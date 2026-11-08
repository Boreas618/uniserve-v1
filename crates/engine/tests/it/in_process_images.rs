//! Image understanding and generation on the simulated worker.
//!
//! These tests run a scheduler (`SchedulerHandle`) over the simulator, so every
//! image call the engine plans executes in the simulator's image and text
//! runtimes: vision encoding of an inline image and the context prefill that
//! places its features; and an image's generation: the conditioning prefill
//! and cache export, latent preparation, guided denoising with guidance prefix
//! prefills, decoding to a PNG and the generated image's feedback encoding.
//! The simulator's denoiser predicts zero velocity, so a generated image is its
//! seeded initial latent decoded: identical for one seed, different across
//! seeds.

#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::Context as _;
use base64::Engine as _;
use engine::{
    Event, FeedbackNextToken, FeedbackSource, FinishReason, GenerationConstraint,
    GenerationFeatures, GenerationLimits, GenerationRequest, ImageEncoderInput,
    ImageGenerationConfig, ImageInput, ImageTrigger, MultimodalInputs, Request,
};
use engine::{SchedulerConfig, SchedulerHandle, WorkerGroupConfig};
use ipc::{ImageParams, RequestId, SamplingParams};
use proto::config::BlockConfig;

/// The name of the component that encodes an input image's features.
const VISION_ENCODER: &str = "vision_encoder";

/// The stub's end-of-image marker, which its head maps to end of sequence.
const END_OF_IMAGE: u32 = 1007;
/// A token that opens an image branch when sampled.
const START_OF_IMAGE: u32 = 1000;

/// One CPU rank holding the language model and its patch encoder.
///
/// `media_overstatement` is declared in every media artifact's report without
/// being written, which states a rank's report of an extent it cannot store.
fn engine(media_overstatement: u64) -> anyhow::Result<SchedulerHandle> {
    logged_engine(media_overstatement).map(|(engine, _)| engine)
}

/// [`engine`], with the log of what each request's calls write into its
/// context.
fn logged_engine(
    media_overstatement: u64,
) -> anyhow::Result<(SchedulerHandle, engine::ContextLog)> {
    let components: BTreeMap<String, BlockConfig> = ["model", "vision_encoder"]
        .into_iter()
        .map(|name| (name.to_owned(), BlockConfig::parallel(vec![0])))
        .collect();
    let mut args = engine::sim_args("stub");
    let facts = args
        .engine_config
        .model
        .facts
        .as_mut()
        .expect("the simulated arguments state the model's facts");
    facts.limits = GenerationLimits {
        features: GenerationFeatures::UNDERSTANDING
            | GenerationFeatures::VISION_ENCODE
            | GenerationFeatures::LATENT_ENCODE
            | GenerationFeatures::IMAGE_GENERATION,
        max_latent_units: 4096,
        latent_downsample: 16,
        max_vae_grid_tokens: 1024,
        max_vit_grid_tokens: 1024,
        // The density the latent feature bytes give over the VAE grid.
        latent_bytes_per_unit: (4 << 20) / 1024,
        max_latent_feature_bytes: 4 << 20,
        max_vision_feature_bytes: 4 << 20,
        commit_marker_tokens: 2,
        max_cfg_branches: 3,
        encoder_cache_entries: 16,
    };
    facts.eos = vec![END_OF_IMAGE];
    facts.end_of_image = END_OF_IMAGE;
    args.engine_config.parallel.worker_groups = vec![WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        1,
        2,
        components.clone(),
    )];

    let mut sim = engine::SimEngine::new();
    // The deployment's two components, and the one that encodes an input
    // image's features: an image input is served by the vision encoder.
    {
        let info = sim.mut_info_for_test();
        info.components = components
            .keys()
            .map(|name| ipc::ComponentInfo {
                name: name.clone(),
                outputs: Vec::new(),
            })
            .collect();
        info.media_components =
            BTreeMap::from([(ipc::MediaCall::VisionEncoding, VISION_ENCODER.to_owned())]);
    }
    sim.overstate_published_media(media_overstatement);
    sim.place_components(components);
    let context = sim.context_log();
    let worker = engine::SimWorker::new(sim);
    let rank = engine::Worker::info(&worker)
        .runtime_info()
        .expect("the sim states its rank");
    let handle = SchedulerHandle::with_worker(&args, SchedulerConfig::default(), Box::new(worker))
        .expect("the simulated worker serves");
    crate::support::install_policy_on(&handle, engine::BatchCapacity::of(&rank));
    Ok((handle, context))
}

/// A base64 PNG of `side` x `side` pixels of one color.
fn png(side: u32, color: [u8; 3]) -> String {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, side, side);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let pixels: Vec<u8> = (0..side * side).flat_map(|_| color).collect();
        writer.write_image_data(&pixels).unwrap();
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Decoded RGB pixels and dimensions of a base64 PNG.
fn pixels(encoded: &str) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)?;
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info()?;
    let mut buffer = vec![0; reader.output_buffer_size().context("unbounded PNG")?];
    let info = reader.next_frame(&mut buffer)?;
    buffer.truncate(info.buffer_size());
    anyhow::ensure!(info.color_type == png::ColorType::Rgb, "PNG is not RGB");
    Ok((info.width, info.height, buffer))
}

fn policy(feedback: bool) -> ImageGenerationConfig {
    ImageGenerationConfig {
        trigger: ImageTrigger::Token {
            token_id: START_OF_IMAGE,
        },
        requires_text_for_image: false,
        feedback_source: feedback.then_some(FeedbackSource::DeviceProduct),
        feedback_next_token: FeedbackNextToken::EndOfImage,
        num_feedback_positions: 2,
        feedback_encoders: if feedback {
            vec![ImageEncoderInput {
                encoder: engine::ImageEncoder::Latent,
                num_cache_tokens: None,
                max_cache_tokens: None,
                height: None,
                width: None,
            }]
        } else {
            Vec::new()
        },
        sample_feedback_continuation: false,
    }
}

fn request(
    request_id: u64,
    prompt_token_ids: Vec<u32>,
    multimodal_inputs: MultimodalInputs,
    constraint: GenerationConstraint,
    image: ImageParams,
    feedback: bool,
) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(request_id),
        prompt_token_ids,
        multimodal_inputs,
        negative_prompt_token_ids: Vec::new(),
        constraint,
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        image,
        max_images: 1,
        max_und_tokens: 4,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        image_generation: policy(feedback),
        readout: Vec::new(),
        generated_token_ids: Vec::new(),
        adapter: 0,
        generated_images: Vec::new(),
        trajectory: Default::default(),
    }
}

/// What a request produced.
#[derive(Default)]
struct Outcome {
    tokens: Vec<u32>,
    /// Scored prompt positions.
    prompt_scores: usize,
    images: Vec<String>,
    reason: Option<FinishReason>,
    /// Every token the engine counted, in order: text and control tokens.
    generated: Vec<u32>,
    /// Each image's identity, its feedback product's values and the number
    /// of counted tokens before it.
    fed_back: Vec<(u32, std::sync::Arc<engine::ImagePixels>, usize)>,
    completion_tokens: usize,
    image_count: usize,
    /// Messages of the artifacts the engine refused to deliver, in arrival
    /// order.
    unavailable: Vec<String>,
}

/// Submit `request` and collect its events until it finishes.
fn run(engine: &SchedulerHandle, request: GenerationRequest) -> anyhow::Result<Outcome> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?;
    let id = request.request_id;
    let mut events = engine.submit(Request::from(request))?;
    let mut outcome = Outcome::default();
    loop {
        let event = runtime
            .block_on(async { tokio::time::timeout(Duration::from_secs(120), events.recv()).await })
            .with_context(|| format!("request {id:?} timed out"))?
            .with_context(|| format!("request {id:?} event stream closed"))?;
        match event {
            Event::TextToken { id, .. } => {
                outcome.tokens.push(id);
                outcome.generated.push(id);
            }
            Event::TextTokens { ids } => {
                outcome.tokens.extend(&ids);
                outcome.generated.extend(ids);
            }
            Event::ControlToken { id } => outcome.generated.push(id),
            Event::PromptLogprobs { positions } => {
                outcome.prompt_scores += positions.len();
            }
            Event::ImageDone {
                image_id,
                pixels_png_b64,
                feedback,
                ..
            } => {
                outcome.images.push(pixels_png_b64);
                if let Some(feedback) = feedback {
                    outcome
                        .fed_back
                        .push((image_id, feedback, outcome.generated.len()));
                }
            }
            Event::ArtifactUnavailable { message } => outcome.unavailable.push(message),
            Event::Rejected { message, .. } => {
                anyhow::bail!("request {id:?} was rejected: {message}")
            }
            Event::Finished {
                reason,
                completion_tokens,
                images,
                ..
            } => {
                outcome.reason = Some(reason);
                outcome.completion_tokens = completion_tokens;
                outcome.image_count = images;
                return Ok(outcome);
            }
            _ => {}
        }
    }
}

fn image(seed: u64, steps: u16, text_scale: f32) -> ImageParams {
    ImageParams {
        steps,
        height: 32,
        width: 32,
        seed: Some(seed),
        cfg_text_scale: text_scale,
        ..ImageParams::default()
    }
}

#[test]
fn generated_images_follow_their_seed_and_feed_back() -> anyhow::Result<()> {
    let engine = engine(0)?;
    let result = (|| -> anyhow::Result<()> {
        // Guided generation evaluates the text-unconditional branch, whose
        // guidance prefix is prefilled into its own slot.
        let first = run(
            &engine,
            request(
                1,
                vec![5, 6, 7],
                MultimodalInputs::default(),
                GenerationConstraint::GenOnly,
                image(7, 3, 4.0),
                false,
            ),
        )?;
        let again = run(
            &engine,
            request(
                2,
                vec![5, 6, 7],
                MultimodalInputs::default(),
                GenerationConstraint::GenOnly,
                image(7, 3, 4.0),
                false,
            ),
        )?;
        let other = run(
            &engine,
            request(
                3,
                vec![5, 6, 7],
                MultimodalInputs::default(),
                GenerationConstraint::GenOnly,
                image(8, 3, 4.0),
                false,
            ),
        )?;
        for outcome in [&first, &again, &other] {
            assert_eq!(outcome.images.len(), 1, "one image per request");
            assert!(
                !matches!(outcome.reason, Some(FinishReason::Error)),
                "image request failed"
            );
        }
        let (width, height, values) = pixels(&first.images[0])?;
        assert_eq!((width, height), (32, 32));
        assert_eq!(values, pixels(&again.images[0])?.2, "one seed, one image");
        assert_ne!(
            values,
            pixels(&other.images[0])?.2,
            "seeds draw distinct latents"
        );

        // An image fed back through the latent encoder conditions the text
        // that follows it. The simulator's text opens no image of itself, so
        // the request's bias opens it.
        let mut fed = request(
            4,
            vec![5, 6, 7],
            MultimodalInputs::default(),
            GenerationConstraint::Default,
            image(7, 2, 1.0),
            true,
        );
        fed.sampling.logit_bias = vec![(START_OF_IMAGE, 100.0)];
        let fed = run(&engine, fed)?;
        assert_eq!(fed.images.len(), 1);
        assert!(
            !matches!(fed.reason, Some(FinishReason::Error)),
            "fed-back request failed"
        );
        Ok(())
    })();
    engine.shutdown();
    result?;
    assert!(!engine.is_dead(), "image requests killed the engine");
    Ok(())
}

/// What request `id`'s calls wrote of its prompt: its four tokens and its
/// image's block, before the tokens it generated.
fn prompt_context(context: &engine::ContextLog, id: u64) -> Vec<engine::ContextWrite> {
    context.admissions(RequestId(id)).remove(0)[..5].to_vec()
}

/// The first `length` tokens the simulator generates for request `id`.
fn simulated_text(id: u64, length: usize) -> Vec<u32> {
    (0..length)
        .map(|index| engine::sim_text_token(RequestId(id), index))
        .collect()
}

#[test]
fn an_input_image_is_encoded_into_its_prompt() -> anyhow::Result<()> {
    let (engine, context) = logged_engine(0)?;
    let result = (|| -> anyhow::Result<()> {
        // A 32x32 image is a 2x2 patch grid: four feature tokens at one
        // temporal position.
        let prompt = |color| MultimodalInputs {
            images: vec![ImageInput {
                digest: crate::support::image_digest(u64::from(color)),
                b64: png(32, [color, 40, 200]),
                position: 2,
                num_positions: 1,
                encoders: vec![ImageEncoderInput {
                    encoder: engine::ImageEncoder::Vision,
                    num_cache_tokens: Some(4),
                    max_cache_tokens: None,
                    height: None,
                    width: None,
                }],
            }],
        };
        let red = run(
            &engine,
            request(
                11,
                vec![5, 6, 7, 8],
                prompt(200),
                GenerationConstraint::UndOnly,
                image(1, 1, 1.0),
                false,
            ),
        )?;
        let cached = run(
            &engine,
            request(
                12,
                vec![5, 6, 7, 8],
                prompt(200),
                GenerationConstraint::UndOnly,
                image(1, 1, 1.0),
                false,
            ),
        )?;
        for outcome in [&red, &cached] {
            assert!(!outcome.tokens.is_empty(), "understanding produced no text");
            assert!(
                !matches!(outcome.reason, Some(FinishReason::Error)),
                "understanding failed"
            );
        }
        // The cached feature conditions the prompt as the encoded one did. The
        // simulator's text follows the request's identity, and these are two
        // requests, so what they share is their prompt's context, the same
        // tokens and image block at the same positions, and each generates its
        // text from its first token.
        let encoded = prompt_context(&context, 11);
        assert!(
            encoded.contains(&engine::ContextWrite::InputBlock { position: 2 }),
            "the prompt holds its image: {encoded:?}"
        );
        assert_eq!(encoded, prompt_context(&context, 12));
        assert_eq!(red.tokens, simulated_text(11, red.tokens.len()));
        assert_eq!(cached.tokens, simulated_text(12, red.tokens.len()));

        // A scored image prompt scores its text: every token but the first,
        // the one after the image predicted from the image's last logits.
        let mut scored = request(
            13,
            vec![5, 6, 7, 8],
            prompt(200),
            GenerationConstraint::UndOnly,
            image(1, 1, 1.0),
            false,
        );
        scored.sampling.return_prompt_logprobs = true;
        let scored = run(&engine, scored)?;
        assert!(
            !matches!(scored.reason, Some(FinishReason::Error)),
            "scored understanding failed"
        );
        assert_eq!(scored.prompt_scores, 3);
        assert_eq!(prompt_context(&context, 13), encoded);
        assert_eq!(scored.tokens, simulated_text(13, red.tokens.len()));
        Ok(())
    })();
    engine.shutdown();
    result?;
    assert!(!engine.is_dead(), "image understanding killed the engine");
    Ok(())
}

/// A request that generated an image and left its engine continues on
/// another from its facts, through the worker's own feedback computation:
/// the engine writes the carried image's decoded values into its context
/// by the request's feedback encoder, which the worker prepares as it
/// prepares the decoded product, and the request generates the images and
/// text the uninterrupted request generates after it, numbered after the
/// carried image, with a usage that counts every token and image.
#[test]
fn a_continuation_feeds_its_carried_image_back_through_the_worker() -> anyhow::Result<()> {
    let interleaved = |id: u64| {
        let mut request = request(
            id,
            vec![5, 6, 7],
            MultimodalInputs::default(),
            GenerationConstraint::Default,
            image(7, 2, 1.0),
            true,
        );
        // Every sample opens an image while the request may open one.
        request.max_images = 2;
        request.max_und_tokens = 6;
        request.sampling.seed = Some(3);
        request.sampling.logit_bias = vec![(START_OF_IMAGE, 100.0)];
        request
    };
    let whole = engine(0)?;
    let uninterrupted = run(&whole, interleaved(21));
    whole.shutdown();
    let uninterrupted = uninterrupted?;
    assert_eq!(uninterrupted.fed_back.len(), 2, "two images fed back");
    assert!(
        !matches!(uninterrupted.reason, Some(FinishReason::Error)),
        "the uninterrupted request failed"
    );

    // The facts a migration carries once the first image is out: the tokens
    // counted before it and the image, with its feedback values.
    let (first, values, tokens) = uninterrupted.fed_back[0].clone();
    assert_eq!(first, 1);
    let mut continuation = interleaved(21);
    continuation.generated_token_ids = uninterrupted.generated[..tokens].to_vec();
    continuation.generated_images = vec![engine::GeneratedImage {
        tokens: tokens as u32,
        image: engine::FeedbackImage::Pixels(values),
    }];
    let resumed = engine(0)?;
    let continued = run(&resumed, continuation);
    resumed.shutdown();
    let continued = continued?;

    assert_eq!(continued.reason, uninterrupted.reason);
    let ids: Vec<u32> = continued.fed_back.iter().map(|(id, _, _)| *id).collect();
    assert_eq!(
        ids,
        [2],
        "the continuation numbers its image after the carried one"
    );
    assert_eq!(continued.images, uninterrupted.images[1..]);
    assert_eq!(
        uninterrupted.generated[tokens..],
        continued.generated[..],
        "the continuation generates what the uninterrupted request did after the image"
    );
    assert_eq!(continued.completion_tokens, uninterrupted.completion_tokens);
    assert_eq!(continued.image_count, uninterrupted.image_count);
    Ok(())
}

/// The engine owns the artifact a call publishes, and refuses one whose
/// declared extent outruns its storage.
///
/// A rank states the payload of a completion by name and extent; the engine
/// claims the object by that statement and keeps the mapping while the result
/// waits. Here the simulator publishes the bytes an honest rank publishes and
/// declares them truthfully, so the image is delivered intact with the
/// published length; with one byte declared beyond the storage, the claim
/// fails (`SharedMediaError::Short`) and the client is told the artifact is
/// unavailable instead of receiving the first bytes the object holds.
#[test]
fn media_is_owned_by_the_engine_and_a_short_payload_fails_its_call() -> anyhow::Result<()> {
    let honest = engine(0)?;
    let result = (|| -> anyhow::Result<()> {
        let delivered = run(
            &honest,
            request(
                1,
                vec![5, 6, 7],
                MultimodalInputs::default(),
                GenerationConstraint::GenOnly,
                image(11, 2, 1.0),
                false,
            ),
        )?;
        assert_eq!(delivered.images.len(), 1, "no image was delivered");
        assert!(
            delivered.unavailable.is_empty(),
            "an honest payload was refused: {:?}",
            delivered.unavailable
        );
        // The claim maps the extent the report declared, so the delivered
        // artifact is exactly the published payload: the simulator's PNG.
        let (_, _, rgba) = pixels(&delivered.images[0])?;
        assert_eq!(rgba.len(), 32 * 32 * 3, "the delivered artifact is whole");
        Ok(())
    })();
    honest.shutdown();
    result?;
    assert!(!honest.is_dead(), "a delivered image killed the engine");

    let short = engine(1)?;
    let result = (|| -> anyhow::Result<()> {
        let refused = run(
            &short,
            request(
                2,
                vec![5, 6, 7],
                MultimodalInputs::default(),
                GenerationConstraint::GenOnly,
                image(11, 2, 1.0),
                false,
            ),
        )?;
        assert!(
            refused.images.is_empty(),
            "a payload declaring more than it stores was delivered"
        );
        assert_eq!(
            refused.unavailable.len(),
            1,
            "the failed call was not reported once: {:?}",
            refused.unavailable
        );
        assert!(
            refused.unavailable[0].contains("fewer than its"),
            "the report does not name the short storage: {}",
            refused.unavailable[0]
        );
        assert!(
            matches!(refused.reason, Some(FinishReason::Error)),
            "a refused artifact did not fail the request: {:?}",
            refused.reason
        );
        Ok(())
    })();
    short.shutdown();
    result?;
    assert!(!short.is_dead(), "a refused payload killed the engine");
    Ok(())
}
