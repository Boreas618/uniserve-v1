//! Image understanding and generation through a native worker process.
//!
//! These tests run a scheduler (`SchedulerHandle`) over one real worker process
//! (`uniserve worker` with the weightless stub model on CPU), so every
//! image call the engine plans executes in the worker's image and text
//! runtimes: vision encoding of an inline image and the context prefill that
//! places its features; and an image's generation: the conditioning prefill
//! and cache export, latent preparation, guided denoising with guidance prefix
//! prefills, decoding to a PNG and the generated image's feedback encoding.
//! The stub's denoiser predicts zero velocity, so a generated image is its
//! seeded initial latent decoded: identical for one seed, different across
//! seeds. The executable comes from `UNISERVE_WORKER_EXECUTABLE` or the
//! repository environment's `.venv/bin/uniserve`.

#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context as _;
use base64::Engine as _;
use engine::{EngineConfig, SchedulerHandle, WorkerGroupConfig, WorkerProcessArgs};
use engine::{
    Event, FeedbackNextToken, FeedbackSource, FinishReason, GenerationConstraint,
    GenerationFeatures, GenerationLimits, GenerationRequest, ImageEncoderInput,
    ImageGenerationConfig, ImageInput, ImageTrigger, MultimodalInputs, Request,
};
use ipc::{ComponentConfig, ImageParams, RequestId, SamplingParams};

/// The stub's end-of-image marker, which its head maps to end of sequence.
const END_OF_IMAGE: u32 = 1007;
/// A token that opens an image branch when sampled.
const START_OF_IMAGE: u32 = 1000;

fn worker_executable() -> PathBuf {
    std::env::var_os("UNISERVE_WORKER_EXECUTABLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(".venv/bin/uniserve")
        })
}

/// One CPU rank holding the language model and its patch encoder.
fn engine() -> anyhow::Result<SchedulerHandle> {
    let components: BTreeMap<String, ComponentConfig> = ["model", "vision_encoder"]
        .into_iter()
        .map(|name| {
            (
                name.to_owned(),
                ComponentConfig::parallel(vec![0], Default::default()),
            )
        })
        .collect();
    let mut config = EngineConfig::sim("stub");
    config.generation_limits = GenerationLimits {
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
    config.eos = vec![END_OF_IMAGE];
    config.end_of_image = END_OF_IMAGE;
    config.worker_groups = vec![WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        1,
        2,
        components.clone(),
    )];
    config.worker_process = WorkerProcessArgs {
        executable: worker_executable(),
        daemon: crate::support::daemon(),
        model: String::new(),
        stub: true,
        // The stub rank encodes many small images, so it keeps a larger
        // encoder cache than a served model's default.
        encoder_cache_entries: 1024,
        ranks: config.worker_groups[0].ranks.clone(),
        components,
        queue_depth: 2,
        max_total_tokens: Some(4096),
        block_size: Some(16),
        max_batch_calls: 16,
        max_batch_tokens: 256,
        attention_backend: "torch".into(),
        ..config.worker_process.clone()
    };
    Ok(SchedulerHandle::start_all(config)?.remove(0))
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
            Event::TextToken { id, .. } => outcome.tokens.push(id),
            Event::TextTokens { ids } => outcome.tokens.extend(ids),
            Event::PromptLogprobs { positions } => {
                outcome.prompt_scores += positions.len();
            }
            Event::ImageDone { pixels_png_b64, .. } => {
                outcome.images.push(pixels_png_b64);
            }
            Event::Rejected { message, .. } => {
                anyhow::bail!("request {id:?} was rejected: {message}")
            }
            Event::Finished { reason, .. } => {
                outcome.reason = Some(reason);
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
    let engine = engine()?;
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
        // that follows it; the stub's head ends that text.
        let fed = run(
            &engine,
            request(
                4,
                vec![5, 6, 7],
                MultimodalInputs::default(),
                GenerationConstraint::Default,
                image(7, 2, 1.0),
                true,
            ),
        )?;
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

#[test]
fn an_input_image_is_encoded_into_its_prompt() -> anyhow::Result<()> {
    let engine = engine()?;
    let result = (|| -> anyhow::Result<()> {
        // A 32x32 image is a 2x2 patch grid: four feature tokens at one
        // temporal position.
        let prompt = |color| MultimodalInputs {
            images: vec![ImageInput {
                hash: u64::from(color),
                b64: png(32, [color, 40, 200]),
                position: 2,
                num_positions: 1,
                encoders: vec![ImageEncoderInput {
                    encoder: engine::ImageEncoder::Vision,
                    num_cache_tokens: Some(4),
                    max_cache_tokens: None,
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
        // The cached feature conditions the prompt as the encoded one did.
        assert_eq!(red.tokens, cached.tokens);

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
        assert_eq!(scored.tokens, red.tokens);
        Ok(())
    })();
    engine.shutdown();
    result?;
    assert!(!engine.is_dead(), "image understanding killed the engine");
    Ok(())
}
