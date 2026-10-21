//! Model-profile prompt preprocessing and generation-policy integration behavior.

use std::collections::HashMap;
use std::fs;
use std::io::{Read as _, Write as _};
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;

use base64::Engine as _;
use daemon::input::chat::HfChatRenderer;
use daemon::input::media::{ImageInput, MediaConnector, MediaListError};
use daemon::input::video::VideoGenerationRequest;
use daemon::input::video::VideoService;
use daemon::input::{InputProcessor, ModelResolutionError, Prompt, RequestRenderer};
use daemon::model::assets::ResolvedModelFiles;
use daemon::model::{ModelConfig, ModelDescription};
use daemon::openai::ApiError;
use daemon::options::ChatTemplateContentFormatOption;
use daemon::options::{ImageFetchOptions, MediaOptions};
use daemon::output::OutputParams;
use daemon::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use models::video::DenoiserInfo;
use models::video::ServedCanvas;
use models::video::plan::{ProcessorConfig, canvas};
use proto::request::ImageEncoder;
use proto::request::{GenerationLimits, GenerationRequest};
use tempfile::tempdir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

/// The model name every test request names and the routes check.
const SERVED_MODEL: &str = "served-model";

/// Base64 payload of a 1x1 PNG image.
const PNG_1X1: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
/// Minimal ChatML template: one `<|im_start|>role\ncontent<|im_end|>` turn per
/// message, then an open assistant turn when a generation prompt is requested.
const CHAT_TEMPLATE: &str = "{%- for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
/// Special tokens added to the synthetic tokenizer: ChatML turn delimiters,
/// the Bagel (`<|vision_*|>`) and SenseNova (`<img>`) image markers, and
/// reasoning and answer delimiters.
const SPECIAL_TOKENS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<img>",
    "</img>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
];

/// What a released base denoiser reports in its handshake: text-to-video
/// over every canvas of the canvas rule, on the 50-point schedule.
fn base_denoiser() -> DenoiserInfo {
    DenoiserInfo {
        tasks: vec!["t2va".to_owned()],
        schedule_points: 50,
        video_shift: 12.0,
        audio_shift: 3.0,
        frame_rate: 24,
        audio_sample_rate: 32_000,
        frames: models::video::FrameClips {
            min_frames: 22,
            clip_frames: 17,
            clip_latent_frames: 5,
            tail_latent_frames: 2,
        },
        // The canvas rule's canvas of each named ratio, at 768p.
        canvases: [(21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16)]
            .map(|(width, height): (u32, u32)| ServedCanvas {
                short_edge: 768,
                aspect_ratio: format!("{width}:{height}"),
                canvas: canvas(f64::from(width), f64::from(height)).unwrap(),
            })
            .to_vec(),
        references: models::video::ReferenceLimits::default(),
    }
}

/// The video service of `denoiser` with a `max_video_seconds` capacity, the
/// released checkpoint's Qwen3-VL processor geometry, a one-rank latent
/// encoder and the default media policy.
fn video_service(
    denoiser: DenoiserInfo,
    max_video_seconds: f64,
    tokenizer: &DynTokenizer,
) -> VideoService {
    VideoService::new(
        denoiser,
        ProcessorConfig {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_min_pixels: 65_536,
            image_max_pixels: 16_777_216,
            video_min_pixels: 4_096,
            video_max_pixels: 25_165_824,
        },
        max_video_seconds,
        1,
        &MediaOptions::default(),
        // A text-to-video denoiser takes no references to size.
        models::video::plan::ReferenceConfig::new(&models::video::ReferenceLimits::default()),
        Arc::clone(tokenizer),
    )
    .unwrap()
}

/// A t2va request body with `fields` merged over a 5-second 16:9 target.
fn video_request(fields: serde_json::Value) -> VideoGenerationRequest {
    let mut body = serde_json::json!({
        "model": "minimax_h3",
        "prompt": "exact token sequence",
        "task": "t2va",
        "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
    });
    body.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    serde_json::from_value(body).unwrap()
}

/// Resolves `description` over a synthetic checkpoint with worker limits that
/// cover every generation feature; see `try_resolved_model`.
fn resolved_model(
    description: ModelDescription,
    model_type: &str,
) -> (tempfile::TempDir, DynTokenizer, InputProcessor) {
    try_resolved_model(description, model_type, runtime_limits()).unwrap()
}

/// Writes a synthetic checkpoint to a temporary directory and binds an
/// `InputProcessor` for `description` over it with the given worker limits.
///
/// The tokenizer maps each ASCII code point from 1 to 127 to one token whose
/// ID is the code point (no merges), plus `SPECIAL_TOKENS`, so outside the
/// special tokens one character is one prompt token. `generation_config.json`
/// sets a 128-token default generation length. The returned directory owns
/// the checkpoint files.
///
/// # Errors
///
/// Returns the `InputProcessor::new` error, for example when `limits` do not
/// cover a feature the profile needs. Other setup failures panic.
fn try_resolved_model(
    description: ModelDescription,
    model_type: &str,
    limits: GenerationLimits,
) -> Result<(tempfile::TempDir, DynTokenizer, InputProcessor), ModelResolutionError> {
    let directory = tempdir().unwrap();

    let mut vocab = Vocab::from_iter([("<unk>".to_string(), 0_u32)]);
    for codepoint in 1_u32..=127 {
        vocab.insert(char::from_u32(codepoint).unwrap().to_string(), codepoint);
    }
    let tokenizer_model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();
    let mut tokenizer_builder = TokenizerBuilder::new(tokenizer_model);
    tokenizer_builder.add_special_tokens(
        &SPECIAL_TOKENS
            .iter()
            .map(|token| AddedToken::from(*token, true))
            .collect::<Vec<_>>(),
    );
    let tokenizer_path = directory.path().join("tokenizer.json");
    tokenizer_builder.save(&tokenizer_path, false).unwrap();

    let config_path = directory.path().join("config.json");
    fs::write(
        &config_path,
        format!(
            r#"{{"model_type":"{model_type}","max_position_embeddings":4096,"num_attention_heads":8}}"#
        ),
    )
    .unwrap();
    let tokenizer_config_path = directory.path().join("tokenizer_config.json");
    fs::write(
        &tokenizer_config_path,
        format!(
            "{{\"eos_token\":\"<|im_end|>\",\"chat_template\":{}}}",
            serde_json::to_string(CHAT_TEMPLATE).unwrap()
        ),
    )
    .unwrap();
    let generation_config_path = directory.path().join("generation_config.json");
    fs::write(
        &generation_config_path,
        r#"{"eos_token_id":2,"max_new_tokens":128}"#,
    )
    .unwrap();
    let files = ResolvedModelFiles {
        tokenizer_path,
        tokenizer_config_path: Some(tokenizer_config_path),
        generation_config_path: Some(generation_config_path),
        chat_template_path: None,
        config_path: Some(config_path),
    };
    // Bagel reads its latent position grid side from the primary checkpoint's
    // safetensors header; a header-only file declaring the published 64x64
    // table suffices.
    if description == ModelDescription::Bagel {
        let header = serde_json::to_vec(&serde_json::json!({
            "latent_pos_embed.pos_embed": {
                "dtype": "BF16", "shape": [4096, 1], "data_offsets": [0, 8192]
            }
        }))
        .unwrap();
        let mut checkpoint = (header.len() as u64).to_le_bytes().to_vec();
        checkpoint.extend(header);
        checkpoint.resize(checkpoint.len() + 8192, 0);
        fs::write(directory.path().join("ema.safetensors"), checkpoint).unwrap();
    }

    let tokenizer: DynTokenizer =
        Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap());
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    // A diffusers pipeline checkpoint has no root `config.json`, so MiniMax H3
    // is built by `ModelConfig::from_pipeline` and served by a video service
    // with the server's default 15-second maximum video duration.
    let config = match description {
        ModelDescription::MiniMaxH3 => ModelConfig::from_pipeline(description, Some(4096)).unwrap(),
        _ => tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(ModelConfig::from_files(
                directory.path().to_str().unwrap(),
                &files,
                Some(4096),
                tokenizer.as_ref(),
            ))
            .unwrap(),
    };
    let renderer = match description {
        ModelDescription::MiniMaxH3 => {
            RequestRenderer::Video(video_service(base_denoiser(), 15.0, &tokenizer))
        }
        _ => RequestRenderer::Chat(renderer),
    };
    let model = InputProcessor::new(config, Arc::clone(&tokenizer), renderer, limits, true)?;
    Ok((directory, tokenizer, model))
}

/// Worker limits that cover every generation feature with capacities no test
/// request reaches.
fn runtime_limits() -> GenerationLimits {
    GenerationLimits {
        features: proto::request::GenerationFeatures::all(),
        max_latent_units: 1_000_000,
        latent_downsample: 16,
        max_vae_grid_tokens: 1_000_000,
        max_vit_grid_tokens: 1_000_000,
        latent_bytes_per_unit: u64::MAX.div_ceil(1_000_000),
        max_latent_feature_bytes: u64::MAX,
        max_vision_feature_bytes: u64::MAX,
        commit_marker_tokens: 2,
        max_cfg_branches: 3,
        encoder_cache_entries: 32,
    }
}

/// Data URL of the `PNG_1X1` image.
fn png_data_url() -> String {
    format!("data:image/png;base64,{PNG_1X1}")
}

/// Chat request with one input image, referenced by `image_url`, between two
/// text parts.
///
/// The leading text contains a literal `</img>`, which the synthetic tokenizer
/// encodes as the SenseNova end-of-image token. The top-level `seed` (9) and
/// the `image_config` seed (17) differ so tests can tell which one wins.
fn image_chat_request(model: &str, image_url: &str) -> daemon::openai::ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "literal </img> before "},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": " after"}
        ]}],
        "seed": 9,
        "image_config": {"seed": 17}
    }))
    .unwrap()
}

/// Preprocesses a chat completion as its route lowers it, answering a
/// refusal as the route does.
fn preprocess_chat(
    processor: &InputProcessor,
    request_id: String,
    request: daemon::openai::ChatCompletionRequest,
    images: Vec<ImageInput>,
) -> Result<(GenerationRequest, OutputParams), ApiError> {
    let (request, _) =
        daemon::openai::chat_completions::lower_chat_request(request_id, SERVED_MODEL, request)?;
    processor
        .preprocess(request, images)
        .map_err(ApiError::from)
}

/// Preprocesses an image-generation request as its route lowers it,
/// answering a refusal as the route does.
fn preprocess_image(
    processor: &InputProcessor,
    request_id: String,
    request: daemon::openai::ImageGenerationRequest,
) -> Result<(GenerationRequest, OutputParams), ApiError> {
    let request = daemon::openai::images::lower_image_request(request_id, SERVED_MODEL, request)?;
    processor
        .preprocess(request, Vec::new())
        .map_err(ApiError::from)
}

/// The input-image URLs of a chat completion, in the order preprocessing
/// pairs resolved images with them.
fn chat_image_urls(request: &daemon::openai::ChatCompletionRequest) -> Vec<String> {
    let (lowered, _) = daemon::openai::chat_completions::lower_chat_request(
        String::new(),
        &request.model,
        request.clone(),
    )
    .unwrap();
    match lowered.prompt {
        Prompt::Chat(chat) => chat.image_urls(),
        Prompt::Text { .. } => Vec::new(),
    }
}

/// Processes a video request through the processor's video service, as
/// `Engine::generate_video` does.
async fn preprocess_video(
    processor: &InputProcessor,
    request_id: &str,
    request: &VideoGenerationRequest,
) -> Result<
    (
        proto::request::DiffusionRequest,
        daemon::input::video::ProcessedVideo,
    ),
    ApiError,
> {
    processor
        .video_service()
        .expect("the processor serves video")
        .process(request_id, request, processor.config().max_model_len)
        .await
        // No worker reads the published condition media here.
        .map(|(request, video, _media)| (request, video))
        .map_err(ApiError::from)
}

/// Runs `MediaConnector::fetch_images` to completion on a current-thread runtime.
fn resolve_images(
    connector: &MediaConnector,
    urls: Vec<String>,
) -> Result<Vec<ImageInput>, MediaListError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(connector.fetch_images(urls))
}

/// A single-turn text chat request to the served model, with the wire
/// `fields` set on top.
fn chat_request(fields: serde_json::Value) -> daemon::openai::ChatCompletionRequest {
    let mut request = serde_json::json!({
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "modalities": ["text"],
    });
    request
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    serde_json::from_value(request).unwrap()
}

/// Resolves a chat request's input images under the default image policy
/// and preprocesses the request with them, as `Engine::generate` does.
fn preprocess_image_chat(
    processor: &InputProcessor,
    request_id: &str,
    request: daemon::openai::ChatCompletionRequest,
) -> (GenerationRequest, OutputParams) {
    let connector = MediaConnector::from_image_options(&ImageFetchOptions::default()).unwrap();
    let images = resolve_images(&connector, chat_image_urls(&request)).unwrap();
    preprocess_chat(processor, String::from(request_id), request, images).unwrap()
}

/// HTTP server on an ephemeral 127.0.0.1 port that answers every request
/// with one `image/png` body from a background thread.
struct ImageServer {
    address: SocketAddr,
}

impl ImageServer {
    /// Starts serving `body`.
    fn start(body: Vec<u8>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else {
                    return;
                };

                // Consume the request head, then answer and close.
                let mut head = Vec::new();
                let mut byte = [0_u8; 1];
                while !head.ends_with(b"\r\n\r\n") && matches!(stream.read(&mut byte), Ok(1)) {
                    head.push(byte[0]);
                }
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(&body);
            }
        });
        Self { address }
    }
}

/// An http image URL prepares exactly the image input its equivalent data URL
/// does, for both omni profiles: the same prompt tokens, encoder-cache
/// identity, worker payload, position, and encoders.
#[test]
fn image_urls_prepare_the_same_image_input_as_data_urls() {
    let png = base64::engine::general_purpose::STANDARD
        .decode(PNG_1X1)
        .unwrap();
    let server = ImageServer::start(png);
    // The server is on loopback, which only the private-destination policy reaches.
    let connector = MediaConnector::from_image_options(&ImageFetchOptions {
        allow_private: true,
        ..ImageFetchOptions::default()
    })
    .unwrap();

    for (description, model_type) in [
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        let prepare = |image_url: &str| {
            let request = image_chat_request(SERVED_MODEL, image_url);
            let images = resolve_images(&connector, chat_image_urls(&request)).unwrap();
            preprocess_chat(&processor, String::from("image-source"), request, images).unwrap()
        };

        let (from_url, url_response) = prepare(&format!("http://{}/image.png", server.address));
        let (from_data, data_response) = prepare(&png_data_url());

        assert_eq!(
            url_response.prompt_token_ids,
            data_response.prompt_token_ids
        );
        assert_eq!(from_url.multimodal_inputs.images.len(), 1, "{model_type}");
        assert_eq!(from_url.multimodal_inputs, from_data.multimodal_inputs);
    }
}

/// `InputProcessor::new` refuses to bind a profile whose worker limits lack a
/// feature the profile's configured branches need, and names the missing
/// feature by its `GenerationFeatures::name` diagnostic. SenseNova appears
/// twice because it needs both ViT encoding and image generation.
#[test]
fn model_resolution_requires_every_configured_runtime_branch() {
    type ChangeLimits = fn(&mut GenerationLimits);
    let cases: [(ModelDescription, &'static str, &'static str, ChangeLimits); 4] = [
        (
            ModelDescription::Qwen3,
            "qwen3",
            "runtime_und_execution",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(proto::request::GenerationFeatures::UNDERSTANDING);
            },
        ),
        (
            ModelDescription::SenseNova,
            "neo_chat",
            "runtime_vit_encode",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(proto::request::GenerationFeatures::VISION_ENCODE);
            },
        ),
        (
            ModelDescription::Bagel,
            "bagel",
            "runtime_vae_encode",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(proto::request::GenerationFeatures::LATENT_ENCODE);
            },
        ),
        (
            ModelDescription::SenseNova,
            "neo_chat",
            "runtime_gen_denoise",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(proto::request::GenerationFeatures::IMAGE_GENERATION);
            },
        ),
    ];

    for (description, model_type, required, remove) in cases {
        let mut limits = runtime_limits();
        remove(&mut limits);
        let error = match try_resolved_model(description, model_type, limits) {
            Ok(_) => panic!("incomplete worker limits must fail model resolution"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(required), "got: {error}");
    }
}

/// The literal `</img>` in the user text also encodes to the end-of-image
/// token, so the prompt holds two of them. The image's position is the index
/// of the end-of-image token of the marker rendered for its slot (the second
/// occurrence), not the first one found in the text.
#[test]
fn sensenova_places_the_input_image_at_its_rendered_slot() {
    let (_directory, tokenizer, model) = resolved_model(ModelDescription::SenseNova, "neo_chat");
    let request = image_chat_request(SERVED_MODEL, &png_data_url());
    let (generation, response) = preprocess_image_chat(&model, "image-params", request);

    // The `image_config` seed takes precedence over the text sampling seed.
    assert_eq!(generation.sampling.seed, Some(17));
    assert_eq!(generation.image.seed, Some(17));

    let end_image = tokenizer.token_to_id("</img>").unwrap();
    let marker_positions = response
        .prompt_token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (*token == end_image).then_some(index as u32))
        .collect::<Vec<_>>();
    assert_eq!(marker_positions.len(), 2);
    let image = &generation.multimodal_inputs.images[0];
    let (position, steps) = (
        image.position,
        image
            .encoders
            .iter()
            .map(|input| input.encoder)
            .collect::<Vec<_>>(),
    );
    assert_eq!(position, marker_positions[1]);
    assert_eq!(steps, &[ImageEncoder::Vision]);

    // Feedback for the default 16:9 canvas (2048x1152) is a 64x36 ViT grid at
    // 32 pixels per token plus one marker token.
    assert_eq!(
        generation
            .image_generation
            .feedback_encoders
            .iter()
            .map(|input| input.num_cache_tokens)
            .collect::<Vec<_>>(),
        vec![Some(2305)]
    );
}

#[test]
fn bagel_places_the_input_image_between_surrounding_chat_text() {
    let (_directory, _tokenizer, model) = resolved_model(ModelDescription::Bagel, "bagel");
    let request = image_chat_request(SERVED_MODEL, &png_data_url());
    let (generation, response) = preprocess_image_chat(&model, "image-params", request);
    assert_eq!(generation.sampling.seed, Some(17));
    assert_eq!(generation.image.seed, Some(17));
    let image = &generation.multimodal_inputs.images[0];
    let (position, steps) = (
        image.position,
        image
            .encoders
            .iter()
            .map(|input| input.encoder)
            .collect::<Vec<_>>(),
    );

    // Bagel removes the slot and places the image at the token count of the
    // text before it, so text on both sides keeps it strictly inside the
    // prompt. The Bagel profile configures a VAE and a ViT encoder input for
    // each input image, in that order.
    assert!(position > 0);
    assert!((position as usize) < response.prompt_token_ids.len());
    assert_eq!(steps, &[ImageEncoder::Latent, ImageEncoder::Vision]);
}

#[tokio::test]
async fn minimax_video_preprocessing_preserves_tokens_seed_and_frame_alignment() {
    let (_directory, tokenizer, model) = resolved_model(ModelDescription::MiniMaxH3, "minimax_h3");
    let id = String::from("video");
    let prepare = |fields: serde_json::Value| {
        let request = video_request(fields);
        let model = &model;
        let id = &id;
        async move { preprocess_video(model, id, &request).await }
    };

    // The t2va presentation is the prompt tokenized without chat framing.
    // The seed defaults to the reference's 42. Five seconds are 120 frames at 24 fps,
    // which extend upward to the next count of the form `17n + 5`: 124
    // frames, decoded as seven 17-frame units plus a tail.
    let (request, accepted) = prepare(serde_json::json!({})).await.unwrap();
    assert_eq!(
        request.prompt_token_ids,
        tokenizer.encode("exact token sequence", false).unwrap()
    );
    let sampling = request.sampling;
    assert_eq!(sampling.seed, 42);
    assert_eq!((sampling.num_frames, sampling.video_units), (124, 7));
    assert_eq!((sampling.width, sampling.height), (1344, 768));
    assert_eq!(accepted.duration_seconds, 5.0);
    let (seeded, _) = prepare(serde_json::json!({"seed": 17})).await.unwrap();
    assert_eq!(seeded.sampling.seed, 17);

    // The admitted interval is [4, 15] seconds inclusive. Fractional
    // durations round half to even before alignment: 5.1875 s is 124.5
    // frames, which rounds to 124 and keeps 124 output frames, where rounding
    // away from zero would reach 125 and extend to 141. 7.3 s is 175.2 frames
    // and 175 is already aligned.
    for (seconds, frames) in [
        (4.0, 107),
        (4.25, 107),
        (5.1875, 124),
        (7.3, 175),
        (10.0, 243),
        (14.0, 345),
        (15.0, 362),
    ] {
        let target = serde_json::json!({
            "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": seconds}
        });
        let sampling = prepare(target).await.unwrap().0.sampling;
        assert_eq!(sampling.num_frames, frames, "{seconds} s");
        assert_eq!(sampling.video_units, (frames - 5) / 17, "{seconds} s");
    }

    // The canvas follows the named aspect ratio; `auto` is 16:9.
    for (ratio, canvas) in [
        ("auto", (1344, 768)),
        ("21:9", (1536, 672)),
        ("4:3", (1024, 768)),
        ("1:1", (768, 768)),
        ("3:4", (768, 1024)),
        ("9:16", (768, 1344)),
    ] {
        let target = serde_json::json!({
            "target": {"short_edge": 768, "aspect_ratio": ratio, "duration_seconds": 5.0}
        });
        let (_, prepared) = prepare(target).await.unwrap();
        assert_eq!(
            (prepared.canvas.width, prepared.canvas.height),
            canvas,
            "{ratio}"
        );
    }

    // A blank prompt, and a task the served denoiser does not serve, are
    // refused naming their field.
    for (fields, param) in [
        (serde_json::json!({"prompt": "  "}), "prompt"),
        (serde_json::json!({"task": "fl2va"}), "task"),
    ] {
        let error = prepare(fields.clone()).await.unwrap_err();
        assert!(
            matches!(
                error,
                daemon::openai::ApiError::InvalidRequest { param: Some(name), .. }
                    if name == param
            ),
            "{fields}: {error:?}"
        );
    }

    // The target alone sets the duration and canvas: a t2va request needs a
    // duration within the finite [4, 15] second range and the 768 short edge.
    for target in [
        serde_json::json!({"short_edge": 768, "aspect_ratio": "16:9"}),
        serde_json::json!({"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 3.99}),
        serde_json::json!({"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.01}),
        serde_json::json!({"short_edge": 720, "aspect_ratio": "16:9", "duration_seconds": 5.0}),
        serde_json::json!({"short_edge": 768, "aspect_ratio": "2:1", "duration_seconds": 5.0}),
    ] {
        let error = prepare(serde_json::json!({"target": target}))
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                daemon::openai::ApiError::InvalidRequest { param: Some(name), .. }
                    if name.starts_with("target.")
            ),
            "{target}: {error:?}"
        );
    }
}

/// A FastH3 DMD deployment serves the training buckets its workers prepared,
/// on the export's own schedule: other canvases are refused, and the
/// capabilities report the canvases with the names that select them, the
/// short edges and ratios they serve, the schedule and the deployment's
/// duration capacity.
#[tokio::test]
async fn video_denoiser_handshake_bounds_requests_and_capabilities() {
    let (_directory, tokenizer, loaded) = resolved_model(ModelDescription::MiniMaxH3, "minimax_h3");
    let canvas = |width, height| models::video::Canvas { width, height };
    let served = |short_edge, aspect_ratio: &str, width, height| ServedCanvas {
        short_edge,
        aspect_ratio: aspect_ratio.to_owned(),
        canvas: canvas(width, height),
    };
    // 768p 16:9, then 480p 21:9 and 16:9.
    let denoiser = DenoiserInfo {
        tasks: vec!["t2va".to_owned()],
        schedule_points: 9,
        video_shift: 10.0,
        audio_shift: 3.0,
        frame_rate: 24,
        audio_sample_rate: 32_000,
        frames: models::video::FrameClips {
            min_frames: 22,
            clip_frames: 17,
            clip_latent_frames: 5,
            tail_latent_frames: 2,
        },
        canvases: vec![
            served(768, "16:9", 1344, 768),
            served(480, "21:9", 992, 416),
            served(480, "16:9", 832, 480),
        ],
        references: models::video::ReferenceLimits::default(),
    };
    let model = InputProcessor::new(
        loaded.config().clone(),
        Arc::clone(&tokenizer),
        RequestRenderer::Video(video_service(denoiser, 10.0, &tokenizer)),
        runtime_limits(),
        true,
    )
    .unwrap();
    let capabilities = model.video_capabilities();
    assert_eq!(capabilities["tasks"], serde_json::json!(["t2va"]));
    assert_eq!(capabilities["min_seconds"], 4.0);
    assert_eq!(capabilities["max_seconds"], 10.0);
    assert_eq!(capabilities["model_max_seconds"], 15.0);
    assert_eq!(capabilities["fps"], 24);
    assert_eq!(
        capabilities["canvas"]["canvases"],
        serde_json::json!([
            {"short_edge": 768, "aspect_ratio": "16:9", "width": 1344, "height": 768},
            {"short_edge": 480, "aspect_ratio": "21:9", "width": 992, "height": 416},
            {"short_edge": 480, "aspect_ratio": "16:9", "width": 832, "height": 480},
        ])
    );
    assert_eq!(
        capabilities["canvas"]["short_edges"],
        serde_json::json!([768, 480])
    );
    assert_eq!(
        capabilities["canvas"]["aspect_ratios"],
        serde_json::json!(["16:9", "21:9"])
    );
    assert_eq!(
        capabilities["canvas"]["sizes"],
        serde_json::json!({
            "768": {"16:9": {"width": 1344, "height": 768}},
            "480": {
                "21:9": {"width": 992, "height": 416},
                "16:9": {"width": 832, "height": 480},
            },
        })
    );
    assert_eq!(
        capabilities["schedule"],
        serde_json::json!({"num_inference_steps": 9, "flow_shift": 10.0, "audio_flow_shift": 3.0})
    );
    assert_eq!(capabilities["max_prompt_tokens"], 4096);

    let id = String::from("capacity");
    let (request, prepared) = preprocess_video(
        &model,
        &id,
        &video_request(serde_json::json!({
            "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 10.0}
        })),
    )
    .await
    .unwrap();
    assert_eq!(request.sampling.num_frames, 243);
    assert_eq!(prepared.num_frames, 243);
    // The checkpoint's nine sigma points fix eight evaluations per request.
    assert_eq!(model.video_service().unwrap().num_inference_steps(), 8);
    assert_eq!(prepared.canvas, canvas(1344, 768));

    // A 480 short edge names the 480p training bucket of the ratio.
    let (narrow_request, narrow) = preprocess_video(
        &model,
        &id,
        &video_request(serde_json::json!({
            "target": {"short_edge": 480, "aspect_ratio": "21:9", "duration_seconds": 5.0}
        })),
    )
    .await
    .unwrap();
    assert_eq!(narrow.canvas, canvas(992, 416));
    assert_eq!(
        (
            narrow_request.sampling.width,
            narrow_request.sampling.height
        ),
        (992, 416)
    );

    // A duration beyond the capacity, an unprepared bucket at either short
    // edge, and an unserved short edge are refused.
    for target in [
        serde_json::json!({"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10.5}),
        serde_json::json!({"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 5.0}),
        serde_json::json!({"short_edge": 768, "aspect_ratio": "21:9", "duration_seconds": 5.0}),
        serde_json::json!({"short_edge": 480, "aspect_ratio": "1:1", "duration_seconds": 5.0}),
        serde_json::json!({"short_edge": 720, "aspect_ratio": "16:9", "duration_seconds": 5.0}),
    ] {
        assert!(
            preprocess_video(
                &model,
                &id,
                &video_request(serde_json::json!({"target": target}))
            )
            .await
            .is_err(),
            "{target}"
        );
    }
}

/// Text preprocessing refuses a prompt longer than the model's
/// `max_model_len`.
#[test]
fn context_length_limits_preprocessed_requests() {
    let (_directory, tokenizer, loaded) = resolved_model(ModelDescription::Qwen3, "qwen3");
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    let mut config = loaded.config().clone();
    config.max_model_len = 8;
    let processor = InputProcessor::new(
        config,
        tokenizer,
        RequestRenderer::Chat(renderer),
        runtime_limits(),
        true,
    )
    .unwrap();

    let error = preprocess_chat(
        &processor,
        String::from("context-capacity"),
        chat_request(serde_json::json!({"messages": [{
            "role": "user",
            "content": "This prompt exceeds eight tokens"
        }]})),
        Vec::new(),
    )
    .err()
    .unwrap();
    assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
    let message = error.to_error_response().error.message;
    assert!(message.contains("8"), "{message}");
}

/// Defaults fill only omitted controls: an explicit zero temperature stays
/// zero rather than taking the default 1.0, while an explicit zero completion
/// length is rejected rather than replaced by the 128-token checkpoint default.
#[test]
fn sampling_defaults_preserve_explicit_zero_controls() {
    let (_directory, _tokenizer, model) = resolved_model(ModelDescription::Qwen3, "qwen3");
    let mut request: daemon::openai::ChatCompletionRequest =
        serde_json::from_value(serde_json::json!({
            "model": SERVED_MODEL, "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    let (default, _) = preprocess_chat(
        &model,
        String::from("sampling-defaults"),
        request.clone(),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(default.max_und_tokens, 128);
    assert_eq!(default.sampling.temperature, 1.0);

    request.temperature = Some(0.0);
    let (greedy, _) = preprocess_chat(
        &model,
        String::from("sampling-defaults"),
        request.clone(),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(greedy.sampling.temperature, 0.0);
    assert_eq!(greedy.max_und_tokens, 128);

    request.max_completion_tokens = Some(0);
    assert!(matches!(
        preprocess_chat(
            &model,
            String::from("sampling-defaults"),
            request,
            Vec::new()
        ),
        Err(daemon::openai::ApiError::InvalidRequest { .. })
    ));
}

/// The image API produces an image-only (`GenOnly`) request that keeps the
/// requested canvas, steps, seed, guidance scale, and negative prompt, the
/// last as its rendered tokens.
/// SenseNova accepts only its resolution buckets, and 1536x1536 is one of
/// them.
#[test]
fn image_api_preserves_requested_dimensions_seed_and_guidance() {
    for (description, model_type, width, height) in [
        (ModelDescription::SenseNova, "neo_chat", 1536, 1536),
        (ModelDescription::Bagel, "bagel", 512, 512),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        let request = |negative_prompt: Option<&str>| {
            let mut body = serde_json::json!({
                "model": SERVED_MODEL,
                "prompt": "a blue bird",
                "size": format!("{width}x{height}"),
                "steps": 4,
                "seed": 17,
                "guidance_scale": 7.5
            });
            if let Some(negative_prompt) = negative_prompt {
                body["negative_prompt"] = negative_prompt.into();
            }
            let request: daemon::openai::ImageGenerationRequest =
                serde_json::from_value(body).unwrap();
            preprocess_image(&processor, String::from("image-api"), request)
                .unwrap()
                .0
        };
        let generation = request(Some("blur"));
        assert_eq!(
            generation.constraint,
            proto::request::GenerationConstraint::GenOnly
        );
        assert_eq!(
            (generation.image.width, generation.image.height),
            (width, height)
        );
        assert_eq!(generation.image.steps, 4);
        assert_eq!(generation.max_images, 1);
        assert_eq!(generation.image.seed, Some(17));
        assert_eq!(generation.sampling.seed, Some(17));
        assert_eq!(generation.image.cfg_text_scale, 7.5);
        // The negative prompt reaches the engine as the tokens the model's
        // profile renders it to, which guidance conditions on.
        assert_ne!(
            generation.negative_prompt_token_ids,
            request(None).negative_prompt_token_ids
        );
    }
}

/// SenseNova bounds each input image by its share of a pixel budget, so the
/// same image preprocessed under a different bound is a different encoder
/// product. Its encoder-cache identity changes once five input images lower
/// the bound, while requests whose images keep the same bound share it.
#[test]
fn sensenova_image_cache_identity_follows_its_pixel_bound() {
    let (_directory, _tokenizer, processor) =
        resolved_model(ModelDescription::SenseNova, "neo_chat");
    let first_image_hash = |count: usize| {
        let content = (0..count)
            .map(|_| {
                serde_json::json!({
                    "type": "image_url",
                    "image_url": {"url": png_data_url()}
                })
            })
            .chain([serde_json::json!({"type": "text", "text": "compare"})])
            .collect::<Vec<_>>();
        let request: daemon::openai::ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": content}]
            }))
            .unwrap();
        let (generation, _) = preprocess_image_chat(&processor, "image-bound", request);
        generation.multimodal_inputs.images[0].hash
    };

    // One to four input images keep the 2048x2048 bound; a fifth lowers it.
    assert_eq!(first_image_hash(1), first_image_hash(4));
    assert_ne!(first_image_hash(4), first_image_hash(5));
}

/// Bagel's learned latent position table covers 64 latent patches per side
/// in the synthetic checkpoint, 1024 pixels at the 16-pixel latent stride. A
/// canvas with a longer side, which would index past a table row or past the
/// table, is an invalid request like other unsupported sizes, while a side
/// of exactly 64 patches is accepted.
#[test]
fn bagel_rejects_a_canvas_side_beyond_its_latent_position_table() {
    let (_directory, _tokenizer, processor) = resolved_model(ModelDescription::Bagel, "bagel");
    let request = |size: &str| -> daemon::openai::ImageGenerationRequest {
        serde_json::from_value(serde_json::json!({
            "model": SERVED_MODEL,
            "prompt": "a banner",
            "size": size
        }))
        .unwrap()
    };

    for size in ["1040x16", "16x1040"] {
        let Err(error) = preprocess_image(&processor, String::from("latent-side"), request(size))
        else {
            panic!("{size} was admitted");
        };
        assert!(
            matches!(error, daemon::openai::ApiError::InvalidRequest { .. }),
            "{size}: {error:?}"
        );
    }
    for size in ["1024x16", "16x1024"] {
        let (generation, _) =
            preprocess_image(&processor, String::from("latent-side"), request(size)).unwrap();
        assert_eq!(
            format!("{}x{}", generation.image.width, generation.image.height),
            size
        );
    }
}

/// The cache salt selects a stable isolation partition, and prompt logprobs
/// disable prefix-cache reads only, because a prefix-cache hit would skip the
/// prompt positions whose logprobs are requested.
#[test]
fn cache_salt_isolates_and_prompt_logprobs_disable_cache_reads() {
    for (description, model_type) in [
        (ModelDescription::Qwen3, "qwen3"),
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        let preprocess = |fields: serde_json::Value| {
            preprocess_chat(
                &processor,
                String::from("cache-controls"),
                chat_request(fields),
                Vec::new(),
            )
            .unwrap()
        };

        let (shared, _) = preprocess(serde_json::json!({}));
        assert_eq!(shared.cache.isolation_key, None);
        assert!(shared.cache.read && shared.cache.write);

        // The same salt always produces the same key, and another salt another.
        let (isolated, _) = preprocess(serde_json::json!({"cache_salt": "ab"}));
        assert!(isolated.cache.isolation_key.is_some());
        assert!(isolated.cache.read && isolated.cache.write);
        let (repeat, _) = preprocess(serde_json::json!({"cache_salt": "ab"}));
        assert_eq!(repeat.cache.isolation_key, isolated.cache.isolation_key);
        let (other, _) = preprocess(serde_json::json!({"cache_salt": "ba"}));
        assert_ne!(other.cache.isolation_key, isolated.cache.isolation_key);

        let (logprobs, response) = preprocess(serde_json::json!({"prompt_logprobs": 0}));
        assert!(!logprobs.cache.read && logprobs.cache.write);
        assert!(response.prompt_logprobs);
    }
}

/// A streamed chat response has no field for prompt logprobs. A streamed
/// request with `prompt_logprobs: 0`, which the route accepts like vLLM does,
/// therefore requests no prompt scoring and keeps prefix-cache reads, while
/// the same buffered request still requests them.
#[test]
fn streamed_chat_does_not_request_undeliverable_prompt_logprobs() {
    for (description, model_type) in [
        (ModelDescription::Qwen3, "qwen3"),
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        for stream in [true, false] {
            let request: daemon::openai::ChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": SERVED_MODEL,
                    "messages": [{"role": "user", "content": "hello"}],
                    "modalities": ["text"],
                    "stream": stream,
                    "prompt_logprobs": 0
                }))
                .unwrap();

            let (generation, response) = preprocess_chat(
                &processor,
                String::from("stream-prompt-logprobs"),
                request,
                Vec::new(),
            )
            .unwrap();

            assert_eq!(
                response.prompt_logprobs, !stream,
                "{model_type} stream={stream}"
            );
            assert_eq!(
                generation.sampling.prompt_logprobs_requested(),
                !stream,
                "{model_type} stream={stream}"
            );
            assert_eq!(
                generation.cache.read, stream,
                "{model_type} stream={stream}"
            );
        }
    }
}

/// Qwen3 chat template shipped by Qwen3 and Qwen3-MoE checkpoints; it asks for
/// JSON tool calls inside `<tool_call>` lines.
const QWEN3_TEMPLATE: &str = include_str!("../templates/qwen3.jinja");
/// Tool instruction in the style of Qwen3-Coder templates, which ask for XML
/// `<function=...>` calls instead of JSON objects.
const XML_CALL_TEMPLATE: &str = "{%- if tools -%}<|im_start|>system\n# Tools\n<tools>{%- for tool in tools -%}{{ tool | tojson }}{%- endfor -%}</tools>\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter>\nvalue\n</parameter>\n</function>\n</tool_call><|im_end|>\n{%- endif -%}{%- for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";

/// Rebinds a resolved Qwen3-family model to `template`.
fn with_template(model_type: &str, template: &str) -> InputProcessor {
    let (_directory, tokenizer, loaded) = resolved_model(ModelDescription::Qwen3, model_type);
    let renderer = HfChatRenderer::new(
        Some(template.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    InputProcessor::new(
        loaded.config().clone(),
        tokenizer,
        RequestRenderer::Chat(renderer),
        runtime_limits(),
        true,
    )
    .unwrap()
}

/// Dense and mixture-of-experts Qwen3 checkpoints serve tool calls only when
/// their chat template asks for the JSON calls the Qwen3 parser reads; a
/// template asking for another call format refuses requests with tools
/// instead of returning their calls as unparsed text.
#[test]
fn qwen3_family_declares_tool_calling_only_for_json_calls() {
    let tool_request = serde_json::json!({
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {
            "name": "weather",
            "parameters": {"type": "object", "properties": {}}
        }}]
    });
    for model_type in ["qwen3", "qwen3_moe"] {
        let json_calls = with_template(model_type, QWEN3_TEMPLATE);
        assert!(json_calls.supports_tool_calling(), "{model_type}");
        preprocess_chat(
            &json_calls,
            String::from("json-tools"),
            serde_json::from_value(tool_request.clone()).unwrap(),
            Vec::new(),
        )
        .unwrap();

        for template in [XML_CALL_TEMPLATE, CHAT_TEMPLATE] {
            let other = with_template(model_type, template);
            assert!(!other.supports_tool_calling(), "{model_type}");
            let error = preprocess_chat(
                &other,
                String::from("other-tools"),
                serde_json::from_value(tool_request.clone()).unwrap(),
                Vec::new(),
            )
            .err()
            .unwrap();
            assert_eq!(error.status_code().as_u16(), 400, "{model_type}");
            let message = error.to_error_response().error.message;
            assert!(message.contains("tool_calling"), "{message}");
        }
    }
}

/// Negative seeds are accepted and reinterpreted as their two's-complement
/// `u64` value, and a logprob count of `-1` requests every candidate
/// (`u32::MAX`), for every token model.
#[test]
fn sampling_controls_have_the_same_meaning_across_token_models() {
    for (description, model_type) in [
        (ModelDescription::Qwen3, "qwen3"),
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _, model) = resolved_model(description, model_type);
        for seed in [-1, -2] {
            let (request, _) = preprocess_chat(
                &model,
                String::from("sampling"),
                chat_request(
                    serde_json::json!({"seed": seed, "logprobs": true, "top_logprobs": -1}),
                ),
                Vec::new(),
            )
            .unwrap();
            assert_eq!(request.sampling.seed, Some(seed as u64));
            assert_eq!(request.sampling.n_logprobs, u32::MAX);
        }
    }
}
