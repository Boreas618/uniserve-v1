//! Model-profile prompt preprocessing and generation-policy integration behavior.

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use daemon::input::chat::HfChatRenderer;
use daemon::input::media::ImageInput;
use daemon::input::video::VideoGenerationRequest;
use daemon::input::video::VideoService;
use daemon::input::{InputProcessor, ModelResolutionError, RequestRenderer};
use daemon::model::ModelConfig;
use daemon::model::assets::ResolvedModelFiles;
use daemon::openai::ApiError;
use daemon::options::ChatTemplateContentFormatOption;
use daemon::options::MediaOptions;
use daemon::output::OutputParams;
use daemon::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use models::ModelDescription;
use models::video::DenoiserInfo;
use models::video::ServedCanvas;
use models::video::plan::{ProcessorConfig, canvas};
use proto::request::{GenerationLimits, GenerationRequest};
use tempfile::tempdir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

/// The model name every test request names and the routes check.
const SERVED_MODEL: &str = "served-model";

/// Minimal ChatML template: one `<|im_start|>role\ncontent<|im_end|>` turn per
/// message, then an open assistant turn when a generation prompt is requested.
const CHAT_TEMPLATE: &str = "{%- for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
/// Special tokens added to the synthetic tokenizer: ChatML turn delimiters,
/// image-marker tokens, and reasoning and answer delimiters.
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
        condition_packing: models::video::ConditionPacking::default(),
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

/// `InputProcessor::new` refuses to bind a profile whose worker limits lack a
/// feature the profile's configured branches need, and names the missing
/// feature by its `GenerationFeatures::name` diagnostic.
#[test]
fn model_resolution_refuses_a_missing_runtime_branch() {
    let mut limits = runtime_limits();
    limits
        .features
        .remove(proto::request::GenerationFeatures::UNDERSTANDING);
    let error = match try_resolved_model(ModelDescription::Qwen3, "qwen3", limits) {
        Ok(_) => panic!("incomplete worker limits must fail model resolution"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("runtime_und_execution"),
        "got: {error}"
    );
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
        condition_packing: models::video::ConditionPacking::default(),
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

/// The cache salt selects a stable isolation partition, and prompt logprobs
/// disable prefix-cache reads only, because a prefix-cache hit would skip the
/// prompt positions whose logprobs are requested.
#[test]
fn cache_salt_isolates_and_prompt_logprobs_disable_cache_reads() {
    let (_directory, _tokenizer, processor) = resolved_model(ModelDescription::Qwen3, "qwen3");
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

/// A streamed chat response has no field for prompt logprobs. A streamed
/// request with `prompt_logprobs: 0`, which the route accepts like vLLM does,
/// therefore requests no prompt scoring and keeps prefix-cache reads, while
/// the same buffered request still requests them.
#[test]
fn streamed_chat_does_not_request_undeliverable_prompt_logprobs() {
    let (_directory, _tokenizer, processor) = resolved_model(ModelDescription::Qwen3, "qwen3");
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
            "stream={stream}"
        );
        assert_eq!(
            generation.sampling.prompt_logprobs_requested(),
            !stream,
            "stream={stream}"
        );
        assert_eq!(
            generation.cache.read, stream,
            "stream={stream}"
        );
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

/// A chat request that asks for image output is refused by name: no served
/// profile generates images, so asking for one alone — or beside text — is a
/// refusal, not a text answer.
#[test]
fn a_chat_request_for_image_output_is_refused_by_name() {
    let (_directory, _tokenizer, processor) = resolved_model(ModelDescription::Qwen3, "qwen3");
    for modalities in [vec!["image"], vec!["text", "image"]] {
        let request: daemon::openai::ChatCompletionRequest = serde_json::from_value(
            serde_json::json!({
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": "draw a bird"}],
                "modalities": modalities,
            }),
        )
        .unwrap();
        let error = preprocess_chat(&processor, String::from("image-output"), request, Vec::new())
            .err()
            .unwrap_or_else(|| panic!("{modalities:?} was admitted"));
        let message = error.to_error_response().error.message;
        assert!(message.contains("image_output"), "{modalities:?}: {message}");
    }
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
/// (`u32::MAX`).
#[test]
fn negative_seeds_and_logprob_counts_are_reinterpreted() {
    let (_directory, _, model) = resolved_model(ModelDescription::Qwen3, "qwen3");
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
