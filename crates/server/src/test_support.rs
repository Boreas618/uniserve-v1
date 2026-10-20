//! Simulated components shared by the crate's tests, one fixture each: the
//! byte tokenizer, in-memory tokenizer loading, the engine over simulated
//! engine cores, the video service, and the server's router.
//!
//! Requests travel through the server's router (`http::router`) into a real
//! `Engine`, whose engine cores drive `SimEngine` through `SimExecutor`. The
//! default simulator provides no video media components, so the engine
//! refuses every video submission as an invalid request.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt as _;
use uniserve_core::ModelDtype;
use uniserve_core::video::plan::{MAX_VIDEO_SECONDS, ProcessorConfig, canvas};
use uniserve_engine::{EngineConfig, EngineCore, SimEngine, SimExecutor};

use crate::engine::Engine;
use crate::openai::video_jobs::VideoJobs;
use crate::options::{ImageFetchOptions, LoggingOptions, ServerOptions, VideoJobOptions};
use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use crate::profile::{ModelConfig, ModelParameters, SamplingDefaults};
use crate::serving::chat::{ChatTemplateContentFormatOption, HfChatRenderer};
use crate::serving::media::MediaConnector;
use crate::serving::video::VideoService;
use crate::serving::{InputProcessor, RequestRenderer};

/// Served model name the simulated engines answer to.
pub(crate) const SERVED_MODEL: &str = "sim-model";

/// Loads a `tokenizers` definition in memory, as the server loads a
/// checkpoint's `tokenizer.json`.
pub(crate) fn load_tokenizer(definition: &TokenizerBuilder) -> HuggingFaceTokenizer {
    let json = definition.to_string(false).expect("serialize tokenizer");
    HuggingFaceTokenizer::from_json(serde_json::from_str(&json).expect("parse tokenizer"))
        .expect("load tokenizer")
}

/// Builds a deterministic byte-oriented tokenizer for serving tests.
///
/// Every printable ASCII character encodes to the token whose ID equals its
/// code point, so tests write IDs as `b'a' as u32`. ID 0 is the `<unk>`
/// token, and IDs 1 through 6 (control characters) are replaced by the chat,
/// vision, and thinking special tokens, which encode atomically. The BPE model
/// has no merges, so encoding is one token per character.
pub(crate) fn configured_tokenizer() -> DynTokenizer {
    let mut vocab = (0_u32..=127)
        .map(|id| {
            let token = if id == 0 {
                "<unk>".to_string()
            } else {
                char::from_u32(id).expect("ASCII code point").to_string()
            };
            (token, id)
        })
        .collect::<Vocab>();
    for (id, token) in [
        (1, "<|im_start|>"),
        (2, "<|im_end|>"),
        (3, "<|vision_start|>"),
        (4, "<|vision_end|>"),
        (5, "<think>"),
        (6, "</think>"),
        (97, "a"),
        (98, "b"),
    ] {
        vocab.retain(|_, existing_id| *existing_id != id);
        vocab.insert(token.to_string(), id);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .expect("build configured tokenizer model");
    let mut tokenizer = TokenizerBuilder::new(model);
    tokenizer.add_special_tokens(&[
        AddedToken::from("<|im_start|>", true),
        AddedToken::from("<|im_end|>", true),
        AddedToken::from("<|vision_start|>", true),
        AddedToken::from("<|vision_end|>", true),
        AddedToken::from("<think>", true),
        AddedToken::from("</think>", true),
    ]);
    Arc::new(load_tokenizer(&tokenizer))
}

/// The server's router over `engine` under the default server options.
pub(crate) fn router(engine: Engine) -> Router {
    router_with(engine, &ServerOptions::default())
}

/// The server's router over `engine` under `server`, with the HTTP body limit
/// for the video submission routes and default video job options.
pub(crate) fn router_with(engine: Engine, server: &ServerOptions) -> Router {
    let jobs = VideoJobs::new(VideoJobOptions::default(), CancellationToken::new());
    crate::http::router(
        Arc::new(engine),
        Arc::new(jobs),
        server,
        crate::http::BODY_LIMIT,
    )
}

/// Builds an engine for a model with `parameters` over one default
/// `SimEngine`.
///
/// The tokenizer is [`configured_tokenizer`], whose
/// `<|im_end|>` (ID 2) is the end-of-sequence token, and the chat template
/// renders only the first message's content. A MiniMax H3 model serves the
/// API's longest duration capacity, as a default video deployment does, and a
/// DiffusionGemma model runs under its own generation limits, as
/// `Engine::start` starts it.
pub(crate) fn sim_engine(parameters: ModelParameters) -> Engine {
    sim_readout_engine(parameters, None)
}

/// Builds an engine like [`sim_engine`] that serves the System One readout
/// with `readout`.
pub(crate) fn sim_readout_engine(
    parameters: ModelParameters,
    readout: Option<crate::serving::systemone::ReadoutEncoder>,
) -> Engine {
    let model = sim_model(parameters);
    let mut config = EngineConfig::sim(SERVED_MODEL);
    if let ModelParameters::DiffusionGemma(_) = model.parameters {
        config.generation_limits = model.generation_limits(ModelDtype::BFloat16);
    }
    let core = sim_core(config, SimEngine::new());
    engine_over(model, configured_tokenizer(), vec![core], readout)
}

/// Starts an engine core under `config` that drives `worker` in process.
pub(crate) fn sim_core(config: EngineConfig, worker: SimEngine) -> EngineCore {
    EngineCore::with_executor(config, Box::new(SimExecutor::new(worker)))
        .expect("start a sim engine core")
}

/// Builds an engine for a model with `parameters` over started engine cores,
/// one per data-parallel rank; see [`sim_engine`].
pub(crate) fn sim_engine_over(parameters: ModelParameters, cores: Vec<EngineCore>) -> Engine {
    engine_over(sim_model(parameters), configured_tokenizer(), cores, None)
}

/// The simulated model's facts.
fn sim_model(parameters: ModelParameters) -> ModelConfig {
    ModelConfig {
        parameters,
        sampling_defaults: SamplingDefaults::default(),
        max_model_len: 4096,
        primary_eos_token_id: Some(2),
        eos_token_ids: BTreeSet::from([2]),
    }
}

/// Serves `model` with `tokenizer` over `cores` without request or stats
/// logging. A MiniMax H3 model serves [`sim_video_service`]; any other
/// model renders chat requests with a template of the first message's
/// content.
pub(crate) fn engine_over(
    model: ModelConfig,
    tokenizer: DynTokenizer,
    cores: Vec<EngineCore>,
    readout: Option<crate::serving::systemone::ReadoutEncoder>,
) -> Engine {
    let renderer = match &model.parameters {
        ModelParameters::MiniMaxH3 => RequestRenderer::Video(sim_video_service(MAX_VIDEO_SECONDS)),
        _ => RequestRenderer::Chat(
            HfChatRenderer::new(
                Some("{{ messages[0].content }}".to_string()),
                HashMap::new(),
                ChatTemplateContentFormatOption::String,
            )
            .unwrap(),
        ),
    };
    let processor = InputProcessor::new(
        model,
        tokenizer,
        renderer,
        cores[0].generation_limits(),
        false,
    )
    .unwrap();
    let images = MediaConnector::from_image_options(&ImageFetchOptions::default()).unwrap();
    let logging = LoggingOptions {
        log_requests: false,
        log_stats: false,
    };
    Engine::new(
        SERVED_MODEL.to_owned(),
        cores,
        processor,
        images,
        readout,
        &logging,
    )
    .unwrap()
}

/// The video service of a four-step text-to-video denoiser that generates
/// every canvas of the canvas rule, with the released checkpoint's vision
/// processor geometry and the default media policy.
pub(crate) fn sim_video_service(max_video_seconds: f64) -> VideoService {
    VideoService::new(
        uniserve_engine::VideoDenoiserInfo {
            tasks: vec!["t2va".to_owned()],
            schedule_points: 5,
            video_shift: 12.0,
            audio_shift: 3.0,
            frame_rate: 24,
            audio_sample_rate: 32_000,
            frames: uniserve_core::FrameClips {
                min_frames: 22,
                clip_frames: 17,
                clip_latent_frames: 5,
                tail_latent_frames: 2,
            },
            // The base checkpoint's canvas of each named ratio at 768p.
            canvases: [(21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16)]
                .map(|(width, height): (u32, u32)| uniserve_core::ServedCanvas {
                    short_edge: 768,
                    aspect_ratio: format!("{width}:{height}"),
                    canvas: canvas(f64::from(width), f64::from(height)).unwrap(),
                })
                .to_vec(),
            references: uniserve_core::ReferenceLimits::default(),
        },
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
        &crate::options::MediaOptions::default(),
        // A text-to-video denoiser takes no references to size.
        uniserve_core::video::plan::ReferenceConfig::new(&uniserve_core::ReferenceLimits::default()),
        configured_tokenizer(),
    )
    .unwrap()
}

/// Sends `request` through `router` and returns the status, the headers, and
/// the body parsed as JSON (`Null` for an empty body).
pub(crate) async fn send(
    router: &Router,
    request: Request<Body>,
) -> (StatusCode, HeaderMap, serde_json::Value) {
    let response = router.clone().oneshot(request).await.unwrap();
    let (parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let json = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (parts.status, parts.headers, json)
}

/// Builds a JSON `POST` request to `path`.
pub(crate) fn post_json(path: &str, body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}
