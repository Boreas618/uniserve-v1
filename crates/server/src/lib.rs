//! UniServe HTTP frontend, model profiles, request lowering, and output assembly.
//!
//! The crate serves one option tree (`options::ServeConfig`) as shared
//! application state and exposes OpenAI-compatible routes backed by the
//! in-process engine.
//!
//! [`build_state`] is the startup path: it resolves the model's assets into a
//! `profile::ModelConfig`, a tokenizer, and a chat renderer, completes the
//! engine configuration the options state with what the model determines,
//! starts the engine (`engine_client`), and binds the model's input
//! processing to the capabilities the engine reports (`serving`). The HTTP
//! layer (`http`) serves requests from the resulting [`AppState`].

/// Server-facing engine submission and health client.
pub mod engine_client;
/// HTTP listener, middleware, and route construction.
pub mod http;
/// OpenAI-compatible schemas and conversion helpers.
pub mod openai;
/// Serving options: the command line and the configuration file.
pub mod options;
pub mod profile;
pub mod serving;
mod state;
mod video_jobs;

use std::sync::Arc;

use crate::engine_client::EngineClient;
pub use crate::profile::ModelDescription;
use crate::profile::assets::ResolvedModelFiles;
use crate::profile::{ModelConfig, ModelParameters};
pub use crate::serving::chat::ChatTemplateContentFormatOption;
use crate::serving::media::ImageFetcher;
use crate::serving::systemone::ReadoutEncoder;
use crate::serving::{InputProcessor, RequestRenderer, ServingRuntime};
use crate::serving::{LoadedModel, LoadedRenderer};
use anyhow::{Context as _, Result};
use tracing::info;
pub use uniserve_engine::SchedulingPolicy;
use uniserve_engine::SpecialTokenIds;

use crate::options::{ReasoningParser, ServeConfig, Stated};

pub use crate::http::{ApiError, build_router, serve};
pub use crate::state::AppState;

/// Resolves the control tokens the engine uses for the loaded model.
///
/// `bos` and `end_of_image` come from the profile's `GenerationControls` and
/// are `0` for profiles without them (`Qwen3` and `MiniMaxH3`). The EOS list
/// is the model's complete EOS set with the primary EOS placed first (and
/// added when the set lacks it). The order matters to the engine scheduler,
/// which substitutes `eos[0]` for the token of a completion that commits
/// none. The primary EOS is the nonzero `GenerationControls::eos` when
/// present, otherwise `ModelConfig::primary_eos_token_id`; when neither
/// exists the list keeps the set's ascending order.
fn special_token_ids(model: &ModelConfig) -> SpecialTokenIds {
    let controls = model.generation_controls();
    let bos = controls.map_or(0, |value| value.bos);
    let end_of_image = controls.map_or(0, |value| value.end_of_image);
    let primary_eos = controls
        .map(|value| value.eos)
        .filter(|value| *value != 0)
        .or(model.primary_eos_token_id);
    let mut eos = model.eos_token_ids.iter().copied().collect::<Vec<_>>();
    if let Some(primary_eos) = primary_eos {
        eos.retain(|value| *value != primary_eos);
        eos.insert(0, primary_eos);
    }
    SpecialTokenIds {
        bos,
        eos,
        end_of_image,
    }
}

/// Builds the shared application state for one resolved model and one engine client.
///
/// Starts the engine and its worker processes as a side effect, so the
/// returned state owns a running engine. `config` must have passed
/// [`ServeConfig::validate`]; `stated` holds the paths the operator set.
///
/// # Errors
///
/// Fails when the model assets cannot be resolved, when an option of another
/// model family was stated ([`Stated::refuse_foreign_options`]), when the
/// options do not derive an engine configuration
/// ([`ServeConfig::engine_config`]), when the engine fails to start, or when
/// the model description cannot be bound to the capabilities the engine
/// reports.
pub async fn build_state(config: &ServeConfig, stated: &Stated) -> Result<Arc<AppState>> {
    let LoadedModel {
        config: mut model_config,
        tokenizer,
        renderer,
    } = ModelConfig::load(config)
        .await
        .with_context(|| format!("failed to resolve model assets for `{}`", config.model()))?;
    // Options of another model family than the served one are refused
    // rather than ignored.
    stated.refuse_foreign_options(model_config.description())?;
    // The server's block-diffusion sampling replaces the checkpoint's
    // defaults once, before any request is lowered.
    if let ModelParameters::DiffusionGemma(profile) = &mut model_config.parameters {
        profile.denoising = profile
            .denoising
            .with_overrides(&config.diffusion_generation_config)
            .map_err(|message| anyhow::anyhow!("invalid diffusion_generation_config: {message}"))?;
    }
    let max_model_len = model_config.max_model_len;

    // The options state the scheduler and the placement; the loaded model
    // completes what it determines.
    let mut engine_config = config.engine_config()?;
    engine_config.generation_limits =
        model_config.generation_limits(engine_config.worker_process.dtype);
    let control_tokens = special_token_ids(&model_config);
    engine_config.bos = control_tokens.bos;
    engine_config.eos = control_tokens.eos;
    engine_config.end_of_image = control_tokens.end_of_image;
    let worker = &mut engine_config.worker_process;
    worker.max_model_len = max_model_len;
    // Video workers prepare exactly the canvases their denoiser names for
    // the configured resolutions and aspect ratios, refusing a pair it does
    // not generate, and report them back as the only canvases requests may
    // target. Other models prepare none.
    if !matches!(model_config.parameters, ModelParameters::MiniMaxH3) {
        worker.video_resolutions.clear();
        worker.video_aspect_ratios.clear();
    }
    // Every generated canvas uses the served sampling, which the worker
    // sizes its canvas state for and prepares its canvas steps with.
    worker.canvas_sampling = match &model_config.parameters {
        ModelParameters::DiffusionGemma(profile) => Some(crate::serving::canvas_sampling(profile)),
        _ => None,
    };
    // A DiffusionGemma prompt, image block or committed block only
    // conditions canvases: its prefill writes the prefix cache alone.
    worker.prefill_outputs = !matches!(model_config.parameters, ModelParameters::DiffusionGemma(_));

    info!(
        workers = ?engine_config.workers,
        block_size = ?engine_config.worker_process.block_size,
        "starting UniServe Rust engine"
    );
    let client =
        EngineClient::connect(engine_config).context("failed to start the UniServe engine")?;

    let engine = Arc::new(client);
    // A video checkpoint serves what its placed denoiser declares in the
    // worker handshake, planned against the conditioner's vision processor.
    let renderer = match renderer {
        LoadedRenderer::Chat(renderer) => {
            // No other served model's processor takes options.
            anyhow::ensure!(
                config.mm_processor_kwargs.is_empty(),
                "invalid mm_processor_kwargs: the processor of `{}` takes no options",
                config.model()
            );
            RequestRenderer::Chat(*renderer)
        }
        LoadedRenderer::Video(vision) => {
            let denoiser = engine
                .video_denoiser()
                .context("the video worker reported no video denoiser")?;
            // Reference images are banded to the units one latent encoding
            // round covers on the deployment's latent encoder while a
            // request's prompt is encoded.
            let latent_encoding_lane = engine
                .media_lane_units(uniserve_engine::MediaCall::LatentEncoding)
                .unwrap_or(1);
            // The video processor's options are its reference sizing, within
            // what the denoiser declares.
            let references = uniserve_core::video::plan::ReferenceConfig::from_kwargs(
                &config.mm_processor_kwargs,
                &denoiser.references,
            )
            .map_err(|message| anyhow::anyhow!("invalid mm_processor_kwargs: {message}"))?;
            RequestRenderer::Video(
                crate::serving::video::VideoService::new(
                    denoiser,
                    vision,
                    config.video_config.max_seconds,
                    latent_encoding_lane,
                    &config.media_config,
                    references,
                    Arc::clone(&tokenizer),
                )
                .context("failed to bind the video denoiser")?,
            )
        }
    };

    // A DiffusionGemma checkpoint serves the System One readout, whose
    // encoder renders prompts with the checkpoint's own chat template.
    let readout = match &model_config.parameters {
        ModelParameters::DiffusionGemma(profile) => {
            let files = ResolvedModelFiles::new(config.model())
                .await
                .context("failed to resolve the readout chat template")?;
            let encoder = ReadoutEncoder::load(
                &files,
                Arc::clone(&tokenizer),
                profile,
                config.readout_config,
                max_model_len,
            )
            .context("failed to prepare the System One readout")?;
            info!(
                layout = ?config.readout_config.layout,
                canvas = ?config.readout_config.canvas,
                "serving the System One readout"
            );
            Some(encoder)
        }
        _ => None,
    };

    let model = InputProcessor::new(
        model_config,
        tokenizer,
        renderer,
        crate::serving::WorkerCapabilities {
            limits: engine.generation_limits(),
        },
        config.chat_config.reasoning_parser == ReasoningParser::Auto,
    )
    .context("failed to bind the configured model description")?;
    let images = ImageFetcher::new(&config.image_fetch_config)
        .context("failed to initialize the image URL fetcher")?;
    let mut runtime = ServingRuntime::new(
        model,
        Arc::clone(&engine),
        images,
        config.logging_config.log_stats,
    );
    if let Some(encoder) = readout {
        runtime = runtime.with_readout(encoder);
    }

    let server = &config.server_config;
    Ok(Arc::new(
        AppState::new(runtime)
            .with_log_requests(config.logging_config.log_requests)
            .with_request_id_headers(server.request_id_headers)
            .with_api_key(server.api_key.clone())
            .with_request_timeout(server.request_timeout)
            .with_max_concurrent_requests(server.max_concurrent_requests.map(|limit| limit.get()))
            .with_video_body_limit(config.media_config.body_limit())
            .with_video_jobs(config.video_job_config.clone()),
    ))
}
