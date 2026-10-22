//! The leader's served model: one model's input processing, its engine
//! process, and the output processing of every request.
//!
//! A [`Leader`] owns the model's [`InputProcessor`] and the client of the
//! engine process that schedules its requests ([`EngineClient`]), as vLLM's
//! `AsyncLLM` owns its input and output processors over its engine core
//! client and SGLang's tokenizer manager drives its schedulers. Every
//! request takes one path:
//!
//! 1. its images are fetched on the async runtime and its prompt rendered and
//!    tokenized on the blocking pool ([`InputProcessor`]);
//! 2. [`Leader::submit`] hands the tokenized request through the migrator
//!    to the engine process, which routes it to one of its schedulers and
//!    returns its events ([`RequestEvents`]), across the engines that serve
//!    it when it migrates;
//! 3. the events are assembled into public [`RequestOutput`] events
//!    ([`output::generation_output`], [`output::media_output`]) and wrapped
//!    in [`RequestStats`], which records the request's metrics once.
//!
//! Dropping the returned stream drops its `RequestEvents`, which cancels the
//! request in the engine that serves it; that is the only cancellation path.
//!
//! [`RequestOutput`]: crate::output::RequestOutput
//! [`output::generation_output`]: crate::output::generation_output
//! [`output::media_output`]: crate::output::media_output

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::metrics::{EngineLabels, METRICS};
use anyhow::Context as _;
use futures::StreamExt as _;
use proto::request::wire;
use proto::request::{GenerationLimits, Request};
use tokio::sync::watch;
use tracing::{info, warn};

use crate::engine::client::{EngineClient, EngineInfo, EngineState, StatsSink, SubmitError};
use crate::engine::control::EngineControl;
use crate::engine::hold::{HoldState, RequestHold};
use crate::input::media::MediaConnector;
use crate::input::readout::ReadoutEncoder;
use crate::input::video::{ProcessedVideo, VideoGenerationRequest, VideoService};
use crate::input::{
    GenerateRequest, InputProcessor, LoadedModel, LoadedRenderer, Prompt, RequestRenderer,
    ServeError, TokenizeError,
};
use crate::metrics::log_stats::StatsLogger;
use crate::metrics::stats::{self as metrics, RequestStats};
use crate::migration::{Migrator, RequestEvents};
use crate::model::{ModelConfig, ModelParameters};
use crate::options::{ReasoningParser, ServeConfig, Stated};
use crate::output::RequestOutputStream;

/// One model served over its engine process.
///
/// Construction spawns Tokio tasks, so every constructor must be called
/// within a Tokio runtime.
pub struct Leader {
    /// The single model name the API serves.
    served_model_name: String,
    /// The model the request and scheduler metrics are labelled with.
    model_name: String,
    processor: Arc<InputProcessor>,
    /// What the engine the model was bound to reported it serves.
    info: EngineInfo,
    /// Every request's way to the engine that serves, across engines.
    migrator: Migrator,
    /// What starts and replaces the engine that serves.
    engines: Arc<Engines>,
    images: MediaConnector,
    /// The System One readout encoder of a model that serves the readout.
    readout: Option<Arc<ReadoutEncoder>>,
    /// Whether every request logs a summary line when it ends.
    log_requests: bool,
    /// Periodic throughput logging, stopped with the engine.
    _stats_logger: Option<StatsLogger>,
}

/// A task an engine serves, which selects the API routes the server
/// registers, as vLLM's `supported_tasks` selects its routers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Chat completions (`/v1/chat/completions`): every model that renders
    /// a chat prompt.
    Generate,
    /// Image generation (`/v1/images/generations`).
    ImageGeneration,
    /// Video generation (`/v1/videos`, `/v1/capabilities`).
    VideoGeneration,
    /// The System One decision readout (`/v1/systemone`).
    Readout,
}

/// Records each scheduler's statistics under `model_name`.
pub(crate) fn stats_sink(model_name: String) -> StatsSink {
    Box::new(move |scheduler, snapshot| {
        metrics::record_scheduler_stats(&METRICS.scheduler, &model_name, scheduler, &snapshot);
    })
}

/// What a leader prepares before it has an engine (`Leader::prepare`).
pub(crate) struct Prepared {
    model_config: ModelConfig,
    tokenizer: crate::tokenizer::DynTokenizer,
    renderer: LoadedRenderer,
    /// The spec of the engine that serves the model.
    pub(crate) spec: wire::EngineSpec,
    max_model_len: u32,
}

impl Leader {
    /// Loads the configured model and starts the engine process that serves
    /// it, with its schedulers and worker processes.
    ///
    /// `config` must have passed [`ServeConfig::validate`]; `stated` holds the
    /// option paths the operator set; `daemon` is this node's daemon's
    /// process socket, through which the engine process and every worker
    /// start. The engine's start blocks until every rank has loaded the
    /// model, so it runs on the blocking pool.
    ///
    /// # Errors
    ///
    /// Fails when the model assets cannot be resolved, when an option of
    /// another model family was stated ([`Stated::refuse_foreign_options`]),
    /// when the options do not form an engine spec
    /// ([`ServeConfig::engine_spec`]), when the engine process fails to
    /// start, or when the model cannot be bound to the capabilities its
    /// workers report.
    pub async fn start(
        config: &ServeConfig,
        stated: &Stated,
        daemon: String,
    ) -> anyhow::Result<Self> {
        let prepared = Self::prepare(config, stated).await?;
        info!(placement = ?prepared.spec.placement, "starting the engine process");
        let control = EngineControl::new(
            &config.request_socket(),
            daemon,
            config.host_identity.clone(),
            config.served_model_name().to_owned(),
        )?;
        let sink = stats_sink(config.model().to_owned());
        let spec = prepared.spec.clone();
        let (client, control) = tokio::task::spawn_blocking(move || {
            control.start(spec, sink).map(|client| (client, control))
        })
        .await
        .context("the engine startup task did not complete")?
        .context("failed to start the UniServe engine")?;
        let served = prepared.spec.clone();
        Self::finish(config, prepared, control, client, served)
    }

    /// What a leader prepares before it has an engine: the model's files
    /// and the spec of the engine that serves it, as [`Leader::start`]
    /// describes.
    ///
    /// # Errors
    ///
    /// As [`Leader::start`], before the engine starts.
    pub(crate) async fn prepare(config: &ServeConfig, stated: &Stated) -> anyhow::Result<Prepared> {
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
            profile.sampling = config
                .diffusion_generation_config
                .apply(profile.sampling)
                .map_err(|message| {
                    anyhow::anyhow!("invalid diffusion_generation_config: {message}")
                })?;
        }
        let max_model_len = model_config.max_model_len;

        // The options state the scheduler and the placement; the loaded model
        // completes what it determines.
        let mut spec = config.engine_spec()?;
        let control_tokens = special_token_ids(&model_config);
        spec.model_facts = Some(wire::ModelFacts {
            limits: Some((&model_config.generation_limits(config.dtype)).into()),
            bos: control_tokens.bos,
            eos: control_tokens.eos,
            end_of_image: control_tokens.end_of_image,
            max_model_len,
            // Every generated canvas uses the served sampling, which the
            // worker sizes its canvas state for and prepares its canvas
            // steps with.
            canvas_sampling: match &model_config.parameters {
                ModelParameters::DiffusionGemma(profile) => Some(wire::CanvasSampling {
                    canvas_length: profile.sampling.canvas_length,
                    max_steps: profile.sampling.max_steps,
                    entropy_bound: profile.sampling.entropy_bound,
                    t_min: profile.sampling.t_min,
                    t_max: profile.sampling.t_max,
                    confidence_threshold: profile.sampling.confidence_threshold,
                    stability_threshold: profile.sampling.stability_threshold,
                }),
                _ => None,
            },
            // A DiffusionGemma prompt, image block or committed block only
            // conditions canvases: its prefill writes the prefix cache alone.
            prefill_outputs: !matches!(model_config.parameters, ModelParameters::DiffusionGemma(_)),
        });
        // Video workers prepare exactly the canvases their denoiser names for
        // the configured resolutions and aspect ratios, refusing a pair it
        // does not generate, and report them back as the only canvases
        // requests may target. Other models prepare none.
        if !matches!(model_config.parameters, ModelParameters::MiniMaxH3)
            && let Some(video) = spec.video.as_mut()
        {
            video.resolutions.clear();
            video.aspect_ratios.clear();
        }

        Ok(Prepared {
            model_config,
            tokenizer,
            renderer,
            spec,
            max_model_len,
        })
    }

    /// The leader of `prepared`'s model over the engine `client` serves,
    /// which `control` started (or adopted) with the spec `served`.
    ///
    /// # Errors
    ///
    /// Fails when the model cannot be bound to the capabilities the
    /// engine's workers report.
    pub(crate) fn finish(
        config: &ServeConfig,
        prepared: Prepared,
        control: EngineControl,
        client: EngineClient,
        served: wire::EngineSpec,
    ) -> anyhow::Result<Self> {
        let Prepared {
            model_config,
            tokenizer,
            renderer,
            spec: _,
            max_model_len,
        } = prepared;
        let model_name = config.model().to_owned();
        let info = client.info().clone();

        // A video checkpoint serves what its placed denoiser declares in the
        // worker handshake, planned against the conditioner's vision
        // processor.
        let mut model_files = None;
        let renderer = match renderer {
            LoadedRenderer::Chat { renderer, files } => {
                // No other served model's processor takes options.
                anyhow::ensure!(
                    config.mm_processor_kwargs.is_empty(),
                    "invalid mm_processor_kwargs: the processor of `{}` takes no options",
                    config.model()
                );
                model_files = Some(files);
                RequestRenderer::Chat(*renderer)
            }
            LoadedRenderer::Video(vision) => {
                anyhow::ensure!(
                    !info.video_denoiser.is_empty(),
                    "the video worker reported no video denoiser"
                );
                let denoiser = models::video::DenoiserInfo::decode(&info.video_denoiser)
                    .context("the video worker described its denoiser unreadably")?;
                denoiser
                    .validate()
                    .map_err(|message| anyhow::anyhow!("the video worker's denoiser: {message}"))?;
                // Reference images are banded to the units one latent
                // encoding round covers on the deployment's latent encoder
                // while a request's prompt is encoded.
                let latent_encoding_lane = info.latent_encoding_units.max(1);
                // The video processor's options are its reference sizing,
                // within what the denoiser declares.
                let references = models::video::plan::ReferenceConfig::from_kwargs(
                    &config.mm_processor_kwargs,
                    &denoiser.references,
                )
                .map_err(|message| anyhow::anyhow!("invalid mm_processor_kwargs: {message}"))?;
                RequestRenderer::Video(
                    VideoService::new(
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
        let readout = match (&model_config.parameters, &model_files) {
            (ModelParameters::DiffusionGemma(profile), Some(files)) => {
                let encoder = ReadoutEncoder::load(
                    files,
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

        let processor = InputProcessor::new(
            model_config,
            tokenizer,
            renderer,
            info.limits.clone(),
            config.chat_config.reasoning_parser == ReasoningParser::Auto,
        )
        .context("failed to bind the configured model description")?;
        let images = MediaConnector::from_image_options(&config.image_fetch_config)
            .context("failed to initialize the image URL fetcher")?;
        let engine = Self::new(
            config.served_model_name().to_owned(),
            model_name,
            client,
            processor,
            images,
            readout,
            config,
        );
        let _ = engine.engines.control.set(Arc::new(control));
        *locked(&engine.engines.served) = Some(served);
        Ok(engine)
    }

    /// Serves `processor`'s model under `served_model_name` over the engine
    /// `client` connects to, labelling metrics with `model_name`.
    ///
    /// `images` resolves the image references of every request, and
    /// `readout` serves `POST /v1/systemone` for a model that has one;
    /// `options` states the request and statistics logging
    /// (`logging_config`) and request migration (`migration_config`).
    pub fn new(
        served_model_name: String,
        model_name: String,
        client: EngineClient,
        processor: InputProcessor,
        images: MediaConnector,
        readout: Option<ReadoutEncoder>,
        options: &ServeConfig,
    ) -> Self {
        let logging = &options.logging_config;
        let info = client.info().clone();
        let stats_logger = logging
            .log_stats
            .then(|| StatsLogger::start(model_name.clone(), (info.schedulers as usize).max(1)));
        let hold = Arc::new(RequestHold::new());
        let engines = Arc::new(Engines {
            hold: Arc::clone(&hold),
            control: std::sync::OnceLock::new(),
            served: std::sync::Mutex::new(None),
            config: std::sync::Mutex::new(options.clone()),
            model_name: model_name.clone(),
            replacing: tokio::sync::Mutex::new(()),
            asleep: AtomicBool::new(false),
            sleep_paused: AtomicBool::new(false),
            adapters: std::sync::Mutex::new(BTreeMap::new()),
        });
        let client = Arc::new(client);
        tokio::spawn(watch_engine(Arc::clone(&engines), Arc::downgrade(&client)));
        hold.serve(client);
        Self {
            served_model_name,
            model_name,
            processor: Arc::new(processor),
            info,
            migrator: Migrator::new(hold, options.migration_config.max_migrations),
            engines,
            images,
            readout: readout.map(Arc::new),
            log_requests: logging.log_requests,
            _stats_logger: stats_logger,
        }
    }

    /// The loaded model's input processor.
    pub fn processor(&self) -> &InputProcessor {
        &self.processor
    }

    /// The single model name the API serves.
    pub fn served_model_name(&self) -> &str {
        &self.served_model_name
    }

    /// The generation limits the engine serves.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.info.limits.clone()
    }

    /// The video denoiser this deployment serves; `None` for a model that
    /// generates no video.
    pub fn video_denoiser(&self) -> Option<&models::video::DenoiserInfo> {
        self.processor.video_service().map(VideoService::denoiser)
    }

    /// The tasks this engine serves.
    ///
    /// A video model serves video generation alone; every other model serves
    /// chat completions, an omni model image generation as well, and a model
    /// with a readout encoder the System One readout.
    pub fn supported_tasks(&self) -> Vec<Task> {
        let mut tasks = Vec::new();
        if self.processor.video_service().is_some() {
            tasks.push(Task::VideoGeneration);
        } else {
            tasks.push(Task::Generate);
        }
        if self.processor.supports_image_output() {
            tasks.push(Task::ImageGeneration);
        }
        if self.readout.is_some() {
            tasks.push(Task::Readout);
        }
        tasks
    }

    /// Whether the engine can accept requests: whether a ready engine
    /// serves them.
    pub fn is_healthy(&self) -> bool {
        self.migrator.hold().is_serving()
    }

    /// The connector every request's image references go through.
    pub(crate) fn images(&self) -> &MediaConnector {
        &self.images
    }

    /// The System One readout encoder, when this model serves the readout.
    pub(crate) fn readout_encoder(&self) -> Option<&Arc<ReadoutEncoder>> {
        self.readout.as_ref()
    }

    /// Fetches the input images of a generation request, renders and
    /// tokenizes it, and streams its output.
    ///
    /// A chat prompt's `image_url` parts are fetched only for a model that
    /// accepts image input; for any other model, preprocessing rejects them
    /// without network access.
    ///
    /// # Errors
    ///
    /// Fails when an image cannot be fetched, preprocessing refuses the
    /// request, or no scheduler accepts it.
    pub async fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<RequestOutputStream, ServeError> {
        // Compilation, and with it the request's latency, is measured from
        // here, so image fetching and time queued for the blocking pool
        // count toward it.
        let started = Instant::now();
        let request_id = request.request_id.clone();
        let image_urls = match &request.prompt {
            Prompt::Chat(chat) if self.processor.supports_image_input() => chat.image_urls(),
            _ => Vec::new(),
        };
        let images = self
            .images
            .fetch_images(image_urls)
            .await
            .map_err(|error| ServeError::ImageInput {
                request_id: request_id.clone(),
                index: error.index,
                source: error.source,
            })?;
        let processor = Arc::clone(&self.processor);
        let (request, output) =
            preprocess(&request_id, move || processor.preprocess(request, images)).await?;

        let compile_us = elapsed_us(started);
        let (events, core) = self.submit(request.into()).await?;
        let output = crate::output::generation_output(
            request_id.clone(),
            started,
            compile_us,
            output,
            events,
        );
        Ok(self.stats(request_id, started, core, output))
    }

    /// Fetches, probes and plans a video request's media, presents its
    /// prompt, and streams its output.
    ///
    /// Also returns what the request generates: its duration, frames and
    /// canvas.
    ///
    /// # Errors
    ///
    /// Fails when the model serves no video, the video service refuses the
    /// request, or no scheduler accepts it.
    pub async fn generate_video(
        &self,
        request_id: String,
        request: VideoGenerationRequest,
    ) -> Result<(ProcessedVideo, RequestOutputStream), ServeError> {
        let started = Instant::now();
        let video =
            self.processor
                .video_service()
                .ok_or_else(|| ServeError::UnsupportedFeature {
                    request_id: request_id.clone(),
                    feature: "video_generation",
                })?;
        let (request, processed, media) = video
            .process(&request_id, &request, self.processor.config().max_model_len)
            .await?;
        let prompt_tokens = request.prompt_token_ids.len();
        let (events, core) = self.submit(Request::Diffusion(request)).await?;
        let output = crate::output::media_output(
            request_id.clone(),
            started,
            elapsed_us(started),
            prompt_tokens,
            events,
        );
        // The workers read the condition media while the request runs, which
        // ends with its event stream.
        let output: RequestOutputStream = Box::pin(output.inspect(move |_| {
            let _ = &media;
        }));
        Ok((processed, self.stats(request_id, started, core, output)))
    }

    /// Records `output`'s request metrics under the scheduler that serves
    /// it.
    fn stats(
        &self,
        request_id: String,
        started: Instant,
        core: u32,
        output: RequestOutputStream,
    ) -> RequestOutputStream {
        let labels = EngineLabels {
            model_name: self.model_name.clone(),
            engine: core,
        };
        let log = self.log_requests.then_some(request_id);
        Box::pin(RequestStats::new(output, labels, started, log))
    }

    /// Submits one tokenized request to the engine, which routes it to its
    /// least loaded live scheduler; while no engine is ready, the request
    /// waits in the hold until one is.
    ///
    /// Returns the request's event stream and the index of the scheduler
    /// that serves it. Dropping the stream before its terminal event cancels
    /// the request.
    ///
    /// # Errors
    ///
    /// Fails when the engine serves no requests.
    pub(crate) async fn submit(
        &self,
        request: Request,
    ) -> Result<(RequestEvents, u32), SubmitError> {
        let (events, scheduler) = self.migrator.submit(request).await?;
        // A request the engine ended at submission names no scheduler; its
        // stream holds its terminal event.
        Ok((events, scheduler.unwrap_or(0)))
    }

    /// Shuts the engine down: every scheduler finishes its outstanding
    /// requests as aborted and closes its workers, and the engine process
    /// exits. Runs on the blocking pool. Repeated calls are harmless.
    ///
    /// # Errors
    ///
    /// Fails when the blocking shutdown task does not complete, which happens
    /// only if it panics or its runtime is shutting down.
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        let hold = self.migrator.hold();
        let client = hold.client();
        hold.close("the daemon is shutting down");
        let Some(client) = client else {
            return Ok(());
        };
        tokio::task::spawn_blocking(move || client.shutdown())
            .await
            .context("engine shutdown did not complete")
    }

    /// Lets go of the engine without shutting it down, as the daemon a hot
    /// restart replaced does once its requests ended: the successor serves
    /// the engine from then on.
    ///
    /// # Errors
    ///
    /// Fails when the client's close does not complete.
    pub async fn detach(&self) -> anyhow::Result<()> {
        let hold = self.migrator.hold();
        let client = hold.client();
        hold.close("the daemon handed the deployment to its successor");
        let Some(client) = client else {
            return Ok(());
        };
        tokio::task::spawn_blocking(move || client.detach())
            .await
            .context("the engine connection's close did not complete")
    }

    /// Holds off engine replacements (hot updates and relaunches) while the
    /// guard lives, as a hot restart hands the engine over.
    pub(crate) async fn quiesce(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.engines.replacing.lock().await
    }

    /// What a hot restart hands the successor of this leader's engine: its
    /// client, its identity and the next engine identity, the spec it
    /// serves and the options it was made from.
    ///
    /// # Errors
    ///
    /// Fails when no engine this leader started serves.
    pub(crate) fn engine_handoff(&self) -> anyhow::Result<EngineHandoff> {
        let client = self
            .migrator
            .hold()
            .client()
            .context("no engine serves the deployment")?;
        let engine = client
            .engine()
            .context("the serving engine is not one this daemon started")?;
        let control = self
            .engines
            .control
            .get()
            .context("this leader starts no engine of its own")?;
        let served = locked(&self.engines.served)
            .clone()
            .context("this leader starts no engine of its own")?;
        Ok(EngineHandoff {
            next_engine: control.next_engine(),
            client,
            engine,
            served,
            config: locked(&self.engines.config).clone(),
            paused: self.migrator.hold().is_paused(),
            asleep: self.engines.asleep.load(Ordering::Acquire),
            sleep_paused: self.engines.sleep_paused.load(Ordering::Acquire),
            adapters: locked(&self.engines.adapters).clone(),
        })
    }

    /// Takes over the pause, the sleep and the adapters of the leader a hot
    /// restart replaced (`EngineHandoff`).
    pub(crate) fn adopt_pause(
        &self,
        paused: bool,
        asleep: bool,
        sleep_paused: bool,
        adapters: BTreeMap<String, Adapter>,
    ) {
        if paused {
            self.migrator.hold().pause();
        }
        self.engines.asleep.store(asleep, Ordering::Release);
        self.engines
            .sleep_paused
            .store(sleep_paused, Ordering::Release);
        *locked(&self.engines.adapters) = adapters;
    }
}

/// A leader's engine as a hot restart hands it over (`Leader::engine_handoff`).
pub(crate) struct EngineHandoff {
    pub(crate) client: Arc<EngineClient>,
    pub(crate) engine: u64,
    pub(crate) next_engine: u64,
    pub(crate) served: wire::EngineSpec,
    pub(crate) config: ServeConfig,
    /// New requests wait (`Leader::pause`).
    pub(crate) paused: bool,
    /// The engine sleeps, and whether its sleep paused new requests.
    pub(crate) asleep: bool,
    pub(crate) sleep_paused: bool,
    /// The LoRA adapters the workers hold, by name.
    pub(crate) adapters: BTreeMap<String, Adapter>,
}

/// What the leader serves, for `GET /admin/status`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct LeaderStatus {
    /// The served model name, the deployment's name.
    pub deployment: String,
    /// The model the deployment serves.
    pub model: String,
    /// `ready`, `standby`, `draining`, `failed` or `closed` for the engine
    /// that serves; `holding` while an engine is being replaced; `stopped`
    /// once none will serve.
    pub engine: String,
    /// Why the engine failed or no engine will serve.
    pub reason: Option<String>,
    /// Whether new requests wait (`/admin/pause_generation`).
    pub paused: bool,
    /// Whether the engine gave back its KV cache and graph memory
    /// (`/admin/sleep`).
    pub asleep: bool,
    /// Whether an engine is being replaced: by an update or a relaunch.
    pub replacing: bool,
    /// The requests the serving engine has in flight.
    pub in_flight: usize,
    /// The executable the serving engine and its workers run.
    pub executable: Option<String>,
}

/// What an update did, and how long each stage of its pause took.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineUpdate {
    /// The schedulers took the new limits in place, at their next step;
    /// nothing paused, and the stages below are zero.
    pub in_place: bool,
    /// How long requests waited for an engine: from holding them until the
    /// promoted engine served them; the sum of the three stages below.
    pub pause: std::time::Duration,
    /// From holding requests until the running engine held none: its step
    /// boundary, and the media requests it let finish.
    pub drain: std::time::Duration,
    /// The drained engine's close: its last pages written to the host KV
    /// cache, its workers closed and its process exited.
    pub close: std::time::Duration,
    /// The standby's promotion, until it reported that it serves.
    pub promotion: std::time::Duration,
}

/// What a hot update changes: the build, and options of the engine.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateRequest {
    /// The `uniserve` executable of the build to update to; the running
    /// engine's executable when unset.
    #[serde(default)]
    pub executable: Option<String>,
    /// `PATH=VALUE` assignments over the serving options, as `--set` states
    /// them; each must change only what a standby engine may serve with
    /// differently ([`updated_spec`]).
    #[serde(default)]
    pub set: Vec<String>,
}

/// Why a hot update did not happen or did not complete.
#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    /// Another update is in progress.
    #[error("an engine update is in progress")]
    InProgress,
    /// The update asks for what a hot update cannot change.
    #[error("{0}")]
    Refused(String),
    /// The leader runs no engine process of its own to update.
    #[error("this leader runs no engine process of its own")]
    NoEngine,
    /// The update failed; the message says whether the running engine
    /// still serves.
    #[error(transparent)]
    Failed(#[from] anyhow::Error),
}

impl Leader {
    /// Updates the engine to the build `executable` runs, or to the build
    /// the running engine's executable holds now when `None`, without
    /// failing a request.
    ///
    /// A standby engine of that build starts beside the running one: its
    /// workers map the running engine's weights and exclusive storages
    /// without writing them and park. Once it stands by, the leader holds
    /// new requests, drains the running engine (its token requests leave as
    /// migratable and wait for the new engine; its other requests finish),
    /// shuts it down, which writes its published pages to the host KV cache
    /// and gives up its storages, promotes the standby, and has it serve
    /// every held and migrated request.
    ///
    /// # Errors
    ///
    /// Fails when another update runs or the leader runs no engine process;
    /// when the standby does not start, after which the running engine
    /// serves on; and when the standby's promotion fails, after which no
    /// engine serves and held requests fail.
    pub async fn update(&self, request: UpdateRequest) -> Result<EngineUpdate, UpdateError> {
        let _replacing = self
            .engines
            .replacing
            .try_lock()
            .map_err(|_| UpdateError::InProgress)?;
        let control = self.engines.control.get().ok_or(UpdateError::NoEngine)?;
        let served = locked(&self.engines.served)
            .clone()
            .ok_or(UpdateError::NoEngine)?;
        let hold = Arc::clone(self.migrator.hold());
        let running = hold.client().ok_or(UpdateError::NoEngine)?;
        if self.engines.asleep.load(Ordering::Acquire) {
            return Err(UpdateError::Refused(
                "the engine sleeps: wake it (/admin/wake_up) before updating it".to_owned(),
            ));
        }

        // The options the standby serves with, over those the running engine
        // was started with.
        let (config, mut spec) = if request.set.is_empty() {
            (locked(&self.engines.config).clone(), served.clone())
        } else {
            let (config, paths) =
                crate::options::overridden(&locked(&self.engines.config), &request.set)
                    .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
            let recomputed = config
                .engine_spec()
                .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
            let spec = updated_spec(&served, &recomputed).map_err(UpdateError::Refused)?;
            if spec == served {
                return Err(UpdateError::Refused(format!(
                    "{} change nothing the engine serves with: the daemon reads them, \
                     and changing them takes a restart of the daemon",
                    paths.join(", ")
                )));
            }
            (config, spec)
        };
        // Scheduling alone, within the batch the running engine's workers
        // were started for, changes in place at the schedulers' next step.
        if request.executable.is_none()
            && configurable(&served, &spec, running.info().capacity.as_ref())
        {
            let scheduling = spec.scheduling.clone().unwrap_or_default();
            running
                .configure(scheduling)
                .await
                .map_err(|error| anyhow::anyhow!("the engine did not take the limits: {error}"))?;
            info!("the engine took the new scheduling limits in place");
            *locked(&self.engines.served) = Some(spec);
            *locked(&self.engines.config) = config;
            return Ok(EngineUpdate {
                in_place: true,
                pause: std::time::Duration::ZERO,
                drain: std::time::Duration::ZERO,
                close: std::time::Duration::ZERO,
                promotion: std::time::Duration::ZERO,
            });
        }
        if let (Some(executable), Some(workers)) = (request.executable, spec.workers.as_mut()) {
            workers.executable = executable;
        }
        // Relaunches and later updates start from what the updated engine
        // serves with.
        let updated = spec.clone();
        spec.standby = Some(wire::Standby {
            cache_units: running.info().cache_units.clone(),
        });
        // The standby needs as much device memory beside the weights and the
        // cache as the running engine's workers hold: the tail of the KV
        // cache makes room for it, and the updated engine creates it again.
        running
            .resize_cache(wire::CacheChange::Shrink)
            .await
            .map_err(|error| {
                anyhow::anyhow!(
                    "the KV cache did not make room for the standby engine; the running engine \
                     serves on: {error}"
                )
            })?;
        let control = Arc::clone(control);
        let sink = stats_sink(self.model_name.clone());
        info!("starting a standby engine for a hot update");
        let started = tokio::task::spawn_blocking(move || control.start(spec, sink))
            .await
            .context("the standby engine's start did not complete")
            .and_then(|started| started.context("the standby engine did not start"));
        let standby = match started {
            Ok(standby) => Arc::new(standby),
            Err(error) => {
                let restored = running.resize_cache(wire::CacheChange::Restore).await;
                return Err(match restored {
                    Ok(_) => error.context("the running engine serves on"),
                    Err(restoring) => error.context(format!(
                        "the running engine serves on without the tail of its KV cache, \
                         which it did not create again: {restoring}"
                    )),
                }
                .into());
            }
        };

        // Requests wait from here until the standby serves.
        let paused = Instant::now();
        hold.hold();
        // An engine lost before it drains ends its token requests as
        // migratable on loss, and the standby replaces it all the same.
        if let Err(error) = running.drain().await {
            warn!("the running engine did not drain: {error}");
        }
        while running.in_flight() > 0 && running.state() == EngineState::Draining {
            tokio::time::sleep(DRAIN_POLL).await;
        }
        let drain = paused.elapsed();
        let drained = Arc::clone(&running);
        let exit = tokio::task::spawn_blocking(move || {
            drained.shutdown();
            drained.await_exit(DRAINED_EXIT)
        })
        .await
        .context("the drained engine's shutdown did not complete")?;
        match exit {
            Ok(Some(status)) => info!(%status, "the drained engine's process exited"),
            Ok(None) => {}
            Err(error) => warn!("the drained engine's process: {error:#}"),
        }
        let close = paused.elapsed() - drain;

        if let Err(error) = standby.promote().await {
            hold.close(format!("the updated engine did not serve: {error}"));
            return Err(anyhow::anyhow!("promoting the standby engine: {error}").into());
        }
        *locked(&self.engines.served) = Some(updated);
        *locked(&self.engines.config) = config;
        tokio::spawn(watch_engine(
            Arc::clone(&self.engines),
            Arc::downgrade(&standby),
        ));
        reload_adapters(&self.engines, &standby).await;
        hold.serve(Arc::clone(&standby));
        let pause = paused.elapsed();
        let promotion = pause - drain - close;
        info!(
            pause_ms = pause.as_millis() as u64,
            drain_ms = drain.as_millis() as u64,
            close_ms = close.as_millis() as u64,
            promotion_ms = promotion.as_millis() as u64,
            "the updated engine serves"
        );
        // The drained engine's memory is free: the updated engine creates
        // the tail of the KV cache the standby went without.
        standby
            .resize_cache(wire::CacheChange::Restore)
            .await
            .map_err(|error| {
                anyhow::anyhow!(
                    "the updated engine serves without the tail of its KV cache, which it did \
                     not create again: {error}"
                )
            })?;
        Ok(EngineUpdate {
            in_place: false,
            pause,
            drain,
            close,
            promotion,
        })
    }
}

/// The spec a standby engine serves with: `served`, with what `recomputed`
/// (the spec of the overridden options) states for the parts a standby may
/// serve with differently. A standby maps the running engine's weights and
/// cache planes and serves its placement, so the model, the placement, the
/// KV cache, the experts and how weights load and are stored stay; the
/// scheduling, CUDA graphs, FlashInfer, the lanes and the workers' other
/// options may change.
///
/// # Errors
///
/// Names the parts `recomputed` changes that a standby must keep.
fn updated_spec(
    served: &wire::EngineSpec,
    recomputed: &wire::EngineSpec,
) -> Result<wire::EngineSpec, String> {
    let mut kept = Vec::new();
    let mut differs = |name: &str, same: bool| {
        if !same {
            kept.push(name.to_owned());
        }
    };
    differs("the deployment", served.deployment == recomputed.deployment);
    differs("the model", served.model == recomputed.model);
    differs("the host", served.host == recomputed.host);
    differs("the placement", served.placement == recomputed.placement);
    differs(
        "the product transport",
        served.transfer == recomputed.transfer,
    );
    differs("the KV cache", served.kv_cache == recomputed.kv_cache);
    differs("the experts", served.experts == recomputed.experts);
    // The daemon clears the video canvases of models whose denoiser names
    // them, as it starts the engine.
    let video = |spec: &wire::EngineSpec| {
        spec.video.clone().map(|mut video| {
            if served.video.as_ref().is_some_and(|served| {
                served.resolutions.is_empty() && served.aspect_ratios.is_empty()
            }) {
                video.resolutions.clear();
                video.aspect_ratios.clear();
            }
            video
        })
    };
    differs("the video options", video(served) == video(recomputed));
    if let (Some(old), Some(new)) = (&served.workers, &recomputed.workers) {
        differs("the worker mesh", old.mesh == new.mesh);
        differs("the dtype", old.dtype == new.dtype);
        differs(
            "the quantization",
            old.quantization_config == new.quantization_config,
        );
        differs("the load format", old.load_format == new.load_format);
        differs("the standby groups", old.standby == new.standby);
    }
    if !kept.is_empty() {
        return Err(format!(
            "a hot update keeps {}: the standby engine maps the running engine's \
             weights and caches and serves its placement",
            kept.join(", ")
        ));
    }
    let mut spec = served.clone();
    spec.scheduling = recomputed.scheduling.clone();
    spec.cuda_graphs = recomputed.cuda_graphs.clone();
    spec.flashinfer = recomputed.flashinfer.clone();
    spec.lanes = recomputed.lanes.clone();
    if let (Some(workers), Some(new)) = (spec.workers.as_mut(), &recomputed.workers) {
        workers.executable = new.executable.clone();
        workers.watchdog_timeout_ms = new.watchdog_timeout_ms;
        workers.distributed_backend = new.distributed_backend.clone();
        workers.attention_backend = new.attention_backend.clone();
        workers.download_dir = new.download_dir.clone();
        workers.load_threads = new.load_threads;
        workers.standby_memory = new.standby_memory.clone();
    }
    Ok(spec)
}

/// Whether `updated` differs from `served` in its scheduling alone, within
/// `capacity`, the batch the running engine's workers were started for: at
/// most as many calls and tokens a step and running requests. The
/// schedulers take such limits in place; any other change starts a standby
/// engine.
fn configurable(
    served: &wire::EngineSpec,
    updated: &wire::EngineSpec,
    capacity: Option<&wire::BatchCapacity>,
) -> bool {
    let (Some(new), Some(capacity)) = (updated.scheduling.as_ref(), capacity) else {
        return false;
    };
    let same_otherwise = wire::EngineSpec {
        scheduling: served.scheduling.clone(),
        ..updated.clone()
    } == *served;
    // A step's tokens left to the engine's default stay as they were.
    let tokens = match new.max_num_batched_tokens {
        Some(tokens) => u64::from(tokens) <= capacity.max_num_batched_tokens,
        None => served
            .scheduling
            .as_ref()
            .is_some_and(|served| served.max_num_batched_tokens.is_none()),
    };
    same_otherwise
        && served.scheduling.as_ref() != Some(new)
        && tokens
        && (capacity.max_batch == 0 || new.max_batch <= capacity.max_batch)
        && new.max_running_requests <= capacity.max_running_requests
}

/// Whether the daemon processes requests for the checkpoint at `path` as
/// it does for the served model at `served`: the same tokenizer, tokenizer
/// settings, generation defaults and chat template, byte for byte.
async fn same_processing(served: &str, path: &str) -> Result<(), String> {
    use crate::model::assets::model_files::ResolvedModelFiles;
    let resolve = |model: &str| {
        let model = model.to_owned();
        async move {
            ResolvedModelFiles::new(&model)
                .await
                .map_err(|error| format!("resolving the files of {model}: {error}"))
        }
    };
    let (old, new) = (resolve(served).await?, resolve(path).await?);
    let read = |file: Option<&std::path::Path>| -> Result<Option<Vec<u8>>, String> {
        file.map(|file| std::fs::read(file).map_err(|error| format!("{}: {error}", file.display())))
            .transpose()
    };
    for (name, old, new) in [
        (
            "tokenizer.json",
            Some(old.tokenizer_path.as_path()),
            Some(new.tokenizer_path.as_path()),
        ),
        (
            "tokenizer_config.json",
            old.tokenizer_config_path.as_deref(),
            new.tokenizer_config_path.as_deref(),
        ),
        (
            "generation_config.json",
            old.generation_config_path.as_deref(),
            new.generation_config_path.as_deref(),
        ),
        (
            "the chat template",
            old.chat_template_path.as_deref(),
            new.chat_template_path.as_deref(),
        ),
    ] {
        if read(old)? != read(new)? {
            return Err(format!(
                "the checkpoint at {path} has another {name} than the served model, which \
                 changes how requests are processed: deploy it instead"
            ));
        }
    }
    Ok(())
}

/// The value behind `mutex`; every holder replaces whole values.
fn locked<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Leader {
    /// What the leader serves now.
    pub fn status(&self) -> LeaderStatus {
        let hold = self.migrator.hold();
        let (engine, reason) = match hold.state() {
            HoldState::Engine(state) => match state {
                EngineState::Ready => ("ready", None),
                EngineState::Standby => ("standby", None),
                EngineState::Draining => ("draining", None),
                EngineState::Failed(reason) => ("failed", Some(reason)),
                EngineState::Closed => ("closed", None),
            },
            HoldState::Holding => ("holding", None),
            HoldState::Closed(reason) => ("stopped", Some(reason)),
        };
        LeaderStatus {
            deployment: self.served_model_name.clone(),
            model: self.model_name.clone(),
            engine: engine.to_owned(),
            reason,
            paused: hold.is_paused(),
            asleep: self.engines.asleep.load(Ordering::Acquire),
            replacing: self.engines.replacing.try_lock().is_err(),
            in_flight: hold.client().map_or(0, |client| client.in_flight()),
            executable: locked(&self.engines.served)
                .as_ref()
                .and_then(|spec| spec.workers.as_ref())
                .map(|workers| workers.executable.clone()),
        }
    }

    /// Holds new requests until [`Self::resume`]; the requests in flight go
    /// on. Returns whether requests were held already.
    pub fn pause(&self) -> bool {
        let paused = self.migrator.hold().pause();
        if !paused {
            info!("new requests wait: generation paused");
        }
        paused
    }

    /// Lets new requests reach the engine again.
    pub fn resume(&self) {
        self.migrator.hold().resume();
        info!("generation continues");
    }

    /// Has the serving engine give back its KV cache and CUDA graph memory
    /// and keep its weights, as vLLM's `sleep` does: new requests wait, the
    /// token requests in flight are recomputed once it wakes, and the other
    /// requests finish first. Returns what each worker reported.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves, or with the reason the engine did not
    /// sleep; requests wait until a wake all the same.
    pub async fn sleep(&self) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.engines.replacing.lock().await;
        let hold = self.migrator.hold();
        let client = hold.client().ok_or(UpdateError::NoEngine)?;
        if self.engines.asleep.swap(true, Ordering::AcqRel) {
            return Ok(Vec::new());
        }
        if !hold.pause() {
            self.engines.sleep_paused.store(true, Ordering::Release);
        }
        let results = client
            .resize_cache(wire::CacheChange::Sleep)
            .await
            .map_err(|error| {
                UpdateError::Failed(anyhow::anyhow!(
                    "the engine did not sleep, and requests wait until /admin/wake_up: {error}"
                ))
            })?;
        info!("the engine sleeps: its KV cache and graph memory are given back");
        Ok(results)
    }

    /// Restores what [`Self::sleep`] gave back, and lets the requests it
    /// held through. Returns what each worker reported.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves, or with the reason the engine did not
    /// wake; requests keep waiting then.
    pub async fn wake(&self) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.engines.replacing.lock().await;
        let hold = self.migrator.hold();
        let client = hold.client().ok_or(UpdateError::NoEngine)?;
        let results = client
            .resize_cache(wire::CacheChange::Wake)
            .await
            .map_err(|error| {
                UpdateError::Failed(anyhow::anyhow!("the engine did not wake: {error}"))
            })?;
        self.engines.asleep.store(false, Ordering::Release);
        if self.engines.sleep_paused.swap(false, Ordering::AcqRel) {
            hold.resume();
        }
        info!("the engine woke");
        Ok(results)
    }

    /// Runs the worker hook `name` with the JSON `payload` on every worker
    /// (`uniserve.hooks`), which a plugin implements, at the engine's next
    /// step boundary, and returns each worker's JSON result.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves, or with the reason a worker gave.
    pub async fn hook(&self, name: &str, payload: &str) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.engines.replacing.lock().await;
        let client = self.migrator.hold().client().ok_or(UpdateError::NoEngine)?;
        let results = client
            .hook(name, payload)
            .await
            .map_err(|error| UpdateError::Failed(anyhow::anyhow!("worker hook {name}: {error}")))?;
        info!(hook = name, workers = results.len(), "ran the worker hook");
        Ok(results)
    }

    /// The LoRA adapter named `name` the workers hold, which a request
    /// naming it as its model computes with.
    pub fn adapter(&self, name: &str) -> Option<crate::input::AdapterSelection> {
        locked(&self.engines.adapters)
            .get(name)
            .map(|adapter| crate::input::AdapterSelection {
                slot: adapter.slot,
                identity: adapter.identity.clone(),
            })
    }

    /// The names of the LoRA adapters the workers hold.
    pub fn adapter_names(&self) -> Vec<String> {
        locked(&self.engines.adapters).keys().cloned().collect()
    }

    /// Loads the LoRA adapter `body` names (`lora_name`, and what the
    /// plugin reads, such as `lora_path`) into a free slot of the workers'
    /// adapter pool, as vLLM's `load_lora_adapter` does: every worker runs
    /// the plugin hook `load_lora_adapter` with the body and the slot
    /// (`adapter`), and requests naming the adapter as their model compute
    /// with it from then on (`uniserve.adapters`). A new engine, after a hot
    /// update or a relaunch, loads it again before it serves.
    ///
    /// # Errors
    ///
    /// Refuses a body without a name, the served model's name and an
    /// adapter loaded already; fails when no engine serves, or with the
    /// reason a worker gave.
    pub async fn load_adapter(
        &self,
        mut body: serde_json::Value,
    ) -> Result<Vec<String>, UpdateError> {
        let name = body["lora_name"]
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| UpdateError::Refused("load_lora_adapter names no lora_name".to_owned()))?
            .to_owned();
        if name == self.served_model_name {
            return Err(UpdateError::Refused(format!(
                "{name} is the served model's name"
            )));
        }
        let _replacing = self.engines.replacing.lock().await;
        let client = self.migrator.hold().client().ok_or(UpdateError::NoEngine)?;
        let slot = {
            let adapters = locked(&self.engines.adapters);
            if adapters.contains_key(&name) {
                return Err(UpdateError::Refused(format!(
                    "the adapter {name} is loaded already"
                )));
            }
            (1..)
                .find(|slot| adapters.values().all(|adapter| adapter.slot != *slot))
                .unwrap_or(u32::MAX)
        };
        body["adapter"] = slot.into();
        let payload = body.to_string();
        let results = client
            .hook("load_lora_adapter", &payload)
            .await
            .map_err(|error| UpdateError::Failed(anyhow::anyhow!("load_lora_adapter: {error}")))?;
        let identity = format!("{name}#{}", uuid::Uuid::new_v4().simple());
        locked(&self.engines.adapters).insert(
            name.clone(),
            Adapter {
                slot,
                identity,
                payload,
            },
        );
        info!(adapter = name, slot, "loaded the adapter");
        Ok(results)
    }

    /// Unloads the LoRA adapter `body` names (`lora_name`): requests stop
    /// naming it at once, and every worker runs the plugin hook
    /// `unload_lora_adapter` with the body and the adapter's slot.
    ///
    /// # Errors
    ///
    /// Refuses a name no loaded adapter has; fails when no engine serves, or
    /// with the reason a worker gave, after which the adapter stays loaded.
    pub async fn unload_adapter(
        &self,
        mut body: serde_json::Value,
    ) -> Result<Vec<String>, UpdateError> {
        let name = body["lora_name"].as_str().unwrap_or_default().to_owned();
        let _replacing = self.engines.replacing.lock().await;
        let client = self.migrator.hold().client().ok_or(UpdateError::NoEngine)?;
        let adapter = locked(&self.engines.adapters)
            .remove(&name)
            .ok_or_else(|| UpdateError::Refused(format!("no adapter {name:?} is loaded")))?;
        body["adapter"] = adapter.slot.into();
        match client.hook("unload_lora_adapter", &body.to_string()).await {
            Ok(results) => {
                info!(adapter = name, slot = adapter.slot, "unloaded the adapter");
                Ok(results)
            }
            Err(error) => {
                locked(&self.engines.adapters).insert(name, adapter);
                Err(UpdateError::Failed(anyhow::anyhow!(
                    "unload_lora_adapter: {error}"
                )))
            }
        }
    }

    /// Rewrites the engine's weights in place from the checkpoint at
    /// `model_path`, another checkpoint of the served model, as SGLang's
    /// `update_weights_from_disk` does, or from a plugin's weight source
    /// (`scheme://...`) that streams the served checkpoint's tensors, and
    /// returns what each worker reported. New requests wait; the token requests in flight are
    /// recomputed with the new weights, and the others finish first. A
    /// relaunch or an update after loads the new checkpoint.
    ///
    /// # Errors
    ///
    /// Refuses a checkpoint the daemon would process requests of
    /// differently (another tokenizer, chat template, generation defaults
    /// or model facts); fails when no engine serves, or with the reason a
    /// worker did not update, after which the weights are those of before.
    pub async fn update_weights(&self, model_path: &str) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.engines.replacing.lock().await;
        let hold = self.migrator.hold();
        let client = hold.client().ok_or(UpdateError::NoEngine)?;
        let mut config = locked(&self.engines.config).clone();
        // A plugin's weight source streams the tensors of the served
        // checkpoint, whose configuration and tokenizer stay; a directory
        // is another checkpoint of the model, checked to process requests
        // alike.
        let streamed = model_path.contains("://");
        if streamed {
            config.load_config.weights = Some(model_path.to_owned());
        } else {
            same_processing(config.model(), model_path)
                .await
                .map_err(UpdateError::Refused)?;
            config.model = Some(model_path.to_owned());
            config.load_config.weights = None;
            let prepared = Leader::prepare(&config, &Stated::from_paths(Vec::new()))
                .await
                .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
            let facts = locked(&self.engines.served)
                .as_ref()
                .map(|spec| spec.model_facts.clone());
            if facts.is_some_and(|facts| facts != prepared.spec.model_facts) {
                return Err(UpdateError::Refused(format!(
                    "the checkpoint at {model_path} has other model facts (generation limits, \
                     special tokens or context length) than the served model: deploy it instead"
                )));
            }
        }

        let paused = !hold.pause();
        let started = Instant::now();
        let results = client.update_weights(model_path).await;
        if paused {
            hold.resume();
        }
        let results = results.map_err(|error| {
            UpdateError::Failed(anyhow::anyhow!("the weights were not updated: {error}"))
        })?;
        if let Some(spec) = locked(&self.engines.served).as_mut() {
            if !streamed {
                spec.model = model_path.to_owned();
            }
            if let Some(workers) = spec.workers.as_mut() {
                workers.weights = config.load_config.weights.clone().unwrap_or_default();
            }
        }
        *locked(&self.engines.config) = config;
        info!(
            model = model_path,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "the engine updated its weights in place"
        );
        Ok(results)
    }

    /// Has the serving engine revoke every cached prefix page; the
    /// requests in flight keep theirs.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves.
    pub fn flush_cache(&self) -> Result<(), UpdateError> {
        let client = self.migrator.hold().client().ok_or(UpdateError::NoEngine)?;
        client.flush_cache();
        info!("flushed the prefix cache");
        Ok(())
    }
}

/// How often a hot update checks whether the drained engine's requests
/// ended.
const DRAIN_POLL: std::time::Duration = std::time::Duration::from_millis(10);

/// How long a hot update waits for the drained engine's process to exit
/// once its connection closed; the daemon ends a process that outlives it
/// with its control connection.
const DRAINED_EXIT: std::time::Duration = std::time::Duration::from_secs(30);

/// What starts and replaces the leader's engines.
struct Engines {
    /// The hold every request passes.
    hold: Arc<RequestHold>,
    /// What starts the leader's engine processes; unset for an engine run
    /// elsewhere.
    control: std::sync::OnceLock<Arc<EngineControl>>,
    /// The spec the serving engine serves with: the one it was started with
    /// and the scheduling it took in place since, which a relaunch and the
    /// next update start from; unset with `control`.
    served: std::sync::Mutex<Option<wire::EngineSpec>>,
    /// The options the serving engine's spec was made from, which an update
    /// overrides.
    config: std::sync::Mutex<ServeConfig>,
    /// The model their statistics are labelled with.
    model_name: String,
    /// Held while an engine is replaced, by a hot update or a relaunch, so
    /// that replacements do not overlap.
    replacing: tokio::sync::Mutex<()>,
    /// The engine gave back its KV cache and graph memory (`Leader::sleep`).
    asleep: AtomicBool,
    /// The sleep paused new requests, which the wake lets through.
    sleep_paused: AtomicBool,
    /// The LoRA adapters the workers hold, by name (`Leader::load_adapter`).
    adapters: std::sync::Mutex<BTreeMap<String, Adapter>>,
}

/// A LoRA adapter the workers hold.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Adapter {
    /// The slot of the workers' adapter pool it lies in.
    pub(crate) slot: u32,
    /// What keeps its cache entries apart from every other adapter's, one
    /// loaded into the same slot before included: its name and a load's
    /// random identity.
    pub(crate) identity: String,
    /// The load's JSON payload, with which a new engine's workers load it
    /// again.
    pub(crate) payload: String,
}

/// Loads every adapter of `engines` into `client`'s workers, as an engine
/// that replaces another starts without them. An adapter that does not load
/// leaves the registry, so that requests naming it are refused instead of
/// computing without it.
async fn reload_adapters(engines: &Engines, client: &EngineClient) {
    let adapters: Vec<(String, Adapter)> = locked(&engines.adapters)
        .iter()
        .map(|(name, adapter)| (name.clone(), adapter.clone()))
        .collect();
    for (name, adapter) in adapters {
        match client.hook("load_lora_adapter", &adapter.payload).await {
            Ok(_) => info!(
                adapter = name,
                slot = adapter.slot,
                "loaded the adapter again"
            ),
            Err(error) => {
                locked(&engines.adapters).remove(&name);
                warn!(
                    adapter = name,
                    "the adapter did not load into the new engine, and requests naming it are \
                     refused: {error}"
                );
            }
        }
    }
}

/// Watches the engine `client` names while the hold serves it, and starts
/// another in its place when it is lost: its workers failed for good or its
/// process ended.
///
/// The hold holds requests while the new engine starts; the lost engine's
/// token requests end as migratable and wait there, and the new engine
/// serves them and every request that arrives meanwhile. The engine's
/// workers find their weights retained at the Storage Servers. An engine the
/// hold no longer serves, as after a hot update or a shutdown, is left as it
/// is. Without an engine process of the leader's own, or when the new
/// engine does not start, the hold closes and held and later requests fail.
async fn watch_engine(engines: Arc<Engines>, mut client: std::sync::Weak<EngineClient>) {
    loop {
        let Some(states) = client.upgrade().map(|client| client.states()) else {
            return;
        };
        let reason = lost(states).await;
        let _replacing = engines.replacing.lock().await;
        if !client
            .upgrade()
            .is_some_and(|client| engines.hold.serves(&client))
        {
            return;
        }
        let (Some(control), Some(spec)) = (engines.control.get(), locked(&engines.served).clone())
        else {
            tracing::error!(%reason, "no engine serves requests");
            engines.hold.close(reason);
            return;
        };
        tracing::error!(%reason, "the engine was lost; starting another");
        engines.hold.hold();
        let control = Arc::clone(control);
        let sink = stats_sink(engines.model_name.clone());
        let started = tokio::task::spawn_blocking(move || control.start(spec, sink))
            .await
            .map_err(anyhow::Error::from)
            .and_then(|started| started);
        match started {
            Ok(relaunched) => {
                let relaunched = Arc::new(relaunched);
                client = Arc::downgrade(&relaunched);
                reload_adapters(&engines, &relaunched).await;
                engines.hold.serve(relaunched);
                info!("the relaunched engine serves");
            }
            Err(error) => {
                let reason =
                    format!("the engine was lost ({reason}) and did not start again: {error:#}");
                tracing::error!(%reason, "no engine serves requests");
                engines.hold.close(reason);
                return;
            }
        }
    }
}

/// Waits until the engine whose states `states` observes stops serving,
/// and returns why. A draining or standing-by engine is no loss.
async fn lost(mut states: watch::Receiver<EngineState>) -> String {
    loop {
        match states.borrow_and_update().clone() {
            EngineState::Ready | EngineState::Draining | EngineState::Standby => {}
            EngineState::Failed(error) => return format!("the engine failed: {error}"),
            EngineState::Closed => return "the engine process ended".to_owned(),
        }
        if states.changed().await.is_err() {
            return "the engine process ended".to_owned();
        }
    }
}

/// Runs synchronous preprocessing (chat rendering, tokenization) on the
/// blocking pool.
///
/// A panicked or cancelled task becomes a tokenization error of
/// `request_id`. Dropping the returned future does not stop the work; its
/// result is discarded.
async fn preprocess<T: Send + 'static>(
    request_id: &str,
    work: impl FnOnce() -> Result<T, ServeError> + Send + 'static,
) -> Result<T, ServeError> {
    tokio::task::spawn_blocking(work)
        .await
        .unwrap_or_else(|error| {
            Err(ServeError::Tokenize {
                request_id: request_id.to_owned(),
                source: TokenizeError::Task(error),
            })
        })
}

/// Microseconds elapsed since `started`.
fn elapsed_us(started: Instant) -> u64 {
    started.elapsed().as_micros() as u64
}

/// The control tokens a model's engine schedules with.
struct SpecialTokenIds {
    bos: u32,
    eos: Vec<u32>,
    end_of_image: u32,
}

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use crate::metrics::METRICS;
    use engine::{EngineConfig, Executor, SchedulerHandle, SimEngine, SimExecutor};
    use futures::{StreamExt as _, poll};
    use proto::request::{
        Event, GenerationConstraint, GenerationRequest, ImageGenerationConfig, Request,
    };
    use proto::request::{ImageParams, SamplingParams};

    use super::Leader;
    use crate::input::GenerateRequest;
    use crate::model::ModelParameters;
    use crate::openai::ChatCompletionRequest;
    use crate::openai::chat_completions::{
        ChatResponseContext, collect_chat_completion, lower_chat_request,
    };
    use crate::output::{FinishReason, RequestOutput};
    use crate::test_support::{SERVED_MODEL, sim_core, sim_engine_over};

    /// A Qwen3 engine over one scheduler, named `engine_model`, that drives
    /// `worker`.
    ///
    /// The scheduler's model name labels the process-wide request metrics,
    /// so a test that reads them names its own engine.
    fn qwen3_engine(worker: SimEngine, engine_model: &str) -> Leader {
        let core = sim_core(EngineConfig::sim(engine_model), worker);
        sim_engine_over(ModelParameters::Qwen3, vec![core])
    }

    /// `request` as the chat route lowers it for the served model.
    fn lowered(id: &str, request: ChatCompletionRequest) -> GenerateRequest {
        lower_chat_request(id.to_owned(), SERVED_MODEL, request)
            .unwrap()
            .0
    }

    /// Greedy chat request of at most `max_completion_tokens` tokens that may
    /// sample only `a`, stopping on each of `stop`.
    fn chat_request(max_completion_tokens: u32, stop: &[&str]) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "prompt"}],
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0.0,
            "allowed_token_ids": [u32::from(b'a')],
            "stop": stop,
        }))
        .unwrap()
    }

    /// Reads `stream` to its end and returns its terminal reason and usage.
    async fn finish(
        mut stream: crate::output::RequestOutputStream,
    ) -> (FinishReason, crate::output::RequestUsage) {
        let mut finished = None;
        while let Some(event) = stream.next().await {
            if let RequestOutput::Finished { reason, usage } = event.unwrap() {
                finished = Some((reason, usage));
            }
        }
        finished.expect("the request finishes")
    }

    /// Returns whether the rendered process-wide metrics hold `line`.
    fn rendered_metrics_hold(line: &str) -> bool {
        let rendered = METRICS.render().unwrap();
        rendered.lines().any(|rendered| rendered == line)
    }

    /// A text request that generates until the simulator's EOS.
    fn text_request() -> Request {
        GenerationRequest {
            request_id: Default::default(),
            prompt_token_ids: vec![1, 2, 3, 4],
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint: GenerationConstraint::UndOnly,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_images: 1,
            max_und_tokens: 64,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: ImageGenerationConfig::default(),
            readout: Vec::new(),
            generated_token_ids: Vec::new(),
            adapter: 0,
        }
        .into()
    }

    /// An engine that is lost while it serves ends the leader's serving: it
    /// reports itself unhealthy and refuses every later request, naming the
    /// loss.
    #[tokio::test]
    async fn a_lost_engine_fails_later_requests() {
        let engine = qwen3_engine(SimEngine::new(), "lost-engine-model");
        assert!(engine.is_healthy());

        // The engine process ends without the leader asking it to.
        let client = engine.migrator.hold().client().expect("an engine serves");
        tokio::task::spawn_blocking(move || client.shutdown())
            .await
            .unwrap();
        for _ in 0..100 {
            if !engine.is_healthy() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!engine.is_healthy());

        let error = engine.submit(text_request()).await.err();
        assert_eq!(
            error,
            Some(crate::engine::client::SubmitError::Unavailable(
                "the engine process ended".to_owned()
            ))
        );
    }

    /// Two requests in flight together land on different schedulers, each
    /// served entirely by its own scheduler.
    #[tokio::test]
    async fn concurrent_requests_spread_over_engine_cores() {
        let mut cores = Vec::new();
        let mut observers = Vec::new();
        for _ in 0..2 {
            let mut executor = SimExecutor::new(SimEngine::new());
            observers.push(executor.observe());
            cores.push(
                SchedulerHandle::with_executor(EngineConfig::sim("sim-model"), Box::new(executor))
                    .expect("start a sim scheduler"),
            );
        }
        let engine = sim_engine_over(ModelParameters::Qwen3, cores);

        // Both requests are submitted before either stream is read, so the
        // first is still in flight when the second is routed.
        let mut streams = Vec::new();
        let mut serving = Vec::new();
        for _ in 0..2 {
            let (stream, core) = engine.submit(text_request()).await.expect("submit request");
            streams.push(stream);
            serving.push(core);
        }
        assert_ne!(serving[0], serving[1]);
        for stream in &mut streams {
            while let Some(event) = stream.next().await {
                if let Event::Finished { reason, .. } = event {
                    assert_eq!(reason, proto::request::FinishReason::Eos);
                    break;
                }
            }
        }
        engine.shutdown().await.expect("shutdown engines");

        let served = observers
            .iter()
            .map(|observer| {
                observer
                    .try_iter()
                    .filter_map(|event| match event {
                        engine::BatchEvent::Submitted(batch) => Some(batch),
                        engine::BatchEvent::Resolved { .. } => None,
                    })
                    .flat_map(|batch| batch.requests)
                    .map(|(call, _)| call.request_key.request_id)
                    .collect::<std::collections::BTreeSet<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(served.iter().map(|ids| ids.len()).sum::<usize>(), 2);
        assert!(served.iter().all(|ids| ids.len() == 1));
    }

    /// Simulated executor whose teardown lasts until another task releases
    /// it, as closing worker processes that drain and exit takes time.
    struct GatedCloseExecutor {
        inner: SimExecutor,
        release: std::sync::mpsc::Receiver<()>,
        released: Arc<AtomicBool>,
    }

    impl Executor for GatedCloseExecutor {
        fn info(&self) -> &engine::ExecutorInfo {
            self.inner.info()
        }

        fn is_ready(&self, worker: &engine::WorkerGroupId) -> bool {
            self.inner.is_ready(worker)
        }

        fn has_capacity(&self, worker: &engine::WorkerGroupId) -> bool {
            self.inner.has_capacity(worker)
        }

        fn command_has_capacity(&self, command: &ipc::BatchCommand) -> bool {
            self.inner.command_has_capacity(command)
        }

        fn submit(
            &mut self,
            batch: engine::ExecutionBatch,
        ) -> Result<(), engine::ExecutorSubmitError> {
            self.inner.submit(batch)
        }

        fn poll(
            &mut self,
            timeout: Duration,
        ) -> Result<Option<engine::BatchResult>, engine::ExecutorError> {
            self.inner.poll(timeout)
        }

        fn close(&mut self) -> Result<(), engine::ExecutorError> {
            // The wait is bounded so that a blocked runtime fails the test
            // instead of hanging it.
            let released = self.release.recv_timeout(Duration::from_secs(5)).is_ok();
            self.released.store(released, Ordering::SeqCst);
            self.inner.close()
        }
    }

    /// Shutdown waits for executor teardown without holding the runtime
    /// thread: on a single-threaded runtime, the task that ends the teardown
    /// runs while `shutdown` is pending.
    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_leaves_the_runtime_free_during_executor_teardown() {
        let (release_tx, release) = std::sync::mpsc::channel();
        let released = Arc::new(AtomicBool::new(false));
        let core = SchedulerHandle::with_executor(
            EngineConfig::sim("sim-model"),
            Box::new(GatedCloseExecutor {
                inner: SimExecutor::new(SimEngine::new()),
                release,
                released: Arc::clone(&released),
            }),
        )
        .expect("start a sim scheduler");
        let engine = sim_engine_over(ModelParameters::Qwen3, vec![core]);

        let releaser = tokio::spawn(async move {
            let _ = release_tx.send(());
        });
        engine.shutdown().await.expect("shutdown engine");
        releaser.await.expect("release task");

        assert!(
            released.load(Ordering::SeqCst),
            "executor teardown held the runtime thread"
        );
    }

    /// A chat prompt that exceeds the total capacity of a two-unit prefix
    /// cache is returned as a stream and then rejected by engine admission.
    /// Collecting the completion must report that rejection as HTTP 400.
    #[tokio::test]
    async fn chat_admission_rejection_remains_a_bad_request() {
        let mut worker = SimEngine::new();
        // Unit zero is the sentinel: two allocatable 64-token pages remain.
        worker.set_num_units(3);
        let engine = qwen3_engine(worker, SERVED_MODEL);
        let request = serde_json::from_value(serde_json::json!({
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "a".repeat(1024)}],
            "max_completion_tokens": 8
        }))
        .unwrap();

        let stream = engine
            .generate(lowered("rejected-chat", request))
            .await
            .unwrap();
        let context = ChatResponseContext {
            request_id: "rejected-chat".to_string(),
            response_model: SERVED_MODEL.to_string(),
            include_usage: false,
            requested_logprobs: false,
            include_prompt_logprobs: false,
            include_reasoning: false,
            return_token_ids: false,
            return_tokens_as_token_ids: false,
        };
        let error = collect_chat_completion(stream, context, 0)
            .await
            .unwrap_err();

        assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
        engine.shutdown().await.unwrap();
    }

    /// A request's reported latency covers the time it waited for the
    /// blocking pool before preprocessing ran.
    #[test]
    fn reported_latency_includes_time_waiting_for_preprocessing() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let engine = qwen3_engine(SimEngine::new(), SERVED_MODEL);

            // Hold the only blocking thread so preprocessing waits in the
            // queue for a measurable interval before it runs.
            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let occupied = tokio::task::spawn_blocking(move || {
                ready_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
            ready_rx.await.unwrap();
            let wait = Duration::from_millis(25);
            let mut pending = Box::pin(engine.generate(lowered("timed", chat_request(8, &["aa"]))));
            assert!(poll!(pending.as_mut()).is_pending());
            tokio::time::sleep(wait).await;
            release_tx.send(()).unwrap();
            occupied.await.unwrap();

            let (_, usage) = finish(pending.await.unwrap()).await;
            assert!(
                usage.timings.total_us >= wait.as_micros() as u64,
                "{:?}",
                usage.timings
            );
            engine.shutdown().await.unwrap();
        });
    }

    /// A completed request is recorded once in the request metrics under its
    /// engine's model name: the token counters and token histograms carry the
    /// counts of its usage report, its finish reason is counted, and each
    /// latency histogram whose input it carries holds one observation.
    #[tokio::test]
    async fn completed_request_is_recorded_in_request_metrics() {
        let engine_model = "request-metrics-model";
        let engine = qwen3_engine(SimEngine::new(), engine_model);

        let stream = engine
            .generate(lowered("metrics", chat_request(3, &[])))
            .await
            .unwrap();
        let (reason, usage) = finish(stream).await;
        assert_eq!(reason, FinishReason::Length);
        assert_eq!(usage.output_tokens, 3);

        let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
        for expected in [
            format!("uniserve:request_success_total{{{labels},finished_reason=\"length\"}} 1"),
            format!(
                "uniserve:prompt_tokens_total{{{labels}}} {}",
                usage.prompt_tokens
            ),
            format!("uniserve:generation_tokens_total{{{labels}}} 3"),
            format!("uniserve:request_prompt_tokens_count{{{labels}}} 1"),
            format!("uniserve:request_generation_tokens_count{{{labels}}} 1"),
            format!("uniserve:request_max_num_generation_tokens_count{{{labels}}} 1"),
            format!("uniserve:e2e_request_latency_seconds_count{{{labels}}} 1"),
            format!("uniserve:request_queue_time_seconds_count{{{labels}}} 1"),
            format!("uniserve:time_to_first_token_seconds_count{{{labels}}} 1"),
        ] {
            assert!(rendered_metrics_hold(&expected), "missing `{expected}`");
        }
        engine.shutdown().await.unwrap();
    }

    /// Each request is recorded under the scheduler that served it.
    #[tokio::test]
    async fn request_metrics_name_the_engine_core_that_served_each_request() {
        let engine_model = "core-metrics-model";
        let cores = (0..2)
            .map(|_| sim_core(EngineConfig::sim(engine_model), SimEngine::new()))
            .collect();
        let engine = sim_engine_over(ModelParameters::Qwen3, cores);

        // Both requests are submitted before either stream is read, so the
        // first is still in flight when the second is routed.
        let mut streams = Vec::new();
        for id in ["core-a", "core-b"] {
            let request = lowered(id, chat_request(3, &[]));
            streams.push(engine.generate(request).await.unwrap());
        }
        for stream in streams {
            assert_eq!(finish(stream).await.0, FinishReason::Length);
        }

        for core in 0..2 {
            let expected = format!(
                "uniserve:request_success_total{{model_name=\"{engine_model}\",\
                 engine=\"{core}\",finished_reason=\"length\"}} 1"
            );
            assert!(rendered_metrics_hold(&expected), "missing `{expected}`");
        }
        engine.shutdown().await.unwrap();
    }

    /// A client that disconnects mid-generation is counted as an aborted
    /// request with its latency, while its partial output, which never
    /// reached a usage report, adds nothing to the token metrics.
    #[tokio::test]
    async fn disconnected_request_is_recorded_as_aborted_without_token_usage() {
        let engine_model = "disconnect-metrics-model";
        let mut worker = SimEngine::new();
        worker.set_text_len(1_000_000);
        let engine = qwen3_engine(worker, engine_model);

        let mut stream = engine
            .generate(lowered("disconnect", chat_request(1_000, &[])))
            .await
            .unwrap();
        loop {
            match stream.next().await.unwrap().unwrap() {
                RequestOutput::TextDelta { text } if !text.is_empty() => break,
                _ => {}
            }
        }
        drop(stream);

        let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
        for expected in [
            format!("uniserve:request_success_total{{{labels},finished_reason=\"abort\"}} 1"),
            format!("uniserve:e2e_request_latency_seconds_count{{{labels}}} 1"),
        ] {
            assert!(rendered_metrics_hold(&expected), "missing `{expected}`");
        }
        let rendered = METRICS.render().unwrap();
        for family in [
            "uniserve:generation_tokens_total",
            "uniserve:request_generation_tokens_count",
        ] {
            let series = format!("{family}{{{labels}}}");
            assert!(
                !rendered.lines().any(|line| line.starts_with(&series)),
                "unexpected `{series}` in:\n{rendered}"
            );
        }
        engine.shutdown().await.unwrap();
    }

    /// Scheduling alone, within the batch the workers were started for, is
    /// taken in place; a limit beyond it, any other change, or no change
    /// starts a standby engine.
    #[test]
    fn scheduling_within_the_workers_batch_changes_in_place() {
        use super::configurable;
        use proto::request::wire;

        let scheduling = |policy: &str, batch, tokens: Option<u32>, running| wire::Scheduling {
            policy: policy.to_owned(),
            max_batch: batch,
            max_num_batched_tokens: tokens,
            max_running_requests: running,
            ..Default::default()
        };
        let spec = |scheduling| wire::EngineSpec {
            scheduling: Some(scheduling),
            ..Default::default()
        };
        let served = spec(scheduling("fcfs", 128, None, 128));
        let capacity = wire::BatchCapacity {
            max_batch: 128,
            max_num_batched_tokens: 8192,
            max_running_requests: 128,
        };
        let within = [
            spec(scheduling("fcfs", 64, Some(4096), 64)),
            spec(scheduling("priority", 128, None, 128)),
            spec(scheduling("fcfs", 128, Some(8192), 128)),
        ];
        for updated in &within {
            assert!(
                configurable(&served, updated, Some(&capacity)),
                "{updated:?}"
            );
            assert!(!configurable(&served, updated, None), "{updated:?}");
        }
        let beyond = [
            spec(scheduling("fcfs", 256, None, 128)),
            spec(scheduling("fcfs", 128, Some(16384), 128)),
            spec(scheduling("fcfs", 128, None, 256)),
            served.clone(),
            wire::EngineSpec {
                cuda_graphs: Some(wire::CudaGraphs::default()),
                ..within[0].clone()
            },
        ];
        for updated in &beyond {
            assert!(
                !configurable(&served, updated, Some(&capacity)),
                "{updated:?}"
            );
        }
        // A step's tokens the served spec states cannot return to the
        // engine's default in place.
        let stated = spec(scheduling("fcfs", 128, Some(4096), 128));
        assert!(!configurable(
            &stated,
            &spec(scheduling("priority", 128, None, 128)),
            Some(&capacity)
        ));
    }
}
