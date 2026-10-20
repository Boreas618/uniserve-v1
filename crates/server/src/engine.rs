//! The serving engine: one model's input processing, its engine cores, and
//! the output processing of every request.
//!
//! An [`Engine`] owns the model's [`InputProcessor`] and one [`EngineCore`]
//! per data-parallel rank, as vLLM's `AsyncLLM` owns its input and output
//! processors over its engine cores and SGLang's `Engine` its tokenizer
//! manager over one scheduler per data-parallel rank. Every request takes one
//! path:
//!
//! 1. its images are fetched on the async runtime and its prompt rendered and
//!    tokenized on the blocking pool ([`InputProcessor`]);
//! 2. [`Engine::submit`] hands the tokenized request to the least loaded live
//!    engine core, which returns the request's [`EventRx`];
//! 3. the events are assembled into public [`RequestOutput`] events
//!    ([`serving::generation_output`], [`serving::media_output`]) and wrapped
//!    in [`RequestStats`], which records the request's metrics once.
//!
//! Dropping the returned stream drops its `EventRx`, which cancels the
//! request in its engine core; that is the only cancellation path.
//!
//! [`RequestOutput`]: crate::serving::RequestOutput

mod log_stats;
pub(crate) mod metrics;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Context as _;
use tokio_util::task::AbortOnDropHandle;
use tracing::info;
use uniserve_core::{GenerationLimits, Request};
use uniserve_engine::{EngineCore, EventRx, SubmitError, VideoDenoiserInfo};
use uniserve_observability::{EngineLabels, METRICS};

use crate::options::{LoggingOptions, ReasoningParser, ServeConfig, Stated};
use crate::profile::{ModelConfig, ModelParameters};
use crate::serving::media::ImageFetcher;
use crate::serving::systemone::ReadoutEncoder;
use crate::serving::video::{ProcessedVideo, VideoGenerationRequest, VideoService};
use crate::serving::{
    self, GenerateRequest, InputProcessor, LoadedModel, LoadedRenderer, Prompt,
    RequestOutputStream, RequestRenderer, ServeError, TokenizeError,
};
use log_stats::StatsLogger;
pub(crate) use metrics::RequestStats;

/// One model served over its engine cores.
///
/// Construction spawns Tokio tasks, so every constructor must be called
/// within a Tokio runtime.
pub struct Engine {
    /// The single model name the API serves.
    served_model_name: String,
    processor: Arc<InputProcessor>,
    /// One engine core per data-parallel rank; every rank serves the same
    /// model with the same limits.
    replicas: Box<[Replica]>,
    /// The rank a load scan starts from, advanced by every pick so that equal
    /// loads rotate across the ranks.
    next_replica: AtomicUsize,
    images: ImageFetcher,
    /// The System One readout encoder of a model that serves the readout.
    readout: Option<Arc<ReadoutEncoder>>,
    /// Whether every request logs a summary line when it ends.
    log_requests: bool,
    /// Tasks exporting each core's scheduler statistics, stopped with the
    /// engine.
    _stats_export: Box<[AbortOnDropHandle<()>]>,
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

/// One data-parallel rank: its engine core and the requests in flight on it.
struct Replica {
    /// Data-parallel rank, the `engine` label of the rank's metrics.
    rank: u32,
    core: Arc<EngineCore>,
    /// Requests submitted to this rank whose event stream has not ended.
    in_flight: Arc<AtomicUsize>,
}

impl Engine {
    /// Loads the configured model and starts its engine cores and worker
    /// processes.
    ///
    /// `config` must have passed [`ServeConfig::validate`]; `stated` holds the
    /// option paths the operator set. Starting the cores blocks until every
    /// rank has loaded the model, so it runs on the blocking pool.
    ///
    /// # Errors
    ///
    /// Fails when the model assets cannot be resolved, when an option of
    /// another model family was stated ([`Stated::refuse_foreign_options`]),
    /// when the options do not derive an engine configuration
    /// ([`ServeConfig::engine_config`]), when the engine cores fail to start,
    /// or when the model cannot be bound to the capabilities its workers
    /// report.
    pub async fn start(config: &ServeConfig, stated: &Stated) -> anyhow::Result<Self> {
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
        // the configured resolutions and aspect ratios, refusing a pair it
        // does not generate, and report them back as the only canvases
        // requests may target. Other models prepare none.
        if !matches!(model_config.parameters, ModelParameters::MiniMaxH3) {
            worker.video_resolutions.clear();
            worker.video_aspect_ratios.clear();
        }
        // Every generated canvas uses the served sampling, which the worker
        // sizes its canvas state for and prepares its canvas steps with.
        worker.canvas_sampling = match &model_config.parameters {
            ModelParameters::DiffusionGemma(profile) => Some(profile.sampling),
            _ => None,
        };
        // A DiffusionGemma prompt, image block or committed block only
        // conditions canvases: its prefill writes the prefix cache alone.
        worker.prefill_outputs =
            !matches!(model_config.parameters, ModelParameters::DiffusionGemma(_));

        info!(
            workers = ?engine_config.workers,
            block_size = ?engine_config.worker_process.block_size,
            "starting UniServe Rust engine"
        );
        let cores = tokio::task::spawn_blocking(move || EngineCore::replicas(engine_config))
            .await
            .context("the engine startup task did not complete")?
            .context("failed to start the UniServe engine")?;
        let first = cores.first().context("the engine started no engine core")?;

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
                let denoiser = first
                    .info()
                    .video_denoiser
                    .clone()
                    .context("the video worker reported no video denoiser")?;
                // Reference images are banded to the units one latent
                // encoding round covers on the deployment's latent encoder
                // while a request's prompt is encoded.
                let latent_encoding_lane = first
                    .media_lane_units(uniserve_engine::MediaCall::LatentEncoding)
                    .unwrap_or(1);
                // The video processor's options are its reference sizing,
                // within what the denoiser declares.
                let references = uniserve_core::video::plan::ReferenceConfig::from_kwargs(
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
            first.generation_limits(),
            config.chat_config.reasoning_parser == ReasoningParser::Auto,
        )
        .context("failed to bind the configured model description")?;
        let images = ImageFetcher::new(&config.image_fetch_config)
            .context("failed to initialize the image URL fetcher")?;
        Self::new(
            config.served_model_name().to_owned(),
            cores,
            processor,
            images,
            readout,
            &config.logging_config,
        )
    }

    /// Serves `processor`'s model under `served_model_name` over started
    /// engine cores, one per data-parallel rank.
    ///
    /// `images` resolves the image references of every request, and
    /// `readout` serves `POST /v1/systemone` for a model that has one. One
    /// task per core publishes a scheduler-stats snapshot every second under
    /// the core's rank.
    ///
    /// # Errors
    ///
    /// Fails when there is no core, or when the cores report different
    /// generation limits: a request is prepared once against them and may
    /// land on any rank.
    pub fn new(
        served_model_name: String,
        cores: Vec<EngineCore>,
        processor: InputProcessor,
        images: ImageFetcher,
        readout: Option<ReadoutEncoder>,
        logging: &LoggingOptions,
    ) -> anyhow::Result<Self> {
        let limits = cores
            .first()
            .context("an engine needs at least one engine core")?
            .generation_limits();
        anyhow::ensure!(
            cores.iter().all(|core| core.generation_limits() == limits),
            "data-parallel ranks report different generation limits"
        );

        let replicas: Box<[Replica]> = (0u32..)
            .zip(cores)
            .map(|(rank, core)| Replica {
                rank,
                core: Arc::new(core),
                in_flight: Arc::default(),
            })
            .collect();
        let model_name = replicas[0].core.model_name().to_owned();
        let stats_export = replicas
            .iter()
            .map(|replica| {
                let stats = Arc::clone(replica.core.stats());
                let model_name = model_name.clone();
                let rank = replica.rank;
                AbortOnDropHandle::new(tokio::spawn(async move {
                    // The reporter turns cumulative scheduler counters into
                    // per-snapshot increments, so one instance lives across
                    // ticks.
                    let mut reporter = uniserve_engine::SchedulerStatsReporter::default();
                    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
                    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                    loop {
                        interval.tick().await;
                        let snapshot = reporter.snapshot(&stats);
                        metrics::record_scheduler_stats(
                            &METRICS.scheduler,
                            &model_name,
                            rank,
                            &snapshot,
                        );
                    }
                }))
            })
            .collect();
        let stats_logger = logging
            .log_stats
            .then(|| StatsLogger::start(model_name, replicas.len()));

        Ok(Self {
            served_model_name,
            processor: Arc::new(processor),
            replicas,
            next_replica: AtomicUsize::new(0),
            images,
            readout: readout.map(Arc::new),
            log_requests: logging.log_requests,
            _stats_export: stats_export,
            _stats_logger: stats_logger,
        })
    }

    /// The loaded model's input processor.
    pub fn processor(&self) -> &InputProcessor {
        &self.processor
    }

    /// The single model name the API serves.
    pub fn served_model_name(&self) -> &str {
        &self.served_model_name
    }

    /// The generation limits every engine core serves.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.replicas[0].core.generation_limits()
    }

    /// The video denoiser this deployment serves; `None` for a model that
    /// generates no video.
    pub fn video_denoiser(&self) -> Option<&VideoDenoiserInfo> {
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

    /// Whether the engine can accept requests: whether any data-parallel
    /// rank's core is alive. Requests go only to live ranks.
    pub fn is_healthy(&self) -> bool {
        self.replicas.iter().any(|replica| !replica.core.is_dead())
    }

    /// The image fetcher every request's image references go through.
    pub(crate) fn images(&self) -> &ImageFetcher {
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
    /// request, or no engine core accepts it.
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
        let images =
            self.images
                .fetch_all(image_urls)
                .await
                .map_err(|source| ServeError::ImageInput {
                    request_id: request_id.clone(),
                    source,
                })?;
        let processor = Arc::clone(&self.processor);
        let (request, output) =
            preprocess(&request_id, move || processor.preprocess(request, images)).await?;

        let compile_us = elapsed_us(started);
        let (events, rank) = self.submit(request.into())?;
        let output =
            serving::generation_output(request_id.clone(), started, compile_us, output, events);
        Ok(self.stats(request_id, started, rank, output))
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
    /// request, or no engine core accepts it.
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
        let (request, processed) = video
            .process(&request_id, &request, self.processor.config().max_model_len)
            .await?;
        let prompt_tokens = request.prompt_token_ids.len();
        let (events, rank) = self.submit(Request::Diffusion(request))?;
        let output = serving::media_output(
            request_id.clone(),
            started,
            elapsed_us(started),
            prompt_tokens,
            events,
        );
        Ok((processed, self.stats(request_id, started, rank, output)))
    }

    /// Records `output`'s request metrics under the rank that serves it.
    fn stats(
        &self,
        request_id: String,
        started: Instant,
        rank: u32,
        output: RequestOutputStream,
    ) -> RequestOutputStream {
        let labels = EngineLabels {
            model_name: self.replicas[0].core.model_name().to_owned(),
            engine: rank,
        };
        let log = self.log_requests.then_some(request_id);
        Box::pin(RequestStats::new(output, labels, started, log))
    }

    /// Submits one tokenized request to the least loaded live engine core.
    ///
    /// Returns the request's event stream and the data-parallel rank that
    /// serves it. The engine request identifier comes from the first core's
    /// allocator alone, so identifiers are unique across the ranks. Dropping
    /// the stream before its terminal event cancels the request.
    ///
    /// # Errors
    ///
    /// Fails with [`SubmitError::Dead`] when no core is alive, and with the
    /// chosen core's submission error.
    pub(crate) fn submit(&self, mut request: Request) -> Result<(EventRx, u32), SubmitError> {
        let replica = self.least_loaded().ok_or(SubmitError::Dead)?;
        let request_id = self.replicas[0].core.next_request_id();
        match &mut request {
            Request::Generation(request) => request.request_id = request_id,
            Request::Diffusion(request) => request.request_id = request_id,
        }
        // Counted before submission, so that a pick racing this one already
        // sees the request.
        replica.in_flight.fetch_add(1, Ordering::Relaxed);
        let mut events = replica.core.submit(request).inspect_err(|_| {
            replica.in_flight.fetch_sub(1, Ordering::Relaxed);
        })?;
        let in_flight = Arc::clone(&replica.in_flight);
        events.set_on_finish(move || {
            in_flight.fetch_sub(1, Ordering::Relaxed);
        });
        Ok((events, replica.rank))
    }

    /// The live data-parallel rank with the fewest requests in flight.
    ///
    /// The scan starts one rank later after every pick, so equal loads rotate
    /// across the ranks. This is SGLang's `total_requests` data-parallel
    /// balancing (the argmin of running plus waiting requests,
    /// `data_parallel_controller.py`) and vLLM's data-parallel client score
    /// with equal waiting and running weights and a rotating start
    /// (`v1/engine/core_client.py`). Both read load snapshots the ranks
    /// publish; the cores here share the engine's process, so the counts are
    /// exact at every pick. Returns `None` when every core is dead.
    fn least_loaded(&self) -> Option<&Replica> {
        let count = self.replicas.len();
        let start = self.next_replica.fetch_add(1, Ordering::Relaxed) % count;
        (0..count)
            .map(|offset| &self.replicas[(start + offset) % count])
            .filter(|replica| !replica.core.is_dead())
            .min_by_key(|replica| replica.in_flight.load(Ordering::Relaxed))
    }

    /// Shuts every engine core down and waits until their scheduler threads
    /// have exited.
    ///
    /// Each scheduler finishes its outstanding requests as aborted and closes
    /// its executor, which on a multi-rank deployment waits for every rank to
    /// drain and exit. The cores shut down concurrently on the blocking pool,
    /// leaving the calling runtime thread free. Repeated calls are harmless.
    ///
    /// # Errors
    ///
    /// Fails when a blocking shutdown task does not complete, which happens
    /// only if it panics or its runtime is shutting down.
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        let tasks = self
            .replicas
            .iter()
            .map(|replica| {
                let core = Arc::clone(&replica.core);
                tokio::task::spawn_blocking(move || core.shutdown())
            })
            .collect::<Vec<_>>();
        for task in tasks {
            task.await.context("engine shutdown did not complete")?;
        }
        Ok(())
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
fn special_token_ids(model: &ModelConfig) -> uniserve_engine::SpecialTokenIds {
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
    uniserve_engine::SpecialTokenIds {
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

    use uniserve_core::{
        EngineCoreOutput, GenerationConstraint, GenerationRequest, ImageGenerationConfig,
        ImageParams, Request, SamplingParams,
    };
    use uniserve_engine::{EngineConfig, EngineCore, Executor, SimEngine, SimExecutor};

    use crate::http::test_support::sim_engine_over;
    use crate::profile::ModelParameters;

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
        }
        .into()
    }

    /// Two requests in flight together land on different data-parallel
    /// ranks, each served entirely by its own rank.
    #[tokio::test]
    async fn concurrent_requests_spread_over_data_parallel_ranks() {
        let mut cores = Vec::new();
        let mut observers = Vec::new();
        for _ in 0..2 {
            let mut executor = SimExecutor::new(SimEngine::new());
            observers.push(executor.observe());
            cores.push(
                EngineCore::with_executor(EngineConfig::sim("sim-model"), Box::new(executor))
                    .expect("start a sim engine core"),
            );
        }
        let engine = sim_engine_over(ModelParameters::Qwen3, cores);

        // Both requests are submitted before either stream is read, so the
        // first is still in flight when the second is routed.
        let mut streams = Vec::new();
        let mut ranks = Vec::new();
        for _ in 0..2 {
            let (stream, rank) = engine.submit(text_request()).expect("submit request");
            streams.push(stream);
            ranks.push(rank);
        }
        assert_ne!(ranks[0], ranks[1]);
        for stream in &mut streams {
            while let Some(event) = stream.next().await {
                if let EngineCoreOutput::Finished { reason, .. } = event {
                    assert_eq!(reason, uniserve_core::FinishReason::Eos);
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
                        uniserve_engine::BatchEvent::Submitted(batch) => Some(batch),
                        uniserve_engine::BatchEvent::Resolved { .. } => None,
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
        fn info(&self) -> &uniserve_engine::ExecutorInfo {
            self.inner.info()
        }

        fn is_ready(&self, worker: &uniserve_engine::WorkerId) -> bool {
            self.inner.is_ready(worker)
        }

        fn has_capacity(&self, worker: &uniserve_engine::WorkerId) -> bool {
            self.inner.has_capacity(worker)
        }

        fn command_has_capacity(&self, command: &uniserve_ipc::BatchCommand) -> bool {
            self.inner.command_has_capacity(command)
        }

        fn submit(
            &mut self,
            batch: uniserve_engine::ExecutionBatch,
        ) -> Result<(), uniserve_engine::ExecutorSubmitError> {
            self.inner.submit(batch)
        }

        fn poll(
            &mut self,
            timeout: Duration,
        ) -> Result<Option<uniserve_engine::BatchResult>, uniserve_engine::ExecutorError> {
            self.inner.poll(timeout)
        }

        fn close(&mut self) -> Result<(), uniserve_engine::ExecutorError> {
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
        let core = EngineCore::with_executor(
            EngineConfig::sim("sim-model"),
            Box::new(GatedCloseExecutor {
                inner: SimExecutor::new(SimEngine::new()),
                release,
                released: Arc::clone(&released),
            }),
        )
        .expect("start a sim engine core");
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
}
