//! The serving engine: one model's input processing, its engine cores, and
//! the output processing of every request.
//!
//! An [`Engine`] owns the model's [`InputProcessor`] and its
//! [`EngineCore`]s, one per data-parallel copy of scheduling, as vLLM's
//! `AsyncLLM` owns its input and output processors over its engine cores and
//! SGLang's `Engine` its tokenizer manager over one scheduler per
//! data-parallel rank. Every request takes one path:
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
use engine::{EngineCore, EventRx, SubmitError, VideoDenoiserInfo};
use engine::{GenerationLimits, Request};
use observability::{EngineLabels, METRICS};
use tokio_util::task::AbortOnDropHandle;
use tracing::info;

use crate::options::{LoggingOptions, ReasoningParser, ServeConfig, Stated};
use crate::profile::{ModelConfig, ModelParameters};
use crate::serving::media::MediaConnector;
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
    /// The engine cores; every engine core serves the same model with the
    /// same limits.
    cores: Box<[Core]>,
    /// The engine core a load scan starts from, advanced by every pick so
    /// that equal loads rotate across the engine cores.
    next_core: AtomicUsize,
    images: MediaConnector,
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

/// One engine core and the requests in flight on it.
struct Core {
    /// The engine core's index, the `engine` label of its metrics.
    index: u32,
    engine: Arc<EngineCore>,
    /// Requests submitted to this engine core whose event stream has not
    /// ended.
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
            worker_groups = ?engine_config.worker_groups,
            block_size = ?engine_config.worker_process.block_size,
            "starting UniServe Rust engine"
        );
        let cores = tokio::task::spawn_blocking(move || EngineCore::start_all(engine_config))
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
                    .media_units(engine::MediaCall::LatentEncoding)
                    .unwrap_or(1);
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
            first.generation_limits(),
            config.chat_config.reasoning_parser == ReasoningParser::Auto,
        )
        .context("failed to bind the configured model description")?;
        let images = MediaConnector::from_image_options(&config.image_fetch_config)
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
    /// engine cores.
    ///
    /// `images` resolves the image references of every request, and
    /// `readout` serves `POST /v1/systemone` for a model that has one. One
    /// task per core publishes a scheduler-stats snapshot every second under
    /// the core's index.
    ///
    /// # Errors
    ///
    /// Fails when there is no core, or when the cores report different
    /// generation limits: a request is prepared once against them and may
    /// land on any engine core.
    pub fn new(
        served_model_name: String,
        cores: Vec<EngineCore>,
        processor: InputProcessor,
        images: MediaConnector,
        readout: Option<ReadoutEncoder>,
        logging: &LoggingOptions,
    ) -> anyhow::Result<Self> {
        let limits = cores
            .first()
            .context("an engine needs at least one engine core")?
            .generation_limits();
        anyhow::ensure!(
            cores.iter().all(|core| core.generation_limits() == limits),
            "engine cores report different generation limits"
        );

        let cores: Box<[Core]> = (0u32..)
            .zip(cores)
            .map(|(index, engine)| Core {
                index,
                engine: Arc::new(engine),
                in_flight: Arc::default(),
            })
            .collect();
        let model_name = cores[0].engine.model_name().to_owned();
        let stats_export = cores
            .iter()
            .map(|core| {
                let stats = Arc::clone(core.engine.stats());
                let model_name = model_name.clone();
                let index = core.index;
                AbortOnDropHandle::new(tokio::spawn(async move {
                    // The reporter turns cumulative scheduler counters into
                    // per-snapshot increments, so one reporter lives across
                    // ticks.
                    let mut reporter = engine::SchedulerStatsReporter::default();
                    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
                    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                    loop {
                        interval.tick().await;
                        let snapshot = reporter.snapshot(&stats);
                        metrics::record_scheduler_stats(
                            &METRICS.scheduler,
                            &model_name,
                            index,
                            &snapshot,
                        );
                    }
                }))
            })
            .collect();
        let stats_logger = logging
            .log_stats
            .then(|| StatsLogger::start(model_name, cores.len()));

        Ok(Self {
            served_model_name,
            processor: Arc::new(processor),
            cores,
            next_core: AtomicUsize::new(0),
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
        self.cores[0].engine.generation_limits()
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

    /// Whether the engine can accept requests: whether any engine core is
    /// alive. Requests go only to live engine cores.
    pub fn is_healthy(&self) -> bool {
        self.cores.iter().any(|core| !core.engine.is_dead())
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
        let (events, core) = self.submit(request.into())?;
        let output =
            serving::generation_output(request_id.clone(), started, compile_us, output, events);
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
        let (events, core) = self.submit(Request::Diffusion(request))?;
        let output = serving::media_output(
            request_id.clone(),
            started,
            elapsed_us(started),
            prompt_tokens,
            events,
        );
        Ok((processed, self.stats(request_id, started, core, output)))
    }

    /// Records `output`'s request metrics under the engine core that serves
    /// it.
    fn stats(
        &self,
        request_id: String,
        started: Instant,
        core: u32,
        output: RequestOutputStream,
    ) -> RequestOutputStream {
        let labels = EngineLabels {
            model_name: self.cores[0].engine.model_name().to_owned(),
            engine: core,
        };
        let log = self.log_requests.then_some(request_id);
        Box::pin(RequestStats::new(output, labels, started, log))
    }

    /// Submits one tokenized request to the least loaded live engine core.
    ///
    /// Returns the request's event stream and the index of the engine core
    /// that serves it. The engine request identifier comes from the first
    /// core's allocator alone, so identifiers are unique across the engine
    /// cores. Dropping the stream before its terminal event cancels the
    /// request.
    ///
    /// # Errors
    ///
    /// Fails with [`SubmitError::Dead`] when no core is alive, and with the
    /// chosen core's submission error.
    pub(crate) fn submit(&self, mut request: Request) -> Result<(EventRx, u32), SubmitError> {
        let core = self.least_loaded().ok_or(SubmitError::Dead)?;
        let request_id = self.cores[0].engine.next_request_id();
        match &mut request {
            Request::Generation(request) => request.request_id = request_id,
            Request::Diffusion(request) => request.request_id = request_id,
        }
        // Counted before submission, so that a pick racing this one already
        // sees the request.
        core.in_flight.fetch_add(1, Ordering::Relaxed);
        let mut events = core.engine.submit(request).inspect_err(|_| {
            core.in_flight.fetch_sub(1, Ordering::Relaxed);
        })?;
        let in_flight = Arc::clone(&core.in_flight);
        events.set_on_finish(move || {
            in_flight.fetch_sub(1, Ordering::Relaxed);
        });
        Ok((events, core.index))
    }

    /// The live engine core with the fewest requests in flight.
    ///
    /// The scan starts one engine core later after every pick, so equal
    /// loads rotate across the engine cores. This is SGLang's `total_requests` data-parallel
    /// balancing (the argmin of running plus waiting requests,
    /// `data_parallel_controller.py`) and vLLM's data-parallel client score
    /// with equal waiting and running weights and a rotating start
    /// (`v1/engine/core_client.py`). Both read load snapshots the ranks
    /// publish; the cores here share the engine's process, so the counts are
    /// exact at every pick. Returns `None` when every core is dead.
    fn least_loaded(&self) -> Option<&Core> {
        let count = self.cores.len();
        let start = self.next_core.fetch_add(1, Ordering::Relaxed) % count;
        (0..count)
            .map(|offset| &self.cores[(start + offset) % count])
            .filter(|core| !core.engine.is_dead())
            .min_by_key(|core| core.in_flight.load(Ordering::Relaxed))
    }

    /// Shuts every engine core down and waits until their scheduler threads
    /// have exited.
    ///
    /// Each scheduler finishes its outstanding requests as aborted and closes
    /// its executor, which on a multi-worker deployment waits for every
    /// worker to drain and exit. The cores shut down concurrently on the blocking pool,
    /// leaving the calling runtime thread free. Repeated calls are harmless.
    ///
    /// # Errors
    ///
    /// Fails when a blocking shutdown task does not complete, which happens
    /// only if it panics or its runtime is shutting down.
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        let tasks = self
            .cores
            .iter()
            .map(|core| {
                let core = Arc::clone(&core.engine);
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
fn special_token_ids(model: &ModelConfig) -> engine::SpecialTokenIds {
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
    engine::SpecialTokenIds {
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

    use engine::{EngineConfig, EngineCore, Executor, SimEngine, SimExecutor};
    use engine::{
        EngineCoreOutput, GenerationConstraint, GenerationRequest, ImageGenerationConfig, Request,
    };
    use futures::{StreamExt as _, poll};
    use ipc::{ImageParams, SamplingParams};
    use observability::METRICS;

    use super::Engine;
    use crate::openai::ChatCompletionRequest;
    use crate::openai::chat_completions::{
        ChatResponseContext, collect_chat_completion, lower_chat_request,
    };
    use crate::profile::ModelParameters;
    use crate::serving::{FinishReason, GenerateRequest, RequestOutput};
    use crate::test_support::{SERVED_MODEL, sim_core, sim_engine_over};

    /// A Qwen3 engine over one engine core, named `engine_model`, that drives
    /// `worker`.
    ///
    /// The engine core's model name labels the process-wide request metrics,
    /// so a test that reads them names its own engine.
    fn qwen3_engine(worker: SimEngine, engine_model: &str) -> Engine {
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
        mut stream: crate::serving::RequestOutputStream,
    ) -> (FinishReason, crate::serving::RequestUsage) {
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
        }
        .into()
    }

    /// Two requests in flight together land on different engine cores, each
    /// served entirely by its own engine core.
    #[tokio::test]
    async fn concurrent_requests_spread_over_engine_cores() {
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
        let mut serving = Vec::new();
        for _ in 0..2 {
            let (stream, core) = engine.submit(text_request()).expect("submit request");
            streams.push(stream);
            serving.push(core);
        }
        assert_ne!(serving[0], serving[1]);
        for stream in &mut streams {
            while let Some(event) = stream.next().await {
                if let EngineCoreOutput::Finished { reason, .. } = event {
                    assert_eq!(reason, engine::FinishReason::Eos);
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

    /// Each request is recorded under the engine core that served it.
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
}
