//! Serving options: one typed configuration tree, read from a TOML file,
//! a few command-line flags and `--set PATH=VALUE` assignments.
//!
//! [`ServeConfig`] is the single declaration of every option, its type and
//! its default. Its top level holds the options most deployments state; the
//! rest sit in `*_config` tables, one per owner of the behavior they
//! configure (`cuda_graph_config` for graph capture, `moe_config` for expert
//! parallelism, `video_config` for MiniMax-H3 video generation, and so on).
//! [`ServeArgs`] reads the tree: a `--config` file first, then the flags that
//! name a path of it, then `--set` assignments, each overriding the last.
//! [`ServeConfig::into_config`] lowers the tree into the server [`Config`].
//!
//! Some tables configure one model family: `chat_config` and
//! `image_fetch_config` chat models, `video_config`, `media_config` and
//! `video_job_config` MiniMax-H3, `readout_config` and
//! `diffusion_generation_config` DiffusionGemma. Startup refuses such a table
//! stated for a checkpoint of another family rather than ignoring it.

mod args;
mod values;

use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uniserve_core::{ModelDtype, PrefixCacheDtype};
use uniserve_engine::{
    DEFAULT_COMPONENT, DEFAULT_MAX_BATCH, DEFAULT_MAX_RUNNING_REQUESTS, Deployment, ExpertExchange,
    FlashInferBackend, GraphPolicy, LaneConfig, LaneSpec, LoadFormat, SchedulingPolicy,
    TransferConfig, WorkerConfig, WorkerProcessArgs,
};

use crate::config::{
    Config, EngineSettings, FamilyOptions, HttpListenerMode, VideoJobSettings, VideoMediaSettings,
};
use crate::profile::diffusion_gemma::DenoisingOverrides;
use crate::serving::chat::ChatTemplateContentFormatOption;
use crate::serving::media::ImageFetchPolicy;
use crate::serving::systemone::{self, CandidateTokens, CanvasMode, ReadoutLayout};

pub use args::{FLAG_PATHS, ServeArgs, Stated, printed};
pub use values::ImageSize;

/// Environment variable read for the API key when `server_config.api_key`
/// is unset.
const API_KEY_ENV: &str = "UNISERVE_API_KEY";

/// The tables of each model family: chat models, MiniMax-H3 video
/// generation, and DiffusionGemma.
const CHAT_TABLES: [&str; 2] = ["chat_config", "image_fetch_config"];
const VIDEO_TABLES: [&str; 3] = ["video_config", "media_config", "video_job_config"];
const DIFFUSION_GEMMA_TABLES: [&str; 2] = ["readout_config", "diffusion_generation_config"];

/// A positive count, for the defaults of non-zero options.
const fn positive(count: usize) -> NonZeroUsize {
    match NonZeroUsize::new(count) {
        Some(count) => count,
        None => NonZeroUsize::MIN,
    }
}

/// Every serving option.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServeConfig {
    /// Local model directory or Hugging Face repository; required.
    pub model: Option<String>,
    /// The single model name the API serves; `model` when unset.
    pub served_model_name: Option<String>,
    /// Longest sequence a request may reach, in tokens. Unset, a checkpoint
    /// with a root `config.json` uses its `max_position_embeddings` (8192
    /// when it declares none) and a MiniMax-H3 pipeline 16384 prompt tokens.
    pub max_model_len: Option<NonZeroU32>,
    /// Element type of parameters and activations.
    pub dtype: ModelDtype,
    /// Precision policy, such as `mode = "balanced"` or per-component
    /// selections under `components`; empty selects the model's default.
    pub quantization_config: serde_json::Map<String, Value>,
    /// Options of the model's multimodal processor; the processor refuses
    /// one it does not take.
    pub mm_processor_kwargs: serde_json::Map<String, Value>,
    /// Attention provider for every attention call, by the name the ranks'
    /// attention registry gives it; `auto` selects a native kernel per call.
    pub attention_backend: String,
    /// Concurrently resident requests, clamped to what the workers hold.
    pub max_running_requests: NonZeroUsize,
    /// Token work admitted in one scheduling step, 8192 when unset; token
    /// models only.
    pub max_num_batched_tokens: Option<NonZeroUsize>,
    /// Each rank's share of its device's total memory for weights, static
    /// allocations and the KV cache, in (0, 1]. A deployment group's
    /// `mem_fraction_static` overrides it.
    pub mem_fraction_static: f64,
    /// Tensor-parallel ranks of each replica when no `workers` file is
    /// given.
    pub tensor_parallel_size: NonZeroUsize,
    /// Independent replicas, each with its own scheduler, KV cache and
    /// ranks. With a `workers` file, its groups form the replicas as equal
    /// consecutive blocks.
    pub data_parallel_size: NonZeroUsize,
    /// Deployment file placing worker groups and their components.
    pub workers: Option<PathBuf>,
    /// This process's host name; it runs the ranks placed on this host, and
    /// a deployment rank entry without a host names it.
    pub host_identity: String,
    /// Execution lanes, each a stream with its own SM partition.
    pub lanes: Vec<LaneSpec>,
    pub server_config: ServerOptions,
    pub load_config: LoadOptions,
    pub kv_cache_config: KvCacheOptions,
    pub scheduler_config: SchedulerOptions,
    pub placement_config: PlacementOptions,
    pub moe_config: MoeOptions,
    pub cuda_graph_config: CudaGraphOptions,
    pub flashinfer_config: FlashInferOptions,
    pub worker_config: WorkerOptions,
    pub logging_config: LoggingOptions,
    pub chat_config: ChatOptions,
    pub image_fetch_config: ImageFetchOptions,
    pub video_config: VideoOptions,
    pub media_config: MediaOptions,
    pub video_job_config: VideoJobOptions,
    pub readout_config: ReadoutOptions,
    pub diffusion_generation_config: DenoisingOverrides,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            model: None,
            served_model_name: None,
            max_model_len: None,
            dtype: WorkerProcessArgs::DEFAULT_DTYPE,
            quantization_config: serde_json::Map::new(),
            mm_processor_kwargs: serde_json::Map::new(),
            attention_backend: WorkerProcessArgs::DEFAULT_ATTENTION_BACKEND.to_owned(),
            max_running_requests: positive(DEFAULT_MAX_RUNNING_REQUESTS),
            max_num_batched_tokens: None,
            mem_fraction_static: WorkerProcessArgs::DEFAULT_MEM_FRACTION_STATIC,
            tensor_parallel_size: NonZeroUsize::MIN,
            data_parallel_size: NonZeroUsize::MIN,
            workers: None,
            host_identity: "localhost".to_owned(),
            lanes: Vec::new(),
            server_config: ServerOptions::default(),
            load_config: LoadOptions::default(),
            kv_cache_config: KvCacheOptions::default(),
            scheduler_config: SchedulerOptions::default(),
            placement_config: PlacementOptions::default(),
            moe_config: MoeOptions::default(),
            cuda_graph_config: CudaGraphOptions::default(),
            flashinfer_config: FlashInferOptions::default(),
            worker_config: WorkerOptions::default(),
            logging_config: LoggingOptions::default(),
            chat_config: ChatOptions::default(),
            image_fetch_config: ImageFetchOptions::default(),
            video_config: VideoOptions::default(),
            media_config: MediaOptions::default(),
            video_job_config: VideoJobOptions::default(),
            readout_config: ReadoutOptions::default(),
            diffusion_generation_config: DenoisingOverrides::default(),
        }
    }
}

/// The HTTP server (`server_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerOptions {
    /// TCP address the server listens on.
    pub host: String,
    /// TCP port the server listens on.
    pub port: u16,
    /// Unix domain socket to listen on instead of TCP; a stale socket file
    /// is replaced and the socket file is removed at shutdown.
    pub uds: Option<String>,
    /// Bearer token the public routes require; `UNISERVE_API_KEY` supplies
    /// it when unset or blank. Never printed.
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
    /// Time until a response head is sent, answered with 504 when it
    /// expires; streamed bodies are not bounded.
    #[serde(with = "values::optional_duration")]
    pub request_timeout: Option<Duration>,
    /// In-flight chat, image generation, System One and synchronous video
    /// requests above which requests are shed with 503.
    pub max_concurrent_requests: Option<NonZeroU64>,
    /// Longest drain of in-flight requests at shutdown; zero stops at once.
    #[serde(with = "values::duration")]
    pub shutdown_timeout: Duration,
    /// Add an `X-Request-Id` response header with the ID the request ran
    /// under.
    pub request_id_headers: bool,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            uds: None,
            api_key: None,
            request_timeout: None,
            max_concurrent_requests: None,
            shutdown_timeout: Duration::from_secs(30),
            request_id_headers: false,
        }
    }
}

/// Checkpoint loading (`load_config`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoadOptions {
    /// Checkpoint reader: `auto`, `safetensors`, `pt`, `layered` (one layer
    /// at a time) or `dummy` (no weights read).
    pub format: LoadFormat,
    /// Hugging Face cache root for repository models.
    pub download_dir: Option<PathBuf>,
    /// Concurrent checkpoint file readers.
    pub threads: Option<NonZeroU32>,
}

/// The KV cache (`kv_cache_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvCacheOptions {
    /// Capacity in tokens; sized from memory when unset.
    pub max_tokens: Option<NonZeroU64>,
    /// Tokens per page of the cache group with the widest token rows, a
    /// power of two; the largest size up to 64 the attention kernels read
    /// when unset.
    pub block_size: Option<NonZeroU32>,
    /// Element type; each layer's compute type when unset.
    pub dtype: Option<PrefixCacheDtype>,
    /// Reuse and retain cached prompt prefixes across requests.
    pub prefix_caching: bool,
}

impl Default for KvCacheOptions {
    fn default() -> Self {
        Self {
            max_tokens: None,
            block_size: None,
            dtype: None,
            prefix_caching: true,
        }
    }
}

/// Admission and batching (`scheduler_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerOptions {
    /// Waiting-queue order: `fcfs` or `priority`.
    pub policy: SchedulingPolicy,
    /// Waiting requests, and finished ones with undelivered events, before
    /// a submission is refused as overloaded; unbounded when unset.
    pub max_queued_requests: Option<NonZeroUsize>,
    /// Prompt tokens one request prefills in one step, 8192 when unset;
    /// token models only.
    pub chunked_prefill_size: Option<NonZeroUsize>,
    /// Calls assembled into one forward batch, clamped to what the workers
    /// accept.
    pub max_batch: NonZeroUsize,
}

impl Default for SchedulerOptions {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Fcfs,
            max_queued_requests: None,
            chunked_prefill_size: None,
            max_batch: positive(DEFAULT_MAX_BATCH),
        }
    }
}

/// Rank placement without a deployment file, and product transport
/// (`placement_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlacementOptions {
    /// Device of every rank; `cuda` numbers each host's GPUs from zero.
    pub device: String,
    /// Hosts after this one the ranks spread across, in equal blocks.
    #[serde(deserialize_with = "values::list::deserialize")]
    pub worker_hosts: Vec<String>,
    /// Batches in flight per rank; a deployment group states its own.
    pub queue_depth: NonZeroUsize,
    /// Product transport per worker edge,
    /// `source[:rank]->destination[:rank]=mechanisms`, comma-separated.
    pub transfer: Option<String>,
}

impl Default for PlacementOptions {
    fn default() -> Self {
        Self {
            device: "cuda".to_owned(),
            worker_hosts: Vec::new(),
            queue_depth: positive(WorkerProcessArgs::DEFAULT_QUEUE_DEPTH),
            transfer: None,
        }
    }
}

/// Mixture-of-experts parallelism (`moe_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MoeOptions {
    /// Shard routed experts across the data-parallel replicas or the
    /// deployment's expert workers.
    pub expert_parallel: bool,
    /// How sharded experts are reached: `alltoall`, `megamoe`, `deepep` or
    /// `dwdp`.
    pub exchange: ExpertExchange,
    /// Microbatches overlapping attention with dedicated expert workers.
    pub microbatches: NonZeroU32,
}

impl Default for MoeOptions {
    fn default() -> Self {
        Self {
            expert_parallel: false,
            exchange: ExpertExchange::default(),
            microbatches: NonZeroU32::MIN,
        }
    }
}

/// CUDA graph capture (`cuda_graph_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaGraphOptions {
    /// `auto` captures graphs unless serving does not capture the attention
    /// provider's calls, `full` requires them, `off` serves eagerly.
    pub policy: GraphPolicy,
    /// Replay a captured graph for every prefill call.
    pub capture_prefill: bool,
    /// Capture and replay image denoising calls.
    pub capture_flow: bool,
    /// Decode batch sizes captured, strictly increasing; the worker's
    /// catalog when unset.
    #[serde(deserialize_with = "values::optional_list::deserialize")]
    pub decode_batch_sizes: Option<Vec<NonZeroU32>>,
    /// Prefill token counts captured, strictly increasing; the worker's
    /// catalog when unset.
    #[serde(deserialize_with = "values::optional_list::deserialize")]
    pub prefill_token_sizes: Option<Vec<NonZeroU32>>,
    /// Image denoising batch sizes captured, strictly increasing; 1 to 4
    /// when unset.
    #[serde(deserialize_with = "values::optional_list::deserialize")]
    pub flow_batch_sizes: Option<Vec<NonZeroU32>>,
    /// Distinct `HEIGHTxWIDTH` image sizes captured; `1152x2048` and
    /// `2048x1152` when unset.
    #[serde(deserialize_with = "values::optional_list::deserialize")]
    pub flow_shapes: Option<Vec<ImageSize>>,
}

impl Default for CudaGraphOptions {
    fn default() -> Self {
        Self {
            policy: GraphPolicy::default(),
            capture_prefill: WorkerProcessArgs::DEFAULT_PREFILL_CUDA_GRAPH,
            capture_flow: WorkerProcessArgs::DEFAULT_FLOW_CUDA_GRAPH,
            decode_batch_sizes: None,
            prefill_token_sizes: None,
            flow_batch_sizes: None,
            flow_shapes: None,
        }
    }
}

/// The FlashInfer attention provider (`flashinfer_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlashInferOptions {
    /// Workspace per rank.
    #[serde(with = "values::bytes")]
    pub workspace_size: u64,
    /// Decode attention on tensor cores; the backend decides when unset.
    pub use_tensor_core: Option<bool>,
    /// Decode kernels: `auto`, `fa2` or `fa3`.
    pub decode_backend: FlashInferBackend,
    /// Prefill kernels: `auto`, `fa2` or `fa3`.
    pub prefill_backend: FlashInferBackend,
    /// Split-KV tile size of decode attention.
    pub decode_split_tile_size: Option<NonZeroU32>,
    /// Split-KV tile size of prefill attention.
    pub prefill_split_tile_size: Option<NonZeroU32>,
    /// Split a KV sequence across thread blocks.
    pub split_kv: bool,
}

impl Default for FlashInferOptions {
    fn default() -> Self {
        Self {
            workspace_size: WorkerProcessArgs::DEFAULT_FLASHINFER_WORKSPACE_SIZE,
            use_tensor_core: None,
            decode_backend: WorkerProcessArgs::DEFAULT_FLASHINFER_DECODE_BACKEND,
            prefill_backend: WorkerProcessArgs::DEFAULT_FLASHINFER_PREFILL_BACKEND,
            decode_split_tile_size: None,
            prefill_split_tile_size: None,
            split_kv: true,
        }
    }
}

/// The worker rank processes (`worker_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerOptions {
    /// Python interpreter that runs the ranks.
    pub python: PathBuf,
    /// Time a batch may run without rank progress before the worker group is
    /// replaced; a rank reports a stalled lane after half of it.
    #[serde(with = "values::duration")]
    pub watchdog_timeout: Duration,
    /// `tower=gen:<device>` places the generation tower on another device
    /// than the rank's.
    pub mesh: Option<String>,
    /// Torch process-group backend for the ranks' collectives.
    pub distributed_backend: Option<String>,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            python: WorkerProcessArgs::default_python(),
            watchdog_timeout: WorkerProcessArgs::DEFAULT_WATCHDOG_TIMEOUT,
            mesh: None,
            distributed_backend: None,
        }
    }
}

/// What the server logs about its serving (`logging_config`). A process's
/// log levels are options of the binary that runs it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingOptions {
    /// Log a summary line for each completed request.
    pub log_requests: bool,
    /// Log engine statistics periodically.
    pub log_stats: bool,
}

impl Default for LoggingOptions {
    fn default() -> Self {
        Self {
            log_requests: false,
            log_stats: true,
        }
    }
}

/// Which reasoning parser splits model output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningParser {
    /// The model description's parser splits `reasoning_content` from
    /// `content`.
    #[default]
    Auto,
    /// Reasoning delimiter tokens stream verbatim as content.
    None,
}

/// Chat models: Qwen3, SenseNova, BAGEL and DiffusionGemma (`chat_config`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatOptions {
    /// Chat template, as a file path or the template itself.
    pub template: Option<String>,
    /// Keyword arguments bound into every chat-template render.
    pub template_kwargs: Option<HashMap<String, Value>>,
    /// How message content renders: `auto`, `string` or `openai`.
    pub template_content_format: ChatTemplateContentFormatOption,
    /// `auto` splits `reasoning_content` from `content`; `none` streams
    /// reasoning delimiters as content.
    pub reasoning_parser: ReasoningParser,
}

/// Fetching input images of chat requests (`image_fetch_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageFetchOptions {
    /// Time to fetch one `http(s)` image URL, redirects and body included.
    #[serde(with = "values::duration")]
    pub timeout: Duration,
    /// Largest input image, fetched or `data:`.
    #[serde(with = "values::bytes")]
    pub max_bytes: u64,
    /// Allow image URLs resolving to loopback, private, link-local,
    /// unique-local or cloud metadata addresses.
    pub allow_private: bool,
}

impl Default for ImageFetchOptions {
    fn default() -> Self {
        Self {
            timeout: ImageFetchPolicy::DEFAULT_TIMEOUT,
            max_bytes: ImageFetchPolicy::DEFAULT_MAX_BYTES,
            allow_private: false,
        }
    }
}

/// MiniMax-H3 video generation (`video_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VideoOptions {
    /// Longest admitted clip, 4 to 15 seconds.
    pub max_seconds: f64,
    /// Resolution classes startup prepares (`768p`, `480p`), each at every
    /// aspect ratio.
    #[serde(deserialize_with = "values::list::deserialize")]
    pub resolutions: Vec<String>,
    /// Aspect ratios startup prepares (`21:9`, `16:9`, `4:3`, `1:1`, `3:4`,
    /// `9:16`).
    #[serde(deserialize_with = "values::list::deserialize")]
    pub aspect_ratios: Vec<String>,
    /// Prompt capacities below `max_model_len`, strictly increasing; 1024,
    /// then steps of 2048, when unset.
    #[serde(deserialize_with = "values::optional_list::deserialize")]
    pub text_capacities: Option<Vec<NonZeroU32>>,
    /// Denoiser rows one request's conditions may take; the largest
    /// condition set the denoiser admits when unset.
    pub max_condition_rows: Option<u32>,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            max_seconds: uniserve_core::video::plan::MAX_VIDEO_SECONDS,
            resolutions: vec!["768p".to_owned()],
            aspect_ratios: vec!["16:9".to_owned(), "9:16".to_owned()],
            text_capacities: None,
            max_condition_rows: None,
        }
    }
}

/// Condition media of video requests (`media_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MediaOptions {
    /// Directory `file://` condition media resolves under; `file://` is
    /// refused when unset.
    pub directory: Option<PathBuf>,
    /// Fetch `http(s)://` condition media.
    pub allow_remote: bool,
    /// Condition media one request may carry in total.
    #[serde(with = "values::bytes")]
    pub max_request_bytes: u64,
    /// Executable that probes video and audio condition media.
    pub ffprobe: PathBuf,
    /// Executable the media reader decodes reference videos with.
    pub ffmpeg: PathBuf,
}

impl Default for MediaOptions {
    fn default() -> Self {
        Self {
            directory: None,
            allow_remote: true,
            max_request_bytes: VideoMediaSettings::DEFAULT_MAX_REQUEST_BYTES,
            ffprobe: PathBuf::from("ffprobe"),
            ffmpeg: PathBuf::from(WorkerProcessArgs::DEFAULT_FFMPEG),
        }
    }
}

/// Asynchronous video jobs (`video_job_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VideoJobOptions {
    /// Time a finished job stays retrievable.
    #[serde(with = "values::duration")]
    pub retention: Duration,
    /// Generated video the jobs keep in shared memory.
    #[serde(with = "values::byte_count")]
    pub max_bytes: usize,
}

impl Default for VideoJobOptions {
    fn default() -> Self {
        Self {
            retention: Duration::from_secs(VideoJobSettings::DEFAULT_RETENTION_SECONDS),
            max_bytes: VideoJobSettings::DEFAULT_MAX_RETAINED_BYTES,
        }
    }
}

/// DiffusionGemma System One readouts (`readout_config`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReadoutOptions {
    /// `joint` packs questions into shared canvases; `independent` gives
    /// each its own prompt and canvas.
    pub layout: ReadoutLayout,
    /// Canvas length: `full`, `compact`, or a positive multiple of 16.
    #[serde(with = "values::canvas")]
    pub canvas: CanvasMode,
    /// `variants` sums supported answer spellings; `primary` reads the
    /// space-prefixed one.
    pub candidates: CandidateTokens,
}

impl ServeConfig {
    /// Lowers the configuration into the server configuration and validates
    /// it.
    ///
    /// The deployment comes from the `workers` file, placed on this
    /// process's `host_identity`, or else from `tensor_parallel_size`,
    /// `data_parallel_size` and `placement_config`. `stated` records the
    /// paths the operator set, so the placement shorthand can be refused
    /// beside a deployment file and the model-family tables checked against
    /// the served model at startup.
    ///
    /// # Errors
    ///
    /// Fails when `model` is unset, a placement-shorthand option is stated
    /// with `workers`, the deployment does not place, a capture list is not
    /// strictly increasing or repeats a shape, a lane or the transfer edges
    /// do not resolve, or [`Config::validate`] refuses the result.
    pub fn into_config(self, stated: &Stated) -> anyhow::Result<Config> {
        let model = self.model.clone().context(
            "`model` is required: name MODEL on the command line or set `model` in --config",
        )?;
        WorkerConfig::validate_mem_fraction_static(self.mem_fraction_static)
            .context("invalid `mem_fraction_static`")?;

        let workers = match &self.workers {
            Some(path) => {
                let shorthand = [
                    "tensor_parallel_size",
                    "placement_config.device",
                    "placement_config.worker_hosts",
                    "placement_config.queue_depth",
                ];
                if let Some(option) = shorthand.iter().find(|option| stated.contains(option)) {
                    anyhow::bail!(
                        "`{option}` places ranks without a deployment file; the `workers` file \
                         states its own placement"
                    );
                }
                Deployment::read(&path.to_string_lossy())?
                    .workers(&self.host_identity)
                    .context("invalid `workers` deployment")?
            }
            None => {
                let tensor_parallel_size = self.tensor_parallel_size.get();
                WorkerConfig::replicated(
                    &self.rank_hosts(),
                    &self.placement_config.device,
                    self.data_parallel_size.get(),
                    tensor_parallel_size,
                    self.placement_config.queue_depth.get(),
                    WorkerConfig::single_component(DEFAULT_COMPONENT, tensor_parallel_size),
                )
            }
        };

        let lanes = self
            .lanes
            .iter()
            .cloned()
            .map(|lane| {
                let id = lane.lane_id.clone();
                LaneConfig::try_from(lane).with_context(|| format!("invalid lane `{id}`"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let transfer = self
            .placement_config
            .transfer
            .as_deref()
            .map(str::parse::<TransferConfig>)
            .transpose()
            .context("invalid `placement_config.transfer`")?
            .unwrap_or_default();

        let graphs = &self.cuda_graph_config;
        let flashinfer = &self.flashinfer_config;
        let worker_process = WorkerProcessArgs {
            python: self.worker_config.python.clone(),
            model: model.clone(),
            host: self.host_identity.clone(),
            queue_depth: self.placement_config.queue_depth.get(),
            watchdog_timeout: self.worker_config.watchdog_timeout,
            max_total_tokens: self.kv_cache_config.max_tokens.map(NonZeroU64::get),
            block_size: self.kv_cache_config.block_size.map(NonZeroU32::get),
            attention_backend: self.attention_backend.clone(),
            expert_microbatches: self.moe_config.microbatches.get(),
            load_format: self.load_config.format,
            download_dir: self.load_config.download_dir.clone(),
            load_threads: self.load_config.threads.map(NonZeroU32::get),
            dtype: self.dtype,
            quantization_config: Value::Object(self.quantization_config.clone()),
            kv_cache_dtype: self.kv_cache_config.dtype,
            mem_fraction_static: self.mem_fraction_static,
            worker_mesh: self.worker_config.mesh.clone(),
            distributed_backend: self.worker_config.distributed_backend.clone(),
            lanes,
            graph_policy: graphs.policy,
            decode_graph_batch_sizes: increasing(
                "cuda_graph_config.decode_batch_sizes",
                &graphs.decode_batch_sizes,
            )?,
            prefill_cuda_graph: graphs.capture_prefill,
            prefill_graph_token_sizes: increasing(
                "cuda_graph_config.prefill_token_sizes",
                &graphs.prefill_token_sizes,
            )?,
            flow_cuda_graph: graphs.capture_flow,
            flow_graph_batch_sizes: increasing(
                "cuda_graph_config.flow_batch_sizes",
                &graphs.flow_batch_sizes,
            )?,
            flow_graph_shapes: distinct("cuda_graph_config.flow_shapes", &graphs.flow_shapes)?,
            video_text_capacities: increasing(
                "video_config.text_capacities",
                &self.video_config.text_capacities,
            )?,
            ffmpeg: self.media_config.ffmpeg.clone(),
            flashinfer_workspace_size: flashinfer.workspace_size,
            flashinfer_use_tensor_core: flashinfer.use_tensor_core,
            flashinfer_decode_backend: flashinfer.decode_backend,
            flashinfer_prefill_backend: flashinfer.prefill_backend,
            flashinfer_decode_split_tile_size: flashinfer
                .decode_split_tile_size
                .map(NonZeroU32::get),
            flashinfer_prefill_split_tile_size: flashinfer
                .prefill_split_tile_size
                .map(NonZeroU32::get),
            flashinfer_split_kv: flashinfer.split_kv,
            ..WorkerProcessArgs::default()
        };

        let engine = EngineSettings {
            max_batch: self.scheduler_config.max_batch.get(),
            max_num_batched_tokens: self.max_num_batched_tokens.map(NonZeroUsize::get),
            max_running_requests: self.max_running_requests.get(),
            max_queued_requests: self
                .scheduler_config
                .max_queued_requests
                .map(NonZeroUsize::get),
            chunked_prefill_size: self
                .scheduler_config
                .chunked_prefill_size
                .map(NonZeroUsize::get),
            prefix_cache: self.kv_cache_config.prefix_caching,
            schedule_policy: self.scheduler_config.policy,
            max_model_len: self.max_model_len.map(NonZeroU32::get),
            max_video_seconds: self.video_config.max_seconds,
            video_resolutions: self.video_config.resolutions.clone(),
            video_aspect_ratios: self.video_config.aspect_ratios.clone(),
            max_condition_rows: self.video_config.max_condition_rows,
            workers,
            transfer,
            data_parallel_size: self.data_parallel_size.get(),
            expert_parallel: self
                .moe_config
                .expert_parallel
                .then_some(self.moe_config.exchange),
            worker_process,
        };

        let server = &self.server_config;
        let listener_mode = match &server.uds {
            Some(path) => HttpListenerMode::BindUnix { path: path.clone() },
            None => HttpListenerMode::BindTcp {
                host: server.host.clone(),
                port: server.port,
            },
        };
        let config = Config {
            engine,
            model,
            served_model_name: self.served_model_name.clone(),
            listener_mode,
            chat_template: self.chat_config.template.clone(),
            default_chat_template_kwargs: self.chat_config.template_kwargs.clone(),
            chat_template_content_format: self.chat_config.template_content_format,
            log_requests: self.logging_config.log_requests,
            request_id_headers: server.request_id_headers,
            log_stats: self.logging_config.log_stats,
            api_key: configured_api_key(server.api_key.as_deref()),
            request_timeout: server.request_timeout,
            max_concurrent_requests: server.max_concurrent_requests.map(NonZeroU64::get),
            shutdown_timeout: server.shutdown_timeout,
            reasoning_parsing: self.chat_config.reasoning_parser == ReasoningParser::Auto,
            video_media: VideoMediaSettings {
                media_directory: self.media_config.directory.clone(),
                remote_media: self.media_config.allow_remote,
                max_request_bytes: self.media_config.max_request_bytes,
                ffprobe: self.media_config.ffprobe.clone(),
            },
            video_jobs: VideoJobSettings {
                retention: self.video_job_config.retention,
                max_retained_bytes: self.video_job_config.max_bytes,
            },
            image_fetch: ImageFetchPolicy {
                timeout: self.image_fetch_config.timeout,
                max_bytes: self.image_fetch_config.max_bytes,
                allow_private: self.image_fetch_config.allow_private,
            },
            readout: systemone::ReadoutOptions {
                layout: self.readout_config.layout,
                canvas: self.readout_config.canvas,
                candidates: self.readout_config.candidates,
            },
            diffusion_generation: self.diffusion_generation_config,
            mm_processor_kwargs: self.mm_processor_kwargs.clone(),
            family_options: FamilyOptions {
                chat: stated.tables(&CHAT_TABLES),
                video: stated.tables(&VIDEO_TABLES),
                diffusion_gemma: stated.tables(&DIFFUSION_GEMMA_TABLES),
            },
        };
        config.validate()?;
        Ok(config)
    }

    /// Returns the hosts the placement shorthand places ranks on, this host
    /// first.
    ///
    /// `WorkerConfig::placed` assigns ranks to these hosts in contiguous
    /// blocks in list order, so leading with this instance's own identity
    /// keeps rank zero and the other lowest ranks on the head's host.
    /// Entries of `worker_hosts` that are empty or repeat this instance's
    /// identity are dropped; other repeated names are kept as given.
    fn rank_hosts(&self) -> Vec<String> {
        let mut hosts = vec![self.host_identity.clone()];
        hosts.extend(
            self.placement_config
                .worker_hosts
                .iter()
                .filter(|host| !host.is_empty() && **host != self.host_identity)
                .cloned(),
        );
        hosts
    }
}

/// Returns the bearer token the server requires, if any.
///
/// The configured key takes precedence over the `UNISERVE_API_KEY`
/// environment variable. Each source is trimmed, and a blank value counts as
/// unset, so a blank configured key falls through to the environment
/// variable.
fn configured_api_key(configured: Option<&str>) -> Option<String> {
    non_empty_secret(configured).or_else(|| {
        std::env::var(API_KEY_ENV)
            .ok()
            .and_then(|value| non_empty_secret(Some(&value)))
    })
}

/// Returns the secret with surrounding whitespace trimmed, or `None` when it
/// is absent or blank.
fn non_empty_secret(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Checks that a stated capture or capacity list is strictly increasing.
fn increasing(path: &str, values: &Option<Vec<NonZeroU32>>) -> anyhow::Result<Option<Vec<u32>>> {
    let Some(values) = values else {
        return Ok(None);
    };
    anyhow::ensure!(
        values.windows(2).all(|pair| pair[0] < pair[1]),
        "`{path}` must be strictly increasing, got {values:?}"
    );
    Ok(Some(values.iter().map(|value| value.get()).collect()))
}

/// Checks that stated image capture sizes are distinct.
fn distinct(path: &str, shapes: &Option<Vec<ImageSize>>) -> anyhow::Result<Option<Vec<[u32; 2]>>> {
    let Some(shapes) = shapes else {
        return Ok(None);
    };
    anyhow::ensure!(
        shapes.iter().collect::<BTreeSet<_>>().len() == shapes.len(),
        "`{path}` repeats an image size"
    );
    Ok(Some(shapes.iter().map(|shape| shape.0).collect()))
}
