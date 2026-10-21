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
//! The tree is the server's configuration: each part of the server reads its
//! own table, and [`ServeConfig::engine_config`] derives the engine
//! configuration the scheduler and placement options state.
//!
//! Some options configure one model family: the token batch budgets,
//! `chat_config` and `image_fetch_config` token models, `video_config`,
//! `media_config` and `video_job_config` MiniMax-H3, `readout_config` and
//! `diffusion_generation_config` DiffusionGemma. Startup refuses such an
//! option stated for a checkpoint of another family rather than ignoring it
//! ([`Stated::refuse_foreign_options`]).

mod args;
mod values;

use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context as _;
use daemon::DaemonOptions;
use engine::{
    DEFAULT_CHUNKED_PREFILL_SIZE, DEFAULT_COMPONENT, DEFAULT_MAX_BATCH,
    DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_RUNNING_REQUESTS, Deployment, EngineConfig,
    ExpertExchange, FlashInferBackend, GraphPolicy, LaneConfig, LaneSpec, LoadFormat,
    SchedulingPolicy, TransferConfig, WorkerGroupConfig, WorkerProcessArgs,
};
use engine::{GenerationLimits, ModelDtype};
use ipc::PrefixCacheDtype;
use models::video::plan::{MAX_VIDEO_SECONDS, MIN_VIDEO_SECONDS};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::profile::ModelDescription;
use crate::profile::diffusion_gemma::DenoisingOverrides;
use crate::serving::chat::ChatTemplateContentFormatOption;
use crate::serving::systemone::{CandidateTokens, CanvasMode, ReadoutLayout};

pub use args::{FLAG_PATHS, ServeArgs, Stated, printed};
pub use values::ImageSize;

/// Environment variable read for the API key when `server_config.api_key`
/// is unset.
const API_KEY_ENV: &str = "UNISERVE_API_KEY";

/// Options that place ranks without a deployment file, refused beside one.
const PLACEMENT_SHORTHAND: [&str; 4] = [
    "tensor_parallel_size",
    "placement_config.device",
    "placement_config.worker_hosts",
    "placement_config.queue_depth",
];

/// The model families whose behavior some options configure, with those
/// options by configuration path.
///
/// Token models schedule token batches and render chat prompts, which a
/// video pipeline does neither of; the video and DiffusionGemma tables
/// configure behavior only their family has.
const FAMILY_OPTIONS: [(Family, &[&str]); 3] = [
    (
        Family::TokenModels,
        &[
            "max_num_batched_tokens",
            "scheduler_config.chunked_prefill_size",
            "chat_config",
            "image_fetch_config",
        ],
    ),
    (
        Family::MiniMaxH3,
        &["video_config", "media_config", "video_job_config"],
    ),
    (
        Family::DiffusionGemma,
        &["readout_config", "diffusion_generation_config"],
    ),
];

/// A group of model families some options belong to.
#[derive(Debug, Clone, Copy)]
enum Family {
    TokenModels,
    MiniMaxH3,
    DiffusionGemma,
}

impl Family {
    /// Whether `description` belongs to this family.
    fn serves(self, description: ModelDescription) -> bool {
        match self {
            Self::TokenModels => !description.is_pipeline(),
            Self::MiniMaxH3 => description == ModelDescription::MiniMaxH3,
            Self::DiffusionGemma => description == ModelDescription::DiffusionGemma,
        }
    }

    /// The family as the refusal names it.
    fn name(self) -> &'static str {
        match self {
            Self::TokenModels => "token models",
            Self::MiniMaxH3 => "MiniMax-H3 video generation",
            Self::DiffusionGemma => "DiffusionGemma",
        }
    }
}

impl Stated {
    /// Refuses a stated option that configures a family other than the
    /// served model's, rather than ignoring it.
    ///
    /// # Errors
    ///
    /// Fails, naming the first such option and the served model, when a
    /// video pipeline states a token-model option, or another model states
    /// a MiniMax-H3 or DiffusionGemma table.
    pub fn refuse_foreign_options(&self, description: ModelDescription) -> anyhow::Result<()> {
        for (family, paths) in FAMILY_OPTIONS {
            if family.serves(description) {
                continue;
            }
            if let Some(path) = paths.iter().find(|path| self.contains(path)) {
                anyhow::bail!(
                    "`{path}` configures {}, but the served checkpoint is a {} model; remove it",
                    family.name(),
                    description.id()
                );
            }
        }
        Ok(())
    }
}

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
    /// Tensor-parallel workers of each scheduler's group when no
    /// `worker_groups` file is given.
    pub tensor_parallel_size: NonZeroUsize,
    /// Schedulers, each with its own scheduler, KV cache and worker
    /// groups. With a `worker_groups` file, its model groups form the engine
    /// cores as equal consecutive blocks.
    pub data_parallel_size: NonZeroUsize,
    /// Deployment file placing worker groups and their components.
    pub worker_groups: Option<PathBuf>,
    /// This node's host name; its daemon runs the ranks placed on this
    /// host, and a deployment rank entry without a host names it.
    pub host_identity: String,
    /// Nodes of the deployment, each running one `uniserve serve`.
    pub nnodes: NonZeroU32,
    /// This node's rank among `nnodes`. The leader, rank 0, serves the API
    /// and runs the engine; every other node runs only its daemon, which
    /// joins the leader's.
    pub node_rank: u32,
    /// The leader's `host:port`, which the other nodes' daemons join at and
    /// the leader binds; required with several nodes.
    pub dist_init_addr: Option<String>,
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
            worker_groups: None,
            host_identity: "localhost".to_owned(),
            nnodes: NonZeroU32::MIN,
            node_rank: 0,
            dist_init_addr: None,
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
    /// Bearer token every route but `/health` and `/metrics` requires;
    /// `UNISERVE_API_KEY` supplies it when unset or blank. Never printed.
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
    /// Copy full prompt pages to the host page tier of each host's Storage
    /// Server and read them back on a later prefix match the device cache
    /// no longer holds. Applies with `prefix_caching`.
    pub host_tier: bool,
    /// Keep every page of the host page tier in Mooncake, the distributed
    /// tier, through the Storage Servers this deployment starts, and
    /// prefetch long hits back. Applies with `host_tier`; none when unset.
    pub mooncake: Option<MooncakeOptions>,
}

impl Default for KvCacheOptions {
    fn default() -> Self {
        Self {
            max_tokens: None,
            block_size: None,
            dtype: None,
            prefix_caching: true,
            host_tier: true,
            mooncake: None,
        }
    }
}

/// Mooncake, the distributed page tier (`kv_cache_config.mooncake`): the
/// settings of each Storage Server's Mooncake client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MooncakeOptions {
    /// Mooncake's master, `host:port`.
    pub master: String,
    /// The transfer engine's metadata server: `P2PHANDSHAKE`, or an
    /// `http://`, `etcd://` or `redis://` URL.
    #[serde(default = "MooncakeOptions::default_metadata_server")]
    pub metadata_server: String,
    /// The transfer protocol: `tcp` or `rdma`.
    #[serde(default = "MooncakeOptions::default_protocol")]
    pub protocol: String,
    /// The RDMA devices; selected automatically when empty.
    #[serde(default)]
    pub device: String,
    /// Host memory each host contributes to Mooncake's pool, in GiB; zero
    /// makes the hosts clients of segments others provide.
    #[serde(default)]
    pub global_segment_gib: u64,
    /// `libmooncake_store.so`; `MOONCAKE_STORE_LIBRARY` or the loader's
    /// search path when unset.
    #[serde(default)]
    pub library: Option<PathBuf>,
}

impl MooncakeOptions {
    fn default_metadata_server() -> String {
        "P2PHANDSHAKE".into()
    }

    fn default_protocol() -> String {
        "tcp".into()
    }

    /// The connector settings of a Storage Server this deployment starts.
    /// Each host names itself to Mooncake.
    fn connector(&self) -> storage::connector::Settings {
        storage::connector::Settings {
            master: self.master.clone(),
            metadata: self.metadata_server.clone(),
            protocol: self.protocol.clone(),
            device: self.device.clone(),
            hostname: None,
            segment_bytes: self.global_segment_gib << 30,
            library: self.library.clone(),
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
    /// Shard routed experts across the schedulers or the
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

/// The worker processes (`worker_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerOptions {
    /// The `uniserve` executable that runs the workers, which runs
    /// them in the Python environment it is installed in.
    pub executable: PathBuf,
    /// Time a batch may run without rank progress before the worker group is
    /// replaced; a rank reports a stalled lane after half of it.
    #[serde(with = "values::duration")]
    pub watchdog_timeout: Duration,
    /// `tower=gen:<device>` places the generation tower on another device
    /// than the rank's.
    pub mesh: Option<String>,
    /// Torch process-group backend for the ranks' collectives.
    pub distributed_backend: Option<String>,
    /// Keep a standby rank group beside each worker group whose ranks all
    /// run on the head's host and that shares no expert world: it maps the
    /// group's retained weights and cache, captures its graphs and parks,
    /// and replaces the group when it fails instead of a relaunch. Each
    /// standby rank holds a CUDA context, graph pools and workspaces of its
    /// own, which the memory `mem_fraction_static` leaves must hold.
    pub standby: bool,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            executable: WorkerProcessArgs::default_executable(),
            watchdog_timeout: WorkerProcessArgs::DEFAULT_WATCHDOG_TIMEOUT,
            mesh: None,
            distributed_backend: None,
            standby: false,
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
            timeout: Duration::from_secs(20),
            // 20 MB.
            max_bytes: 20_000_000,
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
            max_seconds: models::video::plan::MAX_VIDEO_SECONDS,
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

impl MediaOptions {
    /// The HTTP body limit of the video submission routes, in bytes.
    ///
    /// A `data:` URI carries its media base64 encoded, four bytes per three,
    /// so a body holding `max_request_bytes` of media plus the request's own
    /// fields fits; it never falls below the server-wide body limit.
    pub fn body_limit(&self) -> usize {
        let encoded = self.max_request_bytes.div_ceil(3) * 4 + (1 << 20);
        usize::try_from(encoded)
            .unwrap_or(usize::MAX)
            .max(crate::http::BODY_LIMIT)
    }
}

impl Default for MediaOptions {
    fn default() -> Self {
        Self {
            directory: None,
            allow_remote: true,
            // 256 MiB.
            max_request_bytes: 256 << 20,
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
            retention: Duration::from_secs(3600),
            // 1 GiB.
            max_bytes: 1 << 30,
        }
    }
}

/// DiffusionGemma System One readouts (`readout_config`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReadoutOptions {
    /// `joint` packs questions into shared canvases; `independent` gives
    /// each its own prompt and canvas.
    pub layout: ReadoutLayout,
    /// Canvas length: `full`, `compact`, or a positive multiple of 16.
    pub canvas: CanvasMode,
    /// `variants` sums supported answer spellings; `primary` reads the
    /// space-prefixed one.
    pub candidates: CandidateTokens,
}

impl ServeConfig {
    /// The configured model: a local directory or a Hugging Face repository.
    ///
    /// Empty while unstated; [`ServeConfig::validate`] refuses that.
    pub fn model(&self) -> &str {
        self.model.as_deref().unwrap_or_default()
    }

    /// The single model name the API serves: `served_model_name`, else the
    /// model.
    pub fn served_model_name(&self) -> &str {
        self.served_model_name
            .as_deref()
            .unwrap_or_else(|| self.model())
    }

    /// Checks the rules the option types cannot express on their own.
    ///
    /// `stated` records the paths the operator set, so the placement
    /// shorthand is refused beside a deployment file. Rules that need the
    /// served model's family run at startup instead
    /// ([`Stated::refuse_foreign_options`]).
    ///
    /// # Errors
    ///
    /// Fails when `model` is unset, `mem_fraction_static` lies outside
    /// (0, 1], a placement-shorthand option is stated with `worker_groups`, the
    /// server host or Unix socket path is blank, or
    /// `video_config.max_seconds` lies outside the durations the video API
    /// admits.
    pub fn validate(&self, stated: &Stated) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.node_rank < self.nnodes.get(),
            "`node_rank` {} lies outside the {} nodes `nnodes` states",
            self.node_rank,
            self.nnodes
        );
        anyhow::ensure!(
            self.nnodes.get() == 1 || self.dist_init_addr.is_some(),
            "a deployment of several nodes needs the leader's address: set `dist_init_addr`"
        );
        anyhow::ensure!(
            !self.leads() || self.model.is_some(),
            "`model` is required: name MODEL on the command line or set `model` in --config"
        );
        WorkerGroupConfig::validate_mem_fraction_static(self.mem_fraction_static)
            .context("invalid `mem_fraction_static`")?;
        if self.worker_groups.is_some()
            && let Some(option) = PLACEMENT_SHORTHAND
                .iter()
                .find(|option| stated.contains(option))
        {
            anyhow::bail!(
                "`{option}` places ranks without a deployment file; the `worker_groups` file \
                 states its own placement"
            );
        }
        let server = &self.server_config;
        match &server.uds {
            Some(path) => anyhow::ensure!(
                !path.trim().is_empty(),
                "`server_config.uds` must not be empty"
            ),
            None => anyhow::ensure!(
                !server.host.trim().is_empty(),
                "`server_config.host` must not be empty"
            ),
        }
        // A smaller capacity admits no request and a larger one exceeds what
        // the video API accepts.
        let seconds = MIN_VIDEO_SECONDS..=MAX_VIDEO_SECONDS;
        anyhow::ensure!(
            seconds.contains(&self.video_config.max_seconds),
            "`video_config.max_seconds` must lie in [{}, {}], got {}",
            seconds.start(),
            seconds.end(),
            self.video_config.max_seconds
        );
        Ok(())
    }

    /// Whether this node is the leader, which serves the API and runs the
    /// engine.
    pub fn leads(&self) -> bool {
        self.node_rank == 0
    }

    /// The options of this node's daemon. The leader's process socket is
    /// this process's own, in the user's runtime directory; `executable`
    /// starts the node's Storage Server when it has none.
    pub fn daemon_options(&self) -> DaemonOptions {
        let directory = storage::socket::default_socket()
            .parent()
            .map_or_else(std::env::temp_dir, Path::to_path_buf);
        DaemonOptions {
            nnodes: self.nnodes.get(),
            node_rank: self.node_rank,
            host: self.host_identity.clone(),
            dist_init_addr: self.dist_init_addr.clone(),
            socket: directory.join(format!("process-{}.sock", std::process::id())),
            executable: self.worker_config.executable.clone(),
            join_timeout: daemon::JOIN_TIMEOUT,
        }
    }

    /// Derives the engine configuration these options state.
    ///
    /// The deployment comes from the `worker_groups` file, placed on this
    /// process's `host_identity`, or else from `tensor_parallel_size`,
    /// `data_parallel_size` and `placement_config`. What the loaded model
    /// determines keeps its neutral value for `Engine::start` to complete:
    /// the generation limits and control tokens, and the worker launch
    /// fields `max_model_len`, `canvas_sampling` and `prefill_outputs`.
    ///
    /// # Errors
    ///
    /// Fails when the deployment does not place, a capture list is not
    /// strictly increasing or repeats a shape, a lane or the transfer edges
    /// do not resolve, or a batch bound exceeds the worker's `u32` fields.
    pub fn engine_config(&self) -> anyhow::Result<EngineConfig> {
        let worker_groups = match &self.worker_groups {
            Some(path) => Deployment::read(&path.to_string_lossy())?
                .worker_groups(&self.host_identity)
                .context("invalid `worker_groups` deployment")?,
            None => {
                let tensor_parallel_size = self.tensor_parallel_size.get();
                WorkerGroupConfig::data_parallel(
                    &self.rank_hosts(),
                    &self.placement_config.device,
                    self.data_parallel_size.get(),
                    tensor_parallel_size,
                    self.placement_config.queue_depth.get(),
                    WorkerGroupConfig::single_component(DEFAULT_COMPONENT, tensor_parallel_size),
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

        let max_num_batched_tokens = self
            .max_num_batched_tokens
            .map_or(DEFAULT_MAX_NUM_BATCHED_TOKENS, NonZeroUsize::get);
        let graphs = &self.cuda_graph_config;
        let flashinfer = &self.flashinfer_config;
        let worker_process = WorkerProcessArgs {
            executable: self.worker_config.executable.clone(),
            // A deployment is what one head serves under its served name.
            deployment: self.served_model_name().to_owned(),
            page_tier: self.kv_cache_config.prefix_caching && self.kv_cache_config.host_tier,
            storage_connector: self
                .kv_cache_config
                .mooncake
                .as_ref()
                .filter(|_| self.kv_cache_config.prefix_caching && self.kv_cache_config.host_tier)
                .map(MooncakeOptions::connector),
            keep_standby: self.worker_config.standby,
            standby: false,
            cache_units: None,
            model: self.model().to_owned(),
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
            // Each forward batch is bounded by the scheduler's batch and token
            // budgets, which the worker's launch descriptor carries as `u32`.
            max_batch_calls: u32::try_from(self.scheduler_config.max_batch.get())
                .context("`scheduler_config.max_batch` exceeds the worker field width")?,
            max_batch_tokens: u32::try_from(max_num_batched_tokens)
                .context("`max_num_batched_tokens` exceeds the worker field width")?,
            // Video workers size resident media state for the longest
            // duration served and provision only the frame counts the API
            // admits, and prepare exactly the canvases these resolutions and
            // aspect ratios name.
            max_video_seconds: self.video_config.max_seconds,
            min_video_seconds: Some(MIN_VIDEO_SECONDS),
            max_condition_rows: self.video_config.max_condition_rows,
            video_resolutions: self.video_config.resolutions.clone(),
            video_aspect_ratios: self.video_config.aspect_ratios.clone(),
            ..WorkerProcessArgs::default()
        };

        Ok(EngineConfig {
            generation_limits: GenerationLimits::default(),
            max_batch: self.scheduler_config.max_batch.get(),
            max_num_batched_tokens,
            max_running_requests: self.max_running_requests.get(),
            max_queued_requests: self
                .scheduler_config
                .max_queued_requests
                .map(NonZeroUsize::get),
            chunked_prefill_size: self
                .scheduler_config
                .chunked_prefill_size
                .map_or(DEFAULT_CHUNKED_PREFILL_SIZE, NonZeroUsize::get),
            prefix_cache: self.kv_cache_config.prefix_caching,
            schedule_policy: self.scheduler_config.policy,
            worker_groups,
            transfer,
            data_parallel_size: self.data_parallel_size.get(),
            expert_parallel: self
                .moe_config
                .expert_parallel
                .then_some(self.moe_config.exchange),
            worker_process,
            bos: 0,
            eos: Vec::new(),
            end_of_image: 0,
        })
    }

    /// Returns the hosts the placement shorthand places ranks on, this host
    /// first.
    ///
    /// `WorkerGroupConfig::placed` assigns ranks to these hosts in contiguous
    /// blocks in list order, so leading with this head's own identity
    /// keeps rank zero and the other lowest ranks on the head's host.
    /// Entries of `worker_hosts` that are empty or repeat this head's
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
/// [`ServeArgs::resolve`] stores it in `server_config.api_key`.
///
/// The configured key takes precedence over the `UNISERVE_API_KEY`
/// environment variable. Each source is trimmed, and a blank value counts as
/// unset, so a blank configured key falls through to the environment
/// variable.
pub(super) fn configured_api_key(configured: Option<&str>) -> Option<String> {
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
