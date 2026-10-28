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
//! own table, and [`ServeConfig::engine_spec`] states what of it concerns
//! the engine process.
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

use crate::supervision::DaemonOptions;
use anyhow::Context as _;
use models::video::plan::{MAX_VIDEO_SECONDS, MIN_VIDEO_SECONDS};
use proto::request::wire;
use proto::request::{
    ExpertExchange, FlashInferBackend, GraphPolicy, LoadFormat, ModelDtype, PrefixCacheDtype,
    SchedulingPolicy, StandbyMemory,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use models::ModelDescription;
use models::diffusion_gemma::DenoisingOverrides;

pub use args::{FLAG_PATHS, ServeArgs, Stated, overridden, printed, resolve_text};
pub use values::{
    CandidateTokens, CanvasMode, ChatTemplateContentFormatOption, ImageSize, ReadoutLayout,
};

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
            "speculative_config",
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
    pub server_config: ServerOptions,
    pub load_config: LoadOptions,
    pub kv_cache_config: KvCacheOptions,
    pub scheduler_config: SchedulerOptions,
    pub speculative_config: SpeculativeOptions,
    pub migration_config: MigrationOptions,
    pub update_config: UpdateOptions,
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
            dtype: ModelDtype::BFloat16,
            quantization_config: serde_json::Map::new(),
            mm_processor_kwargs: serde_json::Map::new(),
            attention_backend: "auto".to_owned(),
            max_running_requests: positive(128),
            max_num_batched_tokens: None,
            mem_fraction_static: 0.7,
            tensor_parallel_size: NonZeroUsize::MIN,
            data_parallel_size: NonZeroUsize::MIN,
            worker_groups: None,
            host_identity: "localhost".to_owned(),
            nnodes: NonZeroU32::MIN,
            node_rank: 0,
            dist_init_addr: None,
            server_config: ServerOptions::default(),
            load_config: LoadOptions::default(),
            kv_cache_config: KvCacheOptions::default(),
            scheduler_config: SchedulerOptions::default(),
            speculative_config: SpeculativeOptions::default(),
            migration_config: MigrationOptions::default(),
            update_config: UpdateOptions::default(),
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoadOptions {
    /// Checkpoint reader: `auto`, `safetensors`, `pt`, `layered` (one layer
    /// at a time) or `dummy` (no weights read).
    pub format: LoadFormat,
    /// Hugging Face cache root for repository models.
    pub download_dir: Option<PathBuf>,
    /// Concurrent checkpoint file readers.
    pub threads: Option<NonZeroU32>,
    /// A plugin's weight source (`scheme://...`) that streams the
    /// checkpoint's tensors in place of its files; the files when unset.
    pub weights: Option<String>,
    /// The tensor- and expert-parallel degrees the configurations of the
    /// deployment use, which retained weight memory is chunked for so that
    /// a world of any of them maps the chunks it shares with another;
    /// every degree up to the GPUs a worker sees when empty.
    pub parallel_degrees: Vec<NonZeroU32>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            format: LoadFormat::Auto,
            download_dir: None,
            threads: None,
            weights: None,
            parallel_degrees: Vec::new(),
        }
    }
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
    /// Element type (`float16`, `bfloat16`, `float32` or `float8_e4m3fn`);
    /// each layer's compute type when unset.
    pub dtype: Option<PrefixCacheDtype>,
    /// Reuse and retain cached prompt prefixes across requests.
    pub prefix_caching: bool,
    /// Offload full prompt pages to the host KV cache of each host's Storage
    /// Server and load them back on a later prefix match the device cache
    /// no longer holds: every published page of a full-attention group, and
    /// a sliding-window group's window before each aligned boundary.
    /// Applies with `prefix_caching`.
    pub host_tier: bool,
    /// The host KV cache's token count over the device KV cache's, the
    /// same on every rank of a world (as SGLang's `--hicache-ratio`); each
    /// host's Storage Server lowers it for the whole world to what four
    /// fifths of its available memory hold. The engine's default (2.0)
    /// when unset.
    pub host_ratio: Option<f64>,
    /// The deployment's host KV cache on each host, in GiB, over the ranks
    /// there, in place of `host_ratio` (as SGLang's `--hicache-size`).
    pub host_gib: Option<NonZeroU64>,
    /// Keep every page of the host KV cache in Mooncake, the KV storage,
    /// through the Storage Servers this deployment starts, and prefetch long
    /// hits back. Applies with `host_tier`; none when unset.
    pub mooncake: Option<MooncakeOptions>,
    /// The pages of the largest block size between the aligned boundaries
    /// a long prompt offloads its sliding windows before; the engine's
    /// default when unset.
    pub boundary_interval: Option<NonZeroU32>,
    /// The decoded tokens between two checkpoints of the state a model
    /// keeps per request slot (a recurrent state); the engine's default
    /// when unset.
    pub state_interval: Option<NonZeroU32>,
}

impl Default for KvCacheOptions {
    fn default() -> Self {
        Self {
            max_tokens: None,
            block_size: None,
            dtype: None,
            prefix_caching: true,
            host_tier: true,
            host_ratio: None,
            host_gib: None,
            mooncake: None,
            boundary_interval: None,
            state_interval: None,
        }
    }
}

/// Mooncake, the KV storage (`kv_cache_config.mooncake`): the
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
            max_batch: positive(128),
        }
    }
}

/// Speculative decoding with the model's drafter (`speculative_config`).
///
/// A deployment that speculates verifies each running request's draft
/// tokens, which the model's drafter proposes from its draft cache, in one
/// forward. A request whose draft cache is missing, such as one a world
/// change moved, verifies none while its draft cache is refilled with the
/// idle prefill budget of the steps, when the refill policy finds it worth
/// its cost.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculativeOptions {
    /// The most draft tokens one verification carries per request; zero
    /// does not speculate. The draft cache and the drafter's work belong
    /// to the world, so a change of it is a world change.
    pub num_draft_tokens: u32,
    /// The tokens a verification step of a request is expected to emit:
    /// the speedup the refill policy expects over decoding one token a
    /// step; at least one, 2.0 when unset.
    pub expected_speedup: Option<f64>,
    /// The time of evaluating one context token in a refill, in steps of
    /// decoding: the refill policy's cost of each token it evaluates; not
    /// negative, 0.01 when unset.
    pub refill_token_cost: Option<f64>,
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
            queue_depth: positive(2),
            transfer: None,
        }
    }
}

/// Mixture-of-experts parallelism (`moe_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MoeOptions {
    /// Shard routed experts across the schedulers, the tensor-parallel
    /// ranks of a single scheduler's one worker group, or the deployment's
    /// expert workers.
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
            exchange: ExpertExchange::AllToAll,
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
    /// Decode batch sizes captured, strictly increasing; the worker's
    /// catalog when unset.
    #[serde(deserialize_with = "values::optional_list::deserialize")]
    pub decode_batch_sizes: Option<Vec<NonZeroU32>>,
    /// Prefill token counts captured, strictly increasing; the worker's
    /// catalog when unset.
    #[serde(deserialize_with = "values::optional_list::deserialize")]
    pub prefill_token_sizes: Option<Vec<NonZeroU32>>,
    /// Distinct `HEIGHTxWIDTH` image sizes captured; `1152x2048` and
    /// `2048x1152` when unset.
    #[serde(deserialize_with = "values::optional_list::deserialize")]
    pub flow_shapes: Option<Vec<ImageSize>>,
}

impl Default for CudaGraphOptions {
    fn default() -> Self {
        Self {
            policy: GraphPolicy::Auto,
            decode_batch_sizes: None,
            prefill_token_sizes: None,
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
            // 512 MiB.
            workspace_size: 512 << 20,
            use_tensor_core: None,
            decode_backend: FlashInferBackend::Fa2,
            prefill_backend: FlashInferBackend::Auto,
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
    /// replaced; a rank reports a stalled job after half of it.
    #[serde(with = "values::duration")]
    pub watchdog_timeout: Duration,
    /// `tower=gen:<device>` places the generation tower on another device
    /// than the rank's.
    pub mesh: Option<String>,
    /// Torch process-group backend for the ranks' collectives.
    pub distributed_backend: Option<String>,
    /// Keep a standby beside every worker group: it maps the group's
    /// retained weights and cache, captures its graphs and parks, and
    /// replaces the group when it fails instead of a relaunch. A group whose
    /// ranks all run on the engine's host and that shares no expert world
    /// keeps a standby rank group of its own; the groups of an expert world,
    /// or a group on several nodes, keep a standby world: a second launch of
    /// every group of the failure domain, with communicators of its own,
    /// promoted together when one of them fails. Each standby rank holds a
    /// CUDA context, graph pools and workspaces of its own, which the memory
    /// `mem_fraction_static` leaves must hold.
    pub standby: bool,
    /// How a parked standby (a group's, a failure domain's world, or a hot
    /// update's engine) keeps the device memory of its graph pools, scratch,
    /// expert all-to-all workspaces and NCCL communicators: `auto` keeps it
    /// resident while every device has the share a serving worker leaves
    /// for graphs and activations to spare, and suspends it otherwise, every
    /// rank of a world or group alike; `resident`; or `suspended`, released
    /// on the device and restored at the promotion, the graph memory with
    /// its contents at unchanged addresses; workspaces that cannot keep
    /// their addresses (DeepEP's windows, colocated MegaMoE's NVSHMEM heap)
    /// are created again at the promotion, which captures the graphs that
    /// use them anew.
    pub standby_memory: StandbyMemory,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            executable: default_executable(),
            watchdog_timeout: Duration::from_secs(300),
            mesh: None,
            distributed_backend: None,
            standby: false,
            standby_memory: StandbyMemory::Auto,
        }
    }
}

/// The `uniserve` executable of the Python environment this process runs
/// in, which runs the engine process and the workers.
///
/// The executable runs workers in the environment it is installed in, so a
/// daemon built outside one, such as a development build, takes the active
/// environment's; without one, `uniserve` is looked up on the search path.
fn default_executable() -> PathBuf {
    const EXECUTABLE: &str = "uniserve";
    let environment_bin = |directory: &Path| {
        ["python", "python3"]
            .iter()
            .any(|name| directory.join(name).is_file())
            .then(|| directory.join(EXECUTABLE))
            .filter(|candidate| candidate.is_file())
    };
    if let Ok(executable) = std::env::current_exe()
        && let Some(candidate) = executable.parent().and_then(environment_bin)
    {
        return candidate;
    }
    if let Some(environment) = std::env::var_os("VIRTUAL_ENV")
        && let Some(candidate) = environment_bin(&Path::new(&environment).join("bin"))
    {
        return candidate;
    }
    EXECUTABLE.into()
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

impl Default for MediaOptions {
    fn default() -> Self {
        Self {
            directory: None,
            allow_remote: true,
            // 256 MiB.
            max_request_bytes: 256 << 20,
            ffprobe: PathBuf::from("ffprobe"),
            ffmpeg: PathBuf::from("ffmpeg"),
        }
    }
}

/// Moving token requests between engines (`migration_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MigrationOptions {
    /// Migrations caused by lost workers, a lost engine process or a failed
    /// world on trial that a request survives; it fails on the next.
    /// Migrations of a world change or a rollback are not counted.
    pub max_migrations: u32,
}

impl Default for MigrationOptions {
    fn default() -> Self {
        Self { max_migrations: 3 }
    }
}

/// `current` with `target`'s daemon options (those `daemon_options_differ`
/// compares): the configuration a hot restart of the daemon serves when an
/// update changes both the daemon and what a world executes, before the
/// daemon's successor changes the world.
pub fn with_daemon_options(current: &ServeConfig, target: &ServeConfig) -> ServeConfig {
    ServeConfig {
        server_config: target.server_config.clone(),
        logging_config: target.logging_config.clone(),
        chat_config: target.chat_config.clone(),
        image_fetch_config: target.image_fetch_config.clone(),
        media_config: target.media_config.clone(),
        video_job_config: target.video_job_config.clone(),
        readout_config: target.readout_config,
        migration_config: target.migration_config.clone(),
        mm_processor_kwargs: target.mm_processor_kwargs.clone(),
        diffusion_generation_config: target.diffusion_generation_config,
        ..current.clone()
    }
}

/// Whether `current` and `target` differ in the options the daemon reads
/// alone (its HTTP server, the request and output processing, the media
/// and video jobs, logging and migration), which a hot restart of the
/// daemon applies; the engine spec covers every other option.
pub fn daemon_options_differ(current: &ServeConfig, target: &ServeConfig) -> bool {
    current.server_config != target.server_config
        || current.logging_config != target.logging_config
        || current.chat_config != target.chat_config
        || current.image_fetch_config != target.image_fetch_config
        || current.media_config != target.media_config
        || current.video_job_config != target.video_job_config
        || current.readout_config != target.readout_config
        || current.migration_config != target.migration_config
        || current.mm_processor_kwargs != target.mm_processor_kwargs
        || current.diffusion_generation_config != target.diffusion_generation_config
}

/// How the deployment moves to a new configuration in another world
/// (`update_config`, `PUT /admin/config`): what of the previous world its
/// trial keeps for a rollback, when the trial rolls back by itself, and how
/// long the serving world's requests hold the cache a successor needs. A
/// value an update states in its own assignments governs that update alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UpdateOptions {
    /// What the previous world keeps while the new one is on trial.
    pub previous_world: PreviousWorld,
    /// When a trial rolls back without the operator: a fatal worker error,
    /// the engine process's exit, a stalled job and non-finite logits always
    /// roll back; these thresholds, which have no default, add the rest.
    pub rollback: RollbackOptions,
    /// How long the serving world's requests that hold the KV cache's tail a
    /// successor needs run on before they are recomputed; unset to let them
    /// run to their end.
    #[serde(with = "values::optional_duration")]
    pub preempt_after: Option<Duration>,
    /// How much longer than before the change a serving scheduler's step
    /// may take while the successor prepares, as a fraction (`0.1` for
    /// 10%), past which the leader halves the byte budgets of the links the
    /// change's transfers use, and restores them once the steps are back
    /// within it. Steps are compared within their batch-size range. The
    /// default is the value measured on GB200 (`docs/configuration.md`).
    pub transfer_slowdown: f64,
    /// Whether the serving world runs its forwards eagerly while the
    /// successor prepares, giving its CUDA graph memory to the successor's
    /// captures: the serving world steps slower meanwhile, and replays its
    /// graphs again if it serves on or resumes.
    pub eager_prepare: bool,
}

/// The default `update_config.transfer_slowdown`: on GB200, above the
/// serving steps' second-to-second variation without transfers and below
/// the slowdowns copies at a link's whole budget cause
/// (`docs/configuration.md`).
pub const TRANSFER_SLOWDOWN: f64 = 0.1;

impl Default for UpdateOptions {
    fn default() -> Self {
        Self {
            previous_world: PreviousWorld::Parked,
            rollback: RollbackOptions::default(),
            preempt_after: None,
            transfer_slowdown: TRANSFER_SLOWDOWN,
            eager_prepare: false,
        }
    }
}

/// What the previous world keeps during its successor's trial
/// (`update_config.previous_world`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreviousWorld {
    /// Its processes, communicators, graph memory, weights and the KV cache
    /// its successor maps: a rollback resumes it at once.
    Parked,
    /// Its weights (and the KV cache its successor maps): a rollback starts
    /// its processes again, which bind them.
    Weights,
    /// The KV cache its successor maps, and in host memory the weights its
    /// successor does not map: a rollback copies them back and starts its
    /// processes again.
    Host,
}

/// The thresholds past which a trial rolls back by itself
/// (`update_config.rollback`). None has a default: each is measured on the
/// target workload before it is set.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RollbackOptions {
    /// The share of the requests that end during the trial, in `(0, 1]`,
    /// past which it rolls back when they failed.
    pub failure_rate: Option<f64>,
    /// How much longer than the previous world's before its promotion the
    /// new world's average step may take, as a fraction (`0.2` for 20%).
    pub step_time_regression: Option<f64>,
    /// How much lower than the previous world's before its promotion the
    /// new world's token throughput may fall, as a fraction.
    pub throughput_regression: Option<f64>,
    /// The span the step time and throughput are averaged over, before the
    /// promotion and during the trial; required with either regression.
    #[serde(with = "values::optional_duration")]
    pub window: Option<Duration>,
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
            !self.leads() || self.model.is_some() || self.nnodes.get() == 1,
            "`model` is required on the leader of several nodes: name MODEL on the command line \
             or set `model` in --config"
        );
        let speculation = &self.speculative_config;
        if let Some(speedup) = speculation.expected_speedup {
            anyhow::ensure!(
                speedup.is_finite() && speedup >= 1.0,
                "invalid `speculative_config.expected_speedup`: {speedup} must be at least 1"
            );
        }
        if let Some(cost) = speculation.refill_token_cost {
            anyhow::ensure!(
                cost.is_finite() && cost >= 0.0,
                "invalid `speculative_config.refill_token_cost`: {cost} must not be negative"
            );
        }
        let slowdown = self.update_config.transfer_slowdown;
        anyhow::ensure!(
            slowdown.is_finite() && slowdown > 0.0,
            "invalid `update_config.transfer_slowdown`: {slowdown} must be positive"
        );
        let rollback = &self.update_config.rollback;
        if let Some(rate) = rollback.failure_rate {
            anyhow::ensure!(
                rate.is_finite() && rate > 0.0 && rate <= 1.0,
                "invalid `update_config.rollback.failure_rate`: {rate} must be in (0, 1]"
            );
        }
        for (name, ratio) in [
            ("step_time_regression", rollback.step_time_regression),
            ("throughput_regression", rollback.throughput_regression),
        ] {
            if let Some(ratio) = ratio {
                anyhow::ensure!(
                    ratio.is_finite() && ratio > 0.0,
                    "invalid `update_config.rollback.{name}`: {ratio} must be positive"
                );
                anyhow::ensure!(
                    rollback.window.is_some_and(|window| !window.is_zero()),
                    "`update_config.rollback.{name}` is measured over \
                     `update_config.rollback.window`: set it"
                );
            }
        }
        // Each rank grants itself this share of every device's total
        // storage, so any finite value in (0, 1] is meaningful.
        anyhow::ensure!(
            self.mem_fraction_static.is_finite()
                && self.mem_fraction_static > 0.0
                && self.mem_fraction_static <= 1.0,
            "invalid `mem_fraction_static`: {} must be finite and in (0, 1]",
            self.mem_fraction_static
        );
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

    /// The leader's request socket, which its engine processes connect back
    /// to: this process's own, in the user's runtime directory.
    pub fn request_socket(&self) -> PathBuf {
        runtime_directory().join(format!("requests-{}.sock", std::process::id()))
    }

    /// The options of this node's daemon. The leader's process socket is
    /// this process's own, in the user's runtime directory; `executable`
    /// starts the node's Storage Server when it has none.
    pub fn daemon_options(&self) -> DaemonOptions {
        let directory = runtime_directory();
        DaemonOptions {
            nnodes: self.nnodes.get(),
            node_rank: self.node_rank,
            host: self.host_identity.clone(),
            dist_init_addr: self.dist_init_addr.clone(),
            socket: directory.join(format!("process-{}.sock", std::process::id())),
            executable: self.worker_config.executable.clone(),
            join_timeout: crate::supervision::JOIN_TIMEOUT,
        }
    }

    /// The engine spec these options state, without the model's facts,
    /// which the leader resolves from the checkpoint.
    ///
    /// Values the options name by a word (a policy, a backend, an element
    /// type) travel as that word, which the engine reads; the placement
    /// travels as stated, which the engine places.
    ///
    /// # Errors
    ///
    /// Fails, naming the option, when a capture list is not strictly
    /// increasing or repeats a shape, or a batch bound exceeds the worker's
    /// `u32` fields.
    pub fn engine_spec(&self) -> anyhow::Result<wire::EngineSpec> {
        let graphs = &self.cuda_graph_config;
        let flashinfer = &self.flashinfer_config;
        let kv_cache = &self.kv_cache_config;
        let worker = &self.worker_config;
        let text = |path: &Option<PathBuf>| {
            path.as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let bounded = |value: usize, option: &str| {
            u32::try_from(value)
                .with_context(|| format!("`{option}` exceeds the worker field width"))
        };
        let scheduling = wire::Scheduling {
            policy: word(&self.scheduler_config.policy),
            max_batch: bounded(
                self.scheduler_config.max_batch.get(),
                "scheduler_config.max_batch",
            )?,
            max_num_batched_tokens: self
                .max_num_batched_tokens
                .map(|tokens| bounded(tokens.get(), "max_num_batched_tokens"))
                .transpose()?,
            max_running_requests: bounded(self.max_running_requests.get(), "max_running_requests")?,
            max_queued_requests: self
                .scheduler_config
                .max_queued_requests
                .map(|requests| requests.get() as u64),
            chunked_prefill_size: self
                .scheduler_config
                .chunked_prefill_size
                .map(|size| bounded(size.get(), "scheduler_config.chunked_prefill_size"))
                .transpose()?,
            expected_speedup: self.speculative_config.expected_speedup,
            refill_token_cost: self.speculative_config.refill_token_cost,
        };
        Ok(wire::EngineSpec {
            // A deployment is what one leader serves under its served name.
            deployment: self.served_model_name().to_owned(),
            model: self.model().to_owned(),
            host: self.host_identity.clone(),
            placement: Some(wire::Placement {
                worker_groups: text(&self.worker_groups),
                tensor_parallel_size: bounded(
                    self.tensor_parallel_size.get(),
                    "tensor_parallel_size",
                )?,
                data_parallel_size: bounded(self.data_parallel_size.get(), "data_parallel_size")?,
                device: self.placement_config.device.clone(),
                queue_depth: bounded(
                    self.placement_config.queue_depth.get(),
                    "placement_config.queue_depth",
                )?,
                worker_hosts: self.placement_config.worker_hosts.clone(),
                mem_fraction_static: self.mem_fraction_static,
            }),
            transfer: self.placement_config.transfer.clone().unwrap_or_default(),
            scheduling: Some(scheduling),
            kv_cache: Some(wire::KvCache {
                max_tokens: kv_cache.max_tokens.map(NonZeroU64::get),
                block_size: kv_cache.block_size.map(NonZeroU32::get),
                dtype: kv_cache
                    .dtype
                    .map(PrefixCacheDtype::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                prefix_caching: kv_cache.prefix_caching,
                host_tier: kv_cache.host_tier,
                mooncake: kv_cache.mooncake.as_ref().map(|mooncake| wire::Mooncake {
                    master: mooncake.master.clone(),
                    metadata_server: mooncake.metadata_server.clone(),
                    protocol: mooncake.protocol.clone(),
                    device: mooncake.device.clone(),
                    segment_bytes: mooncake.global_segment_gib << 30,
                    library: text(&mooncake.library),
                }),
                boundary_interval: kv_cache.boundary_interval.map(NonZeroU32::get),
                state_interval: kv_cache.state_interval.map(NonZeroU32::get),
                host_ratio: kv_cache.host_ratio,
                host_bytes: kv_cache.host_gib.map(|gib| gib.get() << 30),
            }),
            workers: Some(wire::Workers {
                executable: worker.executable.to_string_lossy().into_owned(),
                watchdog_timeout_ms: u64::try_from(worker.watchdog_timeout.as_millis())
                    .unwrap_or(u64::MAX),
                mesh: worker.mesh.clone().unwrap_or_default(),
                distributed_backend: worker.distributed_backend.clone().unwrap_or_default(),
                standby: worker.standby,
                attention_backend: self.attention_backend.clone(),
                dtype: word(&self.dtype),
                quantization_config: Value::Object(self.quantization_config.clone()).to_string(),
                load_format: word(&self.load_config.format),
                download_dir: text(&self.load_config.download_dir),
                load_threads: self.load_config.threads.map(NonZeroU32::get),
                standby_memory: word(&worker.standby_memory),
                weights: self.load_config.weights.clone().unwrap_or_default(),
                parallel_degrees: self
                    .load_config
                    .parallel_degrees
                    .iter()
                    .map(|degree| degree.get())
                    .collect(),
            }),
            cuda_graphs: Some(wire::CudaGraphs {
                policy: word(&graphs.policy),
                decode_batch_sizes: increasing(
                    "cuda_graph_config.decode_batch_sizes",
                    &graphs.decode_batch_sizes,
                )?,
                prefill_token_sizes: increasing(
                    "cuda_graph_config.prefill_token_sizes",
                    &graphs.prefill_token_sizes,
                )?,
                flow_shapes: distinct("cuda_graph_config.flow_shapes", &graphs.flow_shapes)?,
            }),
            flashinfer: Some(wire::FlashInfer {
                workspace_size: flashinfer.workspace_size,
                use_tensor_core: flashinfer.use_tensor_core,
                decode_backend: flashinfer.decode_backend.as_str().to_owned(),
                prefill_backend: flashinfer.prefill_backend.as_str().to_owned(),
                decode_split_tile_size: flashinfer.decode_split_tile_size.map(NonZeroU32::get),
                prefill_split_tile_size: flashinfer.prefill_split_tile_size.map(NonZeroU32::get),
                split_kv: flashinfer.split_kv,
            }),
            // Video workers size resident media state for the longest
            // duration served and provision only the frame counts the API
            // admits, and prepare exactly the canvases these resolutions and
            // aspect ratios name.
            video: Some(wire::Video {
                max_seconds: self.video_config.max_seconds,
                min_seconds: MIN_VIDEO_SECONDS,
                resolutions: self.video_config.resolutions.clone(),
                aspect_ratios: self.video_config.aspect_ratios.clone(),
                text_capacities: increasing(
                    "video_config.text_capacities",
                    &self.video_config.text_capacities,
                )?,
                max_condition_rows: self.video_config.max_condition_rows,
                ffmpeg: self.media_config.ffmpeg.to_string_lossy().into_owned(),
            }),
            experts: Some(wire::Experts {
                expert_parallel: self.moe_config.expert_parallel,
                exchange: word(&self.moe_config.exchange),
                microbatches: self.moe_config.microbatches.get(),
            }),
            model_facts: None,
            world: 0,
            predecessor: None,
            segments: Vec::new(),
            retire: Vec::new(),
            speculation: (self.speculative_config.num_draft_tokens > 0).then_some(
                wire::Speculation {
                    num_draft_tokens: self.speculative_config.num_draft_tokens,
                },
            ),
        })
    }
}

/// The user's runtime directory, where this node's sockets lie beside its
/// Storage Server's.
fn runtime_directory() -> PathBuf {
    storage::socket::default_socket()
        .parent()
        .map_or_else(std::env::temp_dir, Path::to_path_buf)
}

/// The word a serving option names `value` by, as the option tree prints
/// it.
fn word<T: Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(Value::String(word)) => word,
        _ => String::new(),
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
pub fn configured_api_key(configured: Option<&str>) -> Option<String> {
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
fn increasing(path: &str, values: &Option<Vec<NonZeroU32>>) -> anyhow::Result<Option<wire::Sizes>> {
    let Some(values) = values else {
        return Ok(None);
    };
    anyhow::ensure!(
        values.windows(2).all(|pair| pair[0] < pair[1]),
        "`{path}` must be strictly increasing, got {values:?}"
    );
    Ok(Some(wire::Sizes {
        values: values.iter().map(|value| value.get()).collect(),
    }))
}

/// Checks that stated image capture sizes are distinct.
fn distinct(path: &str, shapes: &Option<Vec<ImageSize>>) -> anyhow::Result<Option<wire::Shapes>> {
    let Some(shapes) = shapes else {
        return Ok(None);
    };
    anyhow::ensure!(
        shapes.iter().collect::<BTreeSet<_>>().len() == shapes.len(),
        "`{path}` repeats an image size"
    );
    Ok(Some(wire::Shapes {
        shapes: shapes
            .iter()
            .map(|shape| wire::Shape {
                height: shape.0[0],
                width: shape.0[1],
            })
            .collect(),
    }))
}
