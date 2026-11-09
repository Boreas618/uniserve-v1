//! The configuration sections every process of a deployment reads, one type each.
//!
//! Each type is the section of the deployment file named after it (`kv_cache_config` is
//! [`KvCacheConfig`]): the daemon parses the file into it, and the same type travels to the
//! frontend and the engine processes. Every type refuses a field it does not know, so a key no
//! section knows fails the parse instead of being ignored.

pub mod block;
pub mod values;
pub mod worker_group;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::request::wire::{ExpertOrder, Predecessor};
use crate::request::{
    ExpertExchange, FlashInferBackend, GraphPolicy, LoadFormat, PrefixCacheDtype, SchedulingPolicy,
    StandbyStorage,
};
pub use block::{
    BlockConfig, BlockConfigError, BlockDistribution, BlockWeights, HeadProjection,
    SequenceParallel,
};
pub use values::ImageSize;
pub use worker_group::{
    DEFAULT_BLOCK, VIDEO_CODEC_BLOCK, VIDEO_DECODER_BLOCK, WorkerGroupConfig, WorkerGroupId,
    WorkerGroupRole, WorkerRank,
};

/// Shortest admitted clip, in seconds.
pub const MIN_VIDEO_SECONDS: f64 = 4.0;
/// Longest admitted clip, in seconds.
pub const MAX_VIDEO_SECONDS: f64 = 15.0;

/// A positive count, for the defaults of non-zero options.
pub const fn positive(count: usize) -> NonZeroUsize {
    match NonZeroUsize::new(count) {
        Some(count) => count,
        None => NonZeroUsize::MIN,
    }
}

/// The HTTP server (`server_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
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

impl Default for ServerConfig {
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
pub struct LoadConfig {
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
    /// deployment use, which retained weight storage is chunked for so that
    /// an epoch of any of them maps the chunks it shares with another;
    /// every degree up to the GPUs a worker sees when empty.
    pub parallel_degrees: Vec<NonZeroU32>,
}

impl Default for LoadConfig {
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
pub struct KvCacheConfig {
    /// Capacity in tokens; sized from storage when unset.
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
    /// same on every rank of an epoch (as SGLang's `--hicache-ratio`); each
    /// host's Storage Server lowers it for the whole epoch to what four
    /// fifths of its available storage hold. The engine's default (2.0)
    /// when unset.
    pub host_ratio: Option<f64>,
    /// The deployment's host KV cache on each host, in GiB, over the ranks
    /// there, in place of `host_ratio` (as SGLang's `--hicache-size`).
    pub host_gib: Option<NonZeroU64>,
    /// Keep every page of the host KV cache in Mooncake, the KV storage,
    /// through the Storage Servers this deployment starts, and prefetch long
    /// hits back. Applies with `host_tier`; none when unset.
    pub mooncake: Option<MooncakeConfig>,
    /// The pages of the largest block size between the aligned boundaries
    /// a long prompt offloads its sliding windows before; the engine's
    /// default when unset.
    pub boundary_interval: Option<NonZeroU32>,
    /// The decoded tokens between two checkpoints of the state a model
    /// keeps per request slot (a recurrent state); the engine's default
    /// when unset.
    pub state_interval: Option<NonZeroU32>,
}

impl Default for KvCacheConfig {
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
pub struct MooncakeConfig {
    /// Mooncake's master, `host:port`.
    pub master: String,
    /// The transfer engine's metadata server: `P2PHANDSHAKE`, or an
    /// `http://`, `etcd://` or `redis://` URL.
    #[serde(default = "MooncakeConfig::default_metadata_server")]
    pub metadata_server: String,
    /// The transfer protocol: `tcp` or `rdma`.
    #[serde(default = "MooncakeConfig::default_protocol")]
    pub protocol: String,
    /// The RDMA devices; selected automatically when empty.
    #[serde(default)]
    pub device: String,
    /// Host storage each host contributes to Mooncake's pool, in GiB; zero
    /// makes the hosts clients of segments others provide.
    #[serde(default)]
    pub global_segment_gib: u64,
    /// `libmooncake_store.so`; `MOONCAKE_STORE_LIBRARY` or the loader's
    /// search path when unset.
    #[serde(default)]
    pub library: Option<PathBuf>,
}

impl MooncakeConfig {
    fn default_metadata_server() -> String {
        "P2PHANDSHAKE".into()
    }

    fn default_protocol() -> String {
        "tcp".into()
    }
}

/// Default maximum calls selected by one generation scheduling pass.
pub const DEFAULT_MAX_BATCH: usize = 128;
/// Default token budget for one generation scheduling pass.
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;
/// Default maximum number of concurrently running requests.
pub const DEFAULT_MAX_RUNNING_REQUESTS: usize = 128;
/// Default token threshold above which a prefill is chunked.
pub const DEFAULT_CHUNKED_PREFILL_SIZE: usize = DEFAULT_MAX_NUM_BATCHED_TOKENS;

/// Admission and batching (`scheduler_config`), and the limits a scheduler runs with once the
/// engine resolved them.
///
/// The section states the policy, the waiting queue, the prefill chunk and the batch. When the
/// engine starts a scheduler it resolves the rest from the configuration and the ranks' captured
/// catalog ([`SchedulerConfig::of`]): those fields are never read from a configuration file and
/// never written to one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Waiting-queue order: `fcfs` or `priority`.
    pub policy: SchedulingPolicy,
    /// Waiting requests, and finished ones with undelivered events, before
    /// a submission is refused as overloaded; unbounded when unset.
    pub max_queued_requests: Option<NonZeroUsize>,
    /// Prompt tokens one request prefills in one step, 8192 when unset;
    /// token models only.
    pub chunked_prefill_size: Option<NonZeroUsize>,
    /// Calls assembled into one forward batch; once resolved, the largest bucket of the ranks'
    /// catalog, which is what the workers accept.
    pub max_batch: NonZeroUsize,
    /// Token budget of one generation scheduling pass, resolved: the configuration's
    /// `max_num_batched_tokens`, or what the ranks' catalog bounds a step to. Each selected call is
    /// charged its token work, which for a denoising call covers its latent units and guidance
    /// branches.
    #[serde(skip)]
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently running requests, token and media together, resolved from the
    /// configuration's `max_running_requests`.
    #[serde(skip)]
    pub max_running_requests: usize,
    /// Block-diffusion sampling of every canvas the deployment generates, resolved from the model.
    /// A block-diffusion runtime has no autoregressive decode, so each request it does not read
    /// out runs under this sampling; without it the runtime serves readouts only.
    #[serde(skip)]
    pub canvas: Option<crate::request::CanvasSampling>,
    /// The most draft tokens a verification carries per request
    /// (`speculative_config.num_draft_tokens`); zero when the deployment does not speculate.
    #[serde(skip)]
    pub draft_tokens: usize,
    /// The captured graph sizes the ranks state, in tokens, ascending and without duplicates:
    /// `Room.graph_tiers`, the bound a plan's batch must fit. Empty when they state none, and on a
    /// reconfigure, which carries no graph catalog: a worker then runs its own captured catalog or
    /// eagerly, and the tier bound is honest but vacuous.
    #[serde(skip)]
    pub graph_tiers: Vec<u32>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Fcfs,
            max_queued_requests: None,
            chunked_prefill_size: None,
            max_batch: positive(DEFAULT_MAX_BATCH),
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_running_requests: DEFAULT_MAX_RUNNING_REQUESTS,
            canvas: None,
            draft_tokens: 0,
            graph_tiers: Vec::new(),
        }
    }
}

impl SchedulerConfig {
    /// The limits of a scheduler of `engine` whose ranks' captured catalog bounds a step to
    /// `max_batch` calls and `max_num_batched_tokens` tokens, `graph_tiers` being the catalog's
    /// sizes in tokens, ascending (§4.4, §4.5).
    ///
    /// The rest is the configuration's: its running requests, its waiting queue and its prefill
    /// chunk, the model's canvas sampling, and the drafter's tokens.
    pub fn of(
        engine: &EngineConfig,
        max_batch: usize,
        max_num_batched_tokens: usize,
        graph_tiers: Vec<u32>,
    ) -> Self {
        Self {
            policy: engine.scheduler.policy,
            max_queued_requests: engine.scheduler.max_queued_requests,
            chunked_prefill_size: Some(
                engine
                    .scheduler
                    .chunked_prefill_size
                    .unwrap_or(positive(DEFAULT_CHUNKED_PREFILL_SIZE)),
            ),
            max_batch: positive(max_batch),
            max_num_batched_tokens,
            max_running_requests: engine.max_running_requests.get(),
            canvas: engine.model.canvas_sampling.clone(),
            draft_tokens: engine.speculative.num_draft_tokens as usize,
            graph_tiers,
        }
    }

    /// The prefill chunk, as resolved: the configured one, or the default.
    pub fn prefill_chunk(&self) -> usize {
        self.chunked_prefill_size
            .map_or(DEFAULT_CHUNKED_PREFILL_SIZE, NonZeroUsize::get)
    }
}

/// Speculative decoding with the model's drafter (`speculative_config`).
///
/// A deployment that speculates verifies each running request's draft
/// tokens, which the model's drafter proposes from its draft cache, in one
/// forward. A request whose draft cache is missing, such as one an epoch
/// change moved, verifies none while its draft cache is refilled with the
/// idle prefill budget of the steps, when the refill policy finds it worth
/// its cost.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculativeConfig {
    /// The most draft tokens one verification carries per request; zero
    /// does not speculate. The draft cache and the drafter's work belong
    /// to the epoch, so a change of it is an epoch change.
    pub num_draft_tokens: u32,
    /// The speedup speculation gives a request, the decode steps' time one
    /// of its verification steps takes the place of, stated in place of
    /// the one the epoch measures; at least one, measured when unset.
    pub expected_speedup: Option<f64>,
    /// The time of evaluating one context token in a refill, in steps of
    /// decoding, stated in place of the one the epoch measures; not
    /// negative, measured when unset.
    pub refill_token_cost: Option<f64>,
}

/// Rank layout without a deployment file, and product transport
/// (`device_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeviceConfig {
    /// Device of every rank; `cuda` numbers each host's GPUs from zero.
    pub device: String,
    /// Each rank's CUDA device index on its host, in rank order; empty
    /// numbers each host's ranks from zero. A change of the parallel
    /// configuration that does not state it takes the order that leaves
    /// the most weights on the GPUs already holding them.
    #[serde(deserialize_with = "values::list::deserialize")]
    pub devices: Vec<u32>,
    /// Hosts after this one the ranks spread across, in equal blocks.
    #[serde(deserialize_with = "values::list::deserialize")]
    pub worker_hosts: Vec<String>,
    /// Batches in flight per rank; a deployment group states its own.
    pub queue_depth: NonZeroUsize,
    /// Product transport per worker edge,
    /// `source[:rank]->destination[:rank]=mechanisms`, comma-separated.
    pub transfer: Option<String>,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device: "cuda".to_owned(),
            devices: Vec::new(),
            worker_hosts: Vec::new(),
            queue_depth: positive(2),
            transfer: None,
        }
    }
}

/// Mixture-of-experts parallelism (`moe_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MoeConfig {
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

impl Default for MoeConfig {
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
pub struct CudaGraphConfig {
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

impl CudaGraphConfig {
    /// The bounds the deployment's captured graphs put on a batch.
    ///
    /// These are the bounds the engine admits a batch under and the bounds a
    /// scheduling policy's declared limits must fit inside
    /// (`crate::catalog::largest_tier`, which both ends read). Absent when the
    /// deployment states no catalog: the worker's own catalogue then bounds
    /// the batch, and there is no stated figure to hold a policy against.
    pub fn tiers(&self) -> crate::catalog::Tiers {
        let sizes = |values: &Option<Vec<NonZeroU32>>| -> Vec<u32> {
            values
                .iter()
                .flatten()
                .map(|size| size.get())
                .collect::<Vec<_>>()
        };
        crate::catalog::largest_tier(
            &sizes(&self.decode_batch_sizes),
            &sizes(&self.prefill_token_sizes),
        )
    }
}

impl Default for CudaGraphConfig {
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
pub struct FlashInferConfig {
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

impl Default for FlashInferConfig {
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
pub struct WorkerConfig {
    /// The `uniserve` executable the deployment's main processes run as.
    pub executable: PathBuf,
    /// Time a batch may run without rank progress before the worker group is
    /// replaced; a rank reports a stalled job after half of it.
    #[serde(with = "values::duration")]
    pub watchdog_timeout: Duration,
    /// `tower=gen:<device>` places the generation tower on another device
    /// than the rank's.
    pub mesh: Option<String>,
    /// Keep a standby beside every worker group: it maps the group's
    /// retained weights and cache, captures its graphs and parks, and
    /// replaces the group when it fails instead of a relaunch. A group whose
    /// ranks all run on the engine's host and that shares no expert world
    /// keeps a standby rank group of its own; the groups of an expert world,
    /// or a group on several nodes, keep a standby epoch: a second launch of
    /// every group of the failure domain, with communicators of its own,
    /// promoted together when one of them fails. Each standby rank holds a
    /// CUDA context, graph pools and workspaces of its own, which the storage
    /// `mem_fraction_static` leaves must hold.
    pub standby: bool,
    /// How a parked standby (a group's, a failure domain's epoch, or a hot
    /// update's engine) keeps the device storage of its graph pools, scratch,
    /// expert all-to-all workspaces and NCCL communicators: `auto` keeps it
    /// resident while every device has the share a serving worker leaves
    /// for graphs and activations to spare, and suspends it otherwise, every
    /// rank of an epoch or group alike; `resident`; or `suspended`, released
    /// on the device and restored at the promotion, the graph storage with
    /// its contents at unchanged addresses; workspaces that cannot keep
    /// their addresses (DeepEP's windows, colocated MegaMoE's NVSHMEM heap)
    /// are created again at the promotion, which captures the graphs that
    /// use them anew.
    pub standby_storage: StandbyStorage,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            executable: default_executable(),
            watchdog_timeout: Duration::from_secs(300),
            mesh: None,
            standby: false,
            standby_storage: StandbyStorage::Auto,
        }
    }
}

/// The `uniserve` executable of the virtual environment this process runs
/// in, which runs the main processes.
///
/// A daemon built outside one, such as a development build, takes the active
/// environment's; without one, `uniserve` is looked up on the search path.
pub fn default_executable() -> PathBuf {
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
pub struct LoggingConfig {
    /// Log a summary line for each completed request.
    pub log_requests: bool,
    /// Log engine statistics periodically.
    pub log_stats: bool,
}

impl Default for LoggingConfig {
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

/// Chat models: `Qwen3` (`chat_config`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatConfig {
    /// Chat template, as a file path or the template itself.
    pub template: Option<String>,
    /// Keyword arguments bound into every chat-template render.
    pub template_kwargs: Option<HashMap<String, Value>>,
    /// How message content renders: `auto`, `string` or `openai`.
    #[serde(with = "content_format")]
    pub template_content_format: Option<ChatTemplateContentFormat>,
    /// `auto` splits `reasoning_content` from `content`; `none` streams
    /// reasoning delimiters as content.
    pub reasoning_parser: ReasoningParser,
}

/// Fetching input images of chat requests (`image_fetch_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageFetchConfig {
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

impl Default for ImageFetchConfig {
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
pub struct VideoConfig {
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

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            max_seconds: MAX_VIDEO_SECONDS,
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
pub struct MediaConfig {
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

impl Default for MediaConfig {
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

/// The workers' always-on profiling (`profiling_config`): the fence
/// probes at the entry and exit of the model's components, which export
/// per-step component durations (`docs/profiling.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProfilingConfig {
    /// Whether the probes record: the device switch every probe reads,
    /// changed in place, at the engine's next step boundary.
    pub fence_probes: bool,
    /// Name patterns (`*` matches any run) of the components whose module
    /// instances carry probes in the CUDA graphs an epoch captures; a change
    /// takes a new epoch. Empty for no probe.
    pub fence_components: Vec<String>,
    /// Name patterns (`*` matches any run) of the probed components whose
    /// probes are off while `fence_probes` is on: the components whose
    /// probes cost more than the always-on budget. Changed in place, at the
    /// engine's next step boundary, as `fence_probes` is.
    pub fence_probes_off: Vec<String>,
    /// Whether every worker enables CUPTI's PC sampling before it captures
    /// its CUDA graphs and keeps it enabled, stopped between sessions, so
    /// that sessions attribute the samples of graph kernels. CUPTI then
    /// refuses PM sampling, the range profiler and whole-graph activity
    /// records, so sessions of those are refused. The graphs are captured
    /// at an epoch's start, so a change takes a new epoch.
    pub graph_pc_sampling: bool,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            fence_probes: true,
            fence_components: DEFAULT_FENCE_COMPONENTS
                .iter()
                .map(|&pattern| pattern.to_owned())
                .collect(),
            fence_probes_off: Vec::new(),
            graph_pc_sampling: false,
        }
    }
}

/// The components probed by default: each model's decoder stack (its
/// `Transformer` component), one instance per step. Each probe costs a
/// kernel launch in every replay, about a microsecond on GB200 even when
/// switched off, so probing every layer exceeds the always-on budget
/// (`docs/profiling.md`).
pub const DEFAULT_FENCE_COMPONENTS: [&str; 1] = ["*.Transformer"];

/// The shape a chat template takes message `content` in.
///
/// A configuration states `string` or `openai`, or leaves the shape to be detected from the
/// template's source (`auto`, which is the field unset). `Preserve` is never stated: only detection
/// selects it, for a template that handles both shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplateContentFormat {
    /// Content is a simple string.
    #[default]
    String,
    /// Content is a list of structured parts (OpenAI format).
    OpenAi,
    /// String messages remain strings and structured messages remain lists.
    Preserve,
}

impl FromStr for ChatTemplateContentFormat {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("string") {
            Ok(Self::String)
        } else if value.eq_ignore_ascii_case("openai") {
            Ok(Self::OpenAi)
        } else {
            Err(format!(
                "invalid content format `{value}`; expected one of: auto, string, openai"
            ))
        }
    }
}

impl fmt::Display for ChatTemplateContentFormat {
    /// Writes the configuration literal.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::String => "string",
            Self::OpenAi => "openai",
            Self::Preserve => "preserve",
        })
    }
}

/// `chat_config.template_content_format`: `auto`, or the field unset, detects the shape.
mod content_format {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::ChatTemplateContentFormat;

    pub(super) fn serialize<S: Serializer>(
        format: &Option<ChatTemplateContentFormat>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match format {
            Some(format) => serializer.collect_str(format),
            None => serializer.serialize_str("auto"),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<ChatTemplateContentFormat>, D::Error> {
        let text = String::deserialize(deserializer)?;
        if text.eq_ignore_ascii_case("auto") {
            return Ok(None);
        }
        text.parse().map(Some).map_err(serde::de::Error::custom)
    }
}

/// Moving token requests between engines (`migration_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MigrationConfig {
    /// Migrations caused by lost workers, a lost engine process or a failed
    /// epoch on trial that a request survives; it fails on the next.
    /// Migrations of an epoch change or a rollback are not counted.
    pub max_migrations: u32,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self { max_migrations: 3 }
    }
}

/// Asynchronous video jobs (`video_job_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VideoJobConfig {
    /// Time a finished job stays retrievable.
    #[serde(with = "values::duration")]
    pub retention: Duration,
    /// Generated video the jobs keep in shared memory.
    #[serde(with = "values::byte_count")]
    pub max_bytes: usize,
}

impl Default for VideoJobConfig {
    fn default() -> Self {
        Self {
            retention: Duration::from_secs(3600),
            // 1 GiB.
            max_bytes: 1 << 30,
        }
    }
}

/// The model an epoch serves, and how its parameters are represented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Local model directory or Hugging Face repository.
    pub model: String,
    /// Longest sequence a request may reach, in tokens: the configured `max_model_len`, else
    /// what the model's files allow. The options state only the configured one (zero when
    /// unset); the leader resolves it when it reads the checkpoint.
    pub max_model_len: u32,
    /// Element type of parameters and activations.
    pub dtype: crate::request::ModelDtype,
    /// The checkpoint's quantization (`quantization`, SGLang's precision policy).
    pub quantization: serde_json::Map<String, Value>,
    /// The model's generation requirements, which the engine intersects with its workers'
    /// capacities. The rest of the fields below are read from the checkpoint as well: they keep
    /// their defaults until the leader, and then the engine process, read it.
    pub generation_limits: crate::request::GenerationLimits,
    pub bos: u32,
    /// Every end-of-sequence token, the primary one first.
    pub eos: Vec<u32>,
    pub end_of_image: u32,
    /// A block-diffusion model's canvas sampling; absent for other models.
    pub canvas_sampling: Option<crate::request::CanvasSampling>,
    /// Whether a prompt's prefill returns outputs, which only conditioning prefills of
    /// block-diffusion models do not.
    pub prefill_outputs: bool,
}

impl ModelConfig {
    /// Whether `other` processes requests as this model does: the same generation limits,
    /// control tokens, context length and canvas sampling, read from the checkpoint.
    pub fn processes_as(&self, other: &Self) -> bool {
        self.generation_limits == other.generation_limits
            && self.bos == other.bos
            && self.eos == other.eos
            && self.end_of_image == other.end_of_image
            && self.max_model_len == other.max_model_len
            && self.canvas_sampling == other.canvas_sampling
            && self.prefill_outputs == other.prefill_outputs
    }
}

/// How an epoch's ranks divide the model and the requests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelConfig {
    /// Tensor-parallel workers of each scheduler's group when no deployment file is given.
    pub tensor_parallel_size: NonZeroUsize,
    /// Schedulers, each with its own KV cache and worker groups.
    pub data_parallel_size: NonZeroUsize,
    /// The host whose engine processes serve requests, where a rank without a host runs.
    pub engine_host: String,
    /// The worker groups: the deployment file's, placed on their hosts, or the command line's
    /// shorthand ([`WorkerGroupConfig::shorthand`]).
    pub worker_groups: Vec<WorkerGroupConfig>,
}

/// Everything an epoch's engine processes are built from (§7.1, §7.7).
///
/// The leader daemon resolves it once from the serving options and hands the same value to every
/// engine process of the epoch. Any change to it is a new epoch, except a change to a value that
/// selects a program's backend or is a program's parameter, which the program layer applies in
/// place; [`EngineConfig::build_hash`] tells the two apart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineConfig {
    /// The build of the engine executable the epoch runs: a replaced executable is a new epoch
    /// even at the same configuration.
    pub executable: String,
    /// The model and its numerical representation.
    pub model: ModelConfig,
    /// The ranks' division of the model and the requests.
    pub parallel: ParallelConfig,
    /// The deployment: its served model name, under which the Storage Servers keep its claims
    /// and pages.
    pub deployment: String,
    /// Concurrently resident requests: the request slots every rank holds.
    pub max_running_requests: NonZeroUsize,
    /// Token work admitted in one scheduling step; the engine's default when unset.
    pub max_num_batched_tokens: Option<NonZeroUsize>,
    /// The attention program's backend, and the ones that fill prompt processing and one decoding
    /// token when they differ.
    pub attention_backend: String,
    pub prefill_attention_backend: Option<String>,
    pub decode_attention_backend: Option<String>,
    /// The grouped expert GEMM program's backend.
    pub moe_runner_backend: Option<String>,
    pub device: DeviceConfig,
    pub scheduler: SchedulerConfig,
    pub media: MediaConfig,
    /// Each rank's share of its device's storage.
    pub mem_fraction_static: f64,
    pub load: LoadConfig,
    pub kv_cache: KvCacheConfig,
    pub moe: MoeConfig,
    pub cuda_graph: CudaGraphConfig,
    pub worker: WorkerConfig,
    pub speculative: SpeculativeConfig,
    pub video: VideoConfig,
    pub profiling: ProfilingConfig,
}

impl EngineConfig {
    /// The identity of the build this configuration makes: two configurations whose hashes differ
    /// need a new epoch, and two whose hashes agree differ at most in what the program layer
    /// changes in place (§7.7).
    ///
    /// Every value enters it except those §7.7 assigns elsewhere: the program backends (attention,
    /// the grouped expert GEMM, the expert exchange); the scheduling policy's parameters (the
    /// scheduler section, the batched token budget, the speculative cost figures); the probes'
    /// switches, which change at a step boundary; and the media section, which the frontend reads.
    pub fn build_hash(&self) -> String {
        let mut build = self.clone();
        build.attention_backend = String::new();
        build.prefill_attention_backend = None;
        build.decode_attention_backend = None;
        build.moe_runner_backend = None;
        build.scheduler = SchedulerConfig::default();
        build.max_num_batched_tokens = None;
        build.media = MediaConfig::default();
        build.moe.exchange = MoeConfig::default().exchange;
        build.speculative.expected_speedup = None;
        build.speculative.refill_token_cost = None;
        build.profiling.fence_probes = ProfilingConfig::default().fence_probes;
        build.profiling.fence_probes_off = Vec::new();
        let bytes = serde_json::to_vec(&build).expect("a configuration serializes");
        let digest = <sha2::Sha256 as sha2::Digest>::digest(&bytes);
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// Everything one engine process starts with (`uniserve engine --launch <path>`): the epoch's
/// engine configuration, which every engine process of the epoch shares; the epoch's number and
/// what the process takes over from the one it succeeds; where this process sits in the epoch; and the
/// interfaces it reaches.
///
/// A replica's record of what serves is the arguments its engine process started with: the daemon
/// keeps one per replica, and starts a replacement or a standby from it. The daemon's engine
/// control states the process's own place and interfaces when it launches one, so arguments built
/// from a configuration alone ([`EngineArgs::new`]) state none yet. The claim token and the
/// Storage Server's socket are runtime facts of the node, which the daemon answers when the
/// process asks, so they are not here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineArgs {
    /// The epoch's engine configuration, the same for every engine process of the epoch.
    pub engine_config: EngineConfig,
    /// The epoch's number, which keys the storage the epoch creates. It is zero until the first
    /// process of a new epoch starts, and that process's engine number becomes the epoch's.
    pub epoch: u64,
    /// The process this one succeeds on the same devices, whose KV cache it maps read-only.
    ///
    /// It is present for the serving process's standby (§2.5), which claims its scope's storage
    /// read-only, parks once it is prepared, and is promoted in place when the serving process of
    /// its failure domain fails. A standby that claimed writable storage would give the scope two
    /// writers, so the role is stated here, before the process claims anything.
    pub predecessor: Option<Predecessor>,
    /// The expert order each MoE module loads, keyed by module, once the daemon has rebalanced the
    /// experts; empty for the checkpoint's own order.
    pub expert_order: BTreeMap<String, ExpertOrder>,
    /// This node's host identity, which keys this rank's storage claim.
    pub host: String,
    /// The leader daemon's request interface, `unix:<path>` or
    /// `<host>:<port>`.
    pub requests: String,
    /// The daemon's process interface, `unix:<path>` or `<host>:<port>`: the
    /// supervision frames this process claims its storage with and reports to,
    /// and the one place a program-layer command can reach it. A different
    /// vocabulary from `requests`, not a second address of the same one.
    pub processes: String,
    /// The identity the daemon started this process under: the engine number,
    /// or the launch generation in the epoch-switch path, where that is the
    /// only allocator.
    pub engine: u64,
    /// The Program Server's socket.
    pub program_server: String,
    /// The devices this rank serves, as CUDA ordinals.
    pub devices: Vec<u32>,
    /// This rank's worker group, which keys its storage claims.
    pub group: String,
    /// This rank's ordinal inside its group.
    pub rank: u32,
    /// Whether this node holds the epoch's schedulers.
    pub scheduler_node: bool,
    /// The addresses on which this node accepts its peer main processes'
    /// connections; empty for a world of one.
    pub peers: Vec<String>,
    /// How long a job may run without progress before it is reported
    /// `Stalled`, when the daemon states it apart from the engine configuration.
    pub watchdog_timeout_s: Option<u64>,
    /// The probation an update must survive before the daemon commits it.
    pub rollback_s: Option<u64>,
    /// The rank layout's queue depth for this rank's worker groups.
    pub queue_depth: u32,
    /// The number of ranks in this process's world, of which this group's rank
    /// is one.
    pub world_size: u32,
}

impl EngineArgs {
    /// The arguments of an engine process of `engine_config` before the daemon's engine control
    /// states the process's place and interfaces: a new epoch, succeeding no process, with the
    /// checkpoint's expert order.
    pub fn new(engine_config: EngineConfig) -> Self {
        Self {
            engine_config,
            epoch: 0,
            predecessor: None,
            expert_order: BTreeMap::new(),
            host: String::new(),
            requests: String::new(),
            processes: String::new(),
            engine: 0,
            program_server: String::new(),
            devices: Vec::new(),
            group: String::new(),
            rank: 0,
            scheduler_node: false,
            peers: Vec::new(),
            watchdog_timeout_s: None,
            rollback_s: None,
            queue_depth: 0,
            world_size: 0,
        }
    }

    /// Whether this process starts as the serving process's standby: it states the process it
    /// succeeds.
    pub fn standby(&self) -> bool {
        self.predecessor.is_some()
    }

    /// §2.4's watchdog bound: this process's when the daemon states one apart from the epoch's.
    pub fn watchdog_timeout(&self) -> Option<Duration> {
        self.watchdog_timeout_s.map(Duration::from_secs)
    }
}
