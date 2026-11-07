//! The configuration sections every process of a deployment reads, one type each.
//!
//! Each type is the section of the deployment file named after it (`kv_cache_config` is
//! [`KvCacheConfig`]): the daemon parses the file into it, and the same type travels to the
//! frontend and the engine processes. A type does not refuse unknown fields, so a process of an
//! older build reads what a newer daemon writes; the daemon refuses a key no section knows when it
//! parses the file (`daemon::options`).

pub mod values;

use std::collections::HashMap;
use std::fmt;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::request::{
    ExpertExchange, FlashInferBackend, GraphPolicy, LoadFormat, PrefixCacheDtype, SchedulingPolicy,
    StandbyStorage,
};
pub use values::ImageSize;

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
#[serde(default)]
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
#[serde(default)]
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
    /// a world of any of them maps the chunks it shares with another;
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
#[serde(default)]
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
    /// same on every rank of a world (as SGLang's `--hicache-ratio`); each
    /// host's Storage Server lowers it for the whole world to what four
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

/// Admission and batching (`scheduler_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SchedulerConfig {
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

impl Default for SchedulerConfig {
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
#[serde(default)]
pub struct SpeculativeConfig {
    /// The most draft tokens one verification carries per request; zero
    /// does not speculate. The draft cache and the drafter's work belong
    /// to the world, so a change of it is a world change.
    pub num_draft_tokens: u32,
    /// The speedup speculation gives a request, the decode steps' time one
    /// of its verification steps takes the place of, stated in place of
    /// the one the world measures; at least one, measured when unset.
    pub expected_speedup: Option<f64>,
    /// The time of evaluating one context token in a refill, in steps of
    /// decoding, stated in place of the one the world measures; not
    /// negative, measured when unset.
    pub refill_token_cost: Option<f64>,
}

/// Rank layout without a deployment file, and product transport
/// (`device_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
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
    /// deployment states no catalog: the executor's own catalogue then bounds
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
#[serde(default)]
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
#[serde(default)]
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
    /// or a group on several nodes, keep a standby world: a second launch of
    /// every group of the failure domain, with communicators of its own,
    /// promoted together when one of them fails. Each standby rank holds a
    /// CUDA context, graph pools and workspaces of its own, which the storage
    /// `mem_fraction_static` leaves must hold.
    pub standby: bool,
    /// How a parked standby (a group's, a failure domain's world, or a hot
    /// update's engine) keeps the device storage of its graph pools, scratch,
    /// expert all-to-all workspaces and NCCL communicators: `auto` keeps it
    /// resident while every device has the share a serving worker leaves
    /// for graphs and activations to spare, and suspends it otherwise, every
    /// rank of a world or group alike; `resident`; or `suspended`, released
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
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
pub struct ProfilingConfig {
    /// Whether the probes record: the device switch every probe reads,
    /// changed in place, at the engine's next step boundary.
    pub fence_probes: bool,
    /// Name patterns (`*` matches any run) of the components whose module
    /// instances carry probes in the CUDA graphs a world captures; a change
    /// takes a new world. Empty for no probe.
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
    /// at a world's start, so a change takes a new world.
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
#[serde(default)]
pub struct MigrationConfig {
    /// Migrations caused by lost workers, a lost engine process or a failed
    /// world on trial that a request survives; it fails on the next.
    /// Migrations of a world change or a rollback are not counted.
    pub max_migrations: u32,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self { max_migrations: 3 }
    }
}

/// Asynchronous video jobs (`video_job_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
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
