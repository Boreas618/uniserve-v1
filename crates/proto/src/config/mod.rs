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
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::request::wire::{ExpertOrder, Predecessor};
use crate::request::{PrefixCacheDtype, StandbyStorage};
pub use block::{
    BlockConfig, BlockConfigError, BlockDistribution, BlockWeights, HeadProjection,
    SequenceParallel,
};
pub use worker_group::{
    DEFAULT_BLOCK, VIDEO_CODEC_BLOCK, VIDEO_DECODER_BLOCK, WorkerGroupConfig, WorkerGroupId,
    WorkerGroupRole, WorkerRank,
};

/// A positive count, for the defaults of non-zero options.
pub const fn positive(count: usize) -> NonZeroUsize {
    match NonZeroUsize::new(count) {
        Some(count) => count,
        None => NonZeroUsize::MIN,
    }
}

/// How the HTTP server answers (`server_config`); where it listens is the daemon's
/// `admin_config`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Bearer token every route but `/health` and `/metrics` requires;
    /// `UNISERVE_API_KEY` supplies it when unset or blank. Never printed.
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
    /// Time until a response head is sent, answered with 504 when it
    /// expires; streamed bodies are not bounded.
    #[serde(with = "values::optional_duration")]
    pub request_timeout: Option<Duration>,
    /// In-flight chat and synchronous video requests above
    /// which requests are shed with 503.
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
pub struct LoadConfig {
    /// A plugin's weight source (`scheme://...`) that streams the
    /// checkpoint's tensors in place of its files; the files when unset.
    pub weights: Option<String>,
}

/// Tokens per KV page when a deployment states no `kv_cache_config.block_size`: the largest page
/// the paged-attention packages read, which keeps a request's page table narrowest. A block of the
/// buffer pool's KV classes is one page (`specs/buffer-pool.md` §6).
pub const DEFAULT_BLOCK_SIZE: u32 = 64;

/// The host KV cache's tokens per device KV cache token when a deployment states neither
/// `kv_cache_config.host_ratio` nor `host_gib`: a ratio, SGLang's `--hicache-ratio` default, so
/// a prefix the device cache evicted is found on the host for as long again as the device held
/// it.
pub const DEFAULT_HOST_RATIO: f64 = 2.0;

/// The bytes of a recurrent state region beyond its per-slot floor per byte of the KV planes when
/// a deployment states no `kv_cache_config.state_ratio`: one, an equal division, near SGLang's
/// Mamba pool at its `mamba_full_memory_ratio` of 0.9, so neither a long prompt's pages nor the
/// checkpoints its prefix reuse needs run out first.
pub const DEFAULT_STATE_RATIO: f64 = 1.0;

/// The host storage of one cache group's converted-load source buffer per rank, in MiB, when a
/// deployment states no `kv_cache_config.conversion_buffer_mib`: a GiB, which holds a long
/// prompt's dense pages of the widest row for the models served while keeping the host storage
/// a group pins on every rank within a GiB.
pub const DEFAULT_CONVERSION_BUFFER_MIB: NonZeroU64 = match NonZeroU64::new(1024) {
    Some(mib) => mib,
    None => NonZeroU64::MIN,
};

/// The KV cache (`kv_cache_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvCacheConfig {
    /// Capacity in tokens; sized from storage when unset.
    pub max_tokens: Option<NonZeroU64>,
    /// Tokens per page of the cache group with the widest token rows, a
    /// power of two the attention kernels read; [`DEFAULT_BLOCK_SIZE`] when
    /// unset ([`KvCacheConfig::block_tokens`]).
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
    /// fifths of its available storage hold. [`DEFAULT_HOST_RATIO`] unless
    /// stated.
    pub host_ratio: f64,
    /// The deployment's host KV cache on each host, in GiB, over the ranks
    /// there, in place of `host_ratio` (as SGLang's `--hicache-size`).
    pub host_gib: Option<NonZeroU64>,
    /// Keep every page of the host KV cache in Mooncake, the KV storage,
    /// through the Mooncake connector of each host's Storage Server, and
    /// prefetch long hits back. Applies with `host_tier`; none when unset.
    pub mooncake: Option<storage::connector::Settings>,
    /// The pages of the largest block size between the aligned boundaries
    /// a long prompt offloads its sliding windows before; the engine's
    /// default when unset.
    pub boundary_interval: Option<NonZeroU32>,
    /// The decoded tokens between two checkpoints of the state a model
    /// keeps per request slot (a recurrent state); the engine's default
    /// when unset.
    pub state_interval: Option<NonZeroU32>,
    /// For a model that keeps a recurrent state, the bytes of its state
    /// region beyond one checkpoint per request slot over the bytes of the
    /// KV planes, which divide what a rank's share leaves between them;
    /// positive. [`DEFAULT_STATE_RATIO`] unless stated.
    pub state_ratio: f64,
    /// The Storage Server endpoints (`host:port`) of other hosts serving the
    /// same model under the same served name, whose published host KV cache
    /// pages admission reads when they deepen a prompt's hit: the pages are
    /// pinned there by key, read into this host's pools by its Transfer
    /// Server and loaded like host pages. The endpoints are unauthenticated;
    /// the hosts rely on the isolation of their network. Applies with
    /// `host_tier`; none when empty.
    pub peers: Vec<String>,
    /// The most host storage, in MiB, one cache group's converted-load
    /// source buffer takes on each rank: the buffer a peer's pages of
    /// another representation are read into before a load converts them
    /// (an FP8 cache reading peers' dense pages). It holds one request's
    /// pages of the widest row within this bound; a page it finds no slot
    /// for is computed instead. [`DEFAULT_CONVERSION_BUFFER_MIB`] unless
    /// stated.
    pub conversion_buffer_mib: NonZeroU64,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            max_tokens: None,
            block_size: None,
            dtype: None,
            prefix_caching: true,
            host_tier: true,
            host_ratio: DEFAULT_HOST_RATIO,
            host_gib: None,
            mooncake: None,
            boundary_interval: None,
            state_interval: None,
            state_ratio: DEFAULT_STATE_RATIO,
            peers: Vec::new(),
            conversion_buffer_mib: DEFAULT_CONVERSION_BUFFER_MIB,
        }
    }
}

impl KvCacheConfig {
    /// The tokens one KV page holds: the stated `block_size`, or [`DEFAULT_BLOCK_SIZE`].
    pub fn block_tokens(&self) -> u32 {
        self.block_size.map_or(DEFAULT_BLOCK_SIZE, NonZeroU32::get)
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
/// The order a scheduling policy's waiting queue admits by, which `scheduler_config.order` states.
pub use scheduling::QueueOrder;

/// The scheduling policy package a deployment runs when its configuration names none
/// (`crates/scheduling/policies/arrival`).
pub const DEFAULT_SCHEDULING_POLICY: &str = "arrival";

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
    /// The scheduling policy package's backend (`arrival`): a package the Build Server holds.
    pub policy: String,
    /// The order the policy's waiting queue admits by (its `order` argument): by arrival, or by
    /// the request's own priority and then arrival; the package's default when unset.
    pub order: Option<QueueOrder>,
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
    /// The most draft tokens a verification carries per request
    /// (`speculative_config.num_draft_tokens`); zero when the deployment does not speculate.
    #[serde(skip)]
    pub draft_tokens: usize,
    /// The captured graph sizes the ranks state, in tokens, ascending and without duplicates:
    /// `Room.graph_tiers`, the bound a scheduled batch must fit. Empty when they state none, and on a
    /// reconfigure, which carries no graph catalog: a worker then runs its own captured catalog or
    /// eagerly, and the tier bound is honest but vacuous.
    #[serde(skip)]
    pub graph_tiers: Vec<u32>,
    /// What the scheduling policy module runs within: the fuel of each call into it, and the
    /// bounds of the starvation timeout and the prefetch threshold its limits may declare.
    pub policy_budget: PolicyBudget,
    /// The tokens of the deployment's KV page, resolved from `kv_cache_config.block_size`
    /// ([`KvCacheConfig::block_tokens`]): the block a class whose blocks hold no tokens is counted
    /// in.
    #[serde(skip)]
    pub block_tokens: u32,
}

/// The budget a scheduling policy module runs under (`scheduler_config.policy_budget`): the
/// engine's policy host meters every call into the module by fuel, wasmtime's count of executed
/// instructions, and checks the limits the module declares against these bounds. A call that
/// runs past its fuel is the module's fault, which ends its instance (`specs/scheduler-design.md`
/// §5.2, "Faults").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PolicyBudget {
    /// The fuel one call into the module may consume, and each export the host enters for that
    /// call (the message buffer's allocation and release), before the message's own share.
    pub call_fuel: NonZeroU64,
    /// The fuel a call is given beyond `call_fuel` for every byte of its message, so that
    /// decoding a message that grows with the cache (an instantiation's snapshot of every cached
    /// block, the events of a scheduler that starts beside a warm cache) is not cut short.
    pub message_fuel_per_byte: u64,
    /// The longest starvation timeout a policy may declare (its limits' `timeout`).
    #[serde(with = "values::duration")]
    pub max_timeout: Duration,
    /// The smallest prefix-hit deepening a policy may declare worth a prefetch (its limits'
    /// `prefetch_min_tokens`), in tokens.
    pub min_prefetch_tokens: u32,
}

/// The fuel of one call into a scheduling policy module when the deployment states none
/// (`scheduler_config.policy_budget.call_fuel`): far above what a real policy's `observe` and
/// `dispatch` of a step consume (the packaged `fcfs` stays below half of it for a step over a
/// thousand running requests), and a bound on a call that never returns.
pub const DEFAULT_POLICY_CALL_FUEL: NonZeroU64 = match NonZeroU64::new(10_000_000) {
    Some(fuel) => fuel,
    None => NonZeroU64::MIN,
};

/// The fuel a call is given for each byte of its message when the deployment states none
/// (`scheduler_config.policy_budget.message_fuel_per_byte`): typed JSON decoding in a wasm32
/// module costs 50 to 100 fuel a byte, and the packaged `fcfs` spends about 70 a byte of cached
/// blocks on its whole `init`. Twice the decoding's dearest leaves a policy as much again for its
/// own state, so a policy instantiates on, and observes, a warm cache of any size.
pub const DEFAULT_POLICY_MESSAGE_FUEL_PER_BYTE: u64 = 200;

/// The longest starvation timeout a policy may declare when the deployment states none
/// (`scheduler_config.policy_budget.max_timeout`): 30 seconds, as `sched_ext` caps a BPF
/// scheduler's `timeout_ms`, a bound past which a starved request is better refused than kept.
pub const DEFAULT_POLICY_MAX_TIMEOUT: Duration = Duration::from_secs(30);

/// The smallest prefetch threshold a policy may declare when the deployment states none
/// (`scheduler_config.policy_budget.min_prefetch_tokens`): a prefix hit deepened by fewer tokens
/// is computed faster than it is fetched from another tier.
pub const DEFAULT_POLICY_MIN_PREFETCH_TOKENS: u32 = 256;

impl Default for PolicyBudget {
    fn default() -> Self {
        Self {
            call_fuel: DEFAULT_POLICY_CALL_FUEL,
            message_fuel_per_byte: DEFAULT_POLICY_MESSAGE_FUEL_PER_BYTE,
            max_timeout: DEFAULT_POLICY_MAX_TIMEOUT,
            min_prefetch_tokens: DEFAULT_POLICY_MIN_PREFETCH_TOKENS,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: DEFAULT_SCHEDULING_POLICY.to_owned(),
            order: None,
            max_queued_requests: None,
            chunked_prefill_size: None,
            max_batch: positive(DEFAULT_MAX_BATCH),
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_running_requests: DEFAULT_MAX_RUNNING_REQUESTS,
            draft_tokens: 0,
            graph_tiers: Vec::new(),
            policy_budget: PolicyBudget::default(),
            block_tokens: DEFAULT_BLOCK_SIZE,
        }
    }
}

impl SchedulerConfig {
    /// The limits of a scheduler of `engine` whose ranks' captured catalog bounds a step to
    /// `max_batch` calls and `max_num_batched_tokens` tokens, `graph_tiers` being the catalog's
    /// sizes in tokens, ascending (§4.4, §4.5).
    ///
    /// The rest is the configuration's: its running requests, its waiting queue and its prefill
    /// chunk, and the drafter's tokens.
    pub fn of(
        engine: &EngineConfig,
        max_batch: usize,
        max_num_batched_tokens: usize,
        graph_tiers: Vec<u32>,
    ) -> Self {
        Self {
            policy: engine.scheduler.policy.clone(),
            order: engine.scheduler.order,
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
            draft_tokens: engine.speculative.num_draft_tokens as usize,
            graph_tiers,
            policy_budget: engine.scheduler.policy_budget,
            block_tokens: engine.kv_cache.block_tokens(),
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

/// The kind of device the command line's shorthand places every rank on (`device_config.device`).
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    /// A CUDA GPU: each host numbers its ranks' GPUs from zero.
    #[default]
    Cuda,
    /// The host's CPU, which every rank of a host shares.
    Cpu,
    /// PyTorch's meta device: the ranks build their modules' shapes and allocate no storage, so a
    /// launch runs on a host without a GPU.
    Meta,
}

impl DeviceKind {
    /// The kind as the configuration and a rank's device string name it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
            Self::Meta => "meta",
        }
    }

    /// The kind [`Self::name`] names; `None` for a name of no kind.
    pub fn of_name(name: &str) -> Option<Self> {
        [Self::Cuda, Self::Cpu, Self::Meta]
            .into_iter()
            .find(|kind| kind.name() == name)
    }

    /// The device of the rank with index `local` among its host's ranks of this kind: GPU
    /// `local` of the host for a GPU, the kind alone for the devices a host's ranks share.
    pub fn rank_device(self, local: u32) -> RankDevice {
        match self {
            Self::Cuda => RankDevice::cuda(local),
            Self::Cpu | Self::Meta => RankDevice {
                kind: self,
                index: None,
            },
        }
    }
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The device one rank runs on: its kind and, for a device its host numbers, the device's index
/// on the rank's host. A deployment file and a launch write it `KIND[:INDEX]`: `cuda:0`, `cpu`.
///
/// A CUDA device always states its ordinal on the host. The host's CPU and the meta device are
/// shared by the host's ranks and state no index, or one that only tells the memories of ranks
/// apart (a host of one scheduler's ranks per index, as a GPU per scheduler would).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RankDevice {
    pub kind: DeviceKind,
    pub index: Option<u32>,
}

impl RankDevice {
    /// The CUDA device of ordinal `ordinal` on the rank's host.
    pub const fn cuda(ordinal: u32) -> Self {
        Self {
            kind: DeviceKind::Cuda,
            index: Some(ordinal),
        }
    }

    /// The ordinal of a CUDA device on its host; `None` for a device of another kind.
    pub fn cuda_ordinal(&self) -> Option<u32> {
        match self.kind {
            DeviceKind::Cuda => self.index,
            DeviceKind::Cpu | DeviceKind::Meta => None,
        }
    }

    /// Whether several ranks of a group may name this device on one host: the host's CPU, which
    /// every rank of the host shares. Every other device is one rank's.
    pub fn is_shared(&self) -> bool {
        self.kind == DeviceKind::Cpu && self.index.is_none()
    }
}

impl fmt::Display for RankDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.index {
            Some(index) => write!(formatter, "{}:{index}", self.kind),
            None => write!(formatter, "{}", self.kind),
        }
    }
}

impl FromStr for RankDevice {
    type Err = anyhow::Error;

    /// Parses `KIND[:INDEX]`, refusing an unknown kind, an index that is no integer, and a CUDA
    /// device that states no ordinal.
    fn from_str(text: &str) -> anyhow::Result<Self> {
        let (name, index) = match text.split_once(':') {
            Some((name, index)) => {
                let index = index.parse::<u32>().map_err(|_| {
                    anyhow::anyhow!("device {text:?}: the index after `:` is no integer")
                })?;
                (name, Some(index))
            }
            None => (text, None),
        };
        let kind = DeviceKind::of_name(name).ok_or_else(|| {
            anyhow::anyhow!("device {text:?}: {name:?} is no device kind (cuda, cpu or meta)")
        })?;
        anyhow::ensure!(
            kind != DeviceKind::Cuda || index.is_some(),
            "device {text:?}: a CUDA device states its ordinal, `cuda:<ordinal>`"
        );
        Ok(Self { kind, index })
    }
}

impl Serialize for RankDevice {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for RankDevice {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// Batches a rank keeps in flight when its deployment states no depth: one that the device
/// executes and one that the host prepares meanwhile, so the host's preparation of a step
/// overlaps the device's execution of the step before it, and no more, since a third would only
/// hold its slot's storage idle.
pub const DEFAULT_QUEUE_DEPTH: usize = 2;

/// Rank layout without a deployment file, and product transport
/// (`device_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeviceConfig {
    /// Device kind of every rank; `cuda` numbers each host's GPUs from zero.
    pub device: DeviceKind,
    /// Each rank's CUDA device index on its host, in rank order; empty
    /// numbers each host's ranks from zero. A change of the parallel
    /// configuration that does not state it takes the order that leaves
    /// the most weights on the GPUs already holding them.
    pub devices: Vec<u32>,
    /// Hosts after this one the ranks spread across, in equal blocks.
    pub worker_hosts: Vec<String>,
    /// Batches in flight per rank; a deployment group states its own.
    pub queue_depth: NonZeroUsize,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device: DeviceKind::Cuda,
            devices: Vec::new(),
            worker_hosts: Vec::new(),
            queue_depth: positive(DEFAULT_QUEUE_DEPTH),
        }
    }
}

/// The expert-exchange package a deployment that shards its experts runs when its configuration
/// names none: FlashInfer's NVLink all-to-all (`crates/ops/kernels/expert_exchange/alltoall`).
pub const DEFAULT_EXPERT_EXCHANGE: &str = "alltoall";

/// Mixture-of-experts parallelism (`moe_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MoeConfig {
    /// Shard routed experts across the deployment's expert workers, across
    /// schedulers of one rank each, or otherwise over the tensor-parallel
    /// ranks of each scheduler's worker group, each rank holding whole
    /// experts ([`EngineConfig::expert_union`]).
    pub expert_parallel: bool,
    /// The expert-exchange package's backend, which reaches sharded experts (`alltoall`,
    /// `megamoe`, `deepep`): a package the Build Server holds.
    pub exchange: String,
    /// Microbatches overlapping attention with dedicated expert workers.
    pub microbatches: NonZeroU32,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            expert_parallel: false,
            exchange: DEFAULT_EXPERT_EXCHANGE.to_owned(),
            microbatches: NonZeroU32::MIN,
        }
    }
}

/// CUDA graph capture (`cuda_graph_config`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaGraphConfig {
    /// Decode batch sizes captured, strictly increasing; the worker's
    /// catalog when unset.
    pub decode_batch_sizes: Option<Vec<NonZeroU32>>,
    /// Prefill token counts captured, strictly increasing; the worker's
    /// catalog when unset.
    pub prefill_token_sizes: Option<Vec<NonZeroU32>>,
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

/// The worker processes (`worker_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerConfig {
    /// The engine package the deployment's main processes run as: the name of an implementation
    /// of the engine interface installed at the Build Server.
    pub engine: String,
    /// The revision of the engine package to run; the latest installed one, fixed when the
    /// configuration is applied, when unset.
    pub engine_revision: Option<u32>,
    /// Time a batch may run without rank progress before the worker group is
    /// replaced; a rank reports a stalled job after half of it.
    #[serde(with = "values::duration")]
    pub watchdog_timeout: Duration,
    /// How long the main processes of an epoch whose ranks span several hosts
    /// wait for one another as they start: the scheduler node for every peer's
    /// registration, a peer for the scheduler node's acceptance, and every
    /// process for the others at the ranks' host-group rendezvous. A process
    /// that is connected and then exits or loses its host fails the start at
    /// once; the bound covers a process that never arrives.
    #[serde(with = "values::duration")]
    pub join_timeout: Duration,
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
            engine: DEFAULT_ENGINE.to_owned(),
            engine_revision: None,
            watchdog_timeout: DEFAULT_WATCHDOG_TIMEOUT,
            join_timeout: DEFAULT_JOIN_TIMEOUT,
            mesh: None,
            standby: false,
            standby_storage: StandbyStorage::Auto,
        }
    }
}

/// How long a batch may run without its ranks' progress before its worker group is replaced, when
/// the deployment states no bound (`worker_config.watchdog_timeout`): five minutes, a policy bound
/// above the longest step a served model takes (a denoising step of the longest video, a prefill
/// of the longest prompt) and below what an operator waits before replacing a stuck group by hand.
pub const DEFAULT_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(300);

/// How long the main processes of a multi-host epoch wait for one another as they start, when the
/// deployment states no bound (`worker_config.join_timeout`): five minutes, a policy bound above
/// the skew between the hosts' spawns of one epoch's processes (the daemons spawn them together
/// once every host prepared the epoch's modules, and each process registers before it loads
/// anything) and below what an operator waits before replacing a start by hand.
pub const DEFAULT_JOIN_TIMEOUT: Duration = Duration::from_secs(300);

/// The engine package a deployment runs when its configuration names none: the implementation
/// of the engine interface this tree holds (`crates/engine/package.toml`).
pub const DEFAULT_ENGINE: &str = "uniserve";

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

/// Chat-template rendering and reasoning output of the chat families (`chat_config`).
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

/// The time one input image's fetch may take when the deployment states none
/// (`image_fetch_config.timeout`): a request-admission policy bound on one remote fetch,
/// redirects and body included.
pub const IMAGE_FETCH_TIMEOUT: Duration = Duration::from_secs(20);

/// The largest input image a request may carry when the deployment states none
/// (`image_fetch_config.max_bytes`), in bytes: a request-admission policy of 20 MB, the
/// per-image limit OpenAI's API states for image inputs.
pub const IMAGE_FETCH_MAX_BYTES: u64 = 20_000_000;

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
            timeout: IMAGE_FETCH_TIMEOUT,
            max_bytes: IMAGE_FETCH_MAX_BYTES,
            allow_private: false,
        }
    }
}

/// Video generation (`video_config`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VideoConfig {
    /// Longest admitted output, in seconds, within the model's own bounds; the model family's
    /// default when unset, which the leader states once it reads the checkpoint.
    pub max_seconds: Option<f64>,
    /// Output sizes startup prepares, each `WxH` in pixels (`1344x768`); the model refuses a
    /// size it does not generate. The model family's default sizes when unset, which the leader
    /// states once it reads the checkpoint; none for a model that generates no video.
    pub sizes: Option<Vec<String>>,
    /// Prompt capacities startup captures, strictly increasing and none above
    /// `max_model_len`, which completes them as the largest; 1024, then steps
    /// of 2048 below `max_model_len`, when unset.
    pub text_capacities: Option<Vec<NonZeroU32>>,
    /// Denoiser rows one request's conditions may take; the largest
    /// condition set the denoiser admits when unset.
    pub max_condition_rows: Option<u32>,
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
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self {
            directory: None,
            allow_remote: true,
            // 256 MiB.
            max_request_bytes: 256 << 20,
        }
    }
}

/// The workers' always-on profiling (`profiling_config`): the fence
/// probes at the entry and exit of the model's modules, which export
/// per-step durations by module class (`docs/profiling.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProfilingConfig {
    /// Whether the probes record: the device switch every probe reads,
    /// changed in place, at the engine's next step boundary.
    pub fence_probes: bool,
    /// Name patterns (`*` matches any run) of the module classes whose
    /// instances carry probes in the CUDA graphs an epoch captures; a change
    /// takes a new epoch. Empty for no probe; the model family's step stack
    /// (the decoder or denoiser stack, one instance per step) when unset.
    pub fence_modules: Option<Vec<String>>,
    /// Name patterns (`*` matches any run) of the probed module classes whose
    /// probes are off while `fence_probes` is on: the classes whose
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
            fence_modules: None,
            fence_probes_off: Vec::new(),
            graph_pc_sampling: false,
        }
    }
}

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

/// The migrations caused by losses a request survives when the deployment states none
/// (`migration_config.max_migrations`): a policy bound, so that a request whose own content
/// fails every engine it reaches fails after a few attempts instead of taking each engine down in
/// turn.
pub const DEFAULT_MAX_MIGRATIONS: u32 = 3;

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
        Self {
            max_migrations: DEFAULT_MAX_MIGRATIONS,
        }
    }
}

/// How long a finished asynchronous video job stays retrievable when the deployment states none
/// (`video_job_config.retention`): a client policy of one hour.
pub const VIDEO_JOB_RETENTION: Duration = Duration::from_secs(3600);

/// The generated video the asynchronous jobs keep in shared memory when the deployment states no
/// budget (`video_job_config.max_bytes`), in bytes: a policy budget of 1 GiB, which an operator
/// sizes to the host storage the daemon may hold.
pub const VIDEO_JOB_MAX_BYTES: usize = 1 << 30;

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
            retention: VIDEO_JOB_RETENTION,
            max_bytes: VIDEO_JOB_MAX_BYTES,
        }
    }
}

/// The model an epoch serves, and how its parameters are represented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Local model directory or Hugging Face repository.
    pub model: String,
    /// The local snapshot of the base release the checkpoint continues, whose directories it
    /// omits; `None` for a checkpoint that holds every directory. The leader resolves it when it
    /// reads the checkpoint, and an engine opens the checkpoint and its base as one.
    pub base: Option<String>,
    /// The weight adapter fused into the checkpoint at load (`lora_path`): an adapter file, or a
    /// directory holding one. The model family names the module it edits, and every request of
    /// the deployment is served by the adapted weights; `None` serves the checkpoint as it is.
    pub lora_path: Option<String>,
    /// Longest sequence a request may reach, in tokens: the configured `max_model_len`, else
    /// what the model's files allow. The options state only the configured one (zero when
    /// unset); the leader resolves it when it reads the checkpoint.
    pub max_model_len: u32,
    /// Element type of parameters and activations.
    pub dtype: crate::request::ModelDtype,
    /// The precision policy (`quantization`), which the model family interprets when it reads
    /// the checkpoint; empty serves the checkpoint's own representation.
    pub quantization: Quantization,
    /// The model's generation requirements, which the engine intersects with its workers'
    /// capacities. The rest of the fields below are read from the checkpoint as well: they keep
    /// their defaults until the leader, and then the engine process, read it.
    pub generation_limits: crate::request::GenerationLimits,
    pub bos: u32,
    /// Every end-of-sequence token, the primary one first.
    pub eos: Vec<u32>,
}

impl ModelConfig {
    /// Whether `other` processes requests as this model does: the same generation limits,
    /// control tokens and context length, read from the checkpoint.
    pub fn processes_as(&self, other: &Self) -> bool {
        self.generation_limits == other.generation_limits
            && self.bos == other.bos
            && self.eos == other.eos
            && self.max_model_len == other.max_model_len
    }
}

/// The precision policy (`quantization`): a mode, and representations selected by module that
/// override what the mode gives those modules.
///
/// Empty selects the checkpoint's own representation. The model family interprets the policy
/// where it reads the checkpoint: it names the modules a selection may state, gives each mode's
/// representation of them, and refuses a policy it does not serve, naming the module, so no
/// statement of the policy is ignored. A checkpoint that declares its own quantization takes
/// none.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Quantization {
    /// The mode every module takes that `modules` does not select.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<PrecisionMode>,
    /// Representations selected by module, under the names the model family declares for its
    /// modules; the family refuses a name it does not declare.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub modules: BTreeMap<String, WeightFormat>,
}

impl Quantization {
    /// Whether the policy states nothing, so the checkpoint's own representation is served.
    pub fn is_empty(&self) -> bool {
        self.mode.is_none() && self.modules.is_empty()
    }
}

impl fmt::Display for Quantization {
    /// The policy as the TOML inline table `--set quantization=…` states it, such as
    /// `{ mode = "performance", modules = { video_vae = "bf16" } }`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut fields = Vec::new();
        if let Some(mode) = self.mode {
            fields.push(format!("mode = \"{mode}\""));
        }
        if !self.modules.is_empty() {
            let modules = self
                .modules
                .iter()
                .map(|(module, format)| format!("{module} = \"{format}\""))
                .collect::<Vec<_>>()
                .join(", ");
            fields.push(format!("modules = {{ {modules} }}"));
        }

        if fields.is_empty() {
            formatter.write_str("{}")
        } else {
            write!(formatter, "{{ {} }}", fields.join(", "))
        }
    }
}

/// A precision mode: what a model family gives each of its modules, from the representation the
/// reference computes in (`quality`) to the narrowest it serves (`maximum`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrecisionMode {
    /// The representation the reference computes in; the default of a dense checkpoint.
    Quality,
    /// The modules whose narrowing changes the output least narrowed.
    Balanced,
    /// More modules narrowed than `balanced`.
    Performance,
    /// The narrowest representation the family states for each module.
    Maximum,
}

impl PrecisionMode {
    /// The mode as the configuration names it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Balanced => "balanced",
            Self::Performance => "performance",
            Self::Maximum => "maximum",
        }
    }
}

impl fmt::Display for PrecisionMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The representation a precision policy gives a module's weights, and with them its products.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightFormat {
    /// BF16 weights, BF16 products.
    Bf16,
    /// FP16 weights, FP16 products.
    Fp16,
    /// E4M3 weights, with activations encoded to E4M3 for the product.
    Fp8,
    /// NVFP4 weights (E2M1 values, E4M3 scales per sixteen elements, one tensor scale), with
    /// activations encoded to NVFP4 for the product.
    Nvfp4,
}

impl WeightFormat {
    /// The representation as the configuration names it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp16 => "fp16",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
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
/// selects a package's backend or is a package's parameter, which the generation layer applies in
/// place; [`EngineConfig::build_part`] tells the two apart.
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
    /// Concurrently resident requests: the request slots every rank holds, or the fewer its share
    /// of its GPU holds.
    pub max_running_requests: NonZeroUsize,
    /// Token work admitted in one scheduling step; the engine's default when unset.
    pub max_num_batched_tokens: Option<NonZeroUsize>,
    /// The most likely tokens a request may ask for at a generated or a prompt position
    /// (`top_logprobs`, `prompt_logprobs`): every sampling graph computes this many for each row
    /// it samples, so it is a property of the captured graphs and a request asking for more is
    /// refused. Zero reports no candidate.
    pub max_logprobs: u32,
    /// The attention package's backend, and the ones that fill prompt processing and one decoding
    /// token when they differ.
    pub attention_backend: String,
    pub prefill_attention_backend: Option<String>,
    pub decode_attention_backend: Option<String>,
    /// The grouped expert GEMM package's backend.
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
    /// Whether `moe_config.expert_parallel` shards the routed experts over an expert union,
    /// whose ranks exchange tokens at every expert layer and so run in one engine: the
    /// deployment's dedicated expert workers (a group of role `experts`) and the ranks that
    /// route tokens to them, or several schedulers of one rank each (`data_parallel_size` above
    /// one). Otherwise the flag shards each serving group's experts over the group's own
    /// tensor-parallel ranks, whose every rank holds every token: no token moves between
    /// groups, so their schedulers are independent replicas, as without the flag.
    pub fn expert_union(&self) -> bool {
        let groups = &self.parallel.worker_groups;
        let dedicated = groups
            .iter()
            .any(|group| group.role == WorkerGroupRole::Experts);
        let single_ranks = groups
            .iter()
            .filter(|group| group.role.serves_requests())
            .all(|group| group.ranks.len() == 1);
        self.moe.expert_parallel
            && (dedicated || (self.parallel.data_parallel_size.get() > 1 && single_ranks))
    }

    /// The part of this configuration that makes the epoch's build ([`BuildPart`]): two
    /// configurations whose build parts differ need a new epoch, and two whose build parts agree
    /// differ at most in what the generation layer changes in place (§7.7).
    ///
    /// Every field of the configuration, and of each of its sections that holds a value of both
    /// layers, is classified by name here, and the destructuring is exhaustive: a field added to
    /// any of these types does not compile until it is placed in the build part or named as a
    /// generation-layer value. The generation layer's values are the package backends
    /// (attention, the grouped expert GEMM, the expert exchange); the scheduling policy's
    /// parameters (the scheduler section, the batched token budget, the speculative cost
    /// figures); the probes' switches, which change at a step boundary; and the media section,
    /// which the frontend reads.
    pub fn build_part(&self) -> BuildPart<'_> {
        let Self {
            executable,
            model,
            parallel,
            deployment,
            max_running_requests,
            max_num_batched_tokens: _,
            max_logprobs,
            attention_backend: _,
            prefill_attention_backend: _,
            decode_attention_backend: _,
            moe_runner_backend: _,
            device,
            scheduler: _,
            media: _,
            mem_fraction_static,
            load,
            kv_cache,
            moe,
            cuda_graph,
            worker,
            speculative,
            video,
            profiling,
        } = self;
        let MoeConfig {
            expert_parallel,
            exchange: _,
            microbatches,
        } = moe;
        let SpeculativeConfig {
            num_draft_tokens,
            expected_speedup: _,
            refill_token_cost: _,
        } = speculative;
        let ProfilingConfig {
            fence_probes: _,
            fence_modules,
            fence_probes_off: _,
            graph_pc_sampling,
        } = profiling;
        BuildPart {
            executable,
            model,
            parallel,
            deployment,
            max_running_requests: *max_running_requests,
            max_logprobs: *max_logprobs,
            device,
            mem_fraction_static: *mem_fraction_static,
            load,
            kv_cache,
            expert_parallel: *expert_parallel,
            microbatches: *microbatches,
            cuda_graph,
            worker,
            num_draft_tokens: *num_draft_tokens,
            video,
            fence_modules,
            graph_pc_sampling: *graph_pc_sampling,
        }
    }

    /// The layout this configuration lays an epoch's device memory out in ([`Layout`]).
    pub fn layout(&self) -> Layout {
        Layout {
            parallel: self.parallel.clone(),
            max_running_requests: self.max_running_requests,
            block_size: self.kv_cache.block_size,
            dtype: self.kv_cache.dtype,
            cuda_graph: self.cuda_graph.clone(),
        }
    }
}

/// The part of an [`EngineConfig`] that makes an epoch's build ([`EngineConfig::build_part`]),
/// every field of it entering the build's identity ([`BuildPart::identity`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BuildPart<'a> {
    pub executable: &'a str,
    pub model: &'a ModelConfig,
    pub parallel: &'a ParallelConfig,
    pub deployment: &'a str,
    pub max_running_requests: NonZeroUsize,
    pub max_logprobs: u32,
    pub device: &'a DeviceConfig,
    pub mem_fraction_static: f64,
    pub load: &'a LoadConfig,
    pub kv_cache: &'a KvCacheConfig,
    pub expert_parallel: bool,
    pub microbatches: NonZeroU32,
    pub cuda_graph: &'a CudaGraphConfig,
    pub worker: &'a WorkerConfig,
    pub num_draft_tokens: u32,
    pub video: &'a VideoConfig,
    pub fence_modules: &'a Option<Vec<String>>,
    pub graph_pc_sampling: bool,
}

impl BuildPart<'_> {
    /// The build's identity: the digest of the whole build part, in lowercase hexadecimal.
    #[expect(
        clippy::expect_used,
        reason = "a configuration is plain data with string keys, which JSON always encodes"
    )]
    pub fn identity(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("a configuration serializes");
        let digest = <sha2::Sha256 as sha2::Digest>::digest(&bytes);
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// What lays an epoch's device memory out (`specs/buffer-pool.md` §14, "A change of layout"): the
/// parallel configuration, whose worker groups state the devices and the hosts; the request slots;
/// the KV cache's block size and element type; and the graph buckets. Two epochs of one layout
/// name the same device KV regions and share their blocks, and an epoch of another layout opens
/// regions of its own, into which the serving epoch's device memory moves as it drains, so the
/// engine names its regions by the layout's identity and the daemon drains with `shrink` exactly
/// when the layouts differ.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Layout {
    pub parallel: ParallelConfig,
    pub max_running_requests: NonZeroUsize,
    pub block_size: Option<NonZeroU32>,
    pub dtype: Option<PrefixCacheDtype>,
    pub cuda_graph: CudaGraphConfig,
}

impl Layout {
    /// The layout's identity: the digest of the whole layout, in lowercase hexadecimal.
    #[expect(
        clippy::expect_used,
        reason = "a configuration is plain data with string keys, which JSON always encodes"
    )]
    pub fn identity(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("a configuration serializes");
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
    /// The engine this one succeeds: present for an epoch switch's target and for the serving
    /// process's standby (§2.5), each of which parks once it is prepared and serves when it is
    /// promoted. It maps nothing of its predecessor: both open sessions with the host's buffer
    /// pool, which keeps two epochs from writing one block.
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
    /// and the one place a generation command can reach it. A different
    /// vocabulary from `requests`, not a second address of the same one.
    pub processes: String,
    /// The identity the daemon started this process under: the engine number,
    /// or the launch generation in the epoch-switch path, where that is the
    /// only allocator.
    pub engine: u64,
    /// The Build Server's socket.
    pub build_server: String,
    /// The epoch's module set, as the daemon fixed it through the Build Server's `cover` when the
    /// epoch's configuration was applied: the node's daemon starts the engine package it names,
    /// and the engine resolves the kernel and policy packages it names, at the revisions it
    /// names, for every key its profile runs state, before it serves or parks.
    pub modules: crate::module::ModuleSet,
    /// The devices this rank serves, as CUDA ordinals.
    pub devices: Vec<u32>,
    /// This rank's worker group, which keys its storage claims.
    pub group: String,
    /// The deployment rank of the first rank this process serves, in the numbering of every worker
    /// group's ranks in the order the groups are stated. A text process's other devices' ranks
    /// follow it; a media process serves every rank the deployment places on its host, which
    /// need not be consecutive.
    pub rank: u32,
    /// Whether this node holds the epoch's schedulers.
    pub scheduler_node: bool,
    /// The epoch's other main processes and where this one meets them, when the epoch's worker
    /// group has ranks on several hosts; `None` when this process holds every rank.
    pub world: Option<World>,
    /// How this process, started as a standby ([`Self::standby`]), keeps the device storage of its
    /// graph pools while it parks: the daemon resolves `worker_config.standby_storage` for the
    /// failure domain, so it states `resident` or `suspended` and never `auto`.
    pub standby_storage: StandbyStorage,
    /// How long a job may run without progress before it is reported
    /// `Stalled`, when the daemon states it apart from the engine configuration.
    pub watchdog_timeout_s: Option<u64>,
    /// The rank layout's queue depth for this rank's worker groups.
    pub queue_depth: u32,
    /// The number of the deployment's ranks, every worker group's on every host: this process's
    /// ranks and those of its peers, the members of the host group they meet in.
    pub world_size: u32,
}

/// Where the main processes of one epoch meet when its worker group has ranks on several hosts
/// (`docs/architecture.md`, The channel).
///
/// The scheduler node's main process holds the group's rank 0 and its schedulers; each other
/// host's main process (a peer) holds that host's ranks. The leader's daemon reserves two sockets
/// on the scheduler node before it starts any process of the epoch, so both addresses are live
/// before a peer dials them: the peer channel's listener and the group's host-collective
/// rendezvous.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct World {
    /// The engine number of the scheduler node's main process, which a peer's registration states:
    /// the channel of another epoch is refused.
    pub scheduler: u64,
    /// Where the scheduler node's main process accepts its peers (`host:port`).
    pub channel: String,
    /// Where the group's ranks form their host collectives (`host:port`), served by the process
    /// that holds rank 0.
    pub store: String,
    /// How many peers the scheduler node's process accepts before its ranks load.
    pub peers: u32,
    /// The descriptors at which the scheduler node's process inherits the two reserved sockets;
    /// `None` on a peer, which dials them.
    pub inherited: Option<Inherited>,
}

/// The descriptors of the sockets the scheduler node's main process inherits from its daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inherited {
    /// The peer channel's listener.
    pub channel: i32,
    /// The host-collective rendezvous.
    pub store: i32,
}

impl EngineArgs {
    /// The arguments of an engine process of `engine_config` that runs `modules`, before the
    /// daemon's engine control states the process's place and interfaces: a new epoch, succeeding
    /// no process, with the checkpoint's expert order.
    pub fn new(engine_config: EngineConfig, modules: crate::module::ModuleSet) -> Self {
        Self {
            engine_config,
            epoch: 0,
            predecessor: None,
            expert_order: BTreeMap::new(),
            host: String::new(),
            requests: String::new(),
            processes: String::new(),
            engine: 0,
            build_server: String::new(),
            modules,
            devices: Vec::new(),
            group: String::new(),
            rank: 0,
            scheduler_node: false,
            world: None,
            standby_storage: StandbyStorage::Resident,
            watchdog_timeout_s: None,
            queue_depth: 0,
            world_size: 0,
        }
    }

    /// Whether this engine starts parked: it states the engine it succeeds, whose storage it
    /// maps read-only until it is promoted. A switch's target and a standby both start so.
    pub fn parked(&self) -> bool {
        self.predecessor.is_some()
    }

    /// §2.4's watchdog bound: this process's when the daemon states one apart from the epoch's.
    pub fn watchdog_timeout(&self) -> Option<Duration> {
        self.watchdog_timeout_s.map(Duration::from_secs)
    }
}
