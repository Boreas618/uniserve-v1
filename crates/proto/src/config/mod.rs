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

use crate::request::wire::Predecessor;
use crate::request::{PrefixCacheDtype, StandbyStorage};
pub use block::{
    BlockConfig, BlockConfigError, BlockDistribution, BlockWeights, HeadProjection, Partition,
    SequenceParallel,
};
pub use worker_group::{
    DEFAULT_BLOCK, HostBlock, WorkerGroupConfig, WorkerGroupId, WorkerGroupRole, WorkerRank,
};

/// A positive count of a `u32` option's default; a zero count fails the constant's evaluation, so
/// the build refuses it.
const fn nonzero(count: u32) -> NonZeroU32 {
    match NonZeroU32::new(count) {
        Some(count) => count,
        None => panic!("the default of a positive option is zero"),
    }
}

/// A positive count, for the defaults of non-zero options.
pub const fn positive(count: usize) -> NonZeroUsize {
    match NonZeroUsize::new(count) {
        Some(count) => count,
        None => NonZeroUsize::MIN,
    }
}

/// The largest request head the HTTP server reads when the deployment states none
/// (`server_config.max_request_head`), in bytes: a head of the API's routes is a few hundred
/// bytes, and 64 KiB bounds what one connection buffers before its head parses, as the default
/// head limits of common HTTP servers do.
pub const REQUEST_HEAD_BYTES: usize = 64 << 10;

/// The largest request body the HTTP server reads when the deployment states none
/// (`server_config.max_request_body`), in bytes: room for a chat request that carries its
/// images inline as base64 data URIs.
pub const REQUEST_BODY_BYTES: usize = 64 << 20;

/// How long a server-sent-events response waits for its next event before a comment keeps its
/// connection alive, when the deployment states none (`server_config.stream_keep_alive`): below
/// the idle timeout of common proxies and load balancers (60 s), so a stream that waits through
/// a long prefill is not cut by one.
pub const STREAM_KEEP_ALIVE: Duration = Duration::from_secs(15);

/// The longest client `X-Request-Id` the server takes as a request's ID when the deployment
/// states none (`server_config.max_request_id`), in bytes: a generated ID is 32 characters and
/// correlation IDs are short, and the bound caps the client data that enters the logs and the
/// echoed header.
pub const REQUEST_ID_BYTES: usize = 128;

/// The `Retry-After` a request shed by the concurrency limit is answered with when the
/// deployment states none (`server_config.retry_after`): a permit frees as soon as one response
/// ends, so the header's resolution, one second.
pub const RETRY_AFTER: Duration = Duration::from_secs(1);

/// How the HTTP server answers (`server_config`); where it listens is the daemon's
/// `admin_config`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Bearer token every route but `/health` and `/metrics` requires; a
    /// stated key is not empty. Never printed.
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
    /// Largest request head (request line and headers), and largest trailer
    /// section of a chunked request body; a larger head is answered with 431.
    #[serde(with = "values::byte_count")]
    pub max_request_head: usize,
    /// Largest request body of every route but the video submissions, which
    /// take what `media_config.max_request_bytes` of base64 media needs and
    /// never less; a larger body is answered with 413.
    #[serde(with = "values::byte_count")]
    pub max_request_body: usize,
    /// Time a server-sent-events response waits for its next event before a
    /// comment keeps the connection alive.
    #[serde(with = "values::duration")]
    pub stream_keep_alive: Duration,
    /// Longest client `X-Request-Id` taken as the request's ID; a longer one
    /// is replaced by a generated ID.
    #[serde(with = "values::byte_count")]
    pub max_request_id: usize,
    /// `Retry-After` of a request the concurrency limit sheds, in whole
    /// seconds.
    #[serde(with = "values::duration")]
    pub retry_after: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            request_timeout: None,
            max_concurrent_requests: None,
            shutdown_timeout: Duration::from_secs(30),
            request_id_headers: false,
            max_request_head: REQUEST_HEAD_BYTES,
            max_request_body: REQUEST_BODY_BYTES,
            stream_keep_alive: STREAM_KEEP_ALIVE,
            max_request_id: REQUEST_ID_BYTES,
            retry_after: RETRY_AFTER,
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

/// The KV cache (`kv_cache_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, crate::reload::Reload)]
#[serde(default, deny_unknown_fields)]
pub struct KvCacheConfig {
    /// Capacity in tokens; sized from storage when unset.
    #[reload(epoch)]
    pub max_tokens: Option<NonZeroU64>,
    /// Tokens per page of the cache group with the widest token rows, a
    /// power of two the attention kernels read; [`DEFAULT_BLOCK_SIZE`] when
    /// unset ([`KvCacheConfig::block_tokens`]).
    #[reload(epoch)]
    pub block_size: Option<NonZeroU32>,
    /// Element type (`float16`, `bfloat16`, `float32` or `float8_e4m3fn`);
    /// each layer's compute type when unset.
    #[reload(epoch)]
    pub dtype: Option<PrefixCacheDtype>,
    /// Reuse and retain cached prompt prefixes across requests. Switched off, the schedulers
    /// neither read nor write cached prefixes from their next step on, and the cached pages
    /// leave by eviction; switched on, a new epoch builds what an epoch that caches prefixes
    /// lays out beside the device cache (the host tier's region) ([`switched_off`]).
    #[reload(by = switched_off)]
    pub prefix_caching: bool,
    /// Offload full prompt pages to the host KV cache of each host's Storage
    /// Server and load them back on a later prefix match the device cache
    /// no longer holds: every published page of a full-attention group, and
    /// a sliding-window group's window before each aligned boundary.
    /// Applies with `prefix_caching`. Switched off, the schedulers offload no page from their
    /// next step on; switched on, a new epoch opens the host region ([`switched_off`]).
    #[reload(by = switched_off)]
    pub host_tier: bool,
    /// The host KV cache's token count over the device KV cache's, the
    /// same on every rank of an epoch (as SGLang's `--hicache-ratio`); each
    /// host's Storage Server lowers it for the whole epoch to what four
    /// fifths of its available storage hold. [`DEFAULT_HOST_RATIO`] unless
    /// stated. The host region is created at its size when an epoch of the deployment first
    /// opens it, under a name its size is no part of, and each later epoch's open keeps its
    /// numbering, so a change of it is an epoch's.
    #[reload(epoch)]
    pub host_ratio: f64,
    /// The deployment's host KV cache on each host, in GiB, over the ranks
    /// there, in place of `host_ratio` (as SGLang's `--hicache-size`); sized as `host_ratio`
    /// is, so a change of it is an epoch's.
    #[reload(epoch)]
    pub host_gib: Option<NonZeroU64>,
    /// Keep every page of the host KV cache in Mooncake, the KV storage,
    /// through the Mooncake connector of each host's Storage Server, and
    /// prefetch long hits back. Applies with `host_tier`; none when unset.
    #[reload(epoch)]
    pub mooncake: Option<storage::connector::Settings>,
    /// The pages of the largest block size between the aligned boundaries
    /// a long prompt offloads its sliding windows before
    /// ([`DEFAULT_BOUNDARY_INTERVAL`] unless stated).
    #[reload(epoch)]
    pub boundary_interval: NonZeroU32,
    /// The decoded tokens between two checkpoints of the state a model
    /// keeps per request slot, a recurrent state ([`DEFAULT_STATE_INTERVAL`]
    /// unless stated).
    #[reload(epoch)]
    pub state_interval: NonZeroU32,
    /// For a model that keeps a recurrent state, the bytes of its state
    /// region beyond one checkpoint per request slot over the bytes of the
    /// KV planes, which divide what a rank's share leaves between them;
    /// positive. [`DEFAULT_STATE_RATIO`] unless stated.
    #[reload(epoch)]
    pub state_ratio: f64,
    /// The time a copy of cached blocks the scheduler asks for may take
    /// whatever its size, before it gives the copy up and the request
    /// computes those tokens instead (as SGLang's `prefetch_timeout_base`):
    /// a prefetch from the remote store, a copy between the
    /// host and the device, and a stored trajectory's restore
    /// ([`DEFAULT_PREFETCH_TIMEOUT_BASE`] unless stated).
    #[serde(with = "values::duration")]
    #[reload(epoch)]
    pub prefetch_timeout_base: Duration,
    /// The time such a copy may take beyond `prefetch_timeout_base` for
    /// every 1024 tokens it moves (as SGLang's
    /// `prefetch_timeout_per_ki_token`)
    /// ([`DEFAULT_PREFETCH_TIMEOUT_PER_KI_TOKEN`] unless stated).
    #[serde(with = "values::duration")]
    #[reload(epoch)]
    pub prefetch_timeout_per_ki_token: Duration,
    /// How long a prefill replica keeps the KV of a request it handed off for the decode
    /// replica to import, at most: the decode replica's import returns it earlier
    /// ([`DEFAULT_LEASE`] unless stated).
    #[serde(with = "values::duration")]
    #[reload(epoch)]
    pub lease: Duration,
}

/// How a switch of a capability the epoch lays out storage for takes effect: switched off at the
/// schedulers' next step, which stop using it; switched on, or stated by a start, by a new epoch,
/// which lays the storage out.
pub fn switched_off(previous: Option<&bool>, next: &bool) -> crate::reload::Effect {
    match previous {
        Some(true) if !*next => crate::reload::Effect::Step,
        Some(previous) if previous == next => crate::reload::Effect::Step,
        _ => crate::reload::Effect::Epoch,
    }
}

/// How long a prefill replica keeps a handoff's KV for its import when the deployment states no
/// `kv_cache_config.lease`: thirty seconds, beyond the time a decode replica takes to admit a
/// request whose prompt another replica computed.
pub const DEFAULT_LEASE: Duration = Duration::from_secs(30);

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
            boundary_interval: DEFAULT_BOUNDARY_INTERVAL,
            state_interval: DEFAULT_STATE_INTERVAL,
            state_ratio: DEFAULT_STATE_RATIO,
            prefetch_timeout_base: DEFAULT_PREFETCH_TIMEOUT_BASE,
            prefetch_timeout_per_ki_token: DEFAULT_PREFETCH_TIMEOUT_PER_KI_TOKEN,
            lease: DEFAULT_LEASE,
        }
    }
}

impl KvCacheConfig {
    /// The tokens one KV page holds: the stated `block_size`, or [`DEFAULT_BLOCK_SIZE`].
    pub fn block_tokens(&self) -> u32 {
        self.block_size.map_or(DEFAULT_BLOCK_SIZE, NonZeroU32::get)
    }
}

/// The pages of the largest block size between two aligned boundaries a long prompt offloads its
/// sliding windows before, when the deployment states none (`kv_cache_config.boundary_interval`):
/// 64, a policy between the windows a long prompt stores to the host (one per interval) and the
/// part of an offloaded prompt a continuation of it reuses (whole intervals).
pub const DEFAULT_BOUNDARY_INTERVAL: NonZeroU32 = nonzero(64);

/// The decoded tokens between two checkpoints of a request's recurrent state, when the deployment
/// states none (`kv_cache_config.state_interval`): 256, as SGLang's `--mamba-track-interval`, a
/// policy between the checkpoints a decode stores (one per interval) and the decoded tokens a
/// resumed request recomputes (at most one interval).
pub const DEFAULT_STATE_INTERVAL: NonZeroU32 = nonzero(256);

/// The time a copy may take whatever its size, when the deployment states none
/// (`kv_cache_config.prefetch_timeout_base`): one second, SGLang's default.
pub const DEFAULT_PREFETCH_TIMEOUT_BASE: Duration = Duration::from_secs(1);

/// The time a copy may take for every 1024 tokens it moves, when the deployment states none
/// (`kv_cache_config.prefetch_timeout_per_ki_token`): a quarter second, SGLang's default.
pub const DEFAULT_PREFETCH_TIMEOUT_PER_KI_TOKEN: Duration = Duration::from_millis(250);

/// Default maximum calls selected by one generation scheduling pass.
pub const DEFAULT_MAX_BATCH: usize = 128;
/// Default token budget for one generation scheduling pass.
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;
/// Default maximum number of concurrently running requests.
pub const DEFAULT_MAX_RUNNING_REQUESTS: usize = 128;
/// The order a scheduling policy's waiting queue admits by, which its `order` argument states.
pub use scheduling::QueueOrder;

/// The installed scheduling program a deployment runs when its policy package lists none
/// (`uniserve:arrival`, `crates/scheduling/policies/arrival`).
pub const DEFAULT_SCHEDULING: &str = "arrival";

/// Admission and batching (`scheduler_config`), and the limits a scheduler runs with once the
/// engine resolved them.
///
/// The section states the waiting queue, the prefill chunk and the batch; the scheduling policy
/// is the policy package's scheduling program (`[programs.<name>]`). When the
/// engine starts a scheduler it resolves the rest from the configuration and the ranks' captured
/// catalog ([`SchedulerConfig::of`]): those fields are never read from a configuration file and
/// never written to one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, crate::reload::Reload)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Waiting requests, and finished ones with undelivered events, before
    /// a submission is refused as overloaded; unbounded when unset.
    #[reload(step)]
    pub max_queued_requests: Option<NonZeroUsize>,
    /// Prompt tokens one request prefills in one step, the scheduling policy's
    /// `chunked_prefill_size` argument; unset, the policy package's manifest declares it
    /// (`[parameters]`), which is its one source. Token models only.
    #[reload(step)]
    pub chunked_prefill_size: Option<NonZeroUsize>,
    /// Calls assembled into one forward batch; once resolved, the largest bucket of the ranks'
    /// catalog, which is what the workers accept.
    #[reload(step)]
    pub max_batch: NonZeroUsize,
    /// Token budget of one generation scheduling pass, resolved: the configuration's
    /// `max_num_batched_tokens`, or what the ranks' catalog bounds a step to. Each selected call is
    /// charged its token work, which for a denoising call covers its latent units and guidance
    /// branches.
    #[serde(skip)]
    #[reload(epoch)]
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently running requests, token and media together, resolved from the
    /// configuration's `max_running_requests`.
    #[serde(skip)]
    #[reload(epoch)]
    pub max_running_requests: usize,
    /// The most draft tokens a verification carries per request
    /// (`speculative_config.num_draft_tokens`); zero when the deployment does not speculate.
    #[serde(skip)]
    #[reload(generation)]
    pub draft_tokens: usize,
    /// The captured graph sizes the ranks state, in tokens, ascending and without duplicates:
    /// `Room.graph_tiers`, the bound a scheduled batch must fit. Empty when they state none, and on a
    /// reconfigure, which carries no graph catalog: a worker then runs its own captured catalog or
    /// eagerly, and the tier bound is honest but vacuous.
    #[serde(skip)]
    #[reload(generation)]
    pub graph_tiers: Vec<u32>,
    /// What the scheduling policy module runs within: the fuel of each call into it, and the
    /// bounds of the starvation timeout and the prefetch threshold its limits may declare.
    #[reload(program)]
    pub policy_budget: PolicyBudget,
    /// The tokens of the deployment's KV page, resolved from `kv_cache_config.block_size`
    /// ([`KvCacheConfig::block_tokens`]): the block a class whose blocks hold no tokens is counted
    /// in.
    #[serde(skip)]
    #[reload(epoch)]
    pub block_tokens: u32,
    /// The time a copy may take whatever its size, resolved from
    /// `kv_cache_config.prefetch_timeout_base`.
    #[serde(skip)]
    #[reload(epoch)]
    pub prefetch_timeout_base: Duration,
    /// The time a copy may take for every 1024 tokens it moves, resolved from
    /// `kv_cache_config.prefetch_timeout_per_ki_token`.
    #[serde(skip)]
    #[reload(epoch)]
    pub prefetch_timeout_per_ki_token: Duration,
    /// How often the scheduler reports its statistics, resolved from
    /// `worker_config.report_period`, which each step's snapshot carries ([`Tuning`]).
    #[serde(skip)]
    #[reload(step)]
    pub report_period: Duration,
    /// The role of each scheduler of the engine, by its index, resolved from the
    /// configuration's `replicas.<id>.role`: which requests each takes.
    #[serde(skip)]
    #[reload(step)]
    pub roles: Vec<Role>,
    /// How long a handoff's lend lasts at most, resolved from `kv_cache_config.lease`.
    #[serde(skip)]
    #[reload(epoch)]
    pub lease: Duration,
    /// The TCP endpoint of the Storage Server of the scheduler's host, which a handoff's lend
    /// names; `None` for a server that listens on none. Resolved from the engine's launch.
    #[serde(skip)]
    #[reload(epoch)]
    pub storage: Option<String>,
}

/// The requests a replica takes (`replicas.<id>.role`).
pub use scheduling::routing::Role;

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
            max_queued_requests: None,
            chunked_prefill_size: None,
            max_batch: positive(DEFAULT_MAX_BATCH),
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_running_requests: DEFAULT_MAX_RUNNING_REQUESTS,
            draft_tokens: 0,
            graph_tiers: Vec::new(),
            policy_budget: PolicyBudget::default(),
            block_tokens: DEFAULT_BLOCK_SIZE,
            prefetch_timeout_base: DEFAULT_PREFETCH_TIMEOUT_BASE,
            prefetch_timeout_per_ki_token: DEFAULT_PREFETCH_TIMEOUT_PER_KI_TOKEN,
            report_period: DEFAULT_REPORT_PERIOD,
            roles: Vec::new(),
            lease: DEFAULT_LEASE,
            storage: None,
        }
    }
}

impl SchedulerConfig {
    /// The limits of a scheduler of `engine` whose ranks' captured catalog bounds a step to
    /// `max_batch` calls and `max_num_batched_tokens` tokens, `graph_tiers` being the catalog's
    /// sizes in tokens, ascending (§4.4, §4.5).
    ///
    /// The rest is the configuration's: its running requests, its waiting queue and the prefill
    /// chunk it states (the policy package's own when it states none), the drafter's tokens, the
    /// KV cache's copy timeouts and the workers' report period.
    pub fn of(
        engine: &EngineConfig,
        max_batch: usize,
        max_num_batched_tokens: usize,
        graph_tiers: Vec<u32>,
    ) -> Self {
        Self {
            max_queued_requests: engine.scheduler.max_queued_requests,
            chunked_prefill_size: engine.scheduler.chunked_prefill_size,
            max_batch: positive(max_batch),
            max_num_batched_tokens,
            max_running_requests: engine.max_running_requests.get(),
            draft_tokens: engine.speculative.num_draft_tokens as usize,
            graph_tiers,
            policy_budget: engine.scheduler.policy_budget,
            block_tokens: engine.kv_cache.block_tokens(),
            prefetch_timeout_base: engine.kv_cache.prefetch_timeout_base,
            prefetch_timeout_per_ki_token: engine.kv_cache.prefetch_timeout_per_ki_token,
            report_period: engine.worker.report_period,
            roles: engine.roles.clone(),
            lease: engine.kv_cache.lease,
            storage: engine.scheduler.storage.clone(),
        }
    }

    /// How long a copy of `tokens` tokens may take before the scheduler gives it up:
    /// `prefetch_timeout_base`, and `prefetch_timeout_per_ki_token` for every 1024 tokens.
    pub fn copy_timeout(&self, tokens: u32) -> Duration {
        let share = self.prefetch_timeout_per_ki_token.saturating_mul(tokens) / 1024;
        self.prefetch_timeout_base.saturating_add(share)
    }
}

/// The options an engine reads at its next step (`specs/configuration.md` §7, step values), with
/// the configuration revision they belong to: the schedulers' limits, and what every main process
/// of the epoch reads beside its batches (its watchdog's bound, its report period, its
/// comparison bound, its NVTX ranges).
///
/// An engine keeps the current values in one snapshot its schedulers share; each step takes the
/// snapshot once when it begins and uses it for the whole step, and every batch it assembles
/// names the snapshot's revision and carries the values its ranks read with it (the copy
/// budget), so all ranks of a batch read one configuration. A change of them replaces the
/// snapshot (the engine command `Tune`), which the scheduler node relays to every peer, and
/// takes effect at each scheduler's next step and each process's next turn of its control
/// loop, without a new policy instance or a new process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tuning {
    /// The configuration revision the values are of.
    pub revision: u64,
    /// Prompt tokens one request prefills in one step; the step's tokens when unstated.
    pub chunked_prefill_size: Option<NonZeroUsize>,
    /// Calls one batch assembles at most.
    pub max_batch: NonZeroUsize,
    /// Token work one step admits; the widest captured prefill bucket's, else the longest
    /// sequence, when unstated.
    pub max_num_batched_tokens: Option<NonZeroUsize>,
    /// Waiting requests before a submission is refused as overloaded; unbounded when unstated.
    pub max_queued_requests: Option<NonZeroUsize>,
    /// Batches each worker group keeps in flight, by the group's id
    /// (`device_config.queue_depth`, or the group's own `queue_depth`).
    pub queue_depths: BTreeMap<String, NonZeroUsize>,
    /// Whether the schedulers read and write cached prefixes (`kv_cache_config.prefix_caching`).
    pub prefix_caching: bool,
    /// Whether the schedulers offload pages to the host tier (`kv_cache_config.host_tier`).
    pub host_tier: bool,
    /// The bytes the copies started with one batch move together over a group's ranks
    /// (`worker_config.copy_budget`).
    pub copy_budget: u64,
    /// How often each process and scheduler reports (`worker_config.report_period`).
    pub report_period: Duration,
    /// How long a job may stay in flight before it is reported stalled
    /// (`worker_config.watchdog_timeout`).
    pub watchdog_timeout: Duration,
    /// How long a generation update's shadow pass and a peer's own preparation are waited for
    /// (`worker_config.compare_timeout`).
    pub compare_timeout: Duration,
    /// Whether the processes emit their NVTX ranges (`profiling_config.nvtx_ranges`).
    pub nvtx_ranges: bool,
    /// Where and how the schedulers publish their KV events (`kv_events_config`).
    #[serde(default)]
    pub kv_events: KvEventsConfig,
    /// The role of each of the engine's schedulers, by its index; a scheduler past them is
    /// mixed.
    pub roles: Vec<Role>,
}

impl Tuning {
    /// The role of the engine's scheduler `scheduler`: mixed unless the values state another.
    pub fn role(&self, scheduler: usize) -> Role {
        self.roles.get(scheduler).copied().unwrap_or(Role::Mixed)
    }

    /// The step values `config` states, as of `revision`.
    pub fn of(config: &EngineConfig, revision: u64) -> Self {
        let queue_depths = config
            .parallel
            .worker_groups
            .iter()
            .map(|group| (group.id.0.clone(), positive(group.queue_depth.max(1))))
            .collect();
        Self {
            revision,
            chunked_prefill_size: config.scheduler.chunked_prefill_size,
            max_batch: config.scheduler.max_batch,
            max_num_batched_tokens: config.max_num_batched_tokens,
            max_queued_requests: config.scheduler.max_queued_requests,
            queue_depths,
            prefix_caching: config.kv_cache.prefix_caching,
            host_tier: config.kv_cache.host_tier,
            copy_budget: config.worker.copy_budget,
            report_period: config.worker.report_period,
            watchdog_timeout: config.worker.watchdog_timeout,
            compare_timeout: config.worker.compare_timeout,
            nvtx_ranges: config.profiling.nvtx_ranges,
            kv_events: config.kv_events.clone(),
            roles: config.roles.clone(),
        }
    }

    /// The engine command that delivers these values, numbered `id`.
    pub fn command(&self, id: u64) -> crate::request::wire::Tune {
        let count = |value: NonZeroUsize| u64::try_from(value.get()).unwrap_or(u64::MAX);
        let millis = |value: Duration| u64::try_from(value.as_millis()).unwrap_or(u64::MAX);
        crate::request::wire::Tune {
            id,
            revision: self.revision,
            chunked_prefill_size: self.chunked_prefill_size.map(count),
            max_batch: count(self.max_batch),
            max_num_batched_tokens: self.max_num_batched_tokens.map(count),
            max_queued_requests: self.max_queued_requests.map(count),
            queue_depths: self
                .queue_depths
                .iter()
                .map(|(group, depth)| (group.clone(), count(*depth)))
                .collect(),
            prefix_caching: self.prefix_caching,
            host_tier: self.host_tier,
            copy_budget: self.copy_budget,
            report_period_ms: millis(self.report_period),
            watchdog_timeout_ms: millis(self.watchdog_timeout),
            compare_timeout_ms: millis(self.compare_timeout),
            nvtx_ranges: self.nvtx_ranges,
            kv_events_endpoint: self.kv_events.endpoint.clone(),
            kv_events_replay_endpoint: self.kv_events.replay_endpoint.clone(),
            kv_events_topic: self.kv_events.topic.clone(),
            kv_events_buffer_steps: count(self.kv_events.buffer_steps),
            roles: self
                .roles
                .iter()
                .map(|role| crate::request::wire::Role::from(*role) as i32)
                .collect(),
        }
    }

    /// The values a `Tune` command delivers; `None` for one whose batch, or a group's depth, is
    /// zero calls, which no deployment states.
    pub fn of_command(command: &crate::request::wire::Tune) -> Option<Self> {
        let count = |value: u64| usize::try_from(value).ok().and_then(NonZeroUsize::new);
        let queue_depths = command
            .queue_depths
            .iter()
            .map(|(group, depth)| Some((group.clone(), count(*depth)?)))
            .collect::<Option<_>>()?;
        Some(Self {
            revision: command.revision,
            chunked_prefill_size: command.chunked_prefill_size.and_then(count),
            max_batch: count(command.max_batch)?,
            max_num_batched_tokens: command.max_num_batched_tokens.and_then(count),
            max_queued_requests: command.max_queued_requests.and_then(count),
            queue_depths,
            prefix_caching: command.prefix_caching,
            host_tier: command.host_tier,
            copy_budget: command.copy_budget,
            report_period: Duration::from_millis(command.report_period_ms),
            watchdog_timeout: Duration::from_millis(command.watchdog_timeout_ms),
            compare_timeout: Duration::from_millis(command.compare_timeout_ms),
            nvtx_ranges: command.nvtx_ranges,
            kv_events: KvEventsConfig {
                endpoint: command.kv_events_endpoint.clone(),
                replay_endpoint: command.kv_events_replay_endpoint.clone(),
                topic: command.kv_events_topic.clone(),
                buffer_steps: count(command.kv_events_buffer_steps)?,
            },
            roles: command
                .roles
                .iter()
                .map(|role| crate::request::wire::Role::try_from(*role).map(Role::from))
                .collect::<Result<_, _>>()
                .ok()?,
        })
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
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, crate::reload::Reload)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculativeConfig {
    /// The most draft tokens one verification carries per request; zero
    /// does not speculate. A change is a kernel generation, whose graphs the
    /// serving engines capture in the background: the verification buckets
    /// hold the drafts, the generation loads or releases the drafter's
    /// weights beside the serving ones, and a model that keeps a recurrent
    /// state lays it out for the drafts beside the serving one. The draft
    /// cache is laid out whatever the drafts.
    #[reload(generation)]
    pub num_draft_tokens: u32,
    /// The speedup speculation gives a request, the decode steps' time one
    /// of its verification steps takes the place of, stated in place of
    /// the one the epoch measures; at least one, measured when unset.
    #[reload(program)]
    pub expected_speedup: Option<f64>,
    /// The time of evaluating one context token in a refill, in steps of
    /// decoding, stated in place of the one the epoch measures; not
    /// negative, measured when unset.
    #[reload(program)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, crate::reload::Reload)]
#[serde(default, deny_unknown_fields)]
pub struct DeviceConfig {
    /// Device kind of every rank; `cuda` numbers each host's GPUs from zero.
    #[reload(epoch)]
    pub device: DeviceKind,
    /// Each rank's CUDA device index on its host, in rank order; empty
    /// numbers each host's ranks from zero. A change of the parallel
    /// configuration that does not state it takes the order that leaves
    /// the most weights on the GPUs already holding them.
    #[reload(epoch)]
    pub devices: Vec<u32>,
    /// Hosts after this one the ranks spread across, in equal blocks.
    #[reload(epoch)]
    pub worker_hosts: Vec<String>,
    /// Batches in flight per rank; a deployment group states its own. It holds no storage: a
    /// scheduler keeps this many of a group's batches in flight from its next step on, and one
    /// that holds more lets them end first.
    #[reload(step)]
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

/// Mixture-of-experts parallelism (`moe_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, crate::reload::Reload)]
#[serde(default, deny_unknown_fields)]
pub struct MoeConfig {
    /// Shard routed experts across the deployment's expert workers, across
    /// schedulers of one rank each, or otherwise over the tensor-parallel
    /// ranks of each scheduler's worker group, each rank holding whole
    /// experts ([`EngineConfig::expert_union`]).
    #[reload(epoch)]
    pub expert_parallel: bool,
    /// Microbatches overlapping attention with dedicated expert workers. The microbatches shape
    /// the graphs an expert union captures, so a change is a kernel generation: the serving
    /// engines capture the buckets it changes in the background and switch bucket by bucket.
    #[reload(generation)]
    pub microbatches: NonZeroU32,
    /// The rows per rank of the middle capture class of an expert union's steps: a step in which
    /// another member prefills a few prompts while this one decodes, which neither the decode
    /// class nor the class of the union's whole capacity fits well. Not captured when the decode
    /// class reaches it or it is within a quarter of the whole capacity
    /// ([`DEFAULT_BESIDE_ROWS`] unless stated).
    #[reload(epoch)]
    pub beside_rows: NonZeroU32,
    /// The experts' order: for each routed expert layer, by its module path, the logical expert
    /// ids in the order of the physical slots that hold them, across the layer's expert ranks; a
    /// layer it does not name keeps the checkpoint's order. A change is prepared on the copy
    /// stream beside the batches and committed at one computation boundary: the epoch, the
    /// graphs' addresses and the requests stay. An epoch starts with the order stated.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    #[reload(order)]
    pub expert_order: BTreeMap<String, Vec<u32>>,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            expert_parallel: false,
            microbatches: NonZeroU32::MIN,
            beside_rows: DEFAULT_BESIDE_ROWS,
            expert_order: BTreeMap::new(),
        }
    }
}

/// The rows per rank of an expert union's middle capture class when the deployment states none
/// (`moe_config.beside_rows`): 1024, the steps of a few prompts' prefill beside the other
/// members' decodes.
pub const DEFAULT_BESIDE_ROWS: NonZeroU32 = nonzero(1024);

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

/// The options a kernel generation captures the engines' CUDA graphs for: the bucket catalog
/// (`cuda_graph_config`), the candidates every sampling graph computes for a scored row
/// (`max_logprobs`), the drafts a verification holds at most
/// (`speculative_config.num_draft_tokens`), and whether CUPTI samples the program counters of
/// the graphs, which it attributes only in graphs captured while it samples
/// (`profiling_config.graph_pc_sampling`). A generation whose options differ from the serving
/// one's captures every graph again, for the shapes they give the engines' runners
/// ([`EngineConfig::graph_options`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphOptions {
    pub cuda_graph: CudaGraphConfig,
    pub max_logprobs: u32,
    pub num_draft_tokens: u32,
    pub graph_pc_sampling: bool,
}

/// The sizes of a wire catalog: `sizes` when `stated`, every one nonzero, and the worker's own
/// catalog otherwise.
fn stated_sizes(
    sizes: Vec<u32>,
    stated: bool,
    what: &str,
) -> Result<Option<Vec<NonZeroU32>>, String> {
    if !stated {
        return Ok(None);
    }
    sizes
        .into_iter()
        .map(|size| NonZeroU32::new(size).ok_or_else(|| format!("a {what} of zero")))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// The wire catalog of `sizes`: the sizes, and whether the catalog states them.
fn wire_sizes(sizes: Option<&Vec<NonZeroU32>>) -> (Vec<u32>, bool) {
    match sizes {
        Some(sizes) => (sizes.iter().map(|size| size.get()).collect(), true),
        None => (Vec::new(), false),
    }
}

impl From<&GraphOptions> for crate::process::GraphOptions {
    fn from(options: &GraphOptions) -> Self {
        let (decode_batch_sizes, decode_stated) =
            wire_sizes(options.cuda_graph.decode_batch_sizes.as_ref());
        let (prefill_token_sizes, prefill_stated) =
            wire_sizes(options.cuda_graph.prefill_token_sizes.as_ref());
        Self {
            decode_batch_sizes,
            decode_stated,
            prefill_token_sizes,
            prefill_stated,
            max_logprobs: options.max_logprobs,
            num_draft_tokens: options.num_draft_tokens,
            graph_pc_sampling: options.graph_pc_sampling,
        }
    }
}

impl TryFrom<crate::process::GraphOptions> for GraphOptions {
    type Error = String;

    /// The options a `Prepare` states; a catalog of a size of zero is refused.
    fn try_from(wire: crate::process::GraphOptions) -> Result<Self, String> {
        Ok(Self {
            cuda_graph: CudaGraphConfig {
                decode_batch_sizes: stated_sizes(
                    wire.decode_batch_sizes,
                    wire.decode_stated,
                    "decode batch size",
                )?,
                prefill_token_sizes: stated_sizes(
                    wire.prefill_token_sizes,
                    wire.prefill_stated,
                    "prefill token size",
                )?,
            },
            max_logprobs: wire.max_logprobs,
            num_draft_tokens: wire.num_draft_tokens,
            graph_pc_sampling: wire.graph_pc_sampling,
        })
    }
}

impl From<&GraphOptions> for crate::request::wire::GraphOptions {
    fn from(options: &GraphOptions) -> Self {
        let process = crate::process::GraphOptions::from(options);
        Self {
            decode_batch_sizes: process.decode_batch_sizes,
            decode_stated: process.decode_stated,
            prefill_token_sizes: process.prefill_token_sizes,
            prefill_stated: process.prefill_stated,
            max_logprobs: process.max_logprobs,
            num_draft_tokens: process.num_draft_tokens,
            graph_pc_sampling: process.graph_pc_sampling,
        }
    }
}

impl TryFrom<crate::request::wire::GraphOptions> for GraphOptions {
    type Error = String;

    /// The options a `MeasureRoom` states, as a `Prepare`'s are read.
    fn try_from(wire: crate::request::wire::GraphOptions) -> Result<Self, String> {
        Self::try_from(crate::process::GraphOptions {
            decode_batch_sizes: wire.decode_batch_sizes,
            decode_stated: wire.decode_stated,
            prefill_token_sizes: wire.prefill_token_sizes,
            prefill_stated: wire.prefill_stated,
            max_logprobs: wire.max_logprobs,
            num_draft_tokens: wire.num_draft_tokens,
            graph_pc_sampling: wire.graph_pc_sampling,
        })
    }
}

/// The worker processes (`worker_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, crate::reload::Reload)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerConfig {
    /// The engine package the deployment's main processes run as: the name of an implementation
    /// of the engine interface installed at the Build Server.
    #[reload(epoch)]
    pub engine: String,
    /// The revision of the engine package to run. Unset, a start fixes the latest installed
    /// one, and every later apply keeps the serving epoch's, so a revision installed later runs
    /// only once a configuration names it. Its change, compared as the deployment runs it, is
    /// an epoch: stating the serving revision changes nothing.
    #[reload(epoch)]
    pub engine_revision: Option<u32>,
    /// The processing program the leader daemon serves the epoch's requests through: the name
    /// of a program package installed at the Build Server, which a source tree installs beside
    /// its engine (`crates/processing/package.toml`).
    #[reload(epoch)]
    pub processing: String,
    /// The revision of the processing program to run, unset as `engine_revision` is: a start
    /// fixes the latest installed one, and every later apply keeps the serving epoch's. A tree
    /// installs its processing program and its engine together, so the latest revisions of both
    /// are one tree's, and an epoch keeps both of its tree.
    #[reload(epoch)]
    pub processing_revision: Option<u32>,
    /// Time a job of a rank's worker may stay in flight before it is reported `Stalled`, which
    /// fails the epoch: its worker group is replaced, and an update on trial rolls back. Every
    /// main process's watchdog takes a change at its engine's next step values ([`Tuning`]).
    #[serde(with = "values::duration")]
    #[reload(step)]
    pub watchdog_timeout: Duration,
    /// How long the main processes of an epoch whose ranks span several hosts
    /// wait for one another as they start: the scheduler node for every peer's
    /// registration, a peer for the scheduler node's acceptance, and every
    /// process for the others at the ranks' host-group rendezvous. A process
    /// that is connected and then exits or loses its host fails the start at
    /// once; the bound covers a process that never arrives. Only a start reads it, so a change
    /// moves no running process: the next start of a process (a relaunch, a standby, another
    /// epoch) waits as long as it states.
    #[serde(with = "values::duration")]
    #[reload(step)]
    pub join_timeout: Duration,
    /// Time a generation update's shadow pass waits for the running requests of every scheduler
    /// to run a step its changed kernels are compared on, and for every peer's own preparation of
    /// the generation to end, before the update is abandoned as uncompared. A change bounds the
    /// next preparation ([`Tuning`]).
    #[serde(with = "values::duration")]
    #[reload(step)]
    pub compare_timeout: Duration,
    /// `tower=gen:<device>` places the generation tower on another device
    /// than the rank's.
    #[reload(epoch)]
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
    #[reload(epoch)]
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
    #[reload(epoch)]
    pub standby_storage: StandbyStorage,
    /// How often each main process states its metrics and each scheduler its statistics to the
    /// daemon, each report covering the period since the last ([`DEFAULT_REPORT_PERIOD`] unless
    /// stated). A change takes effect with the engine's next step values ([`Tuning`]).
    #[serde(with = "values::duration")]
    #[reload(step)]
    pub report_period: Duration,
    /// The bytes, summed over a worker group's ranks, that the copies started with one batch may
    /// move together beside its forward; a larger copy starts alone ([`DEFAULT_COPY_BUDGET`]
    /// unless stated). Each batch carries the budget of the step that assembled it, so every
    /// rank of a group starts the same copies with it, from the next step on.
    #[serde(with = "values::bytes")]
    #[reload(step)]
    pub copy_budget: u64,
    /// The threads of a main process's host work: the width of its media pool, each with
    /// `host_queue_depth` tasks of room, and the threads a conversion of a request's condition
    /// media shares its work over. Half the CPUs the process may run on (its affinity mask within
    /// its cgroup's quota) when unset, the other half left to its workers', schedulers' and
    /// communicators' threads.
    #[reload(epoch)]
    pub host_threads: Option<NonZeroUsize>,
    /// The tasks of host work one thread of a main process's media pool has room for: the
    /// requests a host phase serves at once are the threads times it
    /// ([`DEFAULT_HOST_QUEUE_DEPTH`] unless stated).
    #[reload(epoch)]
    pub host_queue_depth: NonZeroUsize,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            engine: DEFAULT_ENGINE.to_owned(),
            engine_revision: None,
            processing: DEFAULT_PROCESSING.to_owned(),
            processing_revision: None,
            watchdog_timeout: DEFAULT_WATCHDOG_TIMEOUT,
            join_timeout: DEFAULT_JOIN_TIMEOUT,
            compare_timeout: DEFAULT_COMPARE_TIMEOUT,
            mesh: None,
            standby: false,
            standby_storage: StandbyStorage::Auto,
            report_period: DEFAULT_REPORT_PERIOD,
            copy_budget: DEFAULT_COPY_BUDGET,
            host_threads: None,
            host_queue_depth: positive(DEFAULT_HOST_QUEUE_DEPTH),
        }
    }
}

/// How often the engine's processes and schedulers report, when the deployment states no period
/// (`worker_config.report_period`): a second, the interval of the daemon's metrics and of the
/// step times the interference guard compares (`update_config.transfer_slowdown`).
pub const DEFAULT_REPORT_PERIOD: Duration = Duration::from_secs(1);

/// The bytes a batch's copies may move together, summed over a worker group's ranks, when the
/// deployment states no budget (`worker_config.copy_budget`): 512 MiB. Host storage fed a GB200
/// GPU at 209 GB/s in the Transfer Server's interference measurement (`docs/configuration.md`
/// under `transfer_slowdown`), so the budget is at most 2.6 ms of copying on one device of that
/// host, an eighth of the measurement's 20.5 ms decode step, in which copies from host storage at
/// a quarter of the link's bandwidth slowed the steps by 2% and at a sixteenth by none. A host
/// of another link bandwidth or step time states its own.
pub const DEFAULT_COPY_BUDGET: u64 = 512 << 20;

/// The tasks one thread of the media pool has room for when the deployment states none
/// (`worker_config.host_queue_depth`): four, a pipelining depth that keeps tasks queued behind
/// the one each thread runs, so a thread takes its next task without waiting for a step.
pub const DEFAULT_HOST_QUEUE_DEPTH: usize = 4;

/// How long a job may stay in flight before it is reported stalled and its worker group is
/// replaced, when the deployment states no bound (`worker_config.watchdog_timeout`): two and a half
/// minutes, a policy bound above the longest step a served model takes (a denoising step of the
/// longest video, a prefill of the longest prompt) and below what an operator waits before
/// replacing a stuck group by hand.
pub const DEFAULT_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(150);

/// How long a generation update's shadow pass waits for a step of the running requests, when the
/// deployment states no bound (`worker_config.compare_timeout`): five minutes, long enough for an
/// engine whose requests arrive at a low rate to run a step of each scheduler, and short enough
/// that an update on an idle engine is abandoned within the time an operator waits for it.
pub const DEFAULT_COMPARE_TIMEOUT: Duration = Duration::from_secs(300);

/// How long the main processes of a multi-host epoch wait for one another as they start, when the
/// deployment states no bound (`worker_config.join_timeout`): five minutes, a policy bound above
/// the skew between the hosts' spawns of one epoch's processes (the daemons spawn them together
/// once every host prepared the epoch's modules, and each process registers before it loads
/// anything) and below what an operator waits before replacing a start by hand.
pub const DEFAULT_JOIN_TIMEOUT: Duration = Duration::from_secs(300);

/// The engine package a deployment runs when its configuration names none: the implementation
/// of the engine interface this tree holds (`crates/engine/package.toml`).
pub const DEFAULT_ENGINE: &str = "uniserve";

/// The processing program a deployment serves its requests through when its configuration names
/// none: the one this tree holds (`crates/processing/package.toml`).
pub const DEFAULT_PROCESSING: &str = "processing";

/// What the server logs about its serving (`logging_config`). A process's
/// log levels are options of the binary that runs it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    /// Log a summary line for each completed request.
    pub log_requests: bool,
    /// Log engine statistics periodically.
    pub log_stats: bool,
    /// Interval of the statistics log line: the throughput it states is
    /// averaged over it.
    #[serde(with = "values::duration")]
    pub stats_interval: Duration,
}

/// The interval of the statistics log line when the deployment states none
/// (`logging_config.stats_interval`): an operator's reading cadence, long
/// enough that each line averages many scheduler reports.
pub const STATS_LOG_INTERVAL: Duration = Duration::from_secs(10);

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_requests: false,
            log_stats: true,
            stats_interval: STATS_LOG_INTERVAL,
        }
    }
}

/// Which reasoning parser splits model output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningParser {
    /// The model description's parser splits `reasoning` from
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
    /// The chat template every chat request renders with, in place of the checkpoint's own.
    pub template: Option<ChatTemplateSource>,
    /// Keyword arguments bound into every chat-template render.
    pub template_kwargs: Option<HashMap<String, Value>>,
    /// How message content renders: `auto`, `string` or `openai`.
    #[serde(with = "content_format")]
    pub template_content_format: Option<ChatTemplateContentFormat>,
    /// `auto` splits `reasoning` from `content`; `none` streams
    /// reasoning delimiters as content.
    pub reasoning_parser: ReasoningParser,
}

/// Where a configured chat template's Jinja source comes from (`chat_config.template`): a file,
/// `{ file = "/srv/templates/chat.jinja" }`, or the source itself,
/// `{ inline = "{{ messages[0].content }}" }`. Either is the template exactly as written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ChatTemplateSource {
    /// A file holding the template's Jinja source.
    File(PathBuf),
    /// The template's Jinja source.
    Inline(String),
}

/// The smallest prompt capacity, in tokens, a media deployment captures when
/// `video_config.text_capacities` is unset: a short prompt encodes in a graph of at most this
/// many tokens.
pub const DEFAULT_FIRST_TEXT_CAPACITY: u32 = 1024;

/// The tokens between two consecutive prompt capacities a media deployment captures when
/// `video_config.text_capacities` is unset: a prompt pads to at most this many tokens past its
/// own, and each capacity costs one captured encode per request count.
pub const DEFAULT_TEXT_CAPACITY_STEP: u32 = 2048;

/// The bytes of one page of a denoising rank's host latent region, where a preempted request's
/// trajectory is saved, when `video_config.latent_page_bytes` is unset: a mebibyte, a whole
/// number of float32 samples. A saved trajectory takes whole pages, so a page bounds what a save
/// copies past the trajectory, and the region's pages are what its numbering counts.
pub const DEFAULT_LATENT_PAGE_BYTES: u32 = 1 << 20;

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
    /// `max_model_len`, which completes them as the largest;
    /// [`DEFAULT_FIRST_TEXT_CAPACITY`], then steps of [`DEFAULT_TEXT_CAPACITY_STEP`] below
    /// `max_model_len`, when unset.
    pub text_capacities: Option<Vec<NonZeroU32>>,
    /// Denoiser rows one request's conditions may take; the largest
    /// condition set the denoiser admits when unset.
    pub max_condition_rows: Option<u32>,
    /// The bytes of one page of a denoising rank's host latent region, a whole number of
    /// float32 samples; [`DEFAULT_LATENT_PAGE_BYTES`] when unset.
    pub latent_page_bytes: Option<NonZeroU32>,
}

impl VideoConfig {
    /// The prompt capacities the configuration states, or when it states none the default
    /// ladder below `longest`: [`DEFAULT_FIRST_TEXT_CAPACITY`], then steps of
    /// [`DEFAULT_TEXT_CAPACITY_STEP`]. Neither is completed with `longest` here.
    pub fn stated_text_capacities(&self, longest: u32) -> Vec<u32> {
        match &self.text_capacities {
            Some(capacities) => capacities.iter().map(|value| value.get()).collect(),
            None => std::iter::successors(Some(DEFAULT_FIRST_TEXT_CAPACITY), |capacity| {
                capacity.checked_add(DEFAULT_TEXT_CAPACITY_STEP)
            })
            .take_while(|capacity| *capacity < longest)
            .collect(),
        }
    }

    /// The bytes of one page of the host latent region: the stated ones, or
    /// [`DEFAULT_LATENT_PAGE_BYTES`].
    pub fn latent_page_bytes(&self) -> u32 {
        self.latent_page_bytes
            .map_or(DEFAULT_LATENT_PAGE_BYTES, NonZeroU32::get)
    }
}

/// Where a request's media may come from, its size and time limits, and how it is probed: the one
/// schema of the two media tables, input images of chat requests (`image_fetch_config`,
/// [`ImageFetchConfig`]) and condition media of video requests (`media_config`,
/// [`MediaConfig`]). A table's options are the same; what each leaves unset defaults to its own
/// values (`D`, [`MediaDefaults`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct MediaSources<D: MediaDefaults> {
    /// Directory `file://` media resolves under; `file://` is refused when unset. The directory
    /// is named by its canonical path, which `file://` URLs name it by.
    #[serde(default)]
    pub directory: Option<PathBuf>,
    /// Fetch `http(s)://` media.
    #[serde(default = "media::allow_remote")]
    pub allow_remote: bool,
    /// Allow `http(s)` URLs resolving to loopback, private, link-local, unique-local or cloud
    /// metadata addresses.
    #[serde(default)]
    pub allow_private: bool,
    /// Time to fetch one `http(s)` medium, redirects and body included.
    #[serde(default = "D::timeout", with = "values::duration")]
    pub timeout: Duration,
    /// Time to establish one connection.
    #[serde(default = "media::connect_timeout", with = "values::duration")]
    pub connect_timeout: Duration,
    /// Redirects one fetch follows before it fails.
    #[serde(default = "media::max_redirects")]
    pub max_redirects: u32,
    /// `http(s)` downloads in progress at once; further fetches wait for one to end. Each holds
    /// at most `max_bytes`, so the two bound the storage downloads hold.
    #[serde(default = "media::max_concurrent_fetches")]
    pub max_concurrent_fetches: NonZeroUsize,
    /// Largest medium, fetched, read or `data:`.
    #[serde(default = "D::max_bytes", with = "values::bytes")]
    pub max_bytes: u64,
    /// Media one request may carry in total; unbounded when unset.
    #[serde(default = "D::max_request_bytes", with = "values::optional_bytes")]
    pub max_request_bytes: Option<u64>,
    /// Time one medium's probe may take: its demuxer reading the streams of an audio or video
    /// file. An image's probe reads its header alone.
    #[serde(default = "media::probe_timeout", with = "values::duration")]
    pub probe_timeout: Duration,
    /// Media probed at once across requests; the daemon's available parallelism (its CPU set
    /// and quota) when unset.
    #[serde(default)]
    pub max_concurrent_probes: Option<NonZeroUsize>,
    /// The table's defaults.
    #[serde(skip)]
    pub defaults: std::marker::PhantomData<D>,
}

impl<D: MediaDefaults> Default for MediaSources<D> {
    fn default() -> Self {
        Self {
            directory: None,
            allow_remote: media::allow_remote(),
            allow_private: false,
            timeout: D::timeout(),
            connect_timeout: media::connect_timeout(),
            max_redirects: media::max_redirects(),
            max_concurrent_fetches: media::max_concurrent_fetches(),
            max_bytes: D::max_bytes(),
            max_request_bytes: D::max_request_bytes(),
            probe_timeout: media::probe_timeout(),
            max_concurrent_probes: None,
            defaults: std::marker::PhantomData,
        }
    }
}

/// The values a media table's options take when the deployment states none, where the tables
/// differ: each is an admission policy of its kind of media.
pub trait MediaDefaults {
    /// Time to fetch one `http(s)` medium.
    fn timeout() -> Duration;
    /// Largest medium, in bytes.
    fn max_bytes() -> u64;
    /// Media one request may carry in total, in bytes; unbounded for `None`.
    fn max_request_bytes() -> Option<u64>;
}

/// The defaults of input images of chat requests (`image_fetch_config`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageMedia;

impl MediaDefaults for ImageMedia {
    /// 20 seconds: a request-admission bound on one remote fetch, redirects and body included.
    fn timeout() -> Duration {
        Duration::from_secs(20)
    }

    /// 20 MB, the per-image limit OpenAI's API states for image inputs.
    fn max_bytes() -> u64 {
        20_000_000
    }

    /// A chat request's images are bounded each, not in total.
    fn max_request_bytes() -> Option<u64> {
        None
    }
}

/// The defaults of condition media of video requests (`media_config`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConditionMedia;

impl MediaDefaults for ConditionMedia {
    /// 60 seconds: a reference video takes longer to fetch than an image.
    fn timeout() -> Duration {
        Duration::from_secs(60)
    }

    /// 256 MiB, the request total: one condition may use the whole of it.
    fn max_bytes() -> u64 {
        256 << 20
    }

    /// 256 MiB, which also bounds the request body that carries `data:` media.
    fn max_request_bytes() -> Option<u64> {
        Some(256 << 20)
    }
}

/// Fetching input images of chat requests (`image_fetch_config`).
pub type ImageFetchConfig = MediaSources<ImageMedia>;

/// Condition media of video requests (`media_config`).
pub type MediaConfig = MediaSources<ConditionMedia>;

/// The defaults the two media tables share, each a policy of the connector or the prober.
mod media {
    use std::num::NonZeroUsize;
    use std::time::Duration;

    /// `http(s)` media is fetched unless a deployment refuses it.
    pub(super) fn allow_remote() -> bool {
        true
    }

    /// 10 seconds to establish a connection, well past a reachable host's handshake.
    pub(super) fn connect_timeout() -> Duration {
        Duration::from_secs(10)
    }

    /// Three redirects, as many as a CDN chain takes, so a redirect loop ends early.
    pub(super) fn max_redirects() -> u32 {
        3
    }

    /// 64 downloads at once.
    pub(super) fn max_concurrent_fetches() -> NonZeroUsize {
        super::positive(64)
    }

    /// 30 seconds for a demuxer to read one medium's streams, which a crafted file could
    /// otherwise stretch without bound.
    pub(super) fn probe_timeout() -> Duration {
        Duration::from_secs(30)
    }
}

/// The workers' always-on profiling (`profiling_config`): the fence
/// probes at the entry and exit of the model's modules, which export
/// per-step durations by module class, the kernel probes of each kind of
/// layer, and how far back the request timelines reach (`docs/profiling.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, crate::reload::Reload)]
#[serde(default, deny_unknown_fields)]
pub struct ProfilingConfig {
    /// Whether the probes record: the device switch every probe reads,
    /// changed in place, at the engine's next step boundary.
    #[reload(step)]
    pub fence_probes: bool,
    /// Name patterns (`*` matches any run) of the module classes whose
    /// instances carry probes in the CUDA graphs the engines capture. Empty for no probe; the
    /// model family's step stack (the decoder or denoiser stack, one instance per step) when
    /// unset. Probes are captured into the graphs, so a change is a kernel generation: every
    /// bucket is captured again with the probes moved, by the range markers an engine installs
    /// when it starts with probes; a change from no probe takes a new epoch, which installs them
    /// ([`fence_effect`]).
    #[reload(by = fence_effect)]
    pub fence_modules: Option<Vec<String>>,
    /// Name patterns (`*` matches any run) of the probed module classes whose
    /// probes are off while `fence_probes` is on: the classes whose
    /// probes cost more than the always-on budget. Changed in place, at the
    /// engine's next step boundary, as `fence_probes` is.
    #[reload(step)]
    pub fence_probes_off: Vec<String>,
    /// Whether every worker enables CUPTI's PC sampling before it captures
    /// its CUDA graphs and keeps it enabled, stopped between sessions, so
    /// that sessions attribute the samples of graph kernels. CUPTI then
    /// refuses PM sampling, the range profiler and whole-graph activity
    /// records, so sessions of those are refused. CUPTI attributes only the
    /// graphs captured while it samples, so a change is a kernel generation:
    /// the serving engines switch CUPTI's mode, capture every graph again
    /// under it in the background and switch to them whole, and switch the
    /// mode back when the generation is given back.
    #[reload(generation)]
    pub graph_pc_sampling: bool,
    /// Whether the engine's NVTX ranges (the `uniserve.worker` domain and
    /// the scheduler's and control thread's ranges) are emitted from the
    /// epoch's start, for a profiler that records them, such as Nsight
    /// Systems; a session's `timeline` collector turns them on while it
    /// records either way. They are host ranges no graph captures: every main process switches
    /// them with its engine's next step values ([`Tuning`]).
    #[reload(step)]
    pub nvtx_ranges: bool,
    /// A profiling session the serving epoch's workers run (`docs/profiling.md`): its
    /// collectors, its window and its output directory. While it is stated, the session starts
    /// at the engine's next step boundary and records until its window ends, its guard ends it,
    /// or its epoch stops executing; a change ends the running session, every worker writing its
    /// file, and starts the new one, and removing it ends the running one. Each state the
    /// session enters is a `log` event of source `profile`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[reload(step)]
    pub session: Option<crate::profile::SessionRequest>,
    /// Which instance of each kind of layer carries a kernel probe before and after each of its
    /// kernels, by its position among the layers of its kind in launch order (the last when the
    /// kind has fewer), beside a pair around every kernel outside the layers; no kernel probe is
    /// placed when unset (`docs/profiling.md`). Probes are captured into the CUDA graphs, so a
    /// change is a kernel generation: every bucket is captured again with the probes moved.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[reload(generation)]
    pub kernel_instance: Option<u32>,
    /// Whether the kernel probes the graphs carry record: their device switch, changed in place
    /// at the engine's next step boundary, as `fence_probes` is.
    #[reload(step)]
    pub kernel_probes: bool,
    /// How far back a read of the `span` events reaches: the spans that ended longer ago are not
    /// read, whatever the processes' rings still hold. Read by the leader at each query.
    #[serde(with = "values::duration")]
    #[reload(step)]
    pub span_window: Duration,
}

/// How far back a read of the `span` events reaches by default: five minutes.
pub const DEFAULT_SPAN_WINDOW: Duration = Duration::from_secs(300);

/// How a change of `profiling_config.fence_modules` from `previous` to `next` takes effect: a
/// kernel generation, whose captures place the probes on the classes `next` names, in an engine
/// that started with probes and so installed the range markers; from no probe (an empty list),
/// and at a start, a new epoch, which installs them.
pub fn fence_effect(
    previous: Option<&Option<Vec<String>>>,
    _next: &Option<Vec<String>>,
) -> crate::reload::Effect {
    match previous {
        Some(Some(patterns)) if patterns.is_empty() => crate::reload::Effect::Epoch,
        Some(_) => crate::reload::Effect::Generation,
        None => crate::reload::Effect::Epoch,
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            fence_probes: true,
            fence_modules: None,
            fence_probes_off: Vec::new(),
            graph_pc_sampling: false,
            nvtx_ranges: false,
            session: None,
            kernel_instance: None,
            kernel_probes: true,
            span_window: DEFAULT_SPAN_WINDOW,
        }
    }
}

/// The KV events every scheduler publishes (`kv_events_config`): which pages its cache holds
/// on which medium, in SGLang's ZMQ format (`storage::events`), for KV-aware routers outside
/// the deployment (Dynamo's, `sgl-router`, Mooncake's conductor). Each field takes effect at the
/// schedulers' next step: a scheduler that starts publishing states the pages it commits from
/// then on, after an `AllBlocksCleared`, and one whose endpoint changes publishes at the new one
/// from its next step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, crate::reload::Reload)]
#[serde(default, deny_unknown_fields)]
pub struct KvEventsConfig {
    /// The `tcp://<host>:<port>` endpoint the schedulers' publishers bind, `*` for every
    /// interface: the deployment's scheduler `n` (the schedulers numbered across its replicas
    /// in order, [`EngineArgs::replica`]) binds the port plus `n`, as SGLang's data-parallel
    /// rank `n` does. Unset, no scheduler publishes.
    #[reload(step)]
    pub endpoint: Option<String>,
    /// The endpoint each publisher's replay socket binds, offset as `endpoint` is: a subscriber
    /// that missed batches asks it for those from a sequence number on. Unset, there is none.
    #[reload(step)]
    pub replay_endpoint: Option<String>,
    /// The topic every batch is published under; empty unless stated, as SGLang's.
    #[reload(step)]
    pub topic: String,
    /// The batches a publisher keeps for its replay socket ([`DEFAULT_REPLAY_BATCHES`] unless
    /// stated, as SGLang's `buffer_steps`).
    #[reload(step)]
    pub buffer_steps: NonZeroUsize,
}

/// The batches a KV event publisher keeps for replay when the deployment states none
/// (`kv_events_config.buffer_steps`): SGLang's 10000.
pub const DEFAULT_REPLAY_BATCHES: NonZeroUsize = match NonZeroUsize::new(10_000) {
    Some(count) => count,
    None => NonZeroUsize::MIN,
};

impl Default for KvEventsConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            replay_endpoint: None,
            topic: String::new(),
            buffer_steps: DEFAULT_REPLAY_BATCHES,
        }
    }
}

impl KvEventsConfig {
    /// `endpoint` with its port offset by `ordinal` and `*` bound as every interface: what
    /// scheduler `ordinal` of the deployment binds. `None` for an endpoint that is no
    /// `tcp://<host>:<port>` or whose offset port leaves the port range.
    pub fn offset(endpoint: &str, ordinal: u32) -> Option<String> {
        let address = endpoint.strip_prefix("tcp://")?;
        let (host, port) = address.rsplit_once(':')?;
        let port = port.parse::<u16>().ok()?;
        let port = u16::try_from(u32::from(port).checked_add(ordinal)?).ok()?;
        let host = if host == "*" { "0.0.0.0" } else { host };
        Some(format!("tcp://{host}:{port}"))
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
    /// The directory of the weight adapter fused into the checkpoint at load (`lora_path`): the
    /// adapter beside the contract of the student it is, from which the model family reads the
    /// module it edits, and every request of the deployment is served by the adapted weights;
    /// `None` serves the checkpoint as it is.
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
/// is a package's parameter or that the serving engine takes in place;
/// [`EngineConfig::build_part`] tells the two apart. Which package fills each kernel is no part
/// of it: the epoch's module set states that.
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
    pub device: DeviceConfig,
    pub scheduler: SchedulerConfig,
    /// The role of each of the engine's schedulers, by its index (`replicas.<id>.role`): which
    /// requests it takes. A scheduler past them is mixed.
    pub roles: Vec<Role>,
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
    /// The KV events the schedulers publish, a step value ([`Tuning::kv_events`]).
    pub kv_events: KvEventsConfig,
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
    /// generation-layer value. The generation layer's values are the scheduling policy's
    /// parameters (the scheduler section, the batched token budget, the speculative cost
    /// figures); the graph options a kernel generation captures the engine's graphs for (the
    /// bucket catalog, `max_logprobs` and the drafts); the schedulers' roles; the values the serving engine takes at a step boundary
    /// (the probes' switches, the NVTX ranges, the queue depth, the workers' bounds, report
    /// period and copy budget); and the media section, which the frontend reads. A switch the engine takes in place only
    /// when it is switched off (`kv_cache_config.prefix_caching`, `host_tier`) stays in the
    /// build part.
    pub fn build_part(&self) -> BuildPart<'_> {
        let Self {
            executable,
            model,
            parallel,
            max_running_requests,
            max_num_batched_tokens: _,
            max_logprobs: _,
            device,
            scheduler: _,
            roles: _,
            media: _,
            mem_fraction_static,
            load,
            kv_cache,
            moe,
            cuda_graph: _,
            worker,
            speculative,
            video,
            profiling,
            kv_events: _,
        } = self;
        let MoeConfig {
            expert_parallel,
            microbatches: _,
            beside_rows,
            expert_order: _,
        } = moe;
        let WorkerConfig {
            engine,
            engine_revision,
            // The leader daemon's, not the engines': the epoch's module set fixes it.
            processing: _,
            processing_revision: _,
            watchdog_timeout: _,
            join_timeout: _,
            compare_timeout: _,
            mesh,
            standby,
            standby_storage,
            report_period: _,
            copy_budget: _,
            host_threads,
            host_queue_depth,
        } = worker;
        let DeviceConfig {
            device,
            devices,
            worker_hosts,
            queue_depth: _,
        } = device;
        let SpeculativeConfig {
            num_draft_tokens: _,
            expected_speedup: _,
            refill_token_cost: _,
        } = speculative;
        let ProfilingConfig {
            fence_probes: _,
            fence_modules: _,
            fence_probes_off: _,
            graph_pc_sampling: _,
            nvtx_ranges: _,
            session: _,
            kernel_instance: _,
            kernel_probes: _,
            span_window: _,
        } = profiling;
        BuildPart {
            executable,
            model,
            parallel,
            max_running_requests: *max_running_requests,
            device: *device,
            devices,
            worker_hosts,
            mem_fraction_static: *mem_fraction_static,
            load,
            kv_cache,
            expert_parallel: *expert_parallel,
            beside_rows: *beside_rows,
            engine,
            engine_revision: *engine_revision,
            mesh,
            standby: *standby,
            standby_storage: *standby_storage,
            host_threads: *host_threads,
            host_queue_depth: *host_queue_depth,
            video,
        }
    }

    /// Takes from `target` the values a kernel generation moves the serving engines to (the
    /// probes' places, `profiling_config.kernel_instance` and `fence_modules`,
    /// `moe_config.microbatches`, and the graph options: the bucket catalog, `max_logprobs`,
    /// `speculative_config.num_draft_tokens` and `profiling_config.graph_pc_sampling`): what a
    /// relaunch and a standby start from once the generation committed.
    pub fn take_generation(&mut self, target: &EngineConfig) {
        self.profiling.kernel_instance = target.profiling.kernel_instance;
        self.profiling
            .fence_modules
            .clone_from(&target.profiling.fence_modules);
        self.moe.microbatches = target.moe.microbatches;
        self.cuda_graph.clone_from(&target.cuda_graph);
        self.max_logprobs = target.max_logprobs;
        self.speculative.num_draft_tokens = target.speculative.num_draft_tokens;
        self.profiling.graph_pc_sampling = target.profiling.graph_pc_sampling;
    }

    /// The options this configuration's kernel generations capture the CUDA graphs for
    /// ([`GraphOptions`]).
    pub fn graph_options(&self) -> GraphOptions {
        GraphOptions {
            cuda_graph: self.cuda_graph.clone(),
            max_logprobs: self.max_logprobs,
            num_draft_tokens: self.speculative.num_draft_tokens,
            graph_pc_sampling: self.profiling.graph_pc_sampling,
        }
    }

    /// The layout this configuration lays an epoch's device memory out in ([`Layout`]).
    pub fn layout(&self) -> Layout {
        Layout {
            parallel: self.parallel.clone(),
            max_running_requests: self.max_running_requests,
            block_size: self.kv_cache.block_size,
            dtype: self.kv_cache.dtype,
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
    pub max_running_requests: NonZeroUsize,
    pub device: DeviceKind,
    pub devices: &'a [u32],
    pub worker_hosts: &'a [String],
    pub mem_fraction_static: f64,
    pub load: &'a LoadConfig,
    pub kv_cache: &'a KvCacheConfig,
    pub expert_parallel: bool,
    /// The expert union's middle capture class, which the captured graphs are shaped for.
    pub beside_rows: NonZeroU32,
    pub engine: &'a str,
    pub engine_revision: Option<u32>,
    pub mesh: &'a Option<String>,
    pub standby: bool,
    pub standby_storage: StandbyStorage,
    pub host_threads: Option<NonZeroUsize>,
    pub host_queue_depth: NonZeroUsize,
    pub video: &'a VideoConfig,
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
/// and the KV cache's block size and element type. The graph buckets are not part of it: a kernel
/// generation changes them in place, and the KV units lie as they lay. Two epochs of one layout
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
    /// The configuration revision the process starts with: its schedulers' first snapshot of
    /// the step values ([`Tuning`]), which every batch they assemble names until another
    /// revision's values arrive.
    #[serde(default)]
    pub revision: u64,
    /// The ordinal of the process's replica among the epoch's replicas, in the order the
    /// deployment's worker groups state them: what numbers its schedulers across the
    /// deployment, scheduler `s` of replica `r` being the deployment's scheduler `r + s` (an
    /// engine of several schedulers is the deployment's one replica, whose experts they share).
    /// It offsets the ports of each scheduler's own endpoints (`kv_events_config.endpoint`).
    #[serde(default)]
    pub replica: u32,
    /// The deployment, by its policy package's name: the name its daemon, Storage Server and
    /// Build Server on every node serve it under, which keys its claims and pages at the
    /// Storage Server.
    pub deployment: String,
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
    /// no process, with the expert order its configuration states (`moe.expert_order`).
    pub fn new(engine_config: EngineConfig, modules: crate::module::ModuleSet) -> Self {
        Self {
            engine_config,
            epoch: 0,
            predecessor: None,
            revision: 0,
            replica: 0,
            deployment: String::new(),
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
            world_size: 0,
        }
    }

    /// Whether this engine starts parked: it states the engine it succeeds, whose storage it
    /// maps read-only until it is promoted. A switch's target and a standby both start so.
    pub fn parked(&self) -> bool {
        self.predecessor.is_some()
    }
}
