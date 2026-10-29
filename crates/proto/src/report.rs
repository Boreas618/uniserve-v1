//! Serializable scheduler, cache, and worker performance snapshots.
//!
//! The engine's `SchedulerStatsReporter` builds one [`SchedulerStats`] per
//! reporting interval from its cumulative scheduler counters, and the
//! daemon's `record_scheduler_stats` publishes it as Prometheus metrics:
//! interval deltas are added to counters, while gauges and lifetime values
//! are set. Each field's documentation says which kind it is when that is not
//! an interval delta. [`ForwardStats`] also travels from workers to the engine
//! in batch results that report it.
//!
//! A snapshot crosses the request interface JSON-encoded
//! ([`SchedulerStats::encode`]): its fields evolve with every build, each
//! with a default, so a daemon reads the snapshots of engines of other
//! builds.

mod forward;

pub use forward::{FENCE_BUCKETS_US, FenceHistogram, ForwardStats};
use serde::{Deserialize, Serialize};

/// Cache query and hit counters for one reporting interval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BaseCacheStats {
    /// The number of requests in this update.
    pub requests: u64,
    /// The number of queries in these requests.
    pub queries: u64,
    /// The number of hits in these requests.
    pub hits: u64,
}

/// Prefix-cache counters, where query fields count tokens.
///
/// The engine's `SchedulerStatsReporter` fills only `base.queries` and
/// `base.hits`; it reports the request and preemption counters as zero.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefixCacheStats {
    /// Embedded base cache counters.
    #[serde(flatten)]
    pub base: BaseCacheStats,
    /// The part of `base.hits` loaded from the host KV cache, in tokens.
    #[serde(default)]
    pub host_hits: u64,
    /// The number of requests preempted before this stats update.
    pub preempted_requests: u64,
    /// The `queries` number for preempted requests.
    pub preempted_queries: u64,
    /// The `hits` number for preempted requests.
    pub preempted_hits: u64,
}

/// Per-domain scheduler accounting for one stats update.
///
/// The engine aggregates call kinds into the `prefill`, `decode`, and `flow`
/// domains. `active_credits` is a gauge and `peak_credits` a lifetime maximum;
/// every other count and time field is an interval delta. The worker-reported
/// time fields (`launch_us`, `device_us`, `completion_us`) add, once per batch
/// result report that contains calls of the domain, the maximum over those
/// calls of each `TimingCounters` field they draw on. Batches of different
/// domains can overlap on a device, so time fields must not be summed
/// across domains to estimate aggregate GPU busy time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainSchedulerStats {
    /// Stable execution-domain name.
    pub domain: String,
    /// Credits currently occupied by in-flight work.
    #[serde(default)]
    pub active_credits: u64,
    /// Highest active-credit count since the scheduler started.
    #[serde(default)]
    pub peak_credits: u64,
    /// Calls launched in the interval.
    #[serde(default)]
    pub launched_calls: u64,
    /// Call results in the interval, including predicated and error results.
    #[serde(default)]
    pub completed_calls: u64,
    /// Calls skipped by a false predicate.
    #[serde(default)]
    pub predicated_calls: u64,
    /// Calls that returned an error status or whose credit was reclaimed
    /// after an execution failure.
    #[serde(default)]
    pub error_calls: u64,
    /// Submission attempts rejected by executor backpressure.
    #[serde(default)]
    pub backpressure_events: u64,
    /// Scheduling credits returned on completion or execution failure.
    #[serde(default)]
    pub reclaimed_credits: u64,
    /// Batch result reports in the interval that contained calls of this
    /// domain.
    #[serde(default)]
    pub completed_batches: u64,
    /// Scheduler time from planning a call to registering it in flight, in
    /// microseconds. Calls of diffusion media requests contribute zero.
    #[serde(default)]
    pub queue_us: u64,
    /// Worker-reported wait before execution (`TimingCounters::queued_us`),
    /// in microseconds.
    #[serde(default)]
    pub launch_us: u64,
    /// Device execution time in microseconds.
    #[serde(default)]
    pub device_us: u64,
    /// Worker completion copy plus host processing time
    /// (`TimingCounters::copy_us` and `host_us`), in microseconds.
    #[serde(default)]
    pub completion_us: u64,
}

/// Serializable scheduler snapshot for one reporting interval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Running token and media requests (gauge).
    pub num_running_reqs: u64,
    /// Token and media requests waiting for admission (gauge).
    pub num_waiting_reqs: u64,
    /// Generation and command-only batches the scheduler has finalized since
    /// it started (a lifetime value, not a delta). Batches built by the media
    /// scheduling pass for diffusion media requests, including its retirement
    /// batch, are not counted.
    pub step_counter: u64,
    /// Data-parallel wave number. The engine tracks no waves and reports zero.
    pub current_wave: u64,
    /// Fraction of usable cache blocks not free (gauge). `1.0` means 100% usage.
    pub prefix_cache_usage: f64,
    /// Requests admitted since the previous stats snapshot.
    #[serde(default)]
    pub num_admitted_reqs: u64,
    /// Average queue wait for requests admitted since the previous snapshot,
    /// in microseconds; zero when none was admitted.
    #[serde(default)]
    pub avg_queue_wait_us: u64,
    /// Total queue wait for requests admitted since the previous snapshot, in
    /// microseconds.
    #[serde(default)]
    pub queue_wait_us_total: u64,
    /// Maximum queue wait observed by the scheduler so far, in microseconds
    /// (a lifetime value, not a delta).
    #[serde(default)]
    pub max_queue_wait_us: u64,
    /// Local prefix-cache query and hit deltas, in tokens.
    pub prefix_cache_stats: PrefixCacheStats,
    /// Worker-local forward/kernel counters since the previous stats snapshot,
    /// or `None` when none of them changed.
    #[serde(default)]
    pub worker_forward_stats: Option<ForwardStats>,
    /// Worker-reported execution time accumulated in the interval, in microseconds.
    #[serde(default)]
    pub worker_exec_us: u64,
    /// Host-observed submit-to-result time accumulated in the interval, in microseconds.
    #[serde(default)]
    pub batch_roundtrip_us: u64,
    /// Completed batches represented by `worker_exec_us` and
    /// `batch_roundtrip_us` in this interval.
    #[serde(default)]
    pub batch_count: u64,
    /// Calls whose forward produced non-finite logits in the interval.
    #[serde(default)]
    pub numerical_faults: u64,
    /// Per-domain accounting, in `prefill`, `decode`, `flow` order.
    #[serde(default)]
    pub domain_stats: Vec<DomainSchedulerStats>,
}

impl SchedulerStats {
    /// The JSON encoding the request interface carries.
    ///
    /// # Errors
    ///
    /// Fails only when serialization fails, which these plain values do not.
    pub fn encode(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Decodes [`Self::encode`]'s encoding.
    ///
    /// # Errors
    ///
    /// Fails when `bytes` are not a snapshot.
    pub fn decode(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}
