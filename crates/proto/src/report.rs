//! Serializable scheduler, cache, and worker performance snapshots.
//!
//! A scheduler counts into one [`SchedulerStats`] as it works and sends it
//! once a second, starting the next interval's counts from zero, and the
//! daemon's `record_scheduler_stats` publishes it as Prometheus metrics:
//! interval deltas are added to counters, while gauges and lifetime values
//! are set. Each field's documentation says which kind it is when that is not
//! an interval delta. [`ForwardStats`] also travels from workers to the engine
//! in batch results that report it.
//!
//! A snapshot crosses the request interface as its Protobuf message
//! (`request.proto`'s `SchedulerStats`, converted by `wire`).

mod forward;
mod wire;

pub use forward::{
    BlockingCall, FenceStep, ForwardStats, KernelLaunch, ProfileProgress, StreamTime,
};
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
    pub host_hits: u64,
    /// The part of `host_hits` read from peer hosts' pools first, in tokens.
    pub peer_hits: u64,
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
    pub active_credits: u64,
    /// Highest active-credit count since the scheduler started.
    pub peak_credits: u64,
    /// Calls launched in the interval.
    pub launched_calls: u64,
    /// Call results in the interval, including predicated and error results.
    pub completed_calls: u64,
    /// Calls skipped by a false predicate.
    pub predicated_calls: u64,
    /// Calls that returned an error status or whose credit was reclaimed
    /// after an execution failure.
    pub error_calls: u64,
    /// Submission attempts rejected by worker backpressure.
    pub backpressure_events: u64,
    /// Scheduling credits returned on completion or execution failure.
    pub reclaimed_credits: u64,
    /// Batch result reports in the interval that contained calls of this
    /// domain.
    pub completed_batches: u64,
    /// Scheduler time from planning a call to registering it in flight, in
    /// microseconds. Calls of diffusion media requests contribute zero.
    pub queue_us: u64,
    /// Worker-reported wait before execution (`TimingCounters::queued_us`),
    /// in microseconds.
    pub launch_us: u64,
    /// Device execution time in microseconds.
    pub device_us: u64,
    /// Worker completion copy plus host processing time
    /// (`TimingCounters::copy_us` and `host_us`), in microseconds.
    pub completion_us: u64,
}

/// Serializable scheduler snapshot for one reporting interval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// The configuration revision of the step values the scheduler last read
    /// (`config::Tuning`): the revision the interval's measurements ran under.
    pub revision: u64,
    /// Running token and media requests (gauge).
    pub num_running_reqs: u64,
    /// Token and media requests waiting for admission (gauge).
    pub num_waiting_reqs: u64,
    /// Batches the scheduler has submitted to its executor since it started,
    /// of every phase (a lifetime value, not a delta).
    pub step_counter: u64,
    /// Data-parallel wave number. The engine tracks no waves and reports zero.
    pub current_wave: u64,
    /// Fraction of usable cache blocks not free (gauge). `1.0` means 100% usage.
    pub prefix_cache_usage: f64,
    /// Requests admitted since the previous stats snapshot.
    pub num_admitted_reqs: u64,
    /// Average queue wait for requests admitted since the previous snapshot,
    /// in microseconds; zero when none was admitted.
    pub avg_queue_wait_us: u64,
    /// Total queue wait for requests admitted since the previous snapshot, in
    /// microseconds.
    pub queue_wait_us_total: u64,
    /// Maximum queue wait observed by the scheduler so far, in microseconds
    /// (a lifetime value, not a delta).
    pub max_queue_wait_us: u64,
    /// Local prefix-cache query and hit deltas, in tokens.
    pub prefix_cache_stats: PrefixCacheStats,
    /// Worker-local forward/kernel counters since the previous stats snapshot,
    /// or `None` when none of them changed.
    pub worker_forward_stats: Option<ForwardStats>,
    /// Worker-reported execution time accumulated in the interval, in microseconds.
    pub worker_exec_us: u64,
    /// Submit-to-result time of the interval's batches, accumulated, in microseconds: each
    /// batch's wait in its worker's queue, its device time, and its copy and host time
    /// (`TimingCounters`), from its submission to its output as its workers measured them.
    pub batch_roundtrip_us: u64,
    /// Completed batches represented by `worker_exec_us` and
    /// `batch_roundtrip_us` in this interval.
    pub batch_count: u64,
    /// Query tokens the `batch_count` batches computed. The scheduler counts them from the calls it
    /// planned, so a report states the interval's throughput whatever its worker reports
    /// (`worker_forward_stats` is empty for a main process).
    pub batch_tokens: u64,
    /// Calls whose forward produced non-finite logits in the interval.
    pub numerical_faults: u64,
    /// Per-domain accounting, in `prefill`, `decode`, `flow` order.
    pub domain_stats: Vec<DomainSchedulerStats>,
    /// Calls left out of a step in the interval because blocks they write had not arrived from
    /// the buffer pool: the request's `Allocate` was still outstanding when the call was due.
    pub calls_without_blocks: u64,
    /// Transfers of a request between worker groups, from the group that prefilled it to a decode
    /// group, that the buffer pool answered `Transferred` in the interval.
    pub transfers: u64,
    /// Transfers the buffer pool answered `Failed` in the interval; each request ended as
    /// migratable.
    pub failed_transfers: u64,
    /// The blocks the target groups of the interval's transfers received: every entry each
    /// `Transferred` states, by class and position, whether its task copied the block or the
    /// target group's memory held it already.
    pub transferred_blocks: u64,
    /// Running requests the scheduler preempted in the interval because a request's next call
    /// found no room: each ended as migratable (`Migratable { cause: Drain }`, a video request
    /// with its trajectory stored) for the daemon to resubmit, whether its policy named it or it
    /// was the last admitted of a shortage every running request shared.
    pub preempted_requests: u64,
    /// Cached blocks that left the scheduler's regions in the interval: each key a completion
    /// lists as evicted, once per region its request took blocks from (an allocation, a copy, a
    /// prefetch or a shrink taking the eviction order's head), and each key an `Evict` of the
    /// policy's victims removed.
    pub evicted_blocks: u64,
    /// Draft-cache blocks running requests gave up in the interval so that another request's next
    /// call had room, each request its whole draft table.
    pub given_up_draft_blocks: u64,
    /// The scheduler's steps since it started, each one decision of its policy (a lifetime
    /// value). The main process reports it as `Metrics::steps`, `step_counter` as
    /// `Metrics::batches` and the sum of `batch_tokens` as `Metrics::tokens`.
    pub steps: u64,
    /// The calls into the scheduler's policy since it started, over every policy generation it
    /// served (a lifetime value).
    pub policy_calls: u64,
    /// The fuel those calls consumed (a lifetime value).
    pub policy_fuel: u64,
}
