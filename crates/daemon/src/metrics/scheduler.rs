//! Scheduler, cache, and worker execution metrics.
//!
//! [`SchedulerMetrics::record`] writes every family here from one `SchedulerStats` snapshot per
//! reporting interval: it sets gauges and lifetime values, and adds interval deltas to counters.
//! The worker-local forward and kernel families, which come from `ForwardStats`, change only in
//! intervals whose snapshot carries `worker_forward_stats`.
//!
//! Each scalar of a report is declared once ([`scalar_families!`]): the field that holds its
//! family, which is also its exported name after `uniserve:`, its kind, its help and the value of
//! a report it records. The breakdowns keyed by a domain, a mode, a stream or a backend are
//! declared beside them, each once.

use prometheus_client::encoding::{EncodeLabelSet, LabelSetEncoder};
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;

use proto::report::SchedulerStats;

use crate::metrics::{CountedHistogram, F64Gauge, U64Counter, U64Gauge};

/// Labels identifying one scheduler: the served model, the replica (an independent engine of the
/// epoch) and the scheduler (a data-parallel rank of that replica's engine).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct SchedulerLabels {
    /// Served model identity.
    pub model_name: String,
    /// The replica the scheduler belongs to.
    pub replica: u32,
    /// The scheduler's index in its replica's engine.
    pub scheduler: u32,
}

/// [`SchedulerLabels`] with the labels of one entry of a keyed breakdown.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct KeyedLabels<K> {
    /// The scheduler.
    pub scheduler: SchedulerLabels,
    /// The breakdown's entry.
    pub key: K,
}

impl SchedulerLabels {
    /// These labels with the labels of the entry `key` of a keyed breakdown.
    pub fn with<K>(&self, key: K) -> KeyedLabels<K> {
        KeyedLabels {
            scheduler: self.clone(),
            key,
        }
    }
}

impl<K: EncodeLabelSet> EncodeLabelSet for KeyedLabels<K> {
    fn encode(&self, encoder: &mut LabelSetEncoder) -> Result<(), std::fmt::Error> {
        self.scheduler.encode(encoder)?;
        self.key.encode(encoder)
    }
}

/// A scheduler execution domain.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DomainLabel {
    /// Scheduler execution domain: `prefill`, `decode`, or `flow`.
    pub domain: String,
}

/// An event kind within a scheduler domain.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DomainKindLabel {
    /// Scheduler execution domain: `prefill`, `decode`, or `flow`.
    pub domain: String,
    /// Call outcome (`launched`, `completed`, `predicated`, `error`) for
    /// `scheduler_domain_calls`, or timing phase (`queue`, `launch`, `device`,
    /// `completion`) for `scheduler_domain_time_us`.
    pub kind: &'static str,
}

/// A worker runtime mode.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ModeLabel {
    /// Worker runtime mode.
    pub mode: String,
}

/// A speculative-verification resolution path.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PathLabel {
    /// Speculative-verification resolution path reported by the worker.
    pub path: String,
}

/// The attention backend that served a kernel launch.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct BackendLabel {
    /// Attention kernel backend family.
    pub backend: String,
}

/// One state of a worker stream's device time.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct StreamLabel {
    /// Worker stream: `device`, the rank's primary stream, or `copy`, the
    /// stream of its page copies through the host KV cache.
    pub stream: String,
    /// `busy`: device time the stream's operations spanned; `gap`: idle time
    /// of the stream between them; `late`: time of records too far out of
    /// order to merge.
    pub state: String,
}

/// A driver call that made a worker thread wait for the device.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct BlockingLabel {
    /// Worker thread (`builder`, `control`, `device`).
    pub thread: String,
    /// Driver call.
    pub call: String,
}

/// One module class of the fence probes' step durations.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct FenceLabel {
    /// The steps' stage: their call kind, such as `prefill` or `decode`.
    pub stage: String,
    /// The module class (the name its modules give their profiler range),
    /// or `step` for the step's span between its markers.
    pub module: String,
}

/// How one report's value reaches its family: a gauge is set, a counter added to.
trait Record {
    /// The value a report states.
    type Value;

    /// Records `value` on this series.
    fn record(&self, value: Self::Value);
}

impl Record for U64Gauge {
    type Value = u64;

    fn record(&self, value: u64) {
        self.set(value);
    }
}

impl Record for F64Gauge {
    type Value = f64;

    fn record(&self, value: f64) {
        self.set(value);
    }
}

impl Record for U64Counter {
    type Value = u64;

    fn record(&self, value: u64) {
        self.inc_by(value);
    }
}

/// Declares the scalar families of a scheduler's report, each once: the field that holds the
/// family, which is its exported name after `uniserve:`; its series type (a gauge is set, a
/// counter added to); its help; and the value it records from a report, `None` for a report that
/// carries none (the worker counters of an interval in which none changed).
macro_rules! scalar_families {
    ($( $field:ident: $kind:ty, $help:literal, $value:expr; )*) => {
        /// The scalar families of a scheduler's report, one per quantity.
        pub struct ScalarMetrics {
            $(
                #[doc = $help]
                pub $field: Family<SchedulerLabels, $kind>,
            )*
        }

        impl ScalarMetrics {
            /// Registers every scalar family under its name.
            fn register(registry: &mut Registry) -> Self {
                $(
                    let $field = Family::<SchedulerLabels, $kind>::default();
                    registry.register(
                        concat!("uniserve:", stringify!($field)),
                        $help,
                        $field.clone(),
                    );
                )*
                Self { $($field,)* }
            }

            /// Records every scalar `stats` carries under `labels`.
            fn record(&self, labels: &SchedulerLabels, stats: &SchedulerStats) {
                $(
                    let value: fn(&SchedulerStats) -> Option<<$kind as Record>::Value> = $value;
                    if let Some(value) = value(stats) {
                        self.$field.get_or_create(labels).record(value);
                    }
                )*
            }
        }
    };
}

scalar_families! {
    scheduler_running: U64Gauge,
        "Requests in model execution batches.",
        |stats| Some(stats.num_running_reqs);
    scheduler_waiting: U64Gauge,
        "Requests waiting for admission.",
        |stats| Some(stats.num_waiting_reqs);
    prefix_cache_usage: F64Gauge,
        "Fraction of the usable cache blocks that are not free; 1 means full.",
        |stats| Some(stats.prefix_cache_usage);
    scheduler_admitted: U64Counter,
        "Requests admitted into running batches.",
        |stats| Some(stats.num_admitted_reqs);
    scheduler_queue_wait_us: U64Counter,
        "Queue wait of the admitted requests, in microseconds.",
        |stats| Some(stats.queue_wait_us_total);
    scheduler_queue_wait_max_us: U64Gauge,
        "Longest queue wait the scheduler observed since it started, in microseconds.",
        |stats| Some(stats.max_queue_wait_us);
    prefix_cache_queries: U64Counter,
        "Prefix-cache queries, in queried tokens.",
        |stats| Some(stats.prefix_cache_stats.base.queries);
    prefix_cache_hits: U64Counter,
        "Prefix-cache hits, in cached tokens.",
        |stats| Some(stats.prefix_cache_stats.base.hits);
    prefix_cache_host_hits: U64Counter,
        "The prefix-cache hits loaded from the host KV cache, in tokens.",
        |stats| Some(stats.prefix_cache_stats.host_hits);
    prefix_cache_peer_hits: U64Counter,
        "The host KV cache hits read from peer hosts' pools first, in tokens.",
        |stats| Some(stats.prefix_cache_stats.peer_hits);
    request_transfers: U64Counter,
        "Requests moved from the worker group that prefilled them to a decode group: transfers \
         the buffer pool answered as transferred.",
        |stats| Some(stats.transfers);
    request_transfers_failed: U64Counter,
        "Transfers of requests to a decode group that the buffer pool answered as failed; each \
         request ended as migratable.",
        |stats| Some(stats.failed_transfers);
    request_transfer_blocks: U64Counter,
        "Blocks the decode groups received by transfer: every entry of the transferred requests' \
         tables, by class and position.",
        |stats| Some(stats.transferred_blocks);
    scheduler_calls_without_blocks: U64Counter,
        "Calls a scheduler step left out because blocks they write had not arrived from the \
         buffer pool: the request's allocation was still outstanding when the call was due.",
        |stats| Some(stats.calls_without_blocks);
    scheduler_preempted_requests: U64Counter,
        "Running requests a scheduler preempted because a request's next call found no room: \
         each ended as migratable (a video request with its trajectory stored) and was \
         resubmitted.",
        |stats| Some(stats.preempted_requests);
    scheduler_evicted_blocks: U64Counter,
        "Cached blocks evicted from a scheduler's device and host regions: the eviction order's \
         head taken for an allocation, a copy, a prefetch or a shrink, once per region, and the \
         policy's victims evicted by key.",
        |stats| Some(stats.evicted_blocks);
    scheduler_given_up_draft_blocks: U64Counter,
        "Draft-cache blocks running requests gave up so that another request's next call had \
         room.",
        |stats| Some(stats.given_up_draft_blocks);
    worker_exec_us: U64Counter,
        "Worker-reported batch compute time, in microseconds.",
        |stats| Some(stats.worker_exec_us);
    batch_roundtrip_us: U64Counter,
        "Host-observed batch submit-to-result time, in microseconds.",
        |stats| Some(stats.batch_roundtrip_us);
    batch_timing_count: U64Counter,
        "Resolved batches the batch-time counters cover: the denominator of a per-batch average \
         of worker_exec_us and batch_roundtrip_us.",
        |stats| Some(stats.batch_count);
    worker_attention_launches: U64Counter,
        "Worker attention launches.",
        |stats| stats.worker_forward_stats.as_ref().map(|worker| worker.attention_launches);
    worker_attention_us: U64Counter,
        "Worker attention time, in microseconds.",
        |stats| stats.worker_forward_stats.as_ref().map(|worker| worker.attention_us);
    worker_fence_records_lost: U64Counter,
        "Fence probe records a worker's reader lost: overwritten before it read them.",
        |stats| stats.worker_forward_stats.as_ref().map(|worker| worker.fence_records_lost);
    worker_graph_captures: U64Counter,
        "Worker graph captures.",
        |stats| stats.worker_forward_stats.as_ref().map(|worker| worker.cuda_graph_captures);
    worker_graph_replays: U64Counter,
        "Worker graph replays.",
        |stats| stats.worker_forward_stats.as_ref().map(|worker| worker.cuda_graph_replays);
    worker_graph_misses: U64Counter,
        "Worker graph bucket misses.",
        |stats| stats.worker_forward_stats.as_ref().map(|worker| worker.cuda_graph_misses);
    worker_graph_fallbacks: U64Counter,
        "Worker executions that fell back from graph replay to eager execution.",
        |stats| stats.worker_forward_stats.as_ref().map(|worker| worker.cuda_graph_fallbacks);
    worker_graph_unpadded_tokens: U64Counter,
        "Real tokens of the worker's graph executions, before padding.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .map(|worker| worker.cuda_graph_unpadded_tokens)
        };
    worker_graph_padded_tokens: U64Counter,
        "Padding tokens the worker's graph buckets added.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .map(|worker| worker.cuda_graph_padded_tokens)
        };
    worker_text_decode_token_relay_hits: U64Counter,
        "Decode tokens served from the worker's token relay.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .map(|worker| worker.text_decode_token_relay_hits)
        };
    worker_text_decode_token_relay_misses: U64Counter,
        "Decode token relay misses.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .map(|worker| worker.text_decode_token_relay_misses)
        };
    worker_text_decode_position_relay_hits: U64Counter,
        "Decode positions served from the worker's position relay.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .map(|worker| worker.text_decode_position_relay_hits)
        };
    worker_text_decode_position_relay_misses: U64Counter,
        "Decode position relay misses.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .map(|worker| worker.text_decode_position_relay_misses)
        };
    worker_spec_verify_rows: U64Counter,
        "Rows the worker's speculative verification processed.",
        |stats| stats.worker_forward_stats.as_ref().map(|worker| worker.spec_verify_rows);
    worker_spec_verify_draft_tokens: U64Counter,
        "Draft tokens presented to speculative verification.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .map(|worker| worker.spec_verify_draft_tokens)
        };
    worker_spec_verify_accepted_tokens: U64Counter,
        "Draft tokens speculative verification accepted.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .map(|worker| worker.spec_verify_accepted_tokens)
        };
    worker_spec_verify_rejected_tokens: U64Counter,
        "Draft tokens speculative verification rejected.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .map(|worker| worker.spec_verify_rejected_tokens)
        };
    worker_spec_verify_committed_tokens: U64Counter,
        "Tokens committed after speculative verification.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .map(|worker| worker.spec_verify_committed_tokens)
        };
    spec_decode_accepted_length: F64Gauge,
        "Tokens committed per speculative verification step carrying drafts, on average over the \
         last report: its accepted drafts and the target's token after them.",
        |stats| {
            stats
                .worker_forward_stats
                .as_ref()
                .filter(|worker| worker.spec_verify_rows > 0)
                .map(|worker| {
                    worker.spec_verify_committed_tokens as f64 / worker.spec_verify_rows as f64
                })
        };
}

/// A family of one keyed breakdown of a scheduler's report.
type Keyed<K, M> = Family<KeyedLabels<K>, M>;

/// Scheduler/batch-scoped Prometheus families exported from `SchedulerStats`.
///
/// The gauge, lifetime, or interval-delta semantics of each source value are
/// documented on `SchedulerStats`, `DomainSchedulerStats`, and `ForwardStats`.
pub struct SchedulerMetrics {
    /// Every scalar of the report, one family each.
    pub scalars: ScalarMetrics,
    /// Active call credits by execution domain.
    pub scheduler_domain_active_credits: Keyed<DomainLabel, U64Gauge>,
    /// Peak active call credits by execution domain.
    pub scheduler_domain_peak_credits: Keyed<DomainLabel, U64Gauge>,
    /// Call lifecycle events by domain and outcome.
    pub scheduler_domain_calls: Keyed<DomainKindLabel, U64Counter>,
    /// Non-blocking resource-pressure events by domain.
    pub scheduler_domain_backpressure: Keyed<DomainLabel, U64Counter>,
    /// Reclaimed call credits by domain.
    pub scheduler_domain_reclaimed_credits: Keyed<DomainLabel, U64Counter>,
    /// Completed batches by domain.
    pub scheduler_domain_completed_batches: Keyed<DomainLabel, U64Counter>,
    /// Domain time in microseconds by lifecycle phase.
    pub scheduler_domain_time_us: Keyed<DomainKindLabel, U64Counter>,
    /// Attention launches by kernel backend.
    pub worker_attention_backend_counts: Keyed<BackendLabel, U64Counter>,
    /// Graph dispatches by runtime mode.
    pub worker_graph_runtime_mode_counts: Keyed<ModeLabel, U64Counter>,
    /// Forward calls by runtime mode.
    pub worker_forward_mode_counts: Keyed<ModeLabel, U64Counter>,
    /// Forward query tokens by runtime mode; modes without a token notion
    /// (encoders, decoders, module calls) have no series.
    pub worker_forward_mode_tokens: Keyed<ModeLabel, U64Counter>,
    /// Forward wall time by runtime mode.
    pub worker_forward_mode_us: Keyed<ModeLabel, U64Counter>,
    /// Worker stream device time by stream and state.
    pub worker_stream_us: Keyed<StreamLabel, U64Counter>,
    /// Worker driver calls that waited for the device, by thread and call.
    pub worker_blocking_calls: Keyed<BlockingLabel, U64Counter>,
    /// Fence probe step durations by stage and module class.
    pub worker_fence_step_seconds: Keyed<FenceLabel, CountedHistogram>,
    /// Speculative verification rows by resolution path.
    pub worker_spec_verify_path_counts: Keyed<PathLabel, U64Counter>,
}

/// Registers a keyed family under `name` with `help` and returns its handle; the registry keeps
/// a clone, whose state it shares.
fn keyed<K, M>(registry: &mut Registry, name: &str, help: &str) -> Keyed<K, M>
where
    K: EncodeLabelSet + Clone + std::hash::Hash + Eq + std::fmt::Debug + Send + Sync + 'static,
    M: prometheus_client::encoding::EncodeMetric
        + prometheus_client::metrics::TypedMetric
        + Default
        + std::fmt::Debug
        + Send
        + Sync
        + 'static,
{
    let family = Keyed::<K, M>::default();
    registry.register(name, help, family.clone());
    family
}

impl SchedulerMetrics {
    /// Registers every family under its exported name and help, and
    /// returns the handles; the registry keeps a clone of each, whose state
    /// it shares.
    pub fn register(registry: &mut Registry) -> Self {
        Self {
            scalars: ScalarMetrics::register(registry),
            scheduler_domain_active_credits: keyed(
                registry,
                "uniserve:scheduler_domain_active_credits",
                "Scheduler call credits in use, by execution domain.",
            ),
            scheduler_domain_peak_credits: keyed(
                registry,
                "uniserve:scheduler_domain_peak_credits",
                "Most scheduler call credits observed in use, by execution domain.",
            ),
            scheduler_domain_calls: keyed(
                registry,
                "uniserve:scheduler_domain_calls",
                "Scheduler call events by execution domain and kind: launched, completed, \
                 predicated, or error.",
            ),
            scheduler_domain_backpressure: keyed(
                registry,
                "uniserve:scheduler_domain_backpressure",
                "Nonblocking scheduler resource-pressure events by execution domain.",
            ),
            scheduler_domain_reclaimed_credits: keyed(
                registry,
                "uniserve:scheduler_domain_reclaimed_credits",
                "Scheduler call credits reclaimed, by execution domain.",
            ),
            scheduler_domain_completed_batches: keyed(
                registry,
                "uniserve:scheduler_domain_completed_batches",
                "Completed batches by execution domain.",
            ),
            scheduler_domain_time_us: keyed(
                registry,
                "uniserve:scheduler_domain_time_us",
                "Execution-domain time in microseconds by phase: queue, launch, device, or \
                 completion.",
            ),
            worker_attention_backend_counts: keyed(
                registry,
                "uniserve:worker_attention_backend_counts",
                "Worker attention launches by backend kernel family.",
            ),
            worker_graph_runtime_mode_counts: keyed(
                registry,
                "uniserve:worker_graph_runtime_mode_counts",
                "Worker forward dispatches by graph runtime mode.",
            ),
            worker_forward_mode_counts: keyed(
                registry,
                "uniserve:worker_forward_mode_counts",
                "Worker forward calls by forward mode.",
            ),
            worker_forward_mode_tokens: keyed(
                registry,
                "uniserve:worker_forward_mode_tokens",
                "Worker forward query tokens by forward mode.",
            ),
            worker_forward_mode_us: keyed(
                registry,
                "uniserve:worker_forward_mode_us",
                "Worker forward wall time by forward mode, in microseconds.",
            ),
            worker_stream_us: keyed(
                registry,
                "uniserve:worker_stream_us",
                "Worker stream device time in microseconds: for the primary stream (device), from the \
                 profiler's activity records, the union of the kernels, copies and fills on the \
                 stream, graph nodes included (busy), the stream's idle time between them (gap), \
                 and records too far out of order to merge (late); for the host copy streams \
                 (copy), the time between the CUDA events around each issue of copies (busy).",
            ),
            worker_blocking_calls: keyed(
                registry,
                "uniserve:worker_blocking_calls",
                "Driver calls that made a serving worker thread wait for the device, by thread \
                 and call; a serving call of the worker that made one failed.",
            ),
            worker_fence_step_seconds: keyed(
                registry,
                "uniserve:worker_fence_step_seconds",
                "Per-step durations from the workers' fence probes, by stage and module class \
                 (module): a step's summed durations of the class's probed module instances, or \
                 the step's span between its markers for module `step`, measured on the device.",
            ),
            worker_spec_verify_path_counts: keyed(
                registry,
                "uniserve:worker_spec_verify_path_counts",
                "Worker speculative verifier rows by verifier path.",
            ),
        }
    }

    /// Records one scheduler's interval snapshot `stats` under `labels`.
    ///
    /// Interval-delta fields are added to counters, while point-in-time values (running and
    /// waiting requests, cache usage, active credits) and lifetime maxima (`max_queue_wait_us`,
    /// `peak_credits`) are set on gauges. `stats` must therefore be an interval snapshot: passing
    /// a cumulative one, or the same snapshot twice, double-counts every counter.
    pub(crate) fn record(&self, labels: &SchedulerLabels, stats: &SchedulerStats) {
        self.scalars.record(labels, stats);

        for domain in &stats.domain_stats {
            let at = labels.with(DomainLabel {
                domain: domain.domain.clone(),
            });
            self.scheduler_domain_active_credits
                .get_or_create(&at)
                .set(domain.active_credits);
            self.scheduler_domain_peak_credits
                .get_or_create(&at)
                .set(domain.peak_credits);
            self.scheduler_domain_backpressure
                .get_or_create(&at)
                .inc_by(domain.backpressure_events);
            self.scheduler_domain_reclaimed_credits
                .get_or_create(&at)
                .inc_by(domain.reclaimed_credits);
            self.scheduler_domain_completed_batches
                .get_or_create(&at)
                .inc_by(domain.completed_batches);
            let kind = |kind| {
                labels.with(DomainKindLabel {
                    domain: domain.domain.clone(),
                    kind,
                })
            };
            for (outcome, calls) in [
                ("launched", domain.launched_calls),
                ("completed", domain.completed_calls),
                ("predicated", domain.predicated_calls),
                ("error", domain.error_calls),
            ] {
                self.scheduler_domain_calls
                    .get_or_create(&kind(outcome))
                    .inc_by(calls);
            }
            for (phase, us) in [
                ("queue", domain.queue_us),
                ("launch", domain.launch_us),
                ("device", domain.device_us),
                ("completion", domain.completion_us),
            ] {
                self.scheduler_domain_time_us
                    .get_or_create(&kind(phase))
                    .inc_by(us);
            }
        }

        // The worker counters' breakdowns: the reporter sends `None` when no
        // worker counter changed in the interval, and then none of them moves.
        let Some(worker) = &stats.worker_forward_stats else {
            return;
        };
        let mode = |mode: &String| labels.with(ModeLabel { mode: mode.clone() });
        for (name, count) in &worker.mode_counts {
            self.worker_forward_mode_counts
                .get_or_create(&mode(name))
                .inc_by(*count);
        }
        for (name, tokens) in &worker.mode_tokens {
            self.worker_forward_mode_tokens
                .get_or_create(&mode(name))
                .inc_by(*tokens);
        }
        for (name, us) in &worker.mode_us {
            self.worker_forward_mode_us
                .get_or_create(&mode(name))
                .inc_by(*us);
        }
        for (name, count) in &worker.cuda_graph_runtime_mode_counts {
            self.worker_graph_runtime_mode_counts
                .get_or_create(&mode(name))
                .inc_by(*count);
        }
        for (backend, count) in &worker.attention_backend_counts {
            self.worker_attention_backend_counts
                .get_or_create(&labels.with(BackendLabel {
                    backend: backend.clone(),
                }))
                .inc_by(*count);
        }
        for (path, count) in &worker.spec_verify_path_counts {
            self.worker_spec_verify_path_counts
                .get_or_create(&labels.with(PathLabel { path: path.clone() }))
                .inc_by(*count);
        }
        for (time, us) in &worker.stream_us {
            self.worker_stream_us
                .get_or_create(&labels.with(StreamLabel {
                    stream: time.stream.clone(),
                    state: time.state.clone(),
                }))
                .inc_by(*us);
        }
        for (call, count) in &worker.blocking_calls {
            self.worker_blocking_calls
                .get_or_create(&labels.with(BlockingLabel {
                    thread: call.thread.clone(),
                    call: call.call.clone(),
                }))
                .inc_by(*count);
        }
        for (step, histogram) in &worker.fence_us {
            self.worker_fence_step_seconds
                .get_or_create(&labels.with(FenceLabel {
                    stage: step.stage.clone(),
                    module: step.module.clone(),
                }))
                .add(histogram);
        }
    }
}

#[cfg(test)]
mod tests {
    use proto::report::DomainSchedulerStats;

    use super::*;

    /// The labels of scheduler `scheduler` of replica 0.
    fn labels(scheduler: u32) -> SchedulerLabels {
        SchedulerLabels {
            model_name: "model".to_owned(),
            replica: 0,
            scheduler,
        }
    }

    /// Per-domain statistics are set and added under their domain, with the call and phase-time
    /// breakdowns under their kind.
    #[test]
    fn domain_accounting_is_recorded_under_its_domain_and_kind() {
        let metrics = SchedulerMetrics::register(&mut Registry::default());
        let stats = SchedulerStats {
            domain_stats: vec![DomainSchedulerStats {
                domain: "decode".to_owned(),
                active_credits: 2,
                peak_credits: 5,
                launched_calls: 7,
                completed_calls: 6,
                error_calls: 1,
                backpressure_events: 3,
                reclaimed_credits: 6,
                completed_batches: 4,
                queue_us: 11,
                launch_us: 13,
                device_us: 17,
                completion_us: 19,
                ..Default::default()
            }],
            ..Default::default()
        };

        metrics.record(&labels(0), &stats);

        let decode = labels(0).with(DomainLabel {
            domain: "decode".to_owned(),
        });
        let kind = |kind| {
            labels(0).with(DomainKindLabel {
                domain: "decode".to_owned(),
                kind,
            })
        };
        assert_eq!(
            metrics
                .scheduler_domain_active_credits
                .get_or_create(&decode)
                .get(),
            2
        );
        assert_eq!(
            metrics
                .scheduler_domain_completed_batches
                .get_or_create(&decode)
                .get(),
            4
        );
        assert_eq!(
            metrics
                .scheduler_domain_calls
                .get_or_create(&kind("error"))
                .get(),
            1
        );
        assert_eq!(
            metrics
                .scheduler_domain_time_us
                .get_or_create(&kind("device"))
                .get(),
            17
        );
    }

    /// Interval counts add up over reports, under the scheduler that reported them, while a
    /// gauge takes the latest report's value.
    #[test]
    fn counters_add_each_interval_and_gauges_take_the_last() {
        let metrics = SchedulerMetrics::register(&mut Registry::default());
        let first = SchedulerStats {
            num_running_reqs: 3,
            transfers: 2,
            failed_transfers: 1,
            transferred_blocks: 9,
            calls_without_blocks: 3,
            preempted_requests: 2,
            evicted_blocks: 7,
            given_up_draft_blocks: 3,
            ..Default::default()
        };
        let second = SchedulerStats {
            num_running_reqs: 1,
            ..first.clone()
        };

        metrics.record(&labels(1), &first);
        metrics.record(&labels(1), &second);

        let scalars = &metrics.scalars;
        let at = labels(1);
        assert_eq!(scalars.scheduler_running.get_or_create(&at).get(), 1);
        for (family, total) in [
            (&scalars.request_transfers, 4),
            (&scalars.request_transfers_failed, 2),
            (&scalars.request_transfer_blocks, 18),
            (&scalars.scheduler_calls_without_blocks, 6),
            (&scalars.scheduler_preempted_requests, 4),
            (&scalars.scheduler_evicted_blocks, 14),
            (&scalars.scheduler_given_up_draft_blocks, 6),
        ] {
            assert_eq!(family.get_or_create(&at).get(), total);
        }
        assert_eq!(
            scalars.request_transfers.get_or_create(&labels(0)).get(),
            0,
            "another scheduler's series is its own"
        );
    }

    /// The worker counters move only in a report that carries them.
    #[test]
    fn worker_counters_move_only_with_a_worker_report() {
        let metrics = SchedulerMetrics::register(&mut Registry::default());
        metrics.record(&labels(0), &SchedulerStats::default());
        let worker = proto::report::ForwardStats {
            attention_launches: 5,
            spec_verify_rows: 2,
            spec_verify_committed_tokens: 5,
            ..Default::default()
        };
        let carried = SchedulerStats {
            worker_forward_stats: Some(worker),
            ..Default::default()
        };
        metrics.record(&labels(0), &carried);

        let scalars = &metrics.scalars;
        assert_eq!(
            scalars
                .worker_attention_launches
                .get_or_create(&labels(0))
                .get(),
            5
        );
        assert_eq!(
            scalars
                .spec_decode_accepted_length
                .get_or_create(&labels(0))
                .get(),
            2.5
        );
    }
}
