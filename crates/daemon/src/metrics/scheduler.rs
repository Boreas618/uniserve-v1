//! Scheduler, cache, and worker execution metrics.
//!
//! The server's `record_scheduler_stats` writes every family here from one
//! `SchedulerStats` snapshot per reporting interval: it sets gauges and
//! lifetime values, and adds interval deltas to counters. The worker-local
//! forward/kernel families, which come from `ForwardStats`, change only in
//! intervals whose snapshot carries `worker_forward_stats`.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;

use crate::metrics::{F64Gauge, U64Counter, U64Gauge};

/// Labels identifying one model and scheduler.
///
/// The in-process engine client publishes its scheduler snapshots as engine
/// `0`.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
}

/// Labels identifying an engine execution path.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EnginePathLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
    /// Speculative-verification resolution path reported by the worker.
    pub path: String,
}

/// Labels identifying a worker runtime mode.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineModeLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
    /// Worker runtime mode.
    pub mode: String,
}

/// Labels identifying one component of the fence probes' step durations.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineFenceLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
    /// The steps' stage: their call kind, such as `prefill` or `decode`.
    pub stage: String,
    /// The component (`<component>` of its declared function), or `step`
    /// for the step's span between its markers.
    pub component: String,
}

/// Labels identifying one state of a worker stream's device time.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineStreamLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
    /// Worker stream: `device`, the rank's primary stream, or `copy`, the
    /// stream of its page copies through the host KV cache.
    pub stream: String,
    /// `busy`: device time the stream's operations spanned; `gap`: idle time
    /// of the stream between them; `late`: time of records too far out of
    /// order to merge.
    pub state: String,
}

/// Labels identifying a driver call that made a worker thread wait for the
/// device.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineBlockingLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
    /// Worker thread (`builder`, `control`).
    pub thread: String,
    /// Driver call.
    pub call: String,
}

/// Labels identifying an instrumented engine component.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineComponentLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
    /// Instrumented worker component.
    pub component: String,
}

/// Labels identifying a scheduler execution domain.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineDomainLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
    /// Scheduler execution domain: `prefill`, `decode`, or `flow`.
    pub domain: String,
}

/// Labels identifying an event kind within a scheduler domain.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineDomainKindLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
    /// Scheduler execution domain: `prefill`, `decode`, or `flow`.
    pub domain: String,
    /// Call outcome (`launched`, `completed`, `predicated`, `error`) for
    /// `scheduler_domain_calls`, or timing phase (`queue`, `launch`, `device`,
    /// `completion`) for `scheduler_domain_time_us`.
    pub kind: String,
}

/// Labels identifying the attention backend that served a kernel launch.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineBackendLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
    /// Attention kernel backend family.
    pub backend: String,
}

/// Scheduler/batch-scoped Prometheus families exported from `SchedulerStats`.
///
/// The gauge, lifetime, or interval-delta semantics of each source value are
/// documented on `SchedulerStats`, `DomainSchedulerStats`, and `ForwardStats`.
pub struct SchedulerMetrics {
    // Scheduler state gauges.
    /// Requests currently participating in model execution.
    pub scheduler_running: Family<EngineLabels, U64Gauge>,
    /// Requests waiting for scheduler admission.
    pub scheduler_waiting: Family<EngineLabels, U64Gauge>,
    /// Fraction of usable cache blocks that are not free; `1.0` means full.
    pub prefix_cache_usage: Family<EngineLabels, F64Gauge>,
    /// Requests admitted into resident execution.
    pub scheduler_admitted: Family<EngineLabels, U64Counter>,
    /// Cumulative admission queue time in microseconds.
    pub scheduler_queue_wait_us: Family<EngineLabels, U64Counter>,
    /// Maximum admission queue time in microseconds observed since the
    /// scheduler started.
    pub scheduler_queue_wait_max_us: Family<EngineLabels, U64Gauge>,
    /// Active call credits grouped by execution domain.
    pub scheduler_domain_active_credits: Family<EngineDomainLabels, U64Gauge>,
    /// Peak active call credits grouped by execution domain.
    pub scheduler_domain_peak_credits: Family<EngineDomainLabels, U64Gauge>,
    /// Call lifecycle events grouped by domain and outcome.
    pub scheduler_domain_calls: Family<EngineDomainKindLabels, U64Counter>,
    /// Non-blocking resource-pressure events grouped by domain.
    pub scheduler_domain_backpressure: Family<EngineDomainLabels, U64Counter>,
    /// Reclaimed call credits grouped by domain.
    pub scheduler_domain_reclaimed_credits: Family<EngineDomainLabels, U64Counter>,
    /// Completed batches grouped by domain.
    pub scheduler_domain_completed_batches: Family<EngineDomainLabels, U64Counter>,
    /// Cumulative domain time in microseconds grouped by lifecycle phase.
    pub scheduler_domain_time_us: Family<EngineDomainKindLabels, U64Counter>,

    // Local prefix-cache counters, both in tokens.
    /// Prefix-cache query tokens.
    pub prefix_cache_queries: Family<EngineLabels, U64Counter>,
    /// Prefix-cache hit tokens.
    pub prefix_cache_hits: Family<EngineLabels, U64Counter>,

    // Worker-local forward/kernel counters.
    /// Attention kernel launches.
    pub worker_attention_launches: Family<EngineLabels, U64Counter>,
    /// Cumulative attention kernel time in microseconds.
    pub worker_attention_us: Family<EngineLabels, U64Counter>,
    /// Attention launches grouped by kernel backend.
    pub worker_attention_backend_counts: Family<EngineBackendLabels, U64Counter>,
    /// CUDA graph dispatches grouped by runtime mode.
    pub worker_cuda_graph_runtime_mode_counts: Family<EngineModeLabels, U64Counter>,
    /// Forward calls grouped by runtime mode.
    pub worker_forward_mode_counts: Family<EngineModeLabels, U64Counter>,
    /// Forward query tokens grouped by runtime mode; modes without a token
    /// notion (encoders, decoders, module calls) have no series.
    pub worker_forward_mode_tokens: Family<EngineModeLabels, U64Counter>,
    /// Forward wall time grouped by runtime mode.
    pub worker_forward_mode_us: Family<EngineModeLabels, U64Counter>,
    /// Forward wall time grouped by worker component.
    pub worker_forward_component_us: Family<EngineComponentLabels, U64Counter>,
    /// Worker stream device time grouped by stream and state.
    pub worker_stream_us: Family<EngineStreamLabels, U64Counter>,
    /// Worker driver calls that waited for the device, by thread and call.
    pub worker_blocking_calls: Family<EngineBlockingLabels, U64Counter>,
    /// Fence probe step durations by stage and component.
    pub worker_fence_step_seconds: Family<EngineFenceLabels, super::CountedHistogram>,
    /// Fence probe records the workers lost.
    pub worker_fence_records_lost: Family<EngineLabels, U64Counter>,
    /// CUDA graph captures.
    pub worker_cuda_graph_captures: Family<EngineLabels, U64Counter>,
    /// CUDA graph replays.
    pub worker_cuda_graph_replays: Family<EngineLabels, U64Counter>,
    /// CUDA graph bucket misses.
    pub worker_cuda_graph_misses: Family<EngineLabels, U64Counter>,
    /// Executions falling back from CUDA graph replay.
    pub worker_cuda_graph_fallbacks: Family<EngineLabels, U64Counter>,
    /// Real tokens represented by CUDA graph executions.
    pub worker_cuda_graph_unpadded_tokens: Family<EngineLabels, U64Counter>,
    /// Padding tokens represented by CUDA graph executions.
    pub worker_cuda_graph_padded_tokens: Family<EngineLabels, U64Counter>,
    /// Decode tokens served from the token relay.
    pub worker_text_decode_token_relay_hits: Family<EngineLabels, U64Counter>,
    /// Decode token relay misses.
    pub worker_text_decode_token_relay_misses: Family<EngineLabels, U64Counter>,
    /// Decode positions served from the position relay.
    pub worker_text_decode_position_relay_hits: Family<EngineLabels, U64Counter>,
    /// Decode position relay misses.
    pub worker_text_decode_position_relay_misses: Family<EngineLabels, U64Counter>,
    /// FlashInfer decode plan constructions.
    pub worker_flashinfer_decode_plan_calls: Family<EngineLabels, U64Counter>,
    /// FlashInfer decode plan reuses.
    pub worker_flashinfer_decode_plan_reuses: Family<EngineLabels, U64Counter>,
    /// Rows represented by FlashInfer decode plans.
    pub worker_flashinfer_decode_plan_rows: Family<EngineLabels, U64Counter>,
    /// Page indices represented by FlashInfer decode plans.
    pub worker_flashinfer_decode_plan_indices: Family<EngineLabels, U64Counter>,
    /// FlashInfer graph-decode plan constructions.
    pub worker_flashinfer_decode_graph_plan_calls: Family<EngineLabels, U64Counter>,
    /// FlashInfer graph-decode plan reuses.
    pub worker_flashinfer_decode_graph_plan_reuses: Family<EngineLabels, U64Counter>,
    /// Rows processed by speculative verification.
    pub worker_spec_verify_rows: Family<EngineLabels, U64Counter>,
    /// Draft tokens presented for speculative verification.
    pub worker_spec_verify_draft_tokens: Family<EngineLabels, U64Counter>,
    /// Draft tokens accepted by speculative verification.
    pub worker_spec_verify_accepted_tokens: Family<EngineLabels, U64Counter>,
    /// Draft tokens rejected by speculative verification.
    pub worker_spec_verify_rejected_tokens: Family<EngineLabels, U64Counter>,
    /// Tokens committed after speculative verification.
    pub worker_spec_verify_committed_tokens: Family<EngineLabels, U64Counter>,
    /// Speculative verification rows grouped by resolution path.
    pub worker_spec_verify_path_counts: Family<EnginePathLabels, U64Counter>,

    // Batch latency uses cumulative microseconds plus a normalization count.
    /// Cumulative worker-reported batch execution time in microseconds.
    pub worker_exec_us: Family<EngineLabels, U64Counter>,
    /// Cumulative host-observed submit-to-result time in microseconds.
    pub batch_roundtrip_us: Family<EngineLabels, U64Counter>,
    /// Resolved batches contributing to cumulative latency counters.
    pub batch_timing_count: Family<EngineLabels, U64Counter>,
}

impl SchedulerMetrics {
    /// Registers every family under its exported name and help, and
    /// returns the handles; the registry keeps a clone of each, whose state
    /// it shares.
    pub fn register(registry: &mut prometheus_client::registry::Registry) -> Self {
        let scheduler_running: Family<EngineLabels, U64Gauge> = Default::default();
        registry.register(
            "uniserve:num_requests_running",
            "Number of requests in model execution batches",
            scheduler_running.clone(),
        );
        let scheduler_waiting: Family<EngineLabels, U64Gauge> = Default::default();
        registry.register(
            "uniserve:num_requests_waiting",
            "Number of requests waiting to be processed",
            scheduler_waiting.clone(),
        );
        let prefix_cache_usage: Family<EngineLabels, F64Gauge> = Default::default();
        registry.register(
            "uniserve:kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage",
            prefix_cache_usage.clone(),
        );
        let scheduler_admitted: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:num_requests_admitted",
            "Number of requests admitted into running batches.",
            scheduler_admitted.clone(),
        );
        let scheduler_queue_wait_us: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:request_queue_wait_us",
            "Total queue wait in microseconds for admitted requests.",
            scheduler_queue_wait_us.clone(),
        );
        let scheduler_queue_wait_max_us: Family<EngineLabels, U64Gauge> = Default::default();
        registry.register(
            "uniserve:request_queue_wait_max_us",
            "Maximum queue wait in microseconds observed by the scheduler.",
            scheduler_queue_wait_max_us.clone(),
        );
        let scheduler_domain_active_credits: Family<EngineDomainLabels, U64Gauge> =
            Default::default();
        registry.register(
            "uniserve:scheduler_domain_active_credits",
            "Current scheduler call credits in use by execution domain.",
            scheduler_domain_active_credits.clone(),
        );
        let scheduler_domain_peak_credits: Family<EngineDomainLabels, U64Gauge> =
            Default::default();
        registry.register(
            "uniserve:scheduler_domain_peak_credits",
            "Maximum scheduler call credits observed in use by execution domain.",
            scheduler_domain_peak_credits.clone(),
        );
        let scheduler_domain_calls: Family<EngineDomainKindLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:scheduler_domain_calls",
            "Scheduler call events by execution domain and kind: launched, completed, predicated, or error.",
            scheduler_domain_calls.clone(),
        );
        let scheduler_domain_backpressure: Family<EngineDomainLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:scheduler_domain_backpressure",
            "Nonblocking scheduler resource-pressure events by execution domain.",
            scheduler_domain_backpressure.clone(),
        );
        let scheduler_domain_reclaimed_credits: Family<EngineDomainLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:scheduler_domain_reclaimed_credits",
            "Scheduler call credits reclaimed by execution domain.",
            scheduler_domain_reclaimed_credits.clone(),
        );
        let scheduler_domain_completed_batches: Family<EngineDomainLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:scheduler_domain_completed_batches",
            "Completed batches by execution domain.",
            scheduler_domain_completed_batches.clone(),
        );
        let scheduler_domain_time_us: Family<EngineDomainKindLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:scheduler_domain_time_us",
            "Cumulative execution-domain time in microseconds by phase: queue, launch, device, or completion.",
            scheduler_domain_time_us.clone(),
        );
        let prefix_cache_queries: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:prefix_cache_queries",
            "Prefix cache queries, in terms of number of queried tokens",
            prefix_cache_queries.clone(),
        );
        let prefix_cache_hits: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:prefix_cache_hits",
            "Prefix cache hits, in terms of number of cached tokens.",
            prefix_cache_hits.clone(),
        );
        let worker_attention_launches: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_attention_launches",
            "Worker attention launches.",
            worker_attention_launches.clone(),
        );
        let worker_attention_us: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_attention_us",
            "Worker attention time in microseconds.",
            worker_attention_us.clone(),
        );
        let worker_attention_backend_counts: Family<EngineBackendLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_attention_backend_counts",
            "Worker attention launches by backend kernel family.",
            worker_attention_backend_counts.clone(),
        );
        let worker_cuda_graph_runtime_mode_counts: Family<EngineModeLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_cuda_graph_runtime_mode_counts",
            "Worker forward dispatches by CUDA-graph runtime mode.",
            worker_cuda_graph_runtime_mode_counts.clone(),
        );
        let worker_forward_mode_counts: Family<EngineModeLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_forward_mode_counts",
            "Worker forward calls by forward mode.",
            worker_forward_mode_counts.clone(),
        );
        let worker_forward_mode_tokens: Family<EngineModeLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_forward_mode_tokens",
            "Worker forward query tokens by forward mode.",
            worker_forward_mode_tokens.clone(),
        );
        let worker_forward_mode_us: Family<EngineModeLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_forward_mode_us",
            "Worker forward wall time by forward mode in microseconds.",
            worker_forward_mode_us.clone(),
        );
        let worker_forward_component_us: Family<EngineComponentLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_forward_component_us",
            "Worker forward wall time by component in microseconds.",
            worker_forward_component_us.clone(),
        );
        let worker_stream_us: Family<EngineStreamLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_stream_us",
            "Worker stream device time in microseconds from CUPTI activity records, for the \
             primary stream (device) and the host KV cache copy stream (copy): the union of the \
             kernels, copies and fills on the stream, graph nodes included (busy), the stream's \
             idle time between them (gap), and records too far out of order to merge (late).",
            worker_stream_us.clone(),
        );
        let worker_blocking_calls: Family<EngineBlockingLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_blocking_calls",
            "Driver calls that made a serving worker thread wait for the device, by thread and \
             call; a serving call of the executor that made one failed.",
            worker_blocking_calls.clone(),
        );
        let worker_fence_step_seconds: Family<EngineFenceLabels, super::CountedHistogram> =
            Default::default();
        registry.register(
            "uniserve:worker_fence_step_seconds",
            "Per-step durations from the workers' fence probes, by stage and component: a \
             step's summed durations of the component's probed module instances, or the step's \
             span between its markers for component `step`, measured on the device.",
            worker_fence_step_seconds.clone(),
        );
        let worker_fence_records_lost: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_fence_records_lost",
            "Fence probe records a worker's reader lost: overwritten before it read them.",
            worker_fence_records_lost.clone(),
        );
        let worker_cuda_graph_captures: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_cuda_graph_captures",
            "Worker CUDA graph captures.",
            worker_cuda_graph_captures.clone(),
        );
        let worker_cuda_graph_replays: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_cuda_graph_replays",
            "Worker CUDA graph replays.",
            worker_cuda_graph_replays.clone(),
        );
        let worker_cuda_graph_misses: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_cuda_graph_misses",
            "Worker CUDA graph bucket misses.",
            worker_cuda_graph_misses.clone(),
        );
        let worker_cuda_graph_fallbacks: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_cuda_graph_fallbacks",
            "Worker CUDA graph fallbacks to eager execution.",
            worker_cuda_graph_fallbacks.clone(),
        );
        let worker_cuda_graph_unpadded_tokens: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_cuda_graph_unpadded_tokens",
            "Worker CUDA graph real tokens before padding.",
            worker_cuda_graph_unpadded_tokens.clone(),
        );
        let worker_cuda_graph_padded_tokens: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_cuda_graph_padded_tokens",
            "Worker CUDA graph padding tokens.",
            worker_cuda_graph_padded_tokens.clone(),
        );
        let worker_text_decode_token_relay_hits: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_text_decode_token_relay_hits",
            "Worker decode token relay hits.",
            worker_text_decode_token_relay_hits.clone(),
        );
        let worker_text_decode_token_relay_misses: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_text_decode_token_relay_misses",
            "Worker decode token relay misses.",
            worker_text_decode_token_relay_misses.clone(),
        );
        let worker_text_decode_position_relay_hits: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_text_decode_position_relay_hits",
            "Worker decode position relay hits.",
            worker_text_decode_position_relay_hits.clone(),
        );
        let worker_text_decode_position_relay_misses: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_text_decode_position_relay_misses",
            "Worker decode position relay misses.",
            worker_text_decode_position_relay_misses.clone(),
        );
        let worker_flashinfer_decode_plan_calls: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_flashinfer_decode_plan_calls",
            "Worker FlashInfer decode plan calls.",
            worker_flashinfer_decode_plan_calls.clone(),
        );
        let worker_flashinfer_decode_plan_reuses: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_flashinfer_decode_plan_reuses",
            "Worker FlashInfer decode plan reuses.",
            worker_flashinfer_decode_plan_reuses.clone(),
        );
        let worker_flashinfer_decode_plan_rows: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_flashinfer_decode_plan_rows",
            "Worker FlashInfer decode planned rows.",
            worker_flashinfer_decode_plan_rows.clone(),
        );
        let worker_flashinfer_decode_plan_indices: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_flashinfer_decode_plan_indices",
            "Worker FlashInfer decode planned page indices.",
            worker_flashinfer_decode_plan_indices.clone(),
        );
        let worker_flashinfer_decode_graph_plan_calls: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_flashinfer_decode_graph_plan_calls",
            "Worker FlashInfer CUDA-graph decode plan calls.",
            worker_flashinfer_decode_graph_plan_calls.clone(),
        );
        let worker_flashinfer_decode_graph_plan_reuses: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_flashinfer_decode_graph_plan_reuses",
            "Worker FlashInfer CUDA-graph decode plan reuses.",
            worker_flashinfer_decode_graph_plan_reuses.clone(),
        );
        let worker_spec_verify_rows: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_spec_verify_rows",
            "Worker speculative verifier rows.",
            worker_spec_verify_rows.clone(),
        );
        let worker_spec_verify_draft_tokens: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_spec_verify_draft_tokens",
            "Worker speculative verifier draft tokens.",
            worker_spec_verify_draft_tokens.clone(),
        );
        let worker_spec_verify_accepted_tokens: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_spec_verify_accepted_tokens",
            "Worker speculative verifier accepted draft tokens.",
            worker_spec_verify_accepted_tokens.clone(),
        );
        let worker_spec_verify_rejected_tokens: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_spec_verify_rejected_tokens",
            "Worker speculative verifier rejected draft tokens.",
            worker_spec_verify_rejected_tokens.clone(),
        );
        let worker_spec_verify_committed_tokens: Family<EngineLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_spec_verify_committed_tokens",
            "Worker speculative verifier committed tokens.",
            worker_spec_verify_committed_tokens.clone(),
        );
        let worker_spec_verify_path_counts: Family<EnginePathLabels, U64Counter> =
            Default::default();
        registry.register(
            "uniserve:worker_spec_verify_path_counts",
            "Worker speculative verifier rows by verifier path.",
            worker_spec_verify_path_counts.clone(),
        );
        let worker_exec_us: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:worker_exec_us",
            "Cumulative worker-reported batch compute time in microseconds.",
            worker_exec_us.clone(),
        );
        let batch_roundtrip_us: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:batch_roundtrip_us",
            "Cumulative host-observed batch submit-to-result round-trip time in microseconds.",
            batch_roundtrip_us.clone(),
        );
        let batch_timing_count: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:batch_timing_count",
            "Number of resolved batches contributing to the batch-latency counters \
             (the denominator for per-batch averages of worker_exec_us / batch_roundtrip_us).",
            batch_timing_count.clone(),
        );
        Self {
            scheduler_running,
            scheduler_waiting,
            prefix_cache_usage,
            scheduler_admitted,
            scheduler_queue_wait_us,
            scheduler_queue_wait_max_us,
            scheduler_domain_active_credits,
            scheduler_domain_peak_credits,
            scheduler_domain_calls,
            scheduler_domain_backpressure,
            scheduler_domain_reclaimed_credits,
            scheduler_domain_completed_batches,
            scheduler_domain_time_us,
            prefix_cache_queries,
            prefix_cache_hits,
            worker_attention_launches,
            worker_attention_us,
            worker_attention_backend_counts,
            worker_cuda_graph_runtime_mode_counts,
            worker_forward_mode_counts,
            worker_forward_mode_tokens,
            worker_forward_mode_us,
            worker_forward_component_us,
            worker_stream_us,
            worker_blocking_calls,
            worker_fence_step_seconds,
            worker_fence_records_lost,
            worker_cuda_graph_captures,
            worker_cuda_graph_replays,
            worker_cuda_graph_misses,
            worker_cuda_graph_fallbacks,
            worker_cuda_graph_unpadded_tokens,
            worker_cuda_graph_padded_tokens,
            worker_text_decode_token_relay_hits,
            worker_text_decode_token_relay_misses,
            worker_text_decode_position_relay_hits,
            worker_text_decode_position_relay_misses,
            worker_flashinfer_decode_plan_calls,
            worker_flashinfer_decode_plan_reuses,
            worker_flashinfer_decode_plan_rows,
            worker_flashinfer_decode_plan_indices,
            worker_flashinfer_decode_graph_plan_calls,
            worker_flashinfer_decode_graph_plan_reuses,
            worker_spec_verify_rows,
            worker_spec_verify_draft_tokens,
            worker_spec_verify_accepted_tokens,
            worker_spec_verify_rejected_tokens,
            worker_spec_verify_committed_tokens,
            worker_spec_verify_path_counts,
            worker_exec_us,
            batch_roundtrip_us,
            batch_timing_count,
        }
    }
}
