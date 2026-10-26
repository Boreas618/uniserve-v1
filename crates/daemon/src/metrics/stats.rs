//! Prometheus metrics for engine-reported scheduler statistics and finished
//! requests.
//!
//! `Leader::new` runs a once-per-second task per scheduler that asks the
//! core's `SchedulerStatsReporter` for a `SchedulerStats` snapshot and passes
//! it to [`record_scheduler_stats`] under the core's data-parallel rank. The
//! reporter already turns cumulative scheduler counters into per-interval
//! increments; this module only maps fields onto the `SchedulerMetrics`
//! families.
//!
//! [`RequestStats`] wraps the output stream of every request a scheduler
//! accepted and records the request once into the `RequestMetrics` families:
//! its outcome, and its token usage when it finished.

use std::pin::Pin;
use std::task::{Context, Poll, ready};
use std::time::Instant;

use crate::metrics::{
    EngineBackendLabels, EngineBlockingLabels, EngineComponentLabels, EngineDomainKindLabels,
    EngineDomainLabels, EngineLabels, EngineModeLabels, EnginePathLabels, EngineStreamLabels,
    FinishedReasonLabels, METRICS, RequestMetrics, SchedulerMetrics,
};
use futures::{Stream, StreamExt as _};
use tracing::info;

use crate::output::{RequestOutput, RequestOutputStream, RequestUsage};
use proto::report::SchedulerStats;

/// Records the scheduler-stats-backed metrics for one engine at one point in
/// time.
///
/// Interval-delta fields of `stats` are added to Prometheus counters, while
/// point-in-time values (running and waiting requests, cache usage, active
/// credits) and lifetime maxima (`max_queue_wait_us`, `peak_credits`) are set
/// on gauges. `stats` must therefore be an interval snapshot: passing a
/// cumulative one, or the same snapshot twice, double-counts every counter.
pub(crate) fn record_scheduler_stats(
    metrics: &SchedulerMetrics,
    model_name: impl Into<String>,
    engine: u32,
    stats: &SchedulerStats,
) {
    let model_name = model_name.into();
    let labels = EngineLabels {
        model_name: model_name.clone(),
        engine,
    };

    // Request-level scheduler state, admissions, and queue wait.
    metrics
        .scheduler_running
        .get_or_create(&labels)
        .set(stats.num_running_reqs);
    metrics
        .scheduler_waiting
        .get_or_create(&labels)
        .set(stats.num_waiting_reqs);
    metrics
        .prefix_cache_usage
        .get_or_create(&labels)
        .set(stats.prefix_cache_usage);
    metrics
        .scheduler_admitted
        .get_or_create(&labels)
        .inc_by(stats.num_admitted_reqs);
    metrics
        .scheduler_queue_wait_us
        .get_or_create(&labels)
        .inc_by(stats.queue_wait_us_total);
    metrics
        .scheduler_queue_wait_max_us
        .get_or_create(&labels)
        .set(stats.max_queue_wait_us);

    // Per-execution-domain accounting: credit gauges, then call, pressure,
    // and phase-time counters keyed by the domain name.
    for domain in &stats.domain_stats {
        let domain_labels = EngineDomainLabels {
            model_name: model_name.clone(),
            engine,
            domain: domain.domain.clone(),
        };
        metrics
            .scheduler_domain_active_credits
            .get_or_create(&domain_labels)
            .set(domain.active_credits);
        metrics
            .scheduler_domain_peak_credits
            .get_or_create(&domain_labels)
            .set(domain.peak_credits);
        for (kind, value) in [
            ("launched", domain.launched_calls),
            ("completed", domain.completed_calls),
            ("predicated", domain.predicated_calls),
            ("error", domain.error_calls),
        ] {
            metrics
                .scheduler_domain_calls
                .get_or_create(&EngineDomainKindLabels {
                    model_name: model_name.clone(),
                    engine,
                    domain: domain.domain.clone(),
                    kind: kind.to_string(),
                })
                .inc_by(value);
        }
        metrics
            .scheduler_domain_backpressure
            .get_or_create(&domain_labels)
            .inc_by(domain.backpressure_events);
        metrics
            .scheduler_domain_reclaimed_credits
            .get_or_create(&domain_labels)
            .inc_by(domain.reclaimed_credits);
        metrics
            .scheduler_domain_completed_batches
            .get_or_create(&domain_labels)
            .inc_by(domain.completed_batches);
        for (kind, value) in [
            ("queue", domain.queue_us),
            ("launch", domain.launch_us),
            ("device", domain.device_us),
            ("completion", domain.completion_us),
        ] {
            metrics
                .scheduler_domain_time_us
                .get_or_create(&EngineDomainKindLabels {
                    model_name: model_name.clone(),
                    engine,
                    domain: domain.domain.clone(),
                    kind: kind.to_string(),
                })
                .inc_by(value);
        }
    }

    // Prefix-cache query and hit counters, both in tokens.
    metrics
        .prefix_cache_queries
        .get_or_create(&labels)
        .inc_by(stats.prefix_cache_stats.base.queries);
    metrics
        .prefix_cache_hits
        .get_or_create(&labels)
        .inc_by(stats.prefix_cache_stats.base.hits);

    // Worker-local forward/kernel counters. The reporter sends `None` when no
    // worker counter changed in the interval, and then none of these families
    // is touched.
    if let Some(worker_stats) = &stats.worker_forward_stats {
        for (mode, count) in &worker_stats.mode_counts {
            metrics
                .worker_forward_mode_counts
                .get_or_create(&EngineModeLabels {
                    model_name: model_name.clone(),
                    engine,
                    mode: mode.clone(),
                })
                .inc_by(*count);
        }
        for (mode, tokens) in &worker_stats.mode_tokens {
            metrics
                .worker_forward_mode_tokens
                .get_or_create(&EngineModeLabels {
                    model_name: model_name.clone(),
                    engine,
                    mode: mode.clone(),
                })
                .inc_by(*tokens);
        }
        for (mode, us) in &worker_stats.mode_us {
            metrics
                .worker_forward_mode_us
                .get_or_create(&EngineModeLabels {
                    model_name: model_name.clone(),
                    engine,
                    mode: mode.clone(),
                })
                .inc_by(*us);
        }
        for (component, us) in &worker_stats.component_us {
            metrics
                .worker_forward_component_us
                .get_or_create(&EngineComponentLabels {
                    model_name: model_name.clone(),
                    engine,
                    component: component.clone(),
                })
                .inc_by(*us);
        }
        for (key, us) in &worker_stats.stream_us {
            // Keys are `{stream}/{state}` (`ForwardStats::stream_us`).
            let (stream, state) = key.rsplit_once('/').unwrap_or((key.as_str(), ""));
            metrics
                .worker_stream_us
                .get_or_create(&EngineStreamLabels {
                    model_name: model_name.clone(),
                    engine,
                    stream: stream.to_owned(),
                    state: state.to_owned(),
                })
                .inc_by(*us);
        }
        for (key, count) in &worker_stats.blocking_calls {
            // Keys are `{thread}/{call}` (`ForwardStats::blocking_calls`).
            let (thread, call) = key.split_once('/').unwrap_or((key.as_str(), ""));
            metrics
                .worker_blocking_calls
                .get_or_create(&EngineBlockingLabels {
                    model_name: model_name.clone(),
                    engine,
                    thread: thread.to_owned(),
                    call: call.to_owned(),
                })
                .inc_by(*count);
        }
        for (backend, count) in &worker_stats.attention_backend_counts {
            metrics
                .worker_attention_backend_counts
                .get_or_create(&EngineBackendLabels {
                    model_name: model_name.clone(),
                    engine,
                    backend: backend.clone(),
                })
                .inc_by(*count);
        }
        for (mode, count) in &worker_stats.cuda_graph_runtime_mode_counts {
            metrics
                .worker_cuda_graph_runtime_mode_counts
                .get_or_create(&EngineModeLabels {
                    model_name: model_name.clone(),
                    engine,
                    mode: mode.clone(),
                })
                .inc_by(*count);
        }
        metrics
            .worker_attention_launches
            .get_or_create(&labels)
            .inc_by(worker_stats.attention_launches);
        metrics
            .worker_attention_us
            .get_or_create(&labels)
            .inc_by(worker_stats.attention_us);
        metrics
            .worker_cuda_graph_captures
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_captures);
        metrics
            .worker_cuda_graph_replays
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_replays);
        metrics
            .worker_cuda_graph_misses
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_misses);
        metrics
            .worker_cuda_graph_fallbacks
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_fallbacks);
        metrics
            .worker_cuda_graph_unpadded_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_unpadded_tokens);
        metrics
            .worker_cuda_graph_padded_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_padded_tokens);
        metrics
            .worker_text_decode_token_relay_hits
            .get_or_create(&labels)
            .inc_by(worker_stats.text_decode_token_relay_hits);
        metrics
            .worker_text_decode_token_relay_misses
            .get_or_create(&labels)
            .inc_by(worker_stats.text_decode_token_relay_misses);
        metrics
            .worker_text_decode_position_relay_hits
            .get_or_create(&labels)
            .inc_by(worker_stats.text_decode_position_relay_hits);
        metrics
            .worker_text_decode_position_relay_misses
            .get_or_create(&labels)
            .inc_by(worker_stats.text_decode_position_relay_misses);
        metrics
            .worker_flashinfer_decode_plan_calls
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_plan_calls);
        metrics
            .worker_flashinfer_decode_plan_reuses
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_plan_reuses);
        metrics
            .worker_flashinfer_decode_plan_rows
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_plan_rows);
        metrics
            .worker_flashinfer_decode_plan_indices
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_plan_indices);
        metrics
            .worker_flashinfer_decode_graph_plan_calls
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_graph_plan_calls);
        metrics
            .worker_flashinfer_decode_graph_plan_reuses
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_graph_plan_reuses);
        metrics
            .worker_spec_verify_rows
            .get_or_create(&labels)
            .inc_by(worker_stats.spec_verify_rows);
        metrics
            .worker_spec_verify_draft_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.spec_verify_draft_tokens);
        metrics
            .worker_spec_verify_accepted_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.spec_verify_accepted_tokens);
        metrics
            .worker_spec_verify_rejected_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.spec_verify_rejected_tokens);
        metrics
            .worker_spec_verify_committed_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.spec_verify_committed_tokens);
        for (path, count) in &worker_stats.spec_verify_path_counts {
            metrics
                .worker_spec_verify_path_counts
                .get_or_create(&EnginePathLabels {
                    model_name: model_name.clone(),
                    engine,
                    path: path.clone(),
                })
                .inc_by(*count);
        }
    }

    // Worker execution and host-observed round-trip time, in integer
    // microseconds; `batch_timing_count` counts the completed batches they
    // cover and is the denominator for per-batch averages.
    metrics
        .worker_exec_us
        .get_or_create(&labels)
        .inc_by(stats.worker_exec_us);
    metrics
        .batch_roundtrip_us
        .get_or_create(&labels)
        .inc_by(stats.batch_roundtrip_us);
    metrics
        .batch_timing_count
        .get_or_create(&labels)
        .inc_by(stats.batch_count);
}

/// The output stream of one request a scheduler accepted, recording the
/// request into the request metric families once.
///
/// Every event passes through unchanged. The request is recorded when its
/// stream ends: under its finish reason when it finished (`abort` for a
/// cancelled or aborted request), under `abort` when the stream was dropped
/// before its terminal event (a client that disconnected), and under `error`
/// when it failed, its stream yielded an error, or its stream ended without a
/// terminal event. A rejected request, which consumed no engine work, is not
/// recorded. The token metrics count only a request that finished.
pub(crate) struct RequestStats {
    inner: RequestOutputStream,
    labels: EngineLabels,
    started: Instant,
    /// The request whose summary line is logged when it ends, when request
    /// logging is on.
    log: Option<String>,
    /// Queue time from the engine's scheduling timestamps, kept for a request
    /// that ends before it finishes.
    queue_us: Option<u64>,
    recorded: bool,
}

impl RequestStats {
    /// Wraps the output of a request accepted at `started`, recorded under
    /// `labels`.
    pub(crate) fn new(
        inner: RequestOutputStream,
        labels: EngineLabels,
        started: Instant,
        log: Option<String>,
    ) -> Self {
        Self {
            inner,
            labels,
            started,
            log,
            queue_us: None,
            recorded: false,
        }
    }

    /// Folds one output event into the request's statistics, recording the
    /// request at its terminal event.
    fn observe(&mut self, event: &RequestOutput) {
        match event {
            // `queued_at` and `scheduled_at` are engine wall-clock UNIX
            // seconds; a negative difference is clamped to zero.
            RequestOutput::Scheduled {
                queued_at,
                scheduled_at,
            } => {
                self.queue_us = Some(((scheduled_at - queued_at).max(0.0) * 1e6) as u64);
            }
            RequestOutput::Finished { reason, usage } => self.record(reason.as_str(), Some(usage)),
            RequestOutput::Failed { .. } => self.record("error", None),
            RequestOutput::Rejected { .. } => self.recorded = true,
            _ => {}
        }
    }

    /// Records the request under `finished_reason`, with the usage of a
    /// finished request, unless it was recorded already.
    ///
    /// The end-to-end latency is the larger of the reported total and the
    /// time since the request started.
    fn record(&mut self, finished_reason: &'static str, usage: Option<&RequestUsage>) {
        if std::mem::replace(&mut self.recorded, true) {
            return;
        }
        let elapsed_us = self.started.elapsed().as_micros() as u64;
        let metrics = &METRICS.request;
        let total_us = usage.map_or(elapsed_us, |usage| usage.timings.total_us.max(elapsed_us));
        let queue_us = usage
            .and_then(|usage| usage.timings.queue_us)
            .or(self.queue_us);
        metrics
            .request_success
            .get_or_create(&FinishedReasonLabels {
                model_name: self.labels.model_name.clone(),
                engine: self.labels.engine,
                finished_reason,
            })
            .inc();
        metrics
            .e2e_request_latency_seconds
            .get_or_create(&self.labels)
            .observe(seconds(total_us));
        if let Some(queue_us) = queue_us {
            metrics
                .request_queue_time_seconds
                .get_or_create(&self.labels)
                .observe(seconds(queue_us));
        }
        if let Some(usage) = usage {
            record_usage(metrics, &self.labels, usage);
        }
        if let Some(request_id) = &self.log {
            info!(
                %request_id,
                finished_reason,
                prompt_tokens = usage.map_or(0, |usage| usage.prompt_tokens),
                output_tokens = usage.map_or(0, |usage| usage.output_tokens),
                latency_s = seconds(total_us),
                "request finished"
            );
        }
    }
}

impl Stream for RequestStats {
    type Item = Result<RequestOutput, crate::output::OutputError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let next = ready!(self.inner.poll_next_unpin(cx));
        match &next {
            Some(Ok(event)) => self.observe(event),
            Some(Err(_)) | None => self.record("error", None),
        }
        Poll::Ready(next)
    }
}

impl Drop for RequestStats {
    /// Records a request whose stream was dropped before its terminal event
    /// as aborted.
    fn drop(&mut self) {
        self.record("abort", None);
    }
}

/// Records the final usage of one request into the request metric families.
///
/// Its prompt and generated tokens are added to the cumulative token counters
/// and observed in the per-request token histograms. Time to first token is
/// measured from the request's start to its first visible output, and
/// observed only for a request that generated tokens and produced visible
/// output.
fn record_usage(metrics: &RequestMetrics, labels: &EngineLabels, usage: &RequestUsage) {
    let prompt_tokens = u64::from(usage.prompt_tokens);
    let generation_tokens = u64::from(usage.output_tokens);

    metrics
        .prompt_tokens
        .get_or_create(labels)
        .inc_by(prompt_tokens);
    metrics
        .generation_tokens
        .get_or_create(labels)
        .inc_by(generation_tokens);

    metrics
        .request_prompt_tokens
        .get_or_create(labels)
        .observe(prompt_tokens as f64);
    metrics
        .request_generation_tokens
        .get_or_create(labels)
        .observe(generation_tokens as f64);
    // A request samples one sequence, so its largest sequence generation is
    // its own.
    metrics
        .request_max_num_generation_tokens
        .get_or_create(labels)
        .observe(generation_tokens as f64);

    if generation_tokens > 0
        && let Some(first_output_us) = usage.timings.first_visible_output_us
    {
        metrics
            .time_to_first_token_seconds
            .get_or_create(labels)
            .observe(seconds(first_output_us));
    }
}

/// Converts a microsecond duration to the seconds the histograms observe.
fn seconds(microseconds: u64) -> f64 {
    microseconds as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Metrics;
    use proto::report::DomainSchedulerStats;

    // Per-domain scheduler statistics reach the rendered exposition under
    // their `domain` label, with call and phase-time breakdowns under `kind`.
    #[test]
    fn domain_accounting_reaches_openmetrics() {
        let metrics = Metrics::new();
        let stats = SchedulerStats {
            domain_stats: vec![DomainSchedulerStats {
                domain: "decode".to_string(),
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

        record_scheduler_stats(&metrics.scheduler, "model", 0, &stats);
        let rendered = metrics.render().expect("metrics render");

        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:scheduler_domain_active_credits")
                && line.contains("domain=\"decode\"")
                && line.ends_with(" 2")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:scheduler_domain_calls_total")
                && line.contains("domain=\"decode\"")
                && line.contains("kind=\"error\"")
                && line.ends_with(" 1")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:scheduler_domain_time_us_total")
                && line.contains("domain=\"decode\"")
                && line.contains("kind=\"device\"")
                && line.ends_with(" 17")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:scheduler_domain_completed_batches")
                && line.contains("domain=\"decode\"")
                && line.ends_with(" 4")
        }));
    }
}
