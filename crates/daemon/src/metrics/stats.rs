//! Recording finished requests into the request families.
//!
//! A [`StatLogger`] folds one request's output events and records the request once into the
//! `RequestMetrics` families: its outcome, its latency, and its token usage when it finished.
//! [`RequestStats`] is the same recorder around a request's output stream; a response body that
//! drives one engine event at a time (design §7.5.1) reports each output to a `StatLogger`
//! itself, through `api::events::Stats`.

use std::pin::Pin;
use std::task::{Context, Poll, ready};
use std::time::Instant;

use futures::{Stream, StreamExt as _};
use tracing::info;

use crate::api::output::{RequestOutput, RequestOutputStream, RequestUsage};
use crate::metrics::{METRICS, Outcome, OutcomeLabels, RequestLabels, RequestMetrics};

/// How many requests ended since the daemon started, and how many of them
/// failed, which an epoch's trial reads its failure rate from.
pub(crate) struct Outcomes {
    pub(crate) ended: std::sync::atomic::AtomicU64,
    pub(crate) failed: std::sync::atomic::AtomicU64,
}

/// Every request's outcome, as [`StatLogger`] records it.
pub(crate) static OUTCOMES: Outcomes = Outcomes {
    ended: std::sync::atomic::AtomicU64::new(0),
    failed: std::sync::atomic::AtomicU64::new(0),
};

/// The output stream of one request a scheduler accepted, recording the
/// request into the request metric families once.
///
/// Every event passes through unchanged. The request is recorded when its
/// stream ends: under its outcome when it finished, as an abort when the stream
/// was dropped before its terminal event (a client that disconnected), and as
/// an error when it failed, its stream yielded an error, or its stream ended
/// without a terminal event. A rejected request, which consumed no engine
/// work, is not recorded. The token metrics count only a request that
/// finished.
pub(crate) struct RequestStats {
    inner: RequestOutputStream,
    logger: StatLogger,
}

impl RequestStats {
    /// Wraps the output of a request accepted at `started`, recorded under
    /// `labels`.
    pub(crate) fn new(
        inner: RequestOutputStream,
        labels: RequestLabels,
        started: Instant,
        log: Option<String>,
    ) -> Self {
        Self {
            inner,
            logger: StatLogger::new(labels, started, log),
        }
    }
}

impl Stream for RequestStats {
    type Item = Result<RequestOutput, crate::api::output::OutputError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let next = ready!(self.inner.poll_next_unpin(cx));
        match &next {
            Some(Ok(event)) => crate::api::events::Stats::observe(&mut self.logger, event),
            Some(Err(_)) | None => self.logger.record(Outcome::Error, None),
        }
        Poll::Ready(next)
    }
}

/// One request's statistics, folded from its output events and recorded once
/// when the request ends; a recorder dropped before then records an abort.
pub(crate) struct StatLogger {
    labels: RequestLabels,
    started: Instant,
    /// The request whose summary line is logged when it ends, when request
    /// logging is on.
    log: Option<String>,
    recorded: bool,
}

impl StatLogger {
    /// A recorder for a request accepted at `started`, recorded under
    /// `labels`.
    pub(crate) fn new(labels: RequestLabels, started: Instant, log: Option<String>) -> Self {
        Self {
            labels,
            started,
            log,
            recorded: false,
        }
    }

    /// Records the request as `outcome`, with the usage of a finished request,
    /// unless it was recorded already.
    ///
    /// The end-to-end latency is the time since the request started on this
    /// daemon's monotonic clock; the queue time is the engine's own measure,
    /// which a finished request's usage carries.
    fn record(&mut self, outcome: Outcome, usage: Option<&RequestUsage>) {
        if std::mem::replace(&mut self.recorded, true) {
            return;
        }
        // A request the client abandoned neither served nor failed.
        if outcome != Outcome::Abort {
            OUTCOMES
                .ended
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if outcome == Outcome::Error {
                OUTCOMES
                    .failed
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let latency = self.started.elapsed().as_secs_f64();
        let metrics = &METRICS.request;
        metrics
            .request_outcomes
            .get_or_create(&OutcomeLabels {
                served: self.labels.clone(),
                outcome,
            })
            .inc();
        metrics
            .e2e_request_latency_seconds
            .get_or_create(&self.labels)
            .observe(latency);
        if let Some(usage) = usage {
            record_usage(metrics, &self.labels, usage);
            METRICS.log.usage(
                u64::from(usage.prompt_tokens),
                u64::from(usage.output_tokens),
            );
        }
        if let Some(request_id) = &self.log {
            info!(
                %request_id,
                outcome = outcome.name(),
                prompt_tokens = usage.map_or(0, |usage| usage.prompt_tokens),
                output_tokens = usage.map_or(0, |usage| usage.output_tokens),
                latency_s = latency,
                "request finished"
            );
        }
    }
}

impl crate::api::events::Stats for StatLogger {
    fn started(&self) -> Instant {
        self.started
    }

    fn engine(&self) -> u32 {
        self.labels.replica
    }

    fn observe(&mut self, output: &RequestOutput) {
        match output {
            RequestOutput::Finished { reason, usage, .. } => {
                self.record(Outcome::of(reason), Some(usage));
            }
            RequestOutput::Failed { .. } => self.record(Outcome::Error, None),
            RequestOutput::Rejected { .. } => self.recorded = true,
            _ => {}
        }
    }

    fn abandoned(&mut self) {
        self.record(Outcome::Abort, None);
    }
}

impl Drop for StatLogger {
    /// Records a request whose output ended before its terminal event as
    /// aborted, so a handed-over, abandoned or dropped response is counted
    /// once.
    fn drop(&mut self) {
        self.record(Outcome::Abort, None);
    }
}

/// Records the final usage of one request into the request metric families.
///
/// Its prompt and generated tokens are added to the cumulative token counters
/// and observed in the per-request token histograms. Time to first token is
/// measured from the request's start to its first visible output, and
/// observed only for a request that generated tokens and produced visible
/// output; the queue time is observed when the engine measured it.
fn record_usage(metrics: &RequestMetrics, labels: &RequestLabels, usage: &RequestUsage) {
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
    if let Some(queue_us) = usage.timings.queue_us {
        metrics
            .request_queue_time_seconds
            .get_or_create(labels)
            .observe(seconds(queue_us));
    }
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
