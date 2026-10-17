//! Request lifecycle: the terminal-outcome latch of each registered request and
//! the response stream that feeds every event to the request registry, a
//! request's observable lifecycle state and statistics
//! ([`RequestStatsSnapshot`]), the runtime's aggregate lifecycle counters
//! ([`RuntimeMetricsSnapshot`]), and the precedence of external cancel and
//! abort commands over engine output.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    CacheAccounting, RequestOutput, RequestOutputStream, ResourceAccounting, Result,
    RuntimeTimings, ServeError, ServeRequestId,
};
use crate::engine_client::requests::RequestRegistry;

#[derive(Debug, Error)]
/// Failure while applying an external serving control command.
pub enum ServeControlError {
    /// The backing engine rejects or fails the control command.
    #[error(transparent)]
    Engine(#[from] crate::engine_client::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Observable lifecycle phase of one serving request.
pub enum RequestLifecycleState {
    /// The public request is being validated and tokenized.
    Compiling,
    /// The tokenized request is being submitted to the engine.
    Submitting,
    /// The engine has accepted the request.
    Accepted,
    /// The engine has scheduled model execution.
    Scheduled,
    /// Output is being streamed to the caller.
    Streaming,
    /// Cancellation has been requested.
    Cancelling,
    /// Forced abortion has been requested.
    Aborting,
    /// The request completed normally.
    Finished,
    /// The request was rejected before execution.
    Rejected,
    /// The request was cancelled.
    Cancelled,
    /// The request was forcibly aborted.
    Aborted,
    /// The request failed.
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Point-in-time lifecycle state, usage, and timing statistics for one request.
///
/// `RequestRegistry` maintains it from the public events it observes and retains it for a
/// bounded number of completed requests.
pub struct RequestStatsSnapshot {
    /// Caller-visible request identifier.
    pub request_id: ServeRequestId,
    /// Model name used for the request.
    pub served_name: String,
    /// Stable model profile description.
    pub description: String,
    /// Latest observed lifecycle phase.
    pub state: RequestLifecycleState,
    /// Number of prompt tokens submitted to the engine.
    pub prompt_tokens: u32,
    /// Number of generated tokens included in user-visible output.
    pub visible_output_tokens: u32,
    /// Number of generated tokens consumed by internal protocol sections.
    pub internal_tokens: u32,
    /// Number of generated images.
    pub image_count: u32,
    /// Total image-generation steps completed.
    pub image_steps: u32,
    /// Cache policy and pin counts.
    pub cache: CacheAccounting,
    /// Prefix cache and media resources reserved for the request.
    pub resources: ResourceAccounting,
    /// Request phase durations.
    pub timings: RuntimeTimings,
}

impl RequestStatsSnapshot {
    /// Creates a zeroed snapshot for a request entering engine submission.
    pub(crate) fn submitting(
        request_id: ServeRequestId,
        served_name: String,
        description: String,
        compile_us: u64,
    ) -> Self {
        Self {
            request_id,
            served_name,
            description,
            state: RequestLifecycleState::Submitting,
            prompt_tokens: 0,
            visible_output_tokens: 0,
            internal_tokens: 0,
            image_count: 0,
            image_steps: 0,
            cache: CacheAccounting::default(),
            resources: ResourceAccounting::default(),
            timings: RuntimeTimings {
                compile_us,
                ..RuntimeTimings::default()
            },
        }
    }
}

/// Cumulative per-runtime lifecycle counters. They use relaxed ordering: each is read
/// independently for reporting and none orders other memory.
#[derive(Debug, Default)]
pub(super) struct RuntimeLifecycleMetrics {
    pub(super) accepted: AtomicU64,
    pub(super) scheduled: AtomicU64,
    finished: AtomicU64,
    pub(super) rejected: AtomicU64,
    cancelled: AtomicU64,
    aborted: AtomicU64,
    failed: AtomicU64,
}

impl RuntimeLifecycleMetrics {
    /// Captures the current state as an immutable snapshot.
    pub(super) fn snapshot(&self) -> RuntimeMetricsSnapshot {
        RuntimeMetricsSnapshot {
            active: 0,
            accepted: self.accepted.load(Ordering::Relaxed),
            scheduled: self.scheduled.load(Ordering::Relaxed),
            finished: self.finished.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
            aborted: self.aborted.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Aggregate request lifecycle counts: the active requests plus cumulative milestone and
/// outcome counters.
pub struct RuntimeMetricsSnapshot {
    /// Number of registered requests that have not reached a terminal outcome, including
    /// requests still in preprocessing.
    pub active: u64,
    /// Cumulative number of engine-accepted requests.
    pub accepted: u64,
    /// Cumulative number of scheduled requests.
    pub scheduled: u64,
    /// Cumulative number of normally completed requests.
    pub finished: u64,
    /// Cumulative number of requests rejected before execution, including duplicate
    /// identifiers and preprocessing failures.
    pub rejected: u64,
    /// Cumulative number of cancelled requests.
    pub cancelled: u64,
    /// Cumulative number of forcibly aborted requests.
    pub aborted: u64,
    /// Cumulative number of failed requests.
    pub failed: u64,
}

impl RuntimeMetricsSnapshot {
    /// Returns stable lifecycle state names and values exported by the public metrics
    /// route.
    pub const fn state_counts(self) -> [(&'static str, u64); 8] {
        [
            ("active", self.active),
            ("accepted", self.accepted),
            ("scheduled", self.scheduled),
            ("finished", self.finished),
            ("rejected", self.rejected),
            ("cancelled", self.cancelled),
            ("aborted", self.aborted),
            ("failed", self.failed),
        ]
    }
}

/// Terminal-outcome latch for one registered request.
///
/// The guard moves from `generate_with` into the response stream. Its first `terminal` call
/// completes the registry record and increments one outcome counter; later calls do nothing.
/// Dropping it without a terminal records a cancellation or abort.
pub(super) struct LifecycleGuard {
    pub(super) request_id: ServeRequestId,
    metrics: Arc<RuntimeLifecycleMetrics>,
    requests: Arc<RequestRegistry>,
    pub(super) started: Instant,
    terminal: bool,
}

/// Response stream that feeds each event to the request registry and latches the terminal
/// outcome, replacing the engine's terminal with a control terminal when one takes precedence.
pub(super) struct LifecycleTrackedStream {
    inner: RequestOutputStream,
    lifecycle: LifecycleGuard,
}

impl LifecycleTrackedStream {
    /// Transfers the preprocessing guard to the caller's response stream.
    pub(super) fn new(inner: RequestOutputStream, lifecycle: LifecycleGuard) -> Self {
        Self { inner, lifecycle }
    }
}

impl Stream for LifecycleTrackedStream {
    type Item = Result<RequestOutput>;

    /// Polls the inner stream while enforcing control precedence and terminal accounting.
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(mut event))) => {
                let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;

                // A pending control replaces this event and ends the stream; replacing
                // `inner` drops the previous stream and any engine receiver it holds.
                if let Some(control) =
                    self.lifecycle
                        .requests
                        .observe(&self.lifecycle.request_id, &event, elapsed_us)
                {
                    let terminal = self.lifecycle.terminal(control, elapsed_us);
                    event = control_terminal_event(&self.lifecycle.request_id, terminal);
                    self.inner = Box::pin(futures::stream::empty());
                    return Poll::Ready(Some(Ok(event)));
                }

                // `terminal` returns an outcome other than the engine's only when the registry
                // holds a cancel or abort, which is why `control_terminal_event` accepts only
                // those two outcomes.
                let terminal = match &event {
                    RequestOutput::Finished { .. } => Some(LifecycleTerminal::Finished),
                    RequestOutput::Rejected { .. } => Some(LifecycleTerminal::Rejected),
                    RequestOutput::Cancelled { .. } => Some(LifecycleTerminal::Cancelled),
                    RequestOutput::Aborted { .. } => Some(LifecycleTerminal::Aborted),
                    RequestOutput::Failed { .. } => Some(LifecycleTerminal::Failed),
                    _ => None,
                };
                if let Some(terminal) = terminal {
                    let actual = self.lifecycle.terminal(terminal, elapsed_us);
                    if actual != terminal {
                        event = control_terminal_event(&self.lifecycle.request_id, actual);
                    }
                    self.inner = Box::pin(futures::stream::empty());
                }
                Poll::Ready(Some(Ok(event)))
            }
            // An error is a failure unless a control takes precedence, in which case the caller
            // sees the control terminal instead of the error.
            Poll::Ready(Some(Err(error))) => {
                let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;
                let actual = self
                    .lifecycle
                    .terminal(LifecycleTerminal::Failed, elapsed_us);
                self.inner = Box::pin(futures::stream::empty());
                if actual == LifecycleTerminal::Failed {
                    Poll::Ready(Some(Err(error)))
                } else {
                    Poll::Ready(Some(Ok(control_terminal_event(
                        &self.lifecycle.request_id,
                        actual,
                    ))))
                }
            }
            // An inner stream that ends without a terminal event is recorded as failed; only a
            // control outcome is surfaced to the caller as a synthesized event.
            Poll::Ready(None) => {
                if !self.lifecycle.terminal {
                    let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;
                    let actual = self
                        .lifecycle
                        .terminal(LifecycleTerminal::Failed, elapsed_us);
                    if actual != LifecycleTerminal::Failed {
                        return Poll::Ready(Some(Ok(control_terminal_event(
                            &self.lifecycle.request_id,
                            actual,
                        ))));
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl LifecycleGuard {
    /// Creates a guard for a request already registered in `requests`.
    pub(super) fn new(
        request_id: ServeRequestId,
        metrics: Arc<RuntimeLifecycleMetrics>,
        requests: Arc<RequestRegistry>,
        started: Instant,
    ) -> Self {
        Self {
            request_id,
            metrics,
            requests,
            started,
            terminal: false,
        }
    }

    /// Records the first terminal transition and increments its aggregate metric exactly once.
    ///
    /// Returns the recorded outcome: `RequestRegistry::complete` converts `kind` into
    /// `Cancelled` or `Aborted` when a control is pending. After the first call, returns `kind`
    /// unchanged without recording anything.
    pub(super) fn terminal(
        &mut self,
        kind: LifecycleTerminal,
        elapsed_us: u64,
    ) -> LifecycleTerminal {
        if self.terminal {
            return kind;
        }
        let actual = self
            .requests
            .complete(&self.request_id, kind.into(), elapsed_us)
            .and_then(lifecycle_terminal_for_state)
            .unwrap_or(kind);
        let counter = match actual {
            LifecycleTerminal::Finished => &self.metrics.finished,
            LifecycleTerminal::Rejected => &self.metrics.rejected,
            LifecycleTerminal::Cancelled => &self.metrics.cancelled,
            LifecycleTerminal::Aborted => &self.metrics.aborted,
            LifecycleTerminal::Failed => &self.metrics.failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        self.terminal = true;
        actual
    }
}

/// Terminal outcome of one request, as recorded by `LifecycleGuard`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LifecycleTerminal {
    Finished,
    Rejected,
    Cancelled,
    Aborted,
    Failed,
}

/// Returns the terminal lifecycle event represented by a state.
///
/// Pending `Cancelling` and `Aborting` states map to their eventual outcomes.
fn lifecycle_terminal_for_state(state: RequestLifecycleState) -> Option<LifecycleTerminal> {
    match state {
        RequestLifecycleState::Finished => Some(LifecycleTerminal::Finished),
        RequestLifecycleState::Rejected => Some(LifecycleTerminal::Rejected),
        RequestLifecycleState::Cancelled | RequestLifecycleState::Cancelling => {
            Some(LifecycleTerminal::Cancelled)
        }
        RequestLifecycleState::Aborted | RequestLifecycleState::Aborting => {
            Some(LifecycleTerminal::Aborted)
        }
        RequestLifecycleState::Failed => Some(LifecycleTerminal::Failed),
        RequestLifecycleState::Compiling
        | RequestLifecycleState::Submitting
        | RequestLifecycleState::Accepted
        | RequestLifecycleState::Scheduled
        | RequestLifecycleState::Streaming => None,
    }
}

/// Builds the terminal event for a control outcome.
fn control_terminal_event(
    request_id: &ServeRequestId,
    terminal: LifecycleTerminal,
) -> RequestOutput {
    match terminal {
        LifecycleTerminal::Aborted => RequestOutput::Aborted {
            request_id: request_id.clone(),
        },
        LifecycleTerminal::Cancelled => RequestOutput::Cancelled {
            request_id: request_id.clone(),
        },
        LifecycleTerminal::Finished | LifecycleTerminal::Rejected | LifecycleTerminal::Failed => {
            unreachable!("only cancellation and abort can replace an engine terminal")
        }
    }
}

impl From<LifecycleTerminal> for RequestLifecycleState {
    /// Converts the source value into this type.
    fn from(value: LifecycleTerminal) -> Self {
        match value {
            LifecycleTerminal::Finished => Self::Finished,
            LifecycleTerminal::Rejected => Self::Rejected,
            LifecycleTerminal::Cancelled => Self::Cancelled,
            LifecycleTerminal::Aborted => Self::Aborted,
            LifecycleTerminal::Failed => Self::Failed,
        }
    }
}

impl Drop for LifecycleGuard {
    /// Records an unfinished request as aborted when an abort is pending and as cancelled
    /// otherwise. This covers a caller that drops the pending future or the response stream.
    fn drop(&mut self) {
        if !self.terminal {
            let elapsed_us = self.started.elapsed().as_micros() as u64;
            let terminal = self.requests.drop_terminal(&self.request_id);
            let _ = self.terminal(terminal, elapsed_us);
        }
    }
}

#[try_stream]
/// Forwards events until the stream ends or an external control reaches terminal precedence.
///
/// Waiting on the registry alongside the engine stream ends the response promptly even while
/// the engine produces no events. Returning drops the inner stream and with it the engine
/// receiver.
pub(super) async fn control_aware_event_stream(
    request_id: ServeRequestId,
    requests: Arc<RequestRegistry>,
    mut stream: RequestOutputStream,
    mut y: TryYielder<RequestOutput, ServeError>,
) -> Result<()> {
    loop {
        if let Some(terminal) = requests.control_terminal(&request_id) {
            y.yield_ok(control_terminal_event(&request_id, terminal))
                .await;
            return Ok(());
        }
        tokio::select! {
            next = stream.next() => match next {
                Some(Ok(event)) => {
                    // A control recorded while this event was pending still wins over it.
                    if let Some(terminal) = requests.control_terminal(&request_id) {
                        y.yield_ok(control_terminal_event(&request_id, terminal)).await;
                        return Ok(());
                    }
                    let terminal = matches!(
                        event,
                        RequestOutput::Finished { .. }
                            | RequestOutput::Rejected { .. }
                            | RequestOutput::Cancelled { .. }
                            | RequestOutput::Aborted { .. }
                            | RequestOutput::Failed { .. }
                    );
                    y.yield_ok(event).await;
                    if terminal {
                        return Ok(());
                    }
                }
                Some(Err(error)) => return Err(error),
                None => return Ok(()),
            },
            terminal = requests.wait_for_control(&request_id) => {
                y.yield_ok(control_terminal_event(&request_id, terminal)).await;
                return Ok(());
            }
        }
    }
}
