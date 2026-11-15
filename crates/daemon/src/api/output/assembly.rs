//! The timings of a request's response, and the assembly of a diffusion
//! media request's engine events into public serving events.
//!
//! [`StreamClock`] keeps the timings every response reports in its terminal
//! usage; a token request's `OutputProcessor` and [`media_output`] share
//! it. Elapsed times are microseconds since the request started. The queue
//! wait is the difference of the engine's `Scheduled` UNIX timestamps,
//! clamped at zero.
//!
//! [`media_output`] ends its stream with exactly one terminal item:
//! `Finished` with the request's usage, `Rejected`, `Failed`, or an
//! `OutputError` when engine output violates the event protocol.

use std::time::{Duration, Instant};

use crate::api::events::Events;
use asynk_strim_attr::{TryYielder, try_stream};
use proto::request::Event;

use crate::api::output::{FinishReason, OutputError, RequestOutput, RequestUsage, RuntimeTimings};

/// Result of a stream that ends in an output-processing failure.
type Result<T> = std::result::Result<T, OutputError>;

/// The timings of one request's stream.
pub(super) struct StreamClock {
    started: Instant,
    /// When `started` was, in microseconds since the UNIX epoch: the origin a
    /// successor's clock continues from.
    started_unix_us: u64,
    compile_us: u64,
    queue_us: Option<u64>,
    first_visible_output_us: Option<u64>,
}

impl StreamClock {
    pub(super) fn new(started: Instant, compile_us: u64) -> Self {
        Self {
            started,
            started_unix_us: unix_us(),
            compile_us,
            queue_us: None,
            first_visible_output_us: None,
        }
    }

    /// Continues the clock `state` records: when its response entered the server, how long it
    /// took to reach the engine and the timings measured since.
    pub(super) fn resume(state: &proto::handoff::Assembly) -> Self {
        Self {
            started: started_instant(state.started_unix_us),
            started_unix_us: state.started_unix_us,
            compile_us: state.compile_us,
            queue_us: state.queue_us,
            first_visible_output_us: state.first_visible_output_us,
        }
    }

    /// The clock's timings in a state the successor continues from.
    pub(super) fn state(&self, assembly: &mut proto::handoff::Assembly) {
        assembly.queue_us = self.queue_us;
        assembly.first_visible_output_us = self.first_visible_output_us;
        assembly.started_unix_us = self.started_unix_us;
        assembly.compile_us = self.compile_us;
    }

    /// Records the queue wait from the engine's scheduling timestamps, in
    /// UNIX seconds.
    pub(super) fn scheduled(&mut self, queued_at: f64, scheduled_at: f64) {
        self.queue_us = Some(((scheduled_at - queued_at).max(0.0) * 1e6) as u64);
    }

    /// Records the first visible output, once.
    pub(super) fn visible(&mut self) {
        let elapsed = self.elapsed_us();
        self.first_visible_output_us.get_or_insert(elapsed);
    }

    fn elapsed_us(&self) -> u64 {
        self.started.elapsed().as_micros() as u64
    }

    pub(super) fn timings(&self) -> RuntimeTimings {
        RuntimeTimings {
            compile_us: self.compile_us,
            queue_us: self.queue_us,
            first_visible_output_us: self.first_visible_output_us,
            total_us: self.elapsed_us(),
        }
    }
}

/// The current UNIX time in microseconds.
fn unix_us() -> u64 {
    crate::api::since_epoch().as_micros() as u64
}

/// The instant a response started, from the wall-clock origin a handed state
/// carries: the successor's timings continue the predecessor's.
pub fn started_instant(started_unix_us: u64) -> Instant {
    let elapsed = unix_us().saturating_sub(started_unix_us);
    Instant::now() - Duration::from_micros(elapsed)
}

/// The error of engine output that violates the event protocol.
fn malformed(request_id: &str, message: &str) -> OutputError {
    OutputError {
        request_id: request_id.to_owned(),
        source: crate::api::error::Error::MalformedOutput {
            request_id: request_id.to_string(),
            message: message.to_owned(),
        },
    }
}

/// Converts one diffusion media request's engine events into serving events.
///
/// Only scheduling, media progress, artifact, and terminal engine events are
/// valid; any other event ends the stream with an `OutputError`.
/// The terminal usage reports `prompt_tokens` and no output tokens.
#[try_stream]
pub(super) async fn media_output(
    request_id: String,
    started: Instant,
    compile_us: u64,
    prompt_tokens: usize,
    mut events: Box<dyn Events>,
    mut y: TryYielder<RequestOutput, OutputError>,
) -> Result<()> {
    let mut clock = StreamClock::new(started, compile_us);
    while let Some(event) = events.recv().await {
        let output = match event {
            // A diffusion request reads no prefix of cached tokens.
            Event::Scheduled {
                queued_at,
                scheduled_at,
                ..
            } => {
                clock.scheduled(queued_at, scheduled_at);
                RequestOutput::Scheduled {
                    queued_at,
                    scheduled_at,
                }
            }
            Event::MediaProgress {
                phase,
                completed_steps,
            } => RequestOutput::MediaProgress {
                phase,
                completed_steps,
            },
            Event::Artifact(artifact) => {
                clock.visible();
                RequestOutput::Artifact(artifact)
            }
            Event::Finished {
                reason,
                stop_reason,
                ..
            } => {
                let terminal = if reason == FinishReason::Error {
                    RequestOutput::Failed {
                        message: "engine execution failed".to_owned(),
                    }
                } else {
                    RequestOutput::Finished {
                        reason,
                        stop_reason,
                        usage: RequestUsage {
                            prompt_tokens: u32::try_from(prompt_tokens).unwrap_or(u32::MAX),
                            timings: clock.timings(),
                            ..RequestUsage::default()
                        },
                    }
                };
                y.yield_ok(terminal).await;
                return Ok(());
            }
            Event::Rejected { kind, message } => {
                y.yield_ok(RequestOutput::Rejected { kind, message }).await;
                return Ok(());
            }
            Event::Error { message } | Event::ArtifactUnavailable { message } => {
                y.yield_ok(RequestOutput::Failed { message }).await;
                return Ok(());
            }
            _ => {
                return Err(malformed(
                    &request_id,
                    "a diffusion request received a non-media event",
                ));
            }
        };
        y.yield_ok(output).await;
    }
    Err(malformed(
        &request_id,
        "the engine stream closed before a terminal event",
    ))
}
