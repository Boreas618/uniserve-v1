//! Assembly of engine events into public serving events.
//!
//! [`generation_output`] serves every generation request. It pulls decoded
//! text from `decoded_text_event_stream`, runs it through the request's
//! `ChatOutputProcessor`, frames the parsed output as content blocks and tool
//! calls (`StructuredEventState`), and forwards each update's tokens and the
//! image events of an interleaved request. [`media_output`] serves diffusion
//! media requests.
//!
//! Both end their stream with exactly one terminal item: `Finished` with the
//! request's usage, `Rejected`, `Failed`, or an `OutputError` when output processing
//! fails, for example because engine output
//! violates the event protocol. Elapsed times are microseconds since the
//! request started. The queue wait is the difference of the engine's
//! `Scheduled` UNIX timestamps, clamped at zero.

use std::time::Instant;

use crate::engine::client::EventStream;
use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};
use proto::request::Event;

use crate::output::chat::structured::StructuredEventState;
use crate::output::text::decoded_text_event_stream;
use crate::output::text::{DecodedTextEvent, ImageEvent};
use crate::output::{
    FinishReason, OutputError, OutputParams, RequestOutput, RequestUsage, RuntimeTimings,
};

/// Result of a stream that ends in an output-processing failure.
type Result<T> = std::result::Result<T, OutputError>;

/// The timings of one request's stream.
struct StreamClock {
    started: Instant,
    compile_us: u64,
    queue_us: Option<u64>,
    first_visible_output_us: Option<u64>,
}

impl StreamClock {
    fn new(started: Instant, compile_us: u64) -> Self {
        Self {
            started,
            compile_us,
            queue_us: None,
            first_visible_output_us: None,
        }
    }

    /// Records the queue wait from the engine's scheduling timestamps, in
    /// UNIX seconds.
    fn scheduled(&mut self, queued_at: f64, scheduled_at: f64) {
        self.queue_us = Some(((scheduled_at - queued_at).max(0.0) * 1e6) as u64);
    }

    /// Records the first visible output, once.
    fn visible(&mut self) {
        let elapsed = self.elapsed_us();
        self.first_visible_output_us.get_or_insert(elapsed);
    }

    fn elapsed_us(&self) -> u64 {
        self.started.elapsed().as_micros() as u64
    }

    fn timings(&self) -> RuntimeTimings {
        RuntimeTimings {
            compile_us: self.compile_us,
            queue_us: self.queue_us,
            first_visible_output_us: self.first_visible_output_us,
            total_us: self.elapsed_us(),
        }
    }
}

/// Whether `output` is visible output: text, reasoning, a tool call, or
/// generated media.
fn is_visible(output: &RequestOutput) -> bool {
    match output {
        RequestOutput::TextDelta { text } | RequestOutput::ReasoningDelta { text } => {
            !text.is_empty()
        }
        RequestOutput::ToolCallStart { .. } | RequestOutput::Artifact(_) => true,
        _ => false,
    }
}

/// The error of engine output that violates the event protocol.
fn malformed(request_id: &str, message: &str) -> OutputError {
    OutputError {
        request_id: request_id.to_owned(),
        source: crate::output::text::Error::MalformedOutput {
            request_id: request_id.to_string(),
            message: message.to_owned(),
        }
        .into(),
    }
}

/// Decodes and parses one generation request's output.
///
/// Yields `Accepted` and `Scheduled` once the decoder has the prompt
/// metadata, then each decoded update's content-block, tool-call and image
/// events followed by its `TokenDelta`, then the terminal event.
#[try_stream]
pub(super) async fn generation_output(
    request_id: String,
    started: Instant,
    compile_us: u64,
    output: OutputParams,
    events: EventStream,
    mut y: TryYielder<RequestOutput, OutputError>,
) -> Result<()> {
    let OutputParams {
        tokenizer,
        prompt_token_ids,
        decode,
        prompt_logprobs,
        logprobs,
        mut processor,
    } = output;
    let failed = |source: crate::output::chat::Error| OutputError {
        request_id: request_id.clone(),
        source,
    };

    let decoded = decoded_text_event_stream(
        request_id.to_string(),
        tokenizer,
        prompt_token_ids,
        prompt_logprobs,
        logprobs,
        events,
        decode,
    );
    pin_mut!(decoded);
    let mut clock = StreamClock::new(started, compile_us);
    let mut blocks = StructuredEventState::new();
    let mut image_count = 0_u32;
    let mut image_steps = 0_u32;

    while let Some(event) = decoded.next().await {
        let outputs = match event.map_err(|error| failed(error.into()))? {
            DecodedTextEvent::Start {
                prompt,
                queued_at,
                scheduled_at,
            } => {
                processor.initialize(&prompt.token_ids);
                clock.scheduled(queued_at, scheduled_at);
                vec![
                    RequestOutput::Accepted(prompt),
                    RequestOutput::Scheduled {
                        queued_at,
                        scheduled_at,
                    },
                ]
            }
            DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
            } => {
                let mut outputs = Vec::new();
                for event in processor.push(delta).map_err(failed)? {
                    outputs.extend(blocks.push(event).map_err(failed)?);
                }
                if !token_ids.is_empty() || logprobs.is_some() {
                    outputs.push(RequestOutput::TokenDelta {
                        token_ids,
                        logprobs,
                    });
                }
                outputs
            }
            DecodedTextEvent::Image(ImageEvent::Begin) => {
                clock.visible();
                Vec::new()
            }
            DecodedTextEvent::Image(ImageEvent::Step { image_id }) => {
                image_steps = image_steps.saturating_add(1);
                vec![RequestOutput::ImageStep { image_id }]
            }
            DecodedTextEvent::Image(ImageEvent::Done(image)) => {
                image_count = image_count.saturating_add(1);
                vec![RequestOutput::ImageDone(image)]
            }
            DecodedTextEvent::Done(finished) => {
                // Release the text the parsers hold back and close the open
                // block before the terminal event.
                let mut outputs = Vec::new();
                for event in processor.finish().map_err(failed)? {
                    outputs.extend(blocks.push(event).map_err(failed)?);
                }
                outputs.extend(blocks.finish());
                for output in outputs {
                    if is_visible(&output) {
                        clock.visible();
                    }
                    y.yield_ok(output).await;
                }
                let saturate = |count: usize| u32::try_from(count).unwrap_or(u32::MAX);
                y.yield_ok(RequestOutput::Finished {
                    reason: finished.reason,
                    usage: RequestUsage {
                        prompt_tokens: saturate(finished.prompt_tokens),
                        output_tokens: saturate(finished.output_tokens),
                        image_count: image_count.max(saturate(finished.images)),
                        image_steps,
                        timings: clock.timings(),
                    },
                })
                .await;
                return Ok(());
            }
            DecodedTextEvent::Rejected { kind, message } => {
                y.yield_ok(RequestOutput::Rejected { kind, message }).await;
                return Ok(());
            }
            DecodedTextEvent::Failed { message } => {
                y.yield_ok(RequestOutput::Failed { message }).await;
                return Ok(());
            }
        };
        for output in outputs {
            if is_visible(&output) {
                clock.visible();
            }
            y.yield_ok(output).await;
        }
    }

    // The decoder ends with a terminal event or an error.
    Err(malformed(
        &request_id,
        "the decoded stream ended before its terminal event",
    ))
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
    mut events: EventStream,
    mut y: TryYielder<RequestOutput, OutputError>,
) -> Result<()> {
    let mut clock = StreamClock::new(started, compile_us);
    while let Some(event) = events.next().await {
        let output = match event {
            Event::Scheduled {
                queued_at,
                scheduled_at,
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
                let terminal = match FinishReason::from_engine(reason, stop_reason) {
                    Some(reason) => RequestOutput::Finished {
                        reason,
                        usage: RequestUsage {
                            prompt_tokens: u32::try_from(prompt_tokens).unwrap_or(u32::MAX),
                            timings: clock.timings(),
                            ..RequestUsage::default()
                        },
                    },
                    None => RequestOutput::Failed {
                        message: "engine execution failed".to_owned(),
                    },
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
