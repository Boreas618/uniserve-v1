//! The events of a serving stream ([`RequestOutput`]) with their terminal
//! status and stop cause, and the timings they carry.

use serde::{Deserialize, Serialize};

use super::ServeRequestId;
use crate::serving::chat::{AssistantBlockKind, AssistantContentBlock};
use crate::serving::text::{DecodedLogprobs, DecodedPromptLogprobs, FinishReason, StopReason};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Queue, execution, and end-to-end duration totals.
pub struct RuntimeTimings {
    /// Request validation and tokenization duration in microseconds, including time queued
    /// for the blocking pool.
    pub compile_us: u64,
    /// Queue duration in microseconds, when scheduling timestamps are available.
    pub queue_us: Option<u64>,
    /// Time from request start to the first visible output in microseconds.
    pub first_visible_output_us: Option<u64>,
    /// End-to-end request duration in microseconds.
    pub total_us: u64,
}

/// Runtime event consumed by the HTTP response layer.
///
/// A stream returned by [`ServingRuntime`](super::ServingRuntime) yields at most one
/// terminal variant (`Finished`, `Rejected`, `Cancelled`, `Aborted`, or `Failed`) or one
/// error, and then ends.
#[derive(Debug, Clone)]
pub enum RequestOutput {
    /// Immutable generated media retained through response delivery.
    Artifact(uniserve_core::ArtifactEvent),
    /// Latest diffusion progress: the engine-reported phase and the number of completed
    /// denoising steps.
    MediaProgress { phase: String, completed_steps: u32 },
    /// The engine accepted a compiled request.
    Accepted {
        /// Caller-visible request identifier.
        request_id: ServeRequestId,
        /// Validation and tokenization duration in microseconds.
        compile_duration_us: u64,
        /// Number of prompt tokens submitted to the engine.
        prompt_token_count: usize,
        /// Prompt token identifiers submitted to the engine.
        prompt_token_ids: Vec<u32>,
        /// Per-position prompt log probabilities, when requested.
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    /// The engine scheduled the request for model execution.
    Scheduled {
        /// Caller-visible request identifier.
        request_id: ServeRequestId,
        /// Engine wall-clock time, in UNIX seconds, at which the request entered the
        /// scheduler's waiting queue.
        queued_at: Option<f64>,
        /// Engine wall-clock time, in UNIX seconds, at which the scheduler admitted the
        /// request for execution.
        scheduled_at: Option<f64>,
    },
    /// Newly decoded user-visible text.
    TextDelta {
        /// Newly visible decoded text.
        text: String,
        /// Token identifiers represented by this update.
        token_ids: Vec<u32>,
        /// Per-position candidate log probabilities, when requested.
        logprobs: Option<DecodedLogprobs>,
    },
    /// Newly decoded structured reasoning text.
    ReasoningDelta {
        /// Newly decoded reasoning text.
        text: String,
    },
    /// A structured assistant output block has opened.
    OutputBlockStart {
        /// Zero-based content-block index.
        index: usize,
        /// Semantic kind of the opened block.
        kind: AssistantBlockKind,
    },
    /// A structured assistant output block has closed.
    OutputBlockEnd {
        /// Zero-based content-block index.
        index: usize,
        /// Complete normalized block content.
        block: AssistantContentBlock,
    },
    /// A structured function-tool call has opened.
    ToolCallStart {
        /// Zero-based tool-call index.
        index: usize,
        /// Request-local tool-call identifier.
        id: String,
        /// Selected function name.
        name: String,
    },
    /// Newly decoded serialized tool arguments.
    ToolCallArgumentsDelta {
        /// Zero-based tool-call index.
        index: usize,
        /// Newly decoded argument text.
        delta: String,
    },
    /// A structured function-tool call has closed.
    ToolCallEnd {
        /// Zero-based tool-call index.
        index: usize,
        /// Request-local tool-call identifier.
        id: String,
        /// Selected function name.
        name: String,
        /// Complete serialized function arguments.
        arguments: String,
    },
    /// Generation of one image has begun.
    ImageBegin {
        /// Request-local image identifier.
        image_id: String,
        /// Output width in pixels, when known.
        width: Option<u32>,
        /// Output height in pixels, when known.
        height: Option<u32>,
        /// Planned diffusion step count, when known.
        steps: Option<u32>,
        /// Elapsed request duration in microseconds.
        elapsed_us: u64,
    },
    /// One image-generation step has completed.
    ImageStep {
        /// Request-local image identifier.
        image_id: String,
        /// Completed step number.
        step: u32,
        /// Elapsed request duration in microseconds.
        elapsed_us: u64,
    },
    /// Generated image state has been committed to the model context.
    ImageCommit {
        /// Request-local image identifier.
        image_id: String,
        /// Elapsed request duration in microseconds.
        elapsed_us: u64,
    },
    /// One generated image is complete and available to the response layer.
    ImageDone {
        /// Request-local image identifier.
        image_id: String,
        /// Output width in pixels, when reported.
        width: Option<u32>,
        /// Output height in pixels, when reported.
        height: Option<u32>,
        /// Encoded image size in bytes, when reported.
        bytes: Option<u64>,
        /// Base64-encoded PNG payload, when retained for transport.
        pixels_png_b64: Option<String>,
        /// Elapsed request duration in microseconds.
        elapsed_us: u64,
    },
    /// Final request usage and timing totals.
    Usage {
        /// Number of prompt tokens submitted to the engine.
        prompt_tokens: u32,
        /// Number of generated tokens included in user-visible output.
        visible_output_tokens: u32,
        /// Number of generated tokens consumed by internal protocol sections.
        internal_tokens: u32,
        /// Number of generated images.
        image_count: u32,
        /// Total image-generation steps completed.
        image_steps: u32,
        /// Request phase durations.
        timings: RuntimeTimings,
    },
    /// The output candidate reached a terminal condition.
    Finished {
        /// Semantic terminal status.
        reason: FinishStatus,
        /// Optional model- or engine-provided terminal detail.
        finish_detail: Option<String>,
    },
    /// The request was rejected before execution.
    Rejected {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Whether the request is unservable or the engine is full.
        kind: uniserve_core::RejectionKind,
        /// Caller-visible rejection description.
        message: String,
    },
    /// The request was cancelled.
    Cancelled {
        /// Identifier of the cancelled request.
        request_id: ServeRequestId,
    },
    /// The request was forcibly aborted.
    Aborted {
        /// Identifier of the aborted request.
        request_id: ServeRequestId,
    },
    /// The request failed.
    Failed {
        /// Identifier of the failed request.
        request_id: ServeRequestId,
        /// Caller-visible failure description.
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Semantic terminal status of a completed serving stream.
pub enum FinishStatus {
    /// Generation reached a normal stop condition.
    Stop {
        /// Concrete stop cause, when one was observed.
        cause: Option<StopCause>,
    },
    /// The configured generation-length limit was reached.
    Length,
    /// The request was cancelled or aborted.
    Abort,
    /// Generation failed.
    Error,
    /// Repetition detection terminated generation.
    Repetition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
/// Concrete token or string that triggered a stop.
pub enum StopCause {
    /// The model emitted an end-of-sequence token.
    Eos,
    /// The sampler emitted a configured stop-token identifier.
    TokenId(u32),
    /// Decoded output matched a configured stop string.
    Text(String),
}

impl From<&FinishReason> for FinishStatus {
    /// Converts the source value into this type.
    fn from(reason: &FinishReason) -> Self {
        match reason.reason() {
            uniserve_core::FinishReason::Eos
            | uniserve_core::FinishReason::Completed
            | uniserve_core::FinishReason::Stop
            | uniserve_core::FinishReason::ImageDone => Self::Stop {
                cause: reason.as_stop_reason().map(|value| match value {
                    StopReason::Token(id) => StopCause::TokenId(*id),
                    StopReason::String(text) => StopCause::Text(text.clone()),
                }),
            },
            uniserve_core::FinishReason::MaxTokens => Self::Length,
            uniserve_core::FinishReason::Cancelled | uniserve_core::FinishReason::Aborted => {
                Self::Abort
            }
            uniserve_core::FinishReason::Error => Self::Error,
            uniserve_core::FinishReason::Repetition => Self::Repetition,
        }
    }
}
