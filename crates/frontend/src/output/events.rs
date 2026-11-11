//! The deltas of a request's response ([`RequestOutput`]), the name a
//! finish reason is reported under ([`finish_reason_name`]), and what a
//! request consumed ([`RequestUsage`]).

use std::sync::Arc;

use proto::request::{ArtifactEvent, RejectionKind};

pub use proto::request::{FinishReason, StopReason};

use crate::output::chat::AssistantBlockKind;
use proto::handoff::ToolCall;
use proto::handoff::assistant_block::Block;
use proto::handoff::{DecodedLogprobs, DecodedPromptLogprobs};

/// The prompt the engine generates from.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptInfo {
    /// Prompt token IDs submitted to the engine.
    pub token_ids: Arc<[u32]>,
    /// Per-position prompt log-probabilities, when requested.
    pub logprobs: Option<DecodedPromptLogprobs>,
}

/// The name a finished request is reported under, in vLLM's vocabulary:
/// `stop` for an end-of-sequence token, a stop token or string, or the end of
/// the model's own output; `length` for the generation-length limit; `abort`
/// for a cancelled or aborted request; `repetition`; `cache_capacity` for a
/// sequence that reached the length the engine's whole KV cache holds; and
/// `error` for a request the engine failed.
pub fn finish_reason_name(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Completed | FinishReason::Eos | FinishReason::Stop => "stop",
        FinishReason::MaxTokens => "length",
        FinishReason::Cancelled | FinishReason::Aborted => "abort",
        FinishReason::Repetition => "repetition",
        FinishReason::CacheCapacity => "cache_capacity",
        FinishReason::Error => "error",
    }
}

/// Phase durations of one request, in microseconds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeTimings {
    /// Request validation and tokenization, including time queued for the
    /// blocking pool.
    pub compile_us: u64,
    /// Time in the engine's waiting queue, from its scheduling timestamps.
    pub queue_us: Option<u64>,
    /// Time from request start to the first visible output.
    pub first_visible_output_us: Option<u64>,
    /// End-to-end request duration.
    pub total_us: u64,
}

/// What one finished request consumed and produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequestUsage {
    /// Prompt tokens submitted to the engine.
    pub prompt_tokens: u32,
    /// Generated tokens, as the engine counts its completion tokens:
    /// visible text and the tokens of internal sections alike.
    pub output_tokens: u32,
    /// Phase durations.
    pub timings: RuntimeTimings,
}

/// Event of a serving stream, consumed by the API layer.
///
/// A request's output stream yields at most one
/// terminal event (`Finished`, `Rejected`, or `Failed`) or one error, and
/// then ends. A generation stream yields, after `Accepted` and `Scheduled`,
/// the output of each decoded update in order: its reasoning, text and
/// tool-call events, then its `TokenDelta`.
#[derive(Debug, Clone)]
pub enum RequestOutput {
    /// The engine admitted the request and began it: its prompt.
    Accepted(PromptInfo),
    /// The engine scheduled the request for model execution.
    Scheduled {
        /// Engine wall-clock time, in UNIX seconds, at which the request
        /// entered the scheduler's waiting queue.
        queued_at: f64,
        /// Engine wall-clock time, in UNIX seconds, at which the scheduler
        /// admitted the request for execution.
        scheduled_at: f64,
    },
    /// Newly decoded visible text of the open text block.
    TextDelta {
        /// Visible text.
        text: String,
    },
    /// Newly decoded reasoning text of the open reasoning block.
    ReasoningDelta {
        /// Reasoning text.
        text: String,
    },
    /// The tokens of one decoded update, after its text, reasoning and
    /// tool-call events.
    TokenDelta {
        /// Generated token IDs.
        token_ids: Vec<u32>,
        /// Per-position candidate log-probabilities, when requested.
        logprobs: Option<DecodedLogprobs>,
    },
    /// An assistant output block opened.
    OutputBlockStart {
        /// Zero-based content-block index.
        index: usize,
        /// Semantic kind of the block.
        kind: AssistantBlockKind,
    },
    /// An assistant output block closed.
    OutputBlockEnd {
        /// Zero-based content-block index.
        index: usize,
        /// The complete block.
        block: Block,
    },
    /// A function-tool call opened.
    ToolCallStart {
        /// Zero-based tool-call index.
        index: usize,
        /// Request-local tool-call identifier.
        id: String,
        /// Selected function name.
        name: String,
    },
    /// Newly decoded arguments of the open tool call.
    ToolCallArgumentsDelta {
        /// Zero-based tool-call index.
        index: usize,
        /// Argument text.
        delta: String,
    },
    /// A function-tool call closed.
    ToolCallEnd {
        /// Zero-based tool-call index.
        index: usize,
        /// The complete call.
        call: ToolCall,
    },
    /// Latest diffusion progress: the engine-reported phase and the number
    /// of completed denoising steps.
    MediaProgress {
        /// Phase name.
        phase: String,
        /// Completed denoising steps.
        completed_steps: u32,
    },
    /// Immutable generated media retained through response delivery.
    Artifact(ArtifactEvent),
    /// The request finished. The engine's `Error` is never a finish: such a
    /// request ends with `Failed`.
    Finished {
        /// Why it finished.
        reason: FinishReason,
        /// The stop token or string that matched, when one did.
        stop_reason: Option<StopReason>,
        /// What it consumed and produced.
        usage: RequestUsage,
    },
    /// The engine refused the request before executing it.
    Rejected {
        /// Whether the request is unservable or the engine is full.
        kind: RejectionKind,
        /// Caller-visible rejection description.
        message: String,
    },
    /// The engine failed the request.
    Failed {
        /// Caller-visible failure description.
        message: String,
    },
}
