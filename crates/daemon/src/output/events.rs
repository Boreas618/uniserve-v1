//! The events of a serving stream ([`RequestOutput`]), why a request
//! finished ([`FinishReason`]), and what it consumed ([`RequestUsage`]).

use std::sync::Arc;

use proto::request::{ArtifactEvent, RejectionKind};

pub use proto::request::StopReason;

use crate::output::chat::{AssistantBlockKind, AssistantContentBlock, AssistantToolCall};
use crate::output::text::{DecodedLogprobs, DecodedPromptLogprobs};

/// The prompt the engine generates from.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptInfo {
    /// Prompt token IDs submitted to the engine.
    pub token_ids: Arc<[u32]>,
    /// Per-position prompt log-probabilities, when requested.
    pub logprobs: Option<DecodedPromptLogprobs>,
}

/// Why a request finished, in vLLM's vocabulary.
///
/// A request the engine fails ends with [`RequestOutput::Failed`] instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// A normal stop: an end-of-sequence token, a stop token or string, or
    /// the end of the model's own output. Carries the stop token or string
    /// that matched, when one did.
    Stop(Option<StopReason>),
    /// The generation-length limit.
    Length,
    /// The request was cancelled or aborted before it finished.
    Abort,
    /// Repetition detection ended generation.
    Repetition,
}

impl FinishReason {
    /// Maps the engine's finish reason and stop cause; `None` for an engine
    /// error, which fails the request.
    pub fn from_engine(
        reason: proto::request::FinishReason,
        stop_reason: Option<StopReason>,
    ) -> Option<Self> {
        use proto::request::FinishReason as Engine;
        Some(match reason {
            Engine::Completed | Engine::Eos | Engine::Stop | Engine::ImageDone => {
                Self::Stop(stop_reason)
            }
            Engine::MaxTokens => Self::Length,
            Engine::Cancelled | Engine::Aborted => Self::Abort,
            Engine::Repetition => Self::Repetition,
            Engine::Error => return None,
        })
    }

    /// The reason's name: `stop`, `length`, `abort`, or `repetition`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stop(_) => "stop",
            Self::Length => "length",
            Self::Abort => "abort",
            Self::Repetition => "repetition",
        }
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
    /// Generated images.
    pub image_count: u32,
    /// Image-generation steps completed, over every image.
    pub image_steps: u32,
    /// Phase durations.
    pub timings: RuntimeTimings,
}

/// One generated image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedImage {
    /// Request-local image identifier.
    pub image_id: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Encoded PNG size in bytes.
    pub bytes: u64,
    /// The base64-encoded PNG.
    pub png_base64: String,
}

/// Event of a serving stream, consumed by the API layer.
///
/// A stream returned by [`Leader`](crate::Leader) yields at most one
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
        block: AssistantContentBlock,
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
        call: AssistantToolCall,
    },
    /// One image-generation step completed.
    ImageStep {
        /// Request-local image identifier.
        image_id: u32,
    },
    /// One generated image is complete.
    ImageDone(GeneratedImage),
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
    /// The request finished.
    Finished {
        /// Why it finished.
        reason: FinishReason,
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
