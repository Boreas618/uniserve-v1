//! What a model family serves, as request processing serves it: chat in a family's output formats,
//! or video through the processor a video family binds. A family states it in its
//! request-processing description ([`crate::Family::serving`]).

use crate::image::ImageFamily;
use crate::video::VideoFamily;

/// What a family serves.
#[derive(Clone, Copy)]
pub enum Serving {
    /// Chat and completions over a causal language model.
    Chat(ChatProfile),
    /// Video generation over a diffusion pipeline.
    Video(&'static dyn VideoFamily),
}

/// The output formats a chat family's checkpoints write, and the image input it takes.
///
/// A format is a shared parser keyed by its name, as other serving systems key theirs; several
/// families may write one format.
#[derive(Clone, Copy)]
pub struct ChatProfile {
    /// How the model delimits its reasoning, when it reasons.
    pub reasoning: Option<ReasoningFormat>,
    /// How the model writes tool calls, when its chat template asks for them.
    pub tool_calls: Option<ToolCallFormat>,
    /// How the family's checkpoints take images, when they do; a family without it serves text
    /// alone and refuses an image part.
    pub images: Option<&'static dyn ImageFamily>,
}

impl std::fmt::Debug for ChatProfile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatProfile")
            .field("reasoning", &self.reasoning)
            .field("tool_calls", &self.tool_calls)
            .field("images", &self.images.is_some())
            .finish()
    }
}

impl PartialEq for ChatProfile {
    /// Equal formats, and image input of the same family or of none.
    fn eq(&self, other: &Self) -> bool {
        let images = match (self.images, other.images) {
            (Some(left), Some(right)) => std::ptr::addr_eq(left, right),
            (left, right) => left.is_none() && right.is_none(),
        };
        self.reasoning == other.reasoning && self.tool_calls == other.tool_calls && images
    }
}

impl Eq for ChatProfile {}

/// A reasoning output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// Reasoning between `<think>` and `</think>`, as Qwen3 writes it.
    Qwen3,
}

/// A tool call output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// One JSON object per call between `<tool_call>` and `</tool_call>`, as Qwen3 writes it.
    Qwen3,
}
