//! What a model family serves, as request processing serves it: chat in a family's output formats,
//! or video through the processor a video family binds. A family states it beside what execution
//! loads (`nn::family::Family::serving`).

use crate::video::VideoFamily;

/// What a family serves.
#[derive(Clone, Copy)]
pub enum Serving {
    /// Chat and completions over a causal language model.
    Chat(ChatProfile),
    /// Video generation over a diffusion pipeline.
    Video(&'static dyn VideoFamily),
}

/// The output formats a chat family's checkpoints write.
///
/// A format is a shared parser keyed by its name, as other serving systems key theirs; several
/// families may write one format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatProfile {
    /// How the model delimits its reasoning, when it reasons.
    pub reasoning: Option<ReasoningFormat>,
    /// How the model writes tool calls, when its chat template asks for them.
    pub tool_calls: Option<ToolCallFormat>,
}

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
