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
/// A format is named for what the model writes, never for a model: request processing holds one
/// parser per format, and several families may write one format. What a checkpoint's template
/// instructs is the family's to state here; request processing does not read a template to
/// find out.
#[derive(Clone, Copy)]
pub struct ChatProfile {
    /// How the model delimits its reasoning, when it reasons.
    pub reasoning: Option<ReasoningFormat>,
    /// How the model writes tool calls, when it calls tools.
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
    /// Reasoning between `<think>` and `</think>`, each one token of the vocabulary.
    Think,
}

impl ReasoningFormat {
    /// Every reasoning format.
    pub const ALL: [Self; 1] = [Self::Think];

    /// The format's name, which a response's handoff states it by.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Think => "think",
        }
    }

    /// The format named `name` ([`Self::name`]); `None` for no format's name.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|format| format.name() == name)
    }

    /// The text that opens a reasoning section and the text that closes it.
    pub const fn delimiters(self) -> (&'static str, &'static str) {
        match self {
            Self::Think => ("<think>", "</think>"),
        }
    }
}

/// A tool-call output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// One JSON object per call, naming the function and its arguments, between `<tool_call>`
    /// and `</tool_call>`:
    ///
    /// ```text
    /// <tool_call>
    /// {"name": NAME, "arguments": {ARG: VALUE}}
    /// </tool_call>
    /// ```
    Json,
    /// Function and parameter tags between `<tool_call>` and `</tool_call>`, each value as
    /// text that the tool's JSON schema types:
    ///
    /// ```text
    /// <tool_call>
    /// <function=NAME>
    /// <parameter=ARG>
    /// VALUE
    /// </parameter>
    /// </function>
    /// </tool_call>
    /// ```
    Xml,
}

impl ToolCallFormat {
    /// Every tool-call format.
    pub const ALL: [Self; 2] = [Self::Json, Self::Xml];

    /// The format's name, which a response's handoff states it by.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Xml => "xml",
        }
    }

    /// The format named `name` ([`Self::name`]); `None` for no format's name.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|format| format.name() == name)
    }
}
