//! The assistant side of chat: the values an assistant message consists of,
//! and the parsing of a chat request's decoded text.
//!
//! A request's [`ChatParser`] holds the reasoning and tool-call parsers of the
//! formats its model family's chat profile names. The request's
//! `OutputProcessor` feeds decoded text through it and frames the resulting
//! assistant events as response deltas (`structured`).
//!
//! The daemon's `chat_config.reasoning_parser` setting decides whether
//! reasoning sections become reasoning blocks.

pub use parser::{AssistantEvent, ChatParser};

mod parser;
pub mod structured;
mod tool;

use std::ops::Deref;

use serde::{Deserialize, Serialize};
use thiserror::Error;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Error)]
/// Why a chat request's output cannot be processed.
pub enum Error {
    /// A reasoning or tool-call output parser cannot be initialized.
    #[error("failed to initialize {kind} parser `{name}`")]
    ParserInitialization {
        /// Semantic parser category.
        kind: &'static str,
        /// Configured parser name.
        name: String,
        /// Parser-specific initialization error.
        #[source]
        error: BoxedError,
    },
    /// Incremental tool-call events violate the assembler state contract:
    /// arguments arrive with no open tool call, or for a tool index other than
    /// the open one.
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStreamInvariant {
        /// Description of the violated stream invariant.
        message: String,
    },
    /// The text layer fails, for example in detokenization or engine output
    /// validation.
    #[error(transparent)]
    Text(#[from] crate::output::text::Error),
}

/// Result type returned by chat calls.
pub type Result<T> = std::result::Result<T, Error>;

/// One finalized assistant tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantToolCall {
    /// Request-local identifier used to associate a tool response.
    pub id: String,
    /// Function name selected by the model.
    pub name: String,
    /// Function arguments as JSON text. `HfChatRenderer` parses this string
    /// when rendering assistant history and rejects invalid JSON.
    pub arguments: String,
}

/// Semantic kind of one assistant output block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantBlockKind {
    /// Visible final-answer text.
    Text,
    /// Extracted reasoning content.
    Reasoning,
}

/// One structured assistant output block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantContentBlock {
    /// Visible final-answer text.
    Text {
        /// Visible text carried by the block.
        text: String,
    },
    /// Extracted reasoning content.
    Reasoning {
        /// Reasoning text carried by the block.
        text: String,
    },
    /// One finalized tool call.
    ToolCall(AssistantToolCall),
}

impl AssistantContentBlock {
    /// Returns this block as one finalized tool call, if applicable.
    pub fn as_tool_call(&self) -> Option<&AssistantToolCall> {
        match self {
            Self::ToolCall(call) => Some(call),
            _ => None,
        }
    }
}

/// Final structured assistant message assembled from the event stream.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// Ordered semantic content blocks in the assistant response.
    pub content: Vec<AssistantContentBlock>,
}

impl Deref for AssistantMessage {
    type Target = [AssistantContentBlock];

    /// Exposes the ordered content blocks as a slice.
    fn deref(&self) -> &Self::Target {
        &self.content
    }
}

impl AssistantMessage {
    /// Concatenates all visible final-answer text blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                AssistantContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Concatenates all extracted reasoning blocks, or returns `None` when the
    /// concatenation is empty.
    pub fn reasoning(&self) -> Option<String> {
        Some(
            self.content
                .iter()
                .filter_map(|block| match block {
                    AssistantContentBlock::Reasoning { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect(),
        )
        .filter(|text: &String| !text.is_empty())
    }

    /// Returns finalized assistant tool calls in encounter order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &AssistantToolCall> {
        self.content
            .iter()
            .filter_map(AssistantContentBlock::as_tool_call)
    }

    /// Returns whether this assistant message contains any tool-call blocks.
    pub fn has_tool_calls(&self) -> bool {
        self.content
            .iter()
            .any(|block| matches!(block, AssistantContentBlock::ToolCall(_)))
    }

    /// Pushes one new block to the end of the message content.
    pub fn push_block(&mut self, block: AssistantContentBlock) {
        self.content.push(block);
    }
}
