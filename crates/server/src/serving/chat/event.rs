//! Structured assistant message content.
//!
//! These types describe assistant output in both directions. The block
//! assembler in `output::structured` publishes finished text and reasoning
//! blocks in `RequestOutput::OutputBlockEnd`, and the non-streaming Chat
//! Completions response collects finished blocks and tool calls into an
//! [`AssistantMessage`]. Chat requests carry prior assistant turns as
//! `ChatMessage::Assistant` for the template renderer. The incremental events
//! produced while parsing live in `output::processor`.

use std::ops::Deref;

use serde::{Deserialize, Serialize};

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
