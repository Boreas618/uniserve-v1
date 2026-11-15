//! The assistant side of chat: the parsing of a chat request's decoded text
//! into the blocks an assistant message consists of
//! (`proto::handoff::assistant_block::Block`).
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

use serde::{Deserialize, Serialize};

/// Semantic kind of one assistant output block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantBlockKind {
    /// Visible final-answer text.
    Text,
    /// Extracted reasoning content.
    Reasoning,
}
