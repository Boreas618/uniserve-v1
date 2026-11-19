//! The assistant side of chat: the parsing of a chat request's decoded text
//! into the blocks an assistant message consists of
//! (`proto::handoff::assistant_block::Block`).
//!
//! A request's [`ChatParser`] is its instance of the epoch's processing program,
//! which holds the reasoning and tool-call parsers of the formats its model
//! family writes. The request's `OutputProcessor` feeds decoded text through it
//! and frames the resulting assistant events as response deltas (`structured`).
//!
//! The daemon's `chat_config.reasoning_parser` setting, which the program's
//! binding reads, decides whether reasoning sections become reasoning blocks.

pub use parser::{AssistantEvent, ChatParser};

mod parser;
pub mod structured;

use serde::{Deserialize, Serialize};

/// Semantic kind of one assistant output block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantBlockKind {
    /// Visible final-answer text.
    Text,
    /// Extracted reasoning content.
    Reasoning,
}
