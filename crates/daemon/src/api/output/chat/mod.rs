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
    /// Incremental tool-call events violate the assembler's order:
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
    Text(#[from] crate::api::output::text::Error),
}

/// Result type returned by chat calls.
pub type Result<T> = std::result::Result<T, Error>;

/// Semantic kind of one assistant output block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantBlockKind {
    /// Visible final-answer text.
    Text,
    /// Extracted reasoning content.
    Reasoning,
}
