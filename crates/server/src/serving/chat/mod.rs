//! Chat request rendering, structured events, and output processing.
//!
//! This module owns the common chat values and streaming contracts. A chat
//! request flows through it twice. Before submission, `InputProcessor`
//! validates the [`ChatRequest`], builds its request-scoped
//! [`ChatOutputProcessor`] with its model family's parsers, and renders the
//! request into a prompt with [`HfChatRenderer`]. During generation,
//! `serving::assembly` feeds decoded text through that processor and frames
//! the resulting assistant events as public serving events.
//!
//! The template comes from the model files or `chat_config.template`, and
//! the server's `reasoning_parsing` setting decides whether reasoning
//! sections become reasoning blocks.

pub use error::{Error, Result};
pub use event::{AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall};
pub use output::ChatOutputProcessor;
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatToolChoice,
    GenerationPromptMode, ReasoningEffort, Tool,
};
pub use template::renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer};

mod error;
/// Structured assistant message content.
pub mod event;
pub mod output;
/// Chat messages, options, tools, and validation.
pub mod request;
/// Chat-template loading and rendering.
pub mod template;
