//! The request side of chat: the chat template that renders a request's
//! messages, tools and template arguments into a prompt.
//!
//! The template comes from the model files or `chat_config.template`.

pub use template::{GenerationPromptMode, HfChatRenderer};

/// Chat-template loading and rendering.
pub mod template;

use thiserror::Error;

/// Why a chat request cannot be rendered into a prompt.
#[derive(Debug, Error)]
pub enum Error {
    /// The request contains no messages to render.
    #[error("chat request must contain at least one message")]
    EmptyMessages,
    /// Continuation mode is selected for history that does not end with an assistant message.
    #[error("cannot continue the final message when the last message is not from the assistant")]
    ContinueFinalAssistantWithoutFinalAssistant,
    /// Prompt rendering requires a template but no template is configured.
    #[error("chat template is required but none was configured")]
    MissingChatTemplate,
    /// The chat template cannot be loaded, compiled, or rendered, or assistant
    /// history cannot be converted into template values (for example, tool-call
    /// arguments that are not valid JSON).
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    /// Multimodal content reached a text-only conversion or a renderer
    /// configured without multimodal support.
    #[error("unsupported multimodal content: {0}")]
    UnsupportedMultimodalContent(&'static str),
    /// The checkpoint's tokenizer configuration cannot be loaded.
    #[error(transparent)]
    ModelAssets(#[from] crate::api::model::assets::Error),
}

/// Result type of chat rendering.
pub type Result<T> = std::result::Result<T, Error>;
