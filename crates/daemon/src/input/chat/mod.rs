//! The request side of chat: messages, options and tools as a request states
//! them, their validation, and the chat template that renders them into a
//! prompt.
//!
//! The template comes from the model files or `chat_config.template`.

pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatToolChoice,
    GenerationPromptMode, ReasoningEffort,
};
pub use template::HfChatRenderer;

/// Chat messages, options, tools, and validation.
pub mod request;
/// Chat-template loading and rendering.
pub mod template;
