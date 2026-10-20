//! Chat template renderer implementations.
//!
//! The only renderer is the Hugging Face Jinja renderer in `renderer::hf`.

/// Chat-template renderer implementations.
pub mod renderer;

pub use renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer};

pub use crate::serving::chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage,
    ChatRequest, ChatRole, ChatToolChoice, GenerationPromptMode, ReasoningEffort, Tool,
};
