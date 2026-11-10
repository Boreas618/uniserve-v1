//! A chat request as input processing renders it: the wire messages, the
//! request's tools and render options.
//!
//! The messages are the API's own (`openai::ChatMessage`), validated when
//! the route lowers the request; `HfChatRenderer` renders a [`ChatRequest`]
//! into the prompt, and the request's `ChatOutputProcessor` reads its tool
//! configuration to decide whether generated text is parsed for tool calls.

pub use crate::output::tools::Tool;
use serde::{Deserialize, Serialize};

use crate::openai::{ChatMessage, ContentPart, MessageContent};
use crate::output::chat::{Error, Result};

/// Controls how prompt rendering should end after the existing chat history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationPromptMode {
    /// Append a generation prompt for a new assistant turn.
    ///
    /// Equivalent to `add_generation_prompt = true` and `continue_final_message
    /// = false`.
    #[default]
    StartNewAssistant,
    /// Leave the final assistant message open so generation continues it.
    ///
    /// Equivalent to `add_generation_prompt = false` and
    /// `continue_final_message = true`.
    ContinueFinalAssistant,
    /// Render the existing chat history without adding any trailing generation
    /// prompt.
    ///
    /// Equivalent to `add_generation_prompt = false` and
    /// `continue_final_message = false`.
    NoGenerationPrompt,
}

/// Semantic effort level for reasoning models.
///
/// The value reaches the model only through the chat template, which receives
/// it as `reasoning_effort`; its effect depends on that template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Disables model reasoning.
    None,
    /// Requests the smallest available reasoning budget.
    Minimal,
    /// Requests a low reasoning budget.
    Low,
    /// Requests a medium reasoning budget.
    Medium,
    /// Requests a high reasoning budget.
    High,
    /// Requests an extra-high reasoning budget.
    XHigh,
    /// Requests the maximum available reasoning budget.
    Max,
}

/// Chat-template-related request options.
///
/// The options are exposed to prompt templates before tokenization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatConfig {
    /// Controls whether rendering starts a new assistant turn, continues the
    /// final assistant message, or emits no trailing generation prompt at
    /// all.
    pub generation_prompt_mode: GenerationPromptMode,

    /// Effort level exposed to chat templates for reasoning models.
    pub reasoning_effort: Option<ReasoningEffort>,

    /// The request's own template variables, such as `enable_thinking`. They
    /// replace the server's default template kwargs of the same name, and
    /// never name a variable the renderer sets itself
    /// ([`ChatConfig::RESERVED_TEMPLATE_KWARGS`]).
    #[serde(default)]
    pub template_kwargs: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for ChatConfig {
    /// Returns the default value.
    fn default() -> Self {
        Self {
            generation_prompt_mode: GenerationPromptMode::StartNewAssistant,
            reasoning_effort: None,
            template_kwargs: std::collections::HashMap::new(),
        }
    }
}

impl ChatConfig {
    /// Template variables the renderer sets from the request itself, which
    /// request template kwargs cannot replace.
    pub const RESERVED_TEMPLATE_KWARGS: [&'static str; 5] = [
        "messages",
        "tools",
        "documents",
        "add_generation_prompt",
        "continue_final_message",
    ];

    /// Returns whether to add a generation prompt for a new assistant turn after the
    /// existing chat history.
    pub fn add_generation_prompt(&self) -> bool {
        matches!(
            self.generation_prompt_mode,
            GenerationPromptMode::StartNewAssistant
        )
    }

    /// Returns whether to leave the final assistant message open so generation
    /// continues it.
    pub fn continue_final_message(&self) -> bool {
        matches!(
            self.generation_prompt_mode,
            GenerationPromptMode::ContinueFinalAssistant
        )
    }
}

/// Tool-choice semantics supported by `chat`.
///
/// The `Default` value is `None`, whereas the Chat Completions preprocessing
/// maps an absent `tool_choice` to `Auto`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatToolChoice {
    /// Allows the model to choose between text and tool calls. Tools are
    /// passed to the chat template and tool calls parsed only when
    /// `ChatRequest::tools` is non-empty.
    Auto,
    /// Withholds the request's `tools` from the chat template and disables
    /// tool-call parsing. Message-local tools on developer messages are still
    /// passed to the template.
    #[default]
    None,
}

/// One chat request ready to be rendered into a prompt and lowered into a
/// generate request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Ordered chat history to render.
    pub messages: Vec<ChatMessage>,
    /// Chat-specific rendering options.
    pub chat_options: ChatConfig,
    /// Function tools made available to the model for this request.
    pub tools: Vec<Tool>,
    /// Tool-choice behavior for this request.
    pub tool_choice: ChatToolChoice,
}

impl ChatRequest {
    /// Validates basic request invariants before rendering.
    ///
    /// Returns `Error::EmptyMessages` for an empty history, and
    /// `Error::ContinueFinalAssistantWithoutFinalAssistant` when continuation
    /// mode is selected but the last message is not from the assistant.
    pub fn validate(&self) -> Result<()> {
        if self.messages.is_empty() {
            return Err(Error::EmptyMessages);
        }
        let ends_with_assistant =
            matches!(self.messages.last(), Some(ChatMessage::Assistant { .. }));
        if self.chat_options.generation_prompt_mode == GenerationPromptMode::ContinueFinalAssistant
            && !ends_with_assistant
        {
            return Err(Error::ContinueFinalAssistantWithoutFinalAssistant);
        }
        Ok(())
    }

    /// Returns true if this request contains any multimodal content in its
    /// messages.
    pub fn has_multimodal(&self) -> bool {
        self.input_parts()
            .any(|part| matches!(part, ContentPart::ImageUrl { .. }))
    }

    /// Returns the URL of every input image, in prompt order: the
    /// `image_url` parts of system, developer, user, and tool-response
    /// messages, in message order and then part order. Assistant content
    /// carries no input images. Preprocessing pairs the resolved images with
    /// their parts in this order.
    pub fn image_urls(&self) -> Vec<String> {
        self.input_parts()
            .filter_map(|part| match part {
                ContentPart::ImageUrl { image_url, .. } => Some(image_url.url.clone()),
                ContentPart::Text { .. } => None,
            })
            .collect()
    }

    /// The content parts of every message but the assistant's, in message
    /// order and then part order.
    fn input_parts(&self) -> impl Iterator<Item = &ContentPart> {
        self.messages
            .iter()
            .filter_map(|message| match message {
                ChatMessage::System { content, .. }
                | ChatMessage::Developer { content, .. }
                | ChatMessage::User { content, .. }
                | ChatMessage::Tool { content, .. } => Some(content),
                ChatMessage::Assistant { .. } => None,
            })
            .filter_map(|content| match content {
                MessageContent::Parts(parts) => Some(parts),
                MessageContent::Text(_) => None,
            })
            .flatten()
    }

    /// Whether the conversation uses tools: tool definitions, prior
    /// assistant tool calls or tool responses, all of which need a model that
    /// parses tool calls.
    pub fn uses_tools(&self) -> bool {
        !self.tools.is_empty()
            || self.messages.iter().any(|message| match message {
                ChatMessage::Developer { tools, .. } => {
                    tools.as_ref().is_some_and(|tools| !tools.is_empty())
                }
                ChatMessage::Assistant { tool_calls, .. } => {
                    tool_calls.as_ref().is_some_and(|calls| !calls.is_empty())
                }
                ChatMessage::Tool { .. } => true,
                ChatMessage::System { .. } | ChatMessage::User { .. } => false,
            })
    }

    /// Returns true if this request should enable tool parsing based on the tool
    /// choice and tool list.
    ///
    /// `HfChatRenderer` passes the request's `tools` to the template only when
    /// this holds, and the chat output processors apply the same condition to
    /// decide whether to parse tool calls.
    pub fn tool_parsing_enabled(&self) -> bool {
        matches!(self.tool_choice, ChatToolChoice::Auto) && !self.tools.is_empty()
    }
}
