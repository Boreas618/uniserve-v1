//! The chat messages of the processing interface: a request to render ([`Render`]), the prompt
//! it renders to ([`Prompt`]), and the parser of its response ([`Open`], [`Event`]).
//!
//! The daemon validates a request against the OpenAI protocol before it renders it, and states
//! here what the family's prompt is made of: the messages in their OpenAI shape, the tools the
//! model may call, the request's template arguments and reasoning effort, and the sizes of its
//! fetched images.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One chat request to render into its prompt (`render`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Render {
    /// The conversation, in order.
    pub messages: Vec<Message>,
    /// The tools the model may call; `None` when it may call none.
    pub tools: Option<Vec<Tool>>,
    /// The request's `chat_template_kwargs`, which replace the deployment's of the same name.
    pub template_kwargs: Option<Map<String, Value>>,
    /// The request's `reasoning_effort`, as the protocol names it (`low`, `high`, ...).
    pub reasoning_effort: Option<String>,
    /// The displayed size of each fetched image of the request, in the order its image parts
    /// appear in the conversation.
    pub images: Vec<ImageSize>,
}

/// One message of a conversation, in its OpenAI shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// A system instruction.
    System {
        /// Its content.
        content: Content,
    },
    /// A developer instruction, which may declare tools of its own.
    Developer {
        /// Its content.
        content: Content,
        /// The tools it declares.
        tools: Option<Vec<Tool>>,
    },
    /// A user's message.
    User {
        /// Its content.
        content: Content,
    },
    /// An earlier assistant turn, as the client sends it back.
    Assistant {
        /// Its visible content.
        content: Option<Content>,
        /// Its reasoning.
        reasoning: Option<String>,
        /// The calls it made.
        tool_calls: Vec<ToolCall>,
    },
    /// A tool's result.
    Tool {
        /// Its content.
        content: Content,
        /// The call it answers.
        tool_call_id: String,
    },
}

/// A message's content: one text, or typed parts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// Plain text.
    Text(String),
    /// Text and image parts, in order.
    Parts(Vec<ContentPart>),
}

/// One part of a message's content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text.
    Text {
        /// The text.
        text: String,
    },
    /// An image, whose pixels the daemon fetched: the next of [`Render::images`].
    Image,
}

/// A tool the model may call, in the OpenAI shape the client states it in.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    /// A function tool.
    Function {
        /// The function's declaration.
        function: Function,
    },
}

/// A function tool's declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Function {
    /// The function's name.
    pub name: String,
    /// What it does.
    pub description: Option<String>,
    /// The JSON Schema of its arguments.
    pub parameters: Value,
    /// Whether the client asks for arguments that follow the schema exactly.
    pub strict: Option<bool>,
}

/// A call an earlier assistant turn made.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The call's identifier.
    pub id: String,
    /// The function called.
    pub function: FunctionCall,
}

/// The function of a call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The function's name.
    pub name: String,
    /// The arguments, as the JSON text the client sent; `None` when it sent none.
    pub arguments: Option<String>,
}

/// An image's displayed size in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSize {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// The prompt a request renders to: text the daemon tokenizes, and the blocks of its images.
///
/// The daemon encodes each text segment with the checkpoint's tokenizer, which splits the added
/// tokens a segment's text names, and concatenates the ids; an image segment stands where the
/// next of the request's images is, in [`Render::images`] order.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Prompt {
    /// The segments, in order.
    pub segments: Vec<Segment>,
}

/// One segment of a prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Segment {
    /// Text, tokenized without adding special tokens.
    Text(String),
    /// The block of the next image.
    Image(ImageBlock),
}

/// The block an image takes in a request's context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageBlock {
    /// Cache rows the block writes: the vision tower's merged rows of the image.
    pub tokens: u32,
    /// Logical positions the block spans: the advance of the rotary position of the text after
    /// it.
    pub positions: u32,
}

/// Why a request's prompt cannot be rendered.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("{message}")]
pub struct Refusal {
    /// The request field at fault, when one field is.
    pub param: Option<String>,
    /// What is wrong.
    pub message: String,
}

impl Refusal {
    /// A refusal naming no field.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            param: None,
            message: message.into(),
        }
    }

    /// A refusal of the field `param`.
    pub fn of(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            param: Some(param.into()),
            message: message.into(),
        }
    }
}

/// The parser of one response (`parse_open`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Open {
    /// The tools the model may call, against which its tool calls are parsed; `None` when the
    /// request lets it call none.
    pub tools: Option<Vec<Tool>>,
    /// The prompt's ids from its last special token on (the whole prompt when it holds none):
    /// what decides the region generation starts in.
    pub prompt: Vec<u32>,
}

/// What a decoded update parses into, in output order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    /// Reasoning text.
    Reasoning(String),
    /// Visible text.
    Content(String),
    /// A tool call opens; the arguments that follow are its own.
    ToolCall {
        /// The function called.
        name: String,
    },
    /// Arguments text of the open call; a call's increments concatenate to one JSON object.
    Arguments(String),
}
