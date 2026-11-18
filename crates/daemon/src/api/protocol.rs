//! OpenAI-compatible wire schemas: requests, responses and the values they
//! share; `video` holds the video generation request.
//!
//! Every request schema rejects unknown fields (`deny_unknown_fields`), at
//! every level. The chat request's ranges and cross-field rules run through
//! `validator` in the route's `ValidatedJson` extractor; the rules that depend
//! on the served model run when the deployment's `Processor` lowers the
//! request.

pub mod video;

use std::collections::HashMap;
use std::fmt;
use std::slice;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_with::SerializeDisplay;
use validator::Validate;

pub use video::{VideoGenerationRequest, VideoReference, from_form, request_field_names};

/// Returns `true` for fields whose wire default is enabled.
fn default_true() -> bool {
    true
}

/// A wire value accepted as either one string or an array of strings.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StringOrArray {
    /// One string value.
    String(String),
    /// Ordered array of string values.
    Array(Vec<String>),
}

impl StringOrArray {
    /// Borrows the value uniformly as a string slice.
    pub fn as_slice(&self) -> &[String] {
        match self {
            StringOrArray::String(s) => slice::from_ref(s),
            StringOrArray::Array(arr) => arr,
        }
    }

    /// Converts the value into an owned string vector.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StringOrArray::String(s) => vec![s],
            StringOrArray::Array(arr) => arr,
        }
    }
}

/// Rejects empty stop strings.
fn validate_stop(stop: &StringOrArray) -> Result<(), validator::ValidationError> {
    if stop.as_slice().iter().any(|s| s.is_empty()) {
        return Err(validator::ValidationError::new(
            "stop strings cannot be empty",
        ));
    }
    Ok(())
}

/// Validates that `top_p` lies in `(0, 1]`.
fn validate_top_p_value(top_p: f32) -> Result<(), validator::ValidationError> {
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(validator::ValidationError::new(
            "top_p must be in (0, 1] - greater than 0.0 and at most 1.0",
        ));
    }
    Ok(())
}

/// One typed part of multimodal message content.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum ContentPart {
    /// Plain text content.
    #[serde(rename = "text")]
    Text {
        /// Text carried by the content part.
        text: String,
    },
    /// Image content referenced by URL or data URL.
    #[serde(rename = "image_url")]
    ImageUrl {
        /// Image source and requested detail level.
        image_url: ImageUrl,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Remote or data URL and requested image-detail level.
pub struct ImageUrl {
    /// Remote URL or data URL containing the image.
    pub url: String,
    /// Requested image detail. The model family's image processor sizes every image by its own
    /// rule, which is OpenAI's `auto`; `low` and `high` are refused.
    pub detail: Option<ImageDetail>,
}

/// The image detail a request may ask for: OpenAI's `auto`, the server's choice, which the model
/// family's image processor makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// The image processor sizes the image.
    Auto,
}

/// Options that alter streamed chat-completion responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    /// Whether the stream includes a terminal usage-only chunk.
    pub include_usage: Option<bool>,
}

/// Whether the assistant may call a tool, as OpenAI's `tool_choice` states it: one of the
/// strings `auto`, `none` and `required`, or `{"type": "function", "function": {"name": NAME}}`
/// for one function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatToolChoice {
    /// The model chooses between text and tool calls. Tools are passed to the
    /// chat template and tool calls parsed only when the request declares
    /// tools.
    Auto,
    /// The request's `tools` are withheld from the chat template and tool
    /// calls are not parsed. Message-local tools on developer messages are
    /// still passed to the template.
    None,
    /// The model must call a tool. Only constrained decoding can hold a model
    /// to that, so a deployment without it refuses the request.
    Required,
    /// The model must call the function of this name; refused as `Required`
    /// is.
    Function(String),
}

impl<'de> Deserialize<'de> for ChatToolChoice {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// The two shapes the field takes on the wire.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Mode(String),
            Named {
                #[serde(rename = "type")]
                kind: String,
                function: Named,
            },
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Named {
            name: String,
        }

        let invalid = |stated: &str| {
            serde::de::Error::custom(format!(
                "invalid `tool_choice` {stated:?}: expected \"auto\", \"none\", \"required\" or \
                 {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}"
            ))
        };
        match Wire::deserialize(deserializer)? {
            Wire::Mode(mode) => match mode.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::None),
                "required" => Ok(Self::Required),
                other => Err(invalid(other)),
            },
            Wire::Named { kind, function } if kind == "function" => {
                Ok(Self::Function(function.name))
            }
            Wire::Named { kind, .. } => Err(invalid(&kind)),
        }
    }
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

/// A tool definition accepted by chat-completion requests.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Tool {
    /// Function tool callable by the assistant.
    Function {
        /// Function declaration exposed to the model.
        function: Function,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Function-tool name, description, and JSON parameter schema.
pub struct Function {
    /// Function name exposed to the model.
    pub name: String,
    /// Optional human-readable function description.
    pub description: Option<String>,
    /// JSON Schema describing accepted function arguments.
    pub parameters: Value,
    /// Whether the caller asks for arguments that follow the schema exactly, as OpenAI's
    /// function definition states it. The chat template is given the value; generation is
    /// not constrained by it.
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Completed assistant tool call.
pub struct ToolCall {
    /// Model-assigned tool call identifier.
    pub id: String,
    /// Tool type, currently `function`.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function name and serialized arguments.
    pub function: FunctionCallResponse,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Completed function name and serialized arguments.
pub struct FunctionCallResponse {
    /// Function name selected by the model.
    pub name: String,
    /// Serialized JSON arguments, when supplied.
    #[serde(default)]
    pub arguments: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
/// Incremental fields for one indexed tool call.
pub struct ToolCallDelta {
    /// Position of the tool call within the assistant message.
    pub index: u32,
    /// Tool call identifier when first announced.
    pub id: Option<String>,
    /// Tool type when first announced.
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    /// Incremental function fields.
    pub function: Option<FunctionCallDelta>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
/// Incremental function name and argument fragments.
pub struct FunctionCallDelta {
    /// Function name when first announced.
    pub name: Option<String>,
    /// Incremental serialized argument fragment.
    pub arguments: Option<String>,
}

/// A role-tagged OpenAI-compatible chat message.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "role")]
#[serde(deny_unknown_fields)]
pub enum ChatMessage {
    /// System instruction message.
    #[serde(rename = "system")]
    System {
        /// Instruction content.
        content: MessageContent,
        /// Optional participant name.
        name: Option<String>,
    },
    /// User-authored message.
    #[serde(rename = "user")]
    User {
        /// User content.
        content: MessageContent,
        /// Optional participant name.
        name: Option<String>,
    },
    /// Assistant-authored message.
    #[serde(rename = "assistant")]
    Assistant {
        /// Optional visible assistant content.
        content: Option<MessageContent>,
        /// Optional participant name.
        name: Option<String>,
        /// Tool calls requested by the assistant.
        tool_calls: Option<Vec<ToolCall>>,
        /// The assistant's reasoning, under the name responses state it by, so a
        /// client sends an assistant message back as it received it.
        reasoning: Option<String>,
    },
    /// Tool result message.
    #[serde(rename = "tool")]
    Tool {
        /// Tool result content.
        content: MessageContent,
        /// Identifier of the assistant tool call being answered.
        tool_call_id: String,
    },
    /// Developer instruction message with optional tool declarations.
    #[serde(rename = "developer")]
    Developer {
        /// Developer instruction content.
        content: MessageContent,
        /// Tools introduced by the developer message.
        tools: Option<Vec<Tool>>,
        /// Optional participant name.
        name: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
/// Chat message content accepted as text or typed parts.
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Ordered multimodal content parts.
    Parts(Vec<ContentPart>),
}

/// OpenAI usage fields.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    /// Tokens consumed by the prompt.
    pub prompt_tokens: u32,
    /// Sum of prompt and completion tokens.
    pub total_tokens: u32,
    /// Tokens generated by the model, including tokens retained internally
    /// for reasoning or model control.
    pub completion_tokens: u32,
    /// The breakdown of `prompt_tokens`.
    pub prompt_tokens_details: PromptTokensDetails,
}

/// OpenAI's breakdown of a response's prompt tokens.
#[derive(Debug, Clone, Serialize)]
pub struct PromptTokensDetails {
    /// Prompt tokens the request's first admission found cached in any tier
    /// (device, host, peer, remote store) and did not compute again.
    pub cached_tokens: u32,
}

impl Usage {
    /// Creates the usage of one chat response from its token counts, of
    /// which `cached_tokens` are prompt tokens found cached.
    pub fn new(prompt_tokens: u32, completion_tokens: u32, cached_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            // `prompt_tokens + completion_tokens` can exceed `u32::MAX`, which
            // would panic in debug builds and wrap in release builds.
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            completion_tokens,
            prompt_tokens_details: PromptTokensDetails { cached_tokens },
        }
    }
}

/// Generated-token logprob response payload.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbs {
    /// Per-token logprob entries, or `None` when unavailable.
    pub content: Option<Vec<ChatLogProbsContent>>,
}

/// Logprob details for one generated token.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbsContent {
    /// Decoded token text, or the `token_id:<id>` placeholder when the
    /// request sets `return_tokens_as_token_ids`.
    pub token: String,
    /// Natural logarithm of the selected token probability.
    pub logprob: f32,
    /// UTF-8 bytes of `token`.
    pub bytes: Option<Vec<u8>>,
    /// Candidate tokens at this position, including the selected token.
    pub top_logprobs: Vec<TopLogProb>,
}

/// One candidate token and its log probability.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct TopLogProb {
    /// Decoded token text, or the `token_id:<id>` placeholder when the
    /// request sets `return_tokens_as_token_ids`.
    pub token: String,
    /// Natural logarithm of the token probability.
    pub logprob: f32,
    /// UTF-8 bytes of `token`.
    pub bytes: Option<Vec<u8>>,
}

/// Wire response containing one structured API error.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// Structured API error.
    pub error: ErrorDetail,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
/// Structured OpenAI-compatible error details.
pub struct ErrorDetail {
    /// Human-readable error description.
    pub message: String,
    /// Stable OpenAI-compatible error category.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Request parameter associated with the error, when known.
    pub param: Option<String>,
    /// Stable machine-readable error code, when available.
    pub code: Option<String>,
}

/// A single model entry in the `/v1/models` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelObject {
    /// Served model identifier.
    pub id: String,
    /// Object type, always `model`.
    pub object: String,
    /// Unix timestamp in seconds; the listing reports when it was produced.
    pub created: i64,
    /// Owner label; the listing reports `uniserve`.
    pub owned_by: String,
}

/// Response body for `GET /v1/models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ListModelsResponse {
    /// Object type, always `list`.
    pub object: String,
    /// Models served by this configuration.
    pub data: Vec<ModelObject>,
}

/// The configured OpenAI-compatible chat request.
///
/// The route's `ValidatedJson` extractor deserializes it (unknown fields are
/// rejected) and then runs the field validators below and
/// `validate_chat_cross_parameters`. The route checks the served model, and
/// the deployment's `Processor` lowers the request into the engine's request,
/// where the `prompt_logprobs` rules run. Besides the OpenAI fields, the
/// schema accepts sampling and output extensions such as `top_k`, `min_p`,
/// `prompt_logprobs`, and `return_token_ids`.
#[derive(Debug, Clone, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "validate_chat_cross_parameters"))]
pub struct ChatCompletionRequest {
    /// Ordered conversation messages used to build the model prompt.
    #[validate(custom(function = "validate_messages"))]
    pub messages: Vec<ChatMessage>,
    /// Served model name.
    pub model: String,
    /// Frequency-based token penalty in the inclusive range `[-2, 2]`.
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,
    /// Additive sampling biases keyed by decimal token identifier. Keys are
    /// parsed during lowering, not by serde.
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Whether generated-token log probabilities are returned.
    #[serde(default)]
    pub logprobs: bool,
    /// Number of alternate generated-token logprobs, or `-1` for all.
    #[validate(range(min = -1))]
    pub top_logprobs: Option<i32>,
    /// Maximum number of completion tokens.
    #[validate(range(min = 1))]
    pub max_completion_tokens: Option<u32>,
    /// OpenAI's deprecated name of `max_completion_tokens`; a request states one of the two.
    #[validate(range(min = 1))]
    pub max_tokens: Option<u32>,
    /// Presence-based token penalty in the inclusive range `[-2, 2]`.
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,
    /// Deterministic sampling seed. Lowering reinterprets the signed value's
    /// bit pattern as `u64`.
    pub seed: Option<i64>,
    /// Text sequences that terminate generation.
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,
    /// Whether the response is delivered as server-sent event chunks.
    #[serde(default)]
    pub stream: bool,
    /// Options that alter streamed response framing.
    pub stream_options: Option<StreamOptions>,
    /// Output modalities requested from the model.
    #[serde(default = "default_modalities")]
    pub modalities: Vec<ChatModality>,
    /// Sampling temperature in the inclusive range `[0, 2]`.
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass in the interval `(0, 1]`.
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,
    /// Maximum candidate tokens retained before sampling.
    pub top_k: Option<u32>,
    /// Minimum token probability relative to the most likely token.
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,
    /// Multiplicative penalty applied to previously generated tokens.
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,
    /// Function tools available to the assistant.
    pub tools: Option<Vec<Tool>>,
    /// Policy controlling whether the assistant may call a tool.
    pub tool_choice: Option<ChatToolChoice>,
    /// Requested reasoning budget.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Extra chat-template variables, such as `enable_thinking`, which
    /// replace the server's default template kwargs of the same name.
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    /// Token identifiers that terminate generation.
    pub stop_token_ids: Option<Vec<u32>>,
    /// Whether matched stop strings remain in returned text.
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    /// Whether end-of-sequence tokens are excluded from termination checks.
    #[serde(default)]
    pub ignore_eos: bool,
    /// Minimum tokens generated before stop conditions may terminate output.
    #[validate(range(min = 1))]
    pub min_tokens: Option<u32>,
    /// Whether special tokens are omitted during text decoding.
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
    /// Number of alternate prompt-token logprobs, or `-1` for all. The bound
    /// and its interaction with `stream` are checked when the request is
    /// lowered.
    pub prompt_logprobs: Option<i32>,
    /// Optional whitelist of token identifiers eligible for sampling.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Text sequences excluded from generated output.
    pub bad_words: Option<Vec<String>>,
    /// Whether parsed reasoning content is included in responses.
    ///
    /// When false, response assembly also withholds output logprobs and token
    /// IDs that would expose hidden reasoning tokens; a non-streaming
    /// response that contains reasoning omits them for the whole completion.
    #[serde(default = "default_true")]
    pub include_reasoning: bool,
    /// Scheduler priority assigned to the request.
    pub priority: Option<i32>,
    /// Whether logprob token strings are rendered as `token_id:<id>`
    /// placeholders.
    pub return_tokens_as_token_ids: Option<bool>,
    /// Whether response objects include generated token identifiers.
    pub return_token_ids: Option<bool>,
    /// Caller-provided salt folded into the prompt-cache isolation key.
    pub cache_salt: Option<String>,
    /// The caller's key of the requests that share a prompt prefix, OpenAI's field for a
    /// conversation or an application's requests: the request's session, which its `request`
    /// event states and the routing program reads as its session hint. It partitions no cache;
    /// `cache_salt` does.
    pub prompt_cache_key: Option<String>,
}

impl ChatCompletionRequest {
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

    /// The URL of every input image, in prompt order: the `image_url` parts
    /// of system, developer, user, and tool-response messages, in message
    /// order and then part order. Assistant content carries no input images.
    /// Lowering pairs the fetched images with their parts in this order.
    pub fn image_urls(&self) -> Vec<String> {
        self.input_parts()
            .filter_map(|part| match part {
                ContentPart::ImageUrl { image_url, .. } => Some(image_url.url.clone()),
                ContentPart::Text { .. } => None,
            })
            .collect()
    }

    /// Whether the conversation uses tools: tool definitions, prior
    /// assistant tool calls or tool responses, all of which need a model
    /// whose tool calls are parsed.
    pub fn uses_tools(&self) -> bool {
        self.tools.as_ref().is_some_and(|tools| !tools.is_empty())
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

    /// The most tokens the request generates, when it states a limit:
    /// `max_completion_tokens`, or `max_tokens`, OpenAI's deprecated name of it.
    /// [`validate_chat_request`] refuses a request that states both.
    pub fn max_output_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    /// The tools the model may call: the request's tools unless
    /// `tool_choice` is `none`, an absent choice letting the model decide as
    /// OpenAI's default does; `None` when there are none.
    ///
    /// The chat template is given exactly these tools, and tool calls are
    /// parsed from the output exactly when there are some.
    pub fn callable_tools(&self) -> Option<&[Tool]> {
        let allowed = !matches!(self.tool_choice, Some(ChatToolChoice::None));
        self.tools
            .as_deref()
            .filter(|tools| allowed && !tools.is_empty())
    }
}

/// Why a chat completion's choice ended, as its `finish_reason` states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatFinishReason {
    /// The model stopped: at its end of sequence, a stop string or token, the end of its own
    /// output, or a repetition stop, after calling no tool.
    Stop,
    /// The request's generation-length limit ended it.
    Length,
    /// The model stopped after calling tools.
    ToolCalls,
    /// The request was cancelled or aborted.
    Abort,
    /// The sequence reached the length the engine's whole KV cache holds.
    CacheCapacity,
    /// The engine failed the request.
    Error,
}

impl ChatFinishReason {
    /// The finish reason of a choice the engine ended for `reason`, after calling tools when
    /// `called_tools` is set.
    pub fn of(reason: &proto::request::FinishReason, called_tools: bool) -> Self {
        use proto::request::FinishReason;

        match reason {
            FinishReason::Completed
            | FinishReason::Eos
            | FinishReason::Stop
            | FinishReason::Repetition => {
                if called_tools {
                    Self::ToolCalls
                } else {
                    Self::Stop
                }
            }
            FinishReason::MaxTokens => Self::Length,
            FinishReason::Cancelled | FinishReason::Aborted => Self::Abort,
            FinishReason::CacheCapacity => Self::CacheCapacity,
            FinishReason::Error => Self::Error,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// Output modality requested from a chat completion.
pub enum ChatModality {
    /// Text output.
    Text,
    /// Image output.
    Image,
}

/// Returns the default response modalities.
fn default_modalities() -> Vec<ChatModality> {
    vec![ChatModality::Text]
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
/// Non-streaming chat-completion response body.
pub struct ChatCompletionResponse {
    /// OpenAI-compatible completion identifier.
    pub id: String,
    /// Object type, always `chat.completion`.
    pub object: String,
    /// Response creation timestamp in Unix seconds.
    pub created: u64,
    /// Served model name.
    pub model: String,
    /// Completed assistant choices.
    pub choices: Vec<ChatCompletionChoice>,
    /// Request usage accounting, when available.
    pub usage: Option<Usage>,
    /// Prompt logprobs aligned to prompt positions, when requested. The
    /// first entry is `null` because the first prompt token is not scored.
    pub prompt_logprobs: Option<Vec<Option<HashMap<String, f32>>>>,
    /// Prompt token identifiers, when requested.
    pub prompt_token_ids: Option<Vec<u32>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
/// One completed assistant choice.
pub struct ChatCompletionChoice {
    /// Choice position in the response.
    pub index: u32,
    /// Completed assistant message.
    pub message: ChatCompletionMessage,
    /// Generated-token logprob details, when requested.
    pub logprobs: Option<ChatLogProbs>,
    /// Stable terminal reason, when generation is complete.
    pub finish_reason: Option<ChatFinishReason>,
    /// Token identifier or string that triggered stopping, when applicable.
    pub stop_reason: Option<Value>,
    /// Generated token identifiers, when requested.
    pub token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, SerializeDisplay)]
/// Serializer for the constant assistant role label.
pub struct AssistantRole;

impl fmt::Display for AssistantRole {
    /// Writes `assistant`; `SerializeDisplay` uses this as the wire value.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("assistant")
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
/// Assistant message returned by a non-streaming completion.
pub struct ChatCompletionMessage {
    /// Constant assistant role marker.
    pub role: AssistantRole,
    /// Visible assistant text, when produced.
    pub content: Option<String>,
    /// Completed tool calls requested by the assistant.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Parsed reasoning, when requested; an assistant message sent back states it under the
    /// same name.
    pub reasoning: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
/// One server-sent chat-completion chunk.
///
/// Chunks have no `prompt_logprobs` field; lowering rejects a streamed
/// request that asks for a positive or `-1` `prompt_logprobs` count.
pub struct ChatCompletionStreamResponse {
    /// OpenAI-compatible completion identifier.
    pub id: String,
    /// Object type, always `chat.completion.chunk`.
    pub object: String,
    /// Response creation timestamp in Unix seconds.
    pub created: u64,
    /// Served model name.
    pub model: String,
    /// Incremental assistant choices carried by this chunk.
    pub choices: Vec<ChatCompletionStreamChoice>,
    /// Terminal request usage when requested by stream options.
    pub usage: Option<Usage>,
    /// Prompt token identifiers, when supplied with initial metadata.
    pub prompt_token_ids: Option<Vec<u32>>,
}

impl ChatCompletionStreamResponse {
    /// Constructs the initial stream response for one completion.
    pub fn new(id: &str, model: &str, created: u64) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: Vec::new(),
            usage: None,
            prompt_token_ids: None,
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize)]
/// One incremental choice carried by a stream chunk.
pub struct ChatCompletionStreamChoice {
    /// Choice position in the response.
    pub index: u32,
    /// Incremental assistant message fields.
    pub delta: ChatMessageDelta,
    /// Generated-token logprob details for this chunk.
    pub logprobs: Option<ChatLogProbs>,
    /// Stable terminal reason on the final choice chunk.
    pub finish_reason: Option<ChatFinishReason>,
    /// Token identifier or string that triggered stopping, when applicable.
    pub stop_reason: Option<Value>,
    /// Generated token identifiers carried by this chunk.
    pub token_ids: Option<Vec<u32>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize)]
/// Incremental assistant-message fields in a stream chunk.
pub struct ChatMessageDelta {
    /// Assistant role marker, normally present on the first chunk.
    pub role: Option<AssistantRole>,
    /// Incremental visible text.
    pub content: Option<String>,
    /// Incremental tool-call fields.
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    /// Incremental parsed reasoning text.
    pub reasoning: Option<String>,
}

/// Validates conversation-level presence and non-empty user content.
fn validate_messages(messages: &[ChatMessage]) -> Result<(), validator::ValidationError> {
    if messages.is_empty() {
        return Err(validator::ValidationError::new("messages cannot be empty"));
    }
    for message in messages {
        if let ChatMessage::User { content, .. } = message {
            match content {
                MessageContent::Text(text) if text.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content cannot be empty",
                    ));
                }
                MessageContent::Parts(parts) if parts.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content parts cannot be empty",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Validates dependencies and bounds that span multiple chat request fields.
///
/// Each failure is reported by its error code alone. `ValidatedJson` forwards
/// the `validator` display text, which includes that code, as the client's
/// invalid-request message, so the codes are part of the public error text.
fn validate_chat_cross_parameters(
    request: &ChatCompletionRequest,
) -> Result<(), validator::ValidationError> {
    if request.top_logprobs.is_some() && !request.logprobs {
        return Err(validator::ValidationError::new(
            "top_logprobs_requires_logprobs",
        ));
    }

    if request.stream_options.is_some() && !request.stream {
        return Err(validator::ValidationError::new(
            "stream_options_requires_stream",
        ));
    }

    if request.max_completion_tokens.is_some() && request.max_tokens.is_some() {
        return Err(validator::ValidationError::new(
            "max_tokens_and_max_completion_tokens_both_stated",
        ));
    }

    if let (Some(minimum), Some(maximum)) = (request.min_tokens, request.max_output_tokens())
        && minimum > maximum
    {
        return Err(validator::ValidationError::new(
            "min_tokens_exceeds_max_completion_tokens",
        ));
    }

    if request.modalities.is_empty() {
        return Err(validator::ValidationError::new("modalities_empty"));
    }
    let unique = request
        .modalities
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if unique.len() != request.modalities.len() {
        return Err(validator::ValidationError::new("modalities_duplicate"));
    }

    // A choice that lets or makes the model call a tool needs tools to call; an
    // absent choice defaults to `auto`, which serves no tools when the list is
    // empty.
    let calls_tools = matches!(
        request.tool_choice,
        Some(ChatToolChoice::Auto | ChatToolChoice::Required | ChatToolChoice::Function(_))
    );
    if calls_tools
        && !request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
    {
        return Err(validator::ValidationError::new(
            "tool_choice_requires_tools",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tool_choice` takes OpenAI's three strings and its named-function object, and
    /// nothing else.
    #[test]
    fn tool_choice_takes_the_openai_shapes() {
        let choice = |value: serde_json::Value| serde_json::from_value::<ChatToolChoice>(value);

        assert_eq!(
            choice(serde_json::json!("auto")).unwrap(),
            ChatToolChoice::Auto
        );
        assert_eq!(
            choice(serde_json::json!("none")).unwrap(),
            ChatToolChoice::None
        );
        assert_eq!(
            choice(serde_json::json!("required")).unwrap(),
            ChatToolChoice::Required
        );
        assert_eq!(
            choice(serde_json::json!({"type": "function", "function": {"name": "f"}})).unwrap(),
            ChatToolChoice::Function("f".to_owned())
        );
        for refused in [
            serde_json::json!("always"),
            serde_json::json!({"type": "tool", "function": {"name": "f"}}),
            serde_json::json!({"type": "function", "function": {"name": "f", "x": 1}}),
        ] {
            assert!(choice(refused.clone()).is_err(), "{refused}");
        }
    }

    /// `max_tokens`, OpenAI's deprecated name of `max_completion_tokens`, states the limit when
    /// the request states it alone, and a request stating both is refused.
    #[test]
    fn max_tokens_is_the_deprecated_name_of_max_completion_tokens() {
        let request = |fields: serde_json::Value| {
            let mut value = serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            });
            value
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            serde_json::from_value::<ChatCompletionRequest>(value).unwrap()
        };

        assert_eq!(request(serde_json::json!({})).max_output_tokens(), None);
        let deprecated = request(serde_json::json!({"max_tokens": 7}));
        assert_eq!(deprecated.max_output_tokens(), Some(7));
        assert!(deprecated.validate().is_ok());

        let both = request(serde_json::json!({"max_tokens": 7, "max_completion_tokens": 9}));
        let refused = both.validate().unwrap_err();
        assert!(
            refused
                .to_string()
                .contains("max_tokens_and_max_completion_tokens_both_stated"),
            "{refused}"
        );
    }

    /// An image part asks for the processor's own sizing (`auto`) or none, and states no other
    /// field.
    #[test]
    fn an_image_part_takes_only_the_processors_detail() {
        let part = |image: serde_json::Value| {
            serde_json::from_value::<ContentPart>(serde_json::json!({
                "type": "image_url",
                "image_url": image,
            }))
        };

        assert!(part(serde_json::json!({"url": "data:,", "detail": "auto"})).is_ok());
        assert!(part(serde_json::json!({"url": "data:,"})).is_ok());
        assert!(part(serde_json::json!({"url": "data:,", "detail": "high"})).is_err());
        assert!(
            serde_json::from_value::<ContentPart>(serde_json::json!({
                "type": "image_url",
                "image_url": {"url": "data:,"},
                "uuid": "a",
            }))
            .is_err()
        );
    }

    #[test]
    fn configured_request_schema_rejects_unknown_controls() {
        let value = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "unsupported_option": true
        });
        assert!(serde_json::from_value::<ChatCompletionRequest>(value).is_err());
    }
}
