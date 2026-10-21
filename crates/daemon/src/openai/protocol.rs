//! OpenAI-compatible wire schemas: requests, responses and the values they
//! share.
//!
//! Every request schema rejects unknown fields (`deny_unknown_fields`), at
//! every level. The chat request's ranges and cross-field rules run through
//! `validator` in the route's `ValidatedJson` extractor. The video request is
//! the video service's own form (`input::video::VideoGenerationRequest`).

use std::collections::HashMap;
use std::fmt;
use std::slice;

use llm_multimodal::ImageDetail;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_with::SerializeDisplay;
use validator::Validate;

use crate::input::chat::{ChatToolChoice, ReasoningEffort};

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
        /// Optional caller-provided image identity.
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Remote or data URL and requested image-detail level.
pub struct ImageUrl {
    /// Remote URL or data URL containing the image.
    pub url: String,
    /// Requested image preprocessing detail.
    pub detail: Option<ImageDetail>,
}

/// Options that alter streamed chat-completion responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    /// Whether the stream includes a terminal usage-only chunk.
    pub include_usage: Option<bool>,
}

/// A tool definition accepted by chat-completion requests.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Tool {
    /// Function tool callable by the assistant.
    Function {
        /// Function declaration exposed to the model.
        function: Function,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// Function-tool name, description, and JSON parameter schema.
pub struct Function {
    /// Function name exposed to the model.
    pub name: String,
    /// Optional human-readable function description.
    pub description: Option<String>,
    /// JSON Schema describing accepted function arguments.
    pub parameters: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
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
#[derive(Debug, Clone, Deserialize, Serialize)]
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
        /// Reasoning content for reasoning-capable models. The request field
        /// is named `reasoning`, while responses emit `reasoning_content`.
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

/// OpenAI usage fields plus optional generated-image lifecycle accounting.
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
    /// Number of generated images.
    pub image_count: u32,
    /// Total denoising steps across generated images.
    pub image_steps: u32,
    /// Denoising steps observed for each generated image, omitted when no
    /// image completed.
    pub image_steps_per_image: Option<Vec<u32>>,
}

impl Usage {
    /// Creates the usage of one chat response from its token counts, the
    /// engine's image totals, and the denoising steps of each image it
    /// completed.
    pub fn new(
        prompt_tokens: u32,
        completion_tokens: u32,
        image_count: u32,
        image_steps: u32,
        image_steps_per_image: Vec<u32>,
    ) -> Self {
        Self {
            prompt_tokens,
            // `prompt_tokens + completion_tokens` can exceed `u32::MAX`, which
            // would panic in debug builds and wrap in release builds.
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            completion_tokens,
            image_count,
            image_steps,
            image_steps_per_image: (!image_steps_per_image.is_empty())
                .then_some(image_steps_per_image),
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
/// `validate_chat_cross_parameters`. The served-model
/// and `prompt_logprobs` checks run later in `validate_request_compat`, and
/// `chat_completions::lower_chat_request` lowers the request into serving
/// inputs. Besides the OpenAI fields, the schema accepts sampling and output
/// extensions such as `top_k`, `min_p`, `prompt_logprobs`, and
/// `return_token_ids`.
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
    /// parsed by `convert_logit_bias` during lowering, not by serde.
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
    /// Presence-based token penalty in the inclusive range `[-2, 2]`.
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,
    /// Deterministic sampling seed. `apply_sampling` reinterprets the signed
    /// value's bit pattern as `u64`.
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
    /// Image-generation controls when image output is requested.
    pub image_config: Option<ChatImageConfig>,
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
    /// and its interaction with `stream` are checked by
    /// `validate_request_compat`.
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
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// Image-generation controls embedded in a chat request.
pub struct ChatImageConfig {
    /// Named output resolution preset.
    pub resolution: Option<models::omni::resolution::ResolutionName>,
    /// Explicit output height in pixels.
    pub height: Option<u32>,
    /// Explicit output width in pixels.
    pub width: Option<u32>,
    /// Number of denoising steps per image.
    pub steps: Option<u16>,
    /// Text classifier-free-guidance scale.
    pub guidance_scale: Option<f32>,
    /// Image classifier-free-guidance scale.
    pub image_guidance_scale: Option<f32>,
    /// Image-generation random seed.
    pub seed: Option<u64>,
    /// Maximum number of generated images.
    pub num_images: Option<u16>,
    /// Classifier-free-guidance renormalization policy.
    pub cfg_norm: Option<proto::request::CfgRenorm>,
    /// Minimum scale applied by guidance renormalization.
    pub cfg_renorm_min: Option<f32>,
    /// Fractional denoising interval in which guidance is active.
    pub cfg_interval: Option<[f32; 2]>,
    /// Diffusion scheduler timestep shift.
    pub timestep_shift: Option<f32>,
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
    pub finish_reason: Option<String>,
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
    /// Parsed reasoning content, when requested.
    #[serde(rename = "reasoning_content")]
    pub reasoning: Option<String>,
    /// Generated images as `image_url` parts carrying PNG data URLs, when
    /// produced.
    pub images: Option<Vec<ContentPart>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
/// One server-sent chat-completion chunk.
///
/// Chunks have no `prompt_logprobs` field; `validate_request_compat` rejects
/// a streamed request that asks for a positive or `-1` `prompt_logprobs`
/// count.
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
    pub finish_reason: Option<String>,
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
    #[serde(rename = "reasoning_content")]
    pub reasoning: Option<String>,
    /// Generated image content carried by this chunk.
    pub images: Option<Vec<ContentPart>>,
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

    if let (Some(minimum), Some(maximum)) = (request.min_tokens, request.max_completion_tokens)
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

    let image_output = request.modalities.contains(&ChatModality::Image);
    if !image_output && request.image_config.is_some() {
        return Err(validator::ValidationError::new(
            "image_config_without_image",
        ));
    }
    if let Some(config) = &request.image_config {
        if config.width.is_some() != config.height.is_some() {
            return Err(validator::ValidationError::new("image_config_dimensions"));
        }
        if config.num_images == Some(0) || config.steps == Some(0) {
            return Err(validator::ValidationError::new("image_config_bounds"));
        }
    }

    // An explicit `auto` without a non-empty tool list is refused; an absent
    // choice defaults to `auto`, which serves no tools when the list is empty.
    if request.tool_choice == Some(ChatToolChoice::Auto)
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

/// Configured OpenAI-compatible image-generation request for
/// `/v1/images/generations`.
///
/// Serde enforces the field set and types. The route generates exactly one
/// image, so the request carries no image count. The value checks (non-blank
/// prompt, served model name, well-formed `size`) run in
/// `images::lower_image_request` when the request is lowered,
/// and the resolved image controls are checked by `ImageParams::validate`.
#[derive(Debug, Clone, Deserialize, PartialEq, Validate)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationRequest {
    /// Text prompt describing the requested image.
    pub prompt: String,
    /// Served model name; the served-model check is skipped when omitted.
    #[serde(default)]
    pub model: Option<String>,
    /// Output size as `WIDTHxHEIGHT` in pixels. When omitted, the profile's
    /// resolution policy chooses both dimensions.
    #[serde(default)]
    pub size: Option<String>,
    /// Number of denoising steps per image.
    #[serde(default)]
    pub steps: Option<u16>,
    /// Image-generation random seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Text prompt describing content to suppress.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Text classifier-free-guidance scale.
    #[serde(default)]
    pub guidance_scale: Option<f32>,
    /// Image classifier-free-guidance scale.
    #[serde(default)]
    pub image_guidance_scale: Option<f32>,
    /// Classifier-free-guidance renormalization policy.
    #[serde(default)]
    pub cfg_norm: Option<proto::request::CfgRenorm>,
    /// Fractional denoising interval in which guidance is active.
    #[serde(default)]
    pub cfg_interval: Option<[f32; 2]>,
    /// Diffusion scheduler timestep shift.
    #[serde(default)]
    pub timestep_shift: Option<f32>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
/// Completed OpenAI-compatible image-generation response.
pub struct ImageGenerationResponse {
    /// Response creation timestamp in Unix seconds.
    pub created: u64,
    /// Generated images in the order their `ImageDone` events arrived.
    pub data: Vec<GeneratedImageData>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
/// Base64-encoded generated image and optional revised prompt.
pub struct GeneratedImageData {
    /// Base64-encoded PNG image bytes.
    pub b64_json: String,
    /// Image height in pixels.
    pub height: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Encoded image length in bytes.
    pub bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_request_schema_rejects_unknown_controls() {
        let value = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "unsupported_option": true
        });
        assert!(serde_json::from_value::<ChatCompletionRequest>(value).is_err());
    }

    #[test]
    fn image_request_rejects_unknown_or_untyped_fields() {
        let unknown =
            serde_json::from_str::<ImageGenerationRequest>(r#"{"prompt":"draw","width":1024}"#)
                .expect_err("unknown image controls must not bypass adapter validation");
        assert!(unknown.to_string().contains("unknown field `width`"));

        serde_json::from_str::<ImageGenerationRequest>(r#"{"prompt":"draw","steps":70000}"#)
            .expect_err("steps must fit the typed image-step range");
    }
}
