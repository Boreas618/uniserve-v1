//! The request side of chat: the Hugging Face Jinja chat template that renders a request's
//! messages, tools and template arguments into a prompt.
//!
//! A chat family's binding builds one [`HfChatRenderer`] from one template source: the
//! deployment's `chat_config.template` when it states one, else the checkpoint's own template,
//! read where the family keeps it. For each chat prompt, [`HfChatRenderer::render`] renders the
//! request into the text the daemon tokenizes; a family that takes images then cuts the text
//! at each image's placeholder token, where the image's block stands (`crate::program`).
//!
//! Rendering converts each message of the processing interface (`scheduling::processing::Message`)
//! into a `TemplateMessage`, the JSON shape that Hugging Face templates inspect: assistant
//! messages split into visible `content`, `reasoning_content`, and `tool_calls`, and message
//! content takes the shape selected by `ChatTemplateContentFormat` (configured, or detected in
//! `format` when unset). The template is given the tools the model may call. The template
//! environment's compatibility with Python Jinja lives in `python`.

use scheduling::processing::{
    Content, ContentPart, Function, Message, Refusal, Render, Tool, ToolCall,
};
use serde::Serialize;
use serde_json::{Map, Value as JsonValue};

use self::format::ChatTemplateContentFormat;
use self::jinja::{CompiledChatTemplate, RequestVariables, TemplateContext};
use self::metadata::SpecialTokens;
use self::python::{TemplateValue, to_template_value};

mod format;
mod jinja;
pub mod metadata;
mod python;

pub use jinja::TemplateError;

/// How prompt rendering ends after the existing chat history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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

#[derive(Debug, Clone)]
/// Image rendering settings for a renderer that accepts image parts.
///
/// Without this value, rendering refuses any image part. A binding sets it for a family that
/// takes images (`crate::ImageFamily`), with the family's image markup as the placeholder,
/// and leaves it unset for every other family, whose image parts are refused.
pub struct MultimodalRenderInfo {
    /// Text substituted for each image part when content is flattened to a
    /// string. With list-shaped content, image parts become `{"type": "image"}`
    /// items and this token is unused.
    pub placeholder_token: String,
}

/// Hugging Face chat-template renderer for one served model.
///
/// Holds at most one compiled template together with the default template
/// kwargs and tokenizer special tokens installed into every render.
pub struct HfChatRenderer {
    // `None` when neither the deployment nor the checkpoint states a template; `render` then
    // refuses every request.
    default_template: Option<CompiledChatTemplate>,
    default_template_kwargs: Map<String, JsonValue>,
    special_tokens: Option<SpecialTokens>,
    multimodal: Option<MultimodalRenderInfo>,
}

impl HfChatRenderer {
    /// Creates a renderer from an optional template source.
    ///
    /// Resolves the content format (detecting it from the template when unset) and compiles the
    /// template eagerly, so a template that does not compile is refused here rather than on the
    /// first request. A `None` template yields a renderer whose `render` always fails.
    ///
    /// # Errors
    ///
    /// The template's [`TemplateError`] when it is empty or does not compile.
    pub fn new(
        template: Option<String>,
        default_template_kwargs: Map<String, JsonValue>,
        content_format: Option<ChatTemplateContentFormat>,
    ) -> Result<Self, TemplateError> {
        Ok(Self {
            default_template: template
                .map(|template| CompiledChatTemplate::new(template, content_format))
                .transpose()?,
            default_template_kwargs,
            special_tokens: None,
            multimodal: None,
        })
    }

    /// Attaches tokenizer special tokens to the renderer.
    #[must_use]
    pub fn with_special_tokens(mut self, special_tokens: Option<SpecialTokens>) -> Self {
        self.special_tokens = special_tokens;
        self
    }

    /// Attaches image rendering settings; with `None`, any image part fails rendering.
    #[must_use]
    pub fn with_multimodal(mut self, multimodal: Option<MultimodalRenderInfo>) -> Self {
        self.multimodal = multimodal;
        self
    }

    /// Renders one chat request into prompt text for the model tokenizer,
    /// ending as `mode` states.
    ///
    /// The template sees the request's messages, the tools the model may
    /// call, its `reasoning_effort` and its template arguments over the
    /// server's defaults.
    ///
    /// # Errors
    ///
    /// Refuses an empty history, a `mode` that continues a history whose last message is not
    /// the assistant's, a renderer with no template, an image part without
    /// [`MultimodalRenderInfo`] or in an assistant message, template arguments that name a
    /// variable the request itself determines, an assistant tool call without arguments or
    /// whose arguments are not JSON (naming the field at fault), and a template that fails to
    /// render.
    pub fn render(&self, request: &Render, mode: GenerationPromptMode) -> Result<String, Refusal> {
        if request.messages.is_empty() {
            return Err(Refusal::new(
                "chat request must contain at least one message",
            ));
        }
        let ends_with_assistant =
            matches!(request.messages.last(), Some(Message::Assistant { .. }));
        if mode == GenerationPromptMode::ContinueFinalAssistant && !ends_with_assistant {
            return Err(Refusal::new(
                "cannot continue the final message when the last message is not from the \
                 assistant",
            ));
        }
        let template = self
            .default_template
            .as_ref()
            .ok_or_else(|| Refusal::new("chat template is required but none was configured"))?;

        // Template arguments may not replace the variables the render takes from the request.
        if let Some(kwargs) = request.template_kwargs.as_ref()
            && let Some(name) = RequestVariables::names()
                .into_iter()
                .find(|name| kwargs.contains_key(name))
        {
            return Err(Refusal::of(
                "chat_template_kwargs",
                format!(
                    "chat_template_kwargs may not set `{name}`, which the request itself \
                     determines."
                ),
            ));
        }

        let messages = to_template_messages(
            &request.messages,
            template.content_format(),
            self.multimodal.as_ref(),
        )?;
        let tools = request.tools.as_deref().map(to_template_tools);

        // The request's template arguments replace defaults of the same name.
        let merged_kwargs;
        let template_kwargs = match request.template_kwargs.as_ref() {
            Some(request_kwargs) if !request_kwargs.is_empty() => {
                merged_kwargs = self
                    .default_template_kwargs
                    .iter()
                    .chain(request_kwargs)
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect::<Map<_, _>>();
                &merged_kwargs
            }
            _ => &self.default_template_kwargs,
        };
        template
            .apply(TemplateContext {
                request: RequestVariables {
                    messages: &messages,
                    add_generation_prompt: mode == GenerationPromptMode::StartNewAssistant,
                    continue_final_message: mode == GenerationPromptMode::ContinueFinalAssistant,
                    tools: tools.as_deref(),
                    documents: None,
                },
                template_kwargs: Some(template_kwargs),
                special_tokens: self.special_tokens.as_ref(),
                reasoning_effort: request.reasoning_effort.as_deref(),
            })
            .map_err(|error| Refusal::new(error.to_string()))
    }
}

/// The refusal of an image part a renderer cannot place.
fn unsupported_image() -> Refusal {
    Refusal::new("unsupported multimodal content: image_url")
}

/// Chat message in the JSON shape expected by Jinja chat templates.
///
/// `None` fields are omitted rather than serialized as null, so templates can
/// test them with `is defined`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Serialize)]
struct TemplateMessage {
    role: &'static str,
    content: TemplateContent,
    // Developer-role messages may provide message-local tools in the same shape
    // as top-level request tools.
    tools: Option<Vec<TemplateTool>>,
    // Assistant reasoning, kept apart from `content` so each template decides
    // whether to replay it.
    reasoning_content: Option<String>,
    // Function-call-capable templates commonly expect assistant tool calls
    // under this OpenAI-compatible field name.
    tool_calls: Option<Vec<TemplateToolCall>>,
    // Tool-role messages refer back to the assistant call they are answering.
    tool_call_id: Option<String>,
}

/// Chat content in the two shapes HF templates commonly expect: a plain string
/// or an OpenAI-style list of typed parts.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum TemplateContent {
    String(String),
    OpenAi(Vec<TemplateContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TemplateContentPart {
    Text { text: String },
    Image,
}

#[derive(Debug, Serialize)]
struct TemplateToolCall {
    id: String,
    r#type: &'static str, // always "function"
    function: TemplateToolFunction,
}

#[derive(Debug, Serialize)]
struct TemplateToolFunction {
    name: String,
    arguments: TemplateValue,
}

#[derive(Debug, Serialize)]
/// Template-facing function tool representation.
pub(super) struct TemplateTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: TemplateToolDefinition,
}

/// Function definition in the shape of an OpenAI request tool.
///
/// `None` fields are omitted rather than serialized as null, so a template
/// that renders the tool with `tojson` sees only the fields a client sets,
/// as Hugging Face `apply_chat_template` renders the client's tool objects.
#[serde_with::skip_serializing_none]
#[derive(Debug, Serialize)]
struct TemplateToolDefinition {
    name: String,
    description: Option<String>,
    parameters: TemplateValue,
    strict: Option<bool>,
}

/// Converts chat messages into the value shape expected by Jinja chat templates.
fn to_template_messages(
    messages: &[Message],
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<Vec<TemplateMessage>, Refusal> {
    messages
        .iter()
        .map(|message| to_template_message(message, content_format, multimodal))
        .collect()
}

/// Converts one message into the renderer's template value model.
///
/// An assistant message's visible text is its text content, its reasoning a
/// non-empty `reasoning`, and its tool calls carry their arguments as parsed
/// JSON; a developer message's tools reach the template
/// as request tools do, an empty list as none.
fn to_template_message(
    message: &Message,
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<TemplateMessage, Refusal> {
    let message_of = |role, content: &Content| -> Result<TemplateMessage, Refusal> {
        Ok(TemplateMessage {
            role,
            content: to_template_content(content, content_format, multimodal)?,
            tools: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        })
    };
    Ok(match message {
        Message::System { content } => message_of("system", content)?,
        Message::Developer { content, tools } => TemplateMessage {
            tools: tools
                .as_deref()
                .filter(|tools| !tools.is_empty())
                .map(to_template_tools),
            ..message_of("developer", content)?
        },
        Message::User { content } => message_of("user", content)?,
        Message::Assistant {
            content,
            tool_calls,
            reasoning,
        } => {
            let text = match content {
                None => String::new(),
                Some(Content::Text(text)) => text.clone(),
                Some(Content::Parts(parts)) => parts
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text { text } => Ok(text.as_str()),
                        ContentPart::Image => Err(unsupported_image()),
                    })
                    .collect::<Result<String, Refusal>>()?,
            };
            TemplateMessage {
                reasoning_content: reasoning.clone().filter(|text| !text.is_empty()),
                tool_calls: to_template_tool_calls(tool_calls)?,
                ..message_of("assistant", &Content::Text(text))?
            }
        }
        Message::Tool {
            content,
            tool_call_id,
        } => TemplateMessage {
            tool_call_id: Some(tool_call_id.clone()),
            ..message_of("tool", content)?
        },
    })
}

/// Parses assistant tool arguments and converts complete calls into template values.
///
/// # Errors
///
/// Refuses, naming `messages`, a call that states no arguments (the OpenAI schema requires
/// them) or whose arguments are not JSON.
fn to_template_tool_calls(calls: &[ToolCall]) -> Result<Option<Vec<TemplateToolCall>>, Refusal> {
    let mut tool_calls = Vec::new();
    let invalid = |message: String| Refusal::of("messages", message);

    // Arguments reach the template as a parsed JSON value rather than JSON
    // text, so templates can index or iterate them (`arguments.items()`).
    for tool_call in calls {
        let arguments = tool_call.function.arguments.as_deref().ok_or_else(|| {
            invalid(format!(
                "assistant tool call `{}` states no arguments",
                tool_call.id
            ))
        })?;
        let arguments = serde_json::from_str(arguments).map_err(|error| {
            invalid(format!(
                "assistant tool call `{}` has invalid JSON arguments: {error}",
                tool_call.id
            ))
        })?;
        let arguments = to_template_value(arguments);

        tool_calls.push(TemplateToolCall {
            id: tool_call.id.clone(),
            r#type: "function",
            function: TemplateToolFunction {
                name: tool_call.function.name.clone(),
                arguments,
            },
        });
    }

    Ok((!tool_calls.is_empty()).then_some(tool_calls))
}

/// Converts message content according to the template's resolved representation.
///
/// Under `Preserve`, plain text stays a string and part lists stay lists, so
/// templates that branch on `content is string` see the caller's shape.
fn to_template_content(
    content: &Content,
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<TemplateContent, Refusal> {
    Ok(match content_format {
        ChatTemplateContentFormat::String => {
            TemplateContent::String(to_template_string_content(content, multimodal)?)
        }
        ChatTemplateContentFormat::OpenAi => {
            TemplateContent::OpenAi(to_template_openai_content(content, multimodal)?)
        }
        ChatTemplateContentFormat::Preserve => match content {
            Content::Text(text) => TemplateContent::String(text.clone()),
            Content::Parts(_) => {
                TemplateContent::OpenAi(to_template_openai_content(content, multimodal)?)
            }
        },
    })
}

/// Converts message content into an OpenAI-style list of typed parts.
fn to_template_openai_content(
    content: &Content,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<Vec<TemplateContentPart>, Refusal> {
    match content {
        Content::Text(text) => Ok(vec![TemplateContentPart::Text { text: text.clone() }]),
        Content::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(TemplateContentPart::Text { text: text.clone() }),
                // Image parts reach the template as `{"type": "image"}`; the
                // image payload itself is not exposed to the template.
                ContentPart::Image => {
                    multimodal.ok_or_else(unsupported_image)?;
                    Ok(TemplateContentPart::Image)
                }
            })
            .collect(),
    }
}

/// Flattens content into a string while substituting multimodal placeholder tokens.
fn to_template_string_content(
    content: &Content,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<String, Refusal> {
    match content {
        Content::Text(text) => Ok(text.clone()),
        Content::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => out.push_str(text),
                    ContentPart::Image => {
                        let multimodal = multimodal.ok_or_else(unsupported_image)?;
                        out.push_str(&multimodal.placeholder_token);
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Converts request tools into template values.
fn to_template_tools(tools: &[Tool]) -> Vec<TemplateTool> {
    tools
        .iter()
        .map(|Tool::Function { function }| {
            let Function {
                name,
                description,
                parameters,
                strict,
            } = function;
            TemplateTool {
                tool_type: "function",
                function: TemplateToolDefinition {
                    name: name.clone(),
                    description: description.clone(),
                    parameters: to_template_value(parameters.clone()),
                    strict: *strict,
                },
            }
        })
        .collect()
}

/// A ChatML template of the output formats the families parse (`<think>` reasoning, JSON tool
/// calls in `<tool_call>` tags): the template the renderer's and the bindings' tests render
/// realistic prompts with.
#[cfg(test)]
pub(crate) const CHATML_TEMPLATE: &str = include_str!("chatml.jinja");

#[cfg(test)]
mod tests;
