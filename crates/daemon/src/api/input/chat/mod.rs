//! The request side of chat: the Hugging Face Jinja chat template that renders a request's
//! messages, tools and template arguments into a prompt.
//!
//! `ModelConfig::load` builds one [`HfChatRenderer`] for every model except a diffusers pipeline,
//! from one template source: `chat_config.template` when the deployment states one, else the
//! checkpoint's own template. For each chat prompt, request processing calls
//! [`HfChatRenderer::render`] and tokenizes the rendered prompt; a family that takes images then
//! replaces each image's placeholder token by the image's block (`Processor::process`).
//!
//! Rendering converts each wire `ChatMessage` (`crate::api::protocol`) into a `TemplateMessage`,
//! the JSON shape that Hugging Face templates inspect: assistant messages split into visible
//! `content`, `reasoning_content`, and `tool_calls`, and message content takes the shape selected
//! by `ChatTemplateContentFormat` (configured, or detected in `format` when unset). The template
//! is given the tools the model may call (`ChatCompletionRequest::callable_tools`). The template
//! environment's compatibility with Python Jinja lives in `python`.

use std::collections::HashMap;

use crate::api::input::{ModelResolutionError, TokenizeError};
use crate::api::model::assets::{
    HfSpecialTokens, HfTokenizerConfig, ResolvedModelFiles, load_tokenizer_config,
};
use serde::Serialize;
use serde_json::Value as JsonValue;
use tracing::{info, trace};

use self::format::ChatTemplateContentFormat;
use self::jinja::{CompiledChatTemplate, RequestVariables, TemplateContext};
use self::python::{TemplateValue, to_template_value};
use crate::api::protocol::{ChatCompletionRequest, ChatMessage, ContentPart, MessageContent, Tool};
use proto::config::{ChatConfig, ChatTemplateSource};

mod format;
mod jinja;
mod python;

pub use jinja::TemplateError;

/// The name of the template file a multimodal checkpoint's processor configuration states its
/// template in, as an object with a `chat_template` string; any other template file holds the
/// template's Jinja source.
const TEMPLATE_JSON: &str = "chat_template.json";

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
/// Image rendering settings for a renderer that accepts `image_url` parts.
///
/// Without this value, rendering fails with
/// `TokenizeError::UnsupportedMultimodalContent` on any `image_url` part. Loading sets
/// it for a family that takes images (`processing::ImageFamily`), with the
/// family's image markup as the placeholder, and leaves it unset for every
/// other family, whose image parts are refused.
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
    // fails with `TokenizeError::MissingChatTemplate`.
    default_template: Option<CompiledChatTemplate>,
    default_template_kwargs: HashMap<String, JsonValue>,
    special_tokens: Option<HfSpecialTokens>,
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
        default_template_kwargs: HashMap<String, JsonValue>,
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
    pub fn with_special_tokens(mut self, special_tokens: Option<HfSpecialTokens>) -> Self {
        self.special_tokens = special_tokens;
        self
    }

    /// Attaches image rendering settings; with `None`, any `image_url` part
    /// fails rendering.
    pub fn with_multimodal(mut self, multimodal: Option<MultimodalRenderInfo>) -> Self {
        self.multimodal = multimodal;
        self
    }

    /// Creates the renderer of a checkpoint's `files` under the deployment's chat `options`.
    ///
    /// The template is the one `options.template` states, read exactly as written, when it
    /// states one; otherwise it is the checkpoint's own: the standalone template file it ships
    /// (`files.chat_template_path`), else the `chat_template` entry of its
    /// `tokenizer_config.json`. Special tokens always come from the tokenizer config; when it
    /// defines none, templates see them as undefined.
    ///
    /// # Errors
    ///
    /// Fails when the tokenizer config cannot be loaded, the template's file cannot be read, or
    /// the template is empty or does not compile.
    pub fn load(
        files: &ResolvedModelFiles,
        options: &ChatConfig,
        multimodal: Option<MultimodalRenderInfo>,
    ) -> Result<Self, ModelResolutionError> {
        let HfTokenizerConfig {
            special_tokens,
            chat_template,
            ..
        } = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let special_tokens = (!special_tokens.is_empty()).then_some(special_tokens);

        let template = match &options.template {
            Some(ChatTemplateSource::File(path)) => {
                info!(path = %path.display(), "rendering with the configured chat template file");
                Some(jinja::read_jinja(path)?)
            }
            Some(ChatTemplateSource::Inline(source)) => {
                info!("rendering with the configured chat template");
                Some(source.clone())
            }
            None => match files.chat_template_path.as_deref() {
                Some(path) if path.file_name().is_some_and(|name| name == TEMPLATE_JSON) => {
                    Some(jinja::read_template_json(path)?)
                }
                Some(path) => Some(jinja::read_jinja(path)?),
                None => chat_template,
            },
        };

        Ok(Self::new(
            template,
            options.template_kwargs.clone().unwrap_or_default(),
            options.template_content_format,
        )?
        .with_special_tokens(special_tokens)
        .with_multimodal(multimodal))
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
    /// Returns `TokenizeError::EmptyMessages` for an empty history,
    /// `TokenizeError::ContinueWithoutAssistant` when `mode` continues a history whose last
    /// message is not the assistant's, `TokenizeError::MissingChatTemplate` when the renderer has
    /// no template, `TokenizeError::UnsupportedMultimodalContent` for an `image_url` part without
    /// [`MultimodalRenderInfo`], `TokenizeError::InvalidParameter` for template arguments that
    /// name a variable the request itself determines and for an assistant tool call without
    /// arguments or whose arguments are not JSON, and `TokenizeError::Template` when the template
    /// fails to render.
    pub fn render(
        &self,
        request: &ChatCompletionRequest,
        mode: GenerationPromptMode,
    ) -> Result<String, TokenizeError> {
        if request.messages.is_empty() {
            return Err(TokenizeError::EmptyMessages);
        }
        let ends_with_assistant =
            matches!(request.messages.last(), Some(ChatMessage::Assistant { .. }));
        if mode == GenerationPromptMode::ContinueFinalAssistant && !ends_with_assistant {
            return Err(TokenizeError::ContinueWithoutAssistant);
        }
        let template = self
            .default_template
            .as_ref()
            .ok_or(TokenizeError::MissingChatTemplate)?;

        // Template arguments may not replace the variables the render takes from the request.
        if let Some(kwargs) = request.chat_template_kwargs.as_ref()
            && let Some(name) = RequestVariables::names()
                .into_iter()
                .find(|name| kwargs.contains_key(name))
        {
            return Err(TokenizeError::InvalidParameter {
                param: Some("chat_template_kwargs"),
                message: format!(
                    "chat_template_kwargs may not set `{name}`, which the request itself \
                     determines."
                ),
            });
        }

        let messages = to_template_messages(
            &request.messages,
            template.content_format(),
            self.multimodal.as_ref(),
        )?;
        let tools = request.callable_tools().map(to_template_tools);
        trace!(
            message_count = messages.len(),
            content_format = ?template.content_format(),
            ?messages,
            ?tools,
            "applying chat template"
        );

        // The request's template arguments replace defaults of the same name.
        let merged_kwargs;
        let template_kwargs = match request.chat_template_kwargs.as_ref() {
            Some(request_kwargs) if !request_kwargs.is_empty() => {
                merged_kwargs = self
                    .default_template_kwargs
                    .iter()
                    .chain(request_kwargs)
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect::<HashMap<_, _>>();
                &merged_kwargs
            }
            _ => &self.default_template_kwargs,
        };
        let prompt = template.apply(TemplateContext {
            request: RequestVariables {
                messages: &messages,
                add_generation_prompt: mode == GenerationPromptMode::StartNewAssistant,
                continue_final_message: mode == GenerationPromptMode::ContinueFinalAssistant,
                tools: tools.as_deref(),
                documents: None,
            },
            template_kwargs: Some(template_kwargs),
            special_tokens: self.special_tokens.as_ref(),
            reasoning_effort: request.reasoning_effort,
        })?;

        trace!(
            prompt_len = prompt.len(),
            prompt, "rendered chat template prompt"
        );

        Ok(prompt)
    }
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
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<Vec<TemplateMessage>, TokenizeError> {
    messages
        .iter()
        .map(|message| to_template_message(message, content_format, multimodal))
        .collect()
}

/// Converts one wire message into the renderer's template value model.
///
/// An assistant message's visible text is its text content, its reasoning a
/// non-empty `reasoning`, and its tool calls carry their arguments as parsed
/// JSON; a developer message's tools reach the template
/// as request tools do, an empty list as none.
fn to_template_message(
    message: &ChatMessage,
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<TemplateMessage, TokenizeError> {
    let message_of = |role, content: &MessageContent| -> Result<TemplateMessage, TokenizeError> {
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
        ChatMessage::System { content, .. } => message_of("system", content)?,
        ChatMessage::Developer { content, tools, .. } => TemplateMessage {
            tools: tools
                .as_deref()
                .filter(|tools| !tools.is_empty())
                .map(to_template_tools),
            ..message_of("developer", content)?
        },
        ChatMessage::User { content, .. } => message_of("user", content)?,
        ChatMessage::Assistant {
            content,
            tool_calls,
            reasoning,
            ..
        } => {
            let text = match content {
                None => String::new(),
                Some(MessageContent::Text(text)) => text.clone(),
                Some(MessageContent::Parts(parts)) => parts
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text { text } => Ok(text.as_str()),
                        ContentPart::ImageUrl { .. } => {
                            Err(TokenizeError::UnsupportedMultimodalContent("image_url"))
                        }
                    })
                    .collect::<Result<String, TokenizeError>>()?,
            };
            TemplateMessage {
                reasoning_content: reasoning.clone().filter(|text| !text.is_empty()),
                tool_calls: to_template_tool_calls(tool_calls.as_deref().unwrap_or_default())?,
                ..message_of("assistant", &MessageContent::Text(text))?
            }
        }
        ChatMessage::Tool {
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
/// `TokenizeError::InvalidParameter`, naming `messages`, for a call that states no arguments
/// (the OpenAI schema requires them) or whose arguments are not JSON.
fn to_template_tool_calls(
    calls: &[crate::api::protocol::ToolCall],
) -> Result<Option<Vec<TemplateToolCall>>, TokenizeError> {
    let mut tool_calls = Vec::new();
    let invalid = |message: String| TokenizeError::InvalidParameter {
        param: Some("messages"),
        message,
    };

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
    content: &MessageContent,
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<TemplateContent, TokenizeError> {
    Ok(match content_format {
        ChatTemplateContentFormat::String => {
            TemplateContent::String(to_template_string_content(content, multimodal)?)
        }
        ChatTemplateContentFormat::OpenAi => {
            TemplateContent::OpenAi(to_template_openai_content(content, multimodal)?)
        }
        ChatTemplateContentFormat::Preserve => match content {
            MessageContent::Text(text) => TemplateContent::String(text.clone()),
            MessageContent::Parts(_) => {
                TemplateContent::OpenAi(to_template_openai_content(content, multimodal)?)
            }
        },
    })
}

/// Converts message content into an OpenAI-style list of typed parts.
fn to_template_openai_content(
    content: &MessageContent,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<Vec<TemplateContentPart>, TokenizeError> {
    match content {
        MessageContent::Text(text) => Ok(vec![TemplateContentPart::Text { text: text.clone() }]),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(TemplateContentPart::Text { text: text.clone() }),
                // Image parts reach the template as `{"type": "image"}`; the
                // image payload itself is not exposed to the template.
                ContentPart::ImageUrl { .. } => {
                    multimodal.ok_or(TokenizeError::UnsupportedMultimodalContent("image_url"))?;
                    Ok(TemplateContentPart::Image)
                }
            })
            .collect(),
    }
}

/// Flattens content into a string while substituting multimodal placeholder tokens.
fn to_template_string_content(
    content: &MessageContent,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<String, TokenizeError> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => out.push_str(text),
                    ContentPart::ImageUrl { .. } => {
                        let multimodal = multimodal
                            .ok_or(TokenizeError::UnsupportedMultimodalContent("image_url"))?;
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
        .map(|Tool::Function { function }| TemplateTool {
            tool_type: "function",
            function: TemplateToolDefinition {
                name: function.name.clone(),
                description: function.description.clone(),
                parameters: to_template_value(function.parameters.clone()),
                strict: function.strict,
            },
        })
        .collect()
}

/// A ChatML template of the output formats the daemon parses (`<think>` reasoning, JSON tool
/// calls in `<tool_call>` tags): the template the renderer's and request processing's tests render
/// realistic prompts with.
#[cfg(test)]
pub(crate) const CHATML_TEMPLATE: &str = include_str!("chatml.jinja");

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::api::model::assets::{HfSpecialTokens, NamedSpecialToken};
    use expect_test::expect;
    use serde_json::Value;

    use super::{
        CHATML_TEMPLATE, ChatTemplateContentFormat, GenerationPromptMode, HfChatRenderer,
        MultimodalRenderInfo,
    };
    use crate::api::input::TokenizeError;
    use crate::api::protocol::{
        ChatCompletionRequest, ChatMessage, ChatToolChoice, Function, ReasoningEffort, Tool,
    };
    use crate::api::test_support::chat;

    type Result<T> = std::result::Result<T, TokenizeError>;

    fn sample_request(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        chat::request(messages)
    }

    /// Renders `request` for a new assistant turn.
    fn render(template: Option<&str>, request: &ChatCompletionRequest) -> Result<String> {
        render_as(template, request, GenerationPromptMode::StartNewAssistant)
    }

    /// Renders `request`, ending the prompt as `mode` states.
    fn render_as(
        template: Option<&str>,
        request: &ChatCompletionRequest,
        mode: GenerationPromptMode,
    ) -> Result<String> {
        HfChatRenderer::new(template.map(str::to_owned), HashMap::new(), None)?
            .render(request, mode)
    }

    fn render_mm(
        template: &str,
        request: &ChatCompletionRequest,
        content_format: Option<ChatTemplateContentFormat>,
    ) -> Result<String> {
        HfChatRenderer::new(Some(template.to_string()), HashMap::new(), content_format)?
            .with_multimodal(Some(MultimodalRenderInfo {
                placeholder_token: "<image>".to_string(),
            }))
            .render(request, GenerationPromptMode::StartNewAssistant)
    }

    fn image_request() -> ChatCompletionRequest {
        sample_request(vec![chat::user_parts(vec![
            chat::text_part("a"),
            chat::image_part("data:image/png;base64,test"),
            chat::text_part("b"),
        ])])
    }

    #[test]
    fn string_content_format_replaces_image_with_placeholder_text() {
        let rendered = render_mm(
            "{{ messages[0].content }}",
            &image_request(),
            Some(ChatTemplateContentFormat::String),
        )
        .unwrap();

        assert_eq!(rendered, "a<image>b");
    }

    #[test]
    fn openai_content_format_normalizes_image_url_for_template() {
        let rendered = render_mm(
            "{% for item in messages[0].content %}{% if item.type == 'image' %}<|image_pad|>{% else %}{{ item.text }}{% endif %}{% endfor %}",
            &image_request(),
            Some(ChatTemplateContentFormat::OpenAi),
        )
        .unwrap();

        assert_eq!(rendered, "a<|image_pad|>b");
    }

    #[test]
    fn auto_content_format_preserves_mixed_system_and_user_shapes() {
        // The template both loops over content items and tests `content is
        // string`, so `Auto` selects `Preserve`: the system string still
        // supports `+` concatenation while the user parts stay a list.
        let request = sample_request(vec![
            chat::system("policy"),
            chat::user_parts(vec![
                chat::text_part("a"),
                chat::image_part("data:image/png;base64,test"),
                chat::text_part("b"),
            ]),
        ]);
        let rendered = render_mm(
            "{% if messages[0].role == 'system' %}{{ 'S:' + messages[0].content + '|' }}{% endif %}{% for message in messages[1:] %}{% if message.content is string %}{{ message.content }}{% else %}{% for item in message.content %}{% if item.type == 'image' %}<image>{% else %}{{ item.text }}{% endif %}{% endfor %}{% endif %}{% endfor %}",
            &request,
            None,
        )
        .unwrap();

        assert_eq!(rendered, "S:policy|a<image>b");
    }

    #[test]
    fn chat_template_supports_pycompat_templates() {
        let request = sample_request(vec![chat::user("<think>hello")]);

        let rendered = render(
            Some(
                "{% for message in messages %}{% if message.content.startswith('<think>') %}think{% else %}plain{% endif %}{% endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "think");
    }

    #[test]
    fn chat_template_passes_continue_final_message_to_template() {
        let request = sample_request(vec![chat::assistant("The capital of")]);

        assert_eq!(
            render(
                Some("{% if continue_final_message %}continue{% else %}new{% endif %}"),
                &request,
            )
            .unwrap(),
            "new"
        );

        assert_eq!(
            render_as(
                Some("{% if continue_final_message %}continue{% else %}new{% endif %}"),
                &request,
                GenerationPromptMode::ContinueFinalAssistant,
            )
            .unwrap(),
            "continue"
        );
    }

    #[test]
    fn chat_template_flattens_text_parts_for_string_templates() {
        let request = sample_request(vec![chat::user_parts(vec![
            chat::text_part("hello"),
            chat::text_part(" world"),
        ])]);

        let rendered = render(Some("{{ messages[0].content }}"), &request).unwrap();

        assert_eq!(rendered, "hello world");
    }

    #[test]
    fn chat_template_exposes_developer_tools() {
        let request = sample_request(vec![chat::developer(
            "policy",
            Some(vec![Tool::Function {
                function: Function {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    }),
                    strict: Some(true),
                },
            }]),
        )]);

        let rendered = render(
            Some("{{ messages[0].role }}|{{ messages[0].content }}|{{ messages[0].tools[0].function.name }}|{{ messages[0].tools[0].function.parameters.required[0] }}"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "developer|policy|get_weather|city");
    }

    #[test]
    fn chat_template_keeps_string_text_for_openai_detected_templates() {
        // The template tests `content is string` without looping over content
        // items, so `Auto` selects `String` and plain text arrives as a string.
        let request = sample_request(vec![chat::user("hello")]);

        let rendered = render(
            Some(
                "{%- for message in messages %}{%- if message.content is string %}{%- set content = message.content %}{{ content }}{%- endif %}{%- endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "hello");
    }

    #[test]
    fn chat_template_emits_openai_text_blocks_for_structured_templates() {
        let request = sample_request(vec![chat::user_parts(vec![
            chat::text_part("hello"),
            chat::text_part("world"),
        ])]);

        let rendered = render(
            Some(
                "{%- for message in messages %}{%- for item in message.content %}{{ item.text }}|{%- endfor %}{%- endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "hello|world|");
    }

    #[test]
    fn chat_template_requires_a_template() {
        let request = sample_request(vec![chat::user("hello")]);
        let error = render(None, &request).unwrap_err();

        assert!(matches!(error, TokenizeError::MissingChatTemplate));
    }

    #[test]
    fn chat_template_injects_special_tokens_into_context() {
        let request = sample_request(vec![chat::user("hello")]);
        let special_tokens = HfSpecialTokens {
            bos_token: Some(NamedSpecialToken::Text("<bos>".to_string())),
            ..Default::default()
        };

        let rendered = HfChatRenderer::new(
            Some("{{ bos_token }}|{{ bos_token is defined }}".to_string()),
            HashMap::new(),
            None,
        )
        .unwrap()
        .with_special_tokens(Some(special_tokens))
        .render(&request, GenerationPromptMode::StartNewAssistant)
        .unwrap();

        // Chat templates follow Python Jinja2, which prints booleans as
        // Python literals.
        assert_eq!(rendered, "<bos>|True");
    }

    #[test]
    fn chat_template_exposes_assistant_reasoning_separately() {
        let request = sample_request(vec![chat::assistant_turn("inner", "outer", Vec::new())]);

        let rendered = render(
            Some("{{ messages[0].reasoning_content }}|{{ messages[0].content }}"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "inner|outer");
    }

    #[test]
    fn chat_template_forces_string_content_format_when_configured() {
        let request = sample_request(vec![chat::user_parts(vec![
            chat::text_part("hello"),
            chat::text_part(" world"),
        ])]);

        let rendered = HfChatRenderer::new(
            Some(
                "{%- if messages[0].content is string -%}{{ messages[0].content }}{%- else -%}{%- for item in messages[0].content %}{{ item.text }}|{%- endfor -%}{%- endif -%}".to_string(),
            ),
            HashMap::new(),
            Some(ChatTemplateContentFormat::String),
        )
        .unwrap()
        .render(&request, GenerationPromptMode::StartNewAssistant)
        .unwrap();

        assert_eq!(rendered, "hello world");
    }

    #[test]
    fn chat_template_forces_openai_content_format_when_configured() {
        let request = sample_request(vec![chat::user_parts(vec![
            chat::text_part("hello"),
            chat::text_part(" world"),
        ])]);

        let rendered = HfChatRenderer::new(
            Some("{{ messages[0].content[0].text }}{{ messages[0].content[1].text }}".to_string()),
            HashMap::new(),
            Some(ChatTemplateContentFormat::OpenAi),
        )
        .unwrap()
        .render(&request, GenerationPromptMode::StartNewAssistant)
        .unwrap();

        assert_eq!(rendered, "hello world");
    }

    #[test]
    fn chat_template_exposes_default_template_kwargs() {
        let request = sample_request(vec![chat::user("hello")]);
        let renderer = HfChatRenderer::new(
            Some("{{ enable_thinking }}|{{ default_only }}".to_string()),
            HashMap::from([
                ("enable_thinking".to_string(), Value::Bool(true)),
                ("default_only".to_string(), Value::String("x".to_string())),
            ]),
            None,
        )
        .unwrap();

        let rendered = renderer
            .render(&request, GenerationPromptMode::StartNewAssistant)
            .unwrap();

        assert_eq!(rendered, "True|x");
    }

    /// A request's template kwargs replace defaults of the same name and add
    /// their own; the others keep their default values.
    #[test]
    fn request_template_kwargs_override_default_template_kwargs() {
        let mut request = sample_request(vec![chat::user("hello")]);
        request.chat_template_kwargs = Some(HashMap::from([
            ("enable_thinking".to_string(), Value::Bool(true)),
            ("request_only".to_string(), Value::String("y".to_string())),
        ]));
        let renderer = HfChatRenderer::new(
            Some("{{ enable_thinking }}|{{ default_only }}|{{ request_only }}".to_string()),
            HashMap::from([
                ("enable_thinking".to_string(), Value::Bool(false)),
                ("default_only".to_string(), Value::String("x".to_string())),
            ]),
            None,
        )
        .unwrap();

        let rendered = renderer
            .render(&request, GenerationPromptMode::StartNewAssistant)
            .unwrap();

        assert_eq!(rendered, "True|x|y");
    }

    #[test]
    fn chat_template_reasoning_effort_overrides_default_template_kwargs() {
        let mut request = sample_request(vec![chat::user("hello")]);
        request.reasoning_effort = Some(ReasoningEffort::Max);

        let renderer = HfChatRenderer::new(
            Some("{{ reasoning_effort }}".to_string()),
            HashMap::from([(
                "reasoning_effort".to_string(),
                Value::String("medium".to_string()),
            )]),
            None,
        )
        .unwrap();

        let rendered = renderer
            .render(&request, GenerationPromptMode::StartNewAssistant)
            .unwrap();

        assert_eq!(rendered, "max");
    }

    #[test]
    fn chatml_template_omits_reasoning_for_prior_assistant_messages() {
        let request = sample_request(vec![
            chat::user("Hi. Tell me about the capital of France in short"),
            chat::assistant_turn(
                "\nOkay, the user is asking... I think that's all.\n",
                "Paris is the capital of France.",
                Vec::new(),
            ),
            chat::user("Tell me about Paris more."),
        ]);

        let rendered = render(Some(CHATML_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            Hi. Tell me about the capital of France in short<|im_end|>
            <|im_start|>assistant
            Paris is the capital of France.<|im_end|>
            <|im_start|>user
            Tell me about Paris more.<|im_end|>
            <|im_start|>assistant
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn chatml_template_renders_single_user_prompt() {
        let request = sample_request(vec![chat::user("hello")]);

        let rendered = render(Some(CHATML_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            hello<|im_end|>
            <|im_start|>assistant
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn chatml_template_renders_system_user_prompt() {
        let request = sample_request(vec![chat::system("be brief"), chat::user("hello")]);

        let rendered = render(Some(CHATML_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>system
            be brief<|im_end|>
            <|im_start|>user
            hello<|im_end|>
            <|im_start|>assistant
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn chatml_template_respects_forced_openai_content_format() {
        // The ChatML template emits content only when it is a string. A forced
        // `OpenAi` format still wins over detection, so the part list renders
        // as empty user content.
        let request = sample_request(vec![chat::user("hello")]);

        let rendered = HfChatRenderer::new(
            Some(CHATML_TEMPLATE.to_string()),
            HashMap::new(),
            Some(ChatTemplateContentFormat::OpenAi),
        )
        .unwrap()
        .render(&request, GenerationPromptMode::StartNewAssistant)
        .unwrap();

        expect![[r#"
            <|im_start|>user
            <|im_end|>
            <|im_start|>assistant
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn chatml_template_keeps_reasoning_after_the_last_user_query() {
        let request = sample_request(vec![
            chat::user("What is 1 + 1?"),
            chat::assistant_turn("need simple arithmetic", "2", Vec::new()),
        ]);

        let rendered = render_as(
            Some(CHATML_TEMPLATE),
            &request,
            GenerationPromptMode::NoGenerationPrompt,
        )
        .unwrap();

        expect![[r#"
            <|im_start|>user
            What is 1 + 1?<|im_end|>
            <|im_start|>assistant
            <think>
            need simple arithmetic
            </think>

            2<|im_end|>
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn chat_template_exposes_tools_to_templates_when_auto_enabled() {
        let mut request = sample_request(vec![chat::user("hello")]);
        request.tools = Some(vec![Tool::Function {
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }),
                strict: None,
            },
        }]);
        request.tool_choice = Some(ChatToolChoice::Auto);

        let rendered = render(
            Some("{{ tools[0].function.name }}|{{ tools[0].function.parameters.required[0] }}"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "get_weather|city");
    }

    /// Tools reach the template in the shape of the client's tool objects:
    /// optional fields the tool leaves unset are absent rather than null.
    #[test]
    fn chat_template_omits_unset_optional_tool_fields() {
        let parameters = serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
        });
        let mut request = sample_request(vec![chat::user("hello")]);
        request.tools = Some(vec![
            Tool::Function {
                function: Function {
                    name: "get_weather".to_string(),
                    description: None,
                    parameters: parameters.clone(),
                    strict: None,
                },
            },
            Tool::Function {
                function: Function {
                    name: "get_time".to_string(),
                    description: Some("Get time".to_string()),
                    parameters: parameters.clone(),
                    strict: Some(true),
                },
            },
        ]);
        request.tool_choice = Some(ChatToolChoice::Auto);

        let rendered = render(Some("{{ tools | tojson }}"), &request).unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&rendered).unwrap(),
            serde_json::json!([
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": parameters},
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Get time",
                        "parameters": parameters,
                        "strict": true,
                    },
                },
            ])
        );
    }

    #[test]
    fn chat_template_exposes_assistant_tool_calls_and_tool_messages() {
        let request = sample_request(vec![
            chat::assistant_turn(
                "",
                "",
                vec![chat::tool_call(
                    "call_1",
                    "get_weather",
                    r#"{"city":"Paris"}"#,
                )],
            ),
            chat::tool_response("Sunny", "call_1"),
        ]);

        let rendered = render(
            Some(
                "{{ messages[0].tool_calls[0].function.name }}|{{ messages[0].tool_calls[0].function.arguments.city }}|{{ messages[1].tool_call_id }}|{{ messages[1].content }}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "get_weather|Paris|call_1|Sunny");
    }

    #[test]
    fn chat_template_tool_call_argument_items_method_is_not_shadowed_by_field() {
        // An `items` key in tool arguments must not shadow the dict `items()`
        // method (see `TemplateMap`), and iteration follows the key order of
        // the arguments JSON.
        let request = sample_request(vec![chat::assistant_turn(
            "",
            "",
            vec![chat::tool_call(
                "call_1",
                "add",
                r#"{"items":"operands","x":2,"y":1.0}"#,
            )],
        )]);

        let rendered = render(
            Some(
                "{%- set arguments = messages[0].tool_calls[0].function.arguments -%}
{%- for key, value in arguments.items() -%}{{ key }}={{ value }};{%- endfor -%}
|{{ arguments['items'] }}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "items=operands;x=2;y=1.0;|operands");
    }
}
