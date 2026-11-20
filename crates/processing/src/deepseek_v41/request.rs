//! A chat request as the DeepSeek-V4.1 format encodes it, as vLLM's `deepseek_v41` tokenizer
//! mode states it (`vllm/tokenizers/deepseek_v41.py`, `apply_chat_template`), over the port of
//! the reference encoder ([`super::encoding`]).
//!
//! - **Messages.** Each message keeps its role. A message's content parts are its text parts
//!   joined by a blank line; an image part is refused, since the family serves text. An assistant
//!   message's `reasoning` is its reasoning, and its tool calls keep the JSON text of their
//!   arguments.
//! - **Tools.** The request's tools are declared by its first system message, wherever it stands,
//!   or by an empty system message at the conversation's start when it has none. Each tool's
//!   schema is its function object as vLLM dumps it: `name`, `description` (`null` when absent),
//!   `parameters`, and `strict` when the client states it. A developer message's own tools are
//!   rendered as the client states them, which vLLM passes through: an absent field is absent.
//! - **Template arguments.** The request's `chat_template_kwargs` replace the deployment's by
//!   name, and its `reasoning_effort` replaces theirs. Of them the format reads four, as vLLM
//!   does, and ignores the rest:
//!   - `thinking` or `enable_thinking` selects thinking mode when either is true as Python reads
//!     a value (`null`, `false`, `0`, `""`, `[]` and `{}` are false); thinking mode is the default
//!     when neither is stated. A request `reasoning_effort` sets `enable_thinking` to whether it is
//!     other than `none`, unless the request's own arguments state `enable_thinking`.
//!   - `reasoning_effort` is the budget stated at the conversation's start in thinking mode: an
//!     integer from 1 to 100, or `low` (50), `high` (75), `xhigh` (75) or `max` (100); `high` when
//!     unstated. `none` selects chat mode. Any other value is refused, in chat mode too.
//!   - `drop_thinking` (true unless stated otherwise) drops the reasoning of the turns before the
//!     last user turn, unless the conversation declares tools.

use scheduling::processing::{Content, ContentPart, Message, Refusal, Render, Tool, ToolCall};
use serde_json::{Map, Value, json};

use super::encoding::{self, Block, Call, Mode, Options};
use crate::chat::ChatEncoder;

/// The family's prompt encoder: vLLM's reading of a request over the reference's format.
pub struct Encoder;

impl ChatEncoder for Encoder {
    fn encode(&self, request: &Render, defaults: &Map<String, Value>) -> Result<String, Refusal> {
        if request.messages.is_empty() {
            return Err(Refusal::new(
                "chat request must contain at least one message",
            ));
        }
        let options = options(request, defaults)?;
        let messages = conversation(request)?;
        encoding::encode(messages, &options)
    }
}

/// The encoding options a request's template arguments and reasoning effort state, over the
/// deployment's template arguments `defaults`.
///
/// # Errors
///
/// Refuses a reasoning effort the format does not take, naming the field that stated it.
fn options(request: &Render, defaults: &Map<String, Value>) -> Result<Options, Refusal> {
    let stated = request.template_kwargs.clone().unwrap_or_default();
    let mut arguments = defaults.clone();
    arguments.extend(
        stated
            .iter()
            .map(|(name, value)| (name.clone(), value.clone())),
    );
    if let Some(effort) = &request.reasoning_effort {
        arguments.insert("reasoning_effort".to_owned(), Value::String(effort.clone()));
        if !stated.contains_key("enable_thinking") {
            arguments.insert("enable_thinking".to_owned(), Value::Bool(effort != "none"));
        }
    }

    let thinking = arguments.get("thinking");
    let enable_thinking = arguments.get("enable_thinking");
    let reasons = (thinking.is_none() && enable_thinking.is_none())
        || thinking.is_some_and(truthy)
        || enable_thinking.is_some_and(truthy);
    let mut mode = if reasons { Mode::Thinking } else { Mode::Chat };

    let mut effort = arguments
        .get("reasoning_effort")
        .cloned()
        .unwrap_or(Value::Null);
    if effort == "none" {
        mode = Mode::Chat;
        effort = Value::Null;
    }
    let budget = match &effort {
        Value::Null => Some(75),
        Value::String(name) => match name.as_str() {
            "low" => Some(50),
            "high" | "xhigh" => Some(75),
            "max" => Some(100),
            _ => None,
        },
        Value::Number(number) => number
            .as_u64()
            .filter(|budget| (1..=100).contains(budget))
            .map(|budget| budget as u8),
        _ => None,
    };
    let Some(effort) = budget else {
        let field = if request.reasoning_effort.is_some() {
            "reasoning_effort"
        } else {
            "chat_template_kwargs"
        };
        return Err(Refusal::of(
            field,
            format!(
                "DeepSeek-V4.1 takes a reasoning_effort of low, high, xhigh, max or an integer \
                 within [1, 100], not {effort}"
            ),
        ));
    };

    let drop_thinking = arguments.get("drop_thinking").is_none_or(truthy);
    Ok(Options {
        mode,
        drop_thinking,
        effort,
    })
}

/// Whether Python reads `value` as true.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    }
}

/// The conversation `request` states, its tools declared by its first system message.
///
/// # Errors
///
/// Refuses an image part, and an assistant tool call without arguments.
fn conversation(request: &Render) -> Result<Vec<encoding::Message>, Refusal> {
    let mut messages = request
        .messages
        .iter()
        .map(message)
        .collect::<Result<Vec<_>, _>>()?;

    if let Some(tools) = request.tools.as_deref().filter(|tools| !tools.is_empty()) {
        let schemas = schemas(tools);
        let system = messages.iter_mut().find_map(|message| match message {
            encoding::Message::System { tools, .. } => Some(tools),
            _ => None,
        });
        match system {
            Some(declared) => *declared = schemas,
            None => messages.insert(
                0,
                encoding::Message::System {
                    content: String::new(),
                    tools: schemas,
                },
            ),
        }
    }
    Ok(messages)
}

/// One message of the conversation.
fn message(message: &Message) -> Result<encoding::Message, Refusal> {
    Ok(match message {
        Message::System { content } => encoding::Message::System {
            content: text(content)?,
            tools: Vec::new(),
        },
        Message::Developer { content, tools } => encoding::Message::Developer {
            content: text(content)?,
            tools: tools.as_deref().map(stated_schemas).unwrap_or_default(),
        },
        Message::User { content } => encoding::Message::User {
            content: vec![Block::Text(text(content)?)],
            task: None,
        },
        Message::Assistant {
            content,
            reasoning,
            tool_calls,
        } => encoding::Message::Assistant {
            reasoning: reasoning.clone().unwrap_or_default(),
            content: content.as_ref().map(text).transpose()?.unwrap_or_default(),
            tool_calls: tool_calls.iter().map(call).collect::<Result<_, _>>()?,
            task: None,
        },
        Message::Tool {
            content,
            tool_call_id,
        } => encoding::Message::Tool {
            tool_call_id: tool_call_id.clone(),
            content: text(content)?,
        },
    })
}

/// A message's text: its text parts joined by a blank line.
///
/// # Errors
///
/// Refuses an image part by its type: the family serves text.
fn text(content: &Content) -> Result<String, Refusal> {
    match content {
        Content::Text(text) => Ok(text.clone()),
        Content::Parts(parts) => {
            let texts = parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => Ok(text.as_str()),
                    ContentPart::Image => Err(Refusal::of(
                        "messages",
                        "unsupported content part type image_url: DeepSeek-V4.1 is served for \
                         text",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(texts.join("\n\n"))
        }
    }
}

/// An assistant tool call.
///
/// # Errors
///
/// Refuses a call that states no arguments, which the OpenAI schema requires.
fn call(call: &ToolCall) -> Result<Call, Refusal> {
    let arguments = call.function.arguments.clone().ok_or_else(|| {
        Refusal::of(
            "messages",
            format!("assistant tool call `{}` states no arguments", call.id),
        )
    })?;
    Ok(Call {
        id: call.id.clone(),
        name: call.function.name.clone(),
        arguments,
    })
}

/// The schemas of the request's `tools`: each function object as vLLM validates and dumps it,
/// its fields in the model's order and its description `null` when the client states none.
fn schemas(tools: &[Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|Tool::Function { function }| {
            let mut schema = json!({
                "name": function.name,
                "description": function.description,
                "parameters": function.parameters,
            });
            if let Some(strict) = function.strict {
                schema["strict"] = Value::Bool(strict);
            }
            schema
        })
        .collect()
}

/// The schemas of the tools a developer message declares: each function object as the client
/// states it, which vLLM renders as it arrives, an absent field absent.
fn stated_schemas(tools: &[Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|Tool::Function { function }| {
            let mut schema = json!({"name": function.name});
            if let Some(description) = &function.description {
                schema["description"] = Value::String(description.clone());
            }
            schema["parameters"] = function.parameters.clone();
            if let Some(strict) = function.strict {
                schema["strict"] = Value::Bool(strict);
            }
            schema
        })
        .collect()
}
