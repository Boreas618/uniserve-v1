//! The DeepSeek-V4.1 prompt format: a port of the checkpoint's reference encoder,
//! `encoding/encoding.py` (`encode_messages`), with the `developer` role vLLM's port of it adds.
//!
//! A conversation ([`Message`]s, in order) is encoded in the reference's three steps:
//!
//! 1. **Merging.** The format has no tool role: each tool result becomes a `<tool_result>` block
//!    of the user turn before it, or of a user turn of its own, and consecutive user messages
//!    merge into one turn ([`merge`]). The tool results of a turn are then ordered by the order
//!    of the calls of the assistant turn they answer ([`order_tool_results`]).
//! 2. **Dropping reasoning.** In thinking mode, unless some turn declares tools, the reasoning of
//!    the assistant turns before the last user turn is dropped, and so are the developer turns
//!    before it ([`drop_thinking`]). A user turn of tool results alone does not start a turn of
//!    its own ([`last_user_index`]).
//! 3. **Rendering.** The prompt is the BOS token and each turn's text ([`render`]): the reasoning
//!    effort and the system token at the conversation's start, the tools block in the turn that
//!    declares tools, an assistant turn's reasoning, content and DSML tool calls and its EOS
//!    token, and after the turn the model answers, the assistant token and the thinking token
//!    that opens (`<think>`) or closes (`</think>`) its reasoning.
//!
//! Special tokens are written as their text, which the tokenizer encodes as the tokens.
//!
//! The port covers the reference's text format: its roles (with the latest reminder), its task
//! tokens and image placeholders included, so the reference's own cases check it whole
//! (`vectors.json`). What a request can state of it is the program's to decide
//! (`super::request`). Tool namespaces (`ns::name`, which vLLM's port drops too), a message's
//! `response_format` and the reference's `wo_eos` flag are not ported: no request states them.

use std::collections::HashMap;

use scheduling::processing::Refusal;
use serde_json::{Map, Value};

use crate::python_json;

/// The token that begins the conversation.
pub const BOS: &str = "<｜begin▁of▁sentence｜>";
/// The token that ends an assistant turn.
pub const EOS: &str = "<｜end▁of▁sentence｜>";
/// The token that opens reasoning.
pub const THINK_START: &str = "<think>";
/// The token that closes reasoning.
pub const THINK_END: &str = "</think>";
/// The markup token of DSML tool calls.
pub const DSML: &str = "｜DSML｜";
/// The token an image stands at in the prompt.
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const SYSTEM: &str = "<｜System｜>";
const LATEST_REMINDER: &str = "<｜latest_reminder｜>";

/// The opening of a turn's tool calls, a DSML `calls` block, and of one call in it. The tag names
/// of V4.1 begin with a space (V4's were `tool_calls`, `invoke` and `parameter`).
pub const CALLS_START: &str = "<｜DSML｜ calls>";
/// The end of a DSML `calls` block.
pub const CALLS_END: &str = "</｜DSML｜ calls>";
/// The beginning of a call's tag, before the function's name.
pub const INVOKE_START: &str = "<｜DSML｜ invoke name=\"";
/// The end of a call.
pub const INVOKE_END: &str = "</｜DSML｜ invoke>";
/// The beginning of an argument's tag, before its attributes.
pub const PARAMETER_START: &str = "<｜DSML｜ parameter";
/// The end of an argument's value.
pub const PARAMETER_END: &str = "</｜DSML｜ parameter>";
/// What the format writes between a turn's content and its `calls` block.
pub const CALLS_SEPARATOR: &str = "\n\n";

/// The tools block of the turn that declares tools, before the tools' schemas, one JSON object a
/// line (`TOOLS_TEMPLATE`). The thinking tokens it names are written as the tokens.
const TOOLS_PREAMBLE: &str = r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

"#;

/// The tools block's end, after the schemas.
const TOOLS_EPILOGUE: &str = "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

/// Whether the model reasons before it answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The model answers directly: its turn opens with `</think>`.
    Chat,
    /// The model reasons first: its turn opens with `<think>`, and the conversation's start
    /// states the reasoning effort.
    Thinking,
}

/// How a conversation is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Whether the model reasons.
    pub mode: Mode,
    /// Whether the reasoning of turns before the last user turn is dropped, which a turn that
    /// declares tools overrides.
    pub drop_thinking: bool,
    /// The reasoning effort, a budget from 1 to 100, stated at the conversation's start in
    /// thinking mode.
    pub effort: u8,
}

/// A task token, which asks for one of the model's auxiliary outputs in place of a turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    /// Whether the request needs a web search (`<｜action｜>`), asked after the assistant token.
    Action,
    /// Search queries for the request (`<｜query｜>`).
    Query,
    /// How authoritative the request's sources must be (`<｜authority｜>`).
    Authority,
    /// The request's domain (`<｜domain｜>`).
    Domain,
    /// A title for the conversation (`<｜title｜>`), asked after an assistant turn.
    Title,
    /// Whether each URL of the request is read (`<｜read_url｜>`).
    ReadUrl,
}

impl Task {
    /// The task's token.
    fn token(self) -> &'static str {
        match self {
            Self::Action => "<｜action｜>",
            Self::Query => "<｜query｜>",
            Self::Authority => "<｜authority｜>",
            Self::Domain => "<｜domain｜>",
            Self::Title => "<｜title｜>",
            Self::ReadUrl => "<｜read_url｜>",
        }
    }
}

/// One message of a conversation.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    /// A system instruction: the conversation's own at its start, a turn of its own later.
    System {
        /// Its text.
        content: String,
        /// The schemas of the tools it declares: each the function object rendered as JSON.
        tools: Vec<Value>,
    },
    /// A developer instruction, encoded as a user turn.
    Developer {
        /// Its text, which may not be empty.
        content: String,
        /// The schemas of the tools it declares.
        tools: Vec<Value>,
    },
    /// A user's message.
    User {
        /// Its blocks, in order.
        content: Vec<Block>,
        /// The task its turn asks for in place of an answer.
        task: Option<Task>,
    },
    /// The latest reminder (the date, the locale, ...).
    LatestReminder {
        /// Its text.
        content: String,
    },
    /// An earlier assistant turn.
    Assistant {
        /// Its reasoning; empty when it states none.
        reasoning: String,
        /// Its visible text.
        content: String,
        /// The calls it made.
        tool_calls: Vec<Call>,
        /// The task asked after it.
        task: Option<Task>,
    },
    /// A tool's result, merged into a user turn.
    Tool {
        /// The call it answers.
        tool_call_id: String,
        /// Its text.
        content: String,
    },
}

/// One block of a user turn.
#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    /// Text.
    Text(String),
    /// An image, which the prompt places as [`IMAGE_PLACEHOLDER`].
    Image,
    /// A tool's result.
    ToolResult {
        /// The call it answers.
        tool_use_id: String,
        /// Its text.
        content: String,
    },
}

/// A call an earlier assistant turn made.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    /// The call's identifier, which orders the results that answer the turn's calls.
    pub id: String,
    /// The function called.
    pub name: String,
    /// The arguments, as the JSON text the client sent.
    pub arguments: String,
}

/// One turn of the merged conversation: a message, tool results having merged into user turns.
#[derive(Clone, Debug)]
enum Turn {
    System {
        content: String,
        tools: Vec<Value>,
    },
    Developer {
        content: String,
        tools: Vec<Value>,
    },
    User {
        blocks: Vec<Block>,
        task: Option<Task>,
    },
    LatestReminder {
        content: String,
    },
    Assistant {
        reasoning: String,
        content: String,
        tool_calls: Vec<Call>,
        task: Option<Task>,
    },
}

impl Turn {
    /// The task asked after the turn.
    fn task(&self) -> Option<Task> {
        match self {
            Self::User { task, .. } | Self::Assistant { task, .. } => *task,
            _ => None,
        }
    }

    /// The schemas of the tools the turn declares.
    fn tools(&self) -> &[Value] {
        match self {
            Self::System { tools, .. } | Self::Developer { tools, .. } => tools,
            _ => &[],
        }
    }
}

/// Encodes `messages` into the prompt's text as `options` state.
///
/// # Errors
///
/// Refuses, naming `messages`, a text that contains the image token (images are image blocks),
/// and a developer message the prompt renders that has no text.
pub fn encode(messages: Vec<Message>, options: &Options) -> Result<String, Refusal> {
    for message in &messages {
        refuse_image_token(message)?;
    }

    let mut turns = merge(messages);
    order_tool_results(&mut turns);

    // A conversation that declares tools keeps all its reasoning: its turns reason across calls.
    let drop = options.drop_thinking && !turns.iter().any(|turn| !turn.tools().is_empty());
    if options.mode == Mode::Thinking && drop {
        turns = drop_thinking(turns);
    }

    let last_user = last_user_index(&turns);
    let mut prompt = BOS.to_owned();
    for index in 0..turns.len() {
        render(&mut prompt, index, &turns, options, drop, last_user)?;
    }
    Ok(prompt)
}

/// Refuses a message whose text contains [`IMAGE_PLACEHOLDER`], which only an image block may
/// place (`_validate_no_image_sp_tokens`).
fn refuse_image_token(message: &Message) -> Result<(), Refusal> {
    let texts: Vec<&str> = match message {
        Message::System { content, .. }
        | Message::Developer { content, .. }
        | Message::LatestReminder { content }
        | Message::Tool { content, .. } => vec![content],
        Message::User { content, .. } => content
            .iter()
            .filter_map(|block| match block {
                Block::Text(text) | Block::ToolResult { content: text, .. } => Some(text.as_str()),
                Block::Image => None,
            })
            .collect(),
        Message::Assistant {
            reasoning, content, ..
        } => vec![reasoning, content],
    };
    if texts.iter().any(|text| text.contains(IMAGE_PLACEHOLDER)) {
        return Err(Refusal::of(
            "messages",
            format!(
                "message text contains the image token `{IMAGE_PLACEHOLDER}`; images are sent as \
                 image content parts"
            ),
        ));
    }
    Ok(())
}

/// Merges tool results into user turns (`merge_tool_messages`): a tool result joins the user
/// turn before it, or opens a user turn of its own; a user message joins the user turn before
/// it unless that turn asks for a task, and its own task then goes with it.
fn merge(messages: Vec<Message>) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            Message::Tool {
                tool_call_id,
                content,
            } => {
                let block = Block::ToolResult {
                    tool_use_id: tool_call_id,
                    content,
                };
                match turns.last_mut() {
                    Some(Turn::User { blocks, .. }) => blocks.push(block),
                    _ => turns.push(Turn::User {
                        blocks: vec![block],
                        task: None,
                    }),
                }
            }
            Message::User { content, task } => match turns.last_mut() {
                Some(Turn::User { blocks, task: None }) => blocks.extend(content),
                _ => turns.push(Turn::User {
                    blocks: content,
                    task,
                }),
            },
            Message::System { content, tools } => turns.push(Turn::System { content, tools }),
            Message::Developer { content, tools } => {
                turns.push(Turn::Developer { content, tools });
            }
            Message::LatestReminder { content } => turns.push(Turn::LatestReminder { content }),
            Message::Assistant {
                reasoning,
                content,
                tool_calls,
                task,
            } => turns.push(Turn::Assistant {
                reasoning,
                content,
                tool_calls,
                task,
            }),
        }
    }
    turns
}

/// Orders the tool results of each user turn of several by the calls of the last assistant turn
/// before it that made calls (`sort_tool_results_by_call_order`): a stable sort by the position
/// of the call each answers, a result of a call the turn did not make counting as the first.
/// Other blocks keep their places.
fn order_tool_results(turns: &mut [Turn]) {
    let mut order: HashMap<String, usize> = HashMap::new();
    for turn in turns {
        match turn {
            Turn::Assistant { tool_calls, .. } if !tool_calls.is_empty() => {
                order.clear();
                for (index, call) in tool_calls.iter().enumerate() {
                    if !call.id.is_empty() {
                        order.insert(call.id.clone(), index);
                    }
                }
            }
            Turn::User { blocks, .. } => {
                let positions: Vec<usize> = blocks
                    .iter()
                    .enumerate()
                    .filter(|(_, block)| matches!(block, Block::ToolResult { .. }))
                    .map(|(position, _)| position)
                    .collect();
                if positions.len() < 2 || order.is_empty() {
                    continue;
                }
                let mut results: Vec<Block> = positions
                    .iter()
                    .map(|&position| blocks[position].clone())
                    .collect();
                results.sort_by_key(|block| match block {
                    Block::ToolResult { tool_use_id, .. } => {
                        order.get(tool_use_id).copied().unwrap_or(0)
                    }
                    _ => 0,
                });
                for (position, result) in positions.into_iter().zip(results) {
                    blocks[position] = result;
                }
            }
            _ => {}
        }
    }
}

/// The index of the turn the model answers: the last user or developer turn, or a system turn
/// after the conversation's start, that is not a user turn of tool results alone
/// (`find_last_user_index`). `None` when there is none, which keeps every turn's reasoning.
fn last_user_index(turns: &[Turn]) -> Option<usize> {
    turns.iter().enumerate().rev().find_map(|(index, turn)| {
        let starts_turn = match turn {
            Turn::User { blocks, .. } => {
                blocks.is_empty()
                    || blocks
                        .iter()
                        .any(|block| matches!(block, Block::Text(_) | Block::Image))
            }
            Turn::Developer { .. } => true,
            Turn::System { .. } => index > 0,
            Turn::LatestReminder { .. } | Turn::Assistant { .. } => false,
        };
        starts_turn.then_some(index)
    })
}

/// Drops what precedes the last user turn and the model no longer reads
/// (`_drop_thinking_messages`): the reasoning of earlier assistant turns, and earlier developer
/// turns.
fn drop_thinking(turns: Vec<Turn>) -> Vec<Turn> {
    let last_user = last_user_index(&turns);
    turns
        .into_iter()
        .enumerate()
        .filter_map(|(index, turn)| {
            if last_user.is_none_or(|last| index >= last) {
                return Some(turn);
            }
            match turn {
                Turn::Assistant {
                    content,
                    tool_calls,
                    task,
                    ..
                } => Some(Turn::Assistant {
                    reasoning: String::new(),
                    content,
                    tool_calls,
                    task,
                }),
                Turn::Developer { .. } => None,
                kept => Some(kept),
            }
        })
        .collect()
}

/// Appends the text of the turn at `index` to `prompt` (`render_message`), with what follows it
/// when the model answers it: `drop` is whether earlier reasoning is dropped, `last_user` the
/// turn the model answers.
///
/// # Errors
///
/// Refuses a developer turn without text.
fn render(
    prompt: &mut String,
    index: usize,
    turns: &[Turn],
    options: &Options,
    drop: bool,
    last_user: Option<usize>,
) -> Result<(), Refusal> {
    let thinking = options.mode == Mode::Thinking;
    let turn = &turns[index];

    // The conversation opens with the system token when it states the reasoning effort or opens
    // with a system message.
    if index == 0 {
        if thinking || matches!(turn, Turn::System { .. }) {
            prompt.push_str(SYSTEM);
        }
        if thinking {
            prompt.push_str(&format!(
                "Reasoning Effort: {} (range 1-100, the higher the value, the more thorough \
                 the reasoning)\n\n",
                options.effort
            ));
        }
    }

    match turn {
        Turn::System { content, tools } => {
            if index > 0 {
                prompt.push_str(SYSTEM);
            }
            prompt.push_str(content);
            push_tools(prompt, tools);
        }
        Turn::Developer { content, tools } => {
            if content.is_empty() {
                return Err(Refusal::of(
                    "messages",
                    "a developer message must have content",
                ));
            }
            prompt.push_str(USER);
            prompt.push_str(content);
            push_tools(prompt, tools);
        }
        Turn::User { blocks, .. } => {
            prompt.push_str(USER);
            for (position, block) in blocks.iter().enumerate() {
                if position > 0 {
                    prompt.push_str("\n\n");
                }
                match block {
                    Block::Text(text) => prompt.push_str(text),
                    Block::Image => prompt.push_str(IMAGE_PLACEHOLDER),
                    Block::ToolResult { content, .. } => {
                        prompt.push_str("<tool_result>");
                        prompt.push_str(content);
                        prompt.push_str("</tool_result>");
                    }
                }
            }
        }
        Turn::LatestReminder { content } => {
            prompt.push_str(LATEST_REMINDER);
            prompt.push_str(content);
        }
        Turn::Assistant {
            reasoning,
            content,
            tool_calls,
            ..
        } => {
            // A task's answer is not reasoned; a turn before the last user turn keeps its
            // reasoning only when reasoning is not dropped. The turn's `<think>` is the token the
            // turn before it ended with.
            let answers_task = index > 0 && turns[index - 1].task().is_some();
            if thinking && !answers_task && (!drop || last_user.is_none_or(|last| index > last)) {
                prompt.push_str(reasoning);
                prompt.push_str(THINK_END);
            }
            prompt.push_str(content);
            if !tool_calls.is_empty() {
                prompt.push_str(CALLS_SEPARATOR);
                push_calls(prompt, tool_calls);
            }
            prompt.push_str(EOS);
        }
    }

    // The model answers the turn when no message follows it but an assistant turn or the latest
    // reminder.
    if turns
        .get(index + 1)
        .is_some_and(|next| !matches!(next, Turn::Assistant { .. } | Turn::LatestReminder { .. }))
    {
        return Ok(());
    }
    match turn.task() {
        Some(Task::Action) => {
            prompt.push_str(ASSISTANT);
            prompt.push_str(if thinking { THINK_START } else { THINK_END });
            prompt.push_str(Task::Action.token());
        }
        Some(task) => prompt.push_str(task.token()),
        None => {
            let answered = match turn {
                Turn::User { .. } | Turn::Developer { .. } => true,
                Turn::System { .. } => index > 0,
                Turn::LatestReminder { .. } | Turn::Assistant { .. } => false,
            };
            if answered {
                prompt.push_str(ASSISTANT);
                let reasons = thinking && (!drop || last_user.is_none_or(|last| index >= last));
                prompt.push_str(if reasons { THINK_START } else { THINK_END });
            }
        }
    }
    Ok(())
}

/// Appends the tools block of `tools` (`render_tools`), after a blank line, when there are tools.
fn push_tools(prompt: &mut String, tools: &[Value]) {
    if tools.is_empty() {
        return;
    }
    prompt.push_str("\n\n");
    prompt.push_str(TOOLS_PREAMBLE);
    for (position, tool) in tools.iter().enumerate() {
        if position > 0 {
            prompt.push('\n');
        }
        prompt.push_str(&python_json::dumps(tool));
    }
    prompt.push_str(TOOLS_EPILOGUE);
}

/// Appends the `calls` block of an assistant turn's calls: one `invoke` per call, one
/// `parameter` per argument.
fn push_calls(prompt: &mut String, calls: &[Call]) {
    prompt.push_str(CALLS_START);
    prompt.push('\n');
    for (position, call) in calls.iter().enumerate() {
        if position > 0 {
            prompt.push('\n');
        }
        prompt.push_str(INVOKE_START);
        prompt.push_str(&call.name);
        prompt.push_str("\">\n");
        for (index, (name, value)) in arguments(&call.arguments).iter().enumerate() {
            if index > 0 {
                prompt.push('\n');
            }
            // A string is written as it is, any other value as JSON; the name is written as it
            // is, as the reference writes it.
            let (string, text) = match value {
                Value::String(text) => ("true", text.clone()),
                other => ("false", python_json::dumps(other)),
            };
            prompt.push_str(&format!(
                "{PARAMETER_START} name=\"{name}\" string=\"{string}\">{text}{PARAMETER_END}"
            ));
        }
        prompt.push('\n');
        prompt.push_str(INVOKE_END);
    }
    prompt.push('\n');
    prompt.push_str(CALLS_END);
}

/// The arguments object a call's JSON text states (`encode_arguments_to_dsml`): the object it
/// decodes to, decoded twice when it is a JSON string of the object's text; text that decodes to
/// no object is the one argument `arguments`, as the reference tolerates it.
fn arguments(text: &str) -> Map<String, Value> {
    let mut value = Value::String(text.to_owned());
    for _ in 0..2 {
        let Value::String(encoded) = &value else {
            break;
        };
        match serde_json::from_str(encoded) {
            Ok(decoded) => value = decoded,
            Err(_) => break,
        }
    }
    match value {
        Value::Object(arguments) => arguments,
        _ => Map::from_iter([("arguments".to_owned(), Value::String(text.to_owned()))]),
    }
}
