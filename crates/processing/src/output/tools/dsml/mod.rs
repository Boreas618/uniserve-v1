//! The parser of DeepSeek's DSML tool-call format (`crate::ToolCallFormat::Dsml`).
//!
//! A turn ends with one `calls` block, after a blank line, of one `invoke` per call and one
//! `parameter` per argument:
//!
//! ```text
//! <｜DSML｜ calls>
//! <｜DSML｜ invoke name="NAME">
//! <｜DSML｜ parameter name="ARG" string="true">TEXT</｜DSML｜ parameter>
//! <｜DSML｜ parameter name="ARG" string="false">JSON</｜DSML｜ parameter>
//! </｜DSML｜ invoke>
//! </｜DSML｜ calls>
//! ```
//!
//! The text before the block is text, the blank line before the block excepted. A call is
//! published (its first delta names the function) at the end of its `invoke` tag, and its
//! arguments as one JSON object, written as Python's `json.dumps` writes it, in increments: an
//! argument whose `string` attribute is `true` is its text, published as it streams (`{"ARG": "`,
//! the text's escapes, then `"`); any other argument is the JSON value its text states, or its
//! text when the text is no JSON, published once its value ended (`, "ARG": VALUE`); `}` closes
//! the object at the call's end. Parameter tags are read as vLLM's DeepSeek parsers read them
//! (`vllm/parser/deepseek_v4.py`): the `string` attribute may be absent, and a value ends at its
//! closing tag or at the next parameter tag.
//!
//! The parser is lenient, as vLLM's is, and never fails the response: the model's text is shown
//! as the format reads it, and a request's length limit, which an agent's replay may set with
//! the end-of-sequence token ignored, routinely cuts a block or runs past it.
//!
//! - Text inside the block that is no `invoke`, and inside a call that is no `parameter`, is
//!   ignored; so is a parameter whose tag does not parse, and a call whose name is empty.
//! - A value ends at its call's end or the block's end too, as written.
//! - Everything after the block's end is dropped: the turn ends with its calls.
//! - At the end of generation inside the block, the open call is closed as far as the text
//!   reached: a text value is closed where the text ended, a JSON value that did not end is
//!   dropped, and a call whose `invoke` tag did not end is not published. The response's
//!   `length` finish reason states that its calls were cut.

#[cfg(test)]
mod tests;

use std::mem::take;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ToolCallDelta, ToolParser, ToolParserOutput, partial_prefix_len};
use crate::deepseek_v41::encoding::{
    CALLS_END, CALLS_SEPARATOR, CALLS_START, INVOKE_END, INVOKE_START, PARAMETER_END,
    PARAMETER_START,
};
use crate::output::{Error, Result};
use crate::python_json;

/// The block's opening with the blank line the format writes before it.
const SEPARATED_START: &str = "\n\n<｜DSML｜ calls>";
/// The end of an `invoke` tag, after the function's name.
const NAME_END: &str = "\">";

/// The parser of the DSML tool-call format.
#[derive(Default)]
pub struct DsmlToolParser {
    state: State,
}

/// Where the parsed text stands.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Region {
    /// Before the block.
    #[default]
    Text,
    /// Inside the block, between calls.
    Block,
    /// Inside an `invoke` tag, reading the function's name.
    Name,
    /// Inside a published call, between its arguments.
    Call,
    /// Inside a call whose name is empty, which is not published.
    Unnamed,
    /// Inside the value of the argument `name` of the open call.
    Value {
        /// The argument's name.
        name: String,
        /// Whether the value is its text (`string="true"`), whose JSON string is open.
        string: bool,
    },
    /// After the block: the turn ended.
    Done,
}

/// The DSML parser's state.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    /// Input not yet consumed.
    pub buffer: String,
    /// Where the consumed text stands.
    pub region: Region,
    /// Calls published in the turn; the open call is the last.
    pub started: usize,
    /// Arguments the open call published or opened.
    pub arguments: usize,
}

impl DsmlToolParser {
    /// A parser before any input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advances over the buffered input as far as complete tags and values reach.
    fn advance(&mut self, output: &mut ToolParserOutput) {
        while self.step(output) {}
    }

    /// Takes one step in the current region; `false` when the buffer holds nothing the region
    /// can consume yet.
    fn step(&mut self, output: &mut ToolParserOutput) -> bool {
        let state = &mut self.state;
        match state.region.clone() {
            Region::Text => {
                if let Some(at) = state.buffer.find(CALLS_START) {
                    let before = &state.buffer[..at];
                    output.push_text(before.strip_suffix(CALLS_SEPARATOR).unwrap_or(before));
                    state.buffer.drain(..at + CALLS_START.len());
                    state.region = Region::Block;
                    return true;
                }
                // A trailing blank line or partial opening waits for the next chunk to decide.
                let held = partial_prefix_len(&state.buffer, SEPARATED_START)
                    .max(partial_prefix_len(&state.buffer, CALLS_START));
                let text = state.buffer.len() - held;
                output.push_text(&state.buffer[..text]);
                state.buffer.drain(..text);
                false
            }
            Region::Block => match first(&state.buffer, &[INVOKE_START, CALLS_END]) {
                Some((at, INVOKE_START)) => {
                    state.buffer.drain(..at + INVOKE_START.len());
                    state.region = Region::Name;
                    true
                }
                Some((at, _)) => {
                    state.buffer.drain(..at + CALLS_END.len());
                    state.region = Region::Done;
                    true
                }
                None => {
                    discard(&mut state.buffer, &[INVOKE_START, CALLS_END]);
                    false
                }
            },
            Region::Name => {
                let Some(at) = state.buffer.find(NAME_END) else {
                    return false;
                };
                let name: String = state.buffer[..at].to_owned();
                state.buffer.drain(..at + NAME_END.len());
                if name.is_empty() {
                    state.region = Region::Unnamed;
                } else {
                    output.push_call(ToolCallDelta {
                        tool_index: state.started,
                        name: Some(name),
                        arguments: String::new(),
                    });
                    state.started += 1;
                    state.arguments = 0;
                    state.region = Region::Call;
                }
                true
            }
            Region::Call | Region::Unnamed => {
                let published = state.region == Region::Call;
                let tags: &[&str] = if published {
                    &[PARAMETER_START, INVOKE_END, CALLS_END]
                } else {
                    &[INVOKE_END, CALLS_END]
                };
                match first(&state.buffer, tags) {
                    Some((at, PARAMETER_START)) => self.parameter(at, output),
                    Some((at, tag)) => {
                        if published {
                            close_call(state, output);
                        }
                        state.buffer.drain(..at + tag.len());
                        state.region = if tag == INVOKE_END {
                            Region::Block
                        } else {
                            Region::Done
                        };
                        true
                    }
                    None => {
                        discard(&mut state.buffer, tags);
                        false
                    }
                }
            }
            Region::Value { name, string } => self.value(&name, string, output),
            Region::Done => {
                state.buffer.clear();
                false
            }
        }
    }

    /// Reads the parameter tag at `at` of the buffer inside a published call: an argument opens
    /// at a tag that parses, whose text value's JSON string is published open.
    fn parameter(&mut self, at: usize, output: &mut ToolParserOutput) -> bool {
        let state = &mut self.state;
        match head(&state.buffer[at..]) {
            Head::Incomplete => {
                state.buffer.drain(..at);
                false
            }
            Head::Malformed => {
                state.buffer.drain(..at + PARAMETER_START.len());
                true
            }
            Head::Parameter { name, string, len } => {
                state.buffer.drain(..at + len);
                if string {
                    let key = argument_key(state.arguments, &name);
                    push_arguments(state, output, format!("{key}\""));
                    state.arguments += 1;
                }
                state.region = Region::Value { name, string };
                true
            }
        }
    }

    /// Reads the value of the argument `name`: a text value streams as its text arrives; any
    /// value is published once its end is read, which leaves the call's or the block's end to
    /// the call.
    fn value(&mut self, name: &str, string: bool, output: &mut ToolParserOutput) -> bool {
        let state = &mut self.state;
        let Some((end, consumed)) = value_end(&state.buffer) else {
            if string {
                let stable = stable_value_len(&state.buffer);
                let text = python_json::escaped(&state.buffer[..stable]);
                state.buffer.drain(..stable);
                if !text.is_empty() {
                    push_arguments(state, output, text);
                }
            }
            return false;
        };

        let raw = &state.buffer[..end];
        let fragment = if string {
            format!("{}\"", python_json::escaped(raw))
        } else {
            // A value that is no JSON is its text, as vLLM reads it.
            let value = serde_json::from_str::<Value>(raw).map_or_else(
                |_| python_json::string(raw),
                |value| python_json::dumps(&value),
            );
            let key = argument_key(state.arguments, name);
            state.arguments += 1;
            format!("{key}{value}")
        };
        state.buffer.drain(..consumed);
        push_arguments(state, output, fragment);
        state.region = Region::Call;
        true
    }
}

impl ToolParser for DsmlToolParser {
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.state.buffer.push_str(chunk);
        self.advance(output);
        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        let state = &mut self.state;
        let buffer = take(&mut state.buffer);
        match take(&mut state.region) {
            Region::Text => output.push_text(&buffer),
            // The text ended inside a call: a text value closes where it ended (its held-back
            // partial tag dropped), a JSON value that did not end is dropped.
            Region::Value { string: true, .. } => {
                push_arguments(state, &mut output, "\"".to_owned());
                close_call(state, &mut output);
            }
            Region::Value { string: false, .. } | Region::Call => close_call(state, &mut output),
            Region::Block | Region::Name | Region::Unnamed | Region::Done => {}
        }
        state.region = Region::Done;
        Ok(output)
    }

    fn state(&self) -> super::State {
        super::State::Dsml(self.state.clone())
    }

    fn resume(&mut self, state: super::State) -> Result<()> {
        let super::State::Dsml(state) = state else {
            return Err(Error::State {
                part: "DSML tool-call parser",
            });
        };
        let open = matches!(state.region, Region::Call | Region::Value { .. });
        if open && state.started == 0 {
            return Err(Error::State {
                part: "started call for its open DSML call",
            });
        }
        self.state = state;
        Ok(())
    }
}

/// Publishes `arguments` as the next increment of the open call's arguments.
fn push_arguments(state: &State, output: &mut ToolParserOutput, arguments: String) {
    output.push_call(ToolCallDelta {
        tool_index: state.started - 1,
        name: None,
        arguments,
    });
}

/// Closes the open call's arguments object; a call without arguments never opened it.
fn close_call(state: &State, output: &mut ToolParserOutput) {
    let closing = if state.arguments == 0 { "{}" } else { "}" };
    push_arguments(state, output, closing.to_owned());
}

/// The JSON text before the value of the argument `name`, the open call's argument `index`:
/// `{"NAME": ` for the first, `, "NAME": ` for the others.
fn argument_key(index: usize, name: &str) -> String {
    let separator = if index == 0 { "{" } else { ", " };
    format!("{separator}{}: ", python_json::string(name))
}

/// The earliest of `tags` in `buffer`, with its position.
fn first<'t>(buffer: &str, tags: &[&'t str]) -> Option<(usize, &'t str)> {
    tags.iter()
        .filter_map(|tag| buffer.find(tag).map(|at| (at, *tag)))
        .min_by_key(|(at, _)| *at)
}

/// Drops the text of `buffer` that ignored text cannot turn into one of `tags`: all but its
/// longest trailing partial tag.
fn discard(buffer: &mut String, tags: &[&str]) {
    let held = tags
        .iter()
        .map(|tag| partial_prefix_len(buffer, tag))
        .max()
        .unwrap_or(0);
    buffer.drain(..buffer.len() - held);
}

/// A parameter tag, as `<｜DSML｜ parameter\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>` reads
/// it.
enum Head {
    /// The text so far may still become a tag.
    Incomplete,
    /// No tag starts here.
    Malformed,
    /// A tag of `len` bytes: the argument's name and whether its value is its text.
    Parameter {
        name: String,
        string: bool,
        len: usize,
    },
}

/// The parameter tag `text` starts with, which starts with [`PARAMETER_START`].
fn head(text: &str) -> Head {
    let after_start = &text[PARAMETER_START.len()..];
    let rest = after_start.trim_start();
    if rest.is_empty() {
        return Head::Incomplete;
    }
    if rest.len() == after_start.len() {
        return Head::Malformed;
    }
    let Some(rest) = rest.strip_prefix("name=\"") else {
        return if "name=\"".starts_with(rest) {
            Head::Incomplete
        } else {
            Head::Malformed
        };
    };
    let Some(quote) = rest.find('"') else {
        return Head::Incomplete;
    };
    let name = &rest[..quote];
    if name.is_empty() {
        return Head::Malformed;
    }
    let after_name = &rest[quote + 1..];

    // An optional `string` attribute after whitespace, then `>` after optional whitespace.
    let mut string = false;
    let mut tail = after_name.trim_start();
    let spaced = tail.len() < after_name.len();
    if spaced && let Some(attribute) = tail.strip_prefix("string=\"") {
        let Some(quote) = attribute.find('"') else {
            return Head::Incomplete;
        };
        string = &attribute[..quote] == "true";
        tail = attribute[quote + 1..].trim_start();
    } else if spaced && !tail.is_empty() && "string=\"".starts_with(tail) {
        return Head::Incomplete;
    }
    match tail.chars().next() {
        None => Head::Incomplete,
        Some('>') => Head::Parameter {
            name: name.to_owned(),
            string,
            len: text.len() - tail.len() + 1,
        },
        Some(_) => Head::Malformed,
    }
}

/// Where the value at the buffer's start ends, if its end has arrived: the value's length and
/// the bytes its end consumes. A value ends at its closing tag (consumed), and at the next
/// parameter tag, its call's end or the block's end (left to the call).
fn value_end(buffer: &str) -> Option<(usize, usize)> {
    let next = next_parameter(buffer);
    let ends = [
        buffer
            .find(PARAMETER_END)
            .map(|at| (at, at + PARAMETER_END.len())),
        buffer.find(INVOKE_END).map(|at| (at, at)),
        buffer.find(CALLS_END).map(|at| (at, at)),
        match next {
            Lookahead::Parameter(at) => Some((at, at)),
            Lookahead::Pending(_) | Lookahead::None => None,
        },
    ];
    let end = ends.into_iter().flatten().min_by_key(|(at, _)| *at)?;
    // A tag still undecided before the end may yet end the value earlier.
    match next {
        Lookahead::Pending(at) if at < end.0 => None,
        _ => Some(end),
    }
}

/// The length of the text value's prefix that no later text can turn into its end: all but a
/// trailing partial tag, and all before an undecided parameter tag.
fn stable_value_len(buffer: &str) -> usize {
    let held = [PARAMETER_END, INVOKE_END, CALLS_END, PARAMETER_START]
        .iter()
        .map(|tag| partial_prefix_len(buffer, tag))
        .max()
        .unwrap_or(0);
    let stable = buffer.len() - held;
    match next_parameter(buffer) {
        Lookahead::Pending(at) => stable.min(at),
        Lookahead::Parameter(_) | Lookahead::None => stable,
    }
}

/// The first parameter tag of a value's text that ends the value, as vLLM's lookahead
/// `(?=<｜DSML｜ parameter\s+name=)` reads one.
enum Lookahead {
    /// A tag that ends the value, at its position.
    Parameter(usize),
    /// A tag whose text so far may still end the value, at its position.
    Pending(usize),
    /// None.
    None,
}

/// The first tag of `buffer` that ends the value before it, or may.
fn next_parameter(buffer: &str) -> Lookahead {
    for (at, _) in buffer.match_indices(PARAMETER_START) {
        let after_start = &buffer[at + PARAMETER_START.len()..];
        let rest = after_start.trim_start();
        if rest.is_empty() {
            return Lookahead::Pending(at);
        }
        if rest.len() == after_start.len() {
            continue;
        }
        if rest.starts_with("name=") {
            return Lookahead::Parameter(at);
        }
        if "name=".starts_with(rest) {
            return Lookahead::Pending(at);
        }
    }
    Lookahead::None
}
