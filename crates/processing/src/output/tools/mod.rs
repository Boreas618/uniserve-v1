//! Streaming tool-call parsers, one per tool-call format.
//!
//! The output's tool stage (`output::tool`) feeds visible assistant text chunk by chunk to the
//! [`ToolParser`] of the format the request's model family declares
//! ([`ToolCallFormat`]), which [`parser`] builds: [`JsonToolParser`] for one JSON
//! object per call, [`XmlToolParser`] for function and parameter tags. A parser is named for its
//! format, never for a model, and reads the format's region delimiters from its declaration
//! (`ToolCallFormat::delimiters`). It turns each chunk into an ordered sequence of plain text and
//! [`ToolCallDelta`] updates, holding back the bytes it cannot classify yet, such as a partial
//! delimiter or a parameter whose value has not ended.
//!
//! A parser is strict: text inside a call that breaks the format's grammar, and a call the text
//! leaves open, fail with [`Error::ToolCall`], which fails the request's response. A parser
//! repairs nothing and guesses nothing. What it holds is bounded by the text the model wrote,
//! which the request's token budget bounds.

mod json;
mod xml;

use scheduling::processing::Tool;
use serde::{Deserialize, Serialize};

pub use json::JsonToolParser;
pub use xml::XmlToolParser;

use super::{Error, Result};
use crate::serving::ToolCallFormat;

/// A parser's state, by its format.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// The JSON format's parser.
    Json(json::State),
    /// The tag format's parser.
    Xml(xml::State),
}

/// Incremental extraction of tool calls from one assistant turn's visible text.
///
/// Call [`Self::parse_into`] for each text chunk and [`Self::finish`] once at the end of
/// generation.
pub trait ToolParser {
    /// Parses one text chunk into an existing output accumulator.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::ToolCall`] when the text breaks the format's grammar. The events
    /// parsed before the failure remain in `output`.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()>;

    /// Releases what the parser holds at the end of generation: a held-back partial delimiter
    /// is text.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::ToolCall`] when the text ended inside a call.
    fn finish(&mut self) -> Result<ToolParserOutput>;

    /// The parser's state, for the process that goes on writing the response after a handoff.
    fn state(&self) -> State;

    /// Takes over a state a parser of the same format and tools produced ([`Self::state`]).
    ///
    /// # Errors
    ///
    /// Fails with [`Error::State`] when `state` is another format's.
    fn resume(&mut self, state: State) -> Result<()>;
}

/// The parser of `format` for a request whose callable tools are `tools`.
///
/// # Errors
///
/// Fails with [`Error::Parser`] when a tool's parameter schema is no valid JSON Schema.
pub fn parser(format: ToolCallFormat, tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
    Ok(match format {
        ToolCallFormat::Json => Box::new(JsonToolParser::new()),
        ToolCallFormat::Xml => {
            Box::new(XmlToolParser::new(tools).map_err(|error| Error::Parser {
                kind: "tool",
                name: format.name().to_owned(),
                message: error.to_string(),
            })?)
        }
    })
}

/// One tool-call update emitted while parsing assistant text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Parser-local index of the call within one assistant turn, from 0.
    pub tool_index: usize,
    /// Function name, on the first update of a call.
    pub name: Option<String>,
    /// Arguments text this update contributes. The updates of one call concatenate to the
    /// call's arguments as one JSON object.
    pub arguments: String,
}

/// One piece of parser output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParserItem {
    /// Plain assistant text that is part of no tool call.
    Text(String),
    /// One tool-call update.
    Call(ToolCallDelta),
}

/// What advancing a tool parser over one or more chunks produced.
///
/// Items follow the order of the input they were parsed from, so text written before a call
/// precedes the call's deltas and text written after it follows them. Adjacent text is merged
/// into one nonempty item. A call's deltas are contiguous, because text resumes only after the
/// call's end.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolParserOutput {
    /// Text runs and tool-call updates in input order.
    pub items: Vec<ToolParserItem>,
}

impl ToolParserOutput {
    /// Appends plain text, merging it into a trailing text item; empty text is ignored.
    fn push_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        match self.items.last_mut() {
            Some(ToolParserItem::Text(last)) => last.push_str(text),
            _ => self.items.push(ToolParserItem::Text(text.to_owned())),
        }
    }

    /// Appends one tool-call update.
    fn push_call(&mut self, call: ToolCallDelta) {
        self.items.push(ToolParserItem::Call(call));
    }

    /// Appends another output, keeping item order and merging text across the boundary.
    pub fn append(&mut self, other: Self) {
        for item in other.items {
            match item {
                ToolParserItem::Text(text) => self.push_text(&text),
                ToolParserItem::Call(call) => self.push_call(call),
            }
        }
    }

    /// The concatenation of every text item.
    pub fn normal_text(&self) -> String {
        self.items
            .iter()
            .filter_map(|item| match item {
                ToolParserItem::Text(text) => Some(text.as_str()),
                ToolParserItem::Call(_) => None,
            })
            .collect()
    }

    /// The tool-call updates in order.
    pub fn calls(&self) -> impl Iterator<Item = &ToolCallDelta> {
        self.items.iter().filter_map(|item| match item {
            ToolParserItem::Call(call) => Some(call),
            ToolParserItem::Text(_) => None,
        })
    }

    /// Merges consecutive deltas of one call into one item: the first name any of them carries
    /// and the concatenation of their arguments.
    pub fn coalesce_calls(self) -> Self {
        let mut coalesced = Self::default();

        for item in self.items {
            match item {
                ToolParserItem::Text(text) => coalesced.push_text(&text),
                ToolParserItem::Call(call) => match coalesced.items.last_mut() {
                    Some(ToolParserItem::Call(existing))
                        if existing.tool_index == call.tool_index =>
                    {
                        if existing.name.is_none() {
                            existing.name = call.name;
                        }
                        existing.arguments.push_str(&call.arguments);
                    }
                    _ => coalesced.push_call(call),
                },
            }
        }

        coalesced
    }
}

/// The byte length of the longest proper prefix of `delimiter` that `text` ends with: the text a
/// parser holds back because the next chunk may complete the delimiter.
///
/// The length is a character boundary of both strings, so the caller may slice either.
fn partial_prefix_len(text: &str, delimiter: &str) -> usize {
    let longest = text.len().min(delimiter.len().saturating_sub(1));
    (1..=longest)
        .rev()
        .find(|&len| {
            let start = text.len() - len;
            text.is_char_boundary(start)
                && delimiter.is_char_boundary(len)
                && text.as_bytes()[start..] == delimiter.as_bytes()[..len]
        })
        .unwrap_or(0)
}

/// The length of the text a parser outside a call emits from `buffer`: everything before the
/// first `delimiter`, or, without one, everything but a trailing partial delimiter. The second
/// value is whether the delimiter follows that text.
fn text_before(buffer: &str, delimiter: &str) -> (usize, bool) {
    match buffer.find(delimiter) {
        Some(start) => (start, true),
        None => (buffer.len() - partial_prefix_len(buffer, delimiter), false),
    }
}
