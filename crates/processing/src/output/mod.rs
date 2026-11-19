//! The parsing of a chat response's decoded text into the events its response is assembled
//! from (`scheduling::processing::Event`), as vLLM's chat serving does with its reasoning and
//! tool parsers.
//!
//! A response's [`Parser`] holds an optional reasoning parser, which splits reasoning from
//! visible text by the delimiters of the family's reasoning format (`reasoning`), and an
//! optional tool stage, which extracts tool calls from the visible text in the family's
//! tool-call format (`tool`, over the format's parser in `tools`). Which parsers a response has
//! is its family's chat profile, the deployment's choice to answer reasoning apart, and the
//! tools the request lets the model call. Text a parser holds back as a possible partial
//! delimiter is released by a later push or by [`Parser::finish`]. A tool-call format whose calls
//! end the turn (`crate::ToolCallFormat::final_region`) also ends reasoning where its calls open,
//! and from there on the text is the tool stage's alone.
//!
//! A parser's whole state is a value ([`State`]), which a handoff carries to the process that
//! goes on writing the response, as opaque bytes.

pub mod reasoning;
mod tool;
pub mod tools;

use scheduling::processing::{Event, Tool};
use serde::{Deserialize, Serialize};

use self::reasoning::{DelimitedReasoningParser, ReasoningDelta};
use self::tool::ToolStage;
use crate::serving::{ReasoningFormat, ToolCallFormat};

/// A boxed source error.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Why a response's text cannot be parsed, or its parser built.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The reasoning or tool-call parser of the family's format cannot be built for the
    /// request.
    #[error("cannot build the {kind} parser {name:?}: {message}")]
    Parser {
        /// `reasoning` or `tool`.
        kind: &'static str,
        /// The format's name.
        name: String,
        /// Why it cannot be built.
        message: String,
    },
    /// The model's text is not a tool call of the format its family declares: it breaks the
    /// format's grammar, a value is not what its tool's schema states, or the text ended
    /// inside a call.
    #[error("the model wrote a malformed {format} tool call: {message}")]
    ToolCall {
        /// The tool-call format's name.
        format: &'static str,
        /// What the text breaks.
        message: String,
        /// The syntax error of the call's JSON, where the format's JSON broke.
        #[source]
        source: Option<BoxError>,
    },
    /// Tool-call updates arrive out of order: arguments with no open call, or for another call
    /// than the open one.
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStream {
        /// The violated order.
        message: String,
    },
    /// A saved state is not one of this parser's.
    #[error("the parser's state states no {part}")]
    State {
        /// The part it lacks or states otherwise.
        part: &'static str,
    },
}

impl Error {
    /// The error with its sources, as one line: what fails the response.
    pub fn report(&self) -> String {
        let mut report = self.to_string();
        let mut source = std::error::Error::source(self);
        while let Some(cause) = source {
            report.push_str(": ");
            report.push_str(&cause.to_string());
            source = cause.source();
        }
        report
    }
}

/// Result type of parsing.
pub type Result<T> = std::result::Result<T, Error>;

/// A reasoning format with the vocabulary ids of its delimiters, which decide from the prompt
/// whether generation starts inside reasoning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Delimiters {
    /// The format.
    pub format: ReasoningFormat,
    /// The id of the token that opens a section.
    pub open: u32,
    /// The id of the token that closes it.
    pub close: u32,
}

/// What a response's parsers are built from: its reasoning delimiters when reasoning is parsed,
/// and its tool-call format and callable tools when tool calls are.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Options {
    /// Whether reasoning is parsed.
    pub reasoning: bool,
    /// The callable tools, when tool calls are parsed.
    pub tools: Option<Vec<Tool>>,
}

/// The parsers of one response: its reasoning and tool-call parsers, either of which may be
/// absent. Without a reasoning parser all text is visible; without a tool stage visible text
/// passes unchanged.
pub struct Parser {
    options: Options,
    reasoning: Option<DelimitedReasoningParser>,
    tools: Option<ToolStage>,
}

/// A parser's state, the options it was built from included, which [`Parser::restore`] takes
/// back.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    /// What the parser was built from.
    pub options: Options,
    /// The reasoning parser's state, when it parses reasoning.
    pub reasoning: Option<reasoning::State>,
    /// The tool stage's state, when it parses tool calls.
    pub tools: Option<tool::State>,
}

impl Parser {
    /// The parsers `options` state for a response of a family whose reasoning is delimited by
    /// `delimiters` and whose tool calls are written in `format`, each when it has one; the
    /// reasoning parser starts in the region the prompt's tail decides (`prompt`, the prompt's
    /// ids from its last special token on).
    ///
    /// # Errors
    ///
    /// [`Error::Parser`] when reasoning is parsed and the vocabulary lacks a delimiter as one
    /// token, or a tool's parameter schema is no valid JSON Schema.
    pub fn new(
        options: Options,
        delimiters: Result<Option<Delimiters>>,
        format: Option<ToolCallFormat>,
        prompt: &[u32],
    ) -> Result<Self> {
        let mut parser = Self::build(options, delimiters, format)?;
        if let Some(reasoning) = parser.reasoning.as_mut() {
            reasoning.initialize(prompt);
        }
        Ok(parser)
    }

    /// The parsers `state` was saved from, where it left them.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and [`Error::State`] when the state does not state each of the
    /// parsers its options build, or states one of another format.
    pub fn restore(
        state: State,
        delimiters: Result<Option<Delimiters>>,
        format: Option<ToolCallFormat>,
    ) -> Result<Self> {
        let mut parser = Self::build(state.options, delimiters, format)?;
        if let Some(reasoning) = parser.reasoning.as_mut() {
            let saved = state.reasoning.ok_or(Error::State {
                part: "reasoning parser",
            })?;
            reasoning.resume(saved);
        }
        if let Some(tools) = parser.tools.as_mut() {
            let saved = state.tools.ok_or(Error::State { part: "tool stage" })?;
            tools.resume(saved)?;
        }
        Ok(parser)
    }

    /// The parsers `options` state, before any text.
    fn build(
        options: Options,
        delimiters: Result<Option<Delimiters>>,
        format: Option<ToolCallFormat>,
    ) -> Result<Self> {
        // A format whose calls end the turn ends reasoning at its final region's opening, whether
        // or not the request lets the model call tools.
        let opening = format.and_then(ToolCallFormat::final_region);
        let reasoning = match options.reasoning {
            false => None,
            true => {
                delimiters?.map(|delimiters| DelimitedReasoningParser::new(delimiters, opening))
            }
        };
        let tools = match (format, options.tools.as_deref()) {
            (Some(format), Some(tools)) => Some(ToolStage::new(tools::parser(format, tools)?)),
            _ => None,
        };
        Ok(Self {
            options,
            reasoning,
            tools,
        })
    }

    /// The parsers' state, for the process that goes on writing the response.
    pub fn save(&self) -> State {
        State {
            options: self.options.clone(),
            reasoning: self.reasoning.as_ref().map(DelimitedReasoningParser::state),
            tools: self.tools.as_ref().map(ToolStage::state),
        }
    }

    /// Parses one decoded text delta into events.
    ///
    /// One delta may span several delimiters, as when several tokens decode into one update;
    /// its reasoning text precedes its visible text.
    ///
    /// # Errors
    ///
    /// [`Error::ToolCall`] for a malformed tool call and [`Error::ToolCallStream`] when the
    /// tool parser's output is out of order.
    pub fn push(&mut self, delta: &str) -> Result<Vec<Event>> {
        let parsed = match self.reasoning.as_mut() {
            Some(reasoning) => reasoning.push(delta),
            None => ReasoningDelta {
                reasoning: None,
                content: Some(delta.to_owned()),
            },
        };
        self.split(parsed)
    }

    /// Releases the text held back at the end of generation: the reasoning parser's partial
    /// delimiter as text of its current region, then the tool parser's buffered input.
    ///
    /// # Errors
    ///
    /// As [`Self::push`], and [`Error::ToolCall`] when the text ended inside a call.
    pub fn finish(&mut self) -> Result<Vec<Event>> {
        let parsed = self
            .reasoning
            .as_mut()
            .map(DelimitedReasoningParser::finish)
            .unwrap_or_default();
        let mut events = self.split(parsed)?;
        if let Some(tools) = self.tools.as_mut() {
            events.extend(tools.finish()?);
        }
        Ok(events)
    }

    /// Emits the reasoning part of `parsed`, then its visible part through the tool stage.
    fn split(&mut self, parsed: ReasoningDelta) -> Result<Vec<Event>> {
        let mut events = Vec::new();
        if let Some(text) = parsed.reasoning.filter(|text| !text.is_empty()) {
            if let Some(tools) = self.tools.as_mut() {
                tools.interrupt();
            }
            events.push(Event::Reasoning(text));
        }
        if let Some(text) = parsed.content.filter(|text| !text.is_empty()) {
            match self.tools.as_mut() {
                Some(tools) => events.extend(tools.push(&text)?),
                None => events.push(Event::Content(text)),
            }
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests;
