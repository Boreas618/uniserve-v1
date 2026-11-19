//! The tool-call stage of a response's [`Parser`](super::Parser).
//!
//! Visible text passes through the request's `ToolParser`; reasoning passes the stage by
//! ([`ToolStage::interrupt`]). A malformed call fails the stage with the parser's error, which
//! fails the response, so the response reports the model's malformed call instead of passing it
//! on as text. The daemon names each call as it opens.

use scheduling::processing::Event;
use serde::{Deserialize, Serialize};

use super::tools::{self, ToolCallDelta, ToolParser, ToolParserItem, ToolParserOutput};
use super::{Error, Result};

/// Tool-stage state for one response.
pub(super) struct ToolStage {
    parser: Box<dyn ToolParser>,
    /// Parser-local index of the tool call that later argument deltas extend. Cleared when
    /// visible text or reasoning follows the call.
    open_call_index: Option<usize>,
}

/// The stage's state, with its parser's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    /// The parser-local index of the tool call later arguments extend.
    pub open_call_index: Option<usize>,
    /// The parser's state.
    pub parser: tools::State,
}

impl ToolStage {
    /// The stage around the request's tool parser.
    pub(super) fn new(parser: Box<dyn ToolParser>) -> Self {
        Self {
            parser,
            open_call_index: None,
        }
    }

    /// The stage's state, for the process that goes on parsing where this one stopped.
    pub(super) fn state(&self) -> State {
        State {
            open_call_index: self.open_call_index,
            parser: self.parser.state(),
        }
    }

    /// Takes over `state`, which a stage of the same parser produced.
    ///
    /// # Errors
    ///
    /// [`Error::State`] when `state` is another format's.
    pub(super) fn resume(&mut self, state: State) -> Result<()> {
        self.parser.resume(state.parser)?;
        self.open_call_index = state.open_call_index;
        Ok(())
    }

    /// Reasoning passes the stage: it closes the open call.
    pub(super) fn interrupt(&mut self) {
        self.open_call_index = None;
    }

    /// Feeds visible text through the tool parser.
    ///
    /// # Errors
    ///
    /// [`Error::ToolCall`] for a malformed call and [`Error::ToolCallStream`] when the parser's
    /// deltas are out of order.
    pub(super) fn push(&mut self, delta: &str) -> Result<Vec<Event>> {
        let mut output = ToolParserOutput::default();
        self.parser.parse_into(delta, &mut output)?;
        let mut events = Vec::new();
        self.process_parser_output(output, &mut events)?;
        Ok(events)
    }

    /// Emits one parser output's items in their input order.
    ///
    /// The parser resumes text only after a call's end, so a text item closes the open call.
    fn process_parser_output(
        &mut self,
        output: ToolParserOutput,
        events: &mut Vec<Event>,
    ) -> Result<()> {
        for item in output.items {
            match item {
                ToolParserItem::Text(text) => {
                    self.open_call_index = None;
                    if !text.is_empty() {
                        events.push(Event::Content(text));
                    }
                }
                ToolParserItem::Call(call) => self.process_tool_item(call, events)?,
            }
        }
        Ok(())
    }

    /// Converts one parser delta into a call's opening and its argument events.
    fn process_tool_item(&mut self, item: ToolCallDelta, events: &mut Vec<Event>) -> Result<()> {
        if let Some(name) = item.name
            && self.open_call_index != Some(item.tool_index)
        {
            self.open_call_index = Some(item.tool_index);
            events.push(Event::ToolCall { name });
        }

        if item.arguments.is_empty() {
            return Ok(());
        }
        match self.open_call_index {
            Some(open) if open == item.tool_index => {
                events.push(Event::Arguments(item.arguments));
                Ok(())
            }
            Some(open) => Err(Error::ToolCallStream {
                message: format!(
                    "received arguments for tool index {} while tool index {open} is open",
                    item.tool_index
                ),
            }),
            None => Err(Error::ToolCallStream {
                message: format!(
                    "received arguments for tool index {} before any tool-call start",
                    item.tool_index
                ),
            }),
        }
    }

    /// Flushes the parser at the end of generation: held-back text is visible text.
    ///
    /// # Errors
    ///
    /// [`Error::ToolCall`] when the text ended inside a call.
    pub(super) fn finish(&mut self) -> Result<Vec<Event>> {
        let output = self.parser.finish()?;
        let mut events = Vec::new();
        self.process_parser_output(output, &mut events)?;
        Ok(events)
    }
}
