//! The parser of the JSON tool-call format (`crate::ToolCallFormat::Json`): one JSON object
//! per call between the format's delimiters, naming the function and its arguments.
//!
//! ```text
//! <tool_call>
//! {"name": "get_weather", "arguments": {"location": "Tokyo"}}
//! </tool_call>
//! ```
//!
//! Outside a call the parser emits text up to the start delimiter and holds back a trailing
//! partial delimiter; a start marker without the rest of the delimiter, such as a `<tool_call>`
//! tag mentioned in prose, is text. Inside a call a strict push JSON parser ([`actson`]) reads
//! the call's object byte by byte as it arrives. The object states `name`, a string, then
//! `arguments`, an object, and nothing else; the call is published (its first delta names the
//! function) once its arguments object opens, and the arguments' text is streamed as the model
//! writes it, neither parsed into values nor reformatted. After the object closes the end
//! delimiter follows at once. Anything else inside a call is a malformed call, which fails the
//! parse.
//!
//! The arguments are already OpenAI-style JSON text, so the parser reads no tool definition, and
//! the function name is not checked against the request's tools. Parallel calls are repeated
//! calls, not several objects inside one call.

#[cfg(test)]
mod tests;

use actson::feeder::PushJsonFeeder;
use actson::{JsonEvent, JsonParser};
use serde::{Deserialize, Serialize};

use super::{ToolCallDelta, ToolParser, ToolParserOutput, text_before};
use crate::output::{Error, Result};
use crate::serving::ToolCallFormat;

/// The format this module parses.
const FORMAT: ToolCallFormat = ToolCallFormat::Json;

/// A malformed call's failure.
fn malformed(message: impl Into<String>) -> Error {
    Error::ToolCall {
        format: FORMAT.name(),
        message: message.into(),
        source: None,
    }
}

/// The parser of the JSON tool-call format.
pub struct JsonToolParser {
    /// Input after the text emitted and the call's consumed body: a partial start delimiter
    /// outside a call, or the end delimiter's prefix after a call's object closed.
    buffer: String,
    /// The call the text is inside, from its start delimiter to its end delimiter.
    call: Option<Call>,
    /// Calls started since the parser began; the next call's index.
    started: usize,
}

impl Default for JsonToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonToolParser {
    /// A parser before any input.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            call: None,
            started: 0,
        }
    }

    /// Advances over the buffered input as far as it completes the grammar.
    fn advance(&mut self, output: &mut ToolParserOutput) -> Result<()> {
        let (start, end) = FORMAT.delimiters();
        loop {
            match self.call.as_mut() {
                None => {
                    let (text, opens) = text_before(&self.buffer, start);
                    output.push_text(&self.buffer[..text]);
                    if !opens {
                        self.buffer.drain(..text);
                        return Ok(());
                    }
                    self.buffer.drain(..text + start.len());
                    self.call = Some(Call::new(self.started));
                    self.started += 1;
                }
                Some(call) if call.step == Step::End => {
                    if self.buffer.starts_with(end) {
                        self.buffer.drain(..end.len());
                        self.call = None;
                    } else if end.starts_with(self.buffer.as_str()) {
                        return Ok(());
                    } else {
                        return Err(malformed(
                            "the call's object is not followed by the end delimiter",
                        ));
                    }
                }
                Some(call) => match call.read(&self.buffer, output)? {
                    Some(taken) => {
                        self.buffer.drain(..taken);
                    }
                    None => {
                        self.buffer.clear();
                        return Ok(());
                    }
                },
            }
        }
    }
}

impl ToolParser for JsonToolParser {
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);
        self.advance(output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        if self.call.is_some() {
            return Err(malformed("the text ended inside a call"));
        }
        let mut output = ToolParserOutput::default();
        output.push_text(&std::mem::take(&mut self.buffer));
        Ok(output)
    }

    fn state(&self) -> super::State {
        super::State::Json(State {
            buffer: self.buffer.clone(),
            call: self.call.as_ref().map(|call| call.body.clone()),
            started: self.started,
        })
    }

    /// Reads the open call's body again into a new call, which stands where the call of the
    /// process that handed the state over stood; nothing it publishes again is emitted.
    fn resume(&mut self, state: super::State) -> Result<()> {
        let super::State::Json(state) = state else {
            return Err(Error::State {
                part: "JSON tool-call parser",
            });
        };
        self.started = state.started;
        self.call = match &state.call {
            Some(body) => {
                // The open call is the last one started.
                let index = self.started.checked_sub(1).ok_or(Error::State {
                    part: "started call for its open JSON tool call",
                })?;
                let mut call = Call::new(index);
                call.read(body, &mut ToolParserOutput::default())?;
                Some(call)
            }
            None => None,
        };
        self.buffer = state.buffer;
        Ok(())
    }
}

/// The JSON tool-call parser's state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    /// Input after the emitted text and the open call's body: a partial start delimiter outside a
    /// call, or a prefix of the end delimiter after a call's object closed.
    pub buffer: String,
    /// The open call's text after its start delimiter, which a successor's parser reads again to
    /// stand where this one stood; the open call is the last one started.
    pub call: Option<String>,
    /// Calls started in the turn; the next call's index.
    pub started: usize,
}

/// Where a call's object is in the format's grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Before the object opens.
    Object,
    /// Before the `name` key.
    NameKey,
    /// Before the function name.
    Name,
    /// Before the `arguments` key, the name read.
    ArgumentsKey,
    /// Before the arguments object opens.
    Arguments,
    /// Inside the arguments object, published; the body's bytes before `emitted` are emitted.
    Streaming {
        /// Bytes of the body emitted as arguments, or before them.
        emitted: usize,
    },
    /// After the arguments object, before the call's object closes.
    Close,
    /// After the call's object, before the end delimiter.
    End,
}

/// One call's object as far as it was read.
struct Call {
    /// The call's index among the turn's calls.
    index: usize,
    /// The push parser over the call's body.
    json: JsonParser<PushJsonFeeder>,
    /// The call's text after its start delimiter, as far as it was read: what a successor reads
    /// again to stand where this call stands.
    body: String,
    /// The function name, once read.
    name: String,
    /// Nesting depth of the object at the parser's position.
    depth: usize,
    step: Step,
}

impl Call {
    /// The call of index `index`, right after its start delimiter.
    fn new(index: usize) -> Self {
        Self {
            index,
            json: JsonParser::new(PushJsonFeeder::new()),
            body: String::new(),
            name: String::new(),
            depth: 0,
            step: Step::Object,
        }
    }

    /// Reads `input` as the continuation of the call's object, emitting its publication and
    /// its arguments' text, and returns the bytes of `input` the object took when it closed in
    /// it; `None` when every byte of `input` belongs to the still open object.
    fn read(&mut self, input: &str, output: &mut ToolParserOutput) -> Result<Option<usize>> {
        let base = self.body.len();
        self.body.push_str(input);
        let bytes = input.as_bytes();

        let mut pushed = 0;
        loop {
            pushed += self.json.feeder.push_bytes(&bytes[pushed..]);
            loop {
                let event = self.json.next_event().map_err(|error| Error::ToolCall {
                    format: FORMAT.name(),
                    message: "the call's object is not JSON".to_owned(),
                    source: Some(Box::new(error)),
                })?;
                match event {
                    Some(JsonEvent::NeedMoreInput) => break,
                    Some(event) => {
                        // The bytes of the body the parser has read, through this event.
                        let offset = self.json.parsed_bytes();
                        self.apply(event, offset, output)?;
                        if self.step == Step::End {
                            self.body.truncate(offset);
                            return Ok(Some(offset - base));
                        }
                    }
                    // The object's end is seen before its text ends, so the parser never
                    // reaches the end of its input.
                    None => return Err(malformed("the call's object ended twice")),
                }
            }
            if pushed == bytes.len() {
                break;
            }
        }

        // Every byte read lies inside the open object: inside the arguments, it is
        // arguments text.
        if let Step::Streaming { emitted } = &mut self.step {
            output.push_call(ToolCallDelta {
                tool_index: self.index,
                name: None,
                arguments: self.body[*emitted..].to_owned(),
            });
            *emitted = self.body.len();
        }
        Ok(None)
    }

    /// Advances the grammar over one event of the call's object, which ends at byte `offset`
    /// of the body.
    fn apply(
        &mut self,
        event: JsonEvent,
        offset: usize,
        output: &mut ToolParserOutput,
    ) -> Result<()> {
        let text = |json: &JsonParser<PushJsonFeeder>| {
            json.current_str()
                .map(str::to_owned)
                .map_err(|error| Error::ToolCall {
                    format: FORMAT.name(),
                    message: "a string of the call's object is not UTF-8".to_owned(),
                    source: Some(Box::new(error)),
                })
        };

        self.step = match (self.step, event) {
            (Step::Object, JsonEvent::StartObject) => {
                self.depth = 1;
                Step::NameKey
            }
            (Step::Object, _) => return Err(malformed("a call is a JSON object")),
            (Step::NameKey, JsonEvent::FieldName) if text(&self.json)? == "name" => Step::Name,
            (Step::NameKey, _) => return Err(malformed("the call's object states `name` first")),
            (Step::Name, JsonEvent::ValueString) => {
                self.name = text(&self.json)?;
                Step::ArgumentsKey
            }
            (Step::Name, _) => return Err(malformed("the function's name is a string")),
            (Step::ArgumentsKey, JsonEvent::FieldName) if text(&self.json)? == "arguments" => {
                Step::Arguments
            }
            (Step::ArgumentsKey, _) => {
                return Err(malformed(
                    "the call's object states `arguments` after `name`",
                ));
            }
            (Step::Arguments, JsonEvent::StartObject) => {
                self.depth = 2;
                output.push_call(ToolCallDelta {
                    tool_index: self.index,
                    name: Some(std::mem::take(&mut self.name)),
                    arguments: String::new(),
                });
                // The arguments' text starts at the brace just read.
                Step::Streaming {
                    emitted: offset - 1,
                }
            }
            (Step::Arguments, _) => return Err(malformed("the call's arguments are an object")),
            (Step::Streaming { emitted }, JsonEvent::StartObject | JsonEvent::StartArray) => {
                self.depth += 1;
                Step::Streaming { emitted }
            }
            (Step::Streaming { emitted }, JsonEvent::EndObject | JsonEvent::EndArray) => {
                self.depth -= 1;
                if self.depth > 1 {
                    Step::Streaming { emitted }
                } else {
                    output.push_call(ToolCallDelta {
                        tool_index: self.index,
                        name: None,
                        arguments: self.body[emitted..offset].to_owned(),
                    });
                    Step::Close
                }
            }
            (step @ Step::Streaming { .. }, _) => step,
            (Step::Close, JsonEvent::EndObject) => {
                self.depth = 0;
                Step::End
            }
            (Step::Close, _) => {
                return Err(malformed(
                    "the call's object states `name` and `arguments` alone",
                ));
            }
            (Step::End, _) => return Err(malformed("the call's object ended twice")),
        };
        Ok(())
    }
}
