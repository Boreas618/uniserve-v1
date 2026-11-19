//! The parser of the function-and-parameter tag tool-call format
//! (`crate::ToolCallFormat::Xml`).
//!
//! The format names the function and each argument in a tag and writes every value as text:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=ARG>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Its grammar, which the parser holds the text to:
//!
//! - Outside a region, text is plain text up to the region's start delimiter; a trailing partial
//!   delimiter is held back until the next chunk decides.
//! - Inside a region, whitespace separates the tags; a `<function=NAME>` tag opens a call and the
//!   end delimiter closes the region. A region may hold several calls.
//! - Inside a call, whitespace separates the tags; a `<parameter=ARG>` tag opens an argument
//!   whose value runs to its `</parameter>`, and `</function>` closes the call. One newline after
//!   the parameter tag and one before its end tag belong to the format and are not the value's.
//!
//! Any other text inside a region or a call, and text that ends inside a region, is a malformed
//! call. A call is published (its first delta names the function) at its function tag; each
//! argument is published once its value ended, as one fragment of the arguments object
//! (`{"ARG":VALUE`, then `,"ARG":VALUE`), and `</function>` closes the object.
//!
//! A value is typed by the JSON Schema its tool declares for the argument (the `properties` of
//! the tool's `parameters`): it is the text itself when the schema accepts that string, else the
//! JSON value the text states when the schema accepts that value; a value the schema accepts in
//! neither reading is a malformed call. An argument no tool declares a schema for is its text.

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::mem::take;

use jsonschema::Validator;
use scheduling::processing::Tool;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ToolCallDelta, ToolParser, ToolParserOutput, partial_prefix_len, text_before};
use crate::output::{BoxError, Error, Result};
use crate::serving::ToolCallFormat;

/// The format this module parses.
const FORMAT: ToolCallFormat = ToolCallFormat::Xml;

/// The tag that opens a call, before the function's name and its `>`.
const FUNCTION_PREFIX: &str = "<function=";
/// The tag that closes a call.
const FUNCTION_END: &str = "</function>";
/// The tag that opens an argument, before the argument's name and its `>`.
const PARAMETER_PREFIX: &str = "<parameter=";
/// The tag that closes an argument's value.
const PARAMETER_END: &str = "</parameter>";

/// A malformed call's failure.
fn malformed(message: impl Into<String>) -> Error {
    Error::ToolCall {
        format: FORMAT.name(),
        message: message.into(),
        source: None,
    }
}

/// The parser of the function-and-parameter tag tool-call format.
pub struct XmlToolParser {
    /// The schema of each argument of each tool, by function name, then argument name.
    schemas: HashMap<String, HashMap<String, Validator>>,
    /// Input no complete tag or argument consumed.
    buffer: String,
    /// Whether the text parsed so far is inside a tool-call region.
    in_region: bool,
    /// The function whose call is open, from its function tag to its `</function>`.
    function: Option<String>,
    /// Arguments the open call has published.
    parameters: usize,
    /// Calls started in the turn; the next call's index, and one past the open call's.
    started: usize,
}

impl XmlToolParser {
    /// A parser of calls to `tools`, before any input.
    ///
    /// # Errors
    ///
    /// Fails when an argument's schema is no valid JSON Schema.
    pub fn new(tools: &[Tool]) -> std::result::Result<Self, BoxError> {
        let mut schemas = HashMap::new();
        for Tool::Function { function } in tools {
            let Some(Value::Object(properties)) = function.parameters.get("properties") else {
                continue;
            };
            let mut arguments = HashMap::new();
            for (name, schema) in properties {
                arguments.insert(name.clone(), jsonschema::validator_for(schema)?);
            }
            schemas.insert(function.name.clone(), arguments);
        }

        Ok(Self {
            schemas,
            buffer: String::new(),
            in_region: false,
            function: None,
            parameters: 0,
            started: 0,
        })
    }

    /// Advances over the buffered input as far as complete tags and arguments reach.
    fn advance(&mut self, output: &mut ToolParserOutput) -> Result<()> {
        let (start, end) = FORMAT.delimiters();
        loop {
            if !self.in_region {
                let (text, opens) = text_before(&self.buffer, start);
                output.push_text(&self.buffer[..text]);
                let taken = if opens { text + start.len() } else { text };
                self.buffer.drain(..taken);
                if !opens {
                    return Ok(());
                }
                self.in_region = true;
                continue;
            }

            // Inside a region or a call, whitespace between tags carries no meaning.
            let blank = self.buffer.len() - self.buffer.trim_start().len();
            self.buffer.drain(..blank);
            if self.buffer.is_empty() {
                return Ok(());
            }

            let taken = match self.function.take() {
                None => self.region_tag(end, output)?,
                Some(function) => self.call_tag(function, output)?,
            };
            match taken {
                Some(taken) => {
                    self.buffer.drain(..taken);
                }
                None => return Ok(()),
            }
        }
    }

    /// Reads the tag the buffer starts with inside a region, outside a call, and returns its
    /// length; `None` while the tag is incomplete.
    fn region_tag(&mut self, end: &str, output: &mut ToolParserOutput) -> Result<Option<usize>> {
        if self.buffer.starts_with(end) {
            self.in_region = false;
            return Ok(Some(end.len()));
        }
        if self.buffer.starts_with(FUNCTION_PREFIX) {
            let Some((name, taken)) = tag_name(&self.buffer, FUNCTION_PREFIX)? else {
                return Ok(None);
            };
            output.push_call(ToolCallDelta {
                tool_index: self.started,
                name: Some(name.clone()),
                arguments: String::new(),
            });
            self.started += 1;
            self.function = Some(name);
            self.parameters = 0;
            return Ok(Some(taken));
        }
        if is_partial(&self.buffer, &[end, FUNCTION_PREFIX]) {
            return Ok(None);
        }
        Err(malformed(
            "a tool-call region holds function tags and its end delimiter alone",
        ))
    }

    /// Reads the tag the buffer starts with inside the call of `function`, and returns its
    /// length; `None` while the tag, or an argument's value, is incomplete.
    fn call_tag(
        &mut self,
        function: String,
        output: &mut ToolParserOutput,
    ) -> Result<Option<usize>> {
        let index = self.started - 1;
        if self.buffer.starts_with(FUNCTION_END) {
            // A call without arguments never opened its object.
            let closing = if self.parameters == 0 { "{}" } else { "}" };
            output.push_call(ToolCallDelta {
                tool_index: index,
                name: None,
                arguments: closing.to_owned(),
            });
            self.parameters = 0;
            return Ok(Some(FUNCTION_END.len()));
        }

        let taken = if self.buffer.starts_with(PARAMETER_PREFIX) {
            self.parameter(&function, index, output)?
        } else if is_partial(&self.buffer, &[FUNCTION_END, PARAMETER_PREFIX]) {
            None
        } else {
            return Err(malformed(
                "a call holds parameter tags and its `</function>` alone",
            ));
        };
        self.function = Some(function);
        Ok(taken)
    }

    /// Publishes the argument the buffer starts with as one fragment of the open call's
    /// arguments, and returns the bytes it took; `None` while its tag or its value is
    /// incomplete.
    fn parameter(
        &mut self,
        function: &str,
        index: usize,
        output: &mut ToolParserOutput,
    ) -> Result<Option<usize>> {
        let Some((name, tag)) = tag_name(&self.buffer, PARAMETER_PREFIX)? else {
            return Ok(None);
        };
        let rest = &self.buffer[tag..];
        let Some(end) = rest.find(PARAMETER_END) else {
            return Ok(None);
        };

        // One newline after the tag and one before the end tag are the format's.
        let raw = &rest[..end];
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let value = self.typed(function, &name, raw)?;

        let separator = if self.parameters == 0 { "{" } else { "," };
        output.push_call(ToolCallDelta {
            tool_index: index,
            name: None,
            arguments: format!("{separator}{}:{value}", Value::String(name)),
        });
        self.parameters += 1;
        Ok(Some(tag + end + PARAMETER_END.len()))
    }

    /// The JSON value of the argument `name` of `function` whose text is `raw`.
    fn typed(&self, function: &str, name: &str, raw: &str) -> Result<Value> {
        let text = Value::String(raw.to_owned());
        let Some(schema) = self
            .schemas
            .get(function)
            .and_then(|arguments| arguments.get(name))
        else {
            return Ok(text);
        };

        if schema.is_valid(&text) {
            return Ok(text);
        }
        match serde_json::from_str::<Value>(raw) {
            Ok(value) if schema.is_valid(&value) => Ok(value),
            _ => Err(malformed(format!(
                "the value of `{name}` of `{function}` is neither a string nor JSON its schema \
                 accepts: {raw:?}"
            ))),
        }
    }
}

impl ToolParser for XmlToolParser {
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);
        self.advance(output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        if self.in_region {
            return Err(malformed("the text ended inside a tool-call region"));
        }
        let mut output = ToolParserOutput::default();
        output.push_text(&take(&mut self.buffer));
        Ok(output)
    }

    fn state(&self) -> super::State {
        super::State::Xml(State {
            buffer: self.buffer.clone(),
            in_region: self.in_region,
            function: self.function.clone(),
            parameters: self.parameters,
            started: self.started,
        })
    }

    fn resume(&mut self, state: super::State) -> Result<()> {
        let super::State::Xml(state) = state else {
            return Err(Error::State {
                part: "tag tool-call parser",
            });
        };
        if state.function.is_some() && (!state.in_region || state.started == 0) {
            return Err(Error::State {
                part: "region and started call for its open tag tool call",
            });
        }
        self.buffer = state.buffer;
        self.in_region = state.in_region;
        self.function = state.function;
        self.parameters = state.parameters;
        self.started = state.started;
        Ok(())
    }
}

/// The function-and-parameter tag tool-call parser's state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    /// Input no complete tag or parameter consumed.
    pub buffer: String,
    /// Whether the text parsed so far is inside a tool-call region.
    pub in_region: bool,
    /// The function whose call is open, from its `<function=NAME>` tag until its
    /// `</function>`; the open call is the last one started.
    pub function: Option<String>,
    /// Parameters the open call has published.
    pub parameters: usize,
    /// Calls started in the turn; the next call's index.
    pub started: usize,
}

/// The name of the tag `prefix` the buffer starts with, and the tag's length; `None` while its
/// `>` has not arrived.
fn tag_name(buffer: &str, prefix: &str) -> Result<Option<(String, usize)>> {
    let Some(close) = buffer.find('>') else {
        return Ok(None);
    };
    let name = &buffer[prefix.len()..close];
    if name.is_empty() || name.contains(['<', '\n']) {
        return Err(malformed(format!("`{prefix}` names no name")));
    }
    Ok(Some((name.to_owned(), close + 1)))
}

/// Whether the whole buffer is a proper prefix of one of `tags`, which the next chunk may
/// complete.
fn is_partial(buffer: &str, tags: &[&str]) -> bool {
    tags.iter()
        .any(|tag| partial_prefix_len(buffer, tag) == buffer.len())
}
