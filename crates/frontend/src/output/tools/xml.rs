//! Incremental parser for tool calls written as function and parameter tags.
//!
//! The format (`processing::ToolCallFormat::Xml`) names the function and each
//! argument in a tag and writes every value as text:
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
//! [`XmlToolParser`] follows SGLang's `Qwen3CoderDetector`, a cursor over the
//! buffered stream:
//!
//! - `<tool_call>` and `</tool_call>` open and close the region in which text
//!   between tags, such as the newlines above, is dropped. Text outside it is
//!   plain text, before, between and after calls.
//! - `<function=NAME>` publishes a call: its first delta carries the name. A
//!   response may hold several calls, in one region or in several.
//! - `<parameter=ARG>` waits for the end of its value, which is its own end
//!   tag or, where the model left that out, the next parameter or the
//!   function's end. One newline after the tag and one before the end belong
//!   to the format and are stripped. The value is then typed by the schema the
//!   call's tool declares for the argument ([`typed_value`]) and published as
//!   one fragment of the arguments object, `"ARG": VALUE` in JSON.
//! - `</function>` closes the arguments object.
//!
//! A tag may arrive split across fragments: a trailing `<` that may still
//! become a tag is held back until the next fragment decides. A `<` that
//! opens no tag is text.
//!
//! Arguments are written as Python's `json.dumps` writes them (`", "` between
//! items, `": "` after a key), as the reference does, so the concatenated
//! deltas of a call are one JSON object.

use std::mem::take;

use proto::handoff;
use serde_json::{Map, Number, Value};
use serde_json_fmt::JsonFormat;

use super::utils::MAX_BUFFER_BYTES;
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::protocol::Tool;

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION_PREFIX: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_PREFIX: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

/// Every tag of the format: a buffered `<...` that is a prefix of one of them
/// may still become it.
const TAGS: [&str; 6] = [
    TOOL_CALL_START,
    TOOL_CALL_END,
    FUNCTION_PREFIX,
    FUNCTION_END,
    PARAMETER_PREFIX,
    PARAMETER_END,
];

/// The parser of the function-and-parameter tag tool-call format.
#[derive(Debug)]
pub struct XmlToolParser {
    /// The request's tools, whose schemas type the values of their calls.
    tools: Vec<Tool>,
    /// Input no complete tag or parameter consumed.
    buffer: String,
    /// Whether the text parsed so far is inside `<tool_call>`.
    in_tool_call: bool,
    /// The function whose call is open, from its `<function=NAME>` tag until
    /// its `</function>`.
    function: Option<String>,
    /// Parameters the open call has published, which decides whether the next
    /// fragment opens the arguments object or continues it.
    parameters: usize,
    /// Tool index that argument deltas extend; set by a function tag, cleared
    /// by its end.
    active_tool_index: Option<usize>,
    /// Number of tool calls started since the last reset; also the next
    /// call's `tool_index`.
    emitted_tool_count: usize,
}

impl XmlToolParser {
    /// A parser of calls to `tools`, before any input.
    ///
    /// A call to a function that is not among `tools`, and an argument its
    /// tool's schema does not declare, keep their values as strings.
    pub fn new(tools: Vec<Tool>) -> Self {
        Self {
            tools,
            buffer: String::new(),
            in_tool_call: false,
            function: None,
            parameters: 0,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Whether text at the cursor is dropped: inside a tool-call region or an
    /// open call, only tags carry meaning.
    fn drops_text(&self) -> bool {
        self.in_tool_call || self.function.is_some()
    }

    /// Publishes the call of `name`, closing a call the model left open.
    fn open_call(&mut self, name: &str, output: &mut ToolParserOutput) {
        self.close_call(output);

        let tool_index = self.emitted_tool_count;
        self.emitted_tool_count += 1;
        self.active_tool_index = Some(tool_index);
        self.function = Some(name.to_owned());
        self.parameters = 0;
        output.push_call(ToolCallDelta {
            tool_index,
            name: Some(name.to_owned()),
            arguments: String::new(),
        });
    }

    /// Closes the open call's arguments object; nothing when no call is open.
    fn close_call(&mut self, output: &mut ToolParserOutput) {
        let Some(tool_index) = self.active_tool_index.take() else {
            return;
        };

        // A call without parameters never opened its object.
        let closing = if self.parameters == 0 { "{}" } else { "}" };
        output.push_call(ToolCallDelta {
            tool_index,
            name: None,
            arguments: closing.to_owned(),
        });
        self.function = None;
        self.parameters = 0;
    }

    /// Publishes the parameter `rest` starts with as one fragment of the open
    /// call's arguments, and returns the bytes of `rest` it took.
    ///
    /// Returns `None` while the parameter's tag or the end of its value has
    /// not arrived. With `finishing`, a value without an end runs to the end
    /// of `rest`.
    fn parameter(
        &mut self,
        rest: &str,
        finishing: bool,
        output: &mut ToolParserOutput,
    ) -> Option<usize> {
        let (tool_index, function) = self.active_tool_index.zip(self.function.as_deref())?;
        let name_end = rest.find('>')?;
        let name = &rest[PARAMETER_PREFIX.len()..name_end];
        let value = &rest[name_end + 1..];

        // A value ends at its own end tag or, where the model left that out, at
        // the next parameter or the function's end: the earliest of them.
        let end = [
            (PARAMETER_END, PARAMETER_END.len()),
            (PARAMETER_PREFIX, 0),
            (FUNCTION_END, 0),
        ]
        .into_iter()
        .filter_map(|(tag, skipped)| value.find(tag).map(|at| (at, skipped)))
        .min_by_key(|(at, _)| *at);
        let (end, skipped) = match end {
            Some(end) => end,
            None if finishing => (value.len(), 0),
            None => return None,
        };

        // One newline after the tag and one before the end are the format's.
        let raw = &value[..end];
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);

        let schemas = argument_schemas(&self.tools, function);
        let schema = schemas.as_ref().and_then(|schemas| schemas.get(name));
        let typed = typed_value(raw, schema);
        let separator = if self.parameters == 0 { "{" } else { ", " };
        output.push_call(ToolCallDelta {
            tool_index,
            name: None,
            arguments: format!("{separator}{}: {}", json_key(name), json_value(&typed)),
        });
        self.parameters += 1;

        Some(name_end + 1 + end + skipped)
    }

    /// Advances the cursor over `buffer` as far as complete tags and
    /// parameters reach, and returns the bytes it took.
    fn advance(&mut self, buffer: &str, output: &mut ToolParserOutput) -> usize {
        let mut position = 0;
        loop {
            let rest = &buffer[position..];
            if rest.is_empty() {
                break;
            }

            if rest.starts_with(TOOL_CALL_START) {
                self.in_tool_call = true;
                position += TOOL_CALL_START.len();
            } else if rest.starts_with(FUNCTION_PREFIX) {
                // The tag is whole once its `>` arrived.
                let Some(end) = rest.find('>') else {
                    break;
                };
                self.open_call(&rest[FUNCTION_PREFIX.len()..end], output);
                position += end + 1;
            } else if rest.starts_with(PARAMETER_PREFIX) && self.function.is_some() {
                let Some(taken) = self.parameter(rest, false, output) else {
                    break;
                };
                position += taken;
            } else if rest.starts_with(FUNCTION_END) {
                self.close_call(output);
                position += FUNCTION_END.len();
            } else if rest.starts_with(TOOL_CALL_END) {
                self.in_tool_call = false;
                position += TOOL_CALL_END.len();
            } else {
                // Text up to the next `<`, or a `<` that opens no tag. A `<` whose
                // buffered text may still become a tag waits for the next fragment.
                let taken = match rest.find('<') {
                    None => rest.len(),
                    Some(0) if TAGS.iter().any(|tag| tag.starts_with(rest)) => break,
                    Some(0) => 1,
                    Some(next) => next,
                };
                if !self.drops_text() {
                    output.push_text(&rest[..taken]);
                }
                position += taken;
            }
        }
        position
    }

    /// Returns the parser to its state before any input.
    fn clear(&mut self) {
        self.buffer.clear();
        self.in_tool_call = false;
        self.function = None;
        self.parameters = 0;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
    }
}

impl ToolParser for XmlToolParser {
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        let mut buffer = take(&mut self.buffer);
        buffer.push_str(chunk);
        let taken = self.advance(&buffer, output);
        buffer.drain(..taken);
        self.buffer = buffer;

        // A parameter's value is buffered whole, and the generation's length
        // bounds it. A tag that never closes, such as an endless function
        // name, would otherwise grow without that bound mattering to anyone.
        let awaits_value = self.buffer.starts_with(PARAMETER_PREFIX) && self.buffer.contains('>');
        if self.buffer.len() > MAX_BUFFER_BYTES && !awaits_value {
            return Err(parsing_failed!(
                "XML tool-call buffer exceeded {} bytes without completing a tag",
                MAX_BUFFER_BYTES
            ));
        }
        Ok(())
    }

    /// Flushes buffered parser state at end of generation, and resets the
    /// parser.
    ///
    /// A call the model left open, as at a length limit, keeps what it stated,
    /// as the reference's whole-text parse does: a parameter without an end
    /// takes the rest of the text as its value, and the arguments object is
    /// closed. Outside a call, held-back text that became no tag is plain
    /// text. This never fails.
    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        let buffer = take(&mut self.buffer);

        if self.function.is_some() {
            if buffer.starts_with(PARAMETER_PREFIX) {
                self.parameter(&buffer, true, &mut output);
            }
            self.close_call(&mut output);
        } else if !self.in_tool_call {
            output.push_text(&buffer);
        }

        self.clear();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        let unpublished = take(&mut self.buffer);
        self.clear();
        unpublished
    }

    fn state(&self) -> Option<handoff::tool_stage::Parser> {
        Some(handoff::tool_stage::Parser::Xml(handoff::XmlToolCall {
            buffer: self.buffer.clone(),
            in_tool_call: self.in_tool_call,
            function: self.function.clone(),
            parameters: self.parameters as u64,
            active_tool_index: self.active_tool_index.map(|index| index as u64),
            emitted_tool_count: self.emitted_tool_count as u64,
        }))
    }

    fn resume(&mut self, state: &handoff::tool_stage::Parser) {
        if let handoff::tool_stage::Parser::Xml(xml) = state {
            self.buffer.clone_from(&xml.buffer);
            self.in_tool_call = xml.in_tool_call;
            self.function.clone_from(&xml.function);
            self.parameters = xml.parameters as usize;
            self.active_tool_index = xml.active_tool_index.map(|index| index as usize);
            self.emitted_tool_count = xml.emitted_tool_count as usize;
        }
    }
}

/// A parameter name as a JSON object key, written as `json.dumps` writes it.
fn json_key(name: &str) -> String {
    JsonFormat::new()
        .ascii(true)
        .format_to_string(name)
        .unwrap_or_else(|_| Value::from(name).to_string())
}

/// A typed value as JSON text, written as `json.dumps` writes it: `", "`
/// between items, `": "` after a key, non-ASCII characters as they are.
fn json_value(value: &Value) -> String {
    JsonFormat::new()
        .comma(", ")
        .and_then(|format| format.colon(": "))
        .ok()
        .and_then(|format| format.format_to_string(value).ok())
        .unwrap_or_else(|| value.to_string())
}

/// The schemas the tool `function` declares for its arguments, by argument
/// name; `None` when no tool of that name declares a schema object.
///
/// They are the `properties` of the tool's `parameters`; a schema that states
/// its properties only inside `anyOf`, `oneOf` or `allOf` branches gives those
/// of its branches, the first to name an argument winning; and a `parameters`
/// object without any `properties` is itself read as the map, as the
/// reference reads it.
fn argument_schemas(tools: &[Tool], function: &str) -> Option<Map<String, Value>> {
    let Tool::Function { function: tool } = tools
        .iter()
        .find(|Tool::Function { function: tool }| tool.name == function)?;
    let parameters = tool.parameters.as_object()?;

    let properties = schema_properties(&tool.parameters);
    if !properties.is_empty() || parameters.contains_key("properties") {
        Some(properties)
    } else {
        Some(parameters.clone())
    }
}

/// The top-level `properties` of a schema, descending into `anyOf`, `oneOf`
/// and `allOf` branches when the top level declares none.
fn schema_properties(schema: &Value) -> Map<String, Value> {
    let Some(schema) = schema.as_object() else {
        return Map::new();
    };
    if let Some(Value::Object(properties)) = schema.get("properties") {
        return properties.clone();
    }

    let mut merged = Map::new();
    for keyword in ["anyOf", "oneOf", "allOf"] {
        let Some(Value::Array(branches)) = schema.get(keyword) else {
            continue;
        };
        for branch in branches {
            for (name, property) in schema_properties(branch) {
                merged.entry(name).or_insert(property);
            }
        }
    }
    merged
}

/// The primary type a JSON schema states for a value, as the reference's
/// `infer_type_from_json_schema` infers it; `None` when the schema states
/// none.
///
/// In order: a `type` (of a list of types, the first that is not `null`); the
/// branches of `anyOf` or `oneOf` (their common type, the other type of a
/// nullable pair, `string` where they differ and one is a string, else the
/// first); the values of an `enum` (their common kind, else `string`); the
/// first non-string branch of `allOf`; `object` for a schema with
/// `properties`; `array` for one with `items`.
fn schema_type(schema: &Value) -> Option<String> {
    let schema = schema.as_object()?;

    match schema.get("type") {
        Some(Value::String(name)) => return Some(name.clone()),
        Some(Value::Array(names)) if !names.is_empty() => {
            let stated = names
                .iter()
                .filter_map(Value::as_str)
                .find(|name| *name != "null");
            return Some(stated.unwrap_or("string").to_owned());
        }
        _ => {}
    }

    let union = ["anyOf", "oneOf"]
        .into_iter()
        .find_map(|keyword| schema.get(keyword).and_then(Value::as_array))
        .filter(|branches| !branches.is_empty());
    if let Some(branches) = union {
        let types: Vec<String> = branches.iter().filter_map(schema_type).collect();
        if let Some(first) = types.first() {
            let mut distinct = types.clone();
            distinct.sort();
            distinct.dedup();
            if distinct.len() == 1 {
                return Some(first.clone());
            }
            if distinct.len() == 2 && distinct.iter().any(|name| name == "null") {
                return types.into_iter().find(|name| name != "null");
            }
            if distinct.iter().any(|name| name == "string") {
                return Some("string".to_owned());
            }
            return Some(first.clone());
        }
    }

    if let Some(Value::Array(values)) = schema.get("enum") {
        let mut kinds: Vec<&str> = values
            .iter()
            .map(|value| match value {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(number) if number.is_f64() => "number",
                Value::Number(_) => "integer",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            })
            .collect();
        kinds.sort_unstable();
        kinds.dedup();
        return Some(match kinds.as_slice() {
            [kind] => (*kind).to_owned(),
            _ => "string".to_owned(),
        });
    }

    if let Some(Value::Array(branches)) = schema.get("allOf") {
        let stated = branches
            .iter()
            .filter_map(schema_type)
            .find(|name| name != "string");
        return Some(stated.unwrap_or_else(|| "string".to_owned()));
    }

    if schema.contains_key("properties") {
        return Some("object".to_owned());
    }
    if schema.contains_key("items") {
        return Some("array".to_owned());
    }
    None
}

/// The value a parameter's text `raw` states, typed by the `schema` its tool
/// declares for the argument, as the reference's `_convert_param_value` types
/// it. A value that is not what its schema asks for stays the string the
/// model wrote.
///
/// - `null`, in any letter case, is null whatever the schema.
/// - An argument without a schema is a string.
/// - A string type keeps the text. A string whose schema is a union
///   (`anyOf`, `oneOf`) and whose text opens a JSON object or array is that
///   structure when a branch of the union is of its kind; the reference
///   validates the structure against the whole schema, which this parser
///   does not.
/// - An integer type parses a decimal integer. A number type parses a finite
///   float, and writes one without a fraction or exponent in its text as an
///   integer.
/// - A boolean type is true for `true` in any letter case, and false
///   otherwise.
/// - An object or array type parses JSON; failing that, and for any other
///   type, a Python literal (`{'a': 1}`, `(1, 2)`, `True`, `None`).
fn typed_value(raw: &str, schema: Option<&Value>) -> Value {
    let text = || Value::String(raw.to_owned());

    if raw.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    let Some(schema) = schema else {
        return text();
    };
    let stated = schema_type(schema).unwrap_or_else(|| "string".to_owned());
    let stated = stated.trim().to_ascii_lowercase();

    match stated.as_str() {
        "string" | "str" | "text" | "varchar" | "char" | "enum" => {
            let union = ["anyOf", "oneOf"]
                .into_iter()
                .find_map(|keyword| schema.get(keyword).and_then(Value::as_array));
            let opens_structure = raw.trim_start().starts_with(['{', '[']);
            match union {
                Some(branches) if opens_structure => serde_json::from_str::<Value>(raw)
                    .ok()
                    .filter(|value| {
                        let kind = match value {
                            Value::Object(_) => "object",
                            Value::Array(_) => "array",
                            _ => return false,
                        };
                        branches
                            .iter()
                            .any(|branch| schema_type(branch).as_deref() == Some(kind))
                    })
                    .unwrap_or_else(text),
                _ => text(),
            }
        }
        name if ["int", "uint", "long", "short", "unsigned"]
            .iter()
            .any(|prefix| name.starts_with(prefix)) =>
        {
            integer(raw).unwrap_or_else(text)
        }
        name if name.starts_with("num") || name.starts_with("float") => {
            let Some(number) = raw
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
            else {
                return text();
            };
            // `5` for a number is the integer 5; `5.0` and `5e0` stay floats.
            let written_as_integer = !raw.contains(['.', 'e', 'E']);
            if written_as_integer && number.fract() == 0.0 {
                integer(raw).unwrap_or_else(text)
            } else {
                Number::from_f64(number).map_or_else(text, Value::Number)
            }
        }
        "boolean" | "bool" | "binary" => Value::Bool(raw.eq_ignore_ascii_case("true")),
        name => {
            let structured = matches!(name, "object" | "array" | "arr")
                || name.starts_with("dict")
                || name.starts_with("list");
            structured
                .then(|| serde_json::from_str::<Value>(raw).ok())
                .flatten()
                .or_else(|| python_literal(raw))
                .unwrap_or_else(text)
        }
    }
}

/// The decimal integer `raw` states, surrounding whitespace aside.
fn integer(raw: &str) -> Option<Value> {
    let digits = raw.trim();
    digits
        .parse::<i64>()
        .map(Value::from)
        .or_else(|_| digits.parse::<u64>().map(Value::from))
        .ok()
}

/// The value a Python literal states, as JSON: `None`, `True`, `False`,
/// numbers, quoted strings, lists, tuples (as arrays) and dicts whose keys are
/// strings, numbers, booleans or `None` (as `json.dumps` writes such keys).
/// `None` for any other text, a set included, which JSON cannot state.
fn python_literal(text: &str) -> Option<Value> {
    let mut literal = Literal { rest: text.trim() };
    let value = literal.value()?;
    literal.rest.trim_start().is_empty().then_some(value)
}

/// A cursor over the text of a Python literal.
struct Literal<'a> {
    rest: &'a str,
}

impl Literal<'_> {
    /// Takes `token` at the cursor, after whitespace.
    fn take(&mut self, token: &str) -> bool {
        let rest = self.rest.trim_start();
        match rest.strip_prefix(token) {
            Some(after) => {
                self.rest = after;
                true
            }
            None => false,
        }
    }

    /// Takes one value at the cursor.
    fn value(&mut self) -> Option<Value> {
        self.rest = self.rest.trim_start();
        let first = self.rest.chars().next()?;
        match first {
            '[' => self.sequence('[', "]"),
            '(' => self.sequence('(', ")"),
            '{' => self.dict(),
            '\'' | '"' => self.string(first).map(Value::String),
            _ => self.atom(),
        }
    }

    /// Takes the items of a list or a tuple, after its opening bracket.
    fn sequence(&mut self, open: char, close: &str) -> Option<Value> {
        self.rest = &self.rest[open.len_utf8()..];
        let mut items = Vec::new();
        loop {
            if self.take(close) {
                return Some(Value::Array(items));
            }
            items.push(self.value()?);
            if !self.take(",") {
                return self.take(close).then_some(Value::Array(items));
            }
        }
    }

    /// Takes the entries of a dict, after its opening brace.
    fn dict(&mut self) -> Option<Value> {
        self.rest = &self.rest[1..];
        let mut entries = Map::new();
        loop {
            if self.take("}") {
                return Some(Value::Object(entries));
            }
            let key = match self.value()? {
                Value::String(key) => key,
                Value::Null => "null".to_owned(),
                Value::Bool(key) => key.to_string(),
                Value::Number(key) => key.to_string(),
                Value::Array(_) | Value::Object(_) => return None,
            };
            if !self.take(":") {
                return None;
            }
            let value = self.value()?;
            entries.insert(key, value);
            if !self.take(",") {
                return self.take("}").then_some(Value::Object(entries));
            }
        }
    }

    /// Takes a string quoted by `quote`, with its escapes read.
    fn string(&mut self, quote: char) -> Option<String> {
        let mut characters = self.rest[quote.len_utf8()..].char_indices();
        let mut text = String::new();
        while let Some((index, character)) = characters.next() {
            match character {
                _ if character == quote => {
                    // The cursor moves past both quotes.
                    self.rest = &self.rest[quote.len_utf8() + index + quote.len_utf8()..];
                    return Some(text);
                }
                '\\' => {
                    let (_, escaped) = characters.next()?;
                    text.push(match escaped {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '0' => '\0',
                        'x' | 'u' => {
                            let digits = if escaped == 'x' { 2 } else { 4 };
                            let code: String =
                                characters.by_ref().take(digits).map(|(_, c)| c).collect();
                            if code.len() != digits {
                                return None;
                            }
                            char::from_u32(u32::from_str_radix(&code, 16).ok()?)?
                        }
                        other => other,
                    });
                }
                _ => text.push(character),
            }
        }
        None
    }

    /// Takes a keyword or a number.
    fn atom(&mut self) -> Option<Value> {
        let end = self
            .rest
            .find(|character: char| {
                !(character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.' | '_'))
            })
            .unwrap_or(self.rest.len());
        let (atom, rest) = self.rest.split_at(end);
        let value = match atom {
            "None" => Value::Null,
            "True" => Value::Bool(true),
            "False" => Value::Bool(false),
            number => integer(number).or_else(|| {
                let float = number
                    .parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite())?;
                // Only digits state a number: `inf` and `nan` are names in Python.
                number
                    .starts_with(|first: char| {
                        first.is_ascii_digit() || matches!(first, '+' | '-' | '.')
                    })
                    .then(|| Number::from_f64(float).map(Value::Number))
                    .flatten()
            })?,
        };
        self.rest = rest;
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::output::tools::test_utils::{collect_stream, split_by_chars};
    use crate::output::tools::{ToolParserItem, ToolParserOutput};
    use crate::protocol::Function;

    /// The function tool `name`, taking arguments of the JSON schema
    /// `parameters`.
    fn function(name: &str, parameters: Value) -> Tool {
        Tool::Function {
            function: Function {
                name: name.to_owned(),
                description: None,
                parameters,
                strict: None,
            },
        }
    }

    /// A parser of calls to a file writer, a calculator of every scalar type,
    /// and a tool that takes no arguments.
    fn parser() -> XmlToolParser {
        XmlToolParser::new(vec![
            function(
                "write_file",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {"enum": ["append", "replace"]},
                    },
                    "required": ["path", "content"],
                }),
            ),
            function(
                "compute",
                json!({
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "exact": {"type": "boolean"},
                        "limits": {"type": "object"},
                        "values": {"type": "array", "items": {"type": "number"}},
                        "label": {"type": ["string", "null"]},
                        "target": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                    },
                }),
            ),
            function("list_files", json!({"type": "object", "properties": {}})),
        ])
    }

    /// One call as the format writes it, each parameter on its own lines.
    fn call(function: &str, parameters: &[(&str, &str)]) -> String {
        let mut text = format!("<tool_call>\n<function={function}>\n");
        for (name, value) in parameters {
            text.push_str(&format!("<parameter={name}>\n{value}\n</parameter>\n"));
        }
        text.push_str("</function>\n</tool_call>");
        text
    }

    /// The calls of `output` as `(index, name, arguments)`, the arguments
    /// parsed: every call's deltas must concatenate to one JSON object.
    fn calls(output: &ToolParserOutput) -> Vec<(usize, String, Value)> {
        output
            .calls()
            .map(|call| {
                let arguments = serde_json::from_str(&call.arguments)
                    .unwrap_or_else(|error| panic!("{:?}: {error}", call.arguments));
                (
                    call.tool_index,
                    call.name
                        .clone()
                        .expect("a coalesced call names its function"),
                    arguments,
                )
            })
            .collect()
    }

    /// The format of the chapter: a function, its parameters, each value on
    /// its own lines.
    #[test]
    fn a_call_names_its_function_and_its_arguments() {
        let text = call(
            "write_file",
            &[("path", "notes/a.txt"), ("content", "hello")],
        );
        let output = parser().parse_complete(&text).unwrap();

        assert_eq!(output.normal_text(), "");
        assert_eq!(
            calls(&output),
            [(
                0,
                "write_file".to_owned(),
                json!({"path": "notes/a.txt", "content": "hello"})
            )]
        );
        // The arguments are written as `json.dumps` writes them.
        assert_eq!(
            output.calls().next().unwrap().arguments,
            r#"{"path": "notes/a.txt", "content": "hello"}"#
        );
    }

    /// Each value is typed by the schema its tool declares for the argument;
    /// an argument the schema does not declare, and a value that is not what
    /// its schema asks for, stay the strings the model wrote.
    #[test]
    fn values_are_typed_by_the_tools_schema() {
        let text = call(
            "compute",
            &[
                ("count", "42"),
                ("ratio", "2.50"),
                ("exact", "True"),
                ("limits", r#"{"low": 1, "high": [2, 3]}"#),
                ("values", "[1, 2.5]"),
                ("label", "null"),
                ("target", r#"{"id": 7}"#),
                ("undeclared", "7"),
            ],
        );
        let output = parser().parse_complete(&text).unwrap();

        assert_eq!(
            calls(&output),
            [(
                0,
                "compute".to_owned(),
                json!({
                    "count": 42,
                    "ratio": 2.5,
                    "exact": true,
                    "limits": {"low": 1, "high": [2, 3]},
                    "values": [1, 2.5],
                    "label": null,
                    "target": {"id": 7},
                    "undeclared": "7",
                })
            )]
        );

        let mistyped = call(
            "compute",
            &[
                ("count", "many"),
                ("ratio", "5"),
                ("exact", "yes"),
                ("target", "the roof"),
            ],
        );
        let output = parser().parse_complete(&mistyped).unwrap();
        assert_eq!(
            calls(&output)[0].2,
            json!({"count": "many", "ratio": 5, "exact": false, "target": "the roof"})
        );
    }

    /// A value keeps its inner newlines, angle brackets and quotes: only the
    /// newline after the tag and the one before the end are the format's.
    #[test]
    fn a_value_keeps_its_own_text() {
        let content =
            "fn main() {\n    if a < b && b > 0 {\n        println!(\"<ok>\");\n    }\n}\n";
        let text = call(
            "write_file",
            &[("path", "src/main.rs"), ("content", content)],
        );
        let output = parser().parse_complete(&text).unwrap();

        assert_eq!(
            calls(&output)[0].2,
            json!({"path": "src/main.rs", "content": content})
        );
    }

    /// A response holds several calls, in several regions or in one, and the
    /// text before, between and after them stays text in its place.
    #[test]
    fn a_response_holds_several_calls_among_its_text() {
        let text = format!(
            "First the file.\n{}\nThen the listing.{}done",
            call("write_file", &[("path", "a"), ("content", "b")]),
            call("list_files", &[]),
        );
        let output = parser().parse_complete(&text).unwrap();

        let items: Vec<String> = output
            .items
            .iter()
            .map(|item| match item {
                ToolParserItem::Text(text) => format!("text:{text}"),
                ToolParserItem::Call(call) => {
                    format!("call:{}:{}", call.tool_index, call.name.as_deref().unwrap())
                }
            })
            .collect();
        assert_eq!(
            items,
            [
                "text:First the file.\n",
                "call:0:write_file",
                "text:\nThen the listing.",
                "call:1:list_files",
                "text:done",
            ]
        );
        assert_eq!(calls(&output)[1].2, json!({}));

        // Two functions inside one region are two calls.
        let joined = "<tool_call>\n<function=list_files>\n</function>\n\
                      <function=compute>\n<parameter=count>\n3\n</parameter>\n</function>\n</tool_call>";
        let output = parser().parse_complete(joined).unwrap();
        assert_eq!(
            calls(&output),
            [
                (0, "list_files".to_owned(), json!({})),
                (1, "compute".to_owned(), json!({"count": 3})),
            ]
        );
    }

    /// However the stream is cut into fragments, tags included, the parser
    /// publishes the calls and the text of the whole response.
    #[test]
    fn fragments_of_any_size_give_the_whole_response() {
        let text = format!(
            "a < b, so:\n{}{}\nafter",
            call(
                "compute",
                &[
                    ("count", "12"),
                    ("limits", r#"{"low": 1}"#),
                    ("label", "x<y")
                ]
            ),
            call(
                "write_file",
                &[("path", "p"), ("content", "line 1\nline 2")]
            ),
        );
        let whole = parser().parse_complete(&text).unwrap();
        assert_eq!(whole.normal_text(), "a < b, so:\n\nafter");
        assert_eq!(
            calls(&whole),
            [
                (
                    0,
                    "compute".to_owned(),
                    json!({"count": 12, "limits": {"low": 1}, "label": "x<y"})
                ),
                (
                    1,
                    "write_file".to_owned(),
                    json!({"path": "p", "content": "line 1\nline 2"})
                ),
            ]
        );

        for size in [1, 2, 3, 5, 7, 11, 64] {
            let streamed = collect_stream(&mut parser(), &split_by_chars(&text, size));
            assert_eq!(
                streamed.normal_text(),
                whole.normal_text(),
                "fragments of {size}"
            );
            assert_eq!(calls(&streamed), calls(&whole), "fragments of {size}");
        }
    }

    /// A call streams as the OpenAI chunk shape needs it: its name once, then
    /// its arguments in fragments, one per parameter, as each value ends.
    #[test]
    fn a_call_streams_its_name_then_one_fragment_per_parameter() {
        let mut parser = parser();
        let mut streamed = Vec::new();
        for fragment in [
            "Calling.<tool",
            "_call>\n<function=comp",
            "ute>\n<parameter=count>\n4",
            "2\n</parameter>\n<parameter=exact>\nfalse\n</para",
            "meter>\n</function>\n</tool_call>",
        ] {
            let output = parser.parse_chunk(fragment).unwrap();
            streamed.push(
                output
                    .items
                    .iter()
                    .map(|item| match item {
                        ToolParserItem::Text(text) => format!("text:{text}"),
                        ToolParserItem::Call(call) => match &call.name {
                            Some(name) => format!("name:{name}"),
                            None => format!("arguments:{}", call.arguments),
                        },
                    })
                    .collect::<Vec<_>>(),
            );
        }

        assert_eq!(
            streamed,
            [
                vec!["text:Calling."],
                vec![],
                vec!["name:compute"],
                vec![r#"arguments:{"count": 42"#],
                vec![r#"arguments:, "exact": false"#, "arguments:}"],
            ]
        );
        assert!(parser.finish().unwrap().items.is_empty());
    }

    /// A model that leaves out an end tag still states its values: a parameter
    /// ends at the next parameter or at the function's end.
    #[test]
    fn a_parameter_without_its_end_tag_ends_at_the_next_tag() {
        let text = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n\
                    <parameter=content>\nbody\n</function>\n</tool_call>";
        let output = parser().parse_complete(text).unwrap();

        assert_eq!(
            calls(&output)[0].2,
            json!({"path": "a.txt", "content": "body"})
        );
    }

    /// A call the generation's end cut short keeps what it stated: its last
    /// value runs to the end of the text and its arguments close.
    #[test]
    fn a_call_cut_short_keeps_what_it_stated() {
        let text = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n\
                    <parameter=content>\nhalf a li";
        for size in [3, 1000] {
            let output = collect_stream(&mut parser(), &split_by_chars(text, size));
            assert_eq!(
                calls(&output),
                [(
                    0,
                    "write_file".to_owned(),
                    json!({"path": "a.txt", "content": "half a li"})
                )],
                "fragments of {size}"
            );
        }
    }

    /// Text that only resembles a tag is text, and a held-back `<` that
    /// became no tag is released at the end.
    #[test]
    fn text_that_opens_no_tag_stays_text() {
        let mut parser = parser();
        let first = parser.parse_chunk("x <b> y <functional> z <").unwrap();
        assert_eq!(first.normal_text(), "x <b> y <functional> z ");
        let rest = parser.finish().unwrap();
        assert_eq!(rest.normal_text(), "<");
        assert_eq!(rest.calls().count(), 0);
    }

    /// A parser handed over at any point of the stream, as a hot restart
    /// hands a response over, goes on to the same calls and text.
    #[test]
    fn a_handed_over_parser_goes_on_where_it_stopped() {
        let text = format!(
            "before{}after",
            call("compute", &[("count", "5"), ("values", "[1, 2]")])
        );
        let whole = parser().parse_complete(&text).unwrap();

        for split in 1..text.len() {
            let mut first = parser();
            let mut output = first.parse_chunk(&text[..split]).unwrap();
            let state = first.state().expect("the parser states itself");

            let mut second = parser();
            second.resume(&state);
            output.append(second.parse_chunk(&text[split..]).unwrap());
            output.append(second.finish().unwrap());
            let output = output.coalesce_calls();

            assert_eq!(
                output.normal_text(),
                whole.normal_text(),
                "split at {split}"
            );
            assert_eq!(calls(&output), calls(&whole), "split at {split}");
        }
    }

    /// The primary type of a schema, where it does not state `type` alone.
    #[test]
    fn a_schemas_type_is_inferred_from_what_it_states() {
        for (schema, expected) in [
            (json!({"type": ["null", "integer"]}), Some("integer")),
            (
                json!({"anyOf": [{"type": "number"}, {"type": "null"}]}),
                Some("number"),
            ),
            (
                json!({"oneOf": [{"type": "integer"}, {"type": "string"}]}),
                Some("string"),
            ),
            (json!({"enum": [1, 2, 3]}), Some("integer")),
            (json!({"enum": ["a", 1]}), Some("string")),
            (
                json!({"allOf": [{"type": "string"}, {"type": "object"}]}),
                Some("object"),
            ),
            (json!({"properties": {}}), Some("object")),
            (json!({"items": {}}), Some("array")),
            (json!({"description": "anything"}), None),
        ] {
            assert_eq!(schema_type(&schema).as_deref(), expected, "{schema}");
        }
    }

    /// A structured value written as a Python literal is read as one, and
    /// text that is no literal stays a string.
    #[test]
    fn a_structured_value_may_be_a_python_literal() {
        let object = json!({"type": "object"});
        assert_eq!(
            typed_value("{'a': (1, 2.5), 'b': None, 3: True}", Some(&object)),
            json!({"a": [1, 2.5], "b": null, "3": true})
        );
        assert_eq!(
            typed_value("['x', \"q\\n\",]", Some(&json!({"type": "array"}))),
            json!(["x", "q\n"])
        );
        assert_eq!(
            typed_value("not a literal", Some(&object)),
            json!("not a literal")
        );
        assert_eq!(typed_value("{1, 2}", Some(&object)), json!("{1, 2}"));
    }
}
