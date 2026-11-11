//! Incremental parser for JSON tool calls enclosed by text markers.
//!
//! The parser is a three-mode state machine over a buffered stream:
//!
//! - `Text` emits plain text up to the start delimiter and holds back a
//!   trailing partial delimiter. The start delimiter switches to `Header`. A
//!   start marker without the rest of the delimiter, such as a `<tool_call>`
//!   tag mentioned in prose, is plain text.
//! - `Header` parses `{"<name key>": "<name>", "<arguments key>":` with
//!   optional JSON whitespace and completes once the arguments value is seen
//!   to open a JSON object. It then publishes the call by emitting its first
//!   delta (name, empty arguments) and switches to `Arguments`.
//! - `Arguments` streams the raw arguments object lexically as argument
//!   deltas. After the object closes, it expects the wrapper's `}` after
//!   optional JSON whitespace, then the end delimiter, and returns to `Text`.
//!
//! Any other header shape, such as reordered or extra keys or arguments that
//! are not an object, is a parse error.
//!
//! A header is parsed atomically, so until its call is published the parser
//! can still return every byte of the attempted call: after a parse error or
//! at the end of input, `reset` and `finish` yield it as text, start delimiter
//! included. A published call is never withdrawn; after a later failure only
//! the input following its streamed arguments is returned.

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, peek, seq};
use winnow::error::{AddContext, ModalResult, StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::stream::{Partial, Stream};
use winnow::token::literal;

use proto::handoff;

use super::utils::{
    JsonObjectScanPhase, JsonObjectScanState, MAX_BUFFER_BYTES, json_str, parse_buffered_event,
    safe_text_len, take_json_object,
};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};

type JsonToolInput<'i> = Partial<&'i str>;

/// Marker and key vocabulary of one marker-wrapped JSON tool-call format.
#[derive(Debug, Clone, Copy)]
struct JsonToolCallConfig {
    /// Name used in parse error messages.
    parser_name: &'static str,
    /// Exact text that opens a tool call: the start marker together with the
    /// whitespace the format requires after it. Text mode scans for the whole
    /// delimiter, so the marker alone never starts a tool call.
    start_delimiter: &'static str,
    /// Exact text that closes a tool call after the wrapper object: the
    /// whitespace the format requires before the end marker, then the marker.
    end_delimiter: &'static str,
    /// JSON key of the function name, which must be the header's first key.
    name_key: &'static str,
    /// Candidate JSON keys naming the arguments payload, which must follow the
    /// name. The header accepts any one of them.
    arguments_key: &'static [&'static str],
}

/// Parser mode; see the module documentation for the transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonToolCallMode {
    Text,
    Header,
    Arguments { json_scan: JsonObjectScanState },
}

/// One parsed unit of the buffered stream.
///
/// `len` fields count bytes at the head of the parser buffer that
/// `JsonToolParser::apply_event` copies into the output before the buffer
/// drains them.
#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonToolCallEvent {
    Text { len: usize },
    ToolCallStart,
    ToolCallHeader { function_name: String },
    Arguments { len: usize },
    ToolCallEnd,
}

/// The newlines belong to the delimiters, as in the format the chat templates
/// of its families instruct and SGLang's detector matches, so a bare
/// `<tool_call>` tag in prose stays plain text.
const JSON_TOOL_CALL: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "JSON",
    start_delimiter: "<tool_call>\n",
    end_delimiter: "\n</tool_call>",
    name_key: "name",
    arguments_key: &["arguments"],
};

/// The parser of the JSON tool-call format (`processing::ToolCallFormat::Json`):
/// one JSON object per call inside `<tool_call>` tags.
///
/// ```text
/// <tool_call>
/// {"name": "get_weather", "arguments": {"location":"Tokyo"}}
/// </tool_call>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas, neither converted by a schema nor normalized; the parser
/// reads no tool definition, and parsed function names are not checked against
/// the request's tools. Parallel calls are repeated `<tool_call>...</tool_call>`
/// blocks, not several calls inside one tag.
#[derive(Debug)]
pub struct JsonToolParser {
    config: JsonToolCallConfig,
    /// Input not yet consumed by a complete event.
    buffer: String,
    mode: JsonToolCallMode,
    /// Tool index that argument deltas extend; set by a header, cleared by the
    /// end delimiter.
    active_tool_index: Option<usize>,
    /// Number of tool calls started since the last reset; also the next
    /// call's `tool_index`.
    emitted_tool_count: usize,
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
            config: JSON_TOOL_CALL,
            buffer: String::new(),
            mode: JsonToolCallMode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// The parser's state, as `JsonToolCall`'s fields hold it.
    fn json_state(&self) -> handoff::JsonToolCall {
        let mode = match &self.mode {
            JsonToolCallMode::Text => handoff::JsonToolMode {
                kind: handoff::JsonToolModeKind::JsonToolModeText as i32,
                scan: None,
            },
            JsonToolCallMode::Header => handoff::JsonToolMode {
                kind: handoff::JsonToolModeKind::JsonToolModeHeader as i32,
                scan: None,
            },
            JsonToolCallMode::Arguments { json_scan } => handoff::JsonToolMode {
                kind: handoff::JsonToolModeKind::JsonToolModeArguments as i32,
                scan: Some(handoff::JsonObjectScan {
                    object_depth: json_scan.object_depth as u64,
                    array_depth: json_scan.array_depth as u64,
                    in_string: json_scan.in_string,
                    escape: json_scan.escape,
                    phase: match json_scan.phase {
                        JsonObjectScanPhase::Initial => {
                            handoff::json_object_scan::Phase::JsonObjectPhaseInitial
                        }
                        JsonObjectScanPhase::Scanning => {
                            handoff::json_object_scan::Phase::JsonObjectPhaseScanning
                        }
                        JsonObjectScanPhase::Complete => {
                            handoff::json_object_scan::Phase::JsonObjectPhaseComplete
                        }
                    } as i32,
                }),
            },
        };
        handoff::JsonToolCall {
            buffer: self.buffer.clone(),
            mode: Some(mode),
            active_tool_index: self.active_tool_index.map(|index| index as u64),
            emitted_tool_count: self.emitted_tool_count as u64,
        }
    }

    /// Takes over a state this parser produced (`JsonToolParser::json_state`).
    fn resume_json(&mut self, state: &handoff::JsonToolCall) {
        self.buffer.clone_from(&state.buffer);
        let mode = state.mode.as_ref().map_or(JsonToolCallMode::Text, |mode| {
            let kind = handoff::JsonToolModeKind::try_from(mode.kind)
                .unwrap_or(handoff::JsonToolModeKind::JsonToolModeText);
            match kind {
                handoff::JsonToolModeKind::JsonToolModeArguments => {
                    let scan = mode.scan.as_ref().map_or_else(
                        JsonObjectScanState::default,
                        |scan: &handoff::JsonObjectScan| JsonObjectScanState {
                            object_depth: scan.object_depth as usize,
                            array_depth: scan.array_depth as usize,
                            in_string: scan.in_string,
                            escape: scan.escape,
                            phase: match handoff::json_object_scan::Phase::try_from(scan.phase) {
                                Ok(handoff::json_object_scan::Phase::JsonObjectPhaseScanning) => {
                                    JsonObjectScanPhase::Scanning
                                }
                                Ok(handoff::json_object_scan::Phase::JsonObjectPhaseComplete) => {
                                    JsonObjectScanPhase::Complete
                                }
                                _ => JsonObjectScanPhase::Initial,
                            },
                        },
                    );
                    JsonToolCallMode::Arguments { json_scan: scan }
                }
                handoff::JsonToolModeKind::JsonToolModeHeader => JsonToolCallMode::Header,
                handoff::JsonToolModeKind::JsonToolModeText => JsonToolCallMode::Text,
            }
        });
        self.mode = mode;
        self.active_tool_index = state.active_tool_index.map(|index| index as usize);
        self.emitted_tool_count = state.emitted_tool_count as usize;
    }

    /// Applies one parsed JSON tool-call event to parser state and output.
    fn apply_event(
        &mut self,
        event: JsonToolCallEvent,
        output: &mut ToolParserOutput,
    ) -> Result<()> {
        match event {
            JsonToolCallEvent::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            JsonToolCallEvent::ToolCallStart => self.mode = JsonToolCallMode::Header,
            JsonToolCallEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = JsonToolCallMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            JsonToolCallEvent::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "{} arguments without an active tool call",
                        self.config.parser_name
                    ));
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            JsonToolCallEvent::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = JsonToolCallMode::Text;
            }
        }
        Ok(())
    }
}

impl ToolParser for JsonToolParser {
    fn state(&self) -> Option<handoff::tool_stage::Parser> {
        Some(handoff::tool_stage::Parser::Json(self.json_state()))
    }

    fn resume(&mut self, state: &handoff::tool_stage::Parser) {
        if let handoff::tool_stage::Parser::Json(json) = state {
            self.resume_json(json);
        }
    }

    /// Advances incremental parsing and appends every complete event to `output`.
    ///
    /// On error, events parsed before the failure stay in `output` and the
    /// buffer keeps the input that no completed event consumed.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);
        let config = self.config;

        // An event's bytes are applied before they are drained, because text and
        // argument events copy their payload out of the buffer head.
        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_json_tool_call_event(input, &mut self.mode, config)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        // Any bytes still buffered here belong to an event that has not yet
        // completed. Argument bytes leave the buffer as soon as they arrive,
        // but an unterminated tool-call header (for example an endless
        // function-name string) keeps accumulating, so bound that retention.
        if self.buffer.len() > MAX_BUFFER_BYTES {
            return Err(parsing_failed!(
                "{} buffer exceeded {} bytes without completing a tool call",
                self.config.parser_name,
                MAX_BUFFER_BYTES
            ));
        }

        Ok(())
    }

    /// Finalizes buffered input or rejects an incomplete published call.
    ///
    /// Outside a published call, the input `reset` returns becomes plain text
    /// and the parser resets: in text mode the buffer, including any held-back
    /// partial start delimiter; in header mode the start delimiter and the
    /// unfinished header. While a published call is open (from its header
    /// until its end delimiter is parsed) this fails and leaves the parser
    /// state unchanged. The call's argument bytes have already been emitted,
    /// so the buffer then holds at most a prefix of the wrapper's closing text.
    fn finish(&mut self) -> Result<ToolParserOutput> {
        if matches!(self.mode, JsonToolCallMode::Arguments { .. }) {
            return Err(parsing_failed!(
                "incomplete {} tool call",
                self.config.parser_name
            ));
        }

        let mut output = ToolParserOutput::default();
        output.push_text(&self.reset());
        Ok(output)
    }

    /// Resets the incremental parser state and returns the input that no
    /// emitted output represents.
    ///
    /// That is the unconsumed buffer, preceded in header mode by the start
    /// delimiter: its event has been applied, but no call has been published
    /// for it yet. Tool indices restart at 0 afterwards.
    fn reset(&mut self) -> String {
        let mut unpublished = std::mem::take(&mut self.buffer);
        if self.mode == JsonToolCallMode::Header {
            unpublished.insert_str(0, self.config.start_delimiter);
        }

        self.mode = JsonToolCallMode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        unpublished
    }
}

/// Parses a JSON tool-call event for the current parser mode.
fn parse_next_json_tool_call_event(
    input: &mut JsonToolInput<'_>,
    mode: &mut JsonToolCallMode,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    match mode {
        JsonToolCallMode::Text => parse_text_event(input, config),
        JsonToolCallMode::Header => tool_call_header_event(input, config),
        JsonToolCallMode::Arguments { json_scan } => {
            parse_arguments_event(input, json_scan, config)
        }
    }
}

/// Parses a text-mode JSON tool-call event.
fn parse_text_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    alt((
        |input: &mut JsonToolInput<'_>| tool_call_start_event(input, config),
        |input: &mut JsonToolInput<'_>| safe_text_event(input, config),
    ))
    .parse_next(input)
}

/// Parses a marker-wrapped JSON tool-call start delimiter.
fn tool_call_start_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    literal(config.start_delimiter)
        .value(JsonToolCallEvent::ToolCallStart)
        .parse_next(input)
}

/// Parses a marker-wrapped JSON tool-call header before the raw arguments
/// payload.
fn tool_call_header_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    let (function_name,) = seq!(
        _: ws0,
        _: literal("{"),
        _: ws0,
        _: |input: &mut JsonToolInput<'_>| json_key(input, config.name_key),
        _: ws0,
        _: literal(":"),
        _: ws0,
        json_str,
        _: ws0,
        _: literal(","),
        _: ws0,
        _: |input: &mut JsonToolInput<'_>| json_arguments_key(input, config.arguments_key),
        _: ws0,
        _: literal(":"),
        _: ws0,
        // The header waits for the arguments value to open an object, so
        // arguments of any other JSON type fail before the call is published.
        _: peek(literal("{")),
    )
    .context(StrContext::Label(config.parser_name))
    .parse_next(input)?;

    Ok(JsonToolCallEvent::ToolCallHeader { function_name })
}

/// Parses a configured JSON object key.
fn json_key(input: &mut JsonToolInput<'_>, key: &'static str) -> ModalResult<()> {
    seq!(
        _: literal("\""),
        _: literal(key).context(StrContext::Expected(StrContextValue::StringLiteral(key))),
        _: literal("\""),
    )
    .void()
    .parse_next(input)
}

/// Parses a JSON object key accepting any of `candidates`.
///
/// The full quoted key is parsed with `json_str` and then compared against the
/// candidate list, so partial input stays `Incomplete` until the closing quote
/// is buffered, whatever the candidates' lengths.
///
/// On mismatch, each candidate is attached as its own `Expected` context so the
/// error enumerates every valid key ("expected `a`, expected `b`").
/// `StrContextValue::StringLiteral` carries a single `&'static str`, so the
/// contexts are folded over `candidates` rather than chained with `.context`.
fn json_arguments_key(
    input: &mut JsonToolInput<'_>,
    candidates: &'static [&'static str],
) -> ModalResult<()> {
    let start = input.checkpoint();
    json_str
        .verify(|key: &String| candidates.contains(&key.as_str()))
        .void()
        .parse_next(input)
        .map_err(|err| {
            err.map(|context_error| {
                candidates
                    .iter()
                    .fold(context_error, |context_error, candidate| {
                        context_error.add_context(
                            &*input,
                            &start,
                            StrContext::Expected(StrContextValue::StringLiteral(candidate)),
                        )
                    })
            })
        })
}

/// Parses one event inside a marker-wrapped JSON tool-call arguments payload.
fn parse_arguments_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    if json_scan.complete() {
        tool_call_close_event(input, config)
    } else {
        argument_delta_event(input, json_scan)
    }
}

/// Parses a raw JSON arguments delta.
fn argument_delta_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<JsonToolCallEvent> {
    take_json_object(input, json_scan).map(|len| JsonToolCallEvent::Arguments { len })
}

/// Parses the wrapper object's `}` and the end delimiter after the arguments.
///
/// JSON whitespace may separate the arguments object from this `}`, as it may
/// separate every other token of the wrapper object.
fn tool_call_close_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    let _ = (ws0, literal("}")).parse_next(input)?;

    tool_call_end_event(input, config)
}

/// Parses a marker-wrapped JSON tool-call end delimiter.
fn tool_call_end_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    literal(config.end_delimiter)
        .value(JsonToolCallEvent::ToolCallEnd)
        .parse_next(input)
}

/// Parses a safe text run before the next marker-wrapped JSON tool call.
///
/// Scanning for the whole start delimiter keeps a start marker that the
/// delimiter's whitespace does not follow inside the text run, so it neither
/// starts a call nor stops text emission.
fn safe_text_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    safe_text_len(input, config.start_delimiter).map(|len| JsonToolCallEvent::Text { len })
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::JsonToolParser;
    use crate::output::tools::test_utils::{collect_stream, split_by_chars};
    use crate::output::tools::{ToolCallDelta, ToolParser, ToolParserItem, ToolParserOutput};

    /// Collects the tool-call updates of `output` in order.
    fn calls(output: &ToolParserOutput) -> Vec<&ToolCallDelta> {
        output.calls().collect()
    }

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(
            "<tool_call>\n{{\"name\": \"{function_name}\", \"arguments\": {arguments}}}\n</tool_call>"
        )
    }

    #[test]
    fn json_parse_complete_without_tool_call_keeps_text() {
        let mut parser = JsonToolParser::new();
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(calls(&output).is_empty());
    }

    #[test]
    fn json_parse_complete_extracts_raw_json_arguments() {
        let mut parser = JsonToolParser::new();
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let output = parser
            .parse_complete(&format!(
                "Let me check.\n{}",
                build_tool_call("get_weather", arguments)
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Let me check.\n");
        assert_eq!(calls(&output).len(), 1);
        assert_eq!(calls(&output)[0].tool_index, 0);
        assert_eq!(calls(&output)[0].name.as_deref(), Some("get_weather"));
        assert_eq!(calls(&output)[0].arguments, arguments);
    }

    #[test]
    fn json_does_not_validate_or_normalize_arguments() {
        let mut parser = JsonToolParser::new();
        let arguments = r#"{"location":"Tokyo",}"#;
        let output = parser
            .parse_complete(&build_tool_call("get_weather", arguments))
            .unwrap();

        assert_eq!(calls(&output)[0].arguments, arguments);
    }

    /// Argument bytes stream out chunk by chunk: the fifth chunk closes the
    /// arguments object and the last one only closes the wrapper and the tag,
    /// so exactly three argument deltas follow the header.
    #[test]
    fn json_streaming_emits_argument_deltas() {
        let mut parser = JsonToolParser::new();
        let chunks = [
            "<tool_call>",
            "\n{\"name\": \"get_weather\", \"arguments\": ",
            "{\"location\":",
            "\"Beijing\"",
            "}",
            "}\n</tool_call>",
        ];

        let mut output = ToolParserOutput::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.parse_chunk(chunk).unwrap();
            observed_arguments.extend(
                next.calls()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            output.append(next);
        }
        output.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(
            calls(&output.coalesce_calls())[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn json_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = JsonToolParser::new();

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello ");
        assert_eq!(calls(&output).len(), 1);
        assert_eq!(calls(&output)[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn json_keeps_end_marker_literal_inside_json_string() {
        let mut parser = JsonToolParser::new();
        let arguments = r#"{"text":"literal </tool_call> inside"}"#;
        let output = parser
            .parse_complete(&build_tool_call("echo", arguments))
            .unwrap();

        assert_eq!(calls(&output).len(), 1);
        assert_eq!(calls(&output)[0].arguments, arguments);
    }

    #[test]
    fn json_decodes_escaped_function_name() {
        let mut parser = JsonToolParser::new();
        let output = parser
            .parse_complete(
                r#"<tool_call>
{"name":"say_\"hi","arguments":{}}
</tool_call>"#,
            )
            .unwrap();

        assert_eq!(calls(&output)[0].name.as_deref(), Some("say_\"hi"));
    }

    /// Without the newline after `<tool_call>` the block is not a tool call,
    /// and the whole input is plain text.
    #[test]
    fn json_requires_newline_after_tool_call_start() {
        let mut parser = JsonToolParser::new();
        let input = r#"<tool_call>{"name":"get_weather","arguments":{}}
</tool_call>"#;

        let output = parser.parse_complete(input).unwrap();

        assert_eq!(output.normal_text(), input);
        assert!(calls(&output).is_empty());
    }

    /// A `<tool_call>` tag that does not start a tool call streams out as
    /// text with its chunk instead of holding back all later text.
    #[test]
    fn json_streams_text_after_tag_without_newline() {
        let mut parser = JsonToolParser::new();

        let first = parser.parse_chunk("Use the <tool_call> tag.").unwrap();
        let second = parser.parse_chunk(" More.").unwrap();

        assert_eq!(first.normal_text(), "Use the <tool_call> tag.");
        assert_eq!(second.normal_text(), " More.");
    }

    /// A tag that does not start a tool call leaves later well-formed calls
    /// in the same response parseable.
    #[test]
    fn json_parses_tool_call_after_tag_without_newline() {
        let mut parser = JsonToolParser::new();
        let output = parser
            .parse_complete(&format!(
                "<tool_call> x\n{}",
                build_tool_call("add", r#"{"x":1}"#)
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "<tool_call> x\n");
        assert_eq!(calls(&output).len(), 1);
        assert_eq!(calls(&output)[0].name.as_deref(), Some("add"));
        assert_eq!(calls(&output)[0].arguments, r#"{"x":1}"#);
    }

    /// JSON whitespace between the arguments object and the wrapper's closing
    /// brace is insignificant, as it is anywhere else inside the wrapper.
    #[test]
    fn json_accepts_whitespace_before_wrapper_close() {
        let mut parser = JsonToolParser::new();
        let output = parser
            .parse_complete(
                "<tool_call>\n{\"name\": \"add\", \"arguments\": {\"x\": 1} \n}\n</tool_call>",
            )
            .unwrap();

        assert_eq!(output.normal_text(), "");
        assert_eq!(calls(&output).len(), 1);
        assert_eq!(calls(&output)[0].name.as_deref(), Some("add"));
        assert_eq!(calls(&output)[0].arguments, r#"{"x": 1}"#);
    }

    #[test]
    fn json_requires_newline_before_tool_call_end() {
        let mut parser = JsonToolParser::new();
        let error = parser
            .parse_complete(
                r#"<tool_call>
{"name":"get_weather","arguments":{}}</tool_call>"#,
            )
            .unwrap_err();

        assert!(
            error
                .to_report_string()
                .starts_with("tool parser parsing failed:")
        );
    }

    #[test]
    fn json_streaming_extracts_multiple_tool_calls() {
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", r#"{"x":1,"y":2}"#),
        );
        let chunks = split_by_chars(&input, 7);
        let mut parser = JsonToolParser::new();

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(
            output.items,
            [
                ToolParserItem::Call(ToolCallDelta {
                    tool_index: 0,
                    name: Some("get_weather".to_string()),
                    arguments: r#"{"location":"Shanghai"}"#.to_string(),
                }),
                ToolParserItem::Call(ToolCallDelta {
                    tool_index: 1,
                    name: Some("add".to_string()),
                    arguments: r#"{"x":1,"y":2}"#.to_string(),
                }),
            ]
        );
    }

    #[test]
    fn json_finish_fails_incomplete_tool_call() {
        let mut parser = JsonToolParser::new();
        parser
            .parse_chunk(
                r#"<tool_call>
{"name":"get_weather","arguments":{"location""#,
            )
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete JSON tool call"]
            .assert_eq(&error.to_report_string());
    }

    /// An unfinished header has published no call, so finalization returns
    /// it as plain text together with its start delimiter.
    #[test]
    fn json_finish_returns_incomplete_header_as_text() {
        let mut parser = JsonToolParser::new();
        let input = "Checking.<tool_call>\n{\"name\": \"get_wea";

        let streamed = parser.parse_chunk(input).unwrap();
        let finished = parser.finish().unwrap();

        assert_eq!(
            streamed.normal_text() + finished.normal_text().as_str(),
            input
        );
        assert!(calls(&streamed).is_empty());
        assert!(calls(&finished).is_empty());
    }

    /// A header that fails to parse publishes no call, and `reset` returns
    /// the whole attempted call, start delimiter included, even when it
    /// arrived over several chunks.
    #[test]
    fn json_reset_after_header_error_returns_whole_call() {
        let mut parser = JsonToolParser::new();
        let mut output = ToolParserOutput::default();

        parser
            .parse_into("Hi <tool_call>\n{\"na", &mut output)
            .unwrap();
        let result = parser.parse_into("me\": \"add\", \"args\": {}}\n</tool_call>", &mut output);

        assert!(result.is_err());
        assert_eq!(output.normal_text(), "Hi ");
        assert!(calls(&output).is_empty());
        assert_eq!(
            parser.reset(),
            "<tool_call>\n{\"name\": \"add\", \"args\": {}}\n</tool_call>"
        );
    }

    /// Arguments that are not a JSON object fail the header, so no call is
    /// published with unusable arguments and `reset` returns the whole call.
    #[test]
    fn json_rejects_non_object_arguments_before_publishing() {
        let mut parser = JsonToolParser::new();
        let mut output = ToolParserOutput::default();
        let input = "<tool_call>\n{\"name\": \"add\", \"arguments\": \"{}\"}\n</tool_call>";

        let result = parser.parse_into(input, &mut output);

        assert!(result.is_err());
        assert!(calls(&output).is_empty());
        assert_eq!(parser.reset(), input);
    }

    #[test]
    fn json_malformed_field_order_fails_fast() {
        let mut parser = JsonToolParser::new();
        let error = parser
            .parse_chunk(
                r#"<tool_call>
{"arguments":{},"name":"get_weather"}
</tool_call>"#,
            )
            .unwrap_err();

        expect![[r#"
            tool parser parsing failed: invalid JSON
            expected `name`"#]]
        .assert_eq(&error.to_report_string());
    }
}
