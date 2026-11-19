//! The JSON tool-call format's grammar, streaming and handoff.

use serde_json::{Value, json};

use super::JsonToolParser;
use crate::output::Error;
use crate::output::tools::{ToolParser, ToolParserItem, ToolParserOutput};
use crate::testing::tool_parsing::{chunks_of, parse_chunks, parse_whole};

/// One call as the format writes it.
fn call(name: &str, arguments: &str) -> String {
    format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {arguments}}}\n</tool_call>")
}

/// The calls of a coalesced output as `(index, name, arguments text)`.
fn calls(output: &ToolParserOutput) -> Vec<(usize, &str, &str)> {
    output
        .calls()
        .map(|call| {
            (
                call.tool_index,
                call.name
                    .as_deref()
                    .expect("a coalesced call names its function"),
                call.arguments.as_str(),
            )
        })
        .collect()
}

/// Whether `error` reports a malformed JSON tool call.
fn is_malformed(error: &Error) -> bool {
    matches!(error, Error::ToolCall { format: "json", .. })
}

#[test]
fn text_without_a_call_is_text() {
    let output = parse_whole(&mut JsonToolParser::new(), "Hello, world!").unwrap();

    assert_eq!(output.normal_text(), "Hello, world!");
    assert!(calls(&output).is_empty());
}

/// A call's arguments reach the response as the text the model wrote, which is OpenAI's
/// arguments string: neither parsed into values nor reformatted.
#[test]
fn a_call_states_its_name_and_its_arguments_text() {
    let arguments = r#"{ "location": "Tokyo", "days": [1, 2.50],  "unit": null }"#;
    let text = format!("Let me check.\n{}", call("get_weather", arguments));
    let output = parse_whole(&mut JsonToolParser::new(), &text).unwrap();

    assert_eq!(output.normal_text(), "Let me check.\n");
    assert_eq!(calls(&output), [(0, "get_weather", arguments)]);
}

/// The first delta of a call names its function once its arguments object opens, and the
/// arguments then stream as they arrive, a string's characters included.
#[test]
fn arguments_stream_as_they_arrive() {
    let mut parser = JsonToolParser::new();
    let mut output = ToolParserOutput::default();
    parser
        .parse_into(
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"te",
            &mut output,
        )
        .unwrap();
    assert_eq!(output.items.len(), 2);
    let ToolParserItem::Call(first) = &output.items[0] else {
        panic!("{output:?}");
    };
    assert_eq!(
        (
            first.tool_index,
            first.name.as_deref(),
            first.arguments.as_str()
        ),
        (0, Some("f"), "")
    );
    let ToolParserItem::Call(arguments) = &output.items[1] else {
        panic!("{output:?}");
    };
    assert_eq!(arguments.arguments, "{\"te");

    let mut more = ToolParserOutput::default();
    parser
        .parse_into("xt\": \"a}\"}}\n</tool_call>", &mut more)
        .unwrap();
    let streamed: String = more.calls().map(|call| call.arguments.as_str()).collect();
    assert_eq!(streamed, "xt\": \"a}\"}");
    assert!(parser.finish().unwrap().items.is_empty());
}

/// However the text is cut into chunks, delimiters included, the parser publishes the same
/// calls and text.
#[test]
fn chunks_of_any_size_give_the_whole_response() {
    let text = format!(
        "a < b {}\nbetween\n{} after <tool_call",
        call(
            "first",
            r#"{"path": "x}", "nested": {"a": [1, {"b": "\"</tool_call>"}]}}"#
        ),
        call("second", "{}"),
    );
    let whole = parse_whole(&mut JsonToolParser::new(), &text).unwrap();
    assert_eq!(whole.normal_text(), "a < b \nbetween\n after <tool_call");
    assert_eq!(
        calls(&whole),
        [
            (
                0,
                "first",
                r#"{"path": "x}", "nested": {"a": [1, {"b": "\"</tool_call>"}]}}"#
            ),
            (1, "second", "{}"),
        ]
    );

    for size in 1..=9 {
        let chunks = chunks_of(&text, size);
        let output = parse_chunks(
            &mut JsonToolParser::new(),
            chunks.iter().map(String::as_str),
        )
        .unwrap();
        assert_eq!(output, whole, "chunks of {size}");
    }
}

#[test]
fn an_escaped_name_is_decoded() {
    let text = "<tool_call>\n{\"name\": \"get\\u005fweather\", \"arguments\": {}}\n</tool_call>";
    let output = parse_whole(&mut JsonToolParser::new(), text).unwrap();
    assert_eq!(calls(&output), [(0, "get_weather", "{}")]);
}

/// The start delimiter's newline is the format's: a bare tag in prose is text.
#[test]
fn a_tag_without_the_delimiters_newline_is_text() {
    let text = "Use <tool_call> tags. <tool_call>{\"name\": \"f\"}";
    let output = parse_whole(&mut JsonToolParser::new(), text).unwrap();
    assert_eq!(output.normal_text(), text);
    assert!(calls(&output).is_empty());
}

/// JSON whitespace may stand between the tokens of a call's object, but the end delimiter
/// follows the object at once.
#[test]
fn the_end_delimiter_follows_the_object() {
    let spaced = "<tool_call>\n { \"name\" : \"f\" , \"arguments\" : {} } \n</tool_call>";
    let error = parse_whole(&mut JsonToolParser::new(), spaced).unwrap_err();
    assert!(is_malformed(&error), "{error}");

    let tight = "<tool_call>\n { \"name\" : \"f\" , \"arguments\" : {} }\n</tool_call>";
    let output = parse_whole(&mut JsonToolParser::new(), tight).unwrap();
    assert_eq!(calls(&output), [(0, "f", "{}")]);
}

/// A call's object states `name`, a string, then `arguments`, an object, and nothing else.
#[test]
fn a_call_object_of_another_shape_is_malformed() {
    for body in [
        r#"{"arguments": {}, "name": "f"}"#,
        r#"{"name": 7, "arguments": {}}"#,
        r#"{"name": "f", "arguments": "{}"}"#,
        r#"{"name": "f", "arguments": {}, "id": 1}"#,
        r#"["f", {}]"#,
    ] {
        let text = format!("<tool_call>\n{body}\n</tool_call>");
        let error = parse_whole(&mut JsonToolParser::new(), &text).unwrap_err();
        assert!(is_malformed(&error), "{body}: {error}");
    }
}

/// Arguments that are not JSON are a malformed call, not text.
#[test]
fn arguments_that_are_not_json_are_malformed() {
    for arguments in [r#"{"a": 1,}"#, r#"{'a': 1}"#, r#"{"a": tru}"#, r#"{"a" 1}"#] {
        let text = call("f", arguments);
        let error = parse_whole(&mut JsonToolParser::new(), &text).unwrap_err();
        assert!(is_malformed(&error), "{arguments}: {error}");
    }
}

/// Text that ends inside a call is a malformed call; text that ends in a partial start
/// delimiter is text.
#[test]
fn text_that_ends_inside_a_call_is_malformed() {
    for text in [
        "<tool_call>\n{\"na",
        "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1",
        "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_",
    ] {
        let error = parse_whole(&mut JsonToolParser::new(), text).unwrap_err();
        assert!(is_malformed(&error), "{text:?}: {error}");
    }

    let output = parse_whole(&mut JsonToolParser::new(), "see <tool_call").unwrap();
    assert_eq!(output.normal_text(), "see <tool_call");
}

/// A successor's parser resumed from the state goes on where the handing parser stopped, at
/// every point of a response, and publishes nothing twice.
#[test]
fn a_handed_over_parser_goes_on_where_it_stopped() {
    let text = format!(
        "before {}\nafter",
        call("f", r#"{"a": [1, "x"], "b": {"c": true}}"#)
    );
    let whole = parse_whole(&mut JsonToolParser::new(), &text).unwrap();

    for cut in (0..=text.len()).filter(|cut| text.is_char_boundary(*cut)) {
        let mut first = JsonToolParser::new();
        let mut output = ToolParserOutput::default();
        first.parse_into(&text[..cut], &mut output).unwrap();

        let mut second = JsonToolParser::new();
        second.resume(first.state()).unwrap();
        second.parse_into(&text[cut..], &mut output).unwrap();
        output.append(second.finish().unwrap());
        assert_eq!(output.coalesce_calls(), whole, "cut at {cut}");
    }
}

/// The streamed arguments of every call concatenate to the JSON the model wrote.
#[test]
fn streamed_arguments_are_the_written_json() {
    let arguments = r#"{"query": "rust \"streams\"", "limit": 10}"#;
    let chunks = chunks_of(&call("search", arguments), 3);
    let output = parse_chunks(
        &mut JsonToolParser::new(),
        chunks.iter().map(String::as_str),
    )
    .unwrap();
    let parsed: Value = serde_json::from_str(calls(&output)[0].2).unwrap();
    assert_eq!(parsed, json!({"query": "rust \"streams\"", "limit": 10}));
}
