//! The tag tool-call format's grammar, value typing, streaming and handoff.

use serde_json::{Value, json};

use super::XmlToolParser;
use crate::api::error::Error;
use crate::api::output::tools::{ToolParser, ToolParserItem, ToolParserOutput};
use crate::api::protocol::{Function, Tool};
use crate::api::test_support::tool_parsing::{chunks_of, parse_chunks, parse_whole};

/// A function tool named `name` whose parameters are `parameters`.
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

/// A parser of calls to a file-writing tool, a tool whose arguments take every JSON type, and
/// a tool without arguments.
fn parser() -> XmlToolParser {
    XmlToolParser::new(&[
        function(
            "write_file",
            json!({
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
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
                    "label": {"type": ["integer", "null"]},
                    "target": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                },
            }),
        ),
        function("list_files", json!({"type": "object", "properties": {}})),
    ])
    .unwrap()
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

/// The calls of a coalesced output as `(index, name, arguments)`: every call's deltas
/// concatenate to one JSON object.
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

/// Whether `error` reports a malformed tag tool call.
fn is_malformed(error: &Error) -> bool {
    matches!(error, Error::ToolCall { format: "xml", .. })
}

#[test]
fn a_call_names_its_function_and_its_arguments() {
    let text = call(
        "write_file",
        &[("path", "notes/a.txt"), ("content", "hello")],
    );
    let output = parse_whole(&mut parser(), &text).unwrap();

    assert_eq!(output.normal_text(), "");
    assert_eq!(
        calls(&output),
        [(
            0,
            "write_file".to_owned(),
            json!({"path": "notes/a.txt", "content": "hello"})
        )]
    );
}

/// A value is its text when its schema accepts a string, else the JSON its text states.
#[test]
fn values_are_typed_by_the_tools_schema() {
    let text = call(
        "compute",
        &[
            ("count", "42"),
            ("ratio", "2.50"),
            ("exact", "true"),
            ("limits", r#"{"low": 1, "high": [2, 3]}"#),
            ("values", "[1, 2.5]"),
            ("label", "null"),
            ("target", r#"{"id": 7}"#),
            ("undeclared", "7"),
        ],
    );
    let output = parse_whole(&mut parser(), &text).unwrap();

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
                // A string is a reading the union accepts, so the text is kept.
                "target": "{\"id\": 7}",
                "undeclared": "7",
            })
        )]
    );
}

/// A value its schema accepts neither as text nor as JSON is a malformed call.
#[test]
fn a_value_its_schema_refuses_is_malformed() {
    for (name, value) in [
        ("count", "many"),
        ("count", "2.5"),
        ("exact", "True"),
        ("exact", "yes"),
        ("values", "[1, \"two\"]"),
        ("limits", "{'low': 1}"),
        ("label", "None"),
    ] {
        let text = call("compute", &[(name, value)]);
        let error = parse_whole(&mut parser(), &text).unwrap_err();
        assert!(is_malformed(&error), "{name}={value}: {error}");
    }
}

/// A value keeps its inner newlines, angle brackets and quotes: only the newline after the tag
/// and the one before the end tag are the format's.
#[test]
fn a_value_keeps_its_own_text() {
    let content = "fn main() {\n    if a < b && b > 0 {\n        println!(\"<ok>\");\n    }\n}\n";
    let text = call(
        "write_file",
        &[("path", "src/main.rs"), ("content", content)],
    );
    let output = parse_whole(&mut parser(), &text).unwrap();

    assert_eq!(
        calls(&output)[0].2,
        json!({"path": "src/main.rs", "content": content})
    );
}

/// A response holds several calls, in several regions or in one, and the text before, between
/// and after them stays text in its place.
#[test]
fn a_response_holds_several_calls_among_its_text() {
    let text = format!(
        "First the file.\n{}\nThen the listing.{}done",
        call("write_file", &[("path", "a"), ("content", "b")]),
        call("list_files", &[]),
    );
    let output = parse_whole(&mut parser(), &text).unwrap();

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

    let joined = "<tool_call>\n<function=list_files>\n</function>\n\
                  <function=compute>\n<parameter=count>\n3\n</parameter>\n</function>\n</tool_call>";
    let output = parse_whole(&mut parser(), joined).unwrap();
    assert_eq!(
        calls(&output),
        [
            (0, "list_files".to_owned(), json!({})),
            (1, "compute".to_owned(), json!({"count": 3})),
        ]
    );
}

/// A call streams its name at its function tag, then one fragment per argument once the
/// argument's value ended, then the object's end.
#[test]
fn a_call_streams_its_name_then_one_fragment_per_argument() {
    let mut parser = parser();
    let mut output = ToolParserOutput::default();
    parser
        .parse_into(
            "<tool_call>\n<function=compute>\n<parameter=count>\n4",
            &mut output,
        )
        .unwrap();
    assert_eq!(output.calls().count(), 1);
    assert_eq!(
        output.calls().next().unwrap().name.as_deref(),
        Some("compute")
    );

    parser
        .parse_into(
            "2\n</parameter>\n<parameter=exact>\nfalse\n</parameter>\n",
            &mut output,
        )
        .unwrap();
    parser
        .parse_into("</function>\n</tool_call>", &mut output)
        .unwrap();
    let fragments: Vec<&str> = output.calls().map(|call| call.arguments.as_str()).collect();
    assert_eq!(fragments, ["", "{\"count\":42", ",\"exact\":false", "}"]);
}

/// However the text is cut into chunks, tags included, the parser publishes the same calls
/// and text.
#[test]
fn chunks_of_any_size_give_the_whole_response() {
    let text = format!(
        "a < b, so:\n{}{}\nafter",
        call(
            "compute",
            &[
                ("count", "12"),
                ("limits", r#"{"low": 1}"#),
                ("target", "x<y")
            ]
        ),
        call(
            "write_file",
            &[("path", "p"), ("content", "line 1\nline 2")]
        ),
    );
    let whole = parse_whole(&mut parser(), &text).unwrap();
    assert_eq!(whole.normal_text(), "a < b, so:\n\nafter");

    for size in 1..=9 {
        let chunks = chunks_of(&text, size);
        let output = parse_chunks(&mut parser(), chunks.iter().map(String::as_str)).unwrap();
        assert_eq!(output, whole, "chunks of {size}");
    }
}

/// Text that breaks the grammar inside a region or a call is a malformed call.
#[test]
fn text_that_breaks_the_grammar_is_malformed() {
    for text in [
        // A parameter that never ends before the function does.
        "<tool_call>\n<function=compute>\n<parameter=count>\n4\n</function>\n</tool_call>",
        // Text between the tags of a region.
        "<tool_call>\nsure\n<function=list_files>\n</function>\n</tool_call>",
        // A parameter outside a call.
        "<tool_call>\n<parameter=count>\n4\n</parameter>\n</tool_call>",
        // A second function tag inside an open call.
        "<tool_call>\n<function=list_files>\n<function=compute>\n</function>\n</tool_call>",
        // A function tag without a name.
        "<tool_call>\n<function=>\n</function>\n</tool_call>",
    ] {
        let error = parse_whole(&mut parser(), text).unwrap_err();
        assert!(is_malformed(&error), "{text:?}: {error}");
    }
}

/// Text that ends inside a region is a malformed call; text that ends in a partial start
/// delimiter is text.
#[test]
fn text_that_ends_inside_a_region_is_malformed() {
    for text in [
        "<tool_call>\n<function=write_file>\n<parameter=content>\ncut",
        "<tool_call>\n<function=list_files>\n</function>\n",
        "<tool_call>",
    ] {
        let error = parse_whole(&mut parser(), text).unwrap_err();
        assert!(is_malformed(&error), "{text:?}: {error}");
    }

    let output = parse_whole(&mut parser(), "see <tool_").unwrap();
    assert_eq!(output.normal_text(), "see <tool_");
}

/// A `<` that opens no delimiter outside a region is text.
#[test]
fn text_that_opens_no_tag_stays_text() {
    let text = "if a <b and <tool_x> then <tool_call";
    let output = parse_whole(&mut parser(), text).unwrap();
    assert_eq!(output.normal_text(), text);
}

/// A successor's parser resumed from the state goes on where the handing parser stopped, at
/// every point of a response.
#[test]
fn a_handed_over_parser_goes_on_where_it_stopped() {
    let text = format!(
        "before {} after",
        call("compute", &[("count", "3"), ("exact", "true")])
    );
    let whole = parse_whole(&mut parser(), &text).unwrap();

    for cut in (0..=text.len()).filter(|cut| text.is_char_boundary(*cut)) {
        let mut first = parser();
        let mut output = ToolParserOutput::default();
        first.parse_into(&text[..cut], &mut output).unwrap();

        let mut second = parser();
        second.resume(&first.state()).unwrap();
        second.parse_into(&text[cut..], &mut output).unwrap();
        output.append(second.finish().unwrap());
        assert_eq!(output.coalesce_calls(), whole, "cut at {cut}");
    }
}

/// A tool whose parameter schema is not JSON Schema is refused when the parser is built.
#[test]
fn an_invalid_schema_is_refused() {
    let tools = [function(
        "f",
        json!({"type": "object", "properties": {"a": {"type": 7}}}),
    )];
    assert!(XmlToolParser::new(&tools).is_err());
}
