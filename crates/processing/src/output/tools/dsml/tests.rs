//! The DSML format's grammar, value typing, streaming, handoff and cut-off calls. The expected
//! arguments are the texts vLLM's DeepSeek-V4.1 parser (`vllm/parser/deepseek_v41.py`) writes
//! for the same completions, except where a test states otherwise.

use super::DsmlToolParser;
use crate::output::tools::{ToolParser, ToolParserItem, ToolParserOutput};
use crate::testing::tool_parsing::{chunks_of, parse_chunks, parse_whole};

/// A `calls` block of a text argument holding quotes, a newline and thinking tags, a number, an
/// object of every JSON type, and a call without arguments.
const BLOCK: &str = concat!(
    "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n",
    "<｜DSML｜ parameter name=\"path\" string=\"true\">/srv/a \"b\"\n<think>x</think>",
    "</｜DSML｜ parameter>\n",
    "<｜DSML｜ parameter name=\"limit\" string=\"false\">3</｜DSML｜ parameter>\n",
    "<｜DSML｜ parameter name=\"options\" string=\"false\">",
    "{\"deep\": true, \"ratio\": 1.5e-07, \"tags\": [\"x\", null]}</｜DSML｜ parameter>\n",
    "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"list\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
);

/// The arguments of [`BLOCK`]'s `read` call.
const READ_ARGUMENTS: &str = concat!(
    "{\"path\": \"/srv/a \\\"b\\\"\\n<think>x</think>\", \"limit\": 3, ",
    "\"options\": {\"deep\": true, \"ratio\": 1.5e-07, \"tags\": [\"x\", null]}}",
);

/// An output's items, each call as `(name, arguments)`: the visible text and the calls in order.
fn items(output: &ToolParserOutput) -> (String, Vec<(String, String)>) {
    let calls = output
        .calls()
        .map(|call| {
            let name = call
                .name
                .clone()
                .expect("a coalesced call names its function");
            (name, call.arguments.clone())
        })
        .collect();
    (output.normal_text(), calls)
}

/// The text and calls `text` parses into in one chunk.
fn parsed(text: &str) -> (String, Vec<(String, String)>) {
    items(&parse_whole(&mut DsmlToolParser::new(), text).unwrap())
}

fn call(name: &str, arguments: &str) -> (String, String) {
    (name.to_owned(), arguments.to_owned())
}

/// Text before the block is text, the blank line before the block excepted; each call is named
/// once and its arguments are the object vLLM writes, text values as they are and JSON values
/// as Python's `json.dumps` writes them.
#[test]
fn a_block_parses_into_its_calls() {
    let (text, calls) = parsed(&format!("Reading it now.{BLOCK}"));
    assert_eq!(text, "Reading it now.");
    assert_eq!(calls, [call("read", READ_ARGUMENTS), call("list", "{}")]);
}

/// Split at every character boundary, and streamed a character at a time, the text parses into
/// the same text and calls; every call's increments concatenate to its arguments.
#[test]
fn streamed_text_parses_alike_at_every_split() {
    let text = format!("Reading it now.{BLOCK}");
    let whole = parsed(&text);
    for (at, _) in text.char_indices().skip(1) {
        let output = parse_chunks(&mut DsmlToolParser::new(), [&text[..at], &text[at..]]).unwrap();
        assert_eq!(items(&output), whole, "split at {at}");
    }
    let chunks = chunks_of(&text, 1);
    let output = parse_chunks(
        &mut DsmlToolParser::new(),
        chunks.iter().map(String::as_str),
    )
    .unwrap();
    assert_eq!(items(&output), whole);
}

/// A text value streams as its text arrives, before its closing tag, and a JSON value is
/// published once it ended.
#[test]
fn a_text_value_streams_before_its_end() {
    let mut parser = DsmlToolParser::new();
    let mut output = ToolParserOutput::default();
    parser
        .parse_into(
            "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n\
             <｜DSML｜ parameter name=\"content\" string=\"true\">line one\nline",
            &mut output,
        )
        .unwrap();
    let streamed: String = output.calls().map(|call| call.arguments.as_str()).collect();
    assert_eq!(streamed, "{\"content\": \"line one\\nline");

    let mut output = ToolParserOutput::default();
    parser
        .parse_into(
            " two</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"n\" string=\"false\">1",
            &mut output,
        )
        .unwrap();
    let streamed: String = output.calls().map(|call| call.arguments.as_str()).collect();
    assert_eq!(streamed, " two\"", "the JSON value waits for its end");
}

/// The parser's state, saved at any point and restored into a fresh parser, goes on where it
/// left off.
#[test]
fn a_restored_parser_goes_on_where_it_stopped() {
    let text = format!("Reading it now.{BLOCK}");
    let whole = parsed(&text);
    for (at, _) in text.char_indices().skip(1) {
        let mut first = DsmlToolParser::new();
        let mut output = ToolParserOutput::default();
        first.parse_into(&text[..at], &mut output).unwrap();

        let saved = serde_json::to_string(&first.state()).unwrap();
        let mut second = DsmlToolParser::new();
        second
            .resume(serde_json::from_str(&saved).unwrap())
            .unwrap();
        second.parse_into(&text[at..], &mut output).unwrap();
        output.append(second.finish().unwrap());
        assert_eq!(items(&output.coalesce_calls()), whole, "handed off at {at}");
    }
}

/// The block ends the turn: everything after it is dropped, a later block too (vLLM would parse
/// a later block's calls).
#[test]
fn text_after_the_block_is_dropped() {
    let text = format!("c{BLOCK}<｜User｜>more<｜Assistant｜><think>again</think>x{BLOCK}trailing");
    let (text, calls) = parsed(&text);
    assert_eq!(text, "c");
    assert_eq!(calls, [call("read", READ_ARGUMENTS), call("list", "{}")]);
}

/// A call cut by the end of generation is closed as far as the text reached, and nothing is
/// published of a call whose `invoke` tag did not end: a text value is closed where the text
/// ended (vLLM drops it when it parses the whole completion, and leaves its JSON open when it
/// streams it), a JSON value that did not end is dropped, as is a parameter tag that did not end
/// (as vLLM drops them), and a call whose name did not end is not published (vLLM publishes the
/// name's prefix).
#[test]
fn a_cut_call_is_closed_where_the_text_ended() {
    let open = "c\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n";
    let cases = [
        (
            format!("{open}<｜DSML｜ parameter name=\"path\" string=\"true\">/srv/partial"),
            vec![call("read", "{\"path\": \"/srv/partial\"}")],
        ),
        (
            format!(
                "{open}<｜DSML｜ parameter name=\"path\" string=\"true\">/a</｜DSML｜ parameter>\n\
                 <｜DSML｜ parameter name=\"options\" string=\"false\">{{\"deep\": tr"
            ),
            vec![call("read", "{\"path\": \"/a\"}")],
        ),
        (
            format!("{open}<｜DSML｜ parameter name=\"pa"),
            vec![call("read", "{}")],
        ),
        (
            // A partial closing tag of a text value is not the value's text.
            format!("{open}<｜DSML｜ parameter name=\"path\" string=\"true\">/b</｜DSML｜ par"),
            vec![call("read", "{\"path\": \"/b\"}")],
        ),
        (
            "c\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"rea".to_owned(),
            vec![],
        ),
        ("c\n\n<｜DSML｜ calls>\n<｜DSML｜ inv".to_owned(), vec![]),
    ];
    for (text, expected) in cases {
        for (at, _) in text.char_indices().skip(1) {
            let output =
                parse_chunks(&mut DsmlToolParser::new(), [&text[..at], &text[at..]]).unwrap();
            assert_eq!(
                items(&output),
                ("c".to_owned(), expected.clone()),
                "{text:?} at {at}"
            );
        }
    }
}

/// Parameter tags are read as vLLM reads them: the `string` attribute may be absent (a value is
/// then JSON, or its text when it is no JSON), a value ends at the next parameter tag or its
/// call's end, and text that is no tag, a parameter tag that does not parse and a call without a
/// name are ignored.
#[test]
fn lenient_tags_parse_as_vllm_parses_them() {
    let cases = [
        (
            "</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n\
             <｜DSML｜ parameter name=\"limit\">7</｜DSML｜ parameter>\n\
             <｜DSML｜ parameter name=\"path\">not json</｜DSML｜ parameter>\n\
             </｜DSML｜ invoke>\n</｜DSML｜ calls>",
            vec![call("read", "{\"limit\": 7, \"path\": \"not json\"}")],
        ),
        (
            "</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n\
             <｜DSML｜ parameter name=\"path\" string=\"true\">/a\n\
             <｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n\
             </｜DSML｜ invoke>\n</｜DSML｜ calls>",
            vec![call("read", "{\"path\": \"/a\\n\", \"limit\": 2}")],
        ),
        (
            "</think>\n\n<｜DSML｜ calls>\nstray\n<｜DSML｜ invoke name=\"\">\n\
             <｜DSML｜ parameter name=\"x\" string=\"true\">y</｜DSML｜ parameter>\n\
             </｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"read\">\n\
             junk<｜DSML｜ parameter oops>v</｜DSML｜ parameter>\n\
             <｜DSML｜ parameter name=\"path\" string=\"true\">/b</｜DSML｜ parameter>\n\
             </｜DSML｜ invoke>\n</｜DSML｜ calls>",
            vec![call("read", "{\"path\": \"/b\"}")],
        ),
        (
            // A value ends at its call's end, as written (vLLM drops a value without its closing
            // tag).
            "</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n\
             <｜DSML｜ parameter name=\"path\" string=\"true\">/c</｜DSML｜ invoke>\n\
             </｜DSML｜ calls>",
            vec![call("read", "{\"path\": \"/c\"}")],
        ),
    ];
    for (text, expected) in cases {
        let whole = parsed(text);
        assert_eq!(whole, ("</think>".to_owned(), expected), "{text:?}");
        for (at, _) in text.char_indices().skip(1) {
            let output =
                parse_chunks(&mut DsmlToolParser::new(), [&text[..at], &text[at..]]).unwrap();
            assert_eq!(items(&output), whole, "{text:?} at {at}");
        }
    }
}

/// Text without a block is text, a trailing blank line or partial opening released at the end;
/// a block written without the blank line before it is a block too.
#[test]
fn text_without_a_block_is_text() {
    let mut parser = DsmlToolParser::new();
    let mut output = ToolParserOutput::default();
    parser
        .parse_into("plain answer\n\n<｜DS", &mut output)
        .unwrap();
    assert_eq!(
        output.items,
        [ToolParserItem::Text("plain answer".to_owned())]
    );
    let finished = parser.finish().unwrap();
    assert_eq!(
        finished.items,
        [ToolParserItem::Text("\n\n<｜DS".to_owned())]
    );

    let (text, calls) = parsed(
        "a<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"list\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    );
    assert_eq!(text, "a");
    assert_eq!(calls, [call("list", "{}")]);
}
