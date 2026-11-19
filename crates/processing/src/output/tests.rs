//! Reasoning and tool-call parsing of a response's text across stream boundaries, in each
//! tool-call format, and across a handoff of the parser's state.

use scheduling::processing::Event;

use super::{Delimiters, Error, Options, Parser};
use crate::serving::{ReasoningFormat, ToolCallFormat};
use crate::testing::chat;

/// `<think>` reasoning, whose delimiters are ids 5 and 6.
fn think() -> Delimiters {
    Delimiters {
        format: ReasoningFormat {
            open: "<think>",
            close: "</think>",
        },
        open: 5,
        close: 6,
    }
}

/// The id of `<|im_end|>`, a special token that is no delimiter.
const PROMPT_END: u32 = 2;

/// The parsers of a response in `format` whose prompt's tail is `prompt`: reasoning when
/// `reasoning` is set, tool calls against `tools` when the request lets the model call them.
fn parser(
    format: ToolCallFormat,
    reasoning: bool,
    tools: Option<Vec<scheduling::processing::Tool>>,
    prompt: &[u32],
) -> Parser {
    Parser::new(
        Options { reasoning, tools },
        Ok(Some(think())),
        Some(format),
        prompt,
    )
    .unwrap()
}

/// Runs `parser` over `deltas`, then finishes it.
fn parse(mut parser: Parser, deltas: &[&str]) -> Result<Vec<Event>, Error> {
    let mut events = Vec::new();
    for delta in deltas {
        events.extend(parser.push(delta)?);
    }
    events.extend(parser.finish()?);
    Ok(events)
}

/// Output in stream order, with adjacent text events of one kind merged and each call's
/// argument increments joined, so the comparison does not depend on how deltas are split.
#[derive(Debug, PartialEq, Eq)]
enum Ordered {
    Reasoning(String),
    Text(String),
    Call { name: String, arguments: String },
}

/// Folds events into their ordered output.
fn ordered(events: Vec<Event>) -> Vec<Ordered> {
    let mut ordered = Vec::<Ordered>::new();
    for event in events {
        match (event, ordered.last_mut()) {
            (Event::Reasoning(delta), Some(Ordered::Reasoning(text)))
            | (Event::Content(delta), Some(Ordered::Text(text))) => text.push_str(&delta),
            (Event::Reasoning(delta), _) => ordered.push(Ordered::Reasoning(delta)),
            (Event::Content(delta), _) => ordered.push(Ordered::Text(delta)),
            (Event::ToolCall { name }, _) => ordered.push(Ordered::Call {
                name,
                arguments: String::new(),
            }),
            (Event::Arguments(delta), Some(Ordered::Call { arguments, .. })) => {
                arguments.push_str(&delta);
            }
            (Event::Arguments(delta), _) => panic!("arguments {delta:?} follow no tool call"),
        }
    }
    ordered
}

fn text(value: &str) -> Ordered {
    Ordered::Text(value.to_owned())
}

fn lookup(q: &str) -> Ordered {
    Ordered::Call {
        name: "lookup".to_owned(),
        arguments: format!(r#"{{"q":"{q}"}}"#),
    }
}

/// The `lookup` tool, whose `q` is a string and `limit` an integer.
fn lookup_tool() -> scheduling::processing::Tool {
    chat::function(
        "lookup",
        serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
        }),
    )
}

/// One decoded delta holding a reasoning section, text and two calls separates into the
/// reasoning, the visible text with the calls' markup removed, and one call per region.
#[test]
fn reasoning_text_and_json_calls_separate_in_order() {
    // The prompt's tail ends at `<|im_end|>` with no delimiter, so generation starts outside a
    // reasoning section and the `<think>` in the output opens one.
    let parser = parser(ToolCallFormat::Json, true, Some(Vec::new()), &[PROMPT_END]);
    let events = parse(
        parser,
        &[concat!(
            "<think>hidden</think>before",
            "<tool_call>\n",
            r#"{"name":"lookup","arguments":{"q":"x"}}"#,
            "\n</tool_call>",
            "between",
            "<tool_call>\n",
            r#"{"name":"lookup","arguments":{"q":"y"}}"#,
            "\n</tool_call>",
            "after"
        )],
    )
    .unwrap();

    assert_eq!(
        ordered(events),
        vec![
            Ordered::Reasoning("hidden".to_owned()),
            text("before"),
            lookup("x"),
            text("between"),
            lookup("y"),
            text("after"),
        ]
    );
}

/// Text and calls keep their stream order when one delta spans call boundaries, both from
/// outside a call and from inside a call's arguments.
#[test]
fn text_and_json_calls_keep_their_order_within_a_delta() {
    let parser = parser(ToolCallFormat::Json, false, Some(Vec::new()), &[]);
    let events = parse(
        parser,
        &[
            concat!(
                "before",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"x"}}"#,
                "\n</tool_call>",
                "between",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"#,
            ),
            concat!(
                r#""y"}}"#,
                "\n</tool_call>",
                "after",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"z"}}"#,
                "\n</tool_call>",
            ),
        ],
    )
    .unwrap();

    assert_eq!(
        ordered(events),
        vec![
            text("before"),
            lookup("x"),
            text("between"),
            lookup("y"),
            text("after"),
            lookup("z"),
        ]
    );
}

/// A malformed call fails the response instead of reaching the client as text.
#[test]
fn a_malformed_tool_call_fails_the_output() {
    let parser = parser(ToolCallFormat::Json, false, Some(Vec::new()), &[]);
    let error = parse(
        parser,
        &["before<tool_call>\n{\"name\":\"lookup\",\"arguments\":{'q': 1}}\n</tool_call>"],
    )
    .expect_err("a malformed call is no text");

    assert!(
        matches!(error, Error::ToolCall { format: "json", .. }),
        "{error}"
    );
    assert!(
        error.report().contains("malformed json tool call"),
        "{}",
        error.report()
    );
}

/// Without reasoning parsing, `<think>` delimiters stream verbatim as content, also when they
/// arrive in separate deltas; without callable tools, call markup is text.
#[test]
fn unparsed_formats_stream_as_content() {
    let parser = parser(ToolCallFormat::Json, false, None, &[]);
    let events = parse(
        parser,
        &["<think>", "hi</think>after<tool_call>\n{}\n</tool_call>"],
    )
    .unwrap();

    assert_eq!(
        ordered(events),
        vec![text("<think>hi</think>after<tool_call>\n{}\n</tool_call>")]
    );
}

/// Function and parameter tags become calls, each value typed by the schema of the tool the
/// request declared, after the reasoning and among the text.
#[test]
fn function_and_parameter_tags_become_typed_tool_calls() {
    let parser = parser(ToolCallFormat::Xml, true, Some(vec![lookup_tool()]), &[]);

    // The deltas cut the reasoning delimiter, a tag and a value.
    let events = parse(
        parser,
        &[
            "<think>look it up</think>Searching.<tool_call>\n<func",
            "tion=lookup>\n<parameter=q>\nrust tra",
            "its\n</parameter>\n<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=lookup>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>done",
        ],
    )
    .unwrap();

    assert_eq!(
        ordered(events),
        vec![
            Ordered::Reasoning("look it up".to_owned()),
            text("Searching."),
            Ordered::Call {
                name: "lookup".to_owned(),
                arguments: r#"{"q":"rust traits","limit":3}"#.to_owned(),
            },
            Ordered::Call {
                name: "lookup".to_owned(),
                arguments: r#"{"q":"x"}"#.to_owned(),
            },
            text("done"),
        ]
    );
}

/// A parser restored from a saved state inside reasoning, a delimiter and a call goes on
/// where the saved one stood: together the two produce what one parser would.
#[test]
fn a_restored_parser_goes_on_where_the_saved_one_stood() {
    let deltas = [
        "<think>a</th",
        "ink>b<tool_call>\n<function=lookup>\n<parameter=q>\nx",
        "y\n</parameter>\n</function>\n</tool_call>c",
    ];
    let whole = ordered(
        parse(
            parser(ToolCallFormat::Xml, true, Some(vec![lookup_tool()]), &[]),
            &deltas,
        )
        .unwrap(),
    );

    for cut in 1..deltas.len() {
        let mut first = parser(ToolCallFormat::Xml, true, Some(vec![lookup_tool()]), &[]);
        let mut events = Vec::new();
        for delta in &deltas[..cut] {
            events.extend(first.push(delta).unwrap());
        }
        // The state crosses as the bytes a handoff carries.
        let bytes = serde_json::to_vec(&first.save()).unwrap();
        let mut second = Parser::restore(
            serde_json::from_slice(&bytes).unwrap(),
            Ok(Some(think())),
            Some(ToolCallFormat::Xml),
        )
        .unwrap();
        for delta in &deltas[cut..] {
            events.extend(second.push(delta).unwrap());
        }
        events.extend(second.finish().unwrap());
        assert_eq!(ordered(events), whole, "cut after delta {cut}");
    }
}

/// A family whose tokenizer lacks a delimiter as one token fails the responses that parse its
/// reasoning, naming the token, and serves those that do not.
#[test]
fn a_missing_delimiter_fails_the_responses_that_parse_reasoning() {
    let missing = || {
        Err(Error::Parser {
            kind: "reasoning",
            name: "<think> </think>".to_owned(),
            message: "the vocabulary has no token `<think>`".to_owned(),
        })
    };
    let refused = Parser::new(
        Options {
            reasoning: true,
            tools: None,
        },
        missing(),
        None,
        &[],
    );
    assert!(matches!(
        refused,
        Err(Error::Parser {
            kind: "reasoning",
            ..
        })
    ));

    let served = Parser::new(Options::default(), missing(), None, &[]);
    assert!(served.is_ok());
}

/// A DeepSeek-V4.1 completion in thinking mode: reasoning, the turn's text, and a DSML block
/// whose text argument holds thinking tags, after which the model ran on (as a request that
/// ignores the end-of-sequence token lets it).
const DSML_COMPLETION: &str = concat!(
    "I should read it.</think>Reading.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n",
    "<｜DSML｜ parameter name=\"path\" string=\"true\">a <think>b</think> c</｜DSML｜ parameter>\n",
    "<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n",
    "</｜DSML｜ invoke>\n</｜DSML｜ calls><｜User｜>next<｜Assistant｜><think>more</think>text",
);

/// The output of [`DSML_COMPLETION`]: its reasoning, its text without the blank line before the
/// block, and its call, whose argument keeps the thinking tags; what followed the block is
/// dropped.
fn dsml_output() -> Vec<Ordered> {
    vec![
        Ordered::Reasoning("I should read it.".to_owned()),
        text("Reading."),
        Ordered::Call {
            name: "read".to_owned(),
            arguments: r#"{"path": "a <think>b</think> c", "limit": 2}"#.to_owned(),
        },
    ]
}

/// A thinking-mode prompt's tail, which ends with `<think>`: generation starts inside reasoning.
const THINKING_TAIL: [u32; 1] = [5];

/// A DSML completion parses into its reasoning, text and calls at every split of its text, and
/// a parser handed off at any point goes on where it stood.
#[test]
fn a_dsml_completion_parses_alike_at_every_split_and_handoff() {
    let parse_split = |at: usize| {
        let parser = parser(ToolCallFormat::Dsml, true, Some(Vec::new()), &THINKING_TAIL);
        ordered(parse(parser, &[&DSML_COMPLETION[..at], &DSML_COMPLETION[at..]]).unwrap())
    };
    assert_eq!(parse_split(DSML_COMPLETION.len()), dsml_output());

    for (at, _) in DSML_COMPLETION.char_indices().skip(1) {
        assert_eq!(parse_split(at), dsml_output(), "split at {at}");

        let mut first = parser(ToolCallFormat::Dsml, true, Some(Vec::new()), &THINKING_TAIL);
        let mut events = first.push(&DSML_COMPLETION[..at]).unwrap();
        // The state crosses as the bytes a handoff carries.
        let bytes = serde_json::to_vec(&first.save()).unwrap();
        let mut second = Parser::restore(
            serde_json::from_slice(&bytes).unwrap(),
            Ok(Some(think())),
            Some(ToolCallFormat::Dsml),
        )
        .unwrap();
        events.extend(second.push(&DSML_COMPLETION[at..]).unwrap());
        events.extend(second.finish().unwrap());
        assert_eq!(ordered(events), dsml_output(), "handed off at {at}");
    }
}

/// A DSML block opened inside reasoning ends the reasoning, the blank line before it belonging
/// to neither; without callable tools the block is text, and reasoning still ends at it.
#[test]
fn a_dsml_block_ends_unclosed_reasoning() {
    let completion = concat!(
        "thinking\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"list\">\n",
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    );
    let called = parse(
        parser(ToolCallFormat::Dsml, true, Some(Vec::new()), &THINKING_TAIL),
        &[completion],
    )
    .unwrap();
    assert_eq!(
        ordered(called),
        [
            Ordered::Reasoning("thinking".to_owned()),
            Ordered::Call {
                name: "list".to_owned(),
                arguments: "{}".to_owned(),
            },
        ]
    );

    let uncalled = parse(
        parser(ToolCallFormat::Dsml, true, None, &THINKING_TAIL),
        &[completion],
    )
    .unwrap();
    assert_eq!(
        ordered(uncalled),
        [
            Ordered::Reasoning("thinking".to_owned()),
            text(&completion["thinking".len()..]),
        ]
    );
}

/// A DSML call cut by the end of generation reaches the response closed, never as a failure.
/// In chat mode, whose prompt ends with `</think>`, generation starts in the text.
#[test]
fn a_cut_dsml_call_does_not_fail_the_response() {
    let cut = concat!(
        "Writing.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n",
        "<｜DSML｜ parameter name=\"content\" string=\"true\">first line\nsecond",
    );
    let events = parse(
        parser(ToolCallFormat::Dsml, true, Some(Vec::new()), &[6]),
        &[cut],
    )
    .unwrap();
    assert_eq!(
        ordered(events),
        [
            text("Writing."),
            Ordered::Call {
                name: "write".to_owned(),
                arguments: r#"{"content": "first line\nsecond"}"#.to_owned(),
            },
        ]
    );
}
