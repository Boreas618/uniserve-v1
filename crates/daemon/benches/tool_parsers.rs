//! Throughput of incremental JSON tool-call parsing.
//!
//! Each fixture is cut into chunks of a fixed number of characters and streamed through a new
//! `JsonToolParser`, as the chat output stage feeds a request's decoded text to the parser it
//! builds for the request; throughput is reported in input bytes. The chunk sizes are the
//! protocol's parameter: one character (the most calls per byte), four (about one token of
//! English text) and sixty-four (several tokens decoded in one step). Criterion's default sample
//! size, warm-up and measurement time apply. The fixtures are written in the format's own
//! delimiters (`ToolCallFormat::delimiters`).
//!
//! The expected-output checks inside each iteration are `debug_assert`s: they run only in builds
//! with debug assertions and are compiled out of the release-profile build `cargo bench` times.

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use daemon::api::output::tools::{JsonToolParser, ToolParser, ToolParserOutput};
use itertools::Itertools as _;
use processing::ToolCallFormat;

/// Characters per streamed chunk, each a point of the benchmark.
const CHUNK_CHARS: [usize; 3] = [1, 4, 64];

/// Lines of the text-only fixture: enough text that per-stream setup is negligible.
const TEXT_LINES: usize = 2048;

/// One call of `name` with the JSON object `arguments`, in the format's delimiters.
fn tool_call(name: &str, arguments: &str) -> String {
    let (start, end) = ToolCallFormat::Json.delimiters();
    format!("{start}{{\"name\":\"{name}\",\"arguments\":{arguments}}}{end}")
}

/// Text followed by two calls.
fn mixed_fixture() -> String {
    format!(
        "I will check two cities before answering.\n{}{}",
        tool_call("get_weather", r#"{"location":"Hangzhou","days":3}"#),
        tool_call("get_weather", r#"{"location":"San Francisco","days":2}"#),
    )
}

/// Text without a call.
fn text_fixture() -> String {
    "This is ordinary assistant text with no tool-call markers at all.\n".repeat(TEXT_LINES)
}

/// The visible text and the number of calls a new parser reads from `chunks`.
#[expect(
    clippy::unwrap_used,
    reason = "a parser failure on a well-formed fixture invalidates the bench"
)]
fn parse(chunks: &[String]) -> (String, usize) {
    let mut parser = JsonToolParser::new();
    let mut output = ToolParserOutput::default();
    for chunk in chunks {
        parser.parse_into(chunk, &mut output).unwrap();
    }
    output.append(parser.finish().unwrap());
    let output = output.coalesce_calls();
    (output.normal_text(), output.calls().count())
}

/// One group of the fixture `text`, whose visible text is `text_out` and which holds `calls`
/// calls, at every chunk size.
fn stream_group(c: &mut Criterion, name: &str, text: &str, text_out: &str, calls: usize) {
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes(text.len() as u64));
    for size in CHUNK_CHARS {
        let chunks: Vec<String> = text
            .chars()
            .chunks(size)
            .into_iter()
            .map(String::from_iter)
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &chunks, |b, chunks| {
            b.iter_batched(
                || chunks.clone(),
                |chunks| {
                    let result = parse(black_box(&chunks));
                    debug_assert_eq!(result.0, text_out);
                    debug_assert_eq!(result.1, calls);
                    black_box(result);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_json_tool_parser(c: &mut Criterion) {
    let mixed = mixed_fixture();
    stream_group(
        c,
        "json_tool_parser/text_and_calls",
        &mixed,
        "I will check two cities before answering.\n",
        2,
    );

    let text = text_fixture();
    stream_group(c, "json_tool_parser/text", &text, &text, 0);
}

criterion_group!(benches, bench_json_tool_parser);
criterion_main!(benches);
