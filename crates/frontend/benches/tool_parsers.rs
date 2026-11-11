//! Throughput benchmarks for incremental JSON tool-call parsing.
//!
//! Each fixture is split into fixed-size character chunks and streamed
//! through `JsonToolParser` incrementally, as the chat output stage feeds
//! assistant text. Throughput is reported in input bytes. The bench drives
//! the parser through its public interface (`ToolParser`) only.
//!
//! The expected-output checks inside each iteration are `debug_assert`s: they
//! run only in builds with debug assertions and are compiled out of the
//! release-profile build that `cargo bench` times.

use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use frontend::output::tools::{JsonToolParser, ToolParser, ToolParserOutput};

/// Unicode scalar values per streamed chunk.
const CHUNK_CHARS: usize = 7;
const LONG_NORMAL_TEXT_REPEATS: usize = 2048;

/// Splits text into chunks of at most `chunk_chars` Unicode scalar values.
fn split_by_chars(text: &str, chunk_chars: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;
    for (count, (index, _)) in text.char_indices().enumerate() {
        if count > 0 && count % chunk_chars == 0 {
            chunks.push(&text[start..index]);
            start = index;
        }
    }
    if start < text.len() {
        chunks.push(&text[start..]);
    }
    chunks
}

fn tool_call(function_name: &str, arguments: &str) -> String {
    format!("<tool_call>\n{{\"name\":\"{function_name}\",\"arguments\":{arguments}}}\n</tool_call>")
}

fn mixed_fixture() -> String {
    format!(
        "I will check two cities before answering.\n{}{}",
        tool_call("get_weather", r#"{"location":"Hangzhou","days":3}"#),
        tool_call("get_weather", r#"{"location":"San Francisco","days":2}"#),
    )
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no tool-call markers at all.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

/// Feeds chunks into a parser and returns visible text plus parsed-call count.
///
/// The stream is finished after the last chunk, which resets `parser` for the
/// next stream. A parser error panics: every fixture is well formed.
#[expect(
    clippy::unwrap_used,
    reason = "a parser failure on a well-formed fixture invalidates the bench"
)]
fn feed_parser(parser: &mut JsonToolParser, chunks: &[&str]) -> (String, usize) {
    let mut output = ToolParserOutput::default();
    for chunk in chunks {
        parser.parse_into(chunk, &mut output).unwrap();
    }
    output.append(parser.finish().unwrap());
    let output = output.coalesce_calls();
    (output.normal_text(), output.calls().count())
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    text: &str,
    chunk_chars: usize,
    expected_normal_text: &str,
    expected_native_calls_len: usize,
) {
    let chunks = split_by_chars(text, chunk_chars);

    let mut group = c.benchmark_group(name);
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(text.len() as u64));

    // One parser serves every iteration; this relies on `finish`, called at
    // the end of each stream by `feed_parser`, resetting parser state.
    group.bench_function("native_reuse_parser", |b| {
        let mut parser = JsonToolParser::new();
        b.iter(|| {
            let result = feed_parser(&mut parser, black_box(&chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            debug_assert_eq!(result.1, expected_native_calls_len);
            black_box(result);
        })
    });

    // Each iteration parses with a freshly constructed parser. Construction
    // runs in the untimed `iter_batched` setup closure.
    group.bench_function("native_create_parser", |b| {
        b.iter_batched(
            JsonToolParser::new,
            |mut parser| {
                let result = feed_parser(&mut parser, black_box(&chunks));
                debug_assert_eq!(result.0, expected_normal_text);
                debug_assert_eq!(result.1, expected_native_calls_len);
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_json_tool_parser(c: &mut Criterion) {
    let mixed_text = mixed_fixture();
    let long_normal_text = long_normal_text_fixture();

    run_stream_group(
        c,
        "json_tool_parser/mixed_text_tool_call",
        &mixed_text,
        CHUNK_CHARS,
        "I will check two cities before answering.\n",
        2,
    );

    run_stream_group(
        c,
        "json_tool_parser/long_normal_text",
        &long_normal_text,
        CHUNK_CHARS,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_json_tool_parser);
criterion_main!(benches);
