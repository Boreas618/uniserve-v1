//! Shared fixtures and stream collectors for tool-parser tests.

use super::{ToolParser, ToolParserOutput};

/// Push chunks through a streaming parser, finish it, and coalesce its
/// tool-call deltas.
///
/// Panics if there are any parsing errors along the way.
pub(crate) fn collect_stream(parser: &mut impl ToolParser, chunks: &[&str]) -> ToolParserOutput {
    let mut output = ToolParserOutput::default();
    for chunk in chunks {
        output.append(parser.parse_chunk(chunk).unwrap());
    }
    output.append(parser.finish().unwrap());
    output.coalesce_calls()
}

/// Splits text into chunks containing at most `chunk_chars` Unicode scalar
/// values.
pub(crate) fn split_by_chars(text: &str, chunk_chars: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut count = 0;

    for (index, _) in text.char_indices() {
        if count == chunk_chars {
            chunks.push(&text[start..index]);
            start = index;
            count = 0;
        }
        count += 1;
    }

    if start < text.len() {
        chunks.push(&text[start..]);
    }

    chunks
}
