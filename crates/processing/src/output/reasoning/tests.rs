//! Incremental reasoning-parser behavior across delimiter boundaries.

use super::DelimitedReasoningParser;
use crate::output::Delimiters;
use crate::serving::{ReasoningFormat, ToolCallFormat};

/// The id of `<think>` in the tests' vocabulary.
const START: u32 = 5;

/// The id of `</think>`.
const END: u32 = 6;

/// The id of `<|im_end|>`, a special token that is no delimiter.
const PROMPT_END: u32 = 2;

/// The parser of `<think>` reasoning, which waits for an explicit start delimiter when the
/// prompt states no reasoning boundary.
fn think_parser() -> DelimitedReasoningParser {
    DelimitedReasoningParser::new(THINK, None)
}

/// `<think>` reasoning with the ids of the tests' vocabulary.
const THINK: Delimiters = Delimiters {
    format: ReasoningFormat {
        open: "<think>",
        close: "</think>",
    },
    open: START,
    close: END,
};

#[test]
fn delimited_parser_splits_reasoning_across_chunk_boundaries() {
    let mut parser = think_parser();

    // A partial start delimiter is held back until the next delta completes it.
    assert!(parser.push("<thi").is_empty());
    let delta = parser.push("nk>reason</think>answer");
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn delimited_parser_flushes_partial_end_marker() {
    let mut parser = think_parser();
    parser.initialize(&[START]);

    let delta = parser.push("unfinished</thi");
    assert_eq!(delta.reasoning.as_deref(), Some("unfinished"));

    // At end of stream the incomplete delimiter is text of the current region.
    assert_eq!(parser.finish().reasoning.as_deref(), Some("</thi"));
}

#[test]
fn without_a_prompt_boundary_reasoning_waits_for_its_start() {
    let mut parser = think_parser();

    let delta = parser.push("reason</think>answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("reason</think>answer"));
}

#[test]
fn prompt_boundaries_select_the_initial_output_region() {
    let mut content = think_parser();
    content.initialize(&[END]);
    assert_eq!(content.push("answer").content.as_deref(), Some("answer"));

    let mut reasoning = think_parser();
    reasoning.initialize(&[START]);
    let delta = reasoning.push("reason</think>answer");
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

/// A tail from a special token that is no delimiter on holds no boundary: a `<think>` before it
/// belongs to an earlier turn, which the tail leaves out, so the parser keeps its content
/// region.
#[test]
fn a_tail_without_a_delimiter_starts_in_content() {
    let mut parser = think_parser();

    parser.initialize(&[PROMPT_END, u32::from(b'a')]);
    let delta = parser.push("answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

/// The parser of `<think>` reasoning for a family whose DSML calls end the turn.
fn dsml_parser() -> DelimitedReasoningParser {
    DelimitedReasoningParser::new(THINK, ToolCallFormat::Dsml.final_region())
}

/// The final region's opening ends unclosed reasoning, the separator before it going to the
/// content with it, and no delimiter is read after it, at every split of the text.
#[test]
fn the_final_region_ends_reasoning_and_its_delimiters() {
    let text = "reason\n\n<｜DSML｜ calls><think>arg</think>";
    for (at, _) in text.char_indices().skip(1) {
        let mut parser = dsml_parser();
        parser.initialize(&[START]);
        let mut delta = parser.push(&text[..at]);
        for part in [parser.push(&text[at..]), parser.finish()] {
            delta.push_reasoning(part.reasoning.as_deref().unwrap_or_default());
            delta.push_content(part.content.as_deref().unwrap_or_default());
        }
        assert_eq!(delta.reasoning.as_deref(), Some("reason"), "split at {at}");
        assert_eq!(
            delta.content.as_deref(),
            Some("\n\n<｜DSML｜ calls><think>arg</think>"),
            "split at {at}"
        );
    }

    // Opened in the text, the region keeps its thinking tags as text too.
    let mut parser = dsml_parser();
    parser.initialize(&[END]);
    let delta = parser.push("answer<｜DSML｜ calls><think>x");
    assert_eq!(delta.reasoning, None);
    assert_eq!(
        delta.content.as_deref(),
        Some("answer<｜DSML｜ calls><think>x")
    );
}
