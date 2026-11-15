//! Incremental reasoning-parser behavior across delimiter boundaries.

use std::sync::Arc;

use crate::api::test_support::configured_tokenizer as reasoning_tokenizer;
use crate::api::tokenizer::DynTokenizer;

use super::DelimitedReasoningParser;

/// The parser of `<think>` reasoning, which waits for an explicit start
/// delimiter when the prompt states no reasoning boundary.
fn think_parser(tokenizer: DynTokenizer) -> DelimitedReasoningParser {
    DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap()
}

fn token_id(tokenizer: &DynTokenizer, token: &str) -> u32 {
    tokenizer
        .token_to_id(token)
        .unwrap_or_else(|| panic!("tokenizer must contain {token}"))
}

#[test]
fn delimited_parser_splits_reasoning_across_chunk_boundaries() {
    let tokenizer = reasoning_tokenizer();
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    // A partial start delimiter is held back until the next delta completes it.
    assert!(parser.push("<thi").is_empty());
    let delta = parser.push("nk>reason</think>answer");
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn delimited_parser_flushes_partial_end_marker() {
    let tokenizer = reasoning_tokenizer();
    let start = token_id(&tokenizer, "<think>");
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();
    parser.initialize(&[start]);

    let delta = parser.push("unfinished</thi");
    assert_eq!(delta.reasoning.as_deref(), Some("unfinished"));

    // At end of stream the incomplete delimiter is text of the current region.
    assert_eq!(parser.finish().reasoning.as_deref(), Some("</thi"));
}

#[test]
fn without_a_prompt_boundary_reasoning_waits_for_its_start() {
    let tokenizer = reasoning_tokenizer();
    let mut parser = think_parser(tokenizer);

    let delta = parser.push("reason</think>answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("reason</think>answer"));
}

#[test]
fn prompt_boundaries_select_the_initial_output_region() {
    let tokenizer = reasoning_tokenizer();
    let start = token_id(&tokenizer, "<think>");
    let end = token_id(&tokenizer, "</think>");

    let mut content = think_parser(Arc::clone(&tokenizer));
    content.initialize(&[end]);
    assert_eq!(content.push("answer").content.as_deref(), Some("answer"));

    let mut reasoning = think_parser(tokenizer);
    reasoning.initialize(&[start]);
    let delta = reasoning.push("reason</think>answer");
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

/// The backward prompt scan stops at `<|im_end|>` before reaching the earlier
/// `<think>`, so the parser keeps its default content region.
#[test]
fn the_prompt_scan_stops_at_the_last_special_token() {
    let tokenizer = reasoning_tokenizer();
    let start = token_id(&tokenizer, "<think>");
    let prompt_end = token_id(&tokenizer, "<|im_end|>");
    let mut parser = think_parser(tokenizer);

    parser.initialize(&[start, prompt_end]);
    let delta = parser.push("answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}
