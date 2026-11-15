//! Incremental parser for reasoning enclosed by configured text delimiters.
//!
//! Delimiters are matched as text in the decoded stream, so text that may be
//! the start of a delimiter split across deltas is held back until later text
//! completes or rules it out, or the stream ends. Each delimiter begins with a
//! single vocabulary token, whose id `initialize` looks for in the prompt to
//! choose the initial region.

use crate::api::tokenizer::{DynTokenizer, HuggingFaceTokenizer};

use super::{ReasoningDelta, ReasoningError, Result};

/// Shared incremental state machine for configured tag-delimited reasoning.
///
/// One instance serves one generation stream: optionally call `initialize`
/// with the prompt, then `push` for each decoded delta, then `finish` once at
/// end of stream.
pub struct DelimitedReasoningParser {
    tokenizer: DynTokenizer,
    current_in_reasoning: bool,
    /// Text not yet parsed: a trailing partial delimiter, held until a later delta completes or
    /// rules it out, or `finish` flushes it.
    buffer: String,
    /// Text that opens a reasoning section: the start token.
    start_delimiter: String,
    /// Text that closes a reasoning section: the end token.
    end_delimiter: String,
    start_token_id: u32,
    end_token_id: u32,
    default_in_reasoning: bool,
}

impl DelimitedReasoningParser {
    /// Creates one delimited parser state machine.
    ///
    /// `start_token` and `end_token` are the delimiter texts, each a single
    /// vocabulary token.
    /// `default_in_reasoning` is the initial region when `initialize` finds no
    /// delimiter token after the prompt's last other special token, and also
    /// when `initialize` is never called.
    ///
    /// # Errors
    ///
    /// Fails when either delimiter is empty or is not a single token in the
    /// tokenizer vocabulary.
    pub fn new(
        tokenizer: DynTokenizer,
        start_token: impl Into<String>,
        end_token: impl Into<String>,
        default_in_reasoning: bool,
    ) -> Result<Self> {
        let start_token = start_token.into();
        let end_token = end_token.into();
        if start_token.is_empty() {
            return Err(ReasoningError::EmptyDelimiter { field: "start" });
        }
        if end_token.is_empty() {
            return Err(ReasoningError::EmptyDelimiter { field: "end" });
        }
        let start_token_id =
            tokenizer
                .token_to_id(&start_token)
                .ok_or_else(|| ReasoningError::MissingToken {
                    token: start_token.clone(),
                })?;
        let end_token_id =
            tokenizer
                .token_to_id(&end_token)
                .ok_or_else(|| ReasoningError::MissingToken {
                    token: end_token.clone(),
                })?;

        Ok(Self {
            tokenizer,
            current_in_reasoning: default_in_reasoning,
            buffer: String::new(),
            start_delimiter: start_token,
            end_delimiter: end_token,
            start_token_id,
            end_token_id,
            default_in_reasoning,
        })
    }

    /// Parses the buffered text, keeping back what a later delta may still
    /// change: a trailing partial delimiter. `finishing` keeps back nothing.
    fn parse_buffer(&mut self, finishing: bool) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();

        loop {
            let held = if finishing {
                0
            } else {
                self.partial_suffix_len(&self.buffer)
            };
            let stable_len = self.buffer.len() - held;
            let delimiter = if self.current_in_reasoning {
                &self.end_delimiter
            } else {
                &self.start_delimiter
            };
            let found = self.buffer[..stable_len]
                .find(delimiter.as_str())
                .map(|index| (index, index + delimiter.len()));

            // Text up to the next delimiter belongs to the current region;
            // the delimiter itself belongs to neither.
            let (text_len, consumed) = found.unwrap_or((stable_len, stable_len));
            let text: String = self.buffer.drain(..consumed).collect();
            if self.current_in_reasoning {
                delta.push_reasoning(&text[..text_len]);
            } else {
                delta.push_content(&text[..text_len]);
            }
            if found.is_none() {
                return delta;
            }

            self.current_in_reasoning = !self.current_in_reasoning;
        }
    }

    /// The state machine's state, for a response that moves to another
    /// process. Its delimiters and region default are configuration, which
    /// the request's model family restates.
    pub fn state(&self) -> proto::handoff::Reasoning {
        proto::handoff::Reasoning {
            in_reasoning: self.current_in_reasoning,
            buffer: self.buffer.clone(),
        }
    }

    /// Restores the state machine's state.
    pub fn resume(&mut self, state: &proto::handoff::Reasoning) {
        self.current_in_reasoning = state.in_reasoning;
        self.buffer.clone_from(&state.buffer);
    }

    /// Returns the longest trailing suffix that could still complete a
    /// delimiter.
    ///
    /// A trailing suffix can only be a *strict* prefix of a delimiter when it
    /// is shorter than that delimiter, so only the final
    /// `max(start_delimiter, end_delimiter)` bytes of `text` can ever match.
    /// Scanning just that trailing window keeps the cost bounded by the
    /// delimiter length rather than the length of `text`.
    fn partial_suffix_len(&self, text: &str) -> usize {
        let max_delimiter_len = self.start_delimiter.len().max(self.end_delimiter.len());
        // Suffixes at least `max_delimiter_len` bytes long cannot be a strict
        // prefix of either delimiter, so start scanning from there. Clamp to a
        // char boundary so the slice below is always valid.
        let mut window_start = text.len().saturating_sub(max_delimiter_len);
        while window_start < text.len() && !text.is_char_boundary(window_start) {
            window_start += 1;
        }

        let mut best = 0;
        for (idx, _) in text[window_start..].char_indices() {
            let idx = window_start + idx;
            let suffix = &text[idx..];
            if self.start_delimiter.starts_with(suffix) && self.start_delimiter != suffix {
                best = best.max(text.len() - idx);
            }
            if self.end_delimiter.starts_with(suffix) && self.end_delimiter != suffix {
                best = best.max(text.len() - idx);
            }
        }

        best
    }
}

impl DelimitedReasoningParser {
    /// Initializes parser state from the prompt token IDs.
    ///
    /// The last delimiter token in the prompt decides where generation starts
    /// when no other special token follows it: with `<think>` delimiters, a
    /// prompt ending in `<think>` starts inside reasoning and one ending in
    /// `</think>` starts in visible content. Otherwise the parser starts in the
    /// `default_in_reasoning` region.
    pub fn initialize(&mut self, prompt_token_ids: &[u32]) {
        self.current_in_reasoning = last_reasoning_boundary(
            prompt_token_ids,
            self.start_token_id,
            self.end_token_id,
            self.tokenizer.as_ref(),
        )
        .unwrap_or(self.default_in_reasoning);
    }

    /// Parses one decoded text delta and returns its reasoning/content split.
    ///
    /// Text that could begin a delimiter stays buffered, so the returned
    /// delta may be empty.
    pub fn push(&mut self, delta: &str) -> ReasoningDelta {
        self.buffer.push_str(delta);
        self.parse_buffer(false)
    }

    /// Flushes buffered text at end of stream.
    ///
    /// An unfinished delimiter is emitted as ordinary text of the current
    /// region.
    pub fn finish(&mut self) -> ReasoningDelta {
        self.parse_buffer(true)
    }
}

/// Determines the reasoning state implied by the last prompt boundary, if any.
///
/// Scans backwards from the end of the prompt. The first delimiter found
/// decides: the start delimiter means inside reasoning, the end delimiter
/// means outside. Reaching any other special token first, or the start of the
/// prompt, yields `None`.
fn last_reasoning_boundary(
    prompt_token_ids: &[u32],
    start_token_id: u32,
    end_token_id: u32,
    tokenizer: &HuggingFaceTokenizer,
) -> Option<bool> {
    for token_id in prompt_token_ids.iter().rev() {
        if *token_id == start_token_id {
            return Some(true);
        }
        if *token_id == end_token_id {
            return Some(false);
        }
        if tokenizer.is_special_id(*token_id) {
            return None;
        }
    }

    None
}
