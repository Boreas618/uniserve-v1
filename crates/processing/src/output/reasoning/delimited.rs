//! Incremental parser for reasoning enclosed by a family's text delimiters.
//!
//! Delimiters are matched as text in the decoded stream, so text that may be the start of a
//! delimiter split across deltas is held back until later text completes or rules it out, or
//! the stream ends. Each delimiter is one vocabulary token, whose id `initialize` looks for in
//! the prompt's tail to choose the initial region.

use serde::{Deserialize, Serialize};

use super::ReasoningDelta;
use crate::output::Delimiters;

/// Shared incremental state machine for tag-delimited reasoning.
///
/// One instance serves one generation stream: call `initialize` with the prompt's tail, then
/// `push` for each decoded delta, then `finish` once at end of stream. Generation starts in
/// visible content unless the prompt's tail opens a reasoning section.
pub struct DelimitedReasoningParser {
    delimiters: Delimiters,
    current_in_reasoning: bool,
    /// Text not yet parsed: a trailing partial delimiter, held until a later delta completes or
    /// rules it out, or `finish` flushes it.
    buffer: String,
}

/// The state machine's state, for a response that moves to another process. Its delimiters
/// are the family's, which the successor's binding restates.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    /// Whether the text parsed so far is inside a reasoning section.
    pub in_reasoning: bool,
    /// Text a later delta may still complete into a delimiter.
    pub buffer: String,
}

impl DelimitedReasoningParser {
    /// The parser of reasoning between `delimiters`, starting in visible content.
    pub fn new(delimiters: Delimiters) -> Self {
        Self {
            delimiters,
            current_in_reasoning: false,
            buffer: String::new(),
        }
    }

    /// Chooses the region generation starts in from `tail`, the prompt's ids from its last
    /// special token on: the last delimiter in it decides, the open delimiter starting inside
    /// reasoning and the close delimiter in visible content. A tail without either starts in
    /// visible content: a delimiter before the prompt's last special token belongs to an earlier
    /// turn.
    pub fn initialize(&mut self, tail: &[u32]) {
        let open = self.delimiters.open;
        let close = self.delimiters.close;
        self.current_in_reasoning = tail
            .iter()
            .rev()
            .find_map(|&id| match id {
                id if id == open => Some(true),
                id if id == close => Some(false),
                _ => None,
            })
            .unwrap_or(false);
    }

    /// Parses one decoded text delta and returns its reasoning/content split.
    ///
    /// Text that could begin a delimiter stays buffered, so the returned delta may be empty.
    pub fn push(&mut self, delta: &str) -> ReasoningDelta {
        self.buffer.push_str(delta);
        self.parse_buffer(false)
    }

    /// Flushes buffered text at end of stream.
    ///
    /// An unfinished delimiter is emitted as ordinary text of the current region.
    pub fn finish(&mut self) -> ReasoningDelta {
        self.parse_buffer(true)
    }

    /// The state machine's state.
    pub fn state(&self) -> State {
        State {
            in_reasoning: self.current_in_reasoning,
            buffer: self.buffer.clone(),
        }
    }

    /// Restores the state machine's state.
    pub fn resume(&mut self, state: State) {
        self.current_in_reasoning = state.in_reasoning;
        self.buffer = state.buffer;
    }

    /// Parses the buffered text, keeping back what a later delta may still change: a trailing
    /// partial delimiter. `finishing` keeps back nothing.
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
                self.delimiters.format.close
            } else {
                self.delimiters.format.open
            };
            let found = self.buffer[..stable_len]
                .find(delimiter)
                .map(|index| (index, index + delimiter.len()));

            // Text up to the next delimiter belongs to the current region; the delimiter itself
            // belongs to neither.
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

    /// Returns the longest trailing suffix that could still complete a delimiter.
    ///
    /// A trailing suffix can only be a *strict* prefix of a delimiter when it is shorter than
    /// that delimiter, so only the final `max(open, close)` bytes of `text` can ever match.
    /// Scanning just that trailing window keeps the cost bounded by the delimiter length rather
    /// than the length of `text`.
    fn partial_suffix_len(&self, text: &str) -> usize {
        let (open, close) = (self.delimiters.format.open, self.delimiters.format.close);
        let max_delimiter_len = open.len().max(close.len());
        // Suffixes at least `max_delimiter_len` bytes long cannot be a strict prefix of either
        // delimiter, so start scanning from there. Clamp to a char boundary so the slice below
        // is always valid.
        let mut window_start = text.len().saturating_sub(max_delimiter_len);
        while window_start < text.len() && !text.is_char_boundary(window_start) {
            window_start += 1;
        }

        let mut best = 0;
        for (idx, _) in text[window_start..].char_indices() {
            let idx = window_start + idx;
            let suffix = &text[idx..];
            if open.starts_with(suffix) && open != suffix {
                best = best.max(text.len() - idx);
            }
            if close.starts_with(suffix) && close != suffix {
                best = best.max(text.len() - idx);
            }
        }

        best
    }
}
