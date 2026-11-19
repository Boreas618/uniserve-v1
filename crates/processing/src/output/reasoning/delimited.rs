//! Incremental parser for reasoning enclosed by a family's text delimiters.
//!
//! Delimiters are matched as text in the decoded stream, so text that may be the start of a
//! delimiter split across deltas is held back until later text completes or rules it out, or
//! the stream ends. Each delimiter is one vocabulary token, whose id `initialize` looks for in
//! the prompt's tail to choose the initial region.
//!
//! A family whose tool calls end the turn in a final region (`crate::ToolCallFormat::final_region`)
//! states the region's opening: it ends reasoning the model did not close, and from it on the
//! response is the region and what follows it, which the tool stage reads, so no delimiter is read
//! after it (a `<think>` inside a call's argument is the argument's text). The separator the
//! format writes before the opening goes with the opening, to the content.

use serde::{Deserialize, Serialize};

use super::ReasoningDelta;
use crate::output::Delimiters;
use crate::serving::Opening;

/// Shared incremental state machine for tag-delimited reasoning.
///
/// One instance serves one generation stream: call `initialize` with the prompt's tail, then
/// `push` for each decoded delta, then `finish` once at end of stream. Generation starts in
/// visible content unless the prompt's tail opens a reasoning section.
pub struct DelimitedReasoningParser {
    delimiters: Delimiters,
    /// The opening of the family's final tool-call region, when its calls end the turn.
    opening: Option<Opening>,
    current_in_reasoning: bool,
    /// Whether the text reached the final region's opening: all text from it on is content.
    in_final_region: bool,
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
    /// Whether the text reached the opening of the family's final tool-call region.
    pub in_final_region: bool,
    /// Text a later delta may still complete into a delimiter.
    pub buffer: String,
}

impl DelimitedReasoningParser {
    /// The parser of reasoning between `delimiters`, starting in visible content; `opening` is
    /// the opening of the family's final tool-call region, when its tool calls end the turn.
    pub fn new(delimiters: Delimiters, opening: Option<Opening>) -> Self {
        Self {
            delimiters,
            opening,
            current_in_reasoning: false,
            in_final_region: false,
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
            in_final_region: self.in_final_region,
            buffer: self.buffer.clone(),
        }
    }

    /// Restores the state machine's state.
    pub fn resume(&mut self, state: State) {
        self.current_in_reasoning = state.in_reasoning;
        self.in_final_region = state.in_final_region;
        self.buffer = state.buffer;
    }

    /// Parses the buffered text, keeping back what a later delta may still change: a trailing
    /// partial delimiter. `finishing` keeps back nothing.
    fn parse_buffer(&mut self, finishing: bool) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();

        loop {
            if self.in_final_region {
                delta.push_content(&self.buffer);
                self.buffer.clear();
                return delta;
            }

            let held = if finishing {
                0
            } else {
                self.partial_suffix_len(&self.buffer)
            };
            let stable = &self.buffer[..self.buffer.len() - held];
            let delimiter = if self.current_in_reasoning {
                self.delimiters.format.close
            } else {
                self.delimiters.format.open
            };

            // The earlier of the region's delimiter and the final region's opening ends the
            // current region's text: the delimiter belongs to neither region, the opening (with
            // the separator before it) to the content that follows.
            let at_delimiter = stable.find(delimiter);
            let at_opening = self
                .opening
                .and_then(|opening| stable.find(opening.delimiter).map(|at| (at, opening)));
            let boundary = match (at_delimiter, at_opening) {
                (Some(at), Some((opening_at, _))) if at < opening_at => Boundary::Delimiter(at),
                (_, Some((at, opening))) => {
                    let separated =
                        self.current_in_reasoning && stable[..at].ends_with(opening.separator);
                    let at = if separated {
                        at - opening.separator.len()
                    } else {
                        at
                    };
                    Boundary::Opening(at)
                }
                (Some(at), None) => Boundary::Delimiter(at),
                (None, None) => Boundary::None(stable.len()),
            };

            let (text_len, consumed) = match boundary {
                Boundary::Delimiter(at) => (at, at + delimiter.len()),
                Boundary::Opening(at) | Boundary::None(at) => (at, at),
            };
            let text: String = self.buffer.drain(..consumed).collect();
            if self.current_in_reasoning {
                delta.push_reasoning(&text[..text_len]);
            } else {
                delta.push_content(&text[..text_len]);
            }
            match boundary {
                Boundary::Delimiter(_) => self.current_in_reasoning = !self.current_in_reasoning,
                Boundary::Opening(_) => {
                    self.current_in_reasoning = false;
                    self.in_final_region = true;
                }
                Boundary::None(_) => return delta,
            }
        }
    }

    /// Returns the length of the longest trailing suffix that could still complete a delimiter:
    /// either delimiter, the final region's opening, or, inside reasoning, the opening with the
    /// separator before it.
    ///
    /// A trailing suffix can only be a *strict* prefix of a delimiter when it is shorter than
    /// that delimiter, so only the final bytes of `text`, as many as the longest delimiter has,
    /// can ever match. Scanning just that trailing window keeps the cost bounded by the delimiter
    /// length rather than the length of `text`.
    fn partial_suffix_len(&self, text: &str) -> usize {
        let format = self.delimiters.format;
        let separated = self
            .opening
            .filter(|_| self.current_in_reasoning)
            .map(|opening| format!("{}{}", opening.separator, opening.delimiter));
        let candidates: Vec<&str> = [format.open, format.close]
            .into_iter()
            .chain(self.opening.map(|opening| opening.delimiter))
            .chain(separated.as_deref())
            .collect();
        let longest = candidates.iter().map(|candidate| candidate.len()).max();

        // Suffixes at least as long as the longest candidate cannot be a strict prefix of one,
        // so start scanning from there. Clamp to a char boundary so the slice below is always
        // valid.
        let mut window_start = text.len().saturating_sub(longest.unwrap_or(0));
        while window_start < text.len() && !text.is_char_boundary(window_start) {
            window_start += 1;
        }

        // The first (longest) suffix that is a strict prefix of a candidate is the one held.
        text[window_start..]
            .char_indices()
            .map(|(index, _)| &text[window_start + index..])
            .find(|suffix| {
                candidates.iter().any(|candidate| {
                    candidate.starts_with(suffix) && candidate.len() > suffix.len()
                })
            })
            .map_or(0, str::len)
    }
}

/// Where the current region's text ends in the stable text.
enum Boundary {
    /// At the current region's delimiter, which switches the region.
    Delimiter(usize),
    /// At the final region's opening, or at the separator before it.
    Opening(usize),
    /// Nowhere yet: all the stable text is the region's.
    None(usize),
}
