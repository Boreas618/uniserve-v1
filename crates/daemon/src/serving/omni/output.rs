//! SenseNova's reasoning parser.
//!
//! A SenseNova profile's output filter splits text into reasoning and visible
//! output with a `DelimitedReasoningParser`, then strips the profile's
//! visible-wrapper delimiters from the visible part. Both stages match
//! delimiters as text and hold back a trailing fragment that may begin a
//! delimiter split across pushes, which `finish` releases at the end of the
//! stream. [`SenseNovaReasoningParser`] is the reasoning parser of a
//! SenseNova request's `ChatOutputProcessor`.

use crate::profile::omni::{DelimitedTextPolicy, OutputFilterPolicy};
use crate::profile::reasoning::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser};
use crate::profile::tokenizer::DynTokenizer;

/// SenseNova's reasoning and visible-wrapper filter for one request.
pub(crate) struct SenseNovaReasoningParser {
    reasoning: Option<DelimitedReasoningParser>,
    visible_wrappers: VisibleWrapperFilter,
}

impl SenseNovaReasoningParser {
    /// Creates the parser of a SenseNova profile's output filter.
    ///
    /// # Errors
    ///
    /// Fails when a reasoning delimiter is empty or is not a single token in
    /// the tokenizer vocabulary.
    pub(crate) fn new(
        policy: OutputFilterPolicy,
        tokenizer: DynTokenizer,
    ) -> crate::profile::reasoning::Result<Self> {
        let reasoning = policy
            .reasoning
            .map(|reasoning| {
                DelimitedReasoningParser::new(tokenizer, reasoning.start, reasoning.end, false)
            })
            .transpose()?;
        Ok(Self {
            reasoning,
            visible_wrappers: VisibleWrapperFilter::new(policy.visible_wrappers),
        })
    }

    /// Strips the visible wrappers from the visible part of `delta`; a held
    /// partial wrapper delimiter is released with `release_held`.
    fn filter(&mut self, delta: ReasoningDelta, release_held: bool) -> ReasoningDelta {
        let mut visible = self
            .visible_wrappers
            .push(&delta.content.unwrap_or_default());
        if release_held {
            visible.push_str(&self.visible_wrappers.finish());
        }
        ReasoningDelta {
            reasoning: delta.reasoning.filter(|text| !text.is_empty()),
            content: Some(visible).filter(|text| !text.is_empty()),
        }
    }
}

impl ReasoningParser for SenseNovaReasoningParser {
    /// The prompt token IDs decide whether generation starts inside the
    /// reasoning section (see `DelimitedReasoningParser::initialize`).
    fn initialize(&mut self, prompt_token_ids: &[u32]) {
        if let Some(reasoning) = self.reasoning.as_mut() {
            reasoning.initialize(prompt_token_ids);
        }
    }

    /// Either part of the delta may be empty while a possible delimiter
    /// prefix is held back until a later push or `finish` releases it.
    fn push(&mut self, text: &str) -> ReasoningDelta {
        let delta = match self.reasoning.as_mut() {
            Some(reasoning) => reasoning.push(text),
            None => ReasoningDelta {
                reasoning: None,
                content: Some(text.to_owned()),
            },
        };
        self.filter(delta, false)
    }

    /// No later text can complete a delimiter, so a held fragment is
    /// ordinary text of its current region: the reasoning parser's buffered
    /// partial delimiter becomes reasoning or visible text
    /// (`DelimitedReasoningParser::finish`), visible text still passes the
    /// wrapper filter, and the filter's held partial wrapper delimiter is
    /// visible text.
    fn finish(&mut self) -> ReasoningDelta {
        let delta = self
            .reasoning
            .as_mut()
            .map(|reasoning| reasoning.finish())
            .unwrap_or_default();
        self.filter(delta, true)
    }
}

/// Removes wrapper delimiters from visible text while keeping the wrapped text.
struct VisibleWrapperFilter {
    wrappers: Vec<DelimitedTextPolicy>,
    /// Text not yet emitted: at most a trailing fragment that is a proper
    /// prefix of some delimiter.
    pending: String,
}

impl VisibleWrapperFilter {
    /// Creates a filter for the configured wrapper delimiters.
    fn new(wrappers: Vec<DelimitedTextPolicy>) -> Self {
        Self {
            wrappers,
            pending: String::new(),
        }
    }

    /// Removes configured wrapper delimiters while retaining incomplete marker prefixes.
    ///
    /// Start and end delimiters are removed independently, so an unmatched
    /// delimiter is also dropped.
    fn push(&mut self, text: &str) -> String {
        if self.wrappers.is_empty() {
            return text.to_string();
        }
        self.pending.push_str(text);
        let mut visible = String::new();

        // Strip complete markers from the front, then emit everything except a
        // trailing fragment that could still grow into a marker.
        loop {
            if let Some((start, marker_len)) = self.first_marker() {
                visible.push_str(&self.pending[..start]);
                self.pending.drain(..start + marker_len);
                continue;
            }

            let keep = trailing_marker_prefix_len_any(&self.pending, &self.markers());
            let emit_to = self.pending.len().saturating_sub(keep);
            visible.push_str(&self.pending[..emit_to]);
            self.pending.drain(..emit_to);
            break;
        }

        visible
    }

    /// Returns the held trailing fragment at the end of the stream.
    ///
    /// The fragment is a proper prefix of a delimiter that can no longer
    /// complete, so it is visible text.
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Returns all configured output markers.
    fn markers(&self) -> Vec<&str> {
        let mut markers = Vec::new();
        for wrapper in &self.wrappers {
            markers.push(wrapper.start.as_str());
            markers.push(wrapper.end.as_str());
        }
        markers
    }

    /// Returns the byte offset and length of the earliest complete marker in
    /// `pending`.
    fn first_marker(&self) -> Option<(usize, usize)> {
        let mut matches = Vec::new();
        for wrapper in &self.wrappers {
            matches.push(wrapper.start.as_str());
            matches.push(wrapper.end.as_str());
        }
        matches
            .into_iter()
            .filter_map(|marker| self.pending.find(marker).map(|idx| (idx, marker.len())))
            .min_by_key(|(idx, _)| *idx)
    }
}

/// Returns the longest suffix matching any marker prefix.
fn trailing_marker_prefix_len_any(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .map(|marker| trailing_marker_prefix_len(text, marker))
        .max()
        .unwrap_or(0)
}

/// Returns the length of the longest suffix of `text` that is a proper prefix of `marker`.
///
/// A full marker is not counted; `VisibleWrapperFilter::push` strips complete
/// markers before calling this. Lengths are in bytes and only split at UTF-8
/// character boundaries.
fn trailing_marker_prefix_len(text: &str, marker: &str) -> usize {
    let max = text.len().min(marker.len().saturating_sub(1));
    for len in (1..=max).rev() {
        if text.is_char_boundary(text.len() - len)
            && marker.is_char_boundary(len)
            && text[text.len() - len..].eq(&marker[..len])
        {
            return len;
        }
    }
    0
}
