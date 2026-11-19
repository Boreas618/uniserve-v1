//! The streaming reasoning parser and its deltas.
//!
//! [`DelimitedReasoningParser`] splits a stream of decoded text deltas into
//! reasoning text and visible content by matching the delimiters of the
//! family's reasoning format (`crate::ReasoningFormat`), such as
//! `<think>` and `</think>`. One instance serves one generation stream: call
//! `initialize` with the prompt's tail, `push` for each decoded delta, then `finish`
//! once at end of stream.

mod delimited;

pub use self::delimited::{DelimitedReasoningParser, State};

/// One parsed streaming delta split into reasoning and visible content.
///
/// The push methods ignore empty text, so a field they populate always holds
/// non-empty text.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReasoningDelta {
    /// Incremental hidden reasoning text.
    pub reasoning: Option<String>,
    /// Incremental user-visible assistant text.
    pub content: Option<String>,
}

impl ReasoningDelta {
    /// Returns `true` when this delta carries neither reasoning nor content text.
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none()
    }

    /// Appends text to the reasoning portion, creating it on first use.
    pub fn push_reasoning(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.reasoning {
            Some(existing) => existing.push_str(text),
            None => self.reasoning = Some(text.to_string()),
        }
    }

    /// Appends text to the visible content portion, creating it on first use.
    pub fn push_content(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.content {
            Some(existing) => existing.push_str(text),
            None => self.content = Some(text.to_string()),
        }
    }
}

#[cfg(test)]
mod tests;
