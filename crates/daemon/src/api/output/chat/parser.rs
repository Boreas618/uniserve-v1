//! Assistant output processing for chat generation.
//!
//! [`ChatParser`] parses one request's decoded text, as vLLM's chat serving
//! does with its reasoning and tool parsers: an optional reasoning parser
//! splits reasoning from visible text, and an optional tool parser extracts
//! tool calls from the visible text (`tool`). Which parsers a request has is
//! stated by its lowering (`handoff::Parser`): the formats its model family's
//! chat profile names. The request's `OutputProcessor` feeds the parser and
//! frames its [`AssistantEvent`]s as content blocks and tool calls
//! (`structured`).

use proto::handoff;
use uuid::Uuid;

use crate::api::error::{Error, Result};
use crate::api::output::chat::AssistantBlockKind;
use crate::api::output::reasoning::{DelimitedReasoningParser, ReasoningDelta};
use crate::api::output::tools;
use crate::api::tokenizer::DynTokenizer;
use processing::ToolCallFormat;

use super::tool::ToolStage;

/// Parsed assistant output of one decoded update.
#[derive(Debug, Clone, PartialEq)]
pub enum AssistantEvent {
    /// Parsed text of kind `Text` or `Reasoning`.
    TextDelta {
        /// Whether the text is visible or reasoning.
        kind: AssistantBlockKind,
        /// The text.
        delta: String,
    },
    /// Opens a tool call; subsequent argument deltas belong to it.
    ToolCallStart {
        /// Tool-call identifier from `generate_tool_call_id`.
        id: String,
        /// Function name selected by the model.
        name: String,
    },
    /// Argument text for the most recently started tool call.
    ToolCallArgumentsDelta {
        /// Next chunk of the JSON arguments text.
        delta: String,
    },
}

/// The parsers of one chat request's decoded text: its reasoning and
/// tool-call parsers, either of which may be absent.
///
/// Without a reasoning parser all text is visible; without a tool parser
/// visible text passes unchanged. Text a parser holds back as a possible
/// partial delimiter is released by a later push or by [`Self::finish`].
#[derive(Default)]
pub struct ChatParser {
    reasoning: Option<DelimitedReasoningParser>,
    tools: Option<ToolStage>,
}

impl ChatParser {
    /// The parsers `parser` states for a request: the reasoning parser over its delimiters when
    /// reasoning is parsed, and the parser of its tool-call format ([`tools::parser`]) when tool
    /// calls are.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Parser`] when a tool-call format's name is no format's, the tokenizer
    /// lacks a reasoning delimiter as one token, or the request's tools are not the JSON array a
    /// request states.
    pub fn new(parser: &handoff::Parser, tokenizer: DynTokenizer) -> Result<Self> {
        let unknown = |kind: &'static str, name: &str| Error::Parser {
            kind,
            name: name.to_owned(),
            source: format!("no {kind} format is named {name:?}").into(),
        };

        let reasoning = match &parser.reasoning_delimiters {
            None => None,
            Some(delimiters) => {
                let reasoning = DelimitedReasoningParser::new(
                    tokenizer,
                    &delimiters.open,
                    &delimiters.close,
                    false,
                )
                .map_err(|error| Error::Parser {
                    kind: "reasoning",
                    name: format!("{} {}", delimiters.open, delimiters.close),
                    source: Box::new(error),
                })?;
                Some(reasoning)
            }
        };

        let tools = match parser.tool_calls.as_str() {
            "" => None,
            name => {
                let format = ToolCallFormat::named(name).ok_or_else(|| unknown("tool", name))?;
                Some(ToolStage::new(tools::parser(format, &parser.tools)?))
            }
        };

        Ok(Self { reasoning, tools })
    }

    /// The parsers' state, for the process that goes on writing the response after a handoff.
    pub fn state(&self) -> handoff::Processor {
        handoff::Processor {
            reasoning: self.reasoning.as_ref().map(DelimitedReasoningParser::state),
            tools: self.tools.as_ref().map(ToolStage::state),
        }
    }

    /// Puts the parsers back where `state` left them: the successor builds the same parsers for
    /// the request (its family and its tools) and resumes them from the state the predecessor
    /// handed over.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Lowering`] when `state` does not state each of this request's
    /// parsers, or states one of another format.
    pub fn resume(&mut self, state: &handoff::Processor) -> Result<()> {
        if let Some(reasoning) = self.reasoning.as_mut() {
            let reasoning_state = state.reasoning.as_ref().ok_or(Error::Lowering {
                part: "reasoning parser state",
            })?;
            reasoning.resume(reasoning_state);
        }
        if let Some(tools) = self.tools.as_mut() {
            let tools_state = state.tools.as_ref().ok_or(Error::Lowering {
                part: "tool stage state",
            })?;
            tools.resume(tools_state)?;
        }
        Ok(())
    }

    /// Chooses the region generation starts in from the prompt.
    ///
    /// A reasoning delimiter in the prompt suffix after the last other
    /// special token decides whether generation starts inside a reasoning
    /// section; without one, generation starts outside it.
    pub fn initialize(&mut self, prompt_token_ids: &[u32]) {
        if let Some(reasoning) = self.reasoning.as_mut() {
            reasoning.initialize(prompt_token_ids);
        }
    }

    /// Parses one decoded text delta into assistant events.
    ///
    /// One delta may span several delimiters, as when several tokens decode
    /// into one update; its reasoning text precedes its visible text.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ToolCall`] for a malformed tool call and
    /// [`Error::ToolCallStream`] when the tool parser's output is out of order.
    pub fn push(&mut self, delta: String) -> Result<Vec<AssistantEvent>> {
        let parsed = match self.reasoning.as_mut() {
            Some(reasoning) => reasoning.push(&delta),
            None => ReasoningDelta {
                reasoning: None,
                content: Some(delta),
            },
        };
        self.split(parsed)
    }

    /// Releases the text held back at the end of generation: the reasoning
    /// parser's partial delimiter as text of its current region, then the
    /// tool parser's buffered input.
    ///
    /// # Errors
    ///
    /// As [`Self::push`].
    pub fn finish(&mut self) -> Result<Vec<AssistantEvent>> {
        let parsed = self
            .reasoning
            .as_mut()
            .map(|reasoning| reasoning.finish())
            .unwrap_or_default();
        let mut events = self.split(parsed)?;
        if let Some(tools) = self.tools.as_mut() {
            events.extend(tools.finish()?);
        }
        Ok(events)
    }

    /// Emits the reasoning part of `parsed`, then its visible part through
    /// the tool stage.
    fn split(&mut self, parsed: ReasoningDelta) -> Result<Vec<AssistantEvent>> {
        let mut events = Vec::new();
        for (kind, text) in [
            (AssistantBlockKind::Reasoning, parsed.reasoning),
            (AssistantBlockKind::Text, parsed.content),
        ] {
            let Some(text) = text.filter(|text| !text.is_empty()) else {
                continue;
            };
            match self.tools.as_mut() {
                Some(tools) => events.extend(tools.push(kind, text)?),
                None => events.push(AssistantEvent::TextDelta { kind, delta: text }),
            }
        }
        Ok(events)
    }
}

/// Generates an OpenAI-compatible tool-call identifier: `call_` followed by
/// 24 hex digits of a random UUID.
pub(super) fn generate_tool_call_id() -> String {
    format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
}
