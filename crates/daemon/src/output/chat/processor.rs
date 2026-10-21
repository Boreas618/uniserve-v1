//! Assistant output processing for chat generation.
//!
//! [`ChatOutputProcessor`] parses one request's decoded text, as vLLM's
//! default chat output processor does: an optional reasoning parser splits
//! reasoning from visible text, and an optional tool parser extracts tool
//! calls from the visible text (`tool`). Each model family chooses its
//! parsers: Qwen3 `<think>` reasoning and JSON tool calls, Gemma-4 thought
//! channels and native tool calls, SenseNova's delimited reasoning, and no
//! parser for Bagel. `output::assembly` feeds the processor and frames its
//! [`AssistantEvent`]s as content blocks and tool calls (`structured`).

use uuid::Uuid;

use crate::output::chat::{AssistantBlockKind, Error, Result};
use crate::output::reasoning::{
    Gemma4ReasoningParser, Qwen3ReasoningParser, ReasoningDelta, ReasoningParser,
};
use crate::output::text::TextDecodeOptions;
use crate::output::tools::{Gemma4ToolParser, Qwen25ToolParser, Tool, ToolParser};
use crate::tokenizer::DynTokenizer;

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

/// The request-scoped output processor of a chat request: its reasoning and
/// tool-call parsers, either of which may be absent.
///
/// Without a reasoning parser all text is visible; without a tool parser
/// visible text passes unchanged. Text a parser holds back as a possible
/// partial delimiter is released by a later push or by [`Self::finish`].
#[derive(Default)]
pub struct ChatOutputProcessor {
    reasoning: Option<Box<dyn ReasoningParser>>,
    tools: Option<ToolStage>,
}

impl ChatOutputProcessor {
    /// A processor over `reasoning` and `tools`.
    pub fn new(
        reasoning: Option<Box<dyn ReasoningParser>>,
        tools: Option<Box<dyn ToolParser>>,
    ) -> Self {
        Self {
            reasoning,
            tools: tools.map(ToolStage::new),
        }
    }

    /// The Qwen3 processor of a request: `<think>` reasoning when
    /// `parse_reasoning` is set, and JSON tool calls naming `tools` when a
    /// request parses tool calls (`ChatRequest::tool_parsing_enabled`, the
    /// condition under which its chat template sees the tools).
    ///
    /// # Errors
    ///
    /// Returns `Error::ParserInitialization` when the tokenizer lacks the
    /// `<think>` or `</think>` token.
    pub fn qwen3(
        tools: Option<&[Tool]>,
        tokenizer: DynTokenizer,
        parse_reasoning: bool,
    ) -> Result<Self> {
        let reasoning = parse_reasoning
            .then(|| Qwen3ReasoningParser::new(tokenizer))
            .transpose()
            .map_err(|error| Error::ParserInitialization {
                kind: "reasoning",
                name: "qwen3".to_string(),
                error: Box::new(error),
            })?;
        let tools = tools.map(Qwen25ToolParser::new);
        Ok(Self::new(
            reasoning.map(|parser| Box::new(parser) as Box<dyn ReasoningParser>),
            tools.map(|parser| Box::new(parser) as Box<dyn ToolParser>),
        ))
    }

    /// The Gemma-4 processor of a request: thought channels when
    /// `parse_reasoning` is set, and native tool calls when `parse_tools` is.
    ///
    /// The thinking mode needs no separate input: the prompt suffix decides
    /// whether generation starts inside a thought channel, and otherwise the
    /// model's own channel delimiters mark reasoning. Sets
    /// `decode.skip_special_tokens` to `false`: Gemma-4
    /// delimits thought channels, tool calls, and string arguments with
    /// special tokens, which the parsers must see and which stream verbatim
    /// as content when parsing is disabled. DiffusionGemma commits output in
    /// blocks of tokens, so one decoded delta can hold a whole thought
    /// channel, visible text, and complete tool calls; both parsers accept
    /// any number of delimiters per delta, as well as delimiters split across
    /// deltas.
    ///
    /// # Errors
    ///
    /// Returns `Error::ParserInitialization` when the tokenizer lacks the
    /// `<|channel>` or `<channel|>` token.
    pub fn gemma4(
        parse_tools: bool,
        decode: &mut TextDecodeOptions,
        tokenizer: DynTokenizer,
        parse_reasoning: bool,
    ) -> Result<Self> {
        decode.skip_special_tokens = false;
        let reasoning = parse_reasoning
            .then(|| Gemma4ReasoningParser::new(tokenizer))
            .transpose()
            .map_err(|error| Error::ParserInitialization {
                kind: "reasoning",
                name: "gemma4".to_string(),
                error: Box::new(error),
            })?;
        let tools = parse_tools.then(Gemma4ToolParser::new);
        Ok(Self::new(
            reasoning.map(|parser| Box::new(parser) as Box<dyn ReasoningParser>),
            tools.map(|parser| Box::new(parser) as Box<dyn ToolParser>),
        ))
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
    /// One delta may span several delimiters, as when a block-diffusion model
    /// commits a whole block of tokens at once; its reasoning text precedes
    /// its visible text.
    ///
    /// # Errors
    ///
    /// Returns `Error::ToolCallStreamInvariant` when the tool parser's output
    /// is inconsistent.
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
pub(crate) fn generate_tool_call_id() -> String {
    format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
}
