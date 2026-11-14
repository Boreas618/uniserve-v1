//! Output processing of one token request: its engine events become the response's deltas.
//!
//! [`OutputProcessor`] is what a request's response is produced through, as vLLM's
//! `OutputProcessor` holds a request's state: it checks the engine's event order and
//! detokenizes incrementally with the stop strings (`text`), parses reasoning and tool calls
//! in the formats the request's model family writes (`chat`), assembles the content blocks
//! (`chat::structured`), keeps the timings the terminal usage reports, and records the
//! request's statistics. A route's response body pushes one engine event at a time and
//! writes the [`RequestOutput`]s it gets back.
//!
//! Everything it is built from is the lowering input processing stated for the request
//! (`proto::handoff::Lowering`), so a successor daemon rebuilds the same processor from a
//! handed-over response's state and resumes every stage where its predecessor stopped
//! ([`OutputProcessor::resume`]).

use std::sync::Arc;

use proto::handoff;
use proto::request::Event;

use crate::api::events::{Events, Stats};
use crate::api::output::assembly::StreamClock;
use crate::api::output::chat::ChatParser;
use crate::api::output::chat::structured::StructuredEventState;
use crate::api::output::text::{DecodeStream, DecodedTextEvent};
use crate::api::output::{OutputError, RequestOutput, RequestUsage};
use crate::api::tokenizer::DynTokenizer;

/// The output processing of one token request.
pub struct OutputProcessor {
    /// The response's OpenAI fields, which also name the request in errors.
    response: handoff::ChatResponseOptions,
    /// The parsers the request's model family chose, restated to a successor.
    parser: handoff::Parser,
    prompt_logprobs: bool,
    logprobs: bool,
    decode: DecodeStream,
    parsers: ChatParser,
    blocks: StructuredEventState,
    clock: StreamClock,
    /// The request's statistics, recorded at its terminal output.
    stats: Box<dyn Stats>,
    /// Whether the decoder's terminal event was consumed.
    ended: bool,
}

impl OutputProcessor {
    /// The output processor of the request `lowering` describes, before any engine event.
    ///
    /// `stats` records the request's statistics; the instant it started at is the origin of
    /// the timings the response reports.
    ///
    /// # Errors
    ///
    /// Fails when the prompt is empty or the tokenizer lacks a delimiter of the family's
    /// reasoning format.
    pub fn new(
        tokenizer: DynTokenizer,
        lowering: handoff::Lowering,
        stats: Box<dyn Stats>,
    ) -> Result<Self, OutputError> {
        let handoff::Lowering {
            prompt_token_ids,
            decode_options,
            prompt_logprobs,
            logprobs,
            parser,
            response,
            compile_us,
            started_unix_us: _,
        } = lowering;
        let response = response.unwrap_or_default();
        let parser = parser.unwrap_or_default();
        let failed = |source: crate::api::output::chat::Error| OutputError {
            request_id: response.request_id.clone(),
            source,
        };

        let parsers = ChatParser::new(&parser, Arc::clone(&tokenizer)).map_err(failed)?;
        let decode = DecodeStream::new(
            response.request_id.clone(),
            tokenizer,
            prompt_token_ids,
            prompt_logprobs,
            logprobs,
            decode_options.unwrap_or_default(),
        )
        .map_err(|source| failed(source.into()))?;
        let clock = StreamClock::new(stats.started(), compile_us);

        Ok(Self {
            response,
            parser,
            prompt_logprobs,
            logprobs,
            decode,
            parsers,
            blocks: StructuredEventState::new(),
            clock,
            stats,
            ended: false,
        })
    }

    /// Continues the output processing a handed-over response states: the processor is
    /// rebuilt from `lowering` on this process's `tokenizer`, and its decoder, parsers and
    /// block assembly resume from `decode`, `parsers` and `assembly`.
    ///
    /// # Errors
    ///
    /// Fails as [`Self::new`] does.
    pub fn resume(
        tokenizer: DynTokenizer,
        lowering: &handoff::Lowering,
        decode: &handoff::Decode,
        parsers: Option<&handoff::Processor>,
        assembly: &handoff::Assembly,
        stats: Box<dyn Stats>,
    ) -> Result<Self, OutputError> {
        let response = lowering.response.clone().unwrap_or_default();
        let parser = lowering.parser.clone().unwrap_or_default();
        let failed = |source: crate::api::output::chat::Error| OutputError {
            request_id: response.request_id.clone(),
            source,
        };

        let mut chat = ChatParser::new(&parser, Arc::clone(&tokenizer)).map_err(failed)?;
        if let Some(state) = parsers {
            chat.resume(state);
        }
        let decode = DecodeStream::resume(
            response.request_id.clone(),
            tokenizer,
            lowering.prompt_token_ids.clone(),
            lowering.prompt_logprobs,
            lowering.logprobs,
            lowering.decode_options.clone().unwrap_or_default(),
            decode,
        )
        .map_err(|source| failed(source.into()))?;
        let mut blocks = StructuredEventState::new();
        blocks.resume(assembly);

        Ok(Self {
            response,
            parser,
            prompt_logprobs: lowering.prompt_logprobs,
            logprobs: lowering.logprobs,
            decode,
            parsers: chat,
            blocks,
            clock: StreamClock::resume(assembly, lowering.started_unix_us, lowering.compile_us),
            stats,
            ended: false,
        })
    }

    /// The response's OpenAI fields.
    pub fn response(&self) -> &handoff::ChatResponseOptions {
        &self.response
    }

    /// Which engine replica serves the request.
    pub fn engine(&self) -> u32 {
        self.stats.engine()
    }

    /// Whether the request's terminal output was produced.
    pub fn ended(&self) -> bool {
        self.ended
    }

    /// Counts the request as one its client abandoned or a successor took over.
    pub fn abandoned(&mut self) {
        self.stats.abandoned();
    }

    /// The response deltas one engine event produces, in order: the prompt and scheduling
    /// facts first, then each update's content-block and tool-call outputs followed by its
    /// `TokenDelta`, then the terminal output.
    ///
    /// `events` is the request's engine stream: a token that completes a stop string stops
    /// generation there, and every other token is acknowledged.
    ///
    /// # Errors
    ///
    /// Fails when the event violates the engine's event order, its tokens cannot be
    /// decoded, or the tool parser's output is inconsistent.
    pub fn push(
        &mut self,
        event: Event,
        events: &mut dyn Events,
    ) -> Result<Vec<RequestOutput>, OutputError> {
        let request_id = self.response.request_id.clone();
        let failed = |source: crate::api::output::chat::Error| OutputError {
            request_id: request_id.clone(),
            source,
        };
        let decoded = self
            .decode
            .push(event, events)
            .map_err(|source| failed(source.into()))?;

        let mut outputs = Vec::new();
        for event in decoded {
            match event {
                DecodedTextEvent::Start {
                    prompt,
                    queued_at,
                    scheduled_at,
                } => {
                    self.parsers.initialize(&prompt.token_ids);
                    self.clock.scheduled(queued_at, scheduled_at);
                    outputs.push(RequestOutput::Accepted(prompt));
                    outputs.push(RequestOutput::Scheduled {
                        queued_at,
                        scheduled_at,
                    });
                }
                DecodedTextEvent::TextDelta {
                    delta,
                    token_ids,
                    logprobs,
                } => {
                    for event in self.parsers.push(delta).map_err(failed)? {
                        outputs.extend(self.blocks.push(event).map_err(failed)?);
                    }
                    self.visible(&outputs);
                    if !token_ids.is_empty() || logprobs.is_some() {
                        outputs.push(RequestOutput::TokenDelta {
                            token_ids,
                            logprobs,
                        });
                    }
                }
                DecodedTextEvent::Done {
                    reason,
                    stop_reason,
                    prompt_tokens,
                    output_tokens,
                    cached_tokens,
                } => {
                    // Release the text the parsers hold back and close the open block
                    // before the terminal output.
                    for event in self.parsers.finish().map_err(failed)? {
                        outputs.extend(self.blocks.push(event).map_err(failed)?);
                    }
                    outputs.extend(self.blocks.finish());
                    self.visible(&outputs);

                    let saturate = |count: usize| u32::try_from(count).unwrap_or(u32::MAX);
                    outputs.push(RequestOutput::Finished {
                        reason,
                        stop_reason,
                        usage: RequestUsage {
                            prompt_tokens: saturate(prompt_tokens),
                            output_tokens: saturate(output_tokens),
                            cached_tokens: saturate(cached_tokens),
                            timings: self.clock.timings(),
                        },
                    });
                    self.ended = true;
                }
                DecodedTextEvent::Rejected { kind, message } => {
                    outputs.push(RequestOutput::Rejected { kind, message });
                }
                DecodedTextEvent::Failed { message } => {
                    outputs.push(RequestOutput::Failed { message });
                }
            }
        }

        for output in &outputs {
            self.stats.observe(output);
        }
        Ok(outputs)
    }

    /// Starts the clock's first visible output at the first of `outputs` a client sees:
    /// text, reasoning, or a tool call.
    fn visible(&mut self, outputs: &[RequestOutput]) {
        let visible = outputs.iter().any(|output| {
            matches!(
                output,
                RequestOutput::TextDelta { .. }
                    | RequestOutput::ReasoningDelta { .. }
                    | RequestOutput::ToolCallStart { .. }
                    | RequestOutput::ToolCallArgumentsDelta { .. }
            )
        });
        if visible {
            self.clock.visible();
        }
    }

    /// What input processing stated for the request, with the stop strings still to match:
    /// what a successor rebuilds this processor from.
    pub fn lowering(&self) -> handoff::Lowering {
        handoff::Lowering {
            prompt_token_ids: self.decode.prompt_token_ids().to_vec(),
            decode_options: Some(self.decode.options().clone()),
            prompt_logprobs: self.prompt_logprobs,
            logprobs: self.logprobs,
            parser: Some(self.parser.clone()),
            response: Some(self.response.clone()),
            compile_us: self.clock.compile_us(),
            started_unix_us: self.clock.started_unix_us(),
        }
    }

    /// The incremental detokenizer's state and its stop-string decisions.
    pub fn decode_state(&self) -> handoff::Decode {
        self.decode.state()
    }

    /// The reasoning and tool-call parsers' state; `None` when a parser cannot be continued,
    /// in which case the response cannot be handed over.
    pub fn parser_state(&self) -> Option<handoff::Processor> {
        self.parsers.state()
    }

    /// The assembled blocks and the timings recorded so far.
    pub fn assembly(&self) -> handoff::Assembly {
        let mut assembly = self.blocks.state();
        self.clock.state(&mut assembly);
        assembly
    }
}
