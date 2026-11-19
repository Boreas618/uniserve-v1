//! Output processing of one token request: its engine events become the response's deltas.
//!
//! [`OutputProcessor`] is what a request's response is produced through, as vLLM's
//! `OutputProcessor` holds a request's state: it checks the engine's event order and
//! detokenizes incrementally with the stop strings (`text`), parses reasoning and tool calls
//! in the formats the request's model family writes through the request's instance of the
//! epoch's processing program (`chat`), assembles the content blocks (`chat::structured`),
//! keeps the timings the terminal usage reports, and records the request's statistics. A
//! route's response body pushes one engine event at a time and writes the [`RequestOutput`]s it
//! gets back.
//!
//! Everything it is built from is the lowering input processing stated for the request
//! (`proto::handoff::Lowering`) and the parser the program opened for it, whose state is the
//! program's opaque bytes, so a successor daemon rebuilds the same processor from a
//! handed-over response's state and resumes every stage where its predecessor stopped
//! ([`OutputProcessor::resume`]).

use std::sync::Arc;

use proto::handoff;
use proto::request::Event;
use service::span::{self, Stage};

use crate::api::error::Error;
use crate::api::events::{Events, Stats};
use crate::api::output::assembly::StreamClock;
use crate::api::output::chat::ChatParser;
use crate::api::output::chat::structured::StructuredEventState;
use crate::api::output::text::{DecodeStream, DecodedTextEvent};
use crate::api::output::{OutputError, RequestOutput, RequestUsage};
use crate::api::program::{Binding, Session};
use crate::api::tokenizer::DynTokenizer;

/// The output processing of one token request.
pub struct OutputProcessor {
    /// The response's OpenAI fields, which also name the request in errors.
    response: handoff::ChatResponseOptions,
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
    /// The request's trace id, which the spans of its output name; zero for one traced nowhere.
    trace: u64,
}

/// The parts of a lowering every token response is built from, each checked present once.
struct Parts {
    response: handoff::ChatResponseOptions,
    decode: handoff::TextDecode,
}

impl Parts {
    /// The parts `lowering` states.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Lowering`] when it leaves one out.
    fn of(lowering: &handoff::Lowering) -> Result<Self, Error> {
        let missing = |part| Error::Lowering { part };
        Ok(Self {
            response: lowering
                .response
                .clone()
                .ok_or_else(|| missing("response fields"))?,
            decode: lowering
                .decode_options
                .clone()
                .ok_or_else(|| missing("decode options"))?,
        })
    }
}

impl OutputProcessor {
    /// The output processor of the request `lowering` describes, before any engine event, whose
    /// text is parsed in `session`, the request's instance of the processing program.
    ///
    /// `compile_us` is how long the request took to reach the engine, in microseconds, and `stats`
    /// records the request's statistics; the instant it started at is the origin of the timings
    /// the response reports.
    ///
    /// # Errors
    ///
    /// Fails when the lowering leaves out a part every response states or the prompt is empty.
    pub fn new(
        tokenizer: DynTokenizer,
        lowering: handoff::Lowering,
        session: Session,
        compile_us: u64,
        stats: Box<dyn Stats>,
    ) -> Result<Self, OutputError> {
        let request_id = lowering
            .response
            .as_ref()
            .map(|response| response.request_id.clone())
            .unwrap_or_default();
        let failed = |source| OutputError {
            request_id: request_id.clone(),
            source,
        };
        let Parts { response, decode } = Parts::of(&lowering).map_err(failed)?;

        let parsers = ChatParser::new(session);
        let decode = DecodeStream::new(
            response.request_id.clone(),
            tokenizer,
            lowering.prompt_token_ids,
            lowering.prompt_logprobs,
            lowering.logprobs,
            decode,
        )
        .map_err(failed)?;
        let clock = StreamClock::new(stats.started(), compile_us);

        Ok(Self {
            response,
            prompt_logprobs: lowering.prompt_logprobs,
            logprobs: lowering.logprobs,
            decode,
            parsers,
            blocks: StructuredEventState::new(),
            clock,
            stats,
            ended: false,
            trace: 0,
        })
    }

    /// Continues the output processing a handed-over response states: the processor is
    /// rebuilt from `lowering` on this process's `tokenizer`, its parser is restored in an
    /// instance of `binding` (the family this process binds the request's checkpoint to in the
    /// same processing program), and its decoder, parser and block assembly resume from
    /// `decode`, `parsers` and `assembly`.
    ///
    /// # Errors
    ///
    /// Fails as [`Self::new`] does, and when a state does not fit the processor the lowering
    /// builds or the program `binding` is of.
    pub fn resume(
        tokenizer: DynTokenizer,
        binding: &Arc<Binding>,
        lowering: &handoff::Lowering,
        decode: &handoff::Decode,
        parsers: &handoff::Processor,
        assembly: &handoff::Assembly,
        stats: Box<dyn Stats>,
    ) -> Result<Self, OutputError> {
        let request_id = lowering
            .response
            .as_ref()
            .map(|response| response.request_id.clone())
            .unwrap_or_default();
        let failed = |source| OutputError {
            request_id: request_id.clone(),
            source,
        };
        let Parts {
            response,
            decode: options,
        } = Parts::of(lowering).map_err(failed)?;

        let chat = ChatParser::resume(binding, parsers).map_err(failed)?;
        let decode = DecodeStream::resume(
            response.request_id.clone(),
            tokenizer,
            lowering.prompt_token_ids.clone(),
            lowering.prompt_logprobs,
            lowering.logprobs,
            options,
            decode,
        )
        .map_err(failed)?;
        let mut blocks = StructuredEventState::new();
        blocks.resume(assembly);

        Ok(Self {
            response,
            prompt_logprobs: lowering.prompt_logprobs,
            logprobs: lowering.logprobs,
            decode,
            parsers: chat,
            blocks,
            clock: StreamClock::resume(assembly),
            stats,
            ended: false,
            // A successor's processor traces nothing: the trace id stayed with the daemon that
            // received the request.
            trace: 0,
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
        let failed = |source| OutputError {
            request_id: request_id.clone(),
            source,
        };
        // Decoding one event's tokens, and parsing them into the response's blocks.
        let _detokenize = (self.trace != 0).then(|| span::enter(Stage::Detokenize, self.trace));
        let decoded = self.decode.push(event, events).map_err(failed)?;

        let mut outputs = Vec::new();
        for event in decoded {
            match event {
                DecodedTextEvent::Start {
                    prompt,
                    queued_at,
                    scheduled_at,
                } => {
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
                    for event in self.parsers.push(&delta).map_err(failed)? {
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
            response: Some(self.response.clone()),
        }
    }

    /// The processor of a request of trace id `trace`, whose output spans name it.
    #[must_use]
    pub fn traced(mut self, trace: u64) -> Self {
        self.trace = trace;
        self
    }

    /// The request's trace id; zero for one traced nowhere.
    pub fn trace(&self) -> u64 {
        self.trace
    }

    /// The incremental detokenizer's state and its stop-string decisions.
    pub fn decode_state(&self) -> handoff::Decode {
        self.decode.state()
    }

    /// The parser's state, as the processing program saves it; `None` when the program's
    /// instance ended, so the response cannot be handed over.
    pub fn parser_state(&mut self) -> Option<handoff::Processor> {
        self.parsers.state().ok()
    }

    /// The assembled blocks and the timings recorded so far.
    pub fn assembly(&self) -> handoff::Assembly {
        let mut assembly = self.blocks.state();
        self.clock.state(&mut assembly);
        assembly
    }
}
