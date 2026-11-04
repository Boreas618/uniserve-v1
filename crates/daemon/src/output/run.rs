//! A streamed chat completion driven one engine event at a time, so that a
//! hot restart can take it over (design `specs/worker.md` §7.5.1).
//!
//! [`generation_output`](super::assembly::generation_output) pulls the same
//! pipeline through one `try_stream`; a handed-over response cannot be a
//! generator, because a restart boundary is a place to stop: between two
//! engine events, with every frame of the events consumed so far produced.
//! [`GenerationRun`] therefore pushes the pipeline: it takes one event from
//! the request's engine stream, decodes it, runs it through the output
//! processors and the block assembly, turns the resulting outputs into
//! server-sent event frames, and returns.
//!
//! Its state is the whole of what a successor goes on writing from: the
//! incremental decoder, the parsers, the assembled blocks, the stream clock,
//! the image counters, the chunk layer, and the request's identity in the
//! engine with the event position this process consumed. The successor takes
//! the request's engine stream over from that position
//! (`uniserve.request.Command.Takeover`) instead of submitting the request
//! again, so no token is computed twice.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures::FutureExt as _;
use proto::handoff;
use anyhow::Context as _;
use thiserror_ext::AsReport as _;
use tracing::{error, trace};

use crate::http::stream::{Application, Session, Step};
use crate::metrics::stats::RequestOutcome;
use crate::migration::{RequestEvents, Step as EventStep};
use crate::openai::chat_completions::ChatStreamEmitter;
use crate::openai::error::ApiError;
use crate::openai::chat_completions::ChatResponseContext;
use crate::output::assembly::StreamClock;
use crate::output::chat::structured::StructuredEventState;
use crate::output::chat::ChatOutputProcessor;
use crate::output::text::{DecodeStream, DecodedTextEvent, ImageEvent};
use crate::output::text::TextDecodeOptions;
use crate::output::{OutputError, RequestOutput, RequestUsage};
use crate::tokenizer::DynTokenizer;

/// The application part of one streamed chat completion's state: what the
/// successor rebuilds the pipeline from.
pub(crate) struct ChatOutputInputs {
    /// The model's tokenizer.
    pub(crate) tokenizer: DynTokenizer,
    /// The prompt token identifiers the request was lowered to.
    pub(crate) prompt_token_ids: Vec<u32>,
    /// Incremental detokenization and stop strings.
    pub(crate) decode: TextDecodeOptions,
    /// Whether prompt log-probabilities were requested.
    pub(crate) prompt_logprobs: bool,
    /// Whether generated-token log-probabilities were requested.
    pub(crate) logprobs: bool,
    /// The request's reasoning and tool-call parsers.
    pub(crate) processor: ChatOutputProcessor,
    /// The parser set the request's model family chose, which the process
    /// that continues the response rebuilds the same processors from.
    pub(crate) parser: handoff::Parser,
    /// The response's OpenAI fields.
    pub(crate) context: ChatResponseContext,
}

/// One streamed chat completion's body, driven one engine event at a time.
pub(crate) struct GenerationRun {
    request_id: String,
    /// The response timestamp every chunk repeats, in seconds since the UNIX
    /// epoch.
    created: u64,
    tokenizer: DynTokenizer,
    prompt_token_ids: Vec<u32>,
    prompt_logprobs: bool,
    logprobs: bool,
    decode: DecodeStream,
    processor: ChatOutputProcessor,
    parser: handoff::Parser,
    blocks: StructuredEventState,
    clock: StreamClock,
    emitter: ChatStreamEmitter,
    events: RequestEvents,
    image_count: u32,
    image_steps: u32,
    /// The response's OpenAI fields, which a successor rebuilds the emitter
    /// and the chunks from.
    context: ChatResponseContext,
    /// The request's statistics, recorded at its terminal output.
    outcome: RequestOutcome,
    /// The engine replica that serves the request.
    replica: u32,
    /// Whether the decoder's terminal event was consumed.
    ended: bool,
}

impl GenerationRun {
    /// The body of `request_id`, whose pipeline is `inputs` and whose engine
    /// events arrive on `events`, which the caller submitted.
    ///
    /// # Errors
    ///
    /// Fails as [`DecodeStream::new`] does.
    pub(crate) fn new(
        request_id: String,
        compile_us: u64,
        inputs: ChatOutputInputs,
        events: RequestEvents,
        outcome: RequestOutcome,
    ) -> Result<Self, crate::output::text::Error> {
        let ChatOutputInputs {
            tokenizer,
            prompt_token_ids,
            decode,
            prompt_logprobs,
            logprobs,
            processor,
            parser,
            context,
        } = inputs;
        let stream = DecodeStream::new(
            request_id.clone(),
            tokenizer.clone(),
            prompt_token_ids.clone(),
            prompt_logprobs,
            logprobs,
            decode,
        )?;
        let created = (unix_us() / 1_000_000) as u64;
        let started = outcome.started();
        Ok(Self {
            request_id,
            created,
            tokenizer,
            prompt_token_ids,
            prompt_logprobs,
            logprobs,
            decode: stream,
            processor,
            parser,
            blocks: StructuredEventState::new(),
            clock: StreamClock::new(started, compile_us),
            emitter: ChatStreamEmitter::new(&context, created),
            context,
            events,
            replica: outcome.engine(),
            outcome,
            image_count: 0,
            image_steps: 0,
            ended: false,
        })
    }

    /// The frames of one engine event's outputs.
    ///
    /// An output the response fails on ends it: the error event and the
    /// terminal `[DONE]` follow the frames produced before it, as the `Sse`
    /// path wrote them.
    fn push(&mut self, event: proto::request::Event) -> Result<Vec<Bytes>, OutputError> {
        let decoded = self
            .decode
            .push(event, &mut self.events)
            .map_err(|source| OutputError {
                request_id: self.request_id.clone(),
                source: source.into(),
            })?;
        let mut frames = Vec::new();
        for event in decoded.events {
            for output in self.outputs(event)? {
                match self.frames(output) {
                    Ok(output_frames) => frames.extend(output_frames),
                    Err(error) => {
                        frames.push(error_frame(&error));
                        frames.push(done_frame());
                        self.ended = true;
                        return Ok(frames);
                    }
                }
            }
        }
        Ok(frames)
    }

    /// The outputs one decoded text event produces, as
    /// [`generation_output`](super::assembly::generation_output) produces
    /// them.
    fn outputs(&mut self, event: DecodedTextEvent) -> Result<Vec<RequestOutput>, OutputError> {
        let failed = |source: crate::output::chat::Error| OutputError {
            request_id: self.request_id.clone(),
            source,
        };
        Ok(match event {
            DecodedTextEvent::Start {
                prompt,
                queued_at,
                scheduled_at,
            } => {
                self.processor.initialize(&prompt.token_ids);
                self.clock.scheduled(queued_at, scheduled_at);
                vec![
                    RequestOutput::Accepted(prompt),
                    RequestOutput::Scheduled {
                        queued_at,
                        scheduled_at,
                    },
                ]
            }
            DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
            } => {
                let mut outputs = Vec::new();
                for event in self.processor.push(delta).map_err(failed)? {
                    outputs.extend(self.blocks.push(event).map_err(failed)?);
                }
                if !token_ids.is_empty() || logprobs.is_some() {
                    outputs.push(RequestOutput::TokenDelta {
                        token_ids,
                        logprobs,
                    });
                }
                outputs
            }
            DecodedTextEvent::Image(ImageEvent::Begin) => {
                self.clock.visible();
                Vec::new()
            }
            DecodedTextEvent::Image(ImageEvent::Step { image_id }) => {
                self.image_steps = self.image_steps.saturating_add(1);
                vec![RequestOutput::ImageStep { image_id }]
            }
            DecodedTextEvent::Image(ImageEvent::Done(image)) => {
                self.image_count = self.image_count.saturating_add(1);
                vec![RequestOutput::ImageDone(image)]
            }
            DecodedTextEvent::Done(finished) => {
                // Release the text the parsers hold back and close the open
                // block before the terminal event.
                let mut outputs = Vec::new();
                for event in self.processor.finish().map_err(failed)? {
                    outputs.extend(self.blocks.push(event).map_err(failed)?);
                }
                outputs.extend(self.blocks.finish());
                let saturate = |count: usize| u32::try_from(count).unwrap_or(u32::MAX);
                outputs.push(RequestOutput::Finished {
                    reason: finished.reason,
                    usage: RequestUsage {
                        prompt_tokens: saturate(finished.prompt_tokens),
                        output_tokens: saturate(finished.output_tokens),
                        image_count: self.image_count.max(saturate(finished.images)),
                        image_steps: self.image_steps,
                        timings: self.clock.timings(),
                    },
                });
                self.ended = true;
                outputs
            }
            DecodedTextEvent::Rejected { kind, message } => {
                vec![RequestOutput::Rejected { kind, message }]
            }
            DecodedTextEvent::Failed { message } => vec![RequestOutput::Failed { message }],
        })
    }

    /// The frames one public output produces: its chunk's server-sent event,
    /// and, at the end of the response, the terminal `[DONE]` event.
    fn frames(&mut self, output: RequestOutput) -> Result<Vec<Bytes>, ApiError> {
        self.outcome.observe(&output);
        let visible = is_visible(&output);
        let chunks = self.emitter.push(&output)?;
        if visible {
            self.clock.visible();
        }
        let mut frames: Vec<Bytes> = chunks.iter().map(chunk_frame).collect();
        if matches!(output, RequestOutput::Finished { .. }) {
            frames.push(done_frame());
        }
        Ok(frames)
    }

    /// Continues the response `state` describes, whose engine events arrive
    /// on `events`, taken over from the connection that consumed them
    /// (design §7.5.1 step 4): the successor rebuilds the pipeline the state
    /// states from its own tokenizer and model family, then resumes every
    /// stage where the predecessor stopped.
    ///
    /// # Errors
    ///
    /// Fails when the state names no response or no prompt, or when the
    /// family's parsers cannot be built.
    pub(crate) fn resume(
        state: &handoff::ChatStream,
        tokenizer: DynTokenizer,
        events: RequestEvents,
        outcome: RequestOutcome,
    ) -> anyhow::Result<Self> {
        let lowering = state
            .lowering
            .as_ref()
            .context("a handed response states no lowering")?;
        let response = lowering
            .response
            .as_ref()
            .context("a handed response states no response fields")?;
        let context = ChatResponseContext::from(response);
        let request_id = context.request_id.clone();
        let options = lowering
            .decode_options
            .as_ref()
            .map(decode_options)
            .unwrap_or_default();
        let parser = lowering.parser.clone().unwrap_or_default();
        let mut decode_options = options.clone();
        let kind = handoff::parser::Kind::try_from(parser.kind)
            .unwrap_or(handoff::parser::Kind::ParserKindNone);
        let mut processor = match kind {
            handoff::parser::Kind::ParserKindQwen3 => ChatOutputProcessor::qwen3(
                parser.parse_tools.then_some(&[][..]),
                tokenizer.clone(),
                parser.parse_reasoning,
            )?,
            handoff::parser::Kind::ParserKindGemma4 => ChatOutputProcessor::gemma4(
                parser.parse_tools,
                &mut decode_options,
                tokenizer.clone(),
                parser.parse_reasoning,
            )?,
            handoff::parser::Kind::ParserKindSenseNova
            | handoff::parser::Kind::ParserKindNone => ChatOutputProcessor::new(None, None),
        };
        if let Some(parsers) = state.processor.as_ref() {
            processor.resume(parsers);
        }
        let decode = DecodeStream::resume(
            request_id.clone(),
            tokenizer.clone(),
            lowering.prompt_token_ids.clone(),
            lowering.prompt_logprobs,
            lowering.logprobs,
            options,
            state.decode.as_ref().unwrap_or(&Default::default()),
        )
        .context("resuming the incremental decoder")?;
        let assembly = state.assembly.clone().unwrap_or_default();
        let mut blocks = StructuredEventState::new();
        blocks.resume(&assembly);
        let mut emitter = ChatStreamEmitter::new(&context, state.chunks.as_ref().map_or(0, |chunks| chunks.created));
        if let Some(chunks) = state.chunks.as_ref() {
            emitter.resume(chunks);
        }
        let created = state.chunks.as_ref().map_or(0, |chunks| chunks.created);
        let replica = outcome.engine();
        Ok(Self {
            request_id,
            created,
            tokenizer,
            prompt_token_ids: lowering.prompt_token_ids.clone(),
            prompt_logprobs: lowering.prompt_logprobs,
            logprobs: lowering.logprobs,
            decode,
            processor,
            parser,
            blocks,
            clock: StreamClock::resume(&assembly, lowering.started_unix_us, lowering.compile_us),
            emitter,
            events,
            image_count: assembly.images,
            image_steps: assembly.image_steps,
            context,
            replica,
            outcome,
            ended: false,
        })
    }

    /// The state a successor goes on writing from, or `None` when this
    /// response cannot be handed over.
    pub(crate) fn freeze_state(&self) -> Option<handoff::ChatStream> {
        let processor = self.processor.state()?;
        let chunks = self.emitter.state()?;
        let mut assembly = self.blocks.state();
        self.clock.state(&mut assembly);
        assembly.images = self.image_count;
        assembly.image_steps = self.image_steps;
        let (consumed_events, consumed_tokens) = self.events.progress();
        Some(handoff::ChatStream {
            request: self.events.submitted().unwrap_or_default(),
            lowering: Some(handoff::Lowering {
                prompt_token_ids: self.prompt_token_ids.clone(),
                decode_options: Some(decode_state(&self.decode.options())),
                prompt_logprobs: self.prompt_logprobs,
                logprobs: self.logprobs,
                parser: Some(self.parser.clone()),
                response: Some(handoff::ChatResponseOptions::from(&self.context)),
                compile_us: self.clock.compile_us(),
                started_unix_us: self.clock.started_unix_us(),
            }),
            decode: Some(self.decode.state()),
            processor: Some(processor),
            assembly: Some(assembly),
            chunks: Some(chunks),
            consumed_events,
            consumed_tokens,
            usage_owed: self.context.include_usage && !self.emitter.done(),
            engine_request_id: self
                .events
                .engine_request_id()
                .map(|id| id.0.to_string())
                .unwrap_or_default(),
            engine_replica: self.replica,
        })
    }
}

impl Session for GenerationRun {
    fn poll_step(&mut self, cx: &mut Context<'_>) -> Poll<Step> {
        loop {
            if self.ended {
                return Poll::Ready(Step::Ended);
            }
            match self.events.poll_step(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(EventStep::Migrate) => return Poll::Ready(Step::Migrate),
                Poll::Ready(EventStep::Ended) => {
                    // The engine stream ends only after the terminal event,
                    // which the decoder checks.
                    self.ended = true;
                    return Poll::Ready(Step::Ended);
                }
                Poll::Ready(EventStep::Event(event)) => match self.push(event) {
                    Ok(frames) if frames.is_empty() => continue,
                    Ok(frames) => return Poll::Ready(Step::Frames(frames)),
                    Err(error) => {
                        error!(error = %error.as_report(), "a handed-over response failed");
                        return Poll::Ready(Step::Frames(vec![
                            error_frame(&ApiError::server_error("Internal server error")),
                            done_frame(),
                        ]));
                    }
                },
            }
        }
    }

    fn migrate(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(
            self.events
                .migrate_pending()
                .map(|result: Result<(), proto::request::Event>| {
                    if let Err(event) = result {
                        error!(?event, "a handed-over response's migration failed");
                    }
                }),
        )
    }

    fn freeze(&mut self) -> Option<Application> {
        self.freeze_state().map(Application::ChatStream)
    }

    fn abort(&mut self) {
        self.outcome.abandoned();
        self.events.cancel();
    }

    fn detach(&mut self) {
        self.outcome.abandoned();
        self.events.hand_over();
    }
}

/// Whether `output` is visible output: text, reasoning, a tool call, or
/// generated media.
fn is_visible(output: &RequestOutput) -> bool {
    matches!(
        output,
        RequestOutput::TextDelta { .. }
            | RequestOutput::ReasoningDelta { .. }
            | RequestOutput::ToolCallStart { .. }
            | RequestOutput::ToolCallArgumentsDelta { .. }
            | RequestOutput::ImageDone(_)
    )
}

/// One chunk as the server-sent event the connection writes.
fn chunk_frame(chunk: &crate::openai::protocol::ChatCompletionStreamResponse) -> Bytes {
    let payload = serde_json::to_string(chunk).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize chat completion chunk","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "chat completion emitting chunk");
    event_frame(&payload)
}

/// The error event a failed response ends with, as the `Sse` path wrote it.
fn error_frame(error: &ApiError) -> Bytes {
    let payload = serde_json::to_string(&error.to_error_response()).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize error response","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "chat completion emitting error");
    event_frame(&payload)
}

/// One JSON payload as a server-sent `data:` event.
fn event_frame(payload: &str) -> Bytes {
    let payload = payload.replace('\r', "\\r").replace('\n', "\\n");
    Bytes::from(format!("data: {payload}\n\n"))
}

/// The terminal OpenAI sentinel event.
fn done_frame() -> Bytes {
    trace!("chat completion emitting done");
    Bytes::from_static(b"data: [DONE]\n\n")
}

/// The lowerable form of the decode options.
fn decode_state(options: &TextDecodeOptions) -> handoff::TextDecode {
    handoff::TextDecode {
        skip_special_tokens: options.skip_special_tokens,
        include_stop_str_in_output: options.include_stop_str_in_output,
        stop_strings: options.stop_strings.clone().unwrap_or_default(),
        min_tokens: options.min_tokens,
    }
}

/// The decode options a handed state states.
pub(crate) fn decode_options(state: &handoff::TextDecode) -> TextDecodeOptions {
    TextDecodeOptions {
        skip_special_tokens: state.skip_special_tokens,
        include_stop_str_in_output: state.include_stop_str_in_output,
        stop_strings: (!state.stop_strings.is_empty()).then(|| state.stop_strings.clone()),
        min_tokens: state.min_tokens,
    }
}

/// The current UNIX time in microseconds.
fn unix_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_micros() as u64)
        .unwrap_or_default()
}

impl From<&handoff::ChatResponseOptions> for ChatResponseContext {
    fn from(state: &handoff::ChatResponseOptions) -> Self {
        Self {
            request_id: state.request_id.clone(),
            response_model: state.response_model.clone(),
            include_usage: state.include_usage,
            requested_logprobs: state.requested_logprobs,
            include_prompt_logprobs: state.include_prompt_logprobs,
            include_reasoning: state.include_reasoning,
            return_token_ids: state.return_token_ids,
            return_tokens_as_token_ids: state.return_tokens_as_token_ids,
        }
    }
}

impl From<&ChatResponseContext> for handoff::ChatResponseOptions {
    fn from(context: &ChatResponseContext) -> Self {
        Self {
            request_id: context.request_id.clone(),
            response_model: context.response_model.clone(),
            include_usage: context.include_usage,
            requested_logprobs: context.requested_logprobs,
            include_prompt_logprobs: context.include_prompt_logprobs,
            include_reasoning: context.include_reasoning,
            return_token_ids: context.return_token_ids,
            return_tokens_as_token_ids: context.return_tokens_as_token_ids,
            created: 0,
        }
    }
}

/// The instant a response started, from the wall-clock origin a handed state
/// carries: the successor's timings continue the predecessor's.
pub(crate) fn started_instant(started_unix_us: u64) -> Instant {
    let elapsed = unix_us().saturating_sub(started_unix_us);
    Instant::now() - Duration::from_micros(elapsed)
}
