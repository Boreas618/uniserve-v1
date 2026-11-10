//! A streamed chat completion driven one engine event at a time, so that a
//! hot restart can take it over.
//!
//! A handed-over response cannot be a generator, because a restart boundary
//! is a place to stop: between two engine events, with every frame of the
//! events consumed so far produced. [`GenerationRun`] therefore pushes the
//! pipeline: it takes one event from the request's engine stream, decodes
//! it, runs it through the output processors and the block assembly, turns
//! the resulting outputs into its sink's frames (server-sent events, or one
//! JSON document), and returns. It is the one output pipeline of a token
//! request, streamed or not.
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
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use bytes::Bytes;
use futures::FutureExt as _;
use proto::handoff;
use thiserror_ext::AsReport as _;
use tracing::{error, trace};

use crate::body::{Application, Session, Step};
use crate::events::Outcome;
use crate::events::{Events, Next as EventStep};
use crate::openai::chat_completions::ChatResponseContext;
use crate::openai::chat_completions::ChatStreamEmitter;
use crate::openai::error::ApiError;
use crate::output::assembly::StreamClock;
use crate::output::chat::ChatOutputProcessor;
use crate::output::chat::structured::StructuredEventState;
use crate::output::text::TextDecodeOptions;
use crate::output::text::{DecodeStream, DecodedTextEvent, ImageEvent};
use crate::output::{OutputError, RequestOutput, RequestUsage};
use crate::tokenizer::DynTokenizer;

/// The application part of one streamed chat completion's state: what the
/// successor rebuilds the pipeline from.
pub struct ChatOutputInputs {
    /// The model's tokenizer.
    pub tokenizer: DynTokenizer,
    /// The prompt token identifiers the request was lowered to.
    pub prompt_token_ids: Vec<u32>,
    /// Incremental detokenization and stop strings.
    pub decode: TextDecodeOptions,
    /// Whether prompt log-probabilities were requested.
    pub prompt_logprobs: bool,
    /// Whether generated-token log-probabilities were requested.
    pub logprobs: bool,
    /// The request's reasoning and tool-call parsers.
    pub processor: ChatOutputProcessor,
    /// The parser set the request's model family chose, which the process
    /// that continues the response rebuilds the same processors from.
    pub parser: handoff::Parser,
    /// The response's OpenAI fields.
    pub context: ChatResponseContext,
}

/// The parts of a handed state that every response body shares.
pub struct ResumeState<'a> {
    /// How input processing lowered the request.
    pub lowering: Option<&'a handoff::Lowering>,
    /// The incremental detokenizer's state.
    pub decode: Option<&'a handoff::Decode>,
    /// The output parsers' state.
    pub processor: Option<&'a handoff::Processor>,
    /// The assembled blocks.
    pub assembly: Option<&'a handoff::Assembly>,
    /// The response timestamp every chunk repeats.
    pub created: u64,
}

impl<'a> ResumeState<'a> {
    /// The shared parts of a streamed response's state.
    pub fn of_chat_stream(state: &'a handoff::ChatStream) -> Self {
        Self {
            lowering: state.lowering.as_ref(),
            decode: state.decode.as_ref(),
            processor: state.processor.as_ref(),
            assembly: state.assembly.as_ref(),
            created: state.chunks.as_ref().map_or(0, |chunks| chunks.created),
        }
    }

    /// The shared parts of a response answered as one body.
    pub fn of_chat_response(state: &'a handoff::ChatResponse) -> Self {
        Self {
            lowering: state.lowering.as_ref(),
            decode: state.decode.as_ref(),
            processor: state.processor.as_ref(),
            assembly: state.assembly.as_ref(),
            created: state
                .lowering
                .as_ref()
                .and_then(|lowering| lowering.response.as_ref())
                .map_or(0, |response| response.created),
        }
    }
}

impl GenerationRun<ChatStreamEmitter> {
    /// Continues the streamed response `state` describes, whose engine events
    /// arrive on `events`, taken over from the connection that consumed them
    /// (design §7.5.1 step 4).
    ///
    /// # Errors
    ///
    /// Fails when the state names no response or no prompt, or when the
    /// family's parsers cannot be built.
    pub fn resume_stream(
        state: &handoff::ChatStream,
        tokenizer: DynTokenizer,
        events: Box<dyn Events>,
        outcome: Box<dyn Outcome>,
    ) -> anyhow::Result<GenerationRun<ChatStreamEmitter>> {
        let parts = ResumeState::of_chat_stream(state);
        let created = state.chunks.as_ref().map_or(0, |chunks| chunks.created);
        GenerationRun::<ChatStreamEmitter>::resume(parts, tokenizer, events, outcome, |context| {
            let mut emitter = ChatStreamEmitter::new(context, created);
            if let Some(chunks) = state.chunks.as_ref() {
                emitter.resume(chunks);
            }
            emitter
        })
    }
}

impl GenerationRun<ChatBodySink> {
    /// Continues the response answered as one JSON body that `state`
    /// describes, whose engine events arrive on `events`.
    ///
    /// # Errors
    ///
    /// Fails when the state names no response or no prompt, or when the
    /// family's parsers cannot be built.
    pub fn resume_collection(
        state: &handoff::ChatResponse,
        tokenizer: DynTokenizer,
        events: Box<dyn Events>,
        outcome: Box<dyn Outcome>,
    ) -> anyhow::Result<GenerationRun<ChatBodySink>> {
        let parts = ResumeState::of_chat_response(state);
        let created = parts.created;
        GenerationRun::<ChatBodySink>::resume(parts, tokenizer, events, outcome, |context| {
            let mut sink = ChatBodySink::new(context, created);
            if let Some(collection) = state.collection.as_ref() {
                sink.resume(collection);
            }
            sink
        })
    }
}

/// What a response body's outputs become: the frames it writes.
pub trait ResponseSink: Send + 'static {
    /// The frames `output` produces, in order.
    ///
    /// # Errors
    ///
    /// Fails as the response does; the body then ends with its error frame.
    fn push(&mut self, output: &RequestOutput) -> Result<Vec<Bytes>, ApiError>;

    /// Whether the request's terminal output arrived.
    fn done(&self) -> bool;

    /// The sink's own state, as the response's application states it.
    fn body_state(&self) -> Option<BodyState>;

    /// The frames this response writes when it fails, in the sink's own shape:
    /// server-sent events for a streamed response, the one document a
    /// one-body response answers with.
    fn failed(&mut self, error: &ApiError) -> Vec<Bytes>;

    /// Whether the response's head waits for the body's first output, because
    /// the body is one document whose status is its outcome.
    fn holds_head(&self) -> bool;
}

/// The body's own state, which one response's application carries.
pub enum BodyState {
    /// The chunks and server-sent events of a streamed response.
    Stream(handoff::ChatChunks),
    /// What a response answered as one body collected.
    Collection(handoff::ChatCollection),
}

impl ResponseSink for ChatStreamEmitter {
    fn push(&mut self, output: &RequestOutput) -> Result<Vec<Bytes>, ApiError> {
        let chunks = ChatStreamEmitter::push(self, output)?;
        let mut frames: Vec<Bytes> = chunks.iter().map(chunk_frame).collect();
        if matches!(output, RequestOutput::Finished { .. }) {
            frames.push(done_frame());
        }
        Ok(frames)
    }

    fn done(&self) -> bool {
        ChatStreamEmitter::done(self)
    }

    fn body_state(&self) -> Option<BodyState> {
        ChatStreamEmitter::state(self).map(BodyState::Stream)
    }

    fn failed(&mut self, error: &ApiError) -> Vec<Bytes> {
        vec![error_frame(error), done_frame()]
    }

    fn holds_head(&self) -> bool {
        // Server-sent events start at once: the stream's own error event is
        // what a failure looks like inside a response already begun.
        false
    }
}

/// A response answered as one JSON body, collected output by output.
pub struct ChatBodySink {
    collector: crate::openai::chat_completions::ChatCollector,
    context: ChatResponseContext,
    created: u64,
    /// Set once the body was written, so a later output writes nothing.
    written: bool,
}

impl ChatBodySink {
    /// The sink of the response `context` describes, whose timestamp is
    /// `created`.
    pub fn new(context: &ChatResponseContext, created: u64) -> Self {
        Self {
            collector: crate::openai::chat_completions::ChatCollector::new(),
            context: context.clone(),
            created,
            written: false,
        }
    }

    /// Takes over a state this sink produced.
    pub fn resume(&mut self, state: &handoff::ChatCollection) {
        self.collector.resume(state);
    }
}

impl ResponseSink for ChatBodySink {
    fn push(&mut self, output: &RequestOutput) -> Result<Vec<Bytes>, ApiError> {
        self.collector.push(output)?;
        if !self.collector.done() || self.written {
            return Ok(Vec::new());
        }
        self.written = true;
        let response = crate::openai::chat_completions::chat_response(
            self.collector.collected()?,
            &self.context,
            self.created,
        )?;
        let body = serde_json::to_vec(&response).unwrap_or_else(|_| {
            br#"{"error":{"message":"failed to serialize chat completion","type":"server_error"}}"#
                .to_vec()
        });
        Ok(vec![Bytes::from(body)])
    }

    fn done(&self) -> bool {
        self.written
    }

    fn body_state(&self) -> Option<BodyState> {
        Some(BodyState::Collection(self.collector.state()))
    }

    fn failed(&mut self, error: &ApiError) -> Vec<Bytes> {
        // The document `ApiError`'s own response writes, so a failed
        // generation answers what the route answered before this response
        // became a body of the handoff path.
        let body = serde_json::to_vec(&error.to_error_response()).unwrap_or_else(|_| {
            br#"{"error":{"message":"failed to serialize error response","type":"server_error"}}"#
                .to_vec()
        });
        vec![Bytes::from(body)]
    }

    fn holds_head(&self) -> bool {
        // One document, whose status is the outcome: nothing is gained by
        // sending its head before the document exists, and a head that went
        // out cannot state a failure (§7.5.1 step 4 hands such a response over
        // before it has started writing).
        true
    }
}

/// One streamed chat completion's body, driven one engine event at a time.
pub struct GenerationRun<S: ResponseSink> {
    request_id: String,
    /// The response timestamp every chunk repeats, in seconds since the UNIX
    /// epoch.
    created: u64,
    prompt_token_ids: Vec<u32>,
    prompt_logprobs: bool,
    logprobs: bool,
    decode: DecodeStream,
    processor: ChatOutputProcessor,
    parser: handoff::Parser,
    blocks: StructuredEventState,
    clock: StreamClock,
    sink: S,
    events: Box<dyn Events>,
    image_count: u32,
    image_steps: u32,
    /// The response's OpenAI fields, which a successor rebuilds the emitter
    /// and the chunks from.
    context: ChatResponseContext,
    /// The request's statistics, recorded at its terminal output.
    outcome: Box<dyn Outcome>,
    /// The engine replica that serves the request.
    replica: u32,
    /// Whether the decoder's terminal event was consumed.
    ended: bool,
}

impl<S: ResponseSink> GenerationRun<S> {
    /// The body of `request_id`, whose pipeline is `inputs` and whose engine
    /// events arrive on `events`, which the caller submitted.
    ///
    /// # Errors
    ///
    /// Fails as [`DecodeStream::new`] does.
    pub fn new(
        request_id: String,
        compile_us: u64,
        inputs: ChatOutputInputs,
        events: Box<dyn Events>,
        outcome: Box<dyn Outcome>,
        sink: S,
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
            Arc::clone(&tokenizer),
            prompt_token_ids.clone(),
            prompt_logprobs,
            logprobs,
            decode,
        )?;
        let created = unix_us() / 1_000_000;
        let started = outcome.started();
        Ok(Self {
            request_id,
            created,
            prompt_token_ids,
            prompt_logprobs,
            logprobs,
            decode: stream,
            processor,
            parser,
            blocks: StructuredEventState::new(),
            clock: StreamClock::new(started, compile_us),
            sink,
            context,
            events,
            replica: outcome.engine(),
            outcome,
            image_count: 0,
            image_steps: 0,
            ended: false,
        })
    }

    /// The step one engine event's outputs produce: their frames, which are
    /// none while the sink waits for more.
    ///
    /// An output the response fails on ends it in the sink's own shape
    /// ([`Self::failed`]): for a stream, the error event and the terminal
    /// `[DONE]` after the frames produced before it; for one document, the
    /// failure's status and error document.
    fn push(&mut self, event: proto::request::Event) -> Result<Step, OutputError> {
        let decoded = self
            .decode
            .push(event, self.events.as_mut())
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
                        self.ended = true;
                        return Ok(match self.failed(error) {
                            Step::Frames(failure) => {
                                frames.extend(failure);
                                Step::Frames(frames)
                            }
                            failed => failed,
                        });
                    }
                }
            }
        }
        Ok(Step::Frames(frames))
    }

    /// The public outputs one decoded text event produces: the prompt and
    /// scheduling facts first, then each update's content-block, tool-call
    /// and image events followed by its `TokenDelta`, then the terminal
    /// event.
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
        let frames = self.sink.push(&output)?;
        if visible {
            self.clock.visible();
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
    /// Rebuilds the stages every response body shares and takes the body's own
    /// stage over from `sink`.
    ///
    /// # Errors
    ///
    /// Fails when the state names no lowering or response fields, no prompt,
    /// or a family whose parsers cannot be built.
    pub fn resume(
        state: ResumeState<'_>,
        tokenizer: DynTokenizer,
        events: Box<dyn Events>,
        outcome: Box<dyn Outcome>,
        sink: impl FnOnce(&ChatResponseContext) -> S,
    ) -> anyhow::Result<Self> {
        let lowering = state
            .lowering
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
        let parser = lowering.parser.unwrap_or_default();
        let kind = handoff::parser::Kind::try_from(parser.kind)
            .unwrap_or(handoff::parser::Kind::ParserKindNone);
        let mut processor = match kind {
            handoff::parser::Kind::ParserKindQwen3 => ChatOutputProcessor::qwen3(
                parser.parse_tools.then_some(&[][..]),
                Arc::clone(&tokenizer),
                parser.parse_reasoning,
            )?,
            handoff::parser::Kind::ParserKindNone => ChatOutputProcessor::new(None, None),
        };
        if let Some(parsers) = state.processor {
            processor.resume(parsers);
        }
        let decode = DecodeStream::resume(
            request_id.clone(),
            Arc::clone(&tokenizer),
            lowering.prompt_token_ids.clone(),
            lowering.prompt_logprobs,
            lowering.logprobs,
            options,
            state.decode.unwrap_or(&Default::default()),
        )
        .context("resuming the incremental decoder")?;
        let assembly = state.assembly.cloned().unwrap_or_default();
        let mut blocks = StructuredEventState::new();
        blocks.resume(&assembly);
        let created = state.created;
        let sink = sink(&context);
        let replica = outcome.engine();
        Ok(Self {
            request_id,
            created,
            prompt_token_ids: lowering.prompt_token_ids.clone(),
            prompt_logprobs: lowering.prompt_logprobs,
            logprobs: lowering.logprobs,
            decode,
            processor,
            parser,
            blocks,
            clock: StreamClock::resume(&assembly, lowering.started_unix_us, lowering.compile_us),
            sink,
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
    pub fn freeze_body(&self) -> Option<handoff::stream::Application> {
        let processor = self.processor.state()?;
        let mut assembly = self.blocks.state();
        self.clock.state(&mut assembly);
        assembly.images = self.image_count;
        assembly.image_steps = self.image_steps;
        let (consumed_events, consumed_tokens) = self.events.progress();
        let lowering = handoff::Lowering {
            prompt_token_ids: self.prompt_token_ids.clone(),
            decode_options: Some(decode_state(self.decode.options())),
            prompt_logprobs: self.prompt_logprobs,
            logprobs: self.logprobs,
            parser: Some(self.parser),
            response: Some(handoff::ChatResponseOptions {
                created: self.created,
                ..handoff::ChatResponseOptions::from(&self.context)
            }),
            compile_us: self.clock.compile_us(),
            started_unix_us: self.clock.started_unix_us(),
        };
        let body = self.sink.body_state()?;
        Some(match body {
            BodyState::Stream(chunks) => {
                handoff::stream::Application::ChatStream(handoff::ChatStream {
                    request: self.events.submitted().unwrap_or_default(),
                    lowering: Some(lowering),
                    decode: Some(self.decode.state()),
                    processor: Some(processor),
                    assembly: Some(assembly),
                    chunks: Some(chunks),
                    consumed_events,
                    consumed_tokens,
                    usage_owed: self.context.include_usage && !self.sink.done(),
                    engine_request_id: self
                        .events
                        .engine_request_id()
                        .map(|id| id.0.to_string())
                        .unwrap_or_default(),
                    engine_replica: self.replica,
                })
            }
            BodyState::Collection(collection) => {
                handoff::stream::Application::ChatResponse(handoff::ChatResponse {
                    request: self.events.submitted().unwrap_or_default(),
                    lowering: Some(lowering),
                    decode: Some(self.decode.state()),
                    processor: Some(processor),
                    assembly: Some(assembly),
                    collection: Some(collection),
                    engine_request_id: self
                        .events
                        .engine_request_id()
                        .map(|id| id.0.to_string())
                        .unwrap_or_default(),
                    consumed_events,
                    consumed_tokens,
                    engine_replica: self.replica,
                })
            }
        })
    }

    /// The step a failed response ends with.
    ///
    /// A response whose head is still held states the failure there, in the
    /// status it answers with; a response already writing reports it inside
    /// the stream, in its sink's own shape. Either way the sink renders it:
    /// server-sent events inside a stream, one error document for the body
    /// that has none yet.
    fn failed(&mut self, error: ApiError) -> Step {
        let frames = self.sink.failed(&error);
        if self.sink.holds_head() {
            Step::Failed { error, frames }
        } else {
            Step::Frames(frames)
        }
    }
}

impl<S: ResponseSink> Session for GenerationRun<S> {
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
                    // which the decoder checks. An engine stream that ends
                    // before it is a failed response, and it is answered as
                    // one rather than left as a body that stops mid-document.
                    self.ended = true;
                    if self.sink.done() {
                        return Poll::Ready(Step::Ended);
                    }
                    // The engine dropped the request rather than answering it,
                    // and its own log is where the reason is. This is the last
                    // place the daemon sees the request at all, so it says
                    // which one and which engine, and the two logs can be read
                    // together.
                    error!(
                        replica = self.replica,
                        engine_request_id = self
                            .events
                            .engine_request_id()
                            .map(|id| id.0.to_string())
                            .unwrap_or_default(),
                        "an engine ended a response before its terminal event"
                    );
                    return Poll::Ready(self.failed(ApiError::server_error(
                        "the response ended before its terminal output",
                    )));
                }
                Poll::Ready(EventStep::Event(event)) => match self.push(event) {
                    Ok(Step::Frames(frames)) if frames.is_empty() => continue,
                    Ok(step) => return Poll::Ready(step),
                    Err(error) => {
                        error!(error = %error.as_report(), "a handed-over response failed");
                        return Poll::Ready(
                            self.failed(ApiError::server_error("Internal server error")),
                        );
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
        let body = self.freeze_body()?;
        Some(match body {
            handoff::stream::Application::ChatStream(chat) => Application::ChatStream(chat),
            handoff::stream::Application::ChatResponse(chat) => Application::ChatResponse(chat),
            handoff::stream::Application::VideoContent(video) => Application::VideoContent(video),
        })
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
pub fn decode_options(state: &handoff::TextDecode) -> TextDecodeOptions {
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
pub fn started_instant(started_unix_us: u64) -> Instant {
    let elapsed = unix_us().saturating_sub(started_unix_us);
    Instant::now() - Duration::from_micros(elapsed)
}

/// A video job's retained artifact streamed as a response body, one chunk of
/// bytes per step so that a hot restart can stop it between two chunks.
pub struct VideoContentRun<M: AsRef<[u8]> + Send + Sync + 'static> {
    /// The job whose retained media the body streams.
    job_id: String,
    /// The artifact's content type.
    content_type: String,
    /// The retained media, which keeps its retained-byte share until the
    /// body ends.
    media: Arc<M>,
    /// Bytes of the body written so far.
    offset: usize,
    /// Whether the body ended.
    ended: bool,
}

impl<M: AsRef<[u8]> + Send + Sync + 'static> VideoContentRun<M> {
    /// The body streaming `media` for the job `job_id`, from its first byte.
    pub fn new(job_id: String, content_type: String, media: Arc<M>) -> Self {
        Self {
            job_id,
            content_type,
            media,
            offset: 0,
            ended: false,
        }
    }

    /// The same body, going on at the byte `offset` the process that handed
    /// it over reached.
    pub fn resumed_at(mut self, offset: usize) -> Self {
        self.offset = offset.min(self.media.as_ref().as_ref().len());
        self
    }

    /// How many bytes the body writes in one frame.
    const CHUNK: usize = 64 * 1024;
}

impl<M: AsRef<[u8]> + Send + Sync + 'static> Session for VideoContentRun<M> {
    fn poll_step(&mut self, _cx: &mut Context<'_>) -> Poll<Step> {
        if self.ended {
            return Poll::Ready(Step::Ended);
        }
        let media = Arc::clone(&self.media);
        let bytes = media.as_ref().as_ref();
        if self.offset >= bytes.len() {
            self.ended = true;
            return Poll::Ready(Step::Ended);
        }
        let count = (bytes.len() - self.offset).min(Self::CHUNK);
        let frame = Bytes::copy_from_slice(&bytes[self.offset..self.offset + count]);
        self.offset += count;
        Poll::Ready(Step::Frames(vec![frame]))
    }

    fn migrate(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }

    fn freeze(&mut self) -> Option<Application> {
        Some(Application::VideoContent(handoff::VideoContent {
            job_id: self.job_id.clone(),
            content_type: self.content_type.clone(),
            offset: self.offset as u64,
        }))
    }

    /// The client is gone: nothing of the artifact was computed for it.
    fn abort(&mut self) {}

    /// A successor owns the body from `offset` on.
    fn detach(&mut self) {}
}

#[cfg(test)]
mod tests {
    use std::task::Context;

    use proto::request::{Event, TokenLogprob};
    use serde_json::Value;

    use super::*;
    use crate::test_support::{ChannelEvents, Unrecorded};

    /// Drives a body to its end as its connection does and answers what it
    /// wrote, with the failure a held head states.
    async fn answered(run: &mut impl Session) -> (Vec<u8>, Option<ApiError>) {
        let mut bytes = Vec::new();
        loop {
            match std::future::poll_fn(|cx| run.poll_step(cx)).await {
                Step::Frames(frames) => frames
                    .iter()
                    .for_each(|frame| bytes.extend_from_slice(frame)),
                Step::Migrate => run.migrate().await,
                Step::Ended => return (bytes, None),
                Step::Failed { error, frames } => return (frames.concat(), Some(error)),
            }
        }
    }

    /// The response options of a request that asks for `logprobs`.
    fn context(logprobs: bool) -> ChatResponseContext {
        ChatResponseContext {
            request_id: "request".to_owned(),
            response_model: "model".to_owned(),
            include_usage: false,
            requested_logprobs: logprobs,
            include_prompt_logprobs: false,
            include_reasoning: true,
            return_token_ids: false,
            return_tokens_as_token_ids: false,
        }
    }

    /// The body of a request whose prompt is `p`, decoded and parsed as
    /// `processor` does, over the engine events `events`, written by `sink`.
    fn run<S: ResponseSink>(
        processor: ChatOutputProcessor,
        decode: TextDecodeOptions,
        logprobs: bool,
        events: Box<dyn Events>,
        sink: S,
    ) -> GenerationRun<S> {
        let parser = *processor.parser();
        let outcome = Box::new(Unrecorded(Instant::now()));
        let inputs = ChatOutputInputs {
            tokenizer: crate::test_support::configured_tokenizer(),
            prompt_token_ids: vec![u32::from(b'p')],
            decode,
            prompt_logprobs: false,
            logprobs,
            processor,
            parser,
            context: context(logprobs),
        };
        GenerationRun::new("request".to_owned(), 7, inputs, events, outcome, sink).unwrap()
    }

    /// The one JSON document a non-streamed request over `events` answers
    /// with, or the failure its held head states.
    async fn document(
        processor: ChatOutputProcessor,
        decode: TextDecodeOptions,
        logprobs: bool,
        events: Box<dyn Events>,
    ) -> Result<Value, ApiError> {
        let sink = ChatBodySink::new(&context(logprobs), 0);
        let mut body = run(processor, decode, logprobs, events, sink);
        match answered(&mut body).await {
            (bytes, None) => Ok(serde_json::from_slice(&bytes).unwrap()),
            (_, Some(error)) => Err(error),
        }
    }

    /// An engine stream of `events` after the request's scheduling.
    fn engine_events(events: impl IntoIterator<Item = Event>) -> Box<dyn Events> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        for event in events {
            tx.send(event).unwrap();
        }
        Box::new(ChannelEvents::new(rx))
    }

    fn text_token(byte: u8) -> Event {
        Event::TextToken {
            id: u32::from(byte),
            logprob: None,
        }
    }

    fn finished(reason: proto::request::FinishReason, completion_tokens: usize) -> Event {
        Event::Finished {
            reason,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens,
            images: 0,
        }
    }

    /// The text of a document's message, empty when it has none.
    fn content(document: &Value) -> &str {
        document["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
    }

    /// A generated token's logprob reaches the document beside its text, and
    /// a request cut at its token limit finishes with `length`.
    #[tokio::test]
    async fn a_document_carries_each_tokens_logprob() {
        let events = engine_events([
            text_token(b'a'),
            Event::TokenLogprobs {
                id: u32::from(b'a'),
                candidates: vec![TokenLogprob {
                    token_id: u32::from(b'a'),
                    logprob: -0.25,
                    rank: 3,
                }],
            },
            finished(proto::request::FinishReason::MaxTokens, 1),
        ]);
        let document = document(
            ChatOutputProcessor::default(),
            TextDecodeOptions::default(),
            true,
            events,
        )
        .await
        .unwrap();

        assert_eq!(content(&document), "a");
        let token = &document["choices"][0]["logprobs"]["content"][0];
        assert_eq!(token["token"], "a");
        assert_eq!(token["logprob"], -0.25);
        assert_eq!(document["choices"][0]["finish_reason"], "length");
        assert_eq!(document["usage"]["completion_tokens"], 1);
    }

    /// A request that finishes without publishing a text token (the engine
    /// withholds an EOS or excluded stop token) answers no text: the prompt
    /// seeds decoding only.
    #[tokio::test]
    async fn a_request_without_generated_tokens_has_no_text() {
        let events = engine_events([finished(proto::request::FinishReason::Eos, 1)]);
        let document = document(
            ChatOutputProcessor::default(),
            TextDecodeOptions::default(),
            false,
            events,
        )
        .await
        .unwrap();

        assert_eq!(content(&document), "");
        assert_eq!(document["choices"][0]["finish_reason"], "stop");
    }

    /// An engine failure of a running request (`Error`, or
    /// `ArtifactUnavailable` when its output cannot be delivered) answers a
    /// server error, in the head of a document and as the error event of a
    /// stream.
    #[tokio::test]
    async fn an_engine_failure_is_a_server_error() {
        let failures = || {
            [
                Event::Error {
                    message: "worker call failed".to_string(),
                },
                Event::ArtifactUnavailable {
                    message: "worker call failed".to_string(),
                },
            ]
        };
        for failure in failures() {
            let error = document(
                ChatOutputProcessor::default(),
                TextDecodeOptions::default(),
                false,
                engine_events([text_token(b'a'), failure]),
            )
            .await
            .unwrap_err();
            assert_eq!(
                error.status_code(),
                axum::http::StatusCode::INTERNAL_SERVER_ERROR
            );
        }
        for failure in failures() {
            let sink = ChatStreamEmitter::new(&context(false), 0);
            let mut body = run(
                ChatOutputProcessor::default(),
                TextDecodeOptions::default(),
                false,
                engine_events([text_token(b'a'), failure]),
                sink,
            );
            let (bytes, failed) = answered(&mut body).await;
            assert!(failed.is_none());
            let text = String::from_utf8(bytes).unwrap();
            assert!(
                text.contains(r#""type":"server_error""#) && text.ends_with("data: [DONE]\n\n"),
                "{text}"
            );
        }
    }

    /// An engine rejection of a non-streamed request is the status of its
    /// document: its head waits for the outcome.
    #[tokio::test]
    async fn a_rejected_document_states_its_rejection_in_the_head() {
        let events = engine_events([Event::Rejected {
            kind: proto::request::RejectionKind::Invalid,
            message: "the prompt does not fit".to_owned(),
        }]);
        let error = document(
            ChatOutputProcessor::default(),
            TextDecodeOptions::default(),
            false,
            events,
        )
        .await
        .unwrap_err();
        assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
    }

    /// Tokens committed together are decoded as one block, and a stop string
    /// inside a block ends the output at the match with the tokens through
    /// the one that completed it.
    #[tokio::test]
    async fn a_stop_string_inside_a_committed_block_ends_the_output() {
        let block = |text: &str| Event::TextTokens {
            ids: text.bytes().map(u32::from).collect(),
        };
        let decode = TextDecodeOptions {
            stop_strings: Some(vec!["END".to_string()]),
            ..TextDecodeOptions::default()
        };
        let processor =
            ChatOutputProcessor::qwen3(None, crate::test_support::configured_tokenizer(), true)
                .unwrap();
        let document = document(
            processor,
            decode,
            false,
            engine_events([
                block("first block, "),
                block("second one. END and more"),
                block("never decoded"),
            ]),
        )
        .await
        .unwrap();

        assert_eq!(content(&document), "first block, second one. ");
        assert_eq!(document["choices"][0]["finish_reason"], "stop");
        assert_eq!(document["choices"][0]["stop_reason"], "END");
        // The first block and the second through the `D` that completes `END`.
        assert_eq!(document["usage"]["completion_tokens"], 28);
    }

    /// Polls a body to its end and answers the bytes it wrote.
    fn written(run: &mut impl Session) -> Vec<u8> {
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut bytes = Vec::new();
        loop {
            match run.poll_step(&mut cx) {
                Poll::Ready(Step::Frames(frames)) => {
                    for frame in frames {
                        bytes.extend_from_slice(&frame);
                    }
                }
                Poll::Ready(Step::Ended) => return bytes,
                Poll::Ready(Step::Failed { .. }) => panic!("a video body states no failure"),
                Poll::Ready(Step::Migrate) => panic!("a video body performs no migration"),
                Poll::Pending => panic!("a video body never waits on anything"),
            }
        }
    }

    /// An artifact of three and a half chunks, so the last frame is short.
    fn artifact() -> Arc<Vec<u8>> {
        let len = VideoContentRun::<Vec<u8>>::CHUNK * 3 + VideoContentRun::<Vec<u8>>::CHUNK / 2;
        Arc::new((0..len).map(|byte| (byte % 251) as u8).collect())
    }

    fn body(media: &Arc<Vec<u8>>) -> VideoContentRun<Vec<u8>> {
        VideoContentRun::new(
            "job-1".to_owned(),
            "video/mp4".to_owned(),
            Arc::clone(media),
        )
    }

    #[test]
    fn a_video_body_writes_the_artifact_once() {
        let media = artifact();
        assert_eq!(written(&mut body(&media)), media.as_slice());
    }

    /// The state a handoff carries is the byte the predecessor reached, and a
    /// successor that resumes from it writes exactly the rest: no byte twice,
    /// none missing.
    #[test]
    fn a_handed_over_video_body_writes_only_the_bytes_after_the_offset() {
        let media = artifact();
        let mut first = body(&media);
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let Poll::Ready(Step::Frames(frames)) = first.poll_step(&mut cx) else {
            panic!("the first step writes a frame");
        };
        let first_frame = frames.iter().map(Bytes::len).sum::<usize>();
        assert_eq!(first_frame, VideoContentRun::<Vec<u8>>::CHUNK);

        let Some(Application::VideoContent(state)) = first.freeze() else {
            panic!("a video body states itself");
        };
        assert_eq!(state.offset, first_frame as u64);
        assert_eq!(state.job_id, "job-1");
        assert_eq!(state.content_type, "video/mp4");

        let mut successor = body(&media).resumed_at(state.offset as usize);
        assert_eq!(written(&mut successor), &media.as_slice()[first_frame..]);
    }

    /// A body frozen at its last byte, and one handed over beyond it, write
    /// nothing and end: the successor of a finished download does not repeat
    /// it.
    #[test]
    fn a_video_body_at_or_past_its_end_writes_nothing() {
        let media = artifact();
        let len = media.len();
        assert!(written(&mut body(&media).resumed_at(len)).is_empty());
        assert!(written(&mut body(&media).resumed_at(len + 1024)).is_empty());
    }
}
