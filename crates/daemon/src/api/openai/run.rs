//! A chat completion's response body, driven one engine event at a time so
//! that a hot restart can take it over.
//!
//! A handed-over response cannot be a generator, because a restart boundary
//! is a place to stop: between two engine events, with every frame of the
//! events consumed so far produced. [`GenerationRun`] therefore pushes: it
//! takes one event from the request's engine stream, runs it through the
//! request's `OutputProcessor`, turns the resulting outputs into its sink's
//! frames (server-sent events, or one JSON document), and returns. It is the
//! one response body of a token request, streamed or not.
//!
//! Its state is the whole of what a successor goes on writing from: the
//! output processor's (the incremental decoder, the parsers, the assembled
//! blocks, the timings), the sink's, and the request's identity in the engine
//! with the event position this process consumed. The successor takes the
//! request's engine stream over from that position
//! (`uniserve.request.Command.Takeover`) instead of submitting the request
//! again, so no token is computed twice.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::Context as _;
use bytes::Bytes;
use futures::FutureExt as _;
use proto::handoff;
use proto::handoff::stream::Application;
use thiserror_ext::AsReport as _;
use tracing::{error, trace};

use crate::api::body::{Session, Step};
use crate::api::events::{Events, Next as EventStep, Stats};
use crate::api::openai::chat_completions::{ChatCollector, ChatStreamEmitter, chat_response};
use crate::api::openai::error::ApiError;
use crate::api::output::{OutputProcessor, RequestOutput};
use crate::api::protocol::ChatCompletionStreamResponse;
use crate::api::tokenizer::DynTokenizer;

impl GenerationRun<ChatStreamEmitter> {
    /// Continues the streamed response `state` describes, whose engine events
    /// arrive on `events`, taken over from the connection that consumed them:
    /// the successor rebuilds the output processor the state states from its
    /// own tokenizer, then resumes every stage where the predecessor stopped.
    ///
    /// # Errors
    ///
    /// Fails when the state names no lowering or no prompt, or when the
    /// family's parsers cannot be built.
    pub fn resume_stream(
        state: &handoff::ChatStream,
        tokenizer: DynTokenizer,
        events: Box<dyn Events>,
        stats: Box<dyn Stats>,
    ) -> anyhow::Result<Self> {
        let output = resumed_output(
            tokenizer,
            state.lowering.as_ref(),
            state.decode.as_ref(),
            state.processor.as_ref(),
            state.assembly.as_ref(),
            stats,
        )?;
        let mut sink = ChatStreamEmitter::new(output.response());
        if let Some(chunks) = state.chunks.as_ref() {
            sink.resume(chunks);
        }

        Ok(Self::new(events, output, sink))
    }
}

impl GenerationRun<ChatBodySink> {
    /// Continues the response answered as one JSON body that `state`
    /// describes, whose engine events arrive on `events`.
    ///
    /// # Errors
    ///
    /// Fails as [`GenerationRun::resume_stream`] does.
    pub fn resume_collection(
        state: &handoff::ChatResponse,
        tokenizer: DynTokenizer,
        events: Box<dyn Events>,
        stats: Box<dyn Stats>,
    ) -> anyhow::Result<Self> {
        let output = resumed_output(
            tokenizer,
            state.lowering.as_ref(),
            state.decode.as_ref(),
            state.processor.as_ref(),
            state.assembly.as_ref(),
            stats,
        )?;
        let mut sink = ChatBodySink::new(output.response());
        if let Some(collection) = state.collection.as_ref() {
            sink.resume(collection);
        }

        Ok(Self::new(events, output, sink))
    }
}

/// The output processor of a handed-over response, rebuilt from the parts of
/// its state every chat response shares.
fn resumed_output(
    tokenizer: DynTokenizer,
    lowering: Option<&handoff::Lowering>,
    decode: Option<&handoff::Decode>,
    parsers: Option<&handoff::Processor>,
    assembly: Option<&handoff::Assembly>,
    stats: Box<dyn Stats>,
) -> anyhow::Result<OutputProcessor> {
    let lowering = lowering.context("a handed response states no lowering")?;
    lowering
        .response
        .as_ref()
        .context("a handed response states no response fields")?;

    OutputProcessor::resume(
        tokenizer,
        lowering,
        decode.unwrap_or(&Default::default()),
        parsers,
        assembly.unwrap_or(&Default::default()),
        stats,
    )
    .context("resuming the output processor")
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

    /// The response's application state for a successor: the sink's own
    /// state with `output`'s and the position `events` reached; `None` when
    /// the sink or a parser cannot be handed over at this boundary.
    fn freeze(&self, output: &OutputProcessor, events: &dyn Events) -> Option<Application>;

    /// The frames this response writes when it fails, in the sink's own shape:
    /// server-sent events for a streamed response, the one document a
    /// one-body response answers with.
    fn failed(&mut self, error: &ApiError) -> Vec<Bytes>;

    /// Whether the response's head waits for the body's first output, because
    /// the body is one document whose status is its result.
    fn holds_head(&self) -> bool;
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

    fn freeze(&self, output: &OutputProcessor, events: &dyn Events) -> Option<Application> {
        let processor = output.parser_state()?;
        let chunks = ChatStreamEmitter::state(self)?;
        let (consumed_events, consumed_tokens) = events.progress();
        let usage_owed = output.response().include_usage && !ChatStreamEmitter::done(self);

        Some(Application::ChatStream(handoff::ChatStream {
            request: events.submitted().unwrap_or_default(),
            lowering: Some(output.lowering()),
            decode: Some(output.decode_state()),
            processor: Some(processor),
            assembly: Some(output.assembly()),
            chunks: Some(chunks),
            consumed_events,
            consumed_tokens,
            usage_owed,
            engine_request_id: engine_request_id(events),
            engine_replica: output.engine(),
        }))
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
    collector: ChatCollector,
    response: handoff::ChatResponseOptions,
    /// Set once the body was written, so a later output writes nothing.
    written: bool,
}

impl ChatBodySink {
    /// The sink of the response `response` describes.
    pub fn new(response: &handoff::ChatResponseOptions) -> Self {
        Self {
            collector: ChatCollector::new(),
            response: response.clone(),
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
        let response = chat_response(&self.collector, &self.response)?;
        let body = serde_json::to_vec(&response).unwrap_or_else(|_| {
            br#"{"error":{"message":"failed to serialize chat completion","type":"server_error"}}"#
                .to_vec()
        });
        Ok(vec![Bytes::from(body)])
    }

    fn done(&self) -> bool {
        self.written
    }

    fn freeze(&self, output: &OutputProcessor, events: &dyn Events) -> Option<Application> {
        let processor = output.parser_state()?;
        let (consumed_events, consumed_tokens) = events.progress();

        Some(Application::ChatResponse(handoff::ChatResponse {
            request: events.submitted().unwrap_or_default(),
            lowering: Some(output.lowering()),
            decode: Some(output.decode_state()),
            processor: Some(processor),
            assembly: Some(output.assembly()),
            collection: Some(self.collector.state()),
            engine_request_id: engine_request_id(events),
            consumed_events,
            consumed_tokens,
            engine_replica: output.engine(),
        }))
    }

    fn failed(&mut self, error: &ApiError) -> Vec<Bytes> {
        // The document `ApiError`'s own response writes, so a failed
        // generation answers what a refused request answers.
        let body = serde_json::to_vec(&error.to_error_response()).unwrap_or_else(|_| {
            br#"{"error":{"message":"failed to serialize error response","type":"server_error"}}"#
                .to_vec()
        });
        vec![Bytes::from(body)]
    }

    fn holds_head(&self) -> bool {
        // One document, whose status is its result: nothing is gained by
        // sending its head before the document exists, and a head that went
        // out cannot state a failure.
        true
    }
}

/// One chat completion's response body, driven one engine event at a time.
pub struct GenerationRun<S: ResponseSink> {
    output: OutputProcessor,
    sink: S,
    events: Box<dyn Events>,
    /// Whether the body ended: its terminal output was written, or it failed.
    ended: bool,
}

impl<S: ResponseSink> GenerationRun<S> {
    /// The body of the request whose engine events arrive on `events`, which
    /// the caller submitted, processed by `output` and written by `sink`.
    pub fn new(events: Box<dyn Events>, output: OutputProcessor, sink: S) -> Self {
        Self {
            output,
            sink,
            events,
            ended: false,
        }
    }

    /// The step one engine event's outputs produce: their frames, which are
    /// none while the sink waits for more.
    ///
    /// An output the response fails on ends it in the sink's own shape
    /// ([`Self::failed`]): for a stream, the error event and the terminal
    /// `[DONE]` after the frames produced before it; for one document, the
    /// failure's status and error document.
    fn push(
        &mut self,
        event: proto::request::Event,
    ) -> Result<Step, crate::api::output::OutputError> {
        let outputs = self.output.push(event, self.events.as_mut())?;
        let mut frames = Vec::new();
        for output in &outputs {
            match self.sink.push(output) {
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
        if self.output.ended() {
            self.ended = true;
        }
        Ok(Step::Frames(frames))
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

/// The request's identity in the engine that serves it, as a handed state
/// carries it; empty once the request ended or was handed over.
fn engine_request_id(events: &dyn Events) -> String {
    events
        .engine_request_id()
        .map(|id| id.0.to_string())
        .unwrap_or_default()
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
                        replica = self.output.engine(),
                        engine_request_id = engine_request_id(self.events.as_ref()),
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
                        error!(error = %error.as_report(), "a response's output processing failed");
                        self.ended = true;
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
                        error!(?event, "a response's migration failed");
                    }
                }),
        )
    }

    fn freeze(&mut self) -> Option<Application> {
        self.sink.freeze(&self.output, self.events.as_ref())
    }

    fn abort(&mut self) {
        self.output.abandoned();
        self.events.cancel();
    }

    fn detach(&mut self) {
        self.output.abandoned();
        self.events.hand_over();
    }
}

/// One chunk as the server-sent event the connection writes.
fn chunk_frame(chunk: &ChatCompletionStreamResponse) -> Bytes {
    let payload = serde_json::to_string(chunk).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize chat completion chunk","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "chat completion emitting chunk");
    event_frame(&payload)
}

/// The error event a failed response ends with.
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
    use crate::api::test_support::{ChannelEvents, Unrecorded};

    /// Drives a body to its end as its connection does and answers what it
    /// wrote, with the failure a held head states.
    async fn responded(run: &mut impl Session) -> (Vec<u8>, Option<ApiError>) {
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
    fn response(logprobs: bool) -> handoff::ChatResponseOptions {
        handoff::ChatResponseOptions {
            request_id: "request".to_owned(),
            response_model: "model".to_owned(),
            requested_logprobs: logprobs,
            include_reasoning: true,
            ..Default::default()
        }
    }

    /// Decode options that skip special tokens and state no stop string.
    fn decode() -> handoff::TextDecode {
        handoff::TextDecode {
            skip_special_tokens: true,
            ..Default::default()
        }
    }

    /// The body of a request whose prompt is `p`, decoded as `decode` states
    /// and parsed as `parser` states, over the engine events `events`,
    /// written by `sink`.
    fn run<S: ResponseSink>(
        parser: handoff::Parser,
        decode: handoff::TextDecode,
        logprobs: bool,
        events: Box<dyn Events>,
        sink: S,
    ) -> GenerationRun<S> {
        let lowering = handoff::Lowering {
            prompt_token_ids: vec![u32::from(b'p')],
            decode_options: Some(decode),
            prompt_logprobs: false,
            logprobs,
            parser: Some(parser),
            response: Some(response(logprobs)),
            compile_us: 7,
            started_unix_us: 0,
        };
        let output = OutputProcessor::new(
            crate::api::test_support::configured_tokenizer(),
            lowering,
            Box::new(Unrecorded(std::time::Instant::now())),
        )
        .unwrap();
        GenerationRun::new(events, output, sink)
    }

    /// The one JSON document a non-streamed request over `events` answers
    /// with, or the failure its held head states.
    async fn document(
        parser: handoff::Parser,
        decode: handoff::TextDecode,
        logprobs: bool,
        events: Box<dyn Events>,
    ) -> Result<Value, ApiError> {
        let sink = ChatBodySink::new(&response(logprobs));
        let mut body = run(parser, decode, logprobs, events, sink);
        match responded(&mut body).await {
            (bytes, None) => Ok(serde_json::from_slice(&bytes).unwrap()),
            (_, Some(error)) => Err(error),
        }
    }

    /// An engine stream of `events` after the request's scheduling, which
    /// found nothing cached.
    fn engine_events(events: impl IntoIterator<Item = Event>) -> Box<dyn Events> {
        admitted(0, events)
    }

    /// An engine stream of `events` after the request's scheduling, whose
    /// admission found `cached_tokens` of its prompt cached.
    fn admitted(cached_tokens: usize, events: impl IntoIterator<Item = Event>) -> Box<dyn Events> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
            cached_tokens,
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
        let document = document(handoff::Parser::default(), decode(), true, events)
            .await
            .unwrap();

        assert_eq!(content(&document), "a");
        let token = &document["choices"][0]["logprobs"]["content"][0];
        assert_eq!(token["token"], "a");
        assert_eq!(token["logprob"], -0.25);
        assert_eq!(document["choices"][0]["finish_reason"], "length");
        assert_eq!(document["usage"]["completion_tokens"], 1);
    }

    /// The prompt tokens the request's first admission found cached are its
    /// usage's `prompt_tokens_details.cached_tokens`, in a document and in
    /// the usage chunk of a stream that asks for usage; a later admission of
    /// the request (after a preemption) does not replace them.
    #[tokio::test]
    async fn usage_reports_the_cached_tokens_of_the_first_admission() {
        let events = || {
            admitted(
                3,
                [
                    text_token(b'a'),
                    Event::Scheduled {
                        queued_at: 3.0,
                        scheduled_at: 4.0,
                        cached_tokens: 7,
                    },
                    text_token(b'b'),
                    finished(proto::request::FinishReason::MaxTokens, 2),
                ],
            )
        };
        let document = document(handoff::Parser::default(), decode(), false, events())
            .await
            .unwrap();
        assert_eq!(document["usage"]["prompt_tokens"], 1);
        assert_eq!(
            document["usage"]["prompt_tokens_details"]["cached_tokens"],
            3
        );

        let mut options = response(false);
        options.include_usage = true;
        let sink = ChatStreamEmitter::new(&options);
        let mut body = run(handoff::Parser::default(), decode(), false, events(), sink);
        let (bytes, failed) = responded(&mut body).await;
        assert!(failed.is_none());
        let text = String::from_utf8(bytes).unwrap();
        let usage: Vec<Value> = text
            .split("\n\n")
            .filter_map(|event| event.strip_prefix("data: "))
            .filter(|payload| *payload != "[DONE]")
            .map(|payload| serde_json::from_str::<Value>(payload).unwrap())
            .filter_map(|chunk| chunk.get("usage").cloned())
            .collect();
        let [usage] = &usage[..] else {
            panic!("one usage chunk: {text}");
        };
        assert_eq!(usage["completion_tokens"], 2);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 3);
    }

    /// A request that finishes without publishing a text token (the engine
    /// withholds an EOS or excluded stop token) answers no text: the prompt
    /// seeds decoding only.
    #[tokio::test]
    async fn a_request_without_generated_tokens_has_no_text() {
        let events = engine_events([finished(proto::request::FinishReason::Eos, 1)]);
        let document = document(handoff::Parser::default(), decode(), false, events)
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
                handoff::Parser::default(),
                decode(),
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
            let sink = ChatStreamEmitter::new(&response(false));
            let mut body = run(
                handoff::Parser::default(),
                decode(),
                false,
                engine_events([text_token(b'a'), failure]),
                sink,
            );
            let (bytes, failed) = responded(&mut body).await;
            assert!(failed.is_none());
            let text = String::from_utf8(bytes).unwrap();
            assert!(
                text.contains(r#""type":"server_error""#) && text.ends_with("data: [DONE]\n\n"),
                "{text}"
            );
        }
    }

    /// An engine rejection of a non-streamed request is the status of its
    /// document: its head waits for the request's result.
    #[tokio::test]
    async fn a_rejected_document_states_its_rejection_in_the_head() {
        let events = engine_events([Event::Rejected {
            kind: proto::request::RejectionKind::Invalid,
            message: "the prompt does not fit".to_owned(),
        }]);
        let error = document(handoff::Parser::default(), decode(), false, events)
            .await
            .unwrap_err();
        assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
    }

    /// The parsers of a request that may call the tool `lookup`, written as
    /// function and parameter tags.
    fn lookup_parser() -> handoff::Parser {
        handoff::Parser {
            reasoning_delimiters: None,
            tool_calls: "xml".to_owned(),
            tools: serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
                    },
                },
            }])
            .to_string(),
        }
    }

    /// The engine events of a model that writes `text` one token a character
    /// and then stops.
    fn generated(text: &str) -> Box<dyn Events> {
        let tokens = text.bytes().map(text_token);
        let finish = finished(proto::request::FinishReason::Eos, text.len());
        engine_events(tokens.chain(std::iter::once(finish)))
    }

    /// A response of text and two tool calls, as a model writes them.
    const TWO_CALLS: &str = "On it.<tool_call>\n<function=lookup>\n<parameter=q>\nrust\n</parameter>\n\
                             <parameter=limit>\n2\n</parameter>\n</function>\n</tool_call>\
                             <tool_call>\n<function=lookup>\n<parameter=q>\ngo\n</parameter>\n\
                             </function>\n</tool_call>";

    /// A streamed response carries each tool call in the OpenAI chunk shape:
    /// the call's index on every delta, its id, type and function name once
    /// with empty arguments, then its arguments in increments; several calls
    /// follow one another by index, and the stream finishes with
    /// `tool_calls`.
    #[tokio::test]
    async fn a_stream_carries_tool_calls_as_openai_deltas() {
        let sink = ChatStreamEmitter::new(&response(false));
        let mut body = run(lookup_parser(), decode(), false, generated(TWO_CALLS), sink);
        let (bytes, failed) = responded(&mut body).await;
        assert!(failed.is_none());

        let text = String::from_utf8(bytes).unwrap();
        assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
        let chunks: Vec<Value> = text
            .split("\n\n")
            .filter_map(|event| event.strip_prefix("data: "))
            .filter(|payload| *payload != "[DONE]")
            .map(|payload| serde_json::from_str(payload).unwrap())
            .collect();

        let mut content = String::new();
        // Per call index: the ids, the names, and the concatenated arguments.
        let mut calls: Vec<(Vec<String>, Vec<String>, String)> = Vec::new();
        let mut finish_reason = None;
        for chunk in &chunks {
            let choice = &chunk["choices"][0];
            if let Some(text) = choice["delta"]["content"].as_str() {
                content.push_str(text);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
            for delta in choice["delta"]["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
            {
                let index = delta["index"]
                    .as_u64()
                    .expect("every delta states its index") as usize;
                if calls.len() <= index {
                    calls.resize_with(index + 1, Default::default);
                }
                let (ids, names, arguments) = &mut calls[index];
                if let Some(id) = delta["id"].as_str() {
                    assert_eq!(delta["type"], "function");
                    ids.push(id.to_owned());
                }
                if let Some(name) = delta["function"]["name"].as_str() {
                    // The delta that names the call opens its arguments empty.
                    assert_eq!(delta["function"]["arguments"], "");
                    names.push(name.to_owned());
                }
                arguments.push_str(delta["function"]["arguments"].as_str().unwrap_or_default());
            }
        }

        assert_eq!(content, "On it.");
        assert_eq!(finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(calls.len(), 2);
        for (ids, names, _) in &calls {
            assert_eq!(ids.len(), 1, "a call states its id once");
            assert!(ids[0].starts_with("call_"));
            assert_eq!(names, &["lookup"], "a call states its name once");
        }
        assert_ne!(calls[0].0, calls[1].0);
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].2).unwrap(),
            serde_json::json!({"q": "rust", "limit": 2})
        );
        assert_eq!(
            serde_json::from_str::<Value>(&calls[1].2).unwrap(),
            serde_json::json!({"q": "go"})
        );
    }

    /// The same response answered as one document lists both calls in its
    /// message, with their arguments as JSON text, beside the text before
    /// them.
    #[tokio::test]
    async fn a_document_lists_every_tool_call() {
        let document = document(lookup_parser(), decode(), false, generated(TWO_CALLS))
            .await
            .unwrap();

        let choice = &document["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(choice["message"]["content"], "On it.");
        let calls = choice["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        let arguments = |call: &Value| -> Value {
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap()
        };
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "lookup");
        assert_eq!(
            arguments(&calls[0]),
            serde_json::json!({"q": "rust", "limit": 2})
        );
        assert_eq!(arguments(&calls[1]), serde_json::json!({"q": "go"}));
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
