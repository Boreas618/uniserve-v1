//! Model-aware admission, engine submission, and output streaming.
//!
//! Requests follow one ownership chain:
//! `HTTP schema / text prompt -> InputProcessor -> GenerationRequest ->
//! EngineClient::submit_generation -> RequestOutput stream`.
//! Video requests follow the same chain with a `DiffusionRequest`,
//! `EngineClient::submit_media`, and `assembly::assemble_media_event_stream`.
//!
//! [`ServingRuntime`] owns each request from identity reservation to its terminal event. It
//! reserves the external identifier and an engine `RequestId` in the engine client's
//! `RequestRegistry` before preprocessing, resolves the request's image references on the
//! async runtime ([`media::ImageFetcher`]), runs model preprocessing on the blocking pool,
//! submits the prepared request, and wraps the engine receiver in the assembler and in
//! lifecycle tracking (`LifecycleGuard`, `LifecycleTrackedStream`).
//!
//! External cancel and abort commands are recorded in the registry and take precedence over
//! any later engine output: the caller sees `RequestOutput::Cancelled` or
//! `RequestOutput::Aborted` in place of the engine's own terminal event. Each request records
//! exactly one terminal outcome, including when the caller drops the pending future or the
//! response stream. Aggregate lifecycle counters reach the metrics route through
//! [`RuntimeMetricsSnapshot::state_counts`].

mod assembly;
pub mod chat;
mod diffusion_gemma;
pub(crate) use self::diffusion_gemma::canvas_sampling;
mod error;
mod input;
mod lifecycle;
pub mod media;
mod model;
mod omni;
mod output;
mod preprocessing;
mod sampling;
pub mod systemone;
#[cfg(test)]
pub(crate) mod test_support;
/// Text tokenization, decoding, and sampling utilities.
pub mod text;
pub mod video;

use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::engine_client::{EngineClient, EventRx, StreamCancelCause};
use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _};
use serde::{Deserialize, Serialize};
use uniserve_core::EngineCoreOutput;

pub use input::{
    DecodeControls, ImageGenControls, ModalitySelection, OutputDetail, OutputProcessorPolicy,
    PromptInput, ResponseOptions, SamplingConfig, StopConfig, TextPromptRequest,
};
pub use media::ImageInput;
pub(crate) use model::LoadedModel;
pub use model::{InputProcessor, WorkerCapabilities, validate_video_capacity};
pub use preprocessing::chat_image_urls;

use crate::serving::omni::{SenseNovaOutputProcessor, SenseNovaTextDelta};
use crate::serving::text::output::stop_string_holdback_bytes;
use crate::serving::text::{
    DecodedLogprobs, DecodedPromptLogprobs, FinishReason, StopReason, TextDecodeOptions,
};

use assembly::{assemble_chat_event_stream, assemble_event_stream};

use error::malformed_output;
pub use error::{OutputProcessingError, ServeError, TokenizeError};
pub(crate) use lifecycle::LifecycleTerminal;
pub(crate) use lifecycle::RequestLifecycleState;
use lifecycle::{
    LifecycleGuard, LifecycleTrackedStream, RuntimeLifecycleMetrics, control_aware_event_stream,
};
pub use lifecycle::{RuntimeMetricsSnapshot, ServeControlError};
pub use output::{FinishStatus, RequestOutput, RuntimeTimings, StopCause};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
/// Caller-visible request identifier used throughout serving.
///
/// The type performs no validation. The HTTP routes build it from the resolved request ID
/// with an endpoint prefix (for example `chatcmpl-`), and `RequestRegistry::register` refuses
/// an identifier that another live request already holds.
pub struct ServeRequestId(String);

impl ServeRequestId {
    /// Constructs a serving request identifier.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Consumes the identifier and returns its string representation.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for ServeRequestId {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Deref for ServeRequestId {
    type Target = str;

    /// Returns shared access to the wrapped value.
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for ServeRequestId {
    /// Returns shared access to the wrapped value.
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for ServeRequestId {
    /// Borrows the wrapped value.
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl From<String> for ServeRequestId {
    /// Converts the source value into this type.
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ServeRequestId {
    /// Converts the source value into this type.
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<ServeRequestId> for String {
    /// Converts the source value into this type.
    fn from(value: ServeRequestId) -> Self {
        value.0
    }
}

/// Boxed asynchronous stream of serving events.
pub type RequestOutputStream = Pin<Box<dyn Stream<Item = Result<RequestOutput>> + Send>>;
/// Result type returned by serving-runtime calls.
pub type Result<T> = std::result::Result<T, ServeError>;

/// Folds cache namespace/salt into a stable isolation key.
///
/// Returns `None` when both are absent, which leaves the request in the shared cache
/// partition. Otherwise the key is a 64-bit FNV-1a hash (the same function as
/// `model::stable_hash`) over length-prefixed material, so `("ab", "c")` and `("a", "bc")`
/// hash different bytes. The hash is deterministic across processes.
pub(crate) fn cache_isolation_key(namespace: Option<&str>, salt: Option<&str>) -> Option<u64> {
    (namespace.is_some() || salt.is_some()).then(|| {
        let namespace = namespace.unwrap_or_default();
        let salt = salt.unwrap_or_default();
        let material = format!("{}:{namespace}{}:{salt}", namespace.len(), salt.len());
        material
            .as_bytes()
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    })
}

/// Tokenized input retains only the response processing its computation needs.
///
/// Media requests carry no `ResponseOptions`: their stream is assembled from the prompt
/// length alone.
enum Prepared {
    Generation(Box<(uniserve_core::GenerationRequest, ResponseOptions)>),
    Diffusion(uniserve_core::DiffusionRequest),
}

impl From<(uniserve_core::GenerationRequest, ResponseOptions)> for Prepared {
    fn from((request, response): (uniserve_core::GenerationRequest, ResponseOptions)) -> Self {
        Self::Generation(Box::new((request, response)))
    }
}

/// Single-model serving runtime.
pub struct ServingRuntime {
    model: Arc<InputProcessor>,
    engine: Arc<EngineClient>,
    images: media::ImageFetcher,
    // The System One readout encoder of a model that serves the endpoint.
    readout: Option<Arc<systemone::ReadoutEncoder>>,
    // Held only to keep the periodic logging task alive; dropping the logger aborts the task.
    _stats_logger: Option<Arc<crate::engine_client::generation::log_stats::StatsLogger>>,
    // Cumulative lifecycle counters shared with every request's lifecycle guard and assembler.
    metrics: Arc<RuntimeLifecycleMetrics>,
}

impl ServingRuntime {
    /// Creates a serving runtime for one resolved model and engine client.
    ///
    /// `images` resolves the image references of every request this runtime serves.
    pub fn new(
        model: InputProcessor,
        engine: Arc<EngineClient>,
        images: media::ImageFetcher,
        log_stats: bool,
    ) -> Self {
        let stats_logger = log_stats.then(|| {
            Arc::new(
                crate::engine_client::generation::log_stats::StatsLogger::start(
                    engine.model_name().to_string(),
                    engine.engine_count(),
                ),
            )
        });
        Self {
            model: Arc::new(model),
            engine,
            images,
            readout: None,
            _stats_logger: stats_logger,
            metrics: Arc::new(RuntimeLifecycleMetrics::default()),
        }
    }

    /// Serves `POST /v1/systemone` with `encoder`, prepared for this model.
    #[must_use]
    pub fn with_readout(mut self, encoder: systemone::ReadoutEncoder) -> Self {
        self.readout = Some(Arc::new(encoder));
        self
    }

    /// Returns the resolved model owned by this runtime.
    pub fn model(&self) -> &InputProcessor {
        &self.model
    }

    /// Returns the public served-model name.
    pub fn served_model_name(&self) -> &str {
        self.model.served_model_name()
    }

    /// Returns the engine client backing this runtime.
    pub fn engine(&self) -> &EngineClient {
        &self.engine
    }

    /// Returns an aggregate point-in-time metrics snapshot.
    ///
    /// `active` is read from the request registry at call time; the other fields are
    /// cumulative counters since the runtime started.
    pub fn metrics_snapshot(&self) -> RuntimeMetricsSnapshot {
        let mut snapshot = self.metrics.snapshot();
        snapshot.active = self.engine.requests.active_count() as u64;
        snapshot
    }

    /// Waits until every active request leaves the runtime.
    pub async fn drain(&self) {
        self.engine.requests.drain().await;
    }

    /// Preprocesses a chat request while retaining cancellation and identity ownership.
    ///
    /// The `image_url` parts are fetched only for a model that accepts image input; for any
    /// other model, preprocessing rejects them without network access.
    pub async fn generate_chat(
        &self,
        request_id: ServeRequestId,
        request: crate::openai::ChatCompletionRequest,
    ) -> crate::openai::Result<RequestOutputStream> {
        crate::openai::chat_completions::validate_request_compat(
            &request,
            self.served_model_name(),
        )?;
        let image_urls = if self.model.supports_image_input() {
            chat_image_urls(&request)
        } else {
            Vec::new()
        };
        let input_id = request_id.clone();
        self.generate_with(request_id.clone(), move |model| async move {
            let images = self.images.fetch_all(image_urls).await.map_err(|source| {
                crate::openai::serve_error_to_api(ServeError::ImageInput {
                    request_id: input_id.clone(),
                    source,
                })
            })?;
            blocking(request_id, move |model| {
                model
                    .preprocess_chat_request(input_id, request, images)
                    .map(Prepared::from)
            })(model)
            .await
        })
        .await
    }

    /// Preprocesses one image-generation API request and streams its output.
    pub async fn generate_image(
        &self,
        request_id: ServeRequestId,
        request: crate::openai::ImageGenerationRequest,
    ) -> crate::openai::Result<RequestOutputStream> {
        let input_id = request_id.clone();
        self.generate_with(
            request_id.clone(),
            blocking(request_id, move |model| {
                model
                    .preprocess_image_request(input_id, request)
                    .map(Prepared::from)
            }),
        )
        .await
    }

    /// Runs a programmatic text prompt, including model-specific context images.
    pub async fn generate_text(
        &self,
        request: TextPromptRequest,
    ) -> crate::openai::Result<RequestOutputStream> {
        let request_id = request.request_id.clone();
        self.generate_with(
            request_id.clone(),
            blocking(request_id, move |model| {
                model
                    .preprocess_text_request(request)
                    .map(Prepared::from)
                    .map_err(crate::openai::serve_error_to_api)
            }),
        )
        .await
    }

    /// Prepares a video request (media ingestion included) within its request lifecycle and
    /// streams its output.
    ///
    /// Also returns the prepared request, whose duration, canvas and sampling describe the
    /// generation; it is `None` when a control command ended the request before preparation
    /// completed, in which case the stream carries that terminal outcome.
    pub async fn generate_video(
        &self,
        request_id: ServeRequestId,
        request: crate::openai::VideoGenerationRequest,
    ) -> crate::openai::Result<(
        Option<crate::serving::video::PreparedVideo>,
        RequestOutputStream,
    )> {
        let (prepared, outline) = tokio::sync::oneshot::channel();
        let input_id = request_id.clone();
        let stream = self
            .generate_with(request_id, move |model| async move {
                let video = model.preprocess_video_request(&input_id, &request).await?;
                // The caller reads the outline only after this future completed or was
                // dropped, so a closed receiver cannot occur here.
                let _ = prepared.send(video.clone());
                Ok(Prepared::Diffusion(video.request))
            })
            .await?;
        Ok((outline.await.ok(), stream))
    }

    /// Owns the request across image resolution, blocking preprocessing, submission, and
    /// public output.
    ///
    /// Preprocessing resolves media on the async runtime and moves synchronous
    /// tokenization to the blocking pool through `blocking`.
    ///
    /// Registration precedes preprocessing, so a duplicate identifier is refused before any
    /// tokenization work and a cancel or abort can target a request that is still compiling.
    /// Every path after registration either returns a stream that carries the lifecycle guard
    /// or records a terminal outcome through it.
    async fn generate_with<Preprocess>(
        &self,
        request_id: ServeRequestId,
        preprocess: impl FnOnce(Arc<InputProcessor>) -> Preprocess,
    ) -> crate::openai::Result<RequestOutputStream>
    where
        Preprocess: std::future::Future<Output = crate::openai::Result<Prepared>> + Send,
    {
        // `compile_us` and the lifecycle's elapsed times are measured from this instant, so
        // image fetching and time spent queued for the blocking pool count toward
        // compilation.
        let compile_started = Instant::now();

        let engine_request_id = self
            .engine
            .register_request(request_id.to_string(), true)
            .map_err(|error| {
                self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
                match error {
                    crate::engine_client::Error::DuplicateRequestId { .. } => {
                        ServeError::DuplicateRequestId {
                            request_id: request_id.clone(),
                        }
                    }
                    error => ServeError::Engine(error),
                }
            })
            .map_err(crate::openai::serve_error_to_api)?;
        // Created before the first await: if the caller drops this future, the guard's `Drop`
        // records the request as cancelled, or as aborted when an abort is pending.
        let mut lifecycle = LifecycleGuard::new(
            request_id.clone(),
            Arc::clone(&self.metrics),
            Arc::clone(&self.engine.requests),
            compile_started,
        );

        // A control command that arrives during preprocessing ends the request immediately and
        // drops the preprocessing future. Work on the blocking pool (`blocking`) still runs to
        // completion and its result is discarded; asynchronous work such as media fetches
        // stops at its next await.
        let tokenize_result = tokio::select! {
            terminal = self.engine.requests.wait_for_control(&request_id) => {
                return Ok(self.control_event_stream(request_id, terminal, lifecycle));
            }
            tokenized = preprocess(Arc::clone(&self.model)) => tokenized,
        };
        let mut tokenized = match tokenize_result {
            Ok(tokenized) => tokenized,
            Err(error) => {
                lifecycle.terminal(
                    LifecycleTerminal::Rejected,
                    compile_started.elapsed().as_micros() as u64,
                );
                return Err(error);
            }
        };
        let compile_duration_us = compile_started.elapsed().as_micros() as u64;
        self.engine
            .requests
            .mark_submitting(&request_id, compile_duration_us);

        // Preprocessing fills a placeholder engine identifier. `EngineClient::submit_generation`
        // and `submit_media` accept only the identifier reserved at registration.
        match &mut tokenized {
            Prepared::Generation(prepared) => prepared.0.request_id = engine_request_id,
            Prepared::Diffusion(request) => request.request_id = engine_request_id,
        }
        self.submit_and_stream(tokenized, compile_duration_us, lifecycle)
            .await
            .map_err(crate::openai::serve_error_to_api)
    }

    /// Submits a tokenized request and wraps its output with model and lifecycle processing.
    ///
    /// A control command can arrive at any point, so the registry is checked before
    /// submission, after submission, and atomically when the request is accepted. A control
    /// found at any checkpoint yields an immediate terminal stream in place of engine output.
    /// A submission failure with no pending control records `Failed` and returns the error.
    ///
    /// A control found after a successful submission reaches the engine by dropping the
    /// unpolled output stream: its `EventRx` sends the engine a cancel at the acknowledged
    /// prefix, which releases the engine request for a cancel and an abort alike, and then
    /// releases the registry's engine side. The client's `Cancelled` or `Aborted` event comes
    /// from the registry's recorded control, not from the engine.
    async fn submit_and_stream(
        &self,
        prepared: Prepared,
        compile_duration_us: u64,
        mut lifecycle: LifecycleGuard,
    ) -> Result<RequestOutputStream> {
        let request_id = lifecycle.request_id.clone();
        if let Some(terminal) = self.engine.requests.control_terminal(&request_id) {
            return Ok(self.control_event_stream(request_id, terminal, lifecycle));
        }

        let stream_result: Result<RequestOutputStream> = match prepared {
            Prepared::Generation(prepared) => {
                let (request, response) = *prepared;
                let ResponseOptions {
                    tokenizer,
                    prompt_token_ids,
                    decode,
                    emit_token_ids,
                    prompt_logprobs_requested,
                    generated_logprobs_requested,
                    output_processor,
                } = response;
                let engine_stream = self
                    .engine
                    .submit_generation(request_id.to_string(), request)
                    .await;

                let event_context = EventContext {
                    compile_duration_us,
                    started: lifecycle.started,
                    metrics: Arc::clone(&self.metrics),
                };

                match engine_stream {
                    Ok(stream) => {
                        let assembly = StreamInput {
                            request_id: request_id.clone(),
                            event_context,
                            prompt_token_ids,
                            tokenizer,
                            prompt_logprobs_requested,
                            generated_logprobs_requested,
                            emit_token_ids,
                            decode_options: decode,
                            stream,
                        };
                        let output: RequestOutputStream = match output_processor {
                            OutputProcessorPolicy::Chat(processor) => {
                                Box::pin(assemble_chat_event_stream(assembly, processor))
                            }
                            // `assemble_event_stream` applies the remaining policies as its
                            // output sink.
                            output_processor => {
                                Box::pin(assemble_event_stream(assembly, output_processor))
                            }
                        };
                        Ok(output)
                    }
                    Err(error) => Err(ServeError::Engine(error)),
                }
            }
            Prepared::Diffusion(request) => {
                let prompt_tokens = request.prompt_token_ids.len();
                let context = EventContext {
                    compile_duration_us,
                    started: lifecycle.started,
                    metrics: Arc::clone(&self.metrics),
                };
                self.engine
                    .submit_media(request_id.to_string(), request)
                    .await
                    .map(|stream| {
                        Box::pin(assembly::assemble_media_event_stream(
                            request_id.clone(),
                            context,
                            prompt_tokens,
                            stream,
                        )) as RequestOutputStream
                    })
                    .map_err(ServeError::Engine)
            }
        };

        // A control recorded before `claim_submission` was not forwarded to the engine (one
        // recorded after it was). Dropping the unpolled stream drops its `EventRx`, which
        // delivers it as a cancel at the acknowledged prefix and releases the registry's
        // engine side. A failed submission has no engine request to control.
        if let Some(terminal) = self.engine.requests.control_terminal(&request_id) {
            if let Ok(stream) = stream_result {
                drop(stream);
            }
            return Ok(self.control_event_stream(request_id, terminal, lifecycle));
        }
        let stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                lifecycle.terminal(
                    LifecycleTerminal::Failed,
                    lifecycle.started.elapsed().as_micros() as u64,
                );
                return Err(error);
            }
        };

        // `accept` refuses under the registry lock when a control was recorded after the
        // previous check or when the request has no lifecycle record. Such a control was
        // already forwarded to the engine when it was recorded; dropping the stream cancels
        // the engine request in any case.
        if !self.engine.requests.accept(&request_id) {
            let terminal = self
                .engine
                .requests
                .control_terminal(&request_id)
                .unwrap_or(LifecycleTerminal::Cancelled);
            drop(stream);
            return Ok(self.control_event_stream(request_id, terminal, lifecycle));
        }

        // Layering: the assembler produces public events, `control_aware_event_stream` ends
        // them promptly on a control, and `LifecycleTrackedStream` records statistics and the
        // single terminal outcome.
        self.metrics.accepted.fetch_add(1, Ordering::Relaxed);
        let stream = Box::pin(control_aware_event_stream(
            request_id.clone(),
            Arc::clone(&self.engine.requests),
            stream,
        )) as RequestOutputStream;
        Ok(Box::pin(LifecycleTrackedStream::new(stream, lifecycle)))
    }

    /// Builds an immediately terminal stream for a request controlled before engine ownership.
    ///
    /// Callers pass a control outcome (`Cancelled` or `Aborted`); any other terminal maps to
    /// `RequestOutput::Cancelled`.
    fn control_event_stream(
        &self,
        request_id: ServeRequestId,
        terminal: LifecycleTerminal,
        lifecycle: LifecycleGuard,
    ) -> RequestOutputStream {
        let event = match terminal {
            LifecycleTerminal::Aborted => RequestOutput::Aborted {
                request_id: request_id.clone(),
            },
            LifecycleTerminal::Cancelled
            | LifecycleTerminal::Finished
            | LifecycleTerminal::Rejected
            | LifecycleTerminal::Failed => RequestOutput::Cancelled {
                request_id: request_id.clone(),
            },
        };
        let inner = Box::pin(futures::stream::iter([Ok(event)])) as RequestOutputStream;
        Box::pin(LifecycleTrackedStream::new(inner, lifecycle))
    }

    /// Cancels a live request at its acknowledged output prefix.
    pub async fn cancel(
        &self,
        request_id: impl Into<ServeRequestId>,
    ) -> std::result::Result<(), ServeControlError> {
        let request_id = request_id.into();
        self.engine.cancel_request(&request_id).await?;
        Ok(())
    }

    /// Aborts a live request immediately.
    pub async fn abort(
        &self,
        request_id: impl Into<ServeRequestId>,
    ) -> std::result::Result<(), ServeControlError> {
        let request_id = request_id.into();
        self.engine.abort_request(&request_id).await?;
        Ok(())
    }

    /// Drains requests and shuts down the backing engine.
    pub async fn shutdown(self) -> std::result::Result<(), crate::engine_client::Error> {
        self.drain().await;
        self.engine.shutdown().await
    }
}

/// Request-scoped metadata the assemblers stamp onto public events and usage.
#[derive(Debug, Clone)]
struct EventContext {
    started: Instant,
    compile_duration_us: u64,
    // The assemblers increment `scheduled` when the engine reports scheduling.
    metrics: Arc<RuntimeLifecycleMetrics>,
}

/// Engine receiver and the response options a generation assembler consumes.
struct StreamInput {
    request_id: ServeRequestId,
    event_context: EventContext,
    prompt_token_ids: Vec<u32>,
    tokenizer: crate::serving::text::tokenizer::DynTokenizer,
    prompt_logprobs_requested: bool,
    generated_logprobs_requested: bool,
    emit_token_ids: bool,
    decode_options: TextDecodeOptions,
    stream: EventRx,
}

/// Runs synchronous preprocessing (tokenization, chat rendering) on the blocking pool.
///
/// A panicked or cancelled task becomes a tokenization error of `request_id`.
fn blocking(
    request_id: ServeRequestId,
    preprocess: impl FnOnce(&InputProcessor) -> crate::openai::Result<Prepared> + Send + 'static,
) -> impl FnOnce(
    Arc<InputProcessor>,
) -> futures::future::BoxFuture<'static, crate::openai::Result<Prepared>> {
    move |model| {
        Box::pin(async move {
            tokio::task::spawn_blocking(move || preprocess(&model))
                .await
                .unwrap_or_else(|error| {
                    Err(crate::openai::serve_error_to_api(ServeError::Tokenize {
                        request_id,
                        source: TokenizeError::Task(error),
                    }))
                })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::TokenLogprob;
    use uniserve_engine::EventRx;

    fn event_context() -> EventContext {
        EventContext {
            started: Instant::now(),
            compile_duration_us: 7,
            metrics: Arc::new(RuntimeLifecycleMetrics::default()),
        }
    }

    #[tokio::test]
    async fn assembler_attaches_ranked_logprobs_to_text_delta() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(EngineCoreOutput::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.try_send(EngineCoreOutput::TextToken {
            id: b'a' as u32,
            logprob: Some(-0.25),
        })
        .unwrap();
        tx.try_send(EngineCoreOutput::TokenLogprobs {
            id: b'a' as u32,
            candidates: vec![TokenLogprob {
                token_id: b'a' as u32,
                logprob: -0.25,
                rank: 3,
            }],
        })
        .unwrap();
        tx.try_send(EngineCoreOutput::Finished {
            reason: uniserve_core::FinishReason::MaxTokens,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: 1,
            images: 0,
        })
        .unwrap();
        drop(tx);

        let events = assemble_event_stream(
            StreamInput {
                request_id: "req".into(),
                event_context: event_context(),
                prompt_token_ids: vec![b'p' as u32],
                tokenizer,
                prompt_logprobs_requested: false,
                generated_logprobs_requested: true,
                emit_token_ids: true,
                decode_options: TextDecodeOptions::default(),
                stream: EventRx::from_receiver(rx),
            },
            OutputProcessorPolicy::None,
        )
        .collect::<Vec<_>>()
        .await;

        let delta = events
            .iter()
            .find_map(|event| match event {
                Ok(RequestOutput::TextDelta {
                    text,
                    token_ids,
                    logprobs,
                    ..
                }) => Some((text, token_ids, logprobs.as_ref())),
                _ => None,
            })
            .expect("text delta");
        assert_eq!(delta.0, "a");
        assert_eq!(delta.1, &[b'a' as u32]);
        let candidate = &delta.2.expect("decoded logprobs").positions[0].entries[0];
        assert_eq!(candidate.token_id, b'a' as u32);
        assert_eq!(candidate.rank, 3);
        assert!(events.iter().all(std::result::Result::is_ok));
        assert!(matches!(
            events.last(),
            Some(Ok(RequestOutput::Finished {
                reason: FinishStatus::Length,
                ..
            }))
        ));
    }

    /// A request that finishes without publishing a text token (the engine
    /// withholds an EOS or excluded stop token, and image-only generation
    /// publishes none) returns no text. The prompt seeds decoding only.
    #[tokio::test]
    async fn assembler_emits_no_text_when_no_token_was_generated() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(EngineCoreOutput::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.try_send(EngineCoreOutput::Finished {
            reason: uniserve_core::FinishReason::Eos,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: 1,
            images: 0,
        })
        .unwrap();
        drop(tx);

        let events = assemble_event_stream(
            StreamInput {
                request_id: "eos-first".into(),
                event_context: event_context(),
                prompt_token_ids: vec![b'p' as u32],
                tokenizer,
                prompt_logprobs_requested: false,
                generated_logprobs_requested: false,
                emit_token_ids: false,
                decode_options: TextDecodeOptions::default(),
                stream: EventRx::from_receiver(rx),
            },
            OutputProcessorPolicy::None,
        )
        .collect::<Vec<_>>()
        .await;

        assert!(events.iter().all(std::result::Result::is_ok));
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                Ok(RequestOutput::TextDelta { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "");
        assert!(matches!(
            events.last(),
            Some(Ok(RequestOutput::Finished {
                reason: FinishStatus::Stop { .. },
                ..
            }))
        ));
    }

    /// SenseNova filtering holds back a trailing fragment that could begin a
    /// `<think>`, `</think>`, `<answer>` or `</answer>` delimiter. At the end of
    /// the stream no delimiter can complete, so the held fragment is visible
    /// text, whether the engine finished the request or a stop string did.
    #[tokio::test]
    async fn sensenova_assembler_releases_a_trailing_delimiter_prefix_at_the_end() {
        let policy = crate::profile::omni::OutputFilterPolicy {
            reasoning: Some(crate::profile::omni::DelimitedTextPolicy {
                start: "<think>".to_string(),
                end: "</think>".to_string(),
            }),
            visible_wrappers: vec![crate::profile::omni::DelimitedTextPolicy {
                start: "<answer>".to_string(),
                end: "</answer>".to_string(),
            }],
        };
        let text_token = |byte: u8| EngineCoreOutput::TextToken {
            id: u32::from(byte),
            logprob: None,
        };
        let finished = EngineCoreOutput::Finished {
            reason: uniserve_core::FinishReason::MaxTokens,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: 3,
            images: 0,
        };
        let cases = [
            (
                "engine-finished",
                vec![
                    text_token(b'x'),
                    text_token(b' '),
                    text_token(b'<'),
                    finished,
                ],
                None,
                "x <",
            ),
            (
                "stop-string",
                vec![text_token(b'x'), text_token(b'<'), text_token(b'!')],
                Some(vec!["!".to_string()]),
                "x<",
            ),
        ];

        for (request_id, events, stop_strings, expected) in cases {
            let (tx, rx) = tokio::sync::mpsc::channel(8);
            tx.try_send(EngineCoreOutput::Scheduled {
                queued_at: 1.0,
                scheduled_at: 2.0,
            })
            .unwrap();
            for event in events {
                tx.try_send(event).unwrap();
            }
            drop(tx);

            let outputs = assemble_event_stream(
                StreamInput {
                    request_id: request_id.into(),
                    event_context: event_context(),
                    prompt_token_ids: vec![b'p' as u32],
                    tokenizer: crate::serving::test_support::configured_tokenizer(),
                    prompt_logprobs_requested: false,
                    generated_logprobs_requested: false,
                    emit_token_ids: false,
                    decode_options: TextDecodeOptions {
                        stop_strings,
                        ..TextDecodeOptions::default()
                    },
                    stream: EventRx::from_receiver(rx),
                },
                OutputProcessorPolicy::SenseNova(policy.clone()),
            )
            .collect::<Vec<_>>()
            .await;

            assert!(
                outputs.iter().all(std::result::Result::is_ok),
                "{request_id}"
            );
            let visible: String = outputs
                .iter()
                .filter_map(|output| match output {
                    Ok(RequestOutput::TextDelta { text, .. }) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(visible, expected, "{request_id}");
            assert!(
                matches!(outputs.last(), Some(Ok(RequestOutput::Finished { .. }))),
                "{request_id}"
            );
        }
    }

    /// An engine failure of a running request (`Error`, or `ArtifactUnavailable`
    /// when its output cannot be delivered) is a `Failed` terminal carrying the
    /// engine message on every generation assembler, including the Qwen3 chat
    /// assembler, rather than an output-processing error.
    #[tokio::test]
    async fn assemblers_report_an_engine_failure_as_a_failed_terminal() {
        let failures = [
            EngineCoreOutput::Error {
                message: "worker call failed".to_string(),
            },
            EngineCoreOutput::ArtifactUnavailable {
                message: "worker call failed".to_string(),
            },
        ];
        for chat in [false, true] {
            for failure in failures.clone() {
                let tokenizer = crate::serving::test_support::configured_tokenizer();
                let (tx, rx) = tokio::sync::mpsc::channel(8);
                tx.try_send(EngineCoreOutput::Scheduled {
                    queued_at: 1.0,
                    scheduled_at: 2.0,
                })
                .unwrap();
                tx.try_send(EngineCoreOutput::TextToken {
                    id: b'a' as u32,
                    logprob: None,
                })
                .unwrap();
                tx.try_send(failure).unwrap();
                drop(tx);

                let input = StreamInput {
                    request_id: "engine-failure".into(),
                    event_context: event_context(),
                    prompt_token_ids: vec![b'p' as u32],
                    tokenizer: Arc::clone(&tokenizer),
                    prompt_logprobs_requested: false,
                    generated_logprobs_requested: false,
                    emit_token_ids: false,
                    decode_options: TextDecodeOptions::default(),
                    stream: EventRx::from_receiver(rx),
                };
                let events = if chat {
                    let processor = crate::serving::chat::Qwen3ChatOutputProcessor::new(
                        &mut crate::serving::chat::ChatRequest::for_test(),
                        tokenizer,
                        true,
                    )
                    .unwrap();
                    // Boxed like the runtime boxes it, since the chat pipeline
                    // is a large future.
                    Box::pin(assemble_chat_event_stream(
                        input,
                        crate::serving::chat::ChatOutputProcessor::Qwen3(processor),
                    ))
                    .collect::<Vec<_>>()
                    .await
                } else {
                    assemble_event_stream(input, OutputProcessorPolicy::None)
                        .collect::<Vec<_>>()
                        .await
                };

                assert!(
                    events.iter().all(std::result::Result::is_ok),
                    "chat={chat}: {:?}",
                    events.last()
                );
                assert!(
                    matches!(
                        events.last(),
                        Some(Ok(RequestOutput::Failed { message, .. }))
                            if message == "worker call failed"
                    ),
                    "chat={chat}: {:?}",
                    events.last()
                );
            }
        }
    }

    /// Tokens committed together reach the chat stream as one text delta per block, and a
    /// stop string inside a block ends the output at the match with the tokens through the
    /// one that completed it.
    #[tokio::test]
    async fn chat_output_publishes_each_committed_block_as_one_delta() {
        let block = |text: &str| EngineCoreOutput::TextTokens {
            ids: text.bytes().map(u32::from).collect(),
        };
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(EngineCoreOutput::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.try_send(block("first block, ")).unwrap();
        tx.try_send(block("second one. END and more")).unwrap();
        tx.try_send(block("never decoded")).unwrap();
        drop(tx);

        let mut request = crate::serving::chat::ChatRequest::for_test();
        request.decode_options.stop_strings = Some(vec!["END".to_string()]);
        let decode_options = request.decode_options.clone();
        let processor = crate::serving::chat::Qwen3ChatOutputProcessor::new(
            &mut request,
            Arc::clone(&tokenizer),
            true,
        )
        .unwrap();
        let input = StreamInput {
            request_id: "canvas-blocks".into(),
            event_context: event_context(),
            prompt_token_ids: vec![b'p' as u32],
            tokenizer,
            prompt_logprobs_requested: false,
            generated_logprobs_requested: false,
            emit_token_ids: false,
            decode_options,
            stream: EventRx::from_receiver(rx),
        };
        let events = Box::pin(assemble_chat_event_stream(
            input,
            crate::serving::chat::ChatOutputProcessor::Qwen3(processor),
        ))
        .collect::<Vec<_>>()
        .await;

        assert!(events.iter().all(std::result::Result::is_ok), "{events:?}");
        let texts: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Ok(RequestOutput::TextDelta { text, .. }) if !text.is_empty() => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect();
        // The decoder holds back two bytes, one less than the stop string,
        // until the next block rules a match in or out.
        assert_eq!(texts, vec!["first block", ", second one. "]);
        let usage = events.iter().find_map(|event| match event {
            Ok(RequestOutput::Usage {
                visible_output_tokens,
                ..
            }) => Some(*visible_output_tokens),
            _ => None,
        });
        // The first block and the second through the `D` that completes `END`.
        assert_eq!(usage, Some(13 + 15));
        assert!(
            matches!(
                events.last(),
                Some(Ok(RequestOutput::Finished {
                    reason: FinishStatus::Stop { .. },
                    ..
                }))
            ),
            "{:?}",
            events.last()
        );
    }

    /// Committed image events are held back and published immediately before the next text
    /// token (or before a terminal event or the end of the engine stream), so they stay
    /// ordered ahead of that token.
    #[tokio::test]
    async fn assembler_publishes_an_image_with_its_next_text_token() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(EngineCoreOutput::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();

        let events = assemble_event_stream(
            StreamInput {
                request_id: "feedback-output".into(),
                event_context: event_context(),
                prompt_token_ids: vec![b'p' as u32],
                tokenizer,
                prompt_logprobs_requested: false,
                generated_logprobs_requested: false,
                emit_token_ids: false,
                decode_options: TextDecodeOptions::default(),
                stream: EventRx::from_receiver(rx),
            },
            OutputProcessorPolicy::None,
        );
        tokio::pin!(events);
        assert!(matches!(
            events.next().await,
            Some(Ok(RequestOutput::Accepted { .. }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(RequestOutput::Scheduled { .. }))
        ));

        tx.send(EngineCoreOutput::TextToken {
            id: b'a' as u32,
            logprob: None,
        })
        .await
        .unwrap();
        assert!(matches!(
            events.next().await,
            Some(Ok(RequestOutput::TextDelta { text, .. })) if text == "a"
        ));

        tx.send(EngineCoreOutput::ImageCommit { image_id: 0 })
            .await
            .unwrap();
        tx.send(EngineCoreOutput::ImageDone {
            image_id: 0,
            height: 1,
            width: 1,
            bytes: 3,
            pixels_png_b64: "cG5n".to_string(),
        })
        .await
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), events.next())
                .await
                .is_err(),
            "the image became public before a continuation token arrived"
        );

        tx.send(EngineCoreOutput::TextToken {
            id: b'b' as u32,
            logprob: None,
        })
        .await
        .unwrap();
        assert!(matches!(
            events.next().await,
            Some(Ok(RequestOutput::ImageCommit { .. }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(RequestOutput::ImageDone { .. }))
        ));
    }
}
