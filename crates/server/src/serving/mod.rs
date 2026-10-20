//! Model-aware input processing and output assembly.
//!
//! Requests follow one ownership chain:
//! `HTTP schema -> InputProcessor -> GenerationRequest -> Engine::submit ->
//! RequestOutput stream`. Video requests follow the same chain with a
//! `DiffusionRequest` and `assembly::assemble_media_event_stream`.
//!
//! [`InputProcessor`] renders and tokenizes a request against the loaded
//! model; [`generation_output`] and [`media_output`] assemble a submitted
//! request's engine events into public [`RequestOutput`] events. The
//! [`Engine`](crate::engine::Engine) drives both around its engine cores.

mod assembly;
pub mod chat;
mod diffusion_gemma;
pub(crate) use self::diffusion_gemma::canvas_sampling;
mod error;
mod input;
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
use std::time::Instant;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _};
use serde::{Deserialize, Serialize};
use uniserve_core::EngineCoreOutput;
use uniserve_engine::{EventRx, StreamCancelCause};

pub use input::{
    DecodeControls, ImageGenControls, ModalitySelection, OutputDetail, OutputProcessorPolicy,
    PromptInput, ResponseOptions, SamplingConfig, StopConfig,
};
pub use media::ImageInput;
pub use model::{InputProcessor, RequestRenderer, WorkerCapabilities};
pub(crate) use model::{LoadedModel, LoadedRenderer};
pub use preprocessing::chat_image_urls;

use crate::serving::omni::{SenseNovaOutputProcessor, SenseNovaTextDelta};
use crate::serving::text::output::stop_string_holdback_bytes;
use crate::serving::text::{
    DecodedLogprobs, DecodedPromptLogprobs, FinishReason, StopReason, TextDecodeOptions,
};

use assembly::{assemble_chat_event_stream, assemble_event_stream};

use error::malformed_output;
pub use error::{OutputProcessingError, ServeError, TokenizeError};
pub use output::{FinishStatus, RequestOutput, RuntimeTimings, StopCause};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
/// Caller-visible request identifier used throughout serving.
///
/// The type performs no validation. The HTTP routes build it from the resolved request ID
/// with an endpoint prefix (for example `chatcmpl-`); it names the request in its response,
/// events and logs, and two live requests may share it.
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

/// Folds a request's cache salt into a stable isolation key.
///
/// Returns `None` when the salt is absent, which leaves the request in the shared cache
/// partition. Otherwise the key is the 64-bit FNV-1a hash of the salt, which is
/// deterministic across processes.
pub(crate) fn cache_isolation_key(salt: Option<&str>) -> Option<u64> {
    salt.map(|salt| uniserve_core::Fnv1a::hash(salt.as_bytes()))
}

/// Builds the public event stream of one submitted generation request.
///
/// `started` is when the request entered the server and `compile_duration_us`
/// how long it took to reach the engine; both reach the stream's `Accepted`
/// and `Usage` events. Chat output policies run through
/// [`assemble_chat_event_stream`], every other policy through
/// [`assemble_event_stream`].
pub(crate) fn generation_output(
    request_id: ServeRequestId,
    started: Instant,
    compile_duration_us: u64,
    response: ResponseOptions,
    stream: EventRx,
) -> RequestOutputStream {
    let ResponseOptions {
        tokenizer,
        prompt_token_ids,
        decode,
        emit_token_ids,
        prompt_logprobs_requested,
        generated_logprobs_requested,
        output_processor,
    } = response;
    let assembly = StreamInput {
        request_id,
        event_context: EventContext {
            started,
            compile_duration_us,
        },
        prompt_token_ids,
        tokenizer,
        prompt_logprobs_requested,
        generated_logprobs_requested,
        emit_token_ids,
        decode_options: decode,
        stream,
    };
    match output_processor {
        OutputProcessorPolicy::Chat(processor) => {
            Box::pin(assemble_chat_event_stream(assembly, processor))
        }
        // `assemble_event_stream` applies the remaining policies as its output
        // sink.
        output_processor => Box::pin(assemble_event_stream(assembly, output_processor)),
    }
}

/// Builds the public event stream of one submitted media request, whose
/// prompt holds `prompt_tokens` tokens; see [`generation_output`].
pub(crate) fn media_output(
    request_id: ServeRequestId,
    started: Instant,
    compile_duration_us: u64,
    prompt_tokens: usize,
    stream: EventRx,
) -> RequestOutputStream {
    Box::pin(assembly::assemble_media_event_stream(
        request_id,
        EventContext {
            started,
            compile_duration_us,
        },
        prompt_tokens,
        stream,
    ))
}

/// Request-scoped metadata the assemblers stamp onto public events and usage.
#[derive(Debug, Clone)]
struct EventContext {
    started: Instant,
    compile_duration_us: u64,
}

/// Engine receiver and the response options a generation assembler consumes.
struct StreamInput {
    request_id: ServeRequestId,
    event_context: EventContext,
    prompt_token_ids: Vec<u32>,
    tokenizer: crate::profile::tokenizer::DynTokenizer,
    prompt_logprobs_requested: bool,
    generated_logprobs_requested: bool,
    emit_token_ids: bool,
    decode_options: TextDecodeOptions,
    stream: EventRx,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use uniserve_core::TokenLogprob;
    use uniserve_engine::EventRx;

    fn event_context() -> EventContext {
        EventContext {
            started: Instant::now(),
            compile_duration_us: 7,
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
