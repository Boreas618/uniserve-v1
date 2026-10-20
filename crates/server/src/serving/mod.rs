//! Model-aware input processing and output assembly.
//!
//! Requests follow one ownership chain:
//! `HTTP schema -> InputProcessor -> GenerationRequest -> Engine::submit ->
//! RequestOutput stream`. Video requests follow the same chain with a
//! `DiffusionRequest` and [`media_output`].
//!
//! [`InputProcessor`] renders and tokenizes a request against the loaded
//! model; [`generation_output`] and [`media_output`] assemble a submitted
//! request's engine events into public [`RequestOutput`] events. The
//! [`Engine`](crate::engine::Engine) drives both around its engine cores.

mod assembly;
pub mod chat;
mod diffusion_gemma;
mod error;
mod input;
pub mod media;
mod model;
mod omni;
mod output;
mod sampling;
pub mod systemone;
/// Text tokenization, decoding, and sampling utilities.
pub mod text;
pub mod video;

use std::pin::Pin;
use std::time::Instant;

use futures::Stream;
use uniserve_engine::EventRx;

pub use error::{ServeError, TokenizeError};
pub use input::{GenerateRequest, ImageSamplingParams, OutputParams, Prompt, SamplingParams};
pub use media::ImageInput;
pub use model::{InputProcessor, ModelResolutionError, RequestRenderer};
pub(crate) use model::{LoadedModel, LoadedRenderer};
pub use output::{
    FinishReason, GeneratedImage, PromptInfo, RequestOutput, RequestUsage, RuntimeTimings,
    StopReason,
};

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
/// `started` is when the request entered the server and `compile_us` how
/// long it took to reach the engine; both reach the terminal usage.
pub(crate) fn generation_output(
    request_id: String,
    started: Instant,
    compile_us: u64,
    output: OutputParams,
    events: EventRx,
) -> RequestOutputStream {
    Box::pin(assembly::generation_output(
        request_id, started, compile_us, output, events,
    ))
}

/// Builds the public event stream of one submitted media request, whose
/// prompt holds `prompt_tokens` tokens; see [`generation_output`].
pub(crate) fn media_output(
    request_id: String,
    started: Instant,
    compile_us: u64,
    prompt_tokens: usize,
    events: EventRx,
) -> RequestOutputStream {
    Box::pin(assembly::media_output(
        request_id,
        started,
        compile_us,
        prompt_tokens,
        events,
    ))
}

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;
    use uniserve_core::{EngineCoreOutput, TokenLogprob};
    use uniserve_engine::EventRx;

    use super::*;
    use crate::serving::chat::{ChatOutputProcessor, ChatRequest};
    use crate::serving::text::TextDecodeOptions;

    /// The output of a request whose prompt is `p`, decoded and parsed as
    /// `processor` does, over the engine events `events`.
    fn output(
        processor: ChatOutputProcessor,
        decode: TextDecodeOptions,
        logprobs: bool,
        events: EventRx,
    ) -> RequestOutputStream {
        generation_output(
            "request".into(),
            Instant::now(),
            7,
            OutputParams {
                tokenizer: crate::test_support::configured_tokenizer(),
                prompt_token_ids: vec![u32::from(b'p')],
                decode,
                prompt_logprobs: false,
                logprobs,
                processor,
            },
            events,
        )
    }

    /// An engine stream of `events` after the request's scheduling.
    fn engine_events(events: impl IntoIterator<Item = EngineCoreOutput>) -> EventRx {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tx.try_send(EngineCoreOutput::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        for event in events {
            tx.try_send(event).unwrap();
        }
        EventRx::from_receiver(rx)
    }

    fn text_token(byte: u8) -> EngineCoreOutput {
        EngineCoreOutput::TextToken {
            id: u32::from(byte),
            logprob: None,
        }
    }

    fn finished(reason: uniserve_core::FinishReason, completion_tokens: usize) -> EngineCoreOutput {
        EngineCoreOutput::Finished {
            reason,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens,
            images: 0,
        }
    }

    /// The visible text of `outputs`.
    fn text(outputs: &[Result<RequestOutput>]) -> String {
        outputs
            .iter()
            .filter_map(|output| match output {
                Ok(RequestOutput::TextDelta { text }) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn token_deltas_carry_ranked_logprobs() {
        let events = engine_events([
            text_token(b'a'),
            EngineCoreOutput::TokenLogprobs {
                id: u32::from(b'a'),
                candidates: vec![TokenLogprob {
                    token_id: u32::from(b'a'),
                    logprob: -0.25,
                    rank: 3,
                }],
            },
            finished(uniserve_core::FinishReason::MaxTokens, 1),
        ]);
        let outputs = output(
            ChatOutputProcessor::default(),
            TextDecodeOptions::default(),
            true,
            events,
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(text(&outputs), "a");
        let (token_ids, logprobs) = outputs
            .iter()
            .find_map(|output| match output {
                Ok(RequestOutput::TokenDelta {
                    token_ids,
                    logprobs,
                }) => Some((token_ids, logprobs.as_ref())),
                _ => None,
            })
            .expect("a token delta");
        assert_eq!(token_ids, &[u32::from(b'a')]);
        let candidate = &logprobs.expect("decoded logprobs").positions[0].entries[0];
        assert_eq!((candidate.token_id, candidate.rank), (u32::from(b'a'), 3));
        assert!(matches!(
            outputs.last(),
            Some(Ok(RequestOutput::Finished {
                reason: FinishReason::Length,
                usage: RequestUsage {
                    output_tokens: 1,
                    ..
                },
            }))
        ));
    }

    /// A request that finishes without publishing a text token (the engine
    /// withholds an EOS or excluded stop token, and image-only generation
    /// publishes none) returns no text. The prompt seeds decoding only.
    #[tokio::test]
    async fn a_request_without_generated_tokens_has_no_text() {
        let events = engine_events([finished(uniserve_core::FinishReason::Eos, 1)]);
        let outputs = output(
            ChatOutputProcessor::default(),
            TextDecodeOptions::default(),
            false,
            events,
        )
        .collect::<Vec<_>>()
        .await;

        assert!(outputs.iter().all(std::result::Result::is_ok));
        assert_eq!(text(&outputs), "");
        assert!(matches!(
            outputs.last(),
            Some(Ok(RequestOutput::Finished {
                reason: FinishReason::Stop(None),
                ..
            }))
        ));
    }

    /// SenseNova's parser holds back a trailing fragment that could begin a
    /// `<think>`, `</think>`, `<answer>` or `</answer>` delimiter. At the end
    /// of the stream no delimiter can complete, so the held fragment is
    /// visible text, whether the engine finished the request or a stop string
    /// did.
    #[tokio::test]
    async fn sensenova_releases_a_trailing_delimiter_prefix_at_the_end() {
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
        let cases = [
            (
                vec![
                    text_token(b'x'),
                    text_token(b' '),
                    text_token(b'<'),
                    finished(uniserve_core::FinishReason::MaxTokens, 3),
                ],
                None,
                "x <",
            ),
            (
                vec![text_token(b'x'), text_token(b'<'), text_token(b'!')],
                Some(vec!["!".to_string()]),
                "x<",
            ),
        ];

        for (events, stop_strings, expected) in cases {
            let parser = crate::serving::omni::SenseNovaReasoningParser::new(
                policy.clone(),
                crate::test_support::configured_tokenizer(),
            )
            .unwrap();
            let outputs = output(
                ChatOutputProcessor::new(Some(Box::new(parser)), None),
                TextDecodeOptions {
                    stop_strings,
                    ..TextDecodeOptions::default()
                },
                false,
                engine_events(events),
            )
            .collect::<Vec<_>>()
            .await;

            assert!(outputs.iter().all(std::result::Result::is_ok), "{expected}");
            assert_eq!(text(&outputs), expected);
            assert!(matches!(
                outputs.last(),
                Some(Ok(RequestOutput::Finished { .. }))
            ));
        }
    }

    /// An engine failure of a running request (`Error`, or
    /// `ArtifactUnavailable` when its output cannot be delivered) is a
    /// `Failed` terminal carrying the engine message rather than an
    /// output-processing error.
    #[tokio::test]
    async fn an_engine_failure_is_a_failed_terminal() {
        for failure in [
            EngineCoreOutput::Error {
                message: "worker call failed".to_string(),
            },
            EngineCoreOutput::ArtifactUnavailable {
                message: "worker call failed".to_string(),
            },
        ] {
            let outputs = output(
                ChatOutputProcessor::default(),
                TextDecodeOptions::default(),
                false,
                engine_events([text_token(b'a'), failure]),
            )
            .collect::<Vec<_>>()
            .await;

            assert!(
                matches!(
                    outputs.last(),
                    Some(Ok(RequestOutput::Failed { message })) if message == "worker call failed"
                ),
                "{:?}",
                outputs.last()
            );
        }
    }

    /// Tokens committed together reach the stream as one text delta per
    /// block, and a stop string inside a block ends the output at the match
    /// with the tokens through the one that completed it.
    #[tokio::test]
    async fn each_committed_block_is_one_delta() {
        let block = |text: &str| EngineCoreOutput::TextTokens {
            ids: text.bytes().map(u32::from).collect(),
        };
        let request = ChatRequest::for_test();
        let decode = TextDecodeOptions {
            stop_strings: Some(vec!["END".to_string()]),
            ..TextDecodeOptions::default()
        };
        let processor =
            ChatOutputProcessor::qwen3(&request, crate::test_support::configured_tokenizer(), true)
                .unwrap();
        let outputs = output(
            processor,
            decode,
            false,
            engine_events([
                block("first block, "),
                block("second one. END and more"),
                block("never decoded"),
            ]),
        )
        .collect::<Vec<_>>()
        .await;

        assert!(
            outputs.iter().all(std::result::Result::is_ok),
            "{outputs:?}"
        );
        let texts: Vec<_> = outputs
            .iter()
            .filter_map(|output| match output {
                Ok(RequestOutput::TextDelta { text }) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        // The decoder holds back two bytes, one less than the stop string,
        // until the next block rules a match in or out.
        assert_eq!(texts, vec!["first block", ", second one. "]);
        // The first block and the second through the `D` that completes `END`.
        assert!(
            matches!(
                outputs.last(),
                Some(Ok(RequestOutput::Finished {
                    reason: FinishReason::Stop(Some(StopReason::String(stop))),
                    usage: RequestUsage { output_tokens: 28, .. },
                })) if stop == "END"
            ),
            "{:?}",
            outputs.last()
        );
    }

    /// A completed image is held back and published immediately before the
    /// next text token (or the terminal event), so it stays ordered ahead of
    /// that token.
    #[tokio::test]
    async fn an_image_is_published_with_its_next_text_token() {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(EngineCoreOutput::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        let outputs = output(
            ChatOutputProcessor::default(),
            TextDecodeOptions::default(),
            false,
            EventRx::from_receiver(rx),
        );
        tokio::pin!(outputs);
        assert!(matches!(
            outputs.next().await,
            Some(Ok(RequestOutput::Accepted(_)))
        ));
        assert!(matches!(
            outputs.next().await,
            Some(Ok(RequestOutput::Scheduled { .. }))
        ));

        tx.send(text_token(b'a')).await.unwrap();
        while !matches!(
            outputs.next().await,
            Some(Ok(RequestOutput::TokenDelta { .. }))
        ) {}

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
            tokio::time::timeout(std::time::Duration::from_millis(20), outputs.next())
                .await
                .is_err(),
            "the image became public before a continuation token arrived"
        );

        tx.send(text_token(b'b')).await.unwrap();
        assert!(matches!(
            outputs.next().await,
            Some(Ok(RequestOutput::ImageDone(GeneratedImage {
                image_id: 0,
                ..
            })))
        ));
    }
}
