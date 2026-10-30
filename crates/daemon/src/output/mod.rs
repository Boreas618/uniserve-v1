//! Output processing: what turns a submitted request's engine events into
//! its public events.
//!
//! `generation_output` decodes a token request's events (`text`), runs
//! them through the request's chat output processor (`chat`), whose model
//! family chose its reasoning and tool parsers (`reasoning`, `tools`), and
//! frames the result as [`RequestOutput`] events; `media_output` does the
//! same for an image or video request, whose artifacts `video` collects. The
//! parameters each request's output takes ([`OutputParams`]) come from its
//! input processing.

mod assembly;
pub mod chat;
mod events;
pub(crate) mod omni;
pub mod reasoning;
pub mod text;
pub mod tools;
pub mod video;

use std::pin::Pin;
use std::time::Instant;

use futures::Stream;

use crate::migration::RequestEvents;

pub use events::{
    FinishReason, GeneratedImage, PromptInfo, RequestOutput, RequestUsage, RuntimeTimings,
    StopReason,
};

/// Why a submitted request's engine events could not be processed into its
/// public events, for example because they violate the engine's event
/// protocol or its decoded text cannot be parsed.
#[derive(Debug, thiserror::Error)]
#[error("request `{request_id}` output processing failed")]
pub struct OutputError {
    /// Identifier of the affected request.
    pub request_id: String,
    /// Underlying decoding or parsing failure.
    #[source]
    pub source: chat::Error,
}

/// What the daemon keeps of a generation request to decode and parse its
/// output once the request has moved into the engine.
pub struct OutputParams {
    /// The model's tokenizer.
    pub tokenizer: crate::tokenizer::DynTokenizer,
    /// The prompt token IDs submitted to the engine, which seed incremental
    /// detokenization.
    pub prompt_token_ids: Vec<u32>,
    /// Incremental detokenization and stop strings.
    pub decode: text::TextDecodeOptions,
    /// Whether prompt log-probabilities are requested.
    pub prompt_logprobs: bool,
    /// Whether generated-token log-probabilities are requested.
    pub logprobs: bool,
    /// The request's reasoning and tool-call parsers.
    pub processor: chat::ChatOutputProcessor,
}

/// Boxed asynchronous stream of a request's public events.
pub type RequestOutputStream =
    Pin<Box<dyn Stream<Item = Result<RequestOutput, OutputError>> + Send>>;

/// Builds the public event stream of one submitted generation request.
///
/// `started` is when the request entered the server and `compile_us` how
/// long it took to reach the engine; both reach the terminal usage.
pub(crate) fn generation_output(
    request_id: String,
    started: Instant,
    compile_us: u64,
    output: OutputParams,
    events: RequestEvents,
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
    events: RequestEvents,
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
    use crate::migration::RequestEvents;
    use futures::StreamExt as _;
    use proto::request::Event;
    use proto::request::TokenLogprob;

    use super::*;
    use crate::output::chat::ChatOutputProcessor;
    use crate::output::text::TextDecodeOptions;

    /// The output of a request whose prompt is `p`, decoded and parsed as
    /// `processor` does, over the engine events `events`.
    fn output(
        processor: ChatOutputProcessor,
        decode: TextDecodeOptions,
        logprobs: bool,
        events: RequestEvents,
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
    fn engine_events(events: impl IntoIterator<Item = Event>) -> RequestEvents {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        for event in events {
            tx.send(event).unwrap();
        }
        RequestEvents::detached(rx)
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

    /// The visible text of `outputs`.
    fn text(outputs: &[Result<RequestOutput, OutputError>]) -> String {
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
        let events = engine_events([finished(proto::request::FinishReason::Eos, 1)]);
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
        let policy = models::omni::OutputFilterPolicy {
            reasoning: Some(models::omni::DelimitedTextPolicy {
                start: "<think>".to_string(),
                end: "</think>".to_string(),
            }),
            visible_wrappers: vec![models::omni::DelimitedTextPolicy {
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
                    finished(proto::request::FinishReason::MaxTokens, 3),
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
            let parser = crate::output::omni::SenseNovaReasoningParser::new(
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
            Event::Error {
                message: "worker call failed".to_string(),
            },
            Event::ArtifactUnavailable {
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

    /// The engine events of image `image_id`, denoised in one step.
    fn image_events(image_id: u32) -> [Event; 4] {
        [
            Event::ImageBegin {
                image_id,
                height: 1,
                width: 1,
                steps: 1,
            },
            Event::ImageStep { image_id, step: 1 },
            Event::ImageCommit { image_id },
            Event::ImageDone {
                image_id,
                height: 1,
                width: 1,
                bytes: 3,
                pixels_png_b64: "cG5n".to_string(),
                feedback: None,
            },
        ]
    }

    /// The next output of `outputs`, which the events sent so far must
    /// produce: a stream that holds it back fails the test instead of
    /// hanging it.
    async fn next_output(outputs: &mut RequestOutputStream) -> RequestOutput {
        tokio::time::timeout(std::time::Duration::from_secs(10), outputs.next())
            .await
            .expect("the events sent so far produce the next output")
            .expect("the stream goes on")
            .expect("the output is processed")
    }

    /// A completed image is published when the engine reports it, after the
    /// text decoded before it: neither the request's next image, nor its
    /// next text token, nor its end has to arrive first.
    #[tokio::test]
    async fn an_image_is_published_when_it_completes() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.send(text_token(b'a')).unwrap();
        let mut outputs = output(
            ChatOutputProcessor::default(),
            TextDecodeOptions::default(),
            false,
            RequestEvents::detached(rx),
        );

        // The outputs through the token metadata of the text before the
        // first image.
        let mut before = Vec::new();
        loop {
            match next_output(&mut outputs).await {
                RequestOutput::TokenDelta { .. } => break,
                output => before.push(Ok(output)),
            }
        }
        assert_eq!(text(&before), "a");

        // Each image is published before the engine reports anything after
        // it: the first before the second begins, the second before the
        // text after it and the end of the request.
        for image_id in [0, 1] {
            for event in image_events(image_id) {
                tx.send(event).unwrap();
            }
            let step = next_output(&mut outputs).await;
            assert!(
                matches!(step, RequestOutput::ImageStep { image_id: id } if id == image_id),
                "{step:?}"
            );
            let done = next_output(&mut outputs).await;
            assert!(
                matches!(
                    &done,
                    RequestOutput::ImageDone(GeneratedImage { image_id: id, .. })
                        if *id == image_id
                ),
                "{done:?}"
            );
        }

        tx.send(text_token(b'b')).unwrap();
        tx.send(finished(proto::request::FinishReason::MaxTokens, 2))
            .unwrap();
        let after = outputs.collect::<Vec<_>>().await;
        assert_eq!(text(&after), "b");
        assert!(
            matches!(
                after.last(),
                Some(Ok(RequestOutput::Finished {
                    reason: FinishReason::Length,
                    usage: RequestUsage {
                        image_count: 2,
                        image_steps: 2,
                        ..
                    },
                }))
            ),
            "{:?}",
            after.last()
        );
    }
}
