//! Incremental conversion from engine events to decoded text events.
//!
//! [`decoded_text_event_stream`] consumes one generation request's `EventStream`,
//! checks the engine event protocol, detokenizes generated tokens with the
//! tokenizer's `IncrementalDecoder`, decodes requested logprobs, matches stop
//! strings, and passes image events through in order. `serving::assembly`
//! feeds its output to the request's chat output processor.
//!
//! The decoder also reports its stop-string decisions to the engine: it
//! acknowledges each token only after checking that it completes no stop
//! string, and on a match stops generation with
//! `EventStream::cancel_at_consumed_prefix`.

use std::sync::Arc;

use crate::client::EventStream;
use asynk_strim_attr::{TryYielder, try_stream};
use proto::request::{Event, PositionLogprobs, RejectionKind};
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, trace};

use super::logprobs::{DecodedLogprobs, decode_logprobs, decode_prompt_logprobs};
use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer, IncrementalDecoder};
use crate::serving::output::{FinishReason, GeneratedImage, PromptInfo, StopReason};
use crate::serving::text::error::Error;

/// Request-neutral options for incremental text decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDecodeOptions {
    /// Whether tokenizer special tokens are omitted from decoded text.
    pub skip_special_tokens: bool,
    /// Whether the matched stop string remains in emitted output.
    pub include_stop_str_in_output: bool,
    /// Decoded strings that terminate generation.
    pub stop_strings: Option<Vec<String>>,
    /// Minimum number of generated tokens before stop strings can terminate output.
    pub min_tokens: u32,
}

impl Default for TextDecodeOptions {
    /// Skips special tokens, excludes a matched stop string from output, and
    /// configures no stop strings or minimum length.
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
            include_stop_str_in_output: false,
            stop_strings: None,
            min_tokens: 0,
        }
    }
}

/// How a decoded request finished, with its token counts.
#[derive(Debug, Clone, PartialEq)]
pub struct Finished {
    /// Why the request finished.
    pub reason: FinishReason,
    /// Prompt tokens submitted to the engine.
    pub prompt_tokens: usize,
    /// Generated tokens.
    ///
    /// The engine's `completion_tokens` when the engine finishes the request,
    /// which counts the tokens of internal sections it never published as
    /// text; the decoder's own count when a stop string ends the stream.
    pub output_tokens: usize,
    /// Images the engine reports generating.
    pub images: usize,
}

/// An image-generation event of an interleaved text-and-image request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageEvent {
    /// Generation of an image began; the request has visible output.
    Begin,
    /// One denoising step of the image completed.
    Step {
        /// Request-local image identifier.
        image_id: u32,
    },
    /// The image is complete.
    Done(GeneratedImage),
}

/// Decoded output of one generation request.
///
/// A stream yields one `Start`, then `TextDelta` and `Image` events in engine
/// order, then one terminal event: `Done`, `Rejected`, or `Failed`. An
/// engine refusal arrives before `Start`.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedTextEvent {
    /// The engine scheduled the request and its prompt metadata is complete.
    Start {
        /// The prompt, with its log-probabilities when requested.
        prompt: PromptInfo,
        /// Engine wall-clock time, in UNIX seconds, at which the request
        /// entered the engine queue.
        queued_at: f64,
        /// Engine wall-clock time, in UNIX seconds, at which the scheduler
        /// admitted the request.
        scheduled_at: f64,
    },
    /// Newly decoded text and the tokens it decodes.
    ///
    /// Each update covers the tokens of one engine output event: one
    /// generated token, or a block of tokens committed together (its text
    /// may be empty while the decoder holds bytes back). The last update
    /// before `Done` carries the flushed remainder of the text and, when a
    /// stop string ended the stream, the tokens of its event through the
    /// matching one.
    TextDelta {
        /// Newly visible decoded text.
        delta: String,
        /// Token identifiers of this update.
        token_ids: Vec<u32>,
        /// Per-position candidate log-probabilities, when requested.
        logprobs: Option<DecodedLogprobs>,
    },
    /// An image event. A completed image is held until the next generated
    /// token or the end of the request, so it stays ordered ahead of the
    /// text that follows it.
    Image(ImageEvent),
    /// The request finished.
    Done(Finished),
    /// The engine refused the request before executing it.
    Rejected {
        /// Whether the request is unservable or the engine is full.
        kind: RejectionKind,
        /// Engine-supplied rejection detail.
        message: String,
    },
    /// The engine failed the request or could not deliver its output.
    Failed {
        /// Engine-supplied failure detail.
        message: String,
    },
}

/// Result of decoding one generated token.
struct ArDecode {
    /// Newly visible text; empty when a stop string matched.
    delta: String,
    /// Matched stop string and the byte offset of its start in
    /// `IncrementalDecoder::output`.
    stop: Option<(String, usize)>,
}

/// Per-request decoding state for [`decoded_text_event_stream`].
struct DecodeState<'a> {
    decoder: IncrementalDecoder<'a>,
    /// Decode options; matched stop strings are removed from `stop_strings`.
    options: TextDecodeOptions,
    /// Prompt logprob positions received so far.
    prompt_positions: Vec<PositionLogprobs>,
    /// `(queued_at, scheduled_at)` of the engine's scheduling event.
    scheduled: Option<(f64, f64)>,
    /// Whether `DecodedTextEvent::Start` has been yielded.
    started: bool,
    /// Generated token awaiting its `TokenLogprobs` event when generated
    /// logprobs are requested.
    pending_token: Option<u32>,
    /// Completed images held until the next generated token or the end of
    /// the request.
    pending_images: Vec<GeneratedImage>,
    /// Generated text tokens consumed by the decoder.
    output_tokens: usize,
}

impl DecodeState<'_> {
    /// Emits start metadata once scheduling and any requested prompt scores are complete.
    ///
    /// Idempotent: does nothing after `Start` has been yielded or while a
    /// prerequisite is missing. The handlers of generated output check
    /// `started` afterwards to reject output that arrives too early.
    async fn emit_start_if_ready(
        &mut self,
        request_id: &str,
        tokenizer: &HuggingFaceTokenizer,
        prompt_token_ids: &[u32],
        prompt_logprobs_requested: bool,
        y: &mut TryYielder<DecodedTextEvent, Error>,
    ) -> Result<(), Error> {
        // The first prompt token has no left context to score, so prompt
        // logprobs cover every position after it (see `DecodedPromptLogprobs`).
        let prompt_ready =
            !prompt_logprobs_requested || self.prompt_positions.len() == prompt_token_ids.len() - 1;
        let Some((queued_at, scheduled_at)) = self.scheduled else {
            return Ok(());
        };
        if self.started || !prompt_ready {
            return Ok(());
        }
        let logprobs = prompt_logprobs_requested
            .then(|| {
                decode_prompt_logprobs(
                    request_id,
                    tokenizer,
                    prompt_token_ids,
                    &self.prompt_positions,
                    self.options.skip_special_tokens,
                )
            })
            .transpose()?;
        y.yield_ok(DecodedTextEvent::Start {
            prompt: PromptInfo {
                token_ids: Arc::from(prompt_token_ids),
                logprobs,
            },
            queued_at,
            scheduled_at,
        })
        .await;
        self.started = true;
        Ok(())
    }

    /// Fails unless `Start` has been yielded and no generated token awaits
    /// its logprobs, the state every event after `Start` requires.
    fn ensure_output_ready(&self, request_id: &str, event: &str) -> Result<(), Error> {
        if self.started && self.pending_token.is_none() {
            Ok(())
        } else {
            Err(Error::MalformedOutput {
                request_id: request_id.to_owned(),
                message: format!("{event} arrived before output metadata was complete"),
            })
        }
    }

    /// Yields the held completed images in arrival order.
    async fn flush_images(&mut self, y: &mut TryYielder<DecodedTextEvent, Error>) {
        for image in self.pending_images.drain(..) {
            y.yield_ok(DecodedTextEvent::Image(ImageEvent::Done(image)))
                .await;
        }
    }

    /// Decodes committed tokens in order and applies stop-string holdback.
    ///
    /// `token_ids` are the tokens of one engine output event and `positions`
    /// their requested logprobs, if any. Returns `true` when a token completed
    /// a stop string. In that case the engine has been told to stop at the
    /// tokens received so far, the decoder has been flushed and truncated at
    /// the match, and the final `TextDelta`, covering the tokens through the
    /// matching one, and `Done` have been yielded, so the caller must end the
    /// stream. Otherwise the tokens are acknowledged to the engine and
    /// yielded as one `TextDelta`.
    async fn consume_tokens(
        &mut self,
        tokenizer: &HuggingFaceTokenizer,
        prompt_tokens: usize,
        token_ids: &[u32],
        positions: Vec<PositionLogprobs>,
        raw_stream: &mut EventStream,
        y: &mut TryYielder<DecodedTextEvent, Error>,
    ) -> Result<bool, Error> {
        // Images completed before these tokens stay ordered ahead of them.
        self.flush_images(y).await;
        let decoded_logprobs = (!positions.is_empty())
            .then(|| decode_logprobs(tokenizer, &positions, self.options.skip_special_tokens))
            .transpose()?;

        let mut delta = String::new();
        for (index, &token_id) in token_ids.iter().enumerate() {
            let decoded = decode_one_token(
                &mut self.decoder,
                token_id,
                self.output_tokens,
                &mut self.options,
            )?;
            self.output_tokens = self.output_tokens.saturating_add(1);
            delta.push_str(&decoded.delta);
            let Some((stop_string, offset)) = decoded.stop else {
                continue;
            };

            // Stop generation at exactly the tokens received so far, which
            // include this one.
            raw_stream
                .cancel_at_consumed_prefix(crate::client::StreamCancelCause::StopStringMatched);
            let truncate_to = Some(if self.options.include_stop_str_in_output {
                offset + stop_string.len()
            } else {
                offset
            });
            // `last_chunk` is the not-yet-emitted remainder of the truncated
            // text.
            let (last_chunk, _) = self.decoder.flush(truncate_to)?;
            y.yield_ok(DecodedTextEvent::TextDelta {
                delta: delta + last_chunk.unwrap_or_default().as_str(),
                token_ids: token_ids[..=index].to_vec(),
                logprobs: decoded_logprobs,
            })
            .await;
            y.yield_ok(DecodedTextEvent::Done(Finished {
                reason: FinishReason::Stop(Some(StopReason::String(stop_string))),
                prompt_tokens,
                output_tokens: self.output_tokens,
                images: 0,
            }))
            .await;
            return Ok(true);
        }
        raw_stream.acknowledge_consumed_prefix();
        y.yield_ok(DecodedTextEvent::TextDelta {
            delta,
            token_ids: token_ids.to_vec(),
            logprobs: decoded_logprobs,
        })
        .await;
        Ok(false)
    }
}

/// Advances incremental decoding and returns only bytes newly made visible by this token.
///
/// Stop strings are checked only from token number `min_tokens + 1` onward
/// (`output_tokens` counts the tokens before this one). A matched stop
/// string is removed from `options.stop_strings`.
fn decode_one_token(
    decoder: &mut IncrementalDecoder<'_>,
    token_id: u32,
    output_tokens: usize,
    options: &mut TextDecodeOptions,
) -> Result<ArDecode, Error> {
    let new_bytes = decoder.push_token(token_id)?;
    let stop = if output_tokens + 1 > options.min_tokens as usize {
        if let Some(stops) = options.stop_strings.as_mut()
            && let Some((index, offset)) = matches_stop_string(stops, decoder.output(), new_bytes)
        {
            Some((stops.swap_remove(index), offset))
        } else {
            None
        }
    } else {
        None
    };
    let delta = if stop.is_none() {
        decoder.next_chunk().unwrap_or_default()
    } else {
        String::new()
    };
    Ok(ArDecode { delta, stop })
}

/// Decodes one generation request's engine events into decoded text events.
///
/// On success the stream follows the protocol of [`DecodedTextEvent`]. When
/// `generated_logprobs_requested` is set, every `TextToken` must be followed
/// by a `TokenLogprobs` event for the same token whose first candidate is
/// that token. An engine error finish fails the request like an engine error
/// event.
///
/// # Errors
///
/// Fails with [`Error::EmptyPromptTokenIds`] for an empty prompt,
/// [`Error::MalformedOutput`] when engine events violate the protocol,
/// [`Error::StreamClosedBeforeTerminalOutput`] when the channel closes
/// without a terminal event, and a tokenizer error when decoding fails.
#[try_stream]
pub async fn decoded_text_event_stream(
    request_id: String,
    tokenizer: DynTokenizer,
    prompt_token_ids: Vec<u32>,
    prompt_logprobs_requested: bool,
    generated_logprobs_requested: bool,
    mut raw_stream: EventStream,
    decode_options: TextDecodeOptions,
    mut y: TryYielder<DecodedTextEvent, Error>,
) -> crate::serving::text::Result<()> {
    // The prompt seeds incremental decoding and supplies the expected prompt-
    // logprob cardinality before any generated token can be accepted.
    if prompt_token_ids.is_empty() {
        return Err(Error::EmptyPromptTokenIds {
            request_id: request_id.clone(),
        });
    }
    let prompt_tokens = prompt_token_ids.len();
    let decoder = tokenizer.create_decode_stream(
        &prompt_token_ids,
        decode_options.skip_special_tokens,
        stop_string_holdback_bytes(&decode_options),
    );
    let mut state = DecodeState {
        decoder,
        options: decode_options,
        prompt_positions: Vec::new(),
        scheduled: None,
        started: false,
        pending_token: None,
        pending_images: Vec::new(),
        output_tokens: 0,
    };

    // Engine events form an ordered protocol: scheduling and optional prompt
    // metadata establish Start, then each token is paired with optional logprobs.
    while let Some(event) = raw_stream.next().await {
        match event {
            // Scheduling metadata may arrive before prompt logprobs; start is
            // emitted only after both prerequisites are complete.
            Event::Scheduled {
                queued_at,
                scheduled_at,
            } => {
                state.scheduled = Some((queued_at, scheduled_at));
                state
                    .emit_start_if_ready(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        prompt_logprobs_requested,
                        &mut y,
                    )
                    .await?;
            }
            Event::PromptLogprobs { positions } => {
                if !prompt_logprobs_requested {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine returned unrequested prompt logprobs".to_string(),
                    });
                }
                state.prompt_positions.extend(positions);
                if state.prompt_positions.len() >= prompt_tokens {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine returned too many prompt logprob positions".to_string(),
                    });
                }
                state
                    .emit_start_if_ready(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        prompt_logprobs_requested,
                        &mut y,
                    )
                    .await?;
            }
            // Generated logprobs, when requested, defer decoding until the
            // matching TokenLogprobs event validates the token identity.
            Event::TextToken { id, .. } => {
                state
                    .emit_start_if_ready(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        prompt_logprobs_requested,
                        &mut y,
                    )
                    .await?;
                state.ensure_output_ready(&request_id, "a generated token")?;
                if generated_logprobs_requested {
                    state.pending_token = Some(id);
                } else if state
                    .consume_tokens(
                        tokenizer.as_ref(),
                        prompt_tokens,
                        &[id],
                        Vec::new(),
                        &mut raw_stream,
                        &mut y,
                    )
                    .await?
                {
                    return Ok(());
                }
            }
            // Tokens committed together, such as a block-diffusion canvas,
            // carry no logprobs and decode as one update.
            Event::TextTokens { ids } => {
                state
                    .emit_start_if_ready(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        prompt_logprobs_requested,
                        &mut y,
                    )
                    .await?;
                state.ensure_output_ready(&request_id, "a block of committed tokens")?;
                if generated_logprobs_requested {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine emitted committed tokens without their requested logprobs"
                            .to_string(),
                    });
                }
                if state
                    .consume_tokens(
                        tokenizer.as_ref(),
                        prompt_tokens,
                        &ids,
                        Vec::new(),
                        &mut raw_stream,
                        &mut y,
                    )
                    .await?
                {
                    return Ok(());
                }
            }
            Event::TokenLogprobs { id, candidates } => {
                let pending = state
                    .pending_token
                    .take()
                    .ok_or_else(|| Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine returned token logprobs without a pending token"
                            .to_string(),
                    })?;
                // The event protocol lists the sampled token as the first candidate.
                if pending != id || candidates.first().is_none_or(|entry| entry.token_id != id) {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "token logprobs do not match the emitted token".to_string(),
                    });
                }
                if state
                    .consume_tokens(
                        tokenizer.as_ref(),
                        prompt_tokens,
                        &[id],
                        vec![PositionLogprobs {
                            entries: candidates,
                        }],
                        &mut raw_stream,
                        &mut y,
                    )
                    .await?
                {
                    return Ok(());
                }
            }
            Event::ImageBegin { .. } => {
                state.ensure_output_ready(&request_id, "an image start")?;
                y.yield_ok(DecodedTextEvent::Image(ImageEvent::Begin)).await;
            }
            Event::ImageStep { image_id, .. } => {
                state.ensure_output_ready(&request_id, "an image step")?;
                y.yield_ok(DecodedTextEvent::Image(ImageEvent::Step { image_id }))
                    .await;
            }
            // A commit marks the image's entry into the model context; the
            // completed image follows it.
            Event::ImageCommit { .. } => {
                state.ensure_output_ready(&request_id, "an image commit")?;
            }
            Event::ImageDone {
                image_id,
                height,
                width,
                bytes,
                pixels_png_b64,
            } => {
                state.ensure_output_ready(&request_id, "a completed image")?;
                state.pending_images.push(GeneratedImage {
                    image_id,
                    width,
                    height,
                    bytes,
                    png_base64: pixels_png_b64,
                });
            }
            // A terminal event flushes decoder holdback only after every pending
            // token/logprob pair and all prompt metadata have resolved.
            Event::Finished {
                reason,
                stop_reason,
                completion_tokens,
                images,
                ..
            } => {
                state
                    .emit_start_if_ready(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        prompt_logprobs_requested,
                        &mut y,
                    )
                    .await?;
                state.ensure_output_ready(&request_id, "the terminal event")?;
                state.flush_images(&mut y).await;
                let Some(reason) = FinishReason::from_engine(reason, stop_reason) else {
                    y.yield_ok(DecodedTextEvent::Failed {
                        message: "engine execution failed".to_owned(),
                    })
                    .await;
                    return Ok(());
                };

                // Release the decoder's holdback untruncated: a stop-string
                // match would have ended the stream in `consume_tokens`.
                let (last_chunk, text) = state.decoder.flush(None)?;
                debug!(
                    ?reason,
                    output_tokens = state.output_tokens,
                    "request finished with decoded text"
                );
                if tracing::enabled!(Level::TRACE) {
                    trace!(full_text = text, "terminal decoded text");
                }

                y.yield_ok(DecodedTextEvent::TextDelta {
                    delta: last_chunk.unwrap_or_default(),
                    token_ids: Vec::new(),
                    logprobs: None,
                })
                .await;
                y.yield_ok(DecodedTextEvent::Done(Finished {
                    reason,
                    prompt_tokens,
                    output_tokens: completion_tokens,
                    images,
                }))
                .await;
                return Ok(());
            }
            Event::Rejected { kind, message } => {
                y.yield_ok(DecodedTextEvent::Rejected { kind, message })
                    .await;
                return Ok(());
            }
            Event::Error { message } | Event::ArtifactUnavailable { message } => {
                state.flush_images(&mut y).await;
                y.yield_ok(DecodedTextEvent::Failed { message }).await;
                return Ok(());
            }
            // Media artifacts belong to diffusion requests, and readout
            // answers to readout requests, which other consumers collect.
            Event::Artifact(_) | Event::MediaProgress { .. } | Event::Readout { .. } => {
                return Err(Error::MalformedOutput {
                    request_id: request_id.clone(),
                    message: "a generation request received a media or readout event".to_string(),
                });
            }
        }
    }

    Err(Error::StreamClosedBeforeTerminalOutput { request_id })
}

/// Returns the suffix byte count retained to detect cross-chunk stop strings.
///
/// Holding back one byte less than the longest stop string keeps any partial
/// match out of emitted text until the next token confirms or rules it out.
/// No holdback is needed when matched stop strings stay in the output. The
/// result is passed to the decoder as its `min_bytes_to_buffer`.
fn stop_string_holdback_bytes(options: &TextDecodeOptions) -> usize {
    if options.include_stop_str_in_output {
        return 0;
    }
    options
        .stop_strings
        .as_ref()
        .and_then(|stops| stops.iter().map(String::len).max())
        .unwrap_or(1)
        .saturating_sub(1)
}

/// Returns the configured stop string that completes earliest in newly
/// decoded text.
///
/// Only matches that include at least one of the last `new_bytes` bytes of
/// `output` are considered, so earlier text is not rescanned and nothing
/// matches when `new_bytes` is zero. Each stop string is located at its first
/// occurrence in that range, and the one whose match ends earliest wins, with
/// ties going to the earlier entry in `stops`. The result is the one a
/// decoder that received the new bytes one at a time would report, and
/// truncating `output` at the match start removes every stop occurrence
/// (vLLM's `check_stop_strings` selects the same match). Returns the stop
/// string's index in `stops` and the byte offset of the match start in
/// `output`.
///
/// Matching compares bytes. A match of a complete UTF-8 stop string in UTF-8
/// text starts and ends on character boundaries, so the offset is a valid
/// truncation point.
fn matches_stop_string(stops: &[String], output: &str, new_bytes: usize) -> Option<(usize, usize)> {
    let output = output.as_bytes();
    // A match must end at or after this offset to include a new byte.
    let next_offset = (output.len() + 1).saturating_sub(new_bytes);
    stops
        .iter()
        .enumerate()
        .filter_map(|(index, stop)| {
            let stop = stop.as_bytes();
            let search_start = next_offset.saturating_sub(stop.len());
            output[search_start..]
                .windows(stop.len())
                .position(|window| window == stop)
                .map(|position| {
                    let match_start = search_start + position;
                    (index, match_start, match_start + stop.len())
                })
        })
        // `min_by_key` keeps the first of equal keys, so ties go to list order.
        .min_by_key(|&(_, _, match_end)| match_end)
        .map(|(index, match_start, _)| (index, match_start))
}

#[cfg(test)]
mod tests {
    use super::matches_stop_string;

    fn stops(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    /// Truncating at the returned offset must remove every occurrence of the
    /// stop string, so a new chunk holding it twice (Qwen3 encodes `"\n\n"`
    /// as one token) matches at its first occurrence.
    #[test]
    fn stop_string_matches_its_first_occurrence_in_new_text() {
        assert_eq!(
            matches_stop_string(&stops(&["\n"]), "Answer: 42\n\n", 2),
            Some((0, 10))
        );
    }

    /// Among stop strings matching in the same new chunk, the one that
    /// completes earliest wins, as if tokens arrived one byte at a time;
    /// list order breaks ties.
    #[test]
    fn stop_string_that_completes_earliest_wins() {
        assert_eq!(
            matches_stop_string(&stops(&["b", "a"]), "xab", 2),
            Some((1, 1))
        );
        assert_eq!(
            matches_stop_string(&stops(&["ab", "b"]), "xab", 2),
            Some((0, 1))
        );
    }
}
