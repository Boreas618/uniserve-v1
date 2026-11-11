//! Incremental conversion from engine events to decoded text events.
//!
//! [`DecodeStream`] consumes one generation request's `RequestEvents`,
//! checks the engine event protocol, detokenizes generated tokens with the
//! tokenizer's `IncrementalDecoder`, decodes requested logprobs, matches stop
//! strings, and passes image events through in order. The request's
//! `OutputProcessor` feeds its output to the reasoning and tool-call parsers.
//!
//! The decoder also reports its stop-string decisions to the engine: it
//! acknowledges each token only after checking that it completes no stop
//! string, and on a match stops generation with
//! `RequestEvents::cancel_at_consumed_prefix`.

use std::sync::Arc;

use crate::events::Events;
use proto::handoff::TextDecode;
use proto::request::{Event, PositionLogprobs, RejectionKind};
use tracing::{Level, debug, trace};

use super::logprobs::{DecodedLogprobs, decode_logprobs, decode_prompt_logprobs};
use crate::output::events::{FinishReason, PromptInfo, StopReason};
use crate::output::text::error::Error;
use crate::tokenizer::{DynTokenizer, IncrementalDecoder};

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
}

/// Decoded output of one generation request.
///
/// A stream yields one `Start`, then `TextDelta` events in engine
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

/// One generation stream's decoding state, advanced one engine event at a
/// time.
///
/// The stream keeps everything a successor process needs to go on with it
/// (`decode` and `resume`, design §7.5.1): the incremental decoder's window and
/// published text, the stop strings not matched yet, the prompt metadata, and
/// the token awaiting its logprobs.
pub struct DecodeStream {
    request_id: String,
    tokenizer: DynTokenizer,
    prompt_token_ids: Vec<u32>,
    prompt_logprobs_requested: bool,
    generated_logprobs_requested: bool,
    decoder: IncrementalDecoder,
    /// Decode options; matched stop strings are removed from `stop_strings`.
    options: TextDecode,
    /// Prompt logprob positions received so far.
    prompt_positions: Vec<PositionLogprobs>,
    /// `(queued_at, scheduled_at)` of the engine's scheduling event.
    scheduled: Option<(f64, f64)>,
    /// Whether `DecodedTextEvent::Start` has been emitted.
    started: bool,
    /// Generated token awaiting its `TokenLogprobs` event when generated
    /// logprobs are requested.
    pending_token: Option<u32>,
    /// Generated text tokens consumed by the decoder.
    output_tokens: usize,
}

/// The decoded events one engine event produced.
pub struct Decoded {
    /// The events, in order; a terminal event is the last.
    pub events: Vec<DecodedTextEvent>,
}

impl Decoded {
    fn new(events: Vec<DecodedTextEvent>) -> Self {
        Self { events }
    }
}

impl DecodeStream {
    /// A decoder for one generation request, before any engine event.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::EmptyPromptTokenIds`] when the prompt is empty.
    pub fn new(
        request_id: String,
        tokenizer: DynTokenizer,
        prompt_token_ids: Vec<u32>,
        prompt_logprobs_requested: bool,
        generated_logprobs_requested: bool,
        decode_options: TextDecode,
    ) -> Result<Self, Error> {
        if prompt_token_ids.is_empty() {
            return Err(Error::EmptyPromptTokenIds { request_id });
        }
        let decoder = tokenizer.create_decode_stream(
            &prompt_token_ids,
            decode_options.skip_special_tokens,
            stop_string_holdback_bytes(&decode_options),
        );
        Ok(Self {
            request_id,
            tokenizer,
            prompt_token_ids,
            prompt_logprobs_requested,
            generated_logprobs_requested,
            decoder,
            options: decode_options,
            prompt_positions: Vec::new(),
            scheduled: None,
            started: false,
            pending_token: None,
            output_tokens: 0,
        })
    }

    /// The prompt's token identifiers, which seeded the decoder.
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    /// The decode options, with the stop strings not matched yet.
    pub fn options(&self) -> &TextDecode {
        &self.options
    }

    /// Emits start metadata once scheduling and any requested prompt scores
    /// are complete.
    fn start_event(&mut self, prompt_token_ids: &[u32]) -> Result<Option<DecodedTextEvent>, Error> {
        // The first prompt token has no left context to score, so prompt
        // logprobs cover every position after it (see `DecodedPromptLogprobs`).
        let prompt_ready = !self.prompt_logprobs_requested
            || self.prompt_positions.len() + 1 == prompt_token_ids.len();
        let Some((queued_at, scheduled_at)) = self.scheduled else {
            return Ok(None);
        };
        if self.started || !prompt_ready {
            return Ok(None);
        }
        let logprobs = self
            .prompt_logprobs_requested
            .then(|| {
                decode_prompt_logprobs(
                    &self.request_id,
                    self.tokenizer.as_ref(),
                    prompt_token_ids,
                    &self.prompt_positions,
                    self.options.skip_special_tokens,
                )
            })
            .transpose()?;
        self.started = true;
        Ok(Some(DecodedTextEvent::Start {
            prompt: PromptInfo {
                token_ids: Arc::from(prompt_token_ids),
                logprobs,
            },
            queued_at,
            scheduled_at,
        }))
    }

    /// Pushes one engine event and returns the decoded events it produced.
    ///
    /// `raw_stream` is the request's stream: a token that completes a stop
    /// string cancels generation at the tokens received so far and acknowledges
    /// the tokens otherwise.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::MalformedOutput`] when engine events violate the
    /// protocol of [`DecodedTextEvent`], and with a tokenizer error when
    /// decoding fails.
    pub fn push(&mut self, event: Event, raw_stream: &mut dyn Events) -> Result<Decoded, Error> {
        let prompt_token_ids = self.prompt_token_ids.clone();
        let prompt_tokens = prompt_token_ids.len();
        let mut events = Vec::new();
        match event {
            // Scheduling metadata may arrive before prompt logprobs; start is
            // emitted only after both prerequisites are complete.
            Event::Scheduled {
                queued_at,
                scheduled_at,
            } => {
                self.scheduled = Some((queued_at, scheduled_at));
                if let Some(start) = self.start_event(&prompt_token_ids)? {
                    events.push(start);
                }
            }
            Event::PromptLogprobs { positions } => {
                if !self.prompt_logprobs_requested {
                    return Err(Error::MalformedOutput {
                        request_id: self.request_id.clone(),
                        message: "engine returned unrequested prompt logprobs".to_string(),
                    });
                }
                self.prompt_positions.extend(positions);
                if self.prompt_positions.len() >= prompt_tokens {
                    return Err(Error::MalformedOutput {
                        request_id: self.request_id.clone(),
                        message: "engine returned too many prompt logprob positions".to_string(),
                    });
                }
                if let Some(start) = self.start_event(&prompt_token_ids)? {
                    events.push(start);
                }
            }
            // Generated logprobs, when requested, defer decoding until the
            // matching TokenLogprobs event validates the token identity.
            Event::TextToken { id, .. } => {
                if let Some(start) = self.start_event(&prompt_token_ids)? {
                    events.push(start);
                }
                self.ensure_output_ready_from(&events, "a generated token")?;
                if self.generated_logprobs_requested {
                    self.pending_token = Some(id);
                } else if self.consume_tokens(
                    &prompt_token_ids,
                    &[id],
                    Vec::new(),
                    raw_stream,
                    &mut events,
                )? {
                    return Ok(Decoded::new(events));
                }
            }
            // Tokens committed together, such as a block-diffusion canvas,
            // carry no logprobs and decode as one update.
            Event::TextTokens { ids } => {
                if let Some(start) = self.start_event(&prompt_token_ids)? {
                    events.push(start);
                }
                self.ensure_output_ready_from(&events, "a block of committed tokens")?;
                if self.generated_logprobs_requested {
                    return Err(Error::MalformedOutput {
                        request_id: self.request_id.clone(),
                        message: "engine emitted committed tokens without their requested logprobs"
                            .to_string(),
                    });
                }
                if self.consume_tokens(
                    &prompt_token_ids,
                    &ids,
                    Vec::new(),
                    raw_stream,
                    &mut events,
                )? {
                    return Ok(Decoded::new(events));
                }
            }
            Event::TokenLogprobs { id, candidates } => {
                let pending = self
                    .pending_token
                    .take()
                    .ok_or_else(|| Error::MalformedOutput {
                        request_id: self.request_id.clone(),
                        message: "engine returned token logprobs without a pending token"
                            .to_string(),
                    })?;
                // The event protocol lists the sampled token as the first candidate.
                if pending != id || candidates.first().is_none_or(|entry| entry.token_id != id) {
                    return Err(Error::MalformedOutput {
                        request_id: self.request_id.clone(),
                        message: "token logprobs do not match the emitted token".to_string(),
                    });
                }
                if self.consume_tokens(
                    &prompt_token_ids,
                    &[id],
                    vec![PositionLogprobs {
                        entries: candidates,
                    }],
                    raw_stream,
                    &mut events,
                )? {
                    return Ok(Decoded::new(events));
                }
            }
            // A terminal event flushes decoder holdback only after every pending
            // token/logprob pair and all prompt metadata have resolved.
            Event::Finished {
                reason,
                stop_reason,
                completion_tokens,
                ..
            } => {
                if let Some(start) = self.start_event(&prompt_token_ids)? {
                    events.push(start);
                }
                self.ensure_output_ready_from(&events, "the terminal event")?;
                let Some(reason) = FinishReason::from_engine(reason, stop_reason) else {
                    events.push(DecodedTextEvent::Failed {
                        message: "engine execution failed".to_owned(),
                    });
                    return Ok(Decoded::new(events));
                };

                // Release the decoder's holdback untruncated: a stop-string
                // match would have ended the stream in `consume_tokens`.
                let (last_chunk, text) = self.decoder.flush(None)?;
                debug!(
                    ?reason,
                    output_tokens = self.output_tokens,
                    "request finished with decoded text"
                );
                if tracing::enabled!(Level::TRACE) {
                    trace!(full_text = text, "terminal decoded text");
                }

                events.push(DecodedTextEvent::TextDelta {
                    delta: last_chunk.unwrap_or_default(),
                    token_ids: Vec::new(),
                    logprobs: None,
                });
                events.push(DecodedTextEvent::Done(Finished {
                    reason,
                    prompt_tokens,
                    output_tokens: completion_tokens,
                }));
                return Ok(Decoded::new(events));
            }
            Event::Rejected { kind, message } => {
                events.push(DecodedTextEvent::Rejected { kind, message });
                return Ok(Decoded::new(events));
            }
            Event::Error { message } | Event::ArtifactUnavailable { message } => {
                events.push(DecodedTextEvent::Failed { message });
                return Ok(Decoded::new(events));
            }
            // A request's events migrate with it (`crate::migration`), which
            // carries a migratable end over to the next engine.
            Event::Migratable { .. } => {
                return Err(Error::MalformedOutput {
                    request_id: self.request_id.clone(),
                    message: "a migration's event reached the decoder".to_string(),
                });
            }
            // Media artifacts belong to diffusion requests, which other
            // consumers collect.
            Event::Artifact(_) | Event::MediaProgress { .. } => {
                return Err(Error::MalformedOutput {
                    request_id: self.request_id.clone(),
                    message: "a generation request received a media event".to_string(),
                });
            }
        }
        Ok(Decoded::new(events))
    }

    /// Fails unless `Start` is emitted (by `start_event` or earlier) and no
    /// generated token awaits its logprobs.
    fn ensure_output_ready_from(
        &self,
        events: &[DecodedTextEvent],
        event: &str,
    ) -> Result<(), Error> {
        let starting = events
            .iter()
            .any(|event| matches!(event, DecodedTextEvent::Start { .. }));
        if (self.started || starting) && self.pending_token.is_none() {
            Ok(())
        } else {
            Err(Error::MalformedOutput {
                request_id: self.request_id.clone(),
                message: format!("{event} arrived before output metadata was complete"),
            })
        }
    }

    /// Decodes committed tokens in order and applies stop-string holdback.
    ///
    /// Returns `true` when a token completed a stop string; the decoder has
    /// then been flushed and truncated at the match and the final `TextDelta`
    /// and `Done` are pushed, so the caller must end the stream.
    fn consume_tokens(
        &mut self,
        prompt_token_ids: &[u32],
        token_ids: &[u32],
        positions: Vec<PositionLogprobs>,
        raw_stream: &mut dyn Events,
        events: &mut Vec<DecodedTextEvent>,
    ) -> Result<bool, Error> {
        let decoded_logprobs = (!positions.is_empty())
            .then(|| {
                decode_logprobs(
                    self.tokenizer.as_ref(),
                    &positions,
                    self.options.skip_special_tokens,
                )
            })
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
            raw_stream.stop_at_consumed_prefix();
            let truncate_to = Some(if self.options.include_stop_str_in_output {
                offset + stop_string.len()
            } else {
                offset
            });
            // `last_chunk` is the not-yet-emitted remainder of the truncated
            // text.
            let (last_chunk, _) = self.decoder.flush(truncate_to)?;
            events.push(DecodedTextEvent::TextDelta {
                delta: delta + last_chunk.unwrap_or_default().as_str(),
                token_ids: token_ids[..=index].to_vec(),
                logprobs: decoded_logprobs,
            });
            events.push(DecodedTextEvent::Done(Finished {
                reason: FinishReason::Stop(Some(StopReason::String(stop_string))),
                prompt_tokens: prompt_token_ids.len(),
                output_tokens: self.output_tokens,
            }));
            return Ok(true);
        }
        raw_stream.acknowledge_consumed_prefix();
        events.push(DecodedTextEvent::TextDelta {
            delta,
            token_ids: token_ids.to_vec(),
            logprobs: decoded_logprobs,
        });
        Ok(false)
    }

    /// The decoder's state, for a successor process (design §7.5.1).
    pub fn state(&self) -> proto::handoff::Decode {
        use proto::handoff;
        let decoder = self.decoder.state();
        handoff::Decode {
            decoder: Some(handoff::IncrementalDecode {
                ids: decoder.ids,
                prefix: decoder.prefix,
                prefix_index: decoder.prefix_index as u64,
                cumulative_output: decoder.cumulative_output,
                output_index: decoder.output_index as u64,
                prompt_seeded: decoder.prompt_seeded,
                pending_since_emit: decoder.pending_since_emit as u64,
                skip_special_tokens: decoder.skip_special_tokens,
                min_bytes_to_buffer: decoder.min_bytes_to_buffer as u64,
            }),
            stop_strings: self.options.stop_strings.clone(),
            prompt_positions: self
                .prompt_positions
                .iter()
                .map(|position| handoff::PositionLogprobs {
                    entries: position
                        .entries
                        .iter()
                        .map(|entry| handoff::TokenLogprob {
                            token_id: entry.token_id,
                            logprob: entry.logprob,
                            rank: entry.rank,
                        })
                        .collect(),
                })
                .collect(),
            queued_at: self.scheduled.map(|(queued_at, _)| queued_at),
            scheduled_at: self.scheduled.map(|(_, scheduled_at)| scheduled_at),
            started: self.started,
            pending_token: self.pending_token,
            output_tokens: self.output_tokens as u64,
        }
    }

    /// Continues the stream `state` describes, on `tokenizer`.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::EmptyPromptTokenIds`] when the state names no prompt.
    pub fn resume(
        request_id: String,
        tokenizer: DynTokenizer,
        prompt_token_ids: Vec<u32>,
        prompt_logprobs_requested: bool,
        generated_logprobs_requested: bool,
        options: TextDecode,
        state: &proto::handoff::Decode,
    ) -> Result<Self, Error> {
        if prompt_token_ids.is_empty() {
            return Err(Error::EmptyPromptTokenIds { request_id });
        }
        let decoder = match &state.decoder {
            Some(decoder) => IncrementalDecoder::resume(
                crate::tokenizer::IncrementalDecodeState {
                    ids: decoder.ids.clone(),
                    prefix: decoder.prefix.clone(),
                    prefix_index: decoder.prefix_index as usize,
                    cumulative_output: decoder.cumulative_output.clone(),
                    output_index: decoder.output_index as usize,
                    prompt_seeded: decoder.prompt_seeded,
                    pending_since_emit: decoder.pending_since_emit as usize,
                    skip_special_tokens: decoder.skip_special_tokens,
                    min_bytes_to_buffer: decoder.min_bytes_to_buffer as usize,
                },
                Arc::clone(&tokenizer),
            ),
            None => tokenizer.create_decode_stream(
                &prompt_token_ids,
                options.skip_special_tokens,
                stop_string_holdback_bytes(&options),
            ),
        };
        let mut options = options;
        options.stop_strings.clone_from(&state.stop_strings);
        Ok(Self {
            request_id,
            tokenizer,
            prompt_token_ids,
            prompt_logprobs_requested,
            generated_logprobs_requested,
            decoder,
            options,
            prompt_positions: state
                .prompt_positions
                .iter()
                .map(|position| PositionLogprobs {
                    entries: position
                        .entries
                        .iter()
                        .map(|entry| proto::request::TokenLogprob {
                            token_id: entry.token_id,
                            logprob: entry.logprob,
                            rank: entry.rank,
                        })
                        .collect(),
                })
                .collect(),
            scheduled: match (state.queued_at, state.scheduled_at) {
                (Some(queued_at), Some(scheduled_at)) => Some((queued_at, scheduled_at)),
                _ => None,
            },
            started: state.started,
            pending_token: state.pending_token,
            output_tokens: state.output_tokens as usize,
        })
    }
}

/// Advances incremental decoding and returns only bytes newly made visible by this token.
///
/// Stop strings are checked only from token number `min_tokens + 1` onward
/// (`output_tokens` counts the tokens before this one). A matched stop
/// string is removed from `options.stop_strings`.
fn decode_one_token(
    decoder: &mut IncrementalDecoder,
    token_id: u32,
    output_tokens: usize,
    options: &mut TextDecode,
) -> Result<ArDecode, Error> {
    let new_bytes = decoder.push_token(token_id)?;
    let stop = if output_tokens + 1 > options.min_tokens as usize {
        matches_stop_string(&options.stop_strings, decoder.output(), new_bytes)
            .map(|(index, offset)| (options.stop_strings.swap_remove(index), offset))
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

/// Returns the suffix byte count retained to detect cross-chunk stop strings.
///
/// Holding back one byte less than the longest stop string keeps any partial
/// match out of emitted text until the next token confirms or rules it out.
/// No holdback is needed when matched stop strings stay in the output. The
/// result is passed to the decoder as its `min_bytes_to_buffer`.
fn stop_string_holdback_bytes(options: &TextDecode) -> usize {
    if options.include_stop_str_in_output {
        return 0;
    }
    options
        .stop_strings
        .iter()
        .map(String::len)
        .max()
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
