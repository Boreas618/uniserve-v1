//! Client-visible event export and per-request output accounting.
//!
//! Events accumulate in request-local journals when the bounded consumer channel
//! is full, preserving order without blocking the engine owner thread.
//!
//! This module also owns the semantic resolution of completed calls
//! (`Scheduler::resolve`): it turns accepted worker results into public events
//! and the request's next generation phase, and it owns the termination of
//! running token requests (`Scheduler::finish_with`), which releases scheduler
//! resources and publishes the terminal `Finished` event.
//! `RequestState::process_generation_result` has already applied a call's
//! accepted progress (cursors, encoder indices, denoising steps, phase) by the
//! time `resolve` runs.

use super::*;
use crate::completion::CallCompletion;
use ipc::{ForwardMode, MediaCall};

use crate::kv_cache::manager::AttentionManager;
use crate::kv_cache::{CachedTokens, KvOffloadManager, OffloadKind, PageRef};

/// How one understanding token ends or continues its request
/// (`Scheduler::und_token_outcome`).
enum TokenOutcome {
    /// The token is ordinary output.
    Continue,
    /// The token is one of the request's stop tokens, published only with
    /// `include_stop_token`.
    StopToken,
    /// The token ends the request by EOS or the token limit; it is published
    /// unless it is an EOS token.
    Finish(FinishReason),
}

/// What a token request was charged, which its `Finished` event reports.
fn usage(state: &RequestState) -> Usage {
    Usage {
        prompt_tokens: state.charged_prompt_tokens(),
        completion_tokens: state.num_generated_tokens,
    }
}

/// Takes the log-probabilities `record` reports for its committed token at
/// `index`: the token's own value and its entries (the token's own, then its
/// top candidates), or nothing when the request did not ask for them.
fn take_token_logprobs(
    record: &mut CallCompletion,
    index: usize,
) -> (Option<f32>, Option<Vec<TokenLogprob>>) {
    let entries = record
        .token_logprobs
        .get_mut(index)
        .map(std::mem::take)
        .filter(|entries| !entries.is_empty());
    let logprob = entries
        .as_ref()
        .and_then(|entries| entries.first())
        .map(|entry| entry.logprob);
    (logprob, entries)
}

/// Flushes journaled public events into the output channel in order.
///
/// Returns `true` only when the receiver has closed, in which case the journal
/// is discarded. Returns `false` when the journal drained completely or when
/// the channel filled again, leaving the remainder journaled.
fn flush_public_journal(event_tx: &EventTx, journal: &mut VecDeque<Event>) -> bool {
    while let Some(event) = journal.pop_front() {
        match event_tx.send(event) {
            Ok(()) => {}
            Err(EventSendError::Full(event)) => {
                journal.push_front(*event);
                return false;
            }
            Err(EventSendError::Closed(_)) => {
                journal.clear();
                return true;
            }
        }
    }
    false
}

/// Event journal and usage counters for one request.
pub(super) struct RequestOutput {
    pub(super) events: EventJournal,
    /// `TextToken` events accepted into the ordered output.
    pub(super) tokens_sent: usize,
    /// Prompt score positions received from worker completions.
    pub(super) prompt_logprobs_processed: usize,
    /// Prompt score positions published as `PromptLogprobs` events.
    pub(super) prompt_logprobs_emitted: usize,
    /// Ends of the tokens each resolved token-producing call published,
    /// awaiting a stop-string decoder decision, as `tokens_sent` values in
    /// export order. Populated only for requests with stop strings;
    /// `usize::MAX` marks the call being resolved. `can_schedule_next`
    /// refuses another call while the queue holds `max_unresolved_calls`
    /// entries; acknowledgements drain it, and a non-error finish waits until
    /// it is empty.
    pub(super) decoder_boundaries: VecDeque<usize>,
}

impl RequestOutput {
    /// Creates empty output accounting that publishes through `event_tx`.
    pub(super) fn new(event_tx: EventTx) -> Self {
        Self {
            events: EventJournal::new(event_tx),
            tokens_sent: 0,
            prompt_logprobs_processed: 0,
            prompt_logprobs_emitted: 0,
            decoder_boundaries: VecDeque::new(),
        }
    }
}

/// Ordered export shared by token and media requests. Progress can be
/// coalesced while terminal events retain their place until the caller reads.
///
/// The journal holds at most `OUTPUT_JOURNAL_CAPACITY` events beyond what the
/// bounded channel holds, and `enqueue` panics if that bound is exceeded. For
/// token requests the scheduler stays within it by scheduling a call only when
/// `Scheduler::output_window_ready` finds room for the worst-case output of
/// every in-flight call and the next one, plus a terminal reserve.
pub(super) struct EventJournal {
    pub(super) event_tx: EventTx,
    journal: VecDeque<Event>,
}

impl EventJournal {
    pub(super) fn new(event_tx: EventTx) -> Self {
        Self {
            event_tx,
            journal: VecDeque::new(),
        }
    }

    /// Returns whether the request's event receiver has closed.
    pub(super) fn is_closed(&self) -> bool {
        self.event_tx.is_closed()
    }

    /// Returns the number of events the journal can accept: free channel slots
    /// plus the journal's remaining headroom.
    pub(super) fn available_capacity(&self) -> usize {
        self.event_tx
            .capacity()
            .saturating_add(OUTPUT_JOURNAL_CAPACITY.saturating_sub(self.journal.len()))
    }

    /// Flushes pending journal entries; returns `true` when the receiver closed.
    fn flush(&mut self) -> bool {
        flush_public_journal(&self.event_tx, &mut self.journal)
    }

    /// Delivers an event or journals it in order when the public channel is full.
    ///
    /// Returns `true` when the event was delivered, journaled, or coalesced, and
    /// `false` when the receiver has closed and the event was dropped.
    pub(super) fn enqueue(&mut self, event: Event) -> bool {
        if self.flush() {
            return false;
        }

        // A journaled progress event that has not been delivered is superseded
        // by the newer one; coalescing only replaces the journal tail, so no
        // other event loses its place.
        if matches!(event, Event::MediaProgress { .. })
            && let Some(last @ Event::MediaProgress { .. }) = self.journal.back_mut()
        {
            *last = event;
            return true;
        }

        // Send directly only when nothing is journaled; otherwise the event
        // would overtake earlier ones.
        if self.journal.is_empty() {
            match self.event_tx.send(event) {
                Ok(()) => {
                    return true;
                }
                Err(EventSendError::Closed(_)) => return false,
                Err(EventSendError::Full(event)) => self.journal.push_back(*event),
            }
        } else {
            self.journal.push_back(event);
        }
        assert!(
            self.journal.len() <= OUTPUT_JOURNAL_CAPACITY,
            "scheduler exceeded the bounded public output journal"
        );
        true
    }
}

#[derive(Default)]
/// Non-blocking event publisher with request-local ordered buffering.
///
/// Holds the journals of finished token and media requests whose remaining
/// events, including the terminal one, have not yet fit into their channels,
/// so request state can retire while its output drains under backpressure.
pub(crate) struct OutputSender {
    retired: HashMap<RequestId, EventJournal>,
}

impl OutputSender {
    /// Returns the number of retired requests still awaiting output delivery.
    ///
    /// Admission counts these toward `max_queued_requests`, so undelivered
    /// terminal output applies backpressure to new submissions.
    pub(super) fn retained_len(&self) -> usize {
        self.retired.len()
    }

    /// Retains pending events after computational state has retired.
    ///
    /// A journal with nothing pending is dropped immediately.
    pub(super) fn retire(&mut self, id: RequestId, output: EventJournal) {
        if !output.journal.is_empty() {
            self.retired.insert(id, output);
        }
    }

    /// Flushes retired journals, dropping each once it drains or its receiver
    /// closes. Returns whether any journal changed length.
    pub(super) fn flush_retired(&mut self) -> bool {
        let mut progressed = false;
        self.retired.retain(|_, output| {
            let before = output.journal.len();
            let closed = flush_public_journal(&output.event_tx, &mut output.journal);
            progressed |= output.journal.len() != before;
            !closed && !output.event_tx.is_closed() && !output.journal.is_empty()
        });
        progressed
    }

    /// Hands every retained journal to its receiver as the control loop stops.
    ///
    /// A stopped scheduler can no longer flush journals into their full
    /// channels, so each journal's pending events, including the terminal
    /// event, move to the receiver, which yields them after the channel's
    /// buffered events (`EventTx::close_with`).
    pub(super) fn hand_off_retired(&mut self) {
        for (_, output) in self.retired.drain() {
            output.event_tx.close_with(output.journal);
        }
    }
}

impl Scheduler {
    /// Publishes committed autoregressive tokens and advances text-generation state.
    ///
    /// Tokens are applied in order and resolution stops at the first token that
    /// finishes the request. The sampled log probability and top candidates
    /// belong to the last committed token only.
    pub(super) fn resolve_decode_text(&mut self, id: RequestId, mut record: CallCompletion) {
        self.activate_request_tables(id);

        // A completion without committed tokens resolves as the first EOS token.
        let tokens = if record.committed_tokens.is_empty() {
            vec![self.special_tokens.eos[0]]
        } else {
            std::mem::take(&mut record.committed_tokens)
        };
        for (idx, tok) in tokens.iter().copied().enumerate() {
            // Each committed token carries its own log-probabilities when
            // the request asks for them, a verification's accepted drafts
            // as well as the token after them.
            let (logprob, top_logprobs) = take_token_logprobs(&mut record, idx);
            let Some(st) = self.running.get_mut(&id) else {
                return;
            };
            st.num_generated_tokens += 1;
            if self.emit_or_finish_und_token(id, tok, logprob, top_logprobs, true) {
                return;
            }
            if let Some(st) = self.running.get_mut(&id) {
                st.next_token = tok;
                st.phase = Phase::DecodeUnd;
            }
        }
    }

    /// Applies one completed call to its request and emits observable output.
    ///
    /// `RequestState::process_generation_result` has already applied the call's
    /// accepted progress, so the cursors, encoder indices, denoising step count,
    /// and phase read here reflect this completion. Resolution adds the
    /// remaining effects for the call's kind, such as stop conditions,
    /// prefix- and encoder-cache export, and public events.
    ///
    /// The caller resolves only a running request, and every path that
    /// finishes it returns; a request that is no longer running has nothing
    /// left to update, so resolution stops there.
    pub(super) fn resolve(&mut self, id: RequestId, call: Call, mut record: CallCompletion) {
        let call_variant = call.kind;

        // Prompt scores share the completion but precede its phase transition.
        if !record.prompt_logprobs.is_empty() {
            let positions = std::mem::take(&mut record.prompt_logprobs);
            self.resolve_prompt_logprobs(id, positions);
        }

        // A verification commits its accepted drafts and the token after
        // them, which resolve as a decode's tokens do.
        if matches!(
            call_variant,
            CallKind::Forward(ForwardMode::Decode | ForwardMode::Verify)
        ) {
            return self.resolve_decode_text(id, record);
        }

        match call_variant {
            CallKind::Forward(ForwardMode::Prefill) => {
                self.activate_request_tables(id);

                // Latent input-image block ingest. The encoder index wraps to
                // zero once the image's last block has been written.
                if consumes_image_features(&call) {
                    self.free_consumed_products(id, &call);
                    let is_final_step = self
                        .running
                        .get(&id)
                        .is_some_and(|state| state.image_encoder_index == 0);

                    // Writing the last image of a fully consumed prompt
                    // starts a canvas-generating request's first block, or
                    // understanding decode from BOS.
                    if is_final_step
                        && self.running.get(&id).is_some_and(|st| {
                            st.num_ingested_images >= st.req.multimodal_inputs.images.len()
                                && st.num_computed_prompt_tokens
                                    >= st.req.prompt_token_ids.len() as u32
                        })
                    {
                        let bos = self.special_tokens.bos;
                        let generates_canvases = self.config.canvas.is_some();
                        if let Some(st) = self.running.get_mut(&id) {
                            if generates_canvases {
                                st.phase = Phase::Canvas;
                            } else {
                                st.next_token = bos;
                                st.phase = Phase::DecodeUnd;
                            }
                        }
                    }
                    return;
                }

                // Prompt extension. Partial prefill remains in ingest until every
                // text and multimodal position has been consumed. The vision
                // blocks it wrote no longer need their request-local products.
                self.free_consumed_products(id, &call);
                let Some(st) = self.running.get(&id) else {
                    return;
                };
                let (cursor, prompt_len) = (
                    st.num_computed_prompt_tokens as usize,
                    st.req.prompt_token_ids.len(),
                );
                if cursor < prompt_len {
                    return;
                }

                if self.running.get(&id).is_some_and(|st| {
                    st.num_ingested_images < st.req.multimodal_inputs.images.len()
                        || st.num_computed_prompt_tokens < st.req.prompt_token_ids.len() as u32
                }) {
                    return;
                }

                // A canvas-generating request's complete prompt samples
                // nothing, and nor does each of its block commits, which
                // extend the context the same way; the next block follows.
                if self.config.canvas.is_some()
                    && let Some(st) = self.running.get_mut(&id)
                {
                    st.phase = Phase::Canvas;
                    return;
                }

                // A context that ends with an image block sampled nothing:
                // understanding decode starts from BOS, as after a trailing
                // latent block.
                if call.token_output.is_none() {
                    let bos = self.special_tokens.bos;
                    if let Some(st) = self.running.get_mut(&id) {
                        st.next_token = bos;
                        st.phase = Phase::DecodeUnd;
                    }
                    return;
                }

                let Some(st) = self.running.get_mut(&id) else {
                    return;
                };
                st.num_generated_tokens += 1;

                // Prompt scoring covers every prompt position after the first;
                // a complete prompt with unscored positions is an error.
                if self.running.get(&id).is_some_and(|state| {
                    state.req.sampling.prompt_logprobs_requested()
                        && state.output.prompt_logprobs_emitted
                            != state.req.prompt_token_ids.len().saturating_sub(1)
                }) {
                    tracing::error!(
                        request_id = id.0,
                        "prompt logprob scoring ended before every prompt position was resolved"
                    );
                    return self.finish(id, FinishReason::Error);
                }

                let tok = record
                    .committed_tokens
                    .last()
                    .copied()
                    .unwrap_or(self.special_tokens.eos[0]);
                let last = record.committed_tokens.len().saturating_sub(1);
                let (logprob, top_logprobs) = take_token_logprobs(&mut record, last);

                if self.emit_or_finish_und_token(id, tok, logprob, top_logprobs, true) {
                    return;
                }

                // A request prefilled on a prefill group hands its cache to
                // a decode group first; the decode there continues from the
                // committed token, which leaves no device token to relay.
                let hand_off = self.decodes_elsewhere(id);
                if let Some(st) = self.running.get_mut(&id) {
                    st.next_token = tok;
                    if hand_off {
                        st.phase = Phase::HandOff;
                        st.latest_token = None;
                    } else {
                        st.phase = Phase::DecodeUnd;
                    }
                }
            }
            // A canvas step that stopped its block left the block's tokens
            // for their commit; they are published first, in order, and the
            // request finishes on the first that ends it.
            CallKind::Forward(ForwardMode::TokenDenoising) => {
                let tokens = match self.running.get(&id) {
                    Some(st) if st.phase == Phase::CommitCanvas => st.canvas_commit.clone(),
                    _ => return,
                };
                self.emit_or_finish_block(id, &tokens);
            }
            CallKind::Media(
                MediaCall::MediaReading
                | MediaCall::VideoDecoding
                | MediaCall::AudioDecoding
                | MediaCall::AudioEncoding
                | MediaCall::Muxing,
            ) => {}
            // Input-image encoding of the next block the context needs
            // (`next_context_encode`): the product feeds the call that
            // writes the block and, when the request writes the encoder
            // cache, becomes a shared cache entry.
            CallKind::Media(MediaCall::VisionEncoding)
            | CallKind::Media(MediaCall::LatentEncoding) => {
                let Some((block, step, block_cache_key)) =
                    self.next_context_encode(id).and_then(|(block, step)| {
                        let state = self.running.get(&id)?;
                        let encoding = self.encoding(&state.req, block)?;
                        let key = state.req.cache.write.then(|| encoding.device_key());
                        Some((block, step, key))
                    })
                else {
                    return self.finish(id, FinishReason::Error);
                };
                let encoder_cache_key = block_cache_key;
                let Some(feature) = call.encoder_output.clone() else {
                    return self.finish(id, FinishReason::Error);
                };

                let mut free_products = Vec::new();

                // Cache ownership transfers the backing allocation out of the
                // request; uncached products remain request-local transients.
                let selected_product = if let Some(cache_key) = encoder_cache_key {
                    // Insertion may return an evicted entry or, when a
                    // concurrent miss already stored this key, this
                    // call's redundant product; either is freed once
                    // the request's product is selected.
                    if let Some(freed) = self
                        .storage
                        .encoder_cache
                        .insert(cache_key, feature.clone())
                    {
                        free_products.push(freed);
                    }
                    // The entry is live now, whether created or
                    // re-activated; the policy's LRU is told.
                    self.record_encoded(cache_key);

                    let stored =
                        self.storage.encoder_cache.peek_product(cache_key) == Some(feature.clone());

                    // Only a product the cache now holds moves its
                    // buffer from the request to encoder retention.
                    if stored {
                        let allocation = self.running.get_mut(&id).and_then(|state| {
                            state.allocations_mut()?.take_buffer(feature.buffer_id())
                        });
                        let Some(allocation) = allocation else {
                            return self.finish(id, FinishReason::Error);
                        };
                        if let Err(allocation) = self
                            .storage
                            .retain_encoder_buffer(feature.buffer_id(), allocation)
                        {
                            self.storage
                                .pending_buffer_frees
                                .insert(feature.buffer_id(), allocation);
                            self.inflight
                                .pending_commands
                                .push_back(BatchCommand::Free {
                                    buffer: feature.buffer_id(),
                                });
                            return self.finish(id, FinishReason::Error);
                        }
                    }

                    // The request pins the resident entry, which may be an
                    // earlier equivalent product rather than this call's.
                    let Some(product) = self.storage.encoder_cache.acquire(cache_key) else {
                        return self.finish(id, FinishReason::Error);
                    };
                    if let Some(st) = self.running.get_mut(&id) {
                        st.encoder_cache_pins.push(EncoderCachePin {
                            key: cache_key,
                            product: product.clone(),
                        });
                    }
                    // Pinning it is a use: the entry moves in the order
                    // the policy evicts by.
                    self.record_encoded(cache_key);
                    product
                } else if let Some(st) = self.running.get_mut(&id) {
                    st.transient_encoder_products.push(feature.clone());
                    feature.clone()
                } else {
                    feature.clone()
                };

                // A latent block is written by its own extension; a
                // vision block joins the context prefill that reaches
                // it. Encodes follow the blocks' context order.
                if let Some(st) = self.running.get_mut(&id) {
                    if step == crate::ImageEncoder::Latent {
                        st.input_image_features = Some(selected_product);
                        st.phase = Phase::IngestState;
                    } else {
                        debug_assert!(block.image >= st.num_ingested_images);
                        st.context_features.push(selected_product);
                    }
                }

                if !free_products.is_empty() {
                    self.free_buffers(free_products.into_iter().map(|product| product.buffer_id()));
                }
            }
            // These calls publish nothing and change no state here. `Decode`
            // returned above and appears only for exhaustiveness.
            CallKind::Media(MediaCall::TextEncoding)
            | CallKind::Forward(ForwardMode::Decode)
            | CallKind::Forward(ForwardMode::Verify)
            | CallKind::Media(MediaCall::LatentPreparation)
            | CallKind::Media(MediaCall::Denoising)
            | CallKind::Media(MediaCall::VideoEncoding) => {}
        }
    }

    /// Publishes one completion's prompt scores as a `PromptLogprobs` event.
    ///
    /// Every received position counts toward `prompt_logprobs_processed`.
    /// Positions beyond the prompt's scorable range (every prompt position
    /// after the first) are dropped, and nothing is published when no position
    /// remains.
    pub(super) fn resolve_prompt_logprobs(
        &mut self,
        id: RequestId,
        positions: Vec<Vec<TokenLogprob>>,
    ) {
        let total = positions.len();
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        let processed_before = state.output.prompt_logprobs_processed;
        let emitted_before = state.output.prompt_logprobs_emitted;
        let expected_total = state.req.prompt_token_ids.len().saturating_sub(1);
        state.output.prompt_logprobs_processed = processed_before.saturating_add(total);
        let skip = emitted_before.saturating_sub(processed_before);
        let remaining = expected_total.saturating_sub(emitted_before);
        let selected: Vec<_> = positions.into_iter().skip(skip).take(remaining).collect();
        state.output.prompt_logprobs_emitted = emitted_before.saturating_add(selected.len());
        if selected.is_empty() {
            return;
        }
        self.emit(
            id,
            Event::PromptLogprobs {
                positions: selected
                    .into_iter()
                    .map(|entries| PositionLogprobs { entries })
                    .collect(),
            },
        );
    }

    /// Frees the request-local encoder products an image-writing `call`
    /// consumed: its latent feature or its vision blocks' features. Products
    /// the encoder cache holds stay pinned until the request finishes, and
    /// products of blocks the context has not reached yet stay held.
    pub(super) fn free_consumed_products(&mut self, id: RequestId, call: &Call) {
        let consumed: Vec<TensorRef> = call
            .latent_feature_input
            .iter()
            .chain(call.vision_inputs.iter().map(|input| &input.feature))
            .cloned()
            .collect();
        let freed = self
            .running
            .get_mut(&id)
            .map(|state| {
                let (freed, kept) = std::mem::take(&mut state.transient_encoder_products)
                    .into_iter()
                    .partition(|product| consumed.contains(product));
                state.transient_encoder_products = kept;
                freed
            })
            .unwrap_or_default();
        self.free_buffers(freed.into_iter().map(|product| product.buffer_id()));
    }

    /// Flushes the journals of running token and media requests and of retired
    /// requests. Returns whether any journal changed length.
    ///
    /// A running request whose receiver has closed is marked for cancellation
    /// through its terminal intent; it retires once its in-flight calls drain.
    pub(super) fn flush_output_journals(&mut self) -> bool {
        let mut progressed = false;
        for (events, intent) in self
            .running
            .values_mut()
            .map(|state| (&mut state.output.events, &mut state.terminal_intent))
            .chain(
                self.running_media
                    .values_mut()
                    .map(|state| (&mut state.output, &mut state.terminal_intent)),
            )
        {
            let before = events.journal.len();
            if events.is_closed() || events.flush() {
                *intent = Some(Terminal::finish(FinishReason::Cancelled));
            }
            progressed |= events.journal.len() != before;
        }
        progressed |= self.output.flush_retired();
        progressed
    }

    /// Returns whether the request accepted an event into its ordered output.
    ///
    /// Returns `false` for a request that is not running. A closed receiver
    /// also returns `false` and marks the request for cancellation.
    pub(super) fn emit(&mut self, id: RequestId, event: Event) -> bool {
        // A request whose terminal event is published publishes nothing more.
        if self
            .inflight
            .pending_finishes
            .get(&id)
            .is_some_and(|finish| finish.terminal_published)
        {
            return false;
        }
        let Some(state) = self.running.get_mut(&id) else {
            return false;
        };
        let accepted = state.output.events.enqueue(event);
        if !accepted {
            state.terminal_intent = Some(Terminal::finish(FinishReason::Cancelled));
        }
        accepted
    }

    /// Records an accepted text token and publishes it when the request emits text.
    ///
    /// A running request always records the token in `generated_token_ids`.
    /// Returns `true` when the token is visible text, even if a closed receiver
    /// dropped it, and `false` when the request is not running or does not emit
    /// text. `tokens_sent` advances only when the event was accepted.
    pub(super) fn emit_text(&mut self, id: RequestId, tok: u32, logprob: Option<f32>) -> bool {
        let Some(st) = self.running.get_mut(&id) else {
            return false;
        };
        st.generated_token_ids.push(tok);
        let published = self.emit(id, Event::TextToken { id: tok, logprob });
        if published {
            if let Some(state) = self.running.get_mut(&id) {
                state.output.tokens_sent = state.output.tokens_sent.saturating_add(1);
            }
            // The policy is told the request emitted a token, and when.
            let at =
                scheduling::guest::Nanos(crate::clock::now_monotonic_us().saturating_mul(1_000));
            self.recorder
                .record(views::emitted(scheduling::guest::RequestId(id.0), 1, at));
        }
        true
    }

    /// Records accepted text tokens and publishes them as one event when the
    /// request emits text.
    ///
    /// A running request always records the tokens in `generated_token_ids`.
    /// `tokens_sent` advances only when the event was accepted.
    pub(super) fn emit_text_tokens(&mut self, id: RequestId, tokens: Vec<u32>) {
        let Some(st) = self.running.get_mut(&id) else {
            return;
        };
        st.generated_token_ids.extend_from_slice(&tokens);
        if tokens.is_empty() {
            return;
        }
        let count = tokens.len();
        let published = self.emit(id, Event::TextTokens { ids: tokens });
        if published {
            if let Some(state) = self.running.get_mut(&id) {
                state.output.tokens_sent = state.output.tokens_sent.saturating_add(count);
            }
            let at =
                scheduling::guest::Nanos(crate::clock::now_monotonic_us().saturating_mul(1_000));
            self.recorder.record(views::emitted(
                scheduling::guest::RequestId(id.0),
                u32::try_from(count).unwrap_or(u32::MAX),
                at,
            ));
        }
    }

    /// Resolves the stop conditions of a stopped canvas block's tokens in
    /// order and publishes the visible ones as one event.
    ///
    /// Every token counts toward `num_generated_tokens` and is resolved as
    /// `emit_or_finish_und_token` resolves a sampled token: the first token
    /// that ends the request (a stop token, EOS, or the token limit) closes
    /// the block, which is published up to it, and the request finishes after
    /// its in-flight work. Tokens after it are never generated output.
    pub(super) fn emit_or_finish_block(&mut self, id: RequestId, tokens: &[u32]) {
        let mut visible = Vec::with_capacity(tokens.len());
        let mut finish = None;
        for &token_id in tokens {
            let Some(state) = self.running.get_mut(&id) else {
                return;
            };
            state.num_generated_tokens += 1;
            match self.und_token_outcome(id, token_id) {
                TokenOutcome::Continue => visible.push(token_id),
                TokenOutcome::StopToken => {
                    if self
                        .running
                        .get(&id)
                        .is_some_and(|state| state.req.include_stop_token)
                    {
                        visible.push(token_id);
                    }
                    finish = Some((FinishReason::Stop, Some(crate::StopReason::Token(token_id))));
                    break;
                }
                TokenOutcome::Finish(reason) => {
                    if !self.special_tokens.eos.contains(&token_id) {
                        visible.push(token_id);
                    }
                    finish = Some((reason, None));
                    break;
                }
            }
        }
        self.emit_text_tokens(id, visible);
        if let Some((reason, stop_reason)) = finish {
            self.finish_after_inflight(id, reason, stop_reason);
        }
    }

    /// Decides how one understanding token, already counted in
    /// `num_generated_tokens`, ends or continues its request.
    ///
    /// Stop tokens and EOS are honored only once at least `min_tokens` tokens
    /// precede the token; the token limit applies regardless of that floor,
    /// and a stop-token match takes precedence over EOS and the limit. A
    /// request that is not running continues, and its caller finds it gone.
    fn und_token_outcome(&self, id: RequestId, token_id: u32) -> TokenOutcome {
        let Some(state) = self.running.get(&id) else {
            return TokenOutcome::Continue;
        };
        let generated = state.num_generated_tokens;
        let under_floor = generated <= state.req.sampling.min_tokens;
        if !under_floor && state.req.stop_token_ids.contains(&token_id) {
            TokenOutcome::StopToken
        } else if generated >= state.req.max_und_tokens {
            TokenOutcome::Finish(FinishReason::MaxTokens)
        } else if self.special_tokens.eos.contains(&token_id)
            && !state.req.sampling.ignore_eos
            && !under_floor
        {
            TokenOutcome::Finish(FinishReason::Eos)
        } else {
            TokenOutcome::Continue
        }
    }

    /// Emits one visible token and its requested candidate log probabilities.
    pub(super) fn emit_sampled_text(
        &mut self,
        id: RequestId,
        tok: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<TokenLogprob>>,
    ) {
        if self.emit_text(id, tok, logprob)
            && let Some(top_logprobs) = top_logprobs.filter(|entries| !entries.is_empty())
        {
            self.emit(
                id,
                Event::TokenLogprobs {
                    id: tok,
                    candidates: top_logprobs,
                },
            );
        }
    }

    /// Publishes a matched stop token only when the request includes stop tokens
    /// in its output.
    pub(super) fn emit_terminal_stop_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<TokenLogprob>>,
    ) {
        if self
            .running
            .get(&id)
            .is_some_and(|state| state.req.include_stop_token)
        {
            self.emit_sampled_text(id, token_id, logprob, top_logprobs);
        }
    }

    /// Resolves stop conditions for one understanding token and emits it when appropriate.
    ///
    /// The caller has already counted the token in `num_generated_tokens`.
    /// A stop-token match takes precedence and publishes the token only with
    /// `include_stop_token`. A request that finishes on EOS or the token limit
    /// publishes the token unless it is an EOS token; an EOS token that does
    /// not finish the request (under the floor or with `ignore_eos`) is
    /// published as text. Returns `true` when the caller must stop resolving:
    /// the request finished, has a deferred finish, or is no longer running.
    pub(super) fn emit_or_finish_und_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<TokenLogprob>>,
        sampled: bool,
    ) -> bool {
        if !self.running.contains_key(&id) {
            return true;
        }
        match self.und_token_outcome(id, token_id) {
            TokenOutcome::StopToken => {
                self.emit_terminal_stop_token(id, token_id, logprob, top_logprobs);
                self.finish_after_inflight(
                    id,
                    FinishReason::Stop,
                    Some(crate::StopReason::Token(token_id)),
                );
                return true;
            }
            TokenOutcome::Finish(reason) => {
                if !self.special_tokens.eos.contains(&token_id) {
                    if sampled {
                        self.emit_sampled_text(id, token_id, logprob, top_logprobs);
                    } else {
                        self.emit_text(id, token_id, logprob);
                    }
                }
                self.finish_after_inflight(id, reason, None);
                return true;
            }
            TokenOutcome::Continue => {}
        }
        if sampled {
            self.emit_sampled_text(id, token_id, logprob, top_logprobs);
        } else {
            self.emit_text(id, token_id, logprob);
        }
        !self.running.contains_key(&id)
    }

    /// Finishes one request without a concrete stop-token or stop-string cause.
    pub(super) fn finish(&mut self, id: RequestId, reason: FinishReason) {
        self.finish_with(id, reason, None);
    }

    /// Defers terminal cleanup until in-flight work and decoder decisions are drained.
    ///
    /// Finishes immediately when nothing is outstanding; an error finish does
    /// not wait for stop-string decoder decisions. Otherwise records a
    /// `PendingFinish` that `finish_pending_if_idle` applies later. The first
    /// recorded reason is kept, except that an error replaces it.
    ///
    /// A non-error finish that waits only for calls in flight publishes its
    /// terminal event at once. Those calls were queued speculatively behind
    /// the call whose output finished the request, such as the no-op step
    /// behind the step that stopped a block, and a request with a pending
    /// finish resolves none of their results, so its output is already
    /// final. Only retirement, which releases the request's resources and
    /// sends the worker's `Finish`, waits for them to drain. A finish that
    /// waits for a stop-string decision is published at retirement, since
    /// that decision may still end the request earlier.
    pub(super) fn finish_after_inflight(
        &mut self,
        id: RequestId,
        reason: FinishReason,
        stop_reason: Option<crate::StopReason>,
    ) {
        let decoder_pending = !matches!(reason, FinishReason::Error)
            && self
                .running
                .get(&id)
                .is_some_and(|state| !state.output.decoder_boundaries.is_empty());
        if !self.inflight.has_pending_calls(id) && !decoder_pending {
            self.finish_with(id, reason, stop_reason);
            return;
        }
        let recorded = self.inflight.pending_finishes.get(&id);
        if recorded.is_some() && !matches!(reason, FinishReason::Error) {
            return;
        }
        let mut terminal_published = recorded.is_some_and(|finish| finish.terminal_published);
        if !terminal_published && !decoder_pending && !matches!(reason, FinishReason::Error) {
            // A closed receiver drops the event; nothing is published later
            // either way.
            if let Some(state) = self.running.get_mut(&id) {
                let terminal = Terminal::Finished {
                    reason: reason.clone(),
                    stop_reason: stop_reason.clone(),
                };
                for event in terminal.events(usage(state)) {
                    state.output.events.enqueue(event);
                }
                terminal_published = true;
            }
        }
        self.inflight.pending_finishes.insert(
            id,
            PendingFinish {
                reason,
                stop_reason,
                terminal_published,
            },
        );
    }

    /// Fails a running token request whose worker result was rejected.
    ///
    /// The first cause recorded for the request becomes its terminal `Error`
    /// event, sent at once so the client's failure carries it, and appears
    /// in the log line of its error finish, which `finish_after_inflight`
    /// applies once the request's in-flight calls drain. A later cause for
    /// the same request changes neither.
    pub(super) fn fail_after_inflight(&mut self, id: RequestId, cause: String) {
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        if state.failure.is_none() {
            state.failure = Some(cause.clone());
            self.emit(id, Event::Error { message: cause });
        }
        self.finish_after_inflight(id, FinishReason::Error, None);
    }

    /// Applies a deferred finish once no in-flight work or decoder decision remains.
    pub(super) fn finish_pending_if_idle(&mut self, id: RequestId) {
        if self.inflight.has_pending_calls(id)
            || self
                .running
                .get(&id)
                .is_some_and(|state| !state.output.decoder_boundaries.is_empty())
        {
            return;
        }
        // `finish_with` consumes the pending finish, including whether its
        // terminal event is already published.
        if let Some(pending) = self.inflight.pending_finishes.get(&id).cloned() {
            self.finish_with(id, pending.reason, pending.stop_reason);
        }
    }

    /// Publishes terminal accounting and releases every resource owned by one request.
    ///
    /// Applies to token requests in `running`; media requests terminate
    /// through `finish_media`. For a request that is not running, this only
    /// clears a pending finish. Allocations of a request registered with its
    /// workers move to `retiring_requests` until the queued `Finish` command
    /// is acknowledged; an unregistered request's allocations are freed here.
    pub(super) fn finish_with(
        &mut self,
        id: RequestId,
        reason: FinishReason,
        stop_reason: Option<crate::StopReason>,
    ) {
        self.retire_token_request(
            id,
            Terminal::Finished {
                reason,
                stop_reason,
            },
        );
    }

    /// Ends a token request as migratable for `cause`, releasing what it
    /// owns as `finish_with` does; its terminal event is `Migratable`,
    /// naming no stored trajectory.
    pub(super) fn migrate_request(&mut self, id: RequestId, cause: MigrationCause) {
        self.retire_token_request(id, Terminal::migrate(cause));
    }

    /// Publishes `terminal` and releases every resource owned by one token
    /// request; see `finish_with`.
    fn retire_token_request(&mut self, id: RequestId, terminal: Terminal) {
        // A call of this request still queued behind a full worker queue can
        // never be served nor state its layout once the request's allocation
        // goes, so it leaves the queue with the request.
        self.retire_queued_calls(id);
        // The policy sees the request end, with how it ended.
        let end = terminal.end();
        let request = scheduling::guest::RequestId(id.0);
        self.recorder.record(views::finished(request, end));
        self.watchdog.finished(request);
        if terminal.migration().is_some()
            && self.config.canvas.is_none()
            && self
                .running
                .get(&id)
                .is_some_and(|state| state.worker_registered && state.caches_generated_text())
        {
            self.advance_request_cache(id, true);
        }
        let reason = match &terminal {
            Terminal::Finished { reason, .. } => Some(reason.clone()),
            _ => None,
        };
        let terminal_published = self
            .inflight
            .pending_finishes
            .remove(&id)
            .is_some_and(|finish| finish.terminal_published);

        // Report accepted request progress before error teardown removes it.
        if reason == Some(FinishReason::Error)
            && let Some(state) = self.running.get(&id)
        {
            tracing::error!(
                request_id = id.0,
                cause = state.failure.as_deref().unwrap_or("unrecorded"),
                phase = ?state.phase,
                generated_tokens = state.num_generated_tokens,
                logical_position = state.logical_position,
                physical_cache_len = state.cache_visible_len,
                "scheduler request terminated with an internal error"
            );
        }

        // Registered requests retain their allocations until every worker pool
        // acknowledges the ordered close command.
        let mut awaits_close = false;
        let mut allocations = None;
        if let Some(mut st) = self.running.remove(&id) {
            if st.allocations.is_some() {
                // The pool row this request held is free from here on.
                self.rows_held = self.rows_held.saturating_sub(1);
            }
            if matches!(terminal, Terminal::Finished { .. })
                && let Some(resume) = st.req.cache.resume
                && let Some(cache) = &self.storage.cache
            {
                cache.pool().forget_resume(resume);
            }
            let request_key = RequestKey::new(self.engine_id, id, st.request_epoch);
            if st.worker_registered {
                let retained_buffers = self.storage.retained_buffers(request_key);
                let command = BatchCommand::Finish {
                    request_key,
                    retained_buffers,
                };
                self.inflight.pending_commands.push_back(command);
                let mut allocations: Vec<RequestAllocations> = Vec::new();
                let mut buffers = HashMap::new();
                match st.allocations.take() {
                    Some(mut allocation) => {
                        buffers = std::mem::take(&mut allocation.buffers);
                        allocations.push(allocation);
                    }
                    None => self.invariant_broken("a registered request holds its allocations"),
                }
                self.retiring_requests.insert(
                    id,
                    RetiringRequest {
                        request_key,
                        allocations,
                        media_allocations: None,
                        buffers,
                    },
                );
                awaits_close = true;
            } else {
                allocations = st.allocations.take();
            }

            // Remove admission reservations before exposing terminal output so
            // the next scheduler step observes the released capacity.
            self.running_order.retain(|request| *request != id);
            self.storage.reserved_encoder_entries = self
                .storage
                .reserved_encoder_entries
                .saturating_sub(st.req.num_encoder_cache_entries());

            // Its encodings' lookups in the host encoder pools end, those of
            // its calls in flight once they complete.
            if let Some(store) = self.encoder_outputs.as_mut() {
                store.forget(RequestKey::new(self.engine_id, id, st.request_epoch));
            }

            // Releasing the request's encoder pins may make cache products
            // physically reclaimable by their owning worker.
            let mut free_encoder_products = std::mem::take(&mut st.transient_encoder_products);
            for pin in &st.encoder_cache_pins {
                if let Some(product) = self.storage.encoder_cache.release(pin.key, &pin.product) {
                    free_encoder_products.push(product);
                }
            }
            self.free_buffers(
                free_encoder_products
                    .into_iter()
                    .map(|product| product.buffer_id()),
            );

            // A full event channel transfers ownership to the retired-output
            // queue, which drains the terminal event under normal backpressure
            // or hands it to the receiver when the control loop stops. A
            // closed receiver drops the terminal event with the journal. A
            // terminal event published by `finish_after_inflight` may still
            // be journaled, and it is the only one the request publishes.
            let usage = usage(&st);
            let accepted = terminal_published
                || terminal
                    .events(usage)
                    .into_iter()
                    .all(|event| st.output.events.enqueue(event));
            if accepted {
                self.output.retire(id, st.output.events);
            }
        }

        // Unregistered requests have no outstanding worker reference, so their
        // allocations can return to the scheduler immediately.
        if !awaits_close && let Some(allocations) = allocations {
            allocations.free(&mut self.storage);
        }
    }
}

impl Scheduler {
    /// Applies a completed forward's accepted cache extent to the request's
    /// cache tables.
    ///
    /// First publishes the pages the request has now computed, keyed by the
    /// loaded prefill worker that holds them, so every page of a
    /// sliding-window group enters the prefix cache while its table still
    /// holds it: its prompt's, and for a request whose cache holds its
    /// prompt and the text it generates alone, the pages of the tokens it
    /// generated, which its migration's prompt finds. Then, when the host KV
    /// cache serves the worker, reserves the pages to offload ([`offloads`])
    /// while the tables still hold them.
    /// Last retires the sliding-window pages no later reader needs: calls
    /// apply in submission order, so every earlier reader has completed, and
    /// later calls read no history before the accepted extent minus the
    /// window. A request without its prefill worker or prefix cache breaks a
    /// scheduler invariant.
    /// `immutable` also publishes the last partial page, and requires all
    /// writes of this request to have drained permanently.
    pub(super) fn advance_request_cache(&mut self, id: RequestId, immutable: bool) {
        // A canvas deployment's requests write blocks, not their generated
        // tokens in order.
        let tokens_in_order = self.config.canvas.is_none();
        // While a refill restores the request's draft cache, its state
        // tables keep the checkpoint the refill continues from.
        let refill_cursor = self.refill_cursor(id);
        let Some(state) = self.running.get(&id) else {
            return;
        };
        let key = RequestKey::new(self.engine_id, id, state.request_epoch);
        let worker = self
            .dispatch
            .affinity
            .get(&(key, DEFAULT_BLOCK.to_owned()))
            .cloned();
        let source = worker
            .as_ref()
            .and_then(|worker| {
                self.worker
                    .info()
                    .workers
                    .iter()
                    .find(|(id, _)| id == worker)
            })
            .map(|(_, worker)| Arc::new(worker.endpoint.clone()));
        let (Some(source), Some(state), Some(prefix_cache)) = (
            source,
            self.running.get_mut(&id),
            self.storage.cache.as_ref(),
        ) else {
            self.invariant_broken(
                "a prefilled request keeps its loaded prefill worker and prefix cache",
            );
            return;
        };

        // The request's prompt, generated tokens, hashes, page keys and
        // export progress are borrowed beside its tables for the duration of
        // the update.
        let prompt = std::mem::take(&mut state.req.prompt_token_ids);
        let generated_tokens = std::mem::take(&mut state.generated_token_ids);
        let mut hashes = std::mem::take(&mut state.prefix_page_hashes);
        let mut published = std::mem::take(&mut state.prefix_published);
        let published_before = published.clone();
        let visible = state.cache_visible_len as usize;
        let draft_valid = state.draft.valid as usize;
        let cache_write = state.req.cache.write && !state.has_context_images();
        let host_keyed = state.host_keyed;
        let cache_policy = state.req.cache.clone();
        let scope = crate::kv_cache::PrefixScope::of(&cache_policy);
        // Past its prompt, the cache of a request that generates text alone
        // holds the tokens it generated here (a migrated request's earlier
        // ones end its prompt), each written by the decode that reads it:
        // the accepted extent past the prompt, which the tokens recorded
        // cover. Any other request publishes its prompt alone.
        let generated = if tokens_in_order && state.caches_generated_text() {
            let own = &generated_tokens[state
                .req
                .generated_token_ids
                .len()
                .min(generated_tokens.len())..];
            &own[..visible.saturating_sub(prompt.len()).min(own.len())]
        } else {
            &[][..]
        };
        let tokens = CachedTokens::new(&prompt, generated);
        let computed = if generated.is_empty() {
            (state.num_computed_prompt_tokens as usize).min(prompt.len())
        } else {
            tokens.len()
        };
        let offload_from = state.offload_boundary;
        let offload_to = computed.max(offload_from);
        state.offload_boundary = offload_to;
        let kv_offload = self
            .kv_offload
            .as_mut()
            .zip(worker.as_ref())
            .filter(|(kv_offload, worker)| kv_offload.serves(worker));
        if !generated.is_empty() && !hashes.is_empty() {
            prefix_cache.extend_hashes(&mut hashes, tokens, scope);
        }

        let mut reserved = Vec::new();
        let advanced = state.prefix_cache_mut().is_some_and(|allocation| {
            allocation.complete_copies();
            let published_now = prefix_cache.publish_computed(
                allocation,
                tokens,
                &hashes,
                computed,
                draft_valid,
                &mut published,
                cache_write,
                &source,
                scope.resume,
            );
            if immutable {
                prefix_cache.publish_tails(
                    allocation,
                    tokens,
                    computed,
                    scope,
                    cache_write,
                    &source,
                );
            }
            if published_now
                && cache_write
                && let Some((kv_offload, worker)) = kv_offload.as_ref()
            {
                reserved = offloads(
                    kv_offload,
                    worker,
                    id,
                    prefix_cache.pool(),
                    allocation,
                    PublishedPages {
                        tokens,
                        hashes: &hashes,
                        page_keys: if host_keyed { &hashes } else { &[] },
                        before: &published_before,
                        after: &published,
                    },
                    offload_from..offload_to,
                    &source,
                );
            }
            published_now && prefix_cache.release_window(allocation, visible, refill_cursor)
        });
        if let Some((kv_offload, worker)) = kv_offload {
            // Pages published now are offloaded to the host KV cache, but
            // for the host KV cache's own hit, whose loads are now done. Each
            // page keeps its key, under which a withdrawal that evicts it
            // verifies its copy first.
            for (_, keys, pages, _) in &reserved {
                for (key, page) in keys.iter().zip(pages) {
                    prefix_cache.pool().set_host_key(page, *key);
                }
            }
            for (group, keys, pages, kind) in reserved {
                kv_offload.offload(worker, group, keys, pages, kind);
            }
            kv_offload.loaded(id);
        }
        state.req.prompt_token_ids = prompt;
        state.generated_token_ids = generated_tokens;
        state.prefix_page_hashes = hashes;
        state.prefix_published = published;
        if !advanced {
            self.invariant_broken("an admitted request's cache tables match the cache groups");
        }
    }
}

/// A request's tokens as its cache tables publish them: its prompt and the
/// tokens it generated, their page hashes of the device cache and their page
/// keys of the host KV cache by cache group, and the leading pages of each
/// group published before and after the call just accepted.
struct PublishedPages<'a> {
    tokens: CachedTokens<'a>,
    hashes: &'a [Vec<storage::Key>],
    page_keys: &'a [Vec<storage::Key>],
    before: &'a [usize],
    after: &'a [usize],
}

/// The pages of `request`'s tables in `allocation` that the host KV cache
/// offloads after a call on `worker`, by cache group, each with a reference
/// that keeps its units until the offload settles.
///
/// A full-attention group offloads every page published by the call, but
/// for the host KV cache's own hit. A sliding-window group offloads the
/// window before each aligned boundary in `boundaries`
/// ([`KvOffloadManager::offload_boundaries`]), whole or not at all, anchored
/// to the last full-attention page before the boundary. The tables must
/// still hold the windows: the caller runs this before releasing them.
#[expect(
    clippy::too_many_arguments,
    reason = "one request's tables, published pages and offload range"
)]
fn offloads(
    kv_offload: &KvOffloadManager,
    worker: &crate::WorkerGroupId,
    request: RequestId,
    pool: &BlockPool,
    allocation: &CacheAllocation,
    published: PublishedPages<'_>,
    boundaries: std::ops::Range<usize>,
    source: &Arc<ipc::MainProcessEndpoint>,
) -> Vec<(usize, Vec<storage::Key>, Vec<PageRef>, OffloadKind)> {
    let managers = kv_offload.managers(worker);
    let hit = kv_offload.hit_pages(request);
    let anchor_group = kv_offload.anchor_group(worker);
    let boundaries = kv_offload.offload_boundaries(
        worker,
        published.tokens.prompt_len(),
        boundaries.start,
        boundaries.end,
    );
    let mut offloads = Vec::new();
    for (group, manager) in managers.iter().enumerate() {
        let (Some(table), Some(hashes), Some(keys)) = (
            allocation.tables.get(group),
            published.hashes.get(group),
            published.page_keys.get(group),
        ) else {
            continue;
        };
        let block = manager.block_size();
        // A draft cache page's identity covers the token after it.
        let key = table.key_tokens();
        // A reference on a published page of the table, which keeps its
        // units while it is copied.
        let held = |page: usize| {
            let reference = table.page(page)?;
            let tokens = published.tokens.get(page * block..page * block + key)?;
            pool.acquire(
                *reference.units().first()?,
                *hashes.get(page)?,
                group as u32,
                &tokens,
                source,
                None,
            )
        };
        match manager {
            AttentionManager::Full(_) => {
                let hit = hit.get(group).cloned().unwrap_or(0..0);
                let before = published.before.get(group).copied().unwrap_or(0);
                let after = published.after.get(group).copied().unwrap_or(0);
                let (keys, pages): (Vec<_>, Vec<_>) = (before..after)
                    .filter(|page| !hit.contains(page))
                    .filter_map(|page| Some((*keys.get(page)?, held(page)?)))
                    .unzip();
                if !keys.is_empty() {
                    offloads.push((group, keys, pages, OffloadKind::Pages));
                }
            }
            AttentionManager::SlidingWindow(window) => {
                for &boundary in &boundaries {
                    let needed = window.needed_pages(boundary);
                    if needed.is_empty() {
                        continue;
                    }
                    let Some(pages) = needed
                        .map(|page| Some((*keys.get(page)?, held(page)?)))
                        .collect::<Option<Vec<_>>>()
                    else {
                        continue;
                    };
                    let (keys, pages): (Vec<_>, Vec<_>) = pages.into_iter().unzip();
                    let anchor = anchor_group.and_then(|full| {
                        let page = (boundary / managers[full].block_size()).checked_sub(1)?;
                        Some((full, *published.page_keys.get(full)?.get(page)?))
                    });
                    offloads.push((group, keys, pages, OffloadKind::Window { anchor }));
                }
            }
        }
    }
    offloads
}
