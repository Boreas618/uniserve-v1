//! Preparing a request's next call: the call chosen from its accepted state
//! and pending input ranges (context prefill with the encodes its blocks need,
//! the phases of generation), built with the cache units
//! its dispatch owns, and the token constraints that vary with its scheduled
//! sampling position.

use super::*;
use ipc::{ForwardMode, VisionInput};

impl Scheduler {
    /// Builds a call and records the cache units owned by its next dispatch.
    ///
    /// For a forward call, sets `bounds.max_cache_units` to the units every
    /// group's table gained since the last dispatch and advances
    /// `num_cache_units_sent`; a caller that drops the call must roll that back.
    ///
    /// Returns `None` when the request is not running, when preparing it fails
    /// (the request is then finished with `Error`), or, after latching
    /// engine-fatal, when a forward's request holds no cache tables.
    pub(crate) fn prepare_call(
        &mut self,
        id: RequestId,
        build: impl FnOnce(&Self, &GenerationRequest) -> Result<Call, generation::CallError>,
    ) -> Option<Call> {
        let prepared = build(self, &self.requests.tokens(&id)?.req);
        match prepared {
            Ok(mut call) => {
                if matches!(
                    call.kind,
                    CallKind::Forward(ForwardMode::Prefill)
                        | CallKind::Forward(ForwardMode::Decode)
                        | CallKind::Forward(ForwardMode::Verify)
                ) {
                    // The cache owner keeps unit identities. The scheduled record
                    // only needs the number of fresh units for dispatch and drain.
                    let state = self.requests.tokens_mut(&id)?;
                    let Some(allocated) = state
                        .allocations()
                        .map(|allocations| allocations.prefix_cache.allocated_units())
                    else {
                        self.invariant_broken(
                            "a request preparing a forward holds its cache tables",
                        );
                        return None;
                    };
                    call.bounds.max_cache_units = allocated
                        .saturating_sub(state.num_cache_units_sent)
                        .min(u64::from(u32::MAX))
                        as u32;
                    state.num_cache_units_sent = allocated;
                }
                Some(call)
            }
            Err(error) => {
                tracing::error!(
                    request_id = id.0,
                    ?error,
                    "preparing the generation call failed"
                );
                self.finish(id, FinishReason::Error);
                None
            }
        }
    }

    /// Selects the next call using accepted state and pending input ranges.
    ///
    /// Prompt text and input images still to ingest go through
    /// `next_context_call`; otherwise the request's next phase decides.
    /// A block step takes one of the pass's block `rows`, and a context
    /// prefill the segments that fit its context rows within `budget` tokens.
    /// Returns `None` when nothing can be prepared now, including when the
    /// cache capacity a call needs or room for one whole block row is
    /// unavailable.
    pub(crate) fn next_generation_call(
        &mut self,
        id: RequestId,
        budget: usize,
        rows: PassRows,
    ) -> Option<Call> {
        let block_rows = rows.block;
        let (_, cache_visible_len) = self.scheduled_token_lengths(id)?;
        let context_pending = self.requests.tokens(&id).is_some_and(|st| {
            self.num_scheduled_prompt_tokens(id)
                .is_some_and(|count| count < st.req.prompt_token_ids.len() as u32)
                || self
                    .num_scheduled_images(id)
                    .is_some_and(|count| count < st.req.multimodal_inputs.images.len())
        });
        if context_pending {
            return self.next_context_call(id, budget, rows.context);
        }
        let phase = self.next_generation_phase(id)?;
        match phase {
            Phase::Prefill => {
                let st = self.requests.tokens(&id)?;
                let samples = self.config.block_sampling.is_none();
                let n = st.req.prompt_token_ids.len();
                let cursor = self.num_scheduled_prompt_tokens(id)? as usize;
                // Chunked prefill with the clip rule: the chunk is bounded by
                // the remaining step budget and the long-prefill threshold.
                let chunk_cap = (n - cursor)
                    .min(self.config.prefill_chunk())
                    .min(budget.max(1));
                let end = (cursor + chunk_cap.max(1)).min(n);
                // A speculating deployment's prompt chunk writes its draft
                // cache entries: one before the prompt's end reads the
                // token after it, and the last proposes drafts, whose
                // chained steps take the positions after the prompt.
                let draft = if self.speculates() {
                    let prompt = &self.requests.tokens(&id)?.req.prompt_token_ids;
                    ipc::DraftWork {
                        proposals: if end == n {
                            self.proposals_for(id, cache_visible_len)
                        } else {
                            0
                        },
                        lookahead: prompt.get(end).copied(),
                        refill: None,
                    }
                } else {
                    ipc::DraftWork::default()
                };
                if !self.ensure_request_capacity(id, cache_visible_len as usize, end)
                    || !self.ensure_draft_capacity(id, end, end + draft.proposals as usize)
                {
                    return None;
                }
                let sampling_state = self.build_token_masks(id, 0);
                self.prepare_call(id, |_scheduler, request| {
                    generation::prepare_prompt(
                        request,
                        cursor as u32,
                        end as u32,
                        Vec::new(),
                        0,
                        sampling_state,
                        samples,
                    )
                    .map(|mut call| {
                        call.draft = draft;
                        call
                    })
                })
            }
            // A speculating request verifies the drafts its last forward
            // proposed, as many as the scheduler chooses this step: none
            // while its draft cache is incomplete, a verification row of
            // one token. Its successor waits for its result, which states
            // the tokens it accepted and the drafts after them.
            Phase::DecodeUnd if self.speculates() => {
                let sampling_state = self.build_token_masks(id, 0);
                let input_token = self.requests.tokens(&id)?.next_token;
                let drafts = self.verify_drafts(id, cache_visible_len);
                let proposals = self.proposals_for(id, cache_visible_len);
                // The cache holds the verified tokens and, for the drafter's
                // chained steps, the positions of the drafts it proposes.
                // The cache, not the rotary position, is what the tokens occupy: an image's
                // block takes other counts of the two.
                let capacity_target = decode_capacity_target(
                    cache_visible_len as usize,
                    drafts.len() + proposals as usize,
                );
                if !self.ensure_decode_capacity(id, cache_visible_len as usize, capacity_target) {
                    // The request stays where it is for this step, and the pass
                    // stays non-token behind it. A dispatch output the mechanism cannot
                    // build is a stall the rest of the epoch cannot see, so it
                    // is named at `warn`: a policy that keeps asking for a
                    // verification the cache has no room for would otherwise
                    // park the request in silence.
                    tracing::warn!(
                        request = id.0,
                        visible = cache_visible_len,
                        target = capacity_target,
                        drafts = drafts.len(),
                        proposals,
                        "a verification waits: the cache does not cover the positions it lands on"
                    );
                    return None;
                }
                let prepared = self.prepare_call(id, |_scheduler, request| {
                    generation::prepare_verify(
                        request,
                        input_token,
                        &drafts,
                        proposals,
                        sampling_state,
                    )
                });
                if prepared.is_none() {
                    tracing::warn!(
                        request = id.0,
                        drafts = drafts.len(),
                        proposals,
                        "a verification could not be built from the request's state"
                    );
                }
                prepared
            }
            Phase::DecodeUnd => {
                let projected_successor = self.inflight.has_pending_calls(id);
                // Stage minimum-token and force-finish constraints at the
                // successor's scheduled token position. Device predicates
                // prevent an inactive descendant from accepting that position.
                let projected = if projected_successor {
                    self.inflight.num_pending_calls(id)
                } else {
                    0
                };
                let sampling_state = self.build_token_masks(id, projected);
                let st = self.requests.tokens(&id)?;
                // The prior committed token this decode continues from. It is
                // attached as a host input only when no exact selected-point
                // product is eligible for device continuation.
                let input_token = st.next_token;
                let relay_input = projected_successor || self.can_reuse_resolved_token_product(id);
                let capacity_target = decode_capacity_target(cache_visible_len as usize, 0);
                if !self.ensure_decode_capacity(id, cache_visible_len as usize, capacity_target) {
                    return None;
                }

                self.prepare_call(id, |_scheduler, request| {
                    generation::prepare_decode(request, input_token, relay_input, sampling_state)
                })
            }
            // The hand-off copies the request's cache with no call.
            Phase::HandOff => None,
            Phase::IngestState => self.next_context_call(id, budget, rows.context),
            Phase::Block => {
                // One step of the request's block is one read-only block row
                // of the pass. The next step may be queued behind the steps
                // still in flight, predicated on the last one's completion:
                // the worker runs it as a no-op once an earlier step stopped
                // the block, so the block's steps run back to back without
                // waiting for each result. A step past the block's step limit
                // is never queued, since the last step always stops it.
                //
                // A block's first step may likewise be queued behind the
                // prefill that completes its context: the prompt's last
                // chunk, or the previous block's commit, which makes the
                // prepared step the next block's step zero.
                let st = self.requests.tokens(&id)?;
                let sampling = self.config.block_sampling?;
                let block_length = sampling.block_length as usize;
                let pending = self.inflight.num_pending_calls(id);
                if (pending > 0
                    && !self
                        .can_queue_successor(id, CallKind::Forward(ForwardMode::TokenDenoising)))
                    || block_rows == 0
                    || block_length > budget
                {
                    return None;
                }
                // The steps in flight that the prepared step follows are those
                // behind the latest queued prefill: steps queued before a
                // commit are the stopped block's no-ops.
                let queued = self.inflight.pending_calls.get(&id);
                let steps_in_flight = queued.map_or(0, |calls| {
                    calls
                        .iter()
                        .rev()
                        .take_while(|inflight| !is_prompt_extend(&inflight.call))
                        .filter(|inflight| inflight.call.block_step.is_some())
                        .count()
                });
                let commit_in_flight = st.phase == Phase::CommitBlock
                    && queued.is_some_and(|calls| {
                        calls
                            .iter()
                            .any(|inflight| is_prompt_extend(&inflight.call))
                    });
                let (block, first_step) = if commit_in_flight {
                    (st.block_index.saturating_add(1), 0)
                } else {
                    (st.block_index, st.block_step)
                };
                let step =
                    first_step.saturating_add(u32::try_from(steps_in_flight).unwrap_or(u32::MAX));
                if step >= sampling.max_steps {
                    return None;
                }
                self.prepare_call(id, |_scheduler, request| {
                    generation::prepare_block_step(request, &sampling, block, step)
                })
            }
            Phase::CommitBlock => {
                // The stopped block's tokens extend the context as one causal
                // prefill row. The commit is prepared once the host accepts
                // the step that stopped the block, queued without a predicate
                // behind the block's steps still in flight, which are no-ops
                // (`follows_unconditionally`), so the worker launches it
                // while they run.
                let st = self.requests.tokens(&id)?;
                let tokens = st.block_commit.clone();
                if (self.inflight.has_pending_calls(id)
                    && !self.can_queue_successor(id, CallKind::Forward(ForwardMode::Prefill)))
                    || tokens.len() > budget
                {
                    return None;
                }
                let end = (cache_visible_len as usize).saturating_add(tokens.len());
                if !self.ensure_request_capacity(id, cache_visible_len as usize, end) {
                    return None;
                }
                self.prepare_call(id, |_scheduler, request| {
                    generation::prepare_block_commit(request, &tokens)
                })
            }
        }
    }

    /// Prepares the next context-ingest call within `budget` tokens.
    ///
    /// The context is the prompt with the blocks of its input images at their
    /// positions: each encoder of an image writes one block, in encoder order.
    /// A vision block is written by a context prefill together with the prompt
    /// tokens around it, as one attention block of that call; a latent block
    /// is written by an image extension of its own, which ends the preceding
    /// context prefill. Every block's encoder product exists before the call
    /// that writes it: the next block the context needs is encoded first
    /// (`next_context_encode`), or pinned from the encoder cache without a
    /// call.
    ///
    /// A context prefill holds prompt tokens and whole vision blocks, within
    /// the pass's `budget` and the long-prefill threshold; a block that would
    /// not fit waits for the next chunk, which it starts, and a chunk that
    /// starts with a block may exceed the threshold within the budget. It is
    /// one forward row, which `context_rows` must leave room for.
    /// Returns `None` when nothing can be prepared now, including when the
    /// next block does not fit this pass or the request's cache capacity is
    /// unavailable.
    pub(crate) fn next_context_call(
        &mut self,
        id: RequestId,
        budget: usize,
        context_rows: usize,
    ) -> Option<Call> {
        if context_rows == 0 {
            return None;
        }
        let cursor = self.num_scheduled_prompt_tokens(id)? as usize;
        let (_, cache_visible_len) = self.scheduled_token_lengths(id)?;

        // A latent block's encoder product awaits its own extension.
        if self.requests.tokens(&id)?.phase == Phase::IngestState {
            let state = self.requests.tokens(&id)?;
            let feature = state.input_image_features.clone()?;
            let (_, encoder_input) = state.input_block(state.block_cursor())?;
            let encoder_input = *encoder_input;
            // Submission validated every input encoder's capacity.
            let Ok(physical_bound) = encoder_input.cache_token_capacity(&self.generation_limits)
            else {
                self.invariant_broken("a queued input encoder has a valid cache capacity");
                return None;
            };
            if !self.ensure_request_capacity(
                id,
                cache_visible_len as usize,
                cache_visible_len.saturating_add(physical_bound) as usize,
            ) {
                return None;
            }
            return self.prepare_call(id, |scheduler, request| {
                generation::prepare_image_extend(
                    &scheduler.generation_limits,
                    request,
                    encoder_input,
                    feature,
                )
            });
        }

        // Encode the next block the context needs. A product the encoder
        // cache holds is pinned for the request instead, without a call;
        // otherwise the encoding is looked up in the host encoder pools before
        // its call is prepared.
        while let Some((block, step)) = self.next_context_encode(id) {
            let state = self.requests.tokens(&id)?;
            let (image, _) = state.input_block(block)?;
            let Some(encoding) = self.encoding(&state.req, block) else {
                self.invariant_broken("a queued input encoder has a valid cache capacity");
                return None;
            };
            let cache_key = encoding.device_key();
            let image_b64 = image.b64.clone();
            let cached = if state.req.cache.read {
                self.storage.encoder_cache.lookup_product(cache_key)
            } else {
                None
            };
            let Some(cached_product) = cached else {
                // A cache miss schedules the encoder. Nothing else of the
                // request is prepared while it is in flight, so its completion
                // finds the same block here.
                if !self.encoding_ready(id, encoding) {
                    return None;
                }
                let state = self.requests.tokens(&id)?;
                let draw = generation::latent_draw(&state.req, block);
                return self.prepare_call(id, |scheduler, request| {
                    generation::prepare_encode(
                        &scheduler.generation_limits,
                        request,
                        step.into(),
                        image_b64,
                        Some(draw),
                    )
                });
            };
            let product = self.storage.encoder_cache.acquire(cache_key)?;
            // Preparing a call that reads a cached feature re-uses the entry, so
            // the policy's LRU sees the use.
            self.record_encoded(cache_key);
            if product != cached_product {
                return None;
            }
            // A host lookup the request started before the cache held the
            // product ends.
            self.drop_host_encoding(id, encoding);
            // If request ownership disappeared during acquisition, release
            // the pin.
            let Some(state) = self.requests.tokens_mut(&id) else {
                let _ = self.storage.encoder_cache.release(cache_key, &product);
                return None;
            };
            state.encoder_cache_pins.push(EncoderCachePin {
                key: cache_key,
                product: product.clone(),
            });
            if step == crate::ImageEncoder::Latent {
                state.input_image_features = Some(product);
                state.phase = Phase::IngestState;
                return self.next_context_call(id, budget, context_rows);
            }
            state.context_features.push(product);
        }

        // Select the next chunk: prompt tokens up to the budget, and each
        // encoded vision block they reach whole. A latent block or a block
        // not yet encoded ends the chunk.
        let limit = budget.max(1).min(self.config.prefill_chunk());
        let (end, blocks, block_tokens) = {
            let state = self.requests.tokens(&id)?;
            let prompt_len = state.req.prompt_token_ids.len();
            let mut end = cursor;
            let mut used = 0usize;
            let mut blocks = Vec::new();
            let mut block_tokens = 0u32;
            let sc = self.scheduled_context(id);
            let mut block = sc?.blocks;
            // In-flight context prefills hold the leading features.
            let in_flight = state.blocks_between(state.block_cursor(), block);
            loop {
                let next = state.input_block(block);
                let text_stop = next.map_or(prompt_len, |(image, _)| image.position as usize);
                let taken = text_stop
                    .saturating_sub(end)
                    .min(limit.saturating_sub(used));
                end += taken;
                used += taken;
                let Some((image, input)) = next else {
                    break;
                };
                // An image's positions count once its last block is written.
                let positions = if block.encoder + 1 == image.encoders.len() {
                    image.num_positions
                } else {
                    0
                };
                let Some(feature) = state.context_features.get(in_flight + blocks.len()) else {
                    break;
                };
                if end < text_stop || input.encoder != crate::ImageEncoder::Vision {
                    break;
                }
                // Submission validated every vision block's exact cache length.
                let Some(tokens) = input.num_cache_tokens else {
                    self.invariant_broken("an input vision block declares its exact cache length");
                    return None;
                };
                // A block joins only whole: one that would overflow the chunk
                // starts the next one, where it may exceed the threshold
                // within the pass's budget.
                let fits =
                    used + tokens as usize <= limit || (used == 0 && tokens as usize <= budget);
                if !fits {
                    break;
                }
                blocks.push(VisionInput {
                    offset: u32::try_from(end - cursor).unwrap_or(u32::MAX),
                    feature: feature.clone(),
                    rows: tokens,
                    positions,
                });
                used += tokens as usize;
                block_tokens = block_tokens.saturating_add(tokens);
                block = state.next_block(block)?;
            }
            (end, blocks, block_tokens)
        };
        if end == cursor && blocks.is_empty() {
            return None;
        }

        let written = end - cursor + block_tokens as usize;
        if !self.ensure_request_capacity(
            id,
            cache_visible_len as usize,
            cache_visible_len as usize + written,
        ) {
            return None;
        }

        let sampling_state = self.build_token_masks(id, 0);
        let samples = self.config.block_sampling.is_none();
        self.prepare_call(id, |_scheduler, request| {
            generation::prepare_prompt(
                request,
                cursor as u32,
                end as u32,
                blocks,
                block_tokens,
                sampling_state,
                samples,
            )
        })
    }

    /// Returns the next input-image block to encode before the context can
    /// advance, with its encoder, or `None` when the next context call needs
    /// no encode.
    ///
    /// Goes through the blocks after the accepted block cursor, whose leading
    /// products `context_features` holds: the latent block at the scheduled
    /// prompt cursor is encoded for its own extension; vision blocks are
    /// encoded in order ahead of the context prefill that writes them, while
    /// their image lies within the long-prefill threshold of the scheduled
    /// prompt cursor. The search stops at a latent block past that cursor,
    /// which the chunk before it does not reach. A request schedules nothing
    /// while its encode is in flight, so the encode's completion identifies
    /// its block by the same search.
    pub(crate) fn next_context_encode(
        &self,
        id: RequestId,
    ) -> Option<(BlockCursor, crate::ImageEncoder)> {
        let state = self.requests.tokens(&id)?;
        if state.phase == Phase::IngestState {
            return None;
        }
        let cursor = self.num_scheduled_prompt_tokens(id)? as usize;
        let reach = cursor.saturating_add(self.config.prefill_chunk());
        let mut block = state.block_cursor();
        let mut encoded = 0;
        while let Some((image, input)) = state.input_block(block) {
            let position = image.position as usize;
            if input.encoder == crate::ImageEncoder::Latent {
                return (position == cursor).then_some((block, input.encoder));
            }
            if position > cursor && position >= reach {
                return None;
            }
            if encoded == state.context_features.len() {
                return Some((block, input.encoder));
            }
            encoded += 1;
            block = state.next_block(block)?;
        }
        None
    }

    /// Returns the forward-row lengths of a context prefill `call` with vision
    /// blocks, in context order: the prompt tokens before, between and after
    /// its blocks, and each block's exact cache length. The blocks are the
    /// request's next ones after its scheduled block cursor.
    pub(super) fn context_segments(&self, id: RequestId, call: &Call) -> Option<Vec<u32>> {
        let state = self.requests.tokens(&id)?;
        let mut block = self.scheduled_context(id)?.blocks;
        let mut segments = Vec::with_capacity(2 * call.vision_inputs.len() + 1);
        let mut text_start = 0u32;
        for input in &call.vision_inputs {
            if input.offset < text_start {
                return None;
            }
            if input.offset > text_start {
                segments.push(input.offset - text_start);
                text_start = input.offset;
            }
            let (_, encoder) = state.input_block(block)?;
            segments.push(encoder.num_cache_tokens?);
            block = state.next_block(block)?;
        }
        let text = u32::try_from(call.input_token_ids.len()).ok()?;
        if text > text_start {
            segments.push(text - text_start);
        }
        Some(segments)
    }

    /// Builds only constraints that vary with the scheduled sampling position.
    /// Admission supplies the static whitelist and finish tokens. `projected`
    /// counts unresolved predecessors, so minimum-token limits apply at the
    /// successor's position without observing device results. Penalties
    /// remain owned by the worker's accepted-token state.
    ///
    /// Returns `None` when the request is not running or nothing needs
    /// constraining.
    pub(crate) fn build_token_masks(
        &self,
        id: RequestId,
        projected: usize,
    ) -> Option<SamplingState> {
        let n_generated = self.requests.tokens(&id).map_or(0, |state| {
            state.num_generated_tokens.saturating_add(projected)
        });
        let state = self.requests.tokens(&id)?;
        let suppress = suppressed_tokens(
            &state.req.sampling,
            &self.special_tokens.eos,
            &state.generated_token_ids,
            n_generated,
            &[],
        );

        let sampling = SamplingState {
            allowed_token_ids: None,
            suppressed_token_ids: suppress,
            finish_token_ids: Vec::new(),
            force_finish: n_generated.saturating_add(1) >= state.req.max_und_tokens,
        };
        (sampling != SamplingState::default()).then_some(sampling)
    }
}

/// The tokens a request sampling with `sampling` may not sample at the position after its
/// `n_generated` generated tokens followed by `drafts`, where `generated` holds the ids of the
/// generated tokens (a migrated request's included): its end-of-sequence tokens `eos` while it is
/// under its `min_tokens` floor, and the last token of every bad word whose prefix the generated
/// tokens and the drafts end with. The rule is vLLM's (`apply_bad_words`, over the output tokens
/// only), sorted and deduplicated; the worker's sampler masks these tokens to `-inf` before the
/// penalties.
pub(crate) fn suppressed_tokens(
    sampling: &proto::request::SamplingParams,
    eos: &[u32],
    generated: &[u32],
    n_generated: usize,
    drafts: &[u32],
) -> Vec<u32> {
    let mut suppress = Vec::new();
    if n_generated.saturating_add(drafts.len()) < sampling.min_tokens {
        suppress.extend_from_slice(eos);
    }
    for word in &sampling.bad_words_ids {
        let Some((&last, prefix)) = word.split_last() else {
            continue;
        };
        // The history is the generated tokens, then the drafts: a prefix longer than the drafts
        // ends in the generated tokens, a shorter one in the drafts alone.
        let matches = if prefix.len() <= drafts.len() {
            drafts.ends_with(prefix)
        } else {
            let (head, tail) = prefix.split_at(prefix.len() - drafts.len());
            tail == drafts && generated.ends_with(head)
        };
        if matches {
            suppress.push(last);
        }
    }
    suppress.sort_unstable();
    suppress.dedup();
    suppress
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bad word bans its last token exactly where the history ends with its prefix, the history
    /// being the generated tokens and then the drafts; a single-token word bans its token always,
    /// and the `min_tokens` floor bans the end-of-sequence tokens until the floor is met.
    #[test]
    fn a_bad_word_bans_its_last_token_where_its_prefix_ends_the_history() {
        let sampling = proto::request::SamplingParams {
            bad_words_ids: vec![vec![4], vec![7, 8, 9]],
            min_tokens: 3,
            ..proto::request::SamplingParams::default()
        };
        let eos = [2];
        // Under the floor: the end-of-sequence token and the single-token word.
        assert_eq!(suppressed_tokens(&sampling, &eos, &[1], 1, &[]), vec![2, 4]);
        // At the floor, with the three-token word's prefix ending the generated tokens.
        assert_eq!(
            suppressed_tokens(&sampling, &eos, &[1, 7, 8], 3, &[]),
            vec![4, 9]
        );
        // The prefix across the generated tokens and the drafts, and in the drafts alone.
        assert_eq!(
            suppressed_tokens(&sampling, &eos, &[1, 7], 3, &[8]),
            vec![4, 9]
        );
        assert_eq!(
            suppressed_tokens(&sampling, &eos, &[1], 3, &[7, 8]),
            vec![4, 9]
        );
        // A prefix the history does not end with bans nothing more; the drafts count toward the
        // floor.
        assert_eq!(
            suppressed_tokens(&sampling, &eos, &[7, 8], 2, &[5]),
            vec![4]
        );
        assert_eq!(
            suppressed_tokens(&sampling, &eos, &[1], 1, &[5]),
            vec![2, 4]
        );
    }
}
