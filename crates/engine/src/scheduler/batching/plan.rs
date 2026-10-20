//! Planning a request's next call: the call chosen from its accepted state
//! and pending input ranges (context prefill with the encodes its blocks need,
//! the phases of generation, the close of a round), built with the cache units
//! its dispatch owns, and the token constraints that vary with its scheduled
//! sampling position.

use super::*;
use uniserve_ipc::{ForwardMode, VisionInput};

impl Scheduler {
    /// Builds a call and records the cache units owned by its next dispatch.
    ///
    /// For a forward call, sets `bounds.max_cache_units` to the units every
    /// group's table gained since the last dispatch and advances
    /// `num_cache_units_sent`; a caller that drops the call must roll that back.
    ///
    /// Returns `None` when the request is not running, when planning fails
    /// (the request is then finished with `Error`), or, after latching
    /// engine-fatal, when a forward's request holds no cache tables.
    pub(in crate::scheduler) fn plan_call(
        &mut self,
        id: RequestId,
        build: impl FnOnce(&Self, &GenerationRequest) -> Result<Call, generation::PlanningError>,
    ) -> Option<Call> {
        let planned = build(self, &self.running.get(&id)?.req);
        match planned {
            Ok(mut call) => {
                if matches!(
                    call.kind,
                    CallKind::Forward(ForwardMode::Prefill)
                        | CallKind::Forward(ForwardMode::Decode)
                        | CallKind::Forward(ForwardMode::Verify)
                ) {
                    // The cache owner keeps unit identities. The scheduled record
                    // only needs the number of fresh units for dispatch and drain.
                    let state = self.running.get_mut(&id)?;
                    let Some(allocated) = state
                        .allocations()
                        .map(|allocations| allocations.prefix_cache.allocated_units())
                    else {
                        self.invariant_broken(
                            "a request planning a forward holds its cache tables",
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
                tracing::error!(request_id = id.0, ?error, "generation planning failed");
                self.finish(id, FinishReason::Error);
                None
            }
        }
    }

    /// Selects the next call using accepted state and pending input ranges.
    ///
    /// Prompt text and input images still to ingest go through
    /// `next_context_call`; otherwise the request's next phase decides.
    /// A canvas pass takes the next readout rows that fit both `budget`
    /// tokens and the pass's canvas `rows`, and a context prefill the
    /// segments that fit its context rows. Returns `None` when nothing can be
    /// planned now, including when the cache capacity a call needs, the
    /// image-branch reservation, or room for one whole canvas row is
    /// unavailable.
    pub(in crate::scheduler) fn next_generation_call(
        &mut self,
        id: RequestId,
        budget: usize,
        rows: PassRows,
    ) -> Option<Call> {
        let canvas_rows = rows.canvas;
        let (logical_position, cache_visible_len) = self.scheduled_token_lengths(id)?;
        let context_pending = self.running.get(&id).is_some_and(|st| {
            self.num_scheduled_prompt_tokens(id)
                .is_some_and(|count| count < st.req.prompt_token_ids.len() as u32)
                || self
                    .num_scheduled_images(id)
                    .is_some_and(|count| count < st.req.multimodal_inputs.images.len())
        });
        if context_pending {
            return self.next_context_call(id, budget, rows.context);
        }
        if self.running.get(&id)?.image_reservation_pending
            && !self.promote_gen_branch_reservation(id)
        {
            return None;
        }
        let phase = self.next_generation_phase(id)?;
        let image_id = self.running.get(&id)?.image_id;
        match phase {
            Phase::Prefill => {
                let st = self.running.get(&id)?;
                let n = st.req.prompt_token_ids.len();
                let cursor = self.num_scheduled_prompt_tokens(id)? as usize;
                // Chunked prefill with the clip rule: the chunk is bounded by
                // the remaining step budget and the long-prefill threshold.
                let chunk_cap = (n - cursor)
                    .min(self.config.chunked_prefill_size)
                    .min(budget.max(1));
                let end = (cursor + chunk_cap.max(1)).min(n);
                if !self.ensure_request_capacity(id, cache_visible_len as usize, end) {
                    return None;
                }
                let sampling_state = self.build_token_masks(id, 0, 0);

                let generates_canvases = self.config.canvas.is_some();
                self.plan_call(id, |_scheduler, request| {
                    generation::plan_prompt(
                        request,
                        cursor as u32,
                        end as u32,
                        Vec::new(),
                        0,
                        sampling_state,
                        generates_canvases,
                    )
                })
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
                let sampling_state = self.build_token_masks(id, projected, 0);
                let st = self.running.get(&id)?;
                // The prior committed token this decode continues from. It is
                // attached as a host input only when no exact selected-point
                // product is eligible for device continuation.
                let input_token = st.next_token;
                let pos = logical_position;
                let relay_input = projected_successor || self.can_reuse_resolved_token_product(id);
                let capacity_target = decode_capacity_target(pos as usize, 0);
                if !self.ensure_request_capacity(id, cache_visible_len as usize, capacity_target) {
                    return None;
                }

                self.plan_call(id, |_scheduler, request| {
                    generation::plan_decode(request, None, input_token, relay_input, sampling_state)
                })
            }
            Phase::CloseCache => {
                let token = self.running.get(&id)?.next_token;
                let capacity_target = decode_capacity_target(cache_visible_len as usize, 0);
                if !self.ensure_request_capacity(id, cache_visible_len as usize, capacity_target) {
                    return None;
                }

                self.plan_call(id, |_scheduler, request| {
                    generation::plan_close_cache(request, token, false)
                })
            }
            Phase::PublishCache => self.plan_call(id, |scheduler, request| {
                generation::plan_cache_export(
                    scheduler
                        .info
                        .prefix_cache
                        .as_ref()
                        .map_or(0, |cache| cache.transfer_bytes(cache_visible_len)),
                    request,
                )
            }),
            Phase::LatentPreparation => {
                let conditioning = self.cache_conditioning(id)?;
                self.plan_call(id, |scheduler, request| {
                    generation::plan_diffusion_prepare(
                        &scheduler.generation_limits,
                        scheduler.latent_dtype,
                        request,
                        image_id,
                        conditioning,
                    )
                })
            }
            Phase::Denoising => {
                // Each call covers the next step of the schedule, and none
                // is planned once every step is submitted. Accepted
                // completion remains separately tracked on the request.
                let st = self.running.get(&id)?;
                if self.num_scheduled_denoise_steps(id)? >= st.denoising.steps() {
                    return None;
                }
                let conditioning = self.cache_conditioning(id)?;
                let latent = self
                    .pending_output(id, |call| call.latent_output.as_ref())
                    .or(self.running.get(&id)?.denoising.latent())
                    .cloned()?;
                self.plan_call(id, |scheduler, request| {
                    generation::plan_diffusion_step(
                        &scheduler.generation_limits,
                        scheduler.latent_dtype,
                        request,
                        conditioning,
                        latent,
                    )
                })
            }
            Phase::ImageDecoding => {
                let latent = self
                    .pending_output(id, |call| call.latent_output.as_ref())
                    .or(self.running.get(&id)?.denoising.latent())
                    .cloned()?;
                self.plan_call(id, |_scheduler, request| {
                    generation::plan_diffusion_finalize(request, latent)
                })
            }
            Phase::FeedbackEncode => {
                let st = self.running.get(&id)?;
                let feedback = &st.req.image_generation;
                feedback.feedback_source.as_ref()?;
                let encoder_index = self.scheduled_feedback(id)?.0;
                let encoder = feedback.feedback_encoders.get(encoder_index)?.encoder;
                let source = (feedback.feedback_source
                    == Some(uniserve_core::FeedbackSource::DeviceProduct))
                .then(|| {
                    self.pending_output(id, |call| call.image_output.as_ref())
                        .or(st.feedback_source.as_ref())
                        .cloned()
                })
                .flatten();
                let image_b64 = if source.is_none() {
                    st.feedback_image_b64.clone().unwrap_or_default()
                } else {
                    String::new()
                };
                self.plan_call(id, |scheduler, request| {
                    generation::plan_encode(
                        &scheduler.generation_limits,
                        request,
                        encoder,
                        image_b64,
                        source,
                        true,
                    )
                })
            }
            Phase::FeedbackState => {
                let st = self.running.get(&id)?;
                let feedback = &st.req.image_generation;
                feedback.feedback_source.as_ref()?;
                let encoder_index = self.scheduled_feedback(id)?.0;
                let is_final_encoder = encoder_index + 1 == feedback.feedback_encoders.len();
                let encoder_input = *feedback.feedback_encoders.get(encoder_index)?;
                let feature = self
                    .pending_output(id, |call| call.encoder_output.as_ref())
                    .or(st.feedback_features.as_ref())
                    .cloned();
                let sample_continuation = is_final_encoder && feedback.sample_feedback_continuation;
                let Some(feature) = feature else {
                    self.finish(id, FinishReason::Error);
                    return None;
                };

                let sampling_state = self.build_token_masks(id, 0, usize::from(is_final_encoder));
                self.plan_call(id, |scheduler, request| {
                    generation::plan_image_extend(
                        &scheduler.generation_limits,
                        request,
                        encoder_input,
                        feature,
                        true,
                        sample_continuation,
                        if sample_continuation {
                            sampling_state
                        } else {
                            None
                        },
                    )
                })
            }
            Phase::IngestState => self.next_context_call(id, budget, rows.context),
            Phase::Readout => {
                // Whole rows join the pass in report order while they fit its
                // remaining tokens and canvas rows.
                let start = self.scheduled_readout_rows(id)?;
                let st = self.running.get(&id)?;
                let mut end = start;
                let mut tokens = 0usize;
                for row in st.req.readout.get(start..)? {
                    if end - start >= canvas_rows || tokens + row.token_ids.len() > budget {
                        break;
                    }
                    tokens += row.token_ids.len();
                    end += 1;
                }
                if end == start {
                    return None;
                }
                self.plan_call(id, |_scheduler, request| {
                    generation::plan_readout(request, start..end)
                })
            }
            Phase::Canvas => {
                // One step of the request's block is one read-only canvas row
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
                // planned step the next block's step zero.
                let st = self.running.get(&id)?;
                let canvas = self.config.canvas?;
                let canvas_length = canvas.canvas_length as usize;
                let pending = self.inflight.num_pending_calls(id);
                if (pending > 0
                    && !self
                        .can_queue_successor(id, CallKind::Forward(ForwardMode::TokenDenoising)))
                    || canvas_rows == 0
                    || canvas_length > budget
                {
                    return None;
                }
                // The steps in flight that the planned step follows are those
                // behind the latest queued prefill: steps queued before a
                // commit are the stopped block's no-ops.
                let queued = self.inflight.pending_calls.get(&id);
                let steps_in_flight = queued.map_or(0, |calls| {
                    calls
                        .iter()
                        .rev()
                        .take_while(|inflight| !is_prompt_extend(&inflight.call))
                        .filter(|inflight| inflight.call.canvas.is_some())
                        .count()
                });
                let commit_in_flight = st.phase == Phase::CommitCanvas
                    && queued.is_some_and(|calls| {
                        calls
                            .iter()
                            .any(|inflight| is_prompt_extend(&inflight.call))
                    });
                let (block, first_step) = if commit_in_flight {
                    (st.canvas_block.saturating_add(1), 0)
                } else {
                    (st.canvas_block, st.canvas_step)
                };
                let step =
                    first_step.saturating_add(u32::try_from(steps_in_flight).unwrap_or(u32::MAX));
                if step >= canvas.max_steps {
                    return None;
                }
                self.plan_call(id, |_scheduler, request| {
                    generation::plan_canvas_step(request, &canvas, block, step)
                })
            }
            Phase::CommitCanvas => {
                // The stopped block's tokens extend the context as one causal
                // prefill row. The commit is planned once the host accepts
                // the step that stopped the block, queued without a predicate
                // behind the block's steps still in flight, which are no-ops
                // (`follows_unconditionally`), so the worker launches it
                // while they run.
                let st = self.running.get(&id)?;
                let tokens = st.canvas_commit.clone();
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
                self.plan_call(id, |_scheduler, request| {
                    generation::plan_canvas_commit(request, &tokens)
                })
            }
        }
    }

    /// Plans the next context-ingest call within `budget` tokens.
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
    /// starts with a block may exceed the threshold within the budget. Its
    /// forward rows (`generation::prefill_rows`) stay within `context_rows`.
    /// Returns `None` when nothing can be planned now, including when the
    /// next block does not fit this pass or the request's cache capacity is
    /// unavailable.
    pub(in crate::scheduler) fn next_context_call(
        &mut self,
        id: RequestId,
        budget: usize,
        context_rows: usize,
    ) -> Option<Call> {
        let cursor = self.num_scheduled_prompt_tokens(id)? as usize;
        let (_, cache_visible_len) = self.scheduled_token_lengths(id)?;

        // A latent block's encoder product awaits its own extension.
        if self.running.get(&id)?.phase == Phase::IngestState {
            let state = self.running.get(&id)?;
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
            return self.plan_call(id, |scheduler, request| {
                generation::plan_image_extend(
                    &scheduler.generation_limits,
                    request,
                    encoder_input,
                    feature,
                    false,
                    false,
                    None,
                )
            });
        }

        // Encode the next block the context needs. A product the encoder
        // cache holds is pinned for the request instead, without a call.
        while let Some((block, step)) = self.next_context_encode(id) {
            let state = self.running.get(&id)?;
            let (image, _) = state.input_block(block)?;
            let cache_key = encoder_cache_key(image.hash, block.encoder, step);
            let image_b64 = image.b64.clone();
            let cached = if state.req.cache.read {
                self.storage.encoder_cache.lookup_product(cache_key)
            } else {
                None
            };
            let Some(cached_product) = cached else {
                // A cache miss schedules the encoder. Nothing else of the
                // request is planned while it is in flight, so its completion
                // finds the same block here.
                return self.plan_call(id, |scheduler, request| {
                    generation::plan_encode(
                        &scheduler.generation_limits,
                        request,
                        step,
                        image_b64,
                        None,
                        false,
                    )
                });
            };
            let product = self.storage.encoder_cache.acquire(cache_key)?;
            if product != cached_product {
                return None;
            }
            // If request ownership disappeared during acquisition, release
            // the pin.
            let Some(state) = self.running.get_mut(&id) else {
                let _ = self.storage.encoder_cache.release(cache_key, &product);
                return None;
            };
            state.encoder_cache_pins.push(EncoderCachePin {
                key: cache_key,
                product: product.clone(),
            });
            if step == MediaCall::LatentEncoding {
                state.input_image_features = Some(product);
                state.phase = Phase::IngestState;
                return self.next_context_call(id, budget, context_rows);
            }
            state.context_features.push(product);
        }

        // Select the next chunk: prompt tokens up to the budget, and each
        // encoded vision block they reach whole. A latent block or a block
        // not yet encoded ends the chunk.
        let limit = budget.max(1).min(self.config.chunked_prefill_size);
        let (end, blocks, block_tokens) = {
            let state = self.running.get(&id)?;
            let prompt_len = state.req.prompt_token_ids.len();
            let mut end = cursor;
            let mut used = 0usize;
            let mut rows = 0usize;
            let mut blocks = Vec::new();
            let mut block_tokens = 0u32;
            let mut block = self.scheduled_context(id)?.blocks;
            // In-flight context prefills hold the leading features.
            let in_flight = state.blocks_between(state.block_cursor(), block);
            loop {
                let next = state.input_block(block);
                let text_stop = next.map_or(prompt_len, |(image, _)| image.position as usize);
                // A run of prompt tokens is one forward row, as is a block.
                let room = if rows < context_rows {
                    limit.saturating_sub(used)
                } else {
                    0
                };
                let taken = text_stop.saturating_sub(end).min(room);
                if taken > 0 {
                    rows += 1;
                }
                end += taken;
                used += taken;
                let Some((_, input)) = next else {
                    break;
                };
                let Some(feature) = state.context_features.get(in_flight + blocks.len()) else {
                    break;
                };
                if end < text_stop
                    || input.encoder != MediaCall::VisionEncoding
                    || rows >= context_rows
                {
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
                });
                rows += 1;
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

        let sampling_state = self.build_token_masks(id, 0, 0);
        let generates_canvases = self.config.canvas.is_some();
        self.plan_call(id, |_scheduler, request| {
            generation::plan_prompt(
                request,
                cursor as u32,
                end as u32,
                blocks,
                block_tokens,
                sampling_state,
                generates_canvases,
            )
        })
    }

    /// Returns the next input-image block to encode before the context can
    /// advance, with its encoder, or `None` when the next context call needs
    /// no encode.
    ///
    /// Walks the blocks after the accepted block cursor, whose leading
    /// products `context_features` holds: the latent block at the scheduled
    /// prompt cursor is encoded for its own extension; vision blocks are
    /// encoded in order ahead of the context prefill that writes them, while
    /// their image lies within the long-prefill threshold of the scheduled
    /// prompt cursor. The walk stops at a latent block past that cursor,
    /// which the chunk before it does not reach. A request plans nothing
    /// while its encode is in flight, so the encode's completion identifies
    /// its block by the same walk.
    pub(in crate::scheduler) fn next_context_encode(
        &self,
        id: RequestId,
    ) -> Option<(BlockCursor, MediaCall)> {
        let state = self.running.get(&id)?;
        if state.phase == Phase::IngestState {
            return None;
        }
        let cursor = self.num_scheduled_prompt_tokens(id)? as usize;
        let reach = cursor.saturating_add(self.config.chunked_prefill_size);
        let mut block = state.block_cursor();
        let mut encoded = 0;
        while let Some((image, input)) = state.input_block(block) {
            let position = image.position as usize;
            if input.encoder == MediaCall::LatentEncoding {
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
        let state = self.running.get(&id)?;
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

    /// Chooses whether a committed round-close token opens a branch or finishes the request.
    pub(in crate::scheduler) fn close_context_round(&mut self, id: RequestId, close_token: u32) {
        let (triggered, can_open_gen_branch, n_gen, max_tokens) = {
            let Some(st) = self.running.get(&id) else {
                return;
            };
            (
                st.req
                    .image_generation
                    .trigger
                    .matches_round_close(&st.round_token_ids, close_token),
                st.can_open_gen_branch(),
                st.num_generated_tokens,
                st.req.max_und_tokens,
            )
        };
        if triggered && can_open_gen_branch && n_gen < max_tokens {
            return self.begin_image(id);
        }
        self.finish(
            id,
            if n_gen >= max_tokens {
                FinishReason::MaxTokens
            } else {
                FinishReason::Eos
            },
        )
    }

    /// Builds only constraints that vary with the scheduled sampling position.
    /// Admission supplies the static whitelist and finish tokens. `projected`
    /// counts unresolved predecessors, so minimum-token and image limits apply
    /// at the successor's position without observing device results. Penalties
    /// remain owned by the worker's accepted-token state.
    ///
    /// `completing_images` counts image completions the planned call itself
    /// finishes, which the final feedback step does. Returns `None` when the
    /// request is not running or nothing needs constraining.
    pub(in crate::scheduler) fn build_token_masks(
        &self,
        id: RequestId,
        projected: usize,
        completing_images: usize,
    ) -> Option<SamplingState> {
        let n_generated = self.running.get(&id).map_or(0, |state| {
            state.num_generated_tokens.saturating_add(projected)
        });
        let projected_images_done = self.running.get(&id).map_or(0, |state| {
            state
                .num_generated_images
                .saturating_add(self.scheduled_feedback(id).map_or(0, |(_, images)| images))
                .saturating_add(completing_images)
        });
        let state = self.running.get(&id)?;
        let sampling = &state.req.sampling;
        let generated = &state.generated_token_ids;
        let mut suppress = Vec::new();
        if n_generated < sampling.min_tokens {
            suppress.extend_from_slice(&self.ctrl.eos);
        }
        for word in &sampling.bad_words_ids {
            let Some((&last, prefix)) = word.split_last() else {
                continue;
            };
            if generated.ends_with(prefix) {
                suppress.push(last);
            }
        }
        // Disallow unavailable transitions before sampling. The sampled token,
        // its scores, and any device-resident successor input must agree.
        if (!state.req.generates_images() || projected_images_done >= state.req.max_images as usize)
            && let Some(trigger) = state.req.image_generation.trigger.direct_token()
        {
            suppress.push(trigger);
        }
        suppress.sort_unstable();
        suppress.dedup();

        let transition_token_ids = (state.req.generates_images()
            && projected_images_done < state.req.max_images as usize)
            .then(|| state.req.image_generation.trigger.direct_token())
            .flatten()
            .into_iter()
            .collect();
        let sampling = SamplingState {
            allowed_token_ids: None,
            suppressed_token_ids: suppress,
            finish_token_ids: Vec::new(),
            transition_token_ids,
            force_finish: n_generated.saturating_add(1) >= state.req.max_und_tokens,
        };
        (sampling != SamplingState::default()).then_some(sampling)
    }
}
