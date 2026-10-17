//! Successor projection: queries such as `projected_coordinates`,
//! `pending_successor_code` and `can_queue_successor` project a token request's
//! accepted state through its in-flight calls, so the batching path can queue a
//! successor before its predecessor's result returns.

use super::*;
use uniserve_ipc::{CallCoordinates, ForwardMode, MediaCall, TransferMode};

/// A request's context projected through its in-flight calls
/// (`Scheduler::scheduled_context`).
pub(in crate::scheduler) struct ScheduledContext {
    /// Model position after the in-flight calls.
    pub(in crate::scheduler) logical: u32,
    /// Cache length after the in-flight calls, at their submitted maxima.
    pub(in crate::scheduler) physical: u32,
    /// The next input-image block no accepted or in-flight call writes.
    pub(in crate::scheduler) blocks: BlockCursor,
}

impl Scheduler {
    /// Returns whether a flow prefix can be scheduled now.
    ///
    /// Returns `false` only while the request holds a flow prefix that no
    /// successful denoising completion has finalized and a denoising call is
    /// in flight.
    pub(in crate::scheduler) fn flow_prefix_is_schedulable(&self, id: RequestId) -> bool {
        let prefix_is_pending = self
            .running
            .get(&id)
            .and_then(|state| state.flow_prefix.as_ref())
            .is_some_and(|prefix| !prefix.diffusion_finalized);
        !prefix_is_pending
            || !self
                .inflight
                .pending_calls
                .get(&id)
                .into_iter()
                .flatten()
                .any(|call| call.call.code == CallKind::Media(MediaCall::Denoising))
    }

    /// Counts prompt tokens accepted or present in pending forward inputs.
    pub(in crate::scheduler) fn num_scheduled_prompt_tokens(&self, id: RequestId) -> Option<u32> {
        let state = self.running.get(&id)?;
        Some(
            self.inflight
                .pending_calls
                .get(&id)
                .into_iter()
                .flatten()
                .filter(|pending| is_prompt_extend(&pending.call))
                .fold(state.num_computed_prompt_tokens, |count, pending| {
                    count.saturating_add(
                        pending.call.input_token_ids.len().min(u32::MAX as usize) as u32
                    )
                }),
        )
    }

    /// Computes only token extents needed for capacity and successor inputs.
    /// Pending work contributes its submitted maximum; accepted lengths remain
    /// request state and are never advanced by this scheduling calculation.
    pub(in crate::scheduler) fn scheduled_token_lengths(
        &self,
        id: RequestId,
    ) -> Option<(u32, u32)> {
        self.scheduled_context(id)
            .map(|context| (context.logical, context.physical))
    }

    /// Returns the input images whose every block the request's accepted and
    /// in-flight calls write.
    pub(in crate::scheduler) fn num_scheduled_images(&self, id: RequestId) -> Option<usize> {
        self.scheduled_context(id)
            .map(|context| context.blocks.image)
    }

    /// Projects the request's accepted context through its in-flight calls:
    /// logical and cache extents, and the input-image block cursor.
    pub(in crate::scheduler) fn scheduled_context(
        &self,
        id: RequestId,
    ) -> Option<ScheduledContext> {
        let state = self.running.get(&id)?;
        let (mut logical, mut physical) = (state.logical_position, state.cache_visible_len);
        let mut blocks = state.block_cursor();
        let mut feedback_index = state.feedback_encoder_index;
        for pending in self.inflight.pending_calls.get(&id).into_iter().flatten() {
            let call = &pending.call;
            match call.code {
                // A context prefill adds its prompt tokens and vision blocks.
                CallKind::Forward(ForwardMode::Prefill) if is_prompt_extend(call) => {
                    let count = call.input_token_ids.len().min(u32::MAX as usize) as u32;
                    let advance = state.context_advance(count, call.vision_inputs.len(), blocks)?;
                    logical = logical.saturating_add(advance.logical);
                    physical = physical.saturating_add(advance.prefix_cache);
                    blocks = advance.cursor;
                }
                CallKind::Forward(ForwardMode::Prefill) if consumes_image_features(call) => {
                    physical = physical.saturating_add(call.bounds.max_tokens);
                    if is_feedback_computation(call) {
                        feedback_index += 1;
                        if feedback_index == state.req.image_generation.feedback_encoders.len() {
                            logical = logical
                                .saturating_add(state.req.image_generation.num_feedback_positions);
                            feedback_index = 0;
                        }
                    } else {
                        // A latent input-image block.
                        let image = state.req.multimodal_inputs.images.get(blocks.image)?;
                        let next = state.next_block(blocks)?;
                        if next.image != blocks.image {
                            logical = logical.saturating_add(image.num_positions);
                        }
                        blocks = next;
                    }
                }
                CallKind::Forward(ForwardMode::Prefill) => physical = physical.saturating_add(1),
                CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                    logical = logical.saturating_add(1);
                    physical = physical.saturating_add(1);
                }
                CallKind::Media(MediaCall::ImageDecoding) => feedback_index = 0,
                _ => {}
            }
        }
        Some(ScheduledContext {
            logical,
            physical,
            blocks,
        })
    }

    /// The coordinates the next call of this request executes at.
    ///
    /// Each value is the request's own accepted state projected through the
    /// calls already submitted, so a rank receives them instead of chaining
    /// them from its predecessors. A submitted forward collapses the computed
    /// extent onto the visible one: it initializes exactly the tokens it makes
    /// visible. Only a verifier leaves the two apart, and its acceptance
    /// resolves before any successor is scheduled.
    pub(in crate::scheduler) fn projected_coordinates(
        &self,
        id: RequestId,
    ) -> Option<CallCoordinates> {
        let state = self.running.get(&id)?;
        let (logical_position, cache_visible_len) = self.scheduled_token_lengths(id)?;
        let forward_submitted = self
            .inflight
            .pending_calls
            .get(&id)
            .into_iter()
            .flatten()
            .any(|pending| matches!(pending.call.code, CallKind::Forward(_)));
        Some(CallCoordinates {
            logical_position,
            cache_visible_len,
            cache_computed_len: if forward_submitted {
                cache_visible_len
            } else {
                state.cache_computed_len.max(cache_visible_len)
            },
            flow_step: self.num_scheduled_denoise_steps(id)?,
        })
    }

    /// Readout rows accepted or covered by submitted canvas passes.
    ///
    /// Canvas passes cover consecutive rows from the accepted cursor in
    /// submission order, so each in-flight pass continues where the previous
    /// one ends. Returns `None` for a request that is not running or whose
    /// in-flight passes do not cover whole rows.
    pub(in crate::scheduler) fn scheduled_readout_rows(&self, id: RequestId) -> Option<usize> {
        let state = self.running.get(&id)?;
        self.inflight
            .pending_calls
            .get(&id)
            .into_iter()
            .flatten()
            .filter(|pending| pending.call.readout.is_some())
            .try_fold(state.readout_rows, |rows, pending| {
                let covered =
                    state.readout_rows_covering(rows, pending.call.input_token_ids.len())?;
                Some(rows + covered)
            })
    }

    /// Last denoising step covered by submitted intervals, without accepting them.
    pub(in crate::scheduler) fn num_scheduled_denoise_steps(&self, id: RequestId) -> Option<u32> {
        let state = self.running.get(&id)?;
        Some(self.scheduled_steps(id, &state.denoising))
    }

    /// Returns the next feedback encoder index and number of pending image completions.
    /// Only actual feature writes advance the ordered encoder inputs.
    pub(in crate::scheduler) fn scheduled_feedback(&self, id: RequestId) -> Option<(usize, usize)> {
        let state = self.running.get(&id)?;
        let mut index = state.feedback_encoder_index;
        let mut images = 0;
        for pending in self.inflight.pending_calls.get(&id).into_iter().flatten() {
            let call = &pending.call;
            if call.code == CallKind::Media(MediaCall::ImageDecoding) {
                index = 0;
            } else if consumes_image_features(call) && is_feedback_computation(call) {
                index += 1;
                if index == state.req.image_generation.feedback_encoders.len() {
                    index = 0;
                    images += 1;
                }
            }
        }
        Some((index, images))
    }

    /// Returns the cache buffer the request's next call is conditioned on: the
    /// `cache_output` of its latest in-flight call that publishes one, otherwise
    /// the accepted `image_conditioning` of the running request.
    pub(in crate::scheduler) fn cache_conditioning(
        &self,
        id: RequestId,
    ) -> Option<uniserve_ipc::BufferId> {
        self.inflight
            .pending_calls
            .get(&id)
            .and_then(|pending| pending.iter().rev().find_map(|item| item.call.cache_output))
            .or_else(|| {
                self.running
                    .get(&id)
                    .and_then(|request| request.image_conditioning)
            })
    }

    /// Returns the product `select` picks from the request's most recently
    /// submitted in-flight call that has one.
    ///
    /// A successor refers to the producer's registered output generation
    /// directly, before the producer completes.
    pub(in crate::scheduler) fn pending_output(
        &self,
        id: RequestId,
        select: impl for<'a> Fn(&'a Call) -> Option<&'a TensorRef>,
    ) -> Option<&TensorRef> {
        self.inflight
            .pending_calls
            .get(&id)?
            .iter()
            .rev()
            .find_map(|pending| select(&pending.call))
    }

    /// Selects the next computation from the last pending input, or from accepted
    /// request progress when no computation remains. No request state is replayed.
    pub(in crate::scheduler) fn next_generation_phase(&self, id: RequestId) -> Option<Phase> {
        let state = self.running.get(&id)?;
        let pending = self
            .inflight
            .pending_calls
            .get(&id)
            .and_then(|queue| queue.back());
        let phase = match pending.map(|pending| &pending.call) {
            Some(call) => match call.code {
                CallKind::Forward(ForwardMode::Prefill) if is_prompt_extend(call) => Phase::Prefill,
                CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                    Phase::Prefill
                }
                // Steps queued behind the accepted one that stopped the block
                // are no-ops; the block's commit follows them. A step queued
                // behind the commit itself starts the next block.
                CallKind::Forward(ForwardMode::TokenDenoising) if call.canvas.is_some() => {
                    let behind_commit = self.inflight.pending_calls.get(&id).is_some_and(|calls| {
                        calls
                            .iter()
                            .any(|inflight| is_prompt_extend(&inflight.call))
                    });
                    if state.phase == Phase::CommitCanvas && !behind_commit {
                        Phase::CommitCanvas
                    } else {
                        Phase::Canvas
                    }
                }
                CallKind::Forward(ForwardMode::TokenDenoising) => Phase::Readout,
                CallKind::Forward(ForwardMode::Prefill) if consumes_image_features(call) => {
                    if is_feedback_computation(call) {
                        if self.scheduled_feedback(id)?.0 == 0 {
                            Phase::DecodeUnd
                        } else {
                            Phase::FeedbackEncode
                        }
                    } else {
                        // A latent input-image block: context ingestion
                        // continues after it.
                        Phase::Prefill
                    }
                }
                CallKind::Forward(ForwardMode::Prefill) => Phase::PublishCache,
                CallKind::Transfer(TransferMode::CacheExport) => Phase::PrepareGen,
                CallKind::Media(MediaCall::LatentPreparation)
                | CallKind::Media(MediaCall::Denoising) => Phase::DenoiseGen,
                CallKind::Media(MediaCall::ImageDecoding) => Phase::FeedbackEncode,
                CallKind::Media(MediaCall::VisionEncoding)
                | CallKind::Media(MediaCall::LatentEncoding)
                    if is_feedback_computation(call) =>
                {
                    Phase::FeedbackState
                }
                _ => state.phase,
            },
            None => state.phase,
        };
        Some(match phase {
            // A complete prompt conditions a readout's canvases or a
            // canvas-generating request's blocks; any other request decodes
            // from it. A block commit also projects here, since it extends
            // the context the same way, and the next block follows it.
            Phase::Prefill
                if self.num_scheduled_prompt_tokens(id)?
                    >= state.req.prompt_token_ids.len() as u32
                    && self.num_scheduled_images(id)?
                        >= state.req.multimodal_inputs.images.len() =>
            {
                if state.req.is_readout() {
                    Phase::Readout
                } else if state.req.is_canvas_generation() {
                    Phase::Canvas
                } else {
                    Phase::DecodeUnd
                }
            }
            Phase::DenoiseGen
                if self.num_scheduled_denoise_steps(id)? >= state.denoising.steps() =>
            {
                Phase::CommitGen
            }
            phase => phase,
        })
    }

    /// Returns the latest accepted state call identity, or `None` when the
    /// request is not running or its current epoch is not yet registered
    /// with its workers.
    pub(in crate::scheduler) fn state_predecessor(&self, id: RequestId) -> Option<CallId> {
        let state = self.running.get(&id)?;
        state.worker_registered.then_some(())?;
        Some(state.last_state_call_id)
    }

    /// Returns the call that precedes the next state advancement: the latest
    /// in-flight state-advancing call, otherwise `state_predecessor`.
    ///
    /// Returns `None` when the request has no call in flight, and when it
    /// falls back to a `state_predecessor` that is `None`.
    pub(in crate::scheduler) fn execution_predecessor(&self, id: RequestId) -> Option<CallId> {
        let call = self
            .inflight
            .pending_calls
            .get(&id)?
            .iter()
            .rev()
            .find(|inflight| inflight.call.advances_state())
            .map(|inflight| &inflight.call);
        let Some(call) = call else {
            return self.state_predecessor(id);
        };
        // Accepted lengths are published in device request state; the successor
        // needs the producing computation identity while its host report is pending.
        Some(call.call_id)
    }

    /// Applies completed worker commands to scheduler-owned allocation state.
    ///
    /// A `Free` returns its buffer's allocation to the buffer pool when a
    /// pending free, the live request of the same epoch, or its retiring
    /// record still holds it. A `Finish` releases everything its request
    /// parked in `retiring_requests`; a `Finish` that matches no retiring
    /// request of the same epoch latches engine-fatal. `Start` commands are
    /// ignored.
    pub(in crate::scheduler) fn acknowledge_commands(&mut self, commands: &[BatchCommand]) {
        for command in commands {
            if let BatchCommand::Free { buffer } = command {
                // Buffer ownership may sit with an already queued free, a live
                // request, or a request awaiting its close acknowledgement.
                let id = buffer.owner.request_id;
                let allocation = self
                    .storage
                    .pending_buffer_frees
                    .remove(buffer)
                    .or_else(|| {
                        self.running
                            .get_mut(&id)
                            .filter(|state| state.request_epoch == buffer.owner.request_epoch)
                            .and_then(|state| state.allocations_mut()?.take_buffer(*buffer))
                    })
                    .or_else(|| {
                        self.retiring_requests
                            .get_mut(&id)
                            .filter(|state| state.request_key == buffer.owner)
                            .and_then(|state| state.buffers.remove(buffer))
                    });
                if let Some(allocation) = allocation {
                    self.storage.buffer_pool.free(allocation);
                }
            } else if let BatchCommand::Finish { request_key, .. } = command {
                let id = request_key.request_id;
                let retiring = match self.retiring_requests.entry(id) {
                    std::collections::hash_map::Entry::Occupied(entry)
                        if entry.get().request_key == *request_key =>
                    {
                        entry.remove()
                    }
                    _ => {
                        tracing::error!(
                            request_id = id.0,
                            request_epoch = request_key.request_epoch,
                            "close acknowledgement does not match a retiring request"
                        );
                        self.fatal = true;
                        continue;
                    }
                };
                for buffer in retiring.buffers.into_values() {
                    self.storage.buffer_pool.free(buffer);
                }
                for allocation in retiring.allocations {
                    allocation.free(&mut self.storage);
                }
                if let Some(media) = retiring.media_allocations
                    && let Err(unknown) = media.free(&mut self.storage)
                {
                    self.invariant_broken(&unknown.to_string());
                }
            }
        }
    }

    /// Whether a bounded successor can execute from its predecessor's device
    /// results. The submitted input must establish its required lengths and predicate.
    pub(in crate::scheduler) fn can_queue_successor(
        &self,
        id: RequestId,
        target: CallKind,
    ) -> bool {
        // Successor projection requires both live runtime state and an unresolved
        // predecessor whose device products carry the accepted execution result.
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        let Some(queue) = self
            .inflight
            .pending_calls
            .get(&id)
            .filter(|queue| !queue.is_empty())
        else {
            return false;
        };

        // Bound speculation by worker capacity and exclude requests whose
        // terminal or relay state makes the projection unsafe.
        if queue.len() >= self.info.max_unresolved_calls as usize
            || state.terminal_intent.is_terminal()
            || self.inflight.pending_finishes.contains_key(&id)
            || !Self::device_token_relay_eligible(state)
        {
            return false;
        }
        let Some(predecessor) = queue.back() else {
            return false;
        };

        // A cache export is never queued behind an unresolved call, and a
        // request with a `RoundCloseThenSuffix` image trigger queues no
        // successor at all.
        if target == CallKind::Transfer(TransferMode::CacheExport) {
            return false;
        }
        if matches!(
            state.req.image_generation.trigger,
            uniserve_core::ImageTrigger::RoundCloseThenSuffix { .. }
        ) {
            return false;
        }

        // Decode successors support both feedback continuation and ordinary
        // prompt/decode pipelining, using each predecessor's actual inputs.
        if target == CallKind::Forward(ForwardMode::Decode) {
            let feedback_continuation = consumes_image_features(&predecessor.call)
                && is_feedback_computation(&predecessor.call)
                && self
                    .scheduled_feedback(id)
                    .is_some_and(|(index, _)| index == 0)
                && state.req.image_generation.sample_feedback_continuation
                && predecessor.call.token_output.is_some();
            if feedback_continuation {
                return self
                    .pending_successor_code(id)
                    .is_some_and(|variant| variant == CallKind::Forward(ForwardMode::Decode))
                    && state.num_generated_tokens.saturating_add(1) < state.req.max_und_tokens;
            }

            if !matches!(state.phase, Phase::Prefill | Phase::DecodeUnd)
                || (state.phase == Phase::Prefill && state.starts_gen_after_context())
                || queue.iter().any(|call| {
                    !matches!(
                        call.call.code,
                        CallKind::Forward(ForwardMode::Prefill)
                            | CallKind::Forward(ForwardMode::Decode)
                    )
                })
            {
                return false;
            }
            // Ordinary pipelining needs the whole prompt and every context
            // image block submitted, and counts each in-flight call as one
            // token toward `max_und_tokens`.
            return self
                .num_scheduled_prompt_tokens(id)
                .is_some_and(|count| count as usize >= state.req.prompt_token_ids.len())
                && self
                    .num_scheduled_images(id)
                    .is_some_and(|count| count >= state.req.multimodal_inputs.images.len())
                && state.num_generated_tokens.saturating_add(queue.len())
                    < state.req.max_und_tokens;
        }

        // Other successor kinds require an exact projected physical variant
        // match and, unless they follow unconditionally, a completion
        // predicate.
        (predecessor.call.completion_output.is_some() || self.follows_unconditionally(id, target))
            && self
                .pending_successor_code(id)
                .is_some_and(|variant| variant == target)
    }

    /// Returns whether a successor of kind `target`, queued behind the
    /// request's last in-flight call, runs whatever that call's device result
    /// is, and so carries no predicate.
    ///
    /// Two successors follow unconditionally:
    /// - The canvas pass that follows the prefill completing a context (a
    ///   readout's pass or a block's first step), since the prefill cannot
    ///   stop the request. The pass reads the prefix cache the prefill writes, and only
    ///   the worker's device order puts the write first: forwards run in
    ///   submission order on one stream, and a forward on an execution lane
    ///   forks from and joins back into the device's control stream
    ///   (the rank runtime's text runner).
    /// - A stopped block's commit queued behind the block's steps still in
    ///   flight. The host has accepted the step that stopped the block, so
    ///   every step behind it is a no-op whose completion is false, and the
    ///   commit runs regardless. The commit writes the block's tokens to cache
    ///   positions `[P, P + canvas)`, where `P` is the context length each
    ///   step of the block reads; a canvas step reads cache below `P` and writes
    ///   none, so the commit's writes never overlap what those steps read.
    ///   Each rank also runs the commit after them in submission order on
    ///   its device stream, under data and expert parallelism alike.
    pub(in crate::scheduler) fn follows_unconditionally(
        &self,
        id: RequestId,
        target: CallKind,
    ) -> bool {
        let Some(predecessor) = self
            .inflight
            .pending_calls
            .get(&id)
            .and_then(|queue| queue.back())
        else {
            return false;
        };
        match target {
            CallKind::Forward(ForwardMode::TokenDenoising) => is_prompt_extend(&predecessor.call),
            CallKind::Forward(ForwardMode::Prefill) => self.running.get(&id).is_some_and(|state| {
                state.phase == Phase::CommitCanvas
                    && predecessor
                        .call
                        .canvas
                        .is_some_and(|step| step.block == state.canvas_block)
            }),
            _ => false,
        }
    }

    /// Determines the next pipelined computation from its submitted predecessor.
    ///
    /// Returns `None`, among other cases, when the request is not running or
    /// has no call in flight, has not yet submitted its whole prompt or every
    /// block of its context images, cannot chain from its last in-flight
    /// call (an image decode whose request does not feed the device image
    /// product back, or a completed feedback round whose request does not
    /// sample a feedback continuation), or projects to a latent ingest
    /// phase.
    pub(in crate::scheduler) fn pending_successor_code(&self, id: RequestId) -> Option<CallKind> {
        if !self.inflight.has_pending_calls(id) {
            return None;
        }
        let state = self.running.get(&id)?;
        if self.num_scheduled_prompt_tokens(id)? < state.req.prompt_token_ids.len() as u32
            || self.num_scheduled_images(id)? < state.req.multimodal_inputs.images.len()
        {
            return None;
        }
        let last = &self.inflight.pending_calls.get(&id)?.back()?.call;
        let chainable = if last.code == CallKind::Media(MediaCall::ImageDecoding) {
            state.req.feeds_back_images()
                && state.req.image_generation.feedback_source
                    == Some(uniserve_core::FeedbackSource::DeviceProduct)
        } else if consumes_image_features(last)
            && is_feedback_computation(last)
            && self.scheduled_feedback(id)?.0 == 0
        {
            state.req.image_generation.sample_feedback_continuation
        } else {
            true
        };
        if !chainable {
            return None;
        }
        Some(match self.next_generation_phase(id)? {
            Phase::Prefill | Phase::DecodeUnd => CallKind::Forward(ForwardMode::Decode),
            Phase::CloseCache | Phase::FeedbackState => CallKind::Forward(ForwardMode::Prefill),
            Phase::PublishCache => CallKind::Transfer(TransferMode::CacheExport),
            Phase::PrepareGen => CallKind::Media(MediaCall::LatentPreparation),
            Phase::DenoiseGen => CallKind::Media(MediaCall::Denoising),
            Phase::CommitGen => CallKind::Media(MediaCall::ImageDecoding),
            Phase::Readout | Phase::Canvas => CallKind::Forward(ForwardMode::TokenDenoising),
            Phase::CommitCanvas => CallKind::Forward(ForwardMode::Prefill),
            Phase::FeedbackEncode => {
                let feedback = &state.req.image_generation;
                feedback.feedback_source.as_ref()?;
                match feedback
                    .feedback_encoders
                    .get(self.scheduled_feedback(id)?.0)?
                    .encoder
                {
                    ImageIngestStep::VaeEncode => CallKind::Media(MediaCall::LatentEncoding),
                    ImageIngestStep::VitEncode => CallKind::Media(MediaCall::VisionEncoding),
                }
            }
            Phase::IngestState => return None,
        })
    }

    /// Returns whether a successor can consume device-selected tokens before host observation.
    pub(in crate::scheduler) fn device_token_relay_eligible(state: &RequestState) -> bool {
        // A successor may consume the parent's device-selected point before host
        // observation whenever its own sampling state is device-representable
        // from registered coordinates and device products alone. Two request
        // shapes keep the request host-paced:
        // - An image-generating request whose image branch opens on anything
        //   but one direct token. Only a direct trigger is among the device
        //   finish tokens (`finish_token_ids`).
        // - Multi-token bad words, because their next mask depends on the
        //   not-yet-observed suffix.
        // Stop strings do not keep a request host-paced: the frontend decoder
        // matches them after resolution, and the scheduler tracks its pending
        // decisions in `decoder_boundaries`.
        let sampling = &state.req.sampling;
        (!state.req.generates_images()
            || state.req.image_generation.trigger.direct_token().is_some())
            && sampling.bad_words_ids.iter().all(|word| word.len() == 1)
    }

    /// Returns whether the latest host-resolved token may remain the exact device input
    /// to the next decode. CPU stop and EOS decisions are complete before this
    /// check; a continuing request therefore names the same sampled token.
    pub(in crate::scheduler) fn can_reuse_resolved_token_product(&self, id: RequestId) -> bool {
        self.running.get(&id).is_some_and(|state| {
            state.worker_registered
                && state.num_generated_tokens > 0
                && state.latest_token.as_ref().is_some()
                && state.is_replayable_text()
                && state.phase == Phase::DecodeUnd
                && !state.round_closing
                && (!state.req.generates_images()
                    || state.req.image_generation.trigger.direct_token().is_some())
        })
    }

    /// Returns whether the next call can be scheduled.
    pub(in crate::scheduler) fn can_schedule_next(&self, id: RequestId) -> bool {
        !self.inflight.pending_finishes.contains_key(&id)
            && self.output_window_ready(id)
            && self.running.get(&id).is_some_and(|state| {
                !state.speculative_chain_invalidated
                    && state.output.decoder_boundaries.len()
                        < self.info.max_unresolved_calls.max(1) as usize
                    && self.peek_next_call_variant(id).is_none_or(|kind| {
                        self.placement
                            .worker_target(
                                self.executor.as_ref(),
                                &self.info,
                                RequestKey::new(self.engine_id, id, state.request_epoch),
                                kind,
                            )
                            .is_some()
                    })
            })
            && (!self.inflight.has_pending_calls(id)
                || self
                    .pending_successor_code(id)
                    .is_some_and(|target| self.can_queue_successor(id, target)))
    }

    /// Returns whether output capacity can cover every unresolved token-producing call.
    ///
    /// The request's event journal must hold, besides `OUTPUT_TERMINAL_RESERVE`
    /// events kept for its terminal output, the `call_output_bound` of every
    /// in-flight call plus the `next_output_bound` of the next one. Returns
    /// `false` for a request that is not running or whose journal is closed.
    pub(in crate::scheduler) fn output_window_ready(&self, id: RequestId) -> bool {
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        if state.output.events.is_closed() {
            return false;
        }
        let available = state.output.events.available_capacity();
        let reserved = self
            .inflight
            .pending_calls
            .get(&id)
            .into_iter()
            .flatten()
            .map(|pending| call_output_bound(pending.call.code, &pending.call.bounds))
            .sum::<usize>();
        available
            .saturating_sub(reserved)
            .saturating_sub(OUTPUT_TERMINAL_RESERVE)
            >= self.next_output_bound(id)
    }

    /// Returns the output-size bound, in events, for the next call.
    ///
    /// Mirrors the per-kind counts of `call_output_bound`, which charges a
    /// call once it is in flight, for the kinds `peek_next_call_variant`
    /// predicts; a denoising call is budgeted at `denoise_step_burst` steps.
    pub(in crate::scheduler) fn next_output_bound(&self, id: RequestId) -> usize {
        match self.peek_next_call_variant(id) {
            Some(
                CallKind::Forward(ForwardMode::Prefill) | CallKind::Forward(ForwardMode::Decode),
            ) => 4,
            Some(CallKind::Media(MediaCall::Denoising)) => {
                usize::from(self.denoise_step_burst).saturating_add(2)
            }
            Some(CallKind::Media(MediaCall::ImageDecoding)) => 3,
            Some(_) | None => 2,
        }
    }
}
