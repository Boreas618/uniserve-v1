//! Successor projection: queries such as `projected_coordinates`,
//! `pending_successor_code` and `can_queue_successor` project a token request's
//! accepted state through its in-flight calls, so the batching path can queue a
//! successor before its predecessor's result returns.

use super::*;
use ipc::{CallCoordinates, ForwardMode};

/// A request's context projected through its in-flight calls
/// (`Scheduler::scheduled_context`).
pub(crate) struct ScheduledContext {
    /// Model position after the in-flight calls.
    pub(crate) logical: u32,
    /// Cache length after the in-flight calls, at their submitted maxima.
    pub(crate) physical: u32,
    /// The next input-image block no accepted or in-flight call writes.
    pub(crate) blocks: BlockCursor,
}

impl Scheduler {
    /// Counts prompt tokens accepted or present in pending forward inputs.
    pub(crate) fn num_scheduled_prompt_tokens(&self, id: RequestId) -> Option<u32> {
        let state = self.requests.tokens(&id)?;
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
    pub(crate) fn scheduled_token_lengths(&self, id: RequestId) -> Option<(u32, u32)> {
        self.scheduled_context(id)
            .map(|context| (context.logical, context.physical))
    }

    /// Returns the input images whose every block the request's accepted and
    /// in-flight calls write.
    pub(crate) fn num_scheduled_images(&self, id: RequestId) -> Option<usize> {
        self.scheduled_context(id)
            .map(|context| context.blocks.image)
    }

    /// Projects the request's accepted context through its in-flight calls:
    /// logical and cache extents, and the input-image block cursor.
    pub(crate) fn scheduled_context(&self, id: RequestId) -> Option<ScheduledContext> {
        let state = self.requests.tokens(&id)?;
        let (mut logical, mut physical) = (state.logical_position, state.cache_visible_len);
        let mut blocks = state.block_cursor();
        for pending in self.inflight.pending_calls.get(&id).into_iter().flatten() {
            let call = &pending.call;
            match call.kind {
                // A context prefill adds its prompt tokens and vision blocks.
                CallKind::Forward(ForwardMode::Prefill) if is_prompt_extend(call) => {
                    let count = call.input_token_ids.len().min(u32::MAX as usize) as u32;
                    let advance = state.context_advance(count, call.vision_inputs.len(), blocks)?;
                    logical = logical.saturating_add(advance.logical);
                    physical = physical.saturating_add(advance.prefix_cache);
                    blocks = advance.cursor;
                }
                // A latent input-image block.
                CallKind::Forward(ForwardMode::Prefill) if consumes_image_features(call) => {
                    physical = physical.saturating_add(call.bounds.max_tokens);
                    let image = state.req.multimodal_inputs.images.get(blocks.image)?;
                    let next = state.next_block(blocks)?;
                    if next.image != blocks.image {
                        logical = logical.saturating_add(image.num_positions);
                    }
                    blocks = next;
                }
                CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                    logical = logical.saturating_add(1);
                    physical = physical.saturating_add(1);
                }
                // A draft refill evaluates tokens the cache holds already.
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
    pub(crate) fn projected_coordinates(&self, id: RequestId) -> Option<CallCoordinates> {
        let state = self.requests.tokens(&id)?;
        let (logical_position, cache_visible_len) = self.scheduled_token_lengths(id)?;
        let forward_submitted = self
            .inflight
            .pending_calls
            .get(&id)
            .into_iter()
            .flatten()
            .any(|pending| matches!(pending.call.kind, CallKind::Forward(_)));
        Some(CallCoordinates {
            logical_position,
            cache_visible_len,
            cache_computed_len: if forward_submitted {
                cache_visible_len
            } else {
                state.cache_computed_len.max(cache_visible_len)
            },
        })
    }

    /// Selects the next call from the last pending input, or from accepted
    /// request progress when no call remains. No request state is replayed.
    pub(crate) fn next_generation_phase(&self, id: RequestId) -> Option<Phase> {
        let state = self.requests.tokens(&id)?;
        let pending = self
            .inflight
            .pending_calls
            .get(&id)
            .and_then(|queue| queue.back());
        let phase = match pending.map(|pending| &pending.call) {
            Some(call) => match call.kind {
                // A draft refill leaves the request's phase as it is.
                CallKind::Forward(ForwardMode::Prefill) if call.draft.refill.is_some() => {
                    state.phase
                }
                CallKind::Forward(ForwardMode::Prefill) if is_prompt_extend(call) => Phase::Prefill,
                CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                    Phase::Prefill
                }
                // Steps queued behind the accepted one that stopped the block
                // are no-ops; the block's commit follows them. A step queued
                // behind the commit itself starts the next block.
                CallKind::Forward(ForwardMode::TokenDenoising) => {
                    let behind_commit = self.inflight.pending_calls.get(&id).is_some_and(|calls| {
                        calls
                            .iter()
                            .any(|inflight| is_prompt_extend(&inflight.call))
                    });
                    if state.phase == Phase::CommitBlock && !behind_commit {
                        Phase::CommitBlock
                    } else {
                        Phase::Block
                    }
                }
                // A latent input-image block: context ingestion continues
                // after it.
                CallKind::Forward(ForwardMode::Prefill) if consumes_image_features(call) => {
                    Phase::Prefill
                }
                _ => state.phase,
            },
            None => state.phase,
        };
        Some(match phase {
            // A complete prompt conditions a block-generating request's
            // blocks; any other request decodes from it. A block commit also
            // projects here, since it extends the context the same way, and
            // the next block follows it.
            Phase::Prefill
                if self.num_scheduled_prompt_tokens(id)?
                    >= state.req.prompt_token_ids.len() as u32
                    && self.num_scheduled_images(id)?
                        >= state.req.multimodal_inputs.images.len() =>
            {
                if crate::BLOCK_SAMPLING.is_some() {
                    Phase::Block
                } else if self.decodes_elsewhere(id) {
                    Phase::TransferState
                } else {
                    Phase::DecodeUnd
                }
            }
            phase => phase,
        })
    }

    /// Returns the latest accepted state call identity, or `None` when the
    /// request is not running or its current epoch is not yet registered
    /// with its workers.
    pub(crate) fn state_predecessor(&self, id: RequestId) -> Option<CallId> {
        let state = self.requests.tokens(&id)?;
        state.worker_registered.then_some(())?;
        Some(state.last_state_call_id)
    }

    /// Returns the call that precedes the next state advancement: the latest
    /// in-flight state-advancing call, otherwise `state_predecessor`.
    ///
    /// Returns `None` when the request has no call in flight, and when it
    /// falls back to a `state_predecessor` that is `None`.
    pub(crate) fn execution_predecessor(&self, id: RequestId) -> Option<CallId> {
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
        // needs the producing call identity while its host report is pending.
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
    pub(crate) fn acknowledge_commands(&mut self, commands: &[BatchCommand]) {
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
                        self.requests
                            .tokens_mut(&id)
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
                        self.latch_fatal();
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
    pub(crate) fn can_queue_successor(&self, id: RequestId, target: CallKind) -> bool {
        // Successor projection requires both live runtime state and an unresolved
        // predecessor whose device products carry the accepted execution result.
        let Some(state) = self.requests.tokens(&id) else {
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
            || self.requests.finished(&id)
            || self.inflight.pending_finishes.contains_key(&id)
            || !Self::device_token_relay_eligible(state)
        {
            return false;
        }
        let Some(predecessor) = queue.back() else {
            return false;
        };

        // Decode successors pipeline behind the prompt and the decodes,
        // using each predecessor's actual inputs.
        if target == CallKind::Forward(ForwardMode::Decode) {
            if !matches!(state.phase, Phase::Prefill | Phase::DecodeUnd)
                || queue.iter().any(|call| {
                    !matches!(
                        call.call.kind,
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
    /// - The block pass that follows the prefill completing a context (a
    ///   block's first step), since the prefill cannot stop the request.
    ///   The pass reads the prefix cache the prefill writes, and only
    ///   the worker's device order puts the write first: the worker runs the
    ///   forwards over the prefix cache in submission order on the device's
    ///   primary stream, and work on another stream orders behind their
    ///   batches (the retired worker's page worker).
    /// - A stopped block's commit queued behind the block's steps still in
    ///   flight. The host has accepted the step that stopped the block, so
    ///   every step behind it is a no-op whose completion is false, and the
    ///   commit runs regardless. The commit writes the block's tokens to cache
    ///   positions `[P, P + block)`, where `P` is the context length each
    ///   step of the block reads; a block step reads cache below `P` and writes
    ///   none, so the commit's writes never overlap what those steps read.
    ///   Each rank also runs the commit after them in submission order on
    ///   its device stream, under data and expert parallelism alike.
    pub(crate) fn follows_unconditionally(&self, id: RequestId, target: CallKind) -> bool {
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
            CallKind::Forward(ForwardMode::Prefill) => {
                self.requests.tokens(&id).is_some_and(|state| {
                    state.phase == Phase::CommitBlock
                        && predecessor
                            .call
                            .block_step
                            .is_some_and(|step| step.block == state.block_index)
                })
            }
            _ => false,
        }
    }

    /// Determines the next pipelined call from its submitted predecessor.
    ///
    /// Returns `None`, among other cases, when the request is not running or
    /// has no call in flight, has not yet submitted its whole prompt or every
    /// block of its context images, or projects to a latent ingest phase.
    pub(crate) fn pending_successor_code(&self, id: RequestId) -> Option<CallKind> {
        if !self.inflight.has_pending_calls(id) {
            return None;
        }
        let state = self.requests.tokens(&id)?;
        if self.num_scheduled_prompt_tokens(id)? < state.req.prompt_token_ids.len() as u32
            || self.num_scheduled_images(id)? < state.req.multimodal_inputs.images.len()
        {
            return None;
        }
        self.inflight.pending_calls.get(&id)?.back()?;
        Some(match self.next_generation_phase(id)? {
            Phase::Prefill | Phase::DecodeUnd => CallKind::Forward(ForwardMode::Decode),
            // The transfer copies the request's cache with no call.
            Phase::TransferState => return None,
            Phase::Block => CallKind::Forward(ForwardMode::TokenDenoising),
            Phase::CommitBlock => CallKind::Forward(ForwardMode::Prefill),
            Phase::IngestState => return None,
        })
    }

    /// Returns whether a successor can consume device-selected tokens before host observation.
    pub(crate) fn device_token_relay_eligible(state: &TokenState) -> bool {
        // A successor may consume the parent's device-selected point before host
        // observation whenever its own sampling state is device-representable
        // from registered coordinates and device products alone. Multi-token
        // bad words keep the request host-paced, because their next mask
        // depends on the not-yet-observed suffix. Stop strings do not: the
        // frontend decoder matches them after resolution, and the scheduler
        // tracks its pending decisions in `decoder_boundaries`.
        let sampling = &state.req.sampling;
        sampling.bad_words_ids.iter().all(|word| word.len() == 1)
    }

    /// Returns whether the latest host-resolved token may remain the exact device input
    /// to the next decode. CPU stop and EOS decisions are complete before this
    /// check; a continuing request therefore names the same sampled token.
    pub(crate) fn can_reuse_resolved_token_product(&self, id: RequestId) -> bool {
        self.requests.tokens(&id).is_some_and(|state| {
            state.worker_registered
                && state.num_generated_tokens > 0
                && state.latest_token.as_ref().is_some()
                && state.is_replayable_text()
                && state.phase == Phase::DecodeUnd
        })
    }

    /// Whether the request's prompt was prefilled on a group that does not
    /// decode (`WorkerGroupRole::Prefill`): it decodes on a decode group,
    /// to which `Phase::TransferState` moves its KV cache.
    pub(crate) fn decodes_elsewhere(&self, id: RequestId) -> bool {
        let Some(state) = self.requests.tokens(&id) else {
            return false;
        };
        let key = RequestKey::new(self.engine_id, id, state.request_epoch);
        self.dispatch
            .affinity
            .get(&(key, DEFAULT_BLOCK.to_owned()))
            .is_some_and(|owner| self.dispatch.prefill_groups.contains(owner))
    }

    /// Returns whether request `id` may take its next call.
    ///
    /// No request does while a finish of it is in flight. A video request
    /// has no token output window to fill and no decoder boundaries: its
    /// graph names the calls it has ready, and assembly checks its admission
    /// in flight and the stage and host thread pools it occupies when it
    /// builds one; its output is one event per call, so there is no window to
    /// wait for either. A token request also waits for room in its output
    /// window, for the frontend decoder's decisions, for a worker that
    /// serves its next call, and for its calls in flight unless a successor
    /// may be queued behind them.
    pub(crate) fn can_schedule_next(&self, id: RequestId) -> bool {
        if self.inflight.pending_finishes.contains_key(&id) {
            return false;
        }
        if self.requests.video(&id).is_some() {
            return true;
        }
        self.output_window_ready(id)
            && self.requests.tokens(&id).is_some_and(|state| {
                !state.speculative_chain_invalidated
                    && state.output.decoder_boundaries.len()
                        < self.info.max_unresolved_calls.max(1) as usize
                    && self.token_next_call(id).is_none_or(|kind| {
                        self.dispatch
                            .worker_target(
                                &self.worker,
                                &self.info,
                                RequestKey::new(self.engine_id, id, state.request_epoch),
                                kind,
                                &dispatch::Choice::Existing,
                            )
                            .is_some()
                    })
            })
            && (!self.inflight.has_pending_calls(id)
                || (!self.speculation_waits(id)
                    && self
                        .pending_successor_code(id)
                        .is_some_and(|target| self.can_queue_successor(id, target))))
    }

    /// Whether a speculating request's next call waits for its calls in
    /// flight: a verification carries the drafts and starts at the token its
    /// predecessor's result states, and nothing follows a draft refill
    /// until its result.
    fn speculation_waits(&self, id: RequestId) -> bool {
        if !self.speculates() {
            return false;
        }
        let refilling = self
            .inflight
            .pending_calls
            .get(&id)
            .into_iter()
            .flatten()
            .any(|pending| pending.call.draft.refill.is_some());
        refilling
            || self.pending_successor_code(id).is_some_and(|kind| {
                matches!(
                    kind,
                    CallKind::Forward(ForwardMode::Decode | ForwardMode::Verify)
                )
            })
    }

    /// Returns whether output capacity can cover every unresolved token-producing call.
    ///
    /// The request's event journal must hold, besides `OUTPUT_TERMINAL_RESERVE`
    /// events kept for its terminal output, the `output_bound` of every
    /// in-flight call plus that of the call it schedules next. Returns `false`
    /// for a request that is not running or whose journal is closed.
    pub(crate) fn output_window_ready(&self, id: RequestId) -> bool {
        let Some(state) = self.requests.tokens(&id) else {
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
            .map(|pending| output_bound(pending.call.kind))
            .sum::<usize>();
        let next = self
            .token_next_call(id)
            .map_or(DEFAULT_OUTPUT_EVENTS, output_bound);
        available
            .saturating_sub(reserved)
            .saturating_sub(OUTPUT_TERMINAL_RESERVE)
            >= next
    }
}
