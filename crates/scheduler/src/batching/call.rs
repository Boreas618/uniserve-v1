//! Registering a selected call: its device predicate, the
//! prefix-cache and forward-row inputs derived from the lengths its
//! in-flight predecessors project, its outputs, and the cache tables a request
//! grows and activates for it.

use super::*;
use ipc::ForwardMode;

use crate::kv_cache::Checkpoint;

impl Scheduler {
    /// Registers one selected call with its allocated physical inputs and outputs.
    ///
    /// Sets the call's device predicate, derives its prefix cache and
    /// forward-row inputs from the lengths its in-flight predecessors project,
    /// mints product identities, moves `reserved_buffers` into the request's
    /// allocations, selects the executing worker, registers the call in
    /// flight, and appends it with its `RequestDispatch` to `batch`.
    /// `admitted` marks a call whose batch carries the request's `Start`, so
    /// its complete cache tables are sent.
    ///
    /// Returns `None` when a scheduler invariant fails, including an unsafe
    /// projected successor; most such paths latch engine-fatal first.
    pub(super) fn prepare_generation_call(
        &mut self,
        mut call: Call,
        reserved_buffers: Vec<BufferSpan>,
        admitted: bool,
        planned_us: u64,
        batch: &mut ExecutionBatch,
        layouts: &mut Vec<columns::CallLayout>,
    ) -> Option<()> {
        let request_id = call.request_key.request_id;

        // Planning may queue successors speculatively, but registration still
        // requires the request epoch and exact resident predecessor.
        let Some((request_epoch, latest_token)) =
            self.requests.tokens(&request_id).and_then(|state| {
                state
                    .worker_registered
                    .then(|| (state.request_epoch, state.latest_token.clone()))
            })
        else {
            tracing::error!(request_id = request_id.0, "planned call lost its request");
            self.latch_fatal();
            return None;
        };
        if self.inflight.has_pending_calls(request_id)
            && !self.can_queue_successor(request_id, call.kind)
        {
            tracing::error!(
                request_id = request_id.0,
                "scheduler attempted to issue an unsafe projected successor"
            );
            self.latch_fatal();
            return None;
        }

        let request_key = RequestKey::new(self.engine_id, request_id, request_epoch);
        // A device successor consumes the token or completion tensor of
        // either its in-flight predecessor or the latest resolved call as its
        // predicate: a decode follows the predecessor's sampled token, any
        // other successor its completion product unless it follows
        // unconditionally (`follows_unconditionally`). The first call and
        // host-observed transitions use the latest accepted call identity and
        // carry no predicate.
        // A verification carries its current token among its host tokens,
        // whose draft pass reads them, so it never continues from a
        // device-resident token.
        let projected_successor = self.inflight.has_pending_calls(request_id);
        let reusable_device_token = if !projected_successor
            && call.kind != CallKind::Forward(ForwardMode::Verify)
            && self.can_reuse_resolved_token_product(request_id)
        {
            latest_token
        } else {
            None
        };
        let predicate = if projected_successor {
            if self.execution_predecessor(request_id).is_none() {
                tracing::error!(
                    request_id = request_id.0,
                    "projected successor has no execution predecessor"
                );
                self.latch_fatal();
                return None;
            }
            let Some(predecessor) = self
                .inflight
                .pending_calls
                .get(&request_id)
                .and_then(|queue| queue.back())
                .map(|call| &call.call)
            else {
                tracing::error!(
                    request_id = request_id.0,
                    "projected successor lost its predecessor call"
                );
                self.latch_fatal();
                return None;
            };
            if call.kind == CallKind::Forward(ForwardMode::Decode) {
                predecessor.token_output.clone()
            } else if self.follows_unconditionally(request_id, call.kind) {
                None
            } else {
                predecessor.completion_output.clone()
            }
        } else if let Some(parent) = reusable_device_token {
            Some(parent)
        } else {
            if self.state_predecessor(request_id).is_none() {
                self.latch_fatal();
                return None;
            }
            None
        };

        call.predicate = predicate;

        // Freeze physical inputs before registering this call as pending.
        // Accepted progress may stop on cancellation while these inputs still drain.
        // `visible` is the cache length after the in-flight predecessors; `input`
        // counts the tokens this call appends. Export, latent preparation,
        // and denoising only read the request's prefix cache.
        let (_, visible) = self.scheduled_token_lengths(request_id)?;
        let cache_lengths = match call.kind {
            // A draft refill evaluates tokens the cache holds from its start
            // on, below `visible`.
            CallKind::Forward(ForwardMode::Prefill) if call.draft.refill.is_some() => {
                Some(CacheLengths {
                    visible: call.draft.refill.unwrap_or_default(),
                    input: call.bounds.max_tokens,
                })
            }
            // A context prefill's bound counts its prompt tokens and its
            // vision blocks' exact cache lengths.
            CallKind::Forward(ForwardMode::Prefill) => Some(CacheLengths {
                visible,
                input: call.bounds.max_tokens,
            }),
            CallKind::Forward(ForwardMode::Decode) => Some(CacheLengths { visible, input: 1 }),
            // A verification evaluates its current token and its drafts.
            CallKind::Forward(ForwardMode::Verify) => Some(CacheLengths {
                visible,
                input: 1 + generation::verify_drafts(&call).len() as u32,
            }),
            CallKind::Forward(ForwardMode::TokenDenoising) => {
                Some(CacheLengths { visible, input: 0 })
            }
            _ => None,
        };
        // A latent image extension declares the cache tokens of the current
        // input image's encoder that produced its feature.
        let num_image_cache_tokens = if consumes_image_features(&call) {
            let state = self.requests.tokens(&request_id)?;
            state
                .req
                .multimodal_inputs
                .images
                .get(state.num_ingested_images)
                .and_then(|image| image.encoders.get(state.image_encoder_index))
                .and_then(|encoder| encoder.num_cache_tokens)
        } else {
            None
        };

        // A generation step denoises the request's resident canvas of
        // `max_tokens` tokens, one read-only forward row below.
        let canvas_lengths = if call.canvas.is_some() {
            vec![call.bounds.max_tokens]
        } else {
            Vec::new()
        };

        if is_prompt_extend(&call) && !call.vision_inputs.is_empty() {
            let segments = self.context_segments(request_id, &call);
            if !segments.is_some_and(|segments| {
                cache_lengths
                    .as_ref()
                    .is_some_and(|lengths| segments.iter().sum::<u32>() == lengths.input)
            }) {
                self.invariant_broken(
                    "a context prefill's vision blocks are the request's next vision blocks",
                );
                return None;
            }
        }

        // What the call states to the device: its rows, the tables it reads
        // through, its fresh units and host page loads. The batch's device
        // layout is written from this, once (`attach_columns`).
        let mut layout = columns::CallLayout::default();
        if let Some(lengths) = cache_lengths {
            // Registration above found the request running, and a running
            // token request always has its cache tables.
            let Some(state) = self.requests.tokens_mut(&request_id) else {
                self.invariant_broken("a registered token request is admitted with a prefix cache");
                return None;
            };
            let fresh_count = u64::from(call.bounds.max_cache_units);
            let declared = state.num_cache_units_sent;
            // A handed-off request's first call is its first on the decode group.
            let handed_off = state.hand_off == generation::HandOff::Copied;
            state.hand_off = generation::HandOff::None;
            let (Some(request_pool_idx), Some(allocation)) =
                (state.request_pool_idx(), state.prefix_cache_mut())
            else {
                self.invariant_broken("a registered token request is admitted with a prefix cache");
                return None;
            };
            layout = columns::CallLayout::of(request_pool_idx);

            // A table is declared when its held page interval changed since
            // its last declaration (growth, or window retirement in a
            // sliding-window group) or it holds fresh units, which the worker
            // must reset before use. A forward's fresh units are those
            // allocated since the declared watermark (`plan_call`).
            // Admission (or re-admission) declares every table again, with
            // every allocated page fresh; pages acquired from the prefix
            // cache already hold computed keys and values and are never fresh.
            // A handed-off request's first call declares them to the decode
            // group, which has seen none, with no fresh unit: the hand-off
            // copied every unit's pages there.
            if admitted || handed_off {
                allocation.clear_sent();
            }
            let fresh_since = if admitted {
                Some(0)
            } else if handed_off {
                None
            } else {
                (fresh_count > 0).then(|| declared.saturating_sub(fresh_count))
            };
            let copies: Vec<_> = (0..allocation.tables.len())
                .map(|group| {
                    fresh_since
                        .map(|serial| allocation.copies(group, serial))
                        .unwrap_or_default()
                })
                .collect();
            for (table, copies) in allocation.tables.iter_mut().zip(copies) {
                let fresh = fresh_since
                    .map(|serial| table.fresh_units(serial))
                    .unwrap_or_default();
                if table.changed_since_sent() || !fresh.is_empty() {
                    layout.block_tables.push(IpcBlockTable {
                        request_pool_idx,
                        group_id: table.group_id() as u32,
                        start_page: u32::try_from(table.start_page()).unwrap_or(u32::MAX),
                        unit_ids: table.unit_ids(),
                        allocated_tokens: u32::try_from(table.capacity_tokens())
                            .unwrap_or(u32::MAX),
                    });
                    table.mark_sent();
                }
                if !fresh.is_empty() {
                    layout.cache_units.push(CacheUnitAllocation {
                        request_pool_idx,
                        group_id: table.group_id() as u32,
                        unit_ids: fresh,
                        copies,
                    });
                }
            }
            // The first call after admission loads the units of the host KV
            // cache's hit, in every cache group, from their slots before its
            // forward.
            if admitted && let Some(kv_offload) = self.kv_offload.as_mut() {
                let tables = &allocation.tables;
                layout.page_loads = kv_offload.loads(request_id, |group, page| {
                    tables
                        .get(group)
                        .and_then(|table| table.page(page))
                        .map(|page| page.units().to_vec())
                });
            }
            // A call that appends prefix cache has one row, a context prefill's
            // vision blocks included: its prompt tokens and blocks in context
            // order, one causal sequence over the visible prefix, which a
            // recurrent layer advances through as through any prompt.
            if lengths.input > 0 {
                // A draft refill writes no cache of the target: its
                // tokens' K/V are the request's already.
                layout.push(
                    lengths.visible + lengths.input,
                    lengths.input,
                    call.draft.refill.is_none(),
                );
            }
            // One read-only row per canvas: it attends to the visible prefix
            // and its own canvas, and persists nothing.
            for canvas in &canvas_lengths {
                layout.push(lengths.visible + canvas, *canvas, false);
            }
        }

        // Assign the call's final product identities from the scheduler-wide
        // generation counter; exhausting it fails every running request.
        if let Err(error) =
            generation::register_call(&mut call, request_key, &mut self.next_product_generation)
        {
            for allocation in reserved_buffers {
                self.storage.buffer_pool.free(allocation);
            }
            tracing::error!(
                request_id = request_id.0,
                ?error,
                "scheduler authority exhausted product identity space"
            );
            self.latch_fatal();
            self.fail_all_running("scheduler authority exhausted product identity space");
            return None;
        }

        // Persistent product buffers transfer ownership from the local
        // reservation into request state using the minted product identities.
        let persistent_outputs = call
            .buffer_outputs()
            .map(TensorRef::buffer_id)
            .collect::<Vec<_>>();
        if persistent_outputs.len() != reserved_buffers.len() {
            for allocation in reserved_buffers {
                self.storage.buffer_pool.free(allocation);
            }
            tracing::error!(
                request_id = request_id.0,
                "registered call changed its persistent buffer set"
            );
            self.latch_fatal();
            self.fail_all_running("registered call changed its persistent buffer set");
            return None;
        }
        let mut call_buffers = Vec::with_capacity(reserved_buffers.len());
        let Some(allocations) = self
            .requests
            .tokens_mut(&request_id)
            .and_then(|state| state.allocations_mut())
        else {
            for allocation in reserved_buffers {
                self.storage.buffer_pool.free(allocation);
            }
            self.invariant_broken("a registered call's request is admitted");
            return None;
        };
        for (buffer, allocation) in persistent_outputs.into_iter().zip(reserved_buffers) {
            let (offset, bytes) = (allocation.offset, allocation.bytes);
            let replaced = allocations.buffers.insert(buffer, allocation);
            debug_assert!(replaced.is_none(), "buffer identity was reused");
            call_buffers.push(BufferAllocation {
                buffer,
                offset,
                bytes,
            });
        }

        // Worker selection records the request's residency on that worker. A
        // request that has not run this component yet is placed where the
        // policy says (§4.4); one that has goes back to the replica holding its
        // state, whatever the policy would answer.
        let choice = self.place_call(&call)?;
        let Some((worker, entry)) =
            self.dispatch
                .select_worker(&self.worker, &self.info, &call, &choice)
        else {
            self.invariant_broken("a planned call retains an executable component");
            return None;
        };
        call.component = entry;

        // An image extension writes its block's prefix cache, whose extent its
        // result is checked against.
        let image_cache = if consumes_image_features(&call) {
            let Some(lengths) = cache_lengths else {
                self.invariant_broken("an image extension declares its cache input range");
                return None;
            };
            Some((lengths.visible, num_image_cache_tokens))
        } else {
            None
        };
        let submitted_us = crate::clock::now_monotonic_us();
        self.register_inflight(
            call.clone(),
            InflightInput::Generation { image_cache },
            submitted_us.saturating_sub(planned_us),
        );
        // The runtime's buffer pool is one address space, so every rank of
        // the worker holds each span at the same offset.
        let buffers =
            RankBuffer::on_every_rank(&call_buffers, self.worker.info().rank_count(&worker) as u32);
        batch.requests.push((
            call,
            RequestDispatch {
                worker,
                request_pool_idx: None,
                decode: None,
                buffers,
                readers: Default::default(),
            },
        ));
        layouts.push(layout);
        // The resolved token product serves only the first call after its
        // resolution; later calls chain from the in-flight one.
        if let Some(st) = self.requests.tokens_mut(&request_id) {
            st.latest_token = None;
        }
        Some(())
    }
}

impl Scheduler {
    /// Grows the request's cache tables for a call that reads from cache token
    /// `read_start` and holds `total_tokens` tokens and computes prompt
    /// tokens (or loads a reused prefix): a state group checkpoints the last
    /// boundary it reaches.
    ///
    /// A sliding-window group needs only the pages the read's window and the
    /// call's tokens intersect; every group's units come from one pool.
    /// Returns `false` when the request is not running, the cache cannot
    /// grow, or (after latching engine-fatal) a running request lacks its
    /// allocations or the prefix cache.
    pub(crate) fn ensure_request_capacity(
        &mut self,
        id: RequestId,
        read_start: usize,
        total_tokens: usize,
    ) -> bool {
        match self.grow_request_tables(id, read_start, total_tokens, Checkpoint::Boundary) {
            Ok(()) => true,
            Err(needed) => {
                self.note_starved(id, needed);
                false
            }
        }
    }

    /// Grows the request's cache tables for a decode call, as
    /// [`Self::ensure_request_capacity`] does for a prompt call, except that
    /// a state group checkpoints only a boundary that is a multiple of the
    /// state interval.
    pub(crate) fn ensure_decode_capacity(
        &mut self,
        id: RequestId,
        read_start: usize,
        total_tokens: usize,
    ) -> bool {
        let checkpoint = Checkpoint::Interval(self.state_interval);
        match self.grow_request_tables(id, read_start, total_tokens, checkpoint) {
            Ok(()) => true,
            Err(needed) => {
                self.note_starved(id, needed);
                false
            }
        }
    }

    /// Grows the request's tables over the positions `[read_start,
    /// total_tokens)` a call's drafter writes past its tokens, which keep
    /// no state checkpoint.
    pub(crate) fn ensure_draft_capacity(
        &mut self,
        id: RequestId,
        read_start: usize,
        total_tokens: usize,
    ) -> bool {
        total_tokens <= read_start
            || self
                .grow_request_tables(id, read_start, total_tokens, Checkpoint::Never)
                .is_ok()
    }

    /// Grows the request's tables; when the pool lacks the units the tables
    /// need, other requests' draft caches, optional state, release theirs
    /// first ([`Self::release_drafts`]).
    ///
    /// A refusal states the units the growth needed, one when the tables do
    /// not say: what a reclaim must free for the request to go on.
    fn grow_request_tables(
        &mut self,
        id: RequestId,
        read_start: usize,
        total_tokens: usize,
        checkpoint: Checkpoint,
    ) -> Result<(), usize> {
        let read_start = read_start.min(u32::MAX as usize) as u32;
        let total_tokens = total_tokens.min(u32::MAX as usize) as u32;
        let mut needed = None;
        for attempt in 0..2 {
            if attempt == 1 {
                let Some(units) = needed else {
                    return Err(1);
                };
                if !self.release_drafts(units) {
                    return Err(units.max(1));
                }
            }
            let Some(state) = self.requests.tokens_mut(&id) else {
                return Err(1);
            };
            let (Some(allocations), Some(cache)) =
                (state.allocations_mut(), self.storage.cache.as_ref())
            else {
                self.invariant_broken("a running token request is admitted with a prefix cache");
                return Err(1);
            };
            if cache
                .grow(
                    &mut allocations.prefix_cache,
                    read_start,
                    total_tokens,
                    checkpoint,
                )
                .is_ok()
            {
                return Ok(());
            }
            needed = cache.units_needed(
                &allocations.prefix_cache,
                read_start as usize,
                total_tokens as usize,
                checkpoint,
            );
        }
        Err(needed.unwrap_or(1).max(1))
    }

    /// Activates the cache tables reserved for a request.
    ///
    /// A request holds tables only once admitted, and only token requests,
    /// which always have the prefix cache, hold them.
    pub(crate) fn activate_request_tables(&self, id: RequestId) {
        let tables = self.requests.tokens(&id).and_then(TokenState::block_tables);
        if let (Some(tables), Some(prefix_cache)) = (tables, self.storage.cache()) {
            for table in tables {
                table.activate(prefix_cache.pool());
            }
        }
    }
}
