//! Registering a selected computation as a call: its device predicate, the
//! prefix-cache, forward-row and latent inputs derived from the lengths its
//! in-flight predecessors project, its outputs, and the cache tables a request
//! grows and activates for it.

use super::*;
use uniserve_ipc::{ForwardMode, MediaCall, TransferMode};

impl Scheduler {
    /// Registers one selected computation with its allocated physical inputs and outputs.
    ///
    /// Sets the call's device predicate, derives its prefix cache, forward-row, and
    /// latent inputs from the lengths its in-flight predecessors project,
    /// mints product identities, moves `reserved_buffers` into the request's
    /// allocations, selects the executing worker, registers the call in
    /// flight, and appends it with its `RequestPlacement` to `batch`.
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
    ) -> Option<()> {
        let request_id = call.request_key.request_id;

        // Planning may queue successors speculatively, but registration still
        // requires the request epoch and exact resident predecessor.
        let Some((request_epoch, latest_token)) = self.running.get(&request_id).and_then(|state| {
            state
                .worker_registered
                .then(|| (state.request_epoch, state.latest_token.clone()))
        }) else {
            tracing::error!(request_id = request_id.0, "planned call lost its request");
            self.fatal = true;
            return None;
        };
        if self.inflight.has_pending_calls(request_id)
            && !self.can_queue_successor(request_id, call.kind)
        {
            tracing::error!(
                request_id = request_id.0,
                "scheduler attempted to issue an unsafe projected successor"
            );
            self.fatal = true;
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
        let projected_successor = self.inflight.has_pending_calls(request_id);
        let reusable_device_token =
            if !projected_successor && self.can_reuse_resolved_token_product(request_id) {
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
                self.fatal = true;
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
                self.fatal = true;
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
                self.fatal = true;
                return None;
            }
            None
        };

        call.predicate = predicate;

        // Freeze physical inputs before registering this computation as pending.
        // Accepted progress may stop on cancellation while these inputs still drain.
        // `visible` is the cache length after the in-flight predecessors; `input`
        // counts the tokens this call appends. Export, latent preparation,
        // and denoising only read the request's prefix cache.
        let (_, visible) = self.scheduled_token_lengths(request_id)?;
        let cache_lengths = match call.kind {
            // A context prefill's bound counts its prompt tokens and its
            // vision blocks' exact cache lengths.
            CallKind::Forward(ForwardMode::Prefill) => Some(CacheLengths {
                visible,
                input: if is_prompt_extend(&call) || consumes_image_features(&call) {
                    call.bounds.max_tokens
                } else {
                    1
                },
            }),
            CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                Some(CacheLengths { visible, input: 1 })
            }
            CallKind::Forward(ForwardMode::TokenDenoising)
            | CallKind::Transfer(TransferMode::CacheExport)
            | CallKind::Media(MediaCall::LatentPreparation)
            | CallKind::Media(MediaCall::Denoising) => Some(CacheLengths { visible, input: 0 }),
            _ => None,
        };
        // An image-feature extension declares the cache tokens of the encoder
        // that produced its feature: the scheduled feedback encoder, or the
        // current input image's encoder.
        let num_image_cache_tokens = if consumes_image_features(&call) {
            let state = self.running.get(&request_id)?;
            let encoder = if is_feedback_computation(&call) {
                state
                    .req
                    .image_generation
                    .feedback_encoders
                    .get(self.scheduled_feedback(request_id)?.0)
            } else {
                state
                    .req
                    .multimodal_inputs
                    .images
                    .get(state.num_ingested_images)
                    .and_then(|image| image.encoders.get(state.image_encoder_index))
            };
            encoder.and_then(|encoder| encoder.num_cache_tokens)
        } else {
            None
        };

        // A canvas pass covers the readout rows after those already scheduled;
        // each row's canvas becomes one read-only forward row below.
        let canvas_lengths = if call.readout.is_some() {
            let rows = self.scheduled_readout_rows(request_id).and_then(|start| {
                let state = self.running.get(&request_id)?;
                let count = state.readout_rows_covering(start, call.input_token_ids.len())?;
                Some(
                    state.req.readout[start..start + count]
                        .iter()
                        .map(|row| row.token_ids.len() as u32)
                        .collect::<Vec<_>>(),
                )
            });
            let Some(rows) = rows else {
                self.invariant_broken("a canvas pass covers whole scheduled readout rows");
                return None;
            };
            rows
        } else if call.canvas.is_some() {
            // A generation step denoises the request's resident canvas of
            // `max_tokens` tokens.
            vec![call.bounds.max_tokens]
        } else {
            Vec::new()
        };

        let context_segments = if is_prompt_extend(&call) && !call.vision_inputs.is_empty() {
            let segments = self.context_segments(request_id, &call);
            let Some(segments) = segments.filter(|segments| {
                cache_lengths
                    .as_ref()
                    .is_some_and(|lengths| segments.iter().sum::<u32>() == lengths.input)
            }) else {
                self.invariant_broken(
                    "a context prefill's vision blocks are the request's next vision blocks",
                );
                return None;
            };
            segments
        } else {
            Vec::new()
        };

        let mut call_block_tables = Vec::new();
        let mut call_new_cache_units = Vec::new();
        let mut call_forward = ForwardBatch::default();
        if let Some(lengths) = cache_lengths {
            // Registration above found the request running, and a running
            // token request always has its cache tables.
            let Some(state) = self.running.get_mut(&request_id) else {
                self.invariant_broken("a registered token request is admitted with a prefix cache");
                return None;
            };
            let fresh_count = u64::from(call.bounds.max_cache_units);
            let declared = state.num_cache_units_sent;
            let (Some(request_pool_idx), Some(allocation)) =
                (state.request_pool_idx(), state.prefix_cache_mut())
            else {
                self.invariant_broken("a registered token request is admitted with a prefix cache");
                return None;
            };

            // A table is declared when its held page interval changed since
            // its last declaration (growth, or window retirement in a
            // sliding-window group) or it holds fresh units, which the worker
            // must reset before use. A forward's fresh units are those
            // allocated since the declared watermark (`plan_computation`).
            // Admission (or re-admission) declares every table again, with
            // every allocated page fresh; pages acquired from the prefix
            // cache already hold computed keys and values and are never fresh.
            if admitted {
                allocation.clear_sent();
            }
            let fresh_since = if admitted {
                Some(0)
            } else {
                (fresh_count > 0).then(|| declared.saturating_sub(fresh_count))
            };
            for table in &mut allocation.tables {
                let fresh = fresh_since
                    .map(|serial| table.fresh_units(serial))
                    .unwrap_or_default();
                if table.changed_since_sent() || !fresh.is_empty() {
                    call_block_tables.push(IpcBlockTable {
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
                    call_new_cache_units.push(CacheUnitAllocation {
                        request_pool_idx,
                        group_id: table.group_id() as u32,
                        unit_ids: fresh,
                    });
                }
            }
            // A context prefill with vision blocks appends one forward row per
            // segment, in context order: the prompt tokens between blocks, and
            // each block, which the worker evaluates as one attention block.
            // Every other call that appends prefix cache has one row. Each row attends
            // to the visible prefix and the rows before it, and persists its
            // tokens in the prefix cache.
            if !context_segments.is_empty() {
                let mut end = lengths.visible;
                for &segment in &context_segments {
                    end += segment;
                    call_forward.push(0, request_pool_idx, end, segment, true);
                }
            } else if lengths.input > 0 {
                call_forward.push(
                    0,
                    request_pool_idx,
                    lengths.visible + lengths.input,
                    lengths.input,
                    true,
                );
            }
            // One read-only row per canvas: it attends to the visible prefix
            // and its own canvas, and persists nothing.
            for canvas in &canvas_lengths {
                call_forward.push(
                    0,
                    request_pool_idx,
                    lengths.visible + canvas,
                    *canvas,
                    false,
                );
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
            self.fatal = true;
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
            self.fatal = true;
            self.fail_all_running("registered call changed its persistent buffer set");
            return None;
        }
        let mut call_buffers = Vec::with_capacity(reserved_buffers.len());
        let Some(allocations) = self
            .running
            .get_mut(&request_id)
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

        // A denoising call adds one query row per guidance branch; its query
        // is the latent grid plus the commit marker tokens and writes nothing to the prefix cache.
        // Branch 0 attends to the request's own conditioning prefix cache. Every other
        // branch attends to the guidance prefix (the negative-prompt prefix cache in its own
        // request row) when the request holds one, and to the request's own
        // prefix cache otherwise. Until a denoising call completes successfully and
        // finalizes the prefix, the call also prefills a non-empty negative
        // prompt into it; the prefix's tables are resent until then and
        // whenever the prefix has gained pages.
        if call.kind == CallKind::Media(MediaCall::Denoising) {
            let Some(CacheLengths {
                visible: conditioning_tokens,
                ..
            }) = cache_lengths
            else {
                self.invariant_broken("denoising declares its conditioning cache range");
                return None;
            };
            let query_len = u32::try_from(
                self.running
                    .get(&request_id)
                    .map(|state| self.num_vae(&state.req.image))
                    .unwrap_or_default()
                    .saturating_add(u64::from(self.generation_limits.commit_marker_tokens)),
            )
            .unwrap_or(u32::MAX);
            let Some(state) = self.running.get_mut(&request_id) else {
                self.fatal = true;
                return None;
            };
            let Some(main_slot) = state.request_pool_idx() else {
                self.invariant_broken("a denoising request is admitted");
                return None;
            };
            // Each guidance prefix carries its tables and fresh units, and is
            // prefilled until a denoising call completes; then one row per
            // branch, in evaluation order, reads the prefix it attends to.
            let mut text_row = None;
            let mut image_row = None;
            if let Some(state) = state.guidance_prefix.as_mut() {
                let finalized = state.diffusion_finalized;
                for prefix in &mut state.prefixes {
                    if !finalized || !prefix.new_units.is_empty() {
                        for table in prefix.block_tables() {
                            call_block_tables.push(IpcBlockTable {
                                request_pool_idx: prefix.request_pool_idx(),
                                group_id: table.group_id() as u32,
                                start_page: u32::try_from(table.start_page()).unwrap_or(u32::MAX),
                                unit_ids: table.unit_ids(),
                                allocated_tokens: u32::try_from(table.capacity_tokens())
                                    .unwrap_or(u32::MAX),
                            });
                        }
                    }
                    for (group_id, units) in std::mem::take(&mut prefix.new_units) {
                        if !units.is_empty() {
                            call_new_cache_units.push(CacheUnitAllocation {
                                request_pool_idx: prefix.request_pool_idx(),
                                group_id,
                                unit_ids: units,
                            });
                        }
                    }
                    if prefix.tokens > 0 && !finalized {
                        call_forward.push(
                            0,
                            prefix.request_pool_idx(),
                            prefix.tokens,
                            prefix.tokens,
                            true,
                        );
                    }
                }
                text_row = state.row(state.text);
                image_row = state.row(state.image);
            }
            // The conditioned branch reads the request's own context; a
            // branch without a prefix of its own (an image-unconditional
            // branch the worker states none for) reads it too.
            let own = (main_slot, conditioning_tokens);
            let mut rows = vec![own];
            if state.req.image.guides_text() {
                rows.push(text_row.unwrap_or(own));
            }
            if state.req.image.guides_image() {
                rows.push(image_row.unwrap_or(own));
            }
            for (request_pool_index, seq_len) in rows {
                call_forward.push(0, request_pool_index, seq_len + query_len, query_len, false);
            }
        }

        let mut latent = None;

        // Latent allocations describe the request-owned page table this call
        // addresses; the worker's latent pool keeps the trajectory's steps.
        if call.addresses_latent_pages() {
            let Some(state) = self.running.get(&request_id) else {
                self.fatal = true;
                return None;
            };
            let Some(allocations) = state.allocations() else {
                self.invariant_broken("a request with latent work is admitted");
                return None;
            };
            let placement = LatentPlacement {
                page_table: allocations
                    .latent
                    .as_ref()
                    .map(|allocation| allocation.pages.clone())
                    .unwrap_or_default(),
                latent_units: self
                    .worker_image_latent_units_for(state)
                    .max(1)
                    .min(u64::from(u32::MAX)) as u32,
                height: state.req.image.height,
                width: state.req.image.width,
            };
            latent = Some(Denoising::params(
                call.request_key,
                call.call_id,
                &placement,
            ));
        }

        // Worker selection records the request's residency on that worker.
        let Some((worker, entry)) =
            self.placement
                .select_worker(self.executor.as_ref(), &self.info, &call)
        else {
            self.invariant_broken("a planned call retains an executable component");
            return None;
        };
        call.component = entry;

        // An image extension writes its block's prefix cache, and a context prefill
        // with vision blocks exactly its bound: its prompt tokens and blocks.
        let image_cache = if consumes_image_features(&call) || !call.vision_inputs.is_empty() {
            let Some(lengths) = cache_lengths else {
                self.invariant_broken("an image extension declares its cache input range");
                return None;
            };
            Some((
                lengths.visible,
                if consumes_image_features(&call) {
                    num_image_cache_tokens
                } else {
                    Some(lengths.input)
                },
            ))
        } else {
            None
        };
        let submitted_us = uniserve_core::now_monotonic_us();
        self.register_inflight(
            call.clone(),
            InflightInput::Generation { image_cache },
            submitted_us.saturating_sub(planned_us),
        );
        // The runtime's buffer pool is one address space, so every rank of
        // the worker holds each span at the same offset.
        let buffers = RankBuffer::on_every_rank(
            &call_buffers,
            self.executor.info().rank_count(&worker) as u32,
        );
        batch.requests.push((
            call,
            RequestPlacement {
                worker,
                request_pool_idx: None,
                block_tables: call_block_tables,
                new_cache_units: call_new_cache_units,
                forward: call_forward,
                latent,
                decode: None,
                buffers,
                readers: Default::default(),
            },
        ));
        // The resolved token product serves only the first call after its
        // resolution; later calls chain from the in-flight one.
        if let Some(st) = self.running.get_mut(&request_id) {
            st.latest_token = None;
        }
        Some(())
    }
}

impl Scheduler {
    /// Grows the request's cache tables for a call that reads from cache token
    /// `read_start` and holds `total_tokens` tokens.
    ///
    /// A sliding-window group needs only the pages the read's window and the
    /// call's tokens intersect; every group's units come from one pool.
    /// Returns `false` when the request is not running, the cache cannot
    /// grow, or (after latching engine-fatal) a running request lacks its
    /// allocations or the prefix cache.
    pub(in crate::scheduler) fn ensure_request_capacity(
        &mut self,
        id: RequestId,
        read_start: usize,
        total_tokens: usize,
    ) -> bool {
        let Some(state) = self.running.get_mut(&id) else {
            return false;
        };
        let (Some(allocations), Some(cache)) =
            (state.allocations_mut(), self.storage.cache.as_ref())
        else {
            self.invariant_broken("a running token request is admitted with a prefix cache");
            return false;
        };
        cache
            .grow(
                &mut allocations.prefix_cache,
                read_start.min(u32::MAX as usize) as u32,
                total_tokens.min(u32::MAX as usize) as u32,
            )
            .is_ok()
    }

    /// Activates the cache tables reserved for a request.
    ///
    /// A request holds tables only once admitted, and only token requests,
    /// which always have the prefix cache, hold them.
    pub(in crate::scheduler) fn activate_request_tables(&self, id: RequestId) {
        let tables = self.running.get(&id).and_then(RequestState::block_tables);
        if let (Some(tables), Some(prefix_cache)) = (tables, self.storage.cache()) {
            for table in tables {
                table.activate(&prefix_cache.block_pool);
            }
        }
    }
}
