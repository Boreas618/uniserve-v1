//! Batch selection, call planning, and physical params assembly.
//!
//! Each pass selects a compatible execution lane, applies sequence and token
//! budgets, and emits at most one planned call per eligible request.
//!
//! This is the generation path: token requests in `running`, including the
//! image-generation and feedback phases of unified multimodal requests.
//! `Scheduler::schedule_batches` alternates it with the video media path.
//! A pass plans each request's next call from its accepted state plus the
//! calls still in flight (`scheduled_token_lengths`,
//! `num_scheduled_denoise_steps`), so a successor can be planned before its
//! predecessor resolves. Each planned call then reserves its storage, is
//! registered as in flight, and joins the pass's batch for its `CallKind`:
//! one batch per computation, each a single numerical call on one component.
//!
//! The lane (`BatchKind`) only separates prefill from decode. `BatchKind::Media`
//! calls are eligible in every pass, and a pass with no lane applies no filter.
//!
//! A pass's prefill calls, and likewise its decode calls, travel as one
//! numerical call, which workers evaluate with graphs captured at startup; a
//! prefill or decode batch holds at most the calls those graphs hold, as
//! the workers report it (`Scheduler::call_limit`). A context prefill with
//! vision blocks stages several forward rows; a pass's prefill rows, like its
//! canvas rows, stay within the rows a worker stages per call (`PassRows`).

mod call;
mod computation;

use super::*;
use uniserve_ipc::{ForwardMode, MediaCall, TransferMode};

/// Forward rows a pass may still add to its numerical calls, within the rows
/// a worker stages per call: canvas rows across its token-denoising calls,
/// and context rows across its prefill calls.
#[derive(Debug, Clone, Copy)]
pub(super) struct PassRows {
    pub(super) canvas: usize,
    pub(super) context: usize,
}

/// Returns the cache capacity, in tokens, that must be allocated before a decode
/// that writes position `pos` and `spec_len` further positions.
fn decode_capacity_target(pos: usize, spec_len: usize) -> usize {
    pos.saturating_add(1).saturating_add(spec_len)
}

impl Scheduler {
    /// Assembles one scheduling-step batch in priority order.
    ///
    /// Each request contributes at most one call, prefill chunks consume only
    /// the remaining token budget, and first dispatch carries typed admission.
    ///
    /// Returns the pass's batches, empty when nothing can be scheduled or call
    /// preparation failed; most preparation failures latch engine-fatal.
    pub(super) fn assemble(&mut self) -> Vec<ExecutionBatch> {
        let ids = self.assembly_order();
        let lane = self.select_batch_kind(&ids);
        let batches = self.assemble_batch(&ids, lane);
        if batches.is_empty() && !self.fatal && lane == Some(BatchKind::Prefill) {
            // A blocked prefill lane must not prevent already-ready decode work
            // from using the execution slot.
            return self.assemble_batch(&ids, Some(BatchKind::Decode));
        }
        batches
    }

    /// Charges actual token work, including each CFG branch of a denoising step.
    ///
    /// A denoising call costs its latent units times its guidance branches;
    /// it costs nothing once its request has left `running`. A verify call
    /// costs its input tokens plus one, and every other call its `max_tokens`
    /// bound.
    fn computation_token_cost(&self, call: &Call) -> usize {
        match call.code {
            CallKind::Media(MediaCall::Denoising) => {
                let Some(state) = self.running.get(&call.request_key.request_id) else {
                    return 0;
                };
                usize::try_from(self.worker_image_latent_units_for(state).max(1))
                    .unwrap_or(usize::MAX)
                    .saturating_mul(usize::from(state.req.image.cfg_branch_count().max(1)))
            }
            CallKind::Forward(ForwardMode::Verify) => 1 + call.input_token_ids.len(),
            _ => call.bounds.max_tokens as usize,
        }
    }

    /// Reserves persistent buffers, latent pages, and transfer capacity for one computation.
    ///
    /// Returns the call's output buffer spans, in `Call::buffer_outputs`
    /// order, or `None` when the call cannot run now: the guidance prefix of a
    /// denoising call cannot be allocated, transfer capacity is exhausted,
    /// the request is not running, `Placement::worker_target` finds no worker,
    /// a buffer or latent-page allocation fails, or (after latching
    /// engine-fatal) the request holds no allocations.
    ///
    /// Buffer spans reserved before a failure are returned to the pool, but a
    /// failure is not free of side effects: a guidance prefix already allocated
    /// stays with the request, and a failed buffer allocation evicts one
    /// unpinned encoder-cache product, if any, and queues a `Free` for its
    /// buffer. Only a successful reservation counts against the transfer
    /// capacity.
    fn reserve_generation_resources(&mut self, call: &Call) -> Option<Vec<BufferSpan>> {
        let id = call.request_key.request_id;
        if call.code == CallKind::Media(MediaCall::Denoising) && !self.ensure_guidance_prefix(id) {
            return None;
        }
        let uses_transfer = call.bounds.max_transfer_bytes > 0;
        if uses_transfer && self.inflight.num_pending_transfers >= self.transfer_capacity {
            return None;
        }
        let request_key = self
            .running
            .get(&id)
            .map(|state| RequestKey::new(self.engine_id, id, state.request_epoch))?;
        self.placement
            .worker_target(self.executor.as_ref(), &self.info, request_key, call.code)?;

        let mut buffer_allocations = Vec::new();
        for bytes in call.buffer_outputs().map(TensorRef::max_bytes) {
            let allocation = match self.storage.buffer_pool.allocate(request_key, bytes, 256) {
                Ok(allocation) => allocation,
                Err(_) => {
                    for allocation in buffer_allocations {
                        self.storage.buffer_pool.free(allocation);
                    }
                    // The evicted product's span returns to the pool only
                    // when workers acknowledge the queued `Free` command.
                    if let Some(product) = self.storage.encoder_cache.evict_one() {
                        self.free_buffers([product.buffer_id()]);
                    }
                    return None;
                }
            };
            buffer_allocations.push(allocation);
        }

        // Latent pages are reserved only when the worker advertises a paged
        // latent pool (`worker_tracks_image_latent`). They grow in place
        // across the trajectory's calls; a failed grow leaves the existing
        // pages allocated.
        if matches!(
            call.code,
            CallKind::Media(MediaCall::LatentPreparation) | CallKind::Media(MediaCall::Denoising)
        ) && self.worker_tracks_image_latent()
        {
            let latent_units = self
                .running
                .get(&id)
                .map(|state| self.worker_image_latent_units_for(state).max(1))?;
            let state = self.running.get_mut(&id)?;
            let Some(allocations) = state.allocations_mut() else {
                for allocation in buffer_allocations {
                    self.storage.buffer_pool.free(allocation);
                }
                self.invariant_broken("a request planning latent storage is admitted");
                return None;
            };
            let result = if let Some(allocation) = allocations.latent.as_mut() {
                self.storage.latent_pool.grow(allocation, latent_units)
            } else {
                self.storage
                    .latent_pool
                    .allocate(latent_units)
                    .map(|allocation| {
                        allocations.latent = Some(allocation);
                    })
            };
            if result.is_err() {
                for allocation in buffer_allocations {
                    self.storage.buffer_pool.free(allocation);
                }
                return None;
            }
        }

        if uses_transfer {
            self.inflight.num_pending_transfers += 1;
        }
        Some(buffer_allocations)
    }

    /// Returns the most calls one batch of `code` may hold.
    ///
    /// Workers report the rows their captured prefill and decode graphs hold
    /// (`WorkerInfo::max_prefill_calls`, `WorkerInfo::max_decode_calls`); zero
    /// means those calls run eagerly and only the pass's call bound applies,
    /// as it does to every other computation.
    pub(super) fn call_limit(&self, code: CallKind) -> usize {
        let reported = match code {
            CallKind::Forward(ForwardMode::Prefill) => self.info.max_prefill_calls,
            CallKind::Forward(ForwardMode::Decode) => self.info.max_decode_calls,
            _ => 0,
        };
        match reported {
            0 => self.config.max_batch,
            limit => self.config.max_batch.min(limit as usize),
        }
    }

    /// Selects compatible call kinds within one lane's token and sequence budgets.
    ///
    /// Walks `ids` in assembly order until `max_batch` calls are selected or
    /// the token budget is spent. A prefill or decode batch stops growing
    /// at its `call_limit`; requests whose next call is of that computation
    /// then wait for a later pass. Every selected call is registered in flight
    /// before this returns. A failed call preparation returns no batches; most
    /// such failures latch engine-fatal first.
    pub(super) fn assemble_batch(
        &mut self,
        ids: &[RequestId],
        lane: Option<BatchKind>,
    ) -> Vec<ExecutionBatch> {
        // One batch per computation: a batch is one numerical call on one
        // component, so a rank receives it as a single homogeneous group. The
        // component a call binds to follows from its computation, so the two
        // together name one group. Computations keep the order in which their
        // first call was selected.
        let mut code_order: Vec<CallKind> = Vec::new();
        let mut code_batches: HashMap<CallKind, ExecutionBatch> = HashMap::new();
        let submit_at = Instant::now();
        // The per-step token budget is the binding limit; individual prefill
        // chunks are clipped to its remaining capacity.
        let mut budget: usize = self.config.max_num_batched_tokens;
        // Context calls expand into one forward row per text/image segment.
        // Both input buffers and the reported prefill graph capacity bound those
        // numerical rows, independently of the number of requests batched.
        let row_capacity = (self.info.max_batch_calls.min(self.info.request_slots) as usize).max(1);
        let mut rows_left = PassRows {
            canvas: row_capacity,
            context: row_capacity.min(self.call_limit(CallKind::Forward(ForwardMode::Prefill))),
        };
        let limits = [
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::Decode),
        ]
        .map(|code| (code, self.call_limit(code)));
        let full = |batches: &HashMap<CallKind, ExecutionBatch>, code: CallKind| {
            limits.iter().any(|(bounded, limit)| {
                *bounded == code
                    && batches
                        .get(&code)
                        .is_some_and(|batch| batch.requests.len() >= *limit)
            })
        };
        // Host-lane occupancy, built when the pass meets its first encoder
        // call carrying inline image bytes and charged with each one it
        // selects.
        let mut image_lanes: Option<super::execution::LaneLedger> = None;
        let mut selected = 0usize;
        for id in ids.iter().copied() {
            if selected >= self.config.max_batch {
                break;
            }
            if budget == 0 {
                break;
            }
            if !self.can_schedule_next(id) {
                continue;
            }
            let cancelled = self
                .running
                .get(&id)
                .map(|state| state.terminal_intent.is_terminal())
                .unwrap_or(true);
            if cancelled {
                continue;
            }
            let next_type = self.peek_next_call_variant(id);
            if next_type.is_some_and(|code| full(&code_batches, code)) {
                continue;
            }
            // The lane check rejects only a prefill call in a decode pass or
            // the reverse; `BatchKind::Media` calls pass.
            if let (Some(target), Some(call_variant)) = (lane, next_type)
                && {
                    let candidate_lane = batch_kind(call_variant);
                    matches!(candidate_lane, BatchKind::Prefill | BatchKind::Decode)
                        && candidate_lane != target
                }
            {
                continue;
            }
            // The denoiser is an exclusive lane: a request holds it while one
            // of its denoising calls is in flight, so denoising steps of
            // different requests are never in flight together, in any lane.
            // Between two steps of one request, another request may take it.
            if next_type == Some(CallKind::Media(MediaCall::Denoising))
                && (self.inflight.denoiser_lane_held_by_other(id)
                    || !self.guidance_prefix_is_schedulable(id))
            {
                continue;
            }
            // Build one call when its exact resident resources fit.
            if let Some(mut call) = self.next_generation_computation(id, budget, rows_left) {
                // A planned call joins its computation's batch only while it
                // has room; otherwise it waits for a later pass, as a call
                // without resources does below. A context prefill plans
                // within the pass's context rows.
                let prefill = call.code == CallKind::Forward(ForwardMode::Prefill);
                if full(&code_batches, call.code)
                    || (prefill && generation::prefill_rows(&call) > rows_left.context)
                {
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_cache_units_sent = state
                            .num_cache_units_sent
                            .saturating_sub(u64::from(call.bounds.max_cache_units));
                    }
                    continue;
                }
                // An encoder call carrying inline image bytes holds a host
                // task on each rank of its component while the worker
                // prepares the image, so it waits here, as a media call
                // does, until those host lanes have room.
                let image_lane =
                    match call.code {
                        CallKind::Media(media_call) if call.input_image.is_some() => {
                            let Some(request_key) = self.running.get(&id).map(|state| {
                                RequestKey::new(self.engine_id, id, state.request_epoch)
                            }) else {
                                continue;
                            };
                            let demand = self.image_lane_demand(request_key, media_call);
                            let ledger = image_lanes.get_or_insert_with(|| self.lane_occupancy());
                            if !ledger.admits(&demand, id, self) {
                                self.record_domain_backpressure(call.code);
                                if let Some(state) = self.running.get_mut(&id) {
                                    state.num_cache_units_sent = state
                                        .num_cache_units_sent
                                        .saturating_sub(u64::from(call.bounds.max_cache_units));
                                }
                                continue;
                            }
                            Some(demand)
                        }
                        _ => None,
                    };
                let planned_us = uniserve_core::now_monotonic_us();

                // Only a call that obtains its storage is charged to the
                // pass. A call that cannot run now leaves the budget to later
                // requests; in particular, requests waiting for the guidance-prefix
                // row must not spend the pass before the request that holds
                // the row, which is the one whose steps release it.
                let Some(reserved_buffers) = self.reserve_generation_resources(&call) else {
                    self.record_domain_backpressure(call.code);
                    // Planning a forward advanced `num_cache_units_sent` past its
                    // fresh units; roll it back so the next plan declares them.
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_cache_units_sent = state
                            .num_cache_units_sent
                            .saturating_sub(u64::from(call.bounds.max_cache_units));
                    }
                    tracing::debug!(
                        request_id = id.0,
                        "call registration is paused by physical resource pressure"
                    );
                    continue;
                };
                let cost = self.computation_token_cost(&call);
                budget = budget.saturating_sub(cost);
                if call.readout.is_some() {
                    let rows = self.running.get(&id).and_then(|state| {
                        let start = self.scheduled_readout_rows(id)?;
                        state.readout_rows_covering(start, call.input_token_ids.len())
                    });
                    rows_left.canvas = rows_left.canvas.saturating_sub(rows.unwrap_or(0));
                } else if call.canvas.is_some() {
                    rows_left.canvas = rows_left.canvas.saturating_sub(1);
                } else if prefill {
                    rows_left.context = rows_left
                        .context
                        .saturating_sub(generation::prefill_rows(&call));
                }

                // Identify the call by its batch and row, and stamp the
                // coordinates projected through the request's in-flight calls.
                let code = call.code;
                let batch = code_batches.entry(code).or_insert_with(|| {
                    code_order.push(code);
                    ExecutionBatch::new(
                        self.inflight.next_batch_id(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    )
                });
                let Ok(request_index) = u32::try_from(batch.requests.len()) else {
                    self.invariant_broken("a batch's request count fits the IPC index");
                    return Vec::new();
                };
                call.call_id = CallId::new(batch.id, request_index);
                let Some(coordinates) = self.projected_coordinates(id) else {
                    self.invariant_broken("a scheduled request retains its running state");
                    return Vec::new();
                };
                call.coordinates = coordinates;

                // A request's first call carries its `Start` admission in the
                // same batch.
                let finish_token_ids = self
                    .running
                    .get(&id)
                    .map(|state| state.finish_token_ids.clone())
                    .unwrap_or_default();
                let engine_id = self.engine_id;
                let mut admitted = false;
                if let Some(st) = self.running.get_mut(&id)
                    && !st.worker_registered
                {
                    let request_key = RequestKey::new(engine_id, id, st.request_epoch);
                    // Admission reserved the request row, and submission
                    // validated the parameters this admission carries.
                    let admission = st
                        .request_pool_idx()
                        .zip(u32::try_from(st.req.multimodal_inputs.images.len()).ok())
                        .map(|(request_pool_idx, input_images)| {
                            NewRequest::new(
                                request_key,
                                request_pool_idx,
                                Some(ArRequestParams {
                                    sampling: st.req.sampling.clone(),
                                    negative_token_ids: st.req.negative_prompt_token_ids.clone(),
                                    finish_token_ids,
                                    initial_position: st.num_computed_prompt_tokens,
                                    canvas: st.req.canvas,
                                }),
                                st.req.generates_images().then(|| st.req.image.clone()),
                                input_images,
                            )
                        });
                    let Some(Ok(admission)) = admission else {
                        self.invariant_broken("a running request forms a valid worker admission");
                        return Vec::new();
                    };
                    st.worker_registered = true;
                    // Admission is the first request-state dependency.
                    st.last_state_call_id = CallId::default();
                    st.latest_token = None;
                    batch.commands.push(BatchCommand::Start {
                        request: Box::new(admission),
                    });
                    admitted = true;
                }
                let prepared = self.prepare_generation_call(
                    call,
                    reserved_buffers,
                    admitted,
                    planned_us,
                    batch,
                );
                if prepared.is_none() {
                    // Most preparation failures latch engine-fatal first.
                    return Vec::new();
                }
                if let (Some(demand), Some(ledger)) = (&image_lane, image_lanes.as_mut()) {
                    ledger.occupy(demand, id);
                }
                selected += 1;
            }
        }
        code_order.retain(|code| {
            code_batches
                .get(code)
                .is_some_and(|batch| !batch.requests.is_empty())
        });
        let Some(&last) = code_order.last() else {
            return Vec::new();
        };

        // A prefill-only pass takes only the queued commands of its own
        // requests plus `Finish` commands; other requests' `Free` commands
        // wait for a later batch. The round's retirements travel with its last
        // batch so no earlier call loses the state it still reads. They do not
        // travel in a batch of their own: a rank's queue depth bounds
        // submissions, not round trips, so a command-only batch costs the
        // round a queue slot and halves how many rounds a depth-two rank can
        // hold in flight.
        let prompt_only = code_order
            .iter()
            .all(|code| batch_kind(*code) == BatchKind::Prefill);
        let commands = if prompt_only {
            let requests = code_order
                .iter()
                .flat_map(|code| code_batches[code].requests.iter())
                .map(|(call, _)| call.request_key.request_id)
                .collect::<HashSet<_>>();
            self.take_commands(|command| {
                requests.contains(&command.request_key().request_id)
                    || matches!(command, BatchCommand::Finish { .. })
            })
        } else {
            self.take_commands(|_| true)
        };
        let mut batches = Vec::with_capacity(code_order.len());
        for code in code_order {
            let Some(mut batch) = code_batches.remove(&code) else {
                self.invariant_broken("every selected call kind has its batch");
                return Vec::new();
            };
            if code == last {
                batch.commands.extend(commands.iter().cloned());
            }
            batches.push(self.finish_generation_batch(batch, submit_at));
        }
        batches
    }

    /// Chooses the highest-priority execution lane that has schedulable work.
    ///
    /// A ready readout pass takes the decode lane first: one pass answers
    /// the readout and releases its prefix cache. Otherwise prefill wins while fewer
    /// than `PREFILL_WINDOW_CREDITS` batches carrying `BatchKind::Prefill`
    /// calls await their results; past that window, ready decode work takes
    /// the pass, and prefill runs only when no decode is ready.
    ///
    /// A generating block's canvas steps are decode work under this window,
    /// not ahead of it: a block's next step is ready again as soon as the
    /// step before it is submitted, so a prompt ordered behind every ready
    /// step would wait until the running blocks stop instead of joining
    /// their canvas batches.
    ///
    /// `BatchKind::Media` readiness never selects a lane: with only such work
    /// ready this returns `None`, and `assemble_batch` then applies no lane
    /// filter.
    pub(super) fn select_batch_kind(&self, ids: &[RequestId]) -> Option<BatchKind> {
        let mut readout_ready = false;
        let mut decode_ready = false;
        let mut prefill_ready = false;
        for id in ids.iter().copied() {
            let Some(state) = self.running.get(&id) else {
                continue;
            };
            if state.terminal_intent.is_terminal() {
                continue;
            }
            let Some(call_type) = self.peek_next_call_variant(id) else {
                continue;
            };
            let lane = batch_kind(call_type);
            if lane == BatchKind::Media || !self.can_schedule_next(id) {
                continue;
            }
            match lane {
                BatchKind::Decode if state.req.is_readout() => readout_ready = true,
                BatchKind::Decode => decode_ready = true,
                BatchKind::Prefill => prefill_ready = true,
                BatchKind::Media => {}
            }
        }
        let prefill_window_open = self
            .inflight
            .pending_batches
            .values()
            .filter(|batch| batch.prefill)
            .count()
            < PREFILL_WINDOW_CREDITS;
        if readout_ready {
            Some(BatchKind::Decode)
        } else if prefill_ready && prefill_window_open {
            Some(BatchKind::Prefill)
        } else if decode_ready {
            Some(BatchKind::Decode)
        } else if prefill_ready {
            Some(BatchKind::Prefill)
        } else {
            None
        }
    }

    /// Returns running requests in batch-assembly order: by
    /// `assembly_priority`, then by admission order within a priority.
    pub(super) fn assembly_order(&self) -> Vec<RequestId> {
        let mut ids: Vec<(usize, RequestId)> = self
            .running_order
            .iter()
            .enumerate()
            .filter_map(|(index, id)| self.running.contains_key(id).then_some((index, *id)))
            .collect();
        ids.sort_by_key(|(idx, id)| (self.assembly_priority(*id), *idx));
        ids.into_iter().map(|(_, id)| id).collect()
    }

    /// Returns the scheduling priority used during batch assembly; lower
    /// values are assembled first.
    ///
    /// Context ingestion (encoders and prefill) comes first; then token decode,
    /// verification and canvas denoising, image decoding, and cache installation;
    /// then denoising and the video and audio media calls; and last latent
    /// preparation, tensor transfers, cache export, and requests with no
    /// next call.
    pub(super) fn assembly_priority(&self, id: RequestId) -> u8 {
        match self.peek_next_call_variant(id) {
            Some(
                CallKind::Media(MediaCall::TextEncoding)
                | CallKind::Media(MediaCall::VisionEncoding)
                | CallKind::Media(MediaCall::LatentEncoding)
                | CallKind::Forward(ForwardMode::Prefill),
            ) => 0,
            Some(
                CallKind::Forward(
                    ForwardMode::Decode | ForwardMode::Verify | ForwardMode::TokenDenoising,
                )
                | CallKind::Media(MediaCall::ImageDecoding)
                | CallKind::Transfer(TransferMode::CacheInstall),
            ) => 1,
            Some(CallKind::Media(
                MediaCall::MediaReading
                | MediaCall::Denoising
                | MediaCall::VideoDecoding
                | MediaCall::AudioDecoding
                | MediaCall::VideoEncoding
                | MediaCall::AudioEncoding
                | MediaCall::Muxing,
            )) => 2,
            Some(
                CallKind::Media(MediaCall::LatentPreparation)
                | CallKind::Transfer(TransferMode::Tensor)
                | CallKind::Transfer(TransferMode::CacheExport),
            )
            | None => 3,
        }
    }

    /// Determines the next call kind without mutating request or resource state.
    ///
    /// A call chained onto the request's in-flight calls takes precedence.
    /// Otherwise, while the context is ingested, an input-image block the
    /// next context prefill needs is encoded first
    /// (`Scheduler::next_context_encode`); a block whose product the encoder
    /// cache holds still reports its encoder here. Otherwise the request's
    /// phase decides. Returns `None` when the request is not running, waits
    /// on its image-branch reservation, or has no call to issue.
    pub(super) fn peek_next_call_variant(&self, id: RequestId) -> Option<CallKind> {
        let st = self.running.get(&id)?;
        if st.image_reservation_pending {
            return None;
        }
        if let Some(variant) = self.pending_successor_code(id) {
            return Some(variant);
        }
        if st.phase == Phase::Prefill
            && let Some((_, step)) = self.next_context_encode(id)
        {
            return Some(generation::encoder_call(step));
        }
        Some(match st.phase {
            Phase::IngestState => CallKind::Forward(ForwardMode::Prefill),
            Phase::Prefill => CallKind::Forward(ForwardMode::Prefill),
            Phase::DecodeUnd => CallKind::Forward(ForwardMode::Decode),
            Phase::CloseCache => CallKind::Forward(ForwardMode::Prefill),
            Phase::PublishCache => CallKind::Transfer(TransferMode::CacheExport),
            Phase::LatentPreparation => CallKind::Media(MediaCall::LatentPreparation),
            Phase::Denoising if st.denoising.is_complete() => {
                CallKind::Media(MediaCall::ImageDecoding)
            }
            Phase::Denoising => CallKind::Media(MediaCall::Denoising),
            Phase::ImageDecoding => CallKind::Media(MediaCall::ImageDecoding),
            Phase::FeedbackEncode => {
                let feedback = &st.req.image_generation;
                feedback.feedback_source.as_ref()?;
                generation::encoder_call(
                    feedback
                        .feedback_encoders
                        .get(st.feedback_encoder_index)?
                        .encoder,
                )
            }
            Phase::FeedbackState => CallKind::Forward(ForwardMode::Prefill),
            Phase::Readout | Phase::Canvas => CallKind::Forward(ForwardMode::TokenDenoising),
            Phase::CommitCanvas => CallKind::Forward(ForwardMode::Prefill),
        })
    }
}

impl Scheduler {
    /// Finalizes batch statistics and pending-result ownership before submission.
    pub(super) fn finish_generation_batch(
        &mut self,
        mut batch: ExecutionBatch,
        submit_at: Instant,
    ) -> ExecutionBatch {
        // Command-only batches from `refill_executor` arrive with id zero.
        if batch.id == 0 {
            batch.id = self.inflight.next_batch_id();
        }

        // Snapshot scheduler gauges at the batch boundary before ownership moves
        // into the executor.
        self.stats
            .general
            .peak_calls
            .fetch_max(batch.requests.len(), Ordering::Relaxed);
        self.stats.general.steps.fetch_add(1, Ordering::Relaxed);
        self.stats
            .general
            .running
            .store(self.running_request_count(), Ordering::Relaxed);
        self.stats
            .general
            .pending
            .store(self.pending_request_count(), Ordering::Relaxed);
        self.stats
            .prefix_cache
            .free_units
            .store(self.storage.free_units(), Ordering::Relaxed);
        self.inflight.register_pending_batch(&batch, submit_at);
        batch
    }

    /// Returns whether generated output satisfies the trigger.
    pub(super) fn generated_trigger_matches(st: &RequestState) -> bool {
        st.req
            .image_generation
            .trigger
            .matches_generated(&st.generated_token_ids)
    }

    /// Returns whether direct input satisfies the trigger.
    pub(super) fn direct_trigger_matches(st: &RequestState, token_id: u32) -> bool {
        st.req.image_generation.trigger.direct_token() == Some(token_id)
    }

    /// Returns the next token fed back into generation.
    pub(super) fn feedback_next_token(&self, id: RequestId) -> Option<u32> {
        let next = self
            .running
            .get(&id)?
            .req
            .image_generation
            .feedback_next_token;
        match next {
            uniserve_core::FeedbackNextToken::None => None,
            uniserve_core::FeedbackNextToken::Bos => Some(self.ctrl.bos),
            uniserve_core::FeedbackNextToken::EndOfImage => Some(self.ctrl.end_of_image),
            uniserve_core::FeedbackNextToken::Token { token_id } => Some(token_id),
        }
    }

    /// Returns whether the prompt itself matches the image-generation trigger
    /// and the request can open an image branch; `false` for a request that
    /// is not running.
    pub(super) fn prefilled_gen_trigger(&self, id: RequestId) -> bool {
        let Some(st) = self.running.get(&id) else {
            return false;
        };
        st.can_open_gen_branch()
            && st
                .req
                .image_generation
                .trigger
                .matches_generated(&st.req.prompt_token_ids)
    }
}
