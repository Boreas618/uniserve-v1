//! Batch selection, call planning, and physical params assembly.
//!
//! Each pass selects a batch kind, applies sequence and token budgets, and
//! emits at most one planned call per eligible request.
//!
//! This is the generation path: token requests in `running`, including the
//! image-generation and feedback phases of unified multimodal requests.
//! `Scheduler::schedule_batches` alternates it with the video media path.
//! A pass plans each request's next call from its accepted state plus the
//! calls still in flight (`scheduled_token_lengths`,
//! `num_scheduled_denoise_steps`), so a successor can be planned before its
//! predecessor resolves. Each planned call then reserves its storage, is
//! registered as in flight, and joins the pass's batch for its `CallKind`:
//! one batch per call kind, each a single numerical call on one component.
//!
//! The batch kind (`BatchKind`) only separates prefill from decode.
//! `BatchKind::Media` calls are eligible in every pass, and a pass with no
//! batch kind applies no filter.
//!
//! A pass's prefill calls, and likewise its decode calls, travel as one
//! numerical call, bounded by the batch's call and token budgets and the
//! workers' pools; a worker replays a captured graph for a call one holds
//! and runs any other eagerly. A context prefill with vision blocks stages
//! several forward rows; a pass's prefill rows, like its canvas rows, stay
//! within the rows a worker stages per call (`PassRows`).

mod call;
mod plan;
mod slots;

use super::*;
use crate::scheduler::views;
use ipc::{ForwardMode, MediaCall, TransferMode};
use scheduling::guest::{PhaseId, PhaseKind, Plan};

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
    /// Executes one step's plan (design §4.4).
    ///
    /// The plan's work, in the policy's order, becomes this step's calls: the
    /// order of the work list is the decision, and the mechanism plans each
    /// named phase of the named request, clipping it to its own hard limits
    /// (the pass's rows, the request's storage, the transfer capacity). A
    /// request the plan refills gets a refill chunk after the planned work,
    /// which is a prefill call that writes the draft cache.
    ///
    /// A request whose call does not fit this step keeps nothing: the work that
    /// does not fit waits for the next step, as a plan over the free storage
    /// does.
    pub(super) fn assemble_plan(&mut self, plan: &Plan) -> Vec<ExecutionBatch> {
        let refill = views::phase_id(PhaseKind::Refill);
        let mut work: Vec<(RequestId, PhaseId, Option<u32>)> = plan
            .work
            .iter()
            .map(|work| (RequestId(work.request.0), work.phase, Some(work.tokens)))
            .collect();
        work.extend(
            plan.refill
                .iter()
                .map(|request| (RequestId(request.0), refill, None)),
        );
        self.assemble_batch(&work)
    }

    /// The replica a planned call runs on: the mechanism's own residency when
    /// the request already ran, the policy's answer when it has not (design
    /// §4.4).
    ///
    /// The policy sees the candidates exactly as `views::replicas` shows them
    /// and answers one of their indices; `views::replica_targets` names the same
    /// workers in the same order, both built from one filter in `views.rs`, so
    /// an index becomes a worker without two candidate lists that could
    /// disagree. A policy that fails here fails as any decision does: the call
    /// waits this step and the previous policy decides the next one.
    fn place_call(&mut self, call: &Call) -> Option<placement::Choice> {
        if self.placement.resident(call.request_key) {
            return Some(placement::Choice::Existing);
        }
        let replicas = views::replicas(self, call.request_key, call.kind);
        if replicas.is_empty() {
            return Some(placement::Choice::Existing);
        }
        let targets = views::replica_targets(self, call.request_key, call.kind);
        let view = views::call_view(call.request_key, call.kind);
        let Some(mut policy) = self.policy.take() else {
            return None;
        };
        let answer = policy
            .place(&view, &replicas)
            .and_then(|index| mechanism::validate_place(&view, &replicas, index).map(|()| index));
        let target = match answer {
            Ok(index) => targets.get(index).map(|(worker, _)| worker.clone()),
            Err(error) => {
                let _ = self.policy_failed(policy, error);
                return None;
            }
        };
        self.policy = Some(policy);
        target.map(placement::Choice::Target)
    }

    /// Charges actual token work, including each CFG branch of a denoising step.
    ///
    /// A denoising call costs its latent units times its guidance branches;
    /// it costs nothing once its request has left `running`. A verify call
    /// costs its input tokens plus one, and every other call its `max_tokens`
    /// bound.
    fn call_token_cost(&self, call: &Call) -> usize {
        match call.kind {
            CallKind::Media(MediaCall::Denoising) => {
                let Some(state) = self.running.get(&call.request_key.request_id) else {
                    return 0;
                };
                usize::try_from(self.worker_image_latent_units_for(state).max(1))
                    .unwrap_or(usize::MAX)
                    .saturating_mul(usize::from(state.req.image.cfg_branch_count().max(1)))
            }
            CallKind::Forward(ForwardMode::Verify) => 1 + generation::verify_drafts(call).len(),
            _ => call.bounds.max_tokens as usize,
        }
    }

    /// Reserves persistent buffers, latent pages, and transfer capacity for one call.
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
        if call.kind == CallKind::Media(MediaCall::Denoising) && !self.ensure_guidance_prefix(id) {
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
            .worker_target(
                self.executor.as_ref(),
                &self.info,
                request_key,
                call.kind,
                &placement::Choice::Existing,
            )?;

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
            call.kind,
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

    /// Selects compatible call kinds within the step's token and sequence
    /// budgets, in the plan's order.
    ///
    /// Walks the plan's work until `max_batch` calls are selected or the token
    /// budget is spent. Every selected call is registered in flight before this
    /// returns. A failed call preparation returns no batches; most such
    /// failures latch engine-fatal first.
    fn assemble_batch(
        &mut self,
        work: &[(RequestId, PhaseId, Option<u32>)],
    ) -> Vec<ExecutionBatch> {
        // One batch per call kind: a batch is one numerical call on one
        // component, so a rank receives it as a single homogeneous group. The
        // component a call binds to follows from its kind, so the two
        // together name one group. Kinds keep the order in which their
        // first call was selected.
        let mut kind_order: Vec<CallKind> = Vec::new();
        let mut kind_batches: HashMap<CallKind, ExecutionBatch> = HashMap::new();
        let submit_at = Instant::now();
        // The per-step token budget is the binding limit; individual prefill
        // chunks are clipped to its remaining capacity.
        let mut budget: usize = self.config.max_num_batched_tokens;
        // Warming bounds only new prefix writes. Decode and unrelated
        // numerical calls keep their normal scheduling budgets.
        let mut prefill_budget = self.warming.prefill_tokens.unwrap_or(usize::MAX);
        // Context calls expand into one forward row per text/image segment.
        // The worker's input buffers and the pass's call bound limit those
        // numerical rows, independently of the number of requests batched.
        let row_capacity = (self.info.max_batch_calls.min(self.info.request_slots) as usize).max(1);
        let mut rows_left = PassRows {
            canvas: row_capacity,
            context: row_capacity.min(self.config.max_batch),
        };
        // Host task occupancy, built when the pass meets its first encoder
        // call carrying inline image bytes and charged with each one it
        // selects.
        let mut image_occupancy: Option<super::execution::Occupancy> = None;
        let mut selected = 0usize;
        for (id, phase, wanted) in work.iter().copied() {
            let id = id;
            if selected >= self.config.max_batch {
                break;
            }
            if budget == 0 {
                break;
            }
            // The phase names the call kind this entry runs. A phase this epoch
            // has no call for runs nothing.
            let Some(kind) = views::phase_kind(phase).and_then(views::phase_call) else {
                continue;
            };
            let refill = views::phase_kind(phase) == Some(PhaseKind::Refill);
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
            // The denoiser is held exclusively: a request holds it while one
            // of its denoising calls is in flight, so denoising steps of
            // different requests are never in flight together, in any pass.
            // Between two steps of one request, another request may take it.
            if kind == CallKind::Media(MediaCall::Denoising)
                && (self.inflight.denoiser_held_by_other(id)
                    || !self.guidance_prefix_is_schedulable(id))
            {
                continue;
            }
            // Build one call when its exact resident resources fit. A prefill
            // chunk draws on the prefill budget a warming epoch bounds, and its
            // size is the plan's: the mechanism clips it only to what is left.
            let writes_prefix = batch_kind(kind) == BatchKind::Prefill;
            let mut call_budget = if writes_prefix {
                budget.min(prefill_budget)
            } else {
                budget
            };
            if writes_prefix {
                if let Some(tokens) = wanted {
                    call_budget = call_budget.min(tokens as usize);
                }
            }
            if call_budget == 0 {
                continue;
            }
            let planned = if refill {
                self.plan_refill(id, call_budget)
            } else {
                self.next_generation_call(id, call_budget, rows_left)
            };
            if let Some(mut call) = planned {
                if call.kind != kind {
                    // The request's state moved past the phase the plan named:
                    // the plan is executed as far as the state allows.
                    continue;
                }
                // A context prefill plans within the pass's context rows;
                // otherwise it waits for a later pass, as a call without
                // resources does below.
                let prefill = call.kind == CallKind::Forward(ForwardMode::Prefill);
                if prefill && generation::prefill_rows(&call) > rows_left.context {
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
                // does, until those host task pools have room.
                let image_demand =
                    match call.kind {
                        CallKind::Media(media_call)
                            if call.input_image.is_some() || call.carried_image.is_some() =>
                        {
                            let Some(request_key) = self.running.get(&id).map(|state| {
                                RequestKey::new(self.engine_id, id, state.request_epoch)
                            }) else {
                                continue;
                            };
                            let demand = self.image_demand(request_key, media_call);
                            let occupancy =
                                image_occupancy.get_or_insert_with(|| self.media_occupancy());
                            if !occupancy.admits(&demand, id, self) {
                                self.record_domain_backpressure(call.kind);
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
                let planned_us = crate::clock::now_monotonic_us();

                // Only a call that obtains its storage is charged to the
                // pass. A call that cannot run now leaves the budget to later
                // requests; in particular, requests waiting for the guidance-prefix
                // row must not spend the pass before the request that holds
                // the row, which is the one whose steps release it.
                let Some(reserved_buffers) = self.reserve_generation_resources(&call) else {
                    self.record_domain_backpressure(call.kind);
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
                let cost = self.call_token_cost(&call);
                budget = budget.saturating_sub(cost);
                if prefill {
                    prefill_budget = prefill_budget.saturating_sub(cost);
                }
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
                let kind = call.kind;
                let batch = kind_batches.entry(kind).or_insert_with(|| {
                    kind_order.push(kind);
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
                // A hand-off's install admits the request on the decode
                // group, with the token it generated on the prefill group,
                // which its penalties count.
                let hand_off = call.kind == CallKind::Transfer(TransferMode::CacheInstall);
                if let Some(st) = self.running.get_mut(&id)
                    && (!st.worker_registered || hand_off)
                {
                    let request_key = RequestKey::new(engine_id, id, st.request_epoch);
                    let mut generated_token_ids = st.req.generated_token_ids.clone();
                    if hand_off {
                        generated_token_ids.push(st.next_token);
                    }
                    // Admission reserved the request row, and submission
                    // validated the parameters this admission carries.
                    let admission = st
                        .request_pool_idx()
                        .zip(u32::try_from(st.req.multimodal_inputs.images.len()).ok())
                        .map(|(request_pool_idx, input_images)| {
                            NewRequest::new(
                                request_key,
                                request_pool_idx,
                                Some(GenerationParams {
                                    sampling: st.req.sampling.clone(),
                                    negative_token_ids: st.req.negative_prompt_token_ids.clone(),
                                    finish_token_ids,
                                    initial_position: st.num_computed_prompt_tokens,
                                    generated_token_ids,
                                    adapter: st.req.adapter,
                                }),
                                st.req.generates_images().then(|| st.req.image.clone()),
                                input_images,
                            )
                        });
                    let Some(Ok(admission)) = admission else {
                        self.invariant_broken("a running request forms a valid worker admission");
                        return Vec::new();
                    };
                    if !hand_off {
                        st.worker_registered = true;
                        // Admission is the first request-state dependency.
                        st.last_state_call_id = CallId::default();
                        st.latest_token = None;
                        admitted = true;
                    }
                    batch.commands.push(BatchCommand::Start {
                        request: Box::new(admission),
                    });
                }
                // An image encoding reads or writes the output its request
                // looked up in the host encoder pools.
                self.attach_host_encoding(id, &mut call);
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
                if let (Some(demand), Some(occupancy)) = (&image_demand, image_occupancy.as_mut()) {
                    occupancy.occupy(demand, id);
                }
                selected += 1;
            }
        }
        kind_order.retain(|kind| {
            kind_batches
                .get(kind)
                .is_some_and(|batch| !batch.requests.is_empty())
        });
        let Some(&last) = kind_order.last() else {
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
        let prompt_only = kind_order
            .iter()
            .all(|kind| batch_kind(*kind) == BatchKind::Prefill);
        let commands = if prompt_only {
            let requests = kind_order
                .iter()
                .flat_map(|kind| kind_batches[kind].requests.iter())
                .map(|(call, _)| call.request_key.request_id)
                .collect::<HashSet<_>>();
            self.take_commands(|command| {
                requests.contains(&command.request_key().request_id)
                    || matches!(command, BatchCommand::Finish { .. })
            })
        } else {
            self.take_commands(|_| true)
        };
        let mut batches = Vec::with_capacity(kind_order.len());
        for kind in kind_order {
            let Some(mut batch) = kind_batches.remove(&kind) else {
                self.invariant_broken("every selected call kind has its batch");
                return Vec::new();
            };
            if kind == last {
                batch.commands.extend(commands.iter().cloned());
            }
            batches.push(self.finish_generation_batch(batch, submit_at));
        }
        batches
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
        // A context rebuild's image calls (`Scheduler::next_replay_call`).
        if let Some(image) = self
            .num_scheduled_prompt_tokens(id)
            .and_then(|scheduled| st.replay_image_due(scheduled))
        {
            return Some(match image.stage {
                generation::continuation::ReplayStage::Encode(index) => CallKind::Media(
                    st.req
                        .image_generation
                        .feedback_encoders
                        .get(index)?
                        .encoder
                        .into(),
                ),
                generation::continuation::ReplayStage::Close
                | generation::continuation::ReplayStage::Write(_) => {
                    CallKind::Forward(ForwardMode::Prefill)
                }
            });
        }
        if st.phase == Phase::Prefill
            && let Some((_, step)) = self.next_context_encode(id)
        {
            return Some(CallKind::Media(step.into()));
        }
        Some(match st.phase {
            Phase::IngestState => CallKind::Forward(ForwardMode::Prefill),
            Phase::Prefill => CallKind::Forward(ForwardMode::Prefill),
            Phase::DecodeUnd => CallKind::Forward(ForwardMode::Decode),
            Phase::HandOff if st.handoff_export.is_some() => {
                CallKind::Transfer(TransferMode::CacheInstall)
            }
            Phase::HandOff => CallKind::Transfer(TransferMode::CacheExport),
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
                CallKind::Media(
                    feedback
                        .feedback_encoders
                        .get(st.feedback_encoder_index)?
                        .encoder
                        .into(),
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

        // A text batch of one worker group carries the page offloads the
        // host KV cache reserved for that group: the worker copies them on
        // its copy stream beside its forward, and its output settles them.
        if let Some(kv_offload) = self.kv_offload.as_mut()
            && let Some((_, first)) = batch.requests.first()
            && batch.requests.iter().all(|(call, placement)| {
                placement.worker == first.worker
                    && call.component == DEFAULT_COMPONENT
                    && matches!(
                        call.kind,
                        CallKind::Forward(ForwardMode::Prefill | ForwardMode::Decode)
                    )
            })
        {
            let worker = first.worker.clone();
            let pages = kv_offload.take_offloads(&worker, batch.id);
            if !pages.is_empty() {
                batch.page_offloads = Some(crate::executor::PageOffloads {
                    worker,
                    pages,
                    latents: Vec::new(),
                });
            }
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
        // The units the cache holds change with its storage (a successor's
        // segments grow at its commit), so its usage counts them anew.
        self.stats
            .prefix_cache
            .num_units
            .store(self.storage.usable_units(), Ordering::Relaxed);
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
            crate::FeedbackNextToken::None => None,
            crate::FeedbackNextToken::Bos => Some(self.special_tokens.bos),
            crate::FeedbackNextToken::EndOfImage => Some(self.special_tokens.end_of_image),
            crate::FeedbackNextToken::Token { token_id } => Some(token_id),
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
