//! Batch selection, call planning, and physical params assembly.
//!
//! Each pass selects a batch kind, applies sequence and token budgets, and
//! emits at most one planned call per eligible request.
//!
//! This is the generation path: token requests in `running`.
//! `Scheduler::schedule_batches` alternates it with the video media path.
//! A pass schedules each request's next call from its accepted state plus the
//! calls still in flight (`scheduled_token_lengths`), so a successor can be
//! planned before its predecessor resolves. Each planned call then reserves its storage, is
//! registered as in flight, and joins the pass's batch for its phase and
//! component: one batch per `(phase, component)`, each a single numerical call
//! on one component, because a worker's batch carries one phase.
//!
//! Token calls and media calls are planned in the same pass: a token call by
//! the next-call planner, a media call of a video request by
//! `plan_media_call`, which owns that flow's numerical extent, decoder ranges,
//! and worker admission. The output names both, so one policy decides the whole
//! step.
//!
//! A pass's prefill calls, and likewise its decode calls, travel as one
//! numerical call, bounded by the batch's call and token budgets and the
//! workers' pools; a worker replays a captured graph for a call one holds
//! and runs any other eagerly. A context prefill with vision blocks stages
//! several forward rows; a pass's prefill rows, like its canvas rows, stay
//! within the rows a worker stages per call (`PassRows`).

mod call;
pub(crate) mod columns;
pub(crate) mod next;
mod slots;

use super::*;
use crate::views;
use ipc::{ForwardMode, MediaCall};
use scheduling::guest::{DispatchOutput, PhaseId, PhaseKind};

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
    /// Executes one step's output (design §4.4).
    ///
    /// The output's work, in the policy's order, becomes this step's calls: the
    /// order of the work list is the decision, and the mechanism schedules each
    /// named phase of the named request, clipping it to its own hard limits
    /// (the pass's rows, the request's storage, the transfer capacity). A
    /// request the output refills gets a refill chunk after the planned work,
    /// which is a prefill call that writes the draft cache.
    ///
    /// A request whose call does not fit this step keeps nothing: the work that
    /// does not fit waits for the next step, as a dispatch output over the free storage
    /// does.
    pub(super) fn assemble_output(&mut self, output: &DispatchOutput) -> Vec<ExecutionBatch> {
        let refill = views::phase_id(PhaseKind::Refill);
        let mut work: Vec<(RequestId, PhaseId, Option<u32>)> = output
            .work
            .iter()
            .map(|work| (RequestId(work.request.0), work.phase, Some(work.tokens)))
            .collect();
        work.extend(
            output
                .refill
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
    fn place_call(&mut self, call: &Call) -> Option<dispatch::Choice> {
        if self.dispatch.resident(call.request_key) {
            return Some(dispatch::Choice::Existing);
        }
        let replicas = views::replicas(self, call.request_key, call.kind);
        if replicas.is_empty() {
            return Some(dispatch::Choice::Existing);
        }
        let targets = views::replica_targets(self, call.request_key, call.kind);
        let view = views::call_view(call.request_key, call.kind);
        let mut policy = self.policy.take()?;
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
        target.map(dispatch::Choice::Target)
    }

    /// Charges actual token work of a token request's call.
    ///
    /// A verify call costs its input tokens plus one, and every other call
    /// its `max_tokens` bound.
    fn call_token_cost(&self, call: &Call) -> usize {
        match call.kind {
            CallKind::Forward(ForwardMode::Verify) => 1 + generation::verify_drafts(call).len(),
            _ => call.bounds.max_tokens as usize,
        }
    }

    /// Reserves persistent buffers and transfer capacity for one call.
    ///
    /// Returns the call's output buffer spans, in `Call::buffer_outputs`
    /// order, or `None` when the call cannot run now: the request is not
    /// running, `Dispatch::worker_target` finds no worker, or a buffer
    /// allocation fails.
    ///
    /// Buffer spans reserved before a failure are returned to the pool, but a
    /// failure is not free of side effects: a failed buffer allocation evicts
    /// one unpinned encoder-cache product, if any, and queues a `Free` for its
    /// buffer.
    fn reserve_generation_resources(&mut self, call: &Call) -> Option<Vec<BufferSpan>> {
        let id = call.request_key.request_id;
        let request_key = self
            .requests
            .tokens(&id)
            .map(|state| RequestKey::new(self.engine_id, id, state.request_epoch))?;
        self.dispatch.worker_target(
            &self.worker,
            &self.info,
            request_key,
            call.kind,
            &dispatch::Choice::Existing,
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

        Some(buffer_allocations)
    }

    /// Selects compatible call kinds within the step's token and sequence
    /// budgets, in the output's order.
    ///
    /// Walks the output's work until `max_batch` calls are selected or the token
    /// budget is spent. Every selected call is registered in flight before this
    /// returns. A failed call preparation returns no batches; most such
    /// failures latch engine-fatal first.
    fn assemble_batch(
        &mut self,
        work: &[(RequestId, PhaseId, Option<u32>)],
    ) -> Vec<ExecutionBatch> {
        // One batch per (phase, component): a batch is one numerical call of one
        // phase on one component, so a rank receives a homogeneous group and
        // the worker's phase column names it exactly. The output's order is the
        // order the first call of each pair was selected in.
        let mut kind_order: Vec<(PhaseId, String)> = Vec::new();
        // Each batch beside its calls' device layouts, index for index with
        // its requests, which `attach_columns` writes the device batch from.
        let mut kind_batches: HashMap<
            (PhaseId, String),
            (ExecutionBatch, Vec<columns::CallLayout>),
        > = HashMap::new();
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
            context: row_capacity.min(self.config.max_batch.get()),
        };
        // What the media calls in flight occupy — their component's media
        // units and the host task pools preparing their units — built when the
        // pass meets its first media call and charged with each one it selects.
        let mut media_occupancy: Option<super::execution::Occupancy> = None;
        // A media request's worker admission, queued by `plan_media_call`:
        // it travels with the batch of the first call of that request in this
        // pass.
        let mut admissions: Vec<NewRequest> = Vec::new();
        let mut selected = 0usize;
        for (id, phase, wanted) in work.iter().copied() {
            if selected >= self.config.max_batch.get() {
                break;
            }
            // The phase names the call kind this entry runs. A phase this epoch
            // has no call for runs nothing, and says so: an entry the policy
            // named and this loop discards is work the epoch believes it
            // planned, and a pass that names nothing every step is a stall no
            // other layer reports.
            let Some(kind) = views::phase_kind(phase).and_then(views::phase_call) else {
                tracing::debug!(
                    request = id.0,
                    phase = phase.0,
                    "a planned entry names a phase this epoch has no call for"
                );
                continue;
            };
            // A spent token budget ends the pass's token work; a media call
            // draws no tokens and still runs.
            if budget == 0 && !matches!(kind, CallKind::Media(_)) {
                break;
            }
            let refill = views::phase_kind(phase) == Some(PhaseKind::Refill);
            if !self.can_schedule_next(id) {
                continue;
            }
            // A request whose end is decided schedules nothing, and one that is
            // not running is one this pass must not serve.
            let cancelled = self
                .requests
                .get(&id)
                .is_none_or(|request| !request.admitted() || request.status.is_finished());
            if cancelled {
                continue;
            }
            // The denoiser is held exclusively: a request holds it while one
            // of its denoising calls is in flight, so denoising steps of
            // different requests are never in flight together, in any pass.
            // Between two steps of one request, another request may take it.
            if kind == CallKind::Media(MediaCall::Denoising)
                && self.inflight.denoiser_held_by_other(id)
            {
                continue;
            }
            // A media call is planned by the media flow, not by the token
            // planner: the call's numerical extent, its decoder ranges, and its
            // worker admission are that flow's. Its batch is made first, since
            // the call's id names it, and the component is known before
            // planning because a media call's kind fixes it.
            // A media request's call is planned by the media flow. A token
            // request's own encoder call — an input image's block — is
            // planned by the generation flow
            // below, which knows the context segment it serves: routing it
            // here found no media state and planned nothing, so the request
            // held the encode phase and never advanced.
            if let CallKind::Media(media_call) = kind
                && let Some(state) = self.media_state(id)
            {
                // A request offers nothing further while the call carrying its
                // worker admission is in flight.
                if state.admission_state == WorkerRegistration::InFlight
                    || !self.video_ready_calls(state).contains(&media_call)
                {
                    continue;
                }
                let request_key = state.request_key;
                let units = self.call_units(state, media_call);
                let demand = self.media_demand(request_key, media_call, units);
                let occupancy = media_occupancy.get_or_insert_with(|| self.media_occupancy());
                if !occupancy.admits(&demand, id, self)
                    || !self.call_has_queue_capacity(request_key, media_call)
                {
                    self.record_domain_backpressure(kind);
                    continue;
                }
                let Some(component) = self.info.media_components.get(&media_call).cloned() else {
                    continue;
                };
                let key = (phase, component);
                let (batch, layouts) = kind_batches.entry(key.clone()).or_insert_with(|| {
                    kind_order.push(key);
                    (
                        ExecutionBatch::new(self.inflight.next_batch_id(), Vec::new(), Vec::new()),
                        Vec::new(),
                    )
                });
                let Ok(request_index) = u32::try_from(batch.requests.len()) else {
                    self.invariant_broken("a batch's request count fits the IPC index");
                    return Vec::new();
                };
                let planned = self.plan_media_call(
                    id,
                    media_call,
                    CallId::new(batch.id, request_index),
                    &mut admissions,
                );
                match planned {
                    Ok(planned) => {
                        // A media call states no token rows: its batch's device layout is the
                        // computation's own (`attach_media`).
                        batch.requests.push(planned);
                        layouts.push(columns::CallLayout::default());
                    }
                    Err(invariant) => {
                        self.invariant_broken(invariant);
                        return Vec::new();
                    }
                }
                occupancy.occupy(&demand, id);
                selected += 1;
                continue;
            }
            // Build one call when its exact resident resources fit. A prefill
            // chunk draws on the prefill budget a warming epoch bounds, and its
            // size is the output's: the mechanism clips it only to what is left.
            let writes_prefix = batch_kind(kind) == BatchKind::Prefill;
            let mut call_budget = if writes_prefix {
                budget.min(prefill_budget)
            } else {
                budget
            };
            if writes_prefix && let Some(tokens) = wanted {
                call_budget = call_budget.min(tokens as usize);
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
                    // The request's state moved past the phase the output named:
                    // the output is executed as far as the state allows.
                    continue;
                }
                // A generation request's own encoder call — an input image's
                // block — prepares its
                // inline bytes on its rank's host task pool and holds those
                // tasks until the result returns. The pass charges it at the
                // first such call, as the media flow charges a media request's
                // calls.
                if let CallKind::Media(media_call) = call.kind {
                    let demand = self.image_demand(call.request_key, media_call);
                    let occupancy = media_occupancy.get_or_insert_with(|| self.media_occupancy());
                    if !occupancy.admits(&demand, id, self) {
                        self.record_domain_backpressure(kind);
                        continue;
                    }
                    occupancy.occupy(&demand, id);
                }
                // A context prefill schedules within the pass's context rows;
                // otherwise it waits for a later pass, as a call without
                // resources does below.
                let prefill = call.kind == CallKind::Forward(ForwardMode::Prefill);
                if prefill && rows_left.context == 0 {
                    if let Some(state) = self.requests.tokens_mut(&id) {
                        state.num_cache_units_sent = state
                            .num_cache_units_sent
                            .saturating_sub(u64::from(call.bounds.max_cache_units));
                    }
                    continue;
                }
                let scheduled_us = crate::clock::now_monotonic_us();

                // Only a call that obtains its storage is charged to the
                // pass. A call that cannot run now leaves the budget to later
                // requests; in particular, requests waiting for the guidance-prefix
                // row must not spend the pass before the request that holds
                // the row, which is the one whose steps release it.
                let Some(reserved_buffers) = self.reserve_generation_resources(&call) else {
                    self.record_domain_backpressure(call.kind);
                    // Planning a forward advanced `num_cache_units_sent` past its
                    // fresh units; roll it back so the next output declares them.
                    if let Some(state) = self.requests.tokens_mut(&id) {
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
                if call.canvas.is_some() {
                    rows_left.canvas = rows_left.canvas.saturating_sub(1);
                } else if prefill {
                    rows_left.context = rows_left.context.saturating_sub(1);
                }

                // Identify the call by its batch and row, and stamp the
                // coordinates projected through the request's in-flight calls.
                let key = (phase, call.component.clone());
                let (batch, layouts) = kind_batches.entry(key.clone()).or_insert_with(|| {
                    kind_order.push(key);
                    (
                        ExecutionBatch::new(self.inflight.next_batch_id(), Vec::new(), Vec::new()),
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
                    .requests
                    .tokens(&id)
                    .map(|state| state.finish_token_ids.clone())
                    .unwrap_or_default();
                let engine_id = self.engine_id;
                let mut admitted = false;
                // A handed-off request's first call admits it on the decode
                // group, with the token it generated on the prefill group,
                // which its penalties count.
                let hand_off = self
                    .requests
                    .tokens(&id)
                    .is_some_and(|st| st.hand_off == generation::HandOff::Copied);
                if let Some(st) = self.requests.tokens_mut(&id)
                    && (!st.worker_registered || hand_off)
                {
                    let request_key = RequestKey::new(engine_id, id, st.request_epoch);
                    let mut generated_token_ids = st.req.generated_token_ids.clone();
                    if hand_off {
                        generated_token_ids.push(st.next_token);
                        // The decode group's slot holds none of the request's sampling state yet:
                        // its first token batch there starts it.
                        st.sampling_resident = false;
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
                                GenerationParams {
                                    sampling: st.req.sampling.clone(),
                                    finish_token_ids,
                                    initial_position: st.num_computed_prompt_tokens,
                                    generated_token_ids,
                                    adapter: st.req.adapter,
                                },
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
                    scheduled_us,
                    batch,
                    layouts,
                );
                if prepared.is_none() {
                    // Most preparation failures latch engine-fatal first.
                    return Vec::new();
                }
                selected += 1;
            }
        }
        // A media request's admission travels with the batch holding the first
        // call of that request in this pass, so the request is registered on
        // the worker that runs its first call.
        for request in admissions {
            let owner = kind_order
                .iter()
                .find(|key| {
                    kind_batches[*key]
                        .0
                        .requests
                        .iter()
                        .any(|(call, _)| call.request_key == request.request_key)
                })
                .cloned();
            let Some(owner) = owner else {
                self.invariant_broken("an admitted media request has a selected call");
                return Vec::new();
            };
            if let Some((batch, _)) = kind_batches.get_mut(&owner) {
                batch.commands.push(BatchCommand::Start {
                    request: Box::new(request),
                });
            }
        }
        kind_order.retain(|kind| {
            kind_batches
                .get(kind)
                .is_some_and(|(batch, _)| !batch.requests.is_empty())
        });
        let Some(last) = kind_order.last().cloned() else {
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
        let prompt_only = kind_order.iter().all(|(phase, _)| {
            views::phase_kind(*phase)
                .and_then(views::phase_call)
                .is_some_and(|kind| batch_kind(kind) == BatchKind::Prefill)
        });
        let commands = if prompt_only {
            let requests = kind_order
                .iter()
                .flat_map(|kind| kind_batches[kind].0.requests.iter())
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
            let Some((mut batch, layouts)) = kind_batches.remove(&kind) else {
                self.invariant_broken("every selected phase and component has its batch");
                return Vec::new();
            };
            if kind == last {
                batch.commands.extend(commands.iter().cloned());
            }
            let mut batch = self.finish_generation_batch(batch, submit_at);
            // One batch is one computation, and the two families stage differently: a token batch
            // is typed columns the mechanism writes, and a media batch is the request's own stated
            // parameters beside the computation its kernels were captured for.
            if media_batch(&batch) {
                self.attach_media(&mut batch);
            } else {
                self.attach_columns(&mut batch, layouts);
                self.note_state_resident(&batch);
            }
            batches.push(batch);
        }
        batches
    }

    /// Attaches the batch's columns in the layout the kernels read (design
    /// §4.3): the batch carries them, so the submission hands a worker one
    /// value and nothing re-derives the layout from the calls.
    ///
    /// `max_kv_len` is the batch's worker group's `max_model_len`: the longest
    /// sequence the model was built for, not the longest this batch carries,
    /// because a KV view's shape must not move between steps.
    ///
    /// A fact the mechanism cannot state leaves the batch without columns: an
    /// unusable page shape, a worker group the mechanism has no info for, or
    /// a call whose request state is gone. Such a batch is not servable, which
    /// is what its `None` says; it is never filled with plausible values.
    fn attach_columns(&self, batch: &mut ExecutionBatch, mut layouts: Vec<columns::CallLayout>) {
        // A batch that leaves without columns is dropped: the calls it names are
        // never submitted and the step serves nothing. Nothing downstream can
        // tell that from a step the policy planned empty, so each refusal names
        // itself at debug level (`RUST_LOG=scheduler::scheduler::batching=debug`),
        // which is the one way an epoch that dispatches nothing says why.
        let Some(cache) = self.storage.cache() else {
            tracing::debug!("a batch is dropped: this scheduler holds no cache to address");
            return;
        };
        let Some(group) = cache.groups().first() else {
            tracing::debug!("a batch is dropped: the cache kv_cache states no group");
            return;
        };
        let (Ok(block_size), Ok(units_per_page)) = (
            u32::try_from(group.block_size),
            u32::try_from(group.units_per_page),
        ) else {
            tracing::debug!(
                "a batch is dropped: the page shape {}, {} does not fit u32",
                group.block_size,
                group.units_per_page
            );
            return;
        };
        // A page of several units holds different layers' columns in each unit, so one slot
        // number per token addresses only one of them: the kernels index one unit per page, and a
        // multi-unit page would need a slot column per unit.
        if units_per_page != 1 {
            tracing::debug!(
                "a batch is dropped: a page of {units_per_page} cache units has no single slot per \
                 token"
            );
            return;
        }
        // The attention descriptor's static bound: the worker states it once,
        // since a KV view's shape must not move per batch.
        let max_kv_len = self.worker.info().max_model_len;
        if max_kv_len == 0 {
            tracing::debug!(
                "a batch is dropped: the worker states no model length, so a KV view has no \
                 static bound to be shaped to"
            );
            return;
        }
        if layouts.len() != batch.requests.len() {
            tracing::debug!("a batch is dropped: its calls and their layouts disagree");
            return;
        }
        // A model that keeps a recurrent state per request slot has a state group, whose pages the
        // columns name the checkpoints and restores of.
        let state_pages = cache
            .groups()
            .iter()
            .position(|group| group.state())
            .and_then(|group_id| {
                let shape = &cache.groups()[group_id];
                Some(columns::StatePages {
                    group_id: u32::try_from(group_id).ok()?,
                    block_size: u32::try_from(shape.block_size).ok()?,
                    units_per_page: u32::try_from(shape.units_per_page).ok()?,
                })
            });
        let mut resident = Vec::with_capacity(batch.requests.len());
        let mut fresh = Vec::with_capacity(batch.requests.len());
        let mut inputs = Vec::with_capacity(batch.requests.len());
        let mut sampling = Vec::with_capacity(batch.requests.len());
        let mut draws = Vec::with_capacity(batch.requests.len());
        let mut tables = Vec::with_capacity(batch.requests.len());
        // Mutable, because the resolved table is written back to the layout: the worker
        // installs the table the batch states, so a batch that stated only the wire's delta left a
        // decode's row uninstalled and its gather read whatever the slot held before. One value,
        // one statement - the table this batch addresses.
        for ((call, _), placed) in batch.requests.iter().zip(&mut layouts) {
            let Some(state) = self.requests.tokens(&call.request_key.request_id) else {
                tracing::debug!(
                    "a batch is dropped: {:?} names a request this scheduler is not running",
                    call.request_key.request_id
                );
                return;
            };
            // A decode's rows embed the token the request's own state continues
            // from; every other token call states its tokens itself.
            let tokens: Vec<i32> = match call.kind {
                ipc::CallKind::Forward(ipc::ForwardMode::Decode) => (0..placed.rows.len())
                    .map(|_| i32::try_from(state.next_token).unwrap_or(-1))
                    .collect(),
                _ => call
                    .input_token_ids
                    .iter()
                    .map(|token| i32::try_from(*token).unwrap_or(-1))
                    .collect(),
            };
            inputs.push(tokens);
            resident.push(state.state_resident);
            // A slot that does not hold the request's sampling state starts it at the tokens the
            // request accepted so far: a decode group's first batch after a hand-off, and a
            // migrated or preempted request's first batch after its admission, continue penalty
            // counts the request began elsewhere.
            fresh.push((!state.sampling_resident).then_some(state.generated_token_ids.as_slice()));
            sampling.push(&state.req.sampling);
            draws.push(state.sampling_position);
            // The layout addresses the request's current cache tables. A call
            // declares a table on the wire only when it changed, so the
            // declaration is a delta and not the layout: a decode whose pages
            // did not move declares none, and the batch it travels in still
            // needs the table it writes through.
            // The declaration names the groups whose table changed, and every group the columns
            // address keeps its current table: a hybrid model's state group changes on other calls
            // than its K/V group does.
            let declared = placed.block_tables.clone();
            let resolved = if declared.is_empty() || state_pages.is_some() {
                let Some(pool) = state.request_pool_idx() else {
                    tracing::debug!("a batch is dropped: a request holds no pool row");
                    return;
                };
                let current: Vec<IpcBlockTable> = match state.block_tables() {
                    Some(tables) => tables
                        .iter()
                        .map(|table| IpcBlockTable {
                            request_pool_idx: pool,
                            group_id: u32::try_from(table.group_id()).unwrap_or(u32::MAX),
                            start_page: u32::try_from(table.start_page()).unwrap_or(u32::MAX),
                            unit_ids: table.unit_ids(),
                            allocated_tokens: u32::try_from(table.capacity_tokens())
                                .unwrap_or(u32::MAX),
                        })
                        .collect(),
                    None => Vec::new(),
                };
                current
                    .into_iter()
                    .map(|table| {
                        declared
                            .iter()
                            .rev()
                            .find(|stated| stated.group_id == table.group_id)
                            .cloned()
                            .unwrap_or(table)
                    })
                    .collect()
            } else {
                declared
            };
            // The layout now states the table this batch addresses, so the worker installs it
            // and the columns address it: a decode that declared nothing no longer travels with an
            // empty statement while its rows read the request's pages.
            placed.block_tables = resolved.clone();
            tables.push(resolved);
        }
        let calls: Vec<(&Call, &columns::CallLayout)> = batch
            .requests
            .iter()
            .map(|(call, _)| call)
            .zip(&layouts)
            .collect();
        let filled = columns::columns(
            &calls,
            &tables,
            &inputs,
            &sampling,
            &draws,
            block_size,
            max_kv_len,
            state_pages,
            &resident,
            &fresh,
        );

        let Some(filled) = filled else {
            let tokens = batch
                .requests
                .iter()
                .map(|(call, _)| call.input_token_ids.len())
                .sum::<usize>();
            tracing::debug!(
                "a batch is dropped: the fill refuses {} calls over {tokens} tokens at page {} ({max_kv_len} tokens of model)",
                batch.requests.len(),
                block_size,
            );
            return;
        };
        let Some(phase) = batch
            .requests
            .first()
            .and_then(|(call, _)| super::submission::wire_phase(call.kind))
        else {
            tracing::debug!("a batch is dropped: its first call names no device phase");
            return;
        };
        let shape = columns::batch_shape(&filled);
        // The batch as the worker reads it, written once: every call's tables, fresh units and
        // host page loads beside the columns, and the pages the batch publishes to the host KV
        // cache (`finish_generation_batch` attached them).
        let mut cache_units = Vec::new();
        let mut block_tables = Vec::new();
        let mut page_loads = Vec::new();
        for layout in layouts {
            cache_units.extend(layout.cache_units);
            block_tables.extend(layout.block_tables);
            page_loads.extend(layout.page_loads);
        }
        batch.device = Some(ipc::execution::Batch {
            id: ipc::execution::BatchId(batch.id),
            phase,
            // The group the mechanism placed the batch's calls on.
            group: placed_group(batch),
            calls: (0..batch.requests.len() as u64)
                .map(ipc::execution::CallId)
                .collect(),
            shape,
            columns: filled,
            // Stamped at submission: the step reads its generation then.
            generation: 0,
            cache_units,
            block_tables,
            page_loads,
            page_offloads: batch
                .page_offloads
                .as_ref()
                .map(|offloads| offloads.pages.clone())
                .unwrap_or_default(),
            latent_restores: Vec::new(),
            host_outputs: Vec::new(),
            images: Vec::new(),
            // Stated at submission, which reads the batch's commands.
            frees: Vec::new(),
            media: None,
        });
    }

    /// Marks every request a token batch carries as holding its recurrent state in its slot: the
    /// batch's first call of a request restored it there (or started it from zero), so its later
    /// calls continue from the slot's banks and restore nothing.
    fn note_state_resident(&mut self, batch: &ExecutionBatch) {
        if batch.device.is_none() {
            return;
        }
        for (call, _) in &batch.requests {
            if let Some(state) = self.requests.tokens_mut(&call.request_key.request_id) {
                state.state_resident = true;
                // The batch started the request's sampling state on its worker's slot.
                state.sampling_resident = true;
            }
        }
    }

    /// Stamps a media batch with its computation and the facts the mechanism states per request
    /// (2.4, 4.5).
    ///
    /// The call kind is the output's - the phases this pass selected - and the extent is the
    /// worker's own answer: `Batches::media` names the bucket its runner planned the computation
    /// into, and a batch captured for one canvas serves no other, so the mechanism asks instead of
    /// deriving. A computation the worker states no bucket for, or a request whose state this
    /// scheduler no longer holds, leaves the batch without columns; it is then refused at
    /// submission by name rather than served at a canvas nothing captured.
    ///
    /// The per-request record is `ipc::execution::StagedRequest`: the request's conditioning
    /// exactly as the model family planned it, and the step, seed and canvas the request states.
    /// What the model prepares on the device - the encoded conditioning rows, the latent
    /// trajectory - is not here, because the mechanism cannot read it and a placeholder for it
    /// would be an invented tensor.
    fn attach_media(&self, batch: &mut ExecutionBatch) {
        let Some((first, _)) = batch.requests.first() else {
            return;
        };
        let ipc::CallKind::Media(media_call) = first.kind else {
            tracing::debug!(
                "a media batch is dropped: batch {} carries no media call",
                batch.id
            );
            return;
        };
        let Some(phase) = super::submission::wire_phase(first.kind) else {
            tracing::debug!(
                "a media batch is dropped: {:?} names no device phase",
                first.kind
            );
            return;
        };
        // Every media call states the same record: the request's conditioning and the step, seed,
        // canvas and frame count a call of its request is addressed by. A denoising step reads all
        // of it; a decode reads the request and its frames, and the track's decoder the request; the
        // host calls work at the request's canvas and frame count. What a call reads beyond the
        // record is a product, which its runner takes from the store under the request's name.
        //
        // A text encoding also reads its prompt, as the token columns an encoder stages.
        let prompts = if media_call == ipc::MediaCall::TextEncoding {
            let Some(columns) = self.prompt_columns(batch) else {
                return;
            };
            Some(columns)
        } else {
            None
        };
        let Ok(rows) = u32::try_from(batch.requests.len()) else {
            return;
        };
        let tokens = prompts.as_ref().map_or_else(
            || {
                batch
                    .requests
                    .iter()
                    .map(|(call, _)| call.input_token_ids.len())
                    .sum::<usize>()
            },
            |columns| columns.input_ids.len(),
        );
        let Ok(tokens) = u32::try_from(tokens) else {
            return;
        };
        // The batch's requests share a bucket, so the first states the shape it is found by.
        let extent = self.requests.video(&first.request_key.request_id).map_or(
            ipc::execution::Extent::None,
            |state| {
                let sampling = &state.request.sampling;
                ipc::execution::Extent::Request {
                    frames: sampling.num_frames,
                    width: sampling.width,
                    height: sampling.height,
                    text_rows: u32::try_from(state.request.prompt_token_ids.len())
                        .unwrap_or(u32::MAX),
                    condition_rows: sampling.condition_rows,
                }
            },
        );
        let mut stated = ipc::execution::Computation {
            call: media_call,
            rows,
            tokens,
            extent,
        };
        // The worker's answer is the authority on the extent: it is the bucket its own runner
        // planned the computation into, and a batch captured for one canvas serves no other.
        let Some(extent) = self.worker.media(phase, &stated) else {
            tracing::debug!(
                "a media batch is dropped: no device of this rank serves a {:?} computation of \
                 {rows} row(s) and {tokens} token(s) at phase {}",
                media_call,
                phase.0
            );
            return;
        };
        stated.extent = extent;
        let mut requests = Vec::with_capacity(batch.requests.len());
        for (call, placed) in &batch.requests {
            // A token request's vision encoding reads nothing of a video request's record: its
            // image travels with the call (`Batch::images`).
            if media_call == ipc::MediaCall::VisionEncoding
                && call.input_image.is_some()
                && self.requests.tokens(&call.request_key.request_id).is_some()
            {
                requests.push(ipc::execution::StagedRequest {
                    request: call.request_key,
                    conditioning: Vec::new(),
                    step: 0,
                    steps: 0,
                    seed: 0,
                    width: 0,
                    height: 0,
                    frames: 0,
                    handed: 0,
                    unit: 0,
                });
                continue;
            }
            let Some(state) = self.requests.video(&call.request_key.request_id) else {
                tracing::debug!(
                    "a media batch is dropped: {:?} names a request this scheduler is not running",
                    call.request_key.request_id
                );
                return;
            };
            requests.push(ipc::execution::StagedRequest {
                request: call.request_key,
                conditioning: state.request.conditioning.clone(),
                step: self.step_of(call.request_key.request_id, call.call_id, state),
                steps: state.denoising.steps(),
                seed: state.request.sampling.seed,
                width: state.request.sampling.width,
                height: state.request.sampling.height,
                frames: state.request.sampling.num_frames,
                handed: u32::try_from(call.inputs.len()).unwrap_or(u32::MAX),
                unit: placed.decode.as_ref().map_or(0, |range| range.cursor),
            });
        }
        let Some(variant) = stated.variant() else {
            tracing::debug!(
                "a media batch is dropped: no variant serves a {:?} computation",
                media_call
            );
            return;
        };
        batch.device = Some(ipc::execution::Batch {
            id: ipc::execution::BatchId(batch.id),
            phase,
            // The group the mechanism placed the batch's calls on.
            group: placed_group(batch),
            calls: (0..batch.requests.len() as u64)
                .map(ipc::execution::CallId)
                .collect(),
            shape: ipc::execution::Shape {
                rows: stated.rows,
                tokens: stated.tokens,
                variant,
            },
            // A text encoding reads its prompt as token columns. Every other media computation
            // reads none, so its batch carries none: the fields its kernels read are the model's
            // own, staged inside the runner.
            columns: prompts.unwrap_or_default(),
            // Stamped at submission: the step reads its generation then.
            generation: 0,
            cache_units: Vec::new(),
            block_tables: Vec::new(),
            page_loads: Vec::new(),
            page_offloads: Vec::new(),
            // A resumed request's latent preparation opens at the stored trajectory its restore
            // pinned, which the denoiser copies in before the call completes.
            latent_restores: batch
                .requests
                .iter()
                .filter_map(|(call, _)| {
                    call.latent_restore
                        .clone()
                        .map(|restore| (call.request_key, restore))
                })
                .collect(),
            // An encoder call's output in the host encoder pool: read from it, or written to it.
            host_outputs: batch
                .requests
                .iter()
                .filter_map(|(call, _)| {
                    call.host_output
                        .clone()
                        .map(|output| (call.request_key, output))
                })
                .collect(),
            // A token request's vision encoding reads its image and writes the feature product
            // its call declares.
            images: batch
                .requests
                .iter()
                .filter_map(|(call, _)| {
                    Some(ipc::execution::ImageEncoding {
                        request: call.request_key,
                        image: call.input_image.clone()?,
                        output: call.encoder_output.as_ref()?.buffer_id(),
                    })
                })
                .collect(),
            // Stated at submission, which reads the batch's commands.
            frees: Vec::new(),
            media: Some(ipc::execution::MediaColumns::staged(stated, &requests)),
        });
    }

    /// The step a request's call covers: the steps its trajectory has completed and the denoising
    /// calls submitted before this one.
    ///
    /// A request's steps are scheduled ahead of their results, so what has completed is behind what
    /// a call in the pipeline runs: two calls stating the completed count would run one step twice.
    ///
    /// The calls ahead are folded as the trajectory sees them (`Denoising::scheduled`): a latent
    /// preparation still in flight opens it at the step it restores, so the steps after a restore
    /// are numbered from the stored step even before the preparation's result arrives.
    fn step_of(&self, id: RequestId, call_id: CallId, state: &VideoState) -> u32 {
        state.denoising.scheduled(
            self.inflight
                .pending_calls
                .get(&id)
                .into_iter()
                .flatten()
                .take_while(|pending| pending.call.call_id != call_id)
                .filter_map(|pending| match pending.call.kind {
                    CallKind::Media(media_call) => Some((
                        media_call,
                        pending
                            .call
                            .latent_restore
                            .as_ref()
                            .map_or(0, |restore| restore.steps),
                    )),
                    _ => None,
                }),
        )
    }

    /// The token columns of a text encoding batch: each request's prompt, packed, one row per request.
    ///
    /// `None` when the batch names a request this scheduler is not running, which the caller has
    /// the same answer for as for any other media batch it cannot state.
    fn prompt_columns(&self, batch: &ExecutionBatch) -> Option<ipc::execution::BatchColumns> {
        let mut columns = ipc::execution::BatchColumns::default();
        columns.rows.query_offsets.push(0);
        for (call, _) in &batch.requests {
            let state = self.requests.video(&call.request_key.request_id)?;
            let prompt = &state.request.prompt_token_ids;
            let length = i32::try_from(prompt.len()).ok()?;
            columns.input_ids.extend(prompt.iter().map(|id| *id as i32));
            columns.rows.positions.extend(0..length);
            columns.rows.seq_lens.push(length);
            let end = i32::try_from(columns.input_ids.len()).ok()?;
            columns.rows.query_offsets.push(end);
            columns.rows.logits_rows.push(end - 1);
        }
        Some(columns)
    }

    /// Determines the next call kind without mutating request or resource state.
    ///
    /// A call chained onto the request's in-flight calls takes precedence.
    /// Otherwise, while the context is ingested, an input-image block the
    /// next context prefill needs is encoded first
    /// (`Scheduler::next_context_encode`); a block whose product the encoder
    /// cache holds still reports its encoder here. Otherwise the request's
    /// phase decides. Returns `None` when the request is not running or has
    /// no call to issue.
    pub(super) fn token_next_call(&self, id: RequestId) -> Option<CallKind> {
        let st = self.requests.tokens(&id)?;
        if let Some(variant) = self.pending_successor_code(id) {
            return Some(variant);
        }
        if st.phase == Phase::Prefill
            && let Some((_, step)) = self.next_context_encode(id)
        {
            return Some(CallKind::Media(step.into()));
        }
        Some(match st.phase {
            Phase::IngestState => CallKind::Forward(ForwardMode::Prefill),
            Phase::Prefill => CallKind::Forward(ForwardMode::Prefill),
            // A speculating deployment's decode step verifies the drafts its
            // last forward proposed: the call the mechanism builds from
            // `DecodeUnd` is a verification, and the phase a policy reads must
            // be the one that call runs in, or the batch is planned under one
            // kind and built under another and is dropped unnamed.
            Phase::DecodeUnd if self.speculates() => CallKind::Forward(ForwardMode::Verify),
            Phase::DecodeUnd => CallKind::Forward(ForwardMode::Decode),
            // The hand-off copies the request's cache with no call.
            Phase::HandOff => return None,
            Phase::Canvas => CallKind::Forward(ForwardMode::TokenDenoising),
            Phase::CommitCanvas => CallKind::Forward(ForwardMode::Prefill),
        })
    }
}

/// Whether `batch` carries a media computation (2.4).
///
/// One batch is one computation, so the family is the first call's and every call of a media batch
/// is a media call: a batch that mixed them is refused at submission, and deciding the family
/// before staging is what keeps a media battery out of the token layout.
fn media_batch(batch: &ExecutionBatch) -> bool {
    batch
        .requests
        .first()
        .is_some_and(|(call, _)| matches!(call.kind, CallKind::Media(_)))
}

/// The worker group `batch` runs on: the one its calls were placed on (dispatch forms a batch per
/// group, and submission refuses one whose calls name two).
fn placed_group(batch: &ExecutionBatch) -> String {
    batch
        .requests
        .first()
        .map(|(_, dispatch)| dispatch.worker.0.clone())
        .unwrap_or_default()
}

impl Scheduler {
    /// Finalizes batch statistics and pending-result ownership before submission.
    pub(super) fn finish_generation_batch(
        &mut self,
        mut batch: ExecutionBatch,
        submit_at: Instant,
    ) -> ExecutionBatch {
        // Batches of commands alone, which `schedule` forms, arrive with id zero.
        if batch.id == 0 {
            batch.id = self.inflight.next_batch_id();
        }

        // A text batch of one worker group carries the page offloads the
        // host KV cache reserved for that group: the worker copies them on
        // its copy stream beside its forward, and its output settles them.
        if let Some(kv_offload) = self.kv_offload.as_mut()
            && let Some((_, first)) = batch.requests.first()
            && batch.requests.iter().all(|(call, dispatch)| {
                dispatch.worker == first.worker
                    && call.component == DEFAULT_BLOCK
                    && matches!(
                        call.kind,
                        CallKind::Forward(ForwardMode::Prefill | ForwardMode::Decode)
                    )
            })
        {
            let worker = first.worker.clone();
            let pages = kv_offload.take_offloads(&worker, batch.id);
            if !pages.is_empty() {
                batch.page_offloads = Some(crate::worker::PageOffloads {
                    worker,
                    pages,
                    latents: Vec::new(),
                });
            }
        }

        // Snapshot scheduler gauges at the batch boundary before ownership moves
        // into the worker.
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
}
