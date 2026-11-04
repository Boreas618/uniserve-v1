//! Batch submission, completion application, and allocation reclamation.
//!
//! This part of the scheduler sits between request state and the executor:
//!
//! - Refill and dispatch: `refill_executor` admits requests, asks the
//!   generation assembler (`assemble`) and the video media planner
//!   (`prepare_media_batches`) for batches while a rank queue has room, and
//!   `dispatch_submissions` hands them to the executor in order per worker,
//!   each after the batches producing its inputs.
//! - Video media scheduling: a video request runs the calls of its graph
//!   (`graph::VideoGraph`, built at admission from its task and conditions),
//!   offered by `ready_calls` as their inputs become available. Calls occupy
//!   their components' capacity on their workers and, for media reading,
//!   encoding and muxing, host task pools, tracked by the media occupancy,
//!   and `plan_media_call` builds one call with its products, buffer
//!   bindings, latent pages and reading components.
//! - Successor projection: queries such as `projected_coordinates`,
//!   `pending_successor_code` and `can_queue_successor` project a token
//!   request's accepted state through its in-flight calls, so the batching
//!   path can queue a successor before its predecessor's result returns.
//! - Result reconciliation: `apply_result` validates every returned call
//!   identity against the in-flight window before any completion is applied,
//!   stages completions, and applies them in request-local dependency order.
//! - Failure reconciliation: `on_executor_error` retires only the work a
//!   `WorkerFailure` invalidates. Any other executor error, and a worker
//!   failure classified as a control-plane invariant violation, is
//!   engine-fatal.
//!
//! Refill and dispatch live here; video media scheduling, successor
//! projection, result reconciliation and failure reconciliation live in
//! `media`, `projection`, `results` and `failure`.
//!
//! `Scheduler::run` drives these in production; `Scheduler::step` is a
//! single-tick driver for tests and examples.

mod failure;
mod media;
mod projection;
mod results;

use super::*;
use ipc::{BufferId, MediaCall};
use scheduling::PolicyError;
use scheduling::guest::{Plan, Room};

pub(super) use media::Occupancy;

/// The products a batch's calls publish.
fn batch_products(batch: &ExecutionBatch) -> impl Iterator<Item = BufferId> + '_ {
    batch
        .requests
        .iter()
        .flat_map(|(call, _)| call.output_buffers())
}

impl Scheduler {
    /// Dispatches queued work while preserving order at every shared destination
    /// and after every product it reads.
    /// The caller retains blocked destinations for the whole scheduling tick, so
    /// assembling more work cannot repeatedly retry an unavailable worker.
    ///
    /// Each queued batch is tried once per invocation; a batch that cannot be
    /// submitted goes back to the end of the queue. Returns whether any batch
    /// was submitted, or `true` after a terminal submission failure, which is
    /// handed to `on_executor_error` and stops dispatching.
    fn dispatch_submissions(
        &mut self,
        blocked_workers: &mut HashSet<crate::WorkerGroupId>,
    ) -> bool {
        let mut progressed = false;
        // Products of the batches held back in this pass. A call may be
        // submitted before its inputs' producers complete, since its worker
        // waits on them, but not before they are submitted, which is when the
        // executor learns where each product is published. A consumer is
        // queued after its producer and held batches keep their order, so the
        // producer is always tried first.
        let mut held_products = HashSet::new();
        let pending_count = self.inflight.pending_submissions.len();
        for _ in 0..pending_count {
            let Some(batch) = self.inflight.pending_submissions.pop_front() else {
                break;
            };
            let targets = self.placement.batch_workers(&batch);
            let reads_held = batch.requests.iter().any(|(call, _)| {
                call.input_buffers()
                    .any(|input| held_products.contains(&input))
            });
            if reads_held || !targets.is_disjoint(blocked_workers) {
                // A batch waiting on one destination also orders later work at
                // its other destinations and the work reading its products;
                // independent work can proceed.
                blocked_workers.extend(targets);
                held_products.extend(batch_products(&batch));
                self.inflight.pending_submissions.push_back(batch);
                continue;
            }
            match self.executor.submit(batch) {
                Ok(()) => progressed = true,
                Err(ExecutorSubmitError::WouldBlock(batch)) => {
                    blocked_workers.extend(targets);
                    held_products.extend(batch_products(&batch));
                    self.inflight.pending_submissions.push_back(*batch);
                }
                Err(ExecutorSubmitError::Failed(error)) => {
                    // Failure reconciliation owns the pending identities until
                    // every affected call has been retired.
                    self.on_executor_error(error);
                    return true;
                }
            }
        }
        progressed
    }

    /// Returns whether any rank queue can still accept a batch.
    ///
    /// `queue_depth` bounds the batches in flight on one rank, not across the
    /// worker group: a batch carries one call kind to one component, so a media
    /// pass occupies a slot on each component it touches rather than a single
    /// shared slot. Assembling stops once every rank queue is full.
    fn a_rank_queue_admits(&self) -> bool {
        self.executor
            .info()
            .workers
            .iter()
            .any(|(worker, _)| self.executor.has_capacity(worker))
    }

    /// Releases the request latent.
    fn free_request_latent(&mut self, id: RequestId) {
        let allocation = self
            .running
            .get_mut(&id)
            .and_then(|state| state.allocations_mut()?.latent.take());
        if let Some(allocation) = allocation {
            self.storage.latent_pool.free(allocation);
        }
    }

    /// Advances scheduler and executor work, blocking only for an outstanding result.
    ///
    /// Serving drives `run`; `step` lets tests and examples advance the
    /// scheduler deterministically from the calling thread.
    ///
    /// Drains `commands` first; a shutdown command or a disconnected channel
    /// aborts every request, closes the executor and returns `false`. When the
    /// nonblocking tick makes no progress while batches are in flight, this
    /// waits a bounded time for one result.
    ///
    /// Returns whether the loop made progress or handled an executor outcome.
    #[cfg(feature = "testing")]
    pub fn step(&mut self, commands: &Receiver<Command>) -> bool {
        if self.drain_commands(commands) {
            self.abort_all_requests();
            let _ = self.executor.close();
            return false;
        }
        let progressed = self.step_nonblocking();
        if progressed || self.inflight.pending_batches.is_empty() {
            return progressed;
        }
        match self.executor.poll(Duration::from_secs(300)) {
            Ok(Some(result)) => {
                self.apply_result(result);
                self.refill_executor();
                self.publish_cache_stats();
            }
            Ok(None) => return false,
            Err(error) => {
                self.on_executor_error(error);
                self.publish_cache_stats();
            }
        }
        true
    }

    /// Advances one nonblocking schedule-ahead tick for the owner-thread reactor. It
    /// flushes output journals, resolves at most one ready result, reaps
    /// cancellations, and fills available executor slots, but leaves any
    /// blocking result wait to `run`.
    pub(super) fn step_nonblocking(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.step").entered();
        let mut progressed = self.flush_output_journals();
        // Resolve one batch result. Refilling immediately after one
        // completion preserves an occupied execution slot when multiple
        // responses become ready together at queue depth greater than one.
        // The owner loop returns here without parking while progress is being
        // made, so subsequent ready completions are handled on successive turns.
        progressed |= self.poll_one_result();
        // A stage that arrived while a batch was executing is applied at the
        // first batch boundary with nothing in flight: no batch is ever decided
        // by one policy and executed under another (design §7.3).
        self.apply_policy_stage();
        progressed |= self.poll_kv_offload();
        progressed |= self.poll_trajectories();
        progressed |= self
            .encoder_outputs
            .as_mut()
            .is_some_and(crate::encoder_outputs::EncoderStore::poll);

        let submitted = self.refill_executor();
        progressed |= submitted;
        self.poll_resize();
        self.recompute_starved(submitted);
        self.poll_park();

        self.stats
            .general
            .in_flight
            .store(self.inflight.pending_batches.len(), Ordering::Relaxed);
        self.publish_cache_stats();
        progressed
    }

    /// Takes the host KV cache's answers, and ends the pins of requests
    /// that left the scheduler. Returns whether admission pins were
    /// answered.
    fn poll_kv_offload(&mut self) -> bool {
        let Some(kv_offload) = self.kv_offload.as_mut() else {
            return false;
        };
        let answered = kv_offload.poll();
        let waiting: HashSet<RequestId> = self
            .waiting
            .iter()
            .map(|state| state.req.request_id)
            .collect();
        let running = &self.running;
        kv_offload.sweep(|request| waiting.contains(&request) || running.contains_key(&request));
        answered
    }

    /// Dispatches queued batches, reaps cancellations, admits requests, and
    /// fills every available executor slot.
    ///
    /// Returns whether it queued or submitted a batch or handled a submission
    /// failure. Stops early once the engine-fatal latch is set.
    pub(super) fn refill_executor(&mut self) -> bool {
        let mut progressed = false;
        // Drop residency for requests that are no longer running or retiring.
        // A retiring request keeps its entries because
        // `Placement::batch_workers` finds the destinations of its `Finish`
        // and `Free` commands through them.
        let live: HashSet<RequestId> = self
            .running
            .keys()
            .chain(self.running_media.keys())
            .chain(self.retiring_requests.keys())
            .copied()
            .collect();
        self.placement
            .affinity
            .retain(|(request, _), _| live.contains(&request.request_id));
        self.placement
            .decode_affinity
            .retain(|(request, _), _| live.contains(&request.request_id));

        let mut blocked_workers = HashSet::new();
        progressed |= self.dispatch_submissions(&mut blocked_workers);
        if self.fatal {
            return progressed;
        }

        // Reap cancellations before assembling.
        self.reap_cancellations();

        // Admit request and resource residency even while every execution slot
        // is occupied. This lets the next batch see the complete resident
        // cohort instead of admitting only when a slot happens to open.
        self.admit();
        self.admit_media();

        // Submit as many batches as the rank queues admit.
        loop {
            self.admit();
            self.admit_media();
            if !self.a_rank_queue_admits() {
                break;
            }
            let batches = self.schedule_batches();
            if batches.is_empty() {
                break;
            }
            self.inflight.pending_submissions.extend(batches);
            progressed = true;
            self.dispatch_submissions(&mut blocked_workers);
            if self.fatal {
                return progressed;
            }
        }

        // A resize waits for the copies of the idle cached pages it evicts:
        // while no text batch carries them (nothing runs, as on a parked
        // world), batches of page offloads alone do.
        if self.resizing.is_some()
            && self.inflight.pending_submissions.is_empty()
            && self.inflight.pending_batches.is_empty()
            && let Some(workers) = self
                .kv_offload
                .as_ref()
                .map(crate::kv_cache::KvOffloadManager::queued_workers)
        {
            for worker in workers {
                let mut batch = self.finish_generation_batch(
                    ExecutionBatch::new(0, Vec::new(), Vec::new(), Vec::new()),
                    Instant::now(),
                );
                let Some(kv_offload) = self.kv_offload.as_mut() else {
                    break;
                };
                let pages = kv_offload.take_offloads(&worker, batch.id);
                batch.page_offloads = Some(crate::executor::PageOffloads {
                    worker,
                    pages,
                    latents: Vec::new(),
                });
                self.inflight.pending_submissions.push_back(batch);
                progressed = true;
            }
            self.dispatch_submissions(&mut blocked_workers);
            if self.fatal {
                return progressed;
            }
        }

        // Lifecycle commands that no scheduling pass carried go out in
        // command-only batches while nothing else awaits submission.
        while self.inflight.pending_submissions.is_empty()
            && !self.inflight.pending_commands.is_empty()
            && self.a_rank_queue_admits()
        {
            let commands = self.take_commands(|_| true);
            if commands.is_empty() {
                break;
            }
            let batch = self.finish_generation_batch(
                ExecutionBatch::new(0, Vec::new(), commands, Vec::new()),
                Instant::now(),
            );
            self.inflight.pending_submissions.push_back(batch);
            progressed = true;
            self.dispatch_submissions(&mut blocked_workers);
            if self.fatal {
                return progressed;
            }
        }

        progressed
    }

    /// Executes one step of the policy (§4.4).
    ///
    /// The step is the mechanism's whole scheduling act: observe the events
    /// since the previous step, ask for one plan on the room the mechanism has,
    /// validate it against the live state, and execute it. A decision the
    /// validators reject, a trap, or a starved request is a policy failure: the
    /// switch serves on under the previous policy, or the epoch stops.
    fn schedule_batches(&mut self) -> Vec<ExecutionBatch> {
        // A closing scheduler lets what is in flight drain before its
        // workers close; a sleeping one runs nothing.
        if self.closing || self.asleep {
            return Vec::new();
        }
        // Video media work runs the ready calls of its graph through the media
        // planner; the plan's media work entries replace that path when the
        // media planner is cut, so a video request is decided by one policy
        // like every other request.
        let batches = self.prepare_media_batches();
        if !batches.is_empty() {
            return batches;
        }
        let started = std::time::Instant::now();
        let Some(plan) = self.policy_step() else {
            return Vec::new();
        };
        // What the policy gives up comes first: the plan was decided on the
        // state it saw, and freeing a victim makes room for the step's calls.
        // The plan's own page decisions then run, so a prefetch's pins are in
        // flight by the time the calls are built.
        self.reclaim_shortage();
        self.execute_plan_pages(&plan);
        let batches = self.assemble_plan(&plan);
        self.end_step(started.elapsed());
        batches
    }

    /// Asks the policy for the victims of the shortage the mechanism reports,
    /// validates them, and frees exactly those (design §4.4).
    ///
    /// The mechanism decides that there is a shortage and owns every cache; the
    /// policy decides what to give up. A victim list that does not validate is
    /// a policy failure, like any other invalid decision.
    fn reclaim_shortage(&mut self) {
        let Some(shortage) = views::shortage_now(self) else {
            return;
        };
        let Some(mut policy) = self.policy.take() else {
            return;
        };
        let victims = match policy.reclaim(&shortage) {
            Ok(victims) => victims,
            Err(error) => {
                let _ = self.policy_failed(policy, error);
                return;
            }
        };
        let live = self.live();
        if let Err(error) = mechanism::validate_victims(&shortage, &victims, &live) {
            let _ = self.policy_failed(policy, error);
            return;
        }
        self.policy = Some(policy);
        for victim in &victims {
            self.free_victim(victim);
        }
        if !victims.is_empty() {
            tracing::debug!(
                kind = ?shortage.kind,
                need = shortage.need,
                victims = victims.len(),
                "the policy freed what it named"
            );
        }
    }

    /// Frees the one thing a victim names.
    ///
    /// An encoder entry leaves the encoder cache and its worker buffer goes with
    /// it; a device page goes to the host KV cache, exactly as a withdrawal's
    /// pages do, and the units it held return when the copy completes (or at
    /// once when the host cache has nowhere to put it).
    fn free_victim(&mut self, victim: &scheduling::guest::Victim) {
        match victim {
            scheduling::guest::Victim::Encoding(id) => {
                let key = self
                    .storage
                    .encoder_cache
                    .entries()
                    .map(|(key, _)| key)
                    .find(|key| views::encoding_id(key) == *id);
                let freed = key.and_then(|key| self.storage.encoder_cache.evict(&key));
                if let Some(product) = freed {
                    self.free_buffers([product.buffer_id()]);
                }
            }
            scheduling::guest::Victim::Page(page) => {
                if let Some(unit) = views::unit_of(*page) {
                    self.offload_units(&[unit]);
                }
            }
            // Today's capacity preemption, which the design keeps: a request
            // that must give its units away is marked migratable with `Drain`,
            // as `recompute_starved` does when it withdraws the units its
            // requests hold. `Drain` asks the engine to hold the world the
            // stream is in, and there is no world change here, so the daemon
            // resubmits the request as a continuation carrying the state it
            // reached. Only a migratable request is marked: one the daemon
            // could not resubmit would be lost. The request retires at its next
            // call boundary, and what it holds returns then.
            scheduling::guest::Victim::Request(request) => {
                self.migrate_running(MigrationCause::Drain, |id| id.0 == request.0);
            }
        }
    }

    /// Executes the page half of a validated plan (§4.4).
    ///
    /// The policy names pages to hand to the host KV cache and requests whose
    /// host hit should deepen; the mechanism owns the caches and moves exactly
    /// what it named, after validating that every name is a live, unreferenced
    /// device page. A move that the host cache cannot take leaves the pages
    /// where they are: the plan asked, the limit refused, and no request is at
    /// fault.
    fn execute_plan_pages(&mut self, plan: &Plan) {
        let units: Vec<u32> = plan
            .offload
            .iter()
            .filter_map(|page| views::unit_of(*page))
            .collect();
        if !units.is_empty() {
            let moved = self.offload_units(&units);
            if moved < units.len() {
                tracing::debug!(
                    asked = units.len(),
                    moved,
                    "the host KV cache took fewer pages than the plan offered"
                );
            }
        }
        for request in &plan.prefetch {
            self.prefetch_host(ipc::RequestId(request.0));
        }
    }

    /// Deepens one request's host KV cache hit as far as the host holds it.
    ///
    /// The device hit cannot grow, so the deepening is the host boundary
    /// beyond it: asking again after host pages arrive raises the boundary,
    /// and the answer is recorded so the next dispatch sees what the request
    /// can reuse. A request the mechanism cannot place, or whose host lookup
    /// is still in flight, keeps what it had: the plan asked, the cache is not
    /// ready, and no request is at fault.
    fn prefetch_host(&mut self, id: ipc::RequestId) {
        let Some(state) = self.running.get(&id) else {
            return;
        };
        let Some((target, hit)) = self.prefill_target(state) else {
            return;
        };
        let device_hit = hit.cached_tokens.min(state.req.prompt_token_ids.len());
        if let admission::HostHit::Serving(boundary) = admission::host_prefix(
            &mut self.kv_offload,
            state,
            &target,
            device_hit,
            usize::MAX,
        ) {
            self.recorder.prefix_hit(
                scheduling::guest::RequestId(id.0),
                u32::try_from(device_hit).unwrap_or(u32::MAX),
                u32::try_from(boundary.saturating_sub(device_hit)).unwrap_or(u32::MAX),
            );
        }
    }

    /// One step's plan: observe, dispatch, validate (§4.4).
    ///
    /// Nothing decides before the policy is installed: an epoch waits for its
    /// first `Switchover` (§7.3), and a scheduler with no slot schedules
    /// nothing. The slot is taken out of the scheduler for the calls and put
    /// back whatever they answer, so a failure can hand the mechanism's state
    /// to the switch without borrowing the scheduler the slot lives in.
    fn policy_step(&mut self) -> Option<Plan> {
        let Some(mut policy) = self.policy.take() else {
            return None;
        };
        self.step += 1;
        let room = self.room();
        let live = self.live();
        // A resident request that has not advanced within the policy's timeout
        // is the policy's failure, not the request's (design §4.4).
        let now = scheduling::guest::Nanos(crate::clock::now_monotonic_us().saturating_mul(1_000));
        if let Some(request) = self.watchdog.starved(&live, now, policy.limits().timeout) {
            return self.policy_failed(policy, PolicyError::Starved { request });
        }
        // Everything that changed since the previous step, in the order it
        // happened, closed before this step's dispatch.
        self.recorder.close();
        let events = self.recorder.take();
        if let Err(error) = policy.observe(&events) {
            return self.policy_failed(policy, error);
        }
        let plan = match policy.dispatch(&room) {
            Ok(plan) => plan,
            Err(error) => return self.policy_failed(policy, error),
        };
        if let Err(error) = mechanism::validate_plan(&plan, &room, &live) {
            return self.policy_failed(policy, error);
        }
        self.policy = Some(policy);
        Some(plan)
    }

    /// Ends the step's accounting: the step number, and the policy's per-step
    /// cost as the fuel it spent (design §12).
    fn end_step(&mut self, duration: std::time::Duration) {
        let step = self.step;
        self.recorder.record(views::stepped(step, duration));
    }

    /// A policy failure: serve on under the previous policy (§7.3), or stop the
    /// epoch when there is none to serve under.
    ///
    /// The state the previous policy is instantiated on is the mechanism's at
    /// this moment, built here and handed over owned: the slot lives inside the
    /// scheduler, so the mechanism cannot lend it a borrow of the scheduler.
    fn policy_failed(
        &mut self,
        mut policy: policy::PolicySlot,
        error: PolicyError,
    ) -> Option<Plan> {
        let state = self.trial_state();
        match policy.fail(&error, &state) {
            Ok(report) => {
                tracing::warn!(
                    scheduler = ?report.scheduler,
                    backend = %report.program.backend,
                    revision = report.program.revision,
                    reason = %report.reason,
                    "the policy failed; serving on under the previous one"
                );
                self.policy_report = Some(report);
                self.policy = Some(policy);
            }
            Err(fatal) => {
                tracing::error!(reason = %fatal, "the policy failed and cannot be reverted");
                self.fail(format!("the scheduling policy failed: {fatal}"));
                self.policy = Some(policy);
            }
        }
        None
    }

    /// This step's room.
    ///
    /// `views.rs` builds every fact but one: the captured graph tiers, which
    /// the build config states (the spec's `cuda_graphs`).
    fn room(&self) -> Room {
        let mut facts = views::room_facts(self, self.step);
        facts.graph_tiers = self.config.graph_tiers.clone();
        views::room_view(facts)
    }

    /// The mechanism's state at a batch boundary, owned, for the switch.
    fn trial_state(&self) -> policy::TrialState {
        policy::TrialState {
            snapshot: views::snapshot(self, self.step, self.recorder.recorded()),
            room: self.room(),
            shortage: views::shortage_now(self),
            arrival: None,
            call: views::placement(self),
            expert_load: views::expert_load_of(self),
        }
    }

    /// The objects the validators read, as they stand now.
    fn live(&self) -> mechanism::Live {
        let mut live = mechanism::Live {
            queue: self
                .waiting
                .iter()
                .map(|state| scheduling::guest::RequestId(state.req.request_id.0))
                .chain(
                    self.waiting_media
                        .iter()
                        .map(|pending| scheduling::guest::RequestId(pending.request.request_id.0)),
                )
                .collect(),
            ..mechanism::Live::default()
        };
        for request in views::requests(self, self.recorder.recorded()) {
            live.requests.insert(request.id, request);
        }
        for page in views::pages(self) {
            live.pages.insert(page.id, page);
        }
        for entry in views::encodings(self) {
            live.encodings.insert(entry.id, entry);
        }
        for phase in views::phases(&self.info) {
            live.phases.insert(phase.id, phase);
        }
        live
    }

    /// Installs the step's policy: the epoch's first one, or the one a
    /// `Switchover` prepared. Only a batch boundary may adopt, so no work is
    /// ever decided by one policy and executed under another.
    pub(super) fn adopt(&mut self, policy: policy::PolicySlot) {
        debug_assert!(
            self.inflight.pending_batches.is_empty()
                && self.inflight.pending_submissions.is_empty(),
            "a policy is adopted at a batch boundary"
        );
        self.policy = Some(policy);
    }

    /// The report of a policy failure, for the control thread to hand up.
    pub(super) fn take_policy_report(&mut self) -> Option<policy::PolicyReverted> {
        self.policy_report.take()
    }

    /// Allocates the completion product of a state-advancing media call that
    /// declares no outputs.
    ///
    /// `plan_media_call` requests one for latent preparation and every
    /// denoising step but the last; the request's next state-advancing call
    /// names it as its `predicate` while this call is still in flight. The
    /// generation counter is shared with the products the generation path
    /// registers.
    ///
    /// Fails, leaving the generation counter unchanged, once product
    /// generations are exhausted.
    pub(super) fn media_completion_product(
        &mut self,
        request_key: RequestKey,
        call_id: CallId,
    ) -> Result<TensorRef, &'static str> {
        let generation = u32::try_from(self.next_product_generation.max(1))
            .map_err(|_| "media product generations remain within u32")?;
        self.next_product_generation = u64::from(generation).saturating_add(1);
        Ok(TensorRef {
            request_key,
            producer_call_id: call_id,
            output_index: 0,
            generation,
            dtype: DType::U8,
            shape_bound: ShapeBound::default(),
        })
    }
}

/// The calls that read the products of an image generation call, or `None`
/// for a call whose readers are the destinations of its transfer edges.
///
/// Image generation chains its state calls on one denoising component: text
/// features feed latent preparation, which feeds the denoising steps.
/// `WorkerGroup::consumer_slots` reads this for a media call whose placement
/// names no readers; a video request's calls name theirs from the request's
/// graph (`plan_media_call`).
pub(crate) fn generation_consuming_calls(media_call: MediaCall) -> Option<&'static [MediaCall]> {
    Some(match media_call {
        MediaCall::TextEncoding => &[MediaCall::LatentPreparation],
        MediaCall::LatentPreparation | MediaCall::Denoising => &[MediaCall::Denoising],
        _ => return None,
    })
}

impl Scheduler {
    /// Publishes request, in-flight batch, prefix-cache and encoder-cache gauges
    /// to scheduler counters.
    pub(super) fn publish_cache_stats(&mut self) {
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
        self.stats
            .general
            .in_flight
            .store(self.inflight.pending_batches.len(), Ordering::Relaxed);
        if let Some(prefix_cache) = self.storage.cache.as_ref() {
            let pool = prefix_cache.block_pool.stats();
            self.stats
                .prefix_cache
                .pages_evicted
                .store(pool.evictions, Ordering::Relaxed);
            self.stats
                .prefix_cache
                .pages_stored
                .store(pool.pages_stored, Ordering::Relaxed);
            self.stats
                .prefix_cache
                .cached_pages
                .store(prefix_cache.block_pool.cached_pages(), Ordering::Relaxed);
        }
        self.stats
            .encoder
            .cache_queries
            .store(self.storage.encoder_cache.stats.queries, Ordering::Relaxed);
        self.stats
            .encoder
            .cache_hits
            .store(self.storage.encoder_cache.stats.hits, Ordering::Relaxed);
        self.stats
            .encoder
            .cached
            .store(self.storage.encoder_cache.len(), Ordering::Relaxed);
    }

    /// Resolves at most one ready result so assembly can refill the freed slot
    /// before another completion is consumed.
    pub(super) fn poll_one_result(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.poll_one_result").entered();
        match self.executor.poll(Duration::ZERO) {
            Ok(Some(result)) => {
                self.apply_result(result);
                true
            }
            Ok(None) => false,
            Err(error) => {
                self.on_executor_error(error);
                true
            }
        }
    }
}

impl Scheduler {
    /// Retains a submitted call and its inputs as in flight for its request,
    /// and charges one credit to its metrics domain.
    ///
    /// Calls queue per request in the order they are registered, and
    /// `Inflight::take_ready_completions` releases the completion of any call
    /// other than an independent media call only from the front of that
    /// queue. Transfer capacity is charged separately, by
    /// `reserve_generation_resources`.
    pub(super) fn register_inflight(&mut self, call: Call, input: InflightInput, queue_us: u64) {
        let request_id = call.request_key.request_id;
        let domain = self.stats.domains.get(call.kind);
        let active = domain.active_credits.fetch_add(1, Ordering::Relaxed) + 1;
        domain.peak_credits.fetch_max(active, Ordering::Relaxed);
        domain.launched_calls.fetch_add(1, Ordering::Relaxed);
        domain.queue_us.fetch_add(queue_us, Ordering::Relaxed);
        self.inflight
            .pending_calls
            .entry(request_id)
            .or_default()
            .push_back(InflightCall { call, input });
    }

    /// Counts one backpressure event against the metrics domain of `kind`.
    pub(super) fn record_domain_backpressure(&self, kind: ipc::CallKind) {
        self.stats
            .domains
            .get(kind)
            .backpressure_events
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Accumulates timing once per public metric group in a returned batch.
    fn record_domain_batch(stats: &super::stats::DomainStats, timing: TimingCounters) {
        stats.completed_batches.fetch_add(1, Ordering::Relaxed);
        stats
            .launch_us
            .fetch_add(timing.queued_us, Ordering::Relaxed);
        stats
            .device_us
            .fetch_add(timing.device_us, Ordering::Relaxed);
        stats.completion_us.fetch_add(
            timing.copy_us.saturating_add(timing.host_us),
            Ordering::Relaxed,
        );
    }

    /// Counts one returned call, and a predicated or failed status, against
    /// the metrics domain of `kind`.
    pub(super) fn record_domain_completion(&self, kind: ipc::CallKind, status: CallStatus) {
        let stats = self.stats.domains.get(kind);
        stats.completed_calls.fetch_add(1, Ordering::Relaxed);
        match status {
            CallStatus::Ok => {}
            CallStatus::Predicated => {
                stats.predicated_calls.fetch_add(1, Ordering::Relaxed);
            }
            CallStatus::Error => {
                stats.error_calls.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Returns the credit a leaving call holds in its metrics domain, counting
    /// the call as failed when `failed` is set.
    ///
    /// A reclaim with no active credit is logged as an accounting defect and
    /// changes no counter.
    pub(super) fn reclaim_domain_credit(&self, kind: ipc::CallKind, failed: bool) {
        let stats = self.stats.domains.get(kind);
        let active = stats.active_credits.load(Ordering::Relaxed);
        if active == 0 {
            tracing::error!(?kind, "domain call credit accounting underflowed");
            return;
        }
        stats.active_credits.fetch_sub(1, Ordering::Relaxed);
        stats.reclaimed_credits.fetch_add(1, Ordering::Relaxed);
        if failed {
            stats.error_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Reclaims the domain credit of every in-flight call as failed.
    ///
    /// Callers invoke it before `Inflight::clear_failed_calls` drops the
    /// calls whose credits it returns.
    pub(super) fn fail_inflight_domain_credits(&self) {
        for inflight in self.inflight.pending_calls.values().flatten() {
            self.reclaim_domain_credit(inflight.call.kind, true);
        }
    }
}
