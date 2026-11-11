//! The step: what a scheduler does between two waits.
//!
//! - `schedule` forms batches while a rank queue has room, by asking the
//!   policy for one step's decisions and assembling them (`batching`), and
//!   `release` hands out, in order, the batches whose devices have a slot,
//!   each after the batches producing its inputs.
//! - `update` takes one completed batch: `apply_result` (`results`)
//!   validates every returned call against the batches in flight before any
//!   completion is applied, and applies them in request-local order.
//! - Video scheduling (`media`): a video request runs the calls of its graph
//!   (`graph::VideoGraph`), offered by `ready_calls` as their inputs become
//!   available. Calls occupy their stages' capacity on their workers and,
//!   for media reading, encoding and muxing, host thread pools.
//! - Successor projection (`projection`): `projected_coordinates`,
//!   `pending_successor_code` and `can_queue_successor` project a token
//!   request's accepted state through its calls in flight, so a successor
//!   can be queued before its predecessor's result returns.
//! - Failure handling (`failure`): `on_worker_error` retires only the work a
//!   `WorkerFailure` invalidates. Any other error of the devices, and a
//!   failure classified as a broken invariant, stops the engine.

mod failure;
mod media;
pub(crate) use media::VideoState;
mod projection;
mod results;

use super::*;
use crate::request::Request;
use ipc::BufferId;
use scheduling::PolicyError;
use scheduling::guest::{DispatchOutput, Room};

pub(super) use media::Occupancy;

/// The products a batch's calls publish.
fn batch_products(batch: &ExecutionBatch) -> impl Iterator<Item = BufferId> + '_ {
    batch
        .requests
        .iter()
        .flat_map(|(call, _)| call.output_buffers())
}

/// The steps one pass may take in a row that form no batch while the views
/// the policy reads keep moving.
const IDLE_STEPS: usize = 16;

impl Scheduler {
    /// Releases the batches formed and not yet handed out whose devices have
    /// a slot free, in order, preserving order at every shared destination
    /// and after every product a batch reads.
    ///
    /// Each formed batch is tried once per call. One that fits takes its
    /// slots, is laid out for its devices on the step's generation and joins
    /// `released`, which the caller submits in order; one that does not fit
    /// goes back to the end of the queue, and `blocked_workers` keeps its
    /// destinations for the whole pass, so forming more work cannot overtake
    /// it there. Page copies that ride no batch are issued here: they are
    /// storage work with no layout to hand out.
    ///
    /// Returns whether any batch was released or settled, or `true` after a
    /// failure, which is handed to `on_worker_error` and ends the pass.
    fn release(
        &mut self,
        blocked_workers: &mut HashSet<crate::WorkerGroupId>,
        released: &mut Vec<ipc::execution::Batch>,
    ) -> bool {
        let mut progressed = false;
        // Products of the batches held back in this pass. A call may be
        // submitted before its inputs' producers complete, since its worker
        // waits on them, but not before they are submitted, which is when the
        // worker learns where each product is published. A consumer is
        // queued after its producer and held batches keep their order, so the
        // producer is always tried first.
        let mut held_products = HashSet::new();
        // Image features are encoded where a request's prompt is computed, which is the first
        // group of a scheduler that serves several (a deployment's prefill group): a batch of
        // frees alone goes there.
        let frees_group = self
            .worker
            .info()
            .workers
            .first()
            .map(|(group, _)| group.0.clone())
            .unwrap_or_default();
        let pending_count = self.inflight.pending_submissions.len();
        for _ in 0..pending_count {
            let Some(batch) = self.inflight.pending_submissions.pop_front() else {
                break;
            };
            let targets = self.dispatch.batch_workers(&batch);
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
            // A batch of lifecycle commands alone has nothing to tell a device:
            // the slots and buffers a command closes are this scheduler's own
            // pools, so the command is settled here. Its `Finish` has nothing left to reach,
            // nor its `Free` unless a rank holds image features, whose storage is the rank's
            // (`Batch::frees`). A batch that carries page offloads, of the host KV cache or of a
            // preempted trajectory, is work for a device and goes to it.
            if batch.requests.is_empty()
                && batch.page_offloads.is_none()
                && !(self.holds_image_features() && !batch.frees().is_empty())
            {
                self.acknowledge_commands(&batch.commands);
                // Nothing went to a device, so no completion will ever end this batch: it leaves
                // the in-flight window here, or every boundary that waits for an empty window (a
                // policy stage, a park, a drain) waits for it forever.
                self.inflight.pending_batches.remove(&batch.id);
                progressed = true;
                continue;
            }
            // The `Finish` commands this batch carries close the slots of the
            // requests they name once it is handed out: batches reach a rank in
            // the order they are released, so no later `Start` can name a slot
            // before the worker has seen its close.
            let finishing: Vec<RequestKey> = batch
                .commands
                .iter()
                .filter_map(|command| match command {
                    BatchCommand::Finish { request_key, .. } => Some(*request_key),
                    _ => None,
                })
                .collect();
            let id = ipc::execution::BatchId(batch.id);
            // A preempted request's trajectory waits for nothing on the device but the batches
            // that advanced it, and pages that wait while nothing runs ride no forward: each is
            // a copy of its own, whose end the scheduler reads as it reads a batch's.
            let copies = batch
                .page_offloads
                .as_ref()
                .filter(|_| batch.requests.is_empty());
            let copied = match copies {
                Some(offloads) if !offloads.latents.is_empty() => {
                    Some(self.worker.offload_latents(id, offloads.latents.clone()))
                }
                Some(offloads) if !offloads.pages.is_empty() => {
                    Some(self.worker.copy_pages(id, offloads.pages.clone()))
                }
                _ => None,
            };
            match copied {
                Some(Ok(())) => {
                    progressed = true;
                    continue;
                }
                Some(Err(error)) => {
                    self.on_worker_error(error);
                    return true;
                }
                None => {}
            }
            // An update may have asked this step's batches to run in shadow too:
            // the same work against the generation whose kernels are being
            // compared, submitted beside the serving one.
            let shadow = self.shadow_generation(&batch);
            let laid_out = batch
                .device_batch(self.step_generation, &frees_group)
                .and_then(|(device, numbered)| {
                    let reserved = self.worker.reserve(&device, &batch.pools())?;
                    Ok((device, numbered, reserved))
                });
            let (device, numbered) = match laid_out {
                Ok((device, numbered, true)) => (device, numbered),
                // A full queue is not the pass's failure: the batch stays
                // with the scheduler and a later pass tries again.
                Ok((_, _, false)) => {
                    blocked_workers.extend(targets);
                    held_products.extend(batch_products(&batch));
                    self.inflight.pending_submissions.push_back(batch);
                    continue;
                }
                Err(error) => {
                    // Failure reconciliation owns the pending identities until
                    // every affected call has been retired.
                    self.on_worker_error(error);
                    return true;
                }
            };
            if let Some(pending) = self.inflight.pending_batches.get_mut(&batch.id) {
                pending.numbered = numbered;
            }
            progressed = true;
            for request_key in finishing {
                self.close_retiring_row(request_key);
            }
            if let Some(generation) = shadow {
                match batch.device_batch(generation, &frees_group) {
                    Ok((copy, _)) => self
                        .inflight
                        .pending_shadows
                        .push((copy, batch.token_shape())),
                    // A refused pass gets its response at once, because waiting will
                    // not shadow it.
                    Err(error) => self.shadow_failed(format!("{error:#}")),
                }
            }
            released.push(device);
        }
        progressed
    }

    /// Submits `batches`, in order, to the executors, then the shadow copies
    /// an update asked for. A failed submission is the devices' failure
    /// (`on_worker_error`) and ends the submitting.
    fn submit(&mut self, batches: Vec<ipc::execution::Batch>) {
        for batch in batches {
            if let Err(error) = self.worker.submit(batch) {
                self.on_worker_error(error);
                return;
            }
        }
        for (copy, shape) in std::mem::take(&mut self.inflight.pending_shadows) {
            match self.worker.shadow(copy) {
                Ok(()) => self.note_shadowed(shape),
                Err(error) => self.shadow_failed(format!("{error:#}")),
            }
        }
    }

    /// Takes one completed batch `batch` with its `outcome`: gives back the
    /// slots it held, advances its requests in request order, publishes
    /// their pages and emits their events.
    ///
    /// A batch that failed on its device is the engine's failure: the
    /// requests it carried end, and the scheduler stops.
    pub fn update(
        &mut self,
        batch: ipc::execution::BatchId,
        outcome: ipc::execution::Outcome<ipc::execution::BatchOutput>,
    ) {
        self.worker.release(batch);
        let numbered = self
            .inflight
            .pending_batches
            .get(&batch.0)
            .map(|pending| pending.numbered.clone())
            .unwrap_or_default();
        match crate::completion::BatchReport::from_output(batch.0, outcome, &numbered) {
            Ok(report) => self.apply_result(report),
            Err(error) => {
                // A failure retires what the devices held: no slot is kept
                // for a batch that will not complete.
                self.worker.release_all();
                self.on_worker_error(error);
            }
        }
    }

    /// Returns whether any rank queue can still accept a batch.
    ///
    /// `queue_depth` bounds the batches in flight on one rank, not across the
    /// worker group: a batch carries one call kind to one stage, so a media
    /// pass occupies a slot on each stage it touches rather than a single
    /// shared slot. Assembling stops once every rank queue is full.
    fn a_rank_queue_admits(&self) -> bool {
        self.worker
            .info()
            .workers
            .iter()
            .any(|(worker, _)| self.worker.has_capacity(worker))
    }

    /// Closes the scheduler-owned half of a retiring request's record.
    ///
    /// The request row, the prefix-cache table and the latent pages are the
    /// scheduler's own capacity: they return as soon as the `Finish` that names
    /// the request has been submitted, because every later batch on that rank
    /// is ordered behind it and no admission can reuse the row before the
    /// worker has seen the close. The buffers stay in the record until the
    /// acknowledgement, because they are spans the worker still holds.
    fn close_retiring_row(&mut self, request_key: RequestKey) {
        let Some(retiring) = self
            .retiring_requests
            .get_mut(&request_key.request_id)
            .filter(|retiring| retiring.request_key == request_key)
        else {
            return;
        };
        for mut allocation in std::mem::take(&mut retiring.allocations) {
            let buffers = std::mem::take(&mut allocation.buffers);
            retiring.buffers.extend(buffers);
            allocation.free(&mut self.storage);
        }
        if let Some(media) = retiring.media_allocations.as_mut()
            && let Err(unknown) = media.close_rows(&mut self.storage)
        {
            self.invariant_broken(&unknown.to_string());
        }
    }

    /// Advances one nonblocking schedule-ahead tick for the owner-thread reactor. It
    /// flushes output journals, resolves at most one ready result, reaps
    /// cancellations, and fills available worker slots, but leaves any
    /// blocking result wait to `run`.
    /// Strips a retiring request's queued calls, and the admissions among its
    /// commands, from the batches the worker has not accepted yet.
    ///
    /// A batch is prepared from the state of the step that named it, and a
    /// worker queue deeper than one batch keeps it queued while later steps
    /// run. A request that ends in that window leaves calls behind that no
    /// result will resolve, and a batch that still carries one can no longer
    /// state the layout its kernels read: the request's allocation is gone, so
    /// its columns would name no cache table at all. The batches keep their
    /// other work, and one left with nothing is forgotten - the pending
    /// identity goes with it, so no step waits for a result it will never see.
    /// Its `Finish` and `Free` commands stay: they are what settles the
    /// worker's state for the request.
    pub(super) fn retire_queued_calls(&mut self, id: RequestId) {
        let requests: HashSet<RequestKey> = self
            .inflight
            .pending_submissions
            .iter()
            .flat_map(|batch| batch.requests.iter())
            .filter(|(call, _)| call.request_key.request_id == id)
            .map(|(call, _)| call.request_key)
            .collect();
        if requests.is_empty() {
            return;
        }
        let mut queued = VecDeque::new();
        for mut batch in std::mem::take(&mut self.inflight.pending_submissions) {
            for (call, _) in batch.retire_requests(&requests) {
                self.requests.mark(call.request_key.request_id);
                if let Some(pending) =
                    self.inflight
                        .retire_call(batch.id, call.request_key, call.call_id)
                {
                    self.return_domain_credit(pending.call.kind, true);
                }
            }
            if let Some(pending) = self.inflight.pending_batches.get_mut(&batch.id) {
                pending.commands = batch
                    .commands
                    .iter()
                    .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                    .cloned()
                    .collect();
            }
            queued.push_back(batch);
        }
        for batch in queued {
            if batch.requests.is_empty() && batch.commands.is_empty() {
                self.inflight.pending_batches.remove(&batch.id);
            } else {
                self.inflight.pending_submissions.push_back(batch);
            }
        }
    }

    /// Whether a rank of this scheduler holds image features in storage of its own, which the
    /// products' frees must reach (`Batch::frees`).
    pub(super) fn holds_image_features(&self) -> bool {
        self.info
            .image_encoders
            .contains_key(crate::encodings::VISION_FEATURES)
    }

    /// One pass of the step loop, without waiting: flushes the output
    /// journals, takes one completed batch (`update`), reads what the Storage
    /// Servers responded, assembles the batches that fit (`schedule`) and
    /// submits them. Returns whether anything happened; a caller that finds
    /// nothing did waits (`Scheduler::wait`).
    pub(super) fn step(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.step").entered();
        let mut progressed = self.flush_output_journals();
        // Resolve one batch result. Refilling immediately after one
        // completion preserves an occupied execution slot when multiple
        // responses become ready together at queue depth greater than one.
        // The owner loop returns here without parking while progress is being
        // made, so subsequent ready completions are handled on successive turns.
        progressed |= self.take_completed();
        // The wait for a predecessor's index ended: the requests held for it
        // return to the head of the queue.
        if !self.held_for_index.is_empty()
            && self
                .awaiting_index
                .is_some_and(|until| Instant::now() >= until)
        {
            self.release_held_for_index();
            progressed = true;
        }
        // A stage that arrived while a batch was executing is applied at the
        // first batch boundary with nothing in flight: no batch is ever decided
        // by one policy and executed under another (design §7.3).
        self.apply_policy_stage();
        progressed |= self.poll_kv_offload();
        progressed |= self.poll_hand_offs();
        progressed |= self.poll_trajectories();
        progressed |= self
            .encoder_outputs
            .as_mut()
            .is_some_and(crate::encoder_outputs::EncoderStore::poll);

        let mut batches = Vec::new();
        let submitted = self.schedule_into(&mut batches);
        self.submit(batches);
        progressed |= submitted;
        self.poll_resize();
        self.recompute_starved(submitted);
        self.poll_park();

        self.stats
            .general
            .in_flight
            .store(self.inflight.pending_batches.len(), Ordering::Relaxed);
        self.publish_cache_stats();
        if progressed {
            self.progress.wake();
        }
        progressed
    }

    /// Takes the host KV cache's responses, and ends the pins of requests
    /// that left the scheduler. Returns whether admission pins were
    /// responded.
    fn poll_kv_offload(&mut self) -> bool {
        let Some(kv_offload) = self.kv_offload.as_mut() else {
            return false;
        };
        let responded = kv_offload.poll();
        // Nothing pinned means nothing to sweep, so a step gathers no live set
        // to find that out (design §4.4).
        if !kv_offload.holds() {
            return responded;
        }
        let requests = &self.requests;
        kv_offload.sweep(|request| {
            requests
                .get(&request)
                .is_some_and(|request| request.tokens().is_some())
        });
        responded
    }

    /// Assembles the batches that fit now, at most one per worker group and
    /// phase with room in its queue, and returns them for submission, in
    /// order.
    ///
    /// The running requests that can make progress come first, within the
    /// step's call and token budgets; then the waiting queue in the policy's
    /// order, each request admitted when it can be given a slot and the pages
    /// its next call needs. A batch whose devices are full stays with the
    /// scheduler and is returned by a later call.
    pub fn schedule(&mut self) -> Vec<ipc::execution::Batch> {
        let mut released = Vec::new();
        self.schedule_into(&mut released);
        released
    }

    /// [`Self::schedule`] into `released`; returns whether the pass formed,
    /// released or settled a batch, or handled a failure. Stops early once
    /// the engine-fatal latch is set.
    pub(super) fn schedule_into(&mut self, released: &mut Vec<ipc::execution::Batch>) -> bool {
        let mut progressed = false;
        // Drop residency for requests that are no longer running or retiring.
        // A retiring request keeps its entries because
        // `Dispatch::batch_workers` finds the destinations of its `Finish`
        // and `Free` commands through them.
        let requests = &self.requests;
        let retiring = &self.retiring_requests;
        let live = |request: &ipc::RequestId| {
            requests.get(request).is_some_and(Request::admitted) || retiring.contains_key(request)
        };
        self.dispatch
            .affinity
            .retain(|(request, _), _| live(&request.request_id));
        self.dispatch
            .decode_affinity
            .retain(|(request, _), _| live(&request.request_id));

        let mut blocked_workers = HashSet::new();
        progressed |= self.release(&mut blocked_workers, released);
        if self.failed() {
            return progressed;
        }

        // Reap cancellations before assembling.
        self.reap_cancellations();

        // Form as many batches as the rank queues admit. Admission is not
        // done here: it is the policy's `admit`, executed once per step.
        let mut idle_steps = 0;
        loop {
            if !self.a_rank_queue_admits() {
                break;
            }
            // A stage applies at a boundary with nothing in flight, and a scheduler under traffic
            // always has a batch in flight while it keeps forming them. So one that is due stops
            // this loop: what is in flight drains, the stage applies, and forming resumes (7.3).
            if self.policy_stage_due() {
                break;
            }
            let batches = self.schedule_batches();
            if batches.is_empty() {
                // A step that formed no batch may still have moved a request:
                // an image found in the encoder cache needs no encode, so the
                // work the policy gave its encode became nothing and its
                // prefill is the next decision. The policy decides again on
                // the views that moved, within a bound that keeps a pair of
                // views that would move each other for ever from spinning.
                idle_steps += 1;
                if idle_steps <= IDLE_STEPS && self.policy.is_some() && self.refresh_views() {
                    progressed = true;
                    continue;
                }
                break;
            }
            idle_steps = 0;
            self.inflight.pending_submissions.extend(batches);
            progressed = true;
            self.release(&mut blocked_workers, released);
            if self.failed() {
                return progressed;
            }
        }

        // A resize waits for the copies of the idle cached pages it evicts:
        // while no text batch carries them (nothing runs, as on a parked
        // epoch), batches of page offloads alone do.
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
                    ExecutionBatch::new(0, Vec::new(), Vec::new()),
                    Instant::now(),
                );
                let Some(kv_offload) = self.kv_offload.as_mut() else {
                    break;
                };
                let pages = kv_offload.take_offloads(&worker, batch.id);
                batch.page_offloads = Some(crate::worker::PageOffloads {
                    worker,
                    pages,
                    latents: Vec::new(),
                });
                self.inflight.pending_submissions.push_back(batch);
                progressed = true;
            }
            self.release(&mut blocked_workers, released);
            if self.failed() {
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
                ExecutionBatch::new(0, Vec::new(), commands),
                Instant::now(),
            );
            self.inflight.pending_submissions.push_back(batch);
            progressed = true;
            self.release(&mut blocked_workers, released);
            if self.failed() {
                return progressed;
            }
        }

        progressed
    }

    /// Executes one step of the policy (§4.4).
    ///
    /// The step is the mechanism's whole scheduling act: observe the events
    /// since the previous step, ask for one output on the room the mechanism has,
    /// validate it against the live state, and execute it. A decision the
    /// validators reject, a trap, or a starved request is a policy failure: the
    /// switch serves on under the previous policy, or the epoch stops.
    fn schedule_batches(&mut self) -> Vec<ExecutionBatch> {
        // A closing scheduler lets what is in flight drain before its
        // workers close; a sleeping one runs nothing.
        if self.draining || self.asleep || self.held {
            return Vec::new();
        }
        let started = std::time::Instant::now();
        let Some(output) = self.policy_step() else {
            return Vec::new();
        };
        // What the policy gives up comes first: the output was decided on the
        // state it saw, and freeing a victim makes room for the step's calls.
        // The output's own page decisions then run, so a prefetch's pins are in
        // flight by the time the calls are built.
        self.preempt_shortage();
        // The assembly below notes the requests it cannot grow afresh, for the
        // next step's admission and preemption.
        self.starved.clear();
        self.execute_page_work(&output);
        let batches = self.assemble_output(&output);
        self.log_step(&output, &batches);
        self.end_step(started.elapsed());
        batches
    }

    /// Executes the page half of a validated output (§4.4).
    ///
    /// The policy names pages to hand to the host KV cache and requests whose
    /// host hit should deepen; the mechanism owns the caches and moves exactly
    /// what it named, after validating that every name is a live, unreferenced
    /// device page. A move that the host cache cannot take leaves the pages
    /// where they are: the output asked, the limit refused, and no request is at
    /// fault.
    fn execute_page_work(&mut self, output: &DispatchOutput) {
        let units: Vec<u32> = output
            .offload
            .iter()
            .filter_map(|page| views::unit_of(*page))
            .collect();
        // A page the host was offered as it was published has nothing left to move. A policy names
        // the same idle pages step after step, and taking one again pins it out of the free queue
        // until the host responses, so requests that need units would wait behind cached pages.
        let units = self
            .storage
            .cache()
            .map_or(units.clone(), |cache| cache.pool().without_host_key(&units));
        if !units.is_empty() {
            let moved = self.offload_units(&units);
            if moved < units.len() {
                tracing::debug!(
                    asked = units.len(),
                    moved,
                    "the host KV cache took fewer pages than the output offered"
                );
            }
        }
        for request in &output.prefetch {
            self.prefetch_host(ipc::RequestId(request.0));
        }
    }

    /// Deepens one request's host KV cache hit as far as the host holds it.
    ///
    /// The device hit cannot grow, so the deepening is the host boundary
    /// beyond it: asking again after host pages arrive raises the boundary,
    /// and the response is recorded so the next dispatch sees what the request
    /// can reuse. A request the mechanism cannot place, or whose host lookup
    /// is still in flight, keeps what it had: the output asked, the cache is not
    /// ready, and no request is at fault.
    fn prefetch_host(&mut self, id: ipc::RequestId) {
        // §4.4: the mechanism deepens a prefix hit only by at least the floor
        // the policy declared, so a smaller deepening is not a prefetch at all.
        // The floor is the policy's, not a constant here, because a policy that
        // asks for less eager deepenings must get them.
        let floor = self
            .policy
            .as_ref()
            .map_or(0, |policy| policy.limits().prefetch_min_tokens) as usize;
        let Some((_, hit)) = self.lookup_prefix(id, usize::MAX, floor, std::time::Duration::ZERO)
        else {
            return;
        };
        let Some(state) = self.requests.tokens(&id) else {
            return;
        };
        let device_hit = hit.cached_tokens.min(state.req.prompt_token_ids.len());
        if let Some(deepening) = hit.lower_tokens {
            self.recorder.prefix_hit(
                scheduling::guest::RequestId(id.0),
                u32::try_from(device_hit).unwrap_or(u32::MAX),
                u32::try_from(deepening).unwrap_or(u32::MAX),
            );
        }
    }

    /// One step's output: observe, dispatch, validate (§4.4).
    ///
    /// Nothing decides before the policy is installed: an epoch waits for its
    /// first `Switchover` (§7.3), and a scheduler with no slot schedules
    /// nothing. The slot is taken out of the scheduler for the calls and put
    /// back whatever they respond, so a failure can hand the mechanism's state
    /// to the switch without borrowing the scheduler the slot lives in.
    fn policy_step(&mut self) -> Option<DispatchOutput> {
        let mut policy = self.policy.take()?;
        self.step += 1;
        // The kernels this step's batches serve on, read once: a switchover
        // that completes mid-step must not split one step across two epochs
        // (4.2, 7.3).
        let generation = self.serving_generation();
        self.step_generation = generation;
        // The pages that appeared and went since the previous step, then
        // everything else that changed, in the order it happened, closed before
        // this step's dispatch. The room, the live state, and the output's checks
        // are all read after it: a page announced now is a page this step's
        // output may name, so the validators must already see it.
        self.pump_page_changes();
        // The requests whose view moved since the previous step are stated to
        // the policy here, as `Updated` events: a policy's state is
        // append-only, so a phase it is never told about is a phase it does not
        // have, and an admitted request it never hears of again is one it can
        // never schedule work for.
        let _ = self.refresh_views();
        // The policy's knowledge and the mechanism's own must be the same: a
        // request the mechanism holds and the policy never heard of is one it
        // cannot schedule work for, and a request the policy still holds after the
        // mechanism ended it is one it offers in `admit` and the mechanism
        // refuses as "not queued" - a policy failure, which stops the epoch.
        #[cfg(debug_assertions)]
        self.assert_policy_in_sync();
        let room = self.room();
        // A running request that has not advanced within the policy's timeout
        // is the policy's failure, not the request's (design §4.4) - but one
        // request's silence is not the deployment's (§4.4's liveness rule is
        // per request): the request fails by name and the others are served.
        // The epoch ends only when nothing else could have been served, which
        // is the case a reader must be able to see from the log.
        let now = scheduling::guest::Nanos(crate::clock::now_monotonic_us().saturating_mul(1_000));
        self.starvation_judged = now;
        let starved = {
            let live = self.live();
            self.watchdog
                .starved(&live, now, policy.limits().timeout)
                .map(|request| (request, self.starvation_report(&live, now)))
        };
        if let Some((request, report)) = starved {
            // What could have been served instead: another running request, or a waiting request. The
            // watchdog's own view names only the requests it tracks, so this is read from the
            // mechanism's queues rather than from that view.
            let others = self.running.len() > 1 || !self.waiting.is_empty();
            if others {
                tracing::warn!(
                    request = request.0,
                    reason = %report,
                    "a request starved past the policy's timeout: failing that request, not the epoch"
                );
                self.finish_with(RequestId(request.0), FinishReason::Error, None);
            } else {
                tracing::error!(
                    request = request.0,
                    reason = %report,
                    "no batch was formed for anything running"
                );
                self.fail(format!(
                    "no batch formed for {} running request(s) at step {}: {report}",
                    self.running.len(),
                    self.step,
                ));
                self.policy = Some(policy);
                return None;
            }
        }
        let events = self.recorder.take();
        if let Err(error) = policy.observe(&events) {
            return self.policy_failed(policy, error);
        }
        let output = match policy.dispatch(&room) {
            Ok(output) => output,
            Err(error) => return self.policy_failed(policy, error),
        };
        // The admission the output states is checked against the queue the policy
        // was given, and applied before the rest of the output is checked: a
        // request the output admits is running for the work the same output gives
        // it, which is what lets one output both admit and run a request (§4.4).
        // Applying it first would make its own names unqueued.
        let live = self.live();
        if let Err(error) = mechanism::validate_admission(&output, &live) {
            return self.policy_failed(policy, error);
        }
        self.admit(&output);
        // The rest is read against the state admission has left: the admitted
        // requests' views moved, and the pages are the scheduler's own table
        // borrowed for the check (design §4.4).
        let _ = self.refresh_views();
        let live = self.live();
        if let Err(error) = mechanism::validate_after_admission(&output, &room, &live) {
            return self.policy_failed(policy, error);
        }
        self.policy = Some(policy);
        Some(output)
    }

    /// Asserts that the policy knows exactly the requests the mechanism holds.
    ///
    /// A policy's state is append-only and it learns requests only from this
    /// mechanism's events, so the two sets are one fact. The mechanism holds a
    /// request from its arrival (`waiting`) through its admission (`running`)
    /// to its end, and every one of those
    /// transitions is announced (`Arrived`, `Admitted`, `Finished`). A step
    /// boundary is the one place both are complete: nothing is mid-transition
    /// there.
    ///
    /// Debug builds only: it is a statement about the mechanism's bookkeeping,
    /// not a decision the mechanism makes.
    ///
    /// It also asserts that every view the policy was told is the request's
    /// view now: the step states only the views of the requests marked as
    /// moved, so a view that moved without a mark is one the policy never
    /// learns.
    #[cfg(debug_assertions)]
    fn assert_policy_in_sync(&self) {
        // The policy's ids and the mechanism's name the same request; the two
        // newtypes are the interface's, so the sets are compared in the
        // interface's.
        let guest = |id: u64| scheduling::guest::RequestId(id);
        let held: std::collections::BTreeSet<scheduling::guest::RequestId> = self
            .waiting
            .iter()
            .chain(&self.running)
            .map(|id| guest(id.0))
            .collect();
        let told: std::collections::BTreeSet<scheduling::guest::RequestId> =
            self.recorder.known().collect();
        for id in held.difference(&told) {
            debug_assert!(
                false,
                "the policy was never told of request {}: the mechanism holds it, so nothing can                          output its work",
                id.0
            );
        }
        for id in told.difference(&held) {
            debug_assert!(
                false,
                "the policy still holds request {}: the mechanism ended it without saying so",
                id.0
            );
        }
        for id in held.intersection(&told) {
            let current = views::view_of(self, RequestId(id.0), self.recorder.recorded());
            debug_assert_eq!(
                current.as_ref(),
                self.recorder.told().get(id),
                "the view of request {} moved without being marked",
                id.0
            );
        }
    }

    /// States to the recorder the view of every request whose view may have
    /// moved since the last call, which `close` turns into one `Updated` per
    /// view that differs from what the policy was told.
    ///
    /// The marked requests are the running requests accessed mutably, the requests
    /// whose calls entered or left flight, and the admitted ones
    /// (`request::Requests`). A media request's next ready call also depends on the
    /// calls other requests hold in flight on a shared stage, so every
    /// running media request is stated each step; they are few, and a token
    /// running is never visited.
    ///
    /// Returns whether any view differed: the policy has something new to
    /// decide on.
    fn refresh_views(&mut self) -> bool {
        let mut changed = self.requests.take_changed();
        changed.extend(self.requests.running_video().map(|(id, _, _)| id));
        for id in changed {
            if let Some(view) = views::view_of(self, id, self.recorder.recorded()) {
                self.recorder.update(view);
            }
        }
        self.recorder.close()
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
    pub(super) fn policy_failed(
        &mut self,
        mut policy: policy::PolicySlot,
        error: PolicyError,
    ) -> Option<DispatchOutput> {
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
                self.seed_from_policy();
            }
            Err(fatal) => {
                // Both failures are named: the one that ended the policy, which
                // is what the daemon and a reader of the log need first, and the
                // failed takeover that leaves the epoch with no policy at all.
                tracing::error!(
                    reason = %error,
                    takeover = %fatal,
                    "the policy failed and cannot be reverted"
                );
                self.fail(format!(
                    "the scheduling policy failed with `{error}`, and no previous policy can \
                     take over: {fatal}"
                ));
                self.policy = Some(policy);
            }
        }
        None
    }

    /// One line per step: what it prepared and what became a batch.
    ///
    /// A step that serves nothing is the interesting one - a starvation is
    /// otherwise unfalsifiable - so the line names what was running, what the
    /// output asked for, and how much of it became calls. The rows' classes are
    /// the calls' own kinds, which is what a refusal is about (a mixed phase or
    /// a mixed sampling class).
    fn log_step(&self, output: &DispatchOutput, batches: &[ExecutionBatch]) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let calls: usize = batches.iter().map(|batch| batch.requests.len()).sum();
        if calls == 0 {
            tracing::debug!(
                step = self.step,
                running = self.running.len(),
                queued = self.waiting.len(),
                prepared = output.work.len() + output.refill.len(),
                admitted = output.admit.len(),
                in_flight =
                    self.inflight.pending_batches.len() + self.inflight.pending_submissions.len(),
                "a scheduler step formed no batch"
            );
            return;
        }
        // Only when the line is enabled: a step is the hot path, and the sums and the kinds vector
        // exist for the log line and nowhere else.
        let tokens: usize = batches
            .iter()
            .flat_map(|batch| batch.requests.iter())
            .map(|(call, _)| call.input_token_ids.len())
            .sum();
        let kinds: Vec<ipc::CallKind> = batches
            .iter()
            .flat_map(|batch| batch.requests.iter())
            .map(|(call, _)| call.kind)
            .collect();
        tracing::debug!(
            step = self.step,
            batches = batches.len(),
            calls,
            tokens,
            ?kinds,
            "a scheduler step formed its batches"
        );
    }

    /// The starvation rule's report: what was running, in which state, and how
    /// long each request has gone without advancing.
    ///
    /// A starvation with a healthy device must never end a process without
    /// saying what was running (§4.4), so the report is read by the failure
    /// itself and by the log line above it.
    fn starvation_report(
        &self,
        live: &mechanism::Live<'_>,
        now: scheduling::guest::Nanos,
    ) -> String {
        let mut text = format!(
            "{} running, {} waiting, {} in flight, {} waiting to be submitted, {} of {} cache \
             units free",
            self.running.len(),
            self.waiting.len(),
            self.inflight.pending_batches.len(),
            self.inflight.pending_submissions.len(),
            self.storage.free_units(),
            self.storage.capacity_units(),
        );
        for (id, since) in self.watchdog.advancing() {
            let age_ms = now.0.saturating_sub(since.0) / 1_000_000;
            let guest = scheduling::guest::RequestId(id.0);
            match live.requests.get(&guest) {
                Some(view) => {
                    text.push_str(&format!(
                        "; request {} is {:?}{} and last advanced {} ms ago",
                        id.0,
                        view.state,
                        if view.waiting_transfer {
                            ", waiting for a transfer"
                        } else {
                            ""
                        },
                        age_ms,
                    ));
                    if let Some(state) = self.requests.tokens(&RequestId(id.0)) {
                        text.push_str(&format!(
                            ", {} of its {} prompt tokens computed in {} units",
                            state.num_computed_prompt_tokens,
                            state.req.prompt_token_ids.len(),
                            state.block_tables().map_or(0, |tables| tables
                                .iter()
                                .map(BlockTable::held_units)
                                .sum::<usize>()),
                        ));
                    }
                }
                None => text.push_str(&format!(
                    "; request {} is not in the live set and last advanced {} ms ago",
                    id.0, age_ms,
                )),
            }
        }
        text
    }

    /// This step's room.
    ///
    /// `views.rs` builds every fact but one: the captured graph tiers, which
    /// the build config states (the deployment's `cuda_graph_config`).
    fn room(&self) -> Room {
        let mut room = views::room(self, self.step);
        room.graph_tiers = self.config.graph_tiers.clone();
        room
    }

    /// The mechanism's state at a batch boundary, owned, for the switch.
    fn trial_state(&self) -> policy::TrialState {
        policy::TrialState {
            snapshot: views::snapshot(self, self.step, self.recorder.recorded()),
            room: self.room(),
            shortage: views::shortage_now(self),
            arrival: None,
            call: views::dispatch(self),
        }
    }

    /// Announces the pages that appeared and went since the previous step, and
    /// keeps the mechanism's table of live pages current (design §4.4).
    ///
    /// The pool logs both changes, so a step costs what changed rather than the
    /// size of the prefix cache: the page table is built once, from the epoch's
    /// snapshot, and then only by these two events.
    fn pump_page_changes(&mut self) {
        let Some(cache) = self.storage.cache() else {
            return;
        };
        let (published, freed) = cache.pool().take_page_changes();
        if published.is_empty() && freed.is_empty() {
            return;
        }
        if !self.pages_seeded {
            self.pages = views::page_index(self);
            self.pages_seeded = true;
        }
        let announced: Vec<scheduling::guest::PageView> = published
            .into_iter()
            .filter_map(|(head, tokens)| {
                let units = cache.pool().page_units(head)?;
                views::page_of(self, &units, tokens)
            })
            .collect();
        for page in announced {
            self.pages.insert(page.id, page.clone());
            self.recorder.cached(page);
        }
        for head in freed {
            let page = views::device_page(head);
            self.pages.remove(&page);
            self.recorder.freed(page);
        }
    }

    /// The objects the validators read, as they stand now.
    ///
    /// The requests are the views the policy was told, which
    /// [`Self::refresh_views`] keeps current, and the pages the scheduler's own
    /// table, kept current by the page delta: a step borrows both rather than
    /// rebuilding a view of every request or copying every page.
    pub(crate) fn live(&self) -> mechanism::Live<'_> {
        let mut live = mechanism::Live {
            requests: std::borrow::Cow::Borrowed(self.recorder.told()),
            pages: std::borrow::Cow::Borrowed(&self.pages),
            ..mechanism::Live::default()
        };
        for entry in views::encodings(self) {
            live.encodings.insert(entry.id, entry);
        }
        for phase in views::phases(&self.info) {
            live.phases.insert(phase.id, phase);
        }
        // What the policy was told is the recorder's own account of it: the
        // snapshot it was instantiated with, plus every page, encoder entry and
        // request announced since, less what it was told is gone. A preemption is
        // judged against that, not against the mechanism's table, which may
        // have moved on since the policy last heard.
        live.told_pages = std::borrow::Cow::Borrowed(self.recorder.announced_pages());
        live.told_encodings = std::borrow::Cow::Borrowed(self.recorder.announced_encodings());
        live.told_requests = std::borrow::Cow::Borrowed(self.recorder.announced_requests());
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
        self.seed_from_policy();
    }

    /// States the page table and the recorder from the instance that serves
    /// now, which every change of instance does: a mechanism that kept the
    /// table of the instance before it would judge the new one's responds by a
    /// state it was never told.
    pub(super) fn seed_from_policy(&mut self) {
        // The epoch's page table is read once, here, so every later step costs
        // what changed: the policy was handed the same table in its snapshot.
        self.pages = views::page_index(self);
        self.pages_seeded = true;
        // The instance was instantiated on a snapshot and has never been
        // observed. Seeding the recorder from that same snapshot is what keeps
        // the next step's events proportional to what changed after it, rather
        // than one `Updated` per live request to a policy that already knows.
        if let Some(policy) = self.policy.as_ref() {
            let snapshot = policy.instantiated_on().clone();
            self.recorder = mechanism::Recorder::from_snapshot(&snapshot);
        }
    }

    /// Tells the policy that the encoder entry `key` is live, with the last use
    /// the cache now records for it (§4.4's `Encoded`).
    ///
    /// Called where the cache creates, re-activates, or re-uses an entry: a
    /// re-use raises its last use, which is the order a policy's LRU reads.
    pub(super) fn record_encoded(&mut self, key: [u8; 32]) {
        let Some(last_used) = self.storage.encoder_cache.last_used(&key) else {
            return;
        };
        let entry = views::encoding_view(
            views::encoding_id(&key),
            scheduling::guest::Nanos(last_used),
        );
        self.recorder.encoded(entry);
    }

    /// The report of a policy failure, for the control thread to hand up.
    pub(super) fn take_policy_report(&mut self) -> Option<policy::PolicyReverted> {
        self.policy_report.take()
    }

    /// Allocates the completion product of a state-advancing media call that
    /// declares no outputs.
    ///
    /// `prepare_media_call` requests one for latent preparation and every
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
            let pool = prefix_cache.pool().stats();
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
                .store(prefix_cache.pool().cached_pages(), Ordering::Relaxed);
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

    /// Takes at most one completed batch, so the next pass can refill the slot
    /// it freed before another completion is taken.
    pub(super) fn take_completed(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.take_completed").entered();
        match self.worker.take_completed() {
            Some((batch, outcome)) => {
                self.update(batch, outcome);
                true
            }
            None => false,
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
        // A call in flight changes what its request's view states (its next
        // phase, whether it waits for a transfer).
        self.requests.mark(request_id);
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
    /// A return with no active credit is logged as an accounting defect and
    /// changes no counter.
    pub(super) fn return_domain_credit(&self, kind: ipc::CallKind, failed: bool) {
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

    /// Preempts the domain credit of every in-flight call as failed.
    ///
    /// Callers invoke it before `Inflight::clear_failed_calls` drops the
    /// calls whose credits it returns.
    pub(super) fn fail_inflight_domain_credits(&self) {
        for inflight in self.inflight.pending_calls.values().flatten() {
            self.return_domain_credit(inflight.call.kind, true);
        }
    }
}
