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
//!   device lanes and, for media reading, encoding and muxing, host lanes,
//!   tracked by the lane ledger, and `plan_media_call` builds one call with
//!   its products, buffer bindings, latent pages and reading components.
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
use uniserve_ipc::{BufferId, MediaCall};

pub(super) use media::LaneLedger;

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
        progressed |= self.poll_page_tier();

        progressed |= self.refill_executor();

        self.stats
            .general
            .in_flight
            .store(self.inflight.pending_batches.len(), Ordering::Relaxed);
        self.publish_cache_stats();
        progressed
    }

    /// Takes the host page tier's answers, and ends the pins of requests
    /// that left the scheduler. Returns whether admission pins were
    /// answered.
    fn poll_page_tier(&mut self) -> bool {
        let Some(tiers) = self.tiers.as_mut() else {
            return false;
        };
        let answered = tiers.poll();
        let waiting: HashSet<RequestId> = self
            .waiting
            .iter()
            .map(|state| state.req.request_id)
            .collect();
        let running = &self.running;
        tiers.sweep(|request| waiting.contains(&request) || running.contains_key(&request));
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
        self.placement.affinity.retain(|(request, _), _| {
            let id = request.request_id;
            self.running.contains_key(&id)
                || self.running_media.contains_key(&id)
                || self.retiring_requests.contains_key(&id)
        });

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

    /// Selects the batches of one pass, alternating between video media work
    /// and generation work. A pass yields one batch per `(kind, component)` it
    /// selected, each a single numerical call.
    ///
    /// `prefer_media` is set after a generation pass yields batches and
    /// cleared after a media pass does; either kind runs when the other has
    /// nothing to schedule.
    fn schedule_batches(&mut self) -> Vec<ExecutionBatch> {
        // A paused scheduler lets what is in flight drain before it hands
        // over, and a closing one before its workers close.
        if self.paused() || self.closing {
            return Vec::new();
        }
        if self.prefer_media {
            let batches = self.prepare_media_batches();
            if !batches.is_empty() {
                self.prefer_media = false;
                return batches;
            }
        }
        let batches = self.assemble();
        if batches.is_empty() {
            if self.fatal {
                return Vec::new();
            }
            let batches = self.prepare_media_batches();
            if !batches.is_empty() {
                self.prefer_media = false;
            }
            return batches;
        }
        self.prefer_media = true;
        batches
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
    pub(super) fn record_domain_backpressure(&self, kind: uniserve_ipc::CallKind) {
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
    pub(super) fn record_domain_completion(
        &self,
        kind: uniserve_ipc::CallKind,
        status: CallStatus,
    ) {
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
    pub(super) fn reclaim_domain_credit(&self, kind: uniserve_ipc::CallKind, failed: bool) {
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
