//! A scheduler's step (`specs/scheduler-design.md` §3.2 to §3.6): a request's arrival
//! ([`Scheduler::add_request`]), the step that assembles the batches that fit now
//! ([`Scheduler::schedule`]), the admission and placement of waiting requests, the blocks running
//! requests ask for ahead of their next call, and preemption and starvation.
//!
//! The step is vLLM's `schedule()`: the policy names the step's work in its order, and the
//! running requests are visited first, then the waiting queue. Everything the policy decides is
//! checked where it is used: `charge` is the one place a decision meets the limits and the step's
//! budgets, and an id in the wrong state is skipped. Nothing here waits for the buffer pool: a
//! request whose blocks have not arrived is left out of the step, an admission is a round of
//! requests whose answers arrive in later passes, and a shortage is relieved by the policy's
//! victims while the short request waits for the region's `available` word.
//!
//! Regions are named here, as in every per-region field of [`Scheduler`], by their index into the
//! session's regions.

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use ipc::batch::{Batch, BatchId, Columns, Extent, Phase, Shape};
use proto::request::{Event, FinishReason, MigrationCause, RejectionKind, RequestId};
use scheduling_guest::decision::{Shortage, Storage, Victim};
use scheduling_guest::ids::{EncodingId, PageId, PhaseId};
use scheduling_guest::{Duration as Span, Event as PolicyEvent};
use storage::Key;
use storage::runtime::{self, GroupId, OwnerId, Tier};

use super::cache::HOST;
use super::group::variants;
use super::update::migratable;
use super::view::{call_view, replica_views, view_of};
use super::{Error, Request, RequestStatus, Scheduler, WorkerGroup};

/// The rejection of a request that arrives while the waiting queue is at its bound (SD §7.2).
const QUEUE_FULL: &str = "scheduler waiting queue is full";

/// The rejection of a waiting request whose least need exceeds the whole cache (SD §7.2).
const OVER_CAPACITY: &str = "request exceeds total KV capacity";

/// The rejection of a waiting request the policy named in `reject` (SD §7.2).
const POLICY_REJECTED: &str = "the scheduling policy rejected the request as over capacity";

/// The phases an empty batch is formed for, in the order they are preferred: the runner of the
/// phase issues the group's queued copies, and the token and denoising runners hold the memory
/// that the copies of `Load`, `Clone`, `Store`, `Save` and `Restore` read and write.
const COPY_PHASES: [Phase; 4] = [Phase::Prefill, Phase::Decode, Phase::Verify, Phase::Denoise];

/// A step's budgets: the tokens and the calls still to be charged.
type Budget = (u32, usize);

/// A request's arrival and the step (SD §3.2, §3.4).
impl Scheduler {
    /// Whether request `id` takes calls this step: admitted and in no queue, running or waiting
    /// for the transfer of its blocks to the group that decodes it.
    pub(super) fn is_running(&self, id: RequestId) -> bool {
        self.requests.get(&id).is_some_and(|request| {
            matches!(
                request.status,
                RequestStatus::Running | RequestStatus::WaitingForRemoteKvs
            ) && !request.queued
        })
    }

    /// Takes a request from the daemon (SD §3.2). One that fails its validation is rejected as
    /// `Invalid`, and one that arrives while the waiting queue is at the policy's bound as
    /// `Overloaded`; neither is kept. Any other is queued: a resumed request behind the other
    /// resumed requests and ahead of every new one, a new one last. The policy is told of it as
    /// `Arrived`.
    pub fn add_request(&mut self, input: proto::request::Request) {
        let id = input.request_id();

        // Refusals answer at once and keep nothing.
        if let Err(message) = self.validate(&input) {
            let kind = RejectionKind::Invalid;
            return self.send(id, Event::Rejected { kind, message });
        }
        if self.queue_full() {
            let kind = RejectionKind::Overloaded;
            let message = QUEUE_FULL.to_owned();
            return self.send(id, Event::Rejected { kind, message });
        }

        // A resumed request's generated tokens continue its prompt; it is named for its
        // continuation by this epoch unless it carries a name.
        let request = Request::new(input, self.epoch, Instant::now());
        let at = if request.resumes() {
            self.waiting
                .iter()
                .position(|queued| !self.requests.get(queued).is_some_and(Request::resumes))
                .unwrap_or(self.waiting.len())
        } else {
            self.waiting.len()
        };
        self.waiting.insert(at, id);

        let view = view_of(&request, &self.groups);
        self.policy
            .events
            .push(PolicyEvent::Arrived { request: view });
        self.requests.insert(id, request);
    }

    /// Whether the waiting queue is at the bound the serving policy declared.
    fn queue_full(&self) -> bool {
        self.policy
            .limits
            .max_queued_requests
            .is_some_and(|bound| self.waiting.len() >= bound)
    }

    /// Assembles the batches that fit now (SD §3.4, vLLM `Scheduler.schedule`): at most one per
    /// group and phase (per rank for a media phase whose rows name one), no more for a group than
    /// its queue holds beside its batches in flight, each sealed with its number, generation and
    /// shape.
    ///
    /// The policy is told what changed and asked for the step's decision; a fault rolls it back
    /// and forms nothing. The running requests named in the decision's work are scheduled in its
    /// order, then the draft refills it names; then every running request whose blocks no longer
    /// cover its next call and the low watermark asks for more, ahead of the step's admissions;
    /// then the blocks the policy offloads are stored to the host and the waiting requests it
    /// names are prefetched for; then, unless the scheduler drains, the waiting requests it
    /// rejects leave and those it admits start their admission, resumed requests first and in
    /// its order otherwise, until the first that does not fit. A group with copies its executor has not finished, for which the step
    /// formed no batch, is sent an empty one, which runs the copies (SD §2.1).
    ///
    /// Nothing is formed while every group's queue is full, and nothing once the scheduler has
    /// failed (a starved request alone, or a batch the step could not seal).
    pub fn schedule(&mut self) -> Vec<Batch> {
        // No group has room in its queue: nothing can be formed, and the policy is not asked.
        if self.groups.iter().all(|group| !queue_open(group)) {
            return Vec::new();
        }
        let started = Instant::now();
        self.fail_starved(started);
        if self.stopped {
            return Vec::new();
        }

        // The policy chooses the order and the work; everything below enforces the limits.
        let room = self.room();
        let output = match self
            .policy
            .dispatch(&room, &mut self.requests, &self.groups)
        {
            Ok(output) => output,
            Err(fault) => {
                self.policy_failed(fault);
                return Vec::new();
            }
        };
        let limits = self.policy.limits;
        let mut budget = (limits.max_num_batched_tokens, limits.max_batch);
        let mut batches: Vec<Batch> = Vec::new();

        // 1. Running requests first, in the policy's order (vLLM: "First, schedule the RUNNING
        //    requests"). A work entry of a phase no request has is skipped.
        for work in &output.work {
            let id = RequestId(work.request.0);
            if let Some(phase) = phase_of(work.phase)
                && self.is_running(id)
            {
                self.schedule_call(id, phase, work.tokens, &mut budget, &mut batches);
            }
        }

        // Draft-cache refills take the prompt budget the running requests left.
        for refill in &output.refill {
            let id = RequestId(refill.0);
            if self.is_running(id) {
                self.schedule_call(id, Phase::Refill, 0, &mut budget, &mut batches);
            }
        }

        // Blocks ahead of the running requests, held ahead of the step's admissions (SD §4.4).
        self.allocate_ahead();
        self.offload(&output.offload);
        for prefetch in &output.prefetch {
            self.prefetch(RequestId(prefetch.0));
        }

        // 2. Then the waiting queue, in the policy's order, unless the scheduler drains.
        if self.draining.is_none() {
            for reject in &output.reject {
                let kind = RejectionKind::Overloaded;
                self.reject_waiting(RequestId(reject.0), kind, POLICY_REJECTED);
            }
            // Resumed requests are tried before every request that starts (SD §3.2, §7.2 "The
            // queue's order"). A view states no resume, so the policy's order holds among the
            // resumed requests and among the others.
            let mut admitted: Vec<RequestId> = output
                .admit
                .iter()
                .map(|admit| RequestId(admit.0))
                .collect();
            admitted.sort_by_key(|id| !self.requests.get(id).is_some_and(Request::resumes));
            for id in admitted {
                // The first that does not fit stops the queue.
                if !self.admit(id) {
                    break;
                }
            }
        }

        // A batch the step's rows broke is a defect of the scheduler's own.
        let mut sealed = Vec::with_capacity(batches.len());
        for mut batch in batches {
            if let Err(error) = self.seal(&mut batch) {
                let id = batch.id.0;
                self.fail(Error::InvalidRequest(format!(
                    "the step formed batch {id}, which breaks a rule of batches: {error}"
                )));
                return Vec::new();
            }
            sealed.push(batch);
        }
        let empty = self.copy_batches(&sealed);
        sealed.extend(empty);

        // The policy is told the step ended; its next `observe` begins at the next step, which
        // the loop numbers.
        let duration = Span(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        self.policy.events.push(PolicyEvent::Stepped {
            step: self.step.saturating_add(1),
            duration,
        });
        sealed
    }

    /// One call of `phase` of request `id` joins the batch of its group and phase in `batches`,
    /// formed when the step has none yet (SD §3.4). True when it joined.
    ///
    /// The request must be ready for the call, its consumer's window must hold the call's events,
    /// its group must have room in its queue and in the stage that serves the phase, and `charge`
    /// must find room in the step's budgets. A call that writes the cache must lie within the blocks the request holds
    /// and can write; otherwise the budget is returned, the call is counted as left out for want
    /// of blocks, and the request runs once they arrive. A row the request's state cannot state
    /// is left out the same way. The call advances the request when it is scheduled (vLLM
    /// `_update_after_schedule`): its `scheduled` count, and the tokens a prefill, decode or
    /// verification computes.
    fn schedule_call(
        &mut self,
        id: RequestId,
        phase: Phase,
        asked: u32,
        budget: &mut Budget,
        batches: &mut Vec<Batch>,
    ) -> bool {
        let Some(request) = self.requests.get(&id) else {
            return false;
        };
        let Some(group) = request.placement.get(phase.index()).copied().flatten() else {
            return false;
        };

        // The request's own state says whether it has such a call, and the consumer's window
        // whether its output fits.
        if !request.ready(phase, &self.groups) || !request.has_event_room(phase) {
            return false;
        }

        // The group has room in its queue and in the stage that serves the phase; one call per
        // request and phase in a step.
        let Some(entry) = self.group(group) else {
            return false;
        };
        let taken = batches
            .iter()
            .any(|batch| batch.phase == phase && batch.requests.contains(&id));
        if !queue_open(entry) || !entry.has_room(phase, request) || taken {
            return false;
        }

        // The one place a policy's decision meets the limits and the step's budgets.
        let Some(tokens) = self.charge(request, phase, asked, budget) else {
            return false;
        };

        // What the call writes lies within the blocks the request holds and can write; beyond
        // them, its `Allocate` is outstanding and nothing waits for it.
        if is_token(phase) && !self.holds_blocks_for(request, phase, tokens) {
            refund(budget, tokens);
            self.count_without_blocks();
            return false;
        }

        // The request's row, in the step's batch of its group and phase.
        if !self.join_batch(batches, id, group, phase, tokens) {
            refund(budget, tokens);
            return false;
        }

        // Advance when scheduled; `update` corrects only what a verification rejected.
        if let Some(request) = self.requests.get_mut(&id) {
            request.scheduled[phase.index()] += 1;
            if matches!(phase, Phase::Prefill | Phase::Decode | Phase::Verify) {
                request.num_computed_tokens += tokens;
            }
        }
        self.mark(id);
        if let (Some(index), Some(request)) = (self.group_index(group), self.requests.get(&id)) {
            self.groups[index].occupy(phase, request);
        }
        true
    }

    /// Appends request `id`'s row of `phase`, of `tokens` tokens, to the step's batch of `group`
    /// and `phase` while it fits a captured bucket with one more call, or to a batch formed for
    /// it. A media row runs on the one rank its batch names (`Batch::check`), so the rows of
    /// several ranks join batches of their own. A batch is formed only while the group's queue
    /// holds it beside its batches in flight and those the step formed for it, and a token phase
    /// has one batch per group in a step. False, with every batch as it was, when the row does not
    /// fit or the request's state does not state it.
    fn join_batch(
        &mut self,
        batches: &mut Vec<Batch>,
        id: RequestId,
        group: GroupId,
        phase: Phase,
        tokens: u32,
    ) -> bool {
        let Some(entry) = self.group(group) else {
            return false;
        };
        let mut full = false;
        for batch in batches.iter_mut() {
            if batch.group != group || batch.phase != phase {
                continue;
            }
            if !entry.capacity.holds(batch, phase) {
                full = true;
                continue;
            }
            if is_token(phase) {
                return self.push_token_rows(batch, id, phase, tokens);
            }

            // A media row of another rank than the batch's leaves it as it was.
            let (requests, rows) = (batch.requests.len(), batch.media.len());
            if !self.push_media_row(batch, id, phase) {
                return false;
            }
            let rank = |row: Option<&ipc::batch::MediaRow>| row.map(|row| row.rank);
            if rank(batch.media.last()) == rank(batch.media.first()) {
                return true;
            }
            batch.requests.truncate(requests);
            batch.media.truncate(rows);
        }

        // A new batch, which the group's queue must hold.
        let formed = batches.iter().filter(|batch| batch.group == group).count();
        let room = (entry.in_flight as usize).saturating_add(formed) < entry.config.queue_depth;
        if (full && is_token(phase)) || !room {
            return false;
        }
        let Some(mut batch) = self.empty_batch(group, phase) else {
            return false;
        };
        let pushed = if is_token(phase) {
            self.push_token_rows(&mut batch, id, phase, tokens)
        } else {
            self.push_media_row(&mut batch, id, phase)
        };
        if pushed {
            batches.push(batch);
            self.next_batch += 1;
        }
        pushed
    }

    /// The tokens a call of `phase` takes from the step, charged to `budget`; `None` when it does
    /// not fit (SD §3.4, §7.1). A call costs one of the step's calls and:
    ///
    /// - a prefill chunk, what the policy asked (the chunk limit when it asked nothing), cut to
    ///   the chunk limit, the prompt left and the tokens left, then to the recurrent-state
    ///   checkpoints the request holds, with an image block taken whole or the chunk ending
    ///   before it, at least one token;
    /// - a refill, the chunk of the draft cache's gap the tokens left hold, ending on a kept
    ///   checkpoint when it stops short and cut to the checkpoints the request holds (none fits
    ///   when that chunk is empty);
    /// - a decode one token, a verification one and its drafts;
    /// - a media call no token: it runs at a budget of 0.
    fn charge(
        &self,
        request: &Request,
        phase: Phase,
        asked: u32,
        budget: &mut Budget,
    ) -> Option<u32> {
        if budget.1 == 0 {
            return None;
        }
        let limits = &self.policy.limits;
        let capacity = request
            .placement
            .get(phase.index())
            .copied()
            .flatten()
            .and_then(|group| self.group(group))
            .map(|group| &group.capacity);

        let tokens = match phase {
            Phase::Prefill => {
                let left = request
                    .num_prompt_tokens
                    .saturating_sub(request.num_computed_tokens);
                let chunk = match asked {
                    0 => limits.chunked_prefill_size,
                    asked => asked.min(limits.chunked_prefill_size),
                };
                // A model with a recurrent state ends the chunk where it keeps a checkpoint it
                // holds writable.
                let tokens = self.fit_chunk(request, phase, chunk.min(left).min(budget.0));
                request.clip_to_images(tokens).max(1)
            }
            Phase::Refill => {
                let state = self
                    .state_shape(request)
                    .map(|(class, block, _)| (class, block));
                // The rows of a refill start where the draft cache ends; a gap that restarts the
                // draft cache from a checkpoint it no longer holds is not one chunk.
                if request.refill_start(state) != request.num_draft_tokens {
                    return None;
                }
                let chunk = request.refill_chunk(budget.0, state);
                match self.fit_chunk(request, phase, chunk) {
                    0 => return None,
                    tokens => tokens,
                }
            }
            Phase::Decode => 1,
            Phase::Verify => {
                let eos = capacity.map_or(&[][..], |capacity| &capacity.eos_token_ids[..]);
                1 + request.drafts(eos)
            }
            _ => 0,
        };
        if tokens > budget.0 {
            return None;
        }
        budget.0 -= tokens;
        budget.1 -= 1;
        Some(tokens)
    }

    /// A batch of `phase` on `group` with no row yet, under the step's next batch number; `None`
    /// for a phase no batch is launched as.
    fn empty_batch(&self, group: GroupId, phase: Phase) -> Option<Batch> {
        let variant = variants(phase).first().copied()?;
        Some(Batch {
            id: BatchId(self.next_batch),
            group,
            phase,
            generation: self.generation.load(Ordering::Acquire),
            shape: Shape {
                rows: 0,
                tokens: 0,
                variant,
                extent: Extent::None,
            },
            requests: Vec::new(),
            columns: Columns::default(),
            media: Vec::new(),
            new_slots: Vec::new(),
        })
    }

    /// The empty batches of the step (SD §2.1): one for each group with room in its queue,
    /// nothing in flight and no batch among `formed`, whose executor holds copies not yet
    /// finished (`Scheduler::copies`: from its `Copy` until the executor returns its `Finish` or
    /// the scheduler gives it up). Its executor runs the copies its copy budget admits with it,
    /// so that a load an admission waits for never waits for other work. The batch names the
    /// first of [`COPY_PHASES`] the group serves, or else the first phase it serves.
    fn copy_batches(&mut self, formed: &[Batch]) -> Vec<Batch> {
        let copying: HashSet<GroupId> = self.copies.values().map(|&(group, _)| group).collect();
        let mut empty = Vec::new();
        for index in 0..self.groups.len() {
            let group = &self.groups[index];
            let id = group.id();
            let idle = group.in_flight == 0 && queue_open(group);
            if !idle || !copying.contains(&id) || formed.iter().any(|batch| batch.group == id) {
                continue;
            }
            let preferred = COPY_PHASES.into_iter().find(|phase| group.serves(*phase));
            let phase = preferred.or_else(|| Phase::ALL.into_iter().find(|p| group.serves(*p)));
            if let Some(batch) = phase.and_then(|phase| self.empty_batch(id, phase)) {
                empty.push(batch);
                self.next_batch += 1;
            }
        }
        empty
    }

    /// Asks for the blocks of every running request whose blocks asked for no longer cover its
    /// next call and the low watermark beyond it (SD §3.4, §4.4): one `Allocate` per such class,
    /// for the owner that holds the class, at the class's first position not asked for, with the
    /// blocks that restore the cover. `asked` rises by them, and so does what the region's
    /// `available` has given away this step. An `Allocate` of that owner and class still held is
    /// raised instead (`submit`); a class with one outstanding asks again once it is answered.
    ///
    /// A class whose regions' `available` does not cover all those blocks asks for the blocks
    /// the next call writes alone, and the low watermark waits for units. One whose regions do
    /// not cover even those is not asked: the request is short there (SD §3.6), and `preempt`
    /// runs for it. A request that waits for a region's units since its last shortage takes no
    /// further victims: it asks again once `available` shows them there, or once nothing given
    /// up is still on its way back to the region. The low watermark and the draft cache take
    /// only the units the regions have free, and the units other requests wait for there are
    /// not: what a shortage made others give up goes to the request that is short, not to
    /// another request's headroom or draft cache, its own draft table's owner's included.
    ///
    /// Every running request is visited, those the step left out for want of blocks included:
    /// one comparison per request and class.
    fn allocate_ahead(&mut self) {
        // The owners and classes with an `Allocate` outstanding, which the buffer pool answers
        // before they ask again.
        let outstanding: HashSet<(OwnerId, u8)> = self
            .outstanding
            .values()
            .filter_map(|(_, request, _)| match *request {
                runtime::Request::Allocate { owner, class, .. } => Some((owner, class)),
                _ => None,
            })
            .collect();

        let running: Vec<RequestId> = self.running.clone();
        self.short.retain(|id, _| running.contains(id));
        for id in running {
            // A request waiting for its transfer names an owner the buffer pool moves.
            let runs = self
                .requests
                .get(&id)
                .is_some_and(|request| request.status == RequestStatus::Running);
            if !runs {
                continue;
            }
            // Waiting for units since its last shortage: it asks again once they show, or once
            // nothing given up is still on its way back, when it is short again.
            if let Some(&(region, units)) = self.short.get(&id) {
                if self.available.get(region).copied().unwrap_or(0) < units
                    && self.returning(region)
                {
                    continue;
                }
                self.short.remove(&id);
            }
            let Some(request) = self.requests.get(&id) else {
                continue;
            };
            let Some(owner) = self.kv_owner(request) else {
                continue;
            };

            for (class, at, count, needed) in self.ahead(request) {
                if count == 0 || outstanding.contains(&(owner, class)) {
                    continue;
                }
                let Some(regions) = self.class_regions(owner, class) else {
                    continue;
                };
                let each = self.block_units(class, 0).unwrap_or(0);
                // The units of `region` the other short requests wait for.
                let waited = |region: usize| {
                    self.short
                        .iter()
                        .filter(|&(other, &(at, _))| *other != id && at == region)
                        .map(|(_, &(_, units))| units)
                        .fold(0, u32::saturating_add)
                };
                // The region among the class's whose units do not cover `blocks`: of those left
                // free beside what other requests wait for, or, with `waits` false, of all it
                // has available.
                let short = |blocks: u32, waits: bool| {
                    let units = blocks.saturating_mul(each);
                    regions.iter().copied().find(|&region| {
                        let available = self.available.get(region).copied().unwrap_or(0);
                        let free = if waits {
                            available.saturating_sub(waited(region))
                        } else {
                            available
                        };
                        free < units
                    })
                };
                // Every block asked for, or else those the next call writes. The low watermark
                // and the draft cache take only the units the regions have free and wait for
                // them; a next call whose own blocks are not covered leaves the request short.
                let count = match (short(count, true), short(needed, false)) {
                    (None, _) => count,
                    (Some(_), None) if needed > 0 => needed,
                    (Some(_), Some(region)) => {
                        if let Ok(region) = u8::try_from(region) {
                            self.preempt(id, region);
                        }
                        break;
                    }
                    (Some(_), None) => continue,
                };
                let units = count.saturating_mul(each);

                for &region in &regions {
                    if let Some(available) = self.available.get_mut(region) {
                        *available = available.saturating_sub(units);
                    }
                }
                let allocate = runtime::Request::Allocate {
                    owner,
                    class,
                    at,
                    count,
                    units: 0,
                };
                self.count_asked(id, class, at, count);
                self.submit(allocate);
            }
        }
    }

    /// Whether units given up may still come back to `region` (an index): the buffer pool
    /// retires some there, a finished request still holds its blocks, or a `Release`, a `Detach`
    /// or an `Evict` has not been answered.
    fn returning(&self, region: usize) -> bool {
        let gives_up = |request: &runtime::Request| {
            matches!(
                request,
                runtime::Request::Release { .. }
                    | runtime::Request::Detach { .. }
                    | runtime::Request::Evict { .. }
            )
        };
        self.retiring.get(region).is_some_and(|units| *units > 0)
            || self
                .requests
                .values()
                .any(|request| request.status == RequestStatus::Finished)
            || self.held.iter().any(gives_up)
            || self
                .stamping
                .values()
                .any(|(request, _, _)| gives_up(request))
            || self
                .outstanding
                .values()
                .any(|(_, request, _)| gives_up(request))
    }

    /// The policy's `offload` (SD §3.4): one `Store` to the host per group whose memory holds
    /// the device copies of the blocks it names, with their keys in its order. A `PageId` that is
    /// not in `cache`, not on the device, already on the host or being stored there (a block is
    /// offered to the host once, SD §4.4), or of a class without a host region is skipped.
    fn offload(&mut self, pages: &[PageId]) {
        // The keys a `Store` to the host held or outstanding names: until its answer, `cache`
        // does not show them on the host.
        let storing: HashSet<Key> = self
            .held
            .iter()
            .chain(self.outstanding.values().map(|(_, request, _)| request))
            .filter_map(|request| match request {
                runtime::Request::Store {
                    keys,
                    tier: Tier::Host,
                    ..
                } => Some(keys),
                _ => None,
            })
            .flatten()
            .copied()
            .collect();

        let mut stores: Vec<(GroupId, Vec<Key>)> = Vec::new();
        for page in pages {
            let Some(row) = self.cache.get(&page.0) else {
                continue;
            };
            let hosted = self
                .classes
                .get(usize::from(row.class))
                .is_some_and(|class| class.host.is_some());
            if row.tiers & HOST != 0 || !hosted || storing.contains(&row.key) {
                continue;
            }
            let Some(group) = self.holder(row) else {
                continue;
            };

            match stores.iter_mut().find(|(store, _)| *store == group) {
                Some((_, keys)) => keys.push(row.key),
                None => stores.push((group, vec![row.key])),
            }
        }

        for (group, keys) in stores {
            self.submit(runtime::Request::Store {
                group,
                keys,
                tier: Tier::Host,
            });
        }
    }

    /// The policy's `prefetch` of request `id` (SD §3.4): one `Prefetch` per source and class, by
    /// the rule of SD §4.4, for a request that waits and has not started its admission, unless
    /// one of those keys is still being fetched; any other id is skipped.
    fn prefetch(&mut self, id: RequestId) {
        let waits = self
            .requests
            .get(&id)
            .is_some_and(|request| request.queued && request.status == RequestStatus::Waiting);
        if !waits {
            return;
        }
        self.key(id);

        let fetching: HashSet<Key> = self
            .held
            .iter()
            .chain(self.outstanding.values().map(|(_, request, _)| request))
            .filter_map(|request| match request {
                runtime::Request::Prefetch { keys, .. } => keys.first().copied(),
                _ => None,
            })
            .collect();
        let asks = match self.requests.get(&id) {
            Some(request) => self.prefetch_requests(request, true),
            None => Vec::new(),
        };
        for ask in asks {
            if let runtime::Request::Prefetch { keys, .. } = &ask
                && keys.first().is_some_and(|key| fetching.contains(key))
            {
                continue;
            }
            self.submit(ask);
        }
    }

    /// Computes request `id`'s keys and its encoder outputs' keys (SD §4.3) when it has none.
    fn key(&mut self, id: RequestId) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        if request.keys.is_empty() && request.encodings.is_empty() {
            let (keys, encodings) = (self.keys_of(request), self.encodings_of(request));
            if let Some(request) = self.requests.get_mut(&id) {
                request.keys = keys;
                request.encodings = encodings;
            }
        }
    }

    /// Ends the first running request, by id, that made no progress within the policy's timeout
    /// (SD §3.6, §7.2 "Starvation bound"): with `FinishReason::Error` when another request runs
    /// or waits, which could have been served instead; otherwise nothing was formed for anything
    /// running, and the scheduler fails. A request waiting for its transfer is exempt.
    fn fail_starved(&mut self, now: Instant) {
        let timeout = Duration::from_nanos(self.policy.limits.timeout.0);
        let starved = self
            .running
            .iter()
            .filter_map(|id| self.requests.get(id))
            .filter(|request| request.status == RequestStatus::Running)
            .filter(|request| now.saturating_duration_since(request.last_progress) > timeout)
            .map(|request| request.id)
            .min();
        let Some(id) = starved else {
            return;
        };

        if self.running.len() > 1 || !self.waiting.is_empty() {
            let terminal = self.finished(id, FinishReason::Error, None);
            self.finish(id, terminal);
        } else {
            self.fail(Error::Executor(format!(
                "no batch formed for {} running request(s) at step {}",
                self.running.len(),
                self.step
            )));
        }
    }
}

/// Admission and placement (SD §3.5).
impl Scheduler {
    /// Starts or continues the admission of waiting request `id` (SD §3.5). False when it does
    /// not fit: the queue stops here. True when it started a round, when its admission waits for
    /// answers (later requests pass it), when it was rejected, and for an id that names no
    /// request of the waiting queue, which is skipped.
    ///
    /// A request is admitted only while the running requests and the admissions under way are
    /// fewer than `max_running_requests`. It is placed on a group for each phase it runs; one
    /// whose least need exceeds a region's capacity is rejected; one whose need, with the hit
    /// `cache` expects, does not fit what the regions have available beyond what the running
    /// requests still need is placed nowhere again and stops the queue. Otherwise one round
    /// starts, which the buffer pool takes in this order: an `Attach` with a new owner for every
    /// group it is placed on, then the pins, loads, clones, restores, prefetches and allocations
    /// of its expected hit (SD §4.4). It waits in `WaitingForRemoteKvs` while other requests are
    /// scheduled.
    fn admit(&mut self, id: RequestId) -> bool {
        let Some(request) = self.requests.get(&id) else {
            return true;
        };
        // Only a request of the waiting queue is admitted: a transfer also waits in
        // `WaitingForRemoteKvs`, but its request runs, and it ends only by its answer.
        if !request.queued {
            return true;
        }
        match request.status {
            RequestStatus::Waiting => {}
            // Answers of its admission are outstanding: later requests pass it.
            RequestStatus::WaitingForRemoteKvs if request.pending > 0 => return true,
            // Every answer arrived and positions are still missing: one more round asks for
            // them, or stops the queue.
            RequestStatus::WaitingForRemoteKvs => return self.allocate_admission(id),
            _ => return true,
        }
        if self.running.len().saturating_add(self.admitting()) >= self.config.max_running_requests {
            return false;
        }

        // A group for each phase it runs: the regions its classes take units from follow it.
        if !self.place(id) {
            return false;
        }
        let Some(request) = self.requests.get(&id) else {
            return true;
        };
        if self.never_fits(request) {
            let kind = RejectionKind::Invalid;
            self.reject_waiting(id, kind, OVER_CAPACITY);
            return true;
        }

        // The keys of its blocks, and the hit `cache` expects of them; the answers decide again,
        // and the buffer pool's `Full` is final.
        self.key(id);
        let Some(request) = self.requests.get(&id) else {
            return true;
        };
        let expected = self.expected(request);
        if !self.fits(request, expected) {
            self.requeue(id);
            return false;
        }

        // One admission is one submission: the buffer pool takes its requests in this order, so
        // each finds the owner and the positions the ones before it set up. Each owner is the
        // request's attachment to one group, drawn anew at every admission.
        let groups = placed_groups(request);
        for group in groups {
            // The slot the `Attach` takes is not free to the step's later admissions.
            if let Some(slots) = self
                .group(group)
                .map(|group| usize::from(group.layout.slots))
                && let Some(free) = self.available.get_mut(slots)
            {
                *free = free.saturating_sub(1);
            }
            let owner = self.draw_owner();
            self.owners.insert(owner, id);
            if let Some(request) = self.requests.get_mut(&id) {
                request.attachments.push((group, owner, 0, false));
            }
            self.ask(id, runtime::Request::Attach { owner, group });
        }
        // The round sizes the request's tables and sets the hit it expects, which its answers
        // lower to what they show.
        for ask in self.admission_requests(id) {
            self.ask(id, ask);
        }

        // Other requests are scheduled meanwhile.
        if let Some(request) = self.requests.get_mut(&id) {
            request.status = RequestStatus::WaitingForRemoteKvs;
        }
        self.mark(id);
        true
    }

    /// Every answer of the last round of request `id`'s admission arrived and it lacks positions
    /// its first call and the low watermark need: one more round asks for them, against the hit
    /// the answers reported (SD §3.5, §4.4). False when the need does not fit, which stops the
    /// queue. A request that lacks nothing is admitted.
    fn allocate_admission(&mut self, id: RequestId) -> bool {
        let Some(request) = self.requests.get(&id) else {
            return true;
        };
        if !self.fits(request, request.num_cached_tokens) {
            return false;
        }

        if self.settled(request) {
            self.admitted(id);
            return true;
        }
        for ask in self.missing_requests(id) {
            self.ask(id, ask);
        }
        true
    }

    /// Submits `request`, one of request `id`'s admission: its answer is one more the admission
    /// waits for, unless it raised the count of a held `Allocate` of the same owner and class
    /// (SD §3.1), whose one answer the admission already waits for. The round that made it
    /// counted what it asks in the request's tables.
    pub(super) fn ask(&mut self, id: RequestId, request: runtime::Request) {
        if self.submit(request)
            && let Some(row) = self.requests.get_mut(&id)
        {
            row.pending += 1;
        }
    }

    /// Whether `request`'s need fits what the regions have available now (SD §3.5, §7.2): in each
    /// region, the units its admission takes with a prefix hit of `hit` tokens are at most the
    /// region's `available` less the units the running requests still need to reach their next
    /// call and low watermark.
    fn fits(&self, request: &Request, hit: u32) -> bool {
        let reserve = self.admission_reserve();
        let need = self.units_needed(request, hit);
        need.iter().enumerate().all(|(region, &units)| {
            let available = self.available.get(region).copied().unwrap_or(0);
            let kept = reserve.get(region).copied().unwrap_or(0);
            units == 0 || available.saturating_sub(kept) >= units
        })
    }

    /// Whether `request` can never fit (SD §3.5, §7.2): in some region, the fewest units its
    /// admission can take exceed the region's capacity.
    fn never_fits(&self, request: &Request) -> bool {
        let least = self.units_minimum(request);
        least
            .iter()
            .enumerate()
            .any(|(region, &units)| units > self.capacity.get(region).copied().unwrap_or(0))
    }

    /// The units of `region` (an index) the entries of `request`'s tables of fixed-size blocks
    /// hold there.
    fn units_held(&self, request: &Request, region: usize) -> u32 {
        let mut held = 0u32;
        for (class, blocks) in (0u8..).zip(&request.blocks) {
            let owners = request.attachments.iter().map(|attached| attached.1);
            let there = owners
                .filter_map(|owner| self.class_regions(owner, class))
                .any(|regions| regions.contains(&region));
            if there {
                let each = self.block_units(class, 0).unwrap_or(0);
                let entries = blocks.end.saturating_sub(blocks.first);
                held = held.saturating_add(entries.saturating_mul(each));
            }
        }
        held
    }

    /// The admissions under way: waiting requests whose first round was submitted.
    fn admitting(&self) -> usize {
        self.waiting
            .iter()
            .filter_map(|id| self.requests.get(id))
            .filter(|request| request.status == RequestStatus::WaitingForRemoteKvs)
            .count()
    }

    /// Places request `id` on a group for every phase it runs (SD §3.5 `place`, §7.3 "Routes"),
    /// in the order of its route: a group it is placed on already; else, for a request placed
    /// nowhere yet, the group the policy's `place` names among the candidates when there is a
    /// choice; else the least loaded candidate. A request whose prefill lands on a `Prefill`
    /// group is placed for its decode by its transfer, and never refills. False, with the
    /// request's placement as it was,
    /// when a phase has no candidate or the policy faults. Each group the request is newly placed
    /// on counts it once.
    fn place(&mut self, id: RequestId) -> bool {
        let Some(request) = self.requests.get(&id) else {
            return false;
        };
        let before = request.placement;
        let route = self.route(request);

        for phase in route {
            let Some(request) = self.requests.get(&id) else {
                return false;
            };
            if matches!(phase, Phase::Decode | Phase::Verify | Phase::Refill)
                && self.splits(request)
            {
                continue;
            }
            let candidates = self.candidates(request, phase);
            let nowhere = request.placement.iter().all(Option::is_none);

            let named = if nowhere && candidates.len() > 1 {
                let groups = candidates.iter().filter_map(|group| self.group(*group));
                let replicas = replica_views(request, groups);
                match self.policy.place(&call_view(id, phase), &replicas) {
                    Ok(index) => Some(index),
                    Err(fault) => {
                        self.restore_placement(id, before);
                        self.policy_failed(fault);
                        return false;
                    }
                }
            } else {
                None
            };

            let Some(request) = self.requests.get(&id) else {
                return false;
            };
            let Some(group) = self.choose(request, &candidates, named) else {
                self.restore_placement(id, before);
                return false;
            };
            if let Some(request) = self.requests.get_mut(&id) {
                request.placement[phase.index()] = Some(group);
            }
        }

        // Each group the request is newly placed on counts it once.
        let Some(request) = self.requests.get(&id) else {
            return false;
        };
        let joined: Vec<usize> = placed_groups(request)
            .into_iter()
            .filter(|group| !before.contains(&Some(*group)))
            .filter_map(|group| self.group_index(group))
            .collect();
        for index in joined {
            self.groups[index].join();
        }
        true
    }

    /// The phases `request` runs, in the order they are placed (SD §3.5, §7.3 "Routes"). A video
    /// request: the denoiser first, then the calls that feed it, then the decoders, then the
    /// encoders and the muxer, each phase it makes calls of. A token request: its prefill, then
    /// its decode (a verification where the deployment speculates, and the refill of its draft
    /// cache where a group serves it), then the encodings of its images.
    fn route(&self, request: &Request) -> Vec<Phase> {
        if request.is_video() {
            let rank = |phase: Phase| match phase {
                Phase::Denoise => 0,
                Phase::VideoDecode | Phase::AudioDecode => 2,
                Phase::VideoEncode | Phase::AudioEncode | Phase::Mux => 3,
                _ => 1,
            };
            let mut phases: Vec<Phase> = Phase::ALL
                .into_iter()
                .filter(|phase| request.needed(*phase, &self.groups) > 0)
                .collect();
            phases.sort_by_key(|phase| rank(*phase));
            return phases;
        }

        let serves = |phase: Phase| self.groups.iter().any(|group| group.serves(phase));
        let speculates = serves(Phase::Verify);
        let mut phases = vec![Phase::Prefill];
        phases.push(if speculates {
            Phase::Verify
        } else {
            Phase::Decode
        });
        if speculates && serves(Phase::Refill) {
            phases.push(Phase::Refill);
        }
        for phase in [Phase::VisionEncode, Phase::LatentEncode] {
            if request.needed(phase, &self.groups) > 0 {
                phases.push(phase);
            }
        }
        phases
    }

    /// Whether `request`'s prefill is placed on a `Prefill` group, whose blocks its transfer
    /// moves to a decode group.
    fn splits(&self, request: &Request) -> bool {
        request.placement[Phase::Prefill.index()]
            .and_then(|group| self.group(group))
            .is_some_and(|group| group.config.role == proto::config::WorkerGroupRole::Prefill)
    }

    /// Restores request `id`'s placement to `before`; no group counted it yet.
    fn restore_placement(&mut self, id: RequestId, before: [Option<GroupId>; Phase::COUNT]) {
        if let Some(request) = self.requests.get_mut(&id) {
            request.placement = before;
        }
    }

    /// A new owner: the scheduler's data-parallel rank in its top 16 bits, the counter below, so
    /// that the schedulers of one session never draw the same owner (SD §4.1).
    fn draw_owner(&mut self) -> OwnerId {
        let owner = OwnerId(self.next_owner);
        self.next_owner += 1;
        owner
    }

    /// Rejects waiting request `id` with `kind` and `message`, when it waits and has not started
    /// its admission; any other id is skipped. It holds nothing and leaves at once.
    fn reject_waiting(&mut self, id: RequestId, kind: RejectionKind, message: &str) {
        let waits = self
            .requests
            .get(&id)
            .is_some_and(|request| request.queued && request.status == RequestStatus::Waiting);
        if waits {
            let message = message.to_owned();
            self.finish(id, Event::Rejected { kind, message });
        }
    }
}

/// Preemption (SD §3.6).
impl Scheduler {
    /// Running request `id` is short in `region`, an index into the session's regions as a
    /// group's layout names them (not the number completions name it by): `available` did not
    /// cover what it asks for ahead, or the buffer pool answered its `Allocate` with `Full`, after
    /// which its `asked` is back at `end` (SD §3.6).
    ///
    /// A request that can never fit ends with `CacheCapacity`, once its calls in flight have
    /// returned (`finish_outgrown`). Otherwise the units it lacks
    /// beyond the region's `available`, less the region's retiring units, which come back once
    /// the batches that may read them end, are relieved in this order until covered: draft
    /// tables of idle requests, admissions under way, the policy's victims (a running request
    /// ends as migratable, an unpinned cached block or encoder output is evicted by key); and
    /// when the policy named no request and every running token request is short, the latest of
    /// the lowest priority is preempted. The request then waits for the region's `available`
    /// word to show the units it needs, and asks again; until its blocks arrive it is not
    /// scheduled.
    pub(super) fn preempt(&mut self, id: RequestId, region: u8) {
        let region = usize::from(region);

        // A request that can never fit ends with `FinishReason::CacheCapacity`, or waits for its
        // calls in flight to end first.
        if self.finish_outgrown(id, region) {
            return;
        }

        // Retired units come back once the batches that may read them end: the request waits for
        // the capacity words to show them instead of taking victims for them.
        let available = self.available.get(region).copied().unwrap_or(0);
        let retiring = self.retiring.get(region).copied().unwrap_or(0);
        let need = self
            .units_short(id, region)
            .saturating_sub(available)
            .saturating_sub(retiring);
        if need == 0 {
            return self.wait_for_available(id, region);
        }
        let need = self.give_up_drafts(region, need);
        if need == 0 {
            return self.wait_for_available(id, region);
        }
        let need = self.return_admissions(region, need);
        if need == 0 {
            return self.wait_for_available(id, region);
        }

        let shortage = Shortage {
            kind: Storage::KvUnits,
            need: u64::from(need),
        };
        let victims = match self.policy.preempt(&shortage) {
            Ok(victims) => victims,
            Err(fault) => return self.policy_failed(fault),
        };
        let mut evict = Vec::new();
        let mut named = false;
        for victim in victims {
            match victim {
                Victim::Request(victim) => {
                    let victim = RequestId(victim.0);
                    let ends = self.requests.get(&victim).is_some_and(|request| {
                        request.status == RequestStatus::Running
                            && !request.runs_to_end(&self.groups)
                    });
                    if ends {
                        named = true;
                        self.finish(victim, migratable(MigrationCause::Drain));
                    }
                }
                // A cached block or encoder output no request of this scheduler holds, found by
                // its id in `cache`; an id that names nothing is skipped.
                Victim::Page(PageId(page)) | Victim::Encoding(EncodingId(page)) => {
                    if self.cache_unpinned(page)
                        && let Some(row) = self.cache.get(&page)
                    {
                        evict.push(row.key);
                    }
                }
            }
        }
        if !evict.is_empty() {
            self.submit(runtime::Request::Evict { keys: evict });
        }

        // The policy named no request and every running token request is short.
        if !named {
            self.preempt_last_if_all_short();
        }
        if self.is_running(id) {
            self.wait_for_available(id, region);
        }
    }

    /// Whether running request `id` can never fit in `region`: the units it holds there and the
    /// units its next call still needs there exceed the region's capacity, so that no victim
    /// could let it grow (`S/preemption/tokens.rs:133-164`). It ends with
    /// `FinishReason::CacheCapacity` at the length the cache holds, once no call of it is in
    /// flight: those calls write within the blocks it holds and may end it themselves, and their
    /// outputs are applied (a call in flight of a finished request is dropped, SD §3.7). Until
    /// then it waits, and takes no victim. True when it can never fit.
    fn finish_outgrown(&mut self, id: RequestId, region: usize) -> bool {
        let Some(request) = self.requests.get(&id) else {
            return false;
        };
        let held = self.units_held(request, region);
        let need = self.units_short(id, region);
        let capacity = self.capacity.get(region).copied().unwrap_or(0);
        if held.saturating_add(need) <= capacity {
            return false;
        }

        if request.idle() {
            let terminal = self.finished(id, FinishReason::CacheCapacity, None);
            self.finish(id, terminal);
        } else {
            self.wait_for_available(id, region);
        }
        true
    }

    /// The units of `region` (an index) request `id`'s next call needs beyond the blocks it asked
    /// for: the blocks that call writes (SD §3.6, §4.4), neither those of the low watermark nor
    /// those of its draft cache, which take only the units the regions have free.
    fn units_short(&self, id: RequestId, region: usize) -> u32 {
        let Some(request) = self.requests.get(&id) else {
            return 0;
        };
        let Some(owner) = self.kv_owner(request) else {
            return 0;
        };
        let mut short = 0u32;
        for (class, _, _, needed) in self.ahead(request) {
            let there = self
                .class_regions(owner, class)
                .is_some_and(|regions| regions.contains(&region));
            if there {
                let each = self.block_units(class, 0).unwrap_or(0);
                short = short.saturating_add(needed.saturating_mul(each));
            }
        }
        short
    }

    /// Request `id` waits for `region`'s `available` word to show the units its shortage needs;
    /// it asks again once it does (`allocate_ahead`) and is not scheduled until its blocks
    /// arrive.
    fn wait_for_available(&mut self, id: RequestId, region: usize) {
        let units = self.units_short(id, region);
        self.short.insert(id, (region, units));
    }

    /// Gives up draft tables until their units in `region` cover `need`, and returns what is left
    /// of it (SD §3.6, §7.2 "Draft release"; `S/speculation.rs:474-525`). Draft blocks are outside
    /// admission: the idle running requests with the fewest tokens left to generate go first
    /// (`draft_order`), each giving up its whole draft table by one `Release` through the
    /// executors and starting its draft cache again from nothing (`give_up_draft`).
    fn give_up_drafts(&mut self, region: usize, need: u32) -> u32 {
        let mut need = need;
        for id in self.draft_order(region) {
            if need == 0 {
                break;
            }
            let freed = self.give_up_draft(id);
            need = need.saturating_sub(freed.get(region).copied().unwrap_or(0));
        }
        need
    }

    /// Returns admissions under way to the waiting queue until the units their tables hold in
    /// `region` cover `need`, and returns what is left of it (SD §3.6): a running request goes
    /// before a request still admitting. The latest in the queue goes first, as it would be
    /// admitted last.
    fn return_admissions(&mut self, region: usize, need: u32) -> u32 {
        let admitting: Vec<RequestId> = self
            .waiting
            .iter()
            .rev()
            .copied()
            .filter(|id| {
                self.requests.get(id).is_some_and(|request| {
                    request.queued && request.status == RequestStatus::WaitingForRemoteKvs
                })
            })
            .collect();

        let mut need = need;
        for id in admitting {
            if need == 0 {
                break;
            }
            let Some(request) = self.requests.get(&id) else {
                continue;
            };
            let units = self.units_held(request, region);
            if units == 0 {
                continue;
            }
            self.requeue(id);
            need = need.saturating_sub(units);
        }
        need
    }

    /// When at least two running token requests live, every one of them is short, nothing is in
    /// flight on any group and none is leaving, the one of the lowest priority admitted last ends
    /// as migratable: no request would end and let units go, and the policy named none
    /// (`S/preemption/tokens.rs:181-226`). A request running alone gains nothing from giving up
    /// its own units.
    fn preempt_last_if_all_short(&mut self) {
        if self.groups.iter().any(|group| group.in_flight > 0) {
            return;
        }
        let tokens: Vec<&Request> = self
            .running
            .iter()
            .filter_map(|id| self.requests.get(id))
            .filter(|request| !request.is_video())
            .collect();
        let leaving = self
            .requests
            .values()
            .any(|request| request.status == RequestStatus::Finished);
        let short = |request: &&Request| self.ahead(request).iter().any(|&(.., needed)| needed > 0);
        if tokens.len() < 2 || leaving || !tokens.iter().all(short) {
            return;
        }

        let victim = tokens
            .iter()
            .max_by_key(|request| (request.input.priority(), request.arrival))
            .map(|request| request.id);
        if let Some(victim) = victim {
            self.finish(victim, migratable(MigrationCause::Drain));
        }
    }
}

/// The groups `request` is placed on, each once, in the order of its phases.
fn placed_groups(request: &Request) -> Vec<GroupId> {
    let mut groups = Vec::new();
    for group in request.placement.iter().flatten() {
        if !groups.contains(group) {
            groups.push(*group);
        }
    }
    groups
}

/// The phase a policy's work names by its id; `None` for an id no phase carries.
fn phase_of(id: PhaseId) -> Option<Phase> {
    Phase::ALL.into_iter().find(|phase| phase.id() == id)
}

/// Whether `phase` is a token phase, whose calls write the request's cache.
fn is_token(phase: Phase) -> bool {
    matches!(
        phase,
        Phase::Prefill | Phase::Decode | Phase::Verify | Phase::Refill
    )
}

/// Whether `group` takes another batch: fewer of its batches are in flight than its depth.
fn queue_open(group: &WorkerGroup) -> bool {
    (group.in_flight as usize) < group.config.queue_depth
}

/// Returns a call's tokens and the call to the step's budgets.
fn refund(budget: &mut Budget, tokens: u32) {
    budget.0 = budget.0.saturating_add(tokens);
    budget.1 = budget.1.saturating_add(1);
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use ipc::CacheGroupKind;
    use ipc::batch::Phase;
    use proto::config::{KvCacheConfig, WorkerGroupRole};
    use proto::request::{GenerationFeatures, GenerationLimits, GenerationRequest, RequestId};
    use scheduling_guest::Event as PolicyEvent;
    use scheduling_guest::ids::PageId;
    use scheduling_guest::limits::{Limits, Parameters};
    use scheduling_guest::view::Snapshot;
    use storage::runtime::{self, Completion, OwnerId, Tier};

    use super::super::policy::tests::packaged;
    use super::super::tests::{A, Bench, KV, OWNER_A, bench, key};
    use super::super::{Blocks, Error, Output, Policy, Request, RequestStatus};

    /// A token request numbered `id` of `prompt` that may generate `max` tokens.
    fn tokens(id: u64, prompt: &[u32], max: usize) -> GenerationRequest {
        GenerationRequest {
            request_id: RequestId(id),
            prompt_token_ids: prompt.to_vec(),
            multimodal_inputs: Default::default(),
            sampling: Default::default(),
            max_und_tokens: max,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            include_stop_token: false,
            generated_token_ids: Vec::new(),
            adapter: 0,
        }
    }

    /// Group `index` of the bench runs `phases` under `role`.
    fn serve(bench: &mut Bench, index: usize, role: WorkerGroupRole, phases: &[Phase]) {
        let group = &mut bench.scheduler.groups[index];
        group.config.role = role;
        group.capacity.phases = phases.iter().map(|phase| 1 << phase.index()).sum();
    }

    /// Request 1 of the bench, attached to group A, running with its prefill placed there.
    fn running(bench: &mut Bench) -> RequestId {
        bench.attach(&[(A, OWNER_A, 3)]);
        let id = RequestId(1);
        let request = bench.scheduler.requests.get_mut(&id).unwrap();
        request.status = RequestStatus::Running;
        request.queued = false;
        request.placement[Phase::Prefill.index()] = Some(A);
        bench.scheduler.running.push(id);
        id
    }

    /// A prefill chunk is what the policy asked, cut to the chunk limit, the prompt left and the
    /// step's tokens; a decode costs one token and a media call none; every call costs one of the
    /// step's calls, and nothing fits once they or the tokens are spent.
    #[test]
    fn a_call_takes_its_tokens_and_one_call_from_the_step() {
        let mut bench = bench(1024);
        let scheduler = &mut bench.scheduler;
        scheduler.policy.limits.chunked_prefill_size = 8;
        let mut request = Request::new(tokens(1, &[1; 20], 4).into(), 1, Instant::now());

        let mut budget = (100, 3);
        let charge = |request: &Request, phase, asked, budget: &mut (u32, usize)| {
            scheduler.charge(request, phase, asked, budget)
        };
        assert_eq!(charge(&request, Phase::Prefill, 0, &mut budget), Some(8));
        assert_eq!(charge(&request, Phase::Prefill, 5, &mut budget), Some(5));
        assert_eq!(charge(&request, Phase::Prefill, 50, &mut budget), Some(8));
        assert_eq!(budget, (79, 0));
        assert_eq!(
            charge(&request, Phase::Decode, 0, &mut budget),
            None,
            "no call left"
        );

        let mut budget = (6, 4);
        assert_eq!(charge(&request, Phase::Prefill, 0, &mut budget), Some(6));
        request.num_computed_tokens = 17;
        let mut budget = (100, 4);
        assert_eq!(
            charge(&request, Phase::Prefill, 0, &mut budget),
            Some(3),
            "the prompt left"
        );
        assert_eq!(charge(&request, Phase::Decode, 0, &mut budget), Some(1));
        assert_eq!(charge(&request, Phase::Mux, 0, &mut budget), Some(0));
        assert_eq!(budget, (96, 1));

        let mut budget = (0, 4);
        assert_eq!(charge(&request, Phase::Prefill, 0, &mut budget), None);
        assert_eq!(
            charge(&request, Phase::Mux, 0, &mut budget),
            Some(0),
            "a budget of 0"
        );
    }

    /// A resumed request waits behind the resumed requests and ahead of every new one; the
    /// policy is told of each arrival in order. Beyond the policy's queue bound a request is
    /// refused and kept nowhere.
    #[test]
    fn a_resumed_request_waits_ahead_of_new_requests() {
        let mut bench = bench(1024);
        let (output, outputs) = crossbeam_channel::unbounded();
        let scheduler = &mut bench.scheduler;
        scheduler.output = output;
        scheduler.groups[0].capacity.kv_cache = Some(KvCacheConfig::default());
        scheduler.limits = GenerationLimits {
            features: GenerationFeatures::all(),
            ..GenerationLimits::default()
        };

        for (id, resumed) in [(1, false), (2, true), (3, false), (4, true)] {
            let mut input = tokens(id, &[1, 2, 3], 8);
            if resumed {
                input.generated_token_ids = vec![7];
            }
            scheduler.add_request(input.into());
        }
        let waiting: Vec<u64> = scheduler.waiting.iter().map(|id| id.0).collect();
        assert_eq!(waiting, [2, 4, 1, 3]);
        let arrived: Vec<u64> = scheduler
            .policy
            .events
            .iter()
            .filter_map(|event| match event {
                PolicyEvent::Arrived { request } => Some(request.id.0),
                _ => None,
            })
            .collect();
        assert_eq!(arrived, [1, 2, 3, 4]);

        scheduler.policy.limits.max_queued_requests = Some(4);
        scheduler.add_request(tokens(5, &[1], 8).into());
        assert_eq!(scheduler.waiting.len(), 4);
        assert!(!scheduler.requests.contains_key(&RequestId(5)));
        let refused = outputs.try_iter().any(|output| {
            matches!(
                output,
                Output::Event(RequestId(5), proto::request::Event::Rejected { message, .. })
                    if message == "scheduler waiting queue is full"
            )
        });
        assert!(refused);
    }

    /// A token request is placed for its prefill and its decode on one group of the `Model`
    /// role, counted there once; on a `Prefill` group only for its prefill, its transfer choosing
    /// its decode group; with no group for a phase it is placed nowhere.
    #[test]
    fn a_token_request_is_placed_by_the_roles_of_the_groups() {
        let mut bench = bench(1024);
        serve(
            &mut bench,
            0,
            WorkerGroupRole::Model,
            &[Phase::Prefill, Phase::Decode],
        );
        let scheduler = &mut bench.scheduler;
        let place = |scheduler: &mut super::Scheduler, id: u64| {
            let request = Request::new(tokens(id, &[1, 2], 4).into(), 1, Instant::now());
            scheduler.requests.insert(RequestId(id), request);
            let placed = scheduler.place(RequestId(id));
            (placed, scheduler.requests[&RequestId(id)].placement)
        };

        let (placed, placement) = place(scheduler, 1);
        assert!(placed);
        assert_eq!(placement[Phase::Prefill.index()], Some(A));
        assert_eq!(placement[Phase::Decode.index()], Some(A));
        assert_eq!(scheduler.groups[0].num_requests, 1);

        serve(&mut bench, 0, WorkerGroupRole::Prefill, &[Phase::Prefill]);
        serve(&mut bench, 1, WorkerGroupRole::Decode, &[Phase::Decode]);
        let scheduler = &mut bench.scheduler;
        let (placed, placement) = place(scheduler, 2);
        assert!(placed);
        assert_eq!(placement[Phase::Prefill.index()], Some(A));
        assert_eq!(placement[Phase::Decode.index()], None);
        assert_eq!(scheduler.groups[1].num_requests, 0);

        serve(&mut bench, 0, WorkerGroupRole::Decode, &[Phase::Decode]);
        let scheduler = &mut bench.scheduler;
        let (placed, placement) = place(scheduler, 3);
        assert!(!placed);
        assert_eq!(placement, [None; Phase::COUNT]);
        assert_eq!(scheduler.groups[0].num_requests, 2);
    }

    /// Under a shortage, the idle requests with the fewest tokens left give up their whole draft
    /// tables first, each by one `Release` the executors of its entries stamp, until the units
    /// cover it; a request with a call in flight keeps its table.
    #[test]
    fn draft_tables_are_given_up_fewest_tokens_left_first() {
        let mut bench = bench(1024);
        bench.scheduler.groups[0].capacity.cache_groups = vec![(CacheGroupKind::Draft, 16)];
        let first = running(&mut bench);
        let scheduler = &mut bench.scheduler;
        let table = |end| Blocks {
            first: 0,
            shared: 0,
            end,
            asked: end,
        };

        // Request 2 has 2 tokens left to generate, request 1 has 4.
        let mut second = scheduler.requests[&first].clone();
        let owner = OwnerId(12);
        second.id = RequestId(2);
        second.attachments = vec![(A, owner, 5, false)];
        if let proto::request::Request::Generation(input) = &mut second.input {
            input.request_id = RequestId(2);
            input.max_und_tokens = 2;
        }
        second.blocks[usize::from(KV)] = table(2);
        second.num_draft_tokens = 32;
        scheduler.owners.insert(owner, second.id);
        scheduler.requests.insert(second.id, second);
        scheduler.running.push(RequestId(2));
        scheduler.requests.get_mut(&first).unwrap().blocks[usize::from(KV)] = table(3);

        assert_eq!(scheduler.give_up_drafts(0, 2), 0);
        let release = |owner, count| runtime::Request::Release {
            owner,
            class: KV,
            at: 0,
            count,
            after: Vec::new(),
        };
        let stamps: Vec<_> = bench
            .stamps()
            .into_iter()
            .filter(|(group, _)| *group == A)
            .collect();
        assert_eq!(stamps, [(A, release(owner, 2))]);
        let scheduler = &mut bench.scheduler;
        assert_eq!(scheduler.requests[&RequestId(2)].num_draft_tokens, 0);
        assert_eq!(
            scheduler.requests[&RequestId(2)].blocks[0],
            Blocks::default()
        );

        // Request 1 has a call in flight: it keeps its table, and the shortage stays.
        scheduler.requests.get_mut(&first).unwrap().scheduled[Phase::Decode.index()] = 1;
        assert_eq!(scheduler.give_up_drafts(0, 2), 2);
        scheduler.requests.get_mut(&first).unwrap().completed[Phase::Decode.index()] = 1;
        assert_eq!(scheduler.give_up_drafts(0, 5), 2);
        let stamps: Vec<_> = bench
            .stamps()
            .into_iter()
            .filter(|(group, _)| *group == A)
            .collect();
        assert_eq!(stamps, [(A, release(OWNER_A, 3))]);
    }

    /// The low watermark and the draft cache take only the units the regions have free, and the
    /// units a short request waits for are not: a request visited first that would only grow its
    /// headroom leaves them, and the short request asks for the block its next call writes (SD
    /// §3.6, §4.4).
    #[test]
    fn headroom_leaves_the_units_a_short_request_waits_for() {
        let mut bench = bench(1024);
        bench.scheduler.groups[0].capacity.cache_groups = vec![(CacheGroupKind::Full, 4)];
        let first = running(&mut bench);
        let scheduler = &mut bench.scheduler;
        let table = |end| Blocks {
            first: 0,
            shared: 0,
            end,
            asked: end,
        };

        // Request 1 computed its prompt and holds the block its next decode writes: it would ask
        // for its low watermark's block alone.
        let request = scheduler.requests.get_mut(&first).unwrap();
        request.num_computed_tokens = 3;
        request.blocks[usize::from(KV)] = table(1);

        // Request 2 generated a token: its next decode writes a block it has not asked for, and
        // it waits for that block's unit, which the region now has.
        let mut second = scheduler.requests[&first].clone();
        let owner = OwnerId(12);
        second.id = RequestId(2);
        second.attachments = vec![(A, owner, 5, false)];
        if let proto::request::Request::Generation(input) = &mut second.input {
            input.request_id = RequestId(2);
        }
        second.token_ids.push(7);
        second.num_computed_tokens = 4;
        scheduler.owners.insert(owner, second.id);
        scheduler.requests.insert(second.id, second);
        scheduler.running.push(RequestId(2));
        scheduler.short.insert(RequestId(2), (0, 1));
        scheduler.available[0] = 1;

        scheduler.allocate_ahead();
        scheduler.post().unwrap();
        let written: Vec<runtime::Request> = bench
            .take()
            .into_iter()
            .map(|(_, request)| request)
            .collect();
        let asked = runtime::Request::Allocate {
            owner,
            class: KV,
            at: 1,
            count: 1,
            units: 0,
        };
        assert_eq!(written, [asked]);
    }

    /// A step that forms no batch for a group whose executor holds a copy not yet finished sends
    /// it an empty batch of a phase it serves, once its queue has nothing in flight; once the
    /// executor returned the copy's `Finish`, it sends none (SD §2.1).
    #[test]
    fn a_group_with_a_copy_to_run_is_sent_an_empty_batch() {
        let (mut bench, copy) = copying();

        let batches = bench.scheduler.schedule();
        assert_eq!(batches.len(), 1);
        assert_eq!((batches[0].group, batches[0].phase), (A, Phase::Prefill));
        assert!(batches[0].requests.is_empty());

        bench.scheduler.groups[0].in_flight = 1;
        assert!(
            bench.scheduler.schedule().is_empty(),
            "a batch is in flight"
        );
        bench.scheduler.groups[0].in_flight = 0;

        bench.scheduler.stamped(vec![runtime::Request::Finish {
            copy,
            ok: true,
            after: vec![(A, 1)],
        }]);
        assert!(
            bench.scheduler.schedule().is_empty(),
            "the copy is finished"
        );
    }

    /// A copy that has not finished `prefetch_wait` of the tokens it moves after its `Copy`
    /// arrived is given up through its group's executor with `Finish { ok: false }`, and its group
    /// is sent no more empty batches for it (SD §4.4 "The bound of a copy"). A copy of no tokens
    /// is given a second.
    #[test]
    fn a_copy_past_its_bound_is_given_up_through_its_group() {
        let (mut bench, copy) = copying();
        let arrived = Instant::now();

        bench.scheduler.give_up_late_copies(arrived);
        assert!(bench.stamps().is_empty(), "the copy is within its bound");

        bench
            .scheduler
            .give_up_late_copies(arrived + Duration::from_millis(1001));
        let given_up = runtime::Request::Finish {
            copy,
            ok: false,
            after: Vec::new(),
        };
        assert_eq!(bench.stamps(), [(A, given_up)]);
        assert!(bench.scheduler.schedule().is_empty());
    }

    /// Request 1's `Load` of one block on group A, answered `Copy`, which the scheduler handed to
    /// group A's executor; groups A and B serve prefills and decodes under `fcfs`. Returns the
    /// number of the `Load`.
    fn copying() -> (Bench, runtime::RequestId) {
        let mut bench = bench(4096);
        bench.scheduler.policy = Policy::new(
            packaged("fcfs"),
            Parameters::default(),
            Snapshot::default(),
            &Limits::default(),
        )
        .unwrap();
        for index in [0, 1] {
            let phases = [Phase::Prefill, Phase::Decode];
            serve(&mut bench, index, WorkerGroupRole::Model, &phases);
        }
        bench.attach(&[(A, OWNER_A, 3)]);
        bench.scheduler.submit(runtime::Request::Load {
            owner: OWNER_A,
            class: KV,
            at: 0,
            keys: vec![key(1)],
        });
        bench.scheduler.post().unwrap();
        let [(load, _)] = bench.take()[..] else {
            panic!("the `Load` is written");
        };

        // No copy is held before its `Copy` arrives.
        assert!(bench.scheduler.schedule().is_empty());
        let copied = Completion::Copy {
            group: A,
            keys: vec![key(1)],
            from: vec![(5, 1, 1)],
            to: vec![(4, 1, 1)],
            evicted: Vec::new(),
            encodings: None,
        };
        bench.complete(load, &copied);
        assert_eq!(bench.answers().len(), 1);
        (bench, load)
    }

    /// A running request that made no progress within the policy's timeout fails the scheduler
    /// when nothing else runs or waits: nothing was formed for anything running.
    #[test]
    fn a_starved_request_alone_fails_the_scheduler() {
        let mut bench = bench(1024);
        let (output, outputs) = crossbeam_channel::unbounded();
        bench.scheduler.output = output;
        let id = running(&mut bench);
        let scheduler = &mut bench.scheduler;
        let long_ago = Instant::now().checked_sub(Duration::from_secs(31)).unwrap();
        scheduler.requests.get_mut(&id).unwrap().last_progress = long_ago;

        assert!(scheduler.schedule().is_empty());
        let failed = outputs.try_iter().find_map(|output| match output {
            Output::Failed(Error::Executor(message)) => Some(message),
            _ => None,
        });
        assert_eq!(
            failed.as_deref(),
            Some("no batch formed for 1 running request(s) at step 0")
        );
    }

    /// A block the policy's `offload` names is offered to the host once: named again while its
    /// `Store` is held or outstanding, before `cache` shows it on the host, it is not stored
    /// again (SD §3.4 "The policy's `offload`", §4.4 "The write policy").
    #[test]
    fn a_block_the_policy_offloads_is_stored_once_until_its_store_is_answered() {
        let mut bench = bench(4096);
        // Group A's blocks of class `KV` have a host copy region; one is on the device in A.
        bench.scheduler.classes[usize::from(KV)].host = Some((2, 1));
        let block = super::super::Cached {
            key: key(1),
            class: KV,
            tokens: 16,
            tiers: super::super::cache::DEVICE,
            groups: 1 << A.0,
            pins: 1,
            last_used: 1,
        };
        let page = PageId(super::super::Cached::id(&block.key));
        bench.scheduler.cache.insert(page.0, block);

        let store = runtime::Request::Store {
            group: A,
            keys: vec![key(1)],
            tier: Tier::Host,
        };
        let written = |bench: &mut Bench| -> Vec<runtime::Request> {
            bench.scheduler.post().unwrap();
            bench
                .take()
                .into_iter()
                .map(|(_, request)| request)
                .collect()
        };
        bench.scheduler.offload(&[page]);
        bench.scheduler.offload(&[page]);
        assert_eq!(written(&mut bench), [store]);

        // Outstanding, it is not offered again either.
        bench.scheduler.offload(&[page]);
        assert!(written(&mut bench).is_empty());
    }
}
