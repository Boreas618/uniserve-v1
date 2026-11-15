//! The policy: what it decides, from its arguments and what the mechanism tells it.

use std::collections::{BTreeMap, BTreeSet};

use scheduling::{
    CallView, DispatchOutput, EncodingId, Event, Limits, PageId, PageView, Parameters, Phase,
    PhaseClass, Policy, RankView, ReplicaView, RequestClass, RequestId, RequestStatus, RequestView,
    Room, Shortage, Snapshot, Storage, Unstated, Victim, stated,
};

use crate::budget::{Shares, Step, prefill_chunk, work_tokens};
use crate::lru::Lru;
use crate::phases::Phases;
use crate::waiting::Waiting;

/// Whether a page is one a step offloads: a device page no running request
/// continues from.
fn offloadable(view: &PageView) -> bool {
    !view.on_host && view.evictable
}

/// What a draft-cache refill costs and saves, as the deployment states it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Refill {
    /// The speedup speculation gives a request (`expected_speedup`).
    speedup: f64,
    /// The cost of one refilled token, in decode steps (`refill_token_cost`).
    token_cost: f64,
}

/// The policy: it admits in its queue's order and schedules by phase class and admission.
///
/// | decision | rule |
/// | --- | --- |
/// | `admit` | the waiting queue in its order; the mechanism admits until the first request it cannot place |
/// | `work` | requests in `(path, phase class, admission order)`, at most one call each, charged against the step's tokens and calls; a prefill chunk is `prefill_chunk` of the prompt left; a context-class call runs only in a prefill step, and a step is a prefill step while fewer than `prefill_window` batches with prefill work are in flight |
/// | `reject` | a waiting request whose worst-case reservation is more than the device cache's whole capacity |
/// | `offload` | each device page no running request continues from, once: in the step after it became one, in the order it did (least recently used first) |
/// | `prefetch` | the requests whose host hit deepens by at least `Limits::prefetch_min_tokens` |
/// | `refill` | the requests whose draft cache has a gap, when the refill is worth its cost (`Arrival::refill_due`) |
/// | `preempt` | the least recently used page of the short storage, then the lowest-priority last-admitted request |
/// | `route` | the recorded cache rank, else the live rank with the fewest requests in flight, from a rotating cursor |
/// | `place` | the replica that already holds the request; else the one with the fewest running requests among those with capacity; a decode-stage call never lands on a prefill-only replica |
#[derive(Clone, Debug)]
pub struct Arrival {
    /// The limits the arguments declare.
    limits: Limits,
    /// Batches carrying prefill work that may be in flight before token progress takes the step.
    prefill_window: u32,
    /// What a refill costs and saves; `None` unless the deployment states both, and every
    /// refill then runs.
    refill: Option<Refill>,
    /// The epoch's phases, so a request's next phase can be classified.
    phases: Phases,
    /// Requests waiting for admission, in the order the arguments name.
    waiting: Waiting,
    /// The latest view of every request the mechanism has told this policy
    /// about: queued, running, and those that ended.
    views: BTreeMap<RequestId, RequestView>,
    /// The running requests, in the order the mechanism admitted them.
    admission: Vec<RequestId>,
    /// The pages of the device and host caches, least recently used first,
    /// each with its view as the mechanism last stated it: whether it sits in
    /// host memory and whether no running request continues from it.
    pages: Lru<PageId, PageView>,
    /// The pages that became device pages no request holds since the last
    /// `dispatch`, in the order they did: what the next step offloads, each
    /// once. A page the snapshot states is not one, since the instance before
    /// this one had its own step to name it.
    offers: Vec<PageId>,
    /// The encoder cache entries, least recently used first.
    encodings: Lru<EncodingId>,
    /// The rank the last `route` picked, so that ranks of equal load take new requests in turn.
    route_cursor: usize,
    /// Whether the next step puts the media path's work first: the media path and the token path
    /// alternate their precedence from step to step.
    media_first: bool,
}

impl TryFrom<Parameters> for Arrival {
    type Error = Unstated;

    /// The package's arguments construct the policy (design §5.1: the arguments
    /// are part of the package); the state arrives in `init`.
    fn try_from(parameters: Parameters) -> Result<Self, Unstated> {
        let limits = Limits {
            chunked_prefill_size: stated(parameters.chunked_prefill_size, "chunked_prefill_size")?,
            max_queued_requests: parameters.max_queued_requests,
            max_batch: stated(parameters.max_batch, "max_batch")?,
            max_num_batched_tokens: stated(
                parameters.max_num_batched_tokens,
                "max_num_batched_tokens",
            )?,
            prefetch_min_tokens: stated(parameters.prefetch_min_tokens, "prefetch_min_tokens")?,
            timeout: stated(parameters.timeout, "timeout")?,
        };
        let refill = match (parameters.expected_speedup, parameters.refill_token_cost) {
            (Some(speedup), Some(token_cost)) => Some(Refill {
                speedup,
                token_cost,
            }),
            _ => None,
        };
        Ok(Self {
            limits,
            prefill_window: stated(parameters.prefill_window, "prefill_window")?,
            refill,
            phases: Phases::default(),
            waiting: Waiting::new(stated(parameters.order, "order")?),
            views: BTreeMap::new(),
            admission: Vec::new(),
            pages: Lru::new(),
            offers: Vec::new(),
            encodings: Lru::new(),
            route_cursor: 0,
            media_first: false,
        })
    }
}

impl Arrival {
    /// The phase kind of a request's next call, if it has one ready.
    fn next_kind(&self, view: &RequestView) -> Option<Phase> {
        view.phase.and_then(|phase| self.phases.kind(phase))
    }

    /// Whether a running request contributes work of `class` in this step.
    fn has_work_of_class(&self, class: PhaseClass) -> bool {
        self.views.values().any(|view| {
            view.state == RequestStatus::Running
                && self
                    .next_kind(view)
                    .is_some_and(|kind| kind.class() == class)
        })
    }

    /// The pass of this step: a prefill step (context work only) while prefill work is ready and
    /// fewer than `prefill_window` batches with prefill work are in flight; otherwise a token
    /// step (everything but context work) while token progress is ready, a prefill step while
    /// only prefill work is, and no filter while only media or background work is.
    fn pass(&self, room: &Room) -> Pass {
        // A refill is a prefill-class chunk: a step whose only prefill work is a refill's is a
        // prefill step.
        let prefill_ready = self.has_work_of_class(PhaseClass::Context)
            || self
                .views
                .values()
                .any(|view| view.state == RequestStatus::Running && self.refill_due(view));
        let window_open = room.inflight.prefill_batches < self.prefill_window;
        if prefill_ready && window_open {
            return Pass::Prefill;
        }
        if self.has_work_of_class(PhaseClass::Token) {
            return Pass::Token;
        }
        if prefill_ready {
            return Pass::Prefill;
        }
        Pass::Any
    }

    /// The running requests with a call ready, in this step's assembly order: the preferred path
    /// first, then the phase class, then admission order. A running request this policy saw no
    /// admission of sorts after those it did, by id.
    ///
    /// Each request's key is computed once, its admission position read from one
    /// index of the admission order, so a step of `n` running requests sorts in
    /// `n log n` rather than searching the admission order at every comparison.
    fn assembly_order(&self) -> Vec<RequestId> {
        let admitted: BTreeMap<RequestId, usize> = self
            .admission
            .iter()
            .enumerate()
            .map(|(at, id)| (*id, at))
            .collect();
        let mut order: Vec<((u8, PhaseClass, usize, u64), RequestId)> = self
            .views
            .values()
            .filter(|view| view.state == RequestStatus::Running)
            .filter_map(|view| {
                let kind = self.next_kind(view)?;
                let media = view.class == RequestClass::Media;
                // The path the policy did not prefer comes second.
                let path = u8::from(media != self.media_first);
                let at = admitted.get(&view.id).copied().unwrap_or(usize::MAX);
                Some(((path, kind.class(), at, view.id.0), view.id))
            })
            .collect();
        // The keys are distinct, since each ends with the request's id.
        order.sort_unstable_by_key(|(key, _)| *key);
        order.into_iter().map(|(_, id)| id).collect()
    }

    /// The token work of one call of `kind`, and whether it writes the prefix cache.
    ///
    /// A prefill or refill chunk is sized by [`prefill_chunk`] from the
    /// request's prompt progress; every other phase states the charge the
    /// mechanism states for the request's next call ([`work_tokens`]).
    fn work_tokens(&self, view: &RequestView, kind: Phase, room: &Room, shares: &Shares) -> u32 {
        if !kind.writes_prefix_cache() {
            return work_tokens(kind, 0, view.next_tokens);
        }
        // The view already states the further of the two cursors: the prompt
        // tokens the runtime computed, or the prefix the cache held when the
        // request was admitted. Adding them counts the cached prefix twice,
        // which pushes the start past the end of a prompt that reused one — the
        // chunk is then zero for ever and the request is never named.
        let start = view.prefilled_tokens;
        let share = shares.tokens().min(shares.prefill());
        let chunk = prefill_chunk(
            start,
            view.prefill_tokens.max(view.prompt_tokens),
            self.limits.chunked_prefill_size,
            share,
            room.page_tokens,
        );
        if chunk == 0 {
            // Nothing left to ingest: this phase's write is structural (a close
            // cache, a block's commit), so its charge is the
            // one the mechanism states rather than a chunk to size. Without
            // this the request is never named, the pass stays non-token, and
            // the decoders behind it wait for ever.
            return view.next_tokens;
        }
        chunk
    }

    /// Whether a request's missing draft cache is refilled now.
    ///
    /// The mechanism states the cost of the gap (`refill_tokens`). With a stated speedup `g` and
    /// token cost `k`, a refill of `t` tokens of a request that may still generate `remaining`
    /// tokens is worth it when `remaining * (1 - 1/g) > t * k`; with neither stated the refill
    /// runs. The deployment's statement stands in for the mechanism's own measurement of the
    /// speedup, which the views do not carry (design §7.7).
    fn refill_due(&self, view: &RequestView) -> bool {
        // A request with a call in flight is not a candidate. A refill is prefill work of the
        // step that names it, and `output.refill` is the one decision that names it; the
        // request's phase stays the one it will serve next, so no phase kind is asked for here.
        if view.phase.is_none() {
            return false;
        }
        let Some(tokens) = view.refill_tokens else {
            return false;
        };
        if tokens == 0 {
            return false;
        }
        let Some(refill) = self.refill else {
            return true;
        };
        let saved = f64::from(view.remaining_tokens) * (1.0 - 1.0 / refill.speedup.max(1.0));
        saved > f64::from(tokens) * refill.token_cost.max(0.0)
    }

    /// Whether a request's host prefix hit deepens by enough to prefetch
    /// (`Limits::prefetch_min_tokens`): a shorter hit is computed faster than it is fetched.
    fn prefetch_due(&self, view: &RequestView) -> bool {
        view.prefetch_tokens >= self.limits.prefetch_min_tokens
    }

    /// The device pages a step offloads to the host KV cache: each page no running request
    /// continues from, once. A page is named in the step after it became one,
    /// in the order the pages did (least recently used first); one that a
    /// request holds again, that went to the host alone or that left the
    /// cache meanwhile is not, and one that became one twice is named once.
    /// A page named before is not named again: the mechanism offered it, so
    /// a warm cache costs a step nothing.
    fn offload_pages(&mut self) -> Vec<PageId> {
        let mut named = BTreeSet::new();
        let offers = std::mem::take(&mut self.offers);
        offers
            .into_iter()
            .filter(|page| self.pages.get(*page).is_some_and(offloadable))
            .filter(|page| named.insert(*page))
            .collect()
    }

    /// The view of `page`, which an event of kind `event` names as used just now: it moves to the
    /// most recent end of the order.
    ///
    /// # Panics
    ///
    /// When the snapshot and the events before stated no such page. The interface states a page
    /// to the policy before any `Hit`, `Released` or `Held` names it (`scheduling::Event`): the
    /// mechanism names in those events only rows that were in its cache before the change, and a
    /// row that enters it is a `Cached`. A page outside that contract is a defect of the
    /// mechanism or of this policy, so the policy fails (a trap, which the mechanism handles as
    /// it handles every fault of a policy) instead of inventing a view the mechanism never
    /// stated.
    fn used(&mut self, page: PageId, event: &str) -> &mut PageView {
        self.pages
            .touch(page)
            .unwrap_or_else(|| panic!("the mechanism {event} page {} it never stated", page.0))
    }

    /// A page the mechanism states as it stands now (`Cached`): it is the most recently used,
    /// and it is offered to the host once it becomes a device page no request holds.
    fn state(&mut self, page: &PageView) {
        let was = self.pages.get(page.id).is_some_and(offloadable);
        if offloadable(page) && !was {
            self.offers.push(page.id);
        }
        self.pages.insert(page.id, page.clone());
    }

    /// The pages a shortage of `kind` may take, least recently used first.
    ///
    /// A device page a running request still continues from frees no device
    /// units the mechanism may take, and the mechanism refuses a victim it cannot free - so
    /// naming one would stop the epoch at the first shortage instead of freeing what is free. A
    /// host page is the host cache's to take: the mechanism asks only that it is on the host.
    fn page_victims(&self, want_host: bool, need: u64) -> Vec<Victim> {
        let mut victims = Vec::new();
        let mut taken = 0u64;
        for (page, view) in self.pages.iter() {
            if taken >= need {
                break;
            }
            if view.on_host != want_host {
                continue;
            }
            if !want_host && !view.evictable {
                continue;
            }
            victims.push(Victim::Page(page));
            taken += 1;
        }
        victims
    }

    /// The request a shortage takes when no page can relieve it: the lowest
    /// priority, then the latest admission, of the requests that resume from
    /// their generated tokens.
    fn request_victim(&self) -> Option<Victim> {
        self.views
            .values()
            .filter(|view| view.state == RequestStatus::Running && view.migratable)
            .max_by_key(|view| {
                (
                    view.priority,
                    view.admitted_at.map(|at| at.0).unwrap_or_default(),
                    view.id.0,
                )
            })
            .map(|view| Victim::Request(view.id))
    }
}

/// Which work a step assembles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    /// Context ingestion only: encoders and prefill chunks.
    Prefill,
    /// Everything but context ingestion.
    Token,
    /// No filter: only media and background work is ready.
    Any,
}

impl Policy for Arrival {
    fn init(&mut self, state: &Snapshot) {
        self.phases = Phases::new(&state.phases);
        for view in &state.requests {
            self.views.insert(view.id, view.clone());
        }
        // Running requests start in admission order, which the snapshot states as the
        // time each was admitted.
        let mut running: Vec<&RequestView> = state
            .requests
            .iter()
            .filter(|view| view.state == RequestStatus::Running)
            .collect();
        running.sort_by_key(|view| {
            (
                view.admitted_at.map(|at| at.0).unwrap_or_default(),
                view.id.0,
            )
        });
        self.admission = running.into_iter().map(|view| view.id).collect();
        for view in &state.requests {
            if view.state == RequestStatus::Waiting {
                self.waiting.push(view);
            }
        }
        // The cache orders the snapshot states, least recently used first.
        let mut pages: Vec<&PageView> = state.pages.iter().collect();
        pages.sort_unstable_by_key(|page| (page.last_used.0, page.id.0));
        self.pages = Lru::from_least_recent(pages.into_iter().map(|page| (page.id, page.clone())));
        let mut encodings: Vec<(u64, EncodingId)> = state
            .encodings
            .iter()
            .map(|entry| (entry.last_used.0, entry.id))
            .collect();
        encodings.sort_unstable_by_key(|(used, id)| (*used, id.0));
        self.encodings =
            Lru::from_least_recent(encodings.into_iter().map(|(_, entry)| (entry, ())));
    }

    fn observe(&mut self, events: &[Event]) {
        for event in events {
            match event {
                Event::Arrived { request } => {
                    self.views.insert(request.id, request.clone());
                    if request.state == RequestStatus::Waiting {
                        self.waiting.push(request);
                    }
                }
                // The view of a request changed: its next phase, prompt
                // progress, cached prefix or pages. The mechanism states no
                // emission instant; the one `Emitted` gave stays.
                Event::Updated { request } => {
                    let arrived = request.id;
                    if request.state == RequestStatus::Running
                        && !self.admission.contains(&arrived)
                        && !self.waiting.contains(arrived)
                    {
                        self.admission.push(arrived);
                    }
                    if request.state == RequestStatus::Waiting {
                        self.waiting.push(request);
                    }
                    let mut view = request.clone();
                    if view.last_emitted_at.is_none() {
                        view.last_emitted_at = self
                            .views
                            .get(&arrived)
                            .and_then(|known| known.last_emitted_at);
                    }
                    self.views.insert(arrived, view);
                }
                Event::Admitted { request } => {
                    self.waiting.remove(*request);
                    if !self.admission.contains(request) {
                        self.admission.push(*request);
                    }
                    if let Some(view) = self.views.get_mut(request) {
                        view.state = RequestStatus::Running;
                    }
                }
                // The view of the request carries what the policy acts on; the
                // advance itself needs no bookkeeping here.
                Event::Advanced { .. } => {}
                Event::Emitted {
                    request,
                    tokens,
                    at,
                } => {
                    if let Some(view) = self.views.get_mut(request) {
                        view.generated_tokens = view.generated_tokens.saturating_add(*tokens);
                        view.remaining_tokens = view.remaining_tokens.saturating_sub(*tokens);
                        view.last_emitted_at = Some(*at);
                    }
                }
                Event::Finished { request, .. } => {
                    self.waiting.remove(*request);
                    self.views.remove(request);
                    self.admission.retain(|running| running != request);
                }
                Event::Stepped { .. } => {}
                Event::Cached { page } => self.state(page),
                // A hit, a release and a hold name pages the snapshot or a `Cached` stated, each
                // now the most recently used.
                Event::Hit { page } => {
                    self.used(*page, "hit");
                }
                Event::Released { pages } => {
                    for page in pages {
                        let view = self.used(*page, "released");
                        view.evictable = true;
                        // A device page no request holds is offered to the host at the next step.
                        if !view.on_host {
                            self.offers.push(*page);
                        }
                    }
                }
                Event::Held { pages } => {
                    for page in pages {
                        self.used(*page, "held").evictable = false;
                    }
                }
                Event::Freed { page } => {
                    self.pages.remove(*page);
                }
                Event::Encoded { entry } => self.encodings.insert(entry.id, ()),
            }
        }
    }

    fn dispatch(&mut self, room: &Room) -> DispatchOutput {
        // A request whose worst-case reservation is larger than the whole
        // device cache can never run: it is rejected rather than left to block
        // the queue.
        let reject: Vec<RequestId> = self
            .waiting
            .iter()
            .filter(|id| {
                self.views
                    .get(id)
                    .is_some_and(|view| view.reservation_units > room.capacity_units)
            })
            .collect();
        // The step's kind, decided before the output is built: a refill is named
        // as the step's prefill work only when the step is not a token step.
        let pass = self.pass(room);
        let offload = self.offload_pages();
        let mut output = DispatchOutput {
            // Every other waiting request, in order. The mechanism admits them
            // until the first it cannot place, which is the admission
            // decision.
            admit: self
                .waiting
                .iter()
                .filter(|id| !reject.contains(id))
                .collect(),
            reject,
            // The device pages no running request continues from go to the
            // host KV cache, each once, least recently used first.
            offload,
            // Requests whose host hit deepens enough to be worth fetching, in
            // admission order.
            prefetch: self
                .admission
                .iter()
                .filter(|id| {
                    self.views
                        .get(id)
                        .is_some_and(|view| self.prefetch_due(view))
                })
                .copied()
                .collect(),
            ..DispatchOutput::default()
        };
        // A draft cache with a gap is refilled as the step's prefill work. In a
        // token step the request is prepared as its decode instead.
        if pass != Pass::Token {
            output.refill = self
                .admission
                .iter()
                .filter(|id| self.views.get(id).is_some_and(|view| self.refill_due(view)))
                .copied()
                .collect();
        }
        let shares = Shares::of(room);
        let mut step = Step::new(shares);
        let mut media_path = false;
        let mut other_path = false;
        for id in self.assembly_order() {
            if step.shares().full() {
                break;
            }
            // The step's refill is the request's call: preparing its decode too
            // would schedule two calls, the second before the one it depends on.
            if output.refill.contains(&id) {
                continue;
            }
            let Some(view) = self.views.get(&id).cloned() else {
                continue;
            };
            let (Some(phase), Some(kind)) = (view.phase, self.next_kind(&view)) else {
                continue;
            };
            let class = kind.class();
            match pass {
                // A prefill step carries context ingestion only, and a token
                // step every other kind.
                Pass::Prefill if class != PhaseClass::Context => continue,
                Pass::Token if class == PhaseClass::Context => continue,
                Pass::Prefill | Pass::Token | Pass::Any => {}
            }
            let tokens = self.work_tokens(&view, kind, room, &step.shares());
            if tokens == 0 {
                continue;
            }
            if !step.add(id, phase, tokens, kind.writes_prefix_cache()) {
                // The step lacks the room for this call: it is left to later
                // requests and later steps.
                continue;
            }
            if view.class == RequestClass::Media {
                media_path = true;
            } else {
                other_path = true;
            }
        }
        // A step that served both paths gives the other path precedence in the next one.
        if media_path && other_path {
            self.media_first = !self.media_first;
        }
        output.work = step.into_work();
        output
    }

    fn preempt(&mut self, shortage: &Shortage) -> Vec<Victim> {
        let mut victims = match shortage.kind {
            Storage::KvUnits => self.page_victims(false, shortage.need),
            // No page holds a checkpoint unit: the shortage takes requests alone, below.
            Storage::StateUnits => Vec::new(),
            Storage::HostPages => self.page_victims(true, shortage.need),
            Storage::EncoderEntries => {
                let mut victims = Vec::new();
                while (victims.len() as u64) < shortage.need {
                    let Some(entry) = self.encodings.pop_least() else {
                        break;
                    };
                    victims.push(Victim::Encoding(entry));
                }
                victims
            }
        };
        // Units are what the pages above free one of each; when the device
        // cache has no more page no request continues from, a request is the
        // victim, and the mechanism recomputes it later. A shortage of
        // checkpoint units is relieved by a request at once.
        if matches!(shortage.kind, Storage::KvUnits | Storage::StateUnits)
            && (victims.len() as u64) < shortage.need
            && let Some(victim) = self.request_victim()
        {
            victims.push(victim);
        }
        victims
    }

    fn route(&mut self, request: &RequestView, ranks: &[RankView]) -> usize {
        // A request resuming a stored cache goes back to the rank recorded for
        // it; the mechanism asks no routing decision for it.
        if let Some(rank) = request.cache_rank {
            return rank;
        }
        let count = ranks.len();
        if count == 0 {
            return 0;
        }
        let start = self.route_cursor % count;
        self.route_cursor = (start + 1) % count;
        // The live rank with the fewest requests in flight, from the cursor, so
        // ranks of equal load take new requests in turn.
        let chosen = (0..count)
            .map(|offset| (start + offset) % count)
            .filter(|index| ranks[*index].alive)
            .min_by_key(|index| ranks[*index].in_flight)
            .or_else(|| (0..count).map(|offset| (start + offset) % count).next());
        match chosen {
            Some(index) => ranks[index].index,
            None => 0,
        }
    }

    fn place(&mut self, call: &CallView, replicas: &[ReplicaView]) -> usize {
        // A decode-stage call never lands on a replica that only prefills.
        let decode_stage = matches!(call.kind, Phase::Decode | Phase::Verify);
        let mut candidates = replicas.iter().filter(|replica| {
            replica.kinds.contains(&call.kind) && !(decode_stage && replica.prefills_only)
        });
        // The replica that already holds the request's state keeps it: a
        // request's state is never migrated, and a full replica delays the call
        // rather than moving it.
        if let Some(held) = replicas.iter().find(|replica| replica.holds_request) {
            return held.index;
        }
        // Then a replica that already holds another stage of this request,
        // so its state stays in one process and its products are not published.
        if let Some(chosen) = candidates
            .clone()
            .filter(|replica| replica.holds_stage && replica.capacity > 0)
            .min_by_key(|replica| (replica.running_requests, replica.index))
        {
            return chosen.index;
        }
        // Otherwise the replica with the fewest running requests that can take
        // one, ties to the earliest in the list.
        if let Some(chosen) = candidates
            .clone()
            .filter(|replica| replica.capacity > 0)
            .min_by_key(|replica| (replica.running_requests, replica.index))
        {
            return chosen.index;
        }
        // No replica has room: name the least loaded one, and the mechanism
        // holds the call until one does.
        match candidates.next() {
            Some(replica) => replica.index,
            None => 0,
        }
    }

    fn limits(&self) -> Limits {
        self.limits
    }
}

#[cfg(test)]
mod tests {
    use scheduling::{
        Duration, EncodingView, End, Instantiation, Nanos, PhaseId, PhaseView, QueueOrder, StageId,
    };

    use super::*;

    /// The prefill window the tests' arguments state: two batches with prefill work in flight
    /// close it.
    const WINDOW: u32 = 2;

    /// The arguments the tests instantiate the policy with, admitting in `order`: chunks and
    /// steps far wider than any test's room, so a test's room and its requests decide.
    fn arguments(order: QueueOrder) -> Parameters {
        Parameters {
            order: Some(order),
            chunked_prefill_size: Some(8192),
            max_batch: Some(128),
            max_num_batched_tokens: Some(8192),
            prefetch_min_tokens: Some(256),
            timeout: Some(Nanos(30_000_000_000)),
            prefill_window: Some(WINDOW),
            ..Parameters::default()
        }
    }

    /// A policy constructed from the instantiation's arguments and initialized with its state,
    /// as the mechanism instantiates one.
    fn instantiated(init: Instantiation) -> Arrival {
        let mut policy = Arrival::try_from(init.parameters.clone()).unwrap();
        Policy::init(&mut policy, &init.state);
        policy
    }

    /// Arguments that leave out a parameter the policy requires refuse its construction,
    /// naming the parameter.
    #[test]
    fn arguments_without_a_required_parameter_are_refused() {
        let parameters = Parameters {
            prefill_window: None,
            ..arguments(QueueOrder::Arrival)
        };
        assert_eq!(
            Arrival::try_from(parameters).err(),
            Some(Unstated("prefill_window"))
        );
    }

    /// A policy with two phases: 0 prefill, 1 decode.
    fn policy() -> Arrival {
        instantiated(Instantiation {
            parameters: arguments(QueueOrder::Arrival),
            state: Snapshot {
                phases: vec![
                    PhaseView {
                        id: PhaseId(0),
                        kind: Phase::Prefill,
                        block: StageId(0),
                    },
                    PhaseView {
                        id: PhaseId(1),
                        kind: Phase::Decode,
                        block: StageId(0),
                    },
                ],
                ..Snapshot::default()
            },
        })
    }

    fn view(id: u64, priority: i32, arrived: u64) -> RequestView {
        RequestView {
            id: RequestId(id),
            state: RequestStatus::Waiting,
            phase: None,
            priority,
            arrived_at: Nanos(arrived),
            admitted_at: None,
            prompt_tokens: 100,
            // Nothing to commit: a request with no block commit states its prompt length.
            prefill_tokens: 100,
            generated_tokens: 0,
            cached_prefix_tokens: 0,
            prefilled_tokens: 0,
            cached_pages: 0,
            last_emitted_at: None,
            class: RequestClass::Token,
            waiting_transfer: false,
            next_tokens: 1,
            reservation_units: 0,
            prefetch_tokens: 0,
            refill_tokens: None,
            remaining_tokens: 0,
            migratable: true,
            cache_rank: None,
        }
    }

    fn room(tokens: u32, calls: usize) -> Room {
        Room {
            token_budget: tokens,
            call_budget: calls,
            page_tokens: 16,
            ..Room::default()
        }
    }

    fn admit_with_phase(
        policy: &mut Arrival,
        mut request: RequestView,
        phase: PhaseId,
    ) -> RequestId {
        request.state = RequestStatus::Running;
        request.phase = Some(phase);
        request.admitted_at = Some(Nanos(1));
        let id = request.id;
        policy.observe(&[Event::Admitted { request: id }]);
        policy.observe(&[Event::Updated { request }]);
        id
    }

    fn admit(policy: &mut Arrival, mut request: RequestView) -> RequestId {
        request.state = RequestStatus::Running;
        request.phase = Some(PhaseId(0));
        request.admitted_at = Some(Nanos(1));
        let id = request.id;
        policy.observe(&[Event::Admitted { request: id }]);
        policy.observe(&[Event::Updated { request }]);
        id
    }

    /// **An admitted request with a prompt left is prefilled, and the phase
    /// that says which call it is next is the mechanism's to state.**
    ///
    /// A waiting request the mechanism admitted
    /// is prefilled in the next step, chunked only by
    /// `limits().chunked_prefill_size` and the step's budget. It schedules that
    /// work by the phase the views carry (`RequestView::phase`), because the
    /// phase says which of the request's calls runs next and what it costs;
    /// without one it schedules nothing, since a policy cannot invent a phase id
    /// and an invented one would execute a call the mechanism never named.
    ///
    /// The absence is therefore **the mechanism's defect, not the policy's
    /// response**: a running request with prompt left always has a next call,
    /// and its view names that call's phase. Every helper in this module admitted
    /// a request with `phase: Some(..)` already, which is why no case could
    /// fail while the mechanism stated `None` and every text test went silent.
    /// This case pins both halves: no phase, no work; the same request stated
    /// with its prefill phase, the whole prompt in one chunk.
    #[test]
    fn an_admitted_request_is_prefilled_once_the_mechanism_names_its_phase() {
        let mut policy = policy();

        // The mechanism admitted the request and did not name its next call.
        let mut admitted = view(1, 0, 1);
        admitted.state = RequestStatus::Running;
        admitted.admitted_at = Some(Nanos(1));
        admitted.phase = None;
        policy.observe(&[Event::Admitted {
            request: RequestId(1),
        }]);
        policy.observe(&[Event::Updated {
            request: admitted.clone(),
        }]);

        let output = policy.dispatch(&room(1_024, 4));
        assert!(
            output.work.is_empty(),
            "a request whose next phase the mechanism did not state has no call a policy can \
             output; the absence is the mechanism's to fix, not the policy's to work around: \
             {output:?}"
        );

        // The mechanism names the prefill, and nothing else about the request
        // moved: the same admission now has a call to schedule.
        let mut named = admitted;
        named.phase = Some(PhaseId(0));
        policy.observe(&[Event::Updated { request: named }]);

        let output = policy.dispatch(&room(1_024, 4));
        assert_eq!(
            output.work.len(),
            1,
            "the admitted request is prefilled once its phase is stated: {output:?}"
        );
        assert_eq!(output.work[0].request, RequestId(1));
        assert_eq!(output.work[0].phase, PhaseId(0));
        assert_eq!(
            output.work[0].tokens, 100,
            "the whole prompt: it is under the step's budget and the chunk limit, so the rule \
             schedules one chunk"
        );
    }

    #[test]
    fn the_queue_decides_the_admission_order() {
        let mut policy = policy();
        policy.observe(&[
            Event::Arrived {
                request: view(2, 0, 20),
            },
            Event::Arrived {
                request: view(1, 0, 10),
            },
        ]);
        let output = policy.dispatch(&room(1024, 8));
        let order: Vec<u64> = output.admit.iter().map(|id| id.0).collect();
        assert_eq!(order, vec![1, 2]);
        assert!(output.work.is_empty());
    }

    #[test]
    fn a_prefill_chunk_is_clipped_and_the_prompt_advances() {
        let mut policy = policy();
        let mut request = view(1, 0, 0);
        request.prompt_tokens = 100;
        let _ = admit(&mut policy, request);
        let output = policy.dispatch(&room(1024, 8));
        assert_eq!(output.work.len(), 1);
        assert_eq!(output.work[0].tokens, 100);

        policy.observe(&[Event::Updated {
            request: RequestView {
                prefilled_tokens: 32,
                ..policy.views[&RequestId(1)].clone()
            },
        }]);
        let output = policy.dispatch(&room(1024, 8));
        // 32 prefilled of 100 leaves 68, which *is* the prompt end: a chunk
        // that reaches the end may end there, so the page clip does not bite.
        // Clipping needs room smaller than the tokens left.
        assert_eq!(output.work[0].tokens, 68);
        let output = policy.dispatch(&room(40, 8));
        // Under 68 left and a 40-token share: 32 + 40 = 72, clipped down to
        // the last 16-token page boundary at or before it, so 32.
        assert_eq!(output.work[0].tokens, 32);
    }

    #[test]
    fn a_decode_step_skips_context_work() {
        let mut policy = policy();
        let mut first = view(1, 0, 0);
        first.phase = Some(PhaseId(0));
        let _ = admit(&mut policy, first);
        let mut decode = view(2, 0, 1);
        decode.prompt_tokens = 10;
        // On its decode phase: `admit` would put it back on the prefill phase
        // and this step would be two prefills.
        let _ = admit_with_phase(&mut policy, decode, PhaseId(1));
        // One batch with prefill work is in flight and another is allowed, so
        // this step still prefills.
        let output = policy.dispatch(&room(1024, 8));
        assert_eq!(output.work[0].request, RequestId(1));
        // Two batches with prefill work in flight close the window.
        let mut closed = room(1024, 8);
        closed.inflight.prefill_batches = 2;
        let output = policy.dispatch(&closed);
        assert_eq!(output.work[0].request, RequestId(2));
        assert_eq!(output.work[0].tokens, 1);
    }

    #[test]
    fn a_step_stops_when_its_token_budget_is_spent() {
        let mut policy = policy();
        for id in 1..=4 {
            let mut request = view(id, 0, id);
            request.prompt_tokens = 100;
            let _ = admit(&mut policy, request);
        }
        let mut shares = room(80, 8);
        shares.page_tokens = 16;
        let output = policy.dispatch(&shares);
        let charged: u32 = output.work.iter().map(|work| work.tokens).sum();
        assert!(charged <= 80);
        assert!(output.work.len() < 4);
    }

    #[test]
    fn a_shortage_takes_pages_least_recently_used_first() {
        let mut policy = policy();
        policy.observe(&[
            Event::Cached {
                page: PageView {
                    id: PageId(1),
                    tokens: 16,
                    last_used: Nanos(1),
                    on_host: false,
                    evictable: true,
                },
            },
            Event::Cached {
                page: PageView {
                    id: PageId(2),
                    tokens: 16,
                    last_used: Nanos(2),
                    on_host: false,
                    evictable: true,
                },
            },
            Event::Hit { page: PageId(1) },
        ]);
        let victims = policy.preempt(&Shortage {
            kind: Storage::KvUnits,
            need: 1,
        });
        assert_eq!(victims, vec![Victim::Page(PageId(2))]);
    }

    #[test]
    fn a_device_shortage_takes_only_pages_no_request_continues_from() {
        let mut policy = policy();
        policy.observe(&[
            Event::Cached {
                page: PageView {
                    id: PageId(1),
                    tokens: 16,
                    last_used: Nanos(1),
                    on_host: false,
                    evictable: false,
                },
            },
            Event::Cached {
                page: PageView {
                    id: PageId(2),
                    tokens: 16,
                    last_used: Nanos(2),
                    on_host: false,
                    evictable: true,
                },
            },
        ]);
        // The mechanism refuses a page a running request still reads, so the older
        // one is not an offer it can accept: the free page is the victim and
        // the response stays valid.
        let victims = policy.preempt(&Shortage {
            kind: Storage::KvUnits,
            need: 1,
        });
        assert_eq!(victims, vec![Victim::Page(PageId(2))]);
        // Nothing free: no page is named, which is a valid response (the work
        // waits), rather than one the mechanism has to refuse.
        policy.observe(&[Event::Hit { page: PageId(2) }]);
        let victims = policy.preempt(&Shortage {
            kind: Storage::KvUnits,
            need: 1,
        });
        assert!(
            victims
                .iter()
                .all(|victim| *victim != Victim::Page(PageId(1)))
        );
    }

    #[test]
    fn a_host_shortage_takes_host_pages_only() {
        let mut policy = policy();
        policy.observe(&[
            Event::Cached {
                page: PageView {
                    id: PageId(1),
                    tokens: 16,
                    last_used: Nanos(1),
                    on_host: true,
                    evictable: true,
                },
            },
            Event::Cached {
                page: PageView {
                    id: PageId(2),
                    tokens: 16,
                    last_used: Nanos(2),
                    on_host: false,
                    evictable: true,
                },
            },
        ]);
        let victims = policy.preempt(&Shortage {
            kind: Storage::HostPages,
            need: 2,
        });
        assert_eq!(victims, vec![Victim::Page(PageId(1))]);
    }

    #[test]
    fn an_encoder_shortage_takes_the_least_recently_used_entry() {
        let mut policy = policy();
        policy.observe(&[
            Event::Encoded {
                entry: EncodingView {
                    id: EncodingId(5),
                    last_used: Nanos(1),
                },
            },
            Event::Encoded {
                entry: EncodingView {
                    id: EncodingId(6),
                    last_used: Nanos(2),
                },
            },
        ]);
        let victims = policy.preempt(&Shortage {
            kind: Storage::EncoderEntries,
            need: 1,
        });
        assert_eq!(victims, vec![Victim::Encoding(EncodingId(5))]);
    }

    /// A shortage of checkpoint units names no page, however many are evictable: no page holds
    /// one. It takes the request the device shortage would.
    #[test]
    fn a_checkpoint_shortage_takes_a_request_and_no_page() {
        let mut policy = policy();
        policy.observe(&[Event::Cached {
            page: PageView {
                id: PageId(1),
                tokens: 16,
                last_used: Nanos(1),
                on_host: false,
                evictable: true,
            },
        }]);
        admit(&mut policy, view(1, 0, 0));
        admit(&mut policy, view(2, 5, 1));

        let victims = policy.preempt(&Shortage {
            kind: Storage::StateUnits,
            need: 1,
        });
        assert_eq!(victims, vec![Victim::Request(RequestId(2))]);
    }

    #[test]
    fn a_request_victim_is_the_lowest_priority_admitted_last() {
        let mut policy = policy();
        let mut first = view(1, 0, 0);
        first.admitted_at = Some(Nanos(10));
        admit(&mut policy, first);
        let mut second = view(2, 5, 1);
        second.admitted_at = Some(Nanos(20));
        admit(&mut policy, second);
        let mut third = view(3, 0, 2);
        third.admitted_at = Some(Nanos(30));
        admit(&mut policy, third);
        let victims = policy.preempt(&Shortage {
            kind: Storage::KvUnits,
            need: 1,
        });
        assert_eq!(victims, vec![Victim::Request(RequestId(2))]);
    }

    #[test]
    fn routing_takes_the_least_loaded_live_rank_from_the_cursor() {
        let mut policy = policy();
        let ranks = [
            RankView {
                index: 0,
                in_flight: 3,
                alive: true,
            },
            RankView {
                index: 1,
                in_flight: 1,
                alive: true,
            },
            RankView {
                index: 2,
                in_flight: 1,
                alive: false,
            },
        ];
        assert_eq!(policy.route(&view(1, 0, 0), &ranks), 1);
        // The cursor advances by one per pick, not to one past the rank it
        // chose: the second pick starts at 1, so with two ranks the rotation
        // is [1, 0] and equal load keeps rank 1.
        let ranks = [
            RankView {
                index: 0,
                in_flight: 1,
                alive: true,
            },
            RankView {
                index: 1,
                in_flight: 1,
                alive: true,
            },
        ];
        assert_eq!(policy.route(&view(2, 0, 0), &ranks), 1);
    }

    #[test]
    fn a_resuming_request_keeps_its_recorded_rank() {
        let mut policy = policy();
        let mut request = view(1, 0, 0);
        request.cache_rank = Some(3);
        let ranks = [RankView {
            index: 0,
            in_flight: 0,
            alive: true,
        }];
        assert_eq!(policy.route(&request, &ranks), 3);
    }

    #[test]
    fn a_held_replica_keeps_the_request_and_a_decode_avoids_prefill_only() {
        let mut policy = policy();
        let call = CallView {
            request: RequestId(1),
            phase: PhaseId(1),
            kind: Phase::Decode,
        };
        let replicas = [
            ReplicaView {
                index: 0,
                block: StageId(0),
                kinds: vec![Phase::Decode],
                running_requests: 0,
                capacity: 4,
                holds_request: false,
                prefills_only: true,
                holds_stage: false,
            },
            ReplicaView {
                index: 1,
                block: StageId(0),
                kinds: vec![Phase::Decode],
                running_requests: 3,
                capacity: 4,
                holds_request: false,
                prefills_only: false,
                holds_stage: false,
            },
        ];
        assert_eq!(policy.place(&call, &replicas), 1);
        let held = ReplicaView {
            index: 0,
            holds_request: true,
            prefills_only: false,
            ..replicas[0].clone()
        };
        let replicas = [held, replicas[1].clone()];
        assert_eq!(policy.place(&call, &replicas), 0);
    }

    #[test]
    fn priority_admits_and_assembles_by_priority_then_arrival() {
        let mut priority = instantiated(Instantiation {
            parameters: arguments(QueueOrder::Priority),
            state: Snapshot {
                phases: vec![
                    PhaseView {
                        id: PhaseId(0),
                        kind: Phase::Prefill,
                        block: StageId(0),
                    },
                    PhaseView {
                        id: PhaseId(1),
                        kind: Phase::Decode,
                        block: StageId(0),
                    },
                ],
                ..Snapshot::default()
            },
        });
        // The later request carries the more urgent priority.
        priority.observe(&[
            Event::Arrived {
                request: view(1, 5, 10),
            },
            Event::Arrived {
                request: view(2, 0, 20),
            },
        ]);
        let output = priority.dispatch(&room(1024, 8));
        let order: Vec<u64> = output.admit.iter().map(|id| id.0).collect();
        assert_eq!(order, vec![2, 1]);

        // Admitted in that order, they are assembled in it too.
        for id in [2u64, 1] {
            let mut request = view(id, if id == 2 { 0 } else { 5 }, id * 10);
            request.state = RequestStatus::Running;
            request.phase = Some(PhaseId(0));
            request.admitted_at = Some(Nanos(id));
            priority.observe(&[Event::Admitted {
                request: RequestId(id),
            }]);
            priority.observe(&[Event::Updated { request }]);
        }
        let output = priority.dispatch(&room(1024, 8));
        let order: Vec<u64> = output.work.iter().map(|work| work.request.0).collect();
        assert_eq!(order, vec![2, 1]);
    }

    #[test]
    fn a_refill_is_the_steps_prefill_work() {
        let mut policy = policy();
        // A decoding request whose draft cache is 4 tokens short, and a plain
        // decode. No view sits in a refill phase: the refill is a call the
        // mechanism builds for a request the output names.
        let mut short = view(1, 0, 0);
        short.phase = Some(PhaseId(1));
        short.prefilled_tokens = 16;
        short.refill_tokens = Some(4);
        let _ = admit_with_phase(&mut policy, short, PhaseId(1));
        let decoding = view(2, 0, 1);
        let _ = admit_with_phase(&mut policy, decoding, PhaseId(1));

        // A refill-due running is prefill work, so with the window open the
        // step is a prefill step: it names the refill and schedules no work entry
        // for that request — the refill is its call.
        let output = policy.dispatch(&room(1024, 8));
        assert_eq!(output.refill, vec![RequestId(1)]);
        assert!(output.work.is_empty(), "{:?}", output.work);

        // Two prefill batches in flight close the window: the step is a token
        // step, so the refill is not named and both requests decode.
        let mut closed = room(1024, 8);
        closed.inflight.prefill_batches = WINDOW;
        let output = policy.dispatch(&closed);
        assert!(output.refill.is_empty());
        let work: Vec<(u64, u32)> = output
            .work
            .iter()
            .map(|work| (work.request.0, work.tokens))
            .collect();
        assert_eq!(work, vec![(1, 1), (2, 1)]);
    }

    #[test]
    fn a_full_step_decides_every_entry_of_its_output() {
        // Two queued: the first fits the cache, the second never can.
        let mut fits = view(10, 0, 1);
        fits.reservation_units = 10;
        let mut enormous = view(11, 0, 2);
        enormous.reservation_units = 1_000;
        // A running prefill, a running request decode with a draft gap, and a
        // running media request whose host hit deepens.
        let mut prefill = view(20, 0, 3);
        prefill.state = RequestStatus::Running;
        prefill.phase = Some(PhaseId(0));
        prefill.admitted_at = Some(Nanos(1));
        let mut decode = view(21, 0, 4);
        decode.state = RequestStatus::Running;
        decode.phase = Some(PhaseId(1));
        decode.admitted_at = Some(Nanos(2));
        decode.refill_tokens = Some(8);
        let mut media = view(22, 0, 5);
        media.state = RequestStatus::Running;
        media.phase = Some(PhaseId(2));
        media.class = RequestClass::Media;
        media.admitted_at = Some(Nanos(3));
        media.prefetch_tokens = 300;
        let mut policy = instantiated(Instantiation {
            parameters: arguments(QueueOrder::Arrival),
            state: Snapshot {
                requests: vec![fits.clone(), enormous, prefill, decode, media],
                phases: vec![
                    PhaseView {
                        id: PhaseId(0),
                        kind: Phase::Prefill,
                        block: StageId(0),
                    },
                    PhaseView {
                        id: PhaseId(1),
                        kind: Phase::Decode,
                        block: StageId(0),
                    },
                    PhaseView {
                        id: PhaseId(2),
                        kind: Phase::Denoise,
                        block: StageId(1),
                    },
                ],
                pages: vec![
                    PageView {
                        id: PageId(1),
                        tokens: 16,
                        last_used: Nanos(1),
                        on_host: false,
                        evictable: false,
                    },
                    PageView {
                        id: PageId(2),
                        tokens: 16,
                        last_used: Nanos(2),
                        on_host: false,
                        evictable: false,
                    },
                    PageView {
                        id: PageId(3),
                        tokens: 16,
                        last_used: Nanos(3),
                        on_host: true,
                        evictable: true,
                    },
                ],
                ..Snapshot::default()
            },
        });
        // The requests that held pages 1 and 2 gave them up.
        policy.observe(&[Event::Released {
            pages: vec![PageId(1), PageId(2)],
        }]);
        let room = Room {
            token_budget: 1024,
            call_budget: 8,
            page_tokens: 16,
            prefill_tokens: Some(64),
            capacity_units: 100,
            ..Room::default()
        };
        // With the prefill window open, the step carries context work only.
        let output = policy.dispatch(&room);
        let order: Vec<u64> = output.work.iter().map(|work| work.request.0).collect();
        assert_eq!(order, vec![20]);
        // `Room::prefill_tokens` is the warming reservation every
        // prefix-writing call is charged against, so a 100-token chunk is
        // capped at 64.
        assert_eq!(output.work[0].tokens, 64);
        // The over-capacity request is rejected and not also admitted.
        assert_eq!(output.reject, vec![RequestId(11)]);
        assert_eq!(output.admit, vec![RequestId(10)]);
        // The device pages no request continues from any more go to the
        // host, in least-recently-used order; the host page stays.
        assert_eq!(output.offload, vec![PageId(1), PageId(2)]);
        assert_eq!(output.prefetch, vec![RequestId(22)]);
        assert_eq!(output.refill, vec![RequestId(21)]);

        // Two prefill batches in flight close the window: the step carries
        // token progress and media work, in class order within the generation
        // path, and the media path second.
        let mut closed = room.clone();
        closed.inflight.prefill_batches = WINDOW;
        let output = policy.dispatch(&closed);
        // A token step schedules a refill-due request as its decode, so it names
        // no refill; the pages were offloaded once.
        assert!(output.refill.is_empty());
        assert!(output.offload.is_empty());
        let order: Vec<u64> = output.work.iter().map(|work| work.request.0).collect();
        assert_eq!(order, vec![21, 22]);
        assert!(output.work.iter().all(|work| work.tokens == 1));
    }

    #[test]
    fn a_step_does_not_disturb_the_state() {
        let mut policy = policy();
        policy.observe(&[Event::Stepped {
            step: 7,
            duration: Duration(10),
        }]);
        assert!(policy.dispatch(&room(1024, 8)).work.is_empty());
    }
    /// A policy with three phases: 0 prefill, 1 decode, 2 denoise.
    fn policy_with_media() -> Arrival {
        instantiated(Instantiation {
            parameters: arguments(QueueOrder::Arrival),
            state: Snapshot {
                phases: vec![
                    PhaseView {
                        id: PhaseId(0),
                        kind: Phase::Prefill,
                        block: StageId(0),
                    },
                    PhaseView {
                        id: PhaseId(1),
                        kind: Phase::Decode,
                        block: StageId(0),
                    },
                    PhaseView {
                        id: PhaseId(2),
                        kind: Phase::Denoise,
                        block: StageId(1),
                    },
                ],
                ..Snapshot::default()
            },
        })
    }

    /// **The pages a step offloads are the ones no running request continues
    /// from, each once, in the step after it became one.**
    ///
    /// The mechanism refuses to free a page a running request still continues
    /// from, so naming one would fail the policy at the
    /// first offload; `RequestView::cached_pages` and the mechanism's own
    /// `evictable` flag are how the policy knows. The flag moves after a page
    /// is announced (a `Released` names the page its last user gave up), and
    /// the page is offloaded in the next step. The mechanism offers a page to
    /// the host once, so a page named in one step, or stated by the snapshot
    /// a policy starts from (its instance before this one had its step to name
    /// it), is not named again: a step costs what it changed, however many
    /// pages the cache holds. A page a request holds again before the step
    /// is not named, and one given up again after a later hold is.
    #[test]
    fn a_page_no_request_continues_from_is_offloaded_once() {
        let page = |id: u64, used: u64, evictable: bool| PageView {
            id: PageId(id),
            tokens: 16,
            last_used: Nanos(used),
            on_host: false,
            evictable,
        };
        let mut policy = instantiated(Instantiation {
            parameters: arguments(QueueOrder::Arrival),
            state: Snapshot {
                pages: vec![page(1, 1, true), page(2, 2, false), page(3, 3, false)],
                ..Snapshot::default()
            },
        });
        assert!(policy.dispatch(&room(1024, 8)).offload.is_empty());

        // The mechanism states the last users of pages 2 and 3 are gone, and
        // that page 3 is held again before the step: page 2 is now the most
        // recently used page no request continues from.
        policy.observe(&[
            Event::Released {
                pages: vec![PageId(2), PageId(3)],
            },
            Event::Held {
                pages: vec![PageId(3)],
            },
        ]);
        assert_eq!(policy.dispatch(&room(1024, 8)).offload, vec![PageId(2)]);
        assert!(policy.dispatch(&room(1024, 8)).offload.is_empty());

        // A page that enters the cache as one no request holds is offloaded
        // once too, and so is a page given up again after a hold.
        policy.observe(&[
            Event::Cached {
                page: page(4, 4, true),
            },
            Event::Held {
                pages: vec![PageId(2)],
            },
            Event::Released {
                pages: vec![PageId(2)],
            },
        ]);
        assert_eq!(
            policy.dispatch(&room(1024, 8)).offload,
            vec![PageId(4), PageId(2)]
        );
        assert!(policy.dispatch(&room(1024, 8)).offload.is_empty());
    }

    /// A page of the tests' caches, as the mechanism states it.
    fn page_view(id: u64, used: u64, evictable: bool) -> PageView {
        PageView {
            id: PageId(id),
            tokens: 16,
            last_used: Nanos(used),
            on_host: false,
            evictable,
        }
    }

    /// **A hold decides what restating each page as held decides.**
    ///
    /// The mechanism tells the pages an admission takes out of the eviction
    /// order as one `Held`, in the order it takes them, where it could restate
    /// each as a `Cached` page that is no longer evictable: the policy keeps
    /// the same order and the same flags either way, so it offloads the same
    /// pages and names the same victims: neither page held.
    #[test]
    fn a_hold_decides_as_restating_each_page_held() {
        let page = |id: u64, used: u64, evictable: bool| Event::Cached {
            page: page_view(id, used, evictable),
        };
        let free = [
            page(1, 1, true),
            page(2, 2, true),
            page(3, 3, true),
            page(4, 4, true),
        ];

        let mut restated = policy();
        restated.observe(&free);
        restated.observe(&[page(3, 10, false), page(1, 11, false)]);
        let mut held = policy();
        held.observe(&free);
        held.observe(&[Event::Held {
            pages: vec![PageId(3), PageId(1)],
        }]);

        for policy in [&mut restated, &mut held] {
            assert_eq!(
                policy.dispatch(&room(1024, 8)).offload,
                vec![PageId(2), PageId(4)]
            );
            let victims = policy.preempt(&Shortage {
                kind: Storage::KvUnits,
                need: 4,
            });
            assert_eq!(
                victims,
                vec![Victim::Page(PageId(2)), Victim::Page(PageId(4))]
            );
        }
    }

    /// **A release decides what restating each page as evictable decides.**
    ///
    /// The mechanism tells the pages their last holders gave up as one
    /// `Released`, in the order they join the eviction order, where it could
    /// restate each as a `Cached` page that is now evictable: the policy keeps
    /// the same order and the same flags either way, so it offloads the same
    /// pages and names the same victims. A page still held is named by
    /// neither.
    #[test]
    fn a_release_decides_as_restating_each_page_evictable() {
        let page = |id: u64, used: u64, evictable: bool| Event::Cached {
            page: page_view(id, used, evictable),
        };
        let held = [
            page(1, 1, false),
            page(2, 2, false),
            page(3, 3, false),
            page(4, 4, false),
        ];

        let mut restated = policy();
        restated.observe(&held);
        restated.observe(&[page(3, 10, true), page(1, 11, true)]);
        let mut released = policy();
        released.observe(&held);
        released.observe(&[Event::Released {
            pages: vec![PageId(3), PageId(1)],
        }]);

        for policy in [&mut restated, &mut released] {
            assert_eq!(
                policy.dispatch(&room(1024, 8)).offload,
                vec![PageId(3), PageId(1)]
            );
            let victims = policy.preempt(&Shortage {
                kind: Storage::KvUnits,
                need: 2,
            });
            assert_eq!(
                victims,
                vec![Victim::Page(PageId(3)), Victim::Page(PageId(1))]
            );
        }
    }

    /// **A release of a page the mechanism never stated traps.**
    ///
    /// The mechanism names in a hit, a release and a hold only pages the snapshot or a `Cached`
    /// stated; a policy told of another no longer follows the mechanism's view of the cache, and
    /// its trap has the mechanism instantiate it again from a snapshot.
    #[test]
    #[should_panic(expected = "the mechanism released page 9 it never stated")]
    fn a_release_of_a_page_never_stated_traps() {
        let mut policy = policy();
        policy.observe(&[Event::Released {
            pages: vec![PageId(9)],
        }]);
    }

    /// **A host hit shorter than the floor is not worth prefetching.**
    ///
    /// A deepening of fewer tokens than `prefetch_min_tokens` is computed
    /// faster than it is fetched, so the policy asks for the fetches that pay
    /// and leaves the rest. A policy that asked for every deepening would
    /// spend the step's transfers on hits the mechanism would drop anyway.
    #[test]
    fn a_hit_shorter_than_the_floor_is_not_prefetched() {
        let mut policy = policy();
        let floor = Policy::limits(&policy).prefetch_min_tokens;

        let mut short = view(1, 0, 1);
        short.prefetch_tokens = floor - 1;
        let mut deep = view(2, 0, 2);
        deep.prefetch_tokens = floor;
        admit_with_phase(&mut policy, short, PhaseId(1));
        admit_with_phase(&mut policy, deep, PhaseId(1));

        assert_eq!(policy.dispatch(&room(1024, 8)).prefetch, vec![RequestId(2)]);
    }

    /// **The two paths alternate their precedence between steps.**
    ///
    /// After a step that carried both a media call and
    /// a generation call, the path that was second goes first in the next
    /// step, so neither a decoder nor a generation request waits behind the
    /// other indefinitely. A policy that always preferred one path would pass
    /// every case that serves one class of request and starve the other.
    #[test]
    fn the_two_paths_alternate_their_precedence_between_steps() {
        let mut policy = policy_with_media();
        admit_with_phase(&mut policy, view(1, 0, 1), PhaseId(1));
        let mut media = view(2, 0, 2);
        media.class = RequestClass::Media;
        admit_with_phase(&mut policy, media, PhaseId(2));

        let step = room(16, 4);
        let first: Vec<u64> = policy
            .dispatch(&step)
            .work
            .iter()
            .map(|work| work.request.0)
            .collect();
        assert_eq!(first, vec![1, 2]);
        let second: Vec<u64> = policy
            .dispatch(&step)
            .work
            .iter()
            .map(|work| work.request.0)
            .collect();
        assert_eq!(second, vec![2, 1]);
    }

    /// **A rejected request is left behind once its end is announced.**
    ///
    /// The policy rejects a request whose worst-case reservation is larger than
    /// the whole cache, and the mechanism drops it from its queue and reports
    /// `End::Rejected`. From then on the policy names it nowhere: not in
    /// `admit` (the mechanism no longer queues it, and naming it there is a
    /// refused decision that stops the epoch) and not in `reject` again. The
    /// requests behind it are unaffected.
    #[test]
    fn a_rejected_request_is_left_behind_once_its_end_is_announced() {
        let mut policy = policy();
        let mut over = view(1, 0, 1);
        over.reservation_units = 1_000;
        let mut fits = view(2, 0, 2);
        fits.reservation_units = 10;
        policy.observe(&[
            Event::Arrived { request: over },
            Event::Arrived { request: fits },
        ]);

        let step = Room {
            capacity_units: 100,
            ..room(1024, 8)
        };
        let output = policy.dispatch(&step);
        assert_eq!(output.reject, vec![RequestId(1)]);
        assert_eq!(output.admit, vec![RequestId(2)]);

        policy.observe(&[
            Event::Admitted {
                request: RequestId(2),
            },
            Event::Finished {
                request: RequestId(1),
                end: End::Rejected,
            },
        ]);
        let output = policy.dispatch(&step);
        assert!(output.admit.is_empty());
        assert!(output.reject.is_empty());
    }
}
