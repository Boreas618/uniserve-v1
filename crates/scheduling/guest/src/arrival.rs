//! The body every arrival-order policy shares: `fcfs`, `priority`, and the
//! policies that order by a key and differ nowhere else.
//!
//! Design §4.4 fixes what `fcfs` must reproduce item by item (the last column
//! of its table) and says both default policies are built on the guest's
//! common parts, so a new ordering is the only thing a new policy writes. This
//! is that common body; a policy crate names its order key and nothing else:
//!
//! | decision | rule this body implements |
//! | --- | --- |
//! | `admit` | the waiting queue in order ([`Waiting`]); the mechanism admits until the first request it cannot place |
//! | `work` | requests in `(phase class, admission order)`, at most one call each, charged against the step's tokens and calls; a prefill chunk is [`prefill_chunk`] of the prompt left; a prefill-class call runs only in a prefill step; the prefill window holds while [`PREFILL_WINDOW_CREDITS`] batches with prefill work are in flight |
//! | `reject` | a queued request whose worst-case reservation is more than the device cache's whole capacity |
//! | `offload` | the device pages no resident request continues from, least recently used first |
//! | `prefetch` | the requests whose host hit deepens by at least `Limits::prefetch_min_tokens` |
//! | `refill` | the requests whose draft cache has a gap, when the refill is worth its cost (see [`Arrival::refill_due`]) |
//! | `reclaim` | the least recently used page of the short storage, then the lowest-priority last-admitted request |
//! | `route` | the recorded cache rank, else the live rank with the fewest requests in flight, from a rotating cursor |
//! | `place` | the replica that already holds the request; else the one with the fewest resident requests among those with capacity; a decode-stage call never lands on a prefill-only replica |
//! | `warm` | the resumed requests' pages in the order the requests were claimed, each request's pages in prefix order, within `Room::prefill_tokens` |
//! | `experts` | the placement unchanged |
//!
//! # The one judgement the interface cannot state
//!
//! `refill` weighs a refill's cost against the tokens the request may still
//! generate (today `RefillPolicy::worth`), and the views carry the cost
//! ([`RequestView::refill_tokens`]) but not the remaining token budget; see
//! [`Arrival::refill_due`].

use std::collections::HashMap;

use crate::budget::{Shares, Step, prefill_chunk, work_tokens};
use crate::decision::{Plan, Room, Shortage, Storage, Victim};
use crate::event::Event;
use crate::ids::{EncodingId, PageId, RequestId};
use crate::limits::{Limits, Parameters};
use crate::lru::Lru;
use crate::phases::Phases;
use crate::policy::{Init, Policy};
use crate::view::{
    CallView, ExpertLoad, ExpertOrder, PhaseClass, PhaseKind, RankView, ReplicaView, RequestClass,
    RequestState, RequestView,
};
use crate::waiting::{Order, Waiting};

/// How many batches carrying prefill work may be in flight before ready token
/// progress takes the step instead (today's `PREFILL_WINDOW_CREDITS`).
pub const PREFILL_WINDOW_CREDITS: u32 = 2;

/// The assembly class of a phase kind, with a draft-cache refill as the
/// prefill work it is.
///
/// Today a refill chunk is planned only in a pass that is not a decode pass
/// and is assembled with the prefills (`batching.rs`: `refill_due` counts as
/// prefill-ready and `plan_refill` plans a prefill call), so it belongs with
/// the context work whatever class its phase kind carries.
fn class_of(kind: PhaseKind) -> PhaseClass {
    match kind {
        PhaseKind::Refill => PhaseClass::Context,
        other => other.class(),
    }
}

/// A policy that admits by a key derived from the request, and schedules by
/// arrival.
///
/// `K` is the order of the waiting queue: [`crate::ArrivalOrder`] admits by
/// arrival alone (`fcfs`), [`crate::PriorityOrder`] by the request's priority
/// and then arrival (`priority`). Everything else is the same, as design §4.4
/// has it for the two default policies.
#[derive(Clone, Debug)]
pub struct Arrival<K: Order> {
    /// The policy's arguments, as the deployment states them.
    parameters: Parameters,
    /// The limits those arguments declare, read once.
    limits: Limits,
    /// The epoch's phases, so a request's next phase can be classified.
    phases: Phases,
    /// Requests waiting for admission, in this policy's order.
    waiting: Waiting<K>,
    /// The latest view of every request the mechanism has told this policy
    /// about: queued, resident, and those that ended.
    views: HashMap<RequestId, RequestView>,
    /// The residents, in the order the mechanism admitted them.
    admission: Vec<RequestId>,
    /// The pages of the device and host caches, least recently used first.
    pages: Lru<PageId>,
    /// Whether each page sits in host memory.
    on_host: HashMap<PageId, bool>,
    /// Whether no resident request continues from each page.
    evictable: HashMap<PageId, bool>,
    /// The encoder cache entries, least recently used first.
    encodings: Lru<EncodingId>,
    /// The rank the last `route` picked, so ties rotate as they do today.
    route_cursor: usize,
    /// Whether the next step puts the media path's work first, which alternates
    /// the two paths as today's `prefer_media` does.
    media_first: bool,
}

impl<K: Order> Default for Arrival<K> {
    fn default() -> Self {
        Self {
            parameters: Parameters::default(),
            limits: Limits::default(),
            phases: Phases::default(),
            waiting: Waiting::new(),
            views: HashMap::new(),
            admission: Vec::new(),
            pages: Lru::new(),
            on_host: HashMap::new(),
            evictable: HashMap::new(),
            encodings: Lru::new(),
            route_cursor: 0,
            media_first: false,
        }
    }
}

impl<K: Order> Arrival<K> {
    /// The admission position of a request; residents the snapshot listed
    /// before this policy saw an admission event sort after them by admission
    /// time, so the order is the mechanism's.
    fn admission_index(&self, id: RequestId) -> usize {
        self.admission
            .iter()
            .position(|resident| *resident == id)
            .unwrap_or(usize::MAX)
    }

    /// The phase kind of a request's next call, if it has one ready.
    fn next_kind(&self, view: &RequestView) -> Option<PhaseKind> {
        view.phase.map(|phase| self.phases.kind(phase))
    }

    /// Whether a resident request contributes work of `class` in this step.
    fn has_work_of_class(&self, class: PhaseClass) -> bool {
        self.views.values().any(|view| {
            view.state == RequestState::Resident
                && self
                    .next_kind(view)
                    .is_some_and(|kind| kind.class() == class)
        })
    }

    /// Whether a resident request is a readout that can advance this step:
    /// today a ready readout takes the step before prefill does.
    fn has_readout(&self) -> bool {
        self.views.values().any(|view| {
            view.state == RequestState::Resident
                && view.class == RequestClass::Readout
                && self
                    .next_kind(view)
                    .is_some_and(|kind| class_of(kind) == PhaseClass::Token)
        })
    }

    /// The pass of this step: prefill (context work only) or token progress
    /// (everything but context work), as today's `select_batch_kind` decides.
    fn pass(&self, room: &Room) -> Pass {
        if self.has_readout() {
            return Pass::Token;
        }
        let prefill_ready = self.has_work_of_class(PhaseClass::Context);
        let window_open = room.inflight.prefill_batches < PREFILL_WINDOW_CREDITS;
        if prefill_ready && window_open {
            return Pass::Prefill;
        }
        if self.has_work_of_class(PhaseClass::Token) {
            return Pass::Token;
        }
        if prefill_ready {
            return Pass::Prefill;
        }
        // Only media or background work is ready: no kind filter applies.
        Pass::Any
    }

    /// The residents in this step's assembly order: the preferred path first,
    /// then the phase class, then admission order.
    fn assembly_order(&self) -> Vec<RequestId> {
        let mut order: Vec<RequestId> = self
            .views
            .values()
            .filter(|view| view.state == RequestState::Resident && view.phase.is_some())
            .map(|view| view.id)
            .collect();
        order.sort_by_key(|id| {
            let view = &self.views[id];
            let kind = self.next_kind(view).unwrap_or(PhaseKind::Other);
            let media = view.class == RequestClass::Media;
            // The path the policy did not prefer comes second.
            let path = u8::from(media != self.media_first);
            (path, class_of(kind), self.admission_index(*id), id.0)
        });
        order
    }

    /// The token work of one call, and whether it writes the prefix cache.
    ///
    /// A prefill or refill chunk is sized by [`prefill_chunk`] from the
    /// request's prompt progress; every other phase states one token, which
    /// the mechanism charges and validates against the phase
    /// ([`work_tokens`]).
    fn work_tokens(&self, view: &RequestView, room: &Room, shares: &Shares) -> (u32, bool) {
        let Some(phase) = view.phase else {
            return (0, false);
        };
        let kind = self.phases.kind(phase);
        if !kind.writes_prefix_cache() {
            return (work_tokens(kind, 0, view.next_tokens), false);
        }
        let start = view
            .cached_prefix_tokens
            .saturating_add(view.prefilled_tokens);
        let share = shares.tokens().min(shares.prefill());
        let chunk = prefill_chunk(
            start,
            view.prompt_tokens,
            self.limits.chunked_prefill_size,
            share,
            room.page_tokens,
        );
        (chunk, true)
    }

    /// Whether a request's missing draft cache is refilled now.
    ///
    /// The mechanism states the cost of the gap (`refill_tokens`); the
    /// judgement is today's `RefillPolicy::worth`: with a stated speedup `g`
    /// and token cost `k`, a refill of `t` tokens of a request that may still
    /// generate `remaining` tokens is worth it when
    /// `remaining * (1 - 1/g) > t * k`, and with neither stated nor measured
    /// the refill runs. The measured values today's scheduler weighs
    /// (`SpeculationMeter`) are the mechanism's measurement, which the views
    /// do not carry; the policy's parameters replace them, as design §7.7
    /// says they may.
    fn refill_due(&self, view: &RequestView) -> bool {
        let Some(tokens) = view.refill_tokens else {
            return false;
        };
        if tokens == 0 {
            return false;
        }
        // A request already refilling its draft cache has the refill as its
        // next call; asking again would start a second one.
        if view
            .phase
            .is_some_and(|phase| self.phases.kind(phase) == PhaseKind::Refill)
        {
            return false;
        }
        let (Some(speedup), Some(cost)) = (
            self.parameters.expected_speedup,
            self.parameters.refill_token_cost,
        ) else {
            // Nothing stated: today's `worth` has no value to weigh and
            // refills.
            return true;
        };
        let saved = f64::from(view.remaining_tokens) * (1.0 - 1.0 / speedup.max(1.0));
        saved > f64::from(tokens) * cost.max(0.0)
    }

    /// Whether a request's host prefix hit deepens by enough to prefetch: the
    /// design's own floor (`Limits::prefetch_min_tokens`, today
    /// `PREFETCH_MIN_TOKENS`), because a shorter hit is computed faster than it
    /// is fetched.
    fn prefetch_due(&self, view: &RequestView) -> bool {
        view.prefetch_tokens >= self.limits.prefetch_min_tokens
    }

    /// The device pages a step offloads: every page no resident request
    /// continues from, least recently used first, which is today's rule for
    /// writing a prefix to the host KV cache.
    fn offload_pages(&self) -> Vec<PageId> {
        self.pages
            .iter()
            .filter(|page| {
                !self.on_host.get(page).copied().unwrap_or(false)
                    && self.evictable.get(page).copied().unwrap_or(false)
            })
            .collect()
    }

    /// The pages a shortage of `kind` may take, least recently used first.
    fn page_victims(&self, want_host: bool, need: u64) -> Vec<Victim> {
        let mut victims = Vec::new();
        let mut taken = 0u64;
        for page in self.pages.iter() {
            if taken >= need {
                break;
            }
            if self.on_host.get(&page).copied().unwrap_or(false) != want_host {
                continue;
            }
            victims.push(Victim::Page(page));
            taken += 1;
        }
        victims
    }

    /// The request a shortage takes when no page can relieve it: the lowest
    /// priority, then the latest admission, of the requests that resume from
    /// their generated tokens (today's `recompute_starved`).
    fn request_victim(&self) -> Option<Victim> {
        self.views
            .values()
            .filter(|view| view.state == RequestState::Resident && view.migratable)
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

/// Which work a step assembles, as today's batch-kind selection names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    /// Context ingestion only: encoders and prefill chunks.
    Prefill,
    /// Everything but context ingestion.
    Token,
    /// No filter: only media and background work is ready.
    Any,
}

impl<K: Order> Policy for Arrival<K> {
    fn init(&mut self, init: &Init) {
        self.parameters = init.parameters.clone();
        self.limits = Limits::from_parameters(&self.parameters);
        self.phases = Phases::new(&init.state.phases);
        for view in &init.state.requests {
            self.views.insert(view.id, view.clone());
        }
        // Residents start in admission order, which the snapshot states as the
        // time each was admitted.
        let mut residents: Vec<&RequestView> = init
            .state
            .requests
            .iter()
            .filter(|view| view.state == RequestState::Resident)
            .collect();
        residents.sort_by_key(|view| {
            (
                view.admitted_at.map(|at| at.0).unwrap_or_default(),
                view.id.0,
            )
        });
        self.admission = residents.into_iter().map(|view| view.id).collect();
        for view in &init.state.requests {
            if view.state == RequestState::Queued {
                self.waiting.push(view);
            }
        }
        // The cache orders the snapshot states, least recently used first.
        let mut pages: Vec<(u64, PageId, bool, bool)> = init
            .state
            .pages
            .iter()
            .map(|page| (page.last_used.0, page.id, page.on_host, page.evictable))
            .collect();
        pages.sort_unstable_by_key(|(used, id, _, _)| (*used, id.0));
        for (_, page, on_host, evictable) in pages {
            self.pages.touch(page);
            self.on_host.insert(page, on_host);
            self.evictable.insert(page, evictable);
        }
        let mut encodings: Vec<(u64, EncodingId)> = init
            .state
            .encodings
            .iter()
            .map(|entry| (entry.last_used.0, entry.id))
            .collect();
        encodings.sort_unstable_by_key(|(used, id)| (*used, id.0));
        for (_, entry) in encodings {
            self.encodings.touch(entry);
        }
    }

    fn observe(&mut self, events: &[Event]) {
        for event in events {
            match event {
                Event::Arrived { request } => {
                    self.views.insert(request.id, request.clone());
                    if request.state == RequestState::Queued {
                        self.waiting.push(request);
                    }
                }
                // The view of a resident request changed: its next phase,
                // prompt progress, cached prefix or pages.
                Event::Updated { request } => {
                    let arrived = request.id;
                    if request.state == RequestState::Resident
                        && !self.admission.contains(&arrived)
                        && !self.waiting.contains(arrived)
                    {
                        self.admission.push(arrived);
                    }
                    if request.state == RequestState::Queued {
                        self.waiting.push(request);
                    }
                    self.views.insert(arrived, request.clone());
                }
                Event::Admitted { request } => {
                    self.waiting.remove(*request);
                    if !self.admission.contains(request) {
                        self.admission.push(*request);
                    }
                    if let Some(view) = self.views.get_mut(request) {
                        view.state = RequestState::Resident;
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
                        view.last_emitted_at = Some(*at);
                    }
                }
                Event::Finished { request, .. } => {
                    self.waiting.remove(*request);
                    self.views.remove(request);
                    self.admission.retain(|resident| resident != request);
                }
                Event::Stepped { .. } => {}
                Event::Cached { page } => {
                    self.pages.touch(page.id);
                    self.on_host.insert(page.id, page.on_host);
                    self.evictable.insert(page.id, page.evictable);
                }
                Event::Hit { page } => self.pages.touch(*page),
                Event::Freed { page } => {
                    self.pages.remove(*page);
                    self.on_host.remove(page);
                    self.evictable.remove(page);
                }
                Event::Encoded { entry } => self.encodings.touch(entry.id),
                // An event kind from a newer mechanism.
                Event::Unknown => {}
            }
        }
    }

    fn dispatch(&mut self, room: &Room) -> Plan {
        // A request whose worst-case reservation is larger than the whole
        // device cache can never run: it is rejected rather than left to block
        // the queue (today's `admission.rs`, "request exceeds total KV
        // capacity").
        let reject: Vec<RequestId> = self
            .waiting
            .iter()
            .filter(|id| {
                self.views
                    .get(id)
                    .is_some_and(|view| view.reservation_units > room.capacity_units)
            })
            .collect();
        let mut plan = Plan {
            // Every other queued request, in order. The mechanism admits them
            // until the first it cannot place, which is the admission
            // decision.
            admit: self
                .waiting
                .iter()
                .filter(|id| !reject.contains(id))
                .collect(),
            reject,
            // The device pages no resident request continues from go to the
            // host KV cache, least recently used first, which is today's rule
            // for writing a prefix out.
            offload: self.offload_pages(),
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
            ..Plan::default()
        };
        // A draft cache with a gap is refilled, which is prefill work of the
        // request's next step (`PhaseKind::Refill`).
        plan.refill = self
            .admission
            .iter()
            .filter(|id| self.views.get(id).is_some_and(|view| self.refill_due(view)))
            .copied()
            .collect();
        let pass = self.pass(room);
        let shares = Shares::of(room);
        let mut step = Step::new(shares);
        let mut media_path = false;
        let mut other_path = false;
        for id in self.assembly_order() {
            if step.shares().full() {
                break;
            }
            let Some(view) = self.views.get(&id).cloned() else {
                continue;
            };
            let Some(phase) = view.phase else {
                continue;
            };
            let kind = self.phases.kind(phase);
            let class = class_of(kind);
            match pass {
                // A prefill step carries context ingestion only, and a token
                // step every other kind.
                Pass::Prefill if class != PhaseClass::Context => continue,
                Pass::Token if class == PhaseClass::Context => continue,
                Pass::Prefill | Pass::Token | Pass::Any => {}
            }
            let (tokens, writes_prefix) = self.work_tokens(&view, room, &step.shares());
            if tokens == 0 {
                continue;
            }
            if !step.add(id, phase, tokens, writes_prefix) {
                // The step lacks the room for this call: it is left to later
                // requests and later steps, as today's assembly does.
                continue;
            }
            if view.class == RequestClass::Media {
                media_path = true;
            } else {
                other_path = true;
            }
        }
        // Alternate the two paths' precedence, as today's `prefer_media`
        // does between a media pass and a generation pass.
        if media_path && other_path {
            self.media_first = !self.media_first;
        }
        plan.work = step.into_work();
        plan
    }

    fn reclaim(&mut self, shortage: &Shortage) -> Vec<Victim> {
        let mut victims = match shortage.kind {
            Storage::KvUnits => self.page_victims(false, shortage.need),
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
            Storage::Other => Vec::new(),
        };
        // Units are what the pages above free one of each; when the device
        // cache has no more page no request continues from, the request itself
        // is the victim, as today's recompute preemption has it.
        if matches!(shortage.kind, Storage::KvUnits)
            && (victims.len() as u64) < shortage.need
            && let Some(victim) = self.request_victim()
        {
            victims.push(victim);
        }
        victims
    }

    fn route(&mut self, request: &RequestView, ranks: &[RankView]) -> usize {
        // A request resuming a stored cache goes back to the rank recorded for
        // it; the mechanism skips this decision for it.
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
        // equal loads rotate as they do today.
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
        // A decode-stage call never lands on a replica that only prefills
        // (today's prefill groups).
        let decode_stage = matches!(
            call.kind,
            PhaseKind::Decode | PhaseKind::Verify | PhaseKind::CacheInstall
        );
        let mut candidates = replicas.iter().filter(|replica| {
            replica.kinds.contains(&call.kind) && !(decode_stage && replica.prefills_only)
        });
        // The replica that already holds the request's state keeps it: a
        // request's state is never migrated, and a full replica delays the call
        // rather than moving it.
        if let Some(held) = replicas.iter().find(|replica| replica.holds_request) {
            return held.index;
        }
        // Then a replica that already holds another component of this request,
        // so its state stays in one process and its products are not published
        // (today's `component_target`).
        if let Some(chosen) = candidates
            .clone()
            .filter(|replica| replica.holds_component && replica.capacity > 0)
            .min_by_key(|replica| (replica.resident_requests, replica.index))
        {
            return chosen.index;
        }
        // Otherwise the replica with the fewest resident requests that can take
        // one, ties to the earliest in the list.
        if let Some(chosen) = candidates
            .clone()
            .filter(|replica| replica.capacity > 0)
            .min_by_key(|replica| (replica.resident_requests, replica.index))
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

    fn warm(&mut self, room: &Room) -> Vec<PageId> {
        // The resumed requests' pages in the order the mechanism claimed the
        // requests, each request's pages in prefix order (today's
        // `warming.rs`), cut to the tokens a warming epoch may spend on new
        // prefix writes. A page holds `page_tokens` tokens, so the budget is
        // that many pages; a zero budget warms nothing.
        let Some(budget) = room.prefill_tokens else {
            // Outside a precopy the mechanism states no candidates, so this is
            // empty either way.
            return room
                .warming
                .iter()
                .flat_map(|warming| warming.pages.iter().copied())
                .collect();
        };
        let tokens_per_page = room.page_tokens.max(1);
        let mut pages = Vec::new();
        let mut tokens = 0u32;
        for warming in &room.warming {
            for page in &warming.pages {
                if tokens.saturating_add(tokens_per_page) > budget {
                    return pages;
                }
                tokens = tokens.saturating_add(tokens_per_page);
                pages.push(*page);
            }
        }
        pages
    }

    fn experts(&mut self, load: &ExpertLoad) -> ExpertOrder {
        // `fcfs` does not change the expert placement: the identity placement
        // it was launched with, expert `e` on rank `e % ranks`.
        let ranks = load.ranks.max(1);
        ExpertOrder {
            placement: (0..load.experts).map(|expert| expert % ranks).collect(),
        }
    }

    fn limits(&self) -> Limits {
        self.limits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::Warming;
    use crate::ids::{ComponentId, PhaseId};
    use crate::time::{Duration, Nanos};
    use crate::view::{EncodingView, PageView, PhaseView, RankView, ReplicaView, Snapshot};

    /// A policy with two phases: 0 prefill, 1 decode.
    fn policy() -> Arrival<crate::ArrivalOrder> {
        let mut fcfs = Arrival::default();
        fcfs.init(&Init {
            parameters: Parameters::default(),
            state: Snapshot {
                phases: vec![
                    PhaseView {
                        id: PhaseId(0),
                        kind: PhaseKind::Prefill,
                        block: ComponentId(0),
                    },
                    PhaseView {
                        id: PhaseId(1),
                        kind: PhaseKind::Decode,
                        block: ComponentId(0),
                    },
                ],
                ..Snapshot::default()
            },
        });
        fcfs
    }

    fn view(id: u64, priority: i32, arrived: u64) -> RequestView {
        RequestView {
            id: RequestId(id),
            state: RequestState::Queued,
            phase: None,
            priority,
            arrived_at: Nanos(arrived),
            admitted_at: None,
            prompt_tokens: 100,
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
        policy: &mut Arrival<crate::ArrivalOrder>,
        mut request: RequestView,
        phase: PhaseId,
    ) -> RequestId {
        request.state = RequestState::Resident;
        request.phase = Some(phase);
        request.admitted_at = Some(Nanos(1));
        let id = request.id;
        policy.observe(&[Event::Admitted { request: id }]);
        policy.observe(&[Event::Updated { request }]);
        id
    }

    fn admit(policy: &mut Arrival<crate::ArrivalOrder>, mut request: RequestView) -> RequestId {
        request.state = RequestState::Resident;
        request.phase = Some(PhaseId(0));
        request.admitted_at = Some(Nanos(1));
        let id = request.id;
        policy.observe(&[Event::Admitted { request: id }]);
        policy.observe(&[Event::Updated { request }]);
        id
    }

    #[test]
    fn the_queue_decides_the_admission_order() {
        let mut fcfs = policy();
        fcfs.observe(&[
            Event::Arrived {
                request: view(2, 0, 20),
            },
            Event::Arrived {
                request: view(1, 0, 10),
            },
        ]);
        let plan = fcfs.dispatch(&room(1024, 8));
        let order: Vec<u64> = plan.admit.iter().map(|id| id.0).collect();
        assert_eq!(order, vec![1, 2]);
        assert!(plan.work.is_empty());
    }

    #[test]
    fn a_prefill_chunk_is_clipped_and_the_prompt_advances() {
        let mut fcfs = policy();
        let mut request = view(1, 0, 0);
        request.prompt_tokens = 100;
        let _ = admit(&mut fcfs, request);
        let plan = fcfs.dispatch(&room(1024, 8));
        assert_eq!(plan.work.len(), 1);
        assert_eq!(plan.work[0].tokens, 100);

        fcfs.observe(&[Event::Updated {
            request: RequestView {
                prefilled_tokens: 32,
                ..fcfs.views[&RequestId(1)].clone()
            },
        }]);
        let plan = fcfs.dispatch(&room(1024, 8));
        assert_eq!(plan.work[0].tokens, 64);
    }

    #[test]
    fn a_decode_step_skips_context_work() {
        let mut fcfs = policy();
        let mut first = view(1, 0, 0);
        first.phase = Some(PhaseId(0));
        let _ = admit(&mut fcfs, first);
        let mut decode = view(2, 0, 1);
        decode.phase = Some(PhaseId(1));
        decode.prompt_tokens = 10;
        let _ = admit(&mut fcfs, decode);
        // One batch with prefill work is in flight and another is allowed, so
        // this step still prefills.
        let plan = fcfs.dispatch(&room(1024, 8));
        assert_eq!(plan.work[0].request, RequestId(1));
        // Two batches with prefill work in flight close the window.
        let mut closed = room(1024, 8);
        closed.inflight.prefill_batches = 2;
        let plan = fcfs.dispatch(&closed);
        assert_eq!(plan.work[0].request, RequestId(2));
        assert_eq!(plan.work[0].tokens, 1);
    }

    #[test]
    fn a_step_stops_when_its_token_budget_is_spent() {
        let mut fcfs = policy();
        for id in 1..=4 {
            let mut request = view(id, 0, id);
            request.prompt_tokens = 100;
            let _ = admit(&mut fcfs, request);
        }
        let mut shares = room(80, 8);
        shares.page_tokens = 16;
        let plan = fcfs.dispatch(&shares);
        let charged: u32 = plan.work.iter().map(|work| work.tokens).sum();
        assert!(charged <= 80);
        assert!(plan.work.len() < 4);
    }

    #[test]
    fn a_shortage_takes_pages_least_recently_used_first() {
        let mut fcfs = policy();
        fcfs.observe(&[
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
        let victims = fcfs.reclaim(&Shortage {
            kind: Storage::KvUnits,
            need: 1,
        });
        assert_eq!(victims, vec![Victim::Page(PageId(2))]);
    }

    #[test]
    fn a_host_shortage_takes_host_pages_only() {
        let mut fcfs = policy();
        fcfs.observe(&[
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
        let victims = fcfs.reclaim(&Shortage {
            kind: Storage::HostPages,
            need: 2,
        });
        assert_eq!(victims, vec![Victim::Page(PageId(1))]);
    }

    #[test]
    fn an_encoder_shortage_takes_the_least_recently_used_entry() {
        let mut fcfs = policy();
        fcfs.observe(&[
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
        let victims = fcfs.reclaim(&Shortage {
            kind: Storage::EncoderEntries,
            need: 1,
        });
        assert_eq!(victims, vec![Victim::Encoding(EncodingId(5))]);
    }

    #[test]
    fn a_request_victim_is_the_lowest_priority_admitted_last() {
        let mut fcfs = policy();
        let mut first = view(1, 0, 0);
        first.admitted_at = Some(Nanos(10));
        admit(&mut fcfs, first);
        let mut second = view(2, 5, 1);
        second.admitted_at = Some(Nanos(20));
        admit(&mut fcfs, second);
        let mut third = view(3, 0, 2);
        third.admitted_at = Some(Nanos(30));
        admit(&mut fcfs, third);
        let victims = fcfs.reclaim(&Shortage {
            kind: Storage::KvUnits,
            need: 1,
        });
        assert_eq!(victims, vec![Victim::Request(RequestId(2))]);
    }

    #[test]
    fn routing_takes_the_least_loaded_live_rank_from_the_cursor() {
        let mut fcfs = policy();
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
        assert_eq!(fcfs.route(&view(1, 0, 0), &ranks), 1);
        // The cursor moved past rank 1, so the next equal load behind it wins.
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
        assert_eq!(fcfs.route(&view(2, 0, 0), &ranks), 0);
    }

    #[test]
    fn a_resuming_request_keeps_its_recorded_rank() {
        let mut fcfs = policy();
        let mut request = view(1, 0, 0);
        request.cache_rank = Some(3);
        let ranks = [RankView {
            index: 0,
            in_flight: 0,
            alive: true,
        }];
        assert_eq!(fcfs.route(&request, &ranks), 3);
    }

    #[test]
    fn a_held_replica_keeps_the_request_and_a_decode_avoids_prefill_only() {
        let mut fcfs = policy();
        let call = CallView {
            request: RequestId(1),
            phase: PhaseId(1),
            kind: PhaseKind::Decode,
        };
        let replicas = [
            ReplicaView {
                index: 0,
                block: ComponentId(0),
                kinds: vec![PhaseKind::Decode],
                resident_requests: 0,
                capacity: 4,
                holds_request: false,
                prefills_only: true,
                holds_component: false,
            },
            ReplicaView {
                index: 1,
                block: ComponentId(0),
                kinds: vec![PhaseKind::Decode],
                resident_requests: 3,
                capacity: 4,
                holds_request: false,
                prefills_only: false,
                holds_component: false,
            },
        ];
        assert_eq!(fcfs.place(&call, &replicas), 1);
        let held = ReplicaView {
            index: 0,
            holds_request: true,
            prefills_only: false,
            ..replicas[0].clone()
        };
        let replicas = [held, replicas[1].clone()];
        assert_eq!(fcfs.place(&call, &replicas), 0);
    }

    #[test]
    fn the_expert_placement_is_unchanged() {
        let mut fcfs = policy();
        let order = fcfs.experts(&ExpertLoad {
            ranks: 2,
            experts: 5,
            tokens: vec![0; 5],
        });
        assert_eq!(order.placement, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn priority_admits_and_assembles_by_priority_then_arrival() {
        let mut priority = Arrival::<crate::PriorityOrder>::default();
        priority.init(&Init {
            parameters: Parameters::default(),
            state: Snapshot {
                phases: vec![
                    PhaseView {
                        id: PhaseId(0),
                        kind: PhaseKind::Prefill,
                        block: ComponentId(0),
                    },
                    PhaseView {
                        id: PhaseId(1),
                        kind: PhaseKind::Decode,
                        block: ComponentId(0),
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
        let plan = priority.dispatch(&room(1024, 8));
        let order: Vec<u64> = plan.admit.iter().map(|id| id.0).collect();
        assert_eq!(order, vec![2, 1]);

        // Admitted in that order, they are assembled in it too.
        for id in [2u64, 1] {
            let mut request = view(id, if id == 2 { 0 } else { 5 }, id * 10);
            request.state = RequestState::Resident;
            request.phase = Some(PhaseId(0));
            request.admitted_at = Some(Nanos(id));
            priority.observe(&[Event::Admitted {
                request: RequestId(id),
            }]);
            priority.observe(&[Event::Updated { request }]);
        }
        let plan = priority.dispatch(&room(1024, 8));
        let order: Vec<u64> = plan.work.iter().map(|work| work.request.0).collect();
        assert_eq!(order, vec![2, 1]);
    }

    #[test]
    fn a_refill_is_prefill_work() {
        let mut fcfs = Arrival::<crate::ArrivalOrder>::default();
        fcfs.init(&Init {
            parameters: Parameters::default(),
            state: Snapshot {
                phases: vec![
                    PhaseView {
                        id: PhaseId(0),
                        kind: PhaseKind::Refill,
                        block: ComponentId(0),
                    },
                    PhaseView {
                        id: PhaseId(1),
                        kind: PhaseKind::Decode,
                        block: ComponentId(0),
                    },
                ],
                ..Snapshot::default()
            },
        });
        let mut refilling = view(1, 0, 0);
        refilling.phase = Some(PhaseId(0));
        refilling.prefilled_tokens = 16;
        refilling.refill_tokens = Some(4);
        let _ = admit_with_phase(&mut fcfs, refilling, PhaseId(0));
        let mut decoding = view(2, 0, 1);
        decoding.phase = Some(PhaseId(1));
        let _ = admit_with_phase(&mut fcfs, decoding, PhaseId(1));
        // The refill is prefill work, so it is planned in a prefill step and
        // its call draws on the prefill budget.
        let plan = fcfs.dispatch(&room(1024, 8));
        assert_eq!(plan.refill, vec![RequestId(1)]);
        assert_eq!(
            plan.work.first().map(|work| (work.request, work.tokens)),
            Some((RequestId(1), 4))
        );
    }

    #[test]
    fn a_full_step_decides_every_entry_of_its_plan() {
        let mut fcfs = Arrival::<crate::ArrivalOrder>::default();
        // Two queued: the first fits the cache, the second never can.
        let mut fits = view(10, 0, 1);
        fits.reservation_units = 10;
        let mut enormous = view(11, 0, 2);
        enormous.reservation_units = 1_000;
        // A resident prefill, a resident decode with a draft gap, and a
        // resident media request whose host hit deepens.
        let mut prefill = view(20, 0, 3);
        prefill.state = RequestState::Resident;
        prefill.phase = Some(PhaseId(0));
        prefill.admitted_at = Some(Nanos(1));
        let mut decode = view(21, 0, 4);
        decode.state = RequestState::Resident;
        decode.phase = Some(PhaseId(1));
        decode.admitted_at = Some(Nanos(2));
        decode.refill_tokens = Some(8);
        let mut media = view(22, 0, 5);
        media.state = RequestState::Resident;
        media.phase = Some(PhaseId(2));
        media.class = RequestClass::Media;
        media.admitted_at = Some(Nanos(3));
        media.prefetch_tokens = 300;
        fcfs.init(&Init {
            parameters: Parameters::default(),
            state: Snapshot {
                requests: vec![fits.clone(), enormous, prefill, decode, media],
                phases: vec![
                    PhaseView {
                        id: PhaseId(0),
                        kind: PhaseKind::Prefill,
                        block: ComponentId(0),
                    },
                    PhaseView {
                        id: PhaseId(1),
                        kind: PhaseKind::Decode,
                        block: ComponentId(0),
                    },
                    PhaseView {
                        id: PhaseId(2),
                        kind: PhaseKind::Denoise,
                        block: ComponentId(1),
                    },
                ],
                pages: vec![
                    PageView {
                        id: PageId(1),
                        tokens: 16,
                        last_used: Nanos(1),
                        on_host: false,
                        evictable: true,
                    },
                    PageView {
                        id: PageId(2),
                        tokens: 16,
                        last_used: Nanos(2),
                        on_host: false,
                        evictable: true,
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
        let room = Room {
            token_budget: 1024,
            call_budget: 8,
            page_tokens: 16,
            prefill_tokens: Some(64),
            capacity_units: 100,
            warming: vec![Warming {
                request: RequestId(20),
                pages: vec![PageId(7), PageId(8)],
            }],
            ..Room::default()
        };
        // With the prefill window open, the step carries context work only:
        // today a prefill pass schedules no token progress either.
        let plan = fcfs.dispatch(&room);
        let order: Vec<u64> = plan.work.iter().map(|work| work.request.0).collect();
        assert_eq!(order, vec![20]);
        assert_eq!(plan.work[0].tokens, 100);
        // The over-capacity request is rejected and not also admitted.
        assert_eq!(plan.reject, vec![RequestId(11)]);
        assert_eq!(plan.admit, vec![RequestId(10)]);
        // The device pages no request continues from go to the host, in
        // least-recently-used order; the host page stays.
        assert_eq!(plan.offload, vec![PageId(1), PageId(2)]);
        assert_eq!(plan.prefetch, vec![RequestId(22)]);
        assert_eq!(plan.refill, vec![RequestId(21)]);
        assert_eq!(fcfs.warm(&room), vec![PageId(7), PageId(8)]);

        // Two prefill batches in flight close the window: the step carries
        // token progress and media work, in class order within the generation
        // path, and the media path second.
        let mut closed = room.clone();
        closed.inflight.prefill_batches = PREFILL_WINDOW_CREDITS;
        let plan = fcfs.dispatch(&closed);
        let order: Vec<u64> = plan.work.iter().map(|work| work.request.0).collect();
        assert_eq!(order, vec![21, 22]);
        assert!(plan.work.iter().all(|work| work.tokens == 1));
    }

    #[test]
    fn steps_and_unknown_events_do_not_disturb_the_state() {
        let mut fcfs = policy();
        fcfs.observe(&[
            Event::Stepped {
                step: 7,
                duration: Duration(10),
            },
            Event::Unknown,
        ]);
        assert!(fcfs.dispatch(&room(1024, 8)).work.is_empty());
    }
}
