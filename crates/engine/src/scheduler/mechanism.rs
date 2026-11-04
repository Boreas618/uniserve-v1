//! The mechanism's side of the policy boundary (design §4.4, item 3).
//!
//! The mechanism owns the request state, the storage, the hard limits and the
//! execution; a policy decides. This module holds the three parts of the
//! mechanism that exist only because decisions come from outside it:
//!
//! - [`Decisions`], the trait the mechanism asks through. It is the mechanism's
//!   whole view of the policy: one implementation (`scheduler/policy.rs`) wraps
//!   the live policy instance, its previous module and the switch.
//! - The validators ([`validate_plan`], [`validate_victims`],
//!   [`validate_route`], [`validate_place`], [`validate_warm`],
//!   [`validate_experts`]): every answer is checked against the live state and
//!   the step's hard limits before it is executed. A decision that names
//!   something that does not exist, does not fit or cannot be chosen is the
//!   policy's failure (§4.4), never the mechanism's.
//! - The [`Recorder`] and the [`Watchdog`]: the state changes the policy is
//!   told about, and the resident request that did not progress within
//!   `limits().timeout`, which is the other way a policy fails.
//!
//! The validators are pure functions over the interface's own types, so the
//! switch's trial evaluation (§7.3, `policy.rs`) calls the same ones the live
//! path does: a policy that passes probation cannot later be failed by a rule
//! it never met.

use std::collections::{BTreeMap, BTreeSet};

use scheduling::PolicyError;

use crate::scheduler::policy::{PolicyReverted, PolicySlot, TrialState, Validators};
use crate::scheduler::views::{Recorded, RecordedTable};
use scheduling::guest::decision::{Plan, Room, Shortage, Storage, Victim, Work};
use scheduling::guest::event::Event;
use scheduling::guest::wire::Call;
use scheduling::guest::{
    CallView, EncodingId, EncodingView, ExpertLoad, ExpertOrder, Limits, Nanos, PageId, PageView,
    PhaseId, PhaseKind, PhaseView, RankView, ReplicaView, RequestId, RequestState, RequestView,
    Snapshot,
};

/// The mechanism's whole view of the policy: exactly what `policy.rs`'s
/// [`PolicySlot`] offers, so nothing in the mechanism reaches past it.
///
/// Routing is not here. It runs on the control thread's own instance
/// (`policy.rs`'s `RoutePolicy`), which no scheduler thread holds.
///
/// `fail` and `boundary` are the two hooks through which the switch reaches the
/// policy (§7.3): a failure is reported once, with the state at the time of the
/// failure, and the boundary is the only moment a policy is exchanged.
pub(crate) trait Decisions {
    /// The limits the policy declared at instantiation.
    fn limits(&self) -> Limits;

    /// Once per step: every event since the previous step, in order.
    fn observe(&mut self, events: &[Event]) -> Result<(), PolicyError>;

    /// Once per step: what this step does.
    fn dispatch(&mut self, room: &Room) -> Result<Plan, PolicyError>;

    /// When work does not fit the free storage: which objects to free, in order.
    fn reclaim(&mut self, shortage: &Shortage) -> Result<Vec<Victim>, PolicyError>;

    /// When a request is first called on a block: which replica runs it.
    fn place(&mut self, call: &CallView, replicas: &[ReplicaView]) -> Result<usize, PolicyError>;

    /// During an epoch's precopy: which pages to warm, in order.
    fn warm(&mut self, room: &Room) -> Result<Vec<PageId>, PolicyError>;

    /// Periodically: the placement of experts on ranks.
    fn experts(&mut self, load: &ExpertLoad) -> Result<ExpertOrder, PolicyError>;

    /// The policy failed: serve on under the previous policy and report it, or
    /// fail the epoch when no previous policy is kept (§7.3).
    ///
    /// `state` is the mechanism's state at the moment of the failure: a revert
    /// instantiates `previous` on it, so a policy that fails mid-epoch is
    /// replaced by one that sees what actually exists.
    fn fail(
        &mut self,
        error: &PolicyError,
        state: &TrialState,
    ) -> Result<PolicyReverted, PolicyError>;

    /// The batch boundary: the stage a daemon asked for, applied where no batch
    /// and no submission is in flight (§7.3 steps 3 to 6).
    fn boundary(&mut self, state: &TrialState) -> Result<Option<PolicyReverted>, PolicyError>;
}

impl Decisions for PolicySlot {
    fn limits(&self) -> Limits {
        PolicySlot::limits(self)
    }

    fn observe(&mut self, events: &[Event]) -> Result<(), PolicyError> {
        PolicySlot::observe(self, events)
    }

    fn dispatch(&mut self, room: &Room) -> Result<Plan, PolicyError> {
        PolicySlot::dispatch(self, room)
    }

    fn reclaim(&mut self, shortage: &Shortage) -> Result<Vec<Victim>, PolicyError> {
        PolicySlot::reclaim(self, shortage)
    }

    fn place(&mut self, call: &CallView, replicas: &[ReplicaView]) -> Result<usize, PolicyError> {
        PolicySlot::place(self, call, replicas)
    }

    fn warm(&mut self, room: &Room) -> Result<Vec<PageId>, PolicyError> {
        PolicySlot::warm(self, room)
    }

    fn experts(&mut self, load: &ExpertLoad) -> Result<ExpertOrder, PolicyError> {
        PolicySlot::experts(self, load)
    }

    fn fail(
        &mut self,
        error: &PolicyError,
        state: &TrialState,
    ) -> Result<PolicyReverted, PolicyError> {
        PolicySlot::fail(self, error, state)
    }

    fn boundary(&mut self, state: &TrialState) -> Result<Option<PolicyReverted>, PolicyError> {
        PolicySlot::boundary(self, state)
    }
}

/// The mechanism's validators, against the state at one moment (`policy.rs`'s
/// [`Validators`]).
///
/// One type, so the switch's precopy runs exactly the functions the live path
/// runs: a candidate that passes probation is judged by what will judge it once
/// it serves.
pub(crate) struct Checked<'a> {
    live: &'a Live,
    room: &'a Room,
}

impl<'a> Checked<'a> {
    pub(crate) fn new(live: &'a Live, room: &'a Room) -> Self {
        Self { live, room }
    }
}

impl Validators for Checked<'_> {
    fn plan(&self, plan: &Plan) -> Result<(), PolicyError> {
        validate_plan(plan, self.room, self.live)
    }

    fn victims(&self, shortage: &Shortage, victims: &[Victim]) -> Result<(), PolicyError> {
        validate_victims(shortage, victims, self.live)
    }

    fn route(
        &self,
        request: &RequestView,
        ranks: &[RankView],
        index: usize,
    ) -> Result<(), PolicyError> {
        validate_route(request, ranks, index)
    }

    fn place(
        &self,
        call: &CallView,
        replicas: &[ReplicaView],
        index: usize,
    ) -> Result<(), PolicyError> {
        validate_place(call, replicas, index)
    }

    fn warm(&self, pages: &[PageId]) -> Result<(), PolicyError> {
        validate_warm(self.room, pages)
    }

    fn experts(&self, load: &ExpertLoad, order: &ExpertOrder) -> Result<(), PolicyError> {
        validate_experts(load, order)
    }
}

/// The mechanism's objects, as the validators read them.
///
/// `scheduler/views.rs` builds this from the scheduler's own state each step
/// (and the switch's trial evaluation at a batch boundary). The one thing no
/// view carries is the waiting queue's order, which is the mechanism's own
/// admission order, so it is here beside the requests.
#[derive(Clone, Debug, Default)]
pub(crate) struct Live {
    /// Every request the main process holds, by id.
    pub requests: BTreeMap<RequestId, RequestView>,
    /// The requests waiting for admission, in the mechanism's own order.
    ///
    /// Only membership is read: the order a plan admits in is the policy's, and
    /// the mechanism admits in the plan's order.
    pub queue: Vec<RequestId>,
    /// Every device and host page, by id.
    pub pages: BTreeMap<PageId, PageView>,
    /// Every encoder cache entry, by id.
    pub encodings: BTreeMap<EncodingId, EncodingView>,
    /// Every phase of this epoch, by id.
    pub phases: BTreeMap<PhaseId, PhaseView>,
}

impl Live {
    /// The state a policy is instantiated with, as the validators read it.
    pub(crate) fn of(snapshot: &Snapshot, queue: Vec<RequestId>) -> Self {
        let mut live = Live {
            queue,
            ..Live::default()
        };
        for request in &snapshot.requests {
            live.requests.insert(request.id, request.clone());
        }
        for page in &snapshot.pages {
            live.pages.insert(page.id, page.clone());
        }
        for entry in &snapshot.encodings {
            live.encodings.insert(entry.id, entry.clone());
        }
        for phase in &snapshot.phases {
            live.phases.insert(phase.id, phase.clone());
        }
        live
    }

    fn request(&self, id: RequestId) -> Result<&RequestView, PolicyError> {
        self.requests
            .get(&id)
            .ok_or_else(|| invalid(Call::Dispatch, format!("request {} does not exist", id.0)))
    }

    fn page(&self, id: PageId) -> Result<&PageView, PolicyError> {
        self.pages
            .get(&id)
            .ok_or_else(|| invalid(Call::Dispatch, format!("page {} does not exist", id.0)))
    }

    /// Whether `a` waits before `b` in the admission queue.
    fn queued(&self, id: RequestId) -> bool {
        self.queue.contains(&id)
    }
}

fn invalid(call: Call, message: impl Into<String>) -> PolicyError {
    PolicyError::invalid(call, message)
}

/// Whether a phase kind is a decode-stage call, which never lands on a replica
/// that runs prefill only (today's `placement.rs`, prefill-only groups).
fn decodes(kind: PhaseKind) -> bool {
    matches!(
        kind,
        PhaseKind::Decode | PhaseKind::Verify | PhaseKind::TokenDenoise | PhaseKind::CacheInstall
    )
}

/// Checks one step's plan against the live state and the room it was offered
/// (design §4.4).
///
/// A plan is valid when it names only live objects and fits what the mechanism
/// offered: the step's token and call budgets, the prefill budget while an
/// epoch warms, the graph tiers, and the phase each request can run next.
///
/// One request has at most one piece of work in a step: the mechanism plans one
/// call per request per step, because a request's next call depends on the
/// previous one's result. It is a mechanism limit, not a policy's choice, so a
/// plan that names a request twice is rejected here. A plan that fits is still executed
/// conservatively: work that does not fit the free storage after `reclaim`
/// waits for the next step, which is the mechanism's hard limit and not a
/// policy failure.
pub(crate) fn validate_plan(plan: &Plan, room: &Room, live: &Live) -> Result<(), PolicyError> {
    validate_admission(plan, live)?;
    validate_work(plan, room, live)?;
    validate_release(plan, live)?;
    Ok(())
}

/// Admission names queued requests, and never the same one twice or both
/// admitted and rejected.
///
/// The order is the policy's: the mechanism admits in the plan's order and
/// stops at the first request its hard limits refuse, which is how `fcfs`
/// reproduces first-come-first-served while a policy that reorders the queue
/// gets its own order. Membership is all that is validated here.
fn validate_admission(plan: &Plan, live: &Live) -> Result<(), PolicyError> {
    let mut admitted = BTreeSet::new();
    for id in &plan.admit {
        if !live.queued(*id) {
            return Err(invalid(
                Call::Dispatch,
                format!("admit names {}, which is not queued", id.0),
            ));
        }
        if !admitted.insert(*id) {
            return Err(invalid(
                Call::Dispatch,
                format!("admit names {} twice", id.0),
            ));
        }
    }

    let mut rejected = BTreeSet::new();
    for id in &plan.reject {
        if !live.queued(*id) {
            return Err(invalid(
                Call::Dispatch,
                format!("reject names {}, which is not queued", id.0),
            ));
        }
        if admitted.contains(id) {
            return Err(invalid(
                Call::Dispatch,
                format!("{} is both admitted and rejected", id.0),
            ));
        }
        if !rejected.insert(*id) {
            return Err(invalid(
                Call::Dispatch,
                format!("reject names {} twice", id.0),
            ));
        }
    }
    Ok(())
}

fn validate_work(plan: &Plan, room: &Room, live: &Live) -> Result<(), PolicyError> {
    if plan.work.len() > room.call_budget {
        return Err(invalid(
            Call::Dispatch,
            format!(
                "the plan asks for {} calls, the step offers {}",
                plan.work.len(),
                room.call_budget
            ),
        ));
    }

    let mut seen = BTreeSet::new();
    let mut tokens = 0u64;
    let mut prefill_tokens = 0u64;
    // The tokens of each (component, kind) group: a batch is one numerical call
    // on one component, so the graph tier bounds the group, not the step.
    let mut groups: Vec<((u32, PhaseKind), u64)> = Vec::new();

    for work in &plan.work {
        validate_work_entry(work, live)?;
        if !seen.insert(work.request) {
            return Err(invalid(
                Call::Dispatch,
                format!("{} is given two pieces of work in one step", work.request.0),
            ));
        }
        let phase = live.phases.get(&work.phase).ok_or_else(|| {
            invalid(
                Call::Dispatch,
                format!("phase {} is not one of this epoch's", work.phase.0),
            )
        })?;
        let charge = u64::from(work.tokens);
        tokens += charge;
        if phase.kind.writes_prefix_cache() {
            prefill_tokens += charge;
        }
        let key = (phase.block.0, phase.kind);
        match groups.iter_mut().find(|(group, _)| *group == key) {
            Some((_, tokens)) => *tokens += charge,
            None => groups.push((key, charge)),
        }
    }

    if tokens > u64::from(room.token_budget) {
        return Err(invalid(
            Call::Dispatch,
            format!(
                "the plan charges {tokens} tokens, the step offers {}",
                room.token_budget
            ),
        ));
    }
    if let Some(budget) = room.prefill_tokens {
        if prefill_tokens > u64::from(budget) {
            return Err(invalid(
                Call::Dispatch,
                format!(
                    "the plan's prefill work charges {prefill_tokens} tokens, the epoch may still write {budget}"
                ),
            ));
        }
    }
    if let Some(tier) = room.graph_tiers.iter().copied().max() {
        for ((_, kind), tokens) in &groups {
            if *tokens > u64::from(tier) {
                return Err(invalid(
                    Call::Dispatch,
                    format!(
                        "a {kind:?} batch charges {tokens} tokens, the largest captured graph holds {tier}"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_work_entry(work: &Work, live: &Live) -> Result<(), PolicyError> {
    let request = live.request(work.request)?;
    if request.state != RequestState::Resident {
        return Err(invalid(
            Call::Dispatch,
            format!("{} is not resident", work.request.0),
        ));
    }
    if request.phase != Some(work.phase) {
        return Err(invalid(
            Call::Dispatch,
            format!(
                "{} is given phase {}, but its next phase is {}",
                work.request.0,
                work.phase.0,
                request
                    .phase
                    .map(|phase| phase.0.to_string())
                    .unwrap_or_else(|| "none".to_owned())
            ),
        ));
    }
    if work.tokens == 0 {
        return Err(invalid(
            Call::Dispatch,
            format!("{} is given no tokens to run", work.request.0),
        ));
    }
    let Some(phase) = live.phases.get(&work.phase) else {
        return Err(invalid(
            Call::Dispatch,
            format!("phase {} is not one of this epoch's", work.phase.0),
        ));
    };
    match phase.kind {
        PhaseKind::Refill => {
            return Err(invalid(
                Call::Dispatch,
                format!(
                    "{} is given refill work: a refill is named by `plan.refill`, not by `work`",
                    work.request.0
                ),
            ));
        }
        PhaseKind::Prefill => {
            let remaining = request
                .prompt_tokens
                .saturating_sub(request.prefilled_tokens);
            if work.tokens > remaining {
                return Err(invalid(
                    Call::Dispatch,
                    format!(
                        "{} is given a {}-token prefill chunk with {remaining} prompt tokens left",
                        work.request.0, work.tokens
                    ),
                ));
            }
        }
        PhaseKind::Refill => {
            let gap = request.refill_tokens.unwrap_or(0);
            if work.tokens > gap {
                return Err(invalid(
                    Call::Dispatch,
                    format!(
                        "{} is given a {}-token refill chunk, its draft cache is {} tokens short",
                        work.request.0, work.tokens, gap
                    ),
                ));
            }
        }
        _ => {
            let charge = request.next_tokens.max(1);
            if work.tokens != charge {
                return Err(invalid(
                    Call::Dispatch,
                    format!(
                        "{}'s {:?} call is charged {charge} tokens by the mechanism, the plan says {}",
                        work.request.0, phase.kind, work.tokens
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_release(plan: &Plan, live: &Live) -> Result<(), PolicyError> {
    let mut pages = BTreeSet::new();
    for id in &plan.offload {
        let page = live.page(*id)?;
        if page.on_host {
            return Err(invalid(
                Call::Dispatch,
                format!("page {} is already on the host", id.0),
            ));
        }
        if !page.evictable {
            return Err(invalid(
                Call::Dispatch,
                format!(
                    "page {} is still continued from by a resident request",
                    id.0
                ),
            ));
        }
        if !pages.insert(*id) {
            return Err(invalid(
                Call::Dispatch,
                format!("page {} is offloaded twice", id.0),
            ));
        }
    }

    let mut prefetched = BTreeSet::new();
    for id in &plan.prefetch {
        let request = live.request(*id)?;
        if request.state != RequestState::Resident {
            return Err(invalid(
                Call::Dispatch,
                format!("prefetch names {}, which is not resident", id.0),
            ));
        }
        if request.prefetch_tokens == 0 {
            return Err(invalid(
                Call::Dispatch,
                format!("{} has no host hit left to deepen", id.0),
            ));
        }
        if !prefetched.insert(*id) {
            return Err(invalid(
                Call::Dispatch,
                format!("{} is prefetched twice", id.0),
            ));
        }
    }

    let mut refilled = BTreeSet::new();
    for id in &plan.refill {
        let request = live.request(*id)?;
        if request.state != RequestState::Resident {
            return Err(invalid(
                Call::Dispatch,
                format!("refill names {}, which is not resident", id.0),
            ));
        }
        if request.refill_tokens.unwrap_or(0) == 0 {
            return Err(invalid(
                Call::Dispatch,
                format!("{}'s draft cache has no gap to refill", id.0),
            ));
        }
        if !refilled.insert(*id) {
            return Err(invalid(
                Call::Dispatch,
                format!("{} is refilled twice", id.0),
            ));
        }
    }
    Ok(())
}

/// Checks the objects a policy would free for a shortage (design §4.4).
///
/// The legal answers are these, and only these:
///
/// | shortage           | victims                                   |
/// |--------------------|-------------------------------------------|
/// | `KvUnits`          | a free device page, or a migratable request |
/// | `HostPages`        | a host page                                |
/// | `EncoderEntries`   | an encoder cache entry                     |
///
/// A page may be freed only when it is live and no resident request continues
/// from it (`evictable`); a request may be ended only when it is resident and
/// migratable, because that is the only way the mechanism can end one without
/// losing the tokens it generated (the frontend resubmits it). An empty answer
/// is legal: a policy may leave the shortfall to natural completion.
pub(crate) fn validate_victims(
    shortage: &Shortage,
    victims: &[Victim],
    live: &Live,
) -> Result<(), PolicyError> {
    // Freeing nothing is a legal answer: the work that did not fit waits for
    // the next step, and the watchdog bounds a resident request that then never
    // advances (§4.4).
    let mut seen: Vec<Victim> = Vec::new();
    for victim in victims {
        if seen.contains(victim) {
            return Err(invalid(Call::Reclaim, format!("{victim:?} is freed twice")));
        }
        seen.push(*victim);
        match (shortage.kind, victim) {
            (Storage::KvUnits, Victim::Page(page)) => {
                let page = live.page(*page)?;
                if page.on_host {
                    return Err(invalid(
                        Call::Reclaim,
                        format!(
                            "page {} is on the host, it frees no device units",
                            page.id.0
                        ),
                    ));
                }
                if !page.evictable {
                    return Err(invalid(
                        Call::Reclaim,
                        format!(
                            "page {} is still continued from and cannot be freed",
                            page.id.0
                        ),
                    ));
                }
            }
            (Storage::KvUnits, Victim::Request(request)) => {
                let request = live.request(*request)?;
                if request.state != RequestState::Resident || !request.migratable {
                    return Err(invalid(
                        Call::Reclaim,
                        format!(
                            "{} cannot be ended as migratable: it is {} and {} migratable",
                            request.id.0,
                            if request.state == RequestState::Resident {
                                "resident"
                            } else {
                                "not resident"
                            },
                            if request.migratable { "" } else { "not" }
                        ),
                    ));
                }
            }
            (Storage::HostPages, Victim::Page(page)) => {
                let page = live.page(*page)?;
                if !page.on_host {
                    return Err(invalid(
                        Call::Reclaim,
                        format!("page {} is not on the host", page.id.0),
                    ));
                }
            }
            (Storage::EncoderEntries, Victim::Encoding(entry)) => {
                if !live.encodings.contains_key(entry) {
                    return Err(invalid(
                        Call::Reclaim,
                        format!("encoder entry {} does not exist", entry.0),
                    ));
                }
            }
            (Storage::Other, _) => {}
            (kind, victim) => {
                return Err(invalid(
                    Call::Reclaim,
                    format!("{victim:?} frees nothing of a {kind:?} shortage"),
                ));
            }
        }
    }
    Ok(())
}

/// Checks a routing decision: the rank exists, is alive, and is the scheduler
/// index the mechanism routes to.
///
/// `ranks` is in scheduler-index order, so the rotating choice of today's
/// `Routing::pick` is expressible. A request resuming a stored cache is never
/// routed: the mechanism sends it to the rank recorded for it.
pub(crate) fn validate_route(
    request: &RequestView,
    ranks: &[RankView],
    chosen: usize,
) -> Result<(), PolicyError> {
    if ranks.is_empty() {
        return Err(invalid(
            Call::Route,
            format!("request {} was routed with no rank to choose", request.id.0),
        ));
    }
    match ranks.iter().find(|rank| rank.index == chosen) {
        Some(rank) if rank.alive => Ok(()),
        Some(_) => Err(invalid(Call::Route, format!("rank {chosen} is not alive"))),
        None => Err(invalid(
            Call::Route,
            format!(
                "rank {chosen} is not one of the {} ranks offered",
                ranks.len()
            ),
        )),
    }
}

/// Checks a placement decision against the replicas the mechanism offered.
///
/// The replica must be in the list, run the call's kind, and not run prefill
/// only when the call is a decode-stage one (today's `component_target` rules
/// are the policy's; these are the mechanism's hard constraints). Capacity is a
/// preference the policy weighs, not a hard limit: a request that must run
/// somewhere still runs on a full replica rather than not at all.
pub(crate) fn validate_place(
    call: &CallView,
    replicas: &[ReplicaView],
    chosen: usize,
) -> Result<(), PolicyError> {
    let Some(replica) = replicas.iter().find(|replica| replica.index == chosen) else {
        return Err(invalid(
            Call::Place,
            format!(
                "replica {chosen} is not one of the {} replicas offered",
                replicas.len()
            ),
        ));
    };
    if !replica.kinds.contains(&call.kind) {
        return Err(invalid(
            Call::Place,
            format!("replica {chosen} does not run {:?} calls", call.kind),
        ));
    }
    if replica.prefills_only && decodes(call.kind) {
        return Err(invalid(
            Call::Place,
            format!(
                "replica {chosen} runs prefill only, a {:?} call cannot land on it",
                call.kind
            ),
        ));
    }
    Ok(())
}

/// Checks a warm order: every page is one of the pages the epoch's precopy
/// claimed, and none is named twice.
pub(crate) fn validate_warm(room: &Room, pages: &[PageId]) -> Result<(), PolicyError> {
    let claimed: BTreeSet<PageId> = room
        .warming
        .iter()
        .flat_map(|warming| warming.pages.iter().copied())
        .collect();
    let mut seen = BTreeSet::new();
    for page in pages {
        if !claimed.contains(page) {
            return Err(invalid(
                Call::Warm,
                format!(
                    "page {} is not one of the pages this precopy claimed",
                    page.0
                ),
            ));
        }
        if !seen.insert(*page) {
            return Err(invalid(
                Call::Warm,
                format!("page {} is warmed twice", page.0),
            ));
        }
    }
    Ok(())
}

/// Checks an expert order: a placement of every expert on a rank, balanced so
/// that no rank carries more than one expert more than another (design §7.6).
///
/// The load is the input the order is meant to balance; an order that is legal
/// but leaves the load unbalanced is a poor decision, not an invalid one.
pub(crate) fn validate_experts(load: &ExpertLoad, order: &ExpertOrder) -> Result<(), PolicyError> {
    if load.ranks == 0 || load.experts == 0 {
        return Err(invalid(
            Call::Experts,
            "the mechanism asked for an expert order with no ranks or no experts",
        ));
    }
    if order.placement.len() != load.experts {
        return Err(invalid(
            Call::Experts,
            format!(
                "the order places {} experts, the model has {}",
                order.placement.len(),
                load.experts
            ),
        ));
    }
    let mut per_rank = vec![0usize; load.ranks];
    for (expert, rank) in order.placement.iter().enumerate() {
        if *rank >= load.ranks {
            return Err(invalid(
                Call::Experts,
                format!(
                    "expert {expert} is placed on rank {rank}, there are {} ranks",
                    load.ranks
                ),
            ));
        }
        per_rank[*rank] += 1;
    }
    let most = per_rank.iter().copied().max().unwrap_or(0);
    let least = per_rank.iter().copied().min().unwrap_or(0);
    if most - least > 1 {
        return Err(invalid(
            Call::Experts,
            format!(
                "the order leaves {} experts on one rank and {least} on another",
                most
            ),
        ));
    }
    Ok(())
}

/// The tokens a completed call advanced its request by, as a policy counts
/// them: a prompt chunk by the tokens it wrote, a verification by the drafts it
/// checked, a token-advancing phase by one, and a denoising step by one latent
/// unit.
pub(crate) fn advanced_tokens(call: &ipc::Call) -> u32 {
    match call.kind {
        ipc::CallKind::Forward(ipc::ForwardMode::Verify) => {
            1 + u32::try_from(crate::scheduler::generation::verify_drafts(call).len())
                .unwrap_or(u32::MAX)
        }
        ipc::CallKind::Media(ipc::MediaCall::Denoising) => 1,
        ipc::CallKind::Forward(ipc::ForwardMode::Prefill) => call.bounds.max_tokens,
        _ => 1,
    }
}

/// The state changes the policy is told about, in the order they happened.
///
/// Events are proportional to state changes, never to resident requests or
/// cache pages (§4.4): the mechanism records one event per change it makes, and
/// an `Updated` only for a request whose scheduling-relevant view moved since
/// the policy was last told. The list is closed before the step's `dispatch`
/// and handed over once, by `observe`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Recorder {
    /// Events since the last `observe`.
    pending: Vec<Event>,
    /// The last view the policy was told about, per live request.
    told: BTreeMap<RequestId, RequestView>,
    /// Requests whose view changed since the last `close`.
    touched: BTreeMap<RequestId, RequestView>,
    /// The features no view of the request carries any more, kept for
    /// `views.rs` between one step and the next.
    recorded: RecordedTable,
}

impl Recorder {
    /// A recorder that starts from the state a policy is instantiated with: it
    /// was told every request's view in the `Snapshot`, so no `Updated` repeats
    /// one.
    pub(crate) fn from_snapshot(snapshot: &Snapshot) -> Self {
        let told = snapshot
            .requests
            .iter()
            .map(|request| (request.id, request.clone()))
            .collect();
        let recorded = snapshot
            .requests
            .iter()
            .map(|request| {
                (
                    ipc::RequestId(request.id.0),
                    Recorded {
                        admitted_at: request.admitted_at,
                        last_emitted_at: request.last_emitted_at,
                        cached_prefix_tokens: request.cached_prefix_tokens,
                        prefetch_tokens: request.prefetch_tokens,
                        cache_rank: request.cache_rank,
                    },
                )
            })
            .collect();
        Self {
            pending: Vec::new(),
            told,
            touched: BTreeMap::new(),
            recorded,
        }
    }

    /// Records one state change.
    pub(crate) fn record(&mut self, event: Event) {
        match &event {
            Event::Arrived { request } => {
                self.told.insert(request.id, request.clone());
                self.recorded.insert(
                    ipc::RequestId(request.id.0),
                    Recorded {
                        admitted_at: None,
                        last_emitted_at: None,
                        cached_prefix_tokens: request.cached_prefix_tokens,
                        prefetch_tokens: request.prefetch_tokens,
                        cache_rank: request.cache_rank,
                    },
                );
            }
            Event::Admitted { request } => {
                if let Some(view) = self.told.get_mut(request) {
                    view.state = RequestState::Resident;
                }
            }
            Event::Finished { request, .. } => {
                self.told.remove(request);
                self.touched.remove(request);
                self.recorded.remove(&ipc::RequestId(request.0));
            }
            Event::Emitted { request, at, .. } => {
                self.recorded
                    .entry(ipc::RequestId(request.0))
                    .or_default()
                    .last_emitted_at = Some(*at);
            }
            _ => {}
        }
        self.pending.push(event);
    }

    /// Notes a view the mechanism changed. The `Updated` is emitted by `close`,
    /// and only when the view differs from the one the policy was last told.
    pub(crate) fn update(&mut self, request: RequestView) {
        self.touched.insert(request.id, request);
    }

    /// Closes the step: one `Updated` per request whose view moved.
    pub(crate) fn close(&mut self) {
        for (id, view) in std::mem::take(&mut self.touched) {
            let changed = self.told.get(&id).is_none_or(|told| moved(told, &view));
            if changed {
                self.told.insert(id, view.clone());
                self.pending.push(Event::Updated { request: view });
            }
        }
    }

    /// Notes an admission: the event, and the time the views report as
    /// `admitted_at`.
    pub(crate) fn admitted(&mut self, request: RequestId, at: Nanos) {
        self.recorded
            .entry(ipc::RequestId(request.0))
            .or_default()
            .admitted_at = Some(at);
        self.record(Event::Admitted { request });
    }

    /// The features `views.rs` reads from the recorder rather than from the
    /// scheduler's own state.
    pub(crate) fn recorded(&self) -> &RecordedTable {
        &self.recorded
    }

    /// Hands the events to the policy. The policy sees them once; the next step
    /// starts a new list.
    pub(crate) fn take(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.pending)
    }
}

/// Whether a request's view moved in a way the policy must see.
///
/// Only the scheduling-relevant features count: a request's priority and
/// arrival never change, and reporting an unchanged view would tell the policy
/// twice about one state change.
fn moved(told: &RequestView, view: &RequestView) -> bool {
    told.state != view.state
        || told.phase != view.phase
        || told.prompt_tokens != view.prompt_tokens
        || told.generated_tokens != view.generated_tokens
        || told.cached_prefix_tokens != view.cached_prefix_tokens
        || told.prefilled_tokens != view.prefilled_tokens
        || told.cached_pages != view.cached_pages
        || told.waiting_transfer != view.waiting_transfer
        || told.next_tokens != view.next_tokens
        || told.prefetch_tokens != view.prefetch_tokens
        || told.refill_tokens != view.refill_tokens
        || told.remaining_tokens != view.remaining_tokens
        || told.migratable != view.migratable
        || told.cache_rank != view.cache_rank
        || told.last_emitted_at != view.last_emitted_at
}

/// The starvation watchdog (design §4.4).
///
/// A resident request that is not waiting for a transfer and has not advanced
/// within `limits().timeout` is the policy's failure, exactly as sched_ext
/// treats a runnable task its scheduler did not run within `timeout_ms`. The
/// mechanism decides when a request advanced (one completed call it put in a
/// plan); the policy cannot move the deadline, only its own `timeout`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Watchdog {
    /// When each resident request last advanced, or was admitted.
    advancing: BTreeMap<RequestId, Nanos>,
}

impl Watchdog {
    /// A request became resident: its deadline starts now.
    pub(crate) fn admitted(&mut self, request: RequestId, at: Nanos) {
        self.advancing.insert(request, at);
    }

    /// A call of the request completed.
    pub(crate) fn advanced(&mut self, request: RequestId, at: Nanos) {
        self.advancing.insert(request, at);
    }

    /// The request ended; it is no longer observed.
    pub(crate) fn finished(&mut self, request: RequestId) {
        self.advancing.remove(&request);
    }

    /// The first request starved past `timeout`, if any.
    ///
    /// A request waiting for a transfer is not starved: the mechanism cannot
    /// progress it, so waiting is not the policy's doing. The first by id, so a
    /// report is deterministic.
    pub(crate) fn starved(&self, live: &Live, now: Nanos, timeout: Nanos) -> Option<RequestId> {
        for (id, since) in &self.advancing {
            let Some(request) = live.requests.get(id) else {
                continue;
            };
            if request.state != RequestState::Resident || request.waiting_transfer {
                continue;
            }
            if now.0.saturating_sub(since.0) > timeout.0 {
                return Some(*id);
            }
        }
        None
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use scheduling::guest::{ComponentId, RequestClass, Room, Storage, Warming};

    fn request(id: u64, state: RequestState) -> RequestView {
        RequestView {
            id: RequestId(id),
            state,
            phase: None,
            priority: 0,
            arrived_at: Nanos(0),
            admitted_at: None,
            prompt_tokens: 0,
            generated_tokens: 0,
            cached_prefix_tokens: 0,
            prefilled_tokens: 0,
            cached_pages: 0,
            last_emitted_at: None,
            class: RequestClass::Token,
            waiting_transfer: false,
            next_tokens: 0,
            reservation_units: 0,
            prefetch_tokens: 0,
            refill_tokens: None,
            remaining_tokens: 0,
            migratable: false,
            cache_rank: None,
        }
    }

    fn live(requests: Vec<RequestView>, queue: Vec<RequestId>, kinds: &[(u32, PhaseKind)]) -> Live {
        Live {
            requests: requests.into_iter().map(|r| (r.id, r)).collect(),
            queue,
            pages: BTreeMap::new(),
            encodings: BTreeMap::new(),
            phases: kinds
                .iter()
                .map(|(id, kind)| {
                    (
                        PhaseId(*id),
                        PhaseView {
                            id: PhaseId(*id),
                            kind: *kind,
                            block: ComponentId(*id),
                        },
                    )
                })
                .collect(),
        }
    }

    fn room() -> Room {
        Room {
            step: 1,
            kv_units: 10,
            slots: 4,
            token_budget: 128,
            call_budget: 8,
            page_tokens: 16,
            ..Room::default()
        }
    }

    fn plan(work: Vec<Work>) -> Plan {
        Plan {
            admit: Vec::new(),
            reject: Vec::new(),
            work,
            offload: Vec::new(),
            prefetch: Vec::new(),
            refill: Vec::new(),
        }
    }

    /// A fixed policy, standing in for `policy.rs`'s slot in the mechanism's
    /// tests: it exercises the validators and the failure hooks without wasmtime
    /// and without the switch.
    struct Static {
        limits: Limits,
        plan: Plan,
        victims: Vec<Victim>,
        failures: usize,
    }

    impl Decisions for Static {
        fn limits(&self) -> Limits {
            self.limits
        }

        fn observe(&mut self, _events: &[Event]) -> Result<(), PolicyError> {
            Ok(())
        }

        fn dispatch(&mut self, _room: &Room) -> Result<Plan, PolicyError> {
            Ok(self.plan.clone())
        }

        fn reclaim(&mut self, _shortage: &Shortage) -> Result<Vec<Victim>, PolicyError> {
            Ok(self.victims.clone())
        }

        fn place(
            &mut self,
            _call: &CallView,
            replicas: &[ReplicaView],
        ) -> Result<usize, PolicyError> {
            Ok(replicas.first().map(|replica| replica.index).unwrap_or(0))
        }

        fn warm(&mut self, room: &Room) -> Result<Vec<PageId>, PolicyError> {
            Ok(room
                .warming
                .iter()
                .flat_map(|warming| warming.pages.iter().copied())
                .collect())
        }

        fn experts(&mut self, load: &ExpertLoad) -> Result<ExpertOrder, PolicyError> {
            Ok(ExpertOrder {
                placement: (0..load.experts)
                    .map(|expert| expert % load.ranks.max(1))
                    .collect(),
            })
        }

        fn fail(
            &mut self,
            error: &PolicyError,
            _state: &TrialState,
        ) -> Result<PolicyReverted, PolicyError> {
            self.failures += 1;
            Err(PolicyError::invalid(Call::Init, error.to_string()))
        }

        fn boundary(&mut self, _state: &TrialState) -> Result<Option<PolicyReverted>, PolicyError> {
            Ok(None)
        }
    }

    #[test]
    fn a_static_policy_goes_through_the_same_validators_the_live_path_does() {
        let mut live = live(
            vec![
                {
                    let r = request(1, RequestState::Queued);
                    r
                },
                {
                    let mut r = request(2, RequestState::Resident);
                    r.phase = Some(PhaseId(1));
                    r.next_tokens = 1;
                    r
                },
            ],
            vec![RequestId(1)],
            &[(0, PhaseKind::Prefill), (1, PhaseKind::Decode)],
        );
        let mut plan = plan(vec![Work {
            request: RequestId(2),
            phase: PhaseId(1),
            tokens: 1,
        }]);
        plan.admit = vec![RequestId(1)];
        let mut policy = Static {
            limits: Limits::default(),
            plan,
            victims: Vec::new(),
            failures: 0,
        };

        // A good answer validates and executes; the failed one is the policy's.
        let decision = policy.dispatch(&room()).expect("the fixed plan");
        validate_plan(&decision, &room(), &live).expect("a valid plan");

        policy.plan.work[0].tokens = 9;
        let decision = policy.dispatch(&room()).expect("the fixed plan");
        let error = validate_plan(&decision, &room(), &live).expect_err("an invalid plan");
        let report = policy.fail(&error, &TrialState::default());
        assert_eq!(policy.failures, 1);
        assert!(
            report.is_err(),
            "a policy with no previous one cannot serve on"
        );
        assert!(
            policy
                .boundary(&TrialState::default())
                .expect("a boundary")
                .is_none()
        );

        // The routing and placement answers are checked the same way.
        let ranks = vec![RankView {
            index: 0,
            in_flight: 0,
            alive: true,
        }];
        validate_route(&live.requests[&RequestId(2)], &ranks, 0).expect("a live rank");
        live.queue.clear();
    }

    #[test]
    fn admission_names_only_queued_requests_and_never_twice() {
        let live = live(
            vec![
                request(1, RequestState::Queued),
                request(2, RequestState::Queued),
                request(3, RequestState::Resident),
            ],
            vec![RequestId(1), RequestId(2)],
            &[],
        );
        // Any order of queued requests is legal; the order is the policy's.
        let mut p = plan(Vec::new());
        p.admit = vec![RequestId(2), RequestId(1)];
        validate_plan(&p, &room(), &live).expect("a policy may reorder the queue");

        p.admit = vec![RequestId(2), RequestId(2)];
        let error = validate_plan(&p, &room(), &live).expect_err("twice");
        assert!(format!("{error}").contains("twice"));

        p.admit = vec![RequestId(3)];
        let error = validate_plan(&p, &room(), &live).expect_err("not queued");
        assert!(format!("{error}").contains("not queued"));

        p.admit = vec![RequestId(1)];
        p.reject = vec![RequestId(1)];
        let error = validate_plan(&p, &room(), &live).expect_err("both");
        assert!(format!("{error}").contains("both admitted and rejected"));

        p.reject.clear();
        validate_plan(&p, &room(), &live).expect("admit in order");
    }

    #[test]
    fn work_runs_the_phase_the_request_is_at_and_what_the_mechanism_charges() {
        let mut decoding = request(1, RequestState::Resident);
        decoding.phase = Some(PhaseId(1));
        decoding.next_tokens = 1;
        let mut prefilling = request(2, RequestState::Resident);
        prefilling.phase = Some(PhaseId(0));
        prefilling.prompt_tokens = 100;
        prefilling.prefilled_tokens = 40;
        let live = live(
            vec![decoding, prefilling],
            Vec::new(),
            &[(0, PhaseKind::Prefill), (1, PhaseKind::Decode)],
        );

        // The prefill may write what is left of the prompt and not a token more.
        let mut p = plan(vec![Work {
            request: RequestId(2),
            phase: PhaseId(0),
            tokens: 60,
        }]);
        validate_plan(&p, &room(), &live).expect("the rest of the prompt");
        p.work[0].tokens = 61;
        let error = validate_plan(&p, &room(), &live).expect_err("past the prompt");
        assert!(format!("{error}").contains("prompt tokens left"));

        // A decode step is charged what the mechanism charges, not what a
        // policy would like.
        let mut p = plan(vec![Work {
            request: RequestId(1),
            phase: PhaseId(1),
            tokens: 4,
        }]);
        let error = validate_plan(&p, &room(), &live).expect_err("wrong charge");
        assert!(format!("{error}").contains("charged 1 tokens"));
        p.work[0].tokens = 1;
        // ... and the phase must be the one the request is at.
        p.work[0].phase = PhaseId(0);
        let error = validate_plan(&p, &room(), &live).expect_err("wrong phase");
        assert!(format!("{error}").contains("next phase"));

        // A request that is not resident cannot be given work.
        let mut queued = request(3, RequestState::Queued);
        queued.phase = Some(PhaseId(1));
        let mut live = live.clone();
        live.requests.insert(queued.id, queued);
        let p = plan(vec![Work {
            request: RequestId(3),
            phase: PhaseId(1),
            tokens: 1,
        }]);
        let error = validate_plan(&p, &room(), &live).expect_err("queued");
        assert!(format!("{error}").contains("not resident"));
    }

    #[test]
    fn the_step_holds_its_token_call_and_prefill_budgets() {
        let mut live = live(
            vec![
                {
                    let mut r = request(1, RequestState::Resident);
                    r.phase = Some(PhaseId(0));
                    r.prompt_tokens = 1000;
                    r
                },
                {
                    let mut r = request(2, RequestState::Resident);
                    r.phase = Some(PhaseId(0));
                    r.prompt_tokens = 1000;
                    r
                },
            ],
            Vec::new(),
            &[(0, PhaseKind::Prefill)],
        );
        let mut r = room();
        r.token_budget = 100;
        let p = plan(vec![
            Work {
                request: RequestId(1),
                phase: PhaseId(0),
                tokens: 60,
            },
            Work {
                request: RequestId(2),
                phase: PhaseId(0),
                tokens: 60,
            },
        ]);
        let error = validate_plan(&p, &r, &live).expect_err("over the step");
        assert!(format!("{error}").contains("120 tokens"));

        // A warming epoch bounds the prefill work separately.
        r.token_budget = 1000;
        r.prefill_tokens = Some(80);
        let error = validate_plan(&p, &r, &live).expect_err("over the prefill budget");
        assert!(format!("{error}").contains("may still write 80"));

        r.prefill_tokens = None;
        r.call_budget = 1;
        let error = validate_plan(&p, &r, &live).expect_err("over the calls");
        assert!(format!("{error}").contains("2 calls"));

        // A request gets one piece of work in a step.
        r.call_budget = 8;
        let p = plan(vec![
            Work {
                request: RequestId(1),
                phase: PhaseId(0),
                tokens: 10,
            },
            Work {
                request: RequestId(1),
                phase: PhaseId(0),
                tokens: 10,
            },
        ]);
        let error = validate_plan(&p, &r, &live).expect_err("twice");
        assert!(format!("{error}").contains("two pieces of work"));
        live.requests.clear();
    }

    #[test]
    fn only_pages_no_request_continues_from_are_offloaded() {
        let mut live = live(vec![], Vec::new(), &[]);
        live.pages.insert(
            PageId(1),
            PageView {
                id: PageId(1),
                tokens: 16,
                last_used: Nanos(1),
                on_host: false,
                evictable: false,
            },
        );
        live.pages.insert(
            PageId(2),
            PageView {
                id: PageId(2),
                tokens: 16,
                last_used: Nanos(2),
                on_host: true,
                evictable: true,
            },
        );
        live.pages.insert(
            PageId(3),
            PageView {
                id: PageId(3),
                tokens: 16,
                last_used: Nanos(3),
                on_host: false,
                evictable: true,
            },
        );

        let mut p = plan(Vec::new());
        p.offload = vec![PageId(1)];
        let error = validate_plan(&p, &room(), &live).expect_err("continued from");
        assert!(format!("{error}").contains("still continued from"));

        p.offload = vec![PageId(2)];
        let error = validate_plan(&p, &room(), &live).expect_err("already on the host");
        assert!(format!("{error}").contains("already on the host"));

        p.offload = vec![PageId(3), PageId(3)];
        let error = validate_plan(&p, &room(), &live).expect_err("twice");
        assert!(format!("{error}").contains("twice"));

        p.offload = vec![PageId(3)];
        validate_plan(&p, &room(), &live).expect("a free device page");
    }

    #[test]
    fn a_victim_must_free_the_storage_the_shortage_is_about() {
        let mut live = live(
            vec![
                {
                    let mut r = request(1, RequestState::Resident);
                    r.migratable = false;
                    r
                },
                {
                    let mut r = request(2, RequestState::Resident);
                    r.migratable = true;
                    r
                },
            ],
            Vec::new(),
            &[],
        );
        live.pages.insert(
            PageId(1),
            PageView {
                id: PageId(1),
                tokens: 16,
                last_used: Nanos(1),
                on_host: false,
                evictable: true,
            },
        );
        live.pages.insert(
            PageId(2),
            PageView {
                id: PageId(2),
                tokens: 16,
                last_used: Nanos(2),
                on_host: true,
                evictable: true,
            },
        );
        live.encodings.insert(
            EncodingId(7),
            EncodingView {
                id: EncodingId(7),
                last_used: Nanos(1),
            },
        );

        let units = Shortage {
            kind: Storage::KvUnits,
            need: 4,
        };
        validate_victims(&units, &[Victim::Page(PageId(1))], &live).expect("a free page");
        validate_victims(&units, &[Victim::Request(RequestId(2))], &live)
            .expect("a migratable request");

        let error = validate_victims(&units, &[], &live).expect_err("nothing freed");
        assert!(format!("{error}").contains("nothing was freed"));
        let error = validate_victims(&units, &[Victim::Request(RequestId(1))], &live)
            .expect_err("not migratable");
        assert!(format!("{error}").contains("not migratable"));
        let error = validate_victims(&units, &[Victim::Page(PageId(2))], &live)
            .expect_err("a host page frees no device units");
        assert!(format!("{error}").contains("frees no device units"));
        let error = validate_victims(&units, &[Victim::Encoding(EncodingId(7))], &live)
            .expect_err("wrong storage");
        assert!(format!("{error}").contains("frees nothing"));

        let pages = Shortage {
            kind: Storage::HostPages,
            need: 2,
        };
        validate_victims(&pages, &[Victim::Page(PageId(2))], &live).expect("a host page");
        let error = validate_victims(&pages, &[Victim::Page(PageId(1))], &live)
            .expect_err("not on the host");
        assert!(format!("{error}").contains("not on the host"));

        let entries = Shortage {
            kind: Storage::EncoderEntries,
            need: 1,
        };
        validate_victims(&entries, &[Victim::Encoding(EncodingId(7))], &live).expect("an entry");
    }

    #[test]
    fn a_rank_must_be_offered_and_alive_and_a_replica_must_run_the_call() {
        let ranks = vec![
            RankView {
                index: 0,
                in_flight: 3,
                alive: true,
            },
            RankView {
                index: 1,
                in_flight: 0,
                alive: false,
            },
        ];
        let request = request(9, RequestState::Queued);
        validate_route(&request, &ranks, 0).expect("the live rank");
        let error = validate_route(&request, &ranks, 1).expect_err("dead");
        assert!(format!("{error}").contains("not alive"));
        let error = validate_route(&request, &ranks, 7).expect_err("unknown");
        assert!(format!("{error}").contains("not one of"));

        let replicas = vec![
            ReplicaView {
                index: 0,
                block: ComponentId(0),
                kinds: vec![PhaseKind::Prefill],
                resident_requests: 1,
                capacity: 1,
                holds_request: false,
                prefills_only: true,
                holds_component: false,
            },
            ReplicaView {
                index: 1,
                block: ComponentId(1),
                kinds: vec![PhaseKind::Decode],
                resident_requests: 4,
                capacity: 0,
                holds_request: false,
                prefills_only: false,
                holds_component: false,
            },
        ];
        let call = CallView {
            request: RequestId(9),
            phase: PhaseId(1),
            kind: PhaseKind::Decode,
        };
        validate_place(&call, &replicas, 1).expect("a full replica still runs the call");
        let error = validate_place(&call, &replicas, 0).expect_err("prefill only");
        assert!(format!("{error}").contains("prefill only"));
        let call = CallView {
            request: RequestId(9),
            phase: PhaseId(1),
            kind: PhaseKind::Mux,
        };
        let error = validate_place(&call, &replicas, 1).expect_err("cannot run it");
        assert!(format!("{error}").contains("does not run"));
    }

    #[test]
    fn warm_and_experts_must_be_legal() {
        let mut room = room();
        room.warming = vec![Warming {
            request: RequestId(1),
            pages: vec![PageId(1), PageId(2)],
        }];
        validate_warm(&room, &[PageId(2), PageId(1)]).expect("any order of the claimed pages");
        let error = validate_warm(&room, &[PageId(3)]).expect_err("not claimed");
        assert!(format!("{error}").contains("not one of the pages"));

        let load = ExpertLoad {
            ranks: 2,
            experts: 4,
            tokens: vec![10, 0, 9, 1],
        };
        validate_experts(
            &load,
            &ExpertOrder {
                placement: vec![0, 0, 1, 1],
            },
        )
        .expect("balanced");
        let error = validate_experts(
            &load,
            &ExpertOrder {
                placement: vec![0, 0, 0, 0],
            },
        )
        .expect_err("unbalanced");
        assert!(format!("{error}").contains("experts on one rank"));
        let error = validate_experts(
            &load,
            &ExpertOrder {
                placement: vec![0, 1, 0],
            },
        )
        .expect_err("short");
        assert!(format!("{error}").contains("places 3 experts"));
        let error = validate_experts(
            &load,
            &ExpertOrder {
                placement: vec![0, 0, 2, 2],
            },
        )
        .expect_err("no such rank");
        assert!(format!("{error}").contains("there are 2 ranks"));
    }

    #[test]
    fn the_recorder_tells_one_update_per_change_and_none_for_an_unchanged_view() {
        let mut recorder = Recorder::default();
        let mut arrived = request(1, RequestState::Queued);
        arrived.prompt_tokens = 100;
        arrived.phase = Some(PhaseId(0));
        recorder.record(Event::Arrived {
            request: arrived.clone(),
        });

        // A view touched without changing is not an event.
        recorder.update(arrived.clone());
        recorder.close();
        assert_eq!(recorder.take().len(), 1, "the arrival is the only event");

        // A change is one `Updated`, before the step's dispatch.
        arrived.prefilled_tokens = 40;
        recorder.update(arrived.clone());
        recorder.update(arrived.clone());
        recorder.close();
        let events = recorder.take();
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Updated { request } => assert_eq!(request.prefilled_tokens, 40),
            other => panic!("expected an Updated, got {other:?}"),
        }

        // A finished request is not told about again.
        recorder.record(Event::Finished {
            request: RequestId(1),
            end: scheduling::guest::End::Completed,
        });
        recorder.update(arrived);
        recorder.close();
        assert!(
            recorder
                .take()
                .iter()
                .all(|event| !matches!(event, Event::Updated { .. }))
        );
    }

    #[test]
    fn the_watchdog_finds_a_resident_request_that_stopped_advancing_but_not_a_waiting_one() {
        let mut live = live(
            vec![
                {
                    let mut r = request(1, RequestState::Resident);
                    r
                },
                {
                    let mut r = request(2, RequestState::Resident);
                    r.waiting_transfer = true;
                    r
                },
                {
                    let mut r = request(3, RequestState::Resident);
                    r
                },
            ],
            Vec::new(),
            &[],
        );
        let mut watchdog = Watchdog::default();
        watchdog.admitted(RequestId(1), Nanos(0));
        watchdog.admitted(RequestId(2), Nanos(0));
        watchdog.admitted(RequestId(3), Nanos(0));
        watchdog.advanced(RequestId(3), Nanos(5_000_000_000));

        let timeout = Nanos(1_000_000_000);
        assert_eq!(
            watchdog.starved(&live, Nanos(500_000_000), timeout),
            None,
            "within the timeout"
        );
        // Request 1 is past it; 2 waits for a transfer; 3 advanced.
        assert_eq!(
            watchdog.starved(&live, Nanos(2_000_000_000), timeout),
            Some(RequestId(1))
        );
        live.requests.get_mut(&RequestId(1)).expect("live").state = RequestState::Finished;
        assert_eq!(watchdog.starved(&live, Nanos(2_000_000_000), timeout), None);
        watchdog.finished(RequestId(3));
        assert_eq!(
            watchdog.starved(&live, Nanos(20_000_000_000), timeout),
            None
        );
    }
}
