//! What the scheduling policy reads of the scheduler (`specs/scheduler-design.md` §5.2): the view
//! of a request ([`view_of`]), the room of a step, the snapshot an instance starts from, and the
//! views of the groups a request may be placed on.
//!
//! The policy holds the view; the scheduler holds none. A view is computed from the request's own
//! row and its groups whenever the policy is told of the request: at its arrival, once per step
//! after a change no other event carries (`Scheduler::mark`, `Updated`), and in a snapshot. A
//! snapshot is built only when a policy is instantiated, from the scheduler's own state.
//!
//! Time on the policy's wire is nanoseconds of the host's monotonic clock
//! ([`super::monotonic_nanos`]); a policy only compares instants.

use crate::batch::Phase;
use proto::request::RequestId;
use scheduling::decision::{Inflight, Room};
use scheduling::ids::{RequestId as PolicyRequestId, StageId};
use scheduling::time::Nanos;
use scheduling::view::{
    CallView, PhaseView, ReplicaView, RequestClass, RequestStatus as Standing, RequestView,
    Snapshot,
};

use super::{Request, RequestStatus, Scheduler, WorkerGroup, monotonic_nanos};

/// The view of `request` the policy reads, computed from the request's fields and the
/// capacities of `groups` (the scheduler's), the 22 features of `S/views.rs:456-760`:
///
/// - `state`: a request admitting (`WaitingForRemoteKvs` in the waiting queue) still waits; one
///   waiting for its transfer runs. `waiting_transfer` is `WaitingForRemoteKvs`.
/// - `phase`: the first phase the request is ready for; `next_tokens` what that call takes from
///   the step (one for a decode, one and its drafts for a verification, nothing for a prefill
///   chunk, which the policy sizes, or a media call).
/// - Token counts: `prompt_tokens` is the prompt the request was submitted with, which its usage
///   charges; `prefill_tokens` the prompt it computes (a resumed request's generated tokens
///   included); `generated_tokens` every token it generated, on this engine and before;
///   `prefilled_tokens` where its next prompt chunk starts; `cached_pages` the blocks its cache
///   groups' tables hold; `refill_tokens` the draft cache's gap a refill can close, when one
///   is ready and can be given its blocks ([`refill_writable`]), which a step of `tokens`
///   tokens at most sizes its next chunk by; with `tokens` `None` no refill is offered.
/// - `admitted_at` is the request's arrival once it is admitted: the row keeps no admission
///   instant, and arrival orders admitted requests as admission does but where the queue put a
///   resumed request first. `last_emitted_at`, `prefetch_tokens` and `reservation_units` are
///   empty: the policy learns emissions from `Emitted`, and the scheduler applies its own
///   prefetch threshold and capacity rejection at admission (SD §3.5, §4.4).
/// - A token request may be preempted as migratable until it finishes; a media request is not.
pub(super) fn view(request: &Request, groups: &[WorkerGroup], tokens: Option<u32>) -> RequestView {
    let (class, prompt, max_tokens) = match &request.input {
        proto::request::Request::Generation(input) => (
            RequestClass::Token,
            count(input.prompt_token_ids.len()),
            count(input.max_und_tokens),
        ),
        proto::request::Request::Diffusion(input) => {
            (RequestClass::Media, count(input.prompt_token_ids.len()), 0)
        }
    };
    let token = class == RequestClass::Token;

    let state = match request.status {
        RequestStatus::Waiting => Standing::Waiting,
        RequestStatus::WaitingForRemoteKvs if request.queued => Standing::Waiting,
        RequestStatus::WaitingForRemoteKvs | RequestStatus::Running => Standing::Running,
        RequestStatus::Finished => Standing::Finished,
    };
    let phase = match request.status {
        RequestStatus::Finished => None,
        _ => request.ready_phases(groups).next(),
    };
    let next_tokens = match phase {
        Some(Phase::Decode) => 1,
        Some(Phase::Verify) => {
            let eos = groups
                .first()
                .map_or(&[][..], |group| &group.capacity.eos_token_ids[..]);
            1 + request.drafts(eos)
        }
        _ => 0,
    };

    // The blocks of its cache groups' tables: the first classes, one per cache group (SD §4.4),
    // as many as it has key chains.
    let cached_pages = request
        .blocks
        .iter()
        .take(request.keys.len())
        .map(|blocks| blocks.end.saturating_sub(blocks.first))
        .sum();
    let generated = if token {
        count(request.token_ids.len()).saturating_sub(prompt)
    } else {
        0
    };
    // The draft cache's gap a refill can close: up to its end (`Request::refill_end`).
    let refill = request
        .refill_end()
        .saturating_sub(request.num_draft_tokens);

    RequestView {
        id: PolicyRequestId(request.id.0),
        state,
        phase: phase.map(Phase::id),
        priority: request.input.priority(),
        arrived_at: Nanos(monotonic_nanos(request.arrival)),
        admitted_at: (state == Standing::Running).then(|| Nanos(monotonic_nanos(request.arrival))),
        prompt_tokens: prompt,
        prefill_tokens: if token {
            request.num_prompt_tokens
        } else {
            prompt
        },
        generated_tokens: generated,
        cached_prefix_tokens: request.num_cached_tokens,
        prefilled_tokens: request
            .num_computed_tokens
            .min(request.num_prompt_tokens)
            .max(request.num_cached_tokens),
        cached_pages,
        last_emitted_at: None,
        class,
        waiting_transfer: request.status == RequestStatus::WaitingForRemoteKvs,
        next_tokens,
        reservation_units: 0,
        prefetch_tokens: 0,
        refill_tokens: tokens
            .filter(|&tokens| {
                token
                    && refill > 0
                    && request.ready(Phase::Refill, groups)
                    && refill_writable(request, groups, tokens)
            })
            .map(|_| refill),
        remaining_tokens: max_tokens.saturating_sub(generated),
        migratable: token && request.status != RequestStatus::Finished,
    }
}

/// The view of a request no step of a scheduler sizes a refill for ([`view`] with no refill
/// offered): an arrival a scheduler tells its policy.
pub(super) fn view_of(request: &Request, groups: &[WorkerGroup]) -> RequestView {
    view(request, groups, None)
}

/// Whether the scheduler can form `request`'s refill once the blocks it asked for arrive (SD
/// §3.4, §3.6, §7.1): the refill starts at the draft cache's end, and its draft table holds or
/// has asked for every block of its next chunk, which a step of `tokens` tokens sizes as
/// `Scheduler::charge` and `allocate_ahead` do. Draft blocks take only the units the regions
/// have free, so a table given up under a shortage, or one whose next chunk lies beyond what it
/// was given, asks for nothing until units are free: a refill offered then could not be formed,
/// and a policy that runs a refill as the only work of its step would form nothing while the
/// shortage lasts. A drafter that keeps no cache of its own writes no draft block.
fn refill_writable(request: &Request, groups: &[WorkerGroup], tokens: u32) -> bool {
    let placed = request.placement[Phase::Refill.index()];
    let Some(group) = groups.iter().find(|group| Some(group.id()) == placed) else {
        return false;
    };
    let shapes = &group.capacity.cache_groups;
    let state = shapes
        .iter()
        .position(|(kind, _)| kind.is_state())
        .map(|class| (class, shapes[class].1));

    // `charge` forms a chunk that continues the draft cache, of at least one token.
    let start = request.num_draft_tokens;
    let end = start.saturating_add(request.refill_chunk(tokens, state));
    if request.refill_start(state) != start || end == start {
        return false;
    }
    let Some(draft) = shapes.iter().position(|(kind, _)| kind.is_draft()) else {
        return true;
    };
    // Positions are logical block indices: the table covers the tokens before `asked` blocks.
    let block = shapes[draft].1;
    request
        .blocks
        .get(draft)
        .is_some_and(|table| table.asked.saturating_mul(block) >= end)
}

/// The call a `place` decision is about: `request`'s first call of `phase` on a group.
pub(super) fn call_view(request: RequestId, phase: Phase) -> CallView {
    CallView {
        request: PolicyRequestId(request.0),
        phase: phase.id(),
        kind: phase,
    }
}

/// The views of the groups `candidates` a `place` decision chooses among for `request`, in the
/// order of `candidates`, which the decision's index names.
pub(super) fn replica_views<'a>(
    request: &Request,
    candidates: impl IntoIterator<Item = &'a WorkerGroup>,
) -> Vec<ReplicaView> {
    candidates
        .into_iter()
        .enumerate()
        .map(|(index, group)| replica_view(index, group, Some(request)))
        .collect()
}

/// The view of one group at `index` of a replica list: the phases it serves, its placed requests
/// and its free slots, whether it prefills only, and whether `request` is attached to it or
/// placed on it for another phase.
fn replica_view(index: usize, group: &WorkerGroup, request: Option<&Request>) -> ReplicaView {
    let id = group.id();
    ReplicaView {
        index,
        // A group runs one block of the model, so its number names the block.
        block: StageId(u32::from(id.0)),
        kinds: served(group).collect(),
        running_requests: group.num_requests,
        capacity: group.capacity.slots.saturating_sub(group.num_requests),
        holds_request: request.is_some_and(|request| {
            request
                .attachments
                .iter()
                .any(|&(attached, _, _, _)| attached == id)
        }),
        holds_stage: request.is_some_and(|request| request.placement.contains(&Some(id))),
    }
}

/// The snapshot of a scheduler of `groups` that holds no request and knows no cached block, as
/// its policy starts from at the scheduler's construction (SD §5.2): the groups that serve
/// phases, which a request may be placed on, and the phases they serve, each named with the
/// first group that serves it.
pub(super) fn snapshot_of(groups: &[WorkerGroup]) -> Snapshot {
    let serving: Vec<&WorkerGroup> = groups
        .iter()
        .filter(|group| group.capacity.phases != 0)
        .collect();
    let phases = Phase::ALL
        .into_iter()
        .filter_map(|phase| {
            let group = serving
                .iter()
                .find(|group| served(group).any(|p| p == phase))?;
            Some(PhaseView {
                id: phase.id(),
                kind: phase,
                block: StageId(u32::from(group.id().0)),
            })
        })
        .collect();
    let replicas = serving
        .iter()
        .enumerate()
        .map(|(index, group)| replica_view(index, group, None))
        .collect();

    Snapshot {
        replicas,
        phases,
        ..Snapshot::default()
    }
}

/// The phases `group` serves, in the order of their ids.
fn served(group: &WorkerGroup) -> impl Iterator<Item = Phase> + '_ {
    Phase::ALL
        .into_iter()
        .filter(|phase| group.capacity.phases & (1 << phase.index()) != 0)
}

/// A count the policy's wire carries in 32 bits.
fn count(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// The policy's views of the scheduler's state.
impl Scheduler {
    /// Notes that `request`'s view changed in a way no other event tells (SD §5.2): the next
    /// `dispatch` tells it once as `Updated`, however often it is marked before.
    pub(super) fn mark(&mut self, id: RequestId) {
        if let Some(request) = self.requests.get_mut(&id)
            && !request.dirty
        {
            request.dirty = true;
            self.policy.dirty.push(id);
        }
    }

    /// The room of this step (SD §5.1 `dispatch`):
    ///
    /// - `kv_units` and `capacity_units`: the units `available` and the `capacity` of the device
    ///   regions the groups take cache-group blocks from (the classes of fixed-size blocks).
    /// - `slots`: the free slots of the most occupied serving group, at most the requests
    ///   `max_running_requests` still admits (`S/views.rs:368-376`).
    /// - `token_budget` and `call_budget`: the policy's own limits.
    /// - `graph_tiers`: the captured sizes the configuration states; `page_tokens` the largest
    ///   block size, 1 when no group holds a cache; `prefill_tokens` is always `None`.
    /// - `inflight`: the batches submitted and not applied, and of them those that may carry
    ///   prefill work: per group, at most its prefill and refill calls in flight.
    pub(super) fn room(&self) -> Room {
        let regions = self.kv_regions();
        let kv_units = regions
            .iter()
            .map(|&region| u64::from(self.kv.available(region)))
            .sum();
        let capacity_units = regions
            .iter()
            .map(|&region| u64::from(self.kv.capacity(region)))
            .sum();

        let free = self
            .groups
            .iter()
            .filter(|group| group.capacity.phases != 0)
            .map(|group| group.capacity.slots.saturating_sub(group.num_requests) as usize)
            .min()
            .unwrap_or(0);
        let admissible = self
            .config
            .max_running_requests
            .saturating_sub(self.running.len());

        let batches = self.groups.iter().map(|group| group.in_flight).sum();
        let prefill_batches = self
            .groups
            .iter()
            .map(|group| {
                let prefills =
                    group.occupied[Phase::Prefill.index()] + group.occupied[Phase::Refill.index()];
                prefills.min(group.in_flight)
            })
            .sum();

        let page_tokens = self
            .groups
            .iter()
            .filter_map(|group| group.capacity.kv_cache.as_ref())
            .filter_map(|cache| cache.block_size)
            .map(|size| size.get())
            .max()
            .unwrap_or(1);

        Room {
            step: self.step,
            kv_units,
            slots: free.min(admissible),
            token_budget: self.budget.tokens,
            call_budget: self.budget.calls,
            chunk_tokens: self.budget.chunk,
            graph_tiers: self.config.graph_tiers.clone(),
            page_tokens,
            prefill_tokens: None,
            inflight: Inflight {
                batches,
                prefill_batches,
            },
            capacity_units,
        }
    }

    /// Everything a starting policy instance is told (SD §5.2): the view of every waiting
    /// request, in the queue's order, then of every running one; a page for every cached block
    /// of a cache group and an encoding for every other cached block (an encoder output); and
    /// the groups of [`snapshot_of`]. A refill is offered as a step of the serving policy's
    /// token limit sizes it.
    pub(super) fn snapshot(&self) -> Snapshot {
        let tokens = Some(self.budget.tokens);
        let requests = self
            .waiting
            .iter()
            .chain(&self.running)
            .filter_map(|id| self.requests.get(id))
            .map(|request| view(request, &self.groups, tokens))
            .collect();
        let (pages, encodings) = self.kv.cache_views(&self.groups);

        Snapshot {
            step: self.step,
            requests,
            pages,
            encodings,
            ..snapshot_of(&self.groups)
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::time::Instant;

    use proto::request::GenerationRequest;

    use super::super::Blocks;
    use super::*;

    /// A waiting token request of `prompt`, as the scheduler holds it on arrival.
    pub(in super::super) fn waiting(id: u64, priority: i32, prompt: &[u32]) -> Request {
        let input = GenerationRequest {
            trace: 0,
            request_id: RequestId(id),
            prompt_token_ids: prompt.to_vec(),
            multimodal_inputs: Default::default(),
            sampling: Default::default(),
            max_und_tokens: 16,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority,
            cache: Default::default(),
            include_stop_token: false,
            generated_token_ids: Vec::new(),
            adapter: 0,
            hand_off: false,
            lend: None,
        };
        let now = Instant::now();
        Request {
            id: RequestId(id),
            input: input.into(),
            arrival: now,
            status: RequestStatus::Waiting,
            placement: [None; Phase::COUNT],
            attachments: Vec::new(),
            scheduled: [0; Phase::COUNT],
            completed: [0; Phase::COUNT],
            early: Vec::new(),
            blocks: Vec::new(),
            queued: true,
            pending: 0,
            token_ids: prompt.to_vec(),
            num_prompt_tokens: count(prompt.len()),
            num_computed_tokens: 0,
            num_cached_tokens: 0,
            replay: 0..0,
            num_host_tokens: 0,
            listing: 0,
            keys: Vec::new(),
            spec_token_ids: Vec::new(),
            num_draft_tokens: 0,
            num_encoded_inputs: 0,
            encodings: Vec::new(),
            unpinned: Vec::new(),
            crossings: Vec::new(),
            num_acknowledged_tokens: 0,
            terminal: None,
            event_credit: 64,
            event_window: 64,
            draft_limit: 0,
            last_progress: now,
            dirty: false,
            path: None,
        }
    }

    /// A resumed request waits with every token it generated before counted as generated and
    /// as prompt it computes, and its usage's prompt is the one it was submitted with.
    #[test]
    fn a_resumed_request_waits_with_its_generated_tokens() {
        let mut request = waiting(7, 3, &[1, 2, 3, 4]);
        // Two tokens generated on the engine it migrated from continue its prompt.
        request.token_ids.extend([5, 6]);
        request.num_prompt_tokens = 6;

        let view = view_of(&request, &[]);
        assert_eq!(view.id, PolicyRequestId(7));
        assert_eq!(view.state, Standing::Waiting);
        assert_eq!(view.class, RequestClass::Token);
        assert_eq!(view.priority, 3);
        assert_eq!(view.prompt_tokens, 4);
        assert_eq!(view.prefill_tokens, 6);
        assert_eq!(view.generated_tokens, 2);
        assert_eq!(view.remaining_tokens, 14);
        assert_eq!(view.admitted_at, None);
        assert!(view.migratable);
        assert!(!view.waiting_transfer);
    }

    /// An admission under way still waits; a transfer and a run are running, and only the
    /// transfer waits for one. Its next prompt chunk starts past its hit and what is scheduled,
    /// and it holds the blocks of its cache groups' tables.
    #[test]
    fn a_running_request_states_its_progress() {
        let mut request = waiting(1, 0, &[1; 100]);
        request.status = RequestStatus::WaitingForRemoteKvs;
        assert_eq!(view_of(&request, &[]).state, Standing::Waiting);

        request.queued = false;
        let view = view_of(&request, &[]);
        assert_eq!(view.state, Standing::Running);
        assert!(view.waiting_transfer);
        assert!(view.admitted_at.is_some());

        request.status = RequestStatus::Running;
        request.num_cached_tokens = 64;
        request.num_computed_tokens = 80;
        // One cache group's table holds blocks 2 to 5; the encoder-output class after it is not
        // a cache group's.
        request.keys = vec![Vec::new()];
        request.blocks = vec![
            Blocks {
                first: 2,
                shared: 4,
                end: 6,
                asked: 7,
            },
            Blocks {
                first: 0,
                shared: 0,
                end: 3,
                asked: 3,
            },
        ];
        let view = view_of(&request, &[]);
        assert!(!view.waiting_transfer);
        assert_eq!(view.cached_prefix_tokens, 64);
        assert_eq!(view.prefilled_tokens, 80);
        assert_eq!(view.cached_pages, 4);

        request.status = RequestStatus::Finished;
        let view = view_of(&request, &[]);
        assert_eq!(view.state, Standing::Finished);
        assert_eq!(view.phase, None);
        assert!(!view.migratable);
    }
}
