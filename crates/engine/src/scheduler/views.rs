//! The mechanism's state, as a scheduling policy reads it (design §4.4).
//!
//! A view is a set of features of a mechanism object, not the object: the
//! scheduler owns requests, pages, encodings, replicas and ranks, and a policy
//! only ever sees the features [`scheduling::guest::view`] names. This module
//! is the only place that reads scheduler state for a policy, so the interface
//! and the mechanism meet in one file:
//!
//! - [`snapshot`] and [`room`] build what `init` and each step's `dispatch`
//!   read;
//! - [`request`], [`call`], [`replica`]s, [`rank`]s, [`shortage`] and
//!   [`expert_load`] build the argument of one decision;
//! - [`arrived`], [`updated`], [`admitted`], [`advanced`], [`emitted`],
//!   [`finished`], [`stepped`], [`cached`], [`hit`], [`freed`] and
//!   [`encoded`] format a state change the recorder decided to report;
//! - [`phase_id`], [`phase_kind`], [`page_target`], [`unit_of`] and
//!   [`encoding_id`] translate an object of the interface back into the
//!   scheduler's, so the mechanism's validators check a decision through one
//!   table instead of rebuilding maps.
//!
//! Every function is pure: it reads the scheduler and returns a value. Nothing
//! here executes work, mutates state, or decides anything a policy decides.
//!
//! # Two kinds of fact
//!
//! Most features come straight from state the scheduler already holds. The
//! rest are transitions only the recorder observes, or that today's state does
//! not keep at all; they are passed in as [`Recorded`], which the mechanism's
//! recorder fills. They are: when a request was admitted, when it last emitted
//! a token, how many prompt tokens the prefix cache held for it when it was
//! admitted, the host-prefetch deepening of its prefix hit, and the rank
//! recorded for its cache resume (the control thread's routing table, not the
//! scheduler's).
//!
//! # What a scheduler cannot tell a view
//!
//! - `Room::graph_tiers` is the mechanism's: it holds the build config's
//!   captured tiers. It builds the facts with [`room_facts`], fills the tiers,
//!   and turns them into the `Room` with [`room_view`]; [`room`] leaves them
//!   empty. (`Room::warming` is read here, from the claimed precopy pages.)
//! - `Snapshot::ranks` and a trial's `arrival` need the ranks' load, which the
//!   routing table on the control thread holds and a scheduler does not: a
//!   scheduler knows one rank's load, its own. [`trial_state`] therefore trips
//!   `place` and not `route`; [`trial_state_with_ranks`] takes the ranks and
//!   trips both.
//! - `PageView::last_used` is informational: the block pool keeps an allocation
//!   order, not use timestamps. A policy keeps its own LRU from `Cached`, `Hit`
//!   and `Freed`.
//!
//! These report nothing until the subsystem that owns them exposes them, and
//! are open items on the board:
//!
//! - `Snapshot::encodings`: `EncoderCacheManager` lists no entries, so no
//!   encoding is ever a live `EncodingId`.
//! - host KV pages (`PageView::on_host`): host KV pages live in the storage
//!   servers, and the engine's offload manager holds only per-worker pools and
//!   fetch tags, so it lists none; no `PageId` is a live host page and a
//!   `HostPages` shortage has no page victim until the storage side reports its
//!   pages (an open item with `front`).
//! - `ExpertLoad`: no routed-token counters reach the scheduler, so `experts`
//!   is skipped rather than asked about a load nobody measured.

use std::collections::{HashMap, HashSet};

use scheduling::guest::decision::{
    Inflight as InflightView, Room, Shortage, Storage, Warming as WarmingView,
};
use scheduling::guest::event::{End, Event};
use scheduling::guest::ids::{
    ComponentId, EncodingId, PageId, PhaseId, RequestId as ViewRequestId,
};
use scheduling::guest::time::Nanos;
use scheduling::guest::view::{
    CallView, EncodingView, ExpertLoad, PageView, PhaseKind, PhaseView, RankView, ReplicaView,
    RequestClass, RequestState as Standing, RequestView, Snapshot,
};

use super::*;
use crate::scheduler::mechanism::Live;
use crate::scheduler::placement::Placement;
use crate::scheduler::policy::TrialState;

// ---------------------------------------------------------------------------
// The epoch's phases
// ---------------------------------------------------------------------------

/// Every phase kind the interface distinguishes, in the order the ids number
/// them.
///
/// A phase id is the position of its kind here, so it means the same thing in
/// every epoch: `PhaseKind` appends a kind and never renumbers one, and
/// [`phase_id`] is exhaustive so a new kind cannot be forgotten.
const PHASES: [PhaseKind; 21] = [
    PhaseKind::TextEncoding,
    PhaseKind::VisionEncoding,
    PhaseKind::LatentEncoding,
    PhaseKind::Prefill,
    PhaseKind::Decode,
    PhaseKind::Verify,
    PhaseKind::TokenDenoise,
    PhaseKind::Readout,
    PhaseKind::ImageDecode,
    PhaseKind::CacheInstall,
    PhaseKind::MediaRead,
    PhaseKind::Denoise,
    PhaseKind::VideoDecode,
    PhaseKind::AudioDecode,
    PhaseKind::VideoEncode,
    PhaseKind::AudioEncode,
    PhaseKind::Mux,
    PhaseKind::LatentPrepare,
    PhaseKind::TensorTransfer,
    PhaseKind::CacheExport,
    PhaseKind::Refill,
];

/// The id of a phase kind.
///
/// [`PhaseKind::Other`] is a policy-side placeholder for a kind a policy does
/// not know; the mechanism never names a request's phase with it, and it has no
/// id.
pub(crate) const fn phase_id(kind: PhaseKind) -> PhaseId {
    PhaseId(match kind {
        PhaseKind::TextEncoding => 0,
        PhaseKind::VisionEncoding => 1,
        PhaseKind::LatentEncoding => 2,
        PhaseKind::Prefill => 3,
        PhaseKind::Decode => 4,
        PhaseKind::Verify => 5,
        PhaseKind::TokenDenoise => 6,
        PhaseKind::Readout => 7,
        PhaseKind::ImageDecode => 8,
        PhaseKind::CacheInstall => 9,
        PhaseKind::MediaRead => 10,
        PhaseKind::Denoise => 11,
        PhaseKind::VideoDecode => 12,
        PhaseKind::AudioDecode => 13,
        PhaseKind::VideoEncode => 14,
        PhaseKind::AudioEncode => 15,
        PhaseKind::Mux => 16,
        PhaseKind::LatentPrepare => 17,
        PhaseKind::TensorTransfer => 18,
        PhaseKind::CacheExport => 19,
        PhaseKind::Refill => 20,
        PhaseKind::Other => u32::MAX,
    })
}

/// The phase kind an id names; `None` for an id this epoch has no phase for,
/// which no `Plan` may name.
pub(crate) fn phase_kind(id: PhaseId) -> Option<PhaseKind> {
    PHASES.get(usize::try_from(id.0).ok()?).copied()
}

/// The phase a call of `kind` runs.
///
/// The call kinds are the phases' vocabulary: every `CallKind` is one phase,
/// except that a refill chunk is planned as a prefill call and named by its
/// request's draft state instead ([`next_phase`]), and that a readout is a
/// token-denoising call of a readout request.
pub(crate) fn call_phase(kind: CallKind) -> PhaseKind {
    match kind {
        CallKind::Forward(ForwardMode::Prefill) => PhaseKind::Prefill,
        CallKind::Forward(ForwardMode::Decode) => PhaseKind::Decode,
        CallKind::Forward(ForwardMode::Verify) => PhaseKind::Verify,
        CallKind::Forward(ForwardMode::TokenDenoising) => PhaseKind::TokenDenoise,
        CallKind::Media(MediaCall::TextEncoding) => PhaseKind::TextEncoding,
        CallKind::Media(MediaCall::VisionEncoding) => PhaseKind::VisionEncoding,
        CallKind::Media(MediaCall::LatentEncoding) => PhaseKind::LatentEncoding,
        CallKind::Media(MediaCall::Denoising) => PhaseKind::Denoise,
        CallKind::Media(MediaCall::ImageDecoding) => PhaseKind::ImageDecode,
        CallKind::Media(MediaCall::VideoDecoding) => PhaseKind::VideoDecode,
        CallKind::Media(MediaCall::AudioDecoding) => PhaseKind::AudioDecode,
        CallKind::Media(MediaCall::VideoEncoding) => PhaseKind::VideoEncode,
        CallKind::Media(MediaCall::AudioEncoding) => PhaseKind::AudioEncode,
        CallKind::Media(MediaCall::MediaReading) => PhaseKind::MediaRead,
        CallKind::Media(MediaCall::LatentPreparation) => PhaseKind::LatentPrepare,
        CallKind::Media(MediaCall::Muxing) => PhaseKind::Mux,
        CallKind::Transfer(TransferMode::Tensor) => PhaseKind::TensorTransfer,
        CallKind::Transfer(TransferMode::CacheExport) => PhaseKind::CacheExport,
        CallKind::Transfer(TransferMode::CacheInstall) => PhaseKind::CacheInstall,
    }
}

/// The id of the phase a call of `kind` runs.
pub(crate) fn call_phase_id(kind: CallKind) -> PhaseId {
    phase_id(call_phase(kind))
}

/// The call kind that runs a phase, when the phase is a call.
///
/// A refill is a prefill call the mechanism plans for a decoding request, so it
/// answers `Forward(Prefill)`; `Other` runs nothing.
pub(crate) fn phase_call(kind: PhaseKind) -> Option<CallKind> {
    Some(match kind {
        PhaseKind::TextEncoding => CallKind::Media(MediaCall::TextEncoding),
        PhaseKind::VisionEncoding => CallKind::Media(MediaCall::VisionEncoding),
        PhaseKind::LatentEncoding => CallKind::Media(MediaCall::LatentEncoding),
        PhaseKind::Prefill | PhaseKind::Refill => CallKind::Forward(ForwardMode::Prefill),
        PhaseKind::Decode => CallKind::Forward(ForwardMode::Decode),
        PhaseKind::Verify => CallKind::Forward(ForwardMode::Verify),
        PhaseKind::TokenDenoise | PhaseKind::Readout => {
            CallKind::Forward(ForwardMode::TokenDenoising)
        }
        PhaseKind::ImageDecode => CallKind::Media(MediaCall::ImageDecoding),
        PhaseKind::CacheInstall => CallKind::Transfer(TransferMode::CacheInstall),
        PhaseKind::MediaRead => CallKind::Media(MediaCall::MediaReading),
        PhaseKind::Denoise => CallKind::Media(MediaCall::Denoising),
        PhaseKind::VideoDecode => CallKind::Media(MediaCall::VideoDecoding),
        PhaseKind::AudioDecode => CallKind::Media(MediaCall::AudioDecoding),
        PhaseKind::VideoEncode => CallKind::Media(MediaCall::VideoEncoding),
        PhaseKind::AudioEncode => CallKind::Media(MediaCall::AudioEncoding),
        PhaseKind::Mux => CallKind::Media(MediaCall::Muxing),
        PhaseKind::LatentPrepare => CallKind::Media(MediaCall::LatentPreparation),
        PhaseKind::TensorTransfer => CallKind::Transfer(TransferMode::Tensor),
        PhaseKind::CacheExport => CallKind::Transfer(TransferMode::CacheExport),
        PhaseKind::Other => return None,
    })
}

/// The blocks of a worker group: the component names its replicas bind, sorted
/// so a [`ComponentId`] means the same component in every view of the epoch.
pub(crate) fn blocks(info: &WorkerInfo) -> Vec<String> {
    let mut names: Vec<String> = info
        .components
        .iter()
        .map(|component| component.name.clone())
        .collect();
    names.extend(info.media_components.values().cloned());
    names.sort();
    names.dedup();
    names
}

/// The id of a block name in this epoch's table.
pub(crate) fn block_id(blocks: &[String], name: &str) -> Option<ComponentId> {
    blocks
        .iter()
        .position(|block| block == name)
        .map(|index| ComponentId(u32::try_from(index).unwrap_or(u32::MAX)))
}

/// The blocks of every worker of the world: the table `ComponentId`s number, so
/// a phase's block and a replica's block name the same component.
pub(crate) fn world_blocks(executor: &dyn Executor) -> Vec<String> {
    let mut names = Vec::new();
    for (_, info) in &executor.info().workers {
        names.extend(blocks(info));
    }
    names.sort();
    names.dedup();
    names
}

/// The block that runs `kind` on this worker group: the component the worker
/// reports for its media call, or the protocol's default component.
fn block_of(table: &[String], info: &WorkerInfo, kind: PhaseKind) -> ComponentId {
    let name = match phase_call(kind) {
        Some(CallKind::Media(call)) => info
            .media_components
            .get(&call)
            .map_or(DEFAULT_COMPONENT, String::as_str),
        _ => DEFAULT_COMPONENT,
    };
    block_id(table, name).unwrap_or(ComponentId(0))
}

/// Every phase this worker group can run, in id order.
///
/// `Snapshot::phases`, built once per epoch: a policy reads a request's
/// `PhaseId` here to learn its kind and its block.
pub(crate) fn phases(info: &WorkerInfo) -> Vec<PhaseView> {
    phases_in(info, &blocks(info))
}

/// Every phase this worker group can run, numbered in the world's block table.
pub(crate) fn phases_in(info: &WorkerInfo, table: &[String]) -> Vec<PhaseView> {
    PHASES
        .iter()
        .copied()
        .filter(|kind| {
            // A refill is prefill work: it runs wherever a prefill chunk does.
            match phase_call(*kind) {
                Some(call) => info.supported_calls.contains(&call),
                None => false,
            }
        })
        .map(|kind| PhaseView {
            id: phase_id(kind),
            kind,
            block: block_of(table, info, kind),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// The bit that marks a page id as a host page's.
const HOST_PAGE: u64 = 1 << 63;

/// What a [`PageId`] names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PageTarget {
    /// A device page, by the first unit of the page.
    Device(u32),
    /// A host KV page, by its index in the host pool.
    Host(u64),
}

/// The page id of the device page whose first unit is `unit`.
pub(crate) const fn device_page(unit: u32) -> PageId {
    PageId(unit as u64)
}

/// The page id of the host KV page with index `index`.
pub(crate) const fn host_page(index: u64) -> PageId {
    PageId(HOST_PAGE | index)
}

/// What `page` names; `None` for an id of neither space.
pub(crate) const fn page_target(page: PageId) -> Option<PageTarget> {
    if page.0 & HOST_PAGE != 0 {
        Some(PageTarget::Host(page.0 & !HOST_PAGE))
    } else if page.0 <= u32::MAX as u64 {
        Some(PageTarget::Device(page.0 as u32))
    } else {
        None
    }
}

/// The device unit `page` names; `None` for a host page or a foreign id.
pub(crate) const fn unit_of(page: PageId) -> Option<u32> {
    match page_target(page) {
        Some(PageTarget::Device(unit)) => Some(unit),
        Some(PageTarget::Host(_)) | None => None,
    }
}

/// The encoding id of an encoder-cache key: the first eight bytes of its
/// digest, which is what a policy compares and remembers.
pub(crate) fn encoding_id(key: &[u8; 32]) -> EncodingId {
    let mut head = [0u8; 8];
    head.copy_from_slice(&key[..8]);
    EncodingId(u64::from_be_bytes(head))
}

// ---------------------------------------------------------------------------
// Views from facts
// ---------------------------------------------------------------------------

/// The features of one request, as [`request_view`] turns them into a view.
#[derive(Clone, Debug)]
pub(crate) struct RequestFacts {
    pub id: ViewRequestId,
    pub standing: Standing,
    pub phase: Option<PhaseId>,
    pub priority: i32,
    pub arrived_at: Nanos,
    pub admitted_at: Option<Nanos>,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub cached_prefix_tokens: u32,
    pub prefilled_tokens: u32,
    pub cached_pages: u32,
    pub last_emitted_at: Option<Nanos>,
    pub class: RequestClass,
    pub waiting_transfer: bool,
    pub next_tokens: u32,
    pub reservation_units: u64,
    pub prefetch_tokens: u32,
    pub refill_tokens: Option<u32>,
    pub remaining_tokens: u32,
    pub migratable: bool,
    pub cache_rank: Option<usize>,
}

/// The view of one request.
pub(crate) fn request_view(facts: RequestFacts) -> RequestView {
    RequestView {
        id: facts.id,
        state: facts.standing,
        phase: facts.phase,
        priority: facts.priority,
        arrived_at: facts.arrived_at,
        admitted_at: facts.admitted_at,
        prompt_tokens: facts.prompt_tokens,
        generated_tokens: facts.generated_tokens,
        cached_prefix_tokens: facts.cached_prefix_tokens,
        prefilled_tokens: facts.prefilled_tokens,
        cached_pages: facts.cached_pages,
        last_emitted_at: facts.last_emitted_at,
        class: facts.class,
        waiting_transfer: facts.waiting_transfer,
        next_tokens: facts.next_tokens,
        reservation_units: facts.reservation_units,
        prefetch_tokens: facts.prefetch_tokens,
        refill_tokens: facts.refill_tokens,
        remaining_tokens: facts.remaining_tokens,
        migratable: facts.migratable,
        cache_rank: facts.cache_rank,
    }
}

/// The features of one page, as [`page_view`] turns them into a view.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PageFacts {
    pub id: PageId,
    pub tokens: u32,
    pub last_used: Nanos,
    pub on_host: bool,
    pub evictable: bool,
}

/// The view of one page.
pub(crate) fn page_view(facts: PageFacts) -> PageView {
    PageView {
        id: facts.id,
        tokens: facts.tokens,
        last_used: facts.last_used,
        on_host: facts.on_host,
        evictable: facts.evictable,
    }
}

/// The view of one encoder cache entry.
pub(crate) fn encoding_view(id: EncodingId, last_used: Nanos) -> EncodingView {
    EncodingView { id, last_used }
}

/// The features of one replica, as [`replica_view`] turns them into a view.
#[derive(Clone, Debug)]
pub(crate) struct ReplicaFacts {
    pub index: usize,
    pub block: ComponentId,
    pub kinds: Vec<PhaseKind>,
    pub resident_requests: u32,
    pub capacity: u32,
    pub holds_request: bool,
    pub prefills_only: bool,
    pub holds_component: bool,
}

/// The view of one replica.
pub(crate) fn replica_view(facts: ReplicaFacts) -> ReplicaView {
    ReplicaView {
        index: facts.index,
        block: facts.block,
        kinds: facts.kinds,
        resident_requests: facts.resident_requests,
        capacity: facts.capacity,
        holds_request: facts.holds_request,
        prefills_only: facts.prefills_only,
        holds_component: facts.holds_component,
    }
}

/// The features of one step's room, as [`room_view`] turns them into a `Room`.
#[derive(Clone, Debug, Default)]
pub(crate) struct RoomFacts {
    pub step: u64,
    pub kv_units: u64,
    pub slots: usize,
    pub token_budget: u32,
    pub call_budget: usize,
    pub graph_tiers: Vec<u32>,
    pub page_tokens: u32,
    pub prefill_tokens: Option<u32>,
    pub batches: u32,
    pub prefill_batches: u32,
    pub capacity_units: u64,
    pub warming: Vec<WarmingView>,
}

/// This step's room.
pub(crate) fn room_view(facts: RoomFacts) -> Room {
    Room {
        step: facts.step,
        kv_units: facts.kv_units,
        slots: facts.slots,
        token_budget: facts.token_budget,
        call_budget: facts.call_budget,
        graph_tiers: facts.graph_tiers,
        page_tokens: facts.page_tokens,
        prefill_tokens: facts.prefill_tokens,
        inflight: InflightView {
            batches: facts.batches,
            prefill_batches: facts.prefill_batches,
        },
        capacity_units: facts.capacity_units,
        warming: facts.warming,
    }
}

/// The call a `place` decision is about.
pub(crate) fn call_view(request: RequestKey, kind: CallKind) -> CallView {
    let phase = call_phase(kind);
    CallView {
        request: ViewRequestId(request.request_id.0),
        phase: phase_id(phase),
        kind: phase,
    }
}

/// A shortage of one storage class.
pub(crate) const fn shortage(kind: Storage, need: u64) -> Shortage {
    Shortage { kind, need }
}

// ---------------------------------------------------------------------------
// Facts only the recorder keeps
// ---------------------------------------------------------------------------

/// The features of a request that are transitions, not state: the recorder
/// observes them as they happen and hands them back here.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Recorded {
    /// When the mechanism admitted the request; `None` while it is queued.
    pub admitted_at: Option<Nanos>,
    /// When it last emitted a token to its client.
    pub last_emitted_at: Option<Nanos>,
    /// Prompt tokens the prefix cache held for it when it was admitted.
    pub cached_prefix_tokens: u32,
    /// Tokens the host KV cache could add to that hit (the deepening a prefetch
    /// would gain).
    pub prefetch_tokens: u32,
    /// The rank the control thread's routing table records for its cache
    /// resume, when it resumes a stored cache.
    pub cache_rank: Option<usize>,
}

/// What the recorder observed of each request, by request id.
pub(crate) type RecordedTable = HashMap<RequestId, Recorded>;

// ---------------------------------------------------------------------------
// Wall-clock to monotonic time
// ---------------------------------------------------------------------------

/// The monotonic instant of a wall-clock time in fractional Unix seconds.
///
/// Requests carry `queued_at` as Unix time; a policy compares instants of the
/// mechanism's monotonic clock, which the process fixes at its first use.
/// Times before this process started clamp to zero.
pub(crate) fn nanos_at(unix_secs: f64) -> Nanos {
    let now_unix = crate::clock::now_unix_secs();
    let now_monotonic = crate::clock::now_monotonic_us() as f64 / 1e6;
    let elapsed = unix_secs - (now_unix - now_monotonic);
    Nanos((elapsed.max(0.0) * 1e9) as u64)
}

// ---------------------------------------------------------------------------
// One step's room
// ---------------------------------------------------------------------------

/// Free request slots: the sequence budget of the next admission.
///
/// Admission stops at either bound today — the configured count of running
/// requests, and the worker's request rows, which a running token request holds
/// through its allocations (`Storage::request_pool`). A slot is free only when
/// both have room, so the room is their minimum.
fn free_slots(scheduler: &Scheduler) -> usize {
    let held_rows = scheduler
        .running
        .values()
        .filter(|state| state.allocations().is_some())
        .count();
    let by_rows = (scheduler.info.request_slots as usize).saturating_sub(held_rows);
    let by_requests = scheduler
        .config
        .max_running_requests
        .saturating_sub(scheduler.running_request_count());
    by_rows.min(by_requests)
}

/// This step's room.
///
/// The captured graph tiers and the precopy's warming order are the
/// mechanism's, not the scheduler's; it builds the facts, fills those two, and
/// turns them into the `Room` with `room_view`.
pub(crate) fn room(scheduler: &Scheduler, step: u64) -> Room {
    room_view(room_facts(scheduler, step))
}

/// The facts of this step's room.
pub(crate) fn room_facts(scheduler: &Scheduler, step: u64) -> RoomFacts {
    let page_tokens = scheduler
        .storage
        .cache()
        .map_or(1, |cache| cache.coordinator.max_block_size());
    RoomFacts {
        step,
        kv_units: scheduler.storage.free_units() as u64,
        slots: free_slots(scheduler),
        token_budget: u32::try_from(scheduler.config.max_num_batched_tokens).unwrap_or(u32::MAX),
        call_budget: scheduler.config.max_batch,
        // A worker fits a batch into its captured graphs itself; no tier list
        // reaches the scheduler.
        graph_tiers: Vec::new(),
        page_tokens: u32::try_from(page_tokens).unwrap_or(u32::MAX),
        prefill_tokens: scheduler
            .warming
            .prefill_tokens
            .map(|tokens| u32::try_from(tokens).unwrap_or(u32::MAX)),
        batches: u32::try_from(scheduler.inflight.pending_batches.len()).unwrap_or(u32::MAX),
        prefill_batches: u32::try_from(
            scheduler
                .inflight
                .pending_batches
                .values()
                .filter(|batch| batch.prefill)
                .count(),
        )
        .unwrap_or(u32::MAX),
        capacity_units: scheduler.storage.capacity_units() as u64,
        warming: warming(scheduler),
    }
}

/// The requests an epoch's precopy warms, in the order the mechanism claimed
/// them, each with its pages in prefix order.
///
/// The claim order is the reservation list's own: each reserved page names the
/// cache resume it serves, and a request's pages keep the order they were
/// claimed in.
fn warming(scheduler: &Scheduler) -> Vec<WarmingView> {
    let mut views: Vec<WarmingView> = Vec::new();
    for reserved in &scheduler.warming.reserved {
        let Some(unit) = reserved.reference.units().first().copied() else {
            continue;
        };
        let request = ViewRequestId(reserved.resume.request_id);
        let page = device_page(unit.0);
        match views.iter_mut().find(|view| view.request == request) {
            Some(view) => view.pages.push(page),
            None => views.push(WarmingView {
                request,
                pages: vec![page],
            }),
        }
    }
    views
}

/// The storage the mechanism finds short right now, for `reclaim`.
///
/// The mechanism knows the shortfall when executing a plan hits it; a step that
/// hit none has nothing to reclaim.
pub(crate) fn shortage_now(scheduler: &Scheduler) -> Option<Shortage> {
    (!scheduler.starved.is_empty()).then_some(shortage(Storage::KvUnits, 1))
}

// ---------------------------------------------------------------------------
// One request
// ---------------------------------------------------------------------------

/// The phase of a request's next call, when it has one ready.
///
/// A refill chunk is a prefill call named by the request's draft state, and a
/// readout request's token-denoising call is the readout phase.
fn next_phase(scheduler: &Scheduler, state: &RequestState, id: RequestId) -> Option<PhaseId> {
    let kind = scheduler.peek_next_call_variant(id)?;
    let phase = match kind {
        CallKind::Forward(ForwardMode::TokenDenoising) if state.req.is_readout() => {
            PhaseKind::Readout
        }
        // A refill chunk is the policy's decision alone (`Plan::refill`), and
        // the mechanism builds its call: the request's phase is the serving
        // phase it needs next, and its draft gap appears in `refill_tokens`.
        // `PhaseKind::Refill` stays in the phase table as the kind of the call
        // the mechanism builds, not as a phase a request sits in.
        CallKind::Forward(ForwardMode::Prefill) if refill_gap(scheduler, state).is_some() => {
            PhaseKind::Decode
        }
        other => call_phase(other),
    };
    Some(phase_id(phase))
}

/// The token charge of a request's next call, when that call is not a prefill
/// or refill chunk: the mechanism's own cost function, read without planning
/// the call.
fn next_tokens(scheduler: &Scheduler, state: &RequestState, id: RequestId) -> u32 {
    match scheduler.peek_next_call_variant(id) {
        // A prefill or refill chunk is charged the chunk the policy sizes.
        Some(CallKind::Forward(ForwardMode::Prefill)) | None => 0,
        Some(CallKind::Forward(ForwardMode::Decode)) => 1,
        Some(CallKind::Forward(ForwardMode::Verify)) => {
            let drafts = scheduler.verify_drafts(id, state.cache_visible_len).len();
            u32::try_from(1 + drafts).unwrap_or(u32::MAX)
        }
        Some(CallKind::Forward(ForwardMode::TokenDenoising)) if state.req.is_readout() => {
            u32::try_from(state.req.readout.len()).unwrap_or(u32::MAX)
        }
        Some(CallKind::Forward(ForwardMode::TokenDenoising)) => {
            let units = scheduler.worker_image_latent_units_for(state).max(1);
            let branches = u64::from(state.req.image.cfg_branch_count().max(1));
            u32::try_from(units.saturating_mul(branches)).unwrap_or(u32::MAX)
        }
        Some(CallKind::Media(_)) => 1,
        Some(CallKind::Transfer(_)) => 0,
    }
}

/// The worst-case device units a request reserves when it is admitted, or 0
/// when it reserves nothing.
fn reservation_units(scheduler: &Scheduler, state: &RequestState) -> u64 {
    let Some(cache) = scheduler.storage.cache() else {
        return 0;
    };
    if !state.reserve_worstcase {
        return 0;
    }
    cache
        .coordinator
        .units_for_tokens(state.max_reserved_cache_tokens) as u64
}

/// The gap in the request's draft cache: the tokens a refill would evaluate,
/// or `None` when there is none.
///
/// A fact, not a judgement: whether a refill is worth its cost is the policy's
/// to decide (`Plan::refill`). Both `refill_tokens` and the `Refill` phase read
/// it here.
fn refill_gap(scheduler: &Scheduler, state: &RequestState) -> Option<u32> {
    if !scheduler.speculates() {
        return None;
    }
    // The position the refill evaluates up to: the first the request wrote
    // itself after the gap, or the visible length (`DraftCache::refill_end`,
    // which is private to the speculation module).
    let visible = state.cache_visible_len;
    let end = state
        .draft
        .tail
        .map_or(visible, |(start, _)| start.min(visible));
    let gap = end.saturating_sub(state.draft.valid);
    (gap > 0).then_some(gap)
}

/// The tokens a refill of the request's draft cache would cost, or `None` when
/// its draft cache has no gap.
fn refill_tokens(scheduler: &Scheduler, state: &RequestState) -> Option<u32> {
    refill_gap(scheduler, state)
}

/// The KV pages a request holds.
fn cached_pages(state: &RequestState) -> u32 {
    state.allocations().map_or(0, |allocations| {
        allocations
            .block_tables()
            .iter()
            .map(|table| {
                u32::try_from(table.end_page().saturating_sub(table.start_page()))
                    .unwrap_or(u32::MAX)
            })
            .sum()
    })
}

/// Whether a resident request waits for a transfer right now.
fn waiting_transfer(scheduler: &Scheduler, id: RequestId) -> bool {
    scheduler
        .inflight
        .pending_calls
        .get(&id)
        .is_some_and(|calls| {
            calls
                .iter()
                .any(|call| call.call.bounds.max_transfer_bytes > 0)
        })
}

/// The class of a request, as scheduling distinguishes it.
fn request_class(req: &GenerationRequest) -> RequestClass {
    if req.is_readout() {
        RequestClass::Readout
    } else if req.decodes_text() {
        RequestClass::Token
    } else {
        RequestClass::Media
    }
}

/// The view of a token request the scheduler holds.
pub(crate) fn request(
    scheduler: &Scheduler,
    state: &RequestState,
    standing: Standing,
    recorded: &Recorded,
) -> RequestView {
    let id = state.req.request_id;
    // The next prefill chunk starts at whichever is further on: the prompt
    // tokens the runtime computed, or the prefix the cache already held for it
    // at admission.
    let prefilled = state
        .num_computed_prompt_tokens
        .max(recorded.cached_prefix_tokens);
    request_view(RequestFacts {
        id: ViewRequestId(state.req.request_id.0),
        standing,
        phase: next_phase(scheduler, state, id),
        priority: state.req.priority,
        arrived_at: nanos_at(state.queued_at),
        admitted_at: recorded.admitted_at,
        prompt_tokens: u32::try_from(state.prompt_tokens).unwrap_or(u32::MAX),
        generated_tokens: u32::try_from(state.num_generated_tokens).unwrap_or(u32::MAX),
        cached_prefix_tokens: recorded.cached_prefix_tokens,
        prefilled_tokens: prefilled,
        cached_pages: cached_pages(state),
        last_emitted_at: recorded.last_emitted_at,
        class: request_class(&state.req),
        waiting_transfer: waiting_transfer(scheduler, id),
        next_tokens: next_tokens(scheduler, state, id),
        reservation_units: reservation_units(scheduler, state),
        prefetch_tokens: recorded.prefetch_tokens,
        refill_tokens: refill_tokens(scheduler, state),
        remaining_tokens: u32::try_from(
            state
                .req
                .max_und_tokens
                .saturating_sub(state.num_generated_tokens),
        )
        .unwrap_or(u32::MAX),
        migratable: state.req.is_migratable(),
        cache_rank: recorded.cache_rank,
    })
}

/// The phase of a resident media request's next ready call, when its graph has
/// one ready.
fn media_phase(scheduler: &Scheduler, id: RequestId) -> Option<PhaseId> {
    scheduler
        .media_ready_calls(id)
        .first()
        .map(|kind| call_phase_id(*kind))
}

/// The view of a media (diffusion) request the scheduler holds.
///
/// A media request's progress is its call graph's: `phase` is the phase of the
/// graph's first ready call, which is what the mechanism would run next.
pub(crate) fn diffusion_request(
    request: &DiffusionRequest,
    standing: Standing,
    queued_at: f64,
    phase: Option<PhaseId>,
    recorded: &Recorded,
) -> RequestView {
    request_view(RequestFacts {
        id: ViewRequestId(request.request_id.0),
        standing,
        phase,
        priority: request.priority,
        arrived_at: nanos_at(queued_at),
        admitted_at: recorded.admitted_at,
        prompt_tokens: u32::try_from(request.prompt_token_ids.len()).unwrap_or(u32::MAX),
        generated_tokens: 0,
        cached_prefix_tokens: 0,
        prefilled_tokens: 0,
        cached_pages: 0,
        last_emitted_at: recorded.last_emitted_at,
        class: RequestClass::Media,
        waiting_transfer: false,
        next_tokens: 1,
        reservation_units: 0,
        prefetch_tokens: 0,
        refill_tokens: None,
        remaining_tokens: 0,
        migratable: false,
        cache_rank: recorded.cache_rank,
    })
}

/// Every request the scheduler holds, in the order a policy reads them: queued
/// ones in admission order, then resident ones in admission order.
pub(crate) fn requests(scheduler: &Scheduler, recorded: &RecordedTable) -> Vec<RequestView> {
    let empty = Recorded::default();
    let mut views = Vec::with_capacity(
        scheduler.waiting.len() + scheduler.running.len() + scheduler.waiting_media.len(),
    );
    for state in &scheduler.waiting {
        let facts = recorded
            .get(&state.req.request_id)
            .copied()
            .unwrap_or(empty);
        views.push(request(scheduler, state, Standing::Queued, &facts));
    }
    for pending in &scheduler.waiting_media {
        let facts = recorded
            .get(&pending.request.request_id)
            .copied()
            .unwrap_or(empty);
        views.push(diffusion_request(
            &pending.request,
            Standing::Queued,
            pending.queued_at,
            None,
            &facts,
        ));
    }
    let mut seen = HashSet::new();
    for id in &scheduler.running_order {
        if !seen.insert(*id) {
            continue;
        }
        let facts = recorded.get(id).copied().unwrap_or(empty);
        if let Some(state) = scheduler.running.get(id) {
            views.push(request(scheduler, state, Standing::Resident, &facts));
        } else if let Some(flow) = scheduler.running_media.get(id) {
            // A media request keeps no arrival time of its own once it runs.
            views.push(diffusion_request(
                &flow.request,
                Standing::Resident,
                0.0,
                media_phase(scheduler, *id),
                &facts,
            ));
        }
    }
    // A media request the admission order does not name still exists.
    for id in scheduler.running_media.keys() {
        let Some(flow) = scheduler.running_media.get(id) else {
            continue;
        };
        if !seen.insert(*id) {
            continue;
        }
        let facts = recorded.get(id).copied().unwrap_or(empty);
        views.push(diffusion_request(
            &flow.request,
            Standing::Resident,
            0.0,
            media_phase(scheduler, *id),
            &facts,
        ));
    }
    views
}

// ---------------------------------------------------------------------------
// Pages and encodings
// ---------------------------------------------------------------------------

/// The units of every page a resident request continues from.
fn held_units(scheduler: &Scheduler) -> HashSet<u32> {
    scheduler
        .running
        .values()
        .filter_map(|state| state.allocations())
        .flat_map(|allocations| allocations.block_tables().iter())
        .flat_map(|table| table.unit_ids())
        .map(|unit| unit.0)
        .collect()
}

/// Every page of the device KV cache: the pages the prefix cache holds for
/// reuse, each evictable while no resident request continues from it.
///
/// The block pool publishes pages as it completes them; free units are not
/// pages yet, and the pool keeps no use timestamps, so `last_used` is zero for
/// all of them.
pub(crate) fn pages(scheduler: &Scheduler) -> Vec<PageView> {
    let Some(cache) = scheduler.storage.cache() else {
        return Vec::new();
    };
    let held = held_units(scheduler);
    let mut seen = HashSet::new();
    let mut views = Vec::new();
    for page in cache.block_pool.export() {
        let Some(first) = page.units.first().copied() else {
            continue;
        };
        if !seen.insert(first) {
            // One page published under several resumes.
            continue;
        }
        views.push(page_view(PageFacts {
            id: device_page(first),
            tokens: u32::try_from(page.tokens.len()).unwrap_or(u32::MAX),
            last_used: Nanos(0),
            on_host: false,
            // A page is evictable once no resident request continues from any
            // of its units, not just from the one its id names.
            evictable: !page.units.iter().any(|unit| held.contains(unit)),
        }));
    }
    views
}

/// The view of one pool page, from the mechanism's own record of it.
///
/// `units` are all the page's units and `tokens` the tokens it holds: what a
/// cached page carries, and what a change log keeps of a page it freed. The
/// evictability rule is [`pages`]'s — no resident request continues from any of
/// the page's units — so a policy is never told a page is evictable that the
/// mechanism would refuse to evict, and this reads one page rather than the
/// cache.
pub(crate) fn page_of(scheduler: &Scheduler, units: &[u32], tokens: u32) -> Option<PageView> {
    let first = *units.first()?;
    let held = held_units(scheduler);
    Some(page_view(PageFacts {
        id: device_page(first),
        tokens,
        last_used: Nanos(0),
        on_host: false,
        // A page is evictable once no resident request continues from any of
        // its units, not just from the one its id names.
        evictable: !units.iter().any(|unit| held.contains(unit)),
    }))
}

/// Every page of the device KV cache, keyed by id: the table `Live` checks a
/// named page against.
///
/// Named so the cost is visible: it walks the whole prefix cache, so it belongs
/// where a snapshot is built (once per epoch), not in a step. The pages that
/// appear and go reach a policy as `Cached` and `Freed`, which is the per-step
/// source, and the mechanism asks the pool itself whether a named page still
/// exists.
pub(crate) fn page_index(scheduler: &Scheduler) -> std::collections::BTreeMap<PageId, PageView> {
    pages(scheduler)
        .into_iter()
        .map(|page| (page.id, page))
        .collect()
}

/// Every live encoder cache entry.
///
/// Each entry is named by [`encoding_id`] of its key and carries the tick the
/// cache last used it, which is the order its own eviction follows. A pinned
/// entry is listed too: it exists, and whether it may be freed is the eviction
/// call's answer, not the view's.
pub(crate) fn encodings(scheduler: &Scheduler) -> Vec<EncodingView> {
    scheduler
        .storage
        .encoder_cache
        .entries()
        .map(|(key, last_used)| encoding_view(encoding_id(&key), Nanos(last_used)))
        .collect()
}

// ---------------------------------------------------------------------------
// Replicas, ranks and experts
// ---------------------------------------------------------------------------

/// The distinct requests resident on one worker, by the request affinity the
/// mechanism records when it places a call.
fn resident_requests(placement: &Placement, worker: &crate::WorkerGroupId) -> u32 {
    let mut requests = HashSet::new();
    for ((key, _), owner) in &placement.affinity {
        if owner == worker {
            requests.insert(*key);
        }
    }
    for ((key, _), owner) in &placement.decode_affinity {
        if owner == worker {
            requests.insert(*key);
        }
    }
    u32::try_from(requests.len()).unwrap_or(u32::MAX)
}

/// Whether a replica already holds another component of `request`.
fn holds_component(
    placement: &Placement,
    request: RequestKey,
    worker: &crate::WorkerGroupId,
) -> bool {
    placement
        .affinity
        .iter()
        .chain(&placement.decode_affinity)
        .any(|((key, _), owner)| *key == request && owner == worker)
}

/// Whether a replica holds this request's state for the component the call
/// binds.
fn holds_request(
    placement: &Placement,
    request: RequestKey,
    bound: &str,
    decode_stage: bool,
    worker: &crate::WorkerGroupId,
) -> bool {
    let pinned = if decode_stage {
        &placement.decode_affinity
    } else {
        &placement.affinity
    };
    pinned
        .get(&(request, bound.to_owned()))
        .is_some_and(|owner| owner == worker)
}

/// The phase kinds a worker can run.
fn worker_phases(info: &WorkerInfo) -> Vec<PhaseKind> {
    let mut kinds: Vec<PhaseKind> = info
        .supported_calls
        .iter()
        .copied()
        .map(call_phase)
        .collect();
    kinds.sort_by_key(|kind| phase_id(*kind).0);
    kinds.dedup();
    kinds
}

/// The replicas a `place` decision may choose for a request's call of `kind`,
/// in the order the decision's answer indexes.
///
/// The candidates are the workers that serve the call's component, minus the
/// prefill-only workers for a decode-stage call: today's
/// `Placement::component_target` rule, presented instead of decided.
pub(crate) fn replicas(
    scheduler: &Scheduler,
    request: RequestKey,
    kind: CallKind,
) -> Vec<ReplicaView> {
    replica_facts(scheduler, request, kind)
        .into_iter()
        .map(|(view, _, _)| view)
        .collect()
}

/// The workers [`replicas`] offers, in the same order and of the same length:
/// the mechanism's half of a placement decision.
///
/// A policy's `place` answers an index into [`replicas`]; the mechanism turns
/// that index into the worker it binds. Both come from [`replica_facts`], so the
/// filter that decides which replicas are candidates cannot differ between the
/// view a policy saw and the worker the mechanism picks. The second element is
/// the component the worker binds for the call, which a placement records with
/// the worker.
pub(crate) fn replica_targets(
    scheduler: &Scheduler,
    request: RequestKey,
    kind: CallKind,
) -> Vec<(crate::WorkerGroupId, String)> {
    replica_facts(scheduler, request, kind)
        .into_iter()
        .map(|(_, worker, bound)| (worker, bound))
        .collect()
}

/// Every candidate replica of `request`'s call of `kind`, with the worker and
/// component it names. One construction of the filter, which [`replicas`] and
/// [`replica_targets`] are two views of.
fn replica_facts(
    scheduler: &Scheduler,
    request: RequestKey,
    kind: CallKind,
) -> Vec<(ReplicaView, crate::WorkerGroupId, String)> {
    let executor = scheduler.executor.as_ref();
    candidate_facts(
        scheduler,
        request,
        kind,
        scheduler
            .placement
            .worker_candidates(executor, &scheduler.info, kind),
    )
}

/// Every candidate replica of `request`'s call of `kind` that serves the named
/// component: a media call, whose candidates are the workers binding that
/// component rather than the text path's.
pub(crate) fn component_replicas(
    scheduler: &Scheduler,
    request: RequestKey,
    kind: CallKind,
    component: &str,
) -> Vec<ReplicaView> {
    component_facts(scheduler, request, kind, component)
        .into_iter()
        .map(|(view, _, _)| view)
        .collect()
}

/// The workers [`component_replicas`] offers, in the same order and of the same
/// length: the mechanism's half of a named component's placement decision.
pub(crate) fn component_targets(
    scheduler: &Scheduler,
    request: RequestKey,
    kind: CallKind,
    component: &str,
) -> Vec<(crate::WorkerGroupId, String)> {
    component_facts(scheduler, request, kind, component)
        .into_iter()
        .map(|(_, worker, bound)| (worker, bound))
        .collect()
}

/// Every candidate replica of a named component's call, as views.
fn component_facts(
    scheduler: &Scheduler,
    request: RequestKey,
    kind: CallKind,
    component: &str,
) -> Vec<(ReplicaView, crate::WorkerGroupId, String)> {
    let executor = scheduler.executor.as_ref();
    candidate_facts(
        scheduler,
        request,
        kind,
        scheduler
            .placement
            .component_candidates(executor, kind, component),
    )
}

/// One replica view per candidate of `request`'s call, with the worker and
/// component it names.
///
/// The candidates are the caller's list — a call kind's, or a named component's
/// — so a policy's index and the mechanism's worker resolve against the same
/// list, and the decode-stage filter is applied once, here, for every list.
fn candidate_facts<'a>(
    scheduler: &'a Scheduler,
    request: RequestKey,
    kind: CallKind,
    candidates: impl Iterator<Item = (&'a crate::WorkerGroupId, &'a str, &'a WorkerInfo)>,
) -> Vec<(ReplicaView, crate::WorkerGroupId, String)> {
    let executor = scheduler.executor.as_ref();
    let table = world_blocks(executor);
    let decode_stage = matches!(
        kind,
        CallKind::Forward(ForwardMode::Decode | ForwardMode::Verify)
            | CallKind::Transfer(TransferMode::CacheInstall)
    );
    candidates
        .filter(|(id, _, _)| !decode_stage || !scheduler.placement.prefill_groups.contains(*id))
        .enumerate()
        .map(|(index, (worker, bound, worker_info))| {
            let view = replica_view(ReplicaFacts {
                index,
                block: block_id(&table, bound).unwrap_or(ComponentId(0)),
                kinds: worker_phases(worker_info),
                resident_requests: resident_requests(&scheduler.placement, worker),
                // The executor reports whether a worker's queue has room, not
                // how many rows that is; a plan may submit one call's worth.
                capacity: u32::from(executor.has_capacity(worker)),
                holds_request: holds_request(
                    &scheduler.placement,
                    request,
                    bound,
                    decode_stage,
                    worker,
                ),
                prefills_only: scheduler.placement.prefill_groups.contains(worker),
                holds_component: holds_component(&scheduler.placement, request, worker),
            });
            (view, worker.clone(), bound.to_owned())
        })
        .collect()
}

/// A request's first call on a block and the replicas that may run it, for
/// `place`; `None` when no replica can.
pub(crate) fn call(
    scheduler: &Scheduler,
    request: RequestKey,
    kind: CallKind,
) -> Option<(CallView, Vec<ReplicaView>)> {
    let candidates = replicas(scheduler, request, kind);
    (!candidates.is_empty()).then(|| (call_view(request, kind), candidates))
}

/// Every replica a `place` decision may choose, for `Snapshot::replicas`: one
/// per worker of the world, in the executor's order.
pub(crate) fn replica_catalog(scheduler: &Scheduler) -> Vec<ReplicaView> {
    catalog_facts(scheduler)
        .into_iter()
        .map(|(view, _, _)| view)
        .collect()
}

/// The workers [`replica_catalog`] lists, in the same order and of the same
/// length: the control thread's half of a routing decision.
///
/// A policy's `route` answers an index into [`replica_catalog`] and the control
/// thread turns it into the worker the request is sent to. Both come from
/// [`catalog_facts`], so the list a policy was shown and the worker chosen are
/// one list.
pub(crate) fn replica_catalog_targets(
    scheduler: &Scheduler,
) -> Vec<(crate::WorkerGroupId, String)> {
    catalog_facts(scheduler)
        .into_iter()
        .map(|(_, worker, bound)| (worker, bound))
        .collect()
}

/// The world's workers with the component each binds by default, as views.
///
/// The catalog is every worker the world reports, not the candidates of one
/// call's kind: routing happens before a request has a call, and the policy
/// reads each worker's phases and capacity to choose. One construction, which
/// [`replica_catalog`] and [`replica_catalog_targets`] are two views of.
fn catalog_facts(scheduler: &Scheduler) -> Vec<(ReplicaView, crate::WorkerGroupId, String)> {
    let executor = scheduler.executor.as_ref();
    let table = world_blocks(executor);
    executor
        .info()
        .workers
        .iter()
        .enumerate()
        .map(|(index, (worker, info))| {
            let bound = info
                .components
                .first()
                .map_or(DEFAULT_COMPONENT, |component| component.name.as_str());
            let view = replica_view(ReplicaFacts {
                index,
                block: block_id(&table, bound).unwrap_or(ComponentId(0)),
                kinds: worker_phases(info),
                resident_requests: resident_requests(&scheduler.placement, worker),
                capacity: u32::from(executor.has_capacity(worker)),
                holds_request: false,
                prefills_only: scheduler.placement.prefill_groups.contains(worker),
                holds_component: false,
            });
            (view, worker.clone(), bound.to_owned())
        })
        .collect()
}

/// The load the experts are balanced against.
///
/// No worker reports the tokens it routed to each expert, so an epoch's views
/// carry none and the mechanism makes no expert decision.
pub(crate) fn expert_load_of(_scheduler: &Scheduler) -> Option<ExpertLoad> {
    None
}

// ---------------------------------------------------------------------------
// The snapshot and the trial state
// ---------------------------------------------------------------------------

/// The state a policy is instantiated with (`Policy::init`).
///
/// Built once per epoch: every object that already exists, so a policy put in
/// mid-flight can act as if it had been there from the start. The rank views
/// are the control thread's to add; the routing table is the service's, not a
/// scheduler's.
pub(crate) fn snapshot(scheduler: &Scheduler, step: u64, recorded: &RecordedTable) -> Snapshot {
    Snapshot {
        step,
        requests: requests(scheduler, recorded),
        pages: pages(scheduler),
        encodings: encodings(scheduler),
        replicas: replica_catalog(scheduler),
        phases: phases_in(&scheduler.info, &world_blocks(scheduler.executor.as_ref())),
        ranks: Vec::new(),
    }
}

/// The snapshot, with the requests the mechanism no longer holds but still
/// reports: a policy put in mid-flight sees a finished request whose output
/// events are retained, as the guest's `Snapshot` asks.
///
/// The recorder holds the last view it told the policy of each such request;
/// this appends the ones the snapshot does not already name.
pub(crate) fn with_finished(mut snapshot: Snapshot, finished: &[RequestView]) -> Snapshot {
    for view in finished {
        if !snapshot.requests.iter().any(|held| held.id == view.id) {
            snapshot.requests.push(view.clone());
        }
    }
    snapshot
}

/// The first request the mechanism is still to admit.
fn first_queued(scheduler: &Scheduler) -> Option<RequestId> {
    scheduler
        .waiting
        .front()
        .map(|state| state.req.request_id)
        .or_else(|| {
            scheduler
                .waiting_media
                .front()
                .map(|pending| pending.request.request_id)
        })
}

/// A token request's view, by id, when the scheduler holds it.
fn request_of(scheduler: &Scheduler, id: RequestId, standing: Standing) -> Option<RequestView> {
    if let Some(state) = scheduler
        .waiting
        .iter()
        .find(|state| state.req.request_id == id)
    {
        return Some(request(scheduler, state, standing, &Recorded::default()));
    }
    if let Some(state) = scheduler.running.get(&id) {
        return Some(request(scheduler, state, standing, &Recorded::default()));
    }
    scheduler
        .waiting_media
        .iter()
        .find(|pending| pending.request.request_id == id)
        .map(|pending| {
            diffusion_request(
                &pending.request,
                standing,
                pending.queued_at,
                None,
                &Recorded::default(),
            )
        })
}

/// The requests waiting for admission, in the mechanism's own order.
pub(crate) fn queued(scheduler: &Scheduler) -> Vec<ViewRequestId> {
    scheduler
        .waiting
        .iter()
        .map(|state| ViewRequestId(state.req.request_id.0))
        .chain(
            scheduler
                .waiting_media
                .iter()
                .map(|pending| ViewRequestId(pending.request.request_id.0)),
        )
        .collect()
}

/// The live state a step's validators read: this moment's snapshot, with the
/// admission queue in the mechanism's own order.
///
/// The validators read a `Plan`'s names against it, so it is built at the step
/// the plan belongs to and rebuilt after `reclaim` changes the state.
pub(crate) fn live(scheduler: &Scheduler, step: u64, recorded: &RecordedTable) -> Live {
    Live::of(&snapshot(scheduler, step, recorded), queued(scheduler))
}

/// A queued request and the ranks, for `route`: the first request the
/// mechanism is still to admit, and the ranks it may be routed to.
pub(crate) fn arrival(
    scheduler: &Scheduler,
    ranks: Vec<RankView>,
) -> Option<(RequestView, Vec<RankView>)> {
    let id = first_queued(scheduler)?;
    let view = request_of(scheduler, id, Standing::Queued)?;
    Some((view, ranks))
}

/// A resident request's next call on a block and the replicas that may run it,
/// for `place`: the first resident request that has a next call and a replica.
pub(crate) fn placement(scheduler: &Scheduler) -> Option<(CallView, Vec<ReplicaView>)> {
    for id in &scheduler.running_order {
        let Some(state) = scheduler.running.get(id) else {
            continue;
        };
        let Some(kind) = scheduler.peek_next_call_variant(*id) else {
            continue;
        };
        let key = RequestKey::new(scheduler.engine_id, *id, state.request_epoch);
        if let Some(call) = call(scheduler, key, kind) {
            return Some(call);
        }
    }
    None
}

/// The trial state of this moment: the snapshot, the room, and the decisions
/// the scheduler faces without more context.
///
/// [`TrialState`] is `policy.rs`'s: the switch owns the type it hands to a
/// candidate, and this module builds its contents. The rank views are the
/// routing table's (the control thread's), so `route` is trialled only by
/// [`trial_state_with_ranks`]; a scheduler knows one rank's load, its own.
pub(crate) fn trial_state(
    scheduler: &Scheduler,
    step: u64,
    recorded: &RecordedTable,
) -> TrialState {
    TrialState {
        snapshot: snapshot(scheduler, step, recorded),
        room: room(scheduler, step),
        shortage: shortage_now(scheduler),
        arrival: None,
        call: placement(scheduler),
        expert_load: expert_load_of(scheduler),
    }
}

/// The trial state of this moment, with the ranks the routing table holds: the
/// state routing is decided on, for the trial that asks `route` too (§7.3).
pub(crate) fn trial_state_with_ranks(
    scheduler: &Scheduler,
    step: u64,
    recorded: &RecordedTable,
    ranks: Vec<RankView>,
) -> TrialState {
    let call = placement(scheduler);
    let arrival = arrival(scheduler, ranks.clone());
    let mut snapshot = snapshot(scheduler, step, recorded);
    snapshot.ranks = ranks;
    TrialState {
        snapshot,
        room: room(scheduler, step),
        shortage: shortage_now(scheduler),
        arrival,
        call,
        expert_load: expert_load_of(scheduler),
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// A request reached the main process, before `route` decides its rank.
pub(crate) fn arrived(request: RequestView) -> Event {
    Event::Arrived { request }
}

/// The mechanism admitted a queued request.
pub(crate) const fn admitted(request: ViewRequestId) -> Event {
    Event::Admitted { request }
}

/// A resident request's view changed in a way the policy must see.
pub(crate) fn updated(request: RequestView) -> Event {
    Event::Updated { request }
}

/// A call of the request advanced by `tokens` tokens of `phase`.
pub(crate) const fn advanced(request: ViewRequestId, phase: PhaseId, tokens: u32) -> Event {
    Event::Advanced {
        request,
        phase,
        tokens,
    }
}

/// The request emitted `tokens` to its client at `at`.
pub(crate) const fn emitted(request: ViewRequestId, tokens: u32, at: Nanos) -> Event {
    Event::Emitted {
        request,
        tokens,
        at,
    }
}

/// The request ended.
pub(crate) const fn finished(request: ViewRequestId, end: End) -> Event {
    Event::Finished { request, end }
}

/// A scheduling step ended.
pub(crate) fn stepped(step: u64, duration: std::time::Duration) -> Event {
    Event::Stepped {
        step,
        duration: scheduling::guest::time::Duration(
            u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
        ),
    }
}

/// A page entered the cache.
pub(crate) fn cached(page: PageView) -> Event {
    Event::Cached { page }
}

/// A resident (device) page was used again.
pub(crate) const fn hit(page: PageId) -> Event {
    Event::Hit { page }
}

/// A page was freed.
pub(crate) const fn freed(page: PageId) -> Event {
    Event::Freed { page }
}

/// An encoder product entered the encoder cache.
pub(crate) fn encoded(entry: EncodingView) -> Event {
    Event::Encoded { entry }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scheduling::guest::view::PhaseClass;

    fn facts() -> RequestFacts {
        RequestFacts {
            id: ViewRequestId(7),
            standing: Standing::Resident,
            phase: Some(phase_id(PhaseKind::Decode)),
            priority: 3,
            arrived_at: Nanos(10),
            admitted_at: Some(Nanos(20)),
            prompt_tokens: 128,
            generated_tokens: 5,
            cached_prefix_tokens: 64,
            prefilled_tokens: 96,
            cached_pages: 4,
            last_emitted_at: Some(Nanos(30)),
            class: RequestClass::Token,
            waiting_transfer: false,
            next_tokens: 1,
            reservation_units: 0,
            prefetch_tokens: 32,
            refill_tokens: Some(12),
            remaining_tokens: 100,
            migratable: true,
            cache_rank: Some(1),
        }
    }

    #[test]
    fn a_phase_id_numbers_its_kind() {
        for (index, kind) in PHASES.iter().copied().enumerate() {
            assert_eq!(phase_id(kind), PhaseId(index as u32));
            assert_eq!(phase_kind(phase_id(kind)), Some(kind));
        }
        assert_eq!(phase_id(PhaseKind::Other).0, u32::MAX);
        assert_eq!(phase_kind(PhaseId(PHASES.len() as u32)), None);
    }

    #[test]
    fn a_refill_is_prefill_work_of_a_decoding_request() {
        assert_eq!(
            phase_call(PhaseKind::Refill),
            Some(CallKind::Forward(ForwardMode::Prefill))
        );
        assert!(PhaseKind::Refill.is_prefill());
        assert!(PhaseKind::Refill.writes_prefix_cache());
        assert_eq!(PhaseKind::Refill.class(), PhaseClass::Context);
    }

    #[test]
    fn every_call_kind_is_one_phase() {
        let calls = [
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::Decode),
            CallKind::Forward(ForwardMode::Verify),
            CallKind::Forward(ForwardMode::TokenDenoising),
            CallKind::Media(MediaCall::TextEncoding),
            CallKind::Media(MediaCall::VisionEncoding),
            CallKind::Media(MediaCall::LatentEncoding),
            CallKind::Media(MediaCall::Denoising),
            CallKind::Media(MediaCall::ImageDecoding),
            CallKind::Media(MediaCall::VideoDecoding),
            CallKind::Media(MediaCall::AudioDecoding),
            CallKind::Media(MediaCall::VideoEncoding),
            CallKind::Media(MediaCall::AudioEncoding),
            CallKind::Media(MediaCall::MediaReading),
            CallKind::Media(MediaCall::LatentPreparation),
            CallKind::Media(MediaCall::Muxing),
            CallKind::Transfer(TransferMode::Tensor),
            CallKind::Transfer(TransferMode::CacheExport),
            CallKind::Transfer(TransferMode::CacheInstall),
        ];
        for call in calls {
            let phase = call_phase(call);
            assert_ne!(phase, PhaseKind::Other);
            assert_eq!(phase_call(phase), Some(call), "{call:?} -> {phase:?}");
            assert_eq!(call_phase_id(call), phase_id(phase));
        }
    }

    #[test]
    fn an_epoch_offers_the_phases_its_workers_serve() {
        let mut info = WorkerInfo::default();
        info.components = vec![ipc::ComponentInfo {
            name: "model".into(),
            outputs: Vec::new(),
        }];
        info.supported_calls
            .push(CallKind::Media(MediaCall::Denoising));
        info.media_components
            .insert(MediaCall::Denoising, "denoiser".into());

        let phases = phases(&info);
        let kinds: Vec<PhaseKind> = phases.iter().map(|phase| phase.kind).collect();
        // Prefill and decode are served, so a refill chunk of a decoding request
        // is served too; a kind no worker reports is absent.
        assert!(kinds.contains(&PhaseKind::Prefill));
        assert!(kinds.contains(&PhaseKind::Decode));
        assert!(kinds.contains(&PhaseKind::Refill));
        assert!(kinds.contains(&PhaseKind::Denoise));
        assert!(!kinds.contains(&PhaseKind::ImageDecode));

        // A phase's id is its kind's, and its block is the component the worker
        // binds for that call.
        let blocks = blocks(&info);
        for phase in &phases {
            assert_eq!(phase.id, phase_id(phase.kind));
        }
        let denoise = phases
            .iter()
            .find(|phase| phase.kind == PhaseKind::Denoise)
            .expect("the worker serves denoising");
        assert_eq!(Some(denoise.block), block_id(&blocks, "denoiser"));
        let prefill = phases
            .iter()
            .find(|phase| phase.kind == PhaseKind::Prefill)
            .expect("the worker serves prefill");
        assert_eq!(Some(prefill.block), block_id(&blocks, "model"));
    }

    #[test]
    fn a_wall_clock_instant_becomes_a_monotonic_one() {
        let now = nanos_at(crate::clock::now_unix_secs());
        let monotonic = Nanos(crate::clock::now_monotonic_us() * 1_000);
        assert!(
            now.0.abs_diff(monotonic.0) < 1_000_000_000,
            "{now:?} is not within a second of {monotonic:?}"
        );
        // A time before the process started is the clock's zero, not a wrap.
        assert_eq!(nanos_at(0.0), Nanos(0));
    }

    #[test]
    fn a_device_page_id_and_a_host_page_id_do_not_collide() {
        assert_eq!(page_target(device_page(9)), Some(PageTarget::Device(9)));
        assert_eq!(unit_of(device_page(9)), Some(9));
        assert_eq!(page_target(host_page(9)), Some(PageTarget::Host(9)));
        assert_eq!(unit_of(host_page(9)), None);
        assert_eq!(
            page_target(PageId(u64::MAX)),
            Some(PageTarget::Host(u64::MAX & !HOST_PAGE))
        );
        assert_eq!(page_target(PageId(1 << 32)), None);
    }

    #[test]
    fn a_request_view_carries_its_features() {
        let view = request_view(facts());
        assert_eq!(view.id, ViewRequestId(7));
        assert_eq!(view.state, Standing::Resident);
        assert_eq!(view.phase, Some(phase_id(PhaseKind::Decode)));
        assert_eq!(view.cached_prefix_tokens, 64);
        assert_eq!(view.refill_tokens, Some(12));
        assert!(view.migratable);
        assert_eq!(view.cache_rank, Some(1));
    }

    #[test]
    fn a_queued_request_has_no_admission_time() {
        let mut facts = facts();
        facts.standing = Standing::Queued;
        facts.admitted_at = None;
        let view = request_view(facts);
        assert_eq!(view.state, Standing::Queued);
        assert_eq!(view.admitted_at, None);
    }

    #[test]
    fn a_room_carries_the_steps_walls() {
        let room = room_view(RoomFacts {
            step: 4,
            kv_units: 100,
            slots: 3,
            token_budget: 8192,
            call_budget: 128,
            graph_tiers: vec![1, 2, 4],
            page_tokens: 16,
            prefill_tokens: Some(512),
            batches: 2,
            prefill_batches: 1,
            capacity_units: 1024,
            warming: vec![WarmingView {
                request: ViewRequestId(1),
                pages: vec![device_page(2)],
            }],
        });
        assert_eq!(room.step, 4);
        assert_eq!(room.kv_units, 100);
        assert_eq!(room.slots, 3);
        assert_eq!(room.inflight.batches, 2);
        assert_eq!(room.inflight.prefill_batches, 1);
        assert_eq!(room.capacity_units, 1024);
        assert_eq!(room.warming.len(), 1);
        assert_eq!(room.graph_tiers, vec![1, 2, 4]);
    }

    #[test]
    fn replica_indices_number_the_place_list() {
        let replicas: Vec<ReplicaView> = (0..3)
            .map(|index| {
                replica_view(ReplicaFacts {
                    index,
                    block: ComponentId(1),
                    kinds: vec![PhaseKind::Denoise],
                    resident_requests: index as u32,
                    capacity: 1,
                    holds_request: index == 1,
                    prefills_only: false,
                    holds_component: index == 2,
                })
            })
            .collect();
        assert_eq!(
            replicas.iter().map(|view| view.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(replicas[1].holds_request);
        assert!(replicas[2].holds_component);
    }

    #[test]
    fn an_encoding_id_is_the_head_of_its_key() {
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&9u64.to_be_bytes());
        key[8] = 0xff;
        assert_eq!(encoding_id(&key), EncodingId(9));
    }

    #[test]
    fn a_shortage_names_its_storage() {
        let short = shortage(Storage::HostPages, 3);
        assert_eq!(short.kind, Storage::HostPages);
        assert_eq!(short.need, 3);
    }
}
