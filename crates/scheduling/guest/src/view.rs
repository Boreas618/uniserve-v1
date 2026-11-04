//! The snapshot and the views: what a policy reads about the mechanism's state.
//!
//! A view is a set of features of an object, not the object: the mechanism
//! owns the object, and a policy must never ask for anything not here (a
//! decision it cannot make from these features is a mechanism feature to add,
//! which is an epoch layer change). The views and the snapshot are append-only
//! lists; a field added to a view is ignored by a policy that does not know it.

use serde::{Deserialize, Serialize};

use crate::ids::{ComponentId, EncodingId, PageId, PhaseId, RequestId};
use crate::time::Nanos;

/// The whole state a policy is instantiated with (`Policy::init`).
///
/// A policy put in mid-flight sees every object that already exists, so it can
/// act as if it had been there from the start; its own soft state starts
/// empty (design §4.4).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Snapshot {
    /// The step number the snapshot was taken at.
    #[serde(default)]
    pub step: u64,
    /// Every request the main process holds: queued, resident, and those that
    /// finished with output events still retained.
    #[serde(default)]
    pub requests: Vec<RequestView>,
    /// Every page of the device KV cache and of the host KV page pool.
    #[serde(default)]
    pub pages: Vec<PageView>,
    /// Every encoder cache entry.
    #[serde(default)]
    pub encodings: Vec<EncodingView>,
    /// Every replica a `place` decision may choose.
    #[serde(default)]
    pub replicas: Vec<ReplicaView>,
    /// Every phase of this epoch, so a policy can read a request's phase.
    #[serde(default)]
    pub phases: Vec<PhaseView>,
    /// The DP ranks a `route` decision may choose, in scheduler-index order.
    #[serde(default)]
    pub ranks: Vec<RankView>,
}

/// Where a request stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RequestState {
    /// Waiting for admission.
    Queued,
    /// Admitted and the mechanism serves it.
    Resident,
    /// Ended; its output events may still be retained.
    Finished,
    /// A state this policy does not know.
    #[serde(other)]
    Unknown,
}

/// What kind of work a request is, as scheduling distinguishes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "class", rename_all = "snake_case")]
pub enum RequestClass {
    /// Token generation: prefill and decode of a causal language model.
    Token,
    /// Media work: encoding, denoising, decoding, muxing.
    Media,
    /// A readout of a canvas-generating request.
    Readout,
    /// A class this policy does not know.
    #[serde(other)]
    Unknown,
}

/// The request features design §4.4 names, and the ones a `Plan` needs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestView {
    pub id: RequestId,
    pub state: RequestState,
    /// The phase of the request's next call, `None` when it has none ready.
    #[serde(default)]
    pub phase: Option<PhaseId>,
    /// The priority the request itself carries (a request parameter, not the
    /// policy's). Lower values are more urgent in this deployment's
    /// convention; the policy treats it as opaque and only orders by it.
    #[serde(default)]
    pub priority: i32,
    /// When the request reached the main process.
    #[serde(default)]
    pub arrived_at: Nanos,
    /// When the mechanism admitted it; `None` while queued.
    #[serde(default)]
    pub admitted_at: Option<Nanos>,
    /// Tokens of its prompt.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Tokens it has generated so far.
    #[serde(default)]
    pub generated_tokens: u32,
    /// Tokens of its prompt the prefix cache already holds.
    #[serde(default)]
    pub cached_prefix_tokens: u32,
    /// Prompt tokens the runtime has computed so far (the next prefill chunk
    /// starts here). Together with `cached_prefix_tokens` it gives the chunk
    /// start and so the distance to the next KV page boundary.
    #[serde(default)]
    pub prefilled_tokens: u32,
    /// KV pages it currently holds.
    #[serde(default)]
    pub cached_pages: u32,
    /// When it last emitted a token.
    #[serde(default)]
    pub last_emitted_at: Option<Nanos>,
    /// Its class.
    pub class: RequestClass,
    /// Whether a resident request waits for a transfer (KV storage or a host
    /// page read) right now. A request that waits for a transfer is not
    /// starved (design §4.4: "驻留且不在等传输的请求超过 ... timeout").
    #[serde(default)]
    pub waiting_transfer: bool,
    /// The token charge of the request's next call when that call is not a
    /// prefill or refill chunk (a denoising step is charged its latent units
    /// times its guidance branches, a verification one token plus its drafts,
    /// any other call its bound). A prefill or refill chunk is charged the
    /// chunk the policy sizes instead.
    #[serde(default)]
    pub next_tokens: u32,
    /// The worst-case device KV units the request reserves when it is
    /// admitted. A request whose reservation exceeds the cache's total
    /// capacity can never run and is rejected.
    #[serde(default)]
    pub reservation_units: u64,
    /// Tokens the host KV cache could add to the request's prefix hit beyond
    /// the device hit (the deepening a prefetch would gain); 0 when none.
    #[serde(default)]
    pub prefetch_tokens: u32,
    /// Tokens a refill of the request's draft cache would cost; `None` when
    /// its draft cache has no gap.
    #[serde(default)]
    pub refill_tokens: Option<u32>,
    /// Whether the request resumed from generated tokens (a migrated or
    /// preempted request), so ending it as migratable and resubmitting it
    /// loses nothing. Only such a request may be a `Victim::Request`.
    #[serde(default)]
    pub migratable: bool,
    /// The rank recorded for the request's cache resume, when it resumes a
    /// stored cache. `route` returns this rank when it is set; the mechanism
    /// skips the routing decision for it (today `Routing::pick` is skipped the
    /// same way).
    #[serde(default)]
    pub cache_rank: Option<usize>,
}

/// One phase of the epoch: a block's computation for one kind of call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhaseView {
    pub id: PhaseId,
    pub kind: PhaseKind,
    /// The block that runs it.
    pub block: ComponentId,
}

/// The kind of a phase, at the granularity scheduling distinguishes.
///
/// This is the vocabulary today's `CallKind` and generation phases have, named
/// once so a policy orders and charges work by it. Appended, never renumbered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhaseKind {
    // Context ingestion: reads the request's inputs and writes the prefix cache.
    TextEncoding,
    VisionEncoding,
    LatentEncoding,
    Prefill,
    // Token progress.
    Decode,
    Verify,
    TokenDenoise,
    Readout,
    ImageDecode,
    CacheInstall,
    // Media work.
    MediaRead,
    Denoise,
    VideoDecode,
    AudioDecode,
    VideoEncode,
    AudioEncode,
    Mux,
    // Background work.
    LatentPrepare,
    TensorTransfer,
    CacheExport,
    /// A draft-cache refill: a prefill-class chunk of a decoding request
    /// (design §4.4, `Plan::refill`).
    Refill,
    /// A phase kind this policy does not know.
    #[serde(other)]
    Other,
}

/// The four classes work is assembled in (design §4.4, `fcfs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PhaseClass {
    /// Context ingestion.
    Context,
    /// Token progress.
    Token,
    /// Media work.
    Media,
    /// Latent preparation, transfers, cache export, and work with no phase.
    Background,
}

impl PhaseKind {
    /// The assembly class this phase kind belongs to: the three levels today's
    /// `assembly_priority` has, plus background.
    pub fn class(self) -> PhaseClass {
        match self {
            PhaseKind::TextEncoding
            | PhaseKind::VisionEncoding
            | PhaseKind::LatentEncoding
            | PhaseKind::Prefill => PhaseClass::Context,
            PhaseKind::Decode
            | PhaseKind::Verify
            | PhaseKind::TokenDenoise
            | PhaseKind::Readout
            | PhaseKind::ImageDecode
            | PhaseKind::CacheInstall => PhaseClass::Token,
            PhaseKind::MediaRead
            | PhaseKind::Denoise
            | PhaseKind::VideoDecode
            | PhaseKind::AudioDecode
            | PhaseKind::VideoEncode
            | PhaseKind::AudioEncode
            | PhaseKind::Mux => PhaseClass::Media,
            PhaseKind::LatentPrepare
            | PhaseKind::TensorTransfer
            | PhaseKind::CacheExport
            | PhaseKind::Refill
            | PhaseKind::Other => PhaseClass::Background,
        }
    }

    /// Whether a chunk of this phase writes the prefix cache. Only such a
    /// chunk draws on the prefill budget a warming epoch bounds (design §4.4).
    pub fn writes_prefix_cache(self) -> bool {
        matches!(self, PhaseKind::Prefill | PhaseKind::Refill)
    }

    /// Whether work of this kind is prefill work for batching: a prefill or a
    /// refill chunk, the work a decode step must not starve.
    pub fn is_prefill(self) -> bool {
        self.writes_prefix_cache()
    }
}

/// One page of KV cache: a device page or a host page.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PageView {
    pub id: PageId,
    /// Tokens the page holds.
    pub tokens: u32,
    /// When it was last used; larger is more recent. Free pages use the same
    /// clock, so a policy's LRU list orders them.
    #[serde(default)]
    pub last_used: Nanos,
    /// Whether the page currently sits in host memory (an offloaded page).
    #[serde(default)]
    pub on_host: bool,
    /// Whether no resident request continues from the page, so it may be
    /// offloaded to the host KV cache or freed.
    #[serde(default)]
    pub evictable: bool,
}

/// One encoder cache entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodingView {
    pub id: EncodingId,
    /// When it was last used; larger is more recent.
    #[serde(default)]
    pub last_used: Nanos,
}

/// One DP rank, as a `route` decision sees it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankView {
    /// The scheduler index of the rank; `route` returns this index.
    pub index: usize,
    /// Requests the rank holds in flight or resident.
    #[serde(default)]
    pub in_flight: u32,
    /// Whether the rank's scheduler is alive.
    #[serde(default = "default_true")]
    pub alive: bool,
}

fn default_true() -> bool {
    true
}

/// One replica of a block, as a `place` decision sees it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicaView {
    /// The index in this `place` call's replica list; `place` returns it.
    pub index: usize,
    /// The block this replica runs.
    pub block: ComponentId,
    /// Phase kinds this replica can execute.
    #[serde(default)]
    pub kinds: Vec<PhaseKind>,
    /// Distinct requests resident on it.
    #[serde(default)]
    pub resident_requests: u32,
    /// Requests it can still take.
    #[serde(default)]
    pub capacity: u32,
    /// Whether this request's state already lives on it; a request is never
    /// migrated off a replica that holds it (design §4.4, today's
    /// `component_target`).
    #[serde(default)]
    pub holds_request: bool,
    /// Whether it runs prefill only, so a decode-stage call never lands on it
    /// (today `placement.rs`'s prefill-only groups).
    #[serde(default)]
    pub prefills_only: bool,
    /// Whether it already holds another component of this request, which a
    /// placement prefers to a replica that holds none of it.
    #[serde(default)]
    pub holds_component: bool,
}

/// The call a `place` decision is about: a request's first call on a block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CallView {
    pub request: RequestId,
    pub phase: PhaseId,
    pub kind: PhaseKind,
}

/// The expert load a periodic `experts` decision balances.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExpertLoad {
    /// Ranks experts may be placed on.
    #[serde(default)]
    pub ranks: usize,
    /// Experts, in index order.
    #[serde(default)]
    pub experts: usize,
    /// Tokens routed to each expert since the last decision.
    #[serde(default)]
    pub tokens: Vec<u64>,
}

/// An expert-to-rank placement: `placement[expert]` is the rank of that
/// expert, and it must be a permutation of the ranks in the sense the existing
/// expert rearrangement requires (design §7.6).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExpertOrder {
    #[serde(default)]
    pub placement: Vec<usize>,
}
