//! The snapshot and the views: what a policy reads about the mechanism's state.
//!
//! A view is a set of features of an object, not the object: the mechanism
//! owns the object, and a policy must never ask for anything not here (a
//! decision it cannot make from these features is a mechanism feature to add,
//! which is an epoch layer change). The views and the snapshot are append-only
//! lists; a field added to a view is ignored by a policy that does not know it.

use serde::{Deserialize, Serialize};

use crate::ids::{EncodingId, PageId, PhaseId, RequestId, StageId};
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
    /// Every request the main process holds: queued, running, and those that
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
pub enum RequestStatus {
    /// Waiting for admission.
    Waiting,
    /// Admitted and the mechanism serves it.
    Running,
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
    /// A class this policy does not know.
    #[serde(other)]
    Unknown,
}

/// The request features design §4.4 names, and the ones a `DispatchOutput` needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestView {
    pub id: RequestId,
    pub state: RequestStatus,
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
    /// When the mechanism admitted it; `None` while waiting.
    #[serde(default)]
    pub admitted_at: Option<Nanos>,
    /// Tokens of its prompt.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// The context tokens the request's next prefill must write up to: its
    /// prompt, extended by what a block commit adds to its context. A request
    /// with nothing to commit states its prompt length.
    #[serde(default)]
    pub prefill_tokens: u32,
    /// Tokens it has generated so far.
    #[serde(default)]
    pub generated_tokens: u32,
    /// Tokens of its prompt the prefix cache already holds.
    #[serde(default)]
    pub cached_prefix_tokens: u32,
    /// Where the next prefill chunk starts: the prompt tokens the runtime has
    /// computed, or the prefix the cache already held for it at admission,
    /// whichever is further on. `cached_prefix_tokens` says how many of those
    /// the cache holds; it is part of this count, not a value to add to it.
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
    /// Whether a running request waits for a transfer (KV storage or a host
    /// page read) right now. A request that waits for a transfer is not
    /// starved (design §4.4: "驻留且不在等传输的请求超过 ... timeout").
    #[serde(default)]
    pub waiting_transfer: bool,
    /// The token charge of the request's next call when that call is not a
    /// prefill or refill chunk (a denoising step is charged its latent units
    /// times its guidance branches, a verification one token plus its drafts,
    /// any other call its bound). A prefill or refill chunk is charged the
    /// chunk the policy sizes instead, and a prefill with nothing left to
    /// ingest — a close cache, a block's commit — is charged
    /// the one position it writes, since there is no chunk to size.
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
    /// Tokens the request may still generate (`max_und_tokens` less what it
    /// generated), which a refill's worth is judged against.
    #[serde(default)]
    pub remaining_tokens: u32,
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
    pub kind: Phase,
    /// The block that runs it.
    pub block: StageId,
}

/// What one call computes, at the granularity scheduling distinguishes: the one
/// closed set the request, the batch, the worker group and the policy name a
/// call's work by.
///
/// A phase crosses the policy's wire by its tag (`{"kind": "prefill"}`), which
/// never changes; the three encodings' tags are `text_encoding`,
/// `vision_encoding` and `latent_encoding`. Appended, never renumbered
/// ([`Phase::id`]).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Phase {
    // Context ingestion: reads the request's inputs and writes the prefix cache.
    #[serde(rename = "text_encoding")]
    TextEncode,
    #[serde(rename = "vision_encoding")]
    VisionEncode,
    #[serde(rename = "latent_encoding")]
    LatentEncode,
    Prefill,
    // Token progress.
    Decode,
    Verify,
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
    /// A draft-cache refill: a prefill-class chunk of a decoding request
    /// (design §4.4, `DispatchOutput::refill`).
    Refill,
    /// A phase the reader of a message does not know: a tag a later revision
    /// appends, or one no phase carries any more. A policy reads it as
    /// background work; the mechanism never names a request's or a batch's
    /// phase with it.
    #[default]
    #[serde(other)]
    Other,
}

impl core::fmt::Display for Phase {
    /// The phase's name ([`Phase::name`]).
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.name())
    }
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
    /// Latent preparation and work with no phase.
    Background,
}

impl Phase {
    /// The number of phases: the length of a table with one entry per phase,
    /// such as `[u32; Phase::COUNT]`, which [`Phase::index`] indexes.
    pub const COUNT: usize = 15;

    /// Every phase, each at its [`Phase::index`]: the order of [`Phase::id`].
    pub const ALL: [Phase; Phase::COUNT] = [
        Self::TextEncode,
        Self::VisionEncode,
        Self::LatentEncode,
        Self::Prefill,
        Self::Decode,
        Self::Verify,
        Self::MediaRead,
        Self::Denoise,
        Self::VideoDecode,
        Self::AudioDecode,
        Self::VideoEncode,
        Self::AudioEncode,
        Self::Mux,
        Self::LatentPrepare,
        Self::Refill,
    ];

    /// The assembly class this phase belongs to: the three levels today's
    /// `assembly_priority` has, plus background.
    pub fn class(self) -> PhaseClass {
        match self {
            Phase::TextEncode
            | Phase::VisionEncode
            | Phase::LatentEncode
            | Phase::Prefill
            | Phase::Refill => PhaseClass::Context,
            Phase::Decode | Phase::Verify => PhaseClass::Token,
            Phase::MediaRead
            | Phase::Denoise
            | Phase::VideoDecode
            | Phase::AudioDecode
            | Phase::VideoEncode
            | Phase::AudioEncode
            | Phase::Mux => PhaseClass::Media,
            Phase::LatentPrepare | Phase::Other => PhaseClass::Background,
        }
    }

    /// The phase's place in a table of [`Phase::COUNT`] entries: dense from
    /// zero, in the order of [`Phase::id`], so it changes only when a phase is
    /// appended (at the end) or removed. Unlike the id it is not a wire value:
    /// tables indexed by it live in one process.
    ///
    /// [`Phase::Other`] names no phase and has no place: its index is `COUNT`,
    /// past the end of every such table, so indexing one with it panics.
    pub const fn index(self) -> usize {
        match self {
            Self::TextEncode => 0,
            Self::VisionEncode => 1,
            Self::LatentEncode => 2,
            Self::Prefill => 3,
            Self::Decode => 4,
            Self::Verify => 5,
            Self::MediaRead => 6,
            Self::Denoise => 7,
            Self::VideoDecode => 8,
            Self::AudioDecode => 9,
            Self::VideoEncode => 10,
            Self::AudioEncode => 11,
            Self::Mux => 12,
            Self::LatentPrepare => 13,
            Self::Refill => 14,
            Self::Other => Self::COUNT,
        }
    }

    /// The id this phase is numbered with, which "appended, never renumbered" protects: a new
    /// phase takes the next unused id, and the id of a removed phase is never given to another. A
    /// rank's output, a policy's `DispatchOutput` and a batch's phase are all numbered here, so
    /// this is the one table they can agree on.
    ///
    /// Ids 6, 7, 8, 9, 18 and 19 are reserved: they numbered phases that no longer exist (19 was
    /// the cache export, superseded by the scheduler's device-direct transfer).
    ///
    /// [`Phase::Other`] is a policy-side placeholder for a phase a policy does not know: it has
    /// no phase of its own, so its id is `u32::MAX`.
    pub const fn id(self) -> PhaseId {
        PhaseId(match self {
            Self::TextEncode => 0,
            Self::VisionEncode => 1,
            Self::LatentEncode => 2,
            Self::Prefill => 3,
            Self::Decode => 4,
            Self::Verify => 5,
            Self::MediaRead => 10,
            Self::Denoise => 11,
            Self::VideoDecode => 12,
            Self::AudioDecode => 13,
            Self::VideoEncode => 14,
            Self::AudioEncode => 15,
            Self::Mux => 16,
            Self::LatentPrepare => 17,
            Self::Refill => 20,
            Self::Other => u32::MAX,
        })
    }

    /// The phase numbered `id` ([`Phase::id`]); `None` for an id no phase carries.
    pub fn of_id(id: PhaseId) -> Option<Self> {
        Self::ALL.into_iter().find(|phase| phase.id() == id)
    }

    /// The phase's name: its tag on the policy's wire, which a package's selection key and the
    /// `phase` dimension of an operator's descriptor key name it by too. [`Phase::Other`], no
    /// phase, is `other`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::TextEncode => "text_encoding",
            Self::VisionEncode => "vision_encoding",
            Self::LatentEncode => "latent_encoding",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Verify => "verify",
            Self::MediaRead => "media_read",
            Self::Denoise => "denoise",
            Self::VideoDecode => "video_decode",
            Self::AudioDecode => "audio_decode",
            Self::VideoEncode => "video_encode",
            Self::AudioEncode => "audio_encode",
            Self::Mux => "mux",
            Self::LatentPrepare => "latent_prepare",
            Self::Refill => "refill",
            Self::Other => "other",
        }
    }

    /// The phase named `name` ([`Phase::name`]); `None` for a name no phase has.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|phase| phase.name() == name)
    }

    /// Whether a chunk of this phase writes the prefix cache. Only such a
    /// chunk draws on the prefill budget a warming epoch bounds (design §4.4).
    pub fn writes_prefix_cache(self) -> bool {
        matches!(self, Phase::Prefill | Phase::Refill)
    }

    /// Whether work of this phase is prefill work for batching: a prefill or a
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
    /// Whether no running request continues from the page, so it may be
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
    /// Requests the rank holds in flight or running.
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
    pub block: StageId,
    /// Phases this replica can execute.
    #[serde(default)]
    pub kinds: Vec<Phase>,
    /// Distinct requests running on it.
    #[serde(default)]
    pub running_requests: u32,
    /// Requests it can still take.
    #[serde(default)]
    pub capacity: u32,
    /// Whether this request's state already lives on it; a request is never
    /// migrated off a replica that holds it (design §4.4, today's
    /// `stage_target`).
    #[serde(default)]
    pub holds_request: bool,
    /// Whether it runs prefill only, so a decode-stage call never lands on it
    /// (today `dispatch.rs`'s prefill-only groups).
    #[serde(default)]
    pub prefills_only: bool,
    /// Whether it already holds another stage of this request, which a
    /// dispatch prefers to a replica that holds none of it.
    #[serde(default)]
    pub holds_stage: bool,
}

/// The call a `place` decision is about: a request's first call on a block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CallView {
    pub request: RequestId,
    #[serde(default)]
    pub phase: PhaseId,
    #[serde(default)]
    pub kind: Phase,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The phases with the ids and wire tags the policy interface numbers and names them by.
    /// Both are the interface's: a policy built against an earlier revision reads the same.
    const WIRE: [(Phase, u32, &str); Phase::COUNT] = [
        (Phase::TextEncode, 0, "text_encoding"),
        (Phase::VisionEncode, 1, "vision_encoding"),
        (Phase::LatentEncode, 2, "latent_encoding"),
        (Phase::Prefill, 3, "prefill"),
        (Phase::Decode, 4, "decode"),
        (Phase::Verify, 5, "verify"),
        (Phase::MediaRead, 10, "media_read"),
        (Phase::Denoise, 11, "denoise"),
        (Phase::VideoDecode, 12, "video_decode"),
        (Phase::AudioDecode, 13, "audio_decode"),
        (Phase::VideoEncode, 14, "video_encode"),
        (Phase::AudioEncode, 15, "audio_encode"),
        (Phase::Mux, 16, "mux"),
        (Phase::LatentPrepare, 17, "latent_prepare"),
        (Phase::Refill, 20, "refill"),
    ];

    #[test]
    fn a_phase_keeps_its_id_and_its_wire_tag() {
        for (phase, id, tag) in WIRE {
            assert_eq!(phase.id(), PhaseId(id), "{phase:?}");
            assert_eq!(Phase::of_id(PhaseId(id)), Some(phase));
            assert_eq!(phase.name(), tag);
            assert_eq!(Phase::named(tag), Some(phase));

            let encoded = serde_json::to_string(&phase).unwrap();
            assert_eq!(encoded, format!(r#"{{"kind":"{tag}"}}"#), "{phase:?}");
            assert_eq!(serde_json::from_str::<Phase>(&encoded).unwrap(), phase);
        }
    }

    /// A tag no phase carries, whether a later revision's or a removed phase's, is read as
    /// [`Phase::Other`] rather than refused, and `Other` names no phase of its own.
    #[test]
    fn a_tag_no_phase_carries_reads_as_other() {
        for tag in ["token_denoise", "a_later_phase"] {
            let decoded: Phase = serde_json::from_str(&format!(r#"{{"kind":"{tag}"}}"#)).unwrap();
            assert_eq!(decoded, Phase::Other, "{tag}");
        }
        assert_eq!(Phase::Other.id(), PhaseId(u32::MAX));
        assert_eq!(Phase::Other.index(), Phase::COUNT);
    }

    /// The index numbers every phase densely from zero, in the order of its id, so a table of
    /// `Phase::COUNT` entries holds one per phase; `Phase::ALL` lists them at their index.
    #[test]
    fn the_index_is_dense_and_follows_the_id() {
        assert_eq!(Phase::ALL.len(), Phase::COUNT);
        for (index, (phase, _, _)) in WIRE.into_iter().enumerate() {
            assert_eq!(phase.index(), index, "{phase:?}");
            assert_eq!(Phase::ALL[index], phase);
        }
        assert!(
            Phase::ALL
                .windows(2)
                .all(|pair| pair[0].id().0 < pair[1].id().0)
        );
    }
}
