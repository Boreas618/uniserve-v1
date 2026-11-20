//! The snapshot and the views: what a policy reads about the mechanism's state.
//!
//! A view is a set of features of an object, not the object: the mechanism
//! owns the object, and a policy must never ask for anything not here (a
//! decision it cannot make from these features is a mechanism feature to add,
//! which raises the interface's version).

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
    pub step: u64,
    /// Every request the main process holds: queued, running, and those that
    /// finished with output events still retained.
    pub requests: Vec<RequestView>,
    /// Every page of the device KV cache and of the host KV page pool.
    pub pages: Vec<PageView>,
    /// Every encoder cache entry.
    pub encodings: Vec<EncodingView>,
    /// Every replica a `place` decision may choose.
    pub replicas: Vec<ReplicaView>,
    /// Every phase of this epoch, so a policy can read a request's phase.
    pub phases: Vec<PhaseView>,
}

/// Where a request stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    /// Waiting for admission.
    Waiting,
    /// Admitted and the mechanism serves it.
    Running,
    /// Ended; its output events may still be retained.
    Finished,
}

/// What kind of work a request is, as scheduling distinguishes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestClass {
    /// Token generation: prefill and decode of a causal language model.
    Token,
    /// Media work: encoding, denoising, decoding, muxing.
    Media,
}

/// The request features design §4.4 names, and the ones a `DispatchOutput` needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestView {
    pub id: RequestId,
    pub state: RequestStatus,
    /// The phase of the request's next call, `None` when it has none ready.
    pub phase: Option<PhaseId>,
    /// The priority the request itself carries (a request parameter, not the
    /// policy's). Lower values are more urgent in this deployment's
    /// convention; the policy treats it as opaque and only orders by it.
    pub priority: i32,
    /// When the request reached the main process.
    pub arrived_at: Nanos,
    /// When the mechanism admitted it; `None` while waiting.
    pub admitted_at: Option<Nanos>,
    /// Tokens of its prompt.
    pub prompt_tokens: u32,
    /// The context tokens the request's next prefill must write up to: its
    /// prompt, extended by what a block commit adds to its context. A request
    /// with nothing to commit states its prompt length.
    pub prefill_tokens: u32,
    /// Tokens it has generated so far.
    pub generated_tokens: u32,
    /// Tokens of its prompt the prefix cache already holds.
    pub cached_prefix_tokens: u32,
    /// Where the next prefill chunk starts: for a running request the prompt
    /// tokens the runtime has computed, which lie before the end of its cached
    /// prefix while a bounded replay recomputes the sliding windows before it;
    /// for a waiting one the further of those and the prefix the cache holds
    /// for it. `cached_prefix_tokens` is never a value to add to it.
    pub prefilled_tokens: u32,
    /// KV pages it currently holds.
    pub cached_pages: u32,
    /// When it last emitted a token.
    pub last_emitted_at: Option<Nanos>,
    /// Its class.
    pub class: RequestClass,
    /// Whether a running request waits for a transfer (KV storage or a host
    /// page read) right now. A request that waits for a transfer is not
    /// starved (design §4.4: "驻留且不在等传输的请求超过 ... timeout").
    pub waiting_transfer: bool,
    /// The token charge of the request's next call when that call is not a
    /// prefill or refill chunk (a denoising step is charged its latent units
    /// times its guidance branches, a verification one token plus its drafts,
    /// any other call its bound). A prefill or refill chunk is charged the
    /// chunk the policy sizes instead, and a prefill with nothing left to
    /// ingest — a close cache, a block's commit — is charged
    /// the one position it writes, since there is no chunk to size.
    pub next_tokens: u32,
    /// The worst-case device KV units the request reserves when it is
    /// admitted. A request whose reservation exceeds the cache's total
    /// capacity can never run and is rejected.
    pub reservation_units: u64,
    /// Tokens the host KV cache could add to the request's prefix hit beyond
    /// the device hit (the deepening a prefetch would gain); 0 when none.
    pub prefetch_tokens: u32,
    /// Tokens a refill of the request's draft cache would cost; `None` when
    /// its draft cache has no gap.
    pub refill_tokens: Option<u32>,
    /// Tokens the request may still generate (`max_und_tokens` less what it
    /// generated), which a refill's worth is judged against.
    pub remaining_tokens: u32,
    /// Whether the request resumed from generated tokens (a migrated or
    /// preempted request), so ending it as migratable and resubmitting it
    /// loses nothing. Only such a request may be a `Victim::Request`.
    pub migratable: bool,
}

/// One phase of the epoch: a block's computation for one kind of call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhaseView {
    pub id: PhaseId,
    pub kind: Phase,
    /// The block that runs it.
    pub block: StageId,
}

/// The four classes work is assembled in (design §4.4): a step assembles context ingestion,
/// token progress, media work and background work in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PhaseClass {
    /// Context ingestion: reads the request's inputs and writes the prefix cache.
    Context,
    /// Token progress.
    Token,
    /// Media work.
    Media,
    /// Latent preparation.
    Background,
}

/// Declares [`Phase`] and every table over it from one list: each phase's variant, its name (its
/// tag on the policy's wire, which a package's selection key and the `phase` dimension of an
/// operator's descriptor key name it by too) and its assembly class. A phase's place in the list
/// is its index into a table of [`Phase::COUNT`] entries and its id ([`Phase::id`]).
macro_rules! phases {
    ($($(#[$doc:meta])* $variant:ident = $name:literal, $class:ident;)*) => {
        /// What one call computes, at the granularity scheduling distinguishes: the one closed
        /// set the request, the batch, the worker group and the policy name a call's work by.
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        pub enum Phase {
            $($(#[$doc])* #[serde(rename = $name)] $variant,)*
        }

        impl Phase {
            /// Every phase, each at its [`Phase::index`].
            pub const ALL: [Phase; Phase::COUNT] = [$(Phase::$variant,)*];

            /// The number of phases: the length of a table with one entry per phase, such as
            /// `[u32; Phase::COUNT]`, which [`Phase::index`] indexes.
            pub const COUNT: usize = [$(Phase::$variant,)*].len();

            /// The phase's name ([`Phase::named`] reads it back).
            pub const fn name(self) -> &'static str {
                match self {
                    $(Phase::$variant => $name,)*
                }
            }

            /// The assembly class this phase belongs to.
            pub const fn class(self) -> PhaseClass {
                match self {
                    $(Phase::$variant => PhaseClass::$class,)*
                }
            }
        }
    };
}

phases! {
    TextEncode = "text_encoding", Context;
    VisionEncode = "vision_encoding", Context;
    LatentEncode = "latent_encoding", Context;
    Prefill = "prefill", Context;
    Decode = "decode", Token;
    Verify = "verify", Token;
    MediaRead = "media_read", Media;
    Denoise = "denoise", Media;
    VideoDecode = "video_decode", Media;
    AudioDecode = "audio_decode", Media;
    VideoEncode = "video_encode", Media;
    AudioEncode = "audio_encode", Media;
    Mux = "mux", Media;
    LatentPrepare = "latent_prepare", Background;
    /// A draft-cache refill: a prefill-class chunk of a decoding request
    /// (design §4.4, `DispatchOutput::refill`), assembled with the context work it is.
    Refill = "refill", Context;
}

impl core::fmt::Display for Phase {
    /// The phase's name ([`Phase::name`]).
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Phase {
    /// The phase's place in a table of [`Phase::COUNT`] entries: dense from zero, in the order
    /// of the declaration.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The id a rank's output, a policy's `DispatchOutput`, a batch and a module slot number this
    /// phase by on a wire: its index. Both ends of every wire are built from one release of this
    /// crate, so the numbering is the declaration's.
    pub const fn id(self) -> PhaseId {
        PhaseId(self as u32)
    }

    /// The phase numbered `id` ([`Phase::id`]); `None` for an id no phase carries.
    pub fn of_id(id: PhaseId) -> Option<Self> {
        Self::ALL.get(id.0 as usize).copied()
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PageView {
    pub id: PageId,
    /// Tokens the page holds.
    pub tokens: u32,
    /// When it was last used; larger is more recent. Free pages use the same
    /// clock, so a policy's LRU list orders them.
    pub last_used: Nanos,
    /// Whether the page currently sits in host memory (an offloaded page).
    pub on_host: bool,
    /// Whether no running request continues from the page, so it may be
    /// offloaded to the host KV cache or freed.
    pub evictable: bool,
}

/// One encoder cache entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EncodingView {
    pub id: EncodingId,
    /// When it was last used; larger is more recent.
    pub last_used: Nanos,
}

/// One replica of a block, as a `place` decision sees it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicaView {
    /// The index in this `place` call's replica list; `place` returns it.
    pub index: usize,
    /// The block this replica runs.
    pub block: StageId,
    /// Phases this replica can execute.
    pub kinds: Vec<Phase>,
    /// Distinct requests running on it.
    pub running_requests: u32,
    /// Requests it can still take.
    pub capacity: u32,
    /// Whether this request's state already lives on it; a request is never
    /// migrated off a replica that holds it (design §4.4).
    pub holds_request: bool,
    /// Whether it already holds another stage of this request, which a
    /// dispatch prefers to a replica that holds none of it.
    pub holds_stage: bool,
}

/// The call a `place` decision is about: a request's first call on a block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CallView {
    pub request: RequestId,
    pub phase: PhaseId,
    pub kind: Phase,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A phase's id and name read back as the phase, and its name is its tag on the wire.
    #[test]
    fn a_phase_reads_back_from_its_id_and_its_name() {
        for phase in Phase::ALL {
            assert_eq!(Phase::of_id(phase.id()), Some(phase));
            assert_eq!(Phase::named(phase.name()), Some(phase));
            assert_eq!(phase.to_string(), phase.name());

            let encoded = serde_json::to_string(&phase).unwrap();
            assert_eq!(encoded, format!(r#""{}""#, phase.name()));
            assert_eq!(serde_json::from_str::<Phase>(&encoded).unwrap(), phase);
        }
        assert_eq!(Phase::of_id(PhaseId(Phase::COUNT as u32)), None);
        assert_eq!(Phase::named("other"), None);
    }

    /// The index numbers every phase densely from zero, so a table of `Phase::COUNT` entries
    /// holds one per phase, and `Phase::ALL` lists them at their index.
    #[test]
    fn the_index_is_dense() {
        for (index, phase) in Phase::ALL.into_iter().enumerate() {
            assert_eq!(phase.index(), index, "{phase:?}");
        }
    }

    /// A tag no phase carries is refused, not read as some phase.
    #[test]
    fn a_tag_no_phase_carries_is_refused() {
        assert!(serde_json::from_str::<Phase>(r#""a_later_phase""#).is_err());
    }
}
