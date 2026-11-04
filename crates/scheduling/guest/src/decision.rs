//! What a policy decides: the plan of one step, and the victims of a shortage.

use serde::{Deserialize, Serialize};

use crate::ids::{EncodingId, PageId, PhaseId, RequestId};

/// One step's decision (design §4.4). The mechanism executes the entries in
/// order.
///
/// Every field has a default: a policy module older than the mechanism does not
/// write a field the mechanism added, and the mechanism then executes the
/// empty entry.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Plan {
    /// Queued requests to admit, in order. The mechanism admits each; the
    /// first one it cannot place and the ones after it stay queued.
    #[serde(default)]
    pub admit: Vec<RequestId>,
    /// Queued requests to reject as over capacity.
    #[serde(default)]
    pub reject: Vec<RequestId>,
    /// This step's work, in the order it runs: which request, which phase, how
    /// many tokens. A prefill phase's token count is its chunk size, so a
    /// policy that sizes chunks by decode latency or batches by SLO does it in
    /// this one decision.
    #[serde(default)]
    pub work: Vec<Work>,
    /// Device pages to write to the host KV cache, in order.
    #[serde(default)]
    pub offload: Vec<PageId>,
    /// Requests whose prefix hit to deepen through KV storage, in order. The
    /// mechanism deepens only by at least [`crate::Limits::prefetch_min_tokens`].
    #[serde(default)]
    pub prefetch: Vec<RequestId>,
    /// Requests whose draft cache to refill, in order.
    #[serde(default)]
    pub refill: Vec<RequestId>,
}

/// One piece of this step's work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub request: RequestId,
    pub phase: PhaseId,
    /// Tokens the phase is to run: a prefill chunk's size, a decode step's
    /// token count, a denoising step's latent units. The mechanism charges it
    /// against the step's token budget and the phase's class.
    pub tokens: u32,
}

/// What is still in flight from earlier steps, as the mechanism grouped it
/// into numerical calls.
///
/// A batch is one numerical call on one block, so the policy cannot derive
/// these counts from its own dispatched work alone; they are the consumed part
/// of this step's room. `prefill_batches` is what today's
/// `PREFILL_WINDOW_CREDITS` bounds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Inflight {
    /// Batches submitted and not yet complete.
    #[serde(default)]
    pub batches: u32,
    /// Those of them carrying prefill work.
    #[serde(default)]
    pub prefill_batches: u32,
    /// Calls in them.
    #[serde(default)]
    pub calls: u32,
    /// Tokens they were charged.
    #[serde(default)]
    pub tokens: u32,
}

/// The room this step has left (design §4.4).
///
/// The policy emits a `Plan` that fits it; a plan over the step's token, call,
/// graph-tier or slot limits is a policy failure, and a plan whose work does
/// not fit the free KV units after `reclaim` leaves that work to the next step.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Room {
    /// The step number this room belongs to.
    #[serde(default)]
    pub step: u64,
    /// Free device KV cache units.
    #[serde(default)]
    pub kv_units: u64,
    /// Free request slots.
    #[serde(default)]
    pub slots: usize,
    /// This step's token cap (the policy's `max_num_batched_tokens`, already
    /// clamped to the build config's ceiling).
    #[serde(default)]
    pub token_budget: u32,
    /// This step's call cap (`max_batch`).
    #[serde(default)]
    pub call_budget: usize,
    /// The captured graph tiers, in tokens, ascending.
    #[serde(default)]
    pub graph_tiers: Vec<u32>,
    /// The KV page size in tokens, so a policy can clip a prefill chunk at a
    /// page boundary.
    #[serde(default)]
    pub page_tokens: u32,
    /// While an epoch warms, the tokens that may still newly write the prefix
    /// cache; `None` outside warming.
    #[serde(default)]
    pub prefill_tokens: Option<u32>,
    /// Work submitted by earlier steps and still in flight.
    #[serde(default)]
    pub inflight: Inflight,
}

/// Which storage is short.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "storage", rename_all = "snake_case")]
pub enum Storage {
    /// Device KV cache units.
    KvUnits,
    /// Host KV page pool.
    HostPages,
    /// Encoder cache.
    EncoderEntries,
    /// A storage this policy does not know.
    #[serde(other)]
    Other,
}

/// A shortage the mechanism asks the policy to relieve (design §4.4).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Shortage {
    pub kind: Storage,
    /// Units, pages or entries needed.
    pub need: u64,
}

/// An object the mechanism frees, in the order the policy returns them.
///
/// A request is first offloaded to the host KV cache; when there is no room
/// there it ends as `Migratable`, and the frontend resubmits it with the tokens
/// it generated. Pages and encoder entries are simply freed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Victim {
    Request(RequestId),
    Page(PageId),
    Encoding(EncodingId),
}

impl Victim {
    /// The page a victim frees, if it is one.
    pub fn page(self) -> Option<PageId> {
        match self {
            Victim::Page(page) => Some(page),
            _ => None,
        }
    }
}
