//! What a policy decides: the output of one step, and the victims of a shortage.

use serde::{Deserialize, Serialize};

use crate::ids::{EncodingId, PageId, PhaseId, RequestId};

/// One step's decision (design §4.4). The mechanism executes the entries in
/// order.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DispatchOutput {
    /// Waiting requests to admit, in order. The mechanism admits each; the
    /// first one it cannot place and the ones after it stay waiting.
    pub admit: Vec<RequestId>,
    /// Waiting requests to reject as over capacity.
    pub reject: Vec<RequestId>,
    /// This step's work, in the order it runs: which request, which phase, how
    /// many tokens. A prefill phase's token count is its chunk size, so a
    /// policy that sizes chunks by decode latency or batches by SLO does it in
    /// this one decision.
    pub work: Vec<Work>,
    /// Device pages to write to the host KV cache beyond the mechanism's own
    /// write policy, in order. The mechanism offers each page named to the
    /// host once: a page already on the host or being written there, or one
    /// whose last write stored nothing while nothing that could change that
    /// has happened since, is skipped. A page is named in the step it should
    /// be written, not again in every step after it.
    pub offload: Vec<PageId>,
    /// Requests whose prefix hit to deepen through KV storage, in order. The
    /// mechanism deepens only by at least [`crate::Limits::prefetch_min_tokens`].
    pub prefetch: Vec<RequestId>,
    /// Requests whose draft cache to refill, in order.
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

/// What earlier steps submitted and the mechanism has not completed.
///
/// A batch is one numerical call on one block, so a policy cannot derive these
/// counts from its own dispatched work alone. They are not spent out of this
/// step's token and call budgets — those are per step — but they are how many
/// prefill batches the step already has outstanding, which a policy's prefill
/// window bounds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Inflight {
    /// Batches submitted and not yet complete.
    pub batches: u32,
    /// Those of them carrying prefill work.
    pub prefill_batches: u32,
}

/// The room this step has left (design §4.4).
///
/// The policy emits a `DispatchOutput` that fits it; a dispatch output over the step's token, call,
/// graph-tier or slot limits is a policy failure, and a dispatch output whose work does
/// not fit the free KV units after `preempt` leaves that work to the next step.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Room {
    /// The step number this room belongs to.
    pub step: u64,
    /// Free device KV cache units.
    pub kv_units: u64,
    /// Free request slots.
    pub slots: usize,
    /// This step's token cap (the deployment's `max_num_batched_tokens`, clamped to the
    /// captured graphs' ceiling). It is the step's own budget: work submitted in earlier steps
    /// was charged to those steps.
    pub token_budget: u32,
    /// This step's call cap (`scheduler_config.max_batch`, clamped likewise).
    pub call_budget: usize,
    /// The most prompt tokens one request prefills in this step
    /// (`scheduler_config.chunked_prefill_size`, at most `token_budget`).
    pub chunk_tokens: u32,
    /// The captured graph tiers, in tokens, ascending.
    pub graph_tiers: Vec<u32>,
    /// The KV page size in tokens, so a policy can clip a prefill chunk at a
    /// page boundary.
    pub page_tokens: u32,
    /// While an epoch warms, the tokens that may still newly write the prefix
    /// cache; `None` outside warming.
    pub prefill_tokens: Option<u32>,
    /// Work submitted by earlier steps and still in flight.
    pub inflight: Inflight,
    /// The device KV cache's total units, so a policy can reject a request
    /// whose worst-case reservation can never fit.
    pub capacity_units: u64,
}

/// Which storage is short.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Storage {
    /// Device KV cache units.
    KvUnits,
    /// Device units of recurrent-state checkpoints, one checkpoint a unit, in a region of their
    /// own. The pages a policy is shown are KV pages, which free none of these units, and a
    /// cached checkpoint no request holds is the mechanism's to evict: only a running request's
    /// end frees the checkpoints it holds.
    StateUnits,
    /// Host KV page pool.
    HostPages,
    /// Encoder cache.
    EncoderEntries,
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
