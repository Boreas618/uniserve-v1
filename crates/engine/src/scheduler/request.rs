//! One request of a scheduler from its arrival to its end (`specs/scheduler-design.md` §2.2):
//! [`Request`], its status ([`RequestStatus`]) and its table of each class at the buffer pool as
//! the scheduler counts it ([`Blocks`]).
//!
//! A request is one row of plain fields, vLLM's `Request`: what the daemon submitted, where the
//! request runs, the calls of each phase scheduled and applied, its token and video progress, the
//! owners it holds at the buffer pool and the positions of each class's table. It holds no unit
//! and no run: a batch names its slot and positions, and the executor finds its blocks through
//! its device block tables. Everything else about it is derived from these fields (SD §2.2, §3.3).

use std::sync::Arc;
use std::time::Instant;

use ipc::batch::Phase;
use proto::request::{ArtifactMedia, Event, RequestId};
use storage::Key;
use storage::runtime::{GroupId, OwnerId};

/// A request of the scheduler.
///
/// Invariants: `completed[p] <= scheduled[p]` and `num_computed_tokens <= token_ids.len()`; no
/// call writes a block the request does not hold or one that is read-only; it holds one owner
/// per group it is attached to, from that group's `Attach` (or the `Transfer` that moves the
/// owner there) until that owner's `Detach`.
#[derive(Clone, Debug)]
pub struct Request {
    // What the daemon submitted. Immutable.
    pub id: RequestId,
    /// The token or video request: prompt, sampling, images, limits, priority, cache policy, and
    /// what a resumed request generated.
    pub input: proto::request::Request,
    pub arrival: Instant,

    // Where it is.
    pub status: RequestStatus,
    /// Per phase, by [`Phase::index`], the group that serves it; `None` until placed.
    pub placement: [Option<GroupId>; Phase::COUNT],
    /// Per group the request is attached to: its owner there (drawn at each admission), its slot
    /// row there (from `Attached` or `Transferred`; 0, a row no owner holds, until then), and
    /// whether that group has seen it.
    pub attachments: Vec<(GroupId, OwnerId, u32, bool)>,
    /// Per phase, the calls handed to executors.
    pub scheduled: [u32; Phase::COUNT],
    /// Per phase, the calls whose output was applied, in index order.
    pub completed: [u32; Phase::COUNT],
    /// Outputs that returned ahead of an earlier call of the same phase: the phase and the call's
    /// index.
    pub early: Vec<(Phase, u32)>,
    /// Per class, its table at the buffer pool as the scheduler counts it.
    pub blocks: Vec<Blocks>,
    /// Whether it is in the scheduler's waiting queue.
    pub queued: bool,
    /// The answers its admission or its transfer still waits for.
    pub pending: u32,

    // Token progress.
    /// The prompt, then the output; a resumed request's generated tokens are prompt.
    pub token_ids: Vec<u32>,
    pub num_prompt_tokens: u32,
    /// Tokens written or scheduled to be.
    pub num_computed_tokens: u32,
    /// The tokens of its prefix hit, over all tiers.
    pub num_cached_tokens: u32,
    /// Per cache group, the key of every complete block of `token_ids`.
    pub keys: Vec<Vec<Key>>,
    /// The drafts the next verification carries.
    pub spec_token_ids: Vec<u32>,
    /// Tokens the draft group has computed.
    pub num_draft_tokens: u32,
    /// Images whose encoding is placed.
    pub num_encoded_inputs: u32,
    /// The key of each encoder output, and the position the scheduler named for it in the
    /// request's encoder-output table, once held.
    pub encodings: Vec<(Key, Option<u32>)>,
    /// Output tokens the daemon's decoder has passed.
    pub num_acknowledged_tokens: u32,

    // Video progress: the counters above, and the finished artifact.
    pub artifact: Option<Arc<dyn ArtifactMedia>>,

    // Output and clocks.
    /// The terminal event, held until the calls in flight return when it is `Migratable`.
    pub terminal: Option<Event>,
    /// Events the consumer can still take.
    pub event_credit: u32,
    /// Its admission or its last applied call.
    pub last_progress: Instant,
    /// Whether its view changed since the policy was told.
    pub dirty: bool,
}

/// Where a request is (vLLM's `RequestStatus`).
///
/// Only `Running` requests are in batches; `Finished` is absorbing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestStatus {
    Waiting,
    /// Answers of its admission or its transfer are outstanding at the buffer pool.
    WaitingForRemoteKvs,
    Running,
    Finished,
}

/// A request's table of one class at the buffer pool, as the scheduler counts it.
///
/// Invariant: `first <= shared <= end <= asked`. The counts are positions in the table of the
/// owner on the group that writes the class; in a cache group's table a position is the logical
/// index of its block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Blocks {
    /// Entries before it were given up or never asked for: for a sliding window, its window's
    /// first block.
    pub first: u32,
    /// Entries before it are read-only: pinned, loaded, or committed through the executor.
    pub shared: u32,
    /// Entries held. Raised only by completions.
    pub end: u32,
    /// Entries asked for, held or not.
    pub asked: u32,
}

impl Request {
    /// The group the owner `owner` of this request is attached to, and its slot row there.
    pub fn attachment(&self, owner: OwnerId) -> Option<(GroupId, u32)> {
        self.attachments
            .iter()
            .find(|(_, attached, _, _)| *attached == owner)
            .map(|&(group, _, slot, _)| (group, slot))
    }

    /// The slot row of this request's own owner on `group`, if it is attached there.
    pub fn slot_on(&self, group: GroupId) -> Option<u32> {
        self.attachments
            .iter()
            .find(|(attached, _, _, _)| *attached == group)
            .map(|&(_, _, slot, _)| slot)
    }
}
