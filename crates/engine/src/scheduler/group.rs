//! The worker groups a scheduler places requests on (`specs/scheduler-design.md` §2.1
//! `WorkerGroup` and `Capacity`).
//!
//! A worker group is the ranks that run the same blocks under one queue depth. Its
//! [`WorkerGroup`] row holds what the deployment configures for it, what its executors stated
//! when they were ready ([`Capacity`]), how the epoch's session with the buffer pool states it
//! (its [`GroupId`], memory, slot region and per class its device regions), and the occupancy the
//! scheduler counts. Occupancy, room and the placement candidates of a phase are the group's
//! rules of SD §3.4 and §3.5.

use ipc::batch::{Phase, Shape};
use proto::config::{KvCacheConfig, WorkerGroupConfig};
use storage::runtime::{self, GroupId};

/// One worker group of a scheduler.
///
/// Invariants: `in_flight <= config.queue_depth` and `occupied[p] <= capacity.rows[p]` for every
/// phase index `p`.
#[derive(Clone, Debug)]
pub struct WorkerGroup {
    /// The group as the deployment configures it: its role, ranks and queue depth.
    pub config: WorkerGroupConfig,
    /// What the group's executors serve, as they stated it once they were ready.
    pub capacity: Capacity,
    /// The group as the epoch's session with the buffer pool states it: the [`GroupId`] batches,
    /// requests and stamps name it by, the device memory its ranks address, its slot region, and
    /// per class the device regions its owners' blocks are taken from, as indices into the
    /// session's regions (one per class, one per backing rank of a product, none for a class the
    /// group lacks).
    pub layout: runtime::Group,
    /// Per phase, by [`Phase::index`], the requests the stage serving it holds now.
    pub occupied: [u32; Phase::COUNT],
    /// Batches submitted to the group whose output has not been applied.
    pub in_flight: u32,
    /// Requests placed on the group.
    pub num_requests: u32,
}

impl WorkerGroup {
    /// A group with nothing placed on it and nothing in flight.
    pub fn new(config: WorkerGroupConfig, capacity: Capacity, layout: runtime::Group) -> Self {
        Self {
            config,
            capacity,
            layout,
            occupied: [0; Phase::COUNT],
            in_flight: 0,
            num_requests: 0,
        }
    }

    /// The number the session gives the group.
    pub fn id(&self) -> GroupId {
        self.layout.id
    }

    /// The device regions the group's owners take blocks of `class` from, as indices into the
    /// session's regions; empty for a class the group lacks.
    pub fn regions(&self, class: u8) -> &[u8] {
        self.layout
            .regions
            .get(usize::from(class))
            .map_or(&[], Vec::as_slice)
    }
}

/// What a worker group's executors serve, stated once by them when they are ready.
///
/// Immutable after construction, and every rank of a group states the same value. The sizes are
/// the executors' for the model, from which the scheduler sizes every sized block
/// (`specs/buffer-pool.md` §11 "Sizes").
#[derive(Clone, Debug, PartialEq)]
pub struct Capacity {
    /// The phases the group serves: bit [`Phase::index`] per phase.
    pub phases: u32,
    /// Request slots the group holds at once.
    pub slots: u32,
    /// The cache groups with their kind, block size, window and seeds, and the state and boundary
    /// intervals, from which keys and the block rules are computed; `None` for a group that holds
    /// no cache.
    pub kv_cache: Option<KvCacheConfig>,
    /// The shapes the group's executors captured; a batch must fit one of them.
    pub buckets: Vec<Shape>,
    /// Per phase, by [`Phase::index`], the requests the stage serving it holds at once.
    pub rows: [u32; Phase::COUNT],
    /// The longest sequence a request may reach on the group, in tokens.
    pub max_model_len: u32,
    /// The denoising steps of a video request.
    pub denoise_steps: u32,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
    /// The bytes of one token of an encoder output the group computes; 0 when it encodes nothing.
    pub feature_bytes: u32,
    /// Per captured media shape of a group that denoises: the bytes of a slot's trajectory, and
    /// of each product kind's block in the order of the product classes.
    pub media_bytes: Vec<(Shape, u64, Vec<u64>)>,
    /// The calls of one request the group holds unresolved at once: the bound on a request's
    /// pipelined decodes, `min(queue_depth * calls, max(queue_depth, 2))` for batches of up to
    /// `calls` calls (`worker::memory::division::call_window`).
    pub call_window: u32,
}
