//! The main process's resident memory (`specs/worker.md` §4 `MainProcess.memory`).
//!
//! Resident memory is device storage the epoch keeps until it retires: the KV
//! cache planes, the latent pages, the products of cross-call tensors, the
//! request slots, the retained weights and the encoder cache. Every one of
//! them is an exclusive storage at the host's Storage Server, keyed by the
//! inputs `crate::keys` composes, so a restarted main process of the same
//! scope and epoch maps what its predecessor left instead of loading it
//! again.

pub mod allocation;
pub mod buffer;
pub mod encoders;
pub mod latent;
pub mod output;
pub mod pinned_pool;
pub mod prefix_cache;
pub mod products;
pub mod request_slots;
pub mod retained;

pub use latent::LatentPlan;
pub(crate) use prefix_cache::block_tables::AttentionRow;
pub use prefix_cache::block_tables::{
    BlockTableStorage, BlockTableUpdate, BlockTables, GroupShape, GroupTable,
};
pub use prefix_cache::plan::{CacheDescription, CacheGroupPlan, CachePlan, GroupAxis, StateField};
pub use prefix_cache::planes::{CachePlanes, LaidSegment, PlaneLayout, UnitColumn};
pub use prefix_cache::{PendingImport, PrefixCache};
pub(crate) use prefix_cache::{TransferBases, UnitAccesses};

use std::sync::Arc;

use platform::Stream;

use self::buffer::BufferPool;
use self::latent::LatentStorage;
use self::pinned_pool::PinnedPool;
use self::products::ProductStore;
use self::request_slots::RequestSlots;
use self::retained::ExclusiveStorage;
use crate::Result;

/// The main process's resident memory: the owners an epoch keeps until it
/// retires (`specs/worker.md` §4).
///
/// Each owner is built by the process that knows its geometry and handed here,
/// so that the main process holds one value for what a sleep gives back and a
/// wake restores. The retained weights and the encoder cache join as their
/// loaders publish; their storage is already keyed by `crate::keys`.
pub struct Memory {
    /// The KV cache planes, their block tables and their host pages.
    pub prefix_cache: PrefixCache,
    /// The latent pages of diffusion and media calls.
    pub latents: LatentStorage,
    /// The products of cross-call tensors.
    pub products: ProductStore,
    /// The banks of the layers a request keeps per slot.
    pub slots: RequestSlots,
    /// The device buffers a call leases.
    pub buffer: BufferPool,
    /// The page-locked host blocks copies go through.
    pub pinned: Arc<PinnedPool>,
}

impl Memory {
    /// Release the resident cache segments, which the Storage Server keeps for
    /// the next epoch: what a sleep gives back and a wake maps again.
    ///
    /// The device storage the segments held is booked as released before it is
    /// given back (`ExclusiveStorage::release_segments`), so the ledger of the
    /// GPU never counts storage no process maps.
    pub fn sleep(&mut self, storage: &mut dyn ExclusiveStorage) -> Result<()> {
        if self.prefix_cache.planes().segments().is_none() {
            return Ok(());
        }
        self.prefix_cache.planes().release_segments(0, storage)
    }

    /// Map the released cache segments again, `limit` of them or all, booking
    /// their storage first: a booking the GPU's quota refuses leaves the rank
    /// asleep rather than half woken.
    ///
    /// Resets the units of the segments it created, on `stream`, which is
    /// synchronized before returning.
    pub fn wake(
        &mut self,
        storage: &mut dyn ExclusiveStorage,
        stream: &Stream,
        limit: Option<u32>,
    ) -> Result<()> {
        self.prefix_cache
            .planes()
            .restore_segments(storage, stream, limit)
    }
}
