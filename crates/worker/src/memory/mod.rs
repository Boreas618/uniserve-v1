//! The main process's resident memory (`specs/worker.md` §4 `MainProcess.memory`).
//!
//! Resident memory is device storage the epoch keeps until it retires: the KV
//! cache planes, the latent pages, the retained weights and the encoder cache. Every one of
//! them is an exclusive storage at the host's Storage Server, keyed by the
//! inputs `crate::keys` composes, so a restarted main process of the same
//! scope and epoch maps what its predecessor left instead of loading it
//! again.

pub mod allocation;
pub mod encoders;
pub mod latent;
pub mod prefix_cache;
pub mod reload;
pub mod retained;
pub mod weights;

pub use prefix_cache::block_tables::{
    BlockTableStorage, BlockTableUpdate, BlockTables, GroupShape, GroupTable,
};
pub use prefix_cache::plan::{CacheGroupPlan, CachePlan, Checkpoint, GroupAxis};
pub use prefix_cache::planes::{CachePlanes, LaidSegment, PlaneLayout, UnitColumn};
pub use prefix_cache::PrefixCache;

use std::sync::Mutex;

use crate::lock;
use std::sync::Arc;

use platform::Stream;

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
    /// The KV cache planes, their block tables and their host pages, or `None` on a rank that
    /// keeps no paged cache: a media rank's resident state is each request's own footprint.
    pub prefix_cache: Option<Arc<Mutex<PrefixCache>>>,
    /// The device this rank's memory lives on, as the plan states it.
    pub device: crate::config::Device,
    /// The device's pools, which the scratch page set a paired forward writes
    /// is taken from: its bytes are booked against the same allowance as every
    /// other pool, so the shadow never writes memory the report cannot see.
    pub pools: Arc<Mutex<crate::arena::Pools>>,
}

/// What the main process hands [`Memory::new`]: the parts only the deployment
/// knows, beside the memory the capacity negotiation resolved.
///
/// Everything the resolved memory carries is read from it - the slot count, the
/// block-table width, the entry bytes of a product, the relay depth, the import
/// workers - so a caller does not repeat a figure it already states.
pub struct OpenMemory<'a> {
    /// The memory [`capacity::plan_rank`](crate::capacity::plan_rank) resolved.
    pub plan: &'a crate::capacity::MemoryConfig,
    /// The rank's device, or `None` for a host-only rank.
    pub device: Option<platform::DeviceId>,
    /// The device's stream: any reset the owners queue runs here, before the
    /// worker thread exists, and a stream accepts work without one.
    pub stream: Option<&'a Stream>,
    /// The conversion program the import path launches.
    pub conversion: Arc<dyn prefix_cache::conversion::PageConversion>,
    /// The device's pools, which a paired forward's scratch page set is taken
    /// from.
    pub pools: Arc<Mutex<crate::arena::Pools>>,
}

/// Release the resident cache segments of `cache`, which the Storage Server
/// keeps for the next epoch: what a sleep gives back and a wake maps again.
///
/// The device storage the segments held is booked as released before it is
/// given back (`ExclusiveStorage::release_segments`), so the ledger of the GPU
/// never counts storage no process maps. A pool that is not segmented has no
/// segment to give back, which is not an error.
///
/// **Only the segments go back.** The graph pool and the scratch region stay
/// mapped: a capture's addresses are fixed for the life of its pool, so a wake
/// is a remap and never a recapture, and a sleep that released a capture's book
/// while its mapping stayed would let the next capture book the same bytes
/// twice. A park therefore returns the KV and not the graphs.
///
/// One body for both callers: [`Memory::sleep`] on the control thread and the
/// device thread's own command handler, which holds the cache handle and no
/// `Memory`. A second copy of this body is the failure to avoid - it is where a
/// divergence would surface as a rank that wakes with unwritten pages.
///
/// A pool that is not segmented says so and returns: it has nothing to give
/// back, which is a different fact from a release that freed nothing, and the
/// line is what tells the two apart in a run's log.
pub fn sleep(cache: &Arc<Mutex<PrefixCache>>, storage: &mut dyn ExclusiveStorage) -> Result<()> {
    let planes = {
        let cache = lock(cache);
        Arc::clone(cache.planes())
    };
    if planes.segments().is_none() {
        tracing::info!(
            pool = "the cache's segments",
            "this cache holds one segment, so a sleep gives back nothing: the pool has no segment \
             to release and the graph pool and scratch stay as they are"
        );
        return Ok(());
    }
    planes.release_segments(0, storage)
}

/// Map the released cache segments of `cache` again, `limit` of them or all,
/// booking their storage first: a booking the GPU's quota refuses leaves the
/// rank asleep rather than half woken.
///
/// Resets the units of the segments it created, on `stream`, which is
/// synchronized before returning. A wake for a pool that never slept is a
/// no-op, not an error: there is nothing to map again.
pub fn wake(
    cache: &Arc<Mutex<PrefixCache>>,
    storage: &mut dyn ExclusiveStorage,
    stream: &Stream,
    limit: Option<u32>,
) -> Result<()> {
    let planes = {
        let cache = lock(cache);
        Arc::clone(cache.planes())
    };
    if planes.segments().is_none() {
        return Ok(());
    }
    planes.restore_segments(storage, stream, limit)
}

impl Memory {
    /// Allocates every resident owner the plan sized, in one call: the cache
    /// planes and block tables, on a rank that keeps a paged cache.
    ///
    /// This is the epoch's budget becoming real: each owner is booked against
    /// the device's allowance as it is mapped, so the first `mapped <= budget`
    /// reading after warmup counts every byte the plan charged.
    ///
    /// `storage` is the rank's Storage Server client; it stays the main
    /// process's, so `sleep` and `wake` are the only calls that reach the
    /// ledger. The resets the owners queue run on `request.stream` before any
    /// worker thread exists.
    pub fn new(request: OpenMemory<'_>, storage: &mut dyn ExclusiveStorage) -> Result<Self> {
        let plan = request.plan;
        let info = &plan.info;
        let cache = match (plan.cache.as_ref(), plan.planes.as_ref()) {
            (Some(cache), layout) => Some(prefix_cache::PrefixCache::new(
                Arc::clone(cache),
                info.prefix_cache
                    .as_ref()
                    .ok_or_else(|| {
                        crate::Error::Invariant(
                            "a cache plan without the reported unit count".into(),
                        )
                    })?
                    .num_units,
                layout,
                info.request_slots,
                plan.width,
                info.max_unresolved_calls as usize,
                request.device,
                request.stream,
                storage,
                Arc::clone(&request.conversion),
            )?),
            // A rank that keeps no paged cache opens no cache planes and no block tables.
            (None, _) => None,
        };
        Ok(Self {
            prefix_cache: cache.map(|cache| Arc::new(Mutex::new(cache))),
            device: request
                .device
                .map_or(crate::config::Device::Cpu, crate::config::Device::Cuda),
            pools: request.pools,
        })
    }

    /// Releases every owner: the cache planes give their segments back.
    ///
    /// The owners' storage returns itself as each owner goes, so this needs no
    /// storage handle; it is the main process's last call on `Memory`.
    pub fn close(self) -> Result<()> {
        let Memory {
            prefix_cache,
            device,
            pools,
        } = self;
        drop((device, pools));
        if let Some(cache) = prefix_cache {
            lock(&cache).close()?;
        }
        Ok(())
    }




    /// The bytes one token of this rank's pages takes in the host KV cache; zero on a rank that
    /// keeps no paged cache.
    pub fn host_token_bytes(&self) -> u64 {
        self.prefix_cache.as_ref().map_or(0, |cache| {
            prefix_cache::host::token_bytes(lock(cache).plan())
        })
    }

    /// How many cache groups this rank's pages come in.
    pub fn cache_groups(&self) -> usize {
        self.prefix_cache
            .as_ref()
            .map_or(0, |cache| lock(cache).plan().groups.len())
    }


    /// The device's pools, so a paired forward can take its scratch page set
    /// from the same allowance as every other pool.
    pub fn pools(&self) -> Arc<Mutex<crate::arena::Pools>> {
        Arc::clone(&self.pools)
    }

    /// The cache pool as a reader shares it: the runner's provider holds this
    /// clone for its whole life and takes the lock per call, so a read never
    /// sees a half-installed table. The worker is the only writer.
    ///
    /// `None` on a rank that keeps no paged cache, which is what its scheduler
    /// reads to know it takes no paged request.
    pub fn shared_cache(&self) -> Option<Arc<Mutex<PrefixCache>>> {
        self.prefix_cache.clone()
    }
}
