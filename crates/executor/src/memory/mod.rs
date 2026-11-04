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
pub mod weights;

pub use latent::LatentPlan;
pub use prefix_cache::block_tables::{
    BlockTableStorage, BlockTableUpdate, BlockTables, GroupShape, GroupTable,
};
pub use prefix_cache::plan::{CacheDescription, CacheGroupPlan, CachePlan, GroupAxis, StateField};
pub use prefix_cache::planes::{CachePlanes, LaidSegment, PlaneLayout, UnitColumn};
pub use prefix_cache::{PendingImport, PrefixCache};
pub(crate) use prefix_cache::{TransferBases, UnitAccesses};

use std::sync::Mutex;

use crate::lock;
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
    pub prefix_cache: Arc<Mutex<PrefixCache>>,
    /// The latent pages of diffusion and media calls, on a rank that stores a pool.
    pub latents: Option<LatentStorage>,
    /// The products of cross-call tensors.
    pub products: ProductStore,
    /// The banks of the layers a request keeps per slot.
    pub slots: RequestSlots,
    /// The page-locked host blocks copies go through.
    pub pinned: Arc<PinnedPool>,
}

/// What the main process hands [`Memory::new`]: the sized plan and the
/// deployment's own values, so the control thread never assembles the owners
/// itself.
pub struct MemoryRequest<'a> {
    /// The capacity plan this rank resolved (`capacity::Plan`).
    pub plan: &'a crate::capacity::Plan,
    /// The deployed blocks' slot fields, from the model.
    pub slots: &'a [self::request_slots::SlotField],
    /// The rank's device, or `None` for a host-only rank.
    pub device: Option<platform::DeviceId>,
    /// The device's stream: any reset the owners queue runs here, before the
    /// executor thread exists, and a stream accepts work without one.
    pub stream: Option<&'a Stream>,
    /// The block-table width the batch plan holds.
    pub width: u32,
    /// The request slots the rank keeps.
    pub request_pool_size: u32,
    /// The devices the product store's arenas live on.
    pub devices: Vec<crate::memory::allocation::MemoryKind>,
    /// The bytes of one product store entry.
    pub max_feature_bytes: u64,
    /// The relay entries per request slot.
    pub relay_depth: usize,
    /// The cache import workers the pool runs.
    pub import_capacity: usize,
    /// The conversion program the import path launches.
    pub conversion: Arc<dyn prefix_cache::conversion::PageConversion>,
    /// The pool's completion events.
    pub events: Arc<crate::stream::EventPool>,
}

/// What the main process hands [`open_memory`]: the parts only the deployment
/// knows, beside the plan the capacity negotiation resolved.
///
/// Everything the plan carries is read from it - the slot count, the entry
/// bytes of a product, the relay depth, the import workers - so a caller does
/// not repeat a figure the plan already states.
pub struct OpenMemory<'a> {
    /// The plan [`capacity::token_plan`](crate::capacity::token_plan) resolved.
    pub plan: &'a crate::capacity::Plan,
    /// The deployed blocks' slot fields, from the model.
    pub slots: &'a [self::request_slots::SlotField],
    /// The rank's device, or `None` for a host-only rank.
    pub device: Option<platform::DeviceId>,
    /// The device's stream the owners' resets run on.
    pub stream: Option<&'a Stream>,
    /// The block-table width the batch plan holds.
    pub width: u32,
    /// The devices the product store's arenas live on.
    pub devices: Vec<crate::memory::allocation::MemoryKind>,
    /// The conversion program the import path launches.
    pub conversion: Arc<dyn prefix_cache::conversion::PageConversion>,
    /// The pool's completion events.
    pub events: Arc<crate::stream::EventPool>,
}

/// Build every resident owner the plan sized: the cache planes and block
/// tables, the latent pages, the product store, the request slots, the buffer
/// pool and the page-locked blocks.
///
/// This is the epoch's budget becoming real: each owner is booked against the
/// device's allowance as it is mapped, so the first `mapped <= budget` reading
/// after warmup counts every byte the plan charged. `storage` stays the main
/// process's client; `sleep` and `wake` are the only calls that reach the
/// ledger.
pub fn open_memory(request: OpenMemory<'_>, storage: &mut dyn ExclusiveStorage) -> Result<Memory> {
    let plan = request.plan;
    let info = &plan.info;
    Memory::new(
        MemoryRequest {
            plan,
            slots: request.slots,
            device: request.device,
            stream: request.stream,
            width: request.width,
            request_pool_size: info.request_slots,
            devices: request.devices,
            max_feature_bytes: u64::from(info.encoder_entry_bytes),
            relay_depth: plan.arena.relay_depth,
            import_capacity: info.max_unresolved_calls as usize,
            conversion: request.conversion,
            events: request.events,
        },
        storage,
    )
}

impl Memory {
    /// Allocates every resident owner the plan sized, in one call: the cache
    /// planes and block tables, the latent pages, the product store, the
    /// request slots, the buffer pool and the page-locked blocks.
    ///
    /// `storage` is the rank's Storage Server client; it stays the main
    /// process's, so `sleep` and `wake` are the only calls that reach the
    /// ledger. The resets the owners queue run on `request.stream` before any
    /// executor thread exists.
    pub fn new(request: MemoryRequest<'_>, storage: &mut dyn ExclusiveStorage) -> Result<Self> {
        let plan = request.plan;
        let info = &plan.info;
        let cache = match (plan.cache.as_ref(), plan.planes.as_ref()) {
            (Some(cache), layout) => prefix_cache::PrefixCache::new(
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
                request.request_pool_size,
                request.width,
                request.import_capacity,
                request.device,
                request.stream,
                storage,
                Arc::clone(&request.conversion),
            )?,
            (None, _) => {
                return Err(crate::Error::Invalid(
                    "a rank without a cache plan needs no Memory".into(),
                ));
            }
        };
        let latents = match plan.latent.as_ref() {
            Some(latent) => Some(LatentStorage::allocate(
                latent,
                request.device,
                request.stream,
                "latent",
                storage,
            )?),
            None => None,
        };
        // One buffer pool per rank, given to the product store, which binds the scheduler's
        // placements into it; a second pool would allocate the same bytes twice.
        let buffer = BufferPool::allocate(
            plan.info.buffer_pool_bytes.max(1),
            &request.devices,
            storage,
        )?;
        let products = ProductStore::new(
            usize::try_from(plan.arena.device_product_bytes.max(1))
                .map_err(|_| crate::Error::Invalid("product store capacity overflows".into()))?,
            usize::try_from(plan.info.encoder_entry_bytes.max(1))
                .map_err(|_| crate::Error::Invalid("product entry size overflows".into()))?,
            request.devices.clone(),
            usize::try_from(plan.info.request_slots).unwrap_or(0),
            plan.arena.relay_depth,
            buffer,
            Arc::clone(&request.events),
        )?;
        let slots = RequestSlots::allocate(request.slots, plan.info.request_slots, request.device)?;
        let pinned = PinnedPool::new(request.device);
        Ok(Self {
            prefix_cache: Arc::new(Mutex::new(cache)),
            latents,
            products,
            slots,
            pinned,
        })
    }

    /// Releases every owner, in the design's order: the products and the
    /// encoders first, then the cache planes (which give their segments back),
    /// the latents, the slots, the buffer pool and the page-locked blocks.
    ///
    /// The owners' storage returns itself as each owner goes, so this needs no
    /// storage handle; it is the main process's last call on `Memory`.
    pub fn close(self) -> Result<()> {
        let Memory {
            prefix_cache,
            latents,
            mut products,
            slots,
            pinned,
        } = self;
        // The products go first: their reads retain views into the cache, and the cache's
        // segments are the last thing an epoch gives back.
        let drained = products.close();
        drop(drained);
        lock(&prefix_cache).close()?;
        drop(latents);
        drop(slots);
        drop(pinned);
        Ok(())
    }

    /// The device bytes this rank's owners map, which the metrics report states
    /// as the `resident` pool.
    ///
    /// It is the cache planes, the latent pages, the product store's arenas and
    /// the request-slot banks, and it counts MAPPED device storage only: host and
    /// page-locked storage is outside the device's budget, and a sleeping rank's
    /// released segments are not in it. It excludes the graph pools and the
    /// cache's scratch page set as well: those are the executor's and the
    /// paired forward's own regions, so `graph + resident + scratch <= budget`
    /// is three disjoint figures of one device allowance. It reads the owners'
    /// books and asks no driver, so a figure that could not be computed is left
    /// out rather than failing the report.
    pub fn mapped_resident(&self) -> u64 {
        let planes = lock(&self.prefix_cache).mapped_bytes();
        let latents = self
            .latents
            .as_ref()
            .map_or(0, |latents| latents.mapped_bytes());
        planes + latents + self.products.mapped_bytes() + self.slots.mapped_bytes()
    }

    /// Release the resident cache segments, which the Storage Server keeps for
    /// the next epoch: what a sleep gives back and a wake maps again.
    ///
    /// The device storage the segments held is booked as released before it is
    /// given back (`ExclusiveStorage::release_segments`), so the ledger of the
    /// GPU never counts storage no process maps.
    pub fn sleep(&mut self, storage: &mut dyn ExclusiveStorage) -> Result<()> {
        if lock(&self.prefix_cache).planes().segments().is_none() {
            return Ok(());
        }
        lock(&self.prefix_cache)
            .planes()
            .release_segments(0, storage)
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
        lock(&self.prefix_cache)
            .planes()
            .restore_segments(storage, stream, limit)
    }

    /// The cache pool as a reader shares it: the runner's provider holds this
    /// clone for its whole life and takes the lock per call, so a read never
    /// sees a half-installed table. The executor is the only writer.
    pub fn shared_cache(&self) -> Arc<Mutex<PrefixCache>> {
        Arc::clone(&self.prefix_cache)
    }
}
