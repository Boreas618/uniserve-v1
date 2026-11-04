//! A device's allocator: one reserved virtual address range per pool, device
//! storage created and mapped on demand, freed blocks kept and given out
//! again, and every mapping booked against the device's budget before it is
//! made (design 4.1).
//!
//! There is no framework cache allocator behind this crate, so every byte of
//! device storage that is not a weight held by the Storage Server goes through
//! an [`Allocator`]: a captured graph's activations (the graph pool), an
//! operator's workspace (the scratch pool), and the storage that lives for an
//! epoch (the resident pool: KV planes, latents, products, request slots).
//!
//! A pool reserves virtual addresses once ([`Allocator::reserve`]). Physical
//! storage is created only when a block is first allocated, and a block that
//! is freed stays mapped at its address and is given out again, so the
//! addresses a captured graph holds never move while the pool lives. Physical
//! storage returns to the driver when the pool is dropped.
//!
//! The budget is the Storage Server's per-device ledger, which the executor
//! implements over its storage client ([`Budget`]); this crate never talks to
//! the Server. Every mapping books the pool's new cumulative total first, so
//! an allocation that would exceed the device's budget fails before any
//! storage is created and leaves the pool exactly as it was.

use std::fmt;
use std::os::fd::OwnedFd;
use std::sync::Arc;

use crate::pool::VirtualPool;
use crate::vmm::{HandleType, ShareableHandle};
use crate::{Device, DeviceId, Error, PrimaryContext, Protection, Result};

const POOLS: usize = 3;

/// The three uses device storage is separated into, each with its own
/// reserved range (design 4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PoolKind {
    /// Activations a captured graph reads and writes. Their addresses must
    /// stay fixed for the graph's life, and the generations of one stream's
    /// graphs share the same range because the stream runs them in order.
    Graph,
    /// An operator's workspace, valid for one launch.
    Scratch,
    /// Storage that lives as long as the epoch: KV planes, latents, products
    /// and request slots, named and retained by the Storage Server's rules.
    Resident,
}

impl PoolKind {
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Graph => 0,
            Self::Scratch => 1,
            Self::Resident => 2,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Graph => "graph",
            Self::Scratch => "scratch",
            Self::Resident => "resident",
        }
    }
}

/// A device's budget, booked as the Storage Server's ledger books it: `bytes`
/// is the new cumulative total for `(device, pool)`, not a delta, so a
/// booking either fits and is recorded or is refused and changes nothing.
///
/// The executor implements this over the storage client's booking request;
/// `platform` never talks to the Storage Server itself.
pub trait Budget: Send + Sync {
    /// Books `bytes` as the total of `pool` on `device`, failing with no room.
    fn book(&self, device: DeviceId, pool: PoolKind, bytes: u64) -> Result<()>;
}

/// The physical storage behind a [`Buffer`]: the driver's allocation handle,
/// how it is shared, and the context that owns it.
///
/// A block outlives the buffers that describe it inside its pool, which keeps
/// it mapped and gives it out again after it is freed.
pub(crate) struct Block {
    raw: u64,
    bytes: usize,
    handle_type: HandleType,
    context: PrimaryContext,
}

impl Block {
    pub(crate) fn new(
        raw: u64,
        bytes: usize,
        handle_type: HandleType,
        context: PrimaryContext,
    ) -> Self {
        Self {
            raw,
            bytes,
            handle_type,
            context,
        }
    }

    pub(crate) fn raw(&self) -> u64 {
        self.raw
    }

    /// The storage this block maps, rounded up to the allocation granularity.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    /// Exports one reference to the block as a POSIX file descriptor, which a
    /// Unix socket passes to the Storage Server or to another process.
    pub(crate) fn export(&self) -> Result<OwnedFd> {
        match ShareableHandle::export(self.raw, self.handle_type)? {
            ShareableHandle::Descriptor(descriptor) => Ok(descriptor),
            ShareableHandle::Fabric(_) => Err(Error::invalid(format!(
                "{}'s allocator maps POSIX descriptor handles, not fabric handles",
                self.context.device()
            ))),
        }
    }
}

/// A block of device storage given out by an [`Allocator`]: its address, the
/// storage mapped there, its pool, and the descriptor it exports to other
/// processes.
///
/// The buffer borrows the block from its pool; it is valid until it is passed
/// back to [`Allocator::free`] or the allocator is dropped, whichever comes
/// first.
pub struct Buffer {
    block: Arc<Block>,
    address: u64,
    device: DeviceId,
    pool: PoolKind,
}

impl Buffer {
    /// The virtual address the block is mapped at, which stays the same for the
    /// pool's life.
    pub fn address(&self) -> u64 {
        self.address
    }

    /// The storage of the block, rounded up to the allocation granularity.
    pub fn len(&self) -> usize {
        self.block.bytes()
    }

    pub fn is_empty(&self) -> bool {
        self.block.bytes() == 0
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn pool(&self) -> PoolKind {
        self.pool
    }

    /// Exports one reference to the block's physical storage as a POSIX file
    /// descriptor (`cuMemExportToShareableHandle`). Closing the descriptor does
    /// not affect the mapping.
    ///
    /// This is the first half of design 8's NVML rule: the owner imports the
    /// descriptor and maps it, then drops the handle that created the storage,
    /// so NVML attributes the storage to the importer and not to the process
    /// that created it. The Storage Server takes resident storage that way.
    pub fn export(&self) -> Result<OwnedFd> {
        self.block.export()
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Buffer")
            .field("address", &format_args!("{:#x}", self.address))
            .field("len", &self.block.bytes())
            .field("device", &self.device)
            .field("pool", &self.pool)
            .finish()
    }
}

/// One device's allocator; see the module documentation.
pub struct Allocator {
    device: Device,
    budget: Arc<dyn Budget>,
    /// One reserved range per pool kind, indexed by [`PoolKind::index`];
    /// `None` until the pool is reserved.
    pools: [Option<VirtualPool>; POOLS],
}

impl Allocator {
    /// An allocator for `device` that books every mapping against `budget`. No
    /// pool is reserved yet; [`Self::reserve`] sets each one's range.
    pub fn new(device: &Device, budget: Arc<dyn Budget>) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            budget,
            pools: [None, None, None],
        })
    }

    pub fn device(&self) -> DeviceId {
        self.device.id()
    }

    /// Reserves `bytes` of virtual addresses for `pool`, rounded up to the
    /// allocation granularity; nothing is mapped and nothing is booked until
    /// the first allocation from the pool. Reserving a pool that already has a
    /// range fails.
    pub fn reserve(&mut self, pool: PoolKind, bytes: usize) -> Result<()> {
        let index = pool.index();
        if self.pools[index].is_some() {
            return Err(Error::invalid(format!(
                "the {} pool of {} is already reserved",
                pool.name(),
                self.device.id()
            )));
        }
        // The design gives every block a POSIX descriptor handle so that the
        // Storage Server, or another process on this host, can import it.
        let range = VirtualPool::with_budget(
            self.device.id(),
            bytes,
            HandleType::Descriptor,
            pool,
            Arc::clone(&self.budget),
        )?;
        self.pools[index] = Some(range);
        Ok(())
    }

    /// Storage of at least `bytes` from `pool`, at an address the pool keeps:
    /// a freed block of the same rounded size, or new storage created and
    /// mapped, whose mapping is booked against the device's budget first.
    ///
    /// # Errors
    ///
    /// Fails when the pool is not reserved, when the reservation is exhausted,
    /// when the driver cannot create or map the storage, and when the booking
    /// refuses the pool's new total — in which case nothing was created or
    /// mapped.
    pub fn allocate(&mut self, pool: PoolKind, bytes: usize) -> Result<Buffer> {
        let allocator = self.pool(pool)?;
        let (address, block) = allocator.allocate_block(bytes)?;
        Ok(Buffer {
            block,
            address,
            device: self.device.id(),
            pool,
        })
    }

    /// Returns `buffer`'s block to its pool, which keeps it mapped and gives it
    /// out again. A buffer whose allocator was dropped is ignored.
    pub fn free(&mut self, buffer: Buffer) {
        if let Some(allocator) = self.pools[buffer.pool.index()].as_mut() {
            allocator.free(buffer.address);
        }
    }

    /// Changes what the pool's device and its peers may do with `buffer`'s
    /// storage, at its unchanged address: design 8's park makes a standby's
    /// blocks read-only and promotion makes them read-write again, so a
    /// captured graph that addresses them stays valid across the pair.
    ///
    /// # Errors
    ///
    /// Fails when `buffer` belongs to another allocator or pool, when its pool
    /// is suspended, and when the driver refuses the change.
    pub fn set_protection(&mut self, buffer: &Buffer, protection: Protection) -> Result<()> {
        if buffer.device() != self.device.id() {
            return Err(Error::invalid(format!(
                "{buffer:?} belongs to {}, not to {}",
                buffer.device(),
                self.device.id()
            )));
        }
        self.pool(buffer.pool())?
            .protect(buffer.address(), protection)
    }

    /// The physical storage `pool` maps now, or would map once restored:
    /// exactly what has been booked for it.
    pub fn mapped_bytes(&self, pool: PoolKind) -> usize {
        self.pools[pool.index()]
            .as_ref()
            .map_or(0, VirtualPool::mapped_bytes)
    }

    /// Suspends every reserved pool for a parked standby (design 2.5): each
    /// block's contents are copied to page-locked host storage, the physical
    /// storage is released at unchanged addresses, and each pool's booked total
    /// drops to what it maps, so the parked process does not hold the device's
    /// budget for storage it no longer uses.
    ///
    /// Nothing may read or write a buffer between this call and
    /// [`Self::restore`]. A buffer keeps its address and its contents across
    /// the pair, but a descriptor exported before it is invalid afterwards,
    /// since the block's physical storage was released.
    ///
    /// # Errors
    ///
    /// Fails when a copy or an unmap fails; the blocks suspended before the
    /// failure stay suspended and are restored by [`Self::restore`].
    pub fn suspend(&mut self) -> Result<()> {
        let mut outcome = Ok(());
        for pool in self.pools.iter_mut().flatten() {
            outcome = outcome.and(pool.suspend());
        }
        outcome
    }

    /// Maps new physical storage at every suspended block's address and copies
    /// its saved contents back, booking each pool's new total before each
    /// mapping. Restoring a pool that is not suspended does nothing.
    ///
    /// # Errors
    ///
    /// Fails when the driver cannot create or map a block's storage, when a
    /// pool's budget refuses the new total, or when a copy fails; the blocks
    /// not yet restored stay suspended.
    pub fn restore(&mut self) -> Result<()> {
        let mut outcome = Ok(());
        for pool in self.pools.iter_mut().flatten() {
            outcome = outcome.and(pool.restore());
        }
        outcome
    }

    fn pool(&mut self, pool: PoolKind) -> Result<&mut VirtualPool> {
        self.pools[pool.index()].as_mut().ok_or_else(|| {
            Error::invalid(format!(
                "the {} pool of {} is not reserved",
                pool.name(),
                self.device.id()
            ))
        })
    }
}
