//! A pool of device storage at stable addresses: a reserved virtual address
//! range whose physical storage is mapped as it is used, and which can be
//! released and mapped back at the same addresses.
//!
//! A CUDA graph holds the addresses of the storage its nodes read and write,
//! so the storage behind a graph must stay at its addresses for the graph's
//! life. A [`VirtualPool`] gives out chunks of one reservation, each its own
//! physical allocation mapped at an address that never changes. Freed
//! chunks stay mapped and are given out again by the same pool, so the
//! physical storage returns to the driver only when the pool is dropped.
//!
//! Every device that can access the pool's device's storage may read and
//! write the chunks: a module placed on another device (a generation tower)
//! copies its inputs from tensors a capture allocated in the pool.
//!
//! [`VirtualPool::suspend`] copies every chunk's contents to page-locked
//! host storage and releases the physical storage while keeping the
//! reservation; [`VirtualPool::restore`] maps new physical storage at the
//! same addresses and copies the contents back. Nothing may read or write
//! the pool between the two.
//!
//! A pool made for an [`crate::Allocator`] belongs to one of the allocator's
//! pools and books every new mapping against the device's budget before the
//! storage is created ([`crate::Budget`]). The blocks it maps are
//! [`crate::allocator::Block`]s, which the allocator's buffers export as POSIX
//! descriptors.

use std::sync::Arc;

use crate::allocator::{Block, Budget, PoolKind};
use crate::driver::{
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CU_MEM_LOCATION_TYPE_DEVICE, MemAccessDesc,
    MemAllocationProp, MemLocation, in_context, reserve_addresses,
};
use crate::vmm::{HandleType, Protection};
use crate::{DeviceId, Error, PinnedBuffer, PrimaryContext, Result, Stream, reclaim};

const CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: i32 = 1;

/// One chunk of the pool: `bytes` at `offset` of the reservation.
struct Chunk {
    offset: u64,
    bytes: usize,
    /// The physical allocation mapped there; `None` while suspended.
    block: Option<Arc<Block>>,
    /// The chunk's contents while suspended.
    saved: Option<PinnedBuffer>,
    /// What the chunk's device and peers may do with it now. A block is given
    /// out read-write; a holder may park it read-only with
    /// [`VirtualPool::protect`], and the pool puts it back before handing it to
    /// the next holder.
    protection: Protection,
    free: bool,
}

/// The pool's kind and the budget every new mapping is booked against before
/// its storage is created.
struct Booking {
    kind: PoolKind,
    budget: Arc<dyn Budget>,
}

/// Device storage at stable addresses; see the module documentation.
pub struct VirtualPool {
    context: PrimaryContext,
    device: DeviceId,
    /// The ordinals granted access to the chunks: the device's, then each
    /// peer's that can access its storage.
    access: Vec<i32>,
    address: u64,
    reserved: usize,
    granularity: usize,
    /// The shareable handle type of every chunk ([`crate::HandleType::probe`]):
    /// NCCL registers a captured collective's buffers with its peers by
    /// exporting them, a fabric handle to a peer on another host.
    handle_type: HandleType,
    /// The bytes of the reservation given out so far.
    used: usize,
    chunks: Vec<Chunk>,
    suspended: bool,
    /// Set for an allocator's pools, whose mappings are booked.
    booking: Option<Booking>,
}

impl VirtualPool {
    /// Reserves `bytes` of virtual addresses on `device`, rounded up to the
    /// allocation granularity; no physical storage is mapped yet.
    ///
    /// # Errors
    ///
    /// Fails when the driver cannot reserve the range.
    pub fn reserve(device: DeviceId, bytes: usize) -> Result<Self> {
        let handle_type = HandleType::probe(device)?;
        Self::with_handle_type(device, bytes, handle_type, None)
    }

    /// Reserves `bytes` of virtual addresses on `device` for one of an
    /// allocator's pools: every block it maps uses `handle_type` handles and is
    /// booked against `budget` as `kind` before its storage is created.
    pub(crate) fn with_budget(
        device: DeviceId,
        bytes: usize,
        handle_type: HandleType,
        kind: PoolKind,
        budget: Arc<dyn Budget>,
    ) -> Result<Self> {
        Self::with_handle_type(device, bytes, handle_type, Some(Booking { kind, budget }))
    }

    fn with_handle_type(
        device: DeviceId,
        bytes: usize,
        handle_type: HandleType,
        booking: Option<Booking>,
    ) -> Result<Self> {
        let context = PrimaryContext::retain(device)?;
        let ordinal = device.handle()?;
        let mut access = vec![ordinal];
        for peer in (0..DeviceId::count()?).map(DeviceId) {
            if peer != device && peer.can_access_peer(device)? {
                access.push(peer.handle()?);
            }
        }
        let (address, reserved, granularity) = in_context(context.raw(), |driver| {
            let properties = MemAllocationProp::device(ordinal, handle_type.raw());
            let mut granularity = 0;
            driver.check(
                // SAFETY: the driver reads the local properties and writes one
                // size_t to the local `granularity`.
                unsafe {
                    (driver.mem_get_allocation_granularity)(
                        &mut granularity,
                        &properties,
                        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                    )
                },
                "cuMemGetAllocationGranularity",
            )?;
            let granularity = granularity.max(1);
            let reserved = bytes.max(1).next_multiple_of(granularity);
            let address = reserve_addresses(driver, reserved, granularity)?;
            Ok((address, reserved, granularity))
        })?;
        Ok(Self {
            context,
            device,
            access,
            address,
            reserved,
            granularity,
            handle_type,
            used: 0,
            chunks: Vec::new(),
            suspended: false,
            booking,
        })
    }

    /// Whether the pool is parked: its blocks reserve addresses but none is
    /// mapped, and every one holds a saved host copy of its contents.
    ///
    /// An allocator's rollback uses this to park only the pools its own restore
    /// brought back, never one that was live all along.
    pub fn is_suspended(&self) -> bool {
        self.suspended
    }

    /// The bytes of virtual addresses the pool reserves: what it was asked for,
    /// rounded up to the device's allocation granularity.
    pub fn reserved_bytes(&self) -> usize {
        self.reserved
    }

    /// The bytes of physical storage the pool maps now: every block whose
    /// storage is mapped, and so exactly what its budget has booked for it.
    pub fn mapped_bytes(&self) -> usize {
        self.chunks
            .iter()
            .filter(|chunk| chunk.block.as_ref().is_some_and(|block| block.is_mapped()))
            .map(|chunk| chunk.bytes)
            .sum()
    }

    /// Books the pool's current total, after its storage changed. The storage
    /// exists or is gone already, so a refusal cannot undo the change; the
    /// caller reports the failure of the operation that made it.
    fn rebook(&self) {
        if let Some(booking) = &self.booking {
            let _ = booking
                .budget
                .book(self.device, booking.kind, self.mapped_bytes() as u64);
        }
    }

    /// [`Self::allocate`], also answering with the block it gave out, so that a
    /// buffer can export it.
    pub(crate) fn allocate_block(&mut self, bytes: usize) -> Result<(u64, Arc<Block>)> {
        if self.suspended {
            return Err(Error::invalid("a suspended pool gives out no storage"));
        }
        let bytes = bytes.max(1).next_multiple_of(self.granularity);
        if let Some(index) = self
            .chunks
            .iter()
            .position(|chunk| chunk.free && chunk.bytes == bytes)
        {
            // A block is a block from the pool's point of view: the next holder
            // gets it read-write whatever the last one asked the device for.
            // A read-only block handed on would fault the new holder's first
            // write, and a device fault is a sticky error that kills the
            // process rather than failing the call.
            if self.chunks[index].protection != Protection::ReadWrite {
                let offset = self.chunks[index].offset;
                self.set_access(offset, bytes, Protection::ReadWrite)?;
                self.chunks[index].protection = Protection::ReadWrite;
            }
            self.chunks[index].free = false;
            let chunk = &self.chunks[index];
            let block = chunk
                .block
                .clone()
                .ok_or_else(|| Error::invalid("a suspended chunk cannot be given out"))?;
            return Ok((self.address + chunk.offset, block));
        }
        if self.reserved - self.used < bytes {
            return Err(Error::invalid(format!(
                "the pool's reservation of {} bytes on {} cannot give out {bytes} more",
                self.reserved, self.device
            )));
        }
        let offset = self.used as u64;
        let handle = self.map_handle(offset, bytes)?;
        self.used += bytes;
        let block = Arc::new(Block::new(
            handle,
            bytes,
            self.handle_type,
            self.context.clone(),
        ));
        self.chunks.push(Chunk {
            offset,
            bytes,
            block: Some(Arc::clone(&block)),
            saved: None,
            protection: Protection::ReadWrite,
            free: false,
        });
        Ok((self.address + offset, block))
    }

    /// Returns the chunk at `address` to the pool, which keeps it mapped
    /// and gives it out again. An address the pool did not give out is
    /// ignored.
    ///
    /// A freed chunk's storage stays mapped on purpose: a block's physical
    /// memory goes back to the driver only when the pool is suspended
    /// ([`Self::suspend`], which keeps every address and every byte's contents
    /// for [`Self::restore`]) or dropped. There is no partial shrink — a caller
    /// that must give back part of a range while keeping its addresses needs a
    /// new operation, with its own contract and its own test, not this one
    /// quietly unmapping a tail.
    pub fn free(&mut self, address: u64) {
        if let Some(chunk) = self
            .chunks
            .iter_mut()
            .find(|chunk| self.address + chunk.offset == address)
        {
            chunk.free = true;
        }
    }

    /// Changes what the pool's device and its peers may do with the block at
    /// `address`, at its unchanged address (`cuMemSetAccess`): a parked
    /// standby's blocks become read-only and become read-write again when it
    /// is promoted, and a captured graph that addresses them stays valid.
    pub(crate) fn protect(&mut self, address: u64, protection: Protection) -> Result<()> {
        let Some(chunk) = self
            .chunks
            .iter()
            .find(|chunk| self.address + chunk.offset == address)
        else {
            return Err(Error::invalid(format!(
                "the pool on {} gave out no block at {address:#x}",
                self.device
            )));
        };
        if !chunk.block.as_ref().is_some_and(|block| block.is_mapped()) {
            return Err(Error::invalid(
                "a suspended pool's block has no storage to change the access of",
            ));
        }
        let (offset, bytes) = (chunk.offset, chunk.bytes);
        self.set_access(offset, bytes, protection)?;
        if let Some(chunk) = self
            .chunks
            .iter_mut()
            .find(|chunk| self.address + chunk.offset == address)
        {
            chunk.protection = protection;
        }
        Ok(())
    }

    /// What the pool's device and its peers may do with the chunk at `offset`
    /// (`cuMemSetAccess`), at its unchanged address.
    fn set_access(&self, offset: u64, bytes: usize, protection: Protection) -> Result<()> {
        let address = self.address + offset;
        let access: Vec<MemAccessDesc> = self
            .access
            .iter()
            .map(|&id| MemAccessDesc {
                location: MemLocation {
                    kind: CU_MEM_LOCATION_TYPE_DEVICE,
                    id,
                },
                flags: protection.flags(),
            })
            .collect();
        in_context(self.context.raw(), |driver| {
            // SAFETY: the range is mapped by this pool; the descriptors are live
            // for the call.
            driver.check(
                unsafe { (driver.mem_set_access)(address, bytes, access.as_ptr(), access.len()) },
                "cuMemSetAccess",
            )
        })
    }

    /// Creates `bytes` of physical storage and maps it at `offset`, readable
    /// and writable by the pool's device and its peers, booking it against the
    /// pool's budget first when the pool has one. Returns the block.
    fn map_handle(&self, offset: u64, bytes: usize) -> Result<u64> {
        if let Some(booking) = &self.booking {
            let total = self.mapped_bytes() as u64 + bytes as u64;
            // The booking comes first: a refusal must leave nothing mapped.
            booking.budget.book(self.device, booking.kind, total)?;
        }
        let ordinal = self.device.handle()?;
        let address = self.address + offset;
        let mapped = in_context(self.context.raw(), |driver| {
            let properties = MemAllocationProp::device(ordinal, self.handle_type.raw());
            let mut handle = 0;
            driver.check(
                // SAFETY: the driver reads the local properties and writes one
                // handle.
                unsafe { (driver.mem_create)(&mut handle, bytes, &properties, 0) },
                "cuMemCreate",
            )?;
            // SAFETY: the target lies inside the reservation and is unmapped.
            let mapped = driver.check(
                unsafe { (driver.mem_map)(address, bytes, 0, handle, 0) },
                "cuMemMap",
            );
            if let Err(error) = mapped {
                // SAFETY: the handle created above, never mapped.
                let _ = unsafe { (driver.mem_release)(handle) };
                return Err(error);
            }
            let access: Vec<MemAccessDesc> = self
                .access
                .iter()
                .map(|&id| MemAccessDesc {
                    location: MemLocation {
                        kind: CU_MEM_LOCATION_TYPE_DEVICE,
                        id,
                    },
                    flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                })
                .collect();
            // SAFETY: the range was mapped above; the descriptors are live.
            let granted = driver.check(
                unsafe { (driver.mem_set_access)(address, bytes, access.as_ptr(), access.len()) },
                "cuMemSetAccess",
            );
            if let Err(error) = granted {
                // SAFETY: unmaps what was mapped above.
                let _ = unsafe { (driver.mem_unmap)(address, bytes) };
                // SAFETY: releases the handle created above, now unmapped.
                let _ = unsafe { (driver.mem_release)(handle) };
                return Err(error);
            }
            Ok(handle)
        });

        match mapped {
            Ok(handle) => Ok(handle),
            Err(error) => {
                // Nothing stayed mapped, so the pool's total is what it was
                // before the booking; restoring it must not mask the failure.
                if let Some(booking) = &self.booking {
                    let _ =
                        booking
                            .budget
                            .book(self.device, booking.kind, self.mapped_bytes() as u64);
                }
                Err(error)
            }
        }
    }

    /// Copies every chunk's contents to page-locked host storage and releases
    /// its physical storage, keeping the reservation and every address. The
    /// caller has drained all work that reads or writes the pool and starts
    /// none until [`Self::restore`]. Suspending a suspended pool does
    /// nothing.
    ///
    /// A caller that exported a chunk with `Buffer::export` drains whoever
    /// imported it first: the importer holds its own reference to the storage
    /// this call releases, so its mapping outlives the suspension and keeps the
    /// contents the copy took — while [`Self::restore`] maps new storage at the
    /// same address, which the importer's mapping is not.
    ///
    /// # Errors
    ///
    /// Fails when a copy or an unmap fails; the chunks suspended before the
    /// failure stay suspended and are restored by [`Self::restore`].
    pub fn suspend(&mut self) -> Result<()> {
        if self.suspended {
            return Ok(());
        }
        let stream = Stream::new(self.device)?;
        for index in 0..self.chunks.len() {
            let Some(block) = self.chunks[index].block.clone() else {
                continue;
            };
            let bytes = self.chunks[index].bytes;
            let address = self.address + self.chunks[index].offset;
            let saved = PinnedBuffer::new(self.device, bytes)?;
            in_context(self.context.raw(), |driver| {
                // SAFETY: the chunk is mapped and idle; the host buffer is
                // live and as long as the chunk.
                driver.check(
                    unsafe {
                        (driver.memcpy_async)(saved.address(), address, bytes, stream.handle())
                    },
                    "cuMemcpyAsync",
                )
            })?;
            stream.synchronize()?;
            in_context(self.context.raw(), |driver| {
                driver.check(
                    // SAFETY: the chunk was mapped by this pool and nothing
                    // addresses it while the pool is suspended.
                    unsafe { (driver.mem_unmap)(address, bytes) },
                    "cuMemUnmap",
                )?;
                // SAFETY: the handle is this pool's reference.
                driver.check(unsafe { (driver.mem_release)(block.raw()) }, "cuMemRelease")
            })?;
            // The block stays with the chunk: a buffer handed out before the
            // suspension still holds it, and its address and contents are what
            // the restore brings back. Only its handle goes.
            if let Some(block) = &self.chunks[index].block {
                block.unmap();
            }
            self.chunks[index].saved = Some(saved);
            // The device maps this storage no longer, so its budget must stop
            // counting it: an over-count would refuse an allocation that fits.
            self.rebook();
        }
        self.suspended = true;
        Ok(())
    }

    /// Maps new physical storage at every chunk's address and copies its saved
    /// contents back. Restoring a pool that is not suspended does nothing.
    ///
    /// # Errors
    ///
    /// Fails when the driver cannot create or map a chunk, when a booking
    /// refuses the chunk's new total, or when a copy fails — and then maps
    /// nothing: the chunks this call had already mapped are released again
    /// before the error is returned, so the pool ends exactly as suspended as
    /// it began and every chunk keeps its saved contents.
    pub fn restore(&mut self) -> Result<()> {
        if !self.suspended {
            return Ok(());
        }
        let stream = Stream::new(self.device)?;
        let mut mapped: Vec<usize> = Vec::new();
        let mut failure: Option<Error> = None;
        for index in 0..self.chunks.len() {
            if self.chunks[index]
                .block
                .as_ref()
                .is_some_and(|block| block.is_mapped())
            {
                continue;
            }
            let (offset, bytes) = (self.chunks[index].offset, self.chunks[index].bytes);
            let handle = match self.map_handle(offset, bytes) {
                Ok(handle) => handle,
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            };
            // The chunk keeps the block it had, so every buffer handed out
            // before the suspension names the storage this restore created.
            self.chunks[index].protection = Protection::ReadWrite;
            match self.chunks[index].block.clone() {
                Some(block) => block.remap(handle),
                None => {
                    self.chunks[index].block = Some(Arc::new(Block::new(
                        handle,
                        bytes,
                        self.handle_type,
                        self.context.clone(),
                    )));
                    self.chunks[index].protection = Protection::ReadWrite;
                }
            }
            mapped.push(index);
            let Some(saved) = self.chunks[index].saved.as_ref() else {
                continue;
            };
            let address = self.address + offset;
            let copied = in_context(self.context.raw(), |driver| {
                // SAFETY: the chunk was mapped above; the host buffer holds its
                // saved contents and is as long as the chunk.
                driver.check(
                    unsafe {
                        (driver.memcpy_async)(address, saved.address(), bytes, stream.handle())
                    },
                    "cuMemcpyAsync",
                )
            })
            .and_then(|()| stream.synchronize());
            if let Err(error) = copied {
                failure = Some(error);
                break;
            }
        }
        match failure {
            None => {
                self.suspended = false;
                // The contents are back on the device; the host copies are no
                // longer needed.
                for chunk in &mut self.chunks {
                    chunk.saved = None;
                }
                Ok(())
            }
            Some(error) => {
                // A half-restored pool would book part of its storage, which a
                // standby's accounting must not have to reason about.
                for index in mapped {
                    let _ = self.release_chunk(index);
                }
                Err(error)
            }
        }
    }

    /// Releases the chunk `index` this pool mapped, keeping its saved contents
    /// and the booked total true to what the pool maps.
    /// Releases the storage of the chunk at `index` and keeps the block: a
    /// buffer handed out before may still hold it, and a pool that was rolled
    /// back out of a restore may be restored again into the same block.
    fn release_chunk(&mut self, index: usize) -> Result<()> {
        let Some(block) = self.chunks[index].block.clone() else {
            return Ok(());
        };
        if !block.is_mapped() {
            return Ok(());
        }
        let address = self.address + self.chunks[index].offset;
        let bytes = self.chunks[index].bytes;
        in_context(self.context.raw(), |driver| {
            driver.check(
                // SAFETY: the chunk was mapped by this pool and nothing
                // addresses it while the pool is suspended.
                unsafe { (driver.mem_unmap)(address, bytes) },
                "cuMemUnmap",
            )?;
            driver.check(
                // SAFETY: the handle is this pool's reference.
                unsafe { (driver.mem_release)(block.raw()) },
                "cuMemRelease",
            )
        })?;
        block.unmap();
        self.rebook();
        Ok(())
    }
}

impl Drop for VirtualPool {
    fn drop(&mut self) {
        let context = self.context.clone();
        let device = self.device;
        let release = self
            .booking
            .as_ref()
            .map(|booking| (booking.kind, Arc::clone(&booking.budget)));
        let (address, reserved) = (self.address, self.reserved);
        let mapped: Vec<(u64, usize, u64)> = self
            .chunks
            .iter()
            .filter_map(|chunk| {
                chunk.block.as_ref().and_then(|block| {
                    block
                        .is_mapped()
                        .then(|| (address + chunk.offset, chunk.bytes, block.raw()))
                })
            })
            .collect();
        // The owner destroyed every graph that addresses the pool and
        // drained the streams they ran on.
        reclaim::defer(move || {
            let outcome = in_context(context.raw(), |driver| {
                let mut outcome = Ok(());
                for (chunk, bytes, handle) in mapped {
                    // SAFETY: each chunk was mapped by this pool.
                    let status = unsafe { (driver.mem_unmap)(chunk, bytes) };
                    outcome = outcome.and(driver.check(status, "cuMemUnmap"));
                    // SAFETY: each handle is this pool's reference.
                    let status = unsafe { (driver.mem_release)(handle) };
                    outcome = outcome.and(driver.check(status, "cuMemRelease"));
                }
                // SAFETY: the range was reserved by this pool.
                let status = unsafe { (driver.mem_address_free)(address, reserved) };
                outcome.and(driver.check(status, "cuMemAddressFree"))
            });
            // The pool's storage is gone, so the device's budget stops counting
            // it. Releasing the last of a deployment's pools may outlive its
            // ledger client, hence best effort.
            if let Some((kind, budget)) = release {
                let _ = budget.book(device, kind, 0);
            }
            outcome
        });
    }
}
