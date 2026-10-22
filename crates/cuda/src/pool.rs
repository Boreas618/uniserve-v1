//! A pool of device memory at stable addresses: a reserved virtual address
//! range whose physical memory is mapped as it is used, and which can be
//! released and mapped back at the same addresses.
//!
//! A CUDA graph holds the addresses of the memory its nodes read and write,
//! so the memory behind a graph must stay at its addresses for the graph's
//! life. A [`VirtualPool`] gives out chunks of one reservation, each its own
//! physical allocation mapped at an address that never changes. Freed
//! chunks stay mapped and are given out again by the same pool, so the
//! physical memory returns to the driver only when the pool is dropped.
//!
//! Every device that can access the pool's device's memory may read and
//! write the chunks: a module placed on another device (a generation tower)
//! copies its inputs from tensors a capture allocated in the pool.
//!
//! [`VirtualPool::suspend`] copies every chunk's contents to page-locked
//! host memory and releases the physical memory while keeping the
//! reservation; [`VirtualPool::restore`] maps new physical memory at the
//! same addresses and copies the contents back. Nothing may read or write
//! the pool between the two.

use crate::driver::{
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CU_MEM_LOCATION_TYPE_DEVICE, MemAccessDesc,
    MemAllocationProp, MemLocation, in_context,
};
use crate::{DeviceId, Error, PinnedBuffer, PrimaryContext, Result, Stream, reclaim};

/// No shareable handle: the chunks are local to this process.
const CU_MEM_HANDLE_TYPE_NONE: i32 = 0;
const CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: i32 = 1;

/// One chunk of the pool: `bytes` at `offset` of the reservation.
struct Chunk {
    offset: u64,
    bytes: usize,
    /// The physical allocation mapped there; `None` while suspended.
    handle: Option<u64>,
    /// The chunk's contents while suspended.
    saved: Option<PinnedBuffer>,
    free: bool,
}

/// Device memory at stable addresses; see the module documentation.
pub struct VirtualPool {
    context: PrimaryContext,
    device: DeviceId,
    /// The ordinals granted access to the chunks: the device's, then each
    /// peer's that can access its memory.
    access: Vec<i32>,
    address: u64,
    reserved: usize,
    granularity: usize,
    /// The bytes of the reservation given out so far.
    used: usize,
    chunks: Vec<Chunk>,
    suspended: bool,
}

impl VirtualPool {
    /// Reserves `bytes` of virtual addresses on `device`, rounded up to the
    /// allocation granularity; no physical memory is mapped yet.
    ///
    /// # Errors
    ///
    /// Fails when the driver cannot reserve the range.
    pub fn reserve(device: DeviceId, bytes: usize) -> Result<Self> {
        let context = PrimaryContext::retain(device)?;
        let ordinal = device.handle()?;
        let mut access = vec![ordinal];
        for peer in (0..DeviceId::count()?).map(DeviceId) {
            if peer != device && peer.can_access_peer(device)? {
                access.push(peer.handle()?);
            }
        }
        let (address, reserved, granularity) = in_context(context.raw(), |driver| {
            let properties = MemAllocationProp::device(ordinal, CU_MEM_HANDLE_TYPE_NONE);
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
            let mut address = 0;
            driver.check(
                // SAFETY: the driver writes the base of a new range to the
                // local `address`.
                unsafe { (driver.mem_address_reserve)(&mut address, reserved, granularity, 0, 0) },
                "cuMemAddressReserve",
            )?;
            Ok((address, reserved, granularity))
        })?;
        Ok(Self {
            context,
            device,
            access,
            address,
            reserved,
            granularity,
            used: 0,
            chunks: Vec::new(),
            suspended: false,
        })
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The bytes of physical memory the pool maps, or would map once
    /// restored.
    pub fn mapped_bytes(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.bytes).sum()
    }

    /// Whether `address` lies in the pool's reservation.
    pub fn contains(&self, address: u64) -> bool {
        address >= self.address && address - self.address < self.reserved as u64
    }

    /// A chunk of at least `bytes`, readable and writable by the pool's
    /// device and its peers, at an address it keeps for the pool's life: a
    /// freed chunk of
    /// the same rounded size, or new physical memory mapped after the
    /// chunks given out so far.
    ///
    /// # Errors
    ///
    /// Fails while the pool is suspended, when the reservation is
    /// exhausted, and when the driver cannot allocate or map the memory.
    pub fn allocate(&mut self, bytes: usize) -> Result<u64> {
        if self.suspended {
            return Err(Error::invalid("a suspended pool gives out no memory"));
        }
        let bytes = bytes.max(1).next_multiple_of(self.granularity);
        if let Some(chunk) = self
            .chunks
            .iter_mut()
            .find(|chunk| chunk.free && chunk.bytes == bytes)
        {
            chunk.free = false;
            return Ok(self.address + chunk.offset);
        }
        if self.reserved - self.used < bytes {
            return Err(Error::invalid(format!(
                "the pool's reservation of {} bytes on {} cannot give out {bytes} more",
                self.reserved, self.device
            )));
        }
        let offset = self.used as u64;
        let handle = self.map(offset, bytes)?;
        self.used += bytes;
        self.chunks.push(Chunk {
            offset,
            bytes,
            handle: Some(handle),
            saved: None,
            free: false,
        });
        Ok(self.address + offset)
    }

    /// Returns the chunk at `address` to the pool, which keeps it mapped
    /// and gives it out again. An address the pool did not give out is
    /// ignored.
    pub fn free(&mut self, address: u64) {
        if let Some(chunk) = self
            .chunks
            .iter_mut()
            .find(|chunk| self.address + chunk.offset == address)
        {
            chunk.free = true;
        }
    }

    /// Creates `bytes` of physical memory and maps it at `offset`, readable
    /// and writable by the pool's device and its peers. Returns its handle.
    fn map(&self, offset: u64, bytes: usize) -> Result<u64> {
        let ordinal = self.device.handle()?;
        let address = self.address + offset;
        in_context(self.context.raw(), |driver| {
            let properties = MemAllocationProp::device(ordinal, CU_MEM_HANDLE_TYPE_NONE);
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
        })
    }

    /// Copies every chunk's contents to page-locked host memory and releases
    /// its physical memory, keeping the reservation and every address. The
    /// caller has drained all work that reads or writes the pool and starts
    /// none until [`Self::restore`]. Suspending a suspended pool does
    /// nothing.
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
        for chunk in &mut self.chunks {
            let Some(handle) = chunk.handle else {
                continue;
            };
            let saved = PinnedBuffer::new(self.device, chunk.bytes)?;
            let address = self.address + chunk.offset;
            in_context(self.context.raw(), |driver| {
                // SAFETY: the chunk is mapped and idle; the host buffer is
                // live and as long as the chunk.
                driver.check(
                    unsafe {
                        (driver.memcpy_async)(
                            saved.address(),
                            address,
                            chunk.bytes,
                            stream.handle(),
                        )
                    },
                    "cuMemcpyAsync",
                )
            })?;
            stream.synchronize()?;
            in_context(self.context.raw(), |driver| {
                driver.check(
                    // SAFETY: the chunk was mapped by this pool and nothing
                    // addresses it while the pool is suspended.
                    unsafe { (driver.mem_unmap)(address, chunk.bytes) },
                    "cuMemUnmap",
                )?;
                // SAFETY: the handle is this pool's reference.
                driver.check(unsafe { (driver.mem_release)(handle) }, "cuMemRelease")
            })?;
            chunk.handle = None;
            chunk.saved = Some(saved);
        }
        self.suspended = true;
        Ok(())
    }

    /// Maps new physical memory at every chunk's address and copies its
    /// saved contents back. Restoring a pool that is not suspended does
    /// nothing.
    ///
    /// # Errors
    ///
    /// Fails when the driver cannot allocate or map a chunk, or a copy
    /// fails; the chunks not yet restored stay suspended.
    pub fn restore(&mut self) -> Result<()> {
        if !self.suspended {
            return Ok(());
        }
        let stream = Stream::new(self.device)?;
        for index in 0..self.chunks.len() {
            if self.chunks[index].handle.is_some() {
                continue;
            }
            let (offset, bytes) = (self.chunks[index].offset, self.chunks[index].bytes);
            let handle = self.map(offset, bytes)?;
            self.chunks[index].handle = Some(handle);
            if let Some(saved) = self.chunks[index].saved.take() {
                let address = self.address + offset;
                in_context(self.context.raw(), |driver| {
                    // SAFETY: the chunk was mapped above; the host buffer
                    // holds its saved contents.
                    driver.check(
                        unsafe {
                            (driver.memcpy_async)(address, saved.address(), bytes, stream.handle())
                        },
                        "cuMemcpyAsync",
                    )
                })?;
                stream.synchronize()?;
            }
        }
        self.suspended = false;
        Ok(())
    }
}

impl Drop for VirtualPool {
    fn drop(&mut self) {
        let context = self.context.clone();
        let (address, reserved) = (self.address, self.reserved);
        let mapped: Vec<(u64, usize, u64)> = self
            .chunks
            .iter()
            .filter_map(|chunk| {
                chunk
                    .handle
                    .map(|handle| (address + chunk.offset, chunk.bytes, handle))
            })
            .collect();
        // The owner destroyed every graph that addresses the pool and
        // drained the streams they ran on.
        reclaim::defer(move || {
            in_context(context.raw(), |driver| {
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
            })
        });
    }
}
