//! Device allocations, pinned host buffers and host registrations.

use std::ptr::NonNull;

use crate::driver::{MemAllocationProp, driver, in_context};
use crate::vmm::VirtualRange;
use crate::{DeviceId, Error, HandleType, PrimaryContext, Result, ShareableHandle, reclaim};

const CU_MEMHOSTALLOC_PORTABLE: u32 = 1;
const CU_MEMHOSTREGISTER_PORTABLE: u32 = 1;

/// Device memory owned by the worker.
///
/// Two kinds exist:
///
/// - **linear** (`cuMemAlloc`): cheapest, not exportable;
/// - **exportable** (`cuMemCreate` + reserve + map): a physical allocation
///   created with the device's [`HandleType`], so it can be shared in place
///   with [`DeviceAllocation::export`] and aliased with
///   [`crate::PeerMapping::alias`]. Its length is rounded up to the
///   allocation granularity.
///
/// Dropping never calls CUDA on the dropping thread: the release is queued to
/// the [`crate::reclaim`]er, which makes this safe inside a DLPack deleter
/// that runs under the Python GIL. The owner guarantees that no device work
/// still addresses the memory when the last reference drops.
///
/// `Send + Sync` follow from the fields: the type holds only addresses,
/// driver handles behind thread-safe owners, and no host-side views.
pub struct DeviceAllocation {
    device: DeviceId,
    address: u64,
    bytes: usize,
    kind: Kind,
}

enum Kind {
    Linear {
        _memory: LinearMemory,
    },
    Exportable {
        handle: u64,
        handle_type: HandleType,
        _range: VirtualRange,
    },
}

/// A `cuMemAlloc` allocation; freed by the reclaimer on drop.
struct LinearMemory {
    context: PrimaryContext,
    address: u64,
}

impl Drop for LinearMemory {
    fn drop(&mut self) {
        let (context, address) = (self.context.clone(), self.address);
        reclaim::defer(move || {
            in_context(context.raw(), |driver| {
                // SAFETY: the allocation is owned and freed once.
                driver.check(unsafe { (driver.mem_free)(address) }, "cuMemFree_v2")
            })
        });
    }
}

fn require_bytes(bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Err(Error::invalid("an allocation needs a positive length"));
    }
    Ok(())
}

impl DeviceAllocation {
    /// Allocates `bytes` of linear device memory (`cuMemAlloc`, 256-byte aligned).
    pub fn linear(device: DeviceId, bytes: usize) -> Result<Self> {
        require_bytes(bytes)?;
        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;
        let driver = driver()?;

        let mut address = 0;
        // SAFETY: the driver writes the base of a new allocation.
        driver.check(
            unsafe { (driver.mem_alloc)(&mut address, bytes) },
            "cuMemAlloc_v2",
        )?;

        Ok(Self {
            device,
            address,
            bytes,
            kind: Kind::Linear {
                _memory: LinearMemory { context, address },
            },
        })
    }

    /// Allocates exportable device memory of at least `bytes`, rounded up to
    /// [`DeviceAllocation::granularity`], with read-write access for `device`
    /// and every device in `access` (other local GPUs that address it).
    pub fn exportable(device: DeviceId, bytes: usize, access: &[DeviceId]) -> Result<Self> {
        require_bytes(bytes)?;
        let handle_type = HandleType::probe(device)?;
        let granularity = handle_type.granularity(device)?;
        let rounded = bytes
            .checked_next_multiple_of(granularity)
            .ok_or_else(|| Error::invalid("the allocation size overflows"))?;

        let mut devices = vec![device];
        devices.extend(access.iter().filter(|peer| **peer != device));

        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;
        let mut range = VirtualRange::new(context.clone());
        let properties = MemAllocationProp::device(device.handle()?, handle_type.raw());
        let handle = range.create(rounded, &properties)?;
        range.reserve(rounded, granularity)?;
        range.map(0, rounded, handle)?;
        range.grant(&devices)?;

        Ok(Self {
            device,
            address: range.address(),
            bytes: rounded,
            kind: Kind::Exportable {
                handle,
                handle_type,
                _range: range,
            },
        })
    }

    /// The size and alignment unit of exportable allocations on `device`.
    pub fn granularity(device: DeviceId) -> Result<usize> {
        HandleType::probe(device)?.granularity(device)
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The device address of byte 0.
    pub fn address(&self) -> u64 {
        self.address
    }

    /// The usable length in bytes (rounded up for exportable allocations).
    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// The handle type of an exportable allocation; `None` for linear memory.
    pub fn handle_type(&self) -> Option<HandleType> {
        match &self.kind {
            Kind::Linear { .. } => None,
            Kind::Exportable { handle_type, .. } => Some(*handle_type),
        }
    }

    /// Exports the whole physical allocation for another process. `None` for
    /// linear memory, which cannot be shared in place.
    pub fn export(&self) -> Result<Option<ShareableHandle>> {
        match &self.kind {
            Kind::Linear { .. } => Ok(None),
            Kind::Exportable {
                handle,
                handle_type,
                ..
            } => ShareableHandle::export(*handle, *handle_type).map(Some),
        }
    }
}

/// Page-locked host memory (`cuMemHostAlloc`, portable to every context).
///
/// The buffer is addressable by copies on any device. Its owner drains every
/// copy that reads or writes it before reading it on the host, resizing or
/// dropping it. Dropping queues `cuMemFreeHost` to the [`crate::reclaim`]er.
pub struct PinnedBuffer {
    context: PrimaryContext,
    pointer: NonNull<u8>,
    bytes: usize,
}

// SAFETY: the buffer is plain host memory owned by this value. Shared access
// exposes only its address and unsafe views whose callers exclude DMA races.
unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    /// Allocates `bytes` of pinned host memory. `device`'s primary context
    /// performs the allocation and is retained until the buffer is freed.
    pub fn new(device: DeviceId, bytes: usize) -> Result<Self> {
        require_bytes(bytes)?;
        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;
        let driver = driver()?;

        let mut pointer = std::ptr::null_mut();
        // SAFETY: the driver writes the base of a new host allocation.
        driver.check(
            unsafe { (driver.mem_host_alloc)(&mut pointer, bytes, CU_MEMHOSTALLOC_PORTABLE) },
            "cuMemHostAlloc",
        )?;
        let pointer = NonNull::new(pointer.cast::<u8>())
            .ok_or_else(|| Error::invalid("cuMemHostAlloc returned a null pointer"))?;

        Ok(Self {
            context,
            pointer,
            bytes,
        })
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    /// The address as copy operands take it (host and device share one
    /// unified address space).
    pub fn address(&self) -> u64 {
        self.pointer.as_ptr() as u64
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Views the buffer on the host.
    ///
    /// # Safety
    /// No device copy that writes the buffer may be pending.
    pub unsafe fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation spans `bytes` bytes; the caller excludes DMA.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.bytes) }
    }

    /// Views the buffer mutably on the host.
    ///
    /// # Safety
    /// No device copy that reads or writes the buffer may be pending.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the allocation spans `bytes` bytes; the caller excludes DMA.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.bytes) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        let context = self.context.clone();
        let address = self.pointer.as_ptr() as usize;
        reclaim::defer(move || {
            in_context(context.raw(), |driver| {
                // SAFETY: the allocation is owned and freed once.
                driver.check(
                    unsafe { (driver.mem_free_host)(address as *mut _) },
                    "cuMemFreeHost",
                )
            })
        });
    }
}

/// A page-locked registration of existing host memory (`cuMemHostRegister`,
/// portable), for example a POSIX shared-memory mapping used as a DMA source
/// or destination.
///
/// Unregistration is synchronous, on the dropping thread, because the owner
/// unmaps the memory right after: a deferred unregister could run after the
/// mapping disappeared. Use [`HostRegistration::unregister`] to observe
/// failures. The owner drains all device copies that use the range first.
pub struct HostRegistration {
    context: PrimaryContext,
    pointer: NonNull<u8>,
    bytes: usize,
    registered: bool,
}

// SAFETY: the registration only stores the address to unregister; it grants
// no host access to the memory.
unsafe impl Send for HostRegistration {}
unsafe impl Sync for HostRegistration {}

impl HostRegistration {
    /// Registers `[pointer, pointer + bytes)` in `device`'s primary context.
    ///
    /// # Safety
    /// The range must be mapped and stay mapped until the registration is
    /// unregistered or dropped.
    pub unsafe fn new(device: DeviceId, pointer: *mut u8, bytes: usize) -> Result<Self> {
        require_bytes(bytes)?;
        let pointer = NonNull::new(pointer)
            .ok_or_else(|| Error::invalid("cannot register a null host range"))?;
        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;
        let driver = driver()?;

        // SAFETY: the caller guarantees the range is mapped.
        driver.check(
            unsafe {
                (driver.mem_host_register)(
                    pointer.as_ptr().cast(),
                    bytes,
                    CU_MEMHOSTREGISTER_PORTABLE,
                )
            },
            "cuMemHostRegister_v2",
        )?;

        Ok(Self {
            context,
            pointer,
            bytes,
            registered: true,
        })
    }

    pub fn address(&self) -> u64 {
        self.pointer.as_ptr() as u64
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Unregisters the range, reporting failure.
    pub fn unregister(mut self) -> Result<()> {
        self.release()
    }

    fn release(&mut self) -> Result<()> {
        if !self.registered {
            return Ok(());
        }
        self.registered = false;

        let pointer = self.pointer.as_ptr();
        in_context(self.context.raw(), |driver| {
            // SAFETY: the range was registered by this owner and is still mapped.
            driver.check(
                unsafe { (driver.mem_host_unregister)(pointer.cast()) },
                "cuMemHostUnregister",
            )
        })
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        let _ = self.release();
    }
}
