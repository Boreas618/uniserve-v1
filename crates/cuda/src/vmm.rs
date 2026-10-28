//! Virtual memory management: shareable physical allocations, their handles,
//! and mappings of other allocations into this process.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::{Arc, Mutex};

use crate::driver::{
    CU_MEM_ACCESS_FLAGS_PROT_READ, CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CU_MEM_LOCATION_TYPE_DEVICE,
    MemAccessDesc, MemAllocationProp, MemLocation, driver, in_context, reserve_addresses,
};
use crate::memory::DeviceAllocation;
use crate::{DeviceGuard, DeviceId, Error, PrimaryContext, Result, lock, reclaim};

pub(crate) const CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR: i32 = 1;
const CU_MEM_HANDLE_TYPE_FABRIC: i32 = 8;
const CU_MEM_ALLOC_GRANULARITY_MINIMUM: i32 = 0;

/// How a physical allocation is shared with other processes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandleType {
    /// A 64-byte fabric handle, importable by processes on other hosts of the
    /// same NVLink fabric (multi-node NVLink with IMEX).
    Fabric,
    /// A POSIX file descriptor, importable only by processes on this host
    /// after it is passed over a Unix socket.
    Descriptor,
}

impl HandleType {
    pub(crate) fn raw(self) -> i32 {
        match self {
            Self::Fabric => CU_MEM_HANDLE_TYPE_FABRIC,
            Self::Descriptor => CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        }
    }

    /// Whether a handle of this type can be imported on another host.
    pub fn crosses_hosts(self) -> bool {
        self == Self::Fabric
    }

    /// The handle type exportable allocations on `device` use: fabric if the
    /// driver can create a fabric-exportable allocation there, otherwise a
    /// file descriptor.
    ///
    /// The probe creates and releases one granularity-sized allocation, since
    /// device attributes alone do not establish that fabric export works (it
    /// also needs an IMEX channel). The result is cached per device.
    pub fn probe(device: DeviceId) -> Result<Self> {
        static PROBED: Mutex<BTreeMap<i32, HandleType>> = Mutex::new(BTreeMap::new());

        let mut probed = lock(&PROBED);
        if let Some(handle_type) = probed.get(&device.0) {
            return Ok(*handle_type);
        }

        let _guard = DeviceGuard::new(device)?;
        let handle_type = if fabric_usable(device)? {
            Self::Fabric
        } else {
            Self::Descriptor
        };
        probed.insert(device.0, handle_type);
        Ok(handle_type)
    }

    /// The minimum size and alignment of allocations of this type on `device`.
    pub fn granularity(self, device: DeviceId) -> Result<usize> {
        let driver = driver()?;
        let properties = MemAllocationProp::device(device.handle()?, self.raw());
        let mut granularity = 0;
        driver.check(
            // SAFETY: the driver reads the local `properties` and writes one size_t
            // to the local `granularity`.
            unsafe {
                (driver.mem_get_allocation_granularity)(
                    &mut granularity,
                    &properties,
                    CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "cuMemGetAllocationGranularity",
        )?;
        if granularity == 0 {
            return Err(Error::invalid(format!(
                "{device} reports no allocation granularity for {self:?} handles"
            )));
        }
        Ok(granularity)
    }
}

/// Attempts a fabric-exportable allocation in the current context.
fn fabric_usable(device: DeviceId) -> Result<bool> {
    let Ok(granularity) = HandleType::Fabric.granularity(device) else {
        return Ok(false);
    };
    let driver = driver()?;
    let properties = MemAllocationProp::device(device.handle()?, CU_MEM_HANDLE_TYPE_FABRIC);
    let mut handle = 0;

    // SAFETY: the driver reads the properties and writes one handle.
    let status = unsafe { (driver.mem_create)(&mut handle, granularity, &properties, 0) };
    if driver.check(status, "cuMemCreate").is_err() {
        return Ok(false);
    }
    // SAFETY: releases the probe allocation created above; it was never mapped.
    driver.check(unsafe { (driver.mem_release)(handle) }, "cuMemRelease")?;
    Ok(true)
}

/// What the devices granted access to a mapping may do with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protection {
    /// Reads only: a kernel that writes the mapping fails its context with
    /// an illegal address and leaves the bytes unchanged.
    ReadOnly,
    ReadWrite,
}

impl Protection {
    fn flags(self) -> i32 {
        match self {
            Self::ReadOnly => CU_MEM_ACCESS_FLAGS_PROT_READ,
            Self::ReadWrite => CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        }
    }
}

/// An exported allocation handle, owned by the holder.
#[derive(Debug)]
pub enum ShareableHandle {
    Fabric([u8; 64]),
    /// Closing the descriptor does not affect allocations or mappings already
    /// made from it.
    Descriptor(OwnedFd),
}

impl ShareableHandle {
    pub fn handle_type(&self) -> HandleType {
        match self {
            Self::Fabric(_) => HandleType::Fabric,
            Self::Descriptor(_) => HandleType::Descriptor,
        }
    }

    /// Exports one reference of `handle`, a `CUmemGenericAllocationHandle`.
    pub(crate) fn export(handle: u64, handle_type: HandleType) -> Result<Self> {
        let driver = driver()?;
        match handle_type {
            HandleType::Fabric => {
                let mut bytes = [0_u8; 64];
                driver.check(
                    // SAFETY: a fabric handle is a 64-byte CUmemFabricHandle, which
                    // the driver writes to the local `bytes`.
                    unsafe {
                        (driver.mem_export_to_shareable_handle)(
                            bytes.as_mut_ptr().cast(),
                            handle,
                            CU_MEM_HANDLE_TYPE_FABRIC,
                            0,
                        )
                    },
                    "cuMemExportToShareableHandle",
                )?;
                Ok(Self::Fabric(bytes))
            }
            HandleType::Descriptor => {
                let mut fd: i32 = -1;
                driver.check(
                    // SAFETY: a POSIX handle is written as one int, to the local `fd`.
                    unsafe {
                        (driver.mem_export_to_shareable_handle)(
                            (&raw mut fd).cast(),
                            handle,
                            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                            0,
                        )
                    },
                    "cuMemExportToShareableHandle",
                )?;
                // SAFETY: the driver returned a new descriptor owned by us.
                Ok(Self::Descriptor(unsafe { OwnedFd::from_raw_fd(fd) }))
            }
        }
    }

    /// Imports a new reference to the exported physical allocation.
    fn import(&self) -> Result<u64> {
        let driver = driver()?;
        let mut handle = 0;
        let status = match self {
            // The driver reads a fabric handle through the pointer.
            Self::Fabric(bytes) => unsafe {
                // SAFETY: the 64 handle bytes are live for the call.
                (driver.mem_import_from_shareable_handle)(
                    &mut handle,
                    bytes.as_ptr().cast_mut().cast(),
                    CU_MEM_HANDLE_TYPE_FABRIC,
                )
            },
            // A POSIX descriptor travels by value in the pointer argument.
            Self::Descriptor(fd) => unsafe {
                // SAFETY: the descriptor is open for the call.
                (driver.mem_import_from_shareable_handle)(
                    &mut handle,
                    fd.as_raw_fd() as isize as *mut c_void,
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                )
            },
        };
        driver.check(status, "cuMemImportFromShareableHandle")?;
        Ok(handle)
    }
}

/// A reserved virtual address range, the segments mapped into it and the
/// physical allocation references that back them.
///
/// Dropping the range (including a partially built one) queues teardown to
/// the reclaimer: unmap each segment, free the address range, release each
/// handle. Teardown issues no synchronization; see [`crate::reclaim`].
pub(crate) struct VirtualRange {
    context: PrimaryContext,
    address: u64,
    reserved: usize,
    mapped: Vec<(u64, usize)>,
    handles: Vec<u64>,
}

impl VirtualRange {
    /// An empty range. Construction methods run with `context` current.
    pub(crate) fn new(context: PrimaryContext) -> Self {
        Self {
            context,
            address: 0,
            reserved: 0,
            mapped: Vec::new(),
            handles: Vec::new(),
        }
    }

    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    /// Creates a physical allocation owned by this range and returns its handle.
    pub(crate) fn create(&mut self, bytes: usize, properties: &MemAllocationProp) -> Result<u64> {
        let driver = driver()?;
        let mut handle = 0;
        driver.check(
            // SAFETY: the driver reads the borrowed properties and writes one handle
            // to the local `handle`.
            unsafe { (driver.mem_create)(&mut handle, bytes, properties, 0) },
            "cuMemCreate",
        )?;
        self.handles.push(handle);
        Ok(handle)
    }

    /// Takes ownership of one reference to a physical allocation.
    pub(crate) fn adopt(&mut self, handle: u64) {
        self.handles.push(handle);
    }

    pub(crate) fn reserve(&mut self, bytes: usize, alignment: usize) -> Result<()> {
        let address = reserve_addresses(driver()?, bytes, alignment)?;
        self.address = address;
        self.reserved = bytes;
        Ok(())
    }

    /// Maps the whole of `handle` (`bytes` long) at `offset` into the range.
    pub(crate) fn map(&mut self, offset: usize, bytes: usize, handle: u64) -> Result<()> {
        let driver = driver()?;
        let address = self.address + offset as u64;
        driver.check(
            // SAFETY: the target lies inside the reserved range, unmapped so far.
            unsafe { (driver.mem_map)(address, bytes, 0, handle, 0) },
            "cuMemMap",
        )?;
        self.mapped.push((address, bytes));
        Ok(())
    }

    /// Grants `protection` over the whole range to each device. A grant
    /// over a mapped range changes its access in place, at an unchanged
    /// address.
    pub(crate) fn grant(&self, devices: &[DeviceId], protection: Protection) -> Result<()> {
        let descriptors = devices
            .iter()
            .map(|device| {
                Ok(MemAccessDesc {
                    location: MemLocation {
                        kind: CU_MEM_LOCATION_TYPE_DEVICE,
                        id: device.handle()?,
                    },
                    flags: protection.flags(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let driver = driver()?;
        // SAFETY: the range is fully mapped; the descriptors are live.
        driver.check(
            unsafe {
                (driver.mem_set_access)(
                    self.address,
                    self.reserved,
                    descriptors.as_ptr(),
                    descriptors.len(),
                )
            },
            "cuMemSetAccess",
        )
    }
}

impl Drop for VirtualRange {
    fn drop(&mut self) {
        let context = self.context.clone();
        let (address, reserved) = (self.address, self.reserved);
        let mapped = std::mem::take(&mut self.mapped);
        let handles = std::mem::take(&mut self.handles);
        if address == 0 && handles.is_empty() {
            return;
        }
        reclaim::defer(move || {
            in_context(context.raw(), |driver| {
                // Every step runs even after a failure so that as much as
                // possible is released; the first failure is reported.
                let mut outcome = Ok(());
                for (segment, bytes) in mapped {
                    // SAFETY: each segment was mapped by this range.
                    let status = unsafe { (driver.mem_unmap)(segment, bytes) };
                    outcome = outcome.and(driver.check(status, "cuMemUnmap"));
                }
                if address != 0 {
                    // SAFETY: the range was reserved by this owner.
                    let status = unsafe { (driver.mem_address_free)(address, reserved) };
                    outcome = outcome.and(driver.check(status, "cuMemAddressFree"));
                }
                for handle in handles {
                    // SAFETY: each handle is one reference owned by this range.
                    let status = unsafe { (driver.mem_release)(handle) };
                    outcome = outcome.and(driver.check(status, "cuMemRelease"));
                }
                outcome
            })
        });
    }
}

/// Another allocation (or several) mapped into this process on one device.
///
/// The mapping holds its own references to the physical memory, so it stays
/// valid after the exporter or the aliased allocations release theirs.
/// Dropping it queues unmap, address free and release to the reclaimer; the
/// owner guarantees that no device work still reads or writes the mapping.
pub struct PeerMapping {
    device: DeviceId,
    address: u64,
    bytes: usize,
    /// The devices granted access, which a change of protection names
    /// again.
    access: Vec<DeviceId>,
    // Declared last so that the fields above stay valid until teardown.
    range: VirtualRange,
}

impl PeerMapping {
    /// Maps one exported allocation of `bytes` (the exporter's allocation
    /// size, a multiple of the granularity) with read-write access for
    /// `device`.
    pub fn import(device: DeviceId, handle: &ShareableHandle, bytes: usize) -> Result<Self> {
        Self::peers(device, std::slice::from_ref(handle), bytes)
    }

    /// Maps each exported allocation, `bytes_each` long, back to back in
    /// `handles` order, so that segment `i` starts at `i * bytes_each`.
    ///
    /// The exporters keep their allocations importable until every peer has
    /// mapped them (a fabric export stays importable only while its source
    /// allocation exists).
    pub fn peers(device: DeviceId, handles: &[ShareableHandle], bytes_each: usize) -> Result<Self> {
        let Some(first) = handles.first() else {
            return Err(Error::invalid("a peer mapping needs at least one handle"));
        };
        let handle_type = first.handle_type();
        if handles
            .iter()
            .any(|handle| handle.handle_type() != handle_type)
        {
            return Err(Error::invalid("peer handles must share one handle type"));
        }
        let granularity = handle_type.granularity(device)?;
        if bytes_each == 0 || !bytes_each.is_multiple_of(granularity) {
            return Err(Error::invalid(format!(
                "a peer allocation of {bytes_each} bytes is not a positive multiple of \
                 the {granularity}-byte granularity"
            )));
        }
        let total = bytes_each
            .checked_mul(handles.len())
            .ok_or_else(|| Error::invalid("the peer mapping size overflows"))?;

        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;
        let mut range = VirtualRange::new(context.clone());
        let imported = handles
            .iter()
            .map(|handle| {
                let imported = handle.import()?;
                range.adopt(imported);
                Ok(imported)
            })
            .collect::<Result<Vec<_>>>()?;

        range.reserve(total, granularity)?;
        for (index, handle) in imported.into_iter().enumerate() {
            range.map(index * bytes_each, bytes_each, handle)?;
        }
        range.grant(&[device], Protection::ReadWrite)?;

        Ok(Self {
            device,
            address: range.address(),
            bytes: total,
            access: vec![device],
            range,
        })
    }

    /// Maps a physical allocation a Storage Server retains, through its
    /// POSIX descriptor, with `protection` for `device` and every device in
    /// `access`. `bytes` is the allocation's size, a multiple of the
    /// granularity.
    ///
    /// The mapping holds its own reference to the physical memory, which
    /// lives while any process maps it or the server holds its descriptor.
    /// An imported allocation exports no handle of its own: other processes
    /// on the host map the same descriptor, and a fabric handle of the
    /// allocation stopped importing when its creator released it.
    pub fn import_retained(
        device: DeviceId,
        descriptor: &OwnedFd,
        bytes: usize,
        access: &[DeviceId],
        protection: Protection,
    ) -> Result<Self> {
        let granularity = HandleType::Descriptor.granularity(device)?;
        if bytes == 0 || !bytes.is_multiple_of(granularity) {
            return Err(Error::invalid(format!(
                "a retained allocation of {bytes} bytes is not a positive multiple of \
                 the {granularity}-byte granularity"
            )));
        }
        let descriptor = descriptor
            .try_clone()
            .map_err(|error| Error::invalid(format!("duplicating a descriptor: {error}")))?;
        let handle = ShareableHandle::Descriptor(descriptor);

        let mut devices = vec![device];
        devices.extend(access.iter().filter(|peer| **peer != device));
        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;
        let mut range = VirtualRange::new(context.clone());
        let imported = handle.import()?;
        range.adopt(imported);
        range.reserve(bytes, granularity)?;
        range.map(0, bytes, imported)?;
        range.grant(&devices, protection)?;

        Ok(Self {
            device,
            address: range.address(),
            bytes,
            access: devices,
            range,
        })
    }

    /// Change the mapping's access in place, at an unchanged address, so
    /// that pointers into it, captured graphs among them, stay valid. The
    /// caller has drained every access the old protection allowed and the
    /// new one does not.
    pub fn protect(&self, protection: Protection) -> Result<()> {
        let context = PrimaryContext::retain(self.device)?;
        let _guard = context.enter()?;
        self.range.grant(&self.access, protection)
    }

    /// Replace this mapping's backing with another mapping's physical pages.
    ///
    /// Both ranges must have the same device and byte extent. The destination
    /// keeps its virtual address, including pointers captured by CUDA graphs.
    /// The caller drains all accesses before replacing it. Source mappings and
    /// their creating allocations remain owned independently.
    ///
    /// A driver failure after unmapping leaves the destination unusable; its
    /// execution owner must retire it rather than resume numerical work.
    pub fn remap_from(&mut self, source: &Self) -> Result<()> {
        if self.device != source.device || self.bytes != source.bytes {
            return Err(Error::invalid(
                "alias replacement requires one device and byte extent",
            ));
        }
        let context = PrimaryContext::retain(self.device)?;
        let _guard = context.enter()?;
        let driver = driver()?;

        // Acquire every reference before changing a live mapping. The temporary
        // range releases acquired references if the driver rejects an import.
        let mut incoming = VirtualRange::new(context);
        let mut segments = Vec::with_capacity(source.range.mapped.len());
        for &(address, bytes) in &source.range.mapped {
            let mut handle = 0;
            driver.check(
                // SAFETY: source owns a live mapping beginning at address.
                unsafe {
                    (driver.mem_retain_allocation_handle)(&mut handle, address as *mut c_void)
                },
                "cuMemRetainAllocationHandle",
            )?;
            incoming.adopt(handle);
            segments.push(((address - source.address) as usize, bytes, handle));
        }
        while let Some(&(address, bytes)) = self.range.mapped.last() {
            driver.check(
                // SAFETY: this range owns the mapping and all users drained it.
                unsafe { (driver.mem_unmap)(address, bytes) },
                "cuMemUnmap",
            )?;
            self.range.mapped.pop();
        }
        // Old references can retire asynchronously after the aliases vanished.
        std::mem::swap(&mut self.range.handles, &mut incoming.handles);
        for (offset, bytes, handle) in segments {
            self.range.map(offset, bytes, handle)?;
        }
        self.range.grant(&self.access, Protection::ReadWrite)
    }

    /// Maps whole exportable allocations of one device contiguously, without
    /// copying, so that `parts[i]` starts at the sum of the preceding lengths.
    pub fn alias(parts: &[&DeviceAllocation]) -> Result<Self> {
        let Some(first) = parts.first() else {
            return Err(Error::invalid("an alias needs at least one allocation"));
        };
        let device = first.device();
        let Some(handle_type) = first.handle_type() else {
            return Err(Error::invalid("only exportable allocations can be aliased"));
        };
        if parts
            .iter()
            .any(|part| part.device() != device || part.handle_type() != Some(handle_type))
        {
            return Err(Error::invalid(
                "aliased allocations must be exportable allocations of one device and handle type",
            ));
        }
        let total = parts
            .iter()
            .try_fold(0_usize, |total, part| total.checked_add(part.len()))
            .ok_or_else(|| Error::invalid("the alias size overflows"))?;
        let granularity = handle_type.granularity(device)?;

        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;
        let driver = driver()?;
        let mut range = VirtualRange::new(context.clone());
        let mut handles = Vec::with_capacity(parts.len());
        for part in parts {
            let mut handle = 0;
            driver.check(
                // SAFETY: the part's base address is a live VMM mapping; the
                // driver writes one new reference to its physical allocation.
                unsafe {
                    (driver.mem_retain_allocation_handle)(
                        &mut handle,
                        part.address() as *mut c_void,
                    )
                },
                "cuMemRetainAllocationHandle",
            )?;
            range.adopt(handle);
            handles.push((handle, part.len()));
        }

        range.reserve(total, granularity)?;
        let mut offset = 0;
        for (handle, bytes) in handles {
            range.map(offset, bytes, handle)?;
            offset += bytes;
        }
        range.grant(&[device], Protection::ReadWrite)?;

        Ok(Self {
            device,
            address: range.address(),
            bytes: total,
            access: vec![device],
            range,
        })
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn address(&self) -> u64 {
        self.address
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

/// A reserved virtual range whose segments are physical allocations of
/// their own, each mapped, released and mapped again at its own place:
/// memory whose parts are given back while every address stays valid, as a
/// KV cache that shrinks for a standby engine and grows back after.
///
/// Each segment a mapping creates exports a POSIX descriptor, which a
/// Storage Server retains, and another process maps the same segment from
/// it. The mapping keeps the creating handle while the segment stays
/// mapped, so that the segment's fabric handle, which only its creator
/// exports, imports on another host of the NVLink fabric
/// ([`Self::fabric_handle`]); NVML attributes the segment to this process
/// meanwhile ([`created_bytes`]). Dropping the mapping queues the teardown
/// of its mapped segments, its creating handles and the range to the
/// reclaimer.
pub struct SegmentedMapping {
    device: DeviceId,
    address: u64,
    bytes: usize,
    /// The devices granted access, `device` first.
    access: Vec<DeviceId>,
    /// The protection a segment is mapped with unless it states its own.
    protection: Protection,
    /// Each segment's offset and length, its handle while mapped, and the
    /// protection it is mapped with.
    segments: Vec<(usize, usize, Option<u64>, Protection)>,
    /// Each segment's creating handle while this mapping created it and
    /// keeps it mapped, with the fabric handle it exports when it has one.
    created: Vec<Option<Arc<PhysicalAllocation>>>,
    // Declared last so that the fields above stay valid until teardown.
    range: VirtualRange,
}

/// A creating handle independent of a virtual mapping. Keeping it alive
/// preserves the allocation's fabric exports while addresses are remapped.
pub struct PhysicalAllocation {
    device: DeviceId,
    bytes: usize,
    fabric: Option<[u8; 64]>,
    _handle: VirtualRange,
}

impl Drop for PhysicalAllocation {
    fn drop(&mut self) {
        count_created(self.device, self.bytes, false);
    }
}

/// The bytes of segments this process created and keeps the creating
/// handles of, by device ordinal: NVML attributes them to this process
/// although a Storage Server retains them.
static CREATED: Mutex<BTreeMap<i32, u64>> = Mutex::new(BTreeMap::new());

/// The bytes of segments this process created on `device` and keeps the
/// creating handles of ([`SegmentedMapping::create`]), which NVML counts
/// among the process's device memory although a Storage Server retains
/// them: what this process holds of its own is NVML's figure less these.
pub fn created_bytes(device: DeviceId) -> u64 {
    lock(&CREATED).get(&device.0).copied().unwrap_or(0)
}

/// Counts `bytes` more (or, with `held` false, fewer) created bytes on
/// `device`.
pub(crate) fn count_created(device: DeviceId, bytes: usize, held: bool) {
    let mut created = lock(&CREATED);
    let entry = created.entry(device.0).or_default();
    *entry = if held {
        entry.saturating_add(bytes as u64)
    } else {
        entry.saturating_sub(bytes as u64)
    };
}

impl SegmentedMapping {
    /// Reserves a range of segments of `lengths` bytes, back to back, each
    /// a multiple of the descriptor handles' granularity on `device`, which
    /// `device` and every device of `access` address once mapped. No
    /// segment is mapped yet.
    ///
    /// # Errors
    ///
    /// Fails when a length is not a positive multiple of the granularity or
    /// the range cannot be reserved.
    pub fn reserve(device: DeviceId, lengths: &[usize], access: &[DeviceId]) -> Result<Self> {
        let granularity = HandleType::Descriptor.granularity(device)?;
        if lengths.is_empty()
            || lengths
                .iter()
                .any(|bytes| *bytes == 0 || !bytes.is_multiple_of(granularity))
        {
            return Err(Error::invalid(format!(
                "segments must be positive multiples of the {granularity}-byte granularity"
            )));
        }
        let mut segments = Vec::with_capacity(lengths.len());
        let mut total = 0_usize;
        for bytes in lengths {
            segments.push((total, *bytes, None, Protection::ReadWrite));
            total = total
                .checked_add(*bytes)
                .ok_or_else(|| Error::invalid("the segmented range's size overflows"))?;
        }
        let mut devices = vec![device];
        devices.extend(access.iter().filter(|peer| **peer != device));
        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;
        let mut range = VirtualRange::new(context.clone());
        range.reserve(total, granularity)?;
        Ok(Self {
            device,
            address: range.address(),
            bytes: total,
            access: devices,
            protection: Protection::ReadWrite,
            created: segments.iter().map(|_| None).collect(),
            segments,
            range,
        })
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The device address of byte 0.
    pub fn address(&self) -> u64 {
        self.address
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// The segments' count.
    pub fn segments(&self) -> usize {
        self.segments.len()
    }

    /// Segment `index`'s length; zero for a segment that does not exist.
    pub fn segment_len(&self, index: usize) -> usize {
        self.segments
            .get(index)
            .map_or(0, |(_, bytes, _, _)| *bytes)
    }

    /// Whether segment `index` is mapped.
    pub fn is_mapped(&self, index: usize) -> bool {
        self.segments
            .get(index)
            .is_some_and(|(_, _, handle, _)| handle.is_some())
    }

    /// Creates the physical memory of segment `index`, maps it with the
    /// mapping's protection, and returns its POSIX descriptor for a Storage
    /// Server to retain. The segment is mapped from the descriptor, as
    /// another process maps it. The creating handle stays with the mapping
    /// until the segment is released ([`Self::release`]): where the device
    /// creates fabric-exportable memory of the segment's length, the
    /// segment exports a fabric handle ([`Self::fabric_handle`]), which
    /// imports only while its creator holds it. NVML attributes the memory
    /// to this process meanwhile, and [`created_bytes`] counts it.
    ///
    /// # Errors
    ///
    /// Fails when the segment is mapped already or the driver refuses.
    pub fn create(&mut self, index: usize) -> Result<OwnedFd> {
        let (_, bytes) = self.unmapped(index)?;
        let context = PrimaryContext::retain(self.device)?;
        let _guard = context.enter()?;
        // Fabric-exportable too where the device makes such memory and the
        // segment is a multiple of its granularity.
        let fabric = HandleType::probe(self.device)? == HandleType::Fabric
            && bytes.is_multiple_of(HandleType::Fabric.granularity(self.device)?);
        let mut handle_types = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        if fabric {
            handle_types |= CU_MEM_HANDLE_TYPE_FABRIC;
        }
        let properties = MemAllocationProp::device(self.device.handle()?, handle_types);
        let driver = driver()?;
        let mut handle = 0;
        driver.check(
            // SAFETY: the driver reads the borrowed properties and writes one
            // handle to the local `handle`.
            unsafe { (driver.mem_create)(&mut handle, bytes, &properties, 0) },
            "cuMemCreate",
        )?;
        let exported = ShareableHandle::export(handle, HandleType::Descriptor);
        let ShareableHandle::Descriptor(descriptor) = (match exported {
            Ok(exported) => exported,
            Err(error) => {
                // SAFETY: the handle is this function's own reference.
                unsafe { (driver.mem_release)(handle) };
                return Err(error);
            }
        }) else {
            // SAFETY: the handle is this function's own reference.
            unsafe { (driver.mem_release)(handle) };
            return Err(Error::invalid(
                "a descriptor export returned a fabric handle",
            ));
        };
        let fabric_handle = if fabric {
            match ShareableHandle::export(handle, HandleType::Fabric) {
                Ok(ShareableHandle::Fabric(exported)) => Some(exported),
                Ok(ShareableHandle::Descriptor(_)) => None,
                Err(error) => {
                    // SAFETY: the handle is this function's own reference.
                    unsafe { (driver.mem_release)(handle) };
                    return Err(error);
                }
            }
        } else {
            None
        };
        if let Err(error) = self.import(index, &descriptor) {
            // SAFETY: the handle is this function's own reference.
            unsafe { (driver.mem_release)(handle) };
            return Err(error);
        }
        let mut owner = VirtualRange::new(context.clone());
        owner.adopt(handle);
        self.created[index] = Some(Arc::new(PhysicalAllocation {
            device: self.device,
            bytes,
            fabric: fabric_handle,
            _handle: owner,
        }));
        count_created(self.device, bytes, true);
        Ok(descriptor)
    }

    /// The fabric handle of segment `index`, when this mapping created it,
    /// keeps it mapped and the device made it fabric-exportable: another
    /// host of the NVLink fabric imports it while this process holds it.
    pub fn fabric_handle(&self, index: usize) -> Option<[u8; 64]> {
        self.created
            .get(index)
            .and_then(|created| created.as_ref())
            .and_then(|created| created.fabric)
    }

    /// Borrow ownership of an allocation this mapping created. The owner can
    /// outlive this mapping and keeps existing fabric exports importable.
    pub fn allocation(&self, index: usize) -> Option<Arc<PhysicalAllocation>> {
        self.created.get(index).and_then(Clone::clone)
    }

    /// Maps segment `index` from the POSIX descriptor of a physical
    /// allocation of its length, as another process created it, with the
    /// mapping's protection.
    ///
    /// # Errors
    ///
    /// Fails when the segment is mapped already or the driver refuses.
    pub fn import(&mut self, index: usize, descriptor: &OwnedFd) -> Result<()> {
        self.import_protected(index, descriptor, self.protection)
    }

    /// Replace this range's backing at unchanged addresses.
    ///
    /// Each allocation is its byte extent and exported descriptor. Their
    /// total must equal this range's size; segment boundaries may differ.
    /// No device operation may address this range during the call. The caller
    /// keeps any creating handles needed by remote fabric exports alive.
    ///
    /// # Errors
    ///
    /// Fails before unmapping for incompatible allocation extents.
    /// A driver failure during remapping leaves the range unusable; the
    /// caller must terminate its execution rather than replay its graphs.
    pub fn remap(
        &mut self,
        allocations: &[(usize, &OwnedFd)],
        protection: Protection,
    ) -> Result<()> {
        let granularity = HandleType::Descriptor.granularity(self.device)?;
        let mut extent = 0_usize;
        let mut segments = Vec::with_capacity(allocations.len());
        for &(bytes, _) in allocations {
            if bytes == 0 || !bytes.is_multiple_of(granularity) {
                return Err(Error::invalid(
                    "remapped allocations must be whole device granules",
                ));
            }
            segments.push((extent, bytes, None, protection));
            extent = extent
                .checked_add(bytes)
                .ok_or_else(|| Error::invalid("remapped extent overflows"))?;
        }
        if extent != self.bytes {
            return Err(Error::invalid("remapping requires the same byte extent"));
        }
        let context = PrimaryContext::retain(self.device)?;
        let _guard = context.enter()?;
        for index in 0..self.segments.len() {
            self.release(index)?;
        }
        self.segments = segments;
        self.created = self.segments.iter().map(|_| None).collect();
        self.protection = protection;
        for (index, &(_, descriptor)) in allocations.iter().enumerate() {
            self.import_protected(index, descriptor, protection)?;
        }
        Ok(())
    }

    /// Maps segment `index` from the POSIX descriptor of a physical
    /// allocation of its length with `protection`, which the segment keeps
    /// through later changes of the mapping's protection until
    /// [`Self::protect_segment`] changes it: a segment another process
    /// writes is mapped read-only, so a fault here cannot corrupt it.
    ///
    /// # Errors
    ///
    /// Fails when the segment is mapped already or the driver refuses.
    pub fn import_protected(
        &mut self,
        index: usize,
        descriptor: &OwnedFd,
        protection: Protection,
    ) -> Result<()> {
        let (offset, bytes) = self.unmapped(index)?;
        let context = PrimaryContext::retain(self.device)?;
        let _guard = context.enter()?;
        let duplicate = descriptor
            .try_clone()
            .map_err(|error| Error::invalid(format!("duplicating a descriptor: {error}")))?;
        let handle = ShareableHandle::Descriptor(duplicate).import()?;
        self.segments[index].3 = protection;
        self.map(index, offset, bytes, handle)
    }

    /// Changes one mapped segment's access in place, at unchanged
    /// addresses.
    ///
    /// # Errors
    ///
    /// Fails when the segment is not mapped or the driver refuses.
    pub fn protect_segment(&mut self, index: usize, protection: Protection) -> Result<()> {
        let Some((offset, bytes, handle, _)) = self.segments.get(index).copied() else {
            return Err(Error::invalid(format!("segment {index} does not exist")));
        };
        if handle.is_none() {
            return Err(Error::invalid(format!("segment {index} is not mapped")));
        }
        self.segments[index].3 = protection;
        let context = PrimaryContext::retain(self.device)?;
        let _guard = context.enter()?;
        self.grant(offset, bytes, protection)
    }

    /// Unmaps segment `index` and releases this process's reference to its
    /// memory, at once: the caller has drained every device access to it.
    /// The memory is freed once no other process maps it and no Storage
    /// Server holds its descriptor. A segment that is not mapped is left
    /// as it is.
    ///
    /// # Errors
    ///
    /// Fails when the driver refuses.
    pub fn release(&mut self, index: usize) -> Result<()> {
        let Some((offset, bytes, handle, _)) = self.segments.get_mut(index) else {
            return Err(Error::invalid(format!("segment {index} does not exist")));
        };
        let Some(handle) = handle.take() else {
            return Ok(());
        };
        let address = self.address + *offset as u64;
        let bytes = *bytes;
        let created = self.created[index].take();
        let context = PrimaryContext::retain(self.device)?;
        in_context(context.raw(), |driver| {
            // SAFETY: the segment was mapped by this mapping.
            let unmapped =
                driver.check(unsafe { (driver.mem_unmap)(address, bytes) }, "cuMemUnmap");
            // SAFETY: the handle is this mapping's own reference.
            let released = driver.check(unsafe { (driver.mem_release)(handle) }, "cuMemRelease");
            drop(created);
            unmapped.and(released)
        })
    }

    /// Changes every mapped segment's access in place, at unchanged
    /// addresses; segments mapped later take the same protection.
    ///
    /// # Errors
    ///
    /// Fails when the driver refuses.
    pub fn protect(&mut self, protection: Protection) -> Result<()> {
        self.protection = protection;
        let context = PrimaryContext::retain(self.device)?;
        let _guard = context.enter()?;
        for index in 0..self.segments.len() {
            self.segments[index].3 = protection;
            let (offset, bytes, handle, _) = self.segments[index];
            if handle.is_some() {
                self.grant(offset, bytes, protection)?;
            }
        }
        Ok(())
    }

    /// Segment `index`'s offset and length, when it is not mapped.
    fn unmapped(&self, index: usize) -> Result<(usize, usize)> {
        match self.segments.get(index) {
            None => Err(Error::invalid(format!("segment {index} does not exist"))),
            Some((_, _, Some(_), _)) => Err(Error::invalid(format!("segment {index} is mapped"))),
            Some((offset, bytes, None, _)) => Ok((*offset, *bytes)),
        }
    }

    /// Maps `handle` at segment `index` and grants its access. Runs with the
    /// mapping's context current.
    fn map(&mut self, index: usize, offset: usize, bytes: usize, handle: u64) -> Result<()> {
        let driver = driver()?;
        let address = self.address + offset as u64;
        let mapped = driver.check(
            // SAFETY: the target lies inside the reserved range, unmapped.
            unsafe { (driver.mem_map)(address, bytes, 0, handle, 0) },
            "cuMemMap",
        );
        if let Err(error) = mapped {
            // SAFETY: the handle is this mapping's own reference.
            unsafe { (driver.mem_release)(handle) };
            return Err(error);
        }
        self.segments[index].2 = Some(handle);
        self.grant(offset, bytes, self.segments[index].3)
    }

    /// Grants `protection` over one mapped segment to every device of
    /// `access`. Runs with the mapping's context current.
    fn grant(&self, offset: usize, bytes: usize, protection: Protection) -> Result<()> {
        let descriptors = self
            .access
            .iter()
            .map(|device| {
                Ok(MemAccessDesc {
                    location: MemLocation {
                        kind: CU_MEM_LOCATION_TYPE_DEVICE,
                        id: device.handle()?,
                    },
                    flags: protection.flags(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let driver = driver()?;
        driver.check(
            // SAFETY: the segment is mapped; the descriptors are live.
            unsafe {
                (driver.mem_set_access)(
                    self.address + offset as u64,
                    bytes,
                    descriptors.as_ptr(),
                    descriptors.len(),
                )
            },
            "cuMemSetAccess",
        )
    }
}

impl Drop for SegmentedMapping {
    fn drop(&mut self) {
        // The range's teardown unmaps and releases what is still mapped,
        // before it frees the addresses.
        for (index, (offset, bytes, handle, _)) in self.segments.iter_mut().enumerate() {
            if let Some(handle) = handle.take() {
                self.range
                    .mapped
                    .push((self.address + *offset as u64, *bytes));
                self.range.handles.push(handle);
            }
            // An allocation retained by another owner survives this mapping.
            self.created[index].take();
        }
    }
}
