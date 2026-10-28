//! Rust-owned allocations and the typed views Python receives as torch
//! tensors.
//!
//! Runtime storage (cache planes, graph inputs and outputs, workspaces, staging)
//! is allocated here. A [`Storage`] owns one allocation; a [`StorageView`]
//! is a typed, strided window into it that retains the allocation, the way a
//! torch tensor views a torch storage. Python receives a view as one torch tensor, made once from
//! the view's DLPack export and passed by every call that takes the view
//! ([`StorageView::handle`]). Dropping the last Rust and Python reference
//! releases the memory through `cuda`'s reclaimer, which never
//! waits for device work: owners must retire all device accesses to an
//! allocation before dropping it.
//!
//! Exportable device allocations (`cuMemCreate` with a shareable handle) and
//! mappings of memory the host's Storage Server retains are recorded in a
//! process-wide registry by address, with how long they live, so a
//! transport can export any view of them in place
//! ([`Storage::exportable_containing`]).

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use std::os::fd::OwnedFd;

use cuda::{
    DeviceAllocation, DeviceId, HostRegistration, PeerMapping, PinnedBuffer, Protection,
    SegmentedMapping, ShareableHandle, Strided,
};
use tvm_ffi::{Any, AnyView};

use crate::ffi::{
    Backing, DType, Function, Handle, Interpreter, MemoryKind, Tensor, contiguous_strides, export,
};
use crate::{Error, Result, lock};

/// One allocation owned by the worker: device memory, page-locked or
/// pageable host memory, or a mapping of memory another process exported.
pub struct Storage {
    kind: Kind,
}

enum Kind {
    Device(DeviceAllocation),
    Pinned(PinnedBuffer),
    /// Pageable host memory for computations on the host. Boxed so its address
    /// is stable.
    Host(Box<[u8]>),
    /// Another process's (or peer device's) exported allocation mapped into
    /// this process; for memory the Storage Server retains, with the
    /// descriptor that maps it, which this process exports it by.
    Mapped {
        mapping: PeerMapping,
        descriptor: Option<OwnedFd>,
    },
    /// Host memory the Storage Server retains: a memfd mapped into this
    /// process and registered for device copies.
    Retained(HostMapping),
    /// Device memory in segments mapped back to back in one reserved range,
    /// each a physical allocation of its own that the Storage Server retains
    /// by its descriptor, released and mapped again at its place: a KV cache
    /// plane that gives part of its memory back while every address stays
    /// valid.
    Segmented(Segments),
}

/// A segmented storage's mapping.
struct Segments {
    device: DeviceId,
    address: u64,
    bytes: usize,
    mapping: Mutex<SegmentedMapping>,
}

/// A shared mapping of a host memfd, registered with CUDA so device copies
/// read it directly. Read-only once published: mapped `PROT_READ` and
/// registered `CU_MEMHOSTREGISTER_READ_ONLY`.
struct HostMapping {
    address: usize,
    bytes: usize,
    device: DeviceId,
    registration: Mutex<Option<HostRegistration>>,
}

impl HostMapping {
    /// Map `bytes` of `descriptor` from its start, writable or not, and
    /// register the mapping in `device`'s context.
    fn map(descriptor: &OwnedFd, bytes: usize, device: DeviceId, writable: bool) -> Result<Self> {
        use std::os::fd::AsRawFd;

        let protection = if writable {
            libc::PROT_READ | libc::PROT_WRITE
        } else {
            libc::PROT_READ
        };
        // SAFETY: a new shared mapping of a descriptor this call borrows; the
        // mapping keeps the memory alive after the descriptor closes.
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                protection,
                libc::MAP_SHARED,
                descriptor.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(Error::transport_source(
                "mapping retained host memory",
                std::io::Error::last_os_error(),
            ));
        }
        let mut mapping = Self {
            address: address as usize,
            bytes,
            device,
            registration: Mutex::new(None),
        };
        mapping.register(writable)?;
        Ok(mapping)
    }

    fn register(&mut self, writable: bool) -> Result<()> {
        let pointer = self.address as *mut u8;
        // SAFETY: the range is this mapping, which outlives the registration
        // (`Drop` unregisters before unmapping).
        let registration = unsafe {
            if writable {
                HostRegistration::new(self.device, pointer, self.bytes)
            } else {
                HostRegistration::read_only(self.device, pointer, self.bytes)
            }
        }?;
        *lock(&self.registration) = Some(registration);
        Ok(())
    }

    /// Make the mapping read-only for the host and the devices. No copy may
    /// use the range meanwhile.
    fn seal(&self) -> Result<()> {
        let mut registration = lock(&self.registration);
        if let Some(registered) = registration.take() {
            registered.unregister()?;
        }
        // SAFETY: the range is this live mapping.
        if unsafe { libc::mprotect(self.address as *mut _, self.bytes, libc::PROT_READ) } != 0 {
            return Err(Error::transport_source(
                "protecting retained host memory",
                std::io::Error::last_os_error(),
            ));
        }
        // SAFETY: as in `register`.
        *registration = Some(unsafe {
            HostRegistration::read_only(self.device, self.address as *mut u8, self.bytes)
        }?);
        Ok(())
    }
}

impl Drop for HostMapping {
    fn drop(&mut self) {
        drop(lock(&self.registration).take());
        // SAFETY: the last owner of this live mapping, unregistered above.
        unsafe { libc::munmap(self.address as *mut _, self.bytes) };
    }
}

/// How long an exportable allocation lives, which its exports state so a
/// reader knows whether it may keep its mapping for later reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Residency {
    /// Storage the rank keeps until it closes: product arenas, pools, cache
    /// planes, latent pages.
    Resident,
    /// Storage made for one export and released when that export retires.
    Transient,
}

/// Exportable allocations by base address, with their residency. Entries
/// are weak: the registry never extends an allocation's lifetime, and a
/// dropped allocation removes its own entry.
static EXPORTABLE: Mutex<BTreeMap<u64, (Weak<Storage>, Residency)>> = Mutex::new(BTreeMap::new());

impl Storage {
    /// Device memory on `device`. Contents are uninitialized.
    pub fn device(device: DeviceId, bytes: usize) -> Result<Arc<Self>> {
        let allocation = DeviceAllocation::linear(device, bytes)?;
        Ok(Arc::new(Self {
            kind: Kind::Device(allocation),
        }))
    }

    /// Device memory that can be exported in place to other processes, with
    /// read-write access for `device` and every device in `access`, kept by
    /// the rank until it closes. The length is rounded up to the allocation
    /// granularity. Contents are uninitialized.
    pub(crate) fn exportable(
        device: DeviceId,
        bytes: usize,
        access: &[DeviceId],
    ) -> Result<Arc<Self>> {
        Self::register_exportable(device, bytes, access, Residency::Resident)
    }

    /// Exportable device memory, as [`Storage::exportable`], that holds the
    /// payload of one export and is released when that export retires.
    pub(crate) fn export_staging(
        device: DeviceId,
        bytes: usize,
        access: &[DeviceId],
    ) -> Result<Arc<Self>> {
        Self::register_exportable(device, bytes, access, Residency::Transient)
    }

    fn register_exportable(
        device: DeviceId,
        bytes: usize,
        access: &[DeviceId],
        residency: Residency,
    ) -> Result<Arc<Self>> {
        let allocation = DeviceAllocation::exportable(device, bytes, access)?;
        let address = allocation.address();
        let memory = Arc::new(Self {
            kind: Kind::Device(allocation),
        });
        lock(&EXPORTABLE).insert(address, (Arc::downgrade(&memory), residency));
        Ok(memory)
    }

    /// Every other visible device that can address `device`'s memory: the
    /// devices an exportable allocation grants access to, so in-process
    /// readers on those devices can copy from it.
    pub(crate) fn peers(device: DeviceId) -> Result<Vec<DeviceId>> {
        let mut peers = Vec::new();
        for peer in (0..DeviceId::count()?).map(DeviceId) {
            if peer != device && peer.can_access_peer(device)? {
                peers.push(peer);
            }
        }
        Ok(peers)
    }

    /// Page-locked host memory usable as a CUDA copy source or destination
    /// for `device`. Contents are uninitialized.
    pub fn pinned(device: DeviceId, bytes: usize) -> Result<Arc<Self>> {
        let buffer = PinnedBuffer::new(device, bytes)?;
        Ok(Arc::new(Self {
            kind: Kind::Pinned(buffer),
        }))
    }

    /// Zeroed pageable host memory.
    pub fn host(bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            kind: Kind::Host(vec![0; bytes].into_boxed_slice()),
        })
    }

    /// Adopt a mapping of exported device memory.
    pub(crate) fn mapped(mapping: PeerMapping) -> Arc<Self> {
        Arc::new(Self {
            kind: Kind::Mapped {
                mapping,
                descriptor: None,
            },
        })
    }

    /// Device memory of segments of `lengths` bytes back to back, each a
    /// multiple of the allocation granularity, none mapped yet
    /// ([`Storage::create_segment`], [`Storage::import_segment`]), addressed
    /// by `device` and every device in `access` once mapped. Other
    /// processes cannot map it as one allocation, so it is not exportable in
    /// place: an export of it is a copy.
    pub(crate) fn segmented(
        device: DeviceId,
        lengths: &[usize],
        access: &[DeviceId],
    ) -> Result<Arc<Self>> {
        let mapping = SegmentedMapping::reserve(device, lengths, access)?;
        let address = mapping.address();
        let bytes = mapping.len();
        Ok(Arc::new(Self {
            kind: Kind::Segmented(Segments {
                device,
                address,
                bytes,
                mapping: Mutex::new(mapping),
            }),
        }))
    }

    fn segments(&self) -> Result<&Segments> {
        match &self.kind {
            Kind::Segmented(segments) => Ok(segments),
            _ => Err(Error::Invariant("the storage is not segmented".into())),
        }
    }

    /// Creates the memory of segment `index` and returns its descriptor for
    /// the Storage Server to retain. Contents are uninitialized.
    pub(crate) fn create_segment(&self, index: usize) -> Result<OwnedFd> {
        Ok(lock(&self.segments()?.mapping).create(index)?)
    }

    /// The fabric handle of segment `index`, which this process created and
    /// keeps the creating handle of, when its device makes fabric-exportable
    /// memory: what a chunk's record states for other hosts of the NVLink
    /// fabric, which import it while this process holds the segment.
    pub(crate) fn segment_fabric(
        &self,
        index: usize,
    ) -> Result<Option<storage::protocol::FabricHandle>> {
        Ok(lock(&self.segments()?.mapping)
            .fabric_handle(index)
            .map(storage::protocol::FabricHandle))
    }

    /// Maps segment `index` from the descriptor of memory another process
    /// created and the Storage Server retains.
    pub(crate) fn import_segment(&self, index: usize, descriptor: &OwnedFd) -> Result<()> {
        Ok(lock(&self.segments()?.mapping).import(index, descriptor)?)
    }

    /// Maps segment `index` read-only from the descriptor of memory another
    /// world writes: a fault of this process cannot change its bytes.
    pub(crate) fn import_segment_read_only(
        &self,
        index: usize,
        descriptor: &OwnedFd,
    ) -> Result<()> {
        Ok(lock(&self.segments()?.mapping).import_protected(
            index,
            descriptor,
            Protection::ReadOnly,
        )?)
    }

    /// Changes segment `index`'s access in place, as a world takes the
    /// memory of a segment it mapped read-only once its writer is gone.
    pub(crate) fn protect_segment(&self, index: usize, protection: Protection) -> Result<()> {
        Ok(lock(&self.segments()?.mapping).protect_segment(index, protection)?)
    }

    /// Unmaps segment `index` at once, as its memory is given back: no
    /// device work may access it any more.
    pub(crate) fn release_segment(&self, index: usize) -> Result<()> {
        Ok(lock(&self.segments()?.mapping).release(index)?)
    }

    /// Install completed allocations at these addresses, once computations
    /// have drained. Their owner keeps creating handles for fabric exports.
    pub(crate) fn remap(&self, allocations: &[(usize, &OwnedFd)]) -> Result<()> {
        Ok(lock(&self.segments()?.mapping).remap(allocations, Protection::ReadOnly)?)
    }

    /// Keep a segment's creating handle across remapping or alias retirement.
    pub(crate) fn physical_allocation(
        &self,
        index: usize,
    ) -> Result<Option<Arc<cuda::PhysicalAllocation>>> {
        Ok(lock(&self.segments()?.mapping).allocation(index))
    }

    /// Map an allocation the host's Storage Server retains, `bytes` long,
    /// through its `descriptor`, with `protection` for `device` and every
    /// device in `access`. The mapping is registered as resident exportable
    /// memory, which this process exports in place by its descriptor.
    pub(crate) fn import(
        descriptor: OwnedFd,
        bytes: usize,
        device: DeviceId,
        access: &[DeviceId],
        protection: Protection,
    ) -> Result<Arc<Self>> {
        let mapping = PeerMapping::import_retained(device, &descriptor, bytes, access, protection)?;
        let address = mapping.address();
        let storage = Arc::new(Self {
            kind: Kind::Mapped {
                mapping,
                descriptor: Some(descriptor),
            },
        });
        lock(&EXPORTABLE).insert(address, (Arc::downgrade(&storage), Residency::Resident));
        Ok(storage)
    }

    /// New device memory for the Storage Server to retain, of at least
    /// `bytes`, mapped read-write for `device` and every device in `access`,
    /// and the descriptor that publishes it. Contents are uninitialized.
    ///
    /// The creating allocation is released at once and the memory is used
    /// through a mapping of its descriptor, as a later worker maps it, so
    /// that retained memory is uniformly held through descriptors, which
    /// NVML attributes to no process.
    pub(crate) fn retainable(
        device: DeviceId,
        bytes: usize,
        access: &[DeviceId],
    ) -> Result<(Arc<Self>, OwnedFd)> {
        let created = DeviceAllocation::retainable(device, bytes, access)?;
        let descriptor = created.export_descriptor()?.ok_or_else(|| {
            Error::Invariant("a retainable allocation exports a descriptor".into())
        })?;
        let mapped = descriptor
            .try_clone()
            .map_err(|error| Error::transport_source("duplicating a descriptor", error))?;
        let storage = Self::import(mapped, created.len(), device, access, Protection::ReadWrite)?;
        drop(created);
        Ok((storage, descriptor))
    }

    /// New host memory for the Storage Server to retain, of `bytes`, mapped
    /// writable and registered in `device`'s context, and the descriptor
    /// that publishes it. Contents are zero.
    pub(crate) fn host_retainable(device: DeviceId, bytes: usize) -> Result<(Arc<Self>, OwnedFd)> {
        let descriptor =
            rustix::fs::memfd_create("uniserve-retained", rustix::fs::MemfdFlags::CLOEXEC)
                .map_err(|error| {
                    Error::transport_source(
                        "creating retained host memory",
                        std::io::Error::from(error),
                    )
                })?;
        rustix::fs::ftruncate(&descriptor, bytes as u64).map_err(|error| {
            Error::transport_source("sizing retained host memory", std::io::Error::from(error))
        })?;
        let mapping = HostMapping::map(&descriptor, bytes, device, true)?;
        Ok((
            Arc::new(Self {
                kind: Kind::Retained(mapping),
            }),
            descriptor,
        ))
    }

    /// Map host memory the Storage Server keeps, `bytes` long, through its
    /// `descriptor`, writable or read-only, registered in `device`'s
    /// context: a page pool, which the rank's copies write and read.
    pub(crate) fn map_host(
        descriptor: OwnedFd,
        bytes: usize,
        device: DeviceId,
        writable: bool,
    ) -> Result<Arc<Self>> {
        let mapping = HostMapping::map(&descriptor, bytes, device, writable)?;
        Ok(Arc::new(Self {
            kind: Kind::Retained(mapping),
        }))
    }

    /// Map retained host memory, `bytes` long, read-only through its
    /// `descriptor`, registered in `device`'s context.
    pub(crate) fn import_host(
        descriptor: OwnedFd,
        bytes: usize,
        device: DeviceId,
    ) -> Result<Arc<Self>> {
        let mapping = HostMapping::map(&descriptor, bytes, device, false)?;
        Ok(Arc::new(Self {
            kind: Kind::Retained(mapping),
        }))
    }

    /// A handle that exports the whole allocation in place to other
    /// processes: an exportable allocation's own handle, or the descriptor
    /// of retained memory, which reaches readers on this host only (an
    /// imported allocation exports no handle of its own). `None` when the
    /// allocation cannot be exported in place to the readers `cross_hosts`
    /// says there are.
    pub(crate) fn export(&self, cross_hosts: bool) -> Result<Option<ShareableHandle>> {
        match &self.kind {
            Kind::Device(allocation) => Ok(allocation.export()?),
            Kind::Mapped {
                descriptor: Some(descriptor),
                ..
            } if !cross_hosts => descriptor
                .try_clone()
                .map(|descriptor| Some(ShareableHandle::Descriptor(descriptor)))
                .map_err(|error| Error::transport_source("duplicating a descriptor", error)),
            _ => Ok(None),
        }
    }

    /// Change the access of every device mapping the storage has, in place
    /// at an unchanged address: retained weights are downgraded to
    /// read-only once published, so any write through them faults.
    pub(crate) fn protect(&self, protection: Protection) -> Result<()> {
        match &self.kind {
            Kind::Device(allocation) => Ok(allocation.protect(protection)?),
            Kind::Mapped { mapping, .. } => Ok(mapping.protect(protection)?),
            Kind::Segmented(segments) => Ok(lock(&segments.mapping).protect(protection)?),
            Kind::Retained(mapping) if protection == Protection::ReadOnly => mapping.seal(),
            Kind::Retained(_) | Kind::Pinned(_) | Kind::Host(_) => Err(Error::Invariant(
                "only device memory changes its protection, and retained host memory \
                 only to read-only"
                    .into(),
            )),
        }
    }

    /// The device allocation, `None` for host memory and mappings.
    pub(crate) fn allocation(&self) -> Option<&DeviceAllocation> {
        match &self.kind {
            Kind::Device(allocation) => Some(allocation),
            _ => None,
        }
    }

    /// The live exportable allocation containing `address`, with the
    /// address's byte offset inside it and the allocation's residency.
    pub(crate) fn exportable_containing(address: u64) -> Option<(Arc<Self>, u64, Residency)> {
        let registry = lock(&EXPORTABLE);
        let (&base, (memory, residency)) = registry.range(..=address).next_back()?;
        let memory = memory.upgrade()?;
        (address < base + memory.bytes() as u64).then(|| (memory, address - base, *residency))
    }

    /// The allocation's host-visible bytes. `None` for device memory.
    ///
    /// # Safety
    ///
    /// No device or other host access may write the returned range while the
    /// slice is alive.
    pub unsafe fn host_bytes(&self) -> Option<&[u8]> {
        match &self.kind {
            Kind::Device(_) | Kind::Mapped { .. } | Kind::Segmented(_) => None,
            Kind::Pinned(buffer) => {
                // SAFETY: the buffer spans `len()` bytes and is owned by `self`, which
                // the slice borrows; the caller excludes concurrent writers.
                Some(unsafe { std::slice::from_raw_parts(buffer.as_ptr(), buffer.len()) })
            }
            Kind::Host(bytes) => Some(bytes),
            // SAFETY: the mapping spans `bytes` and is owned by `self`; the
            // caller excludes concurrent writers.
            Kind::Retained(mapping) => Some(unsafe {
                std::slice::from_raw_parts(mapping.address as *const u8, mapping.bytes)
            }),
        }
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        let registered = match &self.kind {
            Kind::Device(allocation) if allocation.handle_type().is_some() => {
                Some(allocation.address())
            }
            Kind::Mapped {
                mapping,
                descriptor: Some(_),
            } => Some(mapping.address()),
            _ => None,
        };
        if let Some(address) = registered {
            let mut registry = lock(&EXPORTABLE);
            // A new allocation may already reuse the address; remove only a
            // dead entry.
            if registry
                .get(&address)
                .is_some_and(|(entry, _)| entry.strong_count() == 0)
            {
                registry.remove(&address);
            }
        }
    }
}

impl Backing for Storage {
    fn device(&self) -> MemoryKind {
        match &self.kind {
            Kind::Device(allocation) => MemoryKind::Cuda(allocation.device()),
            Kind::Mapped { mapping, .. } => MemoryKind::Cuda(mapping.device()),
            Kind::Segmented(segments) => MemoryKind::Cuda(segments.device),
            Kind::Pinned(_) | Kind::Retained(_) => MemoryKind::PinnedHost,
            Kind::Host(_) => MemoryKind::Host,
        }
    }

    fn address(&self) -> u64 {
        match &self.kind {
            Kind::Device(allocation) => allocation.address(),
            Kind::Mapped { mapping, .. } => mapping.address(),
            Kind::Segmented(segments) => segments.address,
            Kind::Pinned(buffer) => buffer.address(),
            Kind::Host(bytes) => bytes.as_ptr() as u64,
            Kind::Retained(mapping) => mapping.address as u64,
        }
    }

    fn bytes(&self) -> usize {
        match &self.kind {
            Kind::Device(allocation) => allocation.len(),
            Kind::Mapped { mapping, .. } => mapping.len(),
            Kind::Segmented(segments) => segments.bytes,
            Kind::Pinned(buffer) => buffer.len(),
            Kind::Host(bytes) => bytes.len(),
            Kind::Retained(mapping) => mapping.bytes,
        }
    }

    fn copy_span(&self) -> u64 {
        match &self.kind {
            Kind::Segmented(segments) => {
                let mapping = lock(&segments.mapping);
                (0..mapping.segments())
                    .map(|index| mapping.segment_len(index) as u64)
                    .min()
                    .unwrap_or(0)
            }
            _ => u64::MAX,
        }
    }
}

/// A typed, strided window into an allocation.
///
/// `offset` is in bytes; `shape` and `strides` are in elements. Construction
/// checks that the window lies inside its allocation.
///
/// A view and its clones are one tensor to Python: they share the torch
/// tensor the view is wrapped as (`Self::handle`). A view derived from it
/// (a prefix, a row, a region, a reshape) is another tensor, wrapped on its
/// own; narrowing an axis to its whole extent returns the view itself.
#[derive(Clone)]
pub struct StorageView {
    memory: Arc<dyn Backing>,
    offset: usize,
    dtype: DType,
    shape: Vec<i64>,
    strides: Vec<i64>,
    /// The torch tensor of the view, made by the first call that passes it.
    /// Its last clone's drop releases it on the builder thread.
    tensor: Arc<OnceLock<Handle>>,
}

/// An owned copy operand: a view's address, element width, shape and
/// element strides in the form [`cuda::StridedCopy`] plans.
pub(crate) struct CopyOperand {
    address: u64,
    element_bytes: usize,
    shape: Vec<u64>,
    strides: Vec<i64>,
}

impl CopyOperand {
    pub(crate) fn strided(&self) -> Strided<'_> {
        Strided {
            address: self.address,
            element_bytes: self.element_bytes,
            shape: &self.shape,
            strides: &self.strides,
        }
    }
}

impl StorageView {
    /// A contiguous view of `shape` starting `offset` bytes into `memory`.
    pub fn contiguous(
        memory: Arc<dyn Backing>,
        offset: usize,
        dtype: DType,
        shape: &[i64],
    ) -> Result<Self> {
        Self::strided(memory, offset, dtype, shape, &contiguous_strides(shape))
    }

    pub(crate) fn strided(
        memory: Arc<dyn Backing>,
        offset: usize,
        dtype: DType,
        shape: &[i64],
        strides: &[i64],
    ) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(Error::Invariant(
                "a view needs one stride per dimension".to_owned(),
            ));
        }
        let view = Self {
            memory,
            offset,
            dtype,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            tensor: Arc::default(),
        };
        view.byte_len()?;
        Ok(view)
    }

    /// Another window of the same allocation and dtype, inside this one:
    /// a different tensor, wrapped on its own.
    fn derive(&self, offset: usize, shape: Vec<i64>, strides: Vec<i64>) -> Self {
        Self {
            memory: Arc::clone(&self.memory),
            offset,
            dtype: self.dtype,
            shape,
            strides,
            tensor: Arc::default(),
        }
    }

    /// A whole allocation viewed as `[len / dtype.bytes()]` elements.
    pub fn whole(memory: Arc<dyn Backing>, dtype: DType) -> Result<Self> {
        let elements = memory.bytes() / dtype.bytes();
        Self::contiguous(memory, 0, dtype, &[elements as i64])
    }

    /// The allocation this view retains.
    pub fn memory(&self) -> &Arc<dyn Backing> {
        &self.memory
    }

    /// Byte offset of the first element inside the allocation.
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn device(&self) -> MemoryKind {
        self.memory.device()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub(crate) fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Address of the first element.
    pub fn address(&self) -> u64 {
        self.memory.address() + self.offset as u64
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    /// Elements addressed by the view.
    pub fn numel(&self) -> u64 {
        self.shape.iter().map(|&extent| extent as u64).product()
    }

    /// Bytes of the addressed elements, excluding strided gaps.
    pub(crate) fn nbytes(&self) -> u64 {
        self.numel() * self.dtype.bytes() as u64
    }

    /// Bytes from the first to one past the last element.
    pub fn byte_len(&self) -> Result<usize> {
        if self.shape.contains(&0) {
            return Ok(0);
        }
        if self
            .shape
            .iter()
            .chain(&self.strides)
            .any(|&value| value < 0)
        {
            return Err(Error::Invariant(
                "views require non-negative extents and strides".to_owned(),
            ));
        }
        let last: i64 = self
            .shape
            .iter()
            .zip(&self.strides)
            .map(|(&extent, &stride)| (extent - 1) * stride)
            .sum();
        let bytes = (last as usize + 1) * self.dtype.bytes();
        if self.offset + bytes > self.memory.bytes() {
            return Err(Error::Invariant(format!(
                "a view of {bytes} bytes at offset {} exceeds its {}-byte allocation",
                self.offset,
                self.memory.bytes()
            )));
        }
        Ok(bytes)
    }

    /// Elements `[start, start + length)` of `axis`. The whole axis is this
    /// view itself.
    pub(crate) fn narrow(&self, axis: usize, start: i64, length: i64) -> Result<Self> {
        let extent = *self
            .shape
            .get(axis)
            .ok_or_else(|| Error::Invariant(format!("view has no axis {axis}")))?;
        if start < 0 || length < 0 || start + length > extent {
            return Err(Error::Invariant(format!(
                "range {start}..{} exceeds axis {axis} of extent {extent}",
                start + length
            )));
        }
        if start == 0 && length == extent {
            return Ok(self.clone());
        }
        let offset = self.offset + (start * self.strides[axis]) as usize * self.dtype.bytes();
        let mut shape = self.shape.clone();
        shape[axis] = length;
        Ok(self.derive(offset, shape, self.strides.clone()))
    }

    /// The view at `index` of `axis`, with that axis removed.
    pub(crate) fn select(&self, axis: usize, index: i64) -> Result<Self> {
        let row = self.narrow(axis, index, 1)?;
        let (mut shape, mut strides) = (row.shape, row.strides);
        shape.remove(axis);
        strides.remove(axis);
        Ok(self.derive(row.offset, shape, strides))
    }

    /// The per-axis `[start, end)` region of this view.
    pub(crate) fn region(&self, region: &[Range<u64>]) -> Result<Self> {
        if region.len() != self.shape.len() {
            return Err(Error::Invalid(
                "a region needs one range per view axis".to_owned(),
            ));
        }
        let mut view = self.clone();
        for (axis, range) in region.iter().enumerate() {
            if range.start > range.end {
                return Err(Error::Invalid("a region range is reversed".to_owned()));
            }
            view = view.narrow(axis, range.start as i64, (range.end - range.start) as i64)?;
        }
        Ok(view)
    }

    /// The view as a strided copy operand.
    /// The most bytes one copy may address in the view's memory
    /// ([`Backing::copy_span`]).
    pub(crate) fn copy_span(&self) -> u64 {
        self.memory.copy_span()
    }

    pub(crate) fn operand(&self) -> CopyOperand {
        CopyOperand {
            address: self.address(),
            element_bytes: self.dtype.bytes(),
            shape: self.shape.iter().map(|&extent| extent as u64).collect(),
            strides: self.strides.clone(),
        }
    }

    /// The maximal contiguous byte intervals the view addresses, in address
    /// order. Axes whose stride equals the run below them fold into the run.
    pub(crate) fn intervals(&self) -> Vec<Range<u64>> {
        if self.shape.contains(&0) {
            return Vec::new();
        }
        let element = self.dtype.bytes() as u64;
        let mut axes: Vec<(u64, u64)> = self
            .strides
            .iter()
            .zip(&self.shape)
            .filter(|(_, extent)| **extent > 1)
            .map(|(&stride, &extent)| (stride as u64, extent as u64))
            .collect();
        axes.sort_unstable();

        let mut width = 1;
        let mut outer = Vec::new();
        for (stride, extent) in axes {
            if stride == width && outer.is_empty() {
                width *= extent;
            } else {
                outer.push((stride, extent));
            }
        }

        let base = self.address();
        let mut starts = vec![0_u64];
        for (stride, extent) in outer {
            starts = starts
                .iter()
                .flat_map(|&start| (0..extent).map(move |index| start + index * stride))
                .collect();
        }
        let mut intervals: Vec<_> = starts
            .into_iter()
            .map(|start| {
                let start = base + start * element;
                start..start + width * element
            })
            .collect();
        intervals.sort_unstable_by_key(|interval| interval.start);
        intervals
    }

    /// The view as a TVM-FFI tensor: a DLPack export retaining the
    /// allocation. Python callables receive the view's torch tensor instead
    /// (`Self::handle`); a harness returns this export to its Python
    /// caller.
    pub fn to_tensor(&self) -> Result<Tensor> {
        export(
            Arc::clone(&self.memory),
            self.offset,
            self.dtype,
            &self.shape,
            &self.strides,
        )
    }

    /// The view as the torch tensor every call that takes it passes.
    ///
    /// The first call that passes the view wraps its DLPack export with
    /// `torch.from_dlpack`, called as it is; every later call through any
    /// clone of the view passes that same Python object, which costs no
    /// conversion. The tensor retains the allocation through its DLPack
    /// deleter, so the memory outlives every Python reference to it; the
    /// view releases its reference with its last clone.
    ///
    /// The first wrapping calls Python, a handoff of its own from another
    /// thread; a serving path makes it inside the [`Interpreter::execute`]
    /// of the call that passes the view, so the two take one handoff. A
    /// caller drops the references it makes for a call ([`Handle::to_any`])
    /// before the view or on the builder thread, so that the view's own
    /// reference is the last one and is released on the builder thread.
    pub(crate) fn handle(&self) -> Result<&Handle> {
        if let Some(handle) = self.tensor.get() {
            return Ok(handle);
        }
        let export = Any::from(self.to_tensor()?);
        let handle = Handle::new(from_dlpack()?.call(&[AnyView::from(&export)])?)?;
        Ok(self.tensor.get_or_init(|| handle))
    }
}

/// `torch.from_dlpack`, resolved once per process.
fn from_dlpack() -> Result<&'static Function> {
    static FROM_DLPACK: OnceLock<Function> = OnceLock::new();
    if let Some(function) = FROM_DLPACK.get() {
        return Ok(function);
    }
    let function = Interpreter::import("torch.from_dlpack")?;
    Ok(FROM_DLPACK.get_or_init(|| function))
}

impl std::fmt::Debug for StorageView {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StorageView")
            .field("device", &self.device())
            .field("offset", &self.offset)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_stay_inside_their_allocation() -> Result<()> {
        let memory: Arc<dyn Backing> = Storage::host(64);
        assert!(StorageView::contiguous(Arc::clone(&memory), 0, DType::F32, &[4, 4]).is_ok());
        assert!(StorageView::contiguous(Arc::clone(&memory), 4, DType::F32, &[4, 4]).is_err());

        let rows = StorageView::contiguous(Arc::clone(&memory), 0, DType::I64, &[8])?;
        assert_eq!(rows.narrow(0, 0, 3)?.byte_len()?, 24);
        assert!(rows.narrow(0, 0, 9).is_err());
        Ok(())
    }

    #[test]
    fn regions_select_and_transposes_address_the_same_elements() -> Result<()> {
        let memory: Arc<dyn Backing> = Storage::host(4 * 6 * 4);
        let base = memory.address();
        let grid = StorageView::contiguous(Arc::clone(&memory), 0, DType::F32, &[4, 6])?;

        let window = grid.region(&[1..3, 2..5])?;
        assert_eq!(window.shape(), &[2, 3]);
        assert_eq!(window.address(), base + (6 + 2) * 4);
        assert_eq!(
            window.intervals(),
            vec![base + 32..base + 44, base + 56..base + 68,]
        );

        let column = grid.select(1, 5)?;
        assert_eq!(
            (column.shape(), column.strides()),
            (&[4_i64][..], &[6_i64][..])
        );
        let transposed = StorageView::strided(memory, 0, DType::F32, &[6, 4], &[1, 6])?;
        assert_eq!(transposed.shape(), &[6, 4]);
        // A transposed view addresses the same bytes: one contiguous interval.
        assert_eq!(transposed.intervals(), vec![base..base + 96]);
        assert!(grid.region(&[0..5, 0..6]).is_err());
        Ok(())
    }

    #[test]
    fn dtype_names_round_trip() -> Result<()> {
        for dtype in [
            DType::Bool,
            DType::U8,
            DType::BF16,
            DType::F8E4M3,
            DType::F64,
        ] {
            assert_eq!(DType::parse(dtype.name())?, dtype);
        }
        assert!(DType::parse("complex64").is_err());
        Ok(())
    }
}
