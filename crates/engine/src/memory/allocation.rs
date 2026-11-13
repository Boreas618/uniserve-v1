//! One allocation the worker owns or maps.
//!
//! A [`Storage`] is one allocation: device storage this process created, or a mapping of storage
//! the host's Storage Server retains (a weight's allocation, a cache plane, a rank's part of a
//! host region). Dropping the last reference releases the allocation
//! or the mapping without waiting for device work, so an owner retires every device access to a
//! storage before it drops it.
//!
//! A forward reads and writes a storage through a tensor: [`Storage::external`] gives the tensor
//! over a window of a device allocation, which execution lends for the duration of a call
//! (`ops::external`). The tensor does not keep the allocation alive; its owner does.

use std::os::fd::OwnedFd;
use std::sync::{Arc, Mutex};

use platform::{
    DeviceAllocation, DeviceId, HostRegistration, PeerMapping, Protection, SegmentedMapping,
};
use tensor::{DType, Shape, Tensor};

use crate::worker::{Error, Result};

use service::lock;

/// One allocation owned by the worker: device storage, or a mapping of storage another process
/// created.
pub struct Storage {
    kind: Kind,
}

/// What a [`Storage`] holds.
enum Kind {
    /// Device storage this process created and holds.
    Device(DeviceAllocation),
    /// A device allocation the Storage Server retains, mapped into this process through its
    /// descriptor. Behind a lock because an in-place weight update replaces the physical
    /// allocation under the mapping's address ([`Storage::rebind`]).
    Mapped(Mutex<PeerMapping>),
    /// Host storage the Storage Server keeps: a memfd mapped into this process and registered
    /// for device copies.
    HostMapped(HostMapping),
    /// Device storage in segments mapped back to back in one reserved address range, each a
    /// physical allocation of its own that the Storage Server retains by its descriptor,
    /// released and mapped again at its place: a KV cache plane that gives part of its storage
    /// back while every address stays valid.
    Segmented {
        device: DeviceId,
        /// The first byte of the reserved range.
        address: u64,
        /// The length of the reserved range, mapped or not.
        bytes: usize,
        mapping: Mutex<SegmentedMapping>,
    },
}

/// A shared mapping of a host memfd, registered with CUDA so device copies read and write it
/// directly.
struct HostMapping {
    address: usize,
    bytes: usize,
    /// The registration of the mapped range, ended before the range is unmapped.
    registration: Option<HostRegistration>,
}

impl HostMapping {
    /// Maps `bytes` of `descriptor` from its start, writable or read-only, and registers the
    /// mapping in `device`'s context.
    fn map(descriptor: &OwnedFd, bytes: usize, device: DeviceId, writable: bool) -> Result<Self> {
        use std::os::fd::AsRawFd;

        let protection = if writable {
            libc::PROT_READ | libc::PROT_WRITE
        } else {
            libc::PROT_READ
        };

        // SAFETY: a new shared mapping of a descriptor this call borrows; the mapping keeps the
        // storage alive after the descriptor closes.
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
                "mapping host storage of the storage server",
                std::io::Error::last_os_error(),
            ));
        }

        // The mapping exists from here on, so a refused registration unmaps it on the way out.
        let mut mapping = Self {
            address: address as usize,
            bytes,
            registration: None,
        };
        let pointer = mapping.address as *mut u8;

        // SAFETY: the range is this mapping, which outlives the registration: `Drop` ends the
        // registration before it unmaps the range.
        let registration = unsafe {
            if writable {
                HostRegistration::new(device, pointer, bytes)
            } else {
                HostRegistration::read_only(device, pointer, bytes)
            }
        }?;
        mapping.registration = Some(registration);

        Ok(mapping)
    }
}

impl Drop for HostMapping {
    fn drop(&mut self) {
        drop(self.registration.take());

        // SAFETY: the last owner of this live mapping, whose registration ended above.
        unsafe { libc::munmap(self.address as *mut _, self.bytes) };
    }
}

impl Storage {
    /// Linear device storage on `device`. Contents are uninitialized.
    pub(crate) fn linear(device: DeviceId, bytes: usize) -> Result<Arc<Self>> {
        let allocation = DeviceAllocation::linear(device, bytes)?;

        Ok(Arc::new(Self {
            kind: Kind::Device(allocation),
        }))
    }

    /// Device storage that other processes and devices can map in place, with read-write access
    /// for `device` and every device in `access`. The length is rounded up to the allocation
    /// granularity. Contents are uninitialized.
    pub(crate) fn exportable(
        device: DeviceId,
        bytes: usize,
        access: &[DeviceId],
    ) -> Result<Arc<Self>> {
        let allocation = DeviceAllocation::exportable(device, bytes, access)?;

        Ok(Arc::new(Self {
            kind: Kind::Device(allocation),
        }))
    }

    /// Device storage this process creates and holds, of the kind a Storage Server can retain
    /// through a POSIX descriptor: a table a model writes after its load, which lives as long as
    /// this process. The length is rounded up to the allocation granularity. Contents are
    /// uninitialized.
    pub(crate) fn retainable(device: DeviceId, bytes: usize) -> Result<Arc<Self>> {
        let allocation = DeviceAllocation::retainable(device, bytes, &[])?;

        Ok(Arc::new(Self {
            kind: Kind::Device(allocation),
        }))
    }

    /// Every other visible device that can address `device`'s storage: the devices an exportable
    /// allocation grants access to, so readers on those devices in this process copy from it.
    pub(crate) fn peers(device: DeviceId) -> Result<Vec<DeviceId>> {
        let mut peers = Vec::new();

        for peer in (0..DeviceId::count()?).map(DeviceId) {
            if peer != device && peer.can_access_peer(device)? {
                peers.push(peer);
            }
        }
        Ok(peers)
    }

    /// Device storage of segments of `lengths` bytes back to back, each a multiple of the
    /// allocation granularity, none mapped yet ([`Storage::create_segment`],
    /// [`Storage::import_segment`]), addressed by `device` and every device in `access` once
    /// mapped. Another process maps each segment by its own descriptor, never the storage as one
    /// allocation.
    pub(crate) fn segmented(
        device: DeviceId,
        lengths: &[usize],
        access: &[DeviceId],
    ) -> Result<Arc<Self>> {
        let mapping = SegmentedMapping::reserve(device, lengths, access)?;
        let address = mapping.address();
        let bytes = mapping.len();

        Ok(Arc::new(Self {
            kind: Kind::Segmented {
                device,
                address,
                bytes,
                mapping: Mutex::new(mapping),
            },
        }))
    }

    /// The mapping of a segmented storage.
    fn segments(&self) -> Result<&Mutex<SegmentedMapping>> {
        match &self.kind {
            Kind::Segmented { mapping, .. } => Ok(mapping),
            _ => Err(Error::Invariant("the storage is not segmented".into())),
        }
    }

    /// Creates the storage of segment `index` and returns its descriptor for the Storage Server
    /// to retain. Contents are uninitialized.
    pub(crate) fn create_segment(&self, index: usize) -> Result<OwnedFd> {
        Ok(lock(self.segments()?).create(index)?)
    }

    /// Maps segment `index` from the descriptor of storage another process created and the
    /// Storage Server retains.
    pub(crate) fn import_segment(&self, index: usize, descriptor: &OwnedFd) -> Result<()> {
        Ok(lock(self.segments()?).import(index, descriptor)?)
    }

    /// Unmaps segment `index` at once, as its storage is given back. The caller has retired
    /// every device access to it.
    pub(crate) fn release_segment(&self, index: usize) -> Result<()> {
        Ok(lock(self.segments()?).release(index)?)
    }

    /// Whether segment `index` is mapped: one that is not was given back, or was never created.
    pub(crate) fn segment_mapped(&self, index: usize) -> Result<bool> {
        Ok(lock(self.segments()?).is_mapped(index))
    }

    /// Maps a **device** allocation the Storage Server retains, `bytes` long, through its
    /// `descriptor`, with `protection` for `device` and every device in `access`.
    ///
    /// The descriptor is one `cuMemExportToShareableHandle` produced from a `cuMemCreate`
    /// allocation, which is what the record of a device location carries. The storage of a host
    /// location is a memfd, which [`Storage::map_host`] maps and registers; the record's
    /// location decides between the two.
    pub(crate) fn map_device(
        descriptor: &OwnedFd,
        bytes: usize,
        device: DeviceId,
        access: &[DeviceId],
        protection: Protection,
    ) -> Result<Arc<Self>> {
        let mapping = PeerMapping::import_retained(device, descriptor, bytes, access, protection)?;

        Ok(Arc::new(Self {
            kind: Kind::Mapped(Mutex::new(mapping)),
        }))
    }

    /// New device storage for the Storage Server to retain, of at least `bytes`, mapped
    /// read-write for `device` and every device in `access`, and the descriptor that publishes
    /// it. Contents are uninitialized.
    ///
    /// The creating allocation is released at once and the storage is used through a mapping of
    /// its descriptor, as a later worker maps it, so that retained storage is uniformly held
    /// through descriptors, which NVML attributes to no process.
    pub(crate) fn retained(
        device: DeviceId,
        bytes: usize,
        access: &[DeviceId],
    ) -> Result<(Arc<Self>, OwnedFd)> {
        let created = DeviceAllocation::retainable(device, bytes, access)?;
        let descriptor = created.export_descriptor()?.ok_or_else(|| {
            Error::Invariant("a retainable allocation exports a descriptor".into())
        })?;

        let storage = Self::map_device(
            &descriptor,
            created.len(),
            device,
            access,
            Protection::ReadWrite,
        )?;
        drop(created);

        Ok((storage, descriptor))
    }

    /// Maps host storage the Storage Server keeps, `bytes` long, through its `descriptor`,
    /// writable or read-only, registered in `device`'s context: a rank's part of a host region,
    /// which the rank's copies write and read.
    pub(crate) fn map_host(
        descriptor: OwnedFd,
        bytes: usize,
        device: DeviceId,
        writable: bool,
    ) -> Result<Arc<Self>> {
        let mapping = HostMapping::map(&descriptor, bytes, device, writable)?;

        Ok(Arc::new(Self {
            kind: Kind::HostMapped(mapping),
        }))
    }

    /// Puts the physical allocation `source` maps at this storage's address, read-only: a
    /// retained weight takes another retained allocation's bytes while every pointer to it, a
    /// captured graph's and a weight tensor's among them, stays what it is. `source` keeps its
    /// own mapping until its owner drops it.
    ///
    /// The caller has retired every device read of this storage. A failure after the driver
    /// unmapped the allocation this storage held leaves the address unusable, and the caller
    /// retires the rank.
    ///
    /// # Errors
    ///
    /// [`Error::State`] unless both storages are mappings of retained device allocations, and
    /// two different ones; the driver's refusal.
    pub(crate) fn rebind(&self, source: &Self) -> Result<()> {
        let (Kind::Mapped(target), Kind::Mapped(replacement)) = (&self.kind, &source.kind) else {
            return Err(Error::State(
                "only a retained device mapping takes another retained allocation",
            ));
        };
        if std::ptr::eq(self, source) {
            return Err(Error::State(
                "a retained mapping does not replace its own allocation",
            ));
        }

        lock(target).remap_from(&lock(replacement), Protection::ReadOnly)?;
        Ok(())
    }

    /// The device the allocation is on; `None` for host storage.
    pub(crate) fn device(&self) -> Option<DeviceId> {
        match &self.kind {
            Kind::Device(allocation) => Some(allocation.device()),
            Kind::Mapped(mapping) => Some(lock(mapping).device()),
            Kind::Segmented { device, .. } => Some(*device),
            Kind::HostMapped(_) => None,
        }
    }

    /// The address of the first byte: a device pointer for device storage, a host pointer
    /// otherwise.
    pub(crate) fn address(&self) -> u64 {
        match &self.kind {
            Kind::Device(allocation) => allocation.address(),
            Kind::Mapped(mapping) => lock(mapping).address(),
            Kind::Segmented { address, .. } => *address,
            Kind::HostMapped(mapping) => mapping.address as u64,
        }
    }

    /// The length of the allocation in bytes.
    pub(crate) fn bytes(&self) -> usize {
        match &self.kind {
            Kind::Device(allocation) => allocation.len(),
            Kind::Mapped(mapping) => lock(mapping).len(),
            Kind::Segmented { bytes, .. } => *bytes,
            Kind::HostMapped(mapping) => mapping.bytes,
        }
    }

    /// The tensor of `shape` elements of `dtype` whose first element lies `offset` bytes into
    /// this allocation: memory execution owns and lends to a forward (`ops::external`). A
    /// strided window is a view of the tensor this returns.
    ///
    /// # Errors
    ///
    /// [`Error::Invariant`] when the elements reach past the allocation; [`Error::Unsupported`]
    /// for host storage, which no kernel addresses, and for an element narrower than a byte,
    /// whose plane is stated as the bytes that hold it.
    ///
    /// # Safety
    ///
    /// The caller keeps this storage alive, its window mapped, and the window's bytes unwritten
    /// by anything but the kernels launched on the tensor, until every tensor over the window
    /// is dropped and the stream work issued on them, including the replays of a graph captured
    /// over them, has completed.
    pub(crate) unsafe fn external(
        &self,
        offset: usize,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let element = ops::abi::bytes(dtype).ok_or_else(|| {
            Error::Unsupported(format!(
                "{} has no whole-byte element: its plane is stated as the bytes that hold it",
                ops::abi::name(dtype)
            ))
        })?;

        let end = (shape.elem_count() as u64)
            .checked_mul(element)
            .and_then(|bytes| bytes.checked_add(offset as u64));
        if end.is_none_or(|end| end > self.bytes() as u64) {
            return Err(Error::Invariant(format!(
                "{:?} elements of {} at byte {offset} reach past an allocation of {} bytes",
                shape.dims(),
                ops::abi::name(dtype),
                self.bytes()
            )));
        }

        let Some(device) = self.device() else {
            return Err(Error::Unsupported(
                "host storage is no tensor of a device: a kernel addresses device storage".into(),
            ));
        };

        // SAFETY: the window lies inside this allocation, as checked above, and the caller
        // keeps it valid and unwritten for as long as the tensor's work may read it.
        let tensor = unsafe {
            ops::external(
                self.address() + offset as u64,
                shape,
                dtype,
                &ops::cuda(device.0 as usize),
            )
        }?;
        Ok(tensor)
    }
}
