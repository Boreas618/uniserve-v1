//! The storage the Transfer Server copies between, mapped into its process.
//!
//! The server gives each allocation once, with its descriptor
//! ([`Event::Allocation`](::storage::protocol::Event::Allocation)), and keeps
//! it given until it takes the allocation back. What the Transfer Server
//! maps of it depends on what reaches it:
//!
//! | Allocation | Mapping | For | Lives |
//! | --- | --- | --- | --- |
//! | host storage, page pool | the memfd mapped into the process | the CPU's copies | until the server takes it back and no copy holds it |
//! | the same | that mapping registered with this process's NIXL agent | READs from another host into it, and another host's READs of it | the same |
//! | device storage | its CUDA VMM allocation imported on its GPU, every GPU that can address it granted access | copy engines | while the GPU's context is held ([`super::contexts`]) |
//! | the same | that mapping registered with the NIXL agent | READs, as above | the same |
//! | host storage, page pool | a mapping of the GPU's own, registered in its context (for reads alone where the device supports it) | the GPU's copy engines | the same |
//! | another host's device allocation | its fabric handle imported on the GPU that runs the copy | copy engines | the task |
//!
//! What lives in a GPU's context belongs to that GPU's session, so that the
//! context is released once no work needs it; a session drops what it made
//! for an allocation once the allocation is taken back and no copy holds it
//! ([`Mapped`]'s drop). A mapping that fails stays failed for the
//! allocation, or for the session: the tasks that copy it fail.
//! Read-only content (a storage of read-only tensors) is mapped read-only,
//! a guard beside the server's check that no copy writes it; another host
//! only READs what this host's agent registered.

use std::ffi::c_void;
use std::io;
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use ::storage::protocol::LinkEnd;
use cuda::{DeviceId, PeerMapping, ShareableHandle};

use super::contexts::Contexts;
use super::devices::Gpus;

/// A mapping made once, or why it could not be made.
type Made<T> = OnceLock<Result<T, String>>;

/// The key of the next NIXL registration of the process (`nixl::Region::key`):
/// keys are never reused while the process, and with it its agent, lives.
static NIXL_KEYS: AtomicU64 = AtomicU64::new(1);

/// A key for a new NIXL registration.
pub(super) fn nixl_key() -> u64 {
    NIXL_KEYS.fetch_add(1, Ordering::Relaxed)
}

/// The id of the next allocation given, by which sessions keep what they
/// made for it.
static IDS: AtomicU64 = AtomicU64::new(1);

/// One allocation the server gave.
pub(super) struct Mapped {
    id: u64,
    pub(super) end: LinkEnd,
    pub(super) bytes: u64,
    writable: bool,
    /// For a device allocation: its GPU, as this process numbers it, and
    /// every GPU that can address it; `None` when this process does not
    /// see the GPU.
    device: Option<(DeviceId, Vec<DeviceId>)>,
    /// The sessions that may hold what they made for the allocation.
    contexts: Arc<Contexts>,
    /// A host allocation's registration with the process's NIXL agent, made
    /// at its first READ or exposure. It is undone before the mapping it
    /// registers is unmapped: fields drop in declaration order.
    nixl: Made<nixl::Registration>,
    host: Made<HostMap>,
    /// The allocation's POSIX handle or memfd. Declared last: every
    /// mapping holds its own reference, so the descriptor only needs to
    /// outlive their creation.
    descriptor: OwnedFd,
}

impl Mapped {
    pub(super) fn new(
        end: LinkEnd,
        bytes: u64,
        writable: bool,
        descriptor: OwnedFd,
        gpus: &Gpus,
        contexts: Arc<Contexts>,
    ) -> Self {
        let device = match &end {
            LinkEnd::Device { gpu } => gpus
                .by_uuid(*gpu)
                .map(|found| (found.device, found.readers.clone())),
            LinkEnd::Host | LinkEnd::Remote { .. } => None,
        };
        Self {
            id: IDS.fetch_add(1, Ordering::Relaxed),
            end,
            bytes,
            writable,
            device,
            contexts,
            nixl: OnceLock::new(),
            host: OnceLock::new(),
            descriptor,
        }
    }

    pub(super) fn id(&self) -> u64 {
        self.id
    }

    pub(super) fn writable(&self) -> bool {
        self.writable
    }

    pub(super) fn descriptor(&self) -> &OwnedFd {
        &self.descriptor
    }

    /// The device this device allocation lies on; an error names a GPU this
    /// process does not see.
    pub(super) fn device(&self) -> Result<DeviceId, String> {
        match (&self.end, &self.device) {
            (_, Some((device, _))) => Ok(*device),
            (LinkEnd::Device { gpu }, None) => Err(format!(
                "GPU {} is not visible to the Transfer Server",
                cuda::DeviceUuid(*gpu)
            )),
            (LinkEnd::Host, None) => Err("host storage lies on no GPU".into()),
            (LinkEnd::Remote { .. }, None) => {
                Err("another host's storage lies on no GPU here".into())
            }
        }
    }

    /// The GPUs whose copy engines can address this device allocation,
    /// its own first.
    pub(super) fn readers(&self) -> &[DeviceId] {
        self.device
            .as_ref()
            .map_or(&[], |(_, readers)| readers.as_slice())
    }

    /// Whether `reader`'s copy engines can address this device allocation.
    pub(super) fn readable_by(&self, reader: DeviceId) -> bool {
        self.readers().contains(&reader)
    }

    /// The address of byte 0 of a host allocation in this process, for the
    /// CPU's copies and NIXL.
    pub(super) fn host_address(&self) -> Result<u64, String> {
        let host = self.host.get_or_init(|| {
            HostMap::new(&self.descriptor, self.length()?, self.writable)
                .map_err(|error| format!("mapping host storage: {error}"))
        });
        host.as_ref().map(HostMap::address).map_err(Clone::clone)
    }

    /// A host allocation as NIXL names it, registered with `agent` at the
    /// first call under a key of its own: the range this agent's READs write
    /// into, or another host's READs read. The process has one agent, which
    /// every call names. Registering host storage takes no CUDA context, so
    /// it lasts as long as the allocation; a device allocation registers in
    /// its GPU's session instead
    /// ([`Session::nixl_region`](super::contexts::Session::nixl_region)).
    pub(super) fn nixl_host_region(&self, agent: &nixl::Agent) -> Result<nixl::Region, String> {
        if self.end != LinkEnd::Host {
            return Err("a device allocation registers in its GPU's session".into());
        }
        let registration = self.nixl.get_or_init(|| {
            let region = nixl::Region {
                segment: nixl::Segment::Host,
                address: self.host_address()?,
                len: self.length()?,
                key: nixl_key(),
            };
            // A read-only allocation is mapped read-only, and registered for
            // reads alone: other hosts READ it, and nothing can write it
            // through the registration.
            let access = if self.writable {
                nixl::Access::ReadWrite
            } else {
                nixl::Access::Read
            };
            // SAFETY: the host mapping is a field declared after `nixl`, so
            // it stays mapped (writable for `ReadWrite`) until the
            // registration dropped; the Transfer Server holds no Rust
            // reference to the bytes, which its copies reach through raw
            // addresses only.
            unsafe { agent.register(region, access) }
                .map_err(|error| format!("registering storage with NIXL: {error}"))
        });
        registration
            .as_ref()
            .map(nixl::Registration::region)
            .map_err(Clone::clone)
    }

    pub(super) fn length(&self) -> Result<usize, String> {
        usize::try_from(self.bytes).map_err(|_| "the allocation exceeds the address space".into())
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // The server took the allocation back and no copy holds it: the
        // sessions drop its mappings and registrations, which hold its
        // physical storage.
        self.contexts.forget(self.id);
    }
}

/// A device allocation of another host of the NVLink fabric, imported on
/// the GPU that runs the copies of one task.
pub(super) struct Fabric {
    handle: [u8; 64],
    bytes: u64,
    device: DeviceId,
    mapping: Made<PeerMapping>,
}

impl Fabric {
    pub(super) fn new(handle: [u8; 64], bytes: u64, device: DeviceId) -> Self {
        Self {
            handle,
            bytes,
            device,
            mapping: OnceLock::new(),
        }
    }

    /// The device address of the allocation's byte 0 on the importing GPU.
    pub(super) fn address(&self) -> Result<u64, String> {
        let mapping = self.mapping.get_or_init(|| {
            let bytes = usize::try_from(self.bytes)
                .map_err(|_| "the allocation exceeds the address space".to_owned())?;
            PeerMapping::import(self.device, &ShareableHandle::Fabric(self.handle), bytes)
                .map_err(|error| format!("importing a fabric handle: {error}"))
        });
        mapping
            .as_ref()
            .map(PeerMapping::address)
            .map_err(Clone::clone)
    }
}

/// A shared mapping of a whole memfd, unmapped on drop.
pub(super) struct HostMap {
    address: *mut c_void,
    bytes: usize,
}

// SAFETY: the mapping is plain shared memory; the struct only hands out
// its address, and copies through it are ordered by the lanes' tasks.
unsafe impl Send for HostMap {}
// SAFETY: as for Send above; shared references only read the address.
unsafe impl Sync for HostMap {}

impl HostMap {
    /// Map the memfd `descriptor`, which holds at least `bytes`, for
    /// reading, and for writing when `writable`.
    ///
    /// The whole memfd is mapped: a page pool's memfd may hold huge pages,
    /// whose size it is rounded up to, and a huge-page mapping is unmapped
    /// only whole.
    pub(super) fn new(descriptor: &OwnedFd, bytes: usize, writable: bool) -> io::Result<Self> {
        use rustix::mm::{MapFlags, ProtFlags, mmap};

        let size = usize::try_from(rustix::fs::fstat(descriptor)?.st_size)
            .map_err(|_| io::Error::other("a memfd's size is negative"))?;
        if size < bytes {
            return Err(io::Error::other(format!(
                "a memfd of {size} bytes holds no allocation of {bytes} bytes"
            )));
        }
        let bytes = size;
        let protection = if writable {
            ProtFlags::READ | ProtFlags::WRITE
        } else {
            ProtFlags::READ
        };
        // SAFETY: a new shared mapping of the memfd, which this struct
        // unmaps on drop.
        let address = unsafe {
            mmap(
                std::ptr::null_mut(),
                bytes,
                protection,
                MapFlags::SHARED,
                descriptor,
                0,
            )
        }?;
        Ok(Self { address, bytes })
    }

    pub(super) fn address(&self) -> u64 {
        self.address as u64
    }
}

impl Drop for HostMap {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made; nothing copies through it once
        // its owner drops it.
        if let Err(error) = unsafe { rustix::mm::munmap(self.address, self.bytes) } {
            tracing::warn!(%error, "unmapping host storage failed");
        }
    }
}
