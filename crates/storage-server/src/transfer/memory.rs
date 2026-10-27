//! The memory the Transfer Server copies between, mapped into its process.
//!
//! The server gives each allocation once, with its descriptor
//! ([`Event::Allocation`](storage::protocol::Event::Allocation)), and the
//! Transfer Server maps it at the first copy that needs it, then keeps the
//! mapping until the server takes the allocation back and no copy holds it:
//!
//! | Allocation | Mapping | For |
//! | --- | --- | --- |
//! | device storage | its CUDA VMM allocation imported on its GPU, every GPU that can address it granted access | copy engines |
//! | host storage, page pool | the memfd mapped into the process | the CPU's copies |
//! | the same | that mapping registered with CUDA (portable, so every context's copy engines reach it; for reads alone where the device supports it) | copy engines |
//! | another host's device allocation | its fabric handle imported on the GPU that runs the copy | copy engines |
//! | device or host allocation | the mapping above registered with this process's NIXL agent | READs from another host into it, and another host's READs of it |
//!
//! A mapping that fails stays failed for the allocation: the tasks that
//! copy it fail, and the server's next allocation gives it afresh.
//! Read-only content (a storage of read-only tensors) is mapped read-only,
//! a guard beside the server's check that no copy writes it; another host
//! only READs what this host's agent registered.

use std::ffi::c_void;
use std::io;
use std::os::fd::OwnedFd;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use cuda::{DeviceAttribute, DeviceId, HostRegistration, PeerMapping, Protection, ShareableHandle};
use storage::protocol::Memory;

use super::devices::Gpus;

/// A mapping made once, or why it could not be made.
type Made<T> = OnceLock<Result<T, String>>;

/// The key of the next NIXL registration of the process (`nixl::Region::key`):
/// keys are never reused while the process, and with it its agent, lives.
static NIXL_KEYS: AtomicU64 = AtomicU64::new(1);

/// One allocation the server gave.
pub(super) struct Mapped {
    pub(super) memory: Memory,
    pub(super) bytes: u64,
    writable: bool,
    /// For a device allocation: its GPU, as this process numbers it, and
    /// every GPU that can address it; `None` when this process does not
    /// see the GPU.
    device: Option<(DeviceId, Vec<DeviceId>)>,
    /// The allocation's registration with the process's NIXL agent, made
    /// at its first READ or exposure. It is undone before the mapping it
    /// registers is unmapped: fields drop in declaration order.
    nixl: Made<nixl::Registration>,
    /// The device allocation's mapping.
    mapping: Made<PeerMapping>,
    // A registration is undone before its range is unmapped: fields drop
    // in declaration order.
    registration: Made<HostRegistration>,
    host: Made<HostMap>,
    /// The allocation's POSIX handle or memfd. Declared last: every
    /// mapping above holds its own reference, so the descriptor only
    /// needs to outlive their creation.
    descriptor: OwnedFd,
}

impl Mapped {
    pub(super) fn new(
        memory: Memory,
        bytes: u64,
        writable: bool,
        descriptor: OwnedFd,
        gpus: &Gpus,
    ) -> Self {
        let device = match &memory {
            Memory::Device { gpu } => gpus
                .by_uuid(*gpu)
                .map(|found| (found.device, found.readers.clone())),
            Memory::Host | Memory::Remote { .. } => None,
        };
        Self {
            memory,
            bytes,
            writable,
            device,
            nixl: OnceLock::new(),
            mapping: OnceLock::new(),
            registration: OnceLock::new(),
            host: OnceLock::new(),
            descriptor,
        }
    }

    /// The device this device allocation lies on; an error names a GPU this
    /// process does not see.
    pub(super) fn device(&self) -> Result<DeviceId, String> {
        match (&self.memory, &self.device) {
            (_, Some((device, _))) => Ok(*device),
            (Memory::Device { gpu }, None) => Err(format!(
                "GPU {} is not visible to the Transfer Server",
                cuda::DeviceUuid(*gpu)
            )),
            (Memory::Host, None) => Err("host memory lies on no GPU".into()),
            (Memory::Remote { .. }, None) => {
                Err("another host's memory lies on no GPU here".into())
            }
        }
    }

    /// Whether `reader`'s copy engines can address this device allocation.
    pub(super) fn readable_by(&self, reader: DeviceId) -> bool {
        self.device
            .as_ref()
            .is_some_and(|(_, readers)| readers.contains(&reader))
    }

    /// The device address of byte 0 of a device allocation.
    pub(super) fn device_address(&self) -> Result<u64, String> {
        let mapping = self.mapping.get_or_init(|| {
            let device = self.device()?;
            let readers = self
                .device
                .as_ref()
                .map(|(_, readers)| readers.clone())
                .unwrap_or_default();
            let protection = if self.writable {
                Protection::ReadWrite
            } else {
                Protection::ReadOnly
            };
            PeerMapping::import_retained(
                device,
                &self.descriptor,
                self.length()?,
                &readers,
                protection,
            )
            .map_err(|error| format!("mapping a device allocation: {error}"))
        });
        mapping
            .as_ref()
            .map(PeerMapping::address)
            .map_err(Clone::clone)
    }

    /// The address of byte 0 of a host allocation in this process.
    pub(super) fn host_address(&self) -> Result<u64, String> {
        let host = self.host.get_or_init(|| {
            HostMap::new(&self.descriptor, self.length()?, self.writable)
                .map_err(|error| format!("mapping host memory: {error}"))
        });
        host.as_ref().map(HostMap::address).map_err(Clone::clone)
    }

    /// The address of byte 0 of a host allocation, registered so that copy
    /// engines reach it; the first registration is made in `device`'s
    /// context, and is portable to every other.
    ///
    /// A read-only allocation registers for reads alone where the device
    /// supports it. A device that does not (GB200 reports
    /// `ReadOnlyHostRegisterSupported` 0 and refuses the flag) registers its
    /// read-only mapping as a plain registration. Either way the Transfer
    /// Server only reads a read-only allocation: the server refuses it as a
    /// destination.
    pub(super) fn registered_address(&self, device: DeviceId) -> Result<u64, String> {
        let address = self.host_address()?;
        let length = self.length()?;
        let registration = self.registration.get_or_init(|| {
            let pointer = address as *mut u8;
            let reads_alone = !self.writable
                && device
                    .attribute(DeviceAttribute::ReadOnlyHostRegisterSupported)
                    .is_ok_and(|supported| supported != 0);
            // SAFETY: `host` maps the whole range for as long as this
            // allocation lives, and `registration` drops before it.
            let registered = unsafe {
                if reads_alone {
                    HostRegistration::read_only(device, pointer, length)
                } else {
                    HostRegistration::new(device, pointer, length)
                }
            };
            registered.map_err(|error| format!("registering host memory: {error}"))
        });
        registration.as_ref().map(|_| address).map_err(Clone::clone)
    }

    /// The whole allocation as NIXL names it, registered with `agent` at the
    /// first call: the range this agent's READs write into, or another
    /// host's READs read. The process has one agent, which every call names.
    ///
    /// A device allocation registers its device mapping, host memory its
    /// host mapping, each under a key no other registration of the process
    /// takes: an allocation the server gives later may be mapped at the
    /// address of one it took back, whose registration a peer still knows.
    /// A read-only mapping registers as one: another host only READs it.
    pub(super) fn nixl_region(&self, agent: &nixl::Agent) -> Result<nixl::Region, String> {
        let registration = self.nixl.get_or_init(|| {
            let len = self.length()?;
            let key = NIXL_KEYS.fetch_add(1, Ordering::Relaxed);
            let region = match self.memory {
                Memory::Device { .. } => nixl::Region {
                    memory: nixl::Memory::Device,
                    address: self.device_address()?,
                    len,
                    key,
                },
                Memory::Host => nixl::Region {
                    memory: nixl::Memory::Host,
                    address: self.host_address()?,
                    len,
                    key,
                },
                Memory::Remote { .. } => {
                    return Err("the server gives allocations of this host only".into());
                }
            };
            // SAFETY: the mapping the region lies in is a field declared
            // after `nixl`, so it stays mapped until the registration
            // dropped; the Transfer Server holds no Rust reference to the
            // bytes, which its copies reach through raw addresses only.
            unsafe { agent.register(region) }
                .map_err(|error| format!("registering memory with NIXL: {error}"))
        });
        registration
            .as_ref()
            .map(nixl::Registration::region)
            .map_err(Clone::clone)
    }

    fn length(&self) -> Result<usize, String> {
        usize::try_from(self.bytes).map_err(|_| "the allocation exceeds the address space".into())
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
            tracing::warn!(%error, "unmapping host memory failed");
        }
    }
}
