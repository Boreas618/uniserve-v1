//! Byte ranges an agent registers and transfers move between.

use nixl_sys::{MemType, MemoryRegion, NixlDescriptor, RegistrationHandle};

/// The memory a region lies in, as NIXL distinguishes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Memory {
    /// Host memory of the process (NIXL `DRAM_SEG`).
    Host,
    /// CUDA device memory (NIXL `VRAM_SEG`), named by its unified address,
    /// from which UCX finds its device.
    Device,
}

impl Memory {
    pub(crate) fn kind(self) -> MemType {
        match self {
            Self::Host => MemType::Dram,
            Self::Device => MemType::Vram,
        }
    }
}

/// `len` bytes of `memory` from `address`, a host pointer or a CUDA device
/// address in the process that registers the region, registered under
/// `key`.
///
/// NIXL files a registration under its descriptor's device identifier and
/// address range, in that order, and finds the registration a transfer's
/// descriptor lies in by the same three values: every transfer names a
/// region by the key it was registered under. The UCX backend reads no
/// device from the identifier, so it is free to be a key. Registrations
/// under distinct keys never shadow one another, even at one address: a
/// peer that loaded the metadata of a registration since deregistered, and
/// then of another at the same address, keeps the two apart, where NIXL
/// would otherwise refuse the second or find the first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Region {
    pub memory: Memory,
    pub address: u64,
    pub len: usize,
    pub key: u64,
}

/// One contiguous piece of a READ: the bytes of `source`, which lies in
/// memory a peer registered, into `destination`, which lies in memory this
/// agent registered. Both have the same, nonzero length, and each names the
/// key of the registration it lies in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub source: Region,
    pub destination: Region,
}

/// A region registered with an agent's backends. Dropping it deregisters
/// the region; the agent lives at least as long.
#[derive(Debug)]
pub struct Registration {
    region: Region,
    _handle: RegistrationHandle,
}

impl Registration {
    pub(crate) fn new(region: Region, handle: RegistrationHandle) -> Self {
        Self {
            region,
            _handle: handle,
        }
    }

    pub fn region(&self) -> Region {
        self.region
    }
}

/// A region as `nixl-sys` describes memory to register.
#[derive(Debug)]
pub(crate) struct Described(pub(crate) Region);

impl MemoryRegion for Described {
    unsafe fn as_ptr(&self) -> *const u8 {
        // A device address is a unified address: NIXL and UCX take it as a
        // pointer value and never dereference it on the host.
        self.0.address as usize as *const u8
    }

    fn size(&self) -> usize {
        self.0.len
    }
}

impl NixlDescriptor for Described {
    fn mem_type(&self) -> MemType {
        self.0.memory.kind()
    }

    fn device_id(&self) -> u64 {
        self.0.key
    }
}
