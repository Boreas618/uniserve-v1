//! Byte ranges an agent registers and transfers move between.

use nixl_sys::{MemType, MemoryRegion, NixlDescriptor, RegistrationHandle};

/// The memory a region lies in, as NIXL distinguishes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Memory {
    /// Host memory of the process (NIXL `DRAM_SEG`).
    Host,
    /// Memory of the CUDA device with this ordinal in the process
    /// (NIXL `VRAM_SEG`), which UCX addresses through the unified address.
    Device(u32),
}

impl Memory {
    pub(crate) fn kind(self) -> MemType {
        match self {
            Self::Host => MemType::Dram,
            Self::Device(_) => MemType::Vram,
        }
    }

    /// The device identifier NIXL keeps with a descriptor. NIXL matches a
    /// transfer's descriptors to registered memory by address, length and
    /// this identifier, so a region is named by the same value in its
    /// registration and in every transfer.
    pub(crate) fn device_id(self) -> u64 {
        match self {
            Self::Host => 0,
            Self::Device(ordinal) => u64::from(ordinal),
        }
    }
}

/// `len` bytes of `memory` from `address`, a host pointer or a CUDA device
/// address in the process that registers the region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Region {
    pub memory: Memory,
    pub address: u64,
    pub len: usize,
}

/// One contiguous piece of a READ: the bytes of `source`, which lies in
/// memory a peer registered, into `destination`, which lies in memory this
/// agent registered. Both have the same, nonzero length.
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
        self.0.memory.device_id()
    }
}
