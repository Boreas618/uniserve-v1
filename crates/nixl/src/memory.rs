//! Byte ranges an agent registers and transfers move between.

use std::sync::Arc;

use crate::agent::Shared;
use crate::capi::{MEM_DRAM, MEM_VRAM, MemType};

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
            Self::Host => MEM_DRAM,
            Self::Device => MEM_VRAM,
        }
    }
}

/// What a registration lets this agent's READs and its peers' READs do with
/// its region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    /// Reads and writes: a READ's destination, or a source the process may
    /// also write. The mapping of the region must be writable.
    ReadWrite,
    /// Reads alone, local and remote: a source mapped read-only, which
    /// peers can READ but never write. NIXL's UCX backend registers it so
    /// with the read-only registration patch of UniServe's NIXL build
    /// (`artifacts/native/patches`); without it the registration asks for
    /// write access, which a read-only mapping of host memory refuses.
    Read,
}

impl Access {
    /// The registration descriptor's metadata that asks the UCX backend for
    /// this access: `access=read` for reads alone, none for UCX's default of
    /// reads and writes.
    pub(crate) fn metadata(self) -> &'static [u8] {
        match self {
            Self::ReadWrite => b"",
            Self::Read => b"access=read",
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

/// A region registered with an agent's backend. Dropping it deregisters
/// the region; it keeps the agent alive until then.
pub struct Registration {
    pub(crate) agent: Arc<Shared>,
    pub(crate) region: Region,
    pub(crate) access: Access,
}

impl Registration {
    pub fn region(&self) -> Region {
        self.region
    }

    pub fn access(&self) -> Access {
        self.access
    }
}

impl std::fmt::Debug for Registration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Registration")
            .field("region", &self.region)
            .field("access", &self.access)
            .finish()
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.agent.deregister(self.region);
    }
}
