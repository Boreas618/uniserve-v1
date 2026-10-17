//! Page-locked staging reused across one transport's transfers.
//!
//! A host-side copy to or from device memory goes through page-locked
//! memory. Allocating it per transfer costs a `cuMemHostAlloc` (and a later
//! free) per transfer, so a transport leases blocks from a [`Staging`]
//! cache instead. A lease is a [`Backing`] copies address directly;
//! dropping its last reference returns the block. Whoever drops it has
//! retired every device access to it, as for any [`Memory`], so a returned
//! block is free for the next lease.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use uniserve_cuda::DeviceId;

use crate::Result;
use crate::ffi::{Backing, MemoryKind};
use crate::lock;
use crate::storage::memory::Memory;

/// Free blocks by device, and their total bytes.
struct Free {
    blocks: HashMap<DeviceId, Vec<Arc<Memory>>>,
    bytes: usize,
}

/// Page-locked blocks of one transport, reused by size.
///
/// Blocks are powers of two, so a lease wastes less than half its block and
/// reuse needs no exact size match. The cache keeps at most `retained`
/// bytes of free blocks; a block returned beyond that is released.
pub(crate) struct Staging {
    free: Mutex<Free>,
    retained: usize,
}

impl Staging {
    /// A cache keeping at most `retained` bytes of free blocks: the transfer
    /// capacity of the transport's rank, which bounds the staged bytes of
    /// its concurrent reads.
    pub(crate) fn new(retained: usize) -> Arc<Self> {
        Arc::new(Self {
            free: Mutex::new(Free {
                blocks: HashMap::new(),
                bytes: 0,
            }),
            retained,
        })
    }

    /// A block of at least `bytes` page-locked bytes for copies with
    /// `device`: the smallest free one that fits, else a new one.
    pub(crate) fn lease(self: &Arc<Self>, device: DeviceId, bytes: usize) -> Result<Arc<Lease>> {
        let bytes = bytes.max(1);
        let reused = {
            let mut free = lock(&self.free);
            let taken = free.blocks.get_mut(&device).and_then(|blocks| {
                let index = blocks
                    .iter()
                    .enumerate()
                    .filter(|(_, block)| block.bytes() >= bytes)
                    .min_by_key(|(_, block)| block.bytes())
                    .map(|(index, _)| index)?;
                Some(blocks.swap_remove(index))
            });
            if let Some(block) = &taken {
                free.bytes -= block.bytes();
            }
            taken
        };
        let memory = match reused {
            Some(memory) => memory,
            None => Memory::pinned(device, bytes.next_power_of_two())?,
        };
        Ok(Arc::new(Lease {
            memory,
            device,
            owner: Arc::downgrade(self),
        }))
    }

    fn give_back(&self, device: DeviceId, memory: Arc<Memory>) {
        let mut free = lock(&self.free);
        if free.bytes + memory.bytes() > self.retained {
            // Released outside the lock: the reclaimer frees it without
            // waiting for device work.
            drop(free);
            drop(memory);
            return;
        }
        free.bytes += memory.bytes();
        free.blocks.entry(device).or_default().push(memory);
    }
}

/// One leased block. Its last reference returns it to its cache.
pub(crate) struct Lease {
    memory: Arc<Memory>,
    device: DeviceId,
    owner: Weak<Staging>,
}

impl Lease {
    /// The block's bytes.
    ///
    /// # Safety
    ///
    /// No device or other host access may write the block while the slice
    /// is alive.
    pub(crate) unsafe fn host_bytes(&self) -> &[u8] {
        // SAFETY: the caller excludes concurrent writers; a pinned block
        // always has host bytes.
        unsafe { self.memory.host_bytes() }.unwrap_or_default()
    }
}

impl Backing for Lease {
    fn device(&self) -> MemoryKind {
        self.memory.device()
    }

    fn address(&self) -> u64 {
        self.memory.address()
    }

    fn bytes(&self) -> usize {
        self.memory.bytes()
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        // The cache takes its own reference; this lease's is dropped after.
        if let Some(owner) = self.owner.upgrade() {
            owner.give_back(self.device, Arc::clone(&self.memory));
        }
    }
}
