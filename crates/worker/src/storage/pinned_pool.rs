//! Page-locked host blocks for request inputs prepared on the host.
//!
//! A [`PinnedPool`] leases blocks of page-locked memory (pageable memory for
//! a host device) and takes each back when its last user drops it. Serving
//! allocates page-locked memory only while its peak grows and never frees
//! it, since freeing page-locked memory synchronizes the device. The name is
//! TensorRT-LLM's (`BufferManager::pinnedPool`).
//!
//! A block is reused as soon as its last reference drops: a user whose
//! device work reads a block retains a reference until that work completed.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use crate::Result;
use crate::ffi::{Backing, Device};
use crate::storage::memory::Memory;

/// The smallest block, so that small inputs share block sizes.
const MIN_BLOCK: usize = 1 << 16;

/// Reusable page-locked blocks for one CUDA device's copies.
pub struct PinnedPool {
    /// The device the blocks are page-locked for; `None` for pageable host
    /// memory.
    device: Option<i32>,
    /// Blocks no user holds.
    free: Mutex<Vec<Arc<Memory>>>,
}

impl PinnedPool {
    pub fn new(device: Option<i32>) -> Arc<Self> {
        Arc::new(Self {
            device,
            free: Mutex::new(Vec::new()),
        })
    }

    fn free(&self) -> MutexGuard<'_, Vec<Arc<Memory>>> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Lease a block of at least `bytes`: the smallest free block that
    /// holds them, or a new one of the next power of two. Contents are
    /// whatever the block last held.
    pub fn lease(self: &Arc<Self>, bytes: usize) -> Result<Arc<PinnedBlock>> {
        let reused = {
            let mut free = self.free();
            free.iter()
                .enumerate()
                .filter(|(_, memory)| memory.len() >= bytes)
                .min_by_key(|(_, memory)| memory.len())
                .map(|(index, _)| index)
                .map(|index| free.swap_remove(index))
        };
        let memory = match reused {
            Some(memory) => memory,
            None => {
                let size = bytes.max(MIN_BLOCK).next_power_of_two();
                match self.device {
                    Some(device) => Memory::pinned(device, size)?,
                    None => Memory::host(size),
                }
            }
        };
        Ok(Arc::new(PinnedBlock {
            pool: Arc::downgrade(self),
            memory: Some(memory),
            bytes,
        }))
    }
}

/// A leased block: the leading `len()` bytes of a pool block, which returns
/// to its pool when the lease drops.
pub struct PinnedBlock {
    pool: Weak<PinnedPool>,
    /// Taken back by the pool on drop.
    memory: Option<Arc<Memory>>,
    bytes: usize,
}

impl Backing for PinnedBlock {
    fn device(&self) -> Device {
        self.memory
            .as_ref()
            .map_or(Device::Cpu, |memory| memory.device())
    }

    fn address(&self) -> u64 {
        self.memory.as_ref().map_or(0, |memory| memory.address())
    }

    fn len(&self) -> usize {
        self.bytes
    }
}

impl Drop for PinnedBlock {
    fn drop(&mut self) {
        if let (Some(memory), Some(pool)) = (self.memory.take(), self.pool.upgrade()) {
            pool.free().push(memory);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropped_blocks_are_reused_by_leases_they_hold() -> Result<()> {
        let pool = PinnedPool::new(None);
        let first = pool.lease(100)?;
        let address = first.address();
        assert_eq!(first.len(), 100);

        // A live block is never leased twice.
        let second = pool.lease(100)?;
        assert_ne!(second.address(), address);

        // Once dropped, its block serves a lease it holds.
        drop(first);
        assert_eq!(pool.lease(MIN_BLOCK)?.address(), address);
        Ok(())
    }

    #[test]
    fn a_lease_larger_than_every_free_block_takes_a_new_one() -> Result<()> {
        let pool = PinnedPool::new(None);
        let small = pool.lease(10)?;
        let address = small.address();
        drop(small);
        let large = pool.lease(4 * MIN_BLOCK)?;
        assert_ne!(large.address(), address);
        assert_eq!(large.len(), 4 * MIN_BLOCK);
        Ok(())
    }
}
