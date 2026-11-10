//! Scheduler-owned request slots, cache references, and Tensor spans.
//!
//! Each pool allocates its actual backing. Worker addresses are assembled from
//! these owned allocations when constructing a batch; no region copies persist.
//!
//! The backing storage itself is the worker's, which reports its capacity in
//! `MainProcessInfo`; these pools assign its rows, units, and byte ranges.
//! Row zero of the request state and unit zero of the cache pool are the
//! worker's inactive sentinels, so no pool ever hands them out, and the
//! worker IPC validation rejects a request row or cache unit of zero.
//!
//! An allocation is an owned, non-`Clone` value that the issuing pool's `free`
//! consumes; the value does not record which pool issued it, so the caller
//! returns it to the right one. `CacheAllocation` has no `free`: its units return
//! to the `BlockPool` when its tables drop.
//!
//! [`requests`] holds what a request owns of these pools from its admission
//! to its retirement, and `Storage`, the scheduler's aggregate of every pool.

use ipc::{MainProcessInfo, RequestKey};

mod bytes;
pub(crate) mod requests;
pub(crate) use bytes::ByteAllocator;

pub(crate) use crate::kv_cache::CacheAllocation;
use crate::kv_cache::{BlockPool, Checkpoint, GroupShape, PrefixCacheCoordinator};

/// Owned request row, returned only after its request epoch retires.
///
/// The row index is private so that only [`RequestPool::allocate`] can
/// create a slot.
#[derive(Debug)]
pub(crate) struct RequestSlot {
    index: u32,
}

impl RequestSlot {
    /// Returns the request-state row this slot owns.
    pub(crate) fn index(&self) -> u32 {
        self.index
    }
}

/// A byte span whose export may outlive its producing request.
#[derive(Debug)]
pub(crate) struct BufferSpan {
    pub(crate) owner: RequestKey,
    /// Byte offset, aligned as requested, into the issuing `BufferPool`.
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
}

/// Allocation failure identifying the exhausted resource class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OutOfStorage {
    #[error("request slot capacity is exhausted")]
    /// No request-state row remains available.
    RequestSlots,
    #[error("KV unit capacity is exhausted")]
    /// The paged cache unit pool cannot satisfy the requested token capacity.
    PrefixCache,
    #[error("persistent buffer capacity is exhausted")]
    /// The persistent-buffer arena has no suitable free span.
    Buffer,
    #[error("the requested allocation is invalid")]
    /// The requested shape, count, or alignment is invalid.
    InvalidAllocation,
}

/// Dense pool of stable request-state row identifiers.
///
/// Only [`RequestPool::allocate`] creates a [`RequestSlot`] and only
/// [`RequestPool::free`] consumes one, so a slot cannot be released twice.
/// The slot does not record its pool; the scheduler holds several pools (its
/// main pool and one per media worker) and must free a slot into its issuer.
pub(crate) struct RequestPool {
    /// Unassigned rows as a LIFO stack, initially ordered so that `allocate`
    /// hands out the lowest row first.
    free: Vec<u32>,
}

impl RequestPool {
    /// Creates an allocator for the supplied capacity.
    ///
    /// Rows are numbered `1..=capacity`, matching the worker's one-based
    /// request rows. The capacity is clamped to `1..=u32::MAX` rows.
    pub(crate) fn new(capacity: usize) -> Self {
        let capacity = capacity.clamp(1, u32::MAX as usize);
        Self {
            free: (1..=capacity as u32).rev().collect(),
        }
    }

    /// Returns whether every row is assigned, so `allocate` would fail.
    pub(crate) fn is_empty(&self) -> bool {
        self.free.is_empty()
    }

    /// Returns the number of request rows that can still be assigned.
    pub(crate) fn available(&self) -> usize {
        self.free.len()
    }
}

/// Cache unit pool and cross-group coordination derived from worker
/// capabilities.
pub(crate) struct PrefixCacheManager {
    pub(crate) block_pool: BlockPool,
    pub(crate) coordinator: PrefixCacheCoordinator,
}

impl PrefixCacheManager {
    /// Builds scheduler cache state when a worker advertises a paged unit pool.
    ///
    /// Returns `None` when `info` has no prefix cache. `BlockPool::new` panics on
    /// a pool without an allocatable unit and `BlockTable::new` on an empty
    /// page shape; `PrefixCacheInfo::validate` rejects each of these.
    pub(crate) fn from_main_process_info(info: &MainProcessInfo) -> Option<Self> {
        info.prefix_cache.as_ref().map(|prefix_cache| {
            let block_pool = BlockPool::new(prefix_cache.num_units, prefix_cache.scratch_units);
            // The units of the segments the planes do not hold (a
            // successor's tail its storage does not hold yet, or what a
            // standby's predecessor gave back) stay withdrawn until they are
            // created; the units mapped from a predecessor epoch stay out of
            // allocation until the epoch takes them.
            let usable = prefix_cache.allocatable_units();
            if usable < prefix_cache.num_units {
                block_pool.withdraw(usable);
            }
            block_pool.inherit(prefix_cache.inherited());
            let mut coordinator = PrefixCacheCoordinator::new(
                prefix_cache
                    .groups
                    .iter()
                    .map(GroupShape::from_group)
                    .collect(),
            );
            coordinator.set_seeds(crate::kv_cache::prefix_seeds(prefix_cache));
            PrefixCacheManager {
                block_pool,
                coordinator,
            }
        })
    }

    /// Returns the units the scheduler may allocate, excluding the sentinel.
    pub(crate) fn usable_units(&self) -> usize {
        self.block_pool.usable_units()
    }

    /// Returns the units the scheduler may allocate once none is withdrawn.
    pub(crate) fn capacity_units(&self) -> usize {
        self.block_pool.capacity_units()
    }
}

/// Aligned, byte-addressed persistent Tensor storage.
///
/// A first-fit allocator over free byte extents. Adjacent free extents are
/// coalesced on release.
pub(crate) struct BufferPool {
    bytes: ByteAllocator,
}

impl BufferPool {
    pub(crate) fn new(capacity: u64) -> Self {
        Self {
            bytes: ByteAllocator::new(capacity),
        }
    }
}

impl RequestPool {
    /// Reserve one stable positive row for an admitted request epoch.
    pub(crate) fn allocate(&mut self) -> Result<RequestSlot, OutOfStorage> {
        let index = self.free.pop().ok_or(OutOfStorage::RequestSlots)?;
        Ok(RequestSlot { index })
    }

    /// Return a row after all participating workers acknowledge its release.
    ///
    /// The caller passes the slot this pool issued; consuming it is what
    /// retires the row, so no liveness check remains to fail.
    pub(crate) fn free(&mut self, allocation: RequestSlot) {
        let RequestSlot { index } = allocation;
        self.free.push(index);
    }
}

impl BufferPool {
    /// Reserve an aligned nonempty span; failure does not consume capacity.
    pub(crate) fn allocate(
        &mut self,
        owner: RequestKey,
        bytes: u64,
        alignment: u32,
    ) -> Result<BufferSpan, OutOfStorage> {
        if bytes == 0 || alignment == 0 || !alignment.is_power_of_two() {
            return Err(OutOfStorage::InvalidAllocation);
        }
        let range = self
            .bytes
            .allocate(bytes, alignment)
            .ok_or(OutOfStorage::Buffer)?;
        Ok(BufferSpan {
            owner,
            offset: range.start,
            bytes,
        })
    }

    /// Return the complete span after its export and transport readers retire.
    pub(crate) fn free(&mut self, allocation: BufferSpan) {
        let BufferSpan { offset, bytes, .. } = allocation;
        self.bytes.free(offset..offset + bytes);
    }
}

impl PrefixCacheManager {
    /// Acquire one table per cache group holding `tokens` tokens for a read
    /// from token zero, atomically.
    pub(crate) fn allocate(&self, tokens: u32) -> Result<CacheAllocation, OutOfStorage> {
        let mut allocation = self.coordinator.empty();
        self.coordinator
            .ensure_capacity(
                &self.block_pool,
                &mut allocation,
                0,
                tokens as usize,
                Checkpoint::Boundary,
            )
            .ok_or(OutOfStorage::PrefixCache)?;
        Ok(allocation)
    }

    /// Grow every cache group atomically for a call that reads from token
    /// `read_start` and holds `tokens` tokens, retaining shared prefix
    /// references; its state groups checkpoint as `checkpoint` says.
    pub(crate) fn grow(
        &self,
        allocation: &mut CacheAllocation,
        read_start: u32,
        tokens: u32,
        checkpoint: Checkpoint,
    ) -> Result<(), OutOfStorage> {
        self.coordinator
            .ensure_capacity(
                &self.block_pool,
                allocation,
                read_start as usize,
                tokens as usize,
                checkpoint,
            )
            .ok_or(OutOfStorage::PrefixCache)
    }

    /// Enables or disables prefix reuse.
    pub(crate) fn set_prefix_reuse(&mut self, enabled: bool) {
        self.coordinator.set_prefix_enabled(enabled);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipc::RequestId;

    fn owner(id: u64) -> RequestKey {
        RequestKey::new(1, RequestId(id), 1)
    }

    #[test]
    fn request_rows_remain_exclusive_until_release() {
        let mut pool = RequestPool::new(2);
        let first = pool.allocate().unwrap();
        let second = pool.allocate().unwrap();
        assert_ne!(first.index, second.index);
        assert!(first.index > 0);
        assert!(matches!(pool.allocate(), Err(OutOfStorage::RequestSlots)));
        pool.free(first);
        let replacement = pool.allocate().unwrap();
        assert_ne!(replacement.index, second.index);
        pool.free(second);
        pool.free(replacement);
    }

    #[test]
    fn tensor_spans_are_aligned_exclusive_and_reclaimable() {
        let mut pool = BufferPool::new(1024);
        let first = pool.allocate(owner(1), 200, 256).unwrap();
        let second = pool.allocate(owner(2), 200, 256).unwrap();
        let BufferSpan {
            offset: start,
            bytes,
            ..
        } = &first;
        let BufferSpan {
            offset: other,
            bytes: other_bytes,
            ..
        } = &second;
        assert_eq!(start % 256, 0);
        assert_eq!(other % 256, 0);
        assert!(start + bytes <= *other || other + other_bytes <= *start);
        assert!(matches!(
            pool.allocate(owner(3), 1024, 256),
            Err(OutOfStorage::Buffer)
        ));
        pool.free(first);
        pool.free(second);
        assert!(pool.allocate(owner(3), 1024, 256).is_ok());
    }
}
