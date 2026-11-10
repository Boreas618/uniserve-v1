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

use ipc::RequestKey;

mod bytes;
pub(crate) mod requests;
pub(crate) use bytes::ByteAllocator;

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
