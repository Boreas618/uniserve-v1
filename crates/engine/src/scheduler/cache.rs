//! What a scheduler knows of the cache (`specs/scheduler-design.md` §4.5): the [`Cached`] row of
//! every cached block it knows of, which the scheduler keeps in `Scheduler::cache` by the
//! block's number ([`Cached::id`]), and per region the eviction order of the blocks no request of
//! the scheduler holds (`Scheduler::order`).
//!
//! The table is exact for what this scheduler's queue pair commits and evicts; blocks of other
//! queue pairs are learned from `Listing` and from the answers that name their keys. A belief
//! that is wrong costs reuse, never a wrong block: every request that names a key is answered
//! from the buffer pool's own cache.

use storage::Key;

/// A cached block the scheduler knows of.
///
/// Invariant: it is in its region's eviction order exactly when it is on that region's tier and
/// `pins == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cached {
    pub key: Key,
    pub class: u8,
    /// The tokens the block holds: the class's block size, or fewer for a request's tail.
    pub tokens: u16,
    /// The tiers that hold a copy: bit 0 the device, bit 1 the host, bit 2 the remote store.
    pub tiers: u8,
    /// The worker groups whose memory holds its device copy, one bit per group number.
    pub groups: u8,
    /// Requests of this scheduler holding it.
    pub pins: u32,
    /// Its last use in nanoseconds: its rank in the eviction order.
    pub last_used: u64,
}

impl Cached {
    /// The number the scheduler knows a cached block by: its key's leading eight bytes,
    /// big-endian. It is the policy's `PageId` for a KV block and its `EncodingId` for an encoder
    /// output, so a policy's id finds its block in one lookup.
    pub fn id(key: &Key) -> u64 {
        let mut leading = [0; 8];
        leading.copy_from_slice(&key.0[..8]);
        u64::from_be_bytes(leading)
    }
}
