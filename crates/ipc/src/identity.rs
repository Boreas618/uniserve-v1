//! Identities the engine and its workers name requests, calls and cache
//! units by.

use serde::{Deserialize, Serialize};

/// Index of a physical allocation unit in the paged cache pool.
///
/// The engine's block pool allocates units, and a block table maps each
/// logical page of a cache group onto that group's number of units; worker
/// calls name them as unit ids. The worker treats unit zero as its padding
/// sentinel, valid storage in every plane, so the pool never allocates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct UnitId(pub u32);

/// Logical call identity, preserved when a scheduler batch is split across workers.
/// The request index is the ordinal assigned at selection, before physical row packing.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct CallId {
    /// Scheduler batch the call was selected into.
    pub batch_id: u64,
    /// Ordinal of the call within that batch at selection time.
    pub request_index: u32,
}

impl CallId {
    pub const fn new(batch_id: u64, request_index: u32) -> Self {
        Self {
            batch_id,
            request_index,
        }
    }
}
