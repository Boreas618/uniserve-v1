//! The main process's resident memory (`specs/worker.md` §4 `MainProcess.memory`).
//!
//! Resident memory is device storage the epoch keeps until it retires: the KV
//! cache planes, the latent pages, the products of cross-call tensors, the
//! request slots, the retained weights and the encoder cache. Every one of
//! them is an exclusive storage at the host's Storage Server, keyed by the
//! inputs `crate::keys` composes, so a restarted main process of the same
//! scope and epoch maps what its predecessor left instead of loading it
//! again.

pub mod allocation;
pub mod buffer;
pub mod encoders;
pub mod latent;
pub mod output;
pub mod pinned_pool;
pub mod prefix_cache;
pub mod request_slots;
pub mod retained;

pub(crate) use prefix_cache::block_tables::AttentionRow;
pub use prefix_cache::block_tables::{
    BlockTableStorage, BlockTableUpdate, BlockTables, GroupShape, GroupTable,
};
pub use prefix_cache::plan::{CacheDescription, CacheGroupPlan, CachePlan, GroupAxis, StateField};
pub use prefix_cache::planes::{CachePlanes, LaidSegment, PlaneLayout, UnitColumn};
pub use prefix_cache::{PendingImport, PrefixCache};
pub(crate) use prefix_cache::{TransferBases, UnitAccesses};
