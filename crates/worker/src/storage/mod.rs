//! Physical storage owned by the rank: cache pages, block tables, latent
//! trajectories, cross-call tensors, product buffers and host staging.

mod buffer;
pub(crate) mod latent;
pub(crate) mod memory;
mod output;
mod pinned_pool;
mod prefix_cache;
pub(crate) mod products;
mod request_slots;
pub mod retained;
pub(crate) mod weights;

// The storage owners the integration tests and the storage harness drive.
pub use buffer::{BufferBinding, BufferPool};
pub use latent::{LatentPlan, LatentStorage};
pub use memory::{Storage, StorageView};
pub use output::{LogprobLayout, OutputBuffer, OutputStorage};
pub use prefix_cache::block_tables::{
    BlockTableStorage, BlockTableUpdate, BlockTables, GroupShape, GroupTable,
};
pub use prefix_cache::plan::{CacheGroupPlan, CachePlan, GroupAxis};
pub use prefix_cache::planes::{CachePlanes, UnitColumn};
pub use prefix_cache::{PendingImport, PrefixCache};
pub use products::{
    Metadata, OutputPlacement, ProductBuffer, ProductExport, ProductRead, ProductStore, ValueRange,
};
pub use retained::{Exclusive, ExclusiveMemory, ExclusiveSpec, LocalMemory};

pub(crate) use latent::{LatentCache, Scratch};
pub(crate) use latent::{LatentExport, LatentImport, LatentPool, LatentUpdate, TrajectoryVersion};
pub(crate) use output::{OutputLease, OutputPool};
pub use pinned_pool::PinnedPool;
pub(crate) use prefix_cache::block_tables::{AttentionRow, TablePages};
pub(crate) use prefix_cache::{PageTier, TransferBases, UnitAccesses};
pub(crate) use request_slots::{RequestSlots, SlotField};
