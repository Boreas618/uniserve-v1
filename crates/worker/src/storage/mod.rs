//! Physical storage owned by the rank: cache pages, block tables, latent
//! trajectories, cross-call tensors, product buffers and host staging.

mod buffer;
mod host_buffers;
pub(crate) mod latent;
pub mod memory;
mod output;
mod pinned_pool;
mod prefix_cache;
pub(crate) mod products;
mod request_slots;

pub use buffer::{BufferBinding, BufferPool};
pub use host_buffers::HostBuffers;
pub(crate) use latent::{LatentCache, Scratch};
pub use latent::{LatentExport, LatentImport, LatentPool, LatentStorage, LatentUpdate};
pub use output::{LogprobLayout, OutputBuffer, OutputLease, OutputPool, OutputStorage};
pub use pinned_pool::{PinnedBlock, PinnedPool};
pub use prefix_cache::block_tables::{
    AttentionRow, BlockTableStorage, BlockTableUpdate, BlockTables, GroupShape, GroupTable,
    TablePages, table_pages,
};
pub use prefix_cache::plan::{CacheGroupPlan, CachePlan, GroupAxis};
pub use prefix_cache::planes::{CachePlanes, FIELDS, UnitColumn, UnitSnapshot};
pub use prefix_cache::{PendingImport, PrefixCache, ScratchUnits, TransferBases, UnitAccesses};
pub use products::{
    Metadata, OutputPlacement, ProductBuffer, ProductExport, ProductRead, ProductStore, Scalar,
    ValueRange,
};
pub use request_slots::{RequestSlots, SlotField};
