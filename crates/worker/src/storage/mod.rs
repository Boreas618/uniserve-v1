//! Physical storage owned by the rank: KV pages, block tables, latent
//! trajectories, cross-call tensors, product buffers and host staging.

mod block_tables;
mod buffer;
mod host_buffers;
mod kv_cache;
mod latent;
pub mod memory;
mod output;
pub mod tensor;

pub use block_tables::{
    AttentionRow, BlockTableUpdate, BlockTables, GroupShape, GroupTable, TablePages, table_pages,
};
pub use buffer::{BufferBinding, BufferPool};
pub use host_buffers::HostBuffers;
pub use kv_cache::KVCacheManager;
pub use latent::{LatentExport, LatentImport, LatentPool, LatentUpdate};
pub use output::{LogprobLayout, OutputBuffer, OutputPool, OutputStorage};
