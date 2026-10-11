//! Physical storage owned by the rank: KV pages, block tables, latent
//! trajectories, cross-call tensors, product buffers and host staging.

mod block_tables;
mod buffer;
mod host_buffers;
mod kv;
mod kv_cache;
mod kv_plan;
mod kv_planes;
mod latent;
mod latent_storage;
pub mod memory;
mod output;
mod request_slots;
pub mod tensor;
mod tensor_store;

pub use block_tables::{
    AttentionRow, BlockTableStorage, BlockTableUpdate, BlockTables, GroupShape, GroupTable,
    TablePages, table_pages,
};
pub use buffer::{BufferBinding, BufferPool};
pub use host_buffers::HostBuffers;
pub use kv::{KVCache, PendingImport, ScratchUnits};
pub use kv_cache::KVCacheManager;
pub use kv_plan::{GroupAxis, KVGroupPlan, KVPlan};
pub use kv_planes::{FIELDS, KVPlanes, UnitColumn, UnitSnapshot, tensor_map};
pub(crate) use latent::image::{ImageLatents, Scratch};
pub use latent::{LatentExport, LatentImport, LatentPool, LatentUpdate};
pub use latent_storage::LatentStorage;
pub use output::{LogprobLayout, OutputBuffer, OutputLease, OutputPool, OutputStorage};
pub use request_slots::{RequestSlots, SlotField};
pub use tensor_store::{
    Metadata, OutputPlacement, ProductBuffer, ProductRead, Scalar, TensorStore, ValueRange,
};
