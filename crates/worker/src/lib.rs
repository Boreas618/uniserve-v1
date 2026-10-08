//! The UniServe rank runtime.
//!
//! A worker rank is a Rust process body hosted by a Python interpreter: the
//! `uniserve_models.worker` entry loads this library through TVM-FFI and calls
//! its exported `main` with the numerical interface table. Rust owns request
//! state, storage, streams, graphs policy, transfers and the IPC service;
//! Python computes through the functions in that table.

pub mod capacity;
mod completion;
pub mod config;
pub mod cuda;
mod error;
mod executor;
mod host;
pub mod nccl;
mod parallel;
mod profiling;
mod request;
mod runner;
mod sampling;
mod service;
pub mod storage;
mod stream;
pub mod transfer;

pub use completion::{Completion, Outcome};
pub use config::WorkerConfig;
pub use error::{Error, Result};
pub use executor::{
    Backend, Batch, BatchInputs, BatchResult, Executor, InputReady, InputWait, PendingOutput,
    Submission, request_output,
};
pub use host::{HostAction, HostLane, HostTask};
pub use parallel::{ExpertExchange, Microbatches, WeightPrefetch, yield_microbatch};
pub use request::{Request, RequestPool, RequestProgress};
pub use runner::{
    GraphPool, GraphPools, GraphStorage, ModelBatch, ModelRunners, PrefillShape, TextShapes,
    TokenSelection, graph_storage_budget_bytes, prefill_shapes, prefill_units,
};
pub use sampling::{
    SamplingMetadata, SamplingPath, TOKEN_CONTINUATION_BIT, TOKEN_VALUE_MASK, finish_token_ids,
};
pub use service::{Service, ServiceBackend};
pub use storage::{
    AttentionRow, BlockTableUpdate, BlockTables, BufferBinding, BufferPool, GroupShape,
    GroupTable, HostBuffers, KVCacheManager, LatentExport, LatentImport, LatentPool,
    LatentUpdate, LogprobLayout, OutputBuffer, OutputPool, OutputStorage, TablePages,
    table_pages,
};
pub use stream::{CUDAStream, EventPool};
pub use transfer::{
    BufferRegistry, DescriptorGrants, ImportBackend, ImportCopy, KVImport, KVImporter,
    ReadBackend, ReadRegion, ReadReservation, RegisteredBuffer, SHM_HEADER_BYTES, SharedBuffer,
    SharedMapping, SharedRead, TransferCapacity, TransferPool, TransferRead, TransferTicket,
    fetch_descriptor, plan_reads,
};
