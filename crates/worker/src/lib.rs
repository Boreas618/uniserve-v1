//! The UniServe rank runtime.
//!
//! A worker rank is a Rust process body hosted by a Python interpreter: the
//! `uniserve.worker` entry loads this library through TVM-FFI and calls
//! its exported `main` with the function that resolves import paths. Rust
//! owns request state, storage, streams, graphs policy, transfers and the
//! IPC service; Python computes through the library functions Rust imports
//! by path.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod capacity;
mod completion;
pub mod config;
mod cuda_graph;
pub mod error;
pub mod executor;
pub mod ffi;
mod handoff;
mod host;
pub mod lane;
pub mod launch;
mod logging;
pub mod model;
pub mod parallel;
pub mod profiling;
mod request;
mod runner;
mod sampling;
mod service;
pub mod storage;
mod stream;
pub mod transfer;
pub mod warmup;
pub mod worker;

pub use completion::{Completion, Outcome};
pub use config::WorkerConfig;
pub use cuda_graph::{
    CUDAGraph, Capacity, GraphCounts, GraphMode, GraphPool, GraphPools, GraphStorage, GraphTable,
    Lookup, bucket, capture_order, graph_storage_budget_bytes, reserve_pools,
};
pub use error::{Error, ErrorCode, Result, WorkerError};
pub use executor::{
    Backend, Batch, BatchInputs, BatchResult, Executor, InputReady, InputWait, PendingOutput,
    Submission, request_output,
};
pub use host::{HostAction, HostLane, HostTask};
pub use parallel::{
    ExpertExchange, Microbatches, StreamCommunicator, WeightPrefetch, yield_microbatch,
};
pub use request::diffusion::{
    Branch, BranchSet, BranchSource, Coordinates, ForwardSlot, GuidancePlan, ImageTrajectory,
    Prefix, PrefixFill, Trajectories,
};
pub use request::{Request, RequestPool, RequestProgress};
pub use runner::ImageSetup;
pub use runner::attention::{AttentionCapacity, AttentionMetadata, ROW_SECTIONS, pack_rows};
pub use runner::decoder::{DecodeImage, DecoderRunner};
pub use runner::denoiser::image::{
    ConditionRow, DenoiseRow, DiffusionCapacity, DiffusionShape, DiffusionStorage, ImageDiffusion,
    IntegrateRow, Predicted,
};
pub use runner::denoiser::image_model::ImageDenoiserLayout;
pub use runner::encoder::{
    Encoded, EncoderConfig, EncoderGraph, EncoderInfo, EncoderRunner, LatentEncoder,
    MAX_PACKED_IMAGES, PreparedImage, PromptVision, TextEncoder, Tower, VisionEncoder,
    packed_grids, prepare_input, text_capacity,
};
pub use runner::vision::{
    BlockPositions, FeatureInjection, FeatureLayout, PositionLayout, VisionBlock,
};
pub use runner::{ModelBatch, ModelRunners, TokenSelection};
pub use sampling::{
    SamplingMetadata, SamplingPath, TOKEN_CONTINUATION_BIT, TOKEN_VALUE_MASK, finish_token_ids,
};
pub use service::{Service, ServiceBackend};
pub use storage::{
    AttentionRow, BlockTableStorage, BlockTableUpdate, BlockTables, BufferBinding, BufferPool,
    CachePlan, CachePlanes, GroupShape, GroupTable, HostBuffers, LatentExport, LatentImport,
    LatentPool, LatentStorage, LatentUpdate, LogprobLayout, OutputBuffer, OutputPool,
    OutputStorage, PrefixCache, PrefixCacheManager, ProductBuffer, ProductRead, RequestSlots,
    TablePages, TensorStore, UnitColumn, table_pages,
};
pub use stream::{CUDAStream, Deferred, EventPool, Fill, Issue, RowFill, Submitter};
pub use transfer::{
    BufferRegistry, CacheImport, CacheImporter, DescriptorGrants, ImportBackend, ImportCopy,
    PeerStorage, Read, ReadBackend, ReadRegion, ReadReservation, RegisteredBuffer,
    SHM_HEADER_BYTES, SharedBuffer, SharedMapping, SharedRead, Spans, TransferCapacity,
    TransferPool, TransferRead, TransferTicket, Transport, TransportKind, TransportOptions,
    Transports, fetch_descriptor, plan_reads,
};

/// The library's exported entry point, `main(argv, import) -> int`.
///
/// The Python entry calls it once with the process arguments and the
/// function that resolves an import path to a library callable
/// (`uniserve.worker.import_path`); the returned value is the process exit
/// status.
fn main(argv: tvm_ffi::Array<tvm_ffi::String>, import: tvm_ffi::Function) -> tvm_ffi::Result<i64> {
    let argv: Vec<String> = argv.iter().map(|value| value.as_str().to_owned()).collect();
    Ok(launch::main(&argv, import))
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(main, main);
