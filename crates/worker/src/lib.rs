//! The UniServe rank runtime.
//!
//! A worker rank is a Rust process body hosted by a Python interpreter: the
//! `uniserve.worker` entry loads this library through TVM-FFI and calls
//! its exported `main` with the function that resolves import paths. Rust
//! owns request state, storage, streams, graphs policy, transfers and the
//! IPC service; Python computes through the library functions Rust imports
//! by path.
//!
//! The library also exports the video request rules of `uniserve-core` to
//! the Python library, which plans MiniMax-H3 requests with them.
//!
//! Besides that entry point the crate exposes only what its integration
//! tests (`tests/`), its test harnesses (`examples/`) and its benchmark drive:
//! the launch configuration and registration, the interpreter and DLPack
//! export, the storage owners, the transfer mechanisms, the host groups and
//! collectives, and the image runners. Everything else is internal to the
//! rank runtime.

mod capacity;
mod completion;
pub mod config;
mod cuda_graph;
mod error;
mod executor;
pub mod ffi;
mod host;
mod lane;
pub mod launch;
mod logging;
pub mod parallel;
mod profiling;
mod request;
mod rules;
mod runner;
mod sampling;
mod service;
pub mod storage;
mod stream;
pub mod transfer;
mod warmup;
mod worker;

pub use completion::{Completion, Outcome};
pub use cuda_graph::GraphPool;
pub use error::{Error, ErrorCode, Result, WorkerError};
pub use runner::ImageSetup;
pub use runner::decoder::{DecodeImage, DecoderRunner};
pub use runner::encoder::{
    Encoded, EncoderConfig, EncoderInfo, EncoderRunner, ImagePreparation, LatentEncoder,
    PreparedImage, TextEncoder, Tower, VisionEncoder,
};
pub use stream::{CUDAStream, EventPool, Fill, Issue, RowFill};

pub(crate) use executor::{Batch, Executor, Submission};
pub(crate) use host::HostTask;
pub(crate) use request::{RequestPool, RequestProgress};
pub(crate) use storage::{
    BlockTables, BufferBinding, LatentPool, LatentStorage, LatentUpdate, OutputBuffer, OutputPool,
    ProductBuffer, ProductRead, ProductStore,
};
pub(crate) use transfer::DescriptorGrants;

/// Lock `mutex`, keeping the state a panicking holder left.
///
/// Every build profile sets `panic = "abort"`, so a holder's panic ends the
/// rank before another thread observes the poisoned lock. Only test builds,
/// which ignore that setting and unwind, reach the recovery; it returns the
/// guard rather than raising a second panic in the thread that locks next.
pub(crate) fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Keep `value`, and every resource it owns, until the process exits.
///
/// An aborted close, or a release whose device completion is unknown,
/// cannot tell when device work, a DMA engine or a peer rank stops
/// addressing what `value` owns: releasing it could hand memory, a mapping
/// or a stream that is still in use to a new owner. The value is leaked
/// instead, and process exit returns its resources to the driver.
#[expect(
    clippy::mem_forget,
    reason = "resources whose last device access is unknown must outlive it"
)]
pub(crate) fn leak<T>(value: T) {
    std::mem::forget(value);
}

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
