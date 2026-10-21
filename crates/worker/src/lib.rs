//! The UniServe worker runtime.
//!
//! A worker is a Rust process that hosts the Python interpreter: the
//! `uniserve worker` command loads this library and calls its exported
//! [`worker_main`], which runs the control plane on the process's main
//! thread and the interpreter on a Python thread of its own. Rust owns
//! request state, storage, streams, graphs policy, transfers and the IPC
//! service; Python computes through the library functions Rust imports by
//! path.
//!
//! The library also exports the video request rules of the `models` crate to
//! the Python library, which plans MiniMax-H3 requests with them.
//!
//! Besides that entry point the crate exposes only what its integration
//! tests (`tests/`), its test harnesses (`examples/`) and its benchmark drive:
//! the launch configuration and registration, the interpreter and DLPack
//! export, the storage owners, the transfer mechanisms, the host groups and
//! collectives, and the image runners. Everything else is internal to the
//! worker runtime.

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
pub mod memory;
pub mod parallel;
mod profiling;
mod request;
mod rules;
mod runner;
mod sampling;
mod service;
mod stream;
pub mod transfer;
mod warmup;
mod worker;

use std::os::unix::ffi::OsStrExt;

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
pub(crate) use memory::{
    BlockTables, BufferBinding, LatentPool, LatentStorage, LatentUpdate, OutputBuffer, OutputPool,
    ProductBuffer, ProductRead, ProductStore,
};
pub(crate) use request::{RequestPool, RequestProgress};
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

/// The worker's entry: the `uniserve worker` command calls it once, on
/// the process's main thread, and exits with the status it returns, without
/// Python finalization.
///
/// `argv` holds the `argc` process arguments after the program name, and
/// `python` the path of the environment's interpreter, which the worker's
/// interpreter takes as its executable. The executable has loaded that
/// environment's shared `libpython` into the process's global scope.
///
/// # Safety
///
/// `argv` points to `argc` NUL-terminated strings and `python` to one, all
/// valid for the duration of the call; the function is called at most once
/// per process.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn worker_main(
    argc: std::ffi::c_int,
    argv: *const *const std::ffi::c_char,
    python: *const std::ffi::c_char,
) -> std::ffi::c_int {
    let text = |pointer: *const std::ffi::c_char| {
        // SAFETY: the caller passes valid NUL-terminated strings.
        unsafe { std::ffi::CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    };
    let arguments: Vec<String> = (0..usize::try_from(argc).unwrap_or(0))
        // SAFETY: `argv` holds `argc` pointers.
        .map(|index| text(unsafe { *argv.add(index) }))
        .collect();
    // SAFETY: as above.
    let python = std::ffi::OsString::from(std::ffi::OsStr::from_bytes(unsafe {
        std::ffi::CStr::from_ptr(python).to_bytes()
    }));
    launch::main(&arguments, std::path::Path::new(&python))
}
