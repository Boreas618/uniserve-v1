//! The crate's error.

use std::io;

/// Why a process the Storage Server runs, its Transfer Server or its
/// Mooncake connector, stopped.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] platform::Error),
    #[error("mooncake: {0}")]
    Mooncake(#[from] mooncake_store::StoreError),
    #[error(transparent)]
    Storage(#[from] ::storage::Error),
    #[error(transparent)]
    Io(#[from] io::Error),
    /// The connector's settings name no usable store.
    #[error("{0}")]
    Settings(String),
    /// The host's transfer calibration cannot be read, does not hold together, or names a GPU
    /// the driver does not show (`transfer::calibration`).
    #[error("the transfer calibration {0}")]
    Calibration(String),
    /// NIXL's C API library cannot be used: it is not installed where the loader's default search
    /// finds it, or it lacks an entry point the bindings of `release` call.
    #[error("NIXL's C API ({library}, of NIXL {release}) cannot be used: {reason}")]
    NixlLibrary {
        library: &'static str,
        release: &'static str,
        reason: String,
    },
    /// A NIXL call failed. NIXL's C API reports a few status classes (`ERROR_INVALID_PARAM`,
    /// `ERROR_NOT_FOUND`, `ERROR_BACKEND`, ...), not the backend's own error, which NIXL logs.
    #[error("{call} failed: {name} ({status})", name = crate::transfer::nixl::status_name(*.status))]
    Nixl { call: &'static str, status: i32 },
    /// An argument NIXL would reject or misread, refused before any call.
    #[error("{0}")]
    NixlArgument(String),
}
