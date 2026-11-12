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
}
