//! The crate's error.

use std::io;

use crate::protocol::ErrorKind;

/// Why a request to a Storage Server, or a copy manifest, failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("storage server connection: {0}")]
    Io(#[from] io::Error),
    #[error("the storage server refused the connection: {0}")]
    Refused(String),
    /// The server refused the request ([`crate::protocol::Response::Error`]).
    #[error("the storage server responded {kind:?}: {message}")]
    Server { kind: ErrorKind, message: String },
    #[error("the storage server responded out of protocol: {0}")]
    Protocol(String),
    /// A copy manifest's layout or request contradicts itself.
    #[error("invalid copy manifest input: {0}")]
    Manifest(String),
}

pub type Result<T> = std::result::Result<T, Error>;
