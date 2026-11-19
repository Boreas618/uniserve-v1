//! The crate's error.

use std::io;

use crate::protocol::{ErrorKind, Refusal};

/// Why a request to a Storage Server, or a copy manifest, failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("storage server connection: {0}")]
    Io(#[from] io::Error),
    /// The server refused the hello because the windows of the protocol's
    /// versions it and this client speak share none ([`Refusal::Version`]),
    /// which the refusal names.
    #[error("the storage server refused this client's versions: {0}")]
    Version(crate::protocol::VersionRefusal),
    /// The server refused the connection's hello for another reason.
    #[error("the storage server refused the connection: {0}")]
    Refused(Refusal),
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
