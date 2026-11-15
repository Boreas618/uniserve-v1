//! The crate's error.

use std::io;

use crate::protocol::{ErrorKind, Refusal};

/// Why a request to a Storage Server, or a copy manifest, failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("storage server connection: {0}")]
    Io(#[from] io::Error),
    /// The server refused the hello because it speaks another version of the
    /// protocol than this client ([`Refusal::Version`]): `offered` is this
    /// client's, `spoken` the server's.
    #[error("the storage server speaks version {spoken}, this client {offered}")]
    Version { offered: u32, spoken: u32 },
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
