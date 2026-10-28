//! Typed failures of NIXL calls and of loading NIXL's C API library.

use std::path::PathBuf;

use crate::capi::{Status, status_name};

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A failure reported by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// NIXL's C API library could not be loaded, or it lacks an entry point
    /// this crate calls.
    #[error("cannot load {library}: {reason}", library = .library.display())]
    Load { library: PathBuf, reason: String },

    /// A NIXL call failed. NIXL's C API reports a few status classes
    /// (`ERROR_INVALID_PARAM`, `ERROR_NOT_FOUND`, `ERROR_BACKEND`, ...), not
    /// the backend's own error, which NIXL logs.
    #[error("{call} failed: {name} ({status})", name = status_name(*.status))]
    Nixl { call: &'static str, status: i32 },

    /// An argument NIXL would reject or misread, refused before any call.
    #[error("{0}")]
    Invalid(String),
}

/// `Ok` for a call that returned `SUCCESS`, the failure of `call` otherwise.
pub(crate) fn check(call: &'static str, status: Status) -> Result<()> {
    if status == crate::capi::SUCCESS {
        Ok(())
    } else {
        Err(Error::Nixl { call, status })
    }
}
