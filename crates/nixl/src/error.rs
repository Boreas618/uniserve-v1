//! Typed failures of NIXL calls and of loading NIXL's C API library.

use std::path::PathBuf;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A failure reported by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// NIXL's C API library could not be loaded, or the calls of
    /// `nixl-sys` do not resolve in the library that was.
    #[error("cannot load {library}: {reason}", library = .library.display())]
    Load { library: PathBuf, reason: String },

    /// A NIXL call failed. NIXL's C API reports a few status classes
    /// (`InvalidParam`, `NotFound`, `BackendError`), not the backend's own
    /// error.
    #[error("{call} failed: {source}")]
    Nixl {
        call: &'static str,
        #[source]
        source: nixl_sys::NixlError,
    },

    /// An argument NIXL would reject or misread, refused before any call.
    #[error("{0}")]
    Invalid(String),
}

impl Error {
    /// Wraps the failure of the NIXL call `call`, for `map_err`.
    pub(crate) fn nixl(call: &'static str) -> impl FnOnce(nixl_sys::NixlError) -> Self {
        move |source| Self::Nixl { call, source }
    }
}
