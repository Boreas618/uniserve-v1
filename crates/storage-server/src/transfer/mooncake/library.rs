//! Mooncake Transfer Engine's shared library, loaded once per process and never unloaded.
//!
//! The library is opened by its file name, `libtransfer_engine.so`, from the loader's default
//! search path, where the Mooncake package installs it beside `libmooncake_store.so`; no path is
//! configured, and no environment variable points the loader elsewhere. It is opened with
//! `RTLD_NOW | RTLD_LOCAL`, and every entry point these bindings call is resolved from it at once
//! ([`super::capi::Api`]): a missing library, or one that lacks an entry point, is an
//! [`Error::TransferEngineLibrary`] of the first call that needs the Transfer Engine, never an
//! abort. A process that starts no engine never loads it, and building the Storage Server needs
//! no Mooncake installation.

use std::sync::OnceLock;

use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};

use super::capi::Api;
use super::{Error, Result};

/// The library's file name, as the Mooncake package installs it on the loader's default search
/// path.
pub(crate) const LIBRARY: &str = "libtransfer_engine.so";

/// The loaded library and its entry points, or why it cannot be used, for the process's life.
static LOADED: OnceLock<std::result::Result<(Library, Api), String>> = OnceLock::new();

/// The entry points, loading the library on first use.
///
/// # Errors
///
/// Fails when the library is not on the loader's default search path or lacks an entry point
/// these bindings call (the error names it); the first failure stands for the process.
pub(crate) fn api() -> Result<&'static Api> {
    let loaded = LOADED.get_or_init(|| {
        // SAFETY: opening a library runs its initializers; the Transfer Engine's set up its
        // logging and flags and have no other effect on this process.
        let library = unsafe { Library::open(Some(LIBRARY), RTLD_NOW | RTLD_LOCAL) }
            .map_err(|error| format!("it is not installed where the loader finds it: {error}"))?;
        let api = Api::resolve(&library)
            .map_err(|missing| format!("it lacks the entry point {missing}"))?;
        Ok((library, api))
    });
    match loaded {
        Ok((_, api)) => Ok(api),
        Err(reason) => Err(Error::TransferEngineLibrary {
            library: LIBRARY,
            release: super::RELEASE,
            reason: reason.clone(),
        }),
    }
}
