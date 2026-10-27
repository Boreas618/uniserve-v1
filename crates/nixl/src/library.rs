//! NIXL's C API library, `libnixl_capi.so`, loaded once per process.
//!
//! `nixl-sys` is built with its `stub-api` feature: each `nixl_capi_*`
//! function it calls is a forwarder that, on its first use in the process,
//! opens `libnixl_capi.so` by that bare name (`RTLD_NOW | RTLD_LOCAL`) and
//! calls the symbol of the same name there. A forwarder whose library or
//! symbol is missing aborts the process. Loading the library here first,
//! before any `nixl-sys` call, turns a missing library into an [`Error`]: the
//! dynamic loader then answers the forwarders' open with the library already
//! loaded, which it matches by its soname, `libnixl_capi.so`.
//!
//! The library must come from the NIXL release whose version `nixl-sys`
//! has (the workspace pins both): the forwarders declare the functions of
//! that release's `wrapper.h`.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};

use crate::{Error, Result};

/// The library's file name, which is also its soname and the name the
/// forwarders open.
const NAME: &str = "libnixl_capi.so";

/// The loaded library and the path it was opened by. It is never unloaded:
/// the forwarders keep their own handle to it for the process's life.
static LOADED: Mutex<Option<(PathBuf, Library)>> = Mutex::new(None);

/// Loads NIXL's C API library from `path`, a `libnixl_capi.so` of the NIXL
/// release `nixl-sys` pins (`lib/<arch>-linux-gnu/libnixl_capi.so` of a
/// NIXL installation). Without this call, the first agent loads the library
/// from the dynamic loader's search path.
///
/// The library's own run path, or the loader's search path, must resolve
/// the rest of NIXL and UCX; NIXL finds its backend plugins in `plugins/`
/// beside `libnixl.so`, or in `NIXL_PLUGIN_DIR`.
///
/// # Errors
///
/// Fails when the library cannot be opened, when `nixl-sys`'s calls do not
/// resolve in it (its soname is not `libnixl_capi.so`), or when a library
/// was loaded already from another path.
pub fn load_library(path: &Path) -> Result<()> {
    load(path)
}

/// Loads the library from the loader's search path unless it is loaded.
pub(crate) fn ensure_loaded() -> Result<()> {
    load(Path::new(NAME))
}

fn load(path: &Path) -> Result<()> {
    let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((loaded_path, _)) = loaded.as_ref() {
        // Asking for the library by name, as `Agent::new` does, accepts
        // whichever path loaded it.
        if loaded_path == path || path == Path::new(NAME) {
            return Ok(());
        }
        return Err(Error::Load {
            library: path.to_owned(),
            reason: format!("NIXL is loaded already from {}", loaded_path.display()),
        });
    }

    // SAFETY: opening a library runs its initializers; NIXL's register its
    // plugin manager and logging and have no other effect on this process.
    let library = unsafe { Library::open(Some(path), RTLD_NOW | RTLD_LOCAL) }.map_err(|error| {
        Error::Load {
            library: path.to_owned(),
            reason: error.to_string(),
        }
    })?;

    // The forwarders open `libnixl_capi.so` by name once and remember the
    // outcome; `is_stub` reports whether that open failed. It succeeds only
    // when the loader matches the library opened above by its soname.
    if nixl_sys::is_stub() {
        return Err(Error::Load {
            library: path.to_owned(),
            reason: format!("the library opened, but its soname is not {NAME}"),
        });
    }

    *loaded = Some((path.to_owned(), library));
    Ok(())
}
