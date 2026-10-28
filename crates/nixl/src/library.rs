//! NIXL's C API library, `libnixl_capi.so`, loaded once per process and
//! never unloaded.
//!
//! The library is opened with `RTLD_NOW | RTLD_LOCAL`, and every entry point
//! this crate calls is resolved from it at once ([`crate::capi::Api`]): a
//! missing library, or one that lacks an entry point, is an [`Error`] of the
//! first call that needs NIXL, never an abort. A process that creates no
//! agent never loads NIXL, and building this crate needs no NIXL
//! installation.
//!
//! The library must come from the NIXL release this crate binds, 1.5.1,
//! built as `docs/nixl.md` describes: the entry points are
//! declared as that release's `wrapper.h` declares them, and a registration
//! for reads alone ([`crate::Access::Read`]) relies on the read-only
//! registration patch that build applies to the UCX backend.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};

use crate::capi::Api;
use crate::{Error, Result};

/// The library's file name, which the loader's search path finds when no
/// path was given.
const NAME: &str = "libnixl_capi.so";

/// The loaded library, the path it was opened by and its entry points,
/// leaked: they live for the process.
struct Loaded {
    path: PathBuf,
    api: Api,
    _library: Library,
}

static LOADED: Mutex<Option<&'static Loaded>> = Mutex::new(None);

/// Loads NIXL's C API library from `path`, a `libnixl_capi.so` of the NIXL
/// release this crate binds (`lib/<arch>-linux-gnu/libnixl_capi.so` of a
/// NIXL installation). Without this call, the first agent loads the library
/// from the dynamic loader's search path.
///
/// The library's own run path, or the loader's search path, must resolve
/// the rest of NIXL and UCX; NIXL finds its backend plugins in `plugins/`
/// beside `libnixl.so`, or in `NIXL_PLUGIN_DIR`.
///
/// # Errors
///
/// Fails when the library cannot be opened, when it lacks an entry point
/// this crate calls (the error names it), or when a library was loaded
/// already from another path.
pub fn load_library(path: &Path) -> Result<()> {
    load(path).map(|_| ())
}

/// The entry points, loading the library from the loader's search path
/// unless it is loaded.
pub(crate) fn api() -> Result<&'static Api> {
    load(Path::new(NAME))
}

fn load(path: &Path) -> Result<&'static Api> {
    let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(loaded) = *loaded {
        // Asking for the library by name, as an agent does, accepts
        // whichever path loaded it.
        if loaded.path == path || path == Path::new(NAME) {
            return Ok(&loaded.api);
        }
        return Err(Error::Load {
            library: path.to_owned(),
            reason: format!("NIXL is loaded already from {}", loaded.path.display()),
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
    let api = Api::resolve(&library).map_err(|missing| Error::Load {
        library: path.to_owned(),
        reason: format!("it lacks the entry point {missing}"),
    })?;

    let entry: &'static Loaded = Box::leak(Box::new(Loaded {
        path: path.to_owned(),
        api,
        _library: library,
    }));
    *loaded = Some(entry);
    Ok(&entry.api)
}
