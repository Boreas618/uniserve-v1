//! Native kernel libraries: the `.so` form of a kernel program's module
//! (`specs/worker.md` §5.3, §4.2's `Kernel::Native`).
//!
//! A kernel program whose source needs host code — CuTe DSL's `export_to_c`,
//! FlashInfer's launch code and planner, the cuDNN frontend — is compiled to
//! a shared object with a C ABI, with its cubin embedded. This process
//! `dlopen`s that object and calls the entry point the package declares,
//! passing it the stream the executor issues on and the arguments of the
//! launch. The object's own launches are captured with the stream, so a
//! native kernel is captured into a graph exactly like a cubin one.
//!
//! The entry point's C signature is the package's contract with `ops`
//! (`Kernel::Native { library, entry }`): this module only opens the object
//! and resolves the symbol with the type the caller declares.
//!
//! The library stays loaded while anything holds it and is closed when the
//! last owner drops it; §7.3's purge order (destroy or rebind the graphs
//! that call it first) is the owner's rule, not this module's.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;

use crate::{Error, Result};

/// A native kernel library loaded from `path` with `dlopen`.
///
/// The library is opened local to this process (`RTLD_LOCAL`), so its
/// symbols stay out of the process's global namespace: two kernel programs
/// may export the same entry point without seeing each other's.
pub struct NativeLibrary {
    library: Library,
    path: PathBuf,
}

// SAFETY: the code of a loaded shared object is executable from any thread;
// `dlclose` runs on the last owner's drop, and a call into the library is an
// `unsafe` call whose thread contract is the package's.
unsafe impl Send for NativeLibrary {}
// SAFETY: as for `Send`; a shared reference is only used to resolve an entry
// point, which does not mutate the library.
unsafe impl Sync for NativeLibrary {}

impl NativeLibrary {
    /// Opens the shared object at `path`.
    ///
    /// # Errors
    ///
    /// [`Error::NativeLoad`] when the file cannot be opened.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // SAFETY: `dlopen` runs only the library's own initializers.
        let library = unsafe { Library::new(path) }.map_err(|source| Error::NativeLoad {
            path: path.to_path_buf(),
            source: Arc::new(source),
        })?;
        Ok(Self {
            library,
            path: path.to_path_buf(),
        })
    }

    /// The file the library was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The entry point the package declares under `symbol`, with the C
    /// signature the caller declares for it.
    ///
    /// # Safety
    ///
    /// `F` must be the C signature the package declares for `symbol`: a
    /// wrong signature is undefined behaviour at the call, and the caller
    /// keeps the library (and everything the entry point's arguments
    /// address) alive for every call.
    ///
    /// # Errors
    ///
    /// [`Error::NativeSymbol`] when the object does not export `symbol`.
    pub unsafe fn entry<F: Copy>(&self, symbol: &str) -> Result<F> {
        // SAFETY: forwarded from the caller; the symbol is copied out while
        // the library is alive.
        unsafe { self.library.get::<F>(symbol.as_bytes()) }
            .map(|symbol| *symbol)
            .map_err(|source| Error::NativeSymbol {
                path: self.path.clone(),
                symbol: symbol.to_owned(),
                source: Arc::new(source),
            })
    }
}

impl std::fmt::Debug for NativeLibrary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("NativeLibrary")
            .field(&self.path)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_library_is_named_in_the_error() {
        let error = NativeLibrary::open("/nonexistent/kernel.so")
            .expect_err("no such file")
            .to_string();

        assert!(error.contains("/nonexistent/kernel.so"), "{error}");
    }

    #[test]
    fn an_entry_point_the_library_lacks_is_reported_with_its_symbol() {
        // The C library is a shared object that is always present, so this
        // needs no fixture file.
        let library = NativeLibrary::open("libc.so.6").expect("the C library loads");
        // SAFETY: the symbol is only resolved, never called.
        let missing = unsafe {
            library
                .entry::<unsafe extern "C" fn()>("platform_tests_no_such_entry")
                .expect_err("the C library exports no such entry")
        };

        assert!(
            missing.to_string().contains("platform_tests_no_such_entry"),
            "{missing}"
        );
        assert_eq!(library.path(), Path::new("libc.so.6"));
    }

    #[test]
    fn a_declared_entry_point_resolves_to_a_callable_function() {
        // `strlen` is a C entry point like a kernel program's host code: a
        // known symbol with a declared signature.
        type Strlen = unsafe extern "C" fn(*const std::ffi::c_char) -> usize;

        let library = NativeLibrary::open("libc.so.6").expect("the C library loads");
        // SAFETY: `strlen`'s declared signature is `Strlen`.
        let strlen: Strlen = unsafe { library.entry("strlen") }.expect("strlen is exported");
        let text = c"platform";
        // SAFETY: the NUL-terminated string is `strlen`'s declared argument.
        let len = unsafe { strlen(text.as_ptr()) };

        assert_eq!(len, text.to_bytes().len());
    }
}
