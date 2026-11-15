//! Native kernel libraries: the `.so` form of a kernel package's module
//! (`ops::Kernel::Native`).
//!
//! A kernel package whose source needs host code — CuTe DSL's `export_to_c`,
//! FlashInfer's launch code and planner, the cuDNN frontend — is compiled to
//! a shared object with a C ABI, with its cubin embedded. This process
//! `dlopen`s that object and calls the entry point the package declares,
//! passing it the stream the worker issues on and the arguments of the
//! launch. The object's own launches are captured with the stream, so a
//! native kernel is captured into a graph exactly like a cubin one.
//!
//! The entry point's C signature is the package's interface with `ops`
//! (`Kernel::Native { library, entry }`): this module only opens the object
//! and resolves the symbol with the type the caller declares.
//!
//! The library stays loaded while anything holds it and is closed when the
//! last owner drops it; the order a generation purge follows (destroy or
//! rebind the graphs that call it first) is the owner's rule, not this
//! module's.
//!
//! Before an object is opened, its bytes can be asked what it defines
//! ([`NativeLibrary::defined_functions`]) and where it finds its siblings
//! ([`NativeLibrary::run_path`]): a reader of the ELF dynamic tables that
//! runs no code of the object.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;
use object::read::elf::ElfFile64;
use object::{Endianness, Object as _, ObjectSymbol as _, SymbolKind, elf};

use crate::{Error, Result};

/// A native kernel library loaded from `path` with `dlopen`.
///
/// The library is opened local to this process (`RTLD_LOCAL`), so its
/// symbols stay out of the process's global namespace: two kernel packages
/// may export the same entry point without seeing each other's.
pub struct NativeLibrary {
    library: Library,
    path: PathBuf,
}

// SAFETY: the code of a loaded shared object is executable from any thread;
// `dlclose` runs on the last owner's drop, and a call into the library is an
// `unsafe` call whose thread-safety rules are the package's.
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

    /// Every function name an ELF object defines for the dynamic linker, read from its bytes.
    ///
    /// This answers which file of a multi-file module is its library (the file that defines the
    /// entry the package declared) and what an object defines when `dlsym` refused an entry,
    /// without opening the object. The dynamic symbol table is read with the `object` crate's ELF
    /// reader, which bounds-checks every read against `bytes`. Anything it does not understand -
    /// another format, a table that runs off the end, a name that is not UTF-8 - is `None`, so a
    /// caller diagnosing a fault cannot be handed a second one.
    #[must_use]
    pub fn defined_functions(bytes: &[u8]) -> Option<Vec<String>> {
        let file = ElfFile64::<Endianness>::parse(bytes).ok()?;
        file.dynamic_symbols()
            // An undefined symbol is one the object imports, not one it defines.
            .filter(|symbol| symbol.is_definition() && symbol.kind() == SymbolKind::Text)
            .map(|symbol| symbol.name().ok().map(str::to_owned))
            .collect()
    }

    /// The run path an ELF object states, read from its bytes: the first `DT_RUNPATH` or `DT_RPATH`
    /// entry of its dynamic table. `None` when it states neither, or when the bytes are not an
    /// object the reader understands (as for [`Self::defined_functions`]).
    #[must_use]
    pub fn run_path(bytes: &[u8]) -> Option<String> {
        let file = ElfFile64::<Endianness>::parse(bytes).ok()?;
        let table = file
            .elf_section_table()
            .dynamic_table(file.endian(), bytes)
            .ok()?;
        let entry = table
            .iter()
            .find(|entry| entry.tag == elf::DT_RUNPATH || entry.tag == elf::DT_RPATH)?;
        let path = table.string(entry).ok()?;
        std::str::from_utf8(path).ok().map(str::to_owned)
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
        // `strlen` is a C entry point like a kernel package's host code: a
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

    /// The dynamic-table reader answers about a real object, and says nothing it cannot check.
    ///
    /// Every Linux process has its C library loaded, and the dynamic loader names the object that
    /// defines `malloc` (`dladdr`); that object is asked what it defines. Bytes that are not an
    /// object are no answer at all.
    #[test]
    fn the_functions_an_object_defines_are_read_from_its_bytes() {
        // SAFETY: an all-zero `Dl_info` is a valid value of the plain C struct.
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        // SAFETY: `malloc` is an address in a loaded object; the loader fills `info`.
        let found = unsafe { libc::dladdr(libc::malloc as *const std::ffi::c_void, &mut info) };
        assert_ne!(found, 0, "the loader names the object that defines malloc");
        // SAFETY: the loader returned the object's NUL-terminated path, valid while it is loaded.
        let libc = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) }
            .to_str()
            .unwrap()
            .to_owned();
        let bytes = std::fs::read(&libc).expect("a loaded object is readable");

        let names = NativeLibrary::defined_functions(&bytes)
            .unwrap_or_else(|| panic!("{libc} is a loaded object and its functions were not read"));

        assert!(
            names.len() > 100,
            "{libc} defines {} functions, too few for a C library",
            names.len()
        );
        assert!(
            names.iter().any(|name| name == "malloc"),
            "{libc} does not list malloc"
        );
        assert!(
            names.iter().all(
                |name| name.starts_with(|held: char| held.is_ascii_alphabetic() || held == '_')
            ),
            "a name read out of an object is not a name: {:?}",
            &names[..names.len().min(8)]
        );
        assert_eq!(NativeLibrary::defined_functions(b"not an object"), None);
        assert_eq!(NativeLibrary::run_path(b"not an object"), None);
    }
}
