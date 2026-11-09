//! Native kernel libraries: the `.so` form of a kernel program's module
//! (`ops::Kernel::Native`).
//!
//! A kernel program whose source needs host code — CuTe DSL's `export_to_c`,
//! FlashInfer's launch code and planner, the cuDNN frontend — is compiled to
//! a shared object with a C ABI, with its cubin embedded. This process
//! `dlopen`s that object and calls the entry point the package declares,
//! passing it the stream the worker issues on and the arguments of the
//! launch. The object's own launches are captured with the stream, so a
//! native kernel is captured into a graph exactly like a cubin one.
//!
//! The entry point's C signature is the package's contract with `ops`
//! (`Kernel::Native { library, entry }`): this module only opens the object
//! and resolves the symbol with the type the caller declares.
//!
//! The library stays loaded while anything holds it and is closed when the
//! last owner drops it; the order a program-layer purge follows (destroy or
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

    /// Every function name an ELF object defines for the dynamic linker, read from its bytes.
    ///
    /// This answers which file of a multi-file module is its library (the file that defines the
    /// entry the package declared) and what an object defines when `dlsym` refused an entry,
    /// without opening the object. Only the dynamic symbol table and its string table are read, and
    /// every read is bounds-checked against `bytes`. Anything this reader does not understand -
    /// another class, another byte order, a table that runs off the end - is `None`, so a caller
    /// diagnosing a fault cannot be handed a second one.
    #[must_use]
    pub fn defined_functions(bytes: &[u8]) -> Option<Vec<String>> {
        const SHT_DYNSYM: u32 = 11;
        const STT_FUNC: u8 = 2;
        // An `Elf64_Sym` is 24 bytes: name (u32), info (u8), other (u8), section index (u16),
        // value (u64), size (u64).
        const SYMBOL_SIZE: usize = 24;

        let sections = sections(bytes)?;
        let symbols = sections
            .iter()
            .position(|section| section.kind == SHT_DYNSYM)?;
        let (symbols_at, symbols_size) = (sections[symbols].at, sections[symbols].size);
        let strings = sections.get(sections[symbols].link)?;
        let table = bytes.get(strings.at..strings.at.checked_add(strings.size)?)?;

        let mut names = Vec::new();
        let limit = symbols_at.checked_add(symbols_size)?;
        let mut at = symbols_at;
        while at.checked_add(SYMBOL_SIZE)? <= limit {
            let name = quarter(bytes, at)? as usize;
            let info = *bytes.get(at.checked_add(4)?)?;
            let section = half(bytes, at.checked_add(6)?)?;
            at = at.checked_add(SYMBOL_SIZE)?;
            // An undefined symbol (section index 0) is one the object imports, not one it defines.
            if name == 0 || section == 0 || (info & 0xf) != STT_FUNC {
                continue;
            }
            let tail = table.get(name..)?;
            let end = tail.iter().position(|held| *held == 0)?;
            names.push(std::str::from_utf8(&tail[..end]).ok()?.to_owned());
        }
        Some(names)
    }

    /// The run path an ELF object states, read from its bytes: the first `DT_RUNPATH` or `DT_RPATH`
    /// entry of its dynamic table. `None` when it states neither, or when the bytes are not an
    /// object this reader understands (as for [`Self::defined_functions`]).
    #[must_use]
    pub fn run_path(bytes: &[u8]) -> Option<String> {
        const SHT_DYNAMIC: u32 = 6;
        const DT_RPATH: u64 = 15;
        const DT_RUNPATH: u64 = 29;
        // An `Elf64_Dyn` is 16 bytes: tag (u64), value (u64).
        const ENTRY_SIZE: usize = 16;

        let sections = sections(bytes)?;
        let dynamic = sections
            .iter()
            .find(|section| section.kind == SHT_DYNAMIC)?;
        let strings = sections.get(dynamic.link)?;
        let table = bytes.get(strings.at..strings.at.checked_add(strings.size)?)?;
        let limit = dynamic.at.checked_add(dynamic.size)?;
        let mut at = dynamic.at;
        while at.checked_add(ENTRY_SIZE)? <= limit {
            let tag = word(bytes, at)?;
            let value = word(bytes, at.checked_add(8)?)? as usize;
            at = at.checked_add(ENTRY_SIZE)?;
            if tag != DT_RUNPATH && tag != DT_RPATH {
                continue;
            }
            let tail = table.get(value..)?;
            let end = tail.iter().position(|held| *held == 0)?;
            return std::str::from_utf8(&tail[..end]).ok().map(str::to_owned);
        }
        None
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

/// One section header of an ELF object, as much of it as the dynamic-table reader uses.
struct Section {
    /// `sh_type`.
    kind: u32,
    /// `sh_offset`: where the section's bytes start in the file.
    at: usize,
    /// `sh_size`, in bytes.
    size: usize,
    /// `sh_link`: the index of the section this one names its strings in.
    link: usize,
}

/// The section headers of a 64-bit little-endian ELF object; `None` for any other file.
fn sections(bytes: &[u8]) -> Option<Vec<Section>> {
    const ELFCLASS64: u8 = 2;
    const ELFDATA2LSB: u8 = 1;

    if bytes.get(0..4)? != b"\x7fELF"
        || *bytes.get(4)? != ELFCLASS64
        || *bytes.get(5)? != ELFDATA2LSB
    {
        return None;
    }
    // `e_shoff`, `e_shentsize` and `e_shnum` of the ELF header.
    let at = word(bytes, 0x28)? as usize;
    let entry = half(bytes, 0x3a)? as usize;
    let count = half(bytes, 0x3c)? as usize;
    let mut sections = Vec::with_capacity(count.min(256));
    for index in 0..count {
        let header = at.checked_add(index.checked_mul(entry)?)?;
        sections.push(Section {
            kind: quarter(bytes, header.checked_add(4)?)?,
            at: word(bytes, header.checked_add(0x18)?)? as usize,
            size: word(bytes, header.checked_add(0x20)?)? as usize,
            link: quarter(bytes, header.checked_add(0x28)?)? as usize,
        });
    }
    Some(sections)
}

/// The little-endian `u64` at `at`, when the bytes hold one there.
fn word(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(at..at.checked_add(8)?)?.try_into().ok()?,
    ))
}

/// The little-endian `u16` at `at`, when the bytes hold one there.
fn half(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(at..at.checked_add(2)?)?.try_into().ok()?,
    ))
}

/// The little-endian `u32` at `at`, when the bytes hold one there.
fn quarter(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
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

    /// The dynamic-table reader answers about a real object, and says nothing it cannot check.
    ///
    /// Every Linux process has its C library loaded, and this one is asked about the object its maps
    /// name; a host whose maps name no such path is the one case where the test has nothing to ask.
    /// Bytes that are not an object are no answer at all.
    #[test]
    fn the_functions_an_object_defines_are_read_from_its_bytes() {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        let libc = maps
            .lines()
            .filter_map(|line| line.split_whitespace().last())
            .find(|path| path.contains("libc.so.6") && path.starts_with('/'))
            .map(str::to_owned);
        let Some(libc) = libc else {
            return;
        };
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
