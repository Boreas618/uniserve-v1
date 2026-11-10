//! Installing one module of a program layer by operator name.
//!
//! A program layer arrives as modules keyed by operator name and descriptor key, and the worker's
//! precopy has to turn each one into a kernel of a generation without knowing any operator's type.
//! This is that entry: it takes the operator's name, the descriptor key the module was compiled
//! for, and the module's image or path, and hands back the loaded kernel inside a [`KernelEntry`]
//! for the caller to put in a [`crate::Generation`].
//!
//! Nothing here allocates device memory or launches anything, so it is safe at precopy time and
//! outside any capture.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use crate::generation::KernelEntry;
use crate::kernel::{CubinMetadata, Kernel};
use crate::platform::NativeLibrary;
use crate::{Result, platform};

/// One module of a program layer, as the package delivered it.
pub enum Module<'a> {
    /// A cubin, fatbin or PTX image, with its Triton metadata as JSON when the package carries it,
    /// and the symbol to resolve.
    Cubin {
        /// The image itself.
        image: &'a [u8],
        /// Triton's metadata record for the image, as the package wrote it.
        metadata: Option<&'a [u8]>,
        /// The kernel's symbol in the image.
        entry: &'a str,
    },
    /// A shared object the platform loads, and the C entry the package declares.
    Native {
        /// The library on disk.
        path: &'a Path,
        /// The entry point's name, constants and launch scalars as the package declared them.
        entry: &'a str,
    },
}

/// Load `module` as the kernel of `operator` for `descriptor_key`.
///
/// `Ok(None)` means this build of `ops` has no operator called `operator`: the module belongs to a
/// build that has, and installing it here would let a generation launch a kernel no layer asks for.
/// `Err` means the name is ours and the module does not load.
pub fn install(
    device: &platform::Device,
    operator: &str,
    descriptor_key: &str,
    module: Module<'_>,
) -> Result<Option<KernelEntry>> {
    let Some(name) = proto::program::operator::named(operator) else {
        return Ok(None);
    };
    let kernel = load(device, module)?;
    Ok(Some(KernelEntry::keyed(
        name,
        descriptor_key,
        Arc::new(kernel),
    )))
}

/// Load one module into a kernel.
fn load(device: &platform::Device, module: Module<'_>) -> Result<Kernel> {
    match module {
        Module::Cubin {
            image,
            metadata,
            entry,
        } => {
            // A module's own metadata names the kernel it holds. One package may build a different
            // kernel per descriptor, as paged attention does per phase, so a package-wide entry
            // names only one of them.
            let metadata = match metadata {
                Some(json) => CubinMetadata::parse(json)?,
                None => CubinMetadata::named(entry),
            };
            Kernel::load_cubin(device, image, metadata)
        }
        Module::Native { path, entry } => {
            let entry = intern(entry)?;
            let library = NativeLibrary::open(path)?;
            Kernel::load_native(library, entry)
        }
    }
}

/// `entry` as a `&'static str`, one allocation per distinct name.
///
/// The platform resolves a native entry by a name that outlives the library, and a package may
/// install the same entry many times over (one per descriptor key), so the names are kept once.
fn intern(entry: &str) -> Result<&'static str> {
    static NAMES: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let names = NAMES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut names = names.lock().map_err(|_| crate::Error::Unsupported {
        operator: "install",
        problem: "an entry name was interned while another thread panicked".to_owned(),
    })?;
    if let Some(name) = names.get(entry) {
        return Ok(name);
    }
    let leaked: &'static str = Box::leak(entry.to_owned().into_boxed_str());
    names.insert(entry.to_owned(), leaked);
    Ok(leaked)
}
