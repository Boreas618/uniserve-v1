//! Installing one module of a generation layer by operator name.
//!
//! A generation layer arrives as modules keyed by operator name and descriptor key, and the worker's
//! precopy has to turn each one into a kernel of a generation without knowing any operator's type.
//! This is that entry: it takes the operator's name, the descriptor key the module was compiled
//! for, and the module's image or path, and hands back the loaded kernel inside a [`KernelEntry`]
//! for the caller to put in a [`crate::Generation`].
//!
//! Nothing here allocates device memory or launches anything, so it is safe at precopy time and
//! outside any capture.

use std::path::Path;
use std::sync::Arc;

use platform::NativeLibrary;

use crate::generation::KernelEntry;
use crate::kernel::{CubinMetadata, Kernel};
use crate::{Error, Result};

/// One module of a generation layer, as the package delivered it.
pub enum Module<'a> {
    /// A cubin, fatbin or PTX image with the metadata record written beside it, which names the
    /// kernel to resolve and states its launch shape ([`CubinMetadata`]).
    Cubin {
        /// The image itself.
        image: &'a [u8],
        /// The module's metadata record, as JSON.
        metadata: &'a [u8],
    },
    /// A shared object the platform loads, and the C entries the package declares.
    Native {
        /// The library on disk, in the directory the module's files were published in.
        path: &'a Path,
        /// The entry point's name, constants and launch scalars as the package declared them.
        entry: &'a str,
        /// The module's load entry ([`LOAD_ENTRY`]), for a module that reads files published
        /// beside its library: it is called once the library is open, before any launch, with the
        /// directory of the module's files.
        load: Option<&'a str>,
    },
}

/// The role of a native module's load entry among the C entries its manifest declares
/// (`[module.entries]`): `int32_t load(LoadCall* call, void* stream)`, called once when the module
/// is installed with the directory of the module's files ([`crate::kernel::LoadCall`]).
pub const LOAD_ENTRY: &str = "load";

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
    let Some(name) = proto::module::operator::named(operator) else {
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
        Module::Cubin { image, metadata } => {
            // A module's own metadata names the kernel it holds: one package may build a different
            // kernel per descriptor, as paged attention does per phase.
            Kernel::load_cubin(device, image, CubinMetadata::parse(metadata)?)
        }
        Module::Native { path, entry, load } => {
            let library = NativeLibrary::open(path)?;
            let kernel = Kernel::load_native(library, entry)?;
            if let Some(load) = load {
                // A module's files are published together in one directory, and its library is
                // opened from its own file there, so the library's directory is the module's.
                let directory = path.parent().ok_or_else(|| {
                    Error::Metadata(format!(
                        "the module's library {} is in no directory",
                        path.display()
                    ))
                })?;
                kernel.load_directory(load, directory)?;
            }
            Ok(kernel)
        }
    }
}
