//! A CUDA package's declared compilation: how its translation units become a
//! module (design section 5.2).
//!
//! A CUDA package states its compilation in its manifest, `[compile]`, the way
//! a Triton package states its launch table: the translation units, the C++
//! standard they are written in, the preprocessor definitions each descriptor
//! compiles with (a literal, a field's value, the value a field selects, or a
//! manifest parameter), the libraries the module links beside the CUDA
//! runtime, and the options that are genuinely the package's own, such as host
//! code that must round every floating-point operation. A cubin module adds
//! its kernel: the symbol, the warp count, the dynamic shared memory and the
//! launch signature in parameter order, which the driver writes as the
//! module's launch metadata (`ops::CubinMetadata`'s fields).
//!
//! The manifest reader types and validates the declaration and the Build
//! Server hands it to the compilation in its request; one driver,
//! `uniserve_build_server.cuda`, runs nvcc from the request's toolchain
//! description. A package whose compilation is computed from what it compiles
//! against (a vendor's kernel table, an exported launcher) keeps a build module
//! instead, which calls the same driver's routines.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::launch::{ArgumentType, Constant};

/// A CUDA package's declared compilation, `[compile]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Compile {
    /// The translation units, relative to the package root, each compiled into
    /// one object of the module.
    pub units: Vec<String>,
    /// The C++ standard the units are written in.
    pub standard: Standard,
    /// The preprocessor definitions of every unit, by macro name. A value is
    /// a literal (an integer, a boolean, which is defined as 1 or 0, or the
    /// text of an identifier), a descriptor field's value, the value a field
    /// selects, or a manifest parameter, as a launch table's constant is.
    #[serde(default)]
    pub definitions: BTreeMap<String, Constant>,
    /// The libraries a native module links beside the CUDA runtime, each
    /// found where the toolchain description states it.
    #[serde(default)]
    pub libraries: Vec<Library>,
    /// The header sets the units include beyond the package's own and the
    /// shared kernel sources': each a directory a distribution of the Build
    /// Server's locked environment installs, which the package names in its
    /// `[toolchain] packages`. The lock fixes the distribution by its wheel's
    /// hash, so a header set it installs is not pinned again.
    #[serde(default)]
    pub headers: Vec<Headers>,
    /// Whether `constexpr` functions may be called from host and device code
    /// alike (nvcc's relaxed constexpr), which CuTe's and CUTLASS's headers
    /// require of every unit that includes them.
    #[serde(default)]
    pub relaxed_constexpr: bool,
    /// Whether the host compiler may contract floating-point expressions into
    /// fused multiply-adds, its default. A package whose host arithmetic must
    /// round every operation, as the reference it reproduces does, states
    /// `false`.
    #[serde(default = "contracted")]
    pub host_contraction: bool,
    /// The kernel of a cubin module and its launch metadata; a native module
    /// is launched through its C entries and states none.
    #[serde(default)]
    pub kernel: Option<Kernel>,
}

/// The host compiler's default: contraction allowed.
fn contracted() -> bool {
    true
}

/// A C++ standard a CUDA package is written in, as nvcc spells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Standard {
    #[serde(rename = "c++17")]
    Cxx17,
    #[serde(rename = "c++20")]
    Cxx20,
}

/// A library a native module links, by the name the toolchain description
/// states it under (`uniserve_build_server.toolchain`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Library {
    /// The CUDA driver API, linked against the toolkit's stub; the serving
    /// process loads the driver's own library.
    #[serde(rename = "cuda")]
    Driver,
    /// cuBLASLt, a library of the CUDA toolkit.
    #[serde(rename = "cublasLt")]
    CublasLt,
    /// cuDNN, compiled against the headers and linked against the library of
    /// the distribution the toolchain description names.
    #[serde(rename = "cudnn")]
    Cudnn,
}

/// A header set a locked distribution installs: the distribution's name and
/// the directory, relative to where it installs its files, that the units
/// search.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Headers {
    pub distribution: String,
    pub path: String,
}

/// The kernel of a cubin module: what the module's launch metadata states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    /// The kernel's symbol in the cubin, an `extern "C"` function.
    pub name: String,
    /// The warps of one thread block: a literal, a field or a parameter.
    pub warps: Constant,
    /// The dynamic shared memory of one block, in bytes.
    #[serde(default)]
    pub shared: u32,
    /// The parameters the kernel takes at a launch, in its parameter order:
    /// each name and its type, as a launch table types an argument (a pointer
    /// to the element type a field names, a type a field selects, or a type
    /// as a module's signature spells it).
    pub signature: Vec<(String, ArgumentType)>,
}

impl Compile {
    /// The descriptor fields the declaration reads, so a manifest that reads
    /// one no descriptor declares is refused when it is loaded.
    pub fn fields(&self) -> Vec<&str> {
        let mut fields: Vec<&str> = self
            .definitions
            .values()
            .filter_map(Constant::field)
            .collect();
        if let Some(kernel) = &self.kernel {
            fields.extend(kernel.warps.field());
            fields.extend(kernel.signature.iter().filter_map(|(_, ty)| ty.field()));
        }
        fields
    }

    /// The manifest parameters the declaration reads.
    pub fn parameters(&self) -> Vec<&str> {
        let mut parameters: Vec<&str> = self
            .definitions
            .values()
            .filter_map(Constant::parameter)
            .collect();
        if let Some(kernel) = &self.kernel {
            parameters.extend(kernel.warps.parameter());
        }
        parameters
    }

    /// Why the declaration may not be stated, when it may not.
    pub fn invalid(&self) -> Option<String> {
        if self.units.is_empty() || self.units.iter().any(String::is_empty) {
            return Some("a compilation names its translation units in `units`".into());
        }
        if let Some(headers) = self
            .headers
            .iter()
            .find(|headers| headers.distribution.is_empty() || !relative(&headers.path))
        {
            return Some(format!(
                "the header set {headers:?} names a distribution and a relative directory of it"
            ));
        }
        if let Some(name) = self.definitions.keys().find(|name| !identifier(name)) {
            return Some(format!("the definition `{name}` is no C identifier"));
        }
        if let Some(kernel) = &self.kernel {
            if !identifier(&kernel.name) {
                return Some(format!("the kernel `{}` is no C identifier", kernel.name));
            }
            if self.units.len() != 1 {
                return Some("a cubin module is compiled from one translation unit".into());
            }
            if !self.libraries.is_empty() {
                return Some(
                    "a cubin module links no library: its kernel is loaded into the serving \
                     process's context alone"
                        .into(),
                );
            }
            let mut names: Vec<&str> = kernel
                .signature
                .iter()
                .map(|(name, _)| name.as_str())
                .collect();
            names.sort_unstable();
            if names.windows(2).any(|pair| pair[0] == pair[1]) {
                return Some("the kernel's signature names a parameter twice".into());
            }
        }
        None
    }
}

/// Whether `name` is a C identifier: a letter or `_`, then letters, digits
/// and `_`.
fn identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && characters.all(|rest| rest.is_ascii_alphanumeric() || rest == '_')
}

/// Whether `path` is a non-empty relative path that stays inside the directory
/// it is relative to.
fn relative(path: &str) -> bool {
    let path = std::path::Path::new(path);
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}
