//! A program package's manifest, `package.toml` (design section 5.2).
//!
//! One operator's one backend is one package, in
//! `crates/ops/kernels/<op>/<backend>/` (design section 3). Its manifest says
//! what the package is and how its sources become modules:
//!
//! - the operator and backend names;
//! - the toolchain it needs and the version of it the package was written
//!   for, and the external Python packages it imports (the Program Server's
//!   own environment locks them, design section 5.2);
//! - the module's form — a cubin with metadata, or a native library with a C
//!   entry (design section 5.3);
//! - the source files and the kind of source (Triton, CUDA C++, CuTe DSL, a
//!   vendor cubin, a wasm module, an executable);
//! - the architectures it supports and the descriptors it covers, with the
//!   tile constants and the rule that picks a kernel for a descriptor;
//! - its parameters and their defaults;
//! - its tuning data.
//!
//! A revision fixes all of these (design section 5.1): any change to the
//! sources, the tuning data or a toolchain version is a new installation and
//! a new revision, and an installed revision is never overwritten.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use crate::target::TargetKind;

/// Why a package manifest is not usable.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("reading {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("{0}")]
    Invalid(String),
}

/// The module form a package compiles to (design section 5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleForm {
    /// A cubin and a metadata JSON. The main process loads it with
    /// `cuLibraryLoadData` and launches it with `cuLaunchKernelEx`.
    Cubin,
    /// A native library with a C ABI, loaded with `dlopen`. The package's
    /// host code builds the launches, the TMA descriptors and the grid.
    Native,
}

/// A kind of source: what compiles a package's files into a module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// CUDA C++ through nvcc: a cubin, or a native library when the package
    /// brings a C ABI layer.
    Cuda,
    /// Triton through `triton.compile`, which writes a cubin and its
    /// metadata JSON (design section 5.3).
    Triton,
    /// CuTe DSL through its Python compiler and `export_to_c`, linked into a
    /// native library with `libcuda_dialect_runtime_static.a`.
    CuteDsl,
    /// A vendor cubin the package carries, stored as it is.
    Cubin,
    /// A scheduling policy: wasm32, then precompiled for this host's CPU.
    Wasm,
    /// An executable program: stored as it is.
    Executable,
}

/// The toolchain a package needs (design section 6.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolchainKind {
    /// nvcc, the CUDA toolkit's compiler.
    Nvcc,
    /// The Program Server's own Python environment, with Triton and CuTe DSL.
    Python,
    /// The WebAssembly toolchain: a wasm32 build, then `wasmtime`'s
    /// precompilation for this host's CPU.
    Wasm,
}

impl std::fmt::Display for ToolchainKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl ToolchainKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nvcc => "nvcc",
            Self::Python => "python",
            Self::Wasm => "wasm",
        }
    }
}

/// The toolchain a package declares, with the version it was written for and
/// the external Python packages it imports.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolchainDeps {
    pub kind: ToolchainKind,
    /// The version the package is written for, compared with the toolchain's
    /// own version before a build (design section 6.2). A change is a new
    /// revision.
    pub version: String,
    /// The Python packages the sources import, such as FlashAttention-4 and
    /// FlashInfer's CuTe kernels, with the versions the Program Server's
    /// environment locks them to (design section 5.2).
    #[serde(default)]
    pub packages: Vec<PythonPackage>,
}

/// One external Python package a package's sources import.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonPackage {
    pub name: String,
    pub version: String,
}

/// The module a package compiles to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleSpec {
    pub form: ModuleForm,
    /// The metadata JSON's name inside the module directory, for a cubin
    /// module: Triton's own file for a Triton package, or the file the
    /// package writes in the same fields for nvcc (design section 5.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    /// The C entry symbol, for a native module: what the main process calls
    /// after `dlopen` (design section 4.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
}

/// A package's sources and how they are compiled.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceSpec {
    pub kind: SourceKind,
    /// The source files, relative to the package root. An entry may hold the
    /// placeholder `{descriptor}`, which a build replaces with the descriptor
    /// key of the module it builds: a package that carries one vendored cubin
    /// and one metadata JSON per descriptor names them
    /// `cubin/{descriptor}.cubin` and `cubin/{descriptor}.json`.
    pub files: Vec<String>,
    /// The compile script, for a source kind the Program Server drives with a
    /// Python script; relative to the package root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    /// The kernel's name, for a Triton source: the function `triton.compile`
    /// builds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
}

impl SourceSpec {
    /// The files of one module's build: the declared files with the
    /// descriptor placeholder replaced by `key`.
    pub fn files_for(&self, key: &str) -> Vec<String> {
        self.files
            .iter()
            .map(|file| file.replace(DESCRIPTOR_PLACEHOLDER, key))
            .collect()
    }

    /// Whether the declared files are per descriptor.
    pub fn per_descriptor(&self) -> bool {
        self.files
            .iter()
            .any(|file| file.contains(DESCRIPTOR_PLACEHOLDER))
    }
}

/// The placeholder a source file's name may hold for a module's descriptor
/// key.
pub const DESCRIPTOR_PLACEHOLDER: &str = "{descriptor}";

/// The descriptors a package covers (design section 4.2 and 5.2).
///
/// A descriptor is the set of parameter values one compilation covers; its
/// key is their canonical string, such as `dtype_bf16_head_dim_128_page_64`,
/// and the key is the module path's last name (design section 5.3). A package
/// covers a descriptor when it names the descriptor outright in `keys`, or
/// when every dimension of the key takes one of the values the package
/// declares for it in `dimensions`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptors {
    /// Declared descriptor dimensions and the values the package supports,
    /// e.g. `dtype = ["bf16", "fp16"]`, `head_dim = ["128"]`.
    #[serde(default)]
    pub dimensions: BTreeMap<String, Vec<String>>,
    /// Descriptor keys the package covers that the dimensions do not express.
    #[serde(default)]
    pub keys: Vec<String>,
}

impl Descriptors {
    /// Whether the package covers the descriptor whose key this is.
    ///
    /// A key the package names outright is covered. Otherwise the key must
    /// split into `dimension_value` pairs of declared dimensions, each value
    /// in the dimension's declared list; an undeclared dimension, or a value
    /// it does not declare, leaves the descriptor uncovered, and `prepare`
    /// fails (design section 5.2).
    pub fn covers(&self, key: &str) -> bool {
        if self.keys.iter().any(|named| named == key) {
            return true;
        }
        if self.dimensions.is_empty() {
            return false;
        }
        split_key(&self.dimensions, key).is_some_and(|pairs| {
            pairs.iter().all(|(dimension, value)| {
                self.dimensions
                    .get(dimension)
                    .is_some_and(|values| values.iter().any(|allowed| allowed == value))
            })
        })
    }
}

/// Splits a descriptor key into its `dimension_value` pairs, given the
/// dimensions the package declares. The declared dimension names are matched
/// longest first at each position, and a value runs to the next matched
/// dimension.
fn split_key<'a>(
    dimensions: &BTreeMap<String, Vec<String>>,
    key: &'a str,
) -> Option<Vec<(String, &'a str)>> {
    let mut names: Vec<&str> = dimensions.keys().map(String::as_str).collect();
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));
    let mut pairs = Vec::new();
    let mut rest = key;
    while !rest.is_empty() {
        let name = names.iter().find(|name| rest.starts_with(**name))?;
        let after = rest[name.len()..].strip_prefix('_')?;
        // The value runs to the next place where a declared dimension name is
        // followed by a value, so a value may itself hold another dimension's
        // name.
        let end = names
            .iter()
            .filter_map(|next| after.find(&format!("_{next}_")))
            .min()
            .unwrap_or(after.len());
        let value = &after[..end];
        if value.is_empty() {
            return None;
        }
        pairs.push(((*name).to_owned(), value));
        rest = after[end..].strip_prefix('_').unwrap_or_default();
    }
    Some(pairs)
}

/// The kernel program a package provides, and what picks a kernel for a
/// descriptor: the supported architectures, the descriptors, the tile
/// constants and the launch metadata an nvcc cubin needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSpec {
    /// How the target is named for this package (design section 5.3).
    #[serde(default)]
    pub target: TargetKind,
    /// The architectures the package supports, such as `["sm_100a"]`.
    pub architectures: Vec<String>,
    /// The descriptors the package covers.
    pub descriptors: Descriptors,
    /// The tile constants and the rule that picks a kernel for a descriptor.
    /// The package's own compile script and launch code read these; the
    /// Program Server only keeps them with the revision.
    #[serde(default)]
    pub selection: BTreeMap<String, toml::Value>,
}

/// One program package's manifest: `package.toml` at the package root.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The operator's name, which the package fills: also the first name of
    /// its module paths and the `crates/ops/kernels/<op>/` directory
    /// (design section 4.2).
    pub op: String,
    /// The backend's name, which the configuration field selects.
    pub backend: String,
    /// The toolchain and external packages the sources need.
    pub toolchain: ToolchainDeps,
    /// What the package compiles to.
    pub module: ModuleSpec,
    /// The package's sources.
    pub source: SourceSpec,
    /// The kernel program's architectures, descriptors and selection rules.
    pub kernel: KernelSpec,
    /// The program's parameters and their default values (design section
    /// 5.2). An argument the configuration gives replaces its default.
    #[serde(default)]
    pub parameters: BTreeMap<String, toml::Value>,
    /// The tuning data files, relative to the package root (design section
    /// 5.4). They are part of the revision.
    #[serde(default)]
    pub tuning: Vec<String>,
}

impl Manifest {
    /// The manifest and the package root beside it.
    pub fn load(root: &Path) -> Result<Self, ManifestError> {
        let path = root.join("package.toml");
        let text = std::fs::read_to_string(&path).map_err(|source| ManifestError::Read {
            path: path.clone(),
            source,
        })?;
        let manifest: Manifest = toml::from_str(&text).map_err(|source| ManifestError::Parse {
            path: path.clone(),
            source,
        })?;
        manifest.validate(root)?;
        Ok(manifest)
    }

    /// Checks the manifest against the package's files.
    pub fn validate(&self, root: &Path) -> Result<(), ManifestError> {
        if self.op.is_empty() || self.backend.is_empty() {
            return Err(ManifestError::Invalid(
                "the operator and backend names must not be empty".into(),
            ));
        }
        if self.kernel.architectures.is_empty() {
            return Err(ManifestError::Invalid(
                "the package must name the architectures it supports".into(),
            ));
        }
        let missing: Vec<&String> = self
            .source
            .files
            .iter()
            .chain(self.tuning.iter())
            // A file named per descriptor (`cubin/{descriptor}.cubin`) is
            // checked when that descriptor is built.
            .filter(|file| !file.contains(DESCRIPTOR_PLACEHOLDER))
            .filter(|file| !root.join(file).is_file())
            .collect();
        if !missing.is_empty() {
            return Err(ManifestError::Invalid(format!(
                "the package names files it does not have: {missing:?}"
            )));
        }
        if self.module.form == ModuleForm::Native && self.module.entry.is_none() {
            return Err(ManifestError::Invalid(
                "a native module must name its C entry".into(),
            ));
        }
        Ok(())
    }

    /// The source of this package, ready to build its descriptors.
    ///
    /// The kind the manifest names picks the [`Source`](crate::source::Source)
    /// implementation; an unknown kind is a parse error, so this only fails
    /// when a source kind's package files are inconsistent with it.
    pub fn source(&self, root: &Path) -> crate::Result<Box<dyn crate::source::Source>> {
        crate::source::open(self, root)
    }
}
