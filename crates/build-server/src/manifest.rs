//! A package's manifest, `package.toml` (design section 5.2).
//!
//! One operator's one backend is one package, in
//! `crates/ops/kernels/<op>/<backend>/` (design section 3). Its manifest says
//! what the package is and how its sources become modules:
//!
//! - the operator and backend names;
//! - the toolchain it needs and the version of it the package was written
//!   for, and the external Python packages it imports (the Build Server's
//!   own environment locks them, design section 5.2);
//! - the module's form — a cubin with metadata, or a native library with a C
//!   entry (design section 5.3);
//! - the source files and the kind of source (a build module of Triton, CUDA
//!   C++ or CuTe DSL, a vendor cubin, a crate cargo builds, an executable);
//! - the architectures it supports and the descriptors it covers, with the
//!   tile constants and the rule that picks a kernel for a descriptor;
//! - its parameters and their defaults;
//! - its tuning data.
//!
//! A revision fixes all of these (design section 5.1): any change to the
//! sources, the tuning data or a toolchain version is a new installation and
//! a new revision, and an installed revision is never overwritten. The
//! revision records them, with the compile driver they are built through, as
//! its build inputs ([`crate::inputs`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

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
    /// A wasm32 module, which the engine's wasmtime compiles under its own
    /// configuration when it loads it. A scheduling policy is such a package
    /// revision (design section 5.3); it is a flat byte blob, not a kernel, so
    /// it has neither launch metadata nor a C entry.
    Wasm,
    /// An executable the main process runs as its own process, rather
    /// than loading into itself. It is stored as it is, so it has neither
    /// launch metadata nor a C entry.
    Executable,
}

/// A kind of source: what compiles a package's files into a module.
///
/// The first three are a **build module**, the Python module the manifest's
/// [`SourceSpec::script`] names, compiled through the Build Server's Python
/// environment with a typed compile request (`crate::source`): the kind names
/// how the module states its compilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// CUDA C++: the build module's `compile(request)` is the whole
    /// compilation, to a cubin or a native library.
    Cuda,
    /// Triton through `triton.compile`, which writes a cubin and its
    /// metadata JSON (design section 5.3): the build module states the
    /// descriptor's `spec`.
    Triton,
    /// CuTe DSL through its Python compiler and `export_to_c`, linked into a
    /// native library with `libcuda_dialect_runtime_static.a`: the build
    /// module states the descriptor's `spec`, and the entry unit
    /// [`SourceSpec::unit`] defines the C entry.
    CuteDsl,
    /// A vendor cubin the package carries, stored as it is.
    Cubin,
    /// A scheduling policy: a crate of a Cargo workspace, built for wasm32
    /// from the snapshot an installation takes of the whole workspace.
    Wasm,
    /// An executable: stored as it is.
    Executable,
    /// A crate of a Cargo workspace, built by cargo to the executable
    /// [`SourceSpec::binary`] names, from the snapshot an installation takes
    /// of the whole workspace: the engine.
    Cargo,
}

impl SourceKind {
    /// Whether the source is a build module the Python environment compiles.
    pub fn compiled_by_module(self) -> bool {
        matches!(self, Self::Cuda | Self::Triton | Self::CuteDsl)
    }

    /// Whether the source is a crate cargo builds from a workspace snapshot.
    pub fn builds_with_cargo(self) -> bool {
        matches!(self, Self::Wasm | Self::Cargo)
    }

    /// The toolchains a source of this kind builds with: what a manifest's
    /// `[toolchain] kind` may name for it.
    pub fn toolchains(self) -> &'static [ToolchainKind] {
        match self {
            Self::Cuda | Self::Triton | Self::CuteDsl => &[
                ToolchainKind::Nvcc,
                ToolchainKind::Triton,
                ToolchainKind::CuteDsl,
            ],
            Self::Wasm | Self::Cargo => &[ToolchainKind::Cargo],
            Self::Cubin | Self::Executable => &[ToolchainKind::None],
        }
    }
}

/// The toolchain a package needs (design section 6.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolchainKind {
    /// nvcc, the CUDA toolkit's compiler.
    Nvcc,
    /// Triton, from the Build Server's own Python environment.
    Triton,
    /// CuTe DSL (the `nvidia-cutlass-dsl` distribution), from the same
    /// environment.
    CuteDsl,
    /// cargo, at the compiler release the package states: it builds a crate
    /// of its workspace snapshot, for the host or for wasm32, with the
    /// lockfile and vendored registry sources, offline.
    Cargo,
    /// Nothing: a source that compiles nothing, such as a vendor cubin or an
    /// executable, has no toolchain version to match.
    None,
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
            Self::Triton => "triton",
            Self::CuteDsl => "cute_dsl",
            Self::Cargo => "cargo",
            Self::None => "none",
        }
    }
}

impl ModuleSpec {
    /// The metadata file's name inside the module directory, for one
    /// descriptor: the manifest may name it with `{descriptor}`, because a
    /// package that carries one metadata JSON per descriptor names one file
    /// per descriptor.
    pub fn metadata_name(&self, descriptor: &str) -> Option<String> {
        self.metadata
            .as_ref()
            .map(|name| name.replace(DESCRIPTOR_PLACEHOLDER, descriptor))
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
    ///
    /// An empty version means any version this host has: the package's build
    /// does not depend on the patch release, so a host whose toolchain is
    /// another build of the same release builds it and the store keeps the
    /// module under the revision it was installed as. A fixture that must run
    /// on whichever host the caller's test uses declares it; a serving package
    /// declares the version its modules were validated against.
    pub version: String,
    /// The Python distributions the sources import, such as FlashAttention-4
    /// and FlashInfer's CuTe kernels, by their distribution names
    /// (`flash-attn-4`, `flashinfer-python`) and with the versions the Build
    /// Server's environment locks them to (design section 5.2). A build is
    /// refused when the environment has one at another version or not at all
    /// (`Toolchains::check_packages`).
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
    /// The metadata JSON's name inside the module directory, which every cubin
    /// module carries: Triton's own file for a Triton package, or the file the
    /// package writes in the same fields (design section 5.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    /// A native module's C entry symbols, by role: what the main process
    /// calls after `dlopen` (design section 4.2). [`LAUNCH_ENTRY`] is the
    /// entry that launches the kernel and every native module has it; the
    /// other roles are lifecycle entries with the same calling convention,
    /// such as an expert exchange creating, suspending and restoring a
    /// transport workspace around a launch.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub entries: BTreeMap<String, String>,
    /// A file the package's compile script writes into the build's output:
    /// a JSON object of role to symbol, which overrides [`Self::entries`] for
    /// the module this descriptor builds. A package whose modules each launch
    /// through their own C entry, such as an expert exchange dispatching on
    /// its mode, names it; `[module.entries]` stays the package's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entries_file: Option<String>,
}

/// The entry role every native module has: the one that launches the kernel.
pub const LAUNCH_ENTRY: &str = "launch";

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
    /// The build module of a source a build module compiles
    /// ([`SourceKind::compiled_by_module`]), relative to the package root: the
    /// compilation runs in its directory and imports it by its name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    /// The C++ translation unit that defines a native module's C entries, for
    /// a source whose driver compiles it (a CuTe DSL kernel's, or a shared
    /// build's such as the routed trtllm-gen products'); relative to the
    /// package root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// The cubin of a vendor cubin source, which the module carries as
    /// `module.cubin`; it may hold [`DESCRIPTOR_PLACEHOLDER`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cubin: Option<String>,
    /// The launch metadata of a vendor cubin source, which the module carries
    /// under [`ModuleSpec::metadata`]; it may hold [`DESCRIPTOR_PLACEHOLDER`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    /// The executable a cargo source builds and the module holds, such as
    /// `uniserve-engine`: a binary target of the package's crate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary: Option<String>,
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

/// The key of the one module a package without a `[kernel]` section builds:
/// a scheduling policy and the engine each fill a whole interface, not a set
/// of shapes. No descriptor key is it: a key is `dimension_value` pairs.
pub const SINGLE_KEY: &str = "default";

/// The descriptors a package covers (design section 4.2 and 5.2).
///
/// A descriptor is the set of parameter values one compilation covers; its
/// key is their canonical string, such as `dtype_bf16_head_dim_128_page_64`,
/// and the key is the module path's last name (design section 5.3). A package
/// covers a descriptor when the key splits into `dimension_value` pairs of the
/// dimensions it declares and every one of them admits its value.
///
/// A dimension states what the kernel supports, never the widths of the models
/// that launch it: an enumeration for a value the kernel is specialized on, a
/// [`Constraint`] for a value it handles generically. The Build Server
/// compiles one module per key, so a width the kernel bakes in at compile time
/// is still generic when the kernel compiles for every value that meets the
/// constraint.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptors {
    /// The declared dimensions, by name: `dtype = ["bf16", "f16"]`,
    /// `columns = { multiple = 16 }`.
    #[serde(default)]
    pub dimensions: BTreeMap<String, Dimension>,
}

impl Descriptors {
    /// Whether the package covers the descriptor whose key this is: the key
    /// reads whole against the declared dimensions ([`Self::fields`]). An
    /// undeclared dimension, or a value its dimension does not admit, leaves
    /// the descriptor uncovered, and `prepare` fails (design section 5.2).
    pub fn covers(&self, key: &str) -> bool {
        self.fields(key).is_some()
    }

    /// Reads a descriptor key into its fields, each a dimension and its value
    /// in the key's order, or `None` when the key does not read whole.
    ///
    /// This is the one reader of the key grammar: coverage is whether it reads
    /// a key, and a build script is handed what it read. A key is
    /// `dimension_value` pairs joined with `_`, in whatever order the operator
    /// prints them. Names and enumerated values may hold `_`, so a field is
    /// the longest declared name the rest starts with, and its value the
    /// longest enumerated value that follows it whole or, for a dimension
    /// declared by a constraint, the text up to the next `_` when it meets
    /// the constraint: no integer, extent tuple or bit pattern holds one. A
    /// declared dimension the key leaves out is not refused: the package's
    /// compilation accounts for a field it needs and the key lacks.
    pub fn fields(&self, key: &str) -> Option<Vec<(String, String)>> {
        if self.dimensions.is_empty() {
            return None;
        }
        let mut names: Vec<&str> = self.dimensions.keys().map(String::as_str).collect();
        names.sort_by_key(|name| std::cmp::Reverse(name.len()));

        let mut fields = Vec::new();
        let mut rest = key;
        while !rest.is_empty() {
            let name = names.iter().find(|name| {
                rest.strip_prefix(**name)
                    .is_some_and(|after| after.starts_with('_'))
            })?;
            let after = &rest[name.len() + 1..];
            let value = match self.dimensions.get(*name)? {
                Dimension::Values(values) => values
                    .iter()
                    .filter(|value| {
                        after
                            .strip_prefix(value.as_str())
                            .is_some_and(|tail| tail.is_empty() || tail.starts_with('_'))
                    })
                    .max_by_key(|value| value.len())?
                    .as_str(),
                Dimension::Constraint(constraint) => {
                    let value = after.split('_').next().unwrap_or_default();
                    constraint.admits(value).then_some(value)?
                }
            };
            if value.is_empty() {
                return None;
            }
            fields.push(((*name).to_owned(), value.to_owned()));
            let tail = &after[value.len()..];
            rest = tail.strip_prefix('_').unwrap_or(tail);
        }
        Some(fields)
    }
}

/// What one descriptor dimension admits.
///
/// ```toml
/// dtype = ["bf16", "f16"]            # an enumeration
/// columns = { multiple = 16 }        # a positive multiple of 16
/// block = { power_of_two = true }    # a power of two
/// first = { min = 0 }                # a non-negative integer
/// axes = { min = 1, max = 4 }        # an integer in 1..=4
/// shape = { separator = "x" }        # positive integers joined by `x`: `2048x64x1x128`
/// epsilon = { bits = 32 }            # an `f32` bit pattern: `0x358637bd`
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dimension {
    /// The values the kernel is specialized on: element types, flags, modes,
    /// and the integers whose supported set is a property of the kernel, such
    /// as the head dimensions a CUDA template is instantiated for or the tile
    /// shapes a GEMM is written for.
    Values(Vec<String>),
    /// A value the kernel handles for anything that meets the constraint.
    Constraint(Constraint),
}

impl Dimension {
    /// Whether this dimension admits `value`, exactly as a key spells it.
    pub fn admits(&self, value: &str) -> bool {
        match self {
            Self::Values(values) => values.iter().any(|allowed| allowed == value),
            Self::Constraint(constraint) => constraint.admits(value),
        }
    }

    /// One value this dimension admits: an enumeration's first value, or a
    /// constraint's least one; `None` when it admits nothing.
    pub fn example(&self) -> Option<String> {
        match self {
            Self::Values(values) => values.first().cloned(),
            Self::Constraint(constraint) => constraint.example(),
        }
    }
}

/// The constraint a width-generic dimension imposes on its value.
///
/// Without `bits` the value is a decimal integer, written without leading
/// zeros as a key prints it, that is at least `min` (1 unless stated), at most
/// `max` when stated, a multiple of `multiple` (1 unless stated) and, with
/// `power_of_two`, a power of two. With `separator` the value is one or more
/// such integers joined by it, each meeting the same constraint: an extent
/// tuple the kernel takes whole, such as a block-major shape.
///
/// With `bits` the value is a floating-point number as `ops` prints one into a
/// key: its bit pattern after `0x` in lowercase hexadecimal without leading
/// zeros, of at most `bits` bits. `bits` takes none of the integer fields.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constraint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multiple: Option<u64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub power_of_two: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub separator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits: Option<u32>,
}

impl Constraint {
    /// Whether `value` meets the constraint.
    pub fn admits(&self, value: &str) -> bool {
        if let Some(bits) = self.bits {
            return admits_bits(bits, value);
        }
        match &self.separator {
            Some(separator) => value
                .split(separator.as_str())
                .all(|part| self.admits_integer(part)),
            None => self.admits_integer(value),
        }
    }

    /// The least value the constraint admits, when it admits one.
    pub fn example(&self) -> Option<String> {
        if self.bits.is_some() {
            return Some("0x0".to_owned());
        }
        let multiple = self.multiple.unwrap_or(1).max(1);
        let mut value = self.min.unwrap_or(1).div_ceil(multiple) * multiple;
        while !self.admits_integer(&value.to_string()) {
            value = value.checked_add(multiple)?;
            if self.max.is_some_and(|max| value > max) {
                return None;
            }
        }
        Some(value.to_string())
    }

    /// Why a manifest may not state this constraint, when it may not.
    pub fn invalid(&self) -> Option<&'static str> {
        let integer = self.min.is_some()
            || self.max.is_some()
            || self.multiple.is_some()
            || self.power_of_two
            || self.separator.is_some();
        let separator = self.separator.as_deref().unwrap_or("x");
        match self.bits {
            Some(32 | 64) if integer => Some("`bits` states a float and takes no integer field"),
            Some(32 | 64) => None,
            Some(_) => Some("`bits` is 32 or 64"),
            None if self.multiple == Some(0) => Some("`multiple` must be positive"),
            None if self.max.is_some_and(|max| max < self.min.unwrap_or(1)) => {
                Some("`max` is below `min`")
            }
            None if separator.is_empty()
                || separator.contains(|c: char| c == '_' || c.is_ascii_digit()) =>
            {
                Some("a `separator` is not empty and holds neither `_` nor a digit")
            }
            None => None,
        }
    }

    /// Whether `text` is an integer in canonical decimal that meets the
    /// integer fields.
    fn admits_integer(&self, text: &str) -> bool {
        // A key prints an integer without leading zeros, so `064` names no value.
        let canonical = !text.is_empty()
            && text.bytes().all(|byte| byte.is_ascii_digit())
            && (text == "0" || !text.starts_with('0'));
        let Some(value) = canonical.then(|| text.parse::<u64>().ok()).flatten() else {
            return false;
        };
        value >= self.min.unwrap_or(1)
            && self.max.is_none_or(|max| value <= max)
            && value % self.multiple.unwrap_or(1).max(1) == 0
            && (!self.power_of_two || value.is_power_of_two())
    }
}

/// Whether `text` is a float's bit pattern as a key prints it: `0x`, then
/// lowercase hexadecimal without leading zeros, of at most `bits` bits.
fn admits_bits(bits: u32, text: &str) -> bool {
    let Some(digits) = text.strip_prefix("0x") else {
        return false;
    };
    !digits.is_empty()
        && digits.len() <= (bits / 4) as usize
        && digits
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && (digits == "0" || !digits.starts_with('0'))
}

/// The kernel package a package provides: the architectures it supports and
/// the descriptors it covers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSpec {
    /// The architectures the package supports, such as `["sm_100a"]`: GPU
    /// targets of the Build Server's description (`crate::target`).
    pub architectures: Vec<String>,
    /// The shapes the package covers, one `[[kernel.descriptors]]` table each.
    ///
    /// An operator's key shape varies by variant — `qk_norm_rope` prints
    /// `rotated` only when its rotation is narrowed, `rotary_factors` prints
    /// `axes`, `sections` and `interleaved` only when it is sectioned — and a
    /// package serves an *operator*, not one key shape. A key is covered when
    /// it reads whole against one of the tables, so a backend name keeps
    /// meaning what `prepare` selects by and a deployment's configuration
    /// names, rather than growing a shape vocabulary.
    pub descriptors: Vec<Descriptors>,
}

/// One package's manifest: `package.toml` at the package root.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The operator's name, which the package fills: also the first name of
    /// its module paths and the `crates/ops/kernels/<op>/` directory
    /// (design section 4.2).
    pub op: String,
    /// The backend's name, which the configuration field selects.
    pub backend: String,
    /// The version of the interface the package implements: its operator's
    /// descriptor and argument layout, the scheduling policy's decisions, or
    /// the engine's messages. Versions start at 1 and an interface raises its
    /// own with any change that is not an addition. The Build Server records
    /// it with every module the package builds, and a loader built for
    /// another version refuses the module.
    pub interface_version: u32,
    /// The package's place among its operator's backends: of the packages
    /// that cover a descriptor key, the Build Server selects the one with
    /// the highest priority, after the backend a deployment names (design
    /// section 6.3). Packages of equal priority are ordered by backend name.
    #[serde(default)]
    pub priority: u32,
    /// The toolchain and external packages the sources need.
    pub toolchain: ToolchainDeps,
    /// What the package compiles to.
    pub module: ModuleSpec,
    /// The package's sources.
    pub source: SourceSpec,
    /// The kernel package's architectures, descriptors and selection rules.
    /// A scheduling policy has no kernel package: it names no architecture
    /// and one key, [`SINGLE_KEY`]; neither does the engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<KernelSpec>,
    /// The package revision's parameters and their default values (design section
    /// 5.2). An argument the configuration gives replaces its default.
    #[serde(default)]
    pub parameters: BTreeMap<String, toml::Value>,
    /// The tuning data files, relative to the package root (design section
    /// 5.4). They are part of the revision.
    #[serde(default)]
    pub tuning: Vec<String>,
}

impl Manifest {
    /// The manifest's file name at a package's root.
    pub const FILE: &str = "package.toml";

    /// The manifest and the package root beside it.
    pub fn load(root: &Path) -> Result<Self, ManifestError> {
        let path = root.join(Self::FILE);
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
        if self.interface_version == 0 {
            return Err(ManifestError::Invalid(
                "interface versions start at 1: state the version of the interface the package \
                 implements in `interface_version`"
                    .into(),
            ));
        }
        match (&self.kernel, self.module.form) {
            (Some(_), ModuleForm::Wasm) => {
                return Err(ManifestError::Invalid(
                    "a scheduling policy fills a whole interface, so it names no kernel package"
                        .into(),
                ));
            }
            (Some(kernel), _) if kernel.architectures.is_empty() => {
                return Err(ManifestError::Invalid(
                    "the package must name the architectures it supports".into(),
                ));
            }
            (Some(kernel), _) => {
                let unknown: Vec<&String> = kernel
                    .architectures
                    .iter()
                    .filter(|architecture| {
                        crate::target::Target::new(architecture.as_str())
                            .facts()
                            .is_none()
                    })
                    .collect();
                if !unknown.is_empty() {
                    return Err(ManifestError::Invalid(format!(
                        "the package names architectures the Build Server does not describe: \
                         {unknown:?}"
                    )));
                }
            }
            (None, ModuleForm::Wasm | ModuleForm::Executable) => {}
            (None, _) => {
                return Err(ManifestError::Invalid(
                    "only a package that fills a whole interface with one module, a wasm policy \
                     or an executable, may omit its [kernel] section"
                        .into(),
                ));
            }
        }
        if !self.source.kind.toolchains().contains(&self.toolchain.kind) {
            return Err(ManifestError::Invalid(format!(
                "a {:?} source builds with {:?}, not with the {} toolchain",
                self.source.kind,
                self.source.kind.toolchains(),
                self.toolchain.kind
            )));
        }
        if self.source.kind.builds_with_cargo()
            && (!self.source.files.is_empty() || !self.tuning.is_empty())
        {
            return Err(ManifestError::Invalid(
                "a crate's files are the ones cargo packages of its workspace: a cargo or wasm \
                 source names no [source] files and no tuning"
                    .into(),
            ));
        }
        if self.source.kind.compiled_by_module() && self.source.script.is_none() {
            return Err(ManifestError::Invalid(format!(
                "a {:?} source is compiled by its build module: name it in [source] script",
                self.source.kind
            )));
        }
        if !self.source.kind.compiled_by_module() && self.source.script.is_some() {
            return Err(ManifestError::Invalid(format!(
                "a {:?} source has no build module: it names no [source] script",
                self.source.kind
            )));
        }
        if self.source.kind == SourceKind::CuteDsl && self.source.unit.is_none() {
            return Err(ManifestError::Invalid(
                "a CuTe DSL source names the C++ unit that defines its C entry in [source] unit"
                    .into(),
            ));
        }
        if self.source.kind == SourceKind::Cubin
            && (self.source.cubin.is_none() || self.source.metadata.is_none())
        {
            return Err(ManifestError::Invalid(
                "a vendor cubin source names its cubin and its launch metadata in [source] cubin \
                 and metadata"
                    .into(),
            ));
        }
        if self.source.kind == SourceKind::Cargo {
            if self.module.form != ModuleForm::Executable || self.kernel.is_some() {
                return Err(ManifestError::Invalid(
                    "a cargo source builds one executable for the host's platform: its module's \
                     form is `executable` and it names no [kernel]"
                        .into(),
                ));
            }
            if self.source.binary.as_deref().is_none_or(str::is_empty) {
                return Err(ManifestError::Invalid(
                    "a cargo source names the executable it builds in [source] binary".into(),
                ));
            }
        }
        for shape in self.descriptors() {
            for (name, dimension) in &shape.dimensions {
                if let Dimension::Constraint(constraint) = dimension
                    && let Some(reason) = constraint.invalid()
                {
                    return Err(ManifestError::Invalid(format!(
                        "the descriptor dimension {name}: {reason}"
                    )));
                }
            }
        }
        let missing: Vec<&String> = self
            .source
            .files
            .iter()
            .chain(self.source.script.iter())
            .chain(self.source.unit.iter())
            .chain(self.source.cubin.iter())
            .chain(self.source.metadata.iter())
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
        match self.module.form {
            ModuleForm::Native => {
                if !self.module.entries.contains_key(LAUNCH_ENTRY) {
                    return Err(ManifestError::Invalid(format!(
                        "a native module must name its {LAUNCH_ENTRY} entry, and its other \
                         lifecycle entries, in [module.entries]"
                    )));
                }
            }
            ModuleForm::Cubin | ModuleForm::Wasm | ModuleForm::Executable => {
                // A cubin's kernel symbol and parameters are in its launch metadata. A package that
                // does not name that file serves an image the main process cannot find the kernel
                // of, which reads as a missing symbol on the first load.
                if self.module.form == ModuleForm::Cubin && self.module.metadata.is_none() {
                    return Err(ManifestError::Invalid(
                        "a cubin module must name its metadata file in [module] metadata".into(),
                    ));
                }
                if self.module.entries_file.is_some() {
                    return Err(ManifestError::Invalid(
                        "only a native module has C entries; a cubin or wasm module has none, and \
                         no entries file"
                            .into(),
                    ));
                }
                if !self.module.entries.is_empty() {
                    return Err(ManifestError::Invalid(
                        "only a native module has C entries; a cubin or wasm module has none"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The descriptor shapes this package covers, in the manifest's order: one
    /// table for a package whose keys all have one shape, one per shape for a
    /// package whose operator's key varies by variant, and none for a policy.
    pub fn descriptors(&self) -> &[Descriptors] {
        match &self.kernel {
            Some(kernel) => &kernel.descriptors,
            None => &[],
        }
    }

    /// Whether this package covers a descriptor key: a kernel package by its
    /// declared dimensions, a package of one module by [`SINGLE_KEY`].
    pub fn covers(&self, key: &str) -> bool {
        self.fields(key).is_some()
    }

    /// The fields of the descriptor whose key this is, as the package's
    /// declared dimensions read it ([`Descriptors::fields`]): what a build
    /// script is handed in place of the key's text. `None` for a key the
    /// package does not cover; a package of one module has no field.
    pub fn fields(&self, key: &str) -> Option<Vec<(String, String)>> {
        match &self.kernel {
            Some(kernel) => kernel
                .descriptors
                .iter()
                .find_map(|shape| shape.fields(key)),
            None => (key == SINGLE_KEY).then(Vec::new),
        }
    }

    /// Whether this package builds for a target: a kernel package by its
    /// declared architectures, a package of one module for its
    /// [`Self::single_target`] alone.
    pub fn builds_for(&self, target: &str) -> bool {
        match &self.kernel {
            Some(kernel) => kernel
                .architectures
                .iter()
                .any(|architecture| architecture == target),
            None => target == self.single_target().as_str(),
        }
    }

    /// The one target a package without a `[kernel]` section builds for: what
    /// its module form runs on, wasm32 for a scheduling policy's module and
    /// this host's Rust target triple for an executable.
    pub fn single_target(&self) -> crate::target::Target {
        match self.module.form {
            ModuleForm::Wasm => crate::target::Target::wasm(),
            _ => crate::target::Target::host(),
        }
    }

    /// The source of this package in `root`, ready to build its descriptors:
    /// the kind the manifest names picks the
    /// [`Source`](crate::source::Source) implementation.
    pub fn source(&self, root: &Path) -> Box<dyn crate::source::Source> {
        crate::source::open(self, root)
    }
}
