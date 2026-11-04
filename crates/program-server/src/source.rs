//! A package's source: how its files become a module (design sections 5.2,
//! 5.3 and 6.2).
//!
//! The Program Server drives six kinds of source:
//!
//! - [`CudaSource`]: CUDA C++ through nvcc, to a cubin (with the launch
//!   metadata the package writes in Triton's fields) or, when the package
//!   brings a C ABI layer, to a native library.
//! - [`TritonSource`]: Triton through `triton.compile`, which writes a cubin
//!   and its metadata JSON. Triton's own AOT tool cannot express clusters,
//!   PDL or scratch, so it is not used (design section 5.3).
//! - [`CuteDslSource`]: CuTe DSL through its Python compiler and
//!   `export_to_c`, linked with `libcuda_dialect_runtime_static.a` into a
//!   native library.
//! - [`CubinSource`]: a vendor cubin the package carries, such as
//!   trtllm-gen's launch code and cubins, stored as it is.
//! - [`WasmSource`]: a scheduling policy built for wasm32 and precompiled for
//!   this host's CPU.
//! - [`ExecutableSource`]: an executable program, stored as it is.
//!
//! A source builds one descriptor into a [`Workspace`] and returns the names
//! of the files it wrote; the store publishes them (design section 5.3). The
//! descriptor key is passed through untouched: the package's compile script
//! reads the parameter values out of it and picks the kernel.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::manifest::{Manifest, ModuleForm, SourceKind, ToolchainKind};
use crate::target::Target;
use crate::toolchain::{ToolJob, Toolchain};
use crate::{Error, Result};

/// The name `ops` loads a cubin module by (design section 4.2).
pub const MODULE_CUBIN: &str = "module.cubin";

/// The name `ops` loads a native module by (design section 4.2).
pub const MODULE_SO: &str = "module.so";

/// The name the store keeps a precompiled wasm module under; the scheduling
/// policies are loaded from it (design section 5.3).
pub const MODULE_WASM: &str = "module.wasm";

/// The file a build's compiler writes its output to, inside the build's
/// workspace. It is not part of the module; [`collect`] leaves it out.
const BUILD_LOG: &str = ".build.log";

/// The files a build wrote, relative to its [`Workspace`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Built {
    pub files: Vec<String>,
}

/// The directory one build writes into. It is private to the build and is
/// deleted when the build ends, after the store has published the module.
#[derive(Debug)]
pub struct Workspace {
    directory: tempfile::TempDir,
}

impl Workspace {
    pub fn new() -> crate::Result<Workspace> {
        Ok(Workspace {
            directory: tempfile::tempdir()?,
        })
    }

    pub fn path(&self) -> &Path {
        self.directory.path()
    }
}

/// A program package's source: it builds one descriptor into a module.
pub trait Source: Send + Sync {
    /// The toolchain this source compiles with.
    fn toolchain(&self) -> ToolchainKind;

    /// Builds the module of one descriptor for one target.
    ///
    /// `op`, `key` and `target` are three names of the module's path (design
    /// section 5.3): the operator, the descriptor key and what the module is
    /// compiled for — an architecture such as `sm_100a`, or a GPU model for a
    /// package whose code reads the device. `dir` is the workspace to write
    /// into. The returned file names, and the module's form from the
    /// manifest, are what the store publishes.
    fn build(
        &self,
        op: &str,
        key: &str,
        target: &Target,
        toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Built>;
}

/// Makes the source a package's manifest names.
pub fn open(manifest: &Manifest, root: &Path) -> Result<Box<dyn Source>> {
    let package = root.to_path_buf();
    Ok(match manifest.source.kind {
        SourceKind::Cuda => Box::new(CudaSource::new(package, manifest.clone())),
        SourceKind::Triton => Box::new(TritonSource::new(package, manifest.clone())),
        SourceKind::CuteDsl => Box::new(CuteDslSource::new(package, manifest.clone())),
        SourceKind::Cubin => Box::new(CubinSource::new(package, manifest.clone())),
        SourceKind::Wasm => Box::new(WasmSource::new(package, manifest.clone())),
        SourceKind::Executable => Box::new(ExecutableSource::new(package, manifest.clone())),
    })
}

/// What every source shares: where the package is and what it declares.
///
/// A source is built for exactly one package revision, so the manifest it
/// carries is fixed for its whole life.
#[derive(Clone, Debug)]
pub struct PackageSource {
    pub root: PathBuf,
    pub manifest: Manifest,
}

impl PackageSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self { root, manifest }
    }

    /// The module form the package's manifest names.
    pub fn form(&self) -> ModuleForm {
        self.manifest.module.form
    }

    /// A file of the package, by the manifest's relative path.
    pub fn file(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }

    /// The compile script the package names, or fails.
    pub fn script(&self) -> Result<PathBuf> {
        let script = self
            .manifest
            .source
            .script
            .as_deref()
            .ok_or_else(|| Error::Build {
                op: self.manifest.op.clone(),
                backend: self.manifest.backend.clone(),
                revision: 0,
                target: String::new(),
                descriptor: String::new(),
                message: format!(
                    "a {:?} package needs a compile script",
                    self.manifest.source.kind
                ),
            })?;
        Ok(self.file(script))
    }

    /// The environment the Program Server passes to a compile script: the
    /// package root, the descriptor key, the output directory and the
    /// toolchain's own variables.
    pub fn environment(
        &self,
        key: &str,
        target: &Target,
        dir: &Workspace,
    ) -> Vec<(OsString, OsString)> {
        vec![
            ("UNISERVE_OP".into(), self.manifest.op.clone().into()),
            (
                "UNISERVE_BACKEND".into(),
                self.manifest.backend.clone().into(),
            ),
            ("UNISERVE_DESCRIPTOR".into(), key.into()),
            ("UNISERVE_TARGET".into(), target.to_string().into()),
            (
                "UNISERVE_PACKAGE".into(),
                self.root.clone().into_os_string(),
            ),
            ("UNISERVE_OUT".into(), dir.path().as_os_str().to_owned()),
        ]
    }

    /// The metadata JSON's name in the module directory: the name the
    /// manifest gives, or Triton's own `metadata.json`.
    pub fn metadata_name(&self) -> String {
        self.manifest
            .module
            .metadata
            .clone()
            .unwrap_or_else(|| "metadata.json".to_owned())
    }

    /// A failed build of this package, with the package's identity in the
    /// error so a log line names what was being built.
    pub fn failure(&self, key: &str, target: &Target, message: impl Into<String>) -> Error {
        Error::Build {
            op: self.manifest.op.clone(),
            backend: self.manifest.backend.clone(),
            revision: 0,
            target: target.to_string(),
            descriptor: key.to_owned(),
            message: message.into(),
        }
    }

    /// A failed compiler process, with the package's identity and the
    /// compiler's log.
    pub fn failure_with_log(
        &self,
        key: &str,
        target: &Target,
        error: Error,
        log: &Path,
    ) -> Error {
        let cause = match error {
            Error::Build { message, .. } => message,
            other => other.to_string(),
        };
        self.failure(
            key,
            target,
            format!("{cause}\nthe build log {}:\n{}", log.display(), read_log(log)),
        )
    }
}

/// The file a build's compiler writes its output to.
fn build_log(dir: &Workspace) -> PathBuf {
    dir.path().join(BUILD_LOG)
}

/// The text of a build's log, or a note that it cannot be read.
fn read_log(log: &Path) -> String {
    std::fs::read_to_string(log).unwrap_or_else(|error| format!("<unreadable: {error}>"))
}

/// The files a build wrote into its workspace, relative to the workspace and
/// in a stable order. The compiler's log is not one of them.
fn collect(directory: &Path) -> Result<Vec<String>> {
    fn walk(at: &Path, prefix: &Path, files: &mut Vec<String>) -> Result<()> {
        let mut entries = std::fs::read_dir(at)?.collect::<std::result::Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let name = entry.file_name();
            // A dot file is the build's own bookkeeping (its log), not a
            // module file.
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            let path = entry.path();
            let relative = prefix.join(&name);
            let kind = entry.file_type()?;
            if kind.is_dir() {
                walk(&path, &relative, files)?;
            } else if kind.is_file() {
                files.push(relative.to_string_lossy().replace('\\', "/"));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(directory, Path::new(""), &mut files)?;
    Ok(files)
}

/// Copies one file of a package into the build's workspace, creating the
/// directories its relative path names.
fn copy_file(source: &Path, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(source, destination)?;
    Ok(())
}

/// Copies the package's tuning data into the build's workspace (design section
/// 5.4): tuning data is part of the revision and travels with the module it
/// was measured for, so the main process gets it from the same `resolve`.
fn copy_tuning(
    package: &PackageSource,
    key: &str,
    target: &Target,
    dir: &Workspace,
) -> Result<()> {
    for declared in &package.manifest.tuning {
        let relative = declared.replace(crate::manifest::DESCRIPTOR_PLACEHOLDER, key);
        let source = package.file(&relative);
        if !source.is_file() {
            return Err(package.failure(
                key,
                target,
                format!("the package names tuning data it does not have: {relative}"),
            ));
        }
        copy_file(&source, &dir.path().join(&relative))?;
    }
    Ok(())
}

/// Checks that a cubin module's metadata JSON names the kernel to launch.
///
/// `ops` loads the module with `cuLibraryLoadData` and launches the symbol the
/// metadata's `name` field gives (design sections 4.2 and 5.3), so a metadata
/// file without a name builds a module nothing can launch.
fn check_metadata_symbol(op: &str, metadata: &Path) -> std::result::Result<(), String> {
    let text = std::fs::read_to_string(metadata).map_err(|error| error.to_string())?;
    let parsed: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| error.to_string())?;
    match parsed.get("name").and_then(serde_json::Value::as_str) {
        Some(name) if !name.is_empty() => Ok(()),
        _ => Err(format!("{op}: the metadata has no name field")),
    }
}

/// CUDA C++ through nvcc (design section 5.3).
#[derive(Clone, Debug)]
pub struct CudaSource {
    package: PackageSource,
}

impl CudaSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for CudaSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Nvcc
    }

    fn build(
        &self,
        _op: &str,
        _key: &str,
        _target: &Target,
        _toolchain: &dyn Toolchain,
        _dir: &Workspace,
    ) -> Result<Built> {
        todo!("the nvcc source: compile the package's CUDA C++ into a cubin or a native library")
    }
}

/// Triton through `triton.compile` (design section 5.3).
#[derive(Clone, Debug)]
pub struct TritonSource {
    package: PackageSource,
}

impl TritonSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for TritonSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Python
    }

    fn build(
        &self,
        _op: &str,
        key: &str,
        target: &Target,
        toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Built> {
        let package = &self.package;
        let log = build_log(dir);
        // The package states how its kernel is compiled in its own script: the
        // descriptor's argument signature, the compile-time values the
        // descriptor and the parameters give, and the compilation options are
        // the package's knowledge of its operator, so they live in the
        // revision (design section 5.2), not in the manifest's free-form
        // selection table.
        let script = package.script().map_err(|_| {
            package.failure(
                key,
                target,
                "a Triton package must name the compile script that builds it in [source] script",
            )
        })?;
        if !script.is_file() {
            return Err(package.failure(
                key,
                target,
                format!(
                    "the compile script {} does not exist in the package",
                    script.display()
                ),
            ));
        }
        let interpreter = toolchain.script_interpreter().ok_or_else(|| Error::Toolchain {
            kind: ToolchainKind::Python,
            message: format!(
                "no Python environment to run {}: create crates/program-server/python with `uv sync`",
                script.display()
            ),
        })?;
        let job = ToolJob::new(interpreter, &package.root, &log)
            .argument(&script)
            .environment(package.environment(key, target, dir));
        toolchain
            .run(job)
            .map_err(|error| package.failure_with_log(key, target, error, &log))?;

        // A cubin module is the cubin and the metadata JSON that says how to
        // launch it; a package that wrote neither did not build a module.
        let cubin = dir.path().join(MODULE_CUBIN);
        if !cubin.is_file() {
            return Err(package.failure(
                key,
                target,
                format!(
                    "the compile script wrote no {MODULE_CUBIN}; its output was:\n{}",
                    read_log(&log)
                ),
            ));
        }
        let metadata = package.metadata_name();
        let metadata_path = dir.path().join(&metadata);
        if !metadata_path.is_file() {
            return Err(package.failure(
                key,
                target,
                format!(
                    "the compile script wrote no {metadata}, the launch metadata the module needs"
                ),
            ));
        }
        check_metadata_symbol(&package.manifest.op, &metadata_path).map_err(|message| {
            package.failure(
                key,
                target,
                format!("{metadata} names no kernel to launch: {message}"),
            )
        })?;
        copy_tuning(package, key, target, dir)?;
        let files = collect(dir.path())?;
        Ok(Built { files })
    }
}

/// CuTe DSL through `export_to_c`, linked into a native library (design
/// sections 5.3 and 9).
#[derive(Clone, Debug)]
pub struct CuteDslSource {
    package: PackageSource,
}

impl CuteDslSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for CuteDslSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Python
    }

    fn build(
        &self,
        _op: &str,
        _key: &str,
        _target: &Target,
        _toolchain: &dyn Toolchain,
        _dir: &Workspace,
    ) -> Result<Built> {
        todo!("the CuTe DSL source: export_to_c, then link libcuda_dialect_runtime_static.a")
    }
}

/// A vendor cubin the package carries (design section 5.3).
#[derive(Clone, Debug)]
pub struct CubinSource {
    package: PackageSource,
}

impl CubinSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for CubinSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Nvcc
    }

    fn build(
        &self,
        _op: &str,
        key: &str,
        target: &Target,
        _toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Built> {
        let package = &self.package;
        // A vendor package carries one cubin and one metadata JSON per
        // descriptor (their names hold `{descriptor}`); copying them is the
        // whole build, and it needs no toolchain at all.
        let mut copied_cubin = false;
        let mut copied_metadata = false;
        let metadata = package.metadata_name();
        for relative in package.manifest.source.files_for(key) {
            let source = package.file(&relative);
            if !source.is_file() {
                return Err(package.failure(
                    key,
                    target,
                    format!("the package carries no {relative} for this descriptor"),
                ));
            }
            // The module's own files have the names `ops` loads them by: the
            // cubin is `module.cubin` and the metadata JSON is the name the
            // manifest states. Other files keep their place.
            let name = if relative.ends_with(".cubin") {
                copied_cubin = true;
                MODULE_CUBIN.to_owned()
            } else if relative.ends_with(".json") {
                copied_metadata = true;
                metadata.clone()
            } else {
                relative.clone()
            };
            copy_file(&source, &dir.path().join(&name))?;
        }
        if !copied_cubin {
            return Err(package.failure(
                key,
                target,
                "the package names no .cubin file for this descriptor",
            ));
        }
        if !copied_metadata {
            return Err(package.failure(
                key,
                target,
                "the package names no metadata JSON for this descriptor",
            ));
        }
        check_metadata_symbol(&package.manifest.op, &dir.path().join(&metadata))
            .map_err(|message| package.failure(key, target, message))?;
        copy_tuning(package, key, target, dir)?;
        Ok(Built {
            files: collect(dir.path())?,
        })
    }
}

/// A scheduling policy: wasm32, then precompiled for this host's CPU
/// (design section 5.3).
#[derive(Clone, Debug)]
pub struct WasmSource {
    package: PackageSource,
}

impl WasmSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for WasmSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Wasm
    }

    fn build(
        &self,
        _op: &str,
        _key: &str,
        _target: &Target,
        _toolchain: &dyn Toolchain,
        _dir: &Workspace,
    ) -> Result<Built> {
        todo!("the wasm source: build wasm32, then precompile it with wasmtime")
    }
}

/// An executable program, stored as it is (design section 5.1).
#[derive(Clone, Debug)]
pub struct ExecutableSource {
    package: PackageSource,
}

impl ExecutableSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for ExecutableSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Nvcc
    }

    fn build(
        &self,
        _op: &str,
        _key: &str,
        _target: &Target,
        _toolchain: &dyn Toolchain,
        _dir: &Workspace,
    ) -> Result<Built> {
        todo!("the executable source: store the package's binary as it is")
    }
}
