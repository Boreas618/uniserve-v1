//! A package's source: how its files become a module (design sections 5.2,
//! 5.3 and 6.2).
//!
//! The Build Server drives seven kinds of source:
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
//! - [`WasmSource`]: a scheduling policy, a crate built for wasm32 from its
//!   snapshot and precompiled for this host's CPU.
//! - [`CargoSource`]: a crate cargo builds to an executable from its
//!   snapshot, the engine.
//! - [`ExecutableSource`]: an executable, stored as it is.
//!
//! A source builds one descriptor into a [`Workspace`] and returns the names
//! of the files it wrote; the store publishes them (design section 5.3). The
//! descriptor key is passed through untouched: the package's compile script
//! reads the parameter values out of it and picks the kernel.

use std::collections::BTreeMap;
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

/// A package's source: it builds one descriptor into a module.
pub trait Source: Send + Sync {
    /// The toolchain this source compiles with.
    fn toolchain(&self) -> ToolchainKind;

    /// Builds the module of one descriptor for one target.
    ///
    /// `op`, `key` and `target` are three names of the module's path (design
    /// section 5.3): the operator, the descriptor key and what the module is
    /// compiled for — an architecture such as `sm_100a`, or a GPU model for a
    /// package whose code reads the device. `dir` is the workspace to write
    /// into. The names of the files it wrote, relative to the workspace, and
    /// the module's form from the manifest are what the store publishes.
    fn build(
        &self,
        op: &str,
        key: &str,
        target: &Target,
        toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Vec<String>>;
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
        SourceKind::Cargo => Box::new(CargoSource::new(package, manifest.clone())),
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

    /// The environment the Build Server passes to a compile script: the
    /// package root, the descriptor key and its fields, the target, the
    /// output directory and the toolchain's own variables.
    ///
    /// `UNISERVE_FIELDS` is the key as the manifest's declared dimensions read
    /// it ([`Manifest::fields`]): a JSON object of dimension to value, in the
    /// key's order. A script takes its descriptor's values from there and
    /// never reads the key's text, so the key grammar is read in one place.
    pub fn environment(
        &self,
        key: &str,
        target: &Target,
        dir: &Workspace,
    ) -> Vec<(OsString, OsString)> {
        let fields: serde_json::Map<String, serde_json::Value> = self
            .manifest
            .fields(key)
            .unwrap_or_default()
            .into_iter()
            .map(|(dimension, value)| (dimension, serde_json::Value::String(value)))
            .collect();
        vec![
            (
                "UNISERVE_FIELDS".into(),
                serde_json::Value::Object(fields).to_string().into(),
            ),
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
    /// The metadata file's name for one descriptor: what the package writes
    /// and what the module carries (design section 5.3).
    pub fn metadata_name(&self, descriptor: &str) -> String {
        self.manifest
            .module
            .metadata_name(descriptor)
            .unwrap_or_else(|| "metadata.json".to_owned())
    }

    /// Checks that a build wrote the module its manifest's form requires: a
    /// cubin with metadata naming the kernel to launch, or a native library.
    pub fn check_module(
        &self,
        key: &str,
        target: &Target,
        dir: &Workspace,
        log: &Path,
    ) -> Result<()> {
        match self.manifest.module.form {
            ModuleForm::Cubin => {
                if !dir.path().join(MODULE_CUBIN).is_file() {
                    return Err(self.failure(
                        key,
                        target,
                        format!(
                            "the build wrote no {MODULE_CUBIN}; its log was:\n{}",
                            read_log(log)
                        ),
                    ));
                }
                let metadata = self.metadata_name(key);
                let path = dir.path().join(&metadata);
                if !path.is_file() {
                    return Err(self.failure(
                        key,
                        target,
                        format!(
                            "the build wrote no {metadata}, the launch metadata a cubin module \
                             needs"
                        ),
                    ));
                }
                check_metadata_symbol(&self.manifest.op, &path).map_err(|message| {
                    self.failure(
                        key,
                        target,
                        format!("{metadata} names no kernel to launch: {message}"),
                    )
                })
            }
            ModuleForm::Native => {
                if !dir.path().join(MODULE_SO).is_file() {
                    return Err(self.failure(
                        key,
                        target,
                        format!(
                            "the build wrote no {MODULE_SO}; its log was:\n{}",
                            read_log(log)
                        ),
                    ));
                }
                Ok(())
            }
            other => Err(self.failure(
                key,
                target,
                format!(
                    "a {:?} source writes a cubin or a native module, not {other:?}",
                    self.manifest.source.kind
                ),
            )),
        }
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
    pub fn failure_with_log(&self, key: &str, target: &Target, error: Error, log: &Path) -> Error {
        let cause = match error {
            Error::Build { message, .. } => message,
            other => other.to_string(),
        };
        self.failure(
            key,
            target,
            format!(
                "{cause}\nthe build log {}:\n{}",
                log.display(),
                read_log(log)
            ),
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

/// A string parameter of a package, or a value the manifest states some other
/// way (a number, a boolean).
fn parameter(parameters: &BTreeMap<String, toml::Value>, name: &str) -> Option<String> {
    parameters.get(name).map(|value| match value {
        toml::Value::String(text) => text.clone(),
        other => other.to_string(),
    })
}

/// A list parameter of a package, each item as a string.
fn list(parameters: &BTreeMap<String, toml::Value>, name: &str) -> Option<Vec<String>> {
    parameters.get(name).and_then(|value| {
        value.as_array().map(|items| {
            items
                .iter()
                .map(|item| match item {
                    toml::Value::String(text) => text.clone(),
                    other => other.to_string(),
                })
                .collect()
        })
    })
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
/// Returns the names it copied, relative to the workspace.
fn copy_tuning(
    package: &PackageSource,
    key: &str,
    target: &Target,
    dir: &Workspace,
) -> Result<Vec<String>> {
    let mut copied = Vec::new();
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
        copied.push(relative);
    }
    Ok(copied)
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
        key: &str,
        target: &Target,
        toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Vec<String>> {
        let package = &self.package;
        let log = build_log(dir);
        // A package may state its command line in its own compile script, for
        // flags the manifest's parameters do not express; otherwise the
        // Build Server runs nvcc itself (design section 6.2).
        match package.manifest.source.script.as_deref() {
            Some(script) => {
                let script = package.file(script);
                if !script.is_file() {
                    return Err(package.failure(
                        key,
                        target,
                        format!("the compile script {} does not exist", script.display()),
                    ));
                }
                let interpreter =
                    toolchain
                        .script_interpreter()
                        .ok_or_else(|| Error::Toolchain {
                            kind: ToolchainKind::Nvcc,
                            message: format!(
                                "no Python interpreter to run {}: pass --python",
                                script.display()
                            ),
                        })?;
                // The script compiles with the nvcc this toolchain checked against the
                // package's declared version, never one it looks up itself.
                let nvcc = toolchain.executable().ok_or_else(|| Error::Toolchain {
                    kind: ToolchainKind::Nvcc,
                    message: "no nvcc named: pass --nvcc <path> or set CUDA_HOME".into(),
                })?;
                let job = ToolJob::new(interpreter, &package.root, &log)
                    .argument(&script)
                    .environment(package.environment(key, target, dir))
                    .environment([("UNISERVE_NVCC".into(), nvcc.as_os_str().to_owned())]);
                toolchain
                    .run(job)
                    .map_err(|error| package.failure_with_log(key, target, error, &log))?;
            }
            None => run_nvcc(package, key, target, toolchain, dir, &log)?,
        }
        package.check_module(key, target, dir, &log)?;
        copy_tuning(package, key, target, dir)?;
        collect(dir.path())
    }
}

/// Runs nvcc over the package's CUDA C++ for one descriptor, writing the
/// module its form names.
///
/// The parameters are the ones a CUDA package states for its translation
/// unit: `standard` (`-std=c++20`), `optimization` (passed through, such as
/// `-O3`), `defines` (`NAME=VALUE`), `include` (directories under the package
/// root) and `libraries` (link options). A cubin module also states its launch
/// metadata in `[kernel.selection]`: `name`, `num_warps`, and optionally
/// `shared`, `num_ctas`, `launch_pdl`, `launch_cooperative_grid`,
/// `global_scratch_size` and `global_scratch_align`, in Triton's field names
/// (design section 5.3).
fn run_nvcc(
    package: &PackageSource,
    key: &str,
    target: &Target,
    toolchain: &dyn Toolchain,
    dir: &Workspace,
    log: &Path,
) -> Result<()> {
    let nvcc = toolchain.executable().ok_or_else(|| Error::Toolchain {
        kind: ToolchainKind::Nvcc,
        message: "no nvcc named: pass --nvcc <path> or set CUDA_HOME".into(),
    })?;
    let parameters = &package.manifest.parameters;
    let mut job = ToolJob::new(nvcc, &package.root, log).argument(format!("-arch={target}"));
    if let Some(standard) = parameter(parameters, "standard") {
        job = job.argument(format!("-std={standard}"));
    }
    if let Some(optimization) = parameter(parameters, "optimization") {
        for flag in optimization.split_whitespace() {
            job = job.argument(flag);
        }
    }
    if let Some(defines) = list(parameters, "defines") {
        for define in defines {
            job = job.argument(format!("-D{define}"));
        }
    }
    job = job.argument("-I").argument(package.root.join("src"));
    if let Some(includes) = list(parameters, "include") {
        for include in includes {
            job = job.argument("-I").argument(package.root.join(include));
        }
    }
    let output = match package.manifest.module.form {
        ModuleForm::Native => {
            job = job
                .argument("--shared")
                .argument("-Xcompiler")
                .argument("-fPIC");
            if let Some(libraries) = list(parameters, "libraries") {
                for library in libraries {
                    job = job.argument(library);
                }
            }
            dir.path().join(MODULE_SO)
        }
        ModuleForm::Cubin => {
            job = job.argument("--cubin");
            dir.path().join(MODULE_CUBIN)
        }
        other => {
            return Err(package.failure(
                key,
                target,
                format!("a cuda package compiles a cubin or a native module, not {other:?}"),
            ));
        }
    };
    let mut sources = 0;
    for relative in package.manifest.source.files_for(key) {
        if !relative.ends_with(".cu") {
            continue;
        }
        let source = package.file(&relative);
        if !source.is_file() {
            return Err(package.failure(
                key,
                target,
                format!("the package carries no {relative} for this descriptor"),
            ));
        }
        job = job.argument(source);
        sources += 1;
    }
    if sources == 0 {
        return Err(package.failure(key, target, "the package names no .cu file to compile"));
    }
    job = job.argument("-o").argument(&output);
    toolchain
        .run(job)
        .map_err(|error| package.failure_with_log(key, target, error, log))?;
    if package.manifest.module.form == ModuleForm::Cubin {
        write_cuda_metadata(package, key, target, dir)?;
    }
    Ok(())
}

/// Writes the launch metadata of an nvcc-built cubin in Triton's field names
/// (design section 5.3), from the package's `[kernel.selection]`.
fn write_cuda_metadata(
    package: &PackageSource,
    key: &str,
    target: &Target,
    dir: &Workspace,
) -> Result<()> {
    let kernel = package.manifest.kernel.as_ref().ok_or_else(|| {
        package.failure(
            key,
            target,
            "a cubin module needs a [kernel] section with its launch metadata",
        )
    })?;
    let selection = &kernel.selection;
    let name = selection
        .get("name")
        .and_then(toml::Value::as_str)
        .or(package.manifest.source.kernel.as_deref())
        .ok_or_else(|| {
            package.failure(
                key,
                target,
                "a cubin module needs the kernel's name in [kernel.selection] name",
            )
        })?;
    let num_warps = selection
        .get("num_warps")
        .and_then(toml::Value::as_integer)
        .ok_or_else(|| {
            package.failure(
                key,
                target,
                "a cubin module needs its threads per block in [kernel.selection] num_warps",
            )
        })?;
    let integer = |field: &str, fallback: i64| {
        selection
            .get(field)
            .and_then(toml::Value::as_integer)
            .unwrap_or(fallback)
    };
    let flag = |field: &str| {
        selection
            .get(field)
            .and_then(toml::Value::as_bool)
            .unwrap_or(false)
    };
    let metadata = serde_json::json!({
        "name": name,
        "num_warps": num_warps,
        "num_ctas": integer("num_ctas", 1),
        "shared": integer("shared", 0),
        "launch_pdl": flag("launch_pdl"),
        "launch_cooperative_grid": flag("launch_cooperative_grid"),
        "global_scratch_size": integer("global_scratch_size", 0),
        "global_scratch_align": integer("global_scratch_align", 1),
    });
    let path = dir.path().join(package.metadata_name(key));
    let text = serde_json::to_vec_pretty(&metadata)
        .map_err(|error| package.failure(key, target, format!("writing metadata: {error}")))?;
    std::fs::write(path, text)?;
    Ok(())
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
        ToolchainKind::Triton
    }

    fn build(
        &self,
        _op: &str,
        key: &str,
        target: &Target,
        toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Vec<String>> {
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
        let interpreter = toolchain
            .script_interpreter()
            .ok_or_else(|| Error::Toolchain {
                kind: ToolchainKind::Triton,
                message: format!(
                    "no Python interpreter to run {}: pass --python",
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
        package.check_module(key, target, dir, &log)?;
        copy_tuning(package, key, target, dir)?;
        let files = collect(dir.path())?;
        Ok(files)
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
        ToolchainKind::CuteDsl
    }

    fn build(
        &self,
        _op: &str,
        key: &str,
        target: &Target,
        toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Vec<String>> {
        let package = &self.package;
        let log = build_log(dir);
        // A CuTe DSL kernel is a Python kernel: it is traced, exported to a
        // host launcher and linked with the runtime archive into a native
        // library by the package's own script, because the kernel's argument
        // types are the package's knowledge (design sections 5.2 and 5.3).
        let script = package.script().map_err(|_| {
            package.failure(
                key,
                target,
                "a CuTe DSL package must name the compile script that builds it in [source] \
                 script",
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
        let interpreter = toolchain
            .script_interpreter()
            .ok_or_else(|| Error::Toolchain {
                kind: ToolchainKind::CuteDsl,
                message: format!(
                    "no Python interpreter to run {}: pass --python",
                    script.display()
                ),
            })?;
        let job = ToolJob::new(interpreter, &package.root, &log)
            .argument(&script)
            .environment(package.environment(key, target, dir));
        toolchain
            .run(job)
            .map_err(|error| package.failure_with_log(key, target, error, &log))?;
        package.check_module(key, target, dir, &log)?;
        copy_tuning(package, key, target, dir)?;
        collect(dir.path())
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
        // A vendor cubin is stored as it is: nothing compiles it.
        ToolchainKind::None
    }

    fn build(
        &self,
        _op: &str,
        key: &str,
        target: &Target,
        _toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Vec<String>> {
        let package = &self.package;
        // A vendor package carries one cubin and one metadata JSON per
        // descriptor (their names hold `{descriptor}`); copying them is the
        // whole build, and it needs no toolchain at all.
        let mut copied_cubin = false;
        let mut copied_metadata = false;
        let metadata = package.metadata_name(key);
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
        collect(dir.path())
    }
}

/// A scheduling policy: a crate built for wasm32 from the snapshot of its
/// workspace, then precompiled for this host's CPU (design section 5.3).
///
/// The installed revision holds the policy's crate and the workspace crates
/// it depends on, with the lockfile ([`crate::snapshot`]). The build is the
/// server's own: cargo builds the crate there for `wasm32-unknown-unknown`,
/// locked and offline, and the scheduling precompiler turns the module into
/// the serialized form the scheduler's wasmtime loads. Nothing is built where
/// the package was installed from, so a revision builds on any host that
/// mounts its package root and has the toolchain.
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
        key: &str,
        target: &Target,
        toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Vec<String>> {
        let package = &self.package;
        let log = build_log(dir);
        let missing = |what: &str| Error::Toolchain {
            kind: ToolchainKind::Wasm,
            message: format!("no {what} on this host: the wasm32 build of a policy needs --cargo"),
        };
        let cargo = toolchain.cargo().ok_or_else(|| missing("cargo"))?;
        let target_directory = toolchain
            .build_directory()
            .ok_or_else(|| missing("cargo target directory"))?;
        let workspace = package.root.join(crate::snapshot::WORKSPACE);
        let artifact = wasm_artifact(&workspace)
            .map_err(|error| package.failure(key, target, error.to_string()))?;

        // The raw module: the snapshot's crate, built for wasm32.
        let job = ToolJob::new(cargo, &workspace, &log)
            .argument("build")
            .argument("--release")
            .argument("--locked")
            .argument("--offline")
            .argument("--target")
            .argument(WASM_TRIPLE)
            .argument("--jobs")
            .argument(toolchain.max_jobs().to_string())
            .argument("--target-dir")
            .argument(target_directory);
        toolchain
            .run(job)
            .map_err(|error| package.failure_with_log(key, target, error, &log))?;
        let raw = target_directory
            .join(WASM_TRIPLE)
            .join("release")
            .join(&artifact);
        if !raw.is_file() {
            return Err(package.failure(
                key,
                target,
                format!(
                    "cargo wrote no {}; its log was:\n{}",
                    raw.display(),
                    read_log(&log)
                ),
            ));
        }

        // Precompile it for this host's CPU: a policy is loaded into the
        // scheduler's wasmtime as a serialized module (design section 5.3).
        let precompiler = toolchain.executable().ok_or_else(|| Error::Toolchain {
            kind: ToolchainKind::Wasm,
            message: "no wasm precompiler on this host: crates/scheduling's \
                      scheduling-precompile builds it"
                .into(),
        })?;
        let output = dir.path().join(MODULE_WASM);
        let job = ToolJob::new(precompiler, &workspace, &log)
            .argument("--wasm")
            .argument(&raw)
            .argument("--out")
            .argument(&output);
        toolchain
            .run(job)
            .map_err(|error| package.failure_with_log(key, target, error, &log))?;
        if !output.is_file() {
            return Err(package.failure(
                key,
                target,
                format!(
                    "the precompiler wrote no {MODULE_WASM}; its log was:\n{}",
                    read_log(&log)
                ),
            ));
        }
        let mut files = vec![MODULE_WASM.to_owned()];
        files.extend(copy_tuning(package, key, target, dir)?);
        Ok(files)
    }
}

/// The target triple a policy crate is built for.
const WASM_TRIPLE: &str = "wasm32-unknown-unknown";

/// The file cargo names the wasm32 module of a snapshot's own crate: the
/// library's name, which is the package's with `_` for `-` unless the crate
/// states another, and `.wasm`.
///
/// The snapshot's workspace manifest names the crate as the one cargo builds
/// by default, and cargo's target directory holds the modules of every policy
/// it built, so the name is read rather than searched for.
fn wasm_artifact(workspace: &Path) -> Result<String> {
    let invalid = |message: String| {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            message,
        ))
    };
    let read = |path: &Path| -> Result<toml::Table> {
        let text = std::fs::read_to_string(path)?;
        toml::from_str(&text).map_err(|error| invalid(format!("{}: {error}", path.display())))
    };

    let manifest = read(&workspace.join("Cargo.toml"))?;
    let member = manifest
        .get("workspace")
        .and_then(|workspace| {
            workspace
                .get("default-members")?
                .as_array()?
                .first()?
                .as_str()
        })
        .ok_or_else(|| invalid("the snapshot's workspace names no crate to build".to_owned()))?;
    let member = read(&workspace.join(member).join("Cargo.toml"))?;
    let name = member
        .get("lib")
        .and_then(|lib| lib.get("name")?.as_str())
        .or_else(|| member.get("package")?.get("name")?.as_str())
        .ok_or_else(|| invalid("the policy's crate states no name".to_owned()))?;
    Ok(format!("{}.wasm", name.replace('-', "_")))
}

/// An executable, stored as it is (design section 5.1).
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
        ToolchainKind::None
    }

    fn build(
        &self,
        _op: &str,
        key: &str,
        target: &Target,
        _toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Vec<String>> {
        let package = &self.package;
        // An executable is stored as it is (design section 5.1): the
        // package revision was built with its own toolchain, outside the Build
        // Server, and this source only publishes the bytes under the
        // revision's path.
        for relative in package.manifest.source.files_for(key) {
            let source = package.file(&relative);
            if !source.is_file() {
                return Err(package.failure(
                    key,
                    target,
                    format!("the package carries no {relative}"),
                ));
            }
            copy_file(&source, &dir.path().join(&relative))?;
        }
        copy_tuning(package, key, target, dir)?;
        collect(dir.path())
    }
}

/// A crate cargo builds to an executable from the snapshot of its workspace:
/// the engine.
///
/// The installed revision holds the crate and every workspace crate it depends
/// on under [`crate::snapshot::WORKSPACE`], with the lockfile
/// ([`lock_snapshot`]). The build is `cargo build --release --locked
/// --offline` there: no crate is fetched and no version is chosen, so the
/// executable is a function of the revision and the toolchain. What cargo
/// compiled stays in the toolchain's build directory, where the next revision
/// finds the dependencies it shares.
#[derive(Clone, Debug)]
pub struct CargoSource {
    package: PackageSource,
}

impl CargoSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for CargoSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Cargo
    }

    fn build(
        &self,
        _op: &str,
        key: &str,
        target: &Target,
        toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Vec<String>> {
        let package = &self.package;
        let log = build_log(dir);
        let missing = |what: &str| Error::Toolchain {
            kind: ToolchainKind::Cargo,
            message: format!("no {what} on this host: pass --cargo <path>"),
        };
        let cargo = toolchain.executable().ok_or_else(|| missing("cargo"))?;
        let target_directory = toolchain
            .build_directory()
            .ok_or_else(|| missing("cargo target directory"))?;
        let binary = package.manifest.source.binary.as_deref().ok_or_else(|| {
            package.failure(key, target, "the package names no [source] binary to build")
        })?;

        let job = ToolJob::new(cargo, &package.root.join(crate::snapshot::WORKSPACE), &log)
            .argument("build")
            .argument("--release")
            .argument("--locked")
            .argument("--offline")
            .argument("--bin")
            .argument(binary)
            .argument("--jobs")
            .argument(toolchain.max_jobs().to_string())
            .argument("--target-dir")
            .argument(target_directory);
        toolchain
            .run(job)
            .map_err(|error| package.failure_with_log(key, target, error, &log))?;

        let built = target_directory.join("release").join(binary);
        if !built.is_file() {
            return Err(package.failure(
                key,
                target,
                format!(
                    "cargo wrote no {}; its log was:\n{}",
                    built.display(),
                    read_log(&log)
                ),
            ));
        }
        // `fs::copy` keeps the mode, so the module's file is executable as built.
        copy_file(&built, &dir.path().join(binary))?;
        copy_tuning(package, key, target, dir)?;
        collect(dir.path())
    }
}

/// Reduces the lockfile of a cargo package's staged revision to the crates
/// the snapshot holds, so the revision builds with `--locked`.
///
/// The snapshot took its workspace's lockfile, which also locks what the
/// crates outside the snapshot depend on. cargo rewrites it while resolving
/// the snapshot offline: every crate the snapshot needs keeps the version the
/// lockfile gave it, and the rest are dropped.
///
/// # Errors
///
/// Fails when the snapshot does not resolve offline: a dependency the
/// lockfile does not hold, or one the toolchain's crate cache lacks.
pub fn lock_snapshot(cargo: &dyn Toolchain, staged: &Path) -> Result<()> {
    let executable = cargo.executable().ok_or_else(|| Error::Toolchain {
        kind: ToolchainKind::Cargo,
        message: "no cargo on this host: pass --cargo <path>".into(),
    })?;
    let log = tempfile::NamedTempFile::new()?;
    let job = ToolJob::new(
        executable,
        &staged.join(crate::snapshot::WORKSPACE),
        log.path(),
    )
    .argument("metadata")
    .argument("--offline")
    .argument("--format-version")
    .argument("1");
    cargo.run(job)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toolchain::{ToolchainConfig, Toolchains};

    /// The toolchains of a host that has none: the `None` toolchain always
    /// exists, so a source that compiles nothing can be built anywhere.
    fn nothing() -> ToolchainConfig {
        ToolchainConfig {
            nvcc: None,
            python: Some(PathBuf::from("/nonexistent/python")),
            wasm: None,
            cargo: None,
            max_jobs: 1,
        }
    }

    /// A vendor package carrying one cubin and one metadata JSON per
    /// descriptor, as `CubinSource` reads it (design 5.3).
    fn vendor_package(directory: &Path, metadata: Option<&str>) {
        std::fs::create_dir_all(directory.join("cubin")).unwrap();
        std::fs::write(directory.join("cubin/dtype_bf16.cubin"), b"a cubin").unwrap();
        // A package without launch metadata carries no metadata JSON at all;
        // one that names the metadata file carries a JSON per descriptor.
        let files = if metadata.is_some() {
            std::fs::write(
                directory.join("cubin/dtype_bf16.json"),
                br#"{"name": "vendor"}"#,
            )
            .unwrap();
            r#"["cubin/{descriptor}.cubin", "cubin/{descriptor}.json"]"#
        } else {
            r#"["cubin/{descriptor}.cubin"]"#
        };
        let metadata = metadata.map_or(String::new(), |name| format!("metadata = \"{name}\"\n"));
        std::fs::write(
            directory.join("package.toml"),
            format!(
                r#"
op = "vendor"
backend = "cubin"
interface_version = 1

[toolchain]
kind = "none"
version = ""

[module]
form = "cubin"
{metadata}
[source]
kind = "cubin"
files = {files}

[kernel]
architectures = ["sm_100a"]

[kernel.descriptors]
dimensions = {{ dtype = ["bf16"] }}
"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn a_vendor_cubin_is_stored_under_the_names_ops_loads_it_by() {
        let directory = tempfile::tempdir().unwrap();
        vendor_package(directory.path(), Some("launch.json"));
        let manifest = Manifest::load(directory.path()).unwrap();
        let source = CubinSource::new(directory.path().to_path_buf(), manifest);
        let workspace = Workspace::new().unwrap();
        let toolchains = Toolchains::discover(&nothing()).unwrap();

        let built = source
            .build(
                "vendor",
                "dtype_bf16",
                &Target::new("sm_100a"),
                toolchains.get(ToolchainKind::None).unwrap(),
                &workspace,
            )
            .unwrap();

        assert_eq!(built, vec!["launch.json", "module.cubin"]);
        assert_eq!(
            std::fs::read(workspace.path().join("module.cubin")).unwrap(),
            b"a cubin"
        );
        assert_eq!(
            std::fs::read(workspace.path().join("launch.json")).unwrap(),
            br#"{"name": "vendor"}"#
        );
    }

    /// A build's environment states the key's fields as the manifest read them, in the key's
    /// order, so a compile script reads no key.
    #[test]
    fn a_build_is_handed_the_fields_of_its_key() {
        let directory = tempfile::tempdir().unwrap();
        vendor_package(directory.path(), Some("launch.json"));
        let manifest = Manifest::load(directory.path()).unwrap();
        let package = PackageSource::new(directory.path().to_path_buf(), manifest);
        let workspace = Workspace::new().unwrap();

        let environment = package.environment("dtype_bf16", &Target::new("sm_100a"), &workspace);
        let stated = |name: &str| {
            environment
                .iter()
                .find(|(variable, _)| variable == name)
                .map(|(_, value)| value.to_string_lossy().into_owned())
        };
        assert_eq!(
            stated("UNISERVE_FIELDS").as_deref(),
            Some(r#"{"dtype":"bf16"}"#)
        );
        assert_eq!(stated("UNISERVE_DESCRIPTOR").as_deref(), Some("dtype_bf16"));
    }

    #[test]
    fn a_vendor_cubin_without_its_metadata_does_not_build() {
        let directory = tempfile::tempdir().unwrap();
        vendor_package(directory.path(), None);
        let manifest = Manifest::load(directory.path()).unwrap();
        let source = CubinSource::new(directory.path().to_path_buf(), manifest);
        let workspace = Workspace::new().unwrap();
        let toolchains = Toolchains::discover(&nothing()).unwrap();

        let error = source
            .build(
                "vendor",
                "dtype_bf16",
                &Target::new("sm_100a"),
                toolchains.get(ToolchainKind::None).unwrap(),
                &workspace,
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("metadata"),
            "the error names the missing metadata: {error}"
        );
    }

    #[test]
    fn a_triton_package_without_a_compile_script_does_not_build() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("kernel.py"), "").unwrap();
        std::fs::write(
            directory.path().join("package.toml"),
            r#"
op = "rms_norm"
backend = "triton"
interface_version = 1

[toolchain]
kind = "triton"
version = "3.7.1"

[module]
form = "cubin"
metadata = "metadata.json"

[source]
kind = "triton"
files = ["kernel.py"]

[kernel]
architectures = ["sm_100a"]

[kernel.descriptors]
dimensions = { dtype = ["bf16"] }
"#,
        )
        .unwrap();
        let manifest = Manifest::load(directory.path()).unwrap();
        let source = TritonSource::new(directory.path().to_path_buf(), manifest);
        let workspace = Workspace::new().unwrap();
        let toolchains = Toolchains::discover(&nothing()).unwrap();

        let error = source
            .build(
                "rms_norm",
                "dtype_bf16",
                &Target::new("sm_100a"),
                toolchains.get(ToolchainKind::None).unwrap(),
                &workspace,
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("compile script"),
            "the error says a Triton package needs its compile script: {error}"
        );
    }
}
