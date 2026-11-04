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
                let metadata = self.metadata_name();
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
/// Answers the names it copied, relative to the workspace.
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
    ) -> Result<Built> {
        let package = &self.package;
        let log = build_log(dir);
        // A package may state its command line in its own compile script, for
        // flags the manifest's parameters do not express; otherwise the
        // Program Server runs nvcc itself (design section 6.2).
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
                                "no Python environment to run {}: create \
                                 crates/program-server/python with `uv sync`",
                                script.display()
                            ),
                        })?;
                let job = ToolJob::new(interpreter, &package.root, &log)
                    .argument(&script)
                    .environment(package.environment(key, target, dir));
                toolchain
                    .run(job)
                    .map_err(|error| package.failure_with_log(key, target, error, &log))?;
            }
            None => run_nvcc(package, key, target, toolchain, dir, &log)?,
        }
        package.check_module(key, target, dir, &log)?;
        copy_tuning(package, key, target, dir)?;
        Ok(Built {
            files: collect(dir.path())?,
        })
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
        message: "no nvcc on this host: install the CUDA toolkit".into(),
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
    let path = dir.path().join(package.metadata_name());
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
            kind: ToolchainKind::Triton,
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
        package.check_module(key, target, dir, &log)?;
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
        ToolchainKind::CuteDsl
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
        let interpreter = toolchain.script_interpreter().ok_or_else(|| Error::Toolchain {
            kind: ToolchainKind::CuteDsl,
            message: format!(
                "no Python environment to run {}: create crates/program-server/python with \
                 `uv sync`",
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
        Ok(Built {
            files: collect(dir.path())?,
        })
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
        key: &str,
        target: &Target,
        toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Built> {
        let package = &self.package;
        let log = build_log(dir);
        // The raw module: either the package's own build writes it (it knows
        // how its guest crate is assembled), or the package is a crate that
        // `cargo build` compiles for wasm32 as it stands.
        let raw = match package.manifest.source.script.as_deref() {
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
                            kind: ToolchainKind::Wasm,
                            message: format!(
                                "no Python environment to run {}: create \
                                 crates/program-server/python with `uv sync`",
                                script.display()
                            ),
                        })?;
                let job = ToolJob::new(interpreter, &package.root, &log)
                    .argument(&script)
                    .environment(package.environment(key, target, dir));
                toolchain
                    .run(job)
                    .map_err(|error| package.failure_with_log(key, target, error, &log))?;
                find_wasm(dir.path())?
            }
            None => build_wasm_crate(package, key, target, dir, &log, toolchain.max_jobs())?,
        };

        // Precompile it for this host's CPU: a policy is loaded into the
        // scheduler's wasmtime as a serialized module (design section 5.3).
        let precompiler = toolchain.executable().ok_or_else(|| Error::Toolchain {
            kind: ToolchainKind::Wasm,
            message: "no wasm precompiler on this host: crates/scheduling's \
                      scheduling-precompile builds it"
                .into(),
        })?;
        let output = dir.path().join(MODULE_WASM);
        // A script may write the raw module under the output's own name; move
        // it aside so the precompiler does not overwrite its input.
        let raw = if raw == output {
            let aside = dir.path().join("policy.raw.wasm");
            std::fs::rename(&raw, &aside)?;
            aside
        } else {
            raw
        };
        let job = ToolJob::new(precompiler, &package.root, &log)
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
        Ok(Built { files })
    }
}

/// Compiles a policy package that is a crate by itself: `cargo build
/// --release` for wasm32, in the package root, with its own target directory
/// in the build's workspace.
fn build_wasm_crate(
    package: &PackageSource,
    key: &str,
    target: &Target,
    dir: &Workspace,
    log: &Path,
    max_jobs: usize,
) -> Result<PathBuf> {
    let cargo = crate::toolchain::find_program("cargo").ok_or_else(|| Error::Toolchain {
        kind: ToolchainKind::Wasm,
        message: "no cargo on this host: the wasm32 build of a policy needs the Rust toolchain"
            .into(),
    })?;
    let target_directory = dir.path().join("cargo");
    let job = ToolJob::new(&cargo, &package.root, log)
        .argument("build")
        .argument("--release")
        .argument("--target")
        .argument("wasm32-unknown-unknown")
        .argument("--target-dir")
        .argument(&target_directory)
        .environment(package.environment(key, target, dir));
    crate::toolchain::execute(&job, max_jobs)
        .map_err(|error| package.failure_with_log(key, target, error, log))?;
    find_wasm(&target_directory.join("wasm32-unknown-unknown").join("release"))
}

/// The one `.wasm` file a wasm32 build wrote in `directory`.
fn find_wasm(directory: &Path) -> Result<PathBuf> {
    let mut candidates = std::fs::read_dir(directory)
        .map_err(Error::Io)?
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "wasm"))
        .collect::<Vec<_>>();
    // A Cdylib names its output after the crate; a package that writes the
    // raw module under its own name states which one it is.
    candidates.sort();
    match candidates.len() {
        0 => Err(Error::Build {
            op: String::new(),
            backend: String::new(),
            revision: 0,
            target: String::new(),
            descriptor: String::new(),
            message: format!("no .wasm module in {}", directory.display()),
        }),
        1 => Ok(candidates.remove(0)),
        _ => {
            let chosen = candidates
                .iter()
                .find(|path| path.file_name().is_some_and(|name| name == "policy.wasm"));
            chosen.cloned().ok_or_else(|| Error::Build {
                op: String::new(),
                backend: String::new(),
                revision: 0,
                target: String::new(),
                descriptor: String::new(),
                message: format!(
                    "more than one .wasm module in {}: {candidates:?}",
                    directory.display()
                ),
            })
        }
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
        ToolchainKind::None
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
        // An executable program is stored as it is (design section 5.1): the
        // program was built with its own toolchain, outside the Program
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
        Ok(Built {
            files: collect(dir.path())?,
        })
    }
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
            python: PathBuf::from("/nonexistent/python"),
            wasm: None,
            max_jobs: 1,
        }
    }

    /// A vendor package carrying one cubin and one metadata JSON per
    /// descriptor, as `CubinSource` reads it (design 5.3).
    fn vendor_package(directory: &Path, metadata: Option<&str>) {
        std::fs::create_dir_all(directory.join("cubin")).unwrap();
        std::fs::write(directory.join("cubin/dtype_bf16.cubin"), b"a cubin").unwrap();
        std::fs::write(
            directory.join("cubin/dtype_bf16.json"),
            br#"{"name": "vendor"}"#,
        )
        .unwrap();
        let metadata = metadata.map_or(String::new(), |name| {
            format!("metadata = \"{name}\"\n")
        });
        std::fs::write(
            directory.join("package.toml"),
            format!(
                r#"
op = "vendor"
backend = "cubin"

[toolchain]
kind = "none"
version = ""

[module]
form = "cubin"
{metadata}
[source]
kind = "cubin"
files = ["cubin/{{descriptor}}.cubin", "cubin/{{descriptor}}.json"]

[kernel]
architectures = ["sm_100a"]

[kernel.descriptors]
keys = ["dtype_bf16"]
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

        assert_eq!(built.files, vec!["launch.json", "module.cubin"]);
        assert_eq!(
            std::fs::read(workspace.path().join("module.cubin")).unwrap(),
            b"a cubin"
        );
        assert_eq!(
            std::fs::read(workspace.path().join("launch.json")).unwrap(),
            br#"{"name": "vendor"}"#
        );
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
            error.to_string().contains("dtype_bf16.json"),
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
keys = ["dtype_bf16"]
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
