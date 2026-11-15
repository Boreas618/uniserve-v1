//! A package's source: how its files become a module (design sections 5.2,
//! 5.3 and 6.2).
//!
//! The Build Server drives five kinds of source:
//!
//! - [`ModuleSource`]: a **build module**, the Python module a Triton, CuTe DSL
//!   or CUDA package names in `[source] script`. The Build Server compiles one
//!   descriptor by running its Python environment's interpreter as
//!   `python -m uniserve_build_server compile` in the module's directory with a
//!   typed compile request on standard input ([`Request`]): the package, the
//!   descriptor's fields as the manifest reads them, the target with its
//!   hardware facts, the output directory, the job limit, the cache of pinned
//!   sources and the toolchain description the server checked. The request is
//!   the whole interface; a compilation reads no environment variable.
//! - [`CubinSource`]: a vendor cubin the package carries, stored as it is.
//! - [`WasmSource`]: a scheduling policy, a crate built for wasm32 from the
//!   revision's whole-workspace snapshot (`crate::snapshot`).
//! - [`CargoSource`]: a crate cargo builds to an executable from the
//!   revision's whole-workspace snapshot, the engine.
//! - [`ExecutableSource`]: an executable, stored as it is.
//!
//! A source builds one descriptor into a [`Workspace`] and returns the names
//! of the files it wrote; the store publishes them (design section 5.3).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::manifest::{Manifest, ModuleForm, SourceKind};
use crate::target::{Facts, Target};
use crate::toolchain::{CARGO, Report, Toolchains};
use crate::{Error, Result};

/// The name `ops` loads a cubin module by (design section 4.2).
pub const MODULE_CUBIN: &str = "module.cubin";

/// The name `ops` loads a native module by (design section 4.2).
pub const MODULE_SO: &str = "module.so";

/// The name the store keeps a scheduling policy's wasm32 module under (design
/// section 5.3).
pub const MODULE_WASM: &str = "module.wasm";

/// The file a build's compiler writes its output to, inside the build's
/// workspace. A dot file is the build's own bookkeeping: [`collect`] leaves it
/// out of the module.
const BUILD_LOG: &str = ".build.log";

/// The compile request of a build module, inside the build's workspace.
const REQUEST: &str = ".request.json";

/// The JSON messages cargo prints about what it built, inside the build's
/// workspace.
const CARGO_MESSAGES: &str = ".cargo-messages.json";

/// The Build Server's directories of what builds keep between them, under one
/// cache root: the pinned third-party sources, cargo's target directory, the
/// vendored registry sources of each lockfile, and the workspaces builds write
/// into. One root keeps them on one filesystem, so a build's workspace holds a
/// pinned file as a hard link into the cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    pub fn new(root: PathBuf) -> Cache {
        Cache { root }
    }

    /// The cache of pinned third-party sources (`uniserve_build_server.fetch`).
    pub fn pins(&self) -> PathBuf {
        self.root.join("pins")
    }

    /// cargo's target directory, where what one revision compiled is found by
    /// the next.
    pub fn cargo(&self) -> PathBuf {
        self.root.join("cargo")
    }

    /// The provisioning of the lockfile whose SHA-256 is `lock`: its vendored
    /// registry and git sources and the cargo configuration that names them.
    pub fn vendor(&self, lock: &str) -> PathBuf {
        self.root.join("vendor").join(lock)
    }

    /// The directory builds' workspaces are made in.
    pub fn work(&self) -> PathBuf {
        self.root.join("work")
    }
}

/// The directory one build writes into. It is private to the build and is
/// deleted when the build ends, after the store has published the module.
#[derive(Debug)]
pub struct Workspace {
    directory: tempfile::TempDir,
}

impl Workspace {
    /// A new workspace among the cache's ([`Cache::work`]).
    pub fn new(cache: &Cache) -> Result<Workspace> {
        let work = cache.work();
        std::fs::create_dir_all(&work)?;
        Ok(Workspace {
            directory: tempfile::tempdir_in(work)?,
        })
    }

    pub fn path(&self) -> &Path {
        self.directory.path()
    }
}

/// What one build is given: the module it builds and what builds it.
pub struct Build<'a> {
    /// The operator, the first name of the module's path.
    pub op: &'a str,
    /// The descriptor key, the module path's last name.
    pub key: &'a str,
    /// What the module is compiled for.
    pub target: &'a Target,
    pub toolchains: &'a Toolchains,
    pub cache: &'a Cache,
    /// The compiler processes the build may run at once.
    pub jobs: usize,
    pub workspace: &'a Workspace,
}

/// A package's source: it builds one descriptor into a module.
pub trait Source: Send + Sync {
    /// Builds the module of one descriptor for one target, and returns the
    /// names of the files it wrote into the workspace, relative to it.
    fn build(&self, build: &Build<'_>) -> Result<Vec<String>>;
}

/// Makes the source a package's manifest names.
pub fn open(manifest: &Manifest, root: &Path) -> Box<dyn Source> {
    let package = PackageSource {
        root: root.to_path_buf(),
        manifest: manifest.clone(),
    };
    match manifest.source.kind {
        SourceKind::Cuda | SourceKind::Triton | SourceKind::CuteDsl => {
            Box::new(ModuleSource(package))
        }
        SourceKind::Cubin => Box::new(CubinSource(package)),
        SourceKind::Wasm => Box::new(WasmSource(package)),
        SourceKind::Executable => Box::new(ExecutableSource(package)),
        SourceKind::Cargo => Box::new(CargoSource(package)),
    }
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
    /// A file of the package, by the manifest's relative path.
    pub fn file(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }

    /// The metadata file's name in the module, for one descriptor.
    fn metadata_name(&self, key: &str, target: &Target) -> Result<String> {
        self.manifest.module.metadata_name(key).ok_or_else(|| {
            self.failure(key, target, "a cubin module names its metadata in [module]")
        })
    }

    /// Checks that a build wrote the module its manifest's form requires: a
    /// cubin with metadata naming the kernel to launch, or a native library
    /// that defines every C entry it is launched through.
    fn check_module(&self, key: &str, target: &Target, dir: &Workspace, log: &Path) -> Result<()> {
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
                let metadata = self.metadata_name(key, target)?;
                let path = dir.path().join(&metadata);
                if !path.is_file() {
                    return Err(self.failure(
                        key,
                        target,
                        format!("the build wrote no {metadata}, the launch metadata a cubin module needs"),
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
                let module = dir.path().join(MODULE_SO);
                let bytes = std::fs::read(&module).map_err(|error| {
                    self.failure(
                        key,
                        target,
                        format!(
                            "the build wrote no {MODULE_SO} ({error}); its log was:\n{}",
                            read_log(log)
                        ),
                    )
                })?;
                // The main process resolves each entry by name after loading the module
                // (design section 4.2), so a module that does not define one is refused
                // here rather than at its first launch.
                let defined =
                    platform::NativeLibrary::defined_functions(&bytes).ok_or_else(|| {
                        self.failure(
                            key,
                            target,
                            format!("{MODULE_SO} is no shared object with a symbol table"),
                        )
                    })?;
                let entries = entries(&self.manifest, dir.path())?;
                let missing: Vec<&String> = entries
                    .values()
                    .filter(|entry| !defined.iter().any(|name| name == *entry))
                    .collect();
                if missing.is_empty() {
                    Ok(())
                } else {
                    Err(self.failure(
                        key,
                        target,
                        format!("{MODULE_SO} does not define {missing:?}"),
                    ))
                }
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
    fn failure(&self, key: &str, target: &Target, message: impl Into<String>) -> Error {
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
    fn failure_with_log(&self, key: &str, target: &Target, error: Error, log: &Path) -> Error {
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

    /// Copies the package's tuning data into the build's workspace (design
    /// section 5.4): tuning data is part of the revision and travels with the
    /// module it was measured for, so the main process gets it from the same
    /// `resolve`. Returns the names it copied, relative to the workspace.
    fn copy_tuning(&self, key: &str, target: &Target, dir: &Workspace) -> Result<Vec<String>> {
        let mut copied = Vec::new();
        for declared in &self.manifest.tuning {
            let relative = declared.replace(crate::manifest::DESCRIPTOR_PLACEHOLDER, key);
            let source = self.file(&relative);
            if !source.is_file() {
                return Err(self.failure(
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
}

/// The C entries a module is launched through: its package's
/// `[module.entries]`, overridden by each role the module's entries file names
/// (design section 4.2). The file is part of the module, written by the build
/// for this descriptor, so a package whose modules each launch through their
/// own C entry returns the entries of the module in `directory`.
///
/// # Errors
///
/// [`Error::Protocol`] when the manifest names an entries file the module does
/// not hold, or one that is not a JSON object of roles to symbols.
pub fn entries(manifest: &Manifest, directory: &Path) -> Result<BTreeMap<String, String>> {
    let mut entries = manifest.module.entries.clone();
    let Some(name) = manifest.module.entries_file.as_deref() else {
        return Ok(entries);
    };
    let text = std::fs::read_to_string(directory.join(name)).map_err(|error| {
        Error::Protocol(format!(
            "the module names its entries in {name}, which it does not hold: {error}"
        ))
    })?;
    let written: BTreeMap<String, String> = serde_json::from_str(&text).map_err(|error| {
        Error::Protocol(format!(
            "{name} is not a JSON object of C entry roles to symbols: {error}"
        ))
    })?;
    entries.extend(written);
    Ok(entries)
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
/// in a stable order. A dot entry is the build's own bookkeeping (its log, its
/// request, a scratch directory), not a module file.
fn collect(directory: &Path) -> Result<Vec<String>> {
    fn walk(at: &Path, prefix: &Path, files: &mut Vec<String>) -> Result<()> {
        let mut entries = std::fs::read_dir(at)?.collect::<std::result::Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let name = entry.file_name();
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

/// Copies one file into the build's workspace, creating the directories its
/// relative path names. `fs::copy` keeps the mode, so an executable stays one.
fn copy_file(source: &Path, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(source, destination)?;
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

/// The target of a compile request: its name and its hardware facts.
#[derive(Serialize)]
struct RequestTarget<'a> {
    name: &'a str,
    #[serde(flatten)]
    facts: &'a Facts,
}

/// The compile request of a build module, as `uniserve_build_server.request`
/// reads it.
#[derive(Serialize)]
pub struct Request<'a> {
    op: &'a str,
    backend: &'a str,
    descriptor: &'a str,
    /// The key's fields as the manifest's declared dimensions read them
    /// ([`Manifest::fields`]): a compilation takes its descriptor's values from
    /// here and never reads the key's text, so the key grammar is read in one
    /// place.
    fields: serde_json::Map<String, serde_json::Value>,
    parameters: &'a BTreeMap<String, toml::Value>,
    source: SourceKind,
    module: &'a str,
    unit: Option<&'a str>,
    package: &'a Path,
    output: &'a Path,
    target: RequestTarget<'a>,
    jobs: usize,
    cache: &'a Path,
    toolchain: &'a Report,
}

/// A build module: a Triton, CuTe DSL or CUDA package's compilation, run
/// through the Python environment with a typed compile request.
#[derive(Clone, Debug)]
pub struct ModuleSource(PackageSource);

impl Source for ModuleSource {
    fn build(&self, build: &Build<'_>) -> Result<Vec<String>> {
        let package = &self.0;
        let (key, target) = (build.key, build.target);
        let log = build_log(build.workspace);
        let environment = build
            .toolchains
            .environment(package.manifest.toolchain.kind)?;
        let facts = target.facts().ok_or_else(|| {
            package.failure(
                key,
                target,
                "a build module compiles for a GPU target of the description",
            )
        })?;
        let script = package
            .manifest
            .source
            .script
            .as_deref()
            .map(Path::new)
            .ok_or_else(|| package.failure(key, target, "the package names no build module"))?;
        let (Some(directory), Some(module)) = (
            script.parent().map(|parent| package.root.join(parent)),
            script.file_stem().and_then(|stem| stem.to_str()),
        ) else {
            return Err(package.failure(
                key,
                target,
                format!("{} names no Python module", script.display()),
            ));
        };
        let pins = build.cache.pins();
        let request = Request {
            op: &package.manifest.op,
            backend: &package.manifest.backend,
            descriptor: key,
            fields: package
                .manifest
                .fields(key)
                .unwrap_or_default()
                .into_iter()
                .map(|(dimension, value)| (dimension, serde_json::Value::String(value)))
                .collect(),
            parameters: &package.manifest.parameters,
            source: package.manifest.source.kind,
            module,
            unit: package.manifest.source.unit.as_deref(),
            package: &package.root,
            output: build.workspace.path(),
            target: RequestTarget {
                name: target.as_str(),
                facts,
            },
            jobs: build.jobs,
            cache: &pins,
            toolchain: environment,
        };
        let written = build.workspace.path().join(REQUEST);
        let text = serde_json::to_vec(&request).map_err(|error| {
            package.failure(key, target, format!("writing the compile request: {error}"))
        })?;
        std::fs::write(&written, text)?;

        // The module's directory is the compilation's working directory: `-m`
        // puts it first on the interpreter's import path, so the build module
        // and the package's sources beside it import as any program's own.
        let mut command = Command::new(&environment.interpreter);
        command
            .current_dir(directory)
            .args(["-m", "uniserve_build_server", "compile"])
            .stdin(std::fs::File::open(&written)?);
        crate::toolchain::execute(command, &log)
            .map_err(|error| package.failure_with_log(key, target, error, &log))?;
        package.check_module(key, target, build.workspace, &log)?;
        package.copy_tuning(key, target, build.workspace)?;
        collect(build.workspace.path())
    }
}

/// A vendor cubin the package carries (design section 5.3): the manifest's
/// `[source] cubin` and `[source] metadata`, published as `module.cubin` and
/// under the module's metadata name.
#[derive(Clone, Debug)]
pub struct CubinSource(PackageSource);

impl Source for CubinSource {
    fn build(&self, build: &Build<'_>) -> Result<Vec<String>> {
        let package = &self.0;
        let (key, target) = (build.key, build.target);
        let metadata = package.metadata_name(key, target)?;
        let roles = [
            (
                package.manifest.source.cubin.as_deref(),
                MODULE_CUBIN.to_owned(),
            ),
            (
                package.manifest.source.metadata.as_deref(),
                metadata.clone(),
            ),
        ];
        for (declared, name) in roles {
            let relative = declared
                .ok_or_else(|| {
                    package.failure(key, target, "the package names no cubin or metadata")
                })?
                .replace(crate::manifest::DESCRIPTOR_PLACEHOLDER, key);
            let source = package.file(&relative);
            if !source.is_file() {
                return Err(package.failure(
                    key,
                    target,
                    format!("the package carries no {relative} for this descriptor"),
                ));
            }
            copy_file(&source, &build.workspace.path().join(name))?;
        }
        check_metadata_symbol(
            &package.manifest.op,
            &build.workspace.path().join(&metadata),
        )
        .map_err(|message| package.failure(key, target, message))?;
        package.copy_tuning(key, target, build.workspace)?;
        collect(build.workspace.path())
    }
}

/// An executable, stored as it is (design section 5.1): the package revision
/// was built with its own toolchain, outside the Build Server, and this source
/// only publishes the bytes under the revision's path.
#[derive(Clone, Debug)]
pub struct ExecutableSource(PackageSource);

impl Source for ExecutableSource {
    fn build(&self, build: &Build<'_>) -> Result<Vec<String>> {
        let package = &self.0;
        let (key, target) = (build.key, build.target);
        for relative in package.manifest.source.files_for(key) {
            let source = package.file(&relative);
            if !source.is_file() {
                return Err(package.failure(
                    key,
                    target,
                    format!("the package carries no {relative}"),
                ));
            }
            copy_file(&source, &build.workspace.path().join(&relative))?;
        }
        package.copy_tuning(key, target, build.workspace)?;
        collect(build.workspace.path())
    }
}

/// A scheduling policy: a crate built for wasm32 from the revision's
/// whole-workspace snapshot (design section 5.3). The module is the crate's
/// wasm32 binary, which the engine's wasmtime compiles under its own
/// configuration when it loads it.
#[derive(Clone, Debug)]
pub struct WasmSource(PackageSource);

impl Source for WasmSource {
    fn build(&self, build: &Build<'_>) -> Result<Vec<String>> {
        let package = &self.0;
        let built = cargo_build(package, build, CargoArtifact::Wasm)?;
        let output = build.workspace.path().join(MODULE_WASM);
        copy_file(&built, &output)?;
        let mut files = vec![MODULE_WASM.to_owned()];
        files.extend(package.copy_tuning(build.key, build.target, build.workspace)?);
        Ok(files)
    }
}

/// A crate cargo builds to an executable from the revision's whole-workspace
/// snapshot: the engine.
///
/// The build is `cargo build --release --locked --offline` in the snapshot,
/// against the registry sources the installation vendored for its lockfile:
/// no crate is fetched and no version is chosen, so the executable is a
/// function of the revision and the toolchain. What cargo compiled stays in
/// the Build Server's cargo target directory, where the next revision finds
/// the dependencies it shares.
#[derive(Clone, Debug)]
pub struct CargoSource(PackageSource);

impl Source for CargoSource {
    fn build(&self, build: &Build<'_>) -> Result<Vec<String>> {
        let package = &self.0;
        let binary = package.manifest.source.binary.as_deref().ok_or_else(|| {
            package.failure(
                build.key,
                build.target,
                "the package names no [source] binary to build",
            )
        })?;
        let built = cargo_build(package, build, CargoArtifact::Binary(binary))?;
        copy_file(&built, &build.workspace.path().join(binary))?;
        package.copy_tuning(build.key, build.target, build.workspace)?;
        collect(build.workspace.path())
    }
}

/// What a cargo build of a package's crate produces.
enum CargoArtifact<'a> {
    /// The crate's library for wasm32.
    Wasm,
    /// One of the crate's binaries, for this host.
    Binary(&'a str),
}

/// Builds the package's crate of its revision's workspace snapshot with cargo,
/// and returns the artifact cargo reports it wrote.
///
/// The crate is the workspace member whose `package.toml` states this
/// package's operator and backend; the artifact is read from cargo's JSON
/// messages, not guessed under the target directory.
fn cargo_build(
    package: &PackageSource,
    build: &Build<'_>,
    artifact: CargoArtifact<'_>,
) -> Result<PathBuf> {
    let (key, target) = (build.key, build.target);
    let log = build_log(build.workspace);
    let workspace = package.root.join(crate::snapshot::WORKSPACE);
    let member = crate::snapshot::member(&package.root)
        .map_err(|error| package.failure(key, target, error.to_string()))?;
    let vendor = crate::snapshot::vendored(&workspace, build.cache)
        .map_err(|error| package.failure(key, target, error.to_string()))?;

    let messages = build.workspace.path().join(CARGO_MESSAGES);
    let mut command = crate::toolchain::cargo(&workspace);
    command
        .args([
            "build",
            "--release",
            "--locked",
            "--offline",
            "--message-format=json-render-diagnostics",
        ])
        .arg("--package")
        .arg(&member.id)
        .arg("--jobs")
        .arg(build.jobs.to_string())
        .arg("--target-dir")
        .arg(build.cache.cargo())
        .arg("--config")
        .arg(vendor);
    match artifact {
        CargoArtifact::Wasm => {
            command.args(["--lib", "--target", crate::target::WASM]);
        }
        CargoArtifact::Binary(binary) => {
            command.args(["--bin", binary]);
        }
    }
    // The messages are cargo's standard output; its standard error, with the
    // compiler's rendered diagnostics, is the build's log.
    let errors = std::fs::File::create(&log)?;
    let status = command
        .stdout(std::fs::File::create(&messages)?)
        .stderr(errors)
        .status()
        .map_err(|error| package.failure(key, target, format!("{CARGO} cannot be run: {error}")))?;
    if !status.success() {
        return Err(package.failure(
            key,
            target,
            format!(
                "{CARGO} build failed with {status}; its log was:\n{}",
                read_log(&log)
            ),
        ));
    }

    let text = std::fs::read_to_string(&messages)?;
    let written = text
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|message| {
            message["reason"] == "compiler-artifact" && message["package_id"] == member.id.as_str()
        })
        .find_map(|message| match &artifact {
            CargoArtifact::Wasm => message["filenames"]
                .as_array()?
                .iter()
                .filter_map(serde_json::Value::as_str)
                .find(|file| file.ends_with(".wasm"))
                .map(PathBuf::from),
            CargoArtifact::Binary(binary) => (message["target"]["name"] == *binary)
                .then(|| message["executable"].as_str().map(PathBuf::from))
                .flatten(),
        });
    written.ok_or_else(|| {
        package.failure(
            key,
            target,
            format!(
                "{CARGO} reported no artifact of {}; its log was:\n{}",
                member.id,
                read_log(&log)
            ),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vendor package carrying one cubin and one metadata JSON per descriptor, as
    /// `CubinSource` reads it (design 5.3), with or without its metadata.
    fn vendor_package(directory: &Path, metadata: bool) {
        std::fs::create_dir_all(directory.join("cubin")).unwrap();
        std::fs::write(directory.join("cubin/dtype_bf16.cubin"), b"a cubin").unwrap();
        if metadata {
            std::fs::write(
                directory.join("cubin/dtype_bf16.json"),
                br#"{"name": "vendor"}"#,
            )
            .unwrap();
        }
        std::fs::write(
            directory.join(Manifest::FILE),
            r#"
op = "vendor"
backend = "cubin"
interface_version = 1

[toolchain]
kind = "none"
version = ""

[module]
form = "cubin"
metadata = "launch.json"

[source]
kind = "cubin"
files = []
cubin = "cubin/{descriptor}.cubin"
metadata = "cubin/{descriptor}.json"

[kernel]
architectures = ["sm_100a"]

[[kernel.descriptors]]
dimensions = { dtype = ["bf16"] }
"#,
        )
        .unwrap();
    }

    /// Builds descriptor `dtype_bf16` of the package at `directory` for `sm_100a`.
    fn build_vendor(directory: &Path) -> (Result<Vec<String>>, Workspace, tempfile::TempDir) {
        let manifest = Manifest::load(directory).unwrap();
        let source = open(&manifest, directory);
        let root = tempfile::tempdir().unwrap();
        let cache = Cache::new(root.path().to_path_buf());
        let workspace = Workspace::new(&cache).unwrap();
        let toolchains = Toolchains::from_report(Err("no environment".into()));
        let built = source.build(&Build {
            op: "vendor",
            key: "dtype_bf16",
            target: &Target::new("sm_100a"),
            toolchains: &toolchains,
            cache: &cache,
            jobs: 1,
            workspace: &workspace,
        });
        (built, workspace, root)
    }

    #[test]
    fn a_vendor_cubin_is_stored_under_the_names_ops_loads_it_by() {
        let directory = tempfile::tempdir().unwrap();
        vendor_package(directory.path(), true);
        let (built, workspace, _root) = build_vendor(directory.path());

        assert_eq!(built.unwrap(), vec!["launch.json", "module.cubin"]);
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
        vendor_package(directory.path(), false);
        let error = build_vendor(directory.path()).0.unwrap_err();
        assert!(
            error.to_string().contains("dtype_bf16.json"),
            "the error names the missing metadata: {error}"
        );
    }
}
