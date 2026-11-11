//! Toolchains: what compiles a package (design section 6.2).
//!
//! A toolchain reports its version, which is compared with the version a
//! package declares before the package is built; a peak host memory, which the
//! compile queue reserves before it starts the build; and how to run a
//! compiler process, which every toolchain process caps with `MAX_JOBS`.
//!
//! Design section 5.2 names four compilers, and each is one toolchain:
//!
//! - nvcc, the CUDA toolkit's compiler, for CUDA C++ ([`ToolchainKind::Nvcc`]).
//! - Triton, from the Build Server's own Python environment under `python/`
//!   ([`ToolchainKind::Triton`]).
//! - CuTe DSL, from the same Python environment
//!   ([`ToolchainKind::CuteDsl`]).
//! - wasmtime, which precompiles a scheduling policy for this host's CPU
//!   ([`ToolchainKind::Wasm`]).
//!
//! A source that compiles nothing — a vendor cubin, an executable — needs no
//! toolchain and asks for [`ToolchainKind::None`], which always reports the
//! version the package declares.
//!
//! A compiler the host does not have is not an error of the server: it is
//! discovered as missing, and only a build that needs it fails. That keeps a
//! host without a CUDA toolkit able to serve the packages it already has.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::manifest::{PythonPackage, ToolchainDeps, ToolchainKind};
use crate::{Error, Result};

/// The peak host memory of one Triton compile, in bytes.
///
/// Triton's compiler holds the kernel's IR, its LLVM module and the ptxas
/// process at once; measured Triton builds of these kernels sit below four
/// gigabytes, and eight leaves room for the compiler's own parallelism
/// (`MAX_JOBS` caps the jobs, each of which may start a worker). It is the
/// reservation the queue makes, not a measurement of one build.
const TRITON_PEAK_HOST_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// The peak host memory of one CuTe DSL compile, in bytes: the DSL's Python
/// interpreter, the traced module, `ptxas` and the link of the host launcher.
/// Ten gigabytes leaves room above the largest attention kernel's trace.
const CUTE_DSL_PEAK_HOST_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// The peak host memory of one nvcc compile, in bytes: nvcc, cicc and ptxas
/// over one translation unit. Four gigabytes covers the attention and MoE
/// translation units, whose ptxas step dominates.
const NVCC_PEAK_HOST_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// The peak host memory of one wasm32 build and precompilation, in bytes:
/// `cargo build` of one policy crate plus its precompilation. Two gigabytes
/// covers both.
const WASM_PEAK_HOST_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The peak host memory of one compiler process of a cargo build, in bytes:
/// one rustc over a crate of the engine's workspace in release. A cargo build
/// runs as many of them at once as its job count.
const CARGO_JOB_PEAK_HOST_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The peak host memory of a cargo build's last step, in bytes: the link of
/// the executable under thin LTO, which holds the code of every crate it
/// links while the compiler processes before it have ended.
const CARGO_LINK_PEAK_HOST_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// What compiles a package.
pub trait Toolchain: Send + Sync {
    /// The compiler's version, such as nvcc's `13.0.48`, Triton's `3.7.1` or
    /// CuTe DSL's `4.6.2`. A package declares the version it is written for;
    /// a mismatch refuses the build (design section 6.2). A toolchain that
    /// compiles nothing returns the empty string, which such a package
    /// declares.
    fn version(&self) -> &str;

    /// The peak host memory a build of this descriptor would use, in bytes.
    /// The compile queue starts a build only when this much stays free after
    /// `limits.host_reserve` (design section 6.2).
    fn peak_host_bytes(&self, op: &str, key: &str) -> u64;

    /// Runs one compiler process to its end, capping the compilers it may
    /// start itself with `MAX_JOBS`. `command` states the executable, its
    /// arguments, the directory it runs in and the variables a source adds to
    /// the server's environment; the process's standard output and error are
    /// written to `log`, which a failed build carries as its log.
    fn run(&self, command: Command, log: &Path) -> Result<()>;

    /// The number of compilers a toolchain process may start, which is the
    /// queue's `limits.max_jobs`.
    fn max_jobs(&self) -> usize;

    /// The toolchain's principal executable: nvcc, or the wasm precompiler.
    /// `None` for a toolchain that compiles nothing.
    fn executable(&self) -> Option<&Path> {
        None
    }

    /// The interpreter that runs a package's Python compile script, when this
    /// toolchain has one.
    ///
    /// The Python toolchains are that interpreter. nvcc also runs a
    /// package's compile script, because a CUDA C++ package may state its
    /// command line in a Python driver (design section 6.2: a source drives
    /// the package's compile script), so it carries the Python environment's
    /// interpreter when the host has it.
    fn script_interpreter(&self) -> Option<&Path> {
        None
    }

    /// cargo, which builds a policy's crate for wasm32, when this toolchain
    /// has it.
    fn cargo(&self) -> Option<&Path> {
        None
    }

    /// The directory this toolchain keeps what it compiled between builds:
    /// cargo's target directory, where the dependencies one revision compiled
    /// are found by the next. `None` for a toolchain that keeps nothing.
    fn build_directory(&self) -> Option<&Path> {
        None
    }
}

/// Where each toolchain is on this host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolchainConfig {
    /// nvcc, which a CUDA package needs. `None` takes `$CUDA_HOME/bin/nvcc`, and refuses such a
    /// package when `CUDA_HOME` is not set either.
    pub nvcc: Option<PathBuf>,
    /// The interpreter of the Build Server's Python environment, created from
    /// `python/pyproject.toml` and `python/uv.lock`. Triton and CuTe DSL are the
    /// environment's two toolchains; `None` refuses their packages and runs no
    /// package's Python compile script.
    pub python: Option<PathBuf>,
    /// The wasm precompiler, `scheduling-precompile`, which builds a policy module with
    /// the schedule crate's own engine. `None` refuses a policy package.
    pub wasm: Option<PathBuf>,
    /// cargo, which builds a cargo package (the engine) and a policy's crate from their
    /// snapshots. `None` takes the `cargo` on this process's `PATH`, and refuses such a
    /// package when there is none.
    pub cargo: Option<PathBuf>,
    /// The number of compilers that may run at once (`limits.max_jobs`).
    pub max_jobs: usize,
}

/// The host's toolchains, each discovered with its version, or the reason it
/// is missing.
#[derive(Debug)]
pub struct Toolchains {
    nvcc: Discovered<Nvcc>,
    triton: Discovered<Triton>,
    cute_dsl: Discovered<CuteDsl>,
    wasm: Discovered<Wasm>,
    cargo: Discovered<Cargo>,
    /// The distributions the Build Server's Python environment has installed, by normalized
    /// name (PEP 503), with their versions: what a package's `[toolchain] packages` is compared
    /// with before it is built.
    distributions: Discovered<BTreeMap<String, String>>,
    /// The directory of the compile driver package (`uniserve_build_server`) the Python
    /// environment imports: the Triton and CuTe DSL drivers, the fetcher and its pins, which every
    /// package's compile script runs through. Its files are an input of every module a script
    /// builds ([`Self::identity`]).
    driver: Discovered<PathBuf>,
}

/// A toolchain the host may not have.
type Discovered<T> = std::result::Result<T, String>;

impl Toolchains {
    /// Discovers the host's toolchains and their versions.
    ///
    /// A missing toolchain is recorded with its reason; it does not fail the
    /// server, because only a build that needs it does.
    pub fn discover(config: &ToolchainConfig) -> Result<Toolchains> {
        let interpreter = config.python.as_deref().map(resolve);
        let interpreter = interpreter.as_deref();
        Ok(Toolchains {
            nvcc: Nvcc::discover(config),
            triton: Triton::discover(config, interpreter),
            cute_dsl: CuteDsl::discover(config, interpreter),
            wasm: Wasm::discover(config),
            cargo: Cargo::discover(config),
            distributions: named_interpreter(interpreter).and_then(python_distributions),
            driver: named_interpreter(interpreter).and_then(driver_directory),
        })
    }

    /// What builds a package of toolchain `kind` on this host, as one line: the toolchain's kind
    /// and version and, for a toolchain that runs a package's Python compile script, the digest of
    /// the compile driver's files, read now. Two hosts, or one host before and after an upgrade or
    /// an edit of the driver or of a pin, response the same line only when they build the same bytes
    /// from the same package (`crate::inputs`).
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Toolchain`] when the host has no toolchain of that kind, or its
    /// compile driver cannot be found or read.
    pub fn identity(&self, kind: ToolchainKind) -> Result<String> {
        if kind == ToolchainKind::None {
            return Ok(kind.to_string());
        }
        let toolchain = self.get(kind)?;
        let mut identity = format!("{kind} {}", toolchain.version());
        if kind == ToolchainKind::Wasm {
            // A policy's module is cargo's output before it is the
            // precompiler's, so the compiler is part of what built it.
            let cargo = self.get(ToolchainKind::Cargo)?;
            identity.push_str("; cargo ");
            identity.push_str(cargo.version());
        }
        if toolchain.script_interpreter().is_some() {
            let driver = self.driver.as_ref().map_err(|reason| Error::Toolchain {
                kind,
                message: reason.clone(),
            })?;
            identity.push_str("; driver ");
            identity.push_str(&crate::snapshot::tree_digest(driver)?);
        }
        Ok(identity)
    }

    /// Refuses a build whose package imports a Python distribution this host's environment does
    /// not have at the version the package declares (design section 5.2: a package names the
    /// distributions its sources import, and the environment's lock fixes their versions).
    ///
    /// A package that declares none needs no environment for this. Names are compared as PEP 503
    /// normalizes them, so `flash_attn_4` and `flash-attn-4` are one distribution.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Toolchain`], naming every distribution that is missing or at another
    /// version, or why the environment's distributions could not be read.
    pub fn check_packages(&self, op: &str, backend: &str, toolchain: &ToolchainDeps) -> Result<()> {
        if toolchain.packages.is_empty() {
            return Ok(());
        }
        let refuse = |message: String| Error::Toolchain {
            kind: toolchain.kind,
            message: format!("the package {op} {backend} {message}"),
        };
        let installed = self
            .distributions
            .as_ref()
            .map_err(|reason| refuse(format!("imports Python distributions, and {reason}")))?;
        let disagreements = disagreements(&toolchain.packages, installed);
        if disagreements.is_empty() {
            Ok(())
        } else {
            Err(refuse(format!(
                "imports {} that this host's Python environment does not have",
                disagreements.join(", ")
            )))
        }
    }

    /// The toolchain of a kind, with the host's configuration.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Toolchain`] when the host has no toolchain of that
    /// kind, or it cannot report its version.
    pub fn get(&self, kind: ToolchainKind) -> Result<&dyn Toolchain> {
        match kind {
            ToolchainKind::Nvcc => self
                .nvcc
                .as_ref()
                .map(|toolchain| toolchain as &dyn Toolchain),
            ToolchainKind::Triton => self
                .triton
                .as_ref()
                .map(|toolchain| toolchain as &dyn Toolchain),
            ToolchainKind::CuteDsl => self
                .cute_dsl
                .as_ref()
                .map(|toolchain| toolchain as &dyn Toolchain),
            ToolchainKind::Wasm => self
                .wasm
                .as_ref()
                .map(|toolchain| toolchain as &dyn Toolchain),
            ToolchainKind::Cargo => self
                .cargo
                .as_ref()
                .map(|toolchain| toolchain as &dyn Toolchain),
            ToolchainKind::None => Ok(&NOTHING as &dyn Toolchain),
        }
        .map_err(|reason| Error::Toolchain {
            kind,
            message: reason.clone(),
        })
    }

    /// The toolchain's version, for `list` and the logs; a missing toolchain
    /// returns its reason.
    pub fn version(&self, kind: ToolchainKind) -> std::result::Result<&str, &str> {
        match kind {
            ToolchainKind::Nvcc => self
                .nvcc
                .as_ref()
                .map(|toolchain| toolchain.version())
                .map_err(String::as_str),
            ToolchainKind::Triton => self
                .triton
                .as_ref()
                .map(|toolchain| toolchain.version())
                .map_err(String::as_str),
            ToolchainKind::CuteDsl => self
                .cute_dsl
                .as_ref()
                .map(|toolchain| toolchain.version())
                .map_err(String::as_str),
            ToolchainKind::Wasm => self
                .wasm
                .as_ref()
                .map(|toolchain| toolchain.version())
                .map_err(String::as_str),
            ToolchainKind::Cargo => self
                .cargo
                .as_ref()
                .map(|toolchain| toolchain.version())
                .map_err(String::as_str),
            ToolchainKind::None => Ok(""),
        }
    }
}

/// nvcc, the CUDA toolkit's compiler.
#[derive(Debug)]
struct Nvcc {
    path: PathBuf,
    version: String,
    max_jobs: usize,
    /// The interpreter of the Build Server's Python environment, which runs
    /// a CUDA package's compile script when the package names one.
    interpreter: Option<PathBuf>,
}

impl Nvcc {
    fn discover(config: &ToolchainConfig) -> Discovered<Nvcc> {
        let path = config
            .nvcc
            .clone()
            .or_else(|| {
                std::env::var_os("CUDA_HOME").map(|home| PathBuf::from(home).join("bin/nvcc"))
            })
            .ok_or_else(|| "no nvcc named: pass --nvcc <path> or set CUDA_HOME".to_owned())?;
        // A configured path may be relative to where the server was started; a build
        // runs elsewhere (the store's copy of a package), so it is made absolute once.
        let path = resolve(&path);
        let reported = Command::new(&path)
            .arg("--version")
            .output()
            .map_err(|error| format!("running {}: {error}", path.display()))?;
        let text = String::from_utf8_lossy(&reported.stdout);
        if !reported.status.success() {
            return Err(format!(
                "{} --version failed: {}",
                path.display(),
                text.trim()
            ));
        }
        let version = parse_nvcc_version(&text)
            .ok_or_else(|| format!("{} --version names no release: {text}", path.display()))?;
        Ok(Nvcc {
            path,
            version,
            max_jobs: config.max_jobs,
            interpreter: config
                .python
                .as_deref()
                .map(resolve)
                .filter(|interpreter| interpreter.is_file()),
        })
    }
}

impl Toolchain for Nvcc {
    fn version(&self) -> &str {
        &self.version
    }

    fn peak_host_bytes(&self, _op: &str, _key: &str) -> u64 {
        NVCC_PEAK_HOST_BYTES
    }

    fn run(&self, command: Command, log: &Path) -> Result<()> {
        execute(command, log, self.max_jobs)
    }

    fn max_jobs(&self) -> usize {
        self.max_jobs
    }

    fn executable(&self) -> Option<&Path> {
        Some(&self.path)
    }

    fn script_interpreter(&self) -> Option<&Path> {
        self.interpreter.as_deref()
    }
}

/// Triton, from the Build Server's Python environment.
#[derive(Debug)]
struct Triton {
    interpreter: PathBuf,
    version: String,
    max_jobs: usize,
}

impl Triton {
    fn discover(config: &ToolchainConfig, interpreter: Option<&Path>) -> Discovered<Triton> {
        let interpreter = named_interpreter(interpreter)?;
        let version = python_version(interpreter, "triton")?;
        Ok(Triton {
            interpreter: interpreter.to_path_buf(),
            version,
            max_jobs: config.max_jobs,
        })
    }
}

impl Toolchain for Triton {
    fn version(&self) -> &str {
        &self.version
    }

    fn peak_host_bytes(&self, _op: &str, _key: &str) -> u64 {
        TRITON_PEAK_HOST_BYTES
    }

    fn run(&self, command: Command, log: &Path) -> Result<()> {
        execute(command, log, self.max_jobs)
    }

    fn max_jobs(&self) -> usize {
        self.max_jobs
    }

    fn script_interpreter(&self) -> Option<&Path> {
        Some(&self.interpreter)
    }
}

/// CuTe DSL, from the Build Server's Python environment.
#[derive(Debug)]
struct CuteDsl {
    interpreter: PathBuf,
    version: String,
    max_jobs: usize,
}

impl CuteDsl {
    fn discover(config: &ToolchainConfig, interpreter: Option<&Path>) -> Discovered<CuteDsl> {
        let interpreter = named_interpreter(interpreter)?;
        let version = python_version(interpreter, "cutlass")?;
        Ok(CuteDsl {
            interpreter: interpreter.to_path_buf(),
            version,
            max_jobs: config.max_jobs,
        })
    }
}

impl Toolchain for CuteDsl {
    fn version(&self) -> &str {
        &self.version
    }

    fn peak_host_bytes(&self, _op: &str, _key: &str) -> u64 {
        CUTE_DSL_PEAK_HOST_BYTES
    }

    fn run(&self, command: Command, log: &Path) -> Result<()> {
        execute(command, log, self.max_jobs)
    }

    fn max_jobs(&self) -> usize {
        self.max_jobs
    }

    fn script_interpreter(&self) -> Option<&Path> {
        Some(&self.interpreter)
    }
}

/// The WebAssembly toolchain of a scheduling policy: cargo, which builds the
/// policy's crate for wasm32, and the scheduling crate's precompiler, which
/// serializes the module for this host's CPU (design section 5.3).
#[derive(Debug)]
struct Wasm {
    precompiler: PathBuf,
    /// The precompiler's wasmtime version, which a policy package declares: a
    /// serialized module loads only in the wasmtime that wrote it.
    version: String,
    max_jobs: usize,
    /// cargo, or why this host has none.
    cargo: Discovered<Cargo>,
}

impl Wasm {
    fn discover(config: &ToolchainConfig) -> Discovered<Wasm> {
        let precompiler = config.wasm.clone().ok_or_else(|| {
            "no wasm precompiler named: pass --wasm <scheduling-precompile>".to_owned()
        })?;
        let precompiler = resolve(&precompiler);
        let version = wasmtime_version(&precompiler)?;
        Ok(Wasm {
            precompiler,
            version,
            max_jobs: config.max_jobs,
            cargo: Cargo::discover(config),
        })
    }
}

impl Toolchain for Wasm {
    fn version(&self) -> &str {
        &self.version
    }

    fn peak_host_bytes(&self, _op: &str, _key: &str) -> u64 {
        WASM_PEAK_HOST_BYTES
    }

    fn run(&self, command: Command, log: &Path) -> Result<()> {
        execute(command, log, self.max_jobs())
    }

    fn max_jobs(&self) -> usize {
        self.cargo
            .as_ref()
            .map_or(self.max_jobs, |cargo| cargo.jobs)
    }

    fn executable(&self) -> Option<&Path> {
        Some(&self.precompiler)
    }

    fn cargo(&self) -> Option<&Path> {
        self.cargo.as_ref().ok().map(|cargo| cargo.path.as_path())
    }

    fn build_directory(&self) -> Option<&Path> {
        self.cargo
            .as_ref()
            .ok()
            .map(|cargo| cargo.target_directory.as_path())
    }
}

/// cargo and the compiler it drives: what builds a cargo package, the engine,
/// from its snapshot, and a policy's crate for wasm32.
#[derive(Clone, Debug)]
struct Cargo {
    path: PathBuf,
    version: String,
    /// The compiler processes one build runs at once: `CARGO_BUILD_JOBS` of
    /// the server's environment, else the queue's `limits.max_jobs`.
    jobs: usize,
    /// cargo's target directory: `CARGO_TARGET_DIR` of the server's
    /// environment, else `build-server/cargo` in this user's cache directory.
    target_directory: PathBuf,
}

impl Cargo {
    fn discover(config: &ToolchainConfig) -> Discovered<Cargo> {
        let path = match &config.cargo {
            Some(path) => resolve(path),
            None => std::env::var_os("PATH")
                .and_then(|paths| {
                    std::env::split_paths(&paths)
                        .map(|directory| directory.join("cargo"))
                        .find(|candidate| candidate.is_file())
                })
                .ok_or_else(|| "no cargo named: pass --cargo <path>".to_owned())?,
        };
        let reported = Command::new(&path)
            .arg("--version")
            .output()
            .map_err(|error| format!("running {}: {error}", path.display()))?;
        let text = String::from_utf8_lossy(&reported.stdout);
        // `cargo 1.99.1 (5f94df478 2026-08-27)`: the release is the second word.
        let version = text
            .split_whitespace()
            .nth(1)
            .filter(|_| reported.status.success())
            .ok_or_else(|| format!("{} --version names no release: {text}", path.display()))?
            .to_owned();

        let jobs = std::env::var("CARGO_BUILD_JOBS")
            .ok()
            .and_then(|jobs| jobs.parse::<usize>().ok())
            .filter(|jobs| *jobs > 0)
            .unwrap_or(config.max_jobs);
        let target_directory = match std::env::var_os("CARGO_TARGET_DIR") {
            Some(directory) if !directory.is_empty() => resolve(Path::new(&directory)),
            _ => cache_directory().join("cargo"),
        };
        Ok(Cargo {
            path,
            version,
            jobs,
            target_directory,
        })
    }
}

impl Toolchain for Cargo {
    fn version(&self) -> &str {
        &self.version
    }

    fn peak_host_bytes(&self, _op: &str, _key: &str) -> u64 {
        // The compiler processes run first and the link after them, so the
        // build's peak is the larger of the two, not their sum.
        (self.jobs as u64 * CARGO_JOB_PEAK_HOST_BYTES).max(CARGO_LINK_PEAK_HOST_BYTES)
    }

    fn run(&self, command: Command, log: &Path) -> Result<()> {
        execute(command, log, self.jobs)
    }

    fn max_jobs(&self) -> usize {
        self.jobs
    }

    fn executable(&self) -> Option<&Path> {
        Some(&self.path)
    }

    fn build_directory(&self) -> Option<&Path> {
        Some(&self.target_directory)
    }
}

/// The directory of this user's Build Server caches:
/// `$XDG_CACHE_HOME/uniserve/build-server`, else
/// `~/.cache/uniserve/build-server`.
fn cache_directory() -> PathBuf {
    let cache = match std::env::var_os("XDG_CACHE_HOME") {
        Some(cache) if !cache.is_empty() => PathBuf::from(cache),
        _ => std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".cache"))
            .unwrap_or_else(std::env::temp_dir),
    };
    cache.join("uniserve").join("build-server")
}

/// A toolchain for a source that compiles nothing: a vendor cubin, or an
/// executable stored as it is. It reports the empty version, which such a
/// package declares, and runs nothing.
#[derive(Debug)]
struct Nothing;

static NOTHING: Nothing = Nothing;

impl Toolchain for Nothing {
    fn version(&self) -> &str {
        ""
    }

    fn peak_host_bytes(&self, _op: &str, _key: &str) -> u64 {
        0
    }

    fn run(&self, command: Command, _log: &Path) -> Result<()> {
        Err(Error::Build {
            op: String::new(),
            backend: String::new(),
            revision: 0,
            target: String::new(),
            descriptor: String::new(),
            message: format!(
                "nothing to run: this source compiles nothing ({} was asked)",
                Path::new(command.get_program()).display()
            ),
        })
    }

    fn max_jobs(&self) -> usize {
        1
    }
}

/// The interpreter the configuration names, or why a Python toolchain is
/// missing without one.
fn named_interpreter(interpreter: Option<&Path>) -> Discovered<&Path> {
    interpreter.ok_or_else(|| {
        "no Python interpreter named: pass --python <the Build Server's interpreter>".to_owned()
    })
}

/// A distribution name as PEP 503 normalizes it: lowercase, with every run of `-`, `_` and `.`
/// one `-`.
fn normalized(name: &str) -> String {
    let mut normal = String::with_capacity(name.len());
    let mut separated = false;
    for character in name.chars() {
        if matches!(character, '-' | '_' | '.') {
            separated = true;
            continue;
        }
        if separated && !normal.is_empty() {
            normal.push('-');
        }
        separated = false;
        normal.extend(character.to_lowercase());
    }
    normal
}

/// The declared distributions `installed` does not hold at their declared version, each as
/// `name version (it has ...)`.
fn disagreements(declared: &[PythonPackage], installed: &BTreeMap<String, String>) -> Vec<String> {
    declared
        .iter()
        .filter_map(|package| match installed.get(&normalized(&package.name)) {
            Some(version) if *version == package.version => None,
            Some(version) => Some(format!(
                "{} {} (it has {version})",
                package.name, package.version
            )),
            None => Some(format!(
                "{} {} (it has none)",
                package.name, package.version
            )),
        })
        .collect()
}

/// Every distribution the Python environment has installed, by normalized name, with its version.
fn python_distributions(interpreter: &Path) -> Discovered<BTreeMap<String, String>> {
    if !interpreter.is_file() {
        return Err(format!(
            "no Python interpreter at {}: pass --python <the Build Server's interpreter>",
            interpreter.display()
        ));
    }
    let reported = Command::new(interpreter)
        .args([
            "-c",
            "import importlib.metadata as m, json; \
             print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))",
        ])
        .output()
        .map_err(|error| format!("running {}: {error}", interpreter.display()))?;
    if !reported.status.success() {
        return Err(format!(
            "{} cannot list its distributions: {}",
            interpreter.display(),
            String::from_utf8_lossy(&reported.stderr).trim()
        ));
    }
    let listed: BTreeMap<String, String> = serde_json::from_slice(&reported.stdout)
        .map_err(|error| format!("{} listed no distributions: {error}", interpreter.display()))?;
    Ok(listed
        .into_iter()
        .map(|(name, version)| (normalized(&name), version))
        .collect())
}

/// The directory of the compile driver package the Python environment imports.
fn driver_directory(interpreter: &Path) -> Discovered<PathBuf> {
    if !interpreter.is_file() {
        return Err(format!(
            "no Python interpreter at {}: pass --python <the Build Server's interpreter>",
            interpreter.display()
        ));
    }
    let reported = Command::new(interpreter)
        .args([
            "-c",
            "import pathlib, uniserve_build_server as d; \
             print(pathlib.Path(d.__file__).resolve().parent)",
        ])
        .output()
        .map_err(|error| format!("running {}: {error}", interpreter.display()))?;
    if !reported.status.success() {
        return Err(format!(
            "{} cannot import the compile driver uniserve_build_server: {}",
            interpreter.display(),
            String::from_utf8_lossy(&reported.stderr).trim()
        ));
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&reported.stdout).trim(),
    ))
}

/// Asks the Python environment for the version of one of its distributions.
fn python_version(interpreter: &Path, module: &str) -> Discovered<String> {
    if !interpreter.is_file() {
        return Err(format!(
            "no Python interpreter at {}: pass --python <the Build Server's interpreter>",
            interpreter.display()
        ));
    }
    let reported = Command::new(interpreter)
        .args([
            "-c",
            &format!("import {module}; print({module}.__version__)"),
        ])
        .output()
        .map_err(|error| format!("running {}: {error}", interpreter.display()))?;
    let text = String::from_utf8_lossy(&reported.stdout).trim().to_owned();
    let error = String::from_utf8_lossy(&reported.stderr).trim().to_owned();
    if !reported.status.success() {
        return Err(format!(
            "{} cannot import {module}: {error}",
            interpreter.display()
        ));
    }
    if text.is_empty() {
        return Err(format!(
            "{} reported no {module} version",
            interpreter.display()
        ));
    }
    Ok(text)
}

/// The wasmtime version a precompiler reports, either itself or through the
/// `wasmtime` executable beside it. A precompiled module is only loadable by the
/// wasmtime that wrote it, so this string is what a policy package declares.
fn wasmtime_version(precompiler: &Path) -> Discovered<String> {
    if let Ok(reported) = Command::new(precompiler).arg("--version").output()
        && reported.status.success()
    {
        let text = String::from_utf8_lossy(&reported.stdout);
        if let Some(version) = text.split_whitespace().last() {
            return Ok(version.to_owned());
        }
    }
    Err(format!(
        "{} does not report a wasmtime version; it must print one for `--version`",
        precompiler.display()
    ))
}

/// The version nvcc reports, such as `13.0.48` from
/// `Cuda compilation tools, release 13.0, V13.0.48`.
fn parse_nvcc_version(text: &str) -> Option<String> {
    let release = text.split("release ").nth(1)?;
    let after = release.split(',').nth(1)?.trim();
    // `nvcc --version` follows the release with a second line ("Build
    // cuda_13.0.r13.0/compiler.…"), so the version is the first word after
    // the `V`, not everything to the end of the output.
    let version = after
        .strip_prefix('V')
        .unwrap_or(after)
        .split_whitespace()
        .next()
        .unwrap_or_default();
    if version.is_empty() {
        None
    } else {
        Some(version.to_owned())
    }
}

/// A configured executable as an absolute path, against the directory the server runs in.
///
/// A build runs with its working directory set to the store's copy of a package, so a path left
/// relative would name that copy instead of the executable the caller named.
fn resolve(configured: &Path) -> PathBuf {
    match std::env::current_dir() {
        Ok(current) => resolve_in(configured, &current),
        Err(_) => configured.to_path_buf(),
    }
}

/// [`resolve`] against an explicit starting directory, which is what makes it a function a test
/// can call without moving the process's working directory.
fn resolve_in(configured: &Path, current: &Path) -> PathBuf {
    if configured.is_absolute() {
        configured.to_path_buf()
    } else {
        current.join(configured)
    }
}

/// Runs a compiler process to its end, writing its output to `log`.
///
/// The process runs with the server's environment, `MAX_JOBS` set to
/// `max_jobs` unless the command states its own, and the command's variables.
pub fn execute(mut command: Command, log: &Path, max_jobs: usize) -> Result<()> {
    if !command.get_envs().any(|(name, _)| name == "MAX_JOBS") {
        command.env("MAX_JOBS", max_jobs.to_string());
    }

    let output = std::fs::File::create(log)?;
    let error_output = output.try_clone()?;
    let executable = command.get_program().to_owned();
    let status = command
        .stdout(output)
        .stderr(error_output)
        .status()
        .map_err(|error| Error::Build {
            op: String::new(),
            backend: String::new(),
            revision: 0,
            target: String::new(),
            descriptor: String::new(),
            // A build that cannot find its compiler names the executable, its arguments
            // and the directory it was started in: an errno alone says none of them,
            // and the log is empty because the process never ran.
            message: format!(
                "could not run {} {} in {}: {error}",
                Path::new(&executable).display(),
                command
                    .get_args()
                    .map(|argument| argument.to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join(" "),
                command
                    .get_current_dir()
                    .unwrap_or(Path::new("."))
                    .display()
            ),
        })?;

    if status.success() {
        Ok(())
    } else {
        let text = std::fs::read_to_string(log).unwrap_or_default();
        Err(Error::Build {
            op: String::new(),
            backend: String::new(),
            revision: 0,
            target: String::new(),
            descriptor: String::new(),
            message: format!(
                "{} failed: {status}: {text}",
                Path::new(&executable).display()
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ToolchainConfig, Toolchains, disagreements, execute, normalized, parse_nvcc_version,
        resolve_in,
    };
    use crate::manifest::{PythonPackage, ToolchainKind};
    use std::collections::BTreeMap;
    use std::path::Path;

    /// A declared distribution is compared by its normalized name and its exact version: the
    /// environment's spelling of a name does not matter, its version does, and a distribution it
    /// lacks is named as missing.
    #[test]
    fn a_declared_distribution_must_be_installed_at_its_version() {
        let installed: BTreeMap<String, String> =
            [("flash_attn_4", "4.0.0b30"), ("Torch", "2.13.0")]
                .into_iter()
                .map(|(name, version)| (normalized(name), version.to_owned()))
                .collect();
        let declared = |name: &str, version: &str| PythonPackage {
            name: name.to_owned(),
            version: version.to_owned(),
        };
        assert!(
            disagreements(
                &[
                    declared("flash-attn-4", "4.0.0b30"),
                    declared("torch", "2.13.0")
                ],
                &installed
            )
            .is_empty()
        );
        assert_eq!(
            disagreements(
                &[
                    declared("flash-attn-4", "4.0.0"),
                    declared("flashinfer-python", "0.6.18")
                ],
                &installed
            ),
            [
                "flash-attn-4 4.0.0 (it has 4.0.0b30)",
                "flashinfer-python 0.6.18 (it has none)"
            ]
        );
    }

    /// A Python toolchain's identity names its compile driver's files as they are when it is
    /// asked: an edited driver or pin is another identity, so its modules are other modules.
    #[test]
    fn a_python_toolchain_identity_follows_its_compile_driver() {
        use std::os::unix::fs::PermissionsExt as _;

        let temporary = tempfile::tempdir().unwrap();
        let driver = temporary.path().join("uniserve_build_server");
        std::fs::create_dir_all(driver.join("pins")).unwrap();
        std::fs::write(driver.join("triton.py"), "# the driver\n").unwrap();
        std::fs::write(driver.join("pins/artifact.sha256"), "ab  file\n").unwrap();
        // The Python environment is outside the server: a script returns the questions discovery
        // asks an interpreter (the driver's directory, a version, the distributions).
        let python = temporary.path().join("python");
        std::fs::write(
            &python,
            format!(
                "#!/bin/sh\ncase \"$2\" in\n  *uniserve_build_server*) echo {};;\n  \
                 *distributions*) echo '{{}}';;\n  *) echo 3.7.1;;\nesac\n",
                driver.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
        let toolchains = Toolchains::discover(&ToolchainConfig {
            nvcc: None,
            python: Some(python),
            wasm: None,
            cargo: None,
            max_jobs: 1,
        })
        .unwrap();

        let first = toolchains.identity(ToolchainKind::Triton).unwrap();
        assert!(first.starts_with("triton 3.7.1; driver "), "{first}");
        assert_eq!(toolchains.identity(ToolchainKind::Triton).unwrap(), first);
        std::fs::write(driver.join("triton.py"), "# the driver, edited\n").unwrap();
        let edited = toolchains.identity(ToolchainKind::Triton).unwrap();
        assert_ne!(edited, first);
        std::fs::write(driver.join("pins/artifact.sha256"), "cd  file\n").unwrap();
        assert_ne!(toolchains.identity(ToolchainKind::Triton).unwrap(), edited);
        assert_eq!(toolchains.identity(ToolchainKind::None).unwrap(), "none");
    }

    /// A build whose compiler is not where the host said names the executable, its
    /// arguments and the directory it ran in. An errno alone says none of them and the
    /// log is empty, because the process never started: that combination is what made
    /// a relative toolchain path cost an hour to diagnose.
    #[test]
    fn a_build_that_cannot_find_its_compiler_names_the_executable_and_where_it_ran() {
        let temporary =
            std::env::temp_dir().join(format!("uniserve-tooljob-{}", std::process::id()));
        std::fs::create_dir_all(&temporary).unwrap();
        let log = temporary.join("build.log");
        let mut command = std::process::Command::new("/no-such-toolchain/python");
        command.current_dir(&temporary).arg("--version");
        let error = execute(command, &log, 2).expect_err("a missing executable cannot run");
        let text = error.to_string();
        assert!(text.contains("/no-such-toolchain/python"), "{text}");
        assert!(text.contains("--version"), "{text}");
        assert!(text.contains(&temporary.display().to_string()), "{text}");
        let _ = std::fs::remove_dir_all(&temporary);
    }

    /// A relative toolchain path is returned absolute, against the directory it is resolved in: a
    /// build runs with its working directory set to the store's copy of a package, so a path that
    /// stayed relative would name that copy instead of the executable the caller named.
    #[test]
    fn a_relative_toolchain_path_is_resolved_absolute() {
        let root = Path::new("/srv/host");
        assert_eq!(
            resolve_in(Path::new("bin/nvcc"), root),
            root.join("bin/nvcc")
        );
        assert_eq!(
            resolve_in(Path::new("/opt/nvcc"), root),
            Path::new("/opt/nvcc")
        );
    }

    #[test]
    fn nvcc_version_is_the_v_token() {
        let text = "\nnvcc: NVIDIA (R) Cuda compiler driver\n\
                    Copyright (c) 2005-2025 NVIDIA Corporation\n\
                    Built on Fri_Aug__1_00:00:00_PDT_2025\n\
                    Cuda compilation tools, release 13.0, V13.0.48\n\
                    Build cuda_13.0.r13.0/compiler.36260728_0\n";
        assert_eq!(parse_nvcc_version(text).as_deref(), Some("13.0.48"));
    }

    #[test]
    fn an_unknown_nvcc_output_has_no_version() {
        assert_eq!(parse_nvcc_version("not nvcc"), None);
    }
}
