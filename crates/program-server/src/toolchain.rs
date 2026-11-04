//! Toolchains: what compiles a package (design section 6.2).
//!
//! A toolchain reports its version, which is compared with the version a
//! package declares before the package is built; a peak host memory, which the
//! compile queue reserves before it starts the build; and how to run a
//! compiler process, which every toolchain process caps with `MAX_JOBS`.
//!
//! The three toolchains are nvcc (the CUDA toolkit), the Program Server's own
//! Python environment under `python/` (Triton and CuTe DSL), and the
//! WebAssembly toolchain (a wasm32 build, then `wasmtime`'s precompilation for
//! this host's CPU).
//!
//! A toolchain the host does not have is not an error of the server: it is
//! discovered as missing, and only a build that needs it fails. That keeps a
//! host without a CUDA toolkit able to serve the packages it already has.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::manifest::ToolchainKind;
use crate::{Error, Result};

/// The peak host memory of one Python compile, in bytes.
///
/// Triton's compiler holds the kernel's IR, its LLVM module and the ptxas
/// process at once; measured Triton builds of the attention kernels sit below
/// four gigabytes, and eight leaves room for the compiler's own parallelism
/// (`MAX_JOBS` caps the jobs, each of which may start a worker). The number is
/// the reservation the queue makes, not a measurement of one build.
const PYTHON_PEAK_HOST_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// The peak host memory of one nvcc compile, in bytes: nvcc, cicc and ptxas
/// over one translation unit. Four gigabytes covers the attention and MoE
/// translation units, whose ptxas step dominates.
const NVCC_PEAK_HOST_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// The peak host memory of one wasm32 build and precompilation, in bytes:
/// `cargo build` of one policy crate plus `wasmtime compile`. Two gigabytes
/// covers both.
const WASM_PEAK_HOST_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// One compiler process a source runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolJob {
    /// The program to run: the interpreter, nvcc or cargo.
    pub program: PathBuf,
    pub arguments: Vec<OsString>,
    /// The directory the program runs in.
    pub directory: PathBuf,
    /// Variables added to the toolchain's environment.
    pub environment: Vec<(OsString, OsString)>,
    /// The file the program's standard output and error are written to; a
    /// failed build carries it as its log.
    pub log: PathBuf,
}

impl ToolJob {
    /// A job of a toolchain's own program in `directory`, with its output
    /// going to `log`.
    pub fn new(program: &Path, directory: &Path, log: &Path) -> Self {
        Self {
            program: program.to_path_buf(),
            arguments: Vec::new(),
            directory: directory.to_path_buf(),
            environment: Vec::new(),
            log: log.to_path_buf(),
        }
    }

    /// Adds one argument.
    pub fn argument(mut self, argument: impl Into<OsString>) -> Self {
        self.arguments.push(argument.into());
        self
    }

    /// Adds the variables a source passes to its compiler, such as the
    /// package root and the descriptor key.
    pub fn environment(
        mut self,
        environment: impl IntoIterator<Item = (OsString, OsString)>,
    ) -> Self {
        self.environment.extend(environment);
        self
    }
}

/// What compiles a package.
pub trait Toolchain: Send + Sync {
    /// The compiler's version, such as nvcc's `13.0.48` or Triton's `3.7.1`.
    /// A package declares the version it is written for; a mismatch refuses
    /// the build (design section 6.2).
    fn version(&self) -> &str;

    /// The peak host memory a build of this descriptor would use, in bytes.
    /// The compile queue starts a build only when this much stays free after
    /// `limits.host_reserve` (design section 6.2).
    fn peak_host_bytes(&self, op: &str, key: &str) -> u64;

    /// Runs one compiler process, capping the compilers it may start itself
    /// with `MAX_JOBS`.
    fn run(&self, job: ToolJob) -> Result<()>;

    /// The number of compilers a toolchain process may start, which is the
    /// queue's `limits.max_jobs`.
    fn max_jobs(&self) -> usize;

    /// The toolchain's principal program: the interpreter of the Python
    /// environment, nvcc, or the precompiler of a wasm module. A source runs
    /// it as its [`ToolJob`]'s program.
    fn executable(&self) -> &Path;

    /// The interpreter that runs a package's Python compile script, when this
    /// toolchain has one.
    ///
    /// The Python toolchain is that interpreter. nvcc and the wasm toolchain
    /// also run a package's compile script, because a CUDA C++ or wasm32
    /// package states its command line in a Python driver (design section
    /// 6.2: a source drives the package's compile script), so they carry the
    /// Python environment's interpreter when the host has it.
    fn script_interpreter(&self) -> Option<&Path> {
        None
    }
}

/// Where each toolchain is on this host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolchainConfig {
    /// nvcc, when the host has a CUDA toolkit. `None` looks for `nvcc` on
    /// `PATH` and in `/usr/local/cuda/bin`.
    pub nvcc: Option<PathBuf>,
    /// The Program Server's Python environment: the interpreter under
    /// `python/.venv`, created from `python/pyproject.toml` and
    /// `python/uv.lock`.
    pub python: PathBuf,
    /// wasmtime, when the host can precompile wasm modules. `None` looks for
    /// `wasmtime` on `PATH`.
    pub wasmtime: Option<PathBuf>,
    /// The number of compilers that may run at once (`limits.max_jobs`).
    pub max_jobs: usize,
}

/// The host's toolchains, each discovered with its version, or the reason it
/// is missing.
#[derive(Debug)]
pub struct Toolchains {
    config: ToolchainConfig,
    nvcc: Discovered<Nvcc>,
    python: Discovered<Python>,
    wasm: Discovered<Wasm>,
}

/// A toolchain the host may not have.
type Discovered<T> = std::result::Result<T, String>;

impl Toolchains {
    /// Discovers the host's toolchains and their versions.
    ///
    /// A missing toolchain is recorded with its reason; it does not fail the
    /// server, because only a build that needs it does.
    pub fn discover(config: &ToolchainConfig) -> Result<Toolchains> {
        Ok(Toolchains {
            config: config.clone(),
            nvcc: Nvcc::discover(config),
            python: Python::discover(config),
            wasm: Wasm::discover(config),
        })
    }

    /// The toolchain of a kind, with the host's configuration.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Toolchain`] when the host has no toolchain of that
    /// kind, or it cannot report its version.
    pub fn get(&self, kind: ToolchainKind) -> Result<&dyn Toolchain> {
        match kind {
            ToolchainKind::Nvcc => self.nvcc.as_ref().map(|toolchain| toolchain as &dyn Toolchain),
            ToolchainKind::Python => self
                .python
                .as_ref()
                .map(|toolchain| toolchain as &dyn Toolchain),
            ToolchainKind::Wasm => self.wasm.as_ref().map(|toolchain| toolchain as &dyn Toolchain),
        }
        .map_err(|reason| Error::Toolchain {
            kind,
            message: reason.clone(),
        })
    }

    /// The toolchain's version, for `list` and the logs; a missing toolchain
    /// answers its reason.
    pub fn version(&self, kind: ToolchainKind) -> std::result::Result<&str, &str> {
        match kind {
            ToolchainKind::Nvcc => self
                .nvcc
                .as_ref()
                .map(|toolchain| toolchain.version())
                .map_err(String::as_str),
            ToolchainKind::Python => self
                .python
                .as_ref()
                .map(|toolchain| toolchain.version())
                .map_err(String::as_str),
            ToolchainKind::Wasm => self
                .wasm
                .as_ref()
                .map(|toolchain| toolchain.version())
                .map_err(String::as_str),
        }
    }

    /// The host's configuration of the toolchains.
    pub fn config(&self) -> &ToolchainConfig {
        &self.config
    }
}

/// nvcc, the CUDA toolkit's compiler.
#[derive(Debug)]
struct Nvcc {
    path: PathBuf,
    version: String,
    max_jobs: usize,
    /// The interpreter of the Program Server's Python environment, which runs
    /// a CUDA package's compile script when the package names one.
    interpreter: Option<PathBuf>,
}

impl Nvcc {
    fn discover(config: &ToolchainConfig) -> Discovered<Nvcc> {
        let path = config
            .nvcc
            .clone()
            .or_else(|| find("nvcc"))
            .or_else(|| existing("/usr/local/cuda/bin/nvcc"))
            .ok_or_else(|| "no nvcc on this host: install the CUDA toolkit".to_owned())?;
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
        let interpreter = resolve(&config.python);
        Ok(Nvcc {
            path,
            version,
            max_jobs: config.max_jobs,
            interpreter: interpreter.is_file().then_some(interpreter),
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

    fn run(&self, job: ToolJob) -> Result<()> {
        execute(&job, self.max_jobs)
    }

    fn max_jobs(&self) -> usize {
        self.max_jobs
    }

    fn executable(&self) -> &Path {
        &self.path
    }

    fn script_interpreter(&self) -> Option<&Path> {
        self.interpreter.as_deref()
    }
}

/// The Program Server's Python environment: Triton, CuTe DSL, and the
/// precompiled wasmtime the scheduling policies are stored for.
///
/// Its version is Triton's, the environment's principal compiler: a package
/// declares that version in `[toolchain] version`, and the versions of the
/// other distributions it imports in `[toolchain] packages`.
#[derive(Debug)]
struct Python {
    interpreter: PathBuf,
    version: String,
    max_jobs: usize,
}

impl Python {
    fn discover(config: &ToolchainConfig) -> Discovered<Python> {
        let interpreter = resolve(&config.python);
        if !interpreter.is_file() {
            return Err(format!(
                "no Python environment at {}: create it from crates/program-server/python/pyproject.toml",
                interpreter.display()
            ));
        }
        let reported = Command::new(&interpreter)
            .args(["-c", "import triton; print(triton.__version__)"])
            .output()
            .map_err(|error| format!("running {}: {error}", interpreter.display()))?;
        let text = String::from_utf8_lossy(&reported.stdout).trim().to_owned();
        let error = String::from_utf8_lossy(&reported.stderr).trim().to_owned();
        if !reported.status.success() {
            return Err(format!(
                "{} cannot import Triton: {error}",
                interpreter.display()
            ));
        }
        if text.is_empty() {
            return Err(format!(
                "{} reported no Triton version",
                interpreter.display()
            ));
        }
        Ok(Python {
            interpreter,
            version: text,
            max_jobs: config.max_jobs,
        })
    }
}

impl Toolchain for Python {
    fn version(&self) -> &str {
        &self.version
    }

    fn peak_host_bytes(&self, _op: &str, _key: &str) -> u64 {
        PYTHON_PEAK_HOST_BYTES
    }

    fn run(&self, job: ToolJob) -> Result<()> {
        execute(&job, self.max_jobs)
    }

    fn max_jobs(&self) -> usize {
        self.max_jobs
    }

    fn executable(&self) -> &Path {
        &self.interpreter
    }
}

/// The WebAssembly toolchain: cargo builds a policy for wasm32, and wasmtime
/// precompiles it for this host's CPU (design section 5.3).
#[derive(Debug)]
struct Wasm {
    wasmtime: PathBuf,
    version: String,
    max_jobs: usize,
}

impl Wasm {
    fn discover(config: &ToolchainConfig) -> Discovered<Wasm> {
        let wasmtime = config
            .wasmtime
            .clone()
            .or_else(|| find("wasmtime"))
            .ok_or_else(|| {
                "no wasmtime on this host: install it to precompile scheduling policies".to_owned()
            })?;
        let reported = Command::new(&wasmtime)
            .arg("--version")
            .output()
            .map_err(|error| format!("running {}: {error}", wasmtime.display()))?;
        let text = String::from_utf8_lossy(&reported.stdout);
        if !reported.status.success() {
            return Err(format!(
                "{} --version failed: {}",
                wasmtime.display(),
                text.trim()
            ));
        }
        let version = text
            .split_whitespace()
            .last()
            .filter(|version| !version.is_empty())
            .ok_or_else(|| format!("{} --version names no version", wasmtime.display()))?
            .to_owned();
        Ok(Wasm {
            wasmtime,
            version,
            max_jobs: config.max_jobs,
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

    fn run(&self, job: ToolJob) -> Result<()> {
        execute(&job, self.max_jobs)
    }

    fn max_jobs(&self) -> usize {
        self.max_jobs
    }

    fn executable(&self) -> &Path {
        &self.wasmtime
    }
}

/// The version nvcc reports, such as `13.0.48` from
/// `Cuda compilation tools, release 13.0, V13.0.48`.
fn parse_nvcc_version(text: &str) -> Option<String> {
    let release = text.split("release ").nth(1)?;
    let after = release.split(',').nth(1)?.trim();
    let version = after.strip_prefix('V').unwrap_or(after).trim();
    if version.is_empty() {
        None
    } else {
        Some(version.to_owned())
    }
}

/// The first `name` on `PATH`.
fn find(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

/// A path that exists, if the file is there.
fn existing(path: &str) -> Option<PathBuf> {
    let path = PathBuf::from(path);
    path.is_file().then_some(path)
}

/// Resolves a configured program the way a command started from the
/// workspace root would: as given; and, when it is relative and not there,
/// from each directory above the current one, so the default
/// `python/.venv/bin/python` is found whether the server was started at the
/// repository root or inside `crates/program-server`.
fn resolve(configured: &Path) -> PathBuf {
    if configured.is_absolute() || configured.is_file() {
        return configured.to_path_buf();
    }
    if let Ok(current) = std::env::current_dir() {
        for ancestor in current.ancestors() {
            let candidate = ancestor.join(configured);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    configured.to_path_buf()
}

/// The environment a compiler process runs with: the toolchain's own, with
/// `MAX_JOBS` set and the job's variables added.
pub fn environment(job: &ToolJob, max_jobs: usize) -> Vec<(OsString, OsString)> {
    let mut environment: Vec<(OsString, OsString)> =
        vec![("MAX_JOBS".into(), OsString::from(max_jobs.to_string()))];
    environment.extend(job.environment.iter().cloned());
    environment
}

/// Runs a compiler process, writing its output to the job's log.
pub fn execute(job: &ToolJob, max_jobs: usize) -> Result<()> {
    let log = std::fs::File::create(&job.log)?;
    let error_log = log.try_clone()?;
    let status = Command::new(&job.program)
        .args(&job.arguments)
        .current_dir(&job.directory)
        .envs(environment(job, max_jobs))
        .stdout(log)
        .stderr(error_log)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        let text = std::fs::read_to_string(&job.log).unwrap_or_default();
        Err(Error::Build {
            op: String::new(),
            backend: String::new(),
            revision: 0,
            target: String::new(),
            descriptor: String::new(),
            message: format!("{} failed: {status}: {text}", job.program.display()),
        })
    }
}

/// The program a toolchain runs for a script: the Python interpreter of the
/// Program Server's environment.
pub fn interpreter(config: &ToolchainConfig) -> &Path {
    &config.python
}

#[cfg(test)]
mod tests {
    use super::parse_nvcc_version;

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
