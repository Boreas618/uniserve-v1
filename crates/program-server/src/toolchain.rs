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

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::manifest::ToolchainKind;
use crate::{Error, Result};

/// One compiler process a source runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolJob {
    /// The program to run: the interpreter, nvcc, rustc or wasmtime.
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

/// What compiles a package.
pub trait Toolchain: Send + Sync {
    /// The compiler's version, such as nvcc's `13.0` or Triton's `3.7.1`.
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
}

/// Where each toolchain is on this host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolchainConfig {
    /// nvcc, when the host has a CUDA toolkit.
    pub nvcc: Option<PathBuf>,
    /// The Program Server's Python environment: the interpreter under
    /// `python/.venv`, created from `python/pyproject.toml` and
    /// `python/uv.lock`.
    pub python: PathBuf,
    /// wasmtime, when the host can precompile wasm modules.
    pub wasmtime: Option<PathBuf>,
    /// The number of compilers that may run at once (`limits.max_jobs`).
    pub max_jobs: usize,
}

/// The host's toolchains.
#[derive(Debug)]
pub struct Toolchains {
    config: ToolchainConfig,
}

impl Toolchains {
    /// Discovers the host's toolchains and their versions.
    pub fn discover(config: &ToolchainConfig) -> Result<Toolchains> {
        Ok(Toolchains {
            config: config.clone(),
        })
    }

    /// The toolchain of a kind, with the host's configuration.
    pub fn get(&self, kind: ToolchainKind) -> Result<&dyn Toolchain> {
        self.toolchain(kind)
    }

    fn toolchain(&self, _kind: ToolchainKind) -> Result<&dyn Toolchain> {
        todo!("discover nvcc, the Python environment and wasmtime, and report their versions")
    }
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
