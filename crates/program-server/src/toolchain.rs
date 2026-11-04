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
//! - Triton, from the Program Server's own Python environment under `python/`
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

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::manifest::ToolchainKind;
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

/// One compiler process a source runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolJob {
    /// The program to run: the interpreter, nvcc or the wasm precompiler.
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
    /// A job of a program in `directory`, with its output going to `log`.
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
    /// The compiler's version, such as nvcc's `13.0.48`, Triton's `3.7.1` or
    /// CuTe DSL's `4.6.2`. A package declares the version it is written for;
    /// a mismatch refuses the build (design section 6.2). A toolchain that
    /// compiles nothing answers the empty string, which such a package
    /// declares.
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

    /// The toolchain's principal program: nvcc, or the wasm precompiler.
    /// `None` for a toolchain that compiles nothing.
    fn executable(&self) -> Option<&Path> {
        None
    }

    /// The interpreter that runs a package's Python compile script, when this
    /// toolchain has one.
    ///
    /// The Python toolchains are that interpreter. nvcc and the wasm
    /// toolchain also run a package's compile script, because a CUDA C++ or
    /// wasm32 package states its command line in a Python driver (design
    /// section 6.2: a source drives the package's compile script), so they
    /// carry the Python environment's interpreter when the host has it.
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
    /// `python/uv.lock`. Triton and CuTe DSL are the environment's two
    /// toolchains.
    pub python: PathBuf,
    /// The wasm precompiler, when the host can precompile wasm modules:
    /// `scheduling-precompile`, which builds the policy module with the
    /// schedule crate's own engine. `None` looks for it on `PATH`.
    pub wasm: Option<PathBuf>,
    /// The number of compilers that may run at once (`limits.max_jobs`).
    pub max_jobs: usize,
}

/// The host's toolchains, each discovered with its version, or the reason it
/// is missing.
#[derive(Debug)]
pub struct Toolchains {
    config: ToolchainConfig,
    nvcc: Discovered<Nvcc>,
    triton: Discovered<Triton>,
    cute_dsl: Discovered<CuteDsl>,
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
        let interpreter = resolve(&config.python);
        Ok(Toolchains {
            config: config.clone(),
            nvcc: Nvcc::discover(config),
            triton: Triton::discover(config, &interpreter),
            cute_dsl: CuteDsl::discover(config, &interpreter),
            wasm: Wasm::discover(config, &interpreter),
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
            ToolchainKind::None => Ok(&NOTHING as &dyn Toolchain),
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
            ToolchainKind::None => Ok(""),
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

    fn executable(&self) -> Option<&Path> {
        Some(&self.path)
    }

    fn script_interpreter(&self) -> Option<&Path> {
        self.interpreter.as_deref()
    }
}

/// Triton, from the Program Server's Python environment.
#[derive(Debug)]
struct Triton {
    interpreter: PathBuf,
    version: String,
    max_jobs: usize,
}

impl Triton {
    fn discover(config: &ToolchainConfig, interpreter: &Path) -> Discovered<Triton> {
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

    fn run(&self, job: ToolJob) -> Result<()> {
        execute(&job, self.max_jobs)
    }

    fn max_jobs(&self) -> usize {
        self.max_jobs
    }

    fn script_interpreter(&self) -> Option<&Path> {
        Some(&self.interpreter)
    }
}

/// CuTe DSL, from the Program Server's Python environment.
#[derive(Debug)]
struct CuteDsl {
    interpreter: PathBuf,
    version: String,
    max_jobs: usize,
}

impl CuteDsl {
    fn discover(config: &ToolchainConfig, interpreter: &Path) -> Discovered<CuteDsl> {
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

    fn run(&self, job: ToolJob) -> Result<()> {
        execute(&job, self.max_jobs)
    }

    fn max_jobs(&self) -> usize {
        self.max_jobs
    }

    fn script_interpreter(&self) -> Option<&Path> {
        Some(&self.interpreter)
    }
}

/// The wasm precompiler: the scheduling crate's program that precompiles a
/// wasm32 policy for this host's CPU (design section 5.3).
#[derive(Debug)]
struct Wasm {
    precompiler: PathBuf,
    version: String,
    max_jobs: usize,
    /// The interpreter of the Program Server's Python environment, which runs
    /// a policy package's compile script when the package names one.
    interpreter: Option<PathBuf>,
}

impl Wasm {
    fn discover(config: &ToolchainConfig, interpreter: &Path) -> Discovered<Wasm> {
        let precompiler = config
            .wasm
            .clone()
            .or_else(|| find("scheduling-precompile"))
            .ok_or_else(|| {
                "no wasm precompiler on this host: crates/scheduling's scheduling-precompile \
                 builds it"
                    .to_owned()
            })?;
        let version = wasmtime_version(&precompiler)?;
        Ok(Wasm {
            precompiler,
            version,
            max_jobs: config.max_jobs,
            interpreter: interpreter.is_file().then(|| interpreter.to_path_buf()),
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

    fn executable(&self) -> Option<&Path> {
        Some(&self.precompiler)
    }

    fn script_interpreter(&self) -> Option<&Path> {
        self.interpreter.as_deref()
    }
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

    fn run(&self, job: ToolJob) -> Result<()> {
        Err(Error::Build {
            op: String::new(),
            backend: String::new(),
            revision: 0,
            target: String::new(),
            descriptor: String::new(),
            message: format!(
                "nothing to run: this source compiles no program ({} was asked)",
                job.program.display()
            ),
        })
    }

    fn max_jobs(&self) -> usize {
        1
    }
}

/// Asks the Python environment for the version of one of its distributions.
fn python_version(interpreter: &Path, module: &str) -> Discovered<String> {
    if !interpreter.is_file() {
        return Err(format!(
            "no Python environment at {}: create it from crates/program-server/python/pyproject.toml",
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
/// `wasmtime` program beside it. A precompiled module is only loadable by the
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
    if let Some(wasmtime) = find("wasmtime")
        && let Ok(reported) = Command::new(wasmtime).arg("--version").output()
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

/// A program this host has on `PATH`, such as `cargo` for a wasm32 build.
pub fn find_program(name: &str) -> Option<PathBuf> {
    find(name)
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
