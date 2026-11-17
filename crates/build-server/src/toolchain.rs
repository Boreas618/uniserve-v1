//! Toolchains: what compiles a package (design section 6.2).
//!
//! The Build Server compiles with two declared toolchains, and finds neither by
//! guessing a path:
//!
//! - **Its Python environment** ([`Environment`]): the locked project
//!   `crates/build-server/python` of the source tree the server builds from
//!   (`--source`, [`crate::tree`]), which holds the Triton and CuTe DSL
//!   compilers and describes the CUDA toolkit, cuDNN and the host C++ compiler
//!   it resolved. When the server starts, uv creates the environment or brings
//!   it to the tree's lock, and the server runs the environment's program,
//!   `uniserve-build-server toolchain --json`, from the environment itself and
//!   keeps the description ([`Report`]): the versions, the environment's
//!   distributions, its interpreter and the digest of its compile driver.
//!   Every compilation of a Triton, CuTe DSL or CUDA package runs through that
//!   interpreter with the description handed back in its request
//!   (`crate::source`). A server started without a source tree has no
//!   environment.
//! - **cargo**, from rustup: the `cargo` program runs the toolchain the
//!   workspace's pinned `rust-toolchain.toml` names, and its release is read
//!   from the structured fields of `cargo -vV` in the workspace it builds.
//!
//! A toolchain reports its version, which is compared with the version a
//! package declares before the package is built, and an identity, which an
//! installed revision records with its inputs (`crate::inputs`). A source that
//! compiles nothing, a vendor cubin or an executable, needs no toolchain
//! ([`ToolchainKind::None`]).
//!
//! A toolchain the host does not have is not an error of the server: it is
//! discovered as missing, with the reason, and only a build that needs it
//! fails. That keeps a host without a CUDA toolkit able to serve the packages
//! it already has.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::manifest::{ToolchainDeps, ToolchainKind};
use crate::{Error, Result};

/// The program of the Build Server's Python environment.
pub const ENVIRONMENT_PROGRAM: &str = "uniserve-build-server";

/// uv, which creates the Python environment from its project's lock.
pub const UV: &str = "uv";

/// The Build Server's Python environment: the locked project of a source tree
/// ([`crate::tree::ENVIRONMENT_PROJECT`]) as uv installs it, in the project's
/// own `.venv`.
#[derive(Clone, Debug)]
pub struct Environment {
    /// The environment's program, [`ENVIRONMENT_PROGRAM`] in its `bin`.
    program: PathBuf,
}

impl Environment {
    /// The environment of the source tree `source`, created, or brought to the
    /// tree's lock, by `uv sync --locked`: an environment already at the lock
    /// is left as it is, so servers of one tree share it.
    ///
    /// # Errors
    ///
    /// Why there is no environment: uv cannot be run, or its sync fails, with
    /// what uv said.
    pub fn sync(source: &Path) -> std::result::Result<Environment, String> {
        let project = source.join(crate::tree::ENVIRONMENT_PROJECT);
        // The project's own `.venv`, whatever environment the starter runs in:
        // `uv run` names its own project's environment in `VIRTUAL_ENV`, and
        // `UV_PROJECT_ENVIRONMENT` would move this project's.
        let synced = Command::new(UV)
            .args(["sync", "--locked", "--project"])
            .arg(&project)
            .env_remove("VIRTUAL_ENV")
            .env_remove("UV_PROJECT_ENVIRONMENT")
            .output()
            .map_err(|error| {
                format!(
                    "the Build Server's Python environment cannot be created: {UV} cannot be \
                     run ({error}); install uv"
                )
            })?;
        if !synced.status.success() {
            return Err(format!(
                "{UV} sync --locked --project {} failed with {}: {}",
                project.display(),
                synced.status,
                String::from_utf8_lossy(&synced.stderr).trim()
            ));
        }
        Ok(Environment {
            program: project.join(".venv/bin").join(ENVIRONMENT_PROGRAM),
        })
    }

    /// A command of the environment's program.
    pub fn command(&self) -> Command {
        Command::new(&self.program)
    }
}

/// cargo, as rustup installs it: a proxy that runs the toolchain the
/// directory's `rust-toolchain.toml` names.
pub const CARGO: &str = "cargo";

/// A toolchain entry of the environment's description: what the host has, or
/// why it has none.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Entry<T> {
    Available(T),
    Missing(String),
}

impl<T> Entry<T> {
    /// What the host has, or a [`Error::Toolchain`] naming why it has none.
    fn require(&self, kind: ToolchainKind) -> Result<&T> {
        match self {
            Entry::Available(value) => Ok(value),
            Entry::Missing(reason) => Err(Error::Toolchain {
                kind,
                message: reason.clone(),
            }),
        }
    }
}

/// The CUDA toolkit the environment resolved through the pkg-config modules of
/// the CUDA release its lock fixes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cuda {
    /// nvcc's release and build, such as `13.0.48`.
    pub version: String,
    pub root: PathBuf,
    pub nvcc: PathBuf,
    pub include: PathBuf,
    pub lib: PathBuf,
    /// The driver API's link-time stub library's directory.
    pub stubs: PathBuf,
}

/// The cuDNN the environment resolved: the host distribution package of the
/// release its lock fixes, on the default search paths.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cudnn {
    /// The headers' release, such as `9.12.0`.
    pub version: String,
    pub include: PathBuf,
    pub lib: PathBuf,
}

/// The kernel packages' shared sources (`uniserve_kernels`, `crates/ops/python`)
/// as the environment installs them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kernels {
    /// The SHA-256 of the library's files, in hex: what a module is built
    /// with, and so part of its identity.
    pub digest: String,
    /// The directory of the C ABI headers the library ships, generated from
    /// the operator crate's call structures: the include directory of every
    /// native compilation. Where the environment installs it is no part of a
    /// module's identity; what it holds is, through the digest.
    pub include: PathBuf,
}

/// The toolchain description `uniserve-build-server toolchain --json` prints
/// (`uniserve_build_server.toolchain.Toolchain`), which a compilation is
/// handed back in its request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    /// The environment's interpreter, which runs every compilation.
    pub interpreter: PathBuf,
    /// The SHA-256 of the compile driver's files, in hex.
    pub driver: String,
    /// The kernel packages' shared sources: their digest and the include
    /// directory of their generated C ABI headers.
    pub kernels: Entry<Kernels>,
    /// Every installed distribution by its normalized name, with its version.
    pub distributions: BTreeMap<String, String>,
    pub triton: Entry<String>,
    pub cute_dsl: Entry<String>,
    pub cuda: Entry<Cuda>,
    pub cudnn: Entry<Cudnn>,
    pub host_compiler: Entry<PathBuf>,
}

impl Report {
    /// The environment's own toolchain description, read from its program.
    ///
    /// # Errors
    ///
    /// Why the environment cannot describe itself: its program cannot be run,
    /// fails, or prints no description.
    pub fn read(environment: &Environment) -> std::result::Result<Report, String> {
        let described = environment
            .command()
            .args(["toolchain", "--json"])
            .output()
            .map_err(|error| {
                format!(
                    "the Build Server's Python environment cannot be run ({}: {error})",
                    environment.program.display()
                )
            })?;
        if !described.status.success() {
            return Err(format!(
                "{ENVIRONMENT_PROGRAM} toolchain --json failed with {}: {}",
                described.status,
                String::from_utf8_lossy(&described.stderr).trim()
            ));
        }
        serde_json::from_slice(&described.stdout).map_err(|error| {
            format!("{ENVIRONMENT_PROGRAM} printed no toolchain description: {error}")
        })
    }

    /// The version of a toolchain of this environment.
    fn version(&self, kind: ToolchainKind) -> Result<&str> {
        match kind {
            ToolchainKind::Triton => self.triton.require(kind).map(String::as_str),
            ToolchainKind::CuteDsl => self.cute_dsl.require(kind).map(String::as_str),
            ToolchainKind::Nvcc => self.cuda.require(kind).map(|cuda| cuda.version.as_str()),
            ToolchainKind::Cargo | ToolchainKind::None => Err(Error::Toolchain {
                kind,
                message: "not a toolchain of the Python environment".into(),
            }),
        }
    }

    /// What a module built through this environment is built by, as one line:
    /// the compile driver's digest, the digest of the kernel packages' shared
    /// sources, the version of every compiler and library the environment
    /// describes, and the digest of the locked distributions
    /// ([`Self::distributions_digest`]). A build module may use any of them (a
    /// CuTe DSL kernel linked by nvcc, FlashInfer's sources), so a change of
    /// any is another identity.
    fn identity(&self) -> String {
        let entry = |entry: &Entry<String>| match entry {
            Entry::Available(version) => version.clone(),
            Entry::Missing(_) => "none".to_owned(),
        };
        let cuda = match &self.cuda {
            Entry::Available(cuda) => cuda.version.clone(),
            Entry::Missing(_) => "none".to_owned(),
        };
        let cudnn = match &self.cudnn {
            Entry::Available(cudnn) => cudnn.version.clone(),
            Entry::Missing(_) => "none".to_owned(),
        };
        let kernels = match &self.kernels {
            Entry::Available(kernels) => kernels.digest.clone(),
            Entry::Missing(_) => "none".to_owned(),
        };
        format!(
            "driver {}; kernels {kernels}; triton {}; cute_dsl {}; cuda {cuda}; cudnn {cudnn}; \
             distributions {}",
            self.driver,
            entry(&self.triton),
            entry(&self.cute_dsl),
            self.distributions_digest()
        )
    }

    /// The SHA-256 of the environment's distributions, each `name version` on
    /// a line in name order, in hex: the lock's state, which fixes the version
    /// of every distribution a package imports.
    fn distributions_digest(&self) -> String {
        let mut hasher = Sha256::new();
        for (name, version) in &self.distributions {
            hasher.update(format!("{name} {version}\n").as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
}

/// The peak host memory of one build of each toolchain, in bytes: what the
/// compile queue reserves before it starts a build (design section 6.2).
/// Each is a declared option (`--peak-memory KIND=BYTES`) with the default
/// [`PeakMemory::default`] documents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeakMemory {
    pub triton: u64,
    pub cute_dsl: u64,
    pub nvcc: u64,
    pub cargo: u64,
}

impl PeakMemory {
    /// The defaults for builds of `jobs` compiler processes each.
    ///
    /// Triton holds a kernel's IR, its LLVM module and ptxas at once: 8 GiB.
    /// The CuTe DSL traces the kernel in its interpreter and runs ptxas and the
    /// host link: 10 GiB. nvcc, cicc and ptxas over one translation unit:
    /// 4 GiB. cargo runs `jobs` rustc processes of up to 2 GiB each and then
    /// the thin-LTO link of the executable, which holds every crate's code:
    /// the larger of `jobs` × 2 GiB and 16 GiB.
    pub fn default_for(jobs: usize) -> Self {
        const GIB: u64 = 1 << 30;
        Self {
            triton: 8 * GIB,
            cute_dsl: 10 * GIB,
            nvcc: 4 * GIB,
            cargo: (jobs as u64 * 2 * GIB).max(16 * GIB),
        }
    }

    /// The reservation of one build of toolchain `kind`.
    pub fn of(&self, kind: ToolchainKind) -> u64 {
        match kind {
            ToolchainKind::Triton => self.triton,
            ToolchainKind::CuteDsl => self.cute_dsl,
            ToolchainKind::Nvcc => self.nvcc,
            ToolchainKind::Cargo => self.cargo,
            ToolchainKind::None => 0,
        }
    }

    /// Replaces the reservation of toolchain `kind`.
    pub fn set(&mut self, kind: ToolchainKind, bytes: u64) {
        match kind {
            ToolchainKind::Triton => self.triton = bytes,
            ToolchainKind::CuteDsl => self.cute_dsl = bytes,
            ToolchainKind::Nvcc => self.nvcc = bytes,
            ToolchainKind::Cargo => self.cargo = bytes,
            ToolchainKind::None => {}
        }
    }
}

/// The host's toolchains: the Python environment's description, or why the
/// host has none.
#[derive(Debug)]
pub struct Toolchains {
    environment: std::result::Result<Report, String>,
}

impl Toolchains {
    /// Creates the Python environment of the source tree `source` and reads its
    /// description once.
    ///
    /// No source tree, or an environment that cannot be created or describe
    /// itself, is recorded with its reason; it does not fail the server,
    /// because only a build that needs the environment does.
    pub fn discover(source: Option<&Path>) -> Toolchains {
        let environment = match source {
            Some(source) => {
                Environment::sync(source).and_then(|environment| Report::read(&environment))
            }
            None => Err(
                "this Build Server has no source tree (`--source`, `build_config.source`), \
                 whose Python environment compiles Triton, CuTe DSL and CUDA packages"
                    .to_owned(),
            ),
        };
        Toolchains::from_report(environment)
    }

    /// The toolchains a given description states, or none with the reason.
    pub fn from_report(environment: std::result::Result<Report, String>) -> Toolchains {
        Toolchains { environment }
    }

    /// The Python environment's description, which a compilation is handed.
    ///
    /// # Errors
    ///
    /// [`Error::Toolchain`] naming why the host has no environment.
    pub fn environment(&self, kind: ToolchainKind) -> Result<&Report> {
        self.environment
            .as_ref()
            .map_err(|reason| Error::Toolchain {
                kind,
                message: reason.clone(),
            })
    }

    /// What builds a package of toolchain `kind` on this host, as one line,
    /// read now: the toolchain's kind and version and, for the Python
    /// environment, its identity ([`Report`]). Two hosts, or one host before
    /// and after an upgrade or an edit of the driver or of a pin, state the
    /// same line only when they build the same bytes from the same package
    /// (`crate::inputs`). `workspace` is where a cargo package is built.
    ///
    /// # Errors
    ///
    /// [`Error::Toolchain`] when the host has no toolchain of that kind.
    pub fn identity(&self, kind: ToolchainKind, workspace: &Path) -> Result<String> {
        match kind {
            ToolchainKind::None => Ok(kind.to_string()),
            ToolchainKind::Cargo => Ok(format!("{kind} {}", cargo_release(workspace)?)),
            _ => {
                let environment = self.environment(kind)?;
                Ok(format!(
                    "{kind} {}; {}",
                    environment.version(kind)?,
                    environment.identity()
                ))
            }
        }
    }

    /// Refuses a build whose package imports a Python distribution this host's
    /// environment does not have (design section 5.2: a package names the
    /// distributions its sources import, and the environment's lock fixes
    /// their versions, which the environment's identity carries).
    ///
    /// A package that declares none needs no environment for this. Names are
    /// compared as PEP 503 normalizes them, so `flash_attn_4` and
    /// `flash-attn-4` are one distribution.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Toolchain`], naming every distribution that is
    /// missing, or why the environment's distributions could not be read.
    pub fn check_packages(&self, op: &str, backend: &str, toolchain: &ToolchainDeps) -> Result<()> {
        if toolchain.packages.is_empty() {
            return Ok(());
        }
        let installed = &self.environment(toolchain.kind)?.distributions;
        let missing = missing(&toolchain.packages, installed);
        if missing.is_empty() {
            Ok(())
        } else {
            Err(Error::Toolchain {
                kind: toolchain.kind,
                message: format!(
                    "the package {op} {backend} imports {} that this host's Python environment \
                     does not have",
                    missing.join(", ")
                ),
            })
        }
    }
}

/// A cargo command run in `directory`, whose `rust-toolchain.toml` selects the
/// release rustup runs.
///
/// A toolchain the server's own environment names (`RUSTUP_TOOLCHAIN`, which
/// `cargo run` and `cargo test` set for their children) would override that
/// file, so the command does not inherit it.
pub fn cargo(directory: &Path) -> Command {
    let mut command = Command::new(CARGO);
    command
        .current_dir(directory)
        .env_remove("RUSTUP_TOOLCHAIN");
    command
}

/// cargo's release in `workspace`, from the `release:` field of `cargo -vV`.
fn cargo_release(workspace: &Path) -> Result<String> {
    let missing = |message: String| Error::Toolchain {
        kind: ToolchainKind::Cargo,
        message,
    };
    let reported = cargo(workspace).arg("-vV").output().map_err(|error| {
        missing(format!(
            "{CARGO} cannot be run in {}: {error}; rustup installs it",
            workspace.display()
        ))
    })?;
    if !reported.status.success() {
        return Err(missing(format!(
            "{CARGO} -vV failed in {} with {}: {}",
            workspace.display(),
            reported.status,
            String::from_utf8_lossy(&reported.stderr).trim()
        )));
    }
    String::from_utf8_lossy(&reported.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("release: "))
        .map(|release| release.trim().to_owned())
        .ok_or_else(|| missing(format!("{CARGO} -vV states no release")))
}

/// A distribution name as PEP 503 normalizes it: lowercase, with every run of
/// `-`, `_` and `.` one `-`.
pub fn normalized(name: &str) -> String {
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

/// The declared distributions `installed` does not hold.
fn missing<'a>(declared: &'a [String], installed: &BTreeMap<String, String>) -> Vec<&'a str> {
    declared
        .iter()
        .filter(|name| !installed.contains_key(&normalized(name)))
        .map(String::as_str)
        .collect()
}

/// Runs a compiler process to its end, writing its output to `log`.
///
/// # Errors
///
/// [`Error::Build`] naming the executable, its arguments and the directory it
/// ran in when it cannot start (an errno alone says none of them, and the log
/// is empty because the process never ran), or its exit status and log when it
/// fails.
pub fn execute(mut command: Command, log: &Path) -> Result<()> {
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
    use super::*;

    /// A description as the environment's program prints it, with every
    /// toolchain available.
    fn report() -> Report {
        serde_json::from_value(serde_json::json!({
            "interpreter": "/env/bin/python",
            "driver": "ab".repeat(32),
            "kernels": {"available": {
                "digest": "cd".repeat(32), "include": "/env/uniserve_kernels/include"
            }},
            "distributions": {"triton": "3.7.1", "flash-attn-4": "4.0.0b30"},
            "triton": {"available": "3.7.1"},
            "cute_dsl": {"available": "4.6.2"},
            "cuda": {"available": {
                "version": "13.0.48", "root": "/cuda", "nvcc": "/cuda/bin/nvcc",
                "include": "/cuda/include", "lib": "/cuda/lib", "stubs": "/cuda/lib/stubs"
            }},
            "cudnn": {"available": {
                "version": "9.12.0", "include": "/usr/include", "lib": "/usr/lib"
            }},
            "host_compiler": {"missing": "no c++ on this host's search path"}
        }))
        .unwrap()
    }

    /// A declared distribution is looked up by its normalized name: the environment's spelling
    /// of a name does not matter, and a distribution it lacks is named as missing.
    #[test]
    fn a_declared_distribution_must_be_installed() {
        let installed: BTreeMap<String, String> =
            [("flash_attn_4", "4.0.0b30"), ("Torch", "2.13.0")]
                .into_iter()
                .map(|(name, version)| (normalized(name), version.to_owned()))
                .collect();
        let declared = |names: &[&str]| -> Vec<String> {
            names.iter().map(|name| (*name).to_owned()).collect()
        };
        assert!(missing(&declared(&["flash-attn-4", "torch"]), &installed).is_empty());
        assert_eq!(
            missing(
                &declared(&["flash-attn-4", "flashinfer-python"]),
                &installed
            ),
            ["flashinfer-python"]
        );
    }

    /// A toolchain's identity names its version and everything else the environment builds
    /// with: another driver or another compiler version is another identity, a missing
    /// toolchain is refused with the reason the environment gave, and a source that compiles
    /// nothing is `none` on any host.
    #[test]
    fn an_environment_identity_follows_its_driver_and_compilers() {
        let toolchains = Toolchains::from_report(Ok(report()));
        let here = Path::new(".");
        let first = toolchains.identity(ToolchainKind::Triton, here).unwrap();
        assert!(first.starts_with("triton 3.7.1; driver "), "{first}");
        assert!(first.contains("cuda 13.0.48"), "{first}");

        let edited = Toolchains::from_report(Ok(Report {
            driver: "cd".repeat(32),
            ..report()
        }));
        assert_ne!(edited.identity(ToolchainKind::Triton, here).unwrap(), first);
        let upgraded = Toolchains::from_report(Ok(Report {
            cute_dsl: Entry::Available("4.7.0".into()),
            ..report()
        }));
        assert_ne!(
            upgraded.identity(ToolchainKind::Triton, here).unwrap(),
            first
        );
        // What a package imports is the builder too: the kernel packages' shared sources and the
        // version the lock fixes of every distribution.
        let shared = Toolchains::from_report(Ok(Report {
            kernels: Entry::Available(Kernels {
                digest: "ef".repeat(32),
                include: "/env/uniserve_kernels/include".into(),
            }),
            ..report()
        }));
        assert_ne!(shared.identity(ToolchainKind::Triton, here).unwrap(), first);
        // Where the environment installs them is not: another installation of the same files
        // builds the same modules.
        let moved = Toolchains::from_report(Ok(Report {
            kernels: Entry::Available(Kernels {
                digest: "cd".repeat(32),
                include: "/elsewhere/uniserve_kernels/include".into(),
            }),
            ..report()
        }));
        assert_eq!(moved.identity(ToolchainKind::Triton, here).unwrap(), first);
        let relocked = Toolchains::from_report(Ok(Report {
            distributions: [("flash-attn-4".to_owned(), "4.0.0".to_owned())].into(),
            ..report()
        }));
        assert_ne!(
            relocked.identity(ToolchainKind::Triton, here).unwrap(),
            first
        );

        let none = Toolchains::from_report(Err("no environment".into()));
        let refused = none
            .identity(ToolchainKind::CuteDsl, here)
            .unwrap_err()
            .to_string();
        assert!(refused.contains("no environment"), "{refused}");
        assert_eq!(none.identity(ToolchainKind::None, here).unwrap(), "none");
    }

    /// A build whose compiler is not where the host said names the executable, its arguments and
    /// the directory it ran in: an errno alone says none of them and the log is empty, because
    /// the process never started.
    #[test]
    fn a_build_that_cannot_find_its_compiler_names_the_executable_and_where_it_ran() {
        let directory = tempfile::tempdir().unwrap();
        let log = directory.path().join("build.log");
        let missing = directory.path().join("missing-compiler");
        let mut command = Command::new(&missing);
        command.current_dir(directory.path()).arg("--version");
        let error = execute(command, &log).expect_err("a missing executable cannot run");
        let text = error.to_string();
        assert!(text.contains(&missing.display().to_string()), "{text}");
        assert!(text.contains("--version"), "{text}");
        assert!(
            text.contains(&directory.path().display().to_string()),
            "{text}"
        );
    }
}
