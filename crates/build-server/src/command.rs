//! The `uniserve build-server` command (design sections 6.1 and 6.3).
//!
//! Its subcommands:
//!
//! - `serve --deployment NAME`: runs the deployment's Build Server on this node
//!   until it is stopped. With `--daemon` it leaves its starter's session first
//!   (`service::host::detach`), which is how the deployment's daemon starts it
//!   when none serves (`service::host::start`).
//! - `install <package>`: installs one package on the upstream
//!   Build Server and prints its revision. Only the upstream accepts it;
//!   another server names where the upstream is.
//! - `list`: prints the installed packages, their revisions and their
//!   modules.
//! - `resolve <op> <backend> <revision>`: prints where the store holds one
//!   module's files, building the module first when it is missing.
//! - `fields <package> <key>`: prints the fields a package's declared
//!   dimensions read a descriptor key into, for a tool that needs them outside
//!   a build.
//! - `launch <package> --target <target> [<key>...]`: prints, one JSON line
//!   per descriptor key, the compilation a Triton package's launch table
//!   states for the key, for a check that launches the package's kernel from
//!   its source outside a build ([`state_launches`]).
//! - `build --keys <file> --packages <dir> --package <dir>`: fills a store
//!   directory with the modules of descriptor key lists and of single-module
//!   packages (a policy, the engine), with a Build Server of its own for the
//!   length of the build ([`crate::build`]). The store's `packages` and
//!   `modules` are then a `serve`'s package root and store root, or the
//!   read-only store a release ships.
//!
//! The server launches no kernel and allocates no device memory (design
//! section 6.1). The targets it builds for are the `--target`s it is given,
//! or else the architectures of this host's devices, read from the driver's
//! device attributes without a context. Its toolchains are its own
//! (`crate::toolchain`): no option names a compiler. The source tree it is
//! given (`--source`, [`crate::tree`]) holds its Python environment and the
//! serving packages the upstream server installs when it starts.

use std::collections::BTreeMap;
use std::io::{BufRead as _, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::{Deserialize, Serialize};

use crate::builder::Limits;
use crate::launch::Launch;
use crate::manifest::{ModuleForm, SOURCE_DIRECTORY, ToolchainKind};
use crate::protocol::Client;
use crate::server::{BuildServer, ServerConfig};
use crate::store::Root;
use crate::target::Target;
use crate::toolchain::{ENVIRONMENT_PROGRAM, Environment, PeakMemory};
use crate::{Error, Manifest, Result};

/// The `build-server` command's arguments.
#[derive(Debug, clap::Parser)]
#[command(
    name = "build-server",
    about = "Compile packages, keep the modules, and hand them out"
)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

/// The `build-server` subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Run a deployment's Build Server on this node until stopped.
    Serve(Box<ServeArgs>),
    /// Install a package on the upstream Build Server.
    Install {
        #[command(flatten)]
        server: Address,
        /// The package directory, as the Build Server's host sees it.
        package: PathBuf,
        /// Print the installed package as one JSON object ([`InstalledPackage`])
        /// instead of its text line.
        #[arg(long)]
        json: bool,
    },
    /// List the installed packages, revisions and modules.
    List {
        #[command(flatten)]
        server: Address,
        /// Print the listing as one JSON array ([`ListedPackage`]) instead of
        /// its text rendering.
        #[arg(long)]
        json: bool,
    },
    /// Resolve one module and print the path of each of its files in the
    /// store, building the module first when the store misses it.
    Resolve {
        #[command(flatten)]
        server: Address,
        /// The interface the package implements, such as `engine`.
        op: String,
        /// The package's implementation name.
        backend: String,
        /// The package's revision.
        revision: u32,
        /// The target the module is built for; this host's platform when
        /// unset.
        #[arg(long, value_name = "NAME")]
        target: Option<String>,
        /// The module's key; the key of a single-module package when unset.
        #[arg(long, value_name = "KEY")]
        key: Option<String>,
        /// Print the module as one JSON object ([`ResolvedModule`]) instead of
        /// one file path per line.
        #[arg(long)]
        json: bool,
    },
    /// Read a descriptor key against a package's declared dimensions and
    /// print its fields as a JSON object, typed as a compilation is handed them.
    Fields {
        /// The package directory.
        package: PathBuf,
        /// The descriptor key.
        key: String,
    },
    /// Read descriptor keys against a Triton package's declared dimensions
    /// and print, one JSON line per key, the compilation its launch table
    /// states for the key and a target: the kernel, its signature in parameter
    /// order, the compile-time values, the aligned arguments' attributes and
    /// the options, or why the package states none.
    Launch {
        /// The source tree whose Build Server environment evaluates the
        /// launch table ([`crate::tree`]).
        #[arg(long, value_name = "DIR")]
        source: PathBuf,
        /// The package directory.
        package: PathBuf,
        /// The target the compilations are stated for, such as `sm_100a`.
        #[arg(long, value_name = "NAME")]
        target: String,
        /// The descriptor keys; read from standard input, one per line, when
        /// none is given, each answered before the next is read.
        keys: Vec<String>,
    },
    /// Build the modules of descriptor key lists and of single-module
    /// packages into a store directory.
    Build(Box<BuildArgs>),
}

/// The arguments of `build`.
#[derive(Debug, Clone, clap::Args)]
pub struct BuildArgs {
    /// A key list, `operator<TAB>descriptor key` per line, such as a
    /// deployment's kernel inventory (the `op` and `descriptor` labels of its
    /// `uniserve_kernel_launches_total` stats, read with `/admin/query`);
    /// repeatable.
    #[arg(long = "keys", value_name = "FILE")]
    pub keys: Vec<PathBuf>,
    /// The directory of a package that fills a whole interface with one
    /// module, a scheduling policy or the engine: it is installed and its
    /// module built for its own target; repeatable.
    #[arg(long = "package", value_name = "DIR")]
    pub package: Vec<PathBuf>,
    /// The package tree the key lists' operators are installed from: one
    /// directory per operator, one package directory per backend; required
    /// with `--keys`.
    #[arg(long, value_name = "DIR")]
    pub packages: Option<PathBuf>,
    /// The target the kernel keys' modules are built for, such as `sm_100a`;
    /// required with `--keys`.
    #[arg(long, value_name = "NAME")]
    pub target: Option<String>,
    /// The store directory: installed packages under `packages`, modules
    /// under `modules`. An existing store is extended.
    #[arg(long, value_name = "DIR")]
    pub store: PathBuf,
    /// The source tree whose Build Server environment compiles Triton, CuTe
    /// DSL and CUDA packages ([`crate::tree`]); without it only packages that
    /// need no such toolchain build.
    #[arg(long, value_name = "DIR")]
    pub source: Option<PathBuf>,
    /// How many builds may run at once, and how many compiler processes each
    /// may run (4 when unset).
    #[arg(long, value_name = "COUNT")]
    pub max_jobs: Option<usize>,
    /// The peak host memory one build of a toolchain reserves, as
    /// `KIND=BYTES`; repeatable.
    #[arg(long = "peak-memory", value_name = "KIND=BYTES", value_parser = peak_memory)]
    pub peak_memory: Vec<(ToolchainKind, u64)>,
}

/// The Build Server an operator's command reaches: the one on the socket it names, or the Build
/// Server of the deployment it names on this node ([`crate::socket`]).
#[derive(Debug, Clone, PartialEq, Eq, clap::Args)]
#[group(required = true, multiple = false)]
pub struct Address {
    /// The Build Server's socket.
    #[arg(long, value_name = "PATH")]
    pub socket: Option<PathBuf>,
    /// The deployment whose Build Server on this node is meant.
    #[arg(long, value_name = "NAME")]
    pub deployment: Option<String>,
}

impl Address {
    /// The socket the address names, made absolute.
    ///
    /// # Errors
    ///
    /// Fails when a deployment is named and the user has no runtime directory, or the name is no
    /// deployment's.
    pub fn socket(&self) -> std::io::Result<PathBuf> {
        match (&self.socket, &self.deployment) {
            (Some(socket), _) => Ok(absolute(socket.clone())),
            (None, Some(deployment)) => {
                crate::socket(&service::RuntimeDirectory::of_user()?, deployment)
            }
            (None, None) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "name the Build Server's deployment (--deployment) or its socket (--socket)",
            )),
        }
    }
}

/// The arguments of `serve`.
#[derive(Debug, Clone, clap::Args)]
pub struct ServeArgs {
    /// The deployment this Build Server builds for: its socket, packages, modules and caches are
    /// the deployment's own on this node.
    #[arg(long, value_name = "NAME")]
    pub deployment: String,
    /// The socket to listen on; the deployment's ([`crate::socket`]) when unset.
    #[arg(long, value_name = "PATH")]
    pub socket: Option<PathBuf>,
    /// Leave the starter's session, log beside the socket, and return to the
    /// starter once serving, printing the server's readiness record.
    #[arg(long)]
    pub daemon: bool,
    /// The inherited descriptor the server writes its readiness record to
    /// (`service::Readiness`, one JSON line) once it serves.
    #[arg(long, value_name = "FD")]
    pub ready_fd: Option<i32>,
    #[command(flatten)]
    pub settings: Settings,
}

/// How a Build Server serves: its roots, its upstream, its source tree, its targets and its
/// limits.
///
/// The same settings are `serve`'s flags and a deployment's `build_config` table. The
/// deployment's daemon starts the deployment's Build Server on its node with these settings
/// ([`Settings::arguments`]) when none serves there, and replaces one that serves with others.
#[derive(Debug, Clone, Default, PartialEq, Eq, clap::Args, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    /// The installed packages; writable only on the upstream server.
    /// `build-server/packages` in the deployment's data directory when unset.
    #[arg(long, value_name = "DIR")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_root: Option<PathBuf>,
    /// A module store root; the first is writable, the rest are read-only
    /// (design section 5.3). `build-server/modules` in the deployment's data
    /// directory when none is given.
    #[arg(long = "store-root", value_name = "DIR")]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub store_roots: Vec<PathBuf>,
    /// The upstream Build Server's socket, when this server is not it.
    #[arg(long, value_name = "PATH")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<PathBuf>,
    /// The UniServe source tree this server builds from ([`crate::tree`]):
    /// uv creates its Build Server environment, which compiles Triton, CuTe
    /// DSL and CUDA packages, and the upstream server installs its serving
    /// packages when it starts. Unset, the server has no such toolchain and
    /// installs nothing by itself.
    #[arg(long, value_name = "DIR")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<PathBuf>,
    /// A store this server serves and never writes: a directory `build
    /// --store` made, such as the one a release ships. Its packages and
    /// modules are served with no toolchain; repeatable.
    #[arg(long = "read-only-store", value_name = "DIR")]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub read_only_stores: Vec<PathBuf>,
    /// A target this host builds for, such as `sm_100a`; the architectures
    /// of this host's devices when none is given.
    #[arg(long = "target", value_name = "NAME")]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<String>,
    /// How many builds may run at once, and how many compiler processes each
    /// may run (4 when unset).
    #[arg(long, value_name = "COUNT")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_jobs: Option<usize>,
    /// Host memory, in bytes, that must stay free before a build starts
    /// (8 GiB when unset).
    #[arg(long, value_name = "BYTES")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_reserve: Option<u64>,
    /// The peak host memory one build of a toolchain reserves, in bytes, by
    /// toolchain kind (`--peak-memory KIND=BYTES`, repeatable); a kind not
    /// stated reserves its default ([`PeakMemory::default_for`]).
    #[arg(long = "peak-memory", value_name = "KIND=BYTES", value_parser = peak_memory)]
    #[serde(skip_serializing_if = "Vec::is_empty", with = "peaks")]
    pub peak_memory: Vec<(ToolchainKind, u64)>,
}

/// `peak_memory` as a deployment states it: a table of toolchain kind to bytes.
mod peaks {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::manifest::ToolchainKind;

    pub(super) fn serialize<S: Serializer>(
        peaks: &[(ToolchainKind, u64)],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let table: BTreeMap<&str, u64> = peaks
            .iter()
            .map(|(kind, bytes)| (kind.as_str(), *bytes))
            .collect();
        table.serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<(ToolchainKind, u64)>, D::Error> {
        let table = BTreeMap::<ToolchainKind, u64>::deserialize(deserializer)?;
        Ok(table.into_iter().collect())
    }
}

/// Reads a `--peak-memory` value, `KIND=BYTES`.
fn peak_memory(text: &str) -> std::result::Result<(ToolchainKind, u64), String> {
    let (kind, bytes) = text
        .split_once('=')
        .ok_or_else(|| format!("{text:?} is not KIND=BYTES"))?;
    let kind: ToolchainKind = serde_json::from_value(serde_json::Value::String(kind.to_owned()))
        .map_err(|error| format!("{kind:?} is no toolchain kind: {error}"))?;
    let bytes = bytes
        .parse::<u64>()
        .map_err(|error| format!("{bytes:?} is no number of bytes: {error}"))?;
    Ok((kind, bytes))
}

impl Settings {
    /// The `serve` flags that state these settings: a starter passes them after `--socket`
    /// and `--daemon` (`service::host::start`), with the deployment's name.
    pub fn arguments(&self) -> Vec<String> {
        let mut arguments = Vec::new();
        let mut push = |flag: &str, value: String| {
            arguments.push(format!("--{flag}"));
            arguments.push(value);
        };
        let path = |path: &Path| path.to_string_lossy().into_owned();

        if let Some(root) = &self.package_root {
            push("package-root", path(root));
        }
        for root in &self.store_roots {
            push("store-root", path(root));
        }
        if let Some(upstream) = &self.upstream {
            push("upstream", path(upstream));
        }
        if let Some(source) = &self.source {
            push("source", path(source));
        }
        for store in &self.read_only_stores {
            push("read-only-store", path(store));
        }
        for target in &self.targets {
            push("target", target.clone());
        }
        if let Some(jobs) = self.max_jobs {
            push("max-jobs", jobs.to_string());
        }
        if let Some(reserve) = self.host_reserve {
            push("host-reserve", reserve.to_string());
        }
        for (kind, bytes) in &self.peak_memory {
            push("peak-memory", format!("{kind}={bytes}"));
        }
        arguments
    }

    /// The server configuration these settings describe for a server of `deployment` building
    /// for `targets`.
    ///
    /// Every path is made absolute, and an unset root is the deployment's own:
    /// `build-server/packages` and `build-server/modules` in the deployment's directory of the
    /// user's data directory (`deployments/<name>`, [`service::deployment`]). The server's cache
    /// (the pinned sources, cargo's target directory and the builds' workspaces) is
    /// `build-server` in the deployment's directory of the user's cache directory
    /// ([`service::cache_directory`]): no deployment builds in or reads another's. A server of no
    /// deployment (`build`, which names its store) keeps its cache in the user's cache directory.
    ///
    /// # Errors
    ///
    /// Fails when the user's data or cache directory cannot be resolved, or `deployment` names
    /// no deployment's directory.
    pub fn config(&self, deployment: Option<&str>, targets: Vec<Target>) -> Result<ServerConfig> {
        let of = |directory: PathBuf| -> std::io::Result<PathBuf> {
            let directory = match deployment {
                Some(deployment) => service::deployment(&directory, deployment)?,
                None => directory,
            };
            Ok(directory.join(crate::SERVICE))
        };
        let data = of(service::data_directory()?)?;
        let package_root = self
            .package_root
            .clone()
            .unwrap_or_else(|| data.join("packages"));
        let store_roots = if self.store_roots.is_empty() {
            vec![data.join("modules")]
        } else {
            self.store_roots.clone()
        };

        // The first store root is the one this host writes; the rest are
        // shared read-only roots (design section 5.3).
        let store_roots = store_roots
            .into_iter()
            .enumerate()
            .map(|(index, path)| Root {
                path: absolute(path),
                writable: index == 0,
            })
            .collect();

        let mut limits = Limits::default();
        if let Some(jobs) = self.max_jobs {
            limits.max_jobs = jobs;
            limits.peaks = PeakMemory::default_for(jobs);
        }
        if let Some(reserve) = self.host_reserve {
            limits.host_reserve = reserve;
        }
        for (kind, bytes) in &self.peak_memory {
            limits.peaks.set(*kind, *bytes);
        }

        Ok(ServerConfig {
            package_root: absolute(package_root),
            store_roots,
            upstream: self.upstream.clone().map(absolute),
            read_only_stores: self
                .read_only_stores
                .iter()
                .cloned()
                .map(absolute)
                .collect(),
            limits,
            targets,
            cache: of(service::cache_directory()?)?,
            source: self.source.clone().map(absolute),
        })
    }

    /// The targets this server builds for: each `--target`, or else the
    /// architecture of each of this host's devices, which `host` reads, once.
    ///
    /// # Errors
    ///
    /// Fails when no target is given and `host` cannot report the host's
    /// devices.
    pub fn targets(&self, host: HostTargets<'_>) -> Result<Vec<Target>> {
        if !self.targets.is_empty() {
            return Ok(self.targets.iter().cloned().map(Target::new).collect());
        }
        let mut targets: Vec<Target> = Vec::new();
        for target in host().map_err(Error::Protocol)? {
            if !targets.contains(&target) {
                targets.push(target);
            }
        }
        Ok(targets)
    }
}

/// Reads the architecture target of each of this host's devices
/// ([`Target::arch`]): what `serve` builds for when no `--target` names one.
///
/// The executable that runs the command links the device driver and supplies
/// this; the Build Server links none.
/// It fails with the reason the driver could not report a device.
pub type HostTargets<'a> = &'a dyn Fn() -> std::result::Result<Vec<Target>, String>;

/// Runs the command. `serve --daemon` forks, so the caller has started no
/// thread.
///
/// # Errors
///
/// Fails when the server cannot start, the package cannot be installed, or the
/// running server cannot be reached.
pub fn run(args: Args, host: HostTargets<'_>, levels: &service::logging::Levels) -> Result<()> {
    match args.command {
        Command::Serve(serve) => run_serve(*serve, host, levels),
        Command::Install {
            server,
            package,
            json,
        } => {
            let mut client = Client::connect(&server.socket()?)?;
            let installed = client.install(&absolute(package))?;
            if json {
                write_line(
                    &mut std::io::stdout().lock(),
                    &InstalledPackage {
                        op: &installed.op,
                        backend: &installed.backend,
                        revision: installed.revision,
                    },
                )?;
            } else {
                println!(
                    "{} {} revision {}",
                    installed.op, installed.backend, installed.revision
                );
            }
            Ok(())
        }
        Command::List { server, json } => {
            let mut client = Client::connect(&server.socket()?)?;
            let listing = client.list()?;
            if json {
                let packages: Vec<ListedPackage<'_>> = listing
                    .iter()
                    .map(|(package, origin, modules)| ListedPackage {
                        op: &package.op,
                        backend: &package.backend,
                        revision: package.revision,
                        commit: origin.commit.as_deref(),
                        clean: origin.clean,
                        modules: modules
                            .iter()
                            .map(|module| ListedModule {
                                target: module.target.to_string(),
                                descriptor: &module.descriptor,
                                form: module.form,
                                interface_version: module.interface_version,
                            })
                            .collect(),
                    })
                    .collect();
                write_line(&mut std::io::stdout().lock(), &packages)?;
                return Ok(());
            }
            for (package, origin, modules) in listing {
                // The commit the revision was installed from, marked when its tree was not clean.
                let commit = match (&origin.commit, origin.clean) {
                    (Some(commit), true) => format!(" commit {commit}"),
                    (Some(commit), false) => format!(" commit {commit} (modified)"),
                    (None, _) => String::new(),
                };
                println!(
                    "{} {} revision {}{commit}",
                    package.op, package.backend, package.revision
                );
                for module in modules {
                    println!(
                        "    {} {} {:?}",
                        module.target, module.descriptor, module.form
                    );
                }
            }
            Ok(())
        }
        Command::Resolve {
            server,
            op,
            backend,
            revision,
            target,
            key,
            json,
        } => {
            let mut client = Client::connect(&server.socket()?)?;
            let module = client.resolve(
                &op,
                &proto::module::PackageRevision {
                    backend,
                    revision,
                    arguments: toml::Table::new(),
                },
                &target.map_or_else(Target::host, Target::new),
                key.as_deref().unwrap_or(crate::manifest::SINGLE_KEY),
                &mut |progress| {
                    eprintln!("{} {:?}", progress.path, progress.state);
                },
            )?;
            if json {
                write_line(
                    &mut std::io::stdout().lock(),
                    &ResolvedModule {
                        form: module.form,
                        files: module
                            .files
                            .iter()
                            .map(|file| file.path.as_path())
                            .collect(),
                        executable: module
                            .executable
                            .as_deref()
                            .and_then(|name| module.files.iter().find(|file| file.name == name))
                            .map(|file| file.path.as_path()),
                        commit: module.origin.commit.as_deref(),
                        clean: module.origin.clean,
                    },
                )?;
                return Ok(());
            }
            for file in module.files {
                println!("{}", file.path.display());
            }
            Ok(())
        }
        Command::Fields { package, key } => {
            let manifest = load_manifest(&package)?;
            println!(
                "{}",
                serde_json::Value::Object(typed_fields(&manifest, &key)?)
            );
            Ok(())
        }
        Command::Launch {
            source,
            package,
            target,
            keys,
        } => {
            let environment =
                Environment::sync(&absolute(source)).map_err(|message| Error::Toolchain {
                    kind: ToolchainKind::Triton,
                    message,
                })?;
            let (package, target) = (absolute(package), Target::new(target));
            let mut out = std::io::stdout().lock();
            if keys.is_empty() {
                let lines = std::io::stdin().lock().lines();
                state_launches(&environment, &package, &target, lines, &mut out)
            } else {
                state_launches(
                    &environment,
                    &package,
                    &target,
                    keys.into_iter().map(Ok),
                    &mut out,
                )
            }
        }
        Command::Build(build) => run_build(*build, host),
    }
}

/// A package directory's manifest, or the reason it has none.
fn load_manifest(package: &Path) -> Result<Manifest> {
    Manifest::load(package).map_err(|source| Error::Manifest {
        package: package.to_owned(),
        source: Box::new(source),
    })
}

/// The fields `manifest`'s declared dimensions read `key` into, each typed as
/// its dimension declares it, or the refusal of a key the package does not
/// declare.
fn typed_fields(
    manifest: &Manifest,
    key: &str,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    manifest
        .typed_fields(key)
        .ok_or_else(|| Error::UnsupportedDescriptor {
            op: manifest.op.clone(),
            backend: manifest.backend.clone(),
            revision: 0,
            descriptor: key.to_owned(),
        })
}

/// The first line `uniserve-build-server launch` reads: what every descriptor
/// of the package shares.
///
/// The program states each compilation with the driver a build compiles
/// through (`uniserve_build_server.triton.stated`), so the launch a check takes
/// is the one the package's module is built with.
#[derive(Serialize)]
struct LaunchPackage<'a> {
    parameters: &'a BTreeMap<String, toml::Value>,
    /// The package's source directory, which holds the kernel module.
    source: PathBuf,
    /// The kernel module, `<op>`, as a build imports it.
    module: &'a str,
    launch: &'a Launch,
    /// The target's name: the program reads its hardware facts from the
    /// description this crate reads them from (`targets.toml`).
    target: &'a str,
}

/// Each later line `uniserve-build-server launch` reads: one descriptor, read
/// by the Build Server's reader of the key grammar.
#[derive(Serialize)]
struct LaunchDescriptor<'a> {
    descriptor: &'a str,
    /// The key's fields, typed as the manifest's dimensions declare them.
    fields: serde_json::Map<String, serde_json::Value>,
}

/// The line answering a key the package does not declare, in the shape of the
/// program's refusals.
#[derive(Serialize)]
struct Undeclared<'a> {
    key: &'a str,
    refused: String,
}

/// Prints, one JSON line per key of `keys`, the compilation a Triton package's
/// launch table states for the key on `target`: `{"key": ..., "launch": {...}}`
/// with the kernel's name, its signature in parameter order, the compile-time
/// values, the aligned arguments' attributes and `triton.compile`'s options, or
/// `{"key": ..., "refused": "..."}` for a key the package does not declare or
/// that the table or its derivation refuses.
///
/// Each key is read by the Build Server's own reader of the key grammar, as a
/// build reads it, and its table is evaluated by the Triton driver of the Build
/// Server's Python environment `environment` (`uniserve-build-server launch`),
/// which imports the package's kernel module from its source directory once. A
/// key is answered and `out` flushed before the next key is read, so a caller
/// may write keys one at a time to a running command and read each answer back.
///
/// # Errors
///
/// [`Error::Launch`] for a package without a launch table, a target that is no
/// GPU target, or a program that fails; [`Error::Toolchain`] when the program
/// cannot be run; [`Error::Io`] when a key cannot be read or an answer written.
pub fn state_launches(
    environment: &Environment,
    package: &Path,
    target: &Target,
    keys: impl IntoIterator<Item = std::io::Result<String>>,
    out: &mut impl Write,
) -> Result<()> {
    let manifest = load_manifest(package)?;
    let unstated = |message: String| Error::Launch {
        op: manifest.op.clone(),
        backend: manifest.backend.clone(),
        target: target.to_string(),
        message,
    };
    let launch = manifest
        .launch
        .as_ref()
        .ok_or_else(|| unstated("the package states no launch table ([launch])".to_owned()))?;
    if target.facts().is_none() {
        return Err(unstated(format!(
            "{target} is no GPU target of the description"
        )));
    }

    // The program's standard error stays the command's: a defect of the driver
    // reaches the caller as its own traceback.
    let mut program = environment
        .command()
        .arg("launch")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| Error::Toolchain {
            kind: ToolchainKind::Triton,
            message: format!(
                "{ENVIRONMENT_PROGRAM}, the program of the Build Server's Python environment, \
                 cannot be run: {error}"
            ),
        })?;
    // Both streams were asked for as pipes, so the child holds both.
    let (Some(mut descriptors), Some(answers)) = (program.stdin.take(), program.stdout.take())
    else {
        return Err(unstated(format!(
            "{ENVIRONMENT_PROGRAM} launch started without its pipes"
        )));
    };
    let mut answers = BufReader::new(answers);
    write_line(
        &mut descriptors,
        &LaunchPackage {
            parameters: &manifest.parameters,
            source: package.join(SOURCE_DIRECTORY),
            module: &manifest.op,
            launch,
            target: target.as_str(),
        },
    )?;

    for key in keys {
        let key = key?;
        let answer = match manifest.typed_fields(&key) {
            None => serde_json::to_string(&Undeclared {
                key: &key,
                refused: format!(
                    "{} {} declares no descriptor {key}",
                    manifest.op, manifest.backend
                ),
            })
            .map_err(std::io::Error::from)?,
            Some(fields) => {
                write_line(
                    &mut descriptors,
                    &LaunchDescriptor {
                        descriptor: &key,
                        fields,
                    },
                )?;
                let mut line = String::new();
                if answers.read_line(&mut line)? == 0 {
                    let status = program.wait()?;
                    return Err(unstated(format!(
                        "{ENVIRONMENT_PROGRAM} launch ended with {status} before it answered {key}"
                    )));
                }
                line.trim_end().to_owned()
            }
        };
        writeln!(out, "{answer}")?;
        out.flush()?;
    }

    // The program ends when its input does.
    drop(descriptors);
    let status = program.wait()?;
    if !status.success() {
        return Err(unstated(format!(
            "{ENVIRONMENT_PROGRAM} launch failed with {status}"
        )));
    }
    Ok(())
}

/// What `install --json` prints: the package the server installed and the revision it numbered.
#[derive(Serialize)]
pub struct InstalledPackage<'a> {
    pub op: &'a str,
    pub backend: &'a str,
    pub revision: u32,
}

/// One element of what `list --json` prints: an installed package revision, the commit of the
/// tree it was installed from and whether that tree was clean, and the modules its store holds.
#[derive(Serialize)]
pub struct ListedPackage<'a> {
    pub op: &'a str,
    pub backend: &'a str,
    pub revision: u32,
    pub commit: Option<&'a str>,
    pub clean: bool,
    pub modules: Vec<ListedModule<'a>>,
}

/// One module of a [`ListedPackage`]: the target it was built for, its descriptor key, its form
/// and the interface version it was built for.
#[derive(Serialize)]
pub struct ListedModule<'a> {
    pub target: String,
    pub descriptor: &'a str,
    pub form: ModuleForm,
    pub interface_version: u32,
}

/// What `resolve --json` prints: the module's form, the path of each of its files in the store,
/// for an executable module the file its process runs as, and the commit of the tree its package
/// revision was installed from with whether that tree was clean.
#[derive(Serialize)]
pub struct ResolvedModule<'a> {
    pub form: ModuleForm,
    pub files: Vec<&'a Path>,
    pub executable: Option<&'a Path>,
    pub commit: Option<&'a str>,
    pub clean: bool,
}

/// Writes `value` to `writer` as one JSON line and flushes it.
fn write_line(writer: &mut impl Write, value: &impl Serialize) -> std::io::Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

/// Builds the key lists' modules and the named packages' into the store and
/// prints one line per key: `ok`, the operator, the key and the backends that
/// built it, or `FAILED`, the operator and the key. Every line is
/// tab-separated.
fn run_build(args: BuildArgs, host: HostTargets<'_>) -> Result<()> {
    let invalid = |message: &str| {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            message.to_owned(),
        ))
    };
    let keys = crate::build::read_keys(&args.keys)?;
    if keys.is_empty() && args.package.is_empty() {
        return Err(invalid(
            "nothing to build: name a key list with --keys or a package with --package",
        ));
    }
    if !keys.is_empty() && args.target.is_none() {
        return Err(invalid(
            "kernel keys are built for a device's architecture: name it with --target",
        ));
    }
    let packages = match args.packages {
        Some(packages) => absolute(packages),
        None if keys.is_empty() => PathBuf::new(),
        None => {
            return Err(invalid(
                "kernel keys are built from a package tree: name it with --packages",
            ));
        }
    };

    // The store's two directories are the roots a `serve` is given.
    let store = absolute(args.store);
    std::fs::create_dir_all(&store)?;
    let settings = Settings {
        package_root: Some(store.join(crate::build::PACKAGES)),
        store_roots: vec![store.join(crate::build::MODULES)],
        targets: args.target.iter().cloned().collect(),
        source: args.source,
        max_jobs: args.max_jobs,
        peak_memory: args.peak_memory,
        ..Settings::default()
    };
    // A build of single-module packages alone names no device architecture and
    // reads none from the host.
    let targets = match args.target {
        Some(_) => settings.targets(host)?,
        None => Vec::new(),
    };
    let config = settings.config(None, targets)?;

    let single: Vec<PathBuf> = args.package.into_iter().map(absolute).collect();
    let total = keys.len() + single.len();
    let mut unbuilt = Vec::new();
    let failures = crate::build::build(
        &config,
        &packages,
        &keys,
        args.target.map(Target::new).as_ref(),
        &single,
        &mut |op, key, backends| {
            if backends.is_empty() {
                println!("FAILED\t{op}\t{key}");
                unbuilt.push(format!("{op} {key}"));
            } else {
                let named: Vec<String> = backends
                    .iter()
                    .map(|(backend, files)| format!("{backend} ({files} files)"))
                    .collect();
                println!("ok\t{op}\t{key}\t{}", named.join(", "));
            }
        },
    )?;

    println!(
        "\n{} of {total} modules built into the store at {}",
        total - unbuilt.len(),
        config.store_roots[0].path.display()
    );
    if failures.is_empty() {
        return Ok(());
    }
    eprintln!("\n{} failure(s):", failures.len());
    for failure in &failures {
        eprintln!("  {failure}");
    }
    Err(Error::Unbuilt {
        keys: unbuilt,
        total,
        failures: failures.len(),
    })
}

/// Serves until the process is stopped.
fn run_serve(
    serve: ServeArgs,
    host: HostTargets<'_>,
    levels: &service::logging::Levels,
) -> Result<()> {
    let socket = match &serve.socket {
        Some(socket) => absolute(socket.clone()),
        None => crate::socket(&service::RuntimeDirectory::of_user()?, &serve.deployment)?,
    };
    let inherited = serve
        .ready_fd
        // SAFETY: the starter passed this descriptor for the readiness record; nothing else in
        // this process owns it.
        .map(|fd| unsafe { service::ReadyFd::inherit(fd) })
        .transpose()?;

    // Forked before any thread exists.
    let mut ready = if serve.daemon {
        Some(service::host::detach(&socket, inherited)?)
    } else {
        inherited
    };
    service::logging::init("BuildServer", levels, service::logging::Output::Stderr);
    // A stop signal makes this descriptor readable, which ends the serving.
    let (stop, _flag) = service::stop_signals()?;

    let (listener, starting) = match service::host::bind::<Settings>(&socket)? {
        service::host::Bound::Owner { listener, starting } => (listener, starting),
        service::host::Bound::Serving(record) => {
            tracing::info!(socket = %socket.display(), pid = record.pid, "another build server serves this socket");
            if let Some(ready) = &mut ready {
                ready.report(&record)?;
            }
            return Ok(());
        }
    };

    let config = serve
        .settings
        .config(Some(&serve.deployment), serve.settings.targets(host)?)?;
    let server = std::sync::Arc::new(BuildServer::open(&config)?);
    // The upstream installs its source tree's serving packages before it
    // serves, so the first deployment that asks selects from them.
    if let (Some(source), None) = (&config.source, &config.upstream) {
        server.install_tree(source)?;
    }
    let record = crate::readiness(&socket, &serve.settings);
    let lock = starting.finish(&record)?;
    if let Some(mut ready) = ready {
        ready.report(&record)?;
    }
    tracing::info!(socket = %socket.display(), "build server serving");
    let served = server.serve_on(listener, lock, &stop);
    tracing::info!("build server stopped");
    served
}

/// A path made absolute against the directory the command was started in:
/// every path the command takes is absolute before it reaches the server,
/// whose builds run in other directories (the store's copy of a package).
fn absolute(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    std::path::absolute(&path).unwrap_or(path)
}
