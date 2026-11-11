//! The `uniserve build-server` command (design sections 6.1 and 6.3).
//!
//! Its subcommands:
//!
//! - `serve`: runs this host's Build Server until it is stopped. With
//!   `--daemon` it leaves its starter's session first (`service::detach`),
//!   which is how a daemon starts the Build Server of a host that has none
//!   (`service::ensure`).
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
//! - `build --keys <file> --package <dir>`: fills a store directory with the
//!   modules of descriptor key lists and of single-module packages (a policy,
//!   the engine), with a Build Server of its own for the length of the build
//!   ([`crate::build`]). The store's `packages` and `modules` are then a
//!   `serve`'s package root and store root, or the read-only store a release
//!   ships.
//!
//! The server launches no kernel and allocates no device memory (design
//! section 6.1). The targets it builds for are the `--target`s it is given,
//! or else the architectures of this host's devices, read from the driver's
//! device attributes without a context.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::builder::Limits;
use crate::protocol::Client;
use crate::server::{BuildServer, ServerConfig};
use crate::store::Root;
use crate::target::{Device, Target};
use crate::toolchain::ToolchainConfig;
use crate::{Error, Manifest, Result, default_socket};

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
    /// Run this host's Build Server until stopped.
    Serve(Box<ServeArgs>),
    /// Install a package on the upstream Build Server.
    Install {
        /// The Build Server's socket; this user's default socket when unset.
        #[arg(long)]
        socket: Option<PathBuf>,
        /// The package directory, as the Build Server's host sees it.
        package: PathBuf,
    },
    /// List the installed packages, revisions and modules.
    List {
        /// The Build Server's socket; this user's default socket when unset.
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Resolve one module and print the path of each of its files in the
    /// store, building the module first when the store misses it.
    Resolve {
        /// The Build Server's socket; this user's default socket when unset.
        #[arg(long)]
        socket: Option<PathBuf>,
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
    },
    /// Read a descriptor key against a package's declared dimensions and
    /// print its fields as a JSON object, as a build script is handed them.
    Fields {
        /// The package directory.
        package: PathBuf,
        /// The descriptor key.
        key: String,
    },
    /// Build the modules of descriptor key lists and of single-module
    /// packages into a store directory.
    Build(Box<BuildArgs>),
}

/// The arguments of `build`.
#[derive(Debug, Clone, clap::Args)]
pub struct BuildArgs {
    /// A key list, `operator<TAB>descriptor key` per line; repeatable.
    #[arg(long = "keys", value_name = "FILE")]
    pub keys: Vec<PathBuf>,
    /// The directory of a package that fills a whole interface with one
    /// module, a scheduling policy or the engine: it is installed and its
    /// module built for its own target; repeatable.
    #[arg(long = "package", value_name = "DIR")]
    pub package: Vec<PathBuf>,
    /// The package tree: one directory per operator, one package directory
    /// per backend. Relative to the directory the command runs in.
    #[arg(long, value_name = "DIR", default_value = "crates/ops/kernels")]
    pub packages: PathBuf,
    /// The target the kernel keys' modules are built for, such as `sm_100a`;
    /// required with `--keys`.
    #[arg(long, value_name = "NAME")]
    pub target: Option<String>,
    /// The store directory: installed packages under `packages`, modules
    /// under `modules`. An existing store is extended.
    #[arg(long, value_name = "DIR")]
    pub store: PathBuf,
    /// The interpreter of the Build Server's Python environment, which
    /// Triton and CuTe DSL packages build with.
    #[arg(long, value_name = "PATH")]
    pub python: Option<PathBuf>,
    /// The CUDA toolkit's nvcc; `$CUDA_HOME/bin/nvcc` when unset.
    #[arg(long, value_name = "PATH")]
    pub nvcc: Option<PathBuf>,
    /// The precompiler that turns a scheduling policy into a module for this
    /// host.
    #[arg(long, value_name = "PATH")]
    pub wasm: Option<PathBuf>,
    /// cargo, which builds the engine package; the `cargo` on `PATH` when
    /// unset.
    #[arg(long, value_name = "PATH")]
    pub cargo: Option<PathBuf>,
    /// How many builds may run at once (4 when unset).
    #[arg(long, value_name = "COUNT")]
    pub max_jobs: Option<usize>,
}

/// The arguments of `serve`.
#[derive(Debug, Clone, clap::Args)]
pub struct ServeArgs {
    /// Leave the starter's session, log beside the socket, and return to the
    /// starter once listening.
    #[arg(long)]
    pub daemon: bool,
    #[command(flatten)]
    pub settings: Settings,
}

/// How a Build Server serves: its socket, its roots, its upstream, its
/// targets, its limits and its toolchains.
///
/// The same settings are `serve`'s flags and a deployment's `build_config`
/// table. A daemon prepares an epoch's package revisions at [`Settings::socket`], and
/// starts the Build Server of a host that serves none there with these
/// settings ([`Settings::arguments`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, clap::Args, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    /// The socket to listen on; this user's default socket when unset.
    #[arg(long, value_name = "PATH")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub socket: Option<PathBuf>,
    /// The installed packages; writable only on the upstream server.
    /// `build-server/packages` in this user's data directory when unset.
    #[arg(long, value_name = "DIR")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_root: Option<PathBuf>,
    /// A module store root; the first is writable, the rest are read-only
    /// (design section 5.3). `build-server/modules` in this user's data
    /// directory when none is given.
    #[arg(long = "store-root", value_name = "DIR")]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub store_roots: Vec<PathBuf>,
    /// The upstream Build Server's socket, when this server is not it.
    #[arg(long, value_name = "PATH")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<PathBuf>,
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
    /// How many builds may run at once (4 when unset).
    #[arg(long, value_name = "COUNT")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_jobs: Option<usize>,
    /// Host memory, in bytes, that must stay free before a build starts.
    #[arg(long, value_name = "BYTES")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_reserve: Option<u64>,
    /// The CUDA toolkit's nvcc; `$CUDA_HOME/bin/nvcc` when unset.
    #[arg(long, value_name = "PATH")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvcc: Option<PathBuf>,
    /// The interpreter of the Build Server's Python environment, which
    /// Triton and CuTe DSL packages build with.
    #[arg(long, value_name = "PATH")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub python: Option<PathBuf>,
    /// The precompiler that turns a scheduling policy into a module for this
    /// host.
    #[arg(long, value_name = "PATH")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wasm: Option<PathBuf>,
    /// cargo, which builds a policy crate for wasm32.
    #[arg(long, value_name = "PATH")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cargo: Option<PathBuf>,
}

impl Settings {
    /// The socket these settings name, made absolute, or this user's default
    /// socket.
    pub fn socket(&self) -> PathBuf {
        socket_or_default(self.socket.clone())
    }

    /// The `serve` flags that state these settings, but the socket: a starter
    /// passes them after `--socket` and `--daemon` (`service::ensure`).
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
        for (flag, value) in [
            ("nvcc", &self.nvcc),
            ("python", &self.python),
            ("wasm", &self.wasm),
            ("cargo", &self.cargo),
        ] {
            if let Some(value) = value {
                push(flag, path(value));
            }
        }
        arguments
    }

    /// The server configuration these settings describe for a server
    /// building for `devices`.
    ///
    /// Every path is made absolute, and an unset socket or root is this
    /// user's default one.
    pub fn config(&self, devices: Vec<Device>) -> ServerConfig {
        let data = data_directory();
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
        }
        if let Some(reserve) = self.host_reserve {
            limits.host_reserve = reserve;
        }

        ServerConfig {
            socket: self.socket(),
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
            toolchains: ToolchainConfig {
                nvcc: self.nvcc.clone().map(absolute),
                python: self.python.clone().map(absolute),
                wasm: self.wasm.clone().map(absolute),
                cargo: self.cargo.clone().map(absolute),
                max_jobs: limits.max_jobs,
            },
            devices,
        }
    }

    /// The devices this server builds for: one per `--target`, or else this
    /// host's devices by their architectures, which `host` reads.
    ///
    /// A `--target` names an architecture, and stands for the GPU model too:
    /// the command line does not know the model.
    ///
    /// # Errors
    ///
    /// Fails when no target is given and `host` cannot report the host's
    /// devices.
    pub fn devices(&self, host: HostTargets<'_>) -> Result<Vec<Device>> {
        if !self.targets.is_empty() {
            return Ok(self
                .targets
                .iter()
                .map(|target| Device {
                    arch: Target::new(target.clone()),
                    model: Target::new(target.clone()),
                })
                .collect());
        }

        // The host's devices, each architecture once.
        let mut devices: Vec<Device> = Vec::new();
        for target in host().map_err(Error::Protocol)? {
            if devices.iter().all(|device| device.arch != target) {
                devices.push(Device {
                    arch: target.clone(),
                    model: target,
                });
            }
        }
        Ok(devices)
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
pub fn run(args: Args, host: HostTargets<'_>) -> Result<()> {
    match args.command {
        Command::Serve(serve) => run_serve(*serve, host),
        Command::Install { socket, package } => {
            let mut client = Client::connect(&socket_or_default(socket))?;
            let installed = client.install(&absolute(package))?;
            println!(
                "{} {} revision {}",
                installed.package.op, installed.package.backend, installed.package.revision
            );
            Ok(())
        }
        Command::List { socket } => {
            let mut client = Client::connect(&socket_or_default(socket))?;
            let listing = client.list()?;
            for package in listing.packages {
                println!(
                    "{} {} revision {}",
                    package.op, package.backend, package.revision
                );
                for module in package.modules {
                    println!(
                        "    {} {} {:?}",
                        module.target, module.descriptor, module.form
                    );
                }
            }
            Ok(())
        }
        Command::Resolve {
            socket,
            op,
            backend,
            revision,
            target,
            key,
        } => {
            let mut client = Client::connect(&socket_or_default(socket))?;
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
            for file in module.files {
                println!("{}", file.path.display());
            }
            Ok(())
        }
        Command::Fields { package, key } => {
            let manifest = Manifest::load(&package).map_err(|source| Error::Manifest {
                package: package.clone(),
                source: Box::new(source),
            })?;
            let fields = manifest
                .fields(&key)
                .ok_or_else(|| Error::UnsupportedDescriptor {
                    op: manifest.op.clone(),
                    backend: manifest.backend.clone(),
                    revision: 0,
                    descriptor: key.clone(),
                })?;
            let fields: serde_json::Map<String, serde_json::Value> = fields
                .into_iter()
                .map(|(dimension, value)| (dimension, serde_json::Value::String(value)))
                .collect();
            println!("{}", serde_json::Value::Object(fields));
            Ok(())
        }
        Command::Build(build) => run_build(*build, host),
    }
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
    let python = args.python.map(absolute);
    if let Some(python) = python.as_ref().filter(|python| !python.is_file()) {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "no interpreter at {}: `uv sync --project crates/build-server/python` \
                 builds the Build Server's environment",
                python.display()
            ),
        )));
    }
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

    // The store's two directories are the roots a `serve` is given; the
    // build's own server listens on a socket inside the store while it runs.
    let store = absolute(args.store);
    std::fs::create_dir_all(&store)?;
    let settings = Settings {
        socket: Some(store.join("build-server.sock")),
        package_root: Some(store.join(crate::build::PACKAGES)),
        store_roots: vec![store.join(crate::build::MODULES)],
        targets: args.target.iter().cloned().collect(),
        max_jobs: args.max_jobs,
        nvcc: args.nvcc,
        python,
        wasm: args.wasm,
        cargo: args.cargo,
        ..Settings::default()
    };
    // A build of single-module packages alone names no device architecture and
    // reads none from the host.
    let devices = match args.target {
        Some(_) => settings.devices(host)?,
        None => Vec::new(),
    };
    let config = settings.config(devices);

    let single: Vec<PathBuf> = args.package.into_iter().map(absolute).collect();
    let total = keys.len() + single.len();
    let mut unbuilt = Vec::new();
    let failures = crate::build::build(
        &config,
        &absolute(args.packages),
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
fn run_serve(serve: ServeArgs, host: HostTargets<'_>) -> Result<()> {
    let socket = serve.settings.socket();

    // Forked before any thread exists.
    let ready = if serve.daemon {
        Some(service::detach(&socket)?)
    } else {
        None
    };
    service::logging::init("BuildServer", None, None, service::logging::Output::Stderr);
    let (_wake, stop) = service::stop_signals()?;

    let Some((listener, lock)) = service::bind(&socket)? else {
        tracing::info!(socket = %socket.display(), "another build server serves this socket");
        if let Some(ready) = ready {
            ready.signal();
        }
        return Ok(());
    };

    let config = serve.settings.config(serve.settings.devices(host)?);
    let server = std::sync::Arc::new(BuildServer::open(&config)?);
    if let Some(ready) = ready {
        ready.signal();
    }
    let served = server.serve_on(listener, lock, stop);
    tracing::info!("build server stopped");
    served
}

/// `socket`, made absolute, or this user's default socket.
fn socket_or_default(socket: Option<PathBuf>) -> PathBuf {
    socket.map(absolute).unwrap_or_else(default_socket)
}

/// The directory of this user's installed packages and module store:
/// `$XDG_DATA_HOME/uniserve/build-server`, else `~/.local/share/uniserve/build-server`.
fn data_directory() -> PathBuf {
    let data = match std::env::var_os("XDG_DATA_HOME") {
        Some(data) if !data.is_empty() => PathBuf::from(data),
        _ => std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".local/share"))
            .unwrap_or_else(|| PathBuf::from(".")),
    };
    data.join("uniserve").join("build-server")
}

/// A path made absolute against the directory the command was started in.
///
/// Every path the command takes is absolute before it reaches the server, because a
/// build runs with its working directory set to the store's copy of a package: a
/// relative toolchain path would name that copy instead of this host's compiler, and
/// the failure is an errno from a directory the caller never named.
///
/// `std::path::absolute` rather than `canonicalize`, which follows symlinks: the
/// environment's `python` is a symlink, and canonicalizing it makes toolchain
/// discovery read the host interpreter instead of the pinned one.
fn absolute(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    std::path::absolute(&path).unwrap_or(path)
}
