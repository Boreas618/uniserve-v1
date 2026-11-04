//! The `uniserve program-server` command (design sections 6.1 and 6.3).
//!
//! Three subcommands:
//!
//! - `serve`: runs this host's Program Server until it is stopped.
//! - `install <package>`: installs one program package on the upstream Program
//!   Server and prints its revision. Only the upstream accepts it; another
//!   server answers where the upstream is.
//! - `list`: prints the installed packages, their revisions and their
//!   modules.
//!
//! The server launches no kernel and allocates no device memory (design
//! section 6.1). It is told the devices this host builds for, because only the
//! executable queries the platform crate (design section 4.1); `--target`
//! names them.

use std::path::PathBuf;

use clap::{Arg, ArgAction, ArgMatches, Command};

use crate::builder::Limits;
use crate::protocol::Client;
use crate::server::{ProgramServer, ServerConfig};
use crate::store::Root;
use crate::target::{Device, Target};
use crate::toolchain::ToolchainConfig;
use crate::{Error, Result};

/// What the command was asked to do.
#[derive(Clone, Debug)]
pub struct Options {
    /// The Program Server's socket.
    pub socket: PathBuf,
    pub action: Action,
}

/// The subcommand.
#[derive(Clone, Debug)]
pub enum Action {
    Serve(ServeOptions),
    Install {
        /// The package directory, as the server host sees it.
        package: PathBuf,
    },
    List,
}

/// How to serve.
#[derive(Clone, Debug)]
pub struct ServeOptions {
    /// The package root: writable on the upstream Program Server, a read-only
    /// mount of it everywhere else (design section 6.2).
    pub package_root: PathBuf,
    /// The module store's roots: the first is writable, the others are
    /// shared read-only roots (design section 5.3).
    pub store_roots: Vec<PathBuf>,
    /// The upstream Program Server's socket, when this is not the upstream.
    pub upstream: Option<PathBuf>,
    /// The queue's limits (design section 6.2).
    pub limits: Limits,
    /// Where this host's toolchains are.
    pub toolchains: ToolchainConfig,
    /// The devices this host builds for (design section 5.3).
    pub devices: Vec<Device>,
}

impl ServeOptions {
    /// The server configuration these options describe.
    pub fn config(&self, socket: PathBuf) -> ServerConfig {
        let mut store_roots = self
            .store_roots
            .iter()
            .map(|path| Root {
                path: path.clone(),
                writable: false,
            })
            .collect::<Vec<_>>();
        if let Some(first) = store_roots.first_mut() {
            first.writable = true;
        }
        ServerConfig {
            socket,
            package_root: self.package_root.clone(),
            store_roots,
            upstream: self.upstream.clone(),
            limits: self.limits,
            toolchains: self.toolchains.clone(),
            devices: self.devices.clone(),
        }
    }
}

/// The command's arguments, for the `uniserve` executable to mount as its
/// `program-server` subcommand.
pub fn command() -> Command {
    let socket = Arg::new("socket").long("socket").value_name("PATH").help(
        "the Program Server's Unix socket (default \
             $XDG_RUNTIME_DIR/uniserve/program-server.sock)",
    );
    Command::new("program-server")
        .about("compile program packages, keep the modules, and hand them out")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("serve")
                .about("run this host's Program Server")
                .arg(socket.clone())
                .arg(
                    Arg::new("package-root")
                        .long("package-root")
                        .value_name("DIR")
                        .required(true)
                        .help(
                            "the installed program packages; writable only on the upstream server",
                        ),
                )
                .arg(
                    Arg::new("store-root")
                        .long("store-root")
                        .value_name("DIR")
                        .action(ArgAction::Append)
                        .required(true)
                        .help("a module store root; the first is writable, the rest are read-only"),
                )
                .arg(
                    Arg::new("upstream")
                        .long("upstream")
                        .value_name("PATH")
                        .help("the upstream Program Server's socket, when this server is not it"),
                )
                .arg(
                    Arg::new("target")
                        .long("target")
                        .value_name("NAME")
                        .action(ArgAction::Append)
                        .help("a target this host builds for, such as sm_100a"),
                )
                .arg(
                    Arg::new("max-jobs")
                        .long("max-jobs")
                        .value_name("COUNT")
                        .value_parser(clap::value_parser!(usize))
                        .help("how many builds may run at once (default 4)"),
                )
                .arg(
                    Arg::new("host-reserve")
                        .long("host-reserve")
                        .value_name("BYTES")
                        .value_parser(clap::value_parser!(u64))
                        .help("host memory that must stay free before a build starts"),
                )
                .arg(
                    Arg::new("nvcc")
                        .long("nvcc")
                        .value_name("PATH")
                        .help("the CUDA toolkit's nvcc"),
                )
                .arg(
                    Arg::new("python")
                        .long("python")
                        .value_name("PATH")
                        .help("the interpreter of the Program Server's Python environment"),
                )
                .arg(
                    Arg::new("wasmtime")
                        .long("wasmtime")
                        .value_name("PATH")
                        .help("the wasmtime binary that precompiles the scheduling policies"),
                ),
        )
        .subcommand(
            Command::new("install")
                .about("install a program package on the upstream Program Server")
                .arg(socket.clone())
                .arg(
                    Arg::new("package")
                        .value_name("DIR")
                        .required(true)
                        .help("the package directory, as the Program Server's host sees it"),
                ),
        )
        .subcommand(
            Command::new("list")
                .about("list the installed packages, revisions and modules")
                .arg(socket),
        )
}

/// What the parsed arguments ask for.
///
/// # Errors
///
/// Fails when a subcommand's arguments are not the ones [`command`] defines.
pub fn options(matches: &ArgMatches) -> Result<Options> {
    let (name, matches) = matches
        .subcommand()
        .ok_or_else(|| Error::Protocol("program-server needs a subcommand".into()))?;
    let socket = matches
        .get_one::<String>("socket")
        .map(PathBuf::from)
        .unwrap_or_else(default_socket);
    let action = match name {
        "serve" => {
            let mut limits = Limits::default();
            if let Some(jobs) = matches.get_one::<usize>("max-jobs") {
                limits.max_jobs = *jobs;
            }
            if let Some(reserve) = matches.get_one::<u64>("host-reserve") {
                limits.host_reserve = *reserve;
            }
            let devices = matches
                .get_many::<String>("target")
                .map(|targets| {
                    targets
                        .map(|target| Device {
                            arch: Target::new(target.clone()),
                            model: Target::new(target.clone()),
                        })
                        .collect()
                })
                .unwrap_or_default();
            Action::Serve(ServeOptions {
                package_root: required_path(matches, "package-root")?,
                store_roots: values(matches, "store-root"),
                upstream: matches.get_one::<String>("upstream").map(PathBuf::from),
                limits,
                toolchains: ToolchainConfig {
                    nvcc: matches.get_one::<String>("nvcc").map(PathBuf::from),
                    python: matches
                        .get_one::<String>("python")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("python/.venv/bin/python")),
                    wasmtime: matches.get_one::<String>("wasmtime").map(PathBuf::from),
                    max_jobs: limits.max_jobs,
                },
                devices,
            })
        }
        "install" => Action::Install {
            package: required_path(matches, "package")?,
        },
        "list" => Action::List,
        other => {
            return Err(Error::Protocol(format!(
                "program-server has no {other} subcommand"
            )));
        }
    };
    Ok(Options { socket, action })
}

/// Runs the command.
///
/// # Errors
///
/// Fails when the server cannot start, the package cannot be installed, or the
/// running server cannot be reached.
pub fn run(options: Options) -> Result<()> {
    match options.action {
        Action::Serve(serve) => {
            let server = std::sync::Arc::new(ProgramServer::open(&serve.config(options.socket))?);
            server.serve()
        }
        Action::Install { package } => {
            let mut client = Client::connect(&options.socket)?;
            let installed = client.install(&package)?;
            println!(
                "{} {} revision {}",
                installed.package.op, installed.package.backend, installed.package.revision
            );
            Ok(())
        }
        Action::List => {
            let mut client = Client::connect(&options.socket)?;
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
    }
}

fn required_path(matches: &ArgMatches, name: &str) -> Result<PathBuf> {
    matches
        .get_one::<String>(name)
        .map(PathBuf::from)
        .ok_or_else(|| Error::Protocol(format!("program-server needs --{name}")))
}

fn values(matches: &ArgMatches, name: &str) -> Vec<PathBuf> {
    matches
        .get_many::<String>(name)
        .map(|values| values.map(PathBuf::from).collect())
        .unwrap_or_default()
}

/// The socket of this host's Program Server for the calling user:
/// `$XDG_RUNTIME_DIR/uniserve/program-server.sock`, or
/// `/tmp/uniserve-<uid>/program-server.sock` without a runtime directory.
pub fn default_socket() -> PathBuf {
    let directory = match std::env::var_os("XDG_RUNTIME_DIR") {
        Some(runtime) if !runtime.is_empty() => PathBuf::from(runtime).join("uniserve"),
        // SAFETY: getuid has no preconditions.
        _ => PathBuf::from(format!("/tmp/uniserve-{}", unsafe { libc::getuid() })),
    };
    directory.join("program-server.sock")
}
