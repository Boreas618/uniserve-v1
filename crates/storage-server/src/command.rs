//! The `uniserve storage-server` command: a deployment's Storage Server on one
//! node, and the operator's commands against it.
//!
//! `serve --deployment NAME` holds the deployment's retained storage on this
//! node until it is stopped ([`crate::serve`]), with the settings one document
//! states (`--settings PATH`, or `-` for standard input; `storage::Settings`):
//! the deployment's daemon passes its settings that way when it starts the
//! deployment's Storage Server on a node that has none (`service::host::start`),
//! and an operator writes the same document to a file. With `--daemon` it
//! leaves its starter's session first (`service::host::detach`) and the
//! starter prints the server's readiness record once it serves; `--ready-fd`
//! names a descriptor the record is written to instead. The server runs its
//! children from this executable's image (`crate::child`): with a connector in
//! its settings the Mooncake connector, which keeps the pages of the buffer
//! pools' host regions in Mooncake (L3), and with a transfer calibration in its
//! settings its Transfer Server, which copies between the storage it holds.
//! Each is this command's hidden `child` entry, which reads which child it is
//! from the channel the server made.
//!
//! `calibrate` times this host's copy paths and writes the calibration the
//! Transfer Server copies by (`crate::transfer::calibration`), which the
//! settings name (`transfer_calibration`).
//!
//! `status` prints the running server's readiness record (its endpoints and
//! settings), `stop` stops it, releasing everything it holds, and returns once
//! it ended, `list` prints what it holds, and `restart` passes the running
//! server to a successor of this command's own build. Each names the server it
//! reaches by its deployment (`--deployment NAME`) or by its socket
//! (`--socket PATH`).

use std::ffi::OsString;
use std::io::Read as _;
use std::os::fd::{FromRawFd as _, OwnedFd, RawFd};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use clap::CommandFactory as _;

use crate::child::{CHANNEL_OPTION, Program};

/// How long `stop` waits for the server to end: it ends its children and closes its
/// connections, which takes no longer than a child's exit.
const STOP_TIMEOUT: Duration = Duration::from_secs(60);

/// The option a successor names its handoff socket's descriptor by.
const PREDECESSOR_OPTION: &str = "predecessor";

/// The `storage-server` command's arguments.
#[derive(Debug, clap::Parser)]
#[command(name = "storage-server", about = "A deployment's Storage Server on one node")]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

/// The `storage-server` subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Serve a deployment's retained storage on this node until stopped.
    Serve {
        /// The deployment whose storage the server holds: it listens on the deployment's socket
        /// (`storage::socket`) unless `--socket` names another.
        #[arg(long, value_name = "NAME")]
        deployment: Option<String>,
        /// The socket to listen on.
        #[arg(long, value_name = "PATH")]
        socket: Option<PathBuf>,
        /// Leave the starter's session, log beside the socket, and return to
        /// the starter once serving, printing the server's readiness record.
        #[arg(long)]
        daemon: bool,
        /// The inherited descriptor the server writes its readiness record to
        /// (one JSON line) once it serves.
        #[arg(long, value_name = "FD")]
        ready_fd: Option<i32>,
        /// The settings document (TOML, `storage::Settings`), `-` for
        /// standard input; every setting at its documented default when
        /// unset.
        #[arg(long, value_name = "PATH")]
        settings: Option<PathBuf>,
        /// The descriptor of the handoff socket to the server this one
        /// succeeds in a restart in place (`storage-server restart`), which
        /// passes it the socket and the settings it serves with.
        #[arg(long = PREDECESSOR_OPTION, hide = true, value_name = "FD")]
        predecessor: Option<RawFd>,
    },
    /// One of a server's children, on the channel the server made for it at
    /// this descriptor, which states which child it is.
    #[command(hide = true)]
    Child {
        #[arg(long = CHANNEL_OPTION, value_name = "FD")]
        channel: RawFd,
    },
    /// Time this host's copy paths and write the calibration the Transfer Server copies by.
    Calibrate(crate::transfer::calibration::measure::Arguments),
    /// Print the running server's readiness record: its endpoints and the
    /// settings it serves with, one JSON line.
    Status {
        #[command(flatten)]
        server: Address,
    },
    /// Stop the running server, which releases everything it holds, and return once it ended.
    Stop {
        #[command(flatten)]
        server: Address,
    },
    /// Print the tensors, bytes and deployments the running server holds,
    /// and the buffer pool of each deployment.
    List {
        #[command(flatten)]
        server: Address,
        /// Print the listing as one JSON document (`storage::protocol::Listing`) instead of
        /// its text rendering.
        #[arg(long)]
        json: bool,
    },
    /// Hand the running server to a successor of this command's build, which
    /// goes on with every connection, tensor and buffer pool once nothing is
    /// in flight (a restart in place).
    Restart {
        #[command(flatten)]
        server: Address,
    },
}

/// The Storage Server an operator's command reaches: the one on the socket it names, or the
/// Storage Server of the deployment it names on this node (`storage::socket`).
#[derive(Debug, Clone, PartialEq, Eq, clap::Args)]
#[group(required = true, multiple = false)]
pub struct Address {
    /// The Storage Server's socket.
    #[arg(long, value_name = "PATH")]
    pub socket: Option<PathBuf>,
    /// The deployment whose Storage Server on this node is meant.
    #[arg(long, value_name = "NAME")]
    pub deployment: Option<String>,
}

impl Address {
    /// The socket the address names.
    ///
    /// # Errors
    ///
    /// Fails when a deployment is named and the user has no runtime directory, or the name is
    /// no deployment's.
    pub fn socket(&self) -> anyhow::Result<PathBuf> {
        socket_of(self.socket.clone(), self.deployment.as_deref())
    }
}

/// The entry arguments of a restart's successor of the handoff socket at `predecessor`
/// (`storage-server serve --predecessor FD`), as [`Command::Serve`] reads them.
pub(crate) fn successor_entry(predecessor: RawFd) -> Vec<OsString> {
    vec![
        "serve".into(),
        format!("--{PREDECESSOR_OPTION}").into(),
        predecessor.to_string().into(),
    ]
}

/// The settings document `path` names (`-`: standard input), as `serve --settings` reads it.
///
/// # Errors
///
/// Fails when the document cannot be read or is no settings document.
pub fn read_settings(path: &Path) -> anyhow::Result<::storage::Settings> {
    let document = if path == Path::new("-") {
        let mut document = String::new();
        std::io::stdin()
            .read_to_string(&mut document)
            .context("reading the settings from standard input")?;
        document
    } else {
        std::fs::read_to_string(path)
            .with_context(|| format!("reading the settings {}", path.display()))?
    };
    ::storage::Settings::parse(&document)
        .with_context(|| format!("parsing the settings {}", path.display()))
}

/// `socket`, or the socket of `deployment`'s Storage Server on this node.
fn socket_of(socket: Option<PathBuf>, deployment: Option<&str>) -> anyhow::Result<PathBuf> {
    match (socket, deployment) {
        (Some(socket), _) => Ok(socket),
        (None, Some(deployment)) => {
            let runtime = service::RuntimeDirectory::of_user()
                .context("finding the user's runtime directory")?;
            ::storage::socket(&runtime, deployment)
                .with_context(|| format!("finding the Storage Server socket of {deployment}"))
        }
        (None, None) => anyhow::bail!(
            "name the Storage Server's deployment (--deployment) or its socket (--socket)"
        ),
    }
}

/// This executable's image, running its `storage-server` command at `levels`: the program a
/// server runs its children and a restart's successor from.
fn this_program(levels: &service::logging::Levels) -> anyhow::Result<Program> {
    let mut prefix = levels.arguments();
    prefix.push(Args::command().get_name().to_owned());
    Program::running(prefix).context("opening this executable's image")
}

/// Runs the command. `serve --daemon` forks, so the caller has started no
/// thread.
///
/// # Errors
///
/// Fails when the server cannot start or fails while serving, or the running
/// server cannot be reached or refuses the operator's request.
pub fn run(args: Args, levels: &service::logging::Levels) -> anyhow::Result<()> {
    match args.command {
        Command::Serve {
            deployment,
            socket,
            daemon,
            ready_fd,
            settings,
            predecessor,
        } => {
            let start = match predecessor {
                Some(descriptor) => {
                    // SAFETY: the predecessor placed the handoff socket at this descriptor for
                    // this process, and nothing else in it owns the descriptor.
                    let handoff = unsafe { OwnedFd::from_raw_fd(descriptor) };
                    crate::Start::Successor(handoff)
                }
                None => crate::Start::Fresh {
                    socket: socket_of(socket, deployment.as_deref())?,
                    settings: Box::new(match &settings {
                        Some(path) => read_settings(path)?,
                        None => ::storage::Settings::default(),
                    }),
                },
            };
            crate::serve(crate::ServeOptions {
                start,
                daemon,
                ready_fd,
                program: this_program(levels)?,
                levels: levels.clone(),
            })
        }
        Command::Child { channel } => {
            // SAFETY: the server that started this process made the channel and placed it at
            // this descriptor for it; nothing else in this process owns the descriptor.
            let channel = unsafe { OwnedFd::from_raw_fd(channel) };
            crate::child::run(channel, levels)
        }
        Command::Calibrate(arguments) => crate::transfer::calibration::measure::run(&arguments),
        Command::Status { server } => {
            let socket = server.socket()?;
            let readiness = ::storage::Client::connect(&socket, ::storage::Role::Operator)
                .and_then(|mut operator| operator.status())
                .with_context(|| format!("asking the Storage Server at {}", socket.display()))?;
            let line = readiness.line()?;
            std::io::Write::write_all(&mut std::io::stdout().lock(), &line)?;
            Ok(())
        }
        Command::Stop { server } => {
            let socket = server.socket()?;
            ::storage::Client::connect(&socket, ::storage::Role::Operator)
                .and_then(|operator| operator.stop(STOP_TIMEOUT))
                .with_context(|| format!("stopping the Storage Server at {}", socket.display()))
        }
        Command::List { server, json } => {
            let rendering = if json {
                crate::Rendering::Json
            } else {
                crate::Rendering::Text
            };
            crate::list(
                &server.socket()?,
                rendering,
                &mut std::io::stdout().lock(),
            )
        }
        Command::Restart { server } => {
            let socket = server.socket()?;
            // The successor runs this command's own build: the operator restarts the server
            // into the executable it runs.
            let program = this_program(levels)?;
            let pid = ::storage::Client::connect(&socket, ::storage::Role::Operator)
                .and_then(|mut operator| {
                    operator.restart(program.image(), program.prefix().to_vec(), None)
                })
                .with_context(|| {
                    format!("restarting the Storage Server at {}", socket.display())
                })?;
            println!("the successor server {pid} takes over once nothing is in flight");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser as _;

    use super::*;

    /// A server reads its settings from the document `--settings` names, and serves with every
    /// setting at its default without one.
    #[test]
    fn a_server_reads_its_settings_document() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("storage-server.toml");
        std::fs::write(
            &path,
            "[server]\ntcp = \"10.0.0.2:0\"\ntransfer_deadline_s = 5\n\n\
             [connector]\nmaster = \"10.0.0.1:50051\"\ntransport = \"rdma\"\n",
        )
        .unwrap();
        let settings = read_settings(&path).unwrap();
        assert_eq!(settings.server.transfer_deadline_s, 5);
        assert_eq!(settings.server.put_deadline_s, 60);
        assert_eq!(settings.server.tcp, Some("10.0.0.2:0".parse().unwrap()));
        let connector = settings.connector.unwrap();
        assert_eq!(connector.transport, ::storage::connector::Transport::Rdma);
        assert_eq!(
            connector.metadata,
            ::storage::connector::Metadata::P2pHandshake
        );

        let Command::Serve { settings, .. } = Args::parse_from(["storage-server", "serve"]).command
        else {
            panic!("the arguments are not a `serve` command");
        };
        assert!(settings.is_none());
    }

    /// The entries the server starts its children and its successor with are this command's.
    #[test]
    fn the_entries_a_server_starts_are_its_commands() {
        let parse = |entry: Vec<OsString>| {
            let words = std::iter::once(OsString::from("storage-server")).chain(entry);
            Args::try_parse_from(words).unwrap().command
        };
        assert!(matches!(
            parse(crate::child::child_entry(7)),
            Command::Child { channel: 7 }
        ));
        assert!(matches!(
            parse(successor_entry(9)),
            Command::Serve {
                predecessor: Some(9),
                daemon: false,
                ..
            }
        ));
    }
}
