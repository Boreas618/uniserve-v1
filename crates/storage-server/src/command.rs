//! The `uniserve storage-server` command: the Storage Server of one host,
//! and the operator's commands against it.
//!
//! `serve` holds this host's retained storage for the calling user until it
//! is stopped ([`crate::serve`]), with the settings one document states
//! (`--settings PATH`, or `-` for standard input; `storage::Settings`): a
//! deployment's daemon passes its deployment's settings that way when it
//! starts the Storage Server of a host that has none (`service::host::start`),
//! and an operator writes the same document to a file. With `--daemon` it
//! leaves its starter's session first (`service::host::detach`) and the
//! starter prints the server's readiness record once it serves; `--ready-fd`
//! names a descriptor the record is written to instead. With a connector in
//! its settings, the server also runs the Mooncake connector, which keeps the
//! pages of the buffer pools' host regions in Mooncake (L3): this
//! executable's `storage-server connector` command, as a child process the
//! server starts again when it ends. It always runs its Transfer Server,
//! which copies between the storage it holds, the same way
//! (`storage-server transfer`).
//!
//! `status` prints the running server's readiness record (its endpoints and
//! settings), `stop` stops it and returns once it ended, `list` prints what it
//! holds, `retire` releases the storage of a deployment that is no longer
//! served, and `restart` passes the running server to a successor.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;

/// How long `stop` waits for the server to end: it ends its children and closes its
/// connections, which takes no longer than a child's exit.
const STOP_TIMEOUT: Duration = Duration::from_secs(60);

/// The `storage-server` command's arguments.
#[derive(Debug, clap::Parser)]
#[command(name = "storage-server", about = "The Storage Server of one host")]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

/// The `storage-server` subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Serve this host's retained storage until stopped.
    Serve {
        /// The socket to listen on; this user's default socket when unset.
        #[arg(long)]
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
        /// succeeds in a restart in place (`storage-server restart`), which it
        /// goes on with.
        #[arg(long, hide = true, value_name = "FD")]
        predecessor: Option<i32>,
    },
    /// The Mooncake connector a server runs as its child, with its settings
    /// (`storage::connector::Settings`, TOML) on its standard input.
    #[command(hide = true)]
    Connector {
        #[arg(long)]
        socket: PathBuf,
    },
    /// The Transfer Server a server runs as its child.
    #[command(hide = true)]
    Transfer {
        #[arg(long)]
        socket: PathBuf,
        /// How long one chunk of a copy lasts at its link's bandwidth, in
        /// microseconds.
        #[arg(long)]
        chunk_us: u64,
        /// NIXL's C API library, for copies from other hosts.
        #[arg(long)]
        nixl_library: Option<PathBuf>,
    },
    /// Print the running server's readiness record: its endpoints and the
    /// settings it serves with, one JSON line.
    Status {
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Stop the running server and return once it ended.
    Stop {
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Print the tensors, bytes and deployments the running server holds,
    /// and the buffer pool of each deployment.
    List {
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Release the storage of a deployment that is no longer served; refused
    /// while its head is connected.
    Retire {
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long)]
        deployment: String,
    },
    /// Hand the running server to a successor, of this build or the one
    /// `--executable` names, which goes on with every connection, tensor and
    /// buffer pool once nothing is in flight (a restart in place).
    Restart {
        #[arg(long)]
        socket: Option<PathBuf>,
        /// The `uniserve` executable the successor runs; the running
        /// server's own when unset.
        #[arg(long)]
        executable: Option<String>,
    },
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

/// `socket`, or this user's default socket.
fn socket_or_default(socket: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    match socket {
        Some(socket) => Ok(socket),
        None => ::storage::default_socket().context("finding this user's Storage Server socket"),
    }
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
            socket,
            daemon,
            ready_fd,
            settings,
            predecessor,
        } => {
            let settings = match &settings {
                Some(path) => read_settings(path)?,
                None => ::storage::Settings::default(),
            };
            let executable = std::env::current_exe().context("locating this executable")?;
            // The helpers log at the server's levels; `--log-level` is the executable's own
            // option, before its command.
            let helper = |command: &str| {
                let mut words: Vec<std::ffi::OsString> = vec![executable.clone().into()];
                words.extend(levels.arguments().into_iter().map(Into::into));
                words.extend(["storage-server".into(), command.into()]);
                words
            };
            crate::serve(crate::ServeOptions {
                socket: socket_or_default(socket)?,
                daemon,
                ready_fd,
                settings,
                connector_command: helper("connector"),
                transfer_command: helper("transfer"),
                predecessor,
                levels: levels.clone(),
            })
        }
        Command::Connector { socket } => {
            let mut document = String::new();
            std::io::stdin()
                .read_to_string(&mut document)
                .context("reading the connector's settings from standard input")?;
            let settings: ::storage::connector::Settings =
                toml::from_str(&document).context("parsing the connector's settings")?;
            crate::run_connector(&socket, &settings, levels)
        }
        Command::Transfer {
            socket,
            chunk_us,
            nixl_library,
        } => crate::run_transfer_server(
            &socket,
            &crate::transfer::Options {
                chunk: Duration::from_micros(chunk_us),
                nixl_library,
            },
            levels,
        ),
        Command::Status { socket } => {
            let socket = socket_or_default(socket)?;
            let readiness = ::storage::Client::connect(&socket, ::storage::Role::Operator)
                .and_then(|mut operator| operator.status())
                .with_context(|| format!("asking the Storage Server at {}", socket.display()))?;
            let line = readiness.line()?;
            std::io::Write::write_all(&mut std::io::stdout().lock(), &line)?;
            Ok(())
        }
        Command::Stop { socket } => {
            let socket = socket_or_default(socket)?;
            ::storage::Client::connect(&socket, ::storage::Role::Operator)
                .and_then(|operator| operator.stop(STOP_TIMEOUT))
                .with_context(|| format!("stopping the Storage Server at {}", socket.display()))
        }
        Command::List { socket } => {
            crate::list(&socket_or_default(socket)?, &mut std::io::stdout().lock())
        }
        Command::Retire { socket, deployment } => {
            crate::retire(&socket_or_default(socket)?, &deployment).map_err(anyhow::Error::from)
        }
        Command::Restart { socket, executable } => {
            let socket = socket_or_default(socket)?;
            storage::Client::connect(&socket, ::storage::Role::Operator)
                .and_then(|mut operator| operator.restart(executable))
                .map(|pid| {
                    println!("the successor server {pid} takes over once nothing is in flight")
                })
                .map_err(anyhow::Error::from)
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
}
