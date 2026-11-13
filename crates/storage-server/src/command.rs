//! The `uniserve storage-server` command: the Storage Server of one host,
//! and the operator's commands against it.
//!
//! `serve` holds this host's retained storage for the calling user until it
//! is stopped with SIGTERM or SIGINT ([`crate::serve`]). With `--daemon` it
//! leaves its starter's session first (`service::detach`), which is how a
//! daemon starts the Storage Server of a host that has none
//! (`service::ensure`). With `--mooncake-master`, the server also runs the
//! Mooncake connector, which keeps the pages of the buffer pools' host
//! regions in Mooncake (L3): this
//! executable's `storage-server connector` command, as a child process the
//! server starts again when it ends. It always runs its Transfer Server,
//! which copies between the storage it holds, the same way
//! (`storage-server transfer`).
//!
//! `list` prints what a running server holds, `retire` releases the storage
//! of a deployment that is no longer served, and `restart` passes the running
//! server to a successor.

use std::path::PathBuf;

use anyhow::Context;
use storage::connector::Settings;

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
        /// the starter once listening.
        #[arg(long)]
        daemon: bool,
        /// Host storage, in GiB, that new host storage (a host region, the
        /// host copy of an epoch's weights) leaves available to the host's
        /// other processes: storage the host's available storage less this
        /// reserve cannot hold is refused (SGLang's host KV cache keeps
        /// 10 GiB the same way).
        #[arg(long, default_value_t = 10)]
        host_reserve_gib: u64,
        /// Also accept other hosts' heads and peers on this TCP address
        /// (`host:port`; port 0 picks one, which `Endpoint` names).
        #[arg(long)]
        tcp: Option<String>,
        #[command(flatten)]
        mooncake: Mooncake,
        /// How long one chunk of the Transfer Server's copies lasts at its
        /// link's bandwidth, in microseconds: a copy engine turns to the
        /// executing epoch's own copies, and to transfers of a higher
        /// priority, between chunks, so a chunk is kept far below one
        /// scheduler step.
        #[arg(long, default_value_t = 1000)]
        transfer_chunk_us: u64,
        /// NIXL's C API library (`libnixl_capi.so`, of the NIXL release
        /// this build pins) that the Transfer Server loads for copies from
        /// other hosts; found on the loader's search path when unset.
        #[arg(long)]
        nixl_library: Option<PathBuf>,
        /// How long a buffer pool's transfer of a request between worker
        /// groups may take, in seconds, before it is stopped and answered
        /// as failed.
        #[arg(long, default_value_t = crate::sessions::TRANSFER_DEADLINE.as_secs())]
        transfer_deadline_s: u64,
        /// How long a buffer pool's put of host copies to the remote store
        /// may take, in seconds, before it is stopped and answered with the
        /// keys put so far.
        #[arg(long, default_value_t = crate::sessions::PUT_DEADLINE.as_secs())]
        put_deadline_s: u64,
        /// Record the trace of every buffer pool's inputs in this directory:
        /// per deployment, every input its pool acts on and every completion
        /// it produces (`specs/buffer-pool.md` §18), which
        /// `verification/buffer-pool/replay.sh` replays and checks against
        /// the design model. A failure to write stops the trace, not the
        /// pool.
        #[arg(long, value_name = "DIRECTORY")]
        buffer_pool_trace: Option<PathBuf>,
        /// Check every buffer pool's tables after each input it acts on: an
        /// inconsistency is logged when the pool turns inconsistent and
        /// counted in its listing (`check_failures`), and the pool goes on.
        #[arg(long)]
        buffer_pool_check: bool,
        /// The descriptor of the handoff socket to the server this one
        /// succeeds in a restart in place (`storage-server restart`), which it goes on
        /// with.
        #[arg(long, hide = true, value_name = "FD")]
        predecessor: Option<i32>,
    },
    /// The Mooncake connector a server runs as its child.
    #[command(hide = true)]
    Connector {
        #[arg(long)]
        socket: PathBuf,
        #[command(flatten)]
        mooncake: Mooncake,
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

/// Where the Mooncake connector finds Mooncake: the arguments of its
/// client's `setup` (`storage::connector::Settings`).
#[derive(clap::Args, Debug, Clone)]
pub struct Mooncake {
    /// Run the Mooncake connector, which keeps the pages of the buffer pools'
    /// host regions in Mooncake (L3), with Mooncake's master at this
    /// `host:port`.
    #[arg(long = "mooncake-master")]
    master: Option<String>,
    /// The transfer engine's metadata server: `P2PHANDSHAKE`, or an
    /// `http://`, `etcd://` or `redis://` URL.
    #[arg(long = "mooncake-metadata", default_value = "P2PHANDSHAKE")]
    metadata: String,
    /// The transfer protocol: `tcp` or `rdma`.
    #[arg(long = "mooncake-protocol", default_value = "tcp")]
    protocol: String,
    /// The RDMA devices; empty to select them automatically.
    #[arg(long = "mooncake-device", default_value = "")]
    device: String,
    /// The name other Mooncake clients reach this host by; the host name
    /// when unset.
    #[arg(long = "mooncake-hostname")]
    hostname: Option<String>,
    /// Host storage this host contributes to Mooncake's pool, in bytes.
    #[arg(long = "mooncake-segment-bytes", default_value_t = 0)]
    segment_bytes: u64,
    /// `libmooncake_store.so`; found on the loader's search path when
    /// unset.
    #[arg(long = "mooncake-library")]
    library: Option<PathBuf>,
}

impl Mooncake {
    /// The connector's settings, when a master is given.
    fn settings(&self) -> anyhow::Result<Option<Settings>> {
        let Some(master) = &self.master else {
            return Ok(None);
        };
        let hostname = match &self.hostname {
            Some(hostname) => hostname.clone(),
            None => host_name()?,
        };
        Ok(Some(Settings {
            master: master.clone(),
            metadata: self.metadata.clone(),
            protocol: self.protocol.clone(),
            device: self.device.clone(),
            hostname: Some(hostname),
            segment_bytes: self.segment_bytes,
            library: self.library.clone(),
        }))
    }
}

/// Runs the command. `serve --daemon` forks, so the caller has started no
/// thread.
///
/// # Errors
///
/// Fails when the server cannot start or fails while serving, or the running
/// server cannot be reached or refuses the operator's request.
pub fn run(args: Args) -> anyhow::Result<()> {
    match args.command {
        Command::Serve {
            socket,
            daemon,
            host_reserve_gib,
            tcp,
            mooncake,
            transfer_chunk_us,
            nixl_library,
            transfer_deadline_s,
            put_deadline_s,
            buffer_pool_trace,
            buffer_pool_check,
            predecessor,
        } => mooncake.settings().and_then(|connector| {
            let executable = std::env::current_exe().context("locating this executable")?;
            crate::serve(crate::ServeOptions {
                socket: socket.unwrap_or_else(storage::default_socket),
                daemon,
                host_reserve_bytes: host_reserve_gib << 30,
                tcp,
                connector,
                connector_command: vec![
                    executable.clone().into(),
                    "storage-server".into(),
                    "connector".into(),
                ],
                transfer_command: vec![
                    executable.into(),
                    "storage-server".into(),
                    "transfer".into(),
                ],
                transfer_chunk: std::time::Duration::from_micros(transfer_chunk_us),
                nixl_library,
                transfer_deadline: std::time::Duration::from_secs(transfer_deadline_s),
                put_deadline: std::time::Duration::from_secs(put_deadline_s),
                buffer_pool_trace,
                buffer_pool_check,
                predecessor,
            })
        }),
        Command::Connector { socket, mooncake } => mooncake
            .settings()
            .and_then(|settings| settings.context("the connector needs --mooncake-master"))
            .and_then(|settings| crate::run_connector(&socket, &settings)),
        Command::Transfer {
            socket,
            chunk_us,
            nixl_library,
        } => crate::run_transfer_server(
            &socket,
            &crate::transfer::Options {
                chunk: std::time::Duration::from_micros(chunk_us),
                nixl_library,
            },
        ),
        Command::List { socket } => crate::list(
            &socket.unwrap_or_else(storage::default_socket),
            &mut std::io::stdout().lock(),
        ),
        Command::Retire { socket, deployment } => {
            crate::retire(&socket.unwrap_or_else(storage::default_socket), &deployment)
                .map_err(anyhow::Error::from)
        }
        Command::Restart { socket, executable } => {
            let socket = socket.unwrap_or_else(storage::default_socket);
            // A server of an earlier protocol is restarted in its own.
            storage::Client::connect(&socket, ::storage::Role::Operator)
                .and_then(|mut operator| operator.restart(executable))
                .map(|pid| {
                    println!("the successor server {pid} takes over once nothing is in flight")
                })
                .map_err(anyhow::Error::from)
        }
    }
}

/// This host's name, as `gethostname(2)` gives it.
fn host_name() -> anyhow::Result<String> {
    let mut name = [0_u8; 256];
    // SAFETY: gethostname writes at most `name.len()` bytes into `name`.
    if unsafe { libc::gethostname(name.as_mut_ptr().cast(), name.len()) } != 0 {
        return Err(std::io::Error::last_os_error()).context("reading the host name");
    }
    let length = name
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(name.len());
    String::from_utf8(name[..length].to_vec()).context("the host name is not UTF-8")
}

#[cfg(test)]
mod tests {
    use clap::Parser as _;

    use super::*;

    /// A head states the connector's settings as arguments of `serve`
    /// (`Settings::arguments`); the server must read back the same settings.
    #[test]
    fn connector_settings_survive_the_command_line() {
        let settings = Settings {
            master: "10.0.0.1:50051".into(),
            metadata: "http://10.0.0.1:8080/metadata".into(),
            protocol: "rdma".into(),
            device: "mlx5_0,mlx5_1".into(),
            hostname: Some("10.0.0.2".into()),
            segment_bytes: 4 << 30,
            library: Some("/opt/mooncake/libmooncake_store.so".into()),
        };
        let mut arguments = vec!["storage-server".to_owned(), "serve".to_owned()];
        arguments.extend(settings.arguments());
        let Command::Serve { mooncake, .. } = Args::parse_from(arguments).command else {
            panic!("the arguments are not a `serve` command");
        };
        assert_eq!(mooncake.settings().unwrap(), Some(settings));
    }

    /// A transfer and a put to the remote store may take a minute unless
    /// the server is told otherwise.
    #[test]
    fn the_deadlines_of_transfers_and_puts_are_a_minute_unless_stated() {
        let deadlines = |arguments: &[&str]| {
            let mut command = vec!["storage-server", "serve"];
            command.extend(arguments);
            match Args::parse_from(command).command {
                Command::Serve {
                    transfer_deadline_s,
                    put_deadline_s,
                    ..
                } => (transfer_deadline_s, put_deadline_s),
                other => panic!("{other:?} is not a `serve` command"),
            }
        };
        assert_eq!(deadlines(&[]), (60, 60));
        assert_eq!(
            deadlines(&["--transfer-deadline-s", "5", "--put-deadline-s", "300"]),
            (5, 300)
        );
    }
}
