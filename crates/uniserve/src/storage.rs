//! `uniserve storage`: the Storage Server of one host, and the operator's
//! commands against it.
//!
//! `serve` holds this host's retained storage for the calling user until it
//! is stopped with SIGTERM or SIGINT (`storage_server::serve`). A process
//! that needs a server on a host that has none starts it with `--daemon`.
//! With `--mooncake-master`, the server also runs the Mooncake connector,
//! which keeps committed pages in Mooncake (L3): this executable's `storage
//! connector` command, as a child process the server starts again when it
//! ends. It always runs its Transfer Server, which copies between the
//! storage it holds, the same way (`storage transfer`).
//!
//! `list` prints what a running server holds, and `retire` releases the
//! storage of a deployment that is no longer served.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use clap::{Args, Parser, Subcommand};
use storage::connector::Settings;
use storage::socket;

/// The `storage` command's arguments.
#[derive(Debug, Parser)]
#[command(name = "storage", about = "The Storage Server of one host")]
pub(crate) struct StorageArgs {
    #[command(subcommand)]
    pub(crate) command: StorageCommand,
}

#[derive(Debug, Subcommand)]
pub(crate) enum StorageCommand {
    /// Serve this host's retained storage until stopped.
    Serve {
        /// The socket to listen on; this user's default socket when unset.
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Leave the starter's session, log beside the socket, and return to
        /// the starter once listening.
        #[arg(long)]
        daemon: bool,
        /// The size of a host page pool whose worker states none, in GiB.
        /// The workers that copy through a pool pin it, so it is allocated
        /// whole when the first one maps it.
        #[arg(long, default_value_t = storage_server::DEFAULT_POOL_BYTES >> 30)]
        page_pool_gib: u64,
        /// Host storage, in GiB, that a new page pool leaves available to the
        /// host's other processes: a pool the host's available storage less
        /// this reserve cannot hold is refused (SGLang's host KV cache keeps
        /// 10 GiB the same way).
        #[arg(long, default_value_t = 10)]
        host_reserve_gib: u64,
        /// Also accept other hosts' heads, for page requests, on this TCP
        /// address (`host:port`; port 0 picks one, which `Endpoint` names).
        #[arg(long)]
        tcp: Option<String>,
        #[command(flatten)]
        mooncake: Mooncake,
        /// How long one chunk of the Transfer Server's copies lasts at its
        /// link's bandwidth, in microseconds: a copy engine turns to the
        /// executing world's own copies, and to transfers of a higher
        /// priority, between chunks, so a chunk is kept far below one
        /// scheduler step.
        #[arg(long, default_value_t = 1000)]
        transfer_chunk_us: u64,
        /// NIXL's C API library (`libnixl_capi.so`, of the NIXL release
        /// this build pins) that the Transfer Server loads for copies from
        /// other hosts; found on the loader's search path when unset.
        #[arg(long)]
        nixl_library: Option<PathBuf>,
        /// Go on with the server the predecessor of a hot restart hands over
        /// on the handoff socket at this descriptor (`storage restart`).
        #[arg(long, hide = true)]
        takeover: Option<i32>,
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
    /// Print the tensors, bytes and deployments the running server holds.
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
    /// page pool once nothing is in flight (a hot restart).
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
#[derive(Args, Debug, Clone)]
pub(crate) struct Mooncake {
    /// Run the Mooncake connector, which keeps committed pages in Mooncake
    /// (L3), with Mooncake's master at this `host:port`.
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

/// Runs one `storage` command. `serve --daemon` forks, so the caller has
/// started no thread.
pub(crate) fn run(args: StorageArgs) -> ExitCode {
    let outcome = match args.command {
        StorageCommand::Serve {
            socket,
            daemon,
            page_pool_gib,
            host_reserve_gib,
            tcp,
            mooncake,
            transfer_chunk_us,
            nixl_library,
            takeover,
        } => mooncake.settings().and_then(|connector| {
            let executable = std::env::current_exe().context("locating this executable")?;
            storage_server::serve(storage_server::ServeOptions {
                socket: socket.unwrap_or_else(socket::default_socket),
                daemon,
                page_pool_bytes: page_pool_gib << 30,
                host_reserve_bytes: host_reserve_gib << 30,
                tcp,
                connector,
                connector_command: vec![
                    executable.clone().into(),
                    "storage".into(),
                    "connector".into(),
                ],
                transfer_command: vec![executable.into(), "storage".into(), "transfer".into()],
                transfer_chunk: std::time::Duration::from_micros(transfer_chunk_us),
                nixl_library,
                takeover,
            })
        }),
        StorageCommand::Connector { socket, mooncake } => mooncake
            .settings()
            .and_then(|settings| settings.context("the connector needs --mooncake-master"))
            .and_then(|settings| storage_server::run_connector(&socket, &settings)),
        StorageCommand::Transfer {
            socket,
            chunk_us,
            nixl_library,
        } => storage_server::run_transfer_server(
            &socket,
            &storage_server::transfer::Options {
                chunk: std::time::Duration::from_micros(chunk_us),
                nixl_library,
            },
        ),
        StorageCommand::List { socket } => {
            storage_server::list(&socket.unwrap_or_else(socket::default_socket))
                .map_err(anyhow::Error::from)
        }
        StorageCommand::Retire { socket, deployment } => {
            storage_server::retire(&socket.unwrap_or_else(socket::default_socket), &deployment)
                .map_err(anyhow::Error::from)
        }
        StorageCommand::Restart { socket, executable } => {
            let socket = socket.unwrap_or_else(socket::default_socket);
            // A server of an earlier protocol is restarted in its own.
            storage::Client::connect_operator(&socket)
                .and_then(|mut operator| operator.restart(executable))
                .map(|pid| {
                    println!("the successor server {pid} takes over once nothing is in flight")
                })
                .map_err(anyhow::Error::from)
        }
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("uniserve storage: error: {error:#}");
            ExitCode::FAILURE
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
        let mut arguments = vec!["storage".to_owned(), "serve".to_owned()];
        arguments.extend(settings.arguments());
        let StorageCommand::Serve { mooncake, .. } = StorageArgs::parse_from(arguments).command
        else {
            panic!("the arguments are not a `serve` command");
        };
        assert_eq!(mooncake.settings().unwrap(), Some(settings));
    }
}
