//! `uniserve-storage-server`: the Storage Server of one host.
//!
//! `serve` holds this host's retained memory for the calling user until it
//! is stopped with SIGTERM or SIGINT. The head and each launcher start it
//! with `--daemon` when their host has none: the process then leaves its
//! starter's session and process tree, writes its log beside its socket and
//! signals its starter once it listens, so restarting a head or a launcher
//! never ends it.
//!
//! With `--mooncake-master`, the server also runs the Mooncake connector,
//! which keeps committed pages in Mooncake (L3): the same executable's
//! `connector` command, as a child process the server starts again when it
//! ends.
//!
//! `list` prints what a running server holds, and `retire` releases the
//! memory of a deployment that is no longer served.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use clap::{Args, Parser, Subcommand};
use uniserve_storage::connector::Settings;
use uniserve_storage::{Client, ConnectorLaunch, Role, Server, socket};

#[derive(Parser)]
#[command(
    name = "uniserve-storage-server",
    about = "The Storage Server of one host"
)]
struct Arguments {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Serve this host's retained memory until stopped.
    Serve {
        /// The socket to listen on; this user's default socket when unset.
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Leave the starter's session, log beside the socket, and return to
        /// the starter once listening.
        #[arg(long)]
        daemon: bool,
        /// The size of each host page pool, in GiB. The workers that copy
        /// through a pool pin it, so it is allocated whole when the first
        /// one maps it.
        #[arg(long, default_value_t = uniserve_storage::DEFAULT_POOL_BYTES >> 30)]
        page_pool_gib: u64,
        /// Also accept other hosts' heads, for page requests, on this TCP
        /// address (`host:port`; port 0 picks one, which `Endpoint` names).
        #[arg(long)]
        tcp: Option<String>,
        #[command(flatten)]
        mooncake: Mooncake,
    },
    /// The Mooncake connector a server runs as its child.
    #[command(hide = true)]
    Connector {
        #[arg(long)]
        socket: PathBuf,
        #[command(flatten)]
        mooncake: Mooncake,
    },
    /// Print the tensors, bytes and deployments the running server holds.
    List {
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Release the memory of a deployment that is no longer served; refused
    /// while its head is connected.
    Retire {
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long)]
        deployment: String,
    },
}

/// Where the Mooncake connector finds Mooncake: the arguments of its
/// client's `setup` (`uniserve_storage::connector::Settings`).
#[derive(Args, Debug, Clone)]
struct Mooncake {
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
    /// Host memory this host contributes to Mooncake's pool, in bytes.
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

fn main() -> ExitCode {
    let (socket, daemon, page_pool_bytes, tcp, mooncake) = match Arguments::parse().command {
        Commands::Serve {
            socket,
            daemon,
            page_pool_gib,
            tcp,
            mooncake,
        } => (socket, daemon, page_pool_gib << 30, tcp, mooncake),
        Commands::Connector { socket, mooncake } => return connect(&socket, &mooncake),
        Commands::List { socket } => {
            return operate(socket, |client| {
                let listing = client.list()?;
                for tensor in &listing.tensors {
                    let scopes: Vec<String> =
                        tensor.scopes.iter().map(ToString::to_string).collect();
                    println!(
                        "{} {:?} {} bytes {:?} {} [{}]{}",
                        tensor.key.hex(),
                        tensor.placement,
                        tensor.storage_bytes,
                        tensor.access,
                        tensor.name,
                        scopes.join(", "),
                        tensor
                            .writer
                            .map(|pid| format!(" writer {pid}"))
                            .unwrap_or_default()
                    );
                }
                for deployment in &listing.deployments {
                    println!(
                        "deployment {} head {}",
                        deployment.deployment,
                        if deployment.head_connected {
                            "connected"
                        } else {
                            "gone"
                        }
                    );
                }
                for pool in &listing.page_pools {
                    println!(
                        "page pool {} kv heads {}+{} {} slots of {} bytes: {} ready, {} reserved, \
                         {} pinned, {} fetching, {} uploading",
                        pool.pool,
                        pool.shape.kv_head_offset,
                        pool.shape.kv_head_count,
                        pool.slots,
                        pool.shape.page_bytes,
                        pool.ready,
                        pool.reserved,
                        pool.pinned,
                        pool.fetching,
                        pool.uploading
                    );
                }
                if listing.connector {
                    println!("mooncake connector connected");
                }
                Ok(())
            });
        }
        Commands::Retire { socket, deployment } => {
            return operate(socket, |client| {
                let released = client.retire(deployment.clone())?;
                println!("released {released} tensors of {deployment}");
                Ok(())
            });
        }
    };
    let socket = socket.unwrap_or_else(socket::default_socket);
    // Forked before any thread exists.
    let ready = if daemon {
        match detach(&socket) {
            Ok(ready) => Some(ready),
            Err(error) => {
                eprintln!("uniserve-storage-server: error: {error:#}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    init_logging();
    match serve(&socket, ready, page_pool_bytes, tcp.as_deref(), &mooncake) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!("storage server failed: {error:#}");
            ExitCode::FAILURE
        }
    }
}

/// Run an operator's request against the server on `socket`.
fn operate(
    socket: Option<PathBuf>,
    request: impl FnOnce(&mut Client) -> uniserve_storage::Result<()>,
) -> ExitCode {
    let socket = socket.unwrap_or_else(socket::default_socket);
    let result =
        Client::connect(&socket, Role::Operator).and_then(|mut client| request(&mut client));
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("uniserve-storage-server: error: {error}");
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

/// Run the Mooncake connector for the server on `socket` until the server
/// closes its connection.
fn connect(socket: &Path, mooncake: &Mooncake) -> ExitCode {
    init_logging();
    let result = mooncake.settings().and_then(|settings| {
        let settings = settings.context("the connector needs --mooncake-master")?;
        uniserve_storage::connector::run(socket, &settings).map_err(anyhow::Error::from)
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!("the Mooncake connector failed: {error:#}");
            ExitCode::FAILURE
        }
    }
}

/// Log to standard error at `UNISERVE_STORAGE_LOG`'s level, `info` when
/// unset.
fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("UNISERVE_STORAGE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
}

/// Bind the socket and serve until stopped. `ready` is the starter's pipe,
/// written once the server listens or found another one listening.
fn serve(
    socket: &Path,
    ready: Option<OwnedFd>,
    page_pool_bytes: u64,
    tcp: Option<&str>,
    mooncake: &Mooncake,
) -> anyhow::Result<()> {
    let signal = |ready: Option<OwnedFd>| {
        if let Some(ready) = ready {
            let _ = std::fs::File::from(ready).write_all(b"1");
        }
    };
    let Some((listener, _lock)) =
        socket::bind(socket).with_context(|| format!("binding {}", socket.display()))?
    else {
        tracing::info!(socket = %socket.display(), "another storage server serves this socket");
        signal(ready);
        return Ok(());
    };
    raise_descriptor_limit()?;
    let (wake, stop) = stop_signals()?;
    let connector = match mooncake.settings()? {
        Some(settings) => {
            let mut arguments = vec![
                "connector".into(),
                "--socket".into(),
                socket.as_os_str().to_owned(),
            ];
            arguments.extend(settings.arguments().into_iter().map(Into::into));
            Some(ConnectorLaunch {
                program: std::env::current_exe().context("locating this executable")?,
                arguments,
            })
        }
        None => None,
    };
    let mut server = Server::new(listener, page_pool_bytes, connector)?;
    if let Some(address) = tcp {
        let listener = std::net::TcpListener::bind(address)
            .with_context(|| format!("binding {address} for other hosts"))?;
        let bound = server.listen_tcp(listener)?;
        tracing::info!(address = %bound, "accepting other hosts' page requests");
    }
    tracing::info!(socket = %socket.display(), "storage server listening");
    signal(ready);
    server.run(wake.as_fd(), &|| stop.load(Ordering::Acquire))?;
    tracing::info!("storage server stopped");
    Ok(())
}

/// Raise the soft limit on open descriptors to the hard limit: the server
/// holds one descriptor per retained storage of every worker on the host,
/// thousands for a large model.
fn raise_descriptor_limit() -> anyhow::Result<()> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit fills the struct it is given.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return Err(std::io::Error::last_os_error()).context("reading the descriptor limit");
    }
    limit.rlim_cur = limit.rlim_max;
    // SAFETY: setrlimit reads the struct it is given.
    if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) } != 0 {
        return Err(std::io::Error::last_os_error()).context("raising the descriptor limit");
    }
    Ok(())
}

/// A descriptor that becomes readable, and a flag that is set, when SIGTERM
/// or SIGINT arrives.
fn stop_signals() -> anyhow::Result<(OwnedFd, Arc<AtomicBool>)> {
    let mut ends = [0; 2];
    // SAFETY: pipe2 fills the two descriptors it creates.
    if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) } != 0 {
        return Err(std::io::Error::last_os_error()).context("creating the signal pipe");
    }
    // SAFETY: pipe2 returned two new descriptors this function owns.
    let (reader, writer) =
        unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };
    // The write end lives for the process: a handler may run until exit.
    let writer = Box::leak(Box::new(writer)).as_raw_fd();
    let stop = Arc::new(AtomicBool::new(false));
    for signal in [libc::SIGTERM, libc::SIGINT] {
        let flag = Arc::clone(&stop);
        // SAFETY: the action stores to an atomic and writes to a descriptor
        // that is never closed, both async-signal-safe.
        unsafe {
            signal_hook_registry::register(signal, move || {
                flag.store(true, Ordering::Release);
                let _ = libc::write(writer, b"x".as_ptr().cast(), 1);
            })
        }
        .context("installing a signal handler")?;
    }
    Ok((reader, stop))
}

/// Leave the starter: fork, let the parent wait for the child's readiness
/// and exit, and make the child the leader of a session of its own with its
/// standard streams on `storage-server.log` beside the socket. Returns the
/// child's end of the readiness pipe.
fn detach(socket: &Path) -> anyhow::Result<OwnedFd> {
    let directory = socket.parent().context("the socket has no directory")?;
    std::fs::create_dir_all(directory)
        .with_context(|| format!("creating {}", directory.display()))?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.join("storage-server.log"))
        .context("opening the storage server's log")?;

    let mut ends = [0; 2];
    // SAFETY: pipe2 fills the two descriptors it creates.
    if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(std::io::Error::last_os_error()).context("creating the readiness pipe");
    }
    // SAFETY: pipe2 returned two new descriptors this function owns.
    let (reader, writer) =
        unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };

    // SAFETY: the process has one thread, so the child may run anything.
    match unsafe { libc::fork() } {
        -1 => Err(std::io::Error::last_os_error()).context("forking the storage server"),
        0 => {
            drop(reader);
            // SAFETY: setsid and dup2 on descriptors this process owns.
            unsafe {
                libc::setsid();
                let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDONLY);
                if null >= 0 {
                    libc::dup2(null, 0);
                    libc::close(null);
                }
                libc::dup2(log.as_raw_fd(), 1);
                libc::dup2(log.as_raw_fd(), 2);
            }
            Ok(writer)
        }
        _ => {
            drop(writer);
            // The child writes once it listens; a child that dies first
            // closes the pipe, which reads as end of file.
            let mut byte = [0_u8; 1];
            let started = std::fs::File::from(reader).read(&mut byte).unwrap_or(0) == 1;
            // SAFETY: `_exit` ends the starter without running handlers.
            unsafe { libc::_exit(if started { 0 } else { 1 }) }
        }
    }
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
        let mut arguments = vec!["uniserve-storage-server".to_owned(), "serve".to_owned()];
        arguments.extend(settings.arguments());
        let Commands::Serve { mooncake, .. } = Arguments::parse_from(arguments).command else {
            panic!("the arguments are not a `serve` command");
        };
        assert_eq!(mooncake.settings().unwrap(), Some(settings));
    }
}
