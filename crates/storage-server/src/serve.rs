//! Running a Storage Server, and the operator's commands against one.
//!
//! [`serve`] holds this host's retained memory for the calling user until it
//! is stopped with SIGTERM or SIGINT. A process that starts a server when its
//! host has none starts it with `daemon` set (`storage::socket::ensure`):
//! the server then leaves its starter's session and process tree, writes its
//! log beside its socket and signals its starter once it listens, so a
//! starter's end never ends it.
//!
//! With connector settings, the server also runs the Mooncake connector
//! ([`run_connector`]) as a child process it starts again when it ends. It
//! always runs its Transfer Server ([`run_transfer_server`]) the same way.

use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Context;
use storage::connector::Settings;
use storage::{Client, socket};

use crate::{Launch, Server};

/// How to serve: the arguments of `uniserve storage serve`.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// The socket to listen on.
    pub socket: PathBuf,
    /// Leave the starter's session, log beside the socket, and return to the
    /// starter once listening.
    pub daemon: bool,
    /// The size of each host page pool whose worker states none, in bytes.
    pub page_pool_bytes: u64,
    /// Host memory a new page pool leaves available to the host's other
    /// processes, in bytes.
    pub host_reserve_bytes: u64,
    /// Also accept other hosts' page requests on this TCP address
    /// (`host:port`; port 0 picks one).
    pub tcp: Option<String>,
    /// The Mooncake connector's settings, when the server runs one.
    pub connector: Option<Settings>,
    /// The command that runs the connector, before its own arguments: this
    /// executable and its `storage connector` command.
    pub connector_command: Vec<OsString>,
    /// The command that runs the Transfer Server, before its own arguments:
    /// this executable and its `storage transfer` command.
    pub transfer_command: Vec<OsString>,
    /// How long one chunk of the Transfer Server's copies lasts at its
    /// link's bandwidth.
    pub transfer_chunk: Duration,
    /// Go on with the server the predecessor of a hot restart hands over on
    /// the handoff socket at this descriptor, rather than binding the
    /// socket (`crate::restart`).
    pub takeover: Option<i32>,
}

/// Serve until stopped. Must be called before the process starts a thread:
/// a daemon forks.
///
/// # Errors
///
/// Fails when the starter cannot be left, the socket cannot be bound, the
/// server cannot start, or it fails while serving.
pub fn serve(options: ServeOptions) -> anyhow::Result<()> {
    // Forked before any thread exists. A successor runs detached already.
    let ready = if options.daemon && options.takeover.is_none() {
        Some(detach(&options.socket)?)
    } else {
        None
    };
    init_logging();
    let served = match options.takeover {
        Some(descriptor) => serve_resumed(&options, descriptor),
        None => serve_on(&options, ready),
    };
    if let Err(error) = &served {
        tracing::error!("storage server failed: {error:#}");
    }
    served
}

/// Run the Mooncake connector for the server on `socket` until the server
/// closes its connection.
///
/// # Errors
///
/// Fails when the connector cannot reach Mooncake or the server.
pub fn run_connector(socket: &Path, settings: &Settings) -> anyhow::Result<()> {
    init_logging();
    let ran = crate::mooncake::run(socket, settings).map_err(anyhow::Error::from);
    if let Err(error) = &ran {
        tracing::error!("the Mooncake connector failed: {error:#}");
    }
    ran
}

/// Run the Transfer Server for the server on `socket` until the server
/// closes its connection.
///
/// # Errors
///
/// Fails when a GPU cannot be measured or the server cannot be reached.
pub fn run_transfer_server(
    socket: &Path,
    options: &crate::transfer::Options,
) -> anyhow::Result<()> {
    init_logging();
    let ran = crate::transfer::run(socket, options).map_err(anyhow::Error::from);
    if let Err(error) = &ran {
        tracing::error!("the Transfer Server failed: {error:#}");
    }
    ran
}

/// Print the tensors, bytes and deployments the server on `socket` holds.
///
/// # Errors
///
/// Fails when the server cannot be reached or refuses the listing.
pub fn list(socket: &Path) -> storage::Result<()> {
    let mut client = Client::connect_operator(socket)?;
    let listing = client.list()?;
    for tensor in &listing.tensors {
        let scopes: Vec<String> = tensor.scopes.iter().map(ToString::to_string).collect();
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
            "page pool {} of {} kv heads {}+{} {} slots of {} bytes: {} ready, {} reserved, \
             {} pinned, {} fetching, {} uploading",
            pool.pool,
            pool.deployment,
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
    if let Some(transfer) = &listing.transfer {
        println!("transfer server {} connected", transfer.pid);
        for context in &transfer.contexts {
            println!(
                "transfer server context on {}: {} bytes",
                cuda::DeviceUuid(context.gpu),
                context.bytes
            );
        }
        for measured in &transfer.links {
            println!(
                "link {} -> {}: {:.1} GB/s",
                memory_name(measured.link.source),
                memory_name(measured.link.destination),
                measured.bytes_per_second as f64 / 1e9
            );
        }
        for task in &transfer.tasks {
            println!(
                "transfer {} of {} update {} {:?}: {} of {} bytes{}",
                task.task,
                task.deployment,
                task.update,
                task.priority,
                task.copied,
                task.bytes,
                if task.cancelled { ", cancelled" } else { "" }
            );
        }
    }
    Ok(())
}

/// A link end's name in the listing.
fn memory_name(memory: storage::protocol::Memory) -> String {
    match memory {
        storage::protocol::Memory::Host => "host".into(),
        storage::protocol::Memory::Device { gpu } => cuda::DeviceUuid(gpu).to_string(),
    }
}

/// Release the memory of `deployment`, which is no longer served, at the
/// server on `socket`.
///
/// # Errors
///
/// Fails when the server cannot be reached, or refuses because the
/// deployment's head is connected.
pub fn retire(socket: &Path, deployment: &str) -> storage::Result<()> {
    let mut client = Client::connect_operator(socket)?;
    let released = client.retire(deployment.to_owned())?;
    println!("released {released} tensors of {deployment}");
    Ok(())
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
fn serve_on(options: &ServeOptions, ready: Option<OwnedFd>) -> anyhow::Result<()> {
    let socket = &options.socket;
    let signal = |ready: Option<OwnedFd>| {
        if let Some(ready) = ready {
            let _ = std::fs::File::from(ready).write_all(b"1");
        }
    };
    let Some((listener, lock)) =
        socket::bind(socket).with_context(|| format!("binding {}", socket.display()))?
    else {
        tracing::info!(socket = %socket.display(), "another storage server serves this socket");
        signal(ready);
        return Ok(());
    };
    raise_descriptor_limit()?;
    let (wake, stop) = stop_signals()?;
    let mut server = Server::new(
        listener,
        options.page_pool_bytes,
        options.host_reserve_bytes,
        connector_launch(options),
        Some(transfer_launch(options)),
    )?;
    if let Some(address) = &options.tcp {
        let listener = std::net::TcpListener::bind(address)
            .with_context(|| format!("binding {address} for other hosts"))?;
        let bound = server.listen_tcp(listener)?;
        tracing::info!(address = %bound, "accepting other hosts' page requests");
    }
    server.set_relaunch(crate::relaunch_arguments(std::env::args_os().collect()));
    tracing::info!(socket = %socket.display(), "storage server listening");
    signal(ready);
    run_until_stopped(server, &wake, &stop, lock)
}

/// Go on with the server a hot restart's predecessor hands over on the
/// handoff socket at `descriptor`, and serve until stopped.
fn serve_resumed(options: &ServeOptions, descriptor: i32) -> anyhow::Result<()> {
    raise_descriptor_limit()?;
    let (wake, stop) = stop_signals()?;
    let (server, lock) = Server::resume(
        descriptor,
        connector_launch(options),
        Some(transfer_launch(options)),
    )
    .context("taking the server over from its predecessor")?;
    let lock = socket::SocketLock::resume(lock, options.socket.clone());
    tracing::info!(socket = %options.socket.display(), "storage server listening after a hot restart");
    run_until_stopped(server, &wake, &stop, lock)
}

/// Serve until stopped, or until a hot restart handed the server over.
fn run_until_stopped(
    mut server: Server,
    wake: &OwnedFd,
    stop: &AtomicBool,
    lock: socket::SocketLock,
) -> anyhow::Result<()> {
    match server.run(wake.as_fd(), &|| stop.load(Ordering::Acquire))? {
        Some(pending) => {
            let pid = server
                .hand_over(pending, lock.hand_over())
                .context("handing the server to its successor")?;
            tracing::info!(pid, "handed the server to its successor; exiting");
        }
        None => tracing::info!("storage server stopped"),
    }
    Ok(())
}

/// The connector the server runs, from `options`.
fn connector_launch(options: &ServeOptions) -> Option<Launch> {
    options.connector.as_ref().map(|settings| {
        let mut command = options.connector_command.clone();
        let program = command.remove(0);
        command.extend(["--socket".into(), options.socket.as_os_str().to_owned()]);
        command.extend(settings.arguments().into_iter().map(Into::into));
        Launch {
            program: program.into(),
            arguments: command,
        }
    })
}

/// The Transfer Server the server runs, from `options`.
fn transfer_launch(options: &ServeOptions) -> Launch {
    let mut command = options.transfer_command.clone();
    let program = command.remove(0);
    command.extend([
        "--socket".into(),
        options.socket.as_os_str().to_owned(),
        "--chunk-us".into(),
        options.transfer_chunk.as_micros().to_string().into(),
    ]);
    Launch {
        program: program.into(),
        arguments: command,
    }
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
