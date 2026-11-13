//! Running a Storage Server, and the operator's commands against one.
//!
//! [`serve`] holds this host's retained storage for the calling user until it
//! is stopped with SIGTERM or SIGINT. A process that starts a server when its
//! host has none starts it with `daemon` set (`service::ensure`):
//! the server then leaves its starter's session and process tree, writes its
//! log beside its socket and signals its starter once it listens, so a
//! starter's end never ends it.
//!
//! With connector settings, the server also runs the Mooncake connector
//! ([`run_connector`]) as a child process it starts again when it ends. It
//! always runs its Transfer Server ([`run_transfer_server`]) the same way.

use std::ffi::OsString;
use std::io::Write;
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ::storage::connector::Settings;
use ::storage::protocol::{BlockState, BufferPoolListing, Waiting};
use ::storage::{Client, Role};
use anyhow::Context;

use crate::transfer::storage_name;
use crate::{Launch, Server};

/// How to serve: the arguments of `uniserve storage-server serve`.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// The socket to listen on.
    pub socket: PathBuf,
    /// Leave the starter's session, log beside the socket, and return to the
    /// starter once listening.
    pub daemon: bool,
    /// Host storage new host storage leaves available to the host's other
    /// processes, in bytes.
    pub host_reserve_bytes: u64,
    /// Also accept other hosts' heads and peers on this TCP address
    /// (`host:port`; port 0 picks one).
    pub tcp: Option<String>,
    /// The Mooncake connector's settings, when the server runs one.
    pub connector: Option<Settings>,
    /// The command that runs the connector, before its own arguments: this
    /// executable and its `storage-server connector` command.
    pub connector_command: Vec<OsString>,
    /// The command that runs the Transfer Server, before its own arguments:
    /// this executable and its `storage-server transfer` command.
    pub transfer_command: Vec<OsString>,
    /// How long one chunk of the Transfer Server's copies lasts at its
    /// link's bandwidth.
    pub transfer_chunk: Duration,
    /// NIXL's C API library the Transfer Server's agent loads for copies
    /// between hosts; the loader's search path finds it when `None`.
    pub nixl_library: Option<PathBuf>,
    /// How long a buffer pool's transfer of an owner between worker groups
    /// may take before it is stopped and answered as failed.
    pub transfer_deadline: Duration,
    /// How long a buffer pool's put of host copies to the remote store may
    /// take before it is stopped and answered with the keys put so far.
    pub put_deadline: Duration,
    /// Record the trace of every buffer pool's inputs in this directory.
    pub buffer_pool_trace: Option<PathBuf>,
    /// Check every buffer pool's tables after each input it acts on.
    pub buffer_pool_check: bool,
    /// Go on with the server the predecessor of a restart in place passes on on
    /// the handoff socket at this descriptor, rather than binding the
    /// socket (`crate::restart`).
    pub predecessor: Option<i32>,
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
    let ready = if options.daemon && options.predecessor.is_none() {
        Some(service::detach(&options.socket).context("leaving the starter's session")?)
    } else {
        None
    };
    service::logging::init(
        "StorageServer",
        None,
        None,
        service::logging::Output::Stderr,
    );
    let served = match options.predecessor {
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
    service::logging::init(
        "MooncakeConnector",
        None,
        None,
        service::logging::Output::Stderr,
    );
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
    service::logging::init(
        "TransferServer",
        None,
        None,
        service::logging::Output::Stderr,
    );
    let ran = crate::transfer::run(socket, options).map_err(anyhow::Error::from);
    if let Err(error) = &ran {
        tracing::error!("the Transfer Server failed: {error:#}");
    }
    ran
}

/// Write to `out` what the server on `socket` holds, a line each: its
/// tensors, its deployments, its connector and Transfer Server, the quotas
/// of its GPUs, and the buffer pool of each deployment (`list_pool`).
///
/// # Errors
///
/// Fails when the server cannot be reached or refuses the listing, or `out`
/// cannot be written.
pub fn list(socket: &Path, out: &mut impl Write) -> anyhow::Result<()> {
    let mut client = Client::connect(socket, Role::Operator)?;
    let listing = client.list()?;
    for tensor in &listing.tensors {
        let scopes: Vec<String> = tensor.scopes.iter().map(ToString::to_string).collect();
        writeln!(
            out,
            "{} {:?} {} bytes {:?} {} [{}]{}",
            tensor.key.hex(),
            tensor.location,
            tensor.storage_bytes,
            tensor.access,
            tensor.name,
            scopes.join(", "),
            tensor
                .writer
                .map(|pid| format!(" writer {pid}"))
                .unwrap_or_default()
        )?;
    }
    for deployment in &listing.deployments {
        writeln!(
            out,
            "deployment {} head {}",
            deployment.deployment,
            if deployment.head_connected {
                "connected"
            } else {
                "gone"
            }
        )?;
    }
    if listing.connector {
        writeln!(out, "mooncake connector connected")?;
    }
    if let Some(transfer) = &listing.transfer {
        writeln!(out, "transfer server {} connected", transfer.pid)?;
        for context in &transfer.contexts {
            writeln!(
                out,
                "transfer server context on {}: {} bytes",
                platform::DeviceUuid(context.gpu),
                context.bytes
            )?;
        }
        for measured in &transfer.links {
            writeln!(
                out,
                "link {} -> {}: {:.1} GB/s",
                storage_name(&measured.link.source),
                storage_name(&measured.link.destination),
                measured.bytes_per_second as f64 / 1e9
            )?;
        }
        for task in &transfer.tasks {
            writeln!(
                out,
                "transfer {} of {} update {} {:?}: {} of {} bytes{}",
                task.task,
                task.deployment,
                task.update,
                task.priority,
                task.copied,
                task.bytes,
                if task.cancelled { ", cancelled" } else { "" }
            )?;
        }
    }
    for quota in &listing.quotas {
        writeln!(out, "{}", quota.describe())?;
    }
    for pool in &listing.buffer_pools {
        list_pool(pool, out)?;
    }
    Ok(())
}

/// Write one deployment's buffer pool as `specs/buffer-pool.md` §8 states
/// it, each line led by `buffer pool of <deployment>:`: the inconsistencies
/// its checks found, the blocks of each region and class in each observed
/// state, each worker group with the `completed` word last read and the
/// stamps its deferred work waits for, and each request whose copies are in
/// flight, with the time left to a task's deadline.
fn list_pool(pool: &BufferPoolListing, out: &mut impl Write) -> std::io::Result<()> {
    let of = format!("buffer pool of {}:", pool.deployment);
    writeln!(out, "{of} {} check failures", pool.check_failures)?;

    for count in &pool.blocks {
        let state = match count.state {
            BlockState::Shared => "shared",
            BlockState::Private => "private",
            BlockState::Retired => "retired",
        };
        let what = match count.class {
            Some(class) => format!("blocks of class {class}"),
            None => "slot rows".to_owned(),
        };
        writeln!(
            out,
            "{of} {} {state} {what} in region {}",
            count.blocks, count.region
        )?;
    }

    for group in &pool.groups {
        let waiting: Vec<String> = group
            .waiting
            .iter()
            .map(|(stamp, work)| {
                let work = match work {
                    Waiting::Release => "release",
                    Waiting::Commit => "commit",
                    Waiting::Transfer => "transfer",
                    Waiting::Finish => "finish",
                };
                format!("{work} at {stamp}")
            })
            .collect();
        let waiting = if waiting.is_empty() {
            "nothing waits".to_owned()
        } else {
            format!("waiting {}", waiting.join(", "))
        };
        writeln!(
            out,
            "{of} group {} of epoch {} completed {}, {waiting}",
            group.group, group.epoch, group.completed
        )?;
    }

    // A task's deadline is on the host's monotonic clock, which this
    // process, on the same host, reads too.
    let now = crate::sessions::polling::monotonic();
    for copy in &pool.copies {
        let session = match copy.epoch {
            Some(epoch) => format!("epoch {epoch}"),
            None => "a closed session".to_owned(),
        };
        let deadline = match copy.deadline {
            Some(deadline) if deadline > now => {
                format!(", due in {} ms", (deadline - now) / 1_000_000)
            }
            Some(_) => ", past its deadline".to_owned(),
            None => String::new(),
        };
        writeln!(
            out,
            "{of} {} copies in flight of request {} of scheduler {} of {session}{deadline}",
            copy.copies, copy.request, copy.scheduler
        )?;
    }
    Ok(())
}

/// Release the storage of `deployment`, which is no longer served, at the
/// server on `socket`.
///
/// # Errors
///
/// Fails when the server cannot be reached, or refuses because the
/// deployment's head is connected.
pub fn retire(socket: &Path, deployment: &str) -> ::storage::Result<()> {
    let mut client = Client::connect(socket, Role::Operator)?;
    let released = client.retire(deployment.to_owned())?;
    println!("released {released} tensors of {deployment}");
    Ok(())
}

/// Bind the socket and serve until stopped. `ready` is the starter's pipe,
/// written once the server listens or found another one listening.
fn serve_on(options: &ServeOptions, ready: Option<service::Ready>) -> anyhow::Result<()> {
    let socket = &options.socket;
    let signal = |ready: Option<service::Ready>| {
        if let Some(ready) = ready {
            ready.signal();
        }
    };
    let Some((listener, lock)) =
        service::bind(socket).with_context(|| format!("binding {}", socket.display()))?
    else {
        tracing::info!(socket = %socket.display(), "another storage server serves this socket");
        signal(ready);
        return Ok(());
    };
    service::raise_descriptor_limit().context("raising the descriptor limit")?;
    let (wake, stop) = service::stop_signals().context("installing the stop signals")?;
    let mut server = Server::new(
        listener,
        options.host_reserve_bytes,
        connector_launch(options),
        Some(transfer_launch(options)),
    )?;
    if let Some(address) = &options.tcp {
        let listener = std::net::TcpListener::bind(address)
            .with_context(|| format!("binding {address} for other hosts"))?;
        let bound = server.listen_tcp(listener)?;
        tracing::info!(address = %bound, "accepting other hosts' connections");
    }
    server.set_relaunch(crate::relaunch_arguments(std::env::args_os().collect()));
    server.set_deadlines(options.transfer_deadline, options.put_deadline);
    server.watch_pools(options.buffer_pool_trace.clone(), options.buffer_pool_check);
    tracing::info!(socket = %socket.display(), "storage server listening");
    signal(ready);
    run_until_stopped(server, &wake, &stop, lock)
}

/// Go on with the server a restart in place's predecessor passes on over the
/// handoff socket at `descriptor`, and serve until stopped.
fn serve_resumed(options: &ServeOptions, descriptor: i32) -> anyhow::Result<()> {
    service::raise_descriptor_limit().context("raising the descriptor limit")?;
    let (wake, stop) = service::stop_signals().context("installing the stop signals")?;
    let (mut server, lock) = Server::resume(
        descriptor,
        connector_launch(options),
        Some(transfer_launch(options)),
    )
    .context("taking the server over from its predecessor")?;
    server.set_deadlines(options.transfer_deadline, options.put_deadline);
    server.watch_pools(options.buffer_pool_trace.clone(), options.buffer_pool_check);
    let lock = service::SocketLock::resume(lock, options.socket.clone());
    tracing::info!(socket = %options.socket.display(), "storage server listening after a restart in place");
    run_until_stopped(server, &wake, &stop, lock)
}

/// Serve until stopped, or until a restart in place passed the server on.
fn run_until_stopped(
    mut server: Server,
    wake: &OwnedFd,
    stop: &AtomicBool,
    lock: service::SocketLock,
) -> anyhow::Result<()> {
    match server.run(wake.as_fd(), &|| stop.load(Ordering::Acquire))? {
        Some(pending) => {
            let pid = server
                .handoff(pending, lock.hand_over())
                .context("handing the server to its successor")?;
            tracing::info!(pid, "passed the server to its successor; exiting");
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
    if let Some(library) = &options.nixl_library {
        command.extend(["--nixl-library".into(), library.as_os_str().to_owned()]);
    }
    Launch {
        program: program.into(),
        arguments: command,
    }
}
