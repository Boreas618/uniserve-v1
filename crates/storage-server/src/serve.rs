//! Running a Storage Server, and the operator's commands against one.
//!
//! [`serve`] holds this host's retained storage for the calling user until it
//! is stopped by its `stop` request, SIGTERM or SIGINT. A process that starts a
//! server when its host has none starts it with `daemon` set
//! (`service::host::start`): the server then leaves its starter's session and
//! process tree, writes its log beside its socket and reports its readiness
//! record to its starter once it serves, so a starter's end never ends it.
//! The server binds its socket and its TCP endpoint before anything else, and
//! publishes the record beside its socket for every later client.
//!
//! With connector settings, the server also runs the Mooncake connector
//! ([`run_connector`]) as a child process it starts again when it ends. It
//! always runs its Transfer Server ([`run_transfer_server`]) the same way.

use std::ffi::OsString;
use std::io::Write;
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use ::storage::connector::Settings;
use ::storage::protocol::{BlockState, BufferPoolListing, Waiting};
use ::storage::{Client, Role};
use anyhow::Context;
use service::host::Bound;
use service::{Endpoint, Endpoints, ReadyFd, State};

use crate::transfer::storage_name;
use crate::{Launch, Server};

/// How to serve: the arguments of `uniserve storage-server serve`.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// The socket to listen on.
    pub socket: PathBuf,
    /// Leave the starter's session, log beside the socket, and return to the
    /// starter once serving, which prints the readiness record.
    pub daemon: bool,
    /// The inherited descriptor the readiness record is written to once the
    /// server serves.
    pub ready_fd: Option<i32>,
    /// What the server serves with.
    pub settings: ::storage::Settings,
    /// The command that runs the connector, before its own arguments: this
    /// executable and its `storage-server connector` command.
    pub connector_command: Vec<OsString>,
    /// The command that runs the Transfer Server, before its own arguments:
    /// this executable and its `storage-server transfer` command.
    pub transfer_command: Vec<OsString>,
    /// Go on with the server the predecessor of a restart in place passes on on
    /// the handoff socket at this descriptor, rather than binding the
    /// socket (`crate::restart`).
    pub predecessor: Option<i32>,
    /// The levels the server and the helpers it starts log at.
    pub levels: service::logging::Levels,
}

/// Serve until stopped. Must be called before the process starts a thread:
/// a daemon forks.
///
/// # Errors
///
/// Fails when the starter cannot be left, the socket cannot be bound, the
/// server cannot start, or it fails while serving.
pub fn serve(options: ServeOptions) -> anyhow::Result<()> {
    let inherited = options
        .ready_fd
        // SAFETY: the starter passed this descriptor for the readiness record; nothing else in
        // this process owns it.
        .map(|fd| unsafe { ReadyFd::inherit(fd) })
        .transpose()
        .context("taking the readiness descriptor")?;
    // Forked before any thread exists. A successor runs detached already.
    let ready = if options.daemon && options.predecessor.is_none() {
        Some(
            service::host::detach(&options.socket, inherited)
                .context("leaving the starter's session")?,
        )
    } else {
        inherited
    };
    service::logging::init(
        "StorageServer",
        &options.levels,
        service::logging::Output::Stderr,
    );
    let served = match options.predecessor {
        Some(descriptor) => serve_resumed(&options, descriptor, ready),
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
pub fn run_connector(
    socket: &Path,
    settings: &Settings,
    levels: &service::logging::Levels,
) -> anyhow::Result<()> {
    service::logging::init(
        "MooncakeConnector",
        levels,
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
    levels: &service::logging::Levels,
) -> anyhow::Result<()> {
    service::logging::init("TransferServer", levels, service::logging::Output::Stderr);
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
                    Waiting::Replicate => "replicate",
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

/// Binds the TCP `endpoint` other hosts reach the server at (port 0 picks one).
fn bind_tcp(endpoint: &Endpoint) -> std::io::Result<std::net::TcpListener> {
    match endpoint {
        Endpoint::Tcp { host, port } => std::net::TcpListener::bind((host.as_str(), *port)),
        Endpoint::Unix(_) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("other hosts reach a server over TCP, not at {endpoint}"),
        )),
    }
}

/// The readiness record of a server of this process on `socket`, accepting other hosts on
/// `peer`, serving with `settings`.
pub(crate) fn readiness(
    socket: &Path,
    peer: Option<Endpoint>,
    settings: &::storage::Settings,
) -> ::storage::Readiness {
    service::Readiness::of_process(
        State::Serving,
        None,
        Endpoints {
            storage: Some(Endpoint::unix(socket)),
            peer,
            ..Endpoints::default()
        },
        settings.clone(),
    )
}

/// Writes `record` to the starter's readiness descriptor, which closes then; a starter that is
/// gone loses nothing the server needs.
fn report(ready: Option<ReadyFd>, record: &::storage::Readiness) {
    if let Some(mut ready) = ready
        && let Err(error) = ready.report(record)
    {
        tracing::warn!(%error, "the starter did not take the readiness record");
    }
}

/// Bind the socket and serve until stopped. `ready` is the starter's descriptor, which receives
/// the readiness record once the server serves, or the record of the server found serving.
fn serve_on(options: &ServeOptions, ready: Option<ReadyFd>) -> anyhow::Result<()> {
    let socket = &options.socket;
    let (listener, starting) = match service::host::bind::<::storage::Settings>(socket)
        .with_context(|| format!("binding {}", socket.display()))?
    {
        Bound::Owner { listener, starting } => (listener, starting),
        Bound::Serving(record) => {
            // The server that owns the socket keeps its settings: a starter that states others
            // is refused rather than served with settings it did not state.
            ::storage::settings::Mismatch::check(&record, &options.settings)
                .with_context(|| format!("serving on {}", socket.display()))?;
            tracing::info!(socket = %socket.display(), pid = record.pid, "another storage server serves this socket");
            report(ready, &record);
            return Ok(());
        }
    };
    service::raise_descriptor_limit().context("raising the descriptor limit")?;
    let (wake, stop) = service::stop_signals().context("installing the stop signals")?;
    let server_settings = &options.settings.server;
    let mut server = Server::new(
        listener,
        server_settings.host_reserve_bytes(),
        connector_launch(options),
        Some(transfer_launch(options)),
    )?;
    let peer = match &server_settings.tcp {
        Some(endpoint) => {
            let bound = bind_tcp(endpoint)
                .and_then(|listener| server.listen_tcp(listener))
                .with_context(|| format!("binding {endpoint} for other hosts"))?;
            tracing::info!(endpoint = %bound, "accepting other hosts' connections");
            Some(bound)
        }
        None => None,
    };
    server.set_relaunch(crate::relaunch_arguments(std::env::args_os().collect()));
    server.set_deadlines(
        server_settings.transfer_deadline(),
        server_settings.put_deadline(),
    );
    server.watch_pools(
        server_settings.buffer_pool_trace.clone(),
        server_settings.buffer_pool_check,
    );
    let record = readiness(socket, peer, &options.settings);
    let lock = starting
        .finish(&record)
        .context("publishing the readiness record")?;
    server.set_readiness(record.clone());
    tracing::info!(socket = %socket.display(), "storage server serving");
    report(ready, &record);
    run_until_stopped(server, &wake, &stop, lock)
}

/// Go on with the server a restart in place's predecessor passes on over the
/// handoff socket at `descriptor`, and serve until stopped. The predecessor states its settings
/// on this process's standard input, so the successor serves with them.
fn serve_resumed(
    options: &ServeOptions,
    descriptor: i32,
    ready: Option<ReadyFd>,
) -> anyhow::Result<()> {
    service::raise_descriptor_limit().context("raising the descriptor limit")?;
    let (wake, stop) = service::stop_signals().context("installing the stop signals")?;
    let (mut server, lock) = Server::resume(
        descriptor,
        connector_launch(options),
        Some(transfer_launch(options)),
    )
    .context("taking the server over from its predecessor")?;
    let server_settings = &options.settings.server;
    server.set_deadlines(
        server_settings.transfer_deadline(),
        server_settings.put_deadline(),
    );
    server.watch_pools(
        server_settings.buffer_pool_trace.clone(),
        server_settings.buffer_pool_check,
    );
    let lock = service::host::SocketLock::resume(lock, options.socket.clone());
    // The record names this process now.
    let record = readiness(&options.socket, server.tcp_endpoint(), &options.settings);
    lock.publish(&record)
        .context("publishing the readiness record")?;
    server.set_readiness(record.clone());
    tracing::info!(socket = %options.socket.display(), "storage server serving after a restart in place");
    report(ready, &record);
    run_until_stopped(server, &wake, &stop, lock)
}

/// Serve until stopped, or until a restart in place passed the server on.
fn run_until_stopped(
    mut server: Server,
    wake: &OwnedFd,
    stop: &AtomicBool,
    lock: service::host::SocketLock,
) -> anyhow::Result<()> {
    match server.run(wake.as_fd(), &|| stop.load(Ordering::Acquire))? {
        Some(pending) => {
            let pid = server
                .handoff(pending, lock.hand_over())
                .context("handing the server to its successor")?;
            tracing::info!(pid, "passed the server to its successor; exiting");
        }
        None => {
            // The socket and the record go before the connections close, so a client that sees
            // its connection close finds no server owning the socket.
            drop(lock);
            drop(server);
            tracing::info!("storage server stopped");
        }
    }
    Ok(())
}

/// The connector the server runs, from `options`.
fn connector_launch(options: &ServeOptions) -> Option<Launch> {
    options.settings.connector.as_ref().map(|settings| {
        let mut command = options.connector_command.clone();
        let program = command.remove(0);
        command.extend(["--socket".into(), options.socket.as_os_str().to_owned()]);
        Launch {
            program: program.into(),
            arguments: command,
            input: Some(connector_document(settings)),
        }
    })
}

/// The connector's settings as the document its command reads on its standard input.
fn connector_document(settings: &Settings) -> String {
    // Every field has a TOML form: strings, an endpoint as text and integers.
    toml::to_string(settings).unwrap_or_default()
}

/// The Transfer Server the server runs, from `options`.
fn transfer_launch(options: &ServeOptions) -> Launch {
    let mut command = options.transfer_command.clone();
    let program = command.remove(0);
    command.extend([
        "--socket".into(),
        options.socket.as_os_str().to_owned(),
        "--chunk-us".into(),
        options.settings.server.transfer_chunk_us.to_string().into(),
    ]);
    if let Some(library) = &options.settings.server.nixl_library {
        command.extend(["--nixl-library".into(), library.as_os_str().to_owned()]);
    }
    Launch {
        program: program.into(),
        arguments: command,
        input: None,
    }
}
