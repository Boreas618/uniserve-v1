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
//! The server runs its children from its own executable image ([`crate::child`]): with
//! connector settings the Mooncake connector, and always its Transfer Server, each started
//! again when it ends. A restart's successor goes on with the socket and the settings its
//! predecessor passes it ([`crate::restart`]).

use std::io::Write;
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use ::storage::protocol::{BlockState, BufferPoolListing, Waiting};
use ::storage::{Client, Role};
use anyhow::Context;
use service::host::Bound;
use service::{Endpoint, Endpoints, ReadyFd, State};

use crate::Server;
use crate::child::{Children, Program};
use crate::transfer::storage_name;

/// How a server starts.
#[derive(Debug)]
pub enum Start {
    /// It binds `socket` and serves with `settings`.
    Fresh {
        socket: PathBuf,
        settings: Box<::storage::Settings>,
    },
    /// It goes on with the server the predecessor of a restart in place passes it on this
    /// handoff socket, with that server's socket and settings.
    Successor(OwnedFd),
}

/// How to serve: the arguments of `uniserve storage-server serve`.
#[derive(Debug)]
pub struct ServeOptions {
    pub start: Start,
    /// Leave the starter's session, log beside the socket, and return to the
    /// starter once serving, which prints the readiness record.
    pub daemon: bool,
    /// The inherited descriptor the readiness record is written to once the
    /// server serves.
    pub ready_fd: Option<i32>,
    /// The executable the server runs its children from: its own.
    pub program: Program,
    /// The levels the server logs at.
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
    let ready = match (&options.start, options.daemon) {
        (Start::Fresh { socket, .. }, true) => Some(
            service::host::detach(socket, inherited).context("leaving the starter's session")?,
        ),
        _ => inherited,
    };
    service::logging::init(
        "StorageServer",
        &options.levels,
        service::logging::Output::Stderr,
    );
    let served = match options.start {
        Start::Fresh { socket, settings } => serve_on(&socket, *settings, options.program, ready),
        Start::Successor(handoff) => serve_resumed(handoff, options.program, ready),
    };
    if let Err(error) = &served {
        tracing::error!("storage server failed: {error:#}");
    }
    served
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
/// it, each line led by `buffer pool of <deployment>:`: the blocks of each region and class in each observed
/// state, each worker group with the `completed` word last read and the
/// stamps its deferred work waits for, and each request whose copies are in
/// flight, with the time left to a task's deadline.
fn list_pool(pool: &BufferPoolListing, out: &mut impl Write) -> std::io::Result<()> {
    let of = format!("buffer pool of {}:", pool.deployment);

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

/// Bind `socket` and serve with `settings` until stopped. `ready` is the starter's descriptor,
/// which receives the readiness record once the server serves, or the record of the server found
/// serving.
fn serve_on(
    socket: &Path,
    settings: ::storage::Settings,
    program: Program,
    ready: Option<ReadyFd>,
) -> anyhow::Result<()> {
    let (listener, starting) = match service::host::bind::<::storage::Settings>(socket)
        .with_context(|| format!("binding {}", socket.display()))?
    {
        Bound::Owner { listener, starting } => (listener, starting),
        Bound::Serving(record) => {
            // The server that owns the socket keeps its settings: a starter that states others
            // is refused rather than served with settings it did not state.
            ::storage::settings::Mismatch::check(&record, &settings)
                .with_context(|| format!("serving on {}", socket.display()))?;
            tracing::info!(socket = %socket.display(), pid = record.pid, "another storage server serves this socket");
            report(ready, &record);
            return Ok(());
        }
    };
    service::raise_descriptor_limit().context("raising the descriptor limit")?;
    let (wake, stop) = service::stop_signals().context("installing the stop signals")?;
    let children = Children::of(program, &settings);
    let mut server = Server::new(listener, settings.clone(), Some(children))?;
    let peer = match &settings.server.tcp {
        Some(endpoint) => {
            let bound = bind_tcp(endpoint)
                .and_then(|listener| server.listen_tcp(listener))
                .with_context(|| format!("binding {endpoint} for other hosts"))?;
            tracing::info!(endpoint = %bound, "accepting other hosts' connections");
            Some(bound)
        }
        None => None,
    };
    let record = readiness(socket, peer, &settings);
    let lock = starting
        .finish(&record)
        .context("publishing the readiness record")?;
    server.set_readiness(record.clone());
    tracing::info!(socket = %socket.display(), "storage server serving");
    report(ready, &record);
    run_until_stopped(server, &wake, &stop, lock)
}

/// Go on with the server a restart in place's predecessor passes on over the handoff socket
/// `handoff`, with its socket and settings, and serve until stopped.
fn serve_resumed(handoff: OwnedFd, program: Program, ready: Option<ReadyFd>) -> anyhow::Result<()> {
    service::raise_descriptor_limit().context("raising the descriptor limit")?;
    let (wake, stop) = service::stop_signals().context("installing the stop signals")?;
    let resumed = Server::resume(handoff, Some(program))
        .context("taking the server over from its predecessor")?;
    let mut server = resumed.server;
    let lock = service::host::SocketLock::resume(resumed.lock, resumed.socket.clone());
    // The record names this process now.
    let record = readiness(&resumed.socket, server.tcp_endpoint(), &resumed.settings);
    lock.publish(&record)
        .context("publishing the readiness record")?;
    server.set_readiness(record.clone());
    tracing::info!(socket = %resumed.socket.display(), "storage server serving after a restart in place");
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
