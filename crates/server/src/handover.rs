//! Control-plane hot update: a head hands its deployment to a head of a new
//! build on the same host without failing a request or closing a client
//! connection.
//!
//! Every head offers its deployment on a handover socket ([`socket`]), a
//! Unix socket beside this host's Storage Server socket named after the
//! deployment ([`Offer`]). A head that starts for a deployment another head
//! offers connects to it and becomes its successor ([`Predecessor`]):
//!
//! 1. The successor states its versions and engine core count (`Hello`).
//!    The predecessor's engine cores stop their standby groups, and it
//!    answers with what the successor's workers adopt and a duplicate of its
//!    HTTP listening socket (`Welcome`).
//! 2. The successor launches its workers standing by for the predecessor's
//!    (`EngineCore::start_successors`) and, once they are parked, asks for
//!    the handover (`Ready`).
//! 3. The predecessor quiesces every engine core (`EngineCore::quiesce`),
//!    gives its handover socket up, sends the snapshots (`Snapshots`) and
//!    stops accepting connections: the successor accepts them from then on.
//! 4. The successor resumes each engine core (`EngineCore::resume`), offers
//!    the deployment itself, and serves.
//! 5. The predecessor keeps serving the connections it holds. Every command
//!    its clients' streams send reaches the successor (`Command`), as does a
//!    token request one of them submits (`Submit`), and every event of those
//!    requests comes back (`Event`). It exits once its last connection
//!    closes.
//!
//! A successor that leaves before it asks for the handover leaves the
//! predecessor serving, with its standbys started again; one that leaves
//! later fails the requests it took.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::os::fd::{AsFd as _, AsRawFd as _, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use sha2::Digest as _;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uniserve_core::{EngineCoreOutput, GenerationRequest, RejectionKind, Request, RequestId};
use uniserve_engine::handover::{Adoption, CoreSnapshot, SNAPSHOT_VERSION};
use uniserve_engine::{Command, EngineCore, EventRx, EventTx, Submission};
use uniserve_storage::transport::Connection;

use crate::http::listener::{HandedListener, ListenerAddress};

/// Version of the handover protocol. Each head refuses a peer that states
/// another, or another snapshot version.
const VERSION: u32 = 1;

/// How long a successor waits for its predecessor to give the handover
/// socket up.
const OFFER_TIMEOUT: Duration = Duration::from_secs(10);

/// How often an offering head checks whether it stopped serving.
const OFFER_POLL: Duration = Duration::from_millis(200);

/// The handover socket of `deployment` on this host. Its name is short, as
/// a socket path holds at most 107 bytes; two deployments whose names share
/// it refuse each other (`Hello`).
pub(crate) fn socket(deployment: &str) -> PathBuf {
    let digest = sha2::Sha256::digest(deployment.as_bytes());
    let name: String = digest[..4]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    uniserve_storage::socket::default_socket().with_file_name(format!("handover-{name}.sock"))
}

/// What a successor sends.
#[derive(Debug, Serialize, Deserialize)]
enum ToPredecessor {
    Hello {
        protocol: u32,
        snapshot: u32,
        deployment: String,
        cores: usize,
    },
    /// The successor's workers stand by: quiesce and hand over.
    Ready,
    /// An event of a request the successor runs for the predecessor's
    /// client, by the predecessor's engine core and request identity.
    Event {
        core: usize,
        request: u64,
        event: EngineCoreOutput,
    },
}

/// What a predecessor sends.
#[derive(Debug, Serialize, Deserialize)]
enum ToSuccessor {
    /// Accepts the successor; carries the listening socket's descriptor.
    Welcome {
        adoption: Option<Adoption>,
        listener: ListenerAddress,
    },
    Refused {
        reason: String,
    },
    Snapshots {
        cores: Vec<CoreSnapshot>,
    },
    /// A client's command for one of the requests the successor runs.
    Command {
        core: usize,
        request: u64,
        command: RequestCommand,
    },
    /// A token request a client of the predecessor submitted after the
    /// handover, under the predecessor's request identity.
    Submit {
        core: usize,
        request: Box<GenerationRequest>,
    },
}

/// A client's command for one request (`uniserve_engine::Command`).
#[derive(Debug, Serialize, Deserialize)]
enum RequestCommand {
    Cancel,
    Stop,
    Acknowledge { output_token_count: usize },
    Abort,
}

fn terminal(event: &EngineCoreOutput) -> bool {
    matches!(
        event,
        EngineCoreOutput::Finished { .. }
            | EngineCoreOutput::Rejected { .. }
            | EngineCoreOutput::Error { .. }
    )
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A head's offer of its deployment to a successor.
pub(crate) struct Offer {
    deployment: String,
    listener: UnixListener,
    _lock: uniserve_storage::socket::SocketLock,
}

impl Offer {
    /// Offers `deployment` on its handover socket `path` ([`socket`]). A
    /// predecessor that just handed over gives the socket up before it sends
    /// its snapshots; the offer waits for that.
    ///
    /// Fails when another head keeps offering the deployment, or the socket
    /// cannot be bound.
    pub(crate) fn bind(path: &Path, deployment: &str) -> anyhow::Result<Self> {
        let deadline = Instant::now() + OFFER_TIMEOUT;
        loop {
            if let Some((listener, lock)) = uniserve_storage::socket::bind(path)
                .with_context(|| format!("binding the handover socket {}", path.display()))?
            {
                return Ok(Self {
                    deployment: deployment.to_owned(),
                    listener,
                    _lock: lock,
                });
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "another head offers the deployment at {}",
                path.display()
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Offers the deployment until a successor takes it over or `stopped`
    /// is cancelled, on a thread of its own. `cores` are the head's engine
    /// cores and `listener` its HTTP listening socket. `handed_over` is
    /// cancelled once the successor holds the requests, when the head stops
    /// accepting connections; the relay then serves the head's clients.
    pub(crate) fn serve(
        self,
        cores: Vec<Arc<EngineCore>>,
        listener: HandedListener,
        handed_over: CancellationToken,
        stopped: CancellationToken,
        runtime: Handle,
    ) -> io::Result<std::thread::JoinHandle<()>> {
        std::thread::Builder::new()
            .name("handover".into())
            .spawn(move || self.offer(&cores, &listener, &handed_over, &stopped, &runtime))
    }

    fn offer(
        self,
        cores: &[Arc<EngineCore>],
        listener: &HandedListener,
        handed_over: &CancellationToken,
        stopped: &CancellationToken,
        runtime: &Handle,
    ) {
        while !stopped.is_cancelled() {
            if !readable(&self.listener, OFFER_POLL) {
                continue;
            }
            let stream = match self.listener.accept() {
                Ok((stream, _)) => stream,
                Err(error) => {
                    tracing::warn!(%error, "accepting a successor failed");
                    continue;
                }
            };
            let mut peer = match Peer::new(stream) {
                Ok(peer) => peer,
                Err(error) => {
                    tracing::warn!(%error, "a successor's connection failed");
                    continue;
                }
            };
            if let Err(error) = welcome(&mut peer, &self.deployment, cores, listener) {
                tracing::warn!("a successor was not accepted: {error:#}");
                continue;
            }
            match peer.reader.recv::<ToPredecessor>() {
                Ok((ToPredecessor::Ready, _)) => {}
                outcome => {
                    if let Err(error) = outcome {
                        tracing::warn!(%error, "the successor left before it took over");
                    } else {
                        tracing::warn!("the successor broke the handover protocol");
                    }
                    for core in cores {
                        core.abandon_handover();
                    }
                    continue;
                }
            }
            // From here on the requests move: the offer ends.
            hand_over(self, peer, cores, listener, handed_over, runtime);
            return;
        }
    }
}

/// Whether `listener` has a connection to accept within `timeout`.
fn readable(listener: &UnixListener, timeout: Duration) -> bool {
    let mut poll = libc::pollfd {
        fd: listener.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one valid pollfd for the duration of the call.
    let ready = unsafe { libc::poll(&raw mut poll, 1, timeout.as_millis() as libc::c_int) };
    ready > 0 && poll.revents & libc::POLLIN != 0
}

/// The two directions of a handover connection.
struct Peer {
    reader: Connection,
    writer: Connection,
}

impl Peer {
    fn new(stream: UnixStream) -> io::Result<Self> {
        stream.set_nonblocking(false)?;
        Ok(Self {
            reader: Connection::new(stream.try_clone()?),
            writer: Connection::new(stream),
        })
    }
}

/// Answers a successor's `Hello`: refuses one of other versions, another
/// deployment or another engine core count, and otherwise prepares every
/// engine core and welcomes it.
/// Fails, with every core serving as before, when it refused the successor
/// or the successor left.
fn welcome(
    peer: &mut Peer,
    deployment: &str,
    cores: &[Arc<EngineCore>],
    listener: &HandedListener,
) -> anyhow::Result<()> {
    let (hello, _) = peer.reader.recv::<ToPredecessor>()?;
    let refusal = match hello {
        ToPredecessor::Hello {
            protocol,
            snapshot,
            deployment: theirs,
            cores: count,
        } => {
            if protocol != VERSION || snapshot != SNAPSHOT_VERSION {
                Some(format!(
                    "the successor states handover protocol {protocol} and snapshot {snapshot}; \
                     this head reads protocol {VERSION} and snapshot {SNAPSHOT_VERSION}"
                ))
            } else if theirs != deployment {
                Some(format!(
                    "the successor serves deployment {theirs}; this head serves {deployment}"
                ))
            } else if count != cores.len() {
                Some(format!(
                    "the successor runs {count} engine cores; this head runs {}",
                    cores.len()
                ))
            } else {
                None
            }
        }
        _ => Some("the successor did not begin with its versions".into()),
    };
    if let Some(reason) = refusal {
        peer.writer.send(
            &ToSuccessor::Refused {
                reason: reason.clone(),
            },
            &[],
        )?;
        anyhow::bail!(reason);
    }

    // Each engine core's groups stop their standbys; the successor's
    // workers adopt the claim and cache units they report.
    let mut adoption: Option<Adoption> = None;
    let mut prepared = 0;
    let mut failure = None;
    for core in cores {
        match core.prepare_handover() {
            Ok(Some(core_adoption)) => {
                prepared += 1;
                match adoption.as_mut() {
                    Some(adoption) => adoption.cache_units.extend(core_adoption.cache_units),
                    None => adoption = Some(core_adoption),
                }
            }
            Ok(None) => prepared += 1,
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    if let Some(error) = failure {
        for core in &cores[..prepared] {
            core.abandon_handover();
        }
        let reason = format!("{error:#}");
        peer.writer.send(&ToSuccessor::Refused { reason }, &[])?;
        return Err(error);
    }
    let welcomed = peer.writer.send(
        &ToSuccessor::Welcome {
            adoption,
            listener: listener.address.clone(),
        },
        &[listener.descriptor.as_fd()],
    );
    if let Err(error) = welcomed {
        for core in cores {
            core.abandon_handover();
        }
        return Err(error.into());
    }
    Ok(())
}

/// The predecessor's handover once its successor is ready: quiesces every
/// engine core, gives the offer up, sends the snapshots, stops accepting
/// connections, and relays its clients' requests.
fn hand_over(
    offer: Offer,
    peer: Peer,
    cores: &[Arc<EngineCore>],
    listener: &HandedListener,
    handed_over: &CancellationToken,
    runtime: &Handle,
) {
    let quiesced: Vec<anyhow::Result<uniserve_engine::handover::Quiesced>> =
        std::thread::scope(|scope| {
            let quiescing: Vec<_> = cores
                .iter()
                .map(|core| scope.spawn(|| core.quiesce()))
                .collect();
            quiescing
                .into_iter()
                .map(|thread| {
                    thread
                        .join()
                        .unwrap_or_else(|_| Err(anyhow::anyhow!("quiescing panicked")))
                })
                .collect()
        });
    // The successor offers the deployment once it holds the requests.
    drop(offer);

    let Peer { reader, mut writer } = peer;
    let mut snapshots = Vec::with_capacity(quiesced.len());
    let mut relays = Vec::with_capacity(quiesced.len());
    for (index, quiesced) in quiesced.into_iter().enumerate() {
        match quiesced {
            Ok(quiesced) => {
                snapshots.push(quiesced.snapshot);
                relays.push((index, quiesced.streams, quiesced.commands));
            }
            Err(error) => {
                tracing::error!("engine core {index} could not hand over: {error:#}");
                return;
            }
        }
    }
    if let Err(error) = writer.send(&ToSuccessor::Snapshots { cores: snapshots }, &[]) {
        tracing::error!(%error, "the successor left during the handover");
        return;
    }
    // New connections reach the successor from now on, and the socket file
    // of a Unix-domain listener is its own.
    listener.release();
    handed_over.cancel();
    tracing::info!("handed the deployment over to the successor head");

    let relay = Arc::new(ClientRelay {
        writer: Mutex::new(writer),
        streams: Mutex::new(HashMap::new()),
        runtime: runtime.clone(),
    });
    for (core, streams, commands) in relays {
        for stream in streams {
            relay.add(core, stream.request_id, stream.events, stream.pending);
        }
        let relay = Arc::clone(&relay);
        let spawned = std::thread::Builder::new()
            .name(format!("handover-commands-{core}"))
            .spawn(move || relay.forward_commands(core, &commands));
        if let Err(error) = spawned {
            tracing::error!(%error, "the command relay of engine core {core} could not start");
        }
    }
    relay.forward_events(reader);
}

/// The predecessor's relay between its clients' streams and the successor.
struct ClientRelay {
    writer: Mutex<Connection>,
    /// The ordered delivery of each relayed request's events to its
    /// client, by engine core and request identity.
    streams: Mutex<HashMap<(usize, u64), mpsc::UnboundedSender<EngineCoreOutput>>>,
    runtime: Handle,
}

impl ClientRelay {
    /// Relays a request's events to its client: `pending` first, then what
    /// the successor sends. Delivery waits for room in the client's channel.
    fn add(
        &self,
        core: usize,
        request: RequestId,
        events: EventTx,
        pending: VecDeque<EngineCoreOutput>,
    ) {
        let (sender, mut received) = mpsc::unbounded_channel();
        self.runtime.spawn(async move {
            for event in pending {
                if events.deliver(event).await.is_err() {
                    return;
                }
            }
            while let Some(event) = received.recv().await {
                let last = terminal(&event);
                if events.deliver(event).await.is_err() || last {
                    return;
                }
            }
        });
        lock(&self.streams).insert((core, request.0), sender);
    }

    /// Sends what engine core `core`'s handles command to the successor,
    /// until the core shuts down or the successor leaves.
    fn forward_commands(&self, core: usize, commands: &crossbeam_channel::Receiver<Command>) {
        while let Ok(command) = commands.recv() {
            let (request, command) = match command {
                Command::Cancel(id) => (id, RequestCommand::Cancel),
                Command::Stop(id) => (id, RequestCommand::Stop),
                Command::Acknowledge {
                    request_id,
                    output_token_count,
                } => (
                    request_id,
                    RequestCommand::Acknowledge { output_token_count },
                ),
                Command::Abort(id) => (id, RequestCommand::Abort),
                Command::Submit(submissions) => {
                    for submission in submissions {
                        if self.submit(core, submission).is_err() {
                            return;
                        }
                    }
                    continue;
                }
                Command::Shutdown => return,
                Command::PrepareHandover(reply) => {
                    let _ = reply.send(Err(anyhow::anyhow!("the engine core handed over")));
                    continue;
                }
                Command::Quiesce(reply) => {
                    let _ = reply.send(Err(anyhow::anyhow!("the engine core handed over")));
                    continue;
                }
                Command::Resume { reply, .. } => {
                    let _ = reply.send(Err(anyhow::anyhow!("the engine core handed over")));
                    continue;
                }
                Command::AbandonHandover => continue,
            };
            let message = ToSuccessor::Command {
                core,
                request: request.0,
                command,
            };
            if lock(&self.writer).send(&message, &[]).is_err() {
                return;
            }
        }
    }

    /// Forwards a token request a client submitted after the handover. The
    /// successor serves no other kind for a predecessor's client, whose
    /// events include media artifacts: it is refused as overloaded, and the
    /// client's retry reaches the successor.
    fn submit(&self, core: usize, submission: Submission) -> io::Result<()> {
        let Submission { request, event_tx } = submission;
        let request = match *request {
            Request::Generation(request) => request,
            Request::Diffusion(_) => {
                let _ = event_tx.send(EngineCoreOutput::Rejected {
                    kind: RejectionKind::Overloaded,
                    message: "the server handed its deployment over; retry the request".into(),
                });
                return Ok(());
            }
        };
        self.add(core, request.request_id, event_tx, VecDeque::new());
        lock(&self.writer).send(&ToSuccessor::Submit { core, request }, &[])
    }

    /// Delivers the successor's events until it leaves; the requests still
    /// relayed then fail.
    fn forward_events(&self, mut reader: Connection) {
        loop {
            match reader.recv::<ToPredecessor>() {
                Ok((
                    ToPredecessor::Event {
                        core,
                        request,
                        event,
                    },
                    _,
                )) => {
                    let last = terminal(&event);
                    let mut streams = lock(&self.streams);
                    if let Some(stream) = streams.get(&(core, request)) {
                        let _ = stream.send(event);
                    }
                    if last {
                        streams.remove(&(core, request));
                    }
                }
                Ok((message, _)) => {
                    tracing::warn!(?message, "the successor broke the relay protocol");
                    break;
                }
                Err(error) => {
                    if error.kind() != io::ErrorKind::UnexpectedEof {
                        tracing::warn!(%error, "the relay to the successor failed");
                    }
                    break;
                }
            }
        }
        for (_, stream) in lock(&self.streams).drain() {
            let _ = stream.send(EngineCoreOutput::Error {
                message: "the successor head stopped".into(),
            });
        }
    }
}

/// A successor's connection to the head it succeeds.
pub(crate) struct Predecessor {
    peer: Peer,
    adoption: Option<Adoption>,
    listener: OwnedFd,
    address: ListenerAddress,
}

impl Predecessor {
    /// Connects to the head offering `deployment` on its handover socket
    /// `path` ([`socket`]), when one does, as the successor of a head of
    /// `cores` engine cores. The predecessor's cores stop their standbys for
    /// the successor's workers.
    ///
    /// Fails when the head refuses the successor or breaks the protocol.
    pub(crate) fn connect(
        path: &Path,
        deployment: &str,
        cores: usize,
    ) -> anyhow::Result<Option<Self>> {
        let stream = match UnixStream::connect(path) {
            Ok(stream) => stream,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
                ) =>
            {
                return Ok(None);
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("connecting to the head at {}", path.display()));
            }
        };
        let mut peer = Peer::new(stream)?;
        peer.writer.send(
            &ToPredecessor::Hello {
                protocol: VERSION,
                snapshot: SNAPSHOT_VERSION,
                deployment: deployment.to_owned(),
                cores,
            },
            &[],
        )?;
        match peer.reader.recv::<ToSuccessor>()? {
            (ToSuccessor::Welcome { adoption, listener }, mut descriptors) => {
                let descriptor = descriptors
                    .pop()
                    .context("the predecessor sent no listening socket")?;
                tracing::info!("succeeding the running head of the deployment");
                Ok(Some(Self {
                    peer,
                    adoption,
                    listener: descriptor,
                    address: listener,
                }))
            }
            (ToSuccessor::Refused { reason }, _) => {
                anyhow::bail!("the running head refused the handover: {reason}")
            }
            (message, _) => anyhow::bail!("the running head answered {message:?} to its successor"),
        }
    }

    /// What the successor's workers adopt.
    pub(crate) fn adoption(&self) -> Option<&Adoption> {
        self.adoption.as_ref()
    }

    /// Takes the deployment over once `cores`, started over standby workers,
    /// are ready: asks for the handover, resumes each engine core with the
    /// predecessor's snapshot, and relays its clients' requests until it
    /// leaves. Returns the listening socket the predecessor served on.
    ///
    /// Fails when the predecessor leaves before it sends the snapshots or
    /// an engine core cannot resume.
    pub(crate) fn take_over(
        self,
        cores: Vec<Arc<EngineCore>>,
        runtime: &Handle,
    ) -> anyhow::Result<(OwnedFd, ListenerAddress)> {
        let Peer {
            mut reader,
            mut writer,
        } = self.peer;
        writer.send(&ToPredecessor::Ready, &[])?;
        let snapshots = match reader
            .recv::<ToSuccessor>()
            .context("the predecessor left during the handover")?
        {
            (ToSuccessor::Snapshots { cores }, _) => cores,
            (message, _) => anyhow::bail!("the predecessor answered {message:?} to its successor"),
        };
        anyhow::ensure!(
            snapshots.len() == cores.len(),
            "the predecessor sent {} snapshots for {} engine cores",
            snapshots.len(),
            cores.len()
        );
        let mut resumed = Vec::with_capacity(cores.len());
        for (index, (core, snapshot)) in cores.iter().zip(snapshots).enumerate() {
            resumed.push(
                core.resume(snapshot)
                    .with_context(|| format!("resuming engine core {index}"))?,
            );
        }
        tracing::info!(
            requests = resumed.iter().map(Vec::len).sum::<usize>(),
            "took the deployment over from the predecessor head"
        );

        // One thread writes the events of every relayed request; the
        // relay ends when the predecessor leaves.
        let (outbox, events) = crossbeam_channel::unbounded::<ToPredecessor>();
        let gone = CancellationToken::new();
        {
            let gone = gone.clone();
            std::thread::Builder::new()
                .name("handover-events".into())
                .spawn(move || {
                    while let Ok(message) = events.recv() {
                        if writer.send(&message, &[]).is_err() {
                            break;
                        }
                    }
                    gone.cancel();
                })?;
        }
        let relay = PredecessorRelay {
            cores,
            outbox,
            gone,
            runtime: runtime.clone(),
        };
        for (core, streams) in resumed.into_iter().enumerate() {
            for (request, events) in streams {
                relay.forward(core, request.0, events);
            }
        }
        std::thread::Builder::new()
            .name("handover-commands".into())
            .spawn(move || relay.apply_commands(reader))?;
        Ok((self.listener, self.address))
    }
}

/// The successor's relay of the predecessor's clients' requests.
struct PredecessorRelay {
    cores: Vec<Arc<EngineCore>>,
    outbox: crossbeam_channel::Sender<ToPredecessor>,
    /// Cancelled once the predecessor left: the relayed requests are then
    /// cancelled.
    gone: CancellationToken,
    runtime: Handle,
}

impl PredecessorRelay {
    /// Sends a relayed request's events to the predecessor under its
    /// identity there, until its terminal event or the predecessor leaves,
    /// which drops `events` and so cancels the request.
    fn forward(&self, core: usize, request: u64, mut events: EventRx) {
        let outbox = self.outbox.clone();
        let gone = self.gone.clone();
        self.runtime.spawn(async move {
            loop {
                let event = tokio::select! {
                    event = events.recv() => event,
                    () = gone.cancelled() => return,
                };
                let Some(event) = event else {
                    return;
                };
                let last = terminal(&event);
                let message = ToPredecessor::Event {
                    core,
                    request,
                    event,
                };
                if outbox.send(message).is_err() || last {
                    return;
                }
            }
        });
    }

    /// Applies the predecessor's clients' commands and submissions until it
    /// leaves.
    fn apply_commands(self, mut reader: Connection) {
        // The engine identity of each request the predecessor submitted
        // after the handover; a migrated request keeps its identity.
        let mut submitted: HashMap<(usize, u64), RequestId> = HashMap::new();
        loop {
            let message = match reader.recv::<ToSuccessor>() {
                Ok((message, _)) => message,
                Err(error) => {
                    if error.kind() != io::ErrorKind::UnexpectedEof {
                        tracing::warn!(%error, "the relay from the predecessor failed");
                    }
                    break;
                }
            };
            match message {
                ToSuccessor::Command {
                    core,
                    request,
                    command,
                } => {
                    let Some(engine) = self.cores.get(core) else {
                        continue;
                    };
                    let id = submitted
                        .get(&(core, request))
                        .copied()
                        .unwrap_or(RequestId(request));
                    let handle = engine.handle();
                    match command {
                        RequestCommand::Cancel => handle.cancel(id),
                        RequestCommand::Stop => handle.stop(id),
                        RequestCommand::Acknowledge { output_token_count } => {
                            handle.acknowledge_at(id, output_token_count);
                        }
                        RequestCommand::Abort => handle.abort(id),
                    }
                }
                ToSuccessor::Submit { core, mut request } => {
                    let predecessor_id = request.request_id.0;
                    let Some(engine) = self.cores.get(core) else {
                        continue;
                    };
                    let id = engine.next_request_id();
                    request.request_id = id;
                    submitted.insert((core, predecessor_id), id);
                    match engine.submit(Request::Generation(request)) {
                        Ok(events) => self.forward(core, predecessor_id, events),
                        Err(error) => {
                            let _ = self.outbox.send(ToPredecessor::Event {
                                core,
                                request: predecessor_id,
                                event: EngineCoreOutput::Error {
                                    message: error.to_string(),
                                },
                            });
                        }
                    }
                }
                message => {
                    tracing::warn!(?message, "the predecessor broke the relay protocol");
                    break;
                }
            }
        }
        self.gone.cancel();
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use tokio::sync::mpsc::error::TryRecvError;
    use uniserve_core::{
        EngineCoreOutput, FinishReason, GenerationConstraint, GenerationRequest,
        ImageGenerationConfig, ImageParams, RequestId, SamplingParams,
    };
    use uniserve_engine::{EngineConfig, EngineCore, EventRx, SimEngine, SimExecutor};

    use super::*;
    use crate::http::listener::Listener;

    const TEXT_LEN: usize = 300;

    fn core() -> Arc<EngineCore> {
        let mut sim = SimEngine::new();
        sim.set_text_len(TEXT_LEN);
        Arc::new(
            EngineCore::with_executor(
                EngineConfig::sim("sim-model"),
                Box::new(SimExecutor::new(sim)),
            )
            .unwrap(),
        )
    }

    fn request(id: RequestId) -> Request {
        GenerationRequest {
            request_id: id,
            prompt_token_ids: (0..100).map(|i| i % 89 + 5).collect(),
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint: GenerationConstraint::UndOnly,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_images: 1,
            max_und_tokens: 2 * TEXT_LEN,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: ImageGenerationConfig::default(),
            readout: Vec::new(),
        }
        .into()
    }

    /// Reads `events` until a terminal event: the token ids and the finish.
    async fn read(events: &mut EventRx) -> (Vec<u32>, Option<FinishReason>) {
        let mut tokens = Vec::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
        loop {
            let event = tokio::time::timeout_at(deadline, events.recv())
                .await
                .expect("the request finishes");
            match event.expect("the stream ends with its terminal event") {
                EngineCoreOutput::TextToken { id, .. } => tokens.push(id),
                EngineCoreOutput::Finished { reason, .. } => return (tokens, Some(reason)),
                EngineCoreOutput::Error { message } => panic!("the request failed: {message}"),
                _ => {}
            }
        }
    }

    /// A head hands its deployment to a successor over the handover socket:
    /// a request it was generating finishes through the relay with the
    /// tokens it would have generated alone, the successor serves on the
    /// head's listening socket, and a request a client of the head submits
    /// afterwards runs on the successor.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_head_hands_its_deployment_and_its_clients_to_a_successor() {
        let reference = core();
        let mut events = reference.submit(request(RequestId(1))).unwrap();
        let expected = read(&mut events).await;
        assert_eq!(expected.0.len(), TEXT_LEN);
        reference.shutdown();

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("handover.sock");
        let head = core();
        let listener = Listener::Tcp(tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap());
        let address = listener.local_addr().unwrap();

        // The client reads some tokens, then stops; its request stalls
        // mid-generation on its full channel.
        let mut client = head.submit(request(RequestId(1))).unwrap();
        let mut early = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        while early.len() < 10 {
            assert!(Instant::now() < deadline);
            match client.try_recv() {
                Ok(EngineCoreOutput::TextToken { id, .. }) => early.push(id),
                Ok(_) | Err(TryRecvError::Empty) => tokio::task::yield_now().await,
                Err(TryRecvError::Disconnected) => panic!("the stream closed"),
            }
        }

        let handed_over = CancellationToken::new();
        let stopped = CancellationToken::new();
        let offer = Offer::bind(&path, "sim-model").unwrap();
        let offering = offer
            .serve(
                vec![Arc::clone(&head)],
                listener.handover().unwrap(),
                handed_over.clone(),
                stopped.clone(),
                Handle::current(),
            )
            .unwrap();

        let successor = core();
        let predecessor = {
            let path = path.clone();
            tokio::task::spawn_blocking(move || Predecessor::connect(&path, "sim-model", 1))
                .await
                .unwrap()
                .unwrap()
                .expect("the head offers its deployment")
        };
        assert_eq!(predecessor.adoption(), None);
        let (descriptor, handed) = {
            let cores = vec![Arc::clone(&successor)];
            let runtime = Handle::current();
            tokio::task::spawn_blocking(move || predecessor.take_over(cores, &runtime))
                .await
                .unwrap()
                .unwrap()
        };
        // The offer's thread relays the head's clients from now on.
        drop(offering);
        tokio::time::timeout(Duration::from_secs(10), handed_over.cancelled())
            .await
            .expect("the head stops accepting connections");
        let adopted = Listener::adopt(descriptor, &handed).unwrap();
        assert_eq!(adopted.local_addr().unwrap(), address);

        // The migrated request's events reach the head's client.
        let (rest, finish) = read(&mut client).await;
        early.extend(rest);
        assert_eq!((early, finish), expected);

        // A request the head's client submits after the handover runs on
        // the successor and its events come back.
        let mut late = head.submit(request(head.next_request_id())).unwrap();
        let (tokens, finish) = read(&mut late).await;
        assert_eq!(finish, Some(FinishReason::Eos));
        assert_eq!(tokens.len(), TEXT_LEN);

        // The successor offers the deployment in turn.
        Offer::bind(&path, "sim-model").unwrap();
        stopped.cancel();
        head.shutdown();
        successor.shutdown();
    }

    /// A head refuses a successor of another deployment, as two deployments
    /// whose names share a handover socket do, and keeps serving.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_head_refuses_a_successor_of_another_deployment() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("handover.sock");
        let head = core();
        let listener = Listener::Tcp(tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap());
        let stopped = CancellationToken::new();
        Offer::bind(&path, "sim-model")
            .unwrap()
            .serve(
                vec![Arc::clone(&head)],
                listener.handover().unwrap(),
                CancellationToken::new(),
                stopped.clone(),
                Handle::current(),
            )
            .unwrap();

        let refused = tokio::task::spawn_blocking(move || {
            Predecessor::connect(&path, "another-model", 1).map(|predecessor| predecessor.is_some())
        })
        .await
        .unwrap();
        let error = format!("{:#}", refused.unwrap_err());
        assert!(error.contains("deployment another-model"), "{error}");

        let mut events = head.submit(request(RequestId(1))).unwrap();
        assert_eq!(read(&mut events).await.1, Some(FinishReason::Eos));
        stopped.cancel();
        head.shutdown();
    }
}
