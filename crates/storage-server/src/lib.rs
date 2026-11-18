//! The Storage Server: storage whose contents outlive the head and the
//! workers.
//!
//! The server holds descriptors of physical allocations (CUDA VMM
//! allocations through their POSIX file descriptor handles, and host
//! memfds), the tensors clients publish over them, and which client may
//! write each exclusive tensor. It never maps, reads or writes the storage and
//! never creates a CUDA context: device copies run on the workers' streams.
//! The physical storage of a storage lives while the server holds its
//! descriptor or any process maps it.
//!
//! Its Mooncake connector ([`mooncake`], run as a child process
//! [`kv_storage`] supervises) writes the pages of the buffer pools' host
//! regions to the KV storage and reads them back. Its Transfer Server ([`transfer`], another child process,
//! which [`transfers`] supervises) copies bytes between the storage the
//! server holds, for the tasks heads and workers submit; that process, not
//! the server, holds the CUDA contexts the copies need.
//!
//! The server also keeps one buffer pool per deployment ([`buffer_pool`]):
//! the blocks every request of the deployment's engines holds on the
//! device and on the host, which an engine epoch reaches through the queue
//! pairs of its session ([`sessions`]). The pools live on a thread of their
//! own, which sleeps on the queue pairs' doorbells; the main loop opens and closes the
//! sessions and runs the pools' tasks on the children.
//!
//! [`command`] is the `uniserve storage-server` command: [`serve`] runs a
//! server, and [`list`] is the operator's listing of a running one.
//!
//! One thread serves every client with `poll(2)`. Requests get their responses in
//! order on their connection. A request that must wait for a previous
//! writer to be shot down ([`shoot`]), or for the Transfer Server, holds
//! its connection's queue until then, while every other connection is
//! served on.

pub mod buffer_pool;
mod child;
pub mod command;
mod error;
mod kv_storage;
mod memory;
pub mod mooncake;
mod moves;
mod programs;
mod quota;
mod restart;
mod serve;
mod sessions;
mod shoot;
mod timer;
pub mod transfer;
mod transfers;

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io;
use std::net::TcpListener;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;

use rustix::event::{PollFd, PollFlags, poll};

use ::storage::protocol::{
    Access, Allocation, Child, Claim, Claimant, ClientMessage, DeploymentListing, ErrorKind, Event,
    Hello, HostStorage, Item, Key, Listing, Location, Mode, Quota, Refusal, Request, Response,
    Retainer, Role, Scope, ScopeBytes, ServerMessage, StorageRecord, TensorListing, TensorRecord,
    Tensors, Usage, VERSION, Welcome,
};
pub use child::{Children, Program};
pub use error::Error;
use kv_storage::{KvStorage, Task};
pub use restart::Resumed;
pub use serve::{Rendering, ServeOptions, Start, list, serve};
use service::Connection;
use shoot::ShootDowns;

type ClientId = u64;
type StorageId = u64;
/// One copy of a tensor: its key and where its storage lies.
type TensorId = (Key, Location);

/// A process that holds or may hold access to a tensor.
#[derive(Clone)]
struct Holder {
    /// The client while it is connected.
    client: Option<ClientId>,
    pid: i32,
    pidfd: Arc<OwnedFd>,
    scope: Scope,
    generation: u64,
    claim: Claim,
    /// The epoch the process belongs to.
    epoch: u64,
}

impl Holder {
    /// The retention the holder's requests act under.
    fn retainer(&self) -> Retainer {
        Retainer {
            scope: self.scope.clone(),
            epoch: self.epoch,
        }
    }
}

struct Client {
    connection: Connection,
    /// The peer process of a connection on the host; 0 for a connection
    /// from another host.
    pid: i32,
    /// The peer process, held so that its identity cannot be reused; `None`
    /// for a connection from another host, which may only make a head's
    /// transfer requests or a peer's requests.
    pidfd: Option<Arc<OwnedFd>>,
    role: Option<Role>,
    /// The server's child whose channel the connection is, which alone speaks as that child.
    child: Option<::storage::protocol::Child>,
    /// Requests received and not yet responded, in arrival order.
    queue: VecDeque<(u64, Request, Vec<OwnedFd>)>,
    /// Whether a request taken from the queue waits for the Transfer
    /// Server, which holds the queue until it gets its response.
    held: bool,
}

struct StorageEntry {
    record: StorageRecord,
    /// Keeps the physical storage alive.
    _descriptor: OwnedFd,
    /// Tensors over the storage; the storage is released with its last.
    tensors: usize,
    /// For the host copy of an epoch's weights ([`moves`]): the GPU the
    /// storage left, which a restore returns it to.
    moved_from: Option<[u8; 16]>,
    /// Whether a move is filling the storage, which holds no tensor until
    /// the move ends.
    moving: bool,
    /// Whether the storage is a rank's part of a buffer pool's host region
    /// ([`sessions`]), which holds no tensor and lives with its deployment.
    part: bool,
}

struct TensorEntry {
    name: String,
    storage: StorageId,
    offset: u64,
    dtype: String,
    shape: Vec<u64>,
    strides: Vec<u64>,
    access: Access,
    /// The scopes of the epochs that retain the tensor. An epoch retains what
    /// its workers use until they release it or the head retires the
    /// epoch, so an epoch's storage outlives its processes and is released
    /// with the epoch.
    retainers: BTreeSet<Retainer>,
    /// The deployment and epoch whose worker put the tensor: an epoch on
    /// trial keeps what it publishes to itself until its commit.
    publisher: (String, u64),
    /// The process that may write an exclusive tensor.
    writer: Option<Holder>,
    /// Processes that map an exclusive tensor in standby.
    standby: Vec<Holder>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Deployment {
    claim: Claim,
    head: Option<ClientId>,
    /// The epochs claimed and not retired.
    epochs: BTreeSet<u64>,
    /// Per epoch claimed and not retired, the scopes each of its claimants
    /// named last ([`Request::Claim`]): the epoch retains its storage of
    /// every scope one of them names.
    claimed: BTreeMap<u64, BTreeMap<Claimant, BTreeSet<Scope>>>,
    /// Committed epochs whose tensors every replica may read.
    published: BTreeSet<u64>,
    /// The epochs a rollback retired while on trial: what they publish
    /// while their processes leave stays theirs.
    discarded: BTreeSet<u64>,
}

impl Deployment {
    /// Uncommitted and discarded epochs keep their outputs private.
    fn private(&self) -> BTreeSet<u64> {
        let trial = self.epochs.difference(&self.published);
        trial.chain(&self.discarded).copied().collect()
    }
}

/// What handling a request came to.
enum Handling {
    Response(Response, Vec<Attachment>),
    /// The request waits for these processes to be shot down.
    Wait(Vec<Holder>),
    /// A peer's exposure, checked and resolved, responded once the Transfer
    /// Server registered its regions: the allocations it registers, each
    /// once; for each region the index of its allocation among them and the
    /// offset of its first byte there; and each allocation's fabric handle
    /// and length, when its creator recorded one.
    Exposing {
        allocations: Vec<Allocation>,
        regions: Vec<(usize, u64)>,
        fabric: Vec<Option<(::storage::protocol::FabricHandle, u64)>>,
    },
    /// A response whose descriptors travel with it and are not the
    /// server's to keep: the queue pairs of a session.
    Descriptors(Response, Vec<OwnedFd>),
    /// A head's weight move, checked, responded once its copies ended: its
    /// deployment, its direction and its storages.
    Moving {
        deployment: String,
        to_host: bool,
        items: Vec<moves::Item>,
    },
}

/// A descriptor that travels with a reply or an event.
#[derive(Clone, Copy)]
enum Attachment {
    Storage(StorageId),
}

/// The server's state and its listening socket.
pub struct Server {
    listener: UnixListener,
    /// The TCP listener other hosts reach the server through (heads, peers
    /// and other hosts' Transfer Servers), with the address it is bound to.
    tcp: Option<(TcpListener, std::net::SocketAddr)>,
    clients: BTreeMap<ClientId, Client>,
    next_client: ClientId,
    storages: HashMap<StorageId, StorageEntry>,
    next_storage: StorageId,
    /// Every retained tensor by its key and where its storage lies: a key
    /// names what the bytes are, its storage where they lie, and a key
    /// has at most one copy at each location.
    tensors: HashMap<TensorId, TensorEntry>,
    /// Where each key's copies lie.
    copies: HashMap<Key, Vec<Location>>,
    deployments: HashMap<String, Deployment>,
    shootdowns: ShootDowns,
    /// What the server serves with, which a restart in place's successor serves with too.
    settings: ::storage::Settings,
    /// The GPUs' quotas: what the processes reserve and the quotas
    /// the epochs state ([`quota`]).
    quotas: quota::Quotas,
    kv_storage: KvStorage,
    transfers: transfers::Transfers,
    /// The epochs' weights moving between GPUs and host storage.
    moves: moves::Moves,
    /// Events to send once the request being handled gets its response.
    events: Vec<(ClientId, Event, Vec<Attachment>)>,
    /// The executable the server runs its children from; `None` for a server that runs
    /// none.
    program: Option<Program>,
    /// A restart in place under way, which passes the server over once nothing is
    /// in flight.
    restart: Option<restart::Pending>,
    /// The sessions of engine epochs with the deployments' buffer pools,
    /// and the pools' thread.
    sessions: sessions::Sessions,
    /// The readiness record the server published, which [`Request::Status`] answers; `None`
    /// until the server serves.
    readiness: Option<::storage::Readiness>,
    /// The operator whose [`Request::Stop`] stops the server: the loop ends once its response
    /// is written.
    stopping: Option<ClientId>,
    /// The deployments' KV programs ([`programs`]).
    programs: programs::Programs,
}

impl Server {
    /// Serve on `listener`, which the caller bound with mode 0600, with `settings`, running
    /// `children` when given.
    ///
    /// # Errors
    ///
    /// Fails when the listener cannot be made non-blocking or the buffer pools' thread cannot
    /// start.
    pub fn new(
        listener: UnixListener,
        settings: ::storage::Settings,
        children: Option<Children>,
    ) -> io::Result<Self> {
        listener.set_nonblocking(true)?;
        let (program, connector, transfer) = child::supervisors(children, &settings.server)?;
        let mut server = Self {
            listener,
            tcp: None,
            clients: BTreeMap::new(),
            next_client: 1,
            storages: HashMap::new(),
            next_storage: 1,
            tensors: HashMap::new(),
            copies: HashMap::new(),
            deployments: HashMap::new(),
            shootdowns: ShootDowns::new(settings.server.shoot_down())?,
            quotas: quota::Quotas::default(),
            kv_storage: KvStorage::new(connector),
            transfers: transfers::Transfers::new(transfer),
            program,
            moves: moves::Moves::default(),
            events: Vec::new(),
            restart: None,
            sessions: sessions::Sessions::new(&settings.server)?,
            readiness: None,
            stopping: None,
            programs: programs::Programs::default(),
            settings,
        };
        let trace = server.settings.server.buffer_pool_trace.clone();
        server.watch_pools(trace);
        Ok(server)
    }

    /// The readiness record the server published, which a [`Request::Status`] answers from now
    /// on, and which a restart in place's successor is given the settings of.
    pub fn set_readiness(&mut self, readiness: ::storage::Readiness) {
        self.readiness = Some(readiness);
    }

    /// The TCP endpoint other hosts reach the server at, as it is bound; `None` when it accepts
    /// local connections only.
    pub fn tcp_endpoint(&self) -> Option<service::Endpoint> {
        self.tcp
            .as_ref()
            .map(|(_, address)| service::Endpoint::tcp(*address))
    }

    /// From now on, records the trace of every buffer pool's inputs in the
    /// directory `trace` (`specs/buffer-pool.md` §18;
    /// `buffer_pool::replay` replays it and checks the pool's tables after every input).
    pub fn watch_pools(&mut self, trace: Option<std::path::PathBuf>) {
        self.sessions.watch_pools(trace);
    }

    /// `configure`: see [`Request::Configure`]. Every part of the server that reads a setting
    /// when it acts is given the new value: the admission of host storage reads the settings
    /// themselves, the buffer pools and their helpers take the deadlines, the shoot-downs their
    /// deadline, the children's supervisors their waits, and the Transfer Server, when it runs,
    /// its chunk time and progress period. The published readiness record states the settings
    /// from then on.
    fn configure(
        &mut self,
        settings: &::storage::ServerSettings,
    ) -> Result<Handling, (ErrorKind, String)> {
        if self.settings.server.restarts(settings) {
            return Err(invalid(
                "the settings differ in one a running server cannot take: a restart in place \
                 serves with them",
            ));
        }
        self.sessions.configure(settings);
        self.shootdowns.set_deadline(settings.shoot_down());
        self.kv_storage.process.configure(settings);
        self.transfers.process.configure(settings);
        if let Some(transfer_server) = self.transfers.client {
            self.events.push((
                transfer_server,
                Event::Pace {
                    chunk_us: settings.transfer_chunk_us,
                    progress_ms: settings.transfer_progress_ms,
                },
                Vec::new(),
            ));
        }
        self.settings.server = settings.clone();
        tracing::info!(?settings, "serving with new settings");

        if let Some(record) = &mut self.readiness {
            record.settings = self.settings.clone();
            if let Some(service::Endpoint::Unix(socket)) = &record.endpoints.storage
                && let Err(error) = service::host::publish(socket, record)
            {
                tracing::warn!(%error, "the readiness record states the earlier settings");
            }
        }
        Ok(Handling::Response(Response::Configured, Vec::new()))
    }

    /// Also accept connections from other hosts on `listener`, which carry no descriptors.
    /// Returns the endpoint it is bound at.
    ///
    /// # Errors
    ///
    /// Fails when the listener's address cannot be read or it cannot be made non-blocking.
    pub fn listen_tcp(&mut self, listener: TcpListener) -> io::Result<service::Endpoint> {
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        self.tcp = Some((listener, address));
        Ok(service::Endpoint::tcp(address))
    }

    /// Serve until `stop` reads true. `stop` is checked after every wake;
    /// `wake` is a descriptor whose readability wakes the loop (a signal
    /// pipe), drained when it fires. Returns once `stop` holds, or with the
    /// restart under way once nothing is in flight and its successor speaks
    /// every version in use, which the caller then completes
    /// ([`Server::handoff`]); a restart whose successor does not is given up,
    /// and the server goes on.
    pub fn run(
        &mut self,
        wake: BorrowedFd<'_>,
        stop: &dyn Fn() -> bool,
    ) -> io::Result<Option<restart::Pending>> {
        while !stop() && self.stopping.is_none() {
            self.turn(wake)?;
            if self.ready_to_hand_over()? {
                return Ok(self.restart.take());
            }
        }
        // The operator that stopped the server reads its response before the server's end
        // closes its connection.
        if let Some(operator) = self.stopping
            && let Some(client) = self.clients.get_mut(&operator)
            && let Err(error) = client.connection.flush()
        {
            tracing::warn!(%error, "the stopping operator did not take its response");
        }
        Ok(None)
    }

    /// One wait and the work it makes ready.
    fn turn(&mut self, wake: BorrowedFd<'_>) -> io::Result<()> {
        let ids: Vec<ClientId> = self.clients.keys().copied().collect();
        let pidfds = self.shootdowns.pending_pidfds();
        // The buffer pool's thread's notices.
        let notices = self.sessions.thread.wake();
        let mut fds = Vec::with_capacity(4 + ids.len() + pidfds.len());
        fds.push(PollFd::new(&self.listener, PollFlags::IN));
        fds.push(PollFd::new(&wake, PollFlags::IN));
        fds.push(PollFd::new(&notices, PollFlags::IN));
        let tcp = self.tcp.as_ref().map(|(listener, _)| listener);
        if let Some(listener) = tcp {
            fds.push(PollFd::new(listener, PollFlags::IN));
        }
        let first = fds.len();
        for id in &ids {
            let client = &self.clients[id];
            let mut flags = PollFlags::IN;
            if client.connection.wants_write() {
                flags |= PollFlags::OUT;
            }
            fds.push(PollFd::new(&client.connection, flags));
        }
        for pidfd in &pidfds {
            fds.push(PollFd::new(pidfd.as_ref(), PollFlags::IN));
        }
        // A restart's handoff socket and its successor's pidfd, whose answer or end the loop reads.
        let restart = self.restart_descriptors().map(|descriptors| {
            let at = fds.len();
            fds.extend(descriptors.map(|fd| PollFd::from_borrowed_fd(fd, PollFlags::IN)));
            at
        });
        for process in [&self.kv_storage.process, &self.transfers.process] {
            if let Some(pidfd) = process.pidfd() {
                fds.push(PollFd::new(pidfd, PollFlags::IN));
            }
            // A child's restart, due at the time its timer fires.
            if let Some(timer) = process.restart_timer() {
                fds.push(PollFd::from_borrowed_fd(timer, PollFlags::IN));
            }
        }
        // A shoot-down's next question to NVML.
        if let Some(timer) = self.shootdowns.timer() {
            fds.push(PollFd::from_borrowed_fd(timer, PollFlags::IN));
        }
        // The KV programs' periods.
        for timer in self.programs.timers() {
            fds.push(PollFd::from_borrowed_fd(timer, PollFlags::IN));
        }
        // Every wait is a descriptor's: the loop sleeps until one is ready.
        match poll(&mut fds, None) {
            Ok(_) | Err(rustix::io::Errno::INTR) => {}
            Err(error) => return Err(error.into()),
        }
        let accept = !fds[0].revents().is_empty();
        if !fds[1].revents().is_empty() {
            drain(wake);
        }
        let accept_tcp = tcp.is_some() && !fds[3].revents().is_empty();
        let restart_ready =
            restart.is_some_and(|at| fds[at..at + 2].iter().any(|fd| !fd.revents().is_empty()));
        let ready: Vec<ClientId> = ids
            .iter()
            .zip(&fds[first..first + ids.len()])
            .filter(|(_, fd)| !fd.revents().is_empty())
            .map(|(id, _)| *id)
            .collect();
        drop(fds);

        if accept {
            self.accept();
        }
        if accept_tcp {
            self.accept_tcp();
        }
        for id in ready {
            self.read(id);
        }
        if restart_ready {
            self.read_restart();
        }
        if self.shootdowns.advance() {
            self.settle_shootdowns();
            self.settle_sessions();
        }
        self.kv_storage.process.reap();
        self.transfers.process.reap();
        self.start_children();
        self.step_programs();
        self.take_notices();
        self.settle_lost();
        self.send_events();
        self.serve_queues();
        self.flush_and_reap();
        Ok(())
    }

    /// Take every pending connection.
    fn accept(&mut self) {
        loop {
            let stream = match self.listener.accept() {
                Ok((stream, _)) => stream,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return,
                Err(error) => {
                    tracing::warn!(%error, "accepting a storage client failed");
                    return;
                }
            };
            match self.admit(stream) {
                Ok(client) => {
                    let id = self.next_client;
                    self.next_client += 1;
                    self.clients.insert(id, client);
                }
                Err(error) => tracing::warn!(%error, "a storage client could not be admitted"),
            }
        }
    }

    /// Take every pending connection from another host.
    fn accept_tcp(&mut self) {
        loop {
            let Some((listener, _)) = &self.tcp else {
                return;
            };
            let stream = match listener.accept() {
                Ok((stream, _)) => stream,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return,
                Err(error) => {
                    tracing::warn!(%error, "accepting a remote storage client failed");
                    return;
                }
            };
            let connection = match Connection::new(stream) {
                Ok(connection) => connection,
                Err(error) => {
                    tracing::warn!(%error, "a remote storage client could not be admitted");
                    continue;
                }
            };
            let id = self.next_client;
            self.next_client += 1;
            self.clients.insert(
                id,
                Client {
                    connection,
                    pid: 0,
                    pidfd: None,
                    role: None,
                    child: None,
                    queue: VecDeque::new(),
                    held: false,
                },
            );
        }
    }

    /// Starts each child whose start is due, and admits the server's end of its channel as that
    /// child's client: the connection only it speaks as its child on.
    fn start_children(&mut self) {
        let Some(program) = &self.program else {
            return;
        };
        let started = [&mut self.kv_storage.process, &mut self.transfers.process]
            .into_iter()
            .filter_map(|process| Some((process.child()?, process.start_due(program)?)))
            .collect::<Vec<_>>();
        for (child, started) in started {
            let id = self.next_client;
            self.next_client += 1;
            self.clients.insert(
                id,
                Client {
                    connection: started.connection,
                    pid: started.pid,
                    pidfd: Some(started.pidfd),
                    role: None,
                    child: Some(child),
                    queue: VecDeque::new(),
                    held: false,
                },
            );
        }
    }

    /// A connection's peer process, held by a pidfd so that its identity
    /// cannot be reused while the server refers to it.
    fn admit(&self, stream: UnixStream) -> io::Result<Client> {
        let credentials = rustix::net::sockopt::socket_peercred(&stream)?;
        let pid = credentials.pid;
        let pidfd = rustix::process::pidfd_open(pid, rustix::process::PidfdFlags::empty())?;
        Ok(Client {
            connection: Connection::new(stream)?,
            pid: pid.as_raw_nonzero().get(),
            pidfd: Some(Arc::new(pidfd)),
            role: None,
            child: None,
            queue: VecDeque::new(),
            held: false,
        })
    }

    /// Read what a client sent and queue its requests. A client whose
    /// connection closed has its received requests responded, then is
    /// disconnected.
    fn read(&mut self, id: ClientId) {
        let mut failed = false;
        let mut frames = Vec::new();
        if let Some(client) = self.clients.get_mut(&id) {
            loop {
                match client.connection.try_recv_with::<ClientMessage>() {
                    Ok(Some(frame)) => frames.push(frame),
                    Ok(None) => break,
                    // The client closed its end: the requests it sent before get their
                    // responses below.
                    Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(error) if error.kind() == io::ErrorKind::InvalidData => {
                        tracing::warn!(pid = client.pid, %error, "a storage client sent an unreadable message");
                        failed = true;
                        break;
                    }
                    Err(error) => {
                        tracing::warn!(pid = client.pid, %error, "a storage client failed");
                        failed = true;
                        break;
                    }
                }
            }
        }
        for (message, descriptors) in frames {
            match message {
                ClientMessage::Hello(hello) => {
                    let welcome = self.welcome(id, hello);
                    if let Some(client) = self.clients.get_mut(&id) {
                        let _ = client.connection.send(&ServerMessage::Welcome(welcome));
                    }
                    if self.kv_storage.client == Some(id) {
                        self.offer_parts();
                        self.send_events();
                    }
                }
                ClientMessage::Request {
                    id: request,
                    request: body,
                } => {
                    if let Some(client) = self.clients.get_mut(&id) {
                        client.queue.push_back((request, body, descriptors));
                    }
                }
            }
        }
        let closed = self
            .clients
            .get(&id)
            .is_some_and(|client| client.connection.is_closed());
        if closed && !failed {
            // Requests already received get their responses first: a client may
            // send a final release and close at once.
            self.serve_queue(id);
        }
        if closed || failed {
            self.disconnect(id);
        }
    }

    /// Admit a client's role, or refuse it: a client of another version of
    /// the protocol than this build's is refused before anything else.
    fn welcome(&mut self, id: ClientId, hello: Hello) -> Welcome {
        let refused = |refusal| Welcome::Refused { refusal };
        if hello.version != VERSION {
            return refused(Refusal::Version {
                offered: hello.version,
                spoken: VERSION,
            });
        }
        let Some(client) = self.clients.get_mut(&id) else {
            return refused(Refusal::Repeated);
        };
        if client.role.is_some() {
            return refused(Refusal::Repeated);
        }
        if client.pidfd.is_none() && !matches!(hello.role, Role::Head { .. } | Role::Peer) {
            return refused(Refusal::Remote);
        }
        // A child's role is spoken on that child's channel alone, which the server made, and a
        // channel speaks its child's role alone.
        let stated = match hello.role {
            Role::Connector => Some(Child::Connector),
            Role::Transfer => Some(Child::TransferServer),
            _ => None,
        };
        if let (Some(child), _) | (None, Some(child)) = (stated, client.child)
            && stated != client.child
        {
            return refused(Refusal::Child { child });
        }
        let pid = client.pid;
        client.role = Some(hello.role);
        match stated {
            Some(Child::Connector) => {
                self.kv_storage.client = Some(id);
                self.kv_storage.process.connected();
                tracing::info!(pid, "the Mooncake connector connected");
            }
            Some(Child::TransferServer) => {
                self.transfers.client = Some(id);
                self.transfers.process.connected();
                tracing::info!(pid, "the Transfer Server connected");
            }
            None => {}
        }
        Welcome::Accepted
    }

    /// Respond to every client's queued requests that can be responded now.
    fn serve_queues(&mut self) {
        let ids: Vec<ClientId> = self.clients.keys().copied().collect();
        for id in ids {
            self.serve_queue(id);
        }
    }

    /// Respond to one client's queued requests in order, until one waits.
    fn serve_queue(&mut self, id: ClientId) {
        loop {
            let Some(client) = self.clients.get_mut(&id) else {
                return;
            };
            if client.held {
                return;
            }
            let Some((request_id, request, descriptors)) = client.queue.pop_front() else {
                return;
            };
            let started = std::time::Instant::now();
            let result = self.handle(id, &request, descriptors);
            let handled = started.elapsed();
            match result {
                Ok(Handling::Response(reply, storages)) => {
                    self.respond(id, request_id, reply, &storages);
                    self.send_events();
                    // The server is single-threaded: the time a request holds it is the time
                    // every other client's request waits.
                    let served = started.elapsed();
                    tracing::debug!(
                        request = <&str>::from(&request),
                        handled_us = u64::try_from(handled.as_micros()).unwrap_or(u64::MAX),
                        served_us = u64::try_from(served.as_micros()).unwrap_or(u64::MAX),
                        "a request held the server"
                    );
                }
                Ok(Handling::Descriptors(reply, descriptors)) => {
                    let borrowed: Vec<BorrowedFd<'_>> =
                        descriptors.iter().map(AsFd::as_fd).collect();
                    self.respond_with(id, request_id, reply, &borrowed);
                    self.send_events();
                }
                Ok(Handling::Exposing {
                    allocations,
                    regions,
                    fabric,
                }) => {
                    // Holds the connection until the Transfer Server
                    // reports, unless it is refused at once.
                    self.begin_exposure(id, request_id, allocations, regions, fabric);
                    self.send_events();
                }
                Ok(Handling::Moving {
                    deployment,
                    to_host,
                    items,
                }) => {
                    // Holds the connection until the move's copies end.
                    self.begin_move(id, request_id, deployment, to_host, items);
                    self.send_events();
                }
                Ok(Handling::Wait(victims)) => {
                    for victim in victims {
                        let gpus = self.gpus_held_by(&victim.pidfd);
                        self.shootdowns
                            .start(victim.pid, Arc::clone(&victim.pidfd), gpus);
                    }
                    // The request is handled again, from the start, once the
                    // shoot-downs end; it changed nothing yet.
                    if let Some(client) = self.clients.get_mut(&id) {
                        client.queue.push_front((request_id, request, Vec::new()));
                    }
                    return;
                }
                Err((kind, message)) => {
                    self.respond(id, request_id, Response::Error { kind, message }, &[]);
                }
            }
        }
    }

    /// Send a reply with the descriptors of `storages`, in order.
    fn respond(&mut self, id: ClientId, request: u64, reply: Response, attachments: &[Attachment]) {
        let descriptors: Vec<BorrowedFd<'_>> = attachments
            .iter()
            .filter_map(|attachment| match attachment {
                Attachment::Storage(storage) => self
                    .storages
                    .get(storage)
                    .map(|entry| entry._descriptor.as_fd()),
            })
            .collect();
        send_reply(&mut self.clients, id, request, reply, &descriptors);
    }

    /// Send a reply with `descriptors`, in order.
    fn respond_with(
        &mut self,
        id: ClientId,
        request: u64,
        reply: Response,
        descriptors: &[BorrowedFd<'_>],
    ) {
        send_reply(&mut self.clients, id, request, reply, descriptors);
    }

    /// Handle one request. A request that needs previous writers shot down
    /// changes nothing and returns [`Handling::Wait`].
    fn handle(
        &mut self,
        id: ClientId,
        request: &Request,
        descriptors: Vec<OwnedFd>,
    ) -> Result<Handling, (ErrorKind, String)> {
        let client = &self.clients[&id];
        let role = client
            .role
            .clone()
            .ok_or_else(|| invalid("a request precedes the client's hello"))?;
        match (request, role) {
            (Request::List, _) => Ok(Handling::Response(
                Response::Listing(self.listing()),
                Vec::new(),
            )),
            (Request::Status, _) => match &self.readiness {
                Some(readiness) => Ok(Handling::Response(
                    Response::Status(Box::new(readiness.clone())),
                    Vec::new(),
                )),
                None => Err(resource(io::Error::other("the server does not serve yet"))),
            },
            (Request::Stop, Role::Operator) => {
                tracing::info!("stopping at an operator's request");
                self.stopping = Some(id);
                Ok(Handling::Response(Response::Stopping, Vec::new()))
            }
            (Request::Claim { .. }, Role::Head { .. }) if client.pidfd.is_none() => Err(invalid(
                "a deployment is claimed at its host's server, not from another host",
            )),
            (
                Request::Submit { .. }
                | Request::OpenSession { .. }
                | Request::CloseSession { .. }
                | Request::Cancel { .. }
                | Request::Throttle { .. },
                role,
            ) => self.transfer_request(id, request, &role),
            (
                Request::Measured { .. } | Request::Copied { .. } | Request::Registered { .. },
                Role::Transfer,
            ) => self.transfer_report(request),
            (
                Request::Expose {
                    deployment,
                    regions,
                },
                Role::Peer,
            ) => self.expose(deployment, regions),
            (Request::Unexpose { exposure }, Role::Peer) => Ok(self.unexpose(id, *exposure)),
            (Request::Finished { task, results }, Role::Connector) => {
                self.finished(*task, results);
                Ok(Handling::Response(Response::Finished, Vec::new()))
            }
            (
                Request::Open(open),
                Role::Head {
                    deployment,
                    epoch,
                    generation,
                },
            ) => self.open_pool(id, (&deployment, epoch, generation), open),
            (
                Request::Close,
                Role::Head {
                    deployment,
                    epoch,
                    generation,
                },
            ) => self.close_pool((&deployment, epoch, generation)),
            (
                Request::PlaceRemote {
                    region,
                    memory,
                    part,
                    endpoint,
                    runs,
                },
                Role::Head {
                    deployment, epoch, ..
                },
            ) => self.place_remote(
                id,
                (&deployment, epoch),
                (region, memory, *part),
                endpoint,
                runs,
            ),
            (
                Request::Grant {
                    region,
                    segment,
                    first,
                    units,
                },
                Role::Head {
                    deployment, epoch, ..
                },
            ) => self.grant(id, (&deployment, epoch), region, *segment, (*first, *units)),
            (
                Request::Withdraw { region, segment },
                Role::Head {
                    deployment, epoch, ..
                },
            ) => self.withdraw(id, (&deployment, epoch), region, *segment),
            (Request::PeerHold { deployment, keys }, Role::Peer) => {
                self.peer_hold(id, deployment, keys)
            }
            (Request::PeerRelease { deployment }, Role::Peer) => self.peer_release(id, deployment),
            (
                Request::Pages {
                    region,
                    part,
                    pages,
                    upload,
                },
                Role::Head { deployment, .. },
            ) => self.pages_for(id, &deployment, (region, *part), pages, *upload),
            (
                Request::Restart {
                    arguments,
                    settings,
                },
                Role::Operator,
            ) => self
                .begin_restart(
                    descriptors.into_iter().next(),
                    arguments,
                    settings.as_deref().cloned(),
                )
                .map(|pid| Handling::Response(Response::Restarting { pid }, Vec::new()))
                .map_err(resource),
            (Request::Configure { settings }, Role::Operator) => self.configure(settings),
            (
                Request::Claim {
                    claimant,
                    scopes,
                    epoch,
                    retire,
                },
                Role::Head { deployment, .. },
            ) => self.claim(id, &deployment, (*claimant, scopes), *epoch, retire),
            (Request::Publish { epochs }, Role::Head { deployment, .. }) => {
                self.publish_epochs(id, &deployment, epochs)
            }
            (Request::KeepWeights { epochs, cache_keys }, Role::Head { deployment, .. }) => {
                self.keep_weights(id, &deployment, epochs, cache_keys)
            }
            (Request::Offload { epoch }, Role::Head { deployment, .. }) => {
                self.check_move(id, &deployment, *epoch, true)
            }
            (Request::Restore { epoch }, Role::Head { deployment, .. }) => {
                self.check_move(id, &deployment, *epoch, false)
            }
            (
                Request::Allocated {
                    request,
                    bytes,
                    failure,
                },
                Role::Transfer,
            ) => {
                let descriptor = descriptors.into_iter().next();
                self.allocated(*request, *bytes, failure.as_deref(), descriptor);
                Ok(Handling::Response(Response::Noted, Vec::new()))
            }
            (Request::Usage { gpus }, role) => {
                let deployment = match &role {
                    Role::Head { deployment, .. } => Some(deployment.clone()),
                    Role::Worker { scope, .. } => Some(scope.deployment.clone()),
                    Role::Transfer | Role::Peer | Role::Operator | Role::Connector => None,
                };
                let quotas = self.quotas_of(gpus, deployment.as_deref());
                let tasks = deployment
                    .as_deref()
                    .map(|deployment| self.transfers.totals(deployment))
                    .unwrap_or_default();
                Ok(Handling::Response(
                    Response::Usage(Usage {
                        quotas,
                        host: self.host_storage().map_err(resource)?,
                        tasks,
                    }),
                    Vec::new(),
                ))
            }
            (
                Request::Quota {
                    gpu,
                    total,
                    limit,
                    reservations,
                },
                Role::Worker { scope, epoch, .. },
            ) => self.reserve(
                id,
                quota::Reserver::Worker { scope, epoch },
                *gpu,
                *total,
                *limit,
                reservations,
            ),
            (
                Request::Quota {
                    gpu,
                    total,
                    reservations,
                    ..
                },
                Role::Transfer,
            ) => self.reserve(
                id,
                quota::Reserver::Transfer,
                *gpu,
                *total,
                None,
                reservations,
            ),
            (Request::Program { program }, Role::Head { deployment, .. }) => {
                let start = self
                    .programs
                    .set(&deployment, program.clone())
                    .map_err(|reason| invalid(&reason))?;
                Ok(Handling::Response(
                    Response::Programmed { start },
                    Vec::new(),
                ))
            }
            (Request::Handoff { handoff }, Role::Head { deployment, .. })
            | (
                Request::Handoff { handoff },
                Role::Worker {
                    scope: Scope { deployment, .. },
                    ..
                },
            ) => {
                let path = self
                    .programs
                    .handoff(&deployment, handoff)
                    .map_err(|reason| invalid(&reason))?;
                Ok(Handling::Response(Response::Handoff { path }, Vec::new()))
            }
            (Request::ProgramRecords, Role::Head { deployment, .. }) => {
                let records = self.programs.drain(&deployment);
                Ok(Handling::Response(
                    Response::ProgramRecords { records },
                    Vec::new(),
                ))
            }
            // The rings are the process's, whichever deployment's head reads them.
            (Request::Spans, Role::Head { .. }) => {
                let spans = service::span::spans(&service::span::records(), |_| None);
                Ok(Handling::Response(Response::Spans { spans }, Vec::new()))
            }
            (
                request,
                Role::Worker {
                    scope,
                    generation,
                    claim,
                    epoch,
                },
            ) => {
                let pidfd = client
                    .pidfd
                    .clone()
                    .ok_or_else(|| invalid("a worker connects on its host"))?;
                let holder = Holder {
                    client: Some(id),
                    pid: client.pid,
                    pidfd,
                    scope,
                    generation,
                    claim,
                    epoch,
                };
                match request {
                    Request::Retain { keys, at } => self.retain(&holder, keys, at),
                    Request::Get { keys, mode, at } => self.get(&holder, keys, *mode, at),
                    Request::Put { tensors, storages } => {
                        self.put(&holder, tensors, storages, descriptors)
                    }
                    Request::AllocateHost { key, name, bytes } => {
                        self.allocate_host(&holder, *key, name, *bytes)
                    }
                    Request::Promote { keys } => self.promote(&holder, keys),
                    Request::Release { keys } => self.release(&holder, keys),
                    Request::Place {
                        region,
                        memory,
                        part,
                        runs,
                    } => self.place(&holder, (region, memory, *part), runs),
                    Request::HostPart {
                        region,
                        part,
                        bytes,
                        numa_node,
                    } => self.host_part(&holder, (region, *part), *bytes, *numa_node),
                    _ => Err(invalid(&format!(
                        "{} is not a worker request",
                        <&str>::from(request)
                    ))),
                }
            }
            (request, role) => Err(invalid(&format!(
                "{} is not a request of {role:?}",
                <&str>::from(request)
            ))),
        }
    }

    /// `claim`: see [`Request::Claim`], by `claimant` for `scopes`.
    fn claim(
        &mut self,
        id: ClientId,
        deployment: &str,
        (claimant, scopes): (Claimant, &[Scope]),
        epoch: u64,
        retire: &[u64],
    ) -> Result<Handling, (ErrorKind, String)> {
        if let Some(entry) = self.deployments.get(deployment)
            && let Some(head) = entry.head
            && head != id
            && self.clients.contains_key(&head)
        {
            return Err((
                ErrorKind::Claimed,
                format!("deployment {deployment} has a connected head"),
            ));
        }
        // The head that holds the claim keeps it for every epoch it claims:
        // its workers, of any epoch, stay current.
        let held = self
            .deployments
            .get(deployment)
            .filter(|entry| entry.head == Some(id))
            .map(|entry| entry.claim);
        if held.is_none() {
            // Every process that holds access to the deployment's tensors
            // under a previous claim, and every worker of a previous claim
            // still connected, is out before the new claim stands.
            let current = self.deployments.get(deployment).map(|entry| entry.claim);
            let mut victims: Vec<Holder> = self
                .tensors
                .values()
                .flat_map(|tensor| tensor.writer.iter().chain(&tensor.standby))
                .filter(|holder| holder.scope.deployment == deployment)
                .filter(|holder| Some(holder.claim) != current || current.is_none())
                .cloned()
                .collect();
            victims.extend(
                self.clients
                    .iter()
                    .filter(|(other, _)| **other != id)
                    .filter_map(|(other, client)| match (&client.role, &client.pidfd) {
                        (
                            Some(Role::Worker {
                                scope,
                                generation,
                                claim,
                                epoch,
                            }),
                            Some(pidfd),
                        ) if scope.deployment == deployment => Some(Holder {
                            client: Some(*other),
                            pid: client.pid,
                            pidfd: Arc::clone(pidfd),
                            scope: scope.clone(),
                            generation: *generation,
                            claim: *claim,
                            epoch: *epoch,
                        }),
                        _ => None,
                    }),
            );
            victims.retain(|victim| !self.shootdowns.finished(&victim.pidfd));
            if let Some(failed) = victims
                .iter()
                .find(|victim| self.shootdowns.failed(&victim.pidfd))
            {
                return Err(shoot_down_failed(failed.pid));
            }
            dedup(&mut victims);
            if !victims.is_empty() {
                return Ok(Handling::Wait(victims));
            }
        }

        let claim = match held {
            Some(claim) => claim,
            None => Claim(random_token().map_err(resource)?),
        };
        // The head that holds the claim adds and retires epochs. A new head
        // starts from its own epoch: the epochs a previous head kept on
        // trial are discarded with that head, never committed, and the
        // epochs discarded before are forgotten, so that the new head's
        // epochs may take their numbers.
        let previous = self.deployments.get(deployment);
        let trial: BTreeSet<u64> = previous
            .map(|entry| entry.epochs.difference(&entry.published).copied().collect())
            .unwrap_or_default();
        let (mut discarded, mut epochs, mut published, mut claimed) = previous
            .map(|entry| {
                (
                    entry.discarded.clone(),
                    entry.epochs.clone(),
                    entry.published.clone(),
                    entry.claimed.clone(),
                )
            })
            .unwrap_or_default();
        let mut retired: BTreeSet<u64> = retire.iter().copied().collect();
        if held.is_none() {
            retired.extend(trial.iter().filter(|&&old| old != epoch));
            epochs.clear();
            published.clear();
            claimed.clear();
        }
        epochs.insert(epoch);
        epochs.retain(|epoch| !retired.contains(epoch));
        published.retain(|epoch| epochs.contains(epoch));
        claimed.retain(|epoch, _| epochs.contains(epoch));
        // The claimant's scopes replace those it named before; the epoch
        // keeps every scope one of its claimants names.
        claimed
            .entry(epoch)
            .or_default()
            .insert(claimant, scopes.iter().cloned().collect());
        let named: BTreeSet<Scope> = claimed
            .get(&epoch)
            .into_iter()
            .flat_map(BTreeMap::values)
            .flatten()
            .cloned()
            .collect();
        // Retiring an uncommitted epoch rolls its update back.
        for &old in &trial {
            if retired.contains(&old) {
                self.update_rolled_back(deployment, old);
                discarded.insert(old);
            }
        }
        if held.is_none() {
            discarded.clear();
            // Every process of the previous claim is out (above): nothing it
            // reserved is kept for a relaunch the new head will not make.
            self.quotas.claim_anew(deployment);
        }
        self.deployments.insert(
            deployment.to_owned(),
            Deployment {
                claim,
                head: Some(id),
                epochs,
                claimed,
                published,
                discarded,
            },
        );
        // The epoch's retention of the scopes no claimant of it names ends,
        // and every retired epoch's retention ends.
        let released: Vec<TensorId> = self
            .tensors
            .iter_mut()
            .filter_map(|(id, tensor)| {
                tensor.retainers.retain(|retainer| {
                    retainer.scope.deployment != deployment
                        || !(retired.contains(&retainer.epoch)
                            || (retainer.epoch == epoch && !named.contains(&retainer.scope)))
                });
                tensor.retainers.is_empty().then_some(*id)
            })
            .collect();
        let count = released.len();
        for id in released {
            self.remove_tensor(&id);
        }
        // A retired epoch's quota ends with its storage: the quota then
        // reserves what its successor alone holds, under the successor's.
        self.quotas.retire(deployment, &retired);
        tracing::info!(
            deployment,
            epoch,
            ?claimant,
            ?retire,
            released = count,
            "a head claimed the deployment"
        );
        Ok(Handling::Response(Response::Claimed { claim }, Vec::new()))
    }

    /// Publish a committed set atomically before any reader can observe it.
    fn publish_epochs(
        &mut self,
        id: ClientId,
        deployment: &str,
        epochs: &[u64],
    ) -> Result<Handling, (ErrorKind, String)> {
        let entry = self
            .deployments
            .get_mut(deployment)
            .filter(|entry| entry.head == Some(id))
            .ok_or_else(|| invalid("only the claiming head publishes committed epochs"))?;
        if epochs.iter().any(|epoch| !entry.epochs.contains(epoch)) {
            return Err(invalid("only claimed epochs can be published"));
        }
        let newly: Vec<_> = epochs
            .iter()
            .copied()
            .filter(|epoch| !entry.published.contains(epoch))
            .collect();
        entry.published.extend(epochs);
        for epoch in newly {
            self.update_committed(deployment, epoch);
        }
        Ok(Handling::Response(Response::Published, Vec::new()))
    }

    /// Refuse `bytes` of new host storage that the host cannot hold: its
    /// available storage less the server's reserve and what retained storage
    /// have yet to allocate.
    fn admit_host(&self, bytes: u64) -> Result<(), (ErrorKind, String)> {
        let host = self.host_storage().map_err(resource)?;
        let room = host.room();
        if bytes > room {
            return Err((
                ErrorKind::Resource,
                format!(
                    "{bytes} bytes of host storage exceed the {room} bytes the host can hold: \
                    {} bytes available, less {} bytes retained storage has yet to allocate and \
                     the server's reserve of {} bytes",
                    host.available, host.unallocated, host.reserve
                ),
            ));
        }
        Ok(())
    }

    /// Refuses `bytes` of new device storage on `gpu` that would take
    /// `deployment`'s quota of it past its quota, naming `what`.
    fn admit_device(
        &self,
        deployment: &str,
        gpu: [u8; 16],
        bytes: u64,
        what: &str,
    ) -> Result<(), (ErrorKind, String)> {
        for quota in self.quotas_of(&[gpu], Some(deployment)) {
            let (Some(limit), usage) = (quota.limit, quota.usage()) else {
                continue;
            };
            if usage + bytes > limit {
                return Err((
                    ErrorKind::Quota,
                    format!(
                        "{bytes} bytes of {what} would take the usage of GPU {} to {} \
                         bytes, past its quota of {limit} bytes:\n{}",
                        ::storage::protocol::gpu_name(&gpu),
                        usage + bytes,
                        quota.describe()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// What the host admits of new host storage now ([`Self::admit_host`]).
    fn host_storage(&self) -> io::Result<HostStorage> {
        Ok(HostStorage {
            available: platform::host::available_memory()?,
            reserve: self.settings.server.host_reserve_bytes(),
            // A host region's parts are storages of their own, admitted whole as each is
            // created: what they have yet to allocate counts here.
            unallocated: self
                .storages
                .values()
                .filter(|storage| matches!(storage.record.location, Location::Host { .. }))
                .map(|storage| {
                    let allocated = rustix::fs::fstat(&storage._descriptor)
                        .map_or(0, |stat| (stat.st_blocks as u64).saturating_mul(512));
                    storage.record.bytes.saturating_sub(allocated)
                })
                .sum::<u64>(),
        })
    }

    /// The retained device storages as the quotas see them: each storage's
    /// record, the access of its tensors and the scopes of the epochs that
    /// retain any of them.
    fn retained(&self) -> Vec<quota::Retained<'_>> {
        let mut retained: HashMap<StorageId, quota::Retained<'_>> = HashMap::new();
        for tensor in self.tensors.values() {
            let Some(storage) = self.storages.get(&tensor.storage) else {
                continue;
            };
            if !matches!(storage.record.location, Location::Device { .. }) {
                continue;
            }
            retained
                .entry(tensor.storage)
                .or_insert_with(|| (&storage.record, tensor.access, BTreeSet::new()))
                .2
                .extend(&tensor.retainers);
        }
        retained.into_values().collect()
    }

    /// The quotas of `gpus` (every GPU when empty) of `deployment` (every
    /// deployment when `None`).
    fn quotas_of(&self, gpus: &[[u8; 16]], deployment: Option<&str>) -> Vec<Quota> {
        let mut quotas = self.quotas.quotas(
            self.retained(),
            &self.transfer_contexts(),
            &|gpu| gpus.is_empty() || gpus.contains(gpu),
            &|name| deployment.is_none_or(|deployment| name == Some(deployment)),
        );
        for quota in &mut quotas {
            quota.host_link_bytes_per_second = self.host_link(quota.gpu);
        }
        quotas
    }

    /// `reserve`: see [`Request::Quota`]. A worker reserves in its deployment's
    /// quota, the Transfer Server in every quota of the GPU.
    fn reserve(
        &mut self,
        id: ClientId,
        reserver: quota::Reserver,
        gpu: [u8; 16],
        total: u64,
        quota: Option<u64>,
        items: &[(Item, u64)],
    ) -> Result<Handling, (ErrorKind, String)> {
        let deployment = match &reserver {
            quota::Reserver::Worker { scope, .. } => {
                let worker =
                    |item: &Item| matches!(item, Item::GraphPool | Item::Process | Item::Staging);
                if let Some(&(item, _)) = items.iter().find(|(item, _)| !worker(item)) {
                    return Err(invalid(&format!("a worker reserves no {}", item.name())));
                }
                Some(scope.deployment.clone())
            }
            quota::Reserver::Transfer => {
                let transfer = |item: &Item| matches!(item, Item::Staging | Item::TransferContext);
                if let Some(&(item, _)) = items.iter().find(|(item, _)| !transfer(item)) {
                    return Err(invalid(&format!(
                        "the Transfer Server reserves no {}",
                        item.name()
                    )));
                }
                None
            }
        };
        let before = self.quotas_of(&[gpu], deployment.as_deref());
        let previous = self.quotas.reserve(id, reserver, gpu, total, quota, items);
        let after = self.quotas_of(&[gpu], deployment.as_deref());
        if let Some(refusal) = quota::overrun(&before, &after) {
            self.quotas = previous;
            return Err((ErrorKind::Quota, refusal));
        }
        Ok(Handling::Response(
            Response::Usage(Usage {
                quotas: after,
                host: self.host_storage().map_err(resource)?,
                tasks: Vec::new(),
            }),
            Vec::new(),
        ))
    }

    /// `keep_weights`: see [`Request::KeepWeights`]. Refused unless the
    /// caller is the deployment's head.
    fn keep_weights(
        &mut self,
        id: ClientId,
        deployment: &str,
        epochs: &[u64],
        cache_keys: &[Key],
    ) -> Result<Handling, (ErrorKind, String)> {
        if !self
            .deployments
            .get(deployment)
            .is_some_and(|entry| entry.head == Some(id))
        {
            return Err(invalid(&format!(
                "only the head that claimed {deployment} keeps its epochs' weights"
            )));
        }
        let released: Vec<(Key, Location)> = self
            .tensors
            .iter_mut()
            .filter(|((key, _), tensor)| {
                tensor.access == Access::Exclusive && !cache_keys.contains(key)
            })
            .filter_map(|(key, tensor)| {
                tensor.retainers.retain(|retainer| {
                    retainer.scope.deployment != deployment || !epochs.contains(&retainer.epoch)
                });
                tensor.retainers.is_empty().then_some(*key)
            })
            .collect();
        let count = released.len() as u64;
        for copy in released {
            self.remove_tensor(&copy);
        }
        // The head stopped these epochs' processes: what they reserved ends
        // with them, however they ended.
        self.quotas.keep_weights(deployment, epochs);
        tracing::info!(
            deployment,
            ?epochs,
            released = count,
            "epochs retain their weights and rollback cache"
        );
        Ok(Handling::Response(
            Response::Retained { released: count },
            Vec::new(),
        ))
    }

    /// Whether a worker of `holder`'s epoch reads `tensor`: its own epoch
    /// published it, or an epoch every epoch of the publisher's deployment
    /// reads did. What an epoch on trial publishes stays its own until its
    /// commit, whichever epochs retain it.
    fn sees(&self, holder: &Holder, tensor: &TensorEntry) -> bool {
        let (deployment, epoch) = &tensor.publisher;
        (*deployment == holder.scope.deployment && *epoch == holder.epoch)
            || !self
                .deployments
                .get(deployment)
                .is_some_and(|entry| entry.private().contains(epoch))
    }

    /// Whether `holder` is a worker of its deployment's current head.
    fn current(&self, holder: &Holder) -> bool {
        self.deployments
            .get(&holder.scope.deployment)
            .is_some_and(|entry| entry.claim == holder.claim)
    }

    /// `retain`: the copies on `at` of the listed keys that exist are held
    /// for the caller's scope in its epoch, and the epoch releases every
    /// other tensor of the scope; other epochs keep theirs.
    fn retain(
        &mut self,
        holder: &Holder,
        keys: &[Key],
        at: &[Location],
    ) -> Result<Handling, (ErrorKind, String)> {
        if !self.current(holder) {
            return Err(stale(holder));
        }
        // An epoch retains only what it reads, at the locations it names.
        let listed: std::collections::HashSet<TensorId> = keys
            .iter()
            .flat_map(|key| self.copies_of(key, at))
            .filter(|id| {
                self.tensors
                    .get(id)
                    .is_some_and(|tensor| self.sees(holder, tensor))
            })
            .collect();
        let retainer = holder.retainer();
        let mut released = Vec::new();
        for (id, tensor) in &mut self.tensors {
            if listed.contains(id) {
                tensor.retainers.insert(retainer.clone());
            } else if tensor.retainers.remove(&retainer) && tensor.retainers.is_empty() {
                released.push(*id);
            }
        }
        let count = released.len() as u64;
        for id in released {
            self.remove_tensor(&id);
        }
        if count > 0 {
            tracing::info!(
                scope = %holder.scope,
                epoch = holder.epoch,
                generation = holder.generation,
                pid = holder.pid,
                released = count,
                "a worker's retention released its scope's tensors"
            );
        }
        Ok(Handling::Response(
            Response::Retained { released: count },
            Vec::new(),
        ))
    }

    /// Previous holders of `tensor` that must be out before `holder` writes
    /// it, or the error that refuses `holder`.
    fn writer_out(
        &self,
        holder: &Holder,
        writer: Option<&Holder>,
    ) -> Result<Option<Holder>, (ErrorKind, String)> {
        let Some(writer) = writer else {
            return Ok(None);
        };
        if Arc::ptr_eq(&writer.pidfd, &holder.pidfd) {
            return Ok(None);
        }
        if self.shootdowns.finished(&writer.pidfd) {
            return Ok(None);
        }
        if self.shootdowns.failed(&writer.pidfd) {
            return Err(shoot_down_failed(writer.pid));
        }
        // A live writer of the current head that is not superseded keeps the
        // tensor: two current workers of one scope at once is a location
        // error, never resolved by killing one of them.
        let superseded = !self.current(writer)
            || writer.client.is_none()
            || (writer.scope == holder.scope && writer.generation < holder.generation);
        if !superseded {
            return Err((
                ErrorKind::Held,
                format!(
                    "process {} of {} holds write access",
                    writer.pid, writer.scope
                ),
            ));
        }
        Ok(Some(writer.clone()))
    }

    /// The copies of `key` at the locations `at` selects.
    fn copies_of(&self, key: &Key, at: &[Location]) -> Vec<TensorId> {
        self.copies
            .get(key)
            .into_iter()
            .flatten()
            .filter(|location| selected(at, location))
            .map(|location| (*key, *location))
            .collect()
    }

    /// `get`: see [`Request::Get`].
    fn get(
        &mut self,
        holder: &Holder,
        keys: &[Key],
        mode: Mode,
        at: &[Location],
    ) -> Result<Handling, (ErrorKind, String)> {
        if mode != Mode::Read && !self.current(holder) {
            return Err(stale(holder));
        }
        // The copies at the locations asked for that the caller sees: a
        // tensor another epoch keeps to itself is missing to the caller.
        let ids: Vec<TensorId> = keys
            .iter()
            .flat_map(|key| self.copies_of(key, at))
            .filter(|id| {
                self.tensors
                    .get(id)
                    .is_some_and(|tensor| self.sees(holder, tensor))
            })
            .collect();
        let mut victims = Vec::new();
        for id in &ids {
            let Some(tensor) = self.tensors.get(id) else {
                continue;
            };
            match (tensor.access, mode) {
                (Access::Exclusive, Mode::Read) => {
                    return Err(invalid(&format!(
                        "{} is exclusive and is taken for writing or in standby",
                        tensor.name
                    )));
                }
                (Access::Exclusive, Mode::Write) => {
                    victims.extend(self.writer_out(holder, tensor.writer.as_ref())?);
                }
                // A buffer pool region's storage records no writer: every
                // session maps it writable, and the pool keeps blocks apart.
                _ => {}
            }
        }
        dedup(&mut victims);
        if !victims.is_empty() {
            return Ok(Handling::Wait(victims));
        }

        let mut storages: Vec<StorageId> = Vec::new();
        let mut tensors = Vec::new();
        let found: BTreeSet<Key> = ids.iter().map(|(key, _)| *key).collect();
        let missing = keys
            .iter()
            .filter(|key| !found.contains(key))
            .copied()
            .collect();
        for id in &ids {
            let Some(tensor) = self.tensors.get_mut(id) else {
                continue;
            };
            let key = &id.0;
            if tensor.access == Access::Exclusive {
                match mode {
                    Mode::Write => tensor.writer = Some(holder.clone()),
                    Mode::Standby => {
                        tensor
                            .standby
                            .retain(|standby| !Arc::ptr_eq(&standby.pidfd, &holder.pidfd));
                        tensor.standby.push(holder.clone());
                    }
                    Mode::Read | Mode::Region => {}
                }
            }
            tensor.retainers.insert(holder.retainer());
            let index = match storages
                .iter()
                .position(|storage| *storage == tensor.storage)
            {
                Some(index) => index,
                None => {
                    storages.push(tensor.storage);
                    storages.len() - 1
                }
            };
            tensors.push(record(*key, tensor, index as u32));
        }
        let records = storages
            .iter()
            .map(|storage| self.storages[storage].record)
            .collect();
        Ok(Handling::Response(
            Response::Tensors(Tensors {
                tensors,
                storages: records,
                missing,
            }),
            storages.into_iter().map(Attachment::Storage).collect(),
        ))
    }

    fn allocate_host(
        &mut self,
        holder: &Holder,
        key: Key,
        name: &str,
        bytes: u64,
    ) -> Result<Handling, (ErrorKind, String)> {
        self.admit_host(bytes)?;
        let at = Location::Host { numa_node: None };
        let descriptor =
            memory::shared_memory("uniserve-storage", bytes, None).map_err(resource)?;
        self.put(
            holder,
            &[TensorRecord {
                key,
                name: name.to_owned(),
                storage: 0,
                offset: 0,
                dtype: "uint8".into(),
                shape: vec![bytes],
                strides: vec![1],
                access: Access::Exclusive,
            }],
            &[StorageRecord {
                location: at,
                bytes,
                segment: None,
                fabric: None,
            }],
            vec![descriptor],
        )?;
        self.get(holder, &[key], Mode::Write, &[at])
    }

    /// `put`: see [`Request::Put`].
    fn put(
        &mut self,
        holder: &Holder,
        tensors: &[TensorRecord],
        storages: &[StorageRecord],
        descriptors: Vec<OwnedFd>,
    ) -> Result<Handling, (ErrorKind, String)> {
        if !self.current(holder) {
            return Err(stale(holder));
        }
        if descriptors.len() != storages.len() {
            return Err(invalid(&format!(
                "{} storages arrived with {} descriptors",
                storages.len(),
                descriptors.len()
            )));
        }
        for tensor in tensors {
            let Some(storage) = storages.get(tensor.storage as usize) else {
                return Err(invalid(&format!("{} names no storage", tensor.name)));
            };
            // A key is a content address: a read-only tensor already
            // retained at the same location holds these very bytes, and
            // the put only adds the caller's scope to it. Exclusive tensors
            // are created once.
            if let Some(existing) = self.tensors.get(&(tensor.key, storage.location))
                && (existing.access, tensor.access) != (Access::ReadOnly, Access::ReadOnly)
            {
                return Err(invalid(&format!("{} is already retained", tensor.name)));
            }
            if tensor.shape.len() != tensor.strides.len() {
                return Err(invalid(&format!("{} has one stride per axis", tensor.name)));
            }
        }
        self.admit_storages(holder, tensors, storages)?;
        let ids: Vec<StorageId> = storages
            .iter()
            .zip(descriptors)
            .map(|(record, descriptor)| {
                let id = self.next_storage;
                self.next_storage += 1;
                self.storages.insert(
                    id,
                    StorageEntry {
                        record: *record,
                        _descriptor: descriptor,
                        tensors: 0,
                        moved_from: None,
                        moving: false,
                        part: false,
                    },
                );
                id
            })
            .collect();
        for tensor in tensors {
            let location = storages[tensor.storage as usize].location;
            let unseen = self
                .tensors
                .get(&(tensor.key, location))
                .is_some_and(|existing| !self.sees(holder, existing));
            if let Some(existing) = self.tensors.get_mut(&(tensor.key, location)) {
                existing.retainers.insert(holder.retainer());
                if !unseen {
                    continue;
                }
                // An epoch on trial, which the caller does not read,
                // published the key: the caller's bytes take the place of
                // that epoch's, so that no epoch reads them through the
                // caller. The epoch on trial's processes keep the storage
                // they map.
                let storage = ids[tensor.storage as usize];
                let previous = std::mem::replace(&mut existing.storage, storage);
                existing.publisher = (holder.scope.deployment.clone(), holder.epoch);
                existing.name.clone_from(&tensor.name);
                existing.offset = tensor.offset;
                existing.dtype.clone_from(&tensor.dtype);
                existing.shape.clone_from(&tensor.shape);
                existing.strides.clone_from(&tensor.strides);
                if let Some(entry) = self.storages.get_mut(&storage) {
                    entry.tensors += 1;
                }
                if let Some(entry) = self.storages.get_mut(&previous) {
                    entry.tensors -= 1;
                }
                continue;
            }
            let storage = ids[tensor.storage as usize];
            if let Some(entry) = self.storages.get_mut(&storage) {
                entry.tensors += 1;
            }
            self.insert_tensor(
                (tensor.key, location),
                TensorEntry {
                    name: tensor.name.clone(),
                    storage,
                    offset: tensor.offset,
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    strides: tensor.strides.clone(),
                    access: tensor.access,
                    retainers: BTreeSet::from([holder.retainer()]),
                    publisher: (holder.scope.deployment.clone(), holder.epoch),
                    writer: (tensor.access == Access::Exclusive).then(|| holder.clone()),
                    standby: Vec::new(),
                },
            );
        }
        // A storage no tensor names is not retained, unless a move fills it
        // or it is a part of a host region.
        self.storages
            .retain(|_, entry| entry.tensors > 0 || entry.moving || entry.part);
        Ok(Handling::Response(Response::Put, Vec::new()))
    }

    /// Refuses a put whose new device storages would take a quota of the
    /// holder's deployment past its quota. A storage is new when a tensor
    /// the put creates lies in it; a read-only tensor already retained adds
    /// the caller's scope to its storage and no storage.
    fn admit_storages(
        &self,
        holder: &Holder,
        tensors: &[TensorRecord],
        storages: &[StorageRecord],
    ) -> Result<(), (ErrorKind, String)> {
        let created: BTreeSet<usize> = tensors
            .iter()
            .filter(|tensor| {
                let location = storages
                    .get(tensor.storage as usize)
                    .map(|storage| storage.location);
                location.is_none_or(|location| !self.tensors.contains_key(&(tensor.key, location)))
            })
            .map(|tensor| tensor.storage as usize)
            .collect();
        let mut added: BTreeMap<[u8; 16], u64> = BTreeMap::new();
        for index in created {
            if let Location::Device { gpu } = storages[index].location {
                *added.entry(gpu).or_default() += storages[index].bytes;
            }
        }
        for (gpu, bytes) in added {
            let what = format!("new device storage of {}", holder.scope);
            self.admit_device(&holder.scope.deployment, gpu, bytes, &what)?;
        }
        Ok(())
    }

    /// `promote`: see [`Request::Promote`].
    fn promote(&mut self, holder: &Holder, keys: &[Key]) -> Result<Handling, (ErrorKind, String)> {
        if !self.current(holder) {
            return Err(stale(holder));
        }
        // The copies the caller maps in standby: an exclusive key names one
        // storage, so one copy each.
        let mut promoted = Vec::with_capacity(keys.len());
        for key in keys {
            let id = self
                .copies_of(key, &[])
                .into_iter()
                .find(|id| {
                    self.tensors.get(id).is_some_and(|tensor| {
                        tensor
                            .standby
                            .iter()
                            .any(|standby| Arc::ptr_eq(&standby.pidfd, &holder.pidfd))
                    })
                })
                .ok_or_else(|| invalid(&format!("{key:?} is not held in standby")))?;
            promoted.push(id);
        }
        let mut victims = Vec::new();
        for id in &promoted {
            let Some(tensor) = self.tensors.get(id) else {
                continue;
            };
            // A promotion takes over from a writer of the previous head or of
            // an earlier launch, whatever its generation.
            if let Some(writer) = &tensor.writer
                && !Arc::ptr_eq(&writer.pidfd, &holder.pidfd)
                && !self.shootdowns.finished(&writer.pidfd)
            {
                if self.shootdowns.failed(&writer.pidfd) {
                    return Err(shoot_down_failed(writer.pid));
                }
                if self.current(writer) && writer.client.is_some() && writer.scope != holder.scope {
                    return Err((
                        ErrorKind::Held,
                        format!(
                            "process {} of {} holds write access",
                            writer.pid, writer.scope
                        ),
                    ));
                }
                victims.push(writer.clone());
            }
        }
        dedup(&mut victims);
        if !victims.is_empty() {
            return Ok(Handling::Wait(victims));
        }
        for id in &promoted {
            if let Some(tensor) = self.tensors.get_mut(id) {
                tensor
                    .standby
                    .retain(|standby| !Arc::ptr_eq(&standby.pidfd, &holder.pidfd));
                tensor.writer = Some(holder.clone());
            }
        }
        Ok(Handling::Response(Response::Promoted, Vec::new()))
    }

    /// `release`: see [`Request::Release`].
    fn release(&mut self, holder: &Holder, keys: &[Key]) -> Result<Handling, (ErrorKind, String)> {
        for key in keys {
            for id in self.copies_of(key, &[]) {
                if let Some(tensor) = self.tensors.get_mut(&id)
                    && tensor
                        .writer
                        .as_ref()
                        .is_some_and(|writer| Arc::ptr_eq(&writer.pidfd, &holder.pidfd))
                {
                    tensor.writer = None;
                }
            }
        }
        Ok(Handling::Response(Response::Released, Vec::new()))
    }

    /// Index a tensor under `id`.
    fn insert_tensor(&mut self, id: TensorId, tensor: TensorEntry) {
        let locations = self.copies.entry(id.0).or_default();
        if !locations.contains(&id.1) {
            locations.push(id.1);
        }
        self.tensors.insert(id, tensor);
    }

    /// Take a tensor out of the index, its storage left as it is.
    fn take_tensor(&mut self, id: &TensorId) -> Option<TensorEntry> {
        let tensor = self.tensors.remove(id)?;
        if let Some(locations) = self.copies.get_mut(&id.0) {
            locations.retain(|location| *location != id.1);
            if locations.is_empty() {
                self.copies.remove(&id.0);
            }
        }
        Some(tensor)
    }

    /// Remove a tensor and release its storage with its last tensor.
    fn remove_tensor(&mut self, id: &TensorId) {
        let Some(tensor) = self.take_tensor(id) else {
            return;
        };
        let empty = self.storages.get_mut(&tensor.storage).is_some_and(|entry| {
            entry.tensors -= 1;
            entry.tensors == 0
        });
        if empty {
            self.storages.remove(&tensor.storage);
            self.allocation_released(Allocation::Storage {
                storage: tensor.storage,
            });
        }
    }

    /// The GPUs whose storage process `pid` may still address: those of the
    /// device tensors it holds access to.
    fn gpus_held_by(&self, pidfd: &Arc<OwnedFd>) -> Vec<[u8; 16]> {
        let mut gpus: Vec<[u8; 16]> = self
            .tensors
            .values()
            .filter(|tensor| {
                tensor
                    .writer
                    .iter()
                    .chain(&tensor.standby)
                    .any(|holder| Arc::ptr_eq(&holder.pidfd, pidfd))
            })
            .filter_map(
                |tensor| match self.storages.get(&tensor.storage)?.record.location {
                    Location::Device { gpu } => Some(gpu),
                    Location::Host { .. } => None,
                },
            )
            .collect();
        gpus.sort_unstable();
        gpus.dedup();
        gpus
    }

    /// Clear the access of every process whose shoot-down finished.
    fn settle_shootdowns(&mut self) {
        let shootdowns = &self.shootdowns;
        for tensor in self.tensors.values_mut() {
            if tensor
                .writer
                .as_ref()
                .is_some_and(|writer| shootdowns.finished(&writer.pidfd))
            {
                tensor.writer = None;
            }
            tensor
                .standby
                .retain(|holder| !shootdowns.finished(&holder.pidfd));
        }
    }

    /// A client is gone: its role ends, but its access stays recorded until
    /// it is released or its process shot down, since the process may
    /// still map the storage.
    fn disconnect(&mut self, id: ClientId) {
        let Some(client) = self.clients.remove(&id) else {
            return;
        };
        // What the process reserved ends with it: the storage it reserved is its
        // own, which the driver frees as the process ends. A worker's that is
        // lost is kept for its relaunch while its epoch is claimed.
        let deployments = &self.deployments;
        self.quotas.disconnect(id, |deployment, epoch| {
            deployments
                .get(deployment)
                .is_some_and(|entry| entry.epochs.contains(&epoch))
        });
        match &client.role {
            Some(Role::Connector) if self.kv_storage.client == Some(id) => self.connector_gone(),
            Some(Role::Transfer) if self.transfers.client == Some(id) => {
                self.transfer_server_gone();
            }
            _ => {}
        }
        self.sessions_client_gone(id, &client);
        self.transfer_client_gone(id);
        if let Some(Role::Head { deployment, .. }) = &client.role
            && let Some(entry) = self.deployments.get_mut(deployment)
            && entry.head == Some(id)
        {
            entry.head = None;
            tracing::info!(deployment, "the deployment's head disconnected");
        }
        for tensor in self.tensors.values_mut() {
            for holder in tensor.writer.iter_mut().chain(tensor.standby.iter_mut()) {
                if holder.client == Some(id) {
                    holder.client = None;
                }
            }
        }
        // What the disconnect ended goes out now, not with the next reply:
        // the failures of the Transfer Server's tasks to their submitters,
        // the cancellations of a gone submitter's tasks to the Transfer
        // Server.
        self.send_events();
        // The client reads its end now, whatever other handles of the socket exist (a child the
        // server forks holds them until it executes): the server acted on its close.
        client.connection.shutdown();
    }

    /// Write pending replies and drop closed clients.
    fn flush_and_reap(&mut self) {
        let mut gone = Vec::new();
        for (id, client) in &mut self.clients {
            if client.connection.try_flush().is_err() {
                gone.push(*id);
            }
        }
        for id in gone {
            self.disconnect(id);
        }
    }

    /// See [`Request::List`].
    fn listing(&self) -> Listing {
        let mut device = BTreeMap::<([u8; 16], Retainer), u64>::new();
        let mut host = BTreeMap::<Retainer, u64>::new();
        let mut counted = BTreeSet::<(StorageId, Retainer)>::new();
        let mut tensors = Vec::with_capacity(self.tensors.len());
        for ((key, _), tensor) in &self.tensors {
            let Some(storage) = self.storages.get(&tensor.storage) else {
                continue;
            };
            for retainer in &tensor.retainers {
                if !counted.insert((tensor.storage, retainer.clone())) {
                    continue;
                }
                match storage.record.location {
                    Location::Device { gpu } => {
                        *device.entry((gpu, retainer.clone())).or_default() += storage.record.bytes;
                    }
                    Location::Host { .. } => {
                        *host.entry(retainer.clone()).or_default() += storage.record.bytes;
                    }
                }
            }
            tensors.push(TensorListing {
                key: *key,
                name: tensor.name.clone(),
                scopes: tensor
                    .retainers
                    .iter()
                    .map(|retainer| retainer.scope.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                retainers: tensor.retainers.iter().cloned().collect(),
                location: storage.record.location,
                storage_bytes: storage.record.bytes,
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                access: tensor.access,
                writer: tensor.writer.as_ref().map(|writer| writer.pid),
            });
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        Listing {
            tensors,
            device_bytes: device
                .into_iter()
                .map(|((gpu, retainer), bytes)| ScopeBytes {
                    gpu: Some(gpu),
                    scope: retainer.scope,
                    epoch: retainer.epoch,
                    bytes,
                })
                .collect(),
            host_bytes: host
                .into_iter()
                .map(|(retainer, bytes)| ScopeBytes {
                    gpu: None,
                    scope: retainer.scope,
                    epoch: retainer.epoch,
                    bytes,
                })
                .collect(),
            deployments: self
                .deployments
                .iter()
                .map(|(deployment, entry)| DeploymentListing {
                    deployment: deployment.clone(),
                    head_connected: entry
                        .head
                        .is_some_and(|head| self.clients.contains_key(&head)),
                })
                .collect(),
            connector: self.kv_storage.client.is_some(),
            children: [&self.kv_storage.process, &self.transfers.process]
                .into_iter()
                .filter_map(child::Process::listed)
                .collect(),
            transfer: self.listed_transfer(),
            quotas: self.quotas_of(&[], None),
            buffer_pools: self
                .sessions
                .thread
                .call(|pools| pools.listing())
                .unwrap_or_default(),
        }
    }

    /// Send the events queued while handling requests.
    fn send_events(&mut self) {
        for (id, event, attachments) in std::mem::take(&mut self.events) {
            let descriptors: Vec<BorrowedFd<'_>> = attachments
                .iter()
                .filter_map(|attachment| match attachment {
                    Attachment::Storage(storage) => self
                        .storages
                        .get(storage)
                        .map(|entry| entry._descriptor.as_fd()),
                })
                .collect();
            let Some(client) = self.clients.get_mut(&id) else {
                continue;
            };
            if let Err(error) = client
                .connection
                .send_with(&ServerMessage::Event(event), &descriptors)
            {
                tracing::warn!(pid = client.pid, %error, "an event to a storage client failed");
            }
        }
    }

    /// Hand `task` to the connector; without one, the task fails at once.
    /// Returns the task's number.
    fn assign(&mut self, task: Task) -> u64 {
        let id = self.kv_storage.next_task();
        let Some(connector) = self.kv_storage.client else {
            self.complete(id, task, &[]);
            return id;
        };
        let (pool, pages) = (task.pool, wide(&task.pages));
        let event = if task.upload {
            Event::Upload {
                task: id,
                pool,
                pages,
            }
        } else {
            Event::Download {
                task: id,
                pool,
                pages,
            }
        };
        self.kv_storage.tasks.insert(id, task);
        self.events.push((connector, event, Vec::new()));
        id
    }

    /// The connector reported the results of `task`.
    fn finished(&mut self, task: u64, results: &[bool]) {
        let Some(entry) = self.kv_storage.tasks.remove(&task) else {
            return;
        };
        self.complete(task, entry, results);
    }

    /// End `task`, numbered `id`, with the connector's `results`, one per
    /// page of the task; a page without a result failed.
    fn complete(&mut self, id: u64, task: Task, results: &[bool]) {
        let moved: Vec<bool> = (0..task.pages.len())
            .map(|index| results.get(index).copied().unwrap_or(false))
            .collect();
        match task.owner {
            kv_storage::Owner::Pool { number, keys } => {
                self.pool_pages_moved(number, &keys, &moved);
            }
            kv_storage::Owner::Client(client) => {
                let event = Event::PagesMoved {
                    task: id,
                    results: moved,
                };
                self.events.push((client, event, Vec::new()));
            }
            kv_storage::Owner::Program { holder } => {
                let pages: Vec<Key> = task.pages.iter().map(|(key, _)| *key).collect();
                self.program_moved(holder, &pages, &moved);
            }
        }
    }

    /// A connector task of the KV program's move under `holder` ended: once its last did, the
    /// pool holds its copies no more.
    fn program_moved(&mut self, holder: u64, pages: &[Key], moved: &[bool]) {
        if let Some(deployment) = self.programs.moved(holder, pages, moved) {
            self.sessions.thread.send(move |pools| {
                pools.release_peer(&deployment, holder);
            });
        }
    }

    /// Steps the KV programs whose period came: each saves its state, and on the leader's node
    /// steps its tiering hook over the directory of this node's pool, whose moves into Mooncake
    /// Store the server makes ([`programs`]): it holds the host copies, has its connector write
    /// their pages, and lets them go once every page's task ended.
    fn step_programs(&mut self) {
        let node = rustix::system::uname()
            .nodename()
            .to_string_lossy()
            .into_owned();
        for deployment in self.programs.due() {
            let named = deployment.clone();
            let holdings = match self
                .sessions
                .thread
                .call(move |pools| pools.holdings(&named))
            {
                Ok(holdings) => holdings,
                Err(error) => {
                    tracing::warn!(deployment, %error, "the buffer pool's holdings are not read");
                    Vec::new()
                }
            };
            let upload = self
                .programs
                .tiering(&deployment, programs::directory(node.clone(), holdings));
            if upload.keys.is_empty() {
                continue;
            }

            let holder = self.programs.holder();
            let (named, keys) = (deployment.clone(), upload.keys.clone());
            let held = match self
                .sessions
                .thread
                .call(move |pools| pools.hold(&named, holder, &keys))
            {
                Ok(held) => held,
                Err(error) => {
                    tracing::warn!(deployment, %error, "the KV program's copies are not held");
                    Vec::new()
                }
            };
            for key in &upload.keys {
                if !held.iter().any(|(copy, _)| copy.key == *key) {
                    let reason = "this node holds no committed host copy of it";
                    self.programs.skipped(&deployment, key, reason);
                }
            }
            let (tasks, skipped) = self.upload_held(&deployment, holder, held);
            for (key, reason) in skipped {
                self.programs.skipped(&deployment, &key, &reason);
            }
            if tasks == 0 {
                let named = deployment.clone();
                self.sessions.thread.send(move |pools| {
                    pools.release_peer(&named, holder);
                });
            } else {
                self.programs.moving(&deployment, holder, tasks);
            }
        }
    }

    /// The connector's connection closed: its tasks are lost, and the next
    /// connector maps every pool again. Each ends as failed once the
    /// connector's process has exited, since it may move pages until then.
    fn connector_gone(&mut self) {
        tracing::warn!("the Mooncake connector disconnected");
        self.kv_storage.client = None;
        self.kv_storage.offered.clear();
        let tasks: Vec<(u64, Task)> = self.kv_storage.tasks.drain().collect();
        let pid = self.kv_storage.process.pid();
        for (id, task) in tasks {
            match (task.owner, pid) {
                (kv_storage::Owner::Pool { number, .. }, _) => self.pool_task_lost(number, pid),
                (kv_storage::Owner::Client(client), Some(pid)) => {
                    self.kv_storage
                        .lost
                        .push((pid, client, id, task.pages.len()));
                }
                (kv_storage::Owner::Client(client), None) => {
                    let results = vec![false; task.pages.len()];
                    let event = Event::PagesMoved { task: id, results };
                    self.events.push((client, event, Vec::new()));
                }
                (kv_storage::Owner::Program { holder }, Some(pid)) => {
                    let pages = task.pages.iter().map(|(key, _)| *key).collect();
                    self.kv_storage.lost_moves.push((pid, holder, pages));
                }
                (kv_storage::Owner::Program { holder }, None) => {
                    let pages: Vec<Key> = task.pages.iter().map(|(key, _)| *key).collect();
                    self.program_moved(holder, &pages, &[]);
                }
            }
        }
    }
}

/// Send client `id` the reply to `request`, with `descriptors` in order.
fn send_reply(
    clients: &mut BTreeMap<ClientId, Client>,
    id: ClientId,
    request: u64,
    reply: Response,
    descriptors: &[BorrowedFd<'_>],
) {
    let Some(client) = clients.get_mut(&id) else {
        return;
    };
    let message = ServerMessage::Response {
        id: request,
        response: reply,
    };
    if let Err(error) = client.connection.send_with(&message, descriptors) {
        tracing::warn!(pid = client.pid, %error, "a reply to a storage client failed");
    }
}

/// Slots in the protocol's width.
fn wide(pages: &[(Key, u32)]) -> Vec<(Key, u64)> {
    pages
        .iter()
        .map(|(key, slot)| (*key, u64::from(*slot)))
        .collect()
}

/// Whether a request's locations `at` select `location`: every location
/// when it lists none.
fn selected(at: &[Location], location: &Location) -> bool {
    at.is_empty() || at.contains(location)
}

/// The record of a retained tensor, naming storage `storage` of a reply.
fn record(key: Key, tensor: &TensorEntry, storage: u32) -> TensorRecord {
    TensorRecord {
        key,
        name: tensor.name.clone(),
        storage,
        offset: tensor.offset,
        dtype: tensor.dtype.clone(),
        shape: tensor.shape.clone(),
        strides: tensor.strides.clone(),
        access: tensor.access,
    }
}

/// Keep one holder per process.
fn dedup(holders: &mut Vec<Holder>) {
    let mut seen: Vec<Arc<OwnedFd>> = Vec::new();
    holders.retain(|holder| {
        let new = !seen.iter().any(|pidfd| Arc::ptr_eq(pidfd, &holder.pidfd));
        if new {
            seen.push(Arc::clone(&holder.pidfd));
        }
        new
    });
}

fn invalid(message: &str) -> (ErrorKind, String) {
    (ErrorKind::Invalid, message.to_owned())
}

fn stale(holder: &Holder) -> (ErrorKind, String) {
    (
        ErrorKind::Invalid,
        format!(
            "process {} of {} belongs to a previous head of its deployment",
            holder.pid, holder.scope
        ),
    )
}

fn shoot_down_failed(pid: i32) -> (ErrorKind, String) {
    (
        ErrorKind::ShootDown,
        format!("process {pid} did not leave its GPU after it was killed; the GPU needs a reset"),
    )
}

fn resource(error: io::Error) -> (ErrorKind, String) {
    (ErrorKind::Resource, error.to_string())
}

/// A fresh claim token, from the kernel's random source (`getrandom(2)`).
fn random_token() -> io::Result<[u8; 16]> {
    let mut token = [0_u8; 16];
    let filled = rustix::rand::getrandom(&mut token, rustix::rand::GetRandomFlags::empty())?;
    if filled != token.len() {
        return Err(io::Error::other(
            "the kernel gave fewer random bytes than asked",
        ));
    }
    Ok(token)
}

/// This host's name, as the kernel states it (`uname(2)`'s node name).
///
/// # Errors
///
/// Fails when the name is not UTF-8.
pub(crate) fn host_name() -> io::Result<String> {
    rustix::system::uname()
        .nodename()
        .to_str()
        .map(str::to_owned)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "this host's name is not UTF-8"))
}

/// Read everything a wake descriptor holds.
fn drain(wake: BorrowedFd<'_>) {
    let mut buffer = [0_u8; 64];
    // SAFETY: a nonblocking read into a local buffer.
    while unsafe { libc::read(wake.as_raw_fd(), buffer.as_mut_ptr().cast(), buffer.len()) } > 0 {}
}
