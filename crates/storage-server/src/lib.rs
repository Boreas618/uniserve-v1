//! The Storage Server: memory whose contents outlive the head and the
//! workers.
//!
//! The server holds descriptors of physical allocations (CUDA VMM
//! allocations through their POSIX file descriptor handles, and host
//! memfds), the tensors clients publish over them, and which client may
//! write each exclusive tensor. It never maps, reads or writes the memory and
//! never creates a CUDA context: device copies run on the workers' streams.
//! The physical memory of a storage lives while the server holds its
//! descriptor or any process maps it.
//!
//! The server also keeps the host's page pools ([`pages`]): the slots of
//! the host KV cache and which page each holds. Its Mooncake connector
//! ([`mooncake`], run as a child process [`kv_storage`] supervises) keeps
//! committed pages in the KV storage and reads them back for
//! prefetches. Its Transfer Server ([`transfer`], another child process,
//! which [`transfers`] supervises) copies bytes between the memory the
//! server holds, for the tasks heads and workers submit; that process, not
//! the server, holds the CUDA contexts the copies need.
//!
//! [`serve`] runs a server, the `uniserve storage serve` command, and
//! [`list`] and [`retire`] are the operator's commands against a running
//! one.
//!
//! One thread serves every client with `poll(2)`. Requests are answered in
//! order on their connection. A request that must wait for a previous
//! writer to be shot down ([`shoot`]), or for the connector's lookup, holds
//! its connection's queue until then, while every other connection is
//! served on.

mod child;
mod kv_storage;
mod ledger;
pub mod mooncake;
mod pages;
mod restart;
mod serve;
mod shoot;
pub mod transfer;
mod transfers;

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::{self, Read};
use std::net::TcpListener;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
use std::time::Duration;

use rustix::event::{PollFd, PollFlags, poll};

use crate::mooncake::BATCH_PAGES;
pub use child::Launch;
use kv_storage::{KvStorage, Task};
pub use pages::DEFAULT_POOL_BYTES;
pub use restart::{Pending as RestartPending, TAKEOVER_DESCRIPTOR, relaunch_arguments};
pub use serve::{ServeOptions, list, retire, run_connector, run_transfer_server, serve};
use shoot::ShootDowns;
use storage::protocol::{
    Access, Allocation, Claim, ClientMessage, ErrorKind, Event, GpuLedger, Hello, HostMemory, Item,
    Key, ListedDeployment, ListedTensor, Listing, MIN_VERSION, Mode, PageLevel, Placement, Reply,
    Request, Retainer, Role, Scope, ScopeBytes, ServerMessage, StorageRecord, TensorRecord,
    VERSION, Welcome,
};
use storage::transport::Connection;

/// How often a shoot-down that waits for a GPU to drop its victim polls
/// NVML.
const NVML_POLL: Duration = Duration::from_millis(20);

/// How long one request may hold the single-threaded server before it is
/// logged: every other client's request waits meanwhile.
const SLOW_REQUEST: Duration = Duration::from_millis(100);

type ClientId = u64;
type StorageId = u64;
/// One copy of a tensor: its key and where its storage lies.
type TensorId = (Key, Placement);

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
    /// The world the process belongs to.
    world: u64,
}

impl Holder {
    /// The retention the holder's requests act under.
    fn retainer(&self) -> Retainer {
        Retainer {
            scope: self.scope.clone(),
            world: self.world,
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
    /// page and transfer requests or another host's Transfer Server's
    /// exposures.
    pidfd: Option<Arc<OwnedFd>>,
    role: Option<Role>,
    /// Requests received and not yet answered, in arrival order.
    queue: VecDeque<(u64, Request, Vec<OwnedFd>)>,
    /// Whether a request taken from the queue waits for the connector or
    /// the Transfer Server, which holds the queue until it is answered.
    held: bool,
}

struct StorageEntry {
    record: StorageRecord,
    /// Keeps the physical memory alive.
    _descriptor: OwnedFd,
    /// Tensors over the storage; the storage is released with its last.
    tensors: usize,
}

struct TensorEntry {
    name: String,
    storage: StorageId,
    offset: u64,
    dtype: String,
    shape: Vec<u64>,
    strides: Vec<u64>,
    access: Access,
    /// The scopes of the worlds that retain the tensor. A world retains what
    /// its workers use until they release it or the head retires the
    /// world, so a world's memory outlives its processes and is released
    /// with the world.
    retainers: BTreeSet<Retainer>,
    /// The deployment and world whose worker put the tensor: a world on
    /// trial keeps what it publishes to itself until its commit.
    publisher: (String, u64),
    /// The process that may write an exclusive tensor.
    writer: Option<Holder>,
    /// Processes that map an exclusive tensor in standby.
    standby: Vec<Holder>,
}

struct Deployment {
    claim: Claim,
    head: Option<ClientId>,
    /// The worlds claimed and not retired. The oldest serves; those newer
    /// are on trial and keep what they publish to themselves.
    worlds: BTreeSet<u64>,
    /// The worlds a rollback retired while on trial: what they publish
    /// while their processes leave stays theirs.
    discarded: BTreeSet<u64>,
}

impl Deployment {
    /// The worlds that keep what they publish to themselves: those on
    /// trial, claimed and newer than the oldest, and those discarded.
    fn private(&self) -> BTreeSet<u64> {
        let trial = self.worlds.iter().skip(1);
        trial.chain(&self.discarded).copied().collect()
    }
}

/// What handling a request came to.
enum Outcome {
    Reply(Reply, Vec<Attachment>),
    /// The request waits for these processes to be shot down.
    Wait(Vec<Holder>),
    /// The request is answered when the connector finishes this task.
    KvStorage(Task),
    /// A peer's exposure, answered once the Transfer Server registered its
    /// regions.
    Exposing(transfers::Exposing),
}

/// A descriptor that travels with a reply.
#[derive(Clone, Copy)]
enum Attachment {
    Storage(StorageId),
    Pool(u64),
}

/// The server's state and its listening socket.
pub struct Server {
    listener: UnixListener,
    /// The TCP listener other hosts' heads reach the page pools through,
    /// with the address it is bound to.
    tcp: Option<(TcpListener, String)>,
    clients: BTreeMap<ClientId, Client>,
    next_client: ClientId,
    storages: HashMap<StorageId, StorageEntry>,
    next_storage: StorageId,
    /// Every retained tensor by its key and where its storage lies: a key
    /// names what the bytes are, its storage where they lie, and a key
    /// has at most one copy at each location.
    tensors: HashMap<TensorId, TensorEntry>,
    /// Where each key's copies lie.
    copies: HashMap<Key, Vec<Placement>>,
    deployments: HashMap<String, Deployment>,
    shootdowns: ShootDowns,
    pages: pages::Pools,
    /// Host memory a new page pool leaves available to the host's other
    /// processes.
    host_reserve: u64,
    /// The GPUs' memory ledgers: what the processes book and the quotas
    /// the worlds state ([`ledger`]).
    ledgers: ledger::Ledgers,
    kv_storage: KvStorage,
    transfers: transfers::Transfers,
    /// Events to send once the request being handled is answered.
    events: Vec<(ClientId, Event, Vec<Attachment>)>,
    /// The arguments a hot restart's successor starts with.
    relaunch: Vec<String>,
    /// A hot restart under way, which hands the server over once nothing is
    /// in flight.
    restart: Option<restart::Pending>,
}

impl Server {
    /// Serve on `listener`, which the caller bound with mode 0600, with
    /// page pools of `page_pool_bytes` each unless a worker asks for another
    /// size, leaving `host_reserve` bytes of the host's available memory to
    /// other processes, and run the Mooncake connector `connector` and the
    /// Transfer Server `transfer` when given.
    pub fn new(
        listener: UnixListener,
        page_pool_bytes: u64,
        host_reserve: u64,
        connector: Option<Launch>,
        transfer: Option<Launch>,
    ) -> io::Result<Self> {
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            tcp: None,
            clients: BTreeMap::new(),
            next_client: 1,
            storages: HashMap::new(),
            next_storage: 1,
            tensors: HashMap::new(),
            copies: HashMap::new(),
            deployments: HashMap::new(),
            shootdowns: ShootDowns::default(),
            pages: pages::Pools::new(page_pool_bytes),
            host_reserve,
            ledgers: ledger::Ledgers::default(),
            kv_storage: KvStorage::new(connector),
            transfers: transfers::Transfers::new(transfer),
            events: Vec::new(),
            relaunch: Vec::new(),
            restart: None,
        })
    }

    /// Also accept connections from other hosts on `listener`: a head's
    /// page requests, without descriptors. Returns the bound address.
    pub fn listen_tcp(&mut self, listener: TcpListener) -> io::Result<String> {
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?.to_string();
        self.tcp = Some((listener, address.clone()));
        Ok(address)
    }

    /// Serve until `stop` reads true. `stop` is checked after every wake;
    /// `wake` is a descriptor whose readability wakes the loop (a signal
    /// pipe), drained when it fires. Returns once `stop` holds, or with the
    /// restart under way once nothing is in flight, which the caller then
    /// completes ([`Server::hand_over`]).
    pub fn run(
        &mut self,
        wake: BorrowedFd<'_>,
        stop: &dyn Fn() -> bool,
    ) -> io::Result<Option<restart::Pending>> {
        while !stop() {
            self.turn(wake)?;
            if self.restart.is_some() && self.quiescent() {
                return Ok(self.restart.take());
            }
        }
        Ok(None)
    }

    /// One wait and the work it makes ready.
    fn turn(&mut self, wake: BorrowedFd<'_>) -> io::Result<()> {
        let ids: Vec<ClientId> = self.clients.keys().copied().collect();
        let pidfds = self.shootdowns.pending_pidfds();
        let mut fds = Vec::with_capacity(3 + ids.len() + pidfds.len());
        fds.push(PollFd::new(&self.listener, PollFlags::IN));
        fds.push(PollFd::new(&wake, PollFlags::IN));
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
            fds.push(PollFd::new(client.connection.stream(), flags));
        }
        for pidfd in &pidfds {
            fds.push(PollFd::new(pidfd.as_ref(), PollFlags::IN));
        }
        for process in [&self.kv_storage.process, &self.transfers.process] {
            if let Some((_, pidfd)) = &process.child {
                fds.push(PollFd::new(pidfd, PollFlags::IN));
            }
        }
        let timeout = [
            self.shootdowns.polling().then_some(NVML_POLL),
            self.kv_storage.process.timeout(),
            self.transfers.process.timeout(),
        ]
        .into_iter()
        .flatten()
        .min()
        .map(|timeout| rustix::time::Timespec::try_from(timeout).unwrap_or_default());
        match poll(&mut fds, timeout.as_ref()) {
            Ok(_) | Err(rustix::io::Errno::INTR) => {}
            Err(error) => return Err(error.into()),
        }
        let accept = !fds[0].revents().is_empty();
        if !fds[1].revents().is_empty() {
            drain(wake);
        }
        let accept_tcp = tcp.is_some() && !fds[2].revents().is_empty();
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
        if self.shootdowns.advance() {
            self.settle_shootdowns();
        }
        self.kv_storage.process.reap();
        self.kv_storage.process.start_due();
        self.transfers.process.reap();
        self.transfers.process.start_due();
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
            let admitted = stream
                .set_nonblocking(true)
                .and_then(|()| stream.set_nodelay(true));
            if let Err(error) = admitted {
                tracing::warn!(%error, "a remote storage client could not be admitted");
                continue;
            }
            let id = self.next_client;
            self.next_client += 1;
            self.clients.insert(
                id,
                Client {
                    connection: Connection::new(stream),
                    pid: 0,
                    pidfd: None,
                    role: None,
                    queue: VecDeque::new(),
                    held: false,
                },
            );
        }
    }

    /// A connection's peer process, held by a pidfd so that its identity
    /// cannot be reused while the server refers to it.
    fn admit(&self, stream: UnixStream) -> io::Result<Client> {
        stream.set_nonblocking(true)?;
        let credentials = rustix::net::sockopt::socket_peercred(&stream)?;
        let pid = credentials.pid;
        let pidfd = rustix::process::pidfd_open(pid, rustix::process::PidfdFlags::empty())?;
        Ok(Client {
            connection: Connection::new(stream),
            pid: pid.as_raw_nonzero().get(),
            pidfd: Some(Arc::new(pidfd)),
            role: None,
            queue: VecDeque::new(),
            held: false,
        })
    }

    /// Read what a client sent and queue its requests. A client whose
    /// connection closed has its received requests answered, then is
    /// disconnected.
    fn read(&mut self, id: ClientId) {
        let mut failed = false;
        let mut frames = Vec::new();
        if let Some(client) = self.clients.get_mut(&id) {
            loop {
                match client.connection.receive() {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(error) => {
                        tracing::warn!(pid = client.pid, %error, "a storage client failed");
                        failed = true;
                        break;
                    }
                }
            }
            while !failed {
                match client.connection.next_frame::<ClientMessage>() {
                    Ok(Some(frame)) => frames.push(frame),
                    Ok(None) => break,
                    Err(error) => {
                        tracing::warn!(pid = client.pid, %error, "a storage client sent an unreadable message");
                        failed = true;
                    }
                }
            }
        }
        for (message, descriptors) in frames {
            match message {
                ClientMessage::Hello(hello) => {
                    let welcome = self.welcome(id, hello);
                    if let Some(client) = self.clients.get_mut(&id) {
                        let _ = client
                            .connection
                            .send(&ServerMessage::Welcome(welcome), &[]);
                    }
                    if self.kv_storage.client == Some(id) {
                        self.offer_pools();
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
            // Requests already received are answered first: a client may
            // send a final release and close at once.
            self.serve_queue(id);
        }
        if closed || failed {
            self.disconnect(id);
        }
    }

    /// Admit a client's role, or refuse it.
    fn welcome(&mut self, id: ClientId, hello: Hello) -> Welcome {
        if !(MIN_VERSION..=VERSION).contains(&hello.version) {
            return Welcome::Refused {
                message: format!(
                    "the storage server speaks protocols {MIN_VERSION} to {VERSION}, the client {}",
                    hello.version
                ),
            };
        }
        if let Some(client) = self.clients.get_mut(&id) {
            if client.role.is_some() {
                return Welcome::Refused {
                    message: "the connection already said hello".into(),
                };
            }
            if client.pidfd.is_none() && !matches!(hello.role, Role::Head { .. } | Role::Peer) {
                return Welcome::Refused {
                    message: "a connection from another host is a head's or a Transfer Server's"
                        .into(),
                };
            }
            if matches!(hello.role, Role::Connector) {
                // The connector's client is forgotten when its connection
                // closes, so one that is known is connected.
                if self.kv_storage.client.is_some() {
                    return Welcome::Refused {
                        message: "the server has a connector".into(),
                    };
                }
                self.kv_storage.client = Some(id);
                tracing::info!(pid = client.pid, "the Mooncake connector connected");
            }
        }
        if matches!(hello.role, Role::Transfer) {
            let pid = self.clients.get(&id).map_or(0, |client| client.pid);
            if let Err(message) = self.transfer_hello(id, pid) {
                return Welcome::Refused { message };
            }
        }
        if let Some(client) = self.clients.get_mut(&id) {
            client.role = Some(hello.role);
        }
        Welcome::Accepted { version: VERSION }
    }

    /// Answer every client's queued requests that can be answered now.
    fn serve_queues(&mut self) {
        let ids: Vec<ClientId> = self.clients.keys().copied().collect();
        for id in ids {
            self.serve_queue(id);
        }
    }

    /// Answer one client's queued requests in order, until one waits.
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
            let outcome = self.handle(id, &request, descriptors);
            let handled = started.elapsed();
            match outcome {
                Ok(Outcome::Reply(reply, storages)) => {
                    self.answer(id, request_id, reply, &storages);
                    self.send_events();
                    // The server is single-threaded: a request that holds
                    // it this long delays every other client's.
                    let served = started.elapsed();
                    if served >= SLOW_REQUEST {
                        tracing::info!(
                            request = request_name(&request),
                            handled_ms = handled.as_millis() as u64,
                            served_ms = served.as_millis() as u64,
                            "a request held the server"
                        );
                    }
                }
                Ok(Outcome::KvStorage(mut task)) => {
                    if let Task::Lookup { request, .. } = &mut task {
                        *request = request_id;
                    }
                    if let Some(client) = self.clients.get_mut(&id) {
                        client.held = true;
                    }
                    self.assign(task);
                    self.send_events();
                    return;
                }
                Ok(Outcome::Exposing(exposing)) => {
                    // Holds the connection until the Transfer Server
                    // reports, unless it is refused at once.
                    self.begin_exposure(id, request_id, exposing);
                    self.send_events();
                }
                Ok(Outcome::Wait(victims)) => {
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
                    self.answer(id, request_id, Reply::Error { kind, message }, &[]);
                }
            }
        }
    }

    /// Send a reply with the descriptors of `storages`, in order.
    fn answer(&mut self, id: ClientId, request: u64, reply: Reply, attachments: &[Attachment]) {
        let descriptors: Vec<BorrowedFd<'_>> = attachments
            .iter()
            .filter_map(|attachment| match attachment {
                Attachment::Storage(storage) => self
                    .storages
                    .get(storage)
                    .map(|entry| entry._descriptor.as_fd()),
                Attachment::Pool(pool) => self.pages.get(*pool).map(|pool| pool.descriptor.as_fd()),
            })
            .collect();
        let Some(client) = self.clients.get_mut(&id) else {
            return;
        };
        let message = ServerMessage::Reply { id: request, reply };
        if let Err(error) = client.connection.send(&message, &descriptors) {
            tracing::warn!(pid = client.pid, %error, "a reply to a storage client failed");
        }
    }

    /// Handle one request. A request that needs previous writers shot down
    /// changes nothing and returns [`Outcome::Wait`].
    fn handle(
        &mut self,
        id: ClientId,
        request: &Request,
        descriptors: Vec<OwnedFd>,
    ) -> Result<Outcome, (ErrorKind, String)> {
        let client = &self.clients[&id];
        let role = client
            .role
            .clone()
            .ok_or_else(|| invalid("a request precedes the client's hello"))?;
        match (request, role) {
            (Request::List, _) => Ok(Outcome::Reply(Reply::Listing(self.listing()), Vec::new())),
            (Request::Endpoint, _) => Ok(Outcome::Reply(
                Reply::Endpoint {
                    tcp: self.tcp.as_ref().map(|(_, address)| address.clone()),
                },
                Vec::new(),
            )),
            (Request::Claim { .. }, Role::Head { .. }) if client.pidfd.is_none() => Err(invalid(
                "a deployment is claimed at its host's server, not from another host",
            )),
            (
                Request::PagePool { .. }
                | Request::Lookup { .. }
                | Request::Reserve { .. }
                | Request::Commit { .. }
                | Request::Abort { .. }
                | Request::Pin { .. }
                | Request::Unpin { .. }
                | Request::Prefetch { .. }
                | Request::HostCache { .. },
                role,
            ) => {
                let outcome = self.pages(id, request, &role);
                // A reservation or a prefetch may have evicted pages, and an
                // abort freed some: the windows that followed them end.
                self.pages.settle();
                outcome
            }
            (
                Request::Transfer { .. } | Request::Cancel { .. } | Request::Throttle { .. },
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
                // A download that failed freed the slots it had reserved.
                self.pages.settle();
                Ok(Outcome::Reply(Reply::Finished, Vec::new()))
            }
            (Request::Retire { deployment }, Role::Operator) => self.retire(deployment),
            (Request::Restart { executable }, Role::Operator) => self
                .begin_restart(executable.as_deref())
                .map(|pid| Outcome::Reply(Reply::Restarting { pid }, Vec::new()))
                .map_err(resource),
            (
                Request::Claim {
                    scopes,
                    world,
                    retire,
                },
                Role::Head { deployment, .. },
            ) => self.claim(id, &deployment, scopes, *world, retire),
            (Request::KeepWeights { worlds }, Role::Head { deployment, .. }) => {
                self.keep_weights(id, &deployment, worlds)
            }
            (Request::Ledger { gpus }, role) => {
                let deployment = match &role {
                    Role::Head { deployment, .. } => Some(deployment.clone()),
                    Role::Worker { scope, .. } => Some(scope.deployment.clone()),
                    Role::Transfer | Role::Peer | Role::Operator | Role::Connector => None,
                };
                let ledgers = self.ledgers_of(gpus, deployment.as_deref());
                Ok(Outcome::Reply(
                    Reply::Ledger {
                        ledgers,
                        host: self.host_memory().map_err(resource)?,
                    },
                    Vec::new(),
                ))
            }
            (
                Request::Book {
                    gpu,
                    total,
                    quota,
                    items,
                },
                Role::Worker { scope, world, .. },
            ) => self.book(
                id,
                ledger::Booker::Worker { scope, world },
                *gpu,
                *total,
                *quota,
                items,
            ),
            (
                Request::Book {
                    gpu, total, items, ..
                },
                Role::Transfer,
            ) => self.book(id, ledger::Booker::Transfer, *gpu, *total, None, items),
            (
                request,
                Role::Worker {
                    scope,
                    generation,
                    claim,
                    world,
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
                    world,
                };
                match request {
                    Request::Retain { keys, at } => self.retain(&holder, keys, at),
                    Request::Get { keys, mode, at } => self.get(&holder, keys, *mode, at),
                    Request::Put { tensors, storages } => {
                        self.put(&holder, tensors, storages, descriptors)
                    }
                    Request::Promote { keys } => self.promote(&holder, keys),
                    Request::Release { keys } => self.release(&holder, keys),
                    Request::Rekey { moves, at } => self.rekey(&holder, moves, at),
                    _ => Err(invalid(&format!(
                        "{} is not a worker request",
                        request_name(request)
                    ))),
                }
            }
            (request, role) => Err(invalid(&format!(
                "{} is not a request of {role:?}",
                request_name(request)
            ))),
        }
    }

    /// `claim`: see [`Request::Claim`].
    fn claim(
        &mut self,
        id: ClientId,
        deployment: &str,
        scopes: &[Scope],
        world: u64,
        retire: &[u64],
    ) -> Result<Outcome, (ErrorKind, String)> {
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
        // The head that holds the claim keeps it for every world it claims:
        // its workers, of any world, stay current.
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
                                world,
                            }),
                            Some(pidfd),
                        ) if scope.deployment == deployment => Some(Holder {
                            client: Some(*other),
                            pid: client.pid,
                            pidfd: Arc::clone(pidfd),
                            scope: scope.clone(),
                            generation: *generation,
                            claim: *claim,
                            world: *world,
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
                return Ok(Outcome::Wait(victims));
            }
        }

        let claim = match held {
            Some(claim) => claim,
            None => Claim(random_token().map_err(resource)?),
        };
        // The head that holds the claim adds and retires worlds. A new head
        // starts from its own world: the worlds a previous head kept on
        // trial are discarded with that head, never committed, and the
        // pages of every world discarded before are relabelled as such, so
        // that the new head's worlds may take their numbers.
        let (trial, mut discarded, mut worlds) = match self.deployments.get(deployment) {
            Some(entry) => (
                entry.worlds.iter().skip(1).copied().collect(),
                entry.discarded.clone(),
                entry.worlds.clone(),
            ),
            None => (BTreeSet::new(), BTreeSet::new(), BTreeSet::new()),
        };
        let mut retired: BTreeSet<u64> = retire.iter().copied().collect();
        if held.is_none() {
            retired.extend(trial.iter().filter(|&&old| old != world));
            worlds.clear();
        }
        worlds.insert(world);
        worlds.retain(|world| !retired.contains(world));
        // A world on trial that a claim retires is rolled back: its pages
        // are discarded. A world on trial that the deployment keeps but no
        // longer as a trial is committed: every world reads its pages,
        // which the KV storage now takes.
        let oldest = worlds.first().copied();
        for &old in &trial {
            if retired.contains(&old) {
                self.pages.discard_world(deployment, old);
                discarded.insert(old);
            } else if Some(old) == oldest {
                for (pool, pages) in self.pages.published_by(deployment, old) {
                    self.upload(pool, pages);
                }
            }
        }
        if held.is_none() {
            for &old in &discarded {
                self.pages.discard_world(deployment, old);
            }
            discarded.clear();
        }
        self.deployments.insert(
            deployment.to_owned(),
            Deployment {
                claim,
                head: Some(id),
                worlds,
                discarded,
            },
        );
        // The retired worlds map no page pool any more.
        self.pages.retire_users(deployment, &retired);
        // The world's retention of the scopes it no longer places ends, and
        // every retired world's retention ends.
        let listed: BTreeSet<&Scope> = scopes.iter().collect();
        let released: Vec<TensorId> = self
            .tensors
            .iter_mut()
            .filter_map(|(id, tensor)| {
                tensor.retainers.retain(|retainer| {
                    retainer.scope.deployment != deployment
                        || !(retired.contains(&retainer.world)
                            || (retainer.world == world && !listed.contains(&retainer.scope)))
                });
                tensor.retainers.is_empty().then_some(*id)
            })
            .collect();
        let count = released.len();
        for id in released {
            self.remove_tensor(&id);
        }
        // A retired world's quota ends with its memory: the ledger then
        // books what its successor alone holds, under the successor's.
        self.ledgers.retire(deployment, &retired);
        tracing::info!(
            deployment,
            world,
            ?retire,
            released = count,
            "a head claimed the deployment"
        );
        // The host serves the deployments whose heads run on it: every
        // other one is retired, so its weights and caches do not outlive it.
        for other in self.idle_deployments(deployment) {
            self.release_deployment(&other);
        }
        Ok(Outcome::Reply(Reply::Claimed { claim }, Vec::new()))
    }

    /// The deployments other than `except` that hold a claim, a tensor or a
    /// page pool and whose head is not connected.
    fn idle_deployments(&self, except: &str) -> Vec<String> {
        let mut named: BTreeSet<String> = self.deployments.keys().cloned().collect();
        named.extend(self.tensors.values().flat_map(|tensor| {
            tensor
                .retainers
                .iter()
                .map(|retainer| retainer.scope.deployment.clone())
        }));
        named.extend(self.pages.deployments());
        named
            .into_iter()
            .filter(|deployment| deployment != except)
            .filter(|deployment| {
                !self.deployments.get(deployment).is_some_and(|entry| {
                    entry
                        .head
                        .is_some_and(|head| self.clients.contains_key(&head))
                })
            })
            .collect()
    }

    /// Release every tensor and page pool of `deployment` and forget its
    /// claim. Returns the tensors released.
    fn release_deployment(&mut self, deployment: &str) -> u64 {
        self.deployments.remove(deployment);
        let released: Vec<TensorId> = self
            .tensors
            .iter_mut()
            .filter_map(|(id, tensor)| {
                tensor
                    .retainers
                    .retain(|retainer| retainer.scope.deployment != deployment);
                tensor.retainers.is_empty().then_some(*id)
            })
            .collect();
        let count = released.len() as u64;
        for id in released {
            self.remove_tensor(&id);
        }
        let pools = self.pages.release(deployment);
        self.pools_released(&pools);
        self.ledgers.release(deployment);
        tracing::info!(
            deployment,
            released = count,
            pools = pools.len(),
            "retired a deployment"
        );
        count
    }

    /// Tell the connector and the Transfer Server that they no longer map
    /// `pools`.
    fn pools_released(&mut self, pools: &[u64]) {
        for &pool in pools {
            self.allocation_released(Allocation::Pool { pool });
            if self.kv_storage.offered.remove(&pool)
                && let Some(connector) = self.kv_storage.client
            {
                self.events
                    .push((connector, Event::PoolReleased { pool }, Vec::new()));
            }
        }
    }

    /// Refuse `bytes` of new host memory that the host cannot hold: its
    /// available memory less the server's reserve and what the page pools
    /// have yet to allocate.
    fn admit_host(&self, bytes: u64) -> Result<(), (ErrorKind, String)> {
        let host = self.host_memory().map_err(resource)?;
        let room = host.room();
        if bytes > room {
            return Err((
                ErrorKind::Resource,
                format!(
                    "{bytes} bytes of host memory exceed the {room} bytes the host can hold: \
                     {} bytes available, less {} bytes the page pools have yet to allocate and \
                     the server's reserve of {} bytes",
                    host.available, host.pools_unallocated, host.reserve
                ),
            ));
        }
        Ok(())
    }

    /// What the host admits of new host memory now ([`Self::admit_host`]).
    fn host_memory(&self) -> io::Result<HostMemory> {
        Ok(HostMemory {
            available: available_memory()?,
            reserve: self.host_reserve,
            pools_unallocated: self.pages.unallocated(),
        })
    }

    /// The retained device storages as the ledgers see them: each storage's
    /// record, the access of its tensors and the scopes of the worlds that
    /// retain any of them.
    fn held(&self) -> Vec<ledger::Held<'_>> {
        let mut held: HashMap<StorageId, ledger::Held<'_>> = HashMap::new();
        for tensor in self.tensors.values() {
            let Some(storage) = self.storages.get(&tensor.storage) else {
                continue;
            };
            if !matches!(storage.record.placement, Placement::Device { .. }) {
                continue;
            }
            held.entry(tensor.storage)
                .or_insert_with(|| ledger::Held {
                    record: &storage.record,
                    access: tensor.access,
                    retainers: BTreeSet::new(),
                })
                .retainers
                .extend(&tensor.retainers);
        }
        held.into_values().collect()
    }

    /// The ledgers of `gpus` (every GPU when empty) of `deployment` (every
    /// deployment when `None`).
    fn ledgers_of(&self, gpus: &[[u8; 16]], deployment: Option<&str>) -> Vec<GpuLedger> {
        self.ledgers.ledgers(
            self.held(),
            &self.transfer_contexts(),
            &|gpu| gpus.is_empty() || gpus.contains(gpu),
            &|name| deployment.is_none_or(|deployment| deployment == name),
        )
    }

    /// `book`: see [`Request::Book`]. A worker books in its deployment's
    /// ledger, the Transfer Server in every ledger of the GPU.
    fn book(
        &mut self,
        id: ClientId,
        booker: ledger::Booker,
        gpu: [u8; 16],
        total: u64,
        quota: Option<u64>,
        items: &[(Item, u64)],
    ) -> Result<Outcome, (ErrorKind, String)> {
        let deployment = match &booker {
            ledger::Booker::Worker { scope, .. } => {
                let worker =
                    |item: &Item| matches!(item, Item::GraphPool | Item::Process | Item::Staging);
                if let Some(&(item, _)) = items.iter().find(|(item, _)| !worker(item)) {
                    return Err(invalid(&format!("a worker books no {}", item.name())));
                }
                Some(scope.deployment.clone())
            }
            ledger::Booker::Transfer => {
                let transfer = |item: &Item| matches!(item, Item::Staging | Item::TransferContext);
                if let Some(&(item, _)) = items.iter().find(|(item, _)| !transfer(item)) {
                    return Err(invalid(&format!(
                        "the Transfer Server books no {}",
                        item.name()
                    )));
                }
                None
            }
        };
        let before = self.ledgers_of(&[gpu], deployment.as_deref());
        let previous = self.ledgers.book(id, booker, gpu, total, quota, items);
        let after = self.ledgers_of(&[gpu], deployment.as_deref());
        if let Some(refusal) = ledger::overrun(&before, &after) {
            self.ledgers = previous;
            return Err((ErrorKind::Quota, refusal));
        }
        Ok(Outcome::Reply(
            Reply::Ledger {
                ledgers: after,
                host: self.host_memory().map_err(resource)?,
            },
            Vec::new(),
        ))
    }

    /// `keep_weights`: see [`Request::KeepWeights`]. Refused unless the
    /// caller is the deployment's head.
    fn keep_weights(
        &mut self,
        id: ClientId,
        deployment: &str,
        worlds: &[u64],
    ) -> Result<Outcome, (ErrorKind, String)> {
        if !self
            .deployments
            .get(deployment)
            .is_some_and(|entry| entry.head == Some(id))
        {
            return Err(invalid(&format!(
                "only the head that claimed {deployment} keeps its worlds' weights"
            )));
        }
        let released: Vec<(Key, Placement)> = self
            .tensors
            .iter_mut()
            .filter(|(_, tensor)| tensor.access == Access::Exclusive)
            .filter_map(|(key, tensor)| {
                tensor.retainers.retain(|retainer| {
                    retainer.scope.deployment != deployment || !worlds.contains(&retainer.world)
                });
                tensor.retainers.is_empty().then_some(*key)
            })
            .collect();
        let count = released.len() as u64;
        for copy in released {
            self.remove_tensor(&copy);
        }
        tracing::info!(
            deployment,
            ?worlds,
            released = count,
            "worlds keep their weights alone"
        );
        Ok(Outcome::Reply(
            Reply::Retained { released: count },
            Vec::new(),
        ))
    }

    /// `retire`: see [`Request::Retire`]. Processes that still map the
    /// deployment's memory keep it alive until they end; nobody takes new
    /// access to it.
    fn retire(&mut self, deployment: &str) -> Result<Outcome, (ErrorKind, String)> {
        if let Some(entry) = self.deployments.get(deployment)
            && entry
                .head
                .is_some_and(|head| self.clients.contains_key(&head))
        {
            return Err((
                ErrorKind::Claimed,
                format!("deployment {deployment} has a connected head"),
            ));
        }
        let count = self.release_deployment(deployment);
        Ok(Outcome::Reply(
            Reply::Retired { released: count },
            Vec::new(),
        ))
    }

    /// What a head of `world` of `deployment` reads.
    fn visibility(&self, deployment: &str, world: u64) -> pages::Visibility {
        pages::Visibility {
            world,
            private: self
                .deployments
                .get(deployment)
                .map(Deployment::private)
                .unwrap_or_default(),
        }
    }

    /// What the head connected as `client` reads; nothing of a world on
    /// trial once it is gone.
    fn head_visibility(&self, client: ClientId) -> pages::Visibility {
        match self
            .clients
            .get(&client)
            .and_then(|client| client.role.as_ref())
        {
            Some(Role::Head { deployment, world }) => self.visibility(deployment, *world),
            _ => self.visibility("", 0),
        }
    }

    /// Whether a worker of `holder`'s world reads `tensor`: its own world
    /// published it, or a world every world of the publisher's deployment
    /// reads did. What a world on trial publishes stays its own until its
    /// commit, whichever worlds retain it.
    fn sees(&self, holder: &Holder, tensor: &TensorEntry) -> bool {
        let (deployment, world) = &tensor.publisher;
        (*deployment == holder.scope.deployment && *world == holder.world)
            || !self
                .deployments
                .get(deployment)
                .is_some_and(|entry| entry.private().contains(world))
    }

    /// Whether `holder` is a worker of its deployment's current head.
    fn current(&self, holder: &Holder) -> bool {
        self.deployments
            .get(&holder.scope.deployment)
            .is_some_and(|entry| entry.claim == holder.claim)
    }

    /// `retain`: the copies on `at` of the listed keys that exist are held
    /// for the caller's scope in its world, and the world releases every
    /// other tensor of the scope; other worlds keep theirs.
    fn retain(
        &mut self,
        holder: &Holder,
        keys: &[Key],
        at: &[Placement],
    ) -> Result<Outcome, (ErrorKind, String)> {
        if !self.current(holder) {
            return Err(stale(holder));
        }
        // A world retains only what it reads, at the locations it names.
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
        Ok(Outcome::Reply(
            Reply::Retained { released: count },
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
        // tensor: two current workers of one scope at once is a placement
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
    fn copies_of(&self, key: &Key, at: &[Placement]) -> Vec<TensorId> {
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
        at: &[Placement],
    ) -> Result<Outcome, (ErrorKind, String)> {
        if mode != Mode::Read && !self.current(holder) {
            return Err(stale(holder));
        }
        // The copies at the locations asked for that the caller sees: a
        // tensor another world keeps to itself is missing to the caller.
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
                _ => {}
            }
        }
        dedup(&mut victims);
        if !victims.is_empty() {
            return Ok(Outcome::Wait(victims));
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
                    Mode::Read => {}
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
        Ok(Outcome::Reply(
            Reply::Got {
                tensors,
                storages: records,
                missing,
            },
            storages.into_iter().map(Attachment::Storage).collect(),
        ))
    }

    /// `put`: see [`Request::Put`].
    fn put(
        &mut self,
        holder: &Holder,
        tensors: &[TensorRecord],
        storages: &[StorageRecord],
        descriptors: Vec<OwnedFd>,
    ) -> Result<Outcome, (ErrorKind, String)> {
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
            if let Some(existing) = self.tensors.get(&(tensor.key, storage.placement))
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
                    },
                );
                id
            })
            .collect();
        for tensor in tensors {
            let location = storages[tensor.storage as usize].placement;
            let unseen = self
                .tensors
                .get(&(tensor.key, location))
                .is_some_and(|existing| !self.sees(holder, existing));
            if let Some(existing) = self.tensors.get_mut(&(tensor.key, location)) {
                existing.retainers.insert(holder.retainer());
                if !unseen {
                    continue;
                }
                // A world on trial, which the caller does not read,
                // published the key: the caller's bytes take the place of
                // that world's, so that no world reads them through the
                // caller. The world on trial's processes keep the memory
                // they map.
                let storage = ids[tensor.storage as usize];
                let previous = std::mem::replace(&mut existing.storage, storage);
                existing.publisher = (holder.scope.deployment.clone(), holder.world);
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
                    publisher: (holder.scope.deployment.clone(), holder.world),
                    writer: (tensor.access == Access::Exclusive).then(|| holder.clone()),
                    standby: Vec::new(),
                },
            );
        }
        // A storage no tensor names is not retained.
        self.storages.retain(|_, entry| entry.tensors > 0);
        Ok(Outcome::Reply(Reply::Put, Vec::new()))
    }

    /// Refuses a put whose new device storages would take a ledger of the
    /// holder's deployment past its quota. A storage is new when a tensor
    /// the put creates lies in it; a read-only tensor already retained adds
    /// the caller's scope to its storage and no memory.
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
                    .map(|storage| storage.placement);
                location.is_none_or(|location| !self.tensors.contains_key(&(tensor.key, location)))
            })
            .map(|tensor| tensor.storage as usize)
            .collect();
        let mut added: BTreeMap<[u8; 16], u64> = BTreeMap::new();
        for index in created {
            if let Placement::Device { gpu } = storages[index].placement {
                *added.entry(gpu).or_default() += storages[index].bytes;
            }
        }
        for (gpu, bytes) in added {
            let deployment = holder.scope.deployment.as_str();
            for ledger in self.ledgers_of(&[gpu], Some(deployment)) {
                let (Some(quota), used) = (ledger.quota, ledger.used()) else {
                    continue;
                };
                if used + bytes > quota {
                    return Err((
                        ErrorKind::Quota,
                        format!(
                            "{bytes} bytes of new device storage of {} would take the memory \
                             ledger of GPU {} to {} bytes, past its quota of {quota} bytes:\n{}",
                            holder.scope,
                            storage::protocol::gpu_name(&gpu),
                            used + bytes,
                            ledger.describe()
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// `promote`: see [`Request::Promote`].
    fn promote(&mut self, holder: &Holder, keys: &[Key]) -> Result<Outcome, (ErrorKind, String)> {
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
            return Ok(Outcome::Wait(victims));
        }
        for id in &promoted {
            if let Some(tensor) = self.tensors.get_mut(id) {
                tensor
                    .standby
                    .retain(|standby| !Arc::ptr_eq(&standby.pidfd, &holder.pidfd));
                tensor.writer = Some(holder.clone());
            }
        }
        Ok(Outcome::Reply(Reply::Promoted, Vec::new()))
    }

    /// `release`: see [`Request::Release`].
    fn release(&mut self, holder: &Holder, keys: &[Key]) -> Result<Outcome, (ErrorKind, String)> {
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
        Ok(Outcome::Reply(Reply::Released, Vec::new()))
    }

    /// `rekey`: see [`Request::Rekey`].
    fn rekey(
        &mut self,
        holder: &Holder,
        moves: &[(Key, Key)],
        at: &[Placement],
    ) -> Result<Outcome, (ErrorKind, String)> {
        if !self.current(holder) {
            return Err(stale(holder));
        }
        // Each copy moves to the new key at its own location.
        let mut renamed: Vec<(TensorId, TensorId)> = Vec::new();
        let mut moved = std::collections::HashSet::new();
        for (old, new) in moves {
            let copies = self.copies_of(old, at);
            if copies.is_empty() {
                return Err(invalid(&format!("{old:?} is not retained")));
            }
            for id in copies {
                let Some(tensor) = self.tensors.get(&id) else {
                    continue;
                };
                if tensor.access != Access::ReadOnly {
                    return Err(invalid(&format!("{} is not read-only", tensor.name)));
                }
                if let Some(other) = tensor
                    .retainers
                    .iter()
                    .map(|retainer| &retainer.scope)
                    .find(|scope| scope.deployment != holder.scope.deployment)
                {
                    return Err((
                        ErrorKind::Held,
                        format!("{} is held by {other} as well", tensor.name),
                    ));
                }
                let target = (*new, id.1);
                if self.tensors.contains_key(&target) || !moved.insert(target) {
                    return Err(invalid(&format!(
                        "{} cannot move to {new:?}, which is retained",
                        tensor.name
                    )));
                }
                renamed.push((id, target));
            }
        }
        for (old, new) in renamed {
            if let Some(tensor) = self.take_tensor(&old) {
                self.insert_tensor(new, tensor);
            }
        }
        Ok(Outcome::Reply(Reply::Rekeyed, Vec::new()))
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

    /// The GPUs whose memory process `pid` may still address: those of the
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
                |tensor| match self.storages.get(&tensor.storage)?.record.placement {
                    Placement::Device { gpu } => Some(gpu),
                    Placement::Host { .. } => None,
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
    /// still map the memory.
    fn disconnect(&mut self, id: ClientId) {
        let Some(client) = self.clients.remove(&id) else {
            return;
        };
        // What the process booked ends with it: the memory it booked is its
        // own, which the driver frees as the process ends.
        self.ledgers.disconnect(id);
        match &client.role {
            Some(Role::Head { .. }) => {
                let clients = &self.clients;
                self.pages
                    .head_gone(id, &|scope| connected_worker(clients, scope));
            }
            Some(Role::Worker { scope, .. }) if !connected_worker(&self.clients, scope) => {
                self.pages.worker_gone(scope);
            }
            Some(Role::Connector) if self.kv_storage.client == Some(id) => self.connector_gone(),
            Some(Role::Transfer) if self.transfers.client == Some(id) => {
                self.transfer_server_gone();
            }
            _ => {}
        }
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
    }

    /// Write pending replies and drop closed clients.
    fn flush_and_reap(&mut self) {
        let mut gone = Vec::new();
        for (id, client) in &mut self.clients {
            if client.connection.flush().is_err() {
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
                match storage.record.placement {
                    Placement::Device { gpu } => {
                        *device.entry((gpu, retainer.clone())).or_default() += storage.record.bytes;
                    }
                    Placement::Host { .. } => {
                        *host.entry(retainer.clone()).or_default() += storage.record.bytes;
                    }
                }
            }
            tensors.push(ListedTensor {
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
                placement: storage.record.placement,
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
                    world: retainer.world,
                    bytes,
                })
                .collect(),
            host_bytes: host
                .into_iter()
                .map(|(retainer, bytes)| ScopeBytes {
                    gpu: None,
                    scope: retainer.scope,
                    world: retainer.world,
                    bytes,
                })
                .collect(),
            deployments: self
                .deployments
                .iter()
                .map(|(deployment, entry)| ListedDeployment {
                    deployment: deployment.clone(),
                    head_connected: entry
                        .head
                        .is_some_and(|head| self.clients.contains_key(&head)),
                })
                .collect(),
            page_pools: self.pages.listing(),
            connector: self.kv_storage.client.is_some(),
            transfer: self.listed_transfer(),
            ledgers: self.ledgers_of(&[], None),
        }
    }

    /// Send the events queued while handling requests.
    fn send_events(&mut self) {
        for (id, event, attachments) in std::mem::take(&mut self.events) {
            let descriptors: Vec<BorrowedFd<'_>> = attachments
                .iter()
                .filter_map(|attachment| match attachment {
                    Attachment::Pool(pool) => {
                        self.pages.get(*pool).map(|pool| pool.descriptor.as_fd())
                    }
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
                .send(&ServerMessage::Event(event), &descriptors)
            {
                tracing::warn!(pid = client.pid, %error, "an event to a storage client failed");
            }
        }
    }

    /// Give the connector every page pool it does not map yet.
    fn offer_pools(&mut self) {
        let Some(connector) = self.kv_storage.client else {
            return;
        };
        for listed in self.pages.listing() {
            if !self.pages.released(listed.pool) && self.kv_storage.offered.insert(listed.pool) {
                self.events.push((
                    connector,
                    Event::Pool {
                        pool: listed.pool,
                        shape: listed.shape,
                        slots: listed.slots,
                    },
                    vec![Attachment::Pool(listed.pool)],
                ));
            }
        }
    }

    /// Hand `task` to the connector; without one, the task fails at once.
    fn assign(&mut self, task: Task) {
        let Some(connector) = self.kv_storage.client else {
            self.fail(task);
            return;
        };
        let id = self.kv_storage.next_task();
        let event = match &task {
            Task::Lookup { pool, keys, .. } => Event::Exists {
                task: id,
                pool: *pool,
                keys: keys.clone(),
            },
            Task::Upload { pool, pages } => Event::Upload {
                task: id,
                pool: *pool,
                pages: wide(pages),
            },
            Task::Download { pool, pages, .. } => Event::Download {
                task: id,
                pool: *pool,
                pages: wide(pages),
            },
        };
        self.kv_storage.tasks.insert(id, task);
        self.events.push((connector, event, Vec::new()));
    }

    /// Write pages committed in `pool` to the KV storage, in tasks of
    /// at most [`BATCH_PAGES`] pages, holding their slots until each task
    /// ends.
    fn upload(&mut self, pool: u64, pages: Vec<(Key, u32)>) {
        if self.kv_storage.client.is_none() || pages.is_empty() {
            return;
        }
        let Some(entry) = self.pages.get_mut(pool) else {
            return;
        };
        entry.upload(&pages);
        for chunk in pages.chunks(BATCH_PAGES) {
            self.assign(Task::Upload {
                pool,
                pages: chunk.to_vec(),
            });
        }
    }

    /// `prefetch`: see [`Request::Prefetch`].
    fn prefetch(
        &mut self,
        head: ClientId,
        pool: u64,
        keys: &[Key],
        tag: u64,
        visibility: &pages::Visibility,
    ) -> Result<(), (ErrorKind, String)> {
        let entry = pool_of(&mut self.pages, pool)?;
        let fetches = if self.kv_storage.client.is_some() {
            entry.prefetch(keys, visibility)
        } else {
            vec![pages::Fetch::Full; keys.len()]
        };
        let pages: Vec<(Key, u32)> = keys
            .iter()
            .zip(&fetches)
            .filter_map(|(key, fetch)| match fetch {
                pages::Fetch::Reserved(slot) => Some((*key, *slot)),
                _ => None,
            })
            .collect();
        if pages.is_empty() {
            let results = keys
                .iter()
                .map(|key| entry.holds(key, visibility))
                .collect();
            self.events
                .push((head, Event::Prefetched { tag, results }, Vec::new()));
            return Ok(());
        }
        self.assign(Task::Download {
            head,
            tag,
            pool,
            pages,
            keys: keys.to_vec(),
        });
        Ok(())
    }

    /// The connector reported the results of `task`.
    fn finished(&mut self, task: u64, results: &[bool]) {
        let Some(task) = self.kv_storage.tasks.remove(&task) else {
            return;
        };
        self.complete(task, results);
        let reaped = self.pages.reap();
        self.pools_released(&reaped);
    }

    /// End `task` as if the connector reported every page failed.
    fn fail(&mut self, task: Task) {
        self.complete(task, &[]);
        self.pages.settle();
        let reaped = self.pages.reap();
        self.pools_released(&reaped);
    }

    /// End `task` with the connector's `results`, one per page of the task;
    /// a page without a result failed.
    fn complete(&mut self, task: Task, results: &[bool]) {
        let succeeded = |index: usize| results.get(index).copied().unwrap_or(false);
        match task {
            Task::Lookup {
                client,
                request,
                pool,
                keys,
                held,
            } => {
                // A page the pool did not hold is in L2 when the pool holds
                // it by now, and in L3 when the connector found it.
                let visibility = self.head_visibility(client);
                let mut missing = keys.iter().enumerate();
                let levels = held
                    .iter()
                    .map(|&held| {
                        if held {
                            return PageLevel::Host;
                        }
                        let Some((index, key)) = missing.next() else {
                            return PageLevel::Absent;
                        };
                        if self
                            .pages
                            .get(pool)
                            .is_some_and(|entry| entry.holds(key, &visibility))
                        {
                            PageLevel::Host
                        } else if succeeded(index) {
                            PageLevel::Storage
                        } else {
                            PageLevel::Absent
                        }
                    })
                    .collect();
                self.answer(client, request, Reply::Lookup { levels }, &[]);
                if let Some(entry) = self.clients.get_mut(&client) {
                    entry.held = false;
                }
                self.serve_queue(client);
            }
            Task::Upload { pool, pages } => {
                let failed = (0..pages.len()).filter(|index| !succeeded(*index)).count();
                if failed > 0 {
                    tracing::warn!(pool, failed, "pages were not written to the KV storage");
                }
                if let Some(entry) = self.pages.get_mut(pool) {
                    entry.uploaded(&pages);
                }
            }
            Task::Download {
                head,
                tag,
                pool,
                pages,
                keys,
            } => {
                let visibility = self.head_visibility(head);
                let Some(entry) = self.pages.get_mut(pool) else {
                    return;
                };
                let arrived: Vec<bool> = (0..pages.len()).map(succeeded).collect();
                entry.fetched(&pages, &arrived);
                let results = keys
                    .iter()
                    .map(|key| entry.holds(key, &visibility))
                    .collect();
                self.events
                    .push((head, Event::Prefetched { tag, results }, Vec::new()));
                self.send_events();
            }
        }
    }

    /// The connector's connection closed: its tasks fail, and the next
    /// connector maps every pool again.
    fn connector_gone(&mut self) {
        tracing::warn!("the Mooncake connector disconnected");
        self.kv_storage.client = None;
        self.kv_storage.offered.clear();
        let tasks: Vec<Task> = self
            .kv_storage
            .tasks
            .drain()
            .map(|(_, task)| task)
            .collect();
        for task in tasks {
            self.fail(task);
        }
        self.pages.connector_gone();
        // The connector maps no pool any more: released pools leave.
        self.pages.reap();
    }

    /// A page pool request: `page_pool` from a worker, the others from a
    /// head, for scopes of its deployment.
    fn pages(
        &mut self,
        id: ClientId,
        request: &Request,
        role: &Role,
    ) -> Result<Outcome, (ErrorKind, String)> {
        let reply = |reply| Ok(Outcome::Reply(reply, Vec::new()));
        if let (
            Request::PagePool {
                shape,
                numa_node,
                bytes,
            },
            Role::Worker { scope, .. },
        ) = (request, role)
        {
            let pool = match self.pages.find(&scope.deployment, *shape, *numa_node) {
                Some(pool) => pool,
                None => {
                    let slots = self
                        .pages
                        .slots_of(shape.page_bytes, *bytes)
                        .map_err(resource)?;
                    // A pool the host cannot hold first takes the place of
                    // the deployment's pools no world maps any more, whose
                    // pages a previous world left.
                    if self.admit_host(slots * shape.page_bytes).is_err() {
                        let unreferenced: Vec<u64> = self
                            .pages
                            .unreferenced(&scope.deployment)
                            .into_iter()
                            .map(|(pool, _)| pool)
                            .collect();
                        let released = self.pages.release_pools(&unreferenced);
                        self.pools_released(&released);
                        self.admit_host(slots * shape.page_bytes)?;
                    }
                    self.pages
                        .create(&scope.deployment, *shape, *numa_node, slots)
                        .map_err(resource)?
                }
            };
            if let Role::Worker { scope, world, .. } = role {
                self.pages.add_user(pool, scope, *world);
            }
            let slots = self.pages.get(pool).map_or(0, pages::Pool::slots);
            self.offer_pools();
            return Ok(Outcome::Reply(
                Reply::PagePool {
                    pool,
                    slots,
                    page_bytes: shape.page_bytes,
                },
                vec![Attachment::Pool(pool)],
            ));
        }
        let Role::Head { deployment, world } = role else {
            return Err(invalid(&format!(
                "{} is a request of a head",
                request_name(request)
            )));
        };
        let visibility = self.visibility(deployment, *world);
        let of_deployment = |scope: &Scope| {
            if &scope.deployment == deployment {
                Ok(())
            } else {
                Err(invalid(&format!("{scope} is not a scope of {deployment}")))
            }
        };
        if let Some(pool) = request_pool(request)
            && let Some(entry) = self.pages.get(pool)
            && (&entry.deployment != deployment || self.pages.released(pool))
        {
            return Err(invalid(&format!(
                "page pool {pool} is not a live pool of {deployment}"
            )));
        }
        match request {
            Request::Lookup { pool, keys } => {
                let held = pool_of(&mut self.pages, *pool)?.lookup(keys, &visibility);
                let missing: Vec<Key> = keys
                    .iter()
                    .zip(&held)
                    .filter(|(_, held)| !**held)
                    .map(|(key, _)| *key)
                    .collect();
                if self.kv_storage.client.is_none() || missing.is_empty() {
                    // Without a connector, L3 holds nothing beyond L2.
                    let levels = held
                        .iter()
                        .map(|&held| {
                            if held {
                                PageLevel::Host
                            } else {
                                PageLevel::Absent
                            }
                        })
                        .collect();
                    return reply(Reply::Lookup { levels });
                }
                // The connector looks up the pages the pool does not hold;
                // the request waits for it.
                Ok(Outcome::KvStorage(Task::Lookup {
                    client: id,
                    request: 0,
                    pool: *pool,
                    keys: missing,
                    held,
                }))
            }
            Request::Reserve {
                pool,
                writer,
                keys,
                window,
            } => {
                of_deployment(writer)?;
                let slots = self
                    .pages
                    .reserve(*pool, writer, id, keys, *window, &visibility)
                    .ok_or_else(|| invalid(&format!("page pool {pool} does not exist")))?;
                reply(Reply::Reserved { slots })
            }
            Request::Commit { pool, keys } => {
                let entry = pool_of(&mut self.pages, *pool)?;
                let mut committed = entry.commit(keys);
                // The KV storage takes the pages every world reads; those of
                // a world on trial wait for its commit.
                committed.retain(|(_, slot)| {
                    entry
                        .publisher(*slot)
                        .is_some_and(|published| visibility.shared(published))
                });
                self.upload(*pool, committed);
                reply(Reply::Committed)
            }
            Request::Abort { pool, keys } => {
                pool_of(&mut self.pages, *pool)?.abort(keys);
                reply(Reply::Aborted)
            }
            Request::Pin { pool, reader, keys } => {
                of_deployment(reader)?;
                let slots = pool_of(&mut self.pages, *pool)?.pin(reader, id, keys, &visibility);
                reply(Reply::Pinned { slots })
            }
            Request::Unpin { pool, reader, keys } => {
                of_deployment(reader)?;
                pool_of(&mut self.pages, *pool)?.unpin(reader, keys);
                reply(Reply::Unpinned)
            }
            Request::HostCache {
                tokens,
                token_bytes,
                host_bytes,
            } => {
                let per_token: u64 = token_bytes.iter().sum();
                let budget = match host_bytes {
                    Some(bytes) => *bytes,
                    None => {
                        let releasable: u64 = self
                            .pages
                            .unreferenced(deployment)
                            .iter()
                            .map(|(_, bytes)| bytes)
                            .sum();
                        let available = (available_memory().map_err(resource)? + releasable)
                            .saturating_sub(self.pages.unallocated())
                            .saturating_sub(self.host_reserve);
                        available / 5 * 4
                    }
                };
                let held = match budget.checked_div(per_token) {
                    Some(fits) => fits.min(*tokens),
                    None => *tokens,
                };
                tracing::info!(
                    deployment,
                    world,
                    tokens,
                    per_token,
                    budget,
                    held,
                    "sized the host KV cache of a world"
                );
                reply(Reply::HostCache { tokens: held })
            }
            Request::Prefetch { pool, keys, tag } => {
                self.prefetch(id, *pool, keys, *tag, &visibility)?;
                reply(Reply::Prefetching)
            }
            _ => Err(invalid(&format!(
                "{} is not a page request of a head",
                request_name(request)
            ))),
        }
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
fn selected(at: &[Placement], location: &Placement) -> bool {
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

/// The page pool a head's page request names.
fn request_pool(request: &Request) -> Option<u64> {
    match request {
        Request::Lookup { pool, .. }
        | Request::Reserve { pool, .. }
        | Request::Commit { pool, .. }
        | Request::Abort { pool, .. }
        | Request::Pin { pool, .. }
        | Request::Unpin { pool, .. }
        | Request::Prefetch { pool, .. } => Some(*pool),
        _ => None,
    }
}

/// The host's available memory (`MemAvailable` of `/proc/meminfo`), in
/// bytes.
fn available_memory() -> io::Result<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let available = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.trim().strip_suffix("kB"))
        .and_then(|kib| kib.trim().parse::<u64>().ok())
        .map(|kib| kib * 1024)
        .ok_or_else(|| io::Error::other("/proc/meminfo states no MemAvailable"))?;
    Ok(cgroup_room().map_or(available, |room| available.min(room)))
}

/// What the server's memory cgroup (v2) lets it allocate beyond what the
/// cgroup holds: its limit less its usage; `None` without a limit.
fn cgroup_room() -> Option<u64> {
    let membership = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))?
        .trim();
    let directory = std::path::Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/'));
    let read = |name: &str| std::fs::read_to_string(directory.join(name)).ok();
    let limit = read("memory.max")?.trim().parse::<u64>().ok()?;
    let usage = read("memory.current")?.trim().parse::<u64>().ok()?;
    Some(limit.saturating_sub(usage))
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

fn request_name(request: &Request) -> &'static str {
    match request {
        Request::Claim { .. } => "claim",
        Request::Retain { .. } => "retain",
        Request::KeepWeights { .. } => "keep_weights",
        Request::Get { .. } => "get",
        Request::Put { .. } => "put",
        Request::Promote { .. } => "promote",
        Request::Release { .. } => "release",
        Request::Rekey { .. } => "rekey",
        Request::List => "list",
        Request::Retire { .. } => "retire",
        Request::Restart { .. } => "restart",
        Request::PagePool { .. } => "page_pool",
        Request::Lookup { .. } => "lookup",
        Request::Reserve { .. } => "reserve",
        Request::Commit { .. } => "commit",
        Request::Abort { .. } => "abort",
        Request::Pin { .. } => "pin",
        Request::Unpin { .. } => "unpin",
        Request::Endpoint => "endpoint",
        Request::Prefetch { .. } => "prefetch",
        Request::HostCache { .. } => "host_cache",
        Request::Finished { .. } => "finished",
        Request::Book { .. } => "book",
        Request::Ledger { .. } => "ledger",
        Request::Transfer { .. } => "transfer",
        Request::Cancel { .. } => "cancel",
        Request::Throttle { .. } => "throttle",
        Request::Measured { .. } => "measured",
        Request::Copied { .. } => "copied",
        Request::Expose { .. } => "expose",
        Request::Unexpose { .. } => "unexpose",
        Request::Registered { .. } => "registered",
    }
}

/// Page pool `pool`, which a head request names.
fn pool_of(pages: &mut pages::Pools, pool: u64) -> Result<&mut pages::Pool, (ErrorKind, String)> {
    pages
        .get_mut(pool)
        .ok_or_else(|| invalid(&format!("page pool {pool} does not exist")))
}

/// Whether a worker of `scope` is connected.
fn connected_worker(clients: &BTreeMap<ClientId, Client>, scope: &Scope) -> bool {
    clients.values().any(
        |client| matches!(&client.role, Some(Role::Worker { scope: other, .. }) if other == scope),
    )
}

/// A fresh claim token.
fn random_token() -> io::Result<[u8; 16]> {
    let mut token = [0_u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut token)?;
    Ok(token)
}

/// Read everything a wake descriptor holds.
fn drain(wake: BorrowedFd<'_>) {
    let mut buffer = [0_u8; 64];
    // SAFETY: a nonblocking read into a local buffer.
    while unsafe { libc::read(wake.as_raw_fd(), buffer.as_mut_ptr().cast(), buffer.len()) } > 0 {}
}
