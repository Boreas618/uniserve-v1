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
//! the host page tier and which page each holds. Its Mooncake connector
//! ([`connector`]) keeps committed pages in the distributed tier and reads
//! them back for prefetches.
//!
//! One thread serves every client with `poll(2)`. Requests are answered in
//! order on their connection. A request that must wait for a previous
//! writer to be shot down ([`shoot`]), or for the connector's lookup, holds
//! its connection's queue until then, while every other connection is
//! served on.

mod connector;
mod nvml;
mod pages;
mod shoot;

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::{self, Read};
use std::net::TcpListener;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
use std::time::Duration;

use rustix::event::{PollFd, PollFlags, poll};

use crate::connector::BATCH_PAGES;
use crate::protocol::{
    Access, Claim, ClientMessage, ErrorKind, Event, Hello, Key, ListedDeployment, ListedTensor,
    Listing, Mode, Placement, Reply, Request, Role, Scope, ScopeBytes, ServerMessage,
    StorageRecord, TensorRecord, VERSION, Welcome,
};
use crate::transport::Connection;
pub use connector::Launch as ConnectorLaunch;
use connector::{Connector, Task};
pub use pages::DEFAULT_POOL_BYTES;
use shoot::ShootDowns;

/// How often a shoot-down that waits for a GPU to drop its victim polls
/// NVML.
const NVML_POLL: Duration = Duration::from_millis(20);

type ClientId = u64;
type StorageId = u64;

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
}

struct Client {
    connection: Connection,
    /// The peer process of a connection on the host; 0 for a connection
    /// from another host.
    pid: i32,
    /// The peer process, held so that its identity cannot be reused; `None`
    /// for a connection from another host, which may only make a head's
    /// page requests.
    pidfd: Option<Arc<OwnedFd>>,
    role: Option<Role>,
    /// Requests received and not yet answered, in arrival order.
    queue: VecDeque<(u64, Request, Vec<OwnedFd>)>,
    /// Whether a request taken from the queue waits for the connector, which
    /// holds the queue until it is answered.
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
    /// Scopes whose lifetime holds the tensor.
    scopes: BTreeSet<Scope>,
    /// The process that may write an exclusive tensor.
    writer: Option<Holder>,
    /// Processes that map an exclusive tensor in standby.
    standby: Vec<Holder>,
}

struct Deployment {
    claim: Claim,
    head: Option<ClientId>,
}

/// What handling a request came to.
enum Outcome {
    Reply(Reply, Vec<Attachment>),
    /// The request waits for these processes to be shot down.
    Wait(Vec<Holder>),
    /// The request is answered when the connector finishes this task.
    Connector(Task),
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
    tensors: HashMap<Key, TensorEntry>,
    deployments: HashMap<String, Deployment>,
    shootdowns: ShootDowns,
    pages: pages::Pools,
    connector: Connector,
    /// Events to send once the request being handled is answered.
    events: Vec<(ClientId, Event, Vec<Attachment>)>,
}

impl Server {
    /// Serve on `listener`, which the caller bound with mode 0600, with
    /// page pools of `page_pool_bytes` each, and run the Mooncake connector
    /// `connector` when given.
    pub fn new(
        listener: UnixListener,
        page_pool_bytes: u64,
        connector: Option<ConnectorLaunch>,
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
            deployments: HashMap::new(),
            shootdowns: ShootDowns::default(),
            pages: pages::Pools::new(page_pool_bytes),
            connector: Connector::new(connector),
            events: Vec::new(),
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
    /// pipe), drained when it fires.
    pub fn run(&mut self, wake: BorrowedFd<'_>, stop: &dyn Fn() -> bool) -> io::Result<()> {
        while !stop() {
            self.turn(wake)?;
        }
        Ok(())
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
        if let Some((_, pidfd)) = &self.connector.child {
            fds.push(PollFd::new(pidfd, PollFlags::IN));
        }
        let timeout = [
            self.shootdowns.polling().then_some(NVML_POLL),
            self.connector.timeout(),
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
        self.connector.reap();
        self.connector.start_due();
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
                    if self.connector.client == Some(id) {
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
        if hello.version != VERSION {
            return Welcome::Refused {
                message: format!(
                    "the storage server speaks protocol {VERSION}, the client {}",
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
            if client.pidfd.is_none() && !matches!(hello.role, Role::Head { .. }) {
                return Welcome::Refused {
                    message: "a connection from another host is a head's".into(),
                };
            }
            if matches!(hello.role, Role::Connector) {
                // The connector's client is forgotten when its connection
                // closes, so one that is known is connected.
                if self.connector.client.is_some() {
                    return Welcome::Refused {
                        message: "the server has a connector".into(),
                    };
                }
                self.connector.client = Some(id);
                tracing::info!(pid = client.pid, "the Mooncake connector connected");
            }
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
            let outcome = self.handle(id, &request, descriptors);
            match outcome {
                Ok(Outcome::Reply(reply, storages)) => {
                    self.answer(id, request_id, reply, &storages);
                    self.send_events();
                }
                Ok(Outcome::Connector(mut task)) => {
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
                | Request::Prefetch { .. },
                role,
            ) => self.pages(id, request, &role),
            (Request::Finished { task, results }, Role::Connector) => {
                self.finished(*task, results);
                Ok(Outcome::Reply(Reply::Finished, Vec::new()))
            }
            (Request::Retire { deployment }, Role::Operator) => self.retire(deployment),
            (Request::Claim { scopes }, Role::Head { deployment }) => {
                self.claim(id, &deployment, scopes)
            }
            (
                request,
                Role::Worker {
                    scope,
                    generation,
                    claim,
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
                };
                match request {
                    Request::Retain { keys } => self.retain(&holder, keys),
                    Request::Get { keys, mode } => self.get(&holder, keys, *mode),
                    Request::Put { tensors, storages } => {
                        self.put(&holder, tensors, storages, descriptors)
                    }
                    Request::Promote { keys } => self.promote(&holder, keys),
                    Request::Release { keys } => self.release(&holder, keys),
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
        // Every process that holds access to the deployment's tensors under
        // a previous claim, and every worker of a previous claim still
        // connected, is out before the new claim stands.
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
                        }),
                        Some(pidfd),
                    ) if scope.deployment == deployment => Some(Holder {
                        client: Some(*other),
                        pid: client.pid,
                        pidfd: Arc::clone(pidfd),
                        scope: scope.clone(),
                        generation: *generation,
                        claim: *claim,
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

        let claim = Claim(random_token().map_err(resource)?);
        self.deployments.insert(
            deployment.to_owned(),
            Deployment {
                claim,
                head: Some(id),
            },
        );
        let listed: BTreeSet<&Scope> = scopes.iter().collect();
        let released: Vec<Key> = self
            .tensors
            .iter_mut()
            .filter_map(|(key, tensor)| {
                tensor
                    .scopes
                    .retain(|scope| scope.deployment != deployment || listed.contains(scope));
                tensor.scopes.is_empty().then_some(*key)
            })
            .collect();
        for key in released {
            self.remove_tensor(&key);
        }
        tracing::info!(deployment, "a head claimed the deployment");
        Ok(Outcome::Reply(Reply::Claimed { claim }, Vec::new()))
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
        self.deployments.remove(deployment);
        let released: Vec<Key> = self
            .tensors
            .iter_mut()
            .filter_map(|(key, tensor)| {
                tensor.scopes.retain(|scope| scope.deployment != deployment);
                tensor.scopes.is_empty().then_some(*key)
            })
            .collect();
        let count = released.len() as u64;
        for key in released {
            self.remove_tensor(&key);
        }
        tracing::info!(deployment, released = count, "retired a deployment");
        Ok(Outcome::Reply(
            Reply::Retired { released: count },
            Vec::new(),
        ))
    }

    /// Whether `holder` is a worker of its deployment's current head.
    fn current(&self, holder: &Holder) -> bool {
        self.deployments
            .get(&holder.scope.deployment)
            .is_some_and(|entry| entry.claim == holder.claim)
    }

    /// `retain`: the listed keys that exist are held for the caller's
    /// scope, and every other tensor of the scope is released.
    fn retain(&mut self, holder: &Holder, keys: &[Key]) -> Result<Outcome, (ErrorKind, String)> {
        if !self.current(holder) {
            return Err(stale(holder));
        }
        let listed: BTreeSet<&Key> = keys.iter().collect();
        let mut released = Vec::new();
        for (key, tensor) in &mut self.tensors {
            if listed.contains(key) {
                tensor.scopes.insert(holder.scope.clone());
            } else if tensor.scopes.remove(&holder.scope) && tensor.scopes.is_empty() {
                released.push(*key);
            }
        }
        let count = released.len() as u64;
        for key in released {
            self.remove_tensor(&key);
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

    /// `get`: see [`Request::Get`].
    fn get(
        &mut self,
        holder: &Holder,
        keys: &[Key],
        mode: Mode,
    ) -> Result<Outcome, (ErrorKind, String)> {
        if mode != Mode::Read && !self.current(holder) {
            return Err(stale(holder));
        }
        let mut victims = Vec::new();
        for key in keys {
            let Some(tensor) = self.tensors.get(key) else {
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
        let mut missing = Vec::new();
        for key in keys {
            let Some(tensor) = self.tensors.get_mut(key) else {
                missing.push(*key);
                continue;
            };
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
            tensor.scopes.insert(holder.scope.clone());
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
            // A key is a content address: a read-only tensor already
            // retained holds these very bytes, and the put only adds the
            // caller's scope to it. Exclusive tensors are created once.
            if let Some(existing) = self.tensors.get(&tensor.key)
                && (existing.access, tensor.access) != (Access::ReadOnly, Access::ReadOnly)
            {
                return Err(invalid(&format!("{} is already retained", tensor.name)));
            }
            if tensor.storage as usize >= storages.len() {
                return Err(invalid(&format!("{} names no storage", tensor.name)));
            }
            if tensor.shape.len() != tensor.strides.len() {
                return Err(invalid(&format!("{} has one stride per axis", tensor.name)));
            }
        }
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
            if let Some(existing) = self.tensors.get_mut(&tensor.key) {
                existing.scopes.insert(holder.scope.clone());
                continue;
            }
            let storage = ids[tensor.storage as usize];
            if let Some(entry) = self.storages.get_mut(&storage) {
                entry.tensors += 1;
            }
            self.tensors.insert(
                tensor.key,
                TensorEntry {
                    name: tensor.name.clone(),
                    storage,
                    offset: tensor.offset,
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    strides: tensor.strides.clone(),
                    access: tensor.access,
                    scopes: BTreeSet::from([holder.scope.clone()]),
                    writer: (tensor.access == Access::Exclusive).then(|| holder.clone()),
                    standby: Vec::new(),
                },
            );
        }
        // A storage no tensor names is not retained.
        self.storages.retain(|_, entry| entry.tensors > 0);
        Ok(Outcome::Reply(Reply::Put, Vec::new()))
    }

    /// `promote`: see [`Request::Promote`].
    fn promote(&mut self, holder: &Holder, keys: &[Key]) -> Result<Outcome, (ErrorKind, String)> {
        if !self.current(holder) {
            return Err(stale(holder));
        }
        let mut victims = Vec::new();
        for key in keys {
            let tensor = self
                .tensors
                .get(key)
                .ok_or_else(|| invalid(&format!("{key:?} is not retained")))?;
            if !tensor
                .standby
                .iter()
                .any(|standby| Arc::ptr_eq(&standby.pidfd, &holder.pidfd))
            {
                return Err(invalid(&format!("{} is not held in standby", tensor.name)));
            }
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
        for key in keys {
            if let Some(tensor) = self.tensors.get_mut(key) {
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
            if let Some(tensor) = self.tensors.get_mut(key)
                && tensor
                    .writer
                    .as_ref()
                    .is_some_and(|writer| Arc::ptr_eq(&writer.pidfd, &holder.pidfd))
            {
                tensor.writer = None;
            }
        }
        Ok(Outcome::Reply(Reply::Released, Vec::new()))
    }

    /// Remove a tensor and release its storage with its last tensor.
    fn remove_tensor(&mut self, key: &Key) {
        let Some(tensor) = self.tensors.remove(key) else {
            return;
        };
        let empty = self.storages.get_mut(&tensor.storage).is_some_and(|entry| {
            entry.tensors -= 1;
            entry.tensors == 0
        });
        if empty {
            self.storages.remove(&tensor.storage);
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
        match &client.role {
            Some(Role::Head { .. }) => {
                let clients = &self.clients;
                self.pages
                    .head_gone(id, &|scope| connected_worker(clients, scope));
            }
            Some(Role::Worker { scope, .. }) if !connected_worker(&self.clients, scope) => {
                self.pages.worker_gone(scope);
            }
            Some(Role::Connector) if self.connector.client == Some(id) => self.connector_gone(),
            _ => {}
        }
        if let Some(Role::Head { deployment }) = &client.role
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
        let mut device = BTreeMap::<([u8; 16], Scope), u64>::new();
        let mut host = BTreeMap::<Scope, u64>::new();
        let mut counted = BTreeSet::<(StorageId, Scope)>::new();
        let mut tensors = Vec::with_capacity(self.tensors.len());
        for (key, tensor) in &self.tensors {
            let Some(storage) = self.storages.get(&tensor.storage) else {
                continue;
            };
            for scope in &tensor.scopes {
                if !counted.insert((tensor.storage, scope.clone())) {
                    continue;
                }
                match storage.record.placement {
                    Placement::Device { gpu } => {
                        *device.entry((gpu, scope.clone())).or_default() += storage.record.bytes;
                    }
                    Placement::Host { .. } => {
                        *host.entry(scope.clone()).or_default() += storage.record.bytes;
                    }
                }
            }
            tensors.push(ListedTensor {
                key: *key,
                name: tensor.name.clone(),
                scopes: tensor.scopes.iter().cloned().collect(),
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
                .map(|((gpu, scope), bytes)| ScopeBytes {
                    gpu: Some(gpu),
                    scope,
                    bytes,
                })
                .collect(),
            host_bytes: host
                .into_iter()
                .map(|(scope, bytes)| ScopeBytes {
                    gpu: None,
                    scope,
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
            connector: self.connector.client.is_some(),
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
        let Some(connector) = self.connector.client else {
            return;
        };
        for listed in self.pages.listing() {
            if self.connector.offered.insert(listed.pool) {
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
        let Some(connector) = self.connector.client else {
            self.fail(task);
            return;
        };
        let id = self.connector.next_task();
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
        self.connector.tasks.insert(id, task);
        self.events.push((connector, event, Vec::new()));
    }

    /// Write pages committed in `pool` to the distributed tier, in tasks of
    /// at most [`BATCH_PAGES`] pages, holding their slots until each task
    /// ends.
    fn upload(&mut self, pool: u64, pages: Vec<(Key, u32)>) {
        if self.connector.client.is_none() || pages.is_empty() {
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
    ) -> Result<(), (ErrorKind, String)> {
        let entry = pool_of(&mut self.pages, pool)?;
        let fetches = if self.connector.client.is_some() {
            entry.prefetch(keys)
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
            let results = keys.iter().map(|key| entry.holds(key)).collect();
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
        let Some(task) = self.connector.tasks.remove(&task) else {
            return;
        };
        self.complete(task, results);
    }

    /// End `task` as if the connector reported every page failed.
    fn fail(&mut self, task: Task) {
        self.complete(task, &[]);
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
                l2,
            } => {
                // A page counts in L3 when the connector found it or L2
                // holds it by now.
                let run = self.pages.get(pool).map_or(0, |entry| {
                    keys.iter()
                        .enumerate()
                        .take_while(|(index, key)| succeeded(*index) || entry.holds(key))
                        .count()
                });
                self.answer(
                    client,
                    request,
                    Reply::Lookup {
                        l2,
                        l3: l2 + run as u32,
                    },
                    &[],
                );
                if let Some(entry) = self.clients.get_mut(&client) {
                    entry.held = false;
                }
                self.serve_queue(client);
            }
            Task::Upload { pool, pages } => {
                let failed = (0..pages.len()).filter(|index| !succeeded(*index)).count();
                if failed > 0 {
                    tracing::warn!(
                        pool,
                        failed,
                        "pages were not written to the distributed tier"
                    );
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
                let Some(entry) = self.pages.get_mut(pool) else {
                    return;
                };
                let arrived: Vec<bool> = (0..pages.len()).map(succeeded).collect();
                entry.fetched(&pages, &arrived);
                let results = keys.iter().map(|key| entry.holds(key)).collect();
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
        self.connector.client = None;
        self.connector.offered.clear();
        let tasks: Vec<Task> = self.connector.tasks.drain().map(|(_, task)| task).collect();
        for task in tasks {
            self.fail(task);
        }
        self.pages.connector_gone();
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
        if let (Request::PagePool { shape, numa_node }, Role::Worker { .. }) = (request, role) {
            let pool = self.pages.pool(*shape, *numa_node).map_err(resource)?;
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
        let Role::Head { deployment } = role else {
            return Err(invalid(&format!(
                "{} is a request of a head",
                request_name(request)
            )));
        };
        let of_deployment = |scope: &Scope| {
            if &scope.deployment == deployment {
                Ok(())
            } else {
                Err(invalid(&format!("{scope} is not a scope of {deployment}")))
            }
        };
        match request {
            Request::Lookup { pool, keys } => {
                let l2 = pool_of(&mut self.pages, *pool)?.lookup(keys);
                if self.connector.client.is_none() || l2 as usize >= keys.len() {
                    // Without a connector, L3 holds nothing beyond L2.
                    return reply(Reply::Lookup { l2, l3: l2 });
                }
                // The connector looks up the pages after the L2 run; the
                // request waits for it.
                Ok(Outcome::Connector(Task::Lookup {
                    client: id,
                    request: 0,
                    pool: *pool,
                    keys: keys[l2 as usize..].to_vec(),
                    l2,
                }))
            }
            Request::Reserve { pool, writer, keys } => {
                of_deployment(writer)?;
                let slots = pool_of(&mut self.pages, *pool)?.reserve(writer, id, keys);
                reply(Reply::Reserved { slots })
            }
            Request::Commit { pool, keys } => {
                let committed = pool_of(&mut self.pages, *pool)?.commit(keys);
                self.upload(*pool, committed);
                reply(Reply::Committed)
            }
            Request::Abort { pool, keys } => {
                pool_of(&mut self.pages, *pool)?.abort(keys);
                reply(Reply::Aborted)
            }
            Request::Pin { pool, reader, keys } => {
                of_deployment(reader)?;
                let slots = pool_of(&mut self.pages, *pool)?.pin(reader, id, keys);
                reply(Reply::Pinned { slots })
            }
            Request::Unpin { pool, reader, keys } => {
                of_deployment(reader)?;
                pool_of(&mut self.pages, *pool)?.unpin(reader, keys);
                reply(Reply::Unpinned)
            }
            Request::Prefetch { pool, keys, tag } => {
                self.prefetch(id, *pool, keys, *tag)?;
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

fn request_name(request: &Request) -> &'static str {
    match request {
        Request::Claim { .. } => "claim",
        Request::Retain { .. } => "retain",
        Request::Get { .. } => "get",
        Request::Put { .. } => "put",
        Request::Promote { .. } => "promote",
        Request::Release { .. } => "release",
        Request::List => "list",
        Request::Retire { .. } => "retire",
        Request::PagePool { .. } => "page_pool",
        Request::Lookup { .. } => "lookup",
        Request::Reserve { .. } => "reserve",
        Request::Commit { .. } => "commit",
        Request::Abort { .. } => "abort",
        Request::Pin { .. } => "pin",
        Request::Unpin { .. } => "unpin",
        Request::Endpoint => "endpoint",
        Request::Prefetch { .. } => "prefetch",
        Request::Finished { .. } => "finished",
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
