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
//! One thread serves every client with `poll(2)`. Requests are answered in
//! order on their connection. A request that must wait for a previous
//! writer to be shot down ([`shoot`]) holds its connection's queue until the
//! shoot-down ends, while every other connection is served on.

mod nvml;
mod shoot;

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::{self, Read};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
use std::time::Duration;

use rustix::event::{PollFd, PollFlags, poll};

use crate::protocol::{
    Access, Claim, ClientMessage, ErrorKind, Hello, Key, ListedDeployment, ListedTensor, Listing,
    Mode, Placement, Reply, Request, Role, Scope, ScopeBytes, ServerMessage, StorageRecord,
    TensorRecord, VERSION, Welcome,
};
use crate::transport::Connection;
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
    pid: i32,
    pidfd: Arc<OwnedFd>,
    role: Option<Role>,
    /// Requests received and not yet answered, in arrival order.
    queue: VecDeque<(u64, Request, Vec<OwnedFd>)>,
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
    Reply(Reply, Vec<StorageId>),
    /// The request waits for these processes to be shot down.
    Wait(Vec<Holder>),
}

/// The server's state and its listening socket.
pub struct Server {
    listener: UnixListener,
    clients: BTreeMap<ClientId, Client>,
    next_client: ClientId,
    storages: HashMap<StorageId, StorageEntry>,
    next_storage: StorageId,
    tensors: HashMap<Key, TensorEntry>,
    deployments: HashMap<String, Deployment>,
    shootdowns: ShootDowns,
}

impl Server {
    /// Serve on `listener`, which the caller bound with mode 0600.
    pub fn new(listener: UnixListener) -> io::Result<Self> {
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            clients: BTreeMap::new(),
            next_client: 1,
            storages: HashMap::new(),
            next_storage: 1,
            tensors: HashMap::new(),
            deployments: HashMap::new(),
            shootdowns: ShootDowns::default(),
        })
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
        let mut fds = Vec::with_capacity(2 + ids.len() + pidfds.len());
        fds.push(PollFd::new(&self.listener, PollFlags::IN));
        fds.push(PollFd::new(&wake, PollFlags::IN));
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
        let timeout = self
            .shootdowns
            .polling()
            .then(|| rustix::time::Timespec::try_from(NVML_POLL).unwrap_or_default());
        match poll(&mut fds, timeout.as_ref()) {
            Ok(_) | Err(rustix::io::Errno::INTR) => {}
            Err(error) => return Err(error.into()),
        }
        let accept = !fds[0].revents().is_empty();
        if !fds[1].revents().is_empty() {
            drain(wake);
        }
        let ready: Vec<ClientId> = ids
            .iter()
            .zip(&fds[2..2 + ids.len()])
            .filter(|(_, fd)| !fd.revents().is_empty())
            .map(|(id, _)| *id)
            .collect();
        drop(fds);

        if accept {
            self.accept();
        }
        for id in ready {
            self.read(id);
        }
        if self.shootdowns.advance() {
            self.settle_shootdowns();
        }
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
            pidfd: Arc::new(pidfd),
            role: None,
            queue: VecDeque::new(),
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
            let Some((request_id, request, descriptors)) = client.queue.pop_front() else {
                return;
            };
            match self.handle(id, &request, descriptors) {
                Ok(Outcome::Reply(reply, storages)) => {
                    self.answer(id, request_id, reply, &storages)
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
    fn answer(&mut self, id: ClientId, request: u64, reply: Reply, storages: &[StorageId]) {
        let descriptors: Vec<BorrowedFd<'_>> = storages
            .iter()
            .filter_map(|storage| self.storages.get(storage))
            .map(|entry| entry._descriptor.as_fd())
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
                let holder = Holder {
                    client: Some(id),
                    pid: client.pid,
                    pidfd: Arc::clone(&client.pidfd),
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
                    Request::List | Request::Claim { .. } | Request::Retire { .. } => {
                        Err(invalid("a worker cannot claim or retire a deployment"))
                    }
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
                .filter_map(|(other, client)| match &client.role {
                    Some(Role::Worker {
                        scope,
                        generation,
                        claim,
                    }) if scope.deployment == deployment => Some(Holder {
                        client: Some(*other),
                        pid: client.pid,
                        pidfd: Arc::clone(&client.pidfd),
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
            storages,
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
        }
    }
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
    }
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
