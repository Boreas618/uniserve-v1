//! The two clients of a Storage Server.
//!
//! [`Client`] is the engine's: one connection that speaks every request of
//! the protocol. A worker calls it synchronously, at startup, close and
//! promotion ([`Client::call`] and the typed operations beside it); a
//! scheduler submits requests ([`Client::submit`]) and collects their
//! responses when the connection is readable ([`Client::pump`],
//! [`Client::take`]), from its step loop. The server's own processes (its
//! Transfer Server and connector), a peer host and an operator use the same
//! connection under their roles.
//!
//! [`ClaimClient`] is the daemon's: one deployment's claim at this host's
//! server, which lasts while the client lives and crosses a daemon's restart
//! in place with its connection ([`ClaimClient::handoff`]).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io;
use std::net::ToSocketAddrs;
use std::os::fd::{AsFd, AsRawFd as _, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::protocol::{
    ANSWER, Claim, Claimant, ClientMessage, Event, Hello, Item, Key, KvProgram, Lent, Link,
    Listing, Location, Mode, Priority, ProgramRecord, ProgramStart, Quota, Refusal, Region,
    Registration, Request, Response, Role, Scope, Segment, ServerMessage, Span, StorageRecord,
    TaskStatus, TensorRecord, Tensors, Throttle, UnitLayout, UnitRun, Usage, VERSION, Welcome,
};
use crate::queue::QueueDescriptors;
use crate::runtime::{Open, Opened};
use service::Connection;

/// A connection to a Storage Server.
pub struct Client {
    connection: Connection,
    /// Whether the server is another host's, whose answers the protocol bounds
    /// ([`Request::remote_deadline`]); a server of this host closes the connection when it ends.
    remote: bool,
    /// When each request in flight to another host's server must be answered.
    deadlines: HashMap<u64, Instant>,
    next: u64,
    /// Responses received and not yet taken, by request, each with the
    /// descriptors that came with it.
    responses: HashMap<u64, (Response, Vec<OwnedFd>)>,
    /// Events received and not yet taken, with their descriptors.
    events: VecDeque<(Event, Vec<OwnedFd>)>,
}

impl Client {
    /// Connect to the server at `socket` as `role`.
    pub fn connect(socket: &Path, role: Role) -> Result<Self> {
        Self::open(Connection::new(UnixStream::connect(socket)?)?, role, false)
    }

    /// Say hello as `role` on `connection`, a connection to this host's server that exists
    /// already: a child's channel, whose other end the server made and keeps as the child's own
    /// (its Mooncake connector's or its Transfer Server's).
    pub fn on_connection(connection: Connection, role: Role) -> Result<Self> {
        Self::open(connection, role, false)
    }

    /// Connect to the server of another host at `address` (`host:port`,
    /// its `Endpoint`) as [`Role::Peer`]: the Transfer Server of this host
    /// that READs storage there. Connecting, and every call after it
    /// ([`Client::call`]), fails once the protocol's bound of its request
    /// passes without an answer ([`Request::remote_deadline`]), so a host that
    /// stopped responding fails the caller's task instead of holding it.
    pub fn connect_peer(address: &str) -> Result<Self> {
        Self::connect_remote(address, Role::Peer)
    }

    /// Connect to the server of another host at `address` (`host:port`,
    /// its `Endpoint`) as `role`, as [`Self::connect_peer`] does. A Storage
    /// Server reaches another host's this way for its buffer pools' tasks: as
    /// a [`Role::Peer`] to hold host copies it reads, as a [`Role::Head`] of
    /// the deployment to submit a transfer into a rank of that host.
    pub fn connect_remote(address: &str, role: Role) -> Result<Self> {
        let mut refused = None;
        for candidate in address.to_socket_addrs()? {
            match std::net::TcpStream::connect_timeout(&candidate, ANSWER) {
                Ok(stream) => return Self::open(Connection::new(stream)?, role, true),
                Err(error) => refused = Some(error),
            }
        }
        Err(Error::Io(refused.unwrap_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{address} names no address"),
            )
        })))
    }

    /// Says hello as `role` in this build's [`VERSION`] and reads the
    /// server's welcome, within the protocol's answer bound for another host's server.
    ///
    /// # Errors
    ///
    /// [`Error::Version`] when the server speaks another version,
    /// [`Error::Refused`] when it refuses the hello otherwise.
    fn open(mut connection: Connection, role: Role, remote: bool) -> Result<Self> {
        connection.send(&ClientMessage::Hello(Hello {
            version: VERSION,
            role,
        }))?;
        let deadline = remote.then(|| Instant::now() + ANSWER);
        match receive(&mut connection, deadline)?.0 {
            ServerMessage::Welcome(Welcome::Accepted) => Ok(Self {
                connection,
                remote,
                deadlines: HashMap::new(),
                next: 1,
                responses: HashMap::new(),
                events: VecDeque::new(),
            }),
            ServerMessage::Welcome(Welcome::Refused {
                refusal: Refusal::Version { offered, spoken },
            }) => Err(Error::Version { offered, spoken }),
            ServerMessage::Welcome(Welcome::Refused { refusal }) => Err(Error::Refused(refusal)),
            other => Err(Error::Protocol(format!("{other:?} before a welcome"))),
        }
    }

    /// The connection's socket and the next request identity, for another
    /// process to go on with ([`Client::resume`]): the server knows the
    /// connection under the role it said hello as.
    fn handoff(self) -> Result<(OwnedFd, u64)> {
        if !self.responses.is_empty() || !self.events.is_empty() {
            return Err(Error::Protocol(
                "a connection with responses or events not taken cannot be passed on".into(),
            ));
        }
        let handed = self.connection.hand_over()?;
        // Only the socket crosses: bytes received and not yet taken would be lost.
        if !handed.received.is_empty() {
            return Err(Error::Protocol(
                "a connection with part of a message received cannot be passed on".into(),
            ));
        }
        Ok((handed.descriptor, self.next))
    }

    /// Goes on with a Unix connection on this host another process passed
    /// on.
    fn resume(descriptor: OwnedFd, next: u64) -> Result<Self> {
        Ok(Self {
            connection: Connection::new(UnixStream::from(descriptor))?,
            remote: false,
            deadlines: HashMap::new(),
            next,
            responses: HashMap::new(),
            events: VecDeque::new(),
        })
    }

    /// The connection's descriptor, readable when responses or events
    /// arrive.
    pub fn as_fd(&self) -> BorrowedFd<'_> {
        self.connection.as_fd()
    }

    /// Send `request` with `descriptors` and return its identity. The
    /// response is taken with [`Self::wait`] or [`Self::take`].
    pub fn submit(&mut self, request: &Request, descriptors: &[BorrowedFd<'_>]) -> Result<u64> {
        let id = self.next;
        self.next += 1;
        if self.remote {
            self.deadlines
                .insert(id, Instant::now() + request.remote_deadline());
        }
        self.connection.send_with(
            &ClientMessage::Request {
                id,
                request: request.clone(),
            },
            descriptors,
        )?;
        Ok(id)
    }

    /// Read what the server sent without blocking, and write what waits to be sent.
    pub fn pump(&mut self) -> Result<()> {
        self.route()
    }

    /// The response to request `id`, with the descriptors that came with
    /// it, when it has arrived.
    pub fn take(&mut self, id: u64) -> Option<(Response, Vec<OwnedFd>)> {
        let response = self.responses.remove(&id);
        if response.is_some() {
            self.deadlines.remove(&id);
        }
        response
    }

    /// The next event received, with the descriptors that came with it.
    pub fn next_event(&mut self) -> Option<(Event, Vec<OwnedFd>)> {
        self.events.pop_front()
    }

    /// Blocks until task `task` ended, `interrupt` (when given) becomes readable or reaches its
    /// end, or `deadline` (when given) passes; returns the bytes the task copied and how it
    /// ended. Its
    /// progress events are discarded; other tasks' events stay queued for their owners.
    ///
    /// # Errors
    ///
    /// Fails when the connection fails.
    pub fn wait_task(
        &mut self,
        task: u64,
        interrupt: Option<BorrowedFd<'_>>,
        deadline: Option<Instant>,
    ) -> Result<Waited<(u64, TaskStatus)>> {
        self.wait_event(interrupt, deadline, |events| {
            let mut ended = None;
            events.retain(|(event, _)| match event {
                Event::TaskProgress { task: found, .. } if *found == task => false,
                Event::TaskEnded {
                    task: found,
                    bytes,
                    status,
                } if *found == task => {
                    ended = Some((*bytes, status.clone()));
                    false
                }
                _ => true,
            });
            ended
        })
    }

    /// Blocks until one of the pages tasks `tasks` ([`Request::Pages`]) reported its results,
    /// `interrupt` (when given) becomes readable or reaches its end, or `deadline` (when given)
    /// passes; returns the task and its results, one per page.
    ///
    /// # Errors
    ///
    /// Fails when the connection fails.
    pub fn wait_pages(
        &mut self,
        tasks: &[u64],
        interrupt: Option<BorrowedFd<'_>>,
        deadline: Option<Instant>,
    ) -> Result<Waited<(u64, Vec<bool>)>> {
        self.wait_event(interrupt, deadline, |events| {
            let at = events.iter().position(|(event, _)| {
                matches!(event, Event::PagesMoved { task, .. } if tasks.contains(task))
            })?;
            match events.remove(at)?.0 {
                Event::PagesMoved { task, results } => Some((task, results)),
                _ => None,
            }
        })
    }

    /// Blocks until an event is received, `interrupt` (when given) becomes readable or reaches its
    /// end, or `deadline` (when given) passes; returns the earliest event not yet taken, with the
    /// descriptors that came with it.
    ///
    /// # Errors
    ///
    /// Fails when the connection fails.
    pub fn wait_next_event(
        &mut self,
        interrupt: Option<BorrowedFd<'_>>,
        deadline: Option<Instant>,
    ) -> Result<Waited<(Event, Vec<OwnedFd>)>> {
        self.wait_event(interrupt, deadline, VecDeque::pop_front)
    }

    /// Blocks until `found` takes an event from those received, `interrupt` becomes readable,
    /// or `deadline` passes.
    fn wait_event<T>(
        &mut self,
        interrupt: Option<BorrowedFd<'_>>,
        deadline: Option<Instant>,
        mut found: impl FnMut(&mut VecDeque<(Event, Vec<OwnedFd>)>) -> Option<T>,
    ) -> Result<Waited<T>> {
        loop {
            self.route()?;
            if let Some(event) = found(&mut self.events) {
                return Ok(Waited::Done(event));
            }
            let remaining = deadline.map_or(Duration::MAX, |deadline| {
                deadline.saturating_duration_since(Instant::now())
            });
            if remaining.is_zero() {
                return Ok(Waited::TimedOut);
            }
            let interrupted = match interrupt {
                Some(interrupt) => self.connection.wait_or(interrupt, remaining)?,
                None => self.connection.wait(remaining).map(|()| false)?,
            };
            if interrupted {
                return Ok(Waited::Interrupted);
            }
        }
    }

    /// Block until request `id` has its response: within the protocol's bound of the request
    /// on another host's server ([`Request::remote_deadline`]), unbounded on this host's.
    pub fn wait(&mut self, id: u64) -> Result<(Response, Vec<OwnedFd>)> {
        let deadline = self.deadlines.remove(&id);
        loop {
            if let Some(response) = self.responses.remove(&id) {
                return Ok(response);
            }
            let (message, descriptors) = receive(&mut self.connection, deadline)?;
            self.deliver(message, descriptors)?;
        }
    }

    /// Send `request` and block for its response, which may be the server's
    /// refusal ([`Response::Error`]).
    pub fn call(
        &mut self,
        request: &Request,
        descriptors: &[BorrowedFd<'_>],
    ) -> Result<(Response, Vec<OwnedFd>)> {
        let id = self.submit(request, descriptors)?;
        self.wait(id)
    }

    /// Route every complete message received; the server's close, once every message before it
    /// was routed, is an error.
    fn route(&mut self) -> Result<()> {
        while let Some((message, descriptors)) = self.connection.try_recv_with::<ServerMessage>()? {
            self.deliver(message, descriptors)?;
        }
        Ok(())
    }

    fn deliver(&mut self, message: ServerMessage, descriptors: Vec<OwnedFd>) -> Result<()> {
        match message {
            ServerMessage::Response { id, response } => {
                self.responses.insert(id, (response, descriptors));
            }
            ServerMessage::Event(event) => self.events.push_back((event, descriptors)),
            ServerMessage::Welcome(_) => {
                return Err(Error::Protocol("a second welcome".into()));
            }
        }
        Ok(())
    }

    /// Send `request` and block for its response; the server's refusal is
    /// the error.
    fn request(
        &mut self,
        request: &Request,
        descriptors: &[BorrowedFd<'_>],
    ) -> Result<(Response, Vec<OwnedFd>)> {
        match self.call(request, descriptors)? {
            (Response::Error { kind, message }, _) => Err(Error::Server { kind, message }),
            response => Ok(response),
        }
    }

    /// See [`Request::Claim`]: become the deployment's head, `claimant`
    /// placing `scopes` on the host for `epoch`, and end every epoch of
    /// `retire` there.
    pub fn claim(
        &mut self,
        claimant: Claimant,
        epoch: u64,
        scopes: Vec<Scope>,
        retire: Vec<u64>,
    ) -> Result<Claim> {
        let request = Request::Claim {
            claimant,
            scopes,
            epoch,
            retire,
        };
        match self.request(&request, &[])?.0 {
            Response::Claimed { claim } => Ok(claim),
            other => Err(unexpected("claim", &other)),
        }
    }

    /// See [`Request::Publish`]: make the claimed, committed epochs' tensors
    /// shared together.
    pub fn publish(&mut self, epochs: Vec<u64>) -> Result<()> {
        match self.request(&Request::Publish { epochs }, &[])?.0 {
            Response::Published => Ok(()),
            other => Err(unexpected("publish", &other)),
        }
    }

    /// See [`Request::Unclaim`]: end `epoch`'s retention of `scopes`. Returns how many tensors
    /// were released.
    pub fn unclaim(&mut self, epoch: u64, scopes: Vec<Scope>) -> Result<u64> {
        match self.request(&Request::Unclaim { epoch, scopes }, &[])?.0 {
            Response::Unclaimed { released } => Ok(released),
            other => Err(unexpected("unclaim", &other)),
        }
    }

    /// See [`Request::KeepWeights`]. Returns how many tensors were released.
    pub fn keep_weights(&mut self, epochs: Vec<u64>, cache_keys: Vec<Key>) -> Result<u64> {
        match self
            .request(&Request::KeepWeights { epochs, cache_keys }, &[])?
            .0
        {
            Response::Retained { released } => Ok(released),
            other => Err(unexpected("keep_weights", &other)),
        }
    }

    /// See [`Request::Offload`]. Returns how many storages moved to host
    /// storage and their bytes, once every copy ended.
    pub fn offload(&mut self, epoch: u64) -> Result<(u64, u64)> {
        match self.request(&Request::Offload { epoch }, &[])?.0 {
            Response::Moved { storages, bytes } => Ok((storages, bytes)),
            other => Err(unexpected("offload", &other)),
        }
    }

    /// See [`Request::Restore`]. Returns how many storages returned to
    /// their GPUs and their bytes, once every copy ended.
    pub fn restore(&mut self, epoch: u64) -> Result<(u64, u64)> {
        match self.request(&Request::Restore { epoch }, &[])?.0 {
            Response::Moved { storages, bytes } => Ok((storages, bytes)),
            other => Err(unexpected("restore", &other)),
        }
    }

    /// See [`Request::Retain`]: retain the copies of `keys` at the
    /// locations `at` (every copy when empty). Returns how many tensors
    /// were released.
    pub fn retain(&mut self, keys: Vec<Key>, at: Vec<Location>) -> Result<u64> {
        match self.request(&Request::Retain { keys, at }, &[])?.0 {
            Response::Retained { released } => Ok(released),
            other => Err(unexpected("retain", &other)),
        }
    }

    /// See [`Request::Get`]: the copies of `keys` at the locations `at`
    /// (every copy when empty), with one descriptor per storage of the
    /// response, in order.
    pub fn get(
        &mut self,
        keys: Vec<Key>,
        mode: Mode,
        at: Vec<Location>,
    ) -> Result<(Tensors, Vec<OwnedFd>)> {
        let response = self.request(&Request::Get { keys, mode, at }, &[])?;
        tensors(response)
    }

    /// See [`Request::AllocateHost`]: retained host storage the server
    /// admitted whole, as the tensor over it and its writable descriptor.
    pub fn allocate_host(
        &mut self,
        key: Key,
        name: String,
        bytes: u64,
    ) -> Result<(Tensors, Vec<OwnedFd>)> {
        let response = self.request(&Request::AllocateHost { key, name, bytes }, &[])?;
        tensors(response)
    }

    /// See [`Request::Put`]: publish `tensors` over `storages`, whose
    /// descriptors the server keeps.
    pub fn put(
        &mut self,
        tensors: Vec<TensorRecord>,
        storages: &[(StorageRecord, BorrowedFd<'_>)],
    ) -> Result<()> {
        let records = storages.iter().map(|(record, _)| *record).collect();
        let descriptors: Vec<BorrowedFd<'_>> =
            storages.iter().map(|(_, descriptor)| *descriptor).collect();
        let request = Request::Put {
            tensors,
            storages: records,
        };
        match self.request(&request, &descriptors)?.0 {
            Response::Put => Ok(()),
            other => Err(unexpected("put", &other)),
        }
    }

    /// See [`Request::Promote`].
    pub fn promote(&mut self, keys: Vec<Key>) -> Result<()> {
        match self.request(&Request::Promote { keys }, &[])?.0 {
            Response::Promoted => Ok(()),
            other => Err(unexpected("promote", &other)),
        }
    }

    /// See [`Request::Release`].
    pub fn release(&mut self, keys: Vec<Key>) -> Result<()> {
        match self.request(&Request::Release { keys }, &[])?.0 {
            Response::Released => Ok(()),
            other => Err(unexpected("release", &other)),
        }
    }

    /// See [`Request::Quota`]. Returns the quotas of `gpu` the caller
    /// reserves against; a refusal is [`ErrorKind::Quota`], whose message
    /// names every reservation of the quota it would overrun.
    ///
    /// [`ErrorKind::Quota`]: crate::protocol::ErrorKind::Quota
    pub fn reserve(
        &mut self,
        gpu: [u8; 16],
        total: u64,
        limit: Option<u64>,
        reservations: Vec<(Item, u64)>,
    ) -> Result<Vec<Quota>> {
        let request = Request::Quota {
            gpu,
            total,
            limit,
            reservations,
        };
        match self.request(&request, &[])?.0 {
            Response::Usage(usage) => Ok(usage.quotas),
            other => Err(unexpected("quota", &other)),
        }
    }

    /// See [`Request::Usage`]: the quotas of `gpus` (every GPU when empty),
    /// what the host admits of new host storage, and what the tasks of the
    /// client's deployment copied.
    pub fn usage(&mut self, gpus: Vec<[u8; 16]>) -> Result<Usage> {
        match self.request(&Request::Usage { gpus }, &[])?.0 {
            Response::Usage(usage) => Ok(usage),
            other => Err(unexpected("usage", &other)),
        }
    }

    /// See [`Request::Submit`]. Returns the task, whose progress and end
    /// arrive as events.
    pub fn copy(&mut self, spans: Vec<Span>, priority: Priority, update: u64) -> Result<u64> {
        let request = Request::Submit {
            spans,
            priority,
            update,
        };
        match self.request(&request, &[])?.0 {
            Response::Submitted { task } => Ok(task),
            other => Err(unexpected("submit", &other)),
        }
    }

    /// See [`Request::OpenSession`]: hold the Transfer Server's contexts on
    /// `gpus` between tasks until [`Client::close_session`] or the
    /// connection closes.
    pub fn open_session(&mut self, gpus: Vec<[u8; 16]>) -> Result<u64> {
        match self.request(&Request::OpenSession { gpus }, &[])?.0 {
            Response::SessionOpened { session } => Ok(session),
            other => Err(unexpected("open session", &other)),
        }
    }

    /// See [`Request::CloseSession`]; running tasks keep their own holds.
    pub fn close_session(&mut self, session: u64) -> Result<()> {
        match self.request(&Request::CloseSession { session }, &[])?.0 {
            Response::SessionClosed => Ok(()),
            other => Err(unexpected("close session", &other)),
        }
    }

    /// See [`Request::Cancel`]. Returns how many tasks were cancelled.
    pub fn cancel(&mut self, tasks: Vec<u64>, update: Option<u64>) -> Result<u64> {
        match self.request(&Request::Cancel { tasks, update }, &[])?.0 {
            Response::Cancelled { tasks } => Ok(tasks),
            other => Err(unexpected("cancel", &other)),
        }
    }

    /// See [`Request::Throttle`].
    pub fn throttle(&mut self, links: Vec<Link>, throttle: Throttle) -> Result<()> {
        match self.request(&Request::Throttle { links, throttle }, &[])?.0 {
            Response::Throttled => Ok(()),
            other => Err(unexpected("throttle", &other)),
        }
    }

    /// See [`Request::Expose`]. Returns the exposure, which
    /// [`Client::unexpose`] ends, the NIXL agent metadata of the server's
    /// Transfer Server, and each region's registered range, in order.
    pub fn expose(
        &mut self,
        deployment: String,
        regions: Vec<Region>,
    ) -> Result<(u64, Vec<u8>, Vec<Registration>)> {
        let request = Request::Expose {
            deployment,
            regions,
        };
        match self.request(&request, &[])?.0 {
            Response::Exposed {
                exposure,
                metadata,
                regions,
            } => Ok((exposure, metadata, regions)),
            other => Err(unexpected("expose", &other)),
        }
    }

    /// See [`Request::Unexpose`].
    pub fn unexpose(&mut self, exposure: u64) -> Result<()> {
        match self.request(&Request::Unexpose { exposure }, &[])?.0 {
            Response::Unexposed => Ok(()),
            other => Err(unexpected("unexpose", &other)),
        }
    }

    /// See [`Request::Status`]: the server's readiness record.
    pub fn status(&mut self) -> Result<crate::settings::Readiness> {
        match self.request(&Request::Status, &[])?.0 {
            Response::Status(readiness) => Ok(*readiness),
            other => Err(unexpected("status", &other)),
        }
    }

    /// Closes the connection and returns once the server ended its side of it, which it does
    /// once it acted on the close: what the connection held at the server (its role, its
    /// requests, its processes' write access while the process is gone) has ended, so the caller
    /// observes the server without this client. What arrives meanwhile is dropped.
    ///
    /// # Errors
    ///
    /// Fails when the connection fails otherwise than by the server's close.
    pub fn close(mut self) -> Result<()> {
        self.connection.flush()?;
        // Shutting the socket's sending side ends it at the server whatever other handles of the
        // socket exist, such as a forked child's before it executes.
        // SAFETY: shuts down the socket the connection owns, which stays open.
        if unsafe { libc::shutdown(self.connection.as_fd().as_raw_fd(), libc::SHUT_WR) } == -1 {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        loop {
            match self.connection.recv_with::<ServerMessage>(Duration::MAX) {
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// See [`Request::Stop`]: stops the server and waits until it ended, which it states by
    /// closing this connection last; `timeout` bounds the wait.
    ///
    /// # Errors
    ///
    /// Fails when the server refuses the stop, or has not ended within `timeout`.
    pub fn stop(mut self, timeout: Duration) -> Result<()> {
        match self.request(&Request::Stop, &[])?.0 {
            Response::Stopping => {}
            other => return Err(unexpected("stop", &other)),
        }
        // Nothing more is asked: the server's close of this connection is its end.
        match self.connection.recv::<ServerMessage>(timeout) {
            Ok(None) => Err(Error::Protocol(format!(
                "the server did not end within {timeout:?} of its stop"
            ))),
            Ok(Some(_)) => Err(Error::Protocol(
                "the server sent a message after its stop".to_owned(),
            )),
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    /// See [`Request::List`].
    pub fn list(&mut self) -> Result<Listing> {
        match self.request(&Request::List, &[])?.0 {
            Response::Listing(listing) => Ok(listing),
            other => Err(unexpected("list", &other)),
        }
    }

    /// See [`Request::Restart`]: the successor runs `image`, the descriptor of an executable
    /// opened for reading, whose `storage-server` command `arguments` select, and serves with
    /// `settings`, or with this server's when `None`. Returns the successor's process id.
    pub fn restart(
        &mut self,
        image: BorrowedFd<'_>,
        arguments: Vec<String>,
        settings: Option<crate::settings::Settings>,
    ) -> Result<u32> {
        let request = Request::Restart {
            arguments,
            settings: settings.map(Box::new),
        };
        match self.request(&request, &[image])?.0 {
            Response::Restarting { pid } => Ok(pid),
            other => Err(unexpected("restart", &other)),
        }
    }

    /// See [`Request::Configure`]: the running server serves with `settings` from now on.
    pub fn configure(&mut self, settings: crate::settings::ServerSettings) -> Result<()> {
        match self.request(&Request::Configure { settings }, &[])?.0 {
            Response::Configured => Ok(()),
            other => Err(unexpected("configure", &other)),
        }
    }

    /// See [`Request::Open`]: open the epoch's session with the buffer
    /// pool. Returns the answer and one queue pair's descriptors per
    /// scheduler, in the order the layout lists them, each for
    /// [`QueuePair::open`](crate::QueuePair::open) as the engine's end.
    pub fn open_pool(&mut self, open: Open) -> Result<(Opened, Vec<QueueDescriptors>)> {
        let schedulers = open.schedulers.len();
        match self.request(&Request::Open(open), &[])? {
            (Response::Opened(opened), descriptors) => {
                let count = descriptors.len();
                if count != schedulers * QueueDescriptors::COUNT
                    || opened.registers.len() != schedulers
                {
                    return Err(Error::Protocol(format!(
                        "{schedulers} schedulers were answered with {count} descriptors of queue \
                         pairs"
                    )));
                }
                let mut descriptors = descriptors.into_iter();
                let pairs = std::iter::from_fn(|| QueueDescriptors::take(&mut descriptors));
                Ok((opened, pairs.collect()))
            }
            (other, _) => Err(unexpected("open", &other)),
        }
    }

    /// See [`Request::Close`]: the epoch's workers have stopped.
    pub fn close_pool(&mut self) -> Result<()> {
        match self.request(&Request::Close, &[])?.0 {
            Response::Closed => Ok(()),
            other => Err(unexpected("close", &other)),
        }
    }

    /// See [`Request::Place`]: where this rank's part `part` (its index in
    /// its worker group) of the units of the device region `region` lies in
    /// the device memory `memory`.
    pub fn place(
        &mut self,
        region: String,
        memory: String,
        part: u32,
        runs: Vec<UnitRun>,
        layout: Option<UnitLayout>,
    ) -> Result<()> {
        let request = Request::Place {
            region,
            memory,
            part,
            runs,
            layout,
        };
        match self.request(&request, &[])?.0 {
            Response::Placed => Ok(()),
            other => Err(unexpected("place", &other)),
        }
    }

    /// See [`Request::PlaceRemote`]: where part `part` of the units of the
    /// device region `region` lies in the device memory `memory`, for its
    /// rank of the host whose Storage Server listens at `endpoint`.
    pub fn place_remote(
        &mut self,
        region: String,
        memory: String,
        part: u32,
        endpoint: String,
        runs: Vec<UnitRun>,
        layout: Option<UnitLayout>,
    ) -> Result<()> {
        let request = Request::PlaceRemote {
            region,
            memory,
            part,
            endpoint,
            runs,
            layout,
        };
        match self.request(&request, &[])?.0 {
            Response::Placed => Ok(()),
            other => Err(unexpected("place remote", &other)),
        }
    }

    /// See [`Request::Grant`]: the ranks created `segment` of the device
    /// region `region`, `units` units from `first`.
    pub fn grant(
        &mut self,
        region: String,
        segment: Segment,
        first: u32,
        units: u32,
    ) -> Result<()> {
        let request = Request::Grant {
            region,
            segment,
            first,
            units,
        };
        match self.request(&request, &[])?.0 {
            Response::Granted => Ok(()),
            other => Err(unexpected("grant", &other)),
        }
    }

    /// See [`Request::Withdraw`]: the ranks are to release `segment` of the
    /// memory space of the device region `region`.
    pub fn withdraw(&mut self, region: String, segment: Segment) -> Result<()> {
        match self.request(&Request::Withdraw { region, segment }, &[])?.0 {
            Response::Withdrawn => Ok(()),
            other => Err(unexpected("withdraw", &other)),
        }
    }

    /// See [`Request::HostPart`]: part `part` (the rank's index in its
    /// worker group) of the host region `region`, `bytes` of every unit.
    /// Returns the part's units and its memfd.
    pub fn host_part(
        &mut self,
        region: String,
        part: u32,
        bytes: u64,
        numa_node: Option<u32>,
    ) -> Result<(u32, OwnedFd)> {
        let request = Request::HostPart {
            region,
            part,
            bytes,
            numa_node,
        };
        match self.request(&request, &[])? {
            (Response::HostPart { units }, descriptors) => {
                let descriptor = descriptors.into_iter().next().ok_or_else(|| {
                    Error::Protocol("a host region's part arrived without its memfd".into())
                })?;
                Ok((units, descriptor))
            }
            (other, _) => Err(unexpected("host part", &other)),
        }
    }

    /// See [`Request::Borrow`]: the blocks the owner `owner` of the session `session` of
    /// `deployment`'s buffer pool lent, which this connection holds until it returns them.
    pub fn borrow(&mut self, deployment: String, session: u32, owner: u64) -> Result<Lent> {
        let request = Request::Borrow {
            deployment,
            session,
            owner,
        };
        match self.request(&request, &[])?.0 {
            Response::Borrowed { lent } => Ok(lent),
            other => Err(unexpected("borrow", &other)),
        }
    }

    /// See [`Request::Return`]: the import of the lend of `session`'s owner `owner` ended.
    pub fn give_back(&mut self, deployment: String, session: u32, owner: u64) -> Result<()> {
        let request = Request::Return {
            deployment,
            session,
            owner,
        };
        match self.request(&request, &[])?.0 {
            Response::Returned => Ok(()),
            other => Err(unexpected("return", &other)),
        }
    }

    /// See [`Request::Pages`]: have the connector of this connection's host
    /// move `pages` of part `part` of the host region `region`, to the KV
    /// storage when `upload` and from it otherwise. Returns the task, whose
    /// results arrive as an event ([`Client::wait_pages`]).
    pub fn pages(
        &mut self,
        region: String,
        part: u32,
        pages: Vec<(Key, u32)>,
        upload: bool,
    ) -> Result<u64> {
        let request = Request::Pages {
            region,
            part,
            pages,
            upload,
        };
        match self.request(&request, &[])?.0 {
            Response::Submitted { task } => Ok(task),
            other => Err(unexpected("pages", &other)),
        }
    }

    /// See [`Request::Program`]: run `program` as the deployment's KV program on this server, or
    /// none. Returns how its instance started, `None` when none runs.
    pub fn program(&mut self, program: Option<KvProgram>) -> Result<Option<ProgramStart>> {
        match self.request(&Request::Program { program }, &[])?.0 {
            Response::Programmed { start } => Ok(start),
            other => Err(unexpected("program", &other)),
        }
    }

    /// See [`Request::Handoff`]: how the full pages of `handoff` reach the decode replica, as
    /// the deployment's KV program, or the default rule, decides.
    pub fn handoff_path(
        &mut self,
        handoff: scheduling::kv::Handoff,
    ) -> Result<scheduling::kv::Path> {
        match self.request(&Request::Handoff { handoff }, &[])?.0 {
            Response::Handoff { path } => Ok(path),
            other => Err(unexpected("handoff", &other)),
        }
    }

    /// See [`Request::ProgramRecords`]: what the deployment's KV program logged and how its
    /// outputs fared since the last call, oldest first.
    pub fn program_records(&mut self) -> Result<Vec<ProgramRecord>> {
        match self.request(&Request::ProgramRecords, &[])?.0 {
            Response::ProgramRecords { records } => Ok(records),
            other => Err(unexpected("program_records", &other)),
        }
    }

    /// See [`Request::Spans`]: the spans the server keeps in its rings.
    pub fn spans(&mut self) -> Result<Vec<service::span::Span>> {
        match self.request(&Request::Spans, &[])?.0 {
            Response::Spans { spans } => Ok(spans),
            other => Err(unexpected("spans", &other)),
        }
    }
}

/// The tensors of a response to a `Get`, with one descriptor per storage.
fn tensors((response, descriptors): (Response, Vec<OwnedFd>)) -> Result<(Tensors, Vec<OwnedFd>)> {
    match response {
        Response::Tensors(tensors) => {
            if tensors.storages.len() != descriptors.len() {
                return Err(Error::Protocol(format!(
                    "{} storages arrived with {} descriptors",
                    tensors.storages.len(),
                    descriptors.len()
                )));
            }
            Ok((tensors, descriptors))
        }
        other => Err(unexpected("get", &other)),
    }
}

fn unexpected(request: &str, response: &Response) -> Error {
    Error::Protocol(format!("{request} was responded with {response:?}"))
}

/// One deployment's claim at this host's Storage Server: the daemon's
/// client.
///
/// The claim lasts as long as the client: the claiming connection is the
/// deployment's head, so the claim outlives every engine and every epoch. An
/// engine that succeeds another claims the scopes of its own epoch under the
/// same claim, and its workers state the same token as the serving epoch's,
/// so neither epoch's storage is touched by the other's claim. Each engine
/// of an epoch claims as its own [`Claimant`], so one engine's claim never
/// ends the retention of another's storage.
pub struct ClaimClient {
    client: Client,
    deployment: String,
    /// The claim's token, once the first epoch claimed.
    claim: Option<Claim>,
    /// The scopes each claimant of each epoch claimed last, sorted, for the
    /// epochs not retired.
    claimed: BTreeMap<(u64, Claimant), Vec<Scope>>,
}

/// A [`ClaimClient`]'s state as it crosses to the daemon's successor in a
/// restart in place; the claiming connection's descriptor travels beside it,
/// so the Storage Server sees the same head throughout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimHandoff {
    pub deployment: String,
    pub claim: Option<Claim>,
    /// The scopes each claimant of each epoch claimed last, as a list,
    /// since the map's key is no string.
    pub claimed: Vec<((u64, Claimant), Vec<Scope>)>,
    /// The connection's next request identity.
    pub next: u64,
}

impl ClaimClient {
    /// Connect to the server at `socket` as the head of `deployment`. The
    /// connection holds every epoch of the claim, and reads and publishes
    /// nothing of its own.
    pub fn connect(socket: &Path, deployment: &str) -> Result<Self> {
        let role = Role::Head {
            deployment: deployment.to_owned(),
            epoch: 0,
            generation: 0,
        };
        Ok(Self {
            client: Client::connect(socket, role)?,
            deployment: deployment.to_owned(),
            claim: None,
            claimed: BTreeMap::new(),
        })
    }

    /// Claim the deployment for `epoch` as `claimant`, which places `scopes`
    /// on the host in place of the scopes it claimed before, and end every
    /// epoch of `retire` there ([`Request::Claim`]). The engines of one epoch
    /// on a host each claim the scopes they place, each as its own claimant:
    /// the independent replicas of the epoch's configuration, and a process
    /// that replaces one of them or stands by for it. The server retains the
    /// epoch's storage of every scope some claimant names, so one engine's
    /// claim never ends the retention of another's. A claimant that claims
    /// the scopes it claimed already, retiring nothing, keeps the claim held
    /// without asking the server. Returns the claim's token, which the
    /// deployment's workers state.
    ///
    /// # Errors
    ///
    /// Fails when the server cannot be reached, or refuses the claim
    /// because another head of the deployment is connected.
    pub fn claim(
        &mut self,
        claimant: Claimant,
        epoch: u64,
        mut scopes: Vec<Scope>,
        retire: Vec<u64>,
    ) -> Result<Claim> {
        scopes.sort();
        scopes.dedup();
        if let Some(claim) = self.claim
            && retire.is_empty()
            && self.claimed.get(&(epoch, claimant)) == Some(&scopes)
        {
            return Ok(claim);
        }

        let claim = self
            .client
            .claim(claimant, epoch, scopes.clone(), retire.clone())?;
        self.claimed
            .retain(|(claimed, _), _| !retire.contains(claimed));
        self.claimed.insert((epoch, claimant), scopes);
        self.claim = Some(claim);
        Ok(claim)
    }

    /// The deployment claimed.
    pub fn deployment(&self) -> &str {
        &self.deployment
    }

    /// See [`Request::Publish`]: publish the committed `epochs` together.
    pub fn publish(&mut self, epochs: Vec<u64>) -> Result<()> {
        self.client.publish(epochs)
    }

    /// See [`Request::Unclaim`]: end `epoch`'s retention of `scopes`, which no claim of this
    /// client names from now on. Returns how many tensors were released.
    pub fn unclaim(&mut self, epoch: u64, mut scopes: Vec<Scope>) -> Result<u64> {
        scopes.sort();
        scopes.dedup();
        for ((claimed, _), named) in &mut self.claimed {
            if *claimed == epoch {
                named.retain(|scope| !scopes.contains(scope));
            }
        }
        self.client.unclaim(epoch, scopes)
    }

    /// See [`Request::KeepWeights`]: `epochs` keep their weights and the
    /// rollback copies of `cache_keys` alone. Returns how many tensors were
    /// released.
    pub fn keep_weights(&mut self, epochs: Vec<u64>, cache_keys: Vec<Key>) -> Result<u64> {
        self.client.keep_weights(epochs, cache_keys)
    }

    /// See [`Request::Offload`]: move `epoch`'s weights to host storage.
    /// Returns how many storages moved and their bytes.
    pub fn offload(&mut self, epoch: u64) -> Result<(u64, u64)> {
        self.client.offload(epoch)
    }

    /// See [`Request::Restore`]: return `epoch`'s weights that an offload
    /// left in host storage to the GPUs they left; nothing moves when there
    /// are none. Returns how many storages moved and their bytes.
    pub fn restore(&mut self, epoch: u64) -> Result<(u64, u64)> {
        self.client.restore(epoch)
    }

    /// Halve or restore the byte budgets of the links the deployment's
    /// tasks use ([`Request::Throttle`] with no links): a halving takes
    /// every link a task of the deployment copies over now, a restoration
    /// every link a halving reduced.
    pub fn throttle(&mut self, throttle: Throttle) -> Result<()> {
        self.client.throttle(Vec::new(), throttle)
    }

    /// The deployment's quotas of every GPU of the host, what the host
    /// admits of new host storage and what the deployment's tasks copied
    /// ([`Request::Usage`]).
    pub fn usage(&mut self) -> Result<Usage> {
        self.client.usage(Vec::new())
    }

    /// The server's readiness record ([`Request::Status`]): its endpoints, among them the TCP
    /// endpoint other hosts reach it at when it accepts them, and its settings.
    pub fn status(&mut self) -> Result<crate::settings::Readiness> {
        self.client.status()
    }

    /// Runs `program` as the deployment's KV program on this server, or none
    /// ([`Client::program`]).
    pub fn program(&mut self, program: Option<KvProgram>) -> Result<Option<ProgramStart>> {
        self.client.program(program)
    }

    /// What the deployment's KV program logged and how its outputs fared since the last call
    /// ([`Client::program_records`]).
    pub fn program_records(&mut self) -> Result<Vec<ProgramRecord>> {
        self.client.program_records()
    }

    /// The spans the server keeps in its rings ([`Client::spans`]).
    pub fn spans(&mut self) -> Result<Vec<service::span::Span>> {
        self.client.spans()
    }

    /// Passes the claim to another process ([`ClaimClient::resume`]): its
    /// state and the claiming connection's descriptor.
    ///
    /// # Errors
    ///
    /// Fails when the connection has unread messages.
    pub fn handoff(self) -> Result<(ClaimHandoff, OwnedFd)> {
        let (descriptor, next) = self.client.handoff()?;
        let handoff = ClaimHandoff {
            deployment: self.deployment,
            claim: self.claim,
            claimed: self.claimed.into_iter().collect(),
            next,
        };
        Ok((handoff, descriptor))
    }

    /// Goes on with a claim another process passed on, over its
    /// connection's descriptor.
    ///
    /// # Errors
    ///
    /// Fails when the connection cannot be configured.
    pub fn resume(handoff: ClaimHandoff, descriptor: OwnedFd) -> Result<Self> {
        Ok(Self {
            client: Client::resume(descriptor, handoff.next)?,
            deployment: handoff.deployment,
            claim: handoff.claim,
            claimed: handoff.claimed.into_iter().collect(),
        })
    }
}

/// How a wait for an event ended ([`Client::wait_task`], [`Client::wait_pages`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Waited<T> {
    /// The event arrived.
    Done(T),
    /// The waiter's owner interrupted the wait.
    Interrupted,
    /// The deadline passed first.
    TimedOut,
}

/// The next message from the server, waiting until `deadline` (unbounded for `None`).
///
/// # Errors
///
/// `TimedOut` when none arrives in time, and the connection's failure, `UnexpectedEof` once the
/// server closed it.
fn receive(
    connection: &mut Connection,
    deadline: Option<Instant>,
) -> Result<(ServerMessage, Vec<OwnedFd>)> {
    let timeout = deadline.map_or(Duration::MAX, |deadline| {
        deadline.saturating_duration_since(Instant::now())
    });
    connection
        .recv_with::<ServerMessage>(timeout)?
        .ok_or_else(|| {
            Error::Io(io::Error::new(
                io::ErrorKind::TimedOut,
                "the storage server of another host did not answer within the protocol's bound",
            ))
        })
}
