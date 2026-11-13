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
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::protocol::{
    Claim, ClientMessage, Event, Hello, HostCopy, Item, Key, Link, Listing, Location, Mode,
    Priority, Quota, Region, RegionPart, Registration, Request, Response, Role, Scope, Segment,
    ServerMessage, Span, StorageRecord, TaskStatus, TensorRecord, Tensors, Throttle, UnitRun,
    Usage, VERSION, Welcome,
};
use crate::runtime::{Open, Opened};
use service::Connection;

/// A connection to a Storage Server.
pub struct Client {
    connection: Connection,
    /// How long a blocking call waits for its response: unbounded on this host, the timeout
    /// given for another host's server ([`Client::connect_remote`]).
    timeout: Duration,
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
        Self::open(
            Connection::new(UnixStream::connect(socket)?)?,
            role,
            Duration::MAX,
        )
    }

    /// Connect to the server of another host at `address` (`host:port`,
    /// its `Endpoint`) as [`Role::Peer`]: the Transfer Server of this host
    /// that READs storage there. Connecting, and every blocking
    /// call after it ([`Client::call`]), fails once `timeout` passes without
    /// a response, so a host that stopped responding fails the caller's task
    /// instead of holding it.
    pub fn connect_peer(address: &str, timeout: Duration) -> Result<Self> {
        Self::connect_remote(address, Role::Peer, timeout)
    }

    /// Connect to the server of another host at `address` (`host:port`,
    /// its `Endpoint`) as `role`, as [`Self::connect_peer`] does: connecting
    /// and every blocking call fail once `timeout` passes without a
    /// response. A Storage Server reaches another host's this way for its
    /// buffer pools' tasks: as a [`Role::Peer`] to hold host copies it
    /// reads, as a [`Role::Head`] of the deployment to submit a transfer
    /// into a rank of that host.
    pub fn connect_remote(address: &str, role: Role, timeout: Duration) -> Result<Self> {
        let mut refused = None;
        for candidate in address.to_socket_addrs()? {
            match std::net::TcpStream::connect_timeout(&candidate, timeout) {
                Ok(stream) => return Self::open(Connection::new(stream)?, role, timeout),
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

    fn open(mut connection: Connection, role: Role, timeout: Duration) -> Result<Self> {
        connection.send(&ClientMessage::Hello(Hello {
            version: VERSION,
            role,
        }))?;
        match receive(&mut connection, timeout)?.0 {
            ServerMessage::Welcome(Welcome::Accepted { .. }) => Ok(Self {
                connection,
                timeout,
                next: 1,
                responses: HashMap::new(),
                events: VecDeque::new(),
            }),
            ServerMessage::Welcome(Welcome::Refused { message }) => Err(Error::Refused(message)),
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

    /// Goes on with a Unix connection on this host another process passed on.
    fn resume(descriptor: OwnedFd, next: u64) -> Result<Self> {
        Ok(Self {
            connection: Connection::new(UnixStream::from(descriptor))?,
            timeout: Duration::MAX,
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
        self.responses.remove(&id)
    }

    /// The next event received, with the descriptors that came with it.
    pub fn next_event(&mut self) -> Option<(Event, Vec<OwnedFd>)> {
        self.events.pop_front()
    }

    /// Consume one task's end without waiting, discarding its progress
    /// events: the bytes it copied and how it ended. Other tasks' events
    /// stay queued for their owners. Returns `None` while the task runs or
    /// after its end was consumed.
    pub fn poll_task(&mut self, task: u64) -> Result<Option<(u64, TaskStatus)>> {
        self.pump()?;
        let mut ended = None;
        self.events.retain(|(event, _)| match event {
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
        Ok(ended)
    }

    /// Block until request `id` has its response.
    pub fn wait(&mut self, id: u64) -> Result<(Response, Vec<OwnedFd>)> {
        loop {
            if let Some(response) = self.responses.remove(&id) {
                return Ok(response);
            }
            let (message, descriptors) = receive(&mut self.connection, self.timeout)?;
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

    /// See [`Request::Claim`]: become the deployment's head, `epoch`
    /// placing `scopes` on the host, and end every epoch of `retire` there.
    pub fn claim(&mut self, epoch: u64, scopes: Vec<Scope>, retire: Vec<u64>) -> Result<Claim> {
        let request = Request::Claim {
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

    /// See [`Request::Rekey`]: move the copies at the locations `at`
    /// (every copy when empty).
    pub fn rekey(&mut self, moves: Vec<(Key, Key)>, at: Vec<Location>) -> Result<()> {
        match self.request(&Request::Rekey { moves, at }, &[])?.0 {
            Response::Rekeyed => Ok(()),
            other => Err(unexpected("rekey", &other)),
        }
    }

    /// See [`Request::Read`]: the bytes of `deployment`'s read-only host
    /// tensor `key`.
    pub fn read(&mut self, deployment: String, key: Key) -> Result<Vec<u8>> {
        match self.request(&Request::Read { deployment, key }, &[])?.0 {
            Response::Read { bytes } => Ok(bytes),
            other => Err(unexpected("read", &other)),
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

    /// See [`Request::Endpoint`]: the `host:port` other hosts reach the
    /// server at; `None` when it accepts local connections only.
    pub fn endpoint(&mut self) -> Result<Option<String>> {
        match self.request(&Request::Endpoint, &[])?.0 {
            Response::Endpoint { tcp } => Ok(tcp),
            other => Err(unexpected("endpoint", &other)),
        }
    }

    /// See [`Request::List`].
    pub fn list(&mut self) -> Result<Listing> {
        match self.request(&Request::List, &[])?.0 {
            Response::Listing(listing) => Ok(listing),
            other => Err(unexpected("list", &other)),
        }
    }

    /// See [`Request::Restart`]. Returns the successor's process id.
    pub fn restart(&mut self, executable: Option<String>) -> Result<u32> {
        match self.request(&Request::Restart { executable }, &[])?.0 {
            Response::Restarting { pid } => Ok(pid),
            other => Err(unexpected("restart", &other)),
        }
    }

    /// See [`Request::Retire`]. Returns how many tensors were released.
    pub fn retire(&mut self, deployment: String) -> Result<u64> {
        match self.request(&Request::Retire { deployment }, &[])?.0 {
            Response::Retired { released } => Ok(released),
            other => Err(unexpected("retire", &other)),
        }
    }

    /// See [`Request::Open`]: open the epoch's session with the buffer
    /// pool. Returns the answer and one queue pair's memfd per scheduler,
    /// in the order the layout lists them, each for
    /// [`QueuePair::open`](crate::QueuePair::open) as the engine's end.
    pub fn open_pool(&mut self, open: Open) -> Result<(Opened, Vec<OwnedFd>)> {
        let schedulers = open.schedulers.len();
        match self.request(&Request::Open(open), &[])? {
            (Response::Opened(opened), descriptors) => {
                if descriptors.len() != schedulers || opened.registers.len() != schedulers {
                    return Err(Error::Protocol(format!(
                        "{schedulers} schedulers were answered with {} queue pairs",
                        descriptors.len()
                    )));
                }
                Ok((opened, descriptors))
            }
            (other, _) => Err(unexpected("open", &other)),
        }
    }

    /// See [`Request::Close`]: the epoch's executors have stopped.
    pub fn close_pool(&mut self) -> Result<()> {
        match self.request(&Request::Close, &[])?.0 {
            Response::Closed => Ok(()),
            other => Err(unexpected("close", &other)),
        }
    }

    /// See [`Request::Place`]: where this rank's part of the units of the
    /// device region `region` lies in the device memory `memory`.
    pub fn place(&mut self, region: String, memory: String, runs: Vec<UnitRun>) -> Result<()> {
        let request = Request::Place {
            region,
            memory,
            runs,
        };
        match self.request(&request, &[])?.0 {
            Response::Placed => Ok(()),
            other => Err(unexpected("place", &other)),
        }
    }

    /// See [`Request::PlaceRemote`]: where the part of the units of the
    /// device region `region` lies for rank `rank` of the device memory
    /// `memory`, a rank of the host whose Storage Server listens at
    /// `endpoint`.
    pub fn place_remote(
        &mut self,
        region: String,
        memory: String,
        rank: u32,
        endpoint: String,
        runs: Vec<UnitRun>,
    ) -> Result<()> {
        let request = Request::PlaceRemote {
            region,
            memory,
            rank,
            endpoint,
            runs,
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

    /// See [`Request::HostPart`]: this rank's part of the host region
    /// `region`, `bytes` of every unit. Returns the part's units and its
    /// memfd.
    pub fn host_part(
        &mut self,
        region: String,
        bytes: u64,
        numa_node: Option<u32>,
    ) -> Result<(u32, OwnedFd)> {
        let request = Request::HostPart {
            region,
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

    /// See [`Request::PeerHold`]: hold the host copies of `keys` in
    /// `deployment`'s buffer pool for this connection, a [`Role::Peer`].
    /// Returns the copies held and the parts of their host regions.
    pub fn peer_hold(
        &mut self,
        deployment: String,
        keys: Vec<Key>,
    ) -> Result<(Vec<HostCopy>, Vec<RegionPart>)> {
        match self
            .request(&Request::PeerHold { deployment, keys }, &[])?
            .0
        {
            Response::PeerHeld { copies, parts } => Ok((copies, parts)),
            other => Err(unexpected("peer hold", &other)),
        }
    }

    /// See [`Request::PeerRelease`]: end this connection's holds in
    /// `deployment`'s buffer pool. Returns whether they stood until now
    /// ([`Response::Valid`]), so what was read under them is valid.
    pub fn peer_release(&mut self, deployment: String) -> Result<bool> {
        match self.request(&Request::PeerRelease { deployment }, &[])?.0 {
            Response::Valid => Ok(true),
            Response::Gone => Ok(false),
            other => Err(unexpected("peer release", &other)),
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
/// so neither epoch's storage is touched by the other's claim.
pub struct ClaimClient {
    client: Client,
    deployment: String,
    /// The claim's token, once the first epoch claimed.
    claim: Option<Claim>,
    /// The scopes each epoch claimed and did not retire, sorted.
    epochs: BTreeMap<u64, Vec<Scope>>,
}

/// A [`ClaimClient`]'s state as it crosses to the daemon's successor in a
/// restart in place; the claiming connection's descriptor travels beside it,
/// so the Storage Server sees the same head throughout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimHandoff {
    pub deployment: String,
    pub claim: Option<Claim>,
    pub epochs: BTreeMap<u64, Vec<Scope>>,
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
        };
        Ok(Self {
            client: Client::connect(socket, role)?,
            deployment: deployment.to_owned(),
            claim: None,
            epochs: BTreeMap::new(),
        })
    }

    /// Claim the deployment for `epoch`, which places `scopes` on the host,
    /// and end every epoch of `retire` there ([`Request::Claim`]). An epoch
    /// that claimed the same scopes and retires nothing keeps the claim
    /// held, without asking the server. Returns the claim's token, which
    /// the deployment's workers state.
    ///
    /// # Errors
    ///
    /// Fails when the server cannot be reached, or refuses the claim
    /// because another head of the deployment is connected.
    pub fn claim(&mut self, epoch: u64, mut scopes: Vec<Scope>, retire: Vec<u64>) -> Result<Claim> {
        scopes.sort();
        if let Some(claim) = self.claim
            && retire.is_empty()
            && self.epochs.get(&epoch) == Some(&scopes)
        {
            return Ok(claim);
        }

        let claim = self.client.claim(epoch, scopes.clone(), retire.clone())?;
        for retired in &retire {
            self.epochs.remove(retired);
        }
        self.epochs.insert(epoch, scopes);
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

    /// The `host:port` other hosts reach the server at; `None` when it
    /// accepts local connections only.
    pub fn endpoint(&mut self) -> Result<Option<String>> {
        self.client.endpoint()
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
            epochs: self.epochs,
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
            epochs: handoff.epochs,
        })
    }
}

/// The next message from the server, waiting up to `timeout`.
///
/// # Errors
///
/// `TimedOut` when none arrives in time, and the connection's failure, `UnexpectedEof` once the
/// server closed it.
fn receive(
    connection: &mut Connection,
    timeout: Duration,
) -> Result<(ServerMessage, Vec<OwnedFd>)> {
    connection
        .recv_with::<ServerMessage>(timeout)?
        .ok_or_else(|| {
            Error::Io(io::Error::new(
                io::ErrorKind::TimedOut,
                "the storage server did not respond within the connection's timeout",
            ))
        })
}
