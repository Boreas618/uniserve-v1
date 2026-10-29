//! A connection to a Storage Server.
//!
//! One [`Client`] serves both uses: a worker calls it synchronously, at
//! startup, close and promotion ([`Client::call`] and the typed requests
//! beside it); the head submits requests ([`Client::submit`]) and collects
//! their answers when the connection is readable ([`Client::pump`],
//! [`Client::take`]), from its scheduling loop.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::net::ToSocketAddrs;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use crate::protocol::{
    Claim, ClientMessage, ErrorKind, Event, GpuLedger, Hello, HostStorage, Item, Key, Link,
    Listing, Mode, Placement, Priority, Region, Registered, Reply, Request, Role, Scope,
    ServerMessage, Span, StorageRecord, TensorRecord, Throttle, TransferTotal, VERSION, Welcome,
};
use crate::transport::Connection;

/// Why a request failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("storage server connection: {0}")]
    Io(#[from] io::Error),
    #[error("the storage server refused the connection: {0}")]
    Refused(String),
    #[error("the storage server answered {kind:?}: {message}")]
    Server { kind: ErrorKind, message: String },
    #[error("the storage server answered out of protocol: {0}")]
    Protocol(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A reply and the descriptors that came with it.
#[derive(Debug)]
pub struct Answer {
    pub reply: Reply,
    pub descriptors: Vec<OwnedFd>,
}

/// The tensors a `get` found, over their storages, and the keys it did not.
#[derive(Debug)]
pub struct Got {
    pub tensors: Vec<TensorRecord>,
    /// The storages the tensors name by index, with their descriptors.
    pub storages: Vec<(StorageRecord, OwnedFd)>,
    pub missing: Vec<Key>,
}

/// A connection to the Storage Server of this host.
pub struct Client {
    connection: Connection,
    /// The protocol version the connection speaks: this build's, or an
    /// earlier one an operator's connection to an earlier server said hello
    /// with ([`Client::connect_operator`]).
    version: u32,
    next: u64,
    /// Answers received and not yet taken, by request.
    answers: HashMap<u64, Answer>,
    /// Events received and not yet taken, with their descriptors.
    events: VecDeque<(Event, Vec<OwnedFd>)>,
}

impl Client {
    /// Connect to the server at `socket` as `role`.
    pub fn connect(socket: &Path, role: Role) -> Result<Self> {
        Self::open(Connection::new(UnixStream::connect(socket)?), role)
    }

    /// Connect to the server of another host at `address` (`host:port`,
    /// its `Endpoint`) as `role`. The connection carries page requests of
    /// a head only, and no descriptors.
    pub fn connect_tcp(address: &str, role: Role) -> Result<Self> {
        let stream = std::net::TcpStream::connect(address)?;
        stream.set_nodelay(true)?;
        Self::open(Connection::new(stream), role)
    }

    /// Connect to the server of another host at `address` (`host:port`,
    /// its `Endpoint`) as [`Role::Peer`], the Transfer Server of this host
    /// that READs storage there. Connecting, and every blocking call after
    /// it ([`Client::call`]), fails once `timeout` passes without an
    /// answer, so a host that stopped answering fails the caller's task
    /// instead of holding it.
    pub fn connect_peer(address: &str, timeout: Duration) -> Result<Self> {
        let mut refused = None;
        for candidate in address.to_socket_addrs()? {
            match std::net::TcpStream::connect_timeout(&candidate, timeout) {
                Ok(stream) => {
                    stream.set_nodelay(true)?;
                    stream.set_read_timeout(Some(timeout))?;
                    stream.set_write_timeout(Some(timeout))?;
                    return Self::open(Connection::new(stream), Role::Peer);
                }
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

    /// Connect to the server at `socket` as an operator, in this build's
    /// protocol or, when the server speaks an earlier one, in the latest
    /// earlier version it serves, as the restart of a server whose protocol
    /// predates this build's connects: an operator's requests (`List`,
    /// `Retire`, `Restart`) read alike in every version.
    pub fn connect_operator(socket: &Path) -> Result<Self> {
        let mut refusal = None;
        for version in (crate::protocol::MIN_VERSION..=VERSION).rev() {
            let connection = Connection::new(UnixStream::connect(socket)?);
            match Self::open_version(connection, Role::Operator, version) {
                Ok(client) => return Ok(client),
                Err(Error::Refused(message)) => refusal = Some(message),
                Err(error) => return Err(error),
            }
        }
        Err(Error::Refused(refusal.unwrap_or_default()))
    }

    /// The protocol version the server accepted the connection in.
    pub fn version(&self) -> u32 {
        self.version
    }

    fn open(connection: Connection, role: Role) -> Result<Self> {
        Self::open_version(connection, role, VERSION)
    }

    fn open_version(mut connection: Connection, role: Role, version: u32) -> Result<Self> {
        connection.send(&ClientMessage::Hello(Hello { version, role }), &[])?;
        match connection.recv::<ServerMessage>()?.0 {
            ServerMessage::Welcome(Welcome::Accepted { .. }) => Ok(Self {
                connection,
                version,
                next: 1,
                answers: HashMap::new(),
                events: VecDeque::new(),
            }),
            ServerMessage::Welcome(Welcome::Refused { message }) => Err(Error::Refused(message)),
            other => Err(Error::Protocol(format!("{other:?} before a welcome"))),
        }
    }

    /// The connection's socket and the next request identity, for another
    /// process to go on with ([`Client::resume`]), as a daemon's hot
    /// restart hands its claim to its successor: the server knows the
    /// connection under the role it said hello as.
    ///
    /// # Errors
    ///
    /// Fails when answers, events or frames are buffered.
    pub fn hand_over(self) -> Result<(OwnedFd, u64)> {
        if !self.answers.is_empty() || !self.events.is_empty() {
            return Err(Error::Protocol(
                "a connection with answers or events not taken cannot be handed over".into(),
            ));
        }
        let socket = match self.connection.into_socket()? {
            crate::transport::Socket::Unix(stream) => OwnedFd::from(stream),
            crate::transport::Socket::Tcp(stream) => OwnedFd::from(stream),
        };
        Ok((socket, self.next))
    }

    /// Goes on with a Unix connection another process handed over
    /// ([`Client::hand_over`]).
    pub fn resume(descriptor: OwnedFd, next: u64) -> Self {
        Self {
            connection: Connection::new(UnixStream::from(descriptor)),
            version: VERSION,
            next,
            answers: HashMap::new(),
            events: VecDeque::new(),
        }
    }

    /// The connection's descriptor, readable when answers or events arrive.
    pub fn as_fd(&self) -> BorrowedFd<'_> {
        self.connection.stream().as_fd()
    }

    /// Send `request` with `descriptors` and return its identity. The
    /// answer is taken with [`Self::wait`] or [`Self::take`].
    pub fn submit(&mut self, request: &Request, descriptors: &[BorrowedFd<'_>]) -> Result<u64> {
        let id = self.next;
        self.next += 1;
        self.connection.send(
            &ClientMessage::Request {
                id,
                request: request.clone(),
            },
            descriptors,
        )?;
        Ok(id)
    }

    /// Read what the server sent without blocking.
    pub fn pump(&mut self) -> Result<()> {
        self.connection.stream().set_nonblocking(true)?;
        let read = (|| -> Result<()> {
            while self.connection.receive()? {}
            self.route()
        })();
        self.connection.stream().set_nonblocking(false)?;
        read
    }

    /// The answer to request `id`, when it has arrived.
    pub fn take(&mut self, id: u64) -> Option<Answer> {
        self.answers.remove(&id)
    }

    /// The next event received, with the descriptors that came with it.
    pub fn next_event(&mut self) -> Option<(Event, Vec<OwnedFd>)> {
        self.events.pop_front()
    }

    /// Consume one task's completion without waiting, discarding its progress
    /// notifications. Other tasks' events stay queued for their owners.
    /// Returns `None` while the transfer is pending or after it was consumed.
    pub fn poll_transfer(
        &mut self,
        task: u64,
    ) -> Result<Option<(u64, crate::protocol::TransferEnd)>> {
        self.pump()?;
        let mut completion = None;
        self.events.retain(|(event, _)| match event {
            Event::TransferProgress { task: found, .. } if *found == task => false,
            Event::Transferred {
                task: found,
                bytes,
                end,
            } if *found == task => {
                completion = Some((*bytes, end.clone()));
                false
            }
            _ => true,
        });
        Ok(completion)
    }

    /// Block until request `id` is answered.
    pub fn wait(&mut self, id: u64) -> Result<Answer> {
        loop {
            if let Some(answer) = self.answers.remove(&id) {
                return Ok(answer);
            }
            let (message, descriptors) = self.connection.recv::<ServerMessage>()?;
            self.deliver(message, descriptors)?;
        }
    }

    /// Send `request` and block for its answer.
    pub fn call(&mut self, request: &Request, descriptors: &[BorrowedFd<'_>]) -> Result<Answer> {
        let id = self.submit(request, descriptors)?;
        self.wait(id)
    }

    /// Route every complete message received.
    fn route(&mut self) -> Result<()> {
        while let Some((message, descriptors)) = self.connection.next_frame::<ServerMessage>()? {
            self.deliver(message, descriptors)?;
        }
        if self.connection.is_closed() {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the storage server closed the connection",
            )));
        }
        Ok(())
    }

    fn deliver(&mut self, message: ServerMessage, descriptors: Vec<OwnedFd>) -> Result<()> {
        match message {
            ServerMessage::Reply { id, reply } => {
                self.answers.insert(id, Answer { reply, descriptors });
            }
            ServerMessage::Event(event) => self.events.push_back((event, descriptors)),
            ServerMessage::Welcome(_) => {
                return Err(Error::Protocol("a second welcome".into()));
            }
        }
        Ok(())
    }

    /// The answer, unless the server answered with an error.
    fn expect(answer: Answer) -> Result<Answer> {
        match answer.reply {
            Reply::Error { kind, message } => Err(Error::Server { kind, message }),
            _ => Ok(answer),
        }
    }

    /// See [`Request::Claim`]: become the deployment's head, `world`
    /// placing `scopes` on the host, and end every world of `retire` there.
    pub fn claim(&mut self, world: u64, scopes: Vec<Scope>, retire: Vec<u64>) -> Result<Claim> {
        let request = Request::Claim {
            scopes,
            world,
            retire,
        };
        match Self::expect(self.call(&request, &[])?)?.reply {
            Reply::Claimed { claim } => Ok(claim),
            other => Err(unexpected("claim", &other)),
        }
    }

    /// Make the claimed, committed worlds' tensors and pages shared together.
    pub fn publish_worlds(&mut self, worlds: Vec<u64>) -> Result<()> {
        match Self::expect(self.call(&Request::PublishWorlds { worlds }, &[])?)?.reply {
            Reply::WorldsPublished => Ok(()),
            other => Err(unexpected("publishing worlds", &other)),
        }
    }

    /// See [`Request::Retain`]: retain the copies of `keys` at the
    /// locations `at` (every copy when empty). Returns how many tensors
    /// were released.
    pub fn retain(&mut self, keys: Vec<Key>, at: Vec<Placement>) -> Result<u64> {
        match Self::expect(self.call(&Request::Retain { keys, at }, &[])?)?.reply {
            Reply::Retained { released } => Ok(released),
            other => Err(unexpected("retain", &other)),
        }
    }

    /// See [`Request::Get`]: the copies of `keys` at the locations `at`
    /// (every copy when empty).
    pub fn get(&mut self, keys: Vec<Key>, mode: Mode, at: Vec<Placement>) -> Result<Got> {
        let answer = Self::expect(self.call(&Request::Get { keys, mode, at }, &[])?)?;
        Self::got(answer)
    }

    /// Allocate retained host storage only after the server admits its
    /// entire size. The caller maps the returned writable descriptor.
    pub fn allocate_host(&mut self, key: Key, name: String, bytes: u64) -> Result<Got> {
        let answer = Self::expect(self.call(&Request::AllocateHost { key, name, bytes }, &[])?)?;
        Self::got(answer)
    }

    fn got(answer: Answer) -> Result<Got> {
        match answer.reply {
            Reply::Got {
                tensors,
                storages,
                missing,
            } => {
                if storages.len() != answer.descriptors.len() {
                    return Err(Error::Protocol(format!(
                        "{} storages arrived with {} descriptors",
                        storages.len(),
                        answer.descriptors.len()
                    )));
                }
                Ok(Got {
                    tensors,
                    storages: storages.into_iter().zip(answer.descriptors).collect(),
                    missing,
                })
            }
            other => Err(unexpected("get", &other)),
        }
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
        match Self::expect(self.call(&request, &descriptors)?)?.reply {
            Reply::Put => Ok(()),
            other => Err(unexpected("put", &other)),
        }
    }

    /// See [`Request::Promote`].
    pub fn promote(&mut self, keys: Vec<Key>) -> Result<()> {
        match Self::expect(self.call(&Request::Promote { keys }, &[])?)?.reply {
            Reply::Promoted => Ok(()),
            other => Err(unexpected("promote", &other)),
        }
    }

    /// See [`Request::Release`].
    pub fn release(&mut self, keys: Vec<Key>) -> Result<()> {
        match Self::expect(self.call(&Request::Release { keys }, &[])?)?.reply {
            Reply::Released => Ok(()),
            other => Err(unexpected("release", &other)),
        }
    }

    /// See [`Request::Rekey`]: move the copies at the locations `at`
    /// (every copy when empty).
    pub fn rekey(&mut self, moves: Vec<(Key, Key)>, at: Vec<Placement>) -> Result<()> {
        match Self::expect(self.call(&Request::Rekey { moves, at }, &[])?)?.reply {
            Reply::Rekeyed => Ok(()),
            other => Err(unexpected("rekey", &other)),
        }
    }

    /// See [`Request::Restart`]. Returns the successor's process id.
    pub fn restart(&mut self, executable: Option<String>) -> Result<u32> {
        match Self::expect(self.call(&Request::Restart { executable }, &[])?)?.reply {
            Reply::Restarting { pid } => Ok(pid),
            other => Err(unexpected("restart", &other)),
        }
    }

    /// See [`Request::Retire`]. Returns how many tensors were released.
    pub fn retire(&mut self, deployment: String) -> Result<u64> {
        match Self::expect(self.call(&Request::Retire { deployment }, &[])?)?.reply {
            Reply::Retired { released } => Ok(released),
            other => Err(unexpected("retire", &other)),
        }
    }

    /// See [`Request::Transfer`]. Returns the task, whose progress and end
    /// arrive as events.
    pub fn transfer(&mut self, spans: Vec<Span>, priority: Priority, update: u64) -> Result<u64> {
        let request = Request::Transfer {
            spans,
            priority,
            update,
        };
        match Self::expect(self.call(&request, &[])?)?.reply {
            Reply::Transferring { task } => Ok(task),
            other => Err(unexpected("transfer", &other)),
        }
    }

    /// Hold transfer contexts between tasks until `close_transfer` or
    /// connection closure. Each operation owns its separate session.
    pub fn open_transfer(&mut self, gpus: Vec<[u8; 16]>) -> Result<u64> {
        match Self::expect(self.call(&Request::OpenTransfer { gpus }, &[])?)?.reply {
            Reply::TransferOpened { session } => Ok(session),
            other => Err(unexpected("open transfer", &other)),
        }
    }

    /// Release a session; existing tasks retain their own physical holds.
    pub fn close_transfer(&mut self, session: u64) -> Result<()> {
        match Self::expect(self.call(&Request::CloseTransfer { session }, &[])?)?.reply {
            Reply::TransferClosed => Ok(()),
            other => Err(unexpected("close transfer", &other)),
        }
    }

    /// See [`Request::Cancel`]. Returns how many tasks were cancelled.
    pub fn cancel(&mut self, tasks: Vec<u64>, update: Option<u64>) -> Result<u64> {
        match Self::expect(self.call(&Request::Cancel { tasks, update }, &[])?)?.reply {
            Reply::Cancelled { tasks } => Ok(tasks),
            other => Err(unexpected("cancel", &other)),
        }
    }

    /// See [`Request::Throttle`].
    pub fn throttle(&mut self, links: Vec<Link>, throttle: Throttle) -> Result<()> {
        match Self::expect(self.call(&Request::Throttle { links, throttle }, &[])?)?.reply {
            Reply::Throttled => Ok(()),
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
    ) -> Result<(u64, Vec<u8>, Vec<Registered>)> {
        let request = Request::Expose {
            deployment,
            regions,
        };
        match Self::expect(self.call(&request, &[])?)?.reply {
            Reply::Exposed {
                exposure,
                metadata,
                regions,
            } => Ok((exposure, metadata, regions)),
            other => Err(unexpected("expose", &other)),
        }
    }

    /// See [`Request::Unexpose`].
    pub fn unexpose(&mut self, exposure: u64) -> Result<()> {
        match Self::expect(self.call(&Request::Unexpose { exposure }, &[])?)?.reply {
            Reply::Unexposed => Ok(()),
            other => Err(unexpected("unexpose", &other)),
        }
    }

    /// See [`Request::List`].
    pub fn list(&mut self) -> Result<Listing> {
        match Self::expect(self.call(&Request::List, &[])?)?.reply {
            Reply::Listing(listing) => Ok(listing),
            other => Err(unexpected("list", &other)),
        }
    }

    /// See [`Request::Read`]: the bytes of `deployment`'s read-only host
    /// tensor `key`.
    pub fn read(&mut self, deployment: String, key: Key) -> Result<Vec<u8>> {
        match Self::expect(self.call(&Request::Read { deployment, key }, &[])?)?.reply {
            Reply::Read { bytes } => Ok(bytes),
            other => Err(unexpected("read", &other)),
        }
    }

    /// See [`Request::Offload`] (`to_host`) and [`Request::Restore`].
    /// Returns how many storages moved and their bytes, once every copy
    /// ended.
    pub fn move_weights(&mut self, world: u64, to_host: bool) -> Result<(u64, u64)> {
        let request = if to_host {
            Request::Offload { world }
        } else {
            Request::Restore { world }
        };
        match Self::expect(self.call(&request, &[])?)?.reply {
            Reply::Moved { storages, bytes } => Ok((storages, bytes)),
            other => Err(unexpected("move_weights", &other)),
        }
    }

    /// See [`Request::KeepWeights`]. Returns how many tensors were released.
    pub fn keep_weights(&mut self, worlds: Vec<u64>, cache_keys: Vec<Key>) -> Result<u64> {
        match Self::expect(self.call(&Request::KeepWeights { worlds, cache_keys }, &[])?)?.reply {
            Reply::Retained { released } => Ok(released),
            other => Err(unexpected("keep_weights", &other)),
        }
    }

    /// See [`Request::Book`]. Returns the ledgers of `gpu` the caller books
    /// in; a refusal is [`ErrorKind::Quota`], whose message names every
    /// entry of the ledger it would overrun.
    pub fn book(
        &mut self,
        gpu: [u8; 16],
        total: u64,
        quota: Option<u64>,
        items: Vec<(Item, u64)>,
    ) -> Result<Vec<GpuLedger>> {
        let request = Request::Book {
            gpu,
            total,
            quota,
            items,
        };
        match Self::expect(self.call(&request, &[])?)?.reply {
            Reply::Ledger { ledgers, .. } => Ok(ledgers),
            other => Err(unexpected("book", &other)),
        }
    }

    /// See [`Request::Ledger`]: the ledgers of `gpus` (every GPU when empty),
    /// what the host admits of new host storage, and what the transfer
    /// tasks of the client's deployment copied.
    pub fn ledger(&mut self, gpus: Vec<[u8; 16]>) -> Result<Ledgers> {
        match Self::expect(self.call(&Request::Ledger { gpus }, &[])?)?.reply {
            Reply::Ledger {
                ledgers,
                host,
                transfers,
            } => Ok(Ledgers {
                ledgers,
                host,
                transfers,
            }),
            other => Err(unexpected("ledger", &other)),
        }
    }
}

/// What a [`Request::Ledger`] answers ([`Client::ledger`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ledgers {
    /// The ledgers of the GPUs asked for.
    pub ledgers: Vec<GpuLedger>,
    /// What the host admits of new host storage.
    pub host: HostStorage,
    /// What the transfer tasks of the client's deployment copied on the
    /// host, by priority; empty from a server of protocol 17 or earlier.
    pub transfers: Vec<TransferTotal>,
}

fn unexpected(request: &str, reply: &Reply) -> Error {
    Error::Protocol(format!("{request} was answered with {reply:?}"))
}
