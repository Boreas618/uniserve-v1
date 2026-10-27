//! A connection to a Storage Server.
//!
//! One [`Client`] serves both uses: a worker calls it synchronously, at
//! startup, close and promotion ([`Client::call`] and the typed requests
//! beside it); the head submits requests ([`Client::submit`]) and collects
//! their answers when the connection is readable ([`Client::pump`],
//! [`Client::take`]), from its scheduling loop.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::Path;

use crate::protocol::{
    Claim, ClientMessage, ErrorKind, Event, Hello, Key, Link, Listing, Mode, Priority, Reply,
    Request, Role, Scope, ServerMessage, Span, StorageRecord, TensorRecord, Throttle, VERSION,
    Welcome,
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

    /// See [`Request::Retain`]. Returns how many tensors were released.
    pub fn retain(&mut self, keys: Vec<Key>) -> Result<u64> {
        match Self::expect(self.call(&Request::Retain { keys }, &[])?)?.reply {
            Reply::Retained { released } => Ok(released),
            other => Err(unexpected("retain", &other)),
        }
    }

    /// See [`Request::Get`].
    pub fn get(&mut self, keys: Vec<Key>, mode: Mode) -> Result<Got> {
        let answer = Self::expect(self.call(&Request::Get { keys, mode }, &[])?)?;
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

    /// See [`Request::Rekey`].
    pub fn rekey(&mut self, moves: Vec<(Key, Key)>) -> Result<()> {
        match Self::expect(self.call(&Request::Rekey { moves }, &[])?)?.reply {
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

    /// See [`Request::List`].
    pub fn list(&mut self) -> Result<Listing> {
        match Self::expect(self.call(&Request::List, &[])?)?.reply {
            Reply::Listing(listing) => Ok(listing),
            other => Err(unexpected("list", &other)),
        }
    }
}

fn unexpected(request: &str, reply: &Reply) -> Error {
    Error::Protocol(format!("{request} was answered with {reply:?}"))
}
