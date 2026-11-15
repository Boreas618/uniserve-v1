//! The one connection every process interface of the workspace runs on, over a Unix socket on
//! the host or TCP between hosts, and the listener that accepts it.
//!
//! A [`Connection`] carries the frames of [`crate::frame`], the one frame layout of every
//! protocol: the daemon's interfaces with the engine and the hot-restart handoff, the engines'
//! peer link, and the Storage Server's and the Build Server's connections. The codec is each
//! protocol's: its envelope types implement [`Message`], which encodes a message into a frame's
//! payload. One connection carries the messages of one protocol.
//!
//! A connection's peer and a listener's address are an [`Endpoint`]: a Unix socket of this host
//! or a TCP endpoint. A connection is non-blocking and makes progress only when called: [`Connection::send`] queues a
//! frame and writes what the socket takes, and every receive and wait reads what arrived and
//! writes what is queued, so a frame larger than the kernel's send buffer finishes across later
//! calls and two ends that send at once never wait on each other. [`Connection::flush`] waits
//! until every queued frame is written, which an owner does before it closes a connection on
//! purpose, since dropping one discards what is queued, and [`Connection::deliver`] sends and
//! flushes, for an owner that does not go on calling the connection. An owner blocks in
//! [`Connection::wait`] on the socket together with the connection's local [`Wake`], which its
//! other threads fire for work that did not arrive on the socket.
//!
//! A closed connection is terminal. Frames that arrived whole before the close are still
//! delivered; every call after them fails with `UnexpectedEof`. A TCP connection probes its peer
//! ([`Liveness`]), so a peer whose host died closes the connection like one that exited.
//!
//! Over a Unix socket a frame carries descriptors beside it ([`Connection::send_with`]): they
//! travel with the frame's first byte and the receiver gets its own descriptors of the same open
//! files, exactly those of the frame they came with ([`Connection::try_recv_with`]).

use std::collections::VecDeque;
use std::io::{self, ErrorKind, Read as _, Write as _};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::Endpoint;
use crate::frame::{self, Segment};

/// How a TCP connection notices a peer whose host stopped answering.
///
/// A host that dies sends no close: without a bound a connection to it would stay open for as long
/// as nothing is sent on it (the kernel's keepalive default, two hours, is no bound in practice).
/// An idle connection is probed after half of `bound` and fails when `probes` probes, spread over
/// the other half, go unanswered; data sent and left unacknowledged fails it after the whole
/// `bound` (TCP keepalive and `TCP_USER_TIMEOUT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Liveness {
    /// The time after which a connection whose peer stopped answering fails.
    pub bound: Duration,
    /// The keepalive probes an idle connection sends before it fails.
    pub probes: u32,
}

impl Liveness {
    /// The workspace's policy for every TCP connection: a host that died is noticed within a
    /// minute, inside the deployment's own failure bounds (a member's join, an engine's
    /// watchdog), while a host whose processes are busy for seconds is not taken for dead.
    pub const DEAD_PEER: Self = Self {
        bound: Duration::from_secs(60),
        probes: 3,
    };
}

/// An envelope of one protocol: how it becomes a frame's payload and back. Every message one
/// connection carries in either direction belongs to the same protocol.
pub trait Message: Sized {
    /// The longest payload a peer may send of this message; a longer one is a corrupt stream,
    /// refused before it is buffered whole. The frame header's own limit unless the protocol
    /// states a tighter one for the message.
    const MAX_BYTES: usize = u32::MAX as usize;

    /// The message as a frame's payload.
    ///
    /// # Errors
    ///
    /// Fails when the message cannot be encoded.
    fn encode(&self) -> io::Result<Vec<u8>>;

    /// The message a frame's payload holds.
    ///
    /// # Errors
    ///
    /// Fails with `InvalidData` when the payload is no encoding of this message.
    fn decode(payload: &[u8]) -> io::Result<Self>;
}

/// A connected stream of either kind.
#[derive(Debug)]
pub enum Socket {
    /// A Unix socket of this host, which carries descriptors.
    Unix(UnixStream),
    /// A TCP socket to another host, which carries none.
    Tcp(TcpStream),
}

impl AsFd for Socket {
    fn as_fd(&self) -> BorrowedFd<'_> {
        match self {
            Self::Unix(stream) => stream.as_fd(),
            Self::Tcp(stream) => stream.as_fd(),
        }
    }
}

impl From<UnixStream> for Socket {
    fn from(stream: UnixStream) -> Self {
        Self::Unix(stream)
    }
}

impl From<TcpStream> for Socket {
    fn from(stream: TcpStream) -> Self {
        Self::Tcp(stream)
    }
}

/// A connection's socket as it crosses to another process, as a daemon's or a Storage Server's
/// restart in place hands its connections to its successor: the socket, whether it is a Unix
/// socket, and the bytes received and not yet taken as frames.
pub struct Handed {
    /// The socket.
    pub descriptor: OwnedFd,
    /// Whether the socket is a Unix socket rather than TCP.
    pub unix: bool,
    /// The bytes received and not yet taken as frames.
    pub received: Vec<u8>,
}

/// One end of a connection.
pub struct Connection {
    socket: Socket,
    /// Bytes received and not yet taken as frames.
    inbox: Vec<u8>,
    /// Descriptors received and not yet given to a frame, in arrival order.
    descriptors: VecDeque<OwnedFd>,
    /// Frames, or parts of frames, not yet written.
    outbox: VecDeque<Segment>,
    /// The terminal failure, as its kind and description, reported once the frames that arrived
    /// whole before it were taken: the peer's close (`UnexpectedEof`) or the socket's error.
    failure: Option<(ErrorKind, String)>,
    /// The buffer one receive reads into, allocated at the first receive as large as the
    /// socket's receive buffer (`SO_RCVBUF`): one receive returns at most what the socket holds,
    /// so a larger buffer reads nothing more and a smaller one splits what arrived.
    chunk: Vec<u8>,
    local: LocalWake,
}

impl Connection {
    /// Connects to `endpoint`.
    ///
    /// # Errors
    ///
    /// Fails when nothing listens at the endpoint, or the socket cannot be configured.
    pub fn connect(endpoint: &Endpoint) -> io::Result<Self> {
        let failed = |error: io::Error| {
            io::Error::new(error.kind(), format!("connecting to {endpoint}: {error}"))
        };
        let socket = match endpoint {
            Endpoint::Unix(path) => Socket::Unix(UnixStream::connect(path).map_err(failed)?),
            Endpoint::Tcp { host, port } => {
                Socket::Tcp(TcpStream::connect((host.as_str(), *port)).map_err(failed)?)
            }
        };
        Self::new(socket)
    }

    /// One end of a connected socket: made non-blocking, and over TCP sent without delay and
    /// probed for a dead peer.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be configured or the local wake cannot be made.
    pub fn new(socket: impl Into<Socket>) -> io::Result<Self> {
        let socket = socket.into();
        match &socket {
            Socket::Unix(stream) => stream.set_nonblocking(true)?,
            Socket::Tcp(stream) => {
                // Without Nagle a small frame is sent without waiting for earlier data to be
                // acknowledged.
                stream.set_nodelay(true)?;
                keep_alive(stream, Liveness::DEAD_PEER)?;
                stream.set_nonblocking(true)?;
            }
        }
        Ok(Self {
            socket,
            inbox: Vec::new(),
            descriptors: VecDeque::new(),
            outbox: VecDeque::new(),
            failure: None,
            chunk: Vec::new(),
            local: LocalWake::new()?,
        })
    }

    /// Goes on with a connection another process handed over ([`Connection::hand_over`]).
    ///
    /// # Errors
    ///
    /// As [`Connection::new`].
    pub fn resume(handed: Handed) -> io::Result<Self> {
        let socket = if handed.unix {
            Socket::Unix(UnixStream::from(handed.descriptor))
        } else {
            Socket::Tcp(TcpStream::from(handed.descriptor))
        };
        let mut connection = Self::new(socket)?;
        connection.inbox = handed.received;
        Ok(connection)
    }

    /// The connection's socket and the bytes it received and did not take, for another process to
    /// go on with ([`Connection::resume`]).
    ///
    /// # Errors
    ///
    /// Fails with `WouldBlock` when frames wait to be written or received descriptors wait to be
    /// taken: neither crosses to the other process, so the connection is not passed on.
    pub fn hand_over(self) -> io::Result<Handed> {
        if !self.outbox.is_empty() || !self.descriptors.is_empty() {
            return Err(io::Error::new(
                ErrorKind::WouldBlock,
                "a connection with frames to write or descriptors to take cannot be passed on",
            ));
        }
        let (descriptor, unix) = match self.socket {
            Socket::Unix(stream) => (OwnedFd::from(stream), true),
            Socket::Tcp(stream) => (OwnedFd::from(stream), false),
        };
        Ok(Handed {
            descriptor,
            unix,
            received: self.inbox,
        })
    }

    /// A second end of this connection's socket, for a thread that only sends: this one stays the
    /// sole reader with every frame and descriptor it received. Sends on the two ends are ordered
    /// by their callers.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be duplicated.
    pub fn writer(&self) -> io::Result<Self> {
        let socket = match &self.socket {
            Socket::Unix(stream) => Socket::Unix(stream.try_clone()?),
            Socket::Tcp(stream) => Socket::Tcp(stream.try_clone()?),
        };
        Self::new(socket)
    }

    /// Queues `message` as one frame and writes what the socket takes now; later calls write the
    /// rest.
    ///
    /// # Errors
    ///
    /// The terminal failure once the connection failed, a message that does not encode or does
    /// not fit a frame, and a failure of the write.
    pub fn send<M: Message>(&mut self, message: &M) -> io::Result<()> {
        self.queue(message, Vec::new())
    }

    /// Queues `message` as one frame that carries `descriptors`, each duplicated, and writes what
    /// the socket takes now. The peer receives its own descriptors of the same open files.
    ///
    /// # Errors
    ///
    /// As [`Connection::send`], and `InvalidInput` for descriptors over TCP.
    pub fn send_with<M: Message>(
        &mut self,
        message: &M,
        descriptors: &[BorrowedFd<'_>],
    ) -> io::Result<()> {
        if !descriptors.is_empty() && matches!(self.socket, Socket::Tcp(_)) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "descriptors cross only a Unix socket",
            ));
        }
        let owned = descriptors
            .iter()
            .map(|descriptor| descriptor.try_clone_to_owned())
            .collect::<io::Result<Vec<_>>>()?;
        self.queue(message, owned)
    }

    /// Sends `message` and waits until the socket took it and every frame queued before it: the
    /// send of an owner that does not go on calling this connection, so nothing would write the
    /// rest later.
    ///
    /// # Errors
    ///
    /// As [`Connection::send`] and [`Connection::flush`].
    pub fn deliver<M: Message>(&mut self, message: &M) -> io::Result<()> {
        self.send(message)?;
        self.flush()
    }

    /// Writes every queued frame, waiting while the socket is full.
    ///
    /// # Errors
    ///
    /// The terminal failure, which also ends the wait on a peer that is gone.
    pub fn flush(&mut self) -> io::Result<()> {
        loop {
            self.failed()?;
            if self.try_flush()? {
                return Ok(());
            }
            let mut writable = [pollfd(self.as_raw_fd(), libc::POLLOUT)];
            wait_ready(&mut writable, None)?;
        }
    }

    /// Writes queued frames until the socket would block; whether none is left.
    ///
    /// # Errors
    ///
    /// A failure of the write, which is also recorded as the connection's terminal failure.
    pub fn try_flush(&mut self) -> io::Result<bool> {
        while let Some(segment) = self.outbox.front_mut() {
            match segment.send(self.socket.as_fd()) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(false),
                Err(error) => return Err(self.fail(error, "sending")),
            }
            if segment.sent() {
                self.outbox.pop_front();
            }
        }
        Ok(true)
    }

    /// Whether frames wait to be written: the socket's writability is then worth waiting on.
    pub fn wants_write(&self) -> bool {
        !self.outbox.is_empty()
    }

    /// The next message once it arrived whole, and `None` while it has not. Reads what the socket
    /// holds and writes what is queued when no whole frame is buffered. The descriptors the frame
    /// carried are closed.
    ///
    /// # Errors
    ///
    /// A frame that does not decode or exceeds the message's bound, and the terminal failure once
    /// every frame that arrived whole before it was taken.
    pub fn try_recv<M: Message>(&mut self) -> io::Result<Option<M>> {
        Ok(self.try_recv_with()?.map(|(message, _)| message))
    }

    /// [`Connection::try_recv`], with the descriptors the frame carried.
    ///
    /// # Errors
    ///
    /// As [`Connection::try_recv`].
    pub fn try_recv_with<M: Message>(&mut self) -> io::Result<Option<(M, Vec<OwnedFd>)>> {
        if let Some(frame) = self.take()? {
            return Ok(Some(frame));
        }
        self.pump();
        if let Some(frame) = self.take()? {
            return Ok(Some(frame));
        }
        self.failed().map(|()| None)
    }

    /// Waits up to `timeout` for the next message; `None` when none arrived in time.
    ///
    /// # Errors
    ///
    /// As [`Connection::try_recv`].
    pub fn recv<M: Message>(&mut self, timeout: Duration) -> io::Result<Option<M>> {
        Ok(self.recv_with(timeout)?.map(|(message, _)| message))
    }

    /// [`Connection::recv`], with the descriptors the frame carried.
    ///
    /// # Errors
    ///
    /// As [`Connection::try_recv`].
    pub fn recv_with<M: Message>(
        &mut self,
        timeout: Duration,
    ) -> io::Result<Option<(M, Vec<OwnedFd>)>> {
        // A timeout too long to be a point in time is no deadline.
        let deadline = Instant::now().checked_add(timeout);
        loop {
            if let Some(frame) = self.try_recv_with()? {
                return Ok(Some(frame));
            }
            let remaining =
                deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
            if remaining.is_some_and(|remaining| remaining.is_zero()) {
                return Ok(None);
            }
            self.wait(remaining.unwrap_or(Duration::MAX))?;
        }
    }

    /// Blocks until something happens the owner can act on, or `timeout` passes: bytes arrive, a
    /// queued frame is written, or the local wake fires (a wake already pending ends the wait at
    /// once). It does not say which, so the caller checks every source of progress afterwards.
    ///
    /// Frames already buffered do not end the wait: a caller that can take a message calls
    /// [`Connection::try_recv`] until it returns `None` before it waits, and one that cannot take
    /// one now still blocks here rather than spinning.
    ///
    /// # Errors
    ///
    /// The terminal failure once the connection failed or the peer closed it: at once when that
    /// is known, and otherwise when it is observed without new bytes arriving first.
    pub fn wait(&mut self, timeout: Duration) -> io::Result<()> {
        self.wait_on(None, timeout).map(drop)
    }

    /// [`Connection::wait`], which also ends once `interrupt` becomes readable or reaches its
    /// end; returns whether it did. A thread that waits on a connection on its owner's behalf
    /// blocks here, and the owner stops it by writing to, or closing, the other end of
    /// `interrupt` (a pipe).
    ///
    /// # Errors
    ///
    /// As [`Connection::wait`].
    pub fn wait_or(&mut self, interrupt: BorrowedFd<'_>, timeout: Duration) -> io::Result<bool> {
        self.wait_on(Some(interrupt), timeout)
    }

    /// The wait of [`Connection::wait`] and [`Connection::wait_or`]; whether `interrupt` ended
    /// it.
    fn wait_on(
        &mut self,
        interrupt: Option<BorrowedFd<'_>>,
        timeout: Duration,
    ) -> io::Result<bool> {
        self.failed()?;
        if self.local.take() {
            return Ok(false);
        }

        let deadline = Instant::now().checked_add(timeout);
        let received = self.inbox.len();
        let queued = self.outbox.len();
        loop {
            let remaining =
                deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
            let [socket, local] = self.poll_fds();
            let mut descriptors = [
                socket,
                local,
                pollfd(interrupt.map_or(-1, |fd| fd.as_raw_fd()), libc::POLLIN),
            ];
            // A negative descriptor is one `poll` skips.
            wait_ready(&mut descriptors, remaining)?;
            if descriptors[2].revents != 0 {
                return Ok(true);
            }
            let woken = self.local.take();
            self.pump();

            // Bytes that arrived with the close are reported first; the next wait reports the
            // close.
            let arrived = self.inbox.len() > received;
            let sent = self.outbox.len() < queued;
            if woken || arrived || sent {
                return Ok(false);
            }
            self.failed()?;
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Ok(false);
            }
        }
    }

    /// A handle that fires this connection's local wake, for any thread of this process.
    pub fn wake(&self) -> Wake {
        self.local.wake.clone()
    }

    /// The descriptors whose readiness lets this connection make progress: the socket, writable
    /// only while a frame is queued, and the local wake. An owner of several connections polls
    /// them together and then calls [`Connection::try_recv`] on each.
    pub fn poll_fds(&self) -> [libc::pollfd; 2] {
        let writing = if self.outbox.is_empty() {
            0
        } else {
            libc::POLLOUT
        };
        [
            pollfd(self.as_raw_fd(), libc::POLLIN | writing),
            self.local.poll_fd(),
        ]
    }

    /// Whether the peer closed the connection or it failed; frames that arrived whole before may
    /// still wait to be taken.
    pub fn is_closed(&self) -> bool {
        self.failure.is_some()
    }

    /// Closes the connection both ways at once, whichever handles of its socket this process
    /// holds: the peer reads its end, and a thread waiting on another handle of the socket wakes.
    pub fn shutdown(&self) {
        let _ = match &self.socket {
            Socket::Unix(stream) => stream.shutdown(std::net::Shutdown::Both),
            Socket::Tcp(stream) => stream.shutdown(std::net::Shutdown::Both),
        };
    }

    /// An observer of the peer's close that does not read the connection, for a thread of the
    /// owner's that can wait while the owner's thread is occupied elsewhere.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be duplicated.
    pub fn hangup(&self) -> io::Result<Hangup> {
        Ok(Hangup {
            socket: self.socket.as_fd().try_clone_to_owned()?,
        })
    }

    /// The peer's address over TCP; `None` over a Unix socket.
    pub fn peer_addr(&self) -> Option<SocketAddr> {
        match &self.socket {
            Socket::Tcp(stream) => stream.peer_addr().ok(),
            Socket::Unix(_) => None,
        }
    }

    /// This end's address over TCP, the address the peer reached it at, which is how a process
    /// that binds every interface learns an address its peers route to; `None` over a Unix
    /// socket.
    pub fn local_addr(&self) -> Option<SocketAddr> {
        match &self.socket {
            Socket::Tcp(stream) => stream.local_addr().ok(),
            Socket::Unix(_) => None,
        }
    }

    /// Queues one frame of `message` carrying `descriptors` and writes what the socket takes.
    fn queue<M: Message>(&mut self, message: &M, descriptors: Vec<OwnedFd>) -> io::Result<()> {
        self.failed()?;
        let payload = message.encode()?;
        self.outbox.extend(frame::segments(&payload, descriptors)?);
        self.try_flush().map(drop)
    }

    /// Takes one whole frame from the received bytes, decoded as `M`, with its descriptors.
    fn take<M: Message>(&mut self) -> io::Result<Option<(M, Vec<OwnedFd>)>> {
        if let Some(length) = frame::payload_bytes(&self.inbox)
            && length > M::MAX_BYTES
        {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "a frame of {length} bytes exceeds the {} bytes a message takes",
                    M::MAX_BYTES
                ),
            ));
        }
        let Some((payload, descriptors)) = frame::take(&mut self.inbox, &mut self.descriptors)
        else {
            return Ok(None);
        };
        Ok(Some((M::decode(&payload)?, descriptors)))
    }

    /// Reads what the socket holds and writes what is queued, recording a failure instead of
    /// returning it, so that the frames that arrived whole before it are delivered first.
    fn pump(&mut self) {
        if self.failure.is_some() {
            return;
        }
        if self.chunk.is_empty() {
            match receive_buffer_bytes(self.socket.as_fd()) {
                Ok(bytes) => self.chunk = vec![0_u8; bytes],
                Err(error) => {
                    self.fail(error, "reading the receive buffer's size");
                    return;
                }
            }
        }
        loop {
            match frame::receive_with_descriptors(
                self.socket.as_fd(),
                &mut self.chunk,
                &mut self.descriptors,
            ) {
                // A peer that answers and then exits closes the stream with its last frames
                // unread, so the bytes already taken stay to be formed into frames.
                Ok(0) => {
                    self.failure = Some((
                        ErrorKind::UnexpectedEof,
                        "the peer closed the connection".to_owned(),
                    ));
                    return;
                }
                Ok(count) => self.inbox.extend_from_slice(&self.chunk[..count]),
                Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == ErrorKind::ConnectionReset => {
                    self.failure = Some((
                        ErrorKind::UnexpectedEof,
                        "the peer reset the connection".to_owned(),
                    ));
                    return;
                }
                Err(error) => {
                    self.fail(error, "receiving");
                    return;
                }
            }
        }
        let _ = self.try_flush();
    }

    /// Records `error` of `action` as the terminal failure and returns it.
    fn fail(&mut self, error: io::Error, action: &str) -> io::Error {
        let kind = match error.kind() {
            ErrorKind::BrokenPipe | ErrorKind::ConnectionReset => ErrorKind::UnexpectedEof,
            kind => kind,
        };
        let message = format!("{action}: {error}");
        self.failure = Some((kind, message.clone()));
        io::Error::new(kind, message)
    }

    /// The terminal failure, once there is one; a close that leaves part of a frame unread says
    /// so.
    fn failed(&self) -> io::Result<()> {
        match &self.failure {
            None => Ok(()),
            Some((kind, message)) if self.inbox.is_empty() => {
                Err(io::Error::new(*kind, message.clone()))
            }
            Some((kind, message)) => Err(io::Error::new(
                *kind,
                format!("{message}, with an incomplete frame unread"),
            )),
        }
    }
}

impl AsFd for Connection {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.socket.as_fd()
    }
}

impl AsRawFd for Connection {
    fn as_raw_fd(&self) -> RawFd {
        self.socket.as_fd().as_raw_fd()
    }
}

/// A listening socket that hands out connections. A Unix socket's path is removed when the
/// listener closes, unless it was handed over to another process.
pub struct Listener {
    socket: Listening,
    /// The endpoint peers connect to, with the port a TCP listener picked.
    endpoint: Endpoint,
}

/// A bound, listening socket of either kind.
enum Listening {
    /// A Unix socket and the path it is bound at.
    Unix(UnixListener, UnixPath),
    Tcp(TcpListener),
}

/// The path a listening Unix socket is bound at, removed when it is dropped: the path outlives
/// the socket's descriptor otherwise. `None` once the socket was handed to another process, which
/// keeps serving at the path.
struct UnixPath(Option<PathBuf>);

impl Drop for UnixPath {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl Listener {
    /// Listens at `endpoint`, where TCP port 0 picks a free port.
    ///
    /// # Errors
    ///
    /// Fails when the endpoint cannot be bound, for example a Unix socket path that exists.
    pub fn bind(endpoint: &Endpoint) -> io::Result<Self> {
        let failed =
            |error: io::Error| io::Error::new(error.kind(), format!("binding {endpoint}: {error}"));
        let socket = match endpoint {
            Endpoint::Unix(path) => {
                let listener = UnixListener::bind(path).map_err(failed)?;
                Listening::Unix(listener, UnixPath(Some(path.clone())))
            }
            Endpoint::Tcp { host, port } => {
                Listening::Tcp(TcpListener::bind((host.as_str(), *port)).map_err(failed)?)
            }
        };
        Self::listening(socket)
    }

    /// Accepts on a TCP socket that is already bound and listening: one its process inherited
    /// from the process that reserved it, so the address its peers were given is live before the
    /// process starts.
    ///
    /// # Errors
    ///
    /// Fails when the socket's address cannot be read or it cannot be made non-blocking.
    pub fn from_tcp(listener: TcpListener) -> io::Result<Self> {
        Self::listening(Listening::Tcp(listener))
    }

    /// Goes on with a listening socket another process handed over ([`Listener::hand_over`]),
    /// bound at `endpoint`.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be made non-blocking.
    pub fn resume(descriptor: OwnedFd, endpoint: &Endpoint) -> io::Result<Self> {
        let socket = match endpoint {
            Endpoint::Unix(path) => {
                Listening::Unix(UnixListener::from(descriptor), UnixPath(Some(path.clone())))
            }
            Endpoint::Tcp { .. } => Listening::Tcp(TcpListener::from(descriptor)),
        };
        Self::listening(socket)
    }

    /// The endpoint the listener is bound at, with the port it picked.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// The listening socket and its endpoint, for another process to go on with
    /// ([`Listener::resume`]); the socket's path stays.
    pub fn hand_over(self) -> (OwnedFd, Endpoint) {
        match self.socket {
            Listening::Unix(listener, mut path) => {
                path.0 = None;
                (OwnedFd::from(listener), self.endpoint)
            }
            Listening::Tcp(listener) => (OwnedFd::from(listener), self.endpoint),
        }
    }

    /// Takes one waiting connection without blocking; `None` when none waits.
    ///
    /// A connection whose peer reset it, or whose network failed, while it waited to be accepted
    /// is reported by `accept` as an error of the listener; it is skipped, as accept(2)
    /// prescribes, so that a stray client cannot fail whoever is accepting.
    ///
    /// # Errors
    ///
    /// Fails when accepting fails otherwise, or the connection cannot be configured.
    pub fn try_accept(&self) -> io::Result<Option<Connection>> {
        loop {
            let accepted = match &self.socket {
                Listening::Unix(listener, _) => {
                    listener.accept().map(|(stream, _)| Socket::Unix(stream))
                }
                Listening::Tcp(listener) => {
                    listener.accept().map(|(stream, _)| Socket::Tcp(stream))
                }
            };
            match accepted {
                Ok(socket) => return Connection::new(socket).map(Some),
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(None),
                Err(error)
                    if matches!(
                        error.kind(),
                        ErrorKind::ConnectionAborted
                            | ErrorKind::ConnectionReset
                            | ErrorKind::NetworkDown
                            | ErrorKind::NetworkUnreachable
                            | ErrorKind::HostUnreachable
                    ) => {}
                Err(error) => return Err(error),
            }
        }
    }

    /// `socket`, non-blocking, with the endpoint peers reach it at.
    fn listening(socket: Listening) -> io::Result<Self> {
        let endpoint = match &socket {
            Listening::Unix(listener, path) => {
                listener.set_nonblocking(true)?;
                Endpoint::Unix(path.0.clone().unwrap_or_default())
            }
            Listening::Tcp(listener) => {
                listener.set_nonblocking(true)?;
                Endpoint::tcp(listener.local_addr()?)
            }
        };
        Ok(Self { socket, endpoint })
    }
}

impl AsRawFd for Listener {
    /// The descriptor that becomes readable when a connection waits.
    fn as_raw_fd(&self) -> RawFd {
        match &self.socket {
            Listening::Unix(listener, _) => listener.as_raw_fd(),
            Listening::Tcp(listener) => listener.as_raw_fd(),
        }
    }
}

/// Fires a local wake from any thread of the process that owns it ([`Connection::wake`],
/// [`Waker::wake`]).
///
/// A wake never travels between processes. It ends the owner's wait for work that did not arrive
/// on a socket, such as a completion another of the owner's threads raised. Wakes fired before
/// the owner takes them coalesce into one.
#[derive(Clone)]
pub struct Wake {
    /// The non-blocking end every fire writes one byte to.
    writer: Arc<UnixStream>,
}

impl Wake {
    /// Fires one wake.
    pub fn wake(&self) {
        // A write refused because the buffer is full loses nothing: the descriptor already holds
        // unread bytes, so it is readable.
        let _ = (&*self.writer).write(&[1]);
    }
}

/// The owner's end of a local wake: the descriptor it polls and drains.
///
/// The descriptor's unread bytes are the wake's only state. Every fire writes one byte and a drain
/// consumes them all, so a wake is reported exactly when its byte is read: one fired while the
/// owner drains is either read and reported by that drain or left readable for the next poll.
struct LocalWake {
    /// The non-blocking end the owner polls and drains.
    reader: UnixStream,
    /// The firing end, shared with every [`Wake`] given out.
    wake: Wake,
}

impl LocalWake {
    fn new() -> io::Result<Self> {
        let (reader, writer) = UnixStream::pair()?;
        reader.set_nonblocking(true)?;
        writer.set_nonblocking(true)?;
        Ok(Self {
            reader,
            wake: Wake {
                writer: Arc::new(writer),
            },
        })
    }

    /// The descriptor to poll for a pending wake.
    fn poll_fd(&self) -> libc::pollfd {
        pollfd(self.reader.as_raw_fd(), libc::POLLIN)
    }

    /// Drains the descriptor; whether any wake had fired.
    fn take(&mut self) -> bool {
        let mut fired = false;
        let mut sink = [0_u8; 64];
        while (&self.reader).read(&mut sink).is_ok_and(|read| read > 0) {
            fired = true;
        }
        fired
    }
}

/// A local wake not tied to a connection.
///
/// The owner of a loop waits on its descriptor ([`Waker::poll_fd`]) among the sources it polls;
/// every thread that has the [`Wake`] it hands out ends that wait by firing it, and the loop then
/// reads the state the firing thread changed.
pub struct Waker {
    local: LocalWake,
}

impl Waker {
    /// A waker with no firing pending.
    ///
    /// # Errors
    ///
    /// Fails when its socket pair cannot be made.
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            local: LocalWake::new()?,
        })
    }

    /// A handle that fires this waker, for any thread.
    pub fn wake(&self) -> Wake {
        self.local.wake.clone()
    }

    /// The descriptor a caller that polls several sources includes: it is readable while a
    /// firing is pending.
    pub fn poll_fd(&self) -> libc::pollfd {
        self.local.poll_fd()
    }

    /// Drains a pending firing, whether or not it ended a wait.
    pub fn drain(&mut self) {
        self.local.take();
    }
}

/// Observes whether a connection's peer closed it, without reading it ([`Connection::hangup`]).
///
/// The observer holds its own descriptor of the connection, so it stays valid whatever happens to
/// the connection it came from.
pub struct Hangup {
    socket: OwnedFd,
}

impl Hangup {
    /// Blocks until the peer closed the connection or it failed, and returns `true`; or until
    /// `cancel` becomes readable or is closed, and returns `false`.
    ///
    /// Data the peer sent and the connection has not read does not hide the close: `poll` reports
    /// the peer's hangup separately from readability (`POLLRDHUP`).
    ///
    /// # Errors
    ///
    /// Fails when the wait fails.
    pub fn wait(&self, cancel: &impl AsRawFd) -> io::Result<bool> {
        let mut descriptors = [
            pollfd(self.socket.as_raw_fd(), libc::POLLRDHUP),
            pollfd(cancel.as_raw_fd(), libc::POLLIN),
        ];
        wait_ready(&mut descriptors, None)?;
        let closed = libc::POLLRDHUP | libc::POLLHUP | libc::POLLERR | libc::POLLNVAL;
        Ok(descriptors[1].revents == 0 && descriptors[0].revents & closed != 0)
    }
}

/// A `pollfd` of `fd` for `events`.
fn pollfd(fd: RawFd, events: libc::c_short) -> libc::pollfd {
    libc::pollfd {
        fd,
        events,
        revents: 0,
    }
}

/// Polls `descriptors` once until one is ready or `timeout` passes (forever for `None`), keeping
/// the deadline across signals. Returns after any completed poll, ready or timed out: callers
/// check their own state rather than `revents`.
pub(crate) fn wait_ready(
    descriptors: &mut [libc::pollfd],
    timeout: Option<Duration>,
) -> io::Result<()> {
    let deadline = timeout.and_then(|timeout| Instant::now().checked_add(timeout));
    loop {
        // Rounded up to whole milliseconds, so the poll does not return before the deadline.
        let milliseconds = match deadline {
            Some(deadline) => {
                let left = deadline.saturating_duration_since(Instant::now());
                let rounded =
                    left.as_millis() + u128::from(!left.subsec_nanos().is_multiple_of(1_000_000));
                libc::c_int::try_from(rounded).unwrap_or(libc::c_int::MAX)
            }
            None => -1,
        };
        // SAFETY: poll borrows this live array only for the call.
        let result = unsafe {
            libc::poll(
                descriptors.as_mut_ptr(),
                descriptors.len() as libc::nfds_t,
                milliseconds,
            )
        };
        if result >= 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() != ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

/// The most bytes one receive from `socket` returns: the size of its receive buffer
/// (`SO_RCVBUF`), which the kernel never exceeds in one call. A reader's buffer of this size reads
/// what the socket holds at once; a larger one reads nothing more and a smaller one splits it.
///
/// # Errors
///
/// What `getsockopt` fails with.
pub fn receive_buffer_bytes(socket: BorrowedFd<'_>) -> io::Result<usize> {
    let bytes = rustix::net::sockopt::socket_recv_buffer_size(socket)?;
    Ok(bytes.max(1))
}

/// Makes a TCP connection fail when its peer's host stops answering, as `liveness` states: an
/// idle connection is probed after half of the bound and fails when the probes, spread over the
/// other half, go unanswered; sent data left unacknowledged fails it after the whole bound.
fn keep_alive(stream: &TcpStream, liveness: Liveness) -> io::Result<()> {
    use rustix::net::sockopt;

    let half = liveness.bound / 2;
    let user_timeout = u32::try_from(liveness.bound.as_millis()).unwrap_or(u32::MAX);
    sockopt::set_socket_keepalive(stream, true)?;
    sockopt::set_tcp_keepidle(stream, half)?;
    sockopt::set_tcp_keepintvl(stream, half / liveness.probes.max(1))?;
    sockopt::set_tcp_keepcnt(stream, liveness.probes)?;
    sockopt::set_tcp_user_timeout(stream, user_timeout)?;
    Ok(())
}

#[cfg(test)]
mod tests;
