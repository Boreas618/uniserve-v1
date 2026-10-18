//! The connection between the head and one rank or launcher.
//!
//! A [`Channel`] is one end of a stream connection that carries
//! [`Frame`]s, one after another: each frame's length field tells the reader
//! where the next one starts. The same type serves both ends and both kinds
//! of stream, a Unix socket between processes of one host and TCP between
//! hosts. A [`Listener`] accepts connections and hands each one out as a
//! `Channel`.
//!
//! An address is written `unix:<path>` for a Unix socket and `<host>:<port>`
//! for TCP, the forms gRPC name resolution uses.
//!
//! A channel is non-blocking and makes progress only when called: receive,
//! wait and send calls flush queued writes as far as the socket accepts, so a
//! frame larger than the kernel's send buffer finishes sending across later
//! calls. Its owner blocks in [`Channel::wait`] on the socket together with
//! one local [`Wake`], which the owner's other threads fire for work that did
//! not arrive on the socket. An owner that closes a connection on purpose
//! calls [`Channel::flush`] first, since dropping a channel discards the
//! frames still queued.
//!
//! A closed connection is terminal. Frames that arrived whole before the
//! close are still delivered; every call after them fails. A TCP connection
//! probes its peer (see [`DEAD_PEER_SECS`]), so a peer whose host has died
//! closes the connection like one that exited. [`Hangup`] observes
//! the peer's close without reading the connection, for an owner whose
//! thread is not in a position to call the channel.

use std::collections::VecDeque;
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{IpcError, IpcResult, TransportError};
use crate::frame::Frame;

/// The prefix of a Unix socket address; the rest is the socket's path.
const UNIX_PREFIX: &str = "unix:";

/// Seconds after which a TCP connection whose peer has stopped answering
/// fails.
///
/// A host that dies sends no close. Without this bound its ranks' requests
/// would wait on it, and a rank or launcher whose head's host died would
/// keep its devices, for as long as the connection stayed silent. The system
/// default, two hours, is no bound in practice; this value is chosen, not
/// derived.
pub const DEAD_PEER_SECS: libc::c_int = 60;

/// Wraps an IO failure with what the channel was doing.
fn transport(action: impl Into<std::borrow::Cow<'static, str>>, error: std::io::Error) -> IpcError {
    IpcError::Transport(TransportError::io(action, error))
}

/// A connected stream of either kind.
enum Stream {
    Unix(UnixStream),
    Tcp(TcpStream),
}

impl Stream {
    /// Prepares an accepted or connected stream for a poll-driven channel.
    fn configure(self) -> IpcResult<Self> {
        match &self {
            Self::Unix(stream) => stream.set_nonblocking(true),
            Self::Tcp(stream) => {
                // Disable Nagle so a small segment is sent without waiting
                // for earlier data to be acknowledged.
                stream
                    .set_nodelay(true)
                    .map_err(|error| transport("disabling Nagle", error))?;
                keep_alive(stream)?;
                stream.set_nonblocking(true)
            }
        }
        .map_err(|error| transport("making a channel pollable", error))?;
        Ok(self)
    }
}

/// Makes a TCP connection fail when its peer's host stops answering.
///
/// A host that dies sends no close, so without probes its connection stays
/// open for as long as nothing is sent to it.
fn keep_alive(stream: &TcpStream) -> IpcResult<()> {
    let set = |level: libc::c_int, option: libc::c_int, value: libc::c_int| {
        // SAFETY: the descriptor is this live stream's, and the option value
        // is a `c_int` whose address and size are passed together.
        let result = unsafe {
            libc::setsockopt(
                stream.as_raw_fd(),
                level,
                option,
                (&value as *const libc::c_int).cast(),
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            )
        };
        if result == 0 {
            Ok(())
        } else {
            Err(transport(
                "enabling keepalive",
                std::io::Error::last_os_error(),
            ))
        }
    };

    // An idle connection is probed after half the bound and fails when three
    // probes, spread over the other half, go unanswered. Sent data that stays
    // unacknowledged fails the connection after the whole bound.
    set(libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1)?;
    #[cfg(target_os = "linux")]
    {
        let probes = 3;
        set(libc::IPPROTO_TCP, libc::TCP_KEEPIDLE, DEAD_PEER_SECS / 2)?;
        set(
            libc::IPPROTO_TCP,
            libc::TCP_KEEPINTVL,
            DEAD_PEER_SECS / 2 / probes,
        )?;
        set(libc::IPPROTO_TCP, libc::TCP_KEEPCNT, probes)?;
        set(
            libc::IPPROTO_TCP,
            libc::TCP_USER_TIMEOUT,
            DEAD_PEER_SECS * 1000,
        )?;
    }
    Ok(())
}

impl Read for Stream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Unix(stream) => stream.read(buffer),
            Self::Tcp(stream) => stream.read(buffer),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Unix(stream) => stream.write(buffer),
            Self::Tcp(stream) => stream.write(buffer),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Unix(stream) => stream.flush(),
            Self::Tcp(stream) => stream.flush(),
        }
    }
}

impl AsRawFd for Stream {
    fn as_raw_fd(&self) -> RawFd {
        match self {
            Self::Unix(stream) => stream.as_raw_fd(),
            Self::Tcp(stream) => stream.as_raw_fd(),
        }
    }
}

/// Reads whole frames from a non-blocking stream without losing a partial one.
///
/// A stream delivers bytes, not messages, so a reader that abandons a partial
/// frame loses the bytes it already took. This keeps the incomplete frame and
/// resumes it on the next read.
struct FrameReader {
    /// Bytes taken from the stream and not yet formed into a frame.
    buffer: Vec<u8>,
    /// Frames completed and waiting to be taken.
    ready: VecDeque<Frame>,
}

impl FrameReader {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(64 * 1024),
            ready: VecDeque::new(),
        }
    }

    /// Takes whatever the stream has without blocking, forming whole frames.
    ///
    /// Returns whether the peer closed the connection. Fails on a read error
    /// and on a close that leaves a partial frame; frames completed before
    /// the failure stay in `ready`.
    fn fill(&mut self, stream: &mut Stream) -> Result<bool, TransportError> {
        let mut chunk = [0u8; 64 * 1024];
        let mut closed = false;
        loop {
            match stream.read(&mut chunk) {
                // A peer that answers and then exits closes the stream with
                // its last frames still unread, so the bytes already taken are
                // formed into frames before the close is reported.
                Ok(0) => {
                    closed = true;
                    break;
                }
                Ok(read) => {
                    self.buffer.extend_from_slice(&chunk[..read]);
                    self.split();
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(TransportError::io("reading a channel", error)),
            }
        }
        if closed && !self.buffer.is_empty() {
            return Err(TransportError::closed(
                "channel closed with an incomplete frame",
            ));
        }
        Ok(closed)
    }

    /// Forms every whole frame the buffer now holds.
    fn split(&mut self) {
        while let Some(field) = self.buffer.first_chunk::<{ Frame::LENGTH_BYTES }>() {
            let whole = Frame::LENGTH_BYTES + Frame::message_len(*field);
            if self.buffer.len() < whole {
                return;
            }
            let frame = Frame::from_bytes(self.buffer.drain(..whole).collect());
            self.ready.push_back(frame);
        }
    }
}

/// Retains partial writes until the socket becomes writable.
///
/// The queue is not bounded here: the protocols a channel carries bound what
/// is outstanding on it, as a rank's queue depth bounds its batches and
/// results.
struct FrameWriter {
    /// Frames, each as its bytes, not yet fully written.
    pending: VecDeque<Vec<u8>>,
    /// Bytes of the front frame already written.
    offset: usize,
}

impl FrameWriter {
    fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            offset: 0,
        }
    }

    /// Writes queued bytes until the queue empties or the socket would block.
    fn flush(&mut self, stream: &mut Stream) -> Result<(), TransportError> {
        while let Some(frame) = self.pending.front() {
            match stream.write(&frame[self.offset..]) {
                Ok(0) => return Err(TransportError::closed("channel closed while sending")),
                Ok(count) => {
                    self.offset += count;
                    if self.offset == frame.len() {
                        self.pending.pop_front();
                        self.offset = 0;
                    }
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(()),
                Err(error) => return Err(TransportError::io("sending on a channel", error)),
            }
        }
        Ok(())
    }

    /// Poll interests for the stream: always readable, and writable only while
    /// bytes are queued, since an idle writable socket would end every poll
    /// at once.
    fn events(&self) -> i16 {
        libc::POLLIN
            | if self.pending.is_empty() {
                0
            } else {
                libc::POLLOUT
            }
    }
}

/// Parks on current readiness interests, preserving the deadline across
/// signals; without a timeout it parks until one of them is ready.
///
/// Returns `Ok(())` after any poll completes, whether it saw readiness or timed
/// out; callers re-check their own state rather than reading `revents`.
fn wait_ready(descriptors: &mut [libc::pollfd], timeout: Option<Duration>) -> IpcResult<()> {
    let deadline = timeout.map(|timeout| Instant::now() + timeout);
    loop {
        // Round up to whole milliseconds so poll does not return before the
        // deadline.
        let millis = match deadline {
            Some(deadline) => {
                let left = deadline.saturating_duration_since(Instant::now());
                let millis =
                    left.as_millis() + u128::from(!left.subsec_nanos().is_multiple_of(1_000_000));
                millis.min(i32::MAX as u128) as i32
            }
            None => -1,
        };
        // SAFETY: poll borrows this live array only for the duration of the call.
        let result = unsafe {
            libc::poll(
                descriptors.as_mut_ptr(),
                descriptors.len() as libc::nfds_t,
                millis,
            )
        };
        if result >= 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != ErrorKind::Interrupted {
            return Err(transport("waiting for a channel", error));
        }
    }
}

/// Fires a channel's local wake from any thread of the process that owns it.
///
/// A wake never travels between processes. It interrupts the owner's
/// [`Channel::wait`] for work that did not arrive on the socket, such as a
/// completion raised by another of the owner's threads. Wakes fired before
/// the owner takes them coalesce into one.
#[derive(Clone)]
pub struct Wake {
    /// Non-blocking end that every fire writes one byte to.
    writer: Arc<UnixStream>,
}

impl Wake {
    /// Fires one wake.
    pub fn wake(&self) {
        // A write refused because the buffer is full loses nothing: the
        // descriptor already holds unread bytes, so it is readable.
        let _ = (&*self.writer).write(&[1]);
    }
}

/// The owner's end of a local wake: the descriptor it polls and drains.
///
/// The descriptor's unread bytes are the wake's only state. Every fire writes
/// one byte and a drain consumes them all, so a wake is reported exactly when
/// its byte is read: one fired while the owner drains is either read and
/// reported by that drain or left readable for the owner's next poll.
struct LocalWake {
    /// Non-blocking end the owner polls and drains.
    reader: UnixStream,
    /// The firing end, shared with every [`Wake`] handed out.
    wake: Wake,
}

impl LocalWake {
    fn new() -> IpcResult<Self> {
        let (reader, writer) =
            UnixStream::pair().map_err(|error| transport("creating a local wake", error))?;
        for end in [&reader, &writer] {
            end.set_nonblocking(true)
                .map_err(|error| transport("making a local wake pollable", error))?;
        }
        Ok(Self {
            reader,
            wake: Wake {
                writer: Arc::new(writer),
            },
        })
    }

    /// Drains the descriptor and reports whether any wake had fired.
    fn take(&mut self) -> bool {
        let mut fired = false;
        let mut sink = [0u8; 64];
        while self.reader.read(&mut sink).is_ok_and(|read| read > 0) {
            fired = true;
        }
        fired
    }
}

/// One end of a connection that carries frames.
pub struct Channel {
    stream: Stream,
    reader: FrameReader,
    writer: FrameWriter,
    /// Terminal failure, reported after the frames that arrived whole have
    /// been taken: an IO failure or the peer's close.
    failure: Option<TransportError>,
    local: LocalWake,
}

impl Channel {
    /// Connects to a listening address.
    pub fn connect(address: &str) -> IpcResult<Self> {
        let failed = |error| transport(format!("connecting to {address}"), error);
        let stream = match address.strip_prefix(UNIX_PREFIX) {
            Some(path) => Stream::Unix(UnixStream::connect(path).map_err(failed)?),
            None => Stream::Tcp(TcpStream::connect(address).map_err(failed)?),
        };
        Self::over(stream)
    }

    /// Builds a channel over a connected stream.
    fn over(stream: Stream) -> IpcResult<Self> {
        Ok(Self {
            stream: stream.configure()?,
            reader: FrameReader::new(),
            writer: FrameWriter::new(),
            failure: None,
            local: LocalWake::new()?,
        })
    }

    /// Returns this end's address on a TCP connection, and `None` on a Unix
    /// socket.
    ///
    /// It is the address the peer reached this end at, which is how a process
    /// that binds every interface learns an address its peers route to.
    pub fn local_addr(&self) -> IpcResult<Option<SocketAddr>> {
        match &self.stream {
            Stream::Unix(_) => Ok(None),
            Stream::Tcp(stream) => stream
                .local_addr()
                .map(Some)
                .map_err(|error| transport("reading a channel's local address", error)),
        }
    }

    /// Returns the peer's address on a TCP connection, and `None` on a Unix
    /// socket.
    pub fn peer_addr(&self) -> IpcResult<Option<SocketAddr>> {
        match &self.stream {
            Stream::Unix(_) => Ok(None),
            Stream::Tcp(stream) => stream
                .peer_addr()
                .map(Some)
                .map_err(|error| transport("reading a channel's peer address", error)),
        }
    }

    /// Returns a handle that fires this channel's local wake.
    pub fn wake(&self) -> Wake {
        self.local.wake.clone()
    }

    /// Encodes and sends one message.
    ///
    /// Queues the frame and writes as far as the socket accepts; later calls
    /// write the rest. A message of any length a frame's length field can
    /// state is sent; nothing about a channel bounds it further. Fails with
    /// the recorded failure once the channel has failed, when the message
    /// cannot be encoded or is longer than a frame can state, and on a write
    /// error.
    pub fn send<M: Serialize>(&mut self, message: &M) -> IpcResult<()> {
        if let Some(error) = &self.failure {
            return Err(IpcError::Transport(error.clone()));
        }
        let frame = Frame::encode(message)?.into_bytes();
        self.writer.pending.push_back(frame);
        Ok(self.writer.flush(&mut self.stream)?)
    }

    /// Takes the next message, if one has arrived.
    ///
    /// Reads the stream when no frame is buffered and flushes queued writes
    /// while the connection is open. Fails when the frame taken does not
    /// decode to an `M`, and with the terminal failure once no whole frame
    /// remains buffered.
    pub fn try_recv<M: DeserializeOwned>(&mut self) -> IpcResult<Option<M>> {
        if self.reader.ready.is_empty() {
            self.pump();
        }
        if let Some(frame) = self.reader.ready.pop_front() {
            return frame.decode().map(Some);
        }
        match &self.failure {
            Some(error) => Err(IpcError::Transport(error.clone())),
            None => Ok(None),
        }
    }

    /// Blocks until something happens that the owner can act on, or
    /// `timeout` passes.
    ///
    /// It returns `Ok` when a message arrives, when a queued frame has been
    /// written out, when the local wake fires (a wake already pending is
    /// consumed and ends the wait at once), and at the timeout. It does not say which occurred, so the caller
    /// re-checks every progress source afterwards.
    ///
    /// Messages already buffered do not end the wait. A caller that can take
    /// a message calls [`Self::try_recv`] until it returns `None` before it
    /// waits; a caller that cannot take one right now, because it is busy,
    /// therefore still blocks here instead of spinning.
    ///
    /// Fails with the terminal failure once the connection has failed or
    /// been closed by the peer: at once when that is already known, and
    /// otherwise when it is observed without a new message arriving first.
    pub fn wait(&mut self, timeout: Duration) -> IpcResult<()> {
        if let Some(error) = &self.failure {
            return Err(IpcError::Transport(error.clone()));
        }
        if self.local.take() {
            return Ok(());
        }

        // A timeout too long to be a point in time is no deadline.
        let deadline = Instant::now().checked_add(timeout);
        let buffered = self.reader.ready.len();
        let queued = self.writer.pending.len();
        loop {
            let remaining =
                deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
            wait_ready(&mut self.poll_fds(), remaining)?;
            let woken = self.local.take();
            self.pump();

            // A message that arrived with the close is reported first; the
            // next wait reports the close.
            let arrived = self.reader.ready.len() > buffered;
            let sent = self.writer.pending.len() < queued;
            if woken || arrived || sent {
                return Ok(());
            }
            if let Some(error) = &self.failure {
                return Err(IpcError::Transport(error.clone()));
            }

            // Readiness that completed neither a frame nor a write, such as
            // part of a large message, is not progress the owner can use.
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Ok(());
            }
        }
    }

    /// Writes out every queued frame, waiting for the socket to accept them.
    ///
    /// Dropping a channel discards what is still queued, so an owner calls
    /// this before it closes a connection on purpose: without it a rank's
    /// last response, queued behind a large result the head is still
    /// reading, is lost and the head reports a truncated frame instead of
    /// the response. Fails when the connection fails, which also ends the
    /// wait on a peer that is gone.
    pub fn flush(&mut self) -> IpcResult<()> {
        loop {
            if let Some(error) = &self.failure {
                return Err(IpcError::Transport(error.clone()));
            }
            self.writer.flush(&mut self.stream)?;
            if self.writer.pending.is_empty() {
                return Ok(());
            }

            let mut writable = [libc::pollfd {
                fd: self.stream.as_raw_fd(),
                events: libc::POLLOUT,
                revents: 0,
            }];
            wait_ready(&mut writable, None)?;
        }
    }

    /// Returns an observer of the peer's close that does not read the
    /// connection, and that another thread may hold while this channel is in
    /// use.
    pub fn hangup(&self) -> IpcResult<Hangup> {
        let duplicate = |descriptor: std::os::fd::BorrowedFd<'_>| {
            descriptor
                .try_clone_to_owned()
                .map_err(|error| transport("duplicating a channel's descriptor", error))
        };
        let stream = match &self.stream {
            Stream::Unix(stream) => duplicate(std::os::fd::AsFd::as_fd(stream))?,
            Stream::Tcp(stream) => duplicate(std::os::fd::AsFd::as_fd(stream))?,
        };
        Ok(Hangup { stream })
    }

    /// Returns the descriptors whose readiness means this channel can make
    /// progress: the stream, writable only while a frame is queued, and the
    /// local wake.
    ///
    /// An owner of several channels polls them together and then calls
    /// [`Self::try_recv`] on each.
    pub fn poll_fds(&self) -> [libc::pollfd; 2] {
        [
            libc::pollfd {
                fd: self.stream.as_raw_fd(),
                events: self.writer.events(),
                revents: 0,
            },
            libc::pollfd {
                fd: self.local.reader.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ]
    }

    /// Reads what the stream holds and flushes queued writes, recording a
    /// failure instead of returning it so that buffered frames are delivered
    /// first.
    fn pump(&mut self) {
        if self.failure.is_some() {
            return;
        }
        self.failure = match self.reader.fill(&mut self.stream) {
            Ok(false) => self.writer.flush(&mut self.stream).err(),
            Ok(true) => Some(TransportError::closed("channel closed by its peer")),
            Err(error) => Some(error),
        };
    }
}

/// Observes whether a connection's peer has closed it, without reading it.
///
/// A channel notices its peer's close only when its owner calls it. An owner
/// whose thread is occupied elsewhere, for example while it loads a model,
/// gives a `Hangup` to a thread that can wait. The observer holds its own
/// descriptor of the connection, so it stays valid whatever happens to the
/// channel it came from.
pub struct Hangup {
    stream: std::os::fd::OwnedFd,
}

impl Hangup {
    /// Blocks until the peer has closed the connection or the connection has
    /// failed, and returns `true`; or until `cancel` becomes readable or is
    /// closed, and returns `false`.
    ///
    /// Data the peer sent and the channel has not read yet does not hide the
    /// close: the peer's hangup is reported by `poll` separately from
    /// readability (`POLLRDHUP`).
    pub fn wait(&self, cancel: &impl AsRawFd) -> IpcResult<bool> {
        #[cfg(target_os = "linux")]
        let closing = libc::POLLRDHUP;
        #[cfg(not(target_os = "linux"))]
        let closing = 0;

        let mut descriptors = [
            libc::pollfd {
                fd: self.stream.as_raw_fd(),
                events: closing,
                revents: 0,
            },
            libc::pollfd {
                fd: cancel.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        wait_ready(&mut descriptors, None)?;
        let closed = closing | libc::POLLHUP | libc::POLLERR | libc::POLLNVAL;
        Ok(descriptors[1].revents == 0 && descriptors[0].revents & closed != 0)
    }
}

/// One bound, listening socket of a [`Listener`].
enum Socket {
    /// A Unix socket and the path it is bound at, removed when it closes.
    Unix(UnixListener, PathBuf),
    Tcp(TcpListener),
}

impl Socket {
    fn bind(address: &str) -> IpcResult<Self> {
        let failed = |error| transport(format!("binding {address}"), error);
        match address.strip_prefix(UNIX_PREFIX) {
            Some(path) => {
                let listener = UnixListener::bind(path).map_err(failed)?;
                listener.set_nonblocking(true).map_err(failed)?;
                Ok(Self::Unix(listener, Path::new(path).to_owned()))
            }
            None => {
                let listener = TcpListener::bind(address).map_err(failed)?;
                listener.set_nonblocking(true).map_err(failed)?;
                Ok(Self::Tcp(listener))
            }
        }
    }

    /// Returns the address this socket is bound at, as peers write it.
    fn address(&self) -> IpcResult<String> {
        match self {
            Self::Unix(_, path) => Ok(format!("{UNIX_PREFIX}{}", path.display())),
            Self::Tcp(listener) => listener
                .local_addr()
                .map(|address: SocketAddr| address.to_string())
                .map_err(|error| transport("reading a listening address", error)),
        }
    }

    /// Takes one waiting connection without blocking.
    ///
    /// A connection whose peer reset it, or whose network failed, while it
    /// waited to be accepted is reported by `accept` as an error of the
    /// listener; it is skipped, as accept(2) prescribes, so that a stray
    /// client cannot fail whoever is accepting.
    fn accept(&self) -> IpcResult<Option<Stream>> {
        loop {
            let accepted = match self {
                Self::Unix(listener, _) => {
                    listener.accept().map(|(stream, _)| Stream::Unix(stream))
                }
                Self::Tcp(listener) => listener.accept().map(|(stream, _)| Stream::Tcp(stream)),
            };
            match accepted {
                Ok(stream) => return Ok(Some(stream)),
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
                Err(error) => return Err(transport("accepting a channel", error)),
            }
        }
    }
}

impl AsRawFd for Socket {
    fn as_raw_fd(&self) -> RawFd {
        match self {
            Self::Unix(listener, _) => listener.as_raw_fd(),
            Self::Tcp(listener) => listener.as_raw_fd(),
        }
    }
}

impl Drop for Socket {
    /// A Unix socket's path outlives its descriptor unless it is removed.
    fn drop(&mut self) {
        if let Self::Unix(_, path) = self {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Accepts connections on one or more addresses, each as a [`Channel`].
pub struct Listener {
    sockets: Vec<Socket>,
    /// The bound addresses, in the order they were given.
    addresses: Vec<String>,
}

impl Listener {
    /// Binds every address in `addresses`.
    ///
    /// A TCP address may name port zero, for which the system picks a free
    /// port that [`Self::addresses`] then reports. Fails when an address
    /// cannot be bound, for example a Unix socket path that exists.
    pub fn bind(addresses: &[&str]) -> IpcResult<Self> {
        let sockets = addresses
            .iter()
            .map(|address| Socket::bind(address))
            .collect::<IpcResult<Vec<_>>>()?;
        let addresses = sockets
            .iter()
            .map(Socket::address)
            .collect::<IpcResult<_>>()?;
        Ok(Self { sockets, addresses })
    }

    /// Returns the bound addresses, in the order they were given to
    /// [`Self::bind`].
    pub fn addresses(&self) -> &[String] {
        &self.addresses
    }

    /// Takes one waiting connection, if any, without blocking.
    pub fn try_accept(&self) -> IpcResult<Option<Channel>> {
        for socket in &self.sockets {
            if let Some(stream) = socket.accept()? {
                return Channel::over(stream).map(Some);
            }
        }
        Ok(None)
    }

    /// Returns the descriptors that become readable when a connection waits.
    pub fn poll_fds(&self) -> Vec<libc::pollfd> {
        self.sockets
            .iter()
            .map(|socket| libc::pollfd {
                fd: socket.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WorkerRequest, WorkerResponse, WorkerResponseError};

    const TIMEOUT: Duration = Duration::from_secs(5);

    /// An error response whose encoded message is `filler` bytes of text plus
    /// a fixed envelope.
    fn response_of(message_id: u64, filler: usize) -> WorkerResponse {
        let mut response = WorkerResponse::error(WorkerResponseError {
            message: "m".repeat(filler),
            code: None,
            fatal: false,
            phase: None,
            route: None,
            calls: Vec::new(),
        });
        response.set_message_id(Some(message_id));
        response
    }

    /// The bytes of the frame that carries `message`.
    fn frame_of<M: Serialize>(message: &M) -> Vec<u8> {
        Frame::encode(message).unwrap().into_bytes()
    }

    /// Takes the connection that is on its way to `listener`.
    fn accept(listener: &Listener) -> Channel {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if let Some(channel) = listener.try_accept().unwrap() {
                return channel;
            }
            assert!(Instant::now() < deadline, "no connection arrived");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// A channel and the raw stream of its peer, over loopback TCP.
    fn with_raw_peer() -> (Channel, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let channel = Channel::connect(&address).unwrap();
        let (peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(TIMEOUT)).unwrap();
        (channel, peer)
    }

    /// Waits for the next message, or returns the failure that ends the wait.
    fn recv<M: DeserializeOwned>(channel: &mut Channel) -> IpcResult<M> {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if let Some(message) = channel.try_recv()? {
                return Ok(message);
            }
            assert!(Instant::now() < deadline, "no message arrived");
            channel.wait(Duration::from_millis(50))?;
        }
    }

    /// Restricts `channel`'s socket send buffer so that a large frame spans
    /// several writes.
    fn restrict_send_buffer(channel: &Channel) {
        let capacity: libc::c_int = 4096;
        assert_eq!(
            // SAFETY: the descriptor is the channel's open socket, and the option
            // value points to the local `capacity`, whose size is the option length.
            unsafe {
                libc::setsockopt(
                    channel.stream.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_SNDBUF,
                    (&capacity as *const libc::c_int).cast(),
                    std::mem::size_of_val(&capacity) as libc::socklen_t,
                )
            },
            0
        );
    }

    #[test]
    fn messages_cross_unix_and_tcp_connections_in_both_directions() {
        let directory = tempfile::tempdir().unwrap();
        let unix = format!("unix:{}", directory.path().join("channel.sock").display());
        let listener = Listener::bind(&[&unix, "127.0.0.1:0"]).unwrap();
        assert_eq!(listener.addresses()[0], unix);

        for address in listener.addresses() {
            let mut near = Channel::connect(address).unwrap();
            let mut far = accept(&listener);

            // Only a TCP connection has addresses, and each end's local
            // address is the other's peer address.
            let tcp = !address.starts_with(UNIX_PREFIX);
            assert_eq!(near.local_addr().unwrap().is_some(), tcp);
            assert_eq!(near.local_addr().unwrap(), far.peer_addr().unwrap());
            assert_eq!(near.peer_addr().unwrap(), far.local_addr().unwrap());

            let mut request = WorkerRequest::info();
            request.set_message_id(Some(77));
            near.send(&request).unwrap();
            assert_eq!(recv::<WorkerRequest>(&mut far).unwrap(), request);

            let answer = response_of(77, 16);
            far.send(&answer).unwrap();
            assert_eq!(recv::<WorkerResponse>(&mut near).unwrap(), answer);
        }
    }

    #[test]
    fn a_unix_listener_removes_its_path_when_it_closes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("channel.sock");
        let address = format!("unix:{}", path.display());

        let listener = Listener::bind(&[&address]).unwrap();
        assert!(path.exists());
        assert!(Listener::bind(&[&address]).is_err());

        drop(listener);
        assert!(!path.exists());
    }

    #[test]
    fn completed_messages_survive_a_truncated_last_frame() {
        // The peer writes one complete frame a byte at a time, then a frame cut
        // short, then closes. The complete response must still be delivered;
        // only the truncated one fails, and the channel is then closed.
        let (mut channel, mut peer) = with_raw_peer();
        let first = response_of(1, 5);
        for byte in frame_of(&first) {
            peer.write_all(&[byte]).unwrap();
        }
        let second = frame_of(&response_of(2, 9));
        peer.write_all(&second[..second.len() - 2]).unwrap();
        peer.shutdown(std::net::Shutdown::Write).unwrap();

        assert_eq!(recv::<WorkerResponse>(&mut channel).unwrap(), first);
        let error = recv::<WorkerResponse>(&mut channel).expect_err("truncation must fail");
        assert!(error.to_string().contains("incomplete frame"), "{error}");
        assert!(channel.send(&WorkerRequest::info()).is_err());
    }

    #[test]
    fn a_frame_decodes_only_as_the_message_it_carries() {
        let (mut channel, mut peer) = with_raw_peer();
        peer.write_all(&frame_of(&WorkerRequest::info())).unwrap();
        assert!(matches!(
            recv::<WorkerResponse>(&mut channel),
            Err(IpcError::Encoding(_))
        ));
    }

    /// A wake ends a wait that no message would have ended, fires coalesce
    /// into the one wait that observes them, and the wake rearms afterwards.
    #[test]
    fn a_local_wake_interrupts_a_wait_once_and_rearms() {
        let (mut channel, _peer) = with_raw_peer();
        let wake = channel.wake();
        let long = Duration::from_secs(30);
        let short = Duration::from_millis(50);
        let waited = |channel: &mut Channel, timeout| {
            let started = Instant::now();
            channel.wait(timeout).unwrap();
            started.elapsed()
        };

        assert!(waited(&mut channel, short) >= short, "nothing had fired");

        wake.wake();
        wake.wake();
        assert!(
            waited(&mut channel, long) < long,
            "the wake did not end the wait"
        );
        assert!(
            waited(&mut channel, short) >= short,
            "one wait consumed both fires"
        );

        // A wake may fire from another thread, however many waits preceded it.
        for _ in 0..3 {
            let wake = wake.clone();
            std::thread::spawn(move || wake.wake()).join().unwrap();
            assert!(waited(&mut channel, long) < long);
        }
    }

    /// An owner that cannot take a message right now still blocks in `wait`:
    /// a message it has left buffered does not end the wait, and a new one
    /// does.
    #[test]
    fn a_buffered_message_does_not_end_a_wait_and_a_new_one_does() {
        let (mut channel, mut peer) = with_raw_peer();
        let short = Duration::from_millis(100);
        for message_id in [1, 2] {
            peer.write_all(&frame_of(&response_of(message_id, 5)))
                .unwrap();
        }
        assert_eq!(
            recv::<WorkerResponse>(&mut channel).unwrap(),
            response_of(1, 5)
        );

        // The second response stays buffered, and nothing else happens.
        let started = Instant::now();
        channel.wait(short).unwrap();
        assert!(
            started.elapsed() >= short,
            "a buffered message ended the wait"
        );

        let writer = std::thread::spawn(move || {
            std::thread::sleep(short);
            peer.write_all(&frame_of(&response_of(3, 5))).unwrap();
            peer
        });
        let started = Instant::now();
        channel.wait(TIMEOUT).unwrap();
        assert!(
            started.elapsed() < TIMEOUT,
            "a new message did not end the wait"
        );
        let _peer = writer.join().unwrap();
        for message_id in [2, 3] {
            assert_eq!(
                channel.try_recv::<WorkerResponse>().unwrap(),
                Some(response_of(message_id, 5))
            );
        }
    }

    /// A wait reports the peer's close as an error, after the message that
    /// arrived with it, and at once from then on, so a caller that waits on
    /// a closed channel neither blocks for its timeout nor spins.
    #[test]
    fn a_wait_fails_once_the_peer_has_closed() {
        let (mut channel, mut peer) = with_raw_peer();
        peer.write_all(&frame_of(&response_of(1, 5))).unwrap();
        drop(peer);

        // The last message and the close may be observed by one wait or two.
        let deadline = Instant::now() + TIMEOUT;
        while channel.wait(Duration::from_millis(50)).is_ok() {
            assert!(Instant::now() < deadline, "the close was never reported");
        }
        let started = Instant::now();
        assert!(matches!(channel.wait(TIMEOUT), Err(IpcError::Transport(_))));
        assert!(
            started.elapsed() < TIMEOUT,
            "a closed channel blocked a wait"
        );
        assert_eq!(
            channel.try_recv::<WorkerResponse>().unwrap(),
            Some(response_of(1, 5))
        );
    }

    /// A message queued behind a large, partly written one reaches the peer
    /// when the owner flushes before it closes the connection.
    #[test]
    fn a_flush_delivers_queued_messages_before_a_close() {
        let (mut channel, mut peer) = with_raw_peer();
        restrict_send_buffer(&channel);
        let large = response_of(1, 1 << 20);
        let last = response_of(2, 4);
        channel.send(&large).unwrap();
        channel.send(&last).unwrap();

        let reader = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            peer.read_to_end(&mut bytes).unwrap();
            bytes
        });
        channel.flush().unwrap();
        drop(channel);

        let mut expected = frame_of(&large);
        expected.extend(frame_of(&last));
        assert!(
            reader.join().unwrap() == expected,
            "the peer missed a message"
        );
    }

    /// A frame larger than the socket's send buffer is written across the
    /// owner's waits, each of which ends when the frame has been written out.
    #[test]
    fn a_large_message_is_written_across_waits() {
        let (mut channel, mut peer) = with_raw_peer();
        restrict_send_buffer(&channel);
        let large = response_of(1, 1 << 20);
        channel.send(&large).unwrap();

        let expected = frame_of(&large);
        let length = expected.len();
        let reader = std::thread::spawn(move || {
            let mut bytes = vec![0; length];
            peer.read_exact(&mut bytes).unwrap();
            (bytes, peer)
        });
        let started = Instant::now();
        channel.wait(TIMEOUT).unwrap();
        assert!(
            started.elapsed() < TIMEOUT,
            "the write did not end the wait"
        );
        let (wire, _peer) = reader.join().unwrap();
        assert!(wire == expected, "the frame arrived changed");
    }

    /// A message above 64 MiB, the size of one decoded MiniMax H3 unit,
    /// crosses a TCP connection intact: a channel carries any message its
    /// frame's length field can state.
    #[test]
    fn a_message_above_64_mib_crosses_a_tcp_connection_intact() {
        let listener = Listener::bind(&["127.0.0.1:0"]).unwrap();
        let mut near = Channel::connect(&listener.addresses()[0]).unwrap();
        let mut far = accept(&listener);
        let large = response_of(1, 72 << 20);
        let length = frame_of(&large).len();
        assert!(length > 64 << 20);

        // The sender writes the frame out while the receiver reads it, as
        // the head and a rank on another host do.
        let sent = large.clone();
        let sender = std::thread::spawn(move || {
            near.send(&sent).unwrap();
            near.flush().unwrap();
            near
        });
        let deadline = Instant::now() + Duration::from_secs(60);
        let received = loop {
            if let Some(message) = far.try_recv::<WorkerResponse>().unwrap() {
                break message;
            }
            assert!(
                Instant::now() < deadline,
                "the large message did not arrive"
            );
            far.wait(Duration::from_millis(50)).unwrap();
        };
        let _near = sender.join().unwrap();
        assert!(
            frame_of(&received).len() == length,
            "the frame arrived resized"
        );
        assert!(received == large, "the message arrived changed");
    }

    /// A hangup observer sees the peer's close without the channel being
    /// called; unread data from the peer does not hide or fake it, and its
    /// owner can cancel the wait.
    #[test]
    fn a_hangup_reports_the_peers_close_and_nothing_else() {
        let directory = tempfile::tempdir().unwrap();
        let unix = format!("unix:{}", directory.path().join("channel.sock").display());
        let listener = Listener::bind(&[&unix, "127.0.0.1:0"]).unwrap();
        for address in listener.addresses() {
            let channel = Channel::connect(address).unwrap();
            let mut peer = accept(&listener);
            let hangup = channel.hangup().unwrap();
            let (cancel, canceller) = UnixStream::pair().unwrap();

            peer.send(&WorkerRequest::info()).unwrap();
            let cancelling = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                drop(canceller);
            });
            assert!(
                !hangup.wait(&cancel).unwrap(),
                "unread data read as a close"
            );
            cancelling.join().unwrap();

            let (cancel, _canceller) = UnixStream::pair().unwrap();
            drop(peer);
            assert!(hangup.wait(&cancel).unwrap());
        }
    }
}
