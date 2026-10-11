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
//! A new channel carries messages of up to [`INITIAL_BOUND`] bytes, which is
//! enough for the first exchange on a connection, in which a peer says who it
//! is and learns what it is to do. Its owner then sets the capacity the
//! messages that follow need with [`Channel::set_capacity`].
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
//! probes an idle peer (see [`KEEPALIVE_IDLE_SECS`]), so a peer whose host
//! has died closes the connection like one that exited. [`Hangup`] observes
//! the peer's close without reading the connection, for an owner whose
//! thread is not in a position to call the channel.

use std::collections::VecDeque;
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{IpcError, IpcResult};
use crate::frame::{Frame, Message};

/// The prefix of a Unix socket address; the rest is the socket's path.
const UNIX_PREFIX: &str = "unix:";

/// Largest encoded message, in bytes, a channel carries until its owner
/// calls [`Channel::set_capacity`].
pub const INITIAL_BOUND: usize = 1 << 20;

/// Frames a channel's send queue holds until its owner calls
/// [`Channel::set_capacity`].
const INITIAL_DEPTH: usize = 4;

/// Seconds a TCP connection stays idle before it probes its peer.
pub const KEEPALIVE_IDLE_SECS: libc::c_int = 30;

/// Seconds between the probes of an idle TCP connection.
pub const KEEPALIVE_INTERVAL_SECS: libc::c_int = 10;

/// Unanswered probes after which a TCP connection fails. With the idle time
/// and the interval, a peer whose host died is noticed within a minute.
pub const KEEPALIVE_PROBES: libc::c_int = 3;

/// Milliseconds sent data may stay unacknowledged before a TCP connection
/// fails, which bounds a send to a dead peer as the probes bound an idle one.
pub const USER_TIMEOUT_MILLIS: libc::c_int = 60_000;

/// Wraps an IO failure with what the channel was doing.
fn transport(action: &str, error: std::io::Error) -> IpcError {
    IpcError::Transport(format!("{action}: {error}"))
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

    set(libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1)?;
    #[cfg(target_os = "linux")]
    {
        set(libc::IPPROTO_TCP, libc::TCP_KEEPIDLE, KEEPALIVE_IDLE_SECS)?;
        set(
            libc::IPPROTO_TCP,
            libc::TCP_KEEPINTVL,
            KEEPALIVE_INTERVAL_SECS,
        )?;
        set(libc::IPPROTO_TCP, libc::TCP_KEEPCNT, KEEPALIVE_PROBES)?;
        set(
            libc::IPPROTO_TCP,
            libc::TCP_USER_TIMEOUT,
            USER_TIMEOUT_MILLIS,
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
    /// Returns whether the peer closed the connection. Fails on a read error,
    /// on a length field above `bound`, and on a close that leaves a partial
    /// frame; frames completed before the failure stay in `ready`.
    fn fill(&mut self, stream: &mut Stream, bound: usize) -> IpcResult<bool> {
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
                    self.split(bound)?;
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(transport("reading a channel", error)),
            }
        }
        if closed && !self.buffer.is_empty() {
            return Err(IpcError::Transport(
                "channel closed with an incomplete frame".into(),
            ));
        }
        Ok(closed)
    }

    /// Forms every whole frame the buffer now holds.
    fn split(&mut self, bound: usize) -> IpcResult<()> {
        loop {
            let Some(field) = self.buffer.first_chunk::<{ Frame::LENGTH_BYTES }>() else {
                return Ok(());
            };

            // Check the length as soon as its field is complete, before
            // waiting for the message, so an over-bound length fails without
            // buffering the bytes it announces.
            let whole = Frame::LENGTH_BYTES + Frame::message_len(*field, bound)?;
            if self.buffer.len() < whole {
                return Ok(());
            }
            let frame = Frame::from_bytes(self.buffer.drain(..whole).collect(), bound)?;
            self.ready.push_back(frame);
        }
    }
}

/// Retains partial writes until the socket becomes writable. Queue capacity is
/// the channel's depth, so a stalled peer cannot grow storage use without bound.
struct FrameWriter {
    /// Frames, each as its bytes, not yet fully written.
    pending: VecDeque<Vec<u8>>,
    /// Bytes of the front frame already written.
    offset: usize,
    /// Maximum queued frames, counting a partially written one.
    depth: usize,
}

impl FrameWriter {
    fn new(depth: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            offset: 0,
            depth: depth.max(1),
        }
    }

    /// Queues one frame.
    ///
    /// Fails with [`IpcError::WouldBlock`] when `depth` frames are already
    /// queued.
    fn enqueue(&mut self, frame: Frame) -> IpcResult<()> {
        if self.pending.len() >= self.depth {
            return Err(IpcError::WouldBlock);
        }
        self.pending.push_back(frame.into_bytes());
        Ok(())
    }

    /// Writes queued bytes until the queue empties or the socket would block.
    fn flush(&mut self, stream: &mut Stream) -> IpcResult<()> {
        while let Some(frame) = self.pending.front() {
            match stream.write(&frame[self.offset..]) {
                Ok(0) => {
                    return Err(IpcError::Transport("channel closed while sending".into()));
                }
                Ok(count) => {
                    self.offset += count;
                    if self.offset == frame.len() {
                        self.pending.pop_front();
                        self.offset = 0;
                    }
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(()),
                Err(error) => return Err(transport("sending on a channel", error)),
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

/// Parks on current readiness interests, preserving the deadline across signals.
///
/// Returns `Ok(())` after any poll completes, whether it saw readiness or timed
/// out; callers re-check their own state rather than reading `revents`.
fn wait_ready(descriptors: &mut [libc::pollfd], timeout: Duration) -> IpcResult<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());

        // Round up to whole milliseconds so poll does not return before the
        // deadline.
        let millis = left.as_millis() + u128::from(!left.subsec_nanos().is_multiple_of(1_000_000));
        // SAFETY: poll borrows this live array only for the duration of the call.
        let result = unsafe {
            libc::poll(
                descriptors.as_mut_ptr(),
                descriptors.len() as libc::nfds_t,
                millis.min(i32::MAX as u128) as i32,
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
    /// Largest encoded message either direction may carry.
    bound: usize,
    /// Terminal failure, reported after the frames that arrived whole have
    /// been taken: an IO failure or the peer's close.
    failure: Option<String>,
    local: LocalWake,
}

impl Channel {
    /// Connects to a listening address.
    ///
    /// `timeout` bounds a TCP connect; a Unix socket connects or fails at
    /// once.
    pub fn connect(address: &str, timeout: Duration) -> IpcResult<Self> {
        let stream = match address.strip_prefix(UNIX_PREFIX) {
            Some(path) => Stream::Unix(
                UnixStream::connect(path)
                    .map_err(|error| transport(&format!("connecting to {address}"), error))?,
            ),
            None => Stream::Tcp(connect_tcp(address, timeout)?),
        };
        Self::over(stream)
    }

    /// Builds a channel over a connected stream.
    fn over(stream: Stream) -> IpcResult<Self> {
        Ok(Self {
            stream: stream.configure()?,
            reader: FrameReader::new(),
            writer: FrameWriter::new(INITIAL_DEPTH),
            bound: INITIAL_BOUND,
            failure: None,
            local: LocalWake::new()?,
        })
    }

    /// Sets the capacity the messages from here on need: `bound` is the
    /// largest encoded message in bytes in either direction, and `depth` the
    /// number of frames the send queue holds.
    ///
    /// Both ends of a connection must set the same bound, since each end
    /// refuses a longer message when it sends and when it receives.
    pub fn set_capacity(&mut self, bound: usize, depth: usize) {
        self.bound = bound;
        self.writer.depth = depth.max(1);
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
    /// Queues the frame and writes as far as the socket accepts. Fails with
    /// the recorded failure once the channel has failed, when the message is
    /// invalid or its encoding exceeds the bound, with
    /// [`IpcError::WouldBlock`] when the send queue is full, and on a write
    /// error.
    pub fn send<M: Message>(&mut self, message: &M) -> IpcResult<()> {
        if let Some(error) = &self.failure {
            return Err(IpcError::Transport(error.clone()));
        }
        let frame = Frame::encode(message, self.bound)?;

        // Flush first so frames the socket has since accepted free queue
        // capacity before the depth check.
        self.writer.flush(&mut self.stream)?;
        self.writer.enqueue(frame)?;
        self.writer.flush(&mut self.stream)
    }

    /// Takes the next message, if one has arrived.
    ///
    /// Reads the stream when no frame is buffered and flushes queued writes
    /// while the connection is open. Fails when the frame taken does not
    /// decode to a valid `M`, and with the terminal failure once no whole
    /// frame remains buffered.
    pub fn try_recv<M: Message>(&mut self) -> IpcResult<Option<M>> {
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
    /// written out so that the send queue has room again, when the local wake
    /// fires (a wake already pending is consumed and ends the wait at once),
    /// and at the timeout. It does not say which occurred, so the caller
    /// re-checks every progress source afterwards.
    ///
    /// Messages already buffered do not end the wait. A caller that can take
    /// a message calls [`Self::try_recv`] until it returns `None` before it
    /// waits; a caller that cannot take one right now, because it is busy or
    /// its own queue is full, therefore still blocks here instead of
    /// spinning.
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

        let deadline = Instant::now() + timeout;
        let buffered = self.reader.ready.len();
        let queued = self.writer.pending.len();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
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
            if Instant::now() >= deadline {
                return Ok(());
            }
        }
    }

    /// Writes out every queued frame, waiting up to `timeout` for the socket
    /// to accept them.
    ///
    /// An owner calls this before it closes a connection on purpose, so that
    /// its last messages, such as an acknowledgment queued behind a large
    /// result, reach the peer. Fails when the connection has failed, on a
    /// write error, and when frames remain queued at the deadline.
    pub fn flush(&mut self, timeout: Duration) -> IpcResult<()> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(error) = &self.failure {
                return Err(IpcError::Transport(error.clone()));
            }
            self.writer.flush(&mut self.stream)?;
            if self.writer.pending.is_empty() {
                return Ok(());
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(IpcError::Transport(format!(
                    "{} frames were still queued when the channel closed",
                    self.writer.pending.len()
                )));
            }
            let mut writable = [libc::pollfd {
                fd: self.stream.as_raw_fd(),
                events: libc::POLLOUT,
                revents: 0,
            }];
            wait_ready(&mut writable, remaining)?;
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

    /// Returns whether the connection has failed or been closed by the peer.
    ///
    /// Frames that arrived whole before the close may still be buffered.
    pub fn is_closed(&self) -> bool {
        self.failure.is_some()
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
        self.failure = match self.reader.fill(&mut self.stream, self.bound) {
            Ok(false) => self
                .writer
                .flush(&mut self.stream)
                .err()
                .map(|error| error.to_string()),
            Ok(true) => Some("channel closed by its peer".into()),
            Err(error) => Some(error.to_string()),
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
    /// Waits up to `timeout` and returns whether the peer has closed the
    /// connection or the connection has failed.
    ///
    /// Data the peer sent and the channel has not read yet does not hide the
    /// close: the peer's hangup is reported by `poll` separately from
    /// readability (`POLLRDHUP`).
    pub fn wait(&self, timeout: Duration) -> IpcResult<bool> {
        #[cfg(target_os = "linux")]
        let closing = libc::POLLRDHUP;
        #[cfg(not(target_os = "linux"))]
        let closing = 0;

        let mut descriptor = [libc::pollfd {
            fd: self.stream.as_raw_fd(),
            events: closing,
            revents: 0,
        }];
        wait_ready(&mut descriptor, timeout)?;
        let closed = closing | libc::POLLHUP | libc::POLLERR | libc::POLLNVAL;
        Ok(descriptor[0].revents & closed != 0)
    }
}

/// Connects to the first reachable address `address` resolves to.
fn connect_tcp(address: &str, timeout: Duration) -> IpcResult<TcpStream> {
    let action = format!("connecting to {address}");
    let mut failure = std::io::Error::new(
        ErrorKind::AddrNotAvailable,
        "the address resolves to no endpoint",
    );
    let targets = address
        .to_socket_addrs()
        .map_err(|error| transport(&action, error))?;
    for target in targets {
        match TcpStream::connect_timeout(&target, timeout) {
            Ok(stream) => return Ok(stream),
            Err(error) => failure = error,
        }
    }
    Err(transport(&action, failure))
}

/// One bound, listening socket of a [`Listener`].
enum Socket {
    /// A Unix socket and the path it is bound at, removed when it closes.
    Unix(UnixListener, PathBuf),
    Tcp(TcpListener),
}

impl Socket {
    fn bind(address: &str) -> IpcResult<Self> {
        let failed = |error| transport(&format!("binding {address}"), error);
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
    fn accept(&self) -> IpcResult<Option<Stream>> {
        let accepted = match self {
            Self::Unix(listener, _) => listener.accept().map(|(stream, _)| Stream::Unix(stream)),
            Self::Tcp(listener) => listener.accept().map(|(stream, _)| Stream::Tcp(stream)),
        };
        match accepted {
            Ok(stream) => Ok(Some(stream)),
            Err(error) if error.kind() == ErrorKind::WouldBlock => Ok(None),
            Err(error) => Err(transport("accepting a channel", error)),
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

    /// Waits up to `timeout` for a connection and takes it.
    ///
    /// Returns `Ok(None)` when none arrived in time.
    pub fn accept(&self, timeout: Duration) -> IpcResult<Option<Channel>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(channel) = self.try_accept()? {
                return Ok(Some(channel));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            wait_ready(&mut self.poll_fds(), remaining)?;
        }
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
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
        response.set_call_id(Some(message_id));
        response
    }

    /// The bytes of the frame that carries `message`.
    fn frame_of<M: Message>(message: &M) -> Vec<u8> {
        Frame::encode(message, usize::MAX).unwrap().into_bytes()
    }

    /// A channel and the raw stream of its peer, over loopback TCP.
    fn with_raw_peer(bound: usize, depth: usize) -> (Channel, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let mut channel = Channel::connect(&address, TIMEOUT).unwrap();
        channel.set_capacity(bound, depth);
        let (peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(TIMEOUT)).unwrap();
        (channel, peer)
    }

    /// Waits for the next message, or returns the failure that ends the wait.
    fn recv<M: Message>(channel: &mut Channel) -> IpcResult<M> {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if let Some(message) = channel.try_recv()? {
                return Ok(message);
            }
            assert!(Instant::now() < deadline, "no message arrived");
            channel.wait(Duration::from_millis(50))?;
        }
    }

    #[test]
    fn messages_cross_unix_and_tcp_connections_in_both_directions() {
        let directory = tempfile::tempdir().unwrap();
        let unix = format!("unix:{}", directory.path().join("channel.sock").display());
        let listener = Listener::bind(&[&unix, "127.0.0.1:0"]).unwrap();
        assert_eq!(listener.addresses()[0], unix);

        for address in listener.addresses() {
            let mut near = Channel::connect(address, TIMEOUT).unwrap();
            let mut far = listener.accept(TIMEOUT).unwrap().expect("a connection");

            // Only a TCP connection has addresses, and each end's local
            // address is the other's peer address.
            let tcp = !address.starts_with(UNIX_PREFIX);
            assert_eq!(near.local_addr().unwrap().is_some(), tcp);
            assert_eq!(near.local_addr().unwrap(), far.peer_addr().unwrap());
            assert_eq!(near.peer_addr().unwrap(), far.local_addr().unwrap());

            let mut request = WorkerRequest::info();
            request.set_call_id(Some(77));
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
        // only the truncated one fails, and the channel then reports the close.
        let (mut channel, mut peer) = with_raw_peer(1024, 2);
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
        assert!(channel.is_closed());
        assert!(channel.send(&WorkerRequest::info()).is_err());
    }

    #[test]
    fn a_peer_that_closes_ends_the_channel_after_its_last_message() {
        let (mut channel, mut peer) = with_raw_peer(1024, 2);
        let last = response_of(3, 5);
        peer.write_all(&frame_of(&last)).unwrap();
        drop(peer);

        assert_eq!(recv::<WorkerResponse>(&mut channel).unwrap(), last);
        assert!(recv::<WorkerResponse>(&mut channel).is_err());
        assert!(channel.is_closed());
    }

    #[test]
    fn oversized_and_undecodable_frames_are_refused() {
        // Only the length field is sent: a length above the bound must be
        // refused before any of the message arrives.
        let (mut channel, mut peer) = with_raw_peer(1024, 1);
        peer.write_all(&1025_u32.to_le_bytes()).unwrap();
        assert!(recv::<WorkerResponse>(&mut channel).is_err());
        assert!(channel.is_closed());

        // A whole frame that holds a request is not a response.
        let (mut channel, mut peer) = with_raw_peer(1024, 1);
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
        let (mut channel, _peer) = with_raw_peer(1024, 1);
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
        let (mut channel, mut peer) = with_raw_peer(1024, 1);
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
        let (mut channel, mut peer) = with_raw_peer(1024, 1);
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

    /// Restricts `channel`'s socket send buffer so that a large frame spans
    /// several writes.
    fn restrict_send_buffer(channel: &Channel) {
        let capacity: libc::c_int = 4096;
        assert_eq!(
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

    /// A message queued behind a large, partly written one reaches the peer
    /// when the owner flushes before it closes the connection.
    #[test]
    fn a_flush_delivers_queued_messages_before_a_close() {
        let size = 1 << 20;
        let (mut channel, mut peer) = with_raw_peer(2 * size, 4);
        restrict_send_buffer(&channel);
        let large = response_of(1, size);
        let last = response_of(2, 4);
        channel.send(&large).unwrap();
        channel.send(&last).unwrap();

        let reader = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            peer.read_to_end(&mut bytes).unwrap();
            bytes
        });
        channel.flush(TIMEOUT).unwrap();
        drop(channel);

        let mut expected = frame_of(&large);
        expected.extend(frame_of(&last));
        assert!(
            reader.join().unwrap() == expected,
            "the peer missed a message"
        );
    }

    #[test]
    fn a_flush_fails_at_its_deadline_when_the_peer_does_not_read() {
        let size = 4 << 20;
        let (mut channel, _peer) = with_raw_peer(2 * size, 4);
        restrict_send_buffer(&channel);
        channel.send(&response_of(1, size)).unwrap();
        assert!(channel.flush(Duration::from_millis(100)).is_err());
    }

    /// A hangup observer sees the peer's close without the channel being
    /// called, and unread data from the peer does not hide or fake it.
    #[test]
    fn a_hangup_reports_the_peers_close_and_nothing_else() {
        for address in ["tcp", "unix"] {
            let directory = tempfile::tempdir().unwrap();
            let address = match address {
                "unix" => format!("unix:{}", directory.path().join("channel.sock").display()),
                _ => "127.0.0.1:0".to_owned(),
            };
            let listener = Listener::bind(&[&address]).unwrap();
            let channel = Channel::connect(&listener.addresses()[0], TIMEOUT).unwrap();
            let mut peer = listener.accept(TIMEOUT).unwrap().expect("a connection");
            let hangup = channel.hangup().unwrap();

            peer.send(&WorkerRequest::info()).unwrap();
            assert!(!hangup.wait(Duration::from_millis(100)).unwrap());

            drop(peer);
            assert!(hangup.wait(TIMEOUT).unwrap());
        }
    }

    #[test]
    fn a_stalled_peer_applies_backpressure_and_partial_writes_resume() {
        let size = 1 << 20;
        let (mut channel, mut peer) = with_raw_peer(2 * size, 1);
        restrict_send_buffer(&channel);
        // With depth 1, the partially written first frame fills the queue.
        let first = response_of(1, size);
        channel.send(&first).unwrap();
        assert!(matches!(
            channel.send(&response_of(2, 4)),
            Err(IpcError::WouldBlock)
        ));

        // Once the peer reads, driving the owner's wait loop resumes the
        // partial write until the whole frame arrives.
        let expected = frame_of(&first);
        let length = expected.len();
        let (tx, rx) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let mut bytes = vec![0; length];
            peer.read_exact(&mut bytes).unwrap();
            tx.send(bytes).unwrap();
        });
        let deadline = Instant::now() + TIMEOUT;
        let wire = loop {
            if let Ok(bytes) = rx.try_recv() {
                break bytes;
            }
            assert!(
                Instant::now() < deadline,
                "accepted frame did not finish sending"
            );
            channel.wait(Duration::from_millis(10)).unwrap();
        };
        reader.join().unwrap();
        assert!(wire == expected, "the frame arrived changed");
    }
}
