//! One end of a connection that carries the interfaces' messages.
//!
//! A frame is a little-endian `u32` length followed by one Protobuf message.
//! An address is written `unix:<path>` for a Unix socket and `<host>:<port>`
//! for TCP. A connection is non-blocking: [`Connection::try_recv`] returns a
//! message once one arrived whole, and [`Connection::wait`] blocks until the
//! socket is readable. Sends complete before they return. A TCP connection
//! probes its peer, so a node that died closes the connection like a process
//! that exited.
//!
//! Over a Unix socket a frame may carry a file descriptor beside it
//! ([`Connection::send_with_descriptor`]): the descriptor travels with the
//! frame's first byte, so it has arrived once the frame has, and the
//! receiver takes it after the message that announces it
//! ([`Connection::take_descriptor`]).

use std::collections::VecDeque;
use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::time::Duration;

use prost::Message;

/// The prefix of a Unix socket address; the rest is the socket's path.
const UNIX_PREFIX: &str = "unix:";

/// The largest frame a peer may send. The interfaces' messages are small;
/// a larger length is a corrupt stream.
const MAX_FRAME: usize = 64 << 20;

/// The most descriptors one receive takes; a frame carries at most one, and
/// a receive may take the bytes of several frames.
const MAX_DESCRIPTORS: usize = 16;

/// Bytes of the control messages one receive takes: a control header (16
/// bytes on 64-bit Linux) and [`MAX_DESCRIPTORS`] descriptors.
const CONTROL_BYTES: usize = 16 + MAX_DESCRIPTORS * std::mem::size_of::<RawFd>();

/// Seconds after which a TCP connection whose peer stopped answering fails.
const DEAD_PEER_SECS: libc::c_int = 60;

/// Why a connection failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The peer closed the connection.
    #[error("the peer closed the connection")]
    Closed,
    /// The socket failed.
    #[error("{action}: {source}")]
    Io {
        action: &'static str,
        #[source]
        source: std::io::Error,
    },
    /// A frame holds no message of the expected type.
    #[error("undecodable message: {0}")]
    Decode(#[from] prost::DecodeError),
    /// A frame's length exceeds what any message takes.
    #[error("a frame of {0} bytes exceeds the interface's bound")]
    Oversized(usize),
}

/// The result of a connection's operations.
pub type Result<T> = std::result::Result<T, Error>;

fn io(action: &'static str) -> impl FnOnce(std::io::Error) -> Error {
    move |source| Error::Io { action, source }
}

enum Stream {
    Unix(UnixStream),
    Tcp(TcpStream),
}

impl Stream {
    fn as_raw_fd(&self) -> RawFd {
        match self {
            Self::Unix(stream) => stream.as_raw_fd(),
            Self::Tcp(stream) => stream.as_raw_fd(),
        }
    }

    /// Reads into `buffer`; over a Unix socket, also takes the descriptors
    /// that arrived with the bytes read.
    fn read(
        &mut self,
        buffer: &mut [u8],
        descriptors: &mut VecDeque<OwnedFd>,
    ) -> std::io::Result<usize> {
        match self {
            Self::Unix(stream) => receive_with_descriptors(stream, buffer, descriptors),
            Self::Tcp(stream) => stream.read(buffer),
        }
    }

    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Unix(stream) => stream.write(buffer),
            Self::Tcp(stream) => stream.write(buffer),
        }
    }
}

/// One end of a connection.
pub struct Connection {
    stream: Stream,
    /// Bytes received and not yet taken as frames.
    received: Vec<u8>,
    /// Set once the peer closed; frames that arrived whole before are still
    /// returned.
    closed: bool,
    /// Descriptors received and not yet taken, in arrival order.
    descriptors: VecDeque<OwnedFd>,
}

impl Connection {
    /// Connects to `address`: `unix:<path>` or `<host>:<port>`.
    ///
    /// # Errors
    ///
    /// Fails when nothing listens at the address.
    pub fn connect(address: &str) -> Result<Self> {
        let stream = match address.strip_prefix(UNIX_PREFIX) {
            Some(path) => Stream::Unix(UnixStream::connect(path).map_err(io("connecting"))?),
            None => Stream::Tcp(TcpStream::connect(address).map_err(io("connecting"))?),
        };
        Self::from_stream(stream)
    }

    /// Wraps one end of a connected Unix socket pair.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be made non-blocking.
    pub fn from_unix(stream: UnixStream) -> Result<Self> {
        Self::from_stream(Stream::Unix(stream))
    }

    /// Borrows this connection's socket through a separate sending handle.
    /// The original remains the sole reader and retains all buffered frames
    /// and received descriptors. Concurrent sends require caller ordering.
    ///
    /// # Errors
    ///
    /// Fails when the socket handle cannot be duplicated or configured.
    pub fn writer(&self) -> Result<Self> {
        let stream = match &self.stream {
            Stream::Unix(stream) => Stream::Unix(stream.try_clone().map_err(io("duplicating"))?),
            Stream::Tcp(stream) => Stream::Tcp(stream.try_clone().map_err(io("duplicating"))?),
        };
        Self::from_stream(stream)
    }

    fn from_stream(stream: Stream) -> Result<Self> {
        match &stream {
            Stream::Unix(stream) => stream.set_nonblocking(true),
            Stream::Tcp(stream) => {
                stream.set_nodelay(true).map_err(io("disabling Nagle"))?;
                keep_alive(stream)?;
                stream.set_nonblocking(true)
            }
        }
        .map_err(io("making a connection pollable"))?;
        Ok(Self {
            stream,
            received: Vec::new(),
            closed: false,
            descriptors: VecDeque::new(),
        })
    }

    /// Sends `message` as one frame, waiting until the socket accepted all
    /// of it.
    ///
    /// # Errors
    ///
    /// Fails when the peer closed or the socket failed.
    pub fn send(&mut self, message: &impl Message) -> Result<()> {
        let frame = frame(message);
        self.write_from(&frame, 0)
    }

    /// Sends `message` as one frame that carries `descriptor` beside it,
    /// waiting until the socket accepted all of it. The peer receives its
    /// own descriptor of the same open file.
    ///
    /// # Errors
    ///
    /// Fails over TCP, which carries no descriptors, when the peer closed or
    /// the socket failed.
    pub fn send_with_descriptor(
        &mut self,
        message: &impl Message,
        descriptor: BorrowedFd<'_>,
    ) -> Result<()> {
        let Stream::Unix(stream) = &self.stream else {
            return Err(io("passing a descriptor")(std::io::Error::new(
                ErrorKind::Unsupported,
                "descriptors pass over Unix sockets only",
            )));
        };
        let frame = frame(message);
        // The descriptor travels with the first bytes the socket accepts.
        let written = loop {
            match send_with_descriptor(stream, &frame, descriptor) {
                Ok(0) => return Err(Error::Closed),
                Ok(count) => break count,
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    poll(stream.as_raw_fd(), libc::POLLOUT, None).map_err(io("waiting to send"))?;
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) if error.kind() == ErrorKind::BrokenPipe => return Err(Error::Closed),
                Err(error) => return Err(io("sending")(error)),
            }
        };
        self.write_from(&frame, written)
    }

    /// Takes the next descriptor a received frame carried, after the message
    /// that announces it; `None` when none arrived.
    pub fn take_descriptor(&mut self) -> Option<OwnedFd> {
        self.descriptors.pop_front()
    }

    /// Writes `frame` from byte `written` on, waiting while the socket is
    /// full.
    fn write_from(&mut self, frame: &[u8], mut written: usize) -> Result<()> {
        while written < frame.len() {
            match self.stream.write(&frame[written..]) {
                Ok(0) => return Err(Error::Closed),
                Ok(count) => written += count,
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    poll(self.stream.as_raw_fd(), libc::POLLOUT, None)
                        .map_err(io("waiting to send"))?;
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) if error.kind() == ErrorKind::BrokenPipe => return Err(Error::Closed),
                Err(error) => return Err(io("sending")(error)),
            }
        }
        Ok(())
    }

    /// Returns the next message once it arrived whole, and `None` while it
    /// has not.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Closed`] once the peer closed and every frame
    /// that arrived before was returned, and when a frame does not decode.
    pub fn try_recv<M: Message + Default>(&mut self) -> Result<Option<M>> {
        if let Some(message) = self.take_frame()? {
            return Ok(Some(message));
        }
        if !self.closed {
            let mut chunk = [0_u8; 64 * 1024];
            loop {
                match self.stream.read(&mut chunk, &mut self.descriptors) {
                    Ok(0) => {
                        self.closed = true;
                        break;
                    }
                    Ok(count) => self.received.extend_from_slice(&chunk[..count]),
                    Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                    Err(error) if error.kind() == ErrorKind::Interrupted => {}
                    Err(error) if error.kind() == ErrorKind::ConnectionReset => {
                        self.closed = true;
                        break;
                    }
                    Err(error) => return Err(io("receiving")(error)),
                }
            }
        }
        match self.take_frame()? {
            Some(message) => Ok(Some(message)),
            None if self.closed => Err(Error::Closed),
            None => Ok(None),
        }
    }

    /// Waits up to `timeout` for the next message.
    ///
    /// # Errors
    ///
    /// As [`Connection::try_recv`].
    pub fn recv<M: Message + Default>(&mut self, timeout: Duration) -> Result<Option<M>> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(message) = self.try_recv()? {
                return Ok(Some(message));
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            self.wait(remaining)?;
        }
    }

    /// Waits up to `timeout` until the socket is readable or closed.
    ///
    /// # Errors
    ///
    /// Fails when the wait fails.
    pub fn wait(&self, timeout: Duration) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        poll(self.stream.as_raw_fd(), libc::POLLIN, Some(timeout)).map_err(io("waiting"))?;
        Ok(())
    }

    /// The descriptor that becomes readable when a message arrives or the
    /// peer closes, for an owner that waits on several connections.
    pub fn descriptor(&self) -> RawFd {
        self.stream.as_raw_fd()
    }

    /// The peer's IP address over TCP; `None` over a Unix socket.
    pub fn peer_ip(&self) -> Option<IpAddr> {
        match &self.stream {
            Stream::Tcp(stream) => stream.peer_addr().ok().map(|address| address.ip()),
            Stream::Unix(_) => None,
        }
    }

    /// This end's IP address over TCP, the address the peer reached this
    /// node at; `None` over a Unix socket.
    pub fn local_ip(&self) -> Option<IpAddr> {
        match &self.stream {
            Stream::Tcp(stream) => stream.local_addr().ok().map(|address| address.ip()),
            Stream::Unix(_) => None,
        }
    }

    /// Takes one whole frame from the received bytes.
    fn take_frame<M: Message + Default>(&mut self) -> Result<Option<M>> {
        let Some(header) = self.received.get(..4) else {
            return Ok(None);
        };
        let length = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        if length > MAX_FRAME {
            return Err(Error::Oversized(length));
        }
        if self.received.len() < 4 + length {
            return Ok(None);
        }
        let message = M::decode(&self.received[4..4 + length])?;
        self.received.drain(..4 + length);
        Ok(Some(message))
    }
}

/// One frame of `message`: its length, then its encoding.
fn frame(message: &impl Message) -> Vec<u8> {
    let body = message.encode_to_vec();
    let mut frame = Vec::with_capacity(4 + body.len());
    frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
    frame.extend_from_slice(&body);
    frame
}

/// The control message buffer of up to [`MAX_DESCRIPTORS`] descriptors,
/// aligned as `cmsghdr` requires.
#[repr(C, align(8))]
struct ControlBuffer([u8; CONTROL_BYTES]);

/// Sends the start of `bytes` with `descriptor` attached, returning how many
/// bytes the socket accepted.
fn send_with_descriptor(
    stream: &UnixStream,
    bytes: &[u8],
    descriptor: BorrowedFd<'_>,
) -> std::io::Result<usize> {
    let mut control = ControlBuffer([0; CONTROL_BYTES]);
    let mut iov = libc::iovec {
        iov_base: bytes.as_ptr() as *mut libc::c_void,
        iov_len: bytes.len(),
    };
    // SAFETY: CMSG_SPACE only computes a length.
    let space = unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) } as usize;
    // SAFETY: an all-zero `msghdr` is a valid value to fill in.
    let mut header: libc::msghdr = unsafe { std::mem::zeroed() };
    header.msg_iov = &mut iov;
    header.msg_iovlen = 1;
    header.msg_control = control.0.as_mut_ptr().cast();
    header.msg_controllen = space as _;
    // SAFETY: the header names a control buffer of `space` bytes, room for
    // one descriptor, so the first control header lies inside it and its
    // data holds one `RawFd`.
    unsafe {
        let message = libc::CMSG_FIRSTHDR(&header);
        (*message).cmsg_level = libc::SOL_SOCKET;
        (*message).cmsg_type = libc::SCM_RIGHTS;
        (*message).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as _;
        std::ptr::write_unaligned(
            libc::CMSG_DATA(message).cast::<RawFd>(),
            descriptor.as_raw_fd(),
        );
    }
    // SAFETY: the header and the buffers it names live through the call.
    let sent = unsafe { libc::sendmsg(stream.as_raw_fd(), &header, libc::MSG_NOSIGNAL) };
    if sent < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(sent as usize)
}

/// Receives into `buffer`, taking every descriptor that arrived with the
/// bytes into `descriptors`, close-on-exec.
fn receive_with_descriptors(
    stream: &UnixStream,
    buffer: &mut [u8],
    descriptors: &mut VecDeque<OwnedFd>,
) -> std::io::Result<usize> {
    let mut control = ControlBuffer([0; CONTROL_BYTES]);
    let mut iov = libc::iovec {
        iov_base: buffer.as_mut_ptr().cast(),
        iov_len: buffer.len(),
    };
    // SAFETY: an all-zero `msghdr` is a valid value to fill in.
    let mut header: libc::msghdr = unsafe { std::mem::zeroed() };
    header.msg_iov = &mut iov;
    header.msg_iovlen = 1;
    header.msg_control = control.0.as_mut_ptr().cast();
    header.msg_controllen = control.0.len() as _;
    // SAFETY: the header and the buffers it names live through the call.
    let received =
        unsafe { libc::recvmsg(stream.as_raw_fd(), &mut header, libc::MSG_CMSG_CLOEXEC) };
    if received < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: recvmsg filled `msg_controllen` bytes of control messages; the
    // CMSG macros walk only within them, and each SCM_RIGHTS message's data
    // holds descriptors this process now owns.
    unsafe {
        let mut message = libc::CMSG_FIRSTHDR(&header);
        while !message.is_null() {
            if (*message).cmsg_level == libc::SOL_SOCKET && (*message).cmsg_type == libc::SCM_RIGHTS
            {
                let data = libc::CMSG_DATA(message);
                let length = (*message).cmsg_len as usize - (data as usize - message as usize);
                for index in 0..length / std::mem::size_of::<RawFd>() {
                    let descriptor = std::ptr::read_unaligned(data.cast::<RawFd>().add(index));
                    descriptors.push_back(OwnedFd::from_raw_fd(descriptor));
                }
            }
            message = libc::CMSG_NXTHDR(&header, message);
        }
    }
    Ok(received as usize)
}

/// Makes a TCP connection fail when its peer's node stops answering.
fn keep_alive(stream: &TcpStream) -> Result<()> {
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
            Err(io("enabling keepalive")(std::io::Error::last_os_error()))
        }
    };
    // An idle connection is probed after half the bound and fails when three
    // probes, spread over the other half, go unanswered.
    set(libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1)?;
    set(libc::IPPROTO_TCP, libc::TCP_KEEPIDLE, DEAD_PEER_SECS / 2)?;
    set(libc::IPPROTO_TCP, libc::TCP_KEEPINTVL, DEAD_PEER_SECS / 6)?;
    set(libc::IPPROTO_TCP, libc::TCP_KEEPCNT, 3)?;
    set(
        libc::IPPROTO_TCP,
        libc::TCP_USER_TIMEOUT,
        DEAD_PEER_SECS * 1000,
    )?;
    Ok(())
}

/// Waits until `fd` is ready for `events`, up to `timeout` (forever for
/// `None`).
fn poll(fd: RawFd, events: libc::c_short, timeout: Option<Duration>) -> std::io::Result<()> {
    let mut descriptor = libc::pollfd {
        fd,
        events,
        revents: 0,
    };
    let milliseconds = timeout.map_or(-1, |timeout| {
        libc::c_int::try_from(timeout.as_millis().max(1)).unwrap_or(libc::c_int::MAX)
    });
    loop {
        // SAFETY: one valid pollfd, owned by this frame, for the call.
        let result = unsafe { libc::poll(&mut descriptor, 1, milliseconds) };
        if result >= 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

/// A connection's socket as it crosses to another process, as a daemon's
/// hot restart hands its connections to its successor: the socket, whether
/// it is a Unix socket, and the bytes received and not yet taken as frames.
pub struct Handed {
    pub descriptor: OwnedFd,
    pub unix: bool,
    pub received: Vec<u8>,
}

impl Connection {
    /// Closes the connection both ways at once, whichever handles of its
    /// socket this process holds: the peer reads its end, and a thread
    /// waiting on another handle of the socket wakes.
    pub fn close(&self) {
        let _ = match &self.stream {
            Stream::Unix(stream) => stream.shutdown(std::net::Shutdown::Both),
            Stream::Tcp(stream) => stream.shutdown(std::net::Shutdown::Both),
        };
    }

    /// The connection's socket and the bytes it received and did not take,
    /// for another process to go on with ([`Connection::resume`]). Received
    /// descriptors not yet taken are closed: the connections a daemon hands
    /// over carry none between messages.
    pub fn hand_over(self) -> Handed {
        let (descriptor, unix) = match self.stream {
            Stream::Unix(stream) => (OwnedFd::from(stream), true),
            Stream::Tcp(stream) => (OwnedFd::from(stream), false),
        };
        Handed {
            descriptor,
            unix,
            received: self.received,
        }
    }

    /// Goes on with a connection another process handed over.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be made non-blocking.
    pub fn resume(handed: Handed) -> Result<Self> {
        let stream = if handed.unix {
            Stream::Unix(UnixStream::from(handed.descriptor))
        } else {
            Stream::Tcp(TcpStream::from(handed.descriptor))
        };
        let mut connection = Self::from_stream(stream)?;
        connection.received = handed.received;
        Ok(connection)
    }
}

/// A listening socket that hands out connections.
pub struct Listener {
    socket: ListeningSocket,
    address: String,
}

enum ListeningSocket {
    Unix(UnixListener),
    Tcp(TcpListener),
}

impl Listener {
    /// Listens at `address`: `unix:<path>` or `<host>:<port>`, where port
    /// 0 picks a free one.
    ///
    /// # Errors
    ///
    /// Fails when the address cannot be bound.
    pub fn bind(address: &str) -> Result<Self> {
        let (socket, address) = match address.strip_prefix(UNIX_PREFIX) {
            Some(path) => {
                let listener = UnixListener::bind(path).map_err(io("binding"))?;
                (ListeningSocket::Unix(listener), address.to_owned())
            }
            None => {
                let listener = TcpListener::bind(address).map_err(io("binding"))?;
                let bound: SocketAddr = listener.local_addr().map_err(io("binding"))?;
                (ListeningSocket::Tcp(listener), bound.to_string())
            }
        };
        match &socket {
            ListeningSocket::Unix(listener) => listener.set_nonblocking(true),
            ListeningSocket::Tcp(listener) => listener.set_nonblocking(true),
        }
        .map_err(io("making a listener pollable"))?;
        Ok(Self { socket, address })
    }

    /// The address the listener is bound at, with the port it picked.
    pub fn address(&self) -> &str {
        &self.address
    }

    /// The descriptor that becomes readable when a connection waits.
    pub fn descriptor(&self) -> RawFd {
        match &self.socket {
            ListeningSocket::Unix(listener) => listener.as_raw_fd(),
            ListeningSocket::Tcp(listener) => listener.as_raw_fd(),
        }
    }

    /// The listening socket, whether it is a Unix socket, and its address,
    /// for another process to go on with ([`Listener::resume`]).
    pub fn hand_over(self) -> (OwnedFd, bool, String) {
        match self.socket {
            ListeningSocket::Unix(listener) => (OwnedFd::from(listener), true, self.address),
            ListeningSocket::Tcp(listener) => (OwnedFd::from(listener), false, self.address),
        }
    }

    /// Goes on with a listening socket another process handed over, bound
    /// at `address`.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be made non-blocking.
    pub fn resume(descriptor: OwnedFd, unix: bool, address: String) -> Result<Self> {
        let socket = if unix {
            let listener = UnixListener::from(descriptor);
            listener
                .set_nonblocking(true)
                .map_err(io("making a listener pollable"))?;
            ListeningSocket::Unix(listener)
        } else {
            let listener = TcpListener::from(descriptor);
            listener
                .set_nonblocking(true)
                .map_err(io("making a listener pollable"))?;
            ListeningSocket::Tcp(listener)
        };
        Ok(Self { socket, address })
    }

    /// Accepts one waiting connection; `None` when none waits.
    ///
    /// # Errors
    ///
    /// Fails when accepting fails.
    pub fn try_accept(&self) -> Result<Option<Connection>> {
        let stream = match &self.socket {
            ListeningSocket::Unix(listener) => match listener.accept() {
                Ok((stream, _)) => Stream::Unix(stream),
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(None),
                Err(error) => return Err(io("accepting")(error)),
            },
            ListeningSocket::Tcp(listener) => match listener.accept() {
                Ok((stream, _)) => Stream::Tcp(stream),
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(None),
                Err(error) => return Err(io("accepting")(error)),
            },
        };
        Connection::from_stream(stream).map(Some)
    }
}
