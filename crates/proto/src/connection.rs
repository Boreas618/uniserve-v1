//! One end of a connection that carries the interfaces' messages.
//!
//! A frame is a little-endian `u32` length followed by one Protobuf message.
//! An address is written `unix:<path>` for a Unix socket and `<host>:<port>`
//! for TCP. A connection is non-blocking: [`Connection::try_recv`] returns a
//! message once one arrived whole, and [`Connection::wait`] blocks until the
//! socket is readable. Sends complete before they return. A TCP connection
//! probes its peer, so a node that died closes the connection like a process
//! that exited.

use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::time::Duration;

use prost::Message;

/// The prefix of a Unix socket address; the rest is the socket's path.
const UNIX_PREFIX: &str = "unix:";

/// The largest frame a peer may send. The interfaces' messages are small;
/// a larger length is a corrupt stream.
const MAX_FRAME: usize = 64 << 20;

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

    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Unix(stream) => stream.read(buffer),
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
        })
    }

    /// Sends `message` as one frame, waiting until the socket accepted all
    /// of it.
    ///
    /// # Errors
    ///
    /// Fails when the peer closed or the socket failed.
    pub fn send(&mut self, message: &impl Message) -> Result<()> {
        let body = message.encode_to_vec();
        let mut frame = Vec::with_capacity(4 + body.len());
        frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
        frame.extend_from_slice(&body);
        let mut written = 0;
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
                match self.stream.read(&mut chunk) {
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
