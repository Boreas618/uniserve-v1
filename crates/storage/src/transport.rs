//! Framed messages with file descriptors over a Unix stream socket, or
//! without descriptors over TCP between hosts.
//!
//! A frame is a descriptor frame of the workspace's one framing module
//! (`service::frame`) whose payload is one MessagePack message with named
//! fields: the descriptors that travel with it cross with `SCM_RIGHTS` on
//! its first byte and its carrier bytes.

use std::collections::VecDeque;
use std::io;
use std::net::TcpStream;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;

use serde::Serialize;
use serde::de::DeserializeOwned;
use service::frame::{self, Segment};

/// Descriptors one `sendmsg` carries: the kernel's `SCM_MAX_FD`.
pub use service::frame::MAX_DESCRIPTORS;

/// Bytes one receive reads at most.
const READ_CHUNK: usize = 64 * 1024;

/// The stream a connection runs on: a Unix socket on the host, which
/// carries descriptors, or a TCP socket from another host, which carries
/// none.
pub enum Socket {
    Unix(UnixStream),
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

impl Socket {
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        match self {
            Self::Unix(stream) => stream.set_nonblocking(nonblocking),
            Self::Tcp(stream) => stream.set_nonblocking(nonblocking),
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

/// One end of a framed connection.
pub struct Connection {
    stream: Socket,
    /// Received bytes not yet parsed into frames.
    inbox: Vec<u8>,
    /// Received descriptors not yet given to a frame, in arrival order.
    descriptors: VecDeque<OwnedFd>,
    /// Frames, or parts of frames, not yet written.
    outbox: VecDeque<Segment>,
    /// Whether the peer closed its end.
    closed: bool,
}

impl Connection {
    pub fn new(stream: impl Into<Socket>) -> Self {
        Self {
            stream: stream.into(),
            inbox: Vec::new(),
            descriptors: VecDeque::new(),
            outbox: VecDeque::new(),
            closed: false,
        }
    }

    pub fn stream(&self) -> &Socket {
        &self.stream
    }

    /// The connection's socket and the bytes received and not yet taken
    /// as frames, for another process to go on with ([`Self::resume`]),
    /// once nothing waits to be written and no descriptor is held: a
    /// Storage Server's restart in place passes its clients' connections to its
    /// successor.
    ///
    /// # Errors
    ///
    /// Fails when frames wait to be written or descriptors are held.
    pub fn handoff(self) -> io::Result<(Socket, Vec<u8>)> {
        if !self.descriptors.is_empty() || !self.outbox.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "a connection with frames to write cannot be passed on",
            ));
        }
        Ok((self.stream, self.inbox))
    }

    /// Goes on with a connection another process passed on, with the
    /// bytes it had received.
    pub fn resume(stream: Socket, inbox: Vec<u8>) -> Self {
        Self {
            stream,
            inbox,
            descriptors: VecDeque::new(),
            outbox: VecDeque::new(),
            closed: false,
        }
    }

    /// The connection's socket, for another process to go on with, once
    /// nothing is buffered either way: a daemon's restart in place passes its
    /// idle connections to its successor.
    ///
    /// # Errors
    ///
    /// Fails when bytes or descriptors are buffered.
    pub fn into_socket(self) -> io::Result<Socket> {
        if !self.inbox.is_empty() || !self.descriptors.is_empty() || !self.outbox.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "a connection with buffered frames cannot be passed on",
            ));
        }
        Ok(self.stream)
    }

    /// Whether frames wait to be written.
    pub fn wants_write(&self) -> bool {
        !self.outbox.is_empty()
    }

    /// Queue `message` with `descriptors`, which are duplicated, and write
    /// what the socket takes. A blocking socket writes the whole frame
    /// before this returns.
    pub fn send<T: Serialize>(
        &mut self,
        message: &T,
        descriptors: &[BorrowedFd<'_>],
    ) -> io::Result<()> {
        let payload = rmp_serde::to_vec_named(message)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if !descriptors.is_empty() && matches!(self.stream, Socket::Tcp(_)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "descriptors cross only a Unix socket",
            ));
        }
        let owned = descriptors
            .iter()
            .map(|descriptor| descriptor.try_clone_to_owned())
            .collect::<io::Result<Vec<_>>>()?;
        self.outbox
            .extend(frame::descriptor_frame(&payload, owned)?);
        self.flush().map(drop)
    }

    /// Write queued frames until the socket would block. Returns whether
    /// everything was written.
    pub fn flush(&mut self) -> io::Result<bool> {
        while let Some(segment) = self.outbox.front_mut() {
            match segment.send(self.stream.as_fd()) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(false),
                Err(error) => return Err(error),
            }
            if segment.sent() {
                self.outbox.pop_front();
            }
        }
        Ok(true)
    }

    /// Read what the socket holds without blocking past one read: a
    /// blocking socket waits for at least one byte. Returns whether
    /// anything was read; a closed peer reads nothing and marks the
    /// connection closed.
    pub fn receive(&mut self) -> io::Result<bool> {
        let mut buffer = vec![0_u8; READ_CHUNK];
        let received = match frame::receive_with_descriptors(
            self.stream.as_fd(),
            &mut buffer,
            &mut self.descriptors,
        ) {
            Ok(received) => received,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(false),
            Err(error) => return Err(error),
        };
        if received == 0 {
            self.closed = true;
            return Ok(false);
        }
        self.inbox.extend_from_slice(&buffer[..received]);
        Ok(true)
    }

    /// Whether the peer closed its end.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// The next complete frame received, decoded, with its descriptors.
    pub fn next_frame<T: DeserializeOwned>(&mut self) -> io::Result<Option<(T, Vec<OwnedFd>)>> {
        let Some((payload, descriptors)) =
            frame::take_descriptor_frame(&mut self.inbox, &mut self.descriptors)
        else {
            return Ok(None);
        };
        let message = rmp_serde::from_slice(&payload)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        Ok(Some((message, descriptors)))
    }

    /// Block until a frame arrives and return it. Fails with
    /// `UnexpectedEof` when the peer closes first, and with `TimedOut` when
    /// the socket's receive timeout (`SO_RCVTIMEO`) passes without a byte.
    pub fn recv<T: DeserializeOwned>(&mut self) -> io::Result<(T, Vec<OwnedFd>)> {
        loop {
            if let Some(frame) = self.next_frame()? {
                return Ok(frame);
            }
            if self.closed {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "the storage server connection closed",
                ));
            }
            // A blocking read returns nothing without the peer closing only
            // when the socket's receive timeout passed.
            if !self.receive()? && !self.closed {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "the storage server did not response within the connection's timeout",
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, Write};

    use super::*;

    /// A temporary file holding `text`, as a descriptor to pass.
    fn file(text: &str) -> std::fs::File {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(text.as_bytes()).unwrap();
        file.rewind().unwrap();
        file
    }

    #[test]
    fn frames_carry_their_descriptors_in_order_beyond_one_sendmsg() {
        let (left, right) = UnixStream::pair().unwrap();
        let mut sender = Connection::new(left);
        let mut receiver = Connection::new(right);

        // More descriptors than one sendmsg carries, then a frame without.
        let files: Vec<std::fs::File> = (0..MAX_DESCRIPTORS + 7)
            .map(|index| file(&index.to_string()))
            .collect();
        let borrowed: Vec<BorrowedFd<'_>> = files.iter().map(AsFd::as_fd).collect();
        sender.send(&vec![1_u32, 2, 3], &borrowed).unwrap();
        sender.send(&"after", &[]).unwrap();

        let (first, descriptors): (Vec<u32>, _) = receiver.recv().unwrap();
        assert_eq!(first, vec![1, 2, 3]);
        assert_eq!(descriptors.len(), MAX_DESCRIPTORS + 7);
        for (index, descriptor) in descriptors.into_iter().enumerate() {
            let text = std::io::read_to_string(std::fs::File::from(descriptor)).unwrap();
            assert_eq!(text, index.to_string());
        }
        let (second, descriptors): (String, _) = receiver.recv().unwrap();
        assert_eq!((second.as_str(), descriptors.len()), ("after", 0));
    }

    #[test]
    fn a_closed_peer_ends_a_blocking_receive() {
        let (left, right) = UnixStream::pair().unwrap();
        drop(left);
        let mut receiver = Connection::new(right);
        let error = receiver.recv::<u32>().err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }
}
