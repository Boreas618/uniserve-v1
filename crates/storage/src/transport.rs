//! Framed messages with file descriptors over a Unix stream socket, or
//! without descriptors over TCP between hosts.
//!
//! A frame is an 8-byte header, the payload's length and the number of
//! descriptors that travel with it (both little-endian `u32`), the payload
//! (one MessagePack message with named fields), and one carrier byte for
//! every further [`MAX_DESCRIPTORS`] descriptors. Descriptors cross with
//! `SCM_RIGHTS`, at most [`MAX_DESCRIPTORS`] per `sendmsg`: the first chunk
//! rides on the frame's first byte and each later chunk on its own carrier
//! byte. A receiver queues every descriptor it receives in order and hands a
//! frame the number its header states once all of the frame's bytes, and
//! therefore all of its descriptors, have arrived.

use std::collections::VecDeque;
use std::io::{self, IoSlice, IoSliceMut};
use std::mem::MaybeUninit;
use std::net::TcpStream;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;

use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags, recvmsg, sendmsg,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Descriptors one `sendmsg` carries: the kernel's `SCM_MAX_FD`.
pub const MAX_DESCRIPTORS: usize = 253;

/// Bytes of a frame header.
const HEADER: usize = 8;

/// Bytes one receive reads at most.
const READ_CHUNK: usize = 64 * 1024;

/// Part of an outgoing frame: bytes, and the descriptors that ride on their
/// first byte.
struct Segment {
    bytes: Vec<u8>,
    sent: usize,
    descriptors: Vec<OwnedFd>,
}

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
    /// Received descriptors not yet handed to a frame, in arrival order.
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
    /// Storage Server's hot restart hands its clients' connections to its
    /// successor.
    ///
    /// # Errors
    ///
    /// Fails when frames wait to be written or descriptors are held.
    pub fn hand_over(self) -> io::Result<(Socket, Vec<u8>)> {
        if !self.descriptors.is_empty() || !self.outbox.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "a connection with frames to write cannot be handed over",
            ));
        }
        Ok((self.stream, self.inbox))
    }

    /// Goes on with a connection another process handed over, with the
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
    /// nothing is buffered either way: a daemon's hot restart hands its
    /// idle connections to its successor.
    ///
    /// # Errors
    ///
    /// Fails when bytes or descriptors are buffered.
    pub fn into_socket(self) -> io::Result<Socket> {
        if !self.inbox.is_empty() || !self.descriptors.is_empty() || !self.outbox.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "a connection with buffered frames cannot be handed over",
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
        let length = u32::try_from(payload.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a frame exceeds 4 GiB"))?;
        let count = u32::try_from(descriptors.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many descriptors"))?;
        if count > 0 && matches!(self.stream, Socket::Tcp(_)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "descriptors cross only a Unix socket",
            ));
        }
        let owned = descriptors
            .iter()
            .map(|descriptor| descriptor.try_clone_to_owned())
            .collect::<io::Result<Vec<_>>>()?;

        let mut bytes = Vec::with_capacity(HEADER + payload.len());
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes.extend_from_slice(&payload);
        let mut chunks = owned.into_iter().peekable();
        let first: Vec<OwnedFd> = chunks.by_ref().take(MAX_DESCRIPTORS).collect();
        self.outbox.push_back(Segment {
            bytes,
            sent: 0,
            descriptors: first,
        });
        while chunks.peek().is_some() {
            self.outbox.push_back(Segment {
                bytes: vec![0],
                sent: 0,
                descriptors: chunks.by_ref().take(MAX_DESCRIPTORS).collect(),
            });
        }
        self.flush().map(drop)
    }

    /// Write queued frames until the socket would block. Returns whether
    /// everything was written.
    pub fn flush(&mut self) -> io::Result<bool> {
        while let Some(segment) = self.outbox.front_mut() {
            let mut space =
                [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_DESCRIPTORS))];
            let mut control = SendAncillaryBuffer::new(&mut space);
            let borrowed: Vec<BorrowedFd<'_>> =
                segment.descriptors.iter().map(AsFd::as_fd).collect();
            if !borrowed.is_empty() {
                control.push(SendAncillaryMessage::ScmRights(&borrowed));
            }
            let sent = match sendmsg(
                &self.stream,
                &[IoSlice::new(&segment.bytes[segment.sent..])],
                &mut control,
                SendFlags::NOSIGNAL,
            ) {
                Ok(sent) => sent,
                Err(rustix::io::Errno::AGAIN) => return Ok(false),
                Err(rustix::io::Errno::INTR) => continue,
                Err(error) => return Err(error.into()),
            };
            drop(borrowed);
            // The descriptors crossed with the first byte that was sent.
            segment.descriptors.clear();
            segment.sent += sent;
            if segment.sent == segment.bytes.len() {
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
        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_DESCRIPTORS))];
        let mut control = RecvAncillaryBuffer::new(&mut space);
        let message = loop {
            match recvmsg(
                &self.stream,
                &mut [IoSliceMut::new(&mut buffer)],
                &mut control,
                RecvFlags::CMSG_CLOEXEC,
            ) {
                Ok(message) => break message,
                Err(rustix::io::Errno::AGAIN) => return Ok(false),
                Err(rustix::io::Errno::INTR) => continue,
                Err(error) => return Err(error.into()),
            }
        };
        if message.flags.contains(ReturnFlags::CTRUNC) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "a message carried more descriptors than one receive takes",
            ));
        }
        for received in control.drain() {
            if let RecvAncillaryMessage::ScmRights(descriptors) = received {
                self.descriptors.extend(descriptors);
            }
        }
        if message.bytes == 0 {
            self.closed = true;
            return Ok(false);
        }
        self.inbox.extend_from_slice(&buffer[..message.bytes]);
        Ok(true)
    }

    /// Whether the peer closed its end.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// The next complete frame received, decoded, with its descriptors.
    pub fn next_frame<T: DeserializeOwned>(&mut self) -> io::Result<Option<(T, Vec<OwnedFd>)>> {
        if self.inbox.len() < HEADER {
            return Ok(None);
        }
        let word = |at: usize| {
            let mut bytes = [0_u8; 4];
            bytes.copy_from_slice(&self.inbox[at..at + 4]);
            u32::from_le_bytes(bytes) as usize
        };
        let (length, count) = (word(0), word(4));
        let carriers = count.div_ceil(MAX_DESCRIPTORS).saturating_sub(1);
        let total = HEADER + length + carriers;
        if self.inbox.len() < total || self.descriptors.len() < count {
            return Ok(None);
        }
        let message = rmp_serde::from_slice(&self.inbox[HEADER..HEADER + length])
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        self.inbox.drain(..total);
        let descriptors = self.descriptors.drain(..count).collect();
        Ok(Some((message, descriptors)))
    }

    /// Block until a frame arrives and return it. Fails with
    /// `UnexpectedEof` when the peer closes first.
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
            self.receive()?;
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
