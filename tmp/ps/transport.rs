//! Frames on the Program Server's Unix socket, with the module's files
//! travelling beside a frame as descriptors.
//!
//! A frame is an 8-byte header, the payload's length and the number of
//! descriptors that ride with it (both little-endian `u32`), then the payload
//! (one Protobuf message). The descriptors cross with `SCM_RIGHTS` on the
//! payload's first byte, so a receiver that has the whole frame has them all.
//! A connection is blocking: its two ends are a daemon and a Program Server on
//! one host, and every request either answers at once or waits for one build.

use std::collections::VecDeque;
use std::io::{self, IoSlice, IoSliceMut};
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::net::UnixStream;

use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags, recvmsg, sendmsg,
};

/// The largest frame a peer may send. A package's files travel in one
/// `install` frame; a module's are a cubin plus a JSON document.
pub const MAX_FRAME: usize = 64 << 20;

/// The most descriptors one frame may carry: a module's files (module,
/// metadata, tuning data).
pub const MAX_DESCRIPTORS: usize = 8;

/// Bytes of a frame header: the payload's length and the descriptor count.
const HEADER: usize = 8;

/// Bytes one receive reads at most.
const READ_CHUNK: usize = 64 * 1024;

/// A received frame: one Protobuf message and the files that rode with it,
/// in the order the sender listed them.
pub struct Frame {
    pub payload: Vec<u8>,
    pub descriptors: Vec<OwnedFd>,
}

/// One end of a connection to a Program Server.
pub struct Connection {
    stream: UnixStream,
    /// Bytes received and not yet taken as a frame.
    inbox: Vec<u8>,
    /// Descriptors received and not yet handed to a frame, in arrival order.
    descriptors: VecDeque<OwnedFd>,
}

impl Connection {
    pub fn new(stream: UnixStream) -> Self {
        Self {
            stream,
            inbox: Vec::new(),
            descriptors: VecDeque::new(),
        }
    }

    /// Sends one frame: `payload`, with `descriptors` riding on it.
    ///
    /// # Errors
    ///
    /// Fails when the peer closed its end, the frame exceeds [`MAX_FRAME`],
    /// or the socket fails.
    pub fn send(&mut self, payload: &[u8], descriptors: &[OwnedFd]) -> io::Result<()> {
        if payload.len() > MAX_FRAME {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("a frame of {} bytes exceeds {MAX_FRAME}", payload.len()),
            ));
        }
        if descriptors.len() > MAX_DESCRIPTORS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} descriptors exceed the {MAX_DESCRIPTORS} a frame may carry",
                    descriptors.len()
                ),
            ));
        }

        let mut header = [0u8; HEADER];
        header[..4].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        header[4..].copy_from_slice(&(descriptors.len() as u32).to_le_bytes());
        self.write_all(&header)?;
        if descriptors.is_empty() {
            return self.write_all(payload);
        }

        // The descriptors ride on the payload's first byte. A sendmsg may
        // write only part of the payload; the rest is plain bytes.
        let mut control = SendAncillaryBuffer::new();
        let borrowed: Vec<_> = descriptors.iter().map(|fd| fd.as_raw_fd()).collect();
        control.push(SendAncillaryMessage::ScmRights(&borrowed));
        let (written, _) = sendmsg(
            &self.stream,
            &[IoSlice::new(payload)],
            &mut control,
            SendFlags::empty(),
        )
        .map_err(io::Error::from)?;
        self.write_all(&payload[written..])
    }

    /// Receives one frame, waiting for it to arrive whole.
    ///
    /// # Errors
    ///
    /// Fails when the peer closed its end, a frame exceeds [`MAX_FRAME`] or
    /// carries more descriptors than [`MAX_DESCRIPTORS`], or the socket
    /// fails.
    pub fn recv(&mut self) -> io::Result<Frame> {
        loop {
            if let Some(frame) = self.take_frame()? {
                return Ok(frame);
            }
            self.receive_more()?;
        }
    }

    /// The connection's socket.
    pub fn stream(&self) -> &UnixStream {
        &self.stream
    }

    fn write_all(&mut self, mut bytes: &[u8]) -> io::Result<()> {
        while !bytes.is_empty() {
            match io::Write::write(&mut self.stream, bytes) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(written) => bytes = &bytes[written..],
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Takes the first whole frame out of the inbox, if one arrived.
    fn take_frame(&mut self) -> io::Result<Option<Frame>> {
        if self.inbox.len() < HEADER {
            return Ok(None);
        }
        let length = u32::from_le_bytes(self.inbox[..4].try_into().unwrap_or_default()) as usize;
        let count = u32::from_le_bytes(self.inbox[4..8].try_into().unwrap_or_default()) as usize;
        if length > MAX_FRAME {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("a frame of {length} bytes exceeds {MAX_FRAME}"),
            ));
        }
        if count > MAX_DESCRIPTORS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("a frame carries {count} descriptors, over {MAX_DESCRIPTORS}"),
            ));
        }
        if self.inbox.len() < HEADER + length {
            return Ok(None);
        }
        if self.descriptors.len() < count {
            // The bytes arrived but not every descriptor; receive again.
            return Ok(None);
        }
        let payload = self.inbox[HEADER..HEADER + length].to_vec();
        self.inbox.drain(..HEADER + length);
        let descriptors = self.descriptors.drain(..count).collect();
        Ok(Some(Frame {
            payload,
            descriptors,
        }))
    }

    fn receive_more(&mut self) -> io::Result<()> {
        let mut chunk = vec![0u8; READ_CHUNK];
        let mut buffer = [0u8; descriptor_control_bytes()];
        let mut control = RecvAncillaryBuffer::new(&mut buffer);
        let (read, flags) = recvmsg(
            &self.stream,
            &mut [IoSliceMut::new(&mut chunk)],
            &mut control,
            RecvFlags::empty(),
        )
        .map_err(io::Error::from)?;
        if read == 0 && flags.is_empty() {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        for message in control.peek() {
            match message {
                RecvAncillaryMessage::ScmRights(descriptors) => {
                    self.descriptors.extend(descriptors.iter().map(|fd| {
                        // The receiving side owns the descriptors the kernel
                        // installed for it.
                        unsafe { OwnedFd::from_raw_fd(*fd) }
                    }));
                }
                _ => {}
            }
        }
        self.inbox.extend_from_slice(&chunk[..read]);
        Ok(())
    }
}

use std::os::fd::FromRawFd;

/// Bytes of control data one receive takes: a control header and
/// [`MAX_DESCRIPTORS`] descriptors.
fn descriptor_control_bytes() -> usize {
    16 + MAX_DESCRIPTORS * std::mem::size_of::<i32>()
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::unix::net::UnixStream;

    use super::*;

    fn pair() -> (Connection, Connection) {
        let (left, right) = UnixStream::pair().expect("a socket pair");
        // A short timeout turns a wrong frame into a failure, not a hang.
        left.set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .expect("a timeout");
        right
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .expect("a timeout");
        (Connection::new(left), Connection::new(right))
    }

    #[test]
    fn a_message_crosses_without_descriptors() {
        let (mut sender, mut receiver) = pair();
        sender.send(b"a message", &[]).expect("a send");
        let frame = receiver.recv().expect("a frame");
        assert_eq!(frame.payload, b"a message");
        assert!(frame.descriptors.is_empty());
    }

    #[test]
    fn two_messages_come_back_in_order() {
        let (mut sender, mut receiver) = pair();
        sender.send(b"first", &[]).expect("a send");
        sender.send(b"second", &[]).expect("a send");
        assert_eq!(receiver.recv().expect("a frame").payload, b"first");
        assert_eq!(receiver.recv().expect("a frame").payload, b"second");
    }

    #[test]
    fn a_files_descriptors_ride_on_its_frame() {
        let (mut sender, mut receiver) = pair();
        let mut file = tempfile::tempfile().expect("a temporary file");
        file.write_all(b"a cubin").expect("a write");
        let descriptor = OwnedFd::from(file);
        sender
            .send(b"a module", std::slice::from_ref(&descriptor))
            .expect("a send");

        let frame = receiver.recv().expect("a frame");
        assert_eq!(frame.payload, b"a module");
        assert_eq!(frame.descriptors.len(), 1);
        let mut received = std::fs::File::from(frame.descriptors.into_iter().next().expect("one"));
        received.seek(SeekFrom::Start(0)).expect("a seek");
        let mut contents = Vec::new();
        received.read_to_end(&mut contents).expect("a read");
        assert_eq!(contents, b"a cubin");
    }
}
