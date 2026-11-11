//! The framing every socket transport of the workspace writes and reads, and the passing of file
//! descriptors beside the bytes.
//!
//! Two frame layouts cross sockets, both little-endian:
//!
//! - a **length-prefixed frame** is a `u32` payload length followed by the payload. The daemon's
//!   process interface (`proto::Connection`, Protobuf) and the epoch's peer channel
//!   (`ipc::Channel`, MessagePack) carry these; a descriptor they pass rides on a frame's first
//!   byte;
//! - a **descriptor frame** is an 8-byte header, the payload's length and the number of
//!   descriptors that travel with it (two `u32`), the payload, and one carrier byte for every
//!   further [`MAX_DESCRIPTORS`] descriptors. The host services' connections (the Storage Server,
//!   MessagePack; the Build Server, Protobuf) carry these.
//!
//! Descriptors cross with `SCM_RIGHTS`, at most [`MAX_DESCRIPTORS`] per `sendmsg`: a descriptor
//! frame's first chunk rides on its first byte and each later chunk on its own carrier byte
//! ([`Segment`]). A receiver queues every descriptor it receives in arrival order
//! ([`receive_with_descriptors`]) and hands a frame the number its header states once all of the
//! frame's bytes, and therefore all of its descriptors, have arrived ([`take_descriptor_frame`]).
//! Received descriptors are close-on-exec.
//!
//! The codec is the caller's: this module moves bytes and descriptors only.

use std::collections::VecDeque;
use std::io::{self, IoSlice, IoSliceMut};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};

use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags,
};

/// Descriptors one `sendmsg` carries: the kernel's `SCM_MAX_FD`.
pub const MAX_DESCRIPTORS: usize = 253;

/// Bytes of a length-prefixed frame's length field.
pub const LENGTH_BYTES: usize = 4;

/// Bytes of a descriptor frame's header: the payload's length and its descriptor count.
pub const HEADER_BYTES: usize = 8;

/// `payload` as one length-prefixed frame.
///
/// # Errors
///
/// Fails with `InvalidInput` when the payload is longer than the length field states.
pub fn length_prefixed(payload: &[u8]) -> io::Result<Vec<u8>> {
    let length = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a frame exceeds 4 GiB"))?;
    let mut frame = Vec::with_capacity(LENGTH_BYTES + payload.len());
    frame.extend_from_slice(&length.to_le_bytes());
    frame.extend_from_slice(payload);
    Ok(frame)
}

/// The payload length a length field states.
pub fn stated_length(field: [u8; LENGTH_BYTES]) -> usize {
    u32::from_le_bytes(field) as usize
}

/// The payload length of the length-prefixed frame `inbox` starts with, once its length field
/// arrived.
pub fn peek_length(inbox: &[u8]) -> Option<usize> {
    let field = inbox.get(..LENGTH_BYTES)?;
    Some(stated_length([field[0], field[1], field[2], field[3]]))
}

/// Takes the payload of the length-prefixed frame `inbox` starts with, once it arrived whole.
pub fn take_length_prefixed(inbox: &mut Vec<u8>) -> Option<Vec<u8>> {
    let length = peek_length(inbox)?;
    if inbox.len() < LENGTH_BYTES + length {
        return None;
    }
    let payload = inbox[LENGTH_BYTES..LENGTH_BYTES + length].to_vec();
    inbox.drain(..LENGTH_BYTES + length);
    Some(payload)
}

/// Part of an outgoing descriptor frame: bytes, how many of them were sent, and the descriptors
/// that ride on their first byte.
pub struct Segment {
    bytes: Vec<u8>,
    sent: usize,
    descriptors: Vec<OwnedFd>,
}

impl Segment {
    /// Whether every byte of the segment was sent.
    pub fn sent(&self) -> bool {
        self.sent == self.bytes.len()
    }

    /// Sends what of the segment `socket` takes in one `sendmsg`, with its descriptors on the first
    /// byte sent.
    ///
    /// # Errors
    ///
    /// Fails with `WouldBlock` when a non-blocking socket takes nothing, `WriteZero` when it
    /// accepts no byte, and with the socket's own error otherwise.
    pub fn send(&mut self, socket: BorrowedFd<'_>) -> io::Result<()> {
        let borrowed: Vec<BorrowedFd<'_>> = self.descriptors.iter().map(AsFd::as_fd).collect();
        let sent = send_with_descriptors(socket, &self.bytes[self.sent..], &borrowed)?;
        drop(borrowed);
        if sent == 0 {
            return Err(io::Error::from(io::ErrorKind::WriteZero));
        }
        // The descriptors crossed with the first byte that was sent.
        self.descriptors.clear();
        self.sent += sent;
        Ok(())
    }
}

/// `payload` and `descriptors` as the segments of one descriptor frame, in sending order.
///
/// # Errors
///
/// Fails with `InvalidInput` when the payload or the descriptor count is longer than the header
/// states.
pub fn descriptor_frame(payload: &[u8], descriptors: Vec<OwnedFd>) -> io::Result<Vec<Segment>> {
    let length = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a frame exceeds 4 GiB"))?;
    let count = u32::try_from(descriptors.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many descriptors"))?;
    let mut bytes = Vec::with_capacity(HEADER_BYTES + payload.len());
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(payload);

    let mut chunks = descriptors.into_iter().peekable();
    let mut segments = vec![Segment {
        bytes,
        sent: 0,
        descriptors: chunks.by_ref().take(MAX_DESCRIPTORS).collect(),
    }];
    while chunks.peek().is_some() {
        segments.push(Segment {
            bytes: vec![0],
            sent: 0,
            descriptors: chunks.by_ref().take(MAX_DESCRIPTORS).collect(),
        });
    }
    Ok(segments)
}

/// The bytes the descriptor frame `inbox` starts with occupies, once its header arrived: the
/// header, the payload and the carrier bytes.
pub fn descriptor_frame_bytes(inbox: &[u8]) -> Option<usize> {
    let (length, count) = descriptor_header(inbox)?;
    let carriers = count.div_ceil(MAX_DESCRIPTORS).saturating_sub(1);
    Some(HEADER_BYTES + length + carriers)
}

/// Takes the payload and the descriptors of the descriptor frame `inbox` starts with, once all of
/// its bytes and descriptors arrived.
pub fn take_descriptor_frame(
    inbox: &mut Vec<u8>,
    descriptors: &mut VecDeque<OwnedFd>,
) -> Option<(Vec<u8>, Vec<OwnedFd>)> {
    let (length, count) = descriptor_header(inbox)?;
    let total = descriptor_frame_bytes(inbox)?;
    if inbox.len() < total || descriptors.len() < count {
        return None;
    }
    let payload = inbox[HEADER_BYTES..HEADER_BYTES + length].to_vec();
    inbox.drain(..total);
    Some((payload, descriptors.drain(..count).collect()))
}

/// The payload length and descriptor count a descriptor frame's header states.
fn descriptor_header(inbox: &[u8]) -> Option<(usize, usize)> {
    let header = inbox.get(..HEADER_BYTES)?;
    let word = |at: usize| {
        u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]]) as usize
    };
    Some((word(0), word(4)))
}

/// Sends the start of `bytes` on `socket` in one `sendmsg`, with `descriptors` attached when there
/// are any, and answers how many bytes the socket accepted. An interrupted call is retried.
///
/// # Errors
///
/// Fails with `InvalidInput` for more than [`MAX_DESCRIPTORS`] descriptors, `WouldBlock` when a
/// non-blocking socket is full, and with the socket's own error otherwise (`BrokenPipe` once the
/// peer closed: the call never raises `SIGPIPE`).
pub fn send_with_descriptors(
    socket: BorrowedFd<'_>,
    bytes: &[u8],
    descriptors: &[BorrowedFd<'_>],
) -> io::Result<usize> {
    if descriptors.len() > MAX_DESCRIPTORS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "one send carries at most {MAX_DESCRIPTORS} descriptors, not {}",
                descriptors.len()
            ),
        ));
    }
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_DESCRIPTORS))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    if !descriptors.is_empty() {
        control.push(SendAncillaryMessage::ScmRights(descriptors));
    }
    loop {
        match rustix::net::sendmsg(
            socket,
            &[IoSlice::new(bytes)],
            &mut control,
            SendFlags::NOSIGNAL,
        ) {
            Ok(sent) => return Ok(sent),
            Err(rustix::io::Errno::INTR) => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

/// Receives into `buffer` from `socket` in one `recvmsg`, queueing every descriptor that arrived
/// with the bytes into `descriptors`, and answers how many bytes arrived; zero means the peer
/// closed. An interrupted call is retried.
///
/// # Errors
///
/// Fails with `WouldBlock` when a non-blocking socket holds nothing, `InvalidData` when the
/// descriptors that arrived did not fit one receive, and with the socket's own error otherwise.
pub fn receive_with_descriptors(
    socket: BorrowedFd<'_>,
    buffer: &mut [u8],
    descriptors: &mut VecDeque<OwnedFd>,
) -> io::Result<usize> {
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_DESCRIPTORS))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    let message = loop {
        match rustix::net::recvmsg(
            socket,
            &mut [IoSliceMut::new(buffer)],
            &mut control,
            RecvFlags::CMSG_CLOEXEC,
        ) {
            Ok(message) => break message,
            Err(rustix::io::Errno::INTR) => continue,
            Err(error) => return Err(error.into()),
        }
    };
    for received in control.drain() {
        if let RecvAncillaryMessage::ScmRights(arrived) = received {
            descriptors.extend(arrived);
        }
    }
    if message.flags.contains(ReturnFlags::CTRUNC) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "a message carried more descriptors than one receive takes",
        ));
    }
    Ok(message.bytes)
}

#[cfg(test)]
mod tests {
    use std::io::{Read as _, Seek as _, Write as _};
    use std::os::unix::net::UnixStream;

    use super::*;

    /// A temporary file holding `text`, as a descriptor to pass.
    fn file(text: &str) -> OwnedFd {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(text.as_bytes()).unwrap();
        file.rewind().unwrap();
        file.into()
    }

    /// What the file a received descriptor opens holds.
    fn read(descriptor: OwnedFd) -> String {
        let mut text = String::new();
        std::fs::File::from(descriptor)
            .read_to_string(&mut text)
            .unwrap();
        text
    }

    /// Receives until `inbox` holds one whole descriptor frame and answers it.
    fn receive_frame(
        socket: &UnixStream,
        inbox: &mut Vec<u8>,
        queued: &mut VecDeque<OwnedFd>,
    ) -> (Vec<u8>, Vec<OwnedFd>) {
        loop {
            if let Some(frame) = take_descriptor_frame(inbox, queued) {
                return frame;
            }
            let mut buffer = [0_u8; 7];
            let received = receive_with_descriptors(socket.as_fd(), &mut buffer, queued).unwrap();
            assert!(received > 0, "the sender closed before the frame arrived");
            inbox.extend_from_slice(&buffer[..received]);
        }
    }

    /// A descriptor frame carries more descriptors than one send takes, in order, and the frame
    /// after it carries none: a receiver that reads in small pieces hands each frame its own.
    #[test]
    fn descriptor_frames_carry_their_descriptors_in_order_beyond_one_send() {
        let (sender, receiver) = UnixStream::pair().unwrap();
        let texts: Vec<String> = (0..MAX_DESCRIPTORS + 2)
            .map(|index| format!("file {index}"))
            .collect();
        let sending = std::thread::spawn(move || {
            let descriptors = texts.iter().map(|text| file(text)).collect();
            for payload_descriptors in [(b"first".as_slice(), descriptors), (b"second", Vec::new())]
            {
                for mut segment in
                    descriptor_frame(payload_descriptors.0, payload_descriptors.1).unwrap()
                {
                    while !segment.sent() {
                        segment.send(sender.as_fd()).unwrap();
                    }
                }
            }
        });

        let (mut inbox, mut queued) = (Vec::new(), VecDeque::new());
        let (payload, descriptors) = receive_frame(&receiver, &mut inbox, &mut queued);
        assert_eq!(payload, b"first");
        assert_eq!(descriptors.len(), MAX_DESCRIPTORS + 2);
        for (index, descriptor) in descriptors.into_iter().enumerate() {
            assert_eq!(read(descriptor), format!("file {index}"));
        }
        let (payload, descriptors) = receive_frame(&receiver, &mut inbox, &mut queued);
        assert_eq!(payload, b"second");
        assert!(descriptors.is_empty());
        sending.join().unwrap();
    }

    /// A length-prefixed frame is taken only once it arrived whole, and leaves the bytes after it.
    #[test]
    fn a_length_prefixed_frame_is_taken_once_whole() {
        let mut inbox = length_prefixed(b"payload").unwrap();
        inbox.extend_from_slice(&length_prefixed(b"next").unwrap()[..3]);
        assert_eq!(peek_length(&inbox), Some(7));
        assert_eq!(
            take_length_prefixed(&mut inbox).as_deref(),
            Some(&b"payload"[..])
        );
        assert_eq!(take_length_prefixed(&mut inbox), None);
        assert_eq!(inbox.len(), 3);
    }

    /// More descriptors than one send carries are refused before anything is sent.
    #[test]
    fn one_send_refuses_more_descriptors_than_the_kernel_takes() {
        let (sender, _receiver) = UnixStream::pair().unwrap();
        let files: Vec<OwnedFd> = (0..=MAX_DESCRIPTORS).map(|_| file("x")).collect();
        let borrowed: Vec<BorrowedFd<'_>> = files.iter().map(AsFd::as_fd).collect();
        let refused = send_with_descriptors(sender.as_fd(), b"x", &borrowed).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::InvalidInput);
    }
}
