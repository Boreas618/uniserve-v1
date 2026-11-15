//! The one frame layout every socket transport of the workspace writes and reads, and the passing
//! of file descriptors beside the bytes.
//!
//! A frame is little-endian: an 8-byte header, the payload's length and the number of descriptors
//! that travel with it (two `u32`), then the payload, then one carrier byte for every further
//! [`MAX_DESCRIPTORS`] descriptors. Every protocol's frames take this layout, whatever codec its
//! payload has (Protobuf for the process and request interfaces and the Build Server, MessagePack
//! for the Storage Server and the engines' peer link); over TCP a frame carries no descriptor.
//!
//! Descriptors cross with `SCM_RIGHTS`, at most [`MAX_DESCRIPTORS`] per `sendmsg`: a frame's first
//! chunk rides on its first byte and each later chunk on its own carrier byte ([`Segment`]). A
//! receiver queues every descriptor it receives in arrival order ([`receive_with_descriptors`])
//! and hands a frame the number its header states once all of the frame's bytes, and therefore
//! all of its descriptors, have arrived ([`take`]). Received descriptors are close-on-exec.
//!
//! The codec is the caller's: this module moves bytes and descriptors only. The one connection
//! over these frames is [`crate::Connection`].

use std::collections::VecDeque;
use std::io::{self, IoSlice, IoSliceMut};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};

use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags,
};

/// `SCM_MAX_FD` of the Linux kernel (`include/net/scm.h`): the most descriptors one `sendmsg`
/// carries in an `SCM_RIGHTS` message; more fail with `EINVAL` (unix(7)). The kernel exports it
/// in no user-space header, and neither `libc` nor `rustix` binds it, so it is bound here; a test
/// checks it against the running kernel.
const SCM_MAX_FD: usize = 253;

/// Descriptors one `sendmsg` carries: the kernel's `SCM_MAX_FD`.
pub const MAX_DESCRIPTORS: usize = SCM_MAX_FD;

/// Bytes of a frame's header: the payload's length and its descriptor count.
pub const HEADER_BYTES: usize = 8;

/// Part of an outgoing frame: bytes, how many of them were sent, and the descriptors that ride on
/// their first byte.
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

/// `payload` and `descriptors` as the segments of one frame, in sending order.
///
/// # Errors
///
/// Fails with `InvalidInput` when the payload or the descriptor count is longer than the header
/// states.
pub fn segments(payload: &[u8], descriptors: Vec<OwnedFd>) -> io::Result<Vec<Segment>> {
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

/// The payload length of the frame `inbox` starts with, once its header arrived.
pub fn payload_bytes(inbox: &[u8]) -> Option<usize> {
    header(inbox).map(|(length, _)| length)
}

/// The bytes the frame `inbox` starts with occupies, once its header arrived: the header, the
/// payload and the carrier bytes.
pub fn frame_bytes(inbox: &[u8]) -> Option<usize> {
    let (length, count) = header(inbox)?;
    let carriers = count.div_ceil(MAX_DESCRIPTORS).saturating_sub(1);
    Some(HEADER_BYTES + length + carriers)
}

/// Takes the payload and the descriptors of the frame `inbox` starts with, once all of its bytes
/// and descriptors arrived.
pub fn take(
    inbox: &mut Vec<u8>,
    descriptors: &mut VecDeque<OwnedFd>,
) -> Option<(Vec<u8>, Vec<OwnedFd>)> {
    let (length, count) = header(inbox)?;
    let total = frame_bytes(inbox)?;
    if inbox.len() < total || descriptors.len() < count {
        return None;
    }
    let payload = inbox[HEADER_BYTES..HEADER_BYTES + length].to_vec();
    inbox.drain(..total);
    Some((payload, descriptors.drain(..count).collect()))
}

/// The payload length and descriptor count a frame's header states.
fn header(inbox: &[u8]) -> Option<(usize, usize)> {
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

    /// Receives until `inbox` holds one whole frame and answers it.
    fn receive_frame(
        socket: &UnixStream,
        inbox: &mut Vec<u8>,
        queued: &mut VecDeque<OwnedFd>,
    ) -> (Vec<u8>, Vec<OwnedFd>) {
        loop {
            if let Some(frame) = take(inbox, queued) {
                return frame;
            }
            let mut buffer = [0_u8; 7];
            let received = receive_with_descriptors(socket.as_fd(), &mut buffer, queued).unwrap();
            assert!(received > 0, "the sender closed before the frame arrived");
            inbox.extend_from_slice(&buffer[..received]);
        }
    }

    /// A frame carries more descriptors than one send takes, in order, and the frame after it
    /// carries none: a receiver that reads in small pieces hands each frame its own.
    #[test]
    fn frames_carry_their_descriptors_in_order_beyond_one_send() {
        let (sender, receiver) = UnixStream::pair().unwrap();
        let texts: Vec<String> = (0..MAX_DESCRIPTORS + 2)
            .map(|index| format!("file {index}"))
            .collect();
        let sending = std::thread::spawn(move || {
            let descriptors = texts.iter().map(|text| file(text)).collect();
            for (payload, descriptors) in
                [(b"first".as_slice(), descriptors), (b"second", Vec::new())]
            {
                for mut segment in segments(payload, descriptors).unwrap() {
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

    /// A frame is taken only once it arrived whole, and leaves the bytes after it.
    #[test]
    fn a_frame_is_taken_once_whole() {
        let bytes = |payload: &[u8]| -> Vec<u8> {
            segments(payload, Vec::new())
                .unwrap()
                .into_iter()
                .flat_map(|segment| segment.bytes)
                .collect()
        };
        let mut inbox = bytes(b"payload");
        inbox.extend_from_slice(&bytes(b"next")[..3]);
        assert_eq!(payload_bytes(&inbox), Some(7));
        let mut queued = VecDeque::new();
        assert_eq!(
            take(&mut inbox, &mut queued).map(|(payload, _)| payload),
            Some(b"payload".to_vec())
        );
        assert!(take(&mut inbox, &mut queued).is_none());
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

    /// The bound `SCM_MAX_FD` is the running kernel's: one `sendmsg` carries that many
    /// descriptors, and the kernel refuses one more with `EINVAL`.
    #[test]
    fn the_kernel_carries_exactly_scm_max_fd_descriptors_per_send() {
        let (sender, _receiver) = UnixStream::pair().unwrap();
        let files: Vec<OwnedFd> = (0..=SCM_MAX_FD).map(|_| file("x")).collect();
        let borrowed: Vec<BorrowedFd<'_>> = files.iter().map(AsFd::as_fd).collect();
        let send = |count: usize| {
            let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(SCM_MAX_FD + 1))];
            let mut control = SendAncillaryBuffer::new(&mut space);
            assert!(control.push(SendAncillaryMessage::ScmRights(&borrowed[..count])));
            rustix::net::sendmsg(
                sender.as_fd(),
                &[IoSlice::new(b"x")],
                &mut control,
                SendFlags::NOSIGNAL,
            )
        };
        assert_eq!(send(SCM_MAX_FD), Ok(1));
        assert_eq!(send(SCM_MAX_FD + 1), Err(rustix::io::Errno::INVAL));
    }
}
