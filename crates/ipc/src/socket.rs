//! Rank channel over a stream socket.
//!
//! A rank on the head's host reaches the engine through shared storage; a rank
//! on another host has no shared storage to reach, so it carries the same frames
//! over a stream socket. The stream is one [`Frame`] after another: each
//! frame's length field tells the reader where the next one starts.
//!
//! A response is matched to its request by the envelope's `message_id`, which
//! the engine's end reads from each response it decodes.
//!
//! Socket readiness advances incoming and outgoing frames. Local process-death
//! notifications use a separate descriptor.
//!
//! Both ends are non-blocking and make progress only when called: receive,
//! wait and send calls flush queued writes as far as the socket accepts, so a
//! frame larger than the kernel's send buffer finishes sending across later
//! calls.

use std::collections::{HashMap, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::frame::Frame;
use crate::iceoryx::{IpcError, IpcResult, WakeEvents};
use crate::request::{WorkerRequest, WorkerResponse};

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
    fn fill(&mut self, stream: &mut TcpStream, bound: usize) -> IpcResult<bool> {
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
                Err(error) => {
                    return Err(IpcError::Transport(format!(
                        "reading a rank channel: {error}"
                    )));
                }
            }
        }
        if closed && !self.buffer.is_empty() {
            return Err(IpcError::Transport(
                "rank channel closed with an incomplete frame".into(),
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
/// the channel's in-flight depth, so a stalled peer cannot grow storage use without bound.
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
    fn flush(&mut self, stream: &mut TcpStream) -> IpcResult<()> {
        while let Some(frame) = self.pending.front() {
            match stream.write(&frame[self.offset..]) {
                Ok(0) => {
                    return Err(IpcError::Transport(
                        "rank channel closed while sending".into(),
                    ));
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
                Err(error) => {
                    return Err(IpcError::Transport(format!(
                        "sending on a rank channel: {error}"
                    )));
                }
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
            return Err(IpcError::Transport(format!(
                "waiting for rank channel: {error}"
            )));
        }
    }
}

/// A local wake: one descriptor this process both fires and polls.
///
/// Process death and local worker completion can wake the owner independently
/// of incoming network traffic.
///
/// The descriptor's unread bytes are the wake's only state. Every fire writes
/// one byte and a drain consumes them all, so a wake is reported exactly when
/// its byte is read: one fired while the owner drains is either read and
/// reported by that drain or left readable for the owner's next poll. Wakes
/// fired between two drains coalesce into the one drain that reads them.
struct LocalWake {
    /// Non-blocking end the owner polls and drains.
    reader: UnixStream,
    /// Non-blocking end that senders write one byte to per wake.
    writer: UnixStream,
}

impl LocalWake {
    fn new() -> IpcResult<Self> {
        let (reader, writer) = UnixStream::pair()
            .map_err(|error| IpcError::Transport(format!("creating a local wake: {error}")))?;
        reader.set_nonblocking(true).map_err(|error| {
            IpcError::Transport(format!("making a local wake pollable: {error}"))
        })?;
        writer.set_nonblocking(true).map_err(|error| {
            IpcError::Transport(format!("making a local wake pollable: {error}"))
        })?;
        Ok(Self { reader, writer })
    }

    /// Returns a sender that fires this wake from any thread.
    fn sender(&self) -> Arc<dyn Fn() + Send + Sync> {
        let writer = match self.writer.try_clone() {
            Ok(writer) => writer,
            // Without a cloned writer the returned sender does nothing, so the
            // owner observes the event only through its own re-checks after a
            // wait returns.
            Err(_) => return Arc::new(|| {}),
        };
        Arc::new(move || {
            // A shared reference writes because a wake fires from arbitrary
            // frontend, watcher and completion threads. A write refused
            // because the buffer is full loses nothing: the descriptor
            // already holds unread bytes, so it is readable.
            let _ = (&writer).write(&[1]);
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

/// The engine's end of a socket rank channel.
pub struct SocketClient {
    stream: TcpStream,
    reader: FrameReader,
    writer: FrameWriter,
    /// Responses read from the stream before the caller asked for them.
    ///
    /// Keyed by the response's `message_id` (`None` as zero), so outstanding
    /// requests need distinct identities: a later response with the same
    /// identity replaces an untaken one.
    inbox: HashMap<u64, WorkerResponse>,
    /// Largest encoded message either direction may carry.
    bound: usize,
    /// Terminal failure, reported after complete responses have been
    /// delivered: an IO failure, or a frame that is not a valid response.
    failure: Option<String>,
    /// Descriptor the engine's worker-death watcher fires on.
    local: LocalWake,
}

impl SocketClient {
    /// Connects to the address a rank reported at registration.
    ///
    /// `bound` is the largest encoded message in bytes in either direction,
    /// `depth` the number of frames the send queue holds, and `timeout` bounds
    /// the TCP connect only.
    pub fn connect(
        address: &str,
        bound: usize,
        depth: usize,
        timeout: Duration,
    ) -> IpcResult<Self> {
        let target: SocketAddr = address.parse().map_err(|_| {
            IpcError::Transport(format!("rank channel address {address} is not an address"))
        })?;
        let stream = TcpStream::connect_timeout(&target, timeout)
            .map_err(|error| IpcError::Transport(format!("connecting to rank channel: {error}")))?;

        // Disable Nagle so a small segment is sent without waiting for earlier
        // data to be acknowledged, and make the stream non-blocking for the
        // poll-driven reader and writer.
        stream
            .set_nodelay(true)
            .map_err(|error| IpcError::Transport(format!("disabling Nagle: {error}")))?;
        stream.set_nonblocking(true).map_err(|error| {
            IpcError::Transport(format!("making a rank channel pollable: {error}"))
        })?;
        Ok(Self {
            stream,
            reader: FrameReader::new(),
            writer: FrameWriter::new(depth),
            inbox: HashMap::new(),
            bound,
            failure: None,
            local: LocalWake::new()?,
        })
    }

    /// Returns the sender a worker-death watcher fires.
    pub fn death_wake(&self) -> Arc<dyn Fn() + Send + Sync> {
        self.local.sender()
    }

    /// Returns the descriptor the engine's own wakes fire on.
    pub fn local_wake_fd(&self) -> RawFd {
        self.local.reader.as_raw_fd()
    }

    /// Encodes and sends one request, returning the identity of its response.
    ///
    /// The identity is the request's `message_id`, `None` standing for zero.
    /// Queues the frame and writes as far as the socket accepts. Fails with
    /// the recorded failure once the channel has failed, when the request is
    /// invalid or its encoding exceeds the bound, with
    /// [`IpcError::WouldBlock`] when the send queue is full, and on a write
    /// error.
    pub fn send_request(&mut self, request: &WorkerRequest) -> IpcResult<u64> {
        if let Some(error) = &self.failure {
            return Err(IpcError::Transport(error.clone()));
        }
        let frame = Frame::encode(request, self.bound)?;

        // Flush first so frames the socket has since accepted free queue
        // capacity before the depth check.
        self.writer.flush(&mut self.stream)?;
        self.writer.enqueue(frame)?;
        self.writer.flush(&mut self.stream)?;
        Ok(request.message_id().unwrap_or_default())
    }

    /// Takes the response to one outstanding message, if it has arrived.
    ///
    /// A terminal channel failure is returned only once no received response
    /// remains in the inbox, so complete responses are delivered first.
    pub fn try_recv_response(&mut self, message_id: u64) -> IpcResult<Option<WorkerResponse>> {
        if let Some(response) = self.inbox.remove(&message_id) {
            return Ok(Some(response));
        }
        self.pump()?;
        if let Some(response) = self.inbox.remove(&message_id) {
            return Ok(Some(response));
        }
        if self.inbox.is_empty()
            && let Some(error) = &self.failure
        {
            return Err(IpcError::Transport(error.clone()));
        }
        Ok(None)
    }

    /// Waits for one outstanding response until the deadline passes.
    ///
    /// Returns `Ok(None)` at the deadline, and also once the channel has failed
    /// while other responses still wait in the inbox.
    pub fn recv_response_timeout(
        &mut self,
        message_id: u64,
        timeout: Duration,
    ) -> IpcResult<Option<WorkerResponse>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(response) = self.try_recv_response(message_id)? {
                return Ok(Some(response));
            }
            if self.failure.is_some() || Instant::now() >= deadline {
                return Ok(None);
            }
            self.wait_wake(deadline.saturating_duration_since(Instant::now()))?;
        }
    }

    /// Waits until one of this channel's wake sources fires or the deadline passes.
    pub fn wait_wake(&mut self, timeout: Duration) -> IpcResult<WakeEvents> {
        let wakes = self.drain_wakes()?;
        if wakes.any() {
            return Ok(wakes);
        }
        wait_ready(&mut self.progress_fds(), timeout)?;
        self.drain_wakes()
    }

    /// Readiness interests include writes only while an accepted frame is pending.
    ///
    /// Returns the stream and the local wake descriptor. After either polls
    /// ready, call [`Self::drain_wakes`], which also advances pending writes.
    pub fn progress_fds(&self) -> Vec<libc::pollfd> {
        vec![
            libc::pollfd {
                fd: self.stream.as_raw_fd(),
                events: self.writer.events(),
                revents: 0,
            },
            libc::pollfd {
                fd: self.local_wake_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ]
    }

    /// Reports the wakes already observed without waiting.
    ///
    /// Buffered results and peer death may be reported together; callers drain
    /// complete responses before failing the remaining work.
    pub fn drain_wakes(&mut self) -> IpcResult<WakeEvents> {
        self.pump()?;
        let local = self.local.take();
        Ok(WakeEvents {
            result: !self.inbox.is_empty(),
            death: local || self.failure.is_some(),
            other: false,
        })
    }

    /// Decodes every frame the stream holds into the inbox, keyed by identity.
    ///
    /// Also flushes queued writes while the stream remains open. Transport
    /// failures, and a frame that does not decode to a valid response, are
    /// recorded in `failure` rather than returned, so this returns `Ok` in
    /// every case.
    fn pump(&mut self) -> IpcResult<()> {
        if self.failure.is_none() {
            self.failure = match self.reader.fill(&mut self.stream, self.bound) {
                Ok(false) => self
                    .writer
                    .flush(&mut self.stream)
                    .err()
                    .map(|error| error.to_string()),
                Ok(true) => Some("rank channel closed before its response".into()),
                Err(error) => Some(error.to_string()),
            };
        }
        // A final read may contain complete responses followed by a truncated
        // frame or EOF. Preserve those responses before failing outstanding work.
        while let Some(frame) = self.reader.ready.pop_front() {
            match frame.decode::<WorkerResponse>() {
                Ok(response) => {
                    self.inbox
                        .insert(response.message_id().unwrap_or_default(), response);
                }
                // Frames behind an undecodable one are left unread: the
                // stream's content can no longer be trusted.
                Err(error) => {
                    self.failure.get_or_insert(error.to_string());
                    break;
                }
            }
        }
        Ok(())
    }
}

/// A rank's end of a socket rank channel.
pub struct SocketServer {
    /// Listening socket until the engine's connection is accepted.
    listener: Option<TcpListener>,
    /// The engine's connection once accepted; the channel accepts only one.
    stream: Option<TcpStream>,
    reader: FrameReader,
    writer: FrameWriter,
    /// Largest encoded message either direction may carry.
    bound: usize,
    /// Terminal IO failure, reported after buffered requests have been taken.
    failure: Option<String>,
    /// The address the engine connects to, as it travels in the registration.
    address: String,
    /// Descriptor this rank's own completion callbacks fire on.
    local: LocalWake,
}

impl SocketServer {
    /// Binds the address this rank will report, without awaiting the engine.
    ///
    /// Registration reports an address that must already exist, so binding and
    /// accepting are separate: the rank reports as soon as the address is real
    /// and accepts when the engine connects.
    pub fn bind(host: &str, bound: usize, depth: usize) -> IpcResult<Self> {
        let listener = TcpListener::bind((host, 0))
            .map_err(|error| IpcError::Transport(format!("binding a rank channel: {error}")))?;
        let address = listener
            .local_addr()
            .map_err(|error| {
                IpcError::Transport(format!("reading a rank channel address: {error}"))
            })?
            .to_string();
        Ok(Self {
            listener: Some(listener),
            stream: None,
            reader: FrameReader::new(),
            writer: FrameWriter::new(depth),
            bound,
            failure: None,
            address,
            local: LocalWake::new()?,
        })
    }

    /// Returns the sender this rank's completion callbacks fire.
    ///
    /// A device, transfer or host-lane completion is raised inside the rank,
    /// so it wakes the rank's own service loop and never reaches the engine.
    pub fn completion_wake(&self) -> Arc<dyn Fn() + Send + Sync> {
        self.local.sender()
    }

    /// Returns the address the engine connects to.
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Accepts the engine's connection if one is waiting, without blocking.
    ///
    /// A rank reports its address and then serves; the engine connects once,
    /// some time later. The rank notices that arrival on its next receive
    /// rather than stopping to wait for it.
    pub fn accept_if_pending(&mut self) -> IpcResult<()> {
        if self.stream.is_some() {
            return Ok(());
        }
        let Some(listener) = self.listener.as_ref() else {
            return Ok(());
        };
        listener
            .set_nonblocking(true)
            .map_err(|error| IpcError::Transport(format!("polling for a rank channel: {error}")))?;
        match listener.accept() {
            Ok((stream, _)) => {
                stream
                    .set_nodelay(true)
                    .map_err(|error| IpcError::Transport(format!("disabling Nagle: {error}")))?;
                stream.set_nonblocking(true).map_err(|error| {
                    IpcError::Transport(format!("making a rank channel pollable: {error}"))
                })?;
                self.stream = Some(stream);
                self.listener = None;
                Ok(())
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => Ok(()),
            Err(error) => Err(IpcError::Transport(format!(
                "accepting a rank channel: {error}"
            ))),
        }
    }

    /// Accepts the engine's connection, waiting up to the deadline.
    pub fn accept(&mut self, timeout: Duration) -> IpcResult<()> {
        let deadline = Instant::now() + timeout;
        loop {
            self.accept_if_pending()?;
            if self.stream.is_some() {
                return Ok(());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(IpcError::Transport(
                    "rank channel connection timed out".into(),
                ));
            }
            let listener = self
                .listener
                .as_ref()
                .ok_or_else(|| IpcError::Transport("rank channel has no listener".into()))?;
            wait_ready(
                &mut [libc::pollfd {
                    fd: listener.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                }],
                remaining,
            )?;
        }
    }

    /// Takes one request, if the engine has sent one.
    ///
    /// Returns `Ok(None)` before the engine has connected; it does not accept a
    /// waiting connection itself. When no request is already buffered and no
    /// failure is recorded, it reads the stream and, unless the read fails or
    /// finds the stream closed, flushes queued responses. A terminal failure
    /// is returned once no complete request remains buffered. A frame that
    /// does not decode to a valid request fails the call that takes it.
    pub fn try_recv(&mut self) -> IpcResult<Option<WorkerRequest>> {
        let Some(stream) = self.stream.as_mut() else {
            return Ok(None);
        };
        if let Some(frame) = self.reader.ready.pop_front() {
            return frame.decode().map(Some);
        }
        if self.failure.is_none() {
            self.failure = match self.reader.fill(stream, self.bound) {
                Ok(false) => self
                    .writer
                    .flush(stream)
                    .err()
                    .map(|error| error.to_string()),
                Ok(true) => Some("rank channel is closed".into()),
                Err(error) => Some(error.to_string()),
            };
        }
        if let Some(frame) = self.reader.ready.pop_front() {
            return frame.decode().map(Some);
        }
        if let Some(error) = &self.failure {
            return Err(IpcError::Transport(error.clone()));
        }
        Ok(None)
    }

    /// Waits until a request is readable, a completion fires, or time passes.
    ///
    /// Returns at once when a request is buffered or, failing that, when a
    /// completion wake is already pending, which it consumes. Otherwise it
    /// polls and then flushes queued responses; it never reads the stream,
    /// which the caller's next [`Self::try_recv`] does. Before the engine
    /// connects, the listener stands in for the stream.
    pub fn wait_incoming(&mut self, timeout: Duration) -> IpcResult<()> {
        self.accept_if_pending()?;
        if !self.reader.ready.is_empty() || self.local.take() {
            return Ok(());
        }

        let mut descriptors = vec![libc::pollfd {
            fd: self.local.reader.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        }];
        if let Some(stream) = &self.stream {
            descriptors.push(libc::pollfd {
                fd: stream.as_raw_fd(),
                events: self.writer.events(),
                revents: 0,
            });
        } else if let Some(listener) = &self.listener {
            descriptors.push(libc::pollfd {
                fd: listener.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            });
        }
        wait_ready(&mut descriptors, timeout)?;

        self.accept_if_pending()?;
        if let Some(stream) = &mut self.stream {
            self.writer.flush(stream)?;
        }
        Ok(())
    }

    /// Encodes and sends one response, under the identity it states.
    ///
    /// Queues the frame and writes as far as the socket accepts. Fails before
    /// the engine has connected, when the response is invalid or its encoding
    /// exceeds the bound, with [`IpcError::WouldBlock`] when the send queue is
    /// full, and on a write error. Unlike the shared-storage server, it does
    /// not check the response's identity against received requests.
    pub fn respond(&mut self, response: &WorkerResponse) -> IpcResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| IpcError::Transport("rank channel is not connected".into()))?;
        let frame = Frame::encode(response, self.bound)?;

        self.writer.flush(stream)?;
        self.writer.enqueue(frame)?;
        self.writer.flush(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connected(bound: usize, depth: usize) -> (SocketClient, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = SocketClient::connect(
            &listener.local_addr().unwrap().to_string(),
            bound,
            depth,
            Duration::from_secs(5),
        )
        .unwrap();
        let (peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        (client, peer)
    }

    /// An error response whose encoded message is `filler` bytes of text plus
    /// a fixed envelope.
    fn response_of(message_id: u64, filler: usize) -> WorkerResponse {
        let mut response = WorkerResponse::error(crate::WorkerResponseError {
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
    fn frame_of<M: crate::Message>(message: &M) -> Vec<u8> {
        Frame::encode(message, usize::MAX).unwrap().into_bytes()
    }

    #[test]
    fn completed_responses_survive_a_truncated_last_frame() {
        // The peer writes one complete frame a byte at a time, then a frame cut
        // short, then closes. The complete response must still be delivered;
        // only the truncated one fails, and the channel then reports death.
        let (mut client, mut peer) = connected(1024, 2);
        let first = response_of(1, 5);
        for byte in frame_of(&first) {
            peer.write_all(&[byte]).unwrap();
        }
        let second = frame_of(&response_of(2, 9));
        peer.write_all(&second[..second.len() - 2]).unwrap();
        peer.shutdown(std::net::Shutdown::Write).unwrap();

        let response = client
            .recv_response_timeout(1, Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(response, first);
        let error = client
            .recv_response_timeout(2, Duration::from_secs(5))
            .expect_err("truncation must fail pending work");
        assert!(error.to_string().contains("incomplete frame"), "{error}");
        assert!(client.drain_wakes().unwrap().death);
    }

    #[test]
    fn socket_rejects_oversized_and_undecodable_frames() {
        // Only the length field is sent: a length above the bound must be
        // refused before any of the message arrives.
        let (mut client, mut peer) = connected(1024, 1);
        peer.write_all(&1025_u32.to_le_bytes()).unwrap();
        assert!(
            client
                .recv_response_timeout(0, Duration::from_secs(5))
                .is_err()
        );

        // A whole frame that holds a request is not a response.
        let (mut client, mut peer) = connected(1024, 1);
        peer.write_all(&frame_of(&WorkerRequest::info())).unwrap();
        assert!(
            client
                .recv_response_timeout(0, Duration::from_secs(5))
                .is_err()
        );
        assert!(client.drain_wakes().unwrap().death);
    }

    /// A local wake is reported by the first drain after it fires, wakes
    /// fired before one drain are reported by it together, and a wake fired
    /// after a drain leaves the wake descriptor readable for the owner's next
    /// poll.
    #[test]
    fn a_local_wake_is_reported_once_and_rearms_after_each_drain() {
        let (mut client, _peer) = connected(1024, 1);
        let wake = client.death_wake();
        let readable = |client: &SocketClient| {
            let mut descriptor = [libc::pollfd {
                fd: client.local_wake_fd(),
                events: libc::POLLIN,
                revents: 0,
            }];
            wait_ready(&mut descriptor, Duration::ZERO).unwrap();
            descriptor[0].revents & libc::POLLIN != 0
        };

        assert!(!readable(&client));
        assert!(!client.drain_wakes().unwrap().death);

        wake();
        wake();
        assert!(readable(&client));
        assert!(client.drain_wakes().unwrap().death);
        assert!(!readable(&client));
        assert!(!client.drain_wakes().unwrap().death);

        // Every drain rearms the wake, however many drains preceded it.
        for _ in 0..3 {
            wake();
            assert!(readable(&client), "a wake after a drain must be pollable");
            assert!(client.drain_wakes().unwrap().death);
        }
    }

    #[test]
    fn a_stalled_peer_applies_backpressure_and_partial_writes_resume() {
        let size = 1 << 20;
        let bound = 2 * size;
        let mut server = SocketServer::bind("127.0.0.1", bound, 1).unwrap();
        let mut peer = TcpStream::connect(server.address()).unwrap();
        server.accept(Duration::from_secs(5)).unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

        let capacity: libc::c_int = 4096;
        // Restrict the real socket send buffer so this workload must span writes.
        assert_eq!(
            unsafe {
                libc::setsockopt(
                    server.stream.as_ref().unwrap().as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_SNDBUF,
                    (&capacity as *const libc::c_int).cast(),
                    std::mem::size_of_val(&capacity) as libc::socklen_t,
                )
            },
            0
        );
        // With depth 1, the partially written first frame fills the queue.
        let first = response_of(1, size);
        server.respond(&first).unwrap();
        assert!(matches!(
            server.respond(&response_of(2, 4)),
            Err(IpcError::WouldBlock)
        ));

        // Once the peer reads, driving the rank's wait loop resumes the
        // partial write until the whole frame arrives.
        let expected = frame_of(&first);
        let length = expected.len();
        let (tx, rx) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let mut bytes = vec![0; length];
            peer.read_exact(&mut bytes).unwrap();
            tx.send(bytes).unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        let wire = loop {
            if let Ok(bytes) = rx.try_recv() {
                break bytes;
            }
            assert!(
                Instant::now() < deadline,
                "accepted frame did not finish sending"
            );
            server.wait_incoming(Duration::from_millis(10)).unwrap();
        };
        reader.join().unwrap();
        assert!(wire == expected, "the frame arrived changed");
    }

    #[test]
    fn a_request_and_its_response_cross_a_socket_unchanged() {
        // The channel's contract is that a message arrives as it was sent, and
        // that a response is taken under the identity its request carried, so
        // the engine can match a result to the batch it answers.
        let mut server = SocketServer::bind("127.0.0.1", 1 << 20, 8).expect("rank binds");
        let address = server.address().to_string();
        let answer = response_of(77, 16);

        let sent_answer = answer.clone();
        let accepted = std::thread::spawn(move || {
            server
                .accept(Duration::from_secs(5))
                .expect("engine connects");
            let request = loop {
                if let Some(request) = server.try_recv().expect("request readable") {
                    break request;
                }
            };
            server.respond(&sent_answer).expect("rank answers");
            request
        });

        let mut client = SocketClient::connect(&address, 1 << 20, 8, Duration::from_secs(5))
            .expect("engine connects");
        let mut sent = WorkerRequest::info();
        sent.set_call_id(Some(77));
        let identity = client.send_request(&sent).expect("engine sends");
        assert_eq!(identity, 77);

        let received = accepted.join().expect("rank thread");
        assert_eq!(received, sent);

        let deadline = Instant::now() + Duration::from_secs(5);
        let result = loop {
            if let Some(response) = client.try_recv_response(77).expect("result readable") {
                break response;
            }
            assert!(Instant::now() < deadline, "result did not arrive");
        };
        assert_eq!(result, answer);
    }
}
