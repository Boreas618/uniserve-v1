//! Serving the deployment's HTTP/1.1 connections.
//!
//! hyper cannot hand a connection over: its chunked-encoding position, its
//! write buffer and the request bytes it read ahead all live inside it, and a
//! daemon that a hot restart replaces must know all three to give a response
//! in flight to its successor (design `specs/worker.md` §7.5.1). So the daemon
//! serves its own connections: this module accepts, reads one request at a
//! time, gives the router the same [`Request`] axum's extractors and handlers
//! already take, and frames the response itself — content-length for a body of
//! known size, chunked otherwise, one chunk per body frame, flushed per frame
//! for a streamed response.
//!
//! A response whose body the daemon produces itself carries a [`HandedBody`]
//! in its extensions. The connection takes that body out before it writes the
//! response and drives it through [`crate::http::stream`], so the body stops
//! between two frames when a hot restart asks ([`crate::http::handoff`]): the
//! connection then reports the response frozen, with the HTTP framing it
//! needs, the bytes it did not write, the request bytes it read ahead and a
//! descriptor of the socket, and waits for the restart's decision.
//! [`Decision::Resume`] writes the carried frames and goes on;
//! [`Decision::Adopt`] ends the connection without writing another byte, the
//! successor owning the socket from then on.
//!
//! Only HTTP/1.1 is served, as the design requires: HTTP/2's connection state
//! (the HPACK table, flow-control windows, stream numbering) cannot be
//! exported, while HTTP/1.1's is exactly the framing position and the two byte
//! buffers this module keeps. TLS terminates before the frontend.

use std::io;
use std::os::fd::{AsFd as _, AsRawFd as _, OwnedFd};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use anyhow::Context as _;
use axum::Router;
use axum::body::Body;
use axum::http::header::{self, HeaderName, HeaderValue};
use axum::http::{Method, Request, Response, StatusCode, Version};
use bytes::{Buf as _, Bytes, BytesMut};
use http_body::Body as _;
use tokio::io::{AsyncReadExt as _, AsyncWrite as _, AsyncWriteExt as _};
use tokio::net::{TcpStream, UnixStream};
use tokio::sync::Notify;
use tokio_util::either::Either;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt as _;
use tracing::{debug, trace, warn};

use crate::http::handoff::{self, Decision, FrozenStream};
use crate::http::listener::Listener;
use crate::http::stream::{Handed, HandedBody, Pause};

/// The accepted connection: a TCP stream on a serving address, a Unix stream
/// on a socket address.
type Io = Either<TcpStream, UnixStream>;

/// The largest request head (request line and headers) the server reads.
const MAX_HEAD: usize = 64 << 10;

/// How much is read from the socket at a time.
const READ_CHUNK: usize = 64 << 10;

/// How many de-framed body bytes one pump hands the request's reader before
/// the reader must poll again.
const BODY_CHUNK: usize = 64 << 10;

/// Serves connections accepted on `listener` with `app` until `accepting` is
/// cancelled, then finishes the requests in flight and returns.
///
/// A cancelled `accepting` ends each connection between two requests, as
/// `axum::serve(...).with_graceful_shutdown(...)` did, and the caller bounds
/// the wait (`server_config.shutdown_timeout`).
pub(crate) async fn serve(
    mut listener: Listener,
    app: Router,
    accepting: CancellationToken,
) -> anyhow::Result<()> {
    let mut connections = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            biased;
            () = accepting.cancelled() => break,
            (io, _address) = axum::serve::Listener::accept(&mut listener) => {
                let app = app.clone();
                let stopping = accepting.clone();
                connections.spawn(async move {
                    if let Err(error) = serve_connection(io, app, stopping).await {
                        debug!(%error, "a connection ended with an error");
                    }
                });
            }
        }
    }
    // Stop accepting: the socket closes here, and a Unix socket's file is
    // removed with the listener.
    drop(listener);
    while let Some(joined) = connections.join_next().await {
        if let Err(error) = joined {
            warn!(%error, "a connection task did not finish");
        }
    }
    Ok(())
}

/// Continues one response in flight on a socket the predecessor handed over.
///
/// The predecessor wrote the response head and ended its ownership at a frame
/// boundary, so this process writes the body bytes it produced and the socket
/// did not take (`http.unwritten`, framed as the predecessor frames them: they
/// continue the byte stream exactly where the socket leaves off), then the
/// frames `body` produces from the state the predecessor froze, ends the
/// body's framing, and serves the next request on the connection when it is
/// kept alive. `http.read_ahead` are request bytes the predecessor read and
/// the successor reads as its own.
pub(crate) async fn adopt(
    descriptor: OwnedFd,
    http: proto::handoff::Http,
    mut body: crate::http::stream::ResponseStream,
    app: Router,
) -> anyhow::Result<()> {
    let io = socket_of(descriptor).context("taking over a handed connection")?;
    let mut connection = Connection::new(io, app, CancellationToken::new());
    connection.buffer = BytesMut::from(http.read_ahead.as_slice());
    // The counters continue the record the pause made.
    connection.written = http.written;
    connection.frames = http.frames;
    connection.write(&http.unwritten, None).await?;
    while let Some(frame) = body.next_frame().await {
        connection.write_frame(&frame, http.chunked, None).await?;
    }
    if http.chunked {
        connection.write(b"0\r\n\r\n", None).await?;
    }
    if !http.keep_alive {
        let _ = connection.io.shutdown().await;
        return Ok(());
    }
    connection.run().await
}

/// The accepted stream a handed descriptor names: a Unix socket on a socket
/// address, a TCP socket on an internet address.
fn socket_of(descriptor: OwnedFd) -> io::Result<Io> {
    let mut address = std::mem::MaybeUninit::<libc::sockaddr_storage>::zeroed();
    let mut length = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    // SAFETY: `address` is a writable `sockaddr_storage` and `length` its
    // size, which `getsockname` updates.
    let named = unsafe {
        libc::getsockname(
            descriptor.as_raw_fd(),
            address.as_mut_ptr().cast::<libc::sockaddr>(),
            &mut length,
        )
    };
    if named != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `getsockname` initialised the address's family.
    let family = unsafe { (*address.as_ptr()).ss_family };
    match i32::from(family) {
        libc::AF_UNIX => {
            let stream = std::os::unix::net::UnixStream::from(descriptor);
            stream.set_nonblocking(true)?;
            UnixStream::from_std(stream).map(Either::Right)
        }
        libc::AF_INET | libc::AF_INET6 => {
            let stream = std::net::TcpStream::from(descriptor);
            stream.set_nonblocking(true)?;
            TcpStream::from_std(stream).map(Either::Left)
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a handed descriptor that is not a socket",
        )),
    }
}

/// Serves one accepted connection until the client closes it, the server stops
/// serving, or the connection ends it.
async fn serve_connection(io: Io, app: Router, accepting: CancellationToken) -> anyhow::Result<()> {
    let mut connection = Connection::new(io, app, accepting);
    connection.run().await
}

/// One connection's state between requests: the socket, the bytes read ahead
/// of the parser, and the counters of the response being written.
struct Connection {
    io: Io,
    app: Router,
    accepting: CancellationToken,
    /// Bytes read from the socket and not yet processed: the next request's
    /// head and body, or bytes of the request in flight its reader has not
    /// asked for yet.
    buffer: BytesMut,
    /// Whether the peer closed its writing side.
    peer_closed: bool,
    /// Response bytes queued for the socket that the kernel has not accepted
    /// yet. A pause freezes the response with exactly these bytes, so a slow
    /// client never holds the predecessor here.
    pending: BytesMut,
    /// Response bytes the kernel accepted, head included.
    written: u64,
    /// Body frames of the response being written.
    frames: u64,
}

impl Connection {
    fn new(io: Io, app: Router, accepting: CancellationToken) -> Self {
        Self {
            io,
            app,
            accepting,
            buffer: BytesMut::new(),
            peer_closed: false,
            pending: BytesMut::new(),
            written: 0,
            frames: 0,
        }
    }

    /// Serves requests on the connection until it ends.
    async fn run(&mut self) -> anyhow::Result<()> {
        loop {
            // A server that is shutting down takes no request it did not
            // already read: the request in flight ends and the connection
            // closes.
            if self.accepting.is_cancelled() && self.buffer.is_empty() {
                return Ok(());
            }
            let Some(head) = self.read_head().await? else {
                return Ok(());
            };
            debug!(method = %head.method, target = %head.target, "a request head");

            self.written = 0;
            self.frames = 0;
            let keep_alive = head.keep_alive;
            let mut body = head.body;
            if head.expect_continue {
                self.write(b"HTTP/1.1 100 Continue\r\n\r\n", None).await?;
            }
            let request = body.request(head.method, head.target, head.version, head.headers);
            let Some(response) = self.call(request, &mut body).await? else {
                return Ok(());
            };
            let outcome = self
                .write_response(response, keep_alive, &mut body)
                .await
                .context("writing a response")?;
            debug!(
                keep_alive = matches!(outcome, Outcome::KeepAlive),
                "a response written"
            );
            match outcome {
                Outcome::KeepAlive => {
                    if self.accepting.is_cancelled() {
                        return Ok(());
                    }
                }
                Outcome::Close => {
                    let _ = self.io.shutdown().await;
                    return Ok(());
                }
            }
        }
    }

    /// Runs one request through the router, feeding its body as the handler
    /// polls it; `None` when the client disappeared while the handler ran.
    async fn call(
        &mut self,
        request: Request<Body>,
        body: &mut RequestBody,
    ) -> anyhow::Result<Option<Response<Body>>> {
        let mut future = std::pin::pin!(self.app.clone().oneshot(request));
        loop {
            tokio::select! {
                response = &mut future => {
                    return match response {
                        Ok(response) => Ok(Some(response)),
                        Err(never) => match never {},
                    };
                }
                () = body.wanted.notified() => self.pump(body).await?,
            }
        }
    }

    /// Reads the next request's head, returning `None` at end of stream.
    ///
    /// The bytes the head does not consume stay in [`Self::buffer`] for the
    /// request's body or the next request.
    async fn read_head(&mut self) -> anyhow::Result<Option<Head>> {
        loop {
            if let Some(end) = find_head_end(&self.buffer) {
                let head = self.buffer.split_to(end);
                return match parse_head(&head) {
                    Ok(head) => Ok(Some(head)),
                    Err(error) => {
                        self.answer_bad_request(&error).await?;
                        Ok(None)
                    }
                };
            }
            if self.buffer.len() > MAX_HEAD {
                self.answer_bad_request(&HeadError::TooLarge).await?;
                return Ok(None);
            }
            if !self.read_more().await? {
                if self.buffer.is_empty() {
                    return Ok(None);
                }
                self.answer_bad_request(&HeadError::Incomplete).await?;
                return Ok(None);
            }
        }
    }

    /// Reads more bytes from the socket into the buffer; `false` at end of
    /// stream.
    async fn read_more(&mut self) -> anyhow::Result<bool> {
        if self.peer_closed {
            return Ok(false);
        }
        let mut chunk = [0_u8; READ_CHUNK];
        loop {
            match self.io.read(&mut chunk).await {
                Ok(0) => {
                    self.peer_closed = true;
                    return Ok(false);
                }
                Ok(read) => {
                    self.buffer.extend_from_slice(&chunk[..read]);
                    return Ok(true);
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error).context("reading from a connection"),
            }
        }
    }

    /// De-frames more of the request's body for its reader, reading from the
    /// socket when the buffer holds none.
    async fn pump(&mut self, body: &mut RequestBody) -> anyhow::Result<()> {
        loop {
            let fed = body.de_frame(&mut self.buffer)?;
            if body.complete() || fed {
                return Ok(());
            }
            if !self.read_more().await? {
                // The client stopped inside its body: the request cannot be
                // completed, and the connection cannot serve another.
                self.peer_closed = true;
                body.abandon();
                return Ok(());
            }
        }
    }

    /// Writes a protocol error and closes the connection.
    async fn answer_bad_request(&mut self, error: &HeadError) -> anyhow::Result<()> {
        let status = error.status();
        let body = error.to_string();
        let response = Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .header(header::CONTENT_LENGTH, body.len())
            .header(header::CONNECTION, "close")
            .body(Body::from(body))
            .expect("a fixed error response");
        self.written = 0;
        self.frames = 0;
        let mut absent = RequestBody::empty();
        self.write_response(response, false, &mut absent).await?;
        let _ = self.io.shutdown().await;
        Ok(())
    }

    /// Writes `response`, then ends the body. The returned outcome says
    /// whether the connection serves another request.
    async fn write_response(
        &mut self,
        response: Response<Body>,
        keep_alive: bool,
        body: &mut RequestBody,
    ) -> anyhow::Result<Outcome> {
        let head_only = body.head_request;
        let handed = response
            .extensions()
            .get::<HandedBody>()
            .and_then(HandedBody::take);
        let (parts, mut payload) = response.into_parts();
        let declared = parts
            .headers
            .get(header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok());
        // A handed body has no body of its own: the session produces every
        // frame, so only a length the handler declared is the response's, and
        // without one the body is chunked. A declared zero is the placeholder
        // of the empty body a handed response carries, never its length: a
        // body whose length is zero is chunked as well, and ends at once.
        let (chunked, content_length) = match &handed {
            Some(_) => match declared.filter(|length| *length > 0) {
                Some(length) => (false, Some(length)),
                None => (true, None),
            },
            None => match declared.or_else(|| payload.size_hint().exact()) {
                Some(length) => (false, Some(length)),
                None => (false, None),
            },
        };
        let framed = chunked || content_length.is_some();
        let closing = !keep_alive || !body.complete() || !framed;
        self.write_head(&parts, chunked, content_length, closing)
            .await?;
        if !head_only && !status_without_body(parts.status) {
            let complete = match handed {
                Some(handed) => match self
                    .write_handed(handed, keep_alive, body, chunked, content_length)
                    .await?
                {
                    HandedOutcome::Complete => true,
                    HandedOutcome::Ended => false,
                    HandedOutcome::Adopted => return Ok(Outcome::Close),
                },
                None => self.write_payload(&mut payload, chunked).await?,
            };
            if !complete {
                // Framing cannot be finished over a failed or unhandable body:
                // the connection ends here rather than serving a corrupt one.
                return Ok(Outcome::Close);
            }
            if chunked {
                self.write(b"0\r\n\r\n", None).await?;
            }
        }
        if closing {
            return Ok(Outcome::Close);
        }
        Ok(Outcome::KeepAlive)
    }

    /// Writes the status line and headers of a response.
    async fn write_head(
        &mut self,
        parts: &axum::http::response::Parts,
        chunked: bool,
        content_length: Option<u64>,
        closing: bool,
    ) -> anyhow::Result<()> {
        let reason = parts.status.canonical_reason().unwrap_or("");
        let mut head = BytesMut::with_capacity(256);
        head.extend_from_slice(
            format!("HTTP/1.1 {} {}\r\n", parts.status.as_u16(), reason).as_bytes(),
        );
        for (name, value) in &parts.headers {
            let name = name.as_str();
            // The connection decides the body's framing and the connection's
            // fate, not the handler.
            if name.eq_ignore_ascii_case("connection")
                || name.eq_ignore_ascii_case("transfer-encoding")
                || name.eq_ignore_ascii_case("content-length")
            {
                continue;
            }
            head.extend_from_slice(name.as_bytes());
            head.extend_from_slice(b": ");
            head.extend_from_slice(value.as_bytes());
            head.extend_from_slice(b"\r\n");
        }
        if let Some(length) = content_length {
            head.extend_from_slice(format!("Content-Length: {length}\r\n").as_bytes());
        } else if chunked {
            head.extend_from_slice(b"Transfer-Encoding: chunked\r\n");
        }
        head.extend_from_slice(if closing {
            b"Connection: close\r\n\r\n" as &[u8]
        } else {
            b"Connection: keep-alive\r\n\r\n" as &[u8]
        });
        self.write(&head, None).await.map(|_| ())
    }

    /// Writes the frames of a body the handler produced; `false` when the body
    /// failed before its end.
    async fn write_payload(&mut self, payload: &mut Body, chunked: bool) -> anyhow::Result<bool> {
        loop {
            let frame = std::future::poll_fn(|cx| Pin::new(&mut *payload).poll_frame(cx)).await;
            let Some(frame) = frame else {
                return Ok(true);
            };
            match frame {
                Ok(frame) => {
                    if let Ok(data) = frame.into_data() {
                        self.write_frame(&data, chunked, None).await?;
                    }
                }
                Err(error) => {
                    warn!(%error, "a response body failed while writing");
                    return Ok(false);
                }
            }
        }
    }

    /// Writes a handoffable body, stopping at an event boundary when the hot
    /// restart asks and following its decision.
    async fn write_handed(
        &mut self,
        handed: Handed,
        keep_alive: bool,
        body: &RequestBody,
        chunked: bool,
        content_length: Option<u64>,
    ) -> anyhow::Result<HandedOutcome> {
        let Handed {
            mut stream,
            registration,
            request_id,
        } = handed;
        let mut pause_requested = registration.watch();
        // After the first pause the restart owns the decision: the report is
        // spent, and the response goes on here whatever later requests say.
        let mut paused = false;
        loop {
            if paused {
                // Everything is written here now; a stale request no longer
                // stops the response.
                let _ = self.drain(None).await?;
            } else {
                if *pause_requested.borrow() || registration.paused() {
                    match self
                        .freeze(
                            &mut stream,
                            &registration,
                            &request_id,
                            keep_alive,
                            body,
                            chunked,
                            content_length,
                        )
                        .await?
                    {
                        Frozen::Kept => {
                            paused = true;
                            continue;
                        }
                        Frozen::Complete => return Ok(HandedOutcome::Complete),
                        Frozen::Ended => return Ok(HandedOutcome::Ended),
                        Frozen::Adopted => return Ok(HandedOutcome::Adopted),
                    }
                }
                // Everything queued goes to the socket, but a pause stops the
                // drain where it is: a client that reads slowly never holds
                // this process past the boundary it reached.
                if matches!(self.drain(Some(&pause_requested)).await?, Drain::WouldPause) {
                    continue;
                }
            }
            let next = tokio::select! {
                frame = stream.next_frame() => Some(frame),
                changed = pause_requested.changed(), if !paused => {
                    let _ = changed;
                    None
                }
            };
            match next {
                None => continue,
                Some(Some(frame)) => {
                    let pause = (!paused).then_some(&pause_requested);
                    self.write_frame(&frame, chunked, pause).await?;
                }
                Some(None) => return Ok(HandedOutcome::Complete),
            }
        }
    }

    /// Stops the body at the boundary it reached, reports it frozen and
    /// follows the restart's decision.
    ///
    /// The bytes the socket did not accept and the frames the body produced
    /// before the boundary travel to the successor in order, framed as this
    /// process frames them, so the successor continues the byte stream where
    /// the socket leaves off.
    #[allow(clippy::too_many_arguments)]
    async fn freeze(
        &mut self,
        stream: &mut crate::http::stream::ResponseStream,
        registration: &Arc<crate::http::handoff::Registration>,
        request_id: &str,
        keep_alive: bool,
        body: &RequestBody,
        chunked: bool,
        content_length: Option<u64>,
    ) -> anyhow::Result<Frozen> {
        let mut unwritten = std::mem::take(&mut self.pending);
        let pause = stream
            .pause(|frame| {
                if chunked {
                    unwritten.extend_from_slice(format!("{:x}\r\n", frame.len()).as_bytes());
                    unwritten.extend_from_slice(&frame);
                    unwritten.extend_from_slice(b"\r\n");
                } else {
                    unwritten.extend_from_slice(&frame);
                }
            })
            .await;
        let application = match pause {
            Pause::Complete => {
                // The body ended at its own last boundary: nothing to carry.
                self.pending = unwritten;
                return Ok(Frozen::Complete);
            }
            Pause::Ended => {
                // A covered response whose state cannot be frozen does not
                // keep this process: its connection ends here, as an adopted
                // one does.
                warn!(
                    request_id,
                    "a response in flight cannot be handed over; ending it here"
                );
                let _ = registration.report(None).await;
                return Ok(Frozen::Ended);
            }
            Pause::Frozen(application) => application,
        };
        let http = handoff::Http {
            keep_alive,
            unwritten: unwritten.to_vec(),
            read_ahead: body.read_ahead(&self.buffer),
            written: self.written,
            frames: self.frames,
            chunked,
            content_length,
        };
        let state = application.into_stream(http, request_id.to_owned());
        let descriptor = self.descriptor()?;
        let frozen_at_ns = handoff::now_ns();
        // `report` installs the real decision channel; this one only fills the
        // field until it does.
        let (decision, _undecided) = tokio::sync::oneshot::channel();
        let frozen = FrozenStream {
            id: registration.id(),
            request_id: request_id.to_owned(),
            frozen_at_ns,
            state,
            descriptor,
            decision,
        };
        match registration.report(Some(frozen)).await {
            Decision::Resume => {
                // The carried bytes are written here, from where the socket
                // left off, and the body goes on producing.
                self.pending = unwritten;
                stream.resume().await;
                Ok(Frozen::Kept)
            }
            Decision::Adopt => {
                stream.adopt().await;
                Ok(Frozen::Adopted)
            }
        }
    }

    /// A descriptor of the connection's socket, for the successor to write the
    /// response on.
    fn descriptor(&self) -> anyhow::Result<OwnedFd> {
        match &self.io {
            Either::Left(stream) => stream.as_fd().try_clone_to_owned(),
            Either::Right(stream) => stream.as_fd().try_clone_to_owned(),
        }
        .context("duplicating the connection's socket")
    }

    /// Queues `bytes` for the socket and writes what the kernel takes.
    ///
    /// With `pause` given, the drain stops as soon as the response is asked to
    /// pause, leaving the rest queued: the caller freezes the response with
    /// those bytes. Without it, the drain runs to completion, which the writes
    /// no hot restart hands over (the head of a response, a body the handler
    /// produced) rely on.
    async fn write(
        &mut self,
        bytes: &[u8],
        pause: Option<&tokio::sync::watch::Receiver<bool>>,
    ) -> anyhow::Result<Drain> {
        self.pending.extend_from_slice(bytes);
        self.drain(pause).await
    }

    /// Writes queued bytes while the kernel takes them.
    async fn drain(
        &mut self,
        pause: Option<&tokio::sync::watch::Receiver<bool>>,
    ) -> anyhow::Result<Drain> {
        loop {
            if self.pending.is_empty() {
                return Ok(Drain::Drained);
            }
            if pause.is_some_and(|pause| *pause.borrow()) {
                return Ok(Drain::WouldPause);
            }
            let writing =
                std::future::poll_fn(|cx| Pin::new(&mut self.io).poll_write(cx, &self.pending));
            let accepted = match pause {
                None => writing.await,
                Some(pause) => {
                    let mut asked = pause.clone();
                    tokio::select! {
                        accepted = writing => accepted,
                        changed = asked.changed() => {
                            let _ = changed;
                            if *asked.borrow() {
                                return Ok(Drain::WouldPause);
                            }
                            continue;
                        }
                    }
                }
            };
            let accepted = accepted.context("writing to a connection")?;
            if accepted == 0 {
                anyhow::bail!("the connection closed while writing a response");
            }
            self.pending.advance(accepted);
            self.written += u64::try_from(accepted).unwrap_or(u64::MAX);
            trace!(bytes = accepted, "wrote response bytes");
        }
    }

    /// Queues one body frame, chunked when `chunked`, and writes what the
    /// kernel takes. Only a frame every byte of which the kernel accepted
    /// counts as written.
    async fn write_frame(
        &mut self,
        frame: &Bytes,
        chunked: bool,
        pause: Option<&tokio::sync::watch::Receiver<bool>>,
    ) -> anyhow::Result<Drain> {
        let mut bytes = BytesMut::with_capacity(frame.len() + 16);
        if chunked {
            bytes.extend_from_slice(format!("{:x}\r\n", frame.len()).as_bytes());
            bytes.extend_from_slice(frame);
            bytes.extend_from_slice(b"\r\n");
        } else {
            bytes.extend_from_slice(frame);
        }
        let drain = self.write(&bytes, pause).await?;
        if matches!(drain, Drain::Drained) {
            self.frames += 1;
        }
        Ok(drain)
    }
}

/// What a drain found.
enum Drain {
    /// The kernel accepted every queued byte.
    Drained,
    /// The response was asked to pause before every byte was accepted.
    WouldPause,
}

/// Whether the connection serves another request after a response.
enum Outcome {
    KeepAlive,
    Close,
}

/// What a handoffable body's pause produced.
enum Frozen {
    /// The restart resumed the response here: its carried frames are written
    /// and it goes on.
    Kept,
    /// The body ended at its last boundary.
    Complete,
    /// The response cannot be handed over; its connection closes.
    Ended,
    /// The successor owns the connection now.
    Adopted,
}

/// What a handoffable body's end means for the connection.
enum HandedOutcome {
    /// The body ended at its own last frame.
    Complete,
    /// The body cannot be framed to its end.
    Ended,
    /// The successor owns the socket.
    Adopted,
}

/// Whether a response to `status` carries no body, and so neither framing nor
/// a chunked terminator.
fn status_without_body(status: StatusCode) -> bool {
    status.is_informational()
        || status == StatusCode::NO_CONTENT
        || status == StatusCode::NOT_MODIFIED
}

/// The end of the request head in `buffer`, if it holds one: the index just
/// past the empty line.
fn find_head_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|start| start + 4)
}

/// One request's head.
struct Head {
    method: Method,
    target: axum::http::Uri,
    version: Version,
    headers: axum::http::HeaderMap,
    /// Whether the client takes another request after this response.
    keep_alive: bool,
    /// Whether the client waits for `100 Continue` before its body.
    expect_continue: bool,
    /// The request's body framing and, once the handler polls, its reader.
    body: RequestBody,
}

/// A malformed request head, and the status its answer carries.
#[derive(Debug)]
enum HeadError {
    /// The request line is not `METHOD SP TARGET SP HTTP/1.x`.
    RequestLine,
    /// The request target is not a URI this server serves.
    Target,
    /// A header line is malformed.
    Header,
    /// The body's framing is contradictory.
    Framing,
    /// A request body was declared with a transfer coding this server does not
    /// serve.
    UnsupportedFraming,
    /// The head is larger than [`MAX_HEAD`].
    TooLarge,
    /// The stream ended inside the head.
    Incomplete,
}

impl HeadError {
    fn status(&self) -> StatusCode {
        match self {
            Self::RequestLine | Self::Target | Self::Header | Self::Framing | Self::Incomplete => {
                StatusCode::BAD_REQUEST
            }
            Self::UnsupportedFraming => StatusCode::NOT_IMPLEMENTED,
            Self::TooLarge => StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE,
        }
    }
}

impl std::fmt::Display for HeadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::RequestLine => "malformed request line",
            Self::Target => "malformed request target",
            Self::Header => "malformed request header",
            Self::Framing => "contradictory request framing",
            Self::UnsupportedFraming => "unsupported request transfer coding",
            Self::TooLarge => "request head too large",
            Self::Incomplete => "incomplete request head",
        })
    }
}

/// Parses one request head.
fn parse_head(head: &[u8]) -> Result<Head, HeadError> {
    let head = std::str::from_utf8(head).map_err(|_| HeadError::Header)?;
    let mut lines = head.split("\r\n");
    let request_line = lines.next().ok_or(HeadError::RequestLine)?;
    let mut fields = request_line.split(' ');
    let method = fields.next().ok_or(HeadError::RequestLine)?;
    let target = fields.next().ok_or(HeadError::RequestLine)?;
    let version = fields.next().ok_or(HeadError::RequestLine)?;
    if fields.next().is_some() {
        return Err(HeadError::RequestLine);
    }
    let method = Method::from_bytes(method.as_bytes()).map_err(|_| HeadError::RequestLine)?;
    let target = target
        .parse::<axum::http::Uri>()
        .map_err(|_| HeadError::Target)?;
    let version = match version {
        "HTTP/1.1" => Version::HTTP_11,
        "HTTP/1.0" => Version::HTTP_10,
        _ => return Err(HeadError::RequestLine),
    };
    let mut headers = axum::http::HeaderMap::new();
    for line in lines {
        if line.is_empty() {
            break;
        }
        let (name, value) = line.split_once(':').ok_or(HeadError::Header)?;
        let name = HeaderName::from_bytes(name.trim().as_bytes()).map_err(|_| HeadError::Header)?;
        let value = HeaderValue::from_str(value.trim()).map_err(|_| HeadError::Header)?;
        headers.append(name, value);
    }
    let connection = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(|token| token.trim().to_ascii_lowercase())
        .collect::<Vec<_>>();
    let keep_alive = match version {
        Version::HTTP_11 => !connection.iter().any(|token| token == "close"),
        _ => connection.iter().any(|token| token == "keep-alive"),
    };
    let expect_continue = headers
        .get(header::EXPECT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("100-continue"));
    let body = RequestBody::framing(&headers)?;
    Ok(Head {
        method,
        target,
        version,
        headers,
        keep_alive,
        expect_continue,
        body,
    })
}

/// A request body's framing, and the reader the handler polls for its bytes.
///
/// The connection de-frames the socket's bytes (content-length or chunked)
/// into plain body bytes only when the handler's reader asks: a request whose
/// handler answers without reading its body never has that body read.
struct RequestBody {
    /// What the client still owes.
    framing: Framing,
    /// Whether the request's method is `HEAD`, whose response has no body.
    head_request: bool,
    /// The handler and the connection share their bytes and wakes here.
    shared: Arc<Mutex<BodyShared>>,
    /// The connection is woken when the handler wants bytes.
    wanted: Arc<Notify>,
    /// Whether the body ended.
    ended: bool,
}

/// How the client frames the request body.
#[derive(Debug)]
enum Framing {
    /// No body.
    None,
    /// A body of `remaining` bytes.
    Length { remaining: u64 },
    /// A chunked body.
    Chunked(ChunkedStep),
}

/// Where the chunked parser of a request body stands.
#[derive(Clone, Copy, Debug)]
enum ChunkedStep {
    /// Reading a chunk's size line.
    Size,
    /// Reading the data bytes of a chunk of `remaining` bytes.
    Data { remaining: u64 },
    /// Reading the CRLF that ends a chunk's data.
    DataEnd,
    /// The last chunk arrived; reading the trailer section's empty line.
    Trailers,
}

/// What one step of the chunked parser did.
enum ChunkedOutcome {
    /// It consumed bytes and can be stepped again.
    Progress,
    /// It needs more bytes from the socket.
    NeedMore,
    /// The body ended.
    Ended,
}

impl RequestBody {
    /// A request without a body.
    fn empty() -> Self {
        Self::framed(Framing::None)
    }

    /// The framing a head's headers declare.
    fn framing(headers: &axum::http::HeaderMap) -> Result<Self, HeadError> {
        let transfer = headers.contains_key(header::TRANSFER_ENCODING);
        let length = headers
            .get(header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.parse::<u64>().map_err(|_| HeadError::Framing))
            .transpose()?;
        match (transfer, length) {
            // Both framings at once is the request-smuggling shape; refuse it.
            (true, Some(_)) => Err(HeadError::Framing),
            (true, None) => {
                let chunked = headers
                    .get_all(header::TRANSFER_ENCODING)
                    .iter()
                    .filter_map(|value| value.to_str().ok())
                    .flat_map(|value| value.split(','))
                    .map(|token| token.trim().to_ascii_lowercase())
                    .all(|token| token == "chunked");
                if !chunked {
                    return Err(HeadError::UnsupportedFraming);
                }
                Ok(Self::framed(Framing::Chunked(ChunkedStep::Size)))
            }
            (false, Some(0) | None) => Ok(Self::empty()),
            (false, Some(remaining)) => Ok(Self::framed(Framing::Length { remaining })),
        }
    }

    fn framed(framing: Framing) -> Self {
        let ended = matches!(framing, Framing::None);
        Self {
            framing,
            head_request: false,
            shared: Arc::new(Mutex::new(BodyShared {
                ready: BytesMut::new(),
                ended,
                reader: None,
            })),
            wanted: Arc::new(Notify::new()),
            ended,
        }
    }

    /// The request the handler receives, its body written from [`Self`].
    fn request(
        &mut self,
        method: Method,
        target: axum::http::Uri,
        version: Version,
        headers: axum::http::HeaderMap,
    ) -> Request<Body> {
        self.head_request = method == Method::HEAD;
        let source = BodySource {
            shared: Arc::clone(&self.shared),
            wanted: Arc::clone(&self.wanted),
        };
        let mut request = Request::builder()
            .method(method)
            .uri(target)
            .version(version)
            .body(Body::new(source))
            .expect("a built request");
        *request.headers_mut() = headers;
        request
    }

    /// Whether the body ended, so that the connection serves another request.
    fn complete(&self) -> bool {
        self.ended
    }

    /// Marks the body unreadable because the client stopped inside it.
    fn abandon(&mut self) {
        self.ended = true;
        let mut shared = crate::leader::locked(&self.shared);
        shared.ended = true;
        wake(&mut shared);
    }

    /// The request bytes read from the socket that belong to the next request.
    ///
    /// Only a body that ended leaves the connection's buffer holding the next
    /// request; a body stopped mid-way owns those bytes.
    fn read_ahead(&self, buffer: &BytesMut) -> Vec<u8> {
        if self.ended {
            buffer.to_vec()
        } else {
            Vec::new()
        }
    }

    /// Moves de-framed bytes from `buffer` to the handler until it has
    /// [`BODY_CHUNK`] or the body ends; `true` when it fed the handler
    /// anything.
    fn de_frame(&mut self, buffer: &mut BytesMut) -> anyhow::Result<bool> {
        let mut fed = false;
        loop {
            if self.ended {
                break;
            }
            let mut shared = crate::leader::locked(&self.shared);
            let room = BODY_CHUNK.saturating_sub(shared.ready.len());
            if room == 0 {
                wake(&mut shared);
                return Ok(fed);
            }
            match &mut self.framing {
                Framing::None => {
                    self.ended = true;
                    shared.ended = true;
                    wake(&mut shared);
                    return Ok(fed);
                }
                Framing::Length { remaining } => {
                    if *remaining == 0 {
                        self.ended = true;
                        shared.ended = true;
                        wake(&mut shared);
                        return Ok(fed);
                    }
                    if buffer.is_empty() {
                        wake(&mut shared);
                        return Ok(fed);
                    }
                    let take = usize::try_from((*remaining).min(room as u64).min(buffer.len() as u64))
                        .unwrap_or(usize::MAX);
                    if take == 0 {
                        wake(&mut shared);
                        return Ok(fed);
                    }
                    let chunk = buffer.split_to(take);
                    shared.ready.extend_from_slice(&chunk);
                    *remaining -= u64::try_from(take).unwrap_or(u64::MAX);
                    fed = true;
                }
                Framing::Chunked(state) => {
                    match chunked_step(state, buffer, &mut shared.ready, room)? {
                        ChunkedOutcome::Progress => fed = true,
                        ChunkedOutcome::NeedMore => {
                            wake(&mut shared);
                            return Ok(fed);
                        }
                        ChunkedOutcome::Ended => {
                            self.ended = true;
                            shared.ended = true;
                            wake(&mut shared);
                            return Ok(fed);
                        }
                    }
                }
            }
        }
        let mut shared = crate::leader::locked(&self.shared);
        wake(&mut shared);
        Ok(fed)
    }
}

/// Advances the chunked parser of a request body by at most `room` data bytes.
fn chunked_step(
    state: &mut ChunkedStep,
    buffer: &mut BytesMut,
    ready: &mut BytesMut,
    room: usize,
) -> anyhow::Result<ChunkedOutcome> {
    let mut progressed = false;
    loop {
        let more = if progressed {
            ChunkedOutcome::Progress
        } else {
            ChunkedOutcome::NeedMore
        };
        match *state {
            ChunkedStep::Trailers => match take_line(buffer) {
                None => return Ok(more),
                // The trailer section ends with an empty line. Item 1 hands no
                // request body over, so the server serves no trailers.
                Some(line) if line.is_empty() => return Ok(ChunkedOutcome::Ended),
                Some(_) => return Err(anyhow::anyhow!("request trailers are not served")),
            },
            ChunkedStep::DataEnd => {
                if buffer.len() < 2 {
                    return Ok(more);
                }
                if &buffer[..2] != b"\r\n" {
                    return Err(anyhow::anyhow!("a chunk does not end with CRLF"));
                }
                let _ = buffer.split_to(2);
                *state = ChunkedStep::Size;
                progressed = true;
            }
            ChunkedStep::Size => {
                let Some(line) = take_line(buffer) else {
                    return Ok(more);
                };
                // A chunk's extension follows its size after a semicolon.
                let size = line.split(';').next().unwrap_or_default().trim().to_owned();
                let size = u64::from_str_radix(&size, 16)
                    .map_err(|_| anyhow::anyhow!("malformed chunk size"))?;
                *state = if size == 0 {
                    ChunkedStep::Trailers
                } else {
                    ChunkedStep::Data { remaining: size }
                };
                progressed = true;
            }
            ChunkedStep::Data { remaining } => {
                if buffer.is_empty() || ready.len() >= room {
                    return Ok(more);
                }
                let room = u64::try_from(room - ready.len()).unwrap_or(u64::MAX);
                let take = usize::try_from(remaining.min(room).min(buffer.len() as u64))
                    .unwrap_or(usize::MAX);
                let chunk = buffer.split_to(take);
                ready.extend_from_slice(&chunk);
                let left = remaining - u64::try_from(take).unwrap_or(u64::MAX);
                *state = if left == 0 {
                    ChunkedStep::DataEnd
                } else {
                    ChunkedStep::Data { remaining: left }
                };
                return Ok(ChunkedOutcome::Progress);
            }
        }
    }
}

/// Takes one CRLF-terminated line from `buffer`, without its CRLF; `None` when
/// the line is not complete yet.
fn take_line(buffer: &mut BytesMut) -> Option<String> {
    let end = buffer.windows(2).position(|window| window == b"\r\n")?;
    let line = buffer.split_to(end).split().freeze();
    let _ = buffer.split_to(2);
    Some(String::from_utf8_lossy(&line).into_owned())
}

/// The handler's view of the request body: de-framed bytes, filled by the
/// connection on demand.
struct BodySource {
    shared: Arc<Mutex<BodyShared>>,
    wanted: Arc<Notify>,
}

#[derive(Default)]
struct BodyShared {
    /// De-framed bytes the handler has not polled yet.
    ready: BytesMut,
    /// Whether the body ended.
    ended: bool,
    /// The handler waits for bytes.
    reader: Option<Waker>,
}

/// Wakes the handler's reader, if it waits.
fn wake(shared: &mut BodyShared) {
    if let Some(reader) = shared.reader.take() {
        reader.wake();
    }
}

impl http_body::Body for BodySource {
    type Data = Bytes;
    type Error = io::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Bytes>, io::Error>>> {
        let mut shared = crate::leader::locked(&self.shared);
        if !shared.ready.is_empty() {
            let bytes = shared.ready.split().freeze();
            return Poll::Ready(Some(Ok(http_body::Frame::data(bytes))));
        }
        if shared.ended {
            return Poll::Ready(None);
        }
        shared.reader = Some(cx.waker().clone());
        self.wanted.notify_one();
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use axum::routing::{get, post};
    use tokio::sync::mpsc;

    use super::*;
    use crate::http::handoff::{Handoff, Registration};
    use crate::http::stream::{self, ResponseStream, Session, Step};

    /// The server side of an in-memory socket pair, and the client side.
    fn pair() -> (Io, UnixStream) {
        let (server, client) = UnixStream::pair().expect("a socket pair");
        (Either::Right(server), client)
    }

    /// A pair whose socket buffers are small, so that a write of more than a
    /// few kilobytes cannot be accepted until the peer reads.
    fn small_buffered_pair() -> (Io, UnixStream) {
        let (server, client) = UnixStream::pair().expect("a socket pair");
        for stream in [&server, &client] {
            let fd = std::os::fd::AsRawFd::as_raw_fd(stream);
            for option in [libc::SO_SNDBUF, libc::SO_RCVBUF] {
                let size: libc::c_int = 4096;
                // SAFETY: the descriptor is open and `SO_SNDBUF`/`SO_RCVBUF`
                // take an int.
                let set = unsafe {
                    libc::setsockopt(
                        fd,
                        libc::SOL_SOCKET,
                        option,
                        (&raw const size).cast(),
                        std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                    )
                };
                assert_eq!(set, 0, "a socket buffer of {} bytes", 4096);
            }
        }
        (Either::Right(server), client)
    }

    /// One response read off the wire: status, headers and body.
    struct Wire {
        status: u16,
        headers: String,
        body: String,
    }

    /// The client side of a connection under test, holding the bytes it read
    /// and did not consume yet: two responses may arrive in one read.
    struct TestClient {
        stream: UnixStream,
        buffered: Vec<u8>,
    }

    impl TestClient {
        fn new(stream: UnixStream) -> Self {
            Self {
                stream,
                buffered: Vec::new(),
            }
        }

        /// Writes a request's bytes.
        async fn send(&mut self, bytes: &[u8]) {
            self.stream.write_all(bytes).await.expect("a request");
        }

        /// Reads once from the connection into the buffer, failing the test
        /// rather than hanging.
        async fn fill(&mut self) -> usize {
            let mut chunk = [0_u8; 8192];
            match tokio::time::timeout(std::time::Duration::from_secs(5), self.stream.read(&mut chunk))
                .await
            {
                Ok(Ok(read)) => {
                    self.buffered.extend_from_slice(&chunk[..read]);
                    read
                }
                Ok(Err(error)) => panic!("reading a response: {error}"),
                Err(_) => panic!("a response stalled: nothing arrived for 5 s"),
            }
        }

        /// Reads until the buffer holds at least `len` bytes.
        async fn need(&mut self, len: usize) {
            while self.buffered.len() < len {
                assert!(self.fill().await > 0, "the connection closed early");
            }
        }

        /// Takes the next response's head off the wire.
        async fn read_head(&mut self) -> String {
            let end = loop {
                if let Some(end) = find_head_end(&self.buffered) {
                    break end;
                }
                assert!(self.fill().await > 0, "the connection closed inside a head");
            };
            let head = String::from_utf8_lossy(&self.buffered[..end]).into_owned();
            self.buffered.drain(..end);
            head
        }

        /// Takes the next response off the wire, body included.
        async fn read_response(&mut self) -> Wire {
            let head = self.read_head().await;
            let status = head
                .split(' ')
                .nth(1)
                .and_then(|status| status.parse().ok())
                .expect("a status code");
            let headers = head.to_ascii_lowercase();
            let mut body = Vec::new();
            if headers.contains("transfer-encoding: chunked") {
                loop {
                    let size = loop {
                        if let Some(end) = find_line_end(&self.buffered, 0) {
                            let line = String::from_utf8_lossy(&self.buffered[..end]).into_owned();
                            self.buffered.drain(..end + 2);
                            break line;
                        }
                        assert!(self.fill().await > 0, "the connection closed inside a chunk");
                    };
                    let size =
                        usize::from_str_radix(size.split(';').next().unwrap_or_default().trim(), 16)
                            .expect("a chunk size");
                    if size == 0 {
                        break;
                    }
                    self.need(size + 2).await;
                    body.extend_from_slice(&self.buffered[..size]);
                    self.buffered.drain(..size + 2);
                }
            } else {
                let length = headers
                    .split("\r\n")
                    .find_map(|line| line.strip_prefix("content-length: "))
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                self.need(length).await;
                body.extend_from_slice(&self.buffered[..length]);
                self.buffered.drain(..length);
            }
            Wire {
                status,
                headers,
                body: String::from_utf8_lossy(&body).into_owned(),
            }
        }

        /// Reads everything the connection still writes, and returns it.
        async fn read_to_end(&mut self) -> String {
            while self.fill().await > 0 {}
            String::from_utf8_lossy(&self.buffered).into_owned()
        }
    }

    /// The end of the line starting at `at`, if the buffer holds it.
    fn find_line_end(raw: &[u8], at: usize) -> Option<usize> {
        raw[at..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .map(|end| at + end)
    }

    /// A router over the handlers the connection tests use.
    fn app() -> Router {
        Router::new()
            .route("/hello", get(|| async { "hello" }))
            .route(
                "/echo",
                post(|body: axum::body::Bytes| async move { body }),
            )
    }

    /// Serves `app` on a socket pair and returns the client side.
    fn serve_on_pair(app: Router) -> TestClient {
        let (io, client) = pair();
        tokio::spawn(serve_connection(io, app, CancellationToken::new()));
        TestClient::new(client)
    }

    /// A request head for `request` with the given extra header lines.
    fn head(request: &str, extra: &str) -> String {
        format!("{request}\r\nHost: test\r\n{extra}\r\n")
    }

    #[tokio::test]
    async fn a_content_length_body_reaches_the_handler_and_the_connection_is_kept() {
        let mut client = serve_on_pair(app());
        client
            .send(head("POST /echo HTTP/1.1", "Content-Length: 5\r\n").as_bytes())
            .await;
        client.send(b"hello").await;

        let first = client.read_response().await;
        assert_eq!(first.status, 200);
        assert_eq!(first.body, "hello");
        assert!(
            first.headers.contains("connection: keep-alive"),
            "{}",
            first.headers
        );

        // The same connection serves a second request.
        client.send(head("GET /hello HTTP/1.1", "").as_bytes()).await;
        let second = client.read_response().await;
        assert_eq!(second.body, "hello");
    }

    #[tokio::test]
    async fn a_chunked_body_arrives_chunk_by_chunk() {
        let mut client = serve_on_pair(app());
        client
            .send(head("POST /echo HTTP/1.1", "Transfer-Encoding: chunked\r\n").as_bytes())
            .await;
        for chunk in ["3\r\nhel\r\n", "2\r\nlo\r\n", "0\r\n\r\n"] {
            client.send(chunk.as_bytes()).await;
            tokio::task::yield_now().await;
        }

        let response = client.read_response().await;
        assert_eq!(response.body, "hello");
    }

    #[tokio::test]
    async fn pipelined_requests_are_answered_in_order() {
        let mut client = serve_on_pair(app());
        let mut requests = head("GET /hello HTTP/1.1", "");
        requests.push_str(&head("GET /hello HTTP/1.1", ""));
        client.send(requests.as_bytes()).await;

        let first = client.read_response().await;
        let second = client.read_response().await;
        assert_eq!((first.status, first.body.as_str()), (200, "hello"));
        assert_eq!((second.status, second.body.as_str()), (200, "hello"));
    }

    #[tokio::test]
    async fn a_head_request_answers_without_a_body() {
        let mut client = serve_on_pair(app());
        client.send(head("HEAD /hello HTTP/1.1", "").as_bytes()).await;
        // The HEAD response carries the GET's headers and no body; the next
        // request on the connection proves the framing did not consume it.
        let answered = client.read_head().await;
        assert!(answered.starts_with("HTTP/1.1 200"), "{answered}");
        assert!(answered.contains("Content-Length: 5"), "{answered}");

        client.send(head("GET /hello HTTP/1.1", "").as_bytes()).await;
        let second = client.read_response().await;
        assert_eq!(second.body, "hello");
    }

    /// A session that produces the frames the test sends it, and freezes with
    /// a chat stream's state.
    struct TestSession {
        frames: mpsc::UnboundedReceiver<Bytes>,
        ended: bool,
    }

    impl Session for TestSession {
        fn poll_step(&mut self, cx: &mut Context<'_>) -> Poll<Step> {
            if self.ended {
                return Poll::Ready(Step::Ended);
            }
            match self.frames.poll_recv(cx) {
                Poll::Ready(Some(frame)) => Poll::Ready(Step::Frames(vec![frame])),
                Poll::Ready(None) => {
                    self.ended = true;
                    Poll::Ready(Step::Ended)
                }
                Poll::Pending => Poll::Pending,
            }
        }

        fn migrate(&mut self) -> Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
            Box::pin(std::future::ready(()))
        }

        fn freeze(&mut self) -> Option<stream::Application> {
            Some(stream::Application::ChatStream(Default::default()))
        }

        fn abort(&mut self) {}

        fn detach(&mut self) {}
    }

    /// The response of a handoffable body produced by `session`.
    fn handed_response(
        session: Box<dyn Session>,
        registration: Arc<Registration>,
        request_id: &str,
    ) -> Response<Body> {
        let stream = ResponseStream::start(session, false);
        let body = HandedBody::new(stream, Some(registration), request_id.to_owned())
            .expect("a registration makes a body handoffable");
        let mut response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::empty())
            .expect("a response");
        stream::attach(&mut response, body);
        response
    }

    /// Serves one handoffable response on a socket pair, and hands the test the
    /// client side, the frames' producer and the registry the response
    /// registers in.
    fn serve_handed(
        registry: Handoff,
        registration: Arc<Registration>,
        request_id: &str,
        pair: fn() -> (Io, UnixStream),
    ) -> (TestClient, mpsc::UnboundedSender<Bytes>) {
        let request_id = request_id.to_owned();
        let (frames, produced) = mpsc::unbounded_channel();
        let produced = Arc::new(std::sync::Mutex::new(Some(produced)));
        let taken = Arc::clone(&produced);
        let app = Router::new().route(
            "/stream",
            get(move || {
                let produced = Arc::clone(&taken);
                let registration = Arc::clone(&registration);
                let request_id = request_id.clone();
                async move {
                    let frames = produced
                        .lock()
                        .expect("the frames' receiver")
                        .take()
                        .expect("one streamed request per connection");
                    handed_response(
                        Box::new(TestSession {
                            frames,
                            ended: false,
                        }),
                        registration,
                        &request_id,
                    )
                }
            }),
        );
        let (io, client) = pair();
        tokio::spawn(serve_connection(io, app, CancellationToken::new()));
        let _ = registry;
        (TestClient::new(client), frames)
    }

    /// One streamed response paused at a boundary and resumed by the
    /// predecessor: the client reads the frames from both sides of the pause,
    /// in order, and the connection keeps serving.
    #[tokio::test]
    async fn a_paused_response_resumes_and_writes_its_carried_frames() {
        let registry = Handoff::new();
        let registration = registry.register();
        let (mut client, frames) = serve_handed(
            registry.clone(),
            Arc::clone(&registration),
            "handoff-stream-1",
            pair,
        );
        client.send(head("GET /stream HTTP/1.1", "").as_bytes()).await;
        frames.send(Bytes::from_static(b"data: a\n\n")).unwrap();
        tokio::task::yield_now().await;

        let frozen = tokio::time::timeout(std::time::Duration::from_secs(5), registry.pause())
            .await
            .expect("the pause is answered");
        assert_eq!(frozen.len(), 1, "the response in flight froze");
        let frozen = frozen.into_iter().next().expect("one frozen response");
        assert_eq!(frozen.request_id, "handoff-stream-1");
        assert_eq!(frozen.state.request_id, "handoff-stream-1");
        let http = frozen.state.http.as_ref().expect("the HTTP framing");
        assert!(http.keep_alive, "the connection was kept alive");
        assert!(http.chunked, "a body of unknown length is chunked");
        frozen.decision.send(Decision::Resume).unwrap();

        frames.send(Bytes::from_static(b"data: b\n\n")).unwrap();
        drop(frames);
        let response = client.read_response().await;
        assert_eq!(response.status, 200);
        assert!(response.body.contains("data: a"), "{}", response.body);
        assert!(
            response.body.find("data: a") < response.body.find("data: b"),
            "the frames follow the order they were produced in: {}",
            response.body
        );
    }

    /// One streamed response paused and adopted: this process writes nothing
    /// after the boundary, terminates no body and ends the connection.
    #[tokio::test]
    async fn an_adopted_response_stops_writing_and_closes() {
        let registry = Handoff::new();
        let registration = registry.register();
        let (mut client, frames) = serve_handed(
            registry.clone(),
            Arc::clone(&registration),
            "handoff-stream-2",
            pair,
        );
        client.send(head("GET /stream HTTP/1.1", "").as_bytes()).await;
        frames.send(Bytes::from_static(b"data: a\n\n")).unwrap();
        tokio::task::yield_now().await;

        let frozen = tokio::time::timeout(std::time::Duration::from_secs(5), registry.pause())
            .await
            .expect("the pause is answered");
        assert_eq!(frozen.len(), 1);
        let frozen = frozen.into_iter().next().expect("one frozen response");
        frozen.decision.send(Decision::Adopt).unwrap();
        // The successor writes the carried bytes; this process stops here.
        frames.send(Bytes::from_static(b"data: b\n\n")).unwrap();

        let raw = client.read_to_end().await;
        assert!(raw.starts_with("HTTP/1.1 200"), "{raw}");
        assert!(!raw.contains("data: b"), "nothing follows the boundary: {raw}");
        assert!(
            !raw.ends_with("0\r\n\r\n"),
            "the body was not terminated here: {raw}"
        );
    }

    /// A pause reaches the connection while the kernel refuses the response:
    /// the bytes it did not accept travel to the successor, and the process is
    /// not held by a client that does not read.
    #[tokio::test]
    async fn a_pause_stops_a_write_the_socket_does_not_take() {
        let registry = Handoff::new();
        let registration = registry.register();
        let (mut client, frames) = serve_handed(
            registry.clone(),
            Arc::clone(&registration),
            "handoff-stream-3",
            small_buffered_pair,
        );
        client.send(head("GET /stream HTTP/1.1", "").as_bytes()).await;
        // Far more than these socket buffers take, and the client reads none
        // of it.
        let big = Bytes::from(vec![b'x'; 1 << 20]);
        frames.send(big.clone()).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let frozen = tokio::time::timeout(std::time::Duration::from_secs(5), registry.pause())
            .await
            .expect("a pause is answered while the socket refuses bytes");
        assert_eq!(frozen.len(), 1, "the response in flight froze");
        let frozen = frozen.into_iter().next().expect("one frozen response");
        let http = frozen.state.http.as_ref().expect("the HTTP framing");
        assert!(
            !http.unwritten.is_empty(),
            "the bytes the kernel did not take are carried: written {} of {}",
            http.written,
            big.len()
        );
        assert!(
            http.written < big.len() as u64,
            "only what the kernel took counts as written: {} of {}",
            http.written,
            big.len()
        );
        frozen.decision.send(Decision::Adopt).unwrap();

        // The client reads what the socket holds, and the connection ends
        // without writing the rest.
        let raw = client.read_to_end().await;
        assert!(raw.len() < big.len(), "the rest went to the successor");
        drop(frames);
    }
}
