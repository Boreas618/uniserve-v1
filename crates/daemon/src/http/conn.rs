//! Serving the deployment's HTTP/1.1 connections.
//!
//! hyper cannot hand a connection over: its chunked-encoding position, its
//! write buffer and the request bytes it read ahead all live inside it, and a
//! daemon that a hot restart replaces must know all three to give a response
//! in flight to its successor. So the daemon keeps its own connection state:
//! this module accepts, reads one request at a time, gives the router the same
//! [`Request`] axum's extractors and handlers already take, and frames the
//! response itself. Parsing is `httparse`'s: a request head
//! ([`httparse::Request`]), a chunk size ([`httparse::parse_chunk_size`]) and a
//! trailer section ([`httparse::parse_headers`]); what stays here is the state
//! a handoff carries — the framing position, the bytes queued for the socket
//! and the request bytes read ahead.
//!
//! Every response states its body's framing in its head, and the connection
//! writes exactly that: `Content-Length` for a body whose length is known, a
//! chunked body otherwise, and a body that ends with the connection only for
//! an HTTP/1.0 client, which cannot read chunks. A body that produces more or
//! fewer bytes than its declared length fails the connection instead of
//! writing a corrupt message ([`ResponseFraming`]).
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
//! While a request is served, the connection watches its client
//! ([`Connection::closed`]): a client that closes or resets the connection
//! ends its request at once, whatever the request waits for. A handler still
//! running is dropped, and a response body waiting for its engine's first or
//! next output is dropped; either drops the request's engine events, which
//! cancels the request on the engine. The disconnect is not left for the
//! response's next write to find, which for a request still being admitted
//! comes only after its prefill.
//!
//! Only HTTP/1.1 (and 1.0) is served: HTTP/2's connection state (the HPACK
//! table, flow-control windows, stream numbering) cannot be exported, while
//! HTTP/1.1's is exactly the framing position and the two byte buffers this
//! module keeps. TLS terminates before the daemon.

use std::io;
use std::os::fd::{AsFd as _, OwnedFd};
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
use tracing::{debug, error, trace, warn};

use crate::api::openai::error::ApiError;
use crate::http::handoff::{self, Admissions, Decision, FrozenStream};
use crate::http::listener::Listener;
use crate::http::stream::{Handed, HandedBody, Opened, Pause, failure_document};

/// The accepted connection: a TCP stream on a serving address, a Unix stream
/// on a socket address.
type Io = Either<TcpStream, UnixStream>;

/// What a connection reads requests under, from `server_config`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    /// The largest request head (request line and headers) and the largest
    /// trailer section of a chunked request body, in bytes
    /// (`server_config.max_request_head`).
    pub(crate) head_bytes: usize,
}

impl Limits {
    /// The limits `server` states.
    pub(crate) fn of(server: &crate::options::ServerConfig) -> Self {
        Self {
            head_bytes: server.max_request_head,
        }
    }
}

/// What every connection of one server serves with: its router, the limits it
/// reads requests under, and whether the daemon admits new requests.
#[derive(Clone)]
pub(crate) struct Server {
    /// The routes requests are served by.
    pub(crate) app: Router,
    /// The limits requests are read under.
    pub(crate) limits: Limits,
    /// Whether the daemon admits new requests; a hot restart closes it, and a
    /// connection then ends at its next request boundary.
    pub(crate) admitting: Admissions,
}

/// Serves connections accepted on `listener` with `server` until `accepting` is
/// cancelled, then finishes the requests in flight and returns.
///
/// A cancelled `accepting` ends each connection between two requests, as
/// `axum::serve(...).with_graceful_shutdown(...)` does, and the caller bounds
/// the wait (`server_config.shutdown_timeout`). A closed `admitting` ends them
/// there too, for the other reason a daemon takes no new request: it is
/// handing the deployment over (design §7.5.1 step 1).
pub(crate) async fn serve(
    mut listener: Listener,
    server: Server,
    accepting: CancellationToken,
) -> anyhow::Result<()> {
    let mut connections = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            biased;
            () = accepting.cancelled() => break,
            (io, _address) = axum::serve::Listener::accept(&mut listener) => {
                let server = server.clone();
                let stopping = accepting.clone();
                connections.spawn(async move {
                    if let Err(error) = serve_connection(io, server, stopping).await {
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
/// `started` is called once, before the response's first frame of this
/// process's own — after the predecessor's bytes are out — and `finished` once,
/// after the last frame and the body's terminator.
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
    server: Server,
    started: impl FnOnce(),
    finished: impl FnOnce(),
) -> anyhow::Result<()> {
    let io = socket_of(descriptor).context("taking over a handed connection")?;
    let mut connection = Connection::new(io, server, CancellationToken::new())
        .context("reading the handed connection's receive buffer")?;
    connection.buffer = BytesMut::from(http.read_ahead.as_slice());
    // The counters continue the record the pause made.
    connection.written = http.written;
    connection.frames = http.frames;
    connection.body_bytes = http.body_bytes;
    let framing = ResponseFraming::of_handoff(&http);
    let closing = !http.keep_alive;
    // A head the predecessor held and never wrote is not committed to a status:
    // the bytes in `unwritten` are provisional, so the body's first output
    // decides the head here, exactly as it does on the process that held it.
    let mut started = Some(started);
    if http.head_held && http.written == 0 {
        let opened = tokio::select! {
            opened = body.opened() => opened,
            // Dropping the body on the way out cancels the request.
            () = connection.closed(http.request_body_ended) => {
                anyhow::bail!("the client left before the adopted response's document existed")
            }
        };
        match First::of(opened) {
            First::Document(frame) => {
                connection.write(&http.unwritten, None).await?;
                if let Some(started) = started.take() {
                    started();
                }
                connection.write_frame(&frame, framing, None).await?;
            }
            First::Failure(error, frames) => {
                connection.write_failure(error, frames, closing).await?;
                finished();
                if closing {
                    let _ = connection.io.shutdown().await;
                }
                return Ok(());
            }
        }
    } else {
        connection.write(&http.unwritten, None).await?;
    }
    // The predecessor's bytes are out and the next frame is this process's own:
    // §7.5.1's pause ends here. A body with nothing left is at its end, so it
    // reports the same moment before the terminator rather than not at all.
    loop {
        let frame = tokio::select! {
            frame = body.next_frame() => frame,
            () = connection.closed(http.request_body_ended) => {
                anyhow::bail!("the client left during the adopted response")
            }
        };
        let Some(frame) = frame else {
            break;
        };
        if let Some(started) = started.take() {
            started();
        }
        connection.write_frame(&frame, framing, None).await?;
    }
    if let Some(started) = started.take() {
        started();
    }
    connection.end_body(framing).await?;
    // The response ends here, not when the connection closes: a kept-alive
    // connection serves the next request from `run`, and the handoff's record
    // names this response, not the connection (design §7.5.1).
    finished();
    if closing {
        let _ = connection.io.shutdown().await;
        return Ok(());
    }
    connection.run().await
}

/// The accepted stream a handed descriptor names: a Unix socket on a socket
/// address, a TCP socket on an internet address.
fn socket_of(descriptor: OwnedFd) -> io::Result<Io> {
    use rustix::net::AddressFamily;

    match rustix::net::getsockname(&descriptor)?.address_family() {
        AddressFamily::UNIX => {
            let stream = std::os::unix::net::UnixStream::from(descriptor);
            stream.set_nonblocking(true)?;
            UnixStream::from_std(stream).map(Either::Right)
        }
        AddressFamily::INET | AddressFamily::INET6 => {
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
async fn serve_connection(
    io: Io,
    server: Server,
    accepting: CancellationToken,
) -> anyhow::Result<()> {
    let mut connection = Connection::new(io, server, accepting)
        .context("reading the connection's receive buffer")?;
    connection.run().await
}

/// One connection's state between requests: the socket, the bytes read ahead
/// of the parser, and the counters of the response being written.
struct Connection {
    io: Io,
    app: Router,
    limits: Limits,
    accepting: CancellationToken,
    /// Whether the daemon admits new requests; a hot restart closes it, and
    /// the connection then ends at its next request boundary.
    admitting: Admissions,
    /// Bytes read from the socket and not yet processed: the next request's
    /// head and body, or bytes of the request in flight its reader has not
    /// asked for yet.
    buffer: BytesMut,
    /// Whether the peer closed its writing side, or the connection failed
    /// under a read.
    peer_closed: bool,
    /// Response bytes queued for the socket that the kernel has not accepted
    /// yet. A pause freezes the response with exactly these bytes, so a slow
    /// client never holds the predecessor here.
    pending: BytesMut,
    /// Response bytes the kernel accepted, head included.
    written: u64,
    /// Body frames of the response being written.
    frames: u64,
    /// Body bytes of the response being written that were queued for the
    /// socket, framing excluded: what a declared length is checked against.
    body_bytes: u64,
    /// The most bytes one read takes: the socket's receive buffer
    /// (`service::receive_buffer_bytes`). It is also how many de-framed body
    /// bytes a request's reader may hold unpolled, so a body is read no
    /// further ahead of its handler than one read of the socket.
    read_bytes: usize,
}

/// What one request asks of the response that answers it.
#[derive(Debug, Clone, Copy)]
struct Exchange {
    /// Whether the client takes another request after this response.
    keep_alive: bool,
    /// The request's protocol version, which decides whether an unknown
    /// length can be chunked.
    version: Version,
    /// Whether the request's method is `HEAD`: its response has a head and no
    /// body.
    head_only: bool,
}

impl Connection {
    fn new(io: Io, server: Server, accepting: CancellationToken) -> io::Result<Self> {
        let Server {
            app,
            limits,
            admitting,
        } = server;
        let socket = match &io {
            Either::Left(stream) => stream.as_fd(),
            Either::Right(stream) => stream.as_fd(),
        };
        let read_bytes = service::receive_buffer_bytes(socket)?;
        Ok(Self {
            io,
            app,
            limits,
            accepting,
            admitting,
            buffer: BytesMut::new(),
            peer_closed: false,
            pending: BytesMut::new(),
            written: 0,
            frames: 0,
            body_bytes: 0,
            read_bytes,
        })
    }

    /// Serves requests on the connection until it ends.
    async fn run(&mut self) -> anyhow::Result<()> {
        loop {
            // A server that is shutting down, or handing the deployment
            // over, takes no request it did not already read: the request in
            // flight ends and the connection closes.
            if !self.admitting.open() && self.buffer.is_empty() {
                return Ok(());
            }
            if self.accepting.is_cancelled() && self.buffer.is_empty() {
                return Ok(());
            }
            let Some(head) = self.read_head().await? else {
                return Ok(());
            };
            debug!(method = %head.method, target = %head.target, "a request head");

            self.written = 0;
            self.frames = 0;
            self.body_bytes = 0;
            let exchange = Exchange {
                keep_alive: head.keep_alive,
                version: head.version,
                head_only: head.method == Method::HEAD,
            };
            let mut body = RequestBody::framed(head.framing, self.read_bytes, self.limits);
            if head.expect_continue {
                self.write(b"HTTP/1.1 100 Continue\r\n\r\n", None).await?;
            }
            let request = body.request(head.method, head.target, head.version, head.headers);
            let Some(response) = self.call(request, &mut body).await? else {
                return Ok(());
            };
            let outcome = self
                .write_response(response, exchange, &mut body)
                .await
                .context("writing a response")?;
            debug!(
                keep_alive = matches!(outcome, Outcome::KeepAlive),
                "a response written"
            );
            match outcome {
                Outcome::KeepAlive => {
                    if self.accepting.is_cancelled() || !self.admitting.open() {
                        return Ok(());
                    }
                }
                // A handed-over response's socket belongs to the successor from here: it
                // writes this response's next bytes on the descriptor it was given. Every
                // other ending shuts the socket's write half, and `shutdown` reaches the
                // socket the two processes share, which would fail the successor's first
                // write. Dropping this process's descriptor is all that is left to do.
                Outcome::Adopted => return Ok(()),
                Outcome::Close => {
                    let _ = self.io.shutdown().await;
                    return Ok(());
                }
            }
        }
    }

    /// Runs one request through the router, feeding its body as the handler
    /// polls it; `None` when the client disappeared while the handler ran.
    ///
    /// The handler asks for bytes by bumping the body's ask count, and the
    /// connection serves every ask it has not served yet before it parks. The
    /// count, not the notify, decides: a wakeup can find nothing to feed (the
    /// body already ended, or the socket still owes bytes), and a handler that
    /// parked on those bytes must not be left waiting on a notify that has
    /// already been spent.
    async fn call(
        &mut self,
        request: Request<Body>,
        body: &mut RequestBody,
    ) -> anyhow::Result<Option<Response<Body>>> {
        let mut future = std::pin::pin!(self.app.clone().oneshot(request));
        let mut served = body.asks();
        loop {
            let asked = body.asks();
            if asked != served {
                served = asked;
                self.pump(body).await?;
                continue;
            }
            tokio::select! {
                // The handler runs before the connection parks, so bytes an
                // earlier pump left for it are taken before anything waits.
                biased;
                response = &mut future => {
                    return match response {
                        Ok(response) => Ok(Some(response)),
                        Err(never) => match never {},
                    };
                }
                () = body.wanted.notified() => {}
                // The client left while the handler ran: dropping the handler
                // drops what it holds, so a request it submitted is cancelled
                // (a route that generates before it responds holds its
                // request's engine events here).
                () = self.closed(body.complete()) => return Ok(None),
            }
        }
    }

    /// Reads the next request's head, returning `None` at end of stream.
    ///
    /// The bytes the head does not consume stay in [`Self::buffer`] for the
    /// request's body or the next request.
    async fn read_head(&mut self) -> anyhow::Result<Option<Head>> {
        loop {
            match parse_head(&self.buffer) {
                Ok(Some((head, length))) => {
                    self.buffer.advance(length);
                    return Ok(Some(head));
                }
                Ok(None) => {}
                Err(error) => {
                    self.answer_bad_request(&error).await?;
                    return Ok(None);
                }
            }
            if self.buffer.len() > self.limits.head_bytes {
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
        // Room for as much as one read from the socket takes.
        self.buffer.reserve(self.read_bytes);
        loop {
            match self.io.read_buf(&mut self.buffer).await {
                Ok(0) => {
                    self.peer_closed = true;
                    return Ok(false);
                }
                Ok(_) => return Ok(true),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error).context("reading from a connection"),
            }
        }
    }

    /// Resolves once the client is gone: it closed the connection or its
    /// writing side, or a read failed. Never resolves while the client stays.
    ///
    /// The connection selects this against each wait a request's engine can
    /// hold up (its handler, a held head, a handed body's next frame), so that
    /// a client that leaves ends its request at once. `body_ended` states
    /// whether the request's body was read to its end: before that, the
    /// socket's bytes are the body's and nothing is read here. After it, the
    /// socket's next bytes are the client's next request or its end of
    /// stream. The first read that brings bytes keeps them in
    /// [`Self::buffer`] for that request, and nothing more is read until it
    /// is served, so a client that pipelines is read at most one read ahead.
    /// This is how hyper's HTTP/1 server watches a connection in the middle
    /// of a message, and like its default it takes a client that shuts its
    /// writing side for one that left.
    ///
    /// Cancel safe: a read that completes has moved its bytes into the buffer
    /// before the future can be dropped.
    async fn closed(&mut self, body_ended: bool) {
        loop {
            if self.peer_closed {
                return;
            }
            if !body_ended || !self.buffer.is_empty() {
                return std::future::pending().await;
            }
            // Room for as much as one read from the socket takes.
            self.buffer.reserve(self.read_bytes);
            match self.io.read_buf(&mut self.buffer).await {
                Ok(0) => {
                    debug!("a client closed its connection while its request was served");
                    self.peer_closed = true;
                    return;
                }
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => {
                    debug!(%error, "a connection failed while its request was served");
                    self.peer_closed = true;
                    return;
                }
            }
        }
    }

    /// De-frames more of the request's body for its reader, reading from the
    /// socket when the buffer holds none.
    ///
    /// It never waits on the socket while the reader holds bytes it has not
    /// polled: returning lets the connection poll the handler first, so a body
    /// the buffer already carries cannot leave the handler parked behind a
    /// socket that owes nothing.
    async fn pump(&mut self, body: &mut RequestBody) -> anyhow::Result<()> {
        loop {
            let fed = body.de_frame(&mut self.buffer)?;
            if body.complete() || fed || body.unpolled() {
                return Ok(());
            }
            if !self.read_more().await? {
                // The client stopped inside its body: the request cannot be
                // completed, and the connection cannot serve another.
                self.peer_closed = true;
                body.end();
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
            .body(Body::from(body))
            .map_err(|error| anyhow::anyhow!("building a protocol error response: {error}"))?;
        self.written = 0;
        self.frames = 0;
        self.body_bytes = 0;
        let exchange = Exchange {
            keep_alive: false,
            version: Version::HTTP_11,
            head_only: false,
        };
        let mut absent = RequestBody::framed(BodyFraming::None, self.read_bytes, self.limits);
        self.write_response(response, exchange, &mut absent).await?;
        let _ = self.io.shutdown().await;
        Ok(())
    }

    /// Writes `response`, then ends the body. The returned outcome says
    /// whether the connection serves another request.
    async fn write_response(
        &mut self,
        response: Response<Body>,
        exchange: Exchange,
        body: &mut RequestBody,
    ) -> anyhow::Result<Outcome> {
        let mut handed = response
            .extensions()
            .get::<HandedBody>()
            .and_then(HandedBody::take);
        let (parts, mut payload) = response.into_parts();
        // A handed body states its own length, which the route declared when it
        // attached the body; a body the handler produced states it in its
        // `Content-Length` or its exact size. Without a length the body is
        // chunked, for a client that reads chunks.
        let length = match &handed {
            Some(handed) => handed.length,
            None => declared_length(&parts.headers)?.or_else(|| payload.size_hint().exact()),
        };
        let framing = ResponseFraming::of(length, exchange.version);
        let closing = !exchange.keep_alive || !body.complete() || framing.ends_with_connection();
        let bodiless = exchange.head_only || status_without_body(parts.status);
        let head = head_bytes(&parts, framing, closing);

        // A response answered as one document holds its head until the body's
        // first output: the status is the outcome, so a head sent before the
        // outcome could not state a failure (§7.5.1 step 4 — such a response
        // has not started writing when a handoff pauses it, and the head bytes
        // travel in `http.unwritten` if it does). The head is queued rather
        // than written for the same reason: a pause in this window must be
        // able to freeze it.
        let mut opening = None;
        let held = if bodiless {
            None
        } else {
            handed.as_mut().filter(|handed| handed.stream.head_held())
        };
        if let Some(handed) = held {
            self.pending.extend_from_slice(&head);
            // A one-document response produces nothing until it ends, so this wait can be the
            // request's whole generation. A pause ends it: the restart must not wait for the
            // request, and the queued head travels with the frozen response.
            let mut pause_requested = handed.registration.watch();
            let opened = loop {
                if *pause_requested.borrow() {
                    break None;
                }
                tokio::select! {
                    opened = handed.stream.opened() => break Some(opened),
                    changed = pause_requested.changed() => {
                        if changed.is_err() {
                            break Some(handed.stream.opened().await);
                        }
                    }
                    // The client left before its document existed: nothing of
                    // the response is written, and dropping its body cancels
                    // the request.
                    () = self.closed(body.complete()) => return Ok(Outcome::Close),
                }
            };
            // Frozen or resumed, `write_handed` writes the response from here
            // when the pause came first.
            if let Some(opened) = opened {
                match First::of(opened) {
                    First::Document(frame) => opening = Some(frame),
                    First::Failure(error, frames) => {
                        return self.write_failure(error, frames, closing).await;
                    }
                }
            }
        } else {
            self.write(&head, None).await?;
        }
        if !bodiless {
            let complete = match handed {
                Some(handed) => {
                    let writing = Writing {
                        framing,
                        keep_alive: !closing,
                        request_body_ended: body.complete(),
                    };
                    match self.write_handed(handed, writing, body, opening).await? {
                        HandedOutcome::Complete => true,
                        HandedOutcome::Ended => false,
                        HandedOutcome::Adopted => return Ok(Outcome::Adopted),
                    }
                }
                None => self.write_payload(&mut payload, framing).await?,
            };
            if !complete {
                // Framing cannot be finished over a failed or unhandable body:
                // the connection ends here rather than serving a corrupt one.
                return Ok(Outcome::Close);
            }
            self.end_body(framing).await?;
        }
        if closing {
            return Ok(Outcome::Close);
        }
        Ok(Outcome::KeepAlive)
    }

    /// Ends the response's body as `framing` states: the chunked terminator, or
    /// nothing for a length, which must have been written exactly.
    ///
    /// # Errors
    ///
    /// Fails when the body wrote fewer bytes than its declared length: the
    /// client would wait for the rest, so the connection ends.
    async fn end_body(&mut self, framing: ResponseFraming) -> anyhow::Result<()> {
        match framing {
            ResponseFraming::Chunked => {
                self.write(b"0\r\n\r\n", None).await?;
            }
            ResponseFraming::Length(length) => {
                if self.body_bytes != length {
                    error!(
                        declared = length,
                        written = self.body_bytes,
                        "a response body ended before its declared length"
                    );
                    anyhow::bail!(
                        "the response body wrote {} of its {length} declared bytes",
                        self.body_bytes
                    );
                }
            }
            ResponseFraming::UntilClose => {}
        }
        Ok(())
    }

    /// Answers a response that failed before its head was written: the status
    /// is the failure's, and the body is the sink's document for it.
    async fn write_failure(
        &mut self,
        error: ApiError,
        frames: Vec<Bytes>,
        closing: bool,
    ) -> anyhow::Result<Outcome> {
        // Nothing of the response is written yet: whatever was queued as its
        // head is replaced by the one the failure states.
        self.pending.clear();
        let response = axum::response::IntoResponse::into_response(error);
        let parts = response.into_parts().0;
        let length: u64 = frames.iter().map(|frame| frame.len() as u64).sum();
        let framing = ResponseFraming::Length(length);
        let head = head_bytes(&parts, framing, closing);
        self.pending.extend_from_slice(&head);
        self.body_bytes = 0;
        for frame in &frames {
            self.body_bytes += frame.len() as u64;
            self.pending.extend_from_slice(frame);
        }
        self.drain(None).await?;
        Ok(if closing {
            Outcome::Close
        } else {
            Outcome::KeepAlive
        })
    }

    /// Writes the frames of a body the handler produced; `false` when the body
    /// failed before its end.
    async fn write_payload(
        &mut self,
        payload: &mut Body,
        framing: ResponseFraming,
    ) -> anyhow::Result<bool> {
        loop {
            let frame = std::future::poll_fn(|cx| Pin::new(&mut *payload).poll_frame(cx)).await;
            let Some(frame) = frame else {
                return Ok(true);
            };
            match frame {
                Ok(frame) => {
                    if let Ok(data) = frame.into_data() {
                        self.write_frame(&data, framing, None).await?;
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
        writing: Writing,
        body: &RequestBody,
        opening: Option<Bytes>,
    ) -> anyhow::Result<HandedOutcome> {
        let Handed {
            mut stream,
            registration,
            request_id,
            ..
        } = handed;
        let mut pause_requested = registration.watch();
        // After the first pause the restart owns the decision: the report is
        // spent, and the response goes on here whatever later requests say.
        let mut paused = false;
        // The first frame a held-head body produced before its head was
        // written; it goes out after the queued head, and the pause keeps
        // stopping the response before it.
        let mut opening = opening;
        loop {
            if let Some(frame) = opening.take() {
                self.write_frame(
                    &frame,
                    writing.framing,
                    (!paused).then_some(&pause_requested),
                )
                .await?;
                continue;
            }
            if paused {
                // Everything is written here now; a stale request no longer
                // stops the response.
                let _ = self.drain(None).await?;
            } else {
                if *pause_requested.borrow() {
                    let frozen = self
                        .freeze(&mut stream, &registration, &request_id, writing, body)
                        .await?;
                    match frozen {
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
                // The client left while the body waited for its engine: the
                // body cannot be framed to its end, and dropping it cancels
                // the request.
                () = self.closed(body.complete()) => return Ok(HandedOutcome::Ended),
            };
            match next {
                None => continue,
                Some(Some(frame)) => {
                    let pause = (!paused).then_some(&pause_requested);
                    self.write_frame(&frame, writing.framing, pause).await?;
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
    async fn freeze(
        &mut self,
        stream: &mut crate::http::stream::ResponseStream,
        registration: &crate::http::handoff::Registration,
        request_id: &str,
        writing: Writing,
        body: &RequestBody,
    ) -> anyhow::Result<Frozen> {
        let paused = stream.pause().await;
        // The frames the body produced before the boundary are this
        // response's, whatever the pause found: they join the queued bytes.
        for frame in &paused.frames {
            self.queue_frame(frame, writing.framing)?;
        }
        let application = match paused.pause {
            Pause::Complete => {
                // The body ended at its own last boundary: nothing to carry,
                // and the queued bytes are written here.
                return Ok(Frozen::Complete);
            }
            Pause::Ended => {
                // A response whose state cannot be frozen does not keep this
                // process: its connection ends here, as an adopted one does,
                // and the pause hands it over as nothing.
                warn!(
                    request_id,
                    "a response in flight cannot be handed over; ending it here"
                );
                registration.unhandable();
                return Ok(Frozen::Ended);
            }
            Pause::Frozen(application) => application,
        };
        let unwritten = std::mem::take(&mut self.pending);
        let (framing_chunked, content_length) = writing.framing.handoff();
        let http = handoff::Http {
            keep_alive: writing.keep_alive,
            head_held: stream.head_held(),
            unwritten: unwritten.to_vec(),
            read_ahead: body.read_ahead(&self.buffer),
            written: self.written,
            frames: self.frames,
            chunked: framing_chunked,
            content_length,
            request_body_ended: writing.request_body_ended,
            body_bytes: self.body_bytes,
        };
        let state = proto::handoff::Stream {
            http: Some(http),
            request_id: request_id.to_owned(),
            application: Some(application),
        };
        let descriptor = self.descriptor()?;
        let frozen = FrozenStream::new(request_id.to_owned(), state, descriptor);
        match registration.report(frozen).await {
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
            self.written += accepted as u64;
            trace!(bytes = accepted, "wrote response bytes");
        }
    }

    /// Queues one body frame as `framing` frames it, without writing it.
    ///
    /// An empty frame queues nothing: it carries no byte, and as a chunk it
    /// would be the body's terminator.
    ///
    /// # Errors
    ///
    /// Fails when the frame would take the body past its declared length.
    fn queue_frame(&mut self, frame: &[u8], framing: ResponseFraming) -> anyhow::Result<bool> {
        if frame.is_empty() {
            return Ok(false);
        }
        let length = frame.len() as u64;
        if let ResponseFraming::Length(declared) = framing
            && self.body_bytes + length > declared
        {
            error!(
                declared,
                written = self.body_bytes,
                frame = length,
                "a response body produced more bytes than its declared length"
            );
            anyhow::bail!("the response body exceeds its {declared} declared bytes");
        }
        frame_into(&mut self.pending, frame, framing);
        self.body_bytes += length;
        Ok(true)
    }

    /// Queues one body frame and writes what the kernel takes. Only a frame
    /// every byte of which the kernel accepted counts as written.
    async fn write_frame(
        &mut self,
        frame: &[u8],
        framing: ResponseFraming,
        pause: Option<&tokio::sync::watch::Receiver<bool>>,
    ) -> anyhow::Result<Drain> {
        let queued = self.queue_frame(frame, framing)?;
        let drain = self.drain(pause).await?;
        if queued && matches!(drain, Drain::Drained) {
            self.frames += 1;
        }
        Ok(drain)
    }
}

/// What a handoffable body's writing needs to know of its response.
#[derive(Debug, Clone, Copy)]
struct Writing {
    /// How the body is delimited.
    framing: ResponseFraming,
    /// Whether the connection serves another request after this response.
    keep_alive: bool,
    /// Whether the request's body was read to its end.
    request_body_ended: bool,
}

/// How a response body is delimited on the wire, as its head states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseFraming {
    /// `Content-Length`: exactly this many bytes follow the head.
    Length(u64),
    /// `Transfer-Encoding: chunked`: one chunk per body frame and the
    /// terminating last chunk.
    Chunked,
    /// No framing header: the body ends when the connection closes. Only an
    /// HTTP/1.0 client, which cannot read chunks, gets a body of unknown
    /// length this way.
    UntilClose,
}

impl ResponseFraming {
    /// The framing of a body of `length` bytes, or of unknown length, for a
    /// client speaking `version`.
    fn of(length: Option<u64>, version: Version) -> Self {
        match length {
            Some(length) => Self::Length(length),
            None if version == Version::HTTP_10 => Self::UntilClose,
            None => Self::Chunked,
        }
    }

    /// The framing a handed-over response states in its handoff record.
    fn of_handoff(http: &proto::handoff::Http) -> Self {
        match (http.content_length, http.chunked) {
            (Some(length), _) => Self::Length(length),
            (None, true) => Self::Chunked,
            (None, false) => Self::UntilClose,
        }
    }

    /// The framing as the handoff record states it: whether the body is
    /// chunked, and its declared length.
    fn handoff(self) -> (bool, Option<u64>) {
        match self {
            Self::Length(length) => (false, Some(length)),
            Self::Chunked => (true, None),
            Self::UntilClose => (false, None),
        }
    }

    /// Whether the body ends only with the connection.
    fn ends_with_connection(self) -> bool {
        self == Self::UntilClose
    }
}

/// Appends `frame` to `out` as `framing` frames it: a chunk with its size line
/// and CRLF, or the bytes themselves.
fn frame_into(out: &mut BytesMut, frame: &[u8], framing: ResponseFraming) {
    match framing {
        ResponseFraming::Chunked => {
            out.extend_from_slice(format!("{:x}\r\n", frame.len()).as_bytes());
            out.extend_from_slice(frame);
            out.extend_from_slice(b"\r\n");
        }
        ResponseFraming::Length(_) | ResponseFraming::UntilClose => out.extend_from_slice(frame),
    }
}

/// The body length a handler declared in its `Content-Length`, if it declared
/// one.
///
/// # Errors
///
/// Fails when the header is not a length, which is the handler's defect.
fn declared_length(headers: &axum::http::HeaderMap) -> anyhow::Result<Option<u64>> {
    headers
        .get(header::CONTENT_LENGTH)
        .map(|value| {
            value
                .to_str()
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .with_context(|| format!("a response declared the length {value:?}"))
        })
        .transpose()
}

/// What a held head's body produced first: its document, or the failure the
/// head states instead.
enum First {
    /// The body's first frame, which the held head goes out before.
    Document(Bytes),
    /// The failure the head states, with the sink's document for it.
    Failure(ApiError, Vec<Bytes>),
}

impl First {
    /// What `opened` makes of a held head. One document is promised, so a
    /// body that ends before producing one is a failure rather than an empty
    /// success.
    fn of(opened: Opened) -> Self {
        match opened {
            Opened::Frame(frame) => Self::Document(frame),
            Opened::Failed(error, frames) => Self::Failure(error, frames),
            Opened::Ended => {
                let error = ApiError::server_error(
                    "the response ended before it wrote its document".to_owned(),
                );
                let document = failure_document(&error);
                Self::Failure(error, vec![document])
            }
        }
    }
}

/// The head bytes of one response, as this connection writes them: the status
/// from the response, every header the connection does not own, and the framing
/// and connection headers it does.
fn head_bytes(
    parts: &axum::http::response::Parts,
    framing: ResponseFraming,
    closing: bool,
) -> BytesMut {
    let reason = parts.status.canonical_reason().unwrap_or("");
    let mut head = BytesMut::new();
    head.extend_from_slice(format!("HTTP/1.1 {} {}\r\n", parts.status.as_u16(), reason).as_bytes());
    for (name, value) in &parts.headers {
        // The connection decides the body's framing and the connection's
        // fate, not the handler.
        if name == header::CONNECTION
            || name == header::TRANSFER_ENCODING
            || name == header::CONTENT_LENGTH
        {
            continue;
        }
        head.extend_from_slice(name.as_str().as_bytes());
        head.extend_from_slice(b": ");
        head.extend_from_slice(value.as_bytes());
        head.extend_from_slice(b"\r\n");
    }
    match framing {
        ResponseFraming::Length(length) => {
            head.extend_from_slice(format!("Content-Length: {length}\r\n").as_bytes());
        }
        ResponseFraming::Chunked => head.extend_from_slice(b"Transfer-Encoding: chunked\r\n"),
        ResponseFraming::UntilClose => {}
    }
    head.extend_from_slice(if closing {
        b"Connection: close\r\n\r\n" as &[u8]
    } else {
        b"Connection: keep-alive\r\n\r\n" as &[u8]
    });
    head
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
    /// The successor took the response over and writes it on the socket this
    /// connection shares: nothing more is written here, and the socket is not
    /// shut down, because that would end the successor's half of it too.
    Adopted,
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
    /// The body cannot be framed to its end: it cannot be handed over, or its
    /// client left.
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

/// How many header slots a head or trailer section in `bytes` can need: one
/// per line it holds, since every header takes a line of its own.
fn header_slots(bytes: &[u8]) -> usize {
    bytes.iter().filter(|&&byte| byte == b'\n').count()
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
    /// How the request's body is framed.
    framing: BodyFraming,
}

/// A malformed request head, and the status its answer carries.
#[derive(Debug)]
enum HeadError {
    /// The request line or a header is not HTTP/1.x syntax.
    Syntax(httparse::Error),
    /// The request target is not a URI this server serves.
    Target,
    /// The body's framing is contradictory.
    Framing,
    /// A request body was declared with a transfer coding this server does not
    /// serve.
    UnsupportedFraming,
    /// The head is larger than `server_config.max_request_head`.
    TooLarge,
    /// The stream ended inside the head.
    Incomplete,
}

impl HeadError {
    fn status(&self) -> StatusCode {
        match self {
            Self::Syntax(_) | Self::Target | Self::Framing | Self::Incomplete => {
                StatusCode::BAD_REQUEST
            }
            Self::UnsupportedFraming => StatusCode::NOT_IMPLEMENTED,
            Self::TooLarge => StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE,
        }
    }
}

impl std::fmt::Display for HeadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Syntax(error) => write!(formatter, "malformed request head: {error}"),
            Self::Target => formatter.write_str("malformed request target"),
            Self::Framing => formatter.write_str("contradictory request framing"),
            Self::UnsupportedFraming => formatter.write_str("unsupported request transfer coding"),
            Self::TooLarge => formatter.write_str("request head too large"),
            Self::Incomplete => formatter.write_str("incomplete request head"),
        }
    }
}

/// Parses the request head at the start of `buffer`: the head with the bytes
/// it took, or `None` while the buffer holds only part of one.
fn parse_head(buffer: &[u8]) -> Result<Option<(Head, usize)>, HeadError> {
    let mut slots = vec![httparse::EMPTY_HEADER; header_slots(buffer)];
    let mut request = httparse::Request::new(&mut slots);
    let length = match request.parse(buffer).map_err(HeadError::Syntax)? {
        httparse::Status::Complete(length) => length,
        httparse::Status::Partial => return Ok(None),
    };
    // A complete parse states every part of the request line.
    let (Some(method), Some(target), Some(minor)) = (request.method, request.path, request.version)
    else {
        return Err(HeadError::Incomplete);
    };
    let method = Method::from_bytes(method.as_bytes())
        .map_err(|_| HeadError::Syntax(httparse::Error::Token))?;
    let target = target
        .parse::<axum::http::Uri>()
        .map_err(|_| HeadError::Target)?;
    let version = if minor == 1 {
        Version::HTTP_11
    } else {
        Version::HTTP_10
    };
    let mut headers = axum::http::HeaderMap::with_capacity(request.headers.len());
    for field in request.headers.iter() {
        let name = HeaderName::from_bytes(field.name.as_bytes())
            .map_err(|_| HeadError::Syntax(httparse::Error::HeaderName))?;
        let value = HeaderValue::from_bytes(field.value)
            .map_err(|_| HeadError::Syntax(httparse::Error::HeaderValue))?;
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
    let framing = BodyFraming::of(&headers)?;
    let head = Head {
        method,
        target,
        version,
        headers,
        keep_alive,
        expect_continue,
        framing,
    };
    Ok(Some((head, length)))
}

/// How the client frames a request body (RFC 9112 §6.3).
#[derive(Debug)]
enum BodyFraming {
    /// No body.
    None,
    /// A body of `remaining` bytes.
    Length { remaining: u64 },
    /// A chunked body.
    Chunked(ChunkedDecoder),
}

impl BodyFraming {
    /// The framing a head's headers declare.
    fn of(headers: &axum::http::HeaderMap) -> Result<Self, HeadError> {
        let transfer = headers.contains_key(header::TRANSFER_ENCODING);
        let length = headers
            .get(header::CONTENT_LENGTH)
            .map(|value| {
                value
                    .to_str()
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .ok_or(HeadError::Framing)
            })
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
                Ok(Self::Chunked(ChunkedDecoder::new()))
            }
            (false, Some(0) | None) => Ok(Self::None),
            (false, Some(remaining)) => Ok(Self::Length { remaining }),
        }
    }
}

/// The de-framing of a chunked message body (RFC 9112 §7.1): each chunk's
/// size through [`httparse::parse_chunk_size`], its data, and the trailer
/// section after the last chunk through [`httparse::parse_headers`], whose
/// fields are read and discarded: no route reads a trailer.
#[derive(Debug)]
pub(super) struct ChunkedDecoder {
    step: ChunkedStep,
}

/// Where a chunked decoder stands.
#[derive(Debug, Clone, Copy)]
enum ChunkedStep {
    /// Reading a chunk's size line.
    Size,
    /// Reading the data bytes of a chunk of `remaining` bytes.
    Data { remaining: u64 },
    /// Reading the CRLF that ends a chunk's data.
    DataEnd,
    /// The last chunk arrived; reading the trailer section.
    Trailers,
    /// The body ended.
    Ended,
}

/// What one decoding pass did.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Decoded {
    /// It moved bytes and can be called again.
    Progress,
    /// It needs more bytes, or more room in its output.
    NeedMore,
    /// The body ended.
    Ended,
}

impl ChunkedDecoder {
    pub(super) fn new() -> Self {
        Self {
            step: ChunkedStep::Size,
        }
    }

    /// Moves body bytes from the framed `buffer` to `out` until `out` holds
    /// `room` bytes, the buffer runs out, or the body ends. A trailer section
    /// longer than `trailer_bytes` is refused, as a head that long is.
    ///
    /// # Errors
    ///
    /// Fails on malformed framing: a bad chunk size, a chunk not followed by
    /// CRLF, or a malformed or overlong trailer section.
    pub(super) fn decode(
        &mut self,
        buffer: &mut BytesMut,
        out: &mut BytesMut,
        room: usize,
        trailer_bytes: usize,
    ) -> anyhow::Result<Decoded> {
        let mut progressed = false;
        loop {
            let more = if progressed {
                Decoded::Progress
            } else {
                Decoded::NeedMore
            };
            match self.step {
                ChunkedStep::Ended => return Ok(Decoded::Ended),
                ChunkedStep::Size => match httparse::parse_chunk_size(buffer)
                    .map_err(|_| anyhow::anyhow!("malformed chunk size"))?
                {
                    httparse::Status::Partial => return Ok(more),
                    httparse::Status::Complete((length, size)) => {
                        buffer.advance(length);
                        self.step = if size == 0 {
                            ChunkedStep::Trailers
                        } else {
                            ChunkedStep::Data { remaining: size }
                        };
                        progressed = true;
                    }
                },
                ChunkedStep::Data { remaining } => {
                    let room = room.saturating_sub(out.len());
                    if buffer.is_empty() || room == 0 {
                        return Ok(more);
                    }
                    let take = usize::try_from(remaining)
                        .unwrap_or(usize::MAX)
                        .min(room)
                        .min(buffer.len());
                    out.extend_from_slice(&buffer.split_to(take));
                    let left = remaining - take as u64;
                    self.step = if left == 0 {
                        ChunkedStep::DataEnd
                    } else {
                        ChunkedStep::Data { remaining: left }
                    };
                    progressed = true;
                }
                ChunkedStep::DataEnd => {
                    if buffer.len() < 2 {
                        return Ok(more);
                    }
                    anyhow::ensure!(&buffer[..2] == b"\r\n", "a chunk does not end with CRLF");
                    buffer.advance(2);
                    self.step = ChunkedStep::Size;
                    progressed = true;
                }
                ChunkedStep::Trailers => {
                    let mut slots = vec![httparse::EMPTY_HEADER; header_slots(buffer)];
                    match httparse::parse_headers(buffer, &mut slots)
                        .map_err(|error| anyhow::anyhow!("malformed trailer section: {error}"))?
                    {
                        httparse::Status::Partial => {
                            anyhow::ensure!(
                                buffer.len() <= trailer_bytes,
                                "a trailer section longer than {trailer_bytes} bytes"
                            );
                            return Ok(more);
                        }
                        httparse::Status::Complete((length, _)) => {
                            buffer.advance(length);
                            self.step = ChunkedStep::Ended;
                            return Ok(Decoded::Ended);
                        }
                    }
                }
            }
        }
    }
}

/// A request body's framing, and the reader the handler polls for its bytes.
///
/// The connection de-frames the socket's bytes (content-length or chunked)
/// into plain body bytes only when the handler's reader asks: a request whose
/// handler answers without reading its body never has that body read.
struct RequestBody {
    /// What the client still owes.
    framing: BodyFraming,
    /// How many de-framed bytes the reader may hold unpolled.
    window: usize,
    limits: Limits,
    /// The handler and the connection share their bytes and wakes here.
    shared: Arc<Mutex<BodyShared>>,
    /// The connection is woken when the handler wants bytes.
    wanted: Arc<Notify>,
    /// Whether the body ended.
    ended: bool,
}

impl RequestBody {
    /// The body `framing` frames, de-framed at most `window` bytes ahead of its
    /// reader.
    fn framed(framing: BodyFraming, window: usize, limits: Limits) -> Self {
        let ended = matches!(framing, BodyFraming::None);
        Self {
            framing,
            window,
            limits,
            shared: Arc::new(Mutex::new(BodyShared {
                ready: BytesMut::new(),
                ended,
                reader: None,
                asks: 0,
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
        let source = BodySource {
            shared: Arc::clone(&self.shared),
            wanted: Arc::clone(&self.wanted),
        };
        // Every part is already a valid `http` value — `method`, `target` and
        // `version` come from the parser as parsed values — so building the
        // request cannot fail and there is no error to report to a caller that
        // is the connection itself.
        #[expect(
            clippy::expect_used,
            reason = "a request from parsed parts cannot fail to build"
        )]
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

    /// How many times the handler has asked for bytes it did not have.
    fn asks(&self) -> u64 {
        service::lock(&self.shared).asks
    }

    /// Whether the handler holds de-framed bytes it has not polled yet.
    fn unpolled(&self) -> bool {
        !service::lock(&self.shared).ready.is_empty()
    }

    /// Ends the body, waking its reader: its framing ended, or the client
    /// stopped inside it and nothing more of it can be read.
    fn end(&mut self) {
        self.ended = true;
        let mut shared = service::lock(&self.shared);
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

    /// Moves de-framed bytes from `buffer` to the handler until it holds the
    /// body's window or the body ends; `true` when it fed the handler
    /// anything.
    fn de_frame(&mut self, buffer: &mut BytesMut) -> anyhow::Result<bool> {
        if self.ended {
            return Ok(false);
        }
        let mut shared = service::lock(&self.shared);
        let before = shared.ready.len();
        let ended = match &mut self.framing {
            BodyFraming::None => true,
            BodyFraming::Length { remaining } => {
                let room = self.window.saturating_sub(shared.ready.len());
                let take = usize::try_from(*remaining)
                    .unwrap_or(usize::MAX)
                    .min(room)
                    .min(buffer.len());
                shared.ready.extend_from_slice(&buffer.split_to(take));
                *remaining -= take as u64;
                *remaining == 0
            }
            BodyFraming::Chunked(decoder) => {
                decoder.decode(
                    buffer,
                    &mut shared.ready,
                    self.window,
                    self.limits.head_bytes,
                )? == Decoded::Ended
            }
        };
        let fed = shared.ready.len() > before;
        if ended {
            shared.ended = true;
        }
        wake(&mut shared);
        drop(shared);
        self.ended = ended;
        Ok(fed)
    }
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
    /// How many times the handler asked for bytes it did not have.
    ///
    /// The connection serves every ask it has not served yet, so a wakeup that
    /// finds nothing to feed (the body already ended, or the socket still owes
    /// bytes) cannot leave the handler waiting on a notify already spent.
    asks: u64,
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
        let mut shared = service::lock(&self.shared);
        if !shared.ready.is_empty() {
            let bytes = shared.ready.split().freeze();
            return Poll::Ready(Some(Ok(http_body::Frame::data(bytes))));
        }
        if shared.ended {
            return Poll::Ready(None);
        }
        shared.reader = Some(cx.waker().clone());
        shared.asks = shared.asks.wrapping_add(1);
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

    /// The server of `app` the tests serve with: the schema's default limits,
    /// admitting.
    fn server(app: Router) -> Server {
        Server {
            app,
            limits: Limits::of(&crate::options::ServerConfig::default()),
            admitting: Admissions::default(),
        }
    }

    /// The server side of an in-memory socket pair, and the client side.
    fn pair() -> (Io, UnixStream) {
        let (server, client) = UnixStream::pair().expect("a socket pair");
        (Either::Right(server), client)
    }

    /// A pair whose socket buffers are the smallest the kernel grants, so that
    /// a write larger than [`buffered`] bytes cannot be accepted until the peer
    /// reads.
    fn small_buffered_pair() -> (Io, UnixStream) {
        let (server, client) = UnixStream::pair().expect("a socket pair");
        for stream in [&server, &client] {
            // The kernel raises a smaller request to its own minimum.
            rustix::net::sockopt::set_socket_send_buffer_size(stream, 1).expect("SO_SNDBUF");
            rustix::net::sockopt::set_socket_recv_buffer_size(stream, 1).expect("SO_RCVBUF");
        }
        (Either::Right(server), client)
    }

    /// How many bytes the two ends of `client`'s pair buffer between them: what
    /// a writer can queue before the reader takes any.
    fn buffered(client: &UnixStream) -> usize {
        let send = rustix::net::sockopt::socket_send_buffer_size(client).expect("SO_SNDBUF");
        let receive = rustix::net::sockopt::socket_recv_buffer_size(client).expect("SO_RCVBUF");
        send + receive
    }

    /// One response read off the wire: status, headers and body.
    struct Wire {
        status: u16,
        headers: axum::http::HeaderMap,
        body: String,
    }

    impl Wire {
        /// The value of the header `name`, or `""`.
        fn header(&self, name: &str) -> &str {
            self.headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
        }
    }

    /// The client side of a connection under test, holding the bytes it read
    /// and did not consume yet: two responses may arrive in one read. Heads are
    /// parsed by `httparse` and chunked bodies by the connection's own
    /// [`ChunkedDecoder`], so the client reads exactly what an HTTP/1.1 client
    /// reads.
    struct TestClient {
        stream: UnixStream,
        buffered: BytesMut,
    }

    impl TestClient {
        fn new(stream: UnixStream) -> Self {
            Self {
                stream,
                buffered: BytesMut::new(),
            }
        }

        /// Writes a request's bytes.
        async fn send(&mut self, bytes: &[u8]) {
            self.stream.write_all(bytes).await.expect("a request");
        }

        /// Reads once from the connection into the buffer; `0` at end of
        /// stream.
        async fn fill(&mut self) -> usize {
            self.buffered.reserve(1);
            self.stream
                .read_buf(&mut self.buffered)
                .await
                .expect("reading a response")
        }

        /// Takes the next response's head off the wire.
        async fn read_head(&mut self) -> (u16, axum::http::HeaderMap) {
            loop {
                let mut slots = vec![httparse::EMPTY_HEADER; header_slots(&self.buffered)];
                let mut response = httparse::Response::new(&mut slots);
                if let httparse::Status::Complete(length) =
                    response.parse(&self.buffered).expect("a response head")
                {
                    let status = response.code.expect("a status code");
                    let mut headers = axum::http::HeaderMap::new();
                    for field in response.headers.iter() {
                        headers.append(
                            HeaderName::from_bytes(field.name.as_bytes()).expect("a header name"),
                            HeaderValue::from_bytes(field.value).expect("a header value"),
                        );
                    }
                    self.buffered.advance(length);
                    return (status, headers);
                }
                assert!(self.fill().await > 0, "the connection closed inside a head");
            }
        }

        /// Takes the next response off the wire, body included.
        async fn read_response(&mut self) -> Wire {
            let (status, headers) = self.read_head().await;
            let mut body = BytesMut::new();
            let chunked = headers
                .get(header::TRANSFER_ENCODING)
                .is_some_and(|value| value == "chunked");
            if chunked {
                let mut decoder = ChunkedDecoder::new();
                while decoder
                    .decode(&mut self.buffered, &mut body, usize::MAX, usize::MAX)
                    .expect("a chunked body")
                    != Decoded::Ended
                {
                    assert!(self.fill().await > 0, "the connection closed inside a body");
                }
            } else {
                let length = headers
                    .get(header::CONTENT_LENGTH)
                    .and_then(|value| value.to_str().ok()?.parse::<usize>().ok())
                    .unwrap_or_default();
                while self.buffered.len() < length {
                    assert!(self.fill().await > 0, "the connection closed inside a body");
                }
                body.extend_from_slice(&self.buffered.split_to(length));
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

    /// A router over the handlers the connection tests use.
    fn app() -> Router {
        Router::new()
            .route("/hello", get(|| async { "hello" }))
            .route("/echo", post(|body: axum::body::Bytes| async move { body }))
    }

    /// Serves `app` on a socket pair and returns the client side.
    fn serve_on_pair(app: Router) -> TestClient {
        let (io, client) = pair();
        tokio::spawn(serve_connection(io, server(app), CancellationToken::new()));
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
        assert_eq!(first.header("connection"), "keep-alive");

        // The same connection serves a second request.
        client
            .send(head("GET /hello HTTP/1.1", "").as_bytes())
            .await;
        let second = client.read_response().await;
        assert_eq!(second.body, "hello");
    }

    #[tokio::test]
    async fn a_handler_that_reads_its_body_last_sees_every_byte_the_connection_read_ahead() {
        // The connection reads the request's body with its head, before the
        // handler has asked for a byte, and parks on the handler's response.
        // The handler polls its body only once the test lets it, so no further
        // notify is coming for the bytes the connection already holds: they
        // must still reach it.
        let entered = Arc::new(Notify::new());
        let gate = Arc::new(Notify::new());
        let (entering, gated) = (Arc::clone(&entered), Arc::clone(&gate));
        let app = Router::new()
            .route("/hello", get(|| async { "hello" }))
            .route(
                "/late-echo",
                post(move |request: Request<Body>| {
                    let (entering, gated) = (Arc::clone(&entering), Arc::clone(&gated));
                    async move {
                        entering.notify_one();
                        gated.notified().await;
                        axum::body::to_bytes(request.into_body(), usize::MAX)
                            .await
                            .expect("the request body")
                    }
                }),
            );
        let mut client = serve_on_pair(app);

        // One write: the request's head and body, then the next request's
        // head, which the body's end leaves for the connection to serve.
        let mut requests = head("POST /late-echo HTTP/1.1", "Content-Length: 5\r\n");
        requests.push_str("hello");
        requests.push_str(&head("GET /hello HTTP/1.1", ""));
        client.send(requests.as_bytes()).await;
        // The handler runs once the connection parsed the head, which came
        // with the body in the one read of the one write.
        entered.notified().await;
        gate.notify_one();

        let first = client.read_response().await;
        assert_eq!(first.status, 200);
        assert_eq!(
            first.body, "hello",
            "the handler reads every byte the connection read ahead"
        );
        assert_eq!(first.header("connection"), "keep-alive");

        // The body the handler read was complete, so the connection serves the
        // request that came behind it.
        let second = client.read_response().await;
        assert_eq!((second.status, second.body.as_str()), (200, "hello"));
    }

    #[tokio::test]
    async fn a_chunked_body_with_extensions_and_trailers_reaches_the_handler() {
        let mut client = serve_on_pair(app());
        client
            .send(head("POST /echo HTTP/1.1", "Transfer-Encoding: chunked\r\n").as_bytes())
            .await;
        // A chunk extension, and a trailer field, which RFC 9112 lets a client
        // send and the server discards.
        for chunk in [
            "3;name=value\r\nhel\r\n",
            "2\r\nlo\r\n",
            "0\r\nChecksum: 1\r\n\r\n",
        ] {
            client.send(chunk.as_bytes()).await;
        }

        let response = client.read_response().await;
        assert_eq!(response.body, "hello");
        assert_eq!(response.header("connection"), "keep-alive");

        // The trailer section ended the body: the next request is read as one.
        client
            .send(head("GET /hello HTTP/1.1", "").as_bytes())
            .await;
        assert_eq!(client.read_response().await.body, "hello");
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
        client
            .send(head("HEAD /hello HTTP/1.1", "").as_bytes())
            .await;
        // The HEAD response carries the GET's headers and no body; the next
        // request on the connection proves the framing did not consume it.
        let (status, headers) = client.read_head().await;
        assert_eq!(status, 200);
        assert_eq!(headers[header::CONTENT_LENGTH], "5");

        client
            .send(head("GET /hello HTTP/1.1", "").as_bytes())
            .await;
        let second = client.read_response().await;
        assert_eq!(second.body, "hello");
    }

    /// A body of unknown length is chunked for an HTTP/1.1 client, which keeps
    /// its connection; an HTTP/1.0 client, which cannot read chunks, reads it
    /// to the connection's end.
    #[tokio::test]
    async fn a_body_of_unknown_length_is_chunked_or_ends_with_the_connection() {
        let streamed = || async {
            Body::from_stream(futures::stream::iter([
                Ok::<_, std::convert::Infallible>(Bytes::from_static(b"hel")),
                Ok(Bytes::from_static(b"lo")),
            ]))
        };
        let app = || Router::new().route("/streamed", get(streamed));

        let mut client = serve_on_pair(app());
        client
            .send(head("GET /streamed HTTP/1.1", "").as_bytes())
            .await;
        let chunked = client.read_response().await;
        assert_eq!(chunked.header("transfer-encoding"), "chunked");
        assert_eq!(chunked.header("connection"), "keep-alive");
        assert_eq!(chunked.body, "hello");

        let mut client = serve_on_pair(app());
        client
            .send(head("GET /streamed HTTP/1.0", "").as_bytes())
            .await;
        let raw = client.read_to_end().await;
        let (_, body) = raw.split_once("\r\n\r\n").expect("a head");
        assert!(
            !raw.to_ascii_lowercase().contains("transfer-encoding"),
            "{raw}"
        );
        assert!(raw.contains("Connection: close"), "{raw}");
        assert_eq!(body, "hello");
    }

    /// A body that produces more bytes than the length its handler declared is
    /// not written past that length: the connection ends instead of writing a
    /// message a client would misread.
    #[tokio::test]
    async fn a_body_longer_than_its_declared_length_ends_the_connection() {
        let app = Router::new().route(
            "/overlong",
            get(|| async {
                Response::builder()
                    .header(header::CONTENT_LENGTH, "3")
                    .body(Body::from_stream(futures::stream::iter([Ok::<
                        _,
                        std::convert::Infallible,
                    >(
                        Bytes::from_static(b"hello"),
                    )])))
                    .expect("a response")
            }),
        );
        let mut client = serve_on_pair(app);
        client
            .send(head("GET /overlong HTTP/1.1", "").as_bytes())
            .await;
        let raw = client.read_to_end().await;
        let (head, body) = raw.split_once("\r\n\r\n").expect("a head");
        assert!(head.contains("Content-Length: 3"), "{raw}");
        assert_eq!(body, "", "no byte past the declared length is written");
    }

    /// A session that produces the frames the test sends it, and freezes with
    /// a chat stream's state.
    struct TestSession {
        frames: mpsc::UnboundedReceiver<Bytes>,
        ended: bool,
        /// Whether this response can be handed over; `false` models a
        /// response whose state cannot be frozen.
        freezable: bool,
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
            self.freezable
                .then(|| stream::Application::ChatStream(Default::default()))
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
        let stream = ResponseStream::start(session, None);
        let body = HandedBody::new(stream, registration, request_id.to_owned(), None);
        let mut response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::empty())
            .expect("a response");
        stream::attach(&mut response, body);
        response
    }

    /// A session of a response answered as one document: the one step it takes,
    /// or a body that is over at once.
    struct DocumentSession {
        step: Option<Step>,
    }

    impl Session for DocumentSession {
        fn poll_step(&mut self, _cx: &mut Context<'_>) -> Poll<Step> {
            match self.step.take() {
                Some(step) => Poll::Ready(step),
                None => Poll::Ready(Step::Ended),
            }
        }

        fn migrate(&mut self) -> Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
            Box::pin(std::future::ready(()))
        }

        fn freeze(&mut self) -> Option<stream::Application> {
            Some(stream::Application::ChatResponse(Default::default()))
        }

        fn abort(&mut self) {}

        fn detach(&mut self) {}
    }

    /// The response of a handoffable body that holds its head: one document,
    /// whose status is its outcome.
    fn held_response(
        session: Box<dyn Session>,
        registration: Arc<Registration>,
        request_id: &str,
    ) -> Response<Body> {
        let stream = ResponseStream::start_held(session);
        let body = HandedBody::new(stream, registration, request_id.to_owned(), None);
        let mut response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::empty())
            .expect("a response");
        stream::attach(&mut response, body);
        response
    }

    /// Serves one held-head response on a socket pair and answers the client
    /// side of it.
    fn serve_document(step: Option<Step>) -> TestClient {
        let step = Arc::new(std::sync::Mutex::new(step));
        let registry = Handoff::new();
        let registration = registry.register();
        let app = Router::new().route(
            "/document",
            get(move || {
                let step = service::lock(&step).take();
                let registration = Arc::clone(&registration);
                async move { held_response(Box::new(DocumentSession { step }), registration, "r") }
            }),
        );
        serve_on_pair(app)
    }

    /// One handoffable response served on a socket pair: the client side, the
    /// frames' producer, and the connection's task.
    struct Handing {
        client: TestClient,
        frames: mpsc::UnboundedSender<Bytes>,
        connection: tokio::task::JoinHandle<anyhow::Result<()>>,
    }

    /// Serves one handoffable response on a socket pair: `freezable` false
    /// serves a response that cannot be handed over.
    fn serve_handed(
        registration: Arc<Registration>,
        request_id: &str,
        pair: fn() -> (Io, UnixStream),
        freezable: bool,
    ) -> Handing {
        let request_id = request_id.to_owned();
        let (frames, produced) = mpsc::unbounded_channel();
        let produced = Arc::new(std::sync::Mutex::new(Some(produced)));
        let app = Router::new().route(
            "/stream",
            get(move || {
                let frames = service::lock(&produced)
                    .take()
                    .expect("one streamed request per connection");
                let registration = Arc::clone(&registration);
                let request_id = request_id.clone();
                async move {
                    handed_response(
                        Box::new(TestSession {
                            frames,
                            ended: false,
                            freezable,
                        }),
                        registration,
                        &request_id,
                    )
                }
            }),
        );
        let (io, client) = pair();
        let connection = tokio::spawn(serve_connection(io, server(app), CancellationToken::new()));
        Handing {
            client: TestClient::new(client),
            frames,
            connection,
        }
    }

    /// A daemon that is handing the deployment over takes no new request: a
    /// connection that sends one ends at that boundary without a response,
    /// and its client reaches the successor by reconnecting (design §7.5.1
    /// step 1). A response already in flight is what the pause is for, and it
    /// is not this connection.
    #[tokio::test]
    async fn a_connection_ends_between_requests_when_the_daemon_stops_admitting() {
        let app = Router::new().route("/health", get(|| async { "ok" }));
        let admitting = Admissions::default();
        admitting.close();
        let (io, client) = pair();

        // The request is in the socket before the connection runs, so the
        // case under test is the same on every schedule: a request arrived,
        // and the connection must end without taking it.
        let mut client = TestClient::new(client);
        client
            .send(b"GET /health HTTP/1.1\r\nhost: test\r\n\r\n")
            .await;
        let server = Server {
            admitting,
            ..server(app)
        };
        serve_connection(io, server, CancellationToken::new())
            .await
            .expect("the connection ends without an error");

        // A peer that closes with request bytes it never read resets the
        // connection (`ECONNRESET`); one that read them closes it (end of
        // stream). Either is an ending without a response, which the client
        // answers by reconnecting; a response byte would arrive before either.
        let mut received = Vec::new();
        if let Err(error) = client.stream.read_to_end(&mut received).await {
            assert_eq!(
                error.kind(),
                std::io::ErrorKind::ConnectionReset,
                "the connection ends by closing: {error}"
            );
        }
        assert!(
            received.is_empty(),
            "the connection wrote no response: {:?}",
            String::from_utf8_lossy(&received)
        );
    }

    /// A stream of a handed body that the test sends frames to.
    fn test_stream(freezable: bool) -> (mpsc::UnboundedSender<Bytes>, ResponseStream) {
        let (frames, produced) = mpsc::unbounded_channel();
        let body = ResponseStream::start(
            Box::new(TestSession {
                frames: produced,
                ended: false,
                freezable,
            }),
            None,
        );
        (frames, body)
    }

    /// The server side of a pair as the predecessor's socket, and a duplicate
    /// of it for the successor.
    fn predecessor_socket() -> (UnixStream, OwnedFd, TestClient) {
        let (server, client) = pair();
        let Either::Right(predecessor) = server else {
            unreachable!("a socket pair is a Unix stream")
        };
        let descriptor = predecessor
            .as_fd()
            .try_clone_to_owned()
            .expect("a duplicate of the socket");
        (predecessor, descriptor, TestClient::new(client))
    }

    /// The handoff record of a chunked response whose head the predecessor
    /// wrote (`written` bytes so far) and whose request body ended.
    fn chunked_http(keep_alive: bool, written: u64) -> handoff::Http {
        handoff::Http {
            keep_alive,
            head_held: false,
            unwritten: Vec::new(),
            read_ahead: Vec::new(),
            written,
            frames: 0,
            chunked: true,
            content_length: None,
            request_body_ended: true,
            body_bytes: 0,
        }
    }

    /// The successor serves the request the predecessor read ahead before it
    /// stopped: those bytes are the successor's, not the socket's.
    #[tokio::test]
    async fn an_adopted_connection_serves_the_request_read_ahead() {
        let (frames, body) = test_stream(true);
        let (mut predecessor, descriptor, mut client) = predecessor_socket();
        let head = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n";
        predecessor
            .write_all(head)
            .await
            .expect("the predecessor's bytes");
        drop(frames);

        let http = handoff::Http {
            read_ahead: head_bytes_of("GET /hello HTTP/1.1"),
            ..chunked_http(true, head.len() as u64)
        };
        let ended = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&ended);
        let started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let recorded = Arc::clone(&started);
        let adopted = tokio::spawn(adopt(
            descriptor,
            http,
            body,
            server(app()),
            move || {
                recorded.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            },
            move || {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            },
        ));

        let response = client.read_response().await;
        assert_eq!(response.status, 200);
        assert_eq!(response.body, "");
        // The client sends nothing: these bytes came from the predecessor.
        let ahead = client.read_response().await;
        assert_eq!(ahead.body, "hello");
        assert_eq!(
            started.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a response with no frame left is still recorded as adopted, once"
        );
        assert_eq!(
            ended.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the adopted response ended before its connection went on to the next one"
        );
        drop(client);
        adopted
            .await
            .expect("the adopting task")
            .expect("an adopted connection ends quietly");
    }

    /// An adopted response whose connection is not kept alive ends after its
    /// terminator, and the successor writes nothing on the socket after that.
    #[tokio::test]
    async fn an_adopted_response_of_a_closed_connection_ends_after_its_terminator() {
        let (frames, body) = test_stream(true);
        let (mut predecessor, descriptor, mut client) = predecessor_socket();
        let head = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
        predecessor
            .write_all(head)
            .await
            .expect("the predecessor's bytes");
        drop(predecessor);
        frames.send(Bytes::from_static(b"data: a\n\n")).unwrap();
        drop(frames);

        let recorded = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let going_on = Arc::clone(&recorded);
        let ended = Arc::clone(&recorded);
        let adopted = tokio::spawn(adopt(
            descriptor,
            chunked_http(false, head.len() as u64),
            body,
            server(app()),
            move || going_on.store(true, std::sync::atomic::Ordering::SeqCst),
            move || {
                assert!(
                    ended.load(std::sync::atomic::Ordering::SeqCst),
                    "a response is recorded as adopted before it is recorded as ended"
                );
            },
        ));

        let response = client.read_response().await;
        assert_eq!(response.body, "data: a\n\n");
        adopted
            .await
            .expect("the adopting task")
            .expect("an adopted connection ends quietly");
        // Nothing follows the terminator: the connection is over.
        assert_eq!(client.read_to_end().await, "");
    }

    /// One request head, as a pipelined request puts it in the connection's
    /// buffer.
    fn head_bytes_of(request: &str) -> Vec<u8> {
        format!("{request}\r\nHost: test\r\n\r\n").into_bytes()
    }

    /// One streamed response paused at a boundary and resumed by the
    /// predecessor: the client reads the frames from both sides of the pause,
    /// in order, and the connection keeps serving.
    #[tokio::test]
    async fn a_paused_response_resumes_and_writes_its_carried_frames() {
        let registry = Handoff::new();
        let mut handing = serve_handed(registry.register(), "handoff-stream-1", pair, true);
        handing
            .client
            .send(head("GET /stream HTTP/1.1", "").as_bytes())
            .await;
        handing
            .frames
            .send(Bytes::from_static(b"data: a\n\n"))
            .unwrap();

        let frozen = registry.pause().await;
        assert_eq!(frozen.len(), 1, "the response in flight froze");
        let frozen = frozen.into_iter().next().expect("one frozen response");
        assert_eq!(frozen.request_id, "handoff-stream-1");
        assert_eq!(frozen.state.request_id, "handoff-stream-1");
        let http = frozen.state.http.as_ref().expect("the HTTP framing");
        assert!(http.keep_alive, "the connection was kept alive");
        assert!(http.chunked, "a body of unknown length is chunked");
        frozen.decide(Decision::Resume);

        handing
            .frames
            .send(Bytes::from_static(b"data: b\n\n"))
            .unwrap();
        drop(handing.frames);
        let response = handing.client.read_response().await;
        assert_eq!(response.status, 200);
        assert_eq!(response.body, "data: a\n\ndata: b\n\n");
    }

    /// One streamed response paused and adopted: this process writes nothing
    /// after the boundary, terminates no body and hands its socket over.
    #[tokio::test]
    async fn an_adopted_response_stops_writing_and_hands_its_socket_over() {
        let registry = Handoff::new();
        let mut handing = serve_handed(registry.register(), "handoff-stream-2", pair, true);
        handing
            .client
            .send(head("GET /stream HTTP/1.1", "").as_bytes())
            .await;
        handing
            .frames
            .send(Bytes::from_static(b"data: a\n\n"))
            .unwrap();

        let frozen = registry.pause().await;
        assert_eq!(frozen.len(), 1);
        let frozen = frozen.into_iter().next().expect("one frozen response");
        // The successor's copy of the socket is this record's descriptor, and
        // nothing here shuts it: the response ends for the client when that copy
        // is gone too, not when this process stops writing.
        let successor = frozen
            .descriptor
            .try_clone()
            .expect("the successor's socket");
        frozen.decide(Decision::Adopt);
        drop(successor);
        // The successor writes the carried bytes; this process stops here.
        handing
            .frames
            .send(Bytes::from_static(b"data: b\n\n"))
            .unwrap();

        let raw = handing.client.read_to_end().await;
        assert!(raw.starts_with("HTTP/1.1 200"), "{raw}");
        assert!(
            !raw.contains("data: b"),
            "nothing follows the boundary: {raw}"
        );
        assert!(
            !raw.ends_with("0\r\n\r\n"),
            "the body was not terminated here: {raw}"
        );
    }

    /// The socket an adopted response leaves is still writable by the process
    /// that took it over. The predecessor's copy of the connection shares that
    /// socket, so a `shutdown` here would shut the write half for the
    /// successor too.
    #[tokio::test]
    async fn an_adopted_response_leaves_its_socket_writable_for_the_successor() {
        use std::io::Write as _;

        let registry = Handoff::new();
        let mut handing = serve_handed(registry.register(), "handoff-stream-3", pair, true);
        handing
            .client
            .send(head("GET /stream HTTP/1.1", "").as_bytes())
            .await;
        handing
            .frames
            .send(Bytes::from_static(b"data: a\n\n"))
            .unwrap();

        let frozen = registry.pause().await;
        let frozen = frozen.into_iter().next().expect("one frozen response");
        // The descriptor the successor writes on is the same socket the
        // predecessor's connection holds.
        let mut successor = std::os::unix::net::UnixStream::from(
            frozen
                .descriptor
                .try_clone()
                .expect("the successor's socket"),
        );
        frozen.decide(Decision::Adopt);
        // The predecessor's connection ends, so a `shutdown` there would be in
        // effect before the successor writes.
        handing
            .connection
            .await
            .expect("the connection's task")
            .expect("an adopted connection ends quietly");

        successor
            .write_all(b"data: b\n\n")
            .expect("the adopted socket is still writable");
        drop(successor);

        let raw = handing.client.read_to_end().await;
        assert!(raw.starts_with("HTTP/1.1 200"), "{raw}");
        // The successor's bytes go on the wire unframed because a writer that is
        // not the connection does not chunk them, and that is the point: they
        // arrive.
        assert!(
            raw.contains("data: b"),
            "the successor's frame reached the client: {raw}"
        );
    }

    /// The successor writes the byte stream the predecessor left: the bytes
    /// it did not write first, then its own frames, then the terminator, and
    /// it serves the next request on the connection.
    #[tokio::test]
    async fn an_adopted_response_continues_the_byte_stream() {
        let (frames, body) = test_stream(true);
        // The predecessor wrote the head and one frame, and produced one more
        // frame that the socket did not take.
        let (mut predecessor, descriptor, mut client) = predecessor_socket();
        let written = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n9\r\ndata: a\n\n\r\n";
        predecessor
            .write_all(written)
            .await
            .expect("the predecessor's bytes");

        let http = handoff::Http {
            unwritten: b"9\r\ndata: b\n\n\r\n".to_vec(),
            frames: 1,
            body_bytes: 18,
            ..chunked_http(true, written.len() as u64)
        };
        let adopted = tokio::spawn(adopt(descriptor, http, body, server(app()), || {}, || {}));
        frames.send(Bytes::from_static(b"data: c\n\n")).unwrap();
        drop(frames);

        let response = client.read_response().await;
        assert_eq!(response.status, 200);
        assert_eq!(response.body, "data: a\n\ndata: b\n\ndata: c\n\n");

        // The connection is kept alive and the successor serves on it.
        client
            .send(head("GET /hello HTTP/1.1", "").as_bytes())
            .await;
        let next = client.read_response().await;
        assert_eq!(next.body, "hello");
        drop(client);
        adopted
            .await
            .expect("the adopting task")
            .expect("an adopted connection ends quietly");
    }

    /// A response whose state cannot be frozen does not stay here: it is
    /// handed over as nothing, and its connection ends at the boundary
    /// rather than keeping the process that is leaving.
    #[tokio::test]
    async fn a_response_that_cannot_be_frozen_ends_its_connection() {
        let registry = Handoff::new();
        let mut handing = serve_handed(registry.register(), "handoff-stream-4", pair, false);
        handing
            .client
            .send(head("GET /stream HTTP/1.1", "").as_bytes())
            .await;
        handing
            .frames
            .send(Bytes::from_static(b"data: a\n\n"))
            .unwrap();
        // Pause only once the frame reached the socket: what this process
        // wrote before the boundary stays written.
        while !handing
            .client
            .buffered
            .windows(7)
            .any(|window| window == b"data: a")
        {
            handing.client.fill().await;
        }

        let frozen = registry.pause().await;
        assert!(
            frozen.is_empty(),
            "a response that cannot be frozen is not handed over"
        );

        // What this process wrote stays written; nothing follows it, and the
        // connection is closed rather than kept for a successor.
        let raw = handing.client.read_to_end().await;
        assert!(raw.starts_with("HTTP/1.1 200"), "{raw}");
        assert!(raw.contains("data: a"), "{raw}");
        assert!(
            !raw.ends_with("0\r\n\r\n"),
            "the body was not terminated: {raw}"
        );
    }

    /// A pause reaches the connection while the kernel refuses the response:
    /// the bytes it did not accept travel to the successor, and the process is
    /// not held by a client that does not read.
    #[tokio::test]
    async fn a_pause_stops_a_write_the_socket_does_not_take() {
        let registry = Handoff::new();
        let mut handing = serve_handed(
            registry.register(),
            "handoff-stream-5",
            small_buffered_pair,
            true,
        );
        handing
            .client
            .send(head("GET /stream HTTP/1.1", "").as_bytes())
            .await;
        // Far more than the pair's buffers take, and the client reads only
        // its start.
        let big = Bytes::from(vec![b'x'; 4 * buffered(&handing.client.stream)]);
        handing.frames.send(big.clone()).unwrap();
        // A body byte on the client's side means the connection is writing
        // the frame, which the buffers cannot take whole: the writer is
        // blocked on the socket, or about to be.
        handing.client.read_head().await;
        while handing.client.buffered.is_empty() {
            assert!(handing.client.fill().await > 0, "the body's first bytes");
        }

        let frozen = registry.pause().await;
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
        // The successor's copy of the socket is this record's descriptor, and
        // nothing here shuts it: the response ends for the client when that copy
        // is gone.
        let successor = frozen
            .descriptor
            .try_clone()
            .expect("the successor's socket");
        frozen.decide(Decision::Adopt);
        drop(successor);

        // The client reads what the socket holds, and the connection ends
        // without writing the rest.
        let raw = handing.client.read_to_end().await;
        assert!(raw.len() < big.len(), "the rest went to the successor");
    }

    /// A one-document response whose generation fails before it writes anything
    /// answers the failure's own status and document: the head was held for
    /// exactly this (§7.5.1 step 4).
    #[tokio::test]
    async fn a_one_body_response_states_its_failure_in_the_head_it_held() {
        let document = Bytes::from_static(
            br#"{"error":{"message":"the engine failed","type":"server_error"}}"#,
        );
        let mut client = serve_document(Some(Step::Failed {
            error: ApiError::server_error("the engine failed".to_owned()),
            frames: vec![Bytes::clone(&document)],
        }));
        client.send(&head_bytes_of("GET /document HTTP/1.1")).await;

        let response = client.read_response().await;
        assert_eq!(response.status, 500);
        assert_eq!(response.header("content-type"), "application/json");
        assert_eq!(
            response.header("content-length"),
            document.len().to_string()
        );
        assert_eq!(response.body, String::from_utf8_lossy(&document));
    }

    /// The document a one-document response produces goes out under the head
    /// the connection held for it.
    #[tokio::test]
    async fn a_one_body_response_writes_its_document_under_the_head_it_held() {
        let document = Bytes::from_static(br#"{"id":"r","object":"chat.completion"}"#);
        let mut client = serve_document(Some(Step::Frames(vec![Bytes::clone(&document)])));
        client.send(&head_bytes_of("GET /document HTTP/1.1")).await;

        let response = client.read_response().await;
        assert_eq!(response.status, 200);
        assert_eq!(response.header("content-type"), "application/json");
        assert_eq!(response.body, String::from_utf8_lossy(&document));
    }

    /// A one-document response whose body ends before producing its document is
    /// a failure, not an empty success; an empty frame is no document, so a
    /// body whose only output is one ends the same way.
    #[tokio::test]
    async fn a_one_body_response_without_a_document_states_a_server_error() {
        for step in [None, Some(Step::Frames(vec![Bytes::new()]))] {
            let mut client = serve_document(step);
            client.send(&head_bytes_of("GET /document HTTP/1.1")).await;

            let response = client.read_response().await;
            assert_eq!(response.status, 500);
            assert_eq!(response.header("content-type"), "application/json");
            assert!(
                response
                    .body
                    .contains("the response ended before it wrote its document"),
                "{}",
                response.body
            );
        }
    }

    /// A successor that takes over a response whose head its predecessor held
    /// and never wrote states the outcome itself: the bytes it inherits are a
    /// status the predecessor had not committed to (§7.5.1 step 4).
    #[tokio::test]
    async fn an_adopted_one_body_response_states_its_own_failure() {
        let document = Bytes::from_static(
            br#"{"error":{"message":"the engine failed","type":"server_error"}}"#,
        );
        let body = ResponseStream::start_held(Box::new(DocumentSession {
            step: Some(Step::Failed {
                error: ApiError::server_error("the engine failed".to_owned()),
                frames: vec![Bytes::clone(&document)],
            }),
        }));
        let (_predecessor, descriptor, mut client) = predecessor_socket();
        // The head the predecessor held, with nothing written: a provisional
        // success the successor must not commit to.
        let held = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n";
        let http = handoff::Http {
            head_held: true,
            unwritten: held.to_vec(),
            ..chunked_http(true, 0)
        };
        let adopted = tokio::spawn(adopt(descriptor, http, body, server(app()), || {}, || {}));

        let response = client.read_response().await;
        assert_eq!(response.status, 500);
        assert_eq!(response.body, String::from_utf8_lossy(&document));
        drop(client);
        let _ = adopted.await;
    }

    /// The same successor writes the head it inherited when the body produces
    /// its document: the inherited bytes are right for an outcome that is not a
    /// failure.
    #[tokio::test]
    async fn an_adopted_one_body_response_writes_the_head_its_predecessor_held() {
        let document = Bytes::from_static(br#"{"id":"r","object":"chat.completion"}"#);
        let body = ResponseStream::start_held(Box::new(DocumentSession {
            step: Some(Step::Frames(vec![Bytes::clone(&document)])),
        }));
        let (_predecessor, descriptor, mut client) = predecessor_socket();
        let held = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n";
        let http = handoff::Http {
            head_held: true,
            unwritten: held.to_vec(),
            ..chunked_http(true, 0)
        };
        let adopted = tokio::spawn(adopt(descriptor, http, body, server(app()), || {}, || {}));

        let response = client.read_response().await;
        assert_eq!(response.status, 200);
        assert_eq!(response.body, String::from_utf8_lossy(&document));
        drop(client);
        let _ = adopted.await;
    }

    /// A chat completion's body over a stand-in engine whose events arrive on
    /// `events`: server-sent events when `streamed`, one document otherwise.
    fn chat_session(
        events: mpsc::UnboundedReceiver<proto::request::Event>,
        streamed: bool,
    ) -> Box<dyn Session> {
        use crate::api::openai::chat_completions::ChatStreamEmitter;
        use crate::api::openai::run::{ChatBodySink, GenerationRun};
        use crate::api::test_support::{ChannelEvents, Unrecorded, configured_tokenizer};

        let response = proto::handoff::ChatResponseOptions {
            request_id: "chatcmpl-r".to_owned(),
            response_model: "model".to_owned(),
            ..Default::default()
        };
        let lowering = proto::handoff::Lowering {
            prompt_token_ids: vec![u32::from(b'p')],
            decode_options: Some(proto::handoff::TextDecode {
                skip_special_tokens: true,
                ..Default::default()
            }),
            parser: Some(proto::handoff::Parser::default()),
            response: Some(response.clone()),
            ..Default::default()
        };
        let output = crate::api::output::OutputProcessor::new(
            configured_tokenizer(),
            lowering,
            0,
            Box::new(Unrecorded(std::time::Instant::now())),
        )
        .expect("an output processor");

        let events = Box::new(ChannelEvents::new(events));
        if streamed {
            Box::new(GenerationRun::new(
                events,
                output,
                ChatStreamEmitter::new(&response),
            ))
        } else {
            Box::new(GenerationRun::new(
                events,
                output,
                ChatBodySink::new(&response),
            ))
        }
    }

    /// Routes that serve one request over an engine that never answers, whose
    /// events would arrive on `events`, each waiting where a request waits
    /// for its engine: `/stream` answers server-sent events, its head written
    /// at once; `/document` answers one document, its head held for it;
    /// `/handler` waits inside its handler with the request's events, as a
    /// route that generates before it responds does. `submitted` is notified
    /// once the request holds its events.
    fn silent_engine(
        events: mpsc::UnboundedReceiver<proto::request::Event>,
        submitted: Arc<Notify>,
    ) -> Router {
        let events = Arc::new(std::sync::Mutex::new(Some(events)));
        let submit = move || {
            let events = service::lock(&events)
                .take()
                .expect("one request per engine");
            submitted.notify_one();
            events
        };
        let registration = Handoff::new().register();
        let held = Arc::clone(&registration);
        let (stream, document, handler) = (submit.clone(), submit.clone(), submit);

        Router::new()
            .route(
                "/stream",
                post(move || {
                    let session = chat_session(stream(), true);
                    let registration = Arc::clone(&registration);
                    async move { handed_response(session, registration, "r") }
                }),
            )
            .route(
                "/document",
                post(move || {
                    let session = chat_session(document(), false);
                    let registration = Arc::clone(&held);
                    async move { held_response(session, registration, "r") }
                }),
            )
            .route(
                "/handler",
                post(move || {
                    let events = crate::api::test_support::ChannelEvents::new(handler());
                    async move {
                        // The handler never returns; the events go with it.
                        std::future::pending::<()>().await;
                        drop(events);
                    }
                }),
            )
    }

    /// A client that leaves while its request waits for an engine that never
    /// answers cancels the request before any output exists: while its handler
    /// runs, while a one-document response holds its head, and while a
    /// streamed response waits for its first event. Nothing is written to the
    /// client in any of them (no keep-alive comment either), so the cancel can
    /// come only from the connection noticing that its client left.
    #[tokio::test]
    async fn a_client_that_leaves_cancels_its_request_before_any_output() {
        for route in ["/handler", "/document", "/stream"] {
            let (engine, events) = mpsc::unbounded_channel();
            let submitted = Arc::new(Notify::new());
            let mut client = serve_on_pair(silent_engine(events, Arc::clone(&submitted)));
            let request = head(&format!("POST {route} HTTP/1.1"), "Content-Length: 0\r\n");
            client.send(request.as_bytes()).await;
            submitted.notified().await;
            if route == "/stream" {
                // The head goes out at once; the body then waits for the
                // engine's first event.
                let (status, _) = client.read_head().await;
                assert_eq!(status, 200, "{route}");
            }
            assert!(
                !engine.is_closed(),
                "{route}: the request is in flight before its client leaves"
            );

            drop(client);
            engine.closed().await;
        }
    }
}
