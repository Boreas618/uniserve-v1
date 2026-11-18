//! The response bodies a daemon produces itself, and freezes for a successor.
//!
//! A streamed response is not hyper's to write: the daemon owns the connection
//! (see [`crate::http::conn`]), so that a hot restart can stop a response
//! between two body frames, hand the socket and the state to the successor,
//! and go on writing elsewhere. This module is the body's side of that: a
//! [`Session`] drives one response from a request's engine events, one event
//! boundary at a time, and a [`ResponseStream`] is the handle the connection
//! writes frames from.
//!
//! The session runs in its own task (the driver below) because freezing must
//! stop it at a boundary without losing an engine event, while the connection
//! concurrently drains the frames already produced:
//!
//! - the driver polls the session for one step at a time, and the control
//!   channel before each step, so a pause stops production at the boundary
//!   just reached;
//! - a pause asks the session for its state, answers with it, and produces
//!   nothing until the connection says the successor failed (`Resume`) or the
//!   response is adopted (`Adopted`);
//! - a migration of the request's engine stream is performed outside the
//!   cancellation of the control poll, so an engine that moves the request
//!   never leaves the pipeline half-migrated;
//! - a connection that drops its handle without a decision (its client left,
//!   or it failed) ends the session at once and cancels the request, whatever
//!   the session waits for.
//!
//! The connection translates each frame a body yields into HTTP/1.1 framing,
//! so a frame always ends in the socket or in the predecessor's unwritten
//! bytes, never half-written.
//!
//! Every response the daemon produces has a state a successor continues
//! from: a chat completion streamed as server-sent events (the decoder, the
//! output processors, the assembler and the emitted-chunk counters), a chat
//! completion answered as one body, and a video artifact's byte position. A
//! response whose session reports no state at a boundary (its output
//! processing cannot be continued there) is ended by its driver instead of
//! being kept: the process that hands the deployment over does not wait for a
//! response to end (design §7.5.1), and the pause hands such a response over
//! as nothing.
//!
//! A frame is a non-empty run of body bytes: the driver drops an empty one a
//! session yields, since it carries nothing and, chunked, would be the body's
//! terminator.

use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use axum::response::Response;
use bytes::Bytes;
use tokio::sync::{mpsc, oneshot};

pub(crate) use crate::api::body::{Session, Step};
pub(crate) use proto::handoff::stream::Application;

use crate::api::openai::error::ApiError;
use crate::http::handoff::Registration;

/// The comment a streamed response writes while it waits for its engine.
const KEEP_ALIVE_COMMENT: &[u8] = b":\n\n";

/// How many body frames a session may produce ahead of the connection.
///
/// One frame is enough to keep the body fed while the connection writes and
/// flushes, and it bounds what a freeze has to drain before the socket is
/// stopped.
const AHEAD: usize = 1;

/// A message from the connection to its response's driver.
///
/// The connection has at most one message outstanding: it waits for the
/// answer of a pause before it sends the decision that follows it, and an
/// abort or an adoption is its last message. The channel therefore holds one
/// message ([`CONTROL_DEPTH`]), and every send waits for room rather than
/// assuming it.
enum Control {
    /// Stop at the boundary just reached and answer with the frozen state, or
    /// with `None` when this response cannot be handed over.
    Pause(oneshot::Sender<Option<Application>>),
    /// The successor did not take the response over: go on producing.
    Resume,
    /// The successor owns the response: stop, releasing the request's engine
    /// stream to it (never cancelling the request).
    Adopted,
    /// The connection answered the response without its body, because the
    /// deadline of its held head passed: stop and cancel the request. A
    /// connection that drops its handle ends the response the same way.
    Abort,
}

/// How many control messages a driver's channel holds: the one the
/// connection has outstanding ([`Control`]).
const CONTROL_DEPTH: usize = 1;

/// What a connection's pause of one response found.
#[expect(
    clippy::large_enum_variant,
    reason = "a `Pause` answers one handover of one response and is matched at once; boxing \
              `Frozen` would add an allocation to the only variant that carries state"
)]
pub(crate) enum Pause {
    /// The response stopped at an event boundary; this is the state a
    /// successor resumes it from.
    Frozen(Application),
    /// The response's session reported no state at its boundary, so it cannot
    /// be handed over; its driver ended it, and its connection ends too.
    Ended,
    /// The response ended at its own terminal event; there is nothing to
    /// carry, and the connection is between responses.
    Complete,
}

/// A pause's outcome with the frames the body produced before the boundary,
/// which are the response's whatever the outcome.
pub(crate) struct Paused {
    /// What the pause found.
    pub(crate) pause: Pause,
    /// The frames produced before the boundary, in order.
    pub(crate) frames: Vec<Bytes>,
}

/// What a response's driver hands its connection.
enum Produced {
    /// A body frame.
    Frame(Bytes),
    /// The response failed before its head was written: the connection states
    /// the failure in the head it has not sent yet.
    Failed {
        error: crate::api::openai::error::ApiError,
        frames: Vec<Bytes>,
    },
}

impl Produced {
    /// The body frames this is: a frame, or a failure's document read as
    /// frames of a response whose head is written.
    fn into_frames(self) -> Vec<Bytes> {
        match self {
            Self::Frame(frame) => vec![frame],
            Self::Failed { frames, .. } => frames,
        }
    }
}

/// What a response body produces first, for a connection that has not written
/// the response head yet.
pub(crate) enum Opened {
    /// The body's first frame.
    Frame(Bytes),
    /// The body failed before producing anything: the head states the outcome.
    Failed(crate::api::openai::error::ApiError, Vec<Bytes>),
    /// The body ended without producing anything.
    Ended,
}

/// The handle a connection writes one handoffable response body from.
pub(crate) struct ResponseStream {
    frames: mpsc::Receiver<Produced>,
    /// Frames taken from a failure the connection reads as body frames.
    produced: std::collections::VecDeque<Bytes>,
    /// Whether the response's head waits for the body's first output: one
    /// document, whose status is the outcome.
    head_held: bool,
    /// When a held head must be decided by, and the failure the response
    /// answers with once that passes first: the server's request timeout,
    /// which bounds the time until the head is sent ([`Self::bound_head`]).
    head_deadline: Option<(tokio::time::Instant, ApiError)>,
    control: mpsc::Sender<Control>,
    /// How long the body may wait for its engine before an SSE comment keeps
    /// the connection alive (`server_config.stream_keep_alive`); `None` for a
    /// body that is not server-sent events.
    keep_alive: Option<Duration>,
}

impl ResponseStream {
    /// Starts driving `session` and returns the handle its frames are read
    /// from.
    ///
    /// `keep_alive` writes an SSE comment when the body waits for its engine
    /// longer than that interval; it is for server-sent events only, where a
    /// comment is part of the protocol.
    pub(crate) fn start(session: Box<dyn Session>, keep_alive: Option<Duration>) -> Self {
        Self::opening(session, keep_alive, false)
    }

    /// Starts a body whose head the connection holds until the first output
    /// arrives: a response answered as one document, whose status is its
    /// outcome (§7.5.1 step 4 — such a response has not started writing when a
    /// handoff pauses it). One document writes no keep-alive comment.
    pub(crate) fn start_held(session: Box<dyn Session>) -> Self {
        Self::opening(session, None, true)
    }

    fn opening(session: Box<dyn Session>, keep_alive: Option<Duration>, head_held: bool) -> Self {
        let (frames, produced) = mpsc::channel(AHEAD);
        let (control, controls) = mpsc::channel(CONTROL_DEPTH);
        tokio::spawn(drive(session, frames, controls));
        Self {
            frames: produced,
            produced: std::collections::VecDeque::new(),
            head_held,
            head_deadline: None,
            control,
            keep_alive,
        }
    }

    /// Whether the response head waits for this body's first output.
    pub(crate) fn head_held(&self) -> bool {
        self.head_held
    }

    /// Bounds the wait for a held head: when the body has produced nothing by
    /// `deadline`, [`Self::opened`] answers `error` in its place and the
    /// request is cancelled. A head written at once has nothing to bound.
    pub(crate) fn bound_head(&mut self, deadline: tokio::time::Instant, error: ApiError) {
        if self.head_held {
            self.head_deadline = Some((deadline, error));
        }
    }

    /// What the body produces first, for a connection whose head is not
    /// written yet.
    ///
    /// Cancel safe: the head's deadline is a point in time, so a call that
    /// starts again after a cancelled one waits for the same deadline.
    pub(crate) async fn opened(&mut self) -> Opened {
        if let Some(frame) = self.produced.pop_front() {
            return Opened::Frame(frame);
        }
        let produced = match &self.head_deadline {
            // An output already produced wins over a deadline that passed
            // meanwhile: the receive is polled first.
            Some((deadline, error)) => {
                match tokio::time::timeout_at(*deadline, self.frames.recv()).await {
                    Ok(produced) => produced,
                    Err(_) => {
                        let error = error.clone();
                        return self.expired(error).await;
                    }
                }
            }
            None => self.frames.recv().await,
        };
        match produced {
            Some(Produced::Frame(frame)) => Opened::Frame(frame),
            Some(Produced::Failed { error, frames }) => Opened::Failed(error, frames),
            None => Opened::Ended,
        }
    }

    /// The held head's deadline passed before the body produced anything:
    /// the driver cancels the request, and the response answers `error` with
    /// its document.
    async fn expired(&mut self, error: ApiError) -> Opened {
        // A driver that already ended has no request left to cancel.
        let _ = self.control.send(Control::Abort).await;
        let document = failure_document(&error);
        Opened::Failed(error, vec![document])
    }

    /// The next body frame; `None` once the body ended.
    pub(crate) async fn next_frame(&mut self) -> Option<Bytes> {
        if let Some(frame) = self.produced.pop_front() {
            return Some(frame);
        }
        let frame = match self.keep_alive {
            None => self.frames.recv().await,
            Some(interval) => match tokio::time::timeout(interval, self.frames.recv()).await {
                Ok(frame) => frame,
                Err(_) => return Some(Bytes::from_static(KEEP_ALIVE_COMMENT)),
            },
        };
        match frame? {
            Produced::Frame(frame) => Some(frame),
            // A failure the connection is reading as body frames: its head is
            // already written, so the sink's document is all that is left.
            Produced::Failed { frames, .. } => {
                self.produced.extend(frames);
                self.produced.pop_front()
            }
        }
    }

    /// Freezes the response at the boundary just reached.
    ///
    /// Returns what the pause found with every frame produced before the
    /// boundary. Nothing is produced until [`Self::resume`] or
    /// [`Self::adopt`].
    pub(crate) async fn pause(&mut self) -> Paused {
        let mut frames = Vec::new();
        let pause = self.pause_into(&mut frames).await;
        Paused { pause, frames }
    }

    /// [`Self::pause`], collecting the frames into `frames`.
    async fn pause_into(&mut self, frames: &mut Vec<Bytes>) -> Pause {
        let mut keep = |produced: Produced| frames.extend(produced.into_frames());
        let (answer, paused) = oneshot::channel();
        if self.control.send(Control::Pause(answer)).await.is_err() {
            // The driver ended: the body is over, so there is nothing to
            // hand over but the frames it left.
            while let Ok(produced) = self.frames.try_recv() {
                keep(produced);
            }
            return Pause::Complete;
        }
        let mut paused = paused;
        let state = loop {
            // The driver always answers the pause before it ends the body, so
            // an answer already given outranks a body that ended: without the
            // bias both branches are ready in the same poll when the state
            // cannot be frozen and the body then ends, and the pause would
            // report `Complete` instead of `Ended` about half the time —
            // leaving the report of the registry unsent and the handoff
            // waiting for it.
            tokio::select! {
                biased;
                state = &mut paused => break match state {
                    Ok(Some(state)) => state,
                    Ok(None) => return Pause::Ended,
                    Err(_) => return Pause::Complete,
                },
                // A failure the connection reads as body frames: its head is
                // written, so those frames are the response's own.
                frame = self.frames.recv() => match frame {
                    Some(produced) => keep(produced),
                    None => return Pause::Complete,
                },
            }
        };
        // The driver produces nothing until it is told to go on, so what is
        // left in the channel is everything it produced before the boundary.
        while let Ok(produced) = self.frames.try_recv() {
            keep(produced);
        }
        Pause::Frozen(state)
    }

    /// The successor did not take the response over: go on from the pause.
    pub(crate) async fn resume(&mut self) {
        let _ = self.control.send(Control::Resume).await;
    }

    /// The successor owns the response: stop driving it and give the
    /// request's engine stream up without cancelling it, so the engine sends
    /// the successor the events after the position this process consumed.
    ///
    /// Waits for the driver: the caller is ending the body, so it cannot
    /// deadlock, and a dropped message would leave the request running here.
    pub(crate) async fn adopt(&mut self) {
        let _ = self.control.send(Control::Adopted).await;
    }
}

/// A response body a hot restart can move to a successor: the frames the
/// connection writes, the registry entry that pauses the response, the request
/// identity the pause record names, and the body's length when the route
/// knows it.
pub(crate) struct Handed {
    /// The frames the body is written from.
    pub(crate) stream: ResponseStream,
    /// The registry entry the restart pauses this response through.
    pub(crate) registration: Arc<Registration>,
    /// The request's identity as the deployment named it.
    pub(crate) request_id: String,
    /// The body's exact length in bytes, which its head states; `None` for a
    /// body of unknown length, which is chunked.
    pub(crate) length: Option<u64>,
}

/// The connection's handle on a handoffable body, carried in the response's
/// extensions.
///
/// A response that carries one has no body of its own: the session produces
/// every frame, and the connection takes the handle out before it writes the
/// response, so the body can be paused between two frames.
#[derive(Clone)]
pub(crate) struct HandedBody(Arc<std::sync::Mutex<Option<Handed>>>);

impl HandedBody {
    /// The handle of one handoffable response body of `length` bytes, or of
    /// unknown length.
    pub(crate) fn new(
        stream: ResponseStream,
        registration: Arc<Registration>,
        request_id: String,
        length: Option<u64>,
    ) -> Self {
        Self(Arc::new(std::sync::Mutex::new(Some(Handed {
            stream,
            registration,
            request_id,
            length,
        }))))
    }

    /// Takes the body out; the connection owns it from then on.
    pub(crate) fn take(&self) -> Option<Handed> {
        service::lock(&self.0).take()
    }

    /// Bounds the wait for the body's held head by `deadline`, past which
    /// the response answers `error` ([`ResponseStream::bound_head`]).
    pub(crate) fn bound_head(&self, deadline: tokio::time::Instant, error: ApiError) {
        if let Some(handed) = service::lock(&self.0).as_mut() {
            handed.stream.bound_head(deadline, error);
        }
    }
}

/// The document a failure answers with when the body produced none: the
/// error's own JSON, which is what a sink's `failed` writes for it.
pub(crate) fn failure_document(error: &ApiError) -> Bytes {
    #[expect(
        clippy::expect_used,
        reason = "an error response is a structure of strings, which always serializes"
    )]
    let document = serde_json::to_vec(&error.to_error_response()).expect("an error document");
    Bytes::from(document)
}

/// Attaches a handoffable body to `response`: the connection writes the body
/// from it, in place of the response's own.
pub(crate) fn attach(response: &mut Response, body: HandedBody) {
    response.extensions_mut().insert(body);
}

/// Drives one response body: one session step at a time, the connection's
/// messages before each step.
async fn drive(
    mut session: Box<dyn Session>,
    frames: mpsc::Sender<Produced>,
    mut control: mpsc::Receiver<Control>,
) {
    loop {
        let step = std::future::poll_fn(|cx| {
            // A message takes precedence, so a pause stops at the boundary
            // just reached rather than after the next event.
            match control.poll_recv(cx) {
                Poll::Ready(Some(message)) => return Poll::Ready(Driven::Message(message)),
                // The connection dropped its handle without a decision: its
                // client left, or it failed, and nothing writes this body any
                // more. The request ends now, as an abort ends it, rather
                // than when its next output finds no reader.
                Poll::Ready(None) => return Poll::Ready(Driven::Message(Control::Abort)),
                Poll::Pending => {}
            }
            session.poll_step(cx).map(Driven::Step)
        })
        .await;
        match step {
            Driven::Step(Step::Frames(produced)) => {
                // Sending a step's frames lasts until the connection took the last of them.
                let trace = session.trace();
                let sending = produced.iter().any(|frame| !frame.is_empty());
                if let Some(trace) = trace.filter(|_| sending) {
                    service::span::begin(service::span::Stage::Send, trace);
                }
                for frame in produced.into_iter().filter(|frame| !frame.is_empty()) {
                    if frames.send(Produced::Frame(frame)).await.is_err() {
                        // The connection is gone; cancel the request.
                        session.abort();
                        return;
                    }
                }
                if let Some(trace) = trace.filter(|_| sending) {
                    service::span::end(service::span::Stage::Send, trace);
                }
            }
            Driven::Step(Step::Migrate) => session.migrate().await,
            Driven::Step(Step::Ended) => {
                session.abort();
                return;
            }
            Driven::Step(Step::Failed {
                error,
                frames: produced,
            }) => {
                // The connection has not written the head, so it states the
                // failure there: this step is the body's last.
                let _ = frames
                    .send(Produced::Failed {
                        error,
                        frames: produced,
                    })
                    .await;
                return;
            }
            Driven::Message(Control::Pause(answer)) => {
                let frozen = session.freeze();
                let handable = frozen.is_some();
                let _ = answer.send(frozen);
                if !handable {
                    // A response whose state cannot be frozen must not keep
                    // running here: the process that hands the deployment over
                    // does not wait for a response to end (design §7.5.1). End
                    // it, and let the connection see the body end.
                    tracing::error!(
                        "a response in flight cannot be handed over; ending it rather than keeping it"
                    );
                    session.abort();
                    return;
                }
                loop {
                    match control.recv().await {
                        Some(Control::Resume) => break,
                        // Adopted: the successor consumes the request's events
                        // from here on, and the request keeps generating.
                        Some(Control::Adopted) => {
                            session.detach();
                            return;
                        }
                        // A second pause request finds the response already
                        // frozen; dropping its answer tells the caller that it
                        // cannot be frozen again.
                        Some(Control::Pause(_)) => continue,
                        // The connection went away without a decision: the
                        // client is gone, so the request ends with it. An
                        // abort ends it the same way.
                        Some(Control::Abort) | None => {
                            session.abort();
                            return;
                        }
                    }
                }
            }
            // The connection answered without the body: the request ends.
            Driven::Message(Control::Abort) => {
                session.abort();
                return;
            }
            // A message while no pause is in flight carries nothing to do.
            Driven::Message(Control::Resume | Control::Adopted) => {}
        }
    }
}

/// What a driver poll resolved to.
enum Driven {
    Step(Step),
    Message(Control),
}
