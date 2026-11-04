//! The response bodies a daemon produces itself, and freezes for a successor
//! (design `specs/worker.md` §7.5.1).
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
//!   never leaves the pipeline half-migrated.
//!
//! The connection translates each frame a body yields into HTTP/1.1 framing,
//! so a frame always ends in the socket or in the predecessor's unwritten
//! bytes, never half-written.
//!
//! Every kind item 1 covers has a state a successor continues from: a chat
//! completion streamed as server-sent events (the decoder, the output
//! processors, the assembler and the emitted-chunk counters), a chat
//! completion answered as one body, and a video artifact's byte position. A
//! response of such a kind whose state cannot be frozen is a defect, and the
//! driver ends it instead of keeping it: the process that hands the
//! deployment over does not wait for a response to end (design §7.5.1). Only
//! the families the design retires (the SenseNova and omni readout output
//! filters, whose requests have no facts a continuation submits) have no
//! state, and item 2 removes their routes from the covered set.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use std::sync::Arc;

use axum::response::Response;
use bytes::Bytes;
use tokio::sync::{mpsc, oneshot};

use proto::handoff;

use crate::http::handoff::Registration;

/// How long a streamed response may wait for its next frame before the
/// connection writes an SSE comment to keep the connection alive, as the
/// `Sse` keep-alive the daemon served before did.
const KEEP_ALIVE: std::time::Duration = std::time::Duration::from_secs(15);

/// The comment a streamed response writes while it waits for its engine.
const KEEP_ALIVE_COMMENT: &[u8] = b":\n\n";

/// How many body frames a session may produce ahead of the connection.
///
/// One frame is enough to keep the body fed while the connection writes and
/// flushes, and it bounds what a freeze has to drain before the socket is
/// stopped.
const AHEAD: usize = 1;

/// One step of a response's driver.
pub(crate) enum Step {
    /// The frames the last engine event produced.
    Frames(Vec<Bytes>),
    /// The request's engine stream reported a migration that the driver
    /// performs before the next step.
    Migrate,
    /// The response ended; no later frame follows.
    Ended,
}

/// A response body produced from one request's engine events, freezable at an
/// event boundary.
pub(crate) trait Session: Send + 'static {
    /// Advances to the next boundary.
    ///
    /// Returns the frames one engine event produced, that the request
    /// migrated, or that the response ended. Returns `Pending` while it waits
    /// for the engine, having registered `cx`'s waker.
    fn poll_step(&mut self, cx: &mut Context<'_>) -> Poll<Step>;

    /// Performs the migration `poll_step` reported.
    ///
    /// The driver never cancels this future: a paused handoff waits until the
    /// request's engine stream is whole again.
    fn migrate(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;

    /// Freezes at the boundary just reached and reports the state a successor
    /// continues from; `None` when this response cannot be handed over, for
    /// example because its request cannot migrate.
    fn freeze(&mut self) -> Option<Application>;

    /// Ends the response because the client is gone or it failed: cancel the
    /// request on the engine.
    fn abort(&mut self);

    /// Ends this process's part in the response because a successor took it
    /// over (design §7.5.1): stop reading the request's engine events and
    /// give the stream up, so the engine's next consumer takes it over where
    /// this one stopped. Never cancels the request.
    fn detach(&mut self);
}

/// The application state of one frozen response, as the `application` of
/// `handoff.Stream`.
pub(crate) enum Application {
    /// A chat completion streamed as server-sent events.
    ChatStream(handoff::ChatStream),
    /// A chat completion answered as one JSON body.
    ChatResponse(handoff::ChatResponse),
    /// A video artifact streamed from a job's retained media.
    VideoContent(handoff::VideoContent),
}

impl Application {
    /// The application state as the `application` of `handoff.Stream`.
    pub(crate) fn into_stream(self, http: handoff::Http, request_id: String) -> handoff::Stream {
        let application = match self {
            Self::ChatStream(chat) => handoff::stream::Application::ChatStream(chat),
            Self::ChatResponse(chat) => handoff::stream::Application::ChatResponse(chat),
            Self::VideoContent(video) => handoff::stream::Application::VideoContent(video),
        };
        handoff::Stream {
            http: Some(http),
            request_id,
            application: Some(application),
        }
    }
}

/// A message from the connection to its response's driver.
enum Control {
    /// Stop at the boundary just reached and answer with the frozen state, or
    /// with `None` when this response cannot be handed over.
    Pause(oneshot::Sender<Option<Application>>),
    /// The successor did not take the response over: go on producing.
    Resume,
    /// The successor owns the response: stop, releasing the request's engine
    /// stream to it (never cancelling the request).
    Adopted,
}

/// What a connection's pause of one response found.
pub(crate) enum Pause {
    /// The response stopped at an event boundary; this is the state a
    /// successor resumes it from.
    Frozen(Application),
    /// The response cannot be handed over. No response kind item 1 covers has
    /// this path (design §7.5.1): a covered response whose state cannot be
    /// frozen ends here rather than being kept by the process that is
    /// leaving.
    Ended,
    /// The response ended at its own terminal event; there is nothing to
    /// carry, and the connection is between responses.
    Complete,
}

/// The handle a connection writes one handoffable response body from.
pub(crate) struct ResponseStream {
    frames: mpsc::Receiver<Bytes>,
    control: mpsc::Sender<Control>,
    /// Whether to write an SSE comment while the body waits for its engine.
    keep_alive: bool,
}

impl ResponseStream {
    /// Starts driving `session` and returns the handle its frames are read
    /// from.
    ///
    /// `keep_alive` writes an SSE comment when the body waits for its engine
    /// longer than the keep-alive interval; it is for server-sent events only,
    /// where a comment is part of the protocol.
    pub(crate) fn start(session: Box<dyn Session>, keep_alive: bool) -> Self {
        let (frames, produced) = mpsc::channel(AHEAD);
        let (control, controls) = mpsc::channel(4);
        tokio::spawn(drive(session, frames, controls));
        Self {
            frames: produced,
            control,
            keep_alive,
        }
    }

    /// The next body frame; `None` once the body ended.
    pub(crate) async fn next_frame(&mut self) -> Option<Bytes> {
        if !self.keep_alive {
            return self.frames.recv().await;
        }
        match tokio::time::timeout(KEEP_ALIVE, self.frames.recv()).await {
            Ok(frame) => frame,
            Err(_) => Some(Bytes::from_static(KEEP_ALIVE_COMMENT)),
        }
    }

    /// Takes a frame that is ready without waiting.
    pub(crate) fn try_frame(&mut self) -> Option<Bytes> {
        self.frames.try_recv().ok()
    }

    /// Freezes the response at the boundary just reached.
    ///
    /// Every frame produced before the boundary is given to `keep`; the state
    /// a successor continues from is returned, or `None` when this response
    /// cannot be handed over or its driver ended meanwhile. Nothing is
    /// produced until [`Self::resume`] or [`Self::adopt`].
    pub(crate) async fn pause(&mut self, mut keep: impl FnMut(Bytes)) -> Pause {
        let (answer, paused) = oneshot::channel();
        if self.control.send(Control::Pause(answer)).await.is_err() {
            // The driver ended: the body is over, so there is nothing to
            // hand over.
            return Pause::Complete;
        }
        let mut paused = paused;
        let state = loop {
            tokio::select! {
                state = &mut paused => break match state {
                    Ok(Some(state)) => state,
                    Ok(None) => return Pause::Ended,
                    Err(_) => return Pause::Complete,
                },
                frame = self.frames.recv() => match frame {
                    Some(frame) => keep(frame),
                    None => return Pause::Complete,
                },
            }
        };
        // The driver produces nothing until it is told to go on, so what is
        // left in the channel is everything it produced before the boundary.
        while let Ok(frame) = self.frames.try_recv() {
            keep(frame);
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
/// connection writes, the registry entry that pauses the response, and the
/// request identity the pause record names.
pub(crate) struct Handed {
    /// The frames the body is written from.
    pub(crate) stream: ResponseStream,
    /// The registry entry the restart pauses this response through.
    pub(crate) registration: Arc<Registration>,
    /// The request's identity as the deployment named it.
    pub(crate) request_id: String,
}

/// The connection's handle on a handoffable body, carried in the response's
/// extensions.
///
/// A response that carries one has no body of its own: the session produces
/// every frame, and the connection takes the handle out before it writes the
/// response, so the body can be paused between two frames.
#[derive(Clone, Default)]
pub(crate) struct HandedBody(Arc<std::sync::Mutex<Option<Handed>>>);

impl HandedBody {
    /// The handle of one handoffable response body.
    pub(crate) fn new(
        stream: ResponseStream,
        registration: Option<Arc<Registration>>,
        request_id: String,
    ) -> Option<Self> {
        registration.map(|registration| {
            Self(Arc::new(std::sync::Mutex::new(Some(Handed {
                stream,
                registration,
                request_id,
            }))))
        })
    }

    /// Takes the body out; the connection owns it from then on.
    pub(crate) fn take(&self) -> Option<Handed> {
        crate::leader::locked(&self.0).take()
    }
}

/// Attaches a handoffable body to `response`.
///
/// `registration` is `None` when no hot restart can pause this daemon (a node
/// other than the leader), in which case the response is written without the
/// handoff registry entry.
pub(crate) fn attach(response: &mut Response, body: HandedBody) {
    response.extensions_mut().insert(body);
}

/// Drives one response body: one session step at a time, the connection's
/// messages before each step.
async fn drive(
    mut session: Box<dyn Session>,
    frames: mpsc::Sender<Bytes>,
    mut control: mpsc::Receiver<Control>,
) {
    loop {
        let step = std::future::poll_fn(|cx| {
            // A message takes precedence, so a pause stops at the boundary
            // just reached rather than after the next event.
            if let Poll::Ready(Some(message)) = control.poll_recv(cx) {
                return Poll::Ready(Driven::Message(message));
            }
            session.poll_step(cx).map(Driven::Step)
        })
        .await;
        match step {
            Driven::Step(Step::Frames(produced)) => {
                for frame in produced {
                    if frames.send(frame).await.is_err() {
                        // The connection is gone; cancel the request.
                        session.abort();
                        return;
                    }
                }
            }
            Driven::Step(Step::Migrate) => session.migrate().await,
            Driven::Step(Step::Ended) => {
                session.abort();
                return;
            }
            Driven::Message(Control::Pause(answer)) => {
                let frozen = session.freeze();
                let handable = frozen.is_some();
                let _ = answer.send(frozen);
                if !handable {
                    // A covered response whose state cannot be frozen must not
                    // keep running here: the process that hands the deployment
                    // over does not wait for a response to end (design
                    // §7.5.1). End it, and let the connection see the body end.
                    tracing::error!(
                        "a response in flight cannot be handed over; ending it rather than keeping it"
                    );
                    session.abort();
                    return;
                }
                match control.recv().await {
                    Some(Control::Resume) => continue,
                    // Adopted: the successor consumes the request's events
                    // from here on, and the request keeps generating.
                    Some(Control::Adopted) => {
                        session.detach();
                        return;
                    }
                    // The connection went away without a decision: the client
                    // is gone, so the request ends with it.
                    None => {
                        session.abort();
                        return;
                    }
                }
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
