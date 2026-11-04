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

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use tokio::sync::{mpsc, oneshot};

use proto::handoff;

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

    /// Ends the response: the client is gone, or the successor owns it now.
    fn abort(&mut self);
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
    /// The successor owns the response: stop, dropping the engine request.
    Adopted,
}

/// The handle a connection writes one handoffable response body from.
pub(crate) struct ResponseStream {
    frames: mpsc::Receiver<Bytes>,
    control: mpsc::Sender<Control>,
}

impl ResponseStream {
    /// Starts driving `session` and returns the handle its frames are read
    /// from.
    pub(crate) fn start(session: Box<dyn Session>) -> Self {
        let (frames, produced) = mpsc::channel(AHEAD);
        let (control, controls) = mpsc::channel(4);
        tokio::spawn(drive(session, frames, controls));
        Self {
            frames: produced,
            control,
        }
    }

    /// The next body frame; `None` once the body ended.
    pub(crate) async fn next_frame(&mut self) -> Option<Bytes> {
        self.frames.recv().await
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
    pub(crate) async fn pause(&mut self, mut keep: impl FnMut(Bytes)) -> Option<Application> {
        let (answer, paused) = oneshot::channel();
        self.control.send(Control::Pause(answer)).await.ok()?;
        let mut paused = paused;
        let state = loop {
            tokio::select! {
                state = &mut paused => break state.ok()?,
                frame = self.frames.recv() => match frame {
                    Some(frame) => keep(frame),
                    // The driver ended before it could freeze: the body is
                    // over, so there is nothing to hand over.
                    None => return None,
                },
            }
        };
        // The driver produces nothing until it is told to go on, so what is
        // left in the channel is everything it produced before the boundary.
        while let Ok(frame) = self.frames.try_recv() {
            keep(frame);
        }
        state
    }

    /// The successor did not take the response over: go on from the pause.
    pub(crate) async fn resume(&mut self) {
        let _ = self.control.send(Control::Resume).await;
    }

    /// The successor owns the response: stop driving it, cancelling the
    /// request's engine stream.
    pub(crate) fn adopt(&mut self) {
        let _ = self.control.try_send(Control::Adopted);
    }
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
                    // The response cannot be handed over, so the connection
                    // has nothing to wait for: go on serving it.
                    continue;
                }
                match control.recv().await {
                    Some(Control::Resume) => continue,
                    // Adopted, or the connection went away: this process no
                    // longer owns the response.
                    _ => {
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
