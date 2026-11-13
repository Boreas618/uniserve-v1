//! A response body produced from one request's engine events, freezable at an
//! event boundary for a successor daemon.
//!
//! The frontend produces every response body as a [`Session`]: the body's
//! side of the daemon's own HTTP connection, which polls it one engine event
//! at a time, writes the frames it yields, and pauses it at a boundary when a
//! hot restart hands the deployment to a successor. The connection, its
//! driver and the handoff registry are the daemon's; what a body produces and
//! the state it freezes to (`handoff::stream::Application`) are the frontend's.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use proto::handoff::stream::Application;

use crate::api::openai::ApiError;

/// One step of a response's driver.
pub enum Step {
    /// The frames the last engine event produced.
    Frames(Vec<Bytes>),
    /// The request's engine stream reported a migration that the driver
    /// performs before the next step.
    Migrate,
    /// The response ended; no later frame follows.
    Ended,
    /// The response failed before anything was written: the connection answers
    /// it with `error`'s status and `frames`, which are the sink's own shape of
    /// the failure. Only a response whose head is held reports this — a
    /// response whose head is written reports its failure as frames inside the
    /// stream, where the status is already stated.
    Failed {
        /// The failure, whose status the held head states.
        error: ApiError,
        /// The body the failure answers with.
        frames: Vec<Bytes>,
    },
}

/// A response body produced from one request's engine events, freezable at an
/// event boundary.
pub trait Session: Send + 'static {
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
    /// over: stop reading the request's engine events and give the stream up,
    /// so the engine's next consumer takes it over where this one stopped.
    /// Never cancels the request.
    fn detach(&mut self);
}

/// How a response's head goes out relative to its body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opening {
    /// The head waits for the body's first output: one document, whose
    /// status is its result.
    Held,
    /// The head goes out at once; `keep_alive` writes a server-sent-event
    /// comment while the body waits for its engine.
    Streamed {
        /// Whether to keep the connection alive with SSE comments.
        keep_alive: bool,
    },
}
