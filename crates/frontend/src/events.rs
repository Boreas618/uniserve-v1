//! The engine's side of one request, as output processing reads it: the
//! request's engine events ([`Events`]) and the recorder of its statistics
//! ([`Stats`]).
//!
//! The daemon's engine client implements both: its events follow the request
//! across the engines that serve it when it migrates, and its recorder
//! reports the request's metrics once. Output processing consumes them
//! without knowing either.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use proto::request::{Event, RequestId};

use crate::output::RequestOutput;

/// What one step of a request's events resolved to.
pub enum Next {
    /// The next event.
    Event(Event),
    /// The request's engine is handing the request over: the consumer
    /// migrates ([`Events::migrate_pending`]) before the next step.
    Migrate,
    /// The request ended.
    Ended,
}

/// One submitted request's engine events.
///
/// Dropping the events cancels the request in the engine that serves it,
/// unless they were handed over ([`Events::hand_over`]).
pub trait Events: Send {
    /// Advances the request's events one step, registering `cx`'s waker.
    ///
    /// A response driven from a poll loop (a handoffable body) needs one step
    /// at a time and must not hold a future across a pause.
    fn poll_step(&mut self, cx: &mut Context<'_>) -> Poll<Next>;

    /// Receives the next event, performing any migration on the way; `None`
    /// once the request ended.
    fn recv(&mut self) -> Pin<Box<dyn Future<Output = Option<Event>> + Send + '_>>;

    /// Carries out the migration the last [`Events::poll_step`] reported,
    /// after which the events go on from the engine that serves the request
    /// now.
    ///
    /// Resolves to `Err` with the terminal event the consumer receives
    /// instead when the migration fails.
    fn migrate_pending(&mut self) -> Pin<Box<dyn Future<Output = Result<(), Event>> + Send + '_>>;

    /// Accepts every text token received so far as completing no stop string.
    fn acknowledge_consumed_prefix(&mut self);

    /// Ends generation after the tokens received, because they completed a
    /// stop string.
    fn stop_at_consumed_prefix(&mut self);

    /// Cancels the request.
    fn cancel(&mut self);

    /// Ends this process's consumption of the events without cancelling the
    /// request: the engine's next consumer takes the stream over.
    fn hand_over(&mut self);

    /// The request's identity in the engine that serves it, which a
    /// successor takes its event stream over by; `None` once the request
    /// ended or was handed over.
    fn engine_request_id(&self) -> Option<RequestId>;

    /// The engine events consumed in all, and the public tokens they carried:
    /// the position a successor takes the stream over from.
    fn progress(&self) -> (u64, u64);

    /// The request as submitted, encoded as JSON for a handoff's record.
    fn submitted(&self) -> Option<Vec<u8>>;
}

/// The recorder of one request's statistics, folded from its public outputs.
pub trait Stats: Send {
    /// When the request entered the server.
    fn started(&self) -> Instant;

    /// Which engine replica serves the request.
    fn engine(&self) -> u32;

    /// Folds one public output into the request's statistics, recording them
    /// at its terminal output.
    fn observe(&mut self, output: &RequestOutput);

    /// Counts a request the client abandoned or a successor took over.
    fn abandoned(&mut self);
}
