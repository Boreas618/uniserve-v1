//! One request: what the daemon submitted, what it holds and where it is.
//!
//! A [`Request`] is the same type for a token request and a video request,
//! from its arrival to its end. Where it is, is a small graph of calls
//! ([`Graph`]): a token request's loop of vision encodings, prefill chunks
//! and decode or verification calls (`generation::TokenState`), or a video
//! request's chain of media calls (`execution::VideoState`). Its
//! [`RequestStatus`] is vLLM's: waiting for admission, running, or finished
//! for a reason while its calls in flight drain.
//!
//! Every request the scheduler knows is in one table ([`Requests`]). The
//! waiting queue and the running list name requests of that table by id.

use std::collections::{BTreeSet, HashMap};

use crate::RequestId;
use crate::execution::VideoState;
use crate::generation::TokenState;
use crate::handle::EventTx;
use crate::terminal::Terminal;

/// Where a request is in its life.
///
/// `Waiting` → `Running` at admission; `Running` → `Finished` when its end is
/// decided. A finished request stays in the table until its calls in flight
/// drain, and leaves it with its terminal events. A waiting request that
/// ends (a cancellation, a rejection, a drain) leaves at once.
#[derive(Debug, Clone)]
pub(crate) enum RequestStatus {
    /// In the waiting queue: it holds no slot and no page.
    Waiting,
    /// Admitted: it holds a slot and what its calls need.
    Running,
    /// Its end is decided; it schedules no further call.
    Finished(Terminal),
}

impl RequestStatus {
    /// Whether the request's end is decided.
    pub(crate) const fn is_finished(&self) -> bool {
        matches!(self, Self::Finished(_))
    }

    /// How the request ends, once that is decided.
    pub(crate) const fn terminal(&self) -> Option<&Terminal> {
        match self {
            Self::Finished(terminal) => Some(terminal),
            Self::Waiting | Self::Running => None,
        }
    }

    /// Decides the request's end as `terminal` unless it is already decided:
    /// the first decided terminal wins.
    pub(crate) fn finish(&mut self, terminal: Terminal) {
        if !self.is_finished() {
            *self = Self::Finished(terminal);
        }
    }
}

/// The calls a request runs and where it stands among them.
pub(crate) enum Graph {
    /// A token request's loop.
    Tokens(Box<TokenState>),
    /// A video request's chain.
    Video(Box<VideoState>),
}

/// One request of the scheduler, waiting or running.
pub(crate) struct Request {
    pub(crate) status: RequestStatus,
    /// Monotonic microseconds (`clock::now_monotonic_us`) at which the
    /// request entered the waiting queue: queue waits are measured on the
    /// monotonic clock, and only the public `Scheduled` event states it as
    /// Unix time.
    pub(crate) queued_us: u64,
    pub(crate) graph: Graph,
}

impl Request {
    /// A request that just arrived, with the graph built from what the
    /// daemon submitted.
    pub(crate) fn waiting(graph: Graph) -> Self {
        Self {
            status: RequestStatus::Waiting,
            queued_us: crate::clock::now_monotonic_us(),
            graph,
        }
    }

    /// The request's id.
    pub(crate) fn id(&self) -> RequestId {
        match &self.graph {
            Graph::Tokens(state) => state.req.request_id,
            Graph::Video(state) => state.request.request_id,
        }
    }

    /// Whether the request holds a slot: it was admitted and has not left.
    pub(crate) const fn admitted(&self) -> bool {
        !matches!(self.status, RequestStatus::Waiting)
    }

    /// The token request's state, whatever its status.
    pub(crate) fn tokens(&self) -> Option<&TokenState> {
        match &self.graph {
            Graph::Tokens(state) => Some(state),
            Graph::Video(_) => None,
        }
    }

    /// The token request's state, mutably.
    pub(crate) fn tokens_mut(&mut self) -> Option<&mut TokenState> {
        match &mut self.graph {
            Graph::Tokens(state) => Some(state),
            Graph::Video(_) => None,
        }
    }

    /// The video request's state, whatever its status.
    pub(crate) fn video(&self) -> Option<&VideoState> {
        match &self.graph {
            Graph::Video(state) => Some(state),
            Graph::Tokens(_) => None,
        }
    }

    /// The video request's state, mutably.
    pub(crate) fn video_mut(&mut self) -> Option<&mut VideoState> {
        match &mut self.graph {
            Graph::Video(state) => Some(state),
            Graph::Tokens(_) => None,
        }
    }

    /// Where the request's public events go.
    pub(crate) fn event_tx(&self) -> &EventTx {
        match &self.graph {
            Graph::Tokens(state) => &state.output.events.event_tx,
            Graph::Video(state) => &state.output.event_tx,
        }
    }

    /// The prompt tokens the request's usage charges.
    pub(crate) fn prompt_tokens(&self) -> usize {
        match &self.graph {
            Graph::Tokens(state) => state.charged_prompt_tokens(),
            Graph::Video(state) => state.request.prompt_token_ids.len(),
        }
    }

    /// Whether the request continues work another engine began: a token
    /// request that carries the tokens it generated, or a video request that
    /// names the stored steps of its trajectory. Such a request is admitted
    /// ahead of every request that starts.
    pub(crate) fn resumes(&self) -> bool {
        match &self.graph {
            Graph::Tokens(state) => !state.generated_token_ids.is_empty(),
            Graph::Video(state) => state.request.trajectory.denoised_steps > 0,
        }
    }
}

/// Every request of the scheduler by id, and the requests whose policy view
/// may have moved since the policy was last told.
///
/// A policy keeps its own state from the events it is told, so the mechanism
/// tells it of every request whose view moved, and a step must not visit
/// every running request to find them. Every change of a request's view
/// starts from a mutable access to it, an arrival, an admission or one of its
/// calls entering or leaving flight; each of those marks the request here,
/// and a step computes the views of the marked requests only.
#[derive(Default)]
pub(crate) struct Requests {
    table: HashMap<RequestId, Request>,
    /// The video requests of the table, which are few.
    video: BTreeSet<RequestId>,
    /// The requests marked since the last [`Self::take_changed`].
    changed: BTreeSet<RequestId>,
}

impl Requests {
    /// The request `id`, waiting or running.
    pub(crate) fn get(&self, id: &RequestId) -> Option<&Request> {
        self.table.get(id)
    }

    /// Mutable access to the request `id`, which marks its view as moved.
    pub(crate) fn get_mut(&mut self, id: &RequestId) -> Option<&mut Request> {
        let request = self.table.get_mut(id)?;
        self.changed.insert(*id);
        Some(request)
    }

    /// The state of the admitted token request `id`.
    pub(crate) fn tokens(&self, id: &RequestId) -> Option<&TokenState> {
        self.table
            .get(id)
            .filter(|request| request.admitted())
            .and_then(Request::tokens)
    }

    /// Mutable access to the admitted token request `id`, which marks it.
    pub(crate) fn tokens_mut(&mut self, id: &RequestId) -> Option<&mut TokenState> {
        let state = self
            .table
            .get_mut(id)
            .filter(|request| request.admitted())
            .and_then(Request::tokens_mut)?;
        self.changed.insert(*id);
        Some(state)
    }

    /// The state of the admitted video request `id`.
    pub(crate) fn video(&self, id: &RequestId) -> Option<&VideoState> {
        self.table
            .get(id)
            .filter(|request| request.admitted())
            .and_then(Request::video)
    }

    /// Mutable access to the admitted video request `id`, which marks it.
    pub(crate) fn video_mut(&mut self, id: &RequestId) -> Option<&mut VideoState> {
        let state = self
            .table
            .get_mut(id)
            .filter(|request| request.admitted())
            .and_then(Request::video_mut)?;
        self.changed.insert(*id);
        Some(state)
    }

    /// The status of the request `id`.
    pub(crate) fn status(&self, id: &RequestId) -> Option<&RequestStatus> {
        self.table.get(id).map(|request| &request.status)
    }

    /// Whether the end of the request `id` is decided.
    pub(crate) fn finished(&self, id: &RequestId) -> bool {
        self.status(id).is_some_and(RequestStatus::is_finished)
    }

    /// Adds a request that arrived.
    pub(crate) fn insert(&mut self, request: Request) {
        let id = request.id();
        self.changed.insert(id);
        if request.video().is_some() {
            self.video.insert(id);
        }
        self.table.insert(id, request);
    }

    /// Takes the request `id` out of the table: it ended.
    pub(crate) fn remove(&mut self, id: &RequestId) -> Option<Request> {
        self.changed.insert(*id);
        self.video.remove(id);
        self.table.remove(id)
    }

    /// The admitted token requests, in no order. A step does not call this:
    /// it serves the operations that concern every request at once (a
    /// resize, a sleep, a drain, a failure).
    pub(crate) fn running_tokens(
        &self,
    ) -> impl Iterator<Item = (RequestId, &RequestStatus, &TokenState)> {
        self.table.iter().filter_map(|(id, request)| {
            let state = request.tokens().filter(|_| request.admitted())?;
            Some((*id, &request.status, state))
        })
    }

    /// The admitted video requests, in id order. Video requests are few, and
    /// a step reads them: a video request's next ready call also depends on
    /// the calls other requests hold in flight on a shared stage.
    pub(crate) fn running_video(
        &self,
    ) -> impl Iterator<Item = (RequestId, &RequestStatus, &VideoState)> {
        self.video.iter().filter_map(|id| {
            let request = self.table.get(id)?;
            let state = request.video().filter(|_| request.admitted())?;
            Some((*id, &request.status, state))
        })
    }

    /// Mutable access to every admitted request, marking none: the caller
    /// marks the ones whose view it moves. See [`Self::running_tokens`].
    pub(crate) fn admitted_mut(&mut self) -> impl Iterator<Item = (RequestId, &mut Request)> {
        self.table
            .iter_mut()
            .filter(|(_, request)| request.admitted())
            .map(|(id, request)| (*id, request))
    }

    /// Mutable access to the status of the admitted request `id`, which
    /// marks it.
    pub(crate) fn status_mut(&mut self, id: &RequestId) -> Option<&mut RequestStatus> {
        let request = self
            .table
            .get_mut(id)
            .filter(|request| request.admitted())?;
        self.changed.insert(*id);
        Some(&mut request.status)
    }

    /// Marks a request whose view moved without a mutable access to it: a
    /// queue change, or one of its calls entering or leaving flight.
    pub(crate) fn mark(&mut self, id: RequestId) {
        self.changed.insert(id);
    }

    /// The requests marked since the last call, in id order.
    pub(crate) fn take_changed(&mut self) -> BTreeSet<RequestId> {
        std::mem::take(&mut self.changed)
    }
}
