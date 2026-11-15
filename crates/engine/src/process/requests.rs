//! Requests and their event streams: the submission of each request to a scheduler, the
//! commands for one request, the events each admitted request produces and their journal, and
//! the warm-up requests of admission's first opening.
//!
//! A submitted request is refused at once with a `Rejected` event while the engine does not
//! serve; while admissions are paused (`Pause`, the operator's `pause_generation`) it waits,
//! with every request that arrived after it, in arrival order, and is submitted when the gate
//! opens. A request that passes is routed to a scheduler: with one scheduler to it, with several
//! to the one the route instance of the epoch's policy names (`Policy::route`); an index that
//! names no candidate is replaced by the least loaded candidate, and a fault of the route
//! instance is recovered from as a scheduler recovers from its policy's ([`routed`]). The engine
//! answers `Accepted`, naming the scheduler, and hands the request to it as
//! `Message::AddRequest`.
//!
//! Each accepted request has a [`RequestStream`]: the front its events go to and the events its
//! consumer has not reported consumed. The scheduler bounds those by the event window the front
//! stated in its welcome (`wire::Welcome::event_window`), which the consumer's `Consumed` reports
//! return as credit, so the journal holds at most the window and the terminal event. A `Takeover` moves the stream to
//! the front it arrives on and replays the events past the position it names; a position older
//! than the journal is refused by name rather than served as a stream with a hole in it.

use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::Sender;
use proto::request::wire::{self, from_engine};
use proto::request::{
    ArtifactMedia, CachePolicy, Event, FinishReason, GenerationRequest, MigrationCause,
    RejectionKind, RequestId, SamplingParams,
};
use scheduling::view::{RankView, RequestView, Snapshot};

use super::MainProcess;
use super::lifecycle::Lifecycle;
use crate::scheduler::policy::Policy;
use crate::scheduler::{Message, Request, WARMUP_REQUEST_IDS, view_of};

/// The artifact bytes an event travels with, which a replay writes again with it.
type Media = Option<Arc<dyn ArtifactMedia>>;

/// One accepted request's stream: where its events go and what a takeover replays.
pub(crate) struct RequestStream {
    /// The daemon front its events go to: the one that submitted it, until a `Takeover` moves it.
    pub(crate) front: usize,
    /// The scheduler that serves it, which its commands go to.
    pub(crate) scheduler: usize,
    /// The events written for it so far: the position a takeover names and a consumer reports.
    pub(crate) sent: u64,
    /// The events past the last position its consumer reported consumed, each at its position
    /// with the artifact bytes it travels with: positions `sent - kept.len() + 1 ..= sent`.
    pub(crate) kept: VecDeque<(u64, wire::Event, Media)>,
}

impl RequestStream {
    /// A stream of no event yet, written to `front`, of a request `scheduler` serves.
    pub(crate) fn new(front: usize, scheduler: usize) -> Self {
        Self {
            front,
            scheduler,
            sent: 0,
            kept: VecDeque::new(),
        }
    }

    /// The position the request's consumer reported it consumed.
    fn consumed(&self) -> u64 {
        self.sent - self.kept.len() as u64
    }

    /// Records the next event written for the request; returns its position.
    pub(crate) fn record(&mut self, event: wire::Event, media: Media) -> u64 {
        self.sent += 1;
        self.kept.push_back((self.sent, event, media));
        self.sent
    }

    /// The consumer took `events` of the request's events in all: the ones it took are dropped
    /// from the journal. Returns how many it took since its last report, the credit the
    /// scheduler gets back; a report of no further event returns none.
    pub(crate) fn consume(&mut self, events: u64) -> u64 {
        let taken = events.min(self.sent).saturating_sub(self.consumed());
        self.kept.drain(..taken as usize);
        taken
    }

    /// The events a takeover from position `from` replays: every event past it; or why it is
    /// refused, when the events between it and the journal's oldest were dropped.
    pub(crate) fn replay(&self, from: u64) -> Result<Vec<(wire::Event, Media)>, String> {
        let consumed = self.consumed();
        if from < consumed {
            return Err(format!(
                "the request could not be taken over from position {from}: its consumer \
                 reported position {consumed}, and this engine keeps only the {} event(s) after \
                 it",
                self.kept.len()
            ));
        }

        Ok(self
            .kept
            .iter()
            .filter(|(position, _, _)| *position > from)
            .map(|(_, event, media)| (event.clone(), media.clone()))
            .collect())
    }
}

/// The scheduler of `ranks` a request goes to when the route instance chose `index`: that rank
/// when it is a live candidate, and the least loaded live candidate otherwise (SD §5.2,
/// "Clamping"); the first rank when none lives.
pub(crate) fn candidate(index: usize, ranks: &[RankView]) -> usize {
    if ranks.get(index).is_some_and(|rank| rank.alive) {
        return index;
    }
    ranks
        .iter()
        .filter(|rank| rank.alive)
        .min_by_key(|rank| (rank.in_flight, rank.index))
        .map_or(0, |rank| rank.index)
}

/// The index the routing instance `route` chooses for `arrival` among `ranks`, or, after a fault
/// of it, the reason and whether the fault ends the engine; [`candidate`] then sends the request
/// to the least loaded rank.
///
/// A fault is recovered from by the scheduler's rule (SD §5.2 "Faults", `Policy::recover`): the
/// policy the last switch replaced, or with none a fresh instance of the serving module, starts
/// on a snapshot of `ranks` and routes from the next request on. The fault is fatal when that
/// instantiation faults, or when the faulted instance was itself so made and had not routed a
/// request since.
fn routed(
    route: &mut Policy,
    arrival: &RequestView,
    ranks: &[RankView],
) -> Result<usize, (String, bool)> {
    let fault = match route.route(arrival, ranks) {
        Ok(index) => return Ok(index),
        Err(fault) => fault,
    };
    let reason = format!("the route instance faulted: {fault}");
    if !route.recoverable() {
        return Err((reason, true));
    }

    let snapshot = Snapshot {
        ranks: ranks.to_vec(),
        ..Snapshot::default()
    };
    match route.recover(snapshot) {
        Ok(()) => Err((reason, false)),
        Err(restart) => Err((format!("{reason}; its restart faulted: {restart}"), true)),
    }
}

impl MainProcess {
    /// Takes the requests of one submission, in order: each waits at the gate while admissions
    /// are paused, and is otherwise admitted ([`Self::admit`]).
    pub(super) fn submit(&mut self, front: usize, submit: wire::Submit) {
        for request in submit.requests {
            // A request that arrives while others wait at the gate waits behind them, so the
            // gate releases in arrival order.
            if self.paused && self.lifecycle == Lifecycle::Serving {
                self.held.push_back((front, request));
            } else {
                self.admit(front, request);
            }
        }
    }

    /// Admits every request the gate held, in arrival order.
    pub(super) fn release(&mut self) {
        while !self.paused
            && let Some((front, request)) = self.held.pop_front()
        {
            self.admit(front, request);
        }
    }

    /// Ends every request the gate holds with a refusal the daemon resubmits elsewhere: the
    /// engine stopped serving before the gate opened.
    pub(super) fn refuse_held(&mut self, reason: &str) {
        for (front, request) in std::mem::take(&mut self.held) {
            self.reject(
                front,
                request.id,
                RejectionKind::Overloaded,
                reason.to_owned(),
            );
        }
    }

    /// Admits one request: refused while the engine does not serve or when it does not
    /// convert, routed, accepted, and handed to its scheduler.
    fn admit(&mut self, front: usize, request: wire::Request) {
        let id = request.id;
        if let Some(refusal) = self.lifecycle.refusal() {
            return self.reject(front, id, RejectionKind::Overloaded, refusal.to_owned());
        }
        if self.streams.contains_key(&RequestId(id)) {
            let message = format!("request {id} is in flight here already");
            return self.reject(front, id, RejectionKind::Invalid, message);
        }
        let mut input = match proto::request::Request::from_wire(request) {
            Ok(input) => input,
            Err(error) => {
                return self.reject(front, id, RejectionKind::Invalid, error.to_string());
            }
        };

        if let proto::request::Request::Generation(generation) = &mut input {
            // An adapter's tensors load into their slot, and no kernel package of this epoch
            // computes with an adapter: the base model's text would answer for another model's.
            if generation.adapter != 0 {
                let message = format!(
                    "the adapter in slot {} is loaded, and no kernel package of this epoch \
                     computes with an adapter",
                    generation.adapter
                );
                return self.reject(front, id, RejectionKind::Invalid, message);
            }
            // From a promotion with `recompute` on, a request carried from another epoch
            // computes its state again from its tokens: it reads no block under its keys.
            let carried = generation
                .cache
                .resume
                .as_ref()
                .is_some_and(|resume| resume.epoch != self.launch.epoch);
            if self.recompute && carried {
                generation.cache.read = false;
            }
        }

        let window = self.windows[front];
        let Some(scheduler) = self.route(&input, window) else {
            let message = "this process holds no scheduler: requests go to the epoch's scheduler \
                           node"
                .to_owned();
            return self.reject(front, id, RejectionKind::Invalid, message);
        };
        // A route instance whose fault could not be recovered from failed the engine.
        if let Some(refusal) = self.lifecycle.refusal() {
            return self.reject(front, id, RejectionKind::Overloaded, refusal.to_owned());
        }
        let Some(inbox) = self.inbox_of(scheduler) else {
            let message = format!("scheduler {scheduler} is gone");
            return self.reject(front, id, RejectionKind::Overloaded, message);
        };
        if inbox.send(Message::AddRequest(input, window)).is_err() {
            let message = format!("scheduler {scheduler} is gone");
            return self.reject(front, id, RejectionKind::Overloaded, message);
        }

        self.streams
            .insert(RequestId(id), RequestStream::new(front, scheduler));
        let accepted = wire::Accepted {
            request: id,
            scheduler: u32::try_from(scheduler).unwrap_or(u32::MAX),
        };
        self.send(front, from_engine::Kind::Accepted(accepted), None);
    }

    /// The scheduler that serves `input`; `None` when this process holds none (a peer).
    fn route(&mut self, input: &proto::request::Request, window: NonZeroU32) -> Option<usize> {
        let ranks = self.ranks();
        let engine = self.engine.as_mut()?;
        if ranks.len() <= 1 {
            return (!ranks.is_empty()).then_some(0);
        }
        let Some(route) = engine.route.as_mut() else {
            return Some(candidate(usize::MAX, &ranks));
        };

        // The arrival as the policy reads a request no scheduler took yet: waiting, with no
        // group and no phase ready.
        let arrival = Request::new(input.clone(), self.launch.epoch, Instant::now(), window);
        let index = match routed(route, &view_of(&arrival, &[]), &ranks) {
            Ok(index) => index,
            // This request goes to the least loaded scheduler.
            Err((reason, fatal)) => {
                self.policy_rolled_back(None, reason, fatal);
                usize::MAX
            }
        };
        Some(candidate(index, &ranks))
    }

    /// The schedulers a request may be routed to, each with the requests routed to it that are
    /// still in flight. A scheduler that stopped fails the process, so every one lives.
    pub(super) fn ranks(&self) -> Vec<RankView> {
        let count = self
            .engine
            .as_ref()
            .map_or(0, |engine| engine.schedulers.len());
        let mut in_flight = vec![0_u32; count];
        for stream in self.streams.values() {
            if let Some(held) = in_flight.get_mut(stream.scheduler) {
                *held = held.saturating_add(1);
            }
        }
        in_flight
            .into_iter()
            .enumerate()
            .map(|(index, in_flight)| RankView {
                index,
                in_flight,
                alive: true,
            })
            .collect()
    }

    /// The inbox of scheduler `scheduler`.
    fn inbox_of(&self, scheduler: usize) -> Option<Sender<Message>> {
        let engine = self.engine.as_ref()?;
        let thread = engine.schedulers.get(scheduler)?;
        Some(thread.inbox.clone())
    }

    /// One command of the request interface for one request: its consumer's position and a
    /// takeover are this process's bookkeeping, the rest go to its scheduler. A command for a
    /// request that ended is a no-op: a client that went away a moment too late is no fault.
    pub(super) fn request_command(&mut self, front: usize, command: wire::Command) {
        use wire::command::Kind;

        let id = RequestId(command.request);
        let Some(kind) = command.kind else {
            return;
        };
        let (scheduler, message) = match kind {
            Kind::Consumed(consumed) => {
                let Some(stream) = self.streams.get_mut(&id) else {
                    return;
                };
                let taken = stream.consume(consumed.events);
                if taken == 0 {
                    return;
                }
                let credit = u32::try_from(taken).unwrap_or(u32::MAX);
                (stream.scheduler, Message::Consumed(id, credit))
            }
            Kind::Takeover(takeover) => return self.take_over(front, id, takeover.events),
            Kind::Acknowledge(acknowledge) => {
                let Some(stream) = self.streams.get(&id) else {
                    return;
                };
                let tokens = u32::try_from(acknowledge.output_tokens).unwrap_or(u32::MAX);
                (stream.scheduler, Message::Acknowledge(id, tokens))
            }
            Kind::Cancel(_) => return self.finish(id, FinishReason::Cancelled),
            Kind::Stop(_) => return self.finish(id, FinishReason::Stop),
            Kind::Abort(_) => return self.finish(id, FinishReason::Aborted),
        };
        if let Some(inbox) = self.inbox_of(scheduler) {
            let _ = inbox.send(message);
        }
    }

    /// Ends request `id` for `reason`: its scheduler ends it, or, while it waits at the gate, it
    /// ends here, having computed nothing.
    fn finish(&mut self, id: RequestId, reason: FinishReason) {
        if let Some(stream) = self.streams.get(&id) {
            if let Some(inbox) = self.inbox_of(stream.scheduler) {
                let _ = inbox.send(Message::Cancel(id, reason));
            }
            return;
        }

        let Some(at) = self.held.iter().position(|(_, request)| request.id == id.0) else {
            tracing::debug!(
                request = id.0,
                "a command names a request this engine does not hold"
            );
            return;
        };
        let Some((front, _)) = self.held.remove(at) else {
            return;
        };
        let finished = Event::Finished {
            reason,
            stop_reason: None,
            prompt_tokens: 0,
            completion_tokens: 0,
        };
        let (event, _) = finished.to_wire(id);
        self.send(front, from_engine::Kind::Event(event), None);
    }

    /// A takeover of request `id`'s stream by `front` from position `from`: the stream moves to
    /// `front` and the events past `from` are written there again; a position older than what
    /// is kept is refused by name.
    fn take_over(&mut self, front: usize, id: RequestId, from: u64) {
        let Some(stream) = self.streams.get_mut(&id) else {
            // A request that ended before its takeover, or a front that named another identity:
            // only the log can say which.
            tracing::warn!(
                request = id.0,
                front,
                from,
                "a takeover names no request in flight"
            );
            return;
        };
        match stream.replay(from) {
            Ok(replay) => {
                stream.front = front;
                for (event, media) in replay {
                    self.send(front, from_engine::Kind::Event(event), media.as_ref());
                }
                tracing::info!(request = id.0, front, from, "a request was taken over");
            }
            Err(refusal) => {
                tracing::error!(request = id.0, front, from, %refusal, "a takeover was refused");
                self.reject(front, id.0, RejectionKind::Overloaded, refusal);
            }
        }
    }

    /// The events of one output of a scheduler (`Output::Events`), in order: each written to its
    /// request's stream's front and kept for a takeover; a terminal event ends its stream. They
    /// are written together, each front flushed once after the last, so the daemon reads the
    /// events of one step at once.
    pub(super) fn events(&mut self, events: Vec<(RequestId, Event)>) {
        let mut written: Vec<usize> = Vec::new();
        for (id, event) in events {
            let terminal = event.is_terminal();
            let Some(stream) = self.streams.get_mut(&id) else {
                tracing::debug!(request = id.0, "an event of a request with no stream");
                continue;
            };
            // A timeline shows when each event leaves for its daemon, with its request as the
            // payload.
            let name = if terminal {
                c"uniserve.process.terminal"
            } else {
                c"uniserve.process.event"
            };
            let _range = crate::profiling::ranges::range(name, Some(id.0));
            let (event, media) = event.to_wire(id);
            stream.record(event.clone(), media.clone());
            let front = stream.front;
            if terminal {
                self.streams.remove(&id);
            }
            self.queue(front, from_engine::Kind::Event(event), media.as_ref());
            if !written.contains(&front) {
                written.push(front);
            }
        }
        for front in written {
            self.flush_front(front);
        }
    }

    /// Ends every request this engine holds as migratable for the loss of what served it: each
    /// stream, and each request waiting at the gate. The daemon submits each again elsewhere.
    pub(super) fn lose(&mut self) {
        let lost = Event::Migratable {
            cause: MigrationCause::Loss,
            denoised_steps: 0,
            stored_layout: Vec::new(),
        };
        let held = std::mem::take(&mut self.held)
            .into_iter()
            .map(|(front, request)| (RequestId(request.id), front));
        let streams = std::mem::take(&mut self.streams)
            .into_iter()
            .map(|(id, stream)| (id, stream.front));
        for (id, front) in streams.chain(held).collect::<Vec<_>>() {
            let (event, _) = lost.to_wire(id);
            self.send(front, from_engine::Kind::Event(event), None);
        }
    }

    /// Ends one request before any scheduler took it, with the refusal that says why: the
    /// daemon learns of it no other way.
    pub(super) fn reject(&mut self, front: usize, id: u64, kind: RejectionKind, message: String) {
        let (event, _) = Event::Rejected { kind, message }.to_wire(RequestId(id));
        self.send(front, from_engine::Kind::Event(event), None);
    }

    /// The warm-up requests of admission's first opening (SD §3.2 "warm-up"): one token request
    /// per scheduler of one prompt token and two generated tokens, greedy, ignoring the end of
    /// sequence, reading and writing no cache, so every scheduler's prefill and decode path runs
    /// before a client's request does. Their ids are at or above `WARMUP_REQUEST_IDS`, whose
    /// events the scheduler drops. A deployment that launches nothing on a device (the meta
    /// device) or does not serve such a request warms nothing.
    pub(super) fn warm_up(&mut self) {
        if self.launch.engine_config.device.device != proto::config::DeviceKind::Cuda {
            return;
        }
        let Some(engine) = self.engine.as_ref() else {
            return;
        };

        let bos = self.launch.engine_config.model.bos;
        for (index, scheduler) in engine.schedulers.iter().enumerate() {
            let request = GenerationRequest {
                request_id: RequestId(WARMUP_REQUEST_IDS + index as u64),
                prompt_token_ids: vec![bos],
                multimodal_inputs: Default::default(),
                sampling: SamplingParams {
                    temperature: 0.0,
                    ignore_eos: true,
                    seed: Some(0),
                    ..SamplingParams::default()
                },
                max_und_tokens: 2,
                stop_strings: Vec::new(),
                stop_token_ids: Vec::new(),
                priority: 0,
                cache: CachePolicy {
                    read: false,
                    write: false,
                    ..CachePolicy::default()
                },
                include_stop_token: false,
                generated_token_ids: Vec::new(),
                adapter: 0,
            };
            // A deployment whose limits refuse the request a client could not submit either.
            let servable = engine.limits.features.contains(request.required_features())
                && request.validate_resources(&engine.limits).is_ok();
            if !servable {
                tracing::info!("the deployment's limits refuse the token warm-up request");
                return;
            }
            let input = proto::request::Request::Generation(Box::new(request));
            // A warm-up request's events are dropped: it runs under the launching front's
            // window, which only its scheduling reads.
            if scheduler
                .inbox
                .send(Message::AddRequest(input, self.windows[0]))
                .is_err()
            {
                tracing::warn!(scheduler = index, "a scheduler took no warm-up request");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::policy::tests::{Fixture, ROUTER_LONGEST, fixture};

    /// An event of request 9, distinct by its message.
    fn event(message: &str) -> wire::Event {
        let (event, _) = Event::Error {
            message: message.to_owned(),
        }
        .to_wire(RequestId(9));
        event
    }

    /// A stream that wrote `count` events, the `n`-th stating `n`.
    fn written(count: u64) -> RequestStream {
        let mut stream = RequestStream::new(0, 0);
        for position in 1..=count {
            assert_eq!(stream.record(event(&position.to_string()), None), position);
        }
        stream
    }

    fn messages(replay: Vec<(wire::Event, Media)>) -> Vec<String> {
        replay
            .into_iter()
            .map(|(event, _)| match event.kind {
                Some(wire::event::Kind::Error(failed)) => failed.message,
                other => panic!("an event that was not written: {other:?}"),
            })
            .collect()
    }

    #[test]
    fn a_takeover_replays_exactly_the_events_past_its_position() {
        let stream = written(5);

        assert_eq!(messages(stream.replay(2).unwrap()), ["3", "4", "5"]);
        assert_eq!(
            messages(stream.replay(0).unwrap()),
            ["1", "2", "3", "4", "5"]
        );
        assert!(stream.replay(5).unwrap().is_empty());
    }

    #[test]
    fn a_takeover_from_before_the_consumed_position_is_refused() {
        let mut stream = written(5);
        assert_eq!(stream.consume(3), 3);

        // The events up to the consumed position are kept no more: a replay from before it
        // would skip them.
        assert!(stream.replay(2).is_err());
        assert_eq!(messages(stream.replay(3).unwrap()), ["4", "5"]);
        assert_eq!(messages(stream.replay(4).unwrap()), ["5"]);
    }

    #[test]
    fn a_consumed_report_returns_the_credit_of_the_events_taken_since_the_last() {
        let mut stream = written(5);
        assert_eq!(stream.consume(2), 2);
        // A report of a position already reported takes nothing more.
        assert_eq!(stream.consume(2), 0);
        assert_eq!(stream.consume(1), 0);
        // A report past what was written takes only what was written.
        assert_eq!(stream.consume(9), 3);
        assert!(stream.kept.is_empty());

        // Events written after the report are kept again.
        stream.record(event("6"), None);
        assert_eq!(messages(stream.replay(5).unwrap()), ["6"]);
        assert!(stream.replay(4).is_err());
    }

    fn ranks(in_flight: &[u32]) -> Vec<RankView> {
        in_flight
            .iter()
            .enumerate()
            .map(|(index, &in_flight)| RankView {
                index,
                in_flight,
                alive: true,
            })
            .collect()
    }

    /// A fault of the routing instance is recovered from by the scheduler's rule (SD §5.2
    /// "Faults"): with no policy kept, a fresh instance of the serving module routes from the
    /// next request on, and the request that met the fault goes to the least loaded rank. The
    /// engine fails only when the instance a recovery made faults before it has routed a request;
    /// one that routed a request recovers again.
    #[test]
    fn a_fault_of_the_routing_instance_restarts_it_and_a_second_before_it_routes_is_fatal() {
        // A token request that just arrived, with nothing of it computed yet.
        let arrival = RequestView {
            id: scheduling::RequestId(7),
            state: scheduling::RequestStatus::Waiting,
            phase: None,
            priority: 0,
            arrived_at: scheduling::Nanos(0),
            admitted_at: None,
            prompt_tokens: 0,
            prefill_tokens: 0,
            generated_tokens: 0,
            cached_prefix_tokens: 0,
            prefilled_tokens: 0,
            cached_pages: 0,
            last_emitted_at: None,
            class: scheduling::RequestClass::Token,
            waiting_transfer: false,
            next_tokens: 0,
            reservation_units: 0,
            prefetch_tokens: 0,
            refill_tokens: None,
            remaining_tokens: 0,
            migratable: false,
            cache_rank: None,
        };
        // A load the instance routes, and one stated over so many ranks that its message is
        // longer than the instance routes, which it traps on; rank 1 is the least loaded of both.
        let calm = ranks(&[3, 1]);
        let mut loads = vec![3, 1];
        loads.resize(256, 9);
        let busy = ranks(&loads);
        let length = |load: &[RankView]| scheduling::wire::encode(&(&arrival, load)).unwrap().len();
        assert!(length(&calm) <= ROUTER_LONGEST && length(&busy) > ROUTER_LONGEST);
        let module = fixture(Fixture::Router);
        let start = || {
            let snapshot = Snapshot {
                ranks: calm.clone(),
                ..Snapshot::default()
            };
            let budget = proto::config::PolicyBudget::default();
            let ceilings = crate::scheduler::policy::ceilings([], &budget);
            Policy::new(
                Arc::clone(&module),
                Default::default(),
                snapshot,
                &ceilings,
                &budget,
            )
            .unwrap()
        };

        let mut route = start();
        assert_eq!(routed(&mut route, &arrival, &calm), Ok(0));
        let Err((reason, fatal)) = routed(&mut route, &arrival, &busy) else {
            panic!("the instance traps on the longer message");
        };
        assert!(!fatal, "the first fault restarts the instance: {reason}");
        assert_eq!(candidate(usize::MAX, &busy), 1);
        assert!(matches!(
            routed(&mut route, &arrival, &busy),
            Err((_, true))
        ));

        let mut route = start();
        assert!(matches!(
            routed(&mut route, &arrival, &busy),
            Err((_, false))
        ));
        assert_eq!(routed(&mut route, &arrival, &calm), Ok(0));
        assert!(matches!(
            routed(&mut route, &arrival, &busy),
            Err((_, false))
        ));
    }

    #[test]
    fn a_route_index_naming_a_candidate_is_taken() {
        assert_eq!(candidate(2, &ranks(&[0, 1, 5])), 2);
    }

    #[test]
    fn a_route_index_naming_no_candidate_goes_to_the_least_loaded() {
        let ranks = ranks(&[4, 1, 3, 1]);
        assert_eq!(candidate(4, &ranks), 1);
        assert_eq!(candidate(usize::MAX, &ranks), 1);

        // A rank that does not live is no candidate.
        let mut ranks = ranks;
        ranks[1].alive = false;
        assert_eq!(candidate(1, &ranks), 3);
    }
}
