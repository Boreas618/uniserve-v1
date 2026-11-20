//! Requests and their event streams: the submission of each request to a scheduler, the
//! commands for one request, the events each admitted request produces and their journal, and
//! the warm-up requests of admission's first opening.
//!
//! A submitted request is refused at once with a `Rejected` event while the engine does not
//! serve; while admissions are paused (`Pause`, the operator's `pause_generation`) it waits,
//! with every request that arrived after it, in arrival order, and is submitted when the gate
//! opens. A request that passes goes to the scheduler it names: the leader's routing program
//! chose it among the deployment's replicas, each scheduler being one, so the engine makes no
//! choice of its own. A request that names none, or names no scheduler of this engine (a
//! request resubmitted to another epoch), goes to the scheduler with the fewest requests in
//! flight ([`candidate`]). The engine answers `Accepted`, naming the scheduler, and hands the
//! request to it as `Message::AddRequest`.
//!
//! Each front numbers its own requests. The engine gives every request it admits an identity of
//! its own, which its schedulers, its caches and a takeover name, and states it in the request's
//! `Accepted`; the request's events and commands carry the identity its front gave it on that
//! front. Two fronts' requests are therefore kept apart however their daemons number them.
//!
//! Each accepted request has a [`RequestStream`]: the front its events go to, the identity they
//! carry there, and the events its consumer has not reported consumed. The scheduler bounds those
//! by the event window the front stated in its welcome (`wire::Welcome::event_window`), which the
//! consumer's `Consumed` reports return as credit, so the journal holds at most the window and
//! the terminal event. A `Takeover` names a request by the engine's identity, moves its stream to
//! the front it arrives on under the identity that front gives it, and replays the events past
//! the position it names; a position older than the journal is refused by name rather than served
//! as a stream with a hole in it. The position a takeover names is what the previous consumer
//! took, so it counts as that consumer's last `Consumed` report: the credit the scheduler holds
//! back does not depend on whether the previous front's own last report arrives before the
//! takeover or after it.

use std::collections::VecDeque;
use std::sync::Arc;

use crossbeam_channel::Sender;
use proto::MappedMedia;
use proto::request::wire::{self, from_engine};
use proto::request::{
    ArtifactMedia, CachePolicy, Event, FinishReason, GenerationRequest, MigrationCause,
    RejectionKind, RequestId, SamplingParams,
};

use super::MainProcess;
use super::lifecycle::Lifecycle;
use crate::scheduler::Message;

/// The artifact bytes an event travels with, which a replay writes again with it.
type Media = Option<Arc<dyn ArtifactMedia>>;

/// One accepted request's stream: where its events go and what a takeover replays.
pub(crate) struct RequestStream {
    /// The daemon front its events go to: the one that submitted it, until a `Takeover` moves it.
    pub(crate) front: usize,
    /// The identity that front gives the request, which its events carry there.
    pub(crate) local: RequestId,
    /// The scheduler that serves it, which its commands go to.
    pub(crate) scheduler: usize,
    /// The events written for it so far: the position a takeover names and a consumer reports.
    pub(crate) sent: u64,
    /// The events past the last position its consumer reported consumed, each at its position
    /// with the artifact bytes it travels with: positions `sent - kept.len() + 1 ..= sent`.
    pub(crate) kept: VecDeque<(u64, wire::Event, Media)>,
}

impl RequestStream {
    /// A stream of no event yet, written to `front` under the identity `local` it gives the
    /// request, of a request `scheduler` serves.
    pub(crate) fn new(front: usize, local: RequestId, scheduler: usize) -> Self {
        Self {
            front,
            local,
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

/// The scheduler a request that names `named` goes to, of schedulers holding `in_flight`
/// requests each: the named one when it is one of them, and otherwise the one with the fewest
/// requests in flight, the lowest index of those; the first when there is none.
pub(crate) fn candidate(named: Option<u32>, in_flight: &[u32]) -> usize {
    if let Some(index) = named.and_then(|named| usize::try_from(named).ok())
        && index < in_flight.len()
    {
        return index;
    }
    in_flight
        .iter()
        .enumerate()
        .min_by_key(|(index, held)| (**held, *index))
        .map_or(0, |(index, _)| index)
}

impl MainProcess {
    /// Takes the requests of one submission, in order: each waits at the gate while admissions
    /// are paused, and is otherwise admitted ([`Self::admit`]). `media` are the condition media
    /// that came with the submission, request by request in condition order: each video request
    /// takes as many as it states conditions.
    pub(super) fn submit(
        &mut self,
        front: usize,
        submit: wire::Submit,
        media: Vec<Arc<MappedMedia>>,
    ) {
        let mut media = media.into_iter();
        for request in submit.requests {
            let conditions = match &request.kind {
                Some(wire::request::Kind::Diffusion(diffusion)) => diffusion.conditions.len(),
                _ => 0,
            };
            let taken: Vec<_> = media.by_ref().take(conditions).collect();
            // A request that arrives while others wait at the gate waits behind them, so the
            // gate releases in arrival order.
            if self.paused && self.lifecycle == Lifecycle::Serving {
                self.held.push_back((front, request, taken));
            } else {
                self.admit(front, request, taken);
            }
        }
    }

    /// Admits every request the gate held, in arrival order.
    pub(super) fn release(&mut self) {
        while !self.paused
            && let Some((front, request, media)) = self.held.pop_front()
        {
            self.admit(front, request, media);
        }
    }

    /// Ends every request the gate holds with a refusal the daemon resubmits elsewhere: the
    /// engine stopped serving before the gate opened.
    pub(super) fn refuse_held(&mut self, reason: &str) {
        for (front, request, _) in std::mem::take(&mut self.held) {
            self.reject(
                front,
                request.id,
                RejectionKind::Overloaded,
                reason.to_owned(),
            );
        }
    }

    /// Admits one request with the media of its conditions: refused while the engine does not
    /// serve, when it does not convert or when its media do not match its conditions; routed,
    /// accepted, and handed to its scheduler.
    fn admit(&mut self, front: usize, request: wire::Request, media: Vec<Arc<MappedMedia>>) {
        let id = request.id;
        let named = request.scheduler;
        if let Some(refusal) = self.lifecycle.refusal() {
            return self.reject(front, id, RejectionKind::Overloaded, refusal.to_owned());
        }
        if self.named.contains_key(&(front, RequestId(id))) {
            let message = format!("request {id} is in flight here already");
            return self.reject(front, id, RejectionKind::Invalid, message);
        }
        let mut input = match proto::request::Request::from_wire(request) {
            Ok(input) => input,
            Err(error) => {
                return self.reject(front, id, RejectionKind::Invalid, error.to_string());
            }
        };
        match &mut input {
            proto::request::Request::Diffusion(diffusion)
                if diffusion.conditions.len() == media.len() =>
            {
                diffusion.media = media;
            }
            proto::request::Request::Generation(_) if media.is_empty() => {}
            _ => {
                let message = format!(
                    "request {id} came with {} media, one per condition it states",
                    media.len()
                );
                return self.reject(front, id, RejectionKind::Invalid, message);
            }
        }

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
        let loads = self.loads();
        if loads.is_empty() {
            let message = "this process holds no scheduler: requests go to the epoch's scheduler \
                           node"
                .to_owned();
            return self.reject(front, id, RejectionKind::Invalid, message);
        }
        let scheduler = candidate(named, &loads);
        let Some(inbox) = self.inbox_of(scheduler) else {
            let message = format!("scheduler {scheduler} is gone");
            return self.reject(front, id, RejectionKind::Overloaded, message);
        };
        // The engine's identity of the request, which its scheduler and caches know it by.
        let engine = RequestId(self.next_request);
        match &mut input {
            proto::request::Request::Generation(generation) => generation.request_id = engine,
            proto::request::Request::Diffusion(diffusion) => diffusion.request_id = engine,
        }
        // The request's hand-over to its scheduler, which ends as the scheduler takes it.
        let trace = input.trace();
        if trace != 0 {
            service::span::begin(service::span::Stage::Enqueue, trace);
        }
        if inbox.send(Message::AddRequest(input, window)).is_err() {
            let message = format!("scheduler {scheduler} is gone");
            return self.reject(front, id, RejectionKind::Overloaded, message);
        }
        self.next_request += 1;

        self.streams
            .insert(engine, RequestStream::new(front, RequestId(id), scheduler));
        self.named.insert((front, RequestId(id)), engine);
        let accepted = wire::Accepted {
            request: id,
            scheduler: u32::try_from(scheduler).unwrap_or(u32::MAX),
            engine_request: engine.0,
        };
        self.send(front, from_engine::Kind::Accepted(accepted), None);
    }

    /// The requests in flight on each scheduler of this process, by its index; empty when it
    /// holds none (a peer). A scheduler that stopped fails the process, so every one lives.
    pub(super) fn loads(&self) -> Vec<u32> {
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
    }

    /// The inbox of scheduler `scheduler`.
    fn inbox_of(&self, scheduler: usize) -> Option<Sender<Message>> {
        let engine = self.engine.as_ref()?;
        let thread = engine.schedulers.get(scheduler)?;
        Some(thread.inbox.clone())
    }

    /// One command of the request interface for one request, which names it by the identity
    /// `front` gives it: its consumer's position and a takeover are this process's bookkeeping,
    /// the rest go to its scheduler. A command for a request that ended is a no-op: a client that
    /// went away a moment too late is no fault.
    pub(super) fn request_command(&mut self, front: usize, command: wire::Command) {
        use wire::command::Kind;

        let local = RequestId(command.request);
        let Some(kind) = command.kind else {
            return;
        };
        if let Kind::Takeover(takeover) = kind {
            let engine = RequestId(takeover.engine_request);
            return self.take_over(front, local, engine, takeover.events);
        }
        let Some(&id) = self.named.get(&(front, local)) else {
            // A request of this front with no stream waits at the gate, or ended.
            let reason = match kind {
                Kind::Cancel(_) => FinishReason::Cancelled,
                Kind::Stop(_) => FinishReason::Stop,
                Kind::Abort(_) => FinishReason::Aborted,
                Kind::Consumed(_) | Kind::Acknowledge(_) | Kind::Takeover(_) => return,
            };
            return self.finish_held(front, local, reason);
        };
        let (scheduler, message) = match kind {
            Kind::Consumed(consumed) => {
                let Some(stream) = self.streams.get_mut(&id) else {
                    return;
                };
                let (scheduler, taken) = (stream.scheduler, stream.consume(consumed.events));
                return self.return_credit(id, scheduler, taken);
            }
            Kind::Takeover(_) => return,
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

    /// Returns `taken` events of request `id`'s credit to scheduler `scheduler`, which serves
    /// it: the events its consumer took since the position last counted. None returns nothing.
    fn return_credit(&self, id: RequestId, scheduler: usize, taken: u64) {
        if taken == 0 {
            return;
        }
        let credit = u32::try_from(taken).unwrap_or(u32::MAX);
        if let Some(inbox) = self.inbox_of(scheduler) {
            let _ = inbox.send(Message::Consumed(id, credit));
        }
    }

    /// Ends the admitted request `id` for `reason`: its scheduler ends it.
    fn finish(&mut self, id: RequestId, reason: FinishReason) {
        if let Some(stream) = self.streams.get(&id)
            && let Some(inbox) = self.inbox_of(stream.scheduler)
        {
            let _ = inbox.send(Message::Cancel(id, reason));
        }
    }

    /// Ends the request `front` names `id` for `reason` while it waits at the gate: it ends here,
    /// having computed nothing.
    fn finish_held(&mut self, front: usize, id: RequestId, reason: FinishReason) {
        let Some(at) = self
            .held
            .iter()
            .position(|(held, request, _)| *held == front && request.id == id.0)
        else {
            tracing::debug!(
                request = id.0,
                front,
                "a command names a request this engine does not hold"
            );
            return;
        };
        self.held.remove(at);
        let finished = Event::Finished {
            reason,
            stop_reason: None,
            prompt_tokens: 0,
            completion_tokens: 0,
        };
        let (event, _) = finished.to_wire(id);
        self.send(front, from_engine::Kind::Event(event), None);
    }

    /// A takeover of the stream of the request the engine knows as `id` by `front`, which gives
    /// it the identity `local`, from position `from`: the stream moves to `front` under `local`
    /// and the events past `from` are written there again; a position older than what is kept is
    /// refused by name.
    ///
    /// `from` is the position the previous consumer took the stream to, so the takeover counts
    /// it as consumed and returns the credit of the events taken past the last `Consumed` report
    /// to the scheduler. The previous consumer's own report of that position travels on its own
    /// front, which no order binds to this one: arriving first, it leaves the takeover nothing
    /// to return; arriving after, it names an identity the stream no longer carries and is
    /// dropped. Either way the scheduler gets the credit once, and the new consumer, which counts
    /// its reports from `from`, never waits on credit that only the dropped report would return.
    fn take_over(&mut self, front: usize, local: RequestId, id: RequestId, from: u64) {
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
                self.named.remove(&(stream.front, stream.local));
                stream.front = front;
                stream.local = local;
                self.named.insert((front, local), id);
                let (scheduler, taken) = (stream.scheduler, stream.consume(from));
                for (mut event, media) in replay {
                    // The journal holds the events as the previous front read them.
                    event.request = local.0;
                    self.send(front, from_engine::Kind::Event(event), media.as_ref());
                }
                self.return_credit(id, scheduler, taken);
                tracing::info!(request = id.0, front, from, "a request was taken over");
            }
            Err(refusal) => {
                tracing::error!(request = id.0, front, from, %refusal, "a takeover was refused");
                self.reject(front, local.0, RejectionKind::Overloaded, refusal);
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
            let (event, media) = event.to_wire(stream.local);
            stream.record(event.clone(), media.clone());
            let front = stream.front;
            if terminal {
                let local = stream.local;
                self.streams.remove(&id);
                self.named.remove(&(front, local));
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
            lend: None,
        };
        let held = std::mem::take(&mut self.held)
            .into_iter()
            .map(|(front, request, _)| (RequestId(request.id), front));
        self.named.clear();
        let streams = std::mem::take(&mut self.streams)
            .into_values()
            .map(|stream| (stream.local, stream.front));
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
    /// before a client's request does. Each is the scheduler's warm-up request
    /// (`Message::WarmUp`), which it numbers itself and whose events it drops. A deployment that
    /// launches nothing on a device (the meta device) or does not serve such a request warms
    /// nothing.
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
                trace: 0,
                // The scheduler gives its warm-up request an id of its own (`Scheduler::warm_up`).
                request_id: RequestId::default(),
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
                hand_off: false,
                lend: None,
            };
            // A deployment whose limits refuse the request a client could not submit either.
            let servable = engine.limits.features.contains(request.required_features())
                && request.validate_resources(&engine.limits).is_ok();
            if !servable {
                tracing::info!("the deployment's limits refuse the token warm-up request");
                return;
            }
            // A warm-up request's events are dropped: it runs under the launching front's
            // window, which only its scheduling reads.
            if scheduler
                .inbox
                .send(Message::WarmUp(Box::new(request), self.windows[0]))
                .is_err()
            {
                tracing::warn!(scheduler = index, "a scheduler took no warm-up request");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;
    use std::sync::Mutex;

    use arc_swap::ArcSwap;
    use crossbeam_channel::Receiver;
    use proto::request::wire::{command, to_engine};
    use service::Connection;

    use super::super::start::{Engine, SchedulerThread, step_budget};
    use super::super::{Input, shapes};
    use super::*;

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
        let mut stream = RequestStream::new(0, RequestId(9), 0);
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

    /// The engine's identity of the request a hot restart hands over, and the identities its
    /// launching daemon's front and its successor's give it.
    const ENGINE: RequestId = RequestId(1);
    const PREDECESSOR: RequestId = RequestId(4);
    const SUCCESSOR: RequestId = RequestId(1);

    /// The event window both daemons state in their welcome.
    const WINDOW: u32 = 64;

    /// A control thread of the fixture's launch serving over `front` and `process`, whose engine
    /// runs one scheduler; the returned receiver reads what the control thread sends that
    /// scheduler.
    fn serving(front: UnixStream, process: UnixStream) -> (MainProcess, Receiver<Message>) {
        let launch = testkit::serving::launch();
        let config = &launch.engine_config;
        let mut main = MainProcess::new(
            Connection::new(front).unwrap(),
            std::num::NonZeroU32::new(WINDOW).unwrap(),
            proto::Agreed(proto::ENGINE_INTERFACE.newest()),
            Connection::new(process).unwrap(),
            launch.clone(),
        )
        .unwrap();

        // An engine of no rank: one scheduler, whose inbox the test reads.
        let (inbox, mailbox) = crossbeam_channel::unbounded();
        let (_output, outputs) = crossbeam_channel::unbounded();
        let limits = proto::request::GenerationLimits::default();
        let ceilings = crate::scheduler::policy::ceilings([], &config.scheduler.policy_budget);
        let scheduling = proto::config::SchedulerConfig::of(
            config,
            config.scheduler.max_batch.get(),
            step_budget(config, &ceilings),
            Vec::new(),
        );
        main.engine = Some(Engine {
            schedulers: vec![SchedulerThread {
                inbox,
                outputs,
                shadow: Arc::default(),
                thread: std::thread::spawn(|| {}),
            }],
            graphs: shapes::Bounds::of(Vec::new(), ceilings, limits.clone(), scheduling).scheduling,
            limits,
            capacities: Vec::new(),
            ceilings,
            tuning: Arc::new(ArcSwap::from_pointee(proto::config::Tuning::of(
                config,
                launch.revision,
            ))),
            max_running_requests: config.max_running_requests.get(),
            served: Arc::new(crate::batch::Generations::new(1, 1)),
            session: Box::new(|| Ok(())),
            workers: Vec::new(),
            groups: None,
            binding: Arc::new(crate::process::binding::WeightBinding::new(
                Vec::new(),
                crate::process::binding::Target::default(),
                Vec::new(),
            )),
            head: None,
            build: None,
            kernels: proto::module::Packages::new(),
            remotes: Vec::new(),
            upstream: None,
            prepared: None,
            liveness: None,
            counters: Arc::new(Mutex::new(None)),
            streams: Arc::new(Vec::new()),
            video: Vec::new(),
            shares_regions: None,
        });
        (main, mailbox)
    }

    /// The frame of front `front` that reports `events` of the request it names `local`
    /// consumed.
    fn consumed(front: usize, local: RequestId, events: u64) -> Input {
        let kind = command::Kind::Consumed(wire::Consumed { events });
        frame(front, local, kind)
    }

    /// The frame of front `front` for the request it names `local`, of kind `kind`.
    fn frame(front: usize, local: RequestId, kind: command::Kind) -> Input {
        let command = wire::Command {
            request: local.0,
            kind: Some(kind),
        };
        let frame = wire::ToEngine {
            kind: Some(to_engine::Kind::Command(command)),
        };
        Input::Frame(front, Box::new(frame), Vec::new())
    }

    /// The scheduler writes `count` token events of the request.
    fn generate(main: &mut MainProcess, count: usize) {
        let token = Event::TextToken {
            id: 7,
            logprob: None,
        };
        main.events(vec![(ENGINE, token); count]);
    }

    /// The credit the control thread returned the scheduler for the request so far.
    fn credit(mailbox: &Receiver<Message>) -> u64 {
        mailbox
            .try_iter()
            .map(|message| match message {
                Message::Consumed(ENGINE, events) => u64::from(events),
                _ => 0,
            })
            .sum()
    }

    /// A hot restart hands a request's stream over at the position the predecessor's consumer
    /// took it to, 40, while the engine last counted the report of 32: the predecessor reports
    /// every 32 events. The successor counts its reports from 40, and the predecessor's report of
    /// 40 travels on its own front, which no order binds to the successor's. Whichever arrives
    /// first, and whenever the predecessor's front closes, the scheduler gets the credit of the
    /// 40 events back once, and the successor's reports return the credit of each event past
    /// them: a window of 64 events never holds the 40 the predecessor took.
    #[test]
    fn a_takeover_counts_its_position_as_consumed_whichever_front_reports_first() {
        for report_first in [false, true] {
            let (engine_front, _predecessor) = UnixStream::pair().unwrap();
            let (engine_process, _daemon_process) = UnixStream::pair().unwrap();
            let (mut main, mailbox) = serving(engine_front, engine_process);
            main.streams
                .insert(ENGINE, RequestStream::new(0, PREDECESSOR, 0));
            main.named.insert((0, PREDECESSOR), ENGINE);

            // The predecessor's consumer takes 40 events and reports 32 of them.
            generate(&mut main, 40);
            main.input(consumed(0, PREDECESSOR, 32));
            assert_eq!(credit(&mailbox), 32);

            // The successor attaches and takes the stream over at 40; the predecessor's report
            // of 40 arrives before or after its takeover; then the predecessor's front closes.
            let (engine_successor, _successor) = UnixStream::pair().unwrap();
            let successor = Connection::new(engine_successor).unwrap();
            let window = std::num::NonZeroU32::new(WINDOW).unwrap();
            main.attached("successor", Ok((successor, window)));
            let takeover = command::Kind::Takeover(wire::Takeover {
                events: 40,
                engine_request: ENGINE.0,
            });
            if report_first {
                main.input(consumed(0, PREDECESSOR, 40));
            }
            main.input(frame(1, SUCCESSOR, takeover));
            if !report_first {
                main.input(consumed(0, PREDECESSOR, 40));
            }
            main.input(Input::FrontClosed(0));
            assert_eq!(
                credit(&mailbox),
                8,
                "the predecessor reported first: {report_first}"
            );

            // The request goes on streaming to the successor, whose reports count from 40.
            generate(&mut main, 32);
            main.input(consumed(1, SUCCESSOR, 72));
            assert_eq!(
                credit(&mailbox),
                32,
                "the predecessor reported first: {report_first}"
            );
            assert_eq!(main.streams[&ENGINE].front, 1);
        }
    }

    /// A request that names a scheduler of the engine goes to it, whatever the load.
    #[test]
    fn a_request_goes_to_the_scheduler_it_names() {
        assert_eq!(candidate(Some(2), &[0, 1, 5]), 2);
        assert_eq!(candidate(Some(0), &[7]), 0);
    }

    /// A request that names no scheduler, or one the engine does not have, goes to the
    /// scheduler with the fewest requests in flight, the lowest index of equals.
    #[test]
    fn a_request_naming_no_scheduler_of_the_engine_goes_to_the_least_loaded() {
        let in_flight = [4, 1, 3, 1];
        assert_eq!(candidate(None, &in_flight), 1);
        assert_eq!(candidate(Some(4), &in_flight), 1);
        assert_eq!(candidate(Some(u32::MAX), &in_flight), 1);
        assert_eq!(candidate(None, &[]), 0);
    }
}
