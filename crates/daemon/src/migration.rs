//! Request migration: how a token or video request moves between engines.
//!
//! An engine that parks for an epoch change or a rollback, sleeps, rewrites
//! its weights in place, shrinks its KV cache, or loses the workers that
//! hold a request, ends the request as migratable (`Event::Migratable`): a
//! token request after every token it generated there, a video request at a
//! denoising step boundary, with the steps of the trajectory the engine
//! stored. The leader submits every request through its [`Migrator`], which
//! keeps the facts a continuation needs and, on that event, submits the
//! request again with them through the hold to the engine that serves next:
//!
//! - a token request's generated tokens, every one its engine counted, in
//!   order (`GenerationRequest::generated_token_ids`);
//! - the stored steps of a video request's trajectory and the engine's
//!   record of where they lie (`Trajectory::denoised_steps`,
//!   `Trajectory::stored_layout`).
//!
//! These are facts: the client received every token, and the engine that
//! continues the request computes everything else from them, its context
//! included. The request's consumer reads one stream throughout
//! ([`RequestEvents`]): the new engine continues after the last token it
//! received and charges the request's whole usage, or continues the
//! trajectory from the stored step, and the daemon's detokenizer,
//! stop-string matcher and response assembly keep their state.
//!
//! A migration caused by a drain is not counted. A request survives
//! `max_migrations` migrations caused by losses and fails on the next one.
//!
//! The random stream of a migrated request continues as it began: the
//! migrator gives a token request that names no seed one of its own when it
//! is first submitted, and every draw of a seeded request follows its seed
//! and its token position alone; a latent encoding's posterior draw follows
//! the encoding's place among the request's encodings, which the request's
//! facts fix. A video request's draws follow its seed. The migrator gives a
//! video request the identity that keys its stored trajectories
//! (`Trajectory::id`) when it is first submitted.

use std::sync::Arc;
use std::task::{Context, Poll, ready};
use std::time::Instant;

use proto::request::{
    DiffusionRequest, Event, GenerationRequest, MigrationCause, Request, RequestId, Trajectory,
};
#[cfg(test)]
use tokio::sync::mpsc;

use crate::engine::client::{EventStream, StreamCancelCause, SubmitError};
use crate::engine::hold::{Kept, RequestHold};

/// How long a migration waits for an engine to serve the request it carries.
///
/// The wait itself has no other bound — a deployment between engines is the
/// ordinary case a successor daemon covers — so this is the point at which a
/// request that no engine takes is answered instead of staying open.
const MIGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Submits the leader's requests and moves its token requests between
/// engines.
pub struct Migrator {
    hold: Arc<RequestHold>,
    /// Migrations caused by losses a request survives.
    max_migrations: u32,
}

impl Migrator {
    /// A migrator that submits through `hold`, under which a request survives
    /// `max_migrations` migrations caused by losses.
    pub fn new(hold: Arc<RequestHold>, max_migrations: u32) -> Self {
        Self {
            hold,
            max_migrations,
        }
    }

    /// Submits `request` through the hold and returns its events and the
    /// scheduler serving it, as `RequestHold::submit` does. A token request
    /// that resumes from its tokens, and a video request, are kept for a
    /// later migration; a token request that names no seed takes one first,
    /// so that its draws repeat on the engine that serves it next, and a
    /// video request takes its trajectories' identity.
    ///
    /// # Errors
    ///
    /// Fails as `RequestHold::submit` fails.
    pub async fn submit(
        &self,
        mut request: Request,
    ) -> Result<(RequestEvents, Option<u32>), SubmitError> {
        let resumable = match &mut request {
            Request::Generation(generation) => {
                if generation.sampling.seed.is_none() {
                    generation.sampling.seed = Some(uuid::Uuid::new_v4().as_u64_pair().0);
                }
                Some(Resumable::Generation(generation.clone()))
            }
            Request::Diffusion(diffusion) => {
                if diffusion.trajectory.id == 0 {
                    diffusion.trajectory.id = uuid::Uuid::new_v4().as_u64_pair().0;
                }
                Some(Resumable::Diffusion(diffusion.clone()))
            }
        };
        let (stream, scheduler) = self.hold.submit(request).await?;
        let events = RequestEvents {
            stream,
            request: resumable,
            hold: Some(Arc::clone(&self.hold)),
            max_migrations: self.max_migrations,
            losses: 0,
            pending: None,
            resumed: false,
            scheduled: false,
            migrated_at: None,
            open: true,
            on_finish: None,
        };
        Ok((events, scheduler))
    }
}

/// What one step of a request's events resolved to; an event's tokens are
/// already kept as the request's facts.
pub(crate) use crate::api::events::Next as Step;

/// A request a migration submits again, as last submitted.
#[derive(Clone)]
enum Resumable {
    /// A token request, with the tokens it generated before.
    Generation(Box<GenerationRequest>),
    /// A video request, naming the trajectory steps an engine stored.
    Diffusion(DiffusionRequest),
}

impl Resumable {
    /// The request `migration` submits again: a token request with the
    /// tokens it generated, a video request naming the trajectory steps its
    /// engine stored and its record of where they lie
    /// (`Migration::denoised_steps`, zero for none). `kept` is the state an epoch change keeps of the
    /// requests in flight (`in_flight = keep`), which the request continues
    /// with when the epoch that serves next computes other numerics.
    fn continuation(&self, migration: &Migration, kept: Option<Kept>) -> Request {
        let Migration {
            cause,
            denoised_steps,
            stored_layout,
        } = migration;
        let (cause, denoised_steps) = (*cause, *denoised_steps);
        match self {
            Self::Generation(request) => {
                let mut continuation = request.as_ref().clone();
                // Its prompt's log-probabilities were reported where it began.
                continuation.sampling.return_prompt_logprobs = false;
                continuation.sampling.n_prompt_logprobs = 0;
                // It reads the state it reached and publishes later pages
                // only to its private continuation: they are neither epoch's.
                if let Some(seeds) = kept.as_ref().and_then(|kept| kept.prefix_seeds.as_ref()) {
                    continuation.cache.state_seeds = seeds.clone();
                    continuation.cache.write = false;
                }
                tracing::info!(
                    ?cause,
                    generated = request.generated_token_ids.len(),
                    "a request migrated to the engine that serves next"
                );
                Request::Generation(Box::new(continuation))
            }
            Self::Diffusion(request) => {
                let mut continuation = request.clone();
                stored(
                    &mut continuation.trajectory,
                    denoised_steps,
                    stored_layout,
                    kept,
                );
                tracing::info!(
                    ?cause,
                    denoised_steps = continuation.trajectory.denoised_steps,
                    stored = denoised_steps > 0,
                    "a video request migrated to the engine that serves next"
                );
                Request::Diffusion(continuation)
            }
        }
    }
}

/// Names in `trajectory` the `denoised_steps` of it an engine stored at a
/// step boundary, keyed by what that engine computes, and its record of
/// where they lie (`stored_layout`): the next engine reads them when it
/// computes the same, or when an epoch change keeps the state the request
/// reached (`kept`), whatever its parallel configuration, from where the
/// record says the pages lie. A request the engine stored nothing of names
/// the trajectory it named before, which is the same at that step whenever
/// it is read.
fn stored(
    trajectory: &mut Trajectory,
    denoised_steps: u32,
    stored_layout: &[u8],
    kept: Option<Kept>,
) {
    if denoised_steps > 0 {
        trajectory.denoised_steps = denoised_steps;
        trajectory.state_seed = kept.and_then(|kept| kept.latent_seed);
        trajectory.stored_layout = stored_layout.to_vec();
    }
}

/// Why a request left its engine, and what the engine stored of it
/// (`Event::Migratable`).
struct Migration {
    cause: MigrationCause,
    /// The trajectory steps a video request's engine stored; zero for none.
    denoised_steps: u32,
    /// Where the engine put the stored trajectory's pages and how each
    /// stored rank laid its part out (`Trajectory::stored_layout`).
    stored_layout: Vec<u8>,
}

/// One request's events across the engines that serve it.
///
/// It offers what an `EventStream` offers. An `Event::Migratable` never
/// reaches its consumer: the stream submits the request again and goes on
/// with the next engine's events, or ends with an `Event::Error` when the
/// request does not migrate. Dropping it before its terminal event cancels
/// the request on the engine that serves it.
pub struct RequestEvents {
    /// The events of the engine that serves the request now.
    stream: EventStream,
    /// The request as last submitted, with the tokens it generated before
    /// or the trajectory steps an engine stored; `None` for a request that
    /// does not resume.
    request: Option<Resumable>,
    /// The hold the request migrates through; `None` for a stream no
    /// engine feeds.
    hold: Option<Arc<RequestHold>>,
    max_migrations: u32,
    /// Migrations caused by losses so far.
    losses: u32,
    /// A migration `try_recv` met, which the next `recv` carries out.
    pending: Option<Migration>,
    /// Whether the stream resumed on another engine, whose scheduling event
    /// repeats one the consumer received. A migration that happened *before*
    /// the consumer received any scheduling event leaves this false: the
    /// engine that serves next states the request's scheduling for the first
    /// time, and skipping it would leave the consumer with a token and no
    /// output to put it in.
    resumed: bool,
    /// Whether the consumer received the request's scheduling event. What a
    /// migration skips is decided from this, because it is a fact about what
    /// the consumer read, not about which engine sent it.
    scheduled: bool,
    /// When the request's last migration began, until the engine that serves
    /// it next delivers its first token.
    migrated_at: Option<Instant>,
    /// Whether the request may still be cancelled.
    open: bool,
    on_finish: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl RequestEvents {
    /// The events `events` delivers, which no engine feeds.
    #[cfg(test)]
    pub(crate) fn detached(events: mpsc::UnboundedReceiver<Event>) -> Self {
        Self {
            stream: EventStream::detached(events),
            request: None,
            hold: None,
            max_migrations: 0,
            losses: 0,
            pending: None,
            resumed: false,
            scheduled: false,
            migrated_at: None,
            open: true,
            on_finish: None,
        }
    }

    /// The events of a request this process takes over from another: the
    /// engine's stream `stream`, which goes on after the position the
    /// previous consumer reached (design §7.5.1), and no hold, because the
    /// request is already admitted and migrated by the engine itself.
    pub(crate) fn taken_over(stream: EventStream, consumed: u64) -> Self {
        Self {
            stream,
            request: None,
            hold: None,
            max_migrations: 0,
            losses: 0,
            pending: None,
            resumed: consumed > 0,
            scheduled: consumed > 0,
            migrated_at: None,
            open: true,
            on_finish: None,
        }
    }

    /// Receives the next event; `None` once the request ended.
    pub async fn recv(&mut self) -> Option<Event> {
        loop {
            if let Some(migration) = self.pending.take() {
                match self.migrate(migration).await {
                    Ok(()) => continue,
                    Err(event) => return self.deliver(Some(event)),
                }
            }
            let event = self.stream.recv().await;
            match event {
                Some(Event::Migratable {
                    cause,
                    denoised_steps,
                    stored_layout,
                }) => {
                    self.pending = Some(Migration {
                        cause,
                        denoised_steps,
                        stored_layout,
                    })
                }
                Some(event) if self.skips(&event) => {}
                event => return self.deliver(event),
            }
        }
    }

    /// Receives the next event; the same as [`Self::recv`].
    pub async fn next(&mut self) -> Option<Event> {
        self.recv().await
    }

    /// Advances the request's events one step, registering `cx`'s waker.
    ///
    /// The same stream [`Self::recv`] offers, without awaiting it: a response
    /// driven from a poll loop (a handoffable stream, design §7.5.1) needs one
    /// step at a time and must not hold a future across the pause.
    pub(crate) fn poll_step(&mut self, cx: &mut Context<'_>) -> Poll<Step> {
        loop {
            if self.pending.is_some() {
                return Poll::Ready(Step::Migrate);
            }
            let event = ready!(self.stream.poll_recv(cx));
            match event {
                Some(Event::Migratable {
                    cause,
                    denoised_steps,
                    stored_layout,
                }) => {
                    self.pending = Some(Migration {
                        cause,
                        denoised_steps,
                        stored_layout,
                    });
                    return Poll::Ready(Step::Migrate);
                }
                Some(event) if self.skips(&event) => {}
                Some(event) => {
                    self.observe(&event);
                    return Poll::Ready(Step::Event(event));
                }
                None => {
                    self.finish();
                    return Poll::Ready(Step::Ended);
                }
            }
        }
    }

    /// Carries out the migration the last [`Self::poll_step`] reported, after
    /// which the events go on from the engine that serves the request now.
    ///
    /// # Errors
    ///
    /// Returns the terminal event the consumer receives instead.
    pub(crate) async fn migrate_pending(&mut self) -> Result<(), Event> {
        let Some(migration) = self.pending.take() else {
            return Ok(());
        };
        self.migrate(migration).await
    }

    /// The request's identity in the engine that serves it, which a
    /// successor takes its event stream over by; `None` once the request
    /// ended or was handed over.
    pub(crate) fn engine_request_id(&self) -> Option<RequestId> {
        self.open.then(|| self.stream.id())
    }

    /// The request's engine events its consumer took in all, and the public
    /// tokens they carried: the position a successor takes the stream over
    /// from (`Command::Takeover`).
    pub(crate) fn progress(&self) -> (u64, u64) {
        (
            self.stream.consumed(),
            self.request
                .as_ref()
                .and_then(|request| match request {
                    Resumable::Generation(generation) => {
                        Some(generation.generated_token_ids.len() as u64)
                    }
                    Resumable::Diffusion(_) => None,
                })
                .unwrap_or_default(),
        )
    }

    /// The request as submitted, encoded as JSON for the handoff state's
    /// record of what this process's pipeline processed.
    pub(crate) fn submitted(&self) -> Option<Vec<u8>> {
        let resumable = self.request.as_ref()?;
        let request = match resumable {
            Resumable::Generation(generation) => {
                Request::Generation(Box::new(generation.as_ref().clone()))
            }
            Resumable::Diffusion(diffusion) => Request::Diffusion(diffusion.clone()),
        };
        serde_json::to_vec(&request).ok()
    }

    /// Ends this process's consumption of the request's events without
    /// cancelling it: the engine's next consumer takes the stream over
    /// (design §7.5.1). The stream's owner reports the progress it reached
    /// before this call.
    pub(crate) fn hand_over(&mut self) {
        self.open = false;
        self.request = None;
        self.stream.hand_over();
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
    }

    /// Accepts every text token received from the engine that serves now as
    /// completing no stop string; see `EventStream::acknowledge_consumed_prefix`.
    pub fn acknowledge_consumed_prefix(&mut self) {
        self.stream.acknowledge_consumed_prefix();
    }

    /// Cancels the request.
    pub fn cancel(&mut self) {
        self.cancel_at_consumed_prefix(StreamCancelCause::DroppedStream);
    }

    /// Closes generation after the tokens received; see
    /// `EventStream::cancel_at_consumed_prefix`. A request whose migration is
    /// pending is not submitted again.
    pub fn cancel_at_consumed_prefix(&mut self, cause: StreamCancelCause) {
        self.open = false;
        self.request = None;
        self.stream.cancel_at_consumed_prefix(cause);
    }

    /// Whether `event` repeats what the consumer received from an engine
    /// before: the scheduling of a resumed request.
    fn skips(&mut self, event: &Event) -> bool {
        match event {
            Event::Scheduled { .. } if self.resumed && self.scheduled => {
                self.resumed = false;
                true
            }
            _ => false,
        }
    }

    /// Hands `event` to the consumer, keeping the tokens it carries.
    fn deliver(&mut self, event: Option<Event>) -> Option<Event> {
        match &event {
            Some(event) => self.observe(event),
            None => self.finish(),
        }
        event
    }

    /// Keeps the tokens of `event`, and finishes on a terminal event.
    fn observe(&mut self, event: &Event) {
        match event {
            Event::Scheduled { .. } => self.scheduled = true,
            Event::TextToken { id, .. } => self.keep(&[*id]),
            event if event.is_terminal() => self.finish(),
            _ => {}
        }
    }

    /// Appends generated tokens to the request a migration submits; the
    /// first after a migration ends its wait, which is logged.
    fn keep(&mut self, ids: &[u32]) {
        if let Some(migrated_at) = self.migrated_at.take() {
            tracing::info!(
                wait_ms = migrated_at.elapsed().as_millis() as u64,
                "a migrated request generates on the engine that serves it"
            );
        }
        if let Some(Resumable::Generation(request)) = &mut self.request {
            request.generated_token_ids.extend_from_slice(ids);
        }
    }

    /// Submits the request again, with its generated tokens or the
    /// trajectory steps the engine stored (`Migration::denoised_steps`,
    /// zero for none), to the engine that serves next, and goes on with
    /// that engine's events.
    ///
    /// # Errors
    ///
    /// Returns the terminal event the consumer receives instead: the
    /// request does not resume, was cancelled, lost its engine more often
    /// than it may, or could not be submitted again.
    async fn migrate(&mut self, migration: Migration) -> Result<(), Event> {
        let cause = migration.cause;
        let started = Instant::now();
        tracing::info!(
            ?cause,
            request = self.stream.id().0,
            "a response migrates to the engine that serves it now"
        );
        let failed = |message: String| Event::Error { message };
        if !self.open {
            return Err(failed(
                "the request was cancelled while it migrated".to_owned(),
            ));
        }
        let (Some(request), Some(hold)) = (&self.request, &self.hold) else {
            return Err(failed(
                "the request's engine stopped serving it, and it does not migrate".to_owned(),
            ));
        };
        if cause == MigrationCause::Loss {
            if self.losses >= self.max_migrations {
                return Err(failed(format!(
                    "the request's workers were lost {} times",
                    self.losses + 1
                )));
            }
            self.losses += 1;
        }
        let hold = Arc::clone(hold);
        // A request an epoch change moves to an epoch computing other
        // numerics continues with the state it reached (`in_flight = keep`).
        let previous_epoch = (cause == MigrationCause::Drain)
            .then(|| self.stream.epoch())
            .flatten();
        // Admission resolves the current owner first. After a daemon handoff
        // that owner supplies both the engine and the continuation's state.
        // A migration that finds no engine must say so: the wait for one has no
        // other bound (a deployment between engines is the ordinary case a
        // successor daemon covers), so a request that finds none is answered
        // with its own deadline rather than left open on the engine that went.
        let (client, kept) = tokio::time::timeout(MIGRATION_TIMEOUT, hold.ready(previous_epoch))
            .await
            .map_err(|_| {
                failed(format!(
                    "the request could not migrate: no engine served it within {} s",
                    MIGRATION_TIMEOUT.as_secs()
                ))
            })?
            .map_err(|error| failed(format!("the request could not migrate: {error}")))?;
        let mut continuation = request.continuation(&migration, kept);
        if let Request::Generation(generation) = &mut continuation {
            generation.cache.resume = generation.cache.resume.or(self.stream.cache_resume());
        }
        let (stream, _scheduler) = client
            .submit(continuation.clone())
            .await
            .map_err(|error| failed(format!("the request could not migrate: {error}")))?;
        // The next migration submits the request as submitted now.
        match (&continuation, &mut self.request) {
            (Request::Generation(submitted), Some(Resumable::Generation(request))) => {
                request.cache = submitted.cache.clone();
            }
            (Request::Diffusion(submitted), Some(Resumable::Diffusion(request))) => {
                *request = submitted.clone();
            }
            _ => {}
        }
        // The previous engine's stream ended with its migratable event.
        self.stream = stream;
        // Whether the next engine's scheduling repeats one the consumer read
        // is not knowable here — an engine lost before it scheduled anything
        // leaves the next engine's first scheduling the only one the consumer
        // will ever see — so `skips` answers it from what the consumer read.
        self.resumed = true;
        self.migrated_at = Some(started);
        Ok(())
    }

    /// Ends the stream on its side and runs its callback once.
    fn finish(&mut self) {
        self.open = false;
        self.request = None;
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
    }
}

impl Drop for RequestEvents {
    /// Runs a callback that has not run; the engine's stream cancels a
    /// request that has not ended as it drops.
    fn drop(&mut self) {
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
    }
}

impl crate::api::events::Events for RequestEvents {
    fn poll_step(&mut self, cx: &mut Context<'_>) -> Poll<Step> {
        RequestEvents::poll_step(self, cx)
    }

    fn recv(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<Event>> + Send + '_>> {
        Box::pin(RequestEvents::recv(self))
    }

    fn migrate_pending(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), Event>> + Send + '_>> {
        Box::pin(RequestEvents::migrate_pending(self))
    }

    fn acknowledge_consumed_prefix(&mut self) {
        RequestEvents::acknowledge_consumed_prefix(self);
    }

    fn stop_at_consumed_prefix(&mut self) {
        self.cancel_at_consumed_prefix(StreamCancelCause::StopStringMatched);
    }

    fn cancel(&mut self) {
        RequestEvents::cancel(self);
    }

    fn hand_over(&mut self) {
        RequestEvents::hand_over(self);
    }

    fn engine_request_id(&self) -> Option<RequestId> {
        RequestEvents::engine_request_id(self)
    }

    fn progress(&self) -> (u64, u64) {
        RequestEvents::progress(self)
    }

    fn submitted(&self) -> Option<Vec<u8>> {
        RequestEvents::submitted(self)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use proto::request::{DiffusionSamplingParams, GenerationRequest, SamplingParams, Trajectory};

    use super::*;
    use crate::engine::client::EngineClient;
    use crate::test_support::{SERVED_MODEL, letter_fake};

    /// A text request of a four-token prompt that generates `max_tokens`
    /// tokens.
    fn request(max_tokens: usize) -> GenerationRequest {
        GenerationRequest {
            request_id: Default::default(),
            prompt_token_ids: vec![1, 2, 3, 4],
            multimodal_inputs: Default::default(),
            sampling: SamplingParams::default(),
            max_und_tokens: max_tokens,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            generated_token_ids: Vec::new(),
            adapter: 0,
        }
    }

    /// A client of one scheduler over a fake whose text never ends, so a
    /// request generates until its limit.
    fn client() -> Arc<EngineClient> {
        engine().0
    }

    /// [`client`], with the engine's process link, on which its daemon
    /// drains it.
    fn engine() -> (
        Arc<EngineClient>,
        std::sync::mpsc::Sender<proto::process::Command>,
    ) {
        let (client, commands, _reports) = crate::test_support::fake_engine(
            testkit::args(SERVED_MODEL),
            testkit::Fixture::new(letter_fake),
        );
        (Arc::new(client), commands)
    }

    /// Receives the next event within ten seconds.
    async fn next(events: &mut RequestEvents) -> Option<Event> {
        tokio::time::timeout(Duration::from_secs(10), events.recv())
            .await
            .expect("an event within ten seconds")
    }

    /// A request whose engine drains goes on on the engine that serves next:
    /// its consumer reads every token it generates, one scheduling event,
    /// and a finish that charges its whole usage.
    #[tokio::test]
    async fn a_drained_engines_request_continues_on_the_next_engine() {
        let hold = Arc::new(RequestHold::new());
        let (first, process) = engine();
        hold.serve(Arc::clone(&first));
        let migrator = Migrator::new(Arc::clone(&hold), 0);
        let (mut events, _) = migrator
            .submit(Request::Generation(Box::new(request(200))))
            .await
            .unwrap();

        let (mut tokens, mut scheduled) = (0, 0);
        while tokens < 5 {
            match next(&mut events).await {
                Some(Event::TextToken { .. }) => tokens += 1,
                Some(Event::Scheduled { .. }) => scheduled += 1,
                Some(_) => {}
                None => panic!("the request ended before its engine drained"),
            }
        }
        hold.serve(client());
        process
            .send(proto::process::Command {
                kind: Some(proto::process::command::Kind::Drain(
                    proto::process::Drain {
                        epoch: first.epoch(),
                        ..Default::default()
                    },
                )),
            })
            .unwrap();

        let finished = loop {
            match next(&mut events).await {
                Some(Event::TextToken { .. }) => tokens += 1,
                Some(Event::Scheduled { .. }) => scheduled += 1,
                Some(event @ Event::Finished { .. }) => break event,
                Some(event) if event.is_terminal() => panic!("{event:?}"),
                Some(_) => {}
                None => panic!("the request ended without a finish"),
            }
        };
        assert_eq!(tokens, 200);
        assert_eq!(scheduled, 1, "the resumed scheduling is not repeated");
        assert!(
            matches!(
                finished,
                Event::Finished {
                    prompt_tokens: 4,
                    completion_tokens: 200,
                    ..
                }
            ),
            "{finished:?}"
        );
    }

    /// The events of a request that resumes from `generated` tokens, fed by
    /// `events` and migrating through `hold`.
    fn resumable(
        events: mpsc::UnboundedReceiver<Event>,
        hold: Arc<RequestHold>,
        max_migrations: u32,
    ) -> RequestEvents {
        RequestEvents {
            stream: EventStream::detached(events),
            request: Some(Resumable::Generation(Box::new(request(8)))),
            hold: Some(hold),
            max_migrations,
            losses: 0,
            pending: None,
            resumed: false,
            scheduled: false,
            migrated_at: None,
            open: true,
            on_finish: None,
        }
    }

    /// A request whose engine was lost before it sent any scheduling reads
    /// the next engine's scheduling: the repeat a migration removes is one
    /// the consumer already read, and a consumer that read none has no output
    /// to put a generated token in.
    #[test]
    fn a_resumed_stream_keeps_the_scheduling_its_consumer_never_read() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(tx);
        let mut events = RequestEvents {
            stream: EventStream::detached(rx),
            request: None,
            hold: None,
            max_migrations: 0,
            losses: 1,
            pending: None,
            resumed: true,
            scheduled: false,
            migrated_at: None,
            open: true,
            on_finish: None,
        };
        let scheduling = || Event::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        };
        assert!(
            !events.skips(&scheduling()),
            "the next engine's scheduling is the first the consumer reads, not a repeat"
        );
        events.observe(&scheduling());
        assert!(
            events.skips(&scheduling()),
            "a second scheduling is the repeat a migration removes"
        );
    }

    /// A request survives `max_migrations` migrations caused by losses and
    /// fails on the next; a drain is not counted.
    #[tokio::test]
    async fn losses_beyond_the_limit_fail_the_request() {
        let hold = Arc::new(RequestHold::new());
        hold.close("no engine serves");
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(Event::Migratable {
            cause: MigrationCause::Loss,
            denoised_steps: 0,
            stored_layout: Vec::new(),
        })
        .unwrap();
        let mut events = resumable(rx, hold, 0);
        match next(&mut events).await {
            Some(Event::Error { message }) => {
                assert!(message.contains("lost 1 times"), "{message}")
            }
            other => panic!("expected the loss to fail the request, got {other:?}"),
        }
    }

    /// A video request leaves its engine naming the trajectory steps the
    /// engine stored and where they lie: it is submitted again with them
    /// and its trajectory's identity, reading the stored trajectory under
    /// the next epoch's own latent seed, or under the parked epoch's when an
    /// epoch change keeps the state of the requests in flight. A migration
    /// that stored nothing submits the request naming what it named before.
    #[test]
    fn a_video_request_migrates_with_the_steps_its_engine_stored() {
        let video = DiffusionRequest {
            request_id: Default::default(),
            prompt_token_ids: vec![1, 2],
            conditioning: Vec::new(),
            conditions: Vec::new(),
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: 25,
                video_units: 1,
                seed: 3,
                width: 64,
                height: 32,
                condition_rows: 0,
            },
            trajectory: Trajectory {
                id: 7,
                ..Trajectory::default()
            },
            encoder_inputs: None,
        };
        let trajectory = |request: Request| match request {
            Request::Diffusion(request) => request.trajectory,
            Request::Generation(_) => panic!("a video request migrates as one"),
        };
        let resumable = Resumable::Diffusion(video.clone());
        let migration = |cause, denoised_steps, stored_layout: &[u8]| Migration {
            cause,
            denoised_steps,
            stored_layout: stored_layout.to_vec(),
        };
        let drained = migration(MigrationCause::Drain, 3, b"pages");
        let stored = trajectory(resumable.continuation(&drained, None));
        assert_eq!(
            stored,
            Trajectory {
                id: 7,
                denoised_steps: 3,
                state_seed: None,
                stored_layout: b"pages".to_vec(),
            }
        );
        let kept = Kept {
            epoch: 1,
            prefix_seeds: None,
            latent_seed: Some(11),
        };
        assert_eq!(
            trajectory(resumable.continuation(&drained, Some(kept))).state_seed,
            Some(11)
        );

        let named = Resumable::Diffusion(DiffusionRequest {
            trajectory: Trajectory {
                id: 7,
                denoised_steps: 3,
                state_seed: Some(11),
                stored_layout: b"pages".to_vec(),
            },
            encoder_inputs: None,
            ..video
        });
        assert_eq!(
            trajectory(named.continuation(&migration(MigrationCause::Loss, 0, &[]), None)),
            Trajectory {
                id: 7,
                denoised_steps: 3,
                state_seed: Some(11),
                stored_layout: b"pages".to_vec(),
            }
        );
    }

    /// A request that does not resume from its tokens fails when its engine
    /// ends it as migratable.
    #[tokio::test]
    async fn a_request_that_does_not_migrate_fails_on_a_migratable_end() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(Event::Migratable {
            cause: MigrationCause::Drain,
            denoised_steps: 0,
            stored_layout: Vec::new(),
        })
        .unwrap();
        let mut events = RequestEvents::detached(rx);
        assert!(matches!(next(&mut events).await, Some(Event::Error { .. })));
    }
}
