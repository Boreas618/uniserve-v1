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
//! `max_migrations` migrations caused by losses of its workers or by failed
//! transfers between its worker groups, together, and fails on the next one:
//! a failed transfer leaves the engine serving, and the request's next attempt
//! may take the same path, so without the bound a copy that always fails
//! would resubmit it forever.
//!
//! The random stream of a migrated request continues as it began: every
//! request carries a seed from entry, its own or one drawn for it
//! (`api::input::request_seed`), and every draw of a token request follows
//! its seed and its token position alone; a latent encoding's posterior draw follows
//! the encoding's place among the request's encodings, which the request's
//! facts fix. A video request's draws follow its seed. The migrator gives a
//! video request the identity that keys its stored trajectories
//! (`Trajectory::id`) when it is first submitted.

use std::num::{NonZeroU32, NonZeroU64};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};
use std::time::Instant;

use proto::request::{
    CacheResume, DiffusionRequest, Event, GenerationRequest, MigrationCause, Request, RequestId,
    Trajectory,
};

use crate::api::events::Events;
use crate::engine::client::{EventStream, StreamCancelCause, SubmitError};
use crate::engine::hold::{Kept, RequestHold};

/// Submits the leader's requests and moves its token requests between
/// engines.
pub struct Migrator {
    hold: Arc<RequestHold>,
    /// Migrations caused by losses or failed transfers a request survives.
    max_migrations: u32,
}

impl Migrator {
    /// A migrator that submits through `hold`, under which a request survives
    /// `max_migrations` migrations caused by losses or failed transfers.
    pub fn new(hold: Arc<RequestHold>, max_migrations: u32) -> Self {
        Self {
            hold,
            max_migrations,
        }
    }

    /// Submits `request` through the hold and returns its events and the
    /// scheduler serving it, as `RequestHold::submit` does. A token request
    /// that resumes from its tokens, and a video request, are kept for a
    /// later migration, a token request with the seed it carries from entry
    /// (`api::input::request_seed`), so that its draws repeat on the engine
    /// that serves it next, and a video request with its trajectories'
    /// identity.
    ///
    /// # Errors
    ///
    /// Fails as `RequestHold::submit` fails.
    pub async fn submit(
        &self,
        mut request: Request,
    ) -> Result<(RequestEvents, Option<u32>), SubmitError> {
        let resumable = match &mut request {
            Request::Generation(generation) => Resumable::Generation(generation.clone()),
            Request::Diffusion(diffusion) => {
                if diffusion.trajectory.id.is_none() {
                    diffusion.trajectory.id = Some(trajectory_identity());
                }
                Resumable::Diffusion(diffusion.clone())
            }
        };
        let (stream, scheduler) = self.hold.submit(request).await?;
        let events = RequestEvents {
            stream,
            request: Some(resumable),
            hold: Some(Arc::clone(&self.hold)),
            max_migrations: self.max_migrations,
            counted: 0,
            pending: None,
            resumed: false,
            scheduled: false,
            migrated_at: None,
            open: true,
        };
        Ok((events, scheduler))
    }
}

/// A new identity of a video request's stored trajectories: a random one, so
/// that no two requests of any daemon key their trajectories alike.
fn trajectory_identity() -> NonZeroU64 {
    loop {
        if let Some(identity) = NonZeroU64::new(uuid::Uuid::new_v4().as_u64_pair().0) {
            return identity;
        }
    }
}

/// What one step of a request's events resolved to; an event's tokens are
/// already kept as the request's facts.
pub(crate) use crate::api::events::Next as Step;

/// A request a migration submits again, as last submitted.
enum Resumable {
    /// A token request, with the tokens it generated before.
    Generation(Box<GenerationRequest>),
    /// A video request, naming the trajectory steps an engine stored.
    Diffusion(DiffusionRequest),
}

impl Resumable {
    /// Makes this request the continuation `migration` submits, and returns
    /// it: a token request with the tokens it generated, its private pages
    /// named by `provenance` (none for a request its engine did not admit)
    /// unless an earlier migration named them already;
    /// a video request naming the trajectory steps its engine stored and its
    /// record of where they lie. `kept` is the state an epoch change keeps of
    /// the requests in flight (`in_flight = keep`), which the request
    /// continues with when the epoch that serves next computes other
    /// numerics. The next migration continues from what this one submits.
    fn resume(
        &mut self,
        migration: &Migration,
        kept: Option<Kept>,
        provenance: Option<CacheResume>,
    ) -> Request {
        let cause = migration.cause;
        match self {
            Self::Generation(request) => {
                // Its prompt's log-probabilities were reported where it began.
                request.sampling.return_prompt_logprobs = false;
                request.sampling.n_prompt_logprobs = 0;
                if request.cache.resume.is_none() {
                    request.cache.resume = provenance;
                }
                // It reads the state it reached and publishes later pages
                // only to its private continuation: they are neither epoch's.
                if let Some(seeds) = kept.and_then(|kept| kept.prefix_seeds) {
                    request.cache.state_seeds = seeds;
                    request.cache.write = false;
                }
                tracing::info!(
                    ?cause,
                    generated = request.generated_token_ids.len(),
                    "a request migrated to the engine that serves next"
                );
                Request::Generation(request.clone())
            }
            Self::Diffusion(request) => {
                if let Some(stored) = &migration.stored {
                    stored.name_in(&mut request.trajectory, kept);
                }
                tracing::info!(
                    ?cause,
                    denoised_steps = request.trajectory.denoised_steps,
                    stored = migration.stored.is_some(),
                    "a video request migrated to the engine that serves next"
                );
                Request::Diffusion(request.clone())
            }
        }
    }

    /// The request as last submitted.
    fn request(&self) -> Request {
        match self {
            Self::Generation(generation) => Request::Generation(generation.clone()),
            Self::Diffusion(diffusion) => Request::Diffusion(diffusion.clone()),
        }
    }
}

/// The steps of a video request's trajectory an engine stored when it ended
/// the request at a step boundary, and its record of where they lie.
struct Stored {
    steps: NonZeroU32,
    /// Where the engine put the stored trajectory's pages and how each
    /// stored rank laid its part out (`Trajectory::stored_layout`); opaque to
    /// the daemon.
    layout: Vec<u8>,
}

impl Stored {
    /// Names these steps in `trajectory`, keyed by what the engine that
    /// stored them computes: the next engine reads them when it computes the
    /// same, or when an epoch change keeps the state the request reached
    /// (`kept`), whatever its parallel configuration, from where the record
    /// says the pages lie.
    fn name_in(&self, trajectory: &mut Trajectory, kept: Option<Kept>) {
        trajectory.denoised_steps = self.steps.get();
        trajectory.state_seed = kept.and_then(|kept| kept.latent_seed);
        trajectory.stored_layout.clone_from(&self.layout);
    }
}

/// Why a request left its engine, and what the engine stored of it
/// (`Event::Migratable`).
struct Migration {
    cause: MigrationCause,
    /// The trajectory steps a video request's engine stored; `None` when it
    /// stored none, and the request names the trajectory it named before,
    /// which is the same at that step whenever it is read.
    stored: Option<Stored>,
}

impl Migration {
    /// The migration an `Event::Migratable` states: its cause, and the steps
    /// it stored (none at zero, as the request interface states none).
    fn of(cause: MigrationCause, denoised_steps: u32, stored_layout: Vec<u8>) -> Self {
        Self {
            cause,
            stored: NonZeroU32::new(denoised_steps).map(|steps| Stored {
                steps,
                layout: stored_layout,
            }),
        }
    }
}

/// One request's events across the engines that serve it.
///
/// It offers what an `EventStream` offers, as the API server's [`Events`]. An
/// `Event::Migratable` never reaches its consumer: the stream submits the
/// request again and goes on with the next engine's events, or ends with an
/// `Event::Error` when the request does not migrate. Dropping it before its
/// terminal event cancels the request on the engine that serves it.
pub struct RequestEvents {
    /// The events of the engine that serves the request now.
    stream: EventStream,
    /// The request as last submitted, with the tokens it generated before
    /// or the trajectory steps an engine stored; `None` for a request that
    /// does not resume, and once the request ended or was cancelled.
    request: Option<Resumable>,
    /// The hold the request migrates through; `None` for a request this
    /// process took over, which the engine that serves it admitted.
    hold: Option<Arc<RequestHold>>,
    max_migrations: u32,
    /// Migrations counted against `max_migrations` so far: those caused by
    /// losses and by failed transfers.
    counted: u32,
    /// A migration `poll_step` met, which the next `recv` or
    /// `migrate_pending` carries out.
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
}

impl RequestEvents {
    /// The events of a request this process takes over from another: the
    /// engine's stream `stream`, which goes on after the position the
    /// previous consumer reached, and no hold, because the request is already
    /// admitted and migrated by the engine itself.
    pub(crate) fn taken_over(stream: EventStream, consumed: u64) -> Self {
        Self {
            stream,
            request: None,
            hold: None,
            max_migrations: 0,
            counted: 0,
            pending: None,
            resumed: consumed > 0,
            scheduled: consumed > 0,
            migrated_at: None,
            open: true,
        }
    }

    /// Receives the next event, carrying out any migration on the way;
    /// `None` once the request ended.
    async fn next_event(&mut self) -> Option<Event> {
        loop {
            if let Some(migration) = self.pending.take() {
                match self.migrate(migration).await {
                    Ok(()) => continue,
                    Err(event) => return self.deliver(Some(event)),
                }
            }
            match self.stream.recv().await {
                Some(Event::Migratable {
                    cause,
                    denoised_steps,
                    stored_layout,
                }) => self.pending = Some(Migration::of(cause, denoised_steps, stored_layout)),
                Some(event) if self.skips(&event) => {}
                event => return self.deliver(event),
            }
        }
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

    /// Closes generation after the tokens received, as
    /// `EventStream::cancel_at_consumed_prefix` does; a request whose
    /// migration is pending is not submitted again.
    fn close(&mut self, cause: StreamCancelCause) {
        self.open = false;
        self.request = None;
        self.stream.cancel_at_consumed_prefix(cause);
    }

    /// Submits the request again, with its generated tokens or the
    /// trajectory steps the engine stored, to the engine that serves next,
    /// and goes on with that engine's events.
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
        let (Some(request), Some(hold)) = (&mut self.request, &self.hold) else {
            return Err(failed(
                "the request's engine stopped serving it, and it does not migrate".to_owned(),
            ));
        };
        // A loss and a failed transfer are failures the next attempt may meet
        // again; a drain is the deployment's own decision.
        if matches!(cause, MigrationCause::Loss | MigrationCause::Transfer) {
            if self.counted >= self.max_migrations {
                return Err(failed(format!(
                    "the request's workers were lost or its transfers failed {} times",
                    self.counted + 1
                )));
            }
            self.counted += 1;
        }
        // A request an epoch change moves to an epoch computing other
        // numerics continues with the state it reached (`in_flight = keep`).
        let previous_epoch = (cause == MigrationCause::Drain).then(|| self.stream.epoch());
        // Admission resolves the current owner first. After a daemon handoff
        // that owner supplies both the engine and the continuation's state.
        // The request waits as every held submission does (`RequestHold`): until
        // an engine serves, however long the deployment takes to start one (a
        // relaunch captures every graph again), or until the hold closes because
        // no engine will serve, which fails it with the reason. Its client bounds
        // the wait too: a disconnect drops these events and cancels it.
        let (client, kept) = hold
            .ready(previous_epoch)
            .await
            .map_err(|error| failed(format!("the request could not migrate: {error}")))?;
        let continuation = request.resume(&migration, kept, self.stream.cache_resume());
        let (stream, _scheduler) = client
            .submit(continuation)
            .await
            .map_err(|error| failed(format!("the request could not migrate: {error}")))?;
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

    /// Ends the stream on its side.
    fn finish(&mut self) {
        self.open = false;
        self.request = None;
    }
}

impl Events for RequestEvents {
    fn poll_step(&mut self, cx: &mut Context<'_>) -> Poll<Step> {
        loop {
            if self.pending.is_some() {
                return Poll::Ready(Step::Migrate);
            }
            match ready!(self.stream.poll_recv(cx)) {
                Some(Event::Migratable {
                    cause,
                    denoised_steps,
                    stored_layout,
                }) => {
                    self.pending = Some(Migration::of(cause, denoised_steps, stored_layout));
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

    fn recv(&mut self) -> Pin<Box<dyn Future<Output = Option<Event>> + Send + '_>> {
        Box::pin(self.next_event())
    }

    fn migrate_pending(&mut self) -> Pin<Box<dyn Future<Output = Result<(), Event>> + Send + '_>> {
        Box::pin(async move {
            match self.pending.take() {
                Some(migration) => self.migrate(migration).await,
                None => Ok(()),
            }
        })
    }

    fn acknowledge_consumed_prefix(&mut self) {
        self.stream.acknowledge_consumed_prefix();
    }

    fn stop_at_consumed_prefix(&mut self) {
        self.close(StreamCancelCause::StopStringMatched);
    }

    fn cancel(&mut self) {
        self.close(StreamCancelCause::DroppedStream);
    }

    fn hand_over(&mut self) {
        self.open = false;
        self.request = None;
        self.stream.hand_over();
    }

    fn engine_request_id(&self) -> Option<RequestId> {
        self.open.then(|| self.stream.engine_id()).flatten()
    }

    fn progress(&self) -> (u64, u64) {
        let tokens = match &self.request {
            Some(Resumable::Generation(generation)) => generation.generated_token_ids.len() as u64,
            Some(Resumable::Diffusion(_)) | None => 0,
        };
        (self.stream.consumed(), tokens)
    }

    fn submitted(&self) -> Option<Vec<u8>> {
        // The record is the request interface's own message, the one the
        // handoff schema names (`uniserve.request.Request`).
        let request = self.request.as_ref()?.request();
        Some(prost::Message::encode_to_vec(&request.to_wire()))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use proto::request::{DiffusionSamplingParams, SamplingParams};

    use super::*;
    use crate::engine::client::EngineClient;
    use crate::test_support::{Arrival, SERVED_MODEL, letter_fake, scripted_client};

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

    /// A video request of a two-token prompt, whose trajectory is `trajectory`.
    fn video(trajectory: Trajectory) -> DiffusionRequest {
        DiffusionRequest {
            request_id: Default::default(),
            prompt_token_ids: vec![1, 2],
            conditioning: Vec::new(),
            conditions: Vec::new(),
            media: Vec::new(),
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: 25,
                video_units: 1,
                seed: 3,
                width: 64,
                height: 32,
                condition_rows: 0,
            },
            trajectory,
            encoder_inputs: None,
        }
    }

    /// The migratable end an engine states for `cause`, storing `steps`
    /// trajectory steps laid out as `layout`.
    fn migratable(cause: MigrationCause, steps: u32, layout: &[u8]) -> Event {
        Event::Migratable {
            cause,
            denoised_steps: steps,
            stored_layout: layout.to_vec(),
        }
    }

    /// A finish of `completion_tokens` tokens.
    fn finished(completion_tokens: usize) -> Event {
        Event::Finished {
            reason: proto::request::FinishReason::MaxTokens,
            stop_reason: None,
            prompt_tokens: 4,
            completion_tokens,
        }
    }

    /// An engine of epoch `epoch` that ends every request it is given as
    /// migratable for `cause`, storing nothing.
    fn migrating(epoch: u64, cause: MigrationCause) -> Arc<EngineClient> {
        Arc::new(scripted_client(epoch, move |_| vec![migratable(cause, 0, &[])]).0)
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
            match events.recv().await {
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
            match events.recv().await {
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

    /// A request whose engine was lost before it sent any scheduling reads
    /// the next engine's scheduling: the repeat a migration removes is one
    /// the consumer already read, and a consumer that read none has no output
    /// to put a generated token in.
    #[tokio::test]
    async fn a_resumed_stream_keeps_the_scheduling_its_consumer_never_read() {
        let hold = Arc::new(RequestHold::new());
        hold.serve(migrating(1, MigrationCause::Loss));
        let migrator = Migrator::new(Arc::clone(&hold), 1);
        let (mut events, _) = migrator
            .submit(Request::Generation(Box::new(request(1))))
            .await
            .unwrap();

        let scheduling = || Event::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
            cached_tokens: 0,
        };
        let (next, _) = scripted_client(1, move |_| {
            vec![
                scheduling(),
                Event::TextToken {
                    id: u32::from(b'a'),
                    logprob: None,
                },
                finished(1),
            ]
        });
        hold.serve(Arc::new(next));
        assert!(
            matches!(events.recv().await, Some(Event::Scheduled { .. })),
            "the next engine's scheduling is the first the consumer reads, not a repeat"
        );
        assert!(matches!(events.recv().await, Some(Event::TextToken { .. })));
        assert!(matches!(events.recv().await, Some(Event::Finished { .. })));
    }

    /// A request whose engine was lost waits in the hold for as long as the
    /// deployment takes to put another engine in front of it — a relaunch
    /// captures every graph again, which takes minutes — and then continues on
    /// that engine to its last token: a held request is bounded by its client
    /// and by the hold closing, never by a clock of its own.
    #[tokio::test]
    async fn a_lost_engines_request_waits_through_a_slow_relaunch() {
        let hold = Arc::new(RequestHold::new());
        hold.serve(migrating(1, MigrationCause::Loss));
        let migrator = Migrator::new(Arc::clone(&hold), 3);
        let (mut events, _) = migrator
            .submit(Request::Generation(Box::new(request(8))))
            .await
            .unwrap();
        hold.hold();

        // Ten minutes pass on the paused clock with no engine serving: longer
        // than any engine's start.
        tokio::time::pause();
        let waiting = tokio::spawn(async move {
            let first = events.recv().await;
            (first, events)
        });
        tokio::time::sleep(Duration::from_secs(600)).await;
        assert!(
            !waiting.is_finished(),
            "the request stopped waiting while the deployment had no engine"
        );

        tokio::time::resume();
        hold.serve(client());
        let (first, mut events) = waiting.await.unwrap();
        let mut tokens = usize::from(matches!(first, Some(Event::TextToken { .. })));
        let finished = loop {
            match events.recv().await {
                Some(Event::TextToken { .. }) => tokens += 1,
                Some(event @ Event::Finished { .. }) => break event,
                Some(event) if event.is_terminal() => panic!("{event:?}"),
                Some(_) => {}
                None => panic!("the request ended without a finish"),
            }
        };
        assert_eq!(tokens, 8);
        assert!(
            matches!(
                finished,
                Event::Finished {
                    completion_tokens: 8,
                    ..
                }
            ),
            "{finished:?}"
        );
    }

    /// A request survives `max_migrations` migrations caused by losses or
    /// failed transfers, together, and fails on the next.
    #[tokio::test]
    async fn losses_and_failed_transfers_beyond_the_limit_fail_the_request() {
        for cause in [MigrationCause::Loss, MigrationCause::Transfer] {
            let hold = Arc::new(RequestHold::new());
            hold.serve(migrating(1, cause));
            let migrator = Migrator::new(Arc::clone(&hold), 0);
            let (mut events, _) = migrator
                .submit(Request::Generation(Box::new(request(8))))
                .await
                .unwrap();
            match events.recv().await {
                Some(Event::Error { message }) => {
                    assert!(message.contains("failed 1 times"), "{cause:?}: {message}")
                }
                other => panic!("expected the {cause:?} to fail the request, got {other:?}"),
            }
        }
    }

    /// The continuation of a video request the engine of epoch 1 ends with
    /// `migration`, migrating to an engine that records it, under a hold that
    /// keeps `kept`.
    async fn continuation(
        trajectory: Trajectory,
        migration: Event,
        kept: Option<Kept>,
    ) -> (Trajectory, Trajectory) {
        let hold = Arc::new(RequestHold::new());
        let (first, first_engine) = scripted_client(1, move |_| vec![migration.clone()]);
        hold.serve(Arc::new(first));
        if let Some(kept) = kept {
            hold.keep(kept);
        }
        let migrator = Migrator::new(Arc::clone(&hold), 1);
        let (mut events, _) = migrator
            .submit(Request::Diffusion(video(trajectory)))
            .await
            .unwrap();

        let (next, next_engine) = scripted_client(2, |_| vec![finished(0)]);
        hold.serve(Arc::new(next));
        assert!(matches!(events.recv().await, Some(Event::Finished { .. })));
        let submitted =
            |engine: &crate::test_support::ScriptedEngine| match engine.submitted.recv().unwrap() {
                Request::Diffusion(request) => request.trajectory,
                Request::Generation(_) => panic!("a video request migrates as one"),
            };
        (submitted(&first_engine), submitted(&next_engine))
    }

    /// A video request leaves its engine naming the trajectory steps the
    /// engine stored and where they lie: it is submitted again with them and
    /// its trajectory's identity, reading the stored trajectory under the next
    /// epoch's own latent seed, or under the drained epoch's when an epoch
    /// change keeps the state of the requests in flight. A migration that
    /// stored nothing submits the request naming what it named before.
    #[tokio::test]
    async fn a_video_request_migrates_with_the_steps_its_engine_stored() {
        let drained = migratable(MigrationCause::Drain, 3, b"pages");
        let (first, next) = continuation(Trajectory::default(), drained.clone(), None).await;
        assert!(first.id.is_some(), "the migrator names the trajectory");
        assert_eq!(
            next,
            Trajectory {
                id: first.id,
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
        let (_, next) = continuation(Trajectory::default(), drained, Some(kept)).await;
        assert_eq!(next.state_seed, Some(11));

        let named = Trajectory {
            id: NonZeroU64::new(7),
            denoised_steps: 3,
            state_seed: Some(11),
            stored_layout: b"pages".to_vec(),
        };
        let lost = migratable(MigrationCause::Loss, 0, &[]);
        let (first, next) = continuation(named.clone(), lost, None).await;
        assert_eq!(first, named);
        assert_eq!(next, named);
    }

    /// A request this process took over, which the engine that serves it
    /// admitted, fails when its engine ends it as migratable: it does not
    /// migrate from here.
    #[tokio::test]
    async fn a_request_that_does_not_migrate_fails_on_a_migratable_end() {
        let (client, _engine) = scripted_client(1, |arrival| match arrival {
            Arrival::TakenOver(RequestId(5)) => vec![migratable(MigrationCause::Drain, 0, &[])],
            Arrival::TakenOver(_) | Arrival::Submitted(_) => Vec::new(),
        });
        let stream = client.take_over(RequestId(5), 0).unwrap();
        let mut events = RequestEvents::taken_over(stream, 0);
        assert!(matches!(events.recv().await, Some(Event::Error { .. })));
    }
}
