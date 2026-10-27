//! Request migration: how a token or video request moves between engines.
//!
//! An engine that parks for a world change or a rollback, sleeps, rewrites
//! its weights in place, or loses the workers that hold a request, ends the
//! request as migratable (`Event::Migratable`): a token request after every
//! token it generated there, a video request at a denoising step boundary,
//! with the steps of its trajectory the engine stored. The leader submits
//! every request through its [`Migrator`], which keeps each token request's
//! generated tokens and each video request's stored steps and, on that
//! event, submits the request again with them
//! (`GenerationRequest::generated_token_ids`, `Trajectory::denoised_steps`)
//! through the hold to the engine that serves next. The request's consumer
//! reads one stream throughout ([`RequestEvents`]): the new engine
//! continues after the last token it received and charges the request's
//! whole usage, or continues the trajectory from the stored step, and the
//! daemon's detokenizer, stop-string matcher and response assembly keep
//! their state.
//!
//! A migration caused by a drain is not counted. A request survives
//! `max_migrations` migrations caused by losses and fails on the next one.
//! A request that does not resume (an image, a readout) fails when its
//! engine ends it as migratable.
//!
//! The random stream of a migrated request continues as it began: the
//! migrator gives a request that names no seed one of its own when it is
//! first submitted, and every draw of a seeded request follows its seed and
//! its token position alone. A video request's draws follow its seed, and
//! the migrator gives it the identity that keys its stored trajectory
//! (`Trajectory::id`) when it is first submitted.

use std::sync::Arc;
use std::time::Instant;

use proto::request::{DiffusionRequest, Event, GenerationRequest, MigrationCause, Request};
use tokio::sync::mpsc;

use crate::engine::client::{EventStream, StreamCancelCause, SubmitError};
use crate::engine::hold::{Kept, RequestHold};

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

    /// The hold every submission goes through.
    pub fn hold(&self) -> &Arc<RequestHold> {
        &self.hold
    }

    /// Submits `request` through the hold and returns its events and the
    /// scheduler serving it, as `RequestHold::submit` does. A token request
    /// that resumes from its tokens, and a video request, are kept for a
    /// later migration; a token request that names no seed takes one first,
    /// so that its draws repeat on the engine that serves it next, and a
    /// video request takes its trajectory's identity.
    ///
    /// # Errors
    ///
    /// Fails as `RequestHold::submit` fails.
    pub async fn submit(
        &self,
        mut request: Request,
    ) -> Result<(RequestEvents, Option<u32>), SubmitError> {
        let resumable = match &mut request {
            Request::Generation(generation) if generation.is_migratable() => {
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
            Request::Generation(_) => None,
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
            migrated_at: None,
            open: true,
            on_finish: None,
        };
        Ok((events, scheduler))
    }
}

/// A request a migration submits again, as last submitted.
#[derive(Clone)]
enum Resumable {
    /// A token request, with the tokens it generated before.
    Generation(Box<GenerationRequest>),
    /// A video request, naming the trajectory steps an engine stored.
    Diffusion(DiffusionRequest),
}

impl Resumable {
    /// The request a migration for `cause` submits again: a token request
    /// with the tokens it generated, a video request naming the trajectory
    /// steps its engine stored (`denoised_steps`, zero for none). `kept`
    /// is the state a world change keeps of the requests in flight
    /// (`in_flight = keep`), which the request continues with when the
    /// world that serves next computes other numerics.
    fn continuation(
        &self,
        cause: MigrationCause,
        denoised_steps: u32,
        kept: Option<Kept>,
    ) -> Request {
        match self {
            Self::Generation(request) => {
                let mut continuation = request.as_ref().clone();
                // Its prompt's log-probabilities were reported where it began.
                continuation.sampling.return_prompt_logprobs = false;
                continuation.sampling.n_prompt_logprobs = 0;
                // It reads the pages of the state it reached and writes none,
                // since its later pages are neither world's.
                if let Some(seed) = kept.and_then(|kept| kept.prefix_seed) {
                    continuation.cache.state_seed = Some(seed);
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
                // The engine stored the trajectory at a step boundary, keyed
                // by what it computes: the next engine reads it when it
                // computes the same, or when a world change keeps the state
                // the request reached. A request the engine stored nothing
                // of names the trajectory it named before, which is the
                // same at that step whenever it is read.
                if denoised_steps > 0 {
                    continuation.trajectory.denoised_steps = denoised_steps;
                    continuation.trajectory.state_seed = kept.and_then(|kept| kept.latent_seed);
                }
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
    /// A migration `try_recv` met, which the next `recv` carries out, with
    /// the trajectory steps the engine stored.
    pending: Option<(MigrationCause, u32)>,
    /// Whether the stream resumed on another engine, whose scheduling event
    /// repeats one the consumer received.
    resumed: bool,
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
            migrated_at: None,
            open: true,
            on_finish: None,
        }
    }

    /// Registers a callback to run once, on the terminal event, the end of
    /// the stream or its drop.
    pub fn set_on_finish(&mut self, on_finish: impl FnOnce() + Send + 'static) {
        self.on_finish = Some(Box::new(on_finish));
    }

    /// Receives the next event; `None` once the request ended.
    pub async fn recv(&mut self) -> Option<Event> {
        loop {
            if let Some((cause, denoised_steps)) = self.pending.take() {
                match self.migrate(cause, denoised_steps).await {
                    Ok(()) => continue,
                    Err(event) => return self.deliver(Some(event)),
                }
            }
            let event = self.stream.recv().await;
            match event {
                Some(Event::Migratable {
                    cause,
                    denoised_steps,
                }) => self.pending = Some((cause, denoised_steps)),
                Some(event) if self.skips(&event) => {}
                event => return self.deliver(event),
            }
        }
    }

    /// Receives the next event; the same as [`Self::recv`].
    pub async fn next(&mut self) -> Option<Event> {
        self.recv().await
    }

    /// Receives the next event without waiting.
    ///
    /// A migration it meets waits for the next [`Self::recv`], which carries
    /// it out; until then it reports no event.
    ///
    /// # Errors
    ///
    /// Returns the channel's error when no event waits or the request ended.
    pub fn try_recv(&mut self) -> Result<Event, mpsc::error::TryRecvError> {
        loop {
            if self.pending.is_some() {
                return Err(mpsc::error::TryRecvError::Empty);
            }
            match self.stream.try_recv() {
                Ok(Event::Migratable {
                    cause,
                    denoised_steps,
                }) => self.pending = Some((cause, denoised_steps)),
                Ok(event) if self.skips(&event) => {}
                Ok(event) => {
                    self.observe(&event);
                    return Ok(event);
                }
                Err(error) => {
                    if error == mpsc::error::TryRecvError::Disconnected {
                        self.finish();
                    }
                    return Err(error);
                }
            }
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
        if self.resumed && matches!(event, Event::Scheduled { .. }) {
            self.resumed = false;
            return true;
        }
        false
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
            Event::TextToken { id, .. } => self.keep(&[*id]),
            Event::TextTokens { ids } => self.keep(ids),
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
    /// trajectory steps the engine stored (`denoised_steps`, zero for
    /// none), to the engine that serves next, and goes on with that
    /// engine's events.
    ///
    /// # Errors
    ///
    /// Returns the terminal event the consumer receives instead: the
    /// request does not resume, was cancelled, lost its engine more often
    /// than it may, or could not be submitted again.
    async fn migrate(&mut self, cause: MigrationCause, denoised_steps: u32) -> Result<(), Event> {
        let started = Instant::now();
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
        // A request a world change moves to a world computing other
        // numerics continues with the state it reached (`in_flight = keep`).
        let kept = match cause {
            MigrationCause::Drain => self.stream.world().and_then(|world| hold.kept(world)),
            MigrationCause::Loss => None,
        };
        let continuation = request.continuation(cause, denoised_steps, kept);
        let (stream, _scheduler) = hold
            .submit(continuation.clone())
            .await
            .map_err(|error| failed(format!("the request could not migrate: {error}")))?;
        // The next migration submits the request as submitted now.
        if let (Request::Diffusion(submitted), Some(Resumable::Diffusion(request))) =
            (&continuation, &mut self.request)
        {
            *request = submitted.clone();
        }
        // The previous engine's stream ended with its migratable event.
        self.stream = stream;
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use engine::{EngineConfig, SimEngine};
    use proto::request::{
        DiffusionSamplingParams, GenerationConstraint, GenerationRequest, ImageGenerationConfig,
        ImageParams, SamplingParams, Trajectory,
    };

    use super::*;
    use crate::engine::client::EngineClient;
    use crate::test_support::{SERVED_MODEL, sim_client, sim_core};

    /// A text request of a four-token prompt that generates `max_tokens`
    /// tokens.
    fn request(max_tokens: usize) -> GenerationRequest {
        GenerationRequest {
            request_id: Default::default(),
            prompt_token_ids: vec![1, 2, 3, 4],
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint: GenerationConstraint::UndOnly,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_images: 1,
            max_und_tokens: max_tokens,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: ImageGenerationConfig::default(),
            readout: Vec::new(),
            generated_token_ids: Vec::new(),
            adapter: 0,
        }
    }

    /// A client of one simulated scheduler that generates text until a
    /// request's limit.
    fn client() -> Arc<EngineClient> {
        let mut sim = SimEngine::new();
        sim.set_text_len(100_000);
        Arc::new(sim_client(vec![sim_core(
            EngineConfig::sim(SERVED_MODEL),
            sim,
        )]))
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
    async fn a_parked_engines_request_continues_on_the_next_engine() {
        let hold = Arc::new(RequestHold::new());
        let first = client();
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
                None => panic!("the request ended before its engine parked"),
            }
        }
        hold.serve(client());
        first.park().await.unwrap();

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
            migrated_at: None,
            open: true,
            on_finish: None,
        }
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
    /// engine stored: it is submitted again with them and its trajectory's
    /// identity, reading the stored trajectory under the next world's own
    /// latent seed, or under the parked world's when a world change keeps
    /// the state of the requests in flight. A migration that stored nothing
    /// submits the request naming what it named before.
    #[test]
    fn a_video_request_migrates_with_the_steps_its_engine_stored() {
        let video = DiffusionRequest {
            request_id: Default::default(),
            task: "t2va".to_owned(),
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
            },
            trajectory: Trajectory {
                id: 7,
                ..Trajectory::default()
            },
        };
        let trajectory = |request: Request| match request {
            Request::Diffusion(request) => request.trajectory,
            Request::Generation(_) => panic!("a video request migrates as one"),
        };
        let resumable = Resumable::Diffusion(video.clone());
        let stored = trajectory(resumable.continuation(MigrationCause::Drain, 3, None));
        assert_eq!(
            stored,
            Trajectory {
                id: 7,
                denoised_steps: 3,
                state_seed: None,
            }
        );
        let kept = Kept {
            world: 1,
            prefix_seed: None,
            latent_seed: Some(11),
        };
        assert_eq!(
            trajectory(resumable.continuation(MigrationCause::Drain, 3, Some(kept))).state_seed,
            Some(11)
        );

        let named = Resumable::Diffusion(DiffusionRequest {
            trajectory: Trajectory {
                id: 7,
                denoised_steps: 3,
                state_seed: Some(11),
            },
            ..video
        });
        assert_eq!(
            trajectory(named.continuation(MigrationCause::Loss, 0, None)),
            Trajectory {
                id: 7,
                denoised_steps: 3,
                state_seed: Some(11),
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
        })
        .unwrap();
        let mut events = RequestEvents::detached(rx);
        assert!(matches!(next(&mut events).await, Some(Event::Error { .. })));
    }
}
