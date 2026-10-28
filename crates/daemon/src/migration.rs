//! Request migration: how a token or video request moves between engines.
//!
//! An engine that parks for a world change or a rollback, sleeps, rewrites
//! its weights in place, or loses the workers that hold a request, ends the
//! request as migratable (`Event::Migratable`): a token request after every
//! token and image it generated there, a video request at a denoising step
//! boundary, with the steps of its trajectory the engine stored. The leader
//! submits every request through its [`Migrator`], which keeps the facts a
//! continuation needs and, on that event, submits the request again with
//! them through the hold to the engine that serves next:
//!
//! - a token request's generated tokens, every one its engine counted, the
//!   text it published and the control tokens it did not, such as those
//!   that open its images (`Event::ControlToken`), in order
//!   (`GenerationRequest::generated_token_ids`);
//! - each image a token request generated, with what its feedback encoders
//!   read (the decoded image product's values the engine published with the
//!   image, or its PNG when feedback reads the artifact) and the number of
//!   tokens before it (`GenerationRequest::generated_images`);
//! - a video request's stored steps (`Trajectory::denoised_steps`).
//!
//! These are facts: the client received every token and image, and the
//! engine that continues the request computes everything else from them, its
//! context included. The request's consumer reads one stream throughout
//! ([`RequestEvents`]): the new engine continues after the last token and
//! image it received, numbers its next image after them and charges the
//! request's whole usage, or continues the trajectory from the stored step,
//! and the daemon's detokenizer, stop-string matcher and response assembly
//! keep their state. An image the request had begun and not finished starts
//! again from noise, which follows from its facts; the consumer does not
//! read its start or its steps again.
//!
//! A migration caused by a drain is not counted. A request survives
//! `max_migrations` migrations caused by losses and fails on the next one.
//! A request that does not resume (a readout) fails when its engine ends it
//! as migratable.
//!
//! The random stream of a migrated request continues as it began: the
//! migrator gives a token request that names no seed one of its own when it
//! is first submitted, and every draw of a seeded request follows its seed
//! and its token position alone. An image's noise follows the request's
//! image seed and the image's index, and a latent encoding's posterior draw
//! the encoding's place among the request's encodings, which the request's
//! facts fix. A video request's draws follow its seed, and the migrator
//! gives it the identity that keys its stored trajectory (`Trajectory::id`)
//! when it is first submitted.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use proto::request::{
    DiffusionRequest, Event, FeedbackImage, FeedbackSource, GeneratedImage, GenerationRequest,
    MigrationCause, Request,
};
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
            images: HashMap::new(),
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
    /// The request `migration` submits again: a token request with the
    /// tokens it generated, a video request naming the trajectory steps
    /// its engine stored and its record of where they lie
    /// (`Migration::denoised_steps`, zero for none). `kept` is the state a
    /// world change keeps of the requests in flight (`in_flight = keep`),
    /// which the request continues with when the world that serves next
    /// computes other numerics.
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
                // It reads the pages of the state it reached and writes none,
                // since its later pages are neither world's.
                if let Some(seed) = kept.and_then(|kept| kept.prefix_seed) {
                    continuation.cache.state_seed = Some(seed);
                    continuation.cache.write = false;
                }
                tracing::info!(
                    ?cause,
                    generated = request.generated_token_ids.len(),
                    images = request.generated_images.len(),
                    "a request migrated to the engine that serves next"
                );
                Request::Generation(Box::new(continuation))
            }
            Self::Diffusion(request) => {
                let mut continuation = request.clone();
                // The engine stored the trajectory at a step boundary, keyed
                // by what it computes: the next engine reads it when it
                // computes the same, or when a world change keeps the state
                // the request reached, whatever its parallel configuration,
                // from where the engine's record says the pages lie. A
                // request the engine stored nothing of names the trajectory
                // it named before, which is the same at that step whenever
                // it is read.
                if denoised_steps > 0 {
                    continuation.trajectory.denoised_steps = denoised_steps;
                    continuation.trajectory.state_seed = kept.and_then(|kept| kept.latent_seed);
                    continuation.trajectory.stored_layout = stored_layout.clone();
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
    /// repeats one the consumer received.
    resumed: bool,
    /// The last step the consumer received of each image that began, by
    /// image identity: an image a migration restarts from noise repeats its
    /// start and the steps the consumer received, which it does not read
    /// again.
    images: HashMap<u32, u16>,
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
            images: HashMap::new(),
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
                Some(Event::ControlToken { id }) => self.keep(&[id]),
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
                    stored_layout,
                }) => {
                    self.pending = Some(Migration {
                        cause,
                        denoised_steps,
                        stored_layout,
                    })
                }
                Ok(Event::ControlToken { id }) => self.keep(&[id]),
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
    /// before: the scheduling of a resumed request, and the start and the
    /// steps it received of an image the request restarts from noise.
    fn skips(&mut self, event: &Event) -> bool {
        match event {
            Event::Scheduled { .. } if self.resumed => {
                self.resumed = false;
                true
            }
            Event::ImageBegin { image_id, .. } => {
                if self.images.contains_key(image_id) {
                    return true;
                }
                self.images.insert(*image_id, 0);
                false
            }
            Event::ImageStep { image_id, step } => {
                let received = self.images.entry(*image_id).or_default();
                if *step <= *received {
                    return true;
                }
                *received = *step;
                false
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

    /// Keeps the tokens and images of `event`, and finishes on a terminal
    /// event.
    fn observe(&mut self, event: &Event) {
        match event {
            Event::TextToken { id, .. } => self.keep(&[*id]),
            Event::TextTokens { ids } => self.keep(ids),
            Event::ImageDone {
                pixels_png_b64,
                feedback,
                ..
            } => self.keep_image(pixels_png_b64, feedback.as_ref()),
            event if event.is_terminal() => self.finish(),
            _ => {}
        }
    }

    /// Keeps an image the request generated for a migration to carry, when
    /// the request's context continues after it: what its feedback encoders
    /// read, the decoded product's values (`feedback`) or the image's PNG,
    /// after the tokens it generated so far, the one that opened the image
    /// last among them.
    fn keep_image(&mut self, png_b64: &str, feedback: Option<&Arc<proto::request::ImagePixels>>) {
        let Some(Resumable::Generation(request)) = &mut self.request else {
            return;
        };
        if !request.continues_after_image() {
            return;
        }
        let image = match (&request.image_generation.feedback_source, feedback) {
            (Some(FeedbackSource::DeviceProduct), Some(pixels)) => {
                FeedbackImage::Pixels(Arc::clone(pixels))
            }
            (Some(FeedbackSource::ArtifactProduct), _) => FeedbackImage::Png(Arc::from(png_b64)),
            // A request whose feedback reads the device product resumes
            // from that product's values alone: without them it cannot
            // resume, and its migration fails.
            _ => {
                tracing::error!("a generated image came without the values its feedback reads");
                self.request = None;
                return;
            }
        };
        request.generated_images.push(GeneratedImage {
            tokens: u32::try_from(request.generated_token_ids.len()).unwrap_or(u32::MAX),
            image,
        });
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
        let previous_world = (cause == MigrationCause::Drain)
            .then(|| self.stream.world())
            .flatten();
        // Admission resolves the current owner first. After a daemon handoff
        // that owner supplies both the engine and the continuation's state.
        let (client, kept) = hold
            .ready(previous_world)
            .await
            .map_err(|error| failed(format!("the request could not migrate: {error}")))?;
        let mut continuation = request.continuation(&migration, kept);
        if let Request::Generation(generation) = &mut continuation {
            generation.cache.resume = self.stream.cache_resume();
        }
        let (stream, _scheduler) = client
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

    use engine::{ContextLog, EngineConfig, SimEngine};
    use proto::request::{
        DiffusionSamplingParams, FeedbackNextToken, GenerationConstraint, GenerationRequest,
        ImageEncoder, ImageEncoderInput, ImageGenerationConfig, ImageParams, ImageTrigger,
        RequestId, SamplingParams, Trajectory, wire,
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
            generated_images: Vec::new(),
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
            images: HashMap::new(),
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
    /// the next world's own latent seed, or under the parked world's when a
    /// world change keeps the state of the requests in flight. A migration
    /// that stored nothing submits the request naming what it named before.
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
            world: 1,
            prefix_seed: None,
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

    /// The interleaved request of the image tests: a four-token prompt whose
    /// greedy text opens an image on `trigger`, at most three images of six
    /// denoising steps, fed back by a vision encoder whose feature write
    /// samples the next token (`sampled`, as SenseNova feeds images back), or
    /// by a latent encoder after which decoding continues with a configured
    /// token (as BAGEL does).
    fn interleaved(trigger: u32, sampled: bool) -> GenerationRequest {
        let encoder = ImageEncoderInput {
            encoder: if sampled {
                ImageEncoder::Vision
            } else {
                ImageEncoder::Latent
            },
            num_cache_tokens: None,
            max_cache_tokens: Some(16),
            height: None,
            width: None,
        };
        GenerationRequest {
            constraint: GenerationConstraint::Default,
            sampling: SamplingParams {
                temperature: 0.0,
                seed: Some(5),
                ..SamplingParams::default()
            },
            image: ImageParams {
                steps: 6,
                height: 32,
                width: 32,
                ..ImageParams::default()
            },
            max_images: 3,
            image_generation: ImageGenerationConfig {
                trigger: ImageTrigger::Token { token_id: trigger },
                requires_text_for_image: false,
                feedback_source: Some(FeedbackSource::DeviceProduct),
                feedback_encoders: vec![encoder],
                num_feedback_positions: 2,
                feedback_next_token: FeedbackNextToken::Token { token_id: 7 },
                sample_feedback_continuation: sampled,
            },
            ..request(24)
        }
    }

    /// A simulated engine of the image tests, paced or not, with the log of
    /// what each request's calls wrote into its context.
    fn image_engine(paced: bool) -> (Arc<EngineClient>, Option<engine::Pace>, ContextLog) {
        let mut sim = SimEngine::new();
        sim.set_text_len(100_000);
        let pace = paced.then(|| sim.pace());
        let log = sim.context_log();
        let client = Arc::new(sim_client(vec![sim_core(
            EngineConfig::sim(SERVED_MODEL),
            sim,
        )]));
        (client, pace, log)
    }

    /// What a consumer read of an image request, in order.
    #[derive(Debug, Default)]
    struct Read {
        tokens: Vec<u32>,
        /// Each completed image's identity and the digest of the feedback
        /// values it came with.
        images: Vec<(u32, u64)>,
        /// The text tokens read before the first image completed.
        tokens_before_image: Option<usize>,
        /// Each image start's identity, in order.
        begins: Vec<u32>,
        /// Each step, as (image, step), in order.
        steps: Vec<(u32, u16)>,
        terminal: Option<Event>,
    }

    impl Read {
        fn take(&mut self, event: Event) {
            match event {
                Event::TextToken { id, .. } => self.tokens.push(id),
                Event::TextTokens { ids } => self.tokens.extend(ids),
                Event::ImageBegin { image_id, .. } => self.begins.push(image_id),
                Event::ImageStep { image_id, step } => self.steps.push((image_id, step)),
                Event::ImageDone {
                    image_id, feedback, ..
                } => {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::hash::DefaultHasher::new();
                    feedback
                        .expect("an image fed back from the device comes with its values")
                        .values
                        .hash(&mut hasher);
                    self.tokens_before_image.get_or_insert(self.tokens.len());
                    self.images.push((image_id, hasher.finish()));
                }
                Event::ControlToken { .. } => panic!("a control token reached the consumer"),
                event if event.is_terminal() => self.terminal = Some(event),
                _ => {}
            }
        }

        /// Reads `events` to their end.
        async fn finish(&mut self, events: &mut RequestEvents) {
            while self.terminal.is_none() {
                match next(events).await {
                    Some(event) => self.take(event),
                    None => panic!("the request ended without a terminal event"),
                }
            }
        }
    }

    /// Where an image test preempts its request, by what its consumer read.
    #[derive(Debug, Clone, Copy)]
    enum Phase {
        /// After its first text token, before its first image.
        BeforeImage,
        /// While its first image denoises: after the image's second step.
        Denoising,
        /// Right after its first image.
        AfterImage,
        /// After its first image and text that follows it, before its second
        /// image.
        BetweenImages,
    }

    impl Phase {
        fn reached(self, read: &Read) -> bool {
            match self {
                Self::BeforeImage => !read.tokens.is_empty(),
                Self::Denoising => read.steps.contains(&(1, 2)),
                Self::AfterImage => !read.images.is_empty(),
                Self::BetweenImages => read
                    .tokens_before_image
                    .is_some_and(|before| read.tokens.len() > before),
            }
        }
    }

    /// Steps `pace`'s engine until the consumer of `events` read what
    /// `phase` names, reading every event on the way into `read`.
    async fn advance(
        events: &mut RequestEvents,
        pace: &engine::Pace,
        phase: Phase,
        read: &mut Read,
    ) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !phase.reached(read) {
            assert!(
                Instant::now() < deadline,
                "{phase:?} was not reached: {read:?}"
            );
            pace.step();
            // The batch's events arrive as its result resolves.
            while let Ok(Some(event)) =
                tokio::time::timeout(Duration::from_millis(30), events.recv()).await
            {
                read.take(event);
                if phase.reached(read) {
                    break;
                }
            }
        }
    }

    /// How an image test's engine preempts its request.
    #[derive(Debug, Clone, Copy)]
    enum Preemption {
        /// A park for a world change: the request resumes on another engine.
        Park,
        /// A sleep: the request resumes on the same engine once it wakes.
        Sleep,
        /// An in-place weight update whose requests in flight recompute:
        /// the request resumes on the same engine, on its new weights.
        Recompute,
    }

    /// An interleaved request preempted at `phase` by `preemption` resumes
    /// where it was and completes as one stream: each image and token once,
    /// images numbered in order, each image's start and steps read once,
    /// and a usage that counts every token and image. The engine that
    /// continues it first writes into its context exactly what the
    /// preempted engine wrote, its images included; an image the request
    /// had begun starts again from noise, and every image is the one the
    /// uninterrupted request generates under that image index.
    async fn an_image_request_resumes(phase: Phase, sampled: bool, preemption: Preemption) {
        let (whole, _, whole_log) = image_engine(false);
        let (first, pace, first_log) = image_engine(true);
        let pace = pace.expect("a paced engine");
        // Both engines give the request the identity that keys the
        // simulator's tokens and images.
        let id = RequestId(first.next_request_id());
        assert_eq!(whole.next_request_id(), id.0);
        let request = interleaved(engine::sim_text_token(id, 2), sampled);

        let hold = Arc::new(RequestHold::new());
        hold.serve(Arc::clone(&whole));
        let (mut events, _) = Migrator::new(hold, 0)
            .submit(Request::Generation(Box::new(request.clone())))
            .await
            .unwrap();
        let mut expected = Read::default();
        expected.finish(&mut events).await;

        let hold = Arc::new(RequestHold::new());
        hold.serve(Arc::clone(&first));
        let (mut events, _) = Migrator::new(Arc::clone(&hold), 0)
            .submit(Request::Generation(Box::new(request)))
            .await
            .unwrap();
        let mut read = Read::default();
        advance(&mut events, &pace, phase, &mut read).await;
        let tokens_read = read.tokens.clone();

        // The engine marks the request to leave at its next step boundary,
        // then runs freely: the call in flight completes, and the request
        // leaves.
        let (resumed_log, resumed_id) = match preemption {
            Preemption::Park => {
                let (second, _, second_log) = image_engine(false);
                let resumed_id = RequestId(second.next_request_id());
                hold.serve(second);
                let parking = {
                    let first = Arc::clone(&first);
                    tokio::spawn(async move { first.park().await })
                };
                tokio::time::sleep(Duration::from_millis(100)).await;
                drop(pace);
                parking.await.unwrap().unwrap();
                (second_log, Some(resumed_id))
            }
            Preemption::Recompute => {
                let updating = {
                    let first = Arc::clone(&first);
                    tokio::spawn(async move {
                        first
                            .update_weights(
                                "/checkpoints/next",
                                crate::leader::WeightsInFlight::Recompute,
                            )
                            .await
                    })
                };
                tokio::time::sleep(Duration::from_millis(100)).await;
                drop(pace);
                updating.await.unwrap().unwrap();
                (first_log.clone(), None)
            }
            Preemption::Sleep => {
                let sleeping = {
                    let first = Arc::clone(&first);
                    tokio::spawn(async move { first.resize_cache(wire::CacheChange::Sleep).await })
                };
                tokio::time::sleep(Duration::from_millis(100)).await;
                drop(pace);
                sleeping.await.unwrap().unwrap();
                first.resize_cache(wire::CacheChange::Wake).await.unwrap();
                (first_log.clone(), None)
            }
        };
        read.finish(&mut events).await;
        // The engine numbers the request it admits again anew.
        let resumed_id = resumed_id.unwrap_or_else(|| {
            *resumed_log
                .requests()
                .iter()
                .find(|other| **other != id)
                .expect("the request was admitted again")
        });

        let Some(Event::Finished {
            completion_tokens,
            images,
            ..
        }) = read.terminal
        else {
            panic!("the migrated request did not finish: {:?}", read.terminal);
        };
        let ids: Vec<u32> = read.images.iter().map(|(image, _)| *image).collect();
        assert_eq!(ids, (1..=ids.len() as u32).collect::<Vec<_>>(), "{read:?}");
        assert!(!ids.is_empty());
        assert_eq!(read.begins, ids, "each image starts once");
        for image in &ids {
            let steps: Vec<u16> = read
                .steps
                .iter()
                .filter(|(of, _)| of == image)
                .map(|(_, step)| *step)
                .collect();
            assert_eq!(steps, (1..=6).collect::<Vec<_>>(), "image {image}");
        }
        assert_eq!(images, ids.len());
        // Every token the engines counted reached the consumer once, as
        // text, or as the token that opened an image, which it does not
        // read.
        assert_eq!(completion_tokens, read.tokens.len() + ids.len());
        // Before the preemption the request generated what the uninterrupted
        // one did, and the engine that continues it rebuilt that context.
        assert!(
            expected.tokens.starts_with(&tokens_read),
            "{tokens_read:?} {expected:?}"
        );
        let preempted = first_log.admissions(id).remove(0);
        assert!(whole_log.admissions(id)[0].starts_with(&preempted));
        let resumed = resumed_log.admissions(resumed_id).remove(0);
        assert!(
            resumed.starts_with(&preempted),
            "the continuation wrote {resumed:#?}, the preempted request {preempted:#?}"
        );
        // An image follows the request and its index: under the same request
        // identity, the migrated request's images are the uninterrupted
        // one's.
        if resumed_id == id {
            for (image, values) in &read.images {
                if let Some((_, expected)) = expected.images.get(*image as usize - 1) {
                    assert_eq!(values, expected, "image {image}");
                }
            }
        }
    }

    #[tokio::test]
    async fn a_sampled_feedback_request_parked_before_its_first_image_resumes() {
        an_image_request_resumes(Phase::BeforeImage, true, Preemption::Park).await;
    }

    #[tokio::test]
    async fn a_sampled_feedback_request_parked_while_denoising_resumes() {
        an_image_request_resumes(Phase::Denoising, true, Preemption::Park).await;
    }

    #[tokio::test]
    async fn a_sampled_feedback_request_parked_after_an_image_resumes() {
        an_image_request_resumes(Phase::AfterImage, true, Preemption::Park).await;
    }

    #[tokio::test]
    async fn a_sampled_feedback_request_parked_between_images_resumes() {
        an_image_request_resumes(Phase::BetweenImages, true, Preemption::Park).await;
    }

    #[tokio::test]
    async fn a_token_feedback_request_parked_before_its_first_image_resumes() {
        an_image_request_resumes(Phase::BeforeImage, false, Preemption::Park).await;
    }

    #[tokio::test]
    async fn a_token_feedback_request_parked_while_denoising_resumes() {
        an_image_request_resumes(Phase::Denoising, false, Preemption::Park).await;
    }

    #[tokio::test]
    async fn a_token_feedback_request_parked_after_an_image_resumes() {
        an_image_request_resumes(Phase::AfterImage, false, Preemption::Park).await;
    }

    #[tokio::test]
    async fn a_token_feedback_request_parked_between_images_resumes() {
        an_image_request_resumes(Phase::BetweenImages, false, Preemption::Park).await;
    }

    #[tokio::test]
    async fn an_image_request_resumes_on_its_engine_after_a_sleep_while_denoising() {
        an_image_request_resumes(Phase::Denoising, true, Preemption::Sleep).await;
    }

    #[tokio::test]
    async fn an_image_request_resumes_on_its_engine_after_a_sleep_between_images() {
        an_image_request_resumes(Phase::BetweenImages, false, Preemption::Sleep).await;
    }

    #[tokio::test]
    async fn an_image_request_resumes_after_a_weight_update_that_recomputes() {
        an_image_request_resumes(Phase::AfterImage, true, Preemption::Recompute).await;
    }

    /// The migrator keeps what a token request generated as facts: every
    /// token in order, the control tokens its consumer never reads among
    /// them, and each image with the values its feedback reads, after the
    /// tokens before it; it submits the request again with them. A repeated
    /// image start and the steps the consumer read do not reach it again.
    #[tokio::test]
    async fn the_migrator_keeps_tokens_and_images_and_hides_repeats() {
        let pixels = Arc::new(proto::request::ImagePixels {
            height: 32,
            width: 32,
            range: proto::request::PixelRange::SignedUnit,
            values: vec![3; 32 * 32 * 6],
        });
        let (tx, rx) = mpsc::unbounded_channel();
        for event in [
            Event::TextToken {
                id: 11,
                logprob: None,
            },
            Event::ControlToken { id: 99 },
            Event::ImageBegin {
                image_id: 1,
                height: 32,
                width: 32,
                steps: 6,
            },
            Event::ImageStep {
                image_id: 1,
                step: 1,
            },
            Event::ImageDone {
                image_id: 1,
                height: 32,
                width: 32,
                bytes: 3,
                pixels_png_b64: "cG5n".to_owned(),
                feedback: Some(Arc::clone(&pixels)),
            },
            Event::TextTokens { ids: vec![12, 13] },
            Event::ControlToken { id: 99 },
            Event::ImageBegin {
                image_id: 2,
                height: 32,
                width: 32,
                steps: 6,
            },
            Event::ImageStep {
                image_id: 2,
                step: 1,
            },
            // An image restarted from noise repeats its start and steps.
            Event::ImageBegin {
                image_id: 2,
                height: 32,
                width: 32,
                steps: 6,
            },
            Event::ImageStep {
                image_id: 2,
                step: 1,
            },
            Event::ImageStep {
                image_id: 2,
                step: 2,
            },
        ] {
            tx.send(event).unwrap();
        }
        drop(tx);
        let hold = Arc::new(RequestHold::new());
        let mut events = RequestEvents {
            stream: EventStream::detached(rx),
            request: Some(Resumable::Generation(Box::new(interleaved(99, true)))),
            hold: Some(hold),
            max_migrations: 0,
            losses: 0,
            pending: None,
            resumed: false,
            images: HashMap::new(),
            migrated_at: None,
            open: true,
            on_finish: None,
        };
        let mut read = Vec::new();
        while let Some(event) = next(&mut events).await {
            read.push(event);
        }
        let shown: Vec<String> = read
            .iter()
            .map(|event| match event {
                Event::TextToken { id, .. } => format!("text {id}"),
                Event::TextTokens { ids } => format!("text {ids:?}"),
                Event::ImageBegin { image_id, .. } => format!("begin {image_id}"),
                Event::ImageStep { image_id, step } => format!("step {image_id}.{step}"),
                Event::ImageDone { image_id, .. } => format!("done {image_id}"),
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(
            shown,
            [
                "text 11",
                "begin 1",
                "step 1.1",
                "done 1",
                "text [12, 13]",
                "begin 2",
                "step 2.1",
                "step 2.2"
            ]
        );

        // The facts carried at the end of the stream: a finished stream
        // carries none, so they are read as the events arrive.
        let (tx, rx) = mpsc::unbounded_channel();
        let mut events = RequestEvents {
            stream: EventStream::detached(rx),
            request: Some(Resumable::Generation(Box::new(interleaved(99, true)))),
            hold: None,
            max_migrations: 0,
            losses: 0,
            pending: None,
            resumed: false,
            images: HashMap::new(),
            migrated_at: None,
            open: true,
            on_finish: None,
        };
        for event in [
            Event::TextToken {
                id: 11,
                logprob: None,
            },
            Event::ControlToken { id: 99 },
            Event::ImageDone {
                image_id: 1,
                height: 32,
                width: 32,
                bytes: 3,
                pixels_png_b64: "cG5n".to_owned(),
                feedback: Some(Arc::clone(&pixels)),
            },
            Event::TextToken {
                id: 12,
                logprob: None,
            },
        ] {
            tx.send(event).unwrap();
        }
        for _ in 0..3 {
            next(&mut events).await.unwrap();
        }
        let Some(Resumable::Generation(kept)) = &events.request else {
            panic!("the request resumes");
        };
        assert_eq!(kept.generated_token_ids, [11, 99, 12]);
        assert_eq!(
            kept.generated_images,
            [GeneratedImage {
                tokens: 2,
                image: FeedbackImage::Pixels(pixels),
            }]
        );
        let Request::Generation(continuation) = events.request.as_ref().unwrap().continuation(
            &Migration {
                cause: MigrationCause::Drain,
                denoised_steps: 0,
                stored_layout: Vec::new(),
            },
            None,
        ) else {
            panic!("a token request continues as one");
        };
        assert_eq!(continuation.generated_token_ids, [11, 99, 12]);
        assert_eq!(continuation.generated_images.len(), 1);
        assert_eq!(continuation.validate(), Ok(()));
    }
}
