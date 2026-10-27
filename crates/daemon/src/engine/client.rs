//! The daemon's client of the request interface: one engine process, the
//! requests submitted to it and their events.
//!
//! An [`EngineClient`] holds the connection to an engine process that has
//! reported itself ready. Its reader thread files every frame the engine
//! sends: a request's acceptance and events go to the request's
//! [`EventStream`], an artifact's descriptor is mapped into its event, each
//! scheduler's statistics go to the client's statistics sink, and the
//! engine's state changes to the client's state. When the connection closes,
//! every request still in flight ends with an error event.

use std::collections::HashMap;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use proto::report::SchedulerStats;
use proto::request::wire::{self, from_engine, to_engine};
use proto::request::{EVENT_WINDOW, Event, GenerationLimits, Request, RequestId};
use proto::{Connection, MappedMedia};
use tokio::sync::{mpsc, oneshot, watch};

/// How long a read of the engine's frames waits before it rechecks whether
/// the client closed.
const READ_INTERVAL: Duration = Duration::from_millis(200);

/// What a ready engine reported it serves.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineInfo {
    /// The limits every request is checked against.
    pub limits: GenerationLimits,
    /// The video denoiser as its model family describes it
    /// (`models::video::DenoiserInfo`, JSON); empty when the engine
    /// generates no video.
    pub video_denoiser: Vec<u8>,
    /// Condition units one latent encoding call covers at once; zero when no
    /// worker encodes latents.
    pub latent_encoding_units: u32,
    /// The engine's schedulers, one per DP rank.
    pub schedulers: u32,
    /// Each worker group's KV cache segments as the engine started with
    /// them; a world succeeding this one maps those its engine describes
    /// then ([`EngineClient::describe_cache`]).
    pub segments: Vec<wire::CacheSegments>,
    /// The batch the workers were started for, within which the schedulers
    /// take scheduling limits in place; `None` from an engine that does not
    /// state it.
    pub capacity: Option<wire::BatchCapacity>,
    /// The root of the prefix hash chains of the pages the engine's workers
    /// compute: equal between worlds whose models and numerics agree.
    pub prefix_seed: u64,
}

impl From<wire::EngineInfo> for EngineInfo {
    fn from(info: wire::EngineInfo) -> Self {
        Self {
            limits: info.limits.map(GenerationLimits::from).unwrap_or_default(),
            video_denoiser: info.video_denoiser,
            latent_encoding_units: info.latent_encoding_units,
            schedulers: info.schedulers,
            segments: info.segments,
            capacity: info.capacity,
            prefix_seed: info.prefix_seed,
        }
    }
}

/// Locks `mutex`, keeping the state a panicking holder left: every build
/// profile aborts on panic, so only test builds reach the recovery.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// What an engine is, as its client knows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineState {
    /// It succeeds the serving engine and is prepared beside it: its
    /// workers are loaded and captured, and it serves once promoted.
    Standby,
    /// It serves requests.
    Ready,
    /// It preempted every request and executes nothing, keeping its memory
    /// and prefix cache index until it resumes or shuts down.
    Parked,
    /// It serves nothing more: its workers failed.
    Failed(String),
    /// Its connection closed.
    Closed,
}

/// Why a request could not be submitted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SubmitError {
    /// The engine serves no requests.
    #[error("the engine is unavailable: {0}")]
    Unavailable(String),
}

/// Receives each scheduler's statistics: the scheduler's index and its
/// snapshot of the last interval.
pub type StatsSink = Box<dyn Fn(u64, u32, SchedulerStats) + Send + Sync>;

/// Why the consumer of a request's events closes them before the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamCancelCause {
    /// The consumer dropped or explicitly closed the event stream.
    #[default]
    DroppedStream,
    /// Incremental decoding matched a configured stop string.
    StopStringMatched,
}

/// A request in flight, as the reader files its frames.
struct Pending {
    events: mpsc::UnboundedSender<Event>,
    /// Answered with the scheduler serving the request once the engine
    /// accepts it, or with `None` when the request ends first.
    accepted: Option<oneshot::Sender<Option<u32>>>,
}

/// The connection's state shared between the client, its streams and its
/// reader.
struct Shared {
    writer: Mutex<Connection>,
    requests: Mutex<HashMap<RequestId, Pending>>,
    state: watch::Sender<EngineState>,
    next_id: AtomicU64,
    /// The hooks and configure commands awaiting their answer, by
    /// identity: a hook's answer is each worker's result, a configure's
    /// none.
    answers: Mutex<HashMap<u64, Answer>>,
    /// The world the engine executes, which its statistics are recorded
    /// under; 0 for an engine this daemon did not start.
    world: AtomicU64,
}

/// Where an engine command's answer goes: each worker's result, or why the
/// command failed.
type Answer = oneshot::Sender<Result<Answered, String>>;

/// What the engine answered a command with.
enum Answered {
    /// Each worker's JSON result (`Hooked`), or nothing (`Configured`).
    Results(Vec<String>),
    /// Each worker group's prefix cache index (`Parked`).
    Index(Vec<wire::GroupIndex>),
    /// Each worker group's KV cache segments (`CacheDescribed`).
    Segments(Vec<wire::CacheSegments>),
}

impl Shared {
    /// Sends one frame to the engine; a closed connection is the reader's to
    /// report.
    fn send(&self, kind: to_engine::Kind) {
        let message = wire::ToEngine { kind: Some(kind) };
        let mut writer = lock(&self.writer);
        if let Err(error) = writer.send(&message) {
            tracing::debug!(%error, "the engine's connection refused a frame");
        }
    }

    /// Sends a command for one request.
    fn command(&self, request: RequestId, kind: wire::command::Kind) {
        self.send(to_engine::Kind::Command(wire::Command {
            request: request.0,
            kind: Some(kind),
        }));
    }
}

/// The client of one engine process that serves requests.
pub struct EngineClient {
    shared: Arc<Shared>,
    info: EngineInfo,
    reader: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// The daemon connection the engine's process lives under, which reports
    /// its exit; `None` for an engine whose process this client did not
    /// start, or once handed to a successor daemon ([`Self::take_control`]).
    control: Mutex<Option<Connection>>,
    /// Set once the client let go of the engine without shutting it down
    /// ([`Self::detach`]).
    detached: std::sync::atomic::AtomicBool,
    /// The identity this daemon's engine control gave the engine; `None`
    /// for an engine it did not start.
    engine: Option<u64>,
}

impl EngineClient {
    /// A client of the engine at the other end of `stream`, which has
    /// reported itself ready with `info`; `stats` receives its schedulers'
    /// statistics.
    ///
    /// # Errors
    ///
    /// Fails when the connection cannot be shared between the client and
    /// its reader.
    pub fn over(stream: UnixStream, info: EngineInfo, stats: StatsSink) -> anyhow::Result<Self> {
        let reader = Connection::from_unix(stream.try_clone()?)?;
        let writer = Connection::from_unix(stream)?;
        Self::new(reader, writer, info, EngineState::Ready, stats, None)
    }

    /// A client of the engine read through `reader` and written through
    /// `writer`, two ends of one connection over which the engine reported
    /// itself in `state` (ready or standing by) with `info`; frames `reader`
    /// already holds are filed first. `stats` receives the schedulers'
    /// statistics. `control` is the daemon connection the engine's process
    /// lives under, which the client keeps; the process ends with it.
    ///
    /// # Errors
    ///
    /// Fails when the reader thread cannot start.
    pub fn new(
        reader: Connection,
        writer: Connection,
        info: EngineInfo,
        state: EngineState,
        stats: StatsSink,
        control: Option<Connection>,
    ) -> anyhow::Result<Self> {
        let (state, _) = watch::channel(state);
        let shared = Arc::new(Shared {
            writer: Mutex::new(writer),
            requests: Mutex::default(),
            state,
            next_id: AtomicU64::new(1),
            answers: Mutex::default(),
            world: AtomicU64::new(0),
        });
        let thread = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("engine-reader".into())
                .spawn(move || read_frames(reader, &shared, &stats))?
        };
        Ok(Self {
            shared,
            info,
            reader: Mutex::new(Some(thread)),
            control: Mutex::new(control),
            detached: std::sync::atomic::AtomicBool::new(false),
            engine: None,
        })
    }

    /// The client of engine `engine`, as this daemon's engine control
    /// numbers its engines.
    #[must_use]
    pub fn numbered(mut self, engine: u64) -> Self {
        self.engine = Some(engine);
        self
    }

    /// The client of an engine that executes world `world`.
    #[must_use]
    pub fn in_world(self, world: u64) -> Self {
        self.shared.world.store(world, Ordering::Relaxed);
        self
    }

    /// The engine's identity, when this daemon's engine control started or
    /// adopted it.
    pub fn engine(&self) -> Option<u64> {
        self.engine
    }

    /// The world the engine executes; 0 for an engine this daemon did not
    /// start.
    pub fn world(&self) -> u64 {
        self.shared.world.load(Ordering::Relaxed)
    }

    /// What the engine reported it serves.
    pub fn info(&self) -> &EngineInfo {
        &self.info
    }

    /// The engine's state as of now.
    pub fn state(&self) -> EngineState {
        self.shared.state.borrow().clone()
    }

    /// The engine's state, observed as it changes.
    pub fn states(&self) -> watch::Receiver<EngineState> {
        self.shared.state.subscribe()
    }

    /// Submits `request` under an identity of the client's choosing and
    /// returns its events and the scheduler serving it; the scheduler is
    /// `None` when the engine ended the request at submission, whose
    /// terminal event the stream holds.
    ///
    /// # Errors
    ///
    /// Fails when the engine serves no requests.
    pub async fn submit(
        &self,
        mut request: Request,
    ) -> Result<(EventStream, Option<u32>), SubmitError> {
        if let EngineState::Failed(reason) = self.state() {
            return Err(SubmitError::Unavailable(reason));
        }
        if self.state() == EngineState::Closed {
            return Err(SubmitError::Unavailable("its process ended".to_owned()));
        }
        let id = RequestId(self.shared.next_id.fetch_add(1, Ordering::Relaxed));
        let awaits_decoder = match &mut request {
            Request::Generation(generation) => {
                generation.request_id = id;
                !generation.stop_strings.is_empty()
            }
            Request::Diffusion(diffusion) => {
                diffusion.request_id = id;
                false
            }
        };
        let (events, receiver) = mpsc::unbounded_channel();
        let (accepted, acceptance) = oneshot::channel();
        lock(&self.shared.requests).insert(
            id,
            Pending {
                events,
                accepted: Some(accepted),
            },
        );
        // The stream exists before the acceptance arrives, so that dropping
        // the submission while it waits cancels the request.
        let stream = EventStream {
            id,
            events: receiver,
            shared: Some(Arc::clone(&self.shared)),
            awaits_decoder,
            text_tokens: 0,
            acknowledged: 0,
            consumed: 0,
            reported: 0,
            open: true,
            on_finish: None,
        };
        self.shared.send(to_engine::Kind::Submit(wire::Submit {
            requests: vec![request.to_wire()],
        }));
        // A closed connection drops the acceptance with the request, whose
        // stream then holds the connection's error event.
        let scheduler = acceptance.await.unwrap_or(None);
        Ok((stream, scheduler))
    }

    /// Promotes a prepared engine, which succeeds a parked one: its workers
    /// adopt the predecessor's cache flags, and it imports `index`, the
    /// predecessor's prefix cache index, and serves the requests the
    /// predecessor hands over, which `recompute` their state from their
    /// tokens or not, and for which its cache `grows` into the
    /// predecessor's memory at once or only at its commit. Waits until the
    /// engine reports that it serves; an engine that serves already is left
    /// as it is.
    ///
    /// # Errors
    ///
    /// Fails when the engine reports that its promotion failed, as when its
    /// cache cannot hold those requests, or its connection closes first.
    pub async fn promote(
        &self,
        index: Vec<wire::GroupIndex>,
        recompute: bool,
        grows: bool,
    ) -> Result<(), SubmitError> {
        let mut states = self.states();
        if *states.borrow_and_update() == EngineState::Standby {
            self.shared
                .send(to_engine::Kind::Engine(wire::EngineCommand {
                    kind: Some(wire::engine_command::Kind::Promote(wire::Promote {
                        index,
                        recompute,
                        grows,
                    })),
                }));
        }
        loop {
            match states.borrow_and_update().clone() {
                EngineState::Ready => return Ok(()),
                EngineState::Failed(reason) => return Err(SubmitError::Unavailable(reason)),
                EngineState::Closed | EngineState::Parked => {
                    return Err(SubmitError::Unavailable(
                        "the engine ended before it served".to_owned(),
                    ));
                }
                EngineState::Standby => {}
            }
            if states.changed().await.is_err() {
                return Err(SubmitError::Unavailable(
                    "the engine's client closed".to_owned(),
                ));
            }
        }
    }

    /// Waits at most `timeout` for the engine's process to exit, as the
    /// daemon that started it reports, and returns its exit status; `None`
    /// for an engine whose process this client did not start.
    ///
    /// # Errors
    ///
    /// Fails when the daemon connection fails or the process does not exit
    /// in time.
    pub fn await_exit(&self, timeout: Duration) -> anyhow::Result<Option<String>> {
        use proto::process::{Downstream, Event as ProcessEvent, downstream, event};

        let mut control = lock(&self.control);
        let Some(control) = control.as_mut() else {
            return Ok(None);
        };
        let deadline = std::time::Instant::now() + timeout;
        loop {
            while let Some(message) = control.try_recv::<Downstream>()? {
                if let Some(downstream::Kind::Event(ProcessEvent {
                    kind: Some(event::Kind::Exit(exit)),
                })) = message.kind
                {
                    return Ok(Some(exit.status));
                }
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            anyhow::ensure!(
                !remaining.is_zero(),
                "the engine process did not exit within {timeout:?}"
            );
            control.wait(remaining.min(READ_INTERVAL))?;
        }
    }

    /// The requests submitted to the engine that have not ended.
    pub fn in_flight(&self) -> usize {
        lock(&self.shared.requests).len()
    }

    /// Sends `command`, built with the identity its answer matches, and
    /// awaits the answer.
    async fn ask(
        &self,
        command: impl FnOnce(u64) -> wire::engine_command::Kind,
    ) -> Result<Answered, String> {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let (answer, answered) = oneshot::channel();
        lock(&self.shared.answers).insert(id, answer);
        self.shared
            .send(to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(command(id)),
            }));
        answered
            .await
            .unwrap_or_else(|_| Err("the engine ended before it answered".to_owned()))
    }

    /// Parks the engine: at its next step boundary it ends every token
    /// request that resumes from its generated tokens as migratable, lets
    /// the others finish, and executes nothing more, keeping its memory and
    /// prefix cache index. Returns that index once it parked.
    ///
    /// # Errors
    ///
    /// Fails with the reason a scheduler did not park, or when the engine
    /// ends first.
    pub async fn park(&self) -> Result<Vec<wire::GroupIndex>, String> {
        match self
            .ask(|id| wire::engine_command::Kind::Park(wire::Park { id }))
            .await?
        {
            Answered::Index(index) => Ok(index),
            _ => Err("the engine answered a park with something else".to_owned()),
        }
    }

    /// Resumes a parked engine, ending the memory of the worlds `retire`
    /// names on its nodes first: the successor a rollback discards.
    ///
    /// # Errors
    ///
    /// Fails with the reason a scheduler did not resume, or when the engine
    /// ends first.
    pub async fn resume(&self, retire: Vec<u64>) -> Result<(), String> {
        self.ask(|id| wire::engine_command::Kind::Resume(wire::Resume { id, retire }))
            .await
            .map(drop)
    }

    /// Commits the engine's world: ends the memory of the worlds `retire`
    /// names (its predecessor, whose processes are gone) on its nodes, and
    /// its workers take the predecessor's memory they mapped read-only.
    /// Returns what each worker reported.
    ///
    /// # Errors
    ///
    /// Fails with the reason a worker did not commit, or when the engine
    /// ends first.
    pub async fn commit(&self, retire: Vec<u64>) -> Result<Vec<String>, String> {
        match self
            .ask(|id| wire::engine_command::Kind::Commit(wire::Commit { id, retire }))
            .await?
        {
            Answered::Results(results) => Ok(results),
            _ => Err("the engine answered a commit with something else".to_owned()),
        }
    }

    /// Each worker group's KV cache segments as the engine holds them now,
    /// which a world succeeding it maps.
    ///
    /// # Errors
    ///
    /// Fails with the reason a scheduler did not describe its cache, or
    /// when the engine ends first.
    pub async fn describe_cache(&self) -> Result<Vec<wire::CacheSegments>, String> {
        match self
            .ask(|id| wire::engine_command::Kind::DescribeCache(wire::DescribeCache { id }))
            .await?
        {
            Answered::Segments(segments) => Ok(segments),
            _ => Err("the engine answered a cache description with something else".to_owned()),
        }
    }

    /// Runs the worker hook `name` with the JSON `payload` on every worker
    /// once the engine's batches in flight complete, and returns each
    /// worker's JSON result.
    ///
    /// # Errors
    ///
    /// Fails with the reason a worker gave, or when the engine ends first.
    pub async fn hook(&self, name: &str, payload: &str) -> Result<Vec<String>, String> {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let (answer, answered) = oneshot::channel();
        lock(&self.shared.answers).insert(id, answer);
        self.shared
            .send(to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::Hook(wire::Hook {
                    id,
                    name: name.to_owned(),
                    payload: payload.to_owned(),
                })),
            }));
        answered
            .await
            .unwrap_or_else(|_| Err("the engine ended before it answered".to_owned()))
            .map(Answered::results)
    }

    /// Gives every scheduler of the engine the scheduling limits and policy
    /// `scheduling` states, from its next step on.
    ///
    /// # Errors
    ///
    /// Fails with the reason a scheduler refused them (a limit beyond the
    /// batch the workers were started for), or when the engine ends first.
    pub async fn configure(&self, scheduling: wire::Scheduling) -> Result<(), String> {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let (answer, answered) = oneshot::channel();
        lock(&self.shared.answers).insert(id, answer);
        self.shared
            .send(to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::Configure(wire::Configure {
                    id,
                    scheduling: Some(scheduling),
                })),
            }));
        answered
            .await
            .unwrap_or_else(|_| Err("the engine ended before it answered".to_owned()))
            .map(drop)
    }

    /// Gives back or restores the device memory of the engine's KV cache
    /// (`wire::CacheChange`), and returns what each worker reported.
    ///
    /// # Errors
    ///
    /// Fails with the reason a scheduler could not make the change, or when
    /// the engine ends first.
    pub async fn resize_cache(&self, change: wire::CacheChange) -> Result<Vec<String>, String> {
        self.resize_cache_within(change, None).await
    }

    /// Gives back the tail of the engine's KV cache as
    /// [`Self::resize_cache`] does for a shrink, recomputing the requests
    /// that still hold it once `preempt_after` passed, when given.
    ///
    /// # Errors
    ///
    /// Fails as [`Self::resize_cache`] does.
    pub async fn resize_cache_within(
        &self,
        change: wire::CacheChange,
        preempt_after: Option<Duration>,
    ) -> Result<Vec<String>, String> {
        self.ask(|id| {
            wire::engine_command::Kind::ResizeCache(wire::ResizeCache {
                id,
                change: change as i32,
                preempt_after_ms: preempt_after.map(|after| after.as_millis() as u64),
            })
        })
        .await
        .map(Answered::results)
    }

    /// Has every worker rewrite its weights in place from the checkpoint at
    /// `model_path` once no request runs, and returns what each reported.
    ///
    /// # Errors
    ///
    /// Fails with the reason a worker could not update, or when the engine
    /// ends first.
    pub async fn update_weights(&self, model_path: &str) -> Result<Vec<String>, String> {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let (answer, answered) = oneshot::channel();
        lock(&self.shared.answers).insert(id, answer);
        self.shared
            .send(to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::UpdateWeights(
                    wire::UpdateWeights {
                        id,
                        model_path: model_path.to_owned(),
                    },
                )),
            }));
        answered
            .await
            .unwrap_or_else(|_| Err("the engine ended before it answered".to_owned()))
            .map(Answered::results)
    }

    /// Commands the engine to revoke every cached prefix page; requests in
    /// flight keep their pages.
    pub fn flush_cache(&self) {
        self.shared
            .send(to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::FlushCache(wire::FlushCache {})),
            }));
    }

    /// Commands the engine to connect to the daemon whose request socket is
    /// `requests` beside this one (a hot restart's successor), which it then
    /// serves as well.
    pub fn attach(&self, requests: &str) {
        self.shared
            .send(to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::Attach(wire::Attach {
                    requests: requests.to_owned(),
                })),
            }));
    }

    /// The daemon connection the engine's process lives under, for a
    /// successor daemon to keep; this client reports no exit afterwards.
    pub fn take_control(&self) -> Option<Connection> {
        lock(&self.control).take()
    }

    /// The identity the next request submitted takes.
    pub fn next_request_id(&self) -> u64 {
        self.shared.next_id.load(Ordering::Relaxed)
    }

    /// Numbers this client's requests from `first` on, so that they never
    /// take the identity of another daemon's request to the same engine.
    pub fn number_requests_from(&self, first: u64) {
        self.shared.next_id.fetch_max(first, Ordering::Relaxed);
    }

    /// Closes the connection without shutting the engine down, as the daemon
    /// a hot restart replaced does once its requests ended: the engine goes
    /// on serving its other daemons. Waits until the reader ended.
    pub fn detach(&self) {
        self.detached.store(true, Ordering::Release);
        lock(&self.shared.writer).close();
        let reader = lock(&self.reader).take();
        if let Some(reader) = reader
            && reader.join().is_err()
        {
            tracing::warn!("the engine's reader panicked");
        }
    }

    /// Commands the engine to shut down, finishing every request in flight
    /// as aborted, and waits until its connection closes.
    pub fn shutdown(&self) {
        if self.detached.load(Ordering::Acquire) {
            return;
        }
        self.shared
            .send(to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::Shutdown(wire::Shutdown {})),
            }));
        let reader = lock(&self.reader).take();
        if let Some(reader) = reader
            && reader.join().is_err()
        {
            tracing::warn!("the engine's reader panicked");
        }
    }
}

impl Drop for EngineClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Answered {
    /// The results a `Hooked` answer carries; none for any other answer.
    fn results(self) -> Vec<String> {
        match self {
            Self::Results(results) => results,
            Self::Index(_) | Self::Segments(_) => Vec::new(),
        }
    }
}

/// Files the engine's frames until its connection closes, then ends every
/// request still in flight.
fn read_frames(mut reader: Connection, shared: &Shared, stats: &StatsSink) {
    loop {
        let message = match reader.try_recv::<wire::FromEngine>() {
            Ok(Some(message)) => message,
            Ok(None) => {
                if reader.wait(READ_INTERVAL).is_err() {
                    break;
                }
                continue;
            }
            Err(proto::Error::Closed) => break,
            Err(error) => {
                tracing::error!(%error, "the engine's connection failed");
                break;
            }
        };
        match message.kind {
            Some(from_engine::Kind::Event(event)) => {
                // An artifact's bytes come as the descriptor beside its
                // frame.
                let media = match &event.kind {
                    Some(wire::event::Kind::Artifact(artifact)) => reader
                        .take_descriptor()
                        .and_then(|descriptor| {
                            MappedMedia::map(descriptor, artifact.bytes)
                                .inspect_err(|error| {
                                    tracing::error!(%error, "an artifact does not map");
                                })
                                .ok()
                        })
                        .map(|media| Arc::new(media) as Arc<dyn proto::request::ArtifactMedia>),
                    _ => None,
                };
                match Event::from_wire(event, media) {
                    Ok((id, event)) => file(shared, id, event),
                    Err(error) => tracing::error!(%error, "the engine sent an unreadable event"),
                }
            }
            Some(from_engine::Kind::Accepted(accepted)) => {
                let mut requests = lock(&shared.requests);
                if let Some(answer) = requests
                    .get_mut(&RequestId(accepted.request))
                    .and_then(|pending| pending.accepted.take())
                {
                    let _ = answer.send(Some(accepted.scheduler));
                }
            }
            Some(from_engine::Kind::Stats(report)) => {
                match SchedulerStats::decode(&report.snapshot) {
                    Ok(snapshot) => stats(
                        shared.world.load(Ordering::Relaxed),
                        report.scheduler,
                        snapshot,
                    ),
                    Err(error) => tracing::debug!(%error, "the engine sent unreadable statistics"),
                }
            }
            Some(from_engine::Kind::State(state))
                if state.state == wire::EngineState::Failed as i32 =>
            {
                tracing::error!(error = %state.error, "the engine failed");
                shared.state.send_replace(EngineState::Failed(state.error));
            }
            Some(from_engine::Kind::State(state))
                if state.state == wire::EngineState::Ready as i32 =>
            {
                // A prepared engine was promoted, or a parked one resumed.
                shared.state.send_if_modified(|current| {
                    let changes = matches!(*current, EngineState::Standby | EngineState::Parked);
                    if changes {
                        *current = EngineState::Ready;
                    }
                    changes
                });
            }
            Some(from_engine::Kind::State(state))
                if state.state == wire::EngineState::Parked as i32 =>
            {
                shared.state.send_if_modified(|current| {
                    let ready = *current == EngineState::Ready;
                    if ready {
                        *current = EngineState::Parked;
                    }
                    ready
                });
            }
            Some(from_engine::Kind::Hooked(hooked)) => {
                if let Some(answer) = lock(&shared.answers).remove(&hooked.id) {
                    let _ = answer.send(if hooked.error.is_empty() {
                        Ok(Answered::Results(hooked.results))
                    } else {
                        Err(hooked.error)
                    });
                }
            }
            Some(from_engine::Kind::Configured(configured)) => {
                if let Some(answer) = lock(&shared.answers).remove(&configured.id) {
                    let _ = answer.send(if configured.error.is_empty() {
                        Ok(Answered::Results(Vec::new()))
                    } else {
                        Err(configured.error)
                    });
                }
            }
            Some(from_engine::Kind::Parked(parked)) => {
                if let Some(answer) = lock(&shared.answers).remove(&parked.id) {
                    let _ = answer.send(if parked.error.is_empty() {
                        Ok(Answered::Index(parked.index))
                    } else {
                        Err(parked.error)
                    });
                }
            }
            Some(from_engine::Kind::CacheDescribed(described)) => {
                if let Some(answer) = lock(&shared.answers).remove(&described.id) {
                    let _ = answer.send(if described.error.is_empty() {
                        Ok(Answered::Segments(described.segments))
                    } else {
                        Err(described.error)
                    });
                }
            }
            Some(from_engine::Kind::State(_)) | None => {}
        }
    }
    // A hook or configure the lost engine did not answer fails.
    lock(&shared.answers).clear();
    if !matches!(*shared.state.borrow(), EngineState::Failed(_)) {
        shared.state.send_replace(EngineState::Closed);
    }
    // Every request still in flight leaves with the lost engine as
    // migratable: the daemon submits one that resumes from its generated
    // tokens to the engine that serves next, and fails any other.
    let pending: Vec<(RequestId, Pending)> = lock(&shared.requests).drain().collect();
    for (_, mut request) in pending {
        if let Some(answer) = request.accepted.take() {
            let _ = answer.send(None);
        }
        let _ = request.events.send(Event::Migratable {
            cause: proto::request::MigrationCause::Loss,
        });
    }
}

/// Hands one event to its request's stream; a terminal event ends the
/// request.
fn file(shared: &Shared, id: RequestId, event: Event) {
    let terminal = event.is_terminal();
    let mut requests = lock(&shared.requests);
    let Some(pending) = requests.get_mut(&id) else {
        return;
    };
    // The engine accepts a request before any of its scheduler's events; an
    // event before the acceptance ended the request at submission.
    if let Some(answer) = pending.accepted.take() {
        let _ = answer.send(None);
    }
    let _ = pending.events.send(event);
    if terminal {
        requests.remove(&id);
    }
}

/// One request's events, as the engine sends them.
///
/// The stream reports to the engine how many events its consumer took, which
/// lets the engine send more (`EVENT_WINDOW`). It also carries the decoder's
/// stop-string decisions: the engine holds a token request with stop strings
/// at a bounded number of tokens awaiting a decision, and
/// [`Self::acknowledge_consumed_prefix`] reports the tokens received so far
/// as completing none, [`Self::cancel_at_consumed_prefix`] a match. Dropping
/// the stream before its terminal event cancels the request.
pub struct EventStream {
    id: RequestId,
    events: mpsc::UnboundedReceiver<Event>,
    /// The engine's connection; `None` for a stream no engine feeds.
    shared: Option<Arc<Shared>>,
    /// Whether the request's tokens await the decoder's stop-string
    /// decisions.
    awaits_decoder: bool,
    /// Text tokens of the received events.
    text_tokens: usize,
    /// Text-token count of the last acknowledgement sent.
    acknowledged: usize,
    /// Events received, and the count last reported to the engine.
    consumed: u64,
    reported: u64,
    /// Whether the request may still be cancelled: neither ended nor
    /// cancelled.
    open: bool,
    on_finish: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl EventStream {
    /// The world of the engine that feeds the stream; `None` for a stream no
    /// engine feeds.
    pub fn world(&self) -> Option<u64> {
        self.shared
            .as_ref()
            .map(|shared| shared.world.load(Ordering::Relaxed))
    }

    /// A stream of the events `events` delivers, which no engine feeds.
    #[cfg(test)]
    pub(crate) fn detached(events: mpsc::UnboundedReceiver<Event>) -> Self {
        Self {
            id: RequestId::default(),
            events,
            shared: None,
            awaits_decoder: false,
            text_tokens: 0,
            acknowledged: 0,
            consumed: 0,
            reported: 0,
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
        let event = self.events.recv().await;
        match &event {
            Some(event) => self.observe(event),
            None => self.finish(),
        }
        event
    }

    /// Receives the next event; the same as [`Self::recv`].
    pub async fn next(&mut self) -> Option<Event> {
        self.recv().await
    }

    /// Receives the next event without waiting.
    ///
    /// # Errors
    ///
    /// Returns the channel's error when no event waits or the request ended.
    pub fn try_recv(&mut self) -> Result<Event, mpsc::error::TryRecvError> {
        let event = self.events.try_recv();
        match &event {
            Ok(event) => self.observe(event),
            Err(mpsc::error::TryRecvError::Disconnected) => self.finish(),
            Err(mpsc::error::TryRecvError::Empty) => {}
        }
        event
    }

    /// Counts a received event, reports consumption to the engine, and
    /// finishes the stream on a terminal event.
    fn observe(&mut self, event: &Event) {
        match event {
            Event::TextToken { .. } => self.text_tokens += 1,
            Event::TextTokens { ids } => self.text_tokens += ids.len(),
            event if event.is_terminal() => {
                self.finish();
                return;
            }
            _ => {}
        }
        self.consumed += 1;
        if self.consumed - self.reported >= EVENT_WINDOW / 2
            && let Some(shared) = &self.shared
        {
            shared.command(
                self.id,
                wire::command::Kind::Consumed(wire::Consumed {
                    events: self.consumed,
                }),
            );
            self.reported = self.consumed;
        }
    }

    /// Accepts every text token received as completing no stop string.
    ///
    /// Sends nothing for a request without stop strings, when no token
    /// arrived since the last acknowledgement, or once the request ended or
    /// was cancelled.
    pub fn acknowledge_consumed_prefix(&mut self) {
        if !self.open || !self.awaits_decoder || self.text_tokens <= self.acknowledged {
            return;
        }
        if let Some(shared) = &self.shared {
            shared.command(
                self.id,
                wire::command::Kind::Acknowledge(wire::Acknowledge {
                    output_tokens: self.text_tokens as u64,
                }),
            );
        }
        self.acknowledged = self.text_tokens;
    }

    /// Cancels the request.
    pub fn cancel(&mut self) {
        self.cancel_at_consumed_prefix(StreamCancelCause::DroppedStream);
    }

    /// Closes generation after the tokens received: `DroppedStream` cancels
    /// the request, `StopStringMatched` finishes it with `Stop`. Only the
    /// first cancellation of an open request sends a command.
    pub fn cancel_at_consumed_prefix(&mut self, cause: StreamCancelCause) {
        if !std::mem::replace(&mut self.open, false) {
            return;
        }
        if let Some(shared) = &self.shared {
            let command = match cause {
                StreamCancelCause::DroppedStream => wire::command::Kind::Cancel(wire::Cancel {}),
                StreamCancelCause::StopStringMatched => {
                    wire::command::Kind::Stop(wire::StopRequest {})
                }
            };
            shared.command(self.id, command);
        }
    }

    /// Ends the stream's request on its side and runs its callback once.
    fn finish(&mut self) {
        self.open = false;
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
    }
}

impl Drop for EventStream {
    /// Cancels a request that has not ended and runs a callback that has not
    /// run.
    fn drop(&mut self) {
        self.cancel();
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
    }
}
