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
}

impl From<wire::EngineInfo> for EngineInfo {
    fn from(info: wire::EngineInfo) -> Self {
        Self {
            limits: info.limits.map(GenerationLimits::from).unwrap_or_default(),
            video_denoiser: info.video_denoiser,
            latent_encoding_units: info.latent_encoding_units,
            schedulers: info.schedulers,
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
    /// It serves requests.
    Ready,
    /// It admits nothing more: it ended its token requests as migratable
    /// and finishes the rest.
    Draining,
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
pub type StatsSink = Box<dyn Fn(u32, SchedulerStats) + Send + Sync>;

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
    /// Kept for as long as the engine serves, such as the daemon connection
    /// its process lives under.
    _keep: Option<Box<dyn Send + Sync>>,
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
        Self::new(reader, writer, info, stats, None)
    }

    /// A client of the engine read through `reader` and written through
    /// `writer`, two ends of one connection over which the engine reported
    /// itself ready with `info`; frames `reader` already holds are filed
    /// first. `stats` receives the schedulers' statistics, and `keep` lives
    /// as long as the client.
    ///
    /// # Errors
    ///
    /// Fails when the reader thread cannot start.
    pub fn new(
        reader: Connection,
        writer: Connection,
        info: EngineInfo,
        stats: StatsSink,
        keep: Option<Box<dyn Send + Sync>>,
    ) -> anyhow::Result<Self> {
        let (state, _) = watch::channel(EngineState::Ready);
        let shared = Arc::new(Shared {
            writer: Mutex::new(writer),
            requests: Mutex::default(),
            state,
            next_id: AtomicU64::new(1),
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
            _keep: keep,
        })
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

    /// Commands the engine to drain: it admits nothing more, ends every
    /// token request that resumes from its generated tokens as migratable,
    /// and finishes the rest. Returns at once; the engine reports
    /// [`EngineState::Draining`].
    pub fn drain(&self) {
        self.shared
            .send(to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::Drain(wire::Drain {})),
            }));
    }

    /// Commands the engine to shut down, finishing every request in flight
    /// as aborted, and waits until its connection closes.
    pub fn shutdown(&self) {
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
                    Ok(snapshot) => stats(report.scheduler, snapshot),
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
                if state.state == wire::EngineState::Draining as i32 =>
            {
                shared.state.send_if_modified(|current| {
                    let ready = *current == EngineState::Ready;
                    if ready {
                        *current = EngineState::Draining;
                    }
                    ready
                });
            }
            Some(from_engine::Kind::State(_)) | None => {}
        }
    }
    if !matches!(*shared.state.borrow(), EngineState::Failed(_)) {
        shared.state.send_replace(EngineState::Closed);
    }
    // Every request still in flight ends with the engine.
    let pending: Vec<(RequestId, Pending)> = lock(&shared.requests).drain().collect();
    for (_, mut request) in pending {
        if let Some(answer) = request.accepted.take() {
            let _ = answer.send(None);
        }
        let _ = request.events.send(Event::Error {
            message: "the engine process ended".to_owned(),
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
