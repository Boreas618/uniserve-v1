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
use std::os::fd::BorrowedFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use proto::report::SchedulerStats;
use proto::request::wire::{self, from_engine, to_engine};
use proto::request::{ArtifactMedia as _, Event, GenerationLimits, Request, RequestId};
use proto::{MappedMedia, MediaReceiver};
use service::{Connection, lock};
use tokio::sync::{mpsc, oneshot, watch};

/// What a ready engine reported it serves.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineInfo {
    /// The limits every request is checked against.
    pub limits: GenerationLimits,
    /// The video denoiser as its model family describes it to its request
    /// processing (`processing::VideoBinding::description`); empty when the
    /// engine generates no video.
    pub video_denoiser: Vec<u8>,
    /// The engine's schedulers, one per DP rank.
    pub schedulers: u32,
    /// Each worker group's KV cache segments as the engine started with
    /// them.
    pub segments: Vec<wire::CacheSegments>,
    /// The batch the workers were started for, within which the schedulers
    /// take scheduling limits in place.
    pub capacity: wire::BatchCapacity,
    /// The page seeds of the page chains the engine's workers compute
    /// (`storage::key::page_seed`): equal between epochs whose models,
    /// numerics and page layouts agree.
    pub prefix_seeds: Vec<[u8; 32]>,
    /// The seed of the keys of the trajectories the engine stores at
    /// preemption (a video request's, or an image's): equal between epochs
    /// whose denoising computes the same; zero without a denoiser or host
    /// latent pools.
    pub latent_seed: u64,
    /// How often each scheduler reports its statistics: the period whose
    /// interval each report covers, which the readers of the reports (the
    /// metrics, an epoch's observer and trial) take as the reports' cadence.
    pub stats_period: std::time::Duration,
}

/// Why an engine's report of what it serves is refused: it leaves out a part
/// the request interface requires, or states one in another form.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InfoError {
    /// It states no generation limits.
    #[error("the engine stated no generation limits")]
    Limits,
    /// It states no batch capacity.
    #[error("the engine stated no batch capacity")]
    Capacity,
    /// A page seed is not a 32-byte key.
    #[error("the engine stated a page seed of {0} bytes, not 32")]
    Seed(usize),
}

impl TryFrom<wire::EngineInfo> for EngineInfo {
    type Error = InfoError;

    fn try_from(info: wire::EngineInfo) -> Result<Self, InfoError> {
        let prefix_seeds = info
            .prefix_seeds
            .iter()
            .map(|seed| {
                <[u8; 32]>::try_from(seed.as_slice()).map_err(|_| InfoError::Seed(seed.len()))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            limits: info
                .limits
                .map(GenerationLimits::from)
                .ok_or(InfoError::Limits)?,
            video_denoiser: info.video_denoiser,
            schedulers: info.schedulers,
            segments: info.segments,
            capacity: info.capacity.ok_or(InfoError::Capacity)?,
            prefix_seeds,
            latent_seed: info.latent_seed,
            stats_period: std::time::Duration::from_millis(u64::from(info.stats_period_ms)),
        })
    }
}

/// Where an engine is in its life, as its client knows it: the lifecycle
/// the engine announces on its request link, and `Stopped` once its
/// connection closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineState {
    /// It loads its ranks, resolves its modules and captures its graphs.
    Starting,
    /// It executes and admits nothing, with its storage mapped and its
    /// graphs captured: an update's target or a standby before its
    /// promotion, or an engine that drained.
    Parked,
    /// It admits and executes requests.
    Serving,
    /// It admits nothing more; its requests end as migratable at their step
    /// boundaries.
    Draining,
    /// It shut down, or its connection closed.
    Stopped,
    /// It serves nothing more: its start failed or its ranks are lost.
    Failed(String),
}

/// The device storage a resize of the engine's KV cache makes room for
/// ([`EngineClient::reserve_storage`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Room {
    /// The same bytes on every device the engine's ranks use.
    Each(u64),
    /// The bytes of each named GPU.
    Devices(std::collections::BTreeMap<storage::manifest::Gpu, u64>),
}

/// Why a request could not be submitted to an engine.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SubmitError {
    /// The engine serves no requests.
    #[error("the engine is unavailable: {0}")]
    Unavailable(String),
}

/// What an engine reports of itself at its report period
/// (`EngineInfo::report_period`), which the client hands its [`StatsSink`].
pub enum EngineReport {
    /// One scheduler's statistics of the last interval, in the epoch the
    /// engine serves.
    Scheduler {
        /// The epoch the engine serves.
        epoch: u64,
        /// The scheduler's index in the engine.
        scheduler: u32,
        /// The interval's snapshot.
        stats: Box<SchedulerStats>,
    },
    /// The main process's allocators: each device's budget and the bytes each
    /// of its pools has mapped now.
    Memory(Vec<wire::DeviceMemory>),
}

/// Receives what the engine reports of itself.
pub type StatsSink = Box<dyn Fn(EngineReport) + Send + Sync>;

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
    /// Sent the request's admission once the engine accepts it, or `None`
    /// when the request ends first.
    accepted: Option<oneshot::Sender<Option<Admission>>>,
    /// The scheduler that serves it, once the engine accepted it.
    scheduler: Option<u32>,
}

/// How an engine admitted a request: the scheduler that serves it, and the
/// engine's own identity of it (`wire::Accepted::engine_request`).
type Admission = (u32, RequestId);

/// The connection's state shared between the client, its streams and its
/// reader.
struct Shared {
    writer: Mutex<Connection>,
    requests: Mutex<HashMap<RequestId, Pending>>,
    /// Notified whenever the last request in flight ends ([`EngineClient::drained`]).
    drained: tokio::sync::Notify,
    state: watch::Sender<EngineState>,
    next_id: AtomicU64,
    /// The engine commands awaiting their answer, by identity.
    responses: Mutex<HashMap<u64, Responder>>,
    /// The epoch the engine executes, which its statistics are recorded
    /// under.
    epoch: u64,
}

/// Where an engine command's response goes: each worker's result, or why the
/// command failed.
type Responder = oneshot::Sender<Result<Response, String>>;

/// What the engine responded to a command with: the reports of an `Operated` answer, `None` for
/// a command that reports nothing.
type Response = Option<wire::operated::Reports>;

impl Shared {
    /// Sends one frame to the engine, waiting until the socket took all of it,
    /// and says whether it was taken; a closed connection is ordinarily the
    /// reader's to report, so a caller that has nowhere else to learn of it
    /// reads the result instead.
    fn write(&self, kind: to_engine::Kind) -> std::io::Result<()> {
        let message = wire::ToEngine { kind: Some(kind) };
        let mut writer = lock(&self.writer);
        writer.send(&message)?;
        writer.flush()
    }

    /// Sends one frame to the engine; a closed connection is the reader's to
    /// report.
    fn send(&self, kind: to_engine::Kind) {
        if let Err(error) = self.write(kind) {
            tracing::debug!(%error, "the engine's connection refused a frame");
        }
    }

    /// Submits `request` with the media of its conditions, waiting until the
    /// socket took all of it; a closed connection is the reader's to report.
    ///
    /// An engine on this host maps each medium's memory file, whose
    /// descriptors ride the `Submit` frame itself in condition order. An
    /// engine on another host takes each medium as bounded parts ahead of the
    /// frame, each part written whole before the next is encoded, so a medium
    /// is never held twice. `scheduler` is the engine's scheduler the request
    /// names, the one the leader's routing program chose.
    fn submit(&self, request: &Request, scheduler: Option<u32>) {
        let media = match request {
            Request::Diffusion(diffusion) => diffusion.media.as_slice(),
            Request::Generation(_) => &[],
        };
        let frame = wire::ToEngine {
            kind: Some(to_engine::Kind::Submit(wire::Submit {
                requests: vec![wire::Request {
                    scheduler,
                    ..request.to_wire()
                }],
            })),
        };

        let mut writer = lock(&self.writer);
        let sent = if writer.peer_addr().is_none() {
            let descriptors: Vec<BorrowedFd<'_>> = media
                .iter()
                .filter_map(|medium| medium.descriptor())
                .collect();
            writer.send_with(&frame, &descriptors)
        } else {
            media
                .iter()
                .try_for_each(|medium| {
                    let bytes = medium.bytes();
                    bytes.chunks(proto::media::PART_BYTES).try_for_each(|data| {
                        writer.deliver(&wire::ToEngine {
                            kind: Some(to_engine::Kind::ConditionPart(wire::MediaPart {
                                size: bytes.len() as u64,
                                data: data.to_vec(),
                            })),
                        })
                    })
                })
                .and_then(|()| writer.send(&frame))
        };
        if let Err(error) = sent.and_then(|()| writer.flush()) {
            tracing::debug!(%error, "the engine's connection refused a submission");
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
    /// The host the engine runs on, when this daemon's engine
    /// control started it.
    host: Option<String>,
    /// The engines on the other hosts, when the worker group has ranks on
    /// other hosts: one client per peer, which this client holds so that they
    /// end with it.
    peers: Vec<EngineClient>,
}

impl EngineClient {
    /// A client of the engine of epoch `epoch`, read through `reader` and
    /// written through `writer`, two ends of one connection, over which the
    /// engine reported itself in `state` (ready or standing by) with `info`;
    /// frames `reader` already holds are filed first. `stats` receives the
    /// schedulers' statistics. `control` is the daemon connection the
    /// engine's process lives under, which the client keeps; the process ends
    /// with it.
    ///
    /// # Errors
    ///
    /// Fails when the reader thread cannot start.
    pub fn new(
        reader: Connection,
        writer: Connection,
        epoch: u64,
        info: EngineInfo,
        state: EngineState,
        stats: StatsSink,
        control: Option<Connection>,
    ) -> anyhow::Result<Self> {
        let (state, _) = watch::channel(state);
        let shared = Arc::new(Shared {
            writer: Mutex::new(writer),
            requests: Mutex::default(),
            drained: tokio::sync::Notify::new(),
            state,
            next_id: AtomicU64::new(1),
            responses: Mutex::default(),
            epoch,
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
            host: None,
            peers: Vec::new(),
        })
    }

    /// The client of engine `engine`, as this daemon's engine control
    /// numbers its engines.
    #[must_use]
    pub fn numbered(mut self, engine: u64) -> Self {
        self.engine = Some(engine);
        self
    }

    /// The client of an engine that runs on `host`.
    #[must_use]
    pub fn on_host(mut self, host: &str) -> Self {
        self.host = Some(host.to_owned());
        self
    }

    /// The scheduler node's client of a deployment whose other engines `peers` are: they end
    /// with it, and a promotion promotes them with it.
    #[must_use]
    pub fn with_peers(mut self, peers: Vec<EngineClient>) -> Self {
        self.peers = peers;
        self
    }

    /// Every engine process as the updater commands it, the scheduler node's
    /// first: its host and its number. Empty for an engine this daemon did not start.
    pub fn processes(&self) -> Vec<crate::engine::control::EpochProcess> {
        std::iter::once(self)
            .chain(&self.peers)
            .filter_map(|client| {
                Some(crate::engine::control::EpochProcess {
                    host: client.host.clone()?,
                    engine: client.engine?,
                })
            })
            .collect()
    }

    /// The engine's identity, when this daemon's engine control started or
    /// adopted it.
    pub fn engine(&self) -> Option<u64> {
        self.engine
    }

    /// The epoch the engine executes.
    pub fn epoch(&self) -> u64 {
        self.shared.epoch
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

    /// Submits `request` under an identity of the client's choosing, which
    /// this connection's requests alone are numbered in, to the engine's
    /// scheduler `scheduler` (the one the leader's routing program chose; the
    /// engine's least loaded when `None` or when it has no such scheduler),
    /// and returns its events and the scheduler serving it; the scheduler is
    /// `None` when the engine ended the request at submission, whose terminal
    /// event the stream holds.
    ///
    /// # Errors
    ///
    /// Fails when the engine serves no requests.
    pub async fn submit(
        &self,
        mut request: Request,
        scheduler: Option<u32>,
    ) -> Result<(EventStream, Option<u32>), SubmitError> {
        if let EngineState::Failed(reason) = self.state() {
            return Err(SubmitError::Unavailable(reason));
        }
        if self.state() == EngineState::Stopped {
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
                scheduler: None,
            },
        );
        // The stream exists before the acceptance arrives, so that dropping
        // the submission while it waits cancels the request.
        let mut stream = EventStream {
            id,
            engine: None,
            events: receiver,
            shared: Arc::clone(&self.shared),
            awaits_decoder,
            text_tokens: 0,
            acknowledged: 0,
            consumed: 0,
            reported: 0,
            open: true,
        };
        self.shared.submit(&request, scheduler);
        // A closed connection drops the acceptance with the request, whose
        // stream then holds the connection's error event.
        let admission = match acceptance.await {
            Ok(admission) => admission,
            Err(_) => {
                // Nothing took the sender out of `Pending`, so no event of this
                // request ever arrived and no scheduler was named: the engine
                // forgot the request before admitting it. Named here because a
                // response otherwise cannot tell this from a request that was
                // admitted and then abandoned - both end the event stream.
                tracing::warn!(
                    request = id.0,
                    "an engine dropped a request before accepting it"
                );
                None
            }
        };
        stream.engine = admission.map(|(_, engine)| engine);
        Ok((stream, admission.map(|(scheduler, _)| scheduler)))
    }

    /// Takes the event stream of the request the engine knows as `engine`
    /// over from the connection that consumed it, going on after the
    /// `consumed` events that response took: a hot restart's successor
    /// attaches to a request the engine kept running, which is neither
    /// cancelled nor submitted again. This connection gives the request an
    /// identity of its own, which the stream's events and commands carry.
    ///
    /// `acknowledgement` is where the previous consumer's stop-string
    /// acknowledgements stood, for a request whose tokens await the decoder
    /// ([`EventStream::acknowledgement`]): the stream counts its text tokens on
    /// from what that consumer received, and states what it acknowledged on
    /// this connection right after the takeover. The previous consumer's own
    /// last acknowledgement travels on its connection, which the engine reads
    /// in no order with this one, and one that arrives after the takeover
    /// names an identity the stream no longer carries; the engine keeps the
    /// highest count it is told, so stating it again is harmless.
    ///
    /// The local receiver is registered before the command is sent, so the
    /// engine's replay of the events after `consumed` reaches the returned
    /// stream even if it arrives first. The engine sends no acceptance for a
    /// takeover: the request is already admitted.
    ///
    /// # Errors
    ///
    /// Fails when the engine serves no requests.
    pub fn take_over(
        &self,
        engine: RequestId,
        consumed: u64,
        acknowledgement: Option<&proto::handoff::Acknowledgement>,
    ) -> Result<EventStream, SubmitError> {
        if let EngineState::Failed(reason) = self.state() {
            return Err(SubmitError::Unavailable(reason));
        }
        if self.state() == EngineState::Stopped {
            return Err(SubmitError::Unavailable("its process ended".to_owned()));
        }
        let request = RequestId(self.shared.next_id.fetch_add(1, Ordering::Relaxed));
        let (events, receiver) = mpsc::unbounded_channel();
        lock(&self.shared.requests).insert(
            request,
            Pending {
                events,
                accepted: None,
                scheduler: None,
            },
        );
        // Text-token counts as the engine counts them: of the tokens it
        // generated for the request, which the previous consumer received.
        let count = |tokens: u64| usize::try_from(tokens).unwrap_or(usize::MAX);
        let stream = EventStream {
            id: request,
            engine: Some(engine),
            events: receiver,
            shared: Arc::clone(&self.shared),
            awaits_decoder: acknowledgement.is_some(),
            text_tokens: acknowledgement.map_or(0, |at| count(at.received_tokens)),
            acknowledged: acknowledgement.map_or(0, |at| count(at.acknowledged_tokens)),
            consumed,
            reported: consumed,
            open: true,
        };
        self.shared.command(
            request,
            wire::command::Kind::Takeover(wire::Takeover {
                events: consumed,
                engine_request: engine.0,
            }),
        );
        // Behind the takeover on this connection, so the engine files it
        // under the identity the takeover gave the request.
        if let Some(at) = acknowledgement.filter(|at| at.acknowledged_tokens > 0) {
            self.shared.command(
                request,
                wire::command::Kind::Acknowledge(wire::Acknowledge {
                    output_tokens: at.acknowledged_tokens,
                }),
            );
        }
        Ok(stream)
    }

    /// Waits until this parked engine, and every peer of its world, serves:
    /// its promotion travels on its process link, and it announces here that
    /// it took effect.
    ///
    /// # Errors
    ///
    /// Fails when an engine reports that it failed, as when its cache cannot
    /// hold the requests handed to it, or its connection closes first.
    pub async fn await_serving(&self) -> Result<(), SubmitError> {
        for peer in &self.peers {
            Box::pin(peer.await_serving()).await?;
        }
        let mut states = self.states();
        loop {
            match states.borrow_and_update().clone() {
                EngineState::Serving => return Ok(()),
                EngineState::Failed(reason) => return Err(SubmitError::Unavailable(reason)),
                EngineState::Stopped | EngineState::Draining => {
                    return Err(SubmitError::Unavailable(
                        "the engine ended before it served".to_owned(),
                    ));
                }
                EngineState::Starting | EngineState::Parked => {}
            }
            if states.changed().await.is_err() {
                return Err(SubmitError::Unavailable(
                    "the engine's client closed".to_owned(),
                ));
            }
        }
    }

    /// Waits for the engine's process to exit, as the daemon that started it reports on the
    /// control connection the process lives under, and returns its exit status; `None` for an
    /// engine whose process this client did not start. The daemon reports every exit of a
    /// process it supervises, so the wait ends with the report or with the connection's end.
    ///
    /// # Errors
    ///
    /// Fails when the control connection ends before the exit's report.
    pub fn await_exit(&self) -> anyhow::Result<Option<String>> {
        use proto::process::{Downstream, Event as ProcessEvent, downstream, event};

        let mut control = lock(&self.control);
        let Some(control) = control.as_mut() else {
            return Ok(None);
        };
        loop {
            let message: Downstream = control.recv_next()?;
            if let Some(downstream::Kind::Event(ProcessEvent {
                kind: Some(event::Kind::Exit(exit)),
            })) = message.kind
            {
                return Ok(Some(exit.status));
            }
        }
    }

    /// The requests in flight on each of the engine's schedulers, by its index: those the engine
    /// accepted and that have not ended.
    pub fn in_flight_by_scheduler(&self) -> Vec<u32> {
        let mut counts = vec![0_u32; self.info.schedulers as usize];
        for pending in lock(&self.shared.requests).values() {
            if let Some(count) = pending
                .scheduler
                .and_then(|scheduler| counts.get_mut(scheduler as usize))
            {
                *count = count.saturating_add(1);
            }
        }
        counts
    }

    /// The requests submitted to the engine that have not ended.
    pub fn in_flight(&self) -> usize {
        lock(&self.shared.requests).len()
    }

    /// Resolves once no request submitted to the engine is in flight: at once when none is, else
    /// when the last one ends, by its terminal event or the engine's loss.
    pub async fn drained(&self) {
        loop {
            // Registered before the count is read, so an end between the two is not missed.
            let ended = self.shared.drained.notified();
            tokio::pin!(ended);
            ended.as_mut().enable();
            if self.in_flight() == 0 {
                return;
            }
            ended.await;
        }
    }

    /// Sends `command`, built with the identity its response matches, and
    /// awaits the response.
    async fn ask(
        &self,
        command: impl FnOnce(u64) -> wire::engine_command::Kind,
    ) -> Result<Response, String> {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let (responder, response) = oneshot::channel();
        lock(&self.shared.responses).insert(id, responder);
        self.shared
            .send(to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(command(id)),
            }));
        // The reader marks the engine closed before it drops the responses it holds, so a responder
        // registered after that finds the mark here and no reader ever drops it.
        if matches!(self.state(), EngineState::Stopped | EngineState::Failed(_)) {
            lock(&self.shared.responses).remove(&id);
            return Err("the engine ended before it responded".to_owned());
        }
        response
            .await
            .unwrap_or_else(|_| Err("the engine ended before it responded".to_owned()))
    }

    /// Prepares, on every rank, the copies that put each named routed-expert layer's experts in
    /// `order`, and returns each rank's allocation.
    ///
    /// # Errors
    ///
    /// Fails with the reason a rank refused the order, or when the engine ends first.
    pub async fn prepare_experts(
        &self,
        order: std::collections::HashMap<String, wire::ExpertOrder>,
    ) -> Result<Vec<wire::ExpertAllocation>, String> {
        match self
            .ask(|id| {
                wire::engine_command::Kind::PrepareExperts(wire::PrepareExperts { id, order })
            })
            .await?
        {
            Some(wire::operated::Reports::ExpertAllocations(allocations)) => Ok(allocations.ranks),
            other => Err(unexpected("prepare_experts", other)),
        }
    }

    /// Runs the prepared expert copies beside the batches, and answers once every rank's
    /// completed.
    ///
    /// # Errors
    ///
    /// Fails when no redistribution is prepared, with the reason a rank's copies failed, or
    /// when the engine ends first.
    pub async fn start_experts(&self) -> Result<(), String> {
        self.ask(|id| wire::engine_command::Kind::StartExperts(wire::StartExperts { id }))
            .await
            .map(drop)
    }

    /// Abandons the prepared or copying expert redistribution; the serving order stays.
    ///
    /// # Errors
    ///
    /// Fails when the engine ends first.
    pub async fn abort_experts(&self) -> Result<(), String> {
        self.ask(|id| wire::engine_command::Kind::AbortExperts(wire::AbortExperts { id }))
            .await
            .map(drop)
    }

    /// Loads the LoRA adapter at `path` into `slot` of every rank, and returns each rank's report.
    ///
    /// # Errors
    ///
    /// Fails with the reason a rank could not load it, or when the engine ends first.
    pub async fn load_adapter(
        &self,
        slot: u32,
        path: &str,
    ) -> Result<Vec<wire::AdapterReport>, String> {
        let path = path.to_owned();
        match self
            .ask(|id| wire::engine_command::Kind::LoadAdapter(wire::LoadAdapter { id, slot, path }))
            .await?
        {
            Some(wire::operated::Reports::AdapterReports(reports)) => Ok(reports.ranks),
            other => Err(unexpected("load_adapter", other)),
        }
    }

    /// Releases `slot`'s adapter on every rank, and returns each rank's report.
    ///
    /// # Errors
    ///
    /// Fails when the engine ends first.
    pub async fn unload_adapter(&self, slot: u32) -> Result<Vec<wire::AdapterReport>, String> {
        match self
            .ask(|id| wire::engine_command::Kind::UnloadAdapter(wire::UnloadAdapter { id, slot }))
            .await?
        {
            Some(wire::operated::Reports::AdapterReports(reports)) => Ok(reports.ranks),
            other => Err(unexpected("unload_adapter", other)),
        }
    }

    /// Starts the profiling session `start` on every rank it selects, and returns each rank's
    /// answer.
    ///
    /// # Errors
    ///
    /// Fails with the reason a rank could not start it, or when the engine ends first.
    pub async fn start_profile(
        &self,
        start: &proto::profile::SessionStart,
    ) -> Result<Vec<proto::profile::RankStart>, String> {
        let session = wire::ProfileSession::from(start);
        match self
            .ask(|id| {
                wire::engine_command::Kind::StartProfile(wire::StartProfile {
                    id,
                    session: Some(session),
                })
            })
            .await?
        {
            Some(wire::operated::Reports::ProfileStarts(starts)) => {
                Ok(starts.ranks.iter().map(Into::into).collect())
            }
            other => Err(unexpected("start_profile", other)),
        }
    }

    /// Ends profiling session `session` for `reason` on every rank that records it, and returns
    /// each rank's answer: what its file holds.
    ///
    /// # Errors
    ///
    /// Fails with the reason a rank could not end it, or when the engine ends first.
    pub async fn stop_profile(
        &self,
        session: u64,
        reason: &str,
    ) -> Result<Vec<proto::profile::RankStop>, String> {
        let reason = reason.to_owned();
        match self
            .ask(|id| {
                wire::engine_command::Kind::StopProfile(wire::StopProfile {
                    id,
                    session,
                    reason,
                })
            })
            .await?
        {
            Some(wire::operated::Reports::ProfileStops(stops)) => {
                Ok(stops.ranks.iter().map(Into::into).collect())
            }
            other => Err(unexpected("stop_profile", other)),
        }
    }

    /// Turns `collector` of profiling session `session` off for `reason` on every rank, and
    /// returns whether each ran it.
    ///
    /// # Errors
    ///
    /// Fails with the reason a rank could not release it, or when the engine ends first.
    pub async fn release_profile(
        &self,
        session: u64,
        collector: proto::profile::Collector,
        reason: &str,
    ) -> Result<Vec<proto::profile::RankRelease>, String> {
        let reason = reason.to_owned();
        match self
            .ask(|id| {
                wire::engine_command::Kind::ReleaseProfile(wire::ReleaseProfile {
                    id,
                    session,
                    collector: wire::ProfileCollector::from(collector) as i32,
                    reason,
                })
            })
            .await?
        {
            Some(wire::operated::Reports::ProfileReleases(releases)) => {
                Ok(releases.ranks.iter().map(Into::into).collect())
            }
            other => Err(unexpected("release_profile", other)),
        }
    }

    /// Replaces the step values every scheduler of the engine reads from its next step on
    /// (`proto::config::Tuning`).
    ///
    /// # Errors
    ///
    /// Fails with the reason the engine refused them, or when the engine ends first.
    pub async fn tune(&self, tuning: &proto::config::Tuning) -> Result<(), String> {
        match self
            .ask(|id| wire::engine_command::Kind::Tune(tuning.command(id)))
            .await?
        {
            None => Ok(()),
            other => Err(unexpected("tune", other)),
        }
    }

    /// What a generation update to the kernel packages `kernels` would take on every rank's
    /// device beside the serving graphs, each rank's device's extra bytes, room and the KV units
    /// its cache would give up; measured without installing anything, empty when it replaces no
    /// kernel and, unless `recapture` (a generation that places the probes elsewhere) or
    /// `graphs` (the generation's graph options where they differ from the serving ones, whose
    /// shapes the engine measures), captures no bucket again.
    ///
    /// # Errors
    ///
    /// Fails with the reason the engine could not measure it, or when the engine ends first.
    pub async fn measure_room(
        &self,
        kernels: &proto::module::Packages,
        recapture: bool,
        graphs: Option<&proto::config::GraphOptions>,
    ) -> Result<Vec<wire::DeviceRoom>, String> {
        let graphs = graphs.map(wire::GraphOptions::from);
        let kernels: Vec<wire::RoomKernel> = kernels
            .iter()
            .map(|(slot, package)| wire::RoomKernel {
                op: slot.op.clone(),
                key: slot.key.clone(),
                part: slot.part.clone(),
                backend: package.backend.clone(),
                revision: package.revision,
            })
            .collect();
        match self
            .ask(move |id| {
                wire::engine_command::Kind::MeasureRoom(wire::MeasureRoom {
                    id,
                    kernels,
                    recapture,
                    graphs,
                })
            })
            .await?
        {
            Some(wire::operated::Reports::Rooms(rooms)) => Ok(rooms.devices),
            other => Err(unexpected("measure_room", other)),
        }
    }

    /// Switches the fence probes on (`enabled`) or off, with the probed classes `off` switched
    /// off whatever `enabled` says, in place at the next step boundary; returns each process's
    /// switch.
    ///
    /// # Errors
    ///
    /// Fails with the reason a process could not switch, or when the engine ends first.
    pub async fn switch_fences(
        &self,
        enabled: bool,
        off: Vec<String>,
        kernels: bool,
    ) -> Result<Vec<wire::FencesSwitched>, String> {
        match self
            .ask(|id| {
                wire::engine_command::Kind::SwitchFences(wire::SwitchFences {
                    id,
                    enabled,
                    off,
                    kernels,
                })
            })
            .await?
        {
            Some(wire::operated::Reports::FenceSwitches(switches)) => Ok(switches.processes),
            other => Err(unexpected("switch_fences", other)),
        }
    }

    /// The spans every process of the engine keeps in its rings that ended at or after `since`
    /// (nanoseconds since the Unix epoch), each as its process named it, with the epoch the
    /// engine executes, which qualifies their batch ids. The engine answers each `ReadSpans` with
    /// one page of them (`service::span::page`), which fits a message, and the read asks for
    /// pages until the last.
    ///
    /// # Errors
    ///
    /// Fails with the reason the engine could not read them, a page that does not decode, or
    /// when the engine ends first.
    pub async fn read_spans(&self, since: u64) -> Result<(u64, Vec<service::span::Span>), String> {
        let mut read = Vec::new();
        let mut start = since;
        loop {
            let page = match self
                .ask(|id| {
                    wire::engine_command::Kind::ReadSpans(wire::ReadSpans { id, since: start })
                })
                .await?
            {
                Some(wire::operated::Reports::Spans(page)) => page,
                other => return Err(unexpected("read_spans", other)),
            };
            let spans: Vec<service::span::Span> =
                service::msgpack::decode(&page.spans).map_err(|error| {
                    format!("a page of the engine's spans does not decode: {error}")
                })?;
            read.extend(spans);
            match page.next {
                Some(next) => start = next,
                None => return Ok((page.epoch, read)),
            }
        }
    }

    /// Commit completed expert copies and record the order in worker launches; returns each
    /// rank's report.
    ///
    /// # Errors
    ///
    /// Fails with the reason a rank could not commit, or when the engine ends first.
    pub async fn commit_experts(
        &self,
        order: std::collections::HashMap<String, wire::ExpertOrder>,
    ) -> Result<Vec<wire::WeightReport>, String> {
        match self
            .ask(|id| wire::engine_command::Kind::CommitExperts(wire::CommitExperts { id, order }))
            .await?
        {
            Some(wire::operated::Reports::WeightReports(reports)) => Ok(reports.ranks),
            other => Err(unexpected("commit_experts", other)),
        }
    }

    /// Gives back or restores the device storage of the engine's KV cache
    /// (`wire::CacheChange`).
    ///
    /// # Errors
    ///
    /// Fails with the reason a scheduler could not make the change, or when
    /// the engine ends first.
    pub async fn resize_cache(&self, change: wire::CacheChange) -> Result<(), String> {
        self.ask(|id| {
            wire::engine_command::Kind::ResizeCache(wire::ResizeCache {
                id,
                change: change as i32,
                ..wire::ResizeCache::default()
            })
        })
        .await
        .map(drop)
    }

    /// Gives back the tail of the engine's KV cache until its devices'
    /// quotas have `room`, beside everything they reserve; the requests that
    /// hold the tail run on, and the response comes once they let it go.
    ///
    /// # Errors
    ///
    /// Fails, changing nothing, when even the segments the cache cannot
    /// give back leave a device too little room (naming every device's
    /// bytes), while another change of the cache is in progress, or when
    /// the engine ends first.
    pub async fn reserve_storage(&self, room: Room) -> Result<(), String> {
        let (room_bytes, device_bytes) = match room {
            Room::Each(bytes) => (bytes, Vec::new()),
            Room::Devices(devices) => (
                0,
                devices
                    .into_iter()
                    .map(|(gpu, bytes)| wire::DeviceBytes {
                        gpu: gpu.to_vec(),
                        bytes,
                    })
                    .collect(),
            ),
        };
        self.ask(|id| {
            wire::engine_command::Kind::ResizeCache(wire::ResizeCache {
                id,
                change: wire::CacheChange::MakeRoom as i32,
                room_bytes,
                device_bytes,
                ..wire::ResizeCache::default()
            })
        })
        .await
        .map(drop)
    }

    /// Lowers every rank's share of its GPU to `deployment`, the deployment's
    /// `mem_fraction_static`, and each worker group's own share `groups` states: the cache
    /// gives back what the lower limits leave no room for, demoting the blocks it evicts to the
    /// host tier first, and every rank reserves under them.
    ///
    /// # Errors
    ///
    /// Fails, changing nothing, when even the segments the cache cannot give back leave a
    /// device above its lower limit, or when the engine ends first.
    pub async fn lower_fractions(
        &self,
        deployment: f64,
        groups: std::collections::BTreeMap<String, f64>,
    ) -> Result<(), String> {
        self.ask(move |id| {
            wire::engine_command::Kind::ResizeCache(wire::ResizeCache {
                id,
                change: wire::CacheChange::Fraction as i32,
                deployment_fraction: deployment,
                group_fractions: groups.into_iter().collect(),
                ..wire::ResizeCache::default()
            })
        })
        .await
        .map(drop)
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
    ///
    /// The command is the handover's second step: the successor has no
    /// front on this engine until it arrives, so whether the engine was told is
    /// the difference between a handover that is waiting and one that can never
    /// finish. Everywhere else a refused frame is the reader's to report, and at
    /// a handover the reader may already be gone — so the write states its own
    /// result, at the level a handover's failure deserves.
    pub fn attach(&self, requests: &str) {
        let command = to_engine::Kind::Engine(wire::EngineCommand {
            kind: Some(wire::engine_command::Kind::Attach(wire::Attach {
                requests: requests.to_owned(),
            })),
        });
        match self.shared.write(command) {
            Ok(()) => tracing::info!(
                engine = ?self.engine,
                daemon = requests,
                "asked the engine to dial the successor daemon's request socket"
            ),
            Err(error) => tracing::warn!(
                engine = ?self.engine,
                daemon = requests,
                %error,
                "the engine was not asked to dial the successor daemon: the request connection \
                 refused the command, so the successor waits for a front that will not come"
            ),
        }
    }

    /// The daemon connection the engine's process lives under, for a
    /// successor daemon to keep; this client reports no exit afterwards.
    pub fn take_control(&self) -> Option<Connection> {
        lock(&self.control).take()
    }

    /// Closes the connection without shutting the engine down, as the daemon
    /// a hot restart replaced does once its requests ended: the engine goes
    /// on serving its other daemons. Waits until the reader ended.
    pub fn detach(&self) {
        self.detached.store(true, Ordering::Release);
        lock(&self.shared.writer).shutdown();
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
        // A front of a handoff or a borrowed front owns only its requests. Closing
        // it must not terminate the engine another daemon controls.
        if lock(&self.control).is_some() {
            self.shutdown();
        } else {
            self.detach();
        }
    }
}

/// Why an answer to `command` that carries `reports` of another kind is refused: the engine
/// answered another command under its identity.
fn unexpected(command: &str, reports: Response) -> String {
    format!("the engine answered {command} with {reports:?}")
}

/// Files the engine's frames until its connection closes, then ends every
/// request still in flight.
///
/// The read blocks: it ends with the connection, when the engine closes it or
/// when the client shuts its socket down ([`EngineClient::detach`]).
fn read_frames(mut reader: Connection, shared: &Shared, stats: &StatsSink) {
    let mut artifact_transfer: Option<MediaReceiver> = None;
    loop {
        let (message, mut descriptors) = match reader.recv_next_with::<wire::FromEngine>() {
            Ok(received) => received,
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => {
                tracing::error!(%error, "the engine's connection failed");
                break;
            }
        };
        match message.kind {
            Some(from_engine::Kind::ArtifactPart(part)) => {
                let received = (|| -> std::io::Result<()> {
                    let transfer = match artifact_transfer.as_mut() {
                        Some(transfer) => transfer,
                        None => artifact_transfer.insert(MediaReceiver::new(part.size)?),
                    };
                    transfer.append(part.size, &part.data)
                })();
                if let Err(error) = received {
                    tracing::error!(%error, "receiving an artifact failed");
                    break;
                }
            }
            Some(from_engine::Kind::Event(event)) => {
                // Local artifacts carry a descriptor; remote artifacts
                // follow the bounded parts written into their host storage.
                let media = match (&event.kind, artifact_transfer.take()) {
                    (Some(wire::event::Kind::Artifact(artifact)), Some(transfer)) => {
                        match transfer.finish(artifact.bytes) {
                            Ok(media) => {
                                Some(Arc::new(media) as Arc<dyn proto::request::ArtifactMedia>)
                            }
                            Err(error) => {
                                tracing::error!(%error, "receiving an artifact failed");
                                break;
                            }
                        }
                    }
                    (Some(wire::event::Kind::Artifact(artifact)), None) => descriptors
                        .pop()
                        .and_then(|descriptor| {
                            MappedMedia::map(descriptor, artifact.bytes)
                                .inspect_err(|error| {
                                    tracing::error!(%error, "an artifact does not map");
                                })
                                .ok()
                        })
                        .map(|media| Arc::new(media) as Arc<dyn proto::request::ArtifactMedia>),
                    (_, Some(_)) => {
                        tracing::error!("artifact parts were not followed by their event");
                        break;
                    }
                    (_, None) => None,
                };
                match Event::from_wire(event, media) {
                    Ok((id, event)) => file(shared, id, event),
                    Err(error) => tracing::error!(%error, "the engine sent an unreadable event"),
                }
            }
            Some(from_engine::Kind::Accepted(accepted)) => {
                let mut requests = lock(&shared.requests);
                if let Some(pending) = requests.get_mut(&RequestId(accepted.request)) {
                    pending.scheduler = Some(accepted.scheduler);
                    if let Some(responder) = pending.accepted.take() {
                        let engine = RequestId(accepted.engine_request);
                        let _ = responder.send(Some((accepted.scheduler, engine)));
                    }
                }
            }
            Some(from_engine::Kind::Metrics(metrics)) => {
                // The engine's allocator, which the deployment exports: each device's budget
                // and, per pool, the bytes it has mapped now. The message arrives at the
                // engine's report period while the process serves, so the series follow the
                // allocator rather than a snapshot taken at launch.
                //
                // The message's counters - `steps`, `batches`, `tokens`, `kernel_launches`,
                // `device_ns`, `policy_fuel`, `policy_calls`, and each rank's `queue` depth - have no
                // family: the engine's scheduler and request families are recorded from
                // `SchedulerStats` and `RequestStats`, which arrive beside this message.
                stats(EngineReport::Memory(metrics.memory));
            }
            Some(from_engine::Kind::Stats(report)) => match report.snapshot {
                Some(snapshot) => stats(EngineReport::Scheduler {
                    epoch: shared.epoch,
                    scheduler: report.scheduler,
                    stats: Box::new(SchedulerStats::from(snapshot)),
                }),
                None => tracing::warn!(
                    scheduler = report.scheduler,
                    "the engine sent statistics without a snapshot"
                ),
            },
            Some(from_engine::Kind::State(state)) => {
                // The engine announces its lifecycle; the client keeps what
                // it last announced. A state the interface does not define is
                // a protocol error, which ends the connection as a failure.
                let announced = match wire::EngineState::try_from(state.state) {
                    Ok(wire::EngineState::Starting) => EngineState::Starting,
                    Ok(wire::EngineState::Parked) => EngineState::Parked,
                    Ok(wire::EngineState::Serving) => EngineState::Serving,
                    Ok(wire::EngineState::Draining) => EngineState::Draining,
                    Ok(wire::EngineState::Stopped) => EngineState::Stopped,
                    Ok(wire::EngineState::Failed) => {
                        tracing::error!(error = %state.error, "the engine failed");
                        EngineState::Failed(state.error)
                    }
                    Ok(wire::EngineState::Unspecified) | Err(_) => {
                        let reason = format!(
                            "the engine announced the state {}, which the request interface \
                             does not define",
                            state.state
                        );
                        tracing::error!(%reason, "the engine broke the request interface");
                        shared.state.send_replace(EngineState::Failed(reason));
                        break;
                    }
                };
                shared.state.send_if_modified(|current| {
                    let changes = *current != announced;
                    if changes {
                        *current = announced;
                    }
                    changes
                });
            }
            Some(from_engine::Kind::Operated(operated)) => {
                if let Some(responder) = lock(&shared.responses).remove(&operated.id) {
                    let _ = responder.send(if operated.error.is_empty() {
                        Ok(operated.reports)
                    } else {
                        Err(operated.error)
                    });
                }
            }
            None => {}
        }
    }
    if !matches!(*shared.state.borrow(), EngineState::Failed(_)) {
        shared.state.send_replace(EngineState::Stopped);
    }
    // A command the lost engine did not answer fails.
    lock(&shared.responses).clear();
    // Every request still in flight leaves with the lost engine as
    // migratable: the daemon submits one that resumes from its generated
    // tokens, or a video request, to the engine that serves next, and fails
    // any other. No trajectory was stored anywhere new.
    let pending: Vec<(RequestId, Pending)> = lock(&shared.requests).drain().collect();
    shared.drained.notify_waiters();
    for (_, mut request) in pending {
        if let Some(responder) = request.accepted.take() {
            let _ = responder.send(None);
        }
        let _ = request.events.send(Event::Migratable {
            cause: proto::request::MigrationCause::Loss,
            denoised_steps: 0,
            stored_layout: Vec::new(),
            lend: None,
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
    if let Some(responder) = pending.accepted.take() {
        let _ = responder.send(None);
    }
    let _ = pending.events.send(event);
    if terminal {
        requests.remove(&id);
        if requests.is_empty() {
            shared.drained.notify_waiters();
        }
    }
}

/// The events of one request this daemon takes ahead of its `Consumed` reports, which it states
/// to every engine in its welcome (`wire::Welcome::event_window`): a request's events queue in its
/// stream unboundedly, so the window bounds what a slow consumer leaves queued.
pub(crate) const EVENT_WINDOW: u32 = 64;

/// The events a consumer takes between two `Consumed` reports while its request streams: half the
/// window, so the engine always holds at least half a window of credit for a consumer that keeps
/// up and never waits on a report. The end of a stream reports what was taken since the last
/// report ([`EventStream::report_consumed`]). The window and this cadence are the request
/// interface's flow control as this daemon runs it.
const REPORT_EVERY: u64 = EVENT_WINDOW as u64 / 2;

/// One request's events, as the engine sends them.
///
/// The stream reports to the engine how many events its consumer took, which
/// lets the engine send more ([`EVENT_WINDOW`]). It also carries the decoder's
/// stop-string decisions: the engine holds a token request with stop strings
/// at a bounded number of tokens awaiting a decision, and
/// [`Self::acknowledge_consumed_prefix`] reports the tokens received so far
/// as completing none, [`Self::cancel_at_consumed_prefix`] a match. Dropping
/// the stream before its terminal event cancels the request.
pub struct EventStream {
    /// The identity this connection gave the request, which its events and
    /// commands carry here.
    id: RequestId,
    /// The engine's own identity of the request, which a takeover names and
    /// its private cache pages are kept under; `None` for a request the
    /// engine did not admit.
    engine: Option<RequestId>,
    events: mpsc::UnboundedReceiver<Event>,
    /// The engine's connection.
    shared: Arc<Shared>,
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
}

impl EventStream {
    /// Provenance of private cache pages when this request resumes elsewhere:
    /// the epoch and the engine's identity of the request; `None` for a
    /// request the engine did not admit, which computed no page.
    pub fn cache_resume(&self) -> Option<proto::request::CacheResume> {
        self.engine.map(|engine| proto::request::CacheResume {
            epoch: self.epoch(),
            request_id: engine.0,
        })
    }

    /// The epoch of the engine that feeds the stream.
    pub fn epoch(&self) -> u64 {
        self.shared.epoch
    }

    /// The identity this connection gave the request.
    pub fn id(&self) -> RequestId {
        self.id
    }

    /// The engine's own identity of the request, which a successor takes its
    /// stream over by; `None` for a request the engine did not admit.
    pub fn engine_id(&self) -> Option<RequestId> {
        self.engine
    }

    /// The events its consumer took in all, as the engine counts them.
    pub fn consumed(&self) -> u64 {
        self.consumed
    }

    /// Where the request's stop-string acknowledgements stand, for a request
    /// whose tokens await the decoder: the text tokens received and those
    /// acknowledged as completing no stop string, which a successor that takes
    /// the stream over goes on from ([`EngineClient::take_over`]). `None` for a
    /// request without stop strings.
    pub fn acknowledgement(&self) -> Option<proto::handoff::Acknowledgement> {
        self.awaits_decoder
            .then_some(proto::handoff::Acknowledgement {
                received_tokens: self.text_tokens as u64,
                acknowledged_tokens: self.acknowledged as u64,
            })
    }

    /// Gives the request's event stream up without cancelling the request:
    /// the engine's next consumer takes it over from the position this one
    /// reached.
    pub fn hand_over(&mut self) {
        self.open = false;
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

    /// Polls the next event, registering `cx`'s waker; the same stream
    /// [`Self::recv`] offers.
    ///
    /// A response driven from a poll loop reads its request's events one at a
    /// time this way: it must not hold a future across a pause.
    pub fn poll_recv(&mut self, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Event>> {
        let event = std::task::ready!(self.events.poll_recv(cx));
        match &event {
            Some(event) => self.observe(event),
            None => self.finish(),
        }
        std::task::Poll::Ready(event)
    }

    /// Counts a received event, reports consumption to the engine, and
    /// finishes the stream on a terminal event.
    ///
    /// The terminal event is one the consumer took, and the engine counts it
    /// in the position a takeover replays from. Positions are reported
    /// whether or not the request is still open: the engine owns the
    /// request's retirement, and a position for a request it retired is a
    /// command it ignores.
    fn observe(&mut self, event: &Event) {
        self.consumed += 1;
        match event {
            Event::TextToken { .. } => self.text_tokens += 1,
            event if event.is_terminal() => {
                self.finish();
                return;
            }
            _ => {}
        }
        if self.consumed - self.reported >= REPORT_EVERY {
            self.report_consumed();
        }
    }

    /// Reports the position the consumer has reached, when it moved since the
    /// last report.
    ///
    /// The window's turnover is the ordinary reason to report; the end of the
    /// stream is the other one. A short request never turns the window over, and
    /// an engine that keeps a request until its consumer has taken every event —
    /// the forwarder whose join holds an epoch's shutdown, and a takeover's
    /// replay — would wait for a report that only a long request ever sends.
    fn report_consumed(&mut self) {
        if self.consumed > self.reported {
            self.shared.command(
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
        self.shared.command(
            self.id,
            wire::command::Kind::Acknowledge(wire::Acknowledge {
                output_tokens: self.text_tokens as u64,
            }),
        );
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
        let command = match cause {
            StreamCancelCause::DroppedStream => wire::command::Kind::Cancel(wire::Cancel {}),
            StreamCancelCause::StopStringMatched => wire::command::Kind::Stop(wire::StopRequest {}),
        };
        self.shared.command(self.id, command);
    }

    /// Ends the stream's request on its side.
    fn finish(&mut self) {
        self.open = false;
        self.report_consumed();
    }
}

impl Drop for EventStream {
    /// Reports the position the consumer reached and cancels a request that has
    /// not ended.
    ///
    /// The report goes out on the way out even though the request is about to be
    /// cancelled: the engine's forwarder holds a request until its consumer has
    /// taken what it sent, and a dropped stream is exactly the case where the
    /// last position would otherwise never be stated.
    fn drop(&mut self) {
        self.report_consumed();
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use proto::request::{FinishReason, GenerationRequest, SamplingParams};

    use super::*;
    use crate::test_support::{Arrival, ScriptedEngine, engine_info, scripted_client};

    /// A text request that generates `max_tokens` tokens.
    fn request(max_tokens: usize) -> Request {
        GenerationRequest {
            trace: 0,
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
            hand_off: false,
            lend: None,
        }
        .into()
    }

    /// An engine that answers each token request with as many tokens as it may generate and a
    /// finish.
    fn generating() -> (EngineClient, ScriptedEngine) {
        scripted_client(1, |arrival| match arrival {
            Arrival::Submitted(Request::Generation(request)) => {
                let tokens = request.max_und_tokens;
                (0..tokens)
                    .map(|_| Event::TextToken {
                        id: u32::from(b'a'),
                        logprob: None,
                    })
                    .chain([Event::Finished {
                        reason: FinishReason::MaxTokens,
                        stop_reason: None,
                        prompt_tokens: 4,
                        completion_tokens: tokens,
                    }])
                    .collect()
            }
            Arrival::Submitted(Request::Diffusion(_)) | Arrival::TakenOver(_) => Vec::new(),
        })
    }

    /// The positions the engine was told, in order.
    fn positions(engine: &ScriptedEngine) -> Vec<u64> {
        engine
            .commands
            .try_iter()
            .filter_map(|command| match command.kind {
                Some(wire::command::Kind::Consumed(consumed)) => Some(consumed.events),
                _ => None,
            })
            .collect()
    }

    /// A stream reports the position its consumer reached every half window and at its end,
    /// the terminal event included, so the engine holds the request for no event nobody is
    /// behind.
    #[tokio::test]
    async fn a_stream_reports_its_position_every_half_window_and_at_its_end() {
        let (client, engine) = generating();
        let tokens = EVENT_WINDOW as usize;
        let (mut stream, scheduler) = client.submit(request(tokens), None).await.unwrap();
        assert_eq!(scheduler, Some(0));
        while stream.recv().await.is_some() {}
        drop(stream);
        drop(client);
        // The window's two halves, then the terminal event.
        assert_eq!(
            positions(&engine),
            [REPORT_EVERY, 2 * REPORT_EVERY, tokens as u64 + 1]
        );
    }

    /// A cancelled request's stream goes on reporting the positions its consumer reaches: the
    /// engine owns the request's retirement and ignores a position it no longer files.
    #[tokio::test]
    async fn a_cancelled_stream_still_reports_its_position() {
        let (client, engine) = generating();
        let tokens = EVENT_WINDOW as usize;
        let (mut stream, _) = client.submit(request(tokens), None).await.unwrap();
        assert!(stream.recv().await.is_some());
        stream.cancel();
        while stream.recv().await.is_some() {}
        drop(stream);
        drop(client);
        let commands: Vec<_> = engine.commands.try_iter().filter_map(|c| c.kind).collect();
        let cancelled = commands
            .iter()
            .position(|kind| matches!(kind, wire::command::Kind::Cancel(_)))
            .expect("the request was cancelled");
        assert!(
            commands[cancelled..].iter().any(|kind| matches!(
                kind,
                wire::command::Kind::Consumed(consumed) if consumed.events == tokens as u64 + 1
            )),
            "{commands:?}"
        );
    }

    /// A successor takes a request's stream over by the engine's identity of the request, and
    /// gives it an identity among this connection's own requests, which the replayed events
    /// carry here; a submitted request learns the engine's identity from its acceptance.
    #[tokio::test]
    async fn a_takeover_names_the_engine_s_identity_and_numbers_the_request_here() {
        let (client, engine) = scripted_client(1, |arrival| match arrival {
            Arrival::TakenOver(RequestId(42)) => vec![Event::Finished {
                reason: FinishReason::Stop,
                stop_reason: None,
                prompt_tokens: 4,
                completion_tokens: 3,
            }],
            Arrival::TakenOver(_) | Arrival::Submitted(_) => Vec::new(),
        });
        let mut stream = client.take_over(RequestId(42), 3, None).unwrap();
        assert_eq!(stream.engine_id(), Some(RequestId(42)));
        assert!(matches!(stream.recv().await, Some(Event::Finished { .. })));
        let takeover = engine.commands.recv().unwrap();
        assert_eq!(takeover.request, stream.id().0);
        assert!(matches!(
            takeover.kind,
            Some(wire::command::Kind::Takeover(wire::Takeover {
                events: 3,
                engine_request: 42,
            }))
        ));

        // The scripted engine states the connection's own identity as its own.
        let (submitted, _) = client.submit(request(0), None).await.unwrap();
        assert_eq!(submitted.engine_id(), Some(submitted.id()));
        assert_ne!(submitted.id(), stream.id());
    }

    /// The next count of text tokens the engine is told acknowledged under the identity
    /// `request` on its connection, waiting for it within [`testkit::LONG`].
    fn next_acknowledged(engine: &ScriptedEngine, request: RequestId) -> u64 {
        let deadline = std::time::Instant::now() + testkit::LONG;
        loop {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            let command = engine
                .commands
                .recv_timeout(left)
                .expect("the engine was told no acknowledgement in time");
            if let (true, Some(wire::command::Kind::Acknowledge(acknowledge))) =
                (command.request == request.0, command.kind)
            {
                return acknowledge.output_tokens;
            }
        }
    }

    /// A text token of the scripted engine.
    fn token() -> Event {
        Event::TextToken {
            id: u32::from(b'a'),
            logprob: None,
        }
    }

    /// A request with stop strings is held in the engine at a bounded number of tokens its
    /// consumer has not acknowledged. A successor that takes its stream over goes on
    /// acknowledging where its predecessor stood: it states the predecessor's acknowledged
    /// count on its own connection at once, since the predecessor's last acknowledgement may
    /// reach the engine after the takeover, and counts the tokens it receives on from the
    /// predecessor's, so its decoder's next acknowledgement covers every token the engine
    /// generated for the request.
    #[tokio::test]
    async fn a_takeover_goes_on_acknowledging_a_request_with_stop_strings() {
        let (client, engine) = scripted_client(1, |arrival| match arrival {
            Arrival::Submitted(_) => vec![token(), token(), token()],
            Arrival::TakenOver(_) => vec![token(), token()],
        });
        let Request::Generation(mut stopped) = request(8) else {
            unreachable!("a token request");
        };
        stopped.stop_strings = vec!["stop".to_owned()];

        // The predecessor's decoder passes the three tokens it received, then hands the stream
        // over at its position.
        let (mut predecessor, _) = client
            .submit(Request::Generation(stopped), None)
            .await
            .unwrap();
        for _ in 0..3 {
            assert!(matches!(
                predecessor.recv().await,
                Some(Event::TextToken { .. })
            ));
        }
        predecessor.acknowledge_consumed_prefix();
        let acknowledgement = predecessor.acknowledgement();
        assert_eq!(
            acknowledgement,
            Some(proto::handoff::Acknowledgement {
                received_tokens: 3,
                acknowledged_tokens: 3,
            })
        );
        predecessor.hand_over();
        let engine_id = predecessor.engine_id().unwrap();
        let consumed = predecessor.consumed();

        let mut successor = client
            .take_over(engine_id, consumed, acknowledgement.as_ref())
            .unwrap();
        // Before its decoder passes a token of its own, the engine reads the predecessor's
        // acknowledgement under the successor's identity.
        assert_eq!(next_acknowledged(&engine, successor.id()), 3);
        for _ in 0..2 {
            assert!(matches!(
                successor.recv().await,
                Some(Event::TextToken { .. })
            ));
        }
        successor.acknowledge_consumed_prefix();
        assert_eq!(next_acknowledged(&engine, successor.id()), 5);
    }

    /// An engine's report of what it serves leaves out none of what the request interface
    /// requires, and states every page seed as a 32-byte key.
    #[test]
    fn an_engine_report_without_its_limits_capacity_or_whole_seeds_is_refused() {
        let info = engine_info();
        let reported = wire::EngineInfo {
            limits: Some((&info.limits).into()),
            capacity: Some(info.capacity),
            prefix_seeds: vec![vec![1; 32]],
            ..wire::EngineInfo::default()
        };
        let read = EngineInfo::try_from(reported.clone()).expect("a whole report");
        assert_eq!(read.prefix_seeds, [[1; 32]]);
        assert_eq!(
            EngineInfo::try_from(wire::EngineInfo {
                limits: None,
                ..reported.clone()
            }),
            Err(InfoError::Limits)
        );
        assert_eq!(
            EngineInfo::try_from(wire::EngineInfo {
                capacity: None,
                ..reported.clone()
            }),
            Err(InfoError::Capacity)
        );
        assert_eq!(
            EngineInfo::try_from(wire::EngineInfo {
                prefix_seeds: vec![vec![1; 31]],
                ..reported
            }),
            Err(InfoError::Seed(31))
        );
    }
}
