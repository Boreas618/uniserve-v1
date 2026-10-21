//! The engine side of the request interface: one connection to the leader
//! daemon, over which the engine's schedulers serve the daemon's requests.
//!
//! The daemon submits requests and commands; the engine routes each request
//! to the live scheduler with the fewest requests in flight (data-parallel
//! routing, as SGLang's `DataParallelController` and vLLM's data-parallel
//! client balance by running plus waiting requests), answers it `Accepted`
//! naming the scheduler, and forwards its events. Each scheduler's
//! statistics go to the daemon once a second.
//!
//! A request's events flow under credit: the engine sends at most
//! [`EVENT_WINDOW`] beyond those the daemon's consumer took (`Consumed`), so
//! a slow consumer fills the request's event channel here and its scheduler
//! stalls the request's output, as it does for a consumer in its own process.
//!
//! Three threads share the work so that none waits behind another: this one
//! reads the daemon's frames and acts on them; `request-events` runs one
//! task per request that forwards its events; `request-frames` writes every
//! frame to the daemon, an artifact's with its descriptor. A request's
//! `Accepted` is queued for writing before its events can be, so the daemon
//! always reads it first.

use std::collections::HashMap;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context as _;
use proto::Connection;
use proto::request::wire::{self, from_engine, to_engine};
use proto::request::{ArtifactMedia, EVENT_WINDOW, Event, Request, RequestId};

use crate::core::SchedulerHandle;
use crate::handle::EventRx;
use crate::scheduler::SchedulerStatsReporter;

/// How often each scheduler's statistics are reported.
const STATS_INTERVAL: Duration = Duration::from_secs(1);

/// How long a read of the daemon's frames waits before it rechecks the
/// schedulers.
const READ_INTERVAL: Duration = Duration::from_millis(200);

/// One frame for the daemon, with the media whose descriptor travels beside
/// it.
struct Frame {
    message: wire::FromEngine,
    media: Option<Arc<dyn ArtifactMedia>>,
}

impl Frame {
    fn plain(kind: from_engine::Kind) -> Self {
        Self {
            message: wire::FromEngine { kind: Some(kind) },
            media: None,
        }
    }
}

/// A scheduler with its count of requests in flight.
struct Slot {
    handle: SchedulerHandle,
    in_flight: AtomicUsize,
}

/// A request in flight: the slot of the scheduler serving it, and the count
/// of its events the daemon's consumer took.
struct InFlight {
    slot: usize,
    consumed: tokio::sync::watch::Sender<u64>,
}

/// Shared between the reader and the event tasks: the schedulers, and which
/// one serves each request in flight.
struct Routing {
    slots: Vec<Slot>,
    requests: Mutex<HashMap<RequestId, InFlight>>,
    /// The next slot a load scan starts from, so equal loads rotate.
    next: AtomicUsize,
}

impl Routing {
    /// The live scheduler with the fewest requests in flight.
    fn pick(&self) -> Option<usize> {
        let count = self.slots.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed) % count;
        (0..count)
            .map(|offset| (start + offset) % count)
            .filter(|&index| !self.slots[index].handle.is_dead())
            .min_by_key(|&index| self.slots[index].in_flight.load(Ordering::Relaxed))
    }

    /// The scheduler that serves `request`, while it is in flight.
    fn of(&self, request: RequestId) -> Option<&SchedulerHandle> {
        let index = crate::lock(&self.requests).get(&request)?.slot;
        Some(&self.slots[index].handle)
    }

    /// Records that the daemon's consumer took `events` of `request`'s
    /// events in all.
    fn consumed(&self, request: RequestId, events: u64) {
        if let Some(in_flight) = crate::lock(&self.requests).get(&request) {
            in_flight.consumed.send_if_modified(|consumed| {
                let advanced = events > *consumed;
                *consumed = (*consumed).max(events);
                advanced
            });
        }
    }

    /// Forgets a request that ended.
    fn finish(&self, request: RequestId) {
        if let Some(in_flight) = crate::lock(&self.requests).remove(&request) {
            self.slots[in_flight.slot]
                .in_flight
                .fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Whether every scheduler stopped on a fatal failure.
    fn all_dead(&self) -> bool {
        self.slots.iter().all(|slot| slot.handle.is_dead())
    }
}

/// Serves the daemon's requests on `stream` with `schedulers` until the
/// daemon commands a shutdown or closes the connection, then shuts every
/// scheduler down.
///
/// The connection has said hello and been welcomed, and the engine has
/// reported itself ready; from here on it carries submissions, commands,
/// events, acceptances and statistics. When every scheduler fails, the
/// engine reports itself failed and goes on answering each submission with
/// an error.
///
/// # Errors
///
/// Fails when the connection fails or the event threads cannot start.
pub fn serve_requests(stream: UnixStream, schedulers: Vec<SchedulerHandle>) -> anyhow::Result<()> {
    let writer = stream
        .try_clone()
        .context("sharing the daemon's connection")?;
    serve(
        Connection::from_unix(stream)?,
        Connection::from_unix(writer)?,
        schedulers,
    )
}

/// [`serve_requests`] over a connection read through `reader` and written
/// through `writer`, two ends of the same socket.
fn serve(
    mut reader: Connection,
    mut writer: Connection,
    schedulers: Vec<SchedulerHandle>,
) -> anyhow::Result<()> {
    let routing = Arc::new(Routing {
        slots: schedulers
            .into_iter()
            .map(|handle| Slot {
                handle,
                in_flight: AtomicUsize::new(0),
            })
            .collect(),
        requests: Mutex::default(),
        next: AtomicUsize::new(0),
    });
    anyhow::ensure!(!routing.slots.is_empty(), "the engine runs no scheduler");

    let (frames, outgoing) = crossbeam_channel::unbounded::<Frame>();
    let frames_thread = std::thread::Builder::new()
        .name("request-frames".into())
        .spawn(move || write_frames(&mut writer, &outgoing))
        .context("starting the frame writer")?;
    let (streams, incoming) = tokio::sync::mpsc::unbounded_channel::<Stream>();
    let events_thread = {
        let routing = Arc::clone(&routing);
        let frames = frames.clone();
        std::thread::Builder::new()
            .name("request-events".into())
            .spawn(move || forward_events(&routing, incoming, &frames))
            .context("starting the event forwarder")?
    };

    let served = read_requests(&mut reader, &routing, &frames, &streams);

    // Shutting the schedulers down ends every request in flight, whose
    // terminal events the forwarder still sends; then the forwarder and the
    // writer end with their channels.
    for slot in &routing.slots {
        slot.handle.shutdown();
    }
    drop(streams);
    let _ = events_thread.join();
    drop(frames);
    let written = frames_thread
        .join()
        .map_err(|_| anyhow::anyhow!("the frame writer panicked"))?;
    served.and(written)
}

/// Reads the daemon's frames and acts on them until a shutdown or the
/// connection's close.
fn read_requests(
    reader: &mut Connection,
    routing: &Routing,
    frames: &crossbeam_channel::Sender<Frame>,
    streams: &tokio::sync::mpsc::UnboundedSender<Stream>,
) -> anyhow::Result<()> {
    let mut failed = false;
    loop {
        if !failed && routing.all_dead() {
            failed = true;
            tracing::error!("every scheduler failed; the engine serves nothing more");
            let _ = frames.send(Frame::plain(from_engine::Kind::State(wire::State {
                state: wire::EngineState::Failed as i32,
                info: None,
                error: "every scheduler stopped on a worker failure".to_owned(),
            })));
        }
        let message = match reader.try_recv::<wire::ToEngine>() {
            Ok(Some(message)) => message,
            Ok(None) => {
                reader.wait(READ_INTERVAL)?;
                continue;
            }
            Err(proto::Error::Closed) => {
                tracing::info!("the daemon closed the request connection");
                return Ok(());
            }
            Err(error) => return Err(error).context("reading the daemon's requests"),
        };
        match message.kind {
            Some(to_engine::Kind::Submit(submit)) => submit_all(submit, routing, frames, streams),
            Some(to_engine::Kind::Command(command)) => apply(command, routing),
            Some(to_engine::Kind::Engine(command)) => match command.kind {
                Some(wire::engine_command::Kind::Shutdown(_)) => {
                    tracing::info!("the daemon shuts the engine down");
                    return Ok(());
                }
                None => {}
            },
            None => {}
        }
    }
}

/// Routes and submits one submission's requests. Requests routed to one
/// scheduler are submitted to it together, so it queues all of them before
/// it forms its next batch.
fn submit_all(
    submit: wire::Submit,
    routing: &Routing,
    frames: &crossbeam_channel::Sender<Frame>,
    streams: &tokio::sync::mpsc::UnboundedSender<Stream>,
) {
    let mut batches: HashMap<usize, Vec<(RequestId, Request, Credit)>> = HashMap::new();
    for message in submit.requests {
        let id = RequestId(message.id);
        let request = match Request::from_wire(message) {
            Ok(request) => request,
            Err(error) => {
                refuse(frames, id, error.to_string());
                continue;
            }
        };
        let Some(index) = routing.pick() else {
            let (event, _) = Event::Error {
                message: "the engine is unavailable after a worker failure".to_owned(),
            }
            .to_wire(id);
            let _ = frames.send(Frame::plain(from_engine::Kind::Event(event)));
            continue;
        };
        // Counted before submission, so that a pick racing this one already
        // sees the request.
        routing.slots[index]
            .in_flight
            .fetch_add(1, Ordering::Relaxed);
        let (consumed, credit) = tokio::sync::watch::channel(0);
        crate::lock(&routing.requests).insert(
            id,
            InFlight {
                slot: index,
                consumed,
            },
        );
        batches
            .entry(index)
            .or_default()
            .push((id, request, credit));
    }
    for (index, batch) in batches {
        let mut ids = Vec::with_capacity(batch.len());
        let mut requests = Vec::with_capacity(batch.len());
        let mut credits = Vec::with_capacity(batch.len());
        for (id, request, credit) in batch {
            ids.push(id);
            requests.push(request);
            credits.push(credit);
        }
        let handle = &routing.slots[index].handle;
        match handle.submit_all(requests) {
            Ok(events) => {
                for ((id, events), credit) in ids.into_iter().zip(events).zip(credits) {
                    let _ =
                        frames.send(Frame::plain(from_engine::Kind::Accepted(wire::Accepted {
                            request: id.0,
                            scheduler: index as u32,
                        })));
                    let _ = streams.send(Stream { id, events, credit });
                }
            }
            Err(error) => {
                for id in ids {
                    routing.finish(id);
                    let (event, _) = Event::Error {
                        message: error.to_string(),
                    }
                    .to_wire(id);
                    let _ = frames.send(Frame::plain(from_engine::Kind::Event(event)));
                }
            }
        }
    }
}

/// Answers a submission that does not decode with its rejection.
fn refuse(frames: &crossbeam_channel::Sender<Frame>, id: RequestId, message: String) {
    let (event, _) = Event::Rejected {
        kind: proto::request::RejectionKind::Invalid,
        message,
    }
    .to_wire(id);
    let _ = frames.send(Frame::plain(from_engine::Kind::Event(event)));
}

/// Applies a command to the request it names, on the scheduler serving it;
/// a request that already ended needs none.
fn apply(command: wire::Command, routing: &Routing) {
    let id = RequestId(command.request);
    let Some(handle) = routing.of(id) else {
        return;
    };
    match command.kind {
        Some(wire::command::Kind::Cancel(_)) => handle.cancel(id),
        Some(wire::command::Kind::Stop(_)) => handle.stop(id),
        Some(wire::command::Kind::Acknowledge(acknowledge)) => {
            handle.acknowledge_at(id, acknowledge.output_tokens as usize);
        }
        Some(wire::command::Kind::Abort(_)) => handle.abort(id),
        Some(wire::command::Kind::Consumed(consumed)) => routing.consumed(id, consumed.events),
        None => {}
    }
}

/// Forwards each request's events, and every scheduler's statistics, until
/// no request is in flight and the reader has stopped handing over streams.
fn forward_events(
    routing: &Arc<Routing>,
    mut incoming: tokio::sync::mpsc::UnboundedReceiver<Stream>,
    frames: &crossbeam_channel::Sender<Frame>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::error!(%error, "the event forwarder could not start");
            return;
        }
    };
    runtime.block_on(async {
        let mut tasks = tokio::task::JoinSet::new();
        let mut reporters: Vec<SchedulerStatsReporter> = routing
            .slots
            .iter()
            .map(|_| SchedulerStatsReporter::default())
            .collect();
        let mut interval = tokio::time::interval(STATS_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                handed = incoming.recv() => match handed {
                    Some(stream) => {
                        let routing = Arc::clone(routing);
                        let frames = frames.clone();
                        tasks.spawn(async move { forward(stream, &routing, &frames).await });
                    }
                    None => break,
                },
                _ = interval.tick() => {
                    for (index, (slot, reporter)) in routing.slots.iter().zip(&mut reporters).enumerate() {
                        let snapshot = reporter.snapshot(slot.handle.stats());
                        if let Ok(snapshot) = snapshot.encode() {
                            let _ = frames.send(Frame::plain(from_engine::Kind::Stats(wire::Stats {
                                scheduler: index as u32,
                                snapshot,
                            })));
                        }
                    }
                }
                Some(_) = tasks.join_next(), if !tasks.is_empty() => {}
            }
        }
        // The reader stopped: the requests still in flight end with their
        // schedulers' shutdown.
        while tasks.join_next().await.is_some() {}
    });
}

/// The count of a request's events its consumer took, as the daemon
/// reports it.
type Credit = tokio::sync::watch::Receiver<u64>;

/// One request's events, as the reader hands them to the forwarder.
struct Stream {
    id: RequestId,
    events: EventRx,
    credit: Credit,
}

/// Forwards one request's events until its terminal one, never more than
/// [`EVENT_WINDOW`] beyond those its consumer took.
async fn forward(stream: Stream, routing: &Routing, frames: &crossbeam_channel::Sender<Frame>) {
    let Stream {
        id,
        mut events,
        mut credit,
    } = stream;
    let mut sent = 0_u64;
    loop {
        // A closed credit channel means the request ended in the reader's
        // view; its remaining events still go out.
        while sent >= *credit.borrow() + EVENT_WINDOW {
            if credit.changed().await.is_err() {
                break;
            }
        }
        let Some(event) = events.recv().await else {
            break;
        };
        sent += 1;
        let terminal = matches!(
            event,
            Event::Finished { .. } | Event::Rejected { .. } | Event::Error { .. }
        );
        let (message, media) = event.to_wire(id);
        let _ = frames.send(Frame {
            message: wire::FromEngine {
                kind: Some(from_engine::Kind::Event(message)),
            },
            media,
        });
        if terminal {
            break;
        }
    }
    routing.finish(id);
}

/// Writes every frame to the daemon, in order, until the channel closes.
fn write_frames(
    writer: &mut Connection,
    outgoing: &crossbeam_channel::Receiver<Frame>,
) -> anyhow::Result<()> {
    for frame in outgoing {
        let descriptor = frame.media.as_ref().and_then(|media| media.descriptor());
        match descriptor {
            Some(descriptor) => writer.send_with_descriptor(&frame.message, descriptor),
            None => writer.send(&frame.message),
        }
        .context("writing to the daemon")?;
    }
    Ok(())
}

/// How long an engine process waits for the daemon's welcome.
const WELCOME_TIMEOUT: Duration = Duration::from_secs(60);

/// Runs one engine process: connects to the leader daemon's request socket
/// at `requests` (`unix:<path>`) as engine `engine`, takes the spec the
/// daemon welcomes it with, starts the deployment's workers through the
/// daemons' process socket at `processes` and its schedulers, reports
/// itself ready, and serves the daemon's requests until it shuts the engine
/// down or closes the connection.
///
/// A start that fails is reported to the daemon as the engine's failure
/// before this returns it.
///
/// # Errors
///
/// Fails when the daemon cannot be reached or refuses the engine, the spec
/// does not resolve, the workers or schedulers do not start, or the
/// connection fails.
pub fn run_engine(requests: &str, processes: &str, engine: u64) -> anyhow::Result<()> {
    let path = requests
        .strip_prefix("unix:")
        .with_context(|| format!("the request socket {requests} is not a Unix socket"))?;
    let stream = UnixStream::connect(path)
        .with_context(|| format!("connecting to the daemon's request socket {requests}"))?;
    let writer = stream
        .try_clone()
        .context("sharing the daemon's connection")?;
    let mut reader = Connection::from_unix(stream)?;
    let mut writer = Connection::from_unix(writer)?;

    writer.send(&wire::Hello {
        major: proto::PROTOCOL_MAJOR,
        minor: proto::PROTOCOL_MINOR,
        build: proto::BUILD.to_owned(),
        engine,
    })?;
    let welcome: wire::Welcome = reader
        .recv(WELCOME_TIMEOUT)?
        .context("the daemon did not welcome the engine")?;
    anyhow::ensure!(
        welcome.refusal.is_empty(),
        "the daemon refused the engine: {}",
        welcome.refusal
    );
    let spec = welcome
        .spec
        .context("the daemon's welcome carries no engine spec")?;
    tracing::info!(engine, deployment = %spec.deployment, daemon = %welcome.build, "the daemon welcomed the engine");

    let state = |state: wire::EngineState, info: Option<wire::EngineInfo>, error: String| {
        wire::FromEngine {
            kind: Some(from_engine::Kind::State(wire::State {
                state: state as i32,
                info,
                error,
            })),
        }
    };
    writer.send(&state(wire::EngineState::Starting, None, String::new()))?;
    let started = crate::EngineConfig::from_spec(&spec).and_then(|mut config| {
        config.worker_process.daemon = processes.to_owned();
        SchedulerHandle::start_all(config)
    });
    let schedulers = match started {
        Ok(schedulers) => schedulers,
        Err(error) => {
            let _ = writer.send(&state(
                wire::EngineState::Failed,
                None,
                format!("{error:#}"),
            ));
            return Err(error);
        }
    };
    let first = &schedulers[0];
    let info = wire::EngineInfo {
        limits: Some((&first.generation_limits()).into()),
        video_denoiser: first
            .info()
            .video_denoiser
            .as_ref()
            .map(|denoiser| denoiser.description.clone())
            .unwrap_or_default(),
        latent_encoding_units: first
            .media_units(ipc::MediaCall::LatentEncoding)
            .unwrap_or(0),
        schedulers: schedulers.len() as u32,
    };
    writer.send(&state(wire::EngineState::Ready, Some(info), String::new()))?;
    tracing::info!(engine, schedulers = schedulers.len(), "the engine serves");
    serve(reader, writer, schedulers)
}
