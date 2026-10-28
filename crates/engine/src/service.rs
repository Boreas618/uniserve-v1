//! The engine side of the request interface: a connection to the leader
//! daemon, over which the engine's schedulers serve the daemon's requests.
//! A hot restart of the daemon connects its successor beside it
//! (`Attach`): the engine then serves both, each its own requests, until
//! the old daemon's requests end and it closes its connection.
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
//! Threads share the work so that none waits behind another: for each
//! daemon, `request-reader` reads its frames and acts on them and
//! `request-frames` writes every frame to it, an artifact's with its
//! descriptor; `request-events` runs one task per request that forwards its
//! events to the daemon that submitted it. A request's `Accepted` is queued
//! for writing before its events can be, so the daemon always reads it
//! first.

use std::collections::HashMap;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context as _;
use proto::Connection;
use proto::request::wire::{self, from_engine, to_engine};
use proto::request::{ArtifactMedia, EVENT_WINDOW, Event, MigrationCause, Request, RequestId};

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

/// A request in flight: the slot of the scheduler serving it, the daemon
/// that submitted it, and the count of its events the daemon's consumer
/// took.
struct InFlight {
    slot: usize,
    front: usize,
    consumed: tokio::sync::watch::Sender<u64>,
}

/// Shared between the readers and the event tasks: the schedulers, which
/// one serves each request in flight, and the daemons connected.
struct Routing {
    slots: Vec<Slot>,
    requests: Mutex<HashMap<RequestId, InFlight>>,
    cache_routes: Mutex<HashMap<proto::request::CacheResume, usize>>,
    /// The next slot a load scan starts from, so equal loads rotate.
    next: AtomicUsize,
    /// Each connected daemon's frame queue, by the order it connected; a
    /// closed connection's entry is `None`.
    fronts: Mutex<Vec<Option<crossbeam_channel::Sender<Frame>>>>,
    /// The published lifecycle state; only Ready admits requests.
    state: AtomicI32,
    /// Whether the engine failed, once reported.
    failed: AtomicBool,
    /// Whether the engine's world is on trial, until its commit: the
    /// engine fails once any scheduler stops, not every one, since the
    /// daemon rolls a failing trial back rather than serving on.
    trial: AtomicBool,
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

    /// Why the engine failed, once it did: every scheduler stopped on a
    /// fatal failure, or, on trial, any one did. Names the first error of a
    /// scheduler that stopped.
    fn failure(&self) -> Option<String> {
        let trial = self.trial.load(Ordering::Acquire);
        let dead = self
            .slots
            .iter()
            .filter(|slot| slot.handle.is_dead())
            .count();
        if dead == 0 || (!trial && dead < self.slots.len()) {
            return None;
        }
        let first = self
            .slots
            .iter()
            .find_map(|slot| slot.handle.failure())
            .unwrap_or_else(|| "a scheduler stopped on a worker failure".to_owned());
        Some(if trial {
            format!("a scheduler of the world on trial stopped: {first}")
        } else {
            format!("every scheduler stopped: {first}")
        })
    }

    /// Reports the engine's failure to every connected daemon once it failed
    /// (`Routing::failure`), the first time it is called after.
    fn report_failure(&self) {
        if let Some(failure) = self.failure()
            && !self.failed.swap(true, Ordering::AcqRel)
        {
            tracing::error!(%failure, "the engine failed; it serves nothing more");
            self.state
                .store(wire::EngineState::Failed as i32, Ordering::Release);
            self.broadcast(|| {
                Frame::plain(from_engine::Kind::State(wire::State {
                    state: wire::EngineState::Failed as i32,
                    info: None,
                    error: failure.clone(),
                }))
            });
        }
    }

    /// Queues `frame` for the daemon `front`, while it is connected.
    fn send(&self, front: usize, frame: Frame) {
        if let Some(Some(frames)) = crate::lock(&self.fronts).get(front) {
            let _ = frames.send(frame);
        }
    }

    /// Queues the frame `make` builds for every connected daemon.
    fn broadcast(&self, make: impl Fn() -> Frame) {
        for frames in crate::lock(&self.fronts).iter().flatten() {
            let _ = frames.send(make());
        }
    }

    /// The daemons still connected.
    fn connected(&self) -> usize {
        crate::lock(&self.fronts).iter().flatten().count()
    }
}

/// What a daemon's reader asks of the thread that serves the engine.
enum Control {
    /// The daemon `front` closed its connection.
    Closed(usize),
    /// Connect to another daemon's request socket (`Attach`).
    Attach(String),
    /// A daemon shut the engine down.
    Shutdown,
}

/// What the engine says to a daemon that it connects to while it serves:
/// its identity and what it serves.
struct Attachment {
    engine: u64,
    info: wire::EngineInfo,
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
        None,
    )
}

/// [`serve_requests`] over a connection read through `reader` and written
/// through `writer`, two ends of the same socket, and over the connections
/// of the daemons it attaches to after (`Attach`, with `attachment`): the
/// engine serves every connected daemon, each its own requests, and lives
/// until one shuts it down or the last closes its connection. A daemon's
/// requests in flight when its connection closes are aborted.
fn serve(
    reader: Connection,
    writer: Connection,
    schedulers: Vec<SchedulerHandle>,
    attachment: Option<Attachment>,
) -> anyhow::Result<()> {
    let prepared = schedulers.iter().any(SchedulerHandle::starts_on_trial);
    let routing = Arc::new(Routing {
        slots: schedulers
            .into_iter()
            .map(|handle| Slot {
                handle,
                in_flight: AtomicUsize::new(0),
            })
            .collect(),
        requests: Mutex::default(),
        cache_routes: Mutex::default(),
        next: AtomicUsize::new(0),
        fronts: Mutex::default(),
        state: AtomicI32::new(if prepared {
            wire::EngineState::Standby
        } else {
            wire::EngineState::Ready
        } as i32),
        failed: AtomicBool::new(false),
        trial: AtomicBool::new(false),
    });
    routing.trial.store(
        routing
            .slots
            .iter()
            .any(|slot| slot.handle.starts_on_trial()),
        Ordering::Release,
    );
    anyhow::ensure!(!routing.slots.is_empty(), "the engine runs no scheduler");

    let (streams, incoming) = tokio::sync::mpsc::unbounded_channel::<Stream>();
    let events_thread = {
        let routing = Arc::clone(&routing);
        std::thread::Builder::new()
            .name("request-events".into())
            .spawn(move || forward_events(&routing, incoming))
            .context("starting the event forwarder")?
    };
    let (control, controls) = crossbeam_channel::unbounded::<Control>();
    let stop = Arc::new(AtomicBool::new(false));
    let mut threads = Vec::new();
    let opened = open_front(
        reader,
        writer,
        &routing,
        &streams,
        &control,
        &stop,
        &mut threads,
    );
    if opened.is_ok() {
        for message in &controls {
            match message {
                Control::Closed(front) => {
                    close_front(front, &routing);
                    if routing.connected() == 0 {
                        tracing::info!("the last daemon closed its connection");
                        break;
                    }
                }
                Control::Attach(address) => {
                    let attached = attachment
                        .as_ref()
                        .context("this engine takes no other daemon")
                        .and_then(|attachment| {
                            connect_front(
                                &address,
                                attachment,
                                routing.state.load(Ordering::Acquire),
                            )
                        })
                        .and_then(|(reader, writer)| {
                            open_front(
                                reader,
                                writer,
                                &routing,
                                &streams,
                                &control,
                                &stop,
                                &mut threads,
                            )
                        });
                    match attached {
                        Ok(()) => tracing::info!(daemon = %address, "a daemon attached"),
                        Err(error) => {
                            tracing::warn!(daemon = %address, "attaching a daemon failed: {error:#}");
                        }
                    }
                }
                Control::Shutdown => {
                    tracing::info!("the daemon shuts the engine down");
                    break;
                }
            }
        }
    }

    // Shutting the schedulers down ends every request in flight, whose
    // terminal events the forwarder still sends; then the forwarder ends
    // with its channel, and each writer with its queue.
    stop.store(true, Ordering::Release);
    SchedulerHandle::shutdown_all(routing.slots.iter().map(|slot| &slot.handle));
    drop(streams);
    let _ = events_thread.join();
    crate::lock(&routing.fronts).clear();
    let mut outcome = opened;
    for (writer, reader) in threads {
        let written = writer
            .join()
            .map_err(|_| anyhow::anyhow!("a frame writer panicked"))
            .and_then(|written| written);
        let read = reader
            .join()
            .map_err(|_| anyhow::anyhow!("a request reader panicked"))
            .and_then(|read| read);
        if outcome.is_ok() {
            outcome = written.and(read);
        }
    }
    outcome
}

/// The writer and reader threads of one daemon's connection.
type FrontThreads = (
    std::thread::JoinHandle<anyhow::Result<()>>,
    std::thread::JoinHandle<anyhow::Result<()>>,
);

/// Serves one more daemon over `reader` and `writer`: its frames are
/// written by a thread of their own, and its requests read by another.
fn open_front(
    mut reader: Connection,
    mut writer: Connection,
    routing: &Arc<Routing>,
    streams: &tokio::sync::mpsc::UnboundedSender<Stream>,
    control: &crossbeam_channel::Sender<Control>,
    stop: &Arc<AtomicBool>,
    threads: &mut Vec<FrontThreads>,
) -> anyhow::Result<()> {
    let (frames, outgoing) = crossbeam_channel::unbounded::<Frame>();
    let front = {
        let mut fronts = crate::lock(&routing.fronts);
        fronts.push(Some(frames));
        fronts.len() - 1
    };
    let writing = std::thread::Builder::new()
        .name(format!("request-frames-{front}"))
        .spawn(move || write_frames(&mut writer, &outgoing))
        .context("starting a frame writer")?;
    let reading = {
        let routing = Arc::clone(routing);
        let streams = streams.clone();
        let control = control.clone();
        let stop = Arc::clone(stop);
        std::thread::Builder::new()
            .name(format!("request-reader-{front}"))
            .spawn(move || {
                let read = read_requests(front, &mut reader, &routing, &streams, &control, &stop);
                let _ = control.send(Control::Closed(front));
                read
            })
            .context("starting a request reader")?
    };
    threads.push((writing, reading));
    Ok(())
}

/// Forgets the daemon `front`, whose connection closed, and aborts its
/// requests in flight: no daemon reads their events any more.
fn close_front(front: usize, routing: &Routing) {
    if let Some(entry) = crate::lock(&routing.fronts).get_mut(front) {
        *entry = None;
    }
    let abandoned: Vec<(RequestId, usize)> = crate::lock(&routing.requests)
        .iter()
        .filter(|(_, in_flight)| in_flight.front == front)
        .map(|(id, in_flight)| (*id, in_flight.slot))
        .collect();
    if !abandoned.is_empty() {
        tracing::info!(
            front,
            requests = abandoned.len(),
            "aborting a closed daemon's requests"
        );
    }
    for (id, slot) in abandoned {
        routing.slots[slot].handle.abort(id);
    }
}

/// Connects to the daemon whose request socket is `address` as the engine
/// `attachment` names: says hello, takes the daemon's welcome, and reports
/// itself serving with what it serves.
fn connect_front(
    address: &str,
    attachment: &Attachment,
    state: i32,
) -> anyhow::Result<(Connection, Connection)> {
    let path = address
        .strip_prefix("unix:")
        .with_context(|| format!("the request socket {address} is not a Unix socket"))?;
    let stream = UnixStream::connect(path)
        .with_context(|| format!("connecting to the daemon's request socket {address}"))?;
    let writer = stream
        .try_clone()
        .context("sharing the daemon's connection")?;
    let mut reader = Connection::from_unix(stream)?;
    let mut writer = Connection::from_unix(writer)?;
    writer.send(&wire::Hello {
        major: proto::PROTOCOL_MAJOR,
        minor: proto::PROTOCOL_MINOR,
        build: proto::BUILD.to_owned(),
        engine: attachment.engine,
    })?;
    let welcome: wire::Welcome = reader
        .recv(WELCOME_TIMEOUT)?
        .context("the daemon did not welcome the engine")?;
    anyhow::ensure!(
        welcome.refusal.is_empty(),
        "the daemon refused the engine: {}",
        welcome.refusal
    );
    writer.send(&wire::FromEngine {
        kind: Some(from_engine::Kind::State(wire::State {
            state,
            info: Some(attachment.info.clone()),
            error: String::new(),
        })),
    })?;
    Ok((reader, writer))
}

/// Reads the frames of the daemon `front` and acts on them until it shuts
/// the engine down, its connection closes, or `stop` is set.
fn read_requests(
    front: usize,
    reader: &mut Connection,
    routing: &Arc<Routing>,
    streams: &tokio::sync::mpsc::UnboundedSender<Stream>,
    control: &crossbeam_channel::Sender<Control>,
    stop: &AtomicBool,
) -> anyhow::Result<()> {
    loop {
        if stop.load(Ordering::Acquire) {
            return Ok(());
        }
        routing.report_failure();
        let message = match reader.try_recv::<wire::ToEngine>() {
            Ok(Some(message)) => message,
            Ok(None) => {
                reader.wait(READ_INTERVAL)?;
                continue;
            }
            Err(proto::Error::Closed) => {
                tracing::info!(front, "the daemon closed the request connection");
                return Ok(());
            }
            Err(error) => return Err(error).context("reading the daemon's requests"),
        };
        match message.kind {
            Some(to_engine::Kind::Submit(submit))
                if routing.state.load(Ordering::Acquire) != wire::EngineState::Ready as i32 =>
            {
                refuse_unready(submit, front, routing);
            }
            Some(to_engine::Kind::Submit(submit)) => submit_all(submit, front, routing, streams),
            Some(to_engine::Kind::Command(command)) => apply(command, routing),
            Some(to_engine::Kind::Engine(command)) => match command.kind {
                Some(wire::engine_command::Kind::Shutdown(_)) => {
                    let _ = control.send(Control::Shutdown);
                    return Ok(());
                }
                Some(wire::engine_command::Kind::Park(park)) => {
                    park_schedulers(park.id, front, routing)
                }
                Some(wire::engine_command::Kind::Resume(resume)) => {
                    resume_schedulers(resume, front, routing);
                }
                Some(wire::engine_command::Kind::Commit(commit)) => {
                    commit_schedulers(commit, front, routing);
                }
                Some(wire::engine_command::Kind::DescribeCache(describe)) => {
                    describe_caches(describe.id, front, routing);
                }
                Some(wire::engine_command::Kind::DescribeRequests(describe)) => {
                    describe_requests(describe.id, front, routing);
                }
                Some(wire::engine_command::Kind::CacheWarm(warm)) => {
                    warm_cache(warm, front, routing);
                }
                Some(wire::engine_command::Kind::DescribeMemory(describe)) => {
                    // The deployment's claims cover every node, so the first
                    // scheduler reads them all.
                    let answers: Vec<_> = routing
                        .slots
                        .first()
                        .map(|slot| slot.handle.ledgers())
                        .into_iter()
                        .collect();
                    answer_hooked(describe.id, answers, front, routing);
                }
                Some(wire::engine_command::Kind::ThrottleTransfers(throttle)) => {
                    // The deployment's claims cover every node, so the first
                    // scheduler throttles them all.
                    let answers: Vec<_> = routing
                        .slots
                        .first()
                        .map(|slot| slot.handle.throttle(throttle.halve))
                        .into_iter()
                        .collect();
                    answer_hooked(throttle.id, answers, front, routing);
                }
                Some(wire::engine_command::Kind::FlushCache(_)) => {
                    for slot in &routing.slots {
                        slot.handle.flush_cache();
                    }
                }
                Some(wire::engine_command::Kind::Hook(hook)) => run_hook(hook, front, routing),
                Some(wire::engine_command::Kind::CommitExperts(commit)) => {
                    let order = commit
                        .order
                        .iter()
                        .map(|(path, order)| (path.clone(), order.experts.clone()))
                        .collect::<std::collections::BTreeMap<_, _>>();
                    let together = crate::HookBarrier::new(routing.slots.len());
                    let answers = routing
                        .slots
                        .iter()
                        .map(|slot| {
                            slot.handle
                                .commit_experts(order.clone(), Arc::clone(&together))
                        })
                        .collect();
                    answer_hooked(commit.id, answers, front, routing);
                }
                Some(wire::engine_command::Kind::ResizeCache(resize)) => {
                    resize_cache(&resize, front, routing);
                }
                Some(wire::engine_command::Kind::UpdateWeights(update)) => {
                    let in_flight = match wire::WeightsInFlight::try_from(update.in_flight) {
                        Ok(wire::WeightsInFlight::Keep) => crate::WeightsInFlight::Keep,
                        Ok(wire::WeightsInFlight::Recompute) => crate::WeightsInFlight::Recompute,
                        Ok(wire::WeightsInFlight::Abort) => crate::WeightsInFlight::Abort,
                        Err(_) => {
                            routing.send(
                                front,
                                Frame::plain(from_engine::Kind::Hooked(wire::Hooked {
                                    id: update.id,
                                    results: Vec::new(),
                                    error: format!("unknown in-flight choice {}", update.in_flight),
                                })),
                            );
                            continue;
                        }
                    };
                    tracing::info!(
                        model = update.model_path,
                        ?in_flight,
                        "updating the weights in place"
                    );
                    let answers: Vec<_> = routing
                        .slots
                        .iter()
                        .map(|slot| {
                            slot.handle
                                .update_weights(update.model_path.clone(), in_flight)
                        })
                        .collect();
                    answer_hooked(update.id, answers, front, routing);
                }
                Some(wire::engine_command::Kind::Configure(configure)) => {
                    configure_schedulers(configure, front, routing);
                }
                Some(wire::engine_command::Kind::Attach(attach)) => {
                    let _ = control.send(Control::Attach(attach.requests));
                }
                Some(wire::engine_command::Kind::Promote(promote))
                    if routing.state.load(Ordering::Acquire)
                        == wire::EngineState::Standby as i32 =>
                {
                    promote_schedulers(promote, routing);
                }
                Some(wire::engine_command::Kind::Promote(_)) | None => {}
            },
            None => {}
        }
    }
}

/// Routes and submits one submission's requests, the daemon `front`'s.
/// Requests routed to one scheduler are submitted to it together, so it
/// queues all of them before it forms its next batch.
fn submit_all(
    submit: wire::Submit,
    front: usize,
    routing: &Routing,
    streams: &tokio::sync::mpsc::UnboundedSender<Stream>,
) {
    let mut batches: HashMap<usize, Vec<(RequestId, Request, Credit)>> = HashMap::new();
    for message in submit.requests {
        let id = RequestId(message.id);
        let request = match Request::from_wire(message) {
            Ok(request) => request,
            Err(error) => {
                refuse(front, routing, id, error.to_string());
                continue;
            }
        };
        let recovered = match &request {
            Request::Generation(request) => request
                .cache
                .resume
                .and_then(|resume| crate::lock(&routing.cache_routes).remove(&resume)),
            Request::Diffusion(_) => None,
        };
        let Some(index) = recovered.or_else(|| routing.pick()) else {
            let (event, _) = Event::Error {
                message: "the engine is unavailable after a worker failure".to_owned(),
            }
            .to_wire(id);
            routing.send(front, Frame::plain(from_engine::Kind::Event(event)));
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
                front,
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
                    routing.send(
                        front,
                        Frame::plain(from_engine::Kind::Accepted(wire::Accepted {
                            request: id.0,
                            scheduler: index as u32,
                        })),
                    );
                    let _ = streams.send(Stream {
                        id,
                        front,
                        events,
                        credit,
                    });
                }
            }
            Err(error) => {
                for id in ids {
                    routing.finish(id);
                    let (event, _) = Event::Error {
                        message: error.to_string(),
                    }
                    .to_wire(id);
                    routing.send(front, Frame::plain(from_engine::Kind::Event(event)));
                }
            }
        }
    }
}

/// Runs the worker hook on every scheduler's workers, all schedulers at once
/// (`engine::HookBarrier`), on a thread of its own so that the daemons'
/// other commands go on, and answers the daemon `front` with `Hooked`.
fn run_hook(hook: wire::Hook, front: usize, routing: &Arc<Routing>) {
    tracing::info!(hook = hook.name, "running a worker hook");
    let together = crate::HookBarrier::new(routing.slots.len());
    let answers: Vec<_> = routing
        .slots
        .iter()
        .map(|slot| {
            slot.handle.hook(
                hook.name.clone(),
                hook.payload.clone(),
                Arc::clone(&together),
            )
        })
        .collect();
    answer_hooked(hook.id, answers, front, routing);
}

/// Assigns a continuation to a DP scheduler by its original request identity.
/// Planning and reservation use this same assignment; `schedulers` is nonzero.
pub fn continuation_slot(origin: u64, schedulers: usize) -> usize {
    (origin % schedulers as u64) as usize
}

fn warm_cache(warm: wire::CacheWarm, front: usize, routing: &Arc<Routing>) {
    use storage::manifest::cache::Operation;
    let operation: Operation = match serde_json::from_slice(&warm.operation) {
        Ok(operation) => operation,
        Err(error) => {
            routing.send(
                front,
                Frame::plain(from_engine::Kind::Hooked(wire::Hooked {
                    id: warm.id,
                    results: Vec::new(),
                    error: error.to_string(),
                })),
            );
            return;
        }
    };
    let operations = match operation {
        Operation::Reserve {
            prefixes,
            keep,
            host,
        } => {
            let mut assigned = vec![Vec::new(); routing.slots.len()];
            let mut routes = crate::lock(&routing.cache_routes);
            for prefix in prefixes {
                let resume = proto::request::CacheResume {
                    world: prefix.resume.0,
                    request_id: prefix.resume.1,
                };
                let next = continuation_slot(resume.request_id, routing.slots.len());
                let slot = *routes.entry(resume).or_insert(next);
                assigned[slot].push(prefix);
            }
            assigned
                .into_iter()
                .map(|prefixes| Operation::Reserve {
                    prefixes,
                    keep,
                    host,
                })
                .collect::<Vec<_>>()
        }
        Operation::Coverage { prefixes, keep } => {
            let mut assigned = vec![Vec::new(); routing.slots.len()];
            let routes = crate::lock(&routing.cache_routes);
            let requests = crate::lock(&routing.requests);
            for prefix in prefixes {
                let resume = proto::request::CacheResume {
                    world: prefix.resume.0,
                    request_id: prefix.resume.1,
                };
                // Reserved successor pages have a continuation route; a
                // source still serving its request has its ordinary owner.
                let slot = routes.get(&resume).copied().or_else(|| {
                    requests
                        .get(&RequestId(prefix.request))
                        .map(|request| request.slot)
                });
                if let Some(slot) = slot {
                    assigned[slot].push(prefix);
                }
            }
            assigned
                .into_iter()
                .map(|prefixes| Operation::Coverage { prefixes, keep })
                .collect()
        }
        Operation::Discard => {
            crate::lock(&routing.cache_routes).clear();
            vec![Operation::Discard; routing.slots.len()]
        }
        Operation::Retain { requests } => {
            crate::lock(&routing.cache_routes)
                .retain(|resume, _| requests.contains(&(resume.world, resume.request_id)));
            vec![Operation::Retain { requests }; routing.slots.len()]
        }
        operation => vec![operation; routing.slots.len()],
    };
    let together = crate::HookBarrier::new(routing.slots.len());
    let answers = routing
        .slots
        .iter()
        .zip(operations)
        .map(|(slot, operation)| slot.handle.cache_warm(operation, Arc::clone(&together)))
        .collect();
    answer_hooked(warm.id, answers, front, routing);
}

/// Gives back or restores every scheduler's KV cache memory, and answers
/// the daemon `front` with `Hooked` as `run_hook` does.
fn resize_cache(resize: &wire::ResizeCache, front: usize, routing: &Arc<Routing>) {
    let successor = crate::Successor {
        loads_weights: resize.successor_loads_weights,
        unmapped_cache: resize.successor_unmapped_cache,
        host_cache: resize.successor_host_cache,
        eager_source: resize.eager_source,
        captures_graphs: resize.successor_captures_graphs,
        // A malformed UUID names no GPU of this deployment.
        rank_budgets: resize
            .rank_budgets
            .iter()
            .filter_map(|weights| {
                <[u8; 16]>::try_from(weights.gpu.as_slice())
                    .ok()
                    .map(|gpu| {
                        (
                            gpu,
                            storage::manifest::weights::RankBudget {
                                weights: weights.weights,
                                quota: weights.quota,
                                process: weights.process,
                                graphs: weights.graphs,
                                fixed: weights.fixed,
                                cache: weights.cache,
                                cache_capacity: weights.cache_capacity,
                                cache_segment: weights.cache_segment,
                                in_flight: weights.in_flight.as_ref().map(|cache| {
                                    storage::manifest::weights::CacheReservation {
                                        bytes: cache.bytes,
                                        mapped: cache.mapped,
                                    }
                                }),
                                pages: weights
                                    .pages
                                    .iter()
                                    .map(|page| storage::manifest::weights::CachePage {
                                        tokens: page.tokens,
                                        bytes: page.bytes,
                                        window: page.window,
                                        state: page.state,
                                    })
                                    .collect(),
                            },
                        )
                    })
            })
            .collect(),
    };
    let change = match wire::CacheChange::try_from(resize.change) {
        Ok(wire::CacheChange::Shrink) => crate::CacheChange::Shrink {
            preempt_after: resize.preempt_after_ms.map(Duration::from_millis),
            successor,
        },
        Ok(wire::CacheChange::Grow) => crate::CacheChange::Grow,
        Ok(wire::CacheChange::Plan) => crate::CacheChange::Plan(successor),
        Ok(wire::CacheChange::Evict) => crate::CacheChange::Evict,
        Ok(wire::CacheChange::MakeRoom) => crate::CacheChange::MakeRoom {
            bytes: resize.room_bytes,
            devices: resize
                .device_bytes
                .iter()
                .filter_map(|device| {
                    <[u8; 16]>::try_from(device.gpu.as_slice())
                        .ok()
                        .map(|gpu| (gpu, device.bytes))
                })
                .collect(),
        },
        Ok(wire::CacheChange::Restore) => crate::CacheChange::Restore,
        Ok(wire::CacheChange::Sleep) => crate::CacheChange::Sleep,
        Ok(wire::CacheChange::Wake) => crate::CacheChange::Wake,
        Err(_) => {
            routing.send(
                front,
                Frame::plain(from_engine::Kind::Hooked(wire::Hooked {
                    id: resize.id,
                    results: Vec::new(),
                    error: format!("unknown KV cache change {}", resize.change),
                })),
            );
            return;
        }
    };
    tracing::info!(?change, "changing the KV cache memory");
    if resize.keep_weights.is_empty() {
        let answers: Vec<_> = routing
            .slots
            .iter()
            .map(|slot| slot.handle.resize_cache(change.clone()))
            .collect();
        answer_hooked(resize.id, answers, front, routing);
        return;
    }
    // The worlds that keep their weights alone release the rest of their
    // memory on every node first, and move their weights to host memory when
    // asked (the deployment's claims cover every node, so the first
    // scheduler asks), on a thread of its own: the change then finds it on
    // each GPU's ledger. The answer carries each node's report of the
    // weights kept, wrapped as `{"kept": ...}`, before the schedulers'.
    let (id, worlds, to_host, cache_keys) = (
        resize.id,
        resize.keep_weights.clone(),
        resize.keep_in_host_memory,
        resize.cache_keys.clone(),
    );
    let routing = Arc::clone(routing);
    let spawned = std::thread::Builder::new()
        .name("keep-weights".to_owned())
        .spawn(move || {
            let kept = routing.slots.first().map_or(Ok(Vec::new()), |slot| {
                match slot.handle.keep_weights(worlds, to_host, cache_keys).recv() {
                    Ok(kept) => kept,
                    Err(_) => Err("a scheduler stopped".to_owned()),
                }
            });
            let reports = match kept {
                Ok(reports) => reports,
                Err(error) => {
                    routing.send(
                        front,
                        Frame::plain(from_engine::Kind::Hooked(wire::Hooked {
                            id,
                            results: Vec::new(),
                            error,
                        })),
                    );
                    return;
                }
            };
            let reports = reports
                .into_iter()
                .map(|report| format!("{{\"kept\":{report}}}"))
                .collect();
            let answers: Vec<_> = routing
                .slots
                .iter()
                .map(|slot| slot.handle.resize_cache(change.clone()))
                .collect();
            answer_hooked_after(id, reports, answers, front, &routing);
        });
    if let Err(error) = spawned {
        tracing::error!(%error, "the keep-weights thread did not start");
    }
}

/// Answers the daemon `front` with `Hooked` once every scheduler answered,
/// on a thread of its own.
fn answer_hooked(
    id: u64,
    answers: Vec<crossbeam_channel::Receiver<Result<Vec<String>, String>>>,
    front: usize,
    routing: &Arc<Routing>,
) {
    answer_hooked_after(id, Vec::new(), answers, front, routing);
}

/// [`answer_hooked`], with `first` before the schedulers' results.
fn answer_hooked_after(
    id: u64,
    first: Vec<String>,
    answers: Vec<crossbeam_channel::Receiver<Result<Vec<String>, String>>>,
    front: usize,
    routing: &Arc<Routing>,
) {
    let routing = Arc::clone(routing);
    let spawned = std::thread::Builder::new()
        .name("worker-hook".to_owned())
        .spawn(move || {
            let mut results = first;
            let mut error = String::new();
            for answer in answers {
                match answer.recv() {
                    Ok(Ok(found)) => results.extend(found),
                    Ok(Err(failed)) if error.is_empty() => error = failed,
                    Ok(Err(_)) => {}
                    Err(_) if error.is_empty() => error = "a scheduler stopped".to_owned(),
                    Err(_) => {}
                }
            }
            routing.send(
                front,
                Frame::plain(from_engine::Kind::Hooked(wire::Hooked {
                    id,
                    results,
                    error,
                })),
            );
        });
    if let Err(error) = spawned {
        tracing::error!(%error, "the worker hook thread did not start");
    }
}

/// Gives every scheduler the limits `configure` states, on a thread of its
/// own so that the daemons' other commands go on, and answers the daemon
/// `front` with `Configured`. A scheduler that refuses them keeps its own;
/// the others have taken theirs.
fn configure_schedulers(configure: wire::Configure, front: usize, routing: &Arc<Routing>) {
    let id = configure.id;
    let answer = move |error: String| {
        Frame::plain(from_engine::Kind::Configured(wire::Configured {
            id,
            error,
        }))
    };
    let limits = match configure
        .scheduling
        .as_ref()
        .context("a configure command without scheduling")
        .and_then(crate::spec::scheduler_limits)
    {
        Ok(limits) => limits,
        Err(error) => {
            routing.send(front, answer(format!("{error:#}")));
            return;
        }
    };
    tracing::info!(?limits, "configuring the schedulers");
    let answers: Vec<_> = routing
        .slots
        .iter()
        .map(|slot| slot.handle.configure(limits))
        .collect();
    let routing = Arc::clone(routing);
    let spawned = std::thread::Builder::new()
        .name("configure".to_owned())
        .spawn(move || {
            let mut error = String::new();
            for answer in answers {
                let failed = match answer.recv() {
                    Ok(Ok(())) => continue,
                    Ok(Err(failed)) => failed,
                    Err(_) => "a scheduler stopped".to_owned(),
                };
                if error.is_empty() {
                    error = failed;
                }
            }
            routing.send(front, answer(error));
        });
    if let Err(error) = spawned {
        tracing::error!(%error, "the configure thread did not start");
    }
}

/// Adopts each predecessor index through the ordinary scheduler owners,
/// then opens request admission on the same connection used to prepare.
fn promote_schedulers(promote: wire::Promote, routing: &Arc<Routing>) {
    let routing = Arc::clone(routing);
    let spawned = std::thread::Builder::new()
        .name("promote".into())
        .spawn(move || {
            let promoted = std::thread::scope(|scope| {
                let promoting: Vec<_> = routing
                    .slots
                    .iter()
                    .enumerate()
                    .map(|(slot_index, slot)| {
                        let parked = slot
                            .handle
                            .cache_group()
                            .and_then(|group| {
                                promote
                                    .index
                                    .iter()
                                    .find(|indexed| indexed.worker_group == group)
                            })
                            .map(|indexed| crate::scheduler::ParkedCache {
                                pages: indexed
                                    .pages
                                    .iter()
                                    .map(|page| crate::kv_cache::IndexedPage {
                                        group: page.group,
                                        hash: page.hash,
                                        tokens: page.tokens.clone(),
                                        units: page.units.clone(),
                                        resume: page.resume.as_ref().map(|resume| {
                                            proto::request::CacheResume {
                                                world: resume.world,
                                                request_id: resume.request_id,
                                            }
                                        }),
                                    })
                                    .collect(),
                                held_units: indexed.held_units,
                                requests: indexed.requests,
                                reserved_units: indexed.reserved_units,
                                reserving: indexed.reserving,
                            })
                            .unwrap_or_default();
                        for resume in parked.pages.iter().filter_map(|page| page.resume) {
                            crate::lock(&routing.cache_routes).insert(resume, slot_index);
                        }
                        let (recompute, grows) = (promote.recompute, promote.grows);
                        scope.spawn(move || slot.handle.promote(parked, recompute, grows))
                    })
                    .collect();
                let mut result = Ok(());
                for promotion in promoting {
                    let completed = promotion.join().unwrap_or_else(|_| {
                        Err(anyhow::anyhow!("a scheduler's promotion panicked"))
                    });
                    if result.is_ok() {
                        result = completed;
                    }
                }
                result
            });
            let (state, error) = match promoted {
                Ok(()) => (wire::EngineState::Ready, String::new()),
                Err(error) => {
                    routing.failed.store(true, Ordering::Release);
                    (wire::EngineState::Failed, format!("{error:#}"))
                }
            };
            routing.state.store(state as i32, Ordering::Release);
            routing.broadcast(|| {
                Frame::plain(from_engine::Kind::State(wire::State {
                    state: state as i32,
                    info: None,
                    error: error.clone(),
                }))
            });
        });
    if let Err(error) = spawned {
        tracing::error!(%error, "the promotion thread did not start");
    }
}

/// Parks every scheduler at once, on a thread of its own, and answers the
/// daemon `front` with `Parked` and every scheduler's prefix cache index,
/// each by the worker group whose cache it indexes; every daemon reads the
/// engine's new state.
fn park_schedulers(id: u64, front: usize, routing: &Arc<Routing>) {
    if routing
        .state
        .swap(wire::EngineState::Parked as i32, Ordering::AcqRel)
        == wire::EngineState::Parked as i32
    {
        tracing::info!("the engine parks already");
    }
    tracing::info!("the engine parks");
    let answers: Vec<_> = routing
        .slots
        .iter()
        .enumerate()
        .map(|(index, slot)| (index, slot.handle.cache_group(), slot.handle.park()))
        .collect();
    let routing = Arc::clone(routing);
    let spawned = std::thread::Builder::new()
        .name("park".to_owned())
        .spawn(move || {
            let mut index = Vec::new();
            let mut error = String::new();
            for (slot_index, group, answer) in answers {
                match answer.recv() {
                    Ok(Ok(parked)) => {
                        for resume in parked.pages.iter().filter_map(|page| page.resume) {
                            crate::lock(&routing.cache_routes).insert(resume, slot_index);
                        }
                        if let Some(group) = group {
                            index.push(wire::GroupIndex {
                                worker_group: group,
                                pages: parked
                                    .pages
                                    .into_iter()
                                    .map(|page| wire::IndexedPage {
                                        group: page.group,
                                        hash: page.hash,
                                        tokens: page.tokens,
                                        units: page.units,
                                        resume: page.resume.map(|resume| wire::CacheResume {
                                            world: resume.world,
                                            request_id: resume.request_id,
                                        }),
                                    })
                                    .collect(),
                                held_units: parked.held_units,
                                requests: parked.requests,
                                reserved_units: parked.reserved_units,
                                reserving: parked.reserving,
                            });
                        }
                    }
                    Ok(Err(failed)) if error.is_empty() => error = failed,
                    Ok(Err(_)) => {}
                    Err(_) if error.is_empty() => error = "a scheduler stopped".to_owned(),
                    Err(_) => {}
                }
            }
            if error.is_empty() {
                tracing::info!("the engine parked");
                routing.broadcast(|| {
                    Frame::plain(from_engine::Kind::State(wire::State {
                        state: wire::EngineState::Parked as i32,
                        info: None,
                        error: String::new(),
                    }))
                });
            }
            routing.send(
                front,
                Frame::plain(from_engine::Kind::Parked(wire::Parked { id, error, index })),
            );
        });
    if let Err(error) = spawned {
        tracing::error!(%error, "the park thread did not start");
    }
}

/// Resumes every parked scheduler, retires the worlds `resume` names first,
/// on a thread of its own, and answers the daemon `front` with `Hooked`;
/// every daemon reads that the engine serves again.
fn resume_schedulers(resume: wire::Resume, front: usize, routing: &Arc<Routing>) {
    tracing::info!(retire = ?resume.retire, "the parked engine resumes");
    let retired = routing
        .slots
        .first()
        .map(|slot| slot.handle.retire(resume.retire.clone(), false));
    let answers: Vec<_> = routing
        .slots
        .iter()
        .map(|slot| slot.handle.resume())
        .collect();
    let routing = Arc::clone(routing);
    let spawned = std::thread::Builder::new()
        .name("resume".to_owned())
        .spawn(move || {
            let mut error = String::new();
            if let Some(retired) = retired {
                match retired.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(failed)) => error = failed,
                    Err(_) => error = "a scheduler stopped".to_owned(),
                }
            }
            for answer in answers {
                match answer.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(failed)) if error.is_empty() => error = failed,
                    Ok(Err(_)) => {}
                    Err(_) if error.is_empty() => error = "a scheduler stopped".to_owned(),
                    Err(_) => {}
                }
            }
            if error.is_empty() {
                routing
                    .state
                    .store(wire::EngineState::Ready as i32, Ordering::Release);
                routing.broadcast(|| {
                    Frame::plain(from_engine::Kind::State(wire::State {
                        state: wire::EngineState::Ready as i32,
                        info: None,
                        error: String::new(),
                    }))
                });
            }
            routing.send(
                front,
                Frame::plain(from_engine::Kind::Hooked(wire::Hooked {
                    id: resume.id,
                    results: Vec::new(),
                    error,
                })),
            );
        });
    if let Err(error) = spawned {
        tracing::error!(%error, "the resume thread did not start");
    }
}

/// Commits the engine's world: publishes its pages, retires the worlds `commit` names (the
/// predecessor), then commits every scheduler, on a thread of its own, and
/// answers the daemon `front` with `Hooked` and every worker's report.
fn commit_schedulers(commit: wire::Commit, front: usize, routing: &Arc<Routing>) {
    tracing::info!(
        retire = ?commit.retire,
        discard_pages = commit.discard_pages,
        "the engine's world is committed"
    );
    let routing = Arc::clone(routing);
    let spawned = std::thread::Builder::new()
        .name("commit".to_owned())
        .spawn(move || {
            let mut results = Vec::new();
            let mut error = String::new();
            if let Some(slot) = routing.slots.first() {
                match slot.handle.publish_world().recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(failed)) => error = failed,
                    Err(_) => error = "a scheduler stopped".to_owned(),
                }
            }
            if error.is_empty()
                && let Some(slot) = routing.slots.first()
            {
                match slot
                    .handle
                    .retire(commit.retire.clone(), commit.discard_pages)
                    .recv()
                {
                    Ok(Ok(())) => {}
                    Ok(Err(failed)) => error = failed,
                    Err(_) => error = "a scheduler stopped".to_owned(),
                }
            }
            if error.is_empty() {
                let answers: Vec<_> = routing
                    .slots
                    .iter()
                    .map(|slot| slot.handle.commit())
                    .collect();
                for answer in answers {
                    match answer.recv() {
                        Ok(Ok(found)) => results.extend(found),
                        Ok(Err(failed)) if error.is_empty() => error = failed,
                        Ok(Err(_)) => {}
                        Err(_) if error.is_empty() => error = "a scheduler stopped".to_owned(),
                        Err(_) => {}
                    }
                }
            }
            if error.is_empty() {
                routing.trial.store(false, Ordering::Release);
            }
            routing.send(
                front,
                Frame::plain(from_engine::Kind::Hooked(wire::Hooked {
                    id: commit.id,
                    results,
                    error,
                })),
            );
        });
    if let Err(error) = spawned {
        tracing::error!(%error, "the commit thread did not start");
    }
}

/// Reports every worker group's KV cache as it holds it now, on a thread of
/// its own, and answers the daemon `front` with `CacheDescribed`.
fn describe_caches(id: u64, front: usize, routing: &Arc<Routing>) {
    let answers: Vec<_> = routing
        .slots
        .iter()
        .map(|slot| slot.handle.describe_cache())
        .collect();
    let routing = Arc::clone(routing);
    let spawned = std::thread::Builder::new()
        .name("describe-cache".to_owned())
        .spawn(move || {
            let mut segments = Vec::new();
            let mut error = String::new();
            for answer in answers {
                match answer.recv() {
                    Ok(Ok(groups)) => segments.extend(groups.into_iter().map(wire_cache)),
                    Ok(Err(failed)) if error.is_empty() => error = failed,
                    Ok(Err(_)) => {}
                    Err(_) if error.is_empty() => error = "a scheduler stopped".to_owned(),
                    Err(_) => {}
                }
            }
            routing.send(
                front,
                Frame::plain(from_engine::Kind::CacheDescribed(wire::CacheDescribed {
                    id,
                    error,
                    segments,
                })),
            );
        });
    if let Err(error) = spawned {
        tracing::error!(%error, "the cache description thread did not start");
    }
}

/// Collect each scheduler's accepted request inventory without stopping it.
fn describe_requests(id: u64, front: usize, routing: &Arc<Routing>) {
    let answers: Vec<_> = routing
        .slots
        .iter()
        .map(|slot| slot.handle.describe_requests())
        .collect();
    let routing = Arc::clone(routing);
    let spawned = std::thread::Builder::new()
        .name("describe-requests".to_owned())
        .spawn(move || {
            let mut requests = Vec::new();
            let mut error = String::new();
            for answer in answers {
                match answer.recv() {
                    Ok(prefixes) => requests.extend(prefixes),
                    Err(_) => error = "a scheduler stopped".to_owned(),
                }
            }
            routing.send(
                front,
                Frame::plain(from_engine::Kind::RequestsDescribed(
                    wire::RequestsDescribed {
                        id,
                        error,
                        requests,
                    },
                )),
            );
        });
    if let Err(error) = spawned {
        tracing::error!(%error, "the request inventory thread did not start");
    }
}

/// One worker group's KV cache on the wire.
fn wire_cache(group: crate::scheduler::GroupCache) -> wire::CacheSegments {
    wire::CacheSegments {
        worker_group: group.worker_group,
        segments: group
            .segments
            .into_iter()
            .map(|segment| wire::CacheSegment {
                world: segment.world,
                ordinal: segment.ordinal,
                units: segment.units,
            })
            .collect(),
        held: group.held,
        units: group.units,
        scratch: group.scratch,
    }
}

/// Answers the requests a parked engine is submitted: a token request that
/// resumes from its generated tokens, and a video request, leave at once as
/// migratable, as the ones it held did, and any other is refused.
fn refuse_unready(submit: wire::Submit, front: usize, routing: &Routing) {
    for message in submit.requests {
        let id = RequestId(message.id);
        let event = match Request::from_wire(message) {
            Ok(Request::Generation(request)) if request.is_migratable() => Event::Migratable {
                cause: MigrationCause::Drain,
                denoised_steps: 0,
                stored_layout: Vec::new(),
            },
            // A video request leaves naming the trajectory it named.
            Ok(Request::Diffusion(_)) => Event::Migratable {
                cause: MigrationCause::Drain,
                denoised_steps: 0,
                stored_layout: Vec::new(),
            },
            Ok(_) => Event::Error {
                message: "the engine is not ready to admit requests".to_owned(),
            },
            Err(error) => {
                refuse(front, routing, id, error.to_string());
                continue;
            }
        };
        let (event, _) = event.to_wire(id);
        routing.send(front, Frame::plain(from_engine::Kind::Event(event)));
    }
}

/// Answers a submission that does not decode with its rejection.
fn refuse(front: usize, routing: &Routing, id: RequestId, message: String) {
    let (event, _) = Event::Rejected {
        kind: proto::request::RejectionKind::Invalid,
        message,
    }
    .to_wire(id);
    routing.send(front, Frame::plain(from_engine::Kind::Event(event)));
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

/// Forwards each request's events to the daemon that submitted it, and
/// every scheduler's statistics to every daemon, until no request is in
/// flight and the readers have stopped handing over streams.
fn forward_events(
    routing: &Arc<Routing>,
    mut incoming: tokio::sync::mpsc::UnboundedReceiver<Stream>,
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
                        tasks.spawn(async move { forward(stream, &routing).await });
                    }
                    None => break,
                },
                _ = interval.tick() => {
                    for (index, (slot, reporter)) in routing.slots.iter().zip(&mut reporters).enumerate() {
                        let snapshot = reporter.snapshot(slot.handle.stats());
                        if let Ok(snapshot) = snapshot.encode() {
                            routing.broadcast(|| Frame::plain(from_engine::Kind::Stats(wire::Stats {
                                scheduler: index as u32,
                                snapshot: snapshot.clone(),
                            })));
                        }
                    }
                }
                Some(_) = tasks.join_next(), if !tasks.is_empty() => {}
            }
        }
        // The readers stopped: the requests still in flight end with their
        // schedulers' shutdown.
        while tasks.join_next().await.is_some() {}
    });
}

/// The count of a request's events its consumer took, as the daemon
/// reports it.
type Credit = tokio::sync::watch::Receiver<u64>;

/// One request's events, as a reader hands them to the forwarder, with the
/// daemon that submitted it.
struct Stream {
    id: RequestId,
    front: usize,
    events: EventRx,
    credit: Credit,
}

/// Forwards one request's events until its terminal one, never more than
/// [`EVENT_WINDOW`] beyond those its consumer took.
async fn forward(stream: Stream, routing: &Routing) {
    let Stream {
        id,
        front,
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
        let terminal = event.is_terminal();
        // A failed engine's daemon learns of the failure before the request
        // ends, as migratable for one: it then submits the request to the
        // engine that serves next, never to this one.
        if terminal {
            routing.report_failure();
        }
        let (message, media) = event.to_wire(id);
        routing.send(
            front,
            Frame {
                message: wire::FromEngine {
                    kind: Some(from_engine::Kind::Event(message)),
                },
                media,
            },
        );
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
        capacity: Some(wire::BatchCapacity {
            max_batch: first.info().max_batch_calls,
            max_num_batched_tokens: u64::from(first.info().max_batch_tokens),
            max_running_requests: u32::try_from(
                (first.info().request_slots as usize)
                    .saturating_sub(crate::scheduler::guidance_prefix_rows(first.info()))
                    .max(1),
            )
            .unwrap_or(u32::MAX),
        }),
        segments: describe(&schedulers)?,
        prefix_seed: crate::kv_cache::prefix_seed(
            first
                .info()
                .prefix_cache
                .as_ref()
                .map_or("", |cache| cache.identity.as_str()),
        ),
        latent_seed: first.latent_seed(),
    };
    // Standby and serving engines share one control channel. Standby
    // refuses submissions until promotion, while cache transfers and other
    // preparation commands use their ordinary owners and responses.
    let initial = if spec.predecessor.is_some() {
        wire::EngineState::Standby
    } else {
        wire::EngineState::Ready
    };
    writer.send(&state(initial, Some(info.clone()), String::new()))?;
    tracing::info!(
        engine,
        schedulers = schedulers.len(),
        ?initial,
        "the engine control channel is ready"
    );
    serve(
        reader,
        writer,
        schedulers,
        Some(Attachment { engine, info }),
    )
}

/// Every worker group's KV cache as the engine starts with it.
fn describe(schedulers: &[SchedulerHandle]) -> anyhow::Result<Vec<wire::CacheSegments>> {
    let mut segments = Vec::new();
    for scheduler in schedulers {
        let groups = scheduler
            .describe_cache()
            .recv()
            .map_err(|_| anyhow::anyhow!("a scheduler stopped before it described its cache"))?
            .map_err(|error| anyhow::anyhow!(error))?;
        segments.extend(groups.into_iter().map(wire_cache));
    }
    Ok(segments)
}
