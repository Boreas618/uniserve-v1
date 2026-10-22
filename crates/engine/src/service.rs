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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
    /// The next slot a load scan starts from, so equal loads rotate.
    next: AtomicUsize,
    /// Each connected daemon's frame queue, by the order it connected; a
    /// closed connection's entry is `None`.
    fronts: Mutex<Vec<Option<crossbeam_channel::Sender<Frame>>>>,
    /// Whether the engine drains (`Drain`): it admits nothing more.
    draining: AtomicBool,
    /// Whether every scheduler failed, once reported.
    failed: AtomicBool,
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
        fronts: Mutex::default(),
        draining: AtomicBool::new(false),
        failed: AtomicBool::new(false),
    });
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
                        .and_then(|attachment| connect_front(&address, attachment))
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
            state: wire::EngineState::Ready as i32,
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
        if routing.all_dead() && !routing.failed.swap(true, Ordering::AcqRel) {
            tracing::error!("every scheduler failed; the engine serves nothing more");
            routing.broadcast(|| {
                Frame::plain(from_engine::Kind::State(wire::State {
                    state: wire::EngineState::Failed as i32,
                    info: None,
                    error: "every scheduler stopped on a worker failure".to_owned(),
                }))
            });
        }
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
            Some(to_engine::Kind::Submit(submit)) if routing.draining.load(Ordering::Acquire) => {
                refuse_draining(submit, front, routing);
            }
            Some(to_engine::Kind::Submit(submit)) => submit_all(submit, front, routing, streams),
            Some(to_engine::Kind::Command(command)) => apply(command, routing),
            Some(to_engine::Kind::Engine(command)) => match command.kind {
                Some(wire::engine_command::Kind::Shutdown(_)) => {
                    let _ = control.send(Control::Shutdown);
                    return Ok(());
                }
                Some(wire::engine_command::Kind::Drain(_))
                    if !routing.draining.swap(true, Ordering::AcqRel) =>
                {
                    tracing::info!("the engine drains");
                    for slot in &routing.slots {
                        slot.handle.migrate(MigrationCause::Drain);
                    }
                    routing.broadcast(|| {
                        Frame::plain(from_engine::Kind::State(wire::State {
                            state: wire::EngineState::Draining as i32,
                            info: None,
                            error: String::new(),
                        }))
                    });
                }
                Some(wire::engine_command::Kind::FlushCache(_)) => {
                    for slot in &routing.slots {
                        slot.handle.flush_cache();
                    }
                }
                Some(wire::engine_command::Kind::Hook(hook)) => run_hook(hook, front, routing),
                Some(wire::engine_command::Kind::ResizeCache(resize)) => {
                    resize_cache(&resize, front, routing);
                }
                Some(wire::engine_command::Kind::UpdateWeights(update)) => {
                    tracing::info!(model = update.model_path, "updating the weights in place");
                    let answers: Vec<_> = routing
                        .slots
                        .iter()
                        .map(|slot| slot.handle.update_weights(update.model_path.clone()))
                        .collect();
                    answer_hooked(update.id, answers, front, routing);
                }
                Some(wire::engine_command::Kind::Configure(configure)) => {
                    configure_schedulers(configure, front, routing);
                }
                Some(wire::engine_command::Kind::Attach(attach)) => {
                    let _ = control.send(Control::Attach(attach.requests));
                }
                // A serving engine was promoted, or never stood by.
                Some(
                    wire::engine_command::Kind::Drain(_) | wire::engine_command::Kind::Promote(_),
                )
                | None => {}
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
        let Some(index) = routing.pick() else {
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

/// Runs the worker hook on every scheduler's workers, on a thread of its own
/// so that the daemons' other commands go on, and answers the daemon
/// `front` with `Hooked`.
fn run_hook(hook: wire::Hook, front: usize, routing: &Arc<Routing>) {
    tracing::info!(hook = hook.name, "running a worker hook");
    let answers: Vec<_> = routing
        .slots
        .iter()
        .map(|slot| slot.handle.hook(hook.name.clone(), hook.payload.clone()))
        .collect();
    answer_hooked(hook.id, answers, front, routing);
}

/// Gives back or restores every scheduler's KV cache memory, and answers
/// the daemon `front` with `Hooked` as `run_hook` does.
fn resize_cache(resize: &wire::ResizeCache, front: usize, routing: &Arc<Routing>) {
    let change = match wire::CacheChange::try_from(resize.change) {
        Ok(wire::CacheChange::Shrink) => crate::CacheChange::Shrink,
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
    let answers: Vec<_> = routing
        .slots
        .iter()
        .map(|slot| slot.handle.resize_cache(change))
        .collect();
    answer_hooked(resize.id, answers, front, routing);
}

/// Answers the daemon `front` with `Hooked` once every scheduler answered,
/// on a thread of its own.
fn answer_hooked(
    id: u64,
    answers: Vec<crossbeam_channel::Receiver<Result<Vec<String>, String>>>,
    front: usize,
    routing: &Arc<Routing>,
) {
    let routing = Arc::clone(routing);
    let spawned = std::thread::Builder::new()
        .name("worker-hook".to_owned())
        .spawn(move || {
            let mut results = Vec::new();
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

/// Answers the requests a draining engine is submitted: a token request that
/// resumes from its generated tokens leaves at once as migratable, as the
/// ones it held did, and any other is refused.
fn refuse_draining(submit: wire::Submit, front: usize, routing: &Routing) {
    for message in submit.requests {
        let id = RequestId(message.id);
        let event = match Request::from_wire(message) {
            Ok(Request::Generation(request)) if request.is_migratable() => Event::Migratable {
                cause: MigrationCause::Drain,
            },
            Ok(_) => Event::Error {
                message: "the engine drains and admits no request".to_owned(),
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
        cache_units: schedulers
            .iter()
            .flat_map(|scheduler| scheduler.cache_units())
            .map(|(group, units)| wire::GroupCacheUnits {
                worker_group: group.clone(),
                units: *units,
            })
            .collect(),
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
    };
    // A standing-by engine serves once the daemon promotes it.
    if spec.standby.is_some() {
        writer.send(&state(
            wire::EngineState::Standby,
            Some(info.clone()),
            String::new(),
        ))?;
        tracing::info!(engine, "the engine stands by");
        if !await_promotion(&mut reader)? {
            tracing::info!(engine, "the daemon shut the standing-by engine down");
            SchedulerHandle::shutdown_all(&schedulers);
            return Ok(());
        }
        // Every scheduler promotes at once: the ranks of an expert world
        // that spans schedulers restore their suspended workspaces
        // together, so none may wait for another scheduler's promotion.
        let promoted = std::thread::scope(|scope| {
            let promoting: Vec<_> = schedulers
                .iter()
                .map(|scheduler| scope.spawn(|| scheduler.promote()))
                .collect();
            let mut outcome = Ok(());
            for promotion in promoting {
                let result = promotion
                    .join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("a scheduler's promotion panicked")));
                if outcome.is_ok() {
                    outcome = result;
                }
            }
            outcome
        });
        if let Err(error) = promoted {
            let _ = writer.send(&state(
                wire::EngineState::Failed,
                None,
                format!("{error:#}"),
            ));
            SchedulerHandle::shutdown_all(&schedulers);
            return Err(error.context("promoting the standing-by engine"));
        }
    }
    writer.send(&state(
        wire::EngineState::Ready,
        Some(info.clone()),
        String::new(),
    ))?;
    tracing::info!(engine, schedulers = schedulers.len(), "the engine serves");
    serve(
        reader,
        writer,
        schedulers,
        Some(Attachment { engine, info }),
    )
}

/// Waits on a standing-by engine's connection for the daemon's word: `true`
/// once it promotes the engine, `false` once it shuts the engine down or
/// closes the connection. A standing-by engine is submitted nothing.
fn await_promotion(reader: &mut Connection) -> anyhow::Result<bool> {
    loop {
        let message = match reader.try_recv::<wire::ToEngine>() {
            Ok(Some(message)) => message,
            Ok(None) => {
                reader.wait(READ_INTERVAL)?;
                continue;
            }
            Err(proto::Error::Closed) => return Ok(false),
            Err(error) => return Err(error).context("reading the daemon's requests"),
        };
        match message.kind {
            Some(to_engine::Kind::Engine(command)) => match command.kind {
                Some(wire::engine_command::Kind::Promote(_)) => return Ok(true),
                Some(wire::engine_command::Kind::Shutdown(_)) => return Ok(false),
                // The daemon addresses a serving engine's other commands.
                Some(_) | None => {}
            },
            Some(to_engine::Kind::Submit(_) | to_engine::Kind::Command(_)) | None => {
                tracing::warn!("the daemon addressed a standing-by engine's requests");
            }
        }
    }
}
