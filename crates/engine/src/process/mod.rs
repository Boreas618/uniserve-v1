//! A node's main process for one epoch: its control thread.
//!
//! The daemon starts one main process per node and epoch from the epoch's launch file, and
//! [`run`] serves it until the daemon commands it down, it loses its daemon, or the epoch fails.
//! The control thread is [`MainProcess`]: it holds where the engine is ([`lifecycle`]), the
//! requests and their streams ([`requests`]), and the started epoch ([`Engine`]), and it blocks
//! in one place: a `Select` over its inbox and every scheduler's outputs, until the earliest of
//! the watchdog's deadline and the next metrics second ([`MainProcess::serve`]). Nothing reaches
//! it but messages, and nothing it does waits:
//!
//! | Thread | One per | Blocks on | Sends |
//! | --- | --- | --- | --- |
//! | Control | process | its `Select` | frames to the fronts, reports to the process interface, `Message`s to the schedulers, jobs |
//! | Request reader | daemon front | a read of its connection | `Input::Frame`, then `FrontClosed` |
//! | Command reader | process | a read of the process interface | `Input::Command` |
//! | Start | process, once | the rank start (`start`) | the `Engine` or the failure, as `Done` |
//! | Operations | process | its job channel, then each job's waits | each job's `Done`, in submission order |
//! | Work | job beside the batches | the job's waits | its `Done` |
//! | Scheduler | DP rank | its inbox | `Output`s; batches to its executor |
//!
//! A job is a closure that runs elsewhere and returns the [`Input`] the control thread applies
//! when it ends ([`Input::Done`]). The operations thread runs, one at a time and in order, every
//! job that pauses the schedulers or changes the ranks' device memory (a switch, a commit, a
//! rollback, a weight update, sleep and wake, a flush, an expert commit, making room, growing,
//! the release after a `Shrunk`, the start and the end of a drain), so two pause sequences never
//! interleave; while the engine drains the control thread starts none that pauses the schedulers,
//! since a `Resume` ends a drain as well as a pause (`Lifecycle::boundary_refusal`). A job beside
//! the batches (a preparation, an adapter, a profiling change, a drain's wait, an attach) runs on
//! a thread of its own. The job functions are `generation`, `epoch` and `operations`, which read
//! the engine's parts and never this thread's state.
//!
//! The control thread and the request interface are library code: [`serve`] runs them over two
//! given connections and a [`Start`], the function that builds the [`Engine`]. The binary's
//! start loads ranks and opens a session with the node's Storage Server; a test's builds fakes,
//! and everything above that seam is the same code.

pub(crate) mod binding;
pub(crate) mod catalogue;
mod daemon;
mod epoch;
mod generation;
mod lifecycle;
mod operations;
pub(crate) mod ranks;
pub(crate) mod record;
mod reports;
mod requests;
mod start;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::executor::peers::Liveness;
use crate::executor::peers::messages::{RankOperation, RankReports};
use crate::process::binding::WeightsUpdated;
use crate::worker::watchdog::Watchdog;
use anyhow::{Context as _, Result, bail};
use crossbeam_channel::{Receiver, Select, Sender};
use proto::config::EngineArgs;
use proto::module::{ModuleSlot, Packages};
use proto::process::command;
use proto::request::wire::{self, engine_command, from_engine};
use proto::request::{ArtifactMedia, RequestId};
use scheduling::Parameters;

use scheduling::view::Snapshot;
use service::Connection;

pub use daemon::connect;
use lifecycle::{Lifecycle, Update};
use requests::RequestStream;
pub use start::Engine;

use crate::memory::client::StorageClient;
use crate::modules::load;
use crate::modules::select::changed_operators;
use crate::process::binding::Target;
use crate::scheduler::{Message, Output};

/// Builds the started epoch from the launch: the binary's rank start, or a test's fakes. It
/// runs on a thread of its own while the control thread serves.
pub type Start = Box<dyn FnOnce(&EngineArgs) -> Result<Engine> + Send>;

/// A job: work off the control thread, which returns what the control thread applies when it
/// ends.
type Job = Box<dyn FnOnce() -> Input + Send>;

/// What reaches the control thread's inbox.
pub(crate) enum Input {
    /// A frame of the request interface from front `.0`, with the condition media that came with
    /// it (`Submit`), in the order its requests' conditions state them.
    Frame(usize, Box<wire::ToEngine>, Vec<Arc<proto::MappedMedia>>),
    /// Front `.0` closed.
    FrontClosed(usize),
    /// A command of the process interface; `None` once the daemon closed it.
    Command(Option<proto::process::Command>),
    /// A peer link of the epoch closed: it was lost, or the scheduler node closed this peer's
    /// link in order, which ends the epoch.
    LinkClosed(String),
    /// The worker of rank `.0` of the started epoch found a sticky error on its device's context.
    Faulted(usize),
    /// A job ended: what it produced, applied on the control thread.
    Done(Box<dyn FnOnce(&mut MainProcess) + Send>),
}

/// The `Done` of a job that applies `apply`.
fn done(apply: impl FnOnce(&mut MainProcess) + Send + 'static) -> Input {
    Input::Done(Box::new(apply))
}

/// The slot the scheduling policy's package is kept under in the packages a generation is made
/// of, beside the kernels' slots: the interface it implements, which no operator is named.
fn policy_slot() -> ModuleSlot {
    ModuleSlot::all(proto::module::POLICY)
}

/// `packages` without the policy's slot: the kernel packages of a generation.
fn kernels_of(packages: &Packages) -> Packages {
    let policy = policy_slot();
    packages
        .iter()
        .filter(|(slot, _)| **slot != policy)
        .map(|(slot, package)| (slot.clone(), package.clone()))
        .collect()
}

/// Runs the main process `launch` describes, until its daemon commands
/// it down or its epoch fails. Its checkpoint is served by the family of `registry` that claims
/// the class it declares.
///
/// # Errors
///
/// When the launch is refused, or the daemon refuses the process or its
/// claim. A process whose epoch fails or loses a peer exits with status 1 without its orderly
/// close: a device or a collective may hold every rank.
pub fn run(launch: EngineArgs, registry: &nn::family::Registry) -> Result<()> {
    let mut requests = launch
        .requests
        .parse()
        .map_err(std::io::Error::from)
        .and_then(|endpoint| Connection::connect(&endpoint))
        .with_context(|| format!("connecting to the daemon's requests at {}", launch.requests))?;
    let window = connect(&mut requests, &launch)?;

    // What fails before the control thread runs is said on the request interface: the daemon
    // waits for the state a start ends in, and a connection that closes says only that it went.
    let claimed = check(&launch).and_then(|()| {
        let mut process = launch
            .processes
            .parse()
            .map_err(std::io::Error::from)
            .and_then(|endpoint| Connection::connect(&endpoint))
            .with_context(|| {
                format!(
                    "connecting to the daemon's processes at {}",
                    launch.processes
                )
            })?;
        let (storage, filed) = daemon::claim(&mut process, &launch)?;
        Ok((process, storage, filed))
    });
    let (process, storage, filed) = match claimed {
        Ok(claimed) => claimed,
        Err(error) => {
            let failed = wire::State {
                state: wire::EngineState::Failed as i32,
                info: None,
                error: format!("{error:#}"),
            };
            let _ = daemon::send(&mut requests, from_engine::Kind::State(failed), None);
            return Err(error);
        }
    };

    let main = MainProcess::new(requests, window, process, launch)?;
    let lost = main.link_closed();
    let registry = registry.clone();
    let start: Start = Box::new(move |launch| {
        // The module set's policy is resolved at the node's Build Server, which the ranks'
        // kernels and every later generation are resolved through too.
        let socket = Path::new(&launch.build_server);
        let mut build = build_server::Client::connect(socket)
            .with_context(|| format!("connecting to the Build Server at {}", socket.display()))?;
        let policy = load::policy(&mut build, &launch.modules.policy)?;
        let liveness = Liveness::new(lost);

        let build = Some(Arc::new(Mutex::new(build)));
        start::start(launch, &registry, &storage, build, policy, liveness)
    });
    if let Err(failed) = main.serve(start, filed) {
        let error = format!("{:#}", failed.error);
        tracing::error!(%error, "the main process ends: the epoch failed");
        // The process ends holding the failed epoch: `exit` runs no destructor, so nothing of it
        // is released while a device or a peer may still address it, and the daemon learns of
        // the end from the process's exit.
        std::process::exit(1);
    }
    tracing::warn!("the main process ends: a shutdown was commanded");
    Ok(())
}

/// How the control thread ended when its epoch failed: why, and the started epoch as the failure
/// left it.
///
/// The epoch is handed to the caller, the owner of the process's lifetime, instead of being
/// dropped: dropping a worker joins its thread, which a device that stopped making progress never
/// lets end, and its storage, given back, could still be addressed by device work or a peer rank.
/// The engine's executable ends the process while it holds the epoch ([`run`]).
pub struct Failed {
    /// Why the epoch failed: the start, a scheduler, a device, a stall or a lost peer.
    pub error: anyhow::Error,
    /// The started epoch, boxed as the error of every return it travels in; `None` when the epoch
    /// failed before its start ended.
    pub engine: Option<Box<Engine>>,
}

/// A failure that holds no epoch: one before the control thread or the start ran, or of the
/// orderly close, which consumed the epoch.
impl From<anyhow::Error> for Failed {
    fn from(error: anyhow::Error) -> Self {
        Self {
            error,
            engine: None,
        }
    }
}

/// Refuses a launch that states no epoch or host, or a world whose inherited sockets disagree
/// with the role it states, by name rather than at the claim or the first collective.
fn check(launch: &EngineArgs) -> Result<()> {
    if launch.epoch == 0 {
        bail!("the launch file states epoch 0, which names no epoch for this process to serve");
    }
    if launch.host.is_empty() {
        bail!("the launch file states no host identity, which the storage claim is keyed by");
    }
    match &launch.world {
        Some(world) if world.peers == 0 => {
            bail!("the launch file states a world of several hosts with no peer main process")
        }
        Some(world) if launch.scheduler_node != world.inherited.is_some() => bail!(
            "the scheduler node's main process, and only it, inherits the world's reserved \
             sockets: this one is{} the scheduler node",
            if launch.scheduler_node { "" } else { " not" }
        ),
        None if !launch.scheduler_node => {
            bail!("a main process that holds every rank of its group holds its schedulers too")
        }
        _ => Ok(()),
    }
}

/// Runs the control thread over `requests`, the daemon's request connection after [`connect`],
/// and `process`, its process interface, with the engine `start` builds from `launch`, until the
/// daemon shuts the engine down or closes both interfaces.
///
/// # Errors
///
/// When the control thread cannot start, or the epoch fails: the start, a scheduler, a device, a
/// stall or a lost peer. The started epoch is then handed back as the failure left it, without
/// its orderly close ([`Failed`]).
pub fn serve(
    requests: Connection,
    window: NonZeroU32,
    process: Connection,
    launch: EngineArgs,
    start: Start,
) -> std::result::Result<(), Failed> {
    MainProcess::new(requests, window, process, launch)?.serve(start, Vec::new())
}

/// The control thread's state, touched on that thread only.
pub(crate) struct MainProcess {
    /// The control thread's inbox, and the sender every reader, job and callback is given.
    inbox: (Sender<Input>, Receiver<Input>),
    /// The write half of each daemon front's request connection, by front: the launching
    /// daemon's first, then each that attached; `None` once it closed.
    fronts: Vec<Option<Connection>>,
    /// The event window each front stated in its welcome, by front: a request it submits is sent
    /// no more events ahead of its consumption reports.
    windows: Vec<NonZeroU32>,
    /// The write half of the process interface; `None` once the daemon closed it.
    process: Option<Connection>,
    launch: EngineArgs,
    lifecycle: Lifecycle,
    update: Update,
    /// The generation that executes, and its packages (the kernels' and, under
    /// [`policy_slot`], the policy's); the packages a trial keeps; and a prepared generation's
    /// packages with its policy module and arguments.
    generation: u64,
    serving: Packages,
    kept: Option<Packages>,
    prepared: Option<(Packages, Arc<wasmtime::Module>, Parameters)>,
    /// The started epoch; `None` while starting.
    engine: Option<Engine>,
    /// The admitted requests' streams, by the engine's identity of each request.
    streams: HashMap<RequestId, RequestStream>,
    /// The engine's identity of each admitted request, by its stream's front and the identity
    /// that front gives it: each front numbers its own requests.
    named: HashMap<(usize, RequestId), RequestId>,
    /// The engine identity the next admitted request takes.
    next_request: u64,
    /// The requests the pause gate holds, in arrival order, each with its front and the media of
    /// its conditions.
    held: VecDeque<(usize, wire::Request, Vec<Arc<proto::MappedMedia>>)>,
    paused: bool,
    /// Whether requests carried from another epoch compute their state again
    /// (`Promote.recompute`).
    recompute: bool,
    /// Whether admission opened once and the serving path's warm-up ran.
    warmed: bool,
    /// A grow that arrived while the engine could not apply it (`lifecycle::grow`): during its
    /// start, while it is parked, or during a drain; applied once a promotion makes the engine
    /// serve.
    grow: bool,
    operations: Sender<Job>,
    watchdog: Watchdog,
    /// The metrics' running sums since the start, and each scheduler's lifetime counts as its
    /// last statistics stated them (steps, batches, policy calls, policy fuel).
    metrics: wire::Metrics,
    lifetime: Vec<[u64; 4]>,
    next_metrics: Instant,
    /// What the engine last announced it serves, which an attaching daemon is told.
    announced: Option<wire::EngineInfo>,
    /// Why the control thread ends: a shutdown or the daemon gone (`Ok`), or a failure.
    ending: Option<Result<()>>,
}

impl MainProcess {
    /// The control thread over `requests` and `process`, its readers and its operations thread
    /// started.
    fn new(
        requests: Connection,
        window: NonZeroU32,
        process: Connection,
        launch: EngineArgs,
    ) -> Result<Self> {
        let inbox = crossbeam_channel::unbounded();
        let front = requests
            .writer()
            .context("a write half of the request connection")?;
        let writer = process
            .writer()
            .context("a write half of the process interface")?;
        daemon::read_requests(0, requests, inbox.0.clone())?;
        daemon::read_commands(process, inbox.0.clone())?;

        let (operations, jobs) = crossbeam_channel::unbounded::<Job>();
        let done = inbox.0.clone();
        std::thread::Builder::new()
            .name("operations".to_owned())
            .spawn(move || {
                for job in jobs {
                    if done.send(job()).is_err() {
                        return;
                    }
                }
            })
            .context("starting the operations thread")?;

        // The module set's kernels are candidates, every package installed for an operator; the
        // kernels the first generation serves are those its start selected, added once it ends
        // (`started`).
        let serving = Packages::from([(policy_slot(), launch.modules.policy.clone())]);
        let report_period = launch.engine_config.worker.report_period;
        Ok(Self {
            inbox,
            fronts: vec![Some(front)],
            windows: vec![window],
            process: Some(writer),
            watchdog: Watchdog::new(launch.engine_config.worker.watchdog_timeout)?,
            launch,
            lifecycle: Lifecycle::Starting,
            update: Update::Steady,
            // A start installs the epoch's module set as its first generation, with nothing kept.
            generation: ranks::FIRST_GENERATION,
            serving,
            kept: None,
            prepared: None,
            engine: None,
            streams: HashMap::new(),
            named: HashMap::new(),
            next_request: 1,
            held: VecDeque::new(),
            paused: false,
            recompute: false,
            warmed: false,
            grow: false,
            operations,
            metrics: wire::Metrics::default(),
            lifetime: Vec::new(),
            next_metrics: Instant::now() + report_period,
            announced: None,
            ending: None,
        })
    }

    /// What the start's peer links call once one closes: it tells the control thread.
    fn link_closed(&self) -> crate::worker::completion::Waker {
        let inbox = self.inbox.0.clone();
        Arc::new(move || {
            let reason = "a peer link of this epoch closed".to_owned();
            let _ = inbox.send(Input::LinkClosed(reason));
        })
    }

    /// Starts `start` on its thread, serves `filed` (commands the claim read first), and then
    /// serves what arrives until the control thread ends.
    fn serve(
        mut self,
        start: Start,
        filed: Vec<proto::process::Command>,
    ) -> std::result::Result<(), Failed> {
        let inbox = self.inbox.0.clone();
        let launch = self.launch.clone();
        std::thread::Builder::new()
            .name("start".to_owned())
            .spawn(move || {
                let started = start(&launch);
                let _ = inbox.send(done(move |main| main.started(started)));
            })
            .context("starting the rank start's thread")?;
        for command in filed {
            self.command(command);
        }

        loop {
            if let Some(ending) = self.ending.take() {
                return self.end(ending);
            }

            // The wait ends at the watchdog's deadline while a job is in flight, and at the
            // metrics second otherwise.
            let deadline = self
                .watchdog
                .deadline(&self.in_flight())
                .unwrap_or(self.next_metrics)
                .min(self.next_metrics);
            let inbox = self.inbox.1.clone();
            let outputs = self.outputs();
            let mut select = Select::new();
            select.recv(&inbox);
            for output in &outputs {
                select.recv(output);
            }
            if let Ok(ready) = select.select_deadline(deadline) {
                match ready.index() {
                    0 => {
                        // The control thread holds a sender of its own inbox: it never closes.
                        if let Ok(input) = ready.recv(&inbox) {
                            self.input(input);
                        }
                    }
                    index => {
                        let scheduler = index - 1;
                        match ready.recv(&outputs[scheduler]) {
                            Ok(output) => self.output(scheduler, output),
                            Err(_) => self.fail(
                                format!("scheduler {scheduler} stopped"),
                                lifecycle::fault(
                                    proto::process::FaultKind::Scheduler,
                                    "its thread ended".to_owned(),
                                    None,
                                ),
                            ),
                        }
                    }
                }
            }
            self.watch(Instant::now());
        }
    }

    /// Applies one input of the inbox.
    fn input(&mut self, input: Input) {
        match input {
            Input::Frame(front, frame, media) => match frame.kind {
                Some(wire::to_engine::Kind::Submit(submit)) => self.submit(front, submit, media),
                Some(wire::to_engine::Kind::Command(command)) => {
                    self.request_command(front, command);
                }
                Some(wire::to_engine::Kind::Engine(command)) => {
                    self.engine_command(front, command);
                }
                // A condition's parts are the front's reader's, which hands the medium on with
                // the `Submit` that follows them.
                Some(wire::to_engine::Kind::ConditionPart(_)) | None => {}
            },
            Input::FrontClosed(front) => {
                tracing::warn!(
                    front,
                    "a daemon front closed; its requests wait for a takeover"
                );
                if let Some(slot) = self.fronts.get_mut(front)
                    && let Some(connection) = slot.take()
                {
                    connection.shutdown();
                }
                self.end_when_unserved();
            }
            Input::Command(Some(command)) => self.command(command),
            Input::Command(None) => {
                tracing::warn!("the daemon closed this engine's process interface");
                self.process = None;
                self.end_when_unserved();
            }
            Input::Faulted(rank) => self.faulted(rank),
            Input::LinkClosed(reason) => {
                let liveness = self
                    .engine
                    .as_ref()
                    .and_then(|engine| engine.liveness.as_ref());
                // The scheduler node's orderly end: this peer ends in order with it.
                if liveness.is_some_and(|liveness| liveness.closed() && liveness.lost().is_none()) {
                    tracing::warn!("the scheduler node ended the epoch: this peer ends with it");
                    self.ending.get_or_insert(Ok(()));
                    return;
                }
                let reason = liveness
                    .and_then(|liveness| liveness.lost())
                    .unwrap_or(reason);
                let fault =
                    lifecycle::fault(proto::process::FaultKind::ProcessExit, reason.clone(), None);
                self.fail(
                    format!("a main process of this epoch was lost: {reason}"),
                    fault,
                );
            }
            Input::Done(apply) => apply(self),
        }
    }

    /// Ends the control thread in order once no daemon serves it any more: the process interface
    /// and every front closed. A hot restart closes the launching daemon's interfaces and leaves
    /// the attached front serving.
    fn end_when_unserved(&mut self) {
        if self.process.is_none() && self.fronts.iter().all(Option::is_none) {
            self.ending.get_or_insert(Ok(()));
        }
    }

    /// Applies one output of scheduler `scheduler`.
    fn output(&mut self, scheduler: usize, output: Output) {
        match output {
            Output::Events(events) => self.events(events),
            Output::Stats(stats) => self.stats(scheduler, stats),
            Output::PolicyRolledBack(reason) => {
                self.policy_rolled_back(Some(scheduler), reason, false);
            }
            Output::NumericalFault(batch, requests) => self.numerical_fault(batch, requests),
            Output::Shrunk(region, units) => self.shrunk(region, units),
            Output::Failed(error) => {
                let reason = format!("scheduler {scheduler} failed: {error}");
                self.fail(
                    reason,
                    lifecycle::fault(
                        proto::process::FaultKind::Scheduler,
                        error.to_string(),
                        None,
                    ),
                );
            }
        }
    }

    /// Each scheduler's channel out, in the order of the session's schedulers; none while
    /// starting.
    fn outputs(&self) -> Vec<Receiver<Output>> {
        self.engine
            .iter()
            .flat_map(|engine| &engine.schedulers)
            .map(|scheduler| scheduler.outputs.clone())
            .collect()
    }

    /// Applies every output the schedulers have sent, before what arrived after them.
    fn outputs_sent(&mut self) {
        for (index, outputs) in self.outputs().into_iter().enumerate() {
            for output in outputs.try_iter() {
                self.output(index, output);
            }
        }
    }

    /// Queues `job` on the operations thread, after every job queued before it.
    fn serial(&mut self, job: Job) {
        if self.operations.send(job).is_err() {
            let reason = "the operations thread ended".to_owned();
            self.fail(
                reason.clone(),
                lifecycle::fault(proto::process::FaultKind::Operations, reason, None),
            );
        }
    }

    /// Runs `job` beside the batches, on a thread of its own.
    fn work(&mut self, job: Job) {
        let inbox = self.inbox.0.clone();
        let spawned = std::thread::Builder::new()
            .name("work".to_owned())
            .spawn(move || {
                let _ = inbox.send(job());
            });
        if let Err(error) = spawned {
            let reason = format!("a job beside the batches could not start: {error}");
            self.fail(
                reason.clone(),
                lifecycle::fault(proto::process::FaultKind::Work, reason, None),
            );
        }
    }

    /// Writes one frame to front `front`, with the artifact bytes it travels with. A front that
    /// closed reads nothing; its streams wait for a takeover.
    fn send(
        &mut self,
        front: usize,
        kind: from_engine::Kind,
        media: Option<&Arc<dyn ArtifactMedia>>,
    ) {
        let Some(Some(connection)) = self.fronts.get_mut(front) else {
            return;
        };
        if let Err(error) = daemon::send(connection, kind, media) {
            tracing::warn!(%error, front, "writing to a daemon front failed");
        }
    }

    /// Queues one frame for front `front`, as [`Self::send`] writes it, writing what its socket
    /// takes now; [`Self::flush_front`] writes the rest once the frames that go together are
    /// queued.
    fn queue(
        &mut self,
        front: usize,
        kind: from_engine::Kind,
        media: Option<&Arc<dyn ArtifactMedia>>,
    ) {
        let Some(Some(connection)) = self.fronts.get_mut(front) else {
            return;
        };
        if let Err(error) = daemon::queue(connection, kind, media) {
            tracing::warn!(%error, front, "writing to a daemon front failed");
        }
    }

    /// Writes every frame queued for front `front`, waiting while its socket is full.
    fn flush_front(&mut self, front: usize) {
        let Some(Some(connection)) = self.fronts.get_mut(front) else {
            return;
        };
        if let Err(error) = connection.flush() {
            tracing::warn!(%error, front, "writing to a daemon front failed");
        }
    }

    /// Ends the control thread: in order after a shutdown or once no daemon serves it; without
    /// its orderly close after a failure, handing the engine to the caller as it is, since a
    /// device or a collective may hold every rank ([`Failed`]).
    fn end(mut self, ending: Result<()>) -> std::result::Result<(), Failed> {
        if let Err(error) = ending {
            return Err(Failed {
                error,
                engine: self.engine.take().map(Box::new),
            });
        }

        self.lifecycle = Lifecycle::Stopped;
        self.refuse_held("the engine shut down");
        self.metrics();
        // Every scheduler ends the requests it holds as aborted and stops, and their events are
        // written before the engine states that it stopped.
        let closed = self.engine.take().map_or(Ok(()), |engine| {
            engine.close(|_, output| {
                if let Output::Events(events) = output {
                    self.events(events);
                }
            })
        });
        self.announce(false);
        closed?;

        for connection in self.fronts.iter().flatten() {
            connection.shutdown();
        }
        if let Some(process) = &self.process {
            process.shutdown();
        }
        Ok(())
    }

    /// The start ended: the engine is installed, and parks or opens admission.
    fn started(&mut self, started: Result<Engine>) {
        let engine = match started {
            Ok(engine) => engine,
            Err(error) => {
                let reason = format!("{error:#}");
                let fault =
                    lifecycle::fault(proto::process::FaultKind::Start, reason.clone(), None);
                return self.fail(format!("the start failed: {reason}"), fault);
            }
        };
        self.serving.extend(engine.kernels.clone());
        // Each worker states its device's fault as it finds one, at once for a fault found
        // during the start, and the control thread fails the epoch then.
        for (rank, worker) in engine.workers.iter().enumerate() {
            let inbox = self.inbox.0.clone();
            let faulted: crate::worker::completion::Waker = Arc::new(move || {
                let _ = inbox.send(Input::Faulted(rank));
            });
            worker.on_fault(&faulted);
        }
        self.engine = Some(engine);
        // An engine that failed while it started is left as it is when the control thread ends.
        if self.lifecycle != Lifecycle::Starting {
            return;
        }

        if !self.launch.parked() {
            return self.open_admission();
        }
        // A standby gives its graph pools' storage back while it parks, as its failure domain
        // resolved.
        if self.launch.standby_storage == proto::request::StandbyStorage::Suspended
            && let Some(engine) = self.engine.as_ref()
        {
            let binding = Arc::clone(&engine.binding);
            return self.serial(Box::new(move || {
                let suspended = binding.suspend_graphs();
                done(move |main| match suspended {
                    Ok(()) => main.park(),
                    Err(error) => {
                        let reason = format!("suspending the graph storage: {error}");
                        main.fail(
                            reason.clone(),
                            lifecycle::fault(proto::process::FaultKind::Start, reason, None),
                        );
                    }
                })
            }));
        }
        self.park();
    }

    /// The start ends parked. A grow kept during the start waits for the promotion.
    fn park(&mut self) {
        self.lifecycle = lifecycle::started(true);
        self.announce(true);
        self.report(lifecycle::parked(self.launch.engine, self.generation));
    }

    /// Opens admission: the graph storage restored, what serving holds reserved, the regions
    /// resized to what the epoch states of them and every scheduler resumed on the operations
    /// thread; then the warm-up on the first opening, and `Serving`.
    fn open_admission(&mut self) {
        let Some(engine) = self.engine.as_ref() else {
            return;
        };
        let schedulers = engine.inboxes();
        let head = engine.head.clone();
        let binding = Arc::clone(&engine.binding);
        self.serial(Box::new(move || {
            let opened = epoch::promote(&schedulers, head.as_deref(), &binding);
            done(move |main| main.opened(opened))
        }));
    }

    fn opened(&mut self, opened: Result<()>) {
        if let Err(error) = opened {
            let reason = format!("opening admission: {error:#}");
            return self.fail(
                reason.clone(),
                lifecycle::fault(proto::process::FaultKind::Promote, reason, None),
            );
        }
        if !matches!(self.lifecycle, Lifecycle::Starting | Lifecycle::Parked) {
            return;
        }

        if !self.warmed {
            self.warm_up();
            self.warmed = true;
        }
        self.lifecycle = Lifecycle::Serving;
        self.announce(true);
        self.report(lifecycle::serving(self.launch.engine, self.generation));
        self.release();
        self.grow_kept();
    }

    /// One command of the process interface.
    fn command(&mut self, command: proto::process::Command) {
        let Some(kind) = command.kind else {
            return;
        };
        tracing::info!(
            command = ?kind,
            lifecycle = ?self.lifecycle,
            update = ?self.update,
            "a command of the process interface"
        );
        match kind {
            command::Kind::Prepare(prepare) => self.prepare(prepare),
            command::Kind::Switch(switch) => self.switch(switch.generation),
            command::Kind::Commit(_) => self.commit(),
            command::Kind::Rollback(rollback) => {
                let reason = if rollback.reason.is_empty() {
                    "the daemon rolled the update back".to_owned()
                } else {
                    rollback.reason
                };
                match lifecycle::rollback(self.lifecycle, &mut self.update, &reason) {
                    Ok((generation, executed)) => self.roll_back(generation, executed, reason),
                    Err(report) => self.report(report),
                }
            }
            command::Kind::Drain(drain) => self.drain(drain),
            command::Kind::Promote(promote) => {
                self.recompute = promote.recompute;
                let serving = lifecycle::serving(self.launch.engine, self.generation);
                match lifecycle::promote(&mut self.lifecycle, serving) {
                    // The grow is kept until the engine serves: behind the opening when it is
                    // parked, once it serves again when it drains.
                    Ok(admits) => {
                        if admits {
                            self.open_admission();
                        }
                        if promote.grow {
                            self.grow();
                        }
                    }
                    Err(report) => self.report(report),
                }
            }
            command::Kind::Pause(pause) => {
                self.paused = pause.paused;
                self.release();
            }
            command::Kind::Grow(_) => self.grow(),
        }
    }

    /// `Prepare`: the next generation is made ready beside the one that executes, on a thread
    /// of its own: its kernels selected, installed and prepared on every worker, its policy
    /// instantiated once against the ceilings, and, when a kernel package changed, a shadow pass
    /// of every scheduler.
    fn prepare(&mut self, prepare: proto::process::Prepare) {
        let generation = match lifecycle::prepare(self.lifecycle, &mut self.update, self.generation)
        {
            Ok(generation) => generation,
            Err(report) => return self.report(report),
        };
        let parts = match generation::named(prepare) {
            Ok((packages, policy)) => self
                .engine
                .as_ref()
                .and_then(|engine| Some((engine, engine.build.clone()?)))
                .map(|(engine, build)| (engine, build, packages, policy))
                .ok_or_else(|| {
                    "this engine holds no Build Server to resolve modules at".to_owned()
                }),
            Err(error) => Err(format!("{error:#}")),
        };
        let (engine, build, named, policy) = match parts {
            Ok(parts) => parts,
            Err(reason) => {
                lifecycle::prepare_ended(&mut self.update, generation);
                return self.report(lifecycle::prepare_failed(reason));
            }
        };

        let shadows = engine.shadows();
        let workers = engine.workers.clone();
        let binding = Arc::clone(&engine.binding);
        let ceilings = engine.ceilings;
        let budget = self.launch.engine_config.scheduler.policy_budget;
        let serving = kernels_of(&self.serving);
        // A preparation whose shadow pass no running request reaches within the epoch's
        // comparison bound is abandoned, naming the operators it could not compare.
        let bound = self.launch.engine_config.worker.compare_timeout;
        self.work(Box::new(move || {
            let prepared = generation::kernels(&workers, &build, &named, &serving, generation)
                .and_then(|kernels| {
                    let (module, parameters) =
                        generation::policy(&build.0, &policy, (&ceilings, &budget))?;
                    // Only an operator whose package changed has an old kernel to compare with.
                    // Every peer prepares the generation from its own daemon's command.
                    let changed = changed_operators(&kernels, &serving);
                    let uncompared = generation::compare(
                        &shadows, &workers, &binding, generation, &changed, bound,
                    )?;
                    let mut packages = kernels;
                    packages.insert(policy_slot(), policy);
                    Ok(((packages, module, parameters), uncompared))
                });
            done(move |main| main.prepared(generation, prepared))
        }));
    }

    #[expect(
        clippy::type_complexity,
        reason = "the result of one preparation, as its job returns it"
    )]
    fn prepared(
        &mut self,
        generation: u64,
        prepared: Result<((Packages, Arc<wasmtime::Module>, Parameters), Vec<String>)>,
    ) {
        // A peer's preparation is what the scheduler node's waits for before its shadow pass.
        if let Some(told) = self
            .engine
            .as_ref()
            .and_then(|engine| engine.prepared.as_ref())
        {
            let outcome = match &prepared {
                Ok(_) => Ok(()),
                Err(error) => Err(format!("{error:#}")),
            };
            let _ = told.send((generation, outcome));
        }

        match prepared {
            Ok((prepared, uncompared)) => {
                if !lifecycle::prepared(&mut self.update, generation) {
                    // A rollback or a drain discarded the update while it was prepared.
                    return self.discard(generation);
                }
                self.prepared = Some(prepared);
                self.report(lifecycle::report(proto::process::report::Kind::Prepared(
                    proto::process::Prepared {
                        generation,
                        uncompared,
                    },
                )));
            }
            Err(error) => {
                lifecycle::prepare_ended(&mut self.update, generation);
                self.discard(generation);
                self.report(lifecycle::prepare_failed(format!("{error:#}")));
            }
        }
    }

    /// Discards `generation`, prepared or being prepared, on the operations thread.
    fn discard(&mut self, generation: u64) {
        self.prepared = None;
        let Some(engine) = self.engine.as_ref() else {
            return;
        };
        let shadows = engine.shadows();
        let workers = engine.workers.clone();
        self.serial(Box::new(move || {
            let discarded = generation::discard(&shadows, &workers, generation);
            done(move |_| {
                if let Err(error) = discarded {
                    tracing::warn!(generation, %error, "a discarded generation was not released");
                }
            })
        }));
    }

    /// `Switch`: the prepared generation executes from one batch boundary of every scheduler,
    /// the previous one kept; the route instance switches with it.
    fn switch(&mut self, generation: u64) {
        if let Err(report) = lifecycle::switch(self.lifecycle, &mut self.update, generation) {
            return self.report(report);
        }
        let (Some(engine), Some((packages, module, parameters))) =
            (self.engine.as_ref(), self.prepared.take())
        else {
            lifecycle::switch_failed(&mut self.update, generation);
            let reason = format!("generation {generation} holds nothing prepared to switch to");
            return self.report(lifecycle::prepare_failed(reason));
        };

        // A peer's ranks switch between the scheduler node's batches, on its frame: its own
        // daemon's command is acknowledged at once.
        let policy = (module, parameters);
        if engine.upstream.is_some() {
            return self.switched(generation, packages, policy, Ok(()));
        }
        let schedulers = engine.inboxes();
        let route = engine.groups.clone();
        let binding = Arc::clone(&engine.binding);
        let served = Arc::clone(&engine.served);
        self.serial(Box::new(move || {
            let switched = generation::switch(
                &schedulers,
                route.as_deref(),
                &binding,
                &served,
                generation,
                &policy,
            );
            done(move |main| main.switched(generation, packages, policy, switched))
        }));
    }

    fn switched(
        &mut self,
        generation: u64,
        packages: Packages,
        (module, parameters): (Arc<wasmtime::Module>, Parameters),
        switched: Result<()>,
    ) {
        if let Err(error) = switched {
            if lifecycle::switch_failed(&mut self.update, generation) {
                let reason = format!("the switch to generation {generation} failed: {error:#}");
                self.report(lifecycle::rolled_back(reason));
            }
            return self.discard(generation);
        }

        self.kept = Some(std::mem::replace(&mut self.serving, packages));
        self.generation = generation;
        let snapshot = self.snapshot();
        if let Some(Err(error)) = self.engine.as_mut().and_then(|engine| {
            let ceilings = engine.ceilings;
            let route = engine.route.as_mut()?;
            Some(route.switch(module, parameters, snapshot, &ceilings))
        }) {
            // The workers execute the generation and the route instance cannot: the route
            // keeps the policy it served, and the whole generation goes back as on any fault on
            // trial.
            let reason = format!("the route instance of generation {generation} failed: {error}");
            self.policy_rolled_back(None, reason, false);
            if self.update != Update::Trial(generation) {
                return;
            }
        }
        self.report(lifecycle::report(proto::process::report::Kind::Switched(
            proto::process::Switched { generation },
        )));
    }

    /// The snapshot a route instance starts from: the schedulers and their requests in flight.
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            ranks: self.ranks(),
            ..Snapshot::default()
        }
    }

    /// `Commit`: the generation on trial stays, and the kept one is released after its last
    /// batch.
    fn commit(&mut self) {
        let generation = match lifecycle::commit(&mut self.update, self.generation) {
            Ok(generation) => generation,
            Err(report) => return self.report(report),
        };
        let Some(engine) = self.engine.as_ref() else {
            return;
        };
        // A peer's ranks commit on the scheduler node's frame.
        if engine.upstream.is_some() {
            return self.committed(generation, Ok(()));
        }
        let schedulers = engine.inboxes();
        let route = engine.groups.clone();
        self.serial(Box::new(move || {
            let committed = generation::commit(&schedulers, route.as_deref(), generation);
            done(move |main| main.committed(generation, committed))
        }));
    }

    fn committed(&mut self, generation: u64, committed: Result<()>) {
        if let Err(error) = committed {
            let reason = format!("the commit of generation {generation} failed: {error:#}");
            return self.fail(
                reason.clone(),
                lifecycle::fault(proto::process::FaultKind::Commit, reason, None),
            );
        }
        self.kept = None;
        if let Some(route) = self
            .engine
            .as_mut()
            .and_then(|engine| engine.route.as_mut())
        {
            route.commit();
        }
        self.report(lifecycle::committed());
    }

    /// Leaves `generation` for `reason`: switched back at a batch boundary when it `executed`,
    /// discarded otherwise; acknowledged with `RolledBack` either way.
    fn roll_back(&mut self, generation: u64, executed: bool, reason: String) {
        if !executed {
            self.discard(generation);
            return self.report(lifecycle::rolled_back(reason));
        }
        let Some(engine) = self.engine.as_ref() else {
            return;
        };
        // The trial kept the generation before it: generations are numbered one past the one
        // that executed when each was prepared.
        let kept = generation - 1;
        // A peer's ranks roll back between the scheduler node's batches, on its frame.
        if engine.upstream.is_some() {
            return self.rolled_back(kept, reason, Ok(()));
        }
        let schedulers = engine.inboxes();
        let route = engine.groups.clone();
        let binding = Arc::clone(&engine.binding);
        let served = Arc::clone(&engine.served);
        self.serial(Box::new(move || {
            let rolled = generation::rollback(
                &schedulers,
                route.as_deref(),
                &binding,
                &served,
                generation,
                kept,
            );
            done(move |main| main.rolled_back(kept, reason, rolled))
        }));
    }

    /// A switch back to `kept` ended.
    fn rolled_back(&mut self, kept: u64, reason: String, rolled: Result<()>) {
        if let Err(error) = rolled {
            let reason = format!("the rollback to generation {kept} failed: {error:#}");
            return self.fail(
                reason.clone(),
                lifecycle::fault(proto::process::FaultKind::Rollback, reason, None),
            );
        }
        // A switch that never took effect kept nothing, and nothing returns.
        if let Some(packages) = self.kept.take() {
            self.serving = packages;
            self.generation = kept;
            let snapshot = self.snapshot();
            if let Some(Err(error)) = self
                .engine
                .as_mut()
                .and_then(|engine| engine.route.as_mut())
                .map(|route| route.rollback(snapshot))
            {
                tracing::warn!(%error, "the route instance kept no policy to roll back to");
            }
        }
        self.report(lifecycle::rolled_back(reason));
    }

    /// `Drain`: every scheduler admits nothing more and steps its requests in flight until each
    /// ends by itself or the deadline ends it as migratable; an update no switch took is
    /// discarded. The drain begins on the operations thread, behind every job accepted before it
    /// that pauses the schedulers, whose `Resume` would end it; a thread of its own then awaits
    /// every scheduler's reply and queues the drain's end on the operations thread, behind the
    /// releases of the `Shrunk`s that preceded it.
    fn drain(&mut self, drain: proto::process::Drain) {
        match lifecycle::drain(&mut self.lifecycle, drain.epoch) {
            Ok(true) => {}
            Ok(false) => return,
            Err(report) => return self.report(report),
        }
        if let Some(generation) = lifecycle::discarded_by_drain(&mut self.update) {
            self.discard(generation);
            let reason =
                format!("the engine drained before generation {generation} was switched to");
            self.report(lifecycle::rolled_back(reason));
        }
        // The daemon's client reads this before the first request ends migratable, on the same
        // connection, so a request that moves never finds this engine serving.
        self.announce(false);
        self.refuse_held("the engine drains and admits nothing");

        let Some(engine) = self.engine.as_ref() else {
            return;
        };
        let deadline = Instant::now() + Duration::from_millis(drain.deadline_ms);
        let schedulers = engine.inboxes();
        let (epoch, shrink) = (drain.epoch, drain.shrink);
        self.serial(Box::new(move || {
            let replies = epoch::drain(&schedulers, deadline, shrink);
            done(move |main| main.draining(epoch, replies))
        }));
    }

    /// Every scheduler was told to drain, and `replies` arrive once each drained.
    fn draining(&mut self, epoch: u64, replies: Result<Vec<Receiver<()>>>) {
        let replies = match replies {
            Ok(replies) => replies,
            Err(error) => {
                let reason = format!("draining: {error:#}");
                return self.fail(
                    reason.clone(),
                    lifecycle::fault(proto::process::FaultKind::Drain, reason, None),
                );
            }
        };
        self.work(Box::new(move || {
            // A scheduler that stopped drops its reply; its `Failed` output says why.
            for reply in replies {
                let _ = reply.recv();
            }
            done(move |main| {
                main.outputs_sent();
                main.serial(Box::new(move || done(move |main| main.drained(epoch))));
            })
        }));
    }

    fn drained(&mut self, epoch: u64) {
        if !matches!(self.lifecycle, Lifecycle::Draining { .. }) {
            return;
        }
        let serves_again = lifecycle::drained(&mut self.lifecycle, &mut self.grow);
        self.metrics();
        self.report(lifecycle::drained_report(epoch));
        self.announce(false);
        if serves_again {
            self.open_admission();
        }
    }

    /// A scheduler's region `region` shrank to `units`: the segments it no longer covers are
    /// withdrawn and released on the operations thread on every rank that maps it, the peers'
    /// included, and the daemon is told once per host whose devices gave memory back.
    fn shrunk(&mut self, region: u8, units: u32) {
        // An engine with no session head of its own holds no segment to release.
        let Some((engine, head)) = self
            .engine
            .as_ref()
            .and_then(|engine| Some((engine, engine.head.clone()?)))
        else {
            return;
        };
        let binding = Arc::clone(&engine.binding);
        self.serial(Box::new(move || {
            let released = epoch::release(&head, &binding, region, units);
            done(move |main| match released {
                // One report per host whose quotas the release freed memory in: the daemon has
                // the target engines with ranks there grow into it.
                Ok((devices, bytes)) => {
                    let launch = &main.launch;
                    let hosts =
                        epoch::by_host(&launch.engine_config.parallel, &launch.host, devices);
                    for (host, devices) in hosts {
                        let shrunk = proto::process::Shrunk {
                            devices,
                            bytes,
                            host,
                        };
                        main.report(lifecycle::report(proto::process::report::Kind::Shrunk(
                            shrunk,
                        )));
                    }
                }
                Err(error) => {
                    let reason = format!("releasing region {region} past unit {units}: {error:#}");
                    main.fail(
                        reason.clone(),
                        lifecycle::fault(proto::process::FaultKind::Release, reason, None),
                    );
                }
            })
        }));
    }

    /// `Grow`, or a promotion with `grow`: the ranks create the segments their quotas now hold,
    /// the head grants them, and the schedulers grow their regions into them. While the engine
    /// starts, is parked or drains the grow is kept instead ([`lifecycle::grow`]).
    fn grow(&mut self) {
        if !lifecycle::grow(self.lifecycle, &mut self.grow) {
            return;
        }
        let (schedulers, head, binding) = match self.segment_parts() {
            Ok(parts) => parts,
            Err(error) => {
                tracing::warn!(error = %format!("{error:#}"), "a grow changes nothing");
                return;
            }
        };
        self.serial(Box::new(move || {
            let grown = epoch::grow(&schedulers, &head, &binding);
            done(move |_| {
                if let Err(error) = grown {
                    tracing::warn!(error = %format!("{error:#}"), "growing the cache failed");
                }
            })
        }));
    }

    /// Applies the grow kept while the engine could not grow, once it serves.
    fn grow_kept(&mut self) {
        if std::mem::take(&mut self.grow) {
            self.grow();
        }
    }

    /// One command of the request interface for the whole engine.
    fn engine_command(&mut self, front: usize, command: wire::EngineCommand) {
        use engine_command::Kind;

        let Some(kind) = command.kind else {
            return;
        };
        tracing::info!(front, command = ?kind, "an engine command");
        match kind {
            Kind::Shutdown(_) => {
                self.ending.get_or_insert(Ok(()));
            }
            Kind::Attach(attach) => {
                // The attach waits on another daemon, which welcomes this engine or closes the
                // connection; it waits on a thread of its own, so that no operation waits behind
                // it.
                let engine = self.launch.engine;
                let inbox = self.inbox.0.clone();
                let attaching =
                    std::thread::Builder::new()
                        .name("attach".to_owned())
                        .spawn(move || {
                            let attached = daemon::attach(&attach.requests, engine);
                            let _ = inbox.send(done(move |main| {
                                main.attached(&attach.requests, attached);
                            }));
                        });
                if let Err(error) = attaching {
                    tracing::warn!(%error, "an attach did not start");
                }
            }
            Kind::PrepareExperts(prepare) => {
                let orders = prepare
                    .order
                    .into_iter()
                    .map(|(module, order)| (module, order.experts))
                    .collect();
                self.on_ranks(
                    front,
                    prepare.id,
                    Ok(RankOperation::PrepareExperts { orders }),
                );
            }
            Kind::StartExperts(start) => {
                self.on_ranks(front, start.id, Ok(RankOperation::StartExperts));
            }
            Kind::AbortExperts(abort) => {
                self.on_ranks(front, abort.id, Ok(RankOperation::AbortExperts));
            }
            Kind::LoadAdapter(load) => {
                let operation = RankOperation::LoadAdapter {
                    slot: load.slot,
                    path: PathBuf::from(load.path),
                };
                self.on_ranks(front, load.id, Ok(operation));
            }
            Kind::UnloadAdapter(unload) => {
                let operation = RankOperation::UnloadAdapter { slot: unload.slot };
                self.on_ranks(front, unload.id, Ok(operation));
            }
            Kind::StartProfile(start) => {
                let operation = start
                    .session
                    .as_ref()
                    .ok_or_else(|| "a profiling start states no session".to_owned())
                    .and_then(proto::profile::SessionStart::try_from)
                    .map(|session| RankOperation::StartProfile(Box::new(session)));
                self.on_ranks(front, start.id, operation);
            }
            Kind::StopProfile(stop) => {
                let operation = RankOperation::StopProfile {
                    session: stop.session,
                    reason: stop.reason,
                };
                self.on_ranks(front, stop.id, Ok(operation));
            }
            Kind::ReleaseProfile(release) => {
                let operation =
                    proto::profile::Collector::from_wire(release.collector).map(|collector| {
                        RankOperation::ReleaseProfile {
                            session: release.session,
                            collector,
                            reason: release.reason,
                        }
                    });
                self.on_ranks(front, release.id, operation);
            }
            Kind::SwitchFences(switch) => {
                let operation = RankOperation::SwitchFences {
                    enabled: switch.enabled,
                    off: switch.off,
                };
                self.on_ranks(front, switch.id, Ok(operation));
            }
            Kind::ResizeCache(resize) => self.resize_cache(front, &resize),
            Kind::UpdateWeights(update) => self.update_weights(front, update),
            Kind::CommitExperts(commit) => {
                let order: BTreeMap<String, Vec<u32>> = commit
                    .order
                    .into_iter()
                    .map(|(module, order)| (module, order.experts))
                    .collect();
                if let Some(refusal) = self.lifecycle.boundary_refusal() {
                    return self.operated(front, commit.id, Err(anyhow::anyhow!(refusal)));
                }
                let Some(engine) = self.engine.as_ref() else {
                    return self.operated(front, commit.id, Err(starting()));
                };
                let schedulers = engine.inboxes();
                let binding = Arc::clone(&engine.binding);
                self.serial(Box::new(move || {
                    let reports = operations::commit_experts(&schedulers, &binding, &order)
                        .map(|reports| Some(weight_reports(reports)));
                    done(move |main| main.operated(front, commit.id, reports))
                }));
            }
            Kind::FlushCache(_) => {
                if let Some(refusal) = self.lifecycle.boundary_refusal() {
                    tracing::warn!(refusal, "a flush is refused");
                    return;
                }
                let Some(engine) = self.engine.as_ref() else {
                    return;
                };
                let schedulers = engine.inboxes();
                self.serial(Box::new(move || {
                    let flushed = operations::flush(&schedulers);
                    done(move |_| {
                        if let Err(error) = flushed {
                            tracing::warn!(error = %format!("{error:#}"), "a flush failed");
                        }
                    })
                }));
            }
        }
    }

    /// Runs `operation`, an operation the daemon commands on every rank, beside the batches, and
    /// answers command `id` of front `front` with every rank's reports; an operation the command
    /// could not state is refused, and so is one while the engine starts.
    fn on_ranks(
        &mut self,
        front: usize,
        id: u64,
        operation: std::result::Result<RankOperation, String>,
    ) {
        let operation = match operation {
            Ok(operation) => operation,
            Err(refusal) => return self.operated(front, id, Err(anyhow::anyhow!(refusal))),
        };
        let Some(binding) = self
            .engine
            .as_ref()
            .map(|engine| Arc::clone(&engine.binding))
        else {
            return self.operated(front, id, Err(starting()));
        };
        self.work(Box::new(move || {
            let reports = operations::on_ranks(&binding, &operation).map(rank_reports);
            done(move |main| main.operated(front, id, reports))
        }));
    }

    /// An attach ended: the daemon at `address` is a further front, told what this engine is.
    fn attached(&mut self, address: &str, attached: Result<(Connection, NonZeroU32)>) {
        let (connection, window) = match attached {
            Ok(attached) => attached,
            Err(error) => {
                // A failed hot restart, not this engine's end: the fronts it has serve on.
                tracing::warn!(daemon = address, error = %format!("{error:#}"), "an attach failed");
                return;
            }
        };
        let front = self.fronts.len();
        let started = connection
            .writer()
            .map_err(anyhow::Error::from)
            .and_then(|writer| {
                daemon::read_requests(front, connection, self.inbox.0.clone())?;
                Ok(writer)
            });
        match started {
            Ok(writer) => {
                self.fronts.push(Some(writer));
                self.windows.push(window);
            }
            Err(error) => {
                tracing::warn!(daemon = address, error = %format!("{error:#}"), "an attach failed");
                return;
            }
        }
        let state = wire::State {
            state: self.lifecycle.state() as i32,
            info: self.announced.clone(),
            error: String::new(),
        };
        self.send(front, from_engine::Kind::State(state), None);
        tracing::info!(daemon = address, front, "a daemon attached to this engine");
    }

    /// `ResizeCache`: a sleep, a wake, a growth or room made for a profiling session, each on the
    /// operations thread, answered with `Hooked`.
    fn resize_cache(&mut self, front: usize, resize: &wire::ResizeCache) {
        let id = resize.id;
        let change = wire::CacheChange::try_from(resize.change);
        if matches!(
            change,
            Ok(wire::CacheChange::Sleep | wire::CacheChange::Wake)
        ) && let Some(refusal) = self.lifecycle.boundary_refusal()
        {
            return self.operated(front, id, Err(anyhow::anyhow!(refusal)));
        }
        let (schedulers, head, binding) = match self.segment_parts() {
            Ok(parts) => parts,
            Err(error) => return self.operated(front, id, Err(error)),
        };

        // A change of the cache reports nothing beyond its success.
        let job: Job = match change {
            Ok(wire::CacheChange::Sleep) => Box::new(move || {
                let slept = operations::sleep(&schedulers, &head, &binding);
                done(move |main| main.operated(front, id, slept.map(|()| None)))
            }),
            Ok(wire::CacheChange::Wake) => Box::new(move || {
                let woken = operations::wake(&schedulers, &head, &binding);
                done(move |main| main.operated(front, id, woken.map(|_| None)))
            }),
            Ok(wire::CacheChange::Grow) => Box::new(move || {
                let grown = epoch::grow(&schedulers, &head, &binding);
                done(move |main| main.operated(front, id, grown.map(|_| None)))
            }),
            Ok(wire::CacheChange::MakeRoom) => {
                let room = resize.room_bytes;
                let devices = match device_bytes(&resize.device_bytes) {
                    Ok(devices) => devices,
                    Err(error) => return self.operated(front, id, Err(error)),
                };
                Box::new(move || {
                    let made = operations::make_room(&schedulers, &head, &binding, room, &devices);
                    done(move |main| main.operated(front, id, made.map(|()| None)))
                })
            }
            Ok(wire::CacheChange::Unspecified) | Err(_) => {
                let error = anyhow::anyhow!("a cache change that states no change");
                return self.operated(front, id, Err(error));
            }
        };
        self.serial(job);
    }

    /// `UpdateWeights`: the weights are rewritten in place from another checkpoint of the same
    /// model on the operations thread, answered with each rank's report. With `abort` every
    /// request the schedulers hold ends first.
    fn update_weights(&mut self, front: usize, update: wire::UpdateWeights) {
        let id = update.id;
        let Ok(in_flight) = wire::WeightsInFlight::try_from(update.in_flight) else {
            let error = anyhow::anyhow!("unknown in-flight choice {}", update.in_flight);
            return self.operated(front, id, Err(error));
        };
        if let Some(refusal) = self.lifecycle.boundary_refusal() {
            return self.operated(front, id, Err(anyhow::anyhow!(refusal)));
        }
        let (schedulers, head, binding) = match self.segment_parts() {
            Ok(parts) => parts,
            Err(error) => return self.operated(front, id, Err(error)),
        };

        // The same model from another checkpoint: the experts keep the order they serve in.
        let target = Target {
            checkpoint: PathBuf::from(&update.model_path),
            ..binding.target()
        };
        self.serial(Box::new(move || {
            let updated =
                operations::update_weights(&schedulers, &head, &binding, target, in_flight)
                    .map(|reports| Some(weight_reports(reports)));
            done(move |main| main.operated(front, id, updated))
        }));
    }

    /// What a job that changes the KV planes' segments reads: each scheduler's inbox, the head of
    /// the session, which states each change of them, and the weight binding.
    ///
    /// # Errors
    ///
    /// While the engine starts, or for an engine that holds no session head of its own.
    #[expect(
        clippy::type_complexity,
        reason = "the three parts a segment job takes, each its own owner's type"
    )]
    fn segment_parts(
        &self,
    ) -> Result<(
        Vec<Sender<Message>>,
        Arc<Mutex<StorageClient>>,
        Arc<crate::process::binding::WeightBinding>,
    )> {
        let engine = self.engine.as_ref().ok_or_else(starting)?;
        let head = engine.head.clone().context(
            "this engine holds no session with a Storage Server of its own, whose segments an \
             operation would change",
        )?;
        Ok((engine.inboxes(), head, Arc::clone(&engine.binding)))
    }

    /// Answers the engine command `id` of front `front` with `Hooked`: each rank's result, or why
    /// the command failed.
    fn operated(
        &mut self,
        front: usize,
        id: u64,
        reports: Result<Option<wire::operated::Reports>>,
    ) {
        let operated = match reports {
            Ok(reports) => wire::Operated {
                id,
                error: String::new(),
                reports,
            },
            Err(error) => wire::Operated {
                id,
                error: format!("{error:#}"),
                reports: None,
            },
        };
        self.send(front, from_engine::Kind::Operated(operated), None);
    }
}

/// Why an operation is refused while the engine starts.
fn starting() -> anyhow::Error {
    anyhow::anyhow!("the engine is starting and holds nothing to operate on yet")
}

/// Each rank's report of a change of its weights, as an `Operated` answer states them.
fn weight_reports(reports: Vec<WeightsUpdated>) -> wire::operated::Reports {
    wire::operated::Reports::WeightReports(wire::WeightReports {
        ranks: reports
            .into_iter()
            .map(|report| wire::WeightReport {
                group: report.group,
                rank: report.rank,
                identity: report.identity,
                identities: report.identities.iter().map(storage::Key::hex).collect(),
                moved: report.moved as u64,
            })
            .collect(),
    })
}

/// Every rank's reports of an operation the daemon commands on every rank, as an `Operated`
/// answer states them: none for an operation that reports nothing.
fn rank_reports(reports: RankReports) -> Option<wire::operated::Reports> {
    use wire::operated::Reports;

    Some(match reports {
        RankReports::None => return None,
        RankReports::ExpertAllocations(ranks) => {
            Reports::ExpertAllocations(wire::ExpertAllocations {
                ranks: ranks
                    .into_iter()
                    .map(|allocation| wire::ExpertAllocation {
                        order: allocation
                            .orders
                            .into_iter()
                            .map(|(module, experts)| (module, wire::ExpertOrder { experts }))
                            .collect(),
                        gpu: allocation.gpu.to_vec(),
                        bytes: allocation.bytes,
                    })
                    .collect(),
            })
        }
        RankReports::Adapters(ranks) => Reports::AdapterReports(wire::AdapterReports {
            ranks: ranks
                .into_iter()
                .map(|report| wire::AdapterReport {
                    slot: report.slot,
                    tensors: report.tensors,
                })
                .collect(),
        }),
        RankReports::ProfileStarts(ranks) => Reports::ProfileStarts(wire::ProfileStarts {
            ranks: ranks.iter().map(wire::ProfileStart::from).collect(),
        }),
        RankReports::ProfileStops(ranks) => Reports::ProfileStops(wire::ProfileStops {
            ranks: ranks.iter().map(wire::ProfileStop::from).collect(),
        }),
        RankReports::ProfileReleases(ranks) => Reports::ProfileReleases(wire::ProfileReleases {
            ranks: ranks.iter().map(wire::ProfileRelease::from).collect(),
        }),
        RankReports::Fences(processes) => Reports::FenceSwitches(wire::FenceSwitches {
            processes: processes
                .into_iter()
                .map(|switched| wire::FencesSwitched {
                    probes: switched.probes,
                    enabled: switched.enabled,
                    off: switched.off,
                })
                .collect(),
        }),
    })
}

/// The bytes a `MakeRoom` states for each GPU it names, by the GPU's identity.
///
/// # Errors
///
/// An identity that is not the sixteen bytes of a GPU's UUID.
fn device_bytes(devices: &[wire::DeviceBytes]) -> Result<BTreeMap<[u8; 16], u64>> {
    devices
        .iter()
        .map(|device| {
            let gpu: [u8; 16] = device.gpu.as_slice().try_into().map_err(|_| {
                anyhow::anyhow!(
                    "a GPU's identity is 16 bytes, and a room names one of {}",
                    device.gpu.len()
                )
            })?;
            Ok((gpu, device.bytes))
        })
        .collect()
}
