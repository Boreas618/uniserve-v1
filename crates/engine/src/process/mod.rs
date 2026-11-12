//! `uniserve-engine`: a node's engine process for one epoch.
//!
//! The daemon starts one engine per node and epoch from the epoch's launch file, and [`run`]
//! serves it until the daemon commands it down or it fails. The process computes nothing itself;
//! it assembles
//!
//! - the [`Control`] thread, which serves the daemon's request interface and its process
//!   interface, and blocks on its channels;
//! - one `scheduler::Scheduler` per DP rank, on the node that holds them;
//! - the peer links of an epoch whose worker group has ranks on other hosts ([`worker::peers`]):
//!   on the scheduler node one per peer process, which carries the schedulers' batches to the
//!   peer's ranks, and on a peer the one link that runs them;
//! - one `worker::device::Worker` per device, the only thread that issues to that device's streams;
//! - the node's resident memory ([`Memory`]);
//! - the blocking pool host work runs on ([`ThreadPool`]).
//!
//! ## Lifecycle
//!
//! Where the engine is is one value ([`lifecycle::State`]): its [`lifecycle::Lifecycle`]
//! (`Starting`, `Parked`, `Serving`, `Draining`, `Stopped`, `Failed`) and, while it serves, its
//! generation update ([`generation::Generation`]: `Steady`, `Preparing`, `Prepared`, `Trial`).
//! Both are moved by messages alone: a command of the daemon's process interface, the completion
//! of work a command started, or a failure the engine observed. Nothing is polled: the control
//! thread blocks until a socket is readable, a completion wakes it (a device submission, a host
//! task, a scheduler's response, a work thread), or a timer is due (the watchdog's stall bound,
//! the once-a-second metrics report, the bound of a stage in flight).
//!
//! A new engine starts with its epoch's module set in its launch file: while `Starting` it
//! resolves every module its devices' profile runs name, installs them as its first generation
//! and captures its graphs, and then parks (launched beside a serving engine) or serves. Startup
//! keeps nothing and is no update. An update of a serving engine's generation is the daemon's:
//! `Prepare`, `Switch`, then `Commit` or `Rollback`, each acknowledged by a report
//! (`crates/proto/src/process.proto`). No module content crosses that interface: the engine
//! resolves its modules from the Build Server itself (`crate::modules`).
//!
//! ## Modules
//!
//! - `launch`, `claim`, `ranks`, `start`: the launch file, the handshake, the Storage Server
//!   claim, the ranks' load, the first generation and the process's close.
//! - `record`: a start that records the packages its devices launch and serves nothing
//!   ([`record()`]).
//! - `lifecycle`, `generation`: the two enums, their transitions and the reports that
//!   acknowledge them.
//! - `serve`, `commands`, `requests`, `reports`: the control thread's loop and what it serves -
//!   the daemon's messages, requests and their event streams, reports and metrics.
//! - `prepare`, `installs`, `policy`, `switch`: the generation's preparation, its switch with the
//!   captures that follow, the commit, the rollback and the release.
//! - `epoch`: the drain, the promotion and admission.
//! - `operations`, `weights`: the operator's operations inside an epoch.

mod claim;
mod commands;
mod epoch;
mod generation;
mod installs;
mod launch;
mod lifecycle;
mod operations;
mod policy;
mod prepare;
mod ranks;
mod record;
mod reports;
mod requests;
mod serve;
mod start;
mod switch;
mod weights;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, bail};
use build_server::{Client, Target};
use ipc::execution::Outcome;
use ipc::execution::Submission;
use worker::blocking::ThreadPool;
use worker::control::Control;
use worker::device::Worker;
use worker::host::Lane;
use worker::memory::Memory;
use worker::memory::retained::Retained;
use worker::watchdog::Watchdog;

use crate::modules::load::ModuleWork;
use crate::rank::buckets::PrefillBound;
use epoch::Draining;
use launch::{ProcessInterface, read_engine_args};
use lifecycle::State;
use operations::Operation;
use prepare::Prepare;
pub use record::{Recorded, record};
use requests::RequestStream;
use switch::Switching;
use weights::WeightUpdate;

/// Runs the main process the launch file at `launch_file` describes, until its
/// daemon commands it down or its epoch fails. Its checkpoint is served by the
/// family of `catalogue` that claims the class it declares.
///
/// Must be called while no other thread reads the environment: the profiling
/// counters' injection sets variables before the first NVTX call.
///
/// # Errors
///
/// Fails when the launch file cannot be read, the process cannot start, its
/// control thread fails, or its epoch fails.
pub fn run(launch_file: &Path, catalogue: &nn::family::Catalogue) -> Result<()> {
    worker::profiling::counters::inject();

    let launch = read_engine_args(launch_file)?;
    let mut main = MainProcess::start(&launch, catalogue)?;
    let ending = main.serve();
    // A process that lost another of its epoch's main processes ends without its orderly close:
    // its device's collectives wait for ranks that are gone, so no close of its workers ends. Its
    // requests go with its connection, which the daemon reads as the engine's loss.
    if let Some(reason) = main.peers.lost() {
        tracing::error!(%reason, "the main process ends: a main process of its epoch was lost");
        std::process::exit(1);
    }
    main.close();

    // Why this process ends is the daemon's own diagnostic: the supervisor reports
    // "the engine process ended" and nothing more, so a handover that loses its engine
    // has no line to read unless this one says which way it went. A commanded end and a
    // failed epoch both stop the process, and only the second is a fault.
    match &ending {
        Ok(None) => tracing::warn!("the main process ends: a shutdown was commanded"),
        Ok(Some(failure)) => {
            tracing::warn!(
                reason = failure.reason(),
                "the main process ends: the epoch failed"
            )
        }
        // The whole chain: the outer context names the stage, and only the causes say what failed.
        Err(error) => tracing::warn!(
            error = %format!("{error:#}"),
            "the main process ends: the control thread failed"
        ),
    }
    if let Some(failure) = ending? {
        bail!("the main process ended: {}", failure.reason());
    }
    Ok(())
}

/// A main process's links to the other main processes of its epoch ([`worker::peers`]).
pub(super) enum Peers {
    /// The process holds every rank of its worker group.
    Alone,
    /// The scheduler node's process: one link per peer, which its schedulers' batches reach the
    /// peers' ranks through.
    Scheduler {
        remotes: Vec<Arc<worker::peers::Remote>>,
        liveness: Arc<worker::peers::Liveness>,
    },
    /// A peer's process: the one link the scheduler node's batches arrive on.
    Peer {
        upstream: worker::peers::Upstream,
        liveness: Arc<worker::peers::Liveness>,
    },
}

impl Peers {
    /// Why a link to another main process of the epoch was lost, once one was: the group cannot
    /// run a batch without that process's ranks, so this process ends.
    pub(super) fn lost(&self) -> Option<String> {
        match self {
            Self::Alone => None,
            Self::Scheduler { liveness, .. } | Self::Peer { liveness, .. } => liveness.lost(),
        }
    }

    /// Closes every link, which the processes at their other ends read as this one's end.
    pub(super) fn close(&mut self) {
        match std::mem::replace(self, Self::Alone) {
            Self::Alone => {}
            Self::Scheduler { remotes, .. } => {
                for remote in remotes {
                    remote.close();
                }
            }
            Self::Peer { upstream, .. } => upstream.close(),
        }
    }
}

/// One node's engine process for one epoch: the threads it started and the owners they borrow.
///
/// Every field is owned by exactly one thread: the control thread owns itself,
/// each worker thread owns its runners and streams, each scheduler thread its
/// mechanism. The fields below are the handles those threads were given and the
/// owners that must outlive them.
pub(crate) struct MainProcess {
    control: Control,
    /// Fires the control thread's wake. Every completion of work this thread started calls it,
    /// so the thread blocks instead of looking: a device submission, a host task and a work
    /// thread are each given a clone.
    wake: ipc::Wake,
    /// This process's links to the epoch's other engine processes, when its worker group has
    /// ranks on other hosts.
    peers: Peers,
    /// This node's schedulers, one per DP rank. The handle is the interface: each scheduler's thread
    /// owns the instance, so every call the control thread makes goes through the handle and the
    /// command channel carries the mutation.
    schedulers: Vec<scheduler::SchedulerHandle>,
    /// The routing policy that names the scheduler of every request when this node holds several
    /// (`requests::route`): the control thread's instance of the epoch's policy package, switched
    /// with the schedulers' instances.
    router: scheduler::Router,
    /// The route instance the preparation in flight builds on a thread of its own, taken by the
    /// switch.
    route_prepared: Option<crossbeam_channel::Receiver<scheduler::Candidate>>,
    /// The expert union of this node's ranks, ended before their device threads are.
    union: Option<Arc<worker::runners::experts::Union>>,
    /// One statistics reporter per scheduler, in the schedulers' order.
    ///
    /// A reporter keeps the cumulative baselines of the counters it has already
    /// reported, so each increment appears in exactly one snapshot: the same
    /// reporter must see every pass (the daemon's `/metrics`, `uniserve:num_requests_running`
    /// and its kind).
    stats_reporters: Vec<scheduler::SchedulerStatsReporter>,
    /// CUPTI's always-on counters, for as long as the process serves.
    _counters: Option<worker::profiling::counters::Counters>,
    /// The control thread's driver calls that waited for the device, as of its last report.
    control_calls: platform::cupti::blocking::Calls,
    /// This node's device workers, shared with the schedulers that submit through them: one
    /// worker thread per device, reached both by this process's own verbs (`prepare`, `switch`,
    /// `warmup`, `release`) and by the scheduler's `Batches`.
    workers: Vec<Arc<Worker>>,
    memory: Vec<Memory>,
    /// The rank's Storage Server client, kept for the two calls that reach its ledger and no
    /// other: a park gives the cache's segments back, and a wake maps them again. Every caller takes
    /// `&mut` of its own `&Retained` (`impl ExclusiveStorage for &Retained`), so the device thread can
    /// hold one beside the control thread without a lock.
    storage: Vec<Arc<Retained>>,
    blocking: ThreadPool,
    /// The blocking lane the Build Server's selection and the module loads run on.
    modules: Lane<ModuleWork>,
    /// Where the engine is: its lifecycle and its generation update.
    state: State,
    /// Whether the launch file started this engine parked, beside a serving one: its start ends
    /// parked instead of serving.
    starts_parked: bool,
    /// The stall and sticky-error rules of the control thread.
    watchdog: Watchdog,
    /// The Build Server client, shared with the module-load lane.
    packages: Arc<Mutex<Client>>,
    /// The preparation in flight: the start's first generation, or an update's.
    preparing: Option<Prepare>,
    /// The switch in flight: the captures every device queued for the generation that took over
    /// and the schedulers' responses to its policy.
    switching: Option<Switching>,
    /// The drain in flight: the schedulers still to park.
    draining: Option<Draining>,
    /// Whether the daemon that launched this process closed its front: at a handover the
    /// predecessor leaves, and the attached daemons serve on. Stated once, and not read
    /// again, because a closed connection gives every read the same error.
    daemon_closed: bool,
    /// What this process last announced it is, kept so that a daemon attaching mid-epoch is
    /// told it without asking the schedulers anything: the attach is a handover's step and a
    /// control thread blocked in a cache description cannot serve one.
    announced: Option<proto::request::wire::EngineInfo>,
    /// Whether the closing of the daemon's process interface has been reported: the interface
    /// stays closed, and the line is worth stating once.
    noted_process_closed: bool,
    /// What this node resolves a module for.
    targets: Vec<Target>,
    /// The Build Server's socket, which a policy's source resolves its module through.
    build_server: PathBuf,
    /// The daemon's process interface: the claim, the reports and every command the daemon
    /// sends this engine.
    process: ProcessInterface,
    /// Each admitted request with the stream it produces, the scheduler rank a command for it
    /// reaches, and the daemon front it belongs to: the scheduler responds to a submission with
    /// the stream of what the request produced, and this process is what carries those events
    /// back to the daemon. A stream leaves when its request ends.
    streams: Vec<RequestStream>,
    /// The launch's own request-slot count, which the daemon records as the epoch's capacity: the
    /// worker's row count includes the guidance prefix, which is not a request.
    max_running_requests: u32,
    /// The prefill buckets this rank captures and whether the deployment stated them: the
    /// ceiling a policy's `max_num_batched_tokens` is refused against, named with the policy's
    /// own figure when that refusal is rendered.
    prefill_bound: PrefillBound,
    /// How an engine that starts parked keeps its graph pools' device storage while it parks, as
    /// the daemon resolved `worker_config.standby_storage` for its failure domain.
    standby_storage: proto::request::StandbyStorage,
    /// Whether the graph pools' storage was given back while this engine parks: a promotion takes
    /// it again, at the addresses the graphs hold, before anything is admitted.
    graphs_suspended: bool,
    /// What each scheduler parked with, in `schedulers` order: kept from the drain to the
    /// promotion that serves this engine again, which adopts it.
    parked: Option<Vec<scheduler::ParkedCache>>,
    /// The admission gate, the daemon's `/admin/pause_generation`: submissions are refused while
    /// it is held, and no request in flight is ended by it.
    paused: bool,
    /// What this process's storage operations reach.
    data: Arc<crate::data_plane::DataPlane>,
    /// The operator's operations still working, each acknowledged when its completions arrive.
    operations: Vec<Operation>,
    /// A weight update in progress.
    weights: Option<WeightUpdate>,
    /// Whether the schedulers sleep: their storage is given back until a wake.
    asleep: bool,
}

/// Waits on one device command, and no longer than the watchdog's stall bound.
///
/// The control thread issues a handful of device commands of its own - the switch, a rollback and
/// a release - and none of them may hold it indefinitely: a device that has not finished by half the
/// watchdog timeout is stalled, and a stall is a failure the epoch reports rather than something the
/// control thread waits out.
fn device_command(
    watchdog: &Watchdog,
    device: platform::DeviceId,
    what: &str,
    submission: Submission,
) -> Result<()> {
    match submission.wait_timeout(watchdog.stall_bound()) {
        Some(Outcome::Success(_)) => Ok(()),
        Some(Outcome::Failed(error)) => bail!("the {what} of device {} failed: {error}", device.0),
        Some(Outcome::Cancelled) => bail!("the {what} of device {} was cancelled", device.0),
        None => bail!(
            "the {what} of device {} did not finish within the stall bound",
            device.0
        ),
    }
}

impl MainProcess {
    /// Wakes the control thread when `submission` settles, at once when it already has.
    fn wake_on_submission(&self, submission: &Submission) {
        let wake = self.wake.clone();
        let waker: ipc::execution::Waker = Arc::new(move || wake.wake());
        submission.on_settled(&waker);
    }

    /// Wakes the control thread when `task` ends, at once when it already has.
    fn wake_on_task<A: worker::host::HostAction>(&self, task: &worker::host::HostTask<A>) {
        let wake = self.wake.clone();
        if let Some(settled) = task.completion.subscribe(Box::new(move || wake.wake())) {
            settled();
        }
    }
}

/// Waits on one scheduler's response, no longer than the stall bound.
fn response(
    receiver: crossbeam_channel::Receiver<std::result::Result<(), String>>,
    bound: Duration,
    what: &str,
) -> Result<()> {
    match receiver.recv_timeout(bound) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => bail!("{what} failed: {error}"),
        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
            bail!("{what} did not respond within the stall bound")
        }
        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
            bail!("{what} lost its scheduler")
        }
    }
}
