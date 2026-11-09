//! `uniserve engine`: a node's main process for one epoch.
//!
//! The daemon starts one main process per node and epoch from the epoch's launch file, and [`run`]
//! serves it until the daemon commands it down or the epoch fails. The process computes nothing
//! itself; it assembles
//!
//! - the [`Control`] thread, which serves the daemon's request interface, its process supervision
//!   and the epoch's peer main processes, and never waits for a job;
//! - one `scheduler::Scheduler` per DP rank, on the node that holds them;
//! - one `worker::device::Worker` per device, the only thread that issues to that device's streams;
//! - the node's resident memory ([`Memory`]);
//! - the blocking pool host work runs on ([`ThreadPool`]),
//!
//! and drives the program-layer update between them: precopy on the blocking pool and the
//! workers, shadow execution against the schedulers' resident requests, switchover at a batch
//! boundary, probation, commit or revert, and purge after the old generation's last batch.
//! `docs/architecture.md` states the threads (The threads of a main process) and the epoch switch
//! this process takes part in (Epochs).
//!
//! A serving batch's `BatchOutput` is emitted from here too: the daemon's request interface
//! carries `Emitted` events, and the tokens come back with the scheduler's completed batch in
//! logits order.
//!
//! ## The daemon's program-layer sequence
//!
//! A standby main process starts, and the daemon then drives one update per generation through
//! the process interface's program commands ([`ipc::ProgramCommand`]), at no other entry point:
//!
//! 1. `Precopy { generation, programs, policy }`: the programs of the new generation, as revisions.
//!    Each node prepares it (module `precopy`) and answers `Precopied { generation }`, or
//!    `PrecopyFailed`, which discards the generation and leaves serving on the old one.
//! 2. `Switchover { generation }`: the node swaps at the next batch boundary and answers
//!    `Switched { generation }` once its warmup has run (module `switchover`). A main process
//!    serves nothing before its first switchover; the daemon only lets the frontend submit after
//!    it.
//! 3. `Commit` ends probation; `Revert` puts the previous generation back and keeps the update
//!    unavailable until forced; `Purge` releases a generation after its last batch. A policy that
//!    fails is reported as `PolicyReverted`, and a stall, a device fault or non-finite logits on
//!    probation revert the update automatically ([`ipc::UpdateFailure`]).
//!
//! Nothing program-content crosses that protocol: a node resolves its modules from the Program
//! Server itself (`crate::programs`).
//!
//! ## Modules
//!
//! - `launch`, `claim`, `ranks`, `start`: the launch file, the handshake, the Storage Server
//!   claim, the ranks' load and the process's start and close.
//! - `serve`, `commands`, `requests`, `reports`, `operations`, `weights`: the serve loop and
//!   what it serves - the daemon's and the peers' messages, requests and their event streams,
//!   reports and metrics, storage commands and weight updates.
//! - `policy`, `installs`, `precopy`, `switchover`: the program-layer update.
//! - `epoch`: the epoch layer's drain, resume, promotion and admission.

mod claim;
mod commands;
mod epoch;
mod installs;
mod launch;
mod operations;
mod policy;
mod precopy;
mod ranks;
mod reports;
mod requests;
mod serve;
mod start;
mod switchover;
mod weights;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use ops::Generation;
use program_server::program::Programs;
use program_server::{Client, Target};
use worker::batch::Submission;
use worker::blocking::ThreadPool;
use worker::completion::Outcome;
use worker::control::Control;
use worker::device::Worker;
use worker::host::Lane;
use worker::hot_update::HotUpdate;
use worker::memory::Memory;
use worker::memory::retained::Retained;
use worker::watchdog::Watchdog;

use crate::programs::modules::ModuleWork;
use crate::rank::buckets::PrefillBound;
use epoch::Draining;
use launch::{ProcessInterface, read_engine_args};
use operations::Owed;
use precopy::Precopy;
use requests::Served;
use serve::Wait;
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
pub fn run(launch_file: &Path, catalogue: &crate::Catalogue) -> Result<()> {
    worker::profiling::counters::inject();

    let launch = read_engine_args(launch_file)?;
    let mut main = MainProcess::start(&launch, catalogue)?;
    let ending = main.serve();
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

/// One node's main process for one epoch: the threads it started and the owners they borrow.
///
/// Every field is owned by exactly one thread: the control thread owns itself,
/// each worker thread owns its runners and streams, each scheduler thread its
/// mechanism. The fields below are the handles those threads were handed and the
/// owners that must outlive them.
pub(crate) struct MainProcess {
    control: Control,
    /// This node's schedulers, one per DP rank. The handle is the interface: each scheduler's thread
    /// owns the instance, so every call the control thread makes goes through the handle and the
    /// command channel carries the mutation.
    schedulers: Vec<scheduler::SchedulerHandle>,
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
    /// worker thread per device, reached both by this process's own verbs (`precopy`,
    /// `switchover`, `warmup`, `purge`) and by the scheduler's `Batches`.
    workers: Vec<Arc<Worker>>,
    memory: Vec<Memory>,
    /// The rank's Storage Server client, kept for the two calls that reach its ledger and no
    /// other: a park gives the cache's segments back, and a wake maps them again. Every caller takes
    /// `&mut` of its own `&Retained` (`impl ExclusiveStorage for &Retained`), so the device thread can
    /// hold one beside the control thread without a lock.
    storage: Vec<Arc<Retained>>,
    blocking: ThreadPool,
    /// The blocking lane the store's listing and the module loads run on.
    modules: Lane<ModuleWork>,
    /// The store's listing read for the precopy in flight. The lane reads it once and
    /// every later reader of that precopy - the keys a package declares, the candidates an operator
    /// has - answers from here, so the unbounded call is made where the lane is and never twice.
    store: Option<program_server::Listing>,
    /// The program-layer update's phases and bookkeeping.
    update: HotUpdate,
    /// The stall and sticky-error rules of the control thread.
    watchdog: Watchdog,
    /// The Program Server client, shared with the module-load lane.
    programs: Arc<Mutex<Client>>,
    /// The precopy between serve passes, if one is in flight.
    precopy: Option<Precopy>,
    /// The warmups a switchover queued, one per device, with the generation each was queued for.
    /// They are read on later passes rather than waited for: a warmup is a device job of its
    /// own, one bucket per queue turn behind serving, and a model with many buckets warms for longer
    /// than the daemon waits for `Switched`.
    warming: Vec<(u32, Submission, u64)>,
    /// How many switchovers this epoch has served. The epoch's first one is what
    /// installs a policy, so only it is checked for a policy that never arrived.
    switchovers: u32,
    /// When this process stated itself as `Ready` or `Standby`, and how many program-layer
    /// commands it has been given since the epoch began: a process that was read as ready and is
    /// given no command looks exactly like one that is working, from the daemon's side and from
    /// its own.
    announced_at: Option<Instant>,
    /// The state this process last stated on the request interface. An attaching
    /// daemon is told the same state on the connection it gives: a successor that
    /// is not told what this engine is never counts it as serving.
    reported_state: i32,
    /// Whether the daemon that launched this process closed its front: at a handover the
    /// predecessor leaves, and the attached daemons serve on. Stated once, and not polled
    /// again, because a closed connection answers every pass with the same error.
    daemon_closed: bool,
    /// What this process last announced it is, kept so that a daemon attaching mid-epoch is
    /// told it without asking the schedulers anything: the attach is a handover's step and a
    /// control thread blocked in a cache description cannot serve one.
    announced: Option<proto::request::wire::EngineInfo>,
    /// Whether the closing of the daemon's process interface has been reported: the interface
    /// stays closed, and the line is worth stating once.
    noted_process_closed: bool,
    commands_seen: u64,
    /// The two waits of the epoch's start that no event ends: the daemon's first command, and this
    /// node's warmup.
    command_wait: Wait,
    warmup_wait: Wait,
    /// A switchover whose answer is owed to the warmup it queued: the generation, and the peer to
    /// answer when admission opens. `Switched` means this process serves, so it waits for the
    /// buckets a batch would be planned into; a warmup that fails answers the switchover instead.
    pending_switch: Option<(u64, Option<u32>)>,
    /// When the warmup now running was queued: a warmup that outlives twice the stall bound is
    /// failed by name rather than left to answer nothing.
    warming_since: Option<Instant>,
    /// When the next check for a serving policy is due, and since when none serves.
    ///
    /// `crates/scheduler`'s first switchover with no prepared candidate adopts no policy, calls
    /// `note_policy_serves` never, raises nothing and answers `Ok(None)`; a scheduler with no policy
    /// forms no batch, so the epoch would sit with nothing running while the daemon believes the
    /// cut-over succeeded. The check asks each scheduler's policy slot, which answers within its own
    /// command timeout, and reads the answer in the watchdog's stall bound rather than waiting.
    policy_probe_at: Option<Instant>,
    policy_missing_since: Option<Instant>,
    /// The generations the last precopy built and the switchover has not taken:
    /// one per device, in `workers` order. A revert drops them, because the
    /// generation they hold never serves.
    prepared: Vec<Arc<Generation>>,
    /// The generations serving before the current ones, kept through probation so
    /// a revert can serve them again. Their graphs stay in the
    /// workers either way; this is the process's own record of what to serve.
    retired: Vec<Arc<Generation>>,
    /// The generation each device serves now, in `workers` order.
    generations: Vec<Arc<Generation>>,
    /// The kernel programs the serving generation was built from, by slot. The shadow
    /// pass pairs a *changed* operator's kernel with the one serving, so "changed" is a fact about
    /// programs and never about generation numbers.
    serving_programs: Programs,
    /// The programs of the generation `retired` holds: a revert serves that generation again, and
    /// the changed set of the update after it is measured against the programs it serves under.
    retired_programs: Programs,
    /// The programs of the generation `prepared` holds, waiting for the switchover to make them
    /// serving.
    prepared_programs: Programs,
    /// What this node resolves a module for.
    targets: Vec<Target>,
    /// The Program Server's socket, which a policy's source resolves its module through.
    program_server: PathBuf,
    /// The daemon's process interface: the claim, the program-layer reports and every command the
    /// daemon sends this epoch.
    process: ProcessInterface,
    /// Set by the daemon's `Shutdown`: the epoch ends at the end of this pass.
    stopping: bool,
    /// Each admitted request with the stream it produces, the scheduler rank a command for it
    /// reaches, and the daemon front it belongs to: the scheduler answers a submission with the
    /// stream of what the request produced, and this process is what carries those events back to
    /// the daemon. A stream leaves when its request ends.
    streams: Vec<Served>,
    /// Whether this process admits and executes: a drain parks every scheduler, and only the
    /// `Serve` that names the epoch again resumes it in place. A process that
    /// has not reached its first switchover admits nothing whatever the daemon has said: the epoch's
    /// first generation holds no kernel and no capture until the daemon's precopy fills it and the
    /// switchover warms it, and a batch admitted before then would reach a graph that does not
    /// exist. The boundary is necessary and not sufficient: the warmup queues one capture per
    /// device and runs them behind serving batches, so admission is owed to it as well
    /// ([`Self::open_admission`]).
    serving: bool,
    /// Whether the epoch's first switchover has happened: the boundary the first batch is served
    /// at.
    boundary: bool,
    /// A `Serve` that arrived before the boundary: the epoch may admit, but not yet.
    pending_serve: bool,
    /// A `Serve` that arrived while the epoch drained: it serves again once the drain ended.
    serve_after_drain: Option<proto::process::Serve>,
    /// Whether admission is owed to the warmup: the switchover that opens the epoch queues one
    /// capture per device and does not wait for them, so a batch admitted the moment the boundary
    /// opened could be planned into a bucket whose capture has not run - a replay finds no graph of
    /// it and answers an invariant error that names a capture bug. The epoch opens serving when the
    /// last submission finishes (`advance_warmup`), and a `Drain` drops what was owed.
    admission_owed: bool,
    /// The launch's own request-slot count, which the daemon records as the epoch's capacity: the
    /// worker's row count includes the guidance prefix, which is not a request.
    max_running_requests: u32,
    /// The catalogue of prefill buckets this rank captures and whether the deployment stated it:
    /// the ceiling a policy's `max_num_batched_tokens` is refused against, named
    /// with the policy's own figure when that refusal is rendered.
    catalogue: PrefillBound,
    /// Whether this process was launched as the epoch's standby: it admits nothing until
    /// the daemon promotes it *and* the epoch reaches its boundary.
    standby: bool,
    /// Whether a promoted standby has been promoted: a standby that has not been stays silent
    /// through the boundary, and one that has follows the rule of any other process.
    promoted: bool,
    /// What each scheduler parked with, in `schedulers` order: kept from the drain to the resume,
    /// which adopts it, so a resumed epoch reuses the pages it still holds instead of reading them
    /// again.
    parked: Option<Vec<scheduler::ParkedCache>>,
    /// A drain in progress: its schedulers park on the passes that follow (`advance_drain`).
    draining: Option<Draining>,
    /// The frontend's admission gate, the daemon's `/admin/pause_generation`: submissions are
    /// refused while it is held, and no request in flight is ended by it.
    paused: bool,
    /// The generations this process has released: a purge is a release of what it holds, so a
    /// second request for one of them is answered rather than run again.
    released: std::collections::BTreeSet<u64>,
    /// What this process's storage operations reach.
    data: Arc<worker::data_plane::DataPlane>,
    /// The storage commands still working, each answered by the pass that sees it end: the
    /// control thread waits for none of them.
    owed: Vec<Owed>,
    /// A weight update in progress.
    weights: Option<WeightUpdate>,
    /// Whether the schedulers sleep: their storage is given back until a wake.
    asleep: bool,
    /// The probation an update must survive before it is committed, when the launch file states
    /// one. The daemon decides the commit; this is the process's own guard against
    /// an update that is left on probation with nobody watching: it reverts at the threshold
    /// rather than serving a generation that was never accepted.
    rollback: Option<Duration>,
}

/// Waits on one device command, and no longer than the watchdog's stall bound.
///
/// The control thread issues a handful of device commands of its own - the switchover, a revert and
/// a purge - and none of them may hold it indefinitely: a device that has not answered by half the
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
    /// The program generation this process holds and would serve from now on.
    ///
    /// A prepared generation is the one a switchover takes, so it is what a promotion is about. With
    /// none prepared this is `HotUpdate`'s current generation, which is the daemon's own number from
    /// the last `Precopy` or switchover and `0` before any - never the launch number, which is what
    /// `Promoted.generation` states and a different question entirely.
    fn served_generation(&self) -> u64 {
        self.prepared
            .first()
            .map(|generation| generation.number())
            .unwrap_or_else(|| self.update.current())
    }
}

/// Waits on one scheduler's answer, no longer than the stall bound.
fn answer(
    receiver: crossbeam_channel::Receiver<std::result::Result<(), String>>,
    bound: Duration,
    what: &str,
) -> Result<()> {
    match receiver.recv_timeout(bound) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => bail!("{what} failed: {error}"),
        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
            bail!("{what} did not answer within the stall bound")
        }
        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
            bail!("{what} lost its scheduler")
        }
    }
}
