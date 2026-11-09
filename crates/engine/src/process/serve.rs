//! The serve loop of the control thread: one pass reads the daemon, the peers and the schedulers,
//! advances every update in flight by at most one stage, forwards the requests' events and
//! watches the workers. Nothing here waits for a job: every wait is bounded by the watchdog's
//! stall bound, and a wait that outlasts it is reported by name ([`Wait`]).

use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use ipc::{ProgramStatus, UpdateFailure};
use worker::control::Ready;
use worker::device::Probe;
use worker::hot_update::Action;
use worker::watchdog::Running;

use super::MainProcess;

/// How often the engine reports its counters.
const METRICS_EVERY: Duration = Duration::from_secs(1);

/// How closely an update's probation is watched while one is pending: the phase machine states an
/// expiry, and the control thread asks it at least this often. A rollback window is stated in whole
/// seconds, so the poll costs a wake per tenth of a second for the length of the window.
const PROBATION_POLL: Duration = Duration::from_millis(100);

/// How long a serve pass sleeps while a precopy, a warmup or a drain is in flight.
const TRANSITION_POLL: Duration = Duration::from_millis(1);

/// The window a daemon gives one program-layer command before it gives up on it, as this process
/// must assume it (`crates/daemon/src/engine/control.rs`'s `COMMAND_TIMEOUT`): the engine does not
/// know that constant, it knows that its own reports must arrive inside the window or a run says
/// nothing but a timeout.
const DAEMON_COMMAND_WINDOW: Duration = Duration::from_secs(120);

/// A wait of the control path: when it began and when it was last reported.
///
/// Every step of the program-layer update is polled rather than blocked on, so a step that never
/// advances is a run that hangs without a line. Each such wait reports itself by name, generation
/// and elapsed time once per stall bound, and never ends the epoch: the command's own failure path
/// decides, and a deadline that killed would be a second failure beside the one being chased.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Wait {
    since: Option<Instant>,
    reported_at: Option<Instant>,
}

impl Wait {
    /// Notes whether the wait is on: it begins on the first pass that finds it waiting, and the
    /// next pass that finds it over clears it.
    fn note(&mut self, waiting: bool, now: Instant) {
        if waiting {
            self.since.get_or_insert(now);
        } else {
            self.since = None;
            self.reported_at = None;
        }
    }

    /// Reports the wait when it has stood a stall bound since it began, and again once per bound
    /// after each report. Returns whether this pass reported it.
    fn report(
        &mut self,
        name: &'static str,
        generation: u64,
        bound: Duration,
        now: Instant,
    ) -> bool {
        let Some(since) = self.since else {
            return false;
        };
        if now.duration_since(since) < bound {
            return false;
        }
        if self
            .reported_at
            .is_some_and(|at| now.duration_since(at) < bound)
        {
            return false;
        }
        self.reported_at = Some(now);
        tracing::warn!(
            generation,
            wait = name,
            waited_s = now.duration_since(since).as_secs(),
            "the control path is still waiting on this step and nothing has answered it yet"
        );
        true
    }
}

impl MainProcess {
    /// Serves until the daemon closes the process or a failure ends it.
    ///
    /// The control thread waits on its channels, never later than the watchdog's
    /// deadline; each pass runs the watchdog over the workers and handles what
    /// arrived.
    pub(super) fn serve(&mut self) -> Result<Option<UpdateFailure>> {
        let mut running = Vec::new();
        let mut reported = Instant::now();
        // This thread must not wait for a device: the count says whether it ever did.
        let _watched = platform::cupti::blocking::Watched::enter();
        loop {
            let pass = Instant::now();
            if self.stopping {
                // The report is stated once more on the way out: an epoch's counters are additive
                // and a frontend that only saw the passes would lose the last second of them.
                self.send_metrics();
                return Ok(None);
            }
            // Two obligations the control thread has that no event announces: the counters' report,
            // and a probation running out. An idle epoch has no events at all, so a deadline naming
            // only the stall rule would leave both to whatever traffic arrived - the dashboard would
            // see nothing of a server nobody is asking, and a generation nobody accepted would sit
            // on probation past the window that reverts it. Only a pending probation
            // adds its poll, so an idle epoch without one keeps waking for its counters alone.
            let stall = self.watchdog.deadline(oldest(&running));
            let mut due = reported + METRICS_EVERY;
            if self.rollback.is_some()
                && matches!(
                    self.update.phase(),
                    worker::hot_update::Phase::Probation { .. }
                )
            {
                due = due.min(Instant::now() + PROBATION_POLL);
            }
            // A transition moves one stage a pass, and what ends a stage is a device, a Program
            // Server or a scheduler rather than a socket. Passes run back to back while one is in
            // flight, so each stage costs what its work costs: a recovery is made of these stages,
            // and a client's stream stands still for all of them.
            if self.precopy.is_some()
                || !self.warming.is_empty()
                || self.draining.is_some()
                || !self.owed.is_empty()
                || self.weights.is_some()
            {
                due = due.min(Instant::now() + TRANSITION_POLL);
            }
            let deadline = Some(stall.map_or(due, |stall| stall.min(due)));
            // One metrics report a second, and a final one on the way out
            // (the daemon exports the engine's counters as metrics).
            if reported.elapsed() >= METRICS_EVERY {
                self.send_metrics();
                reported = Instant::now();
            }
            for event in self.control.wait(deadline)? {
                match event {
                    // Every front reads the same way: a frame from an attached daemon is served
                    // exactly as one from the daemon that launched this process, and the front it
                    // came from travels with it.
                    Ready::Daemon | Ready::Attached(_) => {
                        if let Some(failure) = self.daemon_messages()? {
                            return Ok(Some(failure));
                        }
                    }
                    Ready::PeerConnecting => {
                        self.control.accept()?;
                    }
                    Ready::Peer(rank) => {
                        if let Some(failure) = self.peer_messages(rank)? {
                            return Ok(Some(failure));
                        }
                    }
                    // What these announce is read by the pass below, whatever woke it.
                    Ready::Process | Ready::Woken | Ready::Idle => {}
                }
            }

            // What the admitted requests produced since the last pass.
            self.forward_events();

            // The daemon's commands for this epoch: the program layer's stages, the epoch layer's
            // park and serve, and the shutdown. They arrive on the process interface, so a pass
            // that did not look for them would never see an update.
            while let Some(command) = self.process.take_command()? {
                if let Some(failure) = self.process_command(command)? {
                    return Ok(Some(failure));
                }
            }
            // A daemon owns this process's life: an interface it closed is the end of the epoch,
            // not a failure to report to it, so the pass stops by name and the process exits
            // cleanly. At a hot restart the interface that closes is the predecessor's, and the
            // successor holds the deployment: this process is still serving it, and ending here
            // would leave the successor with a front into nothing.
            if self.process.closed() {
                self.control.watch_process(None);
                let serving = self.control.live_attached();
                if serving == 0 {
                    self.stopping = true;
                } else if !self.noted_process_closed {
                    self.noted_process_closed = true;
                    tracing::warn!(
                        serving,
                        "a daemon closed this process's interface; the attached daemons still \
                         hold this engine, so it serves on"
                    );
                }
            }

            // A precopy in flight advances one stage per pass, never waiting on
            // its builds or its devices.
            self.note_waits();
            self.advance_precopy()?;
            self.advance_drain()?;
            self.advance_owed();
            self.advance_weights();
            if let Some(reason) = self.data.poisoned() {
                bail!("a change of this rank's weights failed part of the way: {reason}");
            }
            if let Some(failure) = self.advance_warmup()? {
                return Ok(Some(failure));
            }

            running = self.read_workers();
            if let Some(failure) = self.watch(&running)? {
                return Ok(Some(failure));
            }
            if let Some(failure) = self.watch_schedulers()? {
                return Ok(Some(failure));
            }
            if let Some(failure) = self.advance_policy_check(&running)? {
                return Ok(Some(failure));
            }
            if let Some(failure) = self.watch_probation()? {
                return Ok(Some(failure));
            }
            // A pass that took seconds was blocked somewhere inside it, and a process that is
            // blocked reads as one that never answered: a handover's attach waits on exactly
            // that difference.
            if pass.elapsed() > Duration::from_secs(5) {
                tracing::warn!(
                    millis = pass.elapsed().as_millis(),
                    "a serve pass took longer than five seconds"
                );
            }
        }
    }

    /// The oldest unfinished command of each worker, in rank order, as the watchdog
    /// reads them.
    fn read_workers(&self) -> Vec<Running> {
        self.workers
            .iter()
            .enumerate()
            .filter_map(|(rank, worker)| {
                let reading = worker.watchdog();
                let watch = reading.longest()?;
                Some(Running {
                    rank,
                    batch: watch.batch.map(|batch| batch.0),
                    since: watch.started,
                })
            })
            .collect()
    }

    /// The watchdog's rules: a job past half the watchdog timeout is reported with its
    /// batch, and while a job has run for a second the devices' contexts are
    /// probed once a second for a sticky error. Both are failures of the epoch,
    /// and of an update on probation.
    fn watch(&mut self, running: &[Running]) -> Result<Option<UpdateFailure>> {
        let now = Instant::now();
        for (_, stall) in self.watchdog.stalls(now, running) {
            if let Some(ending) = self.failure(UpdateFailure::Stalled { stall })? {
                return Ok(Some(ending));
            }
        }

        if self.watchdog.probe_due(now, oldest(running)) {
            self.watchdog.probed(now);
            // The probe leaves an answered error latched in the device rather than failing, so a
            // device that errored and one that is healthy are read in one pass, and only then is
            // the update's decision table applied to what was found.
            let mut sticky = Vec::new();
            for (rank, worker) in self.workers.iter().enumerate() {
                let Probe::Sticky(error) = worker.probe()? else {
                    continue;
                };
                let batch = running
                    .iter()
                    .find(|job| job.rank == rank)
                    .and_then(|job| job.batch);
                sticky.push(UpdateFailure::Device {
                    device: u32::try_from(worker.device().0).unwrap_or_default(),
                    error: error.to_string(),
                    batch,
                });
            }
            for failure in sticky {
                if let Some(ending) = self.failure(failure)? {
                    return Ok(Some(ending));
                }
            }
        }
        Ok(None)
    }

    /// Applies the program-layer update's decision table to `failure`: a precopy failure discards
    /// the new generation and serving goes on; a failure on probation reverts
    /// the update; anything else ends the epoch's service. Returns the failure
    /// that ends service, if any.
    pub(super) fn failure(&mut self, failure: UpdateFailure) -> Result<Option<UpdateFailure>> {
        match self.update.action(&failure) {
            Action::Discard => {
                let generation = self.update.phase().generation().unwrap_or_default();
                if let Err(error) = self.update.precopy_failed(generation, failure.clone()) {
                    // The precopy this failure belongs to is no longer the one in flight - a revert
                    // or a later precopy took its place. The failure is still reported, and a
                    // serving process is not ended over bookkeeping that moved on.
                    tracing::warn!(generation, %error, "a precopy failure arrived for no precopy");
                }
                self.report(ProgramStatus::PrecopyFailed {
                    generation,
                    failure,
                });
                Ok(None)
            }
            Action::Revert => {
                let generation = self.update.phase().generation().unwrap_or_default();
                let reason = failure.reason();
                let status = self.revert(generation, failure)?;
                // The answer goes out from here: a revert this process started - a warmup that
                // failed, a device that could not continue - has no command to answer, and the
                // daemon waiting on the switchover learns it from this report or not at all. Two
                // minutes of silence and then a timeout is what its absence looks like.
                self.report(status);
                tracing::info!(generation, %reason, "an update on probation was reverted");
                Ok(None)
            }
            Action::Epoch => {
                // The three failures that end the epoch's service are said before it ends: the
                // daemon replaces a stalled rank on the report, and a report that never arrives
                // leaves it reading a process that exited for no stated reason.
                self.report_ending(&failure);
                Ok(Some(failure))
            }
        }
    }

    /// States the failure that ends this epoch's service to the daemon.
    ///
    /// A failure with no report of its own is left unsaid rather than given a shape it does not
    /// have; the process's exit still reaches the daemon.
    fn report_ending(&mut self, failure: &UpdateFailure) {
        use proto::process::report::Kind;

        let kind = match failure {
            UpdateFailure::Stalled { stall } => Kind::Stalled(proto::process::Stalled {
                generation: self.update.current(),
                batch: stall
                    .batch_id
                    .map_or_else(String::new, |batch| batch.to_string()),
                // The job's own name, as the worker classifies it: what stalled, not which
                // operator it was in.
                job: stall
                    .route
                    .clone()
                    .or_else(|| stall.method.clone())
                    .unwrap_or_default(),
                elapsed_ms: stall.running_ms,
            }),
            UpdateFailure::Device {
                device,
                error,
                batch: _,
            } => Kind::Fault(proto::process::Fault {
                kind: "sticky".to_owned(),
                reason: error.clone(),
                device: *device,
            }),
            UpdateFailure::NonFiniteLogits { request } => {
                Kind::NonFiniteLogits(proto::process::NonFiniteLogits {
                    request: format!("{request:?}"),
                })
            }
            _ => return,
        };
        self.send_report(proto::process::Report { kind: Some(kind) });
    }

    /// Reports the two waits of the epoch's start when they outlast a stall bound: the daemon's
    /// first program-layer command, and this node's warmup. Neither fails anything - the failures
    /// they precede state themselves - but a run that hangs here now says what it was waiting for.
    fn note_waits(&mut self) {
        let now = Instant::now();
        let bound = self.watchdog.stall_bound();
        self.command_wait
            .note(self.announced_at.is_some() && self.commands_seen == 0, now);
        self.command_wait
            .report("the daemon's first program-layer command", 0, bound, now);
        // The report is due inside the daemon's own window for the switchover this warmup answers: a
        // wait reported later is a wait nobody reads, because the daemon gives up first and the log
        // says only that something timed out.
        let warmup_bound = bound.min(DAEMON_COMMAND_WINDOW / 2);
        self.warmup_wait.note(!self.warming.is_empty(), now);
        let generation = self
            .warming
            .first()
            .map_or(0, |(_, _, generation)| *generation);
        if self
            .warmup_wait
            .report("this node's warmup", generation, warmup_bound, now)
        {
            // The thread that is stuck cannot say what it is stuck in, so each device states the job
            // it performs: a bucket's capture and a launch look identical from outside and are
            // different defects.
            for worker in &self.workers {
                tracing::warn!(
                    device = worker.device().0,
                    performing = worker
                        .performing()
                        .unwrap_or_else(|| "between commands".to_owned()),
                    captures = worker.capturing(),
                    "a device that has not finished its warmup"
                );
            }
        }
    }

    /// A scheduler thread that stopped ends the epoch's service, and the epoch says so.
    ///
    /// With no scheduler no batch is formed, so no job runs and no job can stall: the stall rule
    /// reports nothing and every request waits for a batch that will never be planned. The reason
    /// the thread stopped is its own (a policy that trapped or ran out of fuel is the usual one),
    /// and the update's decision table decides whether that ends the epoch or reverts the update
    /// on probation.
    fn watch_schedulers(&mut self) -> Result<Option<UpdateFailure>> {
        for (rank, scheduler) in self.schedulers.iter().enumerate() {
            if !scheduler.is_dead() {
                continue;
            }
            let failure = UpdateFailure::Policy {
                error: scheduler
                    .failure()
                    .unwrap_or_else(|| format!("the scheduler of rank {rank} stopped")),
            };
            return self.failure(failure);
        }
        Ok(None)
    }
}

/// The oldest of the workers' jobs.
fn oldest(running: &[Running]) -> Option<Running> {
    running.iter().min_by_key(|job| job.since).copied()
}
