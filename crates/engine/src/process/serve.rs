//! The control thread's loop: it blocks until the daemon, a peer, a scheduler or a completion of
//! its own work has something for it, or a timer is due, and then serves what arrived - the
//! daemon's messages, the requests' events, the continuation of every piece of work whose
//! completion woke it, and the watchdog's rules.

use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use ipc::UpdateFailure;
use worker::control::Ready;
use worker::device::Probe;
use worker::watchdog::Running;

use super::MainProcess;
use super::generation::Action;
use super::lifecycle::Lifecycle;

/// How often the engine reports its counters.
const METRICS_EVERY: Duration = Duration::from_secs(1);

impl MainProcess {
    /// Serves until the daemon shuts the process down or a failure ends it.
    ///
    /// Nothing here polls: every piece of work this thread starts wakes it when it ends (a
    /// device submission, a host task, a scheduler's response, a work thread), so the wait's
    /// deadline is the next timer alone - the watchdog's stall bound, the metrics report, or the
    /// bound of a stage in flight.
    pub(super) fn serve(&mut self) -> Result<Option<UpdateFailure>> {
        let mut running = Vec::new();
        let mut reported = Instant::now();
        // This thread must not wait for a device: the count says whether it ever did.
        let _watched = platform::cupti::blocking::Watched::enter();
        loop {
            let pass = Instant::now();
            // A lost peer link is the end of the epoch's failure domain: its group cannot run a
            // batch without that process's ranks, and the daemon replaces the domain once this
            // process is gone. It is read first: the rest of a pass may wait on a scheduler whose
            // batch waits for the lost ranks.
            if let Some(reason) = self.peers.lost() {
                bail!("an engine process of this epoch was lost: {reason}");
            }
            if self.state.lifecycle() == Lifecycle::Stopped {
                // The report is stated once more on the way out: an epoch's counters are additive
                // and a daemon that only saw the passes would lose the last second of them.
                self.send_metrics();
                if let Err(error) = self.announce(false) {
                    tracing::debug!(%error, "no daemon was told that this engine stopped");
                }
                return Ok(None);
            }
            let mut deadline = reported + METRICS_EVERY;
            if let Some(stall) = self.watchdog.deadline(oldest(&running)) {
                deadline = deadline.min(stall);
            }
            if let Some(bound) = self.stage_bound() {
                deadline = deadline.min(bound);
            }
            // One metrics report a second, and a final one on the way out
            // (the daemon exports the engine's counters as metrics).
            if reported.elapsed() >= METRICS_EVERY {
                self.send_metrics();
                reported = Instant::now();
            }
            for event in self.control.wait(Some(deadline))? {
                match event {
                    // Every front reads the same way: a frame from an attached daemon is served
                    // exactly as one from the daemon that launched this process, and the front it
                    // came from travels with it.
                    Ready::Daemon | Ready::Attached(_) => {
                        if let Some(failure) = self.daemon_messages()? {
                            return Ok(Some(failure));
                        }
                    }
                    // What these announce is read below, whatever woke the thread.
                    Ready::Process | Ready::Woken | Ready::Idle => {}
                }
            }

            if let Some(reason) = self.peers.lost() {
                bail!("an engine process of this epoch was lost: {reason}");
            }

            // What the admitted requests produced since the thread last ran.
            self.forward_events();

            // The daemon's commands for this engine arrive on the process interface.
            while let Some(command) = self.process.take_command()? {
                if let Some(failure) = self.process_command(command)? {
                    return Ok(Some(failure));
                }
            }
            // A daemon owns this process's life: an interface it closed is the end of the epoch,
            // not a failure to report to it, so the loop stops by name and the process exits
            // cleanly. At a hot restart the interface that closes is the predecessor's, and the
            // successor holds the deployment: this process is still serving it, and ending here
            // would leave the successor with a front into nothing.
            if self.process.closed() {
                self.control.watch_process(None);
                let serving = self.control.live_attached();
                if serving == 0 {
                    self.state.stop();
                } else if !self.noted_process_closed {
                    self.noted_process_closed = true;
                    tracing::warn!(
                        serving,
                        "a daemon closed this process's interface; the attached daemons still \
                         hold this engine, so it serves on"
                    );
                }
            }

            // The work whose completion woke the thread goes on.
            if let Some(failure) = self.continue_work()? {
                return Ok(Some(failure));
            }
            if let Some(reason) = self.data.poisoned() {
                bail!("a change of this rank's weights failed part of the way: {reason}");
            }

            running = self.read_workers();
            if let Some(failure) = self.watch(&running)? {
                return Ok(Some(failure));
            }
            if let Some(failure) = self.watch_schedulers()? {
                return Ok(Some(failure));
            }
            // A pass that took seconds was blocked somewhere inside it, and a process that is
            // blocked reads as one that never responded: a handover's attach waits on exactly
            // that difference.
            if pass.elapsed() > Duration::from_secs(5) {
                tracing::warn!(
                    millis = pass.elapsed().as_millis(),
                    "a pass of the control thread took longer than five seconds"
                );
            }
        }
    }

    /// Continues every piece of work in flight whose completion arrived: the preparation, the
    /// switch's captures, the drain, the operator's operations and the weight update. A piece
    /// whose completion has not arrived is left as it is; its completion wakes this thread.
    fn continue_work(&mut self) -> Result<Option<UpdateFailure>> {
        if let Some(failure) = self.continue_prepare()? {
            return Ok(Some(failure));
        }
        if let Some(failure) = self.continue_switch()? {
            return Ok(Some(failure));
        }
        self.continue_drain()?;
        self.continue_operations();
        self.continue_weights();
        Ok(None)
    }

    /// The earliest instant a stage in flight is failed by name for standing still: a timer, so
    /// a stage whose completion never arrives is still read when its bound passes.
    fn stage_bound(&self) -> Option<Instant> {
        let bound = self.watchdog.stall_bound();
        [
            self.preparing
                .as_ref()
                .and_then(|prepare| prepare.bound(bound)),
            self.switching
                .as_ref()
                .map(|switching| switching.bound(bound)),
            self.draining.as_ref().map(|draining| draining.bound(bound)),
        ]
        .into_iter()
        .flatten()
        .min()
    }

    /// The oldest unfinished command of each worker, in rank order, as the watchdog
    /// reads them.
    fn read_workers(&self) -> Vec<Running> {
        self.workers
            .iter()
            .enumerate()
            .filter_map(|(rank, worker)| worker.running(rank))
            .collect()
    }

    /// The watchdog's rules: a job past half the watchdog timeout is reported with its
    /// batch, and while a job has run for a second the devices' contexts are
    /// probed once a second for a sticky error. Both are failures of the engine,
    /// and of an update on trial.
    fn watch(&mut self, running: &[Running]) -> Result<Option<UpdateFailure>> {
        let now = Instant::now();
        for (_, stall) in self.watchdog.stalls(now, running) {
            if let Some(ending) = self.failure(UpdateFailure::Stalled { stall })? {
                return Ok(Some(ending));
            }
        }

        if self.watchdog.probe_due(now, oldest(running)) {
            self.watchdog.probed(now);
            // The probe leaves an error latched in the device rather than failing, so a
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

    /// Applies the decision table to `failure`: a failed preparation is abandoned and serving
    /// goes on; a failure on trial rolls the update back; anything else fails the engine.
    /// Returns the failure that ends service, if any.
    pub(super) fn failure(&mut self, failure: UpdateFailure) -> Result<Option<UpdateFailure>> {
        match self.state.failure(&failure) {
            Action::Abandon => {
                self.abandon_prepare(failure.reason());
                Ok(None)
            }
            Action::Rollback => {
                let reason = failure.reason();
                let (left, executed) = self.state.rollback_on_failure()?;
                // The acknowledgement goes out from here: a rollback this engine started - a
                // capture that failed, a device that could not continue - has no command to
                // acknowledge, and the daemon learns of it from this report or not at all.
                self.roll_back(left, executed)?;
                let report = self.state.rolled_back(reason.clone());
                self.send_report(report);
                tracing::info!(generation = left, %reason, "an update on trial was rolled back");
                Ok(None)
            }
            Action::Fail => {
                // The failure is said before the engine ends: the daemon replaces a failed
                // engine on the report, and one that never arrives leaves it reading a process
                // that exited for no stated reason.
                self.report_failure(&failure);
                Ok(Some(failure))
            }
        }
    }

    /// States the failure that ends this engine's service to the daemon, on both links.
    fn report_failure(&mut self, failure: &UpdateFailure) {
        use proto::process::report::Kind;

        super::reports::state_failure(&mut self.control, &failure.reason());
        let kind = match failure {
            UpdateFailure::Stalled { stall } => Kind::Stalled(proto::process::Stalled {
                generation: self.state.generations.current(),
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
            UpdateFailure::Device { device, error, .. } => Kind::Fault(proto::process::Fault {
                kind: "sticky".to_owned(),
                reason: error.clone(),
                device: *device,
            }),
            UpdateFailure::NonFiniteLogits { request } => {
                Kind::NonFiniteLogits(proto::process::NonFiniteLogits {
                    request: format!("{request:?}"),
                })
            }
            // A start whose first generation could not be prepared, and a policy no previous
            // module can replace: neither names a device.
            UpdateFailure::Prepare { reason } | UpdateFailure::Threshold { reason } => {
                Kind::Fault(proto::process::Fault {
                    kind: "start".to_owned(),
                    reason: reason.clone(),
                    device: 0,
                })
            }
            UpdateFailure::Policy { error } => Kind::Fault(proto::process::Fault {
                kind: "policy".to_owned(),
                reason: error.clone(),
                device: 0,
            }),
        };
        self.send_report(proto::process::Report { kind: Some(kind) });
    }

    /// A scheduler thread that stopped ends the engine's service, and the engine says so.
    ///
    /// With no scheduler no batch is formed, so no job runs and no job can stall: the stall rule
    /// reports nothing and every request waits for a batch that will never be formed. The reason
    /// the thread stopped is its own (a policy that trapped or ran out of fuel is the usual one),
    /// and the decision table decides whether that fails the engine or rolls back the update on
    /// trial.
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
