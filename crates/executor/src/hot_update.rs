//! The program-layer update inside one main process: its phases, its
//! bookkeeping and its decisions (`specs/worker.md` §7.3).
//!
//! The phases are the virtual machine's live-migration phases: precopy prepares
//! the new generation while serving goes on, switchover cuts over at one batch
//! boundary, probation keeps the old generation until the update is confirmed,
//! and purge releases the old generation once its last batch has completed.
//!
//! This module is the state and the decision table only. It calls no driver, no
//! scheduler and no Program Server:
//!
//! - the precopy driver (`src/engine.rs`) resolves modules, builds the
//!   [`Generation`](crate::generation) and calls each rank's `precopy`;
//! - the scheduler switches the policy at the batch boundary;
//! - the control thread reports [`ipc::ProgramStatus`] and the daemon drives
//!   commit and revert.
//!
//! Keeping the phases here, without those parties, is what lets the rules of
//! §7.3 step 4 be checked as rules: which command is legal in which phase, which
//! failure discards a new generation, which one reverts the update, which one
//! fails the epoch, and when a purge may run.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use ipc::UpdateFailure;

use crate::{Error, Result};

/// What the main process is doing of the program layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// One generation serves and no update is in flight.
    Serving,
    /// §7.3 step 2: the new generation is being resolved, loaded, rebound or
    /// recaptured, and its shadow batches run; serving goes on.
    Precopying {
        /// The generation being installed.
        generation: u64,
    },
    /// §7.3 step 2 finished: the new generation is ready, and serving still
    /// goes on with the old one until switchover.
    Precopied {
        /// The generation ready to serve.
        generation: u64,
    },
    /// §7.3 step 4: the new generation serves and the old one is kept, with
    /// its graphs and its policy module, until commit or revert.
    Probation {
        /// The generation on probation.
        generation: u64,
    },
}

impl Phase {
    /// The generation this phase is about, if any.
    pub const fn generation(self) -> Option<u64> {
        match self {
            Self::Serving => None,
            Self::Precopying { generation }
            | Self::Precopied { generation }
            | Self::Probation { generation } => Some(generation),
        }
    }

    /// Whether an update is in flight.
    pub const fn updating(self) -> bool {
        !matches!(self, Self::Serving)
    }
}

/// What a failure of the given kind means while the main process is in the
/// current phase (§7.3 steps 2 and 4, §2.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// The new generation is discarded and serving goes on with the old one.
    /// Nothing was switched over, so no update is reverted (§7.3 step 2).
    Discard,
    /// The update is reverted automatically and serving goes on with the old
    /// generation (§7.3 step 4).
    Revert,
    /// The failure is the epoch's, not the update's: its service fails (§2.5),
    /// or the scheduler handles it (§4.4).
    Epoch,
}

/// One main process's program-layer update: which generation serves, which one
/// is being installed, and what is left to purge.
#[derive(Debug)]
pub struct HotUpdate {
    /// The generation whose kernels serve now.
    current: u64,
    phase: Phase,
    /// The generation kept for probation and revert: the one serving before the
    /// switchover, held until the purge after commit or revert.
    previous: Option<u64>,
    /// The generation the daemon asked to purge, waiting for its last batch to
    /// complete (§7.3 step 6).
    purging: Option<u64>,
    /// Batches submitted and not completed, by the generation their kernels
    /// belong to: what the purge of a generation waits for.
    in_flight: BTreeMap<u64, usize>,
    /// When probation began, for the rollback threshold of `update_config`
    /// (§7.3 step 4).
    probation_started: Option<Instant>,
    /// The first error that failed an update; the daemon marks the update
    /// unavailable with it until forced (§7.3 step 4).
    failure: Option<UpdateFailure>,
    /// The generation an update was attempted on, kept for the report that
    /// names it after the phase returned to serving.
    failed_generation: Option<u64>,
}

impl HotUpdate {
    /// A main process that started on `generation`.
    pub fn new(generation: u64) -> Self {
        Self {
            current: generation,
            phase: Phase::Serving,
            previous: None,
            purging: None,
            in_flight: BTreeMap::new(),
            probation_started: None,
            failure: None,
            failed_generation: None,
        }
    }

    /// The phase the update is in.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// The generation whose kernels serve now.
    pub fn current(&self) -> u64 {
        self.current
    }

    /// The generation kept for revert, until its purge.
    pub fn previous(&self) -> Option<u64> {
        self.previous
    }

    /// The first error that failed the last update, if one did.
    pub fn failure(&self) -> Option<&UpdateFailure> {
        self.failure.as_ref()
    }

    /// The generation whose update failed last, for the report that names it.
    pub fn failed_generation(&self) -> Option<u64> {
        self.failed_generation
    }

    /// §7.3 step 2 begins: `generation` replaces what serves. Only one update
    /// runs at a time; a generation that is not newer than the serving one is
    /// refused, so a replayed command cannot move time backwards.
    pub fn begin_precopy(&mut self, generation: u64) -> Result<()> {
        if self.phase.updating() {
            return Err(Error::State("a program-layer update is already in flight"));
        }
        if generation <= self.current {
            return Err(Error::Invalid(format!(
                "generation {generation} does not follow the serving generation {}",
                self.current
            )));
        }
        self.phase = Phase::Precopying { generation };
        Ok(())
    }

    /// §7.3 step 2 finished for `generation`.
    pub fn precopied(&mut self, generation: u64) -> Result<()> {
        match self.phase {
            Phase::Precopying {
                generation: expected,
            } if expected == generation => {
                self.phase = Phase::Precopied { generation };
                Ok(())
            }
            phase => Err(unexpected(phase)),
        }
    }

    /// §7.3 step 2 failed: the new generation is discarded, serving is
    /// unaffected, and the failure is kept for the report.
    pub fn precopy_failed(&mut self, generation: u64, failure: UpdateFailure) -> Result<()> {
        match self.phase {
            Phase::Precopying {
                generation: expected,
            } if expected == generation => {
                debug_assert!(!failure.reverts(), "a precopy failure never reverts");
                self.failure = Some(failure);
                self.failed_generation = Some(generation);
                self.phase = Phase::Serving;
                Ok(())
            }
            phase => Err(unexpected(phase)),
        }
    }

    /// §7.3 step 3: the update cuts over at a batch boundary, at `now`. The
    /// generation serving until now is kept for probation and revert.
    pub fn switchover(&mut self, generation: u64, now: Instant) -> Result<()> {
        match self.phase {
            Phase::Precopied {
                generation: expected,
            } if expected == generation => {
                self.previous = Some(self.current);
                self.current = generation;
                self.phase = Phase::Probation { generation };
                self.probation_started = Some(now);
                Ok(())
            }
            phase => Err(unexpected(phase)),
        }
    }

    /// Whether probation has run past the update's rollback threshold, which is
    /// one of §7.3 step 4's automatic-revert conditions. The threshold comes
    /// from `update_config.rollback`; the phase rules live here, and the control
    /// thread only polls this.
    pub fn probation_expired(&self, now: Instant, threshold: Duration) -> bool {
        match (self.phase, self.probation_started) {
            (Phase::Probation { .. }, Some(started)) => {
                now.saturating_duration_since(started) >= threshold
            }
            _ => false,
        }
    }

    /// §7.3 step 5: probation ends in a commit. The old generation stays until
    /// its purge.
    pub fn commit(&mut self, generation: u64) -> Result<()> {
        match self.phase {
            Phase::Probation {
                generation: expected,
            } if expected == generation => {
                self.phase = Phase::Serving;
                self.probation_started = None;
                self.failure = None;
                Ok(())
            }
            phase => Err(unexpected(phase)),
        }
    }

    /// §7.3 step 4: probation ends in a revert, or an update is abandoned
    /// before the switchover. The generation serving before the update serves
    /// again, and the failed generation is purged with the old one.
    pub fn revert(&mut self, generation: u64, failure: UpdateFailure) -> Result<()> {
        match self.phase {
            Phase::Probation {
                generation: expected,
            } if expected == generation => {
                let previous = self
                    .previous
                    .ok_or(Error::State("probation kept no previous generation"))?;
                self.current = previous;
                self.previous = Some(generation);
                self.failure = Some(failure);
                self.failed_generation = Some(generation);
                self.phase = Phase::Serving;
                self.probation_started = None;
                Ok(())
            }
            Phase::Precopied {
                generation: expected,
            } if expected == generation => {
                // Reverting before the switchover changes nothing serving, but
                // the prepared generation has graphs and kernels on the
                // executors: it is recorded for a purge like a reverted one.
                self.failure = Some(failure);
                self.failed_generation = Some(generation);
                self.phase = Phase::Serving;
                // An earlier purge may still be pending; the daemon's purge of
                // this generation follows its last batch, and there is none to
                // wait for because it never served.
                if self.purging.is_none() {
                    self.purging = Some(generation);
                }
                Ok(())
            }
            phase => Err(unexpected(phase)),
        }
    }

    /// §7.3 step 6: the daemon asks for `generation`'s purge. It runs once no
    /// batch that used its kernels is in flight; a purge asked for a
    /// generation that never served, or twice, is refused.
    pub fn request_purge(&mut self, generation: u64) -> Result<()> {
        if self.phase.updating() {
            return Err(Error::State("cannot purge while an update is in flight"));
        }
        if generation == self.current {
            return Err(Error::Invalid(format!(
                "the serving generation {} cannot be purged",
                self.current
            )));
        }
        if self.purging.is_some() {
            return Err(Error::State("a purge is already pending"));
        }
        self.purging = Some(generation);
        Ok(())
    }

    /// The generation whose purge may run now: the daemon asked for it and its
    /// last batch completed. The caller then releases its graphs and kernels
    /// and reports [`Self::purged`].
    pub fn purge_due(&self) -> Option<u64> {
        let generation = self.purging?;
        if self.in_flight.get(&generation).copied().unwrap_or(0) == 0 {
            Some(generation)
        } else {
            None
        }
    }

    /// The purge of `generation` finished.
    pub fn purged(&mut self, generation: u64) -> Result<()> {
        if self.purging != Some(generation) {
            return Err(Error::State("no such purge is pending"));
        }
        if self.purge_due() != Some(generation) {
            return Err(Error::State(
                "a batch of the purged generation is in flight",
            ));
        }
        self.purging = None;
        if self.previous == Some(generation) {
            self.previous = None;
        }
        self.in_flight.remove(&generation);
        Ok(())
    }

    /// A batch of `generation`'s kernels was submitted.
    pub fn batch_started(&mut self, generation: u64) {
        *self.in_flight.entry(generation).or_default() += 1;
    }

    /// A batch of `generation`'s kernels completed.
    pub fn batch_finished(&mut self, generation: u64) {
        let Some(count) = self.in_flight.get_mut(&generation) else {
            return;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.in_flight.remove(&generation);
        }
    }

    /// What `failure` means in the current phase (§7.3 steps 2 and 4, §2.5).
    pub fn action(&self, failure: &UpdateFailure) -> Action {
        if matches!(failure, UpdateFailure::Precopy { .. }) {
            return Action::Discard;
        }
        match self.phase {
            Phase::Probation { .. } => Action::Revert,
            // A precopy failure that is not of the new generation itself, or a
            // failure outside any update, belongs to the epoch or the
            // scheduler, not to this update.
            _ => Action::Epoch,
        }
    }
}

/// The failure of a command that does not fit the current phase. The caller's
/// log record names the command; the state names the phase it found.
fn unexpected(phase: Phase) -> Error {
    Error::State(match phase {
        Phase::Serving => "no program-layer update is in flight",
        _ => "the program-layer update is in another phase",
    })
}

#[cfg(test)]
mod tests {
    use ipc::Stall;

    use super::*;

    fn update() -> HotUpdate {
        HotUpdate::new(7)
    }

    fn stall() -> UpdateFailure {
        UpdateFailure::Stalled {
            stall: Stall {
                batch_id: Some(3),
                route: None,
                method: None,
                calls: Vec::new(),
                running_ms: 15_000,
            },
        }
    }

    fn precopy_failure() -> UpdateFailure {
        UpdateFailure::Precopy {
            reason: "a shadow batch was outside its tolerance".into(),
        }
    }

    #[test]
    fn a_new_main_process_serves_its_first_generation() {
        let update = update();
        assert_eq!(update.phase(), Phase::Serving);
        assert_eq!(update.current(), 7);
        assert_eq!(update.previous(), None);
    }

    #[test]
    fn a_successful_update_runs_its_phases_in_order() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        assert_eq!(update.phase(), Phase::Precopying { generation: 8 });
        update.precopied(8).expect("precopy finished");
        assert_eq!(update.phase(), Phase::Precopied { generation: 8 });
        update.switchover(8, Instant::now()).expect("switchover");
        assert_eq!(update.phase(), Phase::Probation { generation: 8 });
        assert_eq!(update.current(), 8);
        assert_eq!(update.previous(), Some(7));
        update.commit(8).expect("commit");
        assert_eq!(update.phase(), Phase::Serving);
        assert_eq!(update.current(), 8);
        assert_eq!(update.previous(), Some(7));
    }

    #[test]
    fn a_precopy_failure_discards_the_new_generation_and_keeps_serving() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update
            .precopy_failed(8, precopy_failure())
            .expect("precopy failed");
        assert_eq!(update.phase(), Phase::Serving);
        assert_eq!(update.current(), 7);
        assert_eq!(update.failed_generation(), Some(8));
        assert_eq!(update.failure(), Some(&precopy_failure()));
    }

    #[test]
    fn a_probation_failure_reverts_to_the_previous_generation() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.switchover(8, Instant::now()).expect("switchover");
        assert_eq!(update.action(&stall()), Action::Revert);
        update.revert(8, stall()).expect("revert");
        assert_eq!(update.phase(), Phase::Serving);
        assert_eq!(update.current(), 7);
        assert_eq!(update.failure(), Some(&stall()));
    }

    #[test]
    fn a_precopy_failure_is_discarded_rather_than_reverted() {
        let update = update();
        assert_eq!(update.action(&precopy_failure()), Action::Discard);
        assert_eq!(update.action(&stall()), Action::Epoch);
    }

    #[test]
    fn one_update_runs_at_a_time() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        assert!(update.begin_precopy(9).is_err());
        update.precopied(8).expect("precopy finished");
        assert!(update.begin_precopy(9).is_err());
        update.switchover(8, Instant::now()).expect("switchover");
        assert!(update.begin_precopy(9).is_err());
    }

    #[test]
    fn a_generation_does_not_go_backwards() {
        let mut update = update();
        assert!(update.begin_precopy(7).is_err());
        assert!(update.begin_precopy(6).is_err());
    }

    #[test]
    fn a_command_for_another_generation_is_refused() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        assert!(update.precopied(9).is_err());
        assert!(update.switchover(8, Instant::now()).is_err());
        assert!(update.commit(8).is_err());
    }

    #[test]
    fn a_purge_waits_for_the_last_batch_of_its_generation() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.switchover(8, Instant::now()).expect("switchover");
        update.commit(8).expect("commit");
        update.batch_started(7);
        update
            .request_purge(7)
            .expect("the daemon asks for the old purge");
        assert_eq!(update.purge_due(), None);
        update.batch_finished(7);
        assert_eq!(update.purge_due(), Some(7));
        update.purged(7).expect("purged");
        assert_eq!(update.phase(), Phase::Serving);
        assert_eq!(update.previous(), None);
    }

    #[test]
    fn a_purge_of_the_serving_generation_is_refused() {
        let mut update = update();
        assert!(update.request_purge(7).is_err());
        // A generation that is not the serving one may be purged, even one
        // newer than it: a reverted update's generation is exactly that.
        assert!(update.request_purge(8).is_ok());
    }

    #[test]
    fn a_reverted_generation_is_purged_after_the_last_batch_of_its_kind() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.switchover(8, Instant::now()).expect("switchover");
        update.revert(8, stall()).expect("revert");
        assert_eq!(update.current(), 7);
        assert_eq!(update.previous(), Some(8));
        update.batch_started(8);
        update
            .request_purge(8)
            .expect("the reverted generation is purged");
        assert_eq!(update.purge_due(), None);
        update.batch_finished(8);
        assert_eq!(update.purge_due(), Some(8));
        update.purged(8).expect("purged");
        assert_eq!(update.previous(), None);
    }

    #[test]
    fn a_generation_aborted_before_switchover_is_purged() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.revert(8, precopy_failure()).expect("abort");
        assert_eq!(update.current(), 7);
        assert_eq!(update.purge_due(), Some(8));
        update.purged(8).expect("purged");
    }

    #[test]
    fn probation_expires_at_the_rollback_threshold() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        let started = Instant::now();
        update.switchover(8, started).expect("switchover");
        let threshold = Duration::from_secs(60);
        assert!(!update.probation_expired(started + Duration::from_secs(59), threshold));
        assert!(update.probation_expired(started + threshold, threshold));
        update.commit(8).expect("commit");
        assert!(!update.probation_expired(started + threshold, threshold));
    }

    #[test]
    fn a_purge_cannot_be_requested_twice_or_finished_early() {
        let mut update = update();
        update.request_purge(6).expect("a stale generation");
        assert!(update.request_purge(5).is_err());
        assert!(update.purged(6).is_err());
        assert_eq!(update.purge_due(), Some(6));
        update.purged(6).expect("purged");
    }

    #[test]
    fn a_purge_is_refused_while_an_update_is_in_flight() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        assert!(update.request_purge(6).is_err());
    }

    #[test]
    fn a_revert_before_switchover_keeps_serving_untouched() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.revert(8, precopy_failure()).expect("revert");
        assert_eq!(update.phase(), Phase::Serving);
        assert_eq!(update.current(), 7);
        assert_eq!(update.previous(), None);
    }
}
