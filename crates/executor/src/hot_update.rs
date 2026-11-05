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
    /// The generation kept for probation and revert: the one serving before the switchover, held
    /// until the update is committed or reverted. It is the only generation a revert may return to.
    previous: Option<u64>,
    /// The generations a switchover or a revert released, awaiting the purge of §7.3 step 6, in
    /// release order.
    ///
    /// This is a list and not a slot because a purge is a command on a device queue: one may still
    /// be waiting when the next switchover runs, and a release that was overwritten could never be
    /// asked for again - `request_purge` can name only what this holds - so a whole generation's
    /// graphs, kernels and arena regions would leak.
    purgeable: Vec<u64>,
    /// The generation the daemon asked to purge (§7.3 step 6).
    ///
    /// The order that makes the release follow the generation's last batch is the device's,
    /// not this state machine's: a purge is a command in the same FIFO as the batches, so the
    /// device performs it behind every command it already holds, and after a switchover or a
    /// revert nothing new of the replaced generation can be served. A batch that still names it
    /// fails when it resolves its graph, saying which bucket and generation it wanted.
    purging: Option<u64>,
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
            purgeable: Vec::new(),
            purging: None,
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

    /// The generation a revert returns to, while an update is on probation.
    pub fn previous(&self) -> Option<u64> {
        self.previous
    }

    /// The generations released and not yet purged, in release order.
    pub fn awaiting_purge(&self) -> &[u64] {
        &self.purgeable
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
        // `failure` and `failed_generation` describe the last update, so a new one supersedes them:
        // a report that kept the previous verdict would name a failure the update in flight has
        // nothing to do with, and the epoch's own refusals read this.
        self.failure = None;
        self.failed_generation = None;
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
                self.purgeable.push(self.current);
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
                // The update is accepted, so the generation before it can never serve again: it
                // stops being a revert target and stays only as a generation awaiting its purge.
                self.previous = None;
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
                self.previous = None;
                // The generation the revert returns to **serves again**, so it is no generation
                // awaiting a purge: the switchover listed it while it was being replaced, and a
                // purge of it would release the kernels of what serves. Step 6's purge is for the
                // generation the revert went back from.
                self.purgeable.retain(|held| *held != previous);
                self.purgeable.push(generation);
                self.failure = Some(failure);
                self.failed_generation = Some(generation);
                self.phase = Phase::Serving;
                self.probation_started = None;
                Ok(())
            }
            Phase::Precopied {
                generation: expected,
            } if expected == generation => {
                // Reverting before the switchover changes nothing that serves, and the prepared
                // generation is **discarded**, not purged: it never served a batch, so there is no
                // last batch for a purge to follow, and `previous` is the generation a purge
                // releases - it still names whatever the last switchover replaced. Recording this
                // one as pending would leave the purge slot occupied by a generation no purge can
                // name (`request_purge` refuses a number that is not `previous`), and every later
                // purge would be refused for the rest of the epoch.
                self.failure = Some(failure);
                self.failed_generation = Some(generation);
                self.phase = Phase::Serving;
                Ok(())
            }
            phase => Err(unexpected(phase)),
        }
    }

    /// §7.3 step 6: the daemon asks for `generation`'s purge. It runs once no
    /// batch that used its kernels is in flight.
    ///
    /// The purge is a command in the device's FIFO, behind every batch the device already holds, so
    /// the release follows the generation's last batch by the device's own order (§7.3 step 6).
    ///
    /// Only the generation a switchover or a revert replaced is purged: any
    /// other number names a generation this process never held, and releasing
    /// it would report a purge of nothing. A second purge while one is pending
    /// is refused too.
    ///
    /// An update in flight does not stop a purge: what an update prepares is its
    /// own generation, and the generation being released here is the one the last
    /// switchover replaced. What is refused is a purge of the generation the
    /// update is preparing - that one is neither serving nor replaced, and its
    /// graphs are the ones being built.
    pub fn request_purge(&mut self, generation: u64) -> Result<()> {
        if self.phase.generation() == Some(generation) {
            return Err(Error::Invalid(format!(
                "generation {generation} is the one this process is preparing and cannot be purged"
            )));
        }
        if generation == self.current {
            return Err(Error::Invalid(format!(
                "the serving generation {} cannot be purged",
                self.current
            )));
        }
        if self.previous == Some(generation) {
            return Err(Error::State(
                "generation is the one probation may still revert to and cannot be purged",
            ));
        }
        // Only a generation a switchover or a revert released is ever purged (§7.3 step 6): any
        // other number names a generation this process never held, and releasing it would report a
        // purge of nothing.
        if !self.purgeable.contains(&generation) {
            return Err(Error::Invalid(format!(
                "generation {generation} is not one this process released"
            )));
        }
        if self.purging.is_some() {
            return Err(Error::State("a purge is already pending"));
        }
        self.purging = Some(generation);
        Ok(())
    }

    /// The generation whose purge runs now: the daemon asked for it, and the device performs the
    /// release behind every batch its queue already holds (§7.3 step 6). The caller releases its
    /// graphs and kernels and reports [`Self::purged`].
    pub fn purge_due(&self) -> Option<u64> {
        self.purging
    }

    /// The purge of `generation` finished.
    pub fn purged(&mut self, generation: u64) -> Result<()> {
        if self.purging != Some(generation) {
            return Err(Error::State("no such purge is pending"));
        }
        self.purging = None;
        self.purgeable.retain(|pending| *pending != generation);
        Ok(())
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

    /// A new update supersedes the last one's verdict: the failure a report names describes the
    /// update that failed, not one that has since been replaced.
    #[test]
    fn a_new_update_supersedes_the_last_one_s_failure() {
        let mut update = HotUpdate::new(7);
        update.begin_precopy(8).expect("the first update begins");
        update
            .precopy_failed(8, precopy_failure())
            .expect("the first update fails");
        assert_eq!(update.failed_generation(), Some(8));

        update.begin_precopy(9).expect("a second update begins");
        assert_eq!(update.failure(), None, "the failed update is superseded");
        assert_eq!(update.failed_generation(), None);
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
        assert_eq!(update.previous(), None, "the update is accepted, so nothing reverts");
        assert_eq!(
            update.awaiting_purge(),
            &[7],
            "the generation before it is released and awaits its purge"
        );
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

    /// A purge the daemon asks for after a commit is due at once, and the release follows the
    /// device's own order: the purge is a command behind every batch the device already holds, so
    /// no count stands between the request and the release (§7.3 step 6).
    #[test]
    fn a_purge_after_commit_is_due_and_releases_the_replaced_generation() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.switchover(8, Instant::now()).expect("switchover");
        update.commit(8).expect("commit");
        update
            .request_purge(7)
            .expect("the daemon asks for the old purge");
        assert_eq!(update.purge_due(), Some(7));
        update.purged(7).expect("purged");
        assert_eq!(update.phase(), Phase::Serving);
        assert_eq!(update.previous(), None);
    }

    /// A purge of the generation the last switchover replaced still runs while a newer update is in
    /// flight - the two generations are unrelated - and a purge of the generation being prepared is
    /// refused, because its graphs are the ones the update is building.
    #[test]
    fn a_purge_runs_during_a_later_update_and_refuses_the_generation_being_prepared() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.switchover(8, Instant::now()).expect("switchover");
        update.commit(8).expect("commit");

        update.begin_precopy(9).expect("a second update");
        update
            .request_purge(7)
            .expect("the switchover's generation is released while 9 is prepared");
        assert_eq!(update.purge_due(), Some(7));
        let error = update
            .request_purge(9)
            .expect_err("the generation being prepared is not released");
        assert!(matches!(error, Error::Invalid(_)), "{error:?}");
    }

    #[test]
    fn a_purge_of_the_serving_generation_is_refused() {
        let mut update = update();
        assert!(update.request_purge(7).is_err());
    }

    #[test]
    fn a_purge_of_a_generation_this_update_never_replaced_is_refused() {
        let mut update = update();
        // Nothing has been switched over or reverted yet, so nothing is old:
        // a number the state machine never held names no storage to release.
        let error = update
            .request_purge(9)
            .expect_err("there is no replaced generation to purge");
        assert!(matches!(error, Error::Invalid(_)), "{error:?}");
        // Nor is one after a completed update, when only the serving one is left.
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.switchover(8, Instant::now()).expect("switchover");
        update.commit(8).expect("commit");
        update.request_purge(7).expect("the switchover's generation");
        assert_eq!(update.purge_due(), Some(7));
        update.purged(7).expect("purged");
        assert!(update.request_purge(7).is_err());
    }

    /// A revert before the switchover discards the prepared generation and leaves the purge slot
    /// free: the generation never served, so nothing about it can be purged, and a later update's
    /// purge must still be able to run.
    /// A purge is a command on a device queue, so one may still be waiting when the next
    /// switchover runs. Both released generations are recorded and each can be purged: a record
    /// overwritten would be storage nobody could ask for again.
    #[test]
    fn a_switchover_keeps_a_generation_already_awaiting_its_purge() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.switchover(8, Instant::now()).expect("switchover");
        update.commit(8).expect("commit");

        update.begin_precopy(9).expect("a second update");
        update.precopied(9).expect("precopy finished");
        update
            .switchover(9, Instant::now())
            .expect("the second switchover runs while 7 awaits its purge");
        update.commit(9).expect("commit");
        assert_eq!(update.awaiting_purge(), &[7, 8]);

        update.request_purge(7).expect("the older release");
        update.purged(7).expect("purged");
        update.request_purge(8).expect("the newer release");
        update.purged(8).expect("purged");
        assert!(update.awaiting_purge().is_empty());
    }

    #[test]
    fn a_revert_before_switchover_leaves_the_purge_slot_free() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.revert(8, stall()).expect("revert before switchover");
        assert_eq!(update.current(), 7, "nothing was switched over");
        assert_eq!(update.previous(), None, "no generation was replaced");
        assert_eq!(update.purge_due(), None, "a discarded generation is not purged");

        // The next update's purge waits for its commit: until then the generation the switchover
        // replaced is what a revert would return to, and releasing its kernels would leave the
        // update with nothing to go back to.
        update.begin_precopy(9).expect("a second update");
        update.precopied(9).expect("precopy finished");
        update.switchover(9, Instant::now()).expect("switchover");
        update
            .request_purge(7)
            .expect_err("the revert target is not purged on probation");
        update.commit(9).expect("probation ends in a commit");
        update
            .request_purge(7)
            .expect("the generation the commit replaced is purged");
        assert_eq!(update.purge_due(), Some(7));
    }

    #[test]
    fn a_reverted_generation_is_purged() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.switchover(8, Instant::now()).expect("switchover");
        update.revert(8, stall()).expect("revert");
        assert_eq!(update.current(), 7);
        assert_eq!(update.previous(), None, "the revert returned to the older generation");
        assert_eq!(update.awaiting_purge(), &[8]);
        update
            .request_purge(8)
            .expect("the reverted generation is purged");
        assert_eq!(update.purge_due(), Some(8));
        update.purged(8).expect("purged");
        assert!(update.awaiting_purge().is_empty());
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
    fn a_purge_cannot_be_requested_twice_or_finished_out_of_turn() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        update.precopied(8).expect("precopy finished");
        update.switchover(8, Instant::now()).expect("switchover");
        update.commit(8).expect("commit");
        update
            .request_purge(7)
            .expect("the generation the switchover replaced");
        assert!(update.request_purge(7).is_err(), "its purge is pending");
        // A release the state machine did not ask for names no purge of this update.
        assert!(update.purged(6).is_err());
        assert_eq!(update.purge_due(), Some(7));
        update.purged(7).expect("purged");
        assert_eq!(update.purge_due(), None);
    }

    /// A number this process never released names no storage to release - during an update or
    /// outside one. An update in flight is not itself a reason to refuse a purge of a generation it
    /// does not touch.
    #[test]
    fn a_purge_of_a_generation_this_process_never_released_is_refused() {
        let mut update = update();
        update.begin_precopy(8).expect("a fresh update");
        let error = update
            .request_purge(6)
            .expect_err("nothing named 6 was ever released");
        assert!(matches!(error, Error::Invalid(_)), "{error:?}");
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
