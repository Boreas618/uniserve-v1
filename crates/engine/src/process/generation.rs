//! A serving engine's generation update: which generation of kernel and policy modules executes,
//! which one is being made ready beside it, and which one a trial keeps.
//!
//! [`Generation`] is where the update is, and [`Generations`] is the engine's one record of it:
//! the generation numbers and, per device, the generation that serves, the one prepared and the
//! one kept, with the packages each was built from. It calls no device, no scheduler and no Build
//! Server, so the rules can be checked as rules: which transition is legal in which state, and
//! what a failure means in each ([`Action`]).
//!
//! An engine numbers its own generations. The first is installed by its start
//! ([`Generations::install_first`]) with nothing kept; every later one is prepared, switched to
//! and then committed or rolled back. The generation an update leaves behind - the kept one at a
//! commit, the new one at a rollback - is released by the engine itself, as a command in each
//! device's queue behind every batch that queue already holds.

use std::sync::Arc;

use ipc::UpdateFailure;
use proto::module::Packages;

use worker::{Error, Result};

/// Where a serving engine's generation update is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Generation {
    /// One generation executes and no update is in flight.
    Steady,
    /// The next generation is being resolved, loaded, rebound or recaptured, and shadowed;
    /// serving goes on with the one that executes.
    Preparing {
        /// The generation being prepared.
        generation: u64,
    },
    /// The next generation is ready, and serving still goes on with the one that executes until
    /// the switch.
    Prepared {
        /// The generation ready to execute.
        generation: u64,
    },
    /// The new generation executes and the previous one is kept, with its graphs and its policy
    /// module, until the commit or the rollback.
    Trial {
        /// The generation on trial.
        generation: u64,
    },
}

/// What a failure means in the state the engine is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    /// The generation being prepared is dropped and serving goes on: nothing was switched, so
    /// nothing is rolled back.
    Abandon,
    /// The update on trial is rolled back and serving goes on with the kept generation.
    Rollback,
    /// The failure is the engine's, not an update's: it stops serving.
    Fail,
}

/// One engine's generations: the one that executes, the one an update prepares, and the one a
/// trial keeps.
#[derive(Debug)]
pub(crate) struct Generations {
    /// The generation whose kernels execute; zero until the start installed the first.
    current: u64,
    update: Generation,
    /// The generation a rollback returns to, while an update is on trial.
    previous: Option<u64>,
    /// What the last preparation built and no switch has taken: one per device, in the workers'
    /// order.
    pub(super) prepared: Vec<Arc<ops::Generation>>,
    /// What executed before the generation on trial, kept so a rollback can execute it again.
    pub(super) kept: Vec<Arc<ops::Generation>>,
    /// What each device executes now, in the workers' order.
    pub(super) serving: Vec<Arc<ops::Generation>>,
    /// The kernel packages the serving generation was built from, by slot. The shadow pass pairs
    /// a *changed* operator's kernel with the one serving, so "changed" is a fact about packages
    /// and never about generation numbers.
    pub(super) serving_packages: Packages,
    /// The packages of the generation `kept` holds.
    pub(super) kept_packages: Packages,
    /// The packages of the generation `prepared` holds.
    pub(super) prepared_packages: Packages,
}

impl Generations {
    /// An engine whose devices hold `serving`, the empty generations their workers started with:
    /// nothing executes until its start installs the first generation.
    pub(crate) fn new(serving: Vec<Arc<ops::Generation>>) -> Self {
        Self {
            current: 0,
            update: Generation::Steady,
            previous: None,
            prepared: Vec::new(),
            kept: Vec::new(),
            serving,
            serving_packages: Packages::new(),
            kept_packages: Packages::new(),
            prepared_packages: Packages::new(),
        }
    }

    /// Where the update is.
    pub(crate) fn update(&self) -> Generation {
        self.update
    }

    /// The generation whose kernels execute.
    pub(crate) fn current(&self) -> u64 {
        self.current
    }

    /// The number the next generation takes: one more than the one that executes.
    pub(crate) fn next(&self) -> u64 {
        self.current + 1
    }

    /// The start's install: what was prepared executes as `generation`, the engine's first, and
    /// nothing is kept, because nothing executed before it.
    pub(crate) fn install_first(&mut self, generation: u64) -> Result<()> {
        if self.current != 0 || self.update != Generation::Steady {
            return Err(Error::State("the engine already executes a generation"));
        }
        self.current = generation;
        self.serving = std::mem::take(&mut self.prepared);
        self.serving_packages = std::mem::take(&mut self.prepared_packages);
        Ok(())
    }

    /// An update begins: the next generation is being prepared. One update runs at a time.
    /// Returns the number the engine gives it.
    pub(crate) fn begin_prepare(&mut self) -> Result<u64> {
        if self.update != Generation::Steady {
            return Err(Error::State("a generation update is already in flight"));
        }
        let generation = self.next();
        self.update = Generation::Preparing { generation };
        Ok(generation)
    }

    /// The preparation of `generation` finished.
    pub(crate) fn prepared(&mut self, generation: u64) -> Result<()> {
        match self.update {
            Generation::Preparing {
                generation: expected,
            } if expected == generation => {
                self.update = Generation::Prepared { generation };
                Ok(())
            }
            update => Err(unexpected(update)),
        }
    }

    /// The preparation of `generation` failed: what it built is dropped and serving is unaffected.
    pub(crate) fn prepare_failed(&mut self, generation: u64) -> Result<()> {
        match self.update {
            Generation::Preparing {
                generation: expected,
            } if expected == generation => {
                self.drop_prepared();
                Ok(())
            }
            update => Err(unexpected(update)),
        }
    }

    /// The switch: `generation` executes from a batch boundary, and the generation that executed
    /// until now is kept through the trial.
    pub(crate) fn switch(&mut self, generation: u64) -> Result<()> {
        match self.update {
            Generation::Prepared {
                generation: expected,
            } if expected == generation => {
                self.previous = Some(self.current);
                self.current = generation;
                self.update = Generation::Trial { generation };
                // The packages move with the generations, as one record: the next update's
                // changed set is measured against what serves, and a rollback serves the
                // packages the kept pair holds.
                self.kept =
                    std::mem::replace(&mut self.serving, std::mem::take(&mut self.prepared));
                self.kept_packages = std::mem::replace(
                    &mut self.serving_packages,
                    std::mem::take(&mut self.prepared_packages),
                );
                Ok(())
            }
            update => Err(unexpected(update)),
        }
    }

    /// The trial ends in a commit. Returns the generation that was kept, which can never execute
    /// again and is the caller's to release.
    pub(crate) fn commit(&mut self) -> Result<u64> {
        match self.update {
            Generation::Trial { .. } => {
                let kept = self
                    .previous
                    .take()
                    .ok_or(Error::State("the trial kept no previous generation"))?;
                self.update = Generation::Steady;
                self.kept.clear();
                self.kept_packages.clear();
                Ok(kept)
            }
            update => Err(unexpected(update)),
        }
    }

    /// The update ends in a rollback. Returns the generation it leaves, and whether that
    /// generation executed: one on trial did, and the kept generation executes again; one that was
    /// only being prepared, or was prepared and never switched to, did not, and what serves is
    /// untouched.
    pub(crate) fn rollback(&mut self) -> Result<(u64, bool)> {
        match self.update {
            Generation::Trial { generation } => {
                let previous = self
                    .previous
                    .take()
                    .ok_or(Error::State("the trial kept no previous generation"))?;
                self.current = previous;
                self.update = Generation::Steady;
                self.serving = std::mem::take(&mut self.kept);
                self.serving_packages = std::mem::take(&mut self.kept_packages);
                self.prepared.clear();
                self.prepared_packages.clear();
                Ok((generation, true))
            }
            Generation::Preparing { generation } | Generation::Prepared { generation } => {
                self.drop_prepared();
                Ok((generation, false))
            }
            Generation::Steady => Err(unexpected(Generation::Steady)),
        }
    }

    /// What `failure` means for the update in flight.
    pub(crate) fn action(&self, failure: &UpdateFailure) -> Action {
        match (self.update, failure) {
            // A preparation's failure never touches what serves, whenever it is read.
            (_, UpdateFailure::Prepare { .. }) => Action::Abandon,
            (Generation::Trial { .. }, failure) if failure.rolls_back() => Action::Rollback,
            // A failure outside any update, or one of a generation that only waits for its
            // switch, is the engine's.
            _ => Action::Fail,
        }
    }

    /// Drops what an update prepared and returns to steady.
    fn drop_prepared(&mut self) {
        self.update = Generation::Steady;
        self.prepared.clear();
        self.prepared_packages.clear();
    }
}

/// The failure of a transition that does not fit the state the update is in.
fn unexpected(update: Generation) -> Error {
    Error::State(match update {
        Generation::Steady => "no generation update is in flight",
        Generation::Preparing { .. } => "the generation update is still being prepared",
        Generation::Prepared { .. } => "the generation update is prepared and not switched to",
        Generation::Trial { .. } => "the generation update is on trial",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipc::Stall;

    /// An engine that executes its first generation.
    fn serving() -> Generations {
        let mut generations = Generations::new(Vec::new());
        generations.install_first(1).expect("the first generation");
        generations
    }

    fn stall() -> UpdateFailure {
        UpdateFailure::Stalled {
            stall: Stall {
                batch_id: Some(7),
                route: None,
                method: None,
                calls: Vec::new(),
                running_ms: 15_000,
            },
        }
    }

    fn prepare_failure() -> UpdateFailure {
        UpdateFailure::Prepare {
            reason: "a module did not resolve".into(),
        }
    }

    #[test]
    fn a_start_installs_the_first_generation_with_nothing_kept() {
        let mut generations = Generations::new(Vec::new());
        assert_eq!(generations.current(), 0);
        generations.install_first(1).expect("the first generation");
        assert_eq!(generations.current(), 1);
        assert_eq!(generations.update(), Generation::Steady);
        // A second install is refused: the engine executes a generation already.
        assert!(generations.install_first(2).is_err());
        // Nothing was kept, so nothing can be rolled back to.
        assert!(generations.rollback().is_err());
    }

    #[test]
    fn an_update_runs_its_states_in_order_and_releases_the_kept_generation() {
        let mut generations = serving();
        assert_eq!(generations.begin_prepare().expect("an update begins"), 2);
        assert_eq!(
            generations.update(),
            Generation::Preparing { generation: 2 }
        );
        generations.prepared(2).expect("the preparation finished");
        assert_eq!(generations.update(), Generation::Prepared { generation: 2 });
        // Serving has not moved yet.
        assert_eq!(generations.current(), 1);

        generations.switch(2).expect("the switch");
        assert_eq!(generations.update(), Generation::Trial { generation: 2 });
        assert_eq!(generations.current(), 2);

        assert_eq!(generations.commit().expect("the commit"), 1);
        assert_eq!(generations.update(), Generation::Steady);
        assert_eq!(generations.current(), 2);
    }

    #[test]
    fn a_failed_preparation_drops_the_new_generation_and_keeps_serving() {
        let mut generations = serving();
        let generation = generations.begin_prepare().expect("an update begins");
        assert_eq!(generations.action(&prepare_failure()), Action::Abandon);
        generations
            .prepare_failed(generation)
            .expect("the failure is taken");
        assert_eq!(generations.update(), Generation::Steady);
        assert_eq!(generations.current(), 1);
        // The next update takes the same number: the dropped one never executed.
        assert_eq!(generations.begin_prepare().expect("another update"), 2);
    }

    #[test]
    fn a_rollback_on_trial_executes_the_kept_generation_again() {
        let mut generations = serving();
        let generation = generations.begin_prepare().expect("an update begins");
        generations.prepared(generation).expect("prepared");
        generations.switch(generation).expect("switched");
        assert_eq!(generations.action(&stall()), Action::Rollback);

        assert_eq!(
            generations.rollback().expect("the rollback"),
            (generation, true)
        );
        assert_eq!(generations.update(), Generation::Steady);
        assert_eq!(generations.current(), 1);
        // The trial is over: nothing is left to commit or to roll back.
        assert!(generations.commit().is_err());
        assert!(generations.rollback().is_err());
    }

    #[test]
    fn a_rollback_before_the_switch_discards_what_was_prepared() {
        for prepared in [false, true] {
            let mut generations = serving();
            let generation = generations.begin_prepare().expect("an update begins");
            if prepared {
                generations.prepared(generation).expect("prepared");
            }
            assert_eq!(
                generations.rollback().expect("the rollback"),
                (generation, false)
            );
            assert_eq!(generations.update(), Generation::Steady);
            assert_eq!(generations.current(), 1);
        }
    }

    #[test]
    fn one_update_runs_at_a_time() {
        let mut generations = serving();
        generations.begin_prepare().expect("an update begins");
        assert!(generations.begin_prepare().is_err());
        generations.prepared(2).expect("prepared");
        assert!(generations.begin_prepare().is_err());
        generations.switch(2).expect("switched");
        assert!(generations.begin_prepare().is_err());
    }

    #[test]
    fn a_transition_for_another_generation_or_state_is_refused() {
        let mut generations = serving();
        // Nothing is in flight.
        assert!(generations.prepared(2).is_err());
        assert!(generations.switch(2).is_err());
        assert!(generations.commit().is_err());

        generations.begin_prepare().expect("an update begins");
        assert!(generations.prepared(3).is_err());
        assert!(generations.prepare_failed(3).is_err());
        // A switch before the preparation finished is refused.
        assert!(generations.switch(2).is_err());
        generations.prepared(2).expect("prepared");
        assert!(generations.switch(3).is_err());
        // A commit before the switch is refused: nothing is on trial.
        assert!(generations.commit().is_err());
    }

    #[test]
    fn a_failure_outside_a_trial_is_the_engines() {
        let mut generations = serving();
        // No update in flight: a stall names no update. A preparation's failure that arrives
        // late still fails nothing that serves.
        assert_eq!(generations.action(&stall()), Action::Fail);
        assert_eq!(generations.action(&prepare_failure()), Action::Abandon);

        // While a generation is prepared, a stall is still the engine's: the generation that
        // executes is the one that stalled.
        let generation = generations.begin_prepare().expect("an update begins");
        assert_eq!(generations.action(&stall()), Action::Fail);
        generations.prepared(generation).expect("prepared");
        assert_eq!(generations.action(&stall()), Action::Fail);
    }
}
