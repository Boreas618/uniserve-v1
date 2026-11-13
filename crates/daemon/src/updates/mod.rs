//! The deployment's updates: moving it to another set of modules as a trial.
//!
//! [`update`] holds the one phase machine ([`update::Update`]): `Preparing`,
//! `Switching`, `Trial`, and `Committed`, `RolledBack` or `Abandoned`, in its
//! two forms (a generation inside the serving engines, an epoch beside them).
//!
//! [`Updater`] is what the daemon keeps of updates: the engine processes it
//! commands ([`Processes`]), the update in flight, the trial that reads its
//! thresholds, what the last update reported and the configurations a
//! rollback marked unavailable. The update entry (`entry`: `Leader::apply`,
//! `commit`, `rollback`) starts and decides updates through it, and the
//! thread the leader service wakes advances the one in flight with the
//! engines' acknowledgements ([`Updater::advance`]). One update is in flight
//! at a time.
//!
//! [`change`] holds a configuration change's request and what applying it did;
//! [`trial`] the thresholds of `update_config.rollback`.

pub(crate) mod change;
pub(crate) mod entry;
pub(crate) mod trial;
pub(crate) mod update;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use serde_json::Value;

use crate::epoch::{Epoch, Epochs, Failure};
use crate::interference::StepTime;
use crate::leader::UpdateError;
use crate::supervision::processes::Processes;
use trial::Trial;
use update::{Phase, Update};

/// A configuration a rollback marked unavailable, and why: a target that was
/// rolled back is not tried again until it is applied with `force`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Unavailable {
    /// The digest of what the configuration moves the deployment to: its
    /// build configuration and its module set, revisions included. A revision
    /// installed later is another target.
    pub digest: String,
    /// The epoch that executed it and was rolled back.
    pub epoch: u64,
    pub reason: String,
    /// Seconds since the Unix epoch.
    pub at: f64,
}

/// One phase an update entered, for the admin event log.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Transition {
    pub(crate) update: u64,
    pub(crate) phase: Phase,
    /// The update as it stood when it entered the phase.
    pub(crate) report: Value,
}

/// The digest of what `target` moves a deployment to: its build
/// configuration and the modules it runs.
pub(crate) fn target_digest(target: &Epoch) -> String {
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(target.engine.build_hash().as_bytes());
    hasher.update(serde_json::to_vec(&target.modules).unwrap_or_default());
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The daemon's updater: the engine processes it commands and the update in
/// flight.
pub(crate) struct Updater {
    epochs: Arc<Epochs>,
    state: Mutex<State>,
    /// The number and phase of the update in flight, `None` when none is:
    /// whoever waits for a phase waits here instead of reading on a clock.
    phase: tokio::sync::watch::Sender<Option<(u64, Phase)>>,
    /// Every phase an update entered, for the admin event log.
    transitions: tokio::sync::mpsc::UnboundedSender<Transition>,
    /// The log's end, taken once by its reader.
    entered: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<Transition>>>,
}

struct State {
    processes: Processes,
    update: Option<Update>,
    /// The thresholds the update on trial is read against.
    trial: Option<Arc<Trial>>,
    /// The serving epoch's steps against their baseline while an epoch
    /// update's target prepares.
    transfers: Option<Arc<Mutex<StepTime>>>,
    /// What the last update that ended reported, which the status keeps.
    last: Option<Value>,
    /// The targets a rollback marked unavailable, by their digest.
    unavailable: BTreeMap<String, Unavailable>,
    /// Epochs an update dropped whose storage the next claim retires.
    retire: Vec<u64>,
    /// How the last updates ended, by number, oldest first: what a follower
    /// of one reads once the slot is free, or taken by the next update.
    ended: std::collections::VecDeque<(u64, Phase)>,
}

/// How many ended updates the updater remembers the final phase of: more
/// than can end before the follower of the first reads it.
const ENDED_KEPT: usize = 16;

impl State {
    /// `report` with what the update's trial and transfers read.
    fn completed(&self, mut report: Value) -> Value {
        if let Some(trial) = &self.trial {
            report["trial"] = trial.report(crate::leader::request_counts());
        }
        if let Some(steps) = &self.transfers {
            report["transfers"] = serde_json::to_value(*service::lock(steps)).unwrap_or_default();
        }
        report
    }
}

impl std::fmt::Debug for Updater {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Updater")
            .field("in_flight", &self.in_flight())
            .finish_non_exhaustive()
    }
}

impl Updater {
    /// The updater of the deployment whose engines `processes` commands.
    pub(crate) fn new(epochs: Arc<Epochs>, processes: Processes) -> Self {
        let (transitions, entered) = tokio::sync::mpsc::unbounded_channel();
        Self {
            epochs,
            state: Mutex::new(State {
                processes,
                update: None,
                trial: None,
                transfers: None,
                last: None,
                unavailable: BTreeMap::new(),
                retire: Vec::new(),
                ended: std::collections::VecDeque::new(),
            }),
            phase: tokio::sync::watch::Sender::new(None),
            transitions,
            entered: Mutex::new(Some(entered)),
        }
    }

    /// The epoch records, shared with the rest of the daemon.
    pub(crate) fn epochs(&self) -> &Arc<Epochs> {
        &self.epochs
    }

    /// The phases updates enter, for the one reader that logs them; `None`
    /// once taken.
    pub(crate) fn transitions(&self) -> Option<tokio::sync::mpsc::UnboundedReceiver<Transition>> {
        service::lock(&self.entered).take()
    }

    /// The number and phase of the update in flight, as they change.
    pub(crate) fn watch(&self) -> tokio::sync::watch::Receiver<Option<(u64, Phase)>> {
        self.phase.subscribe()
    }

    /// The number and phase of the update in flight.
    pub(crate) fn in_flight(&self) -> Option<(u64, Phase)> {
        *self.phase.borrow()
    }

    /// The phase of update `number`: where it is while it is in flight, and
    /// how it ended afterwards; `None` for an update the updater never ran or
    /// no longer remembers.
    pub(crate) fn phase_of(&self, number: u64) -> Option<Phase> {
        let state = self.lock();
        match state.update.as_ref() {
            Some(update) if update.number() == number => Some(update.phase()),
            _ => state
                .ended
                .iter()
                .rev()
                .find(|(ended, _)| *ended == number)
                .map(|(_, phase)| *phase),
        }
    }

    /// The epochs an earlier update dropped, for the update that starts next
    /// to retire with its first claim.
    pub(crate) fn take_retire(&self) -> Vec<u64> {
        std::mem::take(&mut self.lock().retire)
    }

    /// Starts `update`: its `Preparing` begins, and its acknowledgements
    /// arrive through [`Self::advance`].
    ///
    /// # Errors
    ///
    /// `UpdateError::InProgress` while another update is in flight: one
    /// update runs at a time, so an effect has one cause.
    pub(crate) fn apply(&self, mut update: Update) -> Result<Value, UpdateError> {
        let mut state = self.lock();
        if state.update.is_some() {
            return Err(UpdateError::InProgress);
        }
        state.trial = None;
        state.transfers = None;
        update.start(&mut state.processes);
        let report = update.report();
        state.update = Some(update);
        self.settle(&mut state);
        Ok(report)
    }

    /// Takes the engines' reports and advances the update in flight.
    ///
    /// Called by the thread the leader service wakes; with no update the
    /// reports are still drained, so they cannot grow.
    pub(crate) fn advance(&self) {
        let mut state = self.lock();
        let State {
            processes, update, ..
        } = &mut *state;
        match update.as_mut() {
            Some(update) => update.advance(processes),
            None => drop(processes.events()),
        }
        self.settle(&mut state);
    }

    /// Publishes the phases the update entered, and frees the slot of one
    /// that ended: its report stays as the last update's, a rollback marks
    /// its target unavailable, and the epochs it dropped wait for the next
    /// claim.
    fn settle(&self, state: &mut State) {
        let Some(update) = state.update.as_mut() else {
            return;
        };
        let entered = update.take_entered();
        let (number, phase) = (update.number(), update.phase());
        let report = update.report();
        let report = state.completed(report);
        for phase in entered {
            let mut report = report.clone();
            report["phase"] = Value::from(phase.name());
            let _ = self.transitions.send(Transition {
                update: number,
                phase,
                report,
            });
        }
        if !phase.ended() {
            self.phase.send_replace(Some((number, phase)));
            return;
        }

        let Some(update) = state.update.take() else {
            return;
        };
        if phase == Phase::RolledBack
            && let Some(failure) = update.failure()
        {
            let digest = target_digest(update.target());
            state.unavailable.insert(
                digest.clone(),
                Unavailable {
                    digest,
                    epoch: update.target_epoch(),
                    reason: failure.reason.clone(),
                    at: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0.0, |elapsed| elapsed.as_secs_f64()),
                },
            );
        }
        state.retire.extend_from_slice(update.retire());
        if state.ended.len() == ENDED_KEPT {
            state.ended.pop_front();
        }
        state.ended.push_back((number, phase));
        state.last = Some(report);
        state.trial = None;
        state.transfers = None;
        self.phase.send_replace(None);
    }

    /// Commits the update in flight, and returns it as it stands; `None` when
    /// no update is there to commit.
    pub(crate) fn commit(&self) -> Option<Value> {
        let mut state = self.lock();
        let State {
            processes, update, ..
        } = &mut *state;
        let update = update.as_mut()?;
        if !update.commit(processes) {
            return None;
        }
        let report = update.report();
        self.settle(&mut state);
        Some(report)
    }

    /// Rolls the update in flight back with `failure` as its reason, and
    /// returns it as it stands; `None` when no update is in flight.
    pub(crate) fn rollback(&self, failure: Failure) -> Option<Value> {
        let mut state = self.lock();
        let State {
            processes, update, ..
        } = &mut *state;
        let update = update.as_mut()?;
        update.rollback(processes, failure);
        let report = update.report();
        self.settle(&mut state);
        Some(report)
    }

    /// The trial's `failure` rolls the update on trial back: `false` when no
    /// update stands on trial, so a threshold read while the update ended
    /// decides nothing.
    pub(crate) fn fail_trial(&self, failure: Failure) -> bool {
        let mut state = self.lock();
        let State {
            processes, update, ..
        } = &mut *state;
        let Some(update) = update.as_mut() else {
            return false;
        };
        if update.phase() != Phase::Trial {
            return false;
        }
        update.rollback(processes, failure);
        self.settle(&mut state);
        true
    }

    /// Rolls update `number` back if its deadline passed.
    pub(crate) fn expire(&self, number: u64) {
        let mut state = self.lock();
        let State {
            processes, update, ..
        } = &mut *state;
        let Some(update) = update.as_mut().filter(|update| update.number() == number) else {
            return;
        };
        update.expire(processes, Instant::now());
        self.settle(&mut state);
    }

    /// `trial` decides the update in flight from now on, and the status
    /// reports it.
    pub(crate) fn begin_trial(&self, trial: Arc<Trial>) {
        self.lock().trial = Some(trial);
    }

    /// `steps` holds the serving epoch's steps against their baseline while
    /// the target prepares, which the status reports.
    pub(crate) fn watch_transfers(&self, steps: Arc<Mutex<StepTime>>) {
        self.lock().transfers = Some(steps);
    }

    /// When the update in flight began to switch; `None` before it did.
    pub(crate) fn switch_began(&self) -> Option<Instant> {
        self.lock().update.as_ref()?.switch_began()
    }

    /// The launch generations of the previous epoch's engines that have to
    /// start again for the rollback in flight.
    pub(crate) fn relaunch_due(&self) -> BTreeSet<u64> {
        self.lock()
            .update
            .as_ref()
            .map(Update::relaunch_due)
            .unwrap_or_default()
    }

    /// The previous epoch's engine of launch generation `launch` ended for
    /// the trial.
    pub(crate) fn previous_ended(&self, launch: u64) {
        let mut state = self.lock();
        let State {
            processes, update, ..
        } = &mut *state;
        if let Some(update) = update.as_mut() {
            update.previous_ended(processes, launch);
        }
    }

    /// The previous epoch's engine of launch generation `previous` serves
    /// again at launch generation `launch`, which its owner started it at.
    pub(crate) fn relaunched(&self, previous: u64, launch: u64) {
        let mut state = self.lock();
        let State {
            processes, update, ..
        } = &mut *state;
        if let Some(update) = update.as_mut() {
            update.relaunched(processes, previous, launch);
        }
        self.settle(&mut state);
    }

    /// Promotes the parked engine of launch generation `generation` on
    /// `host`, a standby whose failure domain was lost: no engine drains
    /// beside it, and its `Serving` arrives on its own links.
    pub(crate) fn promote(&self, host: &str, group: &str, rank: u32, generation: u64) {
        let mut state = self.lock();
        state.processes.adopt(host, group, rank, generation);
        state.processes.command(
            host,
            generation,
            proto::process::Command {
                kind: Some(proto::process::command::Kind::Promote(
                    proto::process::Promote::default(),
                )),
            },
        );
    }

    /// Why `target` may not be applied without `force`: a rollback marked it
    /// unavailable.
    pub(crate) fn unavailable(&self, target: &Epoch) -> Option<Unavailable> {
        self.lock().unavailable.get(&target_digest(target)).cloned()
    }

    /// Every target a rollback marked unavailable.
    pub(crate) fn unavailable_marks(&self) -> Vec<Unavailable> {
        self.lock().unavailable.values().cloned().collect()
    }

    /// Takes over the marks of the daemon a hot restart replaced.
    pub(crate) fn adopt_unavailable(&self, marks: Vec<Unavailable>) {
        let mut state = self.lock();
        for mark in marks {
            state.unavailable.insert(mark.digest.clone(), mark);
        }
    }

    /// The update as the status reports it: the one in flight, else the one
    /// that ended last.
    pub(crate) fn status(&self) -> Option<Value> {
        let state = self.lock();
        match state.update.as_ref() {
            Some(update) => Some(state.completed(update.report())),
            None => state.last.clone(),
        }
    }

    /// The state, whether or not a panicking holder poisoned the lock: the
    /// update is written under it and a poisoned lock must not end the daemon.
    fn lock(&self) -> MutexGuard<'_, State> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

#[cfg(test)]
mod tests;
