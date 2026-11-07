//! The epoch layer's driver: the one object the daemon's update entry and the
//! process adapter share (`specs/worker.md` §7.4, §2.5).
//!
//! A switch cannot be driven from `apply` alone. Starting it and advancing it
//! need a [`Roster`] — the daemon's one owner of main processes: claims,
//! spawns, stops, commands, the frontend's serve, and the reports that come
//! back — and the entry (`Leader::apply`) runs on a request, not on a report.
//! So the driver owns both: the roster handle, and the switch that is running
//! or `None`.
//!
//! The adapter calls [`Driver::advance`] whenever it has reports (and after
//! every command batch it delivers); the entry calls [`Driver::apply`] for the
//! epoch layer of a change and [`Driver::commit`]/[`Driver::rollback`] for the
//! operator's decision. Every method is short and synchronous: a roster
//! command is a send, and the reports arrive through `advance`.
//!
//! One change runs at a time. A switch that reached [`Phase::Purged`] or
//! [`Phase::Abandoned`] has ended and leaves the slot free; a report that
//! arrives after that is dropped, since the epoch record it would have
//! advanced is already final.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use serde_json::Value;

use crate::epoch::{Deployment, Epochs, Failure};
use crate::interference::StepTime;
use crate::leader::UpdateError;
use crate::updates::change::Applied;
use crate::updates::epoch::{EpochChange, epoch_arm};
use crate::updates::switch::{Phase, Roster, Switch};
use crate::updates::trial::Trial;

/// The trait object the daemon constructs once: the adapter b02 builds over
/// `supervision`, behind a box so `Leader` names only this type.
pub(crate) type RosterHandle = Box<dyn Roster + Send>;

/// The epoch layer of one daemon: its epochs, the roster it commands, and the
/// switch in flight.
pub(crate) struct Driver {
    epochs: Arc<Epochs>,
    state: Mutex<State>,
}

struct State {
    roster: RosterHandle,
    switch: Option<Switch>,
    /// The trial of the update on probation, whichever layer's it is.
    trial: Option<Arc<Trial>>,
    /// The serving epoch's steps against their baseline while the running
    /// switch's target prepares.
    transfers: Option<Arc<Mutex<StepTime>>>,
    /// What the running switch's warming read.
    warming: Option<Value>,
    /// What the last update that ended reported, which the status keeps.
    last: Option<Value>,
}

impl State {
    /// `report` with what the update's trial and transfers read.
    fn completed(&self, mut report: Value) -> Value {
        if let Some(trial) = &self.trial {
            report["trial"] = trial.report(crate::leader::outcomes());
        }
        if let Some(steps) = &self.transfers {
            report["transfers"] =
                serde_json::to_value(*crate::leader::locked(steps)).unwrap_or_default();
        }
        if let Some(warming) = &self.warming {
            report["warming"] = warming.clone();
        }
        report
    }
}

impl std::fmt::Debug for Driver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Driver")
            .field("phase", &self.phase())
            .finish_non_exhaustive()
    }
}

impl Driver {
    pub(crate) fn new(epochs: Arc<Epochs>, roster: RosterHandle) -> Self {
        Self {
            epochs,
            state: Mutex::new(State {
                roster,
                switch: None,
                trial: None,
                transfers: None,
                warming: None,
                last: None,
            }),
        }
    }

    /// The epoch records, shared with the rest of the daemon.
    pub(crate) fn epochs(&self) -> &Arc<Epochs> {
        &self.epochs
    }

    /// Runs the epoch layer of a change into the switch slot, or answers
    /// `Ok(None)` when the change is another layer's (host, programs alone, or
    /// nothing) — the caller then takes its other path.
    pub(crate) fn apply(
        &self,
        previous: &Deployment,
        target: &Deployment,
        change: EpochChange,
        dry_run: bool,
    ) -> Result<Option<Applied>, UpdateError> {
        let mut state = self.lock();
        let State { roster, switch, .. } = &mut *state;
        epoch_arm(
            previous,
            target,
            change,
            &self.epochs,
            roster,
            switch,
            dry_run,
        )
    }

    /// Takes the reports that arrived and advances the running switch.
    ///
    /// Called by the adapter; does nothing when no switch runs, and drops the
    /// switch once it ended.
    pub(crate) fn advance(&self) {
        let mut state = self.lock();
        let State { roster, switch, .. } = &mut *state;
        let Some(running) = switch.as_mut() else {
            // No switch to advance: a report of a finished update is dropped,
            // but the roster's feed is still drained so it cannot grow.
            let _ = roster.events();
            return;
        };
        running.advance(roster);
        self.settle(&mut state);
    }

    /// Frees the slot of a switch that ended. The update that ended is the
    /// deployment's last, reverted or committed: the epoch layer keeps it, so
    /// applying the same target again asks for `force` (§7.2, §7.3 step 4) and
    /// the status reports why it was unavailable.
    fn settle(&self, state: &mut State) {
        let Some(running) = state.switch.as_ref() else {
            return;
        };
        if !matches!(running.phase(), Phase::Purged | Phase::Abandoned) {
            return;
        }
        self.epochs.set_update(running.update().clone());
        let report = crate::updates::epoch::report(running);
        state.last = Some(state.completed(report));
        state.trial = None;
        state.transfers = None;
        state.warming = None;
        state.switch = None;
    }

    /// Whether no switch runs: the one that ran was committed or returned to the previous epoch.
    pub(crate) fn settled(&self) -> bool {
        self.lock().switch.is_none()
    }

    /// States that the running switch's target on `host` is given its programs by the daemon.
    pub(crate) fn await_programs(&self, host: &str) {
        let mut state = self.lock();
        if let Some(running) = state.switch.as_mut() {
            running.await_programs([host.to_owned()]);
        }
    }

    /// Whether the running switch waits only for what the daemon gives its target: every node's
    /// target parked. `None` when no switch prepares any more, which ends such a wait.
    pub(crate) fn parked(&self) -> Option<bool> {
        let state = self.lock();
        let running = state.switch.as_ref()?;
        (running.phase() == Phase::Preparing).then(|| running.parked_everywhere())
    }

    /// The warming ended as `warming` states, which the update reports.
    pub(crate) fn warmed(&self, warming: Value) {
        self.lock().warming = Some(warming);
    }

    /// The Program Servers selected what the running switch's target executes.
    pub(crate) fn pin(&self, programs: &crate::epoch::FilledPrograms) {
        if let Some(running) = self.lock().switch.as_mut() {
            running.pin(programs);
        }
    }

    /// The running switch's target on `host` took its programs.
    pub(crate) fn handed(&self, host: &str) {
        let mut state = self.lock();
        let State { roster, switch, .. } = &mut *state;
        if let Some(running) = switch.as_mut() {
            running.handed(roster, host);
        }
    }

    /// The operator commits the trial. `true` when a switch was there to
    /// commit; the acknowledgement arrives through [`Driver::advance`].
    pub(crate) fn commit(&self) -> bool {
        let mut state = self.lock();
        let State { roster, switch, .. } = &mut *state;
        let Some(running) = switch.as_mut() else {
            return false;
        };
        running.commit(roster);
        true
    }

    /// The operator (or a failed probation) reverts the trial. `true` when a
    /// switch was there to revert.
    pub(crate) fn rollback(&self, reason: &str) -> bool {
        let mut state = self.lock();
        let State { roster, switch, .. } = &mut *state;
        let Some(running) = switch.as_mut() else {
            return false;
        };
        running.revert(roster, reason.to_owned());
        true
    }

    /// The trial's `failure` reverts the update on probation (§7.3 step 4):
    /// `false` when no switch stands on probation, so a threshold read while
    /// the update ended decides nothing.
    pub(crate) fn fail_trial(&self, failure: Failure) -> bool {
        let mut state = self.lock();
        let State { roster, switch, .. } = &mut *state;
        let Some(running) = switch.as_mut() else {
            return false;
        };
        if running.phase() != Phase::Probation {
            return false;
        }
        running.revert_with(roster, failure);
        true
    }

    /// `trial` decides the update in flight from now on, and the status
    /// reports it.
    pub(crate) fn begin_trial(&self, trial: Arc<Trial>) {
        self.lock().trial = Some(trial);
    }

    /// The trial of a program-layer update ended: what it read stays as the
    /// report of the last update.
    pub(crate) fn end_trial(&self, report: Value) {
        let mut state = self.lock();
        state.last = Some(state.completed(report));
        state.trial = None;
    }

    /// `steps` holds the serving epoch's steps against their baseline while
    /// the target prepares, which the status reports.
    pub(crate) fn watch_transfers(&self, steps: Arc<Mutex<StepTime>>) {
        self.lock().transfers = Some(steps);
    }

    /// When the running switch's switchover began, and the previous epoch's
    /// number and launch generation; `None` before it began.
    pub(crate) fn promotion(&self) -> Option<(Instant, (u64, u64))> {
        let state = self.lock();
        let running = state.switch.as_ref()?;
        Some((running.promotion()?, running.previous()))
    }

    /// The hosts where the previous epoch's processes have to start again for
    /// the running switch's revert.
    pub(crate) fn relaunch_due(&self) -> BTreeSet<String> {
        self.lock()
            .switch
            .as_ref()
            .map(|running| running.relaunch_due().clone())
            .unwrap_or_default()
    }

    /// The previous epoch's process on `host` ended for the trial.
    pub(crate) fn previous_ended(&self, host: &str) {
        let mut state = self.lock();
        let State { roster, switch, .. } = &mut *state;
        if let Some(running) = switch.as_mut() {
            running.previous_ended(roster, host);
        }
    }

    /// The previous epoch serves again on `host`, at the launch generation
    /// `generation` its owner started it at.
    pub(crate) fn relaunched(&self, host: &str, generation: u64) {
        let mut state = self.lock();
        let State { roster, switch, .. } = &mut *state;
        if let Some(running) = switch.as_mut() {
            running.serving(roster, host, generation);
        }
        self.settle(&mut state);
    }

    /// The running switch's phase, `None` when nothing runs.
    pub(crate) fn phase(&self) -> Option<Phase> {
        self.lock().switch.as_ref().map(Switch::phase)
    }

    /// The running switch as the operator sees it, `None` when nothing runs.
    pub(crate) fn report(&self) -> Option<Value> {
        let state = self.lock();
        let report = state.switch.as_ref().map(crate::updates::epoch::report)?;
        Some(state.completed(report))
    }

    /// The update as the status reports it: the one in flight, else the one
    /// that ended last.
    pub(crate) fn status(&self) -> Option<Value> {
        self.report().or_else(|| {
            let state = self.lock();
            if state.trial.is_some() {
                let update = serde_json::json!({ "change": "program", "phase": "probation" });
                return Some(state.completed(update));
            }
            state.last.clone()
        })
    }

    /// Whether an update is in flight.
    pub(crate) fn running(&self) -> bool {
        self.lock().switch.is_some()
    }

    /// The state, whether or not a panicking holder poisoned the lock: the
    /// switch is written under it and a poisoned lock must not end the daemon.
    fn lock(&self) -> MutexGuard<'_, State> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use proto::process::report::Kind;
    use proto::process::{Committed, Drained, Parked, Promoted, Report, Spawn};

    use super::*;
    use crate::epoch::{EpochRecord, Program, Status, Update};
    use crate::options::ServerArgs;
    use crate::updates::switch::{Event, Start};
    use proto::config::EngineConfig;

    /// The reports the test feeds as if the adapter had delivered them.
    type Feed = Arc<Mutex<Vec<Event>>>;

    /// A roster that records what it was asked and replays the fed events.
    #[derive(Default)]
    struct Fake {
        spawns: Vec<Spawn>,
        feed: Feed,
    }

    impl Fake {
        fn with_feed() -> (Self, Feed) {
            let feed = Feed::default();
            (
                Self {
                    spawns: Vec::new(),
                    feed: Arc::clone(&feed),
                },
                feed,
            )
        }
    }

    impl Roster for Fake {
        fn claim(
            &mut self,
            _host: &str,
            _world: u64,
            _scopes: &[proto::process::Scope],
            _retire: &[u64],
        ) -> Result<(), String> {
            Ok(())
        }

        fn spawn(
            &mut self,
            spawn: Spawn,
            _launch: &crate::engine::launch::Launch,
        ) -> Result<crate::updates::switch::Spawned, String> {
            let generation = spawn.generation;
            self.spawns.push(spawn);
            Ok(crate::updates::switch::Spawned { generation, pid: 1 })
        }

        fn stop(&mut self, _host: &str, _group: &str, _generation: Option<u64>) {}

        fn command(&mut self, _host: &str, _generation: u64, _command: proto::process::Command) {}

        fn serve(&mut self, _epoch: u64, _requests: &str) {}

        fn resume(&mut self, _host: &str, _generation: u64, _reason: &str) {}

        fn events(&mut self) -> Vec<Event> {
            std::mem::take(&mut *self.feed.lock().expect("the feed lock"))
        }
    }

    fn record(world: u64) -> EpochRecord {
        EpochRecord {
            config: ServerArgs::default().engine_config(),
            kernels: Default::default(),
            policy: Program {
                backend: "fcfs".to_owned(),
                revision: 0,
                arguments: toml::Table::new(),
            },
            previous_kernels: Default::default(),
            previous_policy: None,
            status: Status::Serving,
            generation: Some(4),
            world,
        }
    }

    fn change(target: u64) -> EpochChange {
        EpochChange {
            previous: 11,
            target,
            update: Update::new(7, Deployment::of(ServerArgs::default())),
            group: "engine".to_owned(),
            start: Start {
                executable: "/opt/uniserve".to_owned(),
                build: Some("2.0.0+new".to_owned()),
                launch: crate::engine::launch::Launch {
                    engine_config: "/run/engine-config-12.json".into(),
                    model: "/models/qwen3".into(),
                    requests: "unix:/run/requests.sock".to_owned(),
                    processes: "unix:/run/processes.sock".to_owned(),
                    engine: 7,
                    program_server: "/run/programs.sock".to_owned(),
                    devices: vec![0],
                    group: "engine".to_owned(),
                    rank: 0,
                    deployment: "qwen3".to_owned(),
                    world: 12,
                    scheduler_node: true,
                    peers: Vec::new(),
                    watchdog_timeout_s: None,
                    rollback_s: None,
                    queue_depth: 4,
                    world_size: 1,
                    standby: false,
                },
                environment: HashMap::new(),
            },
            previous_world: 11,
            previous_generation: 5,
            spawn_generation: 7,
            keeps: crate::options::PreviousWorld::Parked,
        }
    }

    fn reported(host: &str, generation: u64, kind: Kind) -> Event {
        Event::Report {
            host: host.to_owned(),
            generation,
            report: Report { kind: Some(kind) },
        }
    }

    fn deliver(feed: &Feed, events: Vec<Event>) {
        feed.lock().expect("the feed lock").extend(events);
    }

    /// The driver runs one epoch switch from the wire reports to the end: a
    /// change to another executable prepares, promotes, serves, drains the
    /// previous epoch, goes on probation, and a commit frees the slot. Nothing
    /// but `advance` moves it — the adapter's only call.
    #[test]
    fn the_epoch_layer_runs_from_reports_and_frees_its_slot() {
        let epochs = Arc::new(Epochs::new());
        epochs.record(11, record(11));
        let (roster, feed) = Fake::with_feed();
        let driver = Driver::new(Arc::clone(&epochs), Box::new(roster));
        let previous = Deployment::of(ServerArgs::default());
        let mut target_config = ServerArgs::default();
        target_config.worker_config.executable = "/opt/uniserve-b".into();
        let target = Deployment::of(target_config);

        // The switch's nodes are the target configuration's: one node here.
        let host = target.config.host_identity.clone();
        let applied = driver
            .apply(&previous, &target, change(12), false)
            .expect("the epoch arm runs");
        assert!(matches!(applied, Some(Applied::Epoch(_))));
        assert_eq!(driver.phase(), Some(Phase::Preparing));

        deliver(
            &feed,
            vec![reported(&host, 7, Kind::Parked(Parked { generation: 7 }))],
        );
        driver.advance();
        assert_eq!(driver.phase(), Some(Phase::Promoting));

        deliver(
            &feed,
            vec![reported(
                &host,
                7,
                Kind::Promoted(Promoted {
                    generation: 7,
                    program_generation: 0,
                }),
            )],
        );
        driver.advance();
        assert_eq!(driver.phase(), Some(Phase::Serving));

        deliver(
            &feed,
            vec![reported(&host, 5, Kind::Drained(Drained { epoch: 11 }))],
        );
        driver.advance();
        assert_eq!(driver.phase(), Some(Phase::Probation));

        assert!(driver.commit());
        assert_eq!(driver.phase(), Some(Phase::Committing));
        deliver(
            &feed,
            vec![reported(&host, 7, Kind::Committed(Committed {}))],
        );
        driver.advance();
        assert!(!driver.running());
        assert_eq!(
            epochs.get(12).map(|record| record.status),
            Some(Status::Serving)
        );
    }

    /// A commit or a rollback with no switch running does nothing and says so,
    /// so the entry can fall through to the program layer's probation.
    #[test]
    fn no_switch_means_no_decision() {
        let epochs = Arc::new(Epochs::new());
        let (roster, _feed) = Fake::with_feed();
        let driver = Driver::new(epochs, Box::new(roster));
        assert!(!driver.commit());
        assert!(!driver.rollback("nothing runs"));
        assert!(driver.report().is_none());
    }
}
