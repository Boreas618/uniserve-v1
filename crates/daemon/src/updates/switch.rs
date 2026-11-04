//! One epoch switch: the phases that move a deployment's main processes from
//! the serving epoch to a target one (`specs/worker.md` §7.4, §7.3, §2.5).
//!
//! The switch is driven by events the node roster delivers and by the
//! operator's commit or rollback; every phase ends on an acknowledgement a
//! main process makes, never on an assumption about its progress. The
//! participants and their wire are `proto::process`: a node's daemon starts
//! and stops children (`Spawn`, `Stop`), claims a world's storage (`Claim`),
//! commands a main process of one generation (`Command`), and the frontend is
//! commanded down its own connection (`Serve`). The roster behind those verbs
//! is the caller's; this module owns the order.
//!
//! The target takes the standby path, in front's order (§7.4):
//!
//! 1. **Preparing.** Every node starts the target executable at the next
//!    generation of the serving group and rank, then claims the target
//!    world's storage for the node's scopes. The child parks on its storage
//!    and reports `Parked { generation }`; the daemon numbers nothing — the
//!    child's number is authoritative and is the generation the switch
//!    commands from then on. The child's own `Hello` states the build it
//!    runs: the daemon's own executable's build is known up front, another
//!    executable's is fixed by its first child and every other child must
//!    report the same.
//! 2. **Promoting.** Every parked child is promoted in place (`Promote`) and
//!    answers `Promoted { generation }`.
//! 3. **Serving.** The record says serving, the frontend is told to serve the
//!    target epoch and resubmit the kept facts, and the previous epoch's main
//!    processes are drained. The pause is measured from the first promote to
//!    the last drained epoch. From the frontend's `Serve` on, a dead target is
//!    not a drop: the previous epoch must admit and serve again.
//! 4. **Probation**, recorded once the drain completed: a failure the target
//!    reports, or a target process that exits, reverts.
//! 5. **Commit** records the target as serving once it acknowledges, stops
//!    the old generation and purges and retires the previous world; **revert**
//!    acknowledges, serves the previous epoch again, stops the target
//!    generation, records it purged and marks the target configuration
//!    unavailable, so applying it again needs `force`.
//!
//! A world dropped before the switchover is kept in [`Switch::discarded`] and
//! passed in the next claim's `retire`: storage ends only through a claim,
//! and nothing else would ever retire a target that never served.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use proto::process::{self, Report, Spawn};

use crate::epoch::{EpochRecord, Epochs, Failure, Status, Update};

/// A node of the deployment the switch starts main processes on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Node {
    /// The node's host identity, as placements name it.
    pub(crate) host: String,
    /// The node's rank in the deployment.
    pub(crate) rank: u32,
}

/// How the target epoch's main processes are started: the executable, the
/// build identity it must report when the daemon knows it, and its arguments
/// and environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Start {
    /// The target main process's executable.
    pub(crate) executable: String,
    /// The build identity the target must report in its `Hello`: `Some` when
    /// the executable is the daemon's own (`proto::BUILD`), `None` for
    /// another executable, whose first child's `Hello` fixes it and every
    /// other child must then report the same.
    pub(crate) build: Option<String>,
    /// Arguments after the program name.
    pub(crate) arguments: Vec<String>,
    /// Variables set over the daemon's environment.
    pub(crate) environment: HashMap<String, String>,
}

/// What the node roster reports to the switch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Event {
    /// A main process of `generation` on `host` connected, saying what build
    /// it runs.
    Hello {
        host: String,
        generation: u64,
        build: String,
    },
    /// A main process of `generation` on `host` reported.
    Report {
        host: String,
        generation: u64,
        report: Report,
    },
    /// A main process of `generation` on `host` exited.
    Exit {
        host: String,
        generation: u64,
        code: Option<i32>,
    },
}

/// What the switch asks of the nodes: claim a world's storage, start and
/// stop main processes, command one generation's main process, command the
/// frontend, and take the events that arrive. The daemon's supervision is
/// its only real implementation; the crate's tests drive a scripted one.
pub(crate) trait Roster {
    /// Claims `world`'s storage on `host` for `scopes`, retiring `retire`'s
    /// worlds, and holds the claim when it returns `Ok`: §7.4 requires it
    /// held before the target maps storage, so it is synchronous and has no
    /// matching report. `Err` says the node refused and holds nothing.
    fn claim(
        &mut self,
        host: &str,
        world: u64,
        scopes: &[process::Scope],
        retire: &[u64],
    ) -> Result<(), String>;
    /// Starts `spawn` and returns the child's process id.
    fn spawn(&mut self, spawn: Spawn) -> Result<u32, String>;
    /// Stops one launch generation of `group` on `host`; every generation
    /// when `generation` is `None`.
    fn stop(&mut self, host: &str, group: &str, generation: Option<u64>);
    /// Commands the main process of `generation` on `host`.
    fn command(&mut self, host: &str, generation: u64, command: process::Command);
    /// Commands the frontend to serve `epoch` from `requests`.
    fn serve(&mut self, epoch: u64, requests: &str);
    /// Makes the previous epoch's main process of `generation` on `host`
    /// admit again after its drain. Exec settles whether the wire for that is
    /// `Revert` or `Pause { paused: false }` (§2.3, §7.4); this method is
    /// where that answer lands, so the switch states only that the epoch
    /// must admit again.
    fn resume(&mut self, host: &str, generation: u64, reason: &str);
    /// Takes the events that arrived since the last call.
    fn events(&mut self) -> Vec<Event>;
}

/// Where a switch is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// The target's main processes start and park on their claimed storage.
    Preparing,
    /// The parked targets are promoted in place and answer `Promoted`.
    Promoting,
    /// The target serves and the old epoch's main processes are drained.
    Serving,
    /// The target serves on trial.
    Probation,
    /// The operator committed; the target acknowledges.
    Committing,
    /// The update reverted; the target acknowledges.
    Reverting,
    /// The update was committed.
    Purged,
    /// The update returned to the previous epoch.
    Abandoned,
}

/// One epoch switch.
pub(crate) struct Switch {
    epochs: Arc<Epochs>,
    update: Update,
    /// The target epoch's number, which the record is kept under.
    target: u64,
    /// The previous epoch's number, which `Serve` and `Drain` name.
    previous: u64,
    /// The launch generation the target's children are started at: the
    /// serving generation's next one.
    spawn_generation: u64,
    /// The previous epoch's launch generation.
    previous_generation: u64,
    /// The target's generation as its child reported it: authoritative, and
    /// `None` until the first report.
    generation: Option<u64>,
    /// The target epoch's record, whose status the switch advances.
    record: EpochRecord,
    /// The worker group every main process of the deployment is in.
    group: String,
    nodes: Vec<Node>,
    start: Start,
    /// The build the target's children must report, once known.
    build: Option<String>,
    /// The request socket the frontend submits to after the switchover.
    requests: String,
    phase: Phase,
    /// Worlds whose storage must be retired by the next claim.
    retire: Vec<u64>,
    parked: BTreeSet<String>,
    promoted: BTreeSet<String>,
    drained: BTreeSet<String>,
    /// Hosts whose main process of the previous epoch exited during the
    /// switch: their epoch is replaced before it can serve again.
    previous_dead: BTreeSet<String>,
    /// Hosts whose previous-epoch main process reported serving again.
    serving: BTreeSet<String>,
    /// A restore waiting for the previous epoch to be replaced before the
    /// frontend is served it.
    pending_restore: Option<Failure>,
    committed: BTreeSet<String>,
    reverted: BTreeSet<String>,
    failure: Option<Failure>,
    /// Where the switchover began and how long its pause took: from its first
    /// promote to the last drained epoch.
    switchover_started: Option<Instant>,
    pause: Option<Duration>,
    /// Whether a serving main process of the previous epoch exited during the
    /// switch, which hands the failure domain to the standby promotion
    /// (§2.5) instead of a revert.
    promotion_due: bool,
}

impl Switch {
    /// A switch of epoch `previous` (launch generation `previous_generation`)
    /// to epoch `target`, whose record is `record`. The target is started one
    /// generation after the serving one; its own report is authoritative.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        epochs: Arc<Epochs>,
        update: Update,
        target: u64,
        record: EpochRecord,
        previous: u64,
        previous_generation: u64,
        group: String,
        nodes: Vec<Node>,
        start: Start,
        requests: String,
    ) -> Self {
        let mut record = record;
        record.status = Status::Precopy;
        epochs.record(target, record.clone());
        Self {
            epochs,
            update,
            target,
            previous,
            spawn_generation: previous_generation + 1,
            previous_generation,
            generation: None,
            record,
            group,
            nodes,
            start: start.clone(),
            build: start.build,
            requests,
            phase: Phase::Preparing,
            retire: Vec::new(),
            parked: BTreeSet::new(),
            promoted: BTreeSet::new(),
            drained: BTreeSet::new(),
            previous_dead: BTreeSet::new(),
            serving: BTreeSet::new(),
            pending_restore: None,
            committed: BTreeSet::new(),
            reverted: BTreeSet::new(),
            failure: None,
            switchover_started: None,
            pause: None,
            promotion_due: false,
        }
    }

    /// Where the switch is.
    pub(crate) fn phase(&self) -> Phase {
        self.phase
    }

    /// The update this switch belongs to.
    pub(crate) fn update(&self) -> &Update {
        &self.update
    }

    /// The target epoch's record as the switch has advanced it.
    pub(crate) fn record(&self) -> &EpochRecord {
        &self.record
    }

    /// The target's generation as its child reported it, once one did.
    pub(crate) fn generation(&self) -> Option<u64> {
        self.generation
    }

    /// Why the update failed, when it did.
    pub(crate) fn failure(&self) -> Option<&Failure> {
        self.failure.as_ref()
    }

    /// Worlds this switch dropped whose storage a later claim must retire:
    /// the update layer keeps them for the next switch's `start`.
    pub(crate) fn discarded(&self) -> &[u64] {
        &self.retire
    }

    /// The switchover's pause, once it ended.
    pub(crate) fn pause(&self) -> Option<Duration> {
        self.pause
    }

    /// Whether a serving main process died during the switch: the failure
    /// domain's promotion (§2.5) takes over from the revert.
    pub(crate) fn promotion_due(&self) -> bool {
        self.promotion_due
    }

    /// The generation the target's main processes run at: its own reported
    /// number once one reported, the generation it was spawned at before.
    fn target_generation(&self) -> u64 {
        self.generation.unwrap_or(self.spawn_generation)
    }

    /// Whether `generation` names a main process of the target epoch: any
    /// generation from the one the switch spawned on, since a promoted or
    /// relaunched child reports the next one.
    fn is_target(&self, generation: u64) -> bool {
        generation >= self.spawn_generation
    }

    /// Whether `generation` names a main process of the previous epoch.
    fn is_previous(&self, generation: u64) -> bool {
        generation >= self.previous_generation && !self.is_target(generation)
    }

    /// Starts the precopy: every node starts the target's main process, then
    /// claims the target world's storage before the child maps it.
    pub(crate) fn start(&mut self, roster: &mut impl Roster) {
        for node in &self.nodes {
            let spawn = Spawn {
                host: node.host.clone(),
                group: self.group.clone(),
                rank: node.rank,
                generation: self.spawn_generation,
                executable: self.start.executable.clone(),
                arguments: self.start.arguments.clone(),
                environment: self.start.environment.clone(),
                inherit: Vec::new(),
            };
            if let Err(reason) = roster.spawn(spawn) {
                self.failed(
                    roster,
                    Failure {
                        kind: "spawn_failed".to_owned(),
                        reason,
                        device: None,
                    },
                );
                return;
            }
        }
        let retire = std::mem::take(&mut self.retire);
        for node in &self.nodes {
            if let Err(reason) = roster.claim(
                &node.host,
                self.target,
                &self.scopes(node),
                if node.rank == 0 { &retire } else { &[] },
            ) {
                self.retire = retire;
                self.failed(
                    roster,
                    Failure {
                        kind: "claim_failed".to_owned(),
                        reason,
                        device: None,
                    },
                );
                return;
            }
        }
    }

    /// Takes the roster's events and advances the switch as they and the
    /// phase allow.
    pub(crate) fn advance(&mut self, roster: &mut impl Roster) {
        for event in roster.events() {
            self.event(roster, event);
        }
        self.progress(roster);
    }

    /// The target's main process of `generation` on `host` parked on its
    /// storage and is ready to be promoted (`Report::Parked`). The caller's
    /// mapping from the wire is the one line; this input is the phase's.
    pub(crate) fn parked(&mut self, roster: &mut impl Roster, host: &str, generation: u64) {
        if self.phase != Phase::Preparing || !self.accept(roster, host, generation) {
            return;
        }
        self.parked.insert(host.to_owned());
        self.progress(roster);
    }

    /// The target's main process of `generation` on `host` is promoted and
    /// serves in place (`Report::Promoted`).
    pub(crate) fn promoted(&mut self, roster: &mut impl Roster, host: &str, generation: u64) {
        if self.phase != Phase::Promoting || !self.accept(roster, host, generation) {
            return;
        }
        self.promoted.insert(host.to_owned());
        self.progress(roster);
    }

    /// The epoch `epoch` finished draining: it admits nothing and the facts of
    /// its requests are kept for the epoch the frontend serves. Whether the
    /// frontend or the old epoch's main processes report it is the caller's
    /// mapping.
    pub(crate) fn drained(&mut self, roster: &mut impl Roster, host: &str, epoch: u64) {
        if self.phase != Phase::Serving || epoch != self.previous {
            return;
        }
        self.drained.insert(host.to_owned());
        self.progress(roster);
    }

    /// The main process of `generation` on `host` serves again after being
    /// replaced: a restore that waited for the previous epoch can now serve
    /// it.
    pub(crate) fn serving(&mut self, roster: &mut impl Roster, host: &str, generation: u64) {
        if self.pending_restore.is_none() || generation < self.previous_generation {
            return;
        }
        self.previous_generation = self.previous_generation.max(generation);
        self.serving.insert(host.to_owned());
        if self.previous_dead.is_subset(&self.serving)
            && let Some(failure) = self.pending_restore.take()
        {
            self.finish_restore(roster, &failure.reason);
        }
    }

    /// The operator commits: the target serves from now on and the previous
    /// epoch is dropped and retired.
    pub(crate) fn commit(&mut self, roster: &mut impl Roster) {
        if self.phase != Phase::Probation {
            return;
        }
        self.phase = Phase::Committing;
        let generation = self.target_generation();
        for node in &self.nodes {
            roster.command(
                &node.host,
                generation,
                command(process::command::Kind::Commit(process::Commit {})),
            );
        }
    }

    /// The operator rolls back, or the probation did: the previous epoch
    /// serves again and the target is dropped.
    pub(crate) fn revert(&mut self, roster: &mut impl Roster, reason: String) {
        let failure = Failure {
            kind: "reverted".to_owned(),
            reason,
            device: None,
        };
        match self.phase {
            Phase::Probation => self.reverting(roster, failure),
            // The switchover happened and the frontend serves the target:
            // the previous epoch must admit and serve again.
            Phase::Serving => self.restore(roster, failure),
            Phase::Preparing | Phase::Promoting => self.failed(roster, failure),
            Phase::Committing | Phase::Reverting | Phase::Purged | Phase::Abandoned => {}
        }
    }

    /// Accepts `generation` as a target main process's, or refuses the node:
    /// the child numbers the generation it prepared, so any number from the
    /// one the switch spawned on names this epoch, and the highest reported
    /// is the epoch's current generation. A lower number is another epoch's.
    fn accept(&mut self, roster: &mut impl Roster, host: &str, generation: u64) -> bool {
        if generation < self.spawn_generation {
            self.refuse(roster, host, generation);
            return false;
        }
        if self.generation.is_none_or(|recorded| generation > recorded) {
            self.generation = Some(generation);
        }
        true
    }

    /// Stops a node's target whose report names an older generation, and
    /// drops the target epoch.
    fn refuse(&mut self, roster: &mut impl Roster, host: &str, generation: u64) {
        roster.stop(host, &self.group, Some(generation));
        self.failed(
            roster,
            Failure {
                kind: "generation_mismatch".to_owned(),
                reason: format!(
                    "the main process on {host} reported generation {generation}, older than the \
                 generation {} the target was spawned at",
                    self.spawn_generation
                ),
                device: None,
            },
        );
    }

    /// The scopes a claim keeps: the node's own `(group, rank)`, which the
    /// target takes over unchanged.
    fn scopes(&self, node: &Node) -> Vec<process::Scope> {
        vec![process::Scope {
            group: self.group.clone(),
            rank: node.rank,
        }]
    }

    /// Drops the target before the frontend ever served it: its processes
    /// stop, its world is retired by the next claim, and the previous epoch
    /// went on serving untouched.
    fn failed(&mut self, roster: &mut impl Roster, failure: Failure) {
        self.failure = Some(failure);
        self.retire.push(self.target);
        self.record.status = Status::Purged;
        self.epochs.record(self.target, self.record.clone());
        self.update.revert();
        let generation = self.target_generation();
        for node in &self.nodes {
            roster.stop(&node.host, &self.group, Some(generation));
        }
        self.phase = Phase::Abandoned;
    }

    /// Reverts a target that serves on probation: the target's own `Revert`
    /// is acknowledged, then the previous epoch takes over again.
    fn reverting(&mut self, roster: &mut impl Roster, failure: Failure) {
        self.update.fail(failure.clone());
        self.failure = Some(failure.clone());
        self.phase = Phase::Reverting;
        let generation = self.target_generation();
        for node in &self.nodes {
            roster.command(
                &node.host,
                generation,
                command(process::command::Kind::Revert(process::Revert {
                    reason: failure.reason.clone(),
                })),
            );
        }
    }

    /// Returns to the previous epoch after the frontend was told the target
    /// serves: the previous epoch admits again, the frontend serves it, and
    /// the target is stopped and retired.
    fn restore(&mut self, roster: &mut impl Roster, failure: Failure) {
        self.update.fail(failure.clone());
        self.failure = Some(failure.clone());
        if self.previous_dead.is_empty() {
            // The epoch that served before the switchover still runs: it
            // only has to admit again.
            self.finish_restore(roster, &failure.reason);
            return;
        }
        // Its replacement must serve first: serving it before that would
        // route requests to an epoch that is not running.
        self.pending_restore = Some(failure);
        self.phase = Phase::Reverting;
    }

    /// Serves the previous epoch again once it is running, and drops the
    /// target.
    fn finish_restore(&mut self, roster: &mut impl Roster, reason: &str) {
        self.resume(roster, reason);
        roster.serve(self.previous, &self.requests);
        self.retire.push(self.target);
        let retire = std::mem::take(&mut self.retire);
        let generation = self.target_generation();
        for node in &self.nodes {
            roster.stop(&node.host, &self.group, Some(generation));
            let _ = roster.claim(&node.host, self.previous, &self.scopes(node), &retire);
        }
        self.record.status = Status::Purged;
        self.epochs.record(self.target, self.record.clone());
        self.update.revert();
        self.phase = Phase::Abandoned;
    }

    /// The previous epoch admits again after its drain.
    fn resume(&mut self, roster: &mut impl Roster, reason: &str) {
        for node in &self.nodes {
            roster.resume(&node.host, self.previous_generation, reason);
        }
    }

    /// Finishes a revert the target acknowledged.
    fn reverted(&mut self, roster: &mut impl Roster) {
        self.record.status = Status::Purged;
        self.epochs.record(self.target, self.record.clone());
        self.update.revert();
        self.resume(roster, "the update reverted");
        roster.serve(self.previous, &self.requests);
        self.retire.push(self.target);
        let retire = std::mem::take(&mut self.retire);
        let generation = self.target_generation();
        for node in &self.nodes {
            roster.stop(&node.host, &self.group, Some(generation));
            let _ = roster.claim(&node.host, self.previous, &self.scopes(node), &retire);
        }
        self.phase = Phase::Abandoned;
    }

    /// One event, in whatever phase it arrives.
    fn event(&mut self, roster: &mut impl Roster, event: Event) {
        match event {
            Event::Hello {
                host,
                generation,
                build,
            } => self.hello(roster, host, generation, build),
            Event::Report {
                host,
                generation,
                report,
            } => self.report(roster, host, generation, report),
            Event::Exit {
                host,
                generation,
                code,
            } => self.exit(roster, host, generation, code),
        }
    }

    /// A child of `generation` says what build it runs.
    fn hello(&mut self, roster: &mut impl Roster, host: String, generation: u64, build: String) {
        if !self.is_target(generation) {
            return;
        }
        match &self.build {
            Some(expected) if *expected == build => {}
            Some(expected) => {
                // The child is not the build this epoch runs: it is stopped
                // before it counts as the epoch's process.
                roster.stop(&host, &self.group, Some(generation));
                self.failed(
                    roster,
                    Failure {
                        kind: "build_mismatch".to_owned(),
                        reason: format!(
                            "the child speaks build {build}, the epoch runs {expected}"
                        ),
                        device: None,
                    },
                );
            }
            None => {
                // Another executable's identity is known only once it
                // connects; the epoch records what it says.
                self.build = Some(build.clone());
                self.record.build_config.executable = build;
                self.epochs.record(self.target, self.record.clone());
            }
        }
    }

    /// A main process of `generation` exited.
    fn exit(&mut self, roster: &mut impl Roster, host: String, generation: u64, code: Option<i32>) {
        let reason = format!(
            "the main process of generation {generation} on {host} exited{}",
            code.map(|code| format!(" with {code}")).unwrap_or_default()
        );
        let failure = Failure {
            kind: "process_exit".to_owned(),
            reason,
            device: None,
        };
        if self.is_previous(generation) {
            match self.phase {
                Phase::Preparing | Phase::Promoting => {
                    // The serving epoch's main process died: the standby
                    // promotion (§2.5) takes over. The switch itself stops.
                    self.previous_dead.insert(host);
                    self.promotion_due = true;
                    self.failed(roster, failure);
                }
                Phase::Serving => {
                    // This host has nothing left to drain, and the epoch
                    // must be replaced before it could serve again.
                    self.previous_dead.insert(host.clone());
                    self.promotion_due = true;
                    self.drained.insert(host);
                    self.progress(roster);
                }
                Phase::Probation | Phase::Committing | Phase::Reverting => {
                    self.previous_dead.insert(host);
                    self.promotion_due = true;
                }
                _ => {}
            }
        } else if self.is_target(generation) {
            match self.phase {
                Phase::Preparing | Phase::Promoting => self.failed(roster, failure),
                Phase::Serving => self.restore(roster, failure),
                Phase::Probation => self.reverting(roster, failure),
                Phase::Reverting => {
                    // The previous epoch is to serve anyway: finish.
                    self.reverted(roster);
                }
                Phase::Committing => {
                    // The commit is an operator decision retained until it
                    // completes: the promotion or a relaunch of the target
                    // generation takes over, and the commit stays pending.
                    self.promotion_due = true;
                    self.update.fail(failure.clone());
                    self.failure = Some(failure);
                }
                _ => {}
            }
        }
    }

    /// One report of one main process.
    fn report(&mut self, roster: &mut impl Roster, host: String, generation: u64, report: Report) {
        use process::report::Kind;
        // The kind drives the phase; the whole report still states a failure.
        let kind = report.kind.clone();
        match (self.phase, kind) {
            // `Report::Parked` and `Report::Promoted` are not in this
            // branch's `process.proto` yet (front's `026466b20`,
            // `2d477b9b5`); when they land, the daemon's mapping calls
            // `parked` and `promoted` here. Until then those two inputs are
            // the phase's only sources.
            (Phase::Serving, Some(Kind::Drained(drained)))
                if generation == self.previous_generation =>
            {
                self.drained(roster, &host, drained.epoch);
            }
            (Phase::Preparing, Some(Kind::PrecopyFailed(failed))) if self.is_target(generation) => {
                self.failed(
                    roster,
                    Failure {
                        kind: "precopy_failed".to_owned(),
                        reason: failed.reason,
                        device: None,
                    },
                );
            }
            (Phase::Serving, _) if self.is_target(generation) => {
                if let Some(failure) = Failure::of_report(&report) {
                    self.restore(roster, failure);
                }
            }
            (Phase::Probation, _) if self.is_target(generation) => {
                if let Some(failure) = Failure::of_report(&report) {
                    self.reverting(roster, failure);
                }
            }
            (Phase::Committing, Some(Kind::Committed(_))) if self.is_target(generation) => {
                self.committed.insert(host);
            }
            (Phase::Committing, _) if self.is_target(generation) => {
                if let Some(failure) = Failure::of_report(&report) {
                    self.promotion_due = true;
                    self.update.fail(failure.clone());
                    self.failure = Some(failure);
                }
            }
            (Phase::Reverting, Some(Kind::Reverted(_))) if self.is_target(generation) => {
                self.reverted.insert(host);
            }
            _ => {}
        }
        self.progress(roster);
    }

    /// Fires the next phase's actions when the current one has ended.
    fn progress(&mut self, roster: &mut impl Roster) {
        let hosts: BTreeSet<String> = self.nodes.iter().map(|node| node.host.clone()).collect();
        match self.phase {
            Phase::Preparing if hosts.is_subset(&self.parked) => {
                self.phase = Phase::Promoting;
                self.switchover_started = Some(Instant::now());
                let generation = self.target_generation();
                for node in &self.nodes {
                    // One step, first, so that moving it with exec's answer is
                    // a one-line change.
                    roster.command(
                        &node.host,
                        generation,
                        command(process::command::Kind::Promote(process::Promote {})),
                    );
                }
            }
            Phase::Promoting if hosts.is_subset(&self.promoted) => {
                self.record.status = Status::Serving;
                self.epochs.record(self.target, self.record.clone());
                roster.serve(self.target, &self.requests);
                for node in &self.nodes {
                    roster.command(
                        &node.host,
                        self.previous_generation,
                        command(process::command::Kind::Drain(process::Drain {
                            epoch: self.previous,
                        })),
                    );
                }
                self.phase = Phase::Serving;
            }
            Phase::Serving if hosts.is_subset(&self.drained) => {
                self.pause = self.switchover_started.map(|started| started.elapsed());
                self.record.status = Status::Probation;
                self.epochs.record(self.target, self.record.clone());
                self.phase = Phase::Probation;
            }
            Phase::Committing if hosts.is_subset(&self.committed) => {
                self.record.status = Status::Serving;
                self.epochs.record(self.target, self.record.clone());
                let retire = vec![self.previous];
                let generation = self.target_generation();
                for node in &self.nodes {
                    roster.stop(&node.host, &self.group, Some(self.previous_generation));
                    roster.command(
                        &node.host,
                        generation,
                        command(process::command::Kind::Purge(process::Purge {
                            generation: self.previous_generation,
                        })),
                    );
                    // The previous world's storage ends on the node that
                    // served it while the target keeps its own.
                    let _ = roster.claim(&node.host, self.target, &self.scopes(node), &retire);
                }
                self.update.commit();
                self.phase = Phase::Purged;
            }
            Phase::Reverting if hosts.is_subset(&self.reverted) => {
                self.reverted(roster);
            }
            _ => {}
        }
    }
}

/// One command for a main process.
fn command(kind: process::command::Kind) -> process::Command {
    process::Command { kind: Some(kind) }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::epoch::{BuildConfig, Deployment, Program, UpdateState};
    use crate::options::ServeConfig;

    /// A roster that records what the switch asks of it and replays the
    /// events a test scripts.
    #[derive(Default)]
    struct Fake {
        claims: Vec<(String, u64, Vec<process::Scope>, Vec<u64>)>,
        spawns: Vec<Spawn>,
        stops: Vec<(String, String, Option<u64>)>,
        commands: Vec<(String, u64, process::command::Kind)>,
        serves: Vec<(u64, String)>,
        resumes: Vec<(String, u64, String)>,
        scripted: Vec<Event>,
        fail_spawn: Option<String>,
        fail_claim: Option<String>,
    }

    impl Fake {
        fn drains(&self) -> usize {
            self.commands
                .iter()
                .filter(|(_, _, kind)| matches!(kind, process::command::Kind::Drain(_)))
                .count()
        }

        fn commands_of<'a>(
            &'a self,
            generation: u64,
            matches: impl Fn(&process::command::Kind) -> bool + 'a,
        ) -> usize {
            self.commands
                .iter()
                .filter(|(_, which, kind)| *which == generation && matches(kind))
                .count()
        }
    }

    impl Roster for Fake {
        fn claim(
            &mut self,
            host: &str,
            world: u64,
            scopes: &[process::Scope],
            retire: &[u64],
        ) -> Result<(), String> {
            if let Some(reason) = &self.fail_claim {
                return Err(reason.clone());
            }
            self.claims
                .push((host.to_owned(), world, scopes.to_vec(), retire.to_vec()));
            Ok(())
        }

        fn spawn(&mut self, spawn: Spawn) -> Result<u32, String> {
            if let Some(reason) = &self.fail_spawn {
                return Err(reason.clone());
            }
            let pid = self.spawns.len() as u32 + 1;
            self.spawns.push(spawn);
            Ok(pid)
        }

        fn stop(&mut self, host: &str, group: &str, generation: Option<u64>) {
            self.stops
                .push((host.to_owned(), group.to_owned(), generation));
        }

        fn command(&mut self, host: &str, generation: u64, command: process::Command) {
            self.commands.push((
                host.to_owned(),
                generation,
                command.kind.expect("a command"),
            ));
        }

        fn serve(&mut self, epoch: u64, requests: &str) {
            self.serves.push((epoch, requests.to_owned()))
        }

        fn resume(&mut self, host: &str, generation: u64, reason: &str) {
            self.resumes
                .push((host.to_owned(), generation, reason.to_owned()));
        }

        fn events(&mut self) -> Vec<Event> {
            std::mem::take(&mut self.scripted)
        }
    }

    /// The record a target epoch starts with.
    fn record() -> EpochRecord {
        let config = ServeConfig::default();
        EpochRecord {
            build_config: BuildConfig::of(&config),
            kernels: BTreeMap::new(),
            policy: Program {
                backend: "fcfs".to_owned(),
                revision: 0,
                arguments: HashMap::new(),
            },
            previous_kernels: BTreeMap::new(),
            previous_policy: None,
            status: Status::Serving,
        }
    }

    fn nodes() -> Vec<Node> {
        vec![
            Node {
                host: "n0".to_owned(),
                rank: 0,
            },
            Node {
                host: "n1".to_owned(),
                rank: 1,
            },
        ]
    }

    fn start() -> Start {
        Start {
            executable: "/opt/uniserve".to_owned(),
            build: Some("2.0.0+new".to_owned()),
            arguments: vec!["engine".to_owned()],
            environment: HashMap::new(),
        }
    }

    fn setup() -> (Arc<Epochs>, Switch) {
        let epochs = Arc::new(Epochs::new());
        let switch = Switch::new(
            Arc::clone(&epochs),
            Update::new(7, Deployment::of(ServeConfig::default())),
            12,
            record(),
            11,
            5,
            "engine:0".to_owned(),
            nodes(),
            start(),
            "unix:/run/requests.sock".to_owned(),
        );
        (epochs, switch)
    }

    fn report(kind: process::report::Kind) -> Report {
        Report { kind: Some(kind) }
    }

    fn report_event(host: &str, generation: u64, report: Report) -> Event {
        Event::Report {
            host: host.to_owned(),
            generation,
            report,
        }
    }

    fn hello(host: &str, generation: u64, build: &str) -> Event {
        Event::Hello {
            host: host.to_owned(),
            generation,
            build: build.to_owned(),
        }
    }

    fn exit_event(host: &str, generation: u64) -> Event {
        Event::Exit {
            host: host.to_owned(),
            generation,
            code: Some(139),
        }
    }

    /// Spawns, claims and reports the targets parked.
    fn prepared(roster: &mut Fake, switch: &mut Switch) {
        switch.start(roster);
        roster.scripted = vec![hello("n0", 6, "2.0.0+new"), hello("n1", 6, "2.0.0+new")];
        switch.advance(roster);
        switch.parked(roster, "n0", 6);
        switch.parked(roster, "n1", 6);
    }

    /// Promotes on every node and drains the old epoch away.
    fn serve_target(roster: &mut Fake, switch: &mut Switch) {
        prepared(roster, switch);
        switch.promoted(roster, "n0", 6);
        switch.promoted(roster, "n1", 6);
        switch.drained(roster, "n0", 11);
        switch.drained(roster, "n1", 11);
    }

    /// Front's order: spawn, claim, `Parked`, `Promote`, `Promoted`, `Serve`,
    /// `Drain`, `Drained`, and the record moves `Precopy` → `Serving` →
    /// `Probation` only after the drain.
    #[test]
    fn the_switch_moves_through_its_phases_in_order() {
        let (epochs, mut switch) = setup();
        let mut roster = Fake::default();
        switch.start(&mut roster);

        assert_eq!(roster.spawns.len(), 2);
        for (index, spawn) in roster.spawns.iter().enumerate() {
            assert_eq!(spawn.host, nodes()[index].host);
            assert_eq!(spawn.group, "engine:0");
            assert_eq!(spawn.rank, nodes()[index].rank);
            assert_eq!(spawn.generation, 6, "the serving generation's next one");
            assert_eq!(spawn.executable, "/opt/uniserve");
            assert_eq!(spawn.arguments, vec!["engine".to_owned()]);
        }
        assert_eq!(roster.claims.len(), 2);
        for (index, (_, world, scopes, retire)) in roster.claims.iter().enumerate() {
            assert_eq!(*world, 12);
            assert!(retire.is_empty());
            assert_eq!(
                scopes,
                &[process::Scope {
                    group: "engine:0".to_owned(),
                    rank: nodes()[index].rank,
                }]
            );
        }
        assert_eq!(switch.phase(), Phase::Preparing);
        assert_eq!(
            epochs.get(12).map(|record| record.status),
            Some(Status::Precopy)
        );

        roster.scripted = vec![hello("n0", 6, "2.0.0+new"), hello("n1", 6, "2.0.0+new")];
        switch.advance(&mut roster);
        switch.parked(&mut roster, "n0", 6);
        switch.parked(&mut roster, "n1", 6);
        assert_eq!(switch.phase(), Phase::Promoting);
        assert_eq!(switch.generation(), Some(6));
        assert_eq!(roster.commands.len(), 2);
        assert!(
            roster
                .commands
                .iter()
                .all(|(_, generation, kind)| *generation == 6
                    && matches!(kind, process::command::Kind::Promote(_)))
        );

        switch.promoted(&mut roster, "n0", 6);
        switch.promoted(&mut roster, "n1", 6);
        assert_eq!(switch.phase(), Phase::Serving);
        assert_eq!(
            epochs.get(12).map(|record| record.status),
            Some(Status::Serving)
        );
        assert_eq!(
            roster.serves,
            vec![(12, "unix:/run/requests.sock".to_owned())]
        );
        assert_eq!(roster.drains(), 2, "the old epoch on every node");
        assert!(
            roster
                .commands
                .iter()
                .filter(|(_, _, kind)| matches!(kind, process::command::Kind::Drain(_)))
                .all(|(_, generation, _)| *generation == 5)
        );

        switch.drained(&mut roster, "n0", 11);
        assert_eq!(
            switch.phase(),
            Phase::Serving,
            "one host's report does not end the drain"
        );
        switch.drained(&mut roster, "n1", 11);
        assert_eq!(switch.phase(), Phase::Probation);
        assert_eq!(
            epochs.get(12).map(|record| record.status),
            Some(Status::Probation)
        );
        assert!(switch.pause().is_some());
    }

    /// A failed precopy drops the target, retires its world through the next
    /// claim, and leaves the serving epoch alone: the update reverts with no
    /// failure, so applying it again needs no `force`.
    #[test]
    fn a_failed_precopy_leaves_serving_alone() {
        let (epochs, mut switch) = setup();
        let mut roster = Fake::default();
        switch.start(&mut roster);
        roster.scripted = vec![report_event(
            "n0",
            6,
            report(process::report::Kind::PrecopyFailed(
                process::PrecopyFailed {
                    reason: "no storage".to_owned(),
                },
            )),
        )];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(switch.update().state, UpdateState::Reverted);
        assert!(switch.update().unavailable().is_none());
        assert!(roster.commands.is_empty(), "no promote, drain or commit");
        assert!(roster.serves.is_empty(), "the frontend never switched");
        assert_eq!(switch.discarded(), &[12]);
        assert_eq!(
            epochs.get(12).map(|record| record.status),
            Some(Status::Purged)
        );
        assert!(
            roster
                .stops
                .contains(&("n0".to_owned(), "engine:0".to_owned(), Some(6)))
        );
    }

    /// The daemon's own executable's build is known before it connects; a
    /// child of another build is stopped and the switch is abandoned.
    #[test]
    fn a_child_of_another_build_is_stopped() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        switch.start(&mut roster);
        roster.scripted = vec![hello("n0", 6, "2.0.0+old")];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(
            switch.failure().map(|failure| failure.kind.as_str()),
            Some("build_mismatch")
        );
        assert!(
            roster
                .stops
                .contains(&("n0".to_owned(), "engine:0".to_owned(), Some(6)))
        );
        assert!(roster.commands.is_empty());
    }

    /// Another executable's build is unknown until its first child connects:
    /// that child fixes the epoch's build, and a second child of another
    /// build is stopped.
    #[test]
    fn an_unknown_build_is_fixed_by_the_first_child() {
        let (epochs, mut switch) = setup();
        let mut roster = Fake::default();
        switch.start(&mut roster);
        roster.scripted = vec![hello("n0", 6, "3.0.0+theirs")];
        switch.advance(&mut roster);
        assert_eq!(switch.phase(), Phase::Preparing);
        assert_eq!(
            epochs.get(12).map(|record| record.build_config.executable),
            Some("3.0.0+theirs".to_owned())
        );

        roster.scripted = vec![hello("n1", 6, "3.0.0+other")];
        switch.advance(&mut roster);
        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(
            switch.failure().map(|failure| failure.kind.as_str()),
            Some("build_mismatch")
        );
        assert_eq!(switch.discarded(), &[12]);
    }

    /// A child that names a generation older than the one the target was
    /// spawned at is another epoch's: it is refused, stopped, and the target
    /// dropped. A later generation is this epoch's and is accepted.
    #[test]
    fn an_older_generation_is_refused() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        switch.start(&mut roster);
        switch.parked(&mut roster, "n0", 7);
        assert_eq!(switch.generation(), Some(7));
        switch.parked(&mut roster, "n1", 4);
        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(
            switch.failure().map(|failure| failure.kind.as_str()),
            Some("generation_mismatch")
        );
        assert!(
            roster
                .stops
                .contains(&("n1".to_owned(), "engine:0".to_owned(), Some(4)))
        );
    }

    /// A target that dies after `Commit` was sent is replaced by a promoted
    /// or relaunched main process; its `Committed` comes from that newer
    /// generation and still completes the commit.
    #[test]
    fn a_commit_acknowledged_by_a_promoted_generation_completes() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        serve_target(&mut roster, &mut switch);
        switch.commit(&mut roster);
        roster.scripted = vec![
            exit_event("n1", 6),
            report_event(
                "n0",
                7,
                report(process::report::Kind::Committed(process::Committed {})),
            ),
            report_event(
                "n1",
                7,
                report(process::report::Kind::Committed(process::Committed {})),
            ),
        ];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Purged);
        assert_eq!(switch.update().state, UpdateState::Committed);
        assert_eq!(switch.generation(), Some(7));
        assert!(
            roster
                .stops
                .iter()
                .all(|(_, _, generation)| *generation == Some(5))
        );
    }

    /// A revert after the previous epoch's main process died waits for its
    /// replacement to serve before the frontend is routed there.
    #[test]
    fn a_restore_waits_for_the_previous_epoch_to_be_replaced() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        prepared(&mut roster, &mut switch);
        switch.promoted(&mut roster, "n0", 6);
        switch.promoted(&mut roster, "n1", 6);
        roster.scripted = vec![exit_event("n0", 5)];
        switch.advance(&mut roster);
        switch.drained(&mut roster, "n1", 11);
        assert_eq!(switch.phase(), Phase::Probation);
        assert!(switch.promotion_due());
        roster.serves.clear();

        roster.scripted = vec![report_event(
            "n1",
            6,
            report(process::report::Kind::Fault(process::Fault {
                kind: "sticky".to_owned(),
                reason: "an illegal address".to_owned(),
                device: 0,
            })),
        )];
        switch.advance(&mut roster);
        assert_eq!(switch.phase(), Phase::Reverting);
        assert!(
            roster.serves.is_empty(),
            "the replacement of the dead epoch must serve first"
        );

        switch.serving(&mut roster, "n0", 7);
        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(
            roster.serves,
            vec![(11, "unix:/run/requests.sock".to_owned())]
        );
        assert!(
            roster
                .resumes
                .iter()
                .all(|(_, generation, _)| *generation == 7)
        );
    }

    /// A target that dies after the frontend was told it serves does not
    /// leave the deployment unserved: the previous epoch admits again, the
    /// frontend serves it, and the target is stopped and retired.
    #[test]
    fn a_target_death_while_serving_restores_the_previous_epoch() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        serve_target(&mut roster, &mut switch);
        assert_eq!(switch.phase(), Phase::Probation);
        roster.commands.clear();
        roster.serves.clear();
        roster.claims.clear();
        roster.resumes.clear();

        roster.scripted = vec![exit_event("n0", 6)];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(roster.resumes.len(), 2, "the old epoch admits on each node");
        assert!(
            roster
                .resumes
                .iter()
                .all(|(_, generation, _)| *generation == 5)
        );
        assert_eq!(
            roster.serves,
            vec![(11, "unix:/run/requests.sock".to_owned())]
        );
        assert!(
            roster
                .stops
                .contains(&("n0".to_owned(), "engine:0".to_owned(), Some(6)))
        );
        assert!(
            roster
                .claims
                .iter()
                .all(|(_, world, _, retire)| *world == 11 && retire == &[12])
        );
        assert_eq!(
            switch
                .update()
                .unavailable()
                .map(|failure| failure.kind.as_str()),
            Some("process_exit")
        );
    }

    /// A failure the target reports on probation records the failure itself —
    /// the operator sees `sticky`, not a generic revert — reverts the epoch,
    /// marks the configuration unavailable and stops the target.
    #[test]
    fn a_fault_on_probation_reverts_and_marks_unavailable() {
        let (epochs, mut switch) = setup();
        let mut roster = Fake::default();
        serve_target(&mut roster, &mut switch);
        roster.commands.clear();
        roster.serves.clear();

        roster.scripted = vec![report_event(
            "n1",
            6,
            report(process::report::Kind::Fault(process::Fault {
                kind: "sticky".to_owned(),
                reason: "an illegal address".to_owned(),
                device: 3,
            })),
        )];
        switch.advance(&mut roster);
        assert_eq!(switch.phase(), Phase::Reverting);
        assert_eq!(roster.commands.len(), 2, "a revert per node");
        assert!(roster.commands.iter().all(|(_, generation, kind)| {
            *generation == 6 && matches!(kind, process::command::Kind::Revert(_))
        }));
        assert!(roster.serves.is_empty(), "the frontend waits for the ack");

        roster.scripted = vec![
            report_event(
                "n0",
                6,
                report(process::report::Kind::Reverted(process::Reverted {
                    reason: String::new(),
                })),
            ),
            report_event(
                "n1",
                6,
                report(process::report::Kind::Reverted(process::Reverted {
                    reason: String::new(),
                })),
            ),
        ];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(
            roster.serves,
            vec![(11, "unix:/run/requests.sock".to_owned())]
        );
        assert_eq!(switch.update().state, UpdateState::Reverted);
        assert_eq!(
            switch
                .update()
                .unavailable()
                .map(|failure| failure.kind.as_str()),
            Some("sticky"),
            "the first failure's own kind"
        );
        assert_eq!(switch.failure().and_then(|failure| failure.device), Some(3));
        assert_eq!(
            epochs.get(12).map(|record| record.status),
            Some(Status::Purged)
        );
        assert!(
            roster
                .claims
                .iter()
                .all(|(_, world, _, retire)| *world == 11 && retire == &[12])
        );
    }

    /// A commit records the target as serving once it acknowledges, stops the
    /// old generation, purges its programs and retires its storage.
    #[test]
    fn a_commit_retires_the_previous_epoch() {
        let (epochs, mut switch) = setup();
        let mut roster = Fake::default();
        serve_target(&mut roster, &mut switch);
        roster.commands.clear();
        roster.claims.clear();

        switch.commit(&mut roster);
        assert_eq!(switch.phase(), Phase::Committing);
        assert_eq!(roster.commands.len(), 2, "a commit per node");
        assert_eq!(
            epochs.get(12).map(|record| record.status),
            Some(Status::Probation),
            "the target serves on probation until it acknowledges"
        );

        roster.scripted = vec![
            report_event(
                "n0",
                6,
                report(process::report::Kind::Committed(process::Committed {})),
            ),
            report_event(
                "n1",
                6,
                report(process::report::Kind::Committed(process::Committed {})),
            ),
        ];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Purged);
        assert_eq!(switch.update().state, UpdateState::Committed);
        assert_eq!(
            epochs.get(12).map(|record| record.status),
            Some(Status::Serving)
        );
        assert_eq!(roster.stops.len(), 2, "the old generation on each node");
        assert!(
            roster
                .stops
                .iter()
                .all(|(_, group, generation)| group == "engine:0" && *generation == Some(5))
        );
        assert_eq!(
            roster.commands_of(6, |kind| matches!(kind, process::command::Kind::Purge(_))),
            2
        );
        assert_eq!(roster.claims.len(), 2, "each node retires the old world");
        assert!(
            roster
                .claims
                .iter()
                .all(|(_, world, _, retire)| *world == 12 && retire == &[11])
        );
    }

    /// A target that dies after `Commit` was sent keeps the commit pending
    /// and hands the failure domain to the promotion; the switch does not
    /// hang waiting for an acknowledgement that will not come.
    #[test]
    fn a_target_death_during_commit_keeps_it_pending() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        serve_target(&mut roster, &mut switch);
        switch.commit(&mut roster);
        roster.scripted = vec![exit_event("n1", 6)];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Committing);
        assert!(switch.promotion_due());
        assert_eq!(
            switch.update().unavailable(),
            None,
            "a commit in flight is not a revert"
        );
        assert_eq!(
            switch.failure().map(|failure| failure.kind.as_str()),
            Some("process_exit")
        );
    }

    /// A serving main process that exits during the preparing hands the
    /// failure domain to the standby promotion instead of reverting the
    /// update.
    #[test]
    fn a_serving_exit_during_preparing_hands_over_to_promotion() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        switch.start(&mut roster);
        roster.scripted = vec![exit_event("n1", 5)];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Abandoned);
        assert!(switch.promotion_due());
        assert!(
            roster.serves.is_empty(),
            "the promotion path serves, not the switch"
        );
        assert_eq!(switch.discarded(), &[12], "the target world still retires");
    }

    /// A claim the daemon refuses, or a spawn it cannot start, drops the
    /// target before anything else starts.
    #[test]
    fn a_refused_claim_or_spawn_drops_the_target() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake {
            fail_claim: Some("the world is not feasible".to_owned()),
            ..Fake::default()
        };
        switch.start(&mut roster);
        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(roster.spawns.len(), 2, "the children are stopped again");
        assert_eq!(
            switch.failure().map(|failure| failure.kind.as_str()),
            Some("claim_failed")
        );

        let (_epochs, mut switch) = setup();
        let mut roster = Fake {
            fail_spawn: Some("no such executable".to_owned()),
            ..Fake::default()
        };
        switch.start(&mut roster);
        assert_eq!(switch.phase(), Phase::Abandoned);
        assert!(
            roster
                .stops
                .contains(&("n0".to_owned(), "engine:0".to_owned(), Some(6)))
        );
    }
}
