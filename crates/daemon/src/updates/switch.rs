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
//!    not a drop: the previous epoch must serve again, and it was parked at
//!    the switchover rather than ended (§7.4).
//! 4. **Probation**, recorded once the drain completed: a failure the target
//!    reports, or a target process that exits, reverts.
//! 5. **Commit** records the target as serving once it acknowledges, stops
//!    the old generation and purges and retires the previous world; **revert**
//!    commands the target back (`Revert`), returns the parked previous epoch
//!    to serving once every target acknowledged, serves the frontend only
//!    once every host serves, stops the target generation, records it purged
//!    and marks the target configuration
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
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Start {
    /// The target main process's executable.
    pub(crate) executable: String,
    /// The build identity the target must report in its `Hello`: `Some` when
    /// the executable is the daemon's own (`proto::BUILD`), `None` for
    /// another executable, whose first child's `Hello` fixes it and every
    /// other child must then report the same.
    pub(crate) build: Option<String>,
    /// The launch a process of this epoch is started with: the contents of
    /// its launch file, which the adapter writes for the process it starts
    /// (`engine/launch.rs`). Every field is the epoch's; the process's own
    /// identity is the generation the spawn carries, which the adapter puts
    /// into the file's `engine` (§2.3, §7.4).
    pub(crate) launch: crate::engine::launch::Launch,
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

/// What a spawn produced: the process's own launch generation, which is the
/// number the supervisor keys its launches by, and its pid. The switch keeps
/// the generation, because it is what a later stop names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Spawned {
    pub(crate) generation: u64,
    pub(crate) pid: u32,
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
    /// Starts `spawn` and returns the child's own generation and process id.
    fn spawn(
        &mut self,
        spawn: Spawn,
        launch: &crate::engine::launch::Launch,
    ) -> Result<Spawned, String>;
    /// Stops one launch generation of `group` on `host`; every generation
    /// when `generation` is `None`.
    fn stop(&mut self, host: &str, group: &str, generation: Option<u64>);
    /// Commands the main process of `generation` on `host`.
    fn command(&mut self, host: &str, generation: u64, command: process::Command);
    /// Commands the frontend to serve `epoch` from `requests`.
    fn serve(&mut self, epoch: u64, requests: &str);
    /// Returns the previous epoch at `host` to serving.
    ///
    /// Its main processes were parked at the switchover, not ended (§7.4:
    /// the serving epoch is parked at the step boundary, and the old epoch's
    /// main processes exit at purge), so the normal path is `Serve { epoch:
    /// previous }` to them, at the launch generation they always had — the
    /// model generation travels inside the command, not in `generation`.
    /// Only when those processes are gone does the epoch layer start a fresh
    /// generation instead (spawn, claim, park, promote), which the switch
    /// waits for exactly as it waits for the parked ones. Either way the
    /// switch learns the host serves through `serving`.
    fn resume(&mut self, host: &str, generation: u64, reason: &str);
    /// Takes the events that arrived since the last call.
    fn events(&mut self) -> Vec<Event>;
}

/// A boxed roster commands exactly as the roster inside it, so the daemon can
/// hold one trait object for the adapter it constructs once.
impl<T: Roster + ?Sized> Roster for Box<T> {
    fn claim(
        &mut self,
        host: &str,
        world: u64,
        scopes: &[process::Scope],
        retire: &[u64],
    ) -> Result<(), String> {
        (**self).claim(host, world, scopes, retire)
    }

    fn spawn(
        &mut self,
        spawn: Spawn,
        launch: &crate::engine::launch::Launch,
    ) -> Result<Spawned, String> {
        (**self).spawn(spawn, launch)
    }

    fn stop(&mut self, host: &str, group: &str, generation: Option<u64>) {
        (**self).stop(host, group, generation)
    }

    fn command(&mut self, host: &str, generation: u64, command: process::Command) {
        (**self).command(host, generation, command)
    }

    fn serve(&mut self, epoch: u64, requests: &str) {
        (**self).serve(epoch, requests)
    }

    fn resume(&mut self, host: &str, generation: u64, reason: &str) {
        (**self).resume(host, generation, reason)
    }

    fn events(&mut self) -> Vec<Event> {
        (**self).events()
    }
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
    /// The update reverted: the targets acknowledge `Revert` (or one died),
    /// and the previous epoch returns to serving before the frontend is
    /// served it.
    Reverting,
    /// The update was committed.
    Purged,
    /// The update returned to the previous epoch.
    Abandoned,
}

impl Phase {
    /// The phase as the status reports it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Preparing => "preparing",
            Self::Promoting => "promoting",
            Self::Serving => "serving",
            Self::Probation => "probation",
            Self::Committing => "committing",
            Self::Reverting => "reverting",
            Self::Purged => "purged",
            Self::Abandoned => "abandoned",
        }
    }
}

/// One epoch switch.
pub(crate) struct Switch {
    epochs: Arc<Epochs>,
    update: Update,
    /// The target epoch's number, which the record is kept under.
    target: u64,
    /// The previous epoch's number, which `Serve` and `Drain` name.
    previous: u64,
    /// The launch generation the target's children are started at.
    ///
    /// The epoch layer chooses it and owes the switch one property: it is
    /// above every live generation of the group and rank on every node,
    /// including the serving epoch's parked standby, which occupies
    /// `previous_generation + 1` (§2.5). A promoted standby must never reach
    /// this number, or one number would name two epochs.
    spawn_generation: u64,
    /// What the roster actually started: host and the process's own launch
    /// generation, which is what a stop names. A target that fails before it
    /// reports has no model generation yet, so the switch stops these.
    spawned: Vec<(String, u64)>,
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
    /// The hosts whose target main process was told to park: a command can
    /// only be addressed to a generation that has connected, so each park is
    /// sent once, on that child's `Hello`.
    park_sent: BTreeSet<String>,
    parked: BTreeSet<String>,
    promoted: BTreeSet<String>,
    drained: BTreeSet<String>,
    /// Hosts whose previous-epoch main process reported serving again after
    /// the return switch started it.
    serving: BTreeSet<String>,
    /// Generations reported as the previous epoch's: a fresh generation is
    /// minted above the target's spawn generation, so the number alone no
    /// longer tells the two epochs apart.
    previous_serving: BTreeSet<u64>,
    /// The reason a restore is under way: the previous epoch returns to
    /// serving before the frontend is served it.
    pending_restore: Option<Failure>,
    committed: BTreeSet<String>,
    /// Hosts whose target main process acknowledged its revert.
    target_reverted: BTreeSet<String>,
    /// Hosts whose target main process exited: they cannot be commanded, and
    /// their failure is what started the return.
    target_dead: BTreeSet<String>,
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
    /// to epoch `target`, whose record is `record`, with the target's
    /// children started at `spawn_generation` — above every live generation
    /// of the group and rank, the serving epoch's standby included —
    /// retiring `discarded`, the worlds an earlier switch dropped.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        epochs: Arc<Epochs>,
        update: Update,
        target: u64,
        record: EpochRecord,
        previous: u64,
        previous_generation: u64,
        spawn_generation: u64,
        group: String,
        nodes: Vec<Node>,
        start: Start,
        requests: String,
        discarded: Vec<u64>,
    ) -> Self {
        let mut record = record;
        record.status = Status::Precopy;
        epochs.record(target, record.clone());
        Self {
            epochs,
            update,
            target,
            previous,
            spawn_generation,
            spawned: Vec::new(),
            previous_generation,
            generation: None,
            record,
            group,
            nodes,
            start: start.clone(),
            build: start.build,
            requests,
            phase: Phase::Preparing,
            retire: discarded,
            park_sent: BTreeSet::new(),
            parked: BTreeSet::new(),
            promoted: BTreeSet::new(),
            drained: BTreeSet::new(),
            serving: BTreeSet::new(),
            previous_serving: BTreeSet::new(),
            pending_restore: None,
            committed: BTreeSet::new(),
            target_reverted: BTreeSet::new(),
            target_dead: BTreeSet::new(),
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
    /// relaunched child reports the next one, unless it was already reported
    /// as the returning previous epoch's.
    fn is_target(&self, generation: u64) -> bool {
        generation >= self.spawn_generation && !self.previous_serving.contains(&generation)
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
                // The adapter writes the launch file for this process and
                // passes its path: the arguments are its to state.
                arguments: Vec::new(),
                environment: self.start.environment.clone(),
                inherit: Vec::new(),
            };
            let spawned = match roster.spawn(spawn, &self.start.launch) {
                Ok(spawned) => spawned,
                Err(reason) => {
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
            };
            self.spawned.push((node.host.clone(), spawned.generation));
        }
        let retire = std::mem::take(&mut self.retire);
        for node in &self.nodes {
            // A claim retires storage on the node it is made on, so every
            // node is told the same worlds.
            if let Err(reason) = roster.claim(&node.host, self.target, &self.scopes(node), &retire)
            {
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
        if self.phase != Phase::Preparing || !self.accept(generation) {
            return;
        }
        self.parked.insert(host.to_owned());
        self.progress(roster);
    }

    /// The target's main process of `generation` on `host` is promoted and
    /// serves in place (`Report::Promoted`).
    pub(crate) fn promoted(&mut self, roster: &mut impl Roster, host: &str, generation: u64) {
        if self.phase != Phase::Promoting || !self.accept(generation) {
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
        // A fresh generation is minted above the target's, so the number is
        // recorded as the previous epoch's before any report could be read as
        // the target's.
        self.previous_serving.insert(generation);
        self.previous_generation = self.previous_generation.max(generation);
        self.serving.insert(host.to_owned());
        let hosts: BTreeSet<String> = self.nodes.iter().map(|node| node.host.clone()).collect();
        if hosts.is_subset(&self.serving) && self.pending_restore.take().is_some() {
            self.finish_restore(roster);
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

    /// The generation `host` actually started the target at: the number the
    /// node's allocator gave that child, which its reports carry and its
    /// commands are addressed to. `None` before `start` spawned it.
    fn spawned_generation(&self, host: &str) -> Option<u64> {
        self.spawned
            .iter()
            .find(|(spawned, _)| spawned == host)
            .map(|(_, generation)| *generation)
    }

    /// Accepts `generation` as a target main process's: the child numbers the
    /// generation it prepared, so any number from the one the switch spawned
    /// on names this epoch, and the highest reported is its current
    /// generation. A lower number belongs to another epoch — the serving
    /// epoch or its promoted standby — and is ignored.
    fn accept(&mut self, generation: u64) -> bool {
        if generation < self.spawn_generation {
            return false;
        }
        if self.generation.is_none_or(|recorded| generation > recorded) {
            self.generation = Some(generation);
        }
        true
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
        // The target's processes are stopped by the generation each one was
        // actually started at: before the first report no model generation
        // exists, and the epoch's spawn generation is only a floor.
        if self.spawned.is_empty() {
            let generation = self.target_generation();
            for node in &self.nodes {
                roster.stop(&node.host, &self.group, Some(generation));
            }
        } else {
            for (host, generation) in &self.spawned {
                roster.stop(host, &self.group, Some(*generation));
            }
        }
        self.phase = Phase::Abandoned;
    }

    /// Reverts a target that serves on probation: `Revert` is the update
    /// layer's decision, so the target ends its trial and takes the previous
    /// generation back, and the first reason is what marks the configuration
    /// unavailable (§7.3 step 4). The epoch that served before the
    /// switchover is parked, not ended (§7.4), so it can be served again once
    /// every target acknowledges the revert.
    fn reverting(&mut self, roster: &mut impl Roster, failure: Failure) {
        self.update.fail(failure.clone());
        let reason = failure.reason.clone();
        self.failure = Some(failure);
        self.phase = Phase::Reverting;
        let generation = self.target_generation();
        for node in &self.nodes {
            roster.command(
                &node.host,
                generation,
                command(process::command::Kind::Revert(process::Revert {
                    reason: reason.clone(),
                })),
            );
        }
    }

    /// Returns to the previous epoch after the frontend was told the target
    /// serves: the previous epoch admits again, the frontend serves it, and
    /// the target is stopped and retired.
    /// Starts the return to the previous epoch after the target died or
    /// faulted while serving: every target that still runs is reverted, so no
    /// host keeps serving the trial, and then every host returns the parked
    /// previous epoch to serving (§7.4). The frontend is served it only once
    /// every host says it serves.
    fn restore(&mut self, roster: &mut impl Roster, failure: Failure) {
        self.update.fail(failure.clone());
        let reason = failure.reason.clone();
        self.failure = Some(failure.clone());
        self.pending_restore = Some(failure);
        self.phase = Phase::Reverting;
        let generation = self.target_generation();
        for node in &self.nodes {
            // A dead target is not commanded: its exit is the failure.
            if self.target_dead.contains(&node.host) {
                continue;
            }
            roster.command(
                &node.host,
                generation,
                command(process::command::Kind::Revert(process::Revert {
                    reason: reason.clone(),
                })),
            );
        }
        self.resume(roster, &reason);
    }

    /// Serves the previous epoch once every host reports it serves, and drops
    /// the target.
    fn finish_restore(&mut self, roster: &mut impl Roster) {
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

    /// Every host returns its previous epoch to serving: `Serve { epoch:
    /// previous }` to the processes the switchover parked, or a fresh
    /// generation where they are gone.
    fn resume(&mut self, roster: &mut impl Roster, reason: &str) {
        for node in &self.nodes {
            roster.resume(&node.host, self.previous_generation, reason);
        }
    }

    /// Every target acknowledged its revert: the previous epoch returns to
    /// serving on every host, and the frontend is served it when every host
    /// reports serving.
    fn reverted(&mut self, roster: &mut impl Roster) {
        let failure = self.failure.clone().unwrap_or_else(|| Failure {
            kind: "reverted".to_owned(),
            reason: String::new(),
            device: None,
        });
        let reason = failure.reason.clone();
        self.pending_restore = Some(failure);
        self.resume(roster, &reason);
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
        // The build is a precondition of promoting the target, not of a
        // replacement of a target that already serves.
        if !matches!(self.phase, Phase::Preparing | Phase::Promoting) || !self.is_target(generation)
        {
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
        // §7.4 step 1: the child claimed its scopes read-only as the
        // predecessor's standby and parks at the end of its start; `Park` is
        // what turns that into the `Parked` this phase waits for. It is sent
        // now, on the child's `Hello`, because a command is addressed to a
        // generation the roster has connected and a child that was just forked
        // has not. One step per node, at the generation the node allocated, and
        // the child buffers it until its serve loop runs.
        if self.phase == Phase::Preparing && self.park_sent.insert(host.clone()) {
            let generation = self.spawned_generation(&host).unwrap_or(generation);
            roster.command(
                &host,
                generation,
                command(process::command::Kind::Park(process::Park {})),
            );
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
                    self.promotion_due = true;
                    self.failed(roster, failure);
                }
                Phase::Serving => {
                    // This host has nothing left to drain, and the epoch
                    // must be replaced before it could serve again.
                    self.promotion_due = true;
                    self.drained.insert(host);
                    self.progress(roster);
                }
                Phase::Probation | Phase::Committing | Phase::Reverting => {
                    self.promotion_due = true;
                }
                _ => {}
            }
        } else if self.is_target(generation) {
            self.target_dead.insert(host.clone());
            match self.phase {
                Phase::Preparing | Phase::Promoting => self.failed(roster, failure),
                Phase::Serving => self.restore(roster, failure),
                // A dead target cannot acknowledge a revert: the previous
                // epoch must serve again instead.
                Phase::Probation => self.restore(roster, failure),
                Phase::Reverting if self.pending_restore.is_none() => {
                    // The target's process ended instead of acknowledging its
                    // revert: it holds nothing on this host all the same.
                    self.target_reverted.insert(host);
                    self.progress(roster);
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
        // Whichever report it is, it comes from a process of that generation:
        // the highest one seen is the epoch's current generation, which the
        // commit's and the revert's acks therefore keep up to date. While the
        // return is under way the epoch's generation is settled — a successor
        // of the previous epoch reports from a number above the target's, and
        // it must not become the generation the target is stopped at.
        if self.pending_restore.is_none() && self.is_target(generation) {
            self.accept(generation);
        }
        // The kind drives the phase; the whole report still states a failure.
        let kind = report.kind.clone();
        match (self.phase, kind) {
            // The two reports that say a child prepared and serves in
            // place: `Park` took effect, and `Promote` did. The child's
            // generation is the one it was spawned at.
            (Phase::Preparing, Some(Kind::Parked(parked))) if self.is_target(generation) => {
                self.parked(roster, &host, parked.generation);
            }
            (Phase::Promoting, Some(Kind::Promoted(promoted))) if self.is_target(generation) => {
                self.promoted(roster, &host, promoted.generation);
            }
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
            // A target that faults while it prepares — a park that could not
            // take, a device error — never becomes the epoch: the previous one
            // serves on and the target is dropped, exactly as a failed precopy
            // drops it. Without this arm the switch would wait in `Preparing`
            // for a readiness no process of that generation will ever send.
            (Phase::Preparing, _) if self.is_target(generation) => {
                if let Some(failure) = Failure::of_report(&report) {
                    self.failed(roster, failure);
                }
            }
            // The same while it is promoted: `Promote` resumes the target's
            // schedulers, and a device fault as they start is such a report.
            // The frontend is not served the target until every `Promoted`
            // arrives, so a half-promoted epoch serves nobody and is dropped.
            (Phase::Promoting, _) if self.is_target(generation) => {
                if let Some(failure) = Failure::of_report(&report) {
                    self.failed(roster, failure);
                }
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
            (Phase::Reverting, Some(Kind::Reverted(_)))
                if self.pending_restore.is_none() && self.is_target(generation) =>
            {
                self.target_reverted.insert(host);
            }
            // The returning epoch's own process says it serves: the parked one
            // served again, or the fresh generation the epoch layer started
            // where it was gone. It is not the target's generation, which the
            // return has already settled (`a48631c98`), and its number is read
            // by `serving` exactly as a host's report is.
            (Phase::Reverting, Some(Kind::Promoted(promoted)))
                if self.pending_restore.is_some()
                    && promoted.generation >= self.previous_generation
                    && Some(promoted.generation) != self.generation =>
            {
                self.serving(roster, &host, promoted.generation);
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
                for node in &self.nodes {
                    // One step per node, at the generation that node started
                    // its child at: a two-node epoch has two numbers, and the
                    // highest one addresses no process on the other node.
                    let generation = self
                        .spawned_generation(&node.host)
                        .unwrap_or_else(|| self.target_generation());
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
                    // One number addresses the previous epoch on every node,
                    // which is right while the epoch arm runs one node and one
                    // main process (`apply_epoch` refuses more): the switch
                    // never launched the previous epoch, so it holds no
                    // per-node number for it. Lifting that refusal means the
                    // previous epoch's generations must come from the process
                    // table the same way the target's come from `spawned`.
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
                for node in &self.nodes {
                    // Addressed per node for the same reason `Promote` is.
                    let generation = self
                        .spawned_generation(&node.host)
                        .unwrap_or_else(|| self.target_generation());
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
            Phase::Reverting
                if self.pending_restore.is_none() && hosts.is_subset(&self.target_reverted) =>
            {
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
        /// The generation the roster allocates for the first process it
        /// starts, and each one after it: the real adapter allocates every
        /// process's number, which is not the epoch's spawn generation.
        allocate_from: Option<u64>,
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

        fn spawn(
            &mut self,
            spawn: Spawn,
            _launch: &crate::engine::launch::Launch,
        ) -> Result<Spawned, String> {
            if let Some(reason) = &self.fail_spawn {
                return Err(reason.clone());
            }
            let pid = self.spawns.len() as u32 + 1;
            let generation = self
                .allocate_from
                .map_or(spawn.generation, |base| base + u64::from(pid) - 1);
            self.spawns.push(spawn);
            Ok(Spawned { generation, pid })
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
                arguments: toml::Table::new(),
            },
            previous_kernels: BTreeMap::new(),
            previous_policy: None,
            status: Status::Serving,
            generation: None,
            // The record is kept under epsilon 12 in this test's switch, and
            // `apply` fills the world allocation with the epoch number.
            world: 12,
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
            launch: crate::engine::launch::Launch {
                build_config: "/run/build-config-11.json".into(),
                model: "/models/qwen3".into(),
                requests: "unix:/run/requests.sock".to_owned(),
                processes: "unix:/run/processes.sock".to_owned(),
                engine: 7,
                program_server: "/run/programs.sock".to_owned(),
                devices: vec![0],
                group: "engine".to_owned(),
                rank: 0,
                deployment: "qwen3".to_owned(),
                world: 11,
                scheduler_node: true,
                peers: Vec::new(),
                watchdog_timeout_s: None,
                rollback_s: None,
                queue_depth: 4,
                world_size: 1,
                standby: false,
            },
            environment: HashMap::new(),
        }
    }

    fn setup() -> (Arc<Epochs>, Switch) {
        switch_with(start())
    }

    /// A switch started with `start`, for a test that needs another command
    /// than the usual one.
    fn switch_with(start: Start) -> (Arc<Epochs>, Switch) {
        let epochs = Arc::new(Epochs::new());
        let switch = Switch::new(
            Arc::clone(&epochs),
            Update::new(7, Deployment::of(ServeConfig::default())),
            12,
            record(),
            11,
            5,
            6,
            "engine:0".to_owned(),
            nodes(),
            start,
            "unix:/run/requests.sock".to_owned(),
            Vec::new(),
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
            // The adapter writes the launch file and states the argv that
            // reads it ([`crate::engine::launch`]): the switch starts a child,
            // not a command line, so it states none.
            assert!(
                spawn.arguments.is_empty(),
                "a spawn states no arguments: {spawn:?}"
            );
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
        assert!(
            roster.commands.is_empty(),
            "a child that was just forked cannot be commanded: it has not connected"
        );

        // The children connect, and each is told to park on the read-only
        // mapping it claimed. A repeated hello does not park it twice.
        roster.scripted = vec![
            hello("n0", 6, "2.0.0+new"),
            hello("n1", 6, "2.0.0+new"),
            hello("n0", 6, "2.0.0+new"),
        ];
        switch.advance(&mut roster);
        assert_eq!(switch.phase(), Phase::Preparing);
        assert_eq!(roster.commands.len(), 2, "one park per node");
        for (index, (host, generation, kind)) in roster.commands.iter().enumerate() {
            assert_eq!(host, &nodes()[index].host);
            assert_eq!(*generation, 6, "the generation the node spawned it at");
            assert!(matches!(kind, process::command::Kind::Park(_)));
        }
        assert!(
            !roster
                .commands
                .iter()
                .any(|(_, _, kind)| matches!(kind, process::command::Kind::Promote(_))),
            "nothing is promoted before the targets parked"
        );

        switch.parked(&mut roster, "n0", 6);
        assert_eq!(
            switch.phase(),
            Phase::Preparing,
            "one host's readiness promotes nothing"
        );
        switch.parked(&mut roster, "n1", 6);
        assert_eq!(switch.phase(), Phase::Promoting);
        assert_eq!(switch.generation(), Some(6));
        assert_eq!(roster.commands.len(), 4, "two parks, then two promotes");
        assert!(
            roster.commands[2..]
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

    /// The target's children park on their read-only mapping, and the switch
    /// waits for each of them: a readiness of another generation is not this
    /// target's, and no promotion is sent until every node parked.
    #[test]
    fn a_target_is_promoted_only_after_every_node_parked() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        switch.start(&mut roster);
        roster.scripted = vec![hello("n0", 6, "2.0.0+new"), hello("n1", 6, "2.0.0+new")];
        switch.advance(&mut roster);

        // `n0` reports the serving epoch's generation, not the target's.
        switch.parked(&mut roster, "n0", 5);
        switch.parked(&mut roster, "n1", 6);
        assert_eq!(
            switch.phase(),
            Phase::Preparing,
            "a foreign generation's readiness is not the target's"
        );
        assert!(
            !roster
                .commands
                .iter()
                .any(|(_, _, kind)| matches!(kind, process::command::Kind::Promote(_))),
            "nothing is promoted while a node has not parked"
        );

        switch.parked(&mut roster, "n0", 6);
        assert_eq!(switch.phase(), Phase::Promoting);
        assert_eq!(
            roster.commands_of(6, |kind| matches!(kind, process::command::Kind::Promote(_))),
            2
        );
    }

    /// A target that faults while it prepares never becomes the epoch: a park
    /// that could not take must not leave the switch waiting for a readiness
    /// no process of that generation will send.
    #[test]
    fn a_fault_while_preparing_drops_the_target() {
        let (epochs, mut switch) = setup();
        let mut roster = Fake::default();
        switch.start(&mut roster);
        roster.scripted = vec![
            hello("n0", 6, "2.0.0+new"),
            report_event(
                "n0",
                6,
                report(process::report::Kind::Fault(process::Fault {
                    kind: "sticky".to_owned(),
                    reason: "the park faulted the device".to_owned(),
                    device: 0,
                })),
            ),
        ];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(
            switch.failure().map(|failure| failure.kind.as_str()),
            Some("sticky")
        );
        assert_eq!(
            epochs.get(12).map(|record| record.status),
            Some(Status::Purged)
        );
        assert!(
            !roster
                .commands
                .iter()
                .any(|(_, _, kind)| matches!(kind, process::command::Kind::Promote(_))),
            "the failed target was never promoted"
        );
        assert!(
            roster
                .stops
                .contains(&("n0".to_owned(), "engine:0".to_owned(), Some(6)))
        );
    }

    /// Every node allocates its own launch generation, and a command reaches
    /// the process of the generation that node started: addressing the highest
    /// number any node reported would reach no process on the other node and
    /// the phase would wait for an answer that cannot come.
    #[test]
    fn each_node_is_commanded_at_the_generation_it_allocated() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake {
            allocate_from: Some(6),
            ..Fake::default()
        };
        switch.start(&mut roster);
        assert_eq!(
            roster.spawns.len(),
            2,
            "two nodes, each allocated its own generation"
        );

        roster.scripted = vec![hello("n0", 6, "2.0.0+new"), hello("n1", 7, "2.0.0+new")];
        switch.advance(&mut roster);
        let parks: Vec<_> = roster
            .commands
            .iter()
            .filter(|(_, _, kind)| matches!(kind, process::command::Kind::Park(_)))
            .map(|(host, generation, _)| (host.as_str(), *generation))
            .collect();
        assert_eq!(parks, vec![("n0", 6), ("n1", 7)]);

        switch.parked(&mut roster, "n0", 6);
        switch.parked(&mut roster, "n1", 7);
        assert_eq!(switch.phase(), Phase::Promoting);
        let promotes: Vec<_> = roster
            .commands
            .iter()
            .filter(|(_, _, kind)| matches!(kind, process::command::Kind::Promote(_)))
            .map(|(host, generation, _)| (host.as_str(), *generation))
            .collect();
        assert_eq!(
            promotes,
            vec![("n0", 6), ("n1", 7)],
            "each node's own generation, not the highest reported"
        );
    }

    /// A target that faults while it is promoted never serves: one node
    /// answers `Promoted` and the other faults as its schedulers start. The
    /// frontend is not served the target until every node answered, so
    /// dropping it leaves the previous epoch serving and the trial unstarted.
    #[test]
    fn a_fault_while_promoting_drops_the_target() {
        let (epochs, mut switch) = setup();
        let mut roster = Fake::default();
        switch.start(&mut roster);
        roster.scripted = vec![hello("n0", 6, "2.0.0+new"), hello("n1", 6, "2.0.0+new")];
        switch.advance(&mut roster);
        switch.parked(&mut roster, "n0", 6);
        switch.parked(&mut roster, "n1", 6);
        assert_eq!(switch.phase(), Phase::Promoting);

        switch.promoted(&mut roster, "n0", 6);
        assert_eq!(
            switch.phase(),
            Phase::Promoting,
            "one node's promotion does not serve the epoch"
        );
        roster.scripted = vec![report_event(
            "n1",
            6,
            report(process::report::Kind::Fault(process::Fault {
                kind: "illegal_address".to_owned(),
                reason: "the resumed schedulers faulted".to_owned(),
                device: 1,
            })),
        )];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(
            switch.failure().map(|failure| failure.kind.as_str()),
            Some("illegal_address")
        );
        assert_eq!(
            epochs.get(12).map(|record| record.status),
            Some(Status::Purged)
        );
        assert!(
            roster.serves.is_empty(),
            "the frontend never switched to the dropped epoch"
        );
        assert!(
            roster
                .stops
                .contains(&("n1".to_owned(), "engine:0".to_owned(), Some(6)))
        );
    }

    /// A world an earlier switch dropped is retired by the next switch's
    /// claim on every node: a claim retires storage on the node it is made
    /// on, so passing it to one node would leave the others holding it.
    #[test]
    fn a_discarded_world_is_retired_on_every_node() {
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
        assert_eq!(switch.discarded(), &[12]);

        let discarded = switch.discarded().to_vec();
        let mut next = Switch::new(
            Arc::clone(&epochs),
            Update::new(8, Deployment::of(ServeConfig::default())),
            13,
            record(),
            12,
            6,
            8,
            "engine:0".to_owned(),
            nodes(),
            start(),
            "unix:/run/requests.sock".to_owned(),
            discarded,
        );
        let mut roster = Fake::default();
        next.start(&mut roster);
        assert_eq!(roster.claims.len(), 2);
        assert!(
            roster
                .claims
                .iter()
                .all(|(_, world, _, retire)| *world == 13 && retire == &[12])
        );
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
        assert!(
            !roster.commands.iter().any(|(_, _, kind)| matches!(
                kind,
                process::command::Kind::Promote(_)
                    | process::command::Kind::Drain(_)
                    | process::command::Kind::Commit(_)
            )),
            "no promote, drain or commit"
        );
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

    /// The roster allocates every process's own generation, so a target that
    /// dies before it reports is stopped where it was actually started — not
    /// at the epoch's spawn generation, which no process need ever have had.
    #[test]
    fn a_target_that_dies_before_it_reports_is_stopped_where_it_was_started() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake {
            allocate_from: Some(41),
            ..Fake::default()
        };
        switch.start(&mut roster);
        roster.scripted = vec![exit_event("n0", 41)];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Abandoned);
        assert!(
            roster
                .stops
                .contains(&("n0".to_owned(), "engine:0".to_owned(), Some(41)))
        );
        assert!(
            roster
                .stops
                .contains(&("n1".to_owned(), "engine:0".to_owned(), Some(42))),
            "the other node's child, at the generation its spawn allocated"
        );
        assert!(
            !roster
                .stops
                .iter()
                .any(|(_, _, generation)| *generation == Some(6)),
            "no stop names the epoch's spawn generation, which no process had"
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
        assert!(
            !roster
                .commands
                .iter()
                .any(|(_, _, kind)| matches!(kind, process::command::Kind::Promote(_))),
            "the stopped target was never promoted"
        );
    }

    /// Another executable's build is unknown until its first child connects:
    /// that child fixes the epoch's build, and a second child of another
    /// build is stopped.
    #[test]
    fn an_unknown_build_is_fixed_by_the_first_child() {
        // Another executable's build is unknown to the daemon: the child's
        // first `Hello` is what fixes it.
        let (epochs, mut switch) = switch_with(Start {
            build: None,
            ..start()
        });
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

    /// The serving epoch's standby is promoted during the preparing, at the
    /// generation right after the serving one; the target's own generation is
    /// above it, so the promoted child never enters the target's sets.
    #[test]
    fn a_promoted_standby_of_the_previous_epoch_is_not_the_target() {
        let epochs = Arc::new(Epochs::new());
        let mut switch = Switch::new(
            Arc::clone(&epochs),
            Update::new(7, Deployment::of(ServeConfig::default())),
            12,
            record(),
            11,
            5,
            7,
            "engine:0".to_owned(),
            nodes(),
            start(),
            "unix:/run/requests.sock".to_owned(),
            Vec::new(),
        );
        let mut roster = Fake::default();
        switch.start(&mut roster);
        assert_eq!(roster.spawns[0].generation, 7);

        // The previous epoch's parked standby (generation 6) is promoted while
        // the target prepares.
        switch.promoted(&mut roster, "n0", 6);
        switch.parked(&mut roster, "n0", 6);
        assert_eq!(switch.phase(), Phase::Preparing);
        assert_eq!(switch.generation(), None, "6 is the previous epoch's");

        switch.parked(&mut roster, "n0", 7);
        switch.parked(&mut roster, "n1", 7);
        assert_eq!(switch.phase(), Phase::Promoting);
        assert_eq!(switch.generation(), Some(7));
        assert!(
            roster
                .commands
                .iter()
                .all(|(_, generation, _)| *generation == 7)
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
        assert_eq!(
            roster.commands_of(6, |kind| matches!(kind, process::command::Kind::Revert(_))),
            2,
            "the target reverts while the replacement comes up"
        );

        roster.scripted = vec![
            report_event(
                "n0",
                6,
                report(process::report::Kind::Reverted(process::Reverted {
                    reason: "an illegal address".to_owned(),
                })),
            ),
            report_event(
                "n1",
                6,
                report(process::report::Kind::Reverted(process::Reverted {
                    reason: "an illegal address".to_owned(),
                })),
            ),
        ];
        switch.advance(&mut roster);

        switch.serving(&mut roster, "n0", 7);
        assert_eq!(
            switch.phase(),
            Phase::Reverting,
            "every host must report serving"
        );
        switch.serving(&mut roster, "n1", 8);
        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(
            roster.serves,
            vec![(11, "unix:/run/requests.sock".to_owned())]
        );
        assert!(
            roster
                .resumes
                .iter()
                .all(|(_, generation, _)| *generation == 5),
            "the parked epoch returns at its own launch generation, 5"
        );
    }

    /// The return is driven by the reports the adapter forwards: each host's
    /// returning process says `Promoted`, and the frontend is served the
    /// previous epoch once every host has. The target's own generation is not
    /// the return, even while its trial is coming to an end.
    #[test]
    fn a_restore_counts_the_returning_processs_promoted_report() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        serve_target(&mut roster, &mut switch);
        roster.commands.clear();
        roster.serves.clear();

        roster.scripted = vec![exit_event("n0", 6)];
        switch.advance(&mut roster);
        assert_eq!(switch.phase(), Phase::Reverting);
        assert!(roster.serves.is_empty());

        roster.scripted = vec![report_event(
            "n1",
            6,
            report(process::report::Kind::Promoted(process::Promoted {
                generation: 6,
            })),
        )];
        switch.advance(&mut roster);
        assert_eq!(switch.phase(), Phase::Reverting);
        assert!(
            roster.serves.is_empty(),
            "the target's promotion is not the previous epoch's return"
        );

        // The epoch layer started a fresh generation of the previous epoch
        // above the target's spawn generation on each host, and promoted it.
        roster.scripted = vec![
            report_event(
                "n0",
                7,
                report(process::report::Kind::Promoted(process::Promoted {
                    generation: 7,
                })),
            ),
            report_event(
                "n1",
                7,
                report(process::report::Kind::Promoted(process::Promoted {
                    generation: 7,
                })),
            ),
        ];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(
            roster.serves,
            vec![(11, "unix:/run/requests.sock".to_owned())]
        );
        assert!(
            roster
                .stops
                .contains(&("n0".to_owned(), "engine:0".to_owned(), Some(6))),
            "the target generation is stopped once the return serves"
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

        assert_eq!(switch.phase(), Phase::Reverting);
        assert_eq!(
            roster.resumes.len(),
            2,
            "a fresh generation of the old epoch starts on each node"
        );
        assert!(
            roster
                .resumes
                .iter()
                .all(|(_, generation, _)| *generation == 5)
        );
        assert!(
            roster.serves.is_empty(),
            "the frontend is served only once every node reports serving"
        );

        switch.serving(&mut roster, "n0", 7);
        assert!(roster.serves.is_empty(), "one node is not every node");
        switch.serving(&mut roster, "n1", 7);

        assert_eq!(switch.phase(), Phase::Abandoned);
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

    /// A target that dies while its sibling still serves ends the sibling's
    /// trial too: the survivor is reverted, and only the dead host is left
    /// uncommanded.
    #[test]
    fn a_dead_target_reverts_the_sibling_that_still_runs() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        serve_target(&mut roster, &mut switch);
        roster.commands.clear();
        roster.serves.clear();

        roster.scripted = vec![exit_event("n0", 6)];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Reverting);
        assert_eq!(
            roster.commands_of(6, |kind| matches!(kind, process::command::Kind::Revert(_))),
            1,
            "the dead target is not commanded; its sibling is"
        );
        assert!(roster.commands.iter().all(|(host, _, _)| host == "n1"));
        assert!(
            roster.serves.is_empty(),
            "the frontend waits until a previous epoch serves"
        );

        switch.serving(&mut roster, "n0", 7);
        switch.serving(&mut roster, "n1", 7);
        assert_eq!(switch.phase(), Phase::Abandoned);
        assert_eq!(
            roster.serves,
            vec![(11, "unix:/run/requests.sock".to_owned())]
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
        // The precopy spawned the target and claimed its world on each node;
        // only what the revert does is asserted below.
        roster.spawns.clear();
        roster.claims.clear();

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
        assert_eq!(
            roster
                .commands
                .iter()
                .filter(|(_, _, kind)| matches!(kind, process::command::Kind::Revert(_)))
                .count(),
            2,
            "a revert per node"
        );
        assert!(roster.commands.iter().all(|(_, generation, kind)| {
            *generation == 6
                && matches!(kind, process::command::Kind::Revert(revert)
                    if revert.reason == "an illegal address")
        }));
        assert!(
            roster.serves.is_empty(),
            "the frontend waits for the revert"
        );

        roster.scripted = vec![
            report_event(
                "n0",
                6,
                report(process::report::Kind::Reverted(process::Reverted {
                    reason: "an illegal address".to_owned(),
                })),
            ),
            report_event(
                "n1",
                6,
                report(process::report::Kind::Reverted(process::Reverted {
                    reason: "an illegal address".to_owned(),
                })),
            ),
        ];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Reverting);
        assert!(
            roster.serves.is_empty(),
            "the previous epoch returns to serving first"
        );
        assert_eq!(roster.resumes.len(), 2);
        switch.serving(&mut roster, "n0", 7);
        switch.serving(&mut roster, "n1", 7);

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
        assert!(
            roster.spawns.is_empty(),
            "the switch starts nothing: the epoch layer returns the parked epoch"
        );
        assert!(
            roster
                .resumes
                .iter()
                .all(|(_, generation, reason)| *generation == 5 && reason == "an illegal address"),
            "the parked epoch is served again at the launch generation it had"
        );
        assert!(
            roster
                .claims
                .iter()
                .any(|(_, world, _, retire)| *world == 11 && retire == &[12]),
            "the previous world is claimed back, retiring the target's"
        );
        assert!(
            roster
                .stops
                .contains(&("n0".to_owned(), "engine:0".to_owned(), Some(6))),
            "the target generation is stopped"
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

    /// A child of another build that connects after the target serves is a
    /// replacement's, not a precondition of the promotion: it must not drop a
    /// serving epoch.
    #[test]
    fn a_build_mismatch_after_promotion_does_not_drop_a_serving_epoch() {
        let (_epochs, mut switch) = setup();
        let mut roster = Fake::default();
        serve_target(&mut roster, &mut switch);
        roster.stops.clear();
        roster.scripted = vec![hello("n0", 7, "9.9.9+other")];
        switch.advance(&mut roster);
        assert_eq!(switch.phase(), Phase::Probation);
        assert!(roster.stops.is_empty());
        assert!(switch.failure().is_none());
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

    /// The previous epoch's promoted standby is that epoch's serving process:
    /// its exit during the preparing is the serving epoch's death, so the
    /// standby promotion takes over and the target is dropped.
    #[test]
    fn a_promoted_standbys_exit_is_the_previous_epochs() {
        let epochs = Arc::new(Epochs::new());
        let mut switch = Switch::new(
            Arc::clone(&epochs),
            Update::new(7, Deployment::of(ServeConfig::default())),
            12,
            record(),
            11,
            5,
            7,
            "engine:0".to_owned(),
            nodes(),
            start(),
            "unix:/run/requests.sock".to_owned(),
            Vec::new(),
        );
        let mut roster = Fake::default();
        switch.start(&mut roster);
        // The serving epoch's standby was promoted at 6, then died.
        roster.scripted = vec![exit_event("n0", 6)];
        switch.advance(&mut roster);

        assert_eq!(switch.phase(), Phase::Abandoned);
        assert!(switch.promotion_due());
        assert_eq!(switch.discarded(), &[12]);
        assert!(roster.serves.is_empty());
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
