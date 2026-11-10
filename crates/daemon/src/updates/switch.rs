//! One epoch switch: the phases that move a deployment's main processes from
//! the serving epoch to a target one.
//!
//! The switch is driven by events the node roster delivers and by the
//! operator's commit or rollback; every phase ends on an acknowledgement a
//! main process makes, never on an assumption about its progress. The
//! participants and their wire are `proto::process`: a node's daemon starts
//! and stops children (`Spawn`, `Stop`), claims an epoch's storage (`Claim`),
//! commands a main process of one generation (`Command`), and the frontend is
//! commanded down its own connection (`Serve`). The roster behind those verbs
//! is the caller's; this module owns the order.
//!
//! The target takes the standby path, in front's order (§7.4):
//!
//! 1. **Preparing.** Every node starts the target executable at the next
//!    generation of the serving group and rank, then claims the target
//!    epoch's storage for the node's scopes. The child parks on its storage
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
//! 4. **Probation**, recorded once the drain completed. The previous epoch is
//!    kept as `update_config.previous_epoch` states: drained and alive, or its
//!    processes stopped and its weights kept ([`Roster::keep_weights`]). A
//!    failure the target reports, a target process that exits, or a threshold of
//!    `update_config.rollback` ([`crate::updates::trial`]) reverts.
//! 5. **Commit** records the target as serving once it acknowledges, stops
//!    the old generation and purges and retires the previous epoch; **revert**
//!    drains the target (a target that failed is stopped instead), returns the
//!    previous epoch to serving once every target ended its requests, serves
//!    the frontend only once every host serves, stops the target generation,
//!    records it purged and marks the target configuration unavailable, so
//!    applying it again needs `force`. A previous epoch whose processes are gone
//!    is started again on the weights it kept, by whoever owns its engine: the
//!    switch names it ([`Switch::relaunch_due`]) and is told it serves
//!    ([`Switch::serving`]).
//!
//! An epoch dropped before the switchover is kept in [`Switch::discarded`] and
//! passed in the next claim's `retire`: storage ends only through a claim,
//! and nothing else would ever retire a target that never served.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use proto::process::{self, Report, Spawn};

use crate::epoch::{EpochRecord, Epochs, Failure, FilledPrograms, Status, Update};
use crate::options::PreviousEpoch;

/// A node of the deployment the switch starts main processes on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Node {
    /// The node's host identity, as rank layouts name it.
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
    pub(crate) launch: proto::config::EngineArgs,
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

/// What a node did with a previous epoch's storage (`Roster::keep_weights`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct KeptWeights {
    /// The storages the node's server released.
    pub(crate) released: u64,
    /// The bytes of weights it moved to host storage.
    pub(crate) moved_bytes: u64,
    /// Why it kept the weights on its GPUs although host storage was asked.
    pub(crate) refused: Option<String>,
}

/// What the switch asks of the nodes: claim an epoch's storage, start and
/// stop main processes, command one generation's main process, command the
/// frontend, and take the events that arrive. The daemon's supervision
/// implements it ([`crate::supervision::roster::ProcessRoster`]); the trait is
/// the epoch layer's statement of what it needs of the nodes, so the switch's
/// order is written without the supervision's transport.
pub(crate) trait Roster {
    /// Claims `epoch`'s storage on `host` for `scopes`, retiring `retire`'s
    /// epochs, and holds the claim when it returns `Ok`: §7.4 requires it
    /// held before the target maps storage, so it is synchronous and has no
    /// matching report. `Err` says the node refused and holds nothing.
    fn claim(
        &mut self,
        host: &str,
        epoch: u64,
        scopes: &[process::Scope],
        retire: &[u64],
    ) -> Result<(), String>;
    /// Starts `spawn` and returns the child's own generation and process id.
    fn spawn(
        &mut self,
        spawn: Spawn,
        launch: &proto::config::EngineArgs,
    ) -> Result<Spawned, String>;
    /// Stops one launch generation of `group` on `host`; every generation
    /// when `generation` is `None`.
    fn stop(&mut self, host: &str, group: &str, generation: Option<u64>);
    /// Commands the main process of `generation` on `host`.
    fn command(&mut self, host: &str, generation: u64, command: process::Command);
    /// Makes the process of `generation` on `host` commandable: the epoch the switch leaves was
    /// started by the daemon's engine control, so this roster holds no row of it, and a `Drain` to
    /// a process it does not know would reach nothing.
    fn adopt(&mut self, host: &str, group: &str, rank: u32, generation: u64);
    /// Commands the frontend to serve `epoch` from `requests`.
    fn serve(&mut self, epoch: u64, requests: &str);
    /// Returns the previous epoch at `host` to serving.
    ///
    /// Its main processes were parked at the switchover, not ended (§7.4:
    /// the serving epoch is parked at the step boundary, and the old epoch's
    /// main processes exit at purge), so this is `Serve { epoch: previous }`
    /// to them, at the launch generation they always had — the model
    /// generation travels inside the command, not in `generation`. The switch
    /// learns the host serves through `serving`.
    fn resume(&mut self, host: &str, generation: u64, reason: &str);
    /// Retains `epoch`'s weights on `host` once its processes exited, and
    /// releases the rest of its exclusive storage; with `to_host` the weights
    /// it alone retains move to host storage. Answers what the node did.
    fn keep_weights(
        &mut self,
        host: &str,
        epoch: u64,
        to_host: bool,
    ) -> Result<KeptWeights, String>;
    /// Takes the events that arrived since the last call.
    fn events(&mut self) -> Vec<Event>;
}

/// A boxed roster commands exactly as the roster inside it, so the daemon can
/// hold one trait object for the adapter it constructs once.
impl<T: Roster + ?Sized> Roster for Box<T> {
    fn claim(
        &mut self,
        host: &str,
        epoch: u64,
        scopes: &[process::Scope],
        retire: &[u64],
    ) -> Result<(), String> {
        (**self).claim(host, epoch, scopes, retire)
    }

    fn spawn(
        &mut self,
        spawn: Spawn,
        launch: &proto::config::EngineArgs,
    ) -> Result<Spawned, String> {
        (**self).spawn(spawn, launch)
    }

    fn stop(&mut self, host: &str, group: &str, generation: Option<u64>) {
        (**self).stop(host, group, generation)
    }

    fn command(&mut self, host: &str, generation: u64, command: process::Command) {
        (**self).command(host, generation, command)
    }

    fn adopt(&mut self, host: &str, group: &str, rank: u32, generation: u64) {
        (**self).adopt(host, group, rank, generation)
    }

    fn serve(&mut self, epoch: u64, requests: &str) {
        (**self).serve(epoch, requests)
    }

    fn resume(&mut self, host: &str, generation: u64, reason: &str) {
        (**self).resume(host, generation, reason)
    }

    fn keep_weights(
        &mut self,
        host: &str,
        epoch: u64,
        to_host: bool,
    ) -> Result<KeptWeights, String> {
        (**self).keep_weights(host, epoch, to_host)
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
    /// The hosts whose targets the daemon has yet to give their programs.
    awaiting_programs: BTreeSet<String>,
    /// The previous epoch's number, which `Serve` and `Drain` name.
    previous: u64,
    /// The epoch the previous epoch executes, whose storage a claim retires
    /// or keeps: the epoch a deployment starts with is number 0, its epoch the
    /// first engine's.
    previous_epoch: u64,
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
    /// Epochs whose storage must be retired by the next claim.
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
    /// The operator committed before the trial began (`Switch::commit`).
    commit_asked: bool,
    /// The operator reverted before the trial began (`Switch::revert`).
    revert_asked: Option<Failure>,
    pause: Option<Duration>,
    /// What the previous epoch keeps through the trial
    /// (`update_config.previous_epoch`).
    keeps: PreviousEpoch,
    /// What it keeps as the trial goes: `keeps`, or `weights` once a node
    /// refused to move the weights to host storage.
    kept: PreviousEpoch,
    /// Each node's reason for keeping the weights on its GPUs.
    host_refused: Vec<String>,
    /// The bytes of weights the nodes moved to host storage.
    moved_bytes: u64,
    /// Whether the trial stopped the previous epoch's processes: a revert then
    /// starts them again, exited or not.
    previous_stopped: bool,
    /// Hosts whose stopped previous-epoch main process exited.
    previous_exited: BTreeSet<String>,
    /// Hosts that have to start the previous epoch again before it can serve.
    relaunch_due: BTreeSet<String>,
    /// Whether a serving main process of the previous epoch exited during the
    /// switch, which hands the failure domain to the standby promotion
    /// (§2.5) instead of a revert.
    promotion_due: bool,
}

impl Switch {
    /// A switch of epoch `previous` of epoch `previous_epoch` (launch generation
    /// `previous_generation`) to epoch `target`, whose record is `record`, with the target's
    /// children started at `spawn_generation` — above every live generation
    /// of the group and rank, the serving epoch's standby included —
    /// retiring `discarded`, the epochs an earlier switch dropped.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        epochs: Arc<Epochs>,
        update: Update,
        target: u64,
        record: EpochRecord,
        (previous, previous_epoch): (u64, u64),
        previous_generation: u64,
        spawn_generation: u64,
        group: String,
        nodes: Vec<Node>,
        start: Start,
        requests: String,
        discarded: Vec<u64>,
        keeps: PreviousEpoch,
    ) -> Self {
        let mut record = record;
        record.status = Status::Precopy;
        epochs.record(target, record.clone());
        Self {
            epochs,
            update,
            target,
            previous,
            previous_epoch,
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
            awaiting_programs: BTreeSet::new(),
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
            commit_asked: false,
            revert_asked: None,
            pause: None,
            keeps,
            kept: keeps,
            host_refused: Vec::new(),
            moved_bytes: 0,
            previous_stopped: false,
            previous_exited: BTreeSet::new(),
            relaunch_due: BTreeSet::new(),
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

    /// The Program Servers selected `programs` for the target: an update names backends, and a
    /// process that joins the epoch later is handed what its record names, where a revision of
    /// zero names no package.
    pub(crate) fn pin(&mut self, programs: &FilledPrograms) {
        self.record.kernels = programs.kernels.clone();
        self.record.policy = programs.policy.clone();
        self.epochs.record(self.target, self.record.clone());
    }

    /// Epochs this switch dropped whose storage a later claim must retire:
    /// the update layer keeps them for the next switch's `start`.
    pub(crate) fn discarded(&self) -> &[u64] {
        &self.retire
    }

    /// The switchover's pause, once it ended.
    pub(crate) fn pause(&self) -> Option<Duration> {
        self.pause
    }

    /// When the switchover began: what the trial's baseline ends at.
    pub(crate) fn promotion(&self) -> Option<Instant> {
        self.switchover_started
    }

    /// The previous epoch's number and launch generation.
    pub(crate) fn previous(&self) -> (u64, u64) {
        (self.previous, self.previous_generation)
    }

    /// What the previous epoch is kept as through the trial: what the update
    /// asked for, or `weights` where a node could not move them to host storage.
    pub(crate) fn previous_kept(&self) -> PreviousEpoch {
        self.kept
    }

    /// Why the previous epoch's weights stayed on the GPUs although host
    /// storage was asked: each node's refusal.
    pub(crate) fn host_refused(&self) -> &[String] {
        &self.host_refused
    }

    /// The bytes of weights the nodes moved to host storage.
    pub(crate) fn moved_bytes(&self) -> u64 {
        self.moved_bytes
    }

    /// The hosts that have to start the previous epoch again: its processes
    /// exited for the trial, the target stopped, and it serves only once its
    /// owner started it on the weights it kept and said so (`Self::serving`).
    pub(crate) fn relaunch_due(&self) -> &BTreeSet<String> {
        &self.relaunch_due
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
    /// claims the target epoch's storage before the child maps it.
    pub(crate) fn start(&mut self, roster: &mut impl Roster) {
        for node in &self.nodes {
            roster.adopt(&node.host, &self.group, node.rank, self.previous_generation);
        }
        for node in &self.nodes {
            let spawn = Spawn {
                host: node.host.clone(),
                group: self.group.clone(),
                rank: node.rank,
                generation: self.spawn_generation,
                executable: self.start.executable.clone(),
                // The adapter states the launch file for this process and
                // its arguments: they are its to state.
                arguments: Vec::new(),
                environment: self.start.environment.clone(),
                inherit: Vec::new(),
                launch: Vec::new(),
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
            // node is told the same epochs.
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

    /// Whether every node's target parked, so only what the daemon still gives it holds the
    /// switchover back.
    pub(crate) fn parked_everywhere(&self) -> bool {
        self.nodes
            .iter()
            .all(|node| self.parked.contains(&node.host))
    }

    /// States that `hosts`' targets are given their programs by the daemon, so none of them is
    /// promoted before it holds them: a target parks before any `Precopy` names a generation.
    pub(crate) fn await_programs(&mut self, hosts: impl IntoIterator<Item = String>) {
        self.awaiting_programs.extend(hosts);
    }

    /// `host`'s target took its programs.
    pub(crate) fn handed(&mut self, roster: &mut impl Roster, host: &str) {
        self.awaiting_programs.remove(host);
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
        self.relaunch_due.remove(host);
        let hosts: BTreeSet<String> = self.nodes.iter().map(|node| node.host.clone()).collect();
        if hosts.is_subset(&self.serving) && self.pending_restore.take().is_some() {
            self.finish_restore(roster);
        }
    }

    /// The operator commits: the target serves from now on and the previous
    /// epoch is dropped and retired.
    ///
    /// The operator may ask before the trial has begun: the answer to an apply
    /// comes once the target serves, while the drain that starts the trial is
    /// still in flight. The commit is kept and applied when the trial starts, so
    /// an answered commit is never one that did nothing.
    pub(crate) fn commit(&mut self, roster: &mut impl Roster) {
        match self.phase {
            Phase::Preparing | Phase::Promoting | Phase::Serving => {
                self.commit_asked = true;
                return;
            }
            Phase::Probation => {}
            Phase::Committing | Phase::Reverting | Phase::Purged | Phase::Abandoned => return,
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
        self.revert_with(
            roster,
            Failure {
                kind: "reverted".to_owned(),
                reason,
                device: None,
            },
        );
    }

    /// Reverts with `failure` as the update's first one: a threshold of the
    /// trial, or the operator.
    pub(crate) fn revert_with(&mut self, roster: &mut impl Roster, failure: Failure) {
        match self.phase {
            Phase::Probation => self.reverting(roster, failure),
            // The switchover happened and the previous epoch is draining: its
            // requests move to the target, and only a drained process can
            // serve again, so the revert waits for the trial it ends. A target
            // that fails meanwhile returns at once (`Self::restore`).
            Phase::Serving => self.revert_asked = Some(failure),
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
    /// stop, its epoch is retired by the next claim, and the previous epoch
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

    /// Reverts a target that serves on probation: the target's requests end
    /// migratable at their step boundary, as the previous epoch's did at the
    /// switchover, and the first reason is what marks the configuration
    /// unavailable (§7.3 step 4). A target that failed is not commanded again
    /// (§7.4): it is stopped, and its requests are lost to it and migrate. The
    /// previous epoch serves again once every target drained or exited.
    fn reverting(&mut self, roster: &mut impl Roster, failure: Failure) {
        self.reverting_without(roster, failure, None);
    }

    /// [`Self::reverting`], with `failed` the host whose target reported the
    /// failure.
    fn reverting_without(
        &mut self,
        roster: &mut impl Roster,
        failure: Failure,
        failed: Option<&str>,
    ) {
        self.update.fail(failure.clone());
        self.failure = Some(failure);
        self.phase = Phase::Reverting;
        let generation = self.target_generation();
        let nodes = self.nodes.clone();
        for node in &nodes {
            if failed == Some(node.host.as_str()) {
                self.target_reverted.insert(node.host.clone());
                roster.stop(&node.host, &self.group, Some(generation));
                continue;
            }
            if self.target_dead.contains(&node.host) {
                self.target_reverted.insert(node.host.clone());
                continue;
            }
            roster.command(
                &node.host,
                generation,
                command(process::command::Kind::Drain(process::Drain {
                    epoch: self.target,
                })),
            );
        }
    }

    /// Starts the return to the previous epoch after the target died or
    /// faulted before the trial began: every target that still runs drains,
    /// so no host keeps serving it, and every host returns the previous epoch
    /// to serving at once (§7.4). The frontend is served it only once every
    /// host says it serves.
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
                command(process::command::Kind::Drain(process::Drain {
                    epoch: self.target,
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
            let _ = roster.claim(&node.host, self.previous_epoch, &self.scopes(node), &retire);
        }
        self.record.status = Status::Purged;
        self.epochs.record(self.target, self.record.clone());
        self.update.revert();
        self.phase = Phase::Abandoned;
    }

    /// Every host returns its previous epoch to serving: `Serve { epoch:
    /// previous }` to the processes the switchover parked. Where the trial
    /// stopped them the target stops first, since the previous epoch's start
    /// takes the storage it held, and its owner is asked to start it again.
    fn resume(&mut self, roster: &mut impl Roster, reason: &str) {
        let generation = self.target_generation();
        let nodes = self.nodes.clone();
        for node in &nodes {
            if self.previous_stopped {
                roster.stop(&node.host, &self.group, Some(generation));
                self.relaunch_due.insert(node.host.clone());
            } else {
                roster.resume(&node.host, self.previous_generation, reason);
            }
        }
        if !self.relaunch_due.is_empty() {
            // The record is what a process started now reads its programs
            // from: the target's stands no more.
            self.record.status = Status::Purged;
            self.epochs.record(self.target, self.record.clone());
        }
    }

    /// Every target drained or ended: the previous epoch returns to serving
    /// on every host, and the frontend is served it when every host reports
    /// serving.
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
                self.record.config.executable = build;
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
                // The trial stopped it: `update_config.previous_epoch` keeps its
                // weights alone, and its exit is what lets the node release the
                // rest of its storage. A commit retires that storage anyway, and
                // a revert under way starts the epoch on it.
                Phase::Probation if self.previous_stopped => {
                    self.previous_ended(roster, &host);
                }
                Phase::Committing | Phase::Reverting if self.previous_stopped => {
                    self.previous_exited.insert(host);
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

    /// `host`'s previous-epoch process ended for the trial: the node keeps
    /// its weights alone, in host storage when the update asked and the node
    /// can. A node that refuses, or fails to keep them, keeps them on its GPUs:
    /// nothing else of the trial depends on it, and the status says why.
    ///
    /// The process's owner reports this: a process the daemon's engine control
    /// started ends with its connection, which no roster event states.
    pub(crate) fn previous_ended(&mut self, roster: &mut impl Roster, host: &str) {
        if self.phase != Phase::Probation
            || !self.previous_stopped
            || !self.previous_exited.insert(host.to_owned())
        {
            return;
        }
        let to_host = self.keeps == PreviousEpoch::Host;
        match roster.keep_weights(host, self.previous_epoch, to_host) {
            Ok(kept) => {
                self.moved_bytes += kept.moved_bytes;
                if let Some(refused) = kept.refused {
                    self.kept = PreviousEpoch::Weights;
                    self.host_refused.push(format!("{host}: {refused}"));
                }
            }
            Err(reason) => {
                if to_host {
                    self.kept = PreviousEpoch::Weights;
                }
                self.host_refused.push(format!("{host}: {reason}"));
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
                // The programs the promotion reports are the ones the process
                // holds, and the epoch serves the ones its children prepared
                // (`EpochRecord::generation`). A standby whose precopy the
                // epoch reverted holds an older generation, and moving the
                // frontend onto it would serve programs the epoch no longer
                // runs — refused here, with the frontend left where it is and
                // the target abandoned rather than reverted. An unstated `0`
                // is a process that holds no programs, which is the behaviour
                // a minor-4 engine has.
                let held = promoted.program_generation;
                if let (true, Some(epoch)) = (held != 0, self.record.generation)
                    && held != epoch
                {
                    self.failed(
                        roster,
                        Failure {
                            kind: "stale_programs".to_owned(),
                            reason: format!(
                                "the promoted process on {host} holds program generation \
                                     {held} and this epoch's is {epoch}: the frontend is not \
                                     moved onto programs the epoch does not serve"
                            ),
                            device: None,
                        },
                    );
                    return;
                }
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
                    self.reverting_without(roster, failure, Some(&host));
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
            (Phase::Reverting, Some(Kind::Drained(drained)))
                if self.pending_restore.is_none()
                    && drained.epoch == self.target
                    && self.is_target(generation) =>
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
            Phase::Preparing
                if hosts.is_subset(&self.parked) && self.awaiting_programs.is_empty() =>
            {
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
                if let Some(failure) = self.revert_asked.take() {
                    self.reverting(roster, failure);
                    return;
                }
                // What the trial keeps of the previous epoch: a process that
                // serves nothing holds the storage the target could use, and
                // `previous_epoch` says whether a revert is worth that storage.
                if self.keeps != PreviousEpoch::Parked {
                    self.previous_stopped = true;
                    for node in &self.nodes {
                        roster.stop(&node.host, &self.group, Some(self.previous_generation));
                    }
                }
                if std::mem::take(&mut self.commit_asked) {
                    self.commit(roster);
                }
            }
            Phase::Committing if hosts.is_subset(&self.committed) => {
                self.record.status = Status::Serving;
                self.epochs.record(self.target, self.record.clone());
                let retire = vec![self.previous_epoch];
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
                    // The previous epoch's storage ends on the node that
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
