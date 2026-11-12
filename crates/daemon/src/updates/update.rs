//! One update: the phases that move a deployment to another set of modules as
//! a trial.
//!
//! An update has one phase machine whatever it moves:
//!
//! | Phase | Ends when |
//! | --- | --- |
//! | `Preparing` | Every engine acknowledged it is ready, or one failed: the update is `Abandoned` and nothing changed |
//! | `Switching` | Every engine acknowledged the switch |
//! | `Trial` | A commit or a rollback: the submitter's or the operator's decision, a fault, a threshold or the deadline |
//! | `Committed`, `RolledBack` | — |
//!
//! Its two forms differ only in what `Preparing` and `Switching` do:
//!
//! | | Generation | Epoch |
//! | --- | --- | --- |
//! | Preparing | `Prepare` to each serving engine; `Prepared` acknowledges | Each node starts the target engine, which parks; `Parked` acknowledges |
//! | Switching | `Switch` to each serving engine; `Switched` acknowledges | `Drain` to each serving engine; its `Drained` hands the target its state in `Promote`; `Serving` acknowledges |
//! | Kept | The previous generation, inside the engine | The previous epoch: parked, its weights, or its weights in host storage |
//! | Rollback | `Rollback`; `RolledBack` acknowledges | `Drain` to each target; the previous epoch is promoted again, or started again where it was stopped |
//!
//! A rollback is the switch run back, so the phase is `Switching` again until
//! the kept side serves, and `RolledBack` after. A commit records its decision
//! (the phase and the epoch's record) before it releases anything, so an
//! engine lost while the kept side is released is replaced by one of the
//! committed modules.
//!
//! Every phase ends on acknowledgements the engines make, never on an
//! assumption about their progress. The epoch form's target admits nothing
//! until the serving engine stopped stepping: it is promoted with the state
//! `Drained` carried, so two epochs never write one cache.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use proto::process::{self, Report, Spawn, command, report};
use serde_json::{Value, json};

use crate::epoch::{Epoch, EpochRecord, Epochs, Failure, Status};
use crate::options::PreviousEpoch;
use crate::supervision::processes::{Event, Processes};

/// Where an update is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// The new is made ready beside the old; serving is untouched.
    Preparing,
    /// Service moves at a boundary: to the new, or back to the kept side.
    Switching,
    /// The new serves and the old is kept.
    Trial,
    /// The new serves and the kept side is released.
    Committed,
    /// The kept side serves again and the new is released.
    RolledBack,
    /// The new was never ready: nothing changed.
    Abandoned,
}

impl Phase {
    /// The phase as the status and the event log name it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Preparing => "preparing",
            Self::Switching => "switching",
            Self::Trial => "trial",
            Self::Committed => "committed",
            Self::RolledBack => "rolled_back",
            Self::Abandoned => "abandoned",
        }
    }

    /// Whether the update ended: nothing more moves it.
    pub(crate) fn ended(self) -> bool {
        matches!(self, Self::Committed | Self::RolledBack | Self::Abandoned)
    }
}

/// One serving engine the update moves, and what the update made of it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node {
    /// The host identity of the node it runs on, as rank layouts name it.
    host: String,
    /// The node's rank in the deployment.
    rank: u32,
    /// The launch generation of the engine that serves there when the update
    /// begins.
    serving: u64,
    /// The launch generation of the epoch form's target engine there.
    target: Option<u64>,
    /// The generation of modules the update's engine there numbered: what a
    /// serving engine prepared, or what a target holds.
    generation: Option<u64>,
    /// What the engine that drained there hands the one promoted after it.
    handoff: Option<process::Handoff>,
}

impl Node {
    /// The engine of launch generation `serving` on `host`, the node of rank
    /// `rank`.
    pub(crate) fn new(host: String, rank: u32, serving: u64) -> Self {
        Self {
            host,
            rank,
            serving,
            target: None,
            generation: None,
            handoff: None,
        }
    }
}

/// What the epoch form states of its target, beside the nodes.
pub(crate) struct EpochSwitch {
    /// The target epoch's number, which its record is kept under.
    pub(crate) epoch: u64,
    /// The target epoch's record, whose status the update advances.
    pub(crate) record: EpochRecord,
    /// The launch file every target engine is started from: it states the
    /// epoch's module set and the engine it succeeds, so the engine parks by
    /// itself once it is ready.
    pub(crate) launch: proto::config::EngineArgs,
    /// Variables set over the daemon's environment for the target engines.
    pub(crate) environment: std::collections::HashMap<String, String>,
    /// The launch generation the target engines are started at: above every
    /// live launch generation of the group, a standby's included.
    pub(crate) spawn_generation: u64,
    /// The request socket requests are submitted to after the switch.
    pub(crate) requests: String,
    /// What the previous epoch keeps through the trial
    /// (`update_config.previous_epoch`).
    pub(crate) keeps: PreviousEpoch,
    /// Whether the requests that migrate compute their state again
    /// (`in_flight = recompute`).
    pub(crate) recompute: bool,
    /// Epochs an earlier update dropped whose storage the next claim retires.
    pub(crate) retire: Vec<u64>,
}

/// What `Preparing` and `Switching` do.
enum Form {
    /// Kernel or policy packages: the serving engines prepare the next
    /// generation and switch to it at a batch boundary.
    Generation,
    /// The engine package, the build configuration or the checkpoint: target
    /// engines start beside the serving ones and service moves to them.
    Epoch(Box<EpochTrial>),
}

/// The epoch form's state as the update goes. Its sets hold indices into the
/// update's nodes.
struct EpochTrial {
    switch: EpochSwitch,
    /// The build the target engines report, fixed by the first one's `Hello`.
    build: Option<String>,
    /// What the previous epoch is kept as: `keeps`, or `weights` once a node
    /// could not move them to host storage.
    kept: PreviousEpoch,
    /// Each node's reason for keeping the weights on its GPUs.
    host_refused: Vec<String>,
    /// The bytes of weights the nodes moved to host storage.
    moved_bytes: u64,
    /// Nodes whose previous-epoch engine drained and is parked: it can be
    /// promoted again.
    drained: BTreeSet<usize>,
    /// Whether the targets were sent their promotion.
    promoted: bool,
    /// Whether the trial stopped the previous epoch's engines: a rollback
    /// then starts them again.
    previous_stopped: bool,
    /// Nodes whose previous-epoch engine exited.
    previous_exited: BTreeSet<usize>,
    /// Nodes that were asked to keep the previous epoch's weights.
    weights_kept: BTreeSet<usize>,
    /// Nodes whose target engine serves nothing more: it drained, was
    /// stopped or exited.
    target_left: BTreeSet<usize>,
    /// Whether a rollback returned the previous epoch to serving: every node
    /// is told once.
    returning: bool,
    /// Nodes that have to start the previous epoch again before it serves.
    relaunch_due: BTreeSet<usize>,
}

/// One update of one deployment.
pub(crate) struct Update {
    /// The update's number: the admin operation that applied it.
    number: u64,
    /// What the update moves the deployment to.
    target: Epoch,
    phase: Phase,
    /// The phases entered since they were last taken, for the event log.
    entered: Vec<Phase>,
    form: Form,
    epochs: Arc<Epochs>,
    /// The epoch that serves when the update begins.
    serving: u64,
    /// The worker group every engine of the deployment is in.
    group: String,
    nodes: Vec<Node>,
    /// The nodes that acknowledged the step the phase waits for.
    acknowledged: BTreeSet<usize>,
    /// The hosts whose own step the daemon still runs before the switch: the
    /// warming of the host KV tier.
    warming: BTreeSet<String>,
    /// Whether `Switching` runs back to the kept side.
    back: bool,
    /// Why the update did not stand: its first failure.
    failure: Option<Failure>,
    /// A commit asked before the trial began.
    commit_asked: bool,
    /// A rollback asked while the switch ran.
    rollback_asked: Option<Failure>,
    /// When a trial not committed rolls back by itself.
    deadline: Option<Instant>,
    /// When the switch began, and how long service paused for it.
    switch_began: Option<Instant>,
    pause: Option<Duration>,
    /// When the trial began; `None` before the new served.
    trial_began: Option<Instant>,
    /// The operators the engines could not compare while they prepared.
    uncompared: Vec<String>,
    /// Whether a serving engine was lost during the update, which hands its
    /// failure domain to the standby promotion or a relaunch.
    promotion_due: bool,
}

impl Update {
    /// A generation update: the serving engines `nodes` prepare `target`'s
    /// kernel and policy modules and switch to them.
    pub(crate) fn generation(
        number: u64,
        target: Epoch,
        epochs: Arc<Epochs>,
        serving: u64,
        group: String,
        nodes: Vec<Node>,
        deadline: Option<Instant>,
    ) -> Self {
        Self::new(
            number,
            target,
            Form::Generation,
            epochs,
            serving,
            group,
            nodes,
            deadline,
        )
    }

    /// An epoch update: `switch`'s target engines start beside the serving
    /// engines `nodes`, and service moves to them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn epoch(
        number: u64,
        target: Epoch,
        epochs: Arc<Epochs>,
        serving: u64,
        group: String,
        nodes: Vec<Node>,
        switch: EpochSwitch,
        deadline: Option<Instant>,
    ) -> Self {
        let kept = switch.keeps;
        let form = Form::Epoch(Box::new(EpochTrial {
            switch,
            build: None,
            kept,
            host_refused: Vec::new(),
            moved_bytes: 0,
            drained: BTreeSet::new(),
            promoted: false,
            previous_stopped: false,
            previous_exited: BTreeSet::new(),
            weights_kept: BTreeSet::new(),
            target_left: BTreeSet::new(),
            returning: false,
            relaunch_due: BTreeSet::new(),
        }));
        Self::new(
            number, target, form, epochs, serving, group, nodes, deadline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        number: u64,
        target: Epoch,
        form: Form,
        epochs: Arc<Epochs>,
        serving: u64,
        group: String,
        nodes: Vec<Node>,
        deadline: Option<Instant>,
    ) -> Self {
        Self {
            number,
            target,
            phase: Phase::Preparing,
            entered: vec![Phase::Preparing],
            form,
            epochs,
            serving,
            group,
            nodes,
            acknowledged: BTreeSet::new(),
            warming: BTreeSet::new(),
            back: false,
            failure: None,
            commit_asked: false,
            rollback_asked: None,
            deadline,
            switch_began: None,
            pause: None,
            trial_began: None,
            uncompared: Vec::new(),
            promotion_due: false,
        }
    }

    /// The update's number.
    pub(crate) fn number(&self) -> u64 {
        self.number
    }

    /// Where the update is.
    pub(crate) fn phase(&self) -> Phase {
        self.phase
    }

    /// The phases entered since the last call, oldest first.
    pub(crate) fn take_entered(&mut self) -> Vec<Phase> {
        std::mem::take(&mut self.entered)
    }

    /// What the update moves the deployment to.
    pub(crate) fn target(&self) -> &Epoch {
        &self.target
    }

    /// Why the update did not stand, when it did not.
    pub(crate) fn failure(&self) -> Option<&Failure> {
        self.failure.as_ref()
    }

    /// The form's name, as the status and the event log state it.
    pub(crate) fn change(&self) -> &'static str {
        match self.form {
            Form::Generation => "generation",
            Form::Epoch(_) => "epoch",
        }
    }

    /// The epoch the update's side executes in: the target epoch of the epoch
    /// form, the serving epoch of the generation form.
    pub(crate) fn target_epoch(&self) -> u64 {
        match &self.form {
            Form::Generation => self.serving,
            Form::Epoch(trial) => trial.switch.epoch,
        }
    }

    /// When the switch began: what the trial's baseline ends at.
    pub(crate) fn switch_began(&self) -> Option<Instant> {
        self.switch_began
    }

    /// Whether every engine acknowledged `Preparing`, so only the daemon's
    /// own warming holds the switch back.
    pub(crate) fn prepared(&self) -> bool {
        self.phase == Phase::Preparing && self.all_acknowledged()
    }

    /// The hosts that have to start the previous epoch again for a rollback:
    /// its engines were stopped for the trial, the target is gone, and it
    /// serves once its owner started it on the weights it kept and said so
    /// ([`Self::relaunched`]).
    pub(crate) fn relaunch_due(&self) -> BTreeSet<String> {
        match &self.form {
            Form::Generation => BTreeSet::new(),
            Form::Epoch(trial) => trial
                .relaunch_due
                .iter()
                .map(|index| self.nodes[*index].host.clone())
                .collect(),
        }
    }

    /// Whether a serving engine was lost during the update: its failure
    /// domain is the standby promotion's or a relaunch's.
    pub(crate) fn promotion_due(&self) -> bool {
        self.promotion_due
    }

    /// Epochs this update dropped whose storage a later claim retires.
    pub(crate) fn retire(&self) -> &[u64] {
        match &self.form {
            Form::Generation => &[],
            Form::Epoch(trial) => &trial.switch.retire,
        }
    }

    /// The update as the operator reads it.
    pub(crate) fn report(&self) -> Value {
        let mut report = json!({
            "change": self.change(),
            "update": self.number,
            "phase": self.phase.name(),
            "epoch": self.target_epoch(),
            "rollback": self.back,
            "pause_ms": self.pause.map(|pause| pause.as_millis()),
            "trial_s": self.trial_began.map(|began| began.elapsed().as_secs_f64()),
            "deadline_s": self
                .deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now()).as_secs_f64()),
            "failure": self.failure.as_ref().map(|failure| json!({
                "kind": failure.kind,
                "reason": failure.reason,
                "device": failure.device,
            })),
            "uncompared": self.uncompared,
        });
        match &self.form {
            Form::Generation => {
                report["generation"] =
                    json!(self.nodes.iter().filter_map(|node| node.generation).max());
            }
            Form::Epoch(trial) => {
                report["generation"] = json!(trial.switch.record.generation);
                report["build"] = json!(trial.switch.record.config.executable);
                report["previous_epoch_kept"] = json!(trial.kept);
                report["host_refused"] = json!(trial.host_refused);
                report["moved_to_host_bytes"] = json!(trial.moved_bytes);
            }
        }
        report
    }

    /// Whether every node acknowledged the step the phase waits for.
    fn all_acknowledged(&self) -> bool {
        self.acknowledged.len() == self.nodes.len()
    }

    /// Enters `phase`: the step it waits for is acknowledged by nobody yet.
    fn enter(&mut self, phase: Phase) {
        self.phase = phase;
        self.acknowledged.clear();
        self.entered.push(phase);
    }

    /// The scopes a claim keeps on `node`: its own `(group, rank)`.
    fn scopes(&self, node: &Node) -> Vec<process::Scope> {
        vec![process::Scope {
            group: self.group.clone(),
            rank: node.rank,
        }]
    }

    /// The node whose engine of launch generation `generation` runs on
    /// `host`, and whether that engine is the epoch form's target.
    fn node_of(&self, host: &str, generation: u64) -> Option<(usize, bool)> {
        self.nodes.iter().enumerate().find_map(|(index, node)| {
            if node.host != host {
                None
            } else if node.target == Some(generation) {
                Some((index, true))
            } else if node.serving == generation {
                Some((index, false))
            } else {
                None
            }
        })
    }

    /// Starts `Preparing`.
    pub(crate) fn start(&mut self, processes: &mut Processes) {
        for node in &self.nodes {
            processes.adopt(&node.host, &self.group, node.rank, node.serving);
        }
        match &self.form {
            Form::Generation => self.prepare_generation(processes),
            Form::Epoch(_) => self.prepare_epoch(processes),
        }
    }

    /// `Prepare` to every serving engine, with the target's kernel and policy
    /// modules.
    fn prepare_generation(&mut self, processes: &mut Processes) {
        let modules = &self.target.modules;
        let prepare = command(command::Kind::Prepare(process::Prepare {
            kernels: modules
                .kernels
                .iter()
                .map(|(slot, package)| crate::epoch::kernel_to_wire(slot, package))
                .collect(),
            policy: Some(crate::epoch::package_to_wire(&modules.policy)),
        }));
        for node in &self.nodes {
            processes.command(&node.host, node.serving, prepare.clone());
        }
    }

    /// Every node starts the target engine, then claims the target epoch's
    /// storage before the engine maps it.
    fn prepare_epoch(&mut self, processes: &mut Processes) {
        let Form::Epoch(trial) = &mut self.form else {
            return;
        };
        trial.switch.record.status = Status::Preparing;
        self.epochs
            .record(trial.switch.epoch, trial.switch.record.clone());

        let generation = trial.switch.spawn_generation;
        let epoch = trial.switch.epoch;
        let retire = trial.switch.retire.clone();

        for index in 0..self.nodes.len() {
            let spawn = Spawn {
                host: self.nodes[index].host.clone(),
                group: self.group.clone(),
                rank: self.nodes[index].rank,
                generation,
                // The node's daemon resolves the engine package the launch
                // names and writes the launch file for this process.
                executable: String::new(),
                arguments: Vec::new(),
                environment: trial.switch.environment.clone(),
                inherit: Vec::new(),
                launch: Vec::new(),
            };
            match processes.spawn(spawn, &trial.switch.launch) {
                Ok(_) => self.nodes[index].target = Some(generation),
                Err(reason) => return self.abandon(processes, failure("spawn_failed", reason)),
            }
        }
        for index in 0..self.nodes.len() {
            // A claim retires storage on the node it is made on, so every
            // node is told the same epochs.
            let scopes = self.scopes(&self.nodes[index]);
            let host = self.nodes[index].host.clone();
            if let Err(reason) = processes.claim(&host, epoch, &scopes, &retire) {
                return self.abandon(processes, failure("claim_failed", reason));
            }
        }
        // Every node retired them: none is left for a later claim.
        if let Form::Epoch(trial) = &mut self.form {
            trial.switch.retire.clear();
        }
    }

    /// Holds the switch until the daemon finished its own step for `host`:
    /// the requests in flight warm the host KV tier once the target parked.
    pub(crate) fn await_warming(&mut self, host: &str) {
        self.warming.insert(host.to_owned());
    }

    /// The daemon's own step for `host` ended.
    pub(crate) fn warmed(&mut self, processes: &mut Processes, host: &str) {
        self.warming.remove(host);
        self.progress(processes);
    }

    /// Takes the engines' events and advances the update as they and the
    /// phase allow.
    pub(crate) fn advance(&mut self, processes: &mut Processes) {
        for event in processes.events() {
            if self.phase.ended() {
                break;
            }
            match event {
                Event::Hello {
                    host,
                    generation,
                    build,
                } => self.hello(processes, &host, generation, build),
                Event::Report {
                    host,
                    generation,
                    report,
                } => self.report_arrived(processes, &host, generation, report),
                Event::Exit {
                    host,
                    generation,
                    code,
                } => self.exit(processes, &host, generation, code),
            }
        }
        self.progress(processes);
    }

    /// A target engine says what build it runs: the first one fixes the
    /// epoch's, and every other must report the same.
    fn hello(&mut self, processes: &mut Processes, host: &str, generation: u64, build: String) {
        if self.phase != Phase::Preparing
            || !matches!(self.node_of(host, generation), Some((_, true)))
        {
            return;
        }
        let Form::Epoch(trial) = &mut self.form else {
            return;
        };
        match &trial.build {
            Some(expected) if *expected == build => {}
            Some(expected) => {
                let reason =
                    format!("the engine on {host} runs build {build}, the epoch runs {expected}");
                self.abandon(processes, failure("build_mismatch", reason));
            }
            None => {
                trial.build = Some(build.clone());
                trial.switch.record.config.executable = build;
                self.epochs
                    .record(trial.switch.epoch, trial.switch.record.clone());
            }
        }
    }

    /// One report of one engine.
    fn report_arrived(
        &mut self,
        processes: &mut Processes,
        host: &str,
        generation: u64,
        report: Report,
    ) {
        // Another engine's report is not the update's: a standby's, or an
        // epoch that ended.
        let Some((node, target)) = self.node_of(host, generation) else {
            return;
        };
        match &self.form {
            Form::Generation => self.generation_report(processes, node, report),
            Form::Epoch(_) if target => self.target_report(processes, node, report),
            Form::Epoch(_) => self.previous_report(processes, node, report),
        }
    }

    /// A report of a serving engine in a generation update.
    fn generation_report(&mut self, processes: &mut Processes, node: usize, report: Report) {
        match (self.phase, self.back, &report.kind) {
            (Phase::Preparing, _, Some(report::Kind::Prepared(prepared))) => {
                self.uncompared.extend(prepared.uncompared.iter().cloned());
                self.nodes[node].generation = Some(prepared.generation);
                self.acknowledged.insert(node);
            }
            (Phase::Preparing, _, Some(report::Kind::PrepareFailed(failed))) => {
                let reason = failed.reason.clone();
                self.abandon(processes, failure("prepare_failed", reason));
            }
            (Phase::Preparing, _, _) => {
                if let Some(failure) = Failure::of_report(&report) {
                    self.abandon(processes, failure);
                }
            }
            (Phase::Switching, false, Some(report::Kind::Switched(_))) => {
                self.acknowledged.insert(node);
            }
            // The engine went back to the kept generation by itself: a
            // capture that failed while it switched, or a fault on trial.
            (Phase::Switching, false, Some(report::Kind::RolledBack(rolled)))
            | (Phase::Trial, _, Some(report::Kind::RolledBack(rolled))) => {
                let reason = rolled.reason.clone();
                self.roll_back(processes, failure("rolled_back", reason), Some(node));
            }
            (Phase::Switching, true, Some(report::Kind::RolledBack(_))) => {
                self.acknowledged.insert(node);
            }
            (Phase::Switching, false, _) | (Phase::Trial, _, _) => {
                if let Some(failure) = Failure::of_report(&report) {
                    self.roll_back(processes, failure, None);
                }
            }
            _ => {}
        }
    }

    /// A report of a target engine in an epoch update.
    fn target_report(&mut self, processes: &mut Processes, node: usize, report: Report) {
        let Form::Epoch(trial) = &mut self.form else {
            return;
        };
        match (self.phase, self.back, &report.kind) {
            (Phase::Preparing, _, Some(report::Kind::Parked(parked))) => {
                self.nodes[node].generation = Some(parked.generation);
                // The generation the epoch's engines numbered for its
                // modules: a promotion is refused on any other.
                trial.switch.record.generation = Some(parked.generation);
                self.epochs
                    .record(trial.switch.epoch, trial.switch.record.clone());
                self.acknowledged.insert(node);
            }
            (Phase::Preparing, _, _) => {
                if let Some(failure) = Failure::of_report(&report) {
                    self.abandon(processes, failure);
                }
            }
            (Phase::Switching, false, Some(report::Kind::Serving(serving))) => {
                let held = serving.generation;
                match trial.switch.record.generation {
                    Some(epoch) if epoch != held => {
                        let reason = format!(
                            "the promoted engine on {} executes generation {held} and the \
                             epoch's is {epoch}: requests are not moved onto modules the epoch \
                             does not serve",
                            self.nodes[node].host
                        );
                        self.roll_back(processes, failure("stale_modules", reason), Some(node));
                    }
                    _ => {
                        self.acknowledged.insert(node);
                    }
                }
            }
            (Phase::Switching, true, Some(report::Kind::Drained(_))) => {
                trial.target_left.insert(node);
            }
            (Phase::Switching, false, _) | (Phase::Trial, _, _) => {
                if let Some(failure) = Failure::of_report(&report) {
                    self.roll_back(processes, failure, Some(node));
                }
            }
            _ => {}
        }
    }

    /// A report of a previous-epoch engine in an epoch update.
    fn previous_report(&mut self, processes: &mut Processes, node: usize, report: Report) {
        let Form::Epoch(trial) = &mut self.form else {
            return;
        };
        match (self.phase, self.back, report.kind) {
            (Phase::Switching, back, Some(report::Kind::Drained(drained)))
                if drained.epoch == self.serving =>
            {
                trial.drained.insert(node);
                if !back {
                    self.nodes[node].handoff = Some(drained.handoff.unwrap_or_default());
                } else if trial.returning {
                    // Its drain was under way when the switch turned back: it
                    // is parked now, and serves again.
                    self.promote_previous(processes, node);
                }
            }
            (Phase::Switching, true, Some(report::Kind::Serving(_))) => {
                self.acknowledged.insert(node);
            }
            _ => {}
        }
    }

    /// An engine of launch generation `generation` exited.
    fn exit(&mut self, processes: &mut Processes, host: &str, generation: u64, code: Option<i32>) {
        let Some((node, target)) = self.node_of(host, generation) else {
            return;
        };
        let reason = format!(
            "the engine of launch generation {generation} on {host} exited{}",
            code.map(|code| format!(" with {code}")).unwrap_or_default()
        );
        let exited = failure("process_exit", reason);

        let Form::Epoch(trial) = &mut self.form else {
            // The engine that held both generations is gone: its failure
            // domain is replaced by a standby or a relaunch of the modules
            // the epoch's record names, which are the kept ones until a
            // commit.
            self.promotion_due = true;
            match self.phase {
                Phase::Preparing => self.abandon(processes, exited),
                Phase::Switching | Phase::Trial => self.roll_back(processes, exited, Some(node)),
                _ => {}
            }
            return;
        };
        if target {
            trial.target_left.insert(node);
            match (self.phase, self.back) {
                (Phase::Preparing, _) => self.abandon(processes, exited),
                (Phase::Switching, false) | (Phase::Trial, _) => {
                    self.roll_back(processes, exited, Some(node));
                }
                _ => {}
            }
            return;
        }

        trial.previous_exited.insert(node);
        trial.drained.remove(&node);
        match (self.phase, self.back) {
            (Phase::Preparing, _) => {
                self.promotion_due = true;
                self.abandon(processes, exited);
            }
            (Phase::Switching, false) => {
                // Nothing is left to drain there, and nothing to hand over:
                // the target is promoted without its state.
                self.promotion_due = true;
                trial.drained.insert(node);
            }
            (Phase::Switching, true) => {
                // The engine that was to serve again is gone: its owner
                // starts the previous epoch there.
                if trial.returning && !self.acknowledged.contains(&node) {
                    trial.relaunch_due.insert(node);
                }
            }
            // The trial stopped it: its exit lets the node release all but
            // the weights it keeps.
            (Phase::Trial, _) if trial.previous_stopped => self.keep_weights(processes, node),
            (Phase::Trial, _) => self.promotion_due = true,
            _ => {}
        }
    }

    /// `host`'s previous-epoch engine ended for the trial: the node keeps its
    /// weights alone, in host storage when the update asked and the node can.
    ///
    /// An engine the daemon's engine control started ends with its
    /// connection, which no process event states, so its owner reports it
    /// here as well; the node is asked once.
    pub(crate) fn previous_ended(&mut self, processes: &mut Processes, host: &str) {
        if let Some(node) = self.nodes.iter().position(|node| node.host == host) {
            self.keep_weights(processes, node);
        }
    }

    /// Asks `node` to keep the previous epoch's weights. A node that refuses,
    /// or fails to keep them, keeps them on its GPUs: nothing else of the
    /// trial depends on it, and the status says why.
    fn keep_weights(&mut self, processes: &mut Processes, node: usize) {
        let serving = self.serving;
        let Form::Epoch(trial) = &mut self.form else {
            return;
        };
        if self.phase != Phase::Trial || !trial.previous_stopped {
            return;
        }
        trial.previous_exited.insert(node);
        if !trial.weights_kept.insert(node) {
            return;
        }
        let host = &self.nodes[node].host;
        let to_host = trial.switch.keeps == PreviousEpoch::Host;
        match processes.keep_weights(host, serving, to_host) {
            Ok(kept) => {
                trial.moved_bytes += kept.moved_bytes;
                if !kept.refused.is_empty() {
                    trial.kept = PreviousEpoch::Weights;
                    trial.host_refused.push(format!("{host}: {}", kept.refused));
                }
            }
            Err(reason) => {
                if to_host {
                    trial.kept = PreviousEpoch::Weights;
                }
                trial.host_refused.push(format!("{host}: {reason}"));
            }
        }
    }

    /// The previous epoch serves again on `host`, at the launch generation
    /// its owner started it at.
    pub(crate) fn relaunched(&mut self, processes: &mut Processes, host: &str, generation: u64) {
        if !(self.phase == Phase::Switching && self.back) {
            return;
        }
        let Some(node) = self.nodes.iter().position(|node| node.host == host) else {
            return;
        };
        self.nodes[node].serving = generation;
        if let Form::Epoch(trial) = &mut self.form {
            trial.relaunch_due.remove(&node);
        }
        self.acknowledged.insert(node);
        self.progress(processes);
    }

    /// The submitter or the operator commits: the new serves from now on and
    /// the kept side is released. A commit asked before the trial began is
    /// kept and applied when it begins. Returns whether an update was there to
    /// commit.
    pub(crate) fn commit(&mut self, processes: &mut Processes) -> bool {
        match (self.phase, self.back) {
            (Phase::Preparing, _) | (Phase::Switching, false) => self.commit_asked = true,
            (Phase::Trial, _) => self.committed(processes),
            _ => return false,
        }
        true
    }

    /// Records the commit, then releases the kept side.
    fn committed(&mut self, processes: &mut Processes) {
        // The decision first: an engine lost from here on is replaced by one
        // of the committed modules.
        match &mut self.form {
            Form::Generation => self.epochs.committed(self.serving),
            Form::Epoch(trial) => {
                trial.switch.record.status = Status::Serving;
                self.epochs
                    .record(trial.switch.epoch, trial.switch.record.clone());
            }
        }
        self.enter(Phase::Committed);

        match &self.form {
            Form::Generation => {
                for node in &self.nodes {
                    processes.command(
                        &node.host,
                        node.serving,
                        command(command::Kind::Commit(process::Commit {})),
                    );
                }
            }
            Form::Epoch(trial) => {
                let retire = vec![self.serving];
                for node in &self.nodes {
                    processes.stop(&self.group, Some(node.serving));
                    // The previous epoch's storage ends on the node that
                    // served it while the target keeps its own.
                    let _ = processes.claim(
                        &node.host,
                        trial.switch.epoch,
                        &self.scopes(node),
                        &retire,
                    );
                }
            }
        }
    }

    /// The submitter or the operator rolls back, or the trial did (a
    /// threshold, the deadline): the kept side serves again and the new is
    /// released. Before the switch the update is abandoned; while the switch
    /// runs it is rolled back once the switch ended.
    pub(crate) fn rollback(&mut self, processes: &mut Processes, failure: Failure) {
        match (self.phase, self.back) {
            (Phase::Preparing, _) => self.abandon(processes, failure),
            // Only an engine that drained can serve again, so the rollback
            // waits for the switch it undoes.
            (Phase::Switching, false) => {
                self.rollback_asked.get_or_insert(failure);
            }
            (Phase::Trial, _) => self.roll_back(processes, failure, None),
            _ => {}
        }
    }

    /// Rolls the update back when its deadline passed at `now`.
    pub(crate) fn expire(&mut self, processes: &mut Processes, now: Instant) {
        if self.deadline.is_some_and(|deadline| deadline <= now) && !self.phase.ended() {
            self.deadline = None;
            self.rollback(
                processes,
                failure(
                    "deadline",
                    "the trial was not committed by its deadline".to_owned(),
                ),
            );
        }
    }

    /// Drops the new before anything served it: the engines discard what they
    /// prepared, a target's engines stop and its epoch is retired by the next
    /// claim, and what served goes on untouched.
    fn abandon(&mut self, processes: &mut Processes, failure: Failure) {
        self.failure = Some(failure.clone());
        match &mut self.form {
            Form::Generation => {
                let rollback = command(command::Kind::Rollback(process::Rollback {
                    reason: failure.reason,
                }));
                for node in &self.nodes {
                    processes.command(&node.host, node.serving, rollback.clone());
                }
            }
            Form::Epoch(trial) => {
                trial.switch.retire.push(trial.switch.epoch);
                trial.switch.record.status = Status::Released;
                self.epochs
                    .record(trial.switch.epoch, trial.switch.record.clone());
                for node in &self.nodes {
                    if let Some(target) = node.target {
                        processes.stop(&self.group, Some(target));
                    }
                }
            }
        }
        self.enter(Phase::Abandoned);
    }

    /// Runs the switch back with `failure` as the update's first one. `failed`
    /// names the node whose engine is the failure: it rolled back by itself,
    /// failed or exited, and is not commanded again.
    fn roll_back(&mut self, processes: &mut Processes, failure: Failure, failed: Option<usize>) {
        if self.phase == Phase::Switching && self.back {
            return;
        }
        self.failure.get_or_insert(failure.clone());
        // Whether the new ever served: a target no request was moved onto
        // holds nothing to hand back.
        let served = self.phase == Phase::Trial;
        self.back = true;
        self.enter(Phase::Switching);

        match &mut self.form {
            Form::Generation => {
                let rollback = command(command::Kind::Rollback(process::Rollback {
                    reason: failure.reason,
                }));
                for (index, node) in self.nodes.iter().enumerate() {
                    if failed == Some(index) {
                        self.acknowledged.insert(index);
                    } else {
                        processes.command(&node.host, node.serving, rollback.clone());
                    }
                }
            }
            Form::Epoch(trial) => {
                let drain = command(command::Kind::Drain(process::Drain {
                    epoch: trial.switch.epoch,
                }));
                for (index, node) in self.nodes.iter().enumerate() {
                    let Some(target) = node.target else {
                        continue;
                    };
                    if trial.target_left.contains(&index) {
                        continue;
                    }
                    if served && failed != Some(index) {
                        // Its requests end as migratable at their step
                        // boundary, as the previous epoch's did.
                        processes.command(&node.host, target, drain.clone());
                    } else {
                        trial.target_left.insert(index);
                        processes.stop(&self.group, Some(target));
                    }
                }
            }
        }
        self.progress(processes);
    }

    /// Promotes the parked previous-epoch engine of `node` again, with no
    /// state handed to it: the target wrote the cache it left.
    fn promote_previous(&mut self, processes: &mut Processes, node: usize) {
        let node = &self.nodes[node];
        processes.command(
            &node.host,
            node.serving,
            command(command::Kind::Promote(process::Promote { handoff: None })),
        );
    }

    /// Fires the next step when the acknowledgements the phase waits for
    /// arrived.
    fn progress(&mut self, processes: &mut Processes) {
        match (self.phase, self.back) {
            (Phase::Preparing, _) if self.all_acknowledged() && self.warming.is_empty() => {
                self.switch(processes);
            }
            (Phase::Switching, false) => self.progress_switch(processes),
            (Phase::Switching, true) => self.progress_rollback(processes),
            _ => {}
        }
    }

    /// Starts `Switching`.
    fn switch(&mut self, processes: &mut Processes) {
        self.enter(Phase::Switching);
        self.switch_began = Some(Instant::now());
        for node in &self.nodes {
            let kind = match &self.form {
                Form::Generation => command::Kind::Switch(process::Switch {
                    generation: node.generation.unwrap_or_default(),
                }),
                Form::Epoch(_) => command::Kind::Drain(process::Drain {
                    epoch: self.serving,
                }),
            };
            processes.command(&node.host, node.serving, command(kind));
        }
    }

    /// The forward switch: once every serving engine drained the targets are
    /// promoted with what each handed over, and once every engine
    /// acknowledged the trial begins.
    fn progress_switch(&mut self, processes: &mut Processes) {
        if let Form::Epoch(trial) = &mut self.form
            && !trial.promoted
            && trial.drained.len() == self.nodes.len()
        {
            trial.promoted = true;
            // The previous epoch's storage frees at the promotion when its
            // engines are not kept, so the target's cache grows into it.
            let grows = trial.switch.keeps != PreviousEpoch::Parked;
            let recompute = trial.switch.recompute;
            for node in &mut self.nodes {
                let Some(target) = node.target else {
                    continue;
                };
                let mut handoff = node.handoff.take().unwrap_or_default();
                handoff.recompute = recompute;
                handoff.grows = grows;
                processes.command(
                    &node.host,
                    target,
                    command(command::Kind::Promote(process::Promote {
                        handoff: Some(handoff),
                    })),
                );
            }
        }
        if !self.all_acknowledged() {
            return;
        }

        self.pause = self.switch_began.map(|began| began.elapsed());
        match &mut self.form {
            Form::Generation => {
                self.epochs
                    .switched(self.serving, self.target.modules.clone());
            }
            Form::Epoch(trial) => {
                trial.switch.record.status = Status::Trial;
                self.epochs
                    .record(trial.switch.epoch, trial.switch.record.clone());
                processes.serve(trial.switch.epoch, &trial.switch.requests);
            }
        }
        self.trial_began = Some(Instant::now());
        self.enter(Phase::Trial);

        if let Some(failure) = self.rollback_asked.take() {
            return self.roll_back(processes, failure, None);
        }
        if let Form::Epoch(trial) = &mut self.form
            && trial.switch.keeps != PreviousEpoch::Parked
        {
            // What the trial keeps of the previous epoch: an engine that
            // serves nothing holds storage the target could use.
            trial.previous_stopped = true;
            for node in &self.nodes {
                processes.stop(&self.group, Some(node.serving));
            }
        }
        if std::mem::take(&mut self.commit_asked) {
            self.committed(processes);
        }
    }

    /// The switch back: the generation form ends on every `RolledBack`; the
    /// epoch form returns the previous epoch to serving once every target
    /// left, and ends once every node serves it again.
    fn progress_rollback(&mut self, processes: &mut Processes) {
        match &mut self.form {
            Form::Generation => {
                if !self.all_acknowledged() {
                    return;
                }
                // The record took the new modules only once they served.
                if self.trial_began.is_some() {
                    self.epochs.rolled_back(self.serving);
                }
            }
            Form::Epoch(trial) => {
                let targets_left = self.nodes.iter().enumerate().all(|(index, node)| {
                    node.target.is_none() || trial.target_left.contains(&index)
                });
                if !targets_left {
                    return;
                }
                if !trial.returning {
                    trial.returning = true;
                    // The targets hold the storage the previous epoch takes
                    // back, so they end before it serves.
                    for node in &self.nodes {
                        if let Some(target) = node.target {
                            processes.stop(&self.group, Some(target));
                        }
                    }
                    let promote =
                        command(command::Kind::Promote(process::Promote { handoff: None }));
                    for (index, node) in self.nodes.iter().enumerate() {
                        if trial.previous_stopped || trial.previous_exited.contains(&index) {
                            trial.relaunch_due.insert(index);
                        } else if trial.drained.contains(&index) {
                            processes.command(&node.host, node.serving, promote.clone());
                        }
                        // An engine whose drain is still under way is
                        // promoted when its `Drained` arrives.
                    }
                    // The record is what an engine started now reads its
                    // modules from: the target's stands no more.
                    trial.switch.record.status = Status::Released;
                    self.epochs
                        .record(trial.switch.epoch, trial.switch.record.clone());
                }
                if self.acknowledged.len() != self.nodes.len() {
                    return;
                }
                processes.serve(self.serving, &trial.switch.requests);
                trial.switch.retire.push(trial.switch.epoch);
                let retire = std::mem::take(&mut trial.switch.retire);
                for node in &self.nodes {
                    let _ = processes.claim(&node.host, self.serving, &self.scopes(node), &retire);
                }
            }
        }
        self.enter(Phase::RolledBack);
    }
}

/// One command of `kind`.
fn command(kind: command::Kind) -> process::Command {
    process::Command { kind: Some(kind) }
}

/// A failure of `kind` the daemon states itself.
fn failure(kind: &str, reason: String) -> Failure {
    Failure {
        kind: kind.to_owned(),
        reason,
        device: None,
    }
}
