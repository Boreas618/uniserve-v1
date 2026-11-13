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
//! | Preparing | `Prepare` to each serving engine; `Prepared` acknowledges | Every target engine, on every node and of every replica, starts and parks; `Parked` acknowledges |
//! | Switching | `Switch` to each serving engine; `Switched` acknowledges | `Promote` to every target, which serves from its `Serving`; then `Drain` to every serving engine, with the drain's deadline and `shrink` when the layouts differ; `Drained` acknowledges |
//! | Kept | The previous generation, inside the engine | The previous epoch: parked, its weights, or its weights in host storage |
//! | Rollback | `Rollback`; `RolledBack` acknowledges | The switch the other way: the previous epoch is promoted again, or started again where it was stopped, and the targets drain; a failed target is stopped |
//!
//! A rollback is the switch run back, so the phase is `Switching` again until
//! the kept side serves and the new one drained, and `RolledBack` after. A
//! commit records its decision (the phase and the epoch's record) before it
//! releases anything, so an engine lost while the kept side is released is
//! replaced by one of the committed modules.
//!
//! Every phase ends on acknowledgements the engines make, never on an
//! assumption about their progress. The epoch form's two epochs step together
//! while it switches: new requests go to the side promoted from its first
//! `Serving`, and the draining side steps what it holds until each request
//! ends or the deadline makes it migratable. The buffer pool keeps them from
//! writing one block (`specs/buffer-pool.md` §9), and the two epochs need not
//! have the same nodes, main processes, schedulers or ranks: nothing maps one
//! epoch's engines to the other's.

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

/// One engine process the update commands: a serving engine (a scheduler
/// node's main process or a peer's), or a target the epoch form started.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node {
    /// The host identity of the node it runs on, as rank layouts name it.
    host: String,
    /// The group rank of its first rank, which its storage scope is keyed by.
    rank: u32,
    /// Its launch generation, which commands address it by.
    launch: u64,
    /// The generation of modules it numbered: what a serving engine
    /// prepared, or what a target holds.
    generation: Option<u64>,
}

impl Node {
    /// The engine of launch generation `launch` on `host`, whose first rank
    /// is group rank `rank`.
    pub(crate) fn new(host: String, rank: u32, launch: u64) -> Self {
        Self {
            host,
            rank,
            launch,
            generation: None,
        }
    }
}

/// What the epoch form states of its target, beside the serving engines.
pub(crate) struct EpochSwitch {
    /// The target epoch's number, which its record is kept under.
    pub(crate) epoch: u64,
    /// The target epoch's record, whose status the update advances.
    pub(crate) record: EpochRecord,
    /// Every target engine's launch file, one per host of each target
    /// replica's world, with the reservations of that host it inherits (a
    /// scheduler node's peer channel and rendezvous). Each states the epoch's
    /// module set, the engine it succeeds and its own launch generation
    /// (`engine`), so the engine parks by itself once it is ready.
    pub(crate) launches: Vec<(proto::config::EngineArgs, Vec<u64>)>,
    /// Variables set over the daemon's environment for the target engines.
    pub(crate) environment: std::collections::HashMap<String, String>,
    /// The request socket requests are submitted to after the switch.
    pub(crate) requests: String,
    /// What the previous epoch keeps through the trial
    /// (`update_config.previous_epoch`).
    pub(crate) keeps: PreviousEpoch,
    /// Whether the requests that migrate compute their state again
    /// (`in_flight = recompute`).
    pub(crate) recompute: bool,
    /// How long a draining engine steps its requests after its `Drain`
    /// (`update_config.drain_s`).
    pub(crate) drain: Duration,
    /// Whether the two epochs' layouts differ: the side that drains gives its
    /// device memory back as it drains, and the side promoted grows into it.
    pub(crate) shrink: bool,
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
/// update's serving engines (`nodes`) or into its targets (`targets`).
struct EpochTrial {
    switch: EpochSwitch,
    /// Every target engine, in the order of `switch.launches`.
    targets: Vec<Node>,
    /// The build the target engines report, fixed by the first one's `Hello`.
    build: Option<String>,
    /// What the previous epoch is kept as: `keeps`, or `weights` once a node
    /// could not move them to host storage.
    kept: PreviousEpoch,
    /// Each node's reason for keeping the weights on its GPUs.
    host_refused: Vec<String>,
    /// The bytes of weights the nodes moved to host storage.
    moved_bytes: u64,
    /// Targets that acknowledged `Parked`.
    parked: BTreeSet<usize>,
    /// Targets that serve: their `Serving` acknowledged the promotion.
    serving: BTreeSet<usize>,
    /// Whether the serving engines were sent their drain.
    draining: bool,
    /// Serving engines that drained and are parked, or exited during the
    /// switch: nothing is left to drain there.
    drained: BTreeSet<usize>,
    /// Whether the rollback sent the targets their drain.
    targets_draining: bool,
    /// Targets that serve nothing more: they drained, were stopped or
    /// exited.
    target_left: BTreeSet<usize>,
    /// Whether the trial stopped the previous epoch's engines: a rollback
    /// then starts them again.
    previous_stopped: bool,
    /// Serving engines that exited.
    previous_exited: BTreeSet<usize>,
    /// Serving engines whose node was asked to keep their weights.
    weights_kept: BTreeSet<usize>,
    /// Serving engines that have to start again before they serve.
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
    /// The engines that serve when the update begins.
    nodes: Vec<Node>,
    /// The serving engines that acknowledged the step the phase waits for:
    /// a generation's `Prepared`, `Switched` or `RolledBack`, or for an
    /// epoch's rollback their `Serving` again.
    acknowledged: BTreeSet<usize>,
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
    /// When the switch began, and how long it took.
    switch_began: Option<Instant>,
    switched_in: Option<Duration>,
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
            targets: Vec::new(),
            build: None,
            kept,
            host_refused: Vec::new(),
            moved_bytes: 0,
            parked: BTreeSet::new(),
            serving: BTreeSet::new(),
            draining: false,
            drained: BTreeSet::new(),
            targets_draining: false,
            target_left: BTreeSet::new(),
            previous_stopped: false,
            previous_exited: BTreeSet::new(),
            weights_kept: BTreeSet::new(),
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
            back: false,
            failure: None,
            commit_asked: false,
            rollback_asked: None,
            deadline,
            switch_began: None,
            switched_in: None,
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

    /// The launch generations of the previous epoch's engines that have to
    /// start again for a rollback: the trial stopped them, or they exited,
    /// and each serves once its owner started it on the weights it kept and
    /// said so ([`Self::relaunched`]).
    pub(crate) fn relaunch_due(&self) -> BTreeSet<u64> {
        match &self.form {
            Form::Generation => BTreeSet::new(),
            Form::Epoch(trial) => trial
                .relaunch_due
                .iter()
                .map(|index| self.nodes[*index].launch)
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
            "switch_ms": self.switched_in.map(|switched| switched.as_millis()),
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
                report["engines"] = json!({
                    "serving": self.nodes.len(),
                    "target": trial.switch.launches.len(),
                });
                report["drain_s"] = json!(trial.switch.drain.as_secs_f64());
                report["shrink"] = json!(trial.switch.shrink);
                report["previous_epoch_kept"] = json!(trial.kept);
                report["host_refused"] = json!(trial.host_refused);
                report["moved_to_host_bytes"] = json!(trial.moved_bytes);
            }
        }
        report
    }

    /// Whether every serving engine acknowledged the step the phase waits
    /// for.
    fn all_acknowledged(&self) -> bool {
        self.acknowledged.len() == self.nodes.len()
    }

    /// Enters `phase`: the step it waits for is acknowledged by nobody yet.
    fn enter(&mut self, phase: Phase) {
        self.phase = phase;
        self.acknowledged.clear();
        self.entered.push(phase);
    }

    /// The node whose engine of launch generation `launch` runs on `host`,
    /// and whether that engine is one of the epoch form's targets.
    fn node_of(&self, host: &str, launch: u64) -> Option<(usize, bool)> {
        let find = |nodes: &[Node]| {
            nodes
                .iter()
                .position(|node| node.host == host && node.launch == launch)
        };
        if let Form::Epoch(trial) = &self.form
            && let Some(target) = find(&trial.targets)
        {
            return Some((target, true));
        }
        find(&self.nodes).map(|node| (node, false))
    }

    /// Claims `epoch`'s storage on every host either epoch runs on, retiring
    /// `retire`'s epochs: a claim retires storage only on the node it is made
    /// on, and a host only one epoch uses keeps or releases its storage all
    /// the same. Each host's scopes are those of `keeping`'s engines there.
    fn claim_everywhere(
        &self,
        processes: &mut Processes,
        epoch: u64,
        keeping: &[Node],
        retire: &[u64],
    ) -> Result<(), String> {
        let mut hosts: BTreeSet<&str> = self.nodes.iter().map(|node| node.host.as_str()).collect();
        if let Form::Epoch(trial) = &self.form {
            hosts.extend(trial.targets.iter().map(|node| node.host.as_str()));
        }
        for host in hosts {
            let mut scopes: Vec<process::Scope> = Vec::new();
            for node in keeping.iter().filter(|node| node.host == host) {
                let scope = process::Scope {
                    group: self.group.clone(),
                    rank: node.rank,
                };
                if !scopes.contains(&scope) {
                    scopes.push(scope);
                }
            }
            processes.claim(host, epoch, &scopes, retire)?;
        }
        Ok(())
    }

    /// Starts `Preparing`.
    pub(crate) fn start(&mut self, processes: &mut Processes) {
        for node in &self.nodes {
            processes.adopt(&node.host, &self.group, node.rank, node.launch);
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
            processes.command(&node.host, node.launch, prepare.clone());
        }
    }

    /// Every node starts the target engines of its host, every replica's
    /// scheduler node and peers alike, then claims the target epoch's storage
    /// before the engines map it.
    fn prepare_epoch(&mut self, processes: &mut Processes) {
        let Form::Epoch(trial) = &mut self.form else {
            return;
        };
        trial.switch.record.status = Status::Preparing;
        self.epochs
            .record(trial.switch.epoch, trial.switch.record.clone());

        let launches = trial.switch.launches.clone();
        let environment = trial.switch.environment.clone();
        for (launch, inherit) in launches {
            let spawn = Spawn {
                host: launch.host.clone(),
                group: self.group.clone(),
                rank: launch.rank,
                generation: launch.engine,
                // The node's daemon resolves the engine package the launch
                // names and writes the launch file for this process.
                executable: String::new(),
                arguments: Vec::new(),
                environment: environment.clone(),
                inherit,
                launch: Vec::new(),
            };
            if let Err(reason) = processes.spawn(spawn, &launch) {
                return self.abandon(processes, failure("spawn_failed", reason));
            }
            if let Form::Epoch(trial) = &mut self.form {
                trial
                    .targets
                    .push(Node::new(launch.host.clone(), launch.rank, launch.engine));
            }
        }

        let Form::Epoch(trial) = &self.form else {
            return;
        };
        let (epoch, targets) = (trial.switch.epoch, trial.targets.clone());
        let retire = trial.switch.retire.clone();
        if let Err(reason) = self.claim_everywhere(processes, epoch, &targets, &retire) {
            return self.abandon(processes, failure("claim_failed", reason));
        }
        // Every node retired them: none is left for a later claim.
        if let Form::Epoch(trial) = &mut self.form {
            trial.switch.retire.clear();
        }
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
    fn hello(&mut self, processes: &mut Processes, host: &str, launch: u64, build: String) {
        if self.phase != Phase::Preparing || !matches!(self.node_of(host, launch), Some((_, true)))
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
        launch: u64,
        report: Report,
    ) {
        // Another engine's report is not the update's: a standby's, or an
        // epoch that ended.
        let Some((index, target)) = self.node_of(host, launch) else {
            return;
        };
        match &self.form {
            Form::Generation => self.generation_report(processes, index, report),
            Form::Epoch(_) if target => self.target_report(processes, index, report),
            Form::Epoch(_) => self.previous_report(processes, index, report),
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
    fn target_report(&mut self, processes: &mut Processes, target: usize, report: Report) {
        let Form::Epoch(trial) = &mut self.form else {
            return;
        };
        match (self.phase, self.back, &report.kind) {
            (Phase::Preparing, _, Some(report::Kind::Parked(parked))) => {
                trial.targets[target].generation = Some(parked.generation);
                // The generation the epoch's engines numbered for its
                // modules: a promotion is refused on any other.
                trial.switch.record.generation = Some(parked.generation);
                self.epochs
                    .record(trial.switch.epoch, trial.switch.record.clone());
                trial.parked.insert(target);
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
                            trial.targets[target].host
                        );
                        self.roll_back(processes, failure("stale_modules", reason), Some(target));
                    }
                    _ => {
                        // New requests go to the target from its first
                        // `Serving`: the serving epoch still serves what it
                        // has, so none waits.
                        if trial.serving.is_empty() {
                            processes.serve(trial.switch.epoch, &trial.switch.requests);
                        }
                        trial.serving.insert(target);
                    }
                }
            }
            (Phase::Switching, true, Some(report::Kind::Drained(_))) => {
                trial.target_left.insert(target);
            }
            // A target that fails while the switch runs back is stopped: it
            // drains nothing more.
            (Phase::Switching, true, _) => {
                if Failure::of_report(&report).is_some() && trial.target_left.insert(target) {
                    processes.stop(&self.group, Some(trial.targets[target].launch));
                }
            }
            (Phase::Switching, false, _) | (Phase::Trial, _, _) => {
                if let Some(failure) = Failure::of_report(&report) {
                    self.roll_back(processes, failure, Some(target));
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
                // Its drain was under way when the switch turned back: it is
                // parked now, and serves again.
                if back && !trial.relaunch_due.contains(&node) {
                    self.promote_previous(processes, node);
                }
            }
            (Phase::Switching, true, Some(report::Kind::Serving(_))) => {
                self.previous_serves(processes, node);
            }
            _ => {}
        }
    }

    /// The previous epoch's engine `node` serves again for the rollback: new
    /// requests return to the previous epoch from the first such engine.
    fn previous_serves(&mut self, processes: &mut Processes, node: usize) {
        if self.acknowledged.is_empty()
            && let Form::Epoch(trial) = &self.form
        {
            processes.serve(self.serving, &trial.switch.requests);
        }
        self.acknowledged.insert(node);
    }

    /// An engine of launch generation `launch` exited.
    fn exit(&mut self, processes: &mut Processes, host: &str, launch: u64, code: Option<i32>) {
        let Some((index, target)) = self.node_of(host, launch) else {
            return;
        };
        let reason = format!(
            "the engine of launch generation {launch} on {host} exited{}",
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
                Phase::Switching | Phase::Trial => self.roll_back(processes, exited, Some(index)),
                _ => {}
            }
            return;
        };
        if target {
            trial.target_left.insert(index);
            match (self.phase, self.back) {
                (Phase::Preparing, _) => self.abandon(processes, exited),
                (Phase::Switching, false) | (Phase::Trial, _) => {
                    self.roll_back(processes, exited, Some(index));
                }
                _ => {}
            }
            return;
        }

        trial.previous_exited.insert(index);
        match (self.phase, self.back) {
            (Phase::Preparing, _) => {
                self.promotion_due = true;
                self.abandon(processes, exited);
            }
            (Phase::Switching, false) => {
                // Nothing is left to drain there: its requests end as
                // migratable with its loss, and the target takes them.
                self.promotion_due = true;
                trial.drained.insert(index);
            }
            (Phase::Switching, true) => {
                // The engine that was to serve again is gone: its owner
                // starts the previous epoch there.
                trial.drained.remove(&index);
                if !self.acknowledged.contains(&index) {
                    trial.relaunch_due.insert(index);
                }
            }
            // The trial stopped it: its exit lets the node release all but
            // the weights it keeps.
            (Phase::Trial, _) if trial.previous_stopped => self.keep_weights(processes, index),
            (Phase::Trial, _) => self.promotion_due = true,
            _ => {}
        }
    }

    /// The previous epoch's engine of launch generation `launch` ended for
    /// the trial: its node keeps its weights alone, in host storage when the
    /// update asked and the node can.
    ///
    /// An engine the daemon's engine control started ends with its
    /// connection, which no process event states, so its owner reports it
    /// here as well; the node is asked once.
    pub(crate) fn previous_ended(&mut self, processes: &mut Processes, launch: u64) {
        if let Some(node) = self.nodes.iter().position(|node| node.launch == launch) {
            self.keep_weights(processes, node);
        }
    }

    /// Asks `node`'s host to keep the previous epoch's weights. A node that
    /// refuses, or fails to keep them, keeps them on its GPUs: nothing else
    /// of the trial depends on it, and the status says why.
    fn keep_weights(&mut self, processes: &mut Processes, node: usize) {
        let serving = self.serving;
        let Form::Epoch(trial) = &mut self.form else {
            return;
        };
        if self.phase != Phase::Trial || !trial.previous_stopped {
            return;
        }
        trial.previous_exited.insert(node);
        let host = &self.nodes[node].host;
        // A host is asked once, however many of the epoch's engines ran there.
        let asked = trial
            .weights_kept
            .iter()
            .any(|kept| self.nodes[*kept].host == *host);
        trial.weights_kept.insert(node);
        if asked {
            return;
        }
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

    /// The previous epoch's engine of launch generation `previous` serves
    /// again at launch generation `launch`, which its owner started it at.
    pub(crate) fn relaunched(&mut self, processes: &mut Processes, previous: u64, launch: u64) {
        if !(self.phase == Phase::Switching && self.back) {
            return;
        }
        let Some(node) = self.nodes.iter().position(|node| node.launch == previous) else {
            return;
        };
        self.nodes[node].launch = launch;
        if let Form::Epoch(trial) = &mut self.form {
            trial.relaunch_due.remove(&node);
        }
        self.previous_serves(processes, node);
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
                        node.launch,
                        command(command::Kind::Commit(process::Commit {})),
                    );
                }
            }
            Form::Epoch(trial) => {
                for node in &self.nodes {
                    processes.stop(&self.group, Some(node.launch));
                }
                // The previous epoch's storage ends on every node while the
                // target keeps its own.
                let (epoch, targets) = (trial.switch.epoch, trial.targets.clone());
                let _ = self.claim_everywhere(processes, epoch, &targets, &[self.serving]);
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
            // The rollback undoes a switch that ended, so it waits for it.
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
                    processes.command(&node.host, node.launch, rollback.clone());
                }
            }
            Form::Epoch(trial) => {
                trial.switch.retire.push(trial.switch.epoch);
                trial.switch.record.status = Status::Released;
                self.epochs
                    .record(trial.switch.epoch, trial.switch.record.clone());
                for target in &trial.targets {
                    processes.stop(&self.group, Some(target.launch));
                }
            }
        }
        self.enter(Phase::Abandoned);
    }

    /// Runs the switch back with `failure` as the update's first one.
    /// `failed` names the engine that is the failure (a serving engine of
    /// the generation form, a target of the epoch form): it rolled back by
    /// itself, failed or exited, and is not commanded again.
    fn roll_back(&mut self, processes: &mut Processes, failure: Failure, failed: Option<usize>) {
        if self.phase == Phase::Switching && self.back {
            return;
        }
        self.failure.get_or_insert(failure.clone());
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
                        processes.command(&node.host, node.launch, rollback.clone());
                    }
                }
            }
            Form::Epoch(trial) => {
                // The record is what an engine started now reads its modules
                // from: the target's stands no more.
                trial.switch.record.status = Status::Released;
                self.epochs
                    .record(trial.switch.epoch, trial.switch.record.clone());
                // A target that failed is stopped instead of drained: its
                // requests end as migratable from what they committed.
                if let Some(index) = failed
                    && trial.target_left.insert(index)
                {
                    processes.stop(&self.group, Some(trial.targets[index].launch));
                }
                // The previous epoch first: an engine never drained still
                // serves, one that drained is promoted again (one still
                // draining when its `Drained` arrives), and one stopped or
                // gone is started again by its owner.
                let mut still_serving = Vec::new();
                for index in 0..self.nodes.len() {
                    if trial.previous_stopped || trial.previous_exited.contains(&index) {
                        trial.relaunch_due.insert(index);
                    } else if !trial.draining {
                        still_serving.push(index);
                    } else if trial.drained.contains(&index) {
                        let node = &self.nodes[index];
                        processes.command(
                            &node.host,
                            node.launch,
                            promote(trial.switch.recompute, trial.switch.shrink),
                        );
                    }
                }
                for index in still_serving {
                    self.previous_serves(processes, index);
                }
            }
        }
        self.progress(processes);
    }

    /// Promotes the parked previous-epoch engine `node` again: it grows its
    /// regions back where the drain gave them up.
    fn promote_previous(&mut self, processes: &mut Processes, node: usize) {
        let Form::Epoch(trial) = &self.form else {
            return;
        };
        let node = &self.nodes[node];
        processes.command(
            &node.host,
            node.launch,
            promote(trial.switch.recompute, trial.switch.shrink),
        );
    }

    /// Fires the next step when the acknowledgements the phase waits for
    /// arrived.
    fn progress(&mut self, processes: &mut Processes) {
        match (self.phase, self.back) {
            (Phase::Preparing, _) if self.prepared() => self.switch(processes),
            (Phase::Switching, false) => self.progress_switch(processes),
            (Phase::Switching, true) => self.progress_rollback(processes),
            _ => {}
        }
    }

    /// Whether every engine acknowledged `Preparing`: every serving engine
    /// its generation, or every target that it parked.
    fn prepared(&self) -> bool {
        match &self.form {
            Form::Generation => self.all_acknowledged(),
            Form::Epoch(trial) => {
                !trial.targets.is_empty() && trial.parked.len() == trial.targets.len()
            }
        }
    }

    /// Starts `Switching`: `Switch` to every serving engine of a generation
    /// update, `Promote` to every target of an epoch update.
    fn switch(&mut self, processes: &mut Processes) {
        self.enter(Phase::Switching);
        self.switch_began = Some(Instant::now());
        match &self.form {
            Form::Generation => {
                for node in &self.nodes {
                    let switch = command::Kind::Switch(process::Switch {
                        generation: node.generation.unwrap_or_default(),
                    });
                    processes.command(&node.host, node.launch, command(switch));
                }
            }
            Form::Epoch(trial) => {
                let promotion = promote(trial.switch.recompute, trial.switch.shrink);
                for target in &trial.targets {
                    processes.command(&target.host, target.launch, promotion.clone());
                }
            }
        }
    }

    /// The forward switch: once every target serves the serving engines are
    /// drained, and once every engine acknowledged the trial begins.
    fn progress_switch(&mut self, processes: &mut Processes) {
        let serving = self.serving;
        if let Form::Epoch(trial) = &mut self.form {
            if !trial.draining && trial.serving.len() == trial.targets.len() {
                trial.draining = true;
                let drain = drain(serving, &trial.switch);
                for (index, node) in self.nodes.iter().enumerate() {
                    if !trial.drained.contains(&index) {
                        processes.command(&node.host, node.launch, drain.clone());
                    }
                }
            }
            if !(trial.draining && trial.drained.len() == self.nodes.len()) {
                return;
            }
        } else if !self.all_acknowledged() {
            return;
        }

        self.switched_in = self.switch_began.map(|began| began.elapsed());
        match &mut self.form {
            Form::Generation => {
                self.epochs
                    .switched(self.serving, self.target.modules.clone());
            }
            Form::Epoch(trial) => {
                trial.switch.record.status = Status::Trial;
                self.epochs
                    .record(trial.switch.epoch, trial.switch.record.clone());
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
                processes.stop(&self.group, Some(node.launch));
            }
        }
        if std::mem::take(&mut self.commit_asked) {
            self.committed(processes);
        }
    }

    /// The switch back: the generation form ends on every `RolledBack`; the
    /// epoch form drains the targets once every previous-epoch engine serves
    /// again, and ends once every target left.
    fn progress_rollback(&mut self, processes: &mut Processes) {
        let all_serve = self.all_acknowledged();
        match &mut self.form {
            Form::Generation => {
                if !all_serve {
                    return;
                }
                // The record took the new modules only once they served.
                if self.trial_began.is_some() {
                    self.epochs.rolled_back(self.serving);
                }
            }
            Form::Epoch(trial) => {
                if !all_serve {
                    return;
                }
                if !trial.targets_draining {
                    trial.targets_draining = true;
                    let drain = drain(trial.switch.epoch, &trial.switch);
                    for (index, target) in trial.targets.iter().enumerate() {
                        if !trial.target_left.contains(&index) {
                            processes.command(&target.host, target.launch, drain.clone());
                        }
                    }
                }
                if trial.target_left.len() != trial.targets.len() {
                    return;
                }
                for target in &trial.targets {
                    processes.stop(&self.group, Some(target.launch));
                }
                trial.switch.retire.push(trial.switch.epoch);
                let retire = std::mem::take(&mut trial.switch.retire);
                let nodes = self.nodes.clone();
                let _ = self.claim_everywhere(processes, self.serving, &nodes, &retire);
            }
        }
        self.enter(Phase::RolledBack);
    }
}

/// One command of `kind`.
fn command(kind: command::Kind) -> process::Command {
    process::Command { kind: Some(kind) }
}

/// The promotion of an epoch switch: whether the requests that migrate onto
/// the engine compute their state again, and whether it grows its regions as
/// the engine draining beside it gives its memory back.
fn promote(recompute: bool, grow: bool) -> process::Command {
    command(command::Kind::Promote(process::Promote { recompute, grow }))
}

/// The drain of the engines executing `epoch` in `switch`: the deadline from
/// now, and `shrink` when the two epochs' layouts differ.
fn drain(epoch: u64, switch: &EpochSwitch) -> process::Command {
    command(command::Kind::Drain(process::Drain {
        epoch,
        deadline_ms: u64::try_from(switch.drain.as_millis()).unwrap_or(u64::MAX),
        shrink: switch.shrink,
    }))
}

/// A failure of `kind` the daemon states itself.
fn failure(kind: &str, reason: String) -> Failure {
    Failure {
        kind: kind.to_owned(),
        reason,
        device: None,
    }
}
