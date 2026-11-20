//! One update: the phases that move a deployment to another set of modules as
//! a trial.
//!
//! An update has one phase machine whatever it moves:
//!
//! | Phase | Ends when |
//! | --- | --- |
//! | `Preparing` | Every engine acknowledged it is ready, or one failed: the update is `Abandoned` and nothing changed |
//! | `Switching` | Every engine acknowledged the switch |
//! | `Trial` | A commit once the trial lasted `update_config.trial`, or a rollback: a fault or a threshold |
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
//! replaced by one of the committed modules. The epoch form claims its target
//! at every node either epoch runs on; its commit publishes the target at
//! each of those nodes, and only there, before any of them retires the
//! previous epoch, and a rollback publishes nothing and retires the target.
//!
//! Every release of an epoch's storage the epoch form makes is followed by
//! `Grow` to the side that serves on: the commit's of the previous epoch to the
//! targets, the rollback's of the target to the previous epoch, and the first
//! claim's of the epochs earlier updates dropped to the serving engines. A
//! released epoch's engines are stopped, and their nodes reap them before a
//! stop returns; the claim that retires its storage ends its quotas'
//! reservations too. The scheduler nodes then create the segments their
//! regions number as far as the quotas and the devices hold them
//! ([`grow_on`]).
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

use proto::admin::{
    ChangeForm, EngineCounts, EpochState, EpochSwitchReport, Failure, FailureKind, Phase,
    PreviousEpoch, UpdateReport,
};
use proto::process::{self, Report, Spawn, command, report};

use crate::epoch::{Epoch, EpochRecord, Epochs, Recorded};
use crate::supervision::processes::{Event, Processes};

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
    /// The launch generation of the scheduler node's main process of its
    /// world (its own on that node): the process that holds the world's
    /// schedulers and session, and acts on every rank of the world.
    head: u64,
    /// The generation of modules it numbered: what a serving engine
    /// prepared, or what a target holds.
    generation: Option<u64>,
    /// The Storage Server scopes of its epoch's ranks on its host
    /// ([`crate::engine::control::host_scopes`]): the scopes its workers
    /// retain their tensors under, which every claim of the epoch there states.
    scopes: Vec<process::Scope>,
    /// The layout its replica lays its device memory out in
    /// ([`proto::config::EngineConfig::layout`]), which names its device regions: two engines of
    /// one layout share their regions and the segments in them.
    layout: proto::config::Layout,
    /// The epoch it executes: the one its replica served when the update began, or the epoch
    /// form's target.
    epoch: u64,
}

impl Node {
    /// The engine of launch generation `launch` on `host`, whose first rank
    /// is group rank `rank`, of the world whose scheduler node's main
    /// process is of launch generation `head`, whose epoch's ranks on
    /// `host` have the storage scopes `scopes`, whose replica lays its
    /// device memory out in `layout`, and that executes epoch `epoch`.
    pub(crate) fn new(
        host: String,
        rank: u32,
        launch: u64,
        head: u64,
        scopes: Vec<process::Scope>,
        layout: proto::config::Layout,
        epoch: u64,
    ) -> Self {
        Self {
            host,
            rank,
            launch,
            head,
            generation: None,
            scopes,
            layout,
            epoch,
        }
    }

    /// Whether this engine's device memory moves across the switch to or from the engines on the
    /// other side of it, which lay their memory out in `others`: when none of them lays its memory out as this one does,
    /// this one's regions are its own, so its drain gives their memory back (`shrink`) and its
    /// promotion grows into what the other side gives back. An engine whose layout one of
    /// `others` (their layouts) shares names the same regions, which hold the same segments for both: it keeps
    /// them through its drain and grows into nothing at its promotion.
    fn moves_across<'a>(
        &self,
        others: impl IntoIterator<Item = &'a proto::config::Layout>,
    ) -> bool {
        !others.into_iter().any(|other| *other == self.layout)
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
    /// What the previous epoch keeps through the trial
    /// (`update_config.previous_epoch`).
    pub(crate) keeps: PreviousEpoch,
    /// Whether the requests that migrate compute their state again
    /// (`in_flight = recompute`).
    pub(crate) recompute: bool,
    /// How long a draining engine steps its requests after its `Drain`
    /// (`update_config.drain_s`).
    pub(crate) drain: Duration,
    /// Whether the two sides process requests otherwise (other model files): each side that
    /// drains then runs every request it holds to its end, none migrating, whatever `drain`
    /// states (a complete drain).
    pub(crate) complete: bool,
    /// The layouts the target replicas lay their device memory out in, one per replica: a
    /// serving engine whose layout none of them shares drains with `shrink`
    /// ([`Node::moves_across`]).
    pub(crate) target_layouts: Vec<proto::config::Layout>,
    /// Epochs an earlier update dropped whose storage the next claim retires.
    pub(crate) retire: Vec<u64>,
    /// The serving engines of the replicas the update leaves as they are, which go on serving
    /// their epochs through it: a claim of an epoch they execute keeps their scopes, and the
    /// update retires an epoch only once none of them executes it.
    pub(crate) bystanders: Vec<Node>,
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
    serving: Recorded,
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
    /// How long the trial lasts: once it passed without a rollback, the update commits
    /// (`update_config.trial`).
    trial: Duration,
    /// When the switch began, and how long it took.
    switch_began: Option<Instant>,
    switched_in: Option<Duration>,
    /// When the trial began; `None` before the new served.
    trial_began: Option<Instant>,
    /// The operators the engines could not compare while they prepared.
    uncompared: Vec<String>,
    /// Why the claim that ended the update failed, when it did.
    claim_failed: Option<String>,
}

impl Update {
    /// A generation update: the serving engines `nodes` prepare `target`'s
    /// kernel and policy modules and switch to them.
    pub(crate) fn generation(
        number: u64,
        target: Epoch,
        epochs: Arc<Epochs>,
        serving: Recorded,
        group: String,
        nodes: Vec<Node>,
        trial: Duration,
    ) -> Self {
        Self::new(
            number,
            target,
            Form::Generation,
            epochs,
            serving,
            group,
            nodes,
            trial,
        )
    }

    /// An epoch update: `switch`'s target engines start beside the serving
    /// engines `nodes`, and service moves to them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn epoch(
        number: u64,
        target: Epoch,
        epochs: Arc<Epochs>,
        serving: Recorded,
        group: String,
        nodes: Vec<Node>,
        switch: EpochSwitch,
        trial: Duration,
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
        Self::new(number, target, form, epochs, serving, group, nodes, trial)
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        number: u64,
        target: Epoch,
        form: Form,
        epochs: Arc<Epochs>,
        serving: Recorded,
        group: String,
        nodes: Vec<Node>,
        trial: Duration,
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
            trial,
            switch_began: None,
            switched_in: None,
            trial_began: None,
            uncompared: Vec::new(),
            claim_failed: None,
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

    /// The update's form.
    pub(crate) fn change(&self) -> ChangeForm {
        match self.form {
            Form::Generation => ChangeForm::Generation,
            Form::Epoch(_) => ChangeForm::Epoch,
        }
    }

    /// The epoch the update's side executes in: the target epoch of the epoch
    /// form, the serving epoch of the generation form.
    pub(crate) fn target_epoch(&self) -> u64 {
        match &self.form {
            Form::Generation => self.serving.number(),
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

    /// Epochs this update dropped whose storage a later claim retires.
    pub(crate) fn retire(&self) -> &[u64] {
        match &self.form {
            Form::Generation => &[],
            Form::Epoch(trial) => &trial.switch.retire,
        }
    }

    /// The update as the operator reads it, without what its trial and transfers read.
    pub(crate) fn report(&self) -> UpdateReport {
        let (generation, epoch_form) = match &self.form {
            Form::Generation => (
                self.nodes.iter().filter_map(|node| node.generation).max(),
                None,
            ),
            Form::Epoch(trial) => (
                trial.switch.record.generation,
                Some(EpochSwitchReport {
                    build: trial.switch.record.config.executable.clone(),
                    engines: EngineCounts {
                        serving: self.nodes.len(),
                        target: trial.switch.launches.len(),
                    },
                    drain_s: trial.switch.drain.as_secs_f64(),
                    shrink: self
                        .nodes
                        .iter()
                        .any(|node| node.moves_across(&trial.switch.target_layouts)),
                    previous_epoch_kept: trial.kept,
                    host_refused: trial.host_refused.clone(),
                    moved_to_host_bytes: trial.moved_bytes,
                }),
            ),
        };
        UpdateReport {
            change: self.change(),
            update: self.number,
            phase: self.phase,
            epoch: self.target_epoch(),
            rollback: self.back,
            switch_ms: self
                .switched_in
                .map(|switched| u64::try_from(switched.as_millis()).unwrap_or(u64::MAX)),
            trial_s: self.trial_began.map(|began| began.elapsed().as_secs_f64()),
            deadline_s: self
                .ends()
                .map(|ends| ends.saturating_duration_since(Instant::now()).as_secs_f64()),
            failure: self.failure.clone(),
            uncompared: self.uncompared.clone(),
            claim_failed: self.claim_failed.clone(),
            generation,
            epoch_form,
            trial: None,
            transfers: None,
        }
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
    /// the same. Each host's scopes are those of `keeping`'s engines there:
    /// the scopes their workers retain tensors under, since a claim ends its
    /// epoch's retention of every scope it does not state.
    fn claim_everywhere(
        &self,
        processes: &mut Processes,
        epoch: u64,
        keeping: &[Node],
        retire: &[u64],
    ) -> Result<(), String> {
        self.claim_releasing(processes, epoch, keeping, retire, &[])
    }

    /// [`Self::claim_everywhere`], also ending `epoch`'s retention of the scopes of `releasing`'s
    /// engines on each host: the replicas the update moved off an epoch that still serves the
    /// others.
    fn claim_releasing(
        &self,
        processes: &mut Processes,
        epoch: u64,
        keeping: &[Node],
        retire: &[u64],
        releasing: &[Node],
    ) -> Result<(), String> {
        let scopes_on = |nodes: &[Node], host: &str| {
            let mut scopes: Vec<process::Scope> = Vec::new();
            for node in nodes.iter().filter(|node| node.host == host) {
                for scope in &node.scopes {
                    if !scopes.contains(scope) {
                        scopes.push(scope.clone());
                    }
                }
            }
            scopes
        };
        for host in self.hosts() {
            let scopes = scopes_on(keeping, host);
            let released: Vec<process::Scope> = scopes_on(releasing, host)
                .into_iter()
                .filter(|scope| !scopes.contains(scope))
                .collect();
            processes.claim(host, epoch, &scopes, retire, &released)?;
        }
        Ok(())
    }

    /// The epochs the update's serving engines execute, each once, ascending.
    fn previous_epochs(&self) -> BTreeSet<u64> {
        self.nodes.iter().map(|node| node.epoch).collect()
    }

    /// The records of [`Self::previous_epochs`], which a generation update moves together; the
    /// serving epoch's when no engine states one the daemon records.
    fn previous_records(&self) -> Vec<Recorded> {
        let recorded: Vec<Recorded> = self
            .previous_epochs()
            .into_iter()
            .filter_map(|epoch| self.epochs.recorded(epoch))
            .collect();
        if recorded.is_empty() {
            vec![self.serving]
        } else {
            recorded
        }
    }

    /// The engines that execute `epoch` and go on serving through the update: the serving
    /// engines of the replicas it leaves as they are.
    fn bystanders_of(&self, epoch: u64) -> Vec<Node> {
        match &self.form {
            Form::Generation => Vec::new(),
            Form::Epoch(trial) => trial
                .switch
                .bystanders
                .iter()
                .filter(|node| node.epoch == epoch)
                .cloned()
                .collect(),
        }
    }

    /// Publishes the committed `epoch` at every node that claimed it
    /// ([`Self::hosts`]), and only there: a node neither epoch runs on was
    /// never asked to claim it and holds no storage of it. A node that
    /// refuses keeps the epoch's outputs private to the epoch; the commit
    /// stands, and the log names the node and its reason.
    fn publish_everywhere(&self, processes: &mut Processes, epoch: u64) {
        for host in self.hosts() {
            if let Err(reason) = processes.publish(host, &[epoch]) {
                tracing::error!(
                    epoch,
                    host,
                    %reason,
                    "a node that claimed the committed epoch did not publish it"
                );
            }
        }
    }

    /// The hosts either epoch runs on, each once: every node the update
    /// claims at ([`Self::claim_everywhere`]), so every node whose Storage
    /// Server holds a claim of the epoch form's target.
    fn hosts(&self) -> BTreeSet<&str> {
        let mut hosts: BTreeSet<&str> = self.nodes.iter().map(|node| node.host.as_str()).collect();
        if let Form::Epoch(trial) = &self.form {
            hosts.extend(trial.targets.iter().map(|node| node.host.as_str()));
        }
        hosts
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
    /// modules, where its kernel probes lie, and the options its graphs are
    /// captured for.
    fn prepare_generation(&mut self, processes: &mut Processes) {
        let modules = &self.target.modules;
        let prepare = command(command::Kind::Prepare(process::Prepare {
            kernels: modules
                .kernels
                .iter()
                .map(|(slot, package)| proto::module::kernel_to_wire(slot, package))
                .collect(),
            policy: Some((&modules.policy).into()),
            kernel_instance: self.target.engine.profiling.kernel_instance,
            fence_modules: self
                .target
                .engine
                .profiling
                .fence_modules
                .clone()
                .map(|patterns| process::FenceModules { patterns }),
            graphs: Some((&self.target.engine.graph_options()).into()),
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
        trial.switch.record.status = EpochState::Preparing;
        self.epochs
            .record(trial.switch.epoch, trial.switch.record.clone());

        let launches = trial.switch.launches.clone();
        for (launch, inherit) in launches {
            let spawn = Spawn {
                host: launch.host.clone(),
                group: self.group.clone(),
                rank: launch.rank,
                generation: launch.engine,
                // The node's daemon resolves the engine package the launch
                // names and writes the launch file for this process.
                inherit,
                launch: Vec::new(),
            };
            if let Err(reason) = processes.spawn(spawn, &launch) {
                return self.abandon(processes, Failure::new(FailureKind::SpawnFailed, reason));
            }
            if let Form::Epoch(trial) = &mut self.form {
                trial.targets.push(Node::new(
                    launch.host.clone(),
                    launch.rank,
                    launch.engine,
                    launch
                        .world
                        .as_ref()
                        .map_or(launch.engine, |world| world.scheduler),
                    crate::engine::control::host_scopes(&launch.engine_config, &launch.host),
                    launch.engine_config.layout(),
                    launch.epoch,
                ));
            }
        }

        let Form::Epoch(trial) = &self.form else {
            return;
        };
        let (epoch, targets) = (trial.switch.epoch, trial.targets.clone());
        let retire = trial.switch.retire.clone();
        let mut serving = self.nodes.clone();
        serving.extend(trial.switch.bystanders.iter().cloned());
        if let Err(reason) = self.claim_everywhere(processes, epoch, &targets, &retire) {
            return self.abandon(processes, Failure::new(FailureKind::ClaimFailed, reason));
        }
        // Every node retired them: none is left for a later claim. Their engines stopped when
        // their update dropped them, and their storage and its quotas' reservations ended now:
        // the engines that serve through this update grow into what that freed, and the targets
        // starting beside them divide what the quotas leave.
        if let Form::Epoch(trial) = &mut self.form {
            trial.switch.retire.clear();
        }
        if !retire.is_empty() {
            grow_on(processes, self.hosts(), &serving);
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
                    status,
                    reason,
                } => self.exit(processes, &host, generation, &status, &reason),
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
                self.abandon(processes, Failure::new(FailureKind::BuildMismatch, reason));
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
                self.abandon(processes, Failure::new(FailureKind::PrepareFailed, reason));
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
                self.roll_back(
                    processes,
                    Failure::new(FailureKind::RolledBack, reason),
                    Some(node),
                );
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
            // A target's engines prepare by starting and parking, so whatever a target reports
            // failing before it parked is the update's failed preparation, with the engine's
            // reason: the one kind the leader states when the same target does not join, which
            // of the two arrives first.
            (Phase::Preparing, _, _) => {
                if let Some(failure) = Failure::of_report(&report) {
                    let prepare = Failure {
                        kind: FailureKind::PrepareFailed,
                        ..failure
                    };
                    self.abandon(processes, prepare);
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
                        self.roll_back(
                            processes,
                            Failure::new(FailureKind::StaleModules, reason),
                            Some(target),
                        );
                    }
                    _ => {
                        // New requests go to the target from its first
                        // `Serving`: the serving epoch still serves what it
                        // has, so none waits.
                        if trial.serving.is_empty() {
                            processes.serve(trial.switch.epoch);
                        }
                        trial.serving.insert(target);
                    }
                }
            }
            (Phase::Switching, true, Some(report::Kind::Drained(_))) => {
                trial.target_left.insert(target);
            }
            // On the way back the targets drain beside the previous epoch, which grows into what
            // they give back on the host the release names.
            (Phase::Switching, true, Some(report::Kind::Shrunk(shrunk))) => {
                let growing = growing(&self.nodes, trial.targets.iter().map(|node| &node.layout));
                grow_on(processes, [shrunk.host.as_str()], &growing);
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
                if drained.epoch == self.nodes[node].epoch =>
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
            // A serving engine draining across layouts gave device memory back on the host the
            // release names: the targets with ranks there grow into it, one release at a time.
            (Phase::Switching | Phase::Trial, false, Some(report::Kind::Shrunk(shrunk))) => {
                let growing = growing(&trial.targets, self.nodes.iter().map(|node| &node.layout));
                grow_on(processes, [shrunk.host.as_str()], &growing);
            }
            _ => {}
        }
    }

    /// The previous epoch's engine `node` serves again for the rollback: new
    /// requests return to the previous epoch from the first such engine.
    fn previous_serves(&mut self, processes: &mut Processes, node: usize) {
        // Each epoch the update's serving engines execute serves again from its first such engine.
        let epoch = self.nodes[node].epoch;
        let first = !self
            .acknowledged
            .iter()
            .any(|acknowledged| self.nodes[*acknowledged].epoch == epoch);
        if first && matches!(self.form, Form::Epoch(_)) {
            processes.serve(epoch);
        }
        self.acknowledged.insert(node);
    }

    /// An engine of launch generation `launch` exited as `status` states, having stated `reason`
    /// of its end (its exit reason, which its node read bounded): the update's failure, when the
    /// exit is one, names both.
    fn exit(
        &mut self,
        processes: &mut Processes,
        host: &str,
        launch: u64,
        status: &str,
        reason: &str,
    ) {
        let Some((index, target)) = self.node_of(host, launch) else {
            return;
        };
        let stated = if reason.is_empty() {
            String::new()
        } else {
            format!(": {reason}")
        };
        let reason =
            format!("the engine of launch generation {launch} on {host} exited ({status}){stated}");
        let exited = Failure::new(FailureKind::ProcessExit, reason);

        let Form::Epoch(trial) = &mut self.form else {
            // The engine that held both generations is gone: its failure
            // domain is replaced by a standby or a relaunch of the modules
            // the epoch's record names, which are the kept ones until a
            // commit.
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
            (Phase::Preparing, _) => self.abandon(processes, exited),
            (Phase::Switching, false) => {
                // Nothing is left to drain there: its requests end as
                // migratable with its loss, and the target takes them.
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
            _ => {}
        }
    }

    /// Asks `node`'s host to keep the previous epoch's weights. A node that
    /// refuses, or fails to keep them, keeps them on its GPUs: nothing else
    /// of the trial depends on it, and the status says why.
    fn keep_weights(&mut self, processes: &mut Processes, node: usize) {
        let serving = self.nodes[node].epoch;
        // An epoch that serves other replicas on the host keeps its storage there: what it
        // keeps alone would release theirs.
        let shared = self
            .bystanders_of(serving)
            .iter()
            .any(|bystander| bystander.host == self.nodes[node].host);
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
        if shared {
            trial.kept = PreviousEpoch::Parked;
            trial.host_refused.push(format!(
                "{host}: epoch {serving} serves other replicas on the host, which keep its storage"
            ));
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
        // A relaunched scheduler node is its world's at its new launch generation: what the
        // update sends the world from now on reaches it there.
        for other in &mut self.nodes {
            if other.head == previous {
                other.head = launch;
            }
        }
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
            Form::Generation => {
                for epoch in self.previous_records() {
                    self.epochs.committed(epoch);
                }
            }
            Form::Epoch(trial) => {
                trial.switch.record.status = EpochState::Serving;
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
                // The target's outputs are every epoch's from now on: each
                // node that claimed it publishes it before any node retires
                // the previous epoch.
                let (epoch, targets) = (trial.switch.epoch, trial.targets.clone());
                self.publish_everywhere(processes, epoch);
                for node in &self.nodes {
                    processes.stop(&self.group, Some(node.launch));
                }
                // The previous epochs' storage ends on every node while the target keeps its
                // own: an epoch no other replica executes is retired, and one that serves other
                // replicas keeps their scopes and ends the moved replicas'.
                let (retired, shared): (Vec<u64>, Vec<u64>) = self
                    .previous_epochs()
                    .into_iter()
                    .partition(|previous| self.bystanders_of(*previous).is_empty());
                let mut failed = self
                    .claim_everywhere(processes, epoch, &targets, &retired)
                    .err();
                for previous in shared {
                    let moved: Vec<Node> = self
                        .nodes
                        .iter()
                        .filter(|node| node.epoch == previous)
                        .cloned()
                        .collect();
                    let staying = self.bystanders_of(previous);
                    let released = self
                        .claim_releasing(processes, previous, &staying, &[], &moved)
                        .err();
                    failed = failed.or(released);
                }
                self.claim_failed = failed;
                // The stops returned once every node reaped the previous epoch's engines, and
                // the claims ended its storage and its quotas' reservations: the targets grow
                // into what that freed on every host it ran on. Its weights, graph pools and
                // process storage left those quotas whatever the layouts, so every target world
                // with an engine there grows, not only those its drain shrank for.
                let released: BTreeSet<&str> =
                    self.nodes.iter().map(|node| node.host.as_str()).collect();
                grow_on(processes, released, &targets);
            }
        }
    }

    /// The trial rolls back (a fault, a threshold): the kept side serves again and the new is
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

    /// When the trial ends and the update commits by itself, while it stands on trial.
    pub(crate) fn ends(&self) -> Option<Instant> {
        self.trial_began
            .filter(|_| self.phase == Phase::Trial)
            .map(|began| began + self.trial)
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
                trial.switch.record.status = EpochState::Released;
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
                trial.switch.record.status = EpochState::Released;
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
                            promote(
                                trial.switch.recompute,
                                node.moves_across(&trial.switch.target_layouts),
                            ),
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
            promote(
                trial.switch.recompute,
                node.moves_across(&trial.switch.target_layouts),
            ),
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
    #[expect(
        clippy::expect_used,
        reason = "a generation update switches once every serving engine acknowledged its \
                  preparation, which records the generation it prepared"
    )]
    fn switch(&mut self, processes: &mut Processes) {
        self.enter(Phase::Switching);
        self.switch_began = Some(Instant::now());
        match &self.form {
            Form::Generation => {
                for node in &self.nodes {
                    let switch = command::Kind::Switch(process::Switch {
                        generation: node
                            .generation
                            .expect("every serving engine prepared its generation"),
                    });
                    processes.command(&node.host, node.launch, command(switch));
                }
            }
            Form::Epoch(trial) => {
                for target in &trial.targets {
                    let grow = target.moves_across(self.nodes.iter().map(|node| &node.layout));
                    processes.command(
                        &target.host,
                        target.launch,
                        promote(trial.switch.recompute, grow),
                    );
                }
            }
        }
    }

    /// The forward switch: once every target serves the serving engines are
    /// drained, and once every engine acknowledged the trial begins.
    fn progress_switch(&mut self, processes: &mut Processes) {
        if let Form::Epoch(trial) = &mut self.form {
            if !trial.draining && trial.serving.len() == trial.targets.len() {
                trial.draining = true;
                for (index, node) in self.nodes.iter().enumerate() {
                    if !trial.drained.contains(&index) {
                        let shrink = node.moves_across(&trial.switch.target_layouts);
                        let drain = drain(node.epoch, &trial.switch, shrink);
                        processes.command(&node.host, node.launch, drain);
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
                for epoch in self.previous_records() {
                    self.epochs.switched(epoch, self.target.modules.clone());
                }
            }
            Form::Epoch(trial) => {
                trial.switch.record.status = EpochState::Trial;
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
                    for epoch in self.previous_records() {
                        self.epochs.rolled_back(epoch);
                    }
                }
            }
            Form::Epoch(trial) => {
                if !all_serve {
                    return;
                }
                if !trial.targets_draining {
                    trial.targets_draining = true;
                    for (index, target) in trial.targets.iter().enumerate() {
                        if !trial.target_left.contains(&index) {
                            let shrink =
                                target.moves_across(self.nodes.iter().map(|node| &node.layout));
                            let drain = drain(trial.switch.epoch, &trial.switch, shrink);
                            processes.command(&target.host, target.launch, drain);
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
                let released: BTreeSet<String> = trial
                    .targets
                    .iter()
                    .map(|target| target.host.clone())
                    .collect();
                // Each previous epoch keeps every scope it executes: the update's serving
                // engines' and the replicas' it left as they were.
                let mut failed = None;
                for previous in self.previous_epochs() {
                    let mut keeping: Vec<Node> = self
                        .nodes
                        .iter()
                        .filter(|node| node.epoch == previous)
                        .cloned()
                        .collect();
                    keeping.extend(self.bystanders_of(previous));
                    let claimed = self
                        .claim_everywhere(processes, previous, &keeping, &retire)
                        .err();
                    failed = failed.or(claimed);
                }
                self.claim_failed = failed;
                // The targets are reaped and their epoch's storage retired on every host they
                // ran on: the previous epoch, which serves again, grows into what that freed.
                grow_on(processes, released.iter().map(String::as_str), &self.nodes);
            }
        }
        self.enter(Phase::RolledBack);
    }
}

/// One command of `kind`.
fn command(kind: command::Kind) -> process::Command {
    process::Command { kind: Some(kind) }
}

/// Sends `Grow` to the scheduler node of each world of `engines` with an engine on one of `hosts`,
/// once per world: device memory came free in the quotas of GPUs on those hosts, given back by an
/// engine draining across layouts (`Shrunk`) or released with an epoch's storage, and the
/// scheduler node grows its regions into it on every rank it holds, the peers' included
/// (`specs/updates.md` §3.7), creating the segments its regions number as far as each rank's quota
/// and device hold them. A world with no engine on those hosts shares no quota with them.
fn grow_on<'a>(
    processes: &mut Processes,
    hosts: impl IntoIterator<Item = &'a str>,
    engines: &[Node],
) {
    let hosts: BTreeSet<&str> = hosts.into_iter().collect();
    let mut heads: Vec<u64> = Vec::new();
    for engine in engines
        .iter()
        .filter(|engine| hosts.contains(engine.host.as_str()))
    {
        if !heads.contains(&engine.head) {
            heads.push(engine.head);
        }
    }
    for head in heads {
        let Some(scheduler) = engines.iter().find(|engine| engine.launch == head) else {
            tracing::warn!(head, "a world's scheduler node is no engine of the update");
            continue;
        };
        processes.command(
            &scheduler.host,
            head,
            command(command::Kind::Grow(process::Grow {})),
        );
    }
}

/// The promotion of an epoch switch: whether the requests that migrate onto
/// the engine compute their state again, and whether it grows its regions as
/// the engine draining beside it gives its memory back.
pub(crate) fn promote(recompute: bool, grow: bool) -> process::Command {
    command(command::Kind::Promote(process::Promote { recompute, grow }))
}

/// The drain of an engine executing `epoch` in `switch`: the deadline from
/// now, or none for a switch between sides that process requests otherwise
/// (`complete`), and `shrink` when the engine's device memory moves to the
/// other side of the switch ([`Node::moves_across`]).
fn drain(epoch: u64, switch: &EpochSwitch, shrink: bool) -> process::Command {
    command(command::Kind::Drain(process::Drain {
        epoch,
        deadline_ms: u64::try_from(switch.drain.as_millis()).unwrap_or(u64::MAX),
        shrink,
        complete: switch.complete,
    }))
}

/// The engines of `side` that grow into the memory the other side's engines, which lay their
/// memory out in `other`, give back: those whose device memory moves across the switch
/// ([`Node::moves_across`]).
fn growing<'a>(
    side: &[Node],
    other: impl IntoIterator<Item = &'a proto::config::Layout> + Clone,
) -> Vec<Node> {
    side.iter()
        .filter(|engine| engine.moves_across(other.clone()))
        .cloned()
        .collect()
}
