//! The update's phase machine against the fake engines at the process
//! interface (`testkit::process`): every transition, failure and rollback
//! path of both forms, as the commands the engines were sent, the requests
//! the nodes were asked, the epoch records and the status report state them.

use std::sync::Arc;
use std::time::{Duration, Instant};

use proto::admin::{
    EpochState, Failure, FailureKind, Phase, PreviousEpoch, RollbackConfig, Threshold,
};
use proto::process::{self, command};
use proto::report::SchedulerStats;
use testkit::process::Refusals;

use crate::config::{Config, UpdateConfig};
use crate::epoch::{Epoch, EpochRecord, Epochs, PackageRevision, Recorded};
use crate::observe::Observer;
use crate::supervision::processes::Processes;
use crate::test_support::{Engines, Seen};
use crate::updates::entry::{Serving, Sides};
use crate::updates::trial::{Baseline, Trial};
use crate::updates::update::{EpochSwitch, Node, Update};
use crate::updates::{Transition, Updater};

/// A trial long enough that no test reaches its end, so only a decision ends it.
const LONG: Duration = Duration::from_secs(3600);

/// The epoch that serves when a test begins, and the epoch form's target.
const SERVING: u64 = 1;
const TARGET: u64 = 2;

/// The launch generations of the serving engine, of a second serving engine
/// and of the epoch form's targets.
const PREVIOUS: u64 = 4;
const SECOND: u64 = 5;
const SPAWNED: u64 = 9;
const SPAWNED_PEER: u64 = 10;

/// The build the fake engines greet with.
const BUILD: &str = "2.0.0+target";

const HOST: &str = "rank-0";
const OTHER: &str = "rank-1";

/// A deployment whose engines and nodes are fakes: the updater under test,
/// its epoch records and the fakes it commands.
struct Deployment {
    epochs: Arc<Epochs>,
    updater: Updater,
    engines: Engines,
    transitions: tokio::sync::mpsc::UnboundedReceiver<Transition>,
    /// What the deployment serves, and what the update under test moves it to.
    serving: Epoch,
    target: Epoch,
    /// The serving epoch's record.
    recorded: Recorded,
}

impl Deployment {
    /// A deployment that serves epoch [`SERVING`] on one engine, whose nodes
    /// refuse what `refusals` names.
    fn new(refusals: Refusals) -> Self {
        let epochs = Arc::new(Epochs::default());
        let (mut engines, link) = Engines::link(refusals);
        engines.serve(PREVIOUS);
        let processes = Processes::new(
            Arc::clone(&epochs),
            "deployment".to_owned(),
            link,
            crate::test_support::recorder(),
        );
        let updater = Updater::new(Arc::clone(&epochs), processes);
        let transitions = updater.transitions().expect("the log is read once");

        let serving = Epoch::of(Config::default()).expect("the configuration resolves");
        let recorded = epochs.record(
            SERVING,
            EpochRecord {
                config: serving.engine.clone(),
                engine: serving.modules.engine.clone(),
                kernels: serving.modules.kernels.clone(),
                policy: serving.modules.policy.clone(),
                processing: serving.modules.processing.clone(),
                previous: None,
                status: EpochState::Serving,
                generation: Some(1),
            },
        );
        // The target runs another revision of the scheduling policy.
        let mut target = serving.clone();
        target.modules.policy = PackageRevision {
            revision: serving.modules.policy.revision + 1,
            ..serving.modules.policy.clone()
        };
        Self {
            epochs,
            updater,
            engines,
            transitions,
            serving,
            target,
            recorded,
        }
    }

    /// The serving engines a generation update moves: one, or two.
    fn nodes(&mut self, engines: usize) -> Vec<Node> {
        let mut nodes = vec![Node::new(
            HOST.to_owned(),
            0,
            PREVIOUS,
            PREVIOUS,
            scopes(0..1),
            self.layout(),
            SERVING,
        )];
        if engines == 2 {
            self.engines.serve(SECOND);
            nodes.push(Node::new(
                HOST.to_owned(),
                1,
                SECOND,
                SECOND,
                scopes(1..2),
                self.layout(),
                SERVING,
            ));
        }
        nodes
    }

    /// The layout the serving engines and, unless a test moves them across layouts
    /// ([`Self::across_layouts`]), the targets lay their device memory out in.
    fn layout(&self) -> proto::config::Layout {
        self.target.engine.layout()
    }

    /// Moves `switch`'s targets to another layout than the serving engines' (another slot
    /// count): their regions are their own, so the serving engines give their memory back as
    /// they drain and the targets grow into it.
    fn across_layouts(&self, switch: &mut EpochSwitch) {
        for (launch, _) in &mut switch.launches {
            launch.engine_config.max_running_requests = launch
                .engine_config
                .max_running_requests
                .checked_add(1)
                .expect("a slot count below the maximum");
        }
        switch.target_layouts = switch
            .launches
            .iter()
            .map(|(launch, _)| launch.engine_config.layout())
            .collect();
    }

    /// A generation update numbered `number` over `engines` serving engines.
    fn generation(&mut self, number: u64, engines: usize, trial: Duration) -> Update {
        let nodes = self.nodes(engines);
        Update::generation(
            number,
            self.target.clone(),
            Arc::clone(&self.epochs),
            self.recorded,
            "engine".to_owned(),
            nodes,
            trial,
        )
    }

    /// An epoch update numbered `number` from the one serving engine to one
    /// target engine on the same node, whose previous epoch keeps `keeps`.
    fn epoch(&self, number: u64, keeps: PreviousEpoch, trial: Duration) -> Update {
        let mut switch = self.switch(vec![(target_launch(HOST, 0, SPAWNED), Vec::new())]);
        switch.keeps = keeps;
        let nodes = vec![Node::new(
            HOST.to_owned(),
            0,
            PREVIOUS,
            PREVIOUS,
            scopes(0..1),
            self.layout(),
            SERVING,
        )];
        self.epoch_over(number, nodes, switch, trial)
    }

    /// The epoch form's target: epoch [`TARGET`], started from `launches`,
    /// with the previous epoch parked, the default drain and the layout
    /// unchanged.
    fn switch(&self, launches: Vec<(proto::config::EngineArgs, Vec<u64>)>) -> EpochSwitch {
        let mut record = self.epochs.get(SERVING).expect("the serving record");
        record.generation = None;
        let launches = launches
            .into_iter()
            .map(|(mut launch, inherit)| {
                launch.engine_config = self.target.engine.clone();
                launch.modules = self.target.modules.clone();
                (launch, inherit)
            })
            .collect();
        EpochSwitch {
            epoch: TARGET,
            record,
            launches,
            keeps: PreviousEpoch::Parked,
            recompute: false,
            drain: Duration::from_secs_f64(UpdateConfig::default().drain_s),
            complete: false,
            target_layouts: vec![self.layout()],
            retire: Vec::new(),
            bystanders: Vec::new(),
        }
    }

    /// An epoch update numbered `number` from the serving engines `nodes` to
    /// `switch`'s targets.
    fn epoch_over(
        &self,
        number: u64,
        nodes: Vec<Node>,
        switch: EpochSwitch,
        trial: Duration,
    ) -> Update {
        Update::epoch(
            number,
            self.target.clone(),
            Arc::clone(&self.epochs),
            self.recorded,
            "engine".to_owned(),
            nodes,
            switch,
            trial,
        )
    }

    /// The deployment's `count` serving replicas on [`HOST`], as an epoch change finds them:
    /// replica `i` is one engine of launch generation [`PREVIOUS`]` + i` holding group rank `i`,
    /// serving the serving epoch's launch.
    fn serving_replicas(&mut self, count: u32) -> Vec<Serving> {
        (0..count)
            .map(|index| {
                let generation = PREVIOUS + u64::from(index);
                if generation != PREVIOUS {
                    self.engines.serve(generation);
                }
                let mut launch = target_launch(HOST, index, generation);
                launch.engine_config = self.serving.engine.clone();
                launch.modules = self.serving.modules.clone();
                launch.epoch = SERVING;
                launch.predecessor = None;
                launch.replica = index;
                Serving {
                    launch: Some(launch),
                    nodes: vec![Node::new(
                        HOST.to_owned(),
                        index,
                        generation,
                        generation,
                        scopes(index..index + 1),
                        self.layout(),
                        SERVING,
                    )],
                }
            })
            .collect()
    }

    /// The launches of `count` target replicas on [`HOST`]: target `j` is one engine of launch
    /// generation [`SPAWNED`]` + j` holding group rank `j`, running the target's modules.
    fn target_replicas(&self, count: u32) -> Vec<proto::config::EngineArgs> {
        (0..count)
            .map(|index| {
                let mut launch = target_launch(HOST, index, SPAWNED + u64::from(index));
                launch.engine_config = self.target.engine.clone();
                launch.modules = self.target.modules.clone();
                launch.replica = index;
                launch
            })
            .collect()
    }

    /// An epoch update numbered `number` from the `serving` replicas to the `targets`, over the
    /// sides the update's entry gives the change ([`Sides::of`]), whose drains are complete when
    /// `complete` (the target's model files process requests otherwise).
    fn change(
        &self,
        number: u64,
        serving: Vec<Serving>,
        targets: Vec<proto::config::EngineArgs>,
        complete: bool,
    ) -> Update {
        let sides = Sides::of(serving, targets, &self.target.modules);
        let mut switch = self.switch(Vec::new());
        switch.complete = complete;
        switch.launches = sides
            .targets
            .into_iter()
            .map(|(_, launch)| (launch, Vec::new()))
            .collect();
        switch.target_layouts = switch
            .launches
            .iter()
            .map(|(launch, _)| launch.engine_config.layout())
            .collect();
        switch.bystanders = sides.bystanders;
        self.epoch_over(number, sides.draining, switch, LONG)
    }

    /// Sends what the engine of launch generation `launch` withheld, and runs
    /// the update on.
    fn release(&mut self, launch: u64) {
        for report in self.engines.engine(launch).release() {
            self.engines.report(launch, report);
        }
        self.run();
    }

    /// Applies `update` and runs it as far as the fakes take it; returns the update as it
    /// stood when it started, as the admin API serializes it.
    fn apply(&mut self, update: Update) -> serde_json::Value {
        let report = self.updater.apply(update).expect("no update is in flight");
        self.run();
        serde_json::to_value(report).expect("a report serializes")
    }

    /// Runs the update as far as the fakes take it.
    fn run(&mut self) {
        self.engines.run(&self.updater, BUILD);
    }

    /// The phases updates entered since the last call, in order.
    fn entered(&mut self) -> Vec<&'static str> {
        let mut entered = Vec::new();
        while let Ok(transition) = self.transitions.try_recv() {
            entered.push(transition.phase.name());
        }
        entered
    }

    /// The update as the status reports it, serialized as the admin API does.
    fn status(&self) -> serde_json::Value {
        serde_json::to_value(self.updater.status().expect("an update ran"))
            .expect("a report serializes")
    }

    /// The record of epoch `epoch`.
    fn record(&self, epoch: u64) -> EpochRecord {
        self.epochs.get(epoch).expect("the epoch's record")
    }
}

/// The storage scopes of the deployment's ranks `ranks` of its group `model`.
fn scopes(ranks: std::ops::Range<u32>) -> Vec<process::Scope> {
    ranks
        .map(|rank| process::Scope {
            group: "model".to_owned(),
            rank,
        })
        .collect()
}

/// The launch file of the target engine `launch` on `host`, whose first rank
/// is group rank `rank`: it starts parked, succeeding the serving epoch.
fn target_launch(host: &str, rank: u32, launch: u64) -> proto::config::EngineArgs {
    let options = Config::default();
    let mut args = proto::config::EngineArgs::new(
        options.engine_config().expect("the configuration resolves"),
        crate::epoch::modules_of(&options),
    );
    args.epoch = TARGET;
    args.host = host.to_owned();
    args.rank = rank;
    args.engine = launch;
    args.predecessor = Some(proto::request::wire::Predecessor { epoch: SERVING });
    args
}

/// The kinds of `commands`, by name, for an order a test states.
fn names(commands: &[command::Kind]) -> Vec<&'static str> {
    commands
        .iter()
        .map(|kind| match kind {
            command::Kind::Prepare(_) => "prepare",
            command::Kind::Switch(_) => "switch",
            command::Kind::Commit(_) => "commit",
            command::Kind::Rollback(_) => "rollback",
            command::Kind::Drain(_) => "drain",
            command::Kind::Promote(_) => "promote",
            command::Kind::Pause(_) => "pause",
            command::Kind::Grow(_) => "grow",
            command::Kind::Room(_) => "room",
        })
        .collect()
}

fn decision(reason: &str) -> Failure {
    Failure::new(FailureKind::Decision, reason.to_owned())
}

// --- the generation form ------------------------------------------------------------------------

/// A generation update prepares, switches and serves on trial; a commit
/// records the new modules as the epoch's and tells the engine to release
/// the kept generation.
#[test]
fn a_generation_update_serves_on_trial_and_a_commit_keeps_it() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 1, LONG);
    let report = deployment.apply(update);
    assert_eq!(report["change"], "generation");
    assert_eq!(report["phase"], "preparing");

    assert_eq!(deployment.entered(), ["preparing", "switching", "trial"]);
    assert_eq!(
        names(&deployment.engines.received(PREVIOUS)),
        ["prepare", "switch"]
    );
    // The switch names the generation the engine numbered when it prepared.
    assert_eq!(
        deployment.engines.received(PREVIOUS)[1],
        command::Kind::Switch(process::Switch { generation: 2 })
    );
    let status = deployment.status();
    assert_eq!(status["phase"], "trial");
    assert_eq!(status["generation"], 2);

    // On trial the record names the new modules and keeps the previous ones.
    let record = deployment.record(SERVING);
    assert_eq!(record.status, EpochState::Trial);
    assert_eq!(record.policy, deployment.target.modules.policy);
    assert_eq!(
        record.previous.map(|previous| previous.policy),
        Some(deployment.serving.modules.policy.clone())
    );

    assert!(deployment.updater.commit().is_some());
    deployment.run();
    assert_eq!(deployment.entered(), ["committed"]);
    assert_eq!(
        names(&deployment.engines.received(PREVIOUS)),
        ["prepare", "switch", "commit"]
    );
    let record = deployment.record(SERVING);
    assert_eq!(record.status, EpochState::Serving);
    assert_eq!(record.policy, deployment.target.modules.policy);
    assert_eq!(record.previous, None);
    assert_eq!(deployment.updater.in_flight(), None);
    assert_eq!(deployment.status()["phase"], "committed");
}

/// A preparation that fails abandons the update: nothing that serves
/// changed, the record is the serving one's, and the target stays available.
#[test]
fn a_failed_preparation_abandons_the_update_and_nothing_changed() {
    let mut deployment = Deployment::new(Refusals::default());
    deployment
        .engines
        .engine(PREVIOUS)
        .refuse_prepare("the package has no module for this key");
    let update = deployment.generation(1, 1, LONG);
    deployment.apply(update);

    assert_eq!(deployment.entered(), ["preparing", "abandoned"]);
    assert_eq!(deployment.engines.engine(PREVIOUS).generation(), 1);
    let record = deployment.record(SERVING);
    assert_eq!(record.status, EpochState::Serving);
    assert_eq!(record.policy, deployment.serving.modules.policy);
    let status = deployment.status();
    assert_eq!(status["phase"], "abandoned");
    assert_eq!(status["failure"]["kind"], "prepare_failed");
    assert_eq!(
        deployment.updater.unavailable(&deployment.target),
        None,
        "an abandoned update marks nothing unavailable"
    );
}

/// A target whose start fails before it parks abandons the epoch update as a
/// failed preparation with the engine's reason, the kind the leader states
/// when the same target does not join: the serving epoch serves on, and grows into what the
/// target's retired storage freed.
#[test]
fn a_target_that_does_not_start_abandons_the_update_as_a_failed_preparation() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");

    let reason = "the GPU's quota cannot hold the rank's graph pool";
    deployment.engines.report(
        SPAWNED,
        process::Report {
            kind: Some(process::report::Kind::Fault(process::Fault {
                kind: process::FaultKind::Start.into(),
                reason: reason.to_owned(),
                device: None,
            })),
        },
    );
    deployment.run();

    let status = deployment.status();
    assert_eq!(status["phase"], "abandoned");
    assert_eq!(status["failure"]["kind"], "prepare_failed");
    assert_eq!(status["failure"]["reason"], reason);
    assert_eq!(
        names(&deployment.engines.received(PREVIOUS)),
        ["grow"],
        "the serving engine is sent nothing but its growth"
    );
}

/// With two serving engines the switch begins only once both prepared, and
/// one that cannot prepare abandons the update for both.
#[test]
fn every_engine_acknowledges_before_the_phase_moves() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 2, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    // Only the first engine prepares.
    let prepared = deployment
        .engines
        .engine(PREVIOUS)
        .command(process::Command {
            kind: Some(command::Kind::Prepare(process::Prepare::default())),
        });
    deployment.engines.report(PREVIOUS, prepared[0].clone());
    deployment.updater.advance();
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Preparing)));

    deployment.run();
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Trial)));
    assert_eq!(
        names(&deployment.engines.received(SECOND)),
        ["prepare", "switch"]
    );

    // A second deployment whose second engine refuses.
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 2, LONG);
    deployment
        .engines
        .engine(SECOND)
        .refuse_prepare("no module");
    deployment.apply(update);
    assert_eq!(deployment.status()["phase"], "abandoned");
    // The engine that prepared is told to discard what it prepared.
    assert_eq!(
        names(&deployment.engines.received(PREVIOUS)),
        ["prepare", "rollback"]
    );
    assert_eq!(deployment.engines.engine(PREVIOUS).generation(), 1);
}

/// A fault on trial rolls the update back: the engine that went back by
/// itself is not commanded again, the record returns to the previous modules
/// and the target is unavailable with the first failure.
#[test]
fn a_fault_on_trial_rolls_a_generation_update_back() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 1, LONG);
    deployment.apply(update);
    deployment.entered();

    let rolled = deployment
        .engines
        .engine(PREVIOUS)
        .rolled_back("non-finite logits for request 7");
    deployment.engines.report(PREVIOUS, rolled);
    deployment.run();

    assert_eq!(deployment.entered(), ["switching", "rolled_back"]);
    assert_eq!(
        names(&deployment.engines.received(PREVIOUS)),
        ["prepare", "switch"],
        "the engine rolled back by itself"
    );
    let record = deployment.record(SERVING);
    assert_eq!(record.status, EpochState::Serving);
    assert_eq!(record.policy, deployment.serving.modules.policy);
    assert_eq!(record.previous, None);
    let mark = deployment
        .updater
        .unavailable(&deployment.target)
        .expect("a rollback marks the target");
    assert_eq!(mark.reason, "non-finite logits for request 7");
}

/// A stalled job an engine reports on trial rolls every engine back.
#[test]
fn a_stalled_job_on_trial_rolls_every_engine_back() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 2, LONG);
    deployment.apply(update);

    deployment.engines.report(
        SECOND,
        process::Report {
            kind: Some(process::report::Kind::Stalled(process::Stalled {
                generation: 2,
                batch: "batch-3".to_owned(),
                job: "replay".to_owned(),
                elapsed_ms: 900,
            })),
        },
    );
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(deployment.status()["failure"]["kind"], "stalled");
    for generation in [PREVIOUS, SECOND] {
        assert_eq!(
            names(&deployment.engines.received(generation)),
            ["prepare", "switch", "rollback"]
        );
        assert_eq!(deployment.engines.engine(generation).generation(), 1);
    }
}

/// A rollback by decision returns to the kept generation and records the
/// reason; applying the target again is refused until forced.
#[test]
fn a_rollback_by_decision_returns_to_the_kept_generation() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 1, LONG);
    deployment.apply(update);

    let report = deployment
        .updater
        .rollback(decision("the responses got worse"))
        .expect("an update is on trial");
    assert_eq!(report.phase, Phase::Switching);
    assert!(report.rollback);
    deployment.run();

    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(
        deployment.engines.received(PREVIOUS).last(),
        Some(&command::Kind::Rollback(process::Rollback {
            reason: "the responses got worse".to_owned(),
        }))
    );
    assert_eq!(deployment.engines.engine(PREVIOUS).generation(), 1);
    assert_eq!(
        deployment
            .updater
            .unavailable(&deployment.target)
            .map(|mark| mark.reason),
        Some("the responses got worse".to_owned())
    );
    // Another target is not marked.
    assert_eq!(deployment.updater.unavailable(&deployment.serving), None);
}

/// A trial that reached its end with nothing rolling it back commits; one that has not reached
/// it decides nothing.
#[test]
fn a_trial_commits_at_its_end() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 1, Duration::ZERO);
    deployment.apply(update);
    assert_eq!(deployment.status()["phase"], "trial");

    deployment.updater.conclude(1);
    deployment.run();
    assert_eq!(deployment.status()["phase"], "committed");

    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(2, 1, LONG);
    deployment.apply(update);
    deployment.updater.conclude(2);
    assert_eq!(deployment.updater.in_flight(), Some((2, Phase::Trial)));
    assert!(deployment.updater.ends(2).is_some());
}

/// One update is in flight at a time, and a commit asked before the trial is
/// applied when the trial begins.
#[test]
fn one_update_is_in_flight_and_an_early_commit_is_kept() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 1, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    let second = deployment.generation(2, 1, LONG);
    assert!(matches!(
        deployment.updater.apply(second),
        Err(crate::leader::UpdateError::InProgress)
    ));

    assert!(deployment.updater.commit().is_some(), "kept for the trial");
    deployment.run();
    assert_eq!(
        deployment.entered(),
        ["preparing", "switching", "trial", "committed"]
    );
    // The slot is free again, and the update's follower still reads how
    // it ended once another is in flight.
    assert_eq!(deployment.updater.phase_of(1), Some(Phase::Committed));
    let third = deployment.generation(3, 1, LONG);
    assert!(deployment.updater.apply(third).is_ok());
    assert_eq!(deployment.updater.phase_of(1), Some(Phase::Committed));
    assert_eq!(deployment.updater.phase_of(3), Some(Phase::Preparing));
    assert_eq!(deployment.updater.phase_of(2), None, "it never ran");
}

/// A serving engine that exits during the trial ends it: the record returns
/// to the kept modules, which a relaunch or a standby runs.
#[test]
fn an_engine_lost_on_trial_rolls_the_record_back() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 1, LONG);
    deployment.apply(update);

    deployment.engines.exit(HOST, PREVIOUS, "signal: 11");
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(deployment.status()["failure"]["kind"], "process_exit");
    assert_eq!(
        deployment.record(SERVING).policy,
        deployment.serving.modules.policy
    );
}

// --- the epoch form -----------------------------------------------------------------------------

/// An epoch update to model files that process requests otherwise drains the serving engine
/// completely: its `Drain` states `complete`, so every request it holds runs there to its end and
/// none migrates to the target, whatever `update_config.drain_s` states.
#[test]
fn a_switch_to_files_that_process_requests_otherwise_drains_completely() {
    let mut deployment = Deployment::new(Refusals::default());
    let mut switch = deployment.switch(vec![(target_launch(HOST, 0, SPAWNED), Vec::new())]);
    switch.complete = true;
    let nodes = vec![Node::new(
        HOST.to_owned(),
        0,
        PREVIOUS,
        PREVIOUS,
        scopes(0..1),
        deployment.layout(),
        SERVING,
    )];
    let update = deployment.epoch_over(1, nodes, switch, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    deployment.engines.start_spawned(BUILD);
    deployment.run();
    assert_eq!(*deployment.epochs.served().borrow(), TARGET);
    assert_eq!(
        deployment.engines.received(PREVIOUS),
        [command::Kind::Drain(process::Drain {
            epoch: SERVING,
            deadline_ms: 5_000,
            shrink: false,
            complete: true,
        })]
    );
}

/// An epoch update starts the target from a launch file that states the
/// epoch's module set and the engine it succeeds, and switches once it
/// parked: the target is promoted first, and new requests go to it from its
/// `Serving`; then the serving engine drains beside it, with the deadline
/// `update_config.drain_s` states. The trial keeps the previous epoch parked,
/// and a commit stops it and retires its storage.
#[test]
fn an_epoch_update_promotes_the_target_and_then_drains_the_serving_engine() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    assert_eq!(deployment.status()["change"], "epoch");

    // Preparing: one spawn, whose launch states the modules, and the claim.
    let spawns = deployment.engines.spawns();
    assert_eq!(spawns.len(), 1);
    assert_eq!(
        (spawns[0].host.as_str(), spawns[0].generation),
        (HOST, SPAWNED)
    );
    let launch: proto::config::EngineArgs =
        serde_json::from_slice(&spawns[0].launch).expect("a launch file");
    assert_eq!((launch.engine, launch.epoch), (SPAWNED, TARGET));
    assert_eq!(launch.modules, deployment.target.modules);
    assert!(launch.parked());
    assert_eq!(deployment.engines.claims()[0].epoch, TARGET);

    // The target is slow to acknowledge its promotion: until it serves,
    // requests stay with the serving epoch, which is not drained.
    deployment.engines.start_spawned(BUILD);
    deployment.engines.engine(SPAWNED).withhold();
    deployment.run();
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Switching)));
    assert_eq!(names(&deployment.engines.received(SPAWNED)), ["promote"]);
    assert!(deployment.engines.received(PREVIOUS).is_empty());
    assert_eq!(*deployment.epochs.served().borrow(), 0);

    // Its `Serving` routes new requests to it, and the serving engine
    // drains: both epochs step until it drained.
    deployment.release(SPAWNED);
    assert_eq!(*deployment.epochs.served().borrow(), TARGET);
    assert_eq!(
        deployment.engines.received(SPAWNED),
        [command::Kind::Promote(process::Promote {
            recompute: false,
            grow: false,
        })]
    );
    assert_eq!(
        deployment.engines.received(PREVIOUS),
        [command::Kind::Drain(process::Drain {
            epoch: SERVING,
            deadline_ms: 5_000,
            shrink: false,
            complete: false,
        })]
    );

    // Trial: the previous epoch is kept parked.
    assert_eq!(deployment.entered(), ["preparing", "switching", "trial"]);
    assert_eq!(deployment.record(TARGET).status, EpochState::Trial);
    assert_eq!(deployment.record(TARGET).generation, Some(1));
    assert_eq!(deployment.record(TARGET).config.executable, BUILD);
    assert!(deployment.engines.stops().is_empty());
    let status = deployment.status();
    assert_eq!(status["phase"], "trial");
    assert_eq!(status["previous_epoch_kept"], "parked");
    assert!(status["switch_ms"].is_number());

    assert!(deployment.updater.commit().is_some());
    deployment.run();
    assert_eq!(deployment.entered(), ["committed"]);
    assert_eq!(deployment.record(TARGET).status, EpochState::Serving);
    assert_eq!(deployment.engines.stops(), vec![Some(PREVIOUS)]);
    let retiring = deployment
        .engines
        .claims()
        .pop()
        .expect("the commit's claim");
    assert_eq!((retiring.epoch, retiring.retire), (TARGET, vec![SERVING]));
    assert_eq!(
        deployment.epochs.serving().map(|(epoch, _)| epoch.number()),
        Some(TARGET)
    );
}

/// The drain states the deadline `update_config.drain_s` sets, and when the
/// target's layout differs the serving engines give their device memory
/// back as they drain while the targets grow into it. The option's default
/// is five seconds.
#[test]
fn the_drain_states_its_deadline_and_shrinks_across_layouts() {
    assert_eq!(UpdateConfig::default().drain_s, 5.0);

    let mut deployment = Deployment::new(Refusals::default());
    let mut switch = deployment.switch(vec![(target_launch(HOST, 0, SPAWNED), Vec::new())]);
    switch.drain = Duration::from_millis(2_500);
    deployment.across_layouts(&mut switch);
    let nodes = deployment.nodes(1);
    let update = deployment.epoch_over(1, nodes, switch, LONG);
    deployment.apply(update);

    assert_eq!(
        deployment.engines.received(PREVIOUS),
        [command::Kind::Drain(process::Drain {
            epoch: SERVING,
            deadline_ms: 2_500,
            shrink: true,
            complete: false,
        })]
    );
    assert_eq!(
        deployment.engines.received(SPAWNED),
        [command::Kind::Promote(process::Promote {
            recompute: false,
            grow: true,
        })]
    );
    let status = deployment.status();
    assert_eq!(
        (status["drain_s"].as_f64(), status["shrink"].as_bool()),
        (Some(2.5), Some(true))
    );
}

/// A release of `bytes` on device 0 of `host`, as a draining engine reports it.
fn shrunk(host: &str) -> process::Report {
    process::Report {
        kind: Some(process::report::Kind::Shrunk(process::Shrunk {
            devices: vec![0],
            bytes: 1 << 30,
            host: host.to_owned(),
        })),
    }
}

/// A serving engine that gives its device memory back as it drains across layouts reports
/// each release, and the target engines on its host are sent `Grow` for it; without `shrink`
/// nothing is relayed.
#[test]
fn a_release_of_a_draining_engine_grows_the_targets_on_its_host() {
    for shrink in [true, false] {
        let mut deployment = Deployment::new(Refusals::default());
        let mut switch = deployment.switch(vec![(target_launch(HOST, 0, SPAWNED), Vec::new())]);
        if shrink {
            deployment.across_layouts(&mut switch);
        }
        let nodes = deployment.nodes(1);
        let update = deployment.epoch_over(1, nodes, switch, LONG);
        deployment.apply(update);

        deployment.engines.report(PREVIOUS, shrunk(HOST));
        deployment.run();
        let expected: &[&str] = if shrink {
            &["promote", "grow"]
        } else {
            &["promote"]
        };
        assert_eq!(names(&deployment.engines.received(SPAWNED)), expected);
    }
}

/// A release names the host whose quotas it freed memory in, and the daemon sends `Grow` to the
/// scheduler node of each target world with an engine on that host, once: that node grows on
/// every rank it holds, its peers' on other hosts included, so a peer is never sent `Grow`, and
/// a release on a host where no target has an engine grows nothing.
#[test]
fn a_release_grows_the_scheduler_node_of_each_target_world_with_an_engine_on_its_host() {
    let mut deployment = Deployment::new(Refusals::default());
    let world = proto::config::World {
        scheduler: SPAWNED,
        channel: "rank-0:7000".to_owned(),
        store: "rank-0:7001".to_owned(),
        peers: 1,
        inherited: None,
    };
    let mut head = target_launch(HOST, 0, SPAWNED);
    head.world = Some(world.clone());
    let mut peer = target_launch(OTHER, 2, SPAWNED_PEER);
    peer.world = Some(world);
    let mut switch = deployment.switch(vec![(head, vec![1, 2]), (peer, Vec::new())]);
    deployment.across_layouts(&mut switch);
    let nodes = deployment.nodes(1);
    let update = deployment.epoch_over(1, nodes, switch, LONG);
    deployment.apply(update);

    for host in [OTHER, HOST, "rank-9"] {
        deployment.engines.report(PREVIOUS, shrunk(host));
        deployment.run();
    }
    assert_eq!(
        names(&deployment.engines.received(SPAWNED)),
        ["promote", "grow", "grow"]
    );
    assert_eq!(
        names(&deployment.engines.received(SPAWNED_PEER)),
        ["promote"]
    );
}

/// Where `timeline` first holds what `seen` matches.
fn seen_at(timeline: &[Seen], seen: impl Fn(&Seen) -> bool) -> usize {
    timeline
        .iter()
        .position(seen)
        .unwrap_or_else(|| panic!("the leader never did it: {timeline:#?}"))
}

/// Whether `seen` is a `Grow` sent to the engine of launch generation `launch`.
fn grow_to(launch: u64) -> impl Fn(&Seen) -> bool {
    move |seen| matches!(seen, Seen::Command(engine, command::Kind::Grow(_)) if *engine == launch)
}

/// Whether `seen` is a claim on `host` that retires epoch `retired`.
fn retiring(host: &'static str, retired: u64) -> impl Fn(&Seen) -> bool {
    move |seen| match seen {
        Seen::Request(kind) => matches!(
            kind.as_ref(),
            process::request::Kind::Claim(claim) if claim.host == host && claim.retire == [retired]
        ),
        Seen::Command(..) => false,
    }
}

/// Whether `seen` is the stop of the engine of launch generation `launch`.
fn stopping(launch: u64) -> impl Fn(&Seen) -> bool {
    move |seen| match seen {
        Seen::Request(kind) => matches!(
            kind.as_ref(),
            process::request::Kind::Stop(stop) if stop.generation == Some(launch)
        ),
        Seen::Command(..) => false,
    }
}

/// A commit releases the previous epoch's storage, and the target is then sent `Grow` to grow
/// into what that freed: after the stop that reaped the previous epoch's engine and the claim
/// that retired its storage on its host, whatever the trial kept of it and whether or not the
/// layouts differ, since its weights, graph pools and process storage leave the quota either
/// way. Nothing grows the target before the commit but what the drain gives back.
#[test]
fn a_commit_grows_the_target_into_what_the_previous_epochs_release_freed() {
    for keeps in [PreviousEpoch::Parked, PreviousEpoch::Host] {
        for across in [true, false] {
            let mut deployment = Deployment::new(Refusals::default());
            let mut switch = deployment.switch(vec![(target_launch(HOST, 0, SPAWNED), Vec::new())]);
            switch.keeps = keeps;
            if across {
                deployment.across_layouts(&mut switch);
            }
            let nodes = deployment.nodes(1);
            let update = deployment.epoch_over(1, nodes, switch, LONG);
            deployment.apply(update);
            assert_eq!(deployment.status()["phase"], "trial");
            assert_eq!(names(&deployment.engines.received(SPAWNED)), ["promote"]);

            assert!(deployment.updater.commit().is_some());
            deployment.run();
            assert_eq!(
                names(&deployment.engines.received(SPAWNED)),
                ["promote", "grow"],
                "{keeps:?}, across layouts: {across}"
            );
            let timeline = deployment.engines.timeline();
            let grown = seen_at(&timeline, grow_to(SPAWNED));
            assert!(seen_at(&timeline, stopping(PREVIOUS)) < grown);
            assert!(seen_at(&timeline, retiring(HOST, SERVING)) < grown);
        }
    }
}

/// A commit's release on several hosts grows each target world once, through its scheduler
/// node, which grows on every rank it holds, its peer's on the other host included: the `Grow`
/// follows the claims that retired the previous epoch on both hosts, and the peer is sent none.
#[test]
fn a_commit_grows_each_target_world_once_through_its_scheduler_node() {
    let mut deployment = Deployment::new(Refusals::default());
    deployment.engines.serve(SECOND);
    let nodes = vec![
        Node::new(
            HOST.to_owned(),
            0,
            PREVIOUS,
            PREVIOUS,
            scopes(0..2),
            deployment.layout(),
            SERVING,
        ),
        Node::new(
            OTHER.to_owned(),
            2,
            SECOND,
            PREVIOUS,
            scopes(2..4),
            deployment.layout(),
            SERVING,
        ),
    ];
    let world = proto::config::World {
        scheduler: SPAWNED,
        channel: "rank-0:7000".to_owned(),
        store: "rank-0:7001".to_owned(),
        peers: 1,
        inherited: None,
    };
    let mut head = target_launch(HOST, 0, SPAWNED);
    head.world = Some(world.clone());
    let mut peer = target_launch(OTHER, 2, SPAWNED_PEER);
    peer.world = Some(world);
    let mut switch = deployment.switch(vec![(head, vec![1, 2]), (peer, Vec::new())]);
    deployment.across_layouts(&mut switch);
    let update = deployment.epoch_over(1, nodes, switch, LONG);
    deployment.apply(update);
    assert_eq!(deployment.status()["phase"], "trial");

    assert!(deployment.updater.commit().is_some());
    deployment.run();
    assert_eq!(
        names(&deployment.engines.received(SPAWNED)),
        ["promote", "grow"]
    );
    assert_eq!(
        names(&deployment.engines.received(SPAWNED_PEER)),
        ["promote"]
    );
    let timeline = deployment.engines.timeline();
    let grown = seen_at(&timeline, grow_to(SPAWNED));
    for host in [HOST, OTHER] {
        assert!(
            seen_at(&timeline, retiring(host, SERVING)) < grown,
            "{host}"
        );
    }
}

/// A rollback ends by stopping the target and retiring its storage, and the previous epoch,
/// which serves again, is then sent `Grow` to grow into what that freed: a parked one at its own
/// launch generation, and one its owner started again at the launch generation it serves at.
#[test]
fn a_rollback_grows_the_previous_epoch_into_what_the_targets_release_freed() {
    const RELAUNCHED: u64 = 12;

    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment.apply(update);
    deployment
        .updater
        .rollback(decision("the target answers worse"))
        .expect("an update is on trial");
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(
        names(&deployment.engines.received(PREVIOUS)),
        ["drain", "promote", "grow"]
    );
    let timeline = deployment.engines.timeline();
    let grown = seen_at(&timeline, grow_to(PREVIOUS));
    assert!(seen_at(&timeline, stopping(SPAWNED)) < grown);
    assert!(seen_at(&timeline, retiring(HOST, TARGET)) < grown);

    // The trial stopped the previous epoch, and its owner starts it again for the rollback.
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Weights, LONG);
    deployment.apply(update);
    deployment
        .updater
        .rollback(decision("the target answers worse"))
        .expect("an update is on trial");
    deployment.run();
    deployment.engines.start_controlled(HOST, RELAUNCHED, BUILD);
    deployment.updater.relaunched(PREVIOUS, RELAUNCHED);
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(names(&deployment.engines.received(RELAUNCHED)), ["grow"]);
    let timeline = deployment.engines.timeline();
    assert!(seen_at(&timeline, retiring(HOST, TARGET)) < seen_at(&timeline, grow_to(RELAUNCHED)));
}

/// An epoch an abandoned update dropped, whose storage a node could not retire when the update
/// ended, keeps it until the next epoch update's first claim retires it; that release grows the
/// serving engine, which serves on while the next target prepares beside it.
#[test]
fn the_next_update_grows_the_serving_epoch_into_what_a_dropped_epochs_release_freed() {
    const NEXT: u64 = 3;
    const NEXT_SPAWNED: u64 = 11;

    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    // The node cannot retire the abandoned target's storage, which the next update's first claim
    // then retires.
    deployment.engines.refuse(Refusals {
        claim: Some("the Storage Server is restarting".to_owned()),
        ..Refusals::default()
    });
    deployment.engines.exit(HOST, SPAWNED, "signal: 9");
    deployment.updater.advance();
    assert_eq!(deployment.status()["phase"], "abandoned");
    assert!(deployment.engines.received(PREVIOUS).is_empty());
    deployment.engines.refuse(Refusals::default());

    let mut launch = target_launch(HOST, 0, NEXT_SPAWNED);
    launch.epoch = NEXT;
    let mut switch = deployment.switch(vec![(launch, Vec::new())]);
    switch.epoch = NEXT;
    switch.retire = deployment.updater.take_retire();
    let nodes = deployment.nodes(1);
    let update = deployment.epoch_over(2, nodes, switch, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    assert_eq!(deployment.updater.in_flight(), Some((2, Phase::Preparing)));
    let timeline = deployment.engines.timeline();
    assert!(seen_at(&timeline, retiring(HOST, TARGET)) < seen_at(&timeline, grow_to(PREVIOUS)));
    deployment.run();
    assert_eq!(names(&deployment.engines.received(PREVIOUS))[0], "grow");
}

/// The epoch form covers every node of both epochs: a target whose world
/// spans two hosts starts one engine on each, its scheduler node inheriting
/// the world's sockets, every node claims the target's storage, every target
/// is promoted and every serving engine on every node drained.
#[test]
fn an_epoch_update_covers_every_node_of_both_epochs() {
    let mut deployment = Deployment::new(Refusals::default());
    deployment.engines.serve(SECOND);
    // Both epochs place their group's ranks 0 and 1 on HOST, 2 and 3 on OTHER.
    let group = &mut deployment.target.engine.parallel.worker_groups[0];
    group.ranks = [HOST, HOST, OTHER, OTHER]
        .iter()
        .enumerate()
        .map(|(rank, host)| proto::config::WorkerRank {
            host: (*host).to_owned(),
            device: proto::config::RankDevice::cuda(rank as u32 % 2),
        })
        .collect();
    let nodes = vec![
        Node::new(
            HOST.to_owned(),
            0,
            PREVIOUS,
            PREVIOUS,
            scopes(0..2),
            deployment.layout(),
            SERVING,
        ),
        Node::new(
            OTHER.to_owned(),
            2,
            SECOND,
            PREVIOUS,
            scopes(2..4),
            deployment.layout(),
            SERVING,
        ),
    ];
    let switch = deployment.switch(vec![
        (target_launch(HOST, 0, SPAWNED), vec![1, 2]),
        (target_launch(OTHER, 2, SPAWNED_PEER), Vec::new()),
    ]);
    let update = deployment.epoch_over(1, nodes, switch, LONG);
    deployment.apply(update);

    let spawns = deployment.engines.spawns();
    let spawned: Vec<_> = spawns
        .iter()
        .map(|spawn| {
            (
                spawn.host.as_str(),
                spawn.rank,
                spawn.generation,
                spawn.inherit.clone(),
            )
        })
        .collect();
    assert_eq!(
        spawned,
        [
            (HOST, 0, SPAWNED, vec![1, 2]),
            (OTHER, 2, SPAWNED_PEER, Vec::new())
        ]
    );
    let claimed: Vec<_> = deployment
        .engines
        .claims()
        .iter()
        .map(|claim| (claim.host.clone(), claim.epoch, claim.scopes[0].rank))
        .collect();
    assert_eq!(
        claimed,
        [(HOST.to_owned(), TARGET, 0), (OTHER.to_owned(), TARGET, 2)]
    );
    for target in [SPAWNED, SPAWNED_PEER] {
        assert_eq!(names(&deployment.engines.received(target)), ["promote"]);
    }
    for serving in [PREVIOUS, SECOND] {
        assert_eq!(names(&deployment.engines.received(serving)), ["drain"]);
    }
    assert_eq!(deployment.status()["phase"], "trial");

    assert!(deployment.updater.commit().is_some());
    deployment.run();
    assert_eq!(
        deployment.engines.stops(),
        vec![Some(PREVIOUS), Some(SECOND)]
    );
    // The commit's claims state the scopes the target's workers retain
    // their tensors under, so the target keeps them as the serving epoch
    // retires.
    let retired: Vec<_> = deployment
        .engines
        .claims()
        .iter()
        .filter(|claim| claim.retire == [SERVING])
        .map(|claim| (claim.host.clone(), claim.scopes.clone()))
        .collect();
    assert_eq!(
        retired,
        [
            (HOST.to_owned(), scopes(0..2)),
            (OTHER.to_owned(), scopes(2..4))
        ]
    );
}

/// A commit publishes the target at every node that claimed it, before any
/// node retires the previous epoch: a target that places ranks on a node the
/// serving epoch does not use is claimed and published there too. Nothing is
/// published while the target is on trial, and a rollback publishes nothing.
#[test]
fn a_commit_publishes_the_target_at_every_node_that_claimed_it() {
    let start = |deployment: &mut Deployment| {
        let nodes = deployment.nodes(1);
        let switch = deployment.switch(vec![
            (target_launch(HOST, 0, SPAWNED), Vec::new()),
            (target_launch(OTHER, 2, SPAWNED_PEER), Vec::new()),
        ]);
        let update = deployment.epoch_over(1, nodes, switch, LONG);
        deployment.apply(update);
        assert_eq!(deployment.status()["phase"], "trial");
    };

    let mut deployment = Deployment::new(Refusals::default());
    start(&mut deployment);
    let claimed: Vec<_> = deployment
        .engines
        .claims()
        .iter()
        .map(|claim| (claim.host.clone(), claim.epoch))
        .collect();
    assert_eq!(
        claimed,
        [(HOST.to_owned(), TARGET), (OTHER.to_owned(), TARGET)]
    );
    assert!(deployment.engines.publications().is_empty());

    assert!(deployment.updater.commit().is_some());
    deployment.run();
    let requests = deployment.engines.requests();
    let published: Vec<_> = requests
        .iter()
        .filter_map(|kind| match kind {
            process::request::Kind::PublishEpochs(publish) => {
                Some((publish.host.as_str(), publish.epochs.clone()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(published, [(HOST, vec![TARGET]), (OTHER, vec![TARGET])]);
    let last_published = requests
        .iter()
        .rposition(|kind| matches!(kind, process::request::Kind::PublishEpochs(_)))
        .expect("the commit published the target");
    let first_retired = requests
        .iter()
        .position(|kind| {
            matches!(kind, process::request::Kind::Claim(claim) if claim.retire == [SERVING])
        })
        .expect("the commit retired the previous epoch");
    assert!(
        last_published < first_retired,
        "a node retired the previous epoch before every node published the target: {requests:?}"
    );

    let mut deployment = Deployment::new(Refusals::default());
    start(&mut deployment);
    deployment
        .updater
        .rollback(decision("the target answers worse"))
        .expect("an update is on trial");
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert!(deployment.engines.publications().is_empty());
}

/// The two epochs need not have as many replicas: two serving replicas move
/// to one target engine, both drained once it serves.
#[test]
fn an_epoch_update_moves_two_replicas_to_one() {
    let mut deployment = Deployment::new(Refusals::default());
    let nodes = deployment.nodes(2);
    let mut switch = deployment.switch(vec![(target_launch(HOST, 0, SPAWNED), Vec::new())]);
    deployment.across_layouts(&mut switch);
    let update = deployment.epoch_over(1, nodes, switch, LONG);
    deployment.apply(update);

    assert_eq!(deployment.engines.spawns().len(), 1);
    assert_eq!(names(&deployment.engines.received(SPAWNED)), ["promote"]);
    for serving in [PREVIOUS, SECOND] {
        assert_eq!(names(&deployment.engines.received(serving)), ["drain"]);
    }
    assert_eq!(
        deployment.status()["engines"],
        serde_json::json!({ "serving": 2, "target": 1 })
    );
    assert_eq!(deployment.status()["phase"], "trial");
    assert_eq!(*deployment.epochs.served().borrow(), TARGET);
}

/// An epoch update that moves one replica leaves the other serving its epoch: only the moved
/// replica's engine drains and stops, the commit retires no epoch the other still executes, and
/// it claims that epoch again with the other replica's scopes while ending the moved replica's;
/// a rollback claims the epoch with both replicas' scopes.
#[test]
fn an_epoch_update_moves_one_replica_and_the_other_serves_on() {
    let moving = |deployment: &mut Deployment| {
        let mut nodes = deployment.nodes(2);
        let staying = nodes.pop().expect("two serving engines");
        let mut switch = deployment.switch(vec![(target_launch(HOST, 0, SPAWNED), Vec::new())]);
        switch.bystanders = vec![staying];
        let update = deployment.epoch_over(1, nodes, switch, LONG);
        deployment.apply(update);
        assert_eq!(names(&deployment.engines.received(PREVIOUS)), ["drain"]);
        assert!(deployment.engines.received(SECOND).is_empty());
        assert_eq!(deployment.status()["phase"], "trial");
    };

    let mut deployment = Deployment::new(Refusals::default());
    moving(&mut deployment);
    assert!(deployment.updater.commit().is_some());
    deployment.run();
    assert_eq!(deployment.engines.stops(), vec![Some(PREVIOUS)]);
    let claims = deployment.engines.claims();
    assert!(
        claims.iter().all(|claim| !claim.retire.contains(&SERVING)),
        "the epoch another replica executes is retired: {claims:?}"
    );
    let serving: Vec<_> = claims
        .iter()
        .filter(|claim| claim.epoch == SERVING)
        .map(|claim| (claim.scopes.clone(), claim.release.clone()))
        .collect();
    assert_eq!(serving, [(scopes(1..2), scopes(0..1))]);

    let mut deployment = Deployment::new(Refusals::default());
    moving(&mut deployment);
    deployment
        .updater
        .rollback(decision("the target answers worse"))
        .expect("an update is on trial");
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    let claims = deployment.engines.claims();
    let kept: Vec<_> = claims
        .iter()
        .filter(|claim| claim.epoch == SERVING && claim.retire.contains(&TARGET))
        .map(|claim| claim.scopes.clone())
        .collect();
    assert_eq!(kept, [scopes(0..2)]);
}

/// An epoch change to another number of replicas moves every replica, whichever side has more
/// (N → 1, 1 → N, N → M): every target replica starts and is promoted, and once the targets serve
/// every serving engine drains, completely when the target's model files process requests
/// otherwise. The commit stops every serving engine and retires the previous epoch, which no
/// replica serves any longer; a rollback promotes every serving engine again, then drains and
/// stops every target and retires the target epoch, the previous epoch keeping every replica's
/// scopes.
#[test]
fn a_change_of_the_replica_count_moves_every_replica() {
    // Four independent replicas to one expert union of another model, as the drain is complete
    // then; one replica to four; three to two.
    for (serving, targets, complete) in [(4, 1, true), (1, 4, false), (3, 2, false)] {
        for commit in [true, false] {
            let case = format!("{serving} → {targets} replicas, commit: {commit}");
            let mut deployment = Deployment::new(Refusals::default());
            let replicas = deployment.serving_replicas(serving);
            let launches = deployment.target_replicas(targets);
            let update = deployment.change(1, replicas, launches, complete);
            deployment.apply(update);

            let previous: Vec<u64> = (0..serving).map(|i| PREVIOUS + u64::from(i)).collect();
            let started: Vec<u64> = (0..targets).map(|j| SPAWNED + u64::from(j)).collect();
            let spawned: Vec<u64> = deployment
                .engines
                .spawns()
                .iter()
                .map(|spawn| spawn.generation)
                .collect();
            assert_eq!(spawned, started, "{case}");
            for target in &started {
                assert_eq!(
                    names(&deployment.engines.received(*target)),
                    ["promote"],
                    "{case}"
                );
            }
            for engine in &previous {
                assert_eq!(
                    deployment.engines.received(*engine),
                    [command::Kind::Drain(process::Drain {
                        epoch: SERVING,
                        deadline_ms: 5_000,
                        shrink: false,
                        complete,
                    })],
                    "{case}: engine {engine}"
                );
            }
            assert_eq!(
                deployment.status()["engines"],
                serde_json::json!({ "serving": serving, "target": targets }),
                "{case}"
            );
            assert_eq!(deployment.status()["phase"], "trial", "{case}");
            assert_eq!(*deployment.epochs.served().borrow(), TARGET, "{case}");

            if commit {
                assert!(deployment.updater.commit().is_some());
                deployment.run();
                assert_eq!(deployment.status()["phase"], "committed", "{case}");
                let stopped: Vec<Option<u64>> = previous.iter().copied().map(Some).collect();
                assert_eq!(deployment.engines.stops(), stopped, "{case}");
                let claims = deployment.engines.claims();
                assert!(
                    claims
                        .iter()
                        .any(|claim| claim.epoch == TARGET && claim.retire == [SERVING]),
                    "{case}: the commit retires the previous epoch: {claims:?}"
                );
                assert!(
                    claims.iter().all(|claim| claim.epoch != SERVING),
                    "{case}: no replica keeps the previous epoch: {claims:?}"
                );
                assert_eq!(
                    deployment.epochs.serving().map(|(epoch, _)| epoch.number()),
                    Some(TARGET),
                    "{case}"
                );
            } else {
                deployment
                    .updater
                    .rollback(decision("the target answers worse"))
                    .expect("an update is on trial");
                deployment.run();
                assert_eq!(deployment.status()["phase"], "rolled_back", "{case}");
                assert_eq!(*deployment.epochs.served().borrow(), SERVING, "{case}");
                // Each previous engine serves again and grows into what the targets' release
                // freed.
                for engine in &previous {
                    assert_eq!(
                        names(&deployment.engines.received(*engine)),
                        ["drain", "promote", "grow"],
                        "{case}: engine {engine}"
                    );
                }
                for target in &started {
                    assert_eq!(
                        names(&deployment.engines.received(*target)),
                        ["promote", "drain"],
                        "{case}: target {target}"
                    );
                }
                let stopped: Vec<Option<u64>> = started.iter().copied().map(Some).collect();
                assert_eq!(deployment.engines.stops(), stopped, "{case}");
                let retiring = deployment
                    .engines
                    .claims()
                    .pop()
                    .expect("the rollback's claim");
                assert_eq!(
                    (retiring.epoch, retiring.scopes, retiring.retire),
                    (SERVING, scopes(0..serving), vec![TARGET]),
                    "{case}"
                );
            }
        }
    }
}

/// A target of as many replicas as the deployment's pairs each target replica with the serving
/// replica of its index and moves only the replicas whose launch it touches: the target of the
/// touched replica alone starts, that replica's engine drains, and the other's serves on,
/// commanded nothing, its epoch kept rather than retired at the commit.
#[test]
fn a_target_of_as_many_replicas_moves_the_replicas_it_touches() {
    let mut deployment = Deployment::new(Refusals::default());
    let replicas = deployment.serving_replicas(2);
    let mut launches = deployment.target_replicas(2);
    // Target replica 1 holds another slot count, which its build part states.
    let touched = &mut launches[1].engine_config;
    touched.max_running_requests = touched
        .max_running_requests
        .checked_add(1)
        .expect("a slot count below the maximum");
    let update = deployment.change(1, replicas, launches, false);
    deployment.apply(update);

    let spawned: Vec<u64> = deployment
        .engines
        .spawns()
        .iter()
        .map(|spawn| spawn.generation)
        .collect();
    assert_eq!(spawned, [SPAWNED + 1]);
    assert!(deployment.engines.received(PREVIOUS).is_empty());
    assert_eq!(names(&deployment.engines.received(SECOND)), ["drain"]);

    assert!(deployment.updater.commit().is_some());
    deployment.run();
    assert_eq!(deployment.engines.stops(), vec![Some(SECOND)]);
    let claims = deployment.engines.claims();
    assert!(
        claims.iter().all(|claim| !claim.retire.contains(&SERVING)),
        "the epoch the untouched replica serves is retired: {claims:?}"
    );
}

/// A replica added on another host leaves the layout of the replica already serving: the
/// serving engine, whose regions and segments the target of its layout maps, drains without
/// giving its memory back, that target is promoted without growing, and only the new replica's
/// engine, of a layout of its own, grows. The update reports no shrink.
#[test]
fn a_replica_added_beside_a_kept_layout_moves_no_memory_of_it() {
    let mut deployment = Deployment::new(Refusals::default());
    let mut switch = deployment.switch(vec![
        (target_launch(HOST, 0, SPAWNED), Vec::new()),
        (target_launch(OTHER, 1, SPAWNED_PEER), Vec::new()),
    ]);
    let added = &mut switch.launches[1].0.engine_config;
    added.max_running_requests = added
        .max_running_requests
        .checked_add(1)
        .expect("a slot count below the maximum");
    switch.target_layouts = switch
        .launches
        .iter()
        .map(|(launch, _)| launch.engine_config.layout())
        .collect();
    let nodes = deployment.nodes(1);
    let update = deployment.epoch_over(1, nodes, switch, LONG);
    deployment.apply(update);

    assert_eq!(
        deployment.engines.received(SPAWNED),
        [command::Kind::Promote(process::Promote {
            recompute: false,
            grow: false,
        })]
    );
    assert_eq!(
        deployment.engines.received(SPAWNED_PEER),
        [command::Kind::Promote(process::Promote {
            recompute: false,
            grow: true,
        })]
    );
    assert_eq!(
        deployment.engines.received(PREVIOUS),
        [command::Kind::Drain(process::Drain {
            epoch: SERVING,
            deadline_ms: 5_000,
            shrink: false,
            complete: false,
        })]
    );
    assert_eq!(deployment.status()["shrink"].as_bool(), Some(false));
}

/// The switch waits for every target before it drains: a second target that
/// is slow to serve keeps the serving engines stepping, while new requests
/// already go to the target from the first one's `Serving`.
#[test]
fn requests_go_to_the_target_from_its_first_serving() {
    let mut deployment = Deployment::new(Refusals::default());
    let switch = deployment.switch(vec![
        (target_launch(HOST, 0, SPAWNED), Vec::new()),
        (target_launch(HOST, 0, SPAWNED_PEER), Vec::new()),
    ]);
    let nodes = deployment.nodes(1);
    let update = deployment.epoch_over(1, nodes, switch, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    deployment.engines.start_spawned(BUILD);
    deployment.engines.engine(SPAWNED_PEER).withhold();
    deployment.run();

    assert_eq!(*deployment.epochs.served().borrow(), TARGET);
    assert!(deployment.engines.received(PREVIOUS).is_empty());
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Switching)));

    deployment.release(SPAWNED_PEER);
    assert_eq!(names(&deployment.engines.received(PREVIOUS)), ["drain"]);
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Trial)));
}

/// A node that refuses the spawn or the claim abandons the update: the
/// serving epoch is not drained, and a target that started is stopped.
#[test]
fn a_refused_spawn_or_claim_abandons_the_update() {
    let mut deployment = Deployment::new(Refusals {
        spawn: Some("no device is free".to_owned()),
        ..Default::default()
    });
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment.apply(update);
    assert_eq!(deployment.status()["phase"], "abandoned");
    assert_eq!(deployment.status()["failure"]["kind"], "spawn_failed");
    assert!(deployment.engines.received(PREVIOUS).is_empty());
    assert_eq!(deployment.record(TARGET).status, EpochState::Released);

    let mut deployment = Deployment::new(Refusals {
        claim: Some("the Storage Server refused".to_owned()),
        ..Default::default()
    });
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment.apply(update);
    assert_eq!(deployment.status()["failure"]["kind"], "claim_failed");
    assert_eq!(deployment.engines.stops(), vec![Some(SPAWNED)]);
    assert!(deployment.engines.received(PREVIOUS).is_empty());
    assert_eq!(*deployment.epochs.served().borrow(), 0);
}

/// A target that exits while it prepares abandons the update; the storage of the epoch it would
/// have been is retired at once, after its stop, and the serving engine grows into what that
/// freed, so no later update is left to retire it.
#[test]
fn a_target_lost_while_preparing_abandons_the_update() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    deployment.engines.exit(HOST, SPAWNED, "signal: 9");
    deployment.run();

    assert_eq!(deployment.status()["phase"], "abandoned");
    assert_eq!(deployment.status()["failure"]["kind"], "process_exit");
    let timeline = deployment.engines.timeline();
    assert!(seen_at(&timeline, stopping(SPAWNED)) < seen_at(&timeline, retiring(HOST, TARGET)));
    assert!(seen_at(&timeline, retiring(HOST, TARGET)) < seen_at(&timeline, grow_to(PREVIOUS)));
    assert_eq!(names(&deployment.engines.received(PREVIOUS)), ["grow"]);
    assert!(deployment.updater.take_retire().is_empty());
}

/// A GPU's identity, as a target's ask for room names it.
const GPU: [u8; 16] = [7; 16];

/// A target whose start lacks room in its GPU's quota asks for it before it parks: the leader is
/// handed the ask, and once the serving engines made the room the target is answered, parks, and
/// the switch goes on.
#[test]
fn a_target_that_lacks_room_parks_once_the_serving_engines_made_it() {
    let mut deployment = Deployment::new(Refusals::default());
    let mut asks = deployment
        .updater
        .room_asks()
        .expect("the asks are read once");
    deployment
        .engines
        .arm_spawned(|engine| engine.want_room(1, &[(GPU, 8 << 30)]));
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment.apply(update);

    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Preparing)));
    let wanted = asks
        .try_recv()
        .expect("the target's ask reached the leader");
    assert_eq!(
        (
            wanted.update,
            wanted.host.as_str(),
            wanted.launch,
            wanted.id,
            wanted.devices.as_slice()
        ),
        (1, HOST, SPAWNED, 1, [(GPU, 8 << 30)].as_slice())
    );
    assert!(asks.try_recv().is_err(), "one ask, handed over once");
    assert!(deployment.engines.received(PREVIOUS).is_empty());

    deployment.updater.room_made(&wanted, Ok(()));
    deployment.run();
    assert_eq!(
        deployment.engines.received(SPAWNED)[0],
        command::Kind::Room(process::Room {
            id: 1,
            refused: String::new(),
        })
    );
    assert_eq!(
        names(&deployment.engines.received(SPAWNED)),
        ["room", "promote"]
    );
    assert_eq!(names(&deployment.engines.received(PREVIOUS)), ["drain"]);
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Trial)));
}

/// Room the serving engines cannot make abandons the update: the target is answered with their
/// refusal and stopped, its storage is retired at once, and the serving engine grows back into
/// what that freed, the room its cache gave the target included.
#[test]
fn room_the_serving_engines_cannot_make_abandons_the_update_and_they_grow_back() {
    let mut deployment = Deployment::new(Refusals::default());
    let mut asks = deployment
        .updater
        .room_asks()
        .expect("the asks are read once");
    deployment
        .engines
        .arm_spawned(|engine| engine.want_room(1, &[(GPU, 8 << 30)]));
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment.apply(update);
    let wanted = asks
        .try_recv()
        .expect("the target's ask reached the leader");

    let refusal = "a release of every KV cache segment of region 0 past the first leaves a device \
                   too little room";
    deployment
        .updater
        .room_made(&wanted, Err(refusal.to_owned()));
    deployment.run();

    let status = deployment.status();
    assert_eq!(status["phase"], "abandoned");
    assert_eq!(status["failure"]["kind"], "prepare_failed");
    let reason = status["failure"]["reason"]
        .as_str()
        .expect("the failure states a reason");
    assert!(
        reason.contains(HOST) && reason.contains("8589934592 bytes") && reason.ends_with(refusal),
        "{reason}"
    );
    assert_eq!(
        deployment.engines.received(SPAWNED),
        [command::Kind::Room(process::Room {
            id: 1,
            refused: refusal.to_owned(),
        })]
    );
    let timeline = deployment.engines.timeline();
    assert!(seen_at(&timeline, stopping(SPAWNED)) < seen_at(&timeline, retiring(HOST, TARGET)));
    assert!(seen_at(&timeline, retiring(HOST, TARGET)) < seen_at(&timeline, grow_to(PREVIOUS)));
    assert_eq!(names(&deployment.engines.received(PREVIOUS)), ["grow"]);
}

/// An engine that asks for room while no update prepares it, such as an epoch's first start, is
/// answered that no room is made: its start fails as it would for want of room, rather than
/// waiting for an answer.
#[test]
fn an_ask_for_room_of_no_target_is_refused() {
    const CONTROLLED: u64 = 7;
    let mut deployment = Deployment::new(Refusals::default());
    deployment.engines.start_controlled(HOST, CONTROLLED, BUILD);
    deployment.run();
    let ask = deployment
        .engines
        .engine(CONTROLLED)
        .room_wanted(3, &[(GPU, 1 << 30)]);
    deployment.engines.report(CONTROLLED, ask);
    deployment.run();

    assert_eq!(
        deployment.engines.received(CONTROLLED),
        [command::Kind::Room(process::Room {
            id: 3,
            refused: "no epoch update prepares this engine".to_owned(),
        })]
    );
}

/// A target of the serving engine's layout that states its start opened regions of its own (its
/// model lays its pages out otherwise, a drafter's window owners added) takes the serving
/// engine's memory over the switch as a target of another layout does: the serving engine
/// drains with `shrink`, the target is promoted with `grow` and grows on each release. A target
/// that states it maps the serving engine's regions shares them, as one of an earlier build,
/// which states nothing, does.
#[test]
fn a_target_of_the_same_layout_takes_the_memory_only_when_its_regions_are_its_own() {
    for (shares, moves) in [(Some(false), true), (Some(true), false), (None, false)] {
        let mut deployment = Deployment::new(Refusals::default());
        deployment
            .engines
            .arm_spawned(move |engine| engine.state_shares_regions(shares));
        let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
        deployment.apply(update);

        assert_eq!(
            deployment.engines.received(PREVIOUS),
            [command::Kind::Drain(process::Drain {
                epoch: SERVING,
                deadline_ms: 5_000,
                shrink: moves,
                complete: false,
            })],
            "{shares:?}"
        );
        assert_eq!(
            deployment.engines.received(SPAWNED),
            [command::Kind::Promote(process::Promote {
                recompute: false,
                grow: moves,
            })],
            "{shares:?}"
        );
        assert_eq!(deployment.status()["shrink"].as_bool(), Some(moves));

        deployment.engines.report(PREVIOUS, shrunk(HOST));
        deployment.run();
        let expected: &[&str] = if moves {
            &["promote", "grow"]
        } else {
            &["promote"]
        };
        assert_eq!(
            names(&deployment.engines.received(SPAWNED)),
            expected,
            "{shares:?}"
        );
    }
}

/// A target that exits as it starts abandons the update with a failure that names the engine's
/// exit status and the error it stated as it ended (its exit reason): what the `PUT`'s answer, the
/// revision's reason and the update's `failure` state of why the update did not stand.
#[test]
fn a_target_that_exits_as_it_starts_names_its_error_in_the_failure() {
    let deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    let error = "reading the launch at descriptor 92: unknown field `boundary_interval`, \
                 expected one of `max_tokens`, `block_size`";
    deployment
        .engines
        .fail(HOST, SPAWNED, "exit status: 1", error);
    deployment.updater.advance();

    let status = deployment.status();
    assert_eq!(status["phase"], "abandoned");
    assert_eq!(status["failure"]["kind"], "process_exit");
    let reason = status["failure"]["reason"]
        .as_str()
        .expect("the failure states a reason");
    assert!(
        reason.contains(&format!("launch generation {SPAWNED}"))
            && reason.contains("exit status: 1")
            && reason.ends_with(error),
        "{reason}"
    );
}

/// A rollback asked while the target prepares abandons the update.
#[test]
fn a_rollback_before_the_switch_abandons_an_epoch_update() {
    let deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    deployment
        .updater
        .rollback(decision("the wrong revision"))
        .expect("an update is in flight");
    assert_eq!(deployment.status()["phase"], "abandoned");
    assert_eq!(deployment.engines.stops(), vec![Some(SPAWNED)]);
    assert_eq!(deployment.updater.unavailable(&deployment.target), None);
}

/// A rollback on trial is the switch the other way: the parked previous
/// epoch is promoted again first, new requests return to it from its
/// `Serving`, and the target drains beside it; once drained it stops and its
/// storage is retired.
#[test]
fn a_rollback_on_trial_promotes_the_previous_epoch_before_the_target_drains() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment.apply(update);
    deployment.entered();

    deployment.engines.engine(PREVIOUS).withhold();
    deployment
        .updater
        .rollback(decision("the model no longer responds usefully"))
        .expect("an update is on trial");
    deployment.run();
    assert_eq!(
        deployment.engines.received(PREVIOUS)[1],
        command::Kind::Promote(process::Promote {
            recompute: false,
            grow: false,
        })
    );
    assert_eq!(names(&deployment.engines.received(SPAWNED)), ["promote"]);
    assert_eq!(*deployment.epochs.served().borrow(), TARGET);

    deployment.release(PREVIOUS);
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
    assert_eq!(
        deployment.engines.received(SPAWNED)[1],
        command::Kind::Drain(process::Drain {
            epoch: TARGET,
            deadline_ms: 5_000,
            shrink: false,
            complete: false,
        })
    );
    assert_eq!(deployment.entered(), ["switching", "rolled_back"]);
    assert_eq!(deployment.engines.stops(), vec![Some(SPAWNED)]);
    assert_eq!(deployment.record(TARGET).status, EpochState::Released);
    // The target's storage is retired on the node that keeps the previous epoch.
    let retiring = deployment
        .engines
        .claims()
        .pop()
        .expect("the rollback's claim");
    assert_eq!((retiring.epoch, retiring.retire), (SERVING, vec![TARGET]));
    assert!(deployment.updater.unavailable(&deployment.target).is_some());
}

/// A report of `batches` batches of `step_us` each, one query token apiece.
fn steps(step_us: u64, batches: u64) -> SchedulerStats {
    SchedulerStats {
        batch_roundtrip_us: step_us * batches,
        batch_count: batches,
        batch_tokens: batches,
        ..Default::default()
    }
}

/// A trial begins only once the switch ended, and a threshold decides only
/// an update on trial. Over the steps of an exploratory run, a serving epoch
/// at 10666 us whose target averaged 29744 us over its warm-up while both
/// epochs stepped: the switch's steps are not the trial's, and the target,
/// at 9829 us on trial, stands. Once its average passes the regression it
/// rolls back with the threshold as its failure, and the status keeps the
/// rates that decided it, whatever the target computes after the decision.
#[test]
fn a_threshold_reads_the_trial_from_the_switchs_end_and_the_status_keeps_its_inputs() {
    let window = Duration::from_millis(40);
    let thresholds = RollbackConfig {
        step_time_regression: Some(0.05),
        window: Some(window),
        ..Default::default()
    };
    let observer = Arc::new(Observer::new(Duration::from_secs(60)));
    let trial = |observer: &Arc<Observer>, switch: Instant| {
        let replica = Baseline::new(
            Arc::clone(observer),
            TARGET,
            Some(SERVING),
            thresholds.window,
            switch,
        );
        Trial::begin(thresholds.clone(), vec![replica], (0, 0))
    };
    let mut deployment = Deployment::new(Refusals::default());
    assert_eq!(deployment.updater.read_trial(1, (0, 0)), None, "no update");

    observer.record(SERVING, 0, &steps(10_666, 400));
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    deployment.engines.start_spawned(BUILD);
    deployment.engines.engine(SPAWNED).withhold();
    deployment.run();
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Switching)));

    // The target's first batches, while the serving epoch still steps.
    observer.record(TARGET, 0, &steps(29_744, 3));
    assert!(
        !deployment
            .updater
            .begin_trial(|switch| trial(&observer, switch)),
        "the switch has not ended"
    );
    assert_eq!(deployment.updater.read_trial(1, (0, 0)), None);

    deployment.release(SPAWNED);
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Trial)));
    std::thread::sleep(Duration::from_millis(2));
    assert!(
        deployment
            .updater
            .begin_trial(|switch| trial(&observer, switch))
    );
    std::thread::sleep(window);
    observer.record(TARGET, 0, &steps(9_829, 400));
    assert_eq!(deployment.updater.read_trial(1, (0, 0)), None);
    let replica = &deployment.status()["trial"]["replicas"][0];
    assert_eq!(replica["rates"]["step_us"], 9_829.0);
    assert_eq!(replica["baseline"]["step_us"], 10_666.0);

    // Steps of 15,000 us bring the window's average past 5% over the
    // baseline, whether or not it still holds the steps of 9829 us.
    observer.record(TARGET, 0, &steps(15_000, 400));
    let failure = deployment
        .updater
        .read_trial(1, (0, 0))
        .expect("a regression past 5%");
    assert_eq!(
        failure.kind,
        FailureKind::Threshold(Threshold::StepTimeRegression)
    );
    let decided = deployment.status()["trial"]["replicas"][0]["rates"]["step_us"]
        .as_f64()
        .expect("the reading's step time");
    assert!(decided > 10_666.0 * 1.05, "{decided}");
    assert!(
        failure
            .reason
            .contains(&format!("{decided:.0} us against 10666 us")),
        "{failure:?}"
    );

    // The target drains beside the previous epoch, faster: what decided
    // stays what the status states.
    deployment.run();
    observer.record(TARGET, 0, &steps(5_000, 4_000));
    assert_eq!(deployment.updater.read_trial(1, (0, 0)), None);
    let status = deployment.status();
    assert_eq!(status["phase"], "rolled_back");
    assert_eq!(status["failure"]["kind"], "step_time_regression");
    let replica = &status["trial"]["replicas"][0];
    assert_eq!(replica["rates"]["step_us"], decided);
    assert_eq!(replica["baseline"]["step_us"], 10_666.0);
}

/// A target that faults on trial is stopped, not drained, and the previous
/// epoch serves again.
#[test]
fn a_target_that_faults_on_trial_is_stopped_instead_of_drained() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment.apply(update);

    let fault = deployment
        .engines
        .engine(SPAWNED)
        .fault("an illegal address");
    deployment.engines.report(SPAWNED, fault);
    deployment.run();

    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(deployment.status()["failure"]["kind"], "device_fault");
    assert_eq!(
        names(&deployment.engines.received(SPAWNED)),
        ["promote"],
        "a failed target is not commanded again"
    );
    assert!(deployment.engines.stops().contains(&Some(SPAWNED)));
    // It serves again, and grows into what the stopped target's release freed.
    assert_eq!(
        names(&deployment.engines.received(PREVIOUS)),
        ["drain", "promote", "grow"]
    );
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
}

/// A target that exits on trial rolls the update back as a fault does.
#[test]
fn a_target_that_exits_on_trial_rolls_the_update_back() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment.apply(update);

    deployment.engines.exit(HOST, SPAWNED, "signal: 11");
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(deployment.status()["failure"]["kind"], "process_exit");
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
}

/// A target that fails while the switch runs, before the serving engine was
/// drained, rolls back at once: the serving engine still serves, the
/// target is stopped, and the serving engine grows into what its release freed.
#[test]
fn a_target_that_fails_while_switching_leaves_the_serving_engine_serving() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    deployment.engines.start_spawned(BUILD);
    deployment.engines.engine(SPAWNED).withhold();
    deployment.run();

    let fault = deployment
        .engines
        .engine(SPAWNED)
        .fault("an illegal address");
    deployment.engines.report(SPAWNED, fault);
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    // Never drained nor promoted, it only grows into what the stopped target's release freed.
    assert_eq!(names(&deployment.engines.received(PREVIOUS)), ["grow"]);
    assert!(deployment.engines.stops().contains(&Some(SPAWNED)));
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
}

/// An epoch update on trial commits at its trial's end as a decision does: the target serves on.
#[test]
fn an_epoch_trial_commits_at_its_end() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, Duration::ZERO);
    deployment.apply(update);
    deployment.updater.conclude(1);
    deployment.run();
    assert_ne!(deployment.status()["phase"], "rolled_back");
    assert_ne!(*deployment.epochs.served().borrow(), SERVING);
}

/// A previous epoch that keeps its weights alone is stopped when the trial
/// begins; a rollback then needs it started again, and the target drains
/// once its owner says it serves.
#[test]
fn a_previous_epoch_kept_as_weights_is_relaunched_for_a_rollback() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Weights, LONG);
    deployment.apply(update);
    assert_eq!(deployment.engines.stops(), vec![Some(PREVIOUS)]);

    // Its exit lets the node keep the weights alone.
    deployment.engines.exit(HOST, PREVIOUS, "exit status: 0");
    deployment.run();
    let retentions = deployment.engines.retentions();
    assert_eq!(retentions.len(), 1);
    assert_eq!(retentions[0].epochs, vec![SERVING]);
    assert!(!retentions[0].to_host);
    assert_eq!(deployment.status()["previous_epoch_kept"], "weights");

    deployment
        .updater
        .rollback(decision("worse"))
        .expect("an update is on trial");
    deployment.run();
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Switching)));
    assert_eq!(
        deployment
            .updater
            .relaunch_due()
            .into_iter()
            .collect::<Vec<_>>(),
        vec![PREVIOUS]
    );
    assert_eq!(*deployment.epochs.served().borrow(), TARGET);
    assert_eq!(names(&deployment.engines.received(SPAWNED)), ["promote"]);

    // Its owner started it again at a new launch generation: requests return
    // to it, and the target drains.
    deployment.updater.relaunched(PREVIOUS, 12);
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
    deployment.run();
    assert_eq!(
        names(&deployment.engines.received(SPAWNED)),
        ["promote", "drain"]
    );
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert!(deployment.updater.relaunch_due().is_empty());
}

/// A node that cannot move the weights to host storage keeps them on its
/// GPUs, and the status says why.
#[test]
fn a_node_that_refuses_host_storage_keeps_the_weights_on_its_gpus() {
    let mut deployment = Deployment::new(Refusals {
        host_storage: Some("the host holds no room".to_owned()),
        ..Default::default()
    });
    let update = deployment.epoch(1, PreviousEpoch::Host, LONG);
    deployment.apply(update);
    // The previous epoch's engine, which the trial stopped, exits: its node reports the exit as
    // it reports every engine's.
    deployment.engines.exit(HOST, PREVIOUS, "exit status: 0");
    deployment.run();

    let retentions = deployment.engines.retentions();
    assert!(retentions[0].to_host);
    let status = deployment.status();
    assert_eq!(status["previous_epoch_kept"], "weights");
    assert_eq!(status["host_refused"][0], "rank-0: the host holds no room");
    assert_eq!(status["moved_to_host_bytes"], 0);
    assert_eq!(deployment.engines.retentions().len(), 1);
}

/// A serving engine lost during the switch has nothing left to drain: the
/// switch goes on without it, and the loss is the failure domain's to
/// replace.
#[test]
fn a_serving_engine_lost_during_the_switch_is_not_drained() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, LONG);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    deployment.engines.start_spawned(BUILD);
    deployment.engines.engine(SPAWNED).withhold();
    deployment.run();

    deployment.engines.exit(HOST, PREVIOUS, "signal: 9");
    deployment.release(SPAWNED);
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Trial)));
    assert!(deployment.engines.received(PREVIOUS).is_empty());
}

// --- the updater --------------------------------------------------------------------------------

/// A standby whose failure domain was lost is promoted on its process link,
/// with no state handed to it.
#[test]
fn a_standby_is_promoted_on_its_process_link() {
    let mut deployment = Deployment::new(Refusals::default());
    deployment.engines.serve(6);
    deployment.updater.promote(HOST, "engine", 0, 6);
    deployment.run();
    assert_eq!(
        deployment.engines.received(6),
        vec![command::Kind::Promote(process::Promote::default())]
    );
    assert_eq!(
        deployment.engines.engine(6).lifecycle(),
        proto::request::wire::EngineState::Serving
    );
}

/// The marks a rollback left are handed to a successor daemon and refuse the
/// same target there.
#[test]
fn the_unavailable_marks_cross_a_daemon_handover() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 1, LONG);
    deployment.apply(update);
    let fault = deployment.engines.engine(PREVIOUS).fault("worse");
    deployment.engines.report(PREVIOUS, fault);
    deployment.run();
    let marks = deployment.updater.unavailable_marks();
    assert_eq!(marks.len(), 1);
    let reason = marks[0].reason.clone();

    let successor = Deployment::new(Refusals::default());
    assert_eq!(successor.updater.unavailable(&deployment.target), None);
    successor.updater.adopt_unavailable(marks);
    assert_eq!(
        successor
            .updater
            .unavailable(&deployment.target)
            .map(|mark| mark.reason),
        Some(reason)
    );
}
