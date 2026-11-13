//! The update's phase machine against the fake engines at the process
//! interface (`testkit::process`): every transition, failure and rollback
//! path of both forms, as the commands the engines were sent, the requests
//! the nodes were asked, the epoch records and the status report state them.

use std::sync::Arc;
use std::time::{Duration, Instant};

use proto::process::{self, command};
use proto::report::SchedulerStats;
use testkit::process::Refusals;

use crate::epoch::{Epoch, EpochRecord, Epochs, Failure, PackageRevision, Status};
use crate::observe::Observer;
use crate::options::{PreviousEpoch, RollbackConfig, ServerArgs, UpdateConfig};
use crate::supervision::processes::Processes;
use crate::test_support::Engines;
use crate::updates::trial::{Baseline, Trial};
use crate::updates::update::{EpochSwitch, Node, Phase, Update};
use crate::updates::{Transition, Updater};

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
}

impl Deployment {
    /// A deployment that serves epoch [`SERVING`] on one engine, whose nodes
    /// refuse what `refusals` names.
    fn new(refusals: Refusals) -> Self {
        let epochs = Arc::new(Epochs::new());
        let (mut engines, link) = Engines::link(refusals);
        engines.serve(PREVIOUS);
        let processes = Processes::new(
            Arc::clone(&epochs),
            "deployment".to_owned(),
            Vec::new(),
            link,
        );
        let updater = Updater::new(Arc::clone(&epochs), processes);
        let transitions = updater.transitions().expect("the log is read once");

        let serving = Epoch::of(ServerArgs::default()).expect("the configuration resolves");
        epochs.record(
            SERVING,
            EpochRecord {
                config: serving.engine.clone(),
                engine: serving.modules.engine.clone(),
                kernels: serving.modules.kernels.clone(),
                policy: serving.modules.policy.clone(),
                previous_kernels: Default::default(),
                previous_policy: None,
                status: Status::Serving,
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
        }
    }

    /// The serving engines a generation update moves: one, or two.
    fn nodes(&mut self, engines: usize) -> Vec<Node> {
        let mut nodes = vec![Node::new(HOST.to_owned(), 0, PREVIOUS, PREVIOUS)];
        if engines == 2 {
            self.engines.serve(SECOND);
            nodes.push(Node::new(HOST.to_owned(), 1, SECOND, SECOND));
        }
        nodes
    }

    /// A generation update numbered `number` over `engines` serving engines.
    fn generation(&mut self, number: u64, engines: usize, deadline: Option<Instant>) -> Update {
        let nodes = self.nodes(engines);
        Update::generation(
            number,
            self.target.clone(),
            Arc::clone(&self.epochs),
            SERVING,
            "engine".to_owned(),
            nodes,
            deadline,
        )
    }

    /// An epoch update numbered `number` from the one serving engine to one
    /// target engine on the same node, whose previous epoch keeps `keeps`.
    fn epoch(&self, number: u64, keeps: PreviousEpoch, deadline: Option<Instant>) -> Update {
        let mut switch = self.switch(vec![(target_launch(HOST, 0, SPAWNED), Vec::new())]);
        switch.keeps = keeps;
        let nodes = vec![Node::new(HOST.to_owned(), 0, PREVIOUS, PREVIOUS)];
        self.epoch_over(number, nodes, switch, deadline)
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
            environment: Default::default(),
            requests: "unix:/run/requests.sock".to_owned(),
            keeps: PreviousEpoch::Parked,
            recompute: false,
            drain: Duration::from_secs_f64(UpdateConfig::default().drain_s),
            shrink: false,
            retire: Vec::new(),
        }
    }

    /// An epoch update numbered `number` from the serving engines `nodes` to
    /// `switch`'s targets.
    fn epoch_over(
        &self,
        number: u64,
        nodes: Vec<Node>,
        switch: EpochSwitch,
        deadline: Option<Instant>,
    ) -> Update {
        Update::epoch(
            number,
            self.target.clone(),
            Arc::clone(&self.epochs),
            SERVING,
            "engine".to_owned(),
            nodes,
            switch,
            deadline,
        )
    }

    /// Sends what the engine of launch generation `launch` withheld, and runs
    /// the update on.
    fn release(&mut self, launch: u64) {
        for report in self.engines.engine(launch).release() {
            self.engines.report(launch, report);
        }
        self.run();
    }

    /// Applies `update` and runs it as far as the fakes take it.
    fn apply(&mut self, update: Update) -> serde_json::Value {
        let report = self.updater.apply(update).expect("no update is in flight");
        self.run();
        report
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

    /// The update as the status reports it.
    fn status(&self) -> serde_json::Value {
        self.updater.status().expect("an update ran")
    }

    /// The record of epoch `epoch`.
    fn record(&self, epoch: u64) -> EpochRecord {
        self.epochs.get(epoch).expect("the epoch's record")
    }
}

/// The launch file of the target engine `launch` on `host`, whose first rank
/// is group rank `rank`: it starts parked, succeeding the serving epoch.
fn target_launch(host: &str, rank: u32, launch: u64) -> proto::config::EngineArgs {
    let options = ServerArgs::default();
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
            command::Kind::Serve(_) => "serve",
            command::Kind::Grow(_) => "grow",
        })
        .collect()
}

fn decision(reason: &str) -> Failure {
    Failure {
        kind: "rollback".to_owned(),
        reason: reason.to_owned(),
        device: None,
    }
}

// --- the generation form ------------------------------------------------------------------------

/// A generation update prepares, switches and serves on trial; a commit
/// records the new modules as the epoch's and tells the engine to release
/// the kept generation.
#[test]
fn a_generation_update_serves_on_trial_and_a_commit_keeps_it() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 1, None);
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
    assert_eq!(record.status, Status::Trial);
    assert_eq!(record.policy, deployment.target.modules.policy);
    assert_eq!(
        record.previous_policy,
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
    assert_eq!(record.status, Status::Serving);
    assert_eq!(record.policy, deployment.target.modules.policy);
    assert_eq!(record.previous_policy, None);
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
    let update = deployment.generation(1, 1, None);
    deployment.apply(update);

    assert_eq!(deployment.entered(), ["preparing", "abandoned"]);
    assert_eq!(deployment.engines.engine(PREVIOUS).generation(), 1);
    let record = deployment.record(SERVING);
    assert_eq!(record.status, Status::Serving);
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

/// With two serving engines the switch begins only once both prepared, and
/// one that cannot prepare abandons the update for both.
#[test]
fn every_engine_acknowledges_before_the_phase_moves() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 2, None);
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
    let update = deployment.generation(1, 2, None);
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
    let update = deployment.generation(1, 1, None);
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
    assert_eq!(record.status, Status::Serving);
    assert_eq!(record.policy, deployment.serving.modules.policy);
    assert_eq!(record.previous_policy, None);
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
    let update = deployment.generation(1, 2, None);
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
    let update = deployment.generation(1, 1, None);
    deployment.apply(update);

    let report = deployment
        .updater
        .rollback(decision("the responses got worse"))
        .expect("an update is on trial");
    assert_eq!(report["phase"], "switching");
    assert_eq!(report["rollback"], true);
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

/// A trial not committed by its deadline is rolled back by the daemon; one
/// committed before it is not.
#[test]
fn a_trial_not_committed_by_its_deadline_rolls_back() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 1, Some(Instant::now()));
    deployment.apply(update);
    assert_eq!(deployment.status()["phase"], "trial");

    deployment.updater.expire(1);
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(deployment.status()["failure"]["kind"], "deadline");
    assert_eq!(deployment.engines.engine(PREVIOUS).generation(), 1);

    // A deadline that has not passed decides nothing, and a commit ends it.
    let mut deployment = Deployment::new(Refusals::default());
    let later = Instant::now() + Duration::from_secs(3600);
    let update = deployment.generation(2, 1, Some(later));
    deployment.apply(update);
    deployment.updater.expire(2);
    assert_eq!(deployment.updater.in_flight(), Some((2, Phase::Trial)));
    assert!(deployment.updater.commit().is_some());
    deployment.updater.expire(2);
    assert_eq!(deployment.status()["phase"], "committed");
}

/// One update is in flight at a time, and a commit asked before the trial is
/// applied when the trial begins.
#[test]
fn one_update_is_in_flight_and_an_early_commit_is_kept() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.generation(1, 1, None);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    let second = deployment.generation(2, 1, None);
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
    let third = deployment.generation(3, 1, None);
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
    let update = deployment.generation(1, 1, None);
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

/// An epoch update starts the target from a launch file that states the
/// epoch's module set and the engine it succeeds, and switches once it
/// parked: the target is promoted first, and new requests go to it from its
/// `Serving`; then the serving engine drains beside it, with the deadline
/// `update_config.drain_s` states. The trial keeps the previous epoch parked,
/// and a commit stops it and retires its storage.
#[test]
fn an_epoch_update_promotes_the_target_and_then_drains_the_serving_engine() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
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
        })]
    );

    // Trial: the previous epoch is kept parked.
    assert_eq!(deployment.entered(), ["preparing", "switching", "trial"]);
    assert_eq!(deployment.record(TARGET).status, Status::Trial);
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
    assert_eq!(deployment.record(TARGET).status, Status::Serving);
    assert_eq!(deployment.engines.stops(), vec![Some(PREVIOUS)]);
    let retiring = deployment
        .engines
        .claims()
        .pop()
        .expect("the commit's claim");
    assert_eq!((retiring.epoch, retiring.retire), (TARGET, vec![SERVING]));
    assert_eq!(
        deployment.epochs.serving().map(|(epoch, _)| epoch),
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
    switch.shrink = true;
    let nodes = deployment.nodes(1);
    let update = deployment.epoch_over(1, nodes, switch, None);
    deployment.apply(update);

    assert_eq!(
        deployment.engines.received(PREVIOUS),
        [command::Kind::Drain(process::Drain {
            epoch: SERVING,
            deadline_ms: 2_500,
            shrink: true,
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
        switch.shrink = shrink;
        let nodes = deployment.nodes(1);
        let update = deployment.epoch_over(1, nodes, switch, None);
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
    switch.shrink = true;
    let nodes = deployment.nodes(1);
    let update = deployment.epoch_over(1, nodes, switch, None);
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

/// The epoch form covers every node of both epochs: a target whose world
/// spans two hosts starts one engine on each, its scheduler node inheriting
/// the world's sockets, every node claims the target's storage, every target
/// is promoted and every serving engine on every node drained.
#[test]
fn an_epoch_update_covers_every_node_of_both_epochs() {
    let mut deployment = Deployment::new(Refusals::default());
    deployment.engines.serve(SECOND);
    let nodes = vec![
        Node::new(HOST.to_owned(), 0, PREVIOUS, PREVIOUS),
        Node::new(OTHER.to_owned(), 2, SECOND, PREVIOUS),
    ];
    let switch = deployment.switch(vec![
        (target_launch(HOST, 0, SPAWNED), vec![1, 2]),
        (target_launch(OTHER, 2, SPAWNED_PEER), Vec::new()),
    ]);
    let update = deployment.epoch_over(1, nodes, switch, None);
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
    let retired: Vec<_> = deployment
        .engines
        .claims()
        .iter()
        .filter(|claim| claim.retire == [SERVING])
        .map(|claim| claim.host.clone())
        .collect();
    assert_eq!(retired, [HOST, OTHER]);
}

/// The two epochs need not have as many replicas: two serving replicas move
/// to one target engine, both drained once it serves.
#[test]
fn an_epoch_update_moves_two_replicas_to_one() {
    let mut deployment = Deployment::new(Refusals::default());
    let nodes = deployment.nodes(2);
    let mut switch = deployment.switch(vec![(target_launch(HOST, 0, SPAWNED), Vec::new())]);
    switch.shrink = true;
    let update = deployment.epoch_over(1, nodes, switch, None);
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
    let update = deployment.epoch_over(1, nodes, switch, None);
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
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
    deployment.apply(update);
    assert_eq!(deployment.status()["phase"], "abandoned");
    assert_eq!(deployment.status()["failure"]["kind"], "spawn_failed");
    assert!(deployment.engines.received(PREVIOUS).is_empty());
    assert_eq!(deployment.record(TARGET).status, Status::Released);

    let mut deployment = Deployment::new(Refusals {
        claim: Some("the Storage Server refused".to_owned()),
        ..Default::default()
    });
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
    deployment.apply(update);
    assert_eq!(deployment.status()["failure"]["kind"], "claim_failed");
    assert_eq!(deployment.engines.stops(), vec![Some(SPAWNED)]);
    assert!(deployment.engines.received(PREVIOUS).is_empty());
    assert_eq!(*deployment.epochs.served().borrow(), 0);
}

/// A target that exits while it prepares abandons the update, and the epoch
/// it would have been is retired by the next update's claim.
#[test]
fn a_target_lost_while_preparing_abandons_the_update() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    deployment.engines.exit(HOST, SPAWNED, "signal: 9");
    deployment.updater.advance();

    assert_eq!(deployment.status()["phase"], "abandoned");
    assert_eq!(deployment.status()["failure"]["kind"], "process_exit");
    assert!(deployment.engines.received(PREVIOUS).is_empty());
    assert_eq!(deployment.updater.take_retire(), vec![TARGET]);
}

/// A rollback asked while the target prepares abandons the update.
#[test]
fn a_rollback_before_the_switch_abandons_an_epoch_update() {
    let deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
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
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
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
        })
    );
    assert_eq!(deployment.entered(), ["switching", "rolled_back"]);
    assert_eq!(deployment.engines.stops(), vec![Some(SPAWNED)]);
    assert_eq!(deployment.record(TARGET).status, Status::Released);
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
    let observer = Arc::new(Observer::default());
    let trial = |observer: &Arc<Observer>, switch: Instant| {
        let replica = Baseline::new(
            Arc::clone(observer),
            TARGET,
            SERVING,
            thresholds.window,
            switch,
        );
        Trial::begin(thresholds.clone(), vec![replica], (0, 0))
    };
    let mut deployment = Deployment::new(Refusals::default());
    assert_eq!(deployment.updater.read_trial(1, (0, 0)), None, "no update");

    observer.record(SERVING, 0, &steps(10_666, 400));
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    deployment.engines.start_spawned(BUILD);
    deployment.engines.engine(SPAWNED).withhold();
    deployment.run();
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Switching)));
    let switch = deployment.updater.switch_began().expect("the switch began");

    // The target's first batches, while the serving epoch still steps.
    observer.record(TARGET, 0, &steps(29_744, 3));
    assert!(
        !deployment.updater.begin_trial(trial(&observer, switch)),
        "the switch has not ended"
    );
    assert_eq!(deployment.updater.read_trial(1, (0, 0)), None);

    deployment.release(SPAWNED);
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Trial)));
    std::thread::sleep(Duration::from_millis(2));
    assert!(deployment.updater.begin_trial(trial(&observer, switch)));
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
    assert_eq!(failure.kind, "step_time_regression");
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
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
    deployment.apply(update);

    let fault = deployment
        .engines
        .engine(SPAWNED)
        .fault("an illegal address");
    deployment.engines.report(SPAWNED, fault);
    deployment.run();

    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(deployment.status()["failure"]["kind"], "sticky");
    assert_eq!(
        names(&deployment.engines.received(SPAWNED)),
        ["promote"],
        "a failed target is not commanded again"
    );
    assert!(deployment.engines.stops().contains(&Some(SPAWNED)));
    assert_eq!(
        names(&deployment.engines.received(PREVIOUS)),
        ["drain", "promote"]
    );
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
}

/// A target that exits on trial rolls the update back as a fault does.
#[test]
fn a_target_that_exits_on_trial_rolls_the_update_back() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
    deployment.apply(update);

    deployment.engines.exit(HOST, SPAWNED, "signal: 11");
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(deployment.status()["failure"]["kind"], "process_exit");
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
}

/// A target that fails while the switch runs, before the serving engine was
/// drained, rolls back at once: the serving engine still serves, and the
/// target is stopped.
#[test]
fn a_target_that_fails_while_switching_leaves_the_serving_engine_serving() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
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
    assert!(deployment.engines.received(PREVIOUS).is_empty());
    assert!(deployment.engines.stops().contains(&Some(SPAWNED)));
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
}

/// A deadline rolls an epoch update on trial back as a decision does.
#[test]
fn a_deadline_rolls_an_epoch_update_back() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, Some(Instant::now()));
    deployment.apply(update);
    deployment.updater.expire(1);
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(deployment.status()["failure"]["kind"], "deadline");
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
}

/// A previous epoch that keeps its weights alone is stopped when the trial
/// begins; a rollback then needs it started again, and the target drains
/// once its owner says it serves.
#[test]
fn a_previous_epoch_kept_as_weights_is_relaunched_for_a_rollback() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Weights, None);
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
    let update = deployment.epoch(1, PreviousEpoch::Host, None);
    deployment.apply(update);
    deployment.updater.previous_ended(PREVIOUS);

    let retentions = deployment.engines.retentions();
    assert!(retentions[0].to_host);
    let status = deployment.status();
    assert_eq!(status["previous_epoch_kept"], "weights");
    assert_eq!(status["host_refused"][0], "rank-0: the host holds no room");
    assert_eq!(status["moved_to_host_bytes"], 0);

    // The node is asked once, whoever reports the engine's end.
    deployment.engines.exit(HOST, PREVIOUS, "exit status: 0");
    deployment.run();
    assert_eq!(deployment.engines.retentions().len(), 1);
}

/// A serving engine lost during the switch has nothing left to drain: the
/// switch goes on without it, and the loss is the failure domain's to
/// replace.
#[test]
fn a_serving_engine_lost_during_the_switch_is_not_drained() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
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
    let update = deployment.generation(1, 1, None);
    deployment.apply(update);
    deployment.updater.rollback(decision("worse"));
    deployment.run();
    let marks = deployment.updater.unavailable_marks();
    assert_eq!(marks.len(), 1);

    let successor = Deployment::new(Refusals::default());
    assert_eq!(successor.updater.unavailable(&deployment.target), None);
    successor.updater.adopt_unavailable(marks);
    assert_eq!(
        successor
            .updater
            .unavailable(&deployment.target)
            .map(|mark| mark.reason),
        Some("worse".to_owned())
    );
}
