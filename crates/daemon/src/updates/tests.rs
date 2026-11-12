//! The update's phase machine against the fake engines at the process
//! interface (`testkit::process`): every transition, failure and rollback
//! path of both forms, as the commands the engines were sent, the requests
//! the nodes were asked, the epoch records and the status report state them.

use std::sync::Arc;
use std::time::{Duration, Instant};

use proto::process::{self, command};
use testkit::process::Refusals;

use crate::epoch::{Epoch, EpochRecord, Epochs, Failure, PackageRevision, Status};
use crate::options::{PreviousEpoch, ServerArgs};
use crate::supervision::processes::Processes;
use crate::test_support::Engines;
use crate::updates::update::{EpochSwitch, Node, Phase, Update};
use crate::updates::{Transition, Updater};

/// The epoch that serves when a test begins, and the epoch form's target.
const SERVING: u64 = 1;
const TARGET: u64 = 2;

/// The launch generations of the serving engine, of a second serving engine
/// and of the epoch form's target.
const PREVIOUS: u64 = 4;
const SECOND: u64 = 5;
const SPAWNED: u64 = 9;

/// The build the fake engines greet with.
const BUILD: &str = "2.0.0+target";

const HOST: &str = "rank-0";

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
        let mut nodes = vec![Node::new(HOST.to_owned(), 0, PREVIOUS)];
        if engines == 2 {
            self.engines.serve(SECOND);
            nodes.push(Node::new(HOST.to_owned(), 1, SECOND));
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

    /// An epoch update numbered `number` whose previous epoch keeps `keeps`.
    fn epoch(&self, number: u64, keeps: PreviousEpoch, deadline: Option<Instant>) -> Update {
        let mut record = self.epochs.get(SERVING).expect("the serving record");
        record.generation = None;
        let mut launch =
            proto::config::EngineArgs::new(self.target.engine.clone(), self.target.modules.clone());
        launch.epoch = TARGET;
        launch.host = HOST.to_owned();
        // The target starts parked: it names the engine it succeeds.
        launch.predecessor = Some(proto::request::wire::Predecessor {
            epoch: SERVING,
            segments: Vec::new(),
        });
        Update::epoch(
            number,
            self.target.clone(),
            Arc::clone(&self.epochs),
            SERVING,
            "engine".to_owned(),
            vec![Node::new(HOST.to_owned(), 0, PREVIOUS)],
            EpochSwitch {
                epoch: TARGET,
                record,
                launch,
                environment: Default::default(),
                spawn_generation: SPAWNED,
                requests: "unix:/run/requests.sock".to_owned(),
                keeps,
                recompute: false,
                retire: Vec::new(),
            },
            deadline,
        )
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

/// A threshold decides only an update on trial, and rolls it back with the
/// threshold as its failure.
#[test]
fn a_threshold_rolls_back_only_an_update_on_trial() {
    let mut deployment = Deployment::new(Refusals::default());
    let regression = Failure {
        kind: "step_time_regression".to_owned(),
        reason: "steps take 900 us against 600 us".to_owned(),
        device: None,
    };
    assert!(
        !deployment.updater.fail_trial(regression.clone()),
        "no update is in flight"
    );

    let update = deployment.generation(1, 1, None);
    deployment
        .updater
        .apply(update)
        .expect("no update is in flight");
    assert!(
        !deployment.updater.fail_trial(regression.clone()),
        "the update still prepares"
    );
    deployment.run();
    assert!(deployment.updater.fail_trial(regression));
    deployment.run();
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(
        deployment.status()["failure"]["kind"],
        "step_time_regression"
    );
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
    // The slot is free again.
    let third = deployment.generation(3, 1, None);
    assert!(deployment.updater.apply(third).is_ok());
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
/// epoch's module set and the engine it succeeds, switches once it parked
/// (drain, then the promotion with what the drain handed over), serves on
/// trial with the previous epoch parked, and a commit stops the previous
/// epoch and retires its storage.
#[test]
fn an_epoch_update_serves_the_target_and_a_commit_retires_the_previous_epoch() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
    let report = deployment.apply(update);
    assert_eq!(report["change"], "epoch");
    assert_eq!(report["phase"], "preparing");

    // Preparing: one spawn, whose launch states the modules, and the claim.
    let spawns = deployment.engines.spawns();
    assert_eq!(spawns.len(), 1);
    assert_eq!(spawns[0].host, HOST);
    assert_eq!(spawns[0].generation, SPAWNED);
    let launch: proto::config::EngineArgs =
        serde_json::from_slice(&spawns[0].launch).expect("a launch file");
    assert_eq!(launch.engine, SPAWNED);
    assert_eq!(launch.epoch, TARGET);
    assert_eq!(launch.modules, deployment.target.modules);
    assert!(launch.parked());
    assert_eq!(deployment.engines.claims()[0].epoch, TARGET);

    // Switching: the serving engine drained before the target was promoted,
    // and the promotion carried the handoff.
    assert_eq!(deployment.entered(), ["preparing", "switching", "trial"]);
    assert_eq!(names(&deployment.engines.received(PREVIOUS)), ["drain"]);
    let promoted = deployment.engines.received(SPAWNED);
    assert_eq!(names(&promoted), ["promote"]);
    assert!(matches!(
        &promoted[0],
        command::Kind::Promote(process::Promote {
            handoff: Some(handoff)
        }) if !handoff.grows && !handoff.recompute
    ));

    // Trial: requests go to the target, and the previous epoch is kept parked.
    assert_eq!(*deployment.epochs.served().borrow(), TARGET);
    assert_eq!(deployment.record(TARGET).status, Status::Trial);
    assert_eq!(deployment.record(TARGET).generation, Some(1));
    assert_eq!(deployment.record(TARGET).config.executable, BUILD);
    assert!(deployment.engines.stops().is_empty());
    let status = deployment.status();
    assert_eq!(status["phase"], "trial");
    assert_eq!(status["previous_epoch_kept"], "parked");
    assert!(status["pause_ms"].is_number());

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
    assert_eq!(retiring.epoch, TARGET);
    assert_eq!(retiring.retire, vec![SERVING]);
    assert_eq!(
        deployment.epochs.serving().map(|(epoch, _)| epoch),
        Some(TARGET)
    );
}

/// The switch waits for the daemon's own warming as well as for the engines.
#[test]
fn the_switch_waits_for_the_warming() {
    let mut deployment = Deployment::new(Refusals::default());
    let mut update = deployment.epoch(1, PreviousEpoch::Parked, None);
    update.await_warming(HOST);
    deployment.apply(update);
    assert_eq!(deployment.updater.prepared(), Some(true));
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Preparing)));
    assert!(deployment.engines.received(PREVIOUS).is_empty());

    deployment
        .updater
        .warmed(HOST, Some(serde_json::json!({ "rounds": 2 })));
    deployment.run();
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Trial)));
    assert_eq!(deployment.status()["warming"]["rounds"], 2);
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

/// A rollback by decision on trial drains the target, promotes the parked
/// previous epoch again with no state handed to it, and requests return to
/// it once it serves.
#[test]
fn a_rollback_on_trial_returns_to_the_parked_previous_epoch() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Parked, None);
    deployment.apply(update);
    deployment.entered();

    deployment
        .updater
        .rollback(decision("the model no longer responds usefully"))
        .expect("an update is on trial");
    deployment.run();

    assert_eq!(deployment.entered(), ["switching", "rolled_back"]);
    assert_eq!(
        names(&deployment.engines.received(SPAWNED)),
        ["promote", "drain"]
    );
    let previous = deployment.engines.received(PREVIOUS);
    assert_eq!(names(&previous), ["drain", "promote"]);
    assert_eq!(
        previous[1],
        command::Kind::Promote(process::Promote { handoff: None })
    );
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
    assert_eq!(deployment.engines.stops(), vec![Some(SPAWNED)]);
    assert_eq!(deployment.record(TARGET).status, Status::Released);
    // The target's storage is retired on the node that keeps the previous epoch.
    let retiring = deployment
        .engines
        .claims()
        .pop()
        .expect("the rollback's claim");
    assert_eq!(retiring.epoch, SERVING);
    assert_eq!(retiring.retire, vec![TARGET]);
    assert!(deployment.updater.unavailable(&deployment.target).is_some());
}

/// A target that faults on trial is stopped, not drained, and the previous
/// epoch serves again.
#[test]
fn a_fault_on_trial_rolls_an_epoch_update_back() {
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
    assert_eq!(
        deployment.engines.stops(),
        vec![Some(SPAWNED), Some(SPAWNED)]
    );
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
/// begins; a rollback then needs it started again, and ends once its owner
/// says it serves.
#[test]
fn a_previous_epoch_kept_as_weights_is_relaunched_for_a_rollback() {
    let mut deployment = Deployment::new(Refusals::default());
    let update = deployment.epoch(1, PreviousEpoch::Weights, None);
    deployment.apply(update);
    assert_eq!(deployment.engines.stops(), vec![Some(PREVIOUS)]);
    // The promotion said the target's cache grows into the storage that frees.
    assert!(matches!(
        &deployment.engines.received(SPAWNED)[0],
        command::Kind::Promote(process::Promote {
            handoff: Some(handoff)
        }) if handoff.grows
    ));

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
        vec![HOST.to_owned()]
    );
    assert_eq!(*deployment.epochs.served().borrow(), TARGET);

    // Its owner started it again at a new launch generation.
    deployment.updater.relaunched(HOST, 12);
    assert_eq!(deployment.status()["phase"], "rolled_back");
    assert_eq!(*deployment.epochs.served().borrow(), SERVING);
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
    deployment.updater.previous_ended(HOST);

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

/// A serving engine lost during the switch has nothing to hand over: the
/// target is promoted without its state, and the loss is the failure
/// domain's to replace.
#[test]
fn a_serving_engine_lost_during_the_switch_hands_nothing_over() {
    let mut deployment = Deployment::new(Refusals::default());
    let mut update = deployment.epoch(1, PreviousEpoch::Parked, None);
    update.await_warming(HOST);
    deployment.apply(update);

    deployment.engines.exit(HOST, PREVIOUS, "signal: 9");
    deployment.updater.warmed(HOST, None);
    deployment.run();
    assert_eq!(deployment.updater.in_flight(), Some((1, Phase::Trial)));
    assert_eq!(names(&deployment.engines.received(SPAWNED)), ["promote"]);
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
        vec![command::Kind::Promote(process::Promote { handoff: None })]
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
