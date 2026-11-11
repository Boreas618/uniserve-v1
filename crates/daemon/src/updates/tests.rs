//! The epoch layer as its callers see it.
//!
//! Each test drives the [`Driver`] the update entry and the adapter share,
//! over the real [`crate::supervision::roster::ProcessRoster`], with stand-in
//! nodes and main processes behind its link
//! ([`crate::supervision::stand_in`]). What a test reads is what the switch's
//! callers observe: the requests the nodes received (`Spawn`, `Claim`, `Stop`,
//! `KeepWeights`), the commands each main process and the frontend were sent,
//! the update's status and the epoch records and process table
//! `GET /admin/config` reports.
//!
//! The deployment is the one the update entry switches: one node, one main
//! process (`Leader::apply_epoch` refuses more). The serving epoch 11 runs at
//! launch generation 5, which its engine control started; the target epoch 12
//! is spawned at generation 7.

use std::collections::HashMap;
use std::sync::Arc;

use proto::process::{self, command, report};
use serde_json::Value;

use super::in_flight;
use crate::epoch::{Epoch, EpochRecord, Epochs, Failure, ModuleSet, Status, Update, UpdateState};
use crate::leader::UpdateError;
use crate::options::{PreviousEpoch, ServerArgs};
use crate::supervision::stand_in::{self, Nodes, Refusals};
use crate::updates::change::Applied;
use crate::updates::driver::Driver;
use crate::updates::epoch::EpochChange;
use crate::updates::switch::Start;

/// The serving epoch's number, which is also the epoch it executes.
const SERVING: u64 = 11;
/// The target epoch's number.
const TARGET: u64 = 12;
/// The serving epoch's launch generation. Its engine control started it, so
/// its connection is numbered by the engine it states: this generation.
const PREVIOUS: u64 = 5;
/// The launch generation the target's child is spawned at.
const SPAWNED: u64 = 7;
/// The target child's connection to the leader.
const CHILD: u64 = 70;
/// The frontend's connection to the leader.
const FRONTEND: u64 = 90;
/// The build the target's child must report.
const BUILD: &str = "2.0.0+new";

/// A deployment serving epoch 11, its epoch layer, and the stand-in nodes and
/// main processes behind it.
struct Deployment {
    epochs: Arc<Epochs>,
    driver: Driver,
    nodes: Nodes,
    previous: Epoch,
    target: Epoch,
    /// The one node's host identity.
    host: String,
    keeps: PreviousEpoch,
}

impl Deployment {
    /// A deployment whose trial keeps the previous epoch as `keeps`, over
    /// nodes that refuse what `refusals` names.
    fn new(keeps: PreviousEpoch, refusals: Refusals) -> Self {
        let deployment = Self::unrecorded(keeps, refusals);
        deployment
            .epochs
            .record(SERVING, serving_record(&deployment.previous));
        deployment
    }

    /// The same, before the daemon recorded the serving epoch.
    fn unrecorded(keeps: PreviousEpoch, refusals: Refusals) -> Self {
        let epochs = Arc::new(Epochs::new());
        let (nodes, roster) = stand_in::roster(&epochs, refusals);
        nodes.greet_frontend(FRONTEND);
        let previous = Epoch::of(ServerArgs::default()).expect("the configuration resolves");
        // Another executable is a change of what builds the main process:
        // the epoch layer's.
        let mut config = previous.config.clone();
        config.worker_config.executable = "/opt/other-uniserve".into();
        let target = Epoch::with_modules(config, previous.modules.clone())
            .expect("the configuration resolves");
        Self {
            driver: Driver::new(Arc::clone(&epochs), Box::new(roster)),
            epochs,
            nodes,
            host: target.config.host_identity.clone(),
            previous,
            target,
            keeps,
        }
    }

    /// What the update entry hands the epoch arm for this deployment.
    fn change(&self) -> EpochChange {
        EpochChange {
            previous: SERVING,
            target: TARGET,
            update: Update::new(TARGET, self.target.clone()),
            group: "engine".to_owned(),
            start: start(),
            previous_epoch: SERVING,
            previous_generation: PREVIOUS,
            spawn_generation: SPAWNED,
            keeps: self.keeps,
        }
    }

    /// Applies the target as the update entry does: the epoch arm starts the
    /// switch, and the daemon hands the target its modules. Returns the
    /// switch's first report.
    fn apply(&self) -> Value {
        let applied = self
            .driver
            .apply(&self.previous, &self.target, self.change(), false)
            .expect("the epoch layer runs")
            .expect("the epoch layer owns the change");
        let Applied::Epoch(report) = applied else {
            panic!("an epoch change reports the switch, not {applied:?}");
        };
        self.driver.await_modules(&self.host);
        report
    }

    /// The target's child connects and parks on its storage.
    fn park(&self) {
        self.nodes.greet(CHILD, BUILD);
        self.nodes.report(
            CHILD,
            report::Kind::Parked(process::Parked {
                generation: SPAWNED,
            }),
        );
        self.driver.advance();
    }

    /// The daemon handed the parked target its modules.
    fn hand(&self) {
        self.driver.handed(&self.host);
    }

    /// The target's child answers its promotion.
    fn promote(&self) {
        self.nodes.report(
            CHILD,
            report::Kind::Promoted(process::Promoted {
                generation: SPAWNED,
                module_generation: 0,
            }),
        );
        self.driver.advance();
    }

    /// The serving epoch's process ended its requests.
    fn drain(&self) {
        self.drain_with(Vec::new());
    }

    /// The serving epoch's process ended its requests, with the prefix
    /// cache `index` it parked with.
    fn drain_with(&self, index: Vec<proto::request::GroupIndex>) {
        self.nodes.report(
            PREVIOUS,
            report::Kind::Drained(process::Drained {
                epoch: SERVING,
                index,
            }),
        );
        self.driver.advance();
    }

    /// The returning previous epoch's process says it serves again.
    fn previous_serves(&self) {
        self.nodes.report(
            PREVIOUS,
            report::Kind::Promoted(process::Promoted {
                generation: PREVIOUS,
                module_generation: 0,
            }),
        );
        self.driver.advance();
    }

    /// A deployment whose target serves on trial, with the commands sent on
    /// the way there already read.
    fn on_trial(keeps: PreviousEpoch, refusals: Refusals) -> Self {
        let deployment = Self::new(keeps, refusals);
        deployment.apply();
        deployment.park();
        deployment.hand();
        deployment.promote();
        deployment.drain();
        assert_eq!(deployment.status()["phase"], "probation");
        let _ = deployment.nodes.sent();
        deployment
    }

    /// The update as the status reports it.
    fn status(&self) -> Value {
        self.driver.status().expect("an update ran")
    }

    /// Epoch `epoch` as `GET /admin/config` reports it.
    fn epoch(&self, epoch: u64) -> Value {
        let report = self.epochs.report();
        report["epochs"]
            .as_array()
            .expect("the epochs")
            .iter()
            .find(|record| record["epoch"] == epoch)
            .cloned()
            .unwrap_or_else(|| panic!("epoch {epoch} is reported"))
    }

    /// The state of the process of launch generation `generation`, as
    /// `GET /admin/config` reports it.
    fn process(&self, generation: u64) -> Value {
        let report = self.epochs.report();
        report["processes"]
            .as_array()
            .expect("the processes")
            .iter()
            .find(|process| process["generation"] == generation)
            .map(|process| process["state"].clone())
            .unwrap_or_else(|| panic!("generation {generation} is reported"))
    }

    /// The epoch the frontend was last told to serve.
    fn served(&self) -> u64 {
        *self.epochs.served().borrow()
    }

    /// The `Serve` the frontend is sent for `epoch`.
    fn frontend_serves(&self, epoch: u64) -> (u64, command::Kind) {
        (
            FRONTEND,
            command::Kind::Serve(process::Serve {
                epoch,
                requests: self.requests(),
            }),
        )
    }

    /// The `Serve` the previous epoch's parked process is sent to return.
    fn previous_resumes(&self) -> (u64, command::Kind) {
        (
            PREVIOUS,
            command::Kind::Serve(process::Serve {
                epoch: SERVING,
                requests: self.requests(),
            }),
        )
    }

    /// The request socket the frontend submits to.
    fn requests(&self) -> String {
        self.target
            .config
            .request_socket()
            .to_string_lossy()
            .into_owned()
    }
}

/// The serving epoch's record.
fn serving_record(deployment: &Epoch) -> EpochRecord {
    EpochRecord {
        config: deployment.engine.clone(),
        kernels: deployment.modules.kernels.clone(),
        policy: deployment.modules.policy.clone(),
        previous_kernels: Default::default(),
        previous_policy: None,
        status: Status::Serving,
        generation: Some(4),
    }
}

/// How the target's main process is started.
fn start() -> Start {
    Start {
        executable: "/opt/other-uniserve".to_owned(),
        build: Some(BUILD.to_owned()),
        launch: proto::config::EngineArgs {
            engine_config: ServerArgs::default()
                .engine_config()
                .expect("the configuration resolves"),
            predecessor: None,
            expert_order: Default::default(),
            host: "rank-0".to_owned(),
            requests: "unix:/run/requests.sock".to_owned(),
            processes: "unix:/run/processes.sock".to_owned(),
            engine: 0,
            build_server: "/run/modules.sock".to_owned(),
            devices: vec![0],
            group: "engine".to_owned(),
            rank: 0,
            epoch: TARGET,
            scheduler_node: true,
            world: None,
            standby_storage: crate::options::StandbyStorage::Resident,
            watchdog_timeout_s: None,
            rollback_s: None,
            queue_depth: 4,
            world_size: 1,
        },
        environment: HashMap::new(),
    }
}

fn kind(kind: command::Kind) -> (u64, command::Kind) {
    (CHILD, kind)
}

/// The two layers' refusals share one decision: whichever update is in
/// flight, the second change is told which one it collided with.
#[test]
fn a_second_update_is_refused_by_the_one_in_flight() {
    assert_eq!(in_flight(false, false), None);
    assert_eq!(in_flight(true, false), Some("an epoch switch is running"));
    assert_eq!(
        in_flight(false, true),
        Some("a generation update is on probation")
    );
    assert_eq!(in_flight(true, true), Some("an epoch switch is running"));
}

/// Front's order, end to end: the target is started and its storage claimed,
/// it parks, it is promoted only once the daemon handed it its modules, the
/// frontend serves it and the previous epoch drains, the trial begins after
/// the drain, and a commit stops the previous epoch, purges its modules and
/// retires its storage. The revisions the Build Servers selected stay in
/// the target's record through every phase that writes it.
#[test]
fn an_epoch_switch_serves_the_target_and_a_commit_retires_the_previous_epoch() {
    let deployment = Deployment::new(PreviousEpoch::Parked, Refusals::default());

    let report = deployment.apply();
    assert_eq!(report["phase"], "preparing");
    let spawns = deployment.nodes.spawns();
    assert_eq!(spawns.len(), 1);
    assert_eq!(spawns[0].host, deployment.host);
    assert_eq!(spawns[0].generation, SPAWNED);
    assert_eq!(spawns[0].executable, "/opt/other-uniserve");
    let launch: proto::config::EngineArgs =
        serde_json::from_slice(&spawns[0].launch).expect("the launch file the node writes");
    assert_eq!(
        launch.engine, SPAWNED,
        "the launch names the child's own number"
    );
    let claims = deployment.nodes.claims();
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].epoch, TARGET);
    assert!(claims[0].retire.is_empty());
    assert_eq!(deployment.epoch(TARGET)["status"], "precopy");
    assert_eq!(deployment.process(SPAWNED), "starting");

    let mut modules = ModuleSet {
        kernels: deployment.target.modules.kernels.clone(),
        policy: deployment.target.modules.policy.clone(),
    };
    modules.policy.revision = 3;
    deployment.driver.pin(&modules);

    // A parked child waits for its modules: no promotion before the daemon
    // handed them.
    deployment.park();
    assert_eq!(
        deployment.nodes.sent(),
        vec![kind(command::Kind::Park(process::Park {}))]
    );
    assert_eq!(deployment.process(SPAWNED), "parked");
    assert_eq!(deployment.status()["phase"], "preparing");
    deployment.hand();
    assert_eq!(
        deployment.nodes.sent(),
        vec![kind(command::Kind::Promote(process::Promote {
            inherits: true
        }))]
    );

    // Once promoted, the frontend serves the target and the previous epoch
    // drains; the trial waits for the drain.
    deployment.promote();
    assert_eq!(
        deployment.nodes.sent(),
        vec![
            deployment.frontend_serves(TARGET),
            (
                PREVIOUS,
                command::Kind::Drain(process::Drain { epoch: SERVING })
            ),
        ]
    );
    assert_eq!(deployment.served(), TARGET);
    assert_eq!(deployment.epoch(TARGET)["status"], "serving");
    assert_eq!(deployment.process(SPAWNED), "serving");
    assert_eq!(deployment.status()["phase"], "serving");

    // The index the previous epoch drained with goes to the target, whose
    // planes hold its pages.
    let index = vec![proto::request::GroupIndex {
        worker_group: "text".to_owned(),
        pages: vec![proto::request::wire::IndexedPage {
            group: 0,
            hash: vec![7; 32],
            tokens: vec![1, 2, 3, 4],
            units: vec![9],
            resume: None,
        }],
        ..Default::default()
    }];
    deployment.drain_with(index.clone());
    assert_eq!(
        deployment.nodes.sent(),
        vec![kind(command::Kind::Inherit(process::Inherit { index }))]
    );
    assert_eq!(deployment.epoch(TARGET)["status"], "probation");
    let status = deployment.status();
    assert_eq!(status["phase"], "probation");
    assert!(
        status["pause_ms"].is_u64(),
        "the switchover's pause: {status}"
    );
    assert_eq!(status["previous_epoch_kept"], "parked");
    assert!(
        deployment.nodes.stops().is_empty(),
        "`parked` keeps the previous epoch's process through the trial"
    );

    assert!(deployment.driver.commit());
    assert_eq!(
        deployment.nodes.sent(),
        vec![kind(command::Kind::Commit(process::Commit {}))]
    );
    deployment
        .nodes
        .report(CHILD, report::Kind::Committed(process::Committed {}));
    deployment.driver.advance();
    assert_eq!(
        deployment.nodes.sent(),
        vec![kind(command::Kind::Purge(process::Purge {
            generation: PREVIOUS
        }))]
    );
    assert_eq!(deployment.nodes.stops(), vec![Some(PREVIOUS)]);
    let claims = deployment.nodes.claims();
    assert_eq!(claims.len(), 2);
    assert_eq!(claims[1].epoch, TARGET);
    assert_eq!(claims[1].retire, vec![SERVING]);

    assert!(
        !deployment.driver.running(),
        "a committed switch frees its slot"
    );
    assert_eq!(deployment.status()["phase"], "purged");
    let target = deployment.epoch(TARGET);
    assert_eq!(target["status"], "serving");
    assert_eq!(target["policy"]["revision"], 3);
    assert_eq!(
        deployment.epochs.update().map(|update| update.state),
        Some(UpdateState::Committed)
    );
}

/// The answer to an apply comes once the target serves, so the operator may
/// decide while the drain that starts the trial is in flight. The decision is
/// kept for the trial: a commit is sent, and a rollback drains the target,
/// only once the previous epoch drained.
#[test]
fn an_operators_decision_before_the_trial_is_carried_out_when_the_trial_starts() {
    for commit in [true, false] {
        let deployment = Deployment::new(PreviousEpoch::Parked, Refusals::default());
        deployment.apply();
        deployment.park();
        deployment.hand();
        deployment.promote();
        let _ = deployment.nodes.sent();

        if commit {
            assert!(deployment.driver.commit());
        } else {
            assert!(deployment.driver.rollback("the operator rolls back"));
        }
        assert!(
            deployment.nodes.sent().is_empty(),
            "nothing before the drain"
        );
        assert_eq!(deployment.status()["phase"], "serving");

        deployment.drain();
        let expected = if commit {
            command::Kind::Commit(process::Commit {})
        } else {
            command::Kind::Drain(process::Drain { epoch: TARGET })
        };
        // The target hears the drained index first: it holds the requests
        // that continue from the previous epoch's pages until then.
        let inherit = command::Kind::Inherit(process::Inherit { index: Vec::new() });
        assert_eq!(deployment.nodes.sent(), vec![kind(inherit), kind(expected)]);
    }
}

/// A node that cannot start the target drops it before anything else
/// starts: nothing is claimed, the previous epoch serves on, and the update
/// reverts without a failure, so applying it again needs no `force`.
#[test]
fn a_refused_spawn_abandons_the_target_and_the_previous_epoch_serves_on() {
    let deployment = Deployment::new(
        PreviousEpoch::Parked,
        Refusals {
            spawn: Some("no device is free".to_owned()),
            ..Refusals::default()
        },
    );

    let report = deployment.apply();
    assert_eq!(report["phase"], "abandoned");
    assert_eq!(report["failure"]["kind"], "spawn_failed");
    assert_eq!(report["failure"]["reason"], "no device is free");
    assert!(deployment.nodes.claims().is_empty());
    assert_eq!(deployment.nodes.stops(), vec![Some(SPAWNED)]);
    assert_eq!(deployment.epoch(TARGET)["status"], "purged");
    assert_eq!(deployment.epoch(SERVING)["status"], "serving");

    deployment.driver.advance();
    assert!(!deployment.driver.running());
    assert!(deployment.nodes.sent().is_empty(), "nobody was commanded");
    assert_eq!(deployment.served(), 0, "the frontend was never moved");
    let update = deployment.epochs.update().expect("the update ended");
    assert_eq!(update.state, UpdateState::Reverted);
    assert_eq!(deployment.epochs.unavailable(&deployment.target), None);
}

/// A node that refuses the target's claim holds nothing, so the child it
/// already started is stopped, and its exit is the stop the epoch layer
/// asked for rather than a failure.
#[test]
fn a_refused_claim_stops_the_target_it_started() {
    let deployment = Deployment::new(
        PreviousEpoch::Parked,
        Refusals {
            claim: Some("the storage server is full".to_owned()),
            ..Refusals::default()
        },
    );

    let report = deployment.apply();
    assert_eq!(report["phase"], "abandoned");
    assert_eq!(report["failure"]["kind"], "claim_failed");
    assert_eq!(report["failure"]["reason"], "the storage server is full");
    assert_eq!(deployment.nodes.spawns().len(), 1);
    assert_eq!(deployment.nodes.stops(), vec![Some(SPAWNED)]);
    assert_eq!(deployment.epoch(TARGET)["status"], "purged");

    deployment
        .nodes
        .exit(&deployment.host, SPAWNED, "signal: 15");
    deployment.driver.advance();
    assert_eq!(deployment.process(SPAWNED), "stopped");
    assert!(!deployment.driver.running());
}

/// A target that faults before the frontend serves it never becomes the
/// epoch: it is stopped and never promoted, the switch does not wait for a
/// readiness it will not send, and the frontend stays on the previous epoch.
#[test]
fn a_target_that_faults_before_it_serves_is_abandoned() {
    let deployment = Deployment::new(PreviousEpoch::Parked, Refusals::default());
    deployment.apply();
    deployment.park();
    let _ = deployment.nodes.sent();

    deployment.nodes.report(
        CHILD,
        report::Kind::Fault(process::Fault {
            kind: "illegal_address".to_owned(),
            reason: "an illegal memory access".to_owned(),
            device: 0,
        }),
    );
    deployment.driver.advance();
    let status = deployment.status();
    assert_eq!(status["phase"], "abandoned");
    assert_eq!(status["failure"]["kind"], "illegal_address");
    assert_eq!(deployment.nodes.stops(), vec![Some(SPAWNED)]);
    assert_eq!(deployment.epoch(TARGET)["status"], "purged");

    // The modules arriving afterwards promote nothing.
    deployment.hand();
    assert!(deployment.nodes.sent().is_empty());
    assert_eq!(deployment.served(), 0);
    assert!(!deployment.driver.running());
}

/// The daemon's own executable's build is known before the child connects:
/// a child of another build is stopped before it counts as the epoch's
/// process, and it is neither parked nor promoted.
#[test]
fn a_child_of_another_build_is_stopped_before_it_is_promoted() {
    let deployment = Deployment::new(PreviousEpoch::Parked, Refusals::default());
    deployment.apply();

    deployment.nodes.greet(CHILD, "1.0.0+old");
    deployment.nodes.report(
        CHILD,
        report::Kind::Parked(process::Parked {
            generation: SPAWNED,
        }),
    );
    deployment.driver.advance();
    deployment.hand();
    let status = deployment.status();
    assert_eq!(status["phase"], "abandoned");
    assert_eq!(status["failure"]["kind"], "build_mismatch");
    assert!(deployment.nodes.stops().contains(&Some(SPAWNED)));
    assert!(
        deployment.nodes.sent().is_empty(),
        "the child is not commanded"
    );
    assert_eq!(deployment.served(), 0);
}

/// A target that exits on probation cannot acknowledge a revert: the
/// previous epoch's parked process is told to serve again at once, the
/// frontend moves back only once it says it serves, and the target is
/// stopped, purged and marked unavailable with its exit.
#[test]
fn a_target_that_exits_on_probation_returns_the_previous_epoch() {
    let deployment = Deployment::on_trial(PreviousEpoch::Parked, Refusals::default());

    deployment
        .nodes
        .exit(&deployment.host, SPAWNED, "signal: 9");
    deployment.driver.advance();
    assert_eq!(deployment.process(SPAWNED), "failed: signal: 9");
    assert_eq!(
        deployment.nodes.sent(),
        vec![deployment.previous_resumes()],
        "the dead target is not commanded"
    );
    assert_eq!(deployment.status()["phase"], "reverting");
    assert_eq!(
        deployment.served(),
        TARGET,
        "the frontend waits for the return"
    );

    deployment.previous_serves();
    assert_eq!(
        deployment.nodes.sent(),
        vec![deployment.frontend_serves(SERVING)]
    );
    assert_eq!(deployment.served(), SERVING);
    assert_eq!(deployment.nodes.stops(), vec![Some(SPAWNED)]);
    let claims = deployment.nodes.claims();
    let returned = claims.last().expect("the previous epoch's claim");
    assert_eq!(returned.epoch, SERVING);
    assert_eq!(returned.retire, vec![TARGET]);
    assert_eq!(deployment.epoch(TARGET)["status"], "purged");
    assert_eq!(deployment.process(PREVIOUS), "serving");

    assert!(!deployment.driver.running());
    let status = deployment.status();
    assert_eq!(status["phase"], "abandoned");
    assert_eq!(status["failure"]["kind"], "process_exit");
    let unavailable = deployment
        .epochs
        .unavailable(&deployment.target)
        .expect("a target that failed needs `force`");
    assert_eq!(unavailable.kind, "process_exit");
}

/// A threshold of the trial reverts with the threshold as the update's first
/// failure, and the target drains: it did not fail, so its requests end
/// migratable rather than lost. The previous epoch serves again once the
/// target drained, and the frontend once the previous epoch says it serves.
#[test]
fn a_regressed_trial_drains_the_target_and_returns_the_previous_epoch() {
    let deployment = Deployment::on_trial(PreviousEpoch::Parked, Refusals::default());

    let regression = Failure {
        kind: "stalled".to_owned(),
        reason: "the decode job of batch 3 ran 9000 ms".to_owned(),
        device: None,
    };
    assert!(deployment.driver.fail_trial(regression));
    assert_eq!(
        deployment.nodes.sent(),
        vec![kind(command::Kind::Drain(process::Drain { epoch: TARGET }))]
    );
    assert_eq!(deployment.status()["phase"], "reverting");

    deployment.nodes.report(
        CHILD,
        report::Kind::Drained(process::Drained {
            epoch: TARGET,
            index: Vec::new(),
        }),
    );
    deployment.driver.advance();
    assert_eq!(deployment.nodes.sent(), vec![deployment.previous_resumes()]);
    assert_eq!(deployment.served(), TARGET);

    deployment.previous_serves();
    assert_eq!(
        deployment.nodes.sent(),
        vec![deployment.frontend_serves(SERVING)]
    );
    assert_eq!(deployment.nodes.stops(), vec![Some(SPAWNED)]);
    assert_eq!(deployment.epoch(TARGET)["status"], "purged");
    assert!(!deployment.driver.running());
    let unavailable = deployment
        .epochs
        .unavailable(&deployment.target)
        .expect("a regressed target needs `force`");
    assert_eq!(unavailable.kind, "stalled");

    let late = Failure {
        kind: "stalled".to_owned(),
        reason: "read after the update ended".to_owned(),
        device: None,
    };
    assert!(
        !deployment.driver.fail_trial(late),
        "a threshold read after the update ended decides nothing"
    );
}

/// `host` stops the previous epoch's process for the trial and asks its node
/// to move the weights it alone retains to host storage; a node that refuses
/// keeps them on its GPUs, and the status says `weights`, with its reason.
#[test]
fn a_node_that_refuses_host_storage_keeps_the_weights_on_its_gpus() {
    let refusal = "the host cannot hold 1 GiB more";
    let deployment = Deployment::on_trial(
        PreviousEpoch::Host,
        Refusals {
            host_storage: Some(refusal.to_owned()),
            ..Refusals::default()
        },
    );
    assert_eq!(deployment.nodes.stops(), vec![Some(PREVIOUS)]);
    assert_eq!(deployment.status()["previous_epoch_kept"], "host");
    assert!(
        deployment.nodes.retentions().is_empty(),
        "weights are kept once the process exited"
    );

    deployment
        .nodes
        .exit(&deployment.host, PREVIOUS, "signal: 15");
    deployment.driver.advance();
    assert_eq!(deployment.process(PREVIOUS), "stopped");
    let retentions = deployment.nodes.retentions();
    assert_eq!(retentions.len(), 1);
    assert_eq!(retentions[0].host, deployment.host);
    assert_eq!(retentions[0].epochs, vec![SERVING]);
    assert!(retentions[0].to_host);

    let status = deployment.status();
    assert_eq!(status["phase"], "probation", "the trial goes on");
    assert_eq!(status["previous_epoch_kept"], "weights");
    assert_eq!(
        status["host_refused"],
        serde_json::json!([format!("{}: {refusal}", deployment.host)])
    );
    assert_eq!(status["moved_to_host_bytes"], 0);
}

/// A revert of an epoch kept as its weights stops the target first, since
/// the previous epoch's start takes the storage it held, and asks the epoch's
/// owner to start it again; the frontend moves once the owner says the
/// relaunched process serves.
#[test]
fn a_revert_relaunches_a_previous_epoch_whose_processes_were_stopped() {
    let deployment = Deployment::on_trial(PreviousEpoch::Weights, Refusals::default());
    deployment
        .nodes
        .exit(&deployment.host, PREVIOUS, "signal: 15");
    deployment.driver.advance();
    assert_eq!(deployment.status()["previous_epoch_kept"], "weights");
    assert_eq!(deployment.nodes.retentions().len(), 1);

    assert!(deployment.driver.rollback("the operator rolls back"));
    assert_eq!(
        deployment.nodes.sent(),
        vec![kind(command::Kind::Drain(process::Drain { epoch: TARGET }))]
    );
    deployment.nodes.report(
        CHILD,
        report::Kind::Drained(process::Drained {
            epoch: TARGET,
            index: Vec::new(),
        }),
    );
    deployment.driver.advance();
    assert_eq!(
        deployment.nodes.stops(),
        vec![Some(PREVIOUS), Some(SPAWNED)],
        "the target is stopped before the previous epoch starts again"
    );
    assert_eq!(
        deployment.driver.relaunch_due(),
        [deployment.host.clone()].into()
    );
    assert!(
        deployment.nodes.sent().is_empty(),
        "a stopped process is not resumed"
    );
    assert_eq!(deployment.epoch(TARGET)["status"], "purged");
    assert_eq!(deployment.served(), TARGET);

    deployment.driver.relaunched(&deployment.host, 9);
    assert_eq!(
        deployment.nodes.sent(),
        vec![deployment.frontend_serves(SERVING)]
    );
    assert_eq!(deployment.served(), SERVING);
    assert!(!deployment.driver.running());
    assert_eq!(deployment.status()["phase"], "abandoned");
}

/// A change of nothing, of the host layer or of the modules alone is not the
/// epoch layer's: `apply` falls through to its other path, and no process is
/// started and no record written.
#[test]
fn a_change_of_another_layer_starts_no_switch() {
    let deployment = Deployment::new(PreviousEpoch::Parked, Refusals::default());
    let previous = &deployment.previous;

    let mut logged = previous.config.clone();
    logged.logging_config.log_requests = true;
    let host =
        Epoch::with_modules(logged, previous.modules.clone()).expect("the configuration resolves");
    let mut package = previous.clone();
    package.modules.policy.revision += 1;

    for target in [previous, &host, &package] {
        let applied = deployment
            .driver
            .apply(previous, target, deployment.change(), false)
            .expect("another layer's change is not refused");
        assert!(applied.is_none());
    }
    assert!(deployment.nodes.requests().is_empty());
    assert!(!deployment.driver.running());
    assert!(deployment.epochs.get(TARGET).is_none());
}

/// A dry run plans the switch and changes nothing: no switch, no record, no
/// child.
#[test]
fn a_dry_run_plans_and_starts_nothing() {
    let deployment = Deployment::new(PreviousEpoch::Parked, Refusals::default());
    let applied = deployment
        .driver
        .apply(
            &deployment.previous,
            &deployment.target,
            deployment.change(),
            true,
        )
        .expect("a dry run plans")
        .expect("the epoch layer owns the change");
    let Applied::Planned(plan) = applied else {
        panic!("a dry run plans, not {applied:?}");
    };
    assert_eq!(plan["change"], "epoch");
    assert_eq!(plan["epoch"], TARGET);
    assert_eq!(plan["generation"], SPAWNED);
    assert!(deployment.nodes.requests().is_empty());
    assert!(!deployment.driver.running());
    assert!(
        deployment.epochs.get(TARGET).is_none(),
        "a plan writes no record"
    );
}

/// A serving epoch the daemon cannot describe is refused: there is no
/// previous generation to drain or resume, and one is not invented.
#[test]
fn a_serving_epoch_without_a_record_is_refused() {
    let deployment = Deployment::unrecorded(PreviousEpoch::Parked, Refusals::default());
    let error = deployment
        .driver
        .apply(
            &deployment.previous,
            &deployment.target,
            deployment.change(),
            false,
        )
        .expect_err("no serving record");
    assert!(matches!(error, UpdateError::Refused(_)), "{error}");
    assert!(deployment.nodes.requests().is_empty());
    assert!(!deployment.driver.running());
}

/// One change at a time: a switch already running refuses another, which
/// starts nothing.
#[test]
fn one_switch_runs_at_a_time() {
    let deployment = Deployment::new(PreviousEpoch::Parked, Refusals::default());
    deployment.apply();

    let error = deployment
        .driver
        .apply(
            &deployment.previous,
            &deployment.target,
            deployment.change(),
            false,
        )
        .expect_err("another change is in progress");
    assert!(matches!(error, UpdateError::InProgress), "{error}");
    assert_eq!(deployment.nodes.spawns().len(), 1);
}

/// A commit, a rollback or a trial's failure with no switch running does
/// nothing and says so, so the entry can fall through to the generation layer's
/// probation.
#[test]
fn no_switch_means_no_decision() {
    let deployment = Deployment::new(PreviousEpoch::Parked, Refusals::default());
    assert!(!deployment.driver.commit());
    assert!(!deployment.driver.rollback("nothing runs"));
    assert!(!deployment.driver.fail_trial(Failure {
        kind: "stalled".to_owned(),
        reason: "nothing runs".to_owned(),
        device: None,
    }));
    assert!(deployment.driver.report().is_none());
    assert!(deployment.nodes.requests().is_empty());
}
