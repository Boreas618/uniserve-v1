//! The standby rules over fake standbys: what the daemon does to a
//! domain's engines for each lifecycle event.

use std::collections::BTreeMap;

use super::*;
use crate::epoch::{Failure, Status};
use proto::request::StandbyStorage;

/// The serving generation the fake standbys's processes are at.
const SERVING: u64 = 7;

/// A process whose next engine's generation the caller took one after
/// the serving one, as the counter gives it.
fn process(host: &str, epoch: Option<u64>) -> MainProcess {
    MainProcess {
        host: host.to_owned(),
        group: format!("main-{host}"),
        rank: 0,
        generation: SERVING,
        standby_generation: SERVING + 1,
        epoch,
    }
}

/// Facts of nodes whose devices have the share a serving process leaves.
fn roomy(hosts: &[&str]) -> BTreeMap<String, StandbyRoom> {
    hosts
        .iter()
        .map(|host| {
            (
                (*host).to_owned(),
                StandbyRoom {
                    crowded: false,
                    fits_suspended: true,
                },
            )
        })
        .collect()
}

fn options(storage: StandbyStorage) -> StandbyOptions {
    StandbyOptions {
        enabled: true,
        storage,
    }
}

fn failure(kind: &str) -> Failure {
    Failure {
        kind: kind.to_owned(),
        reason: String::new(),
        device: Some(0),
    }
}

fn spawn(host: &str, generation: u64, storage: Storage) -> Action {
    Action::Spawn {
        host: host.to_owned(),
        group: format!("main-{host}"),
        rank: 0,
        generation,
        storage,
    }
}

fn allocate(host: &str) -> Action {
    Action::Allocate {
        host: host.to_owned(),
        group: format!("main-{host}"),
        rank: 0,
    }
}

fn promote(host: &str, generation: u64) -> Action {
    Action::Promote {
        host: host.to_owned(),
        generation,
    }
}

fn stop(host: &str, generation: u64) -> Action {
    Action::Stop {
        host: host.to_owned(),
        group: format!("main-{host}"),
        generation,
    }
}

fn relaunch(host: &str, generation: u64) -> Action {
    Action::Relaunch {
        host: host.to_owned(),
        group: format!("main-{host}"),
        rank: 0,
        generation,
    }
}

fn terminate(host: &str, generation: u64) -> Action {
    Action::Terminate {
        host: host.to_owned(),
        generation,
    }
}

fn migrate(host: &str) -> Action {
    Action::Migrate {
        host: host.to_owned(),
    }
}

/// Two domains: `a` alone, and `b` and `c` sharing cross-node epoch 3.
fn standbys() -> Standbys {
    Standbys::new(vec![
        process("a", None),
        process("b", Some(3)),
        process("c", Some(3)),
    ])
}

/// Starts the standby of every domain.
fn start(standbys: &mut Standbys) {
    let actions = standbys.serve(
        Status::Serving,
        options(StandbyStorage::Auto),
        &roomy(&["a", "b", "c"]),
    );
    assert_eq!(
        actions,
        vec![
            spawn("a", SERVING + 1, Storage::Resident),
            spawn("b", SERVING + 1, Storage::Resident),
            spawn("c", SERVING + 1, Storage::Resident),
        ]
    );
}

/// Reports every standby as prepared and parked.
fn parked(standbys: &mut Standbys) {
    for host in ["a", "b", "c"] {
        standbys.parked(host, SERVING + 1);
    }
}

/// The generations the caller takes for the next engine of `hosts`, as
/// its counter gives them.
fn allocated(standbys: &mut Standbys, hosts: &[&str], generation: u64) {
    for host in hosts {
        standbys.allocated(host, generation);
    }
}

/// The hosts of each domain, as the standbys derives them.
fn domains(standbys: &Standbys) -> Vec<Vec<String>> {
    standbys
        .domains()
        .into_iter()
        .map(|domain| {
            domain
                .into_iter()
                .map(|process| process.host.clone())
                .collect()
        })
        .collect()
}

#[test]
fn the_domains_follow_the_cross_node_epochs() {
    let standbys = Standbys::new(vec![
        process("a", None),
        process("b", Some(3)),
        process("c", Some(3)),
        process("d", None),
        process("e", Some(5)),
    ]);
    assert_eq!(
        domains(&standbys),
        vec![
            vec!["a".to_owned()],
            vec!["b".to_owned(), "c".to_owned()],
            vec!["d".to_owned()],
            vec!["e".to_owned()],
        ]
    );
}

#[test]
fn a_serving_epoch_starts_one_standby_a_generation_ahead_of_each_member() {
    let mut standbys = standbys();
    start(&mut standbys);
    // The standbys exist: serving again starts none beside them.
    let again = standbys.serve(
        Status::Serving,
        options(StandbyStorage::Auto),
        &roomy(&["a", "b", "c"]),
    );
    assert!(again.is_empty());
    let phases: Vec<&str> = standbys
        .state()
        .iter()
        .map(|domain| domain.phase.as_str())
        .collect();
    assert_eq!(phases, vec!["starting", "starting"]);
}

#[test]
fn the_standby_generation_is_the_epoch_layers_number_and_is_never_computed() {
    let mut standbys = Standbys::new(vec![
        MainProcess {
            standby_generation: 1000,
            ..process("a", None)
        },
        MainProcess {
            standby_generation: 2001,
            ..process("b", Some(3))
        },
        MainProcess {
            standby_generation: 2002,
            ..process("c", Some(3))
        },
    ]);
    let actions = standbys.serve(
        Status::Serving,
        options(StandbyStorage::Auto),
        &roomy(&["a", "b", "c"]),
    );
    assert_eq!(
        actions,
        vec![
            spawn("a", 1000, Storage::Resident),
            spawn("b", 2001, Storage::Resident),
            spawn("c", 2002, Storage::Resident),
        ],
        "the rules carry the counter's numbers verbatim, even far above serving"
    );
}

#[test]
fn an_epoch_that_does_not_serve_keeps_no_standby() {
    let mut standbys = standbys();
    for status in [Status::Preparing, Status::Trial, Status::Released] {
        let actions = standbys.serve(
            status,
            options(StandbyStorage::Auto),
            &roomy(&["a", "b", "c"]),
        );
        assert!(actions.is_empty(), "{status:?} keeps a standby");
    }
    let disabled = standbys.serve(
        Status::Serving,
        StandbyOptions {
            enabled: false,
            storage: StandbyStorage::Auto,
        },
        &roomy(&["a", "b", "c"]),
    );
    assert!(disabled.is_empty(), "worker_config.standby is off");
}

#[test]
fn a_parked_domain_is_promoted_on_a_sticky_fault_and_a_new_standby_follows() {
    let mut standbys = standbys();
    start(&mut standbys);
    parked(&mut standbys);
    let phases: Vec<&str> = standbys
        .state()
        .iter()
        .map(|domain| domain.phase.as_str())
        .collect();
    assert_eq!(phases, vec!["parked", "parked"]);

    // One promotion per parked engine of the failing domain, all in one step;
    // the node in no cross-node group keeps its own standby.
    let actions = standbys.failed("b", &failure("sticky"));
    assert_eq!(
        actions,
        vec![promote("b", SERVING + 1), promote("c", SERVING + 1)]
    );

    // A further fault in the same round changes nothing: the first wins.
    assert!(standbys.failed("c", &failure("sticky")).is_empty());

    assert!(standbys.promoted("b", SERVING + 1).is_empty());
    // The last acknowledgement stops the old generation of the domain's members.
    assert_eq!(
        standbys.promoted("c", SERVING + 1),
        vec![stop("b", SERVING), stop("c", SERVING)]
    );

    // The promoted generation is `SERVING + 1`, the generation the epoch
    // record now names; the caller takes the next number from its
    // counter, so the domain's next standby is at `SERVING + 2`, never beside
    // the old generation.
    allocated(&mut standbys, &["b", "c"], SERVING + 2);
    let actions = standbys.serve(
        Status::Serving,
        options(StandbyStorage::Auto),
        &roomy(&["a", "b", "c"]),
    );
    assert_eq!(
        actions,
        vec![
            spawn("b", SERVING + 2, Storage::Resident),
            spawn("c", SERVING + 2, Storage::Resident),
        ]
    );
}

#[test]
fn a_request_level_failure_promotes_nothing() {
    let mut standbys = standbys();
    start(&mut standbys);
    parked(&mut standbys);
    for kind in ["non_finite_logits", "policy_reverted", "input_error"] {
        assert!(
            standbys.failed("b", &failure(kind)).is_empty(),
            "{kind} replaced the domain"
        );
    }
    let phases: Vec<&str> = standbys
        .state()
        .iter()
        .map(|domain| domain.phase.as_str())
        .collect();
    assert_eq!(phases, vec!["parked", "parked"]);
    // The domain still promotes on a process-fatal fault.
    assert_eq!(
        standbys.failed("b", &failure("sticky")),
        vec![promote("b", SERVING + 1), promote("c", SERVING + 1)]
    );
}

#[test]
fn a_domain_with_a_member_that_has_no_parked_standby_relaunches_wholly() {
    let mut standbys = standbys();
    start(&mut standbys);
    // `b` parked; `c` is still starting.
    standbys.parked("b", SERVING + 1);
    // The caller takes the next number for each member's replacement.
    allocated(&mut standbys, &["b", "c"], SERVING + 2);
    let actions = standbys.failed("b", &failure("illegal_address"));
    assert_eq!(
        actions,
        vec![
            terminate("b", SERVING + 1),
            relaunch("b", SERVING + 2),
            terminate("c", SERVING + 1),
            stop("c", SERVING),
            relaunch("c", SERVING + 2),
        ],
        "a member without a parked standby relaunches its whole domain"
    );

    standbys.relaunched("b", SERVING + 2);
    standbys.relaunched("c", SERVING + 2);
    allocated(&mut standbys, &["b", "c"], SERVING + 3);
    let actions = standbys.serve(
        Status::Serving,
        options(StandbyStorage::Auto),
        &roomy(&["a", "b", "c"]),
    );
    assert_eq!(
        actions,
        vec![
            spawn("b", SERVING + 3, Storage::Resident),
            spawn("c", SERVING + 3, Storage::Resident),
        ]
    );
}

#[test]
fn a_replacement_waits_for_the_epoch_layers_generation_numbers() {
    let mut standbys = standbys();
    start(&mut standbys);
    // Both standbys are gone; their numbers were spent when they started.
    standbys.standby_lost("b");
    assert_eq!(standbys.standby_lost("c"), vec![]);
    // The fault cannot start the replacement: the rules do not mint a number.
    assert_eq!(
        standbys.failed("b", &failure("sticky")),
        vec![allocate("b"), allocate("c")]
    );
    assert_eq!(standbys.state()[1].phase.as_str(), "allocating");
    // The first number is not enough; the round starts with the last one.
    assert!(standbys.allocated("b", 41).is_empty());
    // The failed node's engine is gone, so it is not stopped; its sibling
    // still serves at the old generation and is stopped before its
    // replacement starts.
    assert_eq!(
        standbys.allocated("c", 42),
        vec![relaunch("b", 41), stop("c", SERVING), relaunch("c", 42)]
    );
    assert_eq!(standbys.state()[1].phase.as_str(), "relaunching");
}

#[test]
fn a_parked_standby_that_dies_leaves_the_domain_to_a_relaunch() {
    let mut standbys = standbys();
    start(&mut standbys);
    parked(&mut standbys);
    // A standby dies (its device faulted): the domain's standby epoch is
    // one, so the other members' standbys are dropped with it.
    assert_eq!(
        standbys.standby_lost("c"),
        vec![terminate("b", SERVING + 1)]
    );
    allocated(&mut standbys, &["b", "c"], SERVING + 2);
    let actions = standbys.failed("c", &failure("process_exit"));
    assert!(
        actions
            .iter()
            .all(|action| !matches!(action, Action::Promote { .. })),
        "nothing is promoted without a parked standby"
    );
    assert!(actions.contains(&relaunch("b", SERVING + 2)));
    assert!(actions.contains(&relaunch("c", SERVING + 2)));
}

#[test]
fn an_epoch_that_stops_serving_retires_its_standby_and_its_domain_goes_on_short() {
    let mut standbys = standbys();
    start(&mut standbys);
    parked(&mut standbys);
    // The epoch stops serving on `c`: the standby rules own that standby and
    // stop it, so the caller never needs its generation.
    assert_eq!(standbys.retired("c"), vec![terminate("c", SERVING + 1)]);
    // A fault of the retired node replaces nothing.
    assert!(standbys.failed("c", &failure("sticky")).is_empty());
    // The domain's remaining member is promoted alone.
    assert_eq!(
        standbys.failed("b", &failure("sticky")),
        vec![promote("b", SERVING + 1)]
    );
    assert_eq!(
        standbys.promoted("b", SERVING + 1),
        vec![stop("b", SERVING)]
    );
    // Serving again starts no standby on the retired node.
    allocated(&mut standbys, &["b"], SERVING + 2);
    let actions = standbys.serve(
        Status::Serving,
        options(StandbyStorage::Auto),
        &roomy(&["a", "b", "c"]),
    );
    assert_eq!(actions, vec![spawn("b", SERVING + 2, Storage::Resident)]);
}

#[test]
fn a_device_that_needs_a_reset_takes_its_standby_and_migrates_the_node() {
    let mut standbys = standbys();
    start(&mut standbys);
    parked(&mut standbys);
    allocated(&mut standbys, &["b"], SERVING + 2);
    // `c`'s GPU needs a reset: its standby is dead, the node cannot run
    // again, and the rest of its domain is started without it.
    let actions = standbys.unusable("c");
    assert_eq!(
        actions,
        vec![
            terminate("c", SERVING + 1),
            terminate("b", SERVING + 1),
            stop("b", SERVING),
            relaunch("b", SERVING + 2),
            migrate("c"),
        ]
    );
    // The dead node's fault replaces nothing: it is already gone.
    assert!(standbys.failed("c", &failure("sticky")).is_empty());
}

#[test]
fn every_member_is_asked_to_promote_before_any_acknowledgement_is_awaited() {
    let mut standbys = standbys();
    start(&mut standbys);
    parked(&mut standbys);
    // The whole round's promotions are returned in one call, so the caller
    // issues them all before it awaits any acknowledgement (a suspended epoch
    // restores its communication together).
    let actions = standbys.failed("c", &failure("launch_failed"));
    assert_eq!(
        actions,
        vec![promote("b", SERVING + 1), promote("c", SERVING + 1)]
    );
}

#[test]
fn a_promotion_that_fails_relaunches_the_domain() {
    let mut standbys = standbys();
    start(&mut standbys);
    parked(&mut standbys);
    standbys.failed("b", &failure("sticky"));
    standbys.promoted("b", SERVING + 1);
    allocated(&mut standbys, &["b", "c"], SERVING + 2);
    let actions = standbys.promote_failed("c");
    // The node whose promotion failed has no serving engine to stop; every
    // other member of the domain is stopped and started again.
    assert_eq!(
        actions,
        vec![
            terminate("b", SERVING + 1),
            stop("b", SERVING),
            relaunch("b", SERVING + 2),
            terminate("c", SERVING + 1),
            relaunch("c", SERVING + 2),
        ]
    );
    assert!(
        actions
            .iter()
            .all(|action| !matches!(action, Action::Spawn { .. })),
        "no standby starts while the domain relaunches"
    );
}

#[test]
fn one_crowded_device_suspends_the_whole_domain() {
    let mut standbys = standbys();
    let mut facts = roomy(&["a", "b", "c"]);
    facts.get_mut("b").expect("b's facts").crowded = true;
    let actions = standbys.serve(Status::Serving, options(StandbyStorage::Auto), &facts);
    assert_eq!(
        actions,
        vec![
            spawn("a", SERVING + 1, Storage::Resident),
            spawn("b", SERVING + 1, Storage::Suspended),
            spawn("c", SERVING + 1, Storage::Suspended),
        ],
        "b's crowded device suspends b and c together, never a alone"
    );
}

#[test]
fn a_standby_that_cannot_fit_even_suspended_is_not_kept() {
    let mut standbys = standbys();
    let mut facts = roomy(&["a", "b", "c"]);
    let c = facts.get_mut("c").expect("c's facts");
    c.crowded = true;
    c.fits_suspended = false;
    let actions = standbys.serve(Status::Serving, options(StandbyStorage::Auto), &facts);
    assert_eq!(
        actions,
        vec![spawn("a", SERVING + 1, Storage::Resident)],
        "the b/c domain keeps no standby when c cannot hold one"
    );
    // Without a standby the domain is replaced by a relaunch, with the
    // numbers the caller already took (nothing consumed them).
    let actions = standbys.failed("b", &failure("sticky"));
    assert!(actions.contains(&relaunch("b", SERVING + 1)));
    assert!(actions.contains(&relaunch("c", SERVING + 1)));
}

#[test]
fn a_standby_that_fails_to_start_is_left_out() {
    let mut standbys = standbys();
    start(&mut standbys);
    assert!(standbys.standby_lost("a").is_empty());
    assert_eq!(standbys.state()[0].phase.as_str(), "absent");
    allocated(&mut standbys, &["a"], SERVING + 2);
    let actions = standbys.failed("a", &failure("out_of_memory"));
    assert_eq!(actions, vec![relaunch("a", SERVING + 2)]);
    standbys.relaunched("a", SERVING + 2);
    allocated(&mut standbys, &["a"], SERVING + 3);
    let actions = standbys.serve(
        Status::Serving,
        options(StandbyStorage::Auto),
        &roomy(&["a", "b", "c"]),
    );
    assert_eq!(actions, vec![spawn("a", SERVING + 3, Storage::Resident)]);
}

#[test]
fn a_standby_report_of_another_generation_does_not_park_it() {
    let mut standbys = standbys();
    start(&mut standbys);
    standbys.parked("a", SERVING);
    assert_eq!(standbys.state()[0].phase.as_str(), "starting");
    standbys.parked("a", SERVING + 1);
    assert_eq!(standbys.state()[0].phase.as_str(), "parked");
}
