//! The standby rules over a fake roster (§2.5): what the daemon does to a
//! domain's children for each lifecycle event.

use std::collections::BTreeMap;

use super::*;
use crate::epoch::{Failure, Status};
use proto::request::StandbyStorage;

/// The serving generation the fake roster's processes are at.
const SERVING: u64 = 7;

fn process(host: &str, world: Option<u64>) -> MainProcess {
    MainProcess {
        host: host.to_owned(),
        group: format!("main-{host}"),
        rank: 0,
        generation: SERVING,
        world,
    }
}

/// Facts of nodes whose devices have the share a serving process leaves.
fn roomy(hosts: &[&str]) -> BTreeMap<String, StorageFacts> {
    hosts
        .iter()
        .map(|host| {
            (
                (*host).to_owned(),
                StorageFacts {
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

/// Two domains: `a` alone, and `b` and `c` sharing cross-node world 3.
fn roster() -> Roster {
    Roster::new(vec![
        process("a", None),
        process("b", Some(3)),
        process("c", Some(3)),
    ])
}

/// Starts the standby of every domain.
fn start(roster: &mut Roster) {
    let actions = roster.serve(
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
fn parked(roster: &mut Roster) {
    for host in ["a", "b", "c"] {
        roster.parked(host, SERVING + 1);
    }
}

/// The hosts of each domain, as the roster derives them.
fn domains(roster: &Roster) -> Vec<Vec<String>> {
    roster
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
fn the_domains_follow_the_cross_node_worlds() {
    let roster = Roster::new(vec![
        process("a", None),
        process("b", Some(3)),
        process("c", Some(3)),
        process("d", None),
        process("e", Some(5)),
    ]);
    assert_eq!(
        domains(&roster),
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
    let mut roster = roster();
    start(&mut roster);
    // The standbys exist: serving again starts none beside them.
    let again = roster.serve(
        Status::Serving,
        options(StandbyStorage::Auto),
        &roomy(&["a", "b", "c"]),
    );
    assert!(again.is_empty());
    let phases: Vec<&str> = roster
        .state()
        .iter()
        .map(|domain| domain.phase.as_str())
        .collect();
    assert_eq!(phases, vec!["starting", "starting"]);
}

#[test]
fn an_epoch_that_does_not_serve_keeps_no_standby() {
    let mut roster = roster();
    for status in [Status::Precopy, Status::Probation, Status::Purged] {
        let actions = roster.serve(
            status,
            options(StandbyStorage::Auto),
            &roomy(&["a", "b", "c"]),
        );
        assert!(actions.is_empty(), "{status:?} keeps a standby");
    }
    let disabled = roster.serve(
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
    let mut roster = roster();
    start(&mut roster);
    parked(&mut roster);
    let phases: Vec<&str> = roster
        .state()
        .iter()
        .map(|domain| domain.phase.as_str())
        .collect();
    assert_eq!(phases, vec!["parked", "parked"]);

    // One promotion per parked child of the failing domain, all in one step;
    // the node in no cross-node group keeps its own standby.
    let actions = roster.failed("b", &failure("sticky"));
    assert_eq!(
        actions,
        vec![promote("b", SERVING + 1), promote("c", SERVING + 1)]
    );

    // A further fault in the same round changes nothing: the first wins.
    assert!(roster.failed("c", &failure("sticky")).is_empty());

    assert!(roster.promoted("b", SERVING + 1).is_empty());
    // The last answer stops the old generation of the domain's members.
    assert_eq!(
        roster.promoted("c", SERVING + 1),
        vec![stop("b", SERVING), stop("c", SERVING)]
    );

    // The promoted generation is `SERVING + 1`, the generation the epoch
    // record now names; the domain's next standby is therefore at
    // `SERVING + 2`, never beside the old generation.
    let actions = roster.serve(
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
    let mut roster = roster();
    start(&mut roster);
    parked(&mut roster);
    for kind in ["non_finite_logits", "policy_reverted", "input_error"] {
        assert!(
            roster.failed("b", &failure(kind)).is_empty(),
            "{kind} replaced the domain"
        );
    }
    let phases: Vec<&str> = roster
        .state()
        .iter()
        .map(|domain| domain.phase.as_str())
        .collect();
    assert_eq!(phases, vec!["parked", "parked"]);
    // The domain still promotes on a process-fatal fault.
    assert_eq!(
        roster.failed("b", &failure("sticky")),
        vec![promote("b", SERVING + 1), promote("c", SERVING + 1)]
    );
}

#[test]
fn a_domain_with_a_member_that_has_no_parked_standby_relaunches_wholly() {
    let mut roster = roster();
    start(&mut roster);
    // `b` parked; `c` is still starting.
    roster.parked("b", SERVING + 1);
    let actions = roster.failed("b", &failure("illegal_address"));
    assert_eq!(
        actions,
        vec![
            terminate("b", SERVING + 1),
            relaunch("b", SERVING + 1),
            terminate("c", SERVING + 1),
            stop("c", SERVING),
            relaunch("c", SERVING + 1),
        ],
        "a member without a parked standby relaunches its whole domain"
    );

    roster.relaunched("b", SERVING + 1);
    roster.relaunched("c", SERVING + 1);
    let actions = roster.serve(
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
fn a_parked_standby_that_dies_leaves_the_domain_to_a_relaunch() {
    let mut roster = roster();
    start(&mut roster);
    parked(&mut roster);
    // A standby dies (its device faulted): the domain's standby world is
    // one, so the other members' standbys are dropped with it.
    assert_eq!(roster.standby_lost("c"), vec![terminate("b", SERVING + 1)]);
    let actions = roster.failed("c", &failure("process_exit"));
    assert!(
        actions
            .iter()
            .all(|action| !matches!(action, Action::Promote { .. })),
        "nothing is promoted without a parked standby"
    );
    assert!(actions.contains(&relaunch("b", SERVING + 1)));
    assert!(actions.contains(&relaunch("c", SERVING + 1)));
}

#[test]
fn a_standby_whose_preparation_failed_drops_the_domains_standbys() {
    let mut roster = roster();
    start(&mut roster);
    // `b` reports that its preparation failed (no `Parked`): the domain's
    // standby world is kept whole
    // or not at all, so `c`'s standby is dropped with it.
    assert_eq!(roster.standby_lost("b"), vec![terminate("c", SERVING + 1)]);
    assert_eq!(roster.state()[1].phase.as_str(), "absent");
    // The next fault relaunches the domain instead of promoting it.
    let actions = roster.failed("b", &failure("sticky"));
    assert!(actions.contains(&relaunch("b", SERVING + 1)));
    assert!(actions.contains(&relaunch("c", SERVING + 1)));
    assert!(
        actions
            .iter()
            .all(|action| !matches!(action, Action::Promote { .. })),
        "a failed preparation leaves nothing to promote"
    );
}

#[test]
fn an_epoch_that_stops_serving_retires_its_standby_and_its_domain_goes_on_short() {
    let mut roster = roster();
    start(&mut roster);
    parked(&mut roster);
    // The epoch stops serving on `c`: the standby layer owns that child and
    // stops it, so the epoch layer never needs its generation.
    assert_eq!(roster.retired("c"), vec![terminate("c", SERVING + 1)]);
    // A fault of the retired node replaces nothing.
    assert!(roster.failed("c", &failure("sticky")).is_empty());
    // The domain's remaining member is promoted alone.
    assert_eq!(
        roster.failed("b", &failure("sticky")),
        vec![promote("b", SERVING + 1)]
    );
    assert_eq!(roster.promoted("b", SERVING + 1), vec![stop("b", SERVING)]);
    // Serving again starts no standby on the retired node.
    let actions = roster.serve(
        Status::Serving,
        options(StandbyStorage::Auto),
        &roomy(&["a", "b", "c"]),
    );
    assert_eq!(actions, vec![spawn("b", SERVING + 2, Storage::Resident)]);
}

#[test]
fn a_device_that_needs_a_reset_takes_its_standby_and_migrates_the_node() {
    let mut roster = roster();
    start(&mut roster);
    parked(&mut roster);
    // `c`'s GPU needs a reset: its standby is dead, the node cannot run
    // again, and the rest of its domain is started without it.
    let actions = roster.unusable("c");
    assert_eq!(
        actions,
        vec![
            terminate("c", SERVING + 1),
            terminate("b", SERVING + 1),
            stop("b", SERVING),
            relaunch("b", SERVING + 1),
            migrate("c"),
        ]
    );
    // The dead node's fault replaces nothing: it is already gone.
    assert!(roster.failed("c", &failure("sticky")).is_empty());
}

#[test]
fn every_member_is_asked_to_promote_before_any_answer_is_awaited() {
    let mut roster = roster();
    start(&mut roster);
    parked(&mut roster);
    // The whole round's promotions are returned in one call, so the caller
    // issues them all before it awaits any answer (a suspended world
    // restores its communication together).
    let actions = roster.failed("c", &failure("launch_failed"));
    assert_eq!(
        actions,
        vec![promote("b", SERVING + 1), promote("c", SERVING + 1)]
    );
}

#[test]
fn a_promotion_that_fails_relaunches_the_domain() {
    let mut roster = roster();
    start(&mut roster);
    parked(&mut roster);
    roster.failed("b", &failure("sticky"));
    roster.promoted("b", SERVING + 1);
    let actions = roster.promote_failed("c");
    // The node whose promotion failed has no serving child to stop; every
    // other member of the domain is stopped and started again.
    assert_eq!(
        actions,
        vec![
            terminate("b", SERVING + 1),
            stop("b", SERVING),
            relaunch("b", SERVING + 1),
            terminate("c", SERVING + 1),
            relaunch("c", SERVING + 1),
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
    let mut roster = roster();
    let mut facts = roomy(&["a", "b", "c"]);
    facts.get_mut("b").expect("b's facts").crowded = true;
    let actions = roster.serve(Status::Serving, options(StandbyStorage::Auto), &facts);
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
    let mut roster = roster();
    let mut facts = roomy(&["a", "b", "c"]);
    let c = facts.get_mut("c").expect("c's facts");
    c.crowded = true;
    c.fits_suspended = false;
    let actions = roster.serve(Status::Serving, options(StandbyStorage::Auto), &facts);
    assert_eq!(
        actions,
        vec![spawn("a", SERVING + 1, Storage::Resident)],
        "the b/c domain keeps no standby when c cannot hold one"
    );
    // Without a standby the domain is replaced by a relaunch.
    let actions = roster.failed("b", &failure("sticky"));
    assert!(actions.contains(&relaunch("b", SERVING + 1)));
    assert!(actions.contains(&relaunch("c", SERVING + 1)));
}

#[test]
fn a_standby_that_fails_to_start_is_left_out() {
    let mut roster = roster();
    start(&mut roster);
    assert!(roster.standby_lost("a").is_empty());
    assert_eq!(roster.state()[0].phase.as_str(), "absent");
    let actions = roster.failed("a", &failure("out_of_memory"));
    assert_eq!(actions, vec![relaunch("a", SERVING + 1)]);
    roster.relaunched("a", SERVING + 1);
    let actions = roster.serve(
        Status::Serving,
        options(StandbyStorage::Auto),
        &roomy(&["a", "b", "c"]),
    );
    assert_eq!(actions, vec![spawn("a", SERVING + 2, Storage::Resident)]);
}

#[test]
fn a_standby_report_of_another_generation_does_not_park_it() {
    let mut roster = roster();
    start(&mut roster);
    roster.parked("a", SERVING);
    assert_eq!(roster.state()[0].phase.as_str(), "starting");
    roster.parked("a", SERVING + 1);
    assert_eq!(roster.state()[0].phase.as_str(), "parked");
}
