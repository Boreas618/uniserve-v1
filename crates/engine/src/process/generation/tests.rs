//! The generation update's jobs over the parts of a serving engine on the meta device
//! (`operations::tests::Node`), whose rank serves generation 1 of no kernel.

use std::sync::Arc;

use crate::worker::completion::Outcome;
use ops::Generation;
use scheduling::limits::Parameters;

use super::{commit, named, rollback, switch};
use crate::process::operations::ask;
use crate::process::operations::tests::Node;
use crate::scheduler::Message;
use crate::scheduler::policy::tests::{Fixture, fixture};
use testkit::LONG;

/// The generation every rank's graphs serve, in rank order.
fn serving(node: &Node) -> Vec<u64> {
    node.workers
        .iter()
        .map(|worker| {
            worker
                .call("reading the generation the rank serves", |runners| {
                    Ok(runners[0].graphs().generation())
                })
                .unwrap()
        })
        .collect()
}

/// Whether every rank prepares `generation` beside the one it serves, which it refuses while
/// another update is prepared or on trial. A rank that prepared it is left preparing it.
fn prepares(node: &Node, generation: u64) -> bool {
    node.workers.iter().all(|worker| {
        let submission = worker
            .prepare(Arc::new(Generation::new(generation, [])), None)
            .unwrap();
        matches!(submission.wait_timeout(LONG), Some(Outcome::Success(_)))
    })
}

/// Whether every rank holds one generation: it prepares another beside it, which is then given
/// back.
fn holds_one(node: &Node) -> bool {
    let steady = prepares(node, 99);
    for worker in &node.workers {
        let rolled = worker.rollback(99).unwrap();
        assert!(matches!(
            rolled.wait_timeout(LONG),
            Some(Outcome::Success(_))
        ));
    }
    steady
}

/// Whether some scheduler keeps a policy to roll back to. Asked of a scheduler that keeps none,
/// the rollback is refused and changes nothing.
fn keeps_a_policy(node: &Node) -> bool {
    let replies = ask(&node.schedulers, Message::RollbackPolicy).unwrap();
    replies.iter().any(Result::is_ok)
}

/// A policy module the schedulers take: it admits nothing.
fn policy() -> (Arc<wasmtime::Module>, Parameters) {
    (fixture(Fixture::Idle), Parameters::default())
}

/// The node with generation 2 switched to: on trial, generation 1 kept.
fn on_trial() -> Node {
    let node = Node::start();
    assert!(prepares(&node, 2));
    switch(
        &node.schedulers,
        Some(&node.route),
        &node.workers,
        &node.binding,
        &node.served,
        2,
        &policy(),
        None,
    )
    .unwrap();
    node
}

#[test]
fn a_switch_moves_the_schedulers_and_every_worker_to_the_generation() {
    let node = on_trial();

    assert_eq!(node.served.kernels(), 2);
    assert_eq!(serving(&node), [2]);
    // Generation 1 is kept for the trial: no other update is prepared beside it.
    assert!(!holds_one(&node));
}

#[test]
fn a_commit_leaves_the_generation_on_trial_alone() {
    let node = on_trial();

    commit(
        &node.schedulers,
        Some(&node.route),
        (&node.workers, &node.binding),
        2,
    )
    .unwrap();
    assert_eq!(node.served.kernels(), 2);
    assert_eq!(serving(&node), [2]);
    assert!(holds_one(&node));
    assert!(!keeps_a_policy(&node));
}

#[test]
fn a_rollback_leaves_the_kept_generation_alone() {
    let node = on_trial();

    rollback(
        &node.schedulers,
        Some(&node.route),
        &node.binding,
        &node.served,
        (2, 1),
        None,
    )
    .unwrap();
    assert_eq!(node.served.kernels(), 1);
    assert_eq!(serving(&node), [1]);
    assert!(holds_one(&node));
    assert!(!keeps_a_policy(&node));
}

#[test]
fn a_policy_a_scheduler_refuses_rolls_the_switch_back() {
    let node = Node::start();
    assert!(prepares(&node, 2));

    let refused = (fixture(Fixture::Trap), Parameters::default());
    let failed = switch(
        &node.schedulers,
        Some(&node.route),
        &node.workers,
        &node.binding,
        &node.served,
        2,
        &refused,
        None,
    );
    assert!(failed.is_err());

    // Generation 1 serves alone, under the policy it had, and the batches carry it.
    assert_eq!(node.served.kernels(), 1);
    assert_eq!(serving(&node), [1]);
    assert!(holds_one(&node));
    assert!(!keeps_a_policy(&node));
}

/// A preparation's packages read back as the daemon named them: each kernel under its operator
/// and pinned key (none for every key), its arguments as the TOML values whose text the Protobuf
/// carries, and the policy beside them; a kernel without a package is refused by its operator.
#[test]
fn a_preparation_names_its_packages_with_their_arguments() {
    use proto::module::{ModuleSlot, PackageRevision, kernel_to_wire};
    use proto::process::Prepare;

    let package = |backend: &str, arguments: &str| PackageRevision {
        backend: backend.to_owned(),
        revision: 2,
        arguments: toml::from_str(arguments).unwrap(),
    };
    let every_key = ModuleSlot::all("decode_attention");
    let pinned = ModuleSlot::key("rms_norm", "dtype_bf16_width_4096");
    let attention = package("trtllm", "block = 16");
    let norm = package("triton", "");
    let fcfs = package("fcfs", "chunked_prefill_size = 4096");
    let prepare = Prepare {
        kernel_instance: None,
        fence_modules: None,
        graphs: None,
        kernels: vec![
            kernel_to_wire(&every_key, &attention),
            kernel_to_wire(&pinned, &norm),
        ],
        policy: Some((&fcfs).into()),
    };

    let (packages, policy) = named(prepare.clone()).unwrap();
    assert_eq!(packages[&every_key], attention);
    assert_eq!(packages[&pinned], norm);
    assert_eq!(policy, fcfs);

    let mut unnamed = prepare;
    unnamed.kernels[1].package = None;
    let refused = format!("{:#}", named(unnamed).unwrap_err());
    assert!(refused.contains("rms_norm"), "{refused}");
}
