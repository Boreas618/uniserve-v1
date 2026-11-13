//! The generation update's jobs over the parts of a serving engine on the meta device
//! (`operations::tests::Node`), whose rank serves generation 1 of no kernel.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use ops::Generation;
use scheduling_guest::limits::Parameters;
use worker::completion::Outcome;

use super::{commit, named, rollback, switch};
use crate::process::operations::ask;
use crate::process::operations::tests::{LONG, Node};
use crate::scheduler::Message;
use crate::scheduler::tests::toy;

/// The generation every rank's graphs serve, in rank order.
fn serving(node: &Node) -> Vec<u64> {
    node.executors
        .iter()
        .map(|executor| {
            executor
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
    node.executors.iter().all(|executor| {
        let submission = executor
            .prepare(Arc::new(Generation::new(generation, [])))
            .unwrap();
        matches!(submission.wait_timeout(LONG), Some(Outcome::Success(_)))
    })
}

/// Whether every rank holds one generation: it prepares another beside it, which is then given
/// back.
fn holds_one(node: &Node) -> bool {
    let steady = prepares(node, 99);
    for executor in &node.executors {
        let rolled = executor.rollback(99).unwrap();
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
    (toy("(i64.const 0)"), Parameters::default())
}

/// The node with generation 2 switched to: on trial, generation 1 kept.
fn on_trial() -> Node {
    let node = Node::start();
    assert!(prepares(&node, 2));
    switch(
        &node.schedulers,
        &node.executors,
        &node.data,
        &node.served,
        2,
        &policy(),
    )
    .unwrap();
    node
}

#[test]
fn a_switch_moves_the_schedulers_and_every_executor_to_the_generation() {
    let node = on_trial();

    assert_eq!(node.served.load(Ordering::Acquire), 2);
    assert_eq!(serving(&node), [2]);
    // Generation 1 is kept for the trial: no other update is prepared beside it.
    assert!(!holds_one(&node));
}

#[test]
fn a_commit_leaves_the_generation_on_trial_alone() {
    let node = on_trial();

    commit(&node.schedulers, &node.executors, 2).unwrap();
    assert_eq!(node.served.load(Ordering::Acquire), 2);
    assert_eq!(serving(&node), [2]);
    assert!(holds_one(&node));
    assert!(!keeps_a_policy(&node));
}

#[test]
fn a_rollback_leaves_the_kept_generation_alone() {
    let node = on_trial();

    rollback(
        &node.schedulers,
        &node.executors,
        &node.data,
        &node.served,
        2,
        1,
    )
    .unwrap();
    assert_eq!(node.served.load(Ordering::Acquire), 1);
    assert_eq!(serving(&node), [1]);
    assert!(holds_one(&node));
    assert!(!keeps_a_policy(&node));
}

#[test]
fn a_policy_a_scheduler_refuses_rolls_the_switch_back() {
    let node = Node::start();
    assert!(prepares(&node, 2));

    let refused = (toy("(unreachable)"), Parameters::default());
    let failed = switch(
        &node.schedulers,
        &node.executors,
        &node.data,
        &node.served,
        2,
        &refused,
    );
    assert!(failed.is_err());

    // Generation 1 serves alone, under the policy it had, and the batches carry it.
    assert_eq!(node.served.load(Ordering::Acquire), 1);
    assert_eq!(serving(&node), [1]);
    assert!(holds_one(&node));
    assert!(!keeps_a_policy(&node));
}

/// A preparation's packages read back as the daemon named them: each kernel under its operator
/// and phase (none for every phase), its arguments as the TOML values whose text the Protobuf
/// carries, and the policy beside them; a kernel without a package is refused by its operator.
#[test]
fn a_preparation_names_its_packages_with_their_arguments() {
    use proto::module::{ModuleSlot, PhaseKind};
    use proto::process::{KernelPackage, PackageRevision, Prepare};

    let package = |backend: &str, arguments: &[(&str, &str)]| PackageRevision {
        backend: backend.to_owned(),
        revision: 2,
        arguments: arguments
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect(),
    };
    let kernel = |op: &str, phase: Option<proto::process::PhaseKind>, package| KernelPackage {
        op: op.to_owned(),
        phase: phase.map(i32::from),
        package,
        part: None,
    };
    let prepare = Prepare {
        kernels: vec![
            kernel(
                "decode_attention",
                None,
                Some(package("trtllm", &[("block", "16")])),
            ),
            kernel(
                "rms_norm",
                Some(proto::process::PhaseKind::Prefill),
                Some(package("triton", &[])),
            ),
        ],
        policy: Some(package("fcfs", &[("chunked_prefill_size", "4096")])),
    };

    let (packages, policy) = named(prepare.clone()).unwrap();
    let every_phase = &packages[&ModuleSlot::all("decode_attention")];
    assert_eq!(
        (every_phase.backend.as_str(), every_phase.revision),
        ("trtllm", 2)
    );
    assert_eq!(every_phase.arguments["block"], toml::Value::Integer(16));
    assert_eq!(
        packages[&ModuleSlot::phase("rms_norm", PhaseKind::Prefill)].backend,
        "triton"
    );
    assert_eq!(policy.backend, "fcfs");
    assert_eq!(
        policy.arguments["chunked_prefill_size"],
        toml::Value::Integer(4096)
    );

    let mut unnamed = prepare;
    unnamed.kernels[1].package = None;
    let refused = named(unnamed).unwrap_err().to_string();
    assert!(refused.contains("rms_norm"), "{refused}");
}
