//! A rank's stall report becomes the engine's first error, naming the forward
//! that stalled, its batch and the calls it held, over the simulated
//! executor.
//!
//! The launched case stalls a rank's builder thread under a running request
//! and asserts that the failure it surfaces names the same three things. The
//! rank is only the *cause* of the report; the report's shape and the name the
//! engine builds from it are scheduler-side, so the case asserts both here.
//!
//! The consequence for the request — a running text request ends as
//! migratable for a loss — is `migration.rs`'s subject and is not restated.

use std::thread;
use std::time::{Duration, Instant};

use engine::{SimEngine, SimExecutor, SpecialTokenIds, ExecutorGroupId};
use ipc::{CallId, ErrorCallIdentity, RequestKey, Stall};

use crate::support;

/// How long the engine may take to report the stall.
const DEADLINE: Duration = Duration::from_secs(30);

/// The group the simulated executor serves.
fn group() -> ExecutorGroupId {
    ExecutorGroupId("sim".to_owned())
}

/// The call identity a stalled job names: one request's prefill call.
fn request_key(request: u64) -> RequestKey {
    RequestKey {
        engine_id: 0,
        request_id: ipc::RequestId(request),
        request_epoch: 0,
    }
}

/// The special tokens the simulated engine states.
fn special_tokens() -> SpecialTokenIds {
    SimEngine::special_tokens()
}

/// A stall is reported as the engine's first error, naming the forward the
/// job belongs to, its batch and the call it held, and the group it served
/// fails with it.
#[test]
fn a_reported_stall_names_the_forward_and_the_requests_it_held() {
    let executor = SimExecutor::new(SimEngine::new());
    let reports = executor.loss();
    let scheduler = support::scheduler(Box::new(executor), special_tokens(), 32)
        .expect("the scheduler is built over the simulated executor");
    let failure = scheduler.failure_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let commands = engine::CommandSender::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));

    // The rank reports a job of the first batch that has not completed.
    reports.stall(
        group(),
        &Stall {
            batch_id: Some(1),
            route: Some("prefill".to_owned()),
            method: Some("decode".to_owned()),
            calls: vec![ErrorCallIdentity {
                request_key: request_key(31),
                call_id: CallId::new(1, 0),
            }],
            running_ms: 4_000,
        },
    );

    // The group fails with the report, so nothing serves on.
    let stopped = scheduler_thread.join().expect("the scheduler thread ends");
    assert!(
        stopped,
        "the engine served on after a stall failed its group"
    );
    assert!(
        commands
            .submit(engine::GenerationRequest {
                request_id: ipc::RequestId(9),
                prompt_token_ids: vec![7; 8],
                negative_prompt_token_ids: Vec::new(),
                multimodal_inputs: Default::default(),
                constraint: engine::GenerationConstraint::UndOnly,
                sampling: Default::default(),
                image: Default::default(),
                max_images: 1,
                max_und_tokens: 4,
                include_stop_token: false,
                stop_strings: Vec::new(),
                stop_token_ids: Vec::new(),
                priority: 0,
                cache: Default::default(),
                image_generation: Default::default(),
                readout: Vec::new(),
                generated_token_ids: Vec::new(),
                adapter: 0,
                generated_images: Vec::new(),
                trajectory: Default::default(),
            })
            .is_err(),
        "a stopped engine accepted another submission"
    );

    // The engine's first error is the stall, and it names what stalled.
    let deadline = Instant::now() + DEADLINE;
    let mut reported = failure.get();
    while reported.is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
        reported = failure.get();
    }
    let reported = reported.expect("the engine reports its first error");
    for named in ["the builder thread stalled", "batch 1 (prefill", "31/"] {
        assert!(
            reported.contains(named),
            "the failure does not name {named:?}: {reported}"
        );
    }
}
