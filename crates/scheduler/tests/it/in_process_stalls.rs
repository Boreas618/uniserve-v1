//! A batch the device fails after accepting it becomes the engine's first
//! error, naming the batch and what the device said, and the engine serves
//! nothing further.
//!
//! A stalled forward ends this way on a launched rank: the worker's watchdog
//! fails the stalled submission with its report. The consequence for other
//! requests of a lost group is `migration.rs`'s subject and is not restated.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use scheduler::SchedulerConfig;
use testkit::FakeBatches;

use crate::support::fake;

/// How long the engine may take to report the failure.
const DEADLINE: Duration = Duration::from_secs(30);

/// A text request of eight prompt tokens that generates four.
fn request(id: u64) -> scheduler::GenerationRequest {
    scheduler::GenerationRequest {
        request_id: ipc::RequestId(id),
        prompt_token_ids: vec![7; 8],
        multimodal_inputs: Default::default(),
        sampling: Default::default(),
        max_und_tokens: 4,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        generated_token_ids: Vec::new(),
        adapter: 0,
    }
}

/// A batch failed on the device is reported as the engine's first error,
/// naming the batch and the device's report, and the engine stops: a later
/// submission is refused.
#[test]
fn a_batch_failed_on_the_device_is_the_engines_first_error() {
    let pool = Arc::new(FakeBatches::new(vec![fake::text_pool("model", 8)]));
    pool.hold();
    let scheduler = fake::scheduler_over(
        Arc::<testkit::FakeBatches>::clone(&pool),
        SchedulerConfig::default(),
        Default::default(),
    )
    .expect("the scheduler is built over the fake");
    let failure = scheduler.failure_handle();
    let (commands, scheduler_thread) = fake::spawn(scheduler);

    let _events = commands.submit(request(31)).expect("the engine takes it");
    let deadline = Instant::now() + DEADLINE;
    while pool.held() == 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(pool.held(), 1, "the request's prefill reached the device");

    // The device's watchdog fails the batch it stalled on.
    pool.fail_held("the forward stalled for 4.0 s");

    let stopped = scheduler_thread.join().expect("the scheduler thread ends");
    assert!(
        stopped,
        "the engine served on after its device failed a batch"
    );
    assert!(
        commands.submit(request(9)).is_err(),
        "a stopped engine accepted another submission"
    );

    let reported = failure.get().expect("the engine reports its first error");
    for named in ["batch 1", "the forward stalled for 4.0 s"] {
        assert!(
            reported.contains(named),
            "the failure does not name {named:?}: {reported}"
        );
    }
}
