//! Worker groups as the scheduler serves them over the production submission
//! door: an engine whose device fails every batch.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ipc::{RequestId, SamplingParams};
use scheduler::{CachePolicy, GenerationRequest, MultimodalInputs, SchedulerHandle};
use testkit::FakeBatches;

use crate::support;

/// How long a request may take to reach its terminal event.
const RESPONSE: Duration = Duration::from_secs(30);

/// A text request whose prompt is `length` tokens starting at `first`.
fn request(id: u64, length: u32, first: u32) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: (0..length).map(|index| first + index).collect(),
        multimodal_inputs: MultimodalInputs::default(),
        sampling: SamplingParams::default(),
        max_und_tokens: 8,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy::default(),
        include_stop_token: false,
        generated_token_ids: Vec::new(),
        adapter: 0,
    }
}

/// An engine whose device fails a batch, with no replacement coming, stops
/// admitting work.
#[test]
fn losing_every_worker_stops_request_admission() {
    use scheduler::{SchedulerConfig, SubmitError};

    let batches = Arc::new(FakeBatches::new(vec![support::fake::text_pool("model", 8)]));
    let rank = support::fake::rank_of(&batches);
    let engine = SchedulerHandle::over_batches(
        &testkit::args("sim-model"),
        SchedulerConfig {
            max_running_requests: 8,
            ..SchedulerConfig::default()
        },
        Arc::clone(&batches) as Arc<dyn ipc::execution::Batches>,
        scheduler::CommandWaker::noop(),
    )
    .expect("the fake worker serves");
    support::install_policy_on(&engine, scheduler::BatchCapacity::of(&rank));

    // The engine serves while its group is live.
    let mut first = engine
        .submit(request(1, 4, 100))
        .expect("the engine takes the submission");
    let deadline = Instant::now() + RESPONSE;
    let mut served = false;
    while Instant::now() < deadline && !served {
        match first.try_recv() {
            Ok(event) => served = event.is_terminal(),
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert!(served, "the engine serves while its group is live");

    // Every rank is gone: the device fails the batch it holds, and no
    // replacement will stand.
    batches.hold();
    let _second = engine
        .submit(request(3, 4, 300))
        .expect("the engine takes the submission");
    let deadline = Instant::now() + RESPONSE;
    while batches.held() == 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    batches.fail_held("every rank of the group was lost");
    let deadline = Instant::now() + RESPONSE;
    while !engine.is_dead() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        engine.is_dead(),
        "an engine without execution workers kept accepting work"
    );
    match engine.submit(request(2, 4, 200)) {
        Err(SubmitError::Dead) => {}
        Err(other) => panic!("a dead engine refused the request as {other:?}"),
        Ok(_) => panic!("a dead engine admitted a request"),
    }
    engine.shutdown();
}
