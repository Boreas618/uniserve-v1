//! Worker groups as the scheduler serves them over the production submission
//! door: a deployment that splits prefill from decode, and an engine whose
//! device fails every batch.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ipc::{RequestId, SamplingParams};
use scheduler::{CachePolicy, Event, GenerationRequest, MultimodalInputs, SchedulerHandle};
use testkit::FakeBatches;

use crate::support;

/// How long a request may take to reach its terminal event.
const ANSWER: Duration = Duration::from_secs(30);

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

/// A text request of `max_und_tokens` tokens whose prompt is `prompt`.
fn greedy_request(id: u64, prompt: Vec<u32>, max_und_tokens: usize) -> GenerationRequest {
    let mut request = request(id, 1, 0);
    request.prompt_token_ids = prompt;
    request.sampling = SamplingParams {
        ignore_eos: true,
        ..Default::default()
    };
    request.max_und_tokens = max_und_tokens;
    request
}

/// Serves `requests` on `groups` and returns the tokens each generated.
fn generate(
    groups: &[(&str, scheduler::WorkerGroupRole)],
    requests: Vec<GenerationRequest>,
) -> Vec<Vec<u32>> {
    use scheduler::{SchedulerConfig, WorkerGroupId};

    let names: Vec<&str> = if groups.is_empty() {
        vec!["model"]
    } else {
        groups.iter().map(|(name, _)| *name).collect()
    };
    let pools = names
        .iter()
        .map(|name| support::fake::text_pool(name, 8))
        .collect();
    let batches = Arc::new(FakeBatches::new(pools));
    let rank = support::fake::rank_of(&batches);
    let mut args = testkit::args("groups");
    let limits = SchedulerConfig {
        max_running_requests: 4,
        ..SchedulerConfig::default()
    };
    if !groups.is_empty() {
        let worker_groups = &mut args.engine_config.parallel.worker_groups;
        let base = worker_groups.remove(0);
        *worker_groups = groups
            .iter()
            .map(|(name, role)| scheduler::WorkerGroupConfig {
                id: WorkerGroupId((*name).to_owned()),
                role: *role,
                ..base.clone()
            })
            .collect();
        args.group = names[0].to_owned();
    }
    let engine =
        SchedulerHandle::over_batches(&args, limits, batches).expect("the fake worker serves");
    support::install_policy_on(&engine, scheduler::BatchCapacity::of(&rank));

    let mut streams: Vec<_> = requests
        .into_iter()
        .map(|request| {
            engine
                .submit(request)
                .expect("the engine takes the submission")
        })
        .collect();
    let mut generated = Vec::new();
    for events in &mut streams {
        let (tokens, end) = support::wait_for_tokens(events, usize::MAX, ANSWER);
        assert!(
            matches!(end, Some(Event::Finished { .. })),
            "a request did not finish: it generated {tokens:?} and ended {end:?}"
        );
        generated.push(tokens);
    }
    engine.shutdown();
    generated
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
    )
    .expect("the fake worker serves");
    support::install_policy_on(&engine, scheduler::BatchCapacity::of(&rank));

    // The engine serves while its group is live.
    let mut first = engine
        .submit(request(1, 4, 100))
        .expect("the engine takes the submission");
    let deadline = Instant::now() + ANSWER;
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
    let deadline = Instant::now() + ANSWER;
    while batches.held() == 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    batches.fail_held("every rank of the group was lost");
    let deadline = Instant::now() + ANSWER;
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

/// A deployment that splits prefill from decode generates what one group
/// generates: the prefill group hands its requests to the decode group and the
/// tokens are the model's either way.
#[test]
#[ignore = "the in-process worker publishes no cache export (`CallOutput::cache_output`), so a \
            prefill group cannot hand a request to a decode group"]
fn a_prefill_group_hands_its_requests_to_a_decode_group() {
    use scheduler::WorkerGroupRole;

    let prompts = [
        vec![7, 8, 9],
        (0..600u32).map(|index| index % 47 + 5).collect(),
    ];
    let requests = || {
        prompts
            .iter()
            .enumerate()
            .map(|(index, prompt)| greedy_request(index as u64 + 101, prompt.clone(), 6))
            .collect::<Vec<_>>()
    };

    let alone = generate(&[], requests());
    let handed = generate(
        &[
            ("prefill", WorkerGroupRole::Prefill),
            ("decode", WorkerGroupRole::Decode),
        ],
        requests(),
    );
    assert!(
        alone.iter().all(|tokens| tokens.len() == 6),
        "one group generated {alone:?}"
    );
    assert_eq!(handed, alone, "the prefill and decode groups parted ways");
}
