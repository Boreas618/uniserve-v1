//! A group lost while a replacement starts, and what the engine does meanwhile.
//!
//! The launched path states the same events with processes: a rank dies, its
//! group's failure domain starts another launch, and the engine keeps taking
//! work for that group — which waits until the replacement's ranks install and
//! then completes on them. Here the group is the simulator's one group and both
//! events are told to the worker by the test (`SimLoss`), so what is under
//! test is the scheduler's part: admission while the group is not ready, and
//! service once the replacement stands.

use std::thread;
use std::time::{Duration, Instant};

use ipc::{ImageParams, RequestId, SamplingParams};
use scheduler::{
    CachePolicy, CommandSender, Event, GenerationConstraint, GenerationRequest,
    ImageGenerationConfig, MultimodalInputs, SimEngine, SimWorker,
};

use crate::support;

/// How long a request may take to reach its terminal event.
const ANSWER: Duration = Duration::from_secs(30);

/// How long a request that must not be served is watched for.
const QUIET: Duration = Duration::from_millis(300);

/// A text request whose prompt is `length` tokens starting at `first`.
fn request(id: u64, length: u32, first: u32) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: (0..length).map(|index| first + index).collect(),
        negative_prompt_token_ids: Vec::new(),
        multimodal_inputs: MultimodalInputs::default(),
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: 8,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy::default(),
        include_stop_token: false,
        image_generation: ImageGenerationConfig::default(),
        readout: Vec::new(),
        generated_token_ids: Vec::new(),
        adapter: 0,
        generated_images: Vec::new(),
        trajectory: Default::default(),
    }
}

/// An engine whose only group is lost while a replacement starts keeps
/// serving: a request submitted meanwhile waits rather than failing, and it
/// completes on the replacement.
#[test]
fn an_engine_serves_once_its_only_group_is_replaced() {
    let mut sim = SimEngine::new();
    sim.set_text_len(16);
    let worker = SimWorker::new(sim);
    let loss = worker.loss();
    let group = scheduler::WorkerGroupId("sim".to_owned());
    let scheduler = support::scheduler(Box::new(worker), SimEngine::special_tokens(), 8)
        .expect("the scheduler is built over the simulated worker");
    let (tx, rx) = crossbeam_channel::unbounded();
    let commands = CommandSender::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));

    // The group serves its first request.
    let mut first = commands
        .submit(request(1, 4, 100))
        .expect("the engine takes the submission");
    let deadline = Instant::now() + ANSWER;
    let mut served = false;
    while Instant::now() < deadline && !served {
        match first.try_recv() {
            Ok(event) => served = matches!(event, Event::Finished { .. }),
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert!(served, "the engine serves while its group is live");

    // Its only group is lost, and a replacement is starting.
    loss.lose_to_replace(group.clone(), "the launch was lost");
    let mut waiting = commands
        .submit(request(2, 4, 200))
        .expect("a request submitted meanwhile is admitted");
    thread::sleep(QUIET);
    assert!(
        waiting.try_recv().is_err(),
        "a request whose group is not ready waits rather than failing"
    );

    // The replacement stands, and the waiting request completes on it.
    loss.restore(&group);
    let deadline = Instant::now() + ANSWER;
    let mut finished = false;
    while Instant::now() < deadline && !finished {
        match waiting.try_recv() {
            Ok(event) => finished = matches!(event, Event::Finished { .. }),
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert!(finished, "the replacement serves the waiting request");

    commands.shutdown();
    let _ = scheduler_thread.join();
}

/// A group whose replacement is starting does not take the whole worker
/// with it: the deployment's other group keeps its capacity and its readiness
/// while the first is not ready, and the first takes work again once its
/// replacement stands.
#[test]
fn a_group_being_replaced_leaves_its_worker_serving_the_others() {
    use scheduler::{Worker, WorkerGroupId};

    let mut sim = SimEngine::new();
    sim.name_groups(&["model", "vision"]);
    let worker = SimWorker::new(sim);
    let loss = worker.loss();
    let model = WorkerGroupId("model".to_owned());
    let vision = WorkerGroupId("vision".to_owned());

    assert!(worker.is_ready(&model) && worker.is_ready(&vision));
    assert!(worker.has_capacity(&model) && worker.has_capacity(&vision));

    // The model group's rank died and its replacement is starting.
    loss.lose_to_replace(model.clone(), "the launch was lost");
    assert!(
        !worker.is_ready(&model),
        "a group whose replacement is starting is not ready"
    );
    assert!(
        !worker.is_closed(&model),
        "a group whose replacement is starting is not closed"
    );
    assert!(
        worker.is_ready(&vision) && worker.has_capacity(&vision),
        "the other group keeps serving while the replacement starts"
    );
    assert!(
        !worker.has_capacity(&model),
        "the lost group takes no work until its replacement stands"
    );

    // The replacement stands.
    loss.restore(&model);
    assert!(
        worker.is_ready(&model) && worker.has_capacity(&model),
        "the replacement takes work"
    );
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
    use scheduler::{SchedulerConfig, SchedulerHandle, Worker, WorkerGroupId};

    let mut sim = SimEngine::new();
    sim.set_text_len(16);
    if !groups.is_empty() {
        let names: Vec<&str> = groups.iter().map(|(name, _)| *name).collect();
        sim.name_groups(&names);
    }
    let worker = SimWorker::new(sim);
    let rank = worker
        .info()
        .runtime_info()
        .expect("the sim states its rank");
    let mut args = scheduler::sim_args("groups");
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
    }
    let engine = SchedulerHandle::with_worker(&args, limits, Box::new(worker))
        .expect("the simulated worker serves");
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
        let deadline = Instant::now() + ANSWER;
        let mut tokens = Vec::new();
        let mut finished = false;
        while Instant::now() < deadline && !finished {
            match events.try_recv() {
                Ok(Event::TextToken { id, .. }) => tokens.push(id),
                Ok(event) if event.is_terminal() => finished = true,
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        assert!(
            finished,
            "a request did not finish: it generated {tokens:?}"
        );
        generated.push(tokens);
    }
    engine.shutdown();
    generated
}

/// An engine whose every group is lost, with no replacement coming, stops
/// admitting work.
#[test]
fn losing_every_worker_stops_request_admission() {
    use scheduler::{SchedulerConfig, SchedulerHandle, SubmitError, Worker, WorkerGroupId};

    let mut sim = SimEngine::new();
    sim.set_text_len(4);
    let worker = SimWorker::new(sim);
    let loss = worker.loss();
    let rank = worker
        .info()
        .runtime_info()
        .expect("the sim states its rank");
    let engine = SchedulerHandle::with_worker(
        &scheduler::sim_args("sim-model"),
        SchedulerConfig::default(),
        Box::new(worker),
    )
    .expect("the simulated worker serves");
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

    // Every rank is gone, and no replacement will stand.
    loss.lose(
        WorkerGroupId("sim".to_owned()),
        "every rank of the group was lost",
    );
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
fn a_prefill_group_hands_its_requests_to_a_decode_group() {
    use scheduler::WorkerGroupRole;

    let prompts = vec![
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
