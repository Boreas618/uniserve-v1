//! A group lost while a replacement starts, and what the engine does meanwhile.
//!
//! The launched path states the same events with processes: a rank dies, its
//! group's failure domain starts another launch, and the engine keeps taking
//! work for that group — which waits until the replacement's ranks install and
//! then completes on them. Here the group is the simulator's one group and both
//! events are told to the executor by the test (`SimLoss`), so what is under
//! test is the scheduler's part: admission while the group is not ready, and
//! service once the replacement stands.

use std::thread;
use std::time::{Duration, Instant};

use engine::{
    CachePolicy, CommandSender, Event, GenerationConstraint, GenerationRequest,
    ImageGenerationConfig, MultimodalInputs, SimEngine, SimExecutor,
};
use ipc::{ImageParams, RequestId, SamplingParams};

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
fn an_engine_serves_once_its_only_group_is_relaunched() {
    let mut sim = SimEngine::new();
    sim.set_text_len(16);
    let executor = SimExecutor::new(sim);
    let loss = executor.loss();
    let group = engine::ExecutorGroupId("sim".to_owned());
    let scheduler = support::scheduler(Box::new(executor), SimEngine::special_tokens(), 8)
        .expect("the scheduler is built over the simulated executor");
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

/// A group whose replacement is starting does not take the whole executor
/// with it: the deployment's other group keeps its capacity and its readiness
/// while the first is not ready, and the first takes work again once its
/// replacement stands.
///
/// The launched-path case states the same events with two launches: one rank
/// dies, its group's failure domain starts another, and the executor goes on
/// serving the other group while that launch starts.
#[test]
fn a_relaunching_group_leaves_its_executor_serving_the_others() {
    use engine::{Executor, ExecutorGroupId};

    let mut sim = SimEngine::new();
    sim.name_groups(&["model", "vision"]);
    let executor = SimExecutor::new(sim);
    let loss = executor.loss();
    let model = ExecutorGroupId("model".to_owned());
    let vision = ExecutorGroupId("vision".to_owned());

    assert!(executor.is_ready(&model) && executor.is_ready(&vision));
    assert!(executor.has_capacity(&model) && executor.has_capacity(&vision));

    // The model group's rank died and its replacement is starting.
    loss.lose_to_replace(model.clone(), "the launch was lost");
    assert!(
        !executor.is_ready(&model),
        "a group whose replacement is starting is not ready"
    );
    assert!(
        !executor.is_closed(&model),
        "a group whose replacement is starting is not closed"
    );
    assert!(
        executor.is_ready(&vision) && executor.has_capacity(&vision),
        "the other group keeps serving while the replacement starts"
    );
    assert!(
        !executor.has_capacity(&model),
        "the lost group takes no work until its replacement stands"
    );

    // The replacement stands.
    loss.restore(&model);
    assert!(
        executor.is_ready(&model) && executor.has_capacity(&model),
        "the replacement takes work"
    );
}
