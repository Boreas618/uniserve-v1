//! The in-process path serving a request, without a device.
//!
//! `in_process.rs` states what `SchedulerHandle::over_batches` refuses; this
//! states what it does. The pool here is `executor::batch::Batches` and answers
//! every batch it is handed with the mechanism's own values — one
//! `CallOutput::completed` per call it was given, and the batch's timing — so
//! the whole path runs: the policy admits, the mechanism assembles columns and
//! a shape, the arm states them to the executor and files the report it gets
//! back, and the request emits a token.
//!
//! What the stub does not stand in for: the device. A real executor captures
//! graphs, stages columns into its arena and derives the token from the model;
//! this one is handed the mechanism's batch and answers it. Everything between
//! those two points is the code under test.

use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::{
    BatchCapacity, EngineConfig, Event, GenerationConstraint, GenerationRequest, SchedulerHandle,
    Stage,
};
use executor::batch::{Batch, Batches, Inventory, Placement};
use ipc::{ComponentConfig, ImageParams, SamplingParams, MainProcessEndpoint, MainProcessInfo};

use crate::support::SAMPLED;

/// How long a command answer may take.
///
/// Generous because the first one is the whole boot of a policy: the precopy
/// compiles the `fcfs` module inside a debug build, on a machine that runs this
/// suite's tests in parallel. The bound is the harness's, not the mechanism's;
/// a mechanism that stopped answering would still fail here, only later.
const ANSWER: Duration = Duration::from_secs(60);

/// How long a request may take to emit its first token.
const FIRST_TOKEN: Duration = Duration::from_secs(60);

/// Blocks until a command answers, and answers its result.
///
/// `track_caller` so a timeout names the command that was awaited rather than
/// the helper: every wait here shares one bound, and a location inside the
/// helper cannot tell one unanswered command from another.
#[track_caller]
fn answer<T>(receiver: crossbeam_channel::Receiver<Result<T, String>>) -> Result<T, String> {
    match receiver.recv_timeout(ANSWER) {
        Ok(result) => result,
        Err(error) => panic!("the scheduler did not answer: {error}"),
    }
}

/// A greedy text request of `prompt` tokens that generates at most `max_tokens`.
fn text_request(id: u64, prompt: usize, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: ipc::RequestId(id),
        prompt_token_ids: vec![7; prompt],
        multimodal_inputs: Default::default(),
        negative_prompt_token_ids: Vec::new(),
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: max_tokens,
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
    }
}

/// The `fcfs` program, instantiated for the capacity this pool was built for:
/// a policy declares its own limits and the mechanism refuses a declaration
/// above the ceilings the ranks were started for.
fn policy_program(pool: &crate::support::Serving) -> engine::PolicyProgram {
    // A pool's capacity is the rank's own statement: what the launch built it
    // for, which the policy's declaration must not exceed.
    let rank = pool
        .inventory()
        .first()
        .and_then(|pool| pool.ranks.first())
        .expect("the pool states its ranks");
    let ceilings = engine::ceilings_of(BatchCapacity::of(rank));
    engine::PolicyProgram {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: engine::Parameters {
            chunked_prefill_size: Some(ceilings.chunked_prefill_size),
            max_queued_requests: ceilings.max_queued_requests,
            max_batch: Some(ceilings.max_batch),
            max_num_batched_tokens: Some(ceilings.max_num_batched_tokens),
            prefetch_min_tokens: Some(ceilings.prefetch_min_tokens),
            timeout: Some(ceilings.timeout),
            ..Default::default()
        },
        stated: Default::default(),
    }
}

/// The epoch's first policy: prepared, then switched in on the handle's own
/// command surface, which is the launched path's.
fn install(handle: &SchedulerHandle, pool: &crate::support::Serving) {
    let source: Arc<dyn engine::PolicySource + Send + Sync> =
        Arc::new(engine::PolicyBytes::new(crate::support::policy_module()));
    answer(handle.precopy_policy(policy_program(pool), source)).expect("the policy is precopied");
    answer(handle.stage(Stage::Switchover)).expect("the policy serves");
}

#[test]
fn a_request_is_served_through_the_executors_own_batch_path() {
    let serving = Arc::new(crate::support::Serving::new(vec![crate::support::pool(
        "model",
    )]));
    let batches: Arc<dyn Batches> = serving.clone();
    let handle = SchedulerHandle::over_batches(
        crate::support::serving::config(&serving, "in-process-completion"),
        batches,
    )
    .expect("the pool serves");
    install(&handle, &serving);
    // The kernels this epoch serves on, as the control thread's switchover arm
    // publishes it. The handle's stage command is not the arm that numbers a
    // generation — that is the process command's — so the test states the
    // number through the one cell both threads share, and asserts that what the
    // arm stamped is what the scheduler read once for the step.
    const GENERATION: u64 = 7;
    handle
        .generation()
        .store(GENERATION, std::sync::atomic::Ordering::Relaxed);

    let mut events = handle
        .submit(text_request(1, 8, 4))
        .expect("the request is submitted");

    // The request emits a token: the mechanism admitted it, assembled a batch,
    // the arm stated it to the executor, and the report came back through the
    // completion path to the request.
    let deadline = Instant::now() + FIRST_TOKEN;
    let mut first_token = None;
    while first_token.is_none() && Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::TextToken { id, .. }) => first_token = Some(id),
            Ok(Event::TextTokens { ids }) => first_token = ids.first().copied(),
            Ok(_) => {}
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    let first = first_token.expect("the request emitted a token");
    assert_eq!(
        first, SAMPLED,
        "the token is the one this pool sampled, not an invented one"
    );

    // The mechanism asked for a prefill and then a decode: one batch per phase
    // per step, each stated with the shape its columns are served from. The
    // two batches come from two steps, so the second is waited for rather than
    // read off a snapshot: the first token is the prefill's, and reading the
    // pool the instant it arrives would measure the race, not the schedule.
    let phase = |batches: &[Batch], name: &str| {
        batches
            .iter()
            .any(|batch| batch.shape.variant.name().starts_with(name))
    };
    let deadline = Instant::now() + FIRST_TOKEN;
    let batches = loop {
        let batches = serving.submitted();
        if (phase(&batches, "prefill") && phase(&batches, "decode")) || Instant::now() >= deadline {
            break batches;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    assert!(
        phase(&batches, "prefill") && phase(&batches, "decode"),
        "the pool was asked for both phases: {} batches",
        batches.len()
    );
    // Every batch names the generation the step read, so the kernels it ran on
    // are stated rather than assumed, and one step never mixes two worlds.
    assert!(
        batches.iter().all(|batch| batch.generation == GENERATION),
        "every batch of a step carries the generation that step read: {:?}",
        batches
            .iter()
            .map(|batch| batch.generation)
            .collect::<Vec<_>>()
    );

    handle.shutdown();
}

/// The step's free slots are a hard limit the mechanism keeps by stopping: a
/// request that does not fit one waits, and is admitted when a slot frees.
///
/// §4.4's dispatch row puts the error on *work* put into the step, and gives
/// `admit` the other rule in the same row: the mechanism admits in the plan's
/// order, and the first request it cannot place, and those after it, stay
/// queued. `fcfs` names every queued request, so a refusal here would stop every
/// world that queues more than it can run at once; a stop serves them in turn.
#[test]
fn a_request_that_does_not_fit_a_slot_waits_for_one() {
    let serving = Arc::new(crate::support::Serving::new(vec![crate::support::pool(
        "model",
    )]));
    let batches: Arc<dyn Batches> = serving.clone();
    let mut config = crate::support::serving::config(&serving, "in-process-completion");
    config.max_running_requests = 1;
    let handle = SchedulerHandle::over_batches(config, batches).expect("the pool serves");
    install(&handle, &serving);

    let mut first = handle
        .submit(text_request(1, 8, 4))
        .expect("the first request is submitted");
    let mut second = handle
        .submit(text_request(2, 8, 4))
        .expect("the second request is submitted");

    let mut served = [false, false];
    let deadline = Instant::now() + FIRST_TOKEN;
    while !served.iter().all(|done| *done) && Instant::now() < deadline {
        for (index, events) in [&mut first, &mut second].into_iter().enumerate() {
            while let Ok(event) = events.try_recv() {
                match event {
                    Event::TextToken { .. } | Event::TextTokens { .. } => served[index] = true,
                    refused @ (Event::Rejected { .. } | Event::Error { .. }) => {
                        panic!("request {} was refused, not queued: {refused:?}", index + 1)
                    }
                    _ => {}
                }
            }
        }
        if !served.iter().all(|done| *done) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    assert!(
        served.iter().all(|done| *done),
        "both requests were served, one after the other: {served:?}"
    );
    handle.shutdown();
}

/// An admitted request is worked on, in the phase the mechanism tells the
/// policy it serves next.
///
/// A policy's state is append-only: it learns a request from its arrival and
/// nothing else about it unless the mechanism states it. A request that is
/// admitted and never restated is one whose phase the policy never learns, so
/// no policy — `fcfs` included — can plan the work that would prefilled it, and
/// the epoch dispatches nothing while requests queue behind it. This case
/// states the mechanism's half: one text request, the real `fcfs` program, and
/// a batch handed to the executor with no other help.
#[test]
fn an_admitted_request_is_prefilled_on_the_next_step() {
    let serving = Arc::new(crate::support::Serving::new(vec![crate::support::pool(
        "model",
    )]));
    let batches: Arc<dyn Batches> = serving.clone();
    let handle = SchedulerHandle::over_batches(
        crate::support::serving::config(&serving, "in-process-completion"),
        batches,
    )
    .expect("the pool serves");
    install(&handle, &serving);
    // The consumer is kept open for the case's length: a world whose client
    // went away ends its requests, which is not what is under test.
    let _events = handle
        .submit(text_request(1, 8, 4))
        .expect("the request is submitted");

    let deadline = Instant::now() + FIRST_TOKEN;
    while serving.submitted().is_empty() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let handed = serving.submitted();
    let first = handed.first().expect("the pool was handed a batch");
    assert_eq!(
        first.shape.rows, 1,
        "the batch is the one request's, not a full bucket's"
    );
    assert!(
        first.shape.variant.name().starts_with("prefill"),
        "the request's first call is its prefill, not a later phase: {}",
        first.shape.variant.name()
    );
    // The prefill comes first in submission order: the request's prompt is
    // written before any token of it is decoded.
    let decoded_first = handed
        .iter()
        .position(|batch| batch.shape.variant.name().starts_with("decode"));
    assert!(
        decoded_first.is_none_or(|index| index > 0),
        "a decode was handed before the prefill: {:?}",
        handed
            .iter()
            .map(|batch| batch.shape.variant.name())
            .collect::<Vec<_>>()
    );
    handle.shutdown();
}

/// The reported distribution is the model's, whatever nucleus the request
/// samples under: a request cut at half the mass reports the same five
/// candidates with the same log probabilities as one sampling the whole
/// distribution.
///
/// The two runs share the request identity and the prompt, so the simulated
/// head scores them alike; only the nucleus differs.
#[test]
fn logprobs_report_the_model_distribution_under_a_cutoff() {
    use engine::{EngineConfig, Executor, SchedulerHandle, SimEngine, SimExecutor};
    use std::thread;

    let candidates = |top_p: f32| -> Vec<ipc::TokenLogprob> {
        let mut sim = SimEngine::new();
        sim.set_text_len(8);
        let executor = SimExecutor::new(sim);
        let rank = executor.info().runtime_info().expect("the sim states its rank");
        let engine = SchedulerHandle::with_executor(EngineConfig::sim("logprobs"), Box::new(executor))
            .expect("the simulated executor serves");
        crate::support::install_policy_on(&engine, engine::BatchCapacity::of(&rank));

        let mut request = text_request(1, 2, 7);
        request.sampling = SamplingParams {
            temperature: 0.0,
            top_p,
            return_logprobs: true,
            n_logprobs: 5,
            ..Default::default()
        };
        let mut events = engine.submit(request).expect("the engine takes the submission");
        let deadline = Instant::now() + ANSWER;
        let mut reported = None;
        while Instant::now() < deadline && reported.is_none() {
            match events.try_recv() {
                Ok(Event::TokenLogprobs { candidates, .. }) if reported.is_none() => {
                    reported = Some(candidates);
                }
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        engine.shutdown();
        reported.expect("the request reports the distribution of its first token")
    };

    let cut = candidates(0.5);
    let whole = candidates(1.0);
    assert_eq!(cut.len(), 5, "five candidates were asked for: {cut:?}");
    assert!(
        cut.iter().all(|entry| entry.logprob.is_finite()),
        "a reported log probability is not finite: {cut:?}"
    );
    assert_eq!(cut, whole, "the nucleus changed the reported distribution");
}
