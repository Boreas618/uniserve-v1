//! Media a deployment cannot serve, over the in-process pool.
//!
//! The launched case starts a text deployment and submits video requests it
//! never declared support for. Each is rejected through its own event stream,
//! and the engine goes on serving the text it does support. Here the deployment
//! is a handle over `support::Serving` whose inventory states prefill and
//! decode only, so the refusal is the mechanism's own and no process exists to
//! fail first.

use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::{
    BatchCapacity, EngineConfig, Event, GenerationConstraint, GenerationRequest, SchedulerHandle,
    Stage,
};
use executor::batch::Batches;
use ipc::{DiffusionSamplingParams, ImageParams, RequestId, SamplingParams};
use proto::request::DiffusionRequest;

use crate::support::{self, Serving};

/// How long a command answer may take.
const ANSWER: Duration = Duration::from_secs(60);

/// Instantiates the epoch's first policy for `pool` and switches it in.
fn install(handle: &SchedulerHandle, pool: &Serving) {
    let rank = pool
        .inventory()
        .first()
        .and_then(|inventory| inventory.ranks.first())
        .expect("the pool states its ranks");
    let ceilings = engine::ceilings_of(BatchCapacity::of(rank));
    let program = engine::PolicyProgram {
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
    };
    let source: Arc<dyn engine::PolicySource + Send + Sync> =
        Arc::new(engine::PolicyBytes::new(support::policy_module()));
    let installed = handle
        .precopy_policy(program, source)
        .recv_timeout(ANSWER)
        .expect("the scheduler answers a precopy");
    installed.expect("the policy is precopied");
    let switched = handle
        .stage(Stage::Switchover)
        .recv_timeout(ANSWER)
        .expect("the scheduler answers a switchover");
    switched.expect("the policy serves");
    // The kernels this epoch serves on. The switchover command publishes the
    // program but does not number a generation — that is the process command's
    // arm, which a case does not drive — so the number is stated here through
    // the one cell both threads share, as the green text case does.
    handle
        .generation()
        .store(GENERATION, std::sync::atomic::Ordering::Relaxed);
}

/// The generation this deployment's steps are served under.
const GENERATION: u64 = 7;

/// A text request over `length` prompt tokens.
fn text_request(id: u64, length: u32) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: vec![7; length as usize],
        negative_prompt_token_ids: Vec::new(),
        multimodal_inputs: Default::default(),
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: 4,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        include_stop_token: false,
        image_generation: Default::default(),
        readout: Vec::new(),
        generated_token_ids: Vec::new(),
        adapter: 0,
        generated_images: Vec::new(),
        trajectory: Default::default(),
    }
}

/// A video request the text deployment never declared support for.
fn video_request(id: u64) -> DiffusionRequest {
    DiffusionRequest {
        request_id: RequestId(id),
        task: "t2va".to_owned(),
        prompt_token_ids: vec![1; 4],
        conditioning: models::video::Conditioning {
            task: models::video::VideoTask::T2va,
            text_tags: vec![1; 4],
            conditions: Vec::new(),
        }
        .encode()
        .expect("the conditioning encodes"),
        conditions: Vec::new(),
        priority: 0,
        sampling: DiffusionSamplingParams {
            num_frames: 22,
            video_units: 3,
            seed: 0,
            width: 1344,
            height: 768,
        },
        trajectory: Default::default(),
        encoder_inputs: None,
    }
}

/// Waits for the next event of `events`, failing by name after the bound.
fn next(events: &mut engine::EventRx, what: &str) -> Event {
    let deadline = Instant::now() + ANSWER;
    while Instant::now() < deadline {
        if let Ok(event) = events.try_recv() {
            return event;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("{what} did not answer within the bound");
}

/// Requests the deployment cannot serve are rejected on their own streams, and
/// the engine goes on serving the requests it can: rejecting media is not a
/// reason to stop a text deployment.
#[test]
fn unsupported_media_is_rejected_without_stopping_the_engine() {
    let pool = Arc::new(Serving::new(vec![support::pool("model")]));
    // The configuration the green text case uses: a batch bound is not part
    // of what this case asserts, and a construct that differs from the one
    // known to serve text would make a refusal unattributable.
    let config = support::config(&pool, "in-process-media");
    let engine = SchedulerHandle::over_batches(config, pool.clone() as Arc<dyn Batches>)
        .expect("the text deployment serves");
    install(&engine, &pool);

    // Text is served **before** any rejection, so the rejection cannot be what
    // leaves the deployment able to serve: the state this case ends in is not
    // a state it was already in.
    let mut before = engine
        .submit(text_request(0, 8))
        .expect("the first text request is accepted");
    let (served_before, ended_before) = support::wait_for_tokens(&mut before, 1, ANSWER);
    assert!(
        !served_before.is_empty(),
        "the deployment serves text before any rejection: no token arrived, and the request \
         ended as {ended_before:?}"
    );

    let mut rejections = Vec::new();
    for index in 0..2 {
        let mut events = engine
            .submit(proto::request::Request::Diffusion(video_request(10_000 + index)))
            .expect("the submission is accepted");
        let event = next(&mut events, "an unsupported media request");
        assert!(
            matches!(event, Event::Rejected { .. }),
            "unsupported request {index} was not rejected: {event:?}"
        );
        rejections.push(event);
    }

    // The engine serves the text it does support.
    let mut served = engine
        .submit(text_request(1, 8))
        .expect("the text request is accepted");
    let deadline = Instant::now() + ANSWER;
    let mut terminal = None;
    while Instant::now() < deadline && terminal.is_none() {
        match served.try_recv() {
            // A rejected or failed request ends the stream as surely as a
            // finished one; treating only `Finished` as an end is how a
            // rejection reads as a timeout.
            Ok(event) if event.is_terminal() => terminal = Some(event),
            Ok(_) => {}
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    assert!(
        matches!(terminal, Some(Event::Finished { .. })),
        "the engine stops serving text after rejecting media: {terminal:?}"
    );

    engine.shutdown();
    assert!(!engine.is_dead(), "media rejection killed the engine");
}
