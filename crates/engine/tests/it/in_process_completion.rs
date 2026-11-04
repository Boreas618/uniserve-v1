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

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use engine::{
    BatchCapacity, EngineConfig, Event, GenerationConstraint, GenerationRequest, SchedulerHandle,
    Stage,
};
use executor::batch::{
    Batch, BatchOutput, Batches, CallOutput, Inventory, Placement, RowState, Submission,
    SubmissionId,
};
use executor::completion::Outcome;
use ipc::{
    ComponentConfig, ImageParams, SamplingParams, TimingCounters, WorkerEndpoint, WorkerInfo,
};

/// How long a command answer may take.
///
/// Generous because the first one is the whole boot of a policy: the precopy
/// compiles the `fcfs` module inside a debug build, on a machine that runs this
/// suite's tests in parallel. The bound is the harness's, not the mechanism's;
/// a mechanism that stopped answering would still fail here, only later.
const ANSWER: Duration = Duration::from_secs(60);

/// How long a request may take to emit its first token.
const FIRST_TOKEN: Duration = Duration::from_secs(60);

/// The token this pool samples for the `index`th call of every batch.
const SAMPLED: u32 = 1000;

/// Blocks until a command answers, and answers its result.
fn answer<T>(receiver: crossbeam_channel::Receiver<Result<T, String>>) -> Result<T, String> {
    match receiver.recv_timeout(ANSWER) {
        Ok(result) => result,
        Err(error) => panic!("the scheduler did not answer: {error}"),
    }
}

/// One rank of a text pool: prefill and decode, eight request rows, and the
/// batch bounds the launch stated.
fn rank(host: &str) -> WorkerInfo {
    WorkerInfo {
        endpoint: WorkerEndpoint {
            host: host.to_owned(),
            address_space: format!("{host}-space"),
            incarnation: format!("{host}-rank"),
        },
        supported_calls: vec![
            ipc::CallKind::Forward(ipc::ForwardMode::Prefill),
            ipc::CallKind::Forward(ipc::ForwardMode::Decode),
        ],
        request_slots: 8,
        max_batch_calls: 16,
        max_batch_tokens: 512,
        prefix_cache: None,
        buffer_pool_bytes: 1 << 20,
        ..WorkerInfo::default()
    }
}

/// A text pool of one rank named `model`.
fn pool() -> Inventory {
    Inventory {
        group: "model".to_owned(),
        ranks: vec![rank("gpu-a")],
        placement: Placement {
            queue_depth: 1,
            ranks: 1,
            components: std::collections::BTreeMap::from([(
                ipc::DEFAULT_COMPONENT.to_owned(),
                ComponentConfig {
                    ranks: vec![0],
                    parallel_config: Default::default(),
                    units_per_rank: 1,
                    weights: Default::default(),
                    distribution: None,
                },
            )]),
        },
        max_model_len: 4096,
    }
}

/// The configuration a main process holds: the simulation's facts, since this
/// test runs no device.
fn config() -> EngineConfig {
    let mut config = EngineConfig::sim("in-process-completion");
    config.max_running_requests = 4;
    config
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

/// A pool that serves every batch with the values the mechanism reads back.
struct Serving {
    inventory: Vec<Inventory>,
    /// Every batch the mechanism handed over, in submission order: what the
    /// mechanism asked for is half of what this test observes.
    submitted: Mutex<Vec<Batch>>,
    next: Mutex<u64>,
}

impl Serving {
    fn new(inventory: Vec<Inventory>) -> Self {
        Self {
            inventory,
            submitted: Mutex::new(Vec::new()),
            next: Mutex::new(1),
        }
    }

    fn submitted(&self) -> Vec<Batch> {
        self.submitted.lock().expect("the pool's batches").clone()
    }

    /// The batch's calls, each answered as a completed call that sampled
    /// [`SAMPLED`] and leaves the cache extent its row was stated with.
    ///
    /// The extents are the columns', not invented: `seq_lens` is a row's
    /// attention length, cached prefix plus its own query, which is exactly the
    /// cache the call leaves behind - so a prefill of eight prompt tokens
    /// leaves eight, and a decode of one token on top leaves nine. The
    /// mechanism's next call reads them to know what a successor may read.
    fn serve(&self, batch: Batch) -> Submission {
        let calls = batch
            .calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                let seen = batch
                    .columns
                    .rows
                    .seq_lens
                    .get(index)
                    .copied()
                    .unwrap_or(0)
                    .max(0) as u32;
                CallOutput::completed(
                    *call,
                    vec![SAMPLED + index as u32],
                    RowState {
                        // Text carries no denoising trajectory: the step index
                        // is a media row's, and this pool serves forwards.
                        step_index: 0,
                        cache_computed_len: seen,
                        cache_visible_len: seen,
                    },
                )
            })
            .collect::<Vec<_>>();
        let mut next = self.next.lock().expect("the pool's submissions");
        let submission = Submission::completed(
            SubmissionId(*next),
            Outcome::Success(BatchOutput {
                calls,
                timing: TimingCounters::default(),
            }),
        );
        *next += 1;
        submission
    }
}

impl Batches for Serving {
    fn submit(&self, batch: Batch) -> Result<Submission, executor::Error> {
        self.submitted
            .lock()
            .expect("the pool's batches")
            .push(batch.clone());
        Ok(self.serve(batch))
    }

    fn shadow(&self, batch: Batch) -> Result<Submission, executor::Error> {
        Ok(self.serve(batch))
    }

    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }
}

/// The `fcfs` program, instantiated for the capacity this pool was built for:
/// a policy declares its own limits and the mechanism refuses a declaration
/// above the ceilings the ranks were started for.
fn policy_program(pool: &Serving) -> engine::PolicyProgram {
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
fn install(handle: &SchedulerHandle, pool: &Serving) {
    let source: Arc<dyn engine::PolicySource + Send + Sync> =
        Arc::new(engine::PolicyBytes::new(crate::support::policy_module()));
    answer(handle.precopy_policy(policy_program(pool), source)).expect("the policy is precopied");
    answer(handle.stage(Stage::Switchover)).expect("the policy serves");
}

#[test]
fn a_request_is_served_through_the_executors_own_batch_path() {
    let serving = Arc::new(Serving::new(vec![pool()]));
    let batches: Arc<dyn Batches> = serving.clone();
    let handle = SchedulerHandle::over_batches(config(), batches).expect("the pool serves");
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
    // per step, each stated with the shape its columns are served from.
    let batches = serving.submitted();
    let prefilled = batches
        .iter()
        .any(|batch| batch.shape.variant.name().starts_with("prefill"));
    let decoded = batches
        .iter()
        .any(|batch| batch.shape.variant.name().starts_with("decode"));
    assert!(
        prefilled && decoded,
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

/// A request too big for the pool ends there, and the epoch goes on.
///
/// Admission refuses a queued request whose prompt alone cannot fit the device
/// cache — the mechanism ends it, and the policy never named it. A policy's
/// state is append-only, so unless the mechanism says the request ended the
/// policy keeps it in its own queue and offers it in the next `admit`, where the
/// mechanism refuses it as "not queued": a policy failure, and the whole world
/// stops. This case states the behaviour a caller sees: the oversized request
/// ends with a rejection, and the request behind it is still served.
#[test]
fn a_request_too_big_for_the_pool_does_not_stop_the_epoch() {
    let serving = Arc::new(Serving::new(vec![pool()]));
    let batches: Arc<dyn Batches> = serving.clone();
    let handle = SchedulerHandle::over_batches(config(), batches).expect("the pool serves");
    install(&handle, &serving);

    // The pool is 4095 units of 64 tokens: this prompt cannot fit it however
    // the mechanism chunks it.
    let mut oversized = handle
        .submit(text_request(1, 300_000, 4))
        .expect("the request is submitted");
    let deadline = Instant::now() + FIRST_TOKEN;
    let mut rejected = false;
    while !rejected && Instant::now() < deadline {
        match oversized.try_recv() {
            Ok(Event::Rejected { .. }) => rejected = true,
            Ok(_) => {}
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    assert!(
        rejected,
        "the oversized request is refused, not left queued"
    );

    // The epoch is serving: the request behind it emits a token.
    let mut events = handle
        .submit(text_request(2, 8, 4))
        .expect("the second request is submitted");
    let mut served = None;
    let deadline = Instant::now() + FIRST_TOKEN;
    while served.is_none() && Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::TextToken { id, .. }) => served = Some(id),
            Ok(Event::TextTokens { ids }) => served = ids.first().copied(),
            Ok(Event::Rejected { .. }) | Ok(Event::Error { .. }) => {
                panic!("the request behind the refused one was refused too")
            }
            Ok(_) => {}
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    assert!(
        served.is_some(),
        "the epoch kept serving after the mechanism ended a request"
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
    let serving = Arc::new(Serving::new(vec![pool()]));
    let batches: Arc<dyn Batches> = serving.clone();
    let handle = SchedulerHandle::over_batches(config(), batches).expect("the pool serves");
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
