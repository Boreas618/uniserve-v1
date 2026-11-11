//! The in-process boot: a main process that holds its own devices starts its
//! scheduler from its launch arguments and the limits of its pool
//! (`SchedulerConfig::of`), and serves over the worker's own batch path
//! (`SchedulerHandle::over_batches`), here with no device.
//!
//! This test states what that composition has to produce: a handle the epoch's
//! policy commands reach, over a pool whose
//! inventory is the only statement of what it serves, and one request served
//! through it to a token the pool itself sampled.
//!
//! The pool stands in for the worker only: it is handed the mechanism's batch
//! and answers its calls. What the boot derives — the group, the budgets, the
//! cache geometry, the tokenizer's bounds — is the code under test.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ipc::SamplingParams;
use ipc::execution::Batches;
use scheduler::{BatchCapacity, Event, GenerationRequest, SchedulerConfig, SchedulerHandle, Stage};
use testkit::{Extent, FakeBatches, SAMPLED, Tokens};

/// How long a command answer may take.
///
/// Generous because the first one is the whole installation of a policy: the
/// tree's `fcfs` module is built for `wasm32-unknown-unknown` on first use and
/// then loaded and instantiated, on a machine that runs this suite's tests in
/// parallel. The bound is the harness's, not the mechanism's; a mechanism that
/// stopped answering would still fail here, only later.
const ANSWER: Duration = Duration::from_secs(60);

/// How long a request may take to emit its first token.
const FIRST_TOKEN: Duration = Duration::from_secs(20);

/// Blocks until a command answers, and answers its result.
fn answer<T>(receiver: crossbeam_channel::Receiver<Result<T, String>>) -> Result<T, String> {
    match receiver.recv_timeout(ANSWER) {
        Ok(result) => result,
        Err(error) => panic!("the scheduler did not answer: {error}"),
    }
}

/// The arguments one main process starts its scheduler with for a text deployment of one rank,
/// and the mechanism's limits of its pool: a tokenizer whose end of text is the simulation's.
fn launch() -> (proto::config::EngineArgs, SchedulerConfig) {
    let mut args = testkit::args("boot-model");
    args.engine_config.max_running_requests = std::num::NonZeroUsize::new(4).expect("positive");
    // The budgets the pool below is built for, stated rather than left to a default: a plan may
    // not exceed what the epoch was built for, so a budget tighter than the pool's would refuse
    // the very batch a request needs and read as "the policy served nothing".
    let limits = SchedulerConfig::of(&args.engine_config, 16, 512, vec![1, 8]);
    (args, limits)
}

/// A greedy text request of `prompt` tokens that generates at most `max_tokens`.
fn text_request(id: u64, prompt: usize, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: ipc::RequestId(id),
        prompt_token_ids: vec![7; prompt],
        multimodal_inputs: Default::default(),
        sampling: SamplingParams::default(),
        max_und_tokens: max_tokens,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        generated_token_ids: Vec::new(),
        adapter: 0,
    }
}

/// A pool of one text rank that completes every batch at once, sampling [`SAMPLED`] plus each
/// call's place in its batch and standing one token past the extent its row states.
fn serving() -> FakeBatches {
    FakeBatches::new(vec![testkit::pool("model")])
        .with_tokens(Tokens::Positional)
        .with_extent(Extent::OnePast)
}

/// The `fcfs` package, instantiated for the capacity this pool was built for:
/// a policy declares its own limits and the mechanism refuses a declaration
/// above the ceilings the ranks were started for.
fn policy_package(pool: &FakeBatches) -> scheduler::PolicyPackage {
    let rank = pool
        .inventory()
        .first()
        .and_then(|pool| pool.ranks.first())
        .expect("the pool states its ranks");
    let ceilings = scheduler::ceilings_of(BatchCapacity::of(rank));
    scheduler::PolicyPackage {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: scheduler::Parameters {
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
/// command surface, which the launched path's is.
fn install(handle: &SchedulerHandle, pool: &FakeBatches) {
    let source: Arc<dyn scheduler::PolicySource + Send + Sync> =
        Arc::new(scheduler::PolicyBytes::new(crate::support::policy_module()));
    answer(handle.precopy_policy(policy_package(pool), source)).expect("the policy is precopied");
    answer(handle.stage(Stage::Switchover)).expect("the policy serves");
}

#[test]
fn the_facts_boot_a_main_process_that_serves_a_request() {
    let serving = Arc::new(serving());
    let batches = Arc::clone(&serving) as Arc<dyn Batches>;
    // The boot's whole shape: the facts of a launch file, the worker's batch
    // path, one handle. Nothing is launched: the pool is the rank.
    let (args, limits) = launch();
    let handle = SchedulerHandle::over_batches(&args, limits, batches)
        .expect("the launch boots a scheduler");

    // The handle is the launched path's: its commands answer, and a stage waits
    // for a boundary.
    let progress = handle
        .progress()
        .recv_timeout(ANSWER)
        .expect("the scheduler answers with its progress");
    assert_eq!(progress.step, 0, "no step ran before a policy served");

    install(&handle, &serving);
    // The kernels this epoch serves on, as the control thread's switchover arm
    // publishes it.
    handle
        .generation()
        .store(3, std::sync::atomic::Ordering::Relaxed);

    let mut events = handle
        .submit(text_request(1, 8, 4))
        .expect("the request is submitted");

    // A token arrives: the booted scheduler admitted the request, the policy
    // planned it, the batch path served it, and the report reached the request.
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
    let first = first_token.expect("the booted scheduler served a token");
    assert_eq!(
        first, SAMPLED,
        "the token is the one this pool sampled, not an invented one"
    );
    assert!(
        !serving.submitted().is_empty(),
        "the worker was handed the work the facts' budgets admit"
    );
}

/// The serving path's warm-up of a main process on CUDA devices runs its requests only once the
/// process opens admission, not when the epoch's first policy starts serving: the switchover that
/// installs the policy only queues the device's bucket captures, and a warm-up batch reaching the
/// device before them finds no graph to replay, which fails the worker.
#[test]
fn the_serving_warmup_waits_for_admission_to_open() {
    let serving = Arc::new(serving());
    let batches = Arc::clone(&serving) as Arc<dyn Batches>;
    let (mut args, limits) = launch();
    // The warm-up runs only where the ranks execute on CUDA devices.
    for group in &mut args.engine_config.parallel.worker_groups {
        for rank in &mut group.ranks {
            rank.device = "cuda:0".to_owned();
        }
    }
    let handle = SchedulerHandle::over_batches(&args, limits, batches)
        .expect("the launch boots a scheduler");

    install(&handle, &serving);
    // The policy serves and could form a batch from here on; the device's captures are still
    // running, so nothing may reach it. A warm-up started by the policy would have submitted its
    // prefill by now: the scheduler plans a queued request at its next step.
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        serving.submitted().is_empty(),
        "a warm-up batch reached the device before admission opened"
    );

    // Admission opens: the warm-up's prefill and decode reach the device.
    handle.warm_serving_path();
    let deadline = Instant::now() + FIRST_TOKEN;
    while serving.submitted().len() < 2 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let submitted = serving.submitted();
    assert!(
        submitted.len() >= 2,
        "the opened warm-up runs a prefill and a decode, but the device received {} batches",
        submitted.len()
    );
}
