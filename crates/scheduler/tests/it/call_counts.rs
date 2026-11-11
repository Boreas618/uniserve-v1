//! §4.4's call discipline, stated as the numbers an epoch can be asked for.
//!
//! The design fixes how often the mechanism asks a policy: `observe` and
//! `dispatch` once per step whatever they respond, and the limits once per
//! instance. Those calls are the mechanism's own, and a policy that traps on a
//! breach cannot be the witness here: a policy that names no work serves no
//! request, so it takes no steps either. The witness is the epoch's own
//! arithmetic instead - `Progress.usage` reports the policy's calls by decision,
//! beside the step the scheduler reached - and an instance's life is exactly one
//! instantiation, one evaluation of the decision set, one declaration of its
//! limits, and one `observe` and one `dispatch` per step: `observe == step` and
//! `dispatch == step + 1`.
//!
//! So the epoch here runs the real `fcfs` package over a pool that responds to every
//! batch, serves a request through it, and then reads the counts. A mechanism
//! that asked a step's halves twice, or asked a decision outside a step, would
//! report it in the arithmetic; a policy that failed would report itself.
//!
//! The trial a precopy runs is not a step: it asks the decision set once on the
//! state that exists now - `dispatch` without an `observe` - and the instance a
//! switchover adopts is that trial's, which is why the counts start at one
//! dispatch and no observe.
//!
//! What this test does not state: how many *events* the mechanism records. §4.4
//! makes those proportional to state changes, and the events an epoch emits are
//! asserted where they come from, request by request.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ipc::SamplingParams;
use ipc::execution::Batches;
use scheduler::{
    BatchCapacity, Event, GenerationRequest, Parameters, PolicyPackage, PolicySource, Progress,
    SchedulerConfig, SchedulerHandle, Stage,
};
use scheduling::Call;
use testkit::{Extent, FakeBatches, Tokens};

/// How long a command response may take: a precopy loads and instantiates a
/// module on its own thread.
const RESPONSE: Duration = Duration::from_secs(30);

/// How long an epoch may take to reach the step the test asks for.
const STEPS: Duration = Duration::from_secs(20);

/// Blocks until a command responses, and responds to its result.
fn response<T>(receiver: crossbeam_channel::Receiver<Result<T, String>>) -> Result<T, String> {
    match receiver.recv_timeout(RESPONSE) {
        Ok(result) => result,
        Err(error) => panic!("the scheduler did not response: {error}"),
    }
}

/// A pool of one text rank that completes every batch at once, sampling `testkit::SAMPLED` plus each
/// call's place in its batch and standing one token past the extent its row states.
fn serving() -> FakeBatches {
    FakeBatches::new(vec![testkit::pool("model")])
        .with_tokens(Tokens::Positional)
        .with_extent(Extent::OnePast)
}

/// The package the epoch serves under, its arguments the capacity this pool was
/// built for.
fn package(pool: &FakeBatches) -> PolicyPackage {
    let rank = pool
        .inventory()
        .first()
        .and_then(|pool| pool.ranks.first())
        .expect("the pool states its ranks");
    let ceilings = scheduler::ceilings_of(BatchCapacity::of(rank));
    PolicyPackage {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: Parameters {
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

/// The epoch's policy: prepared, then switched in on the handle's own command
/// surface. Both responds are checked, so a trial that broke the discipline would
/// fail the install rather than the test's first assertion.
fn install(handle: &SchedulerHandle, serving: &FakeBatches) {
    let source: Arc<dyn PolicySource + Send + Sync> =
        Arc::new(scheduler::PolicyBytes::new(crate::support::policy_module()));
    response(handle.precopy_policy(package(serving), source)).expect("the policy is precopied");
    response(handle.stage(Stage::Switchover)).expect("the policy serves");
}

/// The epoch's progress once it has taken at least `steps` steps.
fn reached(handle: &SchedulerHandle, steps: u64) -> Progress {
    let deadline = Instant::now() + STEPS;
    loop {
        let progress = handle
            .progress()
            .recv_timeout(RESPONSE)
            .expect("the scheduler responds with its progress");
        if progress.step >= steps || Instant::now() > deadline {
            return progress;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_step_asks_the_policy_to_observe_and_dispatch_once_each() {
    let serving = Arc::new(serving());
    let batches = Arc::clone(&serving) as Arc<dyn Batches>;
    let args = testkit::args("call-counts");
    let limits = SchedulerConfig {
        max_running_requests: 4,
        ..SchedulerConfig::default()
    };
    let handle =
        SchedulerHandle::over_batches(&args, limits, batches, scheduler::CommandWaker::noop())
            .expect("the pool serves");

    // The trial asks the decision set once on the state that exists now - a
    // `dispatch` and no `observe`, since observing a step's events is the step's
    // - and a trial leaves the package as it found it, so the candidate a
    // switchover adopts is a second instance of the same module. The trial's
    // calls are the epoch's policy's and count (§7.3), from the instance that
    // made them, while the serving instance starts its own life: two
    // instantiations, two `limits`, and one `observe` and `dispatch` per step
    // the serving instance has taken. Stated as a relation between the counts, a
    // step this epoch takes on its own does not change it.
    install(&handle, &serving);
    let installed = reached(&handle, 0);
    let usage = installed.usage.expect("the policy has served");
    assert_eq!(
        usage.decisions.calls(Call::Init),
        2,
        "the trial's instance and the serving one"
    );
    assert_eq!(
        usage.decisions.calls(Call::Observe),
        installed.step,
        "one observe per step and none anywhere else"
    );
    assert_eq!(
        usage.decisions.calls(Call::Dispatch),
        installed.step + 1,
        "one dispatch per step, the trial's one evaluation beside them"
    );
    assert_eq!(
        usage.decisions.calls(Call::Limits),
        2,
        "the limits are declared once in each instance's life"
    );

    // Requests make the epoch step: the real package admits them, the pool
    // responds to their batches, and every step is one `observe` and one
    // `dispatch`.
    handle
        .generation()
        .store(1, std::sync::atomic::Ordering::Relaxed);
    let mut events = handle
        .submit(text_request(1, 8, 4))
        .expect("the request is submitted");
    let deadline = Instant::now() + STEPS;
    let mut tokens = 0usize;
    while tokens == 0 && Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::TextToken { .. }) => tokens += 1,
            Ok(Event::TextTokens { ids }) => tokens += ids.len(),
            Ok(_) => {}
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    assert!(
        tokens > 0,
        "the request was served, so the epoch took the steps the counts are read on"
    );

    // The counters say what the epoch was asked: one observe and one dispatch
    // per step, and one evaluation of the decision set before the first of them.
    let progress = reached(&handle, 2);
    let usage = progress.usage.expect("the policy has served");
    assert!(
        progress.step >= 2,
        "the epoch took steps: {}",
        progress.step
    );
    assert_eq!(
        usage.decisions.calls(Call::Observe),
        progress.step,
        "still one observe per step, and none anywhere else"
    );
    assert_eq!(
        usage.decisions.calls(Call::Dispatch),
        progress.step + 1,
        "still one dispatch per step, the trial's evaluation beside them"
    );
    assert_eq!(
        usage.calls,
        usage.decisions.each().map(|(_, calls)| calls).sum::<u64>(),
        "no call is counted twice: the epoch's total is its instances' decisions, summed, and \
         this epoch's only trial is the one it serves under"
    );
    assert!(
        matches!(
            handle.take_policy_report().try_recv(),
            Err(crossbeam_channel::TryRecvError::Empty)
        ),
        "no policy failed, so the counts are a whole life's and not a truncated one"
    );
}
