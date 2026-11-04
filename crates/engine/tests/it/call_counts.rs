//! §4.4's call discipline, stated as the numbers a world can be asked for.
//!
//! The design fixes how often the mechanism asks a policy: `observe` and
//! `dispatch` once per step whatever they answer, and the limits once per
//! instance. Those calls are the mechanism's own, and a policy that traps on a
//! breach cannot be the witness here: a policy that names no work serves no
//! request, so it takes no steps either. The witness is the epoch's own
//! arithmetic instead - `Progress.usage` reports the policy's calls by decision,
//! beside the step the scheduler reached - and an instance's life is exactly one
//! instantiation, one evaluation of the decision set, one declaration of its
//! limits, and one `observe` and one `dispatch` per step: `observe == step` and
//! `dispatch == step + 1`.
//!
//! So the world here runs the real `fcfs` program over a pool that answers every
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
//! makes those proportional to state changes, and the events a world emits are
//! asserted where they come from, request by request.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use engine::{
    BatchCapacity, EngineConfig, Event, GenerationConstraint, GenerationRequest, Parameters,
    PolicyProgram, PolicySource, Progress, SchedulerHandle, Stage,
};
use executor::batch::{
    Batch, BatchOutput, Batches, CallOutput, Inventory, Placement, RowState, Submission,
    SubmissionId,
};
use executor::completion::Outcome;
use ipc::{
    ComponentConfig, ImageParams, SamplingParams, TimingCounters, WorkerEndpoint, WorkerInfo,
};
use scheduling::Call;

/// How long a command answer may take: a precopy loads and instantiates a
/// module on its own thread.
const ANSWER: Duration = Duration::from_secs(30);

/// How long a world may take to reach the step the test asks for.
const STEPS: Duration = Duration::from_secs(20);

/// The token this pool samples for the `index`th call of every batch.
const SAMPLED: u32 = 1000;

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

/// A pool that serves every batch with the values the mechanism reads back.
struct Serving {
    inventory: Vec<Inventory>,
    next: Mutex<u64>,
}

impl Serving {
    fn new(inventory: Vec<Inventory>) -> Self {
        Self {
            inventory,
            next: Mutex::new(1),
        }
    }

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
                        step_index: seen + 1,
                        cache_computed_len: seen + 1,
                        cache_visible_len: seen + 1,
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
        Ok(self.serve(batch))
    }

    fn shadow(&self, batch: Batch) -> Result<Submission, executor::Error> {
        Ok(self.serve(batch))
    }

    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }
}

/// The program the epoch serves under, its arguments the capacity this pool was
/// built for.
fn program(pool: &Serving) -> PolicyProgram {
    let rank = pool
        .inventory()
        .first()
        .and_then(|pool| pool.ranks.first())
        .expect("the pool states its ranks");
    let ceilings = engine::ceilings_of(BatchCapacity::of(rank));
    PolicyProgram {
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

/// The epoch's policy: prepared, then switched in on the handle's own command
/// surface. Both answers are checked, so a trial that broke the discipline would
/// fail the install rather than the test's first assertion.
fn install(handle: &SchedulerHandle, serving: &Serving) {
    let source: Arc<dyn PolicySource + Send + Sync> =
        Arc::new(engine::PolicyBytes::new(crate::support::policy_module()));
    answer(handle.precopy_policy(program(serving), source)).expect("the policy is precopied");
    answer(handle.stage(Stage::Switchover)).expect("the policy serves");
}

/// The epoch's progress once it has taken at least `steps` steps.
fn reached(handle: &SchedulerHandle, steps: u64) -> Progress {
    let deadline = Instant::now() + STEPS;
    loop {
        let progress = handle
            .progress()
            .recv_timeout(ANSWER)
            .expect("the scheduler answers with its progress");
        if progress.step >= steps || Instant::now() > deadline {
            return progress;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_step_asks_the_policy_to_observe_and_dispatch_once_each() {
    let serving = Arc::new(Serving::new(vec![pool()]));
    let batches: Arc<dyn Batches> = serving.clone();
    let mut config = EngineConfig::sim("call-counts");
    config.max_running_requests = 4;
    let handle = SchedulerHandle::over_batches(config, batches).expect("the pool serves");

    // The trial asks the decision set once on the state that exists now - a
    // `dispatch` and no `observe`, since observing a step's events is the step's
    // - and the instance a switchover adopts is the trial's candidate. So the
    // serving instance's life is one instantiation, one evaluation, and one
    // `observe` and `dispatch` per step it has taken since: stated as a relation
    // between the counts, a step this world takes on its own does not change it.
    install(&handle, &serving);
    let installed = reached(&handle, 0);
    let usage = installed.usage.expect("the policy has served");
    assert_eq!(usage.decisions.calls(Call::Init), 1, "one instantiation");
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
        1,
        "the limits are declared once in an instance's life"
    );

    // Requests make the world step: the real program admits them, the pool
    // answers their batches, and every step is one `observe` and one
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
        "the request was served, so the world took the steps the counts are read on"
    );

    // The counters say what the world was asked: one observe and one dispatch
    // per step, and one evaluation of the decision set before the first of them.
    let progress = reached(&handle, 2);
    let usage = progress.usage.expect("the policy has served");
    assert!(
        progress.step >= 2,
        "the world took steps: {}",
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
         this world's only trial is the one it serves under"
    );
    assert!(
        matches!(
            handle.take_policy_report().try_recv(),
            Err(crossbeam_channel::TryRecvError::Empty)
        ),
        "no policy failed, so the counts are a whole life's and not a truncated one"
    );
}
