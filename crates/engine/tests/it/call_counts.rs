//! §4.4's call discipline, as a policy itself can refuse a breach of it.
//!
//! The design fixes how often the mechanism asks a policy: `observe` and
//! `dispatch` once per step whatever they answer, and the limits once per
//! instance. A test cannot see those calls from outside — they are the
//! mechanism's own — so this one puts the discipline *inside* the policy: the
//! module below traps the instant it is asked the same half of a step twice in a
//! row, and a trapped policy is a policy failure, which the world reports. A
//! world that serves many steps under that module therefore proves the
//! discipline by not failing.
//!
//! The trial a precopy runs is not a step: it asks the decision set once on the
//! state that exists now, which is `dispatch` without an `observe`, and the
//! module allows it as the first half it is asked. What the module refuses is
//! two of one kind in a row.
//!
//! The numbers are then read from what the epoch states: `Progress.usage`
//! carries the policy's calls by decision, so the same world is asked what it
//! was actually asked. The trial a precopy runs is the instance a switchover
//! adopts, so its one `observe` and one `dispatch` are visible before the first
//! step and every step adds exactly one of each.
//!
//! What this test does not state: how many *events* the mechanism records. §4.4
//! makes those proportional to state changes, and the events a world emits are
//! asserted where they come from, request by request.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use engine::{
    BatchCapacity, EngineConfig, Event, GenerationConstraint, GenerationRequest, Limits,
    Parameters, PolicyProgram, PolicySource, Progress, SchedulerHandle, Stage,
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

/// The offsets of the module's replies, one 32 KiB block each above the buffer
/// the host writes a call's state into: a world with requests states a larger
/// state than the toys that serve no request, and a reply segment it overwrote
/// would answer a decision with another decision's bytes.
const MESSAGE: u32 = 4096;
const PLAN: u32 = 65536;
const LIMITS: u32 = 98304;
const VICTIMS: u32 = 131072;
const PAGES: u32 = 163840;
const EXPERTS: u32 = 196608;

/// Blocks until a command answers, and answers its result.
fn answer<T>(receiver: crossbeam_channel::Receiver<Result<T, String>>) -> Result<T, String> {
    match receiver.recv_timeout(ANSWER) {
        Ok(result) => result,
        Err(error) => panic!("the scheduler did not answer: {error}"),
    }
}

/// A policy that traps whenever the mechanism asks it out of the order §4.4
/// states, and serves an empty plan otherwise.
///
/// The flags are module globals, so they are per instantiation: the trial's
/// instance has its own, and a switchover does not carry a step over.
fn disciplined() -> Vec<u8> {
    let limits = Limits {
        chunked_prefill_size: 128,
        max_batch: 1,
        max_num_batched_tokens: 128,
        prefetch_min_tokens: 256,
        ..Limits::default()
    };
    let plan = serde_json::to_string(&scheduling::guest::Plan::default()).expect("a plan is JSON");
    let limits = serde_json::to_string(&limits).expect("limits are JSON");
    let experts = serde_json::to_string(&scheduling::guest::ExpertOrder::default())
        .expect("an expert order is JSON");
    let victims = "[]".to_owned();
    let pages = "[]".to_owned();
    let source = format!(
        r#"(module
  (memory (export "memory") 4)
  ;; The limits are declared once in an instance's life, after `init` and before
  ;; it decides anything.
  (global $limited (mut i32) (i32.const 0))
  ;; An observe is open until a dispatch closes it, and a dispatch is the last
  ;; thing asked until an observe follows: the two flags together refuse the same
  ;; half of a step twice in a row. The trial is the exception the design makes
  ;; it - it asks the decision set once on the current state, so a dispatch may
  ;; be the first thing asked after `init`.
  (global $opened (mut i32) (i32.const 0))
  (global $dispatched (mut i32) (i32.const 0))
  (data (i32.const {PLAN}) "{plan}")
  (data (i32.const {LIMITS}) "{limits}")
  (data (i32.const {VICTIMS}) "{victims}")
  (data (i32.const {PAGES}) "{pages}")
  (data (i32.const {EXPERTS}) "{experts}")
  (func (export "sched_alloc") (param i32) (result i32) i32.const {MESSAGE})
  (func (export "sched_free") (param i32 i32))
  (func (export "sched_limits") (result i64)
    global.get $limited
    if unreachable end
    i32.const 1 global.set $limited
    i64.const {limits_reply})
  ;; `init` precedes the limits in an instance's life (`PolicyInstance::new`),
  ;; so it asks nothing of them.
  (func (export "sched_init") (param i32 i32) (result i64) i64.const 0)
  (func (export "sched_observe") (param i32 i32) (result i64)
    global.get $opened
    if unreachable end
    i32.const 1 global.set $opened
    i32.const 0 global.set $dispatched
    i64.const 0)
  (func (export "sched_dispatch") (param i32 i32) (result i64)
    global.get $dispatched
    if unreachable end
    i32.const 0 global.set $opened
    i32.const 1 global.set $dispatched
    i64.const {plan_reply})
  (func (export "sched_reclaim") (param i32 i32) (result i64) i64.const {victims_reply})
  (func (export "sched_warm") (param i32 i32) (result i64) i64.const {pages_reply})
  (func (export "sched_experts") (param i32 i32) (result i64) i64.const {experts_reply})
  (func (export "sched_route") (param i32 i32) (result i32) i32.const 0)
  (func (export "sched_place") (param i32 i32) (result i32) i32.const 0))"#,
        plan = escape(&plan),
        limits = escape(&limits),
        victims = escape(&victims),
        pages = escape(&pages),
        experts = escape(&experts),
        limits_reply = packed(LIMITS, limits.len()),
        plan_reply = packed(PLAN, plan.len()),
        victims_reply = packed(VICTIMS, victims.len()),
        pages_reply = packed(PAGES, pages.len()),
        experts_reply = packed(EXPERTS, experts.len()),
    );
    wat::parse_str(&source).expect("the disciplined policy is valid WebAssembly text")
}

/// A reply of `len` bytes at `offset`, as the call protocol packs it.
fn packed(offset: u32, len: usize) -> u64 {
    (u64::from(offset) << 32) | u64::try_from(len).expect("a reply fits u64")
}

/// `text` escaped for a WebAssembly text data segment.
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
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

/// The program the disciplined module is served as, its arguments the capacity
/// this pool was built for.
fn program(pool: &Serving) -> PolicyProgram {
    let rank = pool
        .inventory()
        .first()
        .and_then(|pool| pool.ranks.first())
        .expect("the pool states its ranks");
    let ceilings = engine::ceilings_of(BatchCapacity::of(rank));
    PolicyProgram {
        backend: "disciplined".to_owned(),
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
        Arc::new(engine::PolicyBytes::new(disciplined()));
    answer(handle.precopy_policy(program(serving), source))
        .expect("the trial obeys the discipline");
    answer(handle.stage(Stage::Switchover)).expect("the switchover obeys the discipline");
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
    // - and the instance a switchover adopts is the trial's candidate, so before
    // the first step the serving instance has been asked exactly that.
    install(&handle, &serving);
    let installed = reached(&handle, 0);
    assert_eq!(installed.step, 0, "no step ran before the first request");
    let usage = installed.usage.expect("the policy has served");
    assert_eq!(usage.decisions.calls(Call::Init), 1, "the trial's init");
    assert_eq!(
        usage.decisions.calls(Call::Dispatch),
        1,
        "the trial's one evaluation of the decision set"
    );
    assert_eq!(
        usage.decisions.calls(Call::Observe),
        0,
        "the trial observes no step, because it is not one"
    );

    // Requests make the world step. Under a policy that traps on a breach, a
    // step taken twice, or a step's halves out of order, is a policy failure:
    // the world would report one and stop serving.
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
        "the request was served under the disciplined policy"
    );

    // The counters say what the world was asked: one observe and one dispatch
    // per step, the trial's included, and nothing else per step.
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
        "one observe per step, and none anywhere else"
    );
    assert_eq!(
        usage.decisions.calls(Call::Dispatch),
        progress.step + 1,
        "one dispatch per step, the trial's one evaluation beside them"
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
        "no policy failed: the discipline held through every step"
    );
}
