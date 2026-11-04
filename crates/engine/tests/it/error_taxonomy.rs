//! §4.4's error taxonomy, driven through the production step.
//!
//! Each case is a real policy program in WebAssembly text, switched in over a
//! good one and asked for a step, so the failure travels the path a serving
//! policy's failure travels: the call fails (a trap, exhausted fuel, limits over
//! the build config's ceilings, a decision the validators refuse), the
//! mechanism reports it once with its reason, the previous program is
//! instantiated on the state the mechanism holds, and that program answers the
//! next decision. Nothing here is a test-only branch: the scheduler, the switch
//! and the validators are the ones the epoch runs.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use engine::{
    CommandSender, GenerationConstraint, GenerationRequest, PolicyBytes, PolicyProgram,
    PolicySource, SimEngine, SimExecutor,
};
use ipc::{ImageParams, SamplingParams};
use scheduling::guest::decision::{Plan, Work};
use scheduling::guest::Nanos;
use scheduling::guest::limits::Limits;
use scheduling::guest::view::{ExpertLoad, ExpertOrder};
use scheduling::guest::RequestId;
use scheduling::guest::wire::Call;
use scheduling::guest::PhaseId;

use crate::support;

/// Where a toy keeps its reply buffers.
const MESSAGE: u32 = 4096;
const PLAN: u32 = 8192;
const LIMITS: u32 = 12288;
const VICTIMS: u32 = 16384;
const PAGES: u32 = 20480;
const EXPERTS: u32 = 24576;

/// How long a command answer may take.
const ANSWER: Duration = Duration::from_secs(30);

/// What a toy's `observe` does.
#[derive(Clone, Copy)]
enum Observe {
    Answers,
    Traps,
}

/// What a toy's `dispatch` does.
#[derive(Clone, Copy)]
enum Dispatch {
    Plan,
    /// Runs until the fuel is gone, which §4.4 counts as a failed call.
    Burns,
}

/// A toy policy program: the reply for every decision is stated, and the two
/// calls that can fail fail the way the case needs.
struct Toy {
    limits: Limits,
    plan: Plan,
    experts: ExpertOrder,
    observe: Observe,
    dispatch: Dispatch,
}

impl Toy {
    /// A policy that answers every decision with nothing.
    fn idle() -> Self {
        Self {
            limits: Limits {
                chunked_prefill_size: 128,
                max_batch: 1,
                max_num_batched_tokens: 128,
                prefetch_min_tokens: 256,
                ..Limits::default()
            },
            plan: Plan::default(),
            // A placement the previous program is recognizable by.
            experts: ExpertOrder { placement: vec![1, 0] },
            observe: Observe::Answers,
            dispatch: Dispatch::Plan,
        }
    }

    fn wasm(&self) -> Vec<u8> {
        let plan = serde_json::to_string(&self.plan).expect("a plan is JSON");
        let limits = serde_json::to_string(&self.limits).expect("limits are JSON");
        let experts = serde_json::to_string(&self.experts).expect("an expert order is JSON");
        let observe = match self.observe {
            Observe::Answers => "(func (export \"sched_observe\") (param i32 i32) (result i64) i64.const 0)".to_owned(),
            Observe::Traps => "(func (export \"sched_observe\") (param i32 i32) (result i64) unreachable)".to_owned(),
        };
        let dispatch = match self.dispatch {
            Dispatch::Plan => format!(
                "(func (export \"{}\") (result i64) i64.const {})",
                Call::Dispatch.export(),
                packed(PLAN, plan.len())
            ),
            Dispatch::Burns => format!(
                "(func (export \"{}\") (result i64) (loop (br 0)) i64.const 0)",
                Call::Dispatch.export()
            ),
        };
        let source = format!(
            "(module (memory (export \"memory\") 1) \
             (data (i32.const {PLAN}) \"{plan}\") \
             (data (i32.const {LIMITS}) \"{limits}\") \
             (data (i32.const {EXPERTS}) \"{experts}\") \
             (data (i32.const {VICTIMS}) \"[]\") \
             (data (i32.const {PAGES}) \"[]\") \
             (func (export \"sched_alloc\") (param i32) (result i32) i32.const {MESSAGE}) \
             (func (export \"sched_free\") (param i32 i32)) \
             (func (export \"sched_init\") (param i32 i32) (result i64) i64.const 0) \
             {observe} \
             {dispatch} \
             (func (export \"{limits_call}\") (param i32 i32) (result i64) i64.const {limits_answer}) \
             (func (export \"{reclaim_call}\") (param i32 i32) (result i64) i64.const {reclaim_answer}) \
             (func (export \"{warm_call}\") (param i32 i32) (result i64) i64.const {warm_answer}) \
             (func (export \"{experts_call}\") (param i32 i32) (result i64) i64.const {experts_answer}) \
             (func (export \"{route_call}\") (param i32 i32) (result i32) i32.const 0) \
             (func (export \"{place_call}\") (param i32 i32) (result i32) i32.const 0))",
            plan = escape(&plan),
            limits = escape(&limits),
            experts = escape(&experts),
            limits_call = Call::Limits.export(),
            limits_answer = packed(LIMITS, limits.len()),
            reclaim_call = Call::Reclaim.export(),
            reclaim_answer = packed(VICTIMS, 2),
            warm_call = Call::Warm.export(),
            warm_answer = packed(PAGES, 2),
            experts_call = Call::Experts.export(),
            experts_answer = packed(EXPERTS, experts.len()),
            route_call = Call::Route.export(),
            place_call = Call::Place.export(),
        );
        wat::parse_str(&source).expect("the toy policy module is valid WebAssembly text")
    }
}

/// A work entry naming the request in `WORK`, for the invalid-decision case.
fn unknown_phase() -> Plan {
    Plan {
        work: vec![Work {
            request: scheduling::guest::RequestId(9),
            phase: PhaseId(999),
            tokens: 4,
        }],
        ..Plan::default()
    }
}

/// A reply of `len` bytes at `offset`, as the call protocol packs it.
fn packed(offset: u32, len: usize) -> u64 {
    (u64::from(offset) << 32) | u64::try_from(len).expect("a reply fits u64")
}

/// `text` escaped for a WebAssembly text data segment.
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// A scheduler thread serving one program, with the one it replaced kept for a
/// revert.
struct World {
    handle: CommandSender,
    thread: thread::JoinHandle<()>,
}

impl World {
    /// Starts a scheduler serving `good`, committed, and then puts `failing` in
    /// so that a failure has a program to serve on under.
    fn with_previous(good: Toy, failing: Toy) -> Self {
        let mut sim = SimEngine::new();
        sim.set_text_len(16);
        let scheduler = support::scheduler(
            Box::new(SimExecutor::new(sim)),
            SimEngine::special_tokens(),
            8,
        )
        .expect("the simulated scheduler starts");
        let (tx, rx) = crossbeam_channel::unbounded();
        let thread = thread::spawn(move || {
            let _ = scheduler.run(rx);
        });
        let world = Self {
            handle: CommandSender::new(tx),
            thread,
        };
        world.install("fcfs", 1, good);
        world.handle.commit_policy();
        world.install("fcfs", 2, failing);
        world
    }

    fn install(&self, backend: &str, revision: u32, toy: Toy) {
        let program = PolicyProgram {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
            stated: Default::default(),
        };
        let source: Arc<dyn PolicySource + Send + Sync> = Arc::new(PolicyBytes::new(toy.wasm()));
        answer(self.handle.precopy_policy(program, source)).expect("the program's precopy");
        answer(self.handle.switchover_policy()).expect("the switchover puts it in");
    }

    /// Waits for the failure report the mechanism records for a failed policy.
    fn failed(&self) -> engine::PolicyReverted {
        eventually(|| self.handle.take_policy_report().recv().ok().flatten())
    }

    /// The placement the serving program answers a published load with: after a
    /// failure, the program that serves on is the one it reverted to.
    fn serving_order(&self) -> ExpertOrder {
        self.handle.publish_expert_load(Some(ExpertLoad {
            ranks: 2,
            experts: 2,
            tokens: vec![1, 1],
        }));
        eventually(|| self.handle.take_expert_order())
    }
}

/// Blocks until a stage command answers, and answers its result without the
/// stage's own report.
fn answer<T>(receiver: crossbeam_channel::Receiver<Result<T, String>>) -> Result<(), String> {
    match receiver.recv_timeout(ANSWER) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) => Err(error),
        Err(error) => Err(format!("the scheduler did not answer: {error}")),
    }
}

/// Polls `poll` until it answers `Some`.
fn eventually<T>(mut poll: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + ANSWER;
    loop {
        if let Some(value) = poll() {
            return value;
        }
        assert!(Instant::now() < deadline, "the scheduler never answered");
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_trap_in_observe_fails_the_policy() {
    let world = World::with_previous(
        Toy::idle(),
        Toy {
            observe: Observe::Traps,
            ..Toy::idle()
        },
    );
    let report = world.failed();
    assert!(
        report.reason.to_lowercase().contains("trap")
            || report.reason.to_lowercase().contains("observ"),
        "the reason names the failed call: {}",
        report.reason
    );
    // The previous program answers the next decision: its placement is the one
    // handed over, and the failed program's is not.
    let order = world.serving_order();
    assert_eq!(order.placement, vec![1, 0]);
    assert!(
        world.handle.take_policy_report().recv().ok().flatten().is_none(),
        "a policy fails once, not on every step"
    );
}

#[test]
fn exhausted_fuel_in_dispatch_fails_the_policy() {
    let world = World::with_previous(
        Toy::idle(),
        Toy {
            dispatch: Dispatch::Burns,
            ..Toy::idle()
        },
    );
    let report = world.failed();
    let reason = report.reason.to_lowercase();
    assert!(
        reason.contains("fuel") || reason.contains("exhaust") || reason.contains("trap"),
        "the reason names the spent budget: {}",
        report.reason
    );
    assert_eq!(world.serving_order().placement, vec![1, 0]);
}

#[test]
fn limits_over_the_build_ceiling_fail_the_policy() {
    let world = World::with_previous(
        Toy::idle(),
        Toy {
            limits: Limits {
                // The simulated deployment was built for eight rows at most.
                max_batch: 1 << 20,
                ..Toy::idle().limits
            },
            ..Toy::idle()
        },
    );
    let report = world.failed();
    let reason = report.reason.to_lowercase();
    assert!(
        reason.contains("limit") || reason.contains("ceiling"),
        "the reason names the limits: {}",
        report.reason
    );
    assert_eq!(world.serving_order().placement, vec![1, 0]);
}

#[test]
fn an_invalid_decision_fails_the_policy_and_applies_nothing() {
    let world = World::with_previous(
        Toy::idle(),
        Toy {
            plan: unknown_phase(),
            ..Toy::idle()
        },
    );
    let report = world.failed();
    let reason = report.reason.to_lowercase();
    assert!(
        reason.contains("phase") || reason.contains("not one of this epoch"),
        "the reason names the refused decision: {}",
        report.reason
    );
    // Nothing of the refused plan ran: the program that serves on answers, so
    // the step the refused plan belonged to decided nothing.
    assert_eq!(world.serving_order().placement, vec![1, 0]);
}

#[test]
fn an_invalid_expert_order_fails_the_policy() {
    let world = World::with_previous(
        Toy::idle(),
        Toy {
            experts: ExpertOrder {
                placement: vec![0, 0],
            },
            ..Toy::idle()
        },
    );
    world.handle.publish_expert_load(Some(ExpertLoad {
        ranks: 2,
        experts: 2,
        tokens: vec![1, 1],
    }));
    let report = world.failed();
    assert!(
        report.reason.to_lowercase().contains("expert"),
        "the reason names the order: {}",
        report.reason
    );
    assert_eq!(world.serving_order().placement, vec![1, 0]);
}

/// A policy that admits `admit` and gives it no work, with `timeout` as its
/// starvation deadline: §4.4's sixth failure is a resident request the policy
/// never advanced.
fn starving(admit: Vec<RequestId>, timeout: Nanos) -> Toy {
    Toy {
        limits: Limits {
            timeout,
            ..Toy::idle().limits
        },
        plan: Plan {
            admit,
            ..Plan::default()
        },
        // Distinct from the previous program's, so the revert is visible.
        experts: ExpertOrder {
            placement: vec![0, 1],
        },
        ..Toy::idle()
    }
}

/// A text request of `prompt` prompt tokens that generates up to `tokens`.
fn text_request(id: u64, prompt: usize, tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: ipc::RequestId(id),
        prompt_token_ids: vec![7; prompt],
        multimodal_inputs: Default::default(),
        negative_prompt_token_ids: Vec::new(),
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: tokens,
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

#[test]
fn a_starved_request_fails_the_policy() {
    // One millisecond of starvation: the request is admitted by the plan and
    // never given work, and §4.4's watchdog is the mechanism's, not the
    // policy's, so a policy that answers every call correctly still fails.
    let world = World::with_previous(
        Toy::idle(),
        starving(vec![RequestId(1)], Nanos(1_000_000)),
    );
    let mut events = world
        .handle
        .submit(text_request(1, 8, 4))
        .expect("the request is submitted");
    let report = world.failed();
    assert!(
        report.reason.to_lowercase().contains("starv")
            && report.reason.contains('1'),
        "the reason names the starved request: {}",
        report.reason
    );
    // The program that serves on is the one before the starved one.
    assert_eq!(world.serving_order().placement, vec![1, 0]);
    // The starved request is still the epoch's: the revert moves the policy,
    // not the request, and the previous program may now serve it.
    assert!(
        !events.try_recv().is_ok_and(|event| event.is_terminal()),
        "the request was not ended by the policy's failure"
    );
}
