//! §4.4's error taxonomy, driven through the production step.
//!
//! Each case is a real policy program in WebAssembly text, and the case states
//! where the mechanism refuses it, which is the one thing the taxonomy divides
//! the six into:
//!
//! - **the precopy trial** refuses five of them. §4.4 has a precopy ask every
//!   decision on the mechanism's current state, so a trap, spent fuel, limits
//!   over the build config's ceilings and a decision or expert order the
//!   validators refuse are found before the program can serve: the command
//!   answers with the reason, the serving program is untouched, and no revert
//!   happens because nothing was replaced;
//! - **the live step** refuses the sixth. A starved request is a condition of
//!   serving, not of a program's answers: the program is switched in, admits a
//!   request and never advances it, and the mechanism fails it once the
//!   policy's own timeout passes — reported once with the request named, the
//!   previous program instantiated on the state the mechanism holds, and that
//!   program answering the next decision.
//!
//! Nothing here is a test-only branch: the scheduler, the trial, the switch and
//! the validators are the ones the epoch runs.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use engine::{
    CommandSender, GenerationConstraint, GenerationRequest, PolicyBytes, PolicyProgram,
    PolicySource, SimEngine, SimExecutor,
};
use ipc::{ImageParams, SamplingParams};
use scheduling::guest::Nanos;
use scheduling::guest::PhaseId;
use scheduling::guest::RequestId;
use scheduling::guest::decision::{Plan, Work};
use scheduling::guest::limits::Limits;
use scheduling::guest::view::{ExpertLoad, ExpertOrder};
use scheduling::guest::wire::Call;

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
    /// Answers the same plan once and an empty one afterwards.
    ///
    /// A real policy names only queued requests, so a program that admits a
    /// request and keeps naming it on later dispatches states an admission of a
    /// resident request, which §4.4 refuses — and the refusal, not the
    /// starvation, is then what the case reports.
    ///
    /// This double does not read the dispatch view: it decides from its own
    /// state, "I admitted once", which is legitimate state for a policy
    /// instance but is not the view-derived decision a deployment makes.
    /// Reading the view from here would mean parsing the request views inside
    /// hand-written WebAssembly; the follow-up is a test policy compiled the
    /// way `fcfs` is, so a fixture can decide admission the way a deployment
    /// does (noted in `notes/pending/error-taxonomy-view-reading.md`).
    PlanOnce,
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
            experts: ExpertOrder {
                placement: vec![1, 0],
            },
            observe: Observe::Answers,
            dispatch: Dispatch::Plan,
        }
    }

    fn wasm(&self) -> Vec<u8> {
        let plan = serde_json::to_string(&self.plan).expect("a plan is JSON");
        let limits = serde_json::to_string(&self.limits).expect("limits are JSON");
        let experts = serde_json::to_string(&self.experts).expect("an expert order is JSON");
        let observe = match self.observe {
            Observe::Answers => {
                "(func (export \"sched_observe\") (param i32 i32) (result i64) i64.const 0)"
                    .to_owned()
            }
            Observe::Traps => {
                "(func (export \"sched_observe\") (param i32 i32) (result i64) unreachable)"
                    .to_owned()
            }
        };
        let dispatch = match self.dispatch {
            Dispatch::Plan => format!(
                "(func (export \"{}\") (param i32 i32) (result i64) i64.const {})",
                Call::Dispatch.export(),
                packed(PLAN, plan.len())
            ),
            Dispatch::PlanOnce => format!(
                "(func (export \"{}\") (param i32 i32) (result i64) \
                 (if (global.get $admits) \
                   (then (global.set $admits (i32.const 0)) \
                         (return (i64.const {})))) \
                 (i64.const {}))",
                Call::Dispatch.export(),
                packed(PLAN, plan.len()),
                packed(VICTIMS, 2)
            ),
            Dispatch::Burns => format!(
                "(func (export \"{}\") (param i32 i32) (result i64) (loop (br 0)) i64.const 0)",
                Call::Dispatch.export()
            ),
        };
        let source = format!(
            "(module (memory (export \"memory\") 1) (global $admits (mut i32) (i32.const 1)) \
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
             (func (export \"{limits_call}\") (result i64) i64.const {limits_answer}) \
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

/// A work entry naming request 1 — the request the invalid-decision case
/// submits first — in a phase this epoch does not have.
///
/// The request must exist in the state the trial judges: a plan naming a
/// request that was never submitted is refused for that, which is a different
/// decision from the one the case is about.
fn unknown_phase() -> Plan {
    Plan {
        // The request is the scheduler's resident when the trial judges the
        // plan, so no admission is stated and the work is judged against the
        // resident view: the refusal that follows is the phase's and not the
        // request's. A plan that admitted it instead would have its work
        // judged on the admission this plan makes (§4.4), which exempts the
        // work from every check this case exists to state.
        admit: Vec::new(),
        work: vec![Work {
            request: scheduling::guest::RequestId(1),
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

/// Waits until the scheduler has taken a step past `from`.
///
/// A submission is a command, and the scheduler handles commands before it
/// steps: once a step past the submission has run, the request is in the state
/// the next decision is judged against. `from` is read with [`World::step`].
fn stepped_past(world: &World, from: u64) {
    eventually(|| {
        let progress = world.handle.progress().recv().ok()?;
        (progress.step > from).then_some(())
    });
}

/// A scheduler thread serving one program, with the one it replaced kept for a
/// revert.
struct World {
    handle: CommandSender,
    /// The scheduler's own thread. It ends when the world's command channel
    /// closes, which is when `handle` is dropped with this struct: nothing here
    /// joins it, so the field is kept for that lifetime rather than read.
    #[allow(dead_code)]
    thread: thread::JoinHandle<()>,
}

impl World {
    /// A scheduler serving `good`, committed: the program that serves on when
    /// a candidate is refused or a live failure reverts one.
    fn serving(good: Toy) -> Self {
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
        world
            .install("fcfs", 1, good)
            .expect("the good program serves");
        world.handle.commit_policy();
        world
    }

    /// Starts a scheduler serving `good`, committed, and then puts `failing` in
    /// so that a live failure has a program to serve on under.
    fn with_previous(good: Toy, failing: Toy) -> Self {
        let world = Self::serving(good);
        world
            .install("fcfs", 2, failing)
            .expect("the failing program is instantiated before it serves");
        world
    }

    /// Offers `failing` as the epoch's next program and answers the trial's
    /// reason for refusing it.
    ///
    /// §4.4's first failures are decided here rather than live: the trial asks
    /// every decision the state names on the mechanism's current state, and a
    /// program whose answers run past their fuel, declare limits above the
    /// build config's ceilings, or name something that does not exist never
    /// becomes the serving one.
    ///
    /// `observe` is not one of them: at a boundary there are no events to
    /// observe, so a program that traps there passes its trial and fails the
    /// first real step - see `a_trap_in_observe_fails_the_policy`.
    fn refuse(&self, failing: Toy) -> String {
        let program = PolicyProgram {
            backend: "fcfs".to_owned(),
            // A revision past any a case installed, so the trial is of a new
            // program rather than a name already serving.
            revision: 3,
            arguments: Default::default(),
            stated: Default::default(),
        };
        let source: Arc<dyn PolicySource + Send + Sync> =
            Arc::new(PolicyBytes::new(failing.wasm()));
        match answer(self.handle.precopy_policy(program, source)) {
            Ok(()) => panic!("the trial refuses this program, and it did not"),
            Err(reason) => reason,
        }
    }

    /// The step this scheduler reached, or zero before its first.
    fn step(&self) -> u64 {
        self.handle
            .progress()
            .recv()
            .expect("the scheduler answers its progress")
            .step
    }

    /// Asserts a program that was never served left no live failure behind.
    fn no_failure(&self) {
        assert!(
            self.handle
                .take_policy_report()
                .try_recv()
                .ok()
                .flatten()
                .is_none(),
            "a refused candidate is not a live failure: nothing was replaced"
        );
    }

    fn install(&self, backend: &str, revision: u32, toy: Toy) -> Result<(), String> {
        let program = PolicyProgram {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
            stated: Default::default(),
        };
        let source: Arc<dyn PolicySource + Send + Sync> = Arc::new(PolicyBytes::new(toy.wasm()));
        answer(self.handle.precopy_policy(program, source))?;
        answer(self.handle.switchover_policy())
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
    // `observe` is the one decision a precopy cannot ask: at a boundary there
    // are no events to observe, so a program that traps in it passes its trial
    // and fails the first real step that hands it one. That is the live
    // failure, and the program before it serves on.
    let world = World::with_previous(
        Toy::idle(),
        Toy {
            observe: Observe::Traps,
            ..Toy::idle()
        },
    );
    let mut events = world
        .handle
        .submit(text_request(1, 8, 4))
        .expect("the request is submitted");
    let report = world.failed();
    let reason = report.reason.to_lowercase();
    assert!(
        reason.contains("trap") && reason.contains("observ"),
        "the reason names the failed call: {reason}"
    );
    // The program before the trapping one answers the next decision: its
    // placement is the one handed over, and the failed program's is not.
    assert_eq!(world.serving_order().placement, vec![1, 0]);
    assert!(
        world
            .handle
            .take_policy_report()
            .recv()
            .ok()
            .flatten()
            .is_none(),
        "a policy fails once, not on every step"
    );
    // A failed policy ends no request: the one that arrived is the previous
    // program's to serve now.
    assert!(
        !events.try_recv().is_ok_and(|event| event.is_terminal()),
        "the request was not ended by the policy's failure"
    );
}

#[test]
fn exhausted_fuel_in_dispatch_fails_the_policy() {
    let world = World::serving(Toy::idle());
    let reason = world
        .refuse(Toy {
            dispatch: Dispatch::Burns,
            ..Toy::idle()
        })
        .to_lowercase();
    assert!(
        reason.contains("fuel") && reason.contains("dispatch"),
        "the reason names the spent budget and the call it was spent in: {reason}"
    );
    assert_eq!(world.serving_order().placement, vec![1, 0]);
    world.no_failure();
}

#[test]
fn limits_over_the_build_ceiling_fail_the_policy() {
    let world = World::serving(Toy::idle());
    let reason = world
        .refuse(Toy {
            limits: Limits {
                // The simulated deployment was built for eight rows at most.
                max_batch: 1 << 20,
                ..Toy::idle().limits
            },
            ..Toy::idle()
        })
        .to_lowercase();
    assert!(
        reason.contains("limit") || reason.contains("ceiling"),
        "the reason names the limits: {reason}"
    );
    assert_eq!(world.serving_order().placement, vec![1, 0]);
    world.no_failure();
}

#[test]
fn an_invalid_decision_fails_the_policy_and_applies_nothing() {
    // The serving program admits the request and gives it no work, so the
    // state the trial judges holds it as a resident and the refused plan's
    // work is judged against that view.
    let mut admitting = Toy::idle();
    admitting.plan = Plan {
        admit: vec![scheduling::guest::RequestId(1)],
        ..Plan::default()
    };
    // It admits once: naming the resident request again on the next dispatch
    // would be a refused admission, which is a different failure from the one
    // this case states.
    admitting.dispatch = Dispatch::PlanOnce;
    // A program may admit only a queued request (§4.4), and the trial of the
    // first program runs before this request exists. So the world serves an
    // idle program first, the request arrives, and the admitting program is
    // installed over it: the world this case is about is one whose serving
    // program has admitted the request and given it no work.
    let world = World::serving(Toy::idle());
    world
        .handle
        .submit(text_request(1, 8, 4))
        .expect("the request is submitted");
    world
        .install("fcfs", 2, admitting)
        .expect("the good program serves");
    stepped_past(&world, world.step());
    let reason = world
        .refuse(Toy {
            plan: unknown_phase(),
            ..Toy::idle()
        })
        .to_lowercase();
    assert!(
        reason.contains("phase") || reason.contains("not one of this epoch"),
        "the reason names the refused decision: {reason}"
    );
    // Nothing of the refused plan ran, and nothing of it serves: the program
    // that answers the next decision is the one that was already serving.
    assert_eq!(world.serving_order().placement, vec![1, 0]);
    world.no_failure();
}

#[test]
fn an_invalid_expert_order_fails_the_policy() {
    let world = World::serving(Toy::idle());
    world.handle.publish_expert_load(Some(ExpertLoad {
        ranks: 2,
        experts: 2,
        tokens: vec![1, 1],
    }));
    let reason = world
        .refuse(Toy {
            experts: ExpertOrder {
                placement: vec![0, 0],
            },
            ..Toy::idle()
        })
        .to_lowercase();
    assert!(
        reason.contains("expert"),
        "the reason names the order: {reason}"
    );
    assert_eq!(world.serving_order().placement, vec![1, 0]);
    world.no_failure();
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
        dispatch: Dispatch::PlanOnce,
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
    // The program that starves must be able to name the request, so the
    // request is queued before it is offered: its plan admits what the world
    // already holds, and then gives it nothing.
    let world = World::serving(Toy::idle());
    let mut events = world
        .handle
        .submit(text_request(1, 8, 4))
        .expect("the request is submitted");
    stepped_past(&world, world.step());
    world
        .install("fcfs", 2, starving(vec![RequestId(1)], Nanos(1_000_000)))
        .expect("the starving program is instantiated before it serves");
    let report = world.failed();
    assert!(
        report.reason.to_lowercase().contains("starv") && report.reason.contains('1'),
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
