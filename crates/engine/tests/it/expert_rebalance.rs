//! §4.4's periodic row: the mechanism asks the policy for an expert placement
//! on its own interval, checks the answer, and hands it to the point that
//! applies it.
//!
//! The load is the executor's to measure, so the test states it: the routed
//! tokens per expert are published through the scheduler's handle, a step asks
//! the policy about them, and the placement it answers is what
//! `take_expert_order` hands to the world setup (§7.6). The programs are toy
//! policy modules written as WebAssembly text, as in the switch's tests, so the
//! answer is whatever the test wants it to be.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use engine::{CommandSender, PolicyBytes, PolicyProgram, PolicySource, SimEngine, SimExecutor};
use scheduling::guest::decision::Plan;
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

/// A toy policy program that answers every decision with an empty one, except
/// `experts`, which answers `order`.
fn toy(order: ExpertOrder) -> Vec<u8> {
    let limits = Limits {
        chunked_prefill_size: 128,
        max_batch: 1,
        max_num_batched_tokens: 128,
        prefetch_min_tokens: 256,
        ..Limits::default()
    };
    let plan = serde_json::to_string(&Plan::default()).expect("a plan is JSON");
    let limits = serde_json::to_string(&limits).expect("limits are JSON");
    let order = serde_json::to_string(&order).expect("an expert order is JSON");
    let source = format!(
        "(module (memory (export \"memory\") 1) \
         (data (i32.const {PLAN}) \"{}\") \
         (data (i32.const {LIMITS}) \"{}\") \
         (data (i32.const {VICTIMS}) \"[]\") \
         (data (i32.const {PAGES}) \"[]\") \
         (data (i32.const {EXPERTS}) \"{}\") \
         (func (export \"sched_alloc\") (param i32) (result i32) i32.const {MESSAGE}) \
         (func (export \"sched_free\") (param i32 i32)) \
         (func (export \"sched_init\") (param i32 i32) (result i64) i64.const 0) \
         (func (export \"sched_observe\") (param i32 i32) (result i64) i64.const 0) \
         (func (export \"{}\") (result i64) i64.const {}) \
         (func (export \"{}\") (param i32 i32) (result i64) i64.const {}) \
         (func (export \"{}\") (param i32 i32) (result i64) i64.const {}) \
         (func (export \"{}\") (param i32 i32) (result i64) i64.const {}) \
         (func (export \"{}\") (param i32 i32) (result i64) i64.const {}) \
         (func (export \"{}\") (param i32 i32) (result i32) i32.const 0) \
         (func (export \"{}\") (param i32 i32) (result i32) i32.const 0))",
        escape(&plan),
        escape(&limits),
        escape(&order),
        Call::Limits.export(),
        packed(LIMITS, limits.len()),
        Call::Dispatch.export(),
        packed(PLAN, plan.len()),
        Call::Reclaim.export(),
        packed(VICTIMS, 2),
        Call::Warm.export(),
        packed(PAGES, 2),
        Call::Experts.export(),
        packed(EXPERTS, order.len()),
        Call::Route.export(),
        Call::Place.export(),
    );
    wat::parse_str(&source).expect("the toy policy module is valid WebAssembly text")
}

/// A reply of `len` bytes at `offset`, as the call protocol packs it.
fn packed(offset: u32, len: usize) -> u64 {
    (u64::from(offset) << 32) | u64::try_from(len).expect("a reply fits u64")
}

/// `text` escaped for a WebAssembly text data segment.
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// The load the executor publishes: two experts, two ranks, a skew toward the
/// first expert.
fn load() -> ExpertLoad {
    ExpertLoad {
        ranks: 2,
        experts: 2,
        tokens: vec![30, 10],
    }
}

/// A scheduler thread serving with a policy that is already switched in.
struct World {
    handle: CommandSender,
    thread: thread::JoinHandle<()>,
}

impl World {
    fn start(order: ExpertOrder) -> Self {
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
        let program = PolicyProgram {
            backend: "fcfs".to_owned(),
            revision: 1,
            arguments: Default::default(),
            stated: Default::default(),
        };
        let source: Arc<dyn PolicySource + Send + Sync> = Arc::new(PolicyBytes::new(toy(order)));
        wait(world.handle.precopy_policy(program, source)).expect("the program's precopy");
        wait(world.handle.switchover_policy()).expect("the switchover puts it in");
        world
    }

    /// Puts another program in, keeping the one it replaces as the policy a
    /// failure reverts to.
    fn replace(&self, order: ExpertOrder) {
        self.handle.commit_policy();
        let program = PolicyProgram {
            backend: "fcfs".to_owned(),
            revision: 2,
            arguments: Default::default(),
            stated: Default::default(),
        };
        let source: Arc<dyn PolicySource + Send + Sync> = Arc::new(PolicyBytes::new(toy(order)));
        wait(self.handle.precopy_policy(program, source)).expect("the second program's precopy");
        wait(self.handle.switchover_policy()).expect("the second switchover");
    }
}

/// Blocks until a command answers, and answers its result.
fn wait(receiver: crossbeam_channel::Receiver<Result<(), String>>) -> Result<(), String> {
    match receiver.recv_timeout(ANSWER) {
        Ok(result) => result,
        Err(error) => Err(format!("the scheduler did not answer: {error}")),
    }
}

/// Polls `poll` until it answers `Some`, so the test waits on the scheduler's
/// step rather than on a sleep.
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
fn a_published_load_is_asked_about_and_the_order_reaches_the_handoff() {
    // Two experts, two ranks: placing them on opposite ranks is a legal answer
    // and the one the world setup would apply.
    let world = World::start(ExpertOrder {
        placement: vec![1, 0],
    });

    // Until the executor publishes the routed tokens there is nothing to
    // decide, so no order is waiting.
    assert!(
        world.handle.take_expert_order().is_none(),
        "no load, no expert decision"
    );

    world.handle.publish_expert_load(Some(load()));

    // The next step asks, the answer is checked, and the placement is waiting
    // at the handoff point.
    let order = eventually(|| world.handle.take_expert_order());
    assert_eq!(order.placement, vec![1, 0]);

    // Taken once: a placement is applied once.
    assert!(
        world.handle.take_expert_order().is_none(),
        "the handoff takes the placement once"
    );
}

#[test]
fn an_illegal_expert_order_fails_the_policy() {
    // Both experts on rank 0 is not a placement: two ranks would leave one
    // holding every expert while the other idles, which `validate_experts`
    // refuses (§4.4: an illegal permutation is the policy's failure).
    let world = World::start(ExpertOrder {
        placement: vec![1, 0],
    });
    // A policy that fails serves on under the one it replaced (§7.3), so the
    // illegal program is put in over a legal one.
    world.replace(ExpertOrder {
        placement: vec![0, 0],
    });

    world.handle.publish_expert_load(Some(load()));

    let report = eventually(|| world.handle.take_policy_report().recv().ok().flatten());
    assert!(
        format!("{report:?}").contains("expert"),
        "the report names the decision that failed: {report:?}"
    );
}
