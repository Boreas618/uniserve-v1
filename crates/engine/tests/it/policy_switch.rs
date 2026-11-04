//! The program-layer switch of design §7.3 through the scheduler's commands: a
//! program is precopied on a blocking thread while the serving policy keeps
//! answering, the switchover puts it in at a batch boundary, the commit keeps
//! the policy it replaced, and a revert instantiates that one again on the
//! state the scheduler holds.
//!
//! The programs are toy policy modules written as WebAssembly text and built
//! here, so the switch is exercised without a compiled policy package: each
//! answers the pipe's calls with the interface's own JSON, and the answers
//! differ per program so that a switch is observable through what the commands
//! answer.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::Receiver;
use engine::{
    CommandSender, PolicyBytes, PolicyProgram, PolicySource, SimEngine, SimExecutor, Stage,
    StageApplied,
};
use scheduling::guest::decision::Plan;
use scheduling::guest::limits::Limits;
use scheduling::guest::view::ExpertOrder;
use scheduling::guest::wire::Call;

use crate::support;

/// Where a toy keeps its reply buffers.
const MESSAGE: u32 = 4096;
const PLAN: u32 = 8192;
const LIMITS: u32 = 12288;
const VICTIMS: u32 = 16384;
const PAGES: u32 = 20480;
const EXPERTS: u32 = 24576;

/// How long a command answer may take: a precopy loads and instantiates a
/// module on its own thread.
const ANSWER: Duration = Duration::from_secs(30);

/// A toy policy program: limits that fit what the simulated workers were
/// started for, and an empty plan.
///
/// An empty plan asks for nothing, so a scheduler that serves under the toy
/// makes no decision this test has to arrange state for; what the test reads is
/// which program the scheduler serves, from the commands' answers.
fn toy() -> Vec<u8> {
    let limits = Limits {
        chunked_prefill_size: 128,
        max_batch: 1,
        max_num_batched_tokens: 128,
        prefetch_min_tokens: 256,
        ..Limits::default()
    };
    let plan = serde_json::to_string(&Plan::default()).expect("a plan is JSON");
    let limits = serde_json::to_string(&limits).expect("limits are JSON");
    let victims = "[]".to_owned();
    let pages = "[]".to_owned();
    let experts = serde_json::to_string(&ExpertOrder::default()).expect("an expert order is JSON");
    let source = format!(
        "(module (memory (export \"memory\") 1) \
         (data (i32.const {PLAN}) \"{}\") \
         (data (i32.const {LIMITS}) \"{}\") \
         (data (i32.const {VICTIMS}) \"{}\") \
         (data (i32.const {PAGES}) \"{}\") \
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
        escape(&victims),
        escape(&pages),
        escape(&experts),
        Call::Limits.export(),
        packed(LIMITS, limits.len()),
        Call::Dispatch.export(),
        packed(PLAN, plan.len()),
        Call::Reclaim.export(),
        packed(VICTIMS, victims.len()),
        Call::Warm.export(),
        packed(PAGES, pages.len()),
        Call::Experts.export(),
        packed(EXPERTS, experts.len()),
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

/// A toy module that traps in `dispatch`, so its precopy fails: the decisions
/// are what the switch trials.
fn trapping() -> Vec<u8> {
    let source = format!(
        "(module (memory (export \"memory\") 1) \
         (func (export \"sched_alloc\") (param i32) (result i32) i32.const {MESSAGE}) \
         (func (export \"sched_free\") (param i32 i32)) \
         (func (export \"sched_init\") (param i32 i32) (result i64) i64.const 0) \
         (func (export \"sched_observe\") (param i32 i32) (result i64) i64.const 0) \
         (func (export \"{}\") (result i64) i64.const {}) \
         (func (export \"{}\") (param i32 i32) (result i64) unreachable) \
         (func (export \"{}\") (param i32 i32) (result i64) unreachable) \
         (func (export \"{}\") (param i32 i32) (result i64) unreachable) \
         (func (export \"{}\") (param i32 i32) (result i64) unreachable) \
         (func (export \"{}\") (param i32 i32) (result i32) unreachable) \
         (func (export \"{}\") (param i32 i32) (result i32) unreachable))",
        Call::Limits.export(),
        packed(MESSAGE, 0),
        Call::Dispatch.export(),
        Call::Reclaim.export(),
        Call::Warm.export(),
        Call::Experts.export(),
        Call::Route.export(),
        Call::Place.export(),
    );
    wat::parse_str(&source).expect("the trapping shape is valid WebAssembly text")
}

/// A scheduler thread serving with no request.
struct World {
    handle: CommandSender,
    thread: thread::JoinHandle<()>,
}

impl World {
    fn start() -> Self {
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
        Self {
            handle: CommandSender::new(tx),
            thread,
        }
    }

    /// Precopy `program` from `wasm`, and answer the result.
    fn precopy(&self, backend: &str, revision: u32, wasm: Vec<u8>) -> Result<(), String> {
        let program = PolicyProgram {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
            stated: Default::default(),
        };
        let source: Arc<dyn PolicySource + Send + Sync> = Arc::new(PolicyBytes::new(wasm));
        answer(self.handle.precopy_policy(program, source))
    }

    /// The switchover's answer: the step it applied at.
    fn switch_over(&self) -> Result<StageApplied, String> {
        answer(self.handle.stage(Stage::Switchover))
    }

    /// The revert's answer, as the caller that does not use the step reads it.
    fn revert(&self) -> Result<(), String> {
        answer(self.handle.revert_policy())
    }

    /// The step the scheduler reached, as it reports it.
    fn step(&self) -> u64 {
        self.handle
            .progress()
            .recv_timeout(ANSWER)
            .expect("the scheduler reports its step")
            .step
    }
}

/// Blocks until a command answers, and answers its result.
fn answer<T>(receiver: Receiver<Result<T, String>>) -> Result<T, String> {
    match receiver.recv_timeout(ANSWER) {
        Ok(result) => result,
        Err(error) => Err(format!("the scheduler did not answer: {error}")),
    }
}

#[test]
fn the_program_layers_commands_reach_one_scheduler_and_name_what_they_need() {
    let world = World::start();

    // A precopy's policy module belongs to the process holding the Program
    // Server's client, which resolves it once and precopies each scheduler; the
    // command is answered with that rather than half applied.
    let precopy = ipc::ProgramCommand::Precopy {
        generation: 1,
        programs: Box::default(),
        policy: program_server::Program {
            backend: "fcfs".to_owned(),
            revision: 1,
            arguments: Default::default(),
        },
    };
    let refused = answer(world.handle.program_command(precopy))
        .expect_err("a precopy needs its source");
    assert!(
        refused.contains("Program Server"),
        "the refusal names the owner of the program: {refused}"
    );

    // A switchover that names the step its expert group agreed applies at that
    // step: this scheduler has taken none, so step zero is now.
    world.precopy("fcfs", 1, toy()).expect("the first precopy");
    answer(world.handle.program_command(ipc::ProgramCommand::Switchover {
        generation: 1,
        step: Some(0),
    }))
    .expect("a switchover at the step the scheduler is at applies");

    // One that names a step still ahead waits for it instead of applying early.
    world.precopy("priority", 2, toy()).expect("the second precopy");
    let waiting = world.handle.program_command(ipc::ProgramCommand::Switchover {
        generation: 2,
        step: Some(3),
    });
    assert!(
        waiting.recv_timeout(Duration::from_millis(200)).is_err(),
        "a switchover for a later step does not apply at this one"
    );
}

#[test]
fn a_consumers_position_is_the_connections_and_a_scheduler_refuses_it() {
    let world = World::start();
    let command = proto::request::wire::Command {
        request: 7,
        kind: Some(proto::request::wire::command::Kind::Consumed(
            proto::request::wire::Consumed { events: 3 },
        )),
    };
    let refused = world
        .handle
        .request_command(command)
        .expect_err("a consumer's position is not a scheduler's to record");
    assert!(
        refused.contains("connection"),
        "the refusal says whose bookkeeping it is: {refused}"
    );
}

#[test]
fn the_switch_passes_through_its_stages_and_a_revert_brings_the_previous_policy_back() {
    let world = World::start();

    // An epoch's first program: its precopy prepares the only instance the
    // policy has, and the switchover puts it in.
    world
        .precopy("fcfs", 1, toy())
        .expect("the first program's precopy");
    let first = world.switch_over().expect("the first switchover");
    // A stage names the boundary it applied at: the step the scheduler had
    // reached, which is zero before any policy step ran.
    assert_eq!(
        first.step,
        world.step(),
        "a switchover answers the step it applied at"
    );

    // The next program is precopied while the first serves, then switched in;
    // the first is kept for a revert.
    world
        .precopy("priority", 2, toy())
        .expect("the second program's precopy");
    let second = world.switch_over().expect("the second switchover");
    assert_eq!(
        second.step,
        world.step(),
        "a later switchover names the boundary it applied at as well"
    );

    // A commit keeps the replaced policy until the next precopy (§7.3 step 5),
    // so a revert still finds it and brings it back.
    answer(world.handle.commit_policy()).expect("the commit applies at a boundary");
    world.revert().expect("the commit kept the previous policy");

    // The revert consumed it: there is nothing older to revert to.
    assert!(
        world.revert().is_err(),
        "a second revert has no previous policy"
    );
}

#[test]
fn a_precopy_whose_decisions_trap_fails_and_the_serving_policy_stays() {
    let world = World::start();
    world
        .precopy("fcfs", 1, toy())
        .expect("the first program's precopy");
    world.switch_over().expect("the first switchover");

    let failed = world.precopy("priority", 2, trapping());
    assert!(
        failed.is_err(),
        "a policy that traps in a decision fails its precopy"
    );
    assert!(
        world.switch_over().is_err(),
        "a switchover without a prepared instance is refused, so the serving policy stays"
    );

    // The serving policy still answers: a later program's precopy and
    // switchover go through.
    world
        .precopy("priority", 3, toy())
        .expect("a later program's precopy");
    world.switch_over().expect("the later switchover");
}
