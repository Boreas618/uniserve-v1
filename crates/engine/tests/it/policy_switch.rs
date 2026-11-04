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

/// The slot's programs as the scheduler reports them.
fn programs(world: &World) -> engine::PolicyPrograms {
    world
        .handle
        .policy_programs()
        .recv_timeout(ANSWER)
        .expect("the scheduler answers with its slot's programs")
}

/// The program a slot names, by the two fields that identify it on the wire.
fn named(program: Option<PolicyProgram>) -> Option<(String, u32)> {
    program.map(|program| (program.backend, program.revision))
}

#[test]
fn the_slot_names_the_program_serving_and_the_one_a_precopy_prepared() {
    let world = World::start();
    // A scheduler with no policy decides nothing, so a test world serves the
    // policy its harness installed from the start; nothing waits for the
    // switchover that would prepare another.
    let serving = programs(&world);
    assert_eq!(
        named(serving.running),
        Some(("fcfs".to_owned(), 1)),
        "the slot names the program serving"
    );
    assert!(named(serving.prepared).is_none());

    // A precopy prepares a candidate without putting it in.
    world.precopy("fcfs", 2, toy()).expect("the precopy");
    let prepared = programs(&world);
    assert_eq!(
        named(prepared.running),
        Some(("fcfs".to_owned(), 1)),
        "the precopy left the serving program alone"
    );
    assert_eq!(
        named(prepared.prepared),
        Some(("fcfs".to_owned(), 2)),
        "the slot names the program a precopy prepared"
    );

    // The switchover puts it in: it serves, and nothing waits.
    world.switch_over().expect("the switchover");
    let switched = programs(&world);
    assert_eq!(
        named(switched.running),
        Some(("fcfs".to_owned(), 2)),
        "the slot names the program the switchover put in"
    );
    assert!(named(switched.prepared).is_none());

    // The slot's counters agree with the step the scheduler reports, and the
    // policy that just took over has cost calls — its trial's, carried in with
    // the candidate (§12).
    let before = world.step();
    let counters = world
        .handle
        .counters()
        .recv_timeout(ANSWER)
        .expect("the scheduler answers with its counters");
    // The two commands read the same counter: a step the scheduler took between
    // them is the only difference, so the figures cannot disagree.
    let after = world.step();
    assert!(
        (before..=after).contains(&counters.steps),
        "the counters' {} steps are the scheduler's ({before} and {after})",
        counters.steps
    );
    assert!(
        counters.policy_calls > 0,
        "the serving policy has answered at least its trial's calls"
    );
    assert_eq!(
        (counters.kernel_launches, counters.device_ns),
        (0, 0),
        "the device's counters are the executor's, not this mechanism's"
    );
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
    // step: one still ahead waits for it, and its answer is the answer of a
    // stage that applied there.
    world.precopy("fcfs", 2, toy()).expect("the first precopy");
    let ahead = world.step() + 2;
    answer(world.handle.program_command(ipc::ProgramCommand::Switchover {
        generation: 1,
        step: Some(ahead),
    }))
    .expect("a switchover at a step ahead applies there");
    assert!(
        world.step() >= ahead,
        "the boundary it applied at is one the scheduler has reached"
    );

    // A step the scheduler has passed is not a boundary any more: it is refused
    // with the step, rather than applied at one the epoch is past.
    let refused_step = answer(world.handle.program_command(ipc::ProgramCommand::Switchover {
        generation: 1,
        step: Some(ahead - 1),
    }))
    .expect_err("a step the scheduler passed cannot be applied at");
    assert!(
        refused_step.contains("passed"),
        "the refusal says the boundary is gone: {refused_step}"
    );

    // One that names a step far ahead waits for it instead of applying early.
    world.precopy("priority", 3, toy()).expect("the second precopy");
    let waiting = world.handle.program_command(ipc::ProgramCommand::Switchover {
        generation: 2,
        step: Some(world.step() + 1000),
    });
    assert!(
        waiting.recv_timeout(Duration::from_millis(200)).is_err(),
        "a switchover for a later step does not apply at this one"
    );

    // The boundary belongs to the stage that took it: a second one is refused
    // with the waiting stage's name, so the first caller's answer is not lost.
    let refused = answer(world.handle.commit_policy())
        .expect_err("a stage waits for the boundary already claimed");
    assert!(
        refused.contains("switchover"),
        "the refusal names the stage that owns the boundary: {refused}"
    );
}

#[test]
fn a_scheduler_on_the_worker_processes_refuses_a_shadow_pass_and_names_why() {
    let world = World::start();

    // §7.3 step 2's paired pass is the in-process arm's: a worker-process
    // executor shadows a world on its own workers, so the ask is answered with
    // that rather than left waiting for a step that cannot run it.
    let refused = answer(world.handle.shadow(7))
        .expect_err("a scheduler over the worker processes runs no shadow pass");
    assert!(
        refused.contains("worker processes"),
        "the refusal says which arm shadows: {refused}"
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

    // A program precopied while the installed one serves: its precopy prepares
    // the next instance, and the switchover puts it in.
    world
        .precopy("fcfs", 3, toy())
        .expect("the first program's precopy");
    let first = world.switch_over().expect("the first switchover");
    // A stage names the boundary it applied at: a step the scheduler had
    // reached. The scheduler runs on past that boundary as soon as a policy
    // serves, so the answer is a step it has reached, not the one it is at when
    // asked — and the slot names the program the stage put in.
    assert!(
        first.step <= world.step(),
        "the boundary {} is one the scheduler reached",
        first.step
    );
    assert_eq!(
        named(programs(&world).running),
        Some(("fcfs".to_owned(), 3)),
        "the switchover's program serves"
    );

    // The next program is precopied while the first serves, then switched in;
    // the first is kept for a revert.
    world
        .precopy("priority", 2, toy())
        .expect("the second program's precopy");
    let second = world.switch_over().expect("the second switchover");
    assert!(
        second.step >= first.step,
        "a later switchover names a boundary no earlier than the first's: {} then {}",
        first.step,
        second.step
    );
    assert_eq!(
        named(programs(&world).running),
        Some(("priority".to_owned(), 2)),
        "the second switchover's program serves"
    );

    // A commit keeps the replaced policy until the next precopy (§7.3 step 5),
    // so a revert still finds it and brings it back.
    answer(world.handle.commit_policy()).expect("the commit applies at a boundary");
    world.revert().expect("the commit kept the previous policy");
    assert_eq!(
        named(programs(&world).running),
        Some(("fcfs".to_owned(), 3)),
        "the revert brought the replaced program back"
    );

    // The revert consumed it: there is nothing older to revert to.
    assert!(
        world.revert().is_err(),
        "a second revert has no previous policy"
    );
}

#[test]
fn a_purge_drops_the_policy_a_revert_would_return_to() {
    let world = World::start();
    world
        .precopy("fcfs", 1, toy())
        .expect("the first program's precopy");
    world.switch_over().expect("the first switchover");
    world
        .precopy("priority", 2, toy())
        .expect("the second program's precopy");
    world.switch_over().expect("the second switchover");

    // §7.3 step 6: the generation before the serving one is released once no
    // batch uses it, so the policy a revert would return to is gone.
    answer(world.handle.purge_policy()).expect("the purge applies at a boundary");
    assert!(
        world.revert().is_err(),
        "a purged generation cannot be reverted to"
    );
}

#[test]
fn the_engines_commands_reach_a_scheduler_and_name_what_is_not_its_own() {
    let world = World::start();
    world.precopy("fcfs", 1, toy()).expect("the precopy");

    // A stage of §7.3 is a scheduler's: it applies at the next boundary.
    let switchover = proto::process::Command {
        kind: Some(proto::process::command::Kind::Switchover(
            proto::process::Switchover { generation: 1 },
        )),
    };
    answer(world.handle.engine_command(switchover))
        .expect("a switchover applies at the scheduler's next boundary");

    // A precopy's module comes from the Program Server, whose client the
    // process holding it resolves: the command is answered with that.
    let precopy = proto::process::Command {
        kind: Some(proto::process::command::Kind::Precopy(
            proto::process::Precopy {
                kernels: Vec::new(),
                policy: None,
            },
        )),
    };
    let refused = answer(world.handle.engine_command(precopy))
        .expect_err("a precopy is not one scheduler's to prepare");
    assert!(
        refused.contains("Program Server"),
        "the refusal names the owner of the program: {refused}"
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
