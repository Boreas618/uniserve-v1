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
use scheduler::{
    CommandSender, PolicyBytes, PolicyProgram, PolicySource, SchedulerConfig, Stage, StageApplied,
};
use scheduling::guest::limits::Limits;
use scheduling::guest::wire::Call;

use crate::support;
use crate::support::toy::{Toy, toy_module};

/// How long a command answer may take: a precopy loads and instantiates a
/// module on its own thread.
const ANSWER: Duration = Duration::from_secs(30);

/// A toy policy program: limits that fit what the fake's pool was started
/// for, and an empty output.
///
/// An empty output asks for nothing, so a scheduler that serves under the toy
/// makes no decision this test has to arrange state for; what the test reads is
/// which program the scheduler serves, from the commands' answers.
fn toy() -> Vec<u8> {
    toy_module(&Toy {
        limits: fitting(),
        ..Toy::default()
    })
}

/// Limits within the ceilings of the pool these cases build.
fn fitting() -> Limits {
    Limits {
        chunked_prefill_size: 128,
        max_batch: 1,
        max_num_batched_tokens: 128,
        prefetch_min_tokens: 256,
        ..Limits::default()
    }
}

/// A toy module that traps in `dispatch`, so its precopy fails: the decisions
/// are what the switch trials.
fn trapping() -> Vec<u8> {
    toy_module(&Toy {
        limits: fitting(),
        trap: Some(Call::Dispatch),
        ..Toy::default()
    })
}

/// A scheduler thread serving with no request.
struct Epoch {
    handle: CommandSender,
}

impl Epoch {
    fn start() -> Self {
        let pool = Arc::new(testkit::FakeBatches::new(vec![support::fake::text_pool(
            "model", 8,
        )]));
        let scheduler = support::fake::scheduler_over(
            pool,
            SchedulerConfig {
                max_batch: proto::config::positive(8),
                ..Default::default()
            },
            Default::default(),
        )
        .expect("the scheduler is built over the fake");
        let (tx, rx) = crossbeam_channel::unbounded();
        // The thread detaches when its handle is dropped, which is what this
        // fixture wants: the scheduler ends when `handle` closes the channel.
        thread::spawn(move || {
            let _ = scheduler.run(rx);
        });
        Self {
            handle: CommandSender::new(tx),
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
fn programs(epoch: &Epoch) -> scheduler::PolicyPrograms {
    epoch
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
    let epoch = Epoch::start();
    // A scheduler with no policy decides nothing, so a test epoch serves the
    // policy its harness installed from the start; nothing waits for the
    // switchover that would prepare another.
    let serving = programs(&epoch);
    assert_eq!(
        named(serving.running),
        Some(("fcfs".to_owned(), 1)),
        "the slot names the program serving"
    );
    assert!(named(serving.prepared).is_none());

    // A precopy prepares a candidate without putting it in.
    epoch.precopy("fcfs", 2, toy()).expect("the precopy");
    let prepared = programs(&epoch);
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
    epoch.switch_over().expect("the switchover");
    let switched = programs(&epoch);
    assert_eq!(
        named(switched.running),
        Some(("fcfs".to_owned(), 2)),
        "the slot names the program the switchover put in"
    );
    assert!(named(switched.prepared).is_none());

    // The slot's counters agree with the step the scheduler reports, and the
    // policy that just took over has cost calls — its trial's, carried in with
    // the candidate (§12).
    let before = epoch.step();
    let counters = epoch
        .handle
        .counters()
        .recv_timeout(ANSWER)
        .expect("the scheduler answers with its counters");
    // The two commands read the same counter: a step the scheduler took between
    // them is the only difference, so the figures cannot disagree.
    let after = epoch.step();
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
        "the device's counters are the worker's, not this mechanism's"
    );
}

#[test]
fn the_program_layers_commands_reach_one_scheduler_and_name_what_they_need() {
    let epoch = Epoch::start();

    // A precopy's policy module belongs to the process holding the Program
    // Server's client, which resolves it once and precopies each scheduler; the
    // command is answered with that rather than half applied.
    let precopy = ipc::ProgramCommand::Precopy {
        generation: 1,
        programs: Box::default(),
        policy: proto::program::Program {
            backend: "fcfs".to_owned(),
            revision: 1,
            arguments: Default::default(),
        },
    };
    let refused =
        answer(epoch.handle.program_command(precopy)).expect_err("a precopy needs its source");
    assert!(
        refused.contains("Program Server"),
        "the refusal names the owner of the program: {refused}"
    );

    // A switchover that names the step its expert group agreed applies at that
    // step: one still ahead waits for it, and its answer is the answer of a
    // stage that applied there.
    epoch.precopy("fcfs", 2, toy()).expect("the first precopy");
    let ahead = epoch.step() + 2;
    answer(
        epoch
            .handle
            .program_command(ipc::ProgramCommand::Switchover {
                generation: 1,
                step: Some(ahead),
            }),
    )
    .expect("a switchover at a step ahead applies there");
    assert!(
        epoch.step() >= ahead,
        "the boundary it applied at is one the scheduler has reached"
    );

    // A step the scheduler has passed is not a boundary any more: it is refused
    // with the step, rather than applied at one the epoch is past. A candidate is
    // prepared first, because the switchover above used the one it applied: with
    // nothing prepared the answer would be that, and the boundary's own refusal
    // would never be reached.
    epoch
        .precopy("fcfs", 3, toy())
        .expect("a candidate the boundary check can leave prepared");
    let refused_step = answer(
        epoch
            .handle
            .program_command(ipc::ProgramCommand::Switchover {
                generation: 1,
                step: Some(ahead - 1),
            }),
    )
    .expect_err("a step the scheduler passed cannot be applied at");
    assert!(
        refused_step.contains("passed"),
        "the refusal says the boundary is gone: {refused_step}"
    );

    // One that names a step far ahead waits for it instead of applying early.
    epoch
        .precopy("priority", 4, toy())
        .expect("the second precopy");
    let waiting = epoch
        .handle
        .program_command(ipc::ProgramCommand::Switchover {
            generation: 2,
            step: Some(epoch.step() + 1000),
        });
    assert!(
        waiting.recv_timeout(Duration::from_millis(200)).is_err(),
        "a switchover for a later step does not apply at this one"
    );

    // The boundary belongs to the stage that took it: a second one is refused
    // with the waiting stage's name, so the first caller's answer is not lost.
    let refused = answer(epoch.handle.commit_policy())
        .expect_err("a stage waits for the boundary already claimed");
    assert!(
        refused.contains("switchover"),
        "the refusal names the stage that owns the boundary: {refused}"
    );
}

#[test]
fn a_consumers_position_is_the_connections_and_a_scheduler_refuses_it() {
    let epoch = Epoch::start();
    let command = proto::request::wire::Command {
        request: 7,
        kind: Some(proto::request::wire::command::Kind::Consumed(
            proto::request::wire::Consumed { events: 3 },
        )),
    };
    let refused = epoch
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
    let epoch = Epoch::start();

    // A program precopied while the installed one serves: its precopy prepares
    // the next instance, and the switchover puts it in.
    epoch
        .precopy("fcfs", 3, toy())
        .expect("the first program's precopy");
    let first = epoch.switch_over().expect("the first switchover");
    // A stage names the boundary it applied at: a step the scheduler had
    // reached. The scheduler runs on past that boundary as soon as a policy
    // serves, so the answer is a step it has reached, not the one it is at when
    // asked — and the slot names the program the stage put in.
    assert!(
        first.step <= epoch.step(),
        "the boundary {} is one the scheduler reached",
        first.step
    );
    assert_eq!(
        named(programs(&epoch).running),
        Some(("fcfs".to_owned(), 3)),
        "the switchover's program serves"
    );

    // The next program is precopied while the first serves, then switched in;
    // the first is kept for a revert.
    epoch
        .precopy("priority", 2, toy())
        .expect("the second program's precopy");
    let second = epoch.switch_over().expect("the second switchover");
    assert!(
        second.step >= first.step,
        "a later switchover names a boundary no earlier than the first's: {} then {}",
        first.step,
        second.step
    );
    assert_eq!(
        named(programs(&epoch).running),
        Some(("priority".to_owned(), 2)),
        "the second switchover's program serves"
    );

    // A commit keeps the replaced policy until the next precopy (§7.3 step 5),
    // so a revert still finds it and brings it back.
    answer(epoch.handle.commit_policy()).expect("the commit applies at a boundary");
    epoch.revert().expect("the commit kept the previous policy");
    assert_eq!(
        named(programs(&epoch).running),
        Some(("fcfs".to_owned(), 3)),
        "the revert brought the replaced program back"
    );

    // The revert consumed it: there is nothing older to revert to.
    assert!(
        epoch.revert().is_err(),
        "a second revert has no previous policy"
    );
}

#[test]
fn a_purge_drops_the_policy_a_revert_would_return_to() {
    let epoch = Epoch::start();
    epoch
        .precopy("fcfs", 1, toy())
        .expect("the first program's precopy");
    epoch.switch_over().expect("the first switchover");
    epoch
        .precopy("priority", 2, toy())
        .expect("the second program's precopy");
    epoch.switch_over().expect("the second switchover");

    // §7.3 step 6: the generation before the serving one is released once no
    // batch uses it, so the policy a revert would return to is gone.
    answer(epoch.handle.purge_policy()).expect("the purge applies at a boundary");
    assert!(
        epoch.revert().is_err(),
        "a purged generation cannot be reverted to"
    );
}

#[test]
fn the_engines_commands_reach_a_scheduler_and_name_what_is_not_its_own() {
    let epoch = Epoch::start();
    epoch.precopy("fcfs", 1, toy()).expect("the precopy");

    // A stage of §7.3 is a scheduler's: it applies at the next boundary.
    let switchover = proto::process::Command {
        kind: Some(proto::process::command::Kind::Switchover(
            proto::process::Switchover { generation: 1 },
        )),
    };
    answer(epoch.handle.engine_command(switchover))
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
    let refused = answer(epoch.handle.engine_command(precopy))
        .expect_err("a precopy is not one scheduler's to prepare");
    assert!(
        refused.contains("Program Server"),
        "the refusal names the owner of the program: {refused}"
    );
}

#[test]
fn a_precopy_whose_decisions_trap_fails_and_the_serving_policy_stays() {
    let epoch = Epoch::start();
    epoch
        .precopy("fcfs", 1, toy())
        .expect("the first program's precopy");
    epoch.switch_over().expect("the first switchover");

    let failed = epoch.precopy("priority", 2, trapping());
    assert!(
        failed.is_err(),
        "a policy that traps in a decision fails its precopy"
    );
    assert!(
        epoch.switch_over().is_err(),
        "a switchover without a prepared instance is refused, so the serving policy stays"
    );

    // The serving policy still answers: a later program's precopy and
    // switchover go through.
    epoch
        .precopy("priority", 3, toy())
        .expect("a later program's precopy");
    epoch.switch_over().expect("the later switchover");
}

/// A candidate whose decision never returns runs out of the fuel a call is
/// given: its precopy fails, and the serving policy stays.
#[test]
fn a_candidate_that_runs_past_its_fuel_fails_the_precopy() {
    let epoch = Epoch::start();
    let spinning = toy_module(&Toy {
        limits: fitting(),
        spin: Some(Call::Dispatch),
        ..Toy::default()
    });
    let failed = epoch
        .precopy("priority", 2, spinning)
        .expect_err("a decision that never returns fails the precopy");
    assert!(
        failed.contains("fuel"),
        "the refusal names the fuel: {failed}"
    );
    assert!(
        epoch.switch_over().is_err(),
        "nothing was prepared, so the serving policy stays"
    );
}

/// A candidate whose limits exceed what the deployment was built for is
/// refused at its precopy.
#[test]
fn a_candidate_whose_limits_exceed_the_build_config_fails_the_precopy() {
    let epoch = Epoch::start();
    let too_large = toy_module(&Toy {
        limits: Limits {
            max_num_batched_tokens: u32::MAX,
            ..fitting()
        },
        ..Toy::default()
    });
    let failed = epoch
        .precopy("priority", 2, too_large)
        .expect_err("limits above the build config fail the precopy");
    assert!(
        failed.contains("max_num_batched_tokens"),
        "the refusal names the limit: {failed}"
    );
}

/// A candidate whose decision names a request the scheduler does not hold is
/// refused at its precopy: every decision is checked before a policy serves.
#[test]
fn a_candidate_whose_decision_is_invalid_fails_the_precopy() {
    let epoch = Epoch::start();
    let admits_nobody = toy_module(&Toy {
        limits: fitting(),
        output: scheduling::guest::decision::DispatchOutput {
            admit: vec![scheduling::guest::ids::RequestId(999)],
            ..Default::default()
        },
        ..Toy::default()
    });
    let failed = epoch
        .precopy("priority", 2, admits_nobody)
        .expect_err("an admission of a request nobody submitted fails the precopy");
    assert!(
        failed.contains("999"),
        "the refusal names the request: {failed}"
    );
}

/// The control thread's router asks the policy for a new request's rank,
/// keeps the previous program when the serving one names a rank it was not
/// offered, and asks no policy on a node of one scheduler.
#[test]
fn the_router_routes_switches_reverts_and_refuses_a_rank_it_was_not_offered() {
    use scheduler::{Arrival, RankView, Router};
    use scheduling::guest::time::Nanos;
    use scheduling::guest::view::RequestClass;

    let ranks: Vec<RankView> = (0..2)
        .map(|index| RankView {
            index,
            in_flight: 0,
            alive: true,
        })
        .collect();
    let arrival = Arrival {
        id: ipc::RequestId(7),
        priority: 0,
        class: RequestClass::Token,
        prompt_tokens: 8,
        generated_tokens: 0,
        remaining_tokens: 8,
        migratable: false,
        cache_rank: None,
        arrived_at: Nanos(0),
    };
    let ceilings = Limits {
        max_batch: 8,
        max_num_batched_tokens: 1024,
        chunked_prefill_size: 512,
        ..Limits::default()
    };
    let program = |backend: &str, revision: u32| PolicyProgram {
        backend: backend.to_owned(),
        revision,
        arguments: Default::default(),
        stated: Default::default(),
    };
    let routing = |route: u32| {
        PolicyBytes::new(toy_module(&Toy {
            limits: fitting(),
            route,
            ..Toy::default()
        }))
    };

    let mut router = Router::new(ceilings);
    assert!(
        router.route(&arrival, &ranks).is_err(),
        "no policy routes yet"
    );
    assert!(router.switchover(&ranks).is_err(), "nothing was prepared");

    let prepared = Router::prepare(&routing(1), program("a", 1), &ranks, &ceilings)
        .expect("the candidate prepares");
    router.adopt(prepared);
    router
        .switchover(&ranks)
        .expect("the switchover installs it");
    assert_eq!(router.route(&arrival, &ranks).expect("a rank"), 1);

    // A program that names a rank it was not offered fails, and the
    // program before it routes the request and those after it.
    let prepared = Router::prepare(&routing(5), program("b", 2), &ranks, &ceilings)
        .expect("the candidate prepares");
    router.adopt(prepared);
    router
        .switchover(&ranks)
        .expect("the switchover installs it");
    assert_eq!(
        router
            .route(&arrival, &ranks)
            .expect("the previous program routes"),
        1
    );
    assert_eq!(router.program().map(|p| p.backend.as_str()), Some("a"));
    assert_eq!(router.route(&arrival, &ranks).expect("a rank"), 1);

    // A node of one scheduler routes every request there and asks no policy.
    assert_eq!(router.route(&arrival, &ranks[..1]).expect("rank 0"), 0);
}
