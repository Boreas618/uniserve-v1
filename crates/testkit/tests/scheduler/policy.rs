//! Policy (SD §8.1): the scheduler's side of the policy interface (SD §5, §3.9). The policy is
//! called `observe` and `dispatch` once per step and `init` and `limits` once per instance; a
//! switch to a module that traps, runs out of fuel or declares limits above a ceiling is refused
//! and leaves the serving policy; a fault of the serving policy rolls back to the one it replaced,
//! or restarts it when none is kept, and fails the scheduler when the policy so made faults again
//! before it decides a step; a decision beyond a limit is cut and an id in the
//! wrong state skipped, neither a fault (SD §7.6); the policy is told one `Updated` per changed
//! request per step; a starting instance's snapshot holds the blocks of `cache`; and the policy's
//! `offload` is stored per group by key.
//!
//! The cases state their own policy modules as WebAssembly text ([`Script`]): modules that serve
//! a fixed decision, fault where a case puts the fault, or probe the messages the scheduler sends
//! them by trapping on what a case looks for, which the scheduler reports. The `fcfs` policy the
//! harness serves is the policy a switch replaces. Every step of every case checks the
//! scheduler's side of A1 to A4.

use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::batch::{Batch, Phase};
use engine::scheduler::{Message, Output, policy};
use proto::report::SchedulerStats;
use proto::request::{Event, FinishReason, MigrationCause, RequestId};
use scheduling::decision::{DispatchOutput, Work};
use scheduling::ids::{PageId, RequestId as PolicyRequest};
use scheduling::limits::{Limits, Parameters};
use scheduling::wire;
use storage::Key;
use storage::runtime::{Completion, GroupId, Request, Tier};
use wasmtime::Module;

use crate::harness::{A, B, BLOCK, Deployment, Harness, token};
use crate::memory::{HYBRID, commits, host_stores, keys_of};

/// Where a script module keeps its data in its memory: its limits, the needle its probes look
/// for, its dispatch replies, and from its second page on the messages the host writes into it
/// (every `sched_alloc` returns that page; a message never outlives its call).
const LIMITS_AT: u32 = 1024;
const NEEDLE_AT: u32 = 4096;
const REPLIES_AT: u32 = 8192;
const MESSAGES_AT: u32 = 65536;

/// The pages of a script module's memory: room for a message of several hundred KiB.
const PAGES: u32 = 8;

/// How an `Updated` event of the policy's wire begins (`scheduling::event::Event`).
const UPDATED: &str = r#""kind":"updated""#;

/// What a script's `init` does.
enum Init {
    /// It initializes.
    Serves,
    /// It traps.
    Traps,
    /// It runs until its fuel is exhausted.
    Spins,
    /// It traps unless the snapshot it is given contains this text.
    Names(String),
}

/// What a script's `dispatch` does.
enum Dispatch {
    /// It replies its decision.
    Replies,
    /// It traps.
    Traps,
    /// It runs until its fuel is exhausted.
    Spins,
}

/// A policy module a case states: it declares `limits`, does what `init` and `dispatch` say, and
/// replies its `n`-th `dispatch` with the `n`-th of `replies` (the last for every later call).
/// `observe` traps once a message names its needle the stated number of times. `place` names the
/// first candidate, or with `alternate` the first and the second in turn. It names no victim and
/// routes to the first rank.
struct Script {
    limits: Limits,
    init: Init,
    observe: Option<(String, u32)>,
    replies: Vec<DispatchOutput>,
    dispatch: Dispatch,
    alternate: bool,
}

impl Script {
    /// A script that admits `requests` and names a prefill and a decode of each in every step,
    /// with the deployment's chunk and steps of [`Deployment::tokens`]: chunks of 16 tokens, 8
    /// calls and 64 tokens a step. An id that names no waiting request, or a phase its request
    /// is not ready for, is the scheduler's to skip.
    fn serving(requests: &[u64]) -> Self {
        Self {
            limits: Limits {
                chunked_prefill_size: 16,
                max_batch: 8,
                max_num_batched_tokens: 64,
                ..Limits::default()
            },
            init: Init::Serves,
            observe: None,
            replies: vec![driving(requests, 0)],
            dispatch: Dispatch::Replies,
            alternate: false,
        }
    }

    /// The module, compiled by the engine the scheduler's policies are compiled by.
    fn module(&self) -> Arc<Module> {
        let mut segments = Vec::new();
        let limits = wire::encode(&self.limits).unwrap();
        let limits_reply = pack(LIMITS_AT, &limits);
        segments.push(data(LIMITS_AT, &limits));

        let needle = match (&self.init, &self.observe) {
            (Init::Names(needle), _) | (_, Some((needle, _))) => needle.as_bytes().to_vec(),
            _ => vec![0],
        };
        segments.push(data(NEEDLE_AT, &needle));

        let mut at = REPLIES_AT;
        let mut replies = Vec::new();
        for reply in &self.replies {
            let bytes = wire::encode(reply).unwrap();
            segments.push(data(at, &bytes));
            replies.push(pack(at, &bytes));
            at += bytes.len() as u32;
        }
        assert!(at < MESSAGES_AT, "the replies overlap the messages");

        let init = match &self.init {
            Init::Serves => "(i64.const 0)".to_owned(),
            Init::Traps => "(unreachable)".to_owned(),
            Init::Spins => "(loop $spin (br $spin)) (i64.const 0)".to_owned(),
            Init::Names(_) => "(if (i32.eqz (call $count (local.get $at) (local.get $len))) \
                               (then (unreachable))) (i64.const 0)"
                .to_owned(),
        };
        let observe = match &self.observe {
            Some((_, times)) => format!(
                "(if (i32.ge_u (call $count (local.get $at) (local.get $len)) (i32.const {times})) \
                 (then (unreachable))) (i64.const 0)"
            ),
            None => "(i64.const 0)".to_owned(),
        };

        // The module counts its calls; after the fault, if any, the `n`-th call returns the first
        // reply whose number the count does not exceed, and every later call the last.
        let fault = match self.dispatch {
            Dispatch::Replies => "",
            Dispatch::Traps => "(unreachable)",
            Dispatch::Spins => "(loop $spin (br $spin))",
        };
        let mut dispatch = format!(
            "(global.set $dispatched (i32.add (global.get $dispatched) (i32.const 1))) {fault}"
        );
        let (last, earlier) = replies.split_last().unwrap();
        for (call, reply) in (1..).zip(earlier) {
            dispatch.push_str(&format!(
                " (if (i32.le_u (global.get $dispatched) (i32.const {call})) \
                 (then (return (i64.const {reply}))))"
            ));
        }
        dispatch.push_str(&format!(" (i64.const {last})"));

        let place = if self.alternate {
            "(global.set $placed (i32.add (global.get $placed) (i32.const 1))) \
             (i32.rem_u (i32.sub (global.get $placed) (i32.const 1)) (i32.const 2))"
        } else {
            "(i32.const 0)"
        };

        let segments = segments.join("\n");
        let count = count(needle.len());
        let text = format!(
            r#"(module
                (memory (export "memory") {PAGES})
                {segments}
                (global $dispatched (mut i32) (i32.const 0))
                (global $placed (mut i32) (i32.const 0))
                {count}
                (func (export "sched_alloc") (param i32) (result i32) (i32.const {MESSAGES_AT}))
                (func (export "sched_free") (param i32 i32))
                (func (export "sched_init") (param $at i32) (param $len i32) (result i64) {init})
                (func (export "sched_observe") (param $at i32) (param $len i32) (result i64)
                    {observe})
                (func (export "sched_dispatch") (param i32 i32) (result i64) {dispatch})
                (func (export "sched_preempt") (param i32 i32) (result i64) (i64.const 0))
                (func (export "sched_route") (param i32 i32) (result i32) (i32.const 0))
                (func (export "sched_place") (param i32 i32) (result i32) {place})
                (func (export "sched_limits") (result i64) (i64.const {limits_reply})))"#
        );
        load(text.as_bytes())
    }
}

/// A reply buffer at `at` holding `bytes`, as a call returns it: `ptr << 32 | len`.
fn pack(at: u32, bytes: &[u8]) -> u64 {
    (u64::from(at) << 32) | bytes.len() as u64
}

/// A data segment writing `bytes` at `at`, every byte escaped.
fn data(at: u32, bytes: &[u8]) -> String {
    let escaped: String = bytes.iter().map(|byte| format!("\\{byte:02x}")).collect();
    format!(r#"(data (i32.const {at}) "{escaped}")"#)
}

/// `$count(at, len)`: how many times the needle, the `length` bytes at `NEEDLE_AT`, occurs in the
/// message of `len` bytes at `at`.
fn count(length: usize) -> String {
    format!(
        "(func $count (param $at i32) (param $len i32) (result i32)
            (local $end i32) (local $i i32) (local $found i32)
            (if (i32.lt_u (local.get $len) (i32.const {length})) (then (return (i32.const 0))))
            (local.set $end (i32.add (local.get $at) (i32.sub (local.get $len) (i32.const {length}))))
            (block $done
                (loop $scan
                    (br_if $done (i32.gt_u (local.get $at) (local.get $end)))
                    (local.set $i (i32.const 0))
                    (block $differs
                        (loop $compare
                            (br_if $differs (i32.ne
                                (i32.load8_u (i32.add (local.get $at) (local.get $i)))
                                (i32.load8_u (i32.add (i32.const {NEEDLE_AT}) (local.get $i)))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br_if $compare (i32.lt_u (local.get $i) (i32.const {length}))))
                        (local.set $found (i32.add (local.get $found) (i32.const 1))))
                    (local.set $at (i32.add (local.get $at) (i32.const 1)))
                    (br $scan)))
            (local.get $found))"
    )
}

/// A decision that admits `requests` and names a prefill chunk of `tokens` (0: the chunk limit)
/// and a decode of each.
fn driving(requests: &[u64], tokens: u32) -> DispatchOutput {
    let work = requests.iter().flat_map(|&id| {
        [(Phase::Prefill, tokens), (Phase::Decode, 0)].map(|(phase, tokens)| Work {
            request: PolicyRequest(id),
            phase: phase.id(),
            tokens,
        })
    });
    DispatchOutput {
        admit: requests.iter().copied().map(PolicyRequest).collect(),
        work: work.collect(),
        ..DispatchOutput::default()
    }
}

/// A policy module of WebAssembly text or binary, compiled by the engine policies run under.
fn load(bytes: &[u8]) -> Arc<Module> {
    Arc::new(policy::load(bytes).unwrap())
}

/// The `fcfs` policy the harness serves.
fn fcfs() -> Arc<Module> {
    load(&testkit::packaged_module())
}

/// The arguments of the `fcfs` policy on [`Deployment::tokens`]: chunks of `chunk` tokens, and
/// the deployment's 8 calls and 64 tokens a step.
fn chunked(chunk: u32) -> Parameters {
    Parameters {
        chunked_prefill_size: Some(chunk),
        max_batch: Some(8),
        max_num_batched_tokens: Some(64),
        ..Parameters::default()
    }
}

/// Sends `SetPolicy`, as the engine switches a generation's policy, and returns its reply.
fn set_policy(
    harness: &mut Harness,
    module: Arc<Module>,
    parameters: Parameters,
) -> Result<(), String> {
    let (reply, replied) = crossbeam_channel::bounded(1);
    harness.apply(Message::SetPolicy(module, parameters, reply));
    replied
        .try_recv()
        .unwrap()
        .map_err(|error| error.to_string())
}

/// Sends `RollbackPolicy` and returns its reply.
fn rollback(harness: &mut Harness) -> Result<(), String> {
    let (reply, replied) = crossbeam_channel::bounded(1);
    harness.apply(Message::RollbackPolicy(reply));
    replied
        .try_recv()
        .unwrap()
        .map_err(|error| error.to_string())
}

/// The faults the scheduler reported rolling back for (`Output::PolicyRolledBack`), in order.
fn rolled_back(harness: &Harness) -> Vec<String> {
    let outputs = harness.outputs().iter();
    outputs
        .filter_map(|output| match output {
            Output::PolicyRolledBack(fault) => Some(fault.clone()),
            _ => None,
        })
        .collect()
}

/// The tokens of call `row` of a token batch.
fn call_tokens(batch: &Batch, row: usize) -> u32 {
    let rows = &batch.columns.rows;
    let call = &rows.calls[row];
    let offset = |row: u32| rows.query_offsets[row as usize];
    (offset(call.rows.end) - offset(call.rows.start)) as u32
}

/// The prefill chunks of request `id`, in order.
fn chunks(harness: &Harness, id: RequestId) -> Vec<u32> {
    let prefills = harness.batches_of(id).into_iter();
    let prefills = prefills.filter(|batch| batch.phase == Phase::Prefill);
    prefills
        .map(|batch| {
            let row = batch.requests.iter().position(|of| *of == id).unwrap();
            call_tokens(&batch, row)
        })
        .collect()
}

/// Steps until the scheduler sent its first statistics report, which it sends once a second has
/// passed since it started, and returns it.
fn first_report(harness: &mut Harness) -> SchedulerStats {
    let started = Instant::now();
    while harness.reports().is_empty() {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "no report was sent"
        );
        std::thread::sleep(Duration::from_millis(50));
        harness.step();
    }
    harness.reports()[0].clone()
}

/// The policy's id of the block under `key`, and how its view begins in a snapshot: the leading
/// eight bytes of the key (SD §4.5).
fn page(key: Key) -> (PageId, String) {
    let id = u64::from_be_bytes(key.0[..8].try_into().unwrap());
    (PageId(id), format!(r#"{{"id":{id},"tokens":{BLOCK},"#))
}

/// The policy is called `init` and `limits` once per instance, and `observe` and `dispatch` once
/// per step (SD §5.1): the lifetime count of calls the statistics report is twice the instances
/// (the scheduler's own, instantiated on its snapshot as it is built, a switch and a rollback
/// here) and twice the steps the scheduler took (SD §7.5 "The engine's counters").
#[test]
fn observe_and_dispatch_are_called_once_per_step_and_limits_once_per_instance() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 10, 20));
    harness.run(10);
    set_policy(&mut harness, fcfs(), chunked(8)).unwrap();
    harness.run(5);
    rollback(&mut harness).unwrap();
    harness.settle(100);
    assert_eq!(harness.tokens(id).len(), 20);

    let report = first_report(&mut harness);
    assert!(report.steps > 0);
    assert_eq!(report.policy_calls, 2 * 3 + 2 * report.steps);
    assert!(report.policy_fuel > 0, "every call is metered");
}

/// A switch to a module that traps or runs out of fuel as it starts, or that declares limits above
/// the ceilings the workers' buckets state, is refused with the fault, and the serving policy
/// is untouched: nothing is kept to roll back to, and it serves on with its own limits (SD §3.9,
/// §5.2 "Limits", "Faults").
#[test]
fn a_trap_exhausted_fuel_or_limits_above_a_ceiling_refuse_a_switch_and_leave_the_serving_policy() {
    let mut harness = Deployment::tokens().chunk(8).build();
    let trapping = Script {
        init: Init::Traps,
        ..Script::serving(&[])
    };
    let spinning = Script {
        init: Init::Spins,
        ..Script::serving(&[])
    };
    let wide = Parameters {
        max_num_batched_tokens: Some(1024),
        ..chunked(8)
    };
    let refusals = [
        (
            trapping.module(),
            Parameters::default(),
            "it trapped in `init`",
        ),
        (
            spinning.module(),
            Parameters::default(),
            "it ran past its fuel in `init`",
        ),
        (
            fcfs(),
            wide,
            "its limits are outside the ceilings: max_num_batched_tokens is 1024, above the \
             ceiling of 64",
        ),
    ];
    for (module, parameters, fault) in refusals {
        let refused = set_policy(&mut harness, module, parameters);
        let Err(error) = refused else {
            panic!("a switch to a policy that faults with `{fault}` was taken");
        };
        assert!(
            error.starts_with("the scheduling policy faulted: ") && error.contains(fault),
            "{error}"
        );
    }

    assert!(
        rollback(&mut harness).is_err(),
        "no refused switch replaced the serving policy"
    );
    let id = harness.submit(token(1, 20, 2));
    harness.settle(100);
    assert_eq!(
        chunks(&harness, id),
        [8, 8, 4],
        "the serving policy's chunk"
    );
    assert_eq!(harness.tokens(id).len(), 2);
}

/// A trap or exhausted fuel of the serving policy during a step rolls back to the policy it
/// replaced, on a fresh snapshot, and is reported as `Output::PolicyRolledBack`: that policy
/// serves again, with its own limits, the running requests run on, and nothing is kept to roll
/// back to (SD §3.9, §5.2 "Faults").
#[test]
fn a_fault_of_the_serving_policy_rolls_back_to_the_one_it_replaced() {
    for (dispatch, fault) in [
        (Dispatch::Traps, "it trapped in `dispatch`"),
        (Dispatch::Spins, "it ran past its fuel in `dispatch`"),
    ] {
        let mut harness = Deployment::tokens().chunk(8).build();
        let first = harness.submit(token(1, 6, 20));
        harness.run_until(30, |harness| harness.tokens(first).len() >= 2);

        let faulting = Script {
            dispatch,
            ..Script::serving(&[1])
        };
        set_policy(&mut harness, faulting.module(), Parameters::default()).unwrap();
        harness.step();
        let rolled = rolled_back(&harness);
        assert!(
            rolled.len() == 1 && rolled[0].starts_with(fault),
            "{rolled:?}"
        );
        assert!(
            rollback(&mut harness).is_err(),
            "nothing is kept after a rollback"
        );

        let second = harness.submit(token(2, 20, 2));
        harness.settle(200);
        assert_eq!(harness.failure(), None);
        assert_eq!(
            harness.tokens(first).len(),
            20,
            "the running request runs on"
        );
        assert_eq!(
            chunks(&harness, second),
            [8, 8, 4],
            "the replaced policy's chunk"
        );
    }
}

/// A fault of the serving policy with no policy kept to roll back to restarts it: a fresh
/// instance of the serving module, on a fresh snapshot, serves on and is reported as
/// `Output::PolicyRolledBack` (SD §5.2 "Faults"). Here the policy traps on the step that tells it
/// of an arrival; the fresh instance learns of that request from its snapshot, admits it and
/// serves it to its end, and the request running before serves on.
#[test]
fn a_fault_of_the_serving_policy_with_none_kept_restarts_it_on_a_fresh_snapshot() {
    let mut harness = Deployment::tokens().build();
    let first = harness.submit(token(1, 6, 20));
    harness.run_until(30, |harness| harness.tokens(first).len() >= 2);

    let arrival_traps = Script {
        observe: Some((r#""kind":"arrived""#.to_owned(), 1)),
        ..Script::serving(&[1, 2])
    };
    set_policy(&mut harness, arrival_traps.module(), Parameters::default()).unwrap();
    harness.apply(Message::CommitPolicy);
    harness.step();
    assert!(rolled_back(&harness).is_empty());

    let second = harness.submit(token(2, 6, 4));
    harness.settle(200);
    assert_eq!(harness.failure(), None);
    let rolled = rolled_back(&harness);
    assert!(
        rolled.len() == 1 && rolled[0].starts_with("it trapped in `observe`"),
        "{rolled:?}"
    );
    assert_eq!(
        harness.tokens(first).len(),
        20,
        "the running request runs on"
    );
    assert_eq!(
        harness.tokens(second).len(),
        4,
        "the request the faulted policy was told of is served"
    );
}

/// A serving policy that a fault restarted, and that faults again before it decides a step, fails
/// the scheduler with the fault, and every request it held leaves as migratable (SD §5.2
/// "Faults", §3.7 `fail`).
#[test]
fn a_restarted_policy_that_faults_again_before_it_decides_a_step_fails_the_scheduler() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 6, 20));
    harness.run_until(30, |harness| harness.tokens(id).len() >= 2);

    let trapping = Script {
        dispatch: Dispatch::Traps,
        ..Script::serving(&[1])
    };
    set_policy(&mut harness, trapping.module(), Parameters::default()).unwrap();
    harness.apply(Message::CommitPolicy);
    harness.run_until(10, |harness| harness.failure().is_some());

    let failure = harness.failure().unwrap();
    assert!(
        failure.starts_with("the scheduling policy faulted: it trapped in `dispatch`"),
        "{failure}"
    );
    let rolled = rolled_back(&harness);
    assert!(
        rolled.len() == 1 && rolled[0].starts_with("it trapped in `dispatch`"),
        "the first fault restarts the policy: {rolled:?}"
    );
    assert!(matches!(
        harness.terminal(id),
        Some(Event::Migratable {
            cause: MigrationCause::Loss,
            ..
        })
    ));
}

/// A decision beyond the limits is cut to them, never a fault (SD §3.4 `charge`, §5.2
/// "Clamping", §7.6): a prefill chunk the policy sizes beyond its chunk limit is cut to the
/// limit, and the work beyond a step's token or call budget is left for a later step.
#[test]
fn a_decision_beyond_a_limit_is_cut_to_it() {
    const CHUNK: u32 = 16;
    const CALLS: usize = 2;
    const TOKENS: u32 = 32;
    let mut harness = Deployment::tokens().build();
    let greedy = Script {
        limits: Limits {
            chunked_prefill_size: CHUNK,
            max_batch: CALLS,
            max_num_batched_tokens: TOKENS,
            ..Limits::default()
        },
        replies: vec![driving(&[1, 2, 3], 1_000)],
        ..Script::serving(&[])
    };
    set_policy(&mut harness, greedy.module(), Parameters::default()).unwrap();
    let ids: Vec<RequestId> = (1..=3).map(|id| harness.submit(token(id, 40, 4))).collect();

    for _ in 0..300 {
        if ids.iter().all(|id| harness.terminal(*id).is_some()) {
            break;
        }
        let before = harness.batches().len();
        harness.step();
        let formed = harness.batches().split_off(before);
        let calls: usize = formed.iter().map(|batch| batch.requests.len()).sum();
        let tokens: usize = formed.iter().map(|batch| batch.columns.tokens()).sum();
        assert!(calls <= CALLS, "{calls} calls in one step");
        assert!(tokens <= TOKENS as usize, "{tokens} tokens in one step");
    }

    assert_eq!(
        chunks(&harness, ids[0]),
        [16, 16, 8],
        "a chunk of 1000 is cut to 16"
    );
    for id in &ids {
        assert!(chunks(&harness, *id).iter().all(|chunk| *chunk <= CHUNK));
        assert_eq!(harness.tokens(*id).len(), 4, "{}", harness.summary());
    }
    assert!(rolled_back(&harness).is_empty());
    assert_eq!(harness.failure(), None);
}

/// The ids a decision names are checked where they are used, and an id in the wrong state is
/// skipped, never a fault (SD §5.2 "Clamping", §7.6): an admission of a request that runs or does
/// not exist, a rejection of one that does not exist, work of a request that does not exist or
/// for a phase its request is not ready for, a refill of a request with nothing to refill, a
/// prefetch of a request that does not wait, and an offload of a block `cache` does not hold.
#[test]
fn ids_in_the_wrong_state_are_skipped() {
    let mut harness = Deployment::tokens().build();
    let mut decision = driving(&[1, 2, 99], 0);
    decision.reject = vec![PolicyRequest(98)];
    decision.work.insert(
        0,
        Work {
            request: PolicyRequest(99),
            phase: Phase::Decode.id(),
            tokens: 0,
        },
    );
    decision.work.insert(
        1,
        Work {
            request: PolicyRequest(1),
            phase: Phase::Denoise.id(),
            tokens: 0,
        },
    );
    decision.refill = vec![PolicyRequest(1), PolicyRequest(2)];
    decision.prefetch = vec![PolicyRequest(1), PolicyRequest(99)];
    decision.offload = vec![PageId(42)];
    let skipping = Script {
        replies: vec![decision],
        ..Script::serving(&[])
    };
    set_policy(&mut harness, skipping.module(), Parameters::default()).unwrap();

    let ids = [
        harness.submit(token(1, 10, 6)),
        harness.submit(token(2, 6, 6)),
    ];
    harness.settle(100);
    for id in ids {
        let Some(Event::Finished { reason, .. }) = harness.terminal(id) else {
            panic!("{:?}", harness.events(id));
        };
        assert_eq!(reason, FinishReason::MaxTokens);
        assert_eq!(harness.tokens(id).len(), 6);
    }
    for unknown in [98, 99] {
        assert!(harness.events(RequestId(unknown)).is_empty());
    }
    let skipped = harness
        .batches()
        .into_iter()
        .any(|batch| matches!(batch.phase, Phase::Refill | Phase::Denoise));
    assert!(!skipped, "no call of a phase its request is not ready for");
    assert!(rolled_back(&harness).is_empty());
    assert_eq!(harness.failure(), None);
}

/// A rollback makes the policy the last switch replaced serve again, once; a committed switch
/// keeps nothing to roll back to (SD §3.9). Which policy serves shows in its chunk.
#[test]
fn rollback_serves_the_replaced_policy_again_and_commit_drops_it() {
    let mut harness = Deployment::tokens().build();
    set_policy(&mut harness, fcfs(), chunked(8)).unwrap();
    let first = harness.submit(token(1, 20, 2));
    harness.settle(100);
    assert_eq!(
        chunks(&harness, first),
        [8, 8, 4],
        "the switch's policy serves"
    );

    rollback(&mut harness).unwrap();
    let second = harness.submit(token(2, 20, 2));
    harness.settle(100);
    assert_eq!(
        chunks(&harness, second),
        [20],
        "the replaced policy serves again"
    );
    assert!(rollback(&mut harness).is_err(), "one rollback per switch");

    set_policy(&mut harness, fcfs(), chunked(8)).unwrap();
    harness.apply(Message::CommitPolicy);
    assert!(
        rollback(&mut harness).is_err(),
        "a committed switch keeps nothing"
    );
    let third = harness.submit(token(3, 20, 2));
    harness.settle(100);
    assert_eq!(
        chunks(&harness, third),
        [8, 8, 4],
        "the committed policy serves"
    );
}

/// The policy is told a request's view in each step after it changed, once, and not while it does
/// not change (SD §5.2 "The incremental view"). A probe that traps at the first `Updated` of a
/// message serves on while a request waits, idle, for the daemon's decoder to pass its tokens, and
/// is rolled back at the step after the decoder passed them; a probe that traps when a message
/// tells a request twice serves a request from its arrival to its end.
#[test]
fn the_policy_is_told_one_update_per_changed_request_per_step() {
    let mut harness = Deployment::tokens().build();
    let mut waiting = token(1, 10, 6);
    waiting.stop_strings = vec!["end".to_owned()];
    let id = harness.submit(waiting);
    harness.run(20);
    assert_eq!(
        harness.tokens(id).len(),
        2,
        "its outputs wait for the decoder"
    );

    let once = Script {
        observe: Some((UPDATED.to_owned(), 1)),
        ..Script::serving(&[1])
    };
    set_policy(&mut harness, once.module(), Parameters::default()).unwrap();
    harness.run(10);
    assert!(
        rolled_back(&harness).is_empty(),
        "a request that did not change is not told"
    );
    harness.apply(Message::Acknowledge(id, 2));
    harness.step();
    let rolled = rolled_back(&harness);
    assert!(
        rolled.len() == 1 && rolled[0].starts_with("it trapped in `observe`"),
        "{rolled:?}"
    );
    harness.run_until(100, |harness| {
        let published = harness.tokens(id).len() as u32;
        harness.apply(Message::Acknowledge(id, published));
        harness.terminal(id).is_some()
    });
    assert_eq!(harness.tokens(id).len(), 6);

    let twice = Script {
        observe: Some((UPDATED.to_owned(), 2)),
        ..Script::serving(&[2])
    };
    set_policy(&mut harness, twice.module(), Parameters::default()).unwrap();
    let id = harness.submit(token(2, 10, 20));
    harness.settle(100);
    assert_eq!(harness.tokens(id).len(), 20);
    assert_eq!(
        rolled_back(&harness).len(),
        1,
        "one request is told once per step"
    );
}

/// A starting instance is initialized with a snapshot that holds every block of `cache`, as the
/// policy's pages under their ids (SD §5.2 `snapshot`): a probe that traps unless its snapshot
/// names a block starts for every block the scheduler committed, and not for a block it never
/// learned of.
#[test]
fn a_starting_instance_is_given_the_blocks_of_cache() {
    let mut harness = Deployment::tokens().build();
    harness.submit(token(1, 13, 1));
    harness.settle(100);
    let committed: Vec<Key> = harness
        .answers()
        .into_iter()
        .filter_map(|answer| match answer {
            (Request::Commit { key, .. }, Completion::Committed) => Some(key),
            _ => None,
        })
        .collect();
    assert_eq!(committed.len(), 3);

    for key in committed {
        let (_, view) = page(key);
        let probe = Script {
            init: Init::Names(view),
            ..Script::serving(&[])
        };
        set_policy(&mut harness, probe.module(), Parameters::default()).unwrap();
        rollback(&mut harness).unwrap();
    }
    let (_, unknown) = page(Key([7; 32]));
    let probe = Script {
        init: Init::Names(unknown),
        ..Script::serving(&[])
    };
    let refused = set_policy(&mut harness, probe.module(), Parameters::default());
    assert!(
        refused.is_err_and(|error| error.contains("it trapped in `init`")),
        "the snapshot names a block the scheduler never learned of"
    );
}

/// The blocks the policy's `offload` names are stored to the host as one `Store` per group whose
/// memory holds them, with their keys in the policy's order; a page `cache` does not hold is
/// skipped (SD §3.4 "The policy's `offload` and `prefetch`", §4.4 "The write policy"). The blocks
/// it names are ones the write policy never offers, nor the scheduler again: the blocks of a
/// sliding window of 8 tokens that left the window before the aligned boundary of a 21-token
/// prompt, 20 tokens.
#[test]
fn the_policys_offload_is_stored_per_group_by_key() {
    let mut harness = Deployment::tokens()
        .cache(&HYBRID)
        .replicas(2)
        .host(64)
        .build();
    let serving = Script {
        alternate: true,
        ..Script::serving(&[1, 2])
    };
    set_policy(&mut harness, serving.module(), Parameters::default()).unwrap();
    let first = harness.submit(token(1, 21, 1));
    let second = harness.submit(token(2, 21, 1));
    harness.settle(100);
    // The write policy offers the blocks it stores as their `Committed` arrives; the copies run
    // with the groups' next batches.
    harness.run(20);
    assert_eq!(harness.batches_of(first)[0].group, A);
    assert_eq!(harness.batches_of(second)[0].group, B);

    // The window's blocks of a request that were never offered to the host.
    let never = |id: RequestId, group: GroupId| -> Vec<Key> {
        let offered = host_stores(&harness);
        let cached = commits(&harness, harness.owner(id, group).unwrap());
        let window = keys_of(&cached, 1).into_iter();
        window.filter(|key| !offered.contains(key)).collect()
    };
    let (on_a, on_b) = (never(first, A), never(second, B));
    assert!(on_a.len() >= 2 && on_b.len() >= 2, "{on_a:?} {on_b:?}");

    // The policy names a block of B, then one of A, then another of B, then a page `cache` does
    // not hold.
    let named = [on_b[0], on_a[0], on_b[1]];
    let mut pages: Vec<PageId> = named.iter().map(|key| page(*key).0).collect();
    pages.push(PageId(42));
    let offload = Script {
        replies: vec![
            DispatchOutput {
                offload: pages,
                ..DispatchOutput::default()
            },
            DispatchOutput::default(),
        ],
        ..Script::serving(&[])
    };
    let from = harness.taken().len();
    set_policy(&mut harness, offload.module(), Parameters::default()).unwrap();
    harness.run(3);

    let stores: Vec<(GroupId, Vec<Key>)> = harness.taken()[from..]
        .iter()
        .filter_map(|request| match request {
            Request::Store {
                group,
                keys,
                tier: Tier::Host,
            } => Some((*group, keys.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(stores, [(B, vec![on_b[0], on_b[1]]), (A, vec![on_a[0]])]);
}
