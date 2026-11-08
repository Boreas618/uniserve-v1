//! A world that succeeds a running one, over the in-process pool.
//!
//! The launched path states the same events with two deployments: a successor
//! starts beside the running world under the same claim, is promoted with the
//! running world's prefix cache index once it parks, commits once the running
//! world closed, and continues a request the running one generated part of.
//! Here both worlds are handles over `support::Serving`, which completes every
//! batch in process, and the commands are the launched path's own: `describe_cache`,
//! `park`, `promote`, `commit`.
//!
//! The pool the case uses samples by cache extent (`Serving::by_extent`), so a
//! token is a function of how much of the request the mechanism states as
//! computed: the continuation samples what the uninterrupted request sampled
//! next, which is the property the launched case asserts.

use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::{
    BatchCapacity, Event, GenerationConstraint, GenerationRequest, SchedulerHandle, Stage,
};
use ipc::{ImageParams, RequestId, SamplingParams};
use worker::batch::Batches;

use crate::support::{self, Serving};

/// How long a command answer may take.
/// How long a command answer may take.
///
/// The first answer of each world is its policy's precopy, which compiles the
/// `fcfs` module inside a debug build; this case builds three worlds, and a
/// bound that does not cover the compile reports a timeout for a world that is
/// booting. Three boots measured near a hundred seconds between them, so the
/// bound is three times that: a careful margin costs nothing when the case is
/// green and costs a whole round trip when it is short. The margin is the
/// compile's, not the mechanism's: a mechanism that stopped answering still
/// fails here, only later.
const ANSWER: Duration = Duration::from_secs(300);

/// How long a park may take once the world has been asked.
///
/// A park waits for the world to hold nothing, which is a drained world rather
/// than a compiling one, so this bound is the mechanism's and not the build's.
const PARK: Duration = Duration::from_secs(30);

/// How many tokens the running world generates before it parks.
const GENERATED: usize = 3;

/// How many tokens the continuation may generate.
const CONTINUED: usize = 3;

/// A greedy text request over `length` prompt tokens starting at `first`.
fn request(id: u64, length: u32, first: u32, generated: Vec<u32>) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: (0..length).map(|index| first + index).collect(),
        negative_prompt_token_ids: Vec::new(),
        multimodal_inputs: Default::default(),
        constraint: GenerationConstraint::UndOnly,
        max_und_tokens: generated.len() + GENERATED + CONTINUED,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_images: 1,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        include_stop_token: false,
        image_generation: Default::default(),
        readout: Vec::new(),
        generated_token_ids: generated,
        adapter: 0,
        generated_images: Vec::new(),
        trajectory: Default::default(),
    }
}

/// Instantiates the epoch's first policy for `pool` and switches it in.
fn install(handle: &SchedulerHandle, pool: &Serving) {
    let rank = pool
        .inventory()
        .first()
        .and_then(|inventory| inventory.ranks.first())
        .expect("the pool states its ranks");
    let ceilings = engine::ceilings_of(BatchCapacity::of(rank));
    let program = engine::PolicyProgram {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: engine::Parameters {
            chunked_prefill_size: Some(ceilings.chunked_prefill_size),
            max_queued_requests: ceilings.max_queued_requests,
            max_batch: Some(ceilings.max_batch),
            max_num_batched_tokens: Some(ceilings.max_num_batched_tokens),
            prefetch_min_tokens: Some(ceilings.prefetch_min_tokens),
            timeout: Some(ceilings.timeout),
            ..Default::default()
        },
        stated: Default::default(),
    };
    let source: Arc<dyn engine::PolicySource + Send + Sync> =
        Arc::new(engine::PolicyBytes::new(support::policy_module()));
    answer("the precopy", handle.precopy_policy(program, source)).expect("the policy is precopied");
    answer("the switchover", handle.stage(Stage::Switchover)).expect("the policy serves");
}

/// Waits for a command's answer.
fn answer<T>(
    what: &str,
    receiver: crossbeam_channel::Receiver<Result<T, String>>,
) -> Result<T, String> {
    receiver.recv_timeout(ANSWER).unwrap_or_else(|error| {
        panic!("the scheduler did not answer {what} within {ANSWER:?}: {error}")
    })
}

/// Collects a request's terminal event, answering what ended it.
fn terminal(events: &mut engine::EventRx) -> Event {
    let deadline = Instant::now() + ANSWER;
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(event) if event.is_terminal() => return event,
            Ok(_) => {}
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    panic!("the request never ended within {ANSWER:?}");
}

/// A world that succeeds a running one takes over the running world's prefix
/// cache index and continues a request the running one generated part of: the
/// continuation samples what the uninterrupted request sampled next.
/// Ignored while the in-process arm cannot park: `poll_park` waits on seven
/// empty sets and `retiring_requests` drains only through a `BatchCommand::
/// Finish` carried by a command-only batch, which no scheduling pass produces
/// once the world's requests have migrated and nothing else runs
/// (`scheduler/execution.rs:453`). The case stays here because it is the arm's
/// requirement written down — a command-only batch produced from the queued
/// `Finish` — and it is the first thing that proves the requirement if it
/// lands. It is ignored rather than red because a known-blocked case that
/// fails every run blocks every other case's evidence.
#[ignore = "the in-process arm produces no command-only batch, so a park never completes"]
#[test]
fn a_successor_world_continues_the_requests_of_the_world_it_replaces() {
    // The reference: the same request, generated without interruption.
    let reference_pool = Arc::new(Serving::by_extent(vec![support::pool("model")]));
    let (mut reference_config, reference_limits) = support::config(&reference_pool, "worlds");
    reference_config.world = 1;
    let reference = SchedulerHandle::over_batches(
        &reference_config,
        reference_limits,
        reference_pool.clone() as Arc<dyn Batches>,
    )
    .expect("the reference world serves");
    install(&reference, &reference_pool);
    let mut reference_events = reference
        .submit(request(91, 8, 100, Vec::new()))
        .expect("the reference takes the request");
    let (expected, finished) =
        support::wait_for_tokens(&mut reference_events, GENERATED + CONTINUED, ANSWER);
    // Two facts with two owners: how many tokens the request produced is its
    // budget's, and what event ended it is the stream's. A single assertion
    // would let either answer for the other.
    assert_eq!(
        expected.len(),
        GENERATED + CONTINUED,
        "the reference generated {} of its {} tokens",
        expected.len(),
        GENERATED + CONTINUED
    );
    let ended = match finished {
        Some(event) => event,
        None => terminal(&mut reference_events),
    };
    assert!(
        matches!(
            ended,
            Event::Finished {
                reason: engine::FinishReason::MaxTokens,
                completion_tokens,
                ..
            } if completion_tokens as usize == GENERATED + CONTINUED
        ),
        "the reference ended as {ended:?} after {} tokens",
        expected.len()
    );
    reference.shutdown();

    // The running world generates the first part.
    let running_pool = Arc::new(Serving::by_extent(vec![support::pool("model")]));
    let (mut running_config, running_limits) = support::config(&running_pool, "worlds");
    running_config.world = 1;
    let running = SchedulerHandle::over_batches(
        &running_config,
        running_limits,
        running_pool.clone() as Arc<dyn Batches>,
    )
    .expect("the running world serves");
    install(&running, &running_pool);
    let mut running_events = running
        .submit(request(91, 8, 100, Vec::new()))
        .expect("the running world takes the request");
    let (first, _) = support::wait_for_tokens(&mut running_events, GENERATED, ANSWER);
    assert_eq!(
        first,
        expected[..GENERATED],
        "the running world generates the reference's first tokens"
    );

    // Its successor starts beside it.
    answer("the running world's cache", running.describe_cache())
        .expect("the running world states its cache");
    let successor_pool = Arc::new(Serving::by_extent(vec![support::pool("model")]));
    let (mut successor_config, successor_limits) = support::config(&successor_pool, "worlds");
    successor_config.world = 2;
    let successor = SchedulerHandle::over_batches(
        &successor_config,
        successor_limits,
        successor_pool.clone() as Arc<dyn Batches>,
    )
    .expect("the successor starts beside the running world");
    install(&successor, &successor_pool);

    // The running world parks, the successor is promoted with its index, and
    // the running world closes.
    // The park is waited for with a bound of its own, and a park that does not
    // answer is reported with what the world was doing: steps taken since the
    // command and whether it is still answering at all.
    //
    // `poll_park` answers only when seven sets are empty: running requests,
    // running media, retiring requests, waiting requests, waiting media,
    // batches in flight and pending submissions (`scheduler/world.rs:232`). A
    // request's `Finish` leaves inside a batch, so a close that never rides one
    // keeps the park from answering: a world still counting steps is that
    // case, and a world answering nothing has a stopped control thread. Two
    // owners, which is why the timeout names which one it met.
    let steps_before = running
        .counters()
        .recv_timeout(PARK)
        .expect("the running world answers its counters")
        .steps;
    let parked = match running.park(true).recv_timeout(PARK) {
        Ok(result) => result,
        Err(error) => {
            let alive = running
                .counters()
                .recv_timeout(PARK)
                .map(|counters| format!("{} steps, {} batches", counters.steps, counters.batches))
                .unwrap_or_else(|why| format!("no counters ({why})"));
            let cache = answer("the running world's cache", running.describe_cache())
                .map(|groups| format!("{} group(s)", groups.len()))
                .unwrap_or_else(|why| format!("no answer ({why})"));
            panic!(
                "the running world did not park within {PARK:?} ({error}): it had taken \
                 {steps_before} steps before the command, and afterwards {alive} and {cache} — \
                 a world still counting steps is parking and never meets the condition, a \
                 world that stopped answering is stuck"
            );
        }
    };
    let index = parked.expect("the running world parks");
    successor
        .promote(index, false, false)
        .expect("the successor is promoted with the parked index");
    running.shutdown();
    answer("the successor's commit", successor.commit()).expect("the successor commits");

    // The continuation samples what the uninterrupted request sampled next.
    let mut continuation = successor
        .submit(request(91, 8, 100, first.clone()))
        .expect("the successor takes the continuation");
    let (continued, _) = support::wait_for_tokens(&mut continuation, CONTINUED, ANSWER);
    assert_eq!(
        continued,
        expected[GENERATED..GENERATED + CONTINUED],
        "the continuation generates what the uninterrupted request generates after those tokens"
    );

    // The tokens alone cannot tell inheritance from trust: a pool that
    // followed `generated_token_ids` and prefilled the whole prompt again
    // would sample the same ids from the same extent. What separates them is
    // the successor's first batch, so the case reads it.
    let batches = successor_pool.submitted();
    let first = batches
        .first()
        .expect("the successor served the continuation");
    assert_eq!(
        first.shape.tokens, 1,
        "the successor decodes one token from the parked cache rather than prefilling the prompt \
         again: its first batch holds {} tokens",
        first.shape.tokens
    );
    assert_eq!(
        first.columns.rows.seq_lens.first().copied().unwrap_or(0) as usize,
        (8 + GENERATED) as usize,
        "the successor's row starts where the predecessor stopped"
    );
    successor.shutdown();
}
