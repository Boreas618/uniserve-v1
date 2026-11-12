//! An epoch that succeeds a running one, over the in-process pool.
//!
//! The launched path states the same events with two deployments: a successor
//! starts beside the running epoch under the same lease, is promoted with the
//! running epoch's prefix cache index once it parks, commits once the running
//! epoch closed, and continues a request the running one generated part of.
//! Here both epochs are handles over `testkit::FakeBatches`, which completes every
//! batch in process, and the commands are the launched path's own: `describe_cache`,
//! `park`, `promote`, `commit`.
//!
//! The pool the case uses samples by cache extent (`testkit::Tokens::Extent`), so a
//! token is a function of how much of the request the mechanism states as
//! computed: the continuation samples what the uninterrupted request sampled
//! next, which is the property the launched case asserts.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ipc::execution::Batches;
use ipc::{RequestId, SamplingParams};
use scheduler::{BatchCapacity, Event, GenerationRequest, SchedulerHandle, Stage};

use crate::support;
use testkit::FakeBatches;

/// How long a command response may take.
/// How long a command response may take.
///
/// The first response of each epoch is its policy's precopy, which compiles the
/// `fcfs` module inside a debug build; this case builds three epochs, and a
/// bound that does not cover the compile reports a timeout for an epoch that is
/// booting. Three boots measured near a hundred seconds between them, so the
/// bound is three times that: a careful margin costs nothing when the case is
/// green and costs a whole round trip when it is short. The margin is the
/// compile's, not the mechanism's: a mechanism that stopped responding still
/// fails here, only later.
const RESPONSE: Duration = Duration::from_secs(300);

/// How long a park may take once the epoch has been asked.
///
/// A park waits for the epoch to hold nothing, which is a drained epoch rather
/// than a compiling one, so this bound is the mechanism's and not the build's.
const PARK: Duration = Duration::from_secs(30);

/// How many tokens the running epoch generates before it parks.
const GENERATED: usize = 3;

/// How many tokens the continuation may generate.
const CONTINUED: usize = 3;

/// A greedy text request over `length` prompt tokens starting at `first`.
fn request(id: u64, length: u32, first: u32, generated: Vec<u32>) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: (0..length).map(|index| first + index).collect(),
        multimodal_inputs: Default::default(),
        max_und_tokens: generated.len() + GENERATED + CONTINUED,
        sampling: SamplingParams::default(),
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        include_stop_token: false,
        generated_token_ids: generated,
        adapter: 0,
    }
}

/// Instantiates the epoch's first policy for `pool` and switches it in.
fn install(handle: &SchedulerHandle, pool: &FakeBatches) {
    let rank = pool
        .inventory()
        .first()
        .and_then(|inventory| inventory.ranks.first())
        .expect("the pool states its ranks");
    let ceilings = scheduler::ceilings_of(BatchCapacity::of(rank));
    let package = scheduler::PolicyPackage {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: scheduler::Parameters {
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
    let source: Arc<dyn scheduler::PolicySource + Send + Sync> =
        Arc::new(scheduler::PolicyBytes::new(support::policy_module()));
    response("the precopy", handle.prepare_policy(package, source))
        .expect("the policy is precopied");
    response("the switchover", handle.stage(Stage::Switch)).expect("the policy serves");
}

/// Waits for a command's response.
fn response<T>(
    what: &str,
    receiver: crossbeam_channel::Receiver<Result<T, String>>,
) -> Result<T, String> {
    receiver.recv_timeout(RESPONSE).unwrap_or_else(|error| {
        panic!("the scheduler did not respond {what} within {RESPONSE:?}: {error}")
    })
}

/// Collects a request's terminal event, responding to what ended it.
fn terminal(events: &mut scheduler::EventRx) -> Event {
    let deadline = Instant::now() + RESPONSE;
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(event) if event.is_terminal() => return event,
            Ok(_) => {}
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    panic!("the request never ended within {RESPONSE:?}");
}

/// An epoch that succeeds a running one parks it, is promoted with its prefix
/// cache index, commits once it closed, and continues a request the running one
/// generated part of: the continuation samples what the uninterrupted request
/// sampled next.
#[test]
fn a_successor_epoch_continues_the_requests_of_the_epoch_it_replaces() {
    // The reference: the same request, generated without interruption.
    let reference_pool = Arc::new(support::by_extent(vec![support::pool("model")]));
    let (mut reference_config, reference_limits) = support::config(&reference_pool, "epochs");
    reference_config.epoch = 1;
    let reference = SchedulerHandle::over_batches(
        &reference_config,
        reference_limits,
        Arc::clone(&reference_pool) as Arc<dyn Batches>,
        scheduler::CommandWaker::noop(),
    )
    .expect("the reference epoch serves");
    install(&reference, &reference_pool);
    let mut reference_events = reference
        .submit(request(91, 8, 100, Vec::new()))
        .expect("the reference takes the request");
    let (expected, finished) =
        support::wait_for_tokens(&mut reference_events, GENERATED + CONTINUED, RESPONSE);
    // Two facts with two owners: how many tokens the request produced is its
    // budget's, and what event ended it is the stream's. A single assertion
    // would let either response for the other.
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
                reason: scheduler::FinishReason::MaxTokens,
                completion_tokens,
                ..
            } if completion_tokens == GENERATED + CONTINUED
        ),
        "the reference ended as {ended:?} after {} tokens",
        expected.len()
    );
    reference.shutdown();

    // The running epoch generates the first part.
    let running_pool = Arc::new(support::by_extent(vec![support::pool("model")]));
    let (mut running_config, running_limits) = support::config(&running_pool, "epochs");
    running_config.epoch = 1;
    let running = SchedulerHandle::over_batches(
        &running_config,
        running_limits,
        Arc::clone(&running_pool) as Arc<dyn Batches>,
        scheduler::CommandWaker::noop(),
    )
    .expect("the running epoch serves");
    install(&running, &running_pool);
    let mut running_events = running
        .submit(request(91, 8, 100, Vec::new()))
        .expect("the running epoch takes the request");
    let (first, _) = support::wait_for_tokens(&mut running_events, GENERATED, RESPONSE);
    assert_eq!(
        first,
        expected[..GENERATED],
        "the running epoch generates the reference's first tokens"
    );

    // Its successor starts beside it.
    response("the running epoch's cache", running.describe_cache())
        .expect("the running epoch states its cache");
    let successor_pool = Arc::new(support::by_extent(vec![support::pool("model")]));
    let (mut successor_config, successor_limits) = support::config(&successor_pool, "epochs");
    successor_config.epoch = 2;
    let successor = SchedulerHandle::over_batches(
        &successor_config,
        successor_limits,
        Arc::clone(&successor_pool) as Arc<dyn Batches>,
        scheduler::CommandWaker::noop(),
    )
    .expect("the successor starts beside the running epoch");
    install(&successor, &successor_pool);

    // The running epoch parks, the successor is promoted with its index, and
    // the running epoch closes.
    // The park is waited for with a bound of its own, and a park that does not
    // response is reported with what the epoch was doing: steps taken since the
    // command and whether it is still responding at all.
    //
    // `poll_park` responds only when seven sets are empty: running requests,
    // running media, retiring requests, waiting requests, waiting media,
    // batches in flight and pending submissions (`scheduler/epoch.rs:232`). A
    // request's `Finish` leaves inside a batch, so a close that never rides one
    // keeps the park from responding: an epoch still counting steps is that
    // case, and an epoch responding nothing has a stopped control thread. Two
    // owners, which is why the timeout names which one it met.
    let steps_before = running
        .counters()
        .recv_timeout(PARK)
        .expect("the running epoch responds to its counters")
        .steps;
    let parked = match running.park(true).recv_timeout(PARK) {
        Ok(result) => result,
        Err(error) => {
            let alive = running
                .counters()
                .recv_timeout(PARK)
                .map(|counters| format!("{} steps, {} batches", counters.steps, counters.batches))
                .unwrap_or_else(|why| format!("no counters ({why})"));
            let cache = response("the running epoch's cache", running.describe_cache())
                .map(|groups| format!("{} group(s)", groups.len()))
                .unwrap_or_else(|why| format!("no response ({why})"));
            panic!(
                "the running epoch did not park within {PARK:?} ({error}): it had taken \
                 {steps_before} steps before the command, and afterwards {alive} and {cache} — \
                 an epoch still counting steps is parking and never meets the condition, a \
                 epoch that stopped responding is stuck"
            );
        }
    };
    let index = parked.expect("the running epoch parks");
    successor
        .promote(index, false, false)
        .expect("the successor is promoted with the parked index");
    running.shutdown();
    response("the successor's commit", successor.commit()).expect("the successor commits");

    // The continuation samples what the uninterrupted request sampled next.
    let mut continuation = successor
        .submit(request(91, 8, 100, first.clone()))
        .expect("the successor takes the continuation");
    let (continued, _) = support::wait_for_tokens(&mut continuation, CONTINUED, RESPONSE);
    assert_eq!(
        continued,
        expected[GENERATED..GENERATED + CONTINUED],
        "the continuation generates what the uninterrupted request generates after those tokens"
    );

    successor.shutdown();
}
