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
    BatchCapacity, EngineConfig, Event, GenerationConstraint, GenerationRequest, SchedulerHandle,
    Stage,
};
use executor::batch::Batches;
use ipc::{ImageParams, RequestId, SamplingParams};

use crate::support::{self, Serving};

/// How long a command answer may take.
const ANSWER: Duration = Duration::from_secs(60);

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
        max_und_tokens: generated.len() + CONTINUED,
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
    answer(handle.precopy_policy(program, source)).expect("the policy is precopied");
    answer(handle.stage(Stage::Switchover)).expect("the policy serves");
}

/// Waits for a command's answer.
fn answer<T>(receiver: crossbeam_channel::Receiver<Result<T, String>>) -> Result<T, String> {
    receiver
        .recv_timeout(ANSWER)
        .expect("the scheduler answers within the bound")
}

/// Collects a request's tokens until its first `want` and its terminal event.
fn tokens(receiver: &mut engine::EventRx, want: usize) -> (Vec<u32>, Option<Event>) {
    let mut tokens = Vec::new();
    let deadline = Instant::now() + ANSWER;
    while Instant::now() < deadline {
        match receiver.try_recv() {
            Ok(Event::TextToken { id, .. }) => {
                tokens.push(id);
                if tokens.len() >= want {
                    return (tokens, None);
                }
            }
            Ok(event) if event.is_terminal() => return (tokens, Some(event)),
            Ok(_) => {}
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    panic!("the request did not reach {want} tokens within the bound");
}

/// A world that succeeds a running one takes over the running world's prefix
/// cache index and continues a request the running one generated part of: the
/// continuation samples what the uninterrupted request sampled next.
#[test]
fn a_successor_world_continues_the_requests_of_the_world_it_replaces() {
    // The reference: the same request, generated without interruption.
    let reference_pool = Arc::new(Serving::by_extent(vec![support::pool("model")]));
    let mut reference_config = EngineConfig::sim("worlds");
    reference_config.world = 1;
    let reference = SchedulerHandle::over_batches(
        reference_config,
        reference_pool.clone() as Arc<dyn Batches>,
    )
    .expect("the reference world serves");
    install(&reference, &reference_pool);
    let mut reference_events = reference
        .submit(request(91, 8, 100, Vec::new()))
        .expect("the reference takes the request");
    let (expected, finished) = tokens(
        &mut reference_events,
        GENERATED + CONTINUED,
    );
    assert!(
        finished.is_none(),
        "the reference generated every token asked for"
    );
    reference.shutdown();

    // The running world generates the first part.
    let running_pool = Arc::new(Serving::by_extent(vec![support::pool("model")]));
    let mut running_config = EngineConfig::sim("worlds");
    running_config.world = 1;
    let running = SchedulerHandle::over_batches(
        running_config,
        running_pool.clone() as Arc<dyn Batches>,
    )
    .expect("the running world serves");
    install(&running, &running_pool);
    let mut running_events = running
        .submit(request(91, 8, 100, Vec::new()))
        .expect("the running world takes the request");
    let (first, _) = tokens(
        &mut running_events,
        GENERATED,
    );
    assert_eq!(first, expected[..GENERATED], "the running world generates the reference's first tokens");

    // Its successor starts beside it, under the same claim, with its cache.
    let caches = answer(running.describe_cache()).expect("the running world states its cache");
    let successor_pool = Arc::new(Serving::by_extent(vec![support::pool("model")]));
    let mut successor_config = EngineConfig::sim("worlds");
    successor_config.world = 2;
    successor_config.predecessor = Some(
        caches
            .into_iter()
            .map(|cache| {
                (
                    cache.worker_group.clone(),
                    ipc::PredecessorCache {
                        world: 1,
                        segments: cache.segments[..cache.held as usize].to_vec(),
                        num_units: cache.units,
                        scratch_units: cache.scratch,
                    },
                )
            })
            .collect(),
    );
    let successor = SchedulerHandle::over_batches(
        successor_config,
        successor_pool.clone() as Arc<dyn Batches>,
    )
    .expect("the successor starts beside the running world");
    install(&successor, &successor_pool);

    // The running world parks, the successor is promoted with its index, and
    // the running world closes.
    let index = answer(running.park(true)).expect("the running world parks");
    successor
        .promote(index, false, false)
        .expect("the successor is promoted with the parked index");
    running.shutdown();
    answer(successor.commit()).expect("the successor commits");

    // The continuation samples what the uninterrupted request sampled next.
    let mut continuation = successor
        .submit(request(91, 8, 100, first.clone()))
        .expect("the successor takes the continuation");
    let (continued, _) = tokens(
        &mut continuation,
        CONTINUED,
    );
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
    let first = batches.first().expect("the successor served the continuation");
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
