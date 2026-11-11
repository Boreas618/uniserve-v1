//! Block-diffusion text generation over the fake at the batch interface.
//!
//! A request of a deployment that generates in blocks prefills its prompt
//! without sampling, then generates text one block at a time: token-denoising
//! steps run on the block the rank holds until a step stops it and reports the
//! block's tokens, which are published and, unless they end the request,
//! committed to the context by a causal prefill before the next block starts.
//! The fake stops block `b` of a request at step
//! `testkit::block_stop_step(key, b, MAX_STEPS)` and fills it with the
//! request's text, `testkit::draw_token` at each logical position, followed by
//! EOS, so the expected output is known exactly. A step may be queued behind
//! the one still in flight; behind the step that stops the block it computes
//! nothing. Cases observe the events a request publishes and the batches the
//! fake accepts.

use crate::support;
use crate::support::fake::Pacer;

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ipc::execution::{Batch, BatchId, RowKind, Variant};
use ipc::{BlockSampling, RequestId, SamplingParams};
use ipc::{CallKind, ForwardMode};
use scheduler::{CachePolicy, Event, FinishReason, GenerationRequest, MultimodalInputs};
use scheduler::{CommandSender, EventRx, RejectionKind, SchedulerConfig};
use testkit::{Accepted, FakeBatches};

/// Tokens of one block.
const BLOCK: u32 = 16;
/// Denoising steps a block runs at most.
const MAX_STEPS: u32 = 4;
/// The block-diffusion sampling the deployment is launched with.
const SAMPLING: BlockSampling = BlockSampling {
    block_length: BLOCK,
    max_steps: MAX_STEPS,
    entropy_bound: 0.1,
    t_min: 0.4,
    t_max: 0.8,
    confidence_threshold: 0.005,
    stability_threshold: 1,
};

/// A scheduler configuration of the deployment that generates `SAMPLING`
/// blocks.
fn config() -> SchedulerConfig {
    SchedulerConfig {
        block_sampling: Some(SAMPLING),
        ..SchedulerConfig::default()
    }
}

/// A block-diffusion deployment over `testkit::FakeBatches`: one rank that
/// prefills prompts and denoises blocks, with 16-token cache pages and
/// `queue_depth` batches in flight and as many unresolved calls a request,
/// whose requests' text holds `text_len` tokens before EOS.
fn fake(text_len: u64, queue_depth: usize) -> Arc<FakeBatches> {
    let mut pool = support::pool("model");
    pool.layout.queue_depth = queue_depth;
    for rank in &mut pool.ranks {
        rank.supported_calls = vec![
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::TokenDenoising),
        ];
        // A request may have as many calls unresolved as the device queues batches.
        rank.max_unresolved_calls = queue_depth as u32;
        for group in rank
            .prefix_cache
            .iter_mut()
            .flat_map(|cache| cache.groups.iter_mut())
        {
            group.block_size = 16;
        }
    }
    Arc::new(
        FakeBatches::new(vec![pool])
            .with_text_len(text_len)
            .with_block_steps(MAX_STEPS),
    )
}

fn block_request(request_id: u64, prompt_len: u32, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(request_id),
        prompt_token_ids: (0..prompt_len).map(|index| 1_000 + index).collect(),
        multimodal_inputs: MultimodalInputs::default(),
        sampling: SamplingParams {
            // Each request names a seed of its own, so each draws its own stream.
            seed: Some(request_id),
            ..SamplingParams::default()
        },
        max_und_tokens: max_tokens,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy::default(),
        include_stop_token: false,
        generated_token_ids: Vec::new(),
        adapter: 0,
    }
}

/// The Philox stream of request `id`, which states its id as its seed: what
/// the fake's text and stopping steps follow. A seeded request's stream is a
/// function of its seed alone.
fn stream(id: u64) -> u64 {
    ipc::philox::request_sampling_key(Some(id), 0, 0, 0)
}

/// Tokens `range` of the text of the request on stream `key` whose prompt
/// holds `prompt` tokens.
fn text(key: u64, prompt: u64, range: std::ops::Range<u64>) -> Vec<u32> {
    range
        .map(|index| {
            testkit::draw_token(ipc::philox::DrawAddress {
                key,
                position: prompt + index,
            })
        })
        .collect()
}

/// A block-diffusion scheduler over a fake, running on its own thread.
struct Running {
    fake: Arc<FakeBatches>,
    handle: CommandSender,
    thread: thread::JoinHandle<bool>,
}

impl Running {
    fn start(fake: Arc<FakeBatches>, config: SchedulerConfig) -> Self {
        let batches: Arc<dyn ipc::execution::Batches> = Arc::clone(&fake) as _;
        let scheduler = support::fake::scheduler_over(batches, config, Default::default())
            .expect("the fake serves block diffusion");
        let (handle, thread) = support::fake::spawn(scheduler);
        Self {
            fake,
            handle,
            thread,
        }
    }

    fn submit(&self, request: GenerationRequest) -> EventRx {
        self.handle.submit(request).unwrap()
    }

    /// Stops the scheduler and returns every batch its fake accepted, in
    /// order, each with the batches in flight when it was accepted.
    fn stop(self) -> Vec<Accepted> {
        self.handle.shutdown();
        assert!(!self.thread.join().unwrap(), "the scheduler failed");
        self.fake.accepted()
    }
}

/// Collects one request's events through its terminal event.
fn events(receiver: EventRx) -> Vec<Event> {
    events_within(receiver, Duration::from_secs(20))
}

/// Collects one request's events through its terminal event, which must
/// arrive within `timeout`.
fn events_within(mut receiver: EventRx, timeout: Duration) -> Vec<Event> {
    let mut events = Vec::new();
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match receiver.try_recv() {
            Ok(event) => {
                let terminal = matches!(
                    event,
                    Event::Finished { .. } | Event::Rejected { .. } | Event::Error { .. }
                );
                events.push(event);
                if terminal {
                    return events;
                }
            }
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    panic!("request did not finish: {events:?}");
}

/// The token blocks a request published, in order.
fn published_blocks(events: &[Event]) -> Vec<Vec<u32>> {
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::TextToken { .. })),
        "block text is published by block: {events:?}"
    );
    events
        .iter()
        .filter_map(|event| match event {
            Event::TextTokens { ids } => Some(ids.clone()),
            _ => None,
        })
        .collect()
}

fn finish_of(events: &[Event]) -> (FinishReason, usize) {
    match events.last() {
        Some(Event::Finished {
            reason,
            completion_tokens,
            ..
        }) => (reason.clone(), *completion_tokens),
        other => panic!("request ended with {other:?}"),
    }
}

/// The denoising steps `batch` runs for the request on stream `key`: each
/// step's block and step.
fn steps_of(batch: &Batch, key: u64) -> Vec<(u32, u32)> {
    if batch.shape.variant != Variant::TokenDenoise {
        return Vec::new();
    }
    let columns = &batch.columns;
    columns
        .rows
        .calls
        .iter()
        .zip(&columns.block_steps)
        .filter(|(call, _)| columns.draws[call.rows.start as usize].key == key)
        .map(|(_, step)| (step.block, step.step))
        .collect()
}

/// The logical position each prefill call of `batch` starts at, for the
/// request on stream `key`.
fn prefill_starts(batch: &Batch, key: u64) -> Vec<u32> {
    let columns = &batch.columns;
    columns
        .rows
        .calls
        .iter()
        .filter(|call| {
            call.kind == RowKind::Prefill && columns.draws[call.rows.start as usize].key == key
        })
        .map(|call| {
            let first_token = columns.rows.query_offsets[call.rows.start as usize] as usize;
            columns.rows.positions[first_token] as u32
        })
        .collect()
}

/// The accepted batch running step `step` of block `block` of the request on
/// stream `key`.
fn step_batch(accepted: &[Accepted], key: u64, block: u32, step: u32) -> &Accepted {
    accepted
        .iter()
        .find(|accepted| steps_of(&accepted.batch, key).contains(&(block, step)))
        .unwrap_or_else(|| panic!("block {block} ran no step {step}"))
}

/// The place of batch `id` in the order the fake accepted batches.
fn order(accepted: &[Accepted], id: BatchId) -> usize {
    accepted
        .iter()
        .position(|accepted| accepted.batch.id == id)
        .expect("the batch was accepted")
}

/// With room for two batches in flight, each step of a block is queued
/// before the step it follows completes. The step behind the one that stops
/// the block computes nothing; the published text is unchanged.
#[test]
fn a_blocks_steps_are_queued_behind_the_step_in_flight() {
    let fake = fake(40, 2);
    let pacer = Pacer::start(&fake, Duration::from_millis(20));
    let running = Running::start(fake, config());
    let key = stream(5);
    let response = events(running.submit(block_request(5, 24, 1_000)));
    drop(pacer);
    let accepted = running.stop();

    assert_eq!(
        published_blocks(&response),
        vec![
            text(key, 24, 0..16),
            text(key, 24, 16..32),
            text(key, 24, 32..40)
        ]
    );
    assert_eq!(finish_of(&response), (FinishReason::Eos, 41));

    for block in 0..3u32 {
        let stop = testkit::block_stop_step(key, block, MAX_STEPS);
        let steps: Vec<(&Accepted, u32)> = accepted
            .iter()
            .flat_map(|accepted| {
                steps_of(&accepted.batch, key)
                    .into_iter()
                    .filter(|step| step.0 == block)
                    .map(move |step| (accepted, step.1))
            })
            .collect();
        // Every step up to the stopping one runs, in order, and at most one
        // more is queued behind it within the step limit.
        let expected_last = (stop + 1).min(MAX_STEPS - 1);
        assert_eq!(
            steps.iter().map(|step| step.1).collect::<Vec<_>>(),
            (0..=expected_last).collect::<Vec<_>>(),
            "block {block}"
        );
        for pair in steps.windows(2) {
            let (previous, next) = (&pair[0], &pair[1]);
            assert!(
                next.0.in_flight.contains(&previous.0.batch.id),
                "step {} of block {block} waited for step {}",
                next.1,
                previous.1
            );
        }
    }
}

/// With room for two batches in flight, each block's first step is queued
/// behind the prefill that completes its context, the prompt's last chunk
/// or the previous block's commit, before that prefill completes; the
/// published text is unchanged.
#[test]
fn a_blocks_first_step_is_queued_behind_the_prefill_before_it() {
    let fake = fake(40, 2);
    let pacer = Pacer::start(&fake, Duration::from_millis(20));
    let running = Running::start(fake, config());
    let key = stream(6);
    let response = events(running.submit(block_request(6, 24, 1_000)));
    drop(pacer);
    let accepted = running.stop();

    assert_eq!(
        published_blocks(&response),
        vec![
            text(key, 24, 0..16),
            text(key, 24, 16..32),
            text(key, 24, 32..40)
        ]
    );
    assert_eq!(finish_of(&response), (FinishReason::Eos, 41));

    let mut first_steps = 0;
    for (position, step) in accepted.iter().enumerate() {
        let Some((block, _)) = steps_of(&step.batch, key)
            .into_iter()
            .find(|step| step.1 == 0)
        else {
            continue;
        };
        let prefill = accepted[..position]
            .iter()
            .rev()
            .find(|earlier| !prefill_starts(&earlier.batch, key).is_empty())
            .expect("a prefill precedes every block");
        assert!(
            step.in_flight.contains(&prefill.batch.id),
            "step zero of block {block} waited for the prefill before it"
        );
        first_steps += 1;
    }
    assert_eq!(first_steps, 3);
}

/// With room for two batches in flight, a stopped block's commit is queued
/// as soon as the step that stopped the block completes: behind the step
/// still in flight that computes nothing, which the commit does not wait
/// for. The next block starts from step zero behind the commit, and the
/// published text is unchanged.
#[test]
fn a_blocks_commit_is_queued_behind_the_step_that_computes_nothing() {
    let fake = fake(40, 2);
    let pacer = Pacer::start(&fake, Duration::from_millis(20));
    let running = Running::start(fake, config());
    let key = stream(7);
    let response = events(running.submit(block_request(7, 24, 1_000)));
    drop(pacer);
    let accepted = running.stop();

    assert_eq!(
        published_blocks(&response),
        vec![
            text(key, 24, 0..16),
            text(key, 24, 16..32),
            text(key, 24, 32..40)
        ]
    );
    assert_eq!(finish_of(&response), (FinishReason::Eos, 41));

    // A commit is a prefill that starts at or after the prompt's end.
    let commits: Vec<&Accepted> = accepted
        .iter()
        .filter(|accepted| {
            prefill_starts(&accepted.batch, key)
                .iter()
                .any(|start| *start >= 24)
        })
        .collect();
    assert_eq!(commits.len(), 2);
    for (block, commit) in (0u32..).zip(commits) {
        let stop = testkit::block_stop_step(key, block, MAX_STEPS);
        assert!(stop + 1 < MAX_STEPS, "block {block} queues a step behind");
        let behind = step_batch(&accepted, key, block, stop + 1);
        assert!(
            commit.in_flight.contains(&behind.batch.id),
            "the commit of block {block} waited for the step behind step {stop}"
        );
        let next = step_batch(&accepted, key, block + 1, 0);
        assert!(
            order(&accepted, next.batch.id) > order(&accepted, commit.batch.id),
            "block {} starts behind the commit of block {block}",
            block + 1
        );
    }
}

/// A request whose last block stops before its step limit publishes its
/// terminal event once that step completes, while the step queued behind it
/// is still on the device; only the request's retirement waits for it.
#[test]
fn the_terminal_event_does_not_wait_for_the_step_behind_it() {
    let fake = fake(1_000, 2);
    fake.hold();
    let running = Running::start(Arc::clone(&fake), config());
    let key = stream(8);
    let stop = testkit::block_stop_step(key, 0, MAX_STEPS);
    // One block: the completion limit ends the request at the block's end.
    let mut receiver = running.submit(block_request(8, 24, BLOCK as usize));

    // Every batch completes in order but the step behind the stopping one,
    // which stays on the device.
    let mut response = Vec::new();
    let mut released = 0;
    let deadline = Instant::now() + Duration::from_secs(5);
    while !matches!(response.last(), Some(Event::Finished { .. })) {
        assert!(Instant::now() < deadline, "no terminal event: {response:?}");
        while let Ok(event) = receiver.try_recv() {
            response.push(event);
        }
        if let Some(next) = fake.accepted().get(released)
            && !steps_of(&next.batch, key).contains(&(0, stop + 1))
        {
            fake.release(1);
            released += 1;
        }
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(fake.held(), 1, "the step behind is still on the device");
    fake.resume();
    let accepted = running.stop();

    assert_eq!(published_blocks(&response), vec![text(key, 24, 0..16)]);
    assert_eq!(finish_of(&response), (FinishReason::MaxTokens, 16));
    let steps: Vec<u32> = accepted
        .iter()
        .flat_map(|accepted| steps_of(&accepted.batch, key))
        .map(|step| step.1)
        .collect();
    assert_eq!(
        steps,
        (0..=stop + 1).collect::<Vec<_>>(),
        "a step was queued behind the step that stopped the block"
    );
}

/// The completion limit truncates the block that reaches it, and no block
/// follows.
#[test]
fn the_completion_limit_truncates_inside_a_block() {
    let running = Running::start(fake(1_000, 1), config());
    let key = stream(2);
    let response = events(running.submit(block_request(2, 8, 20)));
    let accepted = running.stop();

    assert_eq!(
        published_blocks(&response),
        vec![text(key, 8, 0..16), text(key, 8, 16..20)]
    );
    assert_eq!(finish_of(&response), (FinishReason::MaxTokens, 20));
    let commits = accepted
        .iter()
        .flat_map(|accepted| prefill_starts(&accepted.batch, key))
        .filter(|start| *start >= 8)
        .count();
    assert_eq!(commits, 1, "only the first block is committed");
}

/// A stop token inside a block ends the request there; it is published only
/// when the request includes stop tokens in its output.
#[test]
fn a_stop_token_ends_the_request_inside_its_block() {
    let running = Running::start(fake(1_000, 1), config());
    let (excluded_key, included_key) = (stream(3), stream(4));
    let mut excluded = block_request(3, 8, 1_000);
    excluded.stop_token_ids = text(excluded_key, 8, 21..22);
    let mut included = block_request(4, 8, 1_000);
    included.stop_token_ids = text(included_key, 8, 5..6);
    included.include_stop_token = true;
    let excluded_events = events(running.submit(excluded));
    let included_events = events(running.submit(included));
    running.stop();

    assert_eq!(
        published_blocks(&excluded_events),
        vec![text(excluded_key, 8, 0..16), text(excluded_key, 8, 16..21)]
    );
    assert_eq!(finish_of(&excluded_events).0, FinishReason::Stop);
    assert_eq!(
        published_blocks(&included_events),
        vec![text(included_key, 8, 0..6)]
    );
    assert_eq!(finish_of(&included_events).0, FinishReason::Stop);
}

/// A denoising step and a block commit each need a whole block of one step's
/// token budget, so a deployment whose block is longer than the budget
/// fails to start.
#[test]
fn a_block_beyond_the_step_budget_fails_to_start() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 8,
        chunked_prefill_size: std::num::NonZeroUsize::new(8),
        ..config()
    };
    let error = support::fake::scheduler_over(fake(40, 1), config, Default::default())
        .err()
        .expect("a 16-token block exceeds an 8-token step budget");
    assert!(error.to_string().contains("step budget"), "{error}");
}

/// A request of a deployment that generates in blocks names the seed its
/// draws follow.
#[test]
fn an_unseeded_block_request_is_refused() {
    let running = Running::start(fake(40, 1), config());
    let mut request = block_request(6, 8, 64);
    request.sampling.seed = None;
    let response = events(running.submit(request));
    running.stop();

    assert!(
        matches!(
            response.last(),
            Some(Event::Rejected {
                kind: RejectionKind::Invalid,
                ..
            })
        ),
        "{response:?}"
    );
}

/// A prompt that waits while a block denoises is prefilled within the
/// prefill window instead of after the block stops, although the block's
/// next step is ready at every pass.
#[test]
fn a_waiting_prompt_is_prefilled_while_a_block_denoises() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 64,
        chunked_prefill_size: std::num::NonZeroUsize::new(64),
        ..config()
    };
    // The generating request's block stops at its third step, and it
    // generates one block.
    let generating_id = (1u64..)
        .find(|id| testkit::block_stop_step(stream(*id), 0, MAX_STEPS) == 2)
        .expect("some request's first block stops at its third step");
    let waiting_id = generating_id + 1;
    let (generating_key, waiting_key) = (stream(generating_id), stream(waiting_id));

    // Both requests are waiting before the scheduler starts, so the first
    // pass prefills the short prompt and the start of the long one.
    let fake = fake(1_000, 1);
    let batches: Arc<dyn ipc::execution::Batches> = Arc::clone(&fake) as _;
    let scheduler = support::fake::scheduler_over(batches, config, Default::default()).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let generating_events = handle
        .submit(block_request(generating_id, 16, BLOCK as usize))
        .unwrap();
    let waiting_events = handle
        .submit(block_request(waiting_id, 200, BLOCK as usize))
        .unwrap();
    let thread = thread::spawn(move || scheduler.run(rx));
    assert_eq!(
        published_blocks(&events(generating_events)),
        vec![text(generating_key, 16, 0..16)]
    );
    assert_eq!(
        published_blocks(&events(waiting_events)),
        vec![text(waiting_key, 200, 0..16)]
    );
    handle.shutdown();
    assert!(!thread.join().unwrap());

    let accepted = fake.accepted();
    // The batch whose prefill reaches the end of the long prompt: its last
    // row sees all 200 prompt tokens.
    let prompt_done = accepted
        .iter()
        .position(|accepted| {
            let columns = &accepted.batch.columns;
            columns.rows.calls.iter().any(|call| {
                call.kind == RowKind::Prefill
                    && columns.draws[call.rows.start as usize].key == waiting_key
                    && columns.rows.seq_lens[call.rows.end as usize - 1] == 200
            })
        })
        .expect("the long prompt is prefilled");
    let block_stopped = order(
        &accepted,
        step_batch(&accepted, generating_key, 0, 2).batch.id,
    );
    assert!(
        prompt_done < block_stopped,
        "the long prompt finished prefilling in batch {prompt_done}, after the block \
         that stopped in batch {block_stopped}"
    );
}
