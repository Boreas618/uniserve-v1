//! End-to-end engine-loop lifecycle behavior over the one fake at the worker's
//! batch interface.
//!
//! Each test drives a real `Scheduler` through the production submission door
//! (`Submissions::over_batches`) against `testkit::FakeBatches`, either on a
//! spawned thread through `Scheduler::run` or stepped on the test thread with
//! `Scheduler::step`, and observes behavior through `CommandSender` event
//! streams, `SchedulerStats` counters, the scheduler's public accessors and the
//! batches the fake received. Every completion travels the serving completion
//! path.
//!
//! Fake conventions the tests rely on:
//! - A sampled token is `testkit::draw_token` of the row's Philox address (the
//!   request's sampling key and the token's logical position), or, under
//!   `Tokens::Extent`, `1000` plus the keys the sampling row sees: a prompt of
//!   `p` tokens samples its first token at position `p`.
//! - A pool's text ends where its `EndOfText` states: from that logical position
//!   on every row samples `testkit::EOS`, the first EOS of
//!   `testkit::special_tokens`, so a request finishes with `Eos` there.
//! - `support::text_pool`'s queue depth bounds the batches the scheduler may
//!   keep unresolved. Depth 1 is the serial oracle that the depth-invariance
//!   tests compare depth 2 against.
//! - The cache block size is 64 tokens, one unit per page (`MainProcessInfo::default`),
//!   unless a test states other cache groups on the rank.
//! - The `general` request gauges and the `prefix_cache` and `encoder` counters in
//!   `SchedulerStats` are snapshots that `Scheduler::publish_cache_stats`
//!   refreshes as the loop runs, not when an event is sent, so they can trail
//!   the event stream. The `prefix` counters are recorded at admission.
//!
//! Input images reach the device only through a context prefill that writes
//! their vision or latent blocks, which the production column writer cannot
//! lay out yet (chat image input, `specs/tasks.md` T2): the cases that serve an
//! input image are ignored until it can.

use crate::support;

use ipc::execution::{Batch, RowKind};
use ipc::{CallKind, ForwardMode, MediaCall};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};
use testkit::{EndOfText, FakeBatches, Tokens};
use tokio::sync::mpsc::error::TryRecvError;

use ipc::{RequestId, SamplingParams};
use scheduler::{CommandSender, EVENT_BUFFER_CAPACITY, SchedulerConfig};
use scheduler::{Event, FinishReason};
use scheduler::{
    GenerationLimits, GenerationRequest, ImageEncoderInput, ImageInput, MultimodalInputs,
    RejectionKind,
};

/// A fake pool of one text rank at `depth` whose text ends at `end`.
fn text(depth: usize, end: Option<EndOfText>) -> Arc<FakeBatches> {
    Arc::new(
        support::text_pool(support::text_rank(), depth, end)
            .answering_media(Arc::new(testkit::Media::new())),
    )
}

/// A scheduler over `pool` batching at most 32 calls a pass.
fn scheduler_over(pool: &Arc<FakeBatches>) -> support::fake::Serving {
    support::fake::scheduler_over(
        Arc::clone(pool) as Arc<dyn ipc::execution::Batches>,
        SchedulerConfig {
            max_batch: proto::config::positive(32),
            ..Default::default()
        },
        Default::default(),
    )
    .expect("the scheduler is built over the fake")
}

/// How long a case waits between two looks at what a running scheduler
/// published.
const LOOK: Duration = Duration::from_millis(2);

/// Runs `scheduler` on a thread of its own, reading `commands`, and returns
/// the statistics it shares with the thread to join.
fn run(
    scheduler: support::fake::Serving,
    commands: crossbeam_channel::Receiver<scheduler::Command>,
) -> (Arc<scheduler::SchedulerStats>, thread::JoinHandle<bool>) {
    let stats = scheduler.stats_handle();
    (stats, thread::spawn(move || scheduler.run(commands)))
}

/// Waits until `settled` holds of the statistics a scheduler publishes after
/// each step, or five seconds pass.
fn settle(stats: &scheduler::SchedulerStats, settled: impl Fn(&scheduler::GeneralStats) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && !settled(&stats.general) {
        thread::sleep(LOOK);
    }
}

/// Waits until the scheduler over `pool` has submitted and then stopped
/// submitting: no batch is on the device and none arrived for fifty
/// milliseconds, a thousand times what a step of these cases takes.
fn stall(pool: &FakeBatches) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let submitted = pool.submitted().len();
        thread::sleep(Duration::from_millis(50));
        // A scheduler that has not submitted yet is still starting.
        let quiet = submitted > 0 && pool.held() == 0 && pool.submitted().len() == submitted;
        if quiet || Instant::now() >= deadline {
            return;
        }
    }
}

/// The calls of `batch` and the kind of its first call's rows: a token batch
/// states one row kind for all of its calls.
fn token_calls(batch: &Batch) -> Option<(RowKind, usize)> {
    let first = batch.columns.rows.calls.first()?;
    batch
        .media
        .is_none()
        .then_some((first.kind, batch.calls.len()))
}

/// The prompt tokens `batch`'s prefill calls write: the query tokens of every
/// row of a prefill call.
fn prefill_tokens(batch: &Batch) -> usize {
    let rows = &batch.columns.rows;
    rows.calls
        .iter()
        .filter(|call| call.kind == RowKind::Prefill)
        .flat_map(|call| call.rows.clone())
        .map(|row| {
            (rows.query_offsets[row as usize + 1] - rows.query_offsets[row as usize]) as usize
        })
        .sum()
}

/// Advertised generation features follow the worker's supported calls: each
/// encoder call enables only its own feature, and denoising calls enable
/// none of a token request's.
#[test]
fn generation_capabilities_require_complete_paths_and_distinct_encoders() {
    use scheduler::GenerationFeatures;

    let cases = [
        (
            vec![CallKind::Media(MediaCall::VisionEncoding)],
            GenerationFeatures::VISION_ENCODE,
        ),
        (
            vec![CallKind::Media(MediaCall::LatentEncoding)],
            GenerationFeatures::LATENT_ENCODE,
        ),
        (
            vec![
                CallKind::Media(MediaCall::VisionEncoding),
                CallKind::Media(MediaCall::LatentEncoding),
            ],
            GenerationFeatures::VISION_ENCODE | GenerationFeatures::LATENT_ENCODE,
        ),
        (
            vec![
                CallKind::Media(MediaCall::LatentPreparation),
                CallKind::Media(MediaCall::Denoising),
                CallKind::Media(MediaCall::VideoDecoding),
            ],
            GenerationFeatures::empty(),
        ),
    ];
    for (calls, expected) in cases {
        let mut rank = support::text_rank();
        rank.supported_calls = vec![
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::Decode),
        ];
        rank.supported_calls.extend(calls);
        let pool = Arc::new(support::text_pool(rank, 1, None));
        let scheduler = scheduler_over(&pool);
        assert_eq!(
            scheduler.generation_limits().features,
            GenerationFeatures::UNDERSTANDING | expected,
        );
    }
}

/// Encoder feature byte limits clamp to the smaller of the worker's encoder
/// cache entry size and buffer pool size, and the encoder cache entry count
/// clamps to the worker's advertised entries.
#[test]
fn encoder_products_obey_worker_entry_capacity() {
    let mut rank = support::text_rank();
    rank.buffer_pool_bytes = 8 << 30;
    rank.encoder_entry_bytes = 128 << 20;
    rank.encoder_cache_entries = 64;
    let pool = Arc::new(support::text_pool(rank, 1, None));
    let scheduler = scheduler_over(&pool);
    let limits = scheduler.generation_limits();
    assert_eq!(limits.max_vision_feature_bytes, 128 << 20);
    assert_eq!(limits.max_latent_feature_bytes, 128 << 20);
    assert_eq!(limits.encoder_cache_entries, 64);
}

fn text_input(token_ids: Vec<u32>) -> (Vec<u32>, MultimodalInputs) {
    (token_ids, MultimodalInputs::default())
}

/// Builds a prompt with one context image between `before` and `after`.
///
/// The image sits at prompt-token position `before.len()`, the boundary
/// between `before` and `after`, occupies `logical_positions` positions, and
/// contributes `physical_tokens` cache tokens through a single ViT encode. `id` names the image's digest
/// (`support::image_digest`), from which encoder-cache keys are derived; the
/// payload is a placeholder.
fn image_input(
    mut before: Vec<u32>,
    after: Vec<u32>,
    id: u64,
    logical_positions: u32,
    physical_tokens: u32,
) -> (Vec<u32>, MultimodalInputs) {
    let position = before.len() as u32;
    before.extend(after);
    (
        before,
        MultimodalInputs {
            images: vec![ImageInput {
                digest: support::image_digest(id),
                b64: "aW1hZ2U=".to_string(),
                position,
                num_positions: logical_positions,
                encoders: vec![ImageEncoderInput {
                    encoder: scheduler::ImageEncoder::Vision,
                    num_cache_tokens: Some(physical_tokens),
                    max_cache_tokens: None,
                    height: None,
                    width: None,
                }],
            }],
        },
    )
}

/// Builds a generation request.
///
/// # Panics
///
/// Panics when the request fails `GenerationRequest::validate_resources`
/// against the fixed limits below, so every test starts from a well-formed
/// request. The scheduler validates again against its own resolved limits.
fn generation_request(
    request_id: RequestId,
    (prompt_token_ids, multimodal_inputs): (Vec<u32>, MultimodalInputs),
    sampling: SamplingParams,
    max_und_tokens: usize,
) -> GenerationRequest {
    let cache = Default::default();
    let limits = GenerationLimits {
        features: scheduler::GenerationFeatures::UNDERSTANDING
            | scheduler::GenerationFeatures::VISION_ENCODE,
        max_vae_grid_tokens: 65_536,
        max_vit_grid_tokens: 8_192,
        max_latent_feature_bytes: 256 << 20,
        max_vision_feature_bytes: 256 << 20,
        encoder_cache_entries: 256,
        max_logprobs: 20,
    };
    let request = GenerationRequest {
        request_id,
        prompt_token_ids,
        multimodal_inputs,
        sampling,
        max_und_tokens,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache,
        generated_token_ids: Vec::new(),
        adapter: 0,
    };
    request
        .validate_resources(&limits)
        .expect("bounded simulation request");
    request
}

struct Collected {
    text: usize,
    finished: bool,
    reason: Option<FinishReason>,
}

/// Runs `count` text requests through a fresh scheduler at `depth`,
/// returning per-request collected counts.
///
/// Request ids are assigned from 1 in case order. A request that emits no event
/// before the 20-second deadline has no entry in the result, and one that emits
/// events but no terminal event has `finished == false`.
fn batch_requests(depth: usize, count: usize) -> HashMap<RequestId, Collected> {
    let pool = text(depth, Some(EndOfText::after(5, 8)));
    let sched = scheduler_over(&pool);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut rxs: HashMap<RequestId, scheduler::EventRx> = HashMap::new();
    let mut id = 1u64;
    {
        for _ in 0..count {
            let req = generation_request(
                RequestId(id),
                text_input(vec![1, 2, 3, 4, 5]),
                SamplingParams::default(),
                16,
            );
            let erx = handle.submit(req).unwrap();
            rxs.insert(RequestId(id), erx);
            id += 1;
        }
    }

    let total = rxs.len();
    let mut out: HashMap<RequestId, Collected> = HashMap::new();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut done = 0;
    while done < total && Instant::now() < deadline {
        for (rid, erx) in rxs.iter_mut() {
            while let Ok(ev) = erx.try_recv() {
                let c = out.entry(*rid).or_insert(Collected {
                    text: 0,
                    finished: false,
                    reason: None,
                });
                match ev {
                    Event::TextToken { .. } => c.text += 1,
                    Event::Finished { reason, .. } if !c.finished => {
                        c.finished = true;
                        c.reason = Some(reason);
                        done += 1;
                    }
                    _ => {}
                }
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = jh.join();
    out
}

#[test]
fn scheduler_clamps_max_batch_to_main_process_info() {
    let mut rank = support::text_rank();
    rank.max_batch_calls = 3;
    let sched = scheduler_over(&Arc::new(support::text_pool(rank, 1, None)));

    assert_eq!(sched.config().max_batch, proto::config::positive(3));
}

/// Prefill and decode batches hold at most the calls the workers report one
/// batch may hold: eight concurrent generations prefill and decode at most
/// three at a time and all finish.
#[test]
fn forward_batches_hold_at_most_the_workers_batch_calls() {
    let mut rank = support::text_rank();
    rank.max_batch_calls = 3;
    let pool = Arc::new(support::text_pool(rank, 1, Some(EndOfText::after(5, 8))));
    let sched = scheduler_over(&pool);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut receivers = (1..=8u64)
        .map(|id| {
            handle
                .submit(generation_request(
                    RequestId(id),
                    text_input(vec![1, 2, 3, 4, 5]),
                    SamplingParams::default(),
                    16,
                ))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut finished = 0;
    let deadline = Instant::now() + Duration::from_secs(20);
    while finished < receivers.len() && Instant::now() < deadline {
        for receiver in &mut receivers {
            while let Ok(event) = receiver.try_recv() {
                if let Event::Finished { reason, .. } = event {
                    assert_ne!(reason, FinishReason::Error);
                    finished += 1;
                }
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = jh.join();
    assert_eq!(finished, receivers.len(), "every request finishes");

    let (mut prefill_calls, mut decode_calls) = (Vec::new(), Vec::new());
    for batch in pool.submitted() {
        match token_calls(&batch) {
            Some((RowKind::Prefill, calls)) => prefill_calls.push(calls),
            Some((RowKind::Decode, calls)) => decode_calls.push(calls),
            _ => {}
        }
    }
    assert!(
        !prefill_calls.is_empty() && prefill_calls.iter().all(|&calls| calls <= 3),
        "prefill batches {prefill_calls:?}"
    );
    assert_eq!(prefill_calls.iter().sum::<usize>(), 8);
    assert!(
        !decode_calls.is_empty() && decode_calls.iter().all(|&calls| calls <= 3),
        "decode batches {decode_calls:?}"
    );
}

/// Single-worker results must be identical regardless of queue depth:
/// the same prompts produce the same per-request token counts at depth 1 and 2.
#[test]
fn queue_depth_is_token_identical() {
    let d1 = batch_requests(1, 4);
    let d2 = batch_requests(2, 4);
    for id in d1.keys() {
        assert_eq!(
            d1[id].text, d2[id].text,
            "req {id:?} token count differs by depth"
        );
        assert_eq!(d1[id].reason, d2[id].reason);
    }
}

/// Per-domain call accounting balances over a request's lifetime: every
/// launched call completes and returns its credit, and the stats reporter's
/// snapshot shows the same balance.
#[test]
fn call_window_metrics_record_the_full_lifecycle() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = scheduler::CommandSender::new(command_tx);

    let (stats, runner) = run(
        scheduler_over(&text(2, Some(EndOfText::after(5, 6)))),
        commands,
    );
    let request = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3, 4, 5]),
        SamplingParams::default(),
        16,
    );
    let mut events = handle.submit(request).unwrap();
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && !finished {
        while let Ok(event) = events.try_recv() {
            if matches!(event, Event::Finished { .. }) {
                finished = true;
            }
        }
        thread::sleep(LOOK);
    }
    assert!(finished, "request finished");
    // The counters are compared once the loop has drained and stopped.
    handle.shutdown();
    assert!(!runner.join().unwrap(), "the scheduler failed");

    let active_domains = [
        &stats.domains.prefill,
        &stats.domains.decode,
        &stats.domains.flow,
    ]
    .into_iter()
    .filter(|domain| domain.launched_calls.load(Ordering::Relaxed) > 0)
    .collect::<Vec<_>>();
    assert!(!active_domains.is_empty(), "domain launches recorded");
    for domain in &active_domains {
        assert_eq!(domain.active_credits.load(Ordering::Relaxed), 0);
        assert_eq!(
            domain.launched_calls.load(Ordering::Relaxed),
            domain.completed_calls.load(Ordering::Relaxed)
        );
        assert_eq!(
            domain.launched_calls.load(Ordering::Relaxed),
            domain.reclaimed_credits.load(Ordering::Relaxed)
        );
        assert!(domain.completed_batches.load(Ordering::Relaxed) > 0);
    }
    let mut reporter = scheduler::SchedulerStatsReporter::default();
    let snapshot = reporter.snapshot(&stats);
    let decoded_active = snapshot
        .domain_stats
        .iter()
        .filter(|domain| domain.launched_calls > 0)
        .collect::<Vec<_>>();
    assert_eq!(decoded_active.len(), active_domains.len());
    assert!(decoded_active.iter().all(|domain| {
        domain.active_credits == 0
            && domain.launched_calls == domain.completed_calls
            && domain.launched_calls == domain.reclaimed_credits
    }));
}

/// Runs one text request on a fresh scheduler and returns its emitted token
/// ids.
///
/// `depth` is the pool's queue depth: 1 gives the serial oracle, and 2 lets
/// the scheduler submit a successor before its predecessor's token has been
/// observed on the host. The five-token prompt's text ends after `text_len`
/// tokens.
fn relay_run(
    sampling: SamplingParams,
    stop_token_ids: Vec<u32>,
    max_und_tokens: usize,
    text_len: usize,
    depth: usize,
) -> Vec<u32> {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = scheduler::CommandSender::new(command_tx);

    let (_stats, runner) = run(
        scheduler_over(&text(depth, Some(EndOfText::after(5, text_len as u64)))),
        commands,
    );
    let mut request = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3, 4, 5]),
        sampling,
        max_und_tokens,
    );
    request.stop_token_ids = stop_token_ids;
    let mut events = handle.submit(request).unwrap();
    let mut tokens = Vec::new();
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && !finished {
        while let Ok(event) = events.try_recv() {
            match event {
                Event::TextToken { id, .. } => tokens.push(id),
                Event::Finished { .. } => finished = true,
                _ => {}
            }
        }
        thread::sleep(LOOK);
    }
    assert!(finished, "processor request reached a terminal event");
    handle.shutdown();
    assert!(!runner.join().unwrap(), "the scheduler failed");

    tokens
}

/// Requests whose sampling states processors (penalties, logprobs, minimum-token
/// floors and stop tokens) yield the same tokens when successors are
/// launched ahead of host observation at depth 2 as in the serial depth-one run:
/// what the host decides from each token (stop tokens, the minimum-token floor,
/// the logprobs it keeps) does not depend on when it observes it.
#[test]
fn generalized_processor_successors_match_depth_one_before_observation() {
    let cases: Vec<(&str, SamplingParams, Vec<u32>)> = vec![
        (
            "repetition+frequency+presence penalties",
            SamplingParams {
                temperature: 0.9,
                top_k: 40,
                seed: Some(7),
                repetition_penalty: 1.3,
                frequency_penalty: 0.7,
                presence_penalty: 0.4,
                ..SamplingParams::default()
            },
            Vec::new(),
        ),
        (
            "requested logprobs",
            SamplingParams {
                temperature: 0.7,
                seed: Some(11),
                n_logprobs: 5,
                ..SamplingParams::default()
            },
            Vec::new(),
        ),
        (
            "minimum-token floor over stop tokens",
            SamplingParams {
                temperature: 0.0,
                min_tokens: 6,
                ..SamplingParams::default()
            },
            vec![1_002, 1_003],
        ),
        (
            "greedy penalties with logprobs and a stop token",
            SamplingParams {
                temperature: 0.0,
                repetition_penalty: 1.5,
                frequency_penalty: 0.25,
                n_logprobs: 3,
                ..SamplingParams::default()
            },
            vec![1_004],
        ),
    ];

    for (label, sampling, stop_token_ids) in cases {
        let serial = relay_run(sampling.clone(), stop_token_ids.clone(), 24, 12, 1);
        let relayed = relay_run(sampling, stop_token_ids, 24, 12, 2);

        assert!(!serial.is_empty(), "[{label}] produced no tokens");
        assert_eq!(
            serial, relayed,
            "[{label}] device-relay tokens diverged from the depth-one serial oracle",
        );
    }
}

/// Decoding after a context image produces the same eight tokens and EOS at
/// queue depths 1 and 2.
#[test]
fn image_context_decode_is_depth_invariant() {
    let mut runs = Vec::new();
    for queue_depth in [1, 2] {
        let (command_tx, commands) = crossbeam_channel::unbounded();
        let handle = scheduler::CommandSender::new(command_tx);
        // Two prompt tokens, the image's four positions and two more tokens.
        let (_stats, runner) = run(
            scheduler_over(&text(queue_depth, Some(EndOfText::after(8, 8)))),
            commands,
        );
        let request = generation_request(
            RequestId(1),
            image_input(vec![1, 2], vec![3, 4], 0xD3C0DE, 4, 1),
            SamplingParams::default(),
            16,
        );
        let mut events = handle.submit(request).unwrap();
        let mut finish_reason = None;
        let mut tokens = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && finish_reason.is_none() {
            while let Ok(event) = events.try_recv() {
                match event {
                    Event::TextToken { id, .. } => tokens.push(id),
                    Event::Finished { reason, .. } => finish_reason = Some(reason),
                    _ => {}
                }
            }
            thread::sleep(LOOK);
        }
        handle.shutdown();
        assert!(!runner.join().unwrap(), "the scheduler failed");
        assert_eq!(finish_reason, Some(FinishReason::Eos));
        assert_eq!(tokens.len(), 8);
        runs.push(tokens);
    }

    assert_eq!(runs[0], runs[1]);
}

#[test]
fn stop_token_terminates_with_stop() {
    let pool =
        Arc::new(support::text_pool(support::text_rank(), 2, None).with_tokens(Tokens::Extent));
    let sched = scheduler_over(&pool);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut req = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3]),
        SamplingParams::default(),
        64,
    );
    // The three-token prompt samples its first token at position 3, which
    // `Tokens::Extent` makes 1003, so the request stops on its first sample;
    // `include_stop_token` is false, so nothing is emitted.
    req.stop_token_ids = vec![testkit::SAMPLED + 3];
    let mut erx = handle.submit(req).unwrap();

    let mut reason = None;
    let mut text = 0;
    let deadline = Instant::now() + Duration::from_secs(10);
    while reason.is_none() && Instant::now() < deadline {
        if let Ok(ev) = erx.try_recv() {
            match ev {
                Event::TextToken { .. } => text += 1,
                Event::Finished { reason: r, .. } => reason = Some(r),
                _ => {}
            }
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    }
    handle.shutdown();
    let _ = jh.join();
    assert_eq!(reason, Some(FinishReason::Stop));
    assert_eq!(text, 0, "the stop token itself must not be emitted");
}

/// Issues `CommandSender::abort` (when `abort`) or `CommandSender::cancel` once
/// a text request has emitted its first token, and returns its finish reason.
///
/// A text that never ends keeps the request generating, so the command is
/// observed mid-flight.
fn run_until_control(abort: bool) -> FinishReason {
    let sched = scheduler_over(&text(2, None));
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3]),
        SamplingParams::default(),
        1024,
    );
    let mut erx = handle.submit(req).unwrap();

    // Wait until it is generating, then issue the control command.
    let mut saw_token = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !saw_token && Instant::now() < deadline {
        if let Ok(Event::TextToken { .. }) = erx.try_recv() {
            saw_token = true;
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    }
    if abort {
        handle.abort(RequestId(1));
    } else {
        handle.cancel(RequestId(1));
    }

    let mut reason = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while reason.is_none() && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(Event::Finished { reason: r, .. }) => reason = Some(r),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();
    reason.expect("request never finished")
}

#[test]
fn abort_and_cancel_are_distinct() {
    assert_eq!(run_until_control(false), FinishReason::Cancelled);
    assert_eq!(run_until_control(true), FinishReason::Aborted);
}

/// A stop-string cutoff reported through `CommandSender::stop` finishes only
/// its own request with `Stop`; a concurrent request runs to its own end.
#[test]
fn stop_string_cutoff_is_request_local() {
    let scheduler = scheduler_over(&text(2, None));
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));

    let mut unrelated_events = handle
        .submit(generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            6,
        ))
        .unwrap();

    let mut stopping = generation_request(
        RequestId(2),
        text_input(vec![1, 2, 3]),
        SamplingParams::default(),
        1024,
    );
    stopping.stop_strings = vec!["boundary".to_string()];
    let mut stopping_events = handle.submit(stopping).unwrap();

    // A request with stop strings is acknowledged by the frontend decoder
    // (see `CommandSender::submit`). The test plays that role for the first
    // token and, once a second token arrives, stops the request.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut consumed_tokens = 0;
    while consumed_tokens < 2 && Instant::now() < deadline {
        match stopping_events.try_recv() {
            Ok(Event::TextToken { .. }) => {
                consumed_tokens += 1;
                if consumed_tokens == 1 {
                    handle.acknowledge_at(RequestId(2), 1);
                }
            }
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert_eq!(consumed_tokens, 2);
    handle.stop(RequestId(2));

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut stop_reason = None;
    let mut unrelated_reason = None;
    while Instant::now() < deadline {
        while let Ok(event) = stopping_events.try_recv() {
            if let Event::Finished { reason, .. } = event {
                stop_reason = Some(reason);
            }
        }
        while let Ok(event) = unrelated_events.try_recv() {
            if let Event::Finished { reason, .. } = event {
                unrelated_reason = Some(reason);
            }
        }
        if stop_reason.is_some() && unrelated_reason.is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = scheduler_thread.join();

    assert_eq!(stop_reason, Some(FinishReason::Stop));
    assert!(matches!(
        unrelated_reason,
        Some(FinishReason::Eos | FinishReason::MaxTokens)
    ));
}

/// Deployment policy controls prefix reuse without changing generated tokens.
/// Both requests ask to read and write prefixes; the disabled deployment must
/// override that permission while preserving each request's active cache state.
#[test]
fn prefix_cache_configuration_controls_reuse() {
    use scheduler::SchedulerHandle;

    let prompt: Vec<u32> = (0..600u32).map(|i| (i % 53) + 7).collect();
    let mut outputs = Vec::new();
    for enabled in [true, false] {
        let pool = text(1, Some(EndOfText::after(600, 4)));
        let mut args = testkit::args("sim-model");
        args.engine_config.kv_cache.prefix_caching = enabled;
        let engine = SchedulerHandle::over_batches(
            &args,
            testkit::limits(pool.as_ref()),
            Arc::clone(&pool) as Arc<dyn ipc::execution::Batches>,
            scheduler::CommandWaker::noop(),
        )
        .unwrap();
        support::install_policy_on(&engine, scheduler::BatchCapacity::of(&support::text_rank()));
        let stats = engine.stats();
        let mut replies = Vec::new();

        // The second request can reuse nine full pages from the first when
        // caching is enabled. Page tails still require ordinary prefill.
        for rid in 1..=2 {
            let mut events = engine
                .submit(generation_request(
                    RequestId(rid),
                    text_input(prompt.clone()),
                    SamplingParams::default(),
                    8,
                ))
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut tokens = Vec::new();
            let mut finish = None;
            while finish.is_none() && Instant::now() < deadline {
                match events.try_recv() {
                    Ok(Event::TextToken { id, .. }) => tokens.push(id),
                    Ok(Event::Finished { reason, .. }) => finish = Some(reason),
                    Ok(_) => {}
                    Err(_) => thread::sleep(Duration::from_millis(1)),
                }
            }
            assert_eq!(finish, Some(FinishReason::Eos));
            assert!(!tokens.is_empty());
            replies.push(tokens);
        }

        let hits = stats.prefix.hit_tokens.load(Ordering::Relaxed);
        if enabled {
            assert!(
                hits >= 512,
                "the shared prompt did not reuse cached keys and values"
            );
        } else {
            assert_eq!(
                hits, 0,
                "disabled prefix caching still reused keys and values"
            );
            assert_eq!(
                stats.prefix_cache.pages_stored.load(Ordering::Relaxed),
                0,
                "disabled prefix caching retained reusable pages"
            );
        }
        outputs.push(replies);
        engine.shutdown();
    }
    assert_eq!(outputs[0], outputs[1]);
}

/// Workers that keep cache layers per request slot (a recurrent state)
/// checkpoint them in a state group: a request of the same prompt reuses the
/// prefix up to the checkpoint the first request's prompt kept at its last
/// page boundary, and no deeper.
#[test]
fn a_state_group_reuses_prefixes_up_to_a_checkpoint() {
    use scheduler::SchedulerHandle;
    use std::sync::atomic::Ordering;

    let prompt: Vec<u32> = (0..600u32).map(|i| (i % 53) + 7).collect();
    let mut rank = support::text_rank();
    let cache = rank
        .prefix_cache
        .as_mut()
        .expect("a text rank has a prefix cache");
    for group in &mut cache.groups {
        group.block_size = 16;
    }
    // A state group after the K/V groups: a checkpoint of one layer per request
    // slot, a page of two units at every boundary of the first group's pages.
    let first = cache.groups[0].clone();
    let layer = cache
        .groups
        .iter()
        .flat_map(|group| group.layer_ids.iter().copied())
        .max()
        .map_or(0, |layer| layer + 1);
    cache.groups.push(ipc::CacheGroup {
        kind: ipc::CacheGroupKind::State,
        block_size: first.block_size,
        units_per_page: 2,
        layer_ids: vec![layer],
        num_kv_heads: 1,
        total_kv_heads: 1,
        kv_head_offset: 0,
        head_dim: 1,
    });
    let pool = Arc::new(support::text_pool(
        rank.clone(),
        1,
        Some(EndOfText::after(600, 4)),
    ));
    let mut args = testkit::args("sim-model");
    args.engine_config.kv_cache.prefix_caching = true;
    let engine = SchedulerHandle::over_batches(
        &args,
        testkit::limits(pool.as_ref()),
        Arc::clone(&pool) as Arc<dyn ipc::execution::Batches>,
        scheduler::CommandWaker::noop(),
    )
    .unwrap();
    support::install_policy_on(&engine, scheduler::BatchCapacity::of(&rank));
    let stats = engine.stats();
    for rid in 1..=2 {
        let mut events = engine
            .submit(generation_request(
                RequestId(rid),
                text_input(prompt.clone()),
                SamplingParams::default(),
                8,
            ))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut finish = None;
        while finish.is_none() && Instant::now() < deadline {
            match events.try_recv() {
                Ok(Event::Finished { reason, .. }) => finish = Some(reason),
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        assert_eq!(finish, Some(FinishReason::Eos));
    }
    // The prompt of 600 tokens is one chunk, which checkpoints boundary 592:
    // the deepest boundary that leaves a token to compute.
    assert_eq!(stats.prefix.hit_tokens.load(Ordering::Relaxed), 592);
    assert!(stats.prefix_cache.pages_stored.load(Ordering::Relaxed) > 0);
    engine.shutdown();
}

/// Per-request `CachePolicy` governs prefix reuse: isolation keys partition the
/// cache, `read: false` skips lookup but still publishes, and `write: false`
/// looks up but publishes nothing. Each stage uses a fresh isolation key so
/// earlier stages cannot supply hits.
#[test]
fn prefix_cache_enforces_read_write_and_isolation_policy() {
    use std::sync::atomic::Ordering;

    let sched = scheduler_over(&text(1, Some(EndOfText::after(600, 2))));
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));
    let prompt: Vec<u32> = (0..600_u32).map(|index| (index % 47) + 5).collect();

    let run = |id: u64, read: bool, write: bool, isolation_key: u64| {
        let mut request = generation_request(
            RequestId(id),
            text_input(prompt.clone()),
            SamplingParams::default(),
            8,
        );
        request.cache = scheduler::CachePolicy {
            read,
            write,
            isolation_key: Some(isolation_key),
            state_seeds: Vec::new(),
            resume: None,
        };
        let mut events = handle.submit(request).expect("submit cache request");
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match events.try_recv() {
                Ok(Event::Finished { .. }) => return,
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        panic!("cache request {id} did not finish");
    };

    run(1, true, true, 11);
    let cold_hits = stats.prefix.hit_tokens.load(Ordering::Relaxed);
    run(2, true, true, 22);
    assert_eq!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed),
        cold_hits,
        "a different isolation key reused cached pages"
    );
    run(3, true, true, 11);
    assert!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed) >= cold_hits + 128,
        "the matching isolation key did not reuse its prefix"
    );

    let before_bypass = stats.prefix.hit_tokens.load(Ordering::Relaxed);
    run(4, false, true, 33);
    assert_eq!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed),
        before_bypass
    );
    run(5, true, true, 33);
    assert!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed) >= before_bypass + 128,
        "read bypass prevented a write-enabled request from publishing its prefix"
    );

    let before_no_store = stats.prefix.hit_tokens.load(Ordering::Relaxed);
    run(6, true, false, 44);
    run(7, true, true, 44);
    assert_eq!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed),
        before_no_store,
        "a write-disabled request published its prefix"
    );

    handle.shutdown();
    let _ = jh.join();
}

/// A scheduler with no host KV cache has nothing to warm through: it names no
/// request, so an epoch switch's warming waits for none of its requests.
#[test]
fn a_scheduler_without_a_host_cache_warms_nothing() {
    let sched = support::fake::scheduler_over(
        text(1, None),
        SchedulerConfig::default(),
        Default::default(),
    )
    .unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let answer = handle
        .cache_warm(
            storage::manifest::cache::Operation::Warmed,
            scheduler::HookBarrier::new(1),
        )
        .recv()
        .unwrap()
        .unwrap();
    let snapshot: storage::manifest::cache::Snapshot = serde_json::from_str(&answer[0]).unwrap();
    assert!(snapshot.prefixes.is_empty());

    drop(handle);
    jh.join().unwrap();
}

/// Under FCFS with a 256-token step budget and prefill chunks capped at 64
/// tokens, a 1000-token prompt is prefilled in many chunks; it and a
/// concurrent short request must both finish and emit text. The test observes
/// completion and the worker's prefill token budget while warming caps it.
#[test]
fn chunked_prefill_progresses_with_decode() {
    let pool = text(2, None);
    let sched = support::fake::scheduler_over(
        Arc::<testkit::FakeBatches>::clone(&pool),
        SchedulerConfig {
            max_batch: proto::config::positive(32),
            // Caps a prefill chunk at 64 tokens.
            chunked_prefill_size: std::num::NonZeroUsize::new(64),
            // Leaves room for other decodes per step.
            max_num_batched_tokens: 256,
            ..Default::default()
        },
        Default::default(),
    )
    .unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    handle
        .cache_warm(
            storage::manifest::cache::Operation::Prefill { tokens: Some(16) },
            scheduler::HookBarrier::new(1),
        )
        .recv()
        .unwrap()
        .unwrap();

    // A long prompt plus a short concurrent request.
    let long_prompt: Vec<u32> = (0..1000u32).map(|i| (i % 91) + 7).collect();
    let mut erx1 = handle
        .submit(generation_request(
            RequestId(1),
            text_input(long_prompt),
            SamplingParams::default(),
            8,
        ))
        .unwrap();
    let mut erx2 = handle
        .submit(generation_request(
            RequestId(2),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            8,
        ))
        .unwrap();

    let collect = |erx: &mut scheduler::EventRx| -> (usize, bool) {
        let mut text = 0;
        let mut done = false;
        let deadline = Instant::now() + Duration::from_secs(15);
        while !done && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(Event::TextToken { .. }) => text += 1,
                Ok(Event::Finished { .. }) => done = true,
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        (text, done)
    };
    let (t1, d1) = collect(&mut erx1);
    let (t2, d2) = collect(&mut erx2);
    // The prefill tokens of each batch submitted since the last reading.
    let mut read = 0;
    let mut prefilled = || {
        let submitted = pool.submitted();
        let tokens = submitted[read..]
            .iter()
            .map(prefill_tokens)
            .filter(|&tokens| tokens > 0)
            .collect::<Vec<_>>();
        read = submitted.len();
        tokens
    };
    let capped = prefilled();
    assert!(!capped.is_empty() && capped.iter().all(|&tokens| tokens <= 16));
    handle
        .cache_warm(
            storage::manifest::cache::Operation::Prefill { tokens: None },
            scheduler::HookBarrier::new(1),
        )
        .recv()
        .unwrap()
        .unwrap();
    let mut restored = handle
        .submit(generation_request(
            RequestId(3),
            text_input(vec![90; 128]),
            SamplingParams::default(),
            1,
        ))
        .unwrap();
    assert!(collect(&mut restored).1);
    assert!(prefilled().into_iter().any(|tokens| tokens > 16));
    handle.shutdown();
    let _ = jh.join();
    assert!(d1 && d2, "both requests must finish (d1={d1} d2={d2})");
    assert!(t1 > 0 && t2 > 0, "both must produce text (t1={t1} t2={t2})");
}

/// Runs one text request with the given sampling params over a pool whose text
/// ends after `text_len` tokens,
/// returning the emitted token ids, whether any token event carried a logprob,
/// and the finish reason (`None` when the request did not finish within the
/// 10-second deadline).
fn run_sampling(
    sampling: SamplingParams,
    text_len: usize,
    max_tokens: usize,
) -> (Vec<u32>, bool, Option<FinishReason>) {
    let sched = scheduler_over(&text(1, Some(EndOfText::after(3, text_len as u64))));
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3]),
        sampling,
        max_tokens,
    );
    let mut erx = handle.submit(req).unwrap();

    let mut toks = Vec::new();
    let mut any_logprob = false;
    let mut finished = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while finished.is_none() && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(Event::TextToken { id, logprob, .. }) => {
                toks.push(id);
                if logprob.is_some() {
                    any_logprob = true;
                }
            }
            Ok(Event::Finished { reason, .. }) => finished = Some(reason),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();
    (toks, any_logprob, finished)
}

#[test]
fn logprobs_flow_to_events() {
    let sp = SamplingParams {
        n_logprobs: 3,
        ..Default::default()
    };
    let (toks, any_logprob, finished) = run_sampling(sp, 8, 16);
    assert!(finished.is_some());
    assert!(!toks.is_empty());
    assert!(
        any_logprob,
        "logprobs requested but none populated on the event stream"
    );
}

#[test]
fn min_tokens_floor_overrides_early_eos() {
    // With text_len=1 the pool samples EOS from the second token on;
    // min_tokens=5 makes the scheduler pass over EOS until 5 tokens exist.
    let sp = SamplingParams {
        min_tokens: 5,
        ..Default::default()
    };
    let (toks, _lp, finished) = run_sampling(sp, 1, 50);
    assert!(finished.is_some());
    assert!(
        toks.len() >= 5,
        "min_tokens floor not honored, only {} tokens",
        toks.len()
    );
}

/// The first request with a context image misses the encoder cache and leaves
/// its encoder output resident; a second request with the same image hash hits
/// that cached output.
#[test]
fn multimodal_encode_then_cache_hit() {
    let sched = scheduler_over(&text(2, Some(EndOfText::after(8, 8))));
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let run_img = |rid: u64, handle: &CommandSender| -> (bool, Vec<String>) {
        let req = generation_request(
            RequestId(rid),
            image_input(vec![1, 2, 3], vec![9], 0xCAFE, 4, 1),
            SamplingParams::default(),
            8,
        );
        let mut erx = handle.submit(req).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut done = false;
        let mut seen = Vec::new();
        while !done && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(Event::Finished { reason, .. }) => {
                    seen.push(format!("finished:{reason:?}"));
                    done = true;
                }
                Ok(event) => seen.push(format!("{event:?}")),
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        (done, seen)
    };

    let (ok, seen) = run_img(1, &handle);
    assert!(
        ok,
        "context-image request 1 must produce an image, saw {seen:?}"
    );
    assert_eq!(
        stats.encoder.cache_hits.load(Ordering::Relaxed),
        0,
        "first image is a cache miss"
    );
    // Encoder counters are snapshots that can trail the terminal event, so
    // poll them briefly instead of reading once.
    let cache_deadline = Instant::now() + Duration::from_secs(2);
    while stats.encoder.cached.load(Ordering::Relaxed) == 0 && Instant::now() < cache_deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert!(
        stats.encoder.cached.load(Ordering::Relaxed) >= 1,
        "encoder output should be cached"
    );

    let (ok, seen) = run_img(2, &handle);
    assert!(ok, "request 2 must produce an image, saw {seen:?}");
    let hit_deadline = Instant::now() + Duration::from_secs(2);
    while stats.encoder.cache_hits.load(Ordering::Relaxed) == 0 && Instant::now() < hit_deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert!(
        stats.encoder.cache_hits.load(Ordering::Relaxed) >= 1,
        "repeated image must hit the encoder cache"
    );

    handle.shutdown();
    let _ = jh.join();
}

/// Two requests with the same image hash are both submitted before the
/// scheduler first steps; once both finish, the encoder cache holds exactly
/// one entry for that image.
#[test]
fn concurrent_same_image_misses_converge_on_one_exact_cached_product() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = scheduler::CommandSender::new(command_tx);

    // Three prompt tokens, the image's four positions and one more token.
    let scheduler = scheduler_over(&text(2, Some(EndOfText::after(8, 6))));
    let request = |request_id| {
        generation_request(
            RequestId(request_id),
            image_input(vec![1, 2, 3], vec![9], 0xCAFE, 4, 1),
            SamplingParams::default(),
            12,
        )
    };
    let mut first = handle.submit(request(1)).unwrap();
    let mut second = handle.submit(request(2)).unwrap();
    let mut reasons = [None, None];
    let mut text_tokens = [0_usize, 0_usize];
    let mut seen = [Vec::new(), Vec::new()];
    // Both requests wait before the scheduler takes its first step.
    let (stats, runner) = run(scheduler, commands);
    let deadline = Instant::now() + Duration::from_secs(5);

    while Instant::now() < deadline {
        thread::sleep(LOOK);
        for (index, events) in [&mut first, &mut second].into_iter().enumerate() {
            while let Ok(event) = events.try_recv() {
                seen[index].push(format!("{event:?}"));
                match event {
                    Event::TextToken { .. } => text_tokens[index] += 1,
                    Event::Finished { reason, .. } => reasons[index] = Some(reason),
                    _ => {}
                }
            }
        }
        if reasons.iter().all(Option::is_some) {
            break;
        }
    }

    assert_eq!(
        reasons,
        [Some(FinishReason::Eos), Some(FinishReason::Eos)],
        "events={seen:?}"
    );
    assert!(text_tokens.iter().all(|count| *count > 0));
    handle.shutdown();
    assert!(!runner.join().unwrap(), "the scheduler failed");
    assert_eq!(
        stats
            .encoder
            .cached
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn und_only_image_context_encodes_then_produces_text_without_gen_output() {
    let sched = scheduler_over(&text(1, Some(EndOfText::after(8, 20))));
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let request = generation_request(
        RequestId(81),
        image_input(vec![1, 2], vec![3, 4], 0x81, 4, 17),
        SamplingParams::default(),
        16,
    );
    let mut events = handle.submit(request).expect("submit request");
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut text_tokens = 0;
    let mut finished = false;
    while !finished && Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::TextToken { .. }) => text_tokens += 1,
            Ok(Event::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }

    handle.shutdown();
    let _ = jh.join();
    assert!(finished, "Und-only image-context request did not finish");
    assert!(text_tokens > 0, "Und-only request emitted no visible text");
    assert!(
        stats.encoder.cached.load(Ordering::Relaxed) >= 1,
        "input image was not encoded and retained in the encoder cache"
    );
}

/// A context prefill writes the prompt and an input image's vision block in
/// one call, the block whole, so the step budget bounds the **context** the
/// prefill reaches and not the block alone: a request whose context does not
/// fit is refused at submission, and one that fits exactly is served.
///
/// The prompt here is four tokens, so a block of twelve makes the sixteen-token
/// budget exactly and is served, while a block of thirteen does not fit and is
/// refused. A block that fits the budget on its own and not beside its prompt
/// (sixteen, against this prompt) is refused too, and the reason names the
/// context.
#[test]
fn a_vision_block_beyond_the_step_budget_is_refused() {
    const STEP_BUDGET: u32 = 16;

    let sched = support::fake::scheduler_over(
        text(1, Some(EndOfText::after(5, 2))),
        SchedulerConfig {
            max_batch: proto::config::positive(32),
            max_num_batched_tokens: STEP_BUDGET as usize,
            ..Default::default()
        },
        Default::default(),
    )
    .unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    // The first terminal event of a request whose image has one vision
    // block of `block_tokens` cache tokens.
    let terminal = |request_id: u64, block_tokens: u32| {
        let request = generation_request(
            RequestId(request_id),
            image_input(vec![1, 2], vec![3, 4], request_id, 1, block_tokens),
            SamplingParams::default(),
            4,
        );
        let mut events = handle.submit(request).expect("submit request");
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match events.try_recv() {
                Ok(event @ (Event::Rejected { .. } | Event::Finished { .. })) => {
                    return Some(event);
                }
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        None
    };
    // Four prompt tokens and a block of twelve fill the step budget exactly.
    const PROMPT_TOKENS: u32 = 4;
    let oversized = terminal(96, STEP_BUDGET + 1);
    let alone = terminal(98, STEP_BUDGET);
    let fitting = terminal(97, STEP_BUDGET - PROMPT_TOKENS);
    handle.shutdown();
    let _ = jh.join();

    assert!(
        matches!(
            oversized,
            Some(Event::Rejected {
                kind: RejectionKind::Invalid,
                ..
            })
        ),
        "{oversized:?}"
    );
    assert!(
        matches!(
            &alone,
            Some(Event::Rejected {
                kind: RejectionKind::Invalid,
                ..
            })
        ),
        "a block that fills the budget alone does not fit beside its prompt: {alone:?}"
    );
    assert!(
        matches!(
            &fitting,
            Some(Event::Finished { reason, .. }) if *reason != FinishReason::Error
        ),
        "{fitting:?}"
    );
}

#[test]
fn fcfs_policy_completes_text() {
    let out = batch_requests(2, 3);
    assert_eq!(out.len(), 3);
    for c in out.values() {
        assert!(c.finished);
        assert!(c.text > 0);
    }
}

/// Every logical cache reservation returns to the block pool after completion,
/// across text, image-only, and mixed requests, and no request remains running
/// or in flight.
#[test]
fn cache_resources_return_after_completion() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = scheduler::CommandSender::new(command_tx);

    let (stats, runner) = run(
        scheduler_over(&text(1, Some(EndOfText::after(5, 8)))),
        commands,
    );

    // Keep receivers alive — a dropped receiver is treated as a cancellation.
    let mut keep_alive = Vec::new();
    for i in 0..4 {
        let req = generation_request(
            RequestId(i as u64 + 1),
            text_input(vec![1, 2, 3, 4, 5]),
            SamplingParams::default(),
            16,
        );
        keep_alive.push(handle.submit(req).unwrap());
    }

    // Every request finishes, and the scheduler stops once it is idle.
    for events in &mut keep_alive {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !matches!(events.try_recv(), Ok(Event::Finished { .. }))
        {
            thread::sleep(Duration::from_micros(200));
        }
    }
    settle(&stats, |general| {
        general.running.load(Ordering::Relaxed) == 0
            && general.in_flight.load(Ordering::Relaxed) == 0
    });
    handle.shutdown();
    assert!(!runner.join().unwrap(), "the scheduler failed");

    assert_eq!(stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(stats.general.in_flight.load(Ordering::Relaxed), 0);
    assert_eq!(
        stats.prefix_cache.free_units.load(Ordering::Relaxed),
        stats.prefix_cache.num_units.load(Ordering::Relaxed)
    );
}

/// Dropping 128 event receivers at once cancels every request (dropping an
/// `EventRx` sends a cancel command), and all of them retire from the pending,
/// running, and in-flight accounting.
#[test]
fn cancellation_storm_retires_every_request() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = scheduler::CommandSender::new(command_tx);

    let (stats, runner) = run(
        scheduler_over(&text(1, Some(EndOfText::after(3, 8)))),
        commands,
    );
    let receivers = (1..=128)
        .map(|request_id| {
            handle
                .submit(generation_request(
                    RequestId(request_id),
                    text_input(vec![1, 2, 3]),
                    SamplingParams::default(),
                    8,
                ))
                .unwrap()
        })
        .collect::<Vec<_>>();
    drop(receivers);

    // The scheduler stops once it has read every cancellation the dropped
    // receivers sent, so the counters below are its last.
    handle.shutdown();
    assert!(!runner.join().unwrap(), "the scheduler failed");

    assert_eq!(stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(stats.general.pending.load(Ordering::Relaxed), 0);
    assert_eq!(stats.general.in_flight.load(Ordering::Relaxed), 0);
}

/// A client that never reads its events stalls its own request on output
/// capacity without holding execution: the concurrent request finishes, the
/// stalled request stays running with nothing in flight, and it retires once
/// its receiver is dropped.
#[test]
fn slow_client_releases_execution_slots_before_output_capacity_returns() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = scheduler::CommandSender::new(command_tx);

    let pool = text(1, None);
    let (stats, runner) = run(scheduler_over(&pool), commands);
    let slow_events = handle
        .submit(generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            256,
        ))
        .unwrap();
    let mut fast_events = handle
        .submit(generation_request(
            RequestId(2),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            8,
        ))
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut fast_finished = false;
    while Instant::now() < deadline && !fast_finished {
        while let Ok(event) = fast_events.try_recv() {
            fast_finished |= matches!(event, Event::Finished { .. });
        }
        thread::sleep(LOOK);
    }
    assert!(
        fast_finished,
        "the consuming client must complete independently"
    );

    // The loop stops making progress, which leaves the slow request parked on
    // output capacity: it runs, and nothing of it is in flight.
    stall(&pool);
    assert_eq!(
        stats.general.in_flight.load(Ordering::Relaxed),
        0,
        "an output-stalled request cannot retain an execution slot"
    );
    assert_eq!(stats.general.running.load(Ordering::Relaxed), 1);

    drop(slow_events);
    settle(&stats, |general| {
        general.running.load(Ordering::Relaxed) == 0
            && general.in_flight.load(Ordering::Relaxed) == 0
    });
    handle.shutdown();
    assert!(!runner.join().unwrap(), "the scheduler failed");
    assert_eq!(stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(stats.general.in_flight.load(Ordering::Relaxed), 0);
}

/// A worker failure that stops the engine still delivers `Finished { Aborted }`
/// to a running request whose event channel is full, after every event it
/// published before the failure. The failure is a batch the worker accepted
/// failing, as a lost device's does: a second request's, while the first is
/// parked on its output.
#[test]
fn engine_death_delivers_the_terminal_behind_a_full_event_channel() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(command_tx);

    let pool = text(1, None);
    let (stats, runner) = run(scheduler_over(&pool), commands);
    let mut events = handle
        .submit(generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            256,
        ))
        .unwrap();

    // The client reads nothing, so the request fills its event channel and
    // parks on output capacity with nothing in flight.
    stall(&pool);
    assert_eq!(stats.general.in_flight.load(Ordering::Relaxed), 0);
    assert_eq!(stats.general.running.load(Ordering::Relaxed), 1);

    pool.hold();
    let _second = handle
        .submit(generation_request(
            RequestId(2),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            4,
        ))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while pool.held() == 0 && Instant::now() < deadline {
        thread::sleep(LOOK);
    }
    assert_eq!(
        pool.fail_held("the worker was lost"),
        1,
        "the second request's batch runs"
    );
    assert!(runner.join().unwrap(), "a lost worker stops the engine");

    let mut received = Vec::new();
    loop {
        match events.try_recv() {
            Ok(event) => received.push(event),
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => panic!("the stopped engine left the event stream open"),
        }
    }
    assert!(
        received.len() > EVENT_BUFFER_CAPACITY,
        "received {} events, ending with {:?}; the request published more than its channel holds",
        received.len(),
        received.last(),
    );
    let text_tokens = received
        .iter()
        .filter(|event| matches!(event, Event::TextToken { .. }))
        .count();
    match received.last() {
        Some(Event::Finished {
            reason,
            completion_tokens,
            ..
        }) => {
            assert_eq!(*reason, FinishReason::Aborted);
            assert_eq!(*completion_tokens, text_tokens);
        }
        other => panic!("the stream did not end with its terminal event: {other:?}"),
    }
}

/// A scheduler takes scheduling limits in place within its workers' batch
/// and refuses limits beyond it, keeping its own. A change of policy orders
/// the requests already waiting by it.
#[test]
fn scheduler_takes_limits_in_place_within_its_workers_batch() {
    let base = SchedulerConfig {
        max_batch: proto::config::positive(32),
        ..Default::default()
    };
    let sched =
        support::fake::scheduler_over(text(1, None), base.clone(), Default::default()).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let refused = handle
        .configure(SchedulerConfig {
            max_running_requests: 1 << 30,
            ..base.clone()
        })
        .recv()
        .unwrap()
        .unwrap_err();
    assert!(refused.contains("max_running_requests"), "{refused}");
    // One request runs at a time.
    handle
        .configure(SchedulerConfig {
            max_running_requests: 1,
            ..base.clone()
        })
        .recv()
        .unwrap()
        .unwrap();

    let request = |id: u64, priority: i32, tokens: usize| GenerationRequest {
        priority,
        ..generation_request(
            RequestId(id),
            text_input(vec![1, 2, 3, 4, 5]),
            SamplingParams::default(),
            tokens,
        )
    };
    // The first request runs until it is aborted: its events are not read,
    // so it stops once its output window is full. The other two wait behind
    // it, in arrival order.
    let mut first = handle.submit(request(1, 0, 1024)).unwrap();
    let scheduled = |events: &mut scheduler::EventRx| {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            match events.try_recv() {
                Ok(Event::Scheduled { scheduled_at, .. }) => return scheduled_at,
                Ok(Event::Finished { reason, .. }) => panic!("finished unscheduled: {reason:?}"),
                Ok(_) => {}
                Err(TryRecvError::Empty) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("no scheduling event: {error:?}"),
            }
        }
    };
    scheduled(&mut first);
    let mut low = handle.submit(request(2, 5, 4)).unwrap();
    let mut high = handle.submit(request(3, 1, 4)).unwrap();
    handle
        .configure(SchedulerConfig {
            max_running_requests: 1,
            ..base.clone()
        })
        .recv()
        .unwrap()
        .unwrap();
    handle.abort(RequestId(1));
    let first_waiter = scheduled(&mut low);
    let second_waiter = scheduled(&mut high);
    // The serving program is `fcfs`, which admits by arrival and reads no
    // priority: the request that arrived first is served first, whatever its
    // priority states. A policy that orders by priority is a different program;
    // this case states that the limit changed in place and the queue moved on.
    assert!(
        first_waiter < second_waiter,
        "{first_waiter} {second_waiter}"
    );

    handle.shutdown();
    let _ = jh.join();
}

/// An engine whose only worker is lost stops admitting work: a batch it
/// accepted fails, its scheduler thread ends, and a submission after that is
/// refused rather than queued for a worker that will never answer.
#[test]
fn losing_every_worker_stops_request_admission() {
    let mut rank = support::text_rank();
    rank.supported_calls = vec![
        CallKind::Forward(ForwardMode::Prefill),
        CallKind::Forward(ForwardMode::Decode),
    ];
    let pool = Arc::new(support::text_pool(rank, 1, None));
    pool.hold();
    let scheduler = scheduler_over(&pool);
    let (tx, rx) = crossbeam_channel::unbounded();
    let commands = CommandSender::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));

    // The deployment's only worker is lost under the first request's batch.
    let _first = commands
        .submit(generation_request(
            RequestId(1),
            text_input(vec![7; 8]),
            SamplingParams::default(),
            4,
        ))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while pool.held() == 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        pool.fail_held("the worker was lost"),
        1,
        "the request's batch runs"
    );

    let died = scheduler_thread.join().expect("the scheduler thread ends");
    assert!(
        died,
        "an engine without execution workers kept accepting work"
    );
    // A stopped engine refuses submissions instead of queueing them.
    assert!(
        commands
            .submit(generation_request(
                RequestId(2),
                text_input(vec![7; 8]),
                SamplingParams::default(),
                4,
            ))
            .is_err(),
        "a stopped engine accepted another submission"
    );
}
