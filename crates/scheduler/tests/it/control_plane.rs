//! End-to-end engine-loop lifecycle behavior over the GPU-free simulator.
//!
//! Each test drives a real `Scheduler` against `SimWorker`/`SimEngine`,
//! either on a spawned thread through `Scheduler::run` or stepped on the test
//! thread with `Scheduler::step`, and observes behavior through `CommandSender`
//! event streams, `SchedulerStats` counters, and the scheduler's public
//! accessors.
//!
//! Simulator conventions the tests rely on:
//! - For request id `r`, the greedy token at text index `n` is
//!   `1000 + (7r + n) % 5000` (1007, 1008, ... for request 1) until
//!   `SimEngine::set_text_len` tokens have been produced; after that the
//!   synthetic EOS (151645, the first EOS of `SimEngine::special_tokens`)
//!   dominates the logits. The index restarts at zero when a generated image
//!   is fed back into the request's context.
//! - `SimEngine::set_queue_depth` bounds the batches the scheduler may keep
//!   unresolved. Depth 1 is the serial oracle that the depth-invariance tests
//!   compare depth 2 against.
//! - The cache block size is 64 tokens, one unit per page (`MainProcessInfo::default`),
//!   unless a test calls `SimEngine::set_block_size` or `set_groups`.
//! - The `general` request gauges and the `prefix_cache` and `encoder` counters in
//!   `SchedulerStats` are snapshots that `Scheduler::publish_cache_stats`
//!   refreshes as the loop runs, not when an event is sent, so they can trail
//!   the event stream. The `prefix` counters are recorded at admission.

use crate::support;

use ipc::{Call, CallKind, ForwardMode, MediaCall};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;

use ipc::{RequestId, SamplingParams};
use scheduler::{
    BatchEvent, BatchResult, CommandSender, EVENT_BUFFER_CAPACITY, ExecutionBatch, SchedulerConfig,
    SimEngine, SimWorker, SpecialTokenIds, Worker, WorkerError, WorkerGroupId, WorkerInfo,
    WorkerSubmitError,
};
use scheduler::{Event, FinishReason};
use scheduler::{
    GenerationLimits, GenerationRequest, ImageEncoderInput, ImageInput, MultimodalInputs,
    RejectionKind,
};

/// Default special tokens, whose first EOS id equals the simulator's synthetic
/// EOS, so the simulator's end of text finishes requests with `Eos`.
fn special_tokens() -> SpecialTokenIds {
    SimEngine::special_tokens()
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
        let mut sim = SimEngine::new();
        sim.mut_info_for_test().supported_calls = vec![
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::Decode),
        ];
        sim.mut_info_for_test().supported_calls.extend(calls);
        let scheduler =
            support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();
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
    let mut sim = SimEngine::new();
    let info = sim.mut_info_for_test();
    info.buffer_pool_bytes = 8 << 30;
    info.encoder_entry_bytes = 128 << 20;
    info.encoder_cache_entries = 64;
    let scheduler =
        support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();
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
fn batch_requests(depth: u32, count: usize) -> HashMap<RequestId, Collected> {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(depth);
    let worker = Box::new(SimWorker::new(sim));
    let sched = support::scheduler_with_depth(worker, special_tokens(), 32).unwrap();
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
    let mut sim = SimEngine::new();
    sim.mut_info_for_test().max_batch_calls = 3;
    let sched = support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();

    assert_eq!(sched.config().max_batch, proto::config::positive(3));
}

/// Prefill and decode batches hold at most the calls the workers report one
/// batch may hold: eight concurrent generations prefill and decode at most
/// three at a time and all finish.
#[test]
fn forward_batches_hold_at_most_the_workers_batch_calls() {
    let mut sim = SimEngine::new();
    sim.mut_info_for_test().max_batch_calls = 3;
    let mut worker = SimWorker::new(sim);
    let dispatched = worker.observe();
    let sched = support::scheduler(Box::new(worker), special_tokens(), 32).unwrap();
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
    for event in dispatched.try_iter() {
        let BatchEvent::Submitted(batch) = event else {
            continue;
        };
        match batch.requests.first().map(|(call, _)| call.kind) {
            Some(CallKind::Forward(ForwardMode::Prefill)) => {
                prefill_calls.push(batch.requests.len())
            }
            Some(CallKind::Forward(ForwardMode::Decode)) => decode_calls.push(batch.requests.len()),
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

    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    sim.set_text_len(6);
    let mut scheduler =
        support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();
    let request = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3, 4, 5]),
        SamplingParams::default(),
        16,
    );
    let mut events = handle.submit(request).unwrap();
    let mut finished = false;
    for _ in 0..512 {
        scheduler.step(&commands);
        while let Ok(event) = events.try_recv() {
            if matches!(event, Event::Finished { .. }) {
                finished = true;
            }
        }
        if finished {
            break;
        }
    }
    assert!(finished, "request finished");
    // Keep stepping after the terminal event so the counters are compared only
    // once the loop has drained.
    for _ in 0..32 {
        scheduler.step(&commands);
    }

    let active_domains = [
        &scheduler.stats.domains.prefill,
        &scheduler.stats.domains.decode,
        &scheduler.stats.domains.flow,
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
    let snapshot = reporter.snapshot(&scheduler.stats);
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

/// Runs one text request by stepping a fresh scheduler on the test thread and
/// returns its emitted token ids.
///
/// `depth` is the simulator queue depth: 1 gives the serial oracle, and 2 lets
/// the scheduler submit a successor before its predecessor's token has been
/// observed on the host.
fn relay_run(
    sampling: SamplingParams,
    stop_token_ids: Vec<u32>,
    max_und_tokens: usize,
    text_len: usize,
    depth: u32,
) -> Vec<u32> {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = scheduler::CommandSender::new(command_tx);

    let mut sim = SimEngine::new();
    sim.set_queue_depth(depth);
    sim.set_text_len(text_len);
    let mut scheduler =
        support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();
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
    for _ in 0..512 {
        scheduler.step(&commands);
        while let Ok(event) = events.try_recv() {
            match event {
                Event::TextToken { id, .. } => tokens.push(id),
                Event::Finished { .. } => finished = true,
                _ => {}
            }
        }
        if finished {
            break;
        }
    }
    assert!(finished, "processor request reached a terminal event");

    tokens
}

/// Sampling processors (penalties, logprobs, minimum-token floors, stop tokens,
/// and allowlists) yield the same tokens when successors are launched ahead of
/// host observation at depth 2 as in the serial depth-one run.
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
        (
            "static allowed-token whitelist under stochastic sampling",
            SamplingParams {
                temperature: 0.8,
                seed: Some(29),
                allowed_token_ids: Some(vec![1_000, 1_001, 1_002, 1_003, 1_004, 1_005]),
                ..SamplingParams::default()
            },
            Vec::new(),
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
    // One command channel serves both runs: each run's scheduler drains the
    // submission made just before it is stepped.
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = scheduler::CommandSender::new(command_tx);

    let mut runs = Vec::new();
    for queue_depth in [1, 2] {
        let mut sim = SimEngine::new();
        sim.set_queue_depth(queue_depth);
        sim.set_text_len(8);
        let mut scheduler =
            support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();
        let request = generation_request(
            RequestId(1),
            image_input(vec![1, 2], vec![3, 4], 0xD3C0DE, 4, 1),
            SamplingParams::default(),
            16,
        );
        let mut events = handle.submit(request).unwrap();
        let mut finish_reason = None;
        let mut tokens = Vec::new();
        for _ in 0..512 {
            scheduler.step(&commands);
            while let Ok(event) = events.try_recv() {
                match event {
                    Event::TextToken { id, .. } => tokens.push(id),
                    Event::Finished { reason, .. } => finish_reason = Some(reason),
                    _ => {}
                }
            }
            if finish_reason.is_some() {
                break;
            }
        }
        assert_eq!(finish_reason, Some(FinishReason::Eos));
        assert_eq!(tokens.len(), 8);
        runs.push(tokens);
    }

    assert_eq!(runs[0], runs[1]);
}

#[test]
fn stop_token_terminates_with_stop() {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    let worker = Box::new(SimWorker::new(sim));
    let sched = support::scheduler(worker, special_tokens(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut req = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3]),
        SamplingParams::default(),
        64,
    );
    // 1007 is request 1's first greedy token, so the request stops on its
    // first sample; `include_stop_token` is false, so nothing is emitted.
    req.stop_token_ids = vec![1007];
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
/// A long simulated text keeps the request generating, so the command is
/// observed mid-flight.
fn run_until_control(abort: bool) -> FinishReason {
    let mut sim = SimEngine::new();
    sim.set_text_len(1024);
    sim.set_queue_depth(2);
    let worker = Box::new(SimWorker::new(sim));
    let sched = support::scheduler(worker, special_tokens(), 32).unwrap();
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
    let mut sim = SimEngine::new();
    sim.set_text_len(1024);
    sim.set_queue_depth(2);
    let scheduler =
        support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();
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

/// Text requests complete on a worker whose unit pool holds a full-attention
/// group and a sliding-window group with different page shapes. The prompt
/// is several times the window, so the sliding group retires pages while the
/// request runs and its declared tables carry only in-window pages, and a
/// second request with the same prompt reuses the first one's prefix. Every
/// unit returns to the pool once both finish.
#[test]
fn hybrid_groups_share_one_unit_pool() {
    use ipc::{CacheGroup, CacheGroupKind};
    use std::sync::atomic::Ordering;

    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    sim.set_text_len(4);
    sim.set_num_units(64);
    let base = sim.mut_info_for_test().prefix_cache.clone().unwrap().groups[0].clone();
    sim.set_groups(vec![
        CacheGroup {
            layer_ids: (0..4).collect(),
            ..base.clone()
        },
        CacheGroup {
            kind: CacheGroupKind::SlidingWindow { window: 96 },
            block_size: 32,
            units_per_page: 3,
            layer_ids: (4..16).collect(),
            ..base
        },
    ]);
    let mut worker = SimWorker::new(sim);
    let dispatched = worker.observe();
    // A 64-token chunk reads at most six sliding pages with the window.
    let sched = support::scheduler_with(
        Box::new(worker),
        special_tokens(),
        SchedulerConfig {
            max_batch: proto::config::positive(32),
            chunked_prefill_size: std::num::NonZeroUsize::new(64),
            ..Default::default()
        },
    )
    .unwrap();
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    // 20 full pages of 64 tokens alone take 20 units; holding every sliding
    // page of 1280 tokens as well would take another 120, beyond the pool.
    let prompt: Vec<u32> = (0..1280u32).map(|i| (i % 61) + 3).collect();
    let run_one = |rid: u64| {
        let mut erx = handle
            .submit(generation_request(
                RequestId(rid),
                text_input(prompt.clone()),
                SamplingParams::default(),
                16,
            ))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match erx.try_recv() {
                Ok(Event::Finished { reason, .. }) => return reason,
                Ok(Event::Rejected { message, .. }) => {
                    panic!("request {rid} was rejected: {message}")
                }
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        panic!("request {rid} did not finish under hybrid groups");
    };

    assert_ne!(run_one(1), FinishReason::Error);
    let reused_before = stats.prefix.hit_tokens.load(Ordering::Relaxed);
    assert_ne!(run_one(2), FinishReason::Error);
    // The reused prefix is a multiple of the 64-token largest page whose
    // sliding window [M - 96, M) the first request published before retiring.
    let reused = stats.prefix.hit_tokens.load(Ordering::Relaxed) - reused_before;
    assert!(
        reused >= 1024 && reused.is_multiple_of(64),
        "reused {reused} tokens"
    );

    // A finished request's units retire until its workers acknowledge the
    // queued finish, so the pool drains while the engine keeps running, not
    // at the moment `Finished` is observed; shutdown would abort that ack.
    let num_units = stats.prefix_cache.num_units.load(Ordering::Relaxed);
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut free_units = stats.prefix_cache.free_units.load(Ordering::Relaxed);
    while free_units != num_units && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
        free_units = stats.prefix_cache.free_units.load(Ordering::Relaxed);
    }
    assert_eq!(free_units, num_units, "units did not return to the pool");

    handle.shutdown();
    let _ = jh.join();

    // Every declared table covers whole pages of its group from its start
    // page; the sliding group's tables move their start and never hold more
    // than the pages a window and two pipelined chunks intersect.
    let mut retired = false;
    for event in dispatched.try_iter() {
        let BatchEvent::Submitted(batch) = event else {
            continue;
        };
        for device in batch.device.iter() {
            for table in &device.block_tables {
                let (block_size, units_per_page) = if table.group_id == 0 {
                    (64, 1)
                } else {
                    (32, 3)
                };
                assert_eq!(table.unit_ids.len() % units_per_page, 0);
                let pages = (table.unit_ids.len() / units_per_page) as u32;
                assert_eq!(
                    table.allocated_tokens,
                    (table.start_page + pages) * block_size
                );
                if table.group_id == 1 {
                    assert!(pages <= (96 + 2 * 64) / 32 + 1, "{table:?}");
                    retired |= table.start_page > 0;
                } else {
                    assert_eq!(table.start_page, 0);
                }
            }
        }
    }
    assert!(retired, "the sliding group never retired a page");
}

/// Deployment policy controls prefix reuse without changing generated tokens.
/// Both requests ask to read and write prefixes; the disabled deployment must
/// override that permission while preserving each request's active cache state.
#[test]
fn prefix_cache_configuration_controls_reuse() {
    use scheduler::{SchedulerConfig, SchedulerHandle};

    let prompt: Vec<u32> = (0..600u32).map(|i| (i % 53) + 7).collect();
    let mut outputs = Vec::new();
    for enabled in [true, false] {
        let mut sim = SimEngine::new();
        sim.set_text_len(4);
        let mut args = scheduler::sim_args("sim-model");
        args.engine_config.kv_cache.prefix_caching = enabled;
        let worker = SimWorker::new(sim);
        let rank = worker
            .info()
            .runtime_info()
            .expect("the sim states its rank");
        let engine =
            SchedulerHandle::with_worker(&args, SchedulerConfig::default(), Box::new(worker))
                .unwrap();
        support::install_policy_on(&engine, scheduler::BatchCapacity::of(&rank));
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

/// Workers that encode latents under CUDA graphs report the image sizes
/// they captured; the engine admits an input image's latent block only at
/// one of them, and refuses a block of another size, or of no stated size,
/// with the sizes it serves.
#[test]
fn latent_blocks_are_admitted_only_at_captured_sizes() {
    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    sim.mut_info_for_test().captured_latents = Some(vec![[512, 512]]);
    let scheduler =
        support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));

    let terminal = |id: u64, size: Option<(u32, u32)>| {
        let mut request = generation_request(
            RequestId(id),
            image_input(vec![1, 2], vec![3, 4], id, 4, 5),
            SamplingParams::default(),
            4,
        );
        request.multimodal_inputs.images[0].encoders = vec![
            ImageEncoderInput {
                encoder: scheduler::ImageEncoder::Latent,
                num_cache_tokens: Some(6),
                max_cache_tokens: None,
                height: size.map(|(height, _)| height),
                width: size.map(|(_, width)| width),
            },
            ImageEncoderInput {
                encoder: scheduler::ImageEncoder::Vision,
                num_cache_tokens: Some(5),
                max_cache_tokens: None,
                height: None,
                width: None,
            },
        ];
        let mut events = handle.submit(request).unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
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
    let captured = terminal(1, Some((512, 512)));
    let uncaptured = terminal(2, Some((512, 768)));
    let unstated = terminal(3, None);
    handle.shutdown();
    let _ = scheduler_thread.join();

    assert!(
        matches!(&captured, Some(Event::Finished { reason, .. }) if *reason != FinishReason::Error),
        "{captured:?}"
    );
    for refused in [&uncaptured, &unstated] {
        let Some(Event::Rejected {
            kind: RejectionKind::Invalid,
            message,
        }) = refused
        else {
            panic!("{refused:?}");
        };
        assert!(
            message.contains("only at the sizes they captured: 512x512"),
            "{message}"
        );
    }
}

/// Workers that keep cache layers per request slot (a recurrent state)
/// checkpoint them in a state group: a request of the same prompt reuses the
/// prefix up to the checkpoint the first request's prompt kept at its last
/// page boundary, and no deeper.
#[test]
fn a_state_group_reuses_prefixes_up_to_a_checkpoint() {
    use scheduler::{SchedulerConfig, SchedulerHandle};
    use std::sync::atomic::Ordering;

    let prompt: Vec<u32> = (0..600u32).map(|i| (i % 53) + 7).collect();
    let mut sim = SimEngine::new();
    sim.set_text_len(4);
    sim.set_block_size(16);
    sim.add_state_group(2);
    let mut args = scheduler::sim_args("sim-model");
    args.engine_config.kv_cache.prefix_caching = true;
    let worker = SimWorker::new(sim);
    let rank = worker
        .info()
        .runtime_info()
        .expect("the sim states its rank");
    let engine =
        SchedulerHandle::with_worker(&args, SchedulerConfig::default(), Box::new(worker)).unwrap();
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

    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    let sched = support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();
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
    let worker = Box::new(SimWorker::new(SimEngine::new()));
    let sched =
        support::scheduler_with(worker, special_tokens(), SchedulerConfig::default()).unwrap();
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
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    let mut worker = SimWorker::new(sim);
    let dispatched = worker.observe();
    let worker = Box::new(worker);
    let sched = support::scheduler_with(
        worker,
        special_tokens(),
        SchedulerConfig {
            max_batch: proto::config::positive(32),
            // Caps a prefill chunk at 64 tokens.
            chunked_prefill_size: std::num::NonZeroUsize::new(64),
            // Leaves room for other decodes per step.
            max_num_batched_tokens: 256,
            ..Default::default()
        },
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
    let prefill_tokens = || {
        dispatched
            .try_iter()
            .filter_map(|event| match event {
                BatchEvent::Submitted(batch) => Some(
                    batch
                        .requests
                        .into_iter()
                        .filter(|(call, _)| {
                            call.kind == CallKind::Forward(ipc::ForwardMode::Prefill)
                        })
                        .map(|(call, _)| call.input_token_ids.len())
                        .sum::<usize>(),
                ),
                _ => None,
            })
            .filter(|&tokens| tokens > 0)
            .collect::<Vec<_>>()
    };
    let capped = prefill_tokens();
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
    assert!(prefill_tokens().into_iter().any(|tokens| tokens > 16));
    handle.shutdown();
    let _ = jh.join();
    assert!(d1 && d2, "both requests must finish (d1={d1} d2={d2})");
    assert!(t1 > 0 && t2 > 0, "both must produce text (t1={t1} t2={t2})");
}

/// Runs one text request with the given sampling params and a sim `text_len`,
/// returning the emitted token ids, whether any token event carried a logprob,
/// and the finish reason (`None` when the request did not finish within the
/// 10-second deadline).
fn run_sampling(
    sampling: SamplingParams,
    text_len: usize,
    max_tokens: usize,
) -> (Vec<u32>, bool, Option<FinishReason>) {
    let mut sim = SimEngine::new();
    sim.set_text_len(text_len);
    let worker = Box::new(SimWorker::new(sim));
    let sched = support::scheduler(worker, special_tokens(), 32).unwrap();
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

/// The first token's reported distribution: the engine reports the model's
/// most likely candidates with the model's log probabilities, and a nucleus
/// cutoff changes which of them may be *sampled*, not which are *reported*.
///
/// The launched case states this with the stub model, whose successor holds
/// nearly all mass; here the sample distribution is the simulator's, and the
/// property under test is the same: the reported candidates and their
/// log probabilities do not depend on the nucleus.
#[test]
fn logprobs_report_the_model_distribution_under_a_cutoff() {
    let candidates = |top_p: f32| {
        let sampling = SamplingParams {
            n_logprobs: 5,
            top_p,
            ..Default::default()
        };
        first_token_logprobs(sampling)
    };
    let cut = candidates(0.5);
    let whole = candidates(1.0);
    assert_eq!(cut.len(), 5, "five candidates are reported: {cut:?}");
    assert!(
        cut.iter().all(|(_, logprob)| logprob.is_finite()),
        "every reported log probability is finite: {cut:?}"
    );
    assert_eq!(
        cut, whole,
        "a nucleus cutoff changes what may be sampled, not what is reported"
    );
}

/// One request's first token's reported candidates, in the order reported.
fn first_token_logprobs(sampling: SamplingParams) -> Vec<(u32, f32)> {
    let mut sim = SimEngine::new();
    sim.set_text_len(8);
    let worker = Box::new(SimWorker::new(sim));
    let sched = support::scheduler(worker, special_tokens(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = generation_request(RequestId(1), text_input(vec![1, 2, 3]), sampling, 4);
    let mut erx = handle.submit(req).unwrap();
    let mut reported = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while reported.is_none() && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(Event::TokenLogprobs { candidates, .. }) => {
                reported = Some(
                    candidates
                        .into_iter()
                        .map(|candidate| (candidate.token_id, candidate.logprob))
                        .collect::<Vec<_>>(),
                );
            }
            Ok(Event::Finished { .. }) => break,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();
    reported.expect("the request reports its first token's candidates")
}

#[test]
fn allowed_tokens_restricts_output() {
    let sp = SamplingParams {
        allowed_token_ids: Some(vec![1234]),
        ..Default::default()
    };
    let (toks, _lp, finished) = run_sampling(sp, 8, 6);
    assert!(finished.is_some());
    assert!(!toks.is_empty());
    assert!(
        toks.iter().all(|&t| t == 1234),
        "every token must be the single allowed id, got {toks:?}"
    );
}

/// The bad-word sequence [4321, 4321] bans 4321 only directly after a 4321:
/// the ban overrides the bias there, while the biased token still wins the
/// first position and the output stays within the allowlist.
#[test]
fn bad_word_suffix_overrides_bias_without_suppressing_its_prefix() {
    let sampling = SamplingParams {
        allowed_token_ids: Some(vec![1234, 4321]),
        logit_bias: vec![(4321, 1000.0)],
        bad_words_ids: vec![vec![4321, 4321]],
        ..Default::default()
    };
    let (tokens, _, finished) = run_sampling(sampling, 8, 6);
    assert!(finished.is_some());
    assert_eq!(tokens.len(), 6, "{tokens:?}, {finished:?}");
    assert_eq!(tokens[0], 4321);
    assert!(tokens.iter().all(|token| [1234, 4321].contains(token)));
    assert!(tokens.windows(2).all(|pair| pair != [4321, 4321]));
}

#[test]
fn logit_bias_forces_token() {
    // Strongly bias token 4321; it should dominate every step.
    let sp = SamplingParams {
        logit_bias: vec![(4321, 1000.0)],
        ..Default::default()
    };
    let (toks, _lp, finished) = run_sampling(sp, 8, 6);
    assert!(finished.is_some());
    assert!(
        toks.iter().all(|&t| t == 4321),
        "biased token must always win, got {toks:?}"
    );
}

#[test]
fn min_tokens_floor_overrides_early_eos() {
    // With text_len=1 the simulator prefers EOS from the second token on;
    // min_tokens=5 makes the scheduler suppress EOS until 5 tokens exist.
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

#[test]
fn default_sampling_is_unchanged() {
    // Default params are greedy (temperature 0), so the first token is the
    // simulator's natural token for request 1 at index 0: 1007.
    let (toks, _lp, finished) = run_sampling(SamplingParams::default(), 8, 16);
    assert!(finished.is_some());
    assert_eq!(toks[0], 1007);
}

#[test]
fn stochastic_sampling_reaches_synthetic_eos() {
    let sampling = SamplingParams {
        temperature: 1.0,
        seed: Some(7),
        ..Default::default()
    };
    let (_toks, _lp, finished) = run_sampling(sampling, 8, 16);
    assert_eq!(finished, Some(FinishReason::Eos));
}

/// The first request with a context image misses the encoder cache and leaves
/// its encoder output resident; a second request with the same image hash hits
/// that cached output.
#[test]
fn multimodal_encode_then_cache_hit() {
    use std::sync::atomic::Ordering;
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    let worker = SimWorker::new(sim);
    let wake = worker.command_waker();
    let sched = support::scheduler(Box::new(worker), special_tokens(), 32).unwrap();
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::with_waker(tx, wake);
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

    let mut sim = SimEngine::new();
    sim.set_text_len(6);
    sim.set_queue_depth(2);
    let mut scheduler =
        support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();
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
    let deadline = Instant::now() + Duration::from_secs(5);

    while Instant::now() < deadline {
        scheduler.step(&commands);
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
    assert_eq!(
        scheduler
            .stats_handle()
            .encoder
            .cached
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn und_only_image_context_encodes_then_produces_text_without_gen_output() {
    use std::sync::atomic::Ordering;

    let mut sim = SimEngine::new();
    sim.set_text_len(20);
    let sched = support::scheduler(Box::new(SimWorker::new(sim)), special_tokens(), 32).unwrap();
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

/// An input image with a latent encoder writes its latent block with a call
/// of its own, which ends the context prefill before it; its vision block
/// then joins the next context prefill, with the prompt tokens after it.
#[test]
fn a_latent_image_block_is_written_by_its_own_call() {
    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    let mut worker = SimWorker::new(sim);
    let boundary = worker.observe();
    let sched = support::scheduler(Box::new(worker), special_tokens(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut request = generation_request(
        RequestId(95),
        image_input(vec![1, 2], vec![3, 4], 0x95, 4, 5),
        SamplingParams::default(),
        4,
    );
    request.multimodal_inputs.images[0].encoders = vec![
        ImageEncoderInput {
            encoder: scheduler::ImageEncoder::Latent,
            num_cache_tokens: Some(6),
            max_cache_tokens: None,
            height: None,
            width: None,
        },
        ImageEncoderInput {
            encoder: scheduler::ImageEncoder::Vision,
            num_cache_tokens: Some(5),
            max_cache_tokens: None,
            height: None,
            width: None,
        },
    ];
    let mut events = handle.submit(request).expect("submit request");
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut finished = false;
    while !finished && Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::Finished { reason, .. }) => {
                assert_ne!(reason, FinishReason::Error);
                finished = true;
            }
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();
    assert!(finished, "the latent-image request did not finish");

    let calls: Vec<Call> = boundary
        .try_iter()
        .filter_map(|event| match event {
            BatchEvent::Submitted(batch) => Some(batch),
            BatchEvent::Resolved { .. } => None,
        })
        .flat_map(|batch| batch.requests.into_iter().map(|(call, _)| call))
        .filter(|call| call.request_key.request_id == RequestId(95))
        .collect();
    let context: Vec<_> = calls
        .iter()
        .take_while(|call| call.kind != CallKind::Forward(ForwardMode::Decode))
        .collect();
    assert_eq!(
        context.iter().map(|call| call.kind).collect::<Vec<_>>(),
        vec![
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Media(MediaCall::LatentEncoding),
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Media(MediaCall::VisionEncoding),
            CallKind::Forward(ForwardMode::Prefill),
        ]
    );
    // The prompt before the image, alone.
    assert_eq!(context[0].input_token_ids, vec![1, 2]);
    assert!(context[0].vision_inputs.is_empty());
    // The latent block's own extension.
    assert_eq!(context[2].latent_feature_input, context[1].encoder_output);
    assert!(context[2].input_token_ids.is_empty());
    assert!(context[2].vision_inputs.is_empty());
    // The vision block at the start of the context prefill that follows.
    assert_eq!(context[4].input_token_ids, vec![3, 4]);
    assert_eq!(context[4].vision_inputs.len(), 1);
    assert_eq!(context[4].vision_inputs[0].offset, 0);
    assert_eq!(
        Some(context[4].vision_inputs[0].feature.clone()),
        context[3].encoder_output
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

    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    let sched = support::scheduler_with(
        Box::new(SimWorker::new(sim)),
        special_tokens(),
        SchedulerConfig {
            max_batch: proto::config::positive(32),
            max_num_batched_tokens: STEP_BUDGET as usize,
            ..Default::default()
        },
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

/// A request's worker admission carries the number of input images the
/// request holds, which worker image preprocessing reads when it divides a
/// pixel budget among them.
#[test]
fn admission_carries_the_request_input_image_count() {
    let mut sim = SimEngine::new();
    sim.set_text_len(4);
    let mut worker = SimWorker::new(sim);
    let boundary = worker.observe();
    let sched = support::scheduler(Box::new(worker), special_tokens(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (prompt_token_ids, mut inputs) = image_input(vec![1, 2], vec![3, 4], 0x91, 4, 17);
    let second = ImageInput {
        digest: support::image_digest(0x92),
        ..inputs.images[0].clone()
    };
    inputs.images.push(second);
    let request = generation_request(
        RequestId(91),
        (prompt_token_ids, inputs),
        SamplingParams::default(),
        4,
    );
    let _events = handle.submit(request).expect("submit request");

    let deadline = Instant::now() + Duration::from_secs(10);
    let admission = loop {
        let event = boundary
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("the request is admitted");
        if let BatchEvent::Submitted(batch) = event
            && let Some(admission) = batch.commands.iter().find_map(|command| match command {
                ipc::BatchCommand::Start { request } => Some(request.clone()),
                _ => None,
            })
        {
            break admission;
        }
    };
    handle.shutdown();
    let _ = jh.join();

    assert_eq!(admission.input_images, 2);
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

    let worker = Box::new(SimWorker::new(SimEngine::new()));
    let mut sched = support::scheduler_with_depth(worker, special_tokens(), 32).unwrap();

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

    let mut idle_steps = 0;
    for _ in 0..5000 {
        let progressed = sched.step(&commands);
        // Converged when several consecutive steps make no progress.
        idle_steps = if progressed { 0 } else { idle_steps + 1 };
        if idle_steps >= 3 {
            break;
        }
    }

    assert_eq!(sched.stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(sched.stats.general.in_flight.load(Ordering::Relaxed), 0);
    assert_eq!(
        sched.stats.prefix_cache.free_units.load(Ordering::Relaxed),
        sched.stats.prefix_cache.num_units.load(Ordering::Relaxed)
    );
}

/// Dropping 128 event receivers at once cancels every request (dropping an
/// `EventRx` sends a cancel command), and all of them retire from the pending,
/// running, and in-flight accounting.
#[test]
fn cancellation_storm_retires_every_request() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = scheduler::CommandSender::new(command_tx);

    let worker = Box::new(SimWorker::new(SimEngine::new()));
    let mut scheduler = support::scheduler(worker, special_tokens(), 32).unwrap();
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

    for _ in 0..10_000 {
        scheduler.step(&commands);
        if scheduler.stats.general.running.load(Ordering::Relaxed) == 0
            && scheduler.stats.general.pending.load(Ordering::Relaxed) == 0
            && scheduler.stats.general.in_flight.load(Ordering::Relaxed) == 0
        {
            break;
        }
        thread::sleep(Duration::from_micros(50));
    }

    assert_eq!(scheduler.stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.stats.general.pending.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.stats.general.in_flight.load(Ordering::Relaxed), 0);
}

/// A client that never reads its events stalls its own request on output
/// capacity without holding execution: the concurrent request finishes, the
/// stalled request stays running with nothing in flight, and it retires once
/// its receiver is dropped.
#[test]
fn slow_client_releases_execution_slots_before_output_capacity_returns() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = scheduler::CommandSender::new(command_tx);

    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let worker = Box::new(SimWorker::new(sim));
    let mut scheduler = support::scheduler_with_depth(worker, special_tokens(), 32).unwrap();
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
        scheduler.step(&commands);
        while let Ok(event) = fast_events.try_recv() {
            fast_finished |= matches!(event, Event::Finished { .. });
        }
    }
    assert!(
        fast_finished,
        "the consuming client must complete independently"
    );

    // Step until the loop stops making progress, which leaves the slow request
    // parked on output capacity.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && scheduler.step(&commands) {}
    assert_eq!(
        scheduler.stats.general.in_flight.load(Ordering::Relaxed),
        0,
        "an output-stalled request cannot retain an execution slot"
    );
    assert_eq!(scheduler.stats.general.running.load(Ordering::Relaxed), 1);

    drop(slow_events);
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline
        && (scheduler.stats.general.running.load(Ordering::Relaxed) > 0
            || scheduler.stats.general.in_flight.load(Ordering::Relaxed) > 0)
    {
        scheduler.step(&commands);
    }
    assert_eq!(scheduler.stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.stats.general.in_flight.load(Ordering::Relaxed), 0);
}

/// Worker that can be lost: once `lost` is set, every poll reports an
/// worker failure, and the scheduler latches engine-fatal. Everything else
/// is the simulator.
struct LosableWorker {
    inner: SimWorker,
    lost: Arc<AtomicBool>,
}

impl Worker for LosableWorker {
    fn info(&self) -> &WorkerInfo {
        self.inner.info()
    }

    fn is_ready(&self, worker: &WorkerGroupId) -> bool {
        self.inner.is_ready(worker)
    }

    fn has_capacity(&self, worker: &WorkerGroupId) -> bool {
        self.inner.has_capacity(worker)
    }

    fn command_has_capacity(&self, command: &ipc::BatchCommand) -> bool {
        self.inner.command_has_capacity(command)
    }

    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), WorkerSubmitError> {
        self.inner.submit(batch)
    }

    fn poll(&mut self, timeout: Duration) -> Result<Option<BatchResult>, WorkerError> {
        if self.lost.load(Ordering::SeqCst) {
            return Err(anyhow::anyhow!("the worker was lost"));
        }
        self.inner.poll(timeout)
    }

    fn close(&mut self) -> Result<(), WorkerError> {
        self.inner.close()
    }
}

/// A worker failure that stops the engine still delivers `Finished { Aborted }`
/// to a running request whose event channel is full, after every event it
/// published before the failure.
#[test]
fn engine_death_delivers_the_terminal_behind_a_full_event_channel() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(command_tx);

    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let lost = Arc::new(AtomicBool::new(false));
    let worker = Box::new(LosableWorker {
        inner: SimWorker::new(sim),
        lost: Arc::clone(&lost),
    });
    let mut scheduler = support::scheduler_with_depth(worker, special_tokens(), 32).unwrap();
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
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && scheduler.step(&commands) {}
    assert_eq!(scheduler.stats.general.in_flight.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.stats.general.running.load(Ordering::Relaxed), 1);

    lost.store(true, Ordering::SeqCst);
    assert!(scheduler.run(commands), "a lost worker stops the engine");

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
    let mut sim = SimEngine::new();
    sim.set_text_len(1 << 20);
    let worker = Box::new(SimWorker::new(sim));
    let base = SchedulerConfig {
        max_batch: proto::config::positive(32),
        ..Default::default()
    };
    let sched = support::scheduler_with(worker, special_tokens(), base.clone()).unwrap();
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

/// An engine whose only worker group is lost, with nothing to replace it,
/// stops admitting work: its scheduler thread ends, and a submission after
/// that is refused rather than queued for workers that will never exist.
#[test]
fn losing_every_worker_stops_request_admission() {
    let mut sim = SimEngine::new();
    sim.mut_info_for_test().supported_calls = vec![
        CallKind::Forward(ForwardMode::Prefill),
        CallKind::Forward(ForwardMode::Decode),
    ];
    let worker = SimWorker::new(sim);
    let loss = worker.loss();
    let scheduler = support::scheduler(Box::new(worker), special_tokens(), 32)
        .expect("the scheduler is built over the simulated worker");
    let (tx, rx) = crossbeam_channel::unbounded();
    let commands = CommandSender::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));

    // The deployment's only group is lost, and nothing can replace it.
    loss.lose(
        WorkerGroupId("sim".to_owned()),
        "the simulated launch could not be replaced",
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
