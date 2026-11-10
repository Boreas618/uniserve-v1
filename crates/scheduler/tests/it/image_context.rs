//! Input images in a token request's context on the GPU-free simulator.
//!
//! Every input image is encoded before the context prefill that reaches it,
//! and that prefill writes the image's vision block whole between the prompt
//! tokens around it. The requests here are generated in block-diffusion
//! canvases, whose prompt prefill samples nothing, so the context prefills
//! are the only prefills a request runs: the simulator's synthetic text is
//! shorter than one canvas, and the first block, which holds EOS, ends the
//! request uncommitted. Tests observe calls at the worker boundary, the
//! protocol the worker consumes.

use crate::support;

use std::collections::{BTreeMap, HashMap};
use std::thread;
use std::time::{Duration, Instant};

use ipc::{Call, CallKind, ComponentInfo, ForwardMode, MediaCall};
use ipc::{CanvasSampling, ImageParams, RequestId, SamplingParams};
use scheduler::{
    BatchEvent, BlockConfig, CommandSender, EventRx, ExecutionBatch, SchedulerConfig, SimEngine,
    SimWorker, sim_text_token,
};
use scheduler::{
    CachePolicy, Event, FinishReason, GenerationConstraint, GenerationRequest, ImageEncoderInput,
    ImageGenerationConfig, ImageInput, MultimodalInputs,
};

/// Tokens of one canvas block.
const CANVAS: u32 = 16;
/// Synthetic text tokens before EOS: fewer than one block holds.
const TEXT_LEN: usize = 4;
/// The block-diffusion sampling the deployment is launched with.
const SAMPLING: CanvasSampling = CanvasSampling {
    canvas_length: CANVAS,
    max_steps: 4,
    entropy_bound: 0.1,
    t_min: 0.4,
    t_max: 0.8,
    confidence_threshold: 0.005,
    stability_threshold: 1,
};

/// A scheduler configuration of the deployment that generates `SAMPLING`
/// canvases.
fn config() -> SchedulerConfig {
    SchedulerConfig {
        canvas: Some(SAMPLING),
        ..SchedulerConfig::default()
    }
}

/// A block-diffusion worker: prompt prefill, canvas denoising and vision
/// encoding, no token decode, with 16-token cache pages.
fn image_worker() -> SimEngine {
    let mut sim = SimEngine::new();
    sim.mut_info_for_test().supported_calls = vec![
        CallKind::Forward(ForwardMode::Prefill),
        CallKind::Forward(ForwardMode::TokenDenoising),
        CallKind::Media(MediaCall::VisionEncoding),
    ];
    sim.set_block_size(16);
    sim.set_text_len(TEXT_LEN);
    sim.set_canvas_sampling(SAMPLING);
    sim
}

fn prompt(length: u32, first: u32) -> Vec<u32> {
    (0..length).map(|index| first + index).collect()
}

/// A canvas-generating request over `prompt_token_ids`, with no images yet.
fn canvas_request(request_id: u64, prompt_token_ids: Vec<u32>) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(request_id),
        prompt_token_ids,
        negative_prompt_token_ids: Vec::new(),
        multimodal_inputs: MultimodalInputs::default(),
        constraint: GenerationConstraint::Default,
        sampling: SamplingParams {
            seed: Some(7),
            ..SamplingParams::default()
        },
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: 1_000,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy::default(),
        include_stop_token: false,
        image_generation: ImageGenerationConfig::default(),
        generated_token_ids: Vec::new(),
        adapter: 0,
        generated_images: Vec::new(),
        trajectory: Default::default(),
    }
}

/// A vision input image of `tokens` cache tokens and as many positions at
/// prompt position `position`.
fn vision_image(id: u64, position: u32, tokens: u32) -> ImageInput {
    ImageInput {
        digest: crate::support::image_digest(id),
        b64: "aW1hZ2U=".into(),
        position,
        num_positions: tokens,
        encoders: vec![ImageEncoderInput {
            encoder: scheduler::ImageEncoder::Vision,
            num_cache_tokens: Some(tokens),
            max_cache_tokens: None,
            height: None,
            width: None,
        }],
    }
}

/// A block-diffusion scheduler running on its own thread, with the batches
/// its worker accepts.
struct Running {
    handle: CommandSender,
    batches: crossbeam_channel::Receiver<BatchEvent>,
    thread: thread::JoinHandle<bool>,
}

impl Running {
    fn start(sim: SimEngine, config: SchedulerConfig) -> Self {
        let mut worker = SimWorker::new(sim);
        let batches = worker.observe();
        let scheduler =
            support::scheduler_with(Box::new(worker), SimEngine::special_tokens(), config).unwrap();
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = CommandSender::new(tx);
        let thread = thread::spawn(move || scheduler.run(rx));
        Self {
            handle,
            batches,
            thread,
        }
    }

    fn submit(&self, request: GenerationRequest) -> EventRx {
        self.handle.submit(request).unwrap()
    }

    /// Stops the scheduler and returns every batch it submitted, in order.
    fn stop(self) -> Vec<ExecutionBatch> {
        self.stop_with_resolutions()
            .into_iter()
            .filter_map(|event| match event {
                BatchEvent::Submitted(batch) => Some(batch),
                BatchEvent::Resolved { .. } => None,
            })
            .collect()
    }

    /// Stops the scheduler and returns every submission and resolution the
    /// worker saw, in order.
    fn stop_with_resolutions(self) -> Vec<BatchEvent> {
        self.handle.shutdown();
        assert!(!self.thread.join().unwrap(), "the scheduler failed");
        self.batches.try_iter().collect()
    }
}

/// Collects one request's events through its terminal event.
fn events(mut receiver: EventRx) -> Vec<Event> {
    let mut events = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(20);
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

/// Asserts that `request` published its synthetic text as one block and
/// ended on the EOS after it.
fn assert_generated(events: &[Event], request: &GenerationRequest) {
    let blocks: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            Event::TextTokens { ids } => Some(ids.clone()),
            _ => None,
        })
        .collect();
    let text: Vec<_> = (0..TEXT_LEN)
        .map(|index| sim_text_token(request.request_id, index))
        .collect();
    assert_eq!(blocks, vec![text]);
    match events.last() {
        Some(Event::Finished {
            reason,
            completion_tokens,
            ..
        }) => {
            assert_eq!(*reason, FinishReason::Eos);
            assert_eq!(*completion_tokens, TEXT_LEN + 1);
        }
        other => panic!("request ended with {other:?}"),
    }
}

/// The calls of one request in submission order.
fn calls_of(batches: &[ExecutionBatch], request: u64) -> Vec<(Call, crate::support::CallRows)> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .requests
                .iter()
                .enumerate()
                .map(move |(index, (call, _))| (call, crate::support::rows_of(batch, index)))
        })
        .filter(|(call, _)| call.request_key.request_id == RequestId(request))
        .map(|(call, rows)| (call.clone(), rows))
        .collect()
}

/// The context prefills of one request, with their forward rows, in
/// submission order.
fn context_prefills(
    batches: &[ExecutionBatch],
    request: u64,
) -> Vec<(Call, crate::support::CallRows)> {
    calls_of(batches, request)
        .into_iter()
        .filter(|(call, _)| call.kind == CallKind::Forward(ForwardMode::Prefill))
        .collect()
}

/// The position in `history` of the first submission holding a call of
/// `kind`.
fn first_submission(history: &[BatchEvent], kind: CallKind) -> usize {
    history
        .iter()
        .position(|event| match event {
            BatchEvent::Submitted(batch) => {
                batch.requests.iter().any(|(call, _)| call.kind == kind)
            }
            BatchEvent::Resolved { .. } => false,
        })
        .unwrap()
}

/// An image in the prompt is encoded and written into the prefix cache before the canvas
/// reads the prompt, whose positions then include the image's.
#[test]
fn a_canvas_reads_the_encoded_image_with_its_prompt() {
    let running = Running::start(image_worker(), config());
    let mut request = canvas_request(6, prompt(16, 1000));
    request
        .multimodal_inputs
        .images
        .push(vision_image(99, 10, 9));
    let answer = events(running.submit(request.clone()));
    let batches = running.stop();
    assert_generated(&answer, &request);

    let calls = calls_of(&batches, 6);
    assert!(
        calls
            .iter()
            .any(|(call, _)| call.kind == CallKind::Media(MediaCall::VisionEncoding))
    );
    let (canvas, forward) = calls
        .iter()
        .find(|(call, _)| call.kind == CallKind::Forward(ForwardMode::TokenDenoising))
        .unwrap();
    assert_eq!(canvas.coordinates.logical_position, 25);
    assert_eq!(canvas.coordinates.cache_visible_len, 25);
    assert_eq!(forward.seq_lens, vec![25 + CANVAS]);
}

/// A prompt with an image is written into the prefix cache by one context prefill, after
/// the image is encoded: the prefill carries the prompt tokens and the image
/// block at its prompt position, as one forward row per segment, and the
/// canvas then reads the whole context.
#[test]
fn an_image_prompt_is_written_by_one_context_prefill() {
    let running = Running::start(image_worker(), config());
    let mut request = canvas_request(9, prompt(16, 1000));
    request
        .multimodal_inputs
        .images
        .push(vision_image(99, 10, 9));
    let answer = events(running.submit(request.clone()));
    let batches = running.stop();
    assert_generated(&answer, &request);

    let calls = calls_of(&batches, 9);
    let kinds: Vec<_> = calls.iter().map(|(call, _)| call.kind).collect();
    assert_eq!(
        kinds[..2],
        [
            CallKind::Media(MediaCall::VisionEncoding),
            CallKind::Forward(ForwardMode::Prefill),
        ]
    );
    assert!(
        kinds.len() > 2
            && kinds[2..]
                .iter()
                .all(|kind| *kind == CallKind::Forward(ForwardMode::TokenDenoising)),
        "the canvas steps follow the context prefill: {kinds:?}"
    );
    let feature = calls[0].0.encoder_output.clone().unwrap();
    let (prefill, forward) = &calls[1];
    assert_eq!(prefill.input_token_ids, prompt(16, 1000));
    assert_eq!(prefill.vision_inputs.len(), 1);
    assert_eq!(prefill.vision_inputs[0].offset, 10);
    assert_eq!(
        prefill.vision_inputs[0].feature.buffer_id(),
        feature.buffer_id()
    );
    assert_eq!(prefill.coordinates.cache_visible_len, 0);
    // Text before the image, the image block, text after it.
    assert_eq!(forward.query_lens, vec![10, 9, 6]);
    assert_eq!(forward.seq_lens, vec![10, 19, 25]);
    assert_eq!(forward.write_cache, vec![true, true, true]);
    assert_eq!(calls[2].0.coordinates.cache_visible_len, 25);
}

/// Every image a context prefill reaches is encoded before it, and the
/// prefill writes all of their blocks between its prompt tokens.
#[test]
fn a_context_prefill_writes_every_image_it_reaches() {
    let running = Running::start(image_worker(), config());
    let mut request = canvas_request(10, prompt(20, 1000));
    request
        .multimodal_inputs
        .images
        .push(vision_image(101, 4, 5));
    request
        .multimodal_inputs
        .images
        .push(vision_image(102, 12, 7));
    let answer = events(running.submit(request.clone()));
    let batches = running.stop();
    assert_generated(&answer, &request);

    let calls = calls_of(&batches, 10);
    let encodes: Vec<_> = calls
        .iter()
        .filter(|(call, _)| call.kind == CallKind::Media(MediaCall::VisionEncoding))
        .map(|(call, _)| call.encoder_output.clone().unwrap().buffer_id())
        .collect();
    assert_eq!(encodes.len(), 2);
    let prefills = context_prefills(&batches, 10);
    assert_eq!(prefills.len(), 1);
    let (prefill, forward) = &prefills[0];
    assert_eq!(
        prefill
            .vision_inputs
            .iter()
            .map(|input| (input.offset, input.feature.buffer_id()))
            .collect::<Vec<_>>(),
        vec![(4, encodes[0]), (12, encodes[1])]
    );
    assert_eq!(forward.query_lens, vec![4, 5, 8, 7, 8]);
    assert_eq!(forward.seq_lens, vec![4, 9, 17, 24, 32]);
}

/// The rows a worker stages per call bound numerical segments, even when one
/// request's context has more segments than a call can hold.
#[test]
fn image_contexts_respect_the_workers_staged_row_capacity() {
    let mut sim = image_worker();
    sim.mut_info_for_test().max_batch_calls = 3;
    let running = Running::start(sim, config());
    let requests: Vec<_> = (0..8u64)
        .map(|index| {
            let mut request = canvas_request(100 + index, prompt(20, 1000 + 100 * index as u32));
            request.multimodal_inputs.images.extend([
                vision_image(1000 + index * 2, 4, 5),
                vision_image(1001 + index * 2, 12, 7),
            ]);
            request
        })
        .collect();
    let streams: Vec<_> = requests
        .iter()
        .map(|request| running.submit(request.clone()))
        .collect();
    for (stream, request) in streams.into_iter().zip(&requests) {
        assert_generated(&events(stream), request);
    }
    let batches = running.stop();
    for batch in &batches {
        let rows: usize = batch
            .requests
            .iter()
            .enumerate()
            .filter(|(_, (call, _))| call.kind == CallKind::Forward(ForwardMode::Prefill))
            .map(|(index, _)| crate::support::rows_of(batch, index).query_lens.len())
            .sum();
        assert!(rows <= 3, "prefill batch has {rows} numerical rows");
    }
    for request in &requests {
        let prefills = context_prefills(&batches, request.request_id.0);
        let tokens: Vec<_> = prefills
            .iter()
            .flat_map(|(call, _)| call.input_token_ids.iter().copied())
            .collect();
        assert_eq!(tokens, request.prompt_token_ids);
        // Chunking preserves both complete image blocks and every text token.
        assert_eq!(
            prefills
                .iter()
                .map(|(_, forward)| forward.query_lens.iter().sum::<u32>())
                .sum::<u32>(),
            32
        );
        assert_eq!(
            prefills
                .iter()
                .map(|(call, _)| call.vision_inputs.len())
                .sum::<usize>(),
            2
        );
    }
}

/// An image block is never split across context prefills: a chunk that
/// cannot hold the whole block ends before it, and the next chunk starts
/// with the block.
#[test]
fn a_context_chunk_ends_before_an_image_block_it_cannot_hold() {
    let config = SchedulerConfig {
        chunked_prefill_size: std::num::NonZeroUsize::new(16),
        ..config()
    };
    let running = Running::start(image_worker(), config);
    let mut request = canvas_request(11, prompt(30, 1000));
    request
        .multimodal_inputs
        .images
        .push(vision_image(103, 12, 9));
    let answer = events(running.submit(request.clone()));
    let batches = running.stop();
    assert_generated(&answer, &request);

    let chunks: Vec<_> = context_prefills(&batches, 11)
        .into_iter()
        .map(|(call, forward)| {
            (
                call.input_token_ids.len(),
                call.vision_inputs
                    .iter()
                    .map(|input| input.offset)
                    .collect::<Vec<_>>(),
                forward.query_lens,
            )
        })
        .collect();
    assert_eq!(
        chunks,
        vec![
            (12, vec![], vec![12]),
            (7, vec![0], vec![9, 7]),
            (11, vec![], vec![11]),
        ]
    );
}

/// The worker prepares each inline input image as one task on its rank's
/// host task pool, so an image encode waits in the scheduler while the pool
/// is full instead of reaching the worker: with one host task per rank,
/// image encodes of different requests are never in flight together, and
/// every request still completes.
#[test]
fn an_image_encode_waits_for_a_free_host_task() {
    let mut sim = image_worker();
    // A deep queue and results delivered only when the scheduler waits keep
    // every ready encode schedulable at once, leaving the host task pool to
    // bound them.
    sim.set_queue_depth(16);
    sim.set_results_on_wait(true);
    let info = sim.mut_info_for_test();
    info.host_task_capacity = 1;
    info.components = vec![ComponentInfo {
        name: "model".to_owned(),
        outputs: Vec::new(),
    }];
    info.media_components = BTreeMap::from([(MediaCall::VisionEncoding, "model".to_owned())]);
    sim.place_components(BTreeMap::from([(
        "model".to_owned(),
        BlockConfig::parallel(vec![0]),
    )]));
    let running = Running::start(sim, config());

    // Each prompt opens with its image, so every encode is ready at once.
    let requests: Vec<_> = (0..3u64)
        .map(|index| {
            let mut request = canvas_request(20 + index, prompt(16, 1000));
            request
                .multimodal_inputs
                .images
                .push(vision_image(200 + index, 0, 9));
            request
        })
        .collect();
    let streams: Vec<_> = requests
        .iter()
        .map(|request| running.submit(request.clone()))
        .collect();
    for (stream, request) in streams.into_iter().zip(&requests) {
        assert_generated(&events(stream), request);
    }
    let timeline = running.stop_with_resolutions();

    // Encodes in flight after each submission: a batch's encodes stay in
    // flight until its result is resolved.
    let mut in_flight: HashMap<u64, usize> = HashMap::new();
    let mut encodes = 0;
    for event in timeline {
        match event {
            BatchEvent::Submitted(batch) => {
                let count = batch
                    .requests
                    .iter()
                    .filter(|(call, _)| call.kind == CallKind::Media(MediaCall::VisionEncoding))
                    .count();
                encodes += count;
                in_flight.insert(batch.id, count);
                assert!(
                    in_flight.values().sum::<usize>() <= 1,
                    "image encodes exceeded the host task pool: {in_flight:?}"
                );
            }
            BatchEvent::Resolved { batch_id } => {
                in_flight.remove(&batch_id);
            }
        }
    }
    assert_eq!(encodes, requests.len());
}

/// A block's first canvas step is queued behind the context prefill that
/// writes the last image of its prompt, before that prefill resolves, as
/// behind a text prompt's last prefill.
#[test]
fn a_blocks_first_step_is_queued_behind_its_image_context_prefill() {
    let mut sim = image_worker();
    sim.set_queue_depth(2);
    sim.set_results_on_wait(true);
    let running = Running::start(sim, config());
    let mut request = canvas_request(12, prompt(24, 1000));
    request
        .multimodal_inputs
        .images
        .push(vision_image(104, 20, 9));
    let answer = events(running.submit(request.clone()));
    let history = running.stop_with_resolutions();
    assert_generated(&answer, &request);

    let prefill = first_submission(&history, CallKind::Forward(ForwardMode::Prefill));
    let step = first_submission(&history, CallKind::Forward(ForwardMode::TokenDenoising));
    let BatchEvent::Submitted(prefill_batch) = &history[prefill] else {
        unreachable!("the position names a submission");
    };
    assert_eq!(prefill_batch.requests[0].0.vision_inputs.len(), 1);
    let prefill_resolved = history
        .iter()
        .position(|event| {
            matches!(event, BatchEvent::Resolved { batch_id } if *batch_id == prefill_batch.id)
        })
        .unwrap();
    assert!(
        step < prefill_resolved,
        "the canvas step waited for its image context prefill to resolve"
    );
}

/// An image whose encoder product the encoder cache holds joins its context
/// prefill without an encode, reading the cached product.
#[test]
fn a_cached_image_joins_its_context_prefill_without_an_encode() {
    let running = Running::start(image_worker(), config());
    let mut first = canvas_request(13, prompt(16, 1000));
    first.multimodal_inputs.images.push(vision_image(105, 8, 9));
    assert_generated(&events(running.submit(first.clone())), &first);
    let mut second = canvas_request(14, prompt(12, 2000));
    second
        .multimodal_inputs
        .images
        .push(vision_image(105, 4, 9));
    assert_generated(&events(running.submit(second.clone())), &second);
    let batches = running.stop();

    let encoded = calls_of(&batches, 13)
        .into_iter()
        .find(|(call, _)| call.kind == CallKind::Media(MediaCall::VisionEncoding))
        .and_then(|(call, _)| call.encoder_output)
        .unwrap();
    let calls = calls_of(&batches, 14);
    assert!(
        calls
            .iter()
            .all(|(call, _)| call.kind != CallKind::Media(MediaCall::VisionEncoding))
    );
    let prefills = context_prefills(&batches, 14);
    assert_eq!(prefills.len(), 1);
    assert_eq!(prefills[0].0.vision_inputs[0].offset, 4);
    assert_eq!(
        prefills[0].0.vision_inputs[0].feature.buffer_id(),
        encoded.buffer_id()
    );
}
