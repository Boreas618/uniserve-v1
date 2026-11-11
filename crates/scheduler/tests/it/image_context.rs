//! Input images in a token request's context, over the fake at the batch
//! interface.
//!
//! Every input image is encoded before the context prefill that reaches it,
//! and that prefill writes the image's vision block whole between the prompt
//! tokens around it. The requests here are generated in blocks by a diffusion
//! language model, whose prompt prefill samples nothing, so the context
//! prefills are the only prefills a request runs: the fake's text is shorter
//! than one block, and the first block, which holds EOS, ends the request
//! uncommitted. Cases observe the events a request publishes and the batches
//! the fake accepts.

use crate::support;
use crate::support::fake::Pacer;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ipc::execution::{Batch, RowKind, Variant};
use ipc::{BufferId, CallKind, ForwardMode, MediaCall, StageInfo};
use ipc::{CanvasSampling, RequestId, SamplingParams};
use scheduler::{
    CachePolicy, Event, FinishReason, GenerationRequest, ImageEncoderInput, ImageInput,
    MultimodalInputs,
};
use scheduler::{CommandSender, EventRx, SchedulerConfig};
use testkit::{Accepted, FakeBatches};

/// Tokens of one block.
const BLOCK: u32 = 16;
/// Text tokens before EOS: fewer than one block holds.
const TEXT_LEN: u64 = 4;
/// The block-diffusion sampling the deployment is launched with.
const SAMPLING: CanvasSampling = CanvasSampling {
    canvas_length: BLOCK,
    max_steps: 4,
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
        canvas: Some(SAMPLING),
        ..SchedulerConfig::default()
    }
}

/// The pool of a block-diffusion deployment that reads images: one rank that
/// prefills prompts, denoises blocks and encodes images on its own stage,
/// with 16-token cache pages and room for the encoders' features.
fn image_pool() -> ipc::execution::Inventory {
    let mut pool = support::pool("model");
    for rank in &mut pool.ranks {
        rank.supported_calls = vec![
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::TokenDenoising),
            CallKind::Media(MediaCall::VisionEncoding),
        ];
        rank.buffer_pool_bytes = 257_u64 * (256 << 20);
        rank.encoder_cache_entries = 256;
        rank.encoder_entry_bytes = 256 << 20;
        rank.max_batch_calls = 1024;
        rank.max_unresolved_calls = 2;
        rank.stages = vec![StageInfo {
            name: "model".to_owned(),
            outputs: Vec::new(),
        }];
        rank.media_stages = BTreeMap::from([(MediaCall::VisionEncoding, "model".to_owned())]);
        for group in rank
            .prefix_cache
            .iter_mut()
            .flat_map(|cache| cache.groups.iter_mut())
        {
            group.block_size = 16;
        }
    }
    pool
}

/// The fake over `pool`: its text ends after `TEXT_LEN` tokens, its blocks
/// stop within the deployment's step limit, and it serves every media
/// computation the mechanism states.
fn fake(pool: ipc::execution::Inventory) -> Arc<FakeBatches> {
    Arc::new(
        FakeBatches::new(vec![pool])
            .with_text_len(TEXT_LEN)
            .with_block_steps(SAMPLING.max_steps)
            .with_media_outputs(Arc::new(testkit::Media::new())),
    )
}

fn prompt(length: u32, first: u32) -> Vec<u32> {
    (0..length).map(|index| first + index).collect()
}

/// A request of the block-diffusion deployment over `prompt_token_ids`, with
/// no images yet.
fn block_request(request_id: u64, prompt_token_ids: Vec<u32>) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(request_id),
        prompt_token_ids,
        multimodal_inputs: MultimodalInputs::default(),
        sampling: SamplingParams {
            // Each request names a seed of its own, so each draws its own stream.
            seed: Some(request_id),
            ..SamplingParams::default()
        },
        max_und_tokens: 1_000,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy::default(),
        include_stop_token: false,
        generated_token_ids: Vec::new(),
        adapter: 0,
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

/// The Philox stream of request `id`, which states its id as its seed: what
/// the fake's text and stopping steps follow. A seeded request's stream is a
/// function of its seed alone.
fn stream(id: u64) -> u64 {
    ipc::philox::request_sampling_key(Some(id), 0, 0, 0)
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
            .expect("the fake serves block diffusion over images");
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

/// Asserts that the request on stream `key`, whose context holds `context`
/// positions, published its text as one block and ended on the EOS after it.
fn assert_generated(events: &[Event], key: u64, context: u64) {
    let blocks: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            Event::TextTokens { ids } => Some(ids.clone()),
            _ => None,
        })
        .collect();
    let text: Vec<u32> = (0..TEXT_LEN)
        .map(|index| {
            testkit::draw_token(ipc::philox::DrawAddress {
                key,
                position: context + index,
            })
        })
        .collect();
    assert_eq!(blocks, vec![text]);
    match events.last() {
        Some(Event::Finished {
            reason,
            completion_tokens,
            ..
        }) => {
            assert_eq!(*reason, FinishReason::Eos);
            assert_eq!(*completion_tokens, TEXT_LEN as usize + 1);
        }
        other => panic!("request ended with {other:?}"),
    }
}

/// The feature each image encode of `batch` writes, for request `id`.
fn encodes(batch: &Batch, id: u64) -> Vec<BufferId> {
    let encodes = batch
        .media
        .as_ref()
        .is_some_and(|media| media.computation.call == MediaCall::VisionEncoding);
    batch
        .images
        .iter()
        .filter(|image| encodes && image.request.request_id == RequestId(id))
        .map(|image| image.output)
        .collect()
}

/// One context prefill of a request, as its batch states it.
#[derive(Debug, PartialEq)]
struct Prefill {
    /// The prompt tokens the call embeds, in order.
    text: Vec<u32>,
    /// Each image block the call writes: its offset among the call's text
    /// tokens, its rows and the feature it reads.
    blocks: Vec<(u32, u32, BufferId)>,
    /// The tokens each row of the call evaluates.
    query_lens: Vec<u32>,
    /// The keys each row sees.
    seq_lens: Vec<u32>,
}

/// The context prefills `batch` runs for the request on stream `key`.
fn prefills(batch: &Batch, key: u64) -> Vec<Prefill> {
    let columns = &batch.columns;
    let offsets = &columns.rows.query_offsets;
    columns
        .rows
        .calls
        .iter()
        .filter(|call| {
            call.kind == RowKind::Prefill && columns.draws[call.rows.start as usize].key == key
        })
        .map(|call| {
            let rows = call.rows.start as usize..call.rows.end as usize;
            let tokens = offsets[rows.start] as u32..offsets[rows.end] as u32;
            // The image blocks of the call, at their batch token indices.
            let vision: Vec<_> = columns
                .vision
                .iter()
                .filter(|block| tokens.contains(&block.token))
                .collect();
            let in_block = |token: u32| {
                vision
                    .iter()
                    .any(|block| (block.token..block.token + block.rows).contains(&token))
            };
            let mut text = Vec::new();
            let mut blocks = Vec::new();
            for token in tokens.clone() {
                if let Some(block) = vision.iter().find(|block| block.token == token) {
                    blocks.push((text.len() as u32, block.rows, block.feature));
                }
                if !in_block(token) {
                    text.push(columns.input_ids[token as usize] as u32);
                }
            }
            Prefill {
                text,
                blocks,
                query_lens: rows
                    .clone()
                    .map(|row| (offsets[row + 1] - offsets[row]) as u32)
                    .collect(),
                seq_lens: rows.map(|row| columns.rows.seq_lens[row] as u32).collect(),
            }
        })
        .collect()
}

/// Whether `batch` denoises a block of the request on stream `key`.
fn denoises(batch: &Batch, key: u64) -> bool {
    batch.shape.variant == Variant::TokenDenoise
        && batch
            .columns
            .rows
            .calls
            .iter()
            .any(|call| batch.columns.draws[call.rows.start as usize].key == key)
}

/// A prompt with an image is written into the cache by one context prefill,
/// after the image is encoded: the prefill carries the prompt tokens and the
/// image block at its prompt position as one causal row, and the block then
/// reads the whole context.
#[test]
fn an_image_prompt_is_written_by_one_context_prefill() {
    let running = Running::start(fake(image_pool()), config());
    let key = stream(9);
    let mut request = block_request(9, prompt(16, 1000));
    request
        .multimodal_inputs
        .images
        .push(vision_image(99, 10, 9));
    let response = events(running.submit(request));
    let accepted = running.stop();
    assert_generated(&response, key, 25);

    // The image is encoded, the context is prefilled, and the block's steps
    // follow.
    let feature = encodes(&accepted[0].batch, 9);
    assert_eq!(feature.len(), 1, "the image is encoded first");
    let context = prefills(&accepted[1].batch, key);
    assert_eq!(
        context,
        vec![Prefill {
            text: prompt(16, 1000),
            // Text before the image, the image block and text after it are
            // one causal row.
            blocks: vec![(10, 9, feature[0])],
            query_lens: vec![25],
            seq_lens: vec![25],
        }]
    );
    assert!(accepted.len() > 2);
    for step in &accepted[2..] {
        assert!(
            denoises(&step.batch, key),
            "the block's steps follow the context prefill"
        );
        // The block reads the whole context: 25 cached keys and its own 16.
        assert_eq!(step.batch.columns.rows.seq_lens, vec![41]);
    }
}

/// Every image a context prefill reaches is encoded before it, and the
/// prefill writes all of their blocks between its prompt tokens.
#[test]
fn a_context_prefill_writes_every_image_it_reaches() {
    let running = Running::start(fake(image_pool()), config());
    let key = stream(10);
    let mut request = block_request(10, prompt(20, 1000));
    request
        .multimodal_inputs
        .images
        .push(vision_image(101, 4, 5));
    request
        .multimodal_inputs
        .images
        .push(vision_image(102, 12, 7));
    let response = events(running.submit(request));
    let accepted = running.stop();
    assert_generated(&response, key, 32);

    let features: Vec<BufferId> = accepted
        .iter()
        .flat_map(|accepted| encodes(&accepted.batch, 10))
        .collect();
    assert_eq!(features.len(), 2);
    let context: Vec<Prefill> = accepted
        .iter()
        .flat_map(|accepted| prefills(&accepted.batch, key))
        .collect();
    assert_eq!(
        context,
        vec![Prefill {
            text: prompt(20, 1000),
            blocks: vec![(4, 5, features[0]), (12, 7, features[1])],
            query_lens: vec![32],
            seq_lens: vec![32],
        }]
    );
}

/// The calls a rank takes in one batch bound a prefill batch's rows, even
/// when the requests' contexts hold more segments than a batch can.
#[test]
fn image_contexts_respect_the_ranks_batch_call_bound() {
    let mut pool = image_pool();
    for rank in &mut pool.ranks {
        rank.max_batch_calls = 3;
    }
    let running = Running::start(fake(pool), config());
    let requests: Vec<_> = (0..8u64)
        .map(|index| {
            let mut request = block_request(100 + index, prompt(20, 1000 + 100 * index as u32));
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
    for (events_rx, request) in streams.into_iter().zip(&requests) {
        assert_generated(&events(events_rx), stream(request.request_id.0), 32);
    }
    let accepted = running.stop();
    for accepted in &accepted {
        let batch = &accepted.batch;
        if batch
            .columns
            .rows
            .calls
            .iter()
            .any(|call| call.kind == RowKind::Prefill)
        {
            let rows = batch.columns.rows.seq_lens.len();
            assert!(rows <= 3, "prefill batch has {rows} rows");
        }
    }
    for request in &requests {
        let key = stream(request.request_id.0);
        let context: Vec<Prefill> = accepted
            .iter()
            .flat_map(|accepted| prefills(&accepted.batch, key))
            .collect();
        let text: Vec<u32> = context
            .iter()
            .flat_map(|prefill| prefill.text.iter().copied())
            .collect();
        assert_eq!(text, request.prompt_token_ids);
        // Chunking preserves both complete image blocks and every text token.
        assert_eq!(
            context
                .iter()
                .map(|prefill| prefill.query_lens.iter().sum::<u32>())
                .sum::<u32>(),
            32
        );
        assert_eq!(
            context
                .iter()
                .map(|prefill| prefill.blocks.len())
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
    let running = Running::start(fake(image_pool()), config);
    let key = stream(11);
    let mut request = block_request(11, prompt(30, 1000));
    request
        .multimodal_inputs
        .images
        .push(vision_image(103, 12, 9));
    let response = events(running.submit(request));
    let accepted = running.stop();
    assert_generated(&response, key, 39);

    let chunks: Vec<_> = accepted
        .iter()
        .flat_map(|accepted| prefills(&accepted.batch, key))
        .map(|prefill| {
            (
                prefill.text.len(),
                prefill
                    .blocks
                    .iter()
                    .map(|block| block.0)
                    .collect::<Vec<_>>(),
                prefill.query_lens,
            )
        })
        .collect();
    assert_eq!(
        chunks,
        vec![
            (12, vec![], vec![12]),
            (7, vec![0], vec![16]),
            (11, vec![], vec![11]),
        ]
    );
}

/// A rank prepares each inline input image as one task on its host thread
/// pool, so an image encode waits in the scheduler while the pool is full
/// instead of reaching the rank: with one host task per rank, image encodes
/// of different requests are never in flight together, and every request
/// still completes.
#[test]
fn an_image_encode_waits_for_a_free_host_task() {
    let mut pool = image_pool();
    // A deep queue and a slow device keep every ready encode schedulable at
    // once, leaving the host thread pool to bound them.
    pool.layout.queue_depth = 16;
    for rank in &mut pool.ranks {
        rank.host_task_capacity = 1;
    }
    let fake = fake(pool);
    let pacer = Pacer::start(&fake, Duration::from_millis(20));
    let running = Running::start(fake, config());

    // Each prompt opens with its image, so every encode is ready at once.
    let requests: Vec<_> = (0..3u64)
        .map(|index| {
            let mut request = block_request(20 + index, prompt(16, 1000));
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
    for (events_rx, request) in streams.into_iter().zip(&requests) {
        assert_generated(&events(events_rx), stream(request.request_id.0), 25);
    }
    drop(pacer);
    let accepted = running.stop();

    let encoding = |batch: &Batch| {
        requests
            .iter()
            .map(|request| encodes(batch, request.request_id.0).len())
            .sum::<usize>()
    };
    let mut total = 0;
    for encode in accepted
        .iter()
        .filter(|accepted| encoding(&accepted.batch) > 0)
    {
        let count = encoding(&encode.batch);
        total += count;
        let others: usize = accepted
            .iter()
            .filter(|other| encode.in_flight.contains(&other.batch.id))
            .map(|other| encoding(&other.batch))
            .sum();
        assert!(
            count + others <= 1,
            "image encodes exceeded the host thread pool: {count} accepted beside {others}"
        );
    }
    assert_eq!(total, requests.len());
}

/// A block's first step is queued behind the context prefill that writes the
/// last image of its prompt, before that prefill completes, as behind a text
/// prompt's last prefill.
#[test]
fn a_blocks_first_step_is_queued_behind_its_image_context_prefill() {
    let mut pool = image_pool();
    pool.layout.queue_depth = 2;
    let fake = fake(pool);
    let pacer = Pacer::start(&fake, Duration::from_millis(20));
    let running = Running::start(fake, config());
    let key = stream(12);
    let mut request = block_request(12, prompt(24, 1000));
    request
        .multimodal_inputs
        .images
        .push(vision_image(104, 20, 9));
    let response = events(running.submit(request));
    drop(pacer);
    let accepted = running.stop();
    assert_generated(&response, key, 33);

    let prefill = accepted
        .iter()
        .find(|accepted| !prefills(&accepted.batch, key).is_empty())
        .expect("the context is prefilled");
    assert_eq!(prefills(&prefill.batch, key)[0].blocks.len(), 1);
    let step = accepted
        .iter()
        .find(|accepted| denoises(&accepted.batch, key))
        .expect("the block is denoised");
    assert!(
        step.in_flight.contains(&prefill.batch.id),
        "the block's first step waited for its image context prefill to complete"
    );
}

/// An image whose encoder product the encoder cache holds joins its context
/// prefill without an encode, reading the cached product.
#[test]
fn a_cached_image_joins_its_context_prefill_without_an_encode() {
    let running = Running::start(fake(image_pool()), config());
    let mut first = block_request(13, prompt(16, 1000));
    first.multimodal_inputs.images.push(vision_image(105, 8, 9));
    assert_generated(&events(running.submit(first)), stream(13), 25);
    let mut second = block_request(14, prompt(12, 2000));
    second
        .multimodal_inputs
        .images
        .push(vision_image(105, 4, 9));
    assert_generated(&events(running.submit(second)), stream(14), 21);
    let accepted = running.stop();

    let encoded: Vec<BufferId> = accepted
        .iter()
        .flat_map(|accepted| encodes(&accepted.batch, 13))
        .collect();
    assert_eq!(encoded.len(), 1);
    assert!(
        accepted
            .iter()
            .all(|accepted| encodes(&accepted.batch, 14).is_empty()),
        "the cached image is not encoded again"
    );
    let context: Vec<Prefill> = accepted
        .iter()
        .flat_map(|accepted| prefills(&accepted.batch, stream(14)))
        .collect();
    assert_eq!(context.len(), 1);
    assert_eq!(context[0].blocks, vec![(4, 9, encoded[0])]);
}
