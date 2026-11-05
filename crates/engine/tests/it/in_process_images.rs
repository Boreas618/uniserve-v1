//! An image is encoded into its prompt, over the in-process pool.
//!
//! The launched case runs the encoder in a worker process: a request's inline
//! image crosses a vision encode call, the features it produces condition the
//! prompt, and the request generates text. Here the deployment is a handle
//! over `support::Serving` whose rank states the encode capability, so the
//! image crosses the same call without a process.

use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine as _;
use engine::{
    BatchCapacity, EngineConfig, Event, FinishReason, GenerationConstraint, GenerationRequest,
    ImageEncoderInput, ImageInput, MultimodalInputs, SchedulerHandle, Stage,
};
use executor::batch::Batches;
use executor::plan::Variant;
use ipc::{ImageParams, RequestId, SamplingParams};

use crate::support::{self, Serving};

/// How long a command answer may take.
const ANSWER: Duration = Duration::from_secs(60);

/// How long a request may take to reach its terminal event.
const DEADLINE: Duration = Duration::from_secs(60);

/// A base64 PNG of `side` x `side` pixels of one color.
fn png(side: u32, color: [u8; 3]) -> String {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, side, side);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let pixels: Vec<u8> = (0..side * side).flat_map(|_| color).collect();
        writer.write_image_data(&pixels).unwrap();
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// A request whose prompt carries one 32x32 image at position 2 of four
/// prompt tokens, as four feature tokens of one temporal position.
fn image_request(id: u64) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: vec![5, 6, 7, 8],
        negative_prompt_token_ids: Vec::new(),
        multimodal_inputs: MultimodalInputs {
            images: vec![ImageInput {
                digest: support::image_digest(200),
                b64: png(32, [200, 40, 200]),
                position: 2,
                num_positions: 1,
                encoders: vec![ImageEncoderInput {
                    encoder: engine::ImageEncoder::Vision,
                    num_cache_tokens: Some(4),
                    max_cache_tokens: None,
                    height: None,
                    width: None,
                }],
            }],
        },
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        // Understanding reads the image's features; it generates no image, so
        // the image parameters are the deployment's defaults.
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: 4,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        include_stop_token: false,
        image_generation: Default::default(),
        readout: Vec::new(),
        generated_token_ids: Vec::new(),
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
    let installed = handle
        .precopy_policy(program, source)
        .recv_timeout(ANSWER)
        .expect("the scheduler answers a precopy");
    installed.expect("the policy is precopied");
    let switched = handle
        .stage(Stage::Switchover)
        .recv_timeout(ANSWER)
        .expect("the scheduler answers a switchover");
    switched.expect("the policy serves");
}

/// An image request is encoded through a call of its own and generates text:
/// the encode reaches the pool as an `Encode` batch, and the request finishes
/// with tokens rather than an error.
#[test]
fn an_input_image_is_encoded_into_its_prompt() {
    let pool = Arc::new(Serving::by_extent(vec![support::media_pool("model")]));
    let mut config = EngineConfig::sim("in-process-images");
    config.world = 1;
    let handle = SchedulerHandle::over_batches(config, pool.clone() as Arc<dyn Batches>)
        .expect("the deployment serves");
    install(&handle, &pool);

    let mut events = handle
        .submit(image_request(11))
        .expect("the deployment takes an image request");

    let mut tokens = Vec::new();
    let mut terminal = None;
    let deadline = Instant::now() + DEADLINE;
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::TextToken { id, .. }) => tokens.push(id),
            Ok(Event::TextTokens { ids }) => tokens.extend(ids),
            Ok(event) if event.is_terminal() => {
                terminal = Some(event);
                break;
            }
            Ok(_) => {}
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();

    let encodes = pool
        .submitted()
        .iter()
        .filter(|batch| batch.shape.variant == Variant::Encode)
        .count();
    assert!(
        encodes > 0,
        "no encode batch reached the pool: the image never crossed an encoder"
    );
    assert!(
        !tokens.is_empty(),
        "understanding produced no text: the request ended with {terminal:?}"
    );
    assert!(
        matches!(terminal, Some(Event::Finished { .. })),
        "understanding failed: the request ended with {terminal:?}"
    );
    if let Some(Event::Finished { reason, .. }) = &terminal {
        assert!(
            !matches!(reason, FinishReason::Error),
            "understanding failed: the request ended with {reason:?}"
        );
    }
}
