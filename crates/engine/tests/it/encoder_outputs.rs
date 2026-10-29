//! Requests' encoder outputs in the host encoder pool, over the simulator
//! and an in-process Storage Server: an output the pool does not hold is
//! computed and stored, and a request that reads the same inputs (another,
//! or the same one resumed, on another world) reads it from the pool
//! instead of computing it. A video request's encoder outputs, and an image
//! request's encodings of its input images and of the images it generated.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use engine::{
    BatchEvent, CachePolicy, CommandSender, EncoderInputs, Event, EventRx, ExecutionBatch,
    FeedbackImage, FeedbackNextToken, FeedbackSource, GeneratedImage, GenerationConstraint,
    GenerationRequest, ImageEncoder, ImageEncoderInput, ImageGenerationConfig, ImageInput,
    ImageTrigger, MultimodalInputs, Request, SchedulerConfig, SimEngine, SimExecutor,
};
use ipc::{
    CallKind, EncoderPoolInfo, ImageEncoderOutput, ImageParams, MediaCall, RequestId,
    SamplingParams,
};

use crate::support;
use crate::support::storage::{DEPLOYMENT, HOST, StorageServer};
use crate::support::video::{video_request, video_worker};

/// The bytes of one page of the simulated encoder pool, which holds a
/// prompt's conditioning in one page.
const PAGE_BYTES: u64 = 4096;

/// A scheduler of a simulated worker whose rank maps a host encoder pool of
/// a Storage Server, with its loop running: one world of a deployment.
struct Encoders {
    handle: CommandSender,
    thread: thread::JoinHandle<bool>,
    boundary: Receiver<BatchEvent>,
}

impl Encoders {
    /// The world of `sim`, whose rank reports the encoder pool `pool` of
    /// `server`, which stores the outputs `identities` names.
    fn start(
        mut sim: SimEngine,
        server: &StorageServer,
        pool: u64,
        identities: &[(&str, &str)],
    ) -> Self {
        sim.set_encoder_pools(vec![(
            HOST.to_owned(),
            Some(Box::new(EncoderPoolInfo {
                pool,
                page_bytes: PAGE_BYTES,
                identities: identities
                    .iter()
                    .map(|(output, identity)| ((*output).to_owned(), (*identity).to_owned()))
                    .collect(),
                endpoint: None,
            })),
        )]);
        let mut inner = SimExecutor::new(sim);
        let boundary = inner.observe();
        let mut scheduler = support::scheduler_with(
            Box::new(inner),
            SimEngine::special_tokens(),
            SchedulerConfig {
                max_batch: 32,
                ..Default::default()
            },
        )
        .unwrap();
        scheduler.enable_encoder_store(server.socket(), DEPLOYMENT, 0, HOST);
        let (tx, rx) = crossbeam_channel::unbounded();
        let thread = thread::spawn(move || scheduler.run(rx));
        Self {
            handle: CommandSender::new(tx),
            thread,
            boundary,
        }
    }

    /// Shuts the scheduler down and returns every batch it submitted.
    fn stop(self) -> Vec<ExecutionBatch> {
        self.handle.shutdown();
        assert!(!self.thread.join().unwrap(), "the engine loop died");
        self.boundary
            .iter()
            .filter_map(|event| match event {
                BatchEvent::Submitted(batch) => Some(batch),
                BatchEvent::Resolved { .. } => None,
            })
            .collect()
    }
}

/// The `t2va` request `id`, whose text encoding reads what `text` digests;
/// `None` names no digests.
fn video(id: u64, text: Option<[u8; 32]>) -> Request {
    let Request::Diffusion(mut request) = video_request(id, 1) else {
        unreachable!("a video request");
    };
    request.encoder_inputs = text.map(|text| {
        Box::new(EncoderInputs {
            text,
            vision: None,
            video_units: Vec::new(),
            audio: None,
        })
    });
    Request::Diffusion(request)
}

/// Reads `events` to their end, within thirty seconds, and returns them.
fn drain(events: &mut EventRx) -> Vec<Event> {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut read = Vec::new();
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(event) => {
                let terminal = event.is_terminal();
                read.push(event);
                if terminal {
                    return read;
                }
            }
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    panic!("request did not end: {read:?}");
}

/// Whether `events` end with the request's video.
fn completes(events: &mut EventRx) -> bool {
    let read = drain(events);
    read.iter().any(|event| matches!(event, Event::Artifact(_)))
        && matches!(read.last(), Some(Event::Finished { .. }))
}

/// How each call of `kind` treated the host encoder pool, by request:
/// `Some(true)` read its output from the pool, `Some(false)` computed it
/// and wrote it there, `None` neither.
fn encodings(batches: &[ExecutionBatch], kind: MediaCall) -> Vec<(RequestId, Option<bool>)> {
    batches
        .iter()
        .flat_map(|batch| &batch.requests)
        .filter(|(call, _)| call.kind == CallKind::Media(kind))
        .map(|(call, _)| {
            (
                call.request_key.request_id,
                call.host_output.as_ref().map(|output| output.import),
            )
        })
        .collect()
}

/// The first request that reads some inputs finds no output in the pool:
/// its encoder computes the output and writes it there. A later request
/// reading the same inputs reads the output instead of computing it, one
/// reading other inputs computes its own, and a request naming no digests
/// neither reads nor writes the pool; every one completes with its video.
#[test]
fn an_encoder_output_is_computed_when_missing_and_read_when_stored() {
    let mut server = StorageServer::start(1 << 20);
    let pool = server.encoder_pool("sim", PAGE_BYTES, 16);
    let encoders = Encoders::start(
        video_worker(1, 1),
        &server,
        pool,
        &[("conditioning", "text encoder")],
    );
    // Each request is submitted once the one before completed, so the
    // output it wrote was committed before the next looks it up.
    for (id, text) in [
        (1, Some([1; 32])),
        (2, Some([1; 32])),
        (3, Some([2; 32])),
        (4, None),
    ] {
        let mut events = encoders.handle.submit(video(id, text)).unwrap();
        assert!(completes(&mut events), "request {id} completed");
    }

    assert_eq!(
        encodings(&encoders.stop(), MediaCall::TextEncoding),
        [
            (RequestId(1), Some(false)),
            (RequestId(2), Some(true)),
            (RequestId(3), Some(false)),
            (RequestId(4), None),
        ]
    );
}

/// The identities a simulated image rank reports of its encoders' outputs.
const IMAGE_ENCODERS: [(&str, &str); 2] = [("vision_features", "vit"), ("latent_features", "vae")];

/// A simulated image rank whose vision encoder computes its features alike
/// for every draw and whose latent encoder samples its posterior, as
/// BAGEL's do: each writes rows of 64 bytes.
fn image_worker() -> SimEngine {
    let mut sim = SimEngine::new();
    sim.set_text_len(4);
    *sim.mut_info_for_test().image_encoders = BTreeMap::from([
        (
            "vision_features".to_owned(),
            ImageEncoderOutput {
                row_bytes: 64,
                draws: false,
            },
        ),
        (
            "latent_features".to_owned(),
            ImageEncoderOutput {
                row_bytes: 64,
                draws: true,
            },
        ),
    ]);
    sim
}

/// An encoder input of `encoder`, writing `tokens` cache tokens.
fn encoder(encoder: ImageEncoder, tokens: u32) -> ImageEncoderInput {
    ImageEncoderInput {
        encoder,
        num_cache_tokens: Some(tokens),
        max_cache_tokens: None,
        height: None,
        width: None,
    }
}

/// Image request `id` under `seed` and the cache policy `cache`: a prompt
/// whose input image `image` (a digest) BAGEL encodes with its latent and
/// its vision encoder, then text and up to one generated image, which a
/// latent encoder feeds back before text continues. The image parameters it
/// is admitted with carry `seed`, which the latent encoder's draws follow.
fn image_request(id: u64, image: u8, seed: u64, cache: CachePolicy) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: vec![1, 2, 3, 4],
        multimodal_inputs: MultimodalInputs {
            images: vec![ImageInput {
                digest: [image; 32],
                b64: "aW1hZ2U=".into(),
                position: 2,
                num_positions: 1,
                encoders: vec![
                    encoder(ImageEncoder::Latent, 18),
                    encoder(ImageEncoder::Vision, 9),
                ],
            }],
        },
        negative_prompt_token_ids: Vec::new(),
        constraint: GenerationConstraint::Default,
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(seed),
            ..SamplingParams::default()
        },
        image: ImageParams {
            steps: 2,
            height: 32,
            width: 32,
            seed: Some(seed),
            ..ImageParams::default()
        },
        max_images: 1,
        max_und_tokens: 8,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache,
        image_generation: ImageGenerationConfig {
            // The request's second greedy token opens its image.
            trigger: ImageTrigger::Token {
                token_id: engine::sim_text_token(RequestId(id), 1),
            },
            requires_text_for_image: false,
            feedback_source: Some(FeedbackSource::DeviceProduct),
            feedback_next_token: FeedbackNextToken::Token { token_id: 7 },
            num_feedback_positions: 2,
            feedback_encoders: vec![encoder(ImageEncoder::Latent, 6)],
            sample_feedback_continuation: false,
        },
        readout: Vec::new(),
        generated_token_ids: Vec::new(),
        adapter: 0,
        generated_images: Vec::new(),
        trajectory: Default::default(),
    }
}

/// Runs `request` to its end on a world of its own, whose encoder cache
/// starts empty, over `server`'s pool `pool`; returns its events and the
/// batches the world submitted.
fn run_image(
    server: &StorageServer,
    pool: u64,
    request: GenerationRequest,
) -> (Vec<Event>, Vec<ExecutionBatch>) {
    let world = Encoders::start(image_worker(), server, pool, &IMAGE_ENCODERS);
    let id = request.request_id;
    let mut events = world.handle.submit(request).unwrap();
    let read = drain(&mut events);
    assert!(
        matches!(read.last(), Some(Event::Finished { .. })),
        "request {id:?} finished: {read:?}"
    );
    (read, world.stop())
}

/// How request `id`'s input image encodings treated the host encoder pool:
/// its latent encoding's, then its vision encoding's.
fn input_encodings(batches: &[ExecutionBatch], id: u64) -> (Option<bool>, Option<bool>) {
    let of = |kind| {
        encodings(batches, kind)
            .into_iter()
            .find(|(request, _)| *request == RequestId(id))
            .unwrap_or_else(|| panic!("request {id} encodes its input image by {kind:?}"))
            .1
    };
    (of(MediaCall::LatentEncoding), of(MediaCall::VisionEncoding))
}

/// An input image's encodings that no world stored are computed and
/// written to the pool; another world reads them instead of encoding. The
/// vision features follow the image alone, so a request of another seed
/// reads them; the latent features sample their posterior at the request's
/// seed and draw, so a request of another seed computes its own. Another
/// image computes both, and a request whose cache policy neither reads nor
/// writes cached outputs encodes without the pool; every request finishes.
#[test]
fn an_image_encoding_is_computed_when_missing_and_read_on_another_world() {
    let mut server = StorageServer::start(1 << 20);
    let pool = server.encoder_pool("sim", PAGE_BYTES, 64);
    // Whether a request reads and writes cached outputs.
    let shared = (true, true);
    let uncached = (false, false);
    // Each request runs on a world of its own, once the one before
    // finished, so the outputs it wrote were committed before the next
    // looks them up and no world's encoder cache serves another's request.
    let runs = [
        (1, 5, 5, shared, (Some(false), Some(false))),
        (2, 5, 5, shared, (Some(true), Some(true))),
        (3, 5, 6, shared, (Some(false), Some(true))),
        (4, 6, 5, shared, (Some(false), Some(false))),
        (5, 5, 5, uncached, (None, None)),
    ];
    for (id, image, seed, (read, write), expected) in runs {
        let cache = CachePolicy {
            read,
            write,
            ..CachePolicy::default()
        };
        let (_, batches) = run_image(&server, pool, image_request(id, image, seed, cache));
        assert_eq!(input_encodings(&batches, id), expected, "request {id}");
    }
}

/// The facts a migration carries of a request that read `events`: every
/// token it generated, the image-opening tokens among them, and each image
/// with the values its feedback reads, after the tokens before it.
fn continuation(mut request: GenerationRequest, events: &[Event]) -> GenerationRequest {
    for event in events {
        match event {
            Event::TextToken { id, .. } | Event::ControlToken { id } => {
                request.generated_token_ids.push(*id);
            }
            Event::TextTokens { ids } => request.generated_token_ids.extend_from_slice(ids),
            Event::ImageDone { feedback, .. } => request.generated_images.push(GeneratedImage {
                tokens: u32::try_from(request.generated_token_ids.len()).unwrap(),
                image: FeedbackImage::Pixels(Arc::clone(
                    feedback.as_ref().expect("the image's feedback values"),
                )),
            }),
            _ => {}
        }
    }
    request
}

/// A request's feedback encoding of the image it generated, planned once
/// the image's values reached the engine (a simulated rank runs one batch
/// at a time), is computed and written to the pool. The request continued
/// on another world rebuilds its context from its facts: its input image's
/// encodings and the feedback encoding of the image it carries, at the same
/// draw, are read from the pool instead of encoded. Continued on a world
/// whose pool holds none of them, it computes and stores them all, and a
/// further continuation there reads them; every run finishes.
#[test]
fn a_continued_image_request_reads_its_images_encodings_on_another_world() {
    let mut server = StorageServer::start(1 << 20);
    let pool = server.encoder_pool("sim", PAGE_BYTES, 64);
    let request = image_request(1, 5, 5, CachePolicy::default());
    let (events, first) = run_image(&server, pool, request.clone());
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Event::ImageDone { .. })),
        "the request generated its image: {events:?}"
    );
    // The input image's latent encoding, then the feedback encoding of the
    // generated one.
    assert_eq!(
        encodings(&first, MediaCall::LatentEncoding),
        [(RequestId(1), Some(false)), (RequestId(1), Some(false))]
    );

    let continued = continuation(request, &events);
    let (_, second) = run_image(&server, pool, continued.clone());
    assert_eq!(
        encodings(&second, MediaCall::LatentEncoding),
        [(RequestId(1), Some(true)), (RequestId(1), Some(true))]
    );
    assert_eq!(
        encodings(&second, MediaCall::VisionEncoding),
        [(RequestId(1), Some(true))]
    );

    let mut elsewhere = StorageServer::start(1 << 20);
    let other = elsewhere.encoder_pool("sim", PAGE_BYTES, 64);
    for read in [false, true] {
        let (_, batches) = run_image(&elsewhere, other, continued.clone());
        assert_eq!(
            encodings(&batches, MediaCall::LatentEncoding),
            [(RequestId(1), Some(read)), (RequestId(1), Some(read))]
        );
        assert_eq!(
            encodings(&batches, MediaCall::VisionEncoding),
            [(RequestId(1), Some(read))]
        );
    }
}
