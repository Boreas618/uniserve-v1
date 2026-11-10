//! Requests' encoder outputs in the host encoder pool, over the batch fake
//! and an in-process Storage Server: an output the pool does not hold is
//! computed and stored, and a request that reads the same inputs (another,
//! or the same one resumed, on another epoch) reads it from the pool
//! instead of computing it. A video request's encoder outputs, and a
//! request's encodings of its input images.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ipc::execution::{Batch, Batches, Inventory};
use ipc::{EncoderPoolInfo, ImageEncoderOutput, MediaCall, RequestId, SamplingParams};
use scheduler::{
    CachePolicy, CommandSender, EncoderInputs, Event, EventRx, GenerationRequest, ImageEncoder,
    ImageEncoderInput, ImageInput, MultimodalInputs, Request, SchedulerConfig,
};
use testkit::{FakeBatches, Media};

use crate::support;
use crate::support::storage::{DEPLOYMENT, HOST, StorageServer};
use crate::support::video::{video_fake, video_pool, video_request};

/// The bytes of one page of the fake's encoder pool, which holds a prompt's
/// conditioning in one page.
const PAGE_BYTES: u64 = 4096;

/// Every rank of `pools` on the Storage Server's host, mapping its encoder
/// pool `pool`, which stores the outputs `identities` names.
fn mapping(mut pools: Vec<Inventory>, pool: u64, identities: &[(&str, &str)]) -> Vec<Inventory> {
    for rank in pools.iter_mut().flat_map(|pool| pool.ranks.iter_mut()) {
        rank.endpoint.host = HOST.to_owned();
        rank.encoder_pool = Some(Box::new(EncoderPoolInfo {
            pages: ipc::HostPoolInfo {
                pool,
                shape: storage::protocol::PageShape {
                    layout: storage::Key([8; 32]),
                    kv_head_offset: 0,
                    kv_head_count: 0,
                    page_bytes: PAGE_BYTES,
                },
                endpoint: None,
            },
            identities: identities
                .iter()
                .map(|(output, identity)| ((*output).to_owned(), (*identity).to_owned()))
                .collect(),
        }));
    }
    pools
}

/// A scheduler over a fake worker whose ranks map a host encoder pool of a
/// Storage Server, with its loop running: one epoch of a deployment.
struct Encoders {
    handle: CommandSender,
    thread: thread::JoinHandle<bool>,
    fake: Arc<FakeBatches>,
}

impl Encoders {
    /// The epoch of `fake`, whose encoder store is `server`'s.
    fn start(fake: Arc<FakeBatches>, server: &StorageServer) -> Self {
        let mut scheduler = support::fake::scheduler_over(
            Arc::clone(&fake) as Arc<dyn Batches>,
            SchedulerConfig {
                max_batch: proto::config::positive(32),
                ..Default::default()
            },
            Default::default(),
        )
        .unwrap();
        scheduler.enable_encoder_store(server.socket(), DEPLOYMENT, 0, HOST);
        let (tx, rx) = crossbeam_channel::unbounded();
        let thread = thread::spawn(move || scheduler.run(rx));
        Self {
            handle: CommandSender::new(tx),
            thread,
            fake,
        }
    }

    /// Shuts the scheduler down and returns every batch it submitted.
    fn stop(self) -> Vec<Batch> {
        self.handle.shutdown();
        assert!(!self.thread.join().unwrap(), "the engine loop died");
        self.fake.submitted()
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
fn encodings(batches: &[Batch], kind: MediaCall) -> Vec<(RequestId, Option<bool>)> {
    batches
        .iter()
        .filter_map(|batch| Some((batch, batch.media.as_ref()?)))
        .filter(|(_, media)| media.computation.call == kind)
        .flat_map(|(batch, media)| {
            (0..batch.calls.len()).map(move |index| {
                let id = media
                    .request(index)
                    .expect("a media batch stages each of its requests")
                    .request
                    .request_id;
                let output = batch
                    .host_outputs
                    .iter()
                    .find(|(key, _)| key.request_id == id)
                    .map(|(_, output)| output.import);
                (id, output)
            })
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
    let media = Arc::new(Media::new());
    let pools = mapping(
        vec![video_pool(1, 1, 1)],
        pool,
        &[("conditioning", "text encoder")],
    );
    let encoders = Encoders::start(video_fake(pools, &media), &server);
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

/// The identities an image rank reports of its encoders' outputs.
const IMAGE_ENCODERS: [(&str, &str); 2] = [("vision_features", "vit"), ("latent_features", "vae")];

/// A fake image worker whose rank maps `server`'s encoder pool `pool`: its
/// vision encoder computes its features alike for every draw and its latent
/// encoder samples its posterior, each writing rows of 64 bytes. Its media
/// calls `media` answers and its token calls the fake's default rules.
fn image_worker(pool: u64, media: &Arc<Media>) -> Arc<FakeBatches> {
    let mut rank = support::text_rank();
    *rank.image_encoders = BTreeMap::from([
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
    let text = support::text_pool(rank, 2, Some(testkit::EndOfText::after(4, 8)));
    let pools = mapping(text.inventory().to_vec(), pool, &IMAGE_ENCODERS);
    let media = Arc::clone(media);
    Arc::new(
        FakeBatches::new(pools)
            .with_end(testkit::EndOfText::after(4, 8))
            .answering(Arc::new(move |batch| {
                if batch.media.is_some() {
                    media.answer(batch)
                } else {
                    ipc::execution::Outcome::Success(testkit::completed(
                        batch,
                        testkit::Tokens::Draw,
                        testkit::Extent::Stated,
                    ))
                }
            }))
            .with_media(Arc::new(Media::serves)),
    )
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
/// whose input image `image` (a digest) an encoder encodes with its latent
/// and its vision encoder, then text.
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
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(seed),
            ..SamplingParams::default()
        },
        max_und_tokens: 8,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache,
        generated_token_ids: Vec::new(),
        adapter: 0,
    }
}

/// Runs `request` to its end on an epoch of its own, whose encoder cache
/// starts empty, over `server`'s pool `pool`; returns its events and the
/// batches the epoch submitted.
fn run_image(
    server: &StorageServer,
    pool: u64,
    request: GenerationRequest,
) -> (Vec<Event>, Vec<Batch>) {
    let media = Arc::new(Media::new());
    let epoch = Encoders::start(image_worker(pool, &media), server);
    let id = request.request_id;
    let mut events = epoch.handle.submit(request).unwrap();
    let read = drain(&mut events);
    assert!(
        matches!(read.last(), Some(Event::Finished { .. })),
        "request {id:?} finished: {read:?}"
    );
    (read, epoch.stop())
}

/// How request `id`'s input image encodings treated the host encoder pool:
/// its latent encoding's, then its vision encoding's.
fn input_encodings(batches: &[Batch], id: u64) -> (Option<bool>, Option<bool>) {
    let of = |kind| {
        encodings(batches, kind)
            .into_iter()
            .find(|(request, _)| *request == RequestId(id))
            .unwrap_or_else(|| panic!("request {id} encodes its input image by {kind:?}"))
            .1
    };
    (of(MediaCall::LatentEncoding), of(MediaCall::VisionEncoding))
}

/// An input image's encodings that no epoch stored are computed and
/// written to the pool; another epoch reads them instead of encoding. The
/// vision features follow the image alone, and the latent features sample
/// their posterior at the encoding's draw under a seed of zero, so a request
/// of another sampling seed reads both. Another image computes both, and a
/// request whose cache policy neither reads nor writes cached outputs
/// encodes without the pool; every request finishes.
#[test]
#[ignore = "input image: no device layout for a context prefill with a vision block (T2)"]
fn an_image_encoding_is_computed_when_missing_and_read_on_another_epoch() {
    let mut server = StorageServer::start(1 << 20);
    let pool = server.encoder_pool("sim", PAGE_BYTES, 64);
    // Whether a request reads and writes cached outputs.
    let shared = (true, true);
    let uncached = (false, false);
    // Each request runs on an epoch of its own, once the one before
    // finished, so the outputs it wrote were committed before the next
    // looks them up and no epoch's encoder cache serves another's request.
    let runs = [
        (1, 5, 5, shared, (Some(false), Some(false))),
        (2, 5, 5, shared, (Some(true), Some(true))),
        (3, 5, 6, shared, (Some(true), Some(true))),
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
