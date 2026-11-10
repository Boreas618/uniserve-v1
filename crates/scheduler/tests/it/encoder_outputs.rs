//! Requests' encoder outputs in the host encoder pool, over the batch fake
//! and an in-process Storage Server: an output the pool does not hold is
//! computed and stored, and a request that reads the same inputs (another,
//! or the same one resumed, on another epoch) reads it from the pool
//! instead of computing it: a video request's encoder outputs.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ipc::execution::{Batch, Batches, Inventory};
use ipc::{EncoderPoolInfo, MediaCall, RequestId};
use scheduler::{CommandSender, EncoderInputs, Event, EventRx, Request, SchedulerConfig};
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
