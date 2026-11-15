//! Images (SD §8.1): a chat request's input images (SD §3.3 `VisionEncode`; §3.4 "Encoder
//! inputs"; §4.3; §7.1 "A prefill chunk"). Each image block is written whole by one prefill,
//! after its image is encoded, and a chunk that cannot hold a block ends before it; an image's
//! encoder output is a block of the buffer pool, so an encode waits until the buffer pool has room
//! for it; an output the buffer pool holds under the image's key is pinned and its encode
//! skipped, and one it does not hold is computed and committed under that key.
//!
//! Every chat request here has one image of [`ROWS`] cache rows at the middle of its prompt
//! ([`chat_image`]), and its encoder output takes [`UNITS`] units of the device buffers: the
//! harness's encoder states 64 bytes of features a row, and its buffers 256 bytes a unit.

use engine::batch::Phase;
use engine::scheduler::Message;
use proto::request::wire::WeightsInFlight;
use proto::request::{Event, GenerationRequest, ImageEncoder, ImageEncoderInput, RequestId};
use storage::Key;
use storage::key::{Digest, encoder_output};
use storage::runtime::{Completion, Request};

use crate::harness::{A, Deployment, Harness, chat_image, row_positions};
use crate::pause_flush_resize::flush;

/// The cache rows of every image of these cases.
const ROWS: u32 = 6;

/// The units of the device buffers an encoder output of [`ROWS`] rows takes.
const UNITS: u32 = 2;

/// The class of encoder outputs in a deployment of one cache group: the class after it.
const ENCODER_CLASS: u8 = 1;

/// The key of the vision encoder output of an image of `digest` computed by the weights of
/// `identity` (SD §4.3): `encoder_output` over the image's digest and no posterior draw, under
/// the identity the encoding group states (none unless the case states one).
fn output_key(identity: &str, digest: [u8; 32]) -> Key {
    let mut inputs = Digest::new(b"image-encoding");
    inputs.bytes(&digest);
    inputs.number(0);
    Key(encoder_output(
        ImageEncoder::Vision.output(),
        identity,
        &inputs.finish(),
    ))
}

/// Chat request `id` of a prompt of 12 tokens whose image is that of request `image`.
fn showing(id: u64, image: u64, max_tokens: usize) -> GenerationRequest {
    let mut request = chat_image(id, 12, ROWS, max_tokens);
    let shown = chat_image(image, 12, ROWS, 1);
    request.multimodal_inputs.images = shown.multimodal_inputs.images;
    request
}

/// The digest of `request`'s first image.
fn digest(request: &GenerationRequest) -> [u8; 32] {
    request.multimodal_inputs.images[0].digest
}

/// The index, among the batches the executor was handed, of request `id`'s first batch of
/// `phase`.
fn first_of(harness: &Harness, id: RequestId, phase: Phase) -> Option<usize> {
    let batches = harness.batches().into_iter();
    batches
        .enumerate()
        .find(|(_, batch)| batch.phase == phase && batch.requests.contains(&id))
        .map(|(at, _)| at)
}

/// The requests the buffer pool took after the first `skip` it took, of the encoder class.
fn of_outputs(harness: &Harness, skip: usize) -> Vec<Request> {
    let taken = harness.taken().into_iter().skip(skip);
    taken
        .filter(|request| match request {
            Request::Pin { class, .. }
            | Request::Allocate { class, .. }
            | Request::Commit { class, .. } => *class == ENCODER_CLASS,
            _ => false,
        })
        .collect()
}

/// A prompt's image block is written whole by one prefill row, after the encode that computes
/// its encoder output: the row holds the text before the image, the block's rows and the text
/// after it (SD §3.3 `Prefill`: its next image, if the chunk reaches one, is encoded; §7.1 "A
/// prefill chunk").
#[test]
fn an_image_block_is_written_whole_by_one_prefill_after_its_encode() {
    let mut harness = Deployment::tokens().images().build();
    let id = harness.submit(chat_image(1, 12, ROWS, 3));
    harness.settle(100);

    let encodes = harness.batches_of_phase(id, Phase::VisionEncode);
    assert_eq!(encodes.len(), 1);
    let prefills = harness.batches_of_phase(id, Phase::Prefill);
    assert_eq!(prefills.len(), 1);
    assert_eq!(
        row_positions(&prefills[0], id),
        (0..18).collect::<Vec<i32>>()
    );
    let encoded = first_of(&harness, id, Phase::VisionEncode).unwrap();
    assert!(encoded < first_of(&harness, id, Phase::Prefill).unwrap());
    assert_eq!(harness.tokens(id).len(), 3);
}

/// A chunk that cannot hold an image block whole ends before it, and the block is written
/// whole by the chunk that starts with it; the chunks together write the context in order, none
/// larger than the policy's chunk (SD §7.1 "A prefill chunk": an image block joins whole when
/// the chunk holds it or it is the chunk's first; `S/batching/next.rs:484-485`).
#[test]
fn a_chunk_ends_before_an_image_block_it_cannot_hold() {
    let mut harness = Deployment::tokens().images().chunk(8).build();
    // The image's block lies at cache positions 6..12 of an 18-row context.
    let id = harness.submit(chat_image(1, 12, ROWS, 2));
    harness.settle(100);

    let chunks: Vec<Vec<i32>> = harness
        .batches_of_phase(id, Phase::Prefill)
        .iter()
        .map(|batch| row_positions(batch, id))
        .collect();
    assert_eq!(chunks[0], (0..6).collect::<Vec<i32>>(), "{chunks:?}");
    assert!(
        chunks[1].starts_with(&(6..12).collect::<Vec<i32>>()),
        "{chunks:?}"
    );
    assert!(chunks.iter().all(|chunk| chunk.len() <= 8), "{chunks:?}");
    let written: Vec<i32> = chunks.concat();
    assert_eq!(written, (0..18).collect::<Vec<i32>>());
    assert_eq!(harness.tokens(id).len(), 2);
}

/// Every image a prompt's prefill reaches is encoded before it, one encode per image, and the
/// prefill writes each block at its place with the encoder output of its own image (SD §3.3
/// `VisionEncode`: one per image whose output its admission's `Pin` did not find).
#[test]
fn every_image_a_prefill_reaches_is_encoded_before_it() {
    let mut harness = Deployment::tokens().images().build();
    let mut request = chat_image(1, 12, ROWS, 2);
    let mut second = request.multimodal_inputs.images[0].clone();
    second.digest[31] = 1;
    second.position = 9;
    second.num_positions = 4;
    second.encoders = vec![ImageEncoderInput {
        encoder: ImageEncoder::Vision,
        num_cache_tokens: Some(4),
        max_cache_tokens: None,
        height: None,
        width: None,
    }];
    request.multimodal_inputs.images.push(second);
    let id = harness.submit(request);
    harness.settle(100);

    let encodes = harness.batches_of_phase(id, Phase::VisionEncode);
    let images: Vec<Option<std::sync::Arc<[u8]>>> = encodes
        .iter()
        .flat_map(|batch| batch.media.iter().map(|row| row.image.clone()))
        .collect();
    assert_eq!(images.len(), 2);
    let prefills = harness.batches_of_phase(id, Phase::Prefill);
    assert_eq!(prefills.len(), 1);
    let last_encode = harness
        .batches()
        .iter()
        .rposition(|batch| batch.phase == Phase::VisionEncode)
        .unwrap();
    assert!(last_encode < first_of(&harness, id, Phase::Prefill).unwrap());

    // The first block at 6..12, the text up to the second image at 15, its block at 15..19.
    let vision: Vec<(u32, u32, u32)> = prefills[0]
        .columns
        .vision
        .iter()
        .map(|block| (block.token, block.rows, block.feature))
        .collect();
    assert_eq!(vision, [(6, 6, 0), (15, 4, 1)]);
    assert_eq!(
        row_positions(&prefills[0], id),
        (0..22).collect::<Vec<i32>>()
    );
    assert_eq!(harness.tokens(id).len(), 2);
}

/// An image's encoder output is a block of the buffer pool: a request whose output the device
/// buffers have no room for waits in the queue, encodes nothing, and is admitted and encodes once
/// the request ahead of it has ended and its output can be evicted (SD §3.5 `fits`: the units of
/// each encoder output; §4.4 "Classes": encoder outputs are a sized class; §7.2 "Admission to the
/// cache").
#[test]
fn an_encode_waits_for_room_for_its_encoder_output() {
    let mut harness = Deployment::tokens().images().build();
    let buffers = harness.region("buffers");
    let (reply, resized) = crossbeam_channel::bounded(1);
    harness.apply(Message::Resize(buffers, UNITS, reply));
    harness.run_until(20, |_| resized.try_recv().is_ok());

    let first = harness.submit(chat_image(1, 12, ROWS, 6));
    let second = harness.submit(chat_image(2, 12, ROWS, 2));
    harness.run_until(30, |harness| harness.tokens(first).len() >= 2);
    assert!(harness.events(second).is_empty(), "no room for its output");
    assert!(
        harness
            .batches_of_phase(second, Phase::VisionEncode)
            .is_empty()
    );

    harness.settle(200);
    let ended = harness
        .sent()
        .position(|(of, event)| *of == first && matches!(event, Event::Finished { .. }));
    let scheduled = harness
        .sent()
        .position(|(of, event)| *of == second && matches!(event, Event::Scheduled { .. }));
    assert!(ended < scheduled, "it is admitted once the first has ended");
    assert_eq!(
        harness.batches_of_phase(second, Phase::VisionEncode).len(),
        1
    );
    assert_eq!(harness.tokens(first).len(), 6);
    assert_eq!(harness.tokens(second).len(), 2);
}

/// An image whose encoder output the buffer pool holds under the image's key is not encoded
/// again: the admission of a request that shows it pins the output, and its prefill reads it.
/// The first request that showed it computed it and committed it under that key with its rows
/// (SD §3.4 "Encoder inputs": an output that `Pinned` found needs no call; §4.3: the key of an
/// encoder output; "Writing": committed when `cache.write` holds).
#[test]
fn a_cached_image_skips_its_encode() {
    let mut harness = Deployment::tokens().images().build();
    let first = chat_image(1, 12, ROWS, 2);
    let key = output_key("", digest(&first));
    let first = harness.submit(first);
    harness.settle(100);
    let committed = of_outputs(&harness, 0);
    let [
        Request::Allocate {
            at: 0,
            count: 1,
            units,
            ..
        },
        Request::Commit {
            at: 0,
            key: of,
            tokens,
            ..
        },
    ] = &committed[..]
    else {
        panic!("{committed:?}");
    };
    assert_eq!((*units, *of, u32::from(*tokens)), (UNITS, key, ROWS));
    assert_eq!(
        harness.batches_of_phase(first, Phase::VisionEncode).len(),
        1
    );

    let skip = harness.taken().len();
    let second = harness.submit(showing(2, 1, 2));
    harness.settle(100);
    let pinned = of_outputs(&harness, skip);
    let [Request::Pin { at: 0, keys, .. }] = &pinned[..] else {
        panic!("{pinned:?}");
    };
    assert_eq!(keys, &[key]);
    assert!(
        harness
            .batches_of_phase(second, Phase::VisionEncode)
            .is_empty()
    );
    let prefill = harness.batches_of_phase(second, Phase::Prefill).remove(0);
    assert_eq!(prefill.columns.vision.len(), 1);
    assert_eq!(prefill.columns.vision[0].feature, 0);
    assert_eq!(harness.tokens(second).len(), 2);
}

/// A cached encoder output no request holds lies in the head of the device buffers' eviction
/// order, whose units the buffer pool's `available` counts, and a request that pins it takes it
/// out of that count. A request showing that image and a new one, while a running request holds
/// the rest of the buffers, therefore waits in the queue until the running request has ended, and
/// the buffer pool is asked for no admission it answers `Full` (SD §3.5 `fits`: an encoder output
/// whose `Pin` `cache` expects to find and that no request holds takes its size while it lies in
/// the installed head; §4.5 "Installing the order").
#[test]
fn a_cached_encoder_output_no_request_holds_takes_its_units_from_the_admission_that_pins_it() {
    let mut harness = Deployment::tokens().images().build();
    assert_eq!(harness.resize("buffers", 2 * UNITS), 2 * UNITS);
    let first = harness.submit(chat_image(1, 12, ROWS, 1));
    harness.settle(100);
    assert_eq!(harness.tokens(first).len(), 1);

    let running = harness.submit(chat_image(3, 12, ROWS, 24));
    harness.run_until(50, |harness| !harness.tokens(running).is_empty());
    // The image of the first request, cached, and a new one after it.
    let mut request = showing(2, 1, 2);
    let mut new = chat_image(4, 12, ROWS, 1)
        .multimodal_inputs
        .images
        .remove(0);
    new.position = 9;
    request.multimodal_inputs.images.push(new);
    let both = harness.submit(request);
    harness.settle(300);

    let full = harness.answers().into_iter().any(|answer| {
        matches!(
            answer,
            (
                Request::Allocate {
                    class: ENCODER_CLASS,
                    ..
                },
                Completion::Full { .. }
            )
        )
    });
    assert!(
        !full,
        "no admission the scheduler counted as fitting found no room"
    );
    let ended = harness
        .sent()
        .position(|(of, event)| *of == running && matches!(event, Event::Finished { .. }));
    let scheduled = harness
        .sent()
        .position(|(of, event)| *of == both && matches!(event, Event::Scheduled { .. }));
    assert!(
        ended < scheduled,
        "it is admitted once the running request has ended"
    );
    assert_eq!(harness.batches_of_phase(both, Phase::VisionEncode).len(), 1);
    assert_eq!(harness.tokens(running).len(), 24);
    assert_eq!(harness.tokens(both).len(), 2);
}

/// An encoder output the scheduler believes held but the buffer pool no longer holds (another
/// epoch evicted it) is computed again: the admission's `Pin` finds nothing, the output is
/// allocated, the image encoded, and the output committed under its key (SD §4.5: a wrong belief
/// costs one more round, never a wrong block; §3.4 "Encoder inputs").
#[test]
fn an_encoder_output_the_buffer_pool_does_not_hold_is_computed_and_committed() {
    let mut harness = Deployment::tokens().images().build();
    let first = chat_image(1, 12, ROWS, 2);
    let key = output_key("", digest(&first));
    harness.submit(first);
    harness.settle(100);
    let evicted = harness.elsewhere(&[Request::Evict { keys: vec![key] }]);
    assert_eq!(evicted, [Completion::Evicted { keys: vec![key] }]);

    let skip = harness.taken().len();
    let second = harness.submit(showing(2, 1, 2));
    harness.settle(100);
    let asked = of_outputs(&harness, skip);
    let [
        Request::Pin { keys, .. },
        Request::Allocate {
            at: 0,
            count: 1,
            units,
            ..
        },
        Request::Commit { key: of, .. },
    ] = &asked[..]
    else {
        panic!("{asked:?}");
    };
    assert_eq!((keys, *units, *of), (&vec![key], UNITS, key));
    assert_eq!(
        harness.batches_of_phase(second, Phase::VisionEncode).len(),
        1
    );
    let committed = harness.answers().into_iter().any(|answer| {
        matches!(answer, (Request::Commit { key: of, .. }, Completion::Committed) if of == key)
    });
    assert!(committed);
    assert_eq!(harness.tokens(second).len(), 2);
}

/// An encoder output is keyed with the identity of the weights that computed it, which the
/// group that encodes states: the same image is committed under another key by other weights, so
/// no request reads an output other weights computed (SD §4.3: `encoder_output(name, identity,
/// inputs)`; §2.1 `Capacity`, its encoder identity).
#[test]
fn an_encoder_output_is_keyed_with_the_identity_of_the_weights_that_computed_it() {
    let request = chat_image(1, 12, ROWS, 2);
    let keys: Vec<Key> = ["vision-weights-a", "vision-weights-b"]
        .into_iter()
        .map(|identity| {
            let mut harness = Deployment::tokens()
                .images()
                .encoder_identity(identity)
                .build();
            harness.submit(request.clone());
            harness.settle(100);
            let committed = of_outputs(&harness, 0);
            let [.., Request::Commit { key, .. }] = &committed[..] else {
                panic!("{committed:?}");
            };
            assert_eq!(*key, output_key(identity, digest(&request)));
            *key
        })
        .collect();
    assert_ne!(keys[0], keys[1]);
}

/// A weight update keys the encoder outputs computed after it by the new weights' identity, which
/// its `Flush` states for the group that encodes: the same image shown again is not read from the
/// output the old weights computed, and its new output is committed under the new identity (SD
/// §2.4 `Flush`, §4.3; E11: no request reads a block other weights computed).
#[test]
fn a_weight_update_keys_the_encoder_outputs_after_it_by_the_new_weights() {
    let mut harness = Deployment::tokens()
        .images()
        .encoder_identity("vision-weights-a")
        .build();
    let first = chat_image(1, 12, ROWS, 2);
    harness.submit(first.clone());
    harness.settle(100);
    let old = output_key("vision-weights-a", digest(&first));
    let committed = of_outputs(&harness, 0);
    assert!(
        matches!(&committed[..], [.., Request::Commit { key, .. }] if *key == old),
        "{committed:?}"
    );

    let seeds = (
        vec![Key([9; 32])],
        7,
        vec![(A, "vision-weights-b".to_owned())],
    );
    let flushed = flush(&mut harness, Some(seeds), WeightsInFlight::Keep);
    assert!(flushed.is_ok(), "{flushed:?}");
    harness.settle(100);

    let skip = harness.taken().len();
    let again = showing(2, 1, 2);
    let id = harness.submit(again.clone());
    harness.settle(100);
    // The scheduler knows no output under the new identity, so it pins none: the image is
    // encoded again and its output committed under the new weights' key.
    let new = output_key("vision-weights-b", digest(&again));
    assert_ne!(new, old);
    let asked = of_outputs(&harness, skip);
    let [Request::Allocate { .. }, Request::Commit { key, .. }] = &asked[..] else {
        panic!("{asked:?}");
    };
    assert_eq!(*key, new);
    assert_eq!(harness.batches_of_phase(id, Phase::VisionEncode).len(), 1);
    assert_eq!(harness.tokens(id).len(), 2);
}
