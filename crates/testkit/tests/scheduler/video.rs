//! Video (SD §8.1): a video request's calls (SD §3.3, §7.3). A request runs a fixed chain of
//! media phases whose dependencies are the table of SD §3.3: a conditioned request reads its
//! conditions first; its text encoding computes the vision features of the conditions and its
//! latent preparation encodes the condition latents inside those calls, so it makes no
//! `VisionEncode` or `LatentEncode` call; it denoises step by step; it decodes in rounds of the
//! decoder group's width, every rank decoding its share, so a decode row names no rank; it encodes
//! each round and the audio, and muxes the encoded rounds into the artifact. Each call reads the
//! products its predecessors wrote, by their positions in the request's product table, every
//! product reserved at admission. Its conditioning is an encoder output keyed by what the text
//! encoder reads, so a repeated request finds it and makes no text encoding. It reports its
//! progress after every applied call, then its artifact, then its end; a request the deployment
//! cannot serve is rejected at its arrival.
//!
//! The harness states one rank per group and every stage at the group's slot count, so the
//! decoder's width is one media unit. Every step of every case checks the scheduler's side of A1
//! to A4.

use std::collections::HashSet;

use engine::batch::product;
use engine::batch::{MediaRow, Phase};
use proto::config::{BlockConfig, BlockDistribution, VIDEO_CODEC_BLOCK, VIDEO_DECODER_BLOCK};
use proto::request::{
    ConditionExtent, DiffusionRequest, EncoderInputs, Event, FinishReason, MediaKind,
    RejectionKind, RequestId,
};
use storage::Key;
use storage::key::encoder_output;
use storage::runtime::{Completion, Request};

use crate::harness::{A, B, Deployment, Harness, STEPS, VIDEO_PHASES, token, video};

/// The calls of request `id`, in the order their batches were submitted: each call's batch's
/// place in that order, its phase and its row.
fn calls(harness: &Harness, id: RequestId) -> Vec<(usize, Phase, MediaRow)> {
    let batches = harness.batches().into_iter().enumerate();
    batches
        .filter_map(|(at, batch)| {
            let row = batch.requests.iter().position(|of| *of == id)?;
            Some((at, batch.phase, batch.media.get(row)?.clone()))
        })
        .collect()
}

/// The calls of `phase` among `calls`: each one's batch's place and its row.
fn of(calls: &[(usize, Phase, MediaRow)], phase: Phase) -> Vec<(usize, &MediaRow)> {
    let calls = calls.iter().filter(|(_, of, _)| *of == phase);
    calls.map(|(at, _, row)| (*at, row)).collect()
}

/// The positions in its product table a call reads.
fn reads(row: &MediaRow) -> &[u32] {
    &row.results[..row.reads as usize]
}

/// The positions in its product table a call writes.
fn writes(row: &MediaRow) -> &[u32] {
    &row.results[row.reads as usize..]
}

/// Request `id` completed: its artifact, then `Finished` with `Completed`.
fn completed(harness: &Harness, id: RequestId) {
    let events = harness.events(id);
    let [.., Event::Artifact(_), Event::Finished { reason, .. }] = &events[..] else {
        panic!("request {} did not complete: {events:?}", id.0);
    };
    assert_eq!(*reason, FinishReason::Completed);
}

/// A reference request of one media unit (`ref2va`): an image the conditioner reads as two vision
/// tokens, a clip of three latent units with its soundtrack, and that soundtrack again as an
/// audio reference.
fn reference(id: u64) -> DiffusionRequest {
    let frame = 64 * 32 * 3;
    let mut request = video(id, 1);
    request.conditions = vec![
        ConditionExtent {
            pixel_bytes: frame,
            vision_patches: 8,
            vision_tokens: 2,
            latent_units: vec![2],
            ..ConditionExtent::default()
        },
        ConditionExtent {
            pixel_bytes: 39 * frame,
            audio_samples: 16_000,
            vision_patches: 16,
            vision_tokens: 4,
            latent_units: vec![10, 10, 4],
            audio_rows: 40,
        },
        ConditionExtent {
            audio_samples: 16_000,
            audio_rows: 40,
            ..ConditionExtent::default()
        },
    ];
    request.sampling.condition_rows = 106;
    request
}

/// An audio-reference request of one media unit (`fl2va` with a soundtrack alone): one audio
/// track, which no vision block presents and no latent unit encodes from pixels.
fn audio_reference(id: u64) -> DiffusionRequest {
    let mut request = video(id, 1);
    request.conditions = vec![ConditionExtent {
        audio_samples: 16_000,
        audio_rows: 40,
        ..ConditionExtent::default()
    }];
    request.sampling.condition_rows = 40;
    request
}

/// `request` naming the digest of what its text encoder reads, `text`, which keys its
/// conditioning (a request with no conditions has no other encoder input).
fn digested(mut request: DiffusionRequest, text: u8) -> DiffusionRequest {
    request.encoder_inputs = Some(Box::new(EncoderInputs {
        text: [text; 32],
        vision: None,
        video_units: Vec::new(),
        audio: None,
    }));
    request
}

/// A text request (`t2va`) of one media unit makes one call of each phase of its chain in the
/// order of SD §3.3, each reading what the call before it wrote: its text encoding writes its
/// conditioning, which its latent preparation reads; its steps denoise in order and the last
/// writes the video and audio latents, which the decoders read once every step is handed over;
/// each encode reads what its decode wrote, a muxing call hands over the encoded round, and the
/// final one reads nothing and returns the artifact. It reads no media and makes no vision or
/// latent encoding, and no row names a rank.
#[test]
fn a_text_request_makes_its_calls_in_order_each_reading_what_the_one_before_wrote() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit_video(video(1, 1));
    harness.settle(200);
    completed(&harness, id);

    let calls = calls(&harness, id);
    let counts = [
        (Phase::MediaRead, 0),
        (Phase::VisionEncode, 0),
        (Phase::LatentEncode, 0),
        (Phase::TextEncode, 1),
        (Phase::LatentPrepare, 1),
        (Phase::Denoise, STEPS as usize),
        (Phase::VideoDecode, 1),
        (Phase::AudioDecode, 1),
        (Phase::VideoEncode, 1),
        (Phase::AudioEncode, 1),
        (Phase::Mux, 2),
    ];
    for (phase, count) in counts {
        assert_eq!(of(&calls, phase).len(), count, "{phase:?}");
    }
    assert!(
        calls.iter().all(|(_, _, row)| row.rank.is_none()),
        "a video call's row names no rank"
    );

    let one = |phase| of(&calls, phase)[0];
    let (text, prepare) = (one(Phase::TextEncode), one(Phase::LatentPrepare));
    assert!(text.0 < prepare.0);
    assert_eq!(
        writes(text.1).len(),
        1,
        "the text encoding writes the conditioning"
    );
    assert_eq!(reads(prepare.1), writes(text.1), "the preparation reads it");

    let steps = of(&calls, Phase::Denoise);
    let numbered: Vec<u32> = steps.iter().map(|(_, row)| row.step).collect();
    assert_eq!(numbered, (0..STEPS).collect::<Vec<_>>());
    assert!(steps[0].0 > prepare.0);
    let (last_step, denoised) = steps[STEPS as usize - 1];
    assert!(
        steps[..STEPS as usize - 1]
            .iter()
            .all(|(_, row)| row.results.is_empty())
    );
    let latents = writes(denoised);
    assert_eq!(
        latents.len(),
        2,
        "the last step writes the video and audio latents"
    );

    let (video_decode, audio_decode) = (one(Phase::VideoDecode), one(Phase::AudioDecode));
    assert!(video_decode.0 > last_step && audio_decode.0 > last_step);
    assert_eq!(reads(video_decode.1), &latents[..1]);
    assert_eq!(reads(audio_decode.1), &latents[1..]);
    assert_eq!(video_decode.1.unit, 0);

    let (video_encode, audio_encode) = (one(Phase::VideoEncode), one(Phase::AudioEncode));
    assert!(video_encode.0 > video_decode.0 && audio_encode.0 > audio_decode.0);
    assert_eq!(reads(video_encode.1), &writes(video_decode.1)[..1]);
    assert_eq!(reads(audio_encode.1), writes(audio_decode.1));

    let muxes = of(&calls, Phase::Mux);
    let (handing, last) = (muxes[0], muxes[1]);
    assert!(handing.0 > video_encode.0);
    assert_eq!(reads(handing.1), writes(video_encode.1));
    assert!(last.0 > handing.0 && last.0 > audio_encode.0);
    assert!(
        last.1.results.is_empty(),
        "the final muxing call reads nothing"
    );

    // Every product is one position of its table, written by one call.
    let written: Vec<u32> = calls
        .iter()
        .flat_map(|(_, _, row)| writes(row).to_vec())
        .collect();
    let distinct: HashSet<u32> = written.iter().copied().collect();
    assert_eq!(distinct.len(), written.len(), "{written:?}");
}

/// A request of several media units decodes them in rounds of the decoder group's width (one
/// unit on the harness's one-rank groups): round `k` covers unit `k`, starts once round `k - 1`
/// completed and reads the splice tail of its last unit, which that round wrote; each round is
/// encoded once it is decoded and handed to the muxer once encoded, and the final muxing call
/// follows them all (SD §3.3 `VideoDecode`, `VideoEncode`, `Mux`).
#[test]
fn a_request_of_several_units_decodes_in_rounds_of_the_decoder_width_in_order() {
    const UNITS: u32 = 3;
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit_video(video(1, UNITS));
    harness.settle(300);
    completed(&harness, id);

    let calls = calls(&harness, id);
    let decodes = of(&calls, Phase::VideoDecode);
    let encodes = of(&calls, Phase::VideoEncode);
    let muxes = of(&calls, Phase::Mux);
    assert_eq!(decodes.len(), UNITS as usize);
    assert_eq!(encodes.len(), UNITS as usize);
    assert_eq!(muxes.len(), UNITS as usize + 1);

    for round in 0..UNITS as usize {
        let (decoded, decode) = decodes[round];
        let (encoded, encode) = encodes[round];
        assert_eq!(decode.unit, round as u32);
        assert_eq!(encode.unit, round as u32);
        assert!(encoded > decoded);
        assert!(muxes[round].0 > encoded);
        if round > 0 {
            let (before, previous) = decodes[round - 1];
            assert!(decoded > before);
            // It reads the latents and the tail of the unit before its own, which the round
            // before it wrote last.
            let tail = writes(previous).last().unwrap();
            assert_eq!(reads(decode).last(), Some(tail));
        }
    }
    let (last, row) = muxes[UNITS as usize];
    assert!(encodes.iter().all(|(encoded, _)| *encoded < last));
    assert!(row.results.is_empty());
}

/// A reference request reads its conditions first, alone, writing the condition pixels, the
/// condition samples and the vision pixels; its text encoding reads the vision pixels and writes
/// the conditioning, and its latent preparation reads the conditioning, the condition pixels and
/// the condition samples, in that order: it makes no vision or latent encoding of its own (SD §3.3
/// "A video request has no `VisionEncode` or `LatentEncode` call").
#[test]
fn a_reference_request_reads_its_conditions_and_encodes_them_inside_its_text_encoding_and_preparation()
 {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit_video(reference(1));
    harness.settle(200);
    completed(&harness, id);

    let calls = calls(&harness, id);
    assert!(of(&calls, Phase::VisionEncode).is_empty());
    assert!(of(&calls, Phase::LatentEncode).is_empty());
    let (read_at, read) = of(&calls, Phase::MediaRead)[0];
    assert_eq!(calls[0].1, Phase::MediaRead, "the read is its first call");
    assert!(reads(read).is_empty());
    let [pixels, samples, vision] = writes(read) else {
        panic!("{read:?}");
    };

    let (text_at, text) = of(&calls, Phase::TextEncode)[0];
    assert!(text_at > read_at);
    assert_eq!(reads(text), [*vision]);
    let [conditioning] = writes(text) else {
        panic!("{text:?}");
    };

    let (_, prepare) = of(&calls, Phase::LatentPrepare)[0];
    assert_eq!(reads(prepare), [*conditioning, *pixels, *samples]);
    assert!(writes(prepare).is_empty());
}

/// An audio-reference request reads only its condition samples: its text encoding reads no
/// vision pixels, and its latent preparation reads the conditioning and the samples.
#[test]
fn an_audio_reference_request_reads_only_its_samples_and_prepares_its_latents_from_them() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit_video(audio_reference(1));
    harness.settle(200);
    completed(&harness, id);

    let calls = calls(&harness, id);
    let (_, read) = of(&calls, Phase::MediaRead)[0];
    let [samples] = writes(read) else {
        panic!("{read:?}");
    };
    let (_, text) = of(&calls, Phase::TextEncode)[0];
    assert!(reads(text).is_empty(), "no vision block to read");
    let [conditioning] = writes(text) else {
        panic!("{text:?}");
    };
    let (_, prepare) = of(&calls, Phase::LatentPrepare)[0];
    assert_eq!(reads(prepare), [*conditioning, *samples]);
}

/// A video request is told it was scheduled before any progress, and then its progress after
/// every applied call, with the steps its trajectory completed (SD §7.3 "Progress and artifact"):
/// `preparing` after its text encoding, `denoising` after its preparation and each step while
/// steps remain, `decoding` after its last step and after every decode and encode, and
/// `finalizing` after each muxing call.
#[test]
fn a_video_request_is_scheduled_before_its_progress_which_follows_every_applied_call() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit_video(video(1, 1));
    harness.settle(200);

    let events = harness.events(id);
    let Some(Event::Scheduled {
        queued_at,
        scheduled_at,
    }) = events.first()
    else {
        panic!("{events:?}");
    };
    assert!(queued_at <= scheduled_at);
    let progress: Vec<(String, u32)> = events
        .iter()
        .filter_map(|event| match event {
            Event::MediaProgress {
                phase,
                completed_steps,
            } => Some((phase.clone(), *completed_steps)),
            _ => None,
        })
        .collect();
    let expected: Vec<(String, u32)> = [
        ("preparing", 0),
        ("denoising", 0),
        ("denoising", 1),
        ("denoising", 2),
        ("denoising", 3),
        ("decoding", 4),
        ("decoding", 4),
        ("decoding", 4),
        ("decoding", 4),
        ("decoding", 4),
        ("finalizing", 4),
        ("finalizing", 4),
    ]
    .into_iter()
    .map(|(phase, steps)| (phase.to_owned(), steps))
    .collect();
    assert_eq!(progress, expected);
}

/// The artifact reaches the daemon before the request's end, as the final muxing call published
/// it, even to a consumer that took no event (its window holds the request's events), and the
/// request ends `Completed` with its prompt as its usage (SD §7.3 "Progress and artifact").
#[test]
fn the_artifact_reaches_a_consumer_that_took_nothing_before_the_terminal() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit_video(video(1, 1));
    harness.settle(200);

    let events = harness.events(id);
    let [.., Event::Artifact(artifact), terminal] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(artifact.media_kind, MediaKind::Video);
    assert_eq!(artifact.content_type, "video/mp4");
    assert_eq!(artifact.media.bytes(), b"uniserve-fake-video request=1");
    let Event::Finished {
        reason,
        prompt_tokens,
        completion_tokens,
        ..
    } = terminal
    else {
        panic!("{events:?}");
    };
    assert_eq!(*reason, FinishReason::Completed);
    assert_eq!((*prompt_tokens, *completion_tokens), (40, 0));
}

/// A video request's conditioning is an encoder output keyed by the digest of what its text
/// encoder reads and the identity of the weights that compute it, which the text encoder's group
/// states (SD §3.3, §4.3): it is committed under that key once encoded, a repeated request
/// of the same inputs pins it at admission and makes no text encoding, its preparation reading
/// the pinned block, while a request of other inputs encodes its own and one that names no
/// digest keeps its own private.
#[test]
fn a_repeated_request_finds_its_conditioning_and_makes_no_text_encoding() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES])
        .encoder_identity("text-encoder")
        .build();
    let first = digested(video(1, 1), 1);
    let id = harness.submit_video(first.clone());
    harness.settle(200);
    completed(&harness, id);
    let committed = harness
        .answers()
        .into_iter()
        .find_map(|answer| match answer {
            (Request::Commit { key, .. }, Completion::Committed) => Some(key),
            _ => None,
        })
        .expect("the conditioning is committed under its key");
    // Its key covers what computed it: the identity of the text encoder's weights (SD §4.3).
    let expected = encoder_output(product::CONDITIONING, "text-encoder", &[1; 32]);
    assert_eq!(committed, Key(expected));

    let repeated = harness.submit_video(DiffusionRequest {
        request_id: RequestId(2),
        ..first
    });
    harness.settle(200);
    completed(&harness, repeated);
    let pinned = harness.answers().into_iter().any(|answer| {
        matches!(
            answer,
            (Request::Pin { keys, .. }, Completion::Pinned { runs, .. })
                if keys == [committed] && runs.len() == 1
        )
    });
    assert!(pinned, "its admission pins the conditioning");
    let found = calls(&harness, repeated);
    assert!(of(&found, Phase::TextEncode).is_empty(), "no text encoding");
    let (_, prepare) = of(&found, Phase::LatentPrepare)[0];
    assert_eq!(reads(prepare).len(), 1, "it reads the pinned conditioning");

    // Other inputs, and no digest: each encodes its own.
    let other = harness.submit_video(digested(video(3, 1), 2));
    let private = harness.submit_video(video(4, 1));
    harness.settle(300);
    for id in [other, private] {
        completed(&harness, id);
        assert_eq!(of(&calls(&harness, id), Phase::TextEncode).len(), 1);
    }
    let commits = harness
        .taken()
        .into_iter()
        .filter(|request| matches!(request, Request::Commit { .. }))
        .count();
    assert_eq!(commits, 2, "a conditioning no digest keys is not committed");
}

/// A repeated request's admission takes no units for the conditioning its `Pin` will find: it is
/// admitted beside the request that committed the conditioning and still holds it, when the
/// buffers have room for its other products and not for the conditioning a second time (SD §3.5
/// `fits`: the units the admission allocates; §4.4 "Pins, loads, clones and prefetches"; §7.2
/// "Admission to the cache").
#[test]
fn a_conditioning_its_admission_will_pin_takes_no_units_of_the_buffers() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    // A text request of one unit holds 49 units of the buffers, 4 of them its conditioning's
    // (`a_video_request_holds_every_product_from_admission_and_asks_for_no_block_ahead`): room
    // for one such request and the other products of a second.
    assert_eq!(harness.resize("buffers", 49 + 45), 49 + 45);
    let first = digested(video(1, 1), 1);
    let id = harness.submit_video(first.clone());
    harness.run_until(60, |harness| {
        harness
            .answers()
            .into_iter()
            .any(|answer| matches!(answer, (Request::Commit { .. }, Completion::Committed)))
    });

    // The first request stays on the device, holding its products and its conditioning.
    harness.fake.hold();
    let repeated = harness.submit_video(DiffusionRequest {
        request_id: RequestId(2),
        ..first
    });
    harness.run_until(20, |harness| {
        let events = harness.events(repeated);
        events
            .iter()
            .any(|event| matches!(event, Event::Scheduled { .. }))
    });
    assert!(harness.terminal(id).is_none());

    harness.fake.resume();
    harness.settle(300);
    completed(&harness, id);
    completed(&harness, repeated);
    let found = calls(&harness, repeated);
    assert!(of(&found, Phase::TextEncode).is_empty(), "no text encoding");
}

/// A decoder block that deals no units to its ranks decodes one unit per rank a round, whatever
/// its `units_per_rank` states, and a codec beside it that covers such a round encodes every unit
/// on the host that decoded it: the deployment is served, one unit a round (SD §3.3
/// `VideoDecode`, the round width; §7.1 "Startup checks"; §7.3 "Occupancy").
#[test]
fn a_decoder_that_deals_no_units_decodes_one_a_rank_and_a_codec_covering_that_round_serves_it() {
    let decoder = BlockConfig {
        units_per_rank: 2,
        ..BlockConfig::parallel(vec![0])
    };
    let codec = BlockConfig {
        distribution: Some(BlockDistribution::TemporalUnits),
        ..BlockConfig::parallel(vec![0])
    };
    let mut harness = Deployment::video(&[&VIDEO_PHASES])
        .block(0, VIDEO_DECODER_BLOCK, decoder)
        .block(0, VIDEO_CODEC_BLOCK, codec)
        .build();
    let id = harness.submit_video(video(1, 2));
    harness.settle(300);
    completed(&harness, id);

    let calls = calls(&harness, id);
    let decodes = of(&calls, Phase::VideoDecode);
    let units: Vec<u32> = decodes.iter().map(|(_, row)| row.unit).collect();
    assert_eq!(units, [0, 1], "one unit a round");
    assert_eq!(of(&calls, Phase::VideoEncode).len(), 2);
}

/// A decode follows the last denoising step on its group once that step is scheduled only when
/// every rank that decodes denoises too: its worker runs the step ahead of the decode, and the
/// decode reads the latents its own step wrote. A rank that decodes and does not denoise passes
/// the step's batch and receives its latents from another rank's step, so the decode then waits
/// for the last step's output (SD §3.3). While the last step is held on the device, a group whose
/// two ranks both denoise decodes, and one whose second rank only decodes does not.
#[test]
fn a_decode_waits_for_the_last_step_s_output_where_a_decoder_rank_does_not_denoise() {
    // The denoiser's block as a deployment names it (`nn::family::Stage::Transformer`).
    const DENOISER_BLOCK: &str = "transformer";
    let decoder = BlockConfig {
        distribution: Some(BlockDistribution::TemporalUnits),
        ..BlockConfig::parallel(vec![0, 1])
    };
    for (denoisers, follows) in [(vec![0, 1], true), (vec![0], false)] {
        let mut harness = Deployment::video(&[&VIDEO_PHASES])
            .ranks(0, 2)
            .block(0, VIDEO_DECODER_BLOCK, decoder.clone())
            .block(0, DENOISER_BLOCK, BlockConfig::parallel(denoisers))
            .build();
        let id = harness.submit_video(video(1, 2));

        // Every batch runs until the last step is submitted, which stays on the device.
        harness.fake.hold();
        let last = |harness: &mut Harness| {
            let submitted = harness.media_rows(id, Phase::Denoise).len() == STEPS as usize;
            if !submitted {
                harness.fake.release(usize::MAX);
            }
            submitted
        };
        harness.run_until(200, last);
        harness.run(5);
        let decoded = !harness.media_rows(id, Phase::VideoDecode).is_empty();
        assert_eq!(decoded, follows, "follows: {follows}");

        harness.fake.resume();
        harness.settle(300);
        completed(&harness, id);
        let calls = calls(&harness, id);
        let (step, _) = *of(&calls, Phase::Denoise).last().unwrap();
        assert!(of(&calls, Phase::VideoDecode)[0].0 > step);
    }
}

/// A latent preparation follows the text encoding on its group once the encoding is scheduled
/// only when every rank that prepares latents encodes text too: its worker runs the encode ahead
/// of the preparation, and every text encoder rank of the conditioning's process writes the same
/// rows into the conditioning's one run, so the preparation reads after its own rank's write. A
/// rank that prepares latents and encodes no text passes the encode's batch, and nothing orders
/// the first text encoder rank's write before its read, so the preparation then waits for the
/// encode's output (SD §3.3). While the encode is held on the device, a group whose two ranks
/// both encode text prepares, and one whose second rank only denoises does not.
#[test]
fn a_latent_preparation_waits_for_the_encode_s_output_where_a_denoising_rank_encodes_no_text() {
    // The blocks as a deployment names them (`nn::family::Stage::{Transformer, TextEncoder}`).
    const DENOISER_BLOCK: &str = "transformer";
    const TEXT_BLOCK: &str = "text_encoder";
    for (encoders, follows) in [(vec![0, 1], true), (vec![0], false)] {
        let mut harness = Deployment::video(&[&VIDEO_PHASES])
            .ranks(0, 2)
            .block(0, DENOISER_BLOCK, BlockConfig::parallel(vec![0, 1]))
            .block(0, TEXT_BLOCK, BlockConfig::parallel(encoders))
            .build();
        let id = harness.submit_video(video(1, 1));

        // The encode is submitted and stays on the device.
        harness.fake.hold();
        harness.run_until(20, |harness| {
            !harness.media_rows(id, Phase::TextEncode).is_empty()
        });
        harness.run(5);
        let prepared = !harness.media_rows(id, Phase::LatentPrepare).is_empty();
        assert_eq!(prepared, follows, "follows: {follows}");

        harness.fake.resume();
        harness.settle(200);
        completed(&harness, id);
    }
}

/// A video request holds every product block from its admission, one `Allocate` per declared
/// result at its position for its owner, sized from the bytes its captured shape states for the
/// product's kind, and asks for no block after it: the low watermark of SD §4.4 is a token
/// request's (SD §4.4 "Products", §7.3 "Media products").
#[test]
fn a_video_request_holds_every_product_from_admission_and_asks_for_no_block_ahead() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit_video(video(1, 1));
    harness.settle(200);
    completed(&harness, id);

    let owner = harness.owner(id, A).unwrap();
    let allocates: Vec<(u8, u32, u32, u32)> = harness
        .taken()
        .into_iter()
        .filter_map(|request| match request {
            Request::Allocate {
                owner: of,
                class,
                at,
                count,
                units,
            } if of == owner => Some((class, at, count, units)),
            _ => None,
        })
        .collect();
    // The text request declares one result of each product kind but the three conditions'
    // (kinds 3 to 9, in kind order). The harness's session numbers product kind `k` class
    // `1 + k`, after the encoder outputs' class, and its captured shapes state `256 * (k + 1)`
    // bytes for it, which take `k + 1` units of the buffers' 256 bytes.
    let expected: Vec<(u8, u32, u32, u32)> = (3..10)
        .zip(0..)
        .map(|(kind, at)| (1 + kind as u8, at, 1, kind + 1))
        .collect();
    assert_eq!(allocates, expected);
}

/// A request the deployment cannot serve is rejected at its arrival as invalid, naming why, and
/// the deployment serves on (SD §7.2 "Validation of a video request"): a video request on a
/// deployment of token groups, a conditioned request where no group reads conditions, and a
/// raster no captured denoising shape holds.
#[test]
fn unsupported_media_is_rejected_and_the_deployment_serves_on() {
    let rejected = |harness: &Harness, id: RequestId| match &harness.events(id)[..] {
        [
            Event::Rejected {
                kind: RejectionKind::Invalid,
                message,
            },
        ] => message.clone(),
        events => panic!("request {} was not rejected: {events:?}", id.0),
    };

    let mut harness = Deployment::tokens().build();
    let refused = harness.submit_video(video(1, 1));
    assert_eq!(
        rejected(&harness, refused),
        "worker does not provide the video media stages"
    );
    let served = harness.submit(token(2, 6, 3));
    harness.settle(100);
    assert_eq!(harness.tokens(served).len(), 3);

    let unread: Vec<Phase> = VIDEO_PHASES
        .into_iter()
        .filter(|phase| *phase != Phase::MediaRead)
        .collect();
    let mut harness = Deployment::video(&[&unread]).build();
    let conditioned = harness.submit_video(reference(1));
    assert_eq!(
        rejected(&harness, conditioned),
        "the deployment's denoiser takes no conditions"
    );
    let mut wide = video(2, 1);
    wide.sampling.width = 128;
    let wide = harness.submit_video(wide);
    assert_eq!(
        rejected(&harness, wide),
        "loaded media stages cannot hold the request: no captured denoising shape holds 121 \
         frames at 128x32 with 40 prompt rows and 0 condition rows"
    );
    let served = harness.submit_video(video(3, 1));
    harness.settle(200);
    completed(&harness, served);
}

/// A call that reads what a call on another group wrote waits for that call to complete, since
/// nothing else orders the write before the read; one on the same group follows it once it is
/// scheduled, its workers running the two in submission order (SD §3.3). While the batches are
/// held on the device, the latent preparation joins the text encoding on one group, and waits for
/// it on a denoiser of its own.
#[test]
fn a_call_reading_another_groups_product_waits_for_it_to_complete() {
    let others: Vec<Phase> = VIDEO_PHASES
        .into_iter()
        .filter(|phase| !matches!(phase, Phase::LatentPrepare | Phase::Denoise))
        .collect();
    let split: [&[Phase]; 2] = [&[Phase::LatentPrepare, Phase::Denoise], &others];
    for (deployment, apart) in [
        (Deployment::video(&[&VIDEO_PHASES]), false),
        (Deployment::video(&split), true),
    ] {
        let mut harness = deployment.build();
        let id = harness.submit_video(video(1, 1));
        harness.fake.hold();
        harness.run_until(20, |harness| {
            !harness.media_rows(id, Phase::TextEncode).is_empty()
        });
        harness.run(5);
        let prepared = !harness.media_rows(id, Phase::LatentPrepare).is_empty();
        assert_eq!(prepared, !apart, "apart: {apart}");

        harness.fake.resume();
        harness.settle(200);
        completed(&harness, id);
        if apart {
            let groups = |phase| {
                harness
                    .batches_of(id)
                    .into_iter()
                    .find(|batch| batch.phase == phase)
                    .unwrap()
                    .group
            };
            assert_eq!(groups(Phase::Denoise), A);
            assert_eq!(groups(Phase::TextEncode), B);
        }
    }
}
