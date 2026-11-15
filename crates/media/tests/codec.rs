//! MP4 output of the encoders and the container assembly, read back through
//! the crate's own probe and decoders.
//!
//! These tests drive the crate's public boundary only: media units are
//! encoded on their own, several units are assembled with a soundtrack into a
//! request's MP4, and the result is probed ([`media::probe::probe_container`]),
//! its frames decoded ([`media::read_video`]) and its soundtrack decoded
//! ([`media::probe::decoded_track`]) through the FFmpeg libraries the crate
//! links. They assert what the media output must be: codec, pixel format,
//! canvas, frame count, frame rate, per-unit content in order, a stereo AAC
//! track covering the video's duration, and the refusals of malformed
//! assemblies.

// Always true in an integration test crate; it makes clippy treat the helpers
// below as test code, where a failed unwrap is the test's failure report.
#![cfg(test)]

use std::io::Write as _;
use std::time::Duration;

use media::probe::{AudioStream, VideoStream, decoded_track, probe_container};
use media::{
    Container, Error, MediaRunner, MuxSession, Rgb24Frames, VideoClip, audio_samples,
    encode_audio_track, encode_video_unit, read_video, require_codecs,
};

const FRAME_RATE: u32 = 24;
const AUDIO_RATE: u32 = 32_000;

/// The MP4 demuxer, by its FFmpeg name.
const MP4: &[&str] = &["mov"];

/// How long a probe of these small files may take.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// The video stream of an MP4 the crate wrote.
fn video_stream(payload: &[u8]) -> VideoStream {
    probe_container(payload, MP4, PROBE_TIMEOUT)
        .expect("the MP4 probes")
        .video
        .expect("the MP4 has a video stream")
}

/// The audio stream of an MP4 the crate wrote.
fn audio_stream(payload: &[u8]) -> AudioStream {
    probe_container(payload, MP4, PROBE_TIMEOUT)
        .expect("the MP4 probes")
        .audio
        .expect("the MP4 has an audio stream")
}

/// Every video frame of an MP4 of `frames` frames at `width` x `height` and
/// [`FRAME_RATE`], decoded to packed RGB24 at its own rate and canvas: a frame
/// off the stream's timeline would be repeated or dropped by the resampling.
fn decoded_frames(payload: &[u8], frames: u32, width: u32, height: u32) -> Vec<u8> {
    let mut file = tempfile::tempfile().expect("a scratch file");
    file.write_all(payload).expect("writing the MP4");
    let clip = VideoClip {
        frame_rate: FRAME_RATE,
        canvas: (width, height),
        frames,
        vae_frames: frames,
        start_frame: 0,
    };
    let mut pixels = vec![0u8; frames as usize * width as usize * height as usize * 3];
    read_video(&file, &clip, &mut pixels).expect("the frames decode");
    pixels
}

/// Decoded samples of each channel of the MP4's first audio stream.
fn decoded_samples(payload: &[u8]) -> usize {
    decoded_track(payload)
        .expect("the soundtrack decodes")
        .samples
}

/// `frames` frames of one flat gray level at `width` x `height`.
fn capture(frames: usize, width: u32, height: u32, shade: u8) -> Vec<u8> {
    vec![shade; frames * height as usize * width as usize * 3]
}

fn encode(width: u32, height: u32, frames: usize, shade: u8) -> Vec<u8> {
    encode_video_unit(
        Rgb24Frames {
            pixels: &capture(frames, width, height, shade),
            height,
            width,
        },
        FRAME_RATE,
    )
    .expect("a media unit encodes")
}

/// A random stereo track at a high level (about -6 dBFS, as the reference
/// probe of the encoder's bit rate used), whose encoded average the AAC
/// encoder drives to its configured bit rate; a tone or silence compresses far
/// below it.
fn noise(samples: usize) -> Vec<i16> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut pcm = Vec::with_capacity(samples * 2);
    for _ in 0..samples * 2 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // xorshift gives a full-range word; map it onto [-20000, 20000].
        pcm.push((((state >> 48) % 40_001) as i32 - 20_000) as i16);
    }
    pcm
}

#[test]
fn the_serving_encoders_are_available() {
    require_codecs().expect("the runtime provides libx264 and aac");
}

#[test]
fn each_canvas_encodes_to_a_valid_h264_mp4() {
    for (width, height, frames) in [(64u32, 48u32, 3usize), (96, 64, 4), (48, 80, 2)] {
        let payload = encode(width, height, frames, 90);
        let video = video_stream(&payload);
        assert_eq!(video.codec, "h264");
        assert_eq!(video.pixel_format, "yuv420p");
        assert_eq!((video.width, video.height), (width, height));
        assert_eq!(video.frame_rate, Some((FRAME_RATE, 1)));
        assert_eq!(video.frames, frames as u64);
        // Every frame decodes once at the stream's own rate, at the level it
        // was encoded at.
        let pixels = decoded_frames(&payload, frames as u32, width, height);
        let level = pixels.iter().map(|byte| u64::from(*byte)).sum::<u64>() / pixels.len() as u64;
        assert!(
            level.abs_diff(90) <= 3,
            "the frames decoded at level {level}"
        );
    }
}

#[test]
fn units_play_back_in_order_on_one_timeline() {
    let shades = [40u8, 120, 200];
    let frames = [17u32, 17, 5];
    let encoded: Vec<Vec<u8>> = shades
        .into_iter()
        .zip(frames)
        .map(|(shade, count)| encode(64, 48, count as usize, shade))
        .collect();

    let samples = 52_000;
    let silence = vec![0i16; samples * 2];
    let track = encode_audio_track(&silence, samples, AUDIO_RATE).expect("a soundtrack encodes");

    let mut container =
        Container::open(&encoded[0], FRAME_RATE, AUDIO_RATE).expect("the first unit templates it");
    let mut offset = 0;
    for payload in &encoded {
        offset = container.append(payload, offset).expect("a unit appends");
    }
    assert!(offset > 0, "the units advance the container's timeline");
    let payload = container.finish(&track).expect("the assembly finalizes");

    let video = video_stream(&payload);
    assert_eq!(video.codec, "h264");
    assert_eq!(video.pixel_format, "yuv420p");
    assert_eq!((video.width, video.height), (64, 48));
    assert_eq!(video.frame_rate, Some((FRAME_RATE, 1)));
    // The frame count of the assembled timeline is the sum of the units'.
    assert_eq!(video.frames, 39);

    let audio = audio_stream(&payload);
    assert_eq!(audio.codec, "aac");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.rate, AUDIO_RATE);

    // The units keep their order and content on one timeline at the frame
    // rate: decoded at that rate, every frame appears once, in order.
    let pixels = decoded_frames(&payload, 39, 64, 48);
    let frame_bytes = 48 * 64 * 3;
    assert_eq!(pixels.len(), 39 * frame_bytes);
    let levels: Vec<i32> = pixels
        .chunks_exact(frame_bytes)
        .map(|frame| {
            let sum: u64 = frame.iter().map(|byte| u64::from(*byte)).sum();
            (sum / frame_bytes as u64) as i32
        })
        .collect();
    for (index, shade) in shades.iter().enumerate() {
        let start: usize = frames[..index].iter().map(|count| *count as usize).sum();
        let stop = start + frames[index] as usize;
        for level in &levels[start..stop] {
            assert!(
                (level - i32::from(*shade)).abs() <= 3,
                "a frame of unit {index} decoded at level {level}, {shade} was encoded"
            );
        }
    }

    // The AAC track covers the video's duration: 52 000 samples plus at most
    // two frames of encoder padding.
    let decoded = decoded_samples(&payload);
    assert!(
        (samples..samples + 2 * media::AUDIO_FRAME_SAMPLES).contains(&decoded),
        "the soundtrack decoded to {decoded} samples, {samples} were encoded"
    );
}

#[test]
fn the_soundtrack_is_aac_stereo_at_the_model_rate_and_lasts_its_samples() {
    let samples = 52_000;
    let track =
        encode_audio_track(&noise(samples), samples, AUDIO_RATE).expect("a soundtrack encodes");
    let audio = audio_stream(&track);
    assert_eq!(audio.codec, "aac");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.rate, AUDIO_RATE);
    // 144 kb/s is the encoder's configured average. An MP4 records the
    // measured average instead, so the check uses a full-scale track, whose
    // measured average approaches the target (a tone compresses far below
    // it).
    let bit_rate = audio.bit_rate.expect("the MP4 states a bit rate");
    assert!(
        (120_000..=150_000).contains(&bit_rate),
        "the soundtrack's measured bit rate is {bit_rate}"
    );
    let duration = audio
        .duration
        .expect("the MP4 states a duration")
        .as_secs_f64();
    let expected = samples as f64 / f64::from(AUDIO_RATE);
    assert!(
        (duration - expected).abs()
            <= 2.0 * media::AUDIO_FRAME_SAMPLES as f64 / f64::from(AUDIO_RATE),
        "the soundtrack lasts {duration} s, its samples last {expected} s"
    );
    let decoded = decoded_samples(&track);
    assert!((samples..samples + 2 * media::AUDIO_FRAME_SAMPLES).contains(&decoded));
}

#[test]
fn audio_tracks_are_cut_or_padded_to_the_video_duration() {
    let samples = 3_000;
    let long = noise(5_000);
    let short = vec![0i16; 100 * 2];
    for (name, pcm) in [("long", long), ("short", short)] {
        let track =
            encode_audio_track(&pcm, samples, AUDIO_RATE).expect("the track is cut or padded");
        let audio = audio_stream(&track);
        assert_eq!(audio.codec, "aac");
        assert_eq!(audio.channels, 2);
        let decoded = decoded_samples(&track);
        assert!(
            (samples..samples + 2 * media::AUDIO_FRAME_SAMPLES).contains(&decoded),
            "the {name} soundtrack decoded to {decoded} samples"
        );
    }
}

#[test]
fn a_session_assembles_a_complete_request_and_refuses_the_rest() {
    let frames = 5u32;
    let encoded = encode(64, 48, frames as usize, 90);
    let samples = audio_samples(frames as usize, FRAME_RATE, AUDIO_RATE);
    let silence = vec![0i16; samples * 2];
    let track = encode_audio_track(&silence, samples, AUDIO_RATE).expect("a soundtrack encodes");

    let mut session = MuxSession::open(64, 48, FRAME_RATE, AUDIO_RATE, &[frames, frames])
        .expect("a valid session");
    assert_eq!(
        session.samples(),
        audio_samples(2 * frames as usize, FRAME_RATE, AUDIO_RATE)
    );

    // Finalization needs every unit and the audio.
    assert!(matches!(session.finish(), Err(Error::Invalid(_))));
    session
        .append(std::slice::from_ref(&encoded))
        .expect("the first unit");

    // An audio-only file is not a video unit.
    assert!(matches!(
        session.append(std::slice::from_ref(&track)),
        Err(Error::Invalid(_))
    ));

    // A unit of another canvas is not this request's.
    let mut wrong_canvas = MuxSession::open(64, 48, FRAME_RATE, AUDIO_RATE, &[frames, frames])
        .expect("a valid session");
    wrong_canvas
        .append(std::slice::from_ref(&encoded))
        .expect("the first unit");
    assert!(matches!(
        wrong_canvas.append(&[encode(96, 64, frames as usize, 90)]),
        Err(Error::Invalid(_))
    ));

    // More units than the request has are refused before any packet.
    let mut excess =
        MuxSession::open(64, 48, FRAME_RATE, AUDIO_RATE, &[frames]).expect("a valid session");
    assert!(matches!(
        excess.append(&[encoded.clone(), encoded.clone()]),
        Err(Error::Invalid(_))
    ));

    session
        .append(std::slice::from_ref(&encoded))
        .expect("the second unit");
    // The audio track arrives once, before finalization.
    assert!(matches!(session.finish(), Err(Error::Invalid(_))));
    session.set_audio(track.clone()).expect("the soundtrack");
    assert!(matches!(
        session.set_audio(track.clone()),
        Err(Error::Invalid(_))
    ));
    let payload = session.finish().expect("the complete assembly");
    let video = video_stream(&payload);
    assert_eq!(video.codec, "h264");
    assert_eq!(video.frames, 10);
    // The soundtrack is in the file because the request set it.
    let audio = audio_stream(&payload);
    assert_eq!(audio.codec, "aac");
    assert_eq!(audio.channels, 2);

    // No call follows finalization.
    assert!(matches!(session.append(&[]), Err(Error::Invalid(_))));
    assert!(matches!(
        session.set_audio(track.clone()),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(session.finish(), Err(Error::Invalid(_))));

    // A container is templated from a unit with a video stream.
    assert!(matches!(
        Container::open(&track, FRAME_RATE, AUDIO_RATE),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn a_runner_holds_the_assembly_of_a_named_request_only() {
    let runner = MediaRunner::new();
    assert!(matches!(
        runner.append_units(7, &[]),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(runner.finish_mux(7), Err(Error::Invalid(_))));
    let frames = 3u32;
    runner
        .open_mux(7, 64, 48, FRAME_RATE, AUDIO_RATE, &[frames])
        .expect("a valid assembly");
    assert!(matches!(
        runner.open_mux(7, 64, 48, FRAME_RATE, AUDIO_RATE, &[frames]),
        Err(Error::Invalid(_))
    ));
    let encoded = encode(64, 48, frames as usize, 70);
    runner.append_units(7, &[encoded]).expect("the unit");
    let samples = audio_samples(frames as usize, FRAME_RATE, AUDIO_RATE);
    let silence = vec![0i16; samples * 2];
    let track = encode_audio_track(&silence, samples, AUDIO_RATE).expect("a soundtrack");
    runner.set_audio(7, track).expect("the soundtrack");
    let payload = runner.finish_mux(7).expect("the complete assembly");
    assert!(!payload.is_empty());
    // The assembly is gone after finalization.
    assert!(matches!(runner.finish_mux(7), Err(Error::Invalid(_))));
}

#[test]
fn identical_calls_produce_identical_media() {
    let capture = capture(5, 64, 48, 77);
    let frames = Rgb24Frames {
        pixels: &capture,
        height: 48,
        width: 64,
    };
    assert_eq!(
        encode_video_unit(frames, FRAME_RATE).expect("a unit"),
        encode_video_unit(frames, FRAME_RATE).expect("a unit")
    );
    let pcm = noise(4_096);
    assert_eq!(
        encode_audio_track(&pcm, 2_048, AUDIO_RATE).expect("a soundtrack"),
        encode_audio_track(&pcm, 2_048, AUDIO_RATE).expect("a soundtrack")
    );
}
