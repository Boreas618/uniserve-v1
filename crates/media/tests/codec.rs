//! MP4 output of the encoders and the container assembly, checked with the
//! environment's FFmpeg.
//!
//! These tests drive the crate's public boundary only: media units are
//! encoded on their own, several units are assembled with a soundtrack into a
//! request's MP4, and the result is probed and decoded by the external
//! `ffprobe`/`ffmpeg` binaries. They assert what the media checks of the
//! design assert (`specs/worker.md` §10): codec, canvas, frame count, frame
//! rate, per-unit content, a stereo AAC track covering the video's duration,
//! and the refusals of malformed captures and assemblies.

// Always true in an integration test crate; it makes clippy treat the helpers
// below as test code, where a failed unwrap is the test's failure report.
#![cfg(test)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use media::{
    Container, Error, MediaConfig, MediaRunner, MuxSession, Rgb24Frames, audio_samples,
    encode_audio_track, encode_video_unit, require_codecs,
};

const FRAME_RATE: u32 = 24;
const AUDIO_RATE: u32 = 32_000;

/// The external probe these tests read media back with, in the order the
/// deployment resolves it: its own name, the deployed runtime's `bin`, the
/// environment's static FFmpeg, then the loader's path.
fn probe_binary(name: &str) -> PathBuf {
    let variable = format!("UNISERVE_{}", name.to_uppercase());
    if let Some(path) = std::env::var_os(&variable) {
        return PathBuf::from(path);
    }
    let mut candidates = Vec::new();
    if let Some(directory) = std::env::var_os("UNISERVE_FFMPEG_DIR") {
        candidates.push(PathBuf::from(directory).join("bin").join(name));
    }
    candidates.push(PathBuf::from("/opt/uv/ffmpeg-8.1.2/bin").join(name));
    candidates.push(PathBuf::from(name));
    candidates
        .into_iter()
        .find(|path| path.is_file() || path.components().count() == 1)
        .expect("a media tool")
}

fn ffprobe() -> PathBuf {
    probe_binary("ffprobe")
}

fn ffmpeg() -> PathBuf {
    let probe = probe_binary("ffprobe");
    if probe.components().count() == 1 {
        // The probe came from the loader's path, so the decoder is beside it
        // there.
        probe_binary("ffmpeg")
    } else {
        probe.with_file_name("ffmpeg")
    }
}

/// A scratch file of this test binary's own, under cargo's test temp
/// directory, removed when the test ends.
struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(name: &str, bytes: &[u8]) -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let directory = Path::new(env!("CARGO_TARGET_TMPDIR"));
        std::fs::create_dir_all(directory).expect("the test temp directory");
        let path = directory.join(format!("{name}-{}-{unique}.mp4", std::process::id()));
        std::fs::write(&path, bytes).expect("writing a scratch media file");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Runs a media tool and returns its standard output, failing with the tool's
/// message on a failed exit.
fn run(program: &Path, arguments: &[&str]) -> Vec<u8> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("running {}: {error}", program.display()));
    assert!(
        output.status.success(),
        "{} {:?} failed: {}",
        program.display(),
        arguments,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    output.stdout
}

/// The `key=value` entries of one stream, as `ffprobe` prints them.
fn stream_entries(file: &Path, stream: &str, entries: &str) -> BTreeMap<String, String> {
    let output = run(
        &ffprobe(),
        &[
            "-v",
            "error",
            "-select_streams",
            stream,
            "-show_entries",
            &format!("stream={entries}"),
            "-of",
            "default=noprint_wrappers=1",
            file.to_str().expect("a UTF-8 scratch path"),
        ],
    );
    let text = String::from_utf8(output).expect("ffprobe writes UTF-8");
    text.lines()
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect()
}

/// Presentation times, in seconds, of the decoded video frames.
fn frame_times(file: &Path) -> Vec<f64> {
    let output = run(
        &ffprobe(),
        &[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "csv=p=0",
            file.to_str().expect("a UTF-8 scratch path"),
        ],
    );
    let text = String::from_utf8(output).expect("ffprobe writes UTF-8");
    text.lines()
        .filter(|line| !line.trim().is_empty())
        // `csv=p=0` prints one line per frame and adds a field for each
        // side-data entry the frame carries, which can leave a trailing empty
        // column; the presentation time is the first field.
        .map(|line| {
            line.split(',')
                .next()
                .expect("a line has a first field")
                .trim()
                .parse::<f64>()
                .expect("ffprobe prints a frame's presentation time")
        })
        .collect()
}

/// Decodes every video frame as packed RGB24.
fn decode_rgb24(file: &Path) -> Vec<u8> {
    run(
        &ffmpeg(),
        &[
            "-v",
            "error",
            "-i",
            file.to_str().expect("a UTF-8 scratch path"),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
    )
}

/// Decoded samples of a file's first audio stream.
fn decoded_samples(file: &Path) -> usize {
    let output = run(
        &ffmpeg(),
        &[
            "-v",
            "error",
            "-i",
            file.to_str().expect("a UTF-8 scratch path"),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "2",
            "pipe:1",
        ],
    );
    output.len() / 4
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
        let file = Scratch::new("canvas", &payload);
        let video = stream_entries(
            file.path(),
            "v:0",
            "codec_name,width,height,avg_frame_rate,nb_frames,pix_fmt",
        );
        assert_eq!(video.get("codec_name").map(String::as_str), Some("h264"));
        assert_eq!(video.get("pix_fmt").map(String::as_str), Some("yuv420p"));
        let expected_width = width.to_string();
        let expected_height = height.to_string();
        assert_eq!(
            video.get("width").map(String::as_str),
            Some(expected_width.as_str())
        );
        assert_eq!(
            video.get("height").map(String::as_str),
            Some(expected_height.as_str())
        );
        assert_eq!(
            video.get("avg_frame_rate").map(String::as_str),
            Some("24/1")
        );
        let expected_frames = frames.to_string();
        assert_eq!(
            video.get("nb_frames").map(String::as_str),
            Some(expected_frames.as_str())
        );
        let times = frame_times(file.path());
        assert_eq!(times.len(), frames);
        assert!(times.windows(2).all(|pair| pair[1] > pair[0]));
        assert!((times[frames - 1] - (frames - 1) as f64 / f64::from(FRAME_RATE)).abs() < 1e-6);
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
    let file = Scratch::new("assembled", &payload);

    let video = stream_entries(
        file.path(),
        "v:0",
        "codec_name,width,height,avg_frame_rate,nb_frames,pix_fmt",
    );
    assert_eq!(video.get("codec_name").map(String::as_str), Some("h264"));
    assert_eq!(video.get("pix_fmt").map(String::as_str), Some("yuv420p"));
    assert_eq!(video.get("width").map(String::as_str), Some("64"));
    assert_eq!(video.get("height").map(String::as_str), Some("48"));
    assert_eq!(
        video.get("avg_frame_rate").map(String::as_str),
        Some("24/1")
    );
    // The frame count of the assembled timeline is the sum of the units'.
    assert_eq!(video.get("nb_frames").map(String::as_str), Some("39"));

    let audio = stream_entries(file.path(), "a:0", "codec_name,channels,sample_rate");
    assert_eq!(audio.get("codec_name").map(String::as_str), Some("aac"));
    assert_eq!(audio.get("channels").map(String::as_str), Some("2"));
    assert_eq!(audio.get("sample_rate").map(String::as_str), Some("32000"));

    // The units keep their order and content on one monotonic timeline that
    // ends at the last frame's presentation time.
    let times = frame_times(file.path());
    assert_eq!(times.len(), 39);
    assert!(times.windows(2).all(|pair| pair[1] > pair[0]));
    assert!((times[38] - 38.0 / f64::from(FRAME_RATE)).abs() < 1e-6);

    let pixels = decode_rgb24(file.path());
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
    let decoded = decoded_samples(file.path());
    assert!(
        (samples..samples + 2 * media::AUDIO_FRAME_SAMPLES).contains(&decoded),
        "the soundtrack decoded to {decoded} samples, {samples} were planned"
    );
}

#[test]
fn the_soundtrack_is_aac_stereo_at_the_model_rate_and_planned_duration() {
    let samples = 52_000;
    let track =
        encode_audio_track(&noise(samples), samples, AUDIO_RATE).expect("a soundtrack encodes");
    let file = Scratch::new("soundtrack", &track);
    let audio = stream_entries(
        file.path(),
        "a:0",
        "codec_name,channels,sample_rate,bit_rate,duration",
    );
    assert_eq!(audio.get("codec_name").map(String::as_str), Some("aac"));
    assert_eq!(audio.get("channels").map(String::as_str), Some("2"));
    assert_eq!(audio.get("sample_rate").map(String::as_str), Some("32000"));
    // 144 kb/s is the encoder's configured average. An MP4 records the
    // measured average instead, so the check uses a full-scale track, whose
    // measured average approaches the target (a tone compresses far below
    // it).
    let bit_rate: f64 = audio
        .get("bit_rate")
        .expect("ffprobe reports a bit rate")
        .parse()
        .expect("a numeric bit rate");
    assert!(
        (120_000.0..=150_000.0).contains(&bit_rate),
        "the soundtrack's measured bit rate is {bit_rate}"
    );
    let duration: f64 = audio
        .get("duration")
        .expect("ffprobe reports a duration")
        .parse()
        .expect("a numeric duration");
    let planned = samples as f64 / f64::from(AUDIO_RATE);
    assert!(
        (duration - planned).abs()
            <= 2.0 * media::AUDIO_FRAME_SAMPLES as f64 / f64::from(AUDIO_RATE),
        "the soundtrack lasts {duration} s, {planned} s were planned"
    );
    let decoded = decoded_samples(file.path());
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
        let file = Scratch::new(name, &track);
        let audio = stream_entries(file.path(), "a:0", "codec_name,channels,sample_rate");
        assert_eq!(audio.get("codec_name").map(String::as_str), Some("aac"));
        assert_eq!(audio.get("channels").map(String::as_str), Some("2"));
        let decoded = decoded_samples(file.path());
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
    assert!(matches!(session.finish(), Err(Error::Plan(_))));
    session
        .append(std::slice::from_ref(&encoded))
        .expect("the first unit");

    // An audio-only file is not a video unit.
    assert!(matches!(
        session.append(std::slice::from_ref(&track)),
        Err(Error::Plan(_))
    ));

    // A unit of another canvas is not this request's.
    let mut wrong_canvas = MuxSession::open(64, 48, FRAME_RATE, AUDIO_RATE, &[frames, frames])
        .expect("a valid session");
    wrong_canvas
        .append(std::slice::from_ref(&encoded))
        .expect("the first unit");
    assert!(matches!(
        wrong_canvas.append(&[encode(96, 64, frames as usize, 90)]),
        Err(Error::Plan(_))
    ));

    // More units than the request has are refused before any packet.
    let mut excess =
        MuxSession::open(64, 48, FRAME_RATE, AUDIO_RATE, &[frames]).expect("a valid session");
    assert!(matches!(
        excess.append(&[encoded.clone(), encoded.clone()]),
        Err(Error::Plan(_))
    ));

    session
        .append(std::slice::from_ref(&encoded))
        .expect("the second unit");
    // The audio track arrives once, before finalization.
    assert!(matches!(session.finish(), Err(Error::Plan(_))));
    session.set_audio(track.clone()).expect("the soundtrack");
    assert!(matches!(
        session.set_audio(track.clone()),
        Err(Error::Plan(_))
    ));
    let payload = session.finish().expect("the complete assembly");
    let file = Scratch::new("session", &payload);
    let video = stream_entries(file.path(), "v:0", "codec_name,nb_frames");
    assert_eq!(video.get("codec_name").map(String::as_str), Some("h264"));
    assert_eq!(video.get("nb_frames").map(String::as_str), Some("10"));
    // The soundtrack is in the file because the request set it.
    let audio = stream_entries(file.path(), "a:0", "codec_name,channels");
    assert_eq!(audio.get("codec_name").map(String::as_str), Some("aac"));
    assert_eq!(audio.get("channels").map(String::as_str), Some("2"));

    // No call follows finalization.
    assert!(matches!(session.append(&[]), Err(Error::Plan(_))));
    assert!(matches!(
        session.set_audio(track.clone()),
        Err(Error::Plan(_))
    ));
    assert!(matches!(session.finish(), Err(Error::Plan(_))));

    // A container is templated from a unit with a video stream.
    assert!(matches!(
        Container::open(&track, FRAME_RATE, AUDIO_RATE),
        Err(Error::Plan(_))
    ));
}

#[test]
fn a_runner_holds_the_assembly_of_a_named_request_only() {
    let runner = MediaRunner::new(MediaConfig::default());
    assert!(matches!(runner.append_units(7, &[]), Err(Error::Plan(_))));
    assert!(matches!(runner.finish_mux(7), Err(Error::Plan(_))));
    let frames = 3u32;
    runner
        .open_mux(7, 64, 48, FRAME_RATE, AUDIO_RATE, &[frames])
        .expect("a valid assembly");
    assert!(matches!(
        runner.open_mux(7, 64, 48, FRAME_RATE, AUDIO_RATE, &[frames]),
        Err(Error::Plan(_))
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
    assert!(matches!(runner.finish_mux(7), Err(Error::Plan(_))));
}

#[test]
fn malformed_captures_are_refused() {
    let rgb24_frame = vec![0u8; 48 * 64 * 3];
    // Not a whole number of RGB24 frames.
    assert!(matches!(
        encode_video_unit(
            Rgb24Frames {
                pixels: &rgb24_frame[..rgb24_frame.len() - 1],
                height: 48,
                width: 64,
            },
            FRAME_RATE,
        ),
        Err(Error::Plan(_))
    ));
    assert!(matches!(
        encode_video_unit(
            Rgb24Frames {
                pixels: &rgb24_frame,
                height: 48,
                width: 64,
            },
            0,
        ),
        Err(Error::Plan(_))
    ));
    // Not interleaved stereo samples.
    assert!(matches!(
        encode_audio_track(&[0i16; 3], 1, AUDIO_RATE),
        Err(Error::Plan(_))
    ));
    assert!(matches!(
        encode_audio_track(&[0i16; 2], 0, AUDIO_RATE),
        Err(Error::Plan(_))
    ));
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
