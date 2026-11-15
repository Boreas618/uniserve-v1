//! The condition readers against the outputs of the libraries whose contracts
//! they implement.
//!
//! The fixtures under `fixtures/read/` were written by its `generate.py` with
//! the library versions its metadata pins: Pillow's decode, EXIF orientation
//! and LANCZOS resize for the images, PyAV's decode and torchaudio's resample
//! for the tracks, and FFmpeg 8.1's command-line tool for the reference
//! videos' frames. Each kind's `cases.txt` names its cases and their
//! transforms; the media and each case's expected bytes are embedded here
//! from the files beside it.

// Always true in an integration test crate; it makes clippy treat the helpers
// below as test code, where a failed unwrap is the test's failure report.
#![cfg(test)]

use std::io::Write as _;

use media::read::ConditionSource;
use media::{
    AudioTransform, ConditionProducts, ConditionRead, ImageTransform, VideoClip, read_audio,
    read_conditions, read_image, read_video,
};

/// One embedded case: its name, its media and its expected output.
type Embedded = (&'static str, &'static [u8], &'static [u8]);

/// The case `$name` of the fixture kind `$kind`: its media file `$media` and
/// its expected output `$name.out`.
macro_rules! case {
    ($kind:literal, $name:literal, $media:literal) => {
        (
            $name,
            include_bytes!(concat!("fixtures/read/", $kind, "/", $media)).as_slice(),
            include_bytes!(concat!("fixtures/read/", $kind, "/", $name, ".out")).as_slice(),
        )
    };
}

const IMAGE_MANIFEST: &str = include_str!("fixtures/read/images/cases.txt");
const IMAGES: &[Embedded] = &[
    case!("images", "pattern", "pattern.png"),
    case!("images", "crop", "crop.png"),
    case!("images", "same", "same.png"),
    case!("images", "rotated", "rotated.jpeg"),
    case!("images", "upside", "upside.jpeg"),
    case!("images", "mirrored", "mirrored.jpeg"),
];

const AUDIO_MANIFEST: &str = include_str!("fixtures/read/audio/cases.txt");
const AUDIO: &[Embedded] = &[
    case!("audio", "stereo", "stereo.wav"),
    case!("audio", "mono", "mono.wav"),
    case!("audio", "noop", "noop.wav"),
];

const VIDEO_MANIFEST: &str = include_str!("fixtures/read/video/cases.txt");
const VIDEOS: &[Embedded] = &[
    case!("video", "scaled", "clip.mp4"),
    case!("video", "native", "clip.mp4"),
    case!("video", "soundtrack", "soundtrack.mp4"),
];

/// The soundtrack of `soundtrack.mp4`: its transform's fields and the
/// expected `[samples, 2]` floats.
const SOUNDTRACK_TRANSFORM: &str = include_str!("fixtures/read/video/soundtrack.audio.txt");
const SOUNDTRACK: &[u8] = include_bytes!("fixtures/read/video/soundtrack.audio");

/// Each manifest line's name, file and numeric fields, paired with its
/// embedded case; every case of the manifest is embedded and no other.
fn cases(manifest: &str, embedded: &[Embedded]) -> Vec<(Embedded, Vec<usize>)> {
    let cases: Vec<(Embedded, Vec<usize>)> = manifest
        .lines()
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let case = *embedded
                .iter()
                .find(|(name, _, _)| *name == fields[0])
                .unwrap_or_else(|| panic!("the fixture case {} is not embedded", fields[0]));
            let values = fields[2..]
                .iter()
                .map(|value| value.parse().expect("a numeric field"))
                .collect();
            (case, values)
        })
        .collect();
    assert_eq!(
        cases.len(),
        embedded.len(),
        "the embedded cases are not the manifest's"
    );
    cases
}

struct ImageCase {
    name: &'static str,
    media: &'static [u8],
    transform: ImageTransform,
    tolerance: u8,
    expected: &'static [u8],
}

struct AudioCase {
    name: &'static str,
    media: &'static [u8],
    transform: AudioTransform,
    expected: Vec<f32>,
}

struct VideoCase {
    name: &'static str,
    media: &'static [u8],
    clip: VideoClip,
    expected: &'static [u8],
}

fn image_cases() -> Vec<ImageCase> {
    cases(IMAGE_MANIFEST, IMAGES)
        .into_iter()
        .map(|((name, media, expected), values)| ImageCase {
            name,
            media,
            transform: ImageTransform {
                resized: (values[0] as u32, values[1] as u32),
                window: (
                    values[2] as u32,
                    values[3] as u32,
                    values[4] as u32,
                    values[5] as u32,
                ),
            },
            tolerance: values[6] as u8,
            expected,
        })
        .collect()
}

/// An audio transform from its manifest fields:
/// `native_rate start source_samples samples rate`.
fn audio_transform(values: &[usize]) -> AudioTransform {
    AudioTransform {
        native_rate: values[0] as u32,
        start: values[1],
        source_samples: values[2],
        samples: values[3],
        rate: values[4] as u32,
    }
}

fn audio_cases() -> Vec<AudioCase> {
    cases(AUDIO_MANIFEST, AUDIO)
        .into_iter()
        .map(|((name, media, expected), values)| AudioCase {
            name,
            media,
            transform: audio_transform(&values),
            expected: floats(expected),
        })
        .collect()
}

fn video_cases() -> Vec<VideoCase> {
    cases(VIDEO_MANIFEST, VIDEOS)
        .into_iter()
        .map(|((name, media, expected), values)| VideoCase {
            name,
            media,
            clip: VideoClip {
                frame_rate: values[0] as u32,
                canvas: (values[1] as u32, values[2] as u32),
                frames: values[3] as u32,
                // These clips' pixel rows cover every frame they decode.
                vae_frames: values[3] as u32,
                start_frame: values[4] as u32,
            },
            expected,
        })
        .collect()
}

/// The float32 samples of little-endian bytes.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// A reference video as the server hands it to the reader: a seekable file.
fn video_file(media: &[u8]) -> std::fs::File {
    let mut file = tempfile::tempfile().expect("a scratch file");
    file.write_all(media).expect("writing the reference video");
    file
}

/// The clip of `case` as [`read_video`] decodes it on its own: what a
/// condition reading the same clip places in its rows.
fn decoded_frames(case: &VideoCase, clip: &VideoClip) -> Vec<u8> {
    let area = clip.canvas.0 as usize * clip.canvas.1 as usize;
    let mut frames = vec![0u8; clip.frames as usize * area * 3];
    read_video(&video_file(case.media), clip, &mut frames).expect("the clip decodes");
    frames
}

/// The largest per-channel difference between two RGB24 buffers.
fn difference(actual: &[u8], expected: &[u8]) -> u8 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual.abs_diff(*expected))
        .max()
        .unwrap_or(0)
}

/// The largest difference between two float sequences of one length.
fn largest_error(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}

/// The bound on a resampled sample's difference from torchaudio's: the
/// resampler implements torchaudio's mathematics in its own summation order,
/// and both round in float32, so the two differ by a few rounding steps of a
/// unit-amplitude signal.
const SAMPLE_TOLERANCE: f32 = 1e-5;

/// The bound on a decoded reference frame's per-channel difference from the
/// fixture's.
const VIDEO_TOLERANCE: u8 = 2;

#[test]
fn images_decode_and_fit_like_pillow() {
    for case in image_cases() {
        let (left, top, width, height) = case.transform.window;
        assert!(
            (left + width, top + height) <= case.transform.resized,
            "image case {} keeps a window outside its raster",
            case.name
        );
        let mut actual = vec![0u8; width as usize * height as usize * 3];
        read_image(case.media, &case.transform, &mut actual)
            .unwrap_or_else(|error| panic!("image case {} failed: {error}", case.name));
        if case.tolerance == 0 {
            assert_eq!(
                actual, case.expected,
                "image case {} differs from Pillow",
                case.name
            );
        } else {
            let difference = difference(&actual, case.expected);
            assert!(
                difference <= case.tolerance,
                "image case {} differs from Pillow by {difference} (> {}) per channel",
                case.name,
                case.tolerance
            );
        }
    }
}

#[test]
fn audio_decodes_and_resamples_like_torchaudio() {
    for case in audio_cases() {
        let mut actual = vec![0.0f32; case.transform.samples * 2];
        read_audio(case.media, &case.transform, &mut actual)
            .unwrap_or_else(|error| panic!("audio case {} failed: {error}", case.name));
        let error = largest_error(&actual, &case.expected);
        assert!(
            error <= SAMPLE_TOLERANCE,
            "audio case {} differs from torchaudio by {error}",
            case.name
        );
    }
}

#[test]
fn reference_videos_decode_to_the_frames_of_ffmpegs_filter_chain() {
    for case in video_cases() {
        let file = video_file(case.media);
        let mut actual = vec![0u8; case.expected.len()];
        read_video(&file, &case.clip, &mut actual)
            .unwrap_or_else(|error| panic!("video case {} failed: {error}", case.name));
        // The entropy decoder is exact; swscale's LANCZOS is deterministic for
        // one build, and another build (another CPU's vector code) may round a
        // sample differently.
        let difference = difference(&actual, case.expected);
        assert!(
            difference <= VIDEO_TOLERANCE,
            "video case {} differs from FFmpeg's frames by {difference} per channel",
            case.name
        );
    }
}

#[test]
fn conditions_fill_both_products_from_an_image_and_a_track() {
    let image = &image_cases()[0];
    let audio = &audio_cases()[0];
    let image_rows = image.transform.window.2 as usize * image.transform.window.3 as usize;
    let audio_rows = audio.transform.samples;
    let mut pixels = vec![0u8; image_rows * 3];
    let mut samples = vec![0.0f32; audio_rows * 2];
    let conditions = [
        ConditionRead {
            source: ConditionSource::Bytes(image.media),
            image: Some(image.transform),
            video: None,
            audio: None,
            pixels: 0..image_rows,
            samples: 0..0,
        },
        ConditionRead {
            source: ConditionSource::Bytes(audio.media),
            image: None,
            video: None,
            audio: Some(audio.transform),
            pixels: 0..0,
            samples: 0..audio_rows,
        },
    ];
    read_conditions(
        &conditions,
        &mut ConditionProducts {
            pixels: Some(&mut pixels),
            samples: Some(&mut samples),
        },
    )
    .unwrap();
    assert_eq!(pixels, image.expected);
    let error = largest_error(&samples, &audio.expected);
    assert!(
        error <= SAMPLE_TOLERANCE,
        "the condition's soundtrack differs from torchaudio by {error}"
    );
}

#[test]
fn conditions_take_a_reference_video_and_an_image_into_one_product() {
    let video = &video_cases()[0];
    let image = &image_cases()[1];
    let video_rows =
        video.clip.frames as usize * video.clip.canvas.0 as usize * video.clip.canvas.1 as usize;
    let image_rows = image.transform.window.2 as usize * image.transform.window.3 as usize;
    let file = video_file(video.media);
    let mut pixels = vec![0u8; (video_rows + image_rows) * 3];
    let conditions = [
        ConditionRead {
            source: ConditionSource::File(&file),
            image: None,
            video: Some(video.clip),
            audio: None,
            pixels: 0..video_rows,
            samples: 0..0,
        },
        ConditionRead {
            source: ConditionSource::Bytes(image.media),
            image: Some(image.transform),
            video: None,
            audio: None,
            pixels: video_rows..video_rows + image_rows,
            samples: 0..0,
        },
    ];
    let decoded = read_conditions(
        &conditions,
        &mut ConditionProducts {
            pixels: Some(&mut pixels),
            samples: None,
        },
    )
    .unwrap();
    let frames = decoded_frames(video, &video.clip);
    assert_eq!(pixels[..video_rows * 3], frames[..]);
    assert_eq!(pixels[video_rows * 3..], image.expected[..]);
    let video_decoded = decoded[0].as_ref().unwrap();
    assert_eq!(video_decoded.clip, video.clip);
    assert_eq!(video_decoded.pixels, frames);
    assert!(decoded[1].is_none());
}

#[test]
fn conditions_take_a_reference_videos_encoded_prefix_and_return_every_frame() {
    // The `native` case decodes five frames; a model may consume fewer of
    // them and still pack patches from the rest.
    let case = &video_cases()[1];
    assert_eq!(case.clip.frames, 5);
    let mut clip = case.clip;
    clip.vae_frames = 3;
    let area = clip.canvas.0 as usize * clip.canvas.1 as usize;
    let rows = clip.vae_frames as usize * area;
    let file = video_file(case.media);
    let mut pixels = vec![0u8; rows * 3];
    let decoded = read_conditions(
        &[ConditionRead {
            source: ConditionSource::File(&file),
            image: None,
            video: Some(clip),
            audio: None,
            pixels: 0..rows,
            samples: 0..0,
        }],
        &mut ConditionProducts {
            pixels: Some(&mut pixels),
            samples: None,
        },
    )
    .unwrap();
    let frames = decoded_frames(case, &clip);
    assert_eq!(pixels[..], frames[..rows * 3]);
    let decoded = decoded[0].as_ref().unwrap();
    assert_eq!(decoded.clip, clip);
    assert_eq!(decoded.pixels, frames);
    assert_eq!(decoded.pixels.len(), case.clip.frames as usize * area * 3);
}

#[test]
fn conditions_read_a_reference_videos_soundtrack_from_its_file() {
    // The video and its soundtrack come from one file: the frames decode on a
    // thread while the track decodes on another.
    let case = &video_cases()[2];
    let values: Vec<usize> = SOUNDTRACK_TRANSFORM
        .split_whitespace()
        .map(|value| value.parse().expect("a numeric field"))
        .collect();
    let transform = audio_transform(&values);
    let expected = floats(SOUNDTRACK);
    let area = case.clip.canvas.0 as usize * case.clip.canvas.1 as usize;
    let rows = case.clip.vae_frames as usize * area;
    let file = video_file(case.media);
    let mut pixels = vec![0u8; rows * 3];
    let mut samples = vec![0.0f32; transform.samples * 2];
    let decoded = read_conditions(
        &[ConditionRead {
            source: ConditionSource::File(&file),
            image: None,
            video: Some(case.clip),
            audio: Some(transform),
            pixels: 0..rows,
            samples: 0..transform.samples,
        }],
        &mut ConditionProducts {
            pixels: Some(&mut pixels),
            samples: Some(&mut samples),
        },
    )
    .unwrap();
    let frames = decoded_frames(case, &case.clip);
    assert_eq!(pixels, frames);
    let error = largest_error(&samples, &expected);
    assert!(
        error <= SAMPLE_TOLERANCE,
        "the soundtrack differs from torchaudio by {error}"
    );
    let video = decoded[0].as_ref().unwrap();
    assert_eq!(video.clip, case.clip);
    assert_eq!(video.pixels, frames);
}
