//! The condition readers against the reference implementations' outputs.
//!
//! The fixtures under `tests/fixtures/read/` are produced by
//! `tests/fixtures/read/generate.py`: the images and tracks as the reference
//! stack decodes them (Pillow 12.3, PyAV/libavcodec 8.1.2, torchaudio 2.11)
//! and the reference video as today's external `ffmpeg` command decodes it.
//! Each kind's `cases.txt` names its cases and their plans; every case's
//! expected bytes are beside its media.

use std::path::{Path, PathBuf};

use super::{
    AudioPlan, ConditionProducts, ConditionRead, ConditionSource, ImagePlan, VideoClip, read_audio,
    read_conditions, read_image, read_video,
};

/// The committed reading fixtures.
fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/read")
}

/// One image case: its plan and the reference's RGB24 rows.
struct ImageCase {
    name: String,
    media: Vec<u8>,
    plan: ImagePlan,
    tolerance: u8,
    expected: Vec<u8>,
}

/// One audio case: its plan and the reference's `[samples, 2]` floats.
struct AudioCase {
    name: String,
    media: Vec<u8>,
    plan: AudioPlan,
    expected: Vec<f32>,
}

/// One reference video case: its clip and the reference's RGB24 frames.
struct VideoCase {
    name: String,
    file: PathBuf,
    clip: VideoClip,
    expected: Vec<u8>,
}

/// Reads a kind's manifest and its cases' media and expected output.
fn image_cases() -> Vec<ImageCase> {
    let directory = fixtures().join("images");
    let manifest = std::fs::read_to_string(directory.join("cases.txt")).unwrap();
    let mut cases = Vec::new();
    for line in manifest.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let name = fields[0].to_owned();
        let values: Vec<u32> = fields[2..]
            .iter()
            .map(|value| value.parse().unwrap())
            .collect();
        cases.push(ImageCase {
            media: std::fs::read(directory.join(fields[1])).unwrap(),
            plan: ImagePlan {
                resized: (values[0], values[1]),
                window: (values[2], values[3], values[4], values[5]),
            },
            tolerance: values[6] as u8,
            expected: std::fs::read(directory.join(format!("{name}.out"))).unwrap(),
            name,
        });
    }
    assert!(!cases.is_empty(), "the image fixtures are missing");
    cases
}

fn audio_cases() -> Vec<AudioCase> {
    let directory = fixtures().join("audio");
    let manifest = std::fs::read_to_string(directory.join("cases.txt")).unwrap();
    let mut cases = Vec::new();
    for line in manifest.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let name = fields[0].to_owned();
        let values: Vec<usize> = fields[2..]
            .iter()
            .map(|value| value.parse().unwrap())
            .collect();
        cases.push(AudioCase {
            media: std::fs::read(directory.join(fields[1])).unwrap(),
            plan: AudioPlan {
                native_rate: values[0] as u32,
                start: values[1],
                source_samples: values[2],
                samples: values[3],
                rate: values[4] as u32,
            },
            expected: floats(&std::fs::read(directory.join(format!("{name}.out"))).unwrap()),
            name,
        });
    }
    assert!(!cases.is_empty(), "the audio fixtures are missing");
    cases
}

fn video_cases() -> Vec<VideoCase> {
    let directory = fixtures().join("video");
    let manifest = std::fs::read_to_string(directory.join("cases.txt")).unwrap();
    let mut cases = Vec::new();
    for line in manifest.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let name = fields[0].to_owned();
        let values: Vec<u32> = fields[2..]
            .iter()
            .map(|value| value.parse().unwrap())
            .collect();
        cases.push(VideoCase {
            file: directory.join(fields[1]),
            clip: VideoClip {
                frame_rate: values[0],
                canvas: (values[1], values[2]),
                frames: values[3],
                // These clips' plans cover every frame they decode.
                vae_frames: values[3],
                start_frame: values[4],
            },
            expected: std::fs::read(directory.join(format!("{name}.out"))).unwrap(),
            name,
        });
    }
    assert!(!cases.is_empty(), "the video fixtures are missing");
    cases
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

/// The external `ffmpeg` the reference video's decode runs through.
///
/// `UNISERVE_FFMPEG` names it; otherwise the provisioned 8.1.2 build is used,
/// and a plain `ffmpeg` on `PATH` is the last resort. The video fixtures are
/// generated with the same binary.
fn ffmpeg() -> PathBuf {
    if let Some(path) = std::env::var_os("UNISERVE_FFMPEG") {
        return path.into();
    }
    let prefix =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/native/prefix/bin/ffmpeg");
    if prefix.exists() {
        return prefix;
    }
    PathBuf::from("ffmpeg")
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

#[test]
fn images_decode_and_fit_like_pillow() {
    for case in image_cases() {
        let (left, top, width, height) = case.plan.window;
        assert!(
            (left + width, top + height) <= case.plan.resized,
            "image case {} keeps a window outside its raster",
            case.name
        );
        let mut actual = vec![0u8; width as usize * height as usize * 3];
        read_image(&case.media, &case.plan, &mut actual)
            .unwrap_or_else(|error| panic!("image case {} failed: {error}", case.name));
        assert_eq!(
            actual.len(),
            case.expected.len(),
            "image case {} produced another extent",
            case.name
        );
        if case.tolerance == 0 {
            assert_eq!(
                actual, case.expected,
                "image case {} differs from Pillow",
                case.name
            );
        } else {
            let difference = difference(&actual, &case.expected);
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
fn images_refuse_a_product_of_another_extent() {
    let case = &image_cases()[0];
    let (_, _, width, height) = case.plan.window;
    let mut short = vec![0u8; width as usize * height as usize * 3 - 1];
    assert!(read_image(&case.media, &case.plan, &mut short).is_err());
    let mut full = vec![0u8; width as usize * height as usize * 3];
    let media = std::fs::read(fixtures().join("images/pattern.png")).unwrap();
    assert!(read_image(&media[..10], &case.plan, &mut full).is_err());
}

#[test]
fn audio_decodes_and_resamples_like_the_reference() {
    for case in audio_cases() {
        let mut actual = vec![0.0f32; case.plan.samples * 2];
        read_audio(&case.media, &case.plan, &mut actual)
            .unwrap_or_else(|error| panic!("audio case {} failed: {error}", case.name));
        assert_eq!(
            actual.len(),
            case.expected.len(),
            "audio case {} produced another extent",
            case.name
        );
        const TOLERANCE: f32 = 1e-5;
        let difference = actual
            .iter()
            .zip(&case.expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            difference <= TOLERANCE,
            "audio case {} differs from the reference by {difference}",
            case.name
        );
    }
}

#[test]
fn audio_refuses_a_track_at_another_rate_or_with_fewer_samples() {
    let case = &audio_cases()[0];
    let mut other_rate = case.plan;
    other_rate.native_rate = 48000;
    let mut out = vec![0.0f32; case.plan.samples * 2];
    assert!(read_audio(&case.media, &other_rate, &mut out).is_err());
    let mut too_long = case.plan;
    too_long.start = 100_000;
    assert!(read_audio(&case.media, &too_long, &mut out).is_err());
    let mut short = vec![0.0f32; case.plan.samples * 2 - 1];
    assert!(read_audio(&case.media, &case.plan, &mut short).is_err());
}

#[test]
fn reference_videos_decode_like_the_reference_command() {
    let ffmpeg = ffmpeg();
    for case in video_cases() {
        let file = std::fs::File::open(&case.file).unwrap();
        let mut actual = vec![0u8; case.expected.len()];
        read_video(&ffmpeg, &file, &case.clip, &mut actual)
            .unwrap_or_else(|error| panic!("video case {} failed: {error}", case.name));
        // swscale's LANCZOS and the entropy decoder are deterministic for one
        // build; another build may round a sample differently.
        let difference = difference(&actual, &case.expected);
        assert!(
            difference <= 2,
            "video case {} differs from the reference by {difference} per channel",
            case.name
        );
    }
}

#[test]
fn reference_videos_refuse_a_clip_the_video_does_not_hold() {
    let case = &video_cases()[0];
    let file = std::fs::File::open(&case.file).unwrap();
    // The video holds fewer frames after the start than the clip asks for.
    let mut beyond = case.clip;
    beyond.start_frame = 8;
    beyond.frames = 8;
    let mut out = vec![0u8; case.expected.len()];
    assert!(read_video(&ffmpeg(), &file, &beyond, &mut out).is_err());
    // A plan whose rows disagree with the decoded frames is refused.
    let mut wider = case.clip;
    wider.canvas = (case.clip.canvas.0 + 1, case.clip.canvas.1);
    assert!(read_video(&ffmpeg(), &file, &wider, &mut out).is_err());
    // A source that is not a video fails through FFmpeg's own error.
    let bogus = std::fs::File::open(fixtures().join("images/pattern.png")).unwrap();
    assert!(read_video(&ffmpeg(), &bogus, &case.clip, &mut out).is_err());
}

#[test]
fn conditions_fill_both_products_from_an_image_and_a_track() {
    let image = &image_cases()[0];
    let audio = &audio_cases()[0];
    let image_rows = image.plan.window.2 as usize * image.plan.window.3 as usize;
    let audio_rows = audio.plan.samples;
    let mut pixels = vec![0u8; image_rows * 3];
    let mut samples = vec![0.0f32; audio_rows * 2];
    let conditions = [
        ConditionRead {
            source: ConditionSource::Bytes(&image.media),
            image: Some(image.plan),
            video: None,
            audio: None,
            pixels: 0..image_rows,
            samples: 0..0,
        },
        ConditionRead {
            source: ConditionSource::Bytes(&audio.media),
            image: None,
            video: None,
            audio: Some(audio.plan),
            pixels: 0..0,
            samples: 0..audio_rows,
        },
    ];
    read_conditions(
        &ffmpeg(),
        &conditions,
        &mut ConditionProducts {
            pixels: Some(&mut pixels),
            samples: Some(&mut samples),
        },
    )
    .unwrap();
    assert_eq!(pixels, image.expected);
    assert_eq!(samples, audio.expected);
}

#[test]
fn conditions_take_a_reference_video_and_an_image_into_one_product() {
    let video = &video_cases()[0];
    let image = &image_cases()[1];
    let video_rows =
        video.clip.frames as usize * video.clip.canvas.0 as usize * video.clip.canvas.1 as usize;
    let image_rows = image.plan.window.2 as usize * image.plan.window.3 as usize;
    let file = std::fs::File::open(&video.file).unwrap();
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
            source: ConditionSource::Bytes(&image.media),
            image: Some(image.plan),
            video: None,
            audio: None,
            pixels: video_rows..video_rows + image_rows,
            samples: 0..0,
        },
    ];
    let decoded = read_conditions(
        &ffmpeg(),
        &conditions,
        &mut ConditionProducts {
            pixels: Some(&mut pixels),
            samples: None,
        },
    )
    .unwrap();
    assert_eq!(pixels[..video_rows * 3], video.expected[..]);
    assert_eq!(pixels[video_rows * 3..], image.expected[..]);
    let video_decoded = decoded[0].as_ref().unwrap();
    assert_eq!(video_decoded.clip, video.clip);
    assert_eq!(video_decoded.pixels, video.expected);
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
    let file = std::fs::File::open(&case.file).unwrap();
    let mut pixels = vec![0u8; rows * 3];
    let decoded = read_conditions(
        &ffmpeg(),
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
    assert_eq!(pixels[..], case.expected[..rows * 3]);
    let decoded = decoded[0].as_ref().unwrap();
    assert_eq!(decoded.clip, clip);
    assert_eq!(decoded.pixels, case.expected);
    assert_eq!(decoded.pixels.len(), case.clip.frames as usize * area * 3);
}

#[test]
fn conditions_refuse_an_encoded_prefix_longer_than_the_decoded_clip() {
    let case = &video_cases()[0];
    let mut clip = case.clip;
    clip.vae_frames = clip.frames + 1;
    let area = clip.canvas.0 as usize * clip.canvas.1 as usize;
    let mut pixels = vec![0u8; clip.vae_frames as usize * area * 3];
    let file = std::fs::File::open(&case.file).unwrap();
    assert!(
        read_conditions(
            &ffmpeg(),
            &[ConditionRead {
                source: ConditionSource::File(&file),
                image: None,
                video: Some(clip),
                audio: None,
                pixels: 0..clip.vae_frames as usize * area,
                samples: 0..0,
            }],
            &mut ConditionProducts {
                pixels: Some(&mut pixels),
                samples: None,
            },
        )
        .is_err()
    );
}

#[test]
fn conditions_refuse_rows_that_do_not_fill_their_products() {
    let image = &image_cases()[0];
    let rows = image.plan.window.2 as usize * image.plan.window.3 as usize;
    let conditions = [ConditionRead {
        source: ConditionSource::Bytes(&image.media),
        image: Some(image.plan),
        video: None,
        audio: None,
        pixels: 0..rows,
        samples: 0..0,
    }];
    // A product with one row too many is left unfilled.
    let mut wide = vec![0u8; (rows + 1) * 3];
    assert!(
        read_conditions(
            &ffmpeg(),
            &conditions,
            &mut ConditionProducts {
                pixels: Some(&mut wide),
                samples: None,
            },
        )
        .is_err()
    );
    // A product without room for every row is exceeded.
    let mut narrow = vec![0u8; (rows - 1) * 3];
    assert!(
        read_conditions(
            &ffmpeg(),
            &conditions,
            &mut ConditionProducts {
                pixels: Some(&mut narrow),
                samples: None,
            },
        )
        .is_err()
    );
    // A condition that needs a product the request does not hold.
    assert!(
        read_conditions(
            &ffmpeg(),
            &conditions,
            &mut ConditionProducts {
                pixels: None,
                samples: None,
            },
        )
        .is_err()
    );
    // Declared rows that disagree with the plan are refused before decoding.
    let mismatched = [ConditionRead {
        source: ConditionSource::Bytes(&image.media),
        image: Some(image.plan),
        video: None,
        audio: None,
        pixels: 0..rows + 1,
        samples: 0..0,
    }];
    let mut roomy = vec![0u8; (rows + 1) * 3];
    assert!(
        read_conditions(
            &ffmpeg(),
            &mismatched,
            &mut ConditionProducts {
                pixels: Some(&mut roomy),
                samples: None,
            },
        )
        .is_err()
    );
    // Two conditions that claim the same rows cannot both own them.
    let overlapping = [
        ConditionRead {
            source: ConditionSource::Bytes(&image.media),
            image: Some(image.plan),
            video: None,
            audio: None,
            pixels: 0..rows,
            samples: 0..0,
        },
        ConditionRead {
            source: ConditionSource::Bytes(&image.media),
            image: Some(image.plan),
            video: None,
            audio: None,
            pixels: 0..rows,
            samples: 0..0,
        },
    ];
    let mut product = vec![0u8; 2 * rows * 3];
    assert!(
        read_conditions(
            &ffmpeg(),
            &overlapping,
            &mut ConditionProducts {
                pixels: Some(&mut product),
                samples: None,
            },
        )
        .is_err()
    );
}
