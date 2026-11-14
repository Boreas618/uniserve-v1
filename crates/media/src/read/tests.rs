//! The condition readers against the reference implementations' outputs, and
//! their refusals of transforms the media cannot satisfy.
//!
//! The reference fixtures are generated once per test process by
//! `tests/fixtures/read/generate.py` in the interpreter
//! `UNISERVE_REFERENCE_PYTHON` names: the images and tracks as the reference
//! stack decodes them (Pillow 12.3, PyAV/libavcodec 8.1.2, torchaudio 2.11)
//! and the reference video as the external `ffmpeg` command decodes it. Each
//! kind's `cases.txt` names its cases and their transforms; every case's expected
//! bytes are beside its media. The comparisons are ignored by default and fail
//! when requested without the interpreter.
//!
//! The refusals need no reference: they read media made here (a PNG, a PCM WAV
//! and an MP4 this crate encodes) and always run.

use std::io::Write as _;
use std::path::PathBuf;

use crate::Error;

use super::{
    AudioTransform, ConditionProducts, ConditionRead, ConditionSource, ImageTransform, VideoClip,
    read_audio, read_conditions, read_image, read_video,
};

/// The reading fixtures, generated once per test process.
fn fixtures() -> PathBuf {
    static GENERATED: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    GENERATED
        .get_or_init(|| crate::reference::generate("tests/fixtures/read/generate.py"))
        .path()
        .to_path_buf()
}

/// One image case: its transform and the reference's RGB24 rows.
struct ImageCase {
    name: String,
    media: Vec<u8>,
    transform: ImageTransform,
    tolerance: u8,
    expected: Vec<u8>,
}

/// One audio case: its transform and the reference's `[samples, 2]` floats.
struct AudioCase {
    name: String,
    media: Vec<u8>,
    transform: AudioTransform,
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
            transform: ImageTransform {
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
            transform: AudioTransform {
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
                // These clips' pixel rows cover every frame they decode.
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

/// The external `ffmpeg` reference videos are decoded with: the one a deployment resolves.
fn ffmpeg() -> PathBuf {
    crate::MediaConfig::from_env().ffmpeg
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
#[ignore = "requires UNISERVE_REFERENCE_PYTHON: an interpreter with numpy, Pillow, torch and torchaudio"]
fn images_decode_and_fit_like_pillow() {
    for case in image_cases() {
        let (left, top, width, height) = case.transform.window;
        assert!(
            (left + width, top + height) <= case.transform.resized,
            "image case {} keeps a window outside its raster",
            case.name
        );
        let mut actual = vec![0u8; width as usize * height as usize * 3];
        read_image(&case.media, &case.transform, &mut actual)
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
#[ignore = "requires UNISERVE_REFERENCE_PYTHON: an interpreter with numpy, Pillow, torch and torchaudio"]
fn audio_decodes_and_resamples_like_the_reference() {
    for case in audio_cases() {
        let mut actual = vec![0.0f32; case.transform.samples * 2];
        read_audio(&case.media, &case.transform, &mut actual)
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
#[ignore = "requires UNISERVE_REFERENCE_PYTHON: an interpreter with numpy, Pillow, torch and torchaudio"]
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
#[ignore = "requires UNISERVE_REFERENCE_PYTHON: an interpreter with numpy, Pillow, torch and torchaudio"]
fn conditions_fill_both_products_from_an_image_and_a_track() {
    let image = &image_cases()[0];
    let audio = &audio_cases()[0];
    let image_rows = image.transform.window.2 as usize * image.transform.window.3 as usize;
    let audio_rows = audio.transform.samples;
    let mut pixels = vec![0u8; image_rows * 3];
    let mut samples = vec![0.0f32; audio_rows * 2];
    let conditions = [
        ConditionRead {
            source: ConditionSource::Bytes(&image.media),
            image: Some(image.transform),
            video: None,
            audio: None,
            pixels: 0..image_rows,
            samples: 0..0,
        },
        ConditionRead {
            source: ConditionSource::Bytes(&audio.media),
            image: None,
            video: None,
            audio: Some(audio.transform),
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
    // The resampler reproduces torchaudio's mathematics, not its float32
    // rounding: the reader's own acceptance is a per-sample bound.
    let difference = samples
        .iter()
        .zip(&audio.expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        difference <= 1e-5,
        "the condition's soundtrack differs from the reference by {difference}"
    );
}

#[test]
#[ignore = "requires UNISERVE_REFERENCE_PYTHON: an interpreter with numpy, Pillow, torch and torchaudio"]
fn conditions_take_a_reference_video_and_an_image_into_one_product() {
    let video = &video_cases()[0];
    let image = &image_cases()[1];
    let video_rows =
        video.clip.frames as usize * video.clip.canvas.0 as usize * video.clip.canvas.1 as usize;
    let image_rows = image.transform.window.2 as usize * image.transform.window.3 as usize;
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
            image: Some(image.transform),
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
#[ignore = "requires UNISERVE_REFERENCE_PYTHON: an interpreter with numpy, Pillow, torch and torchaudio"]
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
#[ignore = "requires UNISERVE_REFERENCE_PYTHON: an interpreter with numpy, Pillow, torch and torchaudio"]
fn conditions_read_a_reference_videos_soundtrack_from_its_file() {
    // The video and its soundtrack come from one file: the frames decode on a
    // thread while the track decodes on this one.
    let case = &video_cases()[2];
    let directory = fixtures().join("video");
    let values: Vec<usize> = std::fs::read_to_string(directory.join("soundtrack.audio.txt"))
        .unwrap()
        .split_whitespace()
        .map(|value| value.parse().unwrap())
        .collect();
    let transform = AudioTransform {
        native_rate: values[0] as u32,
        start: values[1],
        source_samples: values[2],
        samples: values[3],
        rate: values[4] as u32,
    };
    let expected = floats(&std::fs::read(directory.join("soundtrack.audio")).unwrap());
    let area = case.clip.canvas.0 as usize * case.clip.canvas.1 as usize;
    let rows = case.clip.vae_frames as usize * area;
    let file = std::fs::File::open(&case.file).unwrap();
    let mut pixels = vec![0u8; rows * 3];
    let mut samples = vec![0.0f32; transform.samples * 2];
    let decoded = read_conditions(
        &ffmpeg(),
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
    let frames_difference = difference(&pixels, &case.expected);
    assert!(
        frames_difference <= 2,
        "the soundtrack clip's frames differ by {frames_difference} per channel"
    );
    let samples_difference = samples
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        samples_difference <= 1e-5,
        "the soundtrack differs from the reference by {samples_difference}"
    );
    let video = decoded[0].as_ref().unwrap();
    assert_eq!(video.clip, case.clip);
    assert_eq!(video.pixels, case.expected);
}

/// A `width` x `height` RGB gradient, encoded here as a PNG.
fn synthetic_png(width: u32, height: u32) -> Vec<u8> {
    let image = image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x * 7) as u8, (y * 11) as u8, ((x + y) * 5) as u8])
    });
    let mut encoded = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("a PNG encodes");
    encoded.into_inner()
}

/// The transform that reads [`synthetic_png`]`(16, 12)` at its own raster.
const IMAGE: ImageTransform = ImageTransform {
    resized: (16, 12),
    window: (0, 0, 16, 12),
};

/// The pixel rows [`IMAGE`] fills.
const IMAGE_ROWS: usize = 16 * 12;

/// A condition that reads `media` with [`IMAGE`] into the pixel rows `pixels`.
fn image_condition(media: &[u8], pixels: std::ops::Range<usize>) -> ConditionRead<'_> {
    ConditionRead {
        source: ConditionSource::Bytes(media),
        image: Some(IMAGE),
        video: None,
        audio: None,
        pixels,
        samples: 0..0,
    }
}

/// A stereo 16-bit PCM WAV of `samples` samples at `rate`, written here.
fn synthetic_wav(rate: u32, samples: usize) -> Vec<u8> {
    // Two channels of two bytes: four bytes per sample.
    let data_bytes = u32::try_from(samples * 4).expect("a small track");
    let mut wav = Vec::with_capacity(44 + data_bytes as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    // The 16-byte PCM format chunk: format 1, channels, rate, byte rate,
    // block alignment and bits per sample.
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * 4).to_le_bytes());
    wav.extend_from_slice(&4u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());
    for sample in 0..samples {
        let value = ((sample % 64) as i16 - 32) * 512;
        wav.extend_from_slice(&value.to_le_bytes());
        wav.extend_from_slice(&(-value).to_le_bytes());
    }
    wav
}

/// The transform that keeps 800 samples of [`synthetic_wav`]`(44_100, 1_000)` from
/// sample 100 and brings them to 16 kHz: `ceil(16000 * 800 / 44100)` samples.
const AUDIO: AudioTransform = AudioTransform {
    native_rate: 44_100,
    start: 100,
    source_samples: 800,
    samples: 291,
    rate: 16_000,
};

/// Four flat 32x24 frames at 10 fps, encoded by this crate into a seekable
/// file, as a reference video arrives.
fn synthetic_video() -> std::fs::File {
    let pixels: Vec<u8> = (0..4u8)
        .flat_map(|frame| vec![40 + 50 * frame; 32 * 24 * 3])
        .collect();
    let encoded = crate::encode_video_unit(
        crate::Rgb24Frames {
            pixels: &pixels,
            height: 24,
            width: 32,
        },
        10,
    )
    .expect("a reference video encodes");
    let mut file = tempfile::tempfile().expect("a scratch file");
    file.write_all(&encoded)
        .expect("writing the reference video");
    file
}

/// The clip that decodes all of [`synthetic_video`] at its own rate and canvas.
const VIDEO: VideoClip = VideoClip {
    frame_rate: 10,
    canvas: (32, 24),
    frames: 4,
    vae_frames: 4,
    start_frame: 0,
};

/// RGB24 bytes of every frame `clip` decodes.
fn clip_bytes(clip: &VideoClip) -> usize {
    clip.frames as usize * clip.canvas.0 as usize * clip.canvas.1 as usize * 3
}

#[test]
fn images_refuse_a_product_of_another_extent() {
    let media = synthetic_png(16, 12);
    let mut full = vec![0u8; IMAGE_ROWS * 3];
    read_image(&media, &IMAGE, &mut full).expect("the transform's extent reads");

    let mut short = vec![0u8; IMAGE_ROWS * 3 - 1];
    assert!(matches!(
        read_image(&media, &IMAGE, &mut short),
        Err(Error::Invalid(_))
    ));
    // A truncated image fails in its decoder.
    assert!(matches!(
        read_image(&media[..10], &IMAGE, &mut full),
        Err(Error::Codec(_))
    ));
}

#[test]
fn audio_refuses_a_track_at_another_rate_or_with_fewer_samples() {
    let media = synthetic_wav(44_100, 1_000);
    let mut out = vec![0.0f32; AUDIO.samples * 2];
    read_audio(&media, &AUDIO, &mut out).expect("the transform's track reads");

    let mut other_rate = AUDIO;
    other_rate.native_rate = 48_000;
    assert!(matches!(
        read_audio(&media, &other_rate, &mut out),
        Err(Error::Invalid(_))
    ));
    let mut too_long = AUDIO;
    too_long.start = 100_000;
    assert!(matches!(
        read_audio(&media, &too_long, &mut out),
        Err(Error::Invalid(_))
    ));
    let mut short = vec![0.0f32; AUDIO.samples * 2 - 1];
    assert!(matches!(
        read_audio(&media, &AUDIO, &mut short),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn reference_videos_refuse_a_clip_the_video_does_not_hold() {
    let file = synthetic_video();
    let mut out = vec![0u8; clip_bytes(&VIDEO)];
    read_video(&ffmpeg(), &file, &VIDEO, &mut out).expect("the clip decodes");

    // The video holds fewer frames after the start than the clip asks for.
    let mut beyond = VIDEO;
    beyond.start_frame = 2;
    assert!(matches!(
        read_video(&ffmpeg(), &file, &beyond, &mut out),
        Err(Error::Invalid(_))
    ));
    // A clip whose rows disagree with the decoded frames is refused.
    let mut wider = VIDEO;
    wider.canvas = (VIDEO.canvas.0 + 1, VIDEO.canvas.1);
    assert!(matches!(
        read_video(&ffmpeg(), &file, &wider, &mut out),
        Err(Error::Invalid(_))
    ));
    // A source that is not a video fails through FFmpeg's own error.
    let mut bogus = tempfile::tempfile().expect("a scratch file");
    bogus
        .write_all(b"this is not a video")
        .expect("writing the scratch file");
    assert!(matches!(
        read_video(&ffmpeg(), &bogus, &VIDEO, &mut out),
        Err(Error::ReferenceVideo { .. })
    ));
}

#[test]
fn conditions_refuse_an_encoded_prefix_longer_than_the_decoded_clip() {
    let file = synthetic_video();
    let mut clip = VIDEO;
    clip.vae_frames = clip.frames + 1;
    let rows = clip.vae_frames as usize * clip.canvas.0 as usize * clip.canvas.1 as usize;
    let mut pixels = vec![0u8; rows * 3];
    assert!(matches!(
        read_conditions(
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
        ),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn readers_refuse_a_transform_the_host_cannot_address() {
    // A transform comes from a peer: an extent whose byte count overflows must fail
    // the request instead of panicking in the arithmetic that sizes buffers.
    let huge = ImageTransform {
        resized: (u32::MAX, u32::MAX),
        window: (0, 0, u32::MAX, u32::MAX),
    };
    let mut empty = Vec::new();
    assert!(matches!(
        read_image(&synthetic_png(16, 12), &huge, &mut empty),
        Err(Error::Invalid(_))
    ));

    let mut samples_transform = AUDIO;
    samples_transform.samples = usize::MAX;
    let mut no_samples = Vec::new();
    assert!(matches!(
        read_audio(
            &synthetic_wav(44_100, 1_000),
            &samples_transform,
            &mut no_samples
        ),
        Err(Error::Invalid(_))
    ));

    let file = synthetic_video();
    let mut clip = VIDEO;
    clip.canvas = (65_536, 65_536);
    clip.frames = u32::MAX;
    clip.vae_frames = 1;
    let mut no_frames = Vec::new();
    assert!(matches!(
        read_video(&ffmpeg(), &file, &clip, &mut no_frames),
        Err(Error::Invalid(_))
    ));

    // The same clip is refused before any buffer is sized for it, even when
    // the request holds no products.
    let rows = clip.vae_frames as usize;
    assert!(matches!(
        read_conditions(
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
                pixels: None,
                samples: None,
            },
        ),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn conditions_refuse_rows_that_do_not_fill_their_products() {
    let media = synthetic_png(16, 12);
    let rows = IMAGE_ROWS;
    let read = |conditions: &[ConditionRead<'_>], pixels: Option<&mut [u8]>| {
        read_conditions(
            &ffmpeg(),
            conditions,
            &mut ConditionProducts {
                pixels,
                samples: None,
            },
        )
    };
    let conditions = [image_condition(&media, 0..rows)];

    // The product the transform fills exactly is filled.
    let mut exact = vec![0u8; rows * 3];
    read(&conditions, Some(&mut exact)).expect("the exactly sized product fills");

    // A product with one row too many is left unfilled.
    let mut wide = vec![0u8; (rows + 1) * 3];
    assert!(matches!(
        read(&conditions, Some(&mut wide)),
        Err(Error::Invalid(_))
    ));
    // A product without room for every row is exceeded.
    let mut narrow = vec![0u8; (rows - 1) * 3];
    assert!(matches!(
        read(&conditions, Some(&mut narrow)),
        Err(Error::Invalid(_))
    ));
    // A condition that needs a product the request does not hold.
    assert!(matches!(read(&conditions, None), Err(Error::Invalid(_))));
    // Declared rows that disagree with the transform are refused before decoding.
    let mut roomy = vec![0u8; (rows + 1) * 3];
    assert!(matches!(
        read(&[image_condition(&media, 0..rows + 1)], Some(&mut roomy)),
        Err(Error::Invalid(_))
    ));
    // Two conditions that claim the same rows cannot both own them.
    let overlapping = [
        image_condition(&media, 0..rows),
        image_condition(&media, 0..rows),
    ];
    let mut product = vec![0u8; 2 * rows * 3];
    assert!(matches!(
        read(&overlapping, Some(&mut product)),
        Err(Error::Invalid(_))
    ));
}
