//! The condition readers' refusals of transforms and products the media
//! cannot satisfy. They read media made here (a PNG, a PCM WAV and an MP4 this
//! crate encodes); the readers' agreement with the libraries whose contracts
//! they implement is tested against checked-in fixtures in `tests/read.rs`.

use std::io::Write as _;

use crate::Error;

use super::{
    AudioTransform, ConditionProducts, ConditionRead, ConditionSource, ImageTransform, VideoClip,
    read_audio, read_conditions, read_image, read_video,
};

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
    read_video(&file, &VIDEO, &mut out).expect("the clip decodes");

    // The video holds fewer frames after the start than the clip asks for.
    let mut beyond = VIDEO;
    beyond.start_frame = 2;
    assert!(matches!(
        read_video(&file, &beyond, &mut out),
        Err(Error::Invalid(_))
    ));
    // A clip whose rows disagree with the decoded frames is refused.
    let mut wider = VIDEO;
    wider.canvas = (VIDEO.canvas.0 + 1, VIDEO.canvas.1);
    assert!(matches!(
        read_video(&file, &wider, &mut out),
        Err(Error::Invalid(_))
    ));
    // A source that is not a video fails in FFmpeg's demuxer.
    let mut bogus = tempfile::tempfile().expect("a scratch file");
    bogus
        .write_all(b"this is not a video")
        .expect("writing the scratch file");
    assert!(matches!(
        read_video(&bogus, &VIDEO, &mut out),
        Err(Error::Codec(_))
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
        read_video(&file, &clip, &mut no_frames),
        Err(Error::Invalid(_))
    ));

    // The same clip is refused before any buffer is sized for it, even when
    // the request holds no products.
    let rows = clip.vae_frames as usize;
    assert!(matches!(
        read_conditions(
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
