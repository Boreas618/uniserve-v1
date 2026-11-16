//! Generated media, decoded in full and described by what its content states.
//!
//! An image is held to its detected format; an MP4 must hold one video and one
//! audio stream, both of which decode. What the media must be for the point's
//! task (its codecs, raster, frame count and soundtrack) is the task's check,
//! not this module's.

use std::io::Write as _;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::record::{DecodedImage, DecodedVideo, hex};

/// The formats a generated image may have: its media type and its sample's file
/// extension.
const IMAGE_FORMATS: &[(image::ImageFormat, &str, &str)] = &[
    (image::ImageFormat::Png, "image/png", "png"),
    (image::ImageFormat::Jpeg, "image/jpeg", "jpg"),
    (image::ImageFormat::WebP, "image/webp", "webp"),
    (image::ImageFormat::Gif, "image/gif", "gif"),
    (image::ImageFormat::Bmp, "image/bmp", "bmp"),
];

/// A declared image type in its canonical spelling.
fn canonical_mime(mime: &str) -> String {
    let mime = mime.trim().to_lowercase();
    match mime.as_str() {
        "image/jpg" => "image/jpeg".to_owned(),
        "image/x-png" => "image/png".to_owned(),
        _ => mime,
    }
}

/// Decodes an image fully; its sample is named by the digest of its bytes.
///
/// # Errors
///
/// The stable classifier `response_undecodable_image`,
/// `response_unsupported_image_format` or `response_image_mime_mismatch`.
pub fn inspect_image(data: Vec<u8>, declared: Option<&str>) -> Result<DecodedImage, &'static str> {
    let format = image::guess_format(&data).map_err(|_| "response_undecodable_image")?;
    let (_, mime, extension) = IMAGE_FORMATS
        .iter()
        .find(|(known, _, _)| *known == format)
        .ok_or("response_unsupported_image_format")?;
    let decoded = image::load_from_memory_with_format(&data, format)
        .map_err(|_| "response_undecodable_image")?;
    if decoded.width() == 0 || decoded.height() == 0 {
        return Err("response_undecodable_image");
    }
    if declared.is_some_and(|declared| canonical_mime(declared) != *mime) {
        return Err("response_image_mime_mismatch");
    }
    let sha256 = hex(&Sha256::digest(&data));
    Ok(DecodedImage {
        sample_filename: format!("{sha256}.{extension}"),
        sha256,
        mime: (*mime).to_owned(),
        width: decoded.width(),
        height: decoded.height(),
        data,
    })
}

/// Why an MP4 response is invalid: a stable classifier, which the summary
/// counts records by, and the detail.
#[derive(Debug)]
pub struct VideoError {
    pub classifier: &'static str,
    pub message: String,
}

impl VideoError {
    fn new(classifier: &'static str, message: impl Into<String>) -> Self {
        Self {
            classifier,
            message: message.into(),
        }
    }
}

/// How long the demuxer may take over one response.
const PROBE_TIMEOUT: Duration = Duration::from_secs(60);

/// Decodes one MP4's video and audio streams fully and states what they hold:
/// frame and sample counts from the decode, the rational frame rate, the codecs,
/// the pixels' variance over every RGB value of every frame, and the soundtrack's
/// RMS over every sample of every channel.
///
/// # Errors
///
/// When the body is empty, its declared type is not `video/mp4` (parameters
/// after `;` aside), it does not hold exactly one video and one audio stream,
/// the video stream has no frame rate, or either stream decodes nothing.
pub fn inspect_video(data: &[u8], declared: &str) -> Result<DecodedVideo, VideoError> {
    if data.is_empty() {
        return Err(VideoError::new(
            "response_empty_video",
            "video response body is empty",
        ));
    }
    let mime = declared
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    if mime != "video/mp4" {
        return Err(VideoError::new(
            "response_invalid_video_mime",
            format!("expected video/mp4, received {declared:?}"),
        ));
    }
    let undecodable = |error: &dyn std::fmt::Display| {
        VideoError::new(
            "response_undecodable_video",
            format!("MP4 decode failed: {error}"),
        )
    };
    let container = media::probe::probe_container(data, &["mov", "mp4"], PROBE_TIMEOUT)
        .map_err(|error| undecodable(&error))?;
    let (Some(video), Some(audio)) = (container.video, container.audio) else {
        return Err(VideoError::new(
            "response_invalid_media_streams",
            "the output must contain one video and one audio stream",
        ));
    };
    let (numerator, denominator) = video.frame_rate.ok_or_else(|| {
        VideoError::new(
            "response_missing_video_rate",
            "video stream has no average frame rate",
        )
    })?;
    // The rate as an exact rational in lowest terms.
    let divisor = gcd(numerator, denominator);
    let (fps_numerator, fps_denominator) = (numerator / divisor, denominator / divisor);
    let track = media::probe::decoded_track(data).map_err(|error| undecodable(&error))?;
    if video.frames < 1 || track.samples < 1 {
        return Err(VideoError::new(
            "response_undecodable_video",
            "video or audio stream decoded no frames",
        ));
    }

    // The frames at their coded size and own rate, the samples at their own
    // rate: the variance and the RMS of what the streams hold.
    let frames = u32::try_from(video.frames).map_err(|error| undecodable(&error))?;
    let rate = fps_numerator.div_ceil(fps_denominator);
    let clip = media::read::VideoClip {
        frame_rate: rate,
        canvas: (video.width, video.height),
        frames,
        vae_frames: frames,
        start_frame: 0,
    };
    let mut file = tempfile::tempfile().map_err(|error| undecodable(&error))?;
    file.write_all(data).map_err(|error| undecodable(&error))?;
    let mut pixels = vec![0u8; frames as usize * video.width as usize * video.height as usize * 3];
    media::read_video(&file, &clip, &mut pixels).map_err(|error| undecodable(&error))?;
    let (mut sum, mut squares) = (0.0f64, 0.0f64);
    for value in &pixels {
        let value = f64::from(*value);
        sum += value;
        squares += value * value;
    }
    let count = pixels.len() as f64;
    let variance = (squares / count - (sum / count).powi(2)).max(0.0);

    let transform = media::read::AudioTransform {
        native_rate: track.rate,
        start: 0,
        source_samples: track.samples,
        samples: track.samples,
        rate: track.rate,
    };
    let mut pcm = vec![0f32; track.samples * 2];
    media::read_audio(data, &transform, &mut pcm).map_err(|error| undecodable(&error))?;
    let rms = (pcm
        .iter()
        .map(|sample| f64::from(*sample).powi(2))
        .sum::<f64>()
        / pcm.len() as f64)
        .sqrt();

    let sha256 = hex(&Sha256::digest(data));
    Ok(DecodedVideo {
        data: data.to_vec(),
        sample_filename: format!("{sha256}.mp4"),
        sha256,
        mime,
        width: video.width,
        height: video.height,
        frame_count: video.frames,
        fps_numerator: u64::from(fps_numerator),
        fps_denominator: u64::from(fps_denominator),
        video_codec: video.codec,
        audio_codec: audio.codec,
        audio_channels: audio.channels,
        audio_sample_rate: track.rate,
        audio_samples: track.samples as u64,
        video_variance: variance,
        audio_rms: rms,
    })
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}
