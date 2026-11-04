//! Conditional media reading: what a request's conditions decode to.
//!
//! The server fetches every condition's bytes and plans how each is prepared;
//! the main process reads the bytes and calls these decoders with the plan's
//! values. They apply the reference conditioning's operations in its order, so
//! their results equal the reference's own:
//!
//! - [`read_image`]: decoding with the EXIF orientation applied and RGB
//!   conversion (diffusers' `load_image`), then a LANCZOS resize and the
//!   planned crop: a stretched keyframe, a cover-cropped one, or an image
//!   reference at its planned raster.
//! - [`read_audio`]: decoding to planar float at the track's native rate and
//!   channel layout, the planned native samples kept, a mono track duplicated
//!   to stereo, and one torchaudio resample to the model's rate.
//! - [`read_video`]: one external FFmpeg pass over a reference video.
//!
//! Every function checks that it produced exactly the planned extent and fails
//! with [`crate::Error::Plan`] otherwise, so media that decodes differently
//! from what the server probed fails the request instead of misaligning its
//! rows.
//!
//! Images are decoded by the `image` crate. Its lossless decoders (PNG, BMP,
//! GIF) reproduce Pillow's bytes exactly, which the tests assert; its lossy
//! decoders (JPEG, WebP) round differently from libjpeg-turbo by a few units
//! per channel, which the design's numerical tolerance allows and the §10
//! visual check covers. PNG `eXIf` chunks are not read; JPEG, WebP and TIFF
//! carry their orientation and all eight EXIF cases are applied.

use std::borrow::Cow;
use std::io::{Seek as _, SeekFrom};
use std::ops::Range;
use std::os::unix::fs::FileExt as _;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::ffi;
use crate::{Error, Result};

#[cfg(test)]
mod tests;

/// Where an image reference is put on its raster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImagePlan {
    /// The `(width, height)` the decoded image is resized to.
    pub resized: (u32, u32),
    /// The kept `(left, top, width, height)` of the resized image.
    pub window: (u32, u32, u32, u32),
}

/// Which samples of an audio track are kept and where they are resampled to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioPlan {
    /// The rate the track was probed at; a track that decodes at another rate
    /// fails the request.
    pub native_rate: u32,
    /// The first kept native sample.
    pub start: usize,
    /// Kept native samples.
    pub source_samples: usize,
    /// Samples at the model's rate: the extent of the product rows.
    pub samples: usize,
    /// The model's rate.
    pub rate: u32,
}

/// The frames and canvas of a reference video's kept clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoClip {
    /// The model's frame rate the video is resampled to.
    pub frame_rate: u32,
    /// The clip's `(width, height)` canvas.
    pub canvas: (u32, u32),
    /// Kept frames, starting at `start_frame`, that the decode produces.
    pub frames: u32,
    /// The leading decoded frames the condition's pixel rows cover (the
    /// reference's `vae_frames`): at most `frames`. The frames past this
    /// prefix are still decoded and returned, because the vision packer reads
    /// them.
    pub vae_frames: u32,
    /// First kept frame in the video's own rate.
    pub start_frame: u32,
}

/// A reference video's decoded frames and the clip they were decoded for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedVideo {
    /// The clip the frames were decoded for.
    pub clip: VideoClip,
    /// `[clip.frames, height, width, 3]` RGB24: every decoded frame, not only
    /// the `vae_frames` the condition's pixel rows took.
    pub pixels: Vec<u8>,
}

/// Where one condition's encoded media comes from.
#[derive(Debug)]
pub enum ConditionSource<'a> {
    /// Encoded bytes of an image or of an audio container.
    Bytes(&'a [u8]),
    /// An open, seekable file holding a reference video; the external `ffmpeg`
    /// process reads it.
    File(&'a std::fs::File),
}

/// One condition's media and the plan its preparation follows.
#[derive(Debug)]
pub struct ConditionRead<'a> {
    /// The condition's media.
    pub source: ConditionSource<'a>,
    /// The image's fit plan when the condition is an image.
    pub image: Option<ImagePlan>,
    /// The clip plan when the condition is a reference video.
    pub video: Option<VideoClip>,
    /// The track's plan when the condition carries audio.
    pub audio: Option<AudioPlan>,
    /// The condition's rows of the `condition_pixels` product: RGB24 pixels,
    /// three bytes each.
    pub pixels: std::ops::Range<usize>,
    /// The condition's rows of the `condition_samples` product: stereo float
    /// samples, two each.
    pub samples: std::ops::Range<usize>,
}

/// The host products one request's conditions fill.
#[derive(Debug)]
pub struct ConditionProducts<'a> {
    /// `condition_pixels`: `rows * 3` RGB24 bytes, frame-major.
    pub pixels: Option<&'a mut [u8]>,
    /// `condition_samples`: `rows * 2` float32 samples, sample-major.
    pub samples: Option<&'a mut [f32]>,
}

/// An image decoded to RGB24 rows.
struct Rgb24 {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Decodes an image to RGB24 with its EXIF orientation applied, as Pillow's
/// `ImageOps.exif_transpose` and `convert("RGB")` do.
fn decode_rgb(encoded: &[u8]) -> Result<Rgb24> {
    use image::ImageDecoder as _;

    let reader = image::ImageReader::new(std::io::Cursor::new(encoded))
        .with_guessed_format()
        .map_err(|error| Error::Codec(format!("reading a condition image's format: {error}")))?;
    let mut decoder = reader
        .into_decoder()
        .map_err(|error| Error::Codec(format!("decoding a condition image: {error}")))?;
    // A missing or unknown orientation tag leaves the pixels as they are.
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut decoded = image::DynamicImage::from_decoder(decoder)
        .map_err(|error| Error::Codec(format!("decoding a condition image: {error}")))?;
    decoded.apply_orientation(orientation);
    let rgb = decoded.to_rgb8();
    let (width, height) = rgb.dimensions();
    Ok(Rgb24 {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// Decodes an image and puts it on its planned raster.
///
/// Writes `[height * width * 3]` uint8 RGB of the kept window into `out`.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the image decodes to another raster than
/// planned or `out` has another length; [`crate::Error::Codec`] when decoding
/// fails.
pub fn read_image(encoded: &[u8], plan: &ImagePlan, out: &mut [u8]) -> Result<()> {
    let decoded = decode_rgb(encoded)?;
    // Pillow's resize to the image's own size is a copy, so an image already
    // at its raster passes through unchanged.
    let image = if (decoded.width, decoded.height) == plan.resized {
        decoded
    } else {
        let pixels = crate::resample::lanczos_resize(
            &decoded.pixels,
            decoded.width,
            decoded.height,
            plan.resized.0,
            plan.resized.1,
        )?;
        Rgb24 {
            width: plan.resized.0,
            height: plan.resized.1,
            pixels,
        }
    };

    let (left, top, width, height) = plan.window;
    if (left, top, width, height) != (0, 0, image.width, image.height) {
        let right = left.checked_add(width);
        let bottom = top.checked_add(height);
        if width == 0
            || height == 0
            || right.is_none_or(|right| right > image.width)
            || bottom.is_none_or(|bottom| bottom > image.height)
        {
            return Err(Error::Plan(
                "a condition image's window exceeds its raster".to_owned(),
            ));
        }
    }
    let expected = width as usize * height as usize * 3;
    if out.len() != expected {
        return Err(Error::Plan(
            "a condition image's rows disagree with its plan".to_owned(),
        ));
    }
    if width == image.width && height == image.height && left == 0 && top == 0 {
        if out.len() != image.pixels.len() {
            return Err(Error::Plan(
                "a condition image decoded to another raster".to_owned(),
            ));
        }
        out.copy_from_slice(&image.pixels);
        return Ok(());
    }

    let stride = image.width as usize * 3;
    let row_bytes = width as usize * 3;
    for row in 0..height as usize {
        let source = (top as usize + row) * stride + left as usize * 3;
        let target = row * row_bytes;
        out[target..][..row_bytes].copy_from_slice(&image.pixels[source..][..row_bytes]);
    }
    Ok(())
}

/// Decodes an audio track's kept samples at the model's rate.
///
/// `source` holds the container's encoded bytes; its first audio stream is read
/// at `plan.native_rate`. The `source_samples` native samples from `start` are
/// kept, a mono track is duplicated to stereo, and one resample brings the
/// track to `rate`. Writes `[samples, 2]` float32, sample-major.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the track has no audio stream, another rate, too
/// few samples, or resamples to another count; [`crate::Error::Codec`] when
/// decoding fails.
pub fn read_audio(source: &[u8], plan: &AudioPlan, out: &mut [f32]) -> Result<()> {
    let decoded = ffi::decode_audio(source)?;
    if decoded.rate != plan.native_rate {
        return Err(Error::Plan(format!(
            "an audio track decodes at {} Hz, {} Hz were planned",
            decoded.rate, plan.native_rate
        )));
    }
    let stop = plan.start.checked_add(plan.source_samples);
    if stop.is_none_or(|stop| stop > decoded.samples) || !(1..=2).contains(&decoded.channels) {
        return Err(Error::Plan(
            "an audio track holds fewer samples than planned".to_owned(),
        ));
    }

    // Keep the planned native window per channel; a mono track feeds both
    // stereo channels.
    let mut planar = vec![0.0f32; 2 * plan.source_samples];
    for channel in 0..2 {
        let source_channel = if decoded.channels == 1 { 0 } else { channel };
        let from = source_channel * decoded.samples + plan.start;
        let to = from + plan.source_samples;
        planar[channel * plan.source_samples..][..plan.source_samples]
            .copy_from_slice(&decoded.planar[from..to]);
    }

    let stereo = if plan.native_rate != plan.rate {
        crate::resample::sinc_resample(
            &planar,
            2,
            plan.source_samples,
            plan.native_rate,
            plan.rate,
        )?
    } else {
        planar
    };
    if stereo.len() != 2 * plan.samples {
        return Err(Error::Plan(format!(
            "an audio track resampled to {} samples, {} were planned",
            stereo.len() / 2,
            plan.samples
        )));
    }
    if out.len() != 2 * plan.samples {
        return Err(Error::Plan(
            "a condition track's rows disagree with its plan".to_owned(),
        ));
    }
    for sample in 0..plan.samples {
        out[sample * 2] = stereo[sample];
        out[sample * 2 + 1] = stereo[plan.samples + sample];
    }
    Ok(())
}

/// Decodes a reference video's kept frames with one FFmpeg pass.
///
/// The frames are resampled to the clip's frame rate, the start frames
/// skipped, then scaled per frame with LANCZOS to the clip's canvas (the scale
/// follows the trim, so it never sees a skipped frame). Writes
/// `[clip.frames, height, width, 3]` uint8 RGB: every decoded frame, of which
/// the first `vae_frames` fill the condition's pixel rows.
///
/// # Errors
///
/// [`crate::Error::ReferenceVideo`] when the process exits in failure;
/// [`crate::Error::Plan`] when it decodes another frame count;
/// [`crate::Error::Io`] when the process cannot be started.
pub fn read_video(
    ffmpeg: &Path,
    source: &std::fs::File,
    clip: &VideoClip,
    out: &mut [u8],
) -> Result<()> {
    let (width, height) = clip.canvas;
    let filters = format!(
        "fps={},trim=start_frame={},scale={width}:{height}:flags=lanczos,setsar=1",
        clip.frame_rate, clip.start_frame
    );
    // The shared-memory object or file is a seekable file FFmpeg reads
    // directly; it is the child's standard input, which the `fd:` demuxer
    // reads without a pipe in between.
    let mut input = source
        .try_clone()
        .map_err(|error| Error::io("opening a reference video", error))?;
    // `fd:` reads from the current position; a fresh open (the reference's
    // `/dev/shm/<name>`) starts at zero.
    input
        .seek(SeekFrom::Start(0))
        .map_err(|error| Error::io("positioning a reference video", error))?;
    let process = Command::new(ffmpeg)
        .args([
            "-v",
            "error",
            "-i",
            "fd:",
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            &filters,
            "-frames:v",
            &clip.frames.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdin(Stdio::from(input))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| Error::io(format!("starting {}", ffmpeg.display()), error))?;
    let output = process
        .wait_with_output()
        .map_err(|error| Error::io("decoding a reference video", error))?;
    if !output.status.success() {
        return Err(Error::ReferenceVideo {
            ffmpeg: ffmpeg.to_path_buf(),
            message: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    let frame_bytes = width as usize * height as usize * 3;
    let planned = frame_bytes * clip.frames as usize;
    if output.stdout.len() != planned {
        return Err(Error::Plan(format!(
            "a reference video decoded to {} frames, {} were planned",
            output.stdout.len() / frame_bytes.max(1),
            clip.frames
        )));
    }
    if out.len() != planned {
        return Err(Error::Plan(
            "a reference video's rows disagree with its plan".to_owned(),
        ));
    }
    out.copy_from_slice(&output.stdout);
    Ok(())
}

/// The rows one condition takes of each product, as its plan says.
fn extents(condition: &ConditionRead<'_>) -> (usize, usize) {
    let pixels = if let Some(image) = condition.image {
        image.window.2 as usize * image.window.3 as usize
    } else if let Some(clip) = condition.video {
        clip.vae_frames as usize * clip.canvas.0 as usize * clip.canvas.1 as usize
    } else {
        0
    };
    let samples = condition.audio.map_or(0, |audio| audio.samples);
    (pixels, samples)
}

/// The encoded bytes of a condition: its own bytes, or the bytes of the
/// seekable file a reference video came from (which also carries its
/// soundtrack).
///
/// The file is read at offset zero without moving its offset: a reference
/// video's FFmpeg process shares that file description, and its reading
/// position must not move under it.
fn encoded_bytes<'a>(source: &'a ConditionSource<'_>) -> Result<Cow<'a, [u8]>> {
    match source {
        ConditionSource::Bytes(bytes) => Ok(Cow::Borrowed(bytes)),
        ConditionSource::File(file) => {
            let length = file
                .metadata()
                .map_err(|error| Error::io("reading a condition's media", error))?
                .len() as usize;
            let mut bytes = vec![0u8; length];
            file.read_exact_at(&mut bytes, 0)
                .map_err(|error| Error::io("reading a condition's media", error))?;
            Ok(Cow::Owned(bytes))
        }
    }
}

/// Checks that a product's declared rows are inside it and do not overlap.
fn check_ranges(ranges: &[Range<usize>], rows: usize, name: &str) -> Result<()> {
    let mut ranges = ranges.to_vec();
    ranges.sort_by_key(|range| range.start);
    let mut cursor = 0usize;
    for range in &ranges {
        if range.start < cursor || range.end > rows || range.start > range.end {
            return Err(Error::Plan(format!(
                "the request's conditions exceed their {name}"
            )));
        }
        cursor = range.end;
    }
    Ok(())
}

/// Decodes every condition into its rows of the products.
///
/// Every reference video's FFmpeg decode runs on a thread of its own from the
/// start, beside the library's decoders on this thread: each image and audio
/// track in request order, then each video's frames once they arrived. A
/// video's first `vae_frames` frames fill its pixel rows; all `frames` frames
/// are returned, because the vision packer reads the frames past the product.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the conditions exceed a product, leave it
/// unfilled, decode to another extent, or a video's `vae_frames` exceeds its
/// `frames`; the readers' own errors otherwise.
pub fn read_conditions(
    ffmpeg: &Path,
    conditions: &[ConditionRead<'_>],
    products: &mut ConditionProducts<'_>,
) -> Result<Vec<Option<DecodedVideo>>> {
    let mut pixels = products.pixels.as_deref_mut();
    let mut samples = products.samples.as_deref_mut();

    // Every condition's rows must be what its plan decodes to, and together
    // the conditions must fill each product exactly (`Conditions.rows`).
    let mut pixel_rows = 0usize;
    let mut sample_rows = 0usize;
    let mut declared_pixels = Vec::new();
    let mut declared_samples = Vec::new();
    for condition in conditions {
        if let Some(clip) = condition.video
            && clip.vae_frames > clip.frames
        {
            return Err(Error::Plan(
                "a reference video's encoded frames exceed its decoded frames".to_owned(),
            ));
        }
        let (wanted_pixels, wanted_samples) = extents(condition);
        if condition.pixels.len() != wanted_pixels {
            return Err(Error::Plan(
                "a condition's pixel rows disagree with its plan".to_owned(),
            ));
        }
        if condition.samples.len() != wanted_samples {
            return Err(Error::Plan(
                "a condition's sample rows disagree with its plan".to_owned(),
            ));
        }
        pixel_rows += condition.pixels.len();
        sample_rows += condition.samples.len();
        if !condition.pixels.is_empty() {
            declared_pixels.push(condition.pixels.clone());
        }
        if !condition.samples.is_empty() {
            declared_samples.push(condition.samples.clone());
        }
    }
    match pixels.as_deref() {
        Some(product) => {
            check_ranges(&declared_pixels, product.len(), "pixels")?;
            if pixel_rows != product.len() {
                return Err(Error::Plan(
                    "the request's conditions leave their pixels unfilled".to_owned(),
                ));
            }
        }
        None if pixel_rows != 0 => {
            return Err(Error::Plan(
                "a condition's pixels have no product".to_owned(),
            ));
        }
        None => {}
    }
    match samples.as_deref() {
        Some(product) => {
            check_ranges(&declared_samples, product.len(), "samples")?;
            if sample_rows != product.len() {
                return Err(Error::Plan(
                    "the request's conditions leave their samples unfilled".to_owned(),
                ));
            }
        }
        None if sample_rows != 0 => {
            return Err(Error::Plan(
                "a condition's samples have no product".to_owned(),
            ));
        }
        None => {}
    }

    std::thread::scope(|scope| -> Result<Vec<Option<DecodedVideo>>> {
        // Start every reference video first, so its FFmpeg pass runs beside
        // the library's decoders.
        let mut videos = Vec::with_capacity(conditions.len());
        for condition in conditions {
            if let Some(clip) = condition.video {
                let ConditionSource::File(file) = condition.source else {
                    return Err(Error::Plan(
                        "a reference video needs a seekable file".to_owned(),
                    ));
                };
                videos.push(Some(scope.spawn(move || {
                    let mut frames = vec![
                        0u8;
                        clip.frames as usize
                            * clip.canvas.0 as usize
                            * clip.canvas.1 as usize
                            * 3
                    ];
                    read_video(ffmpeg, file, &clip, &mut frames)?;
                    Ok(frames)
                })));
            } else {
                videos.push(None);
            }
        }

        // Decode the images and audio tracks in request order on this thread,
        // as the reference's media thread does.
        for condition in conditions {
            if let Some(plan) = condition.image {
                let product = pixels.as_deref_mut().ok_or_else(|| {
                    Error::Plan("a condition's pixels have no product".to_owned())
                })?;
                let encoded = encoded_bytes(&condition.source)?;
                read_image(
                    &encoded,
                    &plan,
                    &mut product[condition.pixels.start * 3..condition.pixels.end * 3],
                )?;
            }
            if let Some(plan) = condition.audio {
                let product = samples.as_deref_mut().ok_or_else(|| {
                    Error::Plan("a condition's samples have no product".to_owned())
                })?;
                let encoded = encoded_bytes(&condition.source)?;
                read_audio(
                    &encoded,
                    &plan,
                    &mut product[condition.samples.start * 2..condition.samples.end * 2],
                )?;
            }
        }

        // The videos' frames arrive after the library's decoders; copy each
        // video's encoded prefix into its pixel rows and hand the whole
        // buffer back.
        let mut decoded = Vec::with_capacity(conditions.len());
        for (condition, handle) in conditions.iter().zip(videos) {
            let Some(handle) = handle else {
                decoded.push(None);
                continue;
            };
            let frames = handle.join().map_err(|_| {
                Error::Codec("a reference video's decoding thread failed".to_owned())
            })??;
            let clip = condition
                .video
                .ok_or_else(|| Error::Codec("a video condition has no clip".to_owned()))?;
            let product = pixels
                .as_deref_mut()
                .ok_or_else(|| Error::Plan("a condition's pixels have no product".to_owned()))?;
            product[condition.pixels.start * 3..condition.pixels.end * 3]
                .copy_from_slice(&frames[..condition.pixels.len() * 3]);
            decoded.push(Some(DecodedVideo {
                clip,
                pixels: frames,
            }));
        }
        Ok(decoded)
    })
}
