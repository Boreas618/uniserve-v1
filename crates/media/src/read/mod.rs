//! Conditional media reading: what a request's conditions decode to.
//!
//! The server fetches every condition's bytes and decides how each is prepared
//! (its image or audio transform, or its video clip); the main process reads
//! the bytes and calls these decoders with those values. Each applies its
//! conditioning's operations in their order, so its result is the result of
//! the upstream libraries that define them:
//!
//! - [`read_image`]: decoding with the EXIF orientation applied and RGB
//!   conversion (diffusers' `load_image`), then a LANCZOS resize and the
//!   transform's crop: a stretched keyframe, a cover-cropped one, or an image
//!   reference at its transform's raster.
//! - [`read_audio`]: decoding to planar float at the track's native rate and
//!   channel layout, the transform's native samples kept, a mono track duplicated
//!   to stereo, and one torchaudio resample to the model's rate.
//! - [`read_video`]: FFmpeg's decode, `fps`, `trim` and LANCZOS scale of a
//!   reference video, through the linked libraries.
//!
//! Every function checks that it produced exactly the extent its transform or
//! clip states and fails with [`crate::Error::Invalid`] otherwise, so media that decodes differently
//! from what the server probed fails the request instead of misaligning its
//! rows.
//!
//! Images are decoded by the `image` crate. Its lossless PNG decoder
//! reproduces Pillow's bytes exactly, which the tests assert; JPEG rounds
//! differently from libjpeg-turbo by a few units per channel, which the
//! design's numerical tolerance allows and the §10 visual check covers. The
//! crate's other formats (BMP, GIF, WebP) have no fixtures here, so no
//! equality is claimed for them. Every decoder that exposes EXIF metadata
//! (JPEG, PNG's `eXIf` chunk, WebP, TIFF) has its orientation applied, all
//! eight EXIF cases included.

use std::borrow::Cow;
use std::ops::Range;
use std::os::unix::fs::FileExt as _;

use crate::ffi;
use crate::{Error, Result};

#[cfg(test)]
mod tests;

/// Where an image reference is put on its raster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageTransform {
    /// The `(width, height)` the decoded image is resized to.
    pub resized: (u32, u32),
    /// The kept `(left, top, width, height)` of the resized image.
    pub window: (u32, u32, u32, u32),
}

/// Which samples of an audio track are kept and where they are resampled to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioTransform {
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
    /// An open, seekable file holding a reference video, read without moving
    /// its offset.
    File(&'a std::fs::File),
}

/// One condition's media and the transforms its preparation follows.
#[derive(Debug)]
pub struct ConditionRead<'a> {
    /// The condition's media.
    pub source: ConditionSource<'a>,
    /// The image's fit transform when the condition is an image.
    pub image: Option<ImageTransform>,
    /// The clip to decode when the condition is a reference video.
    pub video: Option<VideoClip>,
    /// The track's transform when the condition carries audio.
    pub audio: Option<AudioTransform>,
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

/// A clip's bytes per frame and in total, checked against the address space.
///
/// A clip comes from a peer, so an extent the host cannot address must fail the
/// request rather than overflow the arithmetic that sizes its buffers.
fn clip_bytes(clip: &VideoClip) -> Result<(usize, usize)> {
    let frame = (clip.canvas.0 as usize)
        .checked_mul(clip.canvas.1 as usize)
        .and_then(|area| area.checked_mul(3))
        .ok_or_else(|| {
            Error::Invalid(format!(
                "a reference video's clip of {}x{} frames exceeds the host's address space",
                clip.canvas.0, clip.canvas.1
            ))
        })?;
    let total = frame.checked_mul(clip.frames as usize).ok_or_else(|| {
        Error::Invalid(format!(
            "a reference video's clip of {} frames of {}x{} exceeds the host's address space",
            clip.frames, clip.canvas.0, clip.canvas.1
        ))
    })?;
    Ok((frame, total))
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
    let (width, height) = decoder.dimensions();
    if u64::from(width) * u64::from(height) > crate::probe::readable::IMAGE_PIXELS {
        return Err(Error::Invalid(format!(
            "a condition image of {width}x{height} pixels exceeds the {} the reader decodes",
            crate::probe::readable::IMAGE_PIXELS
        )));
    }
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

/// Decodes an image and puts it on its transform's raster.
///
/// Writes `[height * width * 3]` uint8 RGB of the kept window into `out`.
///
/// # Errors
///
/// [`crate::Error::Invalid`] when the transform's extent cannot be addressed,
/// the image decodes to another raster than the transform states, or `out` has
/// another length;
/// [`crate::Error::Codec`] when decoding fails.
pub fn read_image(encoded: &[u8], transform: &ImageTransform, out: &mut [u8]) -> Result<()> {
    let (left, top, width, height) = transform.window;
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            Error::Invalid(format!(
                "a condition image's window of {}x{} exceeds the host's address space",
                width, height
            ))
        })?;
    if out.len() != expected {
        return Err(Error::Invalid(
            "a condition image's rows disagree with its transform".to_owned(),
        ));
    }

    let decoded = decode_rgb(encoded)?;
    // Pillow's resize to the image's own size is a copy, so an image already
    // at its raster passes through unchanged.
    let image = if (decoded.width, decoded.height) == transform.resized {
        decoded
    } else {
        let pixels = crate::resample::lanczos_resize(
            &decoded.pixels,
            decoded.width,
            decoded.height,
            transform.resized.0,
            transform.resized.1,
        )?;
        Rgb24 {
            width: transform.resized.0,
            height: transform.resized.1,
            pixels,
        }
    };

    if (left, top, width, height) != (0, 0, image.width, image.height) {
        let right = left.checked_add(width);
        let bottom = top.checked_add(height);
        if width == 0
            || height == 0
            || right.is_none_or(|right| right > image.width)
            || bottom.is_none_or(|bottom| bottom > image.height)
        {
            return Err(Error::Invalid(
                "a condition image's window exceeds its raster".to_owned(),
            ));
        }
    }
    if width == image.width && height == image.height && left == 0 && top == 0 {
        if out.len() != image.pixels.len() {
            return Err(Error::Invalid(
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
/// at `transform.native_rate`. The `source_samples` native samples from `start` are
/// kept, a mono track is duplicated to stereo, and one resample brings the
/// track to `rate`. Writes `[samples, 2]` float32, sample-major.
///
/// # Errors
///
/// [`crate::Error::Invalid`] when the transform's extent cannot be addressed, the
/// track has no audio stream, another rate, too few samples, or resamples to
/// another count; [`crate::Error::Codec`] when decoding fails.
pub fn read_audio(source: &[u8], transform: &AudioTransform, out: &mut [f32]) -> Result<()> {
    let out_samples = transform.samples.checked_mul(2).ok_or_else(|| {
        Error::Invalid(format!(
            "a condition track's transform of {} samples exceeds the host's address space",
            transform.samples
        ))
    })?;
    if out.len() != out_samples {
        return Err(Error::Invalid(
            "a condition track's rows disagree with its transform".to_owned(),
        ));
    }
    let window_samples = transform.source_samples.checked_mul(2).ok_or_else(|| {
        Error::Invalid(format!(
            "a condition track's transform of {} samples exceeds the host's address space",
            transform.source_samples
        ))
    })?;

    let decoded = ffi::decode_audio(source)?;
    if decoded.rate != transform.native_rate {
        return Err(Error::Invalid(format!(
            "an audio track decodes at {} Hz, its transform states {} Hz",
            decoded.rate, transform.native_rate
        )));
    }
    let stop = transform.start.checked_add(transform.source_samples);
    if stop.is_none_or(|stop| stop > decoded.samples) || !(1..=2).contains(&decoded.channels) {
        return Err(Error::Invalid(
            "an audio track holds fewer samples than its transform keeps".to_owned(),
        ));
    }

    // Keep the transform's native window per channel; a mono track feeds both
    // stereo channels.
    let mut planar = vec![0.0f32; window_samples];
    for channel in 0..2 {
        let source_channel = if decoded.channels == 1 { 0 } else { channel };
        let from = source_channel * decoded.samples + transform.start;
        let to = from + transform.source_samples;
        planar[channel * transform.source_samples..][..transform.source_samples]
            .copy_from_slice(&decoded.planar[from..to]);
    }

    let stereo = if transform.native_rate != transform.rate {
        crate::resample::sinc_resample(
            &planar,
            2,
            transform.source_samples,
            transform.native_rate,
            transform.rate,
        )?
    } else {
        planar
    };
    if stereo.len() != out_samples {
        return Err(Error::Invalid(format!(
            "an audio track resampled to {} samples, its transform states {}",
            stereo.len() / 2,
            transform.samples
        )));
    }
    for sample in 0..transform.samples {
        out[sample * 2] = stereo[sample];
        out[sample * 2 + 1] = stereo[transform.samples + sample];
    }
    Ok(())
}

/// Decodes a reference video's kept frames through the linked FFmpeg
/// libraries.
///
/// The video's first stream that is not cover art is decoded, put upright by
/// its display matrix, resampled to the clip's frame rate (FFmpeg's `fps`
/// filter), its first `start_frame` resampled frames skipped, and each kept
/// frame scaled with LANCZOS to the clip's canvas (the scale follows the trim,
/// so it never sees a skipped frame) and converted to RGB24: the frames
/// FFmpeg's command-line tool produces from the file with
/// `-vf fps=R,trim=start_frame=S,scale=W:H:flags=lanczos,setsar=1 -pix_fmt rgb24`.
/// Writes `[clip.frames, height, width, 3]` uint8 RGB: every decoded frame, of
/// which the first `vae_frames` fill the condition's pixel rows. The file is
/// read from its start without moving its offset.
///
/// # Errors
///
/// [`crate::Error::Invalid`] when the clip's extent cannot be addressed, `out`
/// has another length, the video has no video stream or changes its frame
/// size mid-stream, or it decodes to fewer frames than the clip states;
/// [`crate::Error::Codec`] when the video cannot be demuxed or decoded.
pub fn read_video(source: &std::fs::File, clip: &VideoClip, out: &mut [u8]) -> Result<()> {
    let (_, expected) = clip_bytes(clip)?;
    if out.len() != expected {
        return Err(Error::Invalid(
            "a reference video's rows disagree with its clip".to_owned(),
        ));
    }
    let decoded = ffi::decode_video(source, clip, out)?;
    if decoded != clip.frames {
        return Err(Error::Invalid(format!(
            "a reference video decoded to {decoded} frames, its clip states {}",
            clip.frames
        )));
    }
    Ok(())
}

/// The rows one condition takes of each product, as its transforms state.
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
/// The file is read from offset zero without moving its offset, which other
/// holders of the same open file description may rely on.
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
            return Err(Error::Invalid(format!(
                "the request's conditions exceed their {name}"
            )));
        }
        cursor = range.end;
    }
    Ok(())
}

/// Each condition's rows of `product`, in request order, `width` elements to a row.
///
/// The ranges were checked not to overlap, which is what lets every decoder hold its own rows at
/// once.
fn rows_of<'a, T>(
    product: Option<&'a mut [T]>,
    ranges: impl Iterator<Item = Range<usize>>,
    width: usize,
) -> Vec<Option<&'a mut [T]>> {
    let mut ordered: Vec<(usize, Range<usize>)> = ranges.enumerate().collect();
    let mut rows: Vec<Option<&'a mut [T]>> = Vec::new();
    rows.resize_with(ordered.len(), || None);
    let Some(mut rest) = product else {
        return rows;
    };
    ordered.retain(|(_, range)| !range.is_empty());
    ordered.sort_by_key(|(_, range)| range.start);
    let mut cursor = 0;
    for (index, range) in ordered {
        let (_, tail) = rest.split_at_mut((range.start - cursor) * width);
        let (own, tail) = tail.split_at_mut(range.len() * width);
        rows[index] = Some(own);
        rest = tail;
        cursor = range.end;
    }
    rows
}

/// Decodes every condition into its rows of the products.
///
/// Every condition decodes on a thread of its own, the conditions being independent: each reference
/// video, each image and each audio track. A video's first `vae_frames` frames fill
/// its pixel rows once they arrive; all `frames` frames are returned, because the vision packer
/// reads the frames past the product.
///
/// # Errors
///
/// [`crate::Error::Invalid`] when the conditions exceed a product, leave it
/// unfilled, decode to another extent, or a video's `vae_frames` exceeds its
/// `frames`; the readers' own errors otherwise.
pub fn read_conditions(
    conditions: &[ConditionRead<'_>],
    products: &mut ConditionProducts<'_>,
) -> Result<Vec<Option<DecodedVideo>>> {
    let pixels = products.pixels.as_deref_mut();
    let samples = products.samples.as_deref_mut();

    // Every condition's rows must be what its transforms decode to, and together
    // the conditions must fill each product exactly (`Conditions.rows`).
    let mut pixel_rows = 0usize;
    let mut sample_rows = 0usize;
    let mut declared_pixels = Vec::new();
    let mut declared_samples = Vec::new();
    for condition in conditions {
        if let Some(clip) = condition.video {
            if clip.vae_frames > clip.frames {
                return Err(Error::Invalid(
                    "a reference video's encoded frames exceed its decoded frames".to_owned(),
                ));
            }
            // Reject an unaddressable clip before any buffer is sized from it.
            clip_bytes(&clip)?;
        }
        let (wanted_pixels, wanted_samples) = extents(condition);
        if condition.pixels.len() != wanted_pixels {
            return Err(Error::Invalid(
                "a condition's pixel rows disagree with its transform".to_owned(),
            ));
        }
        if condition.samples.len() != wanted_samples {
            return Err(Error::Invalid(
                "a condition's sample rows disagree with its transform".to_owned(),
            ));
        }
        pixel_rows = pixel_rows
            .checked_add(condition.pixels.len())
            .ok_or_else(|| {
                Error::Invalid("the request's conditions exceed their pixels".to_owned())
            })?;
        sample_rows = sample_rows
            .checked_add(condition.samples.len())
            .ok_or_else(|| {
                Error::Invalid("the request's conditions exceed their samples".to_owned())
            })?;
        if !condition.pixels.is_empty() {
            declared_pixels.push(condition.pixels.clone());
        }
        if !condition.samples.is_empty() {
            declared_samples.push(condition.samples.clone());
        }
    }
    match pixels.as_deref() {
        Some(product) => {
            // The product is bytes and the conditions' ranges are rows of
            // three bytes each, so the two are compared in rows.
            if product.len() % 3 != 0 {
                return Err(Error::Invalid(
                    "the condition pixel product is not whole RGB24 rows".to_owned(),
                ));
            }
            let rows = product.len() / 3;
            check_ranges(&declared_pixels, rows, "pixels")?;
            if pixel_rows != rows {
                return Err(Error::Invalid(
                    "the request's conditions leave their pixels unfilled".to_owned(),
                ));
            }
        }
        None if pixel_rows != 0 => {
            return Err(Error::Invalid(
                "a condition's pixels have no product".to_owned(),
            ));
        }
        None => {}
    }
    match samples.as_deref() {
        Some(product) => {
            // The product is interleaved stereo samples and the conditions'
            // ranges are stereo rows, so the two are compared in rows.
            if product.len() % 2 != 0 {
                return Err(Error::Invalid(
                    "the condition sample product is not whole stereo samples".to_owned(),
                ));
            }
            let rows = product.len() / 2;
            check_ranges(&declared_samples, rows, "samples")?;
            if sample_rows != rows {
                return Err(Error::Invalid(
                    "the request's conditions leave their samples unfilled".to_owned(),
                ));
            }
        }
        None if sample_rows != 0 => {
            return Err(Error::Invalid(
                "a condition's samples have no product".to_owned(),
            ));
        }
        None => {}
    }

    std::thread::scope(|scope| -> Result<Vec<Option<DecodedVideo>>> {
        // Start every reference video first: a video's decode is the longest,
        // and it runs beside the image and audio decoders.
        let mut videos = Vec::with_capacity(conditions.len());
        for condition in conditions {
            if let Some(clip) = condition.video {
                let ConditionSource::File(file) = condition.source else {
                    return Err(Error::Invalid(
                        "a reference video needs a seekable file".to_owned(),
                    ));
                };
                let total = clip_bytes(&clip)?.1;
                videos.push(Some(scope.spawn(move || {
                    let mut frames = vec![0u8; total];
                    read_video(file, &clip, &mut frames)?;
                    Ok(frames)
                })));
            } else {
                videos.push(None);
            }
        }

        // Each image and audio track on a thread of its own: an enlarged image and a track's
        // decoding share nothing, and a request waits for the last of them.
        let mut pixels = rows_of(pixels, conditions.iter().map(|c| c.pixels.clone()), 3);
        let mut samples = rows_of(samples, conditions.iter().map(|c| c.samples.clone()), 2);
        let mut decoders = Vec::with_capacity(conditions.len());
        for (index, condition) in conditions.iter().enumerate() {
            if let Some(transform) = condition.image {
                let out = pixels[index].take().ok_or_else(|| {
                    Error::Invalid("a condition's pixels have no product".to_owned())
                })?;
                decoders.push(scope.spawn(move || {
                    read_image(&encoded_bytes(&condition.source)?, &transform, out)
                }));
            }
            if let Some(transform) = condition.audio {
                let out = samples[index].take().ok_or_else(|| {
                    Error::Invalid("a condition's samples have no product".to_owned())
                })?;
                decoders.push(scope.spawn(move || {
                    read_audio(&encoded_bytes(&condition.source)?, &transform, out)
                }));
            }
        }
        for decoder in decoders {
            decoder
                .join()
                .map_err(|_| Error::Codec("a condition's decoding thread failed".to_owned()))??;
        }

        // The videos' frames arrive after the library's decoders; copy each
        // video's encoded prefix into its pixel rows and hand the whole
        // buffer back.
        let mut decoded = Vec::with_capacity(conditions.len());
        for (index, (condition, handle)) in conditions.iter().zip(videos).enumerate() {
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
            let product = pixels[index]
                .take()
                .ok_or_else(|| Error::Invalid("a condition's pixels have no product".to_owned()))?;
            product.copy_from_slice(&frames[..condition.pixels.len() * 3]);
            decoded.push(Some(DecodedVideo {
                clip,
                pixels: frames,
            }));
        }
        Ok(decoded)
    })
}
