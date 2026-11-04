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

use std::path::Path;

use crate::Result;

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
    /// Kept frames, starting at `start_frame`.
    pub frames: u32,
    /// First kept frame in the video's own rate.
    pub start_frame: u32,
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

/// Decodes an image and puts it on its planned raster.
///
/// Returns `[height * width * 3]` uint8 RGB of the kept window.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the image decodes to another raster than
/// planned or `out` has another length; [`crate::Error::Codec`] when decoding
/// fails.
pub fn read_image(encoded: &[u8], plan: &ImagePlan, out: &mut [u8]) -> Result<()> {
    todo!("builder02: read module")
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
    todo!("builder02: read module")
}

/// Decodes a reference video's kept frames with one FFmpeg pass.
///
/// The frames are resampled to the clip's frame rate, the start frames
/// skipped, then scaled per frame with LANCZOS to the clip's canvas (the scale
/// follows the trim, so it never sees a skipped frame). Writes
/// `[frames, height, width, 3]` uint8 RGB.
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
    todo!("builder02: read module")
}

/// Decodes every condition into its rows of the products.
///
/// Every reference video's FFmpeg decode runs on a thread of its own from the
/// start, beside the library's decoders on this thread: each image and audio
/// track in request order, then each video's frames once they arrived.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the conditions exceed a product, leave it
/// unfilled, or decode to another extent; the readers' own errors otherwise.
pub fn read_conditions(
    ffmpeg: &Path,
    conditions: &[ConditionRead<'_>],
    products: &mut ConditionProducts<'_>,
) -> Result<()> {
    todo!("builder02: read module")
}
