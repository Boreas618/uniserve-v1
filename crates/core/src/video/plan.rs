//! MiniMax-H3 request rules.
//!
//! [`plan_request`] resolves everything a request implies before any pixel is
//! decoded: the generated canvas and frame counts, how every condition is
//! prepared (keyframe fit, reference image size, reference clip at the
//! output frame rate, 32 kHz soundtrack), what the Qwen3-VL conditioner sees
//! of it (vision
//! grids, 2 fps samples and block timestamps), and the denoiser rows of each
//! condition's encoder units. [`check_request`] applies the request-only
//! rules, which need no media, so a malformed request is rejected before its
//! media are fetched; [`plan_request`] then plans from the probed media.
//!
//! These are the only implementation of the rules: the server plans every
//! request with them, and the model package
//! (`uniserve.models.minimax_h3.processing`) reads the canvas rule and the
//! frame counts from them through the rank runtime library. The model's own
//! facts, such as its output frame rate, are the rules' parameters
//! ([`PlanLimits`]), as the denoiser declares them. The rules are tested
//! against `tests/python/fixtures/minimax_h3_plan.json`, vectors generated
//! from the reference pipeline.
//!
//! The arithmetic reproduces the diffusers MiniMax-H3 pipeline and the
//! checkpoint's Qwen3-VL processor operation for operation, in `f64` where
//! they compute in Python floats, so that every rounding lands on the same
//! side. Rules the reference leaves open are fixed here:
//!
//! - `start_time_seconds` skips the first `floor(s * fps + 0.5)` frames of a
//!   reference video's timeline at the output frame rate `fps` (24 for the
//!   released checkpoint) and the first `floor(s * rate + 0.5)` samples of
//!   its soundtrack;
//! - a reference video needs at least the fewest frames of the denoiser's
//!   frame rule (one VAE clip and its tail, 22 for the released checkpoint)
//!   on that timeline after the start offset;
//! - `ref2va` keyframes are fitted to the target canvas like `fl2va` ones
//!   (the first stretched, a second cover-cropped) and do not enter the
//!   conditioner;
//! - without `target.duration_seconds`, a `ref2va` request whose references
//!   carry exactly one soundtrack lasts as long as that soundtrack after its
//!   start offset.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{ConditionMedia, ConditionVision, FrameClips, ImageFit, MediaLocator};
use super::{ReferenceLimits, ServedCanvas, VideoCondition, VisionGrid};
use crate::{Canvas, VideoTask};

/// Short edge of the adapt_shape_v1 canvas rule, in pixels.
pub const CANVAS_SHORT_EDGE: u32 = 768;
/// Area cap of the canvas rule, in pixels.
pub const CANVAS_MAX_PIXELS: u32 = 768 * 1344;
/// What every canvas side is a multiple of: the VAE's 16x spatial
/// compression times the 2x2 denoiser patch.
pub const CANVAS_MULTIPLE: u32 = 32;
/// Narrowest accepted width-to-height ratio, 1:4.
pub const MIN_ASPECT_RATIO: f64 = 0.25;
/// Widest accepted width-to-height ratio, 4:1.
pub const MAX_ASPECT_RATIO: f64 = 4.0;
/// The ratio `auto` means for `t2va` and `ref2va`.
const DEFAULT_ASPECT_RATIO: (u32, u32) = (16, 9);

/// Shortest explicit video duration, in seconds, the MiniMax-H3 API admits.
pub const MIN_VIDEO_SECONDS: f64 = 4.0;
/// Longest explicit video duration, in seconds, the MiniMax-H3 API admits. A
/// deployment may serve a lower maximum, never a higher one.
pub const MAX_VIDEO_SECONDS: f64 = 15.0;

/// The audio VAE's sample rate and hop: 40 latents per second.
pub const AUDIO_SAMPLE_RATE: u32 = 32_000;
/// Samples per audio latent.
pub const AUDIO_HOP: u32 = 800;
/// Stereo channels, packed channel-major.
const AUDIO_CHANNELS: u32 = 2;

/// The rate the conditioner samples a reference video at.
const VIDEO_SAMPLE_FPS: f64 = 2.0;

/// The request field a rejection names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestField {
    /// `task`.
    Task,
    /// `prompt`.
    Prompt,
    /// `target.short_edge`.
    TargetShortEdge,
    /// `target.aspect_ratio`.
    TargetAspectRatio,
    /// `target.duration_seconds`.
    TargetDuration,
    /// The condition list as a whole: counts and keyframe order.
    Conditions,
    /// One condition, by its position in the request.
    Condition(usize),
}

impl RequestField {
    /// The OpenAI error `param` of the field: the field path, with a
    /// condition reported as `conditions` (the message names its index).
    pub const fn param(self) -> &'static str {
        match self {
            Self::Task => "task",
            Self::Prompt => "prompt",
            Self::TargetShortEdge => "target.short_edge",
            Self::TargetAspectRatio => "target.aspect_ratio",
            Self::TargetDuration => "target.duration_seconds",
            Self::Conditions | Self::Condition(_) => "conditions",
        }
    }
}

impl fmt::Display for RequestField {
    /// The field path; a condition is `conditions[i]`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Condition(index) => write!(formatter, "conditions[{index}]"),
            field => formatter.write_str(field.param()),
        }
    }
}

/// An aspect the canvas rule cannot resolve ([`canvas`]).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum CanvasError {
    /// A side is zero, negative or not a number.
    #[error("aspect must be positive, got {width}:{height}")]
    NonPositive {
        /// The aspect's width.
        width: f64,
        /// The aspect's height.
        height: f64,
    },
    /// The width-to-height ratio lies outside 1:4 to 4:1.
    #[error("aspect {width}:{height} lies outside 1:4 to 4:1")]
    OutOfRange {
        /// The aspect's width.
        width: f64,
        /// The aspect's height.
        height: f64,
    },
}

/// A duration or media fact the rules cannot size.
///
/// The messages name what is wrong in request terms; [`plan_request`] and
/// [`check_request`] attach the request field at fault
/// ([`VideoInputError::Invalid`]).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum RuleError {
    /// A duration outside `[MIN_VIDEO_SECONDS, max_seconds]`, or not finite.
    #[error(
        "video duration must be a finite number of seconds in [{MIN_VIDEO_SECONDS}, {max_seconds}]"
    )]
    Duration {
        /// The longest duration served.
        max_seconds: f64,
    },
    /// An image or video frame whose sides differ by more than 200 times.
    #[error("an aspect of {width}x{height} exceeds 200:1")]
    VisionAspect {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// Video frames smaller than one merged vision token.
    #[error("video frames of {width}x{height} are too small")]
    VisionFrames {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// An image reference outside 1:4 to 4:1.
    #[error("a reference image must lie within 1:4 and 4:1, got {width}x{height}")]
    ReferenceAspect {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// A reference video too short for one group of conditioner samples.
    #[error(
        "a reference video must sample at least {required} frames at {VIDEO_SAMPLE_FPS} fps, got {sampled}"
    )]
    SampledFrames {
        /// Samples one group holds.
        required: usize,
        /// Samples the video yields.
        sampled: usize,
    },
    /// A start offset at or past the end of a soundtrack.
    #[error("start_time_seconds lies past the end of the audio track")]
    AudioStart,
    /// A soundtrack whose latent count exceeds `u32`.
    #[error("the audio track is too long")]
    AudioLength,
    /// A reference video with fewer than one VAE clip of frames after its
    /// start offset.
    #[error(
        "a reference video needs at least {min_frames} frames at {frame_rate} fps after start_time_seconds, got {available}"
    )]
    ReferenceFrames {
        /// Frames left on the output-rate timeline.
        available: u64,
        /// The fewest frames the VAE encodes: one clip and the tail.
        min_frames: u32,
        /// The output frame rate the timeline is counted at.
        frame_rate: u32,
    },
    /// A reference video whose display aspect sets no canvas.
    #[error("the video's {0}")]
    VideoCanvas(CanvasError),
}

/// A checkpoint's processor configs that [`ProcessorConfig::read`] cannot use.
#[derive(Debug, thiserror::Error)]
pub enum ProcessorConfigError {
    /// A config file could not be read.
    #[error("reading {}", path.display())]
    Read {
        /// The file.
        path: PathBuf,
        /// The read's failure.
        #[source]
        source: io::Error,
    },
    /// A config file is not a processor config.
    #[error("parsing {}", path.display())]
    Parse {
        /// The file.
        path: PathBuf,
        /// The parse's failure.
        #[source]
        source: serde_json::Error,
    },
    /// The image and video processors patch differently.
    #[error("the image and video processors disagree on the patch geometry")]
    Geometry,
}

/// A video request the rules reject, or a failure while preparing valid
/// input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VideoInputError {
    /// The request breaks a rule; the same request fails again on retry.
    #[error("{field}: {message}")]
    Invalid {
        /// The field at fault.
        field: RequestField,
        /// What is wrong with it.
        message: String,
    },
    /// Preparing valid input failed: for example `ffprobe` could not run,
    /// scratch storage failed or the tokenizer lacks a vision token.
    #[error("{0}")]
    Internal(String),
}

impl VideoInputError {
    /// Rejects `field` with `message`.
    pub fn invalid(field: RequestField, message: impl Into<String>) -> Self {
        Self::Invalid {
            field,
            message: message.into(),
        }
    }

    /// Rejects `conditions[index]` with `message`.
    pub fn condition(index: usize, message: impl Into<String>) -> Self {
        Self::invalid(RequestField::Condition(index), message)
    }

    /// Reports a failure while preparing valid input.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    /// The field a rejection names; `None` for a failure.
    pub const fn field(&self) -> Option<RequestField> {
        match self {
            Self::Invalid { field, .. } => Some(*field),
            Self::Internal(_) => None,
        }
    }
}

/// The media type of a condition, by its request name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionType {
    /// A JPEG, PNG or WEBP image.
    Image,
    /// A video; its soundtrack is a condition too when it has one.
    Video,
    /// A video that must have a soundtrack.
    VideoAudio,
    /// A WAV or MP3 audio file.
    Audio,
}

impl ConditionType {
    /// Every condition type.
    pub const ALL: [Self; 4] = [Self::Image, Self::Video, Self::VideoAudio, Self::Audio];

    /// The type's request name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
            Self::VideoAudio => "video_audio",
            Self::Audio => "audio",
        }
    }

    /// The type a request name denotes, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }
}

/// Whether a condition anchors a generated frame or is a reference: a
/// condition's `role` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Anchors the first or last generated frame.
    Keyframe,
    /// Conditions the generation as a whole.
    Reference,
}

impl Role {
    /// The role's request name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Keyframe => "keyframe",
            Self::Reference => "reference",
        }
    }

    /// The role a request name denotes, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Keyframe, Self::Reference]
            .into_iter()
            .find(|role| role.as_str() == name)
    }
}

/// The generated frame a keyframe anchors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FramePosition {
    /// `frame_index` 0.
    First,
    /// `frame_index` -1.
    Last,
}

/// The probed facts of one condition's media.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaFacts {
    /// An image.
    Image(ImageFacts),
    /// A video and its soundtrack.
    Video(VideoFacts),
    /// An audio file.
    Audio(AudioFacts),
}

/// An image as displayed, after its EXIF orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageFacts {
    /// Displayed width in pixels.
    pub width: u32,
    /// Displayed height in pixels.
    pub height: u32,
}

/// A rational frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate {
    /// Frames per `denominator` seconds.
    pub numerator: u32,
    /// Seconds per `numerator` frames.
    pub denominator: u32,
}

/// A video's display aspect and size, frame rate, frame count and
/// soundtrack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFacts {
    /// Width of the display aspect, in lowest terms. Only the ratio to
    /// `display_height` matters.
    pub display_width: u64,
    /// Height of the display aspect, in lowest terms.
    pub display_height: u64,
    /// Displayed width in pixels: the coded width scaled by the sample
    /// aspect ratio, rounded half to even, and turned with `height` by the
    /// display rotation.
    pub width: u32,
    /// Displayed height in pixels.
    pub height: u32,
    /// The stream's average frame rate.
    pub frame_rate: FrameRate,
    /// Frames a decoder outputs for the whole stream.
    pub frames: u64,
    /// The first audio stream, when the video has one.
    pub soundtrack: Option<AudioFacts>,
}

/// A soundtrack's native sample rate and decoded sample count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFacts {
    /// Samples per second.
    pub sample_rate: u32,
    /// Samples per channel a decoder outputs.
    pub samples: u64,
}

/// Denoiser rows of one latent frame of `canvas`: one per 32x32 pixel block.
pub const fn rows_per_frame(canvas: Canvas) -> u32 {
    (canvas.width / CANVAS_MULTIPLE) * (canvas.height / CANVAS_MULTIPLE)
}

/// Resolves a display aspect into a canvas with the adapt_shape_v1 rule.
///
/// Only the ratio of the arguments matters. The short edge starts at 768
/// pixels, an area above `768 * 1344` scales both sides down, and each side
/// rounds half to even to a multiple of 32, so the final area may end up
/// slightly above the cap.
///
/// # Errors
///
/// Returns [`CanvasError`] when the ratio is not finite and positive or lies
/// outside 1:4 to 4:1.
pub fn canvas(aspect_width: f64, aspect_height: f64) -> Result<Canvas, CanvasError> {
    if !(aspect_width > 0.0 && aspect_height > 0.0) {
        return Err(CanvasError::NonPositive {
            width: aspect_width,
            height: aspect_height,
        });
    }
    let ratio = aspect_width / aspect_height;
    if !(MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO).contains(&ratio) {
        return Err(CanvasError::OutOfRange {
            width: aspect_width,
            height: aspect_height,
        });
    }
    let short_edge = f64::from(CANVAS_SHORT_EDGE);
    let (mut width, mut height) = if ratio >= 1.0 {
        (short_edge * ratio, short_edge)
    } else {
        (short_edge, short_edge / ratio)
    };
    let area = width * height;
    let max_pixels = f64::from(CANVAS_MAX_PIXELS);
    if area > max_pixels {
        // `powf(0.5)` rather than `sqrt` reproduces the reference's `** 0.5`.
        let scale = (max_pixels / area).powf(0.5);
        width *= scale;
        height *= scale;
    }
    Ok(Canvas {
        width: nearest_multiple(width),
        height: nearest_multiple(height),
    })
}

/// Rounds a side half to even to the nearest multiple of 32, at least 32.
fn nearest_multiple(value: f64) -> u32 {
    let multiple = f64::from(CANVAS_MULTIPLE);
    // Sides stay far below `u32::MAX`, so the conversion is exact.
    ((value / multiple).round_ties_even() * multiple).max(multiple) as u32
}

/// Every canvas the canvas rule resolves a display aspect to, each once,
/// from the narrowest aspect to the widest.
///
/// [`canvas`] is piecewise constant in the ratio: a side changes only where
/// its unrounded length crosses an odd multiple of half
/// [`CANVAS_MULTIPLE`]. Below the area cap the long side is the short edge
/// times the ratio (its reciprocal for a portrait aspect); above it the long
/// side is `sqrt(cap * ratio)` and the short side `sqrt(cap / ratio)`.
/// Those crossings, the ends of the range and the ratio the cap starts at
/// cut 1:4 to 4:1 into intervals of one canvas each, so the canvases at the
/// cuts and at the midpoints between them are all of them.
pub fn rule_canvases() -> Vec<Canvas> {
    let short_edge = f64::from(CANVAS_SHORT_EDGE);
    let max_pixels = f64::from(CANVAS_MAX_PIXELS);
    let multiple = f64::from(CANVAS_MULTIPLE);
    let cap = max_pixels / (short_edge * short_edge);
    let mut cuts = vec![MIN_ASPECT_RATIO, MAX_ASPECT_RATIO, 1.0, cap, 1.0 / cap];
    // No side of the range is longer than the long side at 4:1 under the cap.
    let longest = (max_pixels * MAX_ASPECT_RATIO).sqrt();
    let mut crossing = multiple / 2.0;
    while crossing <= longest + multiple {
        for ratio in [
            crossing / short_edge,
            crossing * crossing / max_pixels,
            max_pixels / (crossing * crossing),
        ] {
            cuts.extend([ratio, 1.0 / ratio]);
        }
        crossing += multiple;
    }
    cuts.retain(|ratio| (MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO).contains(ratio));
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();

    let midpoints: Vec<f64> = cuts
        .windows(2)
        .map(|pair| f64::midpoint(pair[0], pair[1]))
        .collect();
    let mut probes = cuts;
    probes.extend(midpoints);
    probes.sort_by(f64::total_cmp);
    let mut canvases = Vec::new();
    // Every probe lies within 1:4 to 4:1, which the rule accepts.
    for size in probes
        .into_iter()
        .filter_map(|ratio| canvas(ratio, 1.0).ok())
    {
        if !canvases.contains(&size) {
            canvases.push(size);
        }
    }
    canvases
}

/// Frames of `seconds` of video at `fps`: the product rounded half to even,
/// as the reference implementations round with Python's `round`.
pub fn requested_frames(seconds: f64, fps: u32) -> u32 {
    // Callers pass validated durations, whose products stay far below 2^32.
    (seconds * f64::from(fps)).round_ties_even() as u32
}

/// Converts an explicit duration into its output frame count at
/// `frame_rate` frames per second, the rate the denoiser declares.
///
/// The requested frame count is `seconds * frame_rate` rounded half to even
/// ([`requested_frames`]), and the output extends it upward to the next
/// frame count of the model's frame rule `frames` ([`FrameClips::align_up`]),
/// `17n + 5` for the released checkpoint. The output may therefore last
/// longer than requested: at 24 fps, 4 seconds yields 107 frames and 15
/// seconds 362.
///
/// # Errors
///
/// Returns [`RuleError::Duration`] when `seconds` is not finite or lies
/// outside `[MIN_VIDEO_SECONDS, max_video_seconds]`.
pub fn video_frame_count(
    seconds: f64,
    max_video_seconds: f64,
    frame_rate: u32,
    frames: &FrameClips,
) -> Result<u32, RuleError> {
    if !seconds.is_finite() || !(MIN_VIDEO_SECONDS..=max_video_seconds).contains(&seconds) {
        return Err(RuleError::Duration {
            max_seconds: max_video_seconds,
        });
    }
    Ok(frames.align_up(requested_frames(seconds, frame_rate)))
}

/// The requested output, as the request states it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target<'a> {
    /// `target.short_edge`: the resolution class, in pixels, of a canvas the
    /// deployment serves; 768 is the canvas rule's.
    pub short_edge: u32,
    /// `target.aspect_ratio`: `auto` or `W:H`.
    pub aspect_ratio: &'a str,
    /// `target.duration_seconds`, when given.
    pub duration_seconds: Option<f64>,
}

/// One condition's request fields other than its URI.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionSpec {
    /// `conditions[i].type`.
    pub condition_type: ConditionType,
    /// `conditions[i].role`.
    pub role: Role,
    /// `conditions[i].frame_index`: 0 or -1 for a keyframe.
    pub frame_index: Option<i64>,
    /// `conditions[i].start_time_seconds`, for video references.
    pub start_seconds: Option<f64>,
}

/// How a deployment sizes the references of a `ref2va` request: the options
/// of the MiniMax-H3 video processor (`mm_processor_kwargs`), resolved
/// against what the video denoiser declares ([`ReferenceLimits`]).
///
/// A reference is only ever reduced by default: an image whose short edge
/// exceeds `image_short_edge` scales down to it, and a video with more
/// pixels than the canvas of its aspect scales down to that canvas, while
/// smaller references keep their own size on the 32-pixel grid. The released
/// pipeline also enlarges smaller ones to those sizes, which `enlarge`
/// selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceConfig {
    /// Short edge, in pixels, an image reference is encoded at, at most: a
    /// multiple of 32 up to the edge the denoiser declares.
    pub image_short_edge: u32,
    /// Whether a reference smaller than its encoding size is enlarged to it:
    /// an image to `image_short_edge` and a video to the canvas of its
    /// aspect.
    pub enlarge: bool,
}

/// The video processor's options as a deployment states them, each absent
/// one taking its default.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceOptions {
    reference_image_short_edge: Option<u32>,
    #[serde(default)]
    enlarge_references: bool,
}

impl ReferenceConfig {
    /// The default sizing of a deployment whose denoiser declares `limits`:
    /// images reduced to the declared edge, nothing enlarged.
    pub fn new(limits: &ReferenceLimits) -> Self {
        Self {
            image_short_edge: limits.image_short_edge,
            enlarge: false,
        }
    }

    /// The released pipeline's sizing: every reference at its encoding size
    /// (images at the declared edge), enlarged when smaller.
    pub fn released(limits: &ReferenceLimits) -> Self {
        Self {
            enlarge: true,
            ..Self::new(limits)
        }
    }

    /// Reads the config from the processor options a deployment states
    /// (`mm_processor_kwargs`): `reference_image_short_edge`, by default
    /// the edge the denoiser declares, and `enlarge_references`, by default
    /// false.
    ///
    /// # Errors
    ///
    /// Returns a message for an option this processor does not take, a value
    /// of the wrong type, or a short edge [`ReferenceConfig::validate`]
    /// refuses.
    pub fn from_kwargs(
        kwargs: &serde_json::Map<String, serde_json::Value>,
        limits: &ReferenceLimits,
    ) -> Result<Self, String> {
        let options = ReferenceOptions::deserialize(serde_json::Value::Object(kwargs.clone()))
            .map_err(|error| error.to_string())?;
        let config = Self {
            image_short_edge: options
                .reference_image_short_edge
                .unwrap_or(limits.image_short_edge),
            enlarge: options.enlarge_references,
        };
        config.validate(limits)?;
        Ok(config)
    }

    /// Checks that the short edge is a multiple of 32 from 32 pixels to the
    /// edge the denoiser declares; a denoiser that takes no image reference
    /// sizes none.
    ///
    /// # Errors
    ///
    /// Returns a message naming the short edge.
    pub fn validate(&self, limits: &ReferenceLimits) -> Result<(), String> {
        if limits.images == 0 {
            return Ok(());
        }
        let edge = self.image_short_edge;
        let largest = limits.image_short_edge;
        if !edge.is_multiple_of(CANVAS_MULTIPLE) || !(CANVAS_MULTIPLE..=largest).contains(&edge) {
            return Err(format!(
                "a reference image short edge must be a multiple of {CANVAS_MULTIPLE} from \
                 {CANVAS_MULTIPLE} to {largest} pixels, got {edge}"
            ));
        }
        Ok(())
    }
}

/// What the deployment serves, narrowing the model's own rules.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanLimits {
    /// The tasks of the placed denoiser.
    pub tasks: Vec<VideoTask>,
    /// Longest requested duration served (`video_config.max_seconds`).
    pub max_video_seconds: f64,
    /// Frame rate of every generated video, in frames per second, as the
    /// denoiser declares it: durations and reference videos are counted on
    /// it.
    pub frame_rate: u32,
    /// How the denoiser's VAE divides a video's frames into clips, as it
    /// declares it: the frame counts it generates and a reference video's
    /// encoded frames and latent units follow from it.
    pub frames: FrameClips,
    /// The only canvases the deployment serves, as its workers declare them
    /// in their handshake: each with the `target.short_edge` and
    /// `target.aspect_ratio` that select it, in the order the deployment
    /// names them.
    pub canvases: Vec<ServedCanvas>,
    /// The references one `ref2va` request may carry, as the denoiser
    /// declares them in its handshake.
    pub reference_limits: ReferenceLimits,
    /// How the deployment sizes references.
    pub references: ReferenceConfig,
}

impl PlanLimits {
    /// The `target.short_edge` values the deployment serves, in the order of
    /// its canvases.
    pub fn short_edges(&self) -> Vec<u32> {
        let mut edges = Vec::new();
        for served in &self.canvases {
            if !edges.contains(&served.short_edge) {
                edges.push(served.short_edge);
            }
        }
        edges
    }
}

/// The Qwen3-VL processor geometry: patching and pixel budgets.
///
/// It is the image and video processors' configuration, as transformers and
/// SGLang name it (their `VisionConfig` is the vision tower's model
/// configuration). Read from the checkpoint's
/// `processor/preprocessor_config.json` and
/// `processor/video_preprocessor_config.json` with [`ProcessorConfig::read`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Pixels per patch side.
    pub patch_size: u32,
    /// Frames merged into one temporal patch.
    pub temporal_patch_size: u32,
    /// Patches per merged vision token side.
    pub merge_size: u32,
    /// Lower pixel budget of an image.
    pub image_min_pixels: u64,
    /// Upper pixel budget of an image.
    pub image_max_pixels: u64,
    /// Lower budget of a video: frames times pixels.
    pub video_min_pixels: u64,
    /// Upper budget of a video: frames times pixels.
    pub video_max_pixels: u64,
}

#[derive(Deserialize)]
struct ProcessorFile {
    patch_size: u32,
    temporal_patch_size: u32,
    merge_size: u32,
    size: ProcessorSize,
}

#[derive(Deserialize)]
struct ProcessorSize {
    shortest_edge: u64,
    longest_edge: u64,
}

impl ProcessorConfig {
    /// Reads a checkpoint's image processor config
    /// (`processor/preprocessor_config.json`) and video processor config
    /// (`processor/video_preprocessor_config.json`).
    ///
    /// # Errors
    ///
    /// Returns [`ProcessorConfigError`] when a file cannot be read or parsed, or
    /// when the two configs disagree on the patch geometry.
    pub fn read(image: &Path, video: &Path) -> Result<Self, ProcessorConfigError> {
        let read = |path: &Path| -> Result<ProcessorFile, ProcessorConfigError> {
            let text =
                std::fs::read_to_string(path).map_err(|source| ProcessorConfigError::Read {
                    path: path.to_owned(),
                    source,
                })?;
            serde_json::from_str(&text).map_err(|source| ProcessorConfigError::Parse {
                path: path.to_owned(),
                source,
            })
        };
        let image = read(image)?;
        let video = read(video)?;
        let geometry =
            |file: &ProcessorFile| (file.patch_size, file.temporal_patch_size, file.merge_size);
        if geometry(&image) != geometry(&video) {
            return Err(ProcessorConfigError::Geometry);
        }
        Ok(Self {
            patch_size: image.patch_size,
            temporal_patch_size: image.temporal_patch_size,
            merge_size: image.merge_size,
            image_min_pixels: image.size.shortest_edge,
            image_max_pixels: image.size.longest_edge,
            video_min_pixels: video.size.shortest_edge,
            video_max_pixels: video.size.longest_edge,
        })
    }

    /// What a resized side is a multiple of: one merged token.
    const fn factor(&self) -> u32 {
        self.patch_size * self.merge_size
    }

    /// The patch grid the image processor resizes an image to.
    ///
    /// This is Qwen2-VL's `smart_resize`: each side rounds half to even to
    /// the factor, and a size outside the pixel budget is rescaled with floor
    /// (above) or ceil (below) rounding.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::VisionAspect`] for an aspect beyond 200:1.
    pub fn image_grid(&self, height: u32, width: u32) -> Result<VisionGrid, RuleError> {
        check_vision_aspect(height, width)?;
        let factor = f64::from(self.factor());
        let (height, width) = (f64::from(height), f64::from(width));
        let mut resized_height = (height / factor).round_ties_even() * factor;
        let mut resized_width = (width / factor).round_ties_even() * factor;
        if resized_height * resized_width > self.image_max_pixels as f64 {
            let beta = (height * width / self.image_max_pixels as f64).sqrt();
            resized_height = factor.max((height / beta / factor).floor() * factor);
            resized_width = factor.max((width / beta / factor).floor() * factor);
        } else if resized_height * resized_width < self.image_min_pixels as f64 {
            let beta = (self.image_min_pixels as f64 / (height * width)).sqrt();
            resized_height = (height * beta / factor).ceil() * factor;
            resized_width = (width * beta / factor).ceil() * factor;
        }
        Ok(self.grid(1, resized_height, resized_width))
    }

    /// The patch grid of `frames` sampled video frames.
    ///
    /// This is Qwen3-VL's video `smart_resize`: the budget covers all frames,
    /// which are padded to a whole temporal patch by repeating the last one.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::VisionFrames`] for frames smaller than one merged
    /// token and [`RuleError::VisionAspect`] for an aspect beyond 200:1.
    pub fn video_grid(
        &self,
        frames: u32,
        height: u32,
        width: u32,
    ) -> Result<VisionGrid, RuleError> {
        let factor = self.factor();
        if height < factor || width < factor {
            return Err(RuleError::VisionFrames { width, height });
        }
        check_vision_aspect(height, width)?;
        let temporal = self.temporal_patch_size;
        let padded_frames = frames.div_ceil(temporal) * temporal;
        let factor = f64::from(factor);
        let pixels = f64::from(frames) * f64::from(height) * f64::from(width);
        let (height, width) = (f64::from(height), f64::from(width));
        let mut resized_height = (height / factor).round_ties_even() * factor;
        let mut resized_width = (width / factor).round_ties_even() * factor;
        let budget = f64::from(padded_frames) * resized_height * resized_width;
        if budget > self.video_max_pixels as f64 {
            let beta = (pixels / self.video_max_pixels as f64).sqrt();
            resized_height = factor.max((height / beta / factor).floor() * factor);
            resized_width = factor.max((width / beta / factor).floor() * factor);
        } else if budget < self.video_min_pixels as f64 {
            let beta = (self.video_min_pixels as f64 / pixels).sqrt();
            resized_height = (height * beta / factor).ceil() * factor;
            resized_width = (width * beta / factor).ceil() * factor;
        }
        Ok(self.grid(padded_frames / temporal, resized_height, resized_width))
    }

    /// Vision tokens of one temporal block of `grid`.
    pub const fn block_tokens(&self, grid: VisionGrid) -> u32 {
        grid.h * grid.w / (self.merge_size * self.merge_size)
    }

    fn grid(&self, t: u32, resized_height: f64, resized_width: f64) -> VisionGrid {
        // Resized sides are positive multiples of the factor below 2^32.
        VisionGrid {
            t,
            h: resized_height as u32 / self.patch_size,
            w: resized_width as u32 / self.patch_size,
        }
    }
}

fn check_vision_aspect(height: u32, width: u32) -> Result<(), RuleError> {
    let (long, short) = (height.max(width), height.min(width));
    if f64::from(long) / f64::from(short) > 200.0 {
        return Err(RuleError::VisionAspect { width, height });
    }
    Ok(())
}

/// An aspect-preserving resize followed by a centred crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CoverCrop {
    /// Width the image is resized to before cropping.
    pub width: u32,
    /// Height the image is resized to before cropping.
    pub height: u32,
    /// Left edge of the crop in the resized image.
    pub left: u32,
    /// Top edge of the crop in the resized image.
    pub top: u32,
}

/// Returns the LANCZOS resize and centred crop of a `width` x `height` image
/// that cover `canvas`.
///
/// The scale is the larger of the two side ratios; each resized side rounds
/// half to even and never falls below the canvas, and the crop is centred
/// with floor division.
pub fn cover_crop(width: u32, height: u32, canvas: Canvas) -> CoverCrop {
    let scale = (f64::from(canvas.width) / f64::from(width))
        .max(f64::from(canvas.height) / f64::from(height));
    let resized_width = canvas
        .width
        .max((f64::from(width) * scale).round_ties_even() as u32);
    let resized_height = canvas
        .height
        .max((f64::from(height) * scale).round_ties_even() as u32);
    CoverCrop {
        width: resized_width,
        height: resized_height,
        left: (resized_width - canvas.width) / 2,
        top: (resized_height - canvas.height) / 2,
    }
}

/// How a keyframe is put on the target canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct KeyframeFit {
    /// The generated frame the keyframe anchors.
    pub position: FramePosition,
    /// `None` when the keyframe is stretched onto the canvas (the first
    /// keyframe of a request); the crop otherwise.
    pub cover_crop: Option<CoverCrop>,
}

/// The samples of a soundtrack the request conditions on: the media reader's
/// [`super::AudioClip`] and the latents it encodes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AudioClip {
    /// The native sample rate.
    pub sample_rate: u32,
    /// Native samples skipped at the start.
    pub start_sample: u64,
    /// Native samples kept after the start offset.
    pub source_samples: u64,
    /// Samples after resampling to 32 kHz.
    pub samples: u64,
    /// Audio latents per channel; the clip packs `2 * latents` channel-major
    /// rows.
    pub latents: u32,
}

impl AudioClip {
    /// Denoiser rows of the clip: one per latent per stereo channel.
    pub const fn rows(&self) -> u32 {
        AUDIO_CHANNELS * self.latents
    }

    /// What the media reader decodes.
    const fn media(&self) -> super::AudioClip {
        super::AudioClip {
            sample_rate: self.sample_rate,
            start_sample: self.start_sample,
            source_samples: self.source_samples,
            samples: self.samples,
        }
    }
}

/// The frames of a reference video the request conditions on: the media
/// reader's [`super::VideoClip`], the latent frames it encodes to and its
/// soundtrack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VideoClip {
    /// The canvas of the video's own display aspect the frames are put on.
    pub canvas: Canvas,
    /// Frames of the output-rate timeline skipped at the start.
    pub start_frame: u32,
    /// Output-rate frames kept after the start offset, at most the generated
    /// frame count; the conditioner samples these.
    pub frames: u32,
    /// The leading frames the VAE encodes, a frame count of `vae_clips`.
    pub vae_frames: u32,
    /// How the VAE divides those frames into clips.
    pub vae_clips: FrameClips,
    /// The video's soundtrack, when it has one.
    pub soundtrack: Option<AudioClip>,
}

/// The conditioner's view of one image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImageVision {
    /// The patch grid; `t` is 1.
    pub grid: VisionGrid,
    /// Vision tokens of the image.
    pub tokens: u32,
}

/// The conditioner's view of one reference video.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VideoVision {
    /// The patch grid; `t` counts vision blocks.
    pub grid: VisionGrid,
    /// Vision tokens of each block.
    pub block_tokens: u32,
    /// The output-rate frames sampled at 2 fps.
    pub frame_indices: Vec<u32>,
    /// The timestamp label of each block, in seconds.
    pub block_timestamps: Vec<f64>,
}

/// What the conditioner reads of a condition.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Vision {
    /// One image block.
    Image(ImageVision),
    /// Timestamped video blocks.
    Video(VideoVision),
}

/// How a condition's media is prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Prepared {
    /// A keyframe fitted to the target canvas.
    Keyframe(KeyframeFit),
    /// An image reference resized (LANCZOS) to this size.
    Image(Canvas),
    /// A reference video clip and its soundtrack.
    Video(VideoClip),
    /// An audio reference.
    Audio(AudioClip),
}

/// How one condition is prepared and what it contributes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConditionPlan {
    /// Position of the condition in the request.
    pub index: usize,
    /// The preparation of its media.
    pub prepared: Prepared,
    /// The conditioner's view, for conditions it reads.
    pub vision: Option<Vision>,
    /// Denoiser audio condition rows.
    pub audio_rows: u32,
}

impl ConditionPlan {
    /// The condition's described role and what the media reader decodes its
    /// bytes into. `canvas` is the generated canvas, which a keyframe is
    /// fitted to.
    fn media(&self, canvas: Canvas) -> (super::ConditionRole, ConditionMedia) {
        match &self.prepared {
            Prepared::Keyframe(fit) => {
                let role = match fit.position {
                    FramePosition::First => super::ConditionRole::FirstFrame,
                    FramePosition::Last => super::ConditionRole::LastFrame,
                };
                // A stretched keyframe is resized straight onto the canvas; a
                // cover-cropped one past it, then centred.
                let fit = match fit.cover_crop {
                    None => ImageFit {
                        resized: canvas,
                        left: 0,
                        top: 0,
                        size: canvas,
                    },
                    Some(crop) => ImageFit {
                        resized: Canvas {
                            width: crop.width,
                            height: crop.height,
                        },
                        left: crop.left,
                        top: crop.top,
                        size: canvas,
                    },
                };
                (role, ConditionMedia::Image(fit))
            }
            Prepared::Image(size) => (
                super::ConditionRole::Reference,
                ConditionMedia::Image(ImageFit {
                    resized: *size,
                    left: 0,
                    top: 0,
                    size: *size,
                }),
            ),
            Prepared::Video(clip) => (
                super::ConditionRole::Reference,
                ConditionMedia::Video {
                    clip: super::VideoClip {
                        canvas: clip.canvas,
                        start_frame: clip.start_frame,
                        frames: clip.frames,
                        vae_frames: clip.vae_frames,
                    },
                    soundtrack: clip.soundtrack.as_ref().map(AudioClip::media),
                },
            ),
            Prepared::Audio(clip) => (
                super::ConditionRole::Reference,
                ConditionMedia::Audio(clip.media()),
            ),
        }
    }

    /// Denoiser video rows of each unit the video encoder encodes, in unit
    /// order. `canvas` is the generated canvas.
    ///
    /// A keyframe is one unit, one latent frame of the canvas it is fitted
    /// to. A reference image is `image_bands` units: bands of its rows of
    /// 32x32-pixel patches dealt as evenly as possible in order, so with `g`
    /// patch rows band `b` holds rows `b * g / bands` to
    /// `(b + 1) * g / bands`, rounded down, `image_bands` clamped to one
    /// through `g`. Each band is encoded on its own and the bands assemble
    /// the whole image's encoding; a rank finds a band's patch rows from the
    /// rows of the bands before it. A video's encoded frames are one unit
    /// per clip of its frame rule and one for the tail: each clip yields the
    /// rule's clip latent frames and the tail its tail latent frames (5 and
    /// 2 for the released checkpoint, whose video encoder drops the leading
    /// 3 latent frames of its padded final window). An audio reference has
    /// none.
    pub fn latent_units(&self, canvas: Canvas, image_bands: u32) -> Vec<u32> {
        match &self.prepared {
            Prepared::Keyframe(_) => vec![rows_per_frame(canvas)],
            Prepared::Image(size) => {
                let patch_rows = size.height / CANVAS_MULTIPLE;
                let patches_per_row = size.width / CANVAS_MULTIPLE;
                let bands = image_bands.clamp(1, patch_rows.max(1));
                (0..bands)
                    .map(|band| {
                        let start = band * patch_rows / bands;
                        let stop = (band + 1) * patch_rows / bands;
                        (stop - start) * patches_per_row
                    })
                    .collect()
            }
            Prepared::Video(clip) => {
                let rows_per_frame = rows_per_frame(clip.canvas);
                let clips = clip.vae_clips;
                let mut units = vec![
                    clips.clip_latent_frames * rows_per_frame;
                    clips.units(clip.vae_frames) as usize
                ];
                units.push(clips.tail_latent_frames * rows_per_frame);
                units
            }
            Prepared::Audio(_) => Vec::new(),
        }
    }

    /// Describes the condition for the engine and the workers: its role, the
    /// published media at `source`, what the media reader decodes it into,
    /// what the conditioner reads of it, and the denoiser rows it encodes to,
    /// a reference image in `image_bands` units ([`Self::latent_units`]).
    /// `canvas` is the generated canvas, which a keyframe is fitted to.
    pub fn describe(
        &self,
        canvas: Canvas,
        source: MediaLocator,
        image_bands: u32,
    ) -> VideoCondition {
        let (role, media) = self.media(canvas);
        let vision = self.vision.as_ref().map(|vision| match vision {
            Vision::Image(image) => ConditionVision {
                grid: image.grid,
                tokens: image.tokens,
                frame_indices: Vec::new(),
            },
            Vision::Video(video) => ConditionVision {
                grid: video.grid,
                tokens: video.grid.t * video.block_tokens,
                frame_indices: video.frame_indices.clone(),
            },
        });
        VideoCondition {
            role,
            source,
            media,
            vision,
            latent_units: self.latent_units(canvas, image_bands),
            audio_rows: self.audio_rows,
        }
    }
}

/// The resolved size of a request and of every condition.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RequestPlan {
    /// The request's task.
    pub task: VideoTask,
    /// The generated canvas.
    pub canvas: Canvas,
    /// The requested duration in seconds: `target.duration_seconds`, or the
    /// soundtrack's length after its start offset when that sets it.
    pub duration_seconds: f64,
    /// Generated frames, a frame count of the denoiser's frame rule.
    pub num_frames: u32,
    /// One plan per condition, in request order.
    pub conditions: Vec<ConditionPlan>,
}

impl RequestPlan {
    /// The units each condition's reference image is encoded in when one
    /// latent encoding round covers `lane` units, one entry per condition
    /// (other conditions' entries are unread).
    ///
    /// A lone reference image is the largest visual condition and is a
    /// single rank's work as one unit; its bands spread its encoder tiles
    /// over the ranks of one round. Fewer images than the lane divide it
    /// evenly, at least one unit each. More images than the lane encode
    /// whole, a lane's worth per round, except that when every visual unit
    /// is an image, the `k` images of a final partial round take `lane / k`
    /// bands each whenever `k` divides the lane, so that round is full
    /// rather than leaving ranks idle behind its whole images.
    pub fn image_bands(&self, lane: u32) -> Vec<u32> {
        let is_image = |plan: &ConditionPlan| matches!(plan.prepared, Prepared::Image(_));
        let images = self.conditions.iter().filter(|plan| is_image(plan)).count() as u32;
        let mut bands = vec![(lane / images.max(1)).max(1); self.conditions.len()];
        let only_images = self
            .conditions
            .iter()
            .all(|plan| is_image(plan) || matches!(plan.prepared, Prepared::Audio(_)));
        let partial = images % lane.max(1);
        if only_images && images > lane && partial > 0 && lane.is_multiple_of(partial) {
            // The last `partial` images, in request order, form the final
            // round.
            let mut remaining = partial;
            for (entry, plan) in bands.iter_mut().zip(&self.conditions).rev() {
                if remaining == 0 {
                    break;
                }
                if is_image(plan) {
                    *entry = lane / partial;
                    remaining -= 1;
                }
            }
        }
        bands
    }
}

/// Returns the size an image reference is encoded at.
///
/// An image whose short edge exceeds `config.image_short_edge` scales down
/// to it; a smaller one keeps its size, or scales up to that edge under
/// `config.enlarge`. Each side then rounds half to even to a multiple of 32.
/// Unlike the canvas rule there is no area cap.
///
/// # Errors
///
/// Returns [`RuleError::ReferenceAspect`] when the image lies outside 1:4
/// to 4:1.
pub fn reference_image_size(
    width: u32,
    height: u32,
    config: &ReferenceConfig,
) -> Result<Canvas, RuleError> {
    if u64::from(width) > 4 * u64::from(height) || u64::from(height) > 4 * u64::from(width) {
        return Err(RuleError::ReferenceAspect { width, height });
    }
    let mut scale = f64::from(config.image_short_edge) / f64::from(width.min(height));
    if !config.enlarge {
        scale = scale.min(1.0);
    }
    Ok(Canvas {
        width: nearest_multiple(f64::from(width) * scale),
        height: nearest_multiple(f64::from(height) * scale),
    })
}

/// Samples a clip of `frame_rate` frames per second at 2 fps and labels its
/// vision blocks.
///
/// Every `frame_rate / 2`-th frame is sampled (every twelfth at 24 fps). The
/// conditioner merges the samples in groups of `temporal_patch`, repeating
/// the last one to fill a group, and labels a group with the mean of its
/// timestamps.
///
/// # Errors
///
/// Returns [`RuleError::SampledFrames`] when fewer frames are sampled than
/// one group holds.
pub fn sample_video_frames(
    frames: u32,
    temporal_patch: u32,
    frame_rate: u32,
) -> Result<(Vec<u32>, Vec<f64>), RuleError> {
    let stride = f64::from(frame_rate) / VIDEO_SAMPLE_FPS;
    let mut indices: Vec<u32> = Vec::new();
    let mut cursor = 0.0_f64;
    loop {
        // Python's `round` rounds half to even.
        let index = cursor.round_ties_even();
        if index >= f64::from(frames) {
            break;
        }
        let index = index as u32;
        if indices.last().is_none_or(|&last| index > last) {
            indices.push(index);
        }
        cursor += stride;
    }
    let temporal_patch = temporal_patch as usize;
    if indices.len() < temporal_patch {
        return Err(RuleError::SampledFrames {
            required: temporal_patch,
            sampled: indices.len(),
        });
    }
    let mut timestamps: Vec<f64> = (0..indices.len())
        .map(|index| index as f64 / VIDEO_SAMPLE_FPS)
        .collect();
    let last = timestamps[timestamps.len() - 1];
    timestamps.resize(
        indices.len().div_ceil(temporal_patch) * temporal_patch,
        last,
    );
    let blocks = timestamps
        .chunks(temporal_patch)
        .map(|group| (group[0] + group[temporal_patch - 1]) / 2.0)
        .collect();
    Ok((indices, blocks))
}

/// Units of a `rate` timeline skipped by a start offset of `seconds`.
fn start_offset(seconds: f64, rate: f64) -> u64 {
    // Start offsets are validated finite and non-negative.
    (seconds * rate + 0.5).floor() as u64
}

/// Sizes a soundtrack: start offset, truncation, 32 kHz resampling, latents.
///
/// The first `floor(s * rate + 0.5)` samples are skipped and the rest is
/// truncated at the native rate to the generated duration, `frames` at
/// `frame_rate`, as the reference does before it resamples to 32 kHz.
///
/// # Errors
///
/// Returns [`RuleError::AudioStart`] when the offset lies past the end of
/// the soundtrack and [`RuleError::AudioLength`] when its latents exceed
/// `u32`.
pub fn audio_clip(
    media: &AudioFacts,
    start_seconds: f64,
    frames: u32,
    frame_rate: u32,
) -> Result<AudioClip, RuleError> {
    let rate = media.sample_rate;
    let start_sample = start_offset(start_seconds, f64::from(rate));
    if start_sample >= media.samples {
        return Err(RuleError::AudioStart);
    }
    let available = media.samples - start_sample;
    // The reference truncates to `int(frames / fps * rate)` samples.
    let limit = (f64::from(frames) / f64::from(frame_rate) * f64::from(rate)).trunc() as u64;
    let source = available.min(limit);
    let samples = if rate == AUDIO_SAMPLE_RATE {
        source
    } else {
        // torchaudio's resampler returns ceil(new * length / orig) samples
        // for the gcd-reduced rates.
        let divisor = gcd(u64::from(rate), u64::from(AUDIO_SAMPLE_RATE));
        let new = u64::from(AUDIO_SAMPLE_RATE) / divisor;
        let orig = u64::from(rate) / divisor;
        (source * new).div_ceil(orig)
    };
    Ok(AudioClip {
        sample_rate: rate,
        start_sample,
        source_samples: source,
        samples,
        // The audio VAE pads the waveform to a whole hop.
        latents: u32::try_from(samples.div_ceil(u64::from(AUDIO_HOP)))
            .map_err(|_| RuleError::AudioLength)?,
    })
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Sizes a reference video and the conditioner's view of it.
///
/// The video is resampled to `frame_rate` (each frame held until the slot of
/// the next), its first `floor(s * frame_rate + 0.5)` frames are skipped and
/// the rest is truncated to the generated frame count, on the canvas of its
/// own display aspect. A video with fewer pixels than that canvas keeps its
/// own displayed size, each side rounded half to even to a multiple of 32,
/// unless `config.enlarge` scales it up to the canvas. Its soundtrack is
/// offset by the same start and sized like an audio reference.
///
/// # Errors
///
/// Returns [`RuleError::VideoCanvas`] when the display aspect lies outside
/// 1:4 to 4:1, [`RuleError::ReferenceFrames`] when fewer than the frame
/// rule's fewest frames (22 for the released checkpoint) remain after the
/// offset, and the errors of [`audio_clip`],
/// [`sample_video_frames`] and [`ProcessorConfig::video_grid`].
pub fn video_reference(
    media: &VideoFacts,
    start_seconds: f64,
    frames: u32,
    frame_rate: u32,
    clips: &FrameClips,
    vision: &ProcessorConfig,
    config: &ReferenceConfig,
) -> Result<(VideoClip, VideoVision), RuleError> {
    // Display sizes stay far below 2^53, so the conversions are exact.
    let mut size = canvas(media.display_width as f64, media.display_height as f64)
        .map_err(RuleError::VideoCanvas)?;
    let pixels = |width: u32, height: u32| u64::from(width) * u64::from(height);
    if !config.enlarge && pixels(media.width, media.height) < pixels(size.width, size.height) {
        size = Canvas {
            width: nearest_multiple(f64::from(media.width)),
            height: nearest_multiple(f64::from(media.height)),
        };
    }
    let source_rate =
        f64::from(media.frame_rate.numerator) / f64::from(media.frame_rate.denominator);
    let fps = f64::from(frame_rate);
    let timeline = if source_rate == fps {
        media.frames
    } else {
        // The timeline ends at the output-rate slot the stream's end rounds
        // to.
        (media.frames as f64 * (fps / source_rate) + 0.5).floor() as u64
    };
    let start_frame = start_offset(start_seconds, fps);
    let available = timeline.saturating_sub(start_frame);
    let too_short = RuleError::ReferenceFrames {
        available,
        min_frames: clips.min_frames,
        frame_rate,
    };
    if available < u64::from(clips.min_frames) {
        return Err(too_short);
    }
    let kept = u32::try_from(available).map_or(frames, |available| available.min(frames));
    // The VAE encodes the leading frames that form a frame count of its rule.
    let vae_frames = clips.align_down(kept).ok_or(too_short)?;
    let soundtrack = media
        .soundtrack
        .as_ref()
        .map(|track| audio_clip(track, start_seconds, frames, frame_rate))
        .transpose()?;

    let (frame_indices, block_timestamps) =
        sample_video_frames(kept, vision.temporal_patch_size, frame_rate)?;
    let sampled = u32::try_from(frame_indices.len()).unwrap_or(u32::MAX);
    let grid = vision.video_grid(sampled, size.height, size.width)?;
    let clip = VideoClip {
        canvas: size,
        // Offsets past the timeline were rejected above.
        start_frame: u32::try_from(start_frame).unwrap_or(u32::MAX),
        frames: kept,
        vae_frames,
        vae_clips: *clips,
        soundtrack,
    };
    let seen = VideoVision {
        grid,
        block_tokens: vision.block_tokens(grid),
        frame_indices,
        block_timestamps,
    };
    Ok((clip, seen))
}

/// The largest references a request generating a given frame count may
/// bring, as [`reference_bounds`] states them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceBounds {
    /// The two largest rasters an image reference is encoded at: the
    /// encoding short edge at the widest (4:1) and the tallest (1:4)
    /// accepted aspect.
    pub images: [Canvas; 2],
    /// Encoded frames of the longest reference video.
    pub video_frames: u32,
    /// The largest rasters a reference video is encoded at: the canvas
    /// rule's canvases ([`rule_canvases`]); a smaller video keeps its own
    /// size, within its canvas.
    pub video_canvases: Vec<Canvas>,
    /// The most model-rate samples of a soundtrack or audio reference.
    pub audio_samples: u64,
}

/// Bounds every reference of a request that generates `frames` frames.
///
/// However a deployment sizes references ([`ReferenceConfig`]),
/// [`reference_image_size`] encodes an image's short edge at most at
/// `image_short_edge`, the edge the denoiser declares, so its sides, and the
/// rows any packing gives it, are at most those of that edge at the widest or
/// the tallest accepted aspect. [`video_reference`] keeps at most `frames` frames of a
/// video, whose leading frames forming a count of the frame rule `clips` it
/// encodes on the canvas of the video's aspect or, for a video with fewer
/// pixels than that canvas, at its own size, whose sides on the 32-pixel
/// grid are at most the canvas's.
/// [`audio_clip`] truncates a track to the generated duration and resamples
/// it to 32 kHz, so it returns at most `ceil(frames * 32000 / frame_rate)`
/// samples whatever the source rate, `frame_rate` being the output frame
/// rate the frames are counted at. A model sizes these shapes with its own
/// packing to provision the largest condition set it admits.
pub fn reference_bounds(
    frames: u32,
    frame_rate: u32,
    clips: &FrameClips,
    image_short_edge: u32,
) -> ReferenceBounds {
    let edge = image_short_edge;
    ReferenceBounds {
        images: [
            Canvas {
                width: 4 * edge,
                height: edge,
            },
            Canvas {
                width: edge,
                height: 4 * edge,
            },
        ],
        // No reference video fits a request shorter than the rule's fewest
        // frames.
        video_frames: clips.align_down(frames).unwrap_or(0),
        video_canvases: rule_canvases(),
        audio_samples: (u64::from(frames) * u64::from(AUDIO_SAMPLE_RATE))
            .div_ceil(u64::from(frame_rate)),
    }
}

/// Parses `W:H` into positive integers written in canonical decimal form;
/// `None` when malformed.
fn parse_aspect_ratio(value: &str) -> Option<(u32, u32)> {
    let (width, height) = value.split_once(':')?;
    let side = |text: &str| -> Option<u32> {
        let parsed: u32 = text.parse().ok()?;
        (parsed > 0 && parsed.to_string() == text).then_some(parsed)
    };
    Some((side(width)?, side(height)?))
}

/// The canvas a target names without consulting media; `None` for `fl2va`
/// with `auto`, whose canvas follows the first keyframe.
///
/// `canvases` are the canvases a deployment serves, each with the
/// `(short_edge, aspect_ratio)` that selects it. `auto` is 16:9 for `t2va`
/// and `ref2va`. A `W:H` ratio named at the target's short edge selects that
/// canvas. `fl2va` also takes any ratio from 1:4 to 4:1, which the canvas
/// rule ([`canvas`]) resolves at its 768-pixel short edge; a deployment
/// serves such a canvas only when it is one of `canvases`.
///
/// # Errors
///
/// Returns [`VideoInputError::Invalid`] for `target.aspect_ratio` when no
/// canvas the ratio names is served at the target's short edge and the
/// canvas rule does not apply to it.
pub fn named_canvas(
    task: VideoTask,
    target: &Target<'_>,
    canvases: &[ServedCanvas],
) -> Result<Option<Canvas>, VideoInputError> {
    let name = if target.aspect_ratio != "auto" {
        target.aspect_ratio.to_owned()
    } else if task == VideoTask::Fl2va {
        return Ok(None);
    } else {
        format!("{}:{}", DEFAULT_ASPECT_RATIO.0, DEFAULT_ASPECT_RATIO.1)
    };
    let at_edge = || {
        canvases
            .iter()
            .filter(|served| served.short_edge == target.short_edge)
    };
    if let Some(served) = at_edge().find(|served| served.aspect_ratio == name) {
        return Ok(Some(served.canvas));
    }

    let by_rule = task == VideoTask::Fl2va && target.short_edge == CANVAS_SHORT_EDGE;
    if by_rule && let Some((width, height)) = parse_aspect_ratio(&name) {
        return canvas(f64::from(width), f64::from(height))
            .map(Some)
            .map_err(|error| {
                VideoInputError::invalid(RequestField::TargetAspectRatio, error.to_string())
            });
    }

    let served: Vec<&str> = at_edge()
        .map(|served| served.aspect_ratio.as_str())
        .collect();
    if served.is_empty() && !by_rule {
        return Err(VideoInputError::invalid(
            RequestField::TargetAspectRatio,
            format!(
                "names no canvas: no canvas is served at a {}-pixel short edge, got {:?}",
                target.short_edge, target.aspect_ratio
            ),
        ));
    }
    let allowed = match (task, by_rule) {
        (VideoTask::Fl2va, true) => "auto or W:H within 1:4 to 4:1".to_owned(),
        (VideoTask::Fl2va, false) => format!("one of {}", served.join(", ")),
        _ => format!("auto or one of {}", served.join(", ")),
    };
    Err(VideoInputError::invalid(
        RequestField::TargetAspectRatio,
        format!(
            "must be {allowed} at a {}-pixel short edge, got {:?}",
            target.short_edge, target.aspect_ratio
        ),
    ))
}

/// Checks that a canvas is one the deployment's workers prepared.
fn check_served(size: Canvas, limits: &PlanLimits) -> Result<(), VideoInputError> {
    if limits.canvases.iter().any(|served| served.canvas == size) {
        return Ok(());
    }
    Err(VideoInputError::invalid(
        RequestField::TargetAspectRatio,
        format!(
            "resolves to {}x{}, which this deployment does not serve",
            size.width, size.height
        ),
    ))
}

/// Applies every rule that needs no media, in a fixed order: the served task,
/// the target's short edge and aspect ratio, each condition's fields, the
/// task's condition counts and keyframe order, then the duration.
///
/// [`plan_request`] applies these rules again; calling this first rejects a
/// malformed request before its media are fetched.
///
/// # Errors
///
/// Returns [`VideoInputError::Invalid`] naming the first field at fault.
pub fn check_request(
    task: VideoTask,
    target: &Target<'_>,
    conditions: &[ConditionSpec],
    limits: &PlanLimits,
) -> Result<(), VideoInputError> {
    if !limits.tasks.contains(&task) {
        let served: Vec<&str> = limits.tasks.iter().map(|task| task.as_str()).collect();
        return Err(VideoInputError::invalid(
            RequestField::Task,
            format!(
                "{} is not served; this deployment serves {}",
                task.as_str(),
                served.join(", ")
            ),
        ));
    }
    let short_edges = limits.short_edges();
    if !short_edges.contains(&target.short_edge) {
        let served: Vec<String> = short_edges.iter().map(u32::to_string).collect();
        return Err(VideoInputError::invalid(
            RequestField::TargetShortEdge,
            format!("must be {}, got {}", served.join(" or "), target.short_edge),
        ));
    }

    if let Some(size) = named_canvas(task, target, &limits.canvases)? {
        check_served(size, limits)?;
    }

    check_conditions(task, conditions, &limits.reference_limits)?;

    match target.duration_seconds {
        Some(seconds) => {
            video_frame_count(
                seconds,
                limits.max_video_seconds,
                limits.frame_rate,
                &limits.frames,
            )
            .map_err(|error| {
                VideoInputError::invalid(RequestField::TargetDuration, error.to_string())
            })?;
        }
        None => {
            let may_follow_audio = task == VideoTask::Ref2va
                && conditions
                    .iter()
                    .any(|condition| condition.condition_type != ConditionType::Image);
            if !may_follow_audio {
                return Err(VideoInputError::invalid(
                    RequestField::TargetDuration,
                    "is required unless a ref2va request has exactly one reference with audio",
                ));
            }
        }
    }
    Ok(())
}

/// Checks each condition's fields, then the task's counts and keyframe
/// order.
fn check_conditions(
    task: VideoTask,
    conditions: &[ConditionSpec],
    limits: &ReferenceLimits,
) -> Result<(), VideoInputError> {
    if task == VideoTask::T2va {
        if !conditions.is_empty() {
            return Err(VideoInputError::invalid(
                RequestField::Conditions,
                "t2va takes no conditions",
            ));
        }
        return Ok(());
    }

    for (index, condition) in conditions.iter().enumerate() {
        let reject = |message: String| Err(VideoInputError::condition(index, message));
        match condition.role {
            Role::Keyframe => {
                if condition.condition_type != ConditionType::Image {
                    return reject("a keyframe must be an image".to_owned());
                }
                match condition.frame_index {
                    None => return reject("a keyframe needs frame_index".to_owned()),
                    Some(0 | -1) => {}
                    Some(other) => {
                        return reject(format!("frame_index must be 0 or -1, got {other}"));
                    }
                }
            }
            Role::Reference => {
                if task == VideoTask::Fl2va {
                    return reject("fl2va takes keyframes only".to_owned());
                }
                if condition.frame_index.is_some() {
                    return reject("only a keyframe takes frame_index".to_owned());
                }
            }
        }
        if let Some(start) = condition.start_seconds {
            let video = matches!(
                condition.condition_type,
                ConditionType::Video | ConditionType::VideoAudio
            );
            if condition.role == Role::Keyframe || !video {
                return reject("start_time_seconds applies to video references only".to_owned());
            }
            if !start.is_finite() || start < 0.0 {
                return reject("start_time_seconds must be finite and >= 0".to_owned());
            }
        }
    }

    let signature: Vec<i64> = conditions
        .iter()
        .filter(|condition| condition.role == Role::Keyframe)
        .filter_map(|condition| condition.frame_index)
        .collect();
    let keyframes_valid = matches!(signature.as_slice(), [0] | [-1] | [0, -1]);
    if (task == VideoTask::Fl2va || !signature.is_empty()) && !keyframes_valid {
        return Err(VideoInputError::invalid(
            RequestField::Conditions,
            format!("keyframe frame_index values must be [0], [-1] or [0, -1], got {signature:?}"),
        ));
    }

    if task == VideoTask::Ref2va {
        let references = || {
            conditions
                .iter()
                .filter(|condition| condition.role == Role::Reference)
        };
        let count = |kind: fn(ConditionType) -> bool| {
            references()
                .filter(|condition| kind(condition.condition_type))
                .count()
        };
        let images = count(|kind| kind == ConditionType::Image);
        let audios = count(|kind| kind == ConditionType::Audio);
        let videos = count(|kind| matches!(kind, ConditionType::Video | ConditionType::VideoAudio));
        // Each limit names what it bounds; the last bounds every kind.
        for (bounded, found, limit) in [
            ("image references", images, limits.images),
            ("video references", videos, limits.videos),
            ("audio references", audios, limits.audios),
            ("references", references().count(), limits.total),
        ] {
            if found > limit as usize {
                return Err(VideoInputError::invalid(
                    RequestField::Conditions,
                    format!("at most {limit} {bounded}, got {found}"),
                ));
            }
        }
        if images + videos == 0 {
            return Err(VideoInputError::invalid(
                RequestField::Conditions,
                "ref2va needs at least one image or video reference",
            ));
        }
    }
    Ok(())
}

/// Checks that the probed media fits the condition's type.
fn check_media(
    index: usize,
    condition: &ConditionSpec,
    media: &MediaFacts,
) -> Result<(), VideoInputError> {
    let fits = matches!(
        (condition.condition_type, media),
        (ConditionType::Image, MediaFacts::Image(_))
            | (
                ConditionType::Video | ConditionType::VideoAudio,
                MediaFacts::Video(_)
            )
            | (ConditionType::Audio, MediaFacts::Audio(_))
    );
    if !fits {
        return Err(VideoInputError::condition(
            index,
            "the media does not match the condition type",
        ));
    }
    if let (ConditionType::VideoAudio, MediaFacts::Video(video)) = (condition.condition_type, media)
        && video.soundtrack.is_none()
    {
        return Err(VideoInputError::condition(
            index,
            "a video_audio reference needs an audio track",
        ));
    }
    Ok(())
}

/// The soundtrack a reference conditions on, if any.
fn soundtrack<'a>(condition: &ConditionSpec, media: &'a MediaFacts) -> Option<&'a AudioFacts> {
    if condition.role == Role::Keyframe {
        return None;
    }
    match media {
        MediaFacts::Audio(audio) => Some(audio),
        MediaFacts::Video(video) => video.soundtrack.as_ref(),
        MediaFacts::Image(_) => None,
    }
}

/// The generated frame count, from the target or from the one soundtrack.
fn resolve_frames(
    target: &Target<'_>,
    soundtracks: &[(usize, &AudioFacts, f64)],
    limits: &PlanLimits,
) -> Result<(f64, u32), VideoInputError> {
    if let Some(seconds) = target.duration_seconds {
        return video_frame_count(
            seconds,
            limits.max_video_seconds,
            limits.frame_rate,
            &limits.frames,
        )
        .map(|frames| (seconds, frames))
        .map_err(|error| {
            VideoInputError::invalid(RequestField::TargetDuration, error.to_string())
        });
    }
    let [(index, track, start_seconds)] = soundtracks else {
        return Err(VideoInputError::invalid(
            RequestField::TargetDuration,
            format!(
                "is required unless exactly one reference has audio, got {}",
                soundtracks.len()
            ),
        ));
    };
    let rate = f64::from(track.sample_rate);
    let start_sample = start_offset(*start_seconds, rate);
    if start_sample >= track.samples {
        return Err(VideoInputError::condition(
            *index,
            "start_time_seconds lies past the end of the audio track",
        ));
    }
    let seconds = (track.samples - start_sample) as f64 / rate;
    video_frame_count(
        seconds,
        limits.max_video_seconds,
        limits.frame_rate,
        &limits.frames,
    )
    .map(|frames| (seconds, frames))
    .map_err(|error| {
        VideoInputError::condition(
            *index,
            format!("the audio track sets the duration, but {error}"),
        )
    })
}

/// The generated canvas: from the ratio, or from the first keyframe.
fn target_canvas(
    task: VideoTask,
    target: &Target<'_>,
    media: &[MediaFacts],
    limits: &PlanLimits,
) -> Result<Canvas, VideoInputError> {
    if let Some(size) = named_canvas(task, target, &limits.canvases)? {
        return Ok(size);
    }
    // fl2va with `auto`: the conditions are keyframes and the first one sets
    // the canvas through the canvas rule, at its 768-pixel short edge.
    if target.short_edge != CANVAS_SHORT_EDGE {
        return Err(VideoInputError::invalid(
            RequestField::TargetShortEdge,
            format!(
                "must be {CANVAS_SHORT_EDGE} for an auto aspect ratio, got {}",
                target.short_edge
            ),
        ));
    }
    let Some(MediaFacts::Image(first)) = media.first() else {
        return Err(VideoInputError::condition(
            0,
            "the first keyframe is not an image",
        ));
    };
    canvas(f64::from(first.width), f64::from(first.height)).map_err(|error| {
        VideoInputError::condition(
            0,
            format!("the first keyframe sets the canvas, but its {error}"),
        )
    })
}

fn plan_keyframe(
    index: usize,
    condition: &ConditionSpec,
    image: &ImageFacts,
    size: Canvas,
    order: usize,
    task: VideoTask,
    vision: &ProcessorConfig,
) -> Result<ConditionPlan, VideoInputError> {
    let position = if condition.frame_index == Some(0) {
        FramePosition::First
    } else {
        FramePosition::Last
    };
    // The first keyframe anchors the canvas and is stretched onto it; a
    // second one follows and is cover-cropped.
    let crop = (order > 0).then(|| cover_crop(image.width, image.height, size));
    let seen = if task == VideoTask::Fl2va {
        let grid = vision
            .image_grid(size.height, size.width)
            .map_err(|error| VideoInputError::condition(index, error.to_string()))?;
        Some(Vision::Image(ImageVision {
            grid,
            tokens: vision.block_tokens(grid),
        }))
    } else {
        None
    };
    Ok(ConditionPlan {
        index,
        prepared: Prepared::Keyframe(KeyframeFit {
            position,
            cover_crop: crop,
        }),
        vision: seen,
        audio_rows: 0,
    })
}

fn plan_reference(
    index: usize,
    condition: &ConditionSpec,
    media: &MediaFacts,
    frames: u32,
    vision: &ProcessorConfig,
    limits: &PlanLimits,
) -> Result<ConditionPlan, VideoInputError> {
    let reject = |error: RuleError| VideoInputError::condition(index, error.to_string());
    let plan = |prepared, seen, audio_rows| ConditionPlan {
        index,
        prepared,
        vision: seen,
        audio_rows,
    };
    match media {
        MediaFacts::Image(image) => {
            let size = reference_image_size(image.width, image.height, &limits.references)
                .map_err(reject)?;
            let grid = vision.image_grid(size.height, size.width).map_err(reject)?;
            let seen = Vision::Image(ImageVision {
                grid,
                tokens: vision.block_tokens(grid),
            });
            Ok(plan(Prepared::Image(size), Some(seen), 0))
        }
        MediaFacts::Audio(audio) => {
            let clip = audio_clip(audio, 0.0, frames, limits.frame_rate).map_err(reject)?;
            Ok(plan(Prepared::Audio(clip), None, clip.rows()))
        }
        MediaFacts::Video(video) => {
            let start = condition.start_seconds.unwrap_or(0.0);
            let (clip, seen) = video_reference(
                video,
                start,
                frames,
                limits.frame_rate,
                &limits.frames,
                vision,
                &limits.references,
            )
            .map_err(reject)?;
            let audio_rows = clip.soundtrack.map_or(0, |track| track.rows());
            Ok(plan(
                Prepared::Video(clip),
                Some(Vision::Video(seen)),
                audio_rows,
            ))
        }
    }
}

/// Resolves every size a request implies from the probed media.
///
/// `media` holds the probed facts of each condition, in request order. The
/// caller applies the request-only rules of [`check_request`] first, which
/// need no media and so run before it is fetched.
///
/// # Errors
///
/// Returns [`VideoInputError::Invalid`] naming the first field at fault, and
/// [`VideoInputError::Internal`] when `media` and `conditions` differ in
/// length.
pub fn plan_request(
    task: VideoTask,
    target: &Target<'_>,
    conditions: &[ConditionSpec],
    media: &[MediaFacts],
    vision: &ProcessorConfig,
    limits: &PlanLimits,
) -> Result<RequestPlan, VideoInputError> {
    if media.len() != conditions.len() {
        return Err(VideoInputError::internal(format!(
            "{} conditions were planned with {} probed media",
            conditions.len(),
            media.len()
        )));
    }

    let mut soundtracks = Vec::new();
    for (index, (condition, facts)) in conditions.iter().zip(media).enumerate() {
        check_media(index, condition, facts)?;
        if let Some(track) = soundtrack(condition, facts) {
            soundtracks.push((index, track, condition.start_seconds.unwrap_or(0.0)));
        }
    }
    let (duration_seconds, frames) = resolve_frames(target, &soundtracks, limits)?;

    let size = target_canvas(task, target, media, limits)?;
    check_served(size, limits)?;

    let mut plans = Vec::with_capacity(conditions.len());
    let mut keyframes = 0;
    for (index, (condition, facts)) in conditions.iter().zip(media).enumerate() {
        let plan = match (condition.role, facts) {
            (Role::Keyframe, MediaFacts::Image(image)) => {
                keyframes += 1;
                plan_keyframe(index, condition, image, size, keyframes - 1, task, vision)?
            }
            (Role::Keyframe, _) => {
                return Err(VideoInputError::condition(
                    index,
                    "a keyframe must be an image",
                ));
            }
            (Role::Reference, _) => {
                plan_reference(index, condition, facts, frames, vision, limits)?
            }
        };
        plans.push(plan);
    }
    Ok(RequestPlan {
        task,
        canvas: size,
        duration_seconds,
        num_frames: frames,
        conditions: plans,
    })
}

#[cfg(test)]
pub(super) mod tests {
    use std::collections::HashSet;

    use serde::Deserialize;
    use serde_json::{Value, json};

    use super::super::presentation::present;
    use super::super::presentation::tests::CharacterTokenizer;
    use super::{
        AudioClip, AudioFacts, Canvas, ConditionPlan, ConditionSpec, ConditionType, FrameClips,
        FramePosition, FrameRate, ImageFacts, KeyframeFit, MAX_ASPECT_RATIO, MIN_ASPECT_RATIO,
        MediaFacts, PlanLimits, Prepared, ProcessorConfig, ReferenceConfig, RequestField,
        RequestPlan, Role, Target, VideoFacts, VideoInputError, VideoTask, Vision, audio_clip,
        canvas, check_request, cover_crop, gcd, plan_request, reference_bounds,
        reference_image_size, requested_frames, rows_per_frame, rule_canvases, video_frame_count,
        video_reference,
    };
    use crate::{MediaLocator, ReferenceLimits, ServedCanvas};

    /// The planning vectors, generated from the diffusers reference.
    pub(in crate::video) fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../tests/python/fixtures/minimax_h3_plan.json"
        ))
        .unwrap()
    }

    pub(in crate::video) fn vision(fixture: &Value) -> ProcessorConfig {
        ProcessorConfig::deserialize(&fixture["vision"]).unwrap()
    }

    /// The released checkpoint's output frame rate, which the vectors were
    /// generated at.
    pub(in crate::video) const FRAME_RATE: u32 = 24;

    /// The released checkpoint's frame rule: `17n + 5` frames, five latent
    /// frames per clip and two for the tail.
    pub(in crate::video) const FRAMES: FrameClips = FrameClips {
        min_frames: 22,
        clip_frames: 17,
        clip_latent_frames: 5,
        tail_latent_frames: 2,
    };

    /// The released checkpoint's documented references: nine images, three
    /// videos, three audio tracks and twelve in all, images encoded at a
    /// short edge of at most 2048 pixels.
    pub(in crate::video) const REFERENCE_LIMITS: ReferenceLimits = ReferenceLimits {
        images: 9,
        videos: 3,
        audios: 3,
        total: 12,
        image_short_edge: 2048,
    };

    /// The canvases of the vectors' accepted requests, as a deployment that
    /// prepares them all declares them. A `t2va` or `ref2va` request's ratio
    /// names its canvas (`auto` naming 16:9); an `fl2va` canvas, which the
    /// canvas rule resolves from a free ratio or the first keyframe, is
    /// declared under its own size.
    pub(in crate::video) fn vector_canvases() -> Vec<ServedCanvas> {
        let mut declared: Vec<ServedCanvas> = Vec::new();
        for case in fixture()["requests"].as_array().unwrap() {
            let Some(expected) = case.get("expected") else {
                continue;
            };
            let canvas = size(&expected["canvas"]);
            let target = &case["target"];
            let short_edge = target["short_edge"].as_u64().unwrap() as u32;
            let named = target["aspect_ratio"].as_str().unwrap();
            let aspect_ratio = match (case["task"].as_str().unwrap(), named) {
                ("fl2va", _) => format!("{}:{}", canvas.width, canvas.height),
                (_, "auto") => "16:9".to_owned(),
                (_, named) => named.to_owned(),
            };
            if !declared.iter().any(|served| {
                served.short_edge == short_edge && served.aspect_ratio == aspect_ratio
            }) {
                declared.push(ServedCanvas {
                    short_edge,
                    aspect_ratio,
                    canvas,
                });
            }
        }
        declared
    }

    /// The canvas a deployment declares for a short edge and aspect ratio.
    fn served(short_edge: u32, aspect_ratio: &str, width: u32, height: u32) -> ServedCanvas {
        ServedCanvas {
            short_edge,
            aspect_ratio: aspect_ratio.to_owned(),
            canvas: Canvas { width, height },
        }
    }

    /// Limits that serve every task and duration the model does, at the
    /// canvases of the vectors.
    pub(in crate::video) fn limits() -> PlanLimits {
        PlanLimits {
            tasks: vec![VideoTask::T2va, VideoTask::Fl2va, VideoTask::Ref2va],
            max_video_seconds: 15.0,
            frame_rate: FRAME_RATE,
            frames: FRAMES,
            canvases: vector_canvases(),
            reference_limits: REFERENCE_LIMITS,
            // The fixture's cases follow the released pipeline.
            references: ReferenceConfig::released(&REFERENCE_LIMITS),
        }
    }

    fn pair(value: &Value) -> (u32, u32) {
        (
            value[0].as_u64().unwrap() as u32,
            value[1].as_u64().unwrap() as u32,
        )
    }

    fn size(value: &Value) -> Canvas {
        let (width, height) = pair(value);
        Canvas { width, height }
    }

    fn audio_facts(value: &Value) -> AudioFacts {
        AudioFacts {
            sample_rate: value["sample_rate"].as_u64().unwrap() as u32,
            samples: value["samples"].as_u64().unwrap(),
        }
    }

    fn video_facts(value: &Value) -> VideoFacts {
        let (numerator, denominator) = pair(&value["frame_rate"]);
        // The fixture states a video's displayed size as its aspect.
        let (width, height) = pair(&value["display"]);
        VideoFacts {
            display_width: value["display"][0].as_u64().unwrap(),
            display_height: value["display"][1].as_u64().unwrap(),
            width,
            height,
            frame_rate: FrameRate {
                numerator,
                denominator,
            },
            frames: value["frames"].as_u64().unwrap(),
            soundtrack: value
                .get("soundtrack")
                .filter(|track| !track.is_null())
                .map(audio_facts),
        }
    }

    fn media(value: &Value) -> MediaFacts {
        if value.get("width").is_some() {
            MediaFacts::Image(ImageFacts {
                width: value["width"].as_u64().unwrap() as u32,
                height: value["height"].as_u64().unwrap() as u32,
            })
        } else if value.get("display").is_some() {
            MediaFacts::Video(video_facts(value))
        } else {
            MediaFacts::Audio(audio_facts(value))
        }
    }

    /// The task, target, conditions and media of a vectors request.
    pub(in crate::video) struct Request {
        task: VideoTask,
        aspect_ratio: String,
        target: (u32, Option<f64>),
        conditions: Vec<ConditionSpec>,
        media: Vec<MediaFacts>,
    }

    impl Request {
        pub(in crate::video) fn parse(case: &Value) -> Self {
            let target = &case["target"];
            let conditions = case["conditions"].as_array().unwrap();
            Self {
                task: VideoTask::deserialize(&case["task"]).unwrap(),
                aspect_ratio: target["aspect_ratio"].as_str().unwrap().to_owned(),
                target: (
                    target["short_edge"].as_u64().unwrap() as u32,
                    target.get("duration_seconds").and_then(Value::as_f64),
                ),
                conditions: conditions
                    .iter()
                    .map(|entry| ConditionSpec {
                        condition_type: ConditionType::deserialize(&entry["type"]).unwrap(),
                        role: Role::deserialize(&entry["role"]).unwrap(),
                        frame_index: entry.get("frame_index").and_then(Value::as_i64),
                        start_seconds: entry.get("start_time_seconds").and_then(Value::as_f64),
                    })
                    .collect(),
                media: conditions
                    .iter()
                    .map(|entry| media(&entry["media"]))
                    .collect(),
            }
        }

        fn target(&self) -> Target<'_> {
            Target {
                short_edge: self.target.0,
                aspect_ratio: &self.aspect_ratio,
                duration_seconds: self.target.1,
            }
        }

        /// The request's plan, its request-only rules applied first as the
        /// server applies them before fetching media.
        pub(in crate::video) fn plan(
            &self,
            vision: &ProcessorConfig,
        ) -> Result<RequestPlan, VideoInputError> {
            check_request(self.task, &self.target(), &self.conditions, &limits())?;
            plan_request(
                self.task,
                &self.target(),
                &self.conditions,
                &self.media,
                vision,
                &limits(),
            )
        }
    }

    fn audio_json(clip: &AudioClip) -> Value {
        json!({
            "start_sample": clip.start_sample,
            "source_samples": clip.source_samples,
            "samples": clip.samples,
            "latents": clip.latents,
        })
    }

    /// A condition plan in the vectors' layout; `canvas` is the generated
    /// canvas. A condition's video rows are its encoder units' rows.
    fn condition_json(plan: &ConditionPlan, canvas: Canvas) -> Value {
        let vision = match &plan.vision {
            None => Value::Null,
            Some(Vision::Image(seen)) => json!({
                "grid": [seen.grid.t, seen.grid.h, seen.grid.w],
                "tokens": seen.tokens,
            }),
            Some(Vision::Video(seen)) => json!({
                "grid": [seen.grid.t, seen.grid.h, seen.grid.w],
                "block_tokens": seen.block_tokens,
                "frame_indices": seen.frame_indices,
                "block_timestamps": seen.block_timestamps,
            }),
        };
        let mut entry = json!({
            "index": plan.index,
            "video_rows": plan.latent_units(canvas, 1).iter().sum::<u32>(),
            "audio_rows": plan.audio_rows,
            "vision": vision,
        });
        let fields = match plan.prepared {
            Prepared::Keyframe(fit) => json!({
                "kind": "keyframe",
                "position": fit.position,
                "cover_crop": fit.cover_crop,
            }),
            Prepared::Image(size) => json!({
                "kind": "image",
                "resize": [size.width, size.height],
            }),
            Prepared::Audio(clip) => json!({"kind": "audio", "clip": audio_json(&clip)}),
            Prepared::Video(clip) => json!({
                "kind": "video",
                "canvas": [clip.canvas.width, clip.canvas.height],
                "start_frame": clip.start_frame,
                "clip_frames": clip.frames,
                "vae_frames": clip.vae_frames,
                "latent_frames": clip.vae_clips.latent_frames(clip.vae_frames),
                "soundtrack": clip.soundtrack.as_ref().map(audio_json),
            }),
        };
        let entry_fields = entry.as_object_mut().unwrap();
        for (key, value) in fields.as_object().unwrap() {
            entry_fields.insert(key.clone(), value.clone());
        }
        entry
    }

    /// The canvas rule matches the reference's on named, probed and
    /// out-of-range aspects.
    #[test]
    fn canvases_match_the_reference() {
        let fixture = fixture();
        for case in fixture["canvas_cases"].as_array().unwrap() {
            let (width, height) = pair(&case["aspect"]);
            let resolved = canvas(f64::from(width), f64::from(height));
            if case.get("error").is_some() {
                assert!(resolved.is_err(), "{case}");
            } else {
                assert_eq!(resolved.unwrap(), size(&case["canvas"]), "{case}");
            }
        }
    }

    /// The rule's canvases are exactly those a dense sweep of 1:4 to 4:1
    /// resolves to, each listed once.
    #[test]
    fn rule_canvases_are_every_canvas_of_the_range() {
        let listed = rule_canvases();
        let unique: HashSet<Canvas> = listed.iter().copied().collect();
        assert_eq!(unique.len(), listed.len());

        let steps = 1_000_000;
        let swept: HashSet<Canvas> = (0..=steps)
            .map(|step| {
                let ratio = MIN_ASPECT_RATIO
                    + (MAX_ASPECT_RATIO - MIN_ASPECT_RATIO) * f64::from(step) / f64::from(steps);
                canvas(ratio, 1.0).unwrap()
            })
            .collect();
        assert_eq!(swept, unique);
    }

    /// A duration's frames round half to even, as Python's `round` does:
    /// 1.5 frames are 2 and 4.5 frames are 4.
    #[test]
    fn requested_frames_round_half_to_even() {
        assert_eq!(requested_frames(5.0, 24), 120);
        assert_eq!(requested_frames(0.0625, 24), 2);
        assert_eq!(requested_frames(0.1875, 24), 4);
    }

    /// Durations align up to `17n + 5` frames within the served range, with
    /// the video latent count the reference derives from them.
    #[test]
    fn durations_match_the_reference() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["duration_cases"].as_array().unwrap() {
            let seconds = case["seconds"].as_f64().unwrap();
            let target = Target {
                short_edge: 768,
                aspect_ratio: "auto",
                duration_seconds: Some(seconds),
            };
            let plan = plan_request(VideoTask::T2va, &target, &[], &[], &vision, &limits());
            if case.get("error").is_some() {
                let field = plan.unwrap_err().field();
                assert_eq!(field, Some(RequestField::TargetDuration), "{case}");
                assert!(
                    video_frame_count(seconds, 15.0, FRAME_RATE, &FRAMES).is_err(),
                    "{case}"
                );
                continue;
            }
            let plan = plan.unwrap();
            assert_eq!(u64::from(plan.num_frames), case["num_frames"], "{case}");
            assert_eq!(
                u64::from(video_frame_count(seconds, 15.0, FRAME_RATE, &FRAMES).unwrap()),
                case["num_frames"],
                "{case}"
            );
            assert_eq!(
                u64::from(FRAMES.latent_frames(plan.num_frames)),
                case["latent_frames"],
                "{case}"
            );
        }
    }

    /// Under the released sizing, image references resize to a 2048-pixel
    /// short edge and enter the conditioner at the processor's grid.
    #[test]
    fn reference_images_match_the_reference() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["reference_image_cases"].as_array().unwrap() {
            let (width, height) = pair(&case["size"]);
            let resized =
                reference_image_size(width, height, &ReferenceConfig::released(&REFERENCE_LIMITS));
            if case.get("error").is_some() {
                assert!(resized.is_err(), "{case}");
                continue;
            }
            let resized = resized.unwrap();
            assert_eq!(resized, size(&case["resize"]), "{case}");
            let grid = vision.image_grid(resized.height, resized.width).unwrap();
            assert_eq!(
                json!([grid.t, grid.h, grid.w]),
                case["vision_grid"],
                "{case}"
            );
            assert_eq!(
                u64::from(vision.block_tokens(grid)),
                case["vision_tokens"],
                "{case}"
            );
            assert_eq!(u64::from(rows_per_frame(resized)), case["rows"], "{case}");
        }
    }

    /// A following keyframe's cover crop reproduces the reference's resize
    /// and crop box.
    #[test]
    fn keyframe_crops_match_the_reference() {
        let fixture = fixture();
        for case in fixture["keyframe_cases"].as_array().unwrap() {
            let (width, height) = pair(&case["size"]);
            let crop = cover_crop(width, height, size(&case["canvas"]));
            assert_eq!(json!(crop), case["cover_crop"], "{case}");
        }
    }

    /// A video's encoded frames split into the video encoder's 17-frame
    /// clips: each complete clip yields five latent frames of rows and
    /// the padded last one two, which together are the condition's rows. A
    /// still image is one unit.
    #[test]
    fn latent_units_partition_the_condition_rows() {
        let canvas = Canvas {
            width: 1344,
            height: 768,
        };
        let video = ConditionPlan {
            index: 0,
            prepared: Prepared::Video(super::VideoClip {
                canvas,
                start_frame: 0,
                frames: 124,
                vae_frames: 124,
                vae_clips: FRAMES,
                soundtrack: None,
            }),
            vision: None,
            audio_rows: 0,
        };
        // A video's units are its encoder windows, however many bands an
        // image would take.
        let units = video.latent_units(canvas, 4);
        assert_eq!(units, [vec![5 * 1008; 7], vec![2 * 1008]].concat());
        assert_eq!(units.iter().sum::<u32>(), FRAMES.latent_frames(124) * 1008);

        // A reference image is one latent frame at its own size, a keyframe
        // one at the generated canvas.
        let size = Canvas {
            width: 3648,
            height: 2048,
        };
        let image = ConditionPlan {
            prepared: Prepared::Image(size),
            ..video.clone()
        };
        assert_eq!(image.latent_units(canvas, 1), [7296]);
        let keyframe = ConditionPlan {
            prepared: Prepared::Keyframe(KeyframeFit {
                position: FramePosition::First,
                cover_crop: None,
            }),
            ..video
        };
        assert_eq!(keyframe.latent_units(canvas, 1), [1008]);
    }

    /// A reference image's bands are runs of whole 32-pixel patch rows dealt
    /// as evenly as possible, together its rows; a keyframe stays one unit.
    #[test]
    fn reference_image_bands_partition_its_patch_rows() {
        // A 16:9 reference at its 2048-pixel short edge: 64 patch rows of
        // 114 patches.
        let size = Canvas {
            width: 3648,
            height: 2048,
        };
        let image = ConditionPlan {
            index: 0,
            prepared: Prepared::Image(size),
            vision: None,
            audio_rows: 0,
        };
        assert_eq!(image.latent_units(size, 4), [16 * 114; 4]);
        assert_eq!(image.latent_units(size, 3), [21 * 114, 21 * 114, 22 * 114]);
        assert_eq!(image.latent_units(size, 0), [64 * 114]);
        assert_eq!(image.latent_units(size, 100), [114; 64]);

        let keyframe = ConditionPlan {
            prepared: Prepared::Keyframe(KeyframeFit {
                position: FramePosition::First,
                cover_crop: None,
            }),
            ..image.clone()
        };
        assert_eq!(keyframe.latent_units(size, 4), [rows_per_frame(size)]);
    }

    /// The lane is divided evenly among a request's reference images, at
    /// least one unit each; other conditions take no share.
    #[test]
    fn reference_images_share_the_latent_lane() {
        let image = ConditionPlan {
            index: 0,
            prepared: Prepared::Image(Canvas {
                width: 2048,
                height: 2048,
            }),
            vision: None,
            audio_rows: 0,
        };
        let audio = ConditionPlan {
            prepared: Prepared::Audio(AudioClip {
                sample_rate: 32_000,
                start_sample: 0,
                source_samples: 32_000,
                samples: 32_000,
                latents: 40,
            }),
            audio_rows: 80,
            ..image.clone()
        };
        let plan = |conditions: Vec<ConditionPlan>| RequestPlan {
            task: VideoTask::Ref2va,
            canvas: Canvas {
                width: 1344,
                height: 768,
            },
            duration_seconds: 5.0,
            num_frames: 124,
            conditions,
        };
        assert_eq!(
            plan(vec![image.clone(), audio.clone()]).image_bands(4),
            [4, 4]
        );
        assert_eq!(plan(vec![image.clone(); 3]).image_bands(4), [1; 3]);
        assert_eq!(plan(vec![image.clone(); 2]).image_bands(4), [2; 2]);
        assert_eq!(plan(vec![image.clone()]).image_bands(0), [1]);

        // Images beyond one round encode whole, a round's worth at a time,
        // and the images of a final partial round split into bands that
        // fill it: on a four-unit lane nine images take 1 x 8 + 4 units and
        // six take 1 x 4 + 2 x 2, while seven, whose final three cannot
        // share four units evenly, stay whole. An audio track takes no part.
        let nine = plan(vec![image.clone(); 9]).image_bands(4);
        assert_eq!(nine, [vec![1; 8], vec![4]].concat());
        let six = plan(vec![image.clone(); 6]).image_bands(4);
        assert_eq!(six, [vec![1; 4], vec![2; 2]].concat());
        assert_eq!(plan(vec![image.clone(); 7]).image_bands(4), [1; 7]);
        let mut five = vec![image.clone(); 5];
        five.insert(2, audio);
        assert_eq!(plan(five).image_bands(4), [1, 1, 1, 1, 1, 4]);
    }

    /// Every planned condition of the vectors is described with the rows its
    /// plan counts: a described condition carries the plan's video rows, in
    /// its encoder units however its image is banded, and its audio rows.
    #[test]
    fn described_conditions_carry_their_plans_rows() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let source = MediaLocator {
            name: "uniserve-media-0".to_owned(),
            bytes: 1,
        };
        for case in fixture["requests"].as_array().unwrap() {
            if case.get("expected").is_none() {
                continue;
            }
            let plan = Request::parse(case).plan(&vision).unwrap();
            for condition in &plan.conditions {
                // Bands split an image's rows without changing them.
                let described = condition.describe(plan.canvas, source.clone(), 4);
                assert_eq!(
                    described.latent_units,
                    condition.latent_units(plan.canvas, 4),
                    "{}",
                    case["name"]
                );
                assert_eq!(
                    described.audio_rows, condition.audio_rows,
                    "{}",
                    case["name"]
                );
            }
        }
    }

    /// Reference videos resample to the 24 fps output rate, honour the start
    /// offset, keep a whole VAE clip, and sample the reference's 2 fps
    /// vision blocks.
    #[test]
    fn reference_videos_match_the_reference() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["video_reference_cases"].as_array().unwrap() {
            let facts = video_facts(case);
            let start = case["start_seconds"].as_f64().unwrap();
            let frames = case["num_frames"].as_u64().unwrap() as u32;
            let planned = video_reference(
                &facts,
                start,
                frames,
                FRAME_RATE,
                &FRAMES,
                &vision,
                &ReferenceConfig::released(&REFERENCE_LIMITS),
            );
            if case.get("error").is_some() {
                assert!(planned.is_err(), "{case}");
                continue;
            }
            let (clip, seen) = planned.unwrap();
            assert_eq!(clip.canvas, size(&case["canvas"]), "{case}");
            assert_eq!(u64::from(clip.start_frame), case["start_frame"], "{case}");
            assert_eq!(u64::from(clip.frames), case["clip_frames"], "{case}");
            assert_eq!(u64::from(clip.vae_frames), case["vae_frames"], "{case}");
            let latents = FRAMES.latent_frames(clip.vae_frames);
            assert_eq!(u64::from(latents), case["latent_frames"], "{case}");
            assert_eq!(json!(seen.frame_indices), case["frame_indices"], "{case}");
            assert_eq!(
                json!(seen.block_timestamps),
                case["block_timestamps"],
                "{case}"
            );
            assert_eq!(
                json!([seen.grid.t, seen.grid.h, seen.grid.w]),
                case["vision_grid"]
            );
            assert_eq!(u64::from(seen.block_tokens), case["block_tokens"], "{case}");
            let rows = latents * rows_per_frame(clip.canvas);
            assert_eq!(u64::from(rows), case["rows"], "{case}");
        }
    }

    /// Every reference the planner prepares for a five-second request fits
    /// the bounds of its 124 frames, enlarged or not: an image of any
    /// accepted aspect and size, and a video of any aspect, size and frame
    /// rate with a soundtrack of any rate.
    #[test]
    fn prepared_references_fit_their_bounds() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let frames = video_frame_count(5.0, 15.0, FRAME_RATE, &FRAMES).unwrap();
        let edge = REFERENCE_LIMITS.image_short_edge;
        let bounds = reference_bounds(frames, FRAME_RATE, &FRAMES, edge);
        assert_eq!(
            bounds.images,
            [
                Canvas {
                    width: 4 * edge,
                    height: edge
                },
                Canvas {
                    width: edge,
                    height: 4 * edge
                },
            ]
        );
        assert_eq!((bounds.video_frames, bounds.audio_samples), (124, 165_334));
        assert_eq!(bounds.video_canvases, rule_canvases());
        let within = |size: Canvas, bounds: &[Canvas]| {
            bounds
                .iter()
                .any(|bound| size.width <= bound.width && size.height <= bound.height)
        };

        for config in [
            ReferenceConfig::new(&REFERENCE_LIMITS),
            ReferenceConfig::released(&REFERENCE_LIMITS),
        ] {
            for scale in [1, 3] {
                for width in (250..=4000).step_by(7) {
                    let size = reference_image_size(scale * width, scale * 1000, &config).unwrap();
                    assert!(within(size, &bounds.images), "{width} {config:?}");
                }
            }
            for ((width, height), (numerator, denominator), sample_rate) in [
                ((1920, 1080), (24, 1), 32_000),
                ((480, 854), (30_000, 1001), 44_100),
                ((2048, 512), (60, 1), 48_000),
                ((300, 1200), (25, 1), 22_050),
                ((3840, 1646), (24_000, 1001), 16_000),
            ] {
                let divisor = gcd(u64::from(width), u64::from(height));
                let facts = VideoFacts {
                    display_width: u64::from(width) / divisor,
                    display_height: u64::from(height) / divisor,
                    width,
                    height,
                    frame_rate: FrameRate {
                        numerator,
                        denominator,
                    },
                    frames: 10_000,
                    soundtrack: Some(AudioFacts {
                        sample_rate,
                        samples: 10_000_000,
                    }),
                };
                let (clip, _) =
                    video_reference(&facts, 0.0, frames, FRAME_RATE, &FRAMES, &vision, &config)
                        .unwrap();
                assert!(clip.vae_frames <= bounds.video_frames, "{facts:?}");
                assert!(within(clip.canvas, &bounds.video_canvases), "{facts:?}");
                let track = clip.soundtrack.unwrap();
                assert!(track.samples <= bounds.audio_samples, "{facts:?}");
            }
        }
    }

    /// A reference is only ever reduced unless the deployment enlarges: an
    /// image above the short edge and a video above the canvas of its aspect
    /// scale down as the released pipeline scales them, and smaller ones
    /// keep their own size on the 32-pixel grid.
    #[test]
    fn references_are_enlarged_only_on_request() {
        let sized = |width, height| Canvas { width, height };
        let unenlarged = ReferenceConfig::new(&REFERENCE_LIMITS);
        let image = |width, height, config: &ReferenceConfig| {
            reference_image_size(width, height, config).unwrap()
        };
        // 1080 rounds to 34 patches; 768 is already on the grid.
        assert_eq!(image(1920, 1080, &unenlarged), sized(1920, 1088));
        assert_eq!(image(768, 768, &unenlarged), sized(768, 768));
        assert_eq!(
            image(768, 768, &ReferenceConfig::released(&REFERENCE_LIMITS)),
            sized(2048, 2048)
        );
        // A 2304-pixel short edge exceeds 2048 and is reduced either way.
        for config in [&unenlarged, &ReferenceConfig::released(&REFERENCE_LIMITS)] {
            assert_eq!(image(4096, 2304, config), sized(3648, 2048));
        }
        // A smaller short edge reduces further and still never enlarges.
        let half = ReferenceConfig {
            image_short_edge: 1024,
            ..unenlarged
        };
        assert_eq!(image(2560, 1440, &half), sized(1824, 1024));
        assert_eq!(image(768, 768, &half), sized(768, 768));
        // The 1:4 to 4:1 bound holds at any size.
        assert!(reference_image_size(4100, 1000, &unenlarged).is_err());

        let fixture = fixture();
        let vision = vision(&fixture);
        let video = |width: u32, height: u32, config: &ReferenceConfig| {
            let divisor = gcd(u64::from(width), u64::from(height));
            let facts = VideoFacts {
                display_width: u64::from(width) / divisor,
                display_height: u64::from(height) / divisor,
                width,
                height,
                frame_rate: FrameRate {
                    numerator: 24,
                    denominator: 1,
                },
                frames: 48,
                soundtrack: None,
            };
            video_reference(&facts, 0.0, 124, FRAME_RATE, &FRAMES, &vision, config)
                .unwrap()
                .0
                .canvas
        };
        // 720p holds fewer pixels than the 16:9 canvas: 720 is 22.5 patches
        // and rounds half to even. 1080p is reduced to the canvas.
        assert_eq!(video(1280, 720, &unenlarged), sized(1280, 704));
        assert_eq!(video(1920, 1080, &unenlarged), sized(1344, 768));
        assert_eq!(
            video(1280, 720, &ReferenceConfig::released(&REFERENCE_LIMITS)),
            sized(1344, 768)
        );
    }

    /// A reference config reads the processor options a deployment states,
    /// absent ones keeping their defaults (the image edge the denoiser
    /// declares), and refuses an option the processor does not take; its
    /// short edge lies on the 32-pixel grid up to the declared edge.
    #[test]
    fn a_reference_config_reads_processor_options() {
        let read = |options: Value| {
            ReferenceConfig::from_kwargs(options.as_object().unwrap(), &REFERENCE_LIMITS)
        };
        assert_eq!(read(json!({})), Ok(ReferenceConfig::new(&REFERENCE_LIMITS)));
        assert_eq!(
            read(json!({"enlarge_references": true})),
            Ok(ReferenceConfig::released(&REFERENCE_LIMITS))
        );
        assert_eq!(
            read(json!({"reference_image_short_edge": 1024})),
            Ok(ReferenceConfig {
                image_short_edge: 1024,
                enlarge: false,
            })
        );
        assert!(read(json!({"num_crops": 4})).is_err());
        assert!(read(json!({"enlarge_references": "yes"})).is_err());
        assert!(read(json!({"reference_image_short_edge": 1000})).is_err());

        for (edge, valid) in [
            (2048, true),
            (32, true),
            (0, false),
            (1000, false),
            (4096, false),
        ] {
            let config = ReferenceConfig {
                image_short_edge: edge,
                enlarge: false,
            };
            assert_eq!(config.validate(&REFERENCE_LIMITS).is_ok(), valid, "{edge}");
        }

        // A denoiser that declares a smaller edge bounds the deployment's.
        let smaller = ReferenceLimits {
            image_short_edge: 1024,
            ..REFERENCE_LIMITS
        };
        let read =
            |options: Value| ReferenceConfig::from_kwargs(options.as_object().unwrap(), &smaller);
        assert_eq!(
            read(json!({})).map(|config| config.image_short_edge),
            Ok(1024)
        );
        assert!(read(json!({"reference_image_short_edge": 2048})).is_err());
        // A denoiser that takes no image reference sizes none.
        let none = ReferenceLimits::default();
        assert!(ReferenceConfig::from_kwargs(&serde_json::Map::new(), &none).is_ok());
    }

    /// Soundtracks truncate at their native rate, resample to 32 kHz and pad
    /// to whole 800-sample latents.
    #[test]
    fn audio_clips_match_the_reference() {
        let fixture = fixture();
        for case in fixture["audio_cases"].as_array().unwrap() {
            let clip = audio_clip(
                &audio_facts(case),
                case["start_seconds"].as_f64().unwrap(),
                case["num_frames"].as_u64().unwrap() as u32,
                FRAME_RATE,
            )
            .unwrap();
            assert_eq!(audio_json(&clip), case["clip"], "{case}");
        }
    }

    /// Whole requests plan to the reference's canvas, frame counts, condition
    /// preparation, vision grids and rows.
    #[test]
    fn request_plans_match_the_reference() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["requests"].as_array().unwrap() {
            let Some(expected) = case.get("expected") else {
                continue;
            };
            let name = &case["name"];
            let plan = Request::parse(case).plan(&vision).unwrap();
            assert_eq!(plan.canvas, size(&expected["canvas"]), "{name}");
            assert_eq!(u64::from(plan.num_frames), expected["num_frames"], "{name}");
            // The conditions' encoder units and audio rows are the
            // reference's condition rows.
            let video_rows: u32 = plan
                .conditions
                .iter()
                .flat_map(|condition| condition.latent_units(plan.canvas, 1))
                .sum();
            let audio_rows: u32 = plan.conditions.iter().map(|plan| plan.audio_rows).sum();
            assert_eq!(
                u64::from(video_rows),
                expected["condition_video_rows"],
                "{name}"
            );
            assert_eq!(
                u64::from(audio_rows),
                expected["condition_audio_rows"],
                "{name}"
            );
            let conditions: Vec<Value> = plan
                .conditions
                .iter()
                .map(|condition| condition_json(condition, plan.canvas))
                .collect();
            assert_eq!(Value::from(conditions), expected["conditions"], "{name}");
        }
    }

    /// Each rejected request names the field the serving contract blames,
    /// whether planning or the presentation rejects it.
    #[test]
    fn rejected_requests_name_the_field() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["requests"].as_array().unwrap() {
            let Some(error) = case.get("error") else {
                continue;
            };
            let rejection = match Request::parse(case).plan(&vision) {
                Ok(plan) => present(&CharacterTokenizer, &plan, case["prompt"].as_str().unwrap())
                    .expect_err("the request was accepted"),
                Err(rejection) => rejection,
            };
            assert_eq!(
                rejection.field().unwrap().to_string(),
                error["field"],
                "{}",
                case["name"]
            );
        }
    }

    const FIVE_SECONDS: Target<'static> = Target {
        short_edge: 768,
        aspect_ratio: "auto",
        duration_seconds: Some(5.0),
    };

    /// A deployment serves only the canvases its workers prepare; others are
    /// rejected on the aspect ratio before any media is needed.
    #[test]
    fn prepared_canvases_restrict_the_target() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let limits = PlanLimits {
            canvases: vec![served(768, "16:9", 1344, 768)],
            ..limits()
        };
        let plan =
            plan_request(VideoTask::T2va, &FIVE_SECONDS, &[], &[], &vision, &limits).unwrap();
        assert_eq!((plan.canvas.width, plan.canvas.height), (1344, 768));

        let portrait = Target {
            aspect_ratio: "9:16",
            ..FIVE_SECONDS
        };
        let error = check_request(VideoTask::T2va, &portrait, &[], &limits).unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetAspectRatio));
    }

    /// A deployment that declares 480p canvases accepts their short edge: a
    /// ratio names the canvas the deployment declares for it rather than a
    /// rescaled canvas rule, and a short edge or ratio the deployment did not
    /// prepare is refused before any media is needed.
    #[test]
    fn declared_canvases_resolve_each_short_edge() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let canvas = |width, height| Canvas { width, height };
        let served = PlanLimits {
            canvases: vec![
                served(768, "16:9", 1344, 768),
                served(480, "21:9", 992, 416),
                served(480, "16:9", 832, 480),
            ],
            ..limits()
        };
        assert_eq!(served.short_edges(), vec![768, 480]);
        for (short_edge, aspect_ratio, expected) in [
            (768, "16:9", canvas(1344, 768)),
            (480, "21:9", canvas(992, 416)),
            // `auto` is 16:9 for t2va.
            (480, "auto", canvas(832, 480)),
        ] {
            let target = Target {
                short_edge,
                aspect_ratio,
                ..FIVE_SECONDS
            };
            let plan = plan_request(VideoTask::T2va, &target, &[], &[], &vision, &served).unwrap();
            assert_eq!(plan.canvas, expected, "{short_edge} {aspect_ratio}");
        }
        for (short_edge, aspect_ratio, field) in [
            (720, "16:9", RequestField::TargetShortEdge),
            (768, "21:9", RequestField::TargetAspectRatio),
            (480, "1:1", RequestField::TargetAspectRatio),
        ] {
            let target = Target {
                short_edge,
                aspect_ratio,
                ..FIVE_SECONDS
            };
            let error = check_request(VideoTask::T2va, &target, &[], &served).unwrap_err();
            assert_eq!(error.field(), Some(field), "{short_edge} {aspect_ratio}");
        }

        // A deployment that prepares only 768p canvases serves no other
        // short edge.
        let target = Target {
            short_edge: 480,
            aspect_ratio: "16:9",
            ..FIVE_SECONDS
        };
        let error = check_request(VideoTask::T2va, &target, &[], &limits()).unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetShortEdge));
    }

    /// `fl2va` takes any ratio from 1:4 to 4:1, which the canvas rule
    /// resolves at its 768-pixel short edge, but only onto a canvas the
    /// deployment declares; `t2va` takes only a declared ratio.
    #[test]
    fn free_keyframe_ratios_land_on_a_declared_canvas() {
        let limits = PlanLimits {
            canvases: vec![served(768, "16:9", 1344, 768)],
            ..limits()
        };
        let keyframe = ConditionSpec {
            condition_type: ConditionType::Image,
            role: Role::Keyframe,
            frame_index: Some(0),
            start_seconds: None,
        };
        let target = |aspect_ratio| Target {
            aspect_ratio,
            ..FIVE_SECONDS
        };
        // 7:4 at a 768-pixel short edge is 1344x768, the declared canvas.
        let fl2va = |aspect_ratio| {
            check_request(
                VideoTask::Fl2va,
                &target(aspect_ratio),
                std::slice::from_ref(&keyframe),
                &limits,
            )
        };
        assert!(fl2va("7:4").is_ok());
        assert!(fl2va("16:9").is_ok());
        // 4:3 is 1024x768, which the deployment does not serve.
        let error = fl2va("4:3").unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetAspectRatio));
        // t2va names only the declared ratios, and auto names 16:9.
        assert!(check_request(VideoTask::T2va, &target("auto"), &[], &limits).is_ok());
        let error = check_request(VideoTask::T2va, &target("7:4"), &[], &limits).unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetAspectRatio));
        assert!(error.to_string().contains("auto or one of 16:9"), "{error}");

        // A short edge that serves no canvas names none, for any ratio.
        let narrow = Target {
            short_edge: 480,
            ..target("16:9")
        };
        let error = super::named_canvas(VideoTask::T2va, &narrow, &limits.canvases).unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetAspectRatio));
        assert!(
            error
                .to_string()
                .contains("no canvas is served at a 480-pixel short edge"),
            "{error}"
        );
    }

    /// The served maximum duration narrows the model's own range.
    #[test]
    fn max_video_seconds_bounds_the_duration() {
        let ten_seconds = Target {
            duration_seconds: Some(10.0),
            ..FIVE_SECONDS
        };
        let limits = PlanLimits {
            max_video_seconds: 8.0,
            ..limits()
        };
        let error = check_request(VideoTask::T2va, &ten_seconds, &[], &limits).unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetDuration));
    }

    /// Each reference-count refusal names what it bounds: nine images, three
    /// videos and three audio tracks are each within their own limit, but
    /// thirteen references exceed the twelve of every kind together.
    #[test]
    fn reference_limits_name_what_they_bound() {
        let reference = |condition_type| ConditionSpec {
            condition_type,
            role: Role::Reference,
            frame_index: None,
            start_seconds: None,
        };
        let refusal = |conditions: &[ConditionSpec]| {
            let checked = check_request(VideoTask::Ref2va, &FIVE_SECONDS, conditions, &limits());
            match checked {
                Err(VideoInputError::Invalid { field, message }) => {
                    assert_eq!(field, RequestField::Conditions);
                    message
                }
                other => panic!("the references were not refused: {other:?}"),
            }
        };

        let images = vec![reference(ConditionType::Image); 10];
        assert_eq!(refusal(&images), "at most 9 image references, got 10");

        let mut every_kind = vec![reference(ConditionType::Image); 9];
        every_kind.extend(vec![reference(ConditionType::Video); 3]);
        every_kind.push(reference(ConditionType::Audio));
        assert_eq!(refusal(&every_kind), "at most 12 references, got 13");
    }

    /// A `ref2va` request carries at most the references its denoiser
    /// declares, of each kind and in all.
    #[test]
    fn reference_counts_are_bounded_by_the_declared_limits() {
        let limits = PlanLimits {
            reference_limits: ReferenceLimits {
                images: 2,
                videos: 1,
                audios: 1,
                total: 3,
                ..REFERENCE_LIMITS
            },
            ..limits()
        };
        let reference = |condition_type| ConditionSpec {
            condition_type,
            role: Role::Reference,
            frame_index: None,
            start_seconds: None,
        };
        let image = reference(ConditionType::Image);
        let video = reference(ConditionType::Video);
        let audio = reference(ConditionType::Audio);
        let check = |conditions: &[ConditionSpec]| {
            check_request(VideoTask::Ref2va, &FIVE_SECONDS, conditions, &limits)
        };

        assert!(check(&[image.clone(), image.clone(), video.clone()]).is_ok());
        for (conditions, bound) in [
            (vec![image.clone(); 3], "at most 2 image references"),
            (
                vec![image.clone(), video.clone(), video.clone()],
                "at most 1 video references",
            ),
            (
                vec![image.clone(), audio.clone(), audio.clone()],
                "at most 1 audio references",
            ),
            (
                vec![image.clone(), image.clone(), video.clone(), audio.clone()],
                "at most 3 references",
            ),
        ] {
            let error = check(&conditions).unwrap_err();
            assert_eq!(error.field(), Some(RequestField::Conditions));
            assert!(error.to_string().contains(bound), "{bound}: {error}");
        }
    }

    /// A task outside the placed denoiser's set is rejected first.
    #[test]
    fn only_served_tasks_are_accepted() {
        let limits = PlanLimits {
            tasks: vec![VideoTask::Ref2va],
            ..limits()
        };
        let error = check_request(VideoTask::T2va, &FIVE_SECONDS, &[], &limits).unwrap_err();
        assert_eq!(error.field(), Some(RequestField::Task));
    }

    /// Aspect ratios are accepted only in canonical `W:H` form.
    #[test]
    fn aspect_ratios_must_be_canonical() {
        for (ratio, accepted) in [
            ("16:9", true),
            ("7:4", true),
            ("016:9", false),
            ("+16:9", false),
            ("16:9:1", false),
            ("16/9", false),
            ("0:9", false),
            ("", false),
        ] {
            let target = Target {
                short_edge: 768,
                aspect_ratio: ratio,
                duration_seconds: Some(5.0),
            };
            let keyframe = ConditionSpec {
                condition_type: ConditionType::Image,
                role: Role::Keyframe,
                frame_index: Some(0),
                start_seconds: None,
            };
            let checked = check_request(VideoTask::Fl2va, &target, &[keyframe], &limits());
            assert_eq!(checked.is_ok(), accepted, "{ratio:?}");
        }
    }
}
