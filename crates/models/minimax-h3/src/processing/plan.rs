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
//! These are the only implementation of the rules: the deployment's processor
//! (`super::processor`) plans every request with them. The family-neutral
//! rules they are built from - the frame rule, the canvas rule, the cover crop
//! and the resampled length - are `crates/processing`'s, and the model's own
//! facts, such as its output frame rate and the values those rules are
//! parameterized by, are the rules' parameters ([`PlanLimits`]), as the
//! denoiser declares them. The rules are tested against [`PLAN_VECTORS`],
//! vectors generated from the reference pipeline.
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
//! - without a stated length, a `ref2va` request whose references
//!   carry exactly one soundtrack lasts as long as that soundtrack after its
//!   start offset.

use serde::{Deserialize, Serialize};

use processing::{
    AudioMetadata, CanvasError, CanvasRule, ConditionPlan, CoverCrop, FrameClips, ImageMetadata,
    InputError, Length, MediaMetadata, Reference, ReferenceKind, VideoMetadata, VideoSpec,
    cover_crop, requested_frames, resampled_len,
};
use proto::video::{Canvas, ConditionMedia, ConditionRole, ConditionVision, ImageFit, VisionGrid};

use super::{ConditionPacking, ReferenceLimits, ServedCanvas};
use crate::VideoTask;
use crate::config::MIN_SECONDS;
use crate::encoding::audio_latent_frames;
use crate::packing::{LatentRaster, PATCH_SIDE};
use qwen3::processor::{ProcessorConfig, VisionError, sample_frames};

/// The ratio `auto` means for `t2va` and `ref2va`.
const DEFAULT_ASPECT_RATIO: (u32, u32) = (16, 9);

/// The planning vectors the rules are tested against, generated from the
/// reference pipeline: a file of this crate's, which the daemon's request tests
/// read too.
pub const PLAN_VECTORS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/minimax_h3_plan.json"
);

/// A duration or media fact the rules cannot size.
///
/// The messages name what is wrong in request terms; [`plan_request`] and
/// [`check_request`] attach the request field at fault
/// ([`VideoInputError::Invalid`]).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum RuleError {
    /// A duration outside `[MIN_SECONDS, max_seconds]`, or not finite.
    #[error("video duration must be a finite number of seconds in [{MIN_SECONDS}, {max_seconds}]")]
    Duration {
        /// The longest duration served.
        max_seconds: f64,
    },
    /// An image or video the conditioner's processor cannot size or sample.
    #[error(transparent)]
    Vision(#[from] VisionError),
    /// An image reference outside the canvas rule's aspects.
    #[error(
        "a reference image must lie within {}:{} and {}:{}, got {width}x{height}",
        narrowest.0, narrowest.1, widest.0, widest.1
    )]
    ReferenceAspect {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// The narrowest aspect the canvas rule resolves.
        narrowest: (u32, u32),
        /// The widest aspect the canvas rule resolves.
        widest: (u32, u32),
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

/// The public request parameter a refusal of the output size names.
pub const SIZE: &str = "size";
/// The public request parameter a refusal of the prompt names.
pub const PROMPT: &str = "prompt";
/// The public request parameter a refusal of the references as a whole names; a refusal of one
/// reference names `input_references[i]` ([`reference`]).
pub const REFERENCES: &str = "input_references";

/// Refuses the request's reference `index` with `message`.
pub fn reference(index: usize, message: impl Into<String>) -> InputError {
    InputError::invalid(format!("{REFERENCES}[{index}]"), message)
}

/// The public request parameter that states the request's length: `num_frames` when it states a
/// frame count, `seconds` otherwise (a request that states none is refused for want of
/// `seconds`).
pub fn length_param(request: &VideoSpec) -> &'static str {
    match request.length {
        Some(Length::Frames(_)) => "num_frames",
        Some(Length::Seconds(_)) | None => "seconds",
    }
}

/// The task a request's references select: none is `t2va`, image keyframes alone (`frame_index`
/// 0 or -1) are `fl2va`, and any reference without a frame index makes it `ref2va`.
pub fn task(request: &VideoSpec) -> VideoTask {
    if request
        .references
        .iter()
        .any(|reference| !is_keyframe(reference))
    {
        VideoTask::ReferenceToVideoAudio
    } else if request.references.is_empty() {
        VideoTask::TextToVideoAudio
    } else {
        VideoTask::FirstLastToVideoAudio
    }
}

/// Whether a reference anchors a generated frame: a keyframe states the frame it anchors.
fn is_keyframe(reference: &Reference) -> bool {
    reference.frame_index.is_some()
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
/// outside `[MIN_SECONDS, max_video_seconds]`.
pub fn video_frame_count(
    seconds: f64,
    max_video_seconds: f64,
    frame_rate: u32,
    frames: &FrameClips,
) -> Result<u32, RuleError> {
    if !seconds.is_finite() || !(MIN_SECONDS..=max_video_seconds).contains(&seconds) {
        return Err(RuleError::Duration {
            max_seconds: max_video_seconds,
        });
    }
    Ok(frames.align_up(requested_frames(seconds, frame_rate)))
}

/// How a deployment sizes the references of a `ref2va` request: the options
/// of the MiniMax-H3 video processor (`mm_processor_kwargs`), resolved
/// against what the video denoiser declares ([`ReferenceLimits`]).
///
/// A reference is only ever reduced by default: an image whose short edge
/// exceeds `image_short_edge` scales down to it, and a video with more
/// pixels than the canvas of its aspect scales down to that canvas, while
/// smaller references keep their own size on the canvas grid. The released
/// pipeline also enlarges smaller ones to those sizes, which `enlarge`
/// selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceConfig {
    /// Short edge, in pixels, an image reference is encoded at, at most: a
    /// multiple of the canvas grid up to the edge the denoiser declares.
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
    /// refuses on the canvas grid of `rule`.
    pub fn from_kwargs(
        kwargs: &serde_json::Map<String, serde_json::Value>,
        limits: &ReferenceLimits,
        rule: &CanvasRule,
    ) -> Result<Self, String> {
        let options = ReferenceOptions::deserialize(serde_json::Value::Object(kwargs.clone()))
            .map_err(|error| error.to_string())?;
        let config = Self {
            image_short_edge: options
                .reference_image_short_edge
                .unwrap_or(limits.image_short_edge),
            enlarge: options.enlarge_references,
        };
        config.validate(limits, rule)?;
        Ok(config)
    }

    /// Checks that the short edge is a multiple of the canvas grid of `rule`
    /// from one multiple to the edge the denoiser declares; a denoiser that
    /// takes no image reference sizes none.
    ///
    /// # Errors
    ///
    /// Returns a message naming the short edge.
    pub fn validate(&self, limits: &ReferenceLimits, rule: &CanvasRule) -> Result<(), String> {
        if limits.images == 0 {
            return Ok(());
        }
        let edge = self.image_short_edge;
        let largest = limits.image_short_edge;
        let multiple = rule.multiple;
        if !edge.is_multiple_of(multiple) || !(multiple..=largest).contains(&edge) {
            return Err(format!(
                "a reference image short edge must be a multiple of {multiple} from {multiple} to \
                 {largest} pixels, got {edge}"
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
    /// Sample rate, in Hz, every soundtrack is resampled to, as the denoiser
    /// declares it.
    pub audio_sample_rate: u32,
    /// Samples of one audio latent frame, as the denoiser declares it.
    pub audio_hop: u32,
    /// How a display aspect resolves to a canvas, as the denoiser declares
    /// it.
    pub canvas_rule: CanvasRule,
    /// How the denoiser's VAE divides a video's frames into clips, as it
    /// declares it: the frame counts it generates and a reference video's
    /// encoded frames and latent units follow from it.
    pub frames: FrameClips,
    /// The only canvases the deployment serves, as its workers declare them
    /// in their handshake: each with its short edge and aspect ratio, in the
    /// order the deployment
    /// names them.
    pub canvases: Vec<ServedCanvas>,
    /// The references one `ref2va` request may carry, as the denoiser
    /// declares them in its handshake.
    pub reference_limits: ReferenceLimits,
    /// How the deployment sizes references.
    pub references: ReferenceConfig,
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
/// [`proto::video::AudioClip`] and the latents it encodes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AudioClip {
    /// The native sample rate.
    pub sample_rate: u32,
    /// Native samples skipped at the start.
    pub start_sample: u64,
    /// Native samples kept after the start offset.
    pub source_samples: u64,
    /// Samples after resampling to the model's sample rate.
    pub samples: u64,
    /// Audio latents per channel; the clip packs `2 * latents` channel-major
    /// rows.
    pub latents: u32,
}

impl AudioClip {
    /// Denoiser rows of the clip: one per latent per stereo channel
    /// ([`crate::packing::track_rows`]).
    pub const fn rows(&self) -> u32 {
        crate::packing::track_rows(self.latents)
    }

    /// What the media reader decodes.
    const fn media(&self) -> proto::video::AudioClip {
        proto::video::AudioClip {
            sample_rate: self.sample_rate,
            start_sample: self.start_sample,
            source_samples: self.source_samples,
            samples: self.samples,
        }
    }
}

/// The frames of a reference video the request conditions on: the media
/// reader's [`proto::video::VideoClip`], the latent frames it encodes to and its
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
pub struct ConditionInput {
    /// Position of the condition in the request.
    pub index: usize,
    /// The preparation of its media.
    pub prepared: Prepared,
    /// The conditioner's view, for conditions it reads.
    pub vision: Option<Vision>,
    /// Denoiser audio condition rows.
    pub audio_rows: u32,
}

impl ConditionInput {
    /// The condition as a denoiser lays it out ([`nn::ConditionShape`]): its role, the frames and
    /// raster the video encoder encodes and the samples of its soundtrack, as the worker reads them
    /// from the condition's described media. `canvas` is the generated canvas, which a keyframe is
    /// fitted to.
    pub fn shape(&self, canvas: Canvas) -> nn::ConditionShape {
        let (role, media) = self.media(canvas);
        // Audio alone has no frames and no raster.
        let (frames, raster) = media.pixels().unwrap_or((
            0,
            Canvas {
                width: 0,
                height: 0,
            },
        ));
        nn::ConditionShape {
            role: match role {
                ConditionRole::FirstFrame => nn::ConditionRole::FirstFrame,
                ConditionRole::LastFrame => nn::ConditionRole::LastFrame,
                ConditionRole::Reference => nn::ConditionRole::Reference,
            },
            frames,
            height: raster.height,
            width: raster.width,
            audio_samples: media
                .audio()
                .map_or(0, |clip| u32::try_from(clip.samples).unwrap_or(u32::MAX)),
        }
    }

    /// The condition's described role and what the media reader decodes its
    /// bytes into. `canvas` is the generated canvas, which a keyframe is
    /// fitted to.
    fn media(&self, canvas: Canvas) -> (ConditionRole, ConditionMedia) {
        match &self.prepared {
            Prepared::Keyframe(fit) => {
                let role = match fit.position {
                    FramePosition::First => ConditionRole::FirstFrame,
                    FramePosition::Last => ConditionRole::LastFrame,
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
                ConditionRole::Reference,
                ConditionMedia::Image(ImageFit {
                    resized: *size,
                    left: 0,
                    top: 0,
                    size: *size,
                }),
            ),
            Prepared::Video(clip) => (
                ConditionRole::Reference,
                ConditionMedia::Video {
                    clip: proto::video::VideoClip {
                        canvas: clip.canvas,
                        start_frame: clip.start_frame,
                        frames: clip.frames,
                        vae_frames: clip.vae_frames,
                    },
                    soundtrack: clip.soundtrack.as_ref().map(AudioClip::media),
                },
            ),
            Prepared::Audio(clip) => (
                ConditionRole::Reference,
                ConditionMedia::Audio(clip.media()),
            ),
        }
    }

    /// Denoiser video rows of each unit the video encoder encodes, in unit
    /// order. `canvas` is the generated canvas.
    ///
    /// A keyframe is one unit, one latent frame of the canvas it is fitted
    /// to. A reference image is `image_bands` units: bands of its rows of
    /// denoiser patches (32x32 pixels) dealt as evenly as possible in order, so with `g`
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
            Prepared::Keyframe(_) => vec![LatentRaster::of(&canvas).patch_rows()],
            Prepared::Image(size) => {
                let raster = LatentRaster::of(size);
                let patch_rows = raster.height / PATCH_SIDE;
                let patches_per_row = raster.width / PATCH_SIDE;
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
                let rows_per_frame = LatentRaster::of(&clip.canvas).patch_rows();
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
    pub fn describe(&self, canvas: Canvas, image_bands: u32) -> ConditionPlan {
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
        ConditionPlan {
            role,
            media,
            vision,
            latent_units: self.latent_units(canvas, image_bands),
            audio_rows: self.audio_rows,
        }
    }
}

/// Denoiser rows planned `conditions` take in a layout of `packing`, as the
/// denoiser's packing counts them ([`crate::packing::Tiling::condition_rows`]):
/// each condition's audio rows in whole tiles, then a keyframe's or an image's
/// rows in whole tiles, or the tiles of a video's token grid. `canvas` is the
/// generated canvas, which a keyframe is fitted to.
///
/// # Errors
///
/// Fails when the packing refuses a planned condition, which the rules plan
/// none of.
pub fn condition_rows(
    conditions: &[ConditionInput],
    canvas: Canvas,
    packing: &ConditionPacking,
) -> Result<u32, InputError> {
    let shapes: Vec<crate::packing::Condition> = conditions
        .iter()
        .map(|condition| crate::packing::Condition::of(&condition.shape(canvas)))
        .collect();
    let segments = crate::packing::condition_segments(&shapes, &canvas)
        .map_err(|error| InputError::Internal(format!("packing the conditions: {error}")))?;
    Ok(packing.tiling().condition_rows(&segments))
}

/// The resolved size of a request and of every condition.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VideoParams {
    /// The request's task.
    pub task: VideoTask,
    /// The generated canvas.
    pub canvas: Canvas,
    /// The requested duration in seconds: the stated length, or the
    /// soundtrack's length after its start offset when that sets it.
    pub duration_seconds: f64,
    /// Generated frames, a frame count of the denoiser's frame rule.
    pub num_frames: u32,
    /// One plan per condition, in request order.
    pub conditions: Vec<ConditionInput>,
}

impl VideoParams {
    /// Denoiser rows the request's conditions take in a layout of `packing`
    /// ([`condition_rows`]). The request's layout holds them: what exceeds
    /// `packing.capacity` the deployment does not serve, and within it they
    /// select the request's graph.
    ///
    /// # Errors
    ///
    /// Fails when the packing refuses a planned condition, which the rules
    /// plan none of.
    pub fn condition_rows(&self, packing: &ConditionPacking) -> Result<u32, InputError> {
        condition_rows(&self.conditions, self.canvas, packing)
    }

    /// The units each condition's reference image is encoded in when one
    /// latent encoding round covers `round_units` units, one entry per
    /// condition (other conditions' entries are unread).
    ///
    /// A lone reference image is the largest visual condition and is a
    /// single rank's work as one unit; its bands spread its encoder tiles
    /// over the ranks of one round. Fewer images than a round's units divide
    /// them evenly, at least one unit each. More images than a round's units
    /// encode whole, `round_units` images per round, except that when every
    /// visual unit is an image, the `k` images of a final partial round take
    /// `round_units / k` bands each whenever `k` divides `round_units`, so
    /// that round is full rather than leaving ranks idle behind its whole
    /// images.
    pub fn image_bands(&self, round_units: u32) -> Vec<u32> {
        let is_image = |plan: &ConditionInput| matches!(plan.prepared, Prepared::Image(_));
        let images = self.conditions.iter().filter(|plan| is_image(plan)).count() as u32;
        let mut bands = vec![(round_units / images.max(1)).max(1); self.conditions.len()];
        let only_images = self
            .conditions
            .iter()
            .all(|plan| is_image(plan) || matches!(plan.prepared, Prepared::Audio(_)));
        let partial = images % round_units.max(1);
        if only_images && images > round_units && partial > 0 && round_units.is_multiple_of(partial)
        {
            // The last `partial` images, in request order, form the final
            // round.
            let mut remaining = partial;
            for (entry, plan) in bands.iter_mut().zip(&self.conditions).rev() {
                if remaining == 0 {
                    break;
                }
                if is_image(plan) {
                    *entry = round_units / partial;
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
/// `config.enlarge`. Each side then rounds half to even to the canvas grid
/// of `rule` ([`CanvasRule::nearest_multiple`]). Unlike the canvas rule there
/// is no area cap.
///
/// # Errors
///
/// Returns [`RuleError::ReferenceAspect`] when the image lies outside the
/// rule's aspects.
pub fn reference_image_size(
    width: u32,
    height: u32,
    config: &ReferenceConfig,
    rule: &CanvasRule,
) -> Result<Canvas, RuleError> {
    if !rule.admits(width, height) {
        return Err(RuleError::ReferenceAspect {
            width,
            height,
            narrowest: rule.narrowest,
            widest: rule.widest,
        });
    }
    let mut scale = f64::from(config.image_short_edge) / f64::from(width.min(height));
    if !config.enlarge {
        scale = scale.min(1.0);
    }
    Ok(Canvas {
        width: rule.nearest_multiple(f64::from(width) * scale),
        height: rule.nearest_multiple(f64::from(height) * scale),
    })
}

/// Units of a `rate` timeline skipped by a start offset of `seconds`.
fn start_offset(seconds: f64, rate: f64) -> u64 {
    // Start offsets are validated finite and non-negative.
    (seconds * rate + 0.5).floor() as u64
}

/// Sizes a soundtrack: start offset, truncation, resampling to the model's
/// rate, latents.
///
/// The first `floor(s * rate + 0.5)` samples are skipped and the rest is
/// truncated at the native rate to the generated duration, `frames` at
/// the limits' frame rate, as the reference does before it resamples to the
/// limits' sample rate ([`resampled_len`]); the track encodes to one latent
/// frame per audio hop, the last one padded ([`audio_latent_frames`]).
///
/// # Errors
///
/// Returns [`RuleError::AudioStart`] when the offset lies past the end of
/// the soundtrack and [`RuleError::AudioLength`] when its latents exceed
/// `u32`.
pub fn audio_clip(
    media: &AudioMetadata,
    start_seconds: f64,
    frames: u32,
    limits: &PlanLimits,
) -> Result<AudioClip, RuleError> {
    let frame_rate = limits.frame_rate;
    let rate = media.sample_rate;
    let start_sample = start_offset(start_seconds, f64::from(rate));
    if start_sample >= media.samples {
        return Err(RuleError::AudioStart);
    }
    let available = media.samples - start_sample;
    // The reference truncates to `int(frames / fps * rate)` samples.
    let limit = (f64::from(frames) / f64::from(frame_rate) * f64::from(rate)).trunc() as u64;
    let source = available.min(limit);
    let samples = resampled_len(source, rate, limits.audio_sample_rate);
    Ok(AudioClip {
        sample_rate: rate,
        start_sample,
        source_samples: source,
        samples,
        latents: u32::try_from(audio_latent_frames(samples, u64::from(limits.audio_hop)))
            .map_err(|_| RuleError::AudioLength)?,
    })
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
/// the canvas rule's aspects, [`RuleError::ReferenceFrames`] when fewer than
/// the frame rule's fewest frames (22 for the released checkpoint) remain
/// after the offset, and the errors of [`audio_clip`], [`sample_frames`] and
/// [`ProcessorConfig::video_grid`].
pub fn video_reference(
    media: &VideoMetadata,
    start_seconds: f64,
    frames: u32,
    limits: &PlanLimits,
    vision: &ProcessorConfig,
) -> Result<(VideoClip, VideoVision), RuleError> {
    let (frame_rate, clips, rule) = (limits.frame_rate, &limits.frames, &limits.canvas_rule);
    // Display sizes stay far below 2^53, so the conversions are exact.
    let mut size = rule
        .canvas(media.display_width as f64, media.display_height as f64)
        .map_err(RuleError::VideoCanvas)?;
    let pixels = |width: u32, height: u32| u64::from(width) * u64::from(height);
    if !limits.references.enlarge
        && pixels(media.width, media.height) < pixels(size.width, size.height)
    {
        size = Canvas {
            width: rule.nearest_multiple(f64::from(media.width)),
            height: rule.nearest_multiple(f64::from(media.height)),
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
        .map(|track| audio_clip(track, start_seconds, frames, limits))
        .transpose()?;

    let (frame_indices, block_timestamps) =
        sample_frames(kept, vision.temporal_patch_size, frame_rate)?;
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

/// The canvas a request names without consulting media; `None` for an
/// `fl2va` request without a size, whose canvas follows its first keyframe.
///
/// A stated size must be one of the canvases the deployment serves
/// (`limits.canvases`). Without one, a `t2va` or `ref2va` request takes the
/// 16:9 canvas at the short edge of the deployment's first canvas.
///
/// # Errors
///
/// Refuses `size` when the stated size is not served, or when no 16:9 canvas
/// is served at the first canvas's short edge for a request that states none.
pub fn named_canvas(
    task: VideoTask,
    size: Option<Canvas>,
    limits: &PlanLimits,
) -> Result<Option<Canvas>, InputError> {
    if let Some(size) = size {
        if limits.canvases.iter().any(|served| served.canvas == size) {
            return Ok(Some(size));
        }
        return Err(InputError::invalid(
            SIZE,
            format!(
                "{}x{} is not served; this deployment serves {}",
                size.width,
                size.height,
                served_sizes(limits).join(", ")
            ),
        ));
    }
    if task == VideoTask::FirstLastToVideoAudio {
        return Ok(None);
    }
    let short_edge = default_short_edge(limits);
    let ratio = format!("{}:{}", DEFAULT_ASPECT_RATIO.0, DEFAULT_ASPECT_RATIO.1);
    limits
        .canvases
        .iter()
        .find(|served| served.short_edge == short_edge && served.aspect_ratio == ratio)
        .map(|served| Some(served.canvas))
        .ok_or_else(|| {
            InputError::invalid(
                SIZE,
                format!(
                    "is required: this deployment serves no {ratio} canvas at a \
                     {short_edge}-pixel short edge; it serves {}",
                    served_sizes(limits).join(", ")
                ),
            )
        })
}

/// The short edge a request without a size is generated at: the deployment's
/// first canvas's, or the canvas rule's when it names none.
fn default_short_edge(limits: &PlanLimits) -> u32 {
    limits
        .canvases
        .first()
        .map_or(limits.canvas_rule.short_edge, |served| served.short_edge)
}

/// The sizes the deployment serves, as a request names them.
pub fn served_sizes(limits: &PlanLimits) -> Vec<String> {
    limits
        .canvases
        .iter()
        .map(|served| format!("{}x{}", served.canvas.width, served.canvas.height))
        .collect()
}

/// Checks that a canvas is one the deployment's workers prepared.
fn check_served(size: Canvas, limits: &PlanLimits) -> Result<(), InputError> {
    if limits.canvases.iter().any(|served| served.canvas == size) {
        return Ok(());
    }
    Err(InputError::invalid(
        SIZE,
        format!(
            "resolves to {}x{}, which this deployment does not serve",
            size.width, size.height
        ),
    ))
}

/// The duration a request states, in seconds: a stated frame count, which
/// must be one of the frame rule, lasts its frames at the output frame rate.
///
/// # Errors
///
/// Refuses `num_frames` when it is not a frame count of the denoiser's rule.
fn stated_seconds(request: &VideoSpec, limits: &PlanLimits) -> Result<Option<f64>, InputError> {
    Ok(match request.length {
        None => None,
        Some(Length::Seconds(seconds)) => Some(seconds),
        Some(Length::Frames(frames)) => {
            let rule = limits.frames;
            if !rule.is_legal(frames) {
                return Err(InputError::invalid(
                    "num_frames",
                    format!(
                        "{frames} is not a frame count MiniMax-H3 generates: {} frames and {} \
                         more per clip",
                        rule.min_frames, rule.clip_frames
                    ),
                ));
            }
            Some(f64::from(frames) / f64::from(limits.frame_rate))
        }
    })
}

/// Applies every rule that needs no media, in a fixed order: the served task,
/// the size, each reference's fields, the task's reference counts and
/// keyframe order, then the length.
///
/// [`plan_request`] applies these rules again; calling this first rejects a
/// malformed request before its media are fetched.
///
/// # Errors
///
/// Returns [`InputError::Invalid`] naming the first public parameter at
/// fault.
pub fn check_request(request: &VideoSpec, limits: &PlanLimits) -> Result<(), InputError> {
    let task = task(request);
    if !limits.tasks.contains(&task) {
        let served: Vec<&str> = limits.tasks.iter().map(|task| task.as_str()).collect();
        return Err(InputError::invalid(
            REFERENCES,
            format!(
                "select {}, which is not served; this deployment serves {}",
                task.as_str(),
                served.join(", ")
            ),
        ));
    }

    named_canvas(task, request.size, limits)?;
    check_references(task, &request.references, &limits.reference_limits)?;

    let length = length_param(request);
    match stated_seconds(request, limits)? {
        Some(seconds) => {
            video_frame_count(
                seconds,
                limits.max_video_seconds,
                limits.frame_rate,
                &limits.frames,
            )
            .map_err(|error| InputError::invalid(length, error.to_string()))?;
        }
        None => {
            let may_follow_audio = task == VideoTask::ReferenceToVideoAudio
                && request
                    .references
                    .iter()
                    .any(|reference| reference.kind != ReferenceKind::Image);
            if !may_follow_audio {
                return Err(InputError::invalid(
                    length,
                    "is required unless a ref2va request has exactly one reference with audio",
                ));
            }
        }
    }
    Ok(())
}

/// Checks each reference's fields, then the task's counts and keyframe
/// order.
fn check_references(
    task: VideoTask,
    references: &[Reference],
    limits: &ReferenceLimits,
) -> Result<(), InputError> {
    for (index, reference_) in references.iter().enumerate() {
        let reject = |message: String| Err(reference(index, message));
        if let Some(frame_index) = reference_.frame_index {
            if reference_.kind != ReferenceKind::Image {
                return reject("a keyframe must be an image".to_owned());
            }
            if !matches!(frame_index, 0 | -1) {
                return reject(format!("frame_index must be 0 or -1, got {frame_index}"));
            }
        }
        if let Some(start) = reference_.start_time_seconds {
            if is_keyframe(reference_) || reference_.kind != ReferenceKind::Video {
                return reject("start_time_seconds applies to video references only".to_owned());
            }
            if !start.is_finite() || start < 0.0 {
                return reject("start_time_seconds must be finite and >= 0".to_owned());
            }
        }
    }

    let signature: Vec<i64> = references
        .iter()
        .filter_map(|reference| reference.frame_index)
        .collect();
    let keyframes_valid = matches!(signature.as_slice(), [0] | [-1] | [0, -1]);
    if (task == VideoTask::FirstLastToVideoAudio || !signature.is_empty()) && !keyframes_valid {
        return Err(InputError::invalid(
            REFERENCES,
            format!("keyframe frame_index values must be [0], [-1] or [0, -1], got {signature:?}"),
        ));
    }

    if task == VideoTask::ReferenceToVideoAudio {
        let count = |kind: ReferenceKind| {
            references
                .iter()
                .filter(|reference| !is_keyframe(reference) && reference.kind == kind)
                .count()
        };
        let images = count(ReferenceKind::Image);
        let audios = count(ReferenceKind::Audio);
        let videos = count(ReferenceKind::Video);
        // Each limit names what it bounds; the last bounds every kind.
        for (bounded, found, limit) in [
            ("image references", images, limits.images),
            ("video references", videos, limits.videos),
            ("audio references", audios, limits.audios),
            ("references", images + videos + audios, limits.total),
        ] {
            if found > limit as usize {
                return Err(InputError::invalid(
                    REFERENCES,
                    format!("at most {limit} {bounded}, got {found}"),
                ));
            }
        }
        if images + videos == 0 {
            return Err(InputError::invalid(
                REFERENCES,
                "ref2va needs at least one image or video reference",
            ));
        }
    }
    Ok(())
}

/// Checks that the probed media is the reference's kind.
fn check_media(
    index: usize,
    reference_: &Reference,
    media: &MediaMetadata,
) -> Result<(), InputError> {
    let fits = matches!(
        (reference_.kind, media),
        (ReferenceKind::Image, MediaMetadata::Image(_))
            | (ReferenceKind::Video, MediaMetadata::Video(_))
            | (ReferenceKind::Audio, MediaMetadata::Audio(_))
    );
    if !fits {
        return Err(reference(
            index,
            format!("the media is not {}", reference_.kind.name()),
        ));
    }
    Ok(())
}

/// The soundtrack a reference conditions on, if any.
fn soundtrack<'a>(reference_: &Reference, media: &'a MediaMetadata) -> Option<&'a AudioMetadata> {
    if is_keyframe(reference_) {
        return None;
    }
    match media {
        MediaMetadata::Audio(audio) => Some(audio),
        MediaMetadata::Video(video) => video.soundtrack.as_ref(),
        MediaMetadata::Image(_) => None,
    }
}

/// The generated frame count, from the stated length or from the one
/// soundtrack.
fn resolve_frames(
    request: &VideoSpec,
    soundtracks: &[(usize, &AudioMetadata, f64)],
    limits: &PlanLimits,
) -> Result<(f64, u32), InputError> {
    let length = length_param(request);
    if let Some(seconds) = stated_seconds(request, limits)? {
        return video_frame_count(
            seconds,
            limits.max_video_seconds,
            limits.frame_rate,
            &limits.frames,
        )
        .map(|frames| (seconds, frames))
        .map_err(|error| InputError::invalid(length, error.to_string()));
    }
    let [(index, track, start_seconds)] = soundtracks else {
        return Err(InputError::invalid(
            length,
            format!(
                "is required unless exactly one reference has audio, got {}",
                soundtracks.len()
            ),
        ));
    };
    let rate = f64::from(track.sample_rate);
    let start_sample = start_offset(*start_seconds, rate);
    if start_sample >= track.samples {
        return Err(reference(
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
        reference(
            *index,
            format!("the audio track sets the duration, but {error}"),
        )
    })
}

/// The generated canvas: the one the request names, or the one the first
/// keyframe sets.
fn target_canvas(
    task: VideoTask,
    request: &VideoSpec,
    media: &[MediaMetadata],
    limits: &PlanLimits,
) -> Result<Canvas, InputError> {
    if let Some(size) = named_canvas(task, request.size, limits)? {
        return Ok(size);
    }
    // fl2va without a size: the references are keyframes and the first one
    // sets the canvas through the canvas rule, at its short edge.
    let rule = &limits.canvas_rule;
    if default_short_edge(limits) != rule.short_edge {
        return Err(InputError::invalid(
            SIZE,
            format!(
                "is required: the first keyframe sets a canvas at a {}-pixel short edge, which \
                 this deployment does not serve; it serves {}",
                rule.short_edge,
                served_sizes(limits).join(", ")
            ),
        ));
    }
    let Some(MediaMetadata::Image(first)) = media.first() else {
        return Err(reference(0, "the first keyframe is not an image"));
    };
    rule.canvas(f64::from(first.width), f64::from(first.height))
        .map_err(|error| {
            reference(
                0,
                format!("the first keyframe sets the canvas, but its {error}"),
            )
        })
}

pub(crate) fn plan_keyframe(
    index: usize,
    reference_: &Reference,
    image: &ImageMetadata,
    size: Canvas,
    order: usize,
    task: VideoTask,
    vision: &ProcessorConfig,
) -> Result<ConditionInput, InputError> {
    let position = if reference_.frame_index == Some(0) {
        FramePosition::First
    } else {
        FramePosition::Last
    };
    // The first keyframe anchors the canvas and is stretched onto it; a
    // second one follows and is cover-cropped.
    let crop = (order > 0).then(|| cover_crop(image.width, image.height, size));
    let seen = if task == VideoTask::FirstLastToVideoAudio {
        let grid = vision
            .image_grid(size.height, size.width)
            .map_err(|error| reference(index, error.to_string()))?;
        Some(Vision::Image(ImageVision {
            grid,
            tokens: vision.block_tokens(grid),
        }))
    } else {
        None
    };
    Ok(ConditionInput {
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
    reference_: &Reference,
    media: &MediaMetadata,
    frames: u32,
    vision: &ProcessorConfig,
    limits: &PlanLimits,
) -> Result<ConditionInput, InputError> {
    let reject = |error: RuleError| reference(index, error.to_string());
    let plan = |prepared, seen, audio_rows| ConditionInput {
        index,
        prepared,
        vision: seen,
        audio_rows,
    };
    match media {
        MediaMetadata::Image(image) => {
            let size = reference_image_size(
                image.width,
                image.height,
                &limits.references,
                &limits.canvas_rule,
            )
            .map_err(reject)?;
            let grid = vision
                .image_grid(size.height, size.width)
                .map_err(|error| reject(error.into()))?;
            let seen = Vision::Image(ImageVision {
                grid,
                tokens: vision.block_tokens(grid),
            });
            Ok(plan(Prepared::Image(size), Some(seen), 0))
        }
        MediaMetadata::Audio(audio) => {
            let clip = audio_clip(audio, 0.0, frames, limits).map_err(reject)?;
            Ok(plan(Prepared::Audio(clip), None, clip.rows()))
        }
        MediaMetadata::Video(video) => {
            let start = reference_.start_time_seconds.unwrap_or(0.0);
            let (clip, seen) =
                video_reference(video, start, frames, limits, vision).map_err(reject)?;
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
/// `media` holds the probed facts of each reference, in request order. The
/// caller applies the request-only rules of [`check_request`] first, which
/// need no media and so run before it is fetched.
///
/// # Errors
///
/// Returns [`InputError::Invalid`] naming the first public parameter at
/// fault, and [`InputError::Internal`] when `media` and the references differ
/// in length.
pub fn plan_request(
    request: &VideoSpec,
    media: &[MediaMetadata],
    vision: &ProcessorConfig,
    limits: &PlanLimits,
) -> Result<VideoParams, InputError> {
    let references = &request.references;
    if media.len() != references.len() {
        return Err(InputError::Internal(format!(
            "{} references were planned with {} probed media",
            references.len(),
            media.len()
        )));
    }
    let task = task(request);

    let mut soundtracks = Vec::new();
    for (index, (reference_, facts)) in references.iter().zip(media).enumerate() {
        check_media(index, reference_, facts)?;
        if let Some(track) = soundtrack(reference_, facts) {
            soundtracks.push((index, track, reference_.start_time_seconds.unwrap_or(0.0)));
        }
    }
    let (duration_seconds, frames) = resolve_frames(request, &soundtracks, limits)?;

    let size = target_canvas(task, request, media, limits)?;
    check_served(size, limits)?;

    let mut plans = Vec::with_capacity(references.len());
    let mut keyframes = 0;
    for (index, (reference_, facts)) in references.iter().zip(media).enumerate() {
        let plan = match (is_keyframe(reference_), facts) {
            (true, MediaMetadata::Image(image)) => {
                keyframes += 1;
                plan_keyframe(index, reference_, image, size, keyframes - 1, task, vision)?
            }
            (true, _) => return Err(reference(index, "a keyframe must be an image")),
            (false, _) => plan_reference(index, reference_, facts, frames, vision, limits)?,
        };
        plans.push(plan);
    }
    Ok(VideoParams {
        task,
        canvas: size,
        duration_seconds,
        num_frames: frames,
        conditions: plans,
    })
}

#[cfg(test)]
pub(super) mod tests {

    use serde::Deserialize;
    use serde_json::{Value, json};

    use super::super::presentation::present;
    use super::super::presentation::tests::CharacterTokenizer;
    use super::{
        AudioClip, AudioMetadata, Canvas, ConditionInput, FramePosition, ImageMetadata, InputError,
        KeyframeFit, LatentRaster, Length, MediaMetadata, PlanLimits, Prepared, Reference,
        ReferenceConfig, ReferenceKind, VideoClip, VideoMetadata, VideoParams, VideoSpec,
        VideoTask, Vision, audio_clip, check_request, cover_crop, plan_request,
        reference_image_size, video_frame_count, video_reference,
    };
    use crate::config::{AUDIO_HOP, AUDIO_SAMPLE_RATE, CANVAS_RULE, FPS, FRAMES};
    use crate::processing::{ConditionPacking, ReferenceLimits, ServedCanvas};
    use processing::FrameRate;
    use qwen3::processor::ProcessorConfig;

    /// The planning vectors, generated from the diffusers reference.
    pub(in crate::processing) fn fixture() -> Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/minimax_h3_plan.json"
        )))
        .unwrap()
    }

    pub(in crate::processing) fn vision(fixture: &Value) -> ProcessorConfig {
        ProcessorConfig::deserialize(&fixture["vision"]).unwrap()
    }

    /// The released checkpoint's documented references: nine images, three
    /// videos, three audio tracks and twelve in all, images encoded at a
    /// short edge of at most 2048 pixels.
    pub(in crate::processing) const REFERENCE_LIMITS: ReferenceLimits = ReferenceLimits {
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
    pub(in crate::processing) fn vector_canvases() -> Vec<ServedCanvas> {
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
    /// canvases of the vectors, by the released checkpoint's rules, which the
    /// vectors were generated with.
    pub(in crate::processing) fn limits() -> PlanLimits {
        PlanLimits {
            tasks: vec![
                VideoTask::TextToVideoAudio,
                VideoTask::FirstLastToVideoAudio,
                VideoTask::ReferenceToVideoAudio,
            ],
            max_video_seconds: 15.0,
            frame_rate: FPS,
            audio_sample_rate: AUDIO_SAMPLE_RATE,
            audio_hop: AUDIO_HOP,
            canvas_rule: CANVAS_RULE,
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

    fn audio_facts(value: &Value) -> AudioMetadata {
        AudioMetadata {
            sample_rate: value["sample_rate"].as_u64().unwrap() as u32,
            samples: value["samples"].as_u64().unwrap(),
        }
    }

    fn video_facts(value: &Value) -> VideoMetadata {
        let (numerator, denominator) = pair(&value["frame_rate"]);
        // The fixture states a video's displayed size as its aspect.
        let (width, height) = pair(&value["display"]);
        VideoMetadata {
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

    fn media(value: &Value) -> MediaMetadata {
        if value.get("width").is_some() {
            MediaMetadata::Image(ImageMetadata {
                width: value["width"].as_u64().unwrap() as u32,
                height: value["height"].as_u64().unwrap() as u32,
            })
        } else if value.get("display").is_some() {
            MediaMetadata::Video(video_facts(value))
        } else {
            MediaMetadata::Audio(audio_facts(value))
        }
    }

    /// A size no deployment of the vectors serves: what a request states for a
    /// target the public schema names no served canvas for.
    const UNSERVED: Canvas = Canvas {
        width: 1,
        height: 1,
    };

    /// A vectors request as the public API states it, and the media of its
    /// references.
    ///
    /// The vectors are written in the reference pipeline's schema: a task, a
    /// target of short edge, aspect ratio and duration, and conditions with
    /// roles. A request states them as references, whose frame indices select
    /// the task, a size and a duration.
    pub(in crate::processing) struct Request {
        spec: VideoSpec,
        media: Vec<MediaMetadata>,
    }

    impl Request {
        /// The public request of `case`; `None` for a case the public schema
        /// cannot state: an explicit task its references do not select, or a
        /// `video_audio` condition's required soundtrack.
        pub(in crate::processing) fn stated(case: &Value) -> Option<Self> {
            let target = &case["target"];
            let conditions = case["conditions"].as_array().unwrap();
            let references: Vec<Reference> = conditions
                .iter()
                .map(|entry| Reference {
                    kind: match entry["type"].as_str().unwrap() {
                        "video_audio" => ReferenceKind::Video,
                        kind => ReferenceKind::named(kind).unwrap(),
                    },
                    frame_index: (entry["role"] == "keyframe")
                        .then(|| entry.get("frame_index").and_then(Value::as_i64))
                        .flatten(),
                    start_time_seconds: entry.get("start_time_seconds").and_then(Value::as_f64),
                })
                .collect();
            let spec = VideoSpec {
                prompt: case["prompt"].as_str().unwrap().to_owned(),
                negative_prompt: None,
                size: public_size(case),
                length: target
                    .get("duration_seconds")
                    .and_then(Value::as_f64)
                    .map(Length::Seconds),
                fps: None,
                seed: None,
                num_inference_steps: None,
                guidance_scale: None,
                references,
                extra_params: serde_json::Map::new(),
            };
            let silent_video = case.get("error").is_some()
                && conditions
                    .iter()
                    .any(|entry| entry["type"] == "video_audio");
            let selects = super::task(&spec) == VideoTask::deserialize(&case["task"]).unwrap();
            (selects && !silent_video).then(|| Self {
                spec,
                media: conditions
                    .iter()
                    .map(|entry| media(&entry["media"]))
                    .collect(),
            })
        }

        /// The public request of an accepted case.
        pub(in crate::processing) fn parse(case: &Value) -> Self {
            Self::stated(case).expect("an accepted request is stated in the public schema")
        }

        /// The request's plan, its request-only rules applied first as the
        /// server applies them before fetching media.
        pub(in crate::processing) fn plan(
            &self,
            vision: &ProcessorConfig,
        ) -> Result<VideoParams, InputError> {
            check_request(&self.spec, &limits())?;
            plan_request(&self.spec, &self.media, vision, &limits())
        }
    }

    /// The size a vectors target states: none for `auto` at the vectors'
    /// first short edge, the served canvas its ratio names, the canvas the
    /// canvas rule resolves an `fl2va` ratio to, and otherwise a size no
    /// deployment serves.
    fn public_size(case: &Value) -> Option<Canvas> {
        let target = &case["target"];
        let short_edge = target["short_edge"].as_u64().unwrap() as u32;
        let ratio = target["aspect_ratio"].as_str().unwrap();
        let canvases = vector_canvases();
        if ratio == "auto" {
            return (short_edge != canvases[0].short_edge).then_some(UNSERVED);
        }
        if let Some(served) = canvases
            .iter()
            .find(|served| served.short_edge == short_edge && served.aspect_ratio == ratio)
        {
            return Some(served.canvas);
        }
        let by_rule = case["task"] == "fl2va" && short_edge == CANVAS_RULE.short_edge;
        let resolved = ratio
            .split_once(':')
            .and_then(|(width, height)| {
                Some((width.parse::<u32>().ok()?, height.parse::<u32>().ok()?))
            })
            .filter(|_| by_rule)
            .and_then(|(width, height)| {
                CANVAS_RULE.canvas(f64::from(width), f64::from(height)).ok()
            });
        Some(resolved.unwrap_or(UNSERVED))
    }

    /// The public request parameter a vectors refusal's reference-schema
    /// field is: the target's short edge and aspect ratio are the size, its
    /// duration the seconds, the task and the conditions the references.
    fn public_param(field: &str) -> String {
        match field {
            "target.short_edge" | "target.aspect_ratio" => "size".to_owned(),
            "target.duration_seconds" => "seconds".to_owned(),
            "task" | "conditions" => "input_references".to_owned(),
            field => field.replacen("conditions", "input_references", 1),
        }
    }

    /// The parameter a refusal names.
    fn param(error: InputError) -> String {
        match error {
            InputError::Invalid { param, .. } => param,
            InputError::Internal(message) => panic!("not a refusal: {message}"),
        }
    }

    /// A text request of `seconds` at `size`.
    fn text(size: Option<Canvas>, seconds: f64) -> VideoSpec {
        with_references(size, seconds, Vec::new())
    }

    /// A request of `seconds` at `size` with `references`.
    fn with_references(
        size: Option<Canvas>,
        seconds: f64,
        references: Vec<Reference>,
    ) -> VideoSpec {
        VideoSpec {
            prompt: "a river".to_owned(),
            negative_prompt: None,
            size,
            length: Some(Length::Seconds(seconds)),
            fps: None,
            seed: None,
            num_inference_steps: None,
            guidance_scale: None,
            references,
            extra_params: serde_json::Map::new(),
        }
    }

    /// A reference of `kind` that conditions the generation as a whole.
    fn reference(kind: ReferenceKind) -> Reference {
        Reference {
            kind,
            frame_index: None,
            start_time_seconds: None,
        }
    }

    /// An image keyframe anchoring the first generated frame.
    fn first_keyframe() -> Reference {
        Reference {
            kind: ReferenceKind::Image,
            frame_index: Some(0),
            start_time_seconds: None,
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
    fn condition_json(plan: &ConditionInput, canvas: Canvas) -> Value {
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
            let resolved = CANVAS_RULE.canvas(f64::from(width), f64::from(height));
            if case.get("error").is_some() {
                assert!(resolved.is_err(), "{case}");
            } else {
                assert_eq!(resolved.unwrap(), size(&case["canvas"]), "{case}");
            }
        }
    }

    /// Durations align up to `17n + 5` frames within the served range, with
    /// the video latent count the reference derives from them.
    #[test]
    fn durations_match_the_reference() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["duration_cases"].as_array().unwrap() {
            let seconds = case["seconds"].as_f64().unwrap();
            let plan = plan_request(&text(None, seconds), &[], &vision, &limits());
            if case.get("error").is_some() {
                assert_eq!(param(plan.unwrap_err()), "seconds", "{case}");
                assert!(
                    video_frame_count(seconds, 15.0, FPS, &FRAMES).is_err(),
                    "{case}"
                );
                continue;
            }
            let plan = plan.unwrap();
            assert_eq!(u64::from(plan.num_frames), case["num_frames"], "{case}");
            assert_eq!(
                u64::from(video_frame_count(seconds, 15.0, FPS, &FRAMES).unwrap()),
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
            let resized = reference_image_size(
                width,
                height,
                &ReferenceConfig::released(&REFERENCE_LIMITS),
                &CANVAS_RULE,
            );
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
            assert_eq!(
                u64::from(LatentRaster::of(&resized).patch_rows()),
                case["rows"],
                "{case}"
            );
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
        let video = ConditionInput {
            index: 0,
            prepared: Prepared::Video(VideoClip {
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
        let image = ConditionInput {
            prepared: Prepared::Image(size),
            ..video.clone()
        };
        assert_eq!(image.latent_units(canvas, 1), [7296]);
        let keyframe = ConditionInput {
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
        let image = ConditionInput {
            index: 0,
            prepared: Prepared::Image(size),
            vision: None,
            audio_rows: 0,
        };
        assert_eq!(image.latent_units(size, 4), [16 * 114; 4]);
        assert_eq!(image.latent_units(size, 3), [21 * 114, 21 * 114, 22 * 114]);
        assert_eq!(image.latent_units(size, 0), [64 * 114]);
        assert_eq!(image.latent_units(size, 100), [114; 64]);

        let keyframe = ConditionInput {
            prepared: Prepared::Keyframe(KeyframeFit {
                position: FramePosition::First,
                cover_crop: None,
            }),
            ..image.clone()
        };
        assert_eq!(
            keyframe.latent_units(size, 4),
            [LatentRaster::of(&size).patch_rows()]
        );
    }

    /// A latent encoding round's units are divided evenly among a request's
    /// reference images, at least one unit each; other conditions take no
    /// share.
    #[test]
    fn reference_images_share_the_latent_round() {
        let image = ConditionInput {
            index: 0,
            prepared: Prepared::Image(Canvas {
                width: 2048,
                height: 2048,
            }),
            vision: None,
            audio_rows: 0,
        };
        let audio = ConditionInput {
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
        let plan = |conditions: Vec<ConditionInput>| VideoParams {
            task: VideoTask::ReferenceToVideoAudio,
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
        // fill it: with four units per round nine images take 1 x 8 + 4
        // units and six take 1 x 4 + 2 x 2, while seven, whose final three
        // cannot share four units evenly, stay whole. An audio track takes
        // no part.
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
        for case in fixture["requests"].as_array().unwrap() {
            if case.get("expected").is_none() {
                continue;
            }
            let plan = Request::parse(case).plan(&vision).unwrap();
            for condition in &plan.conditions {
                // Bands split an image's rows without changing them.
                let described = condition.describe(plan.canvas, 4);
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
            let planned = video_reference(&facts, start, frames, &limits(), &vision);
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
            let rows = latents * LatentRaster::of(&clip.canvas).patch_rows();
            assert_eq!(u64::from(rows), case["rows"], "{case}");
        }
    }

    /// A reference is only ever reduced unless the deployment enlarges: an
    /// image above the short edge and a video above the canvas of its aspect
    /// scale down as the released pipeline scales them, and smaller ones
    /// keep their own size on the canvas grid.
    #[test]
    fn references_are_enlarged_only_on_request() {
        let sized = |width, height| Canvas { width, height };
        let unenlarged = ReferenceConfig::new(&REFERENCE_LIMITS);
        let image = |width, height, config: &ReferenceConfig| {
            reference_image_size(width, height, config, &CANVAS_RULE).unwrap()
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
        assert!(reference_image_size(4100, 1000, &unenlarged, &CANVAS_RULE).is_err());

        let fixture = fixture();
        let vision = vision(&fixture);
        let video = |width: u32, height: u32, config: &ReferenceConfig| {
            // Only the display aspect's ratio matters to the canvas rule.
            let facts = VideoMetadata {
                display_width: u64::from(width),
                display_height: u64::from(height),
                width,
                height,
                frame_rate: FrameRate {
                    numerator: 24,
                    denominator: 1,
                },
                frames: 48,
                soundtrack: None,
            };
            let limits = PlanLimits {
                references: *config,
                ..limits()
            };
            video_reference(&facts, 0.0, 124, &limits, &vision)
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
    /// short edge lies on the canvas grid up to the declared edge.
    #[test]
    fn a_reference_config_reads_processor_options() {
        let read = |options: Value| {
            ReferenceConfig::from_kwargs(
                options.as_object().unwrap(),
                &REFERENCE_LIMITS,
                &CANVAS_RULE,
            )
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
            assert_eq!(
                config.validate(&REFERENCE_LIMITS, &CANVAS_RULE).is_ok(),
                valid,
                "{edge}"
            );
        }

        // A denoiser that declares a smaller edge bounds the deployment's.
        let smaller = ReferenceLimits {
            image_short_edge: 1024,
            ..REFERENCE_LIMITS
        };
        let read = |options: Value| {
            ReferenceConfig::from_kwargs(options.as_object().unwrap(), &smaller, &CANVAS_RULE)
        };
        assert_eq!(
            read(json!({})).map(|config| config.image_short_edge),
            Ok(1024)
        );
        assert!(read(json!({"reference_image_short_edge": 2048})).is_err());
        // A denoiser that takes no image reference sizes none.
        let none = ReferenceLimits::default();
        assert!(ReferenceConfig::from_kwargs(&serde_json::Map::new(), &none, &CANVAS_RULE).is_ok());
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
                &limits(),
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
            // A dense denoiser packs them row for row, so the rows a request's layout holds are
            // the reference's too.
            let dense = ConditionPacking {
                capacity: u32::MAX,
                tile_rows: 1,
                video_tile: [1, 1, 1],
            };
            assert_eq!(
                u64::from(plan.condition_rows(&dense).unwrap()),
                expected["condition_video_rows"].as_u64().unwrap()
                    + expected["condition_audio_rows"].as_u64().unwrap(),
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

    /// Each rejected request the public schema states names the public
    /// parameter of the field the reference blames, whether planning or the
    /// presentation rejects it.
    #[test]
    fn rejected_requests_name_the_field() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["requests"].as_array().unwrap() {
            let Some(error) = case.get("error") else {
                continue;
            };
            let Some(request) = Request::stated(case) else {
                continue;
            };
            let rejection = match request.plan(&vision) {
                Ok(plan) => present(&CharacterTokenizer, &plan, case["prompt"].as_str().unwrap())
                    .expect_err("the request was accepted"),
                Err(rejection) => rejection,
            };
            assert_eq!(
                param(rejection),
                public_param(error["field"].as_str().unwrap()),
                "{}",
                case["name"]
            );
        }
    }

    const LANDSCAPE: Canvas = Canvas {
        width: 1344,
        height: 768,
    };

    /// A deployment serves only the canvases its workers prepare; another
    /// size is refused before any media is needed.
    #[test]
    fn prepared_canvases_restrict_the_size() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let limits = PlanLimits {
            canvases: vec![served(768, "16:9", 1344, 768)],
            ..limits()
        };
        let plan = plan_request(&text(None, 5.0), &[], &vision, &limits).unwrap();
        assert_eq!(plan.canvas, LANDSCAPE);

        let portrait = Canvas {
            width: 768,
            height: 1344,
        };
        let error = check_request(&text(Some(portrait), 5.0), &limits).unwrap_err();
        assert_eq!(param(error), "size");
    }

    /// A deployment that declares 480p canvases serves them: a stated size
    /// is the canvas the deployment declares, a request without one takes the
    /// 16:9 canvas at its first canvas's short edge, and a size it did not
    /// prepare is refused before any media is needed.
    #[test]
    fn declared_canvases_are_the_sizes_served() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let canvas = |width, height| Canvas { width, height };
        let declared = PlanLimits {
            canvases: vec![
                served(768, "16:9", 1344, 768),
                served(480, "21:9", 992, 416),
                served(480, "16:9", 832, 480),
            ],
            ..limits()
        };
        for (size, expected) in [
            (Some(canvas(992, 416)), canvas(992, 416)),
            (Some(canvas(832, 480)), canvas(832, 480)),
            (None, LANDSCAPE),
        ] {
            let plan = plan_request(&text(size, 5.0), &[], &vision, &declared).unwrap();
            assert_eq!(plan.canvas, expected, "{size:?}");
        }
        for size in [canvas(1280, 720), canvas(768, 768)] {
            let error = check_request(&text(Some(size), 5.0), &declared).unwrap_err();
            assert!(error.to_string().contains("is not served"), "{error}");
            assert_eq!(param(error), "size");
        }

        // A deployment whose first short edge serves no 16:9 canvas has no
        // default for a text request.
        let portrait_first = PlanLimits {
            canvases: vec![served(480, "21:9", 992, 416)],
            ..limits()
        };
        let error = check_request(&text(None, 5.0), &portrait_first).unwrap_err();
        assert_eq!(param(error), "size");
    }

    /// An `fl2va` request without a size takes the canvas its first keyframe
    /// sets through the canvas rule, at its 768-pixel short edge, but only
    /// onto a canvas the deployment declares.
    #[test]
    fn a_keyframe_canvas_lands_on_a_declared_canvas() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let limits = PlanLimits {
            canvases: vec![served(768, "16:9", 1344, 768)],
            ..limits()
        };
        let request = with_references(None, 5.0, vec![first_keyframe()]);
        assert!(check_request(&request, &limits).is_ok());
        let keyframe = |width, height| [MediaMetadata::Image(ImageMetadata { width, height })];
        // A 7:4 keyframe at a 768-pixel short edge is 1344x768, the declared
        // canvas.
        let plan = plan_request(&request, &keyframe(1400, 800), &vision, &limits).unwrap();
        assert_eq!(plan.canvas, LANDSCAPE);
        // A 4:3 keyframe sets 1024x768, which the deployment does not serve.
        let error = plan_request(&request, &keyframe(1024, 768), &vision, &limits).unwrap_err();
        assert_eq!(param(error), "size");
    }

    /// The served maximum duration narrows the model's own range.
    #[test]
    fn max_video_seconds_bounds_the_duration() {
        let limits = PlanLimits {
            max_video_seconds: 8.0,
            ..limits()
        };
        let error = check_request(&text(None, 10.0), &limits).unwrap_err();
        assert_eq!(param(error), "seconds");
    }

    /// Each reference-count refusal names what it bounds: nine images, three
    /// videos and three audio tracks are each within their own limit, but
    /// thirteen references exceed the twelve of every kind together.
    #[test]
    fn reference_limits_name_what_they_bound() {
        let refusal = |references: Vec<Reference>| match check_request(
            &with_references(None, 5.0, references),
            &limits(),
        ) {
            Err(InputError::Invalid { param, message }) => {
                assert_eq!(param, "input_references");
                message
            }
            other => panic!("the references were not refused: {other:?}"),
        };

        let images = vec![reference(ReferenceKind::Image); 10];
        assert_eq!(refusal(images), "at most 9 image references, got 10");

        let mut every_kind = vec![reference(ReferenceKind::Image); 9];
        every_kind.extend(vec![reference(ReferenceKind::Video); 3]);
        every_kind.push(reference(ReferenceKind::Audio));
        assert_eq!(refusal(every_kind), "at most 12 references, got 13");
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
        let image = reference(ReferenceKind::Image);
        let video = reference(ReferenceKind::Video);
        let audio = reference(ReferenceKind::Audio);
        let check = |references: Vec<Reference>| {
            check_request(&with_references(None, 5.0, references), &limits)
        };

        assert!(check(vec![image, image, video]).is_ok());
        for (references, bound) in [
            (vec![image; 3], "at most 2 image references"),
            (vec![image, video, video], "at most 1 video references"),
            (vec![image, audio, audio], "at most 1 audio references"),
            (vec![image, image, video, audio], "at most 3 references"),
        ] {
            let error = check(references).unwrap_err();
            assert!(error.to_string().contains(bound), "{bound}: {error}");
            assert_eq!(param(error), "input_references");
        }
    }

    /// A task outside the placed denoiser's set is refused first, on the
    /// references that select it.
    #[test]
    fn only_served_tasks_are_accepted() {
        let limits = PlanLimits {
            tasks: vec![VideoTask::ReferenceToVideoAudio],
            ..limits()
        };
        let error = check_request(&text(None, 5.0), &limits).unwrap_err();
        assert_eq!(param(error), "input_references");
    }
}
