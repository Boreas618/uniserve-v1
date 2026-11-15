//! MiniMax-H3 request rules: what a request generates and how every condition is prepared.
//!
//! [`H3Processor::prepare`] resolves everything a request implies before any pixel is decoded: the
//! generated canvas and frame count, how every condition is prepared (keyframe fit, reference image
//! size, reference clip at the output frame rate, 32 kHz soundtrack), what the Qwen3-VL conditioner
//! sees of it (vision grids, 2 fps samples and block timestamps), and the denoiser rows of each
//! condition's encoder units. [`H3Processor::check`] applies the rules that need no media, so a
//! malformed request is refused before its media are fetched.
//!
//! The arithmetic reproduces the diffusers MiniMax-H3 pipeline and the checkpoint's Qwen3-VL
//! processor operation for operation, in `f64` where they compute in Python floats, so that every
//! rounding lands on the same side. The request contract the FastH3 guide publishes
//! (`docs/minimax_h3/minimax_h3.md`, MiniMax-H3 mapping) also states what the public API asks
//! beyond the pipeline's own inputs, and these rules implement it:
//!
//! - `start_time_seconds` skips the first `floor(s * fps + 0.5)` frames of a reference video's
//!   timeline at the output frame rate and the first `floor(s * rate + 0.5)` samples of its
//!   soundtrack;
//! - a reference video runs at 23.976 to 60 frames per second ([`REFERENCE_FRAME_RATES`]) and needs
//!   at least the fewest frames of the frame rule (one VAE clip and its tail, 22 frames) on that
//!   timeline after the start offset;
//! - `ref2va` keyframes are fitted to the target canvas like `fl2va` ones (the first stretched, a
//!   second cover-cropped) and do not enter the conditioner;
//! - without a stated length, a `ref2va` request whose references carry exactly one soundtrack
//!   lasts as long as that soundtrack after its start offset.

use std::num::NonZeroU32;

use proto::video::{Canvas, ConditionMedia, ConditionRole, ConditionVision, ImageFit, VisionGrid};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::processor::H3Processor;
use super::{
    AUDIO_SAMPLE_RATE, CANVAS_RULE, ConditionPacking, FPS, FRAMES, MIN_SECONDS, PATCH_PIXELS,
    ReferenceLimits, VideoTask, audio_latents, patch_rows, track_rows,
};
use crate::canvas::{CoverCrop, cover_crop};
use crate::frames::requested_frames;
use crate::media::{AudioMetadata, ImageMetadata, MediaMetadata, VideoMetadata, resampled_len};
use crate::qwen_vl::{ProcessorConfig, sample_frames};
use crate::video::{ConditionInput, InputError, Length, Reference, ReferenceKind, VideoSpec};

/// The keyframe sets a request may anchor, by the generated frames their `frame_index` values
/// name: the first, the last, or both. A `ref2va` request may also anchor none.
pub const KEYFRAME_ANCHORS: [&[i64]; 3] = [&[0], &[-1], &[0, -1]];
/// The frame rates a reference video may run at, the lowest and the highest, each as
/// `(numerator, denominator)` frames per second: from 23.976 (film at NTSC rates, 24000/1001, to
/// three decimals) to 60, both included.
pub const REFERENCE_FRAME_RATES: [(u64, u64); 2] = [(2997, 125), (60, 1)];

/// The public request parameter a refusal of the output size names.
pub const SIZE: &str = "size";
/// The public request parameter a refusal of the prompt names.
pub const PROMPT: &str = "prompt";
/// The public request parameter a refusal of the references as a whole names; a refusal of one
/// reference names `input_references[i]`.
pub const REFERENCES: &str = "input_references";

/// Refuses the request's reference `index` with `message`.
fn reference(index: usize, message: impl Into<String>) -> InputError {
    InputError::invalid(format!("{REFERENCES}[{index}]"), message)
}

/// The public request parameter that states the request's length: `num_frames` when it states a
/// frame count, `seconds` otherwise (a request that states none is refused for want of `seconds`).
fn length_param(request: &VideoSpec) -> &'static str {
    match request.length {
        Some(Length::Frames(_)) => "num_frames",
        Some(Length::Seconds(_)) | None => "seconds",
    }
}

/// The task a request's references select: none is `t2va`, image keyframes alone (`frame_index` 0
/// or -1) are `fl2va`, and any reference without a frame index makes it `ref2va`.
pub fn task(request: &VideoSpec) -> VideoTask {
    let keyframes = request.references.iter().filter(|r| is_keyframe(r)).count() as u32;
    let references = request.references.len() as u32 - keyframes;
    VideoTask::of(keyframes, 0, references)
}

/// Whether a reference anchors a generated frame: a keyframe states the frame it anchors.
fn is_keyframe(reference: &Reference) -> bool {
    reference.frame_index.is_some()
}

/// The output frame count of an explicit duration: `seconds * FPS` rounded half to even
/// ([`requested_frames`]), extended upward to the next count of the frame rule
/// ([`crate::FrameClips::align_up`]). The output may therefore last longer than requested: 4
/// seconds yield 107 frames and 15 seconds 362.
///
/// # Errors
///
/// When `seconds` is not finite or lies outside `[MIN_SECONDS, max_seconds]`.
pub fn frame_count(seconds: f64, max_seconds: f64) -> Result<u32, String> {
    if !seconds.is_finite() || !(MIN_SECONDS..=max_seconds).contains(&seconds) {
        return Err(format!(
            "video duration must be a finite number of seconds in [{MIN_SECONDS}, {max_seconds}]"
        ));
    }
    Ok(FRAMES.align_up(requested_frames(seconds, FPS)))
}

/// The MiniMax-H3 processor's options a deployment states (`mm_processor_kwargs`); an option of
/// another name is refused.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessorOptions {
    /// Short edge, in pixels, an image reference is encoded at, at most; the edge the denoiser
    /// declares when absent.
    reference_image_short_edge: Option<u32>,
    /// Whether a reference smaller than its encoding size is enlarged to it; false when absent.
    #[serde(default)]
    enlarge_references: bool,
}

/// How a deployment sizes the references of a `ref2va` request: the MiniMax-H3 processor's options
/// (`mm_processor_kwargs`), within what the denoiser declares ([`ReferenceLimits`]).
///
/// A reference is only ever reduced by default: an image whose short edge exceeds
/// `image_short_edge` scales down to it, and a video with more pixels than the canvas of its aspect
/// scales down to that canvas, while smaller references keep their own size on the canvas grid. The
/// released pipeline also enlarges smaller ones to those sizes, which `enlarge` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceSizing {
    /// Short edge, in pixels, an image reference is encoded at, at most: a multiple of the canvas
    /// grid up to the edge the denoiser declares.
    pub image_short_edge: u32,
    /// Whether a reference smaller than its encoding size is enlarged to it: an image to
    /// `image_short_edge` and a video to the canvas of its aspect.
    pub enlarge: bool,
}

impl ReferenceSizing {
    /// The sizing a deployment's processor options state: `reference_image_short_edge`, by default
    /// the edge the denoiser declares, and `enlarge_references`, by default false.
    ///
    /// # Errors
    ///
    /// An option this processor does not take, a value of the wrong type, or a short edge off the
    /// canvas grid or beyond the declared edge, for a denoiser that takes image references.
    pub fn from_options(
        options: &Map<String, Value>,
        limits: &ReferenceLimits,
    ) -> Result<Self, String> {
        let options = ProcessorOptions::deserialize(&Value::Object(options.clone()))
            .map_err(|error| error.to_string())?;
        let sizing = Self {
            image_short_edge: options
                .reference_image_short_edge
                .unwrap_or(limits.image_short_edge),
            enlarge: options.enlarge_references,
        };

        // A denoiser that takes no image reference sizes none.
        let (edge, largest, multiple) = (
            sizing.image_short_edge,
            limits.image_short_edge,
            CANVAS_RULE.multiple,
        );
        if limits.images > 0
            && (!edge.is_multiple_of(multiple) || !(multiple..=largest).contains(&edge))
        {
            return Err(format!(
                "a reference image short edge must be a multiple of {multiple} from {multiple} to \
                 {largest} pixels, got {edge}"
            ));
        }
        Ok(sizing)
    }
}

/// The samples of a soundtrack the request conditions on: what the media reader decodes
/// ([`proto::video::AudioClip`]) and the latents it encodes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AudioClip {
    /// The native sample rate.
    pub sample_rate: u32,
    /// Native samples skipped at the start.
    pub start_sample: u64,
    /// Native samples kept after the start offset.
    pub source_samples: u64,
    /// Samples after resampling to [`AUDIO_SAMPLE_RATE`].
    pub samples: u64,
    /// Audio latents per channel.
    pub latents: u32,
}

impl AudioClip {
    /// Denoiser rows of the clip: one per latent per stereo channel ([`track_rows`]).
    pub const fn rows(&self) -> u32 {
        track_rows(self.latents)
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

/// The frames of a reference video the request conditions on and its soundtrack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VideoClip {
    /// The canvas of the video's own display aspect the frames are put on.
    pub canvas: Canvas,
    /// Frames of the output-rate timeline skipped at the start.
    pub start_frame: u32,
    /// Output-rate frames kept after the start offset, at most the generated frame count; the
    /// conditioner samples these.
    pub frames: u32,
    /// The leading frames the VAE encodes, a frame count of [`FRAMES`].
    pub vae_frames: u32,
    /// The video's soundtrack, when it has one.
    pub soundtrack: Option<AudioClip>,
}

/// What the conditioner reads of a condition.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Vision {
    /// One image block.
    Image {
        /// The patch grid; `t` is 1.
        grid: VisionGrid,
        /// Vision tokens of the image.
        tokens: u32,
    },
    /// Timestamped video blocks.
    Video {
        /// The patch grid; `t` counts vision blocks.
        grid: VisionGrid,
        /// Vision tokens of each block.
        block_tokens: u32,
        /// The output-rate frames sampled at 2 fps.
        frame_indices: Vec<u32>,
        /// The timestamp label of each block, in seconds.
        block_timestamps: Vec<f64>,
    },
}

/// How a condition's media is prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preparation {
    /// A keyframe fitted to the target canvas: stretched onto it (the first keyframe of a
    /// request, `cover_crop` `None`) or cover-cropped.
    Keyframe {
        /// The generated frame it anchors: [`ConditionRole::FirstFrame`] or
        /// [`ConditionRole::LastFrame`].
        role: ConditionRole,
        /// The crop of a following keyframe.
        cover_crop: Option<CoverCrop>,
    },
    /// An image reference resized (LANCZOS) to this size.
    Image(Canvas),
    /// A reference video clip and its soundtrack.
    Video(VideoClip),
    /// An audio reference.
    Audio(AudioClip),
}

/// How one condition is prepared and what it contributes.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedCondition {
    /// Position of the condition in the request.
    pub index: usize,
    /// The preparation of its media.
    pub preparation: Preparation,
    /// The conditioner's view, for conditions it reads.
    pub vision: Option<Vision>,
    /// Denoiser audio condition rows.
    pub audio_rows: u32,
}

impl PreparedCondition {
    /// A keyframe anchoring the frame of `role`, of an image of `image`'s size, fitted to the
    /// generated `canvas`: the request's first keyframe (`order` 0) is stretched onto it and a
    /// following one cover-cropped. An `fl2va` keyframe (`presented`) enters the conditioner at
    /// the canvas's size; a `ref2va` one does not.
    ///
    /// # Errors
    ///
    /// When the processor cannot size the canvas.
    pub fn keyframe(
        index: usize,
        role: ConditionRole,
        image: &ImageMetadata,
        canvas: Canvas,
        order: usize,
        presented: bool,
        vision: &ProcessorConfig,
    ) -> Result<Self, String> {
        let crop = (order > 0).then(|| cover_crop(image.width, image.height, canvas));
        let seen = if presented {
            let grid = vision
                .image_grid(canvas.height, canvas.width)
                .map_err(|error| error.to_string())?;
            Some(Vision::Image {
                grid,
                tokens: vision.block_tokens(grid),
            })
        } else {
            None
        };
        Ok(Self {
            index,
            preparation: Preparation::Keyframe {
                role,
                cover_crop: crop,
            },
            vision: seen,
            audio_rows: 0,
        })
    }

    /// The condition's role and what the media reader decodes its bytes into. `canvas` is the
    /// generated canvas, which a keyframe is fitted to.
    pub fn media(&self, canvas: Canvas) -> (ConditionRole, ConditionMedia) {
        match &self.preparation {
            Preparation::Keyframe { role, cover_crop } => {
                // A stretched keyframe is resized straight onto the canvas; a cover-cropped one
                // past it, then centred.
                let fit = match cover_crop {
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
                (*role, ConditionMedia::Image(fit))
            }
            Preparation::Image(size) => (
                ConditionRole::Reference,
                ConditionMedia::Image(ImageFit {
                    resized: *size,
                    left: 0,
                    top: 0,
                    size: *size,
                }),
            ),
            Preparation::Video(clip) => (
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
            Preparation::Audio(clip) => (
                ConditionRole::Reference,
                ConditionMedia::Audio(clip.media()),
            ),
        }
    }

    /// Denoiser video rows of each unit the video encoder encodes, in unit order. `canvas` is the
    /// generated canvas.
    ///
    /// A keyframe is one unit, one latent frame of the canvas it is fitted to, and a reference
    /// image one latent frame at its own size. A video's encoded frames are one unit per clip of
    /// the frame rule (five latent frames) and one for the tail (two). An audio reference has none.
    pub fn latent_units(&self, canvas: Canvas) -> Vec<u32> {
        match &self.preparation {
            Preparation::Keyframe { .. } => vec![patch_rows(canvas)],
            Preparation::Image(size) => vec![patch_rows(*size)],
            Preparation::Video(clip) => {
                let rows = patch_rows(clip.canvas);
                let mut units =
                    vec![FRAMES.clip_latent_frames * rows; FRAMES.units(clip.vae_frames) as usize];
                units.push(FRAMES.tail_latent_frames * rows);
                units
            }
            Preparation::Audio(_) => Vec::new(),
        }
    }

    /// The condition as the engine and the workers read it, before its media is published: its
    /// role and decoded media, what the conditioner reads of it, and its denoiser rows
    /// ([`Self::latent_units`]). `canvas` is the generated canvas.
    pub fn describe(&self, canvas: Canvas) -> ConditionInput {
        let (role, media) = self.media(canvas);
        let vision = self.vision.as_ref().map(|vision| match vision {
            Vision::Image { grid, tokens } => ConditionVision {
                grid: *grid,
                tokens: *tokens,
                frame_indices: Vec::new(),
            },
            Vision::Video {
                grid,
                block_tokens,
                frame_indices,
                ..
            } => ConditionVision {
                grid: *grid,
                tokens: grid.t * block_tokens,
                frame_indices: frame_indices.clone(),
            },
        });
        ConditionInput {
            role,
            media,
            vision,
            latent_units: self.latent_units(canvas),
            audio_rows: self.audio_rows,
        }
    }

    /// Tiles of `packing` the condition occupies ([`condition_tiles`]).
    fn tiles(&self, canvas: Canvas, packing: &ConditionPacking) -> u32 {
        let visual = match &self.preparation {
            Preparation::Keyframe { .. } => VisualRows::Patches(patch_rows(canvas)),
            Preparation::Image(size) => VisualRows::Patches(patch_rows(*size)),
            Preparation::Video(clip) => VisualRows::Grid([
                FRAMES.latent_frames(clip.vae_frames),
                clip.canvas.height / PATCH_PIXELS,
                clip.canvas.width / PATCH_PIXELS,
            ]),
            Preparation::Audio(_) => VisualRows::None,
        };
        condition_tiles(self.audio_rows, visual, packing)
    }
}

/// How one condition's visual rows enter its layout's tiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualRows {
    /// No pixels: an audio-only reference.
    None,
    /// Patch rows packed row for row into whole tiles: a keyframe or an image.
    Patches(u32),
    /// A `(latent frames, patch rows, patch columns)` token grid covered by video tiles: a
    /// reference video.
    Grid([u32; 3]),
}

/// Tiles of `packing` one condition occupies: its `audio_rows` fill whole tiles, then its visual
/// rows fill whole tiles, or the video tiles that cover its token grid.
///
/// This is the one statement of a condition's extent in a layout: request processing counts a
/// request's condition rows by it ([`condition_rows`]), and the denoiser's packing places the rows
/// it counts.
pub fn condition_tiles(audio_rows: u32, visual: VisualRows, packing: &ConditionPacking) -> u32 {
    let tile = packing.tile_rows.get();
    let [time, height, width] = packing.video_tile.map(NonZeroU32::get);
    let visual = match visual {
        VisualRows::None => 0,
        VisualRows::Patches(rows) => rows.div_ceil(tile),
        VisualRows::Grid([frames, rows, columns]) => {
            frames.div_ceil(time) * rows.div_ceil(height) * columns.div_ceil(width)
        }
    };
    audio_rows.div_ceil(tile) + visual
}

/// Denoiser rows `conditions` take in a layout of `packing`: the rows of the tiles each occupies
/// ([`PreparedCondition`]'s tiles). The denoiser's packing places them in exactly these rows, a
/// deployment's condition capacity bounds them, and the engine selects a request's graph by them.
/// `canvas` is the generated canvas, which a keyframe is fitted to.
pub fn condition_rows(
    conditions: &[PreparedCondition],
    canvas: Canvas,
    packing: &ConditionPacking,
) -> u32 {
    let tile = packing.tile_rows.get();
    conditions
        .iter()
        .map(|condition| condition.tiles(canvas, packing).saturating_mul(tile))
        .fold(0, u32::saturating_add)
}

/// What a request generates, and every condition's preparation.
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    /// The request's task.
    pub task: VideoTask,
    /// The generated canvas.
    pub canvas: Canvas,
    /// The requested duration in seconds: the stated length, or the soundtrack's length after its
    /// start offset when that sets it.
    pub duration_seconds: f64,
    /// Generated frames, a frame count of [`FRAMES`].
    pub num_frames: u32,
    /// One preparation per condition, in request order.
    pub conditions: Vec<PreparedCondition>,
}

impl Generation {
    /// Denoiser rows the request's conditions take in a layout of `packing` ([`condition_rows`]).
    pub fn condition_rows(&self, packing: &ConditionPacking) -> u32 {
        condition_rows(&self.conditions, self.canvas, packing)
    }
}

/// The size an image reference is encoded at.
///
/// An image whose short edge exceeds `sizing.image_short_edge` scales down to it; a smaller one
/// keeps its size, or scales up to that edge under `sizing.enlarge`. Each side then rounds half to
/// even to the canvas grid. Unlike the canvas rule there is no area cap.
///
/// # Errors
///
/// When the image lies outside the canvas rule's aspects.
pub fn reference_image_size(
    width: u32,
    height: u32,
    sizing: &ReferenceSizing,
) -> Result<Canvas, String> {
    let rule = &CANVAS_RULE;
    if !rule.admits(width, height) {
        let ((narrow_w, narrow_h), (wide_w, wide_h)) = (rule.narrowest, rule.widest);
        return Err(format!(
            "a reference image must lie within {narrow_w}:{narrow_h} and {wide_w}:{wide_h}, got \
             {width}x{height}"
        ));
    }
    let mut scale = f64::from(sizing.image_short_edge) / f64::from(width.min(height));
    if !sizing.enlarge {
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

/// Sizes a soundtrack: start offset, truncation, resampling to [`AUDIO_SAMPLE_RATE`], latents.
///
/// The first `floor(s * rate + 0.5)` samples are skipped and the rest is truncated at the native
/// rate to the generated duration, `frames` at [`FPS`], as the reference does before it resamples
/// ([`resampled_len`]); the track encodes to one latent per audio hop, the last one padded
/// ([`audio_latents`]).
///
/// # Errors
///
/// When the offset lies past the end of the soundtrack, or its latents exceed `u32`.
pub fn audio_clip(
    media: &AudioMetadata,
    start_seconds: f64,
    frames: u32,
) -> Result<AudioClip, String> {
    let rate = media.sample_rate;
    let start_sample = start_offset(start_seconds, f64::from(rate));
    if start_sample >= media.samples {
        return Err("start_time_seconds lies past the end of the audio track".to_owned());
    }
    let available = media.samples - start_sample;
    // The reference truncates to `int(frames / fps * rate)` samples.
    let limit = (f64::from(frames) / f64::from(FPS) * f64::from(rate)).trunc() as u64;
    let source = available.min(limit);
    let samples = resampled_len(source, rate, AUDIO_SAMPLE_RATE);
    Ok(AudioClip {
        sample_rate: rate,
        start_sample,
        source_samples: source,
        samples,
        latents: u32::try_from(audio_latents(samples))
            .map_err(|_| "the audio track is too long".to_owned())?,
    })
}

/// Sizes a reference video and the conditioner's view of it.
///
/// The video is resampled to [`FPS`] (each frame held until the slot of the next), its first
/// `floor(s * FPS + 0.5)` frames are skipped and the rest is truncated to the generated frame
/// count, on the canvas of its own display aspect. A video with fewer pixels than that canvas keeps
/// its own displayed size, each side rounded half to even to the canvas grid, unless
/// `sizing.enlarge` scales it up to the canvas. Its soundtrack is offset by the same start and
/// sized like an audio reference.
///
/// # Errors
///
/// When the display aspect lies outside the canvas rule's aspects, fewer than the frame rule's
/// fewest frames remain after the offset, or the soundtrack or the conditioner's processor refuses.
pub fn video_reference(
    media: &VideoMetadata,
    start_seconds: f64,
    frames: u32,
    sizing: &ReferenceSizing,
    vision: &ProcessorConfig,
) -> Result<(VideoClip, Vision), String> {
    let rule = &CANVAS_RULE;
    // Display sizes stay far below 2^53, so the conversions are exact.
    let mut size = rule
        .canvas(media.display_width as f64, media.display_height as f64)
        .map_err(|error| format!("the video's {error}"))?;
    let pixels = |width: u32, height: u32| u64::from(width) * u64::from(height);
    if !sizing.enlarge && pixels(media.width, media.height) < pixels(size.width, size.height) {
        size = Canvas {
            width: rule.nearest_multiple(f64::from(media.width)),
            height: rule.nearest_multiple(f64::from(media.height)),
        };
    }

    let (numerator, denominator) = (
        u64::from(media.frame_rate.numerator),
        u64::from(media.frame_rate.denominator),
    );
    let [(low, low_per), (high, high_per)] = REFERENCE_FRAME_RATES;
    // Compared as fractions, exactly: `n / d >= low / low_per` and `n / d <= high / high_per`.
    if numerator * low_per < low * denominator || numerator * high_per > high * denominator {
        return Err(format!(
            "a reference video must run at {:.3} to {} frames per second, got {:.3}",
            low as f64 / low_per as f64,
            high as f64 / high_per as f64,
            numerator as f64 / denominator as f64
        ));
    }
    let source_rate =
        f64::from(media.frame_rate.numerator) / f64::from(media.frame_rate.denominator);
    let fps = f64::from(FPS);
    // The timeline ends at the output-rate slot the stream's end rounds to; a stream at the output
    // rate keeps its frames, as `fps / source_rate` is then exactly one.
    let timeline = (media.frames as f64 * (fps / source_rate) + 0.5).floor() as u64;
    let start_frame = start_offset(start_seconds, fps);
    let available = timeline.saturating_sub(start_frame);
    let too_short = || {
        format!(
            "a reference video needs at least {} frames at {FPS} fps after start_time_seconds, \
             got {available}",
            FRAMES.min_frames
        )
    };
    if available < u64::from(FRAMES.min_frames) {
        return Err(too_short());
    }
    // At most the generated frames, which a `u32` counts.
    let kept = u32::try_from(available.min(u64::from(frames)))
        .map_err(|_| "a reference clip longer than a frame count holds".to_owned())?;
    // The VAE encodes the leading frames that form a frame count of its rule.
    let vae_frames = FRAMES.align_down(kept).ok_or_else(too_short)?;
    let soundtrack = media
        .soundtrack
        .as_ref()
        .map(|track| audio_clip(track, start_seconds, frames))
        .transpose()?;

    let (frame_indices, block_timestamps) =
        sample_frames(kept, vision.temporal_patch_size, FPS).map_err(|error| error.to_string())?;
    // Distinct indices below `kept`, so a `u32` counts them.
    let sampled = u32::try_from(frame_indices.len())
        .map_err(|_| "more vision samples than a frame count holds".to_owned())?;
    let grid = vision
        .video_grid(sampled, size.height, size.width)
        .map_err(|error| error.to_string())?;
    let clip = VideoClip {
        canvas: size,
        start_frame: u32::try_from(start_frame).map_err(|_| {
            format!("start_time_seconds skips {start_frame} frames, more than a frame count holds")
        })?,
        frames: kept,
        vae_frames,
        soundtrack,
    };
    let seen = Vision::Video {
        grid,
        block_tokens: vision.block_tokens(grid),
        frame_indices,
        block_timestamps,
    };
    Ok((clip, seen))
}

/// The refusal of a request that states no length and cannot take one from a soundtrack: a
/// `ref2va` request exactly one of whose references carries audio lasts as long as that track
/// after its start offset, and every other request states its length. `known` counts the
/// references known to carry audio.
fn unstated_length(request: &VideoSpec, known: usize) -> InputError {
    InputError::invalid(
        length_param(request),
        format!(
            "is required unless a ref2va request has exactly one reference with audio, got {known}"
        ),
    )
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

/// Checks each reference's fields, then the task's counts and keyframe order.
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
            if !KEYFRAME_ANCHORS
                .iter()
                .any(|anchors| anchors.contains(&frame_index))
            {
                return reject(format!(
                    "frame_index must be one of {KEYFRAME_ANCHORS:?}, got {frame_index}"
                ));
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

    let anchors: Vec<i64> = references
        .iter()
        .filter_map(|reference| reference.frame_index)
        .collect();
    let keyframes_valid = KEYFRAME_ANCHORS.contains(&anchors.as_slice());
    if (task == VideoTask::FirstLastToVideoAudio || !anchors.is_empty()) && !keyframes_valid {
        return Err(InputError::invalid(
            REFERENCES,
            format!(
                "keyframe frame_index values must be one of {KEYFRAME_ANCHORS:?}, got {anchors:?}"
            ),
        ));
    }

    if task == VideoTask::ReferenceToVideoAudio {
        let count = |kind: ReferenceKind| {
            references
                .iter()
                .filter(|reference| !is_keyframe(reference) && reference.kind == kind)
                .count()
        };
        let (images, videos, audios) = (
            count(ReferenceKind::Image),
            count(ReferenceKind::Video),
            count(ReferenceKind::Audio),
        );
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

impl H3Processor {
    /// The sizes the deployment serves, as a request names them.
    pub(super) fn served_sizes(&self) -> Vec<String> {
        self.denoiser
            .canvases
            .iter()
            .map(|served| format!("{}x{}", served.canvas.width, served.canvas.height))
            .collect()
    }

    /// The canvas a request names without consulting media; `None` for an `fl2va` request without
    /// a size, whose canvas follows its first keyframe.
    ///
    /// A stated size must be one of the canvases the deployment serves. Without one, a `t2va` or
    /// `ref2va` request takes the deployment's first canvas, the default its configuration
    /// declares by naming it first.
    fn named_canvas(
        &self,
        task: VideoTask,
        size: Option<Canvas>,
    ) -> Result<Option<Canvas>, InputError> {
        let canvases = &self.denoiser.canvases;
        if let Some(size) = size {
            if canvases.iter().any(|served| served.canvas == size) {
                return Ok(Some(size));
            }
            return Err(InputError::invalid(
                SIZE,
                format!(
                    "{}x{} is not served; this deployment serves {}",
                    size.width,
                    size.height,
                    self.served_sizes().join(", ")
                ),
            ));
        }
        if task == VideoTask::FirstLastToVideoAudio {
            return Ok(None);
        }
        Ok(Some(self.default_canvas))
    }

    /// The length a request states, as its duration in seconds and its frame count; `None` when
    /// it states none.
    ///
    /// A duration is aligned up to the frame rule ([`frame_count`]). A frame count must be one of
    /// the frame rule and lie between the counts the shortest and the longest served durations
    /// align to, which are the counts the deployment prepares layouts for: a 5-second deployment
    /// serves the 124 frames its own 5 seconds generate, although 124 frames last 5.17 seconds.
    fn stated_length(&self, request: &VideoSpec) -> Result<Option<(f64, u32)>, InputError> {
        let max_seconds = self.max_seconds;
        Ok(match request.length {
            None => None,
            Some(Length::Seconds(seconds)) => {
                let frames = frame_count(seconds, max_seconds)
                    .map_err(|message| InputError::invalid("seconds", message))?;
                Some((seconds, frames))
            }
            Some(Length::Frames(frames)) => {
                if !FRAMES.is_legal(frames) {
                    return Err(InputError::invalid(
                        "num_frames",
                        format!(
                            "{frames} is not a frame count MiniMax-H3 generates: {} frames and {} \
                             more per clip",
                            FRAMES.min_frames, FRAMES.clip_frames
                        ),
                    ));
                }
                let served =
                    |seconds| frame_count(seconds, max_seconds).map_err(InputError::Internal);
                let (fewest, most) = (served(MIN_SECONDS)?, served(max_seconds)?);
                if !(fewest..=most).contains(&frames) {
                    return Err(InputError::invalid(
                        "num_frames",
                        format!(
                            "must lie between {fewest} and {most} frames, the frame counts of \
                             {MIN_SECONDS} to {max_seconds} seconds at {FPS} fps, got {frames}"
                        ),
                    ));
                }
                Some((f64::from(frames) / f64::from(FPS), frames))
            }
        })
    }

    /// Applies every rule that needs no media, in a fixed order: the served task, the size, each
    /// reference's fields, the task's reference counts and keyframe order, then the length.
    ///
    /// # Errors
    ///
    /// Refuses the first public parameter at fault.
    pub fn check_request(&self, request: &VideoSpec) -> Result<(), InputError> {
        let task = task(request);
        if !self.denoiser.tasks.contains(&task) {
            let served: Vec<&str> = self
                .denoiser
                .tasks
                .iter()
                .map(|task| task.as_str())
                .collect();
            return Err(InputError::invalid(
                REFERENCES,
                format!(
                    "select {}, which is not served; this deployment serves {}",
                    task.as_str(),
                    served.join(", ")
                ),
            ));
        }

        self.named_canvas(task, request.size)?;
        check_references(task, &request.references, &self.denoiser.references)?;

        if self.stated_length(request)?.is_none() {
            // Before its media is probed an audio reference carries audio and a video reference
            // may.
            let count = |kind: ReferenceKind| {
                request
                    .references
                    .iter()
                    .filter(|reference| !is_keyframe(reference) && reference.kind == kind)
                    .count()
            };
            let (audios, videos) = (count(ReferenceKind::Audio), count(ReferenceKind::Video));
            if !(task == VideoTask::ReferenceToVideoAudio && audios <= 1 && audios + videos >= 1) {
                return Err(unstated_length(request, audios));
            }
        }
        Ok(())
    }

    /// The generated frame count, from the stated length or from the one soundtrack.
    fn resolve_frames(
        &self,
        request: &VideoSpec,
        soundtracks: &[(usize, &AudioMetadata, f64)],
    ) -> Result<(f64, u32), InputError> {
        if let Some(stated) = self.stated_length(request)? {
            return Ok(stated);
        }
        let (VideoTask::ReferenceToVideoAudio, [(index, track, start_seconds)]) =
            (task(request), soundtracks)
        else {
            return Err(unstated_length(request, soundtracks.len()));
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
        frame_count(seconds, self.max_seconds)
            .map(|frames| (seconds, frames))
            .map_err(|message| {
                reference(
                    *index,
                    format!("the audio track sets the duration, but {message}"),
                )
            })
    }

    /// The generated canvas: the one the request names, or the one the first keyframe sets through
    /// the canvas rule, which must be a canvas the deployment serves.
    fn target_canvas(
        &self,
        task: VideoTask,
        request: &VideoSpec,
        media: &[MediaMetadata],
    ) -> Result<Canvas, InputError> {
        let canvas = match self.named_canvas(task, request.size)? {
            Some(size) => size,
            None => {
                // `fl2va` without a size: the first keyframe sets the canvas by the canvas rule,
                // which must be one the deployment serves.
                let Some(MediaMetadata::Image(first)) = media.first() else {
                    return Err(reference(0, "the first keyframe is not an image"));
                };
                CANVAS_RULE
                    .canvas(f64::from(first.width), f64::from(first.height))
                    .map_err(|error| {
                        reference(
                            0,
                            format!("the first keyframe sets the canvas, but its {error}"),
                        )
                    })?
            }
        };
        if !self
            .denoiser
            .canvases
            .iter()
            .any(|served| served.canvas == canvas)
        {
            return Err(InputError::invalid(
                SIZE,
                format!(
                    "resolves to {}x{}, which this deployment does not serve",
                    canvas.width, canvas.height
                ),
            ));
        }
        Ok(canvas)
    }

    /// Prepares a reference that is not a keyframe, for a generation of `frames` frames.
    fn prepare_reference(
        &self,
        index: usize,
        reference_: &Reference,
        media: &MediaMetadata,
        frames: u32,
    ) -> Result<PreparedCondition, InputError> {
        let reject = |message: String| reference(index, message);
        let prepared = |preparation, vision, audio_rows| PreparedCondition {
            index,
            preparation,
            vision,
            audio_rows,
        };
        match media {
            MediaMetadata::Image(image) => {
                let size = reference_image_size(image.width, image.height, &self.sizing)
                    .map_err(reject)?;
                let grid = self
                    .vision
                    .image_grid(size.height, size.width)
                    .map_err(|error| reject(error.to_string()))?;
                let seen = Vision::Image {
                    grid,
                    tokens: self.vision.block_tokens(grid),
                };
                Ok(prepared(Preparation::Image(size), Some(seen), 0))
            }
            MediaMetadata::Audio(audio) => {
                let clip = audio_clip(audio, 0.0, frames).map_err(reject)?;
                Ok(prepared(Preparation::Audio(clip), None, clip.rows()))
            }
            MediaMetadata::Video(video) => {
                let start = reference_.start_time_seconds.unwrap_or(0.0);
                let (clip, seen) =
                    video_reference(video, start, frames, &self.sizing, &self.vision)
                        .map_err(reject)?;
                let audio_rows = clip.soundtrack.map_or(0, |track| track.rows());
                Ok(prepared(Preparation::Video(clip), Some(seen), audio_rows))
            }
        }
    }

    /// Resolves what a request generates and prepares every condition from the probed media.
    ///
    /// `media` holds the probed facts of each reference, in request order. The caller applies
    /// [`Self::check_request`] first, which needs no media and so runs before it is fetched.
    ///
    /// # Errors
    ///
    /// Refuses the first public parameter at fault, and fails internally when `media` and the
    /// references differ in length.
    pub fn prepare(
        &self,
        request: &VideoSpec,
        media: &[MediaMetadata],
    ) -> Result<Generation, InputError> {
        let references = &request.references;
        if media.len() != references.len() {
            return Err(InputError::Internal(format!(
                "{} references were prepared with {} probed media",
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
        let (duration_seconds, frames) = self.resolve_frames(request, &soundtracks)?;
        let canvas = self.target_canvas(task, request, media)?;

        let mut conditions = Vec::with_capacity(references.len());
        let mut keyframes = 0;
        for (index, (reference_, facts)) in references.iter().zip(media).enumerate() {
            let condition = match (reference_.frame_index, facts) {
                (Some(frame_index), MediaMetadata::Image(image)) => {
                    let role = if frame_index == 0 {
                        ConditionRole::FirstFrame
                    } else {
                        ConditionRole::LastFrame
                    };
                    let presented = task == VideoTask::FirstLastToVideoAudio;
                    keyframes += 1;
                    PreparedCondition::keyframe(
                        index,
                        role,
                        image,
                        canvas,
                        keyframes - 1,
                        presented,
                        &self.vision,
                    )
                    .map_err(|message| reference(index, message))?
                }
                (Some(_), _) => return Err(reference(index, "a keyframe must be an image")),
                (None, _) => self.prepare_reference(index, reference_, facts, frames)?,
            };
            conditions.push(condition);
        }
        Ok(Generation {
            task,
            canvas,
            duration_seconds,
            num_frames: frames,
            conditions,
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use serde::Deserialize;
    use serde_json::{Value, json};

    use super::super::prompt::present;
    use super::super::prompt::tests::CharacterTokenizer;
    use super::super::{DenoiserInfo, REFERENCE_LIMITS, ServedCanvas, vectors};
    use super::*;
    use crate::media::FrameRate;

    /// The conditioner's processor the vectors were generated with.
    pub(in crate::minimax_h3) fn vision(vectors: &Value) -> ProcessorConfig {
        ProcessorConfig::deserialize(&vectors["vision"]).unwrap()
    }

    /// The canvases of the deployment the vectors' requests are stated to, the first the size of a
    /// request that names none.
    fn vector_canvases() -> Vec<ServedCanvas> {
        vectors()["deployment"]["canvases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|served| ServedCanvas {
                short_edge: served["short_edge"].as_u64().unwrap() as u32,
                aspect_ratio: served["aspect_ratio"].as_str().unwrap().to_owned(),
                canvas: size(&served["size"]),
            })
            .collect()
    }

    /// The sizing of a deployment whose processor options enlarge references, as the released
    /// pipeline sizes them (which the vectors follow).
    fn released() -> ReferenceSizing {
        ReferenceSizing {
            image_short_edge: REFERENCE_LIMITS.image_short_edge,
            enlarge: true,
        }
    }

    /// The sizing of a deployment that states no processor option: references only reduced.
    fn reduced() -> ReferenceSizing {
        ReferenceSizing {
            enlarge: false,
            ..released()
        }
    }

    fn served(short_edge: u32, aspect_ratio: &str, width: u32, height: u32) -> ServedCanvas {
        ServedCanvas {
            short_edge,
            aspect_ratio: aspect_ratio.to_owned(),
            canvas: Canvas { width, height },
        }
    }

    /// A processor that serves every task and duration the model does, at the canvases of the
    /// vectors, sized as the released pipeline sizes references (which the vectors follow).
    pub(in crate::minimax_h3) fn processor() -> H3Processor {
        let vectors = vectors();
        let canvases = vector_canvases();
        H3Processor {
            default_canvas: canvases[0].canvas,
            denoiser: DenoiserInfo {
                tasks: VideoTask::ALL.to_vec(),
                schedule_points: 9,
                video_shift: 12.0,
                audio_shift: 3.0,
                canvases,
                references: REFERENCE_LIMITS,
                condition_packing: ConditionPacking::dense(0),
            },
            max_seconds: 15.0,
            sizing: released(),
            vision: vision(&vectors),
        }
    }

    /// The same processor serving only `canvases`, the first the default.
    fn serving(canvases: Vec<ServedCanvas>) -> H3Processor {
        let mut processor = processor();
        processor.default_canvas = canvases[0].canvas;
        processor.denoiser.canvases = canvases;
        processor
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
        // The vectors state a video's displayed size as its aspect.
        let (width, height) = pair(&value["display"]);
        VideoMetadata {
            display_width: u64::from(width),
            display_height: u64::from(height),
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

    /// A vectors request, stated in the public schema, and the media of its references.
    pub(in crate::minimax_h3) struct Request {
        spec: VideoSpec,
        media: Vec<MediaMetadata>,
    }

    impl Request {
        /// The public request of `case`, which states the task its references select.
        pub(in crate::minimax_h3) fn parse(case: &Value) -> Self {
            let request = &case["request"];
            let references: Vec<Reference> = request["input_references"]
                .as_array()
                .unwrap()
                .iter()
                .map(|reference| Reference {
                    kind: ReferenceKind::named(reference["type"].as_str().unwrap()).unwrap(),
                    frame_index: reference.get("frame_index").and_then(Value::as_i64),
                    start_time_seconds: reference.get("start_time_seconds").and_then(Value::as_f64),
                })
                .collect();
            let spec = VideoSpec {
                prompt: request["prompt"].as_str().unwrap().to_owned(),
                negative_prompt: None,
                size: request.get("size").map(|size| {
                    let (width, height) = size.as_str().unwrap().split_once('x').unwrap();
                    Canvas {
                        width: width.parse().unwrap(),
                        height: height.parse().unwrap(),
                    }
                }),
                length: request
                    .get("seconds")
                    .and_then(Value::as_f64)
                    .map(Length::Seconds),
                fps: None,
                seed: 0,
                num_inference_steps: None,
                guidance_scale: None,
                references,
                extra_params: Map::new(),
            };
            assert_eq!(
                task(&spec),
                VideoTask::deserialize(&case["task"]).unwrap(),
                "{}",
                case["name"]
            );
            Self {
                spec,
                media: case["media"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(media)
                    .collect(),
            }
        }

        /// The request's prompt.
        pub(in crate::minimax_h3) fn prompt(&self) -> &str {
            &self.spec.prompt
        }

        /// What the request generates, its request-only rules applied first as the server applies
        /// them before fetching media.
        pub(in crate::minimax_h3) fn prepare(&self) -> Result<Generation, InputError> {
            let processor = processor();
            processor.check_request(&self.spec)?;
            processor.prepare(&self.spec, &self.media)
        }
    }

    /// The parameter a refusal names.
    fn param(error: InputError) -> String {
        match error {
            InputError::Invalid { param, .. } | InputError::Capacity { param, .. } => param,
            InputError::Internal(message) => panic!("not a refusal: {message}"),
        }
    }

    /// A request of `seconds` at `size` with `references`.
    fn request(size: Option<Canvas>, seconds: f64, references: Vec<Reference>) -> VideoSpec {
        VideoSpec {
            prompt: "a river".to_owned(),
            negative_prompt: None,
            size,
            length: Some(Length::Seconds(seconds)),
            fps: None,
            seed: 0,
            num_inference_steps: None,
            guidance_scale: None,
            references,
            extra_params: Map::new(),
        }
    }

    /// A text request of `seconds` at `size`.
    fn text(size: Option<Canvas>, seconds: f64) -> VideoSpec {
        request(size, seconds, Vec::new())
    }

    /// A reference of `kind` that conditions the generation as a whole.
    fn reference_of(kind: ReferenceKind) -> Reference {
        Reference {
            kind,
            frame_index: None,
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

    /// A prepared condition in the vectors' layout; `canvas` is the generated canvas. A condition's
    /// video rows are its encoder units' rows.
    fn condition_json(condition: &PreparedCondition, canvas: Canvas) -> Value {
        let vision = match &condition.vision {
            None => Value::Null,
            Some(Vision::Image { grid, tokens }) => json!({
                "grid": [grid.t, grid.h, grid.w],
                "tokens": tokens,
            }),
            Some(Vision::Video {
                grid,
                block_tokens,
                frame_indices,
                block_timestamps,
            }) => json!({
                "grid": [grid.t, grid.h, grid.w],
                "block_tokens": block_tokens,
                "frame_indices": frame_indices,
                "block_timestamps": block_timestamps,
            }),
        };
        let mut entry = json!({
            "index": condition.index,
            "video_rows": condition.latent_units(canvas).iter().sum::<u32>(),
            "audio_rows": condition.audio_rows,
            "vision": vision,
        });
        let fields = match condition.preparation {
            Preparation::Keyframe { role, cover_crop } => json!({
                "kind": "keyframe",
                "position": if role == ConditionRole::FirstFrame { "first" } else { "last" },
                "cover_crop": cover_crop,
            }),
            Preparation::Image(size) => {
                json!({"kind": "image", "resize": [size.width, size.height]})
            }
            Preparation::Audio(clip) => json!({"kind": "audio", "clip": audio_json(&clip)}),
            Preparation::Video(clip) => json!({
                "kind": "video",
                "canvas": [clip.canvas.width, clip.canvas.height],
                "start_frame": clip.start_frame,
                "clip_frames": clip.frames,
                "vae_frames": clip.vae_frames,
                "latent_frames": FRAMES.latent_frames(clip.vae_frames),
                "soundtrack": clip.soundtrack.as_ref().map(audio_json),
            }),
        };
        let entry_fields = entry.as_object_mut().unwrap();
        for (key, value) in fields.as_object().unwrap() {
            entry_fields.insert(key.clone(), value.clone());
        }
        entry
    }

    /// Durations align up to `17n + 5` frames within the served range, with the video latent
    /// count the reference derives from them.
    #[test]
    fn durations_match_the_reference() {
        let processor = processor();
        for case in vectors()["duration_cases"].as_array().unwrap() {
            let seconds = case["seconds"].as_f64().unwrap();
            let generation = processor.prepare(&text(None, seconds), &[]);
            if case.get("error").is_some() {
                assert_eq!(param(generation.unwrap_err()), "seconds", "{case}");
                assert!(frame_count(seconds, 15.0).is_err(), "{case}");
                continue;
            }
            let generation = generation.unwrap();
            assert_eq!(
                u64::from(generation.num_frames),
                case["num_frames"],
                "{case}"
            );
            assert_eq!(
                u64::from(frame_count(seconds, 15.0).unwrap()),
                case["num_frames"]
            );
            assert_eq!(
                u64::from(FRAMES.latent_frames(generation.num_frames)),
                case["latent_frames"],
                "{case}"
            );
        }
    }

    /// Under the released sizing, image references resize to a 2048-pixel short edge and enter
    /// the conditioner at the processor's grid.
    #[test]
    fn reference_images_match_the_reference() {
        let vectors = vectors();
        let vision = vision(&vectors);
        for case in vectors["reference_image_cases"].as_array().unwrap() {
            let (width, height) = pair(&case["size"]);
            let resized = reference_image_size(width, height, &released());
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
            assert_eq!(u64::from(patch_rows(resized)), case["rows"], "{case}");
        }
    }

    /// A following keyframe's cover crop reproduces the reference's resize and crop box.
    #[test]
    fn keyframe_crops_match_the_reference() {
        for case in vectors()["keyframe_cases"].as_array().unwrap() {
            let (width, height) = pair(&case["size"]);
            let crop = cover_crop(width, height, size(&case["canvas"]));
            assert_eq!(json!(crop), case["cover_crop"], "{case}");
        }
    }

    /// A video's encoded frames split into the video encoder's 17-frame clips: each complete clip
    /// yields five latent frames of rows and the padded last one two, which together are the
    /// condition's rows. A still image is one unit.
    #[test]
    fn latent_units_partition_the_condition_rows() {
        let canvas = Canvas {
            width: 1344,
            height: 768,
        };
        let video = PreparedCondition {
            index: 0,
            preparation: Preparation::Video(VideoClip {
                canvas,
                start_frame: 0,
                frames: 124,
                vae_frames: 124,
                soundtrack: None,
            }),
            vision: None,
            audio_rows: 0,
        };
        // A video's units are its encoder windows.
        let units = video.latent_units(canvas);
        assert_eq!(units, [vec![5 * 1008; 7], vec![2 * 1008]].concat());
        assert_eq!(units.iter().sum::<u32>(), FRAMES.latent_frames(124) * 1008);

        // A reference image is one latent frame at its own size, a keyframe one at the generated
        // canvas.
        let image = PreparedCondition {
            preparation: Preparation::Image(Canvas {
                width: 3648,
                height: 2048,
            }),
            ..video.clone()
        };
        assert_eq!(image.latent_units(canvas), [7296]);
        let keyframe = PreparedCondition {
            preparation: Preparation::Keyframe {
                role: ConditionRole::FirstFrame,
                cover_crop: None,
            },
            ..video
        };
        assert_eq!(keyframe.latent_units(canvas), [1008]);
    }

    /// A sparse denoiser's tiles hold each condition's audio and pixels apart, and a video fills
    /// the tiles of its token grid: a 1344x768 keyframe of 1008 rows takes eight 128-row tiles, an
    /// audio clip of 80 rows one, and a 124-frame video of 37 latent frames of a 24 x 42 patch grid
    /// the `ceil(37/4) * ceil(24/4) * ceil(42/8)` tiles of a `[4, 4, 8]` tile.
    #[test]
    fn condition_rows_count_whole_tiles() {
        let canvas = Canvas {
            width: 1344,
            height: 768,
        };
        let packing = ConditionPacking::tiled(u32::MAX, 128, [4, 4, 8]).unwrap();
        let keyframe = PreparedCondition {
            index: 0,
            preparation: Preparation::Keyframe {
                role: ConditionRole::FirstFrame,
                cover_crop: None,
            },
            vision: None,
            audio_rows: 0,
        };
        assert_eq!(
            condition_rows(std::slice::from_ref(&keyframe), canvas, &packing),
            8 * 128
        );

        let audio = PreparedCondition {
            preparation: Preparation::Audio(AudioClip {
                sample_rate: 32_000,
                start_sample: 0,
                source_samples: 32_000,
                samples: 32_000,
                latents: 40,
            }),
            audio_rows: 80,
            ..keyframe.clone()
        };
        let video = PreparedCondition {
            preparation: Preparation::Video(VideoClip {
                canvas,
                start_frame: 0,
                frames: 124,
                vae_frames: 124,
                soundtrack: None,
            }),
            ..keyframe.clone()
        };
        let video_tiles = 37_u32.div_ceil(4) * 24_u32.div_ceil(4) * 42_u32.div_ceil(8);
        assert_eq!(
            condition_rows(&[keyframe, audio, video], canvas, &packing),
            (8 + 1 + video_tiles) * 128
        );
    }

    /// Every prepared condition of the vectors is described with the rows its preparation counts,
    /// in its encoder units, and its audio rows.
    #[test]
    fn described_conditions_carry_their_rows() {
        for case in vectors()["requests"].as_array().unwrap() {
            if case.get("expected").is_none() {
                continue;
            }
            let generation = Request::parse(case).prepare().unwrap();
            for condition in &generation.conditions {
                let described = condition.describe(generation.canvas);
                assert_eq!(
                    described.latent_units,
                    condition.latent_units(generation.canvas),
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

    /// Reference videos resample to the 24 fps output rate, honour the start offset, keep a whole
    /// VAE clip, and sample the reference's 2 fps vision blocks.
    #[test]
    fn reference_videos_match_the_reference() {
        let vectors = vectors();
        let vision = vision(&vectors);
        let sizing = released();
        for case in vectors["video_reference_cases"].as_array().unwrap() {
            let facts = video_facts(case);
            let start = case["start_seconds"].as_f64().unwrap();
            let frames = case["num_frames"].as_u64().unwrap() as u32;
            let prepared = video_reference(&facts, start, frames, &sizing, &vision);
            if case.get("error").is_some() {
                assert!(prepared.is_err(), "{case}");
                continue;
            }
            let (clip, seen) = prepared.unwrap();
            assert_eq!(clip.canvas, size(&case["canvas"]), "{case}");
            assert_eq!(u64::from(clip.start_frame), case["start_frame"], "{case}");
            assert_eq!(u64::from(clip.frames), case["clip_frames"], "{case}");
            assert_eq!(u64::from(clip.vae_frames), case["vae_frames"], "{case}");
            let latents = FRAMES.latent_frames(clip.vae_frames);
            assert_eq!(u64::from(latents), case["latent_frames"], "{case}");
            let Vision::Video {
                grid,
                block_tokens,
                frame_indices,
                block_timestamps,
            } = seen
            else {
                panic!("a video is seen as video blocks");
            };
            assert_eq!(json!(frame_indices), case["frame_indices"], "{case}");
            assert_eq!(json!(block_timestamps), case["block_timestamps"], "{case}");
            assert_eq!(json!([grid.t, grid.h, grid.w]), case["vision_grid"]);
            assert_eq!(u64::from(block_tokens), case["block_tokens"], "{case}");
            assert_eq!(
                u64::from(latents * patch_rows(clip.canvas)),
                case["rows"],
                "{case}"
            );
        }
    }

    /// A reference is only ever reduced unless the deployment enlarges: an image above the short
    /// edge and a video above the canvas of its aspect scale down as the released pipeline scales
    /// them, and smaller ones keep their own size on the canvas grid.
    #[test]
    fn references_are_enlarged_only_on_request() {
        let sized = |width, height| Canvas { width, height };
        let (reduced, released) = (reduced(), released());
        let image = |width, height, sizing: &ReferenceSizing| {
            reference_image_size(width, height, sizing).unwrap()
        };
        // 1080 rounds to 34 patches; 768 is already on the grid.
        assert_eq!(image(1920, 1080, &reduced), sized(1920, 1088));
        assert_eq!(image(768, 768, &reduced), sized(768, 768));
        assert_eq!(image(768, 768, &released), sized(2048, 2048));
        // A 2304-pixel short edge exceeds 2048 and is reduced either way.
        for sizing in [&reduced, &released] {
            assert_eq!(image(4096, 2304, sizing), sized(3648, 2048));
        }
        // A smaller short edge reduces further and still never enlarges.
        let half = ReferenceSizing {
            image_short_edge: 1024,
            ..reduced
        };
        assert_eq!(image(2560, 1440, &half), sized(1824, 1024));
        assert_eq!(image(768, 768, &half), sized(768, 768));
        // The 1:4 to 4:1 bound holds at any size.
        assert!(reference_image_size(4100, 1000, &reduced).is_err());

        let vision = vision(&vectors());
        let video = |width: u32, height: u32, sizing: &ReferenceSizing| {
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
            video_reference(&facts, 0.0, 124, sizing, &vision)
                .unwrap()
                .0
                .canvas
        };
        // 720p holds fewer pixels than the 16:9 canvas: 720 is 22.5 patches and rounds half to
        // even. 1080p is reduced to the canvas.
        assert_eq!(video(1280, 720, &reduced), sized(1280, 704));
        assert_eq!(video(1920, 1080, &reduced), sized(1344, 768));
        assert_eq!(video(1280, 720, &released), sized(1344, 768));
    }

    /// A reference video runs at 23.976 to 60 frames per second, both included.
    #[test]
    fn a_reference_video_runs_at_the_frame_rates_the_rules_take() {
        let vision = vision(&vectors());
        let sizing = released();
        let at = |numerator: u32, denominator: u32| {
            let facts = VideoMetadata {
                display_width: 16,
                display_height: 9,
                width: 1344,
                height: 768,
                frame_rate: FrameRate {
                    numerator,
                    denominator,
                },
                frames: 240,
                soundtrack: None,
            };
            video_reference(&facts, 0.0, 124, &sizing, &vision)
        };
        for (numerator, denominator) in [(2997, 125), (24000, 1001), (30, 1), (60, 1)] {
            assert!(
                at(numerator, denominator).is_ok(),
                "{numerator}/{denominator}"
            );
        }
        for (numerator, denominator) in [(2997, 126), (15, 1), (60001, 1000), (120, 1)] {
            let error = at(numerator, denominator).unwrap_err();
            assert!(error.contains("frames per second"), "{error}");
        }
    }

    /// The sizing reads the processor options a deployment states, absent ones keeping their
    /// defaults (the image edge the denoiser declares), and refuses an option the processor does
    /// not take; its short edge lies on the canvas grid up to the declared edge.
    #[test]
    fn the_sizing_reads_processor_options() {
        let read = |options: Value, limits: &ReferenceLimits| {
            ReferenceSizing::from_options(options.as_object().unwrap(), limits)
        };
        let limits = &REFERENCE_LIMITS;
        assert_eq!(read(json!({}), limits), Ok(reduced()));
        assert_eq!(
            read(json!({"enlarge_references": true}), limits),
            Ok(released())
        );
        assert_eq!(
            read(json!({"reference_image_short_edge": 1024}), limits),
            Ok(ReferenceSizing {
                image_short_edge: 1024,
                enlarge: false,
            })
        );
        for refused in [
            json!({"num_crops": 4}),
            json!({"enlarge_references": "yes"}),
            json!({"reference_image_short_edge": 1000}),
            json!({"reference_image_short_edge": 0}),
            json!({"reference_image_short_edge": 4096}),
        ] {
            assert!(read(refused.clone(), limits).is_err(), "{refused}");
        }
        assert!(read(json!({"reference_image_short_edge": 32}), limits).is_ok());

        // A denoiser that declares a smaller edge bounds the deployment's.
        let smaller = ReferenceLimits {
            image_short_edge: 1024,
            ..REFERENCE_LIMITS
        };
        assert_eq!(
            read(json!({}), &smaller).map(|sizing| sizing.image_short_edge),
            Ok(1024)
        );
        assert!(read(json!({"reference_image_short_edge": 2048}), &smaller).is_err());
        // A denoiser that takes no image reference sizes none.
        assert!(read(json!({}), &ReferenceLimits::default()).is_ok());
    }

    /// Soundtracks truncate at their native rate, resample to 32 kHz and pad to whole 800-sample
    /// latents.
    #[test]
    fn audio_clips_match_the_reference() {
        for case in vectors()["audio_cases"].as_array().unwrap() {
            let clip = audio_clip(
                &audio_facts(case),
                case["start_seconds"].as_f64().unwrap(),
                case["num_frames"].as_u64().unwrap() as u32,
            )
            .unwrap();
            assert_eq!(audio_json(&clip), case["clip"], "{case}");
        }
    }

    /// Whole requests resolve to the reference's canvas, frame counts, condition preparation,
    /// vision grids and rows.
    #[test]
    fn requests_match_the_reference() {
        for case in vectors()["requests"].as_array().unwrap() {
            let Some(expected) = case.get("expected") else {
                continue;
            };
            let name = &case["name"];
            let generation = Request::parse(case).prepare().unwrap();
            assert_eq!(generation.canvas, size(&expected["canvas"]), "{name}");
            assert_eq!(
                u64::from(generation.num_frames),
                expected["num_frames"],
                "{name}"
            );
            let video_rows: u32 = generation
                .conditions
                .iter()
                .flat_map(|condition| condition.latent_units(generation.canvas))
                .sum();
            let audio_rows: u32 = generation.conditions.iter().map(|c| c.audio_rows).sum();
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
            let dense = ConditionPacking::dense(u32::MAX);
            assert_eq!(
                u64::from(generation.condition_rows(&dense)),
                expected["condition_video_rows"].as_u64().unwrap()
                    + expected["condition_audio_rows"].as_u64().unwrap(),
                "{name}"
            );
            let conditions: Vec<Value> = generation
                .conditions
                .iter()
                .map(|condition| condition_json(condition, generation.canvas))
                .collect();
            assert_eq!(Value::from(conditions), expected["conditions"], "{name}");
        }
    }

    /// Each refused request names the public parameter the reference blames, whether preparation
    /// or the prompt refuses it.
    #[test]
    fn refused_requests_name_the_field() {
        for case in vectors()["requests"].as_array().unwrap() {
            let Some(error) = case.get("error") else {
                continue;
            };
            let request = Request::parse(case);
            let refusal = match request.prepare() {
                Ok(generation) => present(&CharacterTokenizer, &generation, request.prompt())
                    .expect_err("the request was accepted"),
                Err(refusal) => refusal,
            };
            assert_eq!(param(refusal), error["param"], "{}", case["name"]);
        }
    }

    const LANDSCAPE: Canvas = Canvas {
        width: 1344,
        height: 768,
    };

    /// A deployment serves only the canvases its workers prepare; another size is refused before
    /// any media is needed.
    #[test]
    fn prepared_canvases_restrict_the_size() {
        let processor = serving(vec![served(768, "16:9", 1344, 768)]);
        let generation = processor.prepare(&text(None, 5.0), &[]).unwrap();
        assert_eq!(generation.canvas, LANDSCAPE);

        let portrait = Canvas {
            width: 768,
            height: 1344,
        };
        let error = processor
            .check_request(&text(Some(portrait), 5.0))
            .unwrap_err();
        assert_eq!(param(error), "size");
    }

    /// A deployment that declares 480p canvases serves them: a stated size is the canvas the
    /// deployment declares, a request without one takes the deployment's first canvas, and a size
    /// it did not prepare is refused before any media is needed.
    #[test]
    fn declared_canvases_are_the_sizes_served() {
        let canvas = |width, height| Canvas { width, height };
        let processor = serving(vec![
            served(768, "16:9", 1344, 768),
            served(480, "21:9", 992, 416),
            served(480, "16:9", 832, 480),
        ]);
        for (size, expected) in [
            (Some(canvas(992, 416)), canvas(992, 416)),
            (Some(canvas(832, 480)), canvas(832, 480)),
            (None, LANDSCAPE),
        ] {
            let generation = processor.prepare(&text(size, 5.0), &[]).unwrap();
            assert_eq!(generation.canvas, expected, "{size:?}");
        }
        for size in [canvas(1280, 720), canvas(768, 768)] {
            let error = processor.check_request(&text(Some(size), 5.0)).unwrap_err();
            assert!(error.to_string().contains("is not served"), "{error}");
            assert_eq!(param(error), "size");
        }

        // The first canvas a deployment names is its default, whatever its ratio.
        let wide_first = serving(vec![served(480, "21:9", 992, 416)]);
        let generation = wide_first.prepare(&text(None, 5.0), &[]).unwrap();
        assert_eq!(generation.canvas, canvas(992, 416));
    }

    /// An `fl2va` request without a size takes the canvas its first keyframe sets through the
    /// canvas rule, at its 768-pixel short edge, but only onto a canvas the deployment declares.
    #[test]
    fn a_keyframe_canvas_lands_on_a_declared_canvas() {
        let processor = serving(vec![served(768, "16:9", 1344, 768)]);
        let first = Reference {
            kind: ReferenceKind::Image,
            frame_index: Some(0),
            start_time_seconds: None,
        };
        let keyframed = request(None, 5.0, vec![first]);
        assert!(processor.check_request(&keyframed).is_ok());
        let keyframe = |width, height| [MediaMetadata::Image(ImageMetadata { width, height })];
        // A 7:4 keyframe at a 768-pixel short edge is 1344x768, the declared canvas.
        let generation = processor.prepare(&keyframed, &keyframe(1400, 800)).unwrap();
        assert_eq!(generation.canvas, LANDSCAPE);
        // A 4:3 keyframe sets 1024x768, which the deployment does not serve.
        let error = processor
            .prepare(&keyframed, &keyframe(1024, 768))
            .unwrap_err();
        assert_eq!(param(error), "size");
    }

    /// A stated frame count is served when the deployment's own durations generate it: a 5-second
    /// deployment serves the 124 frames 5 seconds align to, though they last 5.17 seconds, and no
    /// longer count.
    #[test]
    fn a_frame_count_of_the_served_durations_is_served() {
        let mut processor = processor();
        processor.max_seconds = 5.0;
        let frames = |count| VideoSpec {
            length: Some(Length::Frames(count)),
            ..text(None, 5.0)
        };

        let generation = processor.prepare(&frames(124), &[]).unwrap();
        assert_eq!(generation.num_frames, 124);
        assert_eq!(
            generation.num_frames,
            processor.prepare(&text(None, 5.0), &[]).unwrap().num_frames,
            "a frame count and the duration that generates it resolve alike"
        );
        // The fewest frames 4 seconds generate, and none fewer or more.
        assert!(processor.check_request(&frames(107)).is_ok());
        for count in [90, 141] {
            let error = processor.check_request(&frames(count)).unwrap_err();
            assert_eq!(param(error), "num_frames", "{count}");
        }
    }

    /// A request that states no length takes it from its one soundtrack: one whose references
    /// carry no audio, or two audio tracks, is refused before any media is fetched, and a video
    /// whose soundtrack is unknown before its media is probed may set it.
    #[test]
    fn a_length_is_stated_or_follows_one_soundtrack() {
        let processor = processor();
        let unstated = |references: Vec<Reference>| VideoSpec {
            length: None,
            ..request(None, 5.0, references)
        };
        let (image, video, audio) = (
            reference_of(ReferenceKind::Image),
            reference_of(ReferenceKind::Video),
            reference_of(ReferenceKind::Audio),
        );
        for references in [vec![image], vec![image, audio, audio], Vec::new()] {
            let error = processor.check_request(&unstated(references)).unwrap_err();
            assert_eq!(param(error), "seconds");
        }
        assert!(
            processor
                .check_request(&unstated(vec![image, video]))
                .is_ok()
        );
        assert!(
            processor
                .check_request(&unstated(vec![image, audio]))
                .is_ok()
        );
    }

    /// The served maximum duration narrows the model's own range.
    #[test]
    fn max_seconds_bounds_the_duration() {
        let mut processor = processor();
        processor.max_seconds = 8.0;
        let error = processor.check_request(&text(None, 10.0)).unwrap_err();
        assert_eq!(param(error), "seconds");
    }

    /// A `ref2va` request carries at most the references its denoiser declares, of each kind and
    /// in all, and each refusal names what it bounds.
    #[test]
    fn reference_counts_are_bounded_by_the_declared_limits() {
        let refusal = |processor: &H3Processor, references: Vec<Reference>| match processor
            .check_request(&request(None, 5.0, references))
        {
            Err(InputError::Invalid { param, message }) => {
                assert_eq!(param, "input_references");
                message
            }
            other => panic!("the references were not refused: {other:?}"),
        };
        let (image, video, audio) = (
            reference_of(ReferenceKind::Image),
            reference_of(ReferenceKind::Video),
            reference_of(ReferenceKind::Audio),
        );

        let released = processor();
        assert_eq!(
            refusal(&released, vec![image; 10]),
            "at most 9 image references, got 10"
        );
        let mut every_kind = vec![image; 9];
        every_kind.extend([video; 3]);
        every_kind.push(audio);
        assert_eq!(
            refusal(&released, every_kind),
            "at most 12 references, got 13"
        );

        let mut narrow = processor();
        narrow.denoiser.references = ReferenceLimits {
            images: 2,
            videos: 1,
            audios: 1,
            total: 3,
            ..REFERENCE_LIMITS
        };
        assert!(
            narrow
                .check_request(&request(None, 5.0, vec![image, image, video]))
                .is_ok()
        );
        for (references, bound) in [
            (vec![image; 3], "at most 2 image references"),
            (vec![image, video, video], "at most 1 video references"),
            (vec![image, audio, audio], "at most 1 audio references"),
            (vec![image, image, video, audio], "at most 3 references"),
        ] {
            assert!(refusal(&narrow, references).contains(bound), "{bound}");
        }
    }

    /// A task outside the placed denoiser's set is refused first, on the references that select
    /// it.
    #[test]
    fn only_served_tasks_are_accepted() {
        let mut processor = processor();
        processor.denoiser.tasks = vec![VideoTask::ReferenceToVideoAudio];
        let error = processor.check_request(&text(None, 5.0)).unwrap_err();
        assert_eq!(param(error), "input_references");
    }
}
