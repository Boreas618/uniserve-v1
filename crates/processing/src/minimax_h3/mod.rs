//! MiniMax-H3's request processing: the request rules its requests are held to, how each condition
//! is prepared and presented to the conditioner, and the processor a deployment binds.
//!
//! As in the reference pipelines (sglang's MiniMax-H3 model-specific stages, vllm-omni's
//! `minimax_h3/preprocessing.py`), request preprocessing is a stage of its own beside the model's
//! numerics. This module states every request-level fact of the family once, and the model crate
//! reads them from here: the output clock ([`FPS`], [`AUDIO_SAMPLE_RATE`], [`AUDIO_HOP`]), the frame
//! rule ([`FRAMES`]) and duration bounds, the canvas rule ([`CANVAS_RULE`]) and the canvases it
//! names ([`named_canvases`]), the documented reference limits ([`REFERENCE_LIMITS`]), and the
//! arithmetic that counts a condition's denoiser rows ([`patch_rows`], [`track_rows`],
//! [`audio_latents`], [`condition_rows`]).
//!
//! - [`rules`] resolves a request into what it generates and how every condition is prepared;
//! - [`prompt`] lists and tokenizes what the conditioner reads;
//! - [`processor`] is the deployment's [`crate::VideoProcessor`], bound from the description the
//!   denoiser's workers report ([`DenoiserInfo`]).

pub mod processor;
pub mod prompt;
pub mod rules;
pub mod wire;

use std::num::NonZeroU32;

use scheduling::processing::video::Canvas;
use serde::{Deserialize, Serialize};

use crate::canvas::{CanvasError, CanvasRule};
use crate::frames::FrameClips;
use crate::qwen_vl::ProcessorConfig;
use crate::video::{VideoBinding, VideoFamily, VideoProcessor};

pub use processor::H3Processor;
pub use rules::{
    AudioClip, Generation, Preparation, PreparedCondition, ReferenceSizing, VideoClip, Vision,
    VisualRows, condition_rows, condition_tiles,
};

/// Frames per second of every generated video: the clock durations and reference videos are
/// counted on.
pub const FPS: u32 = 24;

/// How the video VAE divides a video's frames into clips: `17n + 5` frames, five latent frames a
/// clip and two for the 5-frame tail.
pub const FRAMES: FrameClips = FrameClips {
    min_frames: 22,
    clip_frames: 17,
    clip_latent_frames: 5,
    tail_latent_frames: 2,
};

/// The shortest output a request may ask for, in seconds.
pub const MIN_SECONDS: f64 = 4.0;

/// The longest output a request may ask for, in seconds: the most a deployment may serve.
pub const MAX_SECONDS: f64 = 15.0;

/// The sizes a deployment prepares when it names none, `WxH` in pixels: the landscape and portrait
/// canvases of the released checkpoint's resolution class (`CANVAS_RULE`, a 768-pixel short edge
/// and `768 * 1344` pixels).
pub const DEFAULT_SIZES: [&str; 2] = ["1344x768", "768x1344"];

/// Samples per second of the generated audio track; every soundtrack is resampled to it.
pub const AUDIO_SAMPLE_RATE: u32 = 32_000;

/// Audio latent steps per second.
pub const AUDIO_LATENTS_PER_SECOND: u32 = 40;

/// Samples one audio latent step covers.
pub const AUDIO_HOP: u32 = AUDIO_SAMPLE_RATE / AUDIO_LATENTS_PER_SECOND;

/// Channels of every audio track: stereo, packed channel-major.
pub const AUDIO_CHANNELS: u32 = 2;

/// The video VAE's spatial compression: one latent cell per `16 x 16` pixels.
pub const LATENT_COMPRESSION: u32 = 16;

/// The transformer's patch side in latent cells: one denoiser row per `2 x 2` patch of a latent
/// frame.
pub const PATCH_SIDE: u32 = 2;

/// Pixels along one side of a denoiser patch: the VAE's spatial compression times the
/// transformer's patch side, 32. Every canvas side is a multiple of it.
pub const PATCH_PIXELS: u32 = LATENT_COMPRESSION * PATCH_SIDE;

/// The canvas rule H3 generates by: a 768-pixel short edge, an area cap of `768 * 1344` pixels,
/// aspects from 1:4 to 4:1, and sides on the denoiser patch's grid ([`PATCH_PIXELS`]), so a side
/// survives the VAE's spatial compression on a whole number of transformer patches.
pub const CANVAS_RULE: CanvasRule = CanvasRule {
    short_edge: 768,
    max_pixels: 768 * 1344,
    multiple: PATCH_PIXELS,
    narrowest: (1, 4),
    widest: (4, 1),
};

/// The aspect ratios a request names a canvas by besides `auto`, landscape first. Every H3
/// checkpoint generates the canvas rule's canvas of each, and a deployment serves a subset.
pub const NAMED_ASPECT_RATIOS: [(u32, u32); 6] =
    [(21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16)];

/// The references the released checkpoint documents for a `ref2va` request: nine images, three
/// videos, three audio tracks and twelve in all, images encoded at a short edge of at most 2048
/// pixels.
pub const REFERENCE_LIMITS: ReferenceLimits = ReferenceLimits {
    images: 9,
    videos: 3,
    audios: 3,
    total: 12,
    image_short_edge: 2048,
};

/// The pipeline stage that holds the conditioner's Qwen3-VL processor.
const PROCESSOR_STAGE: &str = "processor";

/// The checkpoint files the processor reads, as `(stage, file)` of the pipeline: the conditioner's
/// Qwen3-VL processor files ([`ProcessorConfig::FILES`]), which fix how condition media is
/// patched, in the stage that holds them.
pub const PROCESSOR_FILES: [(&str, &str); 2] = [
    (PROCESSOR_STAGE, ProcessorConfig::FILES[0]),
    (PROCESSOR_STAGE, ProcessorConfig::FILES[1]),
];

/// The canvases every H3 checkpoint generates: the canvas rule's canvas of each named aspect ratio,
/// named by the rule's short edge and the ratio, in [`NAMED_ASPECT_RATIOS`]' order (1536x672 for
/// 21:9 to 768x1344 for 9:16). A deployment serves a subset of them.
///
/// # Errors
///
/// Fails when the rule refuses a named ratio, which a rule that states the named ratios' range does
/// not.
pub fn named_canvases() -> Result<Vec<ServedCanvas>, CanvasError> {
    NAMED_ASPECT_RATIOS
        .iter()
        .map(|&(width, height)| {
            Ok(ServedCanvas {
                short_edge: CANVAS_RULE.short_edge,
                aspect_ratio: format!("{width}:{height}"),
                canvas: CANVAS_RULE.canvas(f64::from(width), f64::from(height))?,
            })
        })
        .collect()
}

/// Denoiser rows of one latent frame of `canvas`: one per `2 x 2` patch of its `16x`-compressed
/// latent raster.
pub const fn patch_rows(canvas: Canvas) -> u32 {
    (canvas.height / PATCH_PIXELS) * (canvas.width / PATCH_PIXELS)
}

/// Denoiser rows of a stereo track of `latents` latent steps: one per channel per step,
/// channel-major. A generated track and a condition's alike.
pub const fn track_rows(latents: u32) -> u32 {
    AUDIO_CHANNELS * latents
}

/// Latent steps a track of `samples` samples encodes to: one per [`AUDIO_HOP`], the last one
/// padded with zeros (a ceiling, as the reference's `unpack_samples` right-pads the track).
pub const fn audio_latents(samples: u64) -> u64 {
    samples.div_ceil(AUDIO_HOP as u64)
}

/// A video-and-audio generation task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VideoTask {
    /// Text to video and audio (`t2va`).
    #[serde(rename = "t2va")]
    TextToVideoAudio,
    /// First and/or last keyframe to video and audio (`fl2va`).
    #[serde(rename = "fl2va")]
    FirstLastToVideoAudio,
    /// Image, video and audio references to video and audio (`ref2va`).
    #[serde(rename = "ref2va")]
    ReferenceToVideoAudio,
}

impl VideoTask {
    /// Every task, in canonical order.
    pub const ALL: [Self; 3] = [
        Self::TextToVideoAudio,
        Self::FirstLastToVideoAudio,
        Self::ReferenceToVideoAudio,
    ];

    /// The task's name, as the checkpoint and the denoiser's description state it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TextToVideoAudio => "t2va",
            Self::FirstLastToVideoAudio => "fl2va",
            Self::ReferenceToVideoAudio => "ref2va",
        }
    }

    /// The task named `name`, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|task| task.as_str() == name)
    }

    /// The task a request of `first_frames` and `last_frames` keyframes and `references` references
    /// runs: `ref2va` with any reference, `fl2va` with keyframes alone, `t2va` with neither.
    pub const fn of(first_frames: u32, last_frames: u32, references: u32) -> Self {
        if references > 0 {
            Self::ReferenceToVideoAudio
        } else if first_frames + last_frames > 0 {
            Self::FirstLastToVideoAudio
        } else {
            Self::TextToVideoAudio
        }
    }

    /// Whether the task's requests carry conditions.
    pub const fn conditioned(self) -> bool {
        !matches!(self, Self::TextToVideoAudio)
    }
}

/// A canvas a deployment serves, with the request fields that select it.
///
/// The model names each canvas it generates by the pair a request selects it with; the rank
/// declares the canvases its deployment prepares in its handshake, in the order the operator named
/// them, and the request rules resolve a target against exactly those.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ServedCanvas {
    /// The resolution class, in pixels, such as 768: the canvas's short edge as the model names
    /// it.
    pub short_edge: u32,
    /// The display ratio as `W:H`, such as `16:9`.
    pub aspect_ratio: String,
    /// The raster this pair selects.
    pub canvas: Canvas,
}

/// The references one `ref2va` request may carry, and the largest short edge an image reference is
/// encoded at, as a video denoiser declares them.
///
/// `total` bounds the references of every kind together, keyframes not counted; a video's
/// soundtrack counts with the video, and `audios` bounds the audio references that are not one. A
/// denoiser that takes no references declares zeros.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ReferenceLimits {
    /// Image references.
    pub images: u32,
    /// Video references, with or without soundtrack.
    pub videos: u32,
    /// Audio references that are not a video's soundtrack.
    pub audios: u32,
    /// References of every kind.
    pub total: u32,
    /// Short edge, in pixels, the denoiser encodes an image reference at, at most.
    pub image_short_edge: u32,
}

/// How a denoiser lays a request's conditions into its rows, and how many it holds.
///
/// A sparse denoiser attends whole tiles, so each condition's audio and each condition's pixels
/// fill tiles of their own, and a reference video fills the tiles of its token grid. A dense
/// denoiser packs row for row: its tile is one row and one token ([`Self::dense`]). What a request
/// takes is therefore the denoiser's tiling and not its rows' sum ([`condition_rows`]), and it
/// decides both whether the deployment serves the request and which of its graphs does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditionPacking {
    /// Rows one request's conditions may take: the deployment's capacity, zero for a denoiser
    /// that takes none.
    pub capacity: u32,
    /// Rows of one tile.
    pub tile_rows: NonZeroU32,
    /// Tokens of one video tile along time, height and width.
    pub video_tile: [NonZeroU32; 3],
}

impl ConditionPacking {
    /// A dense denoiser's packing of `capacity` rows: row for row, one token a tile.
    pub const fn dense(capacity: u32) -> Self {
        Self {
            capacity,
            tile_rows: NonZeroU32::MIN,
            video_tile: [NonZeroU32::MIN; 3],
        }
    }

    /// A sparse denoiser's packing of `capacity` rows into tiles of `tile_rows` rows, a video's
    /// token grid into tiles of `video_tile` tokens; `None` when a tile is empty.
    pub fn tiled(capacity: u32, tile_rows: u32, video_tile: [u32; 3]) -> Option<Self> {
        let [time, height, width] = video_tile;
        Some(Self {
            capacity,
            tile_rows: NonZeroU32::new(tile_rows)?,
            video_tile: [
                NonZeroU32::new(time)?,
                NonZeroU32::new(height)?,
                NonZeroU32::new(width)?,
            ],
        })
    }
}

/// What the video denoiser a deployment places serves, as its workers report it to request
/// processing.
///
/// The denoiser fixes its schedule, and with it every request's step count; the tasks, canvases and
/// reference limits bound what a request may ask for, and its condition packing what its
/// conditions may take. The engine carries it as the encoding of the request interface's
/// `VideoDenoiser` message ([`Self::encode`]), which it does not read. The output clock, the frame
/// rule and the canvas rule are the family's ([`FPS`], [`FRAMES`], [`CANVAS_RULE`]), not the
/// deployment's.
#[derive(Debug, Clone, PartialEq)]
pub struct DenoiserInfo {
    /// The tasks the denoiser serves, in canonical order.
    pub tasks: Vec<VideoTask>,
    /// Sigma points of the fixed schedule, the clean endpoint included: one more than the network
    /// evaluations ([`Self::denoise_steps`]).
    pub schedule_points: u32,
    /// Shift of the video schedule.
    pub video_shift: f64,
    /// Shift of the audio schedule.
    pub audio_shift: f64,
    /// The only canvases the deployment serves, each with the `(short_edge, aspect_ratio)` its
    /// model names it by, in the order the deployment names them; the first is the canvas of a
    /// request that states no size.
    pub canvases: Vec<ServedCanvas>,
    /// The references one request may carry and the largest short edge an image reference is
    /// encoded at; zeros for a denoiser that takes none.
    pub references: ReferenceLimits,
    /// How the denoiser packs a request's conditions and how many rows they may take.
    pub condition_packing: ConditionPacking,
}

/// Why bytes are no description of a MiniMax-H3 denoiser.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DescriptionError {
    /// The bytes are no `VideoDenoiser` message.
    #[error("the description does not decode: {0}")]
    Undecodable(String),
    /// The message leaves out a part every description states.
    #[error("the description states no {0}")]
    Missing(&'static str),
    /// A task this family does not have.
    #[error("the description names the unknown task {0:?}")]
    Task(String),
    /// A tile of no rows or tokens.
    #[error("the description's condition packing has an empty tile")]
    Tile,
    /// The description breaks a rule of [`DenoiserInfo::validate`].
    #[error("{0}")]
    Invalid(&'static str),
}

impl DenoiserInfo {
    /// The network evaluations of the fixed schedule: one fewer than its sigma points, and the step
    /// count of every request. A validated description has at least two points.
    pub fn denoise_steps(&self) -> u32 {
        self.schedule_points - 1
    }

    /// The encoding the engine carries: the description's `VideoDenoiser` message.
    pub fn encode(&self) -> Vec<u8> {
        let packing = &self.condition_packing;
        let canvas = |canvas: Canvas| wire::Canvas {
            width: canvas.width,
            height: canvas.height,
        };
        let references = &self.references;
        prost::Message::encode_to_vec(&wire::VideoDenoiser {
            tasks: self
                .tasks
                .iter()
                .map(|task| task.as_str().to_owned())
                .collect(),
            schedule_points: self.schedule_points,
            video_shift: self.video_shift,
            audio_shift: self.audio_shift,
            canvases: self
                .canvases
                .iter()
                .map(|served| wire::ServedCanvas {
                    short_edge: served.short_edge,
                    aspect_ratio: served.aspect_ratio.clone(),
                    canvas: Some(canvas(served.canvas)),
                })
                .collect(),
            references: Some(wire::ReferenceLimits {
                images: references.images,
                videos: references.videos,
                audios: references.audios,
                total: references.total,
                image_short_edge: references.image_short_edge,
            }),
            condition_packing: Some(wire::ConditionPacking {
                capacity: packing.capacity,
                tile_rows: packing.tile_rows.get(),
                video_tile: packing.video_tile.iter().map(|side| side.get()).collect(),
            }),
        })
    }

    /// Decodes [`Self::encode`]'s encoding and validates it ([`Self::validate`]).
    ///
    /// # Errors
    ///
    /// Fails when `bytes` are no `VideoDenoiser` message, leave out a part, name a task the
    /// family does not have, state an empty tile or break a rule of [`Self::validate`].
    pub fn decode(bytes: &[u8]) -> Result<Self, DescriptionError> {
        let message: wire::VideoDenoiser = prost::Message::decode(bytes)
            .map_err(|error| DescriptionError::Undecodable(error.to_string()))?;
        let canvas = |canvas: Option<wire::Canvas>| {
            canvas
                .map(|canvas| Canvas {
                    width: canvas.width,
                    height: canvas.height,
                })
                .ok_or(DescriptionError::Missing("canvas raster"))
        };
        let references = message
            .references
            .ok_or(DescriptionError::Missing("reference limits"))?;
        let packing = message
            .condition_packing
            .ok_or(DescriptionError::Missing("condition packing"))?;
        let [time, height, width] = <[u32; 3]>::try_from(packing.video_tile)
            .map_err(|_| DescriptionError::Missing("three video tile sides"))?;

        let description = Self {
            tasks: message
                .tasks
                .into_iter()
                .map(|name| VideoTask::from_name(&name).ok_or(DescriptionError::Task(name)))
                .collect::<Result<_, _>>()?,
            schedule_points: message.schedule_points,
            video_shift: message.video_shift,
            audio_shift: message.audio_shift,
            canvases: message
                .canvases
                .into_iter()
                .map(|served| {
                    Ok(ServedCanvas {
                        short_edge: served.short_edge,
                        aspect_ratio: served.aspect_ratio,
                        canvas: canvas(served.canvas)?,
                    })
                })
                .collect::<Result<_, DescriptionError>>()?,
            references: ReferenceLimits {
                images: references.images,
                videos: references.videos,
                audios: references.audios,
                total: references.total,
                image_short_edge: references.image_short_edge,
            },
            condition_packing: ConditionPacking::tiled(
                packing.capacity,
                packing.tile_rows,
                [time, height, width],
            )
            .ok_or(DescriptionError::Tile)?,
        };
        description.validate().map_err(DescriptionError::Invalid)?;
        Ok(description)
    }

    /// Checks that tasks are named once each, the schedule has an endpoint and positive finite
    /// shifts, the reference limits bound each kind within the total with an image edge where
    /// images are taken, and at least one canvas is served, each a positive raster named by its own
    /// `(short_edge, aspect_ratio)`.
    ///
    /// # Errors
    ///
    /// The first rule the report breaks.
    pub fn validate(&self) -> Result<(), &'static str> {
        let mut names = std::collections::HashSet::new();
        if self.tasks.is_empty() || !self.tasks.iter().all(|task| names.insert(*task)) {
            return Err("a video denoiser names each of its tasks once");
        }
        if self.schedule_points < 2
            || ![self.video_shift, self.audio_shift]
                .iter()
                .all(|shift| shift.is_finite() && *shift > 0.0)
        {
            return Err("a video denoiser schedule needs points and positive shifts");
        }

        let references = &self.references;
        if references
            .images
            .max(references.videos)
            .max(references.audios)
            > references.total
            || (references.images > 0 && references.image_short_edge == 0)
        {
            return Err(
                "a video denoiser declares reference limits beyond their total or images without an edge",
            );
        }

        if self.canvases.is_empty() {
            return Err("a video denoiser prepared no canvas");
        }
        if !self.canvases.iter().all(|served| {
            served.short_edge > 0
                && !served.aspect_ratio.is_empty()
                && served.canvas.width > 0
                && served.canvas.height > 0
        }) {
            return Err("a video denoiser declares an empty canvas");
        }
        let mut named = std::collections::HashSet::new();
        if !self
            .canvases
            .iter()
            .all(|served| named.insert((served.short_edge, served.aspect_ratio.as_str())))
        {
            return Err("a video denoiser declares a canvas name twice");
        }
        Ok(())
    }
}

/// MiniMax-H3 as request processing serves it: the model crate's family states
/// `Serving::Video(&processing::minimax_h3::H3Family)`.
pub struct H3Family;

/// The pipeline class a MiniMax-H3 checkpoint declares (`modular_model_index.json`'s
/// `_class_name`).
pub const PIPELINE_CLASS: &str = "MiniMaxH3ModularPipeline";

/// Where a FastH3 export of MiniMax-H3 pins the MiniMax-H3 release it continues: the inference
/// description FastVideo publishes at the export's root, whose `base_model_revision` names the
/// release (`hf://MiniMaxAI/MiniMax-H3@<commit>`) whose stage folders the export omits.
const BASE_PIN: crate::BasePin = crate::BasePin {
    file: "fastvideo_inference.json",
    field: "base_model_revision",
};

/// The MiniMax-H3 family as request processing serves it: a diffusers modular pipeline, which every
/// release and export publishes with its `modular_model_index.json`, served through its video
/// processor ([`H3Family`]).
pub const FAMILY: crate::Family = crate::Family {
    id: "minimax_h3",
    root: crate::RootFile::ModularIndex,
    classes: &[PIPELINE_CLASS],
    serving: crate::Serving::Video(&H3Family),
};

impl VideoFamily for H3Family {
    fn processor_files(&self) -> &'static [(&'static str, &'static str)] {
        &PROCESSOR_FILES
    }

    fn default_sizes(&self) -> &'static [&'static str] {
        &DEFAULT_SIZES
    }

    fn default_max_seconds(&self) -> f64 {
        MAX_SECONDS
    }

    fn processing_parts(&self) -> &'static [&'static str] {
        // The conditioner's tokenizer and processor, and the video and audio schedules.
        &["tokenizer", "processor", "scheduler", "audio_scheduler"]
    }

    fn base_pin(&self) -> Option<crate::BasePin> {
        Some(BASE_PIN)
    }

    fn bind(&self, binding: VideoBinding<'_>) -> Result<Box<dyn VideoProcessor>, String> {
        Ok(Box::new(H3Processor::bind(binding)?))
    }
}

/// The vectors the request rules are tested against, generated from the diffusers MiniMax-H3
/// pipeline and the checkpoint's Qwen3-VL processor and tokenizer (`vectors.json`, whose
/// `sources` state the revisions).
#[cfg(test)]
pub(crate) fn vectors() -> serde_json::Value {
    serde_json::from_str(include_str!("vectors.json")).expect("the request vectors are JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canvas rule matches the reference's on named, probed and out-of-range aspects.
    #[test]
    fn canvases_match_the_reference() {
        let vectors = vectors();
        for case in vectors["canvas_cases"].as_array().unwrap() {
            let aspect = &case["aspect"];
            let resolved =
                CANVAS_RULE.canvas(aspect[0].as_f64().unwrap(), aspect[1].as_f64().unwrap());
            if case.get("error").is_some() {
                assert!(resolved.is_err(), "{case}");
            } else {
                let canvas = &case["canvas"];
                assert_eq!(
                    resolved.unwrap(),
                    Canvas {
                        width: canvas[0].as_u64().unwrap() as u32,
                        height: canvas[1].as_u64().unwrap() as u32,
                    },
                    "{case}"
                );
            }
        }
    }

    /// Every named ratio has a canvas, from 1536x672 for 21:9 to 768x1344 for 9:16.
    #[test]
    fn every_named_ratio_names_a_canvas() {
        let canvases = named_canvases().unwrap();
        assert_eq!(canvases.len(), NAMED_ASPECT_RATIOS.len());
        assert_eq!(canvases[0].aspect_ratio, "21:9");
        assert_eq!(
            canvases[0].canvas,
            Canvas {
                width: 1536,
                height: 672
            }
        );
        assert_eq!(
            canvases[5].canvas,
            Canvas {
                width: 768,
                height: 1344
            }
        );
    }

    /// A latent frame of 1344x768 is 42 x 24 latent patches; a track pads its last latent.
    #[test]
    fn rows_follow_the_latent_raster_and_the_hop() {
        let canvas = Canvas {
            width: 1344,
            height: 768,
        };
        assert_eq!(patch_rows(canvas), 42 * 24);
        assert_eq!(audio_latents(800), 1);
        assert_eq!(audio_latents(801), 2);
        assert_eq!(track_rows(40), 80);
    }

    /// A description that names a task twice is refused, and a packing tile is never empty.
    #[test]
    fn a_description_names_each_task_once_and_packs_in_tiles() {
        let report = DenoiserInfo {
            tasks: vec![VideoTask::TextToVideoAudio, VideoTask::TextToVideoAudio],
            schedule_points: 9,
            video_shift: 12.0,
            audio_shift: 3.0,
            canvases: vec![ServedCanvas {
                short_edge: 768,
                aspect_ratio: "16:9".to_owned(),
                canvas: Canvas {
                    width: 1344,
                    height: 768,
                },
            }],
            references: ReferenceLimits::default(),
            condition_packing: ConditionPacking::dense(0),
        };
        assert!(report.validate().is_err());
        assert_eq!(ConditionPacking::tiled(1024, 0, [4, 4, 8]), None);
        assert_eq!(ConditionPacking::tiled(1024, 128, [4, 0, 8]), None);
    }
}
