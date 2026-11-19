//! The video side of the processing interface: what a video family's request rules read
//! ([`VideoSpec`], the probed [`MediaMetadata`] of its references) and what they process a request
//! into ([`VideoInput`]), with the vocabulary of the media every condition is decoded into
//! ([`ConditionMedia`]), which the workers read as `proto::video` re-exports it.
//!
//! A video family declares its pipeline ([`Pipeline`]): the files its processor reads, the parts
//! whose files decide how a request is processed, and the sizes and length a deployment serves
//! when its configuration states none. It binds once its workers described the placed denoiser
//! ([`Binding`]) and reports what it serves ([`Facts`]).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A video raster in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Canvas {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// How a condition relates to the generated video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionRole {
    /// A keyframe anchoring the first generated frame.
    FirstFrame,
    /// A keyframe anchoring the last generated frame.
    LastFrame,
    /// A reference conditioning the generation as a whole.
    Reference,
}

impl ConditionRole {
    /// Whether the condition anchors a generated frame.
    pub const fn is_keyframe(self) -> bool {
        matches!(self, Self::FirstFrame | Self::LastFrame)
    }
}

/// A decoded still image put on a raster: resized with LANCZOS to `resized`,
/// then the `size` window whose top-left corner is (`left`, `top`) kept.
///
/// A stretched image is resized straight to its raster (`resized == size`,
/// no offset); a cover-cropped keyframe is resized past the canvas and
/// centred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageFit {
    /// Size the decoded image is resized to.
    pub resized: Canvas,
    /// Left edge of the kept window in the resized image.
    pub left: u32,
    /// Top edge of the kept window in the resized image.
    pub top: u32,
    /// The kept raster, which the video encoder encodes.
    pub size: Canvas,
}

/// The frames of a reference video the media reader produces.
///
/// The video is resampled to the output frame rate and scaled onto `canvas`
/// in one decoding pass; its first `start_frame` frames are skipped and the
/// next `frames` kept. The conditioner samples the kept frames; the video encoder encodes
/// their leading `vae_frames`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VideoClip {
    /// The canvas of the video's own display aspect.
    pub canvas: Canvas,
    /// Frames of the output-rate timeline skipped at the start.
    pub start_frame: u32,
    /// Frames of the output-rate timeline kept after the start offset.
    pub frames: u32,
    /// The leading kept frames the video encoder encodes.
    pub vae_frames: u32,
}

/// The samples of a soundtrack or audio reference the media reader produces.
///
/// The track is decoded at its native rate, its first `start_sample` samples
/// skipped and the next `source_samples` kept, then resampled once to the
/// model's audio rate, giving `samples` stereo samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioClip {
    /// The track's native sample rate.
    pub sample_rate: u32,
    /// Native samples skipped at the start.
    pub start_sample: u64,
    /// Native samples kept after the start offset.
    pub source_samples: u64,
    /// Samples after resampling to the model's audio rate.
    pub samples: u64,
}

/// What the media reader decodes a condition's bytes into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionMedia {
    /// A keyframe or an image reference: one frame.
    Image(ImageFit),
    /// A reference video and, when the request conditions on it, its
    /// soundtrack.
    Video {
        /// The frames.
        clip: VideoClip,
        /// The soundtrack.
        soundtrack: Option<AudioClip>,
    },
    /// An audio reference.
    Audio(AudioClip),
}

impl ConditionMedia {
    /// The audio track the condition carries, if any.
    pub const fn audio(&self) -> Option<&AudioClip> {
        match self {
            Self::Image(_) => None,
            Self::Video { soundtrack, .. } => soundtrack.as_ref(),
            Self::Audio(clip) => Some(clip),
        }
    }

    /// The frame count and raster of the pixels the video encoder encodes,
    /// if the condition has any.
    pub const fn pixels(&self) -> Option<(u32, Canvas)> {
        match self {
            Self::Image(fit) => Some((1, fit.size)),
            Self::Video { clip, .. } => Some((clip.vae_frames, clip.canvas)),
            Self::Audio(_) => None,
        }
    }
}

/// A vision patch grid: `t` temporal patches of `h` by `w` spatial patches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VisionGrid {
    /// Temporal patches; 1 for an image.
    pub t: u32,
    /// Patch rows.
    pub h: u32,
    /// Patch columns.
    pub w: u32,
}

impl VisionGrid {
    /// Patches of the grid.
    pub const fn patches(&self) -> u64 {
        self.t as u64 * self.h as u64 * self.w as u64
    }
}

/// What the conditioner reads of a visual condition.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConditionVision {
    /// The patch grid the condition's frames are resized to.
    pub grid: VisionGrid,
    /// Vision tokens the condition occupies in the presentation, which is
    /// the count of its vision placeholders.
    pub tokens: u32,
    /// The kept frames of a reference video the conditioner samples, in
    /// order; empty for an image.
    pub frame_indices: Vec<u32>,
}

/// A video generation request, as the public API states it to any video model.
///
/// Every field but the prompt is optional: the model supplies its own default or refuses a request
/// that leaves out what it needs. A reference with a frame index anchors that generated frame; one
/// without conditions the generation as a whole.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoSpec {
    /// The prompt.
    pub prompt: String,
    /// What the generation steers away from.
    pub negative_prompt: Option<String>,
    /// The output raster in pixels.
    pub size: Option<Canvas>,
    /// The output's length.
    pub length: Option<Length>,
    /// The output's frame rate.
    pub fps: Option<u32>,
    /// The noise seed: the request's own, or the one its server draws when it states none, as it
    /// does for every request.
    pub seed: u64,
    /// Denoiser evaluations of the schedule.
    pub num_inference_steps: Option<u32>,
    /// Classifier-free guidance scale.
    pub guidance_scale: Option<f64>,
    /// The request's reference media, in request order.
    pub references: Vec<Reference>,
    /// Options the model defines and validates; an option it does not define is refused.
    pub extra_params: Map<String, Value>,
}

/// How long an output is.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Length {
    /// An exact frame count, which must be one the model generates.
    Frames(u32),
    /// A duration in seconds, which the model aligns to a frame count it generates.
    Seconds(f64),
}

/// One reference medium of a request.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Reference {
    /// What the medium is.
    pub kind: ReferenceKind,
    /// The generated frame it anchors, counted from zero, a negative index from the last;
    /// `None` for a reference that conditions the generation as a whole.
    pub frame_index: Option<i64>,
    /// Where a video or audio reference starts, in seconds.
    pub start_time_seconds: Option<f64>,
}

/// The kind of a reference medium.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    /// A still image.
    Image,
    /// A video, with its soundtrack when it has one.
    Video,
    /// An audio file.
    Audio,
}

impl ReferenceKind {
    /// The kind as the public API names it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }

    /// The kind the public API names `name`.
    pub fn named(name: &str) -> Option<Self> {
        [Self::Image, Self::Video, Self::Audio]
            .into_iter()
            .find(|kind| kind.name() == name)
    }
}

/// A request a processor refuses, or a failure while processing a valid one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(rename_all = "snake_case")]
pub enum InputError {
    /// The request breaks a rule of the model; the same request fails again on retry.
    #[error("{param}: {message}")]
    Invalid {
        /// The request parameter at fault, as the public API names it (`size`,
        /// `input_references[2]`).
        param: String,
        /// What is wrong with it.
        message: String,
    },
    /// The request's conditions take more denoiser rows than the deployment serves; the same
    /// request fails again on this deployment, and one of a larger condition capacity serves it.
    #[error(
        "{param}: the references take {rows} denoiser rows, more than the {capacity} this \
         deployment serves"
    )]
    Capacity {
        /// The request parameter at fault, as the public API names it.
        param: String,
        /// Denoiser rows the request's conditions take.
        rows: u32,
        /// Rows the deployment serves.
        capacity: u32,
    },
    /// Processing valid input failed, for example because the tokenizer lacks a token the model
    /// presents.
    #[error("{0}")]
    Internal(String),
}

impl InputError {
    /// Refuses `param` with `message`.
    pub fn invalid(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Invalid {
            param: param.into(),
            message: message.into(),
        }
    }
}

/// A processed condition, before its media is published: everything the workers read of it but
/// where its bytes are.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionInput {
    /// Anchor or reference.
    pub role: ConditionRole,
    /// What the media reader decodes its bytes into.
    pub media: ConditionMedia,
    /// What the text encoder's vision tower reads of it.
    pub vision: Option<ConditionVision>,
    /// Denoiser rows each temporal unit of its pixels encodes to.
    pub latent_units: Vec<u32>,
    /// Denoiser audio rows of its track.
    pub audio_rows: u32,
}

/// A processed video request: what it generates and everything the engine and the workers read of
/// it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoInput {
    /// The output raster.
    pub size: Canvas,
    /// The frames it generates.
    pub num_frames: u32,
    /// The output's frame rate.
    pub fps: u32,
    /// The noise seed.
    pub seed: u64,
    /// The prompt as the text encoder encodes it, vision content included.
    pub prompt_token_ids: Vec<u32>,
    /// The prompt rows each block of vision content occupies, as `(first row, rows)`.
    pub vision_spans: Vec<(u32, u32)>,
    /// The conditions, one per reference in request order.
    pub conditions: Vec<ConditionInput>,
    /// The decode units of the output.
    pub video_units: u32,
    /// The denoiser rows the conditions take, by which the engine selects the smallest graph
    /// that holds them.
    pub condition_rows: u32,
}

/// The probed facts of one condition's media.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaMetadata {
    /// An image.
    Image(ImageMetadata),
    /// A video and its soundtrack.
    Video(VideoMetadata),
    /// An audio file.
    Audio(AudioMetadata),
}

/// An image as displayed, after its EXIF orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageMetadata {
    /// Displayed width in pixels.
    pub width: u32,
    /// Displayed height in pixels.
    pub height: u32,
}

/// A rational frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRate {
    /// Frames per `denominator` seconds.
    pub numerator: u32,
    /// Seconds per `numerator` frames.
    pub denominator: u32,
}

/// A video's display aspect and size, frame rate, frame count and soundtrack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// Width of the display aspect, in lowest terms. Only the ratio to `display_height` matters.
    pub display_width: u64,
    /// Height of the display aspect, in lowest terms.
    pub display_height: u64,
    /// Displayed width in pixels: the coded width scaled by the sample aspect ratio, rounded
    /// half to even, and turned with `height` by the display rotation.
    pub width: u32,
    /// Displayed height in pixels.
    pub height: u32,
    /// The stream's average frame rate.
    pub frame_rate: FrameRate,
    /// Frames a decoder outputs for the whole stream.
    pub frames: u64,
    /// The first audio stream, when the video has one.
    pub soundtrack: Option<AudioMetadata>,
}

/// A soundtrack's native sample rate and decoded sample count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Samples per second.
    pub sample_rate: u32,
    /// Samples per channel a decoder outputs.
    pub samples: u64,
}

/// One request to process (`video_process`): the request and the probed facts of its
/// references' media, in request order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Process {
    /// The request.
    pub spec: VideoSpec,
    /// Each reference's probed media.
    pub media: Vec<MediaMetadata>,
}

/// A video family's pipeline, as the daemon reads a checkpoint of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pipeline {
    /// The files the processor reads, each as `(stage, file)` of the pipeline.
    pub processor_files: Vec<(String, String)>,
    /// The pipeline's parts (stage directories) whose files decide how a request is processed
    /// beside the weights: a checkpoint whose files in any of them differ from the served one's
    /// is another deployment, not new weights of the served model.
    pub processing_parts: Vec<String>,
    /// Where the family's checkpoints pin the release whose stage folders they continue; `None`
    /// for a family whose checkpoints hold every stage they declare.
    pub base_pin: Option<BasePin>,
    /// The output sizes a deployment prepares when its configuration names none, each `WxH`.
    pub default_sizes: Vec<String>,
    /// The longest output a deployment admits when its configuration states none, in seconds.
    pub default_max_seconds: f64,
}

/// Where a pipeline checkpoint pins the release it continues: a JSON file at the checkpoint's root
/// and the field of it that names the release as `hf://<repository>@<revision>`.
///
/// A checkpoint that pins a release holds the stage folders it replaces and omits the others,
/// which the pinned release's snapshot supplies.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BasePin {
    /// The file at the checkpoint's root.
    pub file: String,
    /// The field of `file` that names the release.
    pub field: String,
}

/// What a deployment binds a video family's processor with, beside the processor files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Binding {
    /// The placed denoiser as the model's workers described it: the encoding of the family's
    /// description message, which the engine carries without reading it.
    pub description: Vec<u8>,
    /// The deployment's longest output, in seconds.
    pub max_seconds: f64,
}

/// What a bound video family serves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Facts {
    /// The model's part of `GET /v1/capabilities`: what a request may ask for.
    pub capabilities: Value,
    /// Denoiser evaluations of every request's schedule.
    pub denoise_steps: u32,
}
