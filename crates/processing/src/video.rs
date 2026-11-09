//! The video request contract: a model-agnostic request, the processor a video family binds, and
//! the plan it returns.

use proto::video::{
    Canvas, Condition, ConditionMedia, ConditionRole, ConditionVision, MediaLocator,
};
use serde_json::{Map, Value};

use crate::media::MediaMetadata;
use crate::vocabulary::Vocabulary;

/// A video generation request, as the public API states it to any video model.
///
/// Every field but the prompt is optional: the model supplies its own default or refuses a request
/// that leaves out what it needs. A reference with a frame index anchors that generated frame; one
/// without conditions the generation as a whole.
#[derive(Debug, Clone, PartialEq)]
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
    /// The noise seed.
    pub seed: Option<u64>,
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Length {
    /// An exact frame count, which must be one the model generates.
    Frames(u32),
    /// A duration in seconds, which the model aligns to a frame count it generates.
    Seconds(f64),
}

/// One reference medium of a request.
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// A request a processor refuses, or a failure while planning a valid one.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
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
    /// Planning valid input failed, for example because the tokenizer lacks a token the model
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

/// A planned condition, before its media is published: everything the workers read of it but
/// where its bytes are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionPlan {
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

impl ConditionPlan {
    /// The condition with its media published at `source`.
    pub fn at(self, source: MediaLocator) -> Condition {
        Condition {
            role: self.role,
            source,
            media: self.media,
            vision: self.vision,
            latent_units: self.latent_units,
            audio_rows: self.audio_rows,
        }
    }
}

/// A planned video request: what it generates and everything the engine and the workers read of
/// it.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoPlan {
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
    pub conditions: Vec<ConditionPlan>,
    /// The decode units of the output.
    pub video_units: u32,
    /// The denoiser rows the conditions take, by which the engine selects the smallest graph
    /// that holds them.
    pub condition_rows: u32,
}

/// A video family's request processor, bound to one deployment.
pub trait VideoProcessor: Send + Sync {
    /// The model's part of `GET /v1/capabilities`: what a request may ask for.
    fn capabilities(&self) -> Value;

    /// Denoiser evaluations of every request's schedule.
    fn denoise_steps(&self) -> u32;

    /// Applies every rule that needs no media, so a malformed request is refused before anything
    /// is fetched.
    ///
    /// # Errors
    ///
    /// Refuses the request, naming the parameter at fault.
    fn check(&self, request: &VideoSpec) -> Result<(), InputError>;

    /// Plans `request`, whose references' media were probed as `media` in request order, with the
    /// checkpoint's `vocabulary`.
    ///
    /// # Errors
    ///
    /// Refuses the request, naming the parameter at fault, or fails when valid input cannot be
    /// planned.
    fn plan(
        &self,
        request: &VideoSpec,
        media: &[MediaMetadata],
        vocabulary: &dyn Vocabulary,
    ) -> Result<VideoPlan, InputError>;
}

/// What a deployment binds a video family's processor with.
#[derive(Debug, Clone, Copy)]
pub struct VideoBinding<'a> {
    /// The placed denoiser as the model's workers described it
    /// (`nn::family::Description::details`).
    pub description: &'a [u8],
    /// Condition units one latent encoding round covers while a request's prompt is encoded.
    pub latent_encoding_units: u32,
    /// The deployment's longest output, in seconds.
    pub max_seconds: f64,
    /// The deployment's processor options (`mm_processor_kwargs`).
    pub processor_options: &'a Map<String, Value>,
    /// The contents of the checkpoint files [`VideoFamily::processor_files`] names, in that
    /// order.
    pub files: &'a [String],
}

/// A video family, as request processing serves it.
pub trait VideoFamily: Send + Sync {
    /// The prompt tokens a deployment admits when it states no `max_model_len`.
    fn default_prompt_tokens(&self) -> u32;

    /// The checkpoint files the processor reads, as `(component, file)` of the pipeline.
    fn processor_files(&self) -> &'static [(&'static str, &'static str)];

    /// Binds the deployment's processor.
    ///
    /// # Errors
    ///
    /// Fails when the description, the options or the files are not the family's.
    fn bind(&self, binding: VideoBinding<'_>) -> Result<Box<dyn VideoProcessor>, String>;
}
