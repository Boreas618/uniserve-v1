//! Video requests: the processor a video family binds, which validates a model-agnostic request
//! and processes it into the engine's input. The request, its media's facts and the input are the
//! processing interface's types (`scheduling::processing::video`).

use serde_json::{Map, Value};

pub use scheduling::processing::video::{
    ConditionInput, InputError, Length, Reference, ReferenceKind, VideoInput, VideoSpec,
};

use crate::media::MediaMetadata;
use crate::vocabulary::Vocabulary;

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

    /// Processes `request`, whose references' media were probed as `media` in request order, with
    /// the checkpoint's `vocabulary`.
    ///
    /// # Errors
    ///
    /// Refuses the request, naming the parameter at fault, or fails when valid input cannot be
    /// processed.
    fn process(
        &self,
        request: &VideoSpec,
        media: &[MediaMetadata],
        vocabulary: &dyn Vocabulary,
    ) -> Result<VideoInput, InputError>;
}

/// What a deployment binds a video family's processor with.
#[derive(Debug, Clone, Copy)]
pub struct VideoBinding<'a> {
    /// The placed denoiser as the model's workers described it
    /// (`nn::family::Description::details`): the encoding of the family's description message,
    /// which the engine carries without reading it.
    pub description: &'a [u8],
    /// The deployment's longest output, in seconds.
    pub max_seconds: f64,
    /// The deployment's processor options (`mm_processor_kwargs`).
    pub processor_options: &'a Map<String, Value>,
    /// The contents of the checkpoint files [`VideoFamily::processor_files`] names, in that
    /// order.
    pub files: &'a [String],
}

/// Where a pipeline checkpoint pins the release it continues: a JSON file at the checkpoint's root
/// and the field of it that names the release as `hf://<repository>@<revision>`.
///
/// A checkpoint that pins a release holds the stage folders it replaces and omits the others,
/// which the pinned release's snapshot supplies: request processing reads each stage from the
/// checkpoint when the checkpoint holds the stage's folder and from the release otherwise, and
/// an engine opens the checkpoint and the release as one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BasePin {
    /// The file at the checkpoint's root.
    pub file: &'static str,
    /// The field of `file` that names the release.
    pub field: &'static str,
}

/// A video family, as request processing serves it.
pub trait VideoFamily: Send + Sync {
    /// The checkpoint files the processor reads, as `(stage, file)` of the pipeline: the stage's
    /// directory and the file in it.
    fn processor_files(&self) -> &'static [(&'static str, &'static str)];

    /// The output sizes a deployment of the family prepares when its configuration names none
    /// (`video_config.sizes`), each `WxH` in pixels.
    fn default_sizes(&self) -> &'static [&'static str];

    /// The longest output a deployment of the family admits when its configuration states none
    /// (`video_config.max_seconds`), in seconds.
    fn default_max_seconds(&self) -> f64;

    /// The pipeline's parts (its stage directories) whose files decide how a request is
    /// processed and generated beside the weights: the prompt's tokenizer, the processor, the
    /// schedules. A checkpoint whose files in any of them differ from the served one's is
    /// another deployment, not new weights of the served model.
    fn processing_parts(&self) -> &'static [&'static str];

    /// Where the family's checkpoints pin the release whose stage folders they continue, for a
    /// family whose checkpoints may hold only some of a pipeline's stages; `None` for a family
    /// whose checkpoints hold every stage they declare.
    fn base_pin(&self) -> Option<BasePin>;

    /// Binds the deployment's processor.
    ///
    /// # Errors
    ///
    /// Fails when the description, the options or the files are not the family's.
    fn bind(&self, binding: VideoBinding<'_>) -> Result<Box<dyn VideoProcessor>, String>;
}
