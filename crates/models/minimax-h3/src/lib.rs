//! MiniMax-H3 with FastH3: the checkpoint configuration read once at the loading boundary and the
//! numerical pipeline.
//!
//! The crate holds one pipeline: the Qwen3-VL text encoder that conditions a request
//! ([`qwen3::Qwen3VlModel`], truncated to [`config::RETAINED_LAYERS`] with no final norm), one or two
//! diffusion transformer (DiT) partitions of the checkpoint ([`denoiser`]), the video and audio
//! latent encoders a conditioned task reads ([`encoding`]), and the video and audio decoders. Each
//! is a stage implementing an [`nn`] capability, and a deployment places a subset of them per worker
//! group under the stage names of the deployment files: `text_encoder`, `transformer`,
//! `transformer_ref`, `video_decoder`, `audio_decoder`, `latent_encoder`, `video_postprocessor`
//! ([`family`]).
//!
//! A checkpoint's JSON sidecars are read and validated once, into [`config::Config`]; every layer
//! below consumes typed fields and borrowed views. Request processing - the request rules, the
//! canvas and frame rules, prompt presentation - is [`processing::minimax_h3`]'s.

pub mod attention;
pub mod attention_sparse;
pub mod audio_vae;
pub mod checkpoint;
pub mod conditioning;
pub mod config;
pub mod denoiser;
pub mod encoding;
pub mod family;
pub mod modulation;
pub mod output;
pub mod packing;
pub mod schedule;
pub mod solver;
pub mod tiled_encode;
pub mod transformer;
pub mod video_vae;

pub use config::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_LATENTS_PER_SECOND, AUDIO_SAMPLE_RATE, AttentionKind,
    AudioVaeConfig, CANVAS_RULE, Config, DenoiserConfig, FPS, FRAMES, MAX_SECONDS, MIN_SECONDS,
    NAMED_ASPECT_RATIOS, NUM_TRAIN_TIMESTEPS, RETAINED_LAYERS, SPARSE_TILE_ROWS, ScheduleGrid,
    ScheduleSpec, Shifts, TransformerConfig, UNIFORM_GRID_POINTS, VideoVaeConfig,
};
pub use family::MinimaxH3;
pub use processing::minimax_h3::VideoTask;

/// What this crate can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A checkpoint sidecar states a value, or a combination of values, this crate cannot serve.
    ///
    /// The message names the sidecar and the field, so a refused checkpoint says what it would
    /// have to state.
    #[error("configuration: {0}")]
    Config(String),

    /// The checkpoint is not a MiniMax-H3 or FastH3 checkpoint this crate can open: a sidecar is
    /// missing, a stage directory holds no shard set, or the export names a task no partition
    /// serves. The message names the file.
    #[error("checkpoint: {0}")]
    Checkpoint(String),

    /// A capability the design names that this crate does not implement yet, said precisely.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A stage's checkpoint could not be opened, or a weight could not be read or converted.
    #[error(transparent)]
    Nn(#[from] nn::Error),
}

impl Error {
    /// A configuration value, or a combination of values, that this crate does not serve.
    pub(crate) fn config(problem: impl Into<String>) -> Self {
        Self::Config(problem.into())
    }

    /// A checkpoint whose layout or files this crate cannot open.
    pub(crate) fn checkpoint(problem: impl Into<String>) -> Self {
        Self::Checkpoint(problem.into())
    }
}

impl From<qwen3::Error> for Error {
    /// A refusal of the Qwen3 family this crate composes, as the shared library reports it.
    fn from(error: qwen3::Error) -> Self {
        Self::Nn(error.into())
    }
}

impl From<ops::Error> for Error {
    /// An operator-level refusal of a view (a narrow, a decode) as the shared library reports it.
    fn from(error: ops::Error) -> Self {
        Self::Nn(error.into())
    }
}

/// A result of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// The family's name in a refusal it reports through the shared library.
const MODEL: &str = "minimax-h3";

impl From<Error> for nn::Error {
    /// A failure as the capability and family traits report it: a shared-library failure is
    /// already one and passes through unchanged, and every refusal of this crate's own is
    /// [`nn::Error::Model`] naming the family.
    fn from(error: Error) -> Self {
        match error {
            Error::Nn(error) => error,
            other => nn::Error::model(MODEL, other),
        }
    }
}
