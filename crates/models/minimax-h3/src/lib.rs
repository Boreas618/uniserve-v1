//! MiniMax-H3 and FastH3: the video-and-audio diffusion family, as a model file holds it.
//!
//! The crate holds what a diffusers or vLLM-Omni model file holds, on the tensor library and
//! [`nn`]:
//!
//! - **configuration** ([`config`]): the checkpoint's sidecars, read and normalized once at the
//!   loading boundary into one typed, immutable struct per numerical submodule;
//! - **the module tree**: the diffusion transformer with its dense or video-sparse attention
//!   ([`transformer`]), the denoiser around it ([`denoiser`]), the video VAE ([`video_vae`]),
//!   the audio VAE with its vocoder and its encoder ([`audio_vae`]) and the clip splice
//!   ([`splice`]). Every `forward` composes tensor operations and `nn` modules;
//! - **the packed sequence** ([`packing`]): the row layout a request's latents, prompt and
//!   conditions are laid out in, as CPU tensor expressions a forward moves to its device;
//! - **the family value** ([`family`]): how a checkpoint opens, which stages a deployment places
//!   on a rank, and what the loaded pipeline states to execution.
//!
//! The request side of the family (the resolution and frame rules, the prompt, the prepared
//! conditions) is [`processing::minimax_h3`]'s; a model consumes those values and redefines
//! none. A model holds no runtime object: the execution context a forward computes under is the
//! thread's, installed by execution (`AGENTS.md`, Model Library Boundary).

pub mod audio_vae;
pub mod config;
pub mod denoiser;
pub mod modulation;
pub mod packing;
pub mod refiner;
pub mod splice;
pub mod transformer;
pub mod video_vae;

pub use config::{
    AudioVaeConfig, Config, DenoiserConfig, ScheduleConfig, TransformerConfig, VideoVaeConfig,
    VsaConfig,
};
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
    /// missing, a stage directory holds no shard set, or the export names a task no network
    /// serves. The message names the file.
    #[error("checkpoint: {0}")]
    Checkpoint(String),

    /// A request, a layout or a tensor a module of this crate was not loaded for.
    #[error("request: {0}")]
    Request(String),

    /// A capability the design names that this crate does not implement yet, said precisely.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// The shared library's failure: a tensor operation, a launch, a weight that could not be
    /// read or converted.
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

    /// A request, a layout or a tensor a module was not loaded for.
    pub(crate) fn request(problem: impl Into<String>) -> Self {
        Self::Request(problem.into())
    }
}

impl From<uniserve_tensor::Error> for Error {
    /// A tensor operation's refusal, as the shared library reports it.
    fn from(error: uniserve_tensor::Error) -> Self {
        Self::Nn(error.into())
    }
}

impl From<qwen3::Error> for Error {
    /// A refusal of the Qwen3 family this crate composes, as the shared library reports it.
    fn from(error: qwen3::Error) -> Self {
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
