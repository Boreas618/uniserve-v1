//! MiniMax-H3 with FastH3: the checkpoint configuration read once at the loading boundary, the
//! mathematical partition of a deployment, and the model's blocks.
//!
//! The crate holds one pipeline (specs/worker.md 4.3, 9): the Qwen3-VL text encoder that conditions
//! a request, one or two diffusion transformer (DiT) partitions of the checkpoint, the video and
//! audio latent encoders `ref2va` uses, and the video and audio decoders. Each is a named block
//! implementing an [`nn`] capability, and a deployment places a subset of them per worker group
//! under the component names of the deployment files: `text_encoder`, `transformer`,
//! `transformer_ref`, `video_decoder`, `audio_decoder`, `latent_encoder`.
//!
//! ## What the loading boundary owns
//!
//! A checkpoint's JSON sidecars are read and validated once, into [`config::Config`]; every layer
//! below consumes typed fields and borrowed views, and nothing reparses a dictionary or applies a
//! dynamic default (`AGENTS.md`, Model Library Boundary). [`checkpoint`] classifies the checkpoint:
//! which DiT partitions it holds, which tasks each serves, and - for a FastH3 export - the contract
//! that states its schedule and its sparse attention.
//!
//! The checkpoint's own numerical representation is used as it is: each tensor keeps the element
//! type it was stored in, every layer declares the precision it computes in, and nothing is
//! converted at load unless the deployment asks for it with `dtype` or `quantization`
//! (specs/worker.md 1).
//!
//! ## What the partition owns
//!
//! [`partition::Partition`] is the deployment's *mathematical* split - the denoiser's Ulysses
//! sequence degree and the text encoder's tensor-parallel position - not a runner, a communicator
//! or a placement. The layers apply it; the executor owns every runtime object.
//!
//! ## The model
//!
//! [`load`] composes the blocks the checkpoint holds into `nn::Model`; `uniserve engine` is the one
//! place a model class is named, and it hands the model to the executor, which serves each block by
//! the capability it implements.

pub mod attention;
pub mod audio_vae;
pub mod checkpoint;
pub mod conditioning;
pub mod config;
pub mod decoding;
pub mod denoiser;
pub mod encoding;
pub mod load;
pub mod modulation;
pub mod output;
pub mod packing;
pub mod partition;
pub mod positions;
pub mod schedule;
pub mod solver;
pub mod text_encoder;
pub mod transformer;
pub mod video_vae;
pub mod vision;

pub use checkpoint::{Contract, ContractSchedule, Dit, Layout, Sidecars, SparseContract};
pub use config::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_LATENTS_PER_SECOND, AUDIO_SAMPLE_RATE, AttentionKind,
    AudioVaeConfig, CANVAS_MULTIPLE, CANVAS_SHORT_EDGE, CLIP_FRAMES, CLIP_LATENTS, Canvas, Config,
    DMD_CANVASES, DROPPED_LATENTS, DenoiserConfig, FPS, NAMED_ASPECT_RATIOS, NAMED_CANVASES,
    NUM_TRAIN_TIMESTEPS, PixelConfig, RETAINED_LAYERS, SPARSE_TILE_ROWS, ScheduleGrid,
    ScheduleSpec, Shifts, SpatialPadding, TextEncoderConfig, TransformerConfig,
    UNIFORM_GRID_POINTS, VideoVaeConfig, VisionConfig,
};
pub use load::{Component, LoadOptions};
pub use partition::Partition;

/// What this crate can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A checkpoint sidecar states a value, or a combination of values, this crate cannot serve.
    ///
    /// The message names the sidecar and the field, so a refused checkpoint says what it would
    /// have to state.
    #[error("configuration: {0}")]
    Config(String),

    /// The checkpoint is not a MiniMax-H3 or FastH3 checkpoint this crate can open: a sidecar a
    /// layout needs is missing, a component directory holds no shard set, or the contract names a
    /// task no partition serves. The message names the file.
    #[error("checkpoint: {0}")]
    Checkpoint(String),

    /// A capability the design names that this crate does not implement yet, said precisely.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A component's checkpoint could not be opened, or a weight could not be read or converted.
    #[error(transparent)]
    Nn(#[from] nn::Error),

    /// A checkpoint file could not be read.
    #[error("reading {path}: {source}")]
    Io {
        /// The path that failed.
        path: std::path::PathBuf,
        /// The underlying filesystem error.
        source: std::io::Error,
    },

    /// A checkpoint's JSON sidecar is not valid JSON.
    #[error("parsing {path}: {source}")]
    Json {
        /// The path that failed.
        path: std::path::PathBuf,
        /// The underlying JSON error.
        source: serde_json::Error,
    },
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

    /// A capability this crate does not implement yet, said precisely.
    pub fn unsupported(problem: impl Into<String>) -> Self {
        Self::Unsupported(problem.into())
    }
}

/// A result of this crate.
pub type Result<T> = std::result::Result<T, Error>;
