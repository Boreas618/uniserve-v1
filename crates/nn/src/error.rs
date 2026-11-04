//! What the shared computation library can fail with.
//!
//! A layer propagates an operator's failure unchanged; it never falls back to another path. The
//! loading boundary adds the failures of reading a checkpoint and converting a weight, and the
//! forward's arena adds the one failure a static activation plan can have: a request that does not
//! fit the region the executor reserved.

use std::path::PathBuf;

/// A failed layer call, checkpoint read or weight conversion.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An operator launch failed.
    #[error(transparent)]
    Ops(#[from] ops::Error),

    /// A call into the CUDA boundary failed.
    #[error("platform: {0}")]
    Platform(#[from] ops::platform::PlatformError),

    /// A file could not be read.
    #[error("io: {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A safetensors shard could not be read.
    #[error("safetensors: {path}: {source}")]
    Safetensors {
        path: PathBuf,
        #[source]
        source: safetensors::SafeTensorError,
    },

    /// A checkpoint's index or configuration is not well formed.
    #[error("checkpoint {path}: {problem}")]
    Checkpoint { path: PathBuf, problem: String },

    /// A tensor a caller asked for is not in the checkpoint.
    #[error("the checkpoint has no tensor `{0}`")]
    MissingTensor(String),

    /// A weight request and what the checkpoint holds disagree.
    #[error("weight `{tensor}`: {problem}")]
    Weight { tensor: String, problem: String },

    /// The arena reserved for one captured bucket cannot hold what this forward asks of it.
    #[error("arena: {needed} bytes requested at offset {at} of a {available}-byte region")]
    Arena {
        needed: u64,
        at: u64,
        available: u64,
    },

    /// A view's shape, dtype or rank does not fit what the layer was built for.
    #[error("{owner}: {problem}")]
    Shape {
        owner: &'static str,
        problem: String,
    },

    /// A capability of the design that this crate does not implement yet, named precisely.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl Error {
    /// A shape disagreement of `owner`.
    pub(crate) fn shape(owner: &'static str, problem: impl Into<String>) -> Self {
        Self::Shape {
            owner,
            problem: problem.into(),
        }
    }

    /// A disagreement between a weight request and the checkpoint.
    pub(crate) fn weight(tensor: &str, problem: impl Into<String>) -> Self {
        Self::Weight {
            tensor: tensor.to_owned(),
            problem: problem.into(),
        }
    }
}

/// A result of this crate.
pub type Result<T> = std::result::Result<T, Error>;
