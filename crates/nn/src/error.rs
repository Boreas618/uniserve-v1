//! What the shared computation library can fail with.
//!
//! A layer propagates an operator's failure unchanged; it never falls back to another path. The
//! loading boundary adds the failures of reading a checkpoint and converting a weight, and the
//! forward's arena adds the one failure a static activation plan can have: a request that does not
//! fit the region the worker reserved.

use std::path::PathBuf;

/// A failed layer call, checkpoint read or weight conversion.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An operator launch failed.
    #[error(transparent)]
    Ops(#[from] ops::Error),

    /// One operator's launch failed, with the program it selected.
    ///
    /// A native entry reports its own failure by the name it was loaded under
    /// (`native entry \`uniserve_gemm\` failed with status 1`), which says nothing about *which*
    /// call it was: one operator is launched at many descriptor keys in one forward, and a runtime
    /// refusal of one cell reads exactly like a refusal of another. The forward, which resolves the
    /// program and holds the descriptor, is the only place both are in scope, so it names them here
    /// rather than leaving a stop to be narrowed by guesswork.
    #[error("operator `{operator}` descriptor {descriptor}: {source}")]
    Launch {
        /// The operator whose program failed, as its package names it.
        operator: &'static str,
        /// The descriptor key of the program that was selected.
        descriptor: String,
        /// What the launch failed with.
        #[source]
        source: ops::Error,
    },

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

    /// A loader's policy does not describe the checkpoint it is asked to read.
    ///
    /// A policy (an element type, a quantization) is chosen once for a model; a component whose
    /// weights are not of that kind would be read as something it is not, which is a wrong answer
    /// rather than a convenience.
    #[error("loader policy: {problem}")]
    Policy {
        /// What the policy asks for and what the checkpoint holds.
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

    /// A policy that does not describe the checkpoint.
    pub(crate) fn policy(problem: impl Into<String>) -> Self {
        Self::Policy {
            problem: problem.into(),
        }
    }

    /// A failed launch, named by its operator and the program it selected.
    pub(crate) fn launch(operator: &'static str, descriptor: String, source: ops::Error) -> Self {
        Self::Launch {
            operator,
            descriptor,
            source,
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
