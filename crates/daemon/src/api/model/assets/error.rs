//! Errors raised while locating and reading model assets.

use thiserror::Error;

#[derive(Debug, Error)]
/// Model-asset discovery and deserialization failure.
pub enum Error {
    #[error("failed to read model asset `{path}`")]
    /// A local model asset could not be read.
    Io {
        /// Path of the unreadable asset.
        path: std::path::PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse model asset `{path}` as JSON")]
    /// A model asset contains invalid JSON.
    Json {
        /// Path of the malformed asset.
        path: std::path::PathBuf,
        /// Underlying JSON parser error.
        #[source]
        source: serde_json::Error,
    },
    #[error("the checkpoint `{model}` has no `{file}`")]
    /// A file the checkpoint's family declares, or its format requires, is absent from the
    /// checkpoint's directory.
    MissingFile {
        /// The checkpoint's directory.
        model: String,
        /// The checkpoint-relative file path.
        file: String,
    },
    #[error("`{model}` is not a checkpoint directory")]
    /// A checkpoint stated as a local directory is not one.
    NotDirectory {
        /// The stated directory.
        model: String,
    },
    #[error(transparent)]
    /// The checkpoint declares no family the executable serves.
    Family(#[from] processing::ResolveError),
    #[error("unsupported diffusers pipeline {class_name}")]
    /// A family that has no pipeline form is resolved as a pipeline.
    UnsupportedPipeline {
        /// The family id `ModelConfig::from_pipeline` rejects.
        class_name: String,
    },
    #[error(
        "a media deployment states its `max_model_len`, the longest prompt it admits: the \
         pipeline keeps no key/value cache, so no checkpoint figure implies one"
    )]
    /// A media pipeline is served without the prompt length its deployment must state.
    MissingPromptLength,
    #[error("failed to resolve `{model}` on the Hugging Face Hub: {message}")]
    /// Listing a revision or downloading a file failed.
    Remote {
        /// The repository and revision.
        model: String,
        /// Human-readable remote resolution failure.
        message: String,
    },
    #[error(
        "the Hugging Face cache holds no snapshot of `{asset}` ({}), and HF_HUB_OFFLINE forbids \
         downloading it: populate the cache first",
        path.display()
    )]
    /// An offline host's cache holds no snapshot of a revision.
    NotCached {
        /// The repository and revision.
        asset: String,
        /// The reference or snapshot the cache lacks.
        path: std::path::PathBuf,
    },
    #[error("invalid model asset: {0}")]
    /// Model asset content violates a profile invariant.
    Invalid(String),
}

/// Result type returned by model-asset calls.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Constructs an invalid-asset error with contextual text.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}
