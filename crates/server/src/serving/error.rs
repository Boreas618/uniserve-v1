//! Serving errors: failures to resolve and tokenize a model input
//! ([`TokenizeError`]), and the runtime's own ([`ServeError`]), which
//! include failures to decode or parse model output.

use thiserror::Error;

use crate::serving::media;

#[derive(Debug, Error)]
/// Failure while resolving and tokenizing a model input.
pub enum TokenizeError {
    /// The model tokenizer cannot encode request content.
    #[error(transparent)]
    Tokenizer(#[from] crate::profile::tokenizer::TokenizerError),
    /// Chat rendering or semantic request validation fails.
    #[error(transparent)]
    Chat(#[from] crate::serving::chat::Error),
    /// Text sampling or decoding options cannot be resolved.
    #[error(transparent)]
    Text(#[from] crate::serving::text::Error),
    /// Sampling controls violate the engine contract.
    #[error(transparent)]
    Sampling(#[from] ipc::SamplingParamsError),
    /// The lowered request violates the canonical generation contract.
    #[error(transparent)]
    Generation(#[from] engine::GenerationRequestError),
    /// The requested output resolution is not served.
    #[error(transparent)]
    Resolution(#[from] crate::profile::omni::resolution::ResolutionError),
    /// The profile cannot size the request's images.
    #[error(transparent)]
    Assets(#[from] crate::profile::assets::Error),
    /// The request exceeds the loaded workers' resource limits.
    #[error(transparent)]
    Resources(#[from] engine::GenerationResourceError),
    /// The minimum generation length exceeds the maximum generation length.
    #[error("min_tokens ({min_tokens}) exceeds max_tokens ({max_tokens})")]
    MinTokensExceedsMaximum {
        /// Requested minimum generated-token count.
        min_tokens: u32,
        /// Effective maximum generated-token count.
        max_tokens: u32,
    },
    /// The request contains an invalid combination of serving controls.
    #[error("{0}")]
    Invalid(String),
    /// The blocking tokenization task cannot complete.
    #[error("tokenization task failed")]
    Task(#[source] tokio::task::JoinError),
}

/// Runtime-local serving errors.
#[derive(Debug, Error)]
pub enum ServeError {
    /// A request requires a feature unavailable on the selected model path.
    #[error("request `{request_id}` requires unsupported feature `{feature}`")]
    UnsupportedFeature {
        /// Identifier of the rejected request.
        request_id: String,
        /// Stable feature name.
        feature: &'static str,
    },
    /// A request sets a sampling control the model's generation does not
    /// define.
    #[error("request `{request_id}` sets `{control}`, but {reason}")]
    UnsupportedSamplingControl {
        /// Identifier of the rejected request.
        request_id: String,
        /// Wire name of the control.
        control: &'static str,
        /// Why the model's generation has no such control.
        reason: &'static str,
    },
    /// The prompt exceeds the profile context limit.
    #[error(
        "request `{request_id}` has {prompt_tokens} prompt tokens, exceeding the {max_tokens}-token profile limit"
    )]
    ContextLengthExceeded {
        /// Identifier of the rejected request.
        request_id: String,
        /// Tokenized prompt length.
        prompt_tokens: usize,
        /// Maximum token count declared by the profile.
        max_tokens: u32,
    },
    /// No live engine core accepts the request.
    #[error(transparent)]
    Submit(#[from] engine::SubmitError),
    /// An input image reference cannot be fetched or decoded.
    #[error("request `{request_id}` has an invalid input image {}", .index + 1)]
    ImageInput {
        /// Identifier of the rejected request.
        request_id: String,
        /// Zero-based position of the failing reference among the request's images.
        index: usize,
        /// Why the reference could not be resolved.
        #[source]
        source: media::MediaError,
    },
    /// The video service refuses a video request or cannot process it.
    #[error(transparent)]
    Video(#[from] models::video::plan::VideoInputError),
    /// Request tokenization fails before engine submission.
    #[error("request `{request_id}` cannot be tokenized")]
    Tokenize {
        /// Identifier of the rejected request.
        request_id: String,
        /// Underlying tokenization failure.
        #[source]
        source: TokenizeError,
    },
    /// Engine output cannot be decoded or parsed into serving events, for
    /// example because it violates the engine event protocol.
    #[error("request `{request_id}` output processing failed")]
    OutputProcessing {
        /// Identifier of the affected request.
        request_id: String,
        /// Underlying decoding or parsing failure.
        #[source]
        source: crate::serving::chat::Error,
    },
}
