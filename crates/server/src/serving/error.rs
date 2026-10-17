//! Serving errors: failures to resolve and tokenize a model input
//! ([`TokenizeError`]), to interpret or assemble model output
//! ([`OutputProcessingError`]), and the runtime's own ([`ServeError`]).

use thiserror::Error;

use super::ServeRequestId;
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
    Sampling(#[from] uniserve_core::SamplingParamsError),
    /// The lowered request violates the canonical generation contract.
    #[error(transparent)]
    Generation(#[from] uniserve_core::GenerationRequestError),
    /// Requested computation exceeds loaded model capacity.
    #[error(transparent)]
    Capacity(#[from] uniserve_core::GenerationResourceError),
    /// Multimodal request controls or geometry are invalid.
    #[error(transparent)]
    Omni(#[from] crate::serving::omni::OmniError),
    /// A requested log-probability count falls outside the accepted range.
    #[error("{field} must be non-negative or -1, got {value}")]
    InvalidLogprobCount {
        /// Invalid request field.
        field: &'static str,
        /// Rejected field value.
        value: i32,
    },
    /// The minimum generation length exceeds the maximum generation length.
    #[error("min_tokens ({min_tokens}) exceeds max_tokens ({max_tokens})")]
    MinTokensExceedsMaximum {
        /// Requested minimum generated-token count.
        min_tokens: u32,
        /// Effective maximum generated-token count.
        max_tokens: u32,
    },
    /// The selected model path cannot report token log probabilities.
    #[error("this model does not support logprobs")]
    UnsupportedLogprobs,
    /// The request contains an invalid combination of serving controls.
    #[error("{0}")]
    Invalid(String),
    /// The blocking tokenization task cannot complete.
    #[error("tokenization task failed")]
    Task(#[source] tokio::task::JoinError),
}

#[derive(Debug, Error)]
/// Failure while interpreting or assembling model output.
pub enum OutputProcessingError {
    /// Generated token identifiers cannot be decoded.
    #[error(transparent)]
    Tokenizer(#[from] crate::profile::tokenizer::TokenizerError),
    /// Structured chat output cannot be parsed or assembled.
    #[error(transparent)]
    Chat(#[from] crate::serving::chat::Error),
    /// Text output cannot be decoded or collected.
    #[error(transparent)]
    Text(#[from] crate::serving::text::Error),
    /// Reasoning delimiters cannot be parsed.
    #[error(transparent)]
    Reasoning(#[from] crate::profile::reasoning::ReasoningError),
    /// Engine output violates a serving-layer invariant.
    #[error("{0}")]
    Malformed(String),
}

/// Runtime-local serving errors.
#[derive(Debug, Error)]
pub enum ServeError {
    /// A request asks for more output candidates than the runtime supports.
    #[error("request `{request_id}` asks for {requested} outputs; only one output is supported")]
    UnsupportedOutputCount {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Requested number of candidates.
        requested: u32,
    },
    /// A request requires a feature unavailable on the selected model path.
    #[error("request `{request_id}` requires unsupported feature `{feature}`")]
    UnsupportedFeature {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Stable feature name.
        feature: &'static str,
    },
    /// A request sets a sampling control the model's generation does not
    /// define.
    #[error("request `{request_id}` sets `{control}`, but {reason}")]
    UnsupportedSamplingControl {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
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
        request_id: ServeRequestId,
        /// Tokenized prompt length.
        prompt_tokens: usize,
        /// Maximum token count declared by the profile.
        max_tokens: u32,
    },
    /// The request's full cache requirement exceeds runtime capacity.
    #[error(
        "request `{request_id}` requires {required_tokens} KV tokens, exceeding the {max_tokens}-token runtime limit"
    )]
    ContextCapacityExceeded {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Total cache tokens required by the request.
        required_tokens: u64,
        /// Maximum cache tokens available to one request.
        max_tokens: u32,
    },
    /// Another live request already owns the same identifier.
    #[error("request `{request_id}` is already active")]
    DuplicateRequestId {
        /// Conflicting request identifier.
        request_id: ServeRequestId,
    },
    /// Model assets or capabilities cannot be resolved.
    #[error(transparent)]
    ModelResolution(#[from] crate::serving::model::ModelResolutionError),
    /// Engine submission or streaming fails.
    #[error(transparent)]
    Engine(#[from] crate::engine_client::Error),
    /// An input image reference cannot be fetched or decoded.
    #[error("request `{request_id}` has an invalid input image")]
    ImageInput {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// The failing reference and its cause.
        #[source]
        source: media::ImageListError,
    },
    /// Request tokenization fails before engine submission.
    #[error("request `{request_id}` cannot be tokenized")]
    Tokenize {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Underlying tokenization failure.
        #[source]
        source: TokenizeError,
    },
    /// Engine output cannot be converted into serving events.
    #[error("request `{request_id}` output processing failed")]
    OutputProcessing {
        /// Identifier of the affected request.
        request_id: ServeRequestId,
        /// Underlying output-processing failure.
        #[source]
        source: OutputProcessingError,
    },
}

/// Builds an output-processing error for malformed engine output.
pub(super) fn malformed_output(
    request_id: impl Into<ServeRequestId>,
    message: impl Into<String>,
) -> ServeError {
    ServeError::OutputProcessing {
        request_id: request_id.into(),
        source: OutputProcessingError::Malformed(message.into()),
    }
}
