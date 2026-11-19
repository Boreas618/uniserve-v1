//! Input processing's errors: failures to lower and tokenize a request
//! ([`TokenizeError`]), and the refusals a request leaves the deployment with
//! before an engine serves it ([`ServeError`]).

use thiserror::Error;

use crate::api::input::media;

#[derive(Debug, Error)]
/// Failure while resolving and tokenizing a model input.
pub enum TokenizeError {
    /// The model tokenizer cannot encode request content.
    #[error(transparent)]
    Tokenizer(#[from] crate::api::tokenizer::TokenizerError),
    /// The epoch's processing program refuses to render the request: its family's prompt
    /// cannot be made of it.
    #[error("{0}")]
    Render(scheduling::processing::Refusal),
    /// The epoch's processing program cannot open the response's parser for the request.
    #[error("{0}")]
    Parser(String),
    /// The tokenized prompt fills or exceeds the model context window, leaving
    /// no room for a generated token.
    #[error(
        "this model's maximum context length is {max_model_len} tokens, \
         but the prompt contains {prompt_len} input tokens"
    )]
    PromptTooLong {
        /// Maximum token count accepted by the model context.
        max_model_len: u32,
        /// Tokenized prompt length.
        prompt_len: u32,
    },
    /// A request parameter states a value the deployment does not take.
    #[error("{message}")]
    InvalidParameter {
        /// The request field at fault, when one field is.
        param: Option<&'static str>,
        /// What is wrong with it.
        message: String,
    },
    /// Sampling controls violate the engine's sampling rules.
    #[error(transparent)]
    Sampling(#[from] proto::request::SamplingParamsError),
    /// The lowered request violates the engine request's rules.
    #[error(transparent)]
    Generation(#[from] proto::request::GenerationRequestError),
    /// The profile cannot size the request's images.
    #[error(transparent)]
    Assets(#[from] crate::api::model::assets::Error),
    /// The request exceeds the loaded workers' resource limits.
    #[error(transparent)]
    Resources(#[from] proto::request::GenerationResourceError),
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
    /// No live scheduler accepts the request.
    #[error(transparent)]
    Submit(#[from] crate::engine::client::SubmitError),
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
    /// The video model refuses a video request, or its processing fails.
    #[error(transparent)]
    Video(#[from] scheduling::processing::video::InputError),
    /// The epoch's processing program failed processing the request: its instance ended.
    #[error(transparent)]
    Processing(#[from] crate::api::program::Error),
    /// Request tokenization fails before engine submission.
    #[error("request `{request_id}` cannot be tokenized")]
    Tokenize {
        /// Identifier of the rejected request.
        request_id: String,
        /// Underlying tokenization failure.
        #[source]
        source: TokenizeError,
    },
}
