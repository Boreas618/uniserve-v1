//! Text tokenization, decoding, and stream errors.
//!
//! `decoded_text_event_stream` reports engine-protocol violations as
//! [`Error::MalformedOutput`]; an engine rejection or failure is a terminal
//! event of its stream, not an error.

use thiserror::Error;

#[derive(Debug, Error)]
/// Text input, tokenization, decoding, or engine-stream failure.
pub enum Error {
    /// The tokenizer cannot encode or decode model text.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// A tokenized text request contains an empty prompt.
    #[error("text request `{request_id}` must contain at least one prompt token ID")]
    EmptyPromptTokenIds {
        /// Identifier of the rejected request.
        request_id: String,
    },
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
    /// The engine stream closes without terminal output.
    #[error("text request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput {
        /// Identifier of the incomplete request.
        request_id: String,
    },
    /// An engine event cannot be represented as valid decoded text output.
    #[error("text request `{request_id}` received malformed output: {message}")]
    MalformedOutput {
        /// Identifier of the affected request.
        request_id: String,
        /// Description of the violated output invariant.
        message: String,
    },
}

/// Result type returned by text serving calls.
pub type Result<T> = std::result::Result<T, Error>;

impl From<crate::profile::tokenizer::TokenizerError> for Error {
    /// Keeps the tokenizer error's message as [`Error::Tokenizer`].
    fn from(error: crate::profile::tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(error.0)
    }
}
