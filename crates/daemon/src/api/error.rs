//! The API server's own failures: what keeps a request's text from becoming tokens, or its
//! engine events from becoming its response.
//!
//! Every layer of the API server that decodes, parses or assembles reports [`Error`], each
//! variant with the source it was caused by; the route handlers turn it into the public error
//! a client sees (`openai::ApiError`).

use crate::api::tokenizer::TokenizerError;

/// Why the API server cannot process a request's text or its engine output.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The tokenizer cannot encode or decode the request's text.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    /// The request's lowering states no prompt token.
    #[error("request `{request_id}` has no prompt token")]
    EmptyPrompt {
        /// Identifier of the request.
        request_id: String,
    },
    /// An engine event violates the request interface's event order, or states tokens the
    /// request cannot have produced.
    #[error("request `{request_id}` received malformed engine output: {message}")]
    MalformedOutput {
        /// Identifier of the affected request.
        request_id: String,
        /// The violated invariant.
        message: String,
    },
    /// The lowering a response is built from, or the state a handed-over response is resumed
    /// from, does not state a part every response states.
    #[error("the response's lowering states no {part}")]
    Lowering {
        /// The missing part.
        part: &'static str,
    },
    /// The family's parsers refuse the model's output: a tool call that breaks its format's
    /// grammar or its tool's schema, text that ended inside a call; the processing program's
    /// report.
    #[error("{message}")]
    Output {
        /// The program's report.
        message: String,
    },
    /// Tool-call events arrive out of the order the response assembles them in: arguments with
    /// no open call.
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStream {
        /// The violated order.
        message: String,
    },
    /// The epoch's processing program failed: the request's instance ended.
    #[error(transparent)]
    Processing(#[from] crate::api::program::Error),
}

impl Error {
    /// The output refusal the processing program reports as `message`.
    pub fn output(message: String) -> Self {
        Self::Output { message }
    }
}

/// Result type of the API server's processing.
pub type Result<T> = std::result::Result<T, Error>;
