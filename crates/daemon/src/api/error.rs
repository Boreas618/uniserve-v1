//! The API server's own failures: what keeps a request's text from becoming tokens, or its
//! engine events from becoming its response.
//!
//! Every layer of the API server that decodes, parses or assembles reports [`Error`], each
//! variant with the source it was caused by; the route handlers turn it into the public error
//! a client sees (`openai::ApiError`).

use crate::api::tokenizer::TokenizerError;

/// A boxed source error.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

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
    /// The reasoning or tool-call parser the request's family declares cannot be built for it.
    #[error("cannot build the {kind} parser {name:?}")]
    Parser {
        /// `reasoning` or `tool`.
        kind: &'static str,
        /// The format's name.
        name: String,
        /// Why it cannot be built.
        #[source]
        source: BoxError,
    },
    /// The model's text is not a tool call of the format its family declares: it breaks the
    /// format's grammar, a value is not what its tool's schema states, or the text ended
    /// inside a call.
    #[error("the model wrote a malformed {format} tool call: {message}")]
    ToolCall {
        /// The tool-call format's name.
        format: &'static str,
        /// What the text breaks.
        message: String,
        /// The syntax error of the call's JSON, where the format's JSON broke.
        #[source]
        source: Option<BoxError>,
    },
    /// Tool-call events arrive out of the order the response assembles them in: arguments with
    /// no open call, or for another call than the open one.
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStream {
        /// The violated order.
        message: String,
    },
}

/// Result type of the API server's processing.
pub type Result<T> = std::result::Result<T, Error>;
