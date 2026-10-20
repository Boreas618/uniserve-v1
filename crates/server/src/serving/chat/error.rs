//! Chat rendering, parsing, and output-processing errors.
//!
//! Request validation, template rendering, and output processing all report
//! this one error type; output-time errors, decoding failures carried in
//! [`Error::Text`] included, reach the caller as
//! `ServeError::OutputProcessing`.

use thiserror::Error;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Error)]
/// Error produced by chat request rendering or output processing.
pub enum Error {
    /// The request contains no messages to render.
    #[error("chat request must contain at least one message")]
    EmptyMessages,
    /// Continuation mode is selected for history that does not end with an assistant message.
    #[error("cannot continue the final message when the last message is not from the assistant")]
    ContinueFinalAssistantWithoutFinalAssistant,
    /// Prompt rendering requires a template but no template is configured.
    #[error("chat template is required but none was configured")]
    MissingChatTemplate,
    /// The chat template cannot be loaded, compiled, or rendered, or assistant
    /// history cannot be converted into template values (for example, tool-call
    /// arguments that are not valid JSON).
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    /// Multimodal content reached a text-only conversion or a renderer
    /// configured without multimodal support.
    #[error("unsupported multimodal content: {0}")]
    UnsupportedMultimodalContent(&'static str),
    /// A reasoning or tool-call output parser cannot be initialized.
    #[error("failed to initialize {kind} parser `{name}`")]
    ParserInitialization {
        /// Semantic parser category.
        kind: &'static str,
        /// Configured parser name.
        name: String,
        /// Parser-specific initialization error.
        #[source]
        error: BoxedError,
    },
    /// Incremental tool-call events violate the assembler state contract:
    /// arguments arrive with no open tool call, or for a tool index other than
    /// the open one.
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStreamInvariant {
        /// Description of the violated stream invariant.
        message: String,
    },
    /// Model profile assets cannot be loaded or validated.
    #[error(transparent)]
    ModelAssets(#[from] crate::profile::assets::Error),
    /// The text layer fails, for example in detokenization or engine output
    /// validation.
    #[error(transparent)]
    Text(#[from] crate::serving::text::Error),
}

/// Result type returned by chat calls.
pub type Result<T> = std::result::Result<T, Error>;
