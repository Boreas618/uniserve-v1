//! Text tokenization, decoding, sampling lowering, and output helpers.
//!
//! Model profiles own request tokenization; this module provides reusable text
//! primitives for profiles and response assemblers.

pub use error::{Error, Result};
pub use output::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTextEvent,
    DecodedTokenLogprob, Finished, ImageEvent, TextDecodeOptions,
};

mod error;
/// Incremental decoded-text and log-probability output values.
pub mod output;
