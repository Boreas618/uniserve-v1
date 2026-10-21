//! Text decoding of a request's engine events.
//!
//! `decoded` turns one request's engine events into [`DecodedTextEvent`]s,
//! which `output::assembly` consumes; `logprobs` converts engine token-ID
//! candidates into decoded strings.

pub use decoded::{
    DecodedTextEvent, Finished, ImageEvent, TextDecodeOptions, decoded_text_event_stream,
};
pub use error::{Error, Result};
pub use logprobs::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};

mod decoded;
mod error;
mod logprobs;
