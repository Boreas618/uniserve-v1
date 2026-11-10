//! Text decoding of a request's engine events.
//!
//! `decoded` turns one request's engine events into [`DecodedTextEvent`]s,
//! which `output::run` consumes; `logprobs` converts engine token-ID
//! candidates into decoded strings.

pub use decoded::DecodeStream;
pub use decoded::{DecodedTextEvent, Finished, TextDecodeOptions};
pub use error::{Error, Result};
pub use logprobs::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};

mod decoded;
mod error;
mod logprobs;
