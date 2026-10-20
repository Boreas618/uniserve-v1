//! Decoded events and logprob conversion shared by output layers.
//!
//! `decoded` turns one request's engine events into [`DecodedTextEvent`]s,
//! which `serving::assembly` consumes; `logprobs` converts engine token-ID
//! candidates into decoded strings.

pub use decoded::{
    DecodedTextEvent, Finished, ImageEvent, TextDecodeOptions, decoded_text_event_stream,
};
pub use logprobs::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};

mod decoded;
mod logprobs;
