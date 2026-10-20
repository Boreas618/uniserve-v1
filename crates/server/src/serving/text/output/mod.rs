//! Decoded events, finish reasons, and logprob conversion shared by output layers.
//!
//! `decoded` turns one request's engine events into [`DecodedTextEvent`]s,
//! which `serving::assembly` consumes; `logprobs` converts engine token-ID
//! candidates into decoded strings; `finish` carries the terminal reason.

pub use decoded::{DecodedTextEvent, Finished, TextDecodeOptions, decoded_text_event_stream};
pub(crate) use decoded::{matches_stop_string, stop_string_holdback_bytes};
pub use finish::{FinishReason, StopReason};
pub use logprobs::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};
pub(crate) use logprobs::{decode_logprobs, decode_prompt_logprobs};

mod decoded;
mod finish;
mod logprobs;
