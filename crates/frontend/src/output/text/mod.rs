//! Text decoding of a request's engine events.
//!
//! `decoded` turns one request's engine events into [`DecodedTextEvent`]s,
//! which the request's `OutputProcessor` consumes; `logprobs` converts engine
//! token-ID candidates into decoded strings.

pub use decoded::DecodeStream;
pub use decoded::{DecodedTextEvent, Finished};
pub use error::{Error, Result};

mod decoded;
mod error;
mod logprobs;
