//! Output processing: what turns a submitted request's engine events into
//! its public events.
//!
//! A token request's body (`run::GenerationRun`) decodes its events
//! (`text`), runs them through the request's chat output processor
//! (`chat`), whose model family chose its reasoning and tool parsers
//! (`reasoning`, `tools`), frames the result as [`RequestOutput`] events and
//! hands each to the sink that writes the response; `media_output` frames a
//! video request's events, whose artifacts `video` collects. The
//! parameters each request's output takes ([`OutputParams`]) come from its
//! input processing.

mod assembly;
pub mod chat;
mod events;
pub mod reasoning;
pub mod run;
pub mod text;
pub mod tools;
pub mod video;

use std::pin::Pin;
use std::time::Instant;

use futures::Stream;

use crate::events::Events;

pub use events::{
    FinishReason, PromptInfo, RequestOutput, RequestUsage, RuntimeTimings, StopReason,
};

/// Why a submitted request's engine events could not be processed into its
/// public events, for example because they violate the engine's event
/// protocol or its decoded text cannot be parsed.
#[derive(Debug, thiserror::Error)]
#[error("request `{request_id}` output processing failed")]
pub struct OutputError {
    /// Identifier of the affected request.
    pub request_id: String,
    /// Underlying decoding or parsing failure.
    #[source]
    pub source: chat::Error,
}

/// What the daemon keeps of a generation request to decode and parse its
/// output once the request has moved into the engine.
pub struct OutputParams {
    /// The model's tokenizer.
    pub tokenizer: crate::tokenizer::DynTokenizer,
    /// The prompt token IDs submitted to the engine, which seed incremental
    /// detokenization.
    pub prompt_token_ids: Vec<u32>,
    /// Incremental detokenization and stop strings.
    pub decode: text::TextDecodeOptions,
    /// Whether prompt log-probabilities are requested.
    pub prompt_logprobs: bool,
    /// Whether generated-token log-probabilities are requested.
    pub logprobs: bool,
    /// The request's reasoning and tool-call parsers.
    pub processor: chat::ChatOutputProcessor,
}

/// Boxed asynchronous stream of a request's public events.
pub type RequestOutputStream =
    Pin<Box<dyn Stream<Item = Result<RequestOutput, OutputError>> + Send>>;

/// Builds the public event stream of one submitted media request, whose
/// prompt holds `prompt_tokens` tokens.
///
/// `started` is when the request entered the server and `compile_us` how
/// long it took to reach the engine; both reach the terminal usage.
pub fn media_output(
    request_id: String,
    started: Instant,
    compile_us: u64,
    prompt_tokens: usize,
    events: Box<dyn Events>,
) -> RequestOutputStream {
    Box::pin(assembly::media_output(
        request_id,
        started,
        compile_us,
        prompt_tokens,
        events,
    ))
}
