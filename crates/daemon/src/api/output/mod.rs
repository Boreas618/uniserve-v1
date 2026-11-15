//! Output processing: what turns a submitted request's engine events into
//! its response's deltas.
//!
//! A token request's [`OutputProcessor`] decodes its events (`text`), parses
//! reasoning and tool calls in the formats the request's model family writes
//! (`chat`, `reasoning`, `tools`) and frames the result as [`RequestOutput`]s,
//! which the route's response body writes; `media_output` frames a video
//! request's events, whose artifacts `video` collects. What a request's
//! output processing is built from is the lowering its input processing
//! stated (`proto::handoff::Lowering`).

mod assembly;
pub mod chat;
mod events;
mod processor;
pub mod reasoning;
pub mod text;
pub mod tools;
pub mod video;

use std::pin::Pin;
use std::time::Instant;

use futures::Stream;

use crate::api::events::Events;

pub use assembly::started_instant;
pub use events::{
    FinishReason, PromptInfo, RequestOutput, RequestUsage, RuntimeTimings, StopReason,
    finish_reason_name,
};
pub use processor::OutputProcessor;

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
    pub source: crate::api::error::Error,
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
