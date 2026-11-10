//! The application layer: everything between an HTTP request and the request
//! the engine serves, and between the engine's events and the HTTP response.
//!
//! It runs no numerical code: models, the scheduler and the workers are the
//! main process's, and of a model family it reads only the request-processing
//! side (`processing::Family`). The daemon supplies what it does not own, the
//! served model's engines and its HTTP connections, through
//! [`openai::Server`], and the engine's side of a request through
//! [`events::Events`] and [`events::Outcome`]. Its modules:
//!
//! - `openai`: the OpenAI-compatible API's wire schemas, errors and route
//!   handlers;
//! - `input`: input processing (chat templates, tokenization, media, each
//!   model family's request lowering) into engine requests;
//! - `output`: output processing (detokenization, stop strings, reasoning and
//!   tool parsing) of engine events into public events and response bodies;
//! - `body`: the freezable response body a connection writes ([`body::Session`]);
//! - `events`: the engine's side of one request;
//! - `tokenizer`: the served checkpoint's tokenizer and incremental decoding;
//! - `model`: the served checkpoint's files, configuration and family
//!   profile.

pub mod body;
pub mod events;
pub mod input;
pub mod model;
pub mod openai;
pub mod output;
pub mod tokenizer;

#[cfg(any(test, feature = "testing"))]
pub mod test_support;
