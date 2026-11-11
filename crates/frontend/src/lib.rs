//! The application layer: everything between an HTTP request and the request
//! the engine serves, and between the engine's events and the HTTP response.
//!
//! It runs no numerical code: models, the scheduler and the workers are the
//! main process's, and of a model family it reads only the request-processing
//! side (`processing::Family`). The daemon supplies what it does not own, the
//! served model's engines and its HTTP connections, through
//! [`openai::Server`], and the engine's side of a request through
//! [`events::Events`] and [`events::Stats`]. A request passes through its
//! layers in one direction, route handler, `Processor`, the daemon's hold and
//! an engine, `OutputProcessor`, response: input processing imports no output
//! processing, and neither imports the route handlers. Its modules:
//!
//! - `protocol`: the OpenAI-compatible API's wire schemas;
//! - `openai`: the API's route handlers, its errors and the response bodies
//!   they write;
//! - `input`: input processing ([`input::Processor`]: chat templates,
//!   tokenization, media, each model family's request lowering) into engine
//!   requests;
//! - `output`: output processing ([`output::OutputProcessor`]:
//!   detokenization, stop strings, reasoning and tool parsing) of engine
//!   events into response deltas;
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
pub mod protocol;
pub mod tokenizer;

#[cfg(any(test, feature = "testing"))]
pub mod test_support;
