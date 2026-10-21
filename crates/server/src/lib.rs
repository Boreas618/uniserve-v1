//! UniServe HTTP frontend, model profiles, request lowering, and output assembly.
//!
//! The crate serves one option tree (`options::ServeConfig`). [`Engine`] is
//! the serving engine: it loads the model, starts one scheduler per
//! data-parallel rank, and turns each request into a stream of public events
//! through the model's input processing and output assembly (`serving`). The
//! HTTP layer (`http`) exposes OpenAI-compatible routes over it.

/// The client of an engine process: its requests and their events.
pub mod client;
/// Starting the deployment's engine process.
pub mod control;
/// The serving engine over the engine process.
pub mod frontend;
/// The HTTP server: router assembly, the listener, and graceful shutdown.
pub mod http;
/// The OpenAI-compatible API: wire schemas, errors, and the route handlers.
pub mod openai;
/// Serving options: the command line and the configuration file.
pub mod options;
pub mod profile;
pub mod serving;
#[cfg(test)]
pub(crate) mod test_support;

pub use crate::frontend::{Engine, Task};
pub use crate::http::serve;
