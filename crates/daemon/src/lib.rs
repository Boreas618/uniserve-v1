//! The daemon: the persistent process of one node of a deployment
//! (`uniserve serve`).
//!
//! Every node runs one daemon; the leader is the node the deployment is
//! started on (node rank 0). Every daemon supervises the processes of its
//! node and holds its node's claim at the Storage Server (`supervision`).
//! The leader also serves the deployment's model: the HTTP API (`http`,
//! `openai`) over one option tree (`options::ServeConfig`), the input and
//! output processing of every request (`serving`, `profile`), and the engine
//! process whose schedulers serve the requests, which it starts and talks to
//! over the request interface (`control`, `client`). [`Engine`] ties these
//! together for the HTTP handlers; `metrics` is the Prometheus registry the
//! daemon exports.

/// The client of an engine process: its requests and their events.
pub mod client;
/// Starting the deployment's engine process.
pub mod control;
/// The serving engine over the engine process.
pub mod frontend;
/// The HTTP server: router assembly, the listener, and graceful shutdown.
pub mod http;
pub mod metrics;
/// The OpenAI-compatible API: wire schemas, errors, and the route handlers.
pub mod openai;
/// Serving options: the command line and the configuration file.
pub mod options;
pub mod profile;
pub mod serving;
pub mod supervision;
#[cfg(test)]
pub(crate) mod test_support;

pub use crate::frontend::{Engine, Task};
pub use crate::http::serve;
pub use crate::supervision::{Daemon, DaemonOptions, JOIN_TIMEOUT};
