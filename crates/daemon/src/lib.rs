//! The daemon: the persistent process of one node of a deployment
//! (`uniserve serve`).
//!
//! Every node runs one daemon; the leader is the node the deployment is
//! started on (node rank 0). Every daemon supervises the processes of its
//! node and holds its node's claim at the Storage Server (`supervision`).
//! The leader also serves the deployment's model through its API server.
//! Its modules are layered, each using only those below it, but for the
//! route handlers of `api::openai`, which reach the served model through
//! the leader and the daemon's connections (`http::served`):
//!
//! - `http`: the HTTP server, its router, middleware and listener, and the
//!   connection that writes every response body (`api::body`);
//! - [`Leader`]: the served model, which carries every request through its
//!   input processing (`api::input`), the engine process and its output
//!   processing (`api::output`);
//! - `api`: the API server: the OpenAI-compatible routes, input and output
//!   processing, tokenizers and the wire protocol;
//! - `migration`: moving token requests between engines with the tokens
//!   they generated, behind one stream of events per request
//!   (`api::events::Events`);
//! - `engine`: starting the engine process and the client of the request
//!   interface it serves;
//! - `options`: the serving option tree (`options::ServerArgs`);
//! - `metrics`: the Prometheus registry the daemon exports;
//! - `restart`: the hot restart that hands a deployment to a successor
//!   daemon;
//! - `keys`: the launches of a deployment's main processes for a start
//!   that records the modules they launch (`uniserve keys`).

pub mod api;
pub mod engine;
pub mod epoch;
pub mod experts;
pub mod http;
mod interference;
pub mod keys;
mod leader;
pub mod metrics;
pub mod migration;
pub mod options;
pub mod profiling;
pub mod restart;
pub mod standby;
pub mod supervision;
#[cfg(test)]
pub(crate) mod test_support;
// The epoch layer lands beside `epochs`, which commit C deletes; nothing drives
// the switch until the update layer that owns it lands too.
pub mod observe;
#[allow(
    dead_code,
    reason = "the switch lands before the update layer that drives it"
)]
pub(crate) mod updates;

pub use crate::http::{node, serve};
pub use crate::leader::{Leader, Task};
pub use crate::supervision::{Daemon, DaemonOptions, DeviceProbe, JOIN_TIMEOUT};
