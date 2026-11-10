//! The daemon: the persistent process of one node of a deployment
//! (`uniserve serve`).
//!
//! Every node runs one daemon; the leader is the node the deployment is
//! started on (node rank 0). Every daemon supervises the processes of its
//! node and holds its node's claim at the Storage Server (`supervision`).
//! The leader also serves the deployment's model: the OpenAI-compatible API,
//! input processing and output processing are `crates/frontend`'s, and the
//! daemon supplies them the served model's engines and its own HTTP
//! connections. Its modules are layered, each using only those below it:
//!
//! - `http`: the HTTP server, its router, middleware and listener, and the
//!   connection that writes every response body (`frontend::body`);
//! - [`Leader`]: the served model, which carries every request through its
//!   input processing (`frontend::input`), the engine process and its output
//!   processing (`frontend::output`);
//! - `migration`: moving token requests between engines with the tokens
//!   they generated, behind one stream of events per request
//!   (`frontend::events::Events`);
//! - `engine`: starting the engine process and the client of the request
//!   interface it serves;
//! - `options`: the serving option tree (`options::ServerArgs`);
//! - `metrics`: the Prometheus registry the daemon exports;
//! - `restart`: the hot restart that hands a deployment to a successor
//!   daemon;
//! - `programs`: the launches of a deployment's main processes for a start
//!   that records the programs they launch (`uniserve programs`).

pub mod engine;
pub mod epoch;
pub mod experts;
pub mod http;
mod interference;
mod leader;
pub mod metrics;
pub mod migration;
pub mod options;
pub mod profiling;
pub mod programs;
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
