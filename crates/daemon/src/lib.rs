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
//! - `config`: the serving option tree (`config::Config`);
//! - `observation`: the deployment's observation store, the `stats` and
//!   `events` tables queried with SQL, and its writers (`observation::record`);
//! - `origins`: where the modules each replica runs come from (`uniserve_replica_module`);
//! - `programs`: the policy package's control and routing programs, which the leader runs in
//!   the program host, and the KV program it hands the Storage Servers;
//! - `restart`: the hot restart that hands a deployment to a successor
//!   daemon.

pub mod api;
pub mod config;
pub mod engine;
pub mod epoch;
pub mod experts;
pub mod http;
mod interference;
mod leader;
pub mod migration;
pub mod observation;
pub mod observe;
pub(crate) mod origins;
pub mod policy;
pub mod profiling;
pub(crate) mod programs;
pub mod restart;
pub mod standby;
pub mod supervision;
#[cfg(test)]
pub(crate) mod test_support;
pub(crate) mod updates;

pub use crate::http::serve;
pub use crate::leader::{Leader, Task};
pub use crate::supervision::{Daemon, DaemonOptions, JOIN_TIMEOUT};
