//! The daemon: the persistent process of one node of a deployment
//! (`uniserve serve`).
//!
//! Every node runs one daemon; the leader is the node the deployment is
//! started on (node rank 0). Every daemon supervises the processes of its
//! node and holds its node's claim at the Storage Server (`supervision`).
//! The leader also serves the deployment's model. Its modules are layered,
//! each using only those below it:
//!
//! - `http`: the HTTP server, its router, middleware and listener;
//! - `openai`: the OpenAI-compatible API's wire schemas and route handlers,
//!   and the System One readout's route;
//! - [`Leader`]: the served model, which carries every request through its
//!   input processing, the engine process and its output processing;
//! - `input`: input processing (chat templates, tokenization, multimodal
//!   inputs, each model family's request lowering) into engine requests;
//! - `output`: output processing (detokenization, stop strings, reasoning
//!   and tool parsing) of engine events into public events;
//! - `migration`: moving token requests between engines with the tokens
//!   they generated, behind one stream of events per request;
//! - `engine`: starting the engine process and the client of the request
//!   interface it serves;
//! - `model`: the served checkpoint: its files, configuration and family
//!   profile; `tokenizer`: its tokenizer;
//! - `options`: the serving option tree (`options::ServeConfig`);
//! - `metrics`: the Prometheus registry the daemon exports;
//! - `restart`: the hot restart that hands a deployment to a successor
//!   daemon.

pub mod engine;
pub mod http;
pub mod input;
mod interference;
mod leader;
pub mod metrics;
pub mod migration;
pub mod model;
pub mod openai;
pub mod options;
pub mod output;
pub mod restart;
pub mod supervision;
#[cfg(test)]
pub(crate) mod test_support;
pub mod tokenizer;
pub mod worlds;

pub use crate::http::{node, serve};
pub use crate::leader::{Leader, Task};
pub use crate::supervision::{Daemon, DaemonOptions, JOIN_TIMEOUT};
