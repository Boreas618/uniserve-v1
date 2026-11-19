//! The API server: everything between an HTTP request and the request the engine serves, and
//! between the engine's events and the HTTP response.
//!
//! It runs no numerical code and links no model family: models, the scheduler and the workers are
//! the main process's, and a family's request side (which classes it claims, how a request
//! becomes a prompt, how a completion is parsed) is the epoch's processing program
//! ([`program`]), which the daemon loads with each epoch's module set. The route
//! handlers reach the served model through the leader ([`crate::Leader`]) and write their bodies
//! on the daemon's own connections (`crate::http::served::Served`); output processing reads the
//! engine's side of a request through [`events::Events`] and [`events::Stats`], which the engine
//! client implements and the tests replace. A request passes through its layers in one direction,
//! route handler, `Processor`, the leader's hold and an engine, `OutputProcessor`, response: input
//! processing imports no output processing, and neither imports the route handlers. Its modules:
//!
//! - `protocol`: the OpenAI-compatible API's wire schemas;
//! - `openai`: the API's route handlers, its errors and the response bodies they write;
//! - `program`: the epoch's processing program, its bindings and each request's instance;
//! - `input`: input processing ([`input::Processor`]: rendering through the program,
//!   tokenization, media, each model family's request lowering) into engine requests;
//! - `output`: output processing ([`output::OutputProcessor`]: detokenization, stop strings,
//!   parsing through the program) of engine events into response deltas;
//! - `body`: the freezable response body a connection writes ([`body::Session`]);
//! - `events`: the engine's side of one request;
//! - `tokenizer`: the served checkpoint's tokenizer and incremental decoding;
//! - `model`: the served checkpoint's files, configuration and family profile.

pub mod body;
pub mod error;
pub mod events;
pub mod input;
pub mod model;
pub mod openai;
pub mod output;
pub mod program;
pub mod protocol;
pub mod tokenizer;

/// The wall-clock time since the Unix epoch, which the API's timestamps state: an OpenAI
/// object's `created` in seconds, a response's start in microseconds. A system clock set before
/// the epoch reads as the epoch.
pub fn since_epoch() -> std::time::Duration {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
}

#[cfg(test)]
pub mod test_support;
/// Input processing and output assembly over real tokenizers and templates.
#[cfg(test)]
mod tests;
