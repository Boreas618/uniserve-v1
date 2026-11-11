//! Test support for the scheduler and its callers.
//!
//! A test of the mechanism runs the production scheduler over the production submission door
//! (`SchedulerHandle::over_batches`) and replaces only what lies beyond it: the worker. This crate
//! holds the one test double at that boundary, [`FakeBatches`], an implementation of
//! [`ipc::execution::Batches`] that answers each batch from the columns the mechanism wrote, so
//! every completion a test observes travels the serving completion path.
//!
//! - [`batches`]: the fake worker and its answer rules.
//! - [`media`]: the answers of media batches, from each request's staged record.
//! - [`pools`]: the inventories a fake states, and the limits and arguments a handle over one is
//!   built with.
//! - [`text`]: where a fake's text ends, and the control tokens a scheduler over it names.
//! - [`policy`]: the `fcfs` module this tree builds, and the package a test epoch installs it
//!   under.

pub mod batches;
pub mod media;
pub mod policy;
pub mod pools;
pub mod service;
pub mod text;

pub use batches::{
    Accepted, Answer, Extent, FakeBatches, Operate, SAMPLED, Tokens, completed, draw_token,
};
pub use media::Media;
pub use policy::{packaged_module, packaged_policy, packaged_revision};
pub use pools::{args, limits, pool, rank};
pub use service::{EpochPolicy, serve_requests};
pub use text::{EOS, EndOfText, special_tokens};
