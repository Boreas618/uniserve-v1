//! Test support for the scheduler and its callers.
//!
//! A test of the mechanism runs the production scheduler over the production submission door
//! (`SchedulerHandle::over_batches`) and replaces only what lies beyond it: the worker. This crate
//! holds the one test double at that boundary, [`FakeBatches`], an implementation of
//! [`ipc::execution::Batches`] that responds to each batch from the columns the mechanism wrote, so
//! every completion a test observes travels the serving completion path.
//!
//! The engine's scheduler (`engine::scheduler`) is tested over two doubles of its own: the
//! executors of its worker groups, [`FakeExecutor`], and the Storage Server's buffer pool behind
//! real queue pairs, [`InProcessPool`].
//!
//! - [`executor`]: the fake executor, its output rule and its records of what it was handed.
//! - [`buffer_pool`]: the in-process buffer pool and the knobs a case paces it with.
//! - [`batches`]: the fake worker and its response rules.
//! - [`media`]: the responses of media batches, from each request's staged record.
//! - [`pools`]: the inventories a fake states, and the limits and arguments a handle over one is
//!   built with.
//! - [`text`]: where a fake's text ends, and the control tokens a scheduler over it names.
//! - [`policy`]: the `fcfs` module this tree builds, and the package a test epoch installs it
//!   under.
//! - [`service`]: a fake engine at the request interface, over production schedulers.
//! - [`process`]: a fake engine at the process interface, and a fake node daemon behind it.

pub mod batches;
pub mod buffer_pool;
pub mod executor;
pub mod media;
pub mod policy;
pub mod pools;
pub mod process;
pub mod service;
pub mod text;

pub use batches::{
    Accepted, Extent, FakeBatches, Operate, OutputFn, SAMPLED, Tokens, block_stop_step, completed,
    draw_token,
};
pub use buffer_pool::InProcessPool;
pub use executor::FakeExecutor;
pub use media::Media;
pub use policy::{packaged_module, packaged_policy, packaged_revision};
pub use pools::{args, limits, pool, rank};
pub use service::{EpochPolicy, ProcessLink, serve_engine, serve_requests};
pub use text::{EOS, EndOfText, special_tokens};
