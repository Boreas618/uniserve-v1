//! Test support for the engine's scheduler and control thread, and for their callers.
//!
//! The engine's scheduler (`engine::scheduler`) is tested over two doubles of what lies below it:
//! the executors of its worker groups, [`FakeExecutor`], and the Storage Server's buffer pool
//! behind real queue pairs, [`InProcessPool`]. A fake engine ([`engine`]) runs the engine's own
//! control thread and request interface over them, which the daemon's tests serve their requests
//! with.
//!
//! - [`executor`]: the fake executor, its output rule and its records of what it was handed.
//! - [`buffer_pool`]: the in-process buffer pool and the knobs a case paces it with.
//! - [`engine`]: the fake engine's start over both, and the thread that runs its control thread.
//! - [`pools`]: the launch a fake engine is started with and what its worker groups serve.
//! - [`policy`]: the `arrival` module this tree builds, and the arguments a test engine's
//!   schedulers instantiate it with.
//! - [`text`]: what a fake's text model samples, and where its text ends.
//! - [`process`]: a fake engine at the process interface, and a fake node daemon behind it.

pub mod buffer_pool;
pub mod engine;
pub mod executor;
pub mod policy;
pub mod pools;
pub mod process;
pub mod text;

pub use self::engine::{Fixture, Pace, UNITS, fake_start, spawn};
pub use buffer_pool::InProcessPool;
pub use executor::FakeExecutor;
pub use policy::{packaged_arguments, packaged_module, packaged_parameters, packaged_policy};
pub use pools::{BLOCK_TOKENS, args, cache, capacity, launch};
pub use text::{EOS, EndOfText, SAMPLED, draw_token};
