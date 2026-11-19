//! Test support for the engine's scheduler and control thread, and for their callers.
//!
//! The engine's scheduler (`engine::scheduler`) is tested over two doubles of what lies below it:
//! the executors of its worker groups, [`FakeExecutor`], and the Storage Server's buffer pool
//! behind real queue pairs, [`InProcessPool`]. A fake engine ([`engine`]) runs the engine's own
//! control thread and request interface over them, which the daemon's tests serve their requests
//! with.
//!
//! - [`executor`]: the fake executor, its output rule and its records of what it was handed.
//! - [`assets`]: released checkpoints by Hugging Face repository and commit, read from the cache.
//! - [`buffer_pool`]: the in-process buffer pool and the knobs a case paces it with.
//! - [`engine`]: the fake engine's start over both, and the thread that runs its control thread.
//! - [`latent`]: a stub of DeepSeek-V4.1-Flash's cache fields, which the engine's cache tests
//!   serve on the meta path.
//! - [`layouts`]: the worker-group layouts the engine's media tests place a pipeline in.
//! - [`pools`]: the launch a fake engine is started with and what its worker groups serve.
//! - [`policy`]: the workspace's policy packages and the modules they build, and the arguments a
//!   test engine's schedulers instantiate the default package with.
//! - [`text`]: what a fake's text model samples, and where its text ends.
//! - [`process`]: a fake engine at the process interface, and a fake node daemon behind it.
//! - [`serving`]: what the engine's own tests serve from: a Storage Server on a thread of the
//!   test's process, the launch of the Qwen3-MoE fixture, and the bound of a test's waits.

pub mod assets;
pub mod buffer_pool;
pub mod engine;
pub mod executor;
pub mod latent;
pub mod layouts;
pub mod policy;
pub mod pools;
pub mod process;
pub mod serving;
pub mod text;

pub use self::engine::{Fixture, Pace, UNITS, fake_start, spawn};
pub use buffer_pool::InProcessPool;
pub use executor::FakeExecutor;
pub use policy::{packaged_arguments, packaged_module, packaged_parameters, packaged_policy};
pub use pools::{BLOCK_TOKENS, args, cache, capacity, launch};
pub use serving::{LONG, StorageServer};
pub use text::{EOS, EndOfText, SAMPLED, draw_token};
