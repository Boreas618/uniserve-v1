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
//! - [`pools`]: the launch a fake engine is started with and what its worker groups serve; the
//!   inventory of the batch-interface fake.
//! - [`policy`]: the `fcfs` module this tree builds, and the arguments a test engine's
//!   schedulers instantiate it with.
//! - [`text`]: where a fake's text ends.
//! - [`process`]: a fake engine at the process interface, and a fake node daemon behind it.
//! - [`batches`] and [`media`]: the batch-interface fake ([`FakeBatches`]) and its responses of
//!   media batches.

pub mod batches;
pub mod buffer_pool;
pub mod engine;
pub mod executor;
pub mod media;
pub mod policy;
pub mod pools;
pub mod process;
pub mod text;

pub use self::engine::{Fixture, Pace, UNITS, fake_start, spawn};
pub use batches::{
    Accepted, Extent, FakeBatches, Operate, OutputFn, SAMPLED, Tokens, block_stop_step, completed,
    draw_token,
};
pub use buffer_pool::InProcessPool;
pub use executor::FakeExecutor;
pub use media::Media;
pub use policy::{packaged_arguments, packaged_module, packaged_policy};
pub use pools::{BLOCK_TOKENS, args, cache, capacity, launch, pool, rank};
pub use text::{EOS, EndOfText};
