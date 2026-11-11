//! The device backend of the tensor library.
//!
//! `uniserve-tensor` computes on the CPU by itself and reaches every other device through its
//! open backend traits. This crate implements them for UniServe: the CUDA device of a GPU and the
//! meta device, storage over memory the runtime owns, the generic primitives as launches of `ops`
//! kernels, and the execution context a forward computes under.
//!
//! The context is the thread's: execution installs it around a model call, and a model never
//! holds, receives or names it. Every kernel reaches the device through [`launch()`], so kernel
//! updates, profile runs, graph rebinding and shadow comparison cover every launch alike. A
//! partitioned forward meets the other ranks of its group through [`collective`], and a module
//! states the range its forward runs in through [`profiler::range`].

pub mod allocator;
pub mod collective;
pub mod context;
pub mod error;
mod input;
mod launch;
pub mod profiler;

pub use allocator::Allocator;
pub use collective::Group;
pub use context::{Context, ContextGuard};
pub use error::{Error, Result};
pub use launch::launch;
pub use profiler::Profiler;
