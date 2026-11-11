//! The device backend of the tensor library.
//!
//! `uniserve-tensor` computes on the CPU by itself and reaches every other device through its
//! open backend traits. This crate implements them for UniServe: the CUDA device of a GPU and the
//! meta device, storage over memory the runtime owns, the generic primitives as launches of `ops`
//! kernels, and the execution context a forward computes under.
//!
//! The context is the thread's: execution installs it around a model call, and a model never
//! holds, receives or names it. Every kernel reaches the device through [`launch`], so kernel
//! updates, profile runs, graph rebinding and shadow comparison cover every launch alike.

pub mod allocator;
pub mod error;

pub use allocator::Allocator;
pub use error::{Error, Result};
