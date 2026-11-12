//! The device backend of the tensor library.
//!
//! `uniserve-tensor` computes on the CPU by itself and reaches every other device through its
//! open backend traits. This crate implements them for UniServe: the CUDA device of a GPU and the
//! meta device, storage over memory the runtime owns, the generic primitives as launches of `ops`
//! kernels, and the execution context a forward computes under.
//!
//! ## What a model sees
//!
//! Tensors. A model computes with the tensor library's operations on tensors whose device is
//! [`cuda`] or [`meta`], and each operation is one launch of a generic primitive operator of
//! `ops` with an output from the context's allocator. A tensor is freed when its last reference
//! drops, which returns its block to the allocator it came from.
//!
//! ## What execution does
//!
//! The context is the thread's: execution builds one [`Context`] per runner and installs it
//! around a model call ([`Context::enter`]), and a model never holds, receives or names it. What
//! a forward does follows from the context and from the capture state of its stream:
//!
//! - under a context that launches, operations issue kernels on its stream, and while the stream
//!   is being captured they become the nodes of the graph;
//! - under a **meta context** nothing is launched. Every operation checks its operands, states
//!   the key of the kernel it would launch and takes its output from the meta context's
//!   allocator, so a profile run yields a forward's launch table and memory peak, and the
//!   forward of `prepare_inputs` writes a replay's host values into the input buffer at the
//!   offsets the capture recorded.
//!
//! Execution makes the tensors a forward starts from: [`retained`] over a weight, [`external`]
//! over memory it lends for the call (KV pages, state, latents, a graph's inputs), and
//! [`empty`] for a new tensor of the installed context. A tensor that has no data, the input of
//! a profile run or of a `prepare_inputs` forward, is [`retained`] or [`external`] over the
//! [`meta`] device: it takes no block of any allocator and its address is never read.
//!
//! ## What a fused operator does
//!
//! Every kernel reaches the device through [`launch()`], so kernel updates, profile runs, graph
//! rebinding and shadow comparison cover every launch alike. A fused operator of `nn` allocates
//! its output with [`empty`], states its arguments with [`operand`] and calls [`launch()`]. A
//! partitioned forward meets the other ranks of its group through [`collective`], a module
//! states the range its forward runs in through [`profiler::range`], and a forward states host
//! work between two of its launches through [`graph_break()`].

pub mod allocator;
pub mod breaks;
pub mod collective;
pub mod context;
mod device;
pub mod error;
mod input;
mod launch;
mod primitives;
pub mod profiler;
mod storage;

pub use allocator::Allocator;
pub use breaks::graph_break;
pub use collective::Group;
pub use context::{Context, ContextGuard};
pub use device::{cuda, meta};
pub use error::{Error, Result};
pub use launch::launch;
pub use profiler::Profiler;
pub use storage::{empty, external, operand, retained};
