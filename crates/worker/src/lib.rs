//! Execution: how an engine runs a batch on a device.
//!
//! A scheduler assembles an [`ipc::execution::Batch`]; this crate executes it on a device and
//! returns its [`ipc::execution::BatchOutput`]. It sits above the numerical library
//! (`worker → nn → backend → ops → platform`) and dispatches by capability: a runner holds a
//! model through a trait of `nn::capability` and never names a concrete model.
//!
//! ## What execution owns
//!
//! A model computes on tensors and holds no runtime object. Everything else a forward needs is
//! this crate's:
//!
//! - the execution context a forward computes under, installed on the executor's thread around
//!   each model call (`backend::Context::enter`);
//! - the tensors a forward starts from: a runner's input tensors at fixed addresses, the weights
//!   as retained storage, and the cache planes and recurrent state as external tensors;
//! - the captured graphs, their buckets and the generations of kernels they launch;
//! - the token selection that ends a causal forward.
//!
//! ## The parts
//!
//! | Module | Holds |
//! | --- | --- |
//! | [`buckets`] | [`Bucket`](buckets::Bucket), the key that fixes a graph's shapes, and the bucket sets a runner captures |
//! | [`completion`] | How an operation ended and the completion its waiters share, over the crate's error |
//! | [`error`] | The crate's one error and its classification for the engine |
//! | [`graphs`] | [`Graphs`]: the profile run, capture, replay through the input buffer, and the switch of generations, for every runner |
//! | [`runner`] | The [`Runner`] interface a stage's runner implements |
//! | [`profiling`] | The range markers that time a model's modules, the engine's `backend::Profiler`; CUPTI's always-on counters; profiling sessions and their collectors and files |
//! | [`thread_pool`] | [`ThreadPool`](thread_pool::ThreadPool), a bounded pool of threads for one kind of blocking host work |
//!
//! ## Rules every module keeps
//!
//! - One executor thread issues all work of a device, and nothing but that thread touches the
//!   device's streams.
//! - No host callback is placed on a stream or in a graph: a replay's host values reach the
//!   device as input copies issued before the graph's launch.
//! - Whatever must wait for the device waits on a stream event, never by polling.
//! - Addresses a captured graph reads stay fixed for as long as the graph exists.
//! - The ranks of a group capture the same buckets in the same order.

pub mod buckets;
pub mod completion;
pub mod error;
pub mod graphs;
pub mod profiling;
pub mod runner;
pub mod thread_pool;

pub use error::{Error, Failure, Result};
pub use graphs::Graphs;
pub use runner::{Runner, RunnerId};

/// Locks `mutex`, keeping the state a panicking holder left.
///
/// Every build profile sets `panic = "abort"`, so a holder's panic ends the process before
/// another thread observes the poisoned lock. Only test builds, which ignore that setting and
/// unwind, reach the recovery; it returns the guard instead of raising a second panic in the
/// thread that locks next.
pub fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Keeps `value`, and every resource it owns, until the process exits.
///
/// An aborted close, or a release whose device completion is unknown, cannot tell when device
/// work, a DMA engine or a peer rank stops addressing what `value` owns: releasing it could give
/// storage, a mapping or a stream that is still in use to another owner. The value is leaked
/// instead, and process exit returns its resources to the driver.
#[expect(
    clippy::mem_forget,
    reason = "resources whose last device access is unknown must outlive it"
)]
pub(crate) fn leak<T>(value: T) {
    std::mem::forget(value);
}
