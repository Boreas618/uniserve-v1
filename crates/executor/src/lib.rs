//! The UniServe executor: the host half of the main process.
//!
//! The executor is the only thread that issues work to a device's streams, and
//! its owners hold everything a step borrows: the resident memory (`memory`),
//! the transfers that move state in and out of it, capacity, host groups, and
//! the keys of the state the Storage Server keeps (`keys`, `specs/worker.md`
//! §8). The runners, graphs, arenas and the program-layer update live in this
//! crate too, beside these modules.
//!
//! Ownership rules this crate keeps from the design:
//!
//! - Device storage is allocated through `platform`'s `Allocator`, which books
//!   every mapping against the device's budget before it maps (`§4.1`);
//! - a storage owner zeroes only storage created for it (§8);
//! - a parked (standby) owner keeps its mappings read-only and its promotion
//!   makes them writable again at the same addresses, so captured graphs stay
//!   valid (§8).

// The runner half of the design: the mechanism-to-executor crossing, captured
// graphs and their buckets, the graph pools, the device executor and its job
// queue, the shadow harness, and the runner of a causal language model
// (§4.4, §4.5, §7.3, §2.4).
pub mod arena;
pub mod batch;
pub mod captured;
pub mod device;
pub mod graph;
pub mod job;
pub mod plan;
pub mod profiling;
pub mod runner;
pub mod runners;
pub mod shadow;

pub mod blocking;
pub mod completion;
pub mod config;
pub mod error;
pub mod host;
pub mod hot_update;
pub mod keys;
pub mod memory;
pub mod parallel;
pub mod stream;
pub mod tensor;
pub mod transfer;
pub mod watchdog;

pub use completion::Outcome;
pub use error::{Error, Failure, Result};
pub use host::{HostPool, HostTask};

/// Lock `mutex`, keeping the state a panicking holder left.
///
/// Every build profile sets `panic = "abort"`, so a holder's panic ends the
/// process before another thread observes the poisoned lock. Only test builds,
/// which ignore that setting and unwind, reach the recovery; it returns the
/// guard rather than raising a second panic in the thread that locks next.
pub(crate) fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Keep `value`, and every resource it owns, until the process exits.
///
/// An aborted close, or a release whose device completion is unknown, cannot
/// tell when device work, a DMA engine or a peer rank stops addressing what
/// `value` owns: releasing it could hand storage, a mapping or a stream that is
/// still in use to a new owner. The value is leaked instead, and process exit
/// returns its resources to the driver.
#[expect(
    clippy::mem_forget,
    reason = "resources whose last device access is unknown must outlive it"
)]
pub(crate) fn leak<T>(value: T) {
    std::mem::forget(value);
}
