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

pub mod error;
pub mod keys;
pub mod tensor;

pub use error::{Error, Failure, Result};

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
