//! Transfer admission, transports, readable results, and physical retirement.
//!
//! A transfer moves Rust-owned storage between this main process and a peer:
//! `Spans` of `StorageView`s, ordered by the rank's `EventPool`. `Spans` and
//! the copy planning live here first; the transports, readable results,
//! export tables and cache import/export follow as they move from the retired
//! worker (`crates/worker/src/transfer`), with their operations unchanged.

pub mod spans;

pub use spans::Spans;
pub(crate) use spans::copy_on_host;

use std::sync::Arc;

use crate::completion::Completion;

/// A completion or readiness observer. Observers only notify (wake a
/// submission or the service loop); they never block or call back into the
/// owner that notifies them.
pub type Notify = Arc<dyn Fn() + Send + Sync>;

/// The physical retirement signal of an export, a read or a cache import.
pub type Retirement = Completion<Notify>;

/// Run the observers of a completion, after its owner unlocked its state.
pub fn notify(observers: Vec<Notify>) {
    for observer in observers {
        observer();
    }
}
