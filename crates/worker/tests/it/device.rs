//! Exclusive use of the CUDA driver by this binary's device tests.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Serializes the device tests of this binary, whatever module holds them.
///
/// One test holds a stream in a host function to observe the work queued
/// behind it (`transports::block`). While a stream is held, a driver call
/// that waits for all of the context's queued work (`cuMemFree`,
/// `cuMemFreeHost`) cannot return, and it keeps driver locks that every
/// other call needs, including the calls the holding test makes before it
/// releases the stream. Every test of the binary shares the process's
/// context, and a pinned or device allocation any of them drops is freed on
/// the reclaim thread, so no other device test may run meanwhile: every
/// device test takes this lock first.
static DRIVER: Mutex<()> = Mutex::new(());

pub(crate) fn exclusive_driver() -> MutexGuard<'static, ()> {
    DRIVER.lock().unwrap_or_else(PoisonError::into_inner)
}
