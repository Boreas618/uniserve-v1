//! Work handed from one thread to another, whose submitter waits for it.
//!
//! The control thread hands a forward to its lane's execution thread, and
//! every thread hands its Python calls to the Python thread. In both cases
//! the work borrows the submitter's stack, so the submitter stays parked
//! until the receiving thread has run it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::Thread;

use crossbeam_channel::Sender;

/// One piece of work in a receiving thread's queue.
pub(crate) struct Handoff {
    work: *mut (dyn FnMut() + 'static),
    /// Set once `work` has returned; the submitter is parked until then.
    done: *const AtomicBool,
    submitter: Thread,
}

// SAFETY: the pointers address the stack of the submitter, which stays
// parked in `hand` until `done` is set and reads nothing before that. The
// state the work captures has no thread affinity: the rank's owners hold
// reference-counted TVM-FFI values with atomic counts, driver handles the
// driver accepts from any thread, and locks they do not hold across a
// handoff.
unsafe impl Send for Handoff {}

impl Handoff {
    /// Run the work on the calling thread and wake its submitter.
    pub(crate) fn run(self) {
        // SAFETY: see the `Send` implementation; the work is run once.
        unsafe { (*self.work)() };

        // The submitter's frame may be gone as soon as `done` is set, so
        // nothing of the handoff is read after the store.
        let submitter = self.submitter;
        // SAFETY: the submitter is still parked.
        unsafe { (*self.done).store(true, Ordering::Release) };
        submitter.unpark();
    }
}

/// Queue `work` on `queue` as the entry `entry` builds from it, and wait
/// until the receiving thread has run it. Returns false, without running
/// the work, when the queue's receiver is gone.
pub(crate) fn hand<E>(
    queue: &Sender<E>,
    entry: impl FnOnce(Handoff) -> E,
    work: &mut dyn FnMut(),
) -> bool {
    let done = AtomicBool::new(false);
    let handoff = Handoff {
        // SAFETY: the lifetime is erased only for the queue's type; this
        // function does not return before the work has run.
        work: unsafe {
            std::mem::transmute::<*mut (dyn FnMut() + '_), *mut (dyn FnMut() + 'static)>(work)
        },
        done: &raw const done,
        submitter: std::thread::current(),
    };
    if queue.send(entry(handoff)).is_err() {
        return false;
    }

    // `park` may return spuriously.
    while !done.load(Ordering::Acquire) {
        std::thread::park();
    }
    true
}
