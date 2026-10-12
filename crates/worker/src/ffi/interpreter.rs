//! The Python thread of a rank.
//!
//! Exactly one thread of a rank process executes Python, so no thread ever
//! waits for the GIL. That thread is the one the Python entry called the
//! library's `main` on: it becomes the [`Interpreter`] and serves the calls
//! every other thread makes through [`super::Function`], one at a time, in
//! arrival order. A caller waits for its call's result; while it waits, the
//! values it passed stay borrowed on its stack.
//!
//! The control thread does not wait: it submits work that owns what it uses
//! ([`Interpreter::submit`]) and hears of it through what the work posts
//! when it ran. Submitted work is served in the same queue, in arrival
//! order with every other thread's calls. Giving up a Python object
//! ([`super::Handle`]'s drop) is such work on every thread: the reference
//! is dropped by the Python thread, in the order of the queue, and nobody
//! waits for it. An owner whose next step needs the releases made (closing
//! the streams or the context that released objects use) waits for the
//! queue once ([`Interpreter::synchronize`]).
//!
//! Three cases make a call on the calling thread instead:
//!
//! - the caller is the interpreter thread itself (a Python function called
//!   back into Rust, which calls another function);
//! - the caller runs a cooperative turn ([`cooperative`]): an eager expert
//!   forward stays suspended inside Python at its dispatch points, so each
//!   microbatch needs a thread of its own, and exactly one of them runs at
//!   a time;
//! - no interpreter was started: the library is used from Python (tests,
//!   harnesses), where the caller's thread already is the Python thread.

use std::cell::Cell;
use std::sync::OnceLock;

use crossbeam_channel::{Receiver, Sender, select, unbounded};
use tvm_ffi::{Any, AnyView};

use crate::handoff::{Handoff, hand};
use crate::{Error, Result};

thread_local! {
    /// Whether this thread makes its Python calls itself.
    static CALLS_HERE: Cell<bool> = const { Cell::new(false) };
}

/// One entry of the Python thread's queue.
enum Request {
    /// Work that borrows its submitter, which is parked until it ran.
    Handed(Handoff),
    /// Work that owns what it uses; nobody waits for it.
    Submitted(Box<dyn FnOnce() + Send>),
}

/// The queue of the started interpreter. It outlives the interpreter: once
/// the interpreter has stopped, sends fail and callers call directly.
static REQUESTS: OnceLock<Sender<Request>> = OnceLock::new();

/// The thread that executes Python.
pub struct Interpreter {
    requests: Receiver<Request>,
}

impl Interpreter {
    /// Make the calling thread the process's Python thread. From now on a
    /// [`super::Function`] called on another thread runs here, inside
    /// [`Self::run`].
    pub fn start() -> Result<Self> {
        let (sender, requests) = unbounded();
        REQUESTS
            .set(sender)
            .map_err(|_| Error::State("the Python thread was already started"))?;
        CALLS_HERE.set(true);
        Ok(Self { requests })
    }

    /// Serve calls until `finished` delivers a value, and return it.
    ///
    /// The sender of `finished` is the thread that drives the rank: it sends
    /// once, after its last call returned.
    #[expect(
        clippy::expect_used,
        reason = "the request queue's sender lives in a static"
    )]
    pub fn run<T>(self, finished: &Receiver<T>) -> Option<T> {
        loop {
            select! {
                recv(self.requests) -> request => {
                    match request.expect("the request queue is never closed") {
                        Request::Handed(handoff) => handoff.run(),
                        Request::Submitted(work) => work(),
                    }
                }
                recv(finished) -> outcome => return outcome.ok(),
            }
        }
    }

    /// Submit `work` to the Python thread and return without waiting for
    /// it. The work owns everything it uses and reports by a message of its
    /// own (a completion it posts), so its submitter goes on at once. Where
    /// the calling thread is the one its Python runs on (see the module
    /// documentation), the work runs before this returns.
    pub fn submit(work: impl FnOnce() + Send + 'static) {
        let work: Box<dyn FnOnce() + Send> = Box::new(work);
        let Some(requests) = REQUESTS.get().filter(|_| !CALLS_HERE.get()) else {
            work();
            return;
        };

        // The interpreter has stopped: the work comes back unsent.
        if let Err(unsent) = requests.send(Request::Submitted(work))
            && let Request::Submitted(work) = unsent.into_inner()
        {
            work();
        }
    }

    /// Wait until the Python thread has run everything submitted or handed
    /// to it before this call: every released object is released when this
    /// returns. For an owner about to close what released objects use; not
    /// for a thread that serves.
    pub fn synchronize() {
        on_python_thread(&mut || {});
    }
}

/// Run `work` on the Python thread and wait for it, or on the calling
/// thread when that is where its Python runs (see the module documentation).
fn on_python_thread(work: &mut dyn FnMut()) {
    let handed = REQUESTS
        .get()
        .filter(|_| !CALLS_HERE.get())
        .is_some_and(|requests| hand(requests, Request::Handed, work));
    if !handed {
        work();
    }
}

/// Call `function` on the Python thread and wait for its result.
#[expect(
    clippy::expect_used,
    reason = "the work stores the result before the handoff completes"
)]
pub(super) fn call(function: &tvm_ffi::Function, args: &[AnyView<'_>]) -> tvm_ffi::Result<Any> {
    let mut result = None;
    on_python_thread(&mut || result = Some(function.call_packed(args)));
    result.expect("a finished call has a result")
}

/// Give up a reference to a Python object: the Python thread drops it, in
/// the order of its queue, and the caller does not wait.
///
/// A Python object can own device resources (a graph pool frees its blocks
/// and synchronizes the streams they were used on), which must go before
/// the streams they refer to: whoever closes those waits for the queue
/// first ([`Interpreter::synchronize`]). A call made after a release runs
/// after it.
pub(super) fn release(value: Any) {
    let released = Released(value);
    Interpreter::submit(move || released.release());
}

/// A reference on its way to the Python thread.
struct Released(Any);

// SAFETY: the value is an immutable reference-counted TVM-FFI object with
// atomic counts, which Rust never dereferences; it is only moved to the
// Python thread and dropped there. A handle never holds a tensor.
unsafe impl Send for Released {}

impl Released {
    /// Drop the reference, on the thread that executes Python.
    fn release(self) {
        drop(self.0);
    }
}

/// Run `body` with this thread's Python calls made on this thread.
///
/// For the threads of eager expert microbatches only: their forwards take
/// turns explicitly and exactly one runs at a time, so they do not contend
/// for the GIL, and a forward suspended at a dispatch point keeps its
/// thread's Python frames.
pub fn cooperative<T>(body: impl FnOnce() -> T) -> T {
    let previous = CALLS_HERE.replace(true);
    let result = body();
    CALLS_HERE.set(previous);
    result
}
