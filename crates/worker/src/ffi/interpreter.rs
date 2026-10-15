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
//!
//! The library is reached by import path ([`Interpreter::import`]) through
//! the one function the Python entry hands the rank,
//! `uniserve.worker.import_path`. There is no table of entries.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::Thread;

use crossbeam_channel::{Receiver, Sender, select, unbounded};
use tvm_ffi::{Any, AnyView};

use super::Function;
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

/// The function that resolves an import path to a Python callable,
/// installed once per process ([`Interpreter::install_import`]).
static IMPORT: OnceLock<Function> = OnceLock::new();

/// The thread that executes Python.
pub struct Interpreter {
    requests: Receiver<Request>,
}

impl Interpreter {
    /// Install the function through which every library function is
    /// reached by its import path ([`Self::import`]): the rank's entry
    /// installs the `import` argument of `main`, and a library loaded into
    /// another Python process (a harness) installs the same function,
    /// `uniserve.worker.import_path`. The first installation stays for the
    /// process; every later one names the same function.
    pub fn install_import(import: tvm_ffi::Function) {
        let _ = IMPORT.set(Function::new("uniserve.worker.import_path", import));
    }

    /// The Python callable at a dotted import path, for example
    /// `uniserve.runner.causal_lm.CausalLMRunner.decode`: a function, a
    /// class (called to construct it) or a method (called with the object
    /// first). Resolving imports the path's module on the Python thread.
    ///
    /// Fails when no import function was installed, when the path does not
    /// resolve (the library's error, which names the path) or when it names
    /// something that is not callable.
    pub fn import(path: &str) -> Result<Function> {
        let import = IMPORT
            .get()
            .ok_or(Error::State("no import function was installed"))?;
        let imported = import.call(&[(&Any::from(tvm_ffi::String::from(path))).into()])?;
        let function = tvm_ffi::Function::try_from(imported)
            .map_err(|_| Error::Interface(format!("`{path}` is not callable")))?;
        Ok(Function::new(path, function))
    }

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

// Work handed from one thread to another, whose submitter waits for it.
// Every thread hands its Python calls to the Python thread. The work borrows
// the submitter's stack, so the submitter stays parked until the receiving
// thread has run it.

/// One piece of work in a receiving thread's queue.
struct Handoff {
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
    fn run(self) {
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
fn hand<E>(queue: &Sender<E>, entry: impl FnOnce(Handoff) -> E, work: &mut dyn FnMut()) -> bool {
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
