//! The builder thread of a worker.
//!
//! Exactly one thread of a worker executes Python, so no thread ever waits
//! for the GIL. That thread is the `python` thread the worker's entry
//! spawns, which hosts the interpreter (`crate::launch`): it becomes the
//! [`Interpreter`] and serves the calls every other thread makes through
//! [`super::Function`], one at a time, in arrival order. A caller waits for its call's result; while it waits, the
//! values it passed stay borrowed on its stack.
//!
//! The control thread does not wait: it submits work that owns what it uses
//! ([`Interpreter::submit`]) and hears of it through what the work posts
//! when it ran. Submitted work is served in the same queue, in arrival
//! order with every other thread's calls. Giving up a Python object
//! ([`super::Handle`]'s drop) is such work on every thread: the reference
//! is dropped by the builder thread, in the order of the queue, and nobody
//! waits for it. An owner whose next step needs the releases made (closing
//! the streams or the context that released objects use) waits for the
//! queue once ([`Interpreter::synchronize`]).
//!
//! Two cases make a call on the calling thread instead:
//!
//! - the caller is the interpreter thread itself (a Python function called
//!   back into Rust, which calls another function);
//! - no interpreter was started: the library is used from Python (tests,
//!   harnesses), where the caller's thread already is the builder thread.
//!
//! The library is reached by import path ([`Interpreter::import`]) through
//! the one function `uniserve.worker.attach` registers,
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

/// One entry of the builder thread's queue.
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
    /// reached by its import path ([`Self::import`]): the worker's Python
    /// thread installs the global function `uniserve.worker.attach`
    /// registers, and a library loaded into another Python process (a
    /// harness) installs the same function, `uniserve.worker.import_path`. The first installation stays for the
    /// process; every later one names the same function.
    pub fn install_import(import: tvm_ffi::Function) {
        let _ = IMPORT.set(Function::new("uniserve.worker.import_path", import));
    }

    /// The Python callable at a dotted import path, for example
    /// `uniserve.executor.Executor.execute`: a function, a class (called to
    /// construct it) or a method (called with the object first). Resolving imports the path's module on the builder thread.
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

    /// Make the calling thread the process's builder thread. From now on a
    /// [`super::Function`] called on another thread runs here, inside
    /// [`Self::run`].
    pub fn start() -> Result<Self> {
        let (sender, requests) = unbounded();
        REQUESTS
            .set(sender)
            .map_err(|_| Error::State("the builder thread was already started"))?;
        CALLS_HERE.set(true);
        Ok(Self { requests })
    }

    /// Serve calls until `finished` delivers a value, and return it.
    ///
    /// The sender of `finished` is the thread that drives the worker: it sends
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

    /// Submit `work` to the builder thread and return without waiting for
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

    /// Wait until the builder thread has run everything submitted or handed
    /// to it before this call: every released object is released when this
    /// returns. For an owner about to close what released objects use; not
    /// for a thread that serves.
    pub fn synchronize() {
        on_builder_thread(&mut || {});
    }

    /// Run `work` on the builder thread and wait for its result, or on the
    /// calling thread when that is where its Python runs (see the module
    /// documentation).
    ///
    /// Every call `work` makes runs where `work` runs, without a handoff of
    /// its own. A call together with the Python work its arguments need,
    /// such as a view's first wrapping as a torch tensor
    /// ([`crate::memory::allocation::StorageView::handle`]), therefore costs
    /// one handoff, and every reference to a Python object that `work`
    /// drops is dropped on the builder thread.
    #[expect(
        clippy::expect_used,
        reason = "the work stores its result before the handoff completes"
    )]
    pub(crate) fn execute<T>(work: impl FnOnce() -> T) -> T {
        let mut work = Some(work);
        let mut result = None;
        on_builder_thread(&mut || result = work.take().map(|work| work()));
        result.expect("a finished handoff ran its work")
    }
}

/// Run `work` on the builder thread and wait for it, or on the calling
/// thread when that is where its Python runs (see the module documentation).
fn on_builder_thread(work: &mut dyn FnMut()) {
    let handed = REQUESTS
        .get()
        .filter(|_| !CALLS_HERE.get())
        .is_some_and(|requests| hand(requests, Request::Handed, work));
    if !handed {
        work();
    }
}

/// Call `function` on the builder thread and wait for its result.
pub(super) fn call(function: &tvm_ffi::Function, args: &[AnyView<'_>]) -> tvm_ffi::Result<Any> {
    Interpreter::execute(|| function.call_packed(args))
}

/// Give up a reference to a Python object: the builder thread drops it, in
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

/// A reference on its way to the builder thread.
struct Released(Any);

// SAFETY: the value is an immutable reference-counted TVM-FFI object with
// atomic counts, which Rust never dereferences; it is only moved to the
// builder thread and dropped there. A handle never holds a TVM-FFI tensor:
// a torch tensor it holds is an opaque Python object.
unsafe impl Send for Released {}

impl Released {
    /// Drop the reference, on the thread that executes Python.
    fn release(self) {
        drop(self.0);
    }
}

// Work handed from one thread to another, whose submitter waits for it.
// Every thread hands its Python calls to the builder thread. The work borrows
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
// state the work captures has no thread affinity: the worker's owners hold
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
