//! The Python thread of a rank.
//!
//! Exactly one thread of a rank process executes Python, so no thread ever
//! waits for the GIL. That thread is the one the Python entry called the
//! library's `main` on: it becomes the [`Interpreter`] and serves the calls
//! every other thread makes through [`super::Function`], one at a time, in
//! arrival order. A caller waits for its call's result; while it waits, the
//! values it passed stay borrowed on its stack. The last use of a Python
//! object ([`super::Handle`]'s drop) is handed over the same way.
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

/// The queue of the started interpreter. It outlives the interpreter: once
/// the interpreter has stopped, sends fail and callers call directly.
static REQUESTS: OnceLock<Sender<Handoff>> = OnceLock::new();

/// The thread that executes Python.
pub struct Interpreter {
    requests: Receiver<Handoff>,
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
                    request.expect("the request queue is never closed").run();
                }
                recv(finished) -> outcome => return outcome.ok(),
            }
        }
    }
}

/// Run `work` on the Python thread and wait for it, or on the calling
/// thread when that is where its Python runs (see the module documentation).
fn on_python_thread(work: &mut dyn FnMut()) {
    let handed = REQUESTS
        .get()
        .filter(|_| !CALLS_HERE.get())
        .is_some_and(|requests| hand(requests, |handoff| handoff, work));
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

/// Give up a reference to a Python object: drop it on the Python thread.
///
/// The caller waits, so the object is released at this point of the
/// caller's teardown order. A Python object can own device resources (a
/// graph pool frees its blocks and synchronizes the streams they were used
/// on), which must go before the streams they refer to.
pub(super) fn release(value: Any) {
    let mut value = Some(value);
    on_python_thread(&mut || drop(value.take()));
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread::ThreadId;

    use super::*;
    use crate::ffi::{Function, Handle};

    /// A function returning the id of the thread that executes it.
    fn executing_thread() -> (Function, Arc<std::sync::Mutex<Option<ThreadId>>>) {
        let seen = Arc::new(std::sync::Mutex::new(None));
        let sink = Arc::clone(&seen);
        let function = tvm_ffi::Function::from_packed(move |args: &[AnyView]| {
            *sink.lock().unwrap() = Some(std::thread::current().id());
            Ok(Any::from(i64::try_from(args[0])? + 1))
        });
        (Function::new("test.increment", function), seen)
    }

    /// The process has one interpreter, so one test covers its behavior:
    /// calls from other threads run on the interpreter thread and return
    /// their results; cooperative threads and the interpreter itself call
    /// in place; a handle dropped elsewhere is released on the interpreter.
    #[test]
    fn calls_and_releases_from_other_threads_run_on_the_interpreter_thread() {
        let (finished, done) = crossbeam_channel::bounded::<()>(1);
        let (started, ready) = crossbeam_channel::bounded(1);
        let interpreter = std::thread::spawn(move || {
            let interpreter = Interpreter::start().unwrap();
            started.send(std::thread::current().id()).unwrap();
            interpreter.run(&done)
        });
        let python = ready.recv().unwrap();

        let (function, seen) = executing_thread();
        let argument = Any::from(41_i64);
        assert_eq!(function.call_as::<i64>(&[(&argument).into()]).unwrap(), 42);
        assert_eq!(*seen.lock().unwrap(), Some(python));

        let here = std::thread::current().id();
        let value = cooperative(|| function.call_as::<i64>(&[(&argument).into()]));
        assert_eq!(value.unwrap(), 42);
        assert_eq!(*seen.lock().unwrap(), Some(here));

        // A handle dropped here is released by the interpreter before the
        // drop returns.
        let released = Arc::new(std::sync::Mutex::new(None));
        let sink = Arc::clone(&released);
        let witness = DropWitness(sink);
        let holder = tvm_ffi::Function::from_packed(move |_: &[AnyView]| {
            let _ = &witness;
            Ok(Any::from(()))
        });
        drop(Handle::new(Any::from(holder)).unwrap());
        assert_eq!(*released.lock().unwrap(), Some(python));

        finished.send(()).unwrap();
        assert_eq!(interpreter.join().unwrap(), Some(()));

        // With the interpreter stopped, a call runs on its caller's thread.
        assert_eq!(function.call_as::<i64>(&[(&argument).into()]).unwrap(), 42);
        assert_eq!(*seen.lock().unwrap(), Some(here));
    }

    /// Records the thread that drops it.
    struct DropWitness(Arc<std::sync::Mutex<Option<ThreadId>>>);

    impl Drop for DropWitness {
        fn drop(&mut self) {
            *self.0.lock().unwrap() = Some(std::thread::current().id());
        }
    }
}
