//! The Python thread of a rank: where calls, releases and submitted work of
//! other threads run. A process has one interpreter, so this file holds one test
//! and runs in a process of its own.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use tvm_ffi::{Any, AnyView};
use uniserve_worker::ffi::{Function, Handle, Interpreter, cooperative};

/// A function that increments its argument and records the thread that
/// executes it.
fn executing_thread() -> (Function, Arc<Mutex<Option<ThreadId>>>) {
    let seen = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&seen);
    let function = tvm_ffi::Function::from_packed(move |args: &[AnyView]| {
        *sink.lock().unwrap() = Some(std::thread::current().id());
        Ok(Any::from(i64::try_from(args[0])? + 1))
    });
    (Function::new("test.increment", function), seen)
}

/// Records the thread that drops it.
struct DropWitness(Arc<Mutex<Option<ThreadId>>>);

impl Drop for DropWitness {
    fn drop(&mut self) {
        *self.0.lock().unwrap() = Some(std::thread::current().id());
    }
}

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

    // A cooperative turn calls in place.
    let here = std::thread::current().id();
    let value = cooperative(|| function.call_as::<i64>(&[(&argument).into()]));
    assert_eq!(value.unwrap(), 42);
    assert_eq!(*seen.lock().unwrap(), Some(here));

    // A handle dropped here is released by the interpreter before the drop
    // returns.
    let released = Arc::new(Mutex::new(None));
    let witness = DropWitness(Arc::clone(&released));
    let holder = tvm_ffi::Function::from_packed(move |_: &[AnyView]| {
        let _ = &witness;
        Ok(Any::from(()))
    });
    drop(Handle::new(Any::from(holder)).unwrap());
    assert_eq!(*released.lock().unwrap(), Some(python));

    // Submitted work runs on the interpreter thread while its submitter
    // goes on: here the submitter reaches the gate only after `submit`
    // returned, and the work cannot end before the gate opens.
    let limit = std::time::Duration::from_secs(30);
    let (open, gate) = crossbeam_channel::bounded::<()>(1);
    let (ran, result) = crossbeam_channel::bounded(1);
    Interpreter::submit(move || {
        gate.recv_timeout(limit).unwrap();
        ran.send(std::thread::current().id()).unwrap();
    });
    open.send(()).unwrap();
    assert_eq!(result.recv_timeout(limit).unwrap(), python);

    finished.send(()).unwrap();
    assert_eq!(interpreter.join().unwrap(), Some(()));

    // With the interpreter stopped, a call and submitted work run on their
    // caller's thread.
    assert_eq!(function.call_as::<i64>(&[(&argument).into()]).unwrap(), 42);
    assert_eq!(*seen.lock().unwrap(), Some(here));
    let (ran, result) = crossbeam_channel::bounded(1);
    Interpreter::submit(move || ran.send(std::thread::current().id()).unwrap());
    assert_eq!(result.try_recv().unwrap(), here);
}
