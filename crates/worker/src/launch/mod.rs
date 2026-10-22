//! The worker process: arguments, registration, descriptor, signals, the
//! builder thread and exit status.
//!
//! This module's `main` is the body of the library's exported
//! `worker_main`, which the `uniserve worker` command calls on the
//! process's main thread. It spawns the worker's builder thread, which hosts
//! the interpreter ([`python::host`]) and serves the Python calls of every
//! other thread ([`Interpreter`]), and runs the control thread's work on
//! the main thread. `main` returns the process exit status:
//!
//! | condition | status |
//! | --- | --- |
//! | argument or descriptor error | 2, after a usage line |
//! | the engine cannot be reached or refuses the registration | 1 |
//! | the interpreter does not start | 1, after reporting to the engine |
//! | SIGINT before serving | 130 |
//! | SIGTERM before serving | signal death |
//! | the engine's connection closing before serving | 1, at once |
//! | startup or serving failure | 1, after an aborted close |
//! | the engine's connection closing while serving with work in flight | 1, at once, from the control thread |
//! | the engine's connection closing while serving with nothing in flight | 1, after an aborted close and the destruction of the worker's device contexts |
//! | close by the engine, or after a fatal response | 0, after the worker's device contexts are destroyed |
//!
//! A worker that ends at once runs no teardown and does not destroy its
//! device contexts (`end`). Every worker that does not close in order with
//! its group removes the names of its CUDA driver's and NCCL's shared
//! memory before it ends (`release_names`).

mod arguments;
mod python;
mod registration;
mod signals;

use std::path::Path;

pub use arguments::Arguments;
pub(crate) use arguments::USAGE;
pub(crate) use registration::EngineWatch;
pub use registration::register;
pub(crate) use signals::Signals;

use crate::config::{DescriptorError, resolve};
use crate::error::WorkerError;
use crate::ffi::Interpreter;
use crate::service::Ending;
use crate::worker::Worker;

/// Run the worker and return its exit status. `python` is the environment's
/// interpreter, which the builder thread's interpreter takes as its
/// executable.
pub(crate) fn main(argv: &[String], python: &Path) -> i32 {
    crate::logging::init();
    // Installed first, before the interpreter exists: Python installs no
    // handlers of its own (`python::host`).
    let signals = match Signals::install() {
        Ok(signals) => signals,
        Err(error) => {
            tracing::error!("worker failed: {error}");
            return 1;
        }
    };
    let usage = |error: DescriptorError| {
        eprintln!("{USAGE}\nuniserve worker: error: {error}");
        2
    };
    let arguments = match Arguments::parse(argv) {
        Ok(arguments) => arguments,
        Err(error) => return usage(error),
    };
    match run(&arguments, python, &signals) {
        Ok(()) => {
            tracing::info!("worker shut down");
            0
        }
        Err(Failure::Descriptor(error)) => usage(error),
        Err(Failure::Worker(error)) => {
            match error.traceback() {
                Some(traceback) => tracing::error!("worker failed: {error}\n{traceback}"),
                None => tracing::error!("worker failed: {error}"),
            }
            1
        }
    }
}

/// Why a worker ended other than by an orderly close.
enum Failure {
    /// The engine's launch descriptor names a launch this worker cannot run.
    Descriptor(DescriptorError),
    Worker(WorkerError),
}

impl<E: Into<WorkerError>> From<E> for Failure {
    fn from(error: E) -> Self {
        Self::Worker(error.into())
    }
}

/// Removes the names of the shared memory that this process's CUDA driver
/// and NCCL keep in `/dev/shm`, for a worker that ends without destroying its
/// communicators with its peers, or its device contexts.
///
/// The driver names its memory for the process
/// ([`cuda::shm::remove_event_memory`]); NCCL's names are random,
/// and those still linked among this process's mappings are its own
/// ([`cuda::shm::NcclSegments`]). Removing a name leaves every
/// mapping of the object, this process's and those its peers opened, valid
/// until they end. The removal is best effort: the worker ends whatever it
/// finds.
fn release_names() {
    // SAFETY: getuid and getpid have no preconditions.
    let (uid, pid) = unsafe { (libc::getuid(), libc::getpid()) };
    let _ = cuda::shm::remove_event_memory(uid, pid as u32);
    if let Ok(segments) = cuda::shm::NcclSegments::mapped_by(None) {
        let _ = segments.remove();
    }
}

/// Ends the process at once with `status`, running no teardown: for a worker
/// that cannot unwind from where its threads wait. Its device contexts and
/// communicators are not destroyed, so the names of their shared memory
/// are removed first ([`release_names`]).
pub(crate) fn end(status: i32) -> ! {
    release_names();
    // SAFETY: `_exit` ends the process without running any handler, which
    // is safe from any thread.
    unsafe { libc::_exit(status) }
}

/// Reports a failed startup to the engine on the channel the worker registered
/// on, as the error that answers the engine's request for its startup report.
/// The engine names the worker and returns its error, and may end the
/// worker's group as soon as it has read it, before the worker logs the
/// error itself.
/// A failure to send leaves the engine to notice the closed connection.
pub(crate) fn report_startup_failure(channel: &mut ipc::Channel, error: &WorkerError) {
    let response = ipc::WorkerResponse::Error {
        message_id: None,
        error: error.response(),
    };
    let _ = channel.send(&response).and_then(|()| channel.flush());
}

/// How the worker's life on the control thread ended.
struct Ended {
    outcome: std::result::Result<(), Failure>,
    /// The worker closed in order with its group, which destroyed the
    /// communicators it shares with its peers.
    orderly: bool,
    /// Nothing the worker issued can still run on its devices, so its device
    /// contexts can be destroyed.
    settled: bool,
}

impl<E: Into<Failure>> From<E> for Ended {
    /// A failure before the worker served leaves nothing settled.
    fn from(error: E) -> Self {
        Self {
            outcome: Err(error.into()),
            orderly: false,
            settled: false,
        }
    }
}

/// Run the worker: start its builder thread, serve as the control thread on
/// the calling thread, then stop the builder thread and destroy the worker's
/// device contexts when nothing it issued can still run on them.
fn run(
    arguments: &Arguments,
    python: &Path,
    signals: &Signals,
) -> std::result::Result<(), Failure> {
    let (started, ready) = crossbeam_channel::bounded(1);
    let (finish, finished) = crossbeam_channel::bounded::<()>(1);
    let python = python.to_owned();
    let builder = std::thread::Builder::new()
        .name("builder".into())
        .spawn(move || {
            match python::host(&python).and_then(|()| Interpreter::start()) {
                Ok(interpreter) => {
                    // The receiver outlives this thread: the control thread
                    // waits for it before its first Python call.
                    let _ = started.send(Ok(()));
                    interpreter.run(&finished);
                }
                Err(error) => {
                    let _ = started.send(Err(error));
                }
            }
        })
        .map_err(|error| crate::Error::transport_source("builder thread", error))?;

    let ended = serve(arguments, signals, &ready);
    // The builder thread finishes the call it is serving, if any, and stops.
    // Only a settled worker waits for it, before it destroys its contexts:
    // any other ends the process with the thread wherever it is, as an
    // aborted worker must.
    let _ = finish.send(());
    if ended.settled {
        let _ = builder.join();
    }
    if !ended.orderly {
        // The communicators were not destroyed with the group's peers, and
        // the contexts are destroyed below only when settled.
        release_names();
    }
    if !ended.settled {
        // Destroying a context waits for the work queued in it, and work an
        // aborted close abandoned may never complete.
        return ended.outcome;
    }

    // The worker was closed and dropped and the builder thread has stopped,
    // so nothing uses a device any more, and nothing the worker issued still
    // runs on one. PyTorch's references keep every context alive until the
    // process ends, and the driver removes the shared memory behind a
    // context's interprocess events only when the context is destroyed, not
    // when the process exits: the contexts are destroyed here.
    // SAFETY: as above, nothing in the process uses a device after this.
    match unsafe { cuda::PrimaryContext::reset_all() } {
        Ok(devices) => tracing::debug!(?devices, "destroyed the device contexts"),
        Err(error) => {
            let error = Failure::from(crate::Error::from(error));
            return ended.outcome.and(Err(error));
        }
    }
    ended.outcome
}

/// The worker's life on the control thread: register with the engine, wait
/// for the builder thread (`ready`), start, warm up, serve and close.
///
/// A close by the engine is collective: the group's workers retire the
/// communicators and process groups they share together, so the close is
/// orderly. Without the engine no worker learns what its peers do, and a peer
/// may have abandoned its work or stopped; an orderly close would wait on it
/// for good, so the worker closes without waiting on device work or peers. Its
/// devices are settled when the engine closed it in order, or when the engine's
/// channel failed while nothing was in flight. When work was in flight, the
/// process ends here at once.
fn serve(
    arguments: &Arguments,
    signals: &Signals,
    ready: &crossbeam_channel::Receiver<crate::Result<()>>,
) -> Ended {
    let (mut channel, descriptor) = match register(arguments) {
        Ok(registered) => registered,
        Err(error) => return error.into(),
    };
    let launch = match resolve(descriptor, &arguments.worker_id, arguments.generation) {
        Ok(launch) => launch,
        Err(error) => return Failure::Descriptor(error).into(),
    };

    // Until the service observes the connection, the engine's close ends the
    // process from the watch's thread.
    let hangup = match channel.hangup() {
        Ok(hangup) => hangup,
        Err(error) => {
            return crate::Error::transport_source("watching the engine connection", error).into();
        }
    };
    let mut starting = match EngineWatch::start(hangup) {
        Ok(watch) => Some(watch),
        Err(error) => return error.into(),
    };

    // Every startup phase from here on calls Python.
    let hosted = ready
        .recv()
        .unwrap_or(Err(crate::Error::State(
            "the builder thread ended without starting",
        )))
        .map_err(WorkerError::from);
    if let Err(error) = hosted {
        report_startup_failure(&mut channel, &error);
        return error.into();
    }

    let stall_bound = launch.ipc.stall_bound;
    let mut worker = match Worker::start(launch) {
        Ok(worker) => worker,
        Err(error) => {
            report_startup_failure(&mut channel, &error);
            return error.into();
        }
    };
    let warmed = worker.warmup();
    if let Err(error) = &warmed {
        report_startup_failure(&mut channel, error);
    }
    let served = warmed.and_then(|()| {
        starting.take();
        signals.serving();
        worker.run(&mut channel, stall_bound)
    });
    // An orderly failure of the startup closes the worker below, not the watch.
    drop(starting);
    let (served, orderly, settled) = match served {
        Ok(Ending::Closed) => (Ok(()), true, true),
        Ok(Ending::Orphaned(error)) => (Err(error), false, true),
        Ok(Ending::Abandoned(error)) => {
            // The builder thread may wait in the abandoned work for good,
            // inside a device call behind a collective whose peer stopped,
            // and an aborted close then waits with it: the worker can neither
            // return through that thread nor close. The process ends from
            // here, as the engine watch ends a starting worker.
            tracing::error!("worker failed: {error}");
            end(1);
        }
        Err(error) => (Err(error), false, false),
    };
    // The worker's readers retire with it, before the connection is
    // released.
    let closed = worker.close(!orderly);
    drop(channel);
    // An orderly close that failed may have left work behind.
    let settled = settled && (!orderly || closed.is_ok());
    Ended {
        outcome: served.and(closed).map_err(Failure::from),
        orderly,
        settled,
    }
}
