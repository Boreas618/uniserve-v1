//! The rank process: arguments, registration, descriptor, signals and exit
//! status.
//!
//! This module's `main` is the body of the worker library's exported
//! `main`. The thread the Python entry calls it on becomes the rank's Python
//! thread ([`Interpreter`]): it serves the Python calls of every other
//! thread until the control thread, which `main` starts and which runs the
//! rank, has finished. `main` returns the process exit status:
//!
//! | condition | status |
//! | --- | --- |
//! | argument or descriptor error | 2, after a usage line |
//! | the head cannot be reached or refuses the registration | 1 |
//! | SIGINT before serving | 130 |
//! | SIGTERM before serving | signal death |
//! | the head's connection closing before serving | 1, at once |
//! | startup or serving failure | 1, after an aborted close |
//! | the head's connection closing while serving with work in flight | 1, at once, from the control thread |
//! | the head's connection closing while serving with nothing in flight | 1, after an aborted close and the destruction of the rank's device contexts |
//! | close by the head, or after a fatal response | 0, after the rank's device contexts are destroyed |
//!
//! A rank that ends at once runs no teardown and does not destroy its
//! device contexts (`end`). Every rank that does not close in order with
//! its group removes the names of its CUDA driver's and NCCL's shared
//! memory before it ends (`release_names`).

mod arguments;
mod registration;
mod signals;

pub use arguments::Arguments;
pub(crate) use arguments::USAGE;
pub(crate) use registration::HeadWatch;
pub use registration::register;
pub(crate) use signals::Signals;

use crate::config::{DescriptorError, resolve};
use crate::error::WorkerError;
use crate::ffi::Interpreter;
use crate::service::Ending;
use crate::worker::Worker;

/// Run the rank and return its exit status.
pub(crate) fn main(argv: &[String], import: tvm_ffi::Function) -> i64 {
    crate::logging::init();
    // Installed first: Python's own SIGINT handler would raise inside a
    // library call.
    let signals = match Signals::install() {
        Ok(signals) => signals,
        Err(error) => {
            tracing::error!("worker failed: {error}");
            return 1;
        }
    };
    let usage = |error: DescriptorError| {
        eprintln!("{USAGE}\nuniserve-worker: error: {error}");
        2
    };
    let arguments = match Arguments::parse(argv) {
        Ok(arguments) => arguments,
        Err(error) => return usage(error),
    };
    match run(&arguments, import, &signals) {
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

/// Why a rank ended other than by an orderly close.
enum Failure {
    /// The head's launch descriptor names a launch this rank cannot run.
    Descriptor(DescriptorError),
    Worker(WorkerError),
}

impl<E: Into<WorkerError>> From<E> for Failure {
    fn from(error: E) -> Self {
        Self::Worker(error.into())
    }
}

/// Removes the names of the shared memory that this process's CUDA driver
/// and NCCL keep in `/dev/shm`, for a rank that ends without destroying its
/// communicators with its peers, or its device contexts.
///
/// The driver names its memory for the process
/// ([`uniserve_ipc::launch::remove_event_memory`]); NCCL's names are random,
/// and those still linked among this process's mappings are its own
/// ([`uniserve_ipc::launch::NcclSegments`]). Removing a name leaves every
/// mapping of the object, this process's and those its peers opened, valid
/// until they end. The removal is best effort: the rank ends whatever it
/// finds.
fn release_names() {
    // SAFETY: getuid and getpid have no preconditions.
    let (uid, pid) = unsafe { (libc::getuid(), libc::getpid()) };
    let _ = uniserve_ipc::launch::remove_event_memory(uid, pid as u32);
    if let Ok(segments) = uniserve_ipc::launch::NcclSegments::mapped_by(None) {
        let _ = segments.remove();
    }
}

/// Ends the process at once with `status`, running no teardown: for a rank
/// that cannot unwind from where its threads wait. Its device contexts and
/// communicators are not destroyed, so the names of their shared memory
/// are removed first ([`release_names`]).
pub(crate) fn end(status: i32) -> ! {
    release_names();
    // SAFETY: `_exit` ends the process without running any handler, which
    // is safe from any thread.
    unsafe { libc::_exit(status) }
}

/// Reports a failed startup to the head on the channel the rank registered
/// on, as the error that answers the head's request for its startup report.
/// The head names the rank and returns its error, and may end the rank's
/// group as soon as it has read it, before the rank logs the error itself.
/// A failure to send leaves the head to notice the closed connection.
fn report_startup_failure(channel: &mut uniserve_ipc::Channel, error: &WorkerError) {
    let response = uniserve_ipc::WorkerResponse::Error {
        message_id: None,
        error: error.response(),
    };
    let _ = channel.send(&response).and_then(|()| channel.flush());
}

/// How the rank's life on the control thread ended.
struct Ended {
    outcome: std::result::Result<(), Failure>,
    /// The worker closed in order with its group, which destroyed the
    /// communicators it shares with its peers.
    orderly: bool,
    /// Nothing the rank issued can still run on its devices, so its device
    /// contexts can be destroyed.
    settled: bool,
}

impl<E: Into<Failure>> From<E> for Ended {
    /// A failure before the rank served leaves nothing settled.
    fn from(error: E) -> Self {
        Self {
            outcome: Err(error.into()),
            orderly: false,
            settled: false,
        }
    }
}

/// Run the rank on a control thread while the calling thread serves its
/// Python calls, then destroy the rank's device contexts when nothing it
/// issued can still run on them.
fn run(
    arguments: &Arguments,
    import: tvm_ffi::Function,
    signals: &Signals,
) -> std::result::Result<(), Failure> {
    Interpreter::install_import(import);
    let interpreter = Interpreter::start()?;
    let (finished, outcome) = crossbeam_channel::bounded(1);

    let ended = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("control".into())
            .spawn_scoped(scope, move || {
                // The receiver outlives this thread: the interpreter holds
                // it until this value arrives.
                let _ = finished.send(serve(arguments, signals));
            })
            .map_err(|error| crate::Error::transport_source("control thread", error))?;

        Ok::<_, Failure>(interpreter.run(&outcome).unwrap_or_else(|| {
            crate::Error::State("the control thread ended without a result").into()
        }))
    })?;
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

    // The worker was closed and dropped, the control thread has finished
    // and this thread makes no Python call again, so nothing uses a device
    // any more, and nothing the rank issued still runs on one. PyTorch's
    // references keep every context alive until the process ends, and the
    // driver removes the shared memory behind a context's interprocess
    // events only when the context is destroyed, not when the process exits:
    // the contexts are destroyed here.
    // SAFETY: as above, nothing in the process uses a device after this.
    match unsafe { uniserve_cuda::PrimaryContext::reset_all() } {
        Ok(devices) => tracing::debug!(?devices, "destroyed the device contexts"),
        Err(error) => {
            let error = Failure::from(crate::Error::from(error));
            return ended.outcome.and(Err(error));
        }
    }
    ended.outcome
}

/// The rank's life on the control thread: register with the head, start,
/// warm up, serve and close.
///
/// A close by the head is collective: the group's ranks retire the
/// communicators and process groups they share together, so the close is
/// orderly. Without the head no rank learns what its peers do, and a peer
/// may have abandoned its work or stopped; an orderly close would wait on it
/// for good, so the rank closes without waiting on device work or peers. Its
/// devices are settled when the head closed it in order, or when the head's
/// channel failed while nothing was in flight. When work was in flight, the
/// process ends here at once.
fn serve(arguments: &Arguments, signals: &Signals) -> Ended {
    let (mut channel, descriptor) = match register(arguments) {
        Ok(registered) => registered,
        Err(error) => return error.into(),
    };
    let launch = match resolve(descriptor) {
        Ok(launch) => launch,
        Err(error) => return Failure::Descriptor(error).into(),
    };
    channel.set_bound(launch.ipc.max_payload_bytes);

    // Until the service observes the connection, the head's close ends the
    // process from the watch's thread.
    let hangup = match channel.hangup() {
        Ok(hangup) => hangup,
        Err(error) => {
            return crate::Error::transport_source("watching the head connection", error).into();
        }
    };
    let mut starting = match HeadWatch::start(hangup) {
        Ok(watch) => Some(watch),
        Err(error) => return error.into(),
    };

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
    // An orderly failure of the startup closes the rank below, not the watch.
    drop(starting);
    let (served, orderly, settled) = match served {
        Ok(Ending::Closed) => (Ok(()), true, true),
        Ok(Ending::Orphaned(error)) => (Err(error), false, true),
        Ok(Ending::Abandoned(error)) => {
            // The Python thread may wait in the abandoned work for good,
            // inside a device call behind a collective whose peer stopped,
            // and an aborted close then waits with it: the rank can neither
            // return through that thread nor close. The process ends from
            // here, as the head watch ends a starting rank.
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
