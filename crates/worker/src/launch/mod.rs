//! The rank process: arguments, registration, descriptor, signals and exit
//! status.
//!
//! [`main`] is the body of the worker library's exported `main`. The thread
//! the Python entry calls it on becomes the rank's Python thread
//! ([`Interpreter`]): it serves the Python calls of every other thread until
//! the control thread, which [`main`] starts and which runs the rank, has
//! finished. [`main`] returns the process exit status:
//!
//! | condition | status |
//! | --- | --- |
//! | argument or descriptor error | 2, after a usage line |
//! | debug build of the library | 1 |
//! | the head cannot be reached or refuses the registration | 1 |
//! | SIGINT before serving | 130 |
//! | SIGTERM before serving | signal death |
//! | the head's connection closing before serving | 1, at once |
//! | startup or serving failure, or the head's connection closing while serving | 1, after an aborted close |
//! | close by the head, or after a fatal response | 0 |

mod arguments;
mod registration;
mod signals;

pub use arguments::{Arguments, USAGE};
pub use registration::{HeadWatch, register};
pub use signals::Signals;

use tvm_ffi::Map;

use crate::config::{DescriptorError, resolve};
use crate::error::WorkerError;
use crate::ffi::Interpreter;
use crate::model::interface::Interface;
use crate::worker::Worker;

/// Run the rank and return its exit status.
pub fn main(argv: &[String], table: Map<tvm_ffi::String, tvm_ffi::Function>) -> i64 {
    crate::logging::init();
    // Installed first: Python's own SIGINT handler would raise inside an
    // interface call.
    let signals = match Signals::install() {
        Ok(signals) => signals,
        Err(error) => {
            tracing::error!(target: "uniserve_worker.launch", "worker failed: {error}");
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
    if cfg!(debug_assertions) {
        tracing::error!(
            target: "uniserve_worker.launch",
            "libuniserve_worker is a debug build; install a release build"
        );
        return 1;
    }
    match run(&arguments, table, &signals) {
        Ok(()) => {
            tracing::info!(target: "uniserve_worker.launch", "worker shut down");
            0
        }
        Err(Failure::Descriptor(error)) => usage(error),
        Err(Failure::Worker(error)) => {
            match error.traceback() {
                Some(traceback) => tracing::error!(
                    target: "uniserve_worker.launch",
                    "worker failed: {error}\n{traceback}"
                ),
                None => tracing::error!(target: "uniserve_worker.launch", "worker failed: {error}"),
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

/// Run the rank on a control thread while the calling thread serves its
/// Python calls.
fn run(
    arguments: &Arguments,
    table: Map<tvm_ffi::String, tvm_ffi::Function>,
    signals: &Signals,
) -> std::result::Result<(), Failure> {
    let interface = Interface::from_table(table)?;
    let interpreter = Interpreter::start()?;
    let (finished, outcome) = crossbeam_channel::bounded(1);

    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("control".into())
            .spawn_scoped(scope, move || {
                // The receiver outlives this thread: the interpreter holds
                // it until this value arrives.
                let _ = finished.send(serve(arguments, interface, signals));
            })
            .map_err(|error| crate::Error::Transport(format!("control thread: {error}")))?;

        interpreter.run(&outcome).unwrap_or_else(|| {
            Err(crate::Error::State("the control thread ended without a result").into())
        })
    })
}

/// The rank's life on the control thread: register with the head, start,
/// warm up, serve and close.
fn serve(
    arguments: &Arguments,
    interface: Interface,
    signals: &Signals,
) -> std::result::Result<(), Failure> {
    let (mut channel, descriptor) = register(arguments)?;
    let launch = resolve(descriptor).map_err(Failure::Descriptor)?;
    channel.set_bound(launch.ipc.max_payload_bytes);

    // Until the service observes the connection, the head's close ends the
    // process from the watch's thread.
    let hangup = channel
        .hangup()
        .map_err(|error| crate::Error::Transport(error.to_string()))?;
    let mut starting = Some(HeadWatch::start(hangup)?);

    let mut worker = Worker::start(launch, interface)?;
    let served = worker.warmup().and_then(|()| {
        starting.take();
        signals.serving();
        worker.run(&mut channel)
    });
    // An orderly failure of the startup closes the rank below, not the watch.
    drop(starting);
    let closed = worker.close(served.is_err());
    // The worker's readers retire before the connection is released.
    drop(worker);
    drop(channel);
    Ok(served.and(closed)?)
}
