//! The rank process: arguments, registration, descriptor, signals and exit
//! status.
//!
//! [`main`] is the body of the worker library's exported `main`. It runs on
//! the interpreter's main thread with the GIL released and returns the
//! process exit status:
//!
//! | condition | status |
//! | --- | --- |
//! | argument or descriptor error | 2, after a usage line |
//! | debug build of the library | 1 |
//! | the head cannot be reached or refuses the registration | 1 |
//! | SIGINT before serving | 130 |
//! | SIGTERM before serving | signal death |
//! | startup or serving failure, or the head's connection closing | 1, after an aborted close |
//! | close by the head, or after a fatal response | 0 |

mod arguments;
mod endpoint;
mod signals;

pub use arguments::{Arguments, USAGE};
pub use endpoint::Connection;
pub use signals::Signals;

use tvm_ffi::Map;

use crate::config::{DescriptorError, WorkerLaunch, resolve};
use crate::error::WorkerError;
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
    let (connection, descriptor) = match Connection::register(&arguments) {
        Ok(registered) => registered,
        Err(error) => {
            tracing::error!(target: "uniserve_worker.launch", "worker failed: {error}");
            return 1;
        }
    };
    let launch = match resolve(descriptor) {
        Ok(launch) => launch,
        Err(error) => return usage(error),
    };
    match run(launch, connection, table, &signals) {
        Ok(()) => {
            tracing::info!(target: "uniserve_worker.launch", "worker shut down");
            0
        }
        Err(error) => {
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

fn run(
    launch: WorkerLaunch,
    mut connection: Connection,
    table: Map<tvm_ffi::String, tvm_ffi::Function>,
    signals: &Signals,
) -> std::result::Result<(), WorkerError> {
    let interface = Interface::from_table(table)?;
    connection.configure(&launch.ipc);

    let mut worker = Worker::start(launch, interface)?;
    let served = worker.warmup().and_then(|()| {
        signals.serving();
        worker.run(&mut connection)
    });
    let closed = worker.close(served.is_err());
    // The worker's readers retire before the connection is released.
    drop(worker);
    drop(connection);
    served.and(closed)
}
