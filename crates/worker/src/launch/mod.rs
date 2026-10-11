//! The rank process: arguments, descriptor, signals, endpoint and exit
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
//! | SIGINT before serving | 130 |
//! | SIGTERM before serving | signal death |
//! | startup or serving failure | 1, after an aborted close |
//! | close by the head, or after a fatal response | 0 |

mod arguments;
mod endpoint;
mod signals;

pub use arguments::{Arguments, USAGE};
pub use endpoint::Endpoint;
pub use signals::Signals;

use tvm_ffi::Map;
use uniserve_ipc::LaunchDescriptor;

use crate::config::{DescriptorError, WorkerLaunch, resolve};
use crate::error::WorkerError;
use crate::ffi::Interpreter;
use crate::model::interface::Interface;
use crate::worker::Worker;

/// Read and resolve the launch descriptor named on the command line.
pub fn configure<S: AsRef<str>>(argv: &[S]) -> Result<WorkerLaunch, DescriptorError> {
    let arguments = Arguments::parse(argv)?;
    let text = std::fs::read_to_string(&arguments.descriptor).map_err(|error| {
        DescriptorError::new(format!(
            "cannot read launch descriptor {}: {error}",
            arguments.descriptor.display()
        ))
    })?;
    let descriptor = LaunchDescriptor::from_json(&text)
        .map_err(|error| DescriptorError::new(format!("invalid launch descriptor: {error}")))?;
    resolve(descriptor)
}

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
    let launch = match configure(argv) {
        Ok(launch) => launch,
        Err(error) => {
            eprintln!("{USAGE}\nuniserve-worker: error: {error}");
            return 2;
        }
    };
    if cfg!(debug_assertions) {
        tracing::error!(
            target: "uniserve_worker.launch",
            "libuniserve_worker is a debug build; install a release build"
        );
        return 1;
    }
    match run(launch, table, &signals) {
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

/// Run the rank on a control thread while the calling thread serves its
/// Python calls.
fn run(
    launch: WorkerLaunch,
    table: Map<tvm_ffi::String, tvm_ffi::Function>,
    signals: &Signals,
) -> std::result::Result<(), WorkerError> {
    let interface = Interface::from_table(table)?;
    let interpreter = Interpreter::start()?;
    let (finished, outcome) = crossbeam_channel::bounded(1);

    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("control".into())
            .spawn_scoped(scope, move || {
                // The receiver outlives this thread: the interpreter holds
                // it until this value arrives.
                let _ = finished.send(serve(launch, interface, signals));
            })
            .map_err(|error| crate::Error::Transport(format!("control thread: {error}")))?;

        interpreter.run(&outcome).unwrap_or_else(|| {
            Err(crate::Error::State("the control thread ended without a result").into())
        })
    })
}

/// The rank's life on the control thread: bind, register, start, warm up,
/// serve and close.
fn serve(
    launch: WorkerLaunch,
    interface: Interface,
    signals: &Signals,
) -> std::result::Result<(), WorkerError> {
    let rank = launch.execution.rank as u32;
    let mut endpoint = Endpoint::bind(&launch.ipc, rank)?;
    endpoint.register(&launch.ipc, &launch.worker_id, rank)?;

    let mut worker = Worker::start(launch, interface)?;
    tracing::info!(target: "uniserve_worker.launch", "worker IPC endpoint bound");
    let served = worker.warmup().and_then(|()| {
        signals.serving();
        worker.run(&mut endpoint)
    });
    let closed = worker.close(served.is_err());
    // The worker's readers retire before the endpoint is released.
    drop(worker);
    drop(endpoint);
    served.and(closed)
}
