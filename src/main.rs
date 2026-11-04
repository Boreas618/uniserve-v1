//! `uniserve`: the one executable of every UniServe process.
//!
//! `main` parses the command line and runs its command, one module each:
//! `serve` (a node's daemon, and on the leader the application), `frontend`
//! (the deployment's application process), `engine` (a node's main process
//! for one epoch), `storage` (a host's Storage Server), `program-server` (a
//! host's Program Server) and `worker` (one worker of the Python runtime the
//! daemon still starts). `serve` installs tracing and builds the Tokio
//! runtime; its module hands the resolved configuration to `daemon::serve`
//! with a cancellation token that fires on Ctrl-C or SIGTERM. `worker` and
//! `storage` run before any thread exists: a worker loads the Python runtime
//! on the main thread, and a Storage Server started with `--daemon` forks.
//!
//! The runtime, the cancellation token and the supervision lifeline are the
//! executable's own: every command that starts threads takes them from here.

mod cli;
mod engine;
mod frontend;
mod logging;
mod program_server;
mod serve;
mod storage;
mod worker;

use std::env;
use std::ffi::OsString;

use anyhow::{Context, Result};
use clap::Parser as _;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::cli::{Cli, Command};

// mimalloc is built with `no_thp` (workspace `Cargo.toml`): its arenas carry
// no transparent-huge-page advice, so khugepaged never collapses the engine's
// heap while the scheduler thread runs.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const TOKIO_WORKER_THREADS_ENV: &str = "TOKIO_WORKER_THREADS";
const DEFAULT_MAX_TOKIO_WORKER_THREADS: usize = 32;
const TOKIO_THREAD_STACK_BYTES: usize = 8 * 1024 * 1024;

/// Caps the default number of Tokio worker threads when `TOKIO_WORKER_THREADS` is unset.
///
/// The cap bounds scheduling overhead on machines with many logical CPUs.
/// Returns `None`, leaving the choice to Tokio's builder, when the variable
/// is set (the builder reads it itself) or when the available parallelism
/// cannot be queried.
fn tokio_worker_threads() -> Option<usize> {
    if env::var_os(TOKIO_WORKER_THREADS_ENV).is_some() {
        return None;
    }

    std::thread::available_parallelism()
        .map(|parallelism| {
            let available = parallelism.get();
            let worker_threads = available.min(DEFAULT_MAX_TOKIO_WORKER_THREADS);
            if worker_threads < available {
                info!(
                    available_parallelism = available,
                    capped_worker_threads = worker_threads,
                    "capping tokio worker threads, set {TOKIO_WORKER_THREADS_ENV} to override"
                );
            }
            worker_threads
        })
        .ok()
}

/// Returns a cancellation token triggered by Ctrl-C or SIGTERM.
///
/// Must be called from within a Tokio runtime, since it spawns the listening
/// task. A handler that fails to install is logged and never fires, leaving
/// the other signal able to cancel the token.
pub(crate) fn shutdown_signal() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();

    tokio::spawn(async move {
        let ctrl_c = async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                warn!(%error, "failed to install Ctrl-C signal handler");
                std::future::pending::<()>().await;
            }
        };

        let sigterm = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut signal) => {
                    signal.recv().await;
                }
                Err(error) => {
                    warn!(%error, "failed to install SIGTERM signal handler");
                    std::future::pending::<()>().await;
                }
            }
        };

        tokio::select! {
            _ = ctrl_c => info!("received shutdown signal (Ctrl-C), shutting down..."),
            _ = sigterm => info!("received shutdown signal (SIGTERM), shutting down..."),
        }

        shutdown.cancel();
    });

    token
}

/// Starts the UniServe command-line process.
fn main() -> Result<std::process::ExitCode> {
    let lifeline = take_lifeline();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => serve::run(*args, &cli.log),
        Command::Frontend(args) => frontend::run(*args, &cli.log),
        Command::Engine(args) => engine::run(&args, lifeline, &cli.log),
        Command::Worker { arguments } => {
            follow_lifeline(lifeline);
            worker::run(arguments)
        }
        Command::Storage(args) => Ok(storage::run(*args)),
        Command::External(arguments) => run_external(&arguments),
    }
}

/// Runs a command whose parser another crate owns, with the arguments the
/// top-level parser did not claim.
///
/// The Program Server's command line is the `program_server` crate's
/// (`command`, `options`, `run`), so this executable mounts it rather than
/// restating its surface. An unknown command is an error like any other.
fn run_external(arguments: &[OsString]) -> Result<std::process::ExitCode> {
    let Some(name) = arguments.first().and_then(|name| name.to_str()) else {
        anyhow::bail!("missing command");
    };
    match name {
        "program-server" => {
            let matches = program_server::command().get_matches_from(arguments);
            let options = program_server::options(&matches).map_err(anyhow::Error::new)?;
            program_server::run(options).map_err(anyhow::Error::new)?;
            Ok(std::process::ExitCode::SUCCESS)
        }
        other => anyhow::bail!("unknown command {other}"),
    }
}

/// The lifeline this process inherited from the daemon that supervises it
/// (`daemon::supervision::LIFELINE_ENV`): the read end of a pipe whose write
/// end the daemon holds. The variable leaves the environment and the
/// descriptor turns close-on-exec, so no process this one starts takes
/// either for its own. Runs before any thread starts.
fn take_lifeline() -> Option<i32> {
    let descriptor = std::env::var(daemon::supervision::LIFELINE_ENV)
        .ok()?
        .parse::<i32>()
        .ok()?;
    // SAFETY: no other thread exists yet to read the environment.
    unsafe { std::env::remove_var(daemon::supervision::LIFELINE_ENV) };
    // SAFETY: fcntl takes integers only; the descriptor is this process's.
    unsafe { libc::fcntl(descriptor, libc::F_SETFD, libc::FD_CLOEXEC) };
    Some(descriptor)
}

/// Ends this process once its daemon is gone: the end of the lifeline's
/// pipe arrives once the daemon, and every successor a hot restart handed
/// its processes to, ended, however they ended.
pub(crate) fn follow_lifeline(lifeline: Option<i32>) {
    let Some(descriptor) = lifeline else {
        return;
    };
    let watching = std::thread::Builder::new()
        .name("lifeline".to_owned())
        .spawn(move || {
            let mut byte = [0_u8; 1];
            loop {
                // SAFETY: reads into a local byte from this process's
                // descriptor.
                let read = unsafe { libc::read(descriptor, byte.as_mut_ptr().cast(), 1) };
                if read == 0 {
                    eprintln!("the daemon that supervised this process is gone; exiting");
                    // SAFETY: ends the process at once, as a signal would.
                    unsafe { libc::_exit(1) };
                }
                if read < 0
                    && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted
                {
                    return;
                }
            }
        });
    if let Err(error) = watching {
        eprintln!("the lifeline's watch did not start: {error}");
    }
}

/// The daemon's Tokio runtime.
pub(crate) fn runtime() -> Result<tokio::runtime::Runtime> {
    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime.enable_all();
    // Chat-template rendering runs on Tokio's blocking pool. Complex model
    // templates require more than the platform's small pthread default.
    runtime.thread_stack_size(TOKIO_THREAD_STACK_BYTES);
    if let Some(worker_threads) = tokio_worker_threads() {
        runtime.worker_threads(worker_threads);
    }
    runtime.build().context("failed to build Tokio runtime")
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    /// Returns the `VmFlags` of the mapping in `/proc/self/smaps` that holds
    /// `address`.
    fn mapping_flags(address: usize) -> Option<String> {
        let smaps = std::fs::read_to_string("/proc/self/smaps").ok()?;
        let mut inside = false;
        for line in smaps.lines() {
            let range = line
                .split_whitespace()
                .next()
                .and_then(|field| field.split_once('-'))
                .and_then(|(start, end)| {
                    Some((
                        usize::from_str_radix(start, 16).ok()?,
                        usize::from_str_radix(end, 16).ok()?,
                    ))
                });
            if let Some((start, end)) = range {
                inside = (start..end).contains(&address);
            } else if inside && let Some(flags) = line.strip_prefix("VmFlags:") {
                return Some(flags.trim().to_owned());
            }
        }
        None
    }

    /// The process heap is never advised for transparent huge pages
    /// (`MADV_HUGEPAGE`, flag `hg`), so khugepaged has nothing of it to
    /// collapse while the engine runs.
    #[test]
    fn the_heap_carries_no_huge_page_advice() {
        // Large enough to be served from an allocator arena rather than a
        // small-object page.
        let block = vec![1_u8; 64 << 20];
        let flags = mapping_flags(block.as_ptr() as usize).expect("the heap block is mapped");
        assert!(
            !flags.split_whitespace().any(|flag| flag == "hg"),
            "the heap mapping is advised for huge pages: {flags}"
        );
    }
}
