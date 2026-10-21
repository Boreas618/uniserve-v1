//! `uniserve`: the one executable of every UniServe process.
//!
//! `main` parses the command line and runs its command. `serve` installs
//! tracing and builds the Tokio runtime; `async_main` hands the resolved
//! configuration to `daemon::serve` together with a cancellation token that
//! fires on Ctrl-C or SIGTERM. `engine` runs the engine process the leader
//! daemon starts. `worker` and `storage` run before any thread exists: a worker loads the Python runtime on the main thread,
//! and a Storage Server started with `--daemon` forks.

mod cli;
mod logging;
mod storage;
mod worker;

use std::env;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use clap::Parser as _;
use daemon::options;

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
fn shutdown_signal() -> CancellationToken {
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
///
/// `serve --print-config` prints the resolved serving configuration, and
/// nothing else, to stdout and exits before tracing or the runtime starts.
fn main() -> Result<std::process::ExitCode> {
    let cli = Cli::parse();
    let args = match cli.command {
        Command::Serve(args) => args,
        Command::Engine(engine) => {
            logging::init_tracing(
                "Engine",
                cli.log.log_level.as_deref(),
                cli.log.log_level_http.as_deref(),
            );
            engine::run_engine(&engine.requests, &engine.processes, engine.engine)?;
            return Ok(std::process::ExitCode::SUCCESS);
        }
        Command::Worker { arguments } => worker::run(arguments),
        Command::Storage(args) => return Ok(storage::run(args)),
    };
    if args.print_config {
        let (config, _) = args.resolve()?;
        print!("{}", options::printed(&config)?);
        return Ok(std::process::ExitCode::SUCCESS);
    }

    logging::init_tracing(
        "Daemon",
        cli.log.log_level.as_deref(),
        cli.log.log_level_http.as_deref(),
    );

    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime.enable_all();
    // Chat-template rendering runs on Tokio's blocking pool. Complex model
    // templates require more than the platform's small pthread default.
    runtime.thread_stack_size(TOKIO_THREAD_STACK_BYTES);
    if let Some(worker_threads) = tokio_worker_threads() {
        runtime.worker_threads(worker_threads);
    }

    runtime
        .build()
        .context("failed to build Tokio runtime")?
        .block_on(async_main(*args))?;
    Ok(std::process::ExitCode::SUCCESS)
}

/// Serves the configuration `args` resolve until Ctrl-C or SIGTERM.
async fn async_main(args: options::ServeArgs) -> Result<()> {
    let (config, stated) = args.resolve()?;
    info!(model = %config.model(),
        resident_requests = config.max_running_requests,
        worker_executable = %config.worker_config.executable.display(), "resolved serving options");
    daemon::serve(config, stated, shutdown_signal()).await
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
