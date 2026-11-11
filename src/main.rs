//! `uniserve`: the one executable of every UniServe process.
//!
//! `main` parses the command line and runs its command: `serve` (a node's
//! daemon, and on the leader the application, [`serve`]), `engine` (a node's
//! main process for one epoch), `programs` (a deployment's recorded key list,
//! [`programs`]), and the host services `storage-server` and
//! `program-server`, whose commands are their crates' own. `serve` installs
//! tracing and builds the Tokio runtime; its module hands the resolved
//! configuration to `daemon::serve` with a cancellation token that fires on
//! Ctrl-C or SIGTERM. The host services run before any thread exists,
//! because a service started with `--daemon` forks.
//!
//! The runtime, the cancellation token and the supervision lifeline are the
//! executable's own: every command that starts threads takes them from here.

mod cli;
mod programs;
mod serve;

use std::env;

use anyhow::{Context, Result};
use service::logging;
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

/// Returns a cancellation token triggered by SIGINT (Ctrl-C) or SIGTERM.
///
/// The signals arrive through the one stop mechanism every `uniserve` process uses
/// (`service::stop_signals`): a self-pipe the runtime waits on. Must be called from within a Tokio
/// runtime, since it spawns the waiting task. When the pipe or its handlers cannot be installed,
/// that is logged and the token never fires.
pub(crate) fn shutdown_signal() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();
    let stopped = service::stop_signals().and_then(|(wake, _)| tokio::io::unix::AsyncFd::new(wake));
    match stopped {
        Ok(wake) => {
            tokio::spawn(async move {
                match wake.readable().await {
                    Ok(_) => {
                        info!("received a shutdown signal (SIGINT or SIGTERM), shutting down...")
                    }
                    Err(error) => {
                        warn!(%error, "waiting for the shutdown signals failed");
                        std::future::pending::<()>().await;
                    }
                }
                shutdown.cancel();
            });
        }
        Err(error) => warn!(%error, "failed to install the shutdown signal handlers"),
    }
    token
}

/// Starts the UniServe command-line process.
fn main() -> Result<std::process::ExitCode> {
    // A supervised process's standard error is its supervisor's pipe, and the supervisor that
    // started it may be gone: after a hot restart the successor adopts the process without its
    // predecessor's pipe, so a panic written only to standard error is invisible in the run's
    // log. Standard output is the log every process of the deployment shares, so a panic is
    // stated there as well - one line, before the process ends either way.
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        println!("a process panicked: {panic}");
        default(panic);
    }));
    let lifeline = take_lifeline();
    // The serving commands read which flags their command line states, so they keep its matches.
    let matches = <Cli as clap::CommandFactory>::command().get_matches();
    let cli = <Cli as clap::FromArgMatches>::from_arg_matches(&matches)
        .unwrap_or_else(|error| error.exit());
    let command_matches = || {
        matches
            .subcommand()
            .map(|(_, matches)| matches.clone())
            .unwrap_or_default()
    };
    match cli.command {
        Command::Serve(args) => serve::run(*args, &command_matches(), &cli.log),
        Command::Programs(args) => programs::run(*args, &command_matches(), &cli.log),
        Command::Engine { launch } => {
            follow_lifeline(lifeline);
            logging::init(
                "Engine",
                cli.log.log_level.as_deref(),
                cli.log.log_level_http.as_deref(),
                logging::Output::Stdout,
            );
            engine::run(&launch, &catalogue()?).map(|()| std::process::ExitCode::SUCCESS)
        }
        Command::StorageServer(args) => {
            storage_server::command::run(*args).map(|()| std::process::ExitCode::SUCCESS)
        }
        Command::ProgramServer(args) => program_server::command::run(*args, &host_targets)
            .map(|()| std::process::ExitCode::SUCCESS)
            .map_err(anyhow::Error::new),
    }
}

/// The architecture target of each device of this host, read from the driver: what a Program
/// Server this executable runs builds for when no `--target` names one.
fn host_targets() -> std::result::Result<Vec<program_server::Target>, String> {
    let count = platform::DeviceId::count()
        .map_err(|error| format!("counting this host's devices: {error}"))?;
    (0..count)
        .map(|ordinal| {
            platform::DeviceId(ordinal)
                .compute_capability()
                .map(program_server::Target::arch)
                .map_err(|error| format!("reading device {ordinal}'s architecture: {error}"))
        })
        .collect()
}

/// The ordinals of this host's idle devices, read through the driver and NVML: what the daemon
/// answers a deployment's planning with (`daemon::DeviceProbe`).
fn idle_devices() -> std::result::Result<Vec<u32>, String> {
    let nvml = platform::Nvml::init().map_err(|error| format!("opening NVML: {error}"))?;
    let count = platform::DeviceId::count()
        .map_err(|error| format!("counting this host's devices: {error}"))?;
    let mut idle = Vec::new();
    for ordinal in 0..count {
        let device = platform::DeviceId(ordinal);
        if nvml
            .is_idle(device)
            .map_err(|error| format!("reading device {ordinal}'s activity: {error}"))?
        {
            idle.push(ordinal as u32);
        }
    }
    Ok(idle)
}

/// The model families this executable serves: the one place a concrete model is named. Its main
/// processes load them and its daemon serves requests of them. A model is added with its crate and
/// one entry here.
const FAMILIES: [nn::family::Family; 3] = [
    qwen3::family::FAMILY,
    qwen3::family::HYBRID_FAMILY,
    minimax_h3::family::FAMILY,
];

/// The families this executable's main processes load ([`FAMILIES`]).
pub(crate) fn catalogue() -> Result<nn::family::Catalogue> {
    Ok(nn::family::Catalogue::new(FAMILIES)?)
}

/// What the daemon serves of [`FAMILIES`]: each family's request-processing side, which is all the
/// daemon reads of a model.
pub(crate) fn served() -> Result<processing::Registry> {
    Ok(processing::Registry::new(
        FAMILIES.map(|family| family.request),
    )?)
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
fn follow_lifeline(lifeline: Option<i32>) {
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
