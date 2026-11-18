//! `uniserve`: the base executable of a deployment, which holds what an
//! update of the deployment does not replace: the daemon, the Storage Server
//! and the Build Server.
//!
//! The engine is not in it. A node's main process is the executable of the
//! engine package (`uniserve-engine`, `crates/engine`), which the Build Server
//! builds and the daemon starts, so this executable links neither the engine
//! nor a model.
//!
//! `main` parses the command line and runs its command: `serve` (a node's
//! daemon, and on the leader the application, [`serve`]) and a deployment's
//! servers `storage-server` and `build-server`, whose commands are their
//! crates' own.
//! `serve` installs tracing and builds the Tokio runtime; its module hands the
//! policy package it reads to `daemon::serve` with a cancellation token that
//! fires on Ctrl-C or SIGTERM. A deployment's servers run before any thread exists,
//! because a server started with `--daemon` forks.
//!
//! The runtime and the cancellation token are the executable's own: every
//! command that starts threads takes them from here.

mod cli;
mod serve;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::cli::{Cli, Command};

// mimalloc is built with `no_thp` (workspace `Cargo.toml`): its arenas carry
// no transparent-huge-page advice, so khugepaged never collapses the engine's
// heap while the scheduler thread runs.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
    let cli = <Cli as clap::Parser>::parse();
    match cli.command {
        Command::Serve(args) => serve::run(*args, &cli.log),
        Command::StorageServer(args) => storage_server::command::run(*args, &cli.log.levels())
            .map(|()| std::process::ExitCode::SUCCESS),
        Command::BuildServer(args) => {
            build_server::command::run(*args, &host_targets, &cli.log.levels())
                .map(|()| std::process::ExitCode::SUCCESS)
                .map_err(anyhow::Error::new)
        }
    }
}

/// The architecture target of each device of this host, read from the driver: what a Build
/// Server this executable runs builds for when no `--target` names one.
fn host_targets() -> std::result::Result<Vec<build_server::Target>, String> {
    let count = platform::DeviceId::count()
        .map_err(|error| format!("counting this host's devices: {error}"))?;
    (0..count)
        .map(|ordinal| {
            let capability = platform::DeviceId(ordinal)
                .compute_capability()
                .map_err(|error| format!("reading device {ordinal}'s architecture: {error}"))?;
            build_server::Target::arch(capability)
                .map_err(|error| format!("device {ordinal}: {error}"))
        })
        .collect()
}

/// What the daemon serves: the request-processing side of every model family
/// (`processing::FAMILIES`), which is all the daemon reads of a model. The
/// engine executable registers the numerical side of the same families.
pub(crate) fn served() -> Result<processing::Registry> {
    Ok(processing::Registry::new(processing::FAMILIES)?)
}

/// The daemon's Tokio runtime: one worker per CPU this process may run on (Tokio's default,
/// the available parallelism, which follows the process's CPU affinity and cgroup quota), at
/// Tokio's default thread stack.
pub(crate) fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime")
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use procfs::process::{MMapExtension, Process, VmFlags};

    /// The process heap is never advised for transparent huge pages
    /// (`MADV_HUGEPAGE`, the mapping flag `hg`), so khugepaged has nothing of it to
    /// collapse while the engine runs.
    #[test]
    fn the_heap_carries_no_huge_page_advice() {
        // Large enough to be served from an allocator arena rather than a
        // small-object page.
        let block = vec![1_u8; 64 << 20];
        let address = block.as_ptr() as u64;
        let maps = Process::myself()
            .and_then(|process| process.smaps())
            .expect("reading this process's mappings");
        let mapping = maps
            .into_iter()
            .find(|map| (map.address.0..map.address.1).contains(&address))
            .expect("the heap block is mapped");
        let MMapExtension { vm_flags, .. } = mapping.extension;
        assert!(
            !vm_flags.contains(VmFlags::HG),
            "the heap mapping is advised for huge pages: {vm_flags:?}"
        );
    }
}
