//! `uniserve engine`: a node's main process for one epoch.
//!
//! The daemon starts one main process per node and epoch. It takes its
//! request interface from the daemon's request socket and its process
//! supervision from the daemon's process socket, and runs this node's
//! schedulers (on the scheduler's node), its executors, and its part of the
//! model (`specs/worker.md` §2.1, §4).
//!
//! The command ends with its daemon: the lifeline the supervising daemon
//! gave this process ends when the daemon, and every daemon a hot restart
//! handed its processes to, ended.

use std::process::ExitCode;

use anyhow::Result;
use clap::Args;

use crate::cli::LogLevels;
use crate::logging;

/// Where an engine process reaches the daemon that started it.
#[derive(Debug, Clone, Args)]
pub(crate) struct EngineArgs {
    /// The leader daemon's request socket, `unix:<path>` or `host:port`.
    #[arg(long)]
    pub requests: String,
    /// The leader daemon's process socket, `unix:<path>` or `host:port`, through which the
    /// engine starts its workers.
    #[arg(long)]
    pub processes: String,
    /// The identity the daemon started the engine under.
    #[arg(long)]
    pub engine: u64,
}

/// Runs the `engine` command with its `args`, logging under `log`, and ends
/// with `lifeline`, the pipe of the daemon that supervises this process.
pub(crate) fn run(args: &EngineArgs, lifeline: Option<i32>, log: &LogLevels) -> Result<ExitCode> {
    crate::follow_lifeline(lifeline);
    logging::init_tracing(
        "Engine",
        log.log_level.as_deref(),
        log.log_level_http.as_deref(),
    );
    engine::run_engine(&args.requests, &args.processes, args.engine)?;
    Ok(ExitCode::SUCCESS)
}
