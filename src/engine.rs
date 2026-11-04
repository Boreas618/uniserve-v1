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

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

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
    /// The Program Server's socket on this node, where the epoch's scheduling
    /// policy modules are resolved (design §5.3, §7.3). Without it a `Precopy`
    /// of a policy is refused and the epoch serves none.
    #[arg(long)]
    pub program_server: Option<PathBuf>,
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
    // The epoch's claims name no program store yet (`EngineSpec` has none), so
    // this is where a program's module comes from: the Program Server's client
    // on this node, which resolves a program's backend, revision and arguments
    // to the module it precompiled for this host.
    let policies: Option<Arc<dyn engine::PolicySource + Send + Sync>> =
        match args.program_server.as_deref() {
            Some(socket) => {
                tracing::info!(
                    socket = %socket.display(),
                    "a scheduling policy's module is resolved through this node's Program Server"
                );
                Some(Arc::new(engine::ProgramSource::connect(socket)?))
            }
            None => {
                tracing::info!(
                    "no Program Server is stated for this epoch: a policy precopy will be refused \
                     with the reason, and the epoch serves no policy"
                );
                None
            }
        };
    engine::run_engine(&args.requests, &args.processes, args.engine, policies)?;
    Ok(ExitCode::SUCCESS)
}
