//! `uniserve serve`: a node's daemon.
//!
//! On the leader (node rank 0) the daemon also serves the deployment's
//! application: the HTTP and OpenAI API, input and output processing, facts
//! and migration, until the application moves to `uniserve frontend`
//! (`crates/frontend`). Every node's daemon supervises the node's processes
//! and holds the node's Storage Server claim.
//!
//! A daemon started as the successor of one this build replaces
//! (`--predecessor FD`, which `POST /admin/restart` passes) first takes the
//! deployment over from the handoff socket at that descriptor
//! (`daemon::restart::take_over`), and its options are the predecessor's.

use std::process::ExitCode;

use anyhow::Result;
use tracing::info;

use crate::cli::LogLevels;
use crate::logging;

/// Runs the `serve` command with the daemon's `args`, logging under `log`.
pub(crate) fn run(args: daemon::options::ServeArgs, log: &LogLevels) -> Result<ExitCode> {
    logging::init_tracing(
        "Daemon",
        log.log_level.as_deref(),
        log.log_level_http.as_deref(),
    );
    let runtime = crate::runtime()?;
    if let Some(descriptor) = args.predecessor {
        runtime.block_on(async move {
            daemon::restart::take_over(descriptor, crate::shutdown_signal()).await
        })?;
        return Ok(ExitCode::SUCCESS);
    }
    runtime.block_on(serve(args))?;
    Ok(ExitCode::SUCCESS)
}

/// Serves the configuration `args` resolve until Ctrl-C or SIGTERM.
async fn serve(args: daemon::options::ServeArgs) -> Result<()> {
    let (config, stated) = args.resolve()?;
    // A model-less leader serves its admin API until a deployment is
    // configured. Followers join the leader without loading a model.
    if config.leads() && config.model.is_none() {
        info!(
            host = %config.server_config.host,
            port = config.server_config.port,
            "serving no deployment until PUT /admin/config"
        );
        return daemon::node(config.server_config, crate::shutdown_signal()).await;
    }
    info!(model = %config.model(),
        resident_requests = config.max_running_requests,
        worker_executable = %config.worker_config.executable.display(), "resolved serving options");
    daemon::serve(config, stated, crate::shutdown_signal()).await
}
