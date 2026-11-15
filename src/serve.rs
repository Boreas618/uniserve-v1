//! `uniserve serve`: a node's daemon.
//!
//! On the leader (node rank 0) the daemon also serves the deployment's
//! application: the HTTP and OpenAI API, input and output processing, facts
//! and migration. Every node's daemon supervises the node's processes, holds
//! the node's Storage Server claim and makes sure its host's Build Server
//! serves.
//!
//! A daemon started as the successor of one this build replaces
//! (`--predecessor FD`, which `POST /admin/restart` passes) first takes the
//! deployment over from the handoff socket at that descriptor
//! (`daemon::restart::take_over`), and its options are the predecessor's.

use std::process::ExitCode;

use anyhow::{Context as _, Result};
use tracing::info;

use crate::cli::LogLevels;
use service::logging;

/// Runs the `serve` command with the daemon's `args`, logging under `log`.
pub(crate) fn run(
    args: daemon::options::ServerArgs,
    matches: &clap::ArgMatches,
    log: &LogLevels,
) -> Result<ExitCode> {
    logging::init("Daemon", &log.serving(), logging::Output::Stdout);
    let runtime = crate::runtime()?;
    if let Some(descriptor) = args.predecessor {
        runtime.block_on(async move {
            daemon::restart::take_over(descriptor, crate::served()?, crate::shutdown_signal()).await
        })?;
        return Ok(ExitCode::SUCCESS);
    }
    let ready = args
        .ready_fd
        // SAFETY: the caller passed this descriptor for the readiness record; nothing else in this
        // process owns it, and no thread has started yet.
        .map(|fd| unsafe { service::ReadyFd::inherit(fd) })
        .transpose()
        .context("taking the readiness descriptor")?;
    let resolved = args.resolve(matches)?;
    runtime.block_on(serve(resolved, ready))?;
    Ok(ExitCode::SUCCESS)
}

/// Serves the resolved configuration until Ctrl-C or SIGTERM, reporting its readiness to
/// `ready`: the deployment it names, or, without one, the deployments `PUT /admin/config` brings
/// up (`daemon::serve`).
async fn serve(
    (config, stated): (daemon::options::ServerArgs, daemon::options::Stated),
    ready: Option<service::ReadyFd>,
) -> Result<()> {
    info!(
        model = config.model(),
        node_rank = config.node_rank,
        "resolved serving options"
    );
    daemon::serve(
        config,
        stated,
        crate::served()?,
        ready,
        crate::shutdown_signal(),
    )
    .await
}
