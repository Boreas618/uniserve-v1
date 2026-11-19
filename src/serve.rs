//! `uniserve serve <package>`: a node's daemon.
//!
//! The package is a policy package (`daemon::policy::Package`), the deployment's whole
//! configuration; its directory's name is the deployment's. On the leader the daemon applies
//! the package as the deployment's first change, the start from nothing, and serves the
//! deployment's application: the HTTP and OpenAI API, input and output processing, and the admin
//! API that applies the next package. Every node's daemon supervises the node's processes.
//!
//! A daemon started as the successor of one this build replaces (`--predecessor FD`, which a
//! daemon restart passes) first takes the deployment over from the handoff socket at that
//! descriptor (`daemon::restart::take_over`); its package is its predecessor's.

use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context as _, Result};

use crate::cli::{LogLevels, ServeArgs};
use service::logging;

/// Runs the `serve` command with `args`, logging under `log`.
pub(crate) fn run(args: ServeArgs, log: &LogLevels) -> Result<ExitCode> {
    logging::init("Daemon", &log.serving(), logging::Output::Stdout);
    let runtime = crate::runtime()?;
    if let Some(descriptor) = args.predecessor {
        runtime.block_on(async move {
            daemon::restart::take_over(descriptor, crate::shutdown_signal()).await
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
    let directory = args
        .package
        .context("`serve` takes the policy package's directory")?;
    let (package, deployment) = read(&directory)?;
    runtime.block_on(async move {
        daemon::serve(package, deployment, ready, crate::shutdown_signal()).await
    })?;
    Ok(ExitCode::SUCCESS)
}

/// The package in `directory`, with the deployment it configures: the directory's name.
pub(crate) fn read(directory: &Path) -> Result<(daemon::policy::Package, String)> {
    let package = daemon::policy::Package::read(directory)?;
    let absolute = std::path::absolute(directory)
        .with_context(|| format!("resolving {}", directory.display()))?;
    let deployment = absolute
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| {
            format!(
                "the package's directory {} has no name to name the deployment by",
                absolute.display()
            )
        })?
        .to_owned();
    Ok((package, deployment))
}
