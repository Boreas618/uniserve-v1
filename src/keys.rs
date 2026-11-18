//! `uniserve keys`: the kernel keys a deployment's main processes launch, as the key list
//! `uniserve build-server build --keys` reads, so a store can be built for the deployment before
//! it serves.
//!
//! The package is read as `uniserve serve` reads it, and the launches are the ones a serving
//! leader starts its main processes with. Each launch is run by the engine package the
//! deployment selects, up to the profile run of every device's runners (`daemon::keys::record`);
//! the key list is the union of what every main process recorded, sorted and each key once,
//! written to standard output. Logs go to standard error.

use std::io::Write as _;
use std::process::ExitCode;

use anyhow::{Context as _, Result};

use crate::cli::LogLevels;
use service::logging;

/// Runs the `keys` command for the package in `directory`, logging under `log`.
pub(crate) fn run(directory: &std::path::Path, log: &LogLevels) -> Result<ExitCode> {
    logging::init("Keys", &log.levels(), logging::Output::Stderr);
    let (package, _) = crate::serve::read(directory)?;
    package
        .config
        .validate(&package.stated)
        .context("invalid serving configuration")?;

    let recorded = crate::runtime()?.block_on(daemon::keys::record(
        &crate::served()?,
        &package.config,
        &package.stated,
    ))?;
    let list = build_server::build::key_list(&recorded)?;
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(list.as_bytes())
        .and_then(|()| stdout.flush())
        .context("writing the key list")?;
    Ok(ExitCode::SUCCESS)
}
