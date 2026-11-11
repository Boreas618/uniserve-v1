//! `uniserve keys`: the kernel keys a deployment's main processes launch, as the key list
//! `uniserve build-server build --keys` reads, so a store can be built for the deployment before
//! it serves.
//!
//! The configuration resolves as `uniserve serve` resolves it, and the launches are the ones a
//! serving leader starts its main processes with. Each launch is run by the engine package the
//! deployment selects, up to the profile run of every device's runners (`daemon::keys::record`);
//! the key list is the union of what every main process recorded, sorted and each key once,
//! written to standard output. Logs go to standard error.

use std::io::Write as _;
use std::process::ExitCode;

use anyhow::{Context as _, Result};

use crate::cli::LogLevels;
use service::logging;

/// Runs the `keys` command with the serving options `args`, logging under `log`.
pub(crate) fn run(
    args: daemon::options::ServerArgs,
    matches: &clap::ArgMatches,
    log: &LogLevels,
) -> Result<ExitCode> {
    logging::init(
        "Keys",
        log.log_level.as_deref(),
        log.log_level_http.as_deref(),
        logging::Output::Stderr,
    );
    anyhow::ensure!(
        args.predecessor.is_none(),
        "`--predecessor` hands a serving deployment to a successor daemon; `keys` serves none"
    );
    let (config, stated) = args.resolve(matches)?;
    config
        .validate(&stated)
        .context("invalid serving configuration")?;
    anyhow::ensure!(
        config.model.is_some(),
        "`keys` records a deployment's model: name MODEL on the command line or set `model` \
         in --config"
    );

    let recorded =
        crate::runtime()?.block_on(daemon::keys::record(&crate::served()?, &config, &stated))?;
    let list = build_server::build::key_list(&recorded)?;
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(list.as_bytes())
        .and_then(|()| stdout.flush())
        .context("writing the key list")?;
    Ok(ExitCode::SUCCESS)
}
