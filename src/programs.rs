//! `uniserve programs`: the programs a deployment's main processes launch, as the key list
//! `uniserve program-server build --keys` reads, so a store can be built for the deployment before
//! it serves.
//!
//! The configuration resolves as `uniserve serve` resolves it, and the launches are the ones a
//! serving leader starts its main processes with (`daemon::programs::launches`). Each launch runs
//! the main process's own start in-process up to the walk every device's runners record
//! (`engine::record`); the key list is the union of what every main process recorded, sorted and
//! each key once, written to standard output. Logs go to standard error.

use std::io::Write as _;
use std::process::ExitCode;

use anyhow::{Context as _, Result};

use crate::cli::LogLevels;
use service::logging;

/// Runs the `programs` command with the serving options `args`, logging under `log`.
pub(crate) fn run(
    args: daemon::options::ServerArgs,
    matches: &clap::ArgMatches,
    log: &LogLevels,
) -> Result<ExitCode> {
    logging::init(
        "Programs",
        log.log_level.as_deref(),
        log.log_level_http.as_deref(),
        logging::Output::Stderr,
    );
    anyhow::ensure!(
        args.predecessor.is_none(),
        "`--predecessor` hands a serving deployment to a successor daemon; `programs` serves none"
    );
    let (config, stated) = args.resolve(matches)?;
    config
        .validate(&stated)
        .context("invalid serving configuration")?;
    anyhow::ensure!(
        config.model.is_some(),
        "`programs` records a deployment's model: name MODEL on the command line or set `model` \
         in --config"
    );

    let catalogue = crate::catalogue()?;
    let launches =
        crate::runtime()?.block_on(daemon::programs::launches(&catalogue, &config, &stated))?;
    let mut recorded = engine::Recorded::new();
    for launch in &launches {
        merge(&mut recorded, engine::record(launch, &catalogue)?);
    }

    let list = program_server::build::key_list(&recorded)?;
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(list.as_bytes())
        .and_then(|()| stdout.flush())
        .context("writing the key list")?;
    Ok(ExitCode::SUCCESS)
}

/// Adds the programs `other` records to `recorded`: what the deployment's main processes launch
/// together.
fn merge(recorded: &mut engine::Recorded, other: engine::Recorded) {
    for (operator, keys) in other {
        recorded.entry(operator).or_default().extend(keys);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorded(entries: &[(&str, &[&str])]) -> engine::Recorded {
        entries
            .iter()
            .map(|(operator, keys)| {
                (
                    (*operator).to_owned(),
                    keys.iter().map(|key| (*key).to_owned()).collect(),
                )
            })
            .collect()
    }

    /// The deployment's programs are every main process's, each key once: a key two replicas
    /// launch is listed once, and an operator only one launches is kept.
    #[test]
    fn the_main_processes_programs_are_merged() {
        let mut merged = recorded(&[("rms_norm", &["a", "b"])]);
        merge(
            &mut merged,
            recorded(&[("rms_norm", &["b", "c"]), ("attention", &["d"])]),
        );
        assert_eq!(
            merged,
            recorded(&[("attention", &["d"]), ("rms_norm", &["a", "b", "c"])])
        );
    }
}
