//! `uniserve frontend`: the deployment's application process.
//!
//! The frontend serves the HTTP and OpenAI API, processes requests' input and
//! output, keeps the facts of the requests in flight, migrates them between
//! epochs, and hands its own streams and their connections to a successor on
//! a hot restart (`specs/executor.md` §2.1, §7.5.1). It takes the deployment's
//! configuration like `serve`: the options it reads are the host layer's, and
//! the daemon starts it with the same file, flags and `--set` assignments the
//! deployment was configured with (§7.7).
//!
//! The deployment file has one parser, and it is the daemon's: this command
//! resolves the same arguments `serve` does and hands the frontend the
//! resolved configuration, whose sections are the types every process reads
//! (`proto::config`). The application itself lives in `crates/frontend`; until its
//! routes move there in one cut, `serve` serves them inside the daemon, and
//! this command says so instead of starting a second copy of them.

use std::process::ExitCode;

use anyhow::Context as _;
use anyhow::Result;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::cli::LogLevels;
use crate::logging;

/// Runs the `frontend` command with the deployment's `args`, logging under
/// `log`.
pub(crate) fn run(
    args: daemon::options::ServerArgs,
    matches: &clap::ArgMatches,
    log: &LogLevels,
) -> Result<ExitCode> {
    logging::init_tracing(
        "Frontend",
        log.log_level.as_deref(),
        log.log_level_http.as_deref(),
    );
    let (config, stated) = args.resolve(matches, daemon::options::Listen::Frontend)?;
    config
        .model
        .as_deref()
        .context("a deployment serves a model: `model` is required")?;
    info!(
        model = %config.model(),
        served_model_name = config.served_model_name(),
        bind = %format!("{}:{}", config.server_config.host, config.server_config.port),
        "resolved frontend options"
    );
    let runtime = crate::runtime()?;
    runtime.block_on(serve(config, stated, crate::shutdown_signal()))?;
    Ok(ExitCode::SUCCESS)
}

/// Serves the deployment's application until `shutdown` is cancelled.
async fn serve(
    _config: daemon::options::ServerArgs,
    _stated: daemon::options::Stated,
    _shutdown: CancellationToken,
) -> Result<()> {
    anyhow::bail!(
        "the frontend process does not serve yet: the application's routes are served inside \
         `uniserve serve` until crates/frontend takes them over (specs/worker.md §2.1)"
    )
}
