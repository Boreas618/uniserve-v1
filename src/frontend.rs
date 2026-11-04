//! `uniserve frontend`: the deployment's application process.
//!
//! The frontend serves the HTTP and OpenAI API, processes requests' input and
//! output, keeps the facts of the requests in flight, migrates them between
//! epochs, and hands its own streams and their connections to a successor on
//! a hot restart (`specs/worker.md` §2.1, §7.5.1). It takes the deployment's
//! configuration like `serve`: the options it reads are the host layer's, and
//! the daemon starts it with the same file, flags and `--set` assignments the
//! deployment was configured with (§7.7).
//!
//! The process is `crates/frontend`, which takes the application out of the
//! daemon (delivery item 2). Until that crate lands the application serves
//! inside `uniserve serve`, and this command says so instead of starting a
//! second copy of it.

use std::process::ExitCode;

use anyhow::Result;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::cli::LogLevels;
use crate::logging;

/// Runs the `frontend` command with the deployment's `args`, logging under
/// `log`.
pub(crate) fn run(args: daemon::options::ServeArgs, log: &LogLevels) -> Result<ExitCode> {
    logging::init_tracing(
        "Frontend",
        log.log_level.as_deref(),
        log.log_level_http.as_deref(),
    );
    let (config, stated) = args.resolve()?;
    info!(model = %config.model(), "resolved frontend options");
    let runtime = crate::runtime()?;
    runtime.block_on(serve(config, stated, crate::shutdown_signal()))?;
    Ok(ExitCode::SUCCESS)
}

/// Serves the deployment's application until `shutdown` is cancelled.
async fn serve(
    _config: daemon::options::ServeConfig,
    _stated: daemon::options::Stated,
    _shutdown: CancellationToken,
) -> Result<()> {
    anyhow::bail!(
        "the frontend process is not built in this tree yet: the application serves inside \
         `uniserve serve` until crates/frontend lands (specs/worker.md §2.1)"
    )
}
