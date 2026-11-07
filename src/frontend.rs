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
//! resolves the same arguments `serve` does and hands the frontend the values
//! its application reads as plain data ([`frontend::options::Options`]). The
//! application itself lives in `crates/frontend` (delivery item 2); until its
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
pub(crate) fn run(args: daemon::options::ServeArgs, log: &LogLevels) -> Result<ExitCode> {
    logging::init_tracing(
        "Frontend",
        log.log_level.as_deref(),
        log.log_level_http.as_deref(),
    );
    let (config, stated) = args.resolve_for(daemon::options::Listen::Frontend)?;
    let options = options(&config)?;
    info!(
        model = %options.model,
        served_model_name = options.served_model_name(),
        bind = %format!("{}:{}", options.server_config.host, options.server_config.port),
        "resolved frontend options"
    );
    let runtime = crate::runtime()?;
    runtime.block_on(serve(options, stated, crate::shutdown_signal()))?;
    Ok(ExitCode::SUCCESS)
}

/// Serves the deployment's application until `shutdown` is cancelled.
async fn serve(
    _options: frontend::options::Options,
    _stated: daemon::options::Stated,
    _shutdown: CancellationToken,
) -> Result<()> {
    anyhow::bail!(
        "the frontend process does not serve yet: the application's routes are served inside \
         `uniserve serve` until crates/frontend takes them over (specs/worker.md §2.1)"
    )
}

/// The frontend's options, from the deployment's resolved configuration.
///
/// The deployment file is the daemon's schema and its one parser
/// (`daemon::options`); the sections the frontend reads are the same types the
/// daemon parses (`proto::config`), so each is handed over whole.
fn options(config: &daemon::options::ServeConfig) -> Result<frontend::options::Options> {
    let model = config
        .model
        .clone()
        .context("a deployment serves a model: `model` is required")?;
    Ok(frontend::options::Options {
        model,
        served_model_name: config.served_model_name.clone(),
        max_model_len: config.max_model_len.filter(|limit| limit.get() > 0),
        dtype: config.dtype,
        mm_processor_kwargs: config.mm_processor_kwargs.clone(),
        host_identity: config.host_identity.clone(),
        nnodes: config.nnodes,
        node_rank: config.node_rank,
        load_config: config.load_config.clone(),
        server_config: config.server_config.clone(),
        logging_config: config.logging_config.clone(),
        chat_config: config.chat_config.clone(),
        image_fetch_config: config.image_fetch_config.clone(),
        video_config: config.video_config.clone(),
        media_config: config.media_config.clone(),
        video_job_config: config.video_job_config.clone(),
        migration_config: config.migration_config.clone(),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    /// The deployment's values reach the frontend's options unchanged, and a
    /// deployment with no model is refused rather than served under a default
    /// name.
    #[test]
    fn the_frontend_options_carry_the_deployments_values() {
        let config = daemon::options::ServeConfig {
            model: Some("Qwen/Qwen3-30B-A3B".to_owned()),
            served_model_name: Some("qwen3".to_owned()),
            max_model_len: NonZeroU32::new(4096),
            host_identity: "node-0".to_owned(),
            ..daemon::options::ServeConfig::default()
        };
        let resolved = options(&config).expect("a deployment with a model resolves");
        assert_eq!(resolved.model, "Qwen/Qwen3-30B-A3B");
        assert_eq!(resolved.served_model_name(), "qwen3");
        assert_eq!(resolved.max_model_len, NonZeroU32::new(4096));
        assert_eq!(resolved.host_identity, "node-0");
        assert_eq!(resolved.server_config.port, config.server_config.port);
        assert_eq!(
            resolved.migration_config.max_migrations,
            config.migration_config.max_migrations
        );
        assert_eq!(resolved.media_config.ffprobe, config.media_config.ffprobe);

        let without_model = daemon::options::ServeConfig::default();
        assert!(
            options(&without_model).is_err(),
            "a deployment with no model is refused"
        );
    }
}
