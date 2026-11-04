//! `uniserve program-server`: a host's Program Server and the operator's
//! commands against it.
//!
//! The Program Server compiles program packages, keeps them, and serves their
//! modules to the main processes (`specs/worker.md` §6). One runs on every
//! host, beside its Storage Server. Only the upstream Program Server — the
//! one whose package root is writable — accepts `install`, and it numbers the
//! revisions; the others mount that root read-only.
//!
//! The process itself is `crates/program-server`, which the programs swarm
//! builds. Until it lands this command says so instead of pretending to
//! serve.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::cli::LogLevels;
use crate::logging;

/// The `program-server` command's arguments.
#[derive(Debug, Parser)]
#[command(
    name = "program-server",
    about = "The Program Server of one host: compile, keep and serve programs"
)]
pub(crate) struct ProgramServerArgs {
    #[command(subcommand)]
    pub(crate) command: Option<ProgramServerCommand>,
    /// The socket to listen on; this host's default when unset.
    #[arg(long, value_name = "PATH")]
    pub(crate) socket: Option<PathBuf>,
    /// The package root this server writes: the upstream's, which alone
    /// accepts `install`; another server mounts it read-only.
    #[arg(long, value_name = "DIRECTORY")]
    pub(crate) packages: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ProgramServerCommand {
    /// Install a program package into the upstream's package root, which
    /// gives it a new revision; a later update picks it up.
    Install {
        /// The package directory holding its `package.toml`.
        #[arg(value_name = "DIRECTORY")]
        package: PathBuf,
    },
}

/// Runs the `program-server` command with its `args`, logging under `log`.
pub(crate) fn run(_args: ProgramServerArgs, log: &LogLevels) -> Result<ExitCode> {
    logging::init_tracing(
        "ProgramServer",
        log.log_level.as_deref(),
        log.log_level_http.as_deref(),
    );
    anyhow::bail!(
        "crates/program-server is not built in this tree yet; `uniserve program-server` \
         serves programs once it is (specs/worker.md §6)"
    )
}
