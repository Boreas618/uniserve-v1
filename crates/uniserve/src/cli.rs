//! Command-line parsing for the `uniserve` binary.
//!
//! One executable runs every UniServe process, each as a command: `serve`
//! (the head), `launcher` (a host's launcher), `worker` (one worker) and
//! `storage` (a host's Storage Server). The binary's own options, its log
//! levels, are global; each command declares its own. `serve` takes the
//! serving command line (`server::options::ServeArgs`), which resolves one
//! serving configuration from a `--config` file, a few flags and `--set`
//! assignments. `worker` hands its arguments to the worker library
//! unparsed.

use std::ffi::OsString;

use clap::{Args, Parser, Subcommand};
use server::options::ServeArgs;

use crate::launcher::LauncherArgs;
use crate::storage::StorageArgs;

/// Top-level parser for the `uniserve` binary.
#[derive(Debug, Parser)]
#[command(
    name = "uniserve",
    about = "UniServe OpenAI-compatible server: Rust engine + scheduler, Python forwards only."
)]
pub(crate) struct Cli {
    #[command(flatten)]
    pub log: LogLevels,
    #[command(subcommand)]
    pub command: Command,
}

/// The process's log levels, which every subcommand takes.
#[derive(Debug, Clone, Args)]
#[command(next_help_heading = "Logging")]
pub(crate) struct LogLevels {
    /// Log level of UniServe components, such as `info` or `debug`.
    #[arg(long, global = true)]
    pub log_level: Option<String>,
    /// Log level of the HTTP server components.
    #[arg(long, global = true)]
    pub log_level_http: Option<String>,
}

/// Supported top-level CLI commands.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Run the UniServe OpenAI server: the Rust engine and scheduler run
    /// in-process, driving a forward-only worker.
    #[command(
        override_usage = "uniserve serve [MODEL] [OPTIONS] [--config FILE] [--set PATH=VALUE]..."
    )]
    Serve(Box<ServeArgs>),
    /// Launch and supervise the workers a head places on this host.
    Launcher(LauncherArgs),
    /// Run one worker: the process the head and the launchers start for
    /// each worker of a worker group.
    #[command(
        hide = true,
        disable_help_flag = true,
        override_usage = "uniserve worker --head ADDRESS --worker-id ID --rank R --generation G"
    )]
    Worker {
        /// The worker's arguments, which the worker library parses.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        arguments: Vec<OsString>,
    },
    /// The Storage Server of this host and the operator's commands against
    /// it.
    Storage(StorageArgs),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The log levels are the binary's global options, stated before the
    /// subcommand or after it.
    #[test]
    fn log_levels_are_global() {
        let levels = |args: &[&str]| {
            let cli = Cli::try_parse_from(args).expect("parses");
            (cli.log.log_level, cli.log.log_level_http)
        };
        assert_eq!(
            levels(&["uniserve", "--log-level", "debug", "serve", "m"]),
            (Some("debug".into()), None)
        );
        assert_eq!(
            levels(&["uniserve", "serve", "m", "--log-level-http", "warn"]),
            (None, Some("warn".into()))
        );
    }
}
