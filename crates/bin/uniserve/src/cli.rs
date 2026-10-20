//! Command-line parsing for the `uniserve` binary.
//!
//! The binary's own options, its log levels, are global: every subcommand
//! takes them. Each subcommand declares its own options; `serve` takes the
//! serving command line (`uniserve_server::options::ServeArgs`), which
//! resolves one serving configuration from a `--config` file, a few flags
//! and `--set` assignments.

use clap::{Args, Parser, Subcommand};
use uniserve_server::options::ServeArgs;

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
