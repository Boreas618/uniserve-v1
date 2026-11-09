//! Command-line parsing for the `uniserve` binary.
//!
//! One executable runs every UniServe process, each as a command. The public
//! commands are `serve` (a node's daemon, and on the leader the application),
//! `storage-server` (a host's Storage Server) and `program-server` (a host's
//! Program Server); the host services' commands are their crates' own
//! (`storage_server::command`, `program_server::command`). `engine` (a node's
//! main process for one epoch) is started by the daemon and hidden from the
//! help. The binary's own options, its log levels, are global; each command
//! declares its own. `serve` takes the serving command line
//! (`daemon::options::ServerArgs`), which resolves one configuration from a
//! `--config` file, a few flags and `--set` assignments.

use clap::{Args, Parser, Subcommand};
use daemon::options::ServerArgs;

/// Top-level parser for the `uniserve` binary.
#[derive(Debug, Parser)]
#[command(name = "uniserve", about = "UniServe OpenAI-compatible server.")]
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
    /// Run this node's daemon; on the leader (node rank 0), also the
    /// UniServe OpenAI server, whose Rust engine and scheduler run
    /// in-process and drive forward-only workers. Without a model, a
    /// single-node daemon serves its admin API alone until `PUT
    /// /admin/config` brings a deployment up.
    #[command(
        override_usage = "uniserve serve [MODEL] [OPTIONS] [--config FILE] [--set PATH=VALUE]..."
    )]
    Serve(Box<ServerArgs>),
    /// Run the main process of this node and epoch: its schedulers and the
    /// request interface.
    #[command(hide = true)]
    Engine {
        /// The launch file the daemon wrote for this engine process (`proto::config::EngineArgs`).
        #[arg(long)]
        launch: std::path::PathBuf,
    },
    /// The Storage Server of this host and the operator's commands against
    /// it.
    StorageServer(Box<storage_server::command::Args>),
    /// The Program Server of this host and the operator's commands against
    /// it.
    ProgramServer(Box<program_server::command::Args>),
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

    /// Every process of a deployment is one of this binary's commands, and
    /// the commands that the daemon starts are not in the help.
    #[test]
    fn each_process_is_a_command() {
        for (args, expected) in [
            (vec!["uniserve", "serve", "model"], "serve"),
            (vec!["uniserve", "storage-server", "list"], "storage-server"),
            (vec!["uniserve", "program-server", "list"], "program-server"),
        ] {
            let cli = Cli::try_parse_from(args).expect("parses");
            let name = match cli.command {
                Command::Serve(_) => "serve",
                Command::StorageServer(_) => "storage-server",
                Command::ProgramServer(_) => "program-server",
                Command::Engine { .. } => "other",
            };
            assert_eq!(name, expected);
        }
    }
}
