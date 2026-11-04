//! Command-line parsing for the `uniserve` binary.
//!
//! One executable runs every UniServe process, each as a command: `serve` (a
//! node's daemon, and on the leader the application), `frontend` (the
//! deployment's application process), `engine` (a node's main process for one
//! epoch), `storage` (a host's Storage Server), `program-server` (a host's
//! Program Server) and `worker` (one worker of the Python runtime the daemon
//! still starts). The binary's own options, its log levels, are global; each
//! command declares its own. `serve` and `frontend` take the serving command
//! line (`daemon::options::ServeArgs`), which resolves one configuration from
//! a `--config` file, a few flags and `--set` assignments. `worker` hands its
//! arguments to the worker library unparsed.

use std::ffi::OsString;

use clap::{Args, Parser, Subcommand};
use daemon::options::ServeArgs;

use crate::engine::EngineArgs;
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
    /// Run this node's daemon; on the leader (node rank 0), also the
    /// UniServe OpenAI server, whose Rust engine and scheduler run
    /// in-process and drive forward-only workers. Without a model, a
    /// single-node daemon serves its admin API alone until `PUT
    /// /admin/config` brings a deployment up.
    #[command(
        override_usage = "uniserve serve [MODEL] [OPTIONS] [--config FILE] [--set PATH=VALUE]..."
    )]
    Serve(Box<ServeArgs>),
    /// Run the deployment's application process: the HTTP and OpenAI API,
    /// input and output processing, facts and migration, on the leader node.
    /// The node's daemon starts it with the deployment's configuration.
    #[command(
        hide = true,
        override_usage = "uniserve frontend [MODEL] [OPTIONS] [--config FILE] [--set PATH=VALUE]..."
    )]
    Frontend(Box<ServeArgs>),
    /// Run the main process of this node and epoch: its schedulers and the
    /// request interface.
    #[command(hide = true)]
    Engine(EngineArgs),
    /// Run one worker: the process a node's daemon starts for each worker
    /// of a worker group placed on the node.
    #[command(
        hide = true,
        disable_help_flag = true,
        override_usage = "uniserve worker --engine ADDRESS --worker-id ID --rank R --generation G"
    )]
    Worker {
        /// The worker's arguments, which the worker library parses.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        arguments: Vec<OsString>,
    },
    /// The Storage Server of this host and the operator's commands against
    /// it.
    Storage(Box<StorageArgs>),
    /// A command whose parser belongs to another crate, such as the Program
    /// Server's: the executable hands the arguments to that parser instead of
    /// restating its surface here.
    #[command(hide = true, external_subcommand)]
    External(Vec<OsString>),
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
            (vec!["uniserve", "frontend", "model"], "frontend"),
            (vec!["uniserve", "storage", "list"], "storage"),
            (vec!["uniserve", "program-server"], "program-server"),
        ] {
            let cli = Cli::try_parse_from(args).expect("parses");
            let name = match cli.command {
                Command::Serve(_) => "serve",
                Command::Frontend(_) => "frontend",
                Command::Storage(_) => "storage",
                Command::External(arguments) => match arguments.first().and_then(|name| name.to_str()) {
                    Some("program-server") => "program-server",
                    _ => "other",
                },
                Command::Engine(_) | Command::Worker { .. } => "other",
            };
            assert_eq!(name, expected);
        }
    }
}
