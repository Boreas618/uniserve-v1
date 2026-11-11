//! Command-line parsing for the `uniserve` binary.
//!
//! One executable runs every UniServe process, each as a command. The public
//! commands are `serve` (a node's daemon, and on the leader the application),
//! `keys` (the key list a deployment's main processes launch),
//! `storage-server` (a host's Storage Server) and `build-server` (a host's
//! Build Server); the host services' commands are their crates' own
//! (`storage_server::command`, `build_server::command`). `engine` (a node's
//! main process for one epoch) is started by the daemon and hidden from the
//! help. The binary's own options, its log levels, are global; each command
//! declares its own. `serve` and `keys` take the serving command line
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
    /// Print the modules the deployment's main processes launch, as the key
    /// list `uniserve build-server build --keys` reads (`operator<TAB>key`
    /// lines, sorted, each once), and exit. Takes `serve`'s options and starts
    /// the deployment's ranks on this node's devices as `serve` does, loading
    /// the model from its checkpoint's headers; no weight is read, and no
    /// daemon, Storage Server, communicator or Build Server is started.
    #[command(
        override_usage = "uniserve keys MODEL [OPTIONS] [--config FILE] [--set PATH=VALUE]..."
    )]
    Keys(Box<ServerArgs>),
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
    /// The Build Server of this host and the operator's commands against
    /// it.
    BuildServer(Box<build_server::command::Args>),
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
            (vec!["uniserve", "keys", "model"], "keys"),
            (vec!["uniserve", "storage-server", "list"], "storage-server"),
            (vec!["uniserve", "build-server", "list"], "build-server"),
        ] {
            let cli = Cli::try_parse_from(args).expect("parses");
            let name = match cli.command {
                Command::Serve(_) => "serve",
                Command::Keys(_) => "keys",
                Command::StorageServer(_) => "storage-server",
                Command::BuildServer(_) => "build-server",
                Command::Engine { .. } => "other",
            };
            assert_eq!(name, expected);
        }
    }

    /// `keys` takes `serve`'s options and resolves them to the same configuration, so the
    /// key list it prints is the one of the deployment `serve` starts with the same command line.
    #[test]
    fn keys_resolves_the_serving_configuration() {
        let resolve = |command: &str| {
            let matches = <Cli as clap::CommandFactory>::command().get_matches_from([
                "uniserve",
                command,
                "/models/qwen3",
                "--max-model-len",
                "2048",
                "--max-running-requests",
                "8",
                "--set",
                "dtype=bfloat16",
                "--set",
                "kv_cache_config.max_tokens=8192",
            ]);
            let cli = <Cli as clap::FromArgMatches>::from_arg_matches(&matches).expect("parses");
            let (_, command_matches) = matches.subcommand().expect("a command");
            let args = match cli.command {
                Command::Serve(args) | Command::Keys(args) => args,
                _ => panic!("{command} takes the serving options"),
            };
            args.resolve(command_matches)
                .expect("the options resolve")
                .0
        };
        let keys = resolve("keys");
        assert_eq!(keys, resolve("serve"));
        assert_eq!(keys.model.as_deref(), Some("/models/qwen3"));
        assert_eq!(keys.max_model_len.map(|length| length.get()), Some(2048));
    }
}
