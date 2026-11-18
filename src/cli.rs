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
//! declares its own. `serve` and `keys` take a policy package, a directory
//! whose `config.toml` is the deployment's whole configuration
//! (`daemon::policy::Package`); no option comes from the command line.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use service::logging;
use tracing::level_filters::LevelFilter;

/// Top-level parser for the `uniserve` binary.
#[derive(Debug, Parser)]
#[command(name = "uniserve", about = "UniServe OpenAI-compatible server.")]
pub(crate) struct Cli {
    #[command(flatten)]
    pub log: LogLevels,
    #[command(subcommand)]
    pub command: Command,
}

/// The process's log levels, which every subcommand takes: tracing's level names (`off`,
/// `error`, `warn`, `info`, `debug`, `trace`), any other value refused.
#[derive(Debug, Clone, Args)]
#[command(next_help_heading = "Logging")]
pub(crate) struct LogLevels {
    /// Log level of every target, `info` when unset.
    #[arg(long, global = true)]
    pub log_level: Option<LevelFilter>,
    /// Log level of the daemon's HTTP stack (`daemon::http::LOG_TARGETS`), the
    /// `--log-level` when unset.
    #[arg(long, global = true)]
    pub log_level_http: Option<LevelFilter>,
}

impl LogLevels {
    /// The levels of a process that runs no HTTP stack.
    pub(crate) fn levels(&self) -> logging::Levels {
        logging::Levels::new(self.log_level)
    }

    /// The levels of a process that serves the daemon's HTTP stack.
    pub(crate) fn serving(&self) -> logging::Levels {
        self.levels()
            .with(daemon::http::LOG_TARGETS, self.log_level_http)
    }
}

/// Supported top-level CLI commands.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Serve the policy package in PACKAGE on this node: the node finds itself in the
    /// package's `hosts` by its host name; the leader (the first host, or this host when the
    /// package lists none) applies the package and serves the OpenAI API, and every other node
    /// joins it. While it serves, `PUT /admin/policy` applies another package.
    Serve(Box<ServeArgs>),
    /// Print the modules the main processes of the package in PACKAGE launch, as the key list
    /// `uniserve build-server build --keys` reads (`operator<TAB>key` lines, sorted, each
    /// once), and exit. Starts the deployment's ranks on this node's devices as `serve` does,
    /// loading the model from its checkpoint's headers; no weight is read, and no daemon,
    /// Storage Server or communicator is started.
    Keys(Box<KeysArgs>),
    /// The Storage Server of this host and the operator's commands against
    /// it.
    StorageServer(Box<storage_server::command::Args>),
    /// The Build Server of this host and the operator's commands against
    /// it.
    BuildServer(Box<build_server::command::Args>),
}

/// `uniserve serve`'s arguments: the package, and the descriptors a caller or a predecessor
/// passes, which are no option of the configuration.
#[derive(Debug, Args)]
pub(crate) struct ServeArgs {
    /// The policy package: a directory holding `config.toml` and, optionally, `programs/`.
    /// Its name is the deployment's, which names its persistent state on every node.
    #[arg(value_name = "PACKAGE", required_unless_present = "predecessor")]
    pub package: Option<PathBuf>,
    /// The inherited descriptor the daemon writes its readiness record to (one JSON line per
    /// state): once its listeners are bound (`listening`, with their endpoints) and once it
    /// serves (`serving`).
    #[arg(long, value_name = "FD")]
    pub ready_fd: Option<i32>,
    /// The descriptor of the handoff socket to the daemon this one succeeds, whose deployment
    /// it takes over, as a daemon restart starts its successor (`daemon::restart`).
    #[arg(long, hide = true, value_name = "FD")]
    pub predecessor: Option<i32>,
}

/// `uniserve keys`'s arguments.
#[derive(Debug, Args)]
pub(crate) struct KeysArgs {
    /// The policy package whose deployment's keys are recorded.
    #[arg(value_name = "PACKAGE")]
    pub package: PathBuf,
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
            (Some(LevelFilter::DEBUG), None)
        );
        assert_eq!(
            levels(&["uniserve", "serve", "m", "--log-level-http", "warn"]),
            (None, Some(LevelFilter::WARN))
        );
    }

    /// A level that is not one of tracing's names is refused, not read as another.
    #[test]
    fn an_unknown_log_level_is_refused() {
        for level in ["bogus", "warning", "critical"] {
            assert!(
                Cli::try_parse_from(["uniserve", "--log-level", level, "serve", "m"]).is_err(),
                "{level} was accepted"
            );
        }
    }

    /// Every process of the fixed base is one of this binary's commands.
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
            };
            assert_eq!(name, expected);
        }
    }

    /// `serve` takes a package and nothing else of the configuration: a serving option is no
    /// flag.
    #[test]
    fn serve_takes_a_package_alone() {
        let cli = Cli::try_parse_from(["uniserve", "serve", "qwen3"]).expect("parses");
        let Command::Serve(args) = cli.command else {
            panic!("serve parses as serve");
        };
        assert_eq!(args.package.as_deref(), Some(std::path::Path::new("qwen3")));
        for flag in ["--max-running-requests", "--config", "--set", "--port"] {
            assert!(
                Cli::try_parse_from(["uniserve", "serve", "qwen3", flag, "1"]).is_err(),
                "{flag} is no flag of serve"
            );
        }
        assert!(Cli::try_parse_from(["uniserve", "serve"]).is_err());
    }
}
