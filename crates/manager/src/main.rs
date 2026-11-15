//! `uniserve-manager`: an optional manager of UniServe deployments across
//! nodes.
//!
//! The manager is a client of each node's admin API (`/admin`) and nothing
//! else: no daemon, engine, worker or Storage Server waits for it or keeps
//! state of it. It reads and writes the interface crate's admin schema
//! (`proto::admin`), the one the daemons serve. A node's admin API listens on a Unix socket
//! whose file permissions are its access (`unix:<path>`); an `http://` address names an
//! admin socket forwarded from another host. It attaches to a node by reading its status
//! (`GET /admin/status`, which states whether the node is idle, starting, serving or stopping)
//! and its event log, and detaches by stopping; a write it sends is carried out by the
//! node whether the manager stays or not. Every command acts on the nodes
//! `--node` names, in the order given:
//!
//! - `status`: what each node serves.
//! - `watch`: follows each node's event log until interrupted (attach and
//!   detach).
//! - `config`: each node's configuration status, as the node answers it.
//! - `apply`: changes each node's configuration (`PUT /admin/config`): an
//!   idle node (`uniserve serve` without a model) brings the deployment
//!   up, a serving one changes it in place or in a new epoch on trial,
//!   which `--commit-after` commits once it served that long. Each node is
//!   one service's leader: the manager decides the order of their updates
//!   from the GPUs their epochs hold ([`updates`]), so that two services that share a GPU
//!   never prepare at once.
//! - `commit`, `rollback`: end each node's trial.
//! - `shutdown`: brings each node's deployment down, releasing its storage
//!   unless `--keep-storage`.
//! - `restart`: a rolling hot restart of each node's daemon, one node after
//!   another, stopping at the first that fails.
//! - `pause`, `continue`, `flush-cache`: the generation and cache operations
//!   of each node.
//!
//! What the manager waits for, the end of an operation a node runs on its own and the end of
//! a trial, it reads from the node's event log by long poll (`GET /admin/events`).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use proto::admin::{
    ApplyRequest, InFlight, NodeState, OperationResponse, Paused, Restart, Restarted,
    RollbackRequest, Shutdown, UpdateReport,
};
use serde::de::DeserializeOwned;

use node::{Node, Polling};

mod node;
mod updates;

/// The manager's command line.
#[derive(Debug, Parser)]
#[command(
    name = "uniserve-manager",
    about = "Manage UniServe deployments through their nodes' admin API"
)]
struct Cli {
    /// A node's admin API: its Unix socket, as its daemon's readiness record states it
    /// (`unix:/run/user/1000/uniserve/daemon.Ab12Cd/admin.sock`),
    /// or the base URL the socket is forwarded at; repeated for several,
    /// before the command.
    #[arg(long = "node", required = true)]
    nodes: Vec<String>,
    /// Seconds one request of a node's event log waits for an event before the manager asks
    /// again; the node waits at most its `admin_config.event_log.longest_wait`.
    #[arg(long, default_value_t = 30)]
    event_wait: u64,
    /// Seconds the manager waits before it asks again a node it could not reach.
    #[arg(long, default_value_t = 5)]
    retry_interval: u64,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// What each node serves.
    Status,
    /// Follow each node's event log until interrupted.
    Watch {
        /// Start from the events the node still keeps instead of new ones.
        #[arg(long)]
        from_start: bool,
    },
    /// Each node's configuration status.
    Config,
    /// Change each node's configuration: an idle node brings the
    /// deployment up, a serving one changes it in place or in a new epoch
    /// on trial. Nodes whose services share a GPU update one after another;
    /// a failure stops the nodes not started.
    Apply {
        /// The whole TOML configuration, as `uniserve serve --config` reads
        /// it; each node's current one when unset.
        #[arg(long)]
        config: Option<PathBuf>,
        /// `PATH=VALUE` assignments over it.
        #[arg(long = "set")]
        set: Vec<String>,
        /// Continue the requests in flight at a promotion with the state
        /// they reached (`keep`) or compute it again (`recompute`).
        #[arg(long, default_value = "keep", value_parser = wire::<InFlight>)]
        in_flight: InFlight,
        /// Report each node's dry run and the order the manager decides, and
        /// change nothing.
        #[arg(long)]
        dry_run: bool,
        /// Apply a configuration a rollback marked unavailable.
        #[arg(long)]
        force: bool,
        /// Commit a node's new epoch once it served on trial this many
        /// seconds, before a node that shares a GPU with it changes; a
        /// trial that rolled back meanwhile stops the nodes not started.
        /// Without it, each new epoch stays on trial for the operator, and
        /// the nodes that share its GPUs wait.
        #[arg(long)]
        commit_after: Option<u64>,
    },
    /// Commit each node's epoch on trial.
    Commit,
    /// Roll each node's epoch on trial back to the previous epoch.
    Rollback,
    /// Bring each node's deployment down.
    Shutdown {
        /// Keep the deployment's weights and caches retained for a restart
        /// of the same deployment.
        #[arg(long)]
        keep_storage: bool,
    },
    /// A rolling hot restart of each node's daemon, which re-executes its own image: each node
    /// in turn, stopping at the first that fails.
    Restart,
    /// Hold new requests on each node; the requests in flight go on.
    Pause,
    /// Let new requests through again on each node.
    Continue,
    /// Revoke each node's cached prefix pages.
    FlushCache,
}

/// A command-line value as the admin schema spells it (`keep`, `recompute`).
fn wire<T: DeserializeOwned>(text: &str) -> Result<T, String> {
    serde_json::from_value(serde_json::Value::String(text.to_owned()))
        .map_err(|error| error.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let nodes: Vec<Node> = cli
        .nodes
        .iter()
        .map(|address| Node::new(address))
        .collect::<Result<_>>()?;
    let polling = Polling {
        wait: Duration::from_secs(cli.event_wait),
        retry: Duration::from_secs(cli.retry_interval),
    };
    match cli.command {
        Command::Status => status(&nodes).await,
        Command::Watch { from_start } => watch(nodes, from_start, polling).await,
        Command::Config => {
            for node in &nodes {
                let config = node.config().await?;
                println!("{}: {}", node.base, serde_json::to_string_pretty(&config)?);
            }
            Ok(())
        }
        Command::Apply {
            config,
            set,
            in_flight,
            dry_run,
            force,
            commit_after,
        } => {
            let config = match config {
                Some(path) => Some(
                    std::fs::read_to_string(&path)
                        .with_context(|| format!("reading {}", path.display()))?,
                ),
                None => None,
            };
            let options = updates::Options {
                request: ApplyRequest {
                    config,
                    set,
                    in_flight,
                    force,
                    ..ApplyRequest::default()
                },
                commit_after: commit_after.map(Duration::from_secs),
                polling,
            };
            if dry_run {
                updates::dry_run(&nodes, &options).await
            } else {
                updates::apply(&nodes, &options).await
            }
        }
        Command::Commit => {
            for node in &nodes {
                let response: OperationResponse<UpdateReport> = node
                    .send(reqwest::Method::POST, "/admin/config/commit", &())
                    .await?;
                println!(
                    "{}: {}",
                    node.base,
                    serde_json::to_string(&response.result)?
                );
            }
            Ok(())
        }
        Command::Rollback => {
            for node in &nodes {
                let response: OperationResponse<UpdateReport> = node
                    .send(
                        reqwest::Method::POST,
                        "/admin/config/rollback",
                        &RollbackRequest::default(),
                    )
                    .await?;
                println!(
                    "{}: {}",
                    node.base,
                    serde_json::to_string(&response.result)?
                );
            }
            Ok(())
        }
        Command::Shutdown { keep_storage } => {
            let request = Shutdown {
                retire: !keep_storage,
            };
            for node in &nodes {
                let _: OperationResponse<Shutdown> = node
                    .send(reqwest::Method::POST, "/admin/shutdown", &request)
                    .await?;
                println!("{}: shutting down", node.base);
            }
            Ok(())
        }
        Command::Restart => {
            for node in &nodes {
                println!("{}: restarting", node.base);
                let response: OperationResponse<Restarted> = node
                    .send(
                        reqwest::Method::POST,
                        "/admin/restart",
                        &Restart { wait: true },
                    )
                    .await
                    .with_context(|| {
                        format!("{} did not restart; later nodes are untouched", node.base)
                    })?;
                let restarted = response
                    .result
                    .with_context(|| format!("{} answered no successor", node.base))?;
                println!("{}: the daemon {} serves", node.base, restarted.pid);
            }
            Ok(())
        }
        Command::Pause => each::<Paused>(&nodes, "/admin/pause_generation").await,
        Command::Continue => each::<()>(&nodes, "/admin/continue_generation").await,
        Command::FlushCache => each::<()>(&nodes, "/admin/flush_cache").await,
    }
}

/// Posts `path`, an operation whose result is `R`, to every node.
async fn each<R: DeserializeOwned>(nodes: &[Node], path: &str) -> Result<()> {
    for node in nodes {
        let response: OperationResponse<R> = node.send(reqwest::Method::POST, path, &()).await?;
        let operation = response.operation;
        println!("{}: {} {}", node.base, operation.kind, operation.state);
    }
    Ok(())
}

/// Prints what each node serves.
async fn status(nodes: &[Node]) -> Result<()> {
    for node in nodes {
        let status = node.status().await?;
        let operation = status
            .operation
            .map_or_else(|| "none".to_owned(), |operation| operation.kind.to_string());
        let line = match status.node {
            NodeState::Idle { last_error } => format!(
                "idle{}",
                last_error
                    .map(|error| format!(" (last error: {error})"))
                    .unwrap_or_default()
            ),
            NodeState::Starting { name } => format!("starting {name}"),
            NodeState::Serving { deployment, .. } => format!(
                "{} engine {}{} in flight {} operation {operation}",
                deployment.deployment,
                deployment.engine,
                if deployment.paused { " (paused)" } else { "" },
                deployment.in_flight,
            ),
            NodeState::Stopping { name } => format!("stopping {name}"),
        };
        println!("{}: {line}", node.base);
    }
    Ok(())
}

/// Follows every node's event log until interrupted, from the events each still keeps or from
/// now. A node that cannot be reached is asked again (`--retry-interval`); one whose log moved
/// past what was read says so.
async fn watch(nodes: Vec<Node>, from_start: bool, polling: Polling) -> Result<()> {
    let mut tasks = tokio::task::JoinSet::new();
    for node in nodes {
        tasks.spawn(async move {
            let mut after = if from_start {
                0
            } else {
                node.last_event().await?
            };
            loop {
                let page = node.events(after, polling).await?;
                if page.gap {
                    println!("{}: events were lost; read its status again", node.base);
                }
                for event in &page.events {
                    println!("{}: {}", node.base, serde_json::to_string(event)?);
                }
                after = page.next;
            }
        });
    }
    tokio::select! {
        interrupted = tokio::signal::ctrl_c() => {
            interrupted.context("waiting for an interrupt")?;
        }
        Some(ended) = tasks.join_next() => {
            let ended: Result<()> = ended.context("a node's watch ended without its result")?;
            ended?;
        }
    }
    tasks.abort_all();
    println!("detached; the nodes serve on");
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory as _;

    use super::*;

    /// The command line's definition is one clap accepts, and it reads the admin schema's
    /// spelling of its values.
    #[test]
    fn the_command_line_is_well_formed() {
        Cli::command().debug_assert();
        let cli = Cli::try_parse_from([
            "uniserve-manager",
            "--node",
            "http://10.0.0.2:8000",
            "--node",
            "http://10.0.0.3:8000",
            "apply",
            "--set",
            "max_running_requests=8",
            "--in-flight",
            "recompute",
            "--commit-after",
            "60",
        ])
        .unwrap();
        assert_eq!(cli.nodes.len(), 2);
        assert!(matches!(
            cli.command,
            Command::Apply {
                commit_after: Some(60),
                in_flight: InFlight::Recompute,
                ..
            }
        ));
        let refused = Cli::try_parse_from([
            "uniserve-manager",
            "--node",
            "http://10.0.0.2:8000",
            "apply",
            "--in-flight",
            "abort",
        ]);
        assert!(refused.is_err());
    }
}
