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
//! - `apply`: applies a policy package to each node's deployment (`PUT /admin/policy`), one
//!   node after another, stopping at the first whose change does not converge. Each node is
//!   one deployment's leader, and a GPU belongs to one deployment, so the changes of several
//!   deployments never contend for a device.
//! - `shutdown`: brings each node's deployment down, releasing its storage
//!   unless `--keep-storage`.
//! - `restart`: a rolling hot restart of each node's daemon, one node after
//!   another, stopping at the first that fails.
//! - `pause`, `continue`, `flush-cache`: the generation and cache operations
//!   of each node.
//!
//! A node answers an apply once its change converged or did not; what else the manager
//! follows it reads from the node's event log by long poll (`GET /admin/events`).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use proto::admin::{NodeState, OperationResponse, Paused, Restart, Restarted, Shutdown};
use serde::de::DeserializeOwned;

use node::{Node, Polling};

mod node;

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
    /// Apply the policy package in a directory (its `config.toml` and `programs/`) to each
    /// node's deployment, one node after another; a change that does not converge stops the
    /// nodes not reached.
    Apply {
        /// The package's directory.
        package: PathBuf,
        /// Apply a target a rollback marked unavailable.
        #[arg(long)]
        force: bool,
    },
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
        Command::Apply { package, force } => {
            let archive = archive(&package)?;
            for node in &nodes {
                let record = node
                    .put_policy(archive.clone(), force)
                    .await
                    .with_context(|| {
                        format!("{} did not converge; later nodes are untouched", node.base)
                    })?;
                println!("{}: {}", node.base, serde_json::to_string(&record)?);
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

/// The policy package in `directory` as a tar archive of its directory, as `PUT /admin/policy`
/// carries it: `config.toml`, and `programs/` when the package has programs.
fn archive(directory: &std::path::Path) -> Result<Vec<u8>> {
    let mut builder = tar::Builder::new(Vec::new());
    builder
        .append_path_with_name(directory.join("config.toml"), "config.toml")
        .with_context(|| format!("archiving {}/config.toml", directory.display()))?;
    let programs = directory.join("programs");
    if programs.is_dir() {
        builder
            .append_dir_all("programs", &programs)
            .with_context(|| format!("archiving {}", programs.display()))?;
    }
    builder
        .into_inner()
        .context("finishing the package archive")
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

    /// The command line's definition is one clap accepts, and `apply` takes a package's
    /// directory.
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
            "qwen3",
            "--force",
        ])
        .unwrap();
        assert_eq!(cli.nodes.len(), 2);
        assert!(matches!(cli.command, Command::Apply { force: true, .. }));
        assert!(
            Cli::try_parse_from(["uniserve-manager", "--node", "x", "apply", "--set", "a=1"])
                .is_err()
        );
    }

    /// A package's archive holds its `config.toml` and its programs, under the names the
    /// package's directory gives them.
    #[test]
    fn a_package_is_archived_as_its_directory() {
        let package = tempfile::tempdir().unwrap();
        std::fs::write(package.path().join("config.toml"), "model = \"m\"\n").unwrap();
        std::fs::create_dir_all(package.path().join("programs/chunk")).unwrap();
        std::fs::write(package.path().join("programs/chunk/lib.rs"), "").unwrap();
        let archive = archive(package.path()).unwrap();
        let mut names: Vec<String> = tar::Archive::new(archive.as_slice())
            .entries()
            .unwrap()
            .map(Result::unwrap)
            .filter(|entry| entry.header().entry_type().is_file())
            .map(|entry| entry.path().unwrap().display().to_string())
            .collect();
        names.sort();
        assert_eq!(names, ["config.toml", "programs/chunk/lib.rs"]);
    }
}
