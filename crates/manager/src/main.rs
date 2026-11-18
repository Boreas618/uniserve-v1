//! `uniserve-manager`: an optional manager of UniServe deployments across
//! nodes.
//!
//! The manager is a client of each node's admin API (`/admin`) and nothing
//! else: no daemon, engine, worker or Storage Server waits for it or keeps
//! state of it. A node's admin API listens on a Unix socket whose file permissions are its
//! access (`unix:<path>`); an `http://` address names an admin socket forwarded from another
//! host. The API has two endpoints: `/admin/policy`, the deployment's policy package, the one
//! way it changes, and `/admin/query`, SQL over its observation store (`stats` and `events`),
//! the one way its state is read. Every command acts on the nodes `--node` names, in the order
//! given:
//!
//! - `status`: what each node serves, from its latest `node` event and its replicas' sampled
//!   state.
//! - `watch`: follows each node's `events` until interrupted (attach and detach), from the
//!   events it keeps or from now.
//! - `apply`: applies a policy package to each node's deployment (`PUT /admin/policy`), one
//!   node after another, stopping at the first whose change does not converge. Each node is
//!   one deployment's leader, and a GPU belongs to one deployment, so the changes of several
//!   deployments never contend for a device.
//!
//! What an operator does to a deployment is an option of its package: holding new requests
//! (`paused`), sleeping (`asleep`), flushing the prefix cache (`flush`), releasing its storage
//! when it comes down (`retire`), and a hot restart of its daemon (any daemon option). A node
//! comes down at its process's stop signal.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use serde_json::Value;

use node::Node;

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
    /// Follow each node's events until interrupted.
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let nodes: Vec<Node> = cli
        .nodes
        .iter()
        .map(|address| Node::new(address))
        .collect::<Result<_>>()?;
    let retry = Duration::from_secs(cli.retry_interval);
    match cli.command {
        Command::Status => status(&nodes).await,
        Command::Watch { from_start } => watch(nodes, from_start, retry).await,
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

/// The latest `node` event of a node: where its deployment is in its life.
const NODE: &str = "SELECT fields['state'] AS state, fields['deployment'] AS deployment, \
                    fields['error'] AS error FROM events \
                    WHERE kind = 'log' AND labels['source'] = 'node' ORDER BY time DESC LIMIT 1";

/// Each replica's state and the requests the deployment holds, as the leader last sampled them.
const REPLICAS: &str = "SELECT name, labels['replica'] AS replica, labels['state'] AS state, \
                        value FROM stats \
                        WHERE name IN ('uniserve_replica_state', 'uniserve_requests_held', \
                        'uniserve_requests_in_flight') \
                        AND time = (SELECT max(time) FROM stats \
                        WHERE name = 'uniserve_requests_held') ORDER BY replica";

/// Prints what each node serves: its deployment's state, each replica's engine, and the
/// requests in flight and held.
async fn status(nodes: &[Node]) -> Result<()> {
    for node in nodes {
        let latest = node.query(NODE).await?;
        let Some(latest) = latest.first() else {
            println!("{}: idle", node.base);
            continue;
        };
        let text = |value: &Value| value.as_str().unwrap_or_default().to_owned();
        let mut line = format!("{} {}", text(&latest["state"]), text(&latest["deployment"]));
        if let Some(error) = latest["error"].as_str() {
            line.push_str(&format!(" (error: {error})"));
        }
        let (mut held, mut in_flight, mut replicas) = (0.0, 0.0, Vec::new());
        for row in node.query(REPLICAS).await? {
            match row["name"].as_str() {
                Some("uniserve_requests_held") => held = row["value"].as_f64().unwrap_or(0.0),
                Some("uniserve_requests_in_flight") => {
                    in_flight += row["value"].as_f64().unwrap_or(0.0);
                }
                _ => replicas.push(format!("{} {}", text(&row["replica"]), text(&row["state"]))),
            }
        }
        if !replicas.is_empty() {
            line.push_str(&format!(
                " replicas [{}] in flight {in_flight} held {held}",
                replicas.join(", ")
            ));
        }
        println!("{}: {line}", node.base);
    }
    Ok(())
}

/// Follows every node's events until interrupted, from the events each still keeps or from
/// now. A node that cannot be reached is asked again (`--retry-interval`).
async fn watch(nodes: Vec<Node>, from_start: bool, retry: Duration) -> Result<()> {
    let mut tasks = tokio::task::JoinSet::new();
    for node in nodes {
        tasks.spawn(async move {
            let since = if from_start {
                String::new()
            } else {
                format!(" WHERE CAST(time AS BIGINT) > {}", node.now().await?)
            };
            let sql = format!("SELECT time, kind, labels, fields, revision FROM events{since}");
            node.follow(&sql, retry, |row| {
                println!("{}: {row}", node.base);
            })
            .await
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
