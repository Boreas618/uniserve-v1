//! `uniserve-manager`: an optional manager of UniServe deployments across
//! nodes.
//!
//! The manager is a client of each node's admin API (`/admin`) and nothing
//! else: no daemon, engine, worker or Storage Server waits for it or keeps
//! state of it. It attaches to a node by reading its status and its event
//! log and detaches by stopping; a write it sends is carried out by the
//! node whether the manager stays or not. Every command acts on the nodes
//! `--node` names, in the order given:
//!
//! - `status`: what each node serves.
//! - `watch`: follows each node's event log until interrupted (attach and
//!   detach).
//! - `config`: each node's configuration, worlds and trial.
//! - `apply`: changes each node's configuration (`PUT /admin/config`): an
//!   idle node (`uniserve serve` without a model) brings the deployment
//!   up, a serving one changes it in place or in a new world on trial,
//!   which `--commit-after` commits once it served that long. Each node is
//!   one service's leader: the manager decides the order of their updates
//!   from the GPUs and hosts they share and the storage their plans
//!   require ([`updates`]), so that two services that share a GPU never
//!   prepare at once.
//! - `commit`, `rollback`: end each node's trial.
//! - `shutdown`: brings each node's deployment down, releasing its storage
//!   unless `--keep-storage`.
//! - `restart`: a rolling hot restart of each node's daemon, one node after
//!   another, stopping at the first that fails.
//! - `pause`, `continue`, `flush-cache`: the generation and cache operations
//!   of each node.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

mod updates;

/// The manager's command line.
#[derive(Debug, Parser)]
#[command(
    name = "uniserve-manager",
    about = "Manage UniServe deployments through their nodes' admin API"
)]
struct Cli {
    /// A node's base URL, such as `http://10.0.0.2:8000`; repeated for
    /// several, before the command.
    #[arg(long = "node", required = true)]
    nodes: Vec<String>,
    /// The nodes' API key.
    #[arg(long, env = "UNISERVE_API_KEY", global = true, hide_env_values = true)]
    api_key: Option<String>,
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
    /// Each node's configuration, worlds and trial.
    Config,
    /// Change each node's configuration: an idle node brings the
    /// deployment up, a serving one changes it in place or in a new world
    /// on trial. Nodes whose services share a GPU update one after another,
    /// each once its plan fits; a failure stops the nodes not started.
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
        #[arg(long, default_value = "keep")]
        in_flight: String,
        /// Report each node's plan and the order the manager decides, and
        /// change nothing.
        #[arg(long)]
        dry_run: bool,
        /// Apply a configuration a rollback marked unavailable.
        #[arg(long)]
        force: bool,
        /// Commit a node's new world once it served on trial this many
        /// seconds, before a node that shares a GPU with it changes; a
        /// trial that rolled back meanwhile stops the nodes not started.
        /// Without it, each new world stays on trial for the operator, and
        /// the nodes that share its GPUs wait.
        #[arg(long)]
        commit_after: Option<u64>,
    },
    /// Commit each node's world on trial.
    Commit,
    /// Roll each node's world on trial back to the previous world.
    Rollback,
    /// Bring each node's deployment down.
    Shutdown {
        /// Keep the deployment's weights and caches retained for a restart
        /// of the same deployment.
        #[arg(long)]
        keep_storage: bool,
    },
    /// A rolling hot restart of each node's daemon, to its own build or the
    /// one `--executable` names: each node in turn, stopping at the first
    /// that fails.
    Restart {
        /// The `uniserve` executable of the build to restart as.
        #[arg(long)]
        executable: Option<String>,
    },
    /// Hold new requests on each node; the requests in flight go on.
    Pause,
    /// Let new requests through again on each node.
    Continue,
    /// Revoke each node's cached prefix pages.
    FlushCache,
}

/// One node's admin API.
#[derive(Clone)]
struct Node {
    base: String,
    http: reqwest::Client,
    api_key: Option<String>,
}

impl Node {
    fn new(base: &str, http: &reqwest::Client, api_key: Option<&str>) -> Self {
        Self {
            base: base.trim_end_matches('/').to_owned(),
            http: http.clone(),
            api_key: api_key.map(str::to_owned),
        }
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let request = self.http.request(method, format!("{}{path}", self.base));
        match &self.api_key {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }

    /// `GET path` as JSON; `None` when the node answers 404 or 503.
    async fn get(&self, path: &str) -> Result<Option<Value>> {
        let response = self
            .request(reqwest::Method::GET, path)
            .send()
            .await
            .with_context(|| format!("reaching {}", self.base))?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND
            || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
        {
            return Ok(None);
        }
        let body: Value = response
            .json()
            .await
            .with_context(|| format!("reading {}{path}", self.base))?;
        if !status.is_success() {
            bail!("{}{path} answered {status}: {body}", self.base);
        }
        Ok(Some(body))
    }

    /// `POST path` with a JSON body; the answer, which must succeed.
    async fn post(&self, path: &str, body: &Value) -> Result<Value> {
        self.send(reqwest::Method::POST, path, body).await
    }

    /// `PUT path` with a JSON body; the answer, which must succeed.
    async fn put(&self, path: &str, body: &Value) -> Result<Value> {
        self.send(reqwest::Method::PUT, path, body).await
    }

    async fn send(&self, method: reqwest::Method, path: &str, body: &Value) -> Result<Value> {
        let response = self
            .request(method, path)
            .json(body)
            .send()
            .await
            .with_context(|| format!("reaching {}", self.base))?;
        let status = response.status();
        let body: Value = response
            .json()
            .await
            .with_context(|| format!("reading {}{path}", self.base))?;
        if !status.is_success() {
            bail!("{}{path} answered {status}: {body}", self.base);
        }
        Ok(body)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let http = reqwest::Client::builder()
        .build()
        .context("building the HTTP client")?;
    let nodes: Vec<Node> = cli
        .nodes
        .iter()
        .map(|base| Node::new(base, &http, cli.api_key.as_deref()))
        .collect();
    match cli.command {
        Command::Status => status(&nodes).await,
        Command::Watch { from_start } => watch(nodes, from_start).await,
        Command::Config => {
            for node in &nodes {
                let config = node.get("/admin/config").await?.unwrap_or(Value::Null);
                println!(
                    "{}: {}",
                    node.base,
                    serde_json::to_string_pretty(&json!({
                        "digest": config["digest"],
                        "slots": config["slots"],
                        "update": config["update"],
                        "trial": config["trial"],
                        "replicas": config["replicas"],
                        "pending_replicas": config["pending_replicas"],
                        "commit": config["commit"],
                        "unavailable": config["unavailable"],
                    }))?
                );
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
                body: json!({
                    "config": config,
                    "set": set,
                    "in_flight": in_flight,
                    "force": force,
                    "wait": true,
                }),
                commit_after: commit_after.map(Duration::from_secs),
                poll: Duration::from_secs(1),
            };
            if dry_run {
                updates::plan(&nodes, &options).await
            } else {
                updates::apply(&nodes, &options).await
            }
        }
        Command::Commit => {
            for node in &nodes {
                let answer = node.post("/admin/config/commit", &json!({})).await?;
                println!("{}: {}", node.base, answer["result"]);
            }
            Ok(())
        }
        Command::Rollback => {
            for node in &nodes {
                let answer = node.post("/admin/config/rollback", &json!({})).await?;
                println!("{}: {}", node.base, answer["result"]);
            }
            Ok(())
        }
        Command::Shutdown { keep_storage } => {
            for node in &nodes {
                node.post("/admin/shutdown", &json!({ "retire": !keep_storage }))
                    .await?;
                println!("{}: shutting down", node.base);
            }
            Ok(())
        }
        Command::Restart { executable } => {
            for node in &nodes {
                println!("{}: restarting", node.base);
                let answer = node
                    .post(
                        "/admin/restart",
                        &json!({ "executable": executable, "wait": true }),
                    )
                    .await
                    .with_context(|| {
                        format!("{} did not restart; later nodes are untouched", node.base)
                    })?;
                println!(
                    "{}: the daemon {} serves",
                    node.base, answer["result"]["pid"]
                );
            }
            Ok(())
        }
        Command::Pause => each(&nodes, "/admin/pause_generation").await,
        Command::Continue => each(&nodes, "/admin/continue_generation").await,
        Command::FlushCache => each(&nodes, "/admin/flush_cache").await,
    }
}

/// Posts `path` to every node.
async fn each(nodes: &[Node], path: &str) -> Result<()> {
    for node in nodes {
        let answer = node.post(path, &json!({})).await?;
        println!("{}: {} {}", node.base, answer["kind"], answer["state"]);
    }
    Ok(())
}

/// Prints what each node serves.
async fn status(nodes: &[Node]) -> Result<()> {
    for node in nodes {
        // An idle node (`uniserve serve` without a model) states it; a
        // serving one states its deployment.
        let idle = node.get("/admin/node").await?;
        let serving = node.get("/admin/status").await?;
        let line = match (&idle, &serving) {
            (_, Some(status)) => {
                let deployment = &status["deployment"];
                format!(
                    "{} engine {}{} in flight {} operation {}",
                    deployment["deployment"].as_str().unwrap_or("?"),
                    deployment["engine"].as_str().unwrap_or("?"),
                    if deployment["paused"] == json!(true) {
                        " (paused)"
                    } else {
                        ""
                    },
                    deployment["in_flight"],
                    status["operation"]["kind"].as_str().unwrap_or("none"),
                )
            }
            (Some(node), None) => format!(
                "node {}{}",
                node["state"].as_str().unwrap_or("?"),
                node["last_error"]
                    .as_str()
                    .map(|error| format!(" (last error: {error})"))
                    .unwrap_or_default()
            ),
            (None, None) => "serves no admin API".to_owned(),
        };
        println!("{}: {line}", node.base);
    }
    Ok(())
}

/// Follows every node's event log until interrupted. A node that cannot be
/// reached is retried; one whose log moved past what was read says so.
async fn watch(nodes: Vec<Node>, from_start: bool) -> Result<()> {
    let mut tasks = Vec::new();
    for node in nodes {
        tasks.push(tokio::spawn(async move {
            let mut after = if from_start {
                0
            } else {
                match node.get("/admin/status").await {
                    Ok(Some(status)) => {
                        status["next_event"].as_u64().unwrap_or(1).saturating_sub(1)
                    }
                    _ => 0,
                }
            };
            loop {
                match node
                    .get(&format!("/admin/events?after={after}&wait=30"))
                    .await
                {
                    Ok(Some(page)) => {
                        if page["gap"] == json!(true) {
                            println!("{}: events were lost; read its status again", node.base);
                        }
                        for event in page["events"].as_array().into_iter().flatten() {
                            println!(
                                "{}: [{}] {}",
                                node.base,
                                event["kind"].as_str().unwrap_or("?"),
                                event["message"].as_str().unwrap_or("?")
                            );
                        }
                        after = page["next"].as_u64().unwrap_or(after);
                    }
                    Ok(None) => tokio::time::sleep(Duration::from_secs(5)).await,
                    Err(error) => {
                        println!("{}: {error:#}; retrying", node.base);
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                }
            }
        }));
    }
    tokio::signal::ctrl_c()
        .await
        .context("waiting for an interrupt")?;
    for task in tasks {
        task.abort();
    }
    println!("detached; the nodes serve on");
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory as _;

    use super::*;

    /// The command line's definition is one clap accepts.
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
            "--commit-after",
            "60",
        ])
        .unwrap();
        assert_eq!(cli.nodes.len(), 2);
        assert!(matches!(
            cli.command,
            Command::Apply {
                commit_after: Some(60),
                ..
            }
        ));
    }
}
