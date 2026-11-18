//! A node's daemon in a process of its own, for the daemon's tests of what ends with a daemon
//! that is killed, and of what a node holds at its own host's Storage Server.
//!
//! `daemon-host --socket PATH` starts the daemon of a single node `node` of the deployment
//! `deployment`, whose process socket is `PATH` and whose deployment's Build Server and Storage
//! Server serve in the process's runtime directory, writes the line `serving` to its standard
//! output once the daemon serves, and serves until the process is killed.
//!
//! With `--host NAME --nnodes N --node-rank R --join ENDPOINT`, the daemon is node `R` on host
//! `NAME` of a deployment of `N` nodes. Its leader (node 0) binds its join listener at `ENDPOINT`
//! (port 0 picks one) and writes `joining ENDPOINT`, the endpoint it bound, before it waits for
//! its members; a member joins its leader at `ENDPOINT`. The leader writes `address ADDRESS`,
//! where engines reach it, after `serving`. The process's runtime directory (`XDG_RUNTIME_DIR`) is
//! its host's, where the node's daemon claims storage.

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::Context as _;

/// The value of option `name` on this process's command line; `None` when it names none.
fn option(arguments: &[String], name: &str) -> anyhow::Result<Option<String>> {
    let Some(position) = arguments.iter().position(|argument| argument == name) else {
        return Ok(None);
    };
    arguments
        .get(position + 1)
        .cloned()
        .map(Some)
        .with_context(|| format!("{name} names no value"))
}

/// The value of option `name`, which the command line must name.
fn required(arguments: &[String], name: &str) -> anyhow::Result<String> {
    option(arguments, name)?.with_context(|| format!("the command line names no {name}"))
}

/// Writes `line` to this process's standard output at once: the test reads it while the process
/// goes on.
fn report(line: &str) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{line}").context("reporting the daemon's state")?;
    stdout.flush().context("reporting the daemon's state")
}

fn main() -> anyhow::Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    let host = option(&arguments, "--host")?.unwrap_or_else(|| "node".to_owned());
    let nnodes: u32 = option(&arguments, "--nnodes")?.map_or(Ok(1), |nnodes| nnodes.parse())?;
    let node_rank: u32 =
        option(&arguments, "--node-rank")?.map_or(Ok(0), |node_rank| node_rank.parse())?;
    let join: Option<service::Endpoint> = option(&arguments, "--join")?
        .map(|endpoint| endpoint.parse())
        .transpose()
        .map_err(|error| anyhow::anyhow!("--join names no endpoint: {error}"))?;

    // The leader of several nodes binds where its members join before it waits for them.
    let (listener, dist_init_addr) = match join {
        Some(endpoint) if node_rank == 0 => {
            let (listener, bound) = daemon::Daemon::bind_join(&endpoint)?;
            report(&format!("joining {bound}"))?;
            (Some(listener), Some(bound))
        }
        joined => (None, joined),
    };
    let daemon = daemon::Daemon::start(
        daemon::DaemonOptions {
            // The deployment's servers are the ones its runtime directory, the variable's, holds.
            deployment: "deployment".to_owned(),
            runtime: service::RuntimeDirectory::of_user()?,
            // The host's registration file, in its runtime directory: each process is a host.
            gpus: service::RuntimeDirectory::of_user()?.path().join("gpus"),
            nnodes,
            node_rank,
            host,
            dist_init_addr,
            socket: PathBuf::from(required(&arguments, "--socket")?),
            join_timeout: daemon::JOIN_TIMEOUT,
            build_server: build_server::command::Settings::default(),
            storage: storage::Settings::default(),
        },
        listener,
    )
    .context("starting the daemon")?;
    report("serving")?;
    // A single node's caller named its process socket; the leader of several nodes states the
    // address it bound on the network.
    if nnodes > 1
        && let Some(address) = daemon.address()
    {
        report(&format!("address {address}"))?;
    }
    // The daemon's service runs on its own thread until this process is killed.
    loop {
        std::thread::park();
    }
}
