//! A node's daemon in a process of its own, for the daemon's tests of what ends with a daemon
//! that is killed.
//!
//! `daemon-host --socket PATH --build-server PATH` starts the daemon of a single node `node`
//! whose process socket is `PATH` and whose host's Build Server serves at the second `PATH`,
//! writes the line `serving` to its standard output once the daemon serves, and serves until the
//! process is killed.

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::Context as _;

/// The value of option `name` on this process's command line.
fn option(arguments: &[String], name: &str) -> anyhow::Result<PathBuf> {
    let position = arguments
        .iter()
        .position(|argument| argument == name)
        .with_context(|| format!("the command line names no {name}"))?;
    arguments
        .get(position + 1)
        .map(PathBuf::from)
        .with_context(|| format!("{name} names no path"))
}

fn main() -> anyhow::Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    let _daemon = daemon::Daemon::start(
        daemon::DaemonOptions {
            nnodes: 1,
            node_rank: 0,
            host: "node".to_owned(),
            dist_init_addr: None,
            socket: option(&arguments, "--socket")?,
            join_timeout: daemon::JOIN_TIMEOUT,
            build_server: build_server::command::Settings {
                socket: Some(option(&arguments, "--build-server")?),
                ..build_server::command::Settings::default()
            },
            storage: storage::Settings::default(),
        },
        None,
    )
    .context("starting the daemon")?;
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "serving").context("reporting that the daemon serves")?;
    stdout.flush().context("reporting that the daemon serves")?;
    drop(stdout);
    // The daemon's service runs on its own thread until this process is killed.
    loop {
        std::thread::park();
    }
}
