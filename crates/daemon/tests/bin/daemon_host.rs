//! A node's daemon in a process of its own, for the daemon's tests of what ends with a daemon
//! that is killed.
//!
//! `daemon-host --socket PATH --build-server PATH` starts the daemon of a single node `node`
//! whose process socket is `PATH` and whose host's Build Server serves at the second `PATH`,
//! writes the line `serving` to its standard output once the daemon serves, and serves until the
//! process is killed.

use std::io::Write as _;
use std::path::PathBuf;

/// The value of option `name` on this process's command line.
fn option(arguments: &[String], name: &str) -> PathBuf {
    let position = arguments
        .iter()
        .position(|argument| argument == name)
        .unwrap_or_else(|| panic!("the command line names no {name}"));
    PathBuf::from(&arguments[position + 1])
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let _daemon = daemon::Daemon::start(
        daemon::DaemonOptions {
            nnodes: 1,
            node_rank: 0,
            host: "node".to_owned(),
            dist_init_addr: None,
            socket: option(&arguments, "--socket"),
            join_timeout: daemon::JOIN_TIMEOUT,
            build_server: build_server::command::Settings {
                socket: Some(option(&arguments, "--build-server")),
                ..build_server::command::Settings::default()
            },
            storage: storage::Settings::default(),
        },
        None,
    )
    .expect("the daemon starts");
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "serving").expect("the readiness line is written");
    stdout.flush().expect("the readiness line is written");
    drop(stdout);
    // The daemon's service runs on its own thread until this process is killed.
    loop {
        std::thread::park();
    }
}
