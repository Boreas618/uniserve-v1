//! The serving command's follower role needs no model on its command line.

use std::process::{Child, Command};
use std::time::Duration;

use daemon::{Daemon, DaemonOptions};

struct Follower(Child);

impl Drop for Follower {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn a_follower_joins_without_a_model() {
    let directory = tempfile::tempdir().unwrap();
    // The leader's join listener is bound first, on a port the kernel picks, and the follower is
    // given the endpoint it bound.
    let (join, endpoint) = Daemon::bind_join(&"127.0.0.1:0".parse().unwrap()).unwrap();
    let address = endpoint.to_string();
    let _follower = Follower(
        Command::new(env!("CARGO_BIN_EXE_uniserve"))
            .args([
                "serve",
                "--nnodes",
                "2",
                "--node-rank",
                "1",
                "--host-identity",
                "follower",
                "--dist-init-addr",
                &address,
            ])
            .env("XDG_RUNTIME_DIR", directory.path())
            .spawn()
            .unwrap(),
    );
    // Startup returns only after the required follower has joined. Neither
    // node starts a model or needs a GPU to establish membership.
    let leader = Daemon::start(
        DaemonOptions {
            nnodes: 2,
            node_rank: 0,
            host: "leader".into(),
            dist_init_addr: Some(endpoint),
            socket: directory.path().join("leader.sock"),
            executable: env!("CARGO_BIN_EXE_uniserve").into(),
            join_timeout: Duration::from_secs(10),
            devices: || Ok(Vec::new()),
            build_server: build_server::command::Settings::default(),
            storage: Default::default(),
        },
        Some(join),
    )
    .unwrap();
    assert!(leader.address().is_some());
}
