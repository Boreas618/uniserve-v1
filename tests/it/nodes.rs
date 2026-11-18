//! A node other than the leader finds its place in the package's `hosts` by its host name and
//! joins the leader at the first host's join port; it starts no model of its own.

use std::process::{Child, Command};

use daemon::{Daemon, DaemonOptions, JOIN_TIMEOUT};

struct Follower(Child);

impl Drop for Follower {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// This host's name, which a node finds itself by in `hosts`.
fn this_host() -> String {
    rustix::system::uname()
        .nodename()
        .to_string_lossy()
        .into_owned()
}

#[test]
fn a_follower_joins_the_first_host() {
    let directory = tempfile::tempdir().unwrap();
    // The leader's join listener is bound first, on a port the kernel picks, and the package
    // names the leader by an address this host reaches and the port it bound.
    let (join, endpoint) = Daemon::bind_join(&"127.0.0.1:0".parse().unwrap()).unwrap();
    let port = endpoint.port().unwrap();
    let package = directory.path().join("model");
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(
        package.join("config.toml"),
        format!(
            "model = \"org/model\"\nhosts = [\"127.0.0.1\", \"{}\"]\n[admin_config]\njoin = {port}\n",
            this_host()
        ),
    )
    .unwrap();
    let _follower = Follower(
        Command::new(env!("CARGO_BIN_EXE_uniserve"))
            .arg("serve")
            .arg(&package)
            // The follower creates its instance directory in the user's runtime directory, which
            // a process names by the XDG variable (`service::RuntimeDirectory`).
            .env("XDG_RUNTIME_DIR", directory.path())
            .env("XDG_STATE_HOME", directory.path())
            .spawn()
            .unwrap(),
    );
    // Startup returns only after the required follower has joined. Neither
    // node starts a model or needs a GPU to establish membership.
    let leader = Daemon::start(
        DaemonOptions {
            // The deployment is named by its package's directory.
            deployment: "model".into(),
            runtime: service::RuntimeDirectory::within(directory.path()).unwrap(),
            gpus: directory.path().join("gpus"),
            nnodes: 2,
            node_rank: 0,
            host: "leader".into(),
            dist_init_addr: Some(endpoint),
            socket: directory.path().join("leader.sock"),
            join_timeout: JOIN_TIMEOUT,
            build_server: build_server::command::Settings::default(),
            storage: Default::default(),
        },
        Some(join),
    )
    .unwrap();
    assert!(leader.address().is_some());
}
