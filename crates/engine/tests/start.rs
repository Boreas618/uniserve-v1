//! The engine executable started as a daemon starts it, end to end on the meta device.
//!
//! The test plays the daemon on two Unix sockets (the request interface and the process
//! interface) and runs a Storage Server and a Build Server on threads of its own process, the
//! Build Server holding the tree's `fcfs` policy package. It starts the built `uniserve-engine`
//! with a launch of the Qwen3-MoE fixture on one rank of one group, `device = "meta"`: the
//! engine says hello and `Starting`, claims the node's Storage Server, starts its rank, opens
//! its session, resolves its policy and serves; it takes a request, states its metrics, and
//! exits with status 0 once shut down, its session closed.
//!
//! No GPU is touched. The Build Server builds the `fcfs` module from its installed snapshot, so
//! the test builds the policy precompiler first, into a target directory of its own.
#![cfg(test)]

mod support;

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::sync::Arc;
use std::time::{Duration, Instant};

use proto::Connection;
use proto::config::EngineArgs;
use proto::process::{downstream, report, upstream};
use proto::request::wire::{self, from_engine};
use support::{DEPLOYMENT, StorageServer, workspace};

/// A bound on a read only a defect reaches.
const READ: Duration = Duration::from_secs(60);

/// A bound on the engine's start: the Build Server builds the policy module from its snapshot on
/// the first resolve.
const START: Duration = Duration::from_secs(900);

/// The policy precompiler a Build Server builds a policy package with, built by the cargo that
/// runs the tests into a target directory of its own, so the build does not contend for the lock
/// of the tree the tests were built in.
fn precompiler() -> PathBuf {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| workspace().join("target"), PathBuf::from)
        .join("it-policy");
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(&cargo)
        .current_dir(workspace())
        .args([
            "build",
            "-q",
            "-p",
            "scheduling",
            "--bin",
            "scheduling-precompile",
        ])
        .arg("--target-dir")
        .arg(&target)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "building the policy precompiler: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    target.join("debug/scheduling-precompile")
}

/// A Build Server on a thread of the test process at `directory`'s socket, holding the tree's
/// `fcfs` policy package as revision 1. Returns its socket.
fn build_server(directory: &Path) -> PathBuf {
    let config = build_server::server::ServerConfig {
        socket: directory.join("build-server.sock"),
        package_root: directory.join("packages"),
        store_roots: vec![build_server::store::Root {
            path: directory.join("store"),
            writable: true,
        }],
        upstream: None,
        read_only_stores: Vec::new(),
        limits: build_server::builder::Limits::default(),
        toolchains: build_server::toolchain::ToolchainConfig {
            nvcc: None,
            python: None,
            wasm: Some(precompiler()),
            cargo: std::env::var_os("CARGO").map(PathBuf::from),
            max_jobs: 4,
        },
        devices: Vec::new(),
    };
    std::fs::create_dir_all(&config.package_root).unwrap();
    std::fs::create_dir_all(&config.store_roots[0].path).unwrap();

    let server = Arc::new(build_server::server::BuildServer::open(&config).unwrap());
    std::thread::spawn(move || {
        if let Err(error) = server.serve() {
            eprintln!("the Build Server stopped: {error}");
        }
    });

    let deadline = Instant::now() + READ;
    let mut client = loop {
        match build_server::protocol::Client::connect(&config.socket) {
            Ok(client) => break client,
            Err(error) => {
                assert!(
                    Instant::now() < deadline,
                    "the Build Server never listened: {error}"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    };
    let installed = client
        .install(&workspace().join("crates/scheduling/policies/fcfs"))
        .unwrap();
    assert_eq!(installed.revision, 1);
    config.socket
}

/// The launch of epoch 1's one main process, serving the Qwen3-MoE fixture on one rank of one
/// group on the meta device, with one scheduler, as engine 1 of the scheduler node; its daemon's
/// sockets are in `directory`, and its Build Server's at `build_server`.
fn launch(directory: &Path, build_server: &Path) -> EngineArgs {
    let mut launch = support::launch();
    launch.engine_config.device.device = "meta".to_owned();
    launch.engine = 1;
    launch.scheduler_node = true;
    launch.requests = format!("unix:{}", directory.join("requests.sock").display());
    launch.processes = format!("unix:{}", directory.join("processes.sock").display());
    launch.build_server = build_server.to_string_lossy().into_owned();
    launch
}

/// The engine process, killed when the test ends before it exited.
struct Engine(Child);

impl Engine {
    /// The process's exit status, once it exited within `bound`.
    fn exited(&mut self, bound: Duration) -> ExitStatus {
        let deadline = Instant::now() + bound;
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "the engine did not exit");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

/// The first connection to `listener` within `bound`.
fn accept(listener: &proto::Listener, engine: &mut Engine, bound: Duration) -> Connection {
    let deadline = Instant::now() + bound;
    loop {
        if let Some(connection) = listener.try_accept().unwrap() {
            return connection;
        }
        if let Some(status) = engine.0.try_wait().unwrap() {
            panic!("the engine exited with {status} before it connected");
        }
        assert!(Instant::now() < deadline, "the engine did not connect");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The next frame of the request interface the engine sends within `bound`.
fn next(requests: &mut Connection, bound: Duration) -> from_engine::Kind {
    let frame: wire::FromEngine = requests
        .recv(bound)
        .unwrap()
        .expect("the engine sent no frame in time");
    frame.kind.expect("a frame states its kind")
}

/// The frames of the request interface until one `wanted` takes, within `bound`.
fn until<T>(
    requests: &mut Connection,
    bound: Duration,
    mut wanted: impl FnMut(from_engine::Kind) -> Option<T>,
) -> T {
    let deadline = Instant::now() + bound;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        assert!(!left.is_zero(), "the engine did not send the frame in time");
        match next(requests, left) {
            from_engine::Kind::State(state) if state.state == wire::EngineState::Failed as i32 => {
                panic!("the engine failed: {}", state.error)
            }
            kind => {
                if let Some(found) = wanted(kind) {
                    return found;
                }
            }
        }
    }
}

#[test]
fn an_engine_started_on_the_meta_device_serves_and_shuts_down() {
    let directory = tempfile::tempdir().unwrap();
    let storage = StorageServer::start();
    let build_server = build_server(directory.path());
    let launch = launch(directory.path(), &build_server);

    let requests_socket = proto::Listener::bind(&launch.requests).unwrap();
    let processes_socket = proto::Listener::bind(&launch.processes).unwrap();
    let launch_file = directory.path().join("launch.json");
    std::fs::write(&launch_file, serde_json::to_vec(&launch).unwrap()).unwrap();
    let mut engine = Engine(
        Command::new(env!("CARGO_BIN_EXE_uniserve-engine"))
            .arg("--launch")
            .arg(&launch_file)
            .spawn()
            .unwrap(),
    );

    // The request interface: hello, welcome, `Starting`.
    let mut requests = accept(&requests_socket, &mut engine, READ);
    let hello: wire::Hello = requests.recv(READ).unwrap().unwrap();
    assert_eq!(hello.engine, launch.engine);
    assert_eq!(hello.interface, proto::process::Interface::Requests as i32);
    requests
        .send(&wire::Welcome {
            major: proto::ENGINE_INTERFACE_VERSION,
            build: proto::BUILD.to_owned(),
            refusal: String::new(),
        })
        .unwrap();
    let from_engine::Kind::State(starting) = next(&mut requests, READ) else {
        panic!("the engine's first frame states no state");
    };
    assert_eq!(starting.state, wire::EngineState::Starting as i32);

    // The process interface: hello, welcome, the claim answered with the Storage Server.
    let mut processes = accept(&processes_socket, &mut engine, READ);
    let hello: proto::process::Hello = processes.recv(READ).unwrap().unwrap();
    assert!(matches!(
        hello.peer,
        Some(proto::process::hello::Peer::Engine(
            proto::process::Engine { engine: 1, .. }
        ))
    ));
    processes
        .send(&proto::process::Welcome {
            major: proto::ENGINE_INTERFACE_VERSION,
            build: proto::BUILD.to_owned(),
            refusal: String::new(),
            nodes: vec![proto::process::Node {
                node_rank: 0,
                host: "localhost".to_owned(),
                address: String::new(),
            }],
        })
        .unwrap();
    let claim: proto::process::Upstream = processes.recv(READ).unwrap().unwrap();
    let Some(upstream::Kind::Request(request)) = claim.kind else {
        panic!("the engine's first frame on its process interface is no request");
    };
    assert!(matches!(
        request.kind,
        Some(proto::process::request::Kind::Claim(_))
    ));
    processes
        .send(&proto::process::Downstream {
            kind: Some(downstream::Kind::Reply(proto::process::Reply {
                id: request.id,
                error: String::new(),
                kind: Some(proto::process::reply::Kind::Claimed(
                    proto::process::Claimed {
                        socket: storage.socket.to_string_lossy().into_owned(),
                        claim: vec![7; 16],
                        tcp: String::new(),
                    },
                )),
            })),
        })
        .unwrap();

    // The start ends serving: `Serving` with what it serves on the request interface, and the
    // first generation on the process interface.
    let info = until(&mut requests, START, |kind| match kind {
        from_engine::Kind::State(state) if state.state == wire::EngineState::Serving as i32 => {
            Some(state.info.expect("a serving engine states what it serves"))
        }
        _ => None,
    });
    assert_eq!(info.schedulers, 1);
    let serving: proto::process::Upstream = processes.recv(READ).unwrap().unwrap();
    assert_eq!(
        serving.kind,
        Some(upstream::Kind::Report(proto::process::Report {
            kind: Some(report::Kind::Serving(proto::process::Serving {
                launch_generation: launch.engine,
                generation: 1,
            })),
        }))
    );

    // One request is accepted by the one scheduler.
    let request =
        proto::request::Request::Generation(Box::new(proto::request::GenerationRequest {
            request_id: proto::request::RequestId(11),
            prompt_token_ids: vec![1, 5, 6],
            multimodal_inputs: Default::default(),
            sampling: Default::default(),
            max_und_tokens: 2,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            include_stop_token: false,
            generated_token_ids: Vec::new(),
            adapter: 0,
        }));
    requests
        .send(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Submit(wire::Submit {
                requests: vec![request.to_wire()],
            })),
        })
        .unwrap();
    let accepted = until(&mut requests, READ, |kind| match kind {
        from_engine::Kind::Accepted(accepted) => Some(accepted),
        _ => None,
    });
    assert_eq!(
        accepted,
        wire::Accepted {
            request: 11,
            scheduler: 0
        }
    );

    // The metrics, once a second.
    until(&mut requests, READ, |kind| match kind {
        from_engine::Kind::Metrics(metrics) => Some(metrics),
        _ => None,
    });

    // A shutdown ends the process in order: it exits with status 0, its session closed.
    requests
        .send(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::Shutdown(wire::Shutdown {})),
            })),
        })
        .unwrap();
    let status = engine.exited(READ);
    assert!(status.success(), "the engine exited with {status}");

    // The epoch's session is open no more: a head of the epoch opens one again.
    let mut head =
        engine::storage_client::StorageClient::connect(&storage.socket, DEPLOYMENT, launch.epoch)
            .unwrap();
    let empty = storage::runtime::Open {
        version: storage::runtime::VERSION,
        epoch: launch.epoch,
        trial: false,
        regions: Vec::new(),
        classes: Vec::new(),
        trajectory: 0,
        groups: Vec::new(),
        schedulers: Vec::new(),
        peers: Vec::new(),
    };
    if let Err(error) = head.open(empty) {
        let error = format!("{error:#}");
        assert!(
            !error.contains("is open"),
            "the Storage Server holds the epoch's session after its engine exited: {error}"
        );
    }
}
