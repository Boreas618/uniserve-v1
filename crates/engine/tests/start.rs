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
//! No GPU is touched. The Build Server builds the `fcfs` module from its installed snapshot and
//! precompiles it with the engine executable under test (`uniserve-engine precompile`).
#![cfg(test)]

mod support;

use std::net::TcpListener;
use std::os::fd::{AsRawFd as _, RawFd};
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::sync::Arc;
use std::time::{Duration, Instant};

use proto::config::{EngineArgs, Inherited, WorkerGroupConfig, World};
use proto::process::{downstream, report, upstream};
use proto::request::wire::{self, from_engine};
use service::{Connection, Listener};
use support::{DEPLOYMENT, StorageServer, workspace};

/// A bound on a read only a defect reaches.
const READ: Duration = Duration::from_secs(60);

/// A bound on the engine's start: the Build Server builds the policy module from its snapshot on
/// the first resolve.
const START: Duration = Duration::from_secs(900);

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
            wasm: Some(PathBuf::from(env!("CARGO_BIN_EXE_uniserve-engine"))),
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
fn accept(listener: &Listener, engine: &mut Engine, bound: Duration) -> Connection {
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

/// An engine process the test plays the daemon of: its two interfaces, after the handshake and
/// the claim.
struct Daemon {
    engine: Engine,
    requests: Connection,
    processes: Connection,
}

/// Starts the built `uniserve-engine` with `launch`, written into `directory`, and plays its
/// daemon up to the claim, which it answers with `storage`: the engine says hello and `Starting`
/// on its request interface, says hello on its process interface and claims. The process
/// inherits the descriptors `inherited` open, as the daemon's supervisor places the sockets it
/// reserved for a scheduler node.
fn started(
    directory: &Path,
    launch: &EngineArgs,
    storage: &StorageServer,
    inherited: &[RawFd],
) -> Daemon {
    let requests_socket = Listener::bind(&launch.requests).unwrap();
    let processes_socket = Listener::bind(&launch.processes).unwrap();
    let launch_file = directory.join("launch.json");
    std::fs::write(&launch_file, serde_json::to_vec(launch).unwrap()).unwrap();

    let mut command = Command::new(env!("CARGO_BIN_EXE_uniserve-engine"));
    command.arg("--launch").arg(&launch_file);
    let inherited = inherited.to_vec();
    // SAFETY: the hook runs in the forked child before it executes the engine, and only clears
    // the close-on-exec flag of descriptors the test process holds open.
    unsafe {
        command.pre_exec(move || {
            for descriptor in &inherited {
                if libc::fcntl(*descriptor, libc::F_SETFD, 0) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let mut engine = Engine(command.spawn().unwrap());

    // The request interface: hello, welcome, `Starting`.
    let mut requests = accept(&requests_socket, &mut engine, READ);
    let hello: wire::Hello = requests.recv(READ).unwrap().unwrap();
    assert_eq!(hello.engine, launch.engine);
    assert_eq!(hello.interface, proto::process::Interface::Requests as i32);
    requests
        .deliver(&wire::Welcome {
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
        Some(proto::process::hello::Peer::Engine(proto::process::Engine { engine, .. }))
            if engine == launch.engine
    ));
    processes
        .deliver(&proto::process::Welcome {
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
        .deliver(&proto::process::Downstream {
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

    Daemon {
        engine,
        requests,
        processes,
    }
}

/// Whether `storage` holds the session of the engine of launch `generation` of `epoch`: a head of
/// that engine cannot open one again while it is open, or while its processes have not left.
fn holds_session(storage: &StorageServer, epoch: u64, generation: u64) -> bool {
    let mut head = engine::memory::client::StorageClient::connect_head(
        &storage.socket,
        DEPLOYMENT,
        epoch,
        generation,
    )
    .unwrap();
    let empty = storage::runtime::Open {
        version: storage::runtime::VERSION,
        epoch,
        trial: false,
        regions: Vec::new(),
        classes: Vec::new(),
        trajectory: 0,
        groups: Vec::new(),
        schedulers: Vec::new(),
        peers: Vec::new(),
    };
    match head.open(empty) {
        Ok(_) => false,
        Err(error) => format!("{error:#}").contains("is open"),
    }
}

/// Sends the engine command `kind` on `requests` and returns its `Hooked` answer's error, empty
/// when every rank ran it.
fn hooked(requests: &mut Connection, id: u64, kind: wire::engine_command::Kind) -> String {
    requests
        .deliver(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(kind),
            })),
        })
        .unwrap();
    until(requests, READ, |kind| match kind {
        from_engine::Kind::Hooked(hooked) if hooked.id == id => Some(hooked.error),
        _ => None,
    })
}

#[test]
fn an_engine_started_on_the_meta_device_serves_and_shuts_down() {
    let directory = tempfile::tempdir().unwrap();
    let storage = StorageServer::start();
    let build_server = build_server(directory.path());
    let launch = launch(directory.path(), &build_server);
    let Daemon {
        mut engine,
        mut requests,
        mut processes,
    } = started(directory.path(), &launch, &storage, &[]);

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
        .deliver(&wire::ToEngine {
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
        .deliver(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::Shutdown(wire::Shutdown {})),
            })),
        })
        .unwrap();
    let status = engine.exited(READ);
    assert!(status.success(), "the engine exited with {status}");

    // The epoch's session is open no more: a head of the epoch opens one again.
    assert!(!holds_session(&storage, launch.epoch, launch.engine));
}

/// The launch of the main process of host `host` in an epoch whose one worker group, of tensor
/// degree 2, has rank 0 on host `a` and rank 1 on host `b`: the Qwen3-MoE fixture on the meta
/// device, one scheduler, as engine 1 (the scheduler node, which holds rank 0) or 2 (its peer,
/// which holds rank 1). Both hosts are this one, each process with sockets of its own in
/// `directory` and a Storage Server of its own.
fn two_host_launch(directory: &Path, build_server: &Path, host: &str, world: World) -> EngineArgs {
    let mut launch = launch(directory, build_server);
    let group = WorkerGroupConfig::placed(
        &["a".to_owned(), "b".to_owned()],
        "cuda",
        2,
        2,
        WorkerGroupConfig::single_component(proto::config::DEFAULT_BLOCK, 2),
    );
    let parallel = &mut launch.engine_config.parallel;
    parallel.tensor_parallel_size = std::num::NonZeroUsize::new(2).unwrap();
    parallel.engine_host = "a".to_owned();
    parallel.worker_groups = vec![group];

    let rank = u32::from(host == "b");
    launch.engine = u64::from(rank) + 1;
    launch.scheduler_node = rank == 0;
    launch.rank = rank;
    launch.world_size = 2;
    launch.host = host.to_owned();
    launch.world = Some(world);
    launch
}

#[test]
fn an_epoch_of_two_processes_of_one_host_standing_for_two_hosts_serves_and_shuts_down() {
    let directory = tempfile::tempdir().unwrap();
    let (here, there) = (directory.path().join("a"), directory.path().join("b"));
    std::fs::create_dir_all(&here).unwrap();
    std::fs::create_dir_all(&there).unwrap();
    // Each host's Storage Server.
    let (storage_a, storage_b) = (StorageServer::start(), StorageServer::start());
    let build_server = build_server(directory.path());

    // The two sockets the daemon reserves on the scheduler node for the epoch: the peer link's
    // listener and the rendezvous of the ranks' host group. The scheduler node inherits both; its
    // peer dials them.
    let channel = TcpListener::bind("127.0.0.1:0").unwrap();
    let store = TcpListener::bind("127.0.0.1:0").unwrap();
    let world = |inherited| World {
        scheduler: 1,
        channel: channel.local_addr().unwrap().to_string(),
        store: store.local_addr().unwrap().to_string(),
        peers: 1,
        inherited,
    };
    let inherited = Inherited {
        channel: channel.as_raw_fd(),
        store: store.as_raw_fd(),
    };
    let head = two_host_launch(&here, &build_server, "a", world(Some(inherited)));
    let peer = two_host_launch(&there, &build_server, "b", world(None));

    // The scheduler node accepts its peer before its rank loads; the peer registers once its
    // daemon answered its claim.
    let mut a = started(
        &here,
        &head,
        &storage_a,
        &[inherited.channel, inherited.store],
    );
    let mut b = started(&there, &peer, &storage_b, &[]);

    // Both serve: the scheduler node with its one scheduler, its peer with none, each having
    // started its first generation.
    let info = until(&mut a.requests, START, |kind| match kind {
        from_engine::Kind::State(state) if state.state == wire::EngineState::Serving as i32 => {
            Some(state.info.expect("a serving engine states what it serves"))
        }
        _ => None,
    });
    assert_eq!(info.schedulers, 1);
    for daemon in [&mut a, &mut b] {
        let serving: proto::process::Upstream = daemon.processes.recv(START).unwrap().unwrap();
        assert!(matches!(
            serving.kind,
            Some(upstream::Kind::Report(proto::process::Report {
                kind: Some(report::Kind::Serving(proto::process::Serving {
                    generation: 1,
                    ..
                })),
            }))
        ));
    }

    // The scheduler node holds the epoch's session; the peer's host holds the session of its
    // host regions, where the peer's ranks take their parts.
    assert!(holds_session(&storage_a, head.epoch, head.engine));
    assert!(holds_session(&storage_b, peer.epoch, peer.engine));

    // One request is accepted by the one scheduler, whose group's batches reach both ranks.
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
    a.requests
        .deliver(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Submit(wire::Submit {
                requests: vec![request.to_wire()],
            })),
        })
        .unwrap();
    let accepted = until(&mut a.requests, READ, |kind| match kind {
        from_engine::Kind::Accepted(accepted) => Some(accepted),
        _ => None,
    });
    assert_eq!(accepted.scheduler, 0);

    // A sleep and a wake reach the peer's ranks as well as the scheduler node's.
    let resize = |id, change: wire::CacheChange| {
        wire::engine_command::Kind::ResizeCache(wire::ResizeCache {
            id,
            change: change as i32,
            ..Default::default()
        })
    };
    let slept = hooked(&mut a.requests, 21, resize(21, wire::CacheChange::Sleep));
    assert_eq!(slept, "");
    let woken = hooked(&mut a.requests, 22, resize(22, wire::CacheChange::Wake));
    assert_eq!(woken, "");

    // A shutdown of the scheduler node ends the epoch in order: its peer reads the close of its
    // link and ends too, both with status 0, and neither host holds a session of the epoch.
    a.requests
        .deliver(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::Shutdown(wire::Shutdown {})),
            })),
        })
        .unwrap();
    let status = a.engine.exited(READ);
    assert!(status.success(), "the scheduler node exited with {status}");
    let status = b.engine.exited(READ);
    assert!(status.success(), "the peer exited with {status}");
    assert!(!holds_session(&storage_a, head.epoch, head.engine));
    assert!(!holds_session(&storage_b, peer.epoch, peer.engine));
}
