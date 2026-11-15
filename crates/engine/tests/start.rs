//! The engine executable started as a daemon starts it, end to end on the meta device.
//!
//! The test plays the daemon on two Unix sockets (the request interface and the process
//! interface) and runs a Storage Server and a Build Server on threads of its own process, the
//! Build Server holding the tree's `fcfs` policy package. It starts the built `uniserve-engine`
//! with a launch of the Qwen3-MoE fixture on one rank of one group, `device = "meta"`: the
//! engine says hello and `Starting`, claims the node's Storage Server, starts its rank, opens
//! its session, resolves its policy and serves; it takes a request, states its metrics, and
//! exits with status 0 once shut down, its session closed. The others start an epoch of two
//! processes of this host standing for two hosts, each with a Storage Server of its own: a text
//! group with a rank on each; and released media checkpoints (which the repository does not
//! hold, so those cases are ignored by default) as the two-host files of `configs/minimax_h3/`
//! place them, a replica on each host, and one replica across both whose products cross between
//! the two processes.
//!
//! No GPU is touched. The Build Server builds the `fcfs` module for wasm32 from its installed
//! snapshot of the workspace, and the engine compiles the module when it loads it.
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

/// A bound on the engine's start: the Build Server vendors the snapshot's registry sources and
/// builds the policy module from its snapshot on the first resolve.
const START: Duration = Duration::from_secs(900);

/// A Build Server on a thread of the test process at `directory`'s socket, holding the tree's
/// `fcfs` policy package as revision 1, which serves until the returned stop end is dropped.
/// Returns its socket and that end.
fn build_server(directory: &Path) -> (PathBuf, std::os::unix::net::UnixStream) {
    let config = build_server::server::ServerConfig {
        package_root: directory.join("packages"),
        store_roots: vec![build_server::store::Root {
            path: directory.join("store"),
            writable: true,
        }],
        upstream: None,
        read_only_stores: Vec::new(),
        limits: build_server::builder::Limits::default(),
        targets: Vec::new(),
        cache: directory.join("cache"),
    };
    let socket = directory.join("build-server.sock");
    // The socket is bound before the server serves, so a client connects at once.
    let (listener, starting) = service::host::bind::<()>(&socket)
        .unwrap()
        .owned(&socket)
        .unwrap();
    let lock = starting
        .finish(&build_server::readiness(&socket))
        .unwrap();
    let server = Arc::new(build_server::server::BuildServer::open(&config).unwrap());
    let (stop, stopped) = std::os::unix::net::UnixStream::pair().unwrap();
    std::thread::spawn(move || {
        if let Err(error) = server.serve_on(listener, lock, &stopped) {
            eprintln!("the Build Server stopped: {error}");
        }
    });

    let mut client = build_server::protocol::Client::connect(&socket).unwrap();
    let installed = client
        .install(&workspace().join("crates/scheduling/policies/fcfs"))
        .unwrap();
    assert_eq!(installed.revision, 1);
    (socket, stop)
}

/// The launch of epoch 1's one main process, serving the Qwen3-MoE fixture on one rank of one
/// group on the meta device, with one scheduler, as engine 1 of the scheduler node; its daemon's
/// sockets are in `directory`, and its Build Server's at `build_server`.
fn launch(directory: &Path, build_server: &Path) -> EngineArgs {
    let mut launch = support::launch();
    launch.engine_config.device.device = proto::config::DeviceKind::Meta;
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
    let requests_socket = Listener::bind(&launch.requests.parse().unwrap()).unwrap();
    let processes_socket = Listener::bind(&launch.processes.parse().unwrap()).unwrap();
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
            build: proto::BUILD.to_owned(),
            refusal: None,
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
            build: proto::BUILD.to_owned(),
            refusal: None,
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
    let (build_server, _serving) = build_server(directory.path());
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
        proto::config::DeviceKind::Cuda,
        2,
        2,
        WorkerGroupConfig::single_block(proto::config::DEFAULT_BLOCK, 2),
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

/// The released FastH3 checkpoint a media launch serves, loaded from its shard headers alone.
const FAST_H3: &str = "/workspace/models/FastVideo-FastH3-8-Step-V2";

/// The base MiniMax-H3 release, loaded from its shard headers alone: its dense denoiser takes an
/// eight-way Ulysses split of the shortest timeline, which the sparse attention of the FastH3
/// exports refuses in the profile run, as it composes rows in blocks the split does not leave
/// whole.
const MINIMAX_H3: &str = "/workspace/models/MiniMax-H3";

/// The launch of the main process of host `host` in an epoch that serves `checkpoint` as `file`,
/// a deployment file of `configs/minimax_h3/`, places it on hosts `rank-0` and `rank-1`, four
/// devices each. The meta device, one scheduler, square videos of the shortest timeline and
/// prompts of up to 256 tokens; engine 1 (the scheduler node, `rank-0`) or 2 (its peer,
/// `rank-1`), each serving from the first rank its host holds. Both hosts are this one, each
/// process with sockets of its own in `directory` and a Storage Server of its own.
fn media_two_host_launch(
    directory: &Path,
    build_server: &Path,
    (file, checkpoint): (&str, &str),
    host: &str,
    world: World,
) -> EngineArgs {
    let mut launch = launch(directory, build_server);
    let file = workspace().join("configs/minimax_h3").join(file);
    let text = std::fs::read_to_string(file).unwrap();
    let groups = WorkerGroupConfig::parse_file(&text, "rank-0").unwrap();
    let hosts: Vec<&str> = groups
        .iter()
        .flat_map(|group| &group.ranks)
        .map(|rank| rank.host.as_str())
        .collect();
    launch.world_size = u32::try_from(hosts.len()).unwrap();
    launch.rank = u32::try_from(hosts.iter().position(|held| *held == host).unwrap()).unwrap();

    let config = &mut launch.engine_config;
    config.model.model = checkpoint.to_owned();
    config.model.max_model_len = 256;
    config.parallel.engine_host = "rank-0".to_owned();
    config.parallel.worker_groups = groups;
    config.video = proto::config::VideoConfig {
        max_seconds: Some(4.0),
        sizes: Some(vec!["768x768".to_owned()]),
        text_capacities: None,
        max_condition_rows: None,
    };

    let scheduler_node = host == "rank-0";
    launch.engine = if scheduler_node { 1 } else { 2 };
    launch.scheduler_node = scheduler_node;
    launch.host = host.to_owned();
    launch.devices = vec![0, 1, 2, 3];
    launch.world = Some(world);
    launch
}

/// The two sockets the daemon reserves on the scheduler node for an epoch of two hosts: the peer
/// link's listener and the rendezvous of the ranks' host group. The scheduler node inherits both;
/// its peer dials them.
struct Reserved {
    channel: TcpListener,
    store: TcpListener,
}

impl Reserved {
    fn new() -> Self {
        Self {
            channel: TcpListener::bind("127.0.0.1:0").unwrap(),
            store: TcpListener::bind("127.0.0.1:0").unwrap(),
        }
    }

    /// The descriptors the scheduler node inherits.
    fn inherited(&self) -> Inherited {
        Inherited {
            channel: self.channel.as_raw_fd(),
            store: self.store.as_raw_fd(),
        }
    }

    /// Where the epoch's two processes meet: with the inherited descriptors for the scheduler
    /// node, without for its peer.
    fn world(&self, inherited: Option<Inherited>) -> World {
        World {
            scheduler: 1,
            channel: self.channel.local_addr().unwrap().to_string(),
            store: self.store.local_addr().unwrap().to_string(),
            peers: 1,
            inherited,
        }
    }
}

/// Reads the start of an epoch of two processes to its end: the scheduler node `a` states
/// `Serving` with its one scheduler, and both report their first generation to their daemons.
fn both_serve(a: &mut Daemon, b: &mut Daemon) {
    let info = until(&mut a.requests, START, |kind| match kind {
        from_engine::Kind::State(state) if state.state == wire::EngineState::Serving as i32 => {
            Some(state.info.expect("a serving engine states what it serves"))
        }
        _ => None,
    });
    assert_eq!(info.schedulers, 1);
    for daemon in [a, b] {
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
}

/// Shuts the scheduler node `a` down, which ends the epoch in order: its peer `b` reads the close
/// of its link and ends too, both with status 0, and neither host's Storage Server (`storage`, the
/// scheduler node's and the peer's, each with the launch generation of its host's engine) holds
/// that engine's session of `epoch`.
fn both_shut_down(a: &mut Daemon, b: &mut Daemon, storage: [(&StorageServer, u64); 2], epoch: u64) {
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
    for (storage, engine) in storage {
        assert!(!holds_session(storage, epoch, engine));
    }
}

#[test]
fn an_epoch_of_two_processes_of_one_host_standing_for_two_hosts_serves_and_shuts_down() {
    let directory = tempfile::tempdir().unwrap();
    let (here, there) = (directory.path().join("a"), directory.path().join("b"));
    std::fs::create_dir_all(&here).unwrap();
    std::fs::create_dir_all(&there).unwrap();
    // Each host's Storage Server.
    let (storage_a, storage_b) = (StorageServer::start(), StorageServer::start());
    let (build_server, _serving) = build_server(directory.path());

    let reserved = Reserved::new();
    let inherited = reserved.inherited();
    let head = two_host_launch(&here, &build_server, "a", reserved.world(Some(inherited)));
    let peer = two_host_launch(&there, &build_server, "b", reserved.world(None));

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
    both_serve(&mut a, &mut b);

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

    // A shutdown of the scheduler node ends the epoch in order.
    let storage = [(&storage_a, head.engine), (&storage_b, peer.engine)];
    both_shut_down(&mut a, &mut b, storage, head.epoch);
}

/// Starts an epoch of two processes of this host standing for hosts `rank-0` and `rank-1` that
/// serves `deployment`, a deployment file of `configs/minimax_h3/` and the checkpoint it places,
/// the scheduler node serving from rank 0 and its peer from `peer_rank`, the first rank of
/// `rank-1`; reads both to `Serving`, each holding its session on its own host's Storage Server,
/// and shuts the epoch down in order.
fn media_epoch_serves_and_shuts_down(deployment: (&str, &str), peer_rank: u32) {
    let checkpoint = deployment.1;
    assert!(Path::new(checkpoint).is_dir(), "{checkpoint} is missing");
    let directory = tempfile::tempdir().unwrap();
    let (here, there) = (directory.path().join("a"), directory.path().join("b"));
    std::fs::create_dir_all(&here).unwrap();
    std::fs::create_dir_all(&there).unwrap();
    let (storage_a, storage_b) = (StorageServer::start(), StorageServer::start());
    let (build_server, _serving) = build_server(directory.path());

    let reserved = Reserved::new();
    let inherited = reserved.inherited();
    let world = reserved.world(Some(inherited));
    let head = media_two_host_launch(&here, &build_server, deployment, "rank-0", world);
    let world = reserved.world(None);
    let peer = media_two_host_launch(&there, &build_server, deployment, "rank-1", world);
    assert_eq!((head.rank, peer.rank), (0, peer_rank));

    let mut a = started(
        &here,
        &head,
        &storage_a,
        &[inherited.channel, inherited.store],
    );
    let mut b = started(&there, &peer, &storage_b, &[]);
    both_serve(&mut a, &mut b);
    assert!(holds_session(&storage_a, head.epoch, head.engine));
    assert!(holds_session(&storage_b, peer.epoch, peer.engine));

    let storage = [(&storage_a, head.engine), (&storage_b, peer.engine)];
    both_shut_down(&mut a, &mut b, storage, head.epoch);
}

/// A media epoch whose two replicas each lie on one of two hosts, as
/// `configs/minimax_h3/fasth3-ulysses4x2-two-host.toml` places them, passes no product between
/// its two processes: the scheduler node and its peer each start their own host's ranks, and
/// both serve.
#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_media_epoch_whose_replicas_lie_on_two_hosts_serves_and_shuts_down() {
    media_epoch_serves_and_shuts_down(("fasth3-ulysses4x2-two-host.toml", FAST_H3), 4);
}

/// A media epoch whose one Ulysses8 replica spans two hosts, as
/// `configs/minimax_h3/ulysses8-two-host.toml` places it, carries the products one host's ranks
/// write and the other's read between its processes: the scheduler node, which holds ranks 0 to
/// 3 of the `model` group and the `muxer` and `reader` groups, and its peer, which holds ranks 4
/// to 7, both start and serve.
#[test]
#[ignore = "reads the headers of the released MiniMax-H3 checkpoint, which the repository does not hold"]
fn a_media_epoch_whose_replica_spans_two_hosts_serves_and_shuts_down() {
    media_epoch_serves_and_shuts_down(("ulysses8-two-host.toml", MINIMAX_H3), 4);
}

/// The two-host reference deployment, `configs/minimax_h3/ulysses8-reference-two-host.toml`,
/// whose text encoder lies on `rank-0`'s four devices and whose latent encoder deals condition
/// units to `rank-1`'s first: both processes start and serve.
#[test]
#[ignore = "reads the headers of the released MiniMax-H3 checkpoint, which the repository does not hold"]
fn the_two_host_reference_epoch_serves_and_shuts_down() {
    media_epoch_serves_and_shuts_down(("ulysses8-reference-two-host.toml", MINIMAX_H3), 4);
}
