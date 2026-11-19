//! The engine executable started as a daemon starts it, end to end on the meta device.
//!
//! The test plays the daemon on two Unix sockets (the request interface and the process
//! interface) and runs a Storage Server and a Build Server on threads of its own process, the
//! Build Server holding the tree's `arrival` policy package. It starts the built `uniserve-engine`
//! with a launch of the Qwen3-MoE fixture on one rank of one group, `device = "meta"`: the
//! engine says hello and `Starting`, claims the node's Storage Server, starts its rank, opens
//! its session, resolves its policy and serves; it states its metrics, takes a request while
//! asleep, which it holds (a meta engine computes nothing, so no request may run on it), and
//! exits with status 0 once shut down, its session closed. The others start an epoch of two
//! processes of this host standing for two hosts, each with a Storage Server of its own: a text
//! group with a rank on each; and released media checkpoints (which the repository does not
//! hold, so those cases are ignored by default) as the two-host layouts of `testkit::layouts`
//! place them, a replica on each host, and one replica across both whose products cross between
//! the two processes.
//!
//! No GPU is touched, but by one ignored case: an engine of the released Qwen3-30B-A3B on two GPUs
//! of this host, its kernels and policy from the host's `device-cases` Build Server, that sleeps
//! and is shut down asleep. Otherwise the Build Server builds the `arrival` module for wasm32 from
//! its installed snapshot of the workspace, and the engine compiles the module when it loads it.
#![cfg(test)]

use std::net::TcpListener;
use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd, OwnedFd, RawFd};
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::sync::Arc;
use std::time::{Duration, Instant};

use proto::config::{EngineArgs, Inherited, WorkerGroupConfig, World};
use proto::process::{downstream, report, upstream};
use proto::request::wire::{self, from_engine};
use rustix::event::{PollFd, PollFlags, Timespec};
use rustix::process::{Pid, PidfdFlags, pidfd_open};
use service::{Connection, Listener};
use testkit::LONG;
use testkit::serving::{DEPLOYMENT, StorageServer};

/// A Build Server on a thread of the test process at `directory`'s socket, holding the
/// workspace's `arrival` policy package (`testkit::policy::package`) as revision 1 with its module
/// built, as a daemon's prepare step leaves
/// a deployment's packages before it starts the engine, so an engine's start resolves a module
/// the store holds. The server serves until the returned stop end is dropped. Returns its socket
/// and that end.
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
        // The policy package is built by cargo, which needs no Python environment.
        source: None,
    };
    let socket = directory.join("build-server.sock");
    // The socket is bound before the server serves, so a client connects at once.
    let (listener, starting) = service::host::bind::<build_server::command::Settings>(&socket)
        .unwrap()
        .owned(&socket)
        .unwrap();
    let lock = starting
        .finish(&build_server::readiness(
            &socket,
            &build_server::command::Settings::default(),
        ))
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
        .install(&testkit::policy::package(proto::config::DEFAULT_SCHEDULING))
        .unwrap();
    assert_eq!(installed.revision, 1);

    // The policy module the launches resolve, built from the package's snapshot now: its first
    // build vendors the snapshot's registry sources and compiles the crate for wasm32.
    let policy = testkit::serving::launch().modules.policy;
    client
        .resolve(
            proto::module::PROGRAM,
            &policy,
            &build_server::Target::wasm(),
            build_server::manifest::SINGLE_KEY,
            &mut |_| {},
        )
        .unwrap();
    (socket, stop)
}

/// The launch of epoch 1's one main process, serving the Qwen3-MoE fixture on one rank of one
/// group on the meta device, with one scheduler, as engine 1 of the scheduler node; its daemon's
/// sockets are in `directory`, and its Build Server's at `build_server`.
fn launch(directory: &Path, build_server: &Path) -> EngineArgs {
    let mut launch = testkit::serving::launch();
    launch.engine_config.device.device = proto::config::DeviceKind::Meta;
    launch.engine = 1;
    launch.scheduler_node = true;
    launch.requests = format!("unix:{}", directory.join("requests.sock").display());
    launch.processes = format!("unix:{}", directory.join("processes.sock").display());
    launch.build_server = build_server.to_string_lossy().into_owned();
    launch
}

/// The engine process, killed when the test ends before it exited, with its pidfd, which is
/// readable once the process exited.
struct Engine {
    child: Child,
    exit: OwnedFd,
}

impl Engine {
    /// Starts `command` as the engine process.
    fn spawn(command: &mut Command) -> Self {
        let child = command.spawn().unwrap();
        let exit = pidfd_open(Pid::from_child(&child), PidfdFlags::empty()).unwrap();
        Self { child, exit }
    }

    /// The process's exit status, once it exited within `bound`.
    fn exited(&mut self, bound: Duration) -> ExitStatus {
        let [exited] = readable([self.exit.as_fd()], bound);
        assert!(exited, "the engine did not exit");
        self.child.wait().unwrap()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Which of `descriptors` became readable within `bound`, waiting until one does.
fn readable<const N: usize>(descriptors: [BorrowedFd<'_>; N], bound: Duration) -> [bool; N] {
    let mut polled =
        descriptors.map(|descriptor| PollFd::from_borrowed_fd(descriptor, PollFlags::IN));
    let timeout = Timespec::try_from(bound).unwrap();
    // A signal that interrupts the wait ends it early; the callers' assertions report what is
    // still not readable.
    match rustix::event::poll(&mut polled, Some(&timeout)) {
        Ok(_) | Err(rustix::io::Errno::INTR) => {}
        Err(error) => panic!("polling the engine's descriptors: {error}"),
    }
    polled.map(|descriptor| !descriptor.revents().is_empty())
}

/// The first connection to `listener` within `bound`, which the engine makes unless it exits.
fn accept(listener: &Listener, engine: &mut Engine, bound: Duration) -> Connection {
    // SAFETY: the listener's descriptor stays open for the borrow, which ends with the poll.
    let socket = unsafe { BorrowedFd::borrow_raw(listener.as_raw_fd()) };
    let [connected, exited] = readable([socket, engine.exit.as_fd()], bound);
    if let Some(connection) = listener.try_accept().unwrap() {
        return connection;
    }
    assert!(
        !exited,
        "the engine exited with {} before it connected",
        engine.child.wait().unwrap()
    );
    panic!("the engine did not connect (a connection waits: {connected})");
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
/// the claim, and the daemon's connection the claim lasts for.
struct Daemon {
    engine: Engine,
    requests: Connection,
    processes: Connection,
    _claim: storage::ClaimClient,
}

/// Starts the built `uniserve-engine` with `launch`, written into `directory` and passed as the
/// inherited descriptor of that file (`proto::LAUNCH_OPTION`), and plays its daemon up to the
/// claim, which it makes at `storage` for the engine and answers with: the engine says hello and
/// `Starting` on its request interface, says hello on its process interface and claims. The process also inherits the
/// descriptors `inherited` open, as the daemon's supervisor places the sockets it reserved for a
/// scheduler node.
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
    // The engine reads its launch whole from the descriptor it inherits; the test's copy closes
    // once the child is spawned.
    let launch_file = std::fs::File::open(&launch_file).unwrap();

    let mut command = Command::new(env!("CARGO_BIN_EXE_uniserve-engine"));
    command
        .arg(proto::LAUNCH_OPTION)
        .arg(launch_file.as_raw_fd().to_string());
    let mut inherited = inherited.to_vec();
    inherited.push(launch_file.as_raw_fd());
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
    let mut engine = Engine::spawn(&mut command);
    drop(launch_file);

    // The request interface: hello, welcome, `Starting`.
    let mut requests = accept(&requests_socket, &mut engine, LONG);
    let hello: wire::Hello = requests.recv(LONG).unwrap().unwrap();
    assert_eq!(hello.engine, launch.engine);
    assert_eq!(hello.interface, proto::process::Interface::Requests as i32);
    requests
        .deliver(&wire::Welcome {
            build: proto::BUILD.to_owned(),
            refusal: None,
            event_window: 64,
        })
        .unwrap();
    let from_engine::Kind::State(starting) = next(&mut requests, LONG) else {
        panic!("the engine's first frame states no state");
    };
    assert_eq!(starting.state, wire::EngineState::Starting as i32);

    // The process interface: hello, welcome, the claim answered with the Storage Server.
    let mut processes = accept(&processes_socket, &mut engine, LONG);
    let hello: proto::process::Hello = processes.recv(LONG).unwrap().unwrap();
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
    let claim: proto::process::Upstream = processes.recv(LONG).unwrap().unwrap();
    let Some(upstream::Kind::Request(request)) = claim.kind else {
        panic!("the engine's first frame on its process interface is no request");
    };
    let Some(proto::process::request::Kind::Claim(claim)) = &request.kind else {
        panic!("the engine's first request is no claim");
    };

    // The claim is the daemon's, at the node's Storage Server, for as long as its connection
    // lasts: what the server holds the storage the engine's processes publish to.
    let scopes = claim
        .scopes
        .iter()
        .map(|scope| storage::Scope {
            deployment: claim.deployment.clone(),
            group: scope.group.clone(),
            rank: scope.rank,
        })
        .collect();
    let mut claim_client =
        storage::ClaimClient::connect(&storage.socket, &claim.deployment).unwrap();
    let token = claim_client
        .claim(
            storage::Claimant::Engine(launch.engine),
            claim.epoch,
            scopes,
            claim.retire.clone(),
        )
        .unwrap();
    processes
        .deliver(&proto::process::Downstream {
            kind: Some(downstream::Kind::Reply(proto::process::Reply {
                id: request.id,
                error: String::new(),
                kind: Some(proto::process::reply::Kind::Claimed(
                    proto::process::Claimed {
                        socket: storage.socket.to_string_lossy().into_owned(),
                        claim: token.0.to_vec(),
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
        _claim: claim_client,
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
    };
    match head.open(empty) {
        Ok(_) => false,
        Err(error) => format!("{error:#}").contains("is open"),
    }
}

/// Sends the engine command `kind` on `requests` and returns its `Operated` answer's error, empty
/// when every rank ran it.
fn operated(requests: &mut Connection, id: u64, kind: wire::engine_command::Kind) -> String {
    requests
        .deliver(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(kind),
            })),
        })
        .unwrap();
    until(requests, LONG, |kind| match kind {
        from_engine::Kind::Operated(operated) if operated.id == id => Some(operated.error),
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
        _claim: _held,
    } = started(directory.path(), &launch, &storage, &[]);

    // The start ends serving: `Serving` with what it serves on the request interface, and the
    // first generation on the process interface.
    let info = until(&mut requests, LONG, |kind| match kind {
        from_engine::Kind::State(state) if state.state == wire::EngineState::Serving as i32 => {
            Some(state.info.expect("a serving engine states what it serves"))
        }
        _ => None,
    });
    assert_eq!(info.schedulers, 1);
    let serving: proto::process::Upstream = processes.recv(LONG).unwrap().unwrap();
    assert_eq!(
        serving.kind,
        Some(upstream::Kind::Report(proto::process::Report {
            kind: Some(report::Kind::Serving(proto::process::Serving {
                launch_generation: launch.engine,
                generation: 1,
            })),
        }))
    );

    // The metrics, once a second.
    until(&mut requests, LONG, |kind| match kind {
        from_engine::Kind::Metrics(metrics) => Some(metrics),
        _ => None,
    });

    // A meta engine computes nothing, so the request is sent to it asleep: its scheduler takes
    // the request and holds it, forming no batch until a wake, and the shutdown ends it.
    let sleep = wire::engine_command::Kind::ResizeCache(wire::ResizeCache {
        id: 21,
        change: wire::CacheChange::Sleep as i32,
        ..Default::default()
    });
    assert_eq!(operated(&mut requests, 21, sleep), "");
    let request =
        proto::request::Request::Generation(Box::new(proto::request::GenerationRequest {
            trace: 0,
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
            hand_off: false,
            lend: None,
        }));
    requests
        .deliver(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Submit(wire::Submit {
                requests: vec![request.to_wire()],
            })),
        })
        .unwrap();
    let accepted = until(&mut requests, LONG, |kind| match kind {
        from_engine::Kind::Accepted(accepted) => Some(accepted),
        _ => None,
    });
    // The request keeps the identity this connection gave it, and the engine states its own.
    assert_eq!((accepted.request, accepted.scheduler), (11, 0));
    assert_ne!(accepted.engine_request, 0);

    // A shutdown ends the process in order: it exits with status 0, its session closed.
    requests
        .deliver(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::Shutdown(wire::Shutdown {})),
            })),
        })
        .unwrap();
    let status = engine.exited(LONG);
    assert!(status.success(), "the engine exited with {status}");

    // The epoch's session is open no more: a head of the epoch opens one again.
    assert!(!holds_session(&storage, launch.epoch, launch.engine));
}

/// The deployment whose Build Server serves the kernels and the policy of an engine on a device:
/// the device cases' (`uniserve build-server serve --deployment device-cases`, which holds the
/// tree's packages).
const DEVICE_CASES: &str = "device-cases";

/// The bound of each step of an engine on a device: its start reads and maps the release's
/// weights and has the Build Server build the kernels its store lacks, its sleep copies the graph
/// pools to host storage, and its close waits for the devices' work.
const DEVICE_BOUND: Duration = Duration::from_secs(1800);

/// An engine serving the released Qwen3-30B-A3B at tensor degree 2 on GPUs 0 and 1 that sleeps
/// and is then shut down ends in order: the sleep gives back its KV cache and suspends its graph
/// pools, and the shutdown closes every owner of its devices' state, streams included, before
/// the devices' contexts are reset, so the process exits with status 0 and its session closed
/// (`docs/admin.md`, Sleep, and wake; `Engine::close`), and its workers destroy the tensor
/// group's communicators, so none of the NCCL shared memory the serving process mapped is left
/// in `/dev/shm` (`docs/architecture.md`, the orderly close). The driver reads a stream of a reset
/// context through freed state once another device's context was reset too, so the case takes
/// two devices; the FP32 fixture has no kernels on a device.
#[test]
#[ignore = "needs GPUs 0 and 1, this host's device-cases Build Server and the released Qwen3-30B-A3B"]
fn an_engine_on_a_device_shut_down_while_asleep_exits_in_order() {
    let directory = tempfile::tempdir().unwrap();
    let storage = StorageServer::start();
    let build_server =
        build_server::socket(&service::RuntimeDirectory::of_user().unwrap(), DEVICE_CASES).unwrap();
    let mut launch = launch(directory.path(), &build_server);
    launch.engine_config.device.device = proto::config::DeviceKind::Cuda;
    launch.engine_config.model.model = testkit::assets::QWEN3_30B_A3B
        .snapshot()
        .to_string_lossy()
        .into_owned();
    launch.devices = vec![0, 1];
    let parallel = &mut launch.engine_config.parallel;
    parallel.tensor_parallel_size = std::num::NonZeroUsize::new(2).unwrap();
    parallel.worker_groups = vec![WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        proto::config::DeviceKind::Cuda,
        2,
        2,
        WorkerGroupConfig::single_block(proto::config::DEFAULT_BLOCK, 2),
    )];
    // The process interface and the claim stay open until the engine exits, as its daemon's do.
    let Daemon {
        mut engine,
        mut requests,
        processes: _processes,
        _claim: _held,
    } = started(directory.path(), &launch, &storage, &[]);

    // A start on a device captures its graphs and creates its KV cache before it serves.
    until(&mut requests, DEVICE_BOUND, |kind| match kind {
        from_engine::Kind::State(state) if state.state == wire::EngineState::Serving as i32 => {
            Some(())
        }
        _ => None,
    });
    // The tensor group's two ranks are threads of the one process, whose communicators' proxies
    // map NCCL's shared memory while they serve.
    let shared = nccl_shared_memory(engine.child.id());
    assert!(
        !shared.is_empty(),
        "the serving engine maps no NCCL shared memory"
    );

    let sleep = wire::engine_command::Kind::ResizeCache(wire::ResizeCache {
        id: 31,
        change: wire::CacheChange::Sleep as i32,
        ..Default::default()
    });
    requests
        .deliver(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(sleep),
            })),
        })
        .unwrap();
    let slept = until(&mut requests, DEVICE_BOUND, |kind| match kind {
        from_engine::Kind::Operated(operated) if operated.id == 31 => Some(operated.error),
        _ => None,
    });
    assert_eq!(slept, "");

    requests
        .deliver(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Engine(wire::EngineCommand {
                kind: Some(wire::engine_command::Kind::Shutdown(wire::Shutdown {})),
            })),
        })
        .unwrap();
    let status = engine.exited(DEVICE_BOUND);
    assert!(status.success(), "the engine exited with {status}");
    assert!(!holds_session(&storage, launch.epoch, launch.engine));

    let left: Vec<&PathBuf> = shared.iter().filter(|path| path.exists()).collect();
    assert!(
        left.is_empty(),
        "the engine's close left NCCL's shared memory {left:?}"
    );
}

/// The files of NCCL's host shared memory (`/dev/shm/nccl-*`) that process `pid` maps and that
/// are still linked: what its communicators' proxies exchange through, which NCCL removes when
/// it destroys them.
fn nccl_shared_memory(pid: u32) -> std::collections::BTreeSet<PathBuf> {
    let maps = std::fs::read_to_string(format!("/proc/{pid}/maps")).unwrap();

    // A mapping's path is its line's last field; an unlinked file's ends with " (deleted)".
    maps.lines()
        .filter_map(|line| line.split_whitespace().nth(5))
        .filter(|path| path.starts_with("/dev/shm/nccl-"))
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .collect()
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
fn fast_h3() -> PathBuf {
    testkit::assets::FAST_H3_8_STEP.snapshot()
}

/// The base MiniMax-H3 release, loaded from its shard headers alone: its dense denoiser takes an
/// eight-way Ulysses split of the shortest timeline, which the sparse attention of the FastH3
/// exports refuses in the profile run, as it composes rows in blocks the split does not leave
/// whole.
fn minimax_h3() -> PathBuf {
    testkit::assets::MINIMAX_H3.snapshot()
}

/// A media deployment of two hosts: a two-host layout of `testkit::layouts`, whose leader is the
/// host it is given, and the checkpoint it places.
type Deployment = (fn(&str) -> Vec<WorkerGroupConfig>, PathBuf);

/// The launch of the main process of host `host` in an epoch that serves `checkpoint` as
/// `layout`, a two-host layout of `testkit::layouts`, places it on hosts `rank-0` and `rank-1`,
/// four devices each. The meta device, one scheduler, square videos of the shortest timeline and
/// prompts of up to 256 tokens; engine 1 (the scheduler node, `rank-0`) or 2 (its peer,
/// `rank-1`), each serving from the first rank its host holds. Both hosts are this one, each
/// process with sockets of its own in `directory` and a Storage Server of its own.
fn media_two_host_launch(
    directory: &Path,
    build_server: &Path,
    (layout, checkpoint): Deployment,
    host: &str,
    world: World,
) -> EngineArgs {
    let mut launch = launch(directory, build_server);
    let groups = layout("rank-0");
    let hosts: Vec<&str> = groups
        .iter()
        .flat_map(|group| &group.ranks)
        .map(|rank| rank.host.as_str())
        .collect();
    launch.world_size = u32::try_from(hosts.len()).unwrap();
    launch.rank = u32::try_from(hosts.iter().position(|held| *held == host).unwrap()).unwrap();

    let config = &mut launch.engine_config;
    config.model.model = checkpoint.to_string_lossy().into_owned();
    config.model.max_model_len = 256;
    config.parallel.engine_host = "rank-0".to_owned();
    config.parallel.worker_groups = groups;
    config.video = proto::config::VideoConfig {
        max_seconds: Some(4.0),
        sizes: Some(vec!["768x768".to_owned()]),
        ..proto::config::VideoConfig::default()
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
    let info = until(&mut a.requests, LONG, |kind| match kind {
        from_engine::Kind::State(state) if state.state == wire::EngineState::Serving as i32 => {
            Some(state.info.expect("a serving engine states what it serves"))
        }
        _ => None,
    });
    assert_eq!(info.schedulers, 1);
    for daemon in [a, b] {
        let serving: proto::process::Upstream = daemon.processes.recv(LONG).unwrap().unwrap();
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
    let status = a.engine.exited(LONG);
    assert!(status.success(), "the scheduler node exited with {status}");
    let status = b.engine.exited(LONG);
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

    // A sleep and a wake reach the peer's ranks as well as the scheduler node's.
    let resize = |id, change: wire::CacheChange| {
        wire::engine_command::Kind::ResizeCache(wire::ResizeCache {
            id,
            change: change as i32,
            ..Default::default()
        })
    };
    let slept = operated(&mut a.requests, 21, resize(21, wire::CacheChange::Sleep));
    assert_eq!(slept, "");
    let woken = operated(&mut a.requests, 22, resize(22, wire::CacheChange::Wake));
    assert_eq!(woken, "");

    // One request is accepted by the one scheduler, sent while the epoch sleeps again: a meta
    // epoch computes nothing, so no request may run on it, and a sleeping scheduler holds what
    // it takes and forms no batch until a wake.
    let slept = operated(&mut a.requests, 23, resize(23, wire::CacheChange::Sleep));
    assert_eq!(slept, "");
    let request =
        proto::request::Request::Generation(Box::new(proto::request::GenerationRequest {
            trace: 0,
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
            hand_off: false,
            lend: None,
        }));
    a.requests
        .deliver(&wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Submit(wire::Submit {
                requests: vec![request.to_wire()],
            })),
        })
        .unwrap();
    let accepted = until(&mut a.requests, LONG, |kind| match kind {
        from_engine::Kind::Accepted(accepted) => Some(accepted),
        _ => None,
    });
    assert_eq!(accepted.scheduler, 0);

    // A shutdown of the scheduler node ends the epoch in order.
    let storage = [(&storage_a, head.engine), (&storage_b, peer.engine)];
    both_shut_down(&mut a, &mut b, storage, head.epoch);
}

/// Starts an epoch of two processes of this host standing for hosts `rank-0` and `rank-1` that
/// serves `deployment`, a two-host layout of `testkit::layouts` and the checkpoint it places,
/// the scheduler node serving from rank 0 and its peer from `peer_rank`, the first rank of
/// `rank-1`; reads both to `Serving`, each holding its session on its own host's Storage Server,
/// and shuts the epoch down in order.
fn media_epoch_serves_and_shuts_down(deployment: Deployment, peer_rank: u32) {
    let directory = tempfile::tempdir().unwrap();
    let (here, there) = (directory.path().join("a"), directory.path().join("b"));
    std::fs::create_dir_all(&here).unwrap();
    std::fs::create_dir_all(&there).unwrap();
    let (storage_a, storage_b) = (StorageServer::start(), StorageServer::start());
    let (build_server, _serving) = build_server(directory.path());

    let reserved = Reserved::new();
    let inherited = reserved.inherited();
    let world = reserved.world(Some(inherited));
    let head = media_two_host_launch(&here, &build_server, deployment.clone(), "rank-0", world);
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
/// `testkit::layouts::fasth3_ulysses4x2_two_host` places them, passes no product between
/// its two processes: the scheduler node and its peer each start their own host's ranks, and
/// both serve.
#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_media_epoch_whose_replicas_lie_on_two_hosts_serves_and_shuts_down() {
    media_epoch_serves_and_shuts_down((testkit::layouts::fasth3_ulysses4x2_two_host, fast_h3()), 4);
}

/// A media epoch whose one Ulysses8 replica spans two hosts, as
/// `testkit::layouts::ulysses8_two_host` places it, carries the products one host's ranks
/// write and the other's read between its processes: the scheduler node, which holds ranks 0 to
/// 3 of the `model` group and the `muxer` and `reader` groups, and its peer, which holds ranks 4
/// to 7, both start and serve.
#[test]
#[ignore = "reads the headers of the released MiniMax-H3 checkpoint, which the repository does not hold"]
fn a_media_epoch_whose_replica_spans_two_hosts_serves_and_shuts_down() {
    media_epoch_serves_and_shuts_down((testkit::layouts::ulysses8_two_host, minimax_h3()), 4);
}

/// The two-host reference deployment, `testkit::layouts::ulysses8_reference_two_host`,
/// whose text encoder lies on `rank-0`'s four devices and whose latent encoder deals condition
/// units to `rank-1`'s first: both processes start and serve.
#[test]
#[ignore = "reads the headers of the released MiniMax-H3 checkpoint, which the repository does not hold"]
fn the_two_host_reference_epoch_serves_and_shuts_down() {
    media_epoch_serves_and_shuts_down(
        (testkit::layouts::ulysses8_reference_two_host, minimax_h3()),
        4,
    );
}
