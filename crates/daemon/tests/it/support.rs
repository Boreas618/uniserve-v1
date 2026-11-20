//! An engine's side of the process interface, which the tests drive a daemon with; the stand-in
//! Build Server a daemon's node resolves its engines at; and the stand-in engines it starts, which
//! are the test crate's `stand-in-engine` binary.

use std::collections::VecDeque;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use daemon::DaemonOptions;
use proto::process::{
    Downstream, Engine as EnginePeer, Exit, Hello, Reply, Request, Reserve, Reserved, Spawn, Stop,
    Upstream, Welcome, downstream, event, hello, reply, request, upstream,
};
use service::Connection;

/// The engine packages the stand-in engine (`tests/bin/stand_in_engine.rs`) stands for, by the
/// backend a launch's module set names.
pub(crate) mod package {
    /// Exits with status 1.
    pub(crate) const FAILS: &str = "fails";
    /// Exits with status 0.
    pub(crate) const SUCCEEDS: &str = "succeeds";
    /// Runs until its lifeline ends or it is killed.
    pub(crate) const FOLLOWS_LIFELINE: &str = "follows-lifeline";
    /// Serves one peer on the store socket its launch's world states it inherited.
    pub(crate) const SERVES_STORE: &str = "serves-store";
}

/// The deployment the cases' daemons serve, whose servers the stand-ins are.
pub(crate) const DEPLOYMENT: &str = "deployment";

/// The options of the single node `host`, whose process socket is `socket` and whose
/// deployment's Build Server is `build_server`, in the node's runtime directory.
pub(crate) fn single_node(
    host: &str,
    socket: PathBuf,
    build_server: &BuildServerStandIn,
) -> DaemonOptions {
    DaemonOptions {
        deployment: DEPLOYMENT.to_owned(),
        runtime: build_server.runtime(),
        gpus: build_server.runtime().path().join("gpus"),
        nnodes: 1,
        node_rank: 0,
        host: host.to_owned(),
        dist_init_addr: None,
        socket,
        join_timeout: daemon::JOIN_TIMEOUT,
        build_server: build_server.settings(),
        storage: storage::Settings::default(),
    }
}

/// An engine connected to a leader's process socket.
pub(crate) struct Engine {
    connection: Connection,
    next: u64,
    /// Exits that arrived while a reply was awaited.
    exits: VecDeque<Exit>,
}

impl Engine {
    /// Connects to the leader at `address` as an engine of this build.
    pub(crate) fn connect(address: &str) -> Self {
        let (engine, welcome) = Self::hello(address, proto::ENGINE_INTERFACE);
        assert_eq!(welcome.refusal, None, "the leader refused this engine");
        engine
    }

    /// Connects to the leader at `address`, saying hello as an engine whose
    /// build speaks the window `versions`, and returns the leader's welcome.
    pub(crate) fn hello(address: &str, versions: service::Versions) -> (Self, Welcome) {
        let mut engine = Self::dial(address, versions);
        let welcome = engine.welcome();
        (engine, welcome)
    }

    /// Connects to the leader at `address` as an engine of this build that
    /// sends its first request, `kind`, right behind its hello rather than
    /// after the welcome, and returns the request's answer.
    pub(crate) fn connect_asking(address: &str, kind: request::Kind) -> (Self, Reply) {
        let mut engine = Self::dial(address, proto::ENGINE_INTERFACE);
        let id = engine.send(kind);
        assert_eq!(
            engine.welcome().refusal,
            None,
            "the leader refused this engine"
        );
        let reply = engine.reply(id);
        (engine, reply)
    }

    /// Connects to the leader at `address`, which is bound once its daemon's start returned, and
    /// says hello as an engine whose build speaks the window `versions`.
    fn dial(address: &str, versions: service::Versions) -> Self {
        let mut connection = Connection::connect(&address.parse().unwrap())
            .unwrap_or_else(|error| panic!("the leader is not reachable at {address}: {error}"));
        connection
            .deliver(&Hello {
                oldest: versions.oldest(),
                newest: versions.newest(),
                build: proto::BUILD.to_owned(),
                peer: Some(hello::Peer::Engine(EnginePeer {
                    deployment: "deployment".into(),
                    // The harness client asks and claims; it is not a main process, so it names no
                    // engine number, no interface and no link of one, and no command is addressed to
                    // it.
                    engine: 0,
                    interface: 0,
                })),
            })
            .unwrap();
        Self {
            connection,
            next: 1,
            exits: VecDeque::new(),
        }
    }

    /// The leader's answer to the hello.
    fn welcome(&mut self) -> Welcome {
        self.connection
            .recv_next::<Welcome>()
            .expect("the leader answers a hello")
    }

    /// Sends one request and returns its answer.
    pub(crate) fn request(&mut self, kind: request::Kind) -> Reply {
        let id = self.send(kind);
        self.reply(id)
    }

    /// Sends one request and returns its id, without waiting for its answer
    /// ([`Engine::reply`]).
    pub(crate) fn send(&mut self, kind: request::Kind) -> u64 {
        let id = self.next;
        self.next += 1;
        self.connection
            .deliver(&Upstream {
                kind: Some(upstream::Kind::Request(Request {
                    id,
                    kind: Some(kind),
                })),
            })
            .unwrap();
        id
    }

    /// The answer to request `id`, keeping the exits that arrive before it.
    pub(crate) fn reply(&mut self, id: u64) -> Reply {
        loop {
            match self.receive() {
                downstream::Kind::Reply(reply) => {
                    assert_eq!(reply.id, id, "the daemon answered another request");
                    return reply;
                }
                downstream::Kind::Event(event) => {
                    if let Some(event::Kind::Exit(exit)) = event.kind {
                        self.exits.push_back(exit);
                    }
                }
            }
        }
    }

    /// Starts `spawn` and returns its process id.
    pub(crate) fn spawn(&mut self, spawn: Spawn) -> u32 {
        match self.request(request::Kind::Spawn(spawn)) {
            Reply {
                kind: Some(reply::Kind::Spawned(spawned)),
                ..
            } => spawned.pid,
            other => panic!("the spawn failed: {other:?}"),
        }
    }

    /// Starts `spawn`, a process that runs until it is ended, and returns it.
    pub(crate) fn start(&mut self, spawn: Spawn) -> Process {
        Process::of(self.spawn(spawn))
    }

    /// Reserves a socket on `host` for `group`.
    pub(crate) fn reserve(&mut self, host: &str, group: &str, routable: bool) -> Reserved {
        let reserve = Reserve {
            host: host.into(),
            group: group.into(),
            routable,
            // Reserved for the owner the asking connection is served for.
            engine: None,
        };
        match self.request(request::Kind::Reserve(reserve)) {
            Reply {
                kind: Some(reply::Kind::Reserved(reserved)),
                ..
            } => reserved,
            other => panic!("the reservation failed: {other:?}"),
        }
    }

    /// Stops `group` on every node.
    pub(crate) fn stop(&mut self, group: &str) {
        let reply = self.request(request::Kind::Stop(Stop {
            group: group.into(),
            generation: None,
        }));
        assert_eq!(reply.error, "", "the stop failed");
        assert!(matches!(reply.kind, Some(reply::Kind::Stopped(_))));
    }

    /// How many exits arrived that nothing read yet.
    pub(crate) fn reported_exits(&self) -> usize {
        self.exits.len()
    }

    /// Waits for the next exit the daemon reports.
    pub(crate) fn exit(&mut self) -> Exit {
        if let Some(exit) = self.exits.pop_front() {
            return exit;
        }
        loop {
            if let downstream::Kind::Event(event) = self.receive()
                && let Some(event::Kind::Exit(exit)) = event.kind
            {
                return exit;
            }
        }
    }

    fn receive(&mut self) -> downstream::Kind {
        let message: Downstream = self
            .connection
            .recv_next()
            .expect("the daemon closed the connection");
        message.kind.expect("the daemon sent an empty message")
    }
}

/// The launch of a stand-in engine of package `package` as `rank` of `group`'s launch
/// `generation` on `host`: an engine's launch, whose module set names the stand-in engine package
/// the node resolves at its Build Server.
pub(crate) fn launch(host: &str, group: &str, rank: u32, generation: u64, package: &str) -> Spawn {
    let mut args = engine_args(package);
    args.engine = generation;
    Spawn {
        host: host.into(),
        group: group.into(),
        rank,
        generation,
        inherit: Vec::new(),
        launch: serde_json::to_vec(&args).unwrap(),
    }
}

/// The launch of a stand-in engine that runs until it is killed or its daemon ends.
pub(crate) fn sleeper(host: &str, group: &str, rank: u32, generation: u64) -> Spawn {
    launch(host, group, rank, generation, package::FOLLOWS_LIFELINE)
}

/// The arguments of a stand-in engine of package `package`: a default deployment's engine
/// configuration with a module set whose engine package is `package`.
pub(crate) fn engine_args(package: &str) -> proto::config::EngineArgs {
    let config = daemon::config::Config::default()
        .engine_config()
        .expect("the default configuration resolves");
    let mut modules = module_set();
    modules.engine.backend = package.to_owned();
    proto::config::EngineArgs::new(config, modules)
}

/// A process a daemon started, observed through its pidfd, which names it whatever process
/// later takes its id and turns readable once it exited.
pub(crate) struct Process {
    pidfd: OwnedFd,
}

impl Process {
    /// The process `pid`, which runs.
    pub(crate) fn of(pid: u32) -> Self {
        // SAFETY: the call takes integers only.
        let descriptor = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0) };
        assert!(
            descriptor >= 0,
            "process {pid} is gone: {}",
            std::io::Error::last_os_error()
        );
        Self {
            // SAFETY: the call returned a new descriptor this test owns.
            pidfd: unsafe { OwnedFd::from_raw_fd(descriptor as libc::c_int) },
        }
    }

    /// Whether the process runs: its pidfd is not readable.
    pub(crate) fn runs(&self) -> bool {
        !self.poll(0)
    }

    /// Waits until the process exited.
    pub(crate) fn wait(&self) {
        self.poll(-1);
    }

    /// Whether the pidfd turned readable within `milliseconds` (forever for -1).
    fn poll(&self, milliseconds: libc::c_int) -> bool {
        let mut polled = libc::pollfd {
            fd: self.pidfd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        loop {
            // SAFETY: one valid pollfd for the call.
            let ready = unsafe { libc::poll(&mut polled, 1, milliseconds) };
            if ready >= 0 {
                return ready > 0;
            }
            assert_eq!(
                std::io::Error::last_os_error().kind(),
                std::io::ErrorKind::Interrupted
            );
        }
    }
}

/// A stand-in for the deployment's Build Server on a node, which a daemon prepares an epoch's
/// module set at and resolves its engines at: it serves `prepare` at its socket and answers every slot with
/// the package revision the request fixed for it, raised by `raise`, as a Build Server that holds
/// every module answers; and it resolves every engine package to the stand-in engine. Each
/// connection is served on a thread of its own, as the Build Server serves them.
pub(crate) struct BuildServerStandIn {
    /// The node's runtime directory, which holds the deployment's directory and the socket.
    runtime: service::RuntimeDirectory,
    socket: PathBuf,
    /// One token per `prepare` that arrived.
    asked: Receiver<()>,
    /// One token per engine package resolved: a node resolves it while it serves a spawn,
    /// before it starts the process and answers.
    resolved: Receiver<()>,
    /// One token lets one held `prepare` answer.
    release: Sender<()>,
}

impl BuildServerStandIn {
    /// Serves as the Build Server of [`DEPLOYMENT`] in the runtime directory within `node`, a
    /// node's own directory, owning its socket as a Build Server does; with `held`, each
    /// `prepare` answers only once [`Self::release`] lets it, as one building the engine's
    /// executable does.
    pub(crate) fn serve(node: &std::path::Path, held: bool, raise: u32) -> Self {
        Self::serve_for(node, DEPLOYMENT, held, raise)
    }

    /// Serves as the Build Server of `deployment` in the runtime directory within `node`, as
    /// [`Self::serve`] serves [`DEPLOYMENT`]'s.
    pub(crate) fn serve_for(
        node: &std::path::Path,
        deployment: &str,
        held: bool,
        raise: u32,
    ) -> Self {
        std::fs::create_dir_all(node).unwrap();
        let runtime = service::RuntimeDirectory::within(node).unwrap();
        let socket = build_server::socket(&runtime, deployment).unwrap();
        let (listener, starting) = service::host::bind::<build_server::command::Settings>(&socket)
            .unwrap()
            .owned(&socket)
            .expect("no server owns the stand-in's socket");
        let lock = starting
            .finish(&build_server::readiness(
                &socket,
                &build_server::command::Settings::default(),
            ))
            .unwrap();
        let (asked_sender, asked) = std::sync::mpsc::channel();
        let (resolved_sender, resolved) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let released = Arc::new(Mutex::new(released));
        std::thread::spawn(move || {
            let _lock = lock;
            for stream in listener.incoming() {
                let Ok(stream) = stream else {
                    return;
                };
                let asked = asked_sender.clone();
                let resolved = resolved_sender.clone();
                let released = Arc::clone(&released);
                std::thread::spawn(move || {
                    answer(
                        Connection::new(stream).unwrap(),
                        held,
                        raise,
                        (&asked, &resolved),
                        &released,
                    );
                });
            }
        });
        Self {
            runtime,
            socket,
            asked,
            resolved,
            release,
        }
    }

    /// The settings of the Build Server a daemon of this node states, which this one serves with.
    pub(crate) fn settings(&self) -> build_server::command::Settings {
        build_server::command::Settings::default()
    }

    /// The runtime directory of the node this Build Server serves.
    pub(crate) fn runtime(&self) -> service::RuntimeDirectory {
        self.runtime.clone()
    }

    /// Its socket.
    pub(crate) fn socket(&self) -> PathBuf {
        self.socket.clone()
    }

    /// Waits until a `prepare` arrived.
    pub(crate) fn asked(&self) {
        self.asked
            .recv()
            .expect("the Build Server was asked to prepare");
    }

    /// Waits until an engine package was resolved, for at most `timeout`; whether one was.
    pub(crate) fn resolved(&self, timeout: std::time::Duration) -> bool {
        self.resolved.recv_timeout(timeout).is_ok()
    }

    /// Lets one held `prepare` answer.
    pub(crate) fn release(&self) {
        self.release.send(()).unwrap();
    }
}

/// Answers the one request a client of the stand-in Build Server sends on `connection`, with a
/// token on `asked` for a `prepare` and on `resolved` once an engine package is resolved.
fn answer(
    mut connection: Connection,
    held: bool,
    raise: u32,
    (asked, resolved): (&Sender<()>, &Sender<()>),
    released: &Mutex<Receiver<()>>,
) {
    use build_server::protocol::wire;

    // A connection that closes before its request asked nothing.
    let Ok(request) = connection.recv_next::<wire::Request>() else {
        return;
    };
    match request.request {
        Some(wire::request::Request::Prepare(prepare)) => {
            let _ = asked.send(());
            if held && released.lock().unwrap().recv().is_err() {
                return;
            }
            let packages = prepare
                .packages
                .into_iter()
                .map(|requested| wire::SlotPackage {
                    op: requested.op,
                    key: requested.key,
                    part: requested.part,
                    package: match requested.selector {
                        Some(wire::package_request::Selector::Package(mut revision)) => {
                            revision.revision += raise;
                            Some(revision)
                        }
                        _ => None,
                    },
                })
                .collect();
            let _ = connection.deliver(&wire::ServerMessage {
                message: Some(wire::server_message::Message::Prepared(
                    wire::PrepareResponse { packages },
                )),
            });
        }
        Some(wire::request::Request::Resolve(_)) => {
            // Every engine package is the stand-in engine, which the test crate built.
            let path = env!("CARGO_BIN_EXE_stand-in-engine");
            let file = std::fs::File::open(path).unwrap();
            let size = file.metadata().unwrap().len();
            let name = "stand-in-engine".to_owned();
            let response = wire::ServerMessage {
                message: Some(wire::server_message::Message::Resolved(
                    wire::ResolveResponse {
                        form: wire::ModuleForm::Executable as i32,
                        files: vec![wire::ModuleFile {
                            name: name.clone(),
                            size,
                            path: path.to_owned(),
                        }],
                        entries: std::collections::HashMap::new(),
                        metadata: String::new(),
                        interface_version: proto::ENGINE_INTERFACE.newest(),
                        interface_oldest: proto::ENGINE_INTERFACE.oldest(),
                        executable: name,
                        // The stand-in is no revision of a repository.
                        commit: None,
                        clean: false,
                    },
                )),
            };
            use std::os::fd::AsFd as _;
            connection.send_with(&response, &[file.as_fd()]).unwrap();
            connection.flush().unwrap();
            let _ = resolved.send(());
        }
        _ => {}
    }
}

/// The module set of an epoch: the engine package, a scheduling policy and one kernel package,
/// each at a fixed revision.
pub(crate) fn module_set() -> proto::module::ModuleSet {
    let package = |backend: &str, revision| proto::module::PackageRevision {
        backend: backend.to_owned(),
        revision,
        arguments: toml::Table::new(),
    };
    proto::module::ModuleSet {
        engine: package("uniserve", 3),
        policy: package("fcfs", 2),
        kernels: [(
            proto::module::ModuleSlot::all("rms_norm"),
            package("triton", 4),
        )]
        .into_iter()
        .collect(),
        programs: Default::default(),
        processing: None,
    }
}

/// A stand-in for the deployment's Storage Server on a node, which a node's daemon claims the deployment's storage
/// at: it owns its socket and states its readiness as a Storage Server serving the default
/// settings does, and answers the requests a claim and a publication make of it (`Claim`,
/// `Restore`, `Publish`) as a server that holds no storage does. As the Storage Server, it
/// publishes only epochs a claim at it named. Each connection is served on a thread of its own.
pub(crate) struct StorageServerStandIn {
    state: Arc<Mutex<StorageState>>,
}

/// What a [`StorageServerStandIn`] was asked.
#[derive(Default)]
struct StorageState {
    /// The epochs claimed at the server and not retired.
    claimed: std::collections::BTreeSet<u64>,
    /// Each publication it took, in order.
    published: Vec<Vec<u64>>,
}

impl StorageServerStandIn {
    /// Serves at `socket`, the socket of the deployment's Storage Server in the runtime directory
    /// a daemon claims storage in (`storage::socket`).
    pub(crate) fn serve(socket: PathBuf) -> Self {
        let (listener, starting) = service::host::bind::<storage::Settings>(&socket)
            .unwrap()
            .owned(&socket)
            .expect("no server owns the stand-in's socket");
        let lock = starting
            .finish(&service::Readiness::of_process(
                service::State::Serving,
                None,
                service::Endpoints {
                    storage: Some(service::Endpoint::unix(&socket)),
                    ..service::Endpoints::default()
                },
                storage::Settings::default(),
            ))
            .unwrap();
        let state = Arc::new(Mutex::new(StorageState::default()));
        let serving = Arc::clone(&state);
        std::thread::spawn(move || {
            let _lock = lock;
            for stream in listener.incoming() {
                let Ok(stream) = stream else {
                    return;
                };
                let state = Arc::clone(&serving);
                std::thread::spawn(move || serve_storage(Connection::new(stream).unwrap(), &state));
            }
        });
        Self { state }
    }

    /// The publications the server took, in order: the epochs each published.
    pub(crate) fn published(&self) -> Vec<Vec<u64>> {
        self.state.lock().unwrap().published.clone()
    }
}

/// Serves one connection of a [`StorageServerStandIn`] until it closes.
fn serve_storage(mut connection: Connection, state: &Mutex<StorageState>) {
    use storage::protocol::{
        ClientMessage, ErrorKind, Request as StorageRequest, Response, ServerMessage,
        Welcome as StorageWelcome,
    };

    while let Ok(message) = connection.recv_next::<ClientMessage>() {
        let reply = match message {
            ClientMessage::Hello(_) => ServerMessage::Welcome(StorageWelcome::Accepted),
            ClientMessage::Request { id, request } => {
                let mut state = state.lock().unwrap();
                let response = match request {
                    StorageRequest::Claim { epoch, retire, .. } => {
                        state.claimed.insert(epoch);
                        state.claimed.retain(|claimed| !retire.contains(claimed));
                        Response::Claimed {
                            claim: storage::Claim([7; 16]),
                        }
                    }
                    StorageRequest::Restore { .. } => Response::Moved {
                        storages: 0,
                        bytes: 0,
                    },
                    StorageRequest::Publish { epochs }
                        if epochs.iter().all(|epoch| state.claimed.contains(epoch)) =>
                    {
                        state.published.push(epochs);
                        Response::Published
                    }
                    StorageRequest::Publish { .. } => Response::Error {
                        kind: ErrorKind::Invalid,
                        message: "only claimed epochs can be published".to_owned(),
                    },
                    other => Response::Error {
                        kind: ErrorKind::Invalid,
                        message: format!("the stand-in serves no {other:?}"),
                    },
                };
                ServerMessage::Response { id, response }
            }
        };
        if connection.deliver(&reply).is_err() {
            return;
        }
    }
}
