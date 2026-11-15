//! An engine's side of the process interface, which the tests drive a
//! daemon with, and the stand-in processes they have it start.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use daemon::DaemonOptions;
use proto::process::{
    Downstream, Engine as EnginePeer, Exit, Hello, Reply, Request, Reserve, Reserved, Spawn, Stop,
    Upstream, Welcome, downstream, event, hello, reply, request, upstream,
};
use service::Connection;

/// The bound for a daemon or a process it supervises to react.
pub(crate) const TIMEOUT: Duration = Duration::from_secs(30);

/// The options of the single node `host`, whose process socket is `socket`.
pub(crate) fn single_node(host: &str, socket: PathBuf) -> DaemonOptions {
    DaemonOptions {
        nnodes: 1,
        node_rank: 0,
        host: host.to_owned(),
        dist_init_addr: None,
        socket,
        // No test claims storage, which is the only use of the executable.
        executable: PathBuf::from("uniserve"),
        join_timeout: TIMEOUT,
        devices: || Ok(Vec::new()),
        build_server: build_server::command::Settings::default(),
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
        let (engine, welcome) = Self::hello(address, proto::ENGINE_INTERFACE_VERSION);
        assert_eq!(welcome.refusal, "", "the leader refused this engine");
        engine
    }

    /// Connects to the leader at `address`, saying hello as an engine whose
    /// build speaks `major`, and returns the leader's welcome.
    pub(crate) fn hello(address: &str, major: u32) -> (Self, Welcome) {
        let mut engine = Self::dial(address, major);
        let welcome = engine.welcome();
        (engine, welcome)
    }

    /// Connects to the leader at `address` as an engine of this build that
    /// sends its first request, `kind`, right behind its hello rather than
    /// after the welcome, and returns the request's answer.
    pub(crate) fn connect_asking(address: &str, kind: request::Kind) -> (Self, Reply) {
        let mut engine = Self::dial(address, proto::ENGINE_INTERFACE_VERSION);
        let id = engine.send(kind);
        assert_eq!(
            engine.welcome().refusal,
            "",
            "the leader refused this engine"
        );
        let reply = engine.reply(id);
        (engine, reply)
    }

    /// Connects to the leader at `address` and says hello as an engine whose
    /// build speaks `major`.
    fn dial(address: &str, major: u32) -> Self {
        let deadline = Instant::now() + TIMEOUT;
        let mut connection = loop {
            match Connection::connect(&address.parse().unwrap()) {
                Ok(connection) => break connection,
                Err(error) => {
                    assert!(
                        Instant::now() < deadline,
                        "the leader is not reachable: {error}"
                    );
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
        };
        connection
            .deliver(&Hello {
                major,
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
            .recv::<Welcome>(TIMEOUT)
            .unwrap()
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

    /// Reserves a socket on `host` for `group`.
    pub(crate) fn reserve(&mut self, host: &str, group: &str, routable: bool) -> Reserved {
        let reserve = Reserve {
            host: host.into(),
            group: group.into(),
            routable,
            // Reserved by the connection that also spawns it.
            engine: 0,
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
            .recv(TIMEOUT)
            .unwrap()
            .expect("the daemon sent nothing in time");
        message.kind.expect("the daemon sent an empty message")
    }
}

/// The launch of `executable` with `arguments` as `rank` of `group`'s
/// launch `generation` on `host`.
pub(crate) fn launch(
    host: &str,
    group: &str,
    rank: u32,
    generation: u64,
    executable: &str,
    arguments: &[&str],
) -> Spawn {
    Spawn {
        host: host.into(),
        group: group.into(),
        rank,
        generation,
        executable: executable.into(),
        arguments: arguments
            .iter()
            .map(|&argument| argument.to_owned())
            .collect(),
        ..Spawn::default()
    }
}

/// The launch of a process that sleeps until it is killed.
pub(crate) fn sleeper(host: &str, group: &str, rank: u32, generation: u64) -> Spawn {
    launch(host, group, rank, generation, "/bin/sleep", &["600"])
}

/// Whether process `pid` exists and is not a zombie awaiting its reaper.
pub(crate) fn alive(pid: u32) -> bool {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    // The state letter follows the parenthesized command name.
    let state = stat.rsplit(')').next().unwrap_or("").trim_start();
    !state.starts_with('Z')
}

/// Polls `condition` for up to [`TIMEOUT`] and returns its last value.
pub(crate) fn eventually(mut condition: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + TIMEOUT;
    while Instant::now() < deadline {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    condition()
}

/// A stand-in for a host's Build Server, the host service a daemon prepares an epoch's module set
/// at: it serves `prepare` at its socket and answers every slot with the package revision the
/// request fixed for it, raised by `raise`, as a Build Server that holds every module answers.
pub(crate) struct BuildServerStandIn {
    socket: PathBuf,
    /// One token per `prepare` that arrived.
    asked: Receiver<()>,
    /// One token lets one held `prepare` answer.
    release: Sender<()>,
}

impl BuildServerStandIn {
    /// Serves at `socket`, owning it as a Build Server does; with `held`, each `prepare` answers
    /// only once [`Self::release`] lets it, as one building the engine's executable does.
    pub(crate) fn serve(socket: PathBuf, held: bool, raise: u32) -> Self {
        use build_server::protocol::wire;

        let (listener, starting) = service::host::bind::<()>(&socket)
            .unwrap()
            .owned(&socket)
            .expect("no server owns the stand-in's socket");
        let lock = starting.finish(&build_server::readiness(&socket)).unwrap();
        let (asked_sender, asked) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel::<()>();
        std::thread::spawn(move || {
            let _lock = lock;
            for stream in listener.incoming() {
                let Ok(stream) = stream else {
                    return;
                };
                let mut connection = Connection::new(stream).unwrap();
                // A daemon probes whether a server listens with a connection that sends nothing.
                let Ok(Some(request)) = connection.recv::<wire::Request>(TIMEOUT) else {
                    continue;
                };
                let Some(wire::request::Request::Prepare(prepare)) = request.request else {
                    continue;
                };
                let _ = asked_sender.send(());
                if held && released.recv().is_err() {
                    return;
                }
                let packages = prepare
                    .packages
                    .into_iter()
                    .map(|requested| wire::SlotPackage {
                        op: requested.op,
                        phase: requested.phase,
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
        });
        Self {
            socket,
            asked,
            release,
        }
    }

    /// The settings of a daemon whose host this Build Server serves.
    pub(crate) fn settings(&self) -> build_server::command::Settings {
        build_server::command::Settings {
            socket: Some(self.socket.clone()),
            ..build_server::command::Settings::default()
        }
    }

    /// Its socket.
    pub(crate) fn socket(&self) -> PathBuf {
        self.socket.clone()
    }

    /// Waits until a `prepare` arrived.
    pub(crate) fn asked(&self) {
        self.asked
            .recv_timeout(TIMEOUT)
            .expect("the Build Server was asked to prepare");
    }

    /// Lets one held `prepare` answer.
    pub(crate) fn release(&self) {
        self.release.send(()).unwrap();
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
    }
}
