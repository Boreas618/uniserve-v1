//! An engine's side of the process interface, which the tests drive a
//! daemon with, and the stand-in processes they have it start.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use daemon::DaemonOptions;
use proto::Connection;
use proto::process::{
    Downstream, Engine as EnginePeer, Exit, Hello, Reply, Request, Reserve, Reserved, Spawn, Stop,
    Upstream, Welcome, downstream, event, hello, reply, request, upstream,
};

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
        let (engine, welcome) = Self::hello(address, proto::PROTOCOL_MAJOR);
        assert_eq!(welcome.refusal, "", "the leader refused this engine");
        engine
    }

    /// Connects to the leader at `address`, saying hello as an engine whose
    /// build speaks `major`, and returns the leader's welcome.
    pub(crate) fn hello(address: &str, major: u32) -> (Self, Welcome) {
        let deadline = Instant::now() + TIMEOUT;
        let mut connection = loop {
            match Connection::connect(address) {
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
            .send(&Hello {
                major,
                minor: proto::PROTOCOL_MINOR,
                build: proto::BUILD.to_owned(),
                peer: Some(hello::Peer::Engine(EnginePeer {
                    deployment: "deployment".into(),
                })),
            })
            .unwrap();
        let welcome = connection
            .recv::<Welcome>(TIMEOUT)
            .unwrap()
            .expect("the leader answers a hello");
        let engine = Self {
            connection,
            next: 1,
            exits: VecDeque::new(),
        };
        (engine, welcome)
    }

    /// Sends one request and returns its answer.
    pub(crate) fn request(&mut self, kind: request::Kind) -> Reply {
        let id = self.next;
        self.next += 1;
        self.connection
            .send(&Upstream {
                kind: Some(upstream::Kind::Request(Request {
                    id,
                    kind: Some(kind),
                })),
            })
            .unwrap();
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
