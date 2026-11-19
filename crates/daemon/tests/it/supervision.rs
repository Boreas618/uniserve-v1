//! One node's daemon, driven over its process socket as an engine drives it:
//! the processes it starts, their exits, the sockets it holds for them, and
//! what ends them.

use std::io::BufRead as _;
use std::net::TcpStream;

use daemon::Daemon;
use proto::process::{Exit, reply, request};

use crate::support::{
    BuildServerStandIn, Engine, engine_args, launch, package, single_node, sleeper,
};

/// A single-node daemon on `host` whose process socket lies in `directory`, with the stand-in
/// Build Server its node resolves its engines at.
struct Node {
    daemon: Daemon,
    _build_server: BuildServerStandIn,
}

impl Node {
    fn start(directory: &tempfile::TempDir, host: &str) -> Self {
        let build_server = BuildServerStandIn::serve(directory.path(), false, 0);
        let daemon = Daemon::start(
            single_node(host, directory.path().join("process.sock"), &build_server),
            None,
        )
        .unwrap();
        Self {
            daemon,
            _build_server: build_server,
        }
    }

    fn address(&self) -> &str {
        self.daemon.address().unwrap()
    }
}

#[test]
fn exits_arrive_unasked_and_name_their_launch() {
    let directory = tempfile::tempdir().unwrap();
    let node = Node::start(&directory, "node");
    let mut engine = Engine::connect(node.address());

    engine.spawn(launch("node", "first", 0, 3, package::FAILS));

    // No further request is sent: the report comes from the daemon's own
    // observation of its children. It names the generation the spawn carried,
    // which is how an engine tells it from an exit of the same rank before
    // its group was relaunched.
    assert_eq!(
        engine.exit(),
        Exit {
            host: "node".into(),
            group: "first".into(),
            rank: 0,
            generation: 3,
            status: "exit status: 1".into(),
        }
    );
}

/// A launch asked for right behind the hello, before the welcome arrived, is
/// the launch of the connection that asked: its exit comes back on it, as
/// the exit of a launch asked for later does.
#[test]
fn a_launch_sent_with_the_hello_reports_its_exit_to_its_connection() {
    let directory = tempfile::tempdir().unwrap();
    let node = Node::start(&directory, "node");
    let (mut engine, reply) = Engine::connect_asking(
        node.address(),
        request::Kind::Spawn(launch("node", "first", 0, 3, package::FAILS)),
    );
    assert!(
        matches!(reply.kind, Some(reply::Kind::Spawned(_))),
        "{reply:?}"
    );

    let exit = engine.exit();
    assert_eq!(
        (exit.group.as_str(), exit.generation, exit.status.as_str()),
        ("first", 3, "exit status: 1")
    );
}

#[test]
fn an_exit_goes_to_the_connection_that_asked_for_the_launch() {
    let directory = tempfile::tempdir().unwrap();
    let node = Node::start(&directory, "node");
    let mut asker = Engine::connect(node.address());
    let mut other = Engine::connect(node.address());

    // Both clients' launches are the deployment's, so the report cannot be
    // routed by owner: a client waiting for its own process must not read
    // another client's exit as that process's.
    asker.spawn(launch("node", "first", 0, 0, package::FAILS));
    assert_eq!(asker.exit().group, "first");

    // The exit was sent before this answer, so a report for `other` is
    // already filed behind it when the answer arrives.
    other.request(request::Kind::Nodes(Default::default()));
    assert_eq!(other.reported_exits(), 0);
}

#[test]
fn the_first_rank_serves_its_store_on_the_reserved_socket() {
    let directory = tempfile::tempdir().unwrap();
    let node = Node::start(&directory, "node");
    let mut engine = Engine::connect(node.address());

    // A world's two sockets, the peer channel and the store, as engine control reserves them.
    let channel = engine.reserve("node", "group", false);
    let store = engine.reserve("node", "group", false);
    let other = engine.reserve("node", "other", false);
    assert_ne!(store.address, other.address);

    // A reservation is its group's: another group's process cannot take it.
    let mut stolen = launch("node", "other", 0, 0, package::SUCCEEDS);
    stolen.inherit = vec![store.reservation];
    let refused = engine.request(request::Kind::Spawn(stolen));
    assert!(
        refused.error.contains("other"),
        "the refusal names the group: {refused:?}"
    );

    // The stand-in rank accepts one peer on the store it inherits at the number the reservation
    // reported, which its launch's world states as an engine's does, and answers it with the
    // engine number its launch states.
    let mut args = engine_args(package::SERVES_STORE);
    args.engine = 4;
    args.world = Some(proto::config::World {
        scheduler: 4,
        channel: channel.address.clone(),
        store: store.address.clone(),
        peers: 0,
        inherited: Some(proto::config::Inherited {
            channel: channel.descriptor,
            store: store.descriptor,
        }),
    });
    let mut rank = launch("node", "group", 0, 4, package::SERVES_STORE);
    rank.launch = serde_json::to_vec(&args).unwrap();
    rank.inherit = vec![channel.reservation, store.reservation];
    engine.spawn(rank);

    // The reserved socket is already listening, so this connection waits in
    // its backlog until the rank starts and accepts it.
    let peer = TcpStream::connect(&store.address).unwrap();
    let mut served = String::new();
    std::io::BufReader::new(peer)
        .read_line(&mut served)
        .unwrap();
    assert_eq!(served, "served 4\n");
    assert_eq!(engine.exit().status, "exit status: 0");

    // The daemon handed the socket over and holds it no longer: a second launch cannot inherit
    // it.
    let mut again = launch("node", "group", 0, 5, package::SUCCEEDS);
    again.inherit = vec![store.reservation];
    let refused = engine.request(request::Kind::Spawn(again));
    assert!(
        refused.error.contains("no reservation"),
        "a reservation is inherited once: {refused:?}"
    );
}

#[test]
fn a_reservation_serves_a_launch_another_connection_sends() {
    // A worker group's client reserves a socket and launches the rank that inherits it; the two
    // are separate connections of one deployment, both served for its head, so the reservation
    // made on the first serves the launch sent on the second.
    let directory = tempfile::tempdir().unwrap();
    let node = Node::start(&directory, "node");
    let mut first = Engine::connect(node.address());
    let reserved = first.reserve("node", "group", false);

    let mut second = Engine::connect(node.address());
    let mut rank = launch("node", "group", 0, 1, package::SUCCEEDS);
    rank.inherit = vec![reserved.reservation];
    second.spawn(rank);
    assert_eq!(second.exit().status, "exit status: 0");
}

#[test]
fn a_stop_ends_every_generation_of_one_group_only() {
    let directory = tempfile::tempdir().unwrap();
    let node = Node::start(&directory, "node");
    let mut engine = Engine::connect(node.address());

    // A group's standby ranks run beside its serving ones under a later
    // generation.
    let serving = engine.start(sleeper("node", "stopped", 0, 0));
    let standby = engine.start(sleeper("node", "stopped", 0, 1));
    let kept = engine.start(sleeper("node", "kept", 0, 0));

    // The stop is answered once the group's processes were killed and reaped.
    engine.stop("stopped");
    assert!(!serving.runs() && !standby.runs());
    assert!(kept.runs(), "a stop ended another group");
}

/// A stop that names one generation ends that launch of the group, reports
/// its exit as every launch's exit is reported, and leaves the standby
/// launched beside it running.
#[test]
fn a_stop_of_one_generation_leaves_the_groups_other_launches() {
    let directory = tempfile::tempdir().unwrap();
    let node = Node::start(&directory, "node");
    let mut engine = Engine::connect(node.address());
    let serving = engine.start(sleeper("node", "group", 0, 0));
    let standby = engine.start(sleeper("node", "group", 0, 1));

    let reply = engine.request(request::Kind::Stop(proto::process::Stop {
        group: "group".into(),
        generation: Some(0),
    }));
    assert_eq!(reply.error, "");
    assert!(!serving.runs());
    assert!(standby.runs(), "a stop of one generation ended another");

    // The update that stopped a previous epoch's engine keeps its weights on
    // this exit.
    let exit = engine.exit();
    assert_eq!(
        (exit.host.as_str(), exit.group.as_str(), exit.generation),
        ("node", "group", 0)
    );
    engine.request(request::Kind::Nodes(Default::default()));
    assert_eq!(
        engine.reported_exits(),
        0,
        "the standby's exit was reported"
    );
}

#[test]
fn an_engine_takes_its_processes_with_it() {
    let directory = tempfile::tempdir().unwrap();
    let node = Node::start(&directory, "node");

    // Two engines of one deployment, as during an epoch change, name the same
    // groups and ranks.
    let mut leaving = Engine::connect(node.address());
    let mut staying = Engine::connect(node.address());
    let left = leaving.start(sleeper("node", "group", 0, 0));
    let stayed = staying.start(sleeper("node", "group", 0, 0));

    drop(leaving);
    left.wait();
    assert!(
        stayed.runs(),
        "an engine's close ended another engine's process"
    );
    assert!(matches!(
        staying
            .request(request::Kind::Stop(proto::process::Stop {
                group: "group".into(),
                generation: None,
            }))
            .kind,
        Some(reply::Kind::Stopped(_))
    ));
    assert!(!stayed.runs());
}

#[test]
fn a_stopped_daemon_takes_its_processes_with_it() {
    let directory = tempfile::tempdir().unwrap();
    let mut node = Node::start(&directory, "node");
    let mut engine = Engine::connect(node.address());
    let process = engine.start(sleeper("node", "group", 0, 0));

    node.daemon.stop().unwrap();
    assert!(!process.runs(), "a process outlived its daemon's stop");
}

/// A daemon serves an engine in the newest version of the engine interface both windows hold,
/// which its welcome states, whether the engine's build is this one, a later one whose window
/// holds this build's newest or an earlier one of one version this build speaks; it refuses an
/// engine whose window shares no version with its own at its hello, with the typed refusal naming
/// both windows.
#[test]
fn a_daemon_serves_engines_in_the_newest_version_both_speak() {
    use proto::process::refusal::Reason;
    use proto::process::{Refusal, VersionRefusal};
    use service::Versions;

    let directory = tempfile::tempdir().unwrap();
    let node = Node::start(&directory, "node");

    let ours = proto::ENGINE_INTERFACE;
    for (offered, agreed) in [
        (ours, ours.newest()),
        (
            Versions::new(ours.newest(), ours.newest() + 1),
            ours.newest(),
        ),
        (Versions::one(ours.oldest()), ours.oldest()),
    ] {
        let (_, welcome) = Engine::hello(node.address(), offered);
        assert_eq!(
            welcome.refusal, None,
            "an engine of versions {offered} is refused"
        );
        assert_eq!(welcome.version, agreed, "an engine of versions {offered}");
    }
    for offered in [
        Versions::one(ours.oldest() - 1),
        Versions::new(ours.newest() + 1, ours.newest() + 2),
    ] {
        let (_, welcome) = Engine::hello(node.address(), offered);
        assert_eq!(
            welcome.refusal,
            Some(Refusal {
                reason: Some(Reason::Version(VersionRefusal {
                    offered: offered.oldest(),
                    spoken: ours.newest(),
                    offered_newest: offered.newest(),
                    spoken_oldest: ours.oldest(),
                })),
            }),
            "an engine of versions {offered}"
        );
    }
}

/// A daemon killed outright takes its processes with it: each follows its lifeline, whose last
/// write end the kernel closes with the daemon.
#[test]
fn a_killed_daemon_takes_its_processes_with_it() {
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("process.sock");
    let _build_server = BuildServerStandIn::serve(directory.path(), false, 0);
    let mut host = std::process::Command::new(env!("CARGO_BIN_EXE_daemon-host"))
        .arg("--socket")
        .arg(&socket)
        // The host's runtime directory holds its deployment's Build Server, the stand-in.
        .env("XDG_RUNTIME_DIR", directory.path())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    // The host reports that its daemon serves.
    let mut ready = String::new();
    std::io::BufReader::new(host.stdout.take().unwrap())
        .read_line(&mut ready)
        .unwrap();
    assert_eq!(ready, "serving\n", "the daemon did not start");
    let mut engine = Engine::connect(&format!("unix:{}", socket.display()));
    let process = engine.start(sleeper("node", "group", 0, 0));

    // SIGKILL skips every exit path of the daemon's own; the kernel closes
    // the lifeline's write end with the daemon.
    host.kill().unwrap();
    host.wait().unwrap();
    process.wait();
}

/// A GPU belongs to one deployment: on a node whose GPU an engine of one deployment runs on, a
/// second deployment's engine that names the GPU is refused before it starts, with an error that
/// names the first deployment, and starts once the first deployment's daemon ended.
#[test]
fn a_gpu_of_another_deployment_refuses_the_engine_naming_it() {
    let directory = tempfile::tempdir().unwrap();
    // Both deployments' daemons run on one node: they claim in its one registration file.
    let registration = directory.path().join("gpus");
    let start = |deployment: &str| {
        let build_server =
            BuildServerStandIn::serve_for(&directory.path().join(deployment), deployment, false, 0);
        let mut options = single_node(
            "node",
            directory.path().join(format!("{deployment}.sock")),
            &build_server,
        );
        options.deployment = deployment.to_owned();
        options.gpus = registration.clone();
        (Daemon::start(options, None).unwrap(), build_server)
    };
    let on_gpus = |generation: u64, gpus: Vec<u32>| {
        let mut spawn = sleeper("node", "group", 0, generation);
        let mut args: proto::config::EngineArgs = serde_json::from_slice(&spawn.launch).unwrap();
        args.devices = gpus;
        spawn.launch = serde_json::to_vec(&args).unwrap();
        spawn
    };

    let (first, _first_builds) = start("first");
    let mut engine = Engine::connect(first.address().unwrap());
    let running = engine.start(on_gpus(1, vec![0, 1]));
    assert!(running.runs());

    let (second, _second_builds) = start("second");
    let mut other = Engine::connect(second.address().unwrap());
    let refused = other.request(request::Kind::Spawn(on_gpus(1, vec![1])));
    assert!(refused.kind.is_none(), "{refused:?}");
    assert!(
        refused.error.contains("GPU 1") && refused.error.contains("deployment first"),
        "{}",
        refused.error
    );

    // The first deployment's claims end with its daemon.
    drop(engine);
    drop(first);
    let started = other.start(on_gpus(2, vec![1]));
    assert!(started.runs());
}
