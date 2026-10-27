//! One node's daemon, driven over its process socket as an engine drives it:
//! the processes it starts, their exits, the sockets it holds for them, and
//! what ends them.

use std::io::BufRead as _;
use std::net::TcpStream;
use std::time::Duration;

use daemon::Daemon;
use proto::process::{Exit, reply, request};

use crate::support::{Engine, TIMEOUT, alive, eventually, launch, single_node, sleeper};

/// Starts a single-node daemon on `host` whose process socket lies in
/// `directory`.
fn start(directory: &tempfile::TempDir, host: &str) -> Daemon {
    Daemon::start(single_node(host, directory.path().join("process.sock"))).unwrap()
}

#[test]
fn exits_arrive_unasked_and_name_their_launch() {
    let directory = tempfile::tempdir().unwrap();
    let daemon = start(&directory, "node");
    let mut engine = Engine::connect(daemon.address().unwrap());

    // /bin/false is a real process with a known exit status.
    engine.spawn(launch("node", "first", 0, 3, "/bin/false", &[]));

    // No further request is sent: the report comes from the daemon's own
    // polling of its children. It names the generation the spawn carried,
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

#[test]
fn the_first_rank_serves_its_store_on_the_reserved_socket() {
    let directory = tempfile::tempdir().unwrap();
    let daemon = start(&directory, "node");
    let mut engine = Engine::connect(daemon.address().unwrap());

    let reserved = engine.reserve("node", "group", false);
    let other = engine.reserve("node", "other", false);
    assert_ne!(reserved.address, other.address);

    // A reservation is its group's: another group's process cannot take it.
    let mut stolen = launch("node", "other", 0, 0, "/bin/true", &[]);
    stolen.inherit = vec![reserved.reservation];
    let refused = engine.request(request::Kind::Spawn(stolen));
    assert!(
        refused.error.contains("other"),
        "the refusal names the group: {refused:?}"
    );

    // The stand-in rank accepts one peer on the socket it inherits at the
    // number the reservation reported, which is how a collective store
    // serves on an inherited socket, and answers it with the launch
    // generation its command line names. An engine names the number in the
    // rank's launch descriptor; this stand-in reads it from its environment.
    let store = "import os, socket, sys\n\
         store = socket.socket(fileno=int(os.environ['STORE_FD']))\n\
         peer, _ = store.accept()\n\
         generation = sys.argv[sys.argv.index('--generation') + 1]\n\
         peer.sendall(f'served {generation}\\n'.encode())\n";
    let mut rank = launch(
        "node",
        "group",
        0,
        4,
        "python3",
        &["-c", store, "--generation", "4"],
    );
    rank.inherit = vec![reserved.reservation];
    rank.environment = [("STORE_FD".to_owned(), reserved.descriptor.to_string())].into();
    engine.spawn(rank);

    // The reserved socket is already listening, so this connection waits in
    // its backlog until the rank starts and accepts it.
    let peer = TcpStream::connect(&reserved.address).unwrap();
    peer.set_read_timeout(Some(TIMEOUT)).unwrap();
    let mut served = String::new();
    std::io::BufReader::new(peer)
        .read_line(&mut served)
        .unwrap();
    assert_eq!(served, "served 4\n");
    assert_eq!(engine.exit().status, "exit status: 0");

    // The daemon kept no copy of the socket it handed over, so the port
    // closes with the rank that served it.
    assert_eq!(
        TcpStream::connect(&reserved.address).unwrap_err().kind(),
        std::io::ErrorKind::ConnectionRefused
    );
}

#[test]
fn a_stop_ends_every_generation_of_one_group_only() {
    let directory = tempfile::tempdir().unwrap();
    let daemon = start(&directory, "node");
    let mut engine = Engine::connect(daemon.address().unwrap());

    // A group's standby ranks run beside its serving ones under a later
    // generation.
    let serving = engine.spawn(sleeper("node", "stopped", 0, 0));
    let standby = engine.spawn(sleeper("node", "stopped", 0, 1));
    let kept = engine.spawn(sleeper("node", "kept", 0, 0));

    engine.stop("stopped");
    assert!(eventually(|| !alive(serving) && !alive(standby)));
    assert!(alive(kept), "a stop ended another group");
}

/// A stop that names one generation ends that launch of the group, and
/// leaves the standby launched beside it running.
#[test]
fn a_stop_of_one_generation_leaves_the_groups_other_launches() {
    let directory = tempfile::tempdir().unwrap();
    let daemon = start(&directory, "node");
    let mut engine = Engine::connect(daemon.address().unwrap());
    let serving = engine.spawn(sleeper("node", "group", 0, 0));
    let standby = engine.spawn(sleeper("node", "group", 0, 1));

    let reply = engine.request(request::Kind::Stop(proto::process::Stop {
        group: "group".into(),
        generation: Some(0),
    }));
    assert_eq!(reply.error, "");
    assert!(eventually(|| !alive(serving)));
    assert!(alive(standby), "a stop of one generation ended another");
}

#[test]
fn an_engine_takes_its_processes_with_it() {
    let directory = tempfile::tempdir().unwrap();
    let daemon = start(&directory, "node");

    // Two engines of one deployment, as during a world change, name the same
    // groups and ranks.
    let mut leaving = Engine::connect(daemon.address().unwrap());
    let mut staying = Engine::connect(daemon.address().unwrap());
    let left = leaving.spawn(sleeper("node", "group", 0, 0));
    let stayed = staying.spawn(sleeper("node", "group", 0, 0));

    drop(leaving);
    assert!(eventually(|| !alive(left)), "a process outlived its engine");
    assert!(
        alive(stayed),
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
    assert!(eventually(|| !alive(stayed)));
}

#[test]
fn a_stopped_daemon_takes_its_processes_with_it() {
    let directory = tempfile::tempdir().unwrap();
    let mut daemon = start(&directory, "node");
    let mut engine = Engine::connect(daemon.address().unwrap());
    let pid = engine.spawn(sleeper("node", "group", 0, 0));

    daemon.stop().unwrap();
    assert!(!alive(pid), "a process outlived its daemon's stop");
}

#[test]
fn an_engine_of_another_major_version_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let daemon = start(&directory, "node");

    let (_, welcome) = Engine::hello(daemon.address().unwrap(), proto::PROTOCOL_MAJOR + 1);
    assert!(
        welcome.refusal.contains("major version"),
        "the refusal says why: {welcome:?}"
    );
}

/// The daemon of [`a_killed_daemon_takes_its_processes_with_it`], in a
/// process of its own: this test binary runs it with the socket's path in
/// [`DAEMON_SOCKET`].
#[test]
#[ignore = "runs as the daemon process of a supervision test"]
fn daemon_host() {
    let Some(socket) = std::env::var_os(DAEMON_SOCKET) else {
        return;
    };
    let _daemon = Daemon::start(single_node("node", socket.into())).unwrap();
    loop {
        std::thread::sleep(Duration::from_secs(60));
    }
}

const DAEMON_SOCKET: &str = "UNISERVE_TEST_DAEMON_SOCKET";

#[test]
fn a_killed_daemon_takes_its_processes_with_it() {
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("process.sock");
    let mut host = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "supervision::daemon_host", "--ignored"])
        .env(DAEMON_SOCKET, &socket)
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    assert!(eventually(|| socket.exists()), "the daemon did not start");
    let mut engine = Engine::connect(&format!("unix:{}", socket.display()));
    // A process that follows its lifeline, as the engine and the workers do.
    let pid = engine.spawn(launch(
        "node",
        "group",
        0,
        0,
        "/usr/bin/python3",
        &[
            "-c",
            "import os; os.read(int(os.environ['UNISERVE_LIFELINE']), 1)",
        ],
    ));
    assert!(alive(pid));

    // SIGKILL skips every exit path of the daemon's own; the kernel closes
    // the lifeline's write end with the daemon.
    host.kill().unwrap();
    host.wait().unwrap();
    assert!(eventually(|| !alive(pid)), "a process outlived its daemon");
}
