//! A leader's waits for engines' dials on its request socket ([`EngineControl`]), each of which
//! ends with what it serves: an epoch update's join with the update, a start with the process it
//! started, every wait with the request socket's close once the daemon serves its deployment no
//! more, and a wait for one engine whatever other engines dial meanwhile. The daemon runs these
//! waits on its runtime's blocking pool, so one left behind keeps that thread, and with it the
//! daemon's process, alive after its deployment ended.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use daemon::Daemon;
use daemon::engine::control::EngineControl;
use proto::request::wire;
use service::{Cancel, Connection};

use crate::support::{
    BuildServerStandIn, DEPLOYMENT, Engine, engine_args, launch, package, single_node,
};

/// How long a wait whose end has come may take to return. Its end is immediate; the bound only
/// keeps a regression from hanging the suite.
const RETURNS: Duration = Duration::from_secs(30);

/// A single node's daemon on this host, with the leader's engine control on its request socket.
struct Leader {
    control: Arc<EngineControl>,
    requests: PathBuf,
    daemon: Daemon,
    build_server: BuildServerStandIn,
    _directory: tempfile::TempDir,
}

impl Leader {
    fn start() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let build_server = BuildServerStandIn::serve(directory.path(), false, 0);
        // The host a default launch places its engine on.
        let host = daemon::config::Config::default().leader_host();
        let daemon = Daemon::start(
            single_node(&host, directory.path().join("process.sock"), &build_server),
            None,
        )
        .unwrap();
        let requests = directory.path().join("requests.sock");
        let control = EngineControl::new(
            &requests,
            daemon.address().unwrap().to_owned(),
            host,
            DEPLOYMENT.into(),
            build_server.socket(),
        )
        .unwrap();
        Self {
            control: Arc::new(control),
            requests,
            daemon,
            build_server,
            _directory: directory,
        }
    }

    fn host(&self) -> String {
        self.control.host().to_owned()
    }

    /// Dials the request socket as engine `engine` of this build, as an engine does first, and
    /// returns its link with the daemon's welcome, which a wait for that engine sends once it
    /// accepted the link.
    fn dial(&self, engine: u64) -> (Connection, wire::Welcome) {
        let mut link = Connection::connect(&service::Endpoint::unix(&self.requests)).unwrap();
        link.deliver(&wire::Hello {
            oldest: proto::ENGINE_INTERFACE.oldest(),
            newest: proto::ENGINE_INTERFACE.newest(),
            build: proto::BUILD.to_owned(),
            engine,
            interface: proto::process::Interface::Requests as i32,
        })
        .unwrap();
        let welcome = link
            .recv::<wire::Welcome>(RETURNS)
            .unwrap()
            .unwrap_or_else(|| panic!("no wait accepted engine {engine}'s link"));
        (link, welcome)
    }
}

/// Runs `wait` on a thread of its own, as the daemon runs its waits on its blocking pool, and
/// returns where its outcome arrives.
fn spawn<T: Send + 'static>(wait: impl FnOnce() -> T + Send + 'static) -> mpsc::Receiver<T> {
    let (outcome, arrived) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = outcome.send(wait());
    });
    arrived
}

/// An epoch update's target that exits before it dials (here at its start, as an engine whose
/// launch does not decode does) is reported to the update, which abandons itself; the update's
/// follower then cancels its join, which fails, and the daemon's runtime, which the join ran on,
/// drops: nothing of the update holds the daemon's process once its deployment ends.
#[test]
fn an_update_target_that_exits_before_it_dials_ends_its_join_with_the_update() {
    let leader = Leader::start();
    let mut updater = Engine::connect(leader.daemon.address().unwrap());
    updater.spawn(launch(&leader.host(), "engine", 0, 5, package::FAILS));
    let exit = updater.exit();
    assert_eq!(
        (exit.generation, exit.status.as_str()),
        (5, "exit status: 1")
    );

    // The daemon's runtime, as `uniserve serve` builds it.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let control = Arc::clone(&leader.control);
    let host = leader.host();
    let joined = runtime.block_on(async move {
        let (abandon, abandoned) = Cancel::new().unwrap();
        let joining = tokio::task::spawn_blocking(move || {
            control
                .join_target(&[(host, 5)], 5, Box::new(|_| {}), &abandoned)
                .map(drop)
        });
        // The update ended: its follower lets go of the join.
        abandon.cancel();
        tokio::time::timeout(RETURNS, joining)
            .await
            .expect("the join outlived its update")
            .unwrap()
    });
    let error = joined.expect_err("a target that never dialled joined");
    assert!(format!("{error:#}").contains("abandoned"), "{error:#}");

    let dropped = spawn(move || drop(runtime));
    dropped
        .recv_timeout(RETURNS)
        .expect("a wait of the update held the daemon's runtime");
}

/// The request socket's close, the end of the daemon's deployment, ends every wait for a dial in
/// progress, whatever else would end it: an update's join its update has not abandoned, and a
/// borrowed front's adoption that nothing else ends. A later wait fails at once.
#[test]
fn closing_the_request_socket_ends_every_wait_for_a_dial() {
    let leader = Leader::start();
    // An update still in flight: its follower holds the join.
    let (_update, abandoned) = Cancel::new().unwrap();
    let (control, host) = (Arc::clone(&leader.control), leader.host());
    let join = {
        let abandoned = abandoned.clone();
        spawn(move || {
            control
                .join_target(&[(host, 7)], 7, Box::new(|_| {}), &abandoned)
                .map(drop)
        })
    };
    let control = Arc::clone(&leader.control);
    let front = spawn(move || {
        control
            .adopt(8, 9, 1, None, Box::new(|_| {}), None)
            .map(drop)
    });

    leader.control.close();
    for (wait, ended) in [("the join", join), ("the borrowed front", front)] {
        let error = ended
            .recv_timeout(RETURNS)
            .unwrap_or_else(|_| panic!("{wait} outlived the request socket's close"))
            .expect_err("an engine that never dialled was accepted");
        assert!(format!("{error:#}").contains("closed"), "{error:#}");
    }
    let refused =
        leader
            .control
            .join_target(&[(leader.host(), 10)], 10, Box::new(|_| {}), &abandoned);
    assert!(refused.is_err(), "a closed request socket awaited a dial");
}

/// A wait for one engine is not held up by a wait for another: each takes its own engine's
/// link, whichever wait took the dial off the socket and whichever engine dials first.
#[test]
fn a_wait_takes_its_engine_whatever_other_engines_dial_first() {
    let leader = Leader::start();
    let (_update, abandoned) = Cancel::new().unwrap();
    let joins: Vec<_> = [7, 8]
        .into_iter()
        .map(|engine| {
            let (control, host) = (Arc::clone(&leader.control), leader.host());
            let abandoned = abandoned.clone();
            spawn(move || {
                control
                    .join_target(&[(host, engine)], engine, Box::new(|_| {}), &abandoned)
                    .map(drop)
            })
        })
        .collect();

    // Engine 8 dials while engine 7's wait is still waiting, then engine 7.
    for engine in [8, 7] {
        let (link, welcome) = leader.dial(engine);
        assert_eq!(welcome.refusal, None, "engine {engine} was refused");
        // The link closes before the engine states its start, which ends that join.
        drop(link);
    }
    for joined in joins {
        let joined = joined
            .recv_timeout(RETURNS)
            .expect("a join outlived its engine's link");
        assert!(joined.is_err(), "an engine that stated no start joined");
    }
}

/// A start ends when the daemon's service ends the process it awaits before the process dialled:
/// the control connection the process lived under closes with it, as a daemon stopping during an
/// engine's start closes it.
#[test]
fn a_start_ends_when_the_daemon_ends_the_process_it_awaits() {
    let mut leader = Leader::start();
    let control = Arc::clone(&leader.control);
    // The stand-in runs until its daemon ends it, and never dials; a launch names its model.
    let mut args = engine_args(package::FOLLOWS_LIFELINE);
    args.engine_config.model.model = "model".to_owned();
    let started = spawn(move || control.start(args, Box::new(|_| {})).map(drop));

    // The node resolves the engine package as it serves the spawn, which it then answers before
    // it serves the stop.
    assert!(
        leader.build_server.resolved(RETURNS),
        "the node resolved no engine package: the start ended {:?}",
        started.try_recv()
    );
    leader.daemon.stop().unwrap();
    let error = started
        .recv_timeout(RETURNS)
        .expect("the start outlived the process it awaited")
        .expect_err("a process that never dialled started");
    assert!(format!("{error:#}").contains("closed"), "{error:#}");
}
