//! The daemon's process interface as the main process sees it: a command for an
//! epoch arrives without a request being in flight, a report reaches the
//! daemon, and closing the link ends the connection.
//!
//! The daemon is a socket of this test's own, speaking the process interface's
//! frames: it welcomes the engine, sends it one command, and reads back what the
//! engine reports.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use engine::ProcessLink;
use proto::process::{
    self, Command, Downstream, Event, Hello, Upstream, Welcome, downstream, event, upstream,
};
use proto::Listener;

/// How long a step of the exchange may take.
const PATIENCE: Duration = Duration::from_secs(10);

/// A socket path of this test's own, in a directory it removes afterwards.
struct Socket {
    path: PathBuf,
}

impl Socket {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "sched-process-link-{}-{name}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        Self { path }
    }

    /// The address the interface listens at.
    fn address(&self) -> String {
        format!("unix:{}", self.path.display())
    }
}

impl Drop for Socket {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The daemon's side of the interface: welcomes the engine, sends `command`, and
/// answers with every report it received, ending when the connection closes.
fn daemon(listener: Listener, command: Command) -> std::thread::JoinHandle<Vec<process::Report>> {
    std::thread::spawn(move || {
        let mut connection = loop {
            match listener.try_accept() {
                Ok(Some(connection)) => break connection,
                Ok(None) => std::thread::sleep(Duration::from_millis(5)),
                Err(error) => panic!("the test's daemon accepts no engine: {error}"),
            }
        };
        let hello = connection
            .recv::<Hello>(PATIENCE)
            .expect("the engine's hello arrives")
            .expect("the engine sends a hello");
        assert!(
            matches!(
                hello.peer,
                Some(process::hello::Peer::Engine(process::Engine { .. }))
            ),
            "an engine connects, not another peer"
        );
        connection
            .send(&Welcome {
                major: proto::PROTOCOL_MAJOR,
                minor: proto::PROTOCOL_MINOR,
                build: proto::BUILD.to_owned(),
                refusal: String::new(),
                nodes: Vec::new(),
            })
            .expect("the welcome is sent");
        // The command an epoch's main process receives, unasked.
        connection
            .send(&Downstream {
                kind: Some(downstream::Kind::Event(Event {
                    kind: Some(event::Kind::Command(command)),
                })),
            })
            .expect("the command is sent");

        let mut reports = Vec::new();
        loop {
            match connection.recv::<Upstream>(PATIENCE) {
                Ok(Some(Upstream {
                    kind: Some(upstream::Kind::Report(report)),
                })) => reports.push(report),
                Ok(Some(_)) => {}
                // The engine closed its side: the daemon sees the connection end.
                Ok(None) | Err(proto::Error::Closed) => return reports,
                Err(error) => panic!("the test's daemon reads no more: {error}"),
            }
        }
    })
}

/// A switchover the daemon commands an epoch's main process to perform.
fn switchover(generation: u64) -> Command {
    Command {
        kind: Some(process::command::Kind::Switchover(process::Switchover {
            generation,
        })),
    }
}

#[test]
fn a_command_arrives_unasked_and_a_report_returns_to_the_daemon() {
    let socket = Socket::new("exchange");
    let listener = Listener::bind(&socket.address()).expect("the test's daemon listens");
    let daemon = daemon(listener, switchover(3));

    let link = ProcessLink::connect(&socket.address(), "test-deployment")
        .expect("the engine connects to its daemon");
    // No request is in flight, and the command still arrives.
    let command = link
        .commands()
        .recv_timeout(PATIENCE)
        .expect("the daemon's command reaches the engine");
    match command.kind {
        Some(process::command::Kind::Switchover(command)) => {
            assert_eq!(command.generation, 3, "the daemon's own number")
        }
        other => panic!("the command is the one the daemon sent, not {other:?}"),
    }

    link.report(process::Report {
        kind: Some(process::report::Kind::Switched(process::Switched {
            generation: 3,
        })),
    })
    .expect("the report is sent");

    // Closing the link ends the connection, which is what the daemon reads as
    // this main process leaving.
    link.close();
    let reports = daemon.join().expect("the test's daemon ends");
    assert_eq!(reports.len(), 1, "one report arrived");
    match &reports[0].kind {
        Some(process::report::Kind::Switched(switched)) => assert_eq!(switched.generation, 3),
        other => panic!("the daemon received the report the engine sent, not {other:?}"),
    }
}

#[test]
fn one_process_has_one_link_to_its_daemons() {
    let socket = Socket::new("one-link");
    let listener = Listener::bind(&socket.address()).expect("the test's daemon listens");
    let daemon = daemon(listener, switchover(1));

    let link = ProcessLink::connect(&socket.address(), "test-deployment")
        .expect("the first link connects");
    link.close();
    let _ = daemon.join();

    // The commands of a main process are read once, by the link it made: a
    // second reader would take half of them.
    let refused = ProcessLink::connect(&socket.address(), "test-deployment")
        .expect_err("a second link in one process is refused");
    let refused = format!("{refused:#}");
    assert!(refused.contains("one link"), "{refused}");
}

#[test]
fn a_link_to_no_daemon_is_refused_by_address() {
    let socket = Socket::new("absent");
    let refused = ProcessLink::connect(&socket.address(), "test-deployment")
        .expect_err("nothing listens there");
    let refused = format!("{refused:#}");
    assert!(refused.contains("connecting to the daemon"), "{refused}");
}

/// The link is `Send + Sync`: a commander thread reports while the epoch's own
/// thread reads commands.
#[test]
fn a_link_serves_both_directions_from_several_threads() {
    fn both(_: &(dyn Send + Sync)) {}
    let socket = Socket::new("threads");
    let listener = Listener::bind(&socket.address()).expect("the test's daemon listens");
    let daemon = daemon(listener, switchover(2));
    let link = Arc::new(
        ProcessLink::connect(&socket.address(), "test-deployment")
            .expect("the engine connects to its daemon"),
    );
    both(&*link);
    let reporting = Arc::clone(&link);
    let reporter = std::thread::spawn(move || {
        reporting
            .report(process::Report {
                kind: Some(process::report::Kind::Committed(process::Committed {})),
            })
            .expect("the report is sent from another thread");
    });
    let command = link
        .commands()
        .recv_timeout(PATIENCE)
        .expect("the command arrives while another thread reports");
    assert!(matches!(
        command.kind,
        Some(process::command::Kind::Switchover(_))
    ));
    reporter.join().expect("the reporter ends");
    let link = Arc::into_inner(link).expect("the reporter's share is gone");
    link.close();
    let reports = daemon.join().expect("the test's daemon ends");
    assert_eq!(reports.len(), 1, "the report from the other thread arrived");
}
