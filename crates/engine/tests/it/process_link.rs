//! The daemon's process interface as the main process sees it: a command for an
//! epoch arrives without a request being in flight, a report reaches the
//! daemon, one process has one reader, and closing the link ends the connection.
//!
//! The daemon is a socket of this test's own, speaking the process interface's
//! frames: it welcomes the engine, sends it its commands, and reads back what the
//! engine reports.
//!
//! One test carries the whole exchange because a process's commands are read
//! once: [`engine::ProcessLink::connect`] takes them for the process, so a second
//! successful link cannot exist in this test binary.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use engine::ProcessLink;
use proto::process::{
    self, downstream, event, upstream, Command, Downstream, Event, Hello, Upstream, Welcome,
};
use proto::Listener;

/// How long a step of the exchange may take.
const PATIENCE: Duration = Duration::from_secs(10);

/// A socket path of this test's own, removed when the test ends.
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

/// The daemon's side of the interface, for `accepts` engines in turn: each is
/// welcomed, sent `commands`, and answered with the reports it sent before it
/// closed.
fn daemon(
    listener: Listener,
    commands: Vec<Command>,
    accepts: usize,
) -> std::thread::JoinHandle<Vec<Vec<process::Report>>> {
    std::thread::spawn(move || {
        let mut rounds = Vec::new();
        while rounds.len() < accepts {
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
            for command in &commands {
                connection
                    .send(&Downstream {
                        kind: Some(downstream::Kind::Event(Event {
                            kind: Some(event::Kind::Command(command.clone())),
                        })),
                    })
                    .expect("the command is sent");
            }

            let mut reports = Vec::new();
            loop {
                match connection.recv::<Upstream>(PATIENCE) {
                    Ok(Some(Upstream {
                        kind: Some(upstream::Kind::Report(report)),
                    })) => reports.push(report),
                    Ok(Some(_)) => {}
                    // The engine closed its side: the daemon sees the
                    // connection end.
                    Ok(None) | Err(proto::Error::Closed) => break,
                    Err(error) => panic!("the test's daemon reads no more: {error}"),
                }
            }
            rounds.push(reports);
        }
        rounds
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

/// A commit, the report of which carries no number.
fn commit() -> Command {
    Command {
        kind: Some(process::command::Kind::Commit(process::Commit {})),
    }
}

#[test]
fn the_link_receives_the_daemons_commands_unasked_and_reports_back() {
    let socket = Socket::new("exchange");
    let listener = Listener::bind(&socket.address()).expect("the test's daemon listens");
    // Two connections: the engine's, and one a second link would open.
    let daemon = daemon(listener, vec![switchover(3), commit()], 2);

    let link = Arc::new(
        ProcessLink::connect(&socket.address(), "test-deployment")
            .expect("the engine connects to its daemon"),
    );
    // No request is in flight, and the commands still arrive, in the daemon's
    // order.
    let commands = link.commands();
    match commands
        .recv_timeout(PATIENCE)
        .expect("the daemon's first command reaches the engine")
        .kind
    {
        Some(process::command::Kind::Switchover(command)) => {
            assert_eq!(command.generation, 3, "the number the daemon names")
        }
        other => panic!("the command is the one the daemon sent, not {other:?}"),
    }
    assert!(matches!(
        commands
            .recv_timeout(PATIENCE)
            .expect("the daemon's second command reaches the engine")
            .kind,
        Some(process::command::Kind::Commit(_))
    ));

    // A report reaches the daemon, from this thread and from another one.
    link.report(process::Report {
        kind: Some(process::report::Kind::Switched(process::Switched {
            generation: 3,
        })),
    })
    .expect("the report is sent");
    let reporting = Arc::clone(&link);
    let reporter = std::thread::spawn(move || {
        reporting
            .report(process::Report {
                kind: Some(process::report::Kind::Committed(process::Committed {})),
            })
            .expect("the report is sent from another thread");
    });
    reporter.join().expect("the reporter ends");

    // One process has one reader: a second link would take half the commands.
    let refused = ProcessLink::connect(&socket.address(), "test-deployment")
        .err()
        .expect("a second link in one process is refused");
    let refused = format!("{refused:#}");
    assert!(refused.contains("one link"), "{refused}");

    // Closing the link ends the connection, which is what the daemon reads as
    // this main process leaving.
    let link = Arc::into_inner(link).expect("the reporter's share is gone");
    link.close();
    let rounds = daemon.join().expect("the test's daemon ends");
    let reports = &rounds[0];
    assert_eq!(reports.len(), 2, "both reports arrived");
    assert!(
        matches!(
            reports[0].kind,
            Some(process::report::Kind::Switched(process::Switched { generation: 3 }))
        ),
        "the switchover's report arrived first"
    );
    assert!(
        matches!(
            reports[1].kind,
            Some(process::report::Kind::Committed(process::Committed {}))
        ),
        "the other thread's report arrived second"
    );
    assert!(
        rounds[1].is_empty(),
        "the refused link reported nothing: it never read the interface"
    );
}

#[test]
fn a_link_to_no_daemon_is_refused_by_address() {
    let socket = Socket::new("absent");
    let refused = ProcessLink::connect(&socket.address(), "test-deployment")
        .err()
        .expect("nothing listens there");
    let refused = format!("{refused:#}");
    assert!(refused.contains("connecting to the daemon"), "{refused}");
}
