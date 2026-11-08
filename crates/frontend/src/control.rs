//! The frontend's link to the daemon that manages the deployment (design
//! `specs/worker.md` §2.3, §7.4).
//!
//! The daemon starts the frontend process and commands it down one
//! `process.proto` connection that the frontend opens once it runs:
//!
//! - `Serve { epoch, requests }`: epoch `epoch` serves from now on, and its
//!   main process takes requests at `requests` (`unix:<path>` or
//!   `host:port`). The frontend admits requests again and resubmits the facts
//!   its drain kept, which it alone holds.
//! - `Drain { epoch }`: no request is admitted, every request in flight ends
//!   migratable at its step boundary with its facts kept, and nothing is
//!   executed. The frontend waits for the `Serve` that names the next epoch.
//! - `Pause { paused }`: admissions are held or released without ending a
//!   request, the `/admin/pause_generation` gate.
//!
//! Commands go only downward (§2.3): the frontend reports nothing on this
//! connection. The main process reports its own `Drained`, and a fact a drain
//! kept is resubmitted by this process on the next `Serve`, so the daemon
//! waits for neither.
//!
//! The connection is a blocking one (`proto::Connection`), so it is read on a
//! thread of its own and its commands are published as a [`State`] watch: the
//! serving side keeps a receiver and acts on every change, without polling.

use std::time::Duration;

use anyhow::Context as _;
use proto::process::{self, command, downstream, event, hello};
use tokio::sync::watch;

/// How long the daemon takes to answer a frontend's hello.
const WELCOME_TIMEOUT: Duration = Duration::from_secs(90);

/// How long one read waits for the daemon's next command.
///
/// The read is blocking; this bounds how long the reader thread sleeps before
/// it looks at the connection again, and is not a deadline for the daemon's
/// next command — an idle frontend gets one `None` after another.
const COMMAND_TIMEOUT: Duration = Duration::from_millis(500);

/// The epoch that serves, and where its main process takes requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Serving {
    /// The epoch's number, as the daemon numbers it.
    pub epoch: u64,
    /// The main process's request socket: `unix:<path>` or `host:port`.
    pub requests: String,
}

/// What the daemon last commanded the frontend.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct State {
    /// The epoch that serves; `None` before the daemon switches one over.
    serving: Option<Serving>,
    /// Whether the daemon holds admissions.
    paused: bool,
    /// Whether the daemon drained the frontend and no `Serve` has followed.
    draining: bool,
}

impl State {
    /// The epoch that serves and the address of its main process.
    pub fn serving(&self) -> Option<&Serving> {
        self.serving.as_ref()
    }

    /// Whether the daemon holds admissions (`/admin/pause_generation`).
    pub fn paused(&self) -> bool {
        self.paused
    }

    /// Whether the daemon drained the frontend and no epoch was switched over
    /// since.
    pub fn draining(&self) -> bool {
        self.draining
    }

    /// Whether a request may be admitted now: an epoch serves, the daemon
    /// holds no admissions, and the frontend is not draining.
    pub fn admits(&self) -> bool {
        self.serving.is_some() && !self.paused && !self.draining
    }

    /// Applies one command the daemon sent.
    ///
    /// # Errors
    ///
    /// Fails for a command the frontend does not take, and for a `Serve` that
    /// names no request address: either is a defect of the daemon, and the
    /// frontend stops rather than serving a deployment it cannot reach.
    fn apply(&mut self, command: process::Command) -> anyhow::Result<()> {
        let Some(kind) = command.kind else {
            anyhow::bail!("the daemon sent a command with no kind");
        };
        match kind {
            command::Kind::Serve(serve) => {
                anyhow::ensure!(
                    !serve.requests.is_empty(),
                    "epoch {} serves with no request address",
                    serve.epoch
                );
                // A second `Serve` of the epoch that already serves is the
                // daemon's revert route: the frontend keeps serving it and
                // resubmits the facts it kept, which the epoch's main process
                // executes again.
                self.serving = Some(Serving {
                    epoch: serve.epoch,
                    requests: serve.requests,
                });
                self.draining = false;
            }
            command::Kind::Drain(drain) => match self.serving.as_ref() {
                Some(serving) if serving.epoch == drain.epoch => self.draining = true,
                Some(serving) => tracing::warn!(
                    epoch = drain.epoch,
                    serving = serving.epoch,
                    "the daemon drained an epoch this frontend does not serve"
                ),
                None => tracing::warn!(
                    epoch = drain.epoch,
                    "the daemon drained the frontend before an epoch served"
                ),
            },
            command::Kind::Pause(pause) => self.paused = pause.paused,
            other => {
                anyhow::bail!("the daemon sent the frontend a command it does not take: {other:?}")
            }
        }
        Ok(())
    }
}

/// The frontend's link to the daemon that manages it.
///
/// The link outlives the daemon's commands: when the connection ends, the
/// state stays at its last value and [`Self::ended`] says why.
#[derive(Clone)]
pub struct Control {
    state: watch::Sender<State>,
    /// Keeps one receiver of `state` alive, so the reader thread can always
    /// publish a command: a `watch` sender without a receiver publishes
    /// nothing. Never read itself.
    #[expect(dead_code, reason = "it exists to keep the sender's receiver alive")]
    current: watch::Receiver<State>,
    ended: watch::Sender<Option<String>>,
    /// Keeps one receiver of `ended` alive, for the same reason.
    #[expect(dead_code, reason = "it exists to keep the sender's receiver alive")]
    end: watch::Receiver<Option<String>>,
}

impl Control {
    /// Links to the daemon at `address` (`unix:<path>` or `<host>:<port>`),
    /// saying hello as the frontend of `deployment`, and follows its commands
    /// from a thread of its own.
    ///
    /// # Errors
    ///
    /// Fails when the daemon cannot be reached, refuses the frontend, or
    /// cannot be answered in time; a link that ends later is reported by
    /// [`Self::ended`].
    pub fn connect(address: &str, deployment: &str) -> anyhow::Result<Self> {
        let mut connection = proto::Connection::connect(address)
            .with_context(|| format!("connecting to the daemon at {address}"))?;
        connection
            .send(&process::Hello {
                major: proto::PROTOCOL_MAJOR,
                build: proto::BUILD.to_owned(),
                peer: Some(hello::Peer::Frontend(process::Frontend {
                    deployment: deployment.to_owned(),
                })),
            })
            .context("saying hello to the daemon")?;
        let welcome: process::Welcome = connection
            .recv(WELCOME_TIMEOUT)?
            .context("the daemon did not answer the frontend's hello")?;
        anyhow::ensure!(
            welcome.refusal.is_empty(),
            "the daemon refused the frontend: {}",
            welcome.refusal
        );

        let (state, current) = watch::channel(State::default());
        let (ended, end) = watch::channel::<Option<String>>(None);
        let read_state = state.clone();
        let read_ended = ended.clone();
        std::thread::Builder::new()
            .name("frontend-control".to_owned())
            .spawn(move || follow(connection, read_state, read_ended))
            .context("starting the frontend's control thread")?;
        Ok(Self {
            state,
            current,
            ended,
            end,
        })
    }

    /// The state the daemon last commanded, and its changes after that.
    pub fn state(&self) -> watch::Receiver<State> {
        self.state.subscribe()
    }

    /// The state the daemon last commanded.
    pub fn current(&self) -> State {
        self.state.borrow().clone()
    }

    /// Why the link ended, once it has: the daemon closed it, or sent
    /// something the frontend cannot take.
    pub fn ended(&self) -> Option<String> {
        self.ended.borrow().clone()
    }
}

/// Reads the daemon's commands until the link ends.
fn follow(
    mut connection: proto::Connection,
    state: watch::Sender<State>,
    ended: watch::Sender<Option<String>>,
) {
    loop {
        let message = match connection.recv::<process::Downstream>(COMMAND_TIMEOUT) {
            Ok(Some(message)) => message,
            // No command yet: the link is idle, not over.
            Ok(None) => continue,
            Err(error) => {
                let _ = ended.send(Some(format!("reading from the daemon: {error:#}")));
                return;
            }
        };
        let Some(downstream::Kind::Event(event)) = message.kind else {
            // The frontend asks the daemon nothing, so a reply answers no
            // request of this process.
            tracing::debug!("the daemon sent the frontend a message the frontend does not use");
            continue;
        };
        match event.kind {
            Some(event::Kind::Command(command)) => {
                let mut next = state.borrow().clone();
                if let Err(error) = next.apply(command) {
                    let _ = ended.send(Some(format!("{error:#}")));
                    return;
                }
                // Every command is published, even one that changes nothing,
                // so a serving side that waited for a `Serve` wakes on it.
                let _ = state.send(next);
            }
            Some(event::Kind::Exit(exit)) => {
                let _ = ended.send(Some(format!(
                    "the daemon reports the exit of {} {} rank {}: {}",
                    exit.host, exit.group, exit.rank, exit.status
                )));
                return;
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// A command of one kind, as the daemon sends it.
    fn command(kind: command::Kind) -> process::Command {
        process::Command { kind: Some(kind) }
    }

    /// A `Serve` of `epoch` at `requests`.
    fn serve(epoch: u64, requests: &str) -> process::Command {
        command(command::Kind::Serve(process::Serve {
            epoch,
            requests: requests.to_owned(),
        }))
    }

    /// A `Drain` of `epoch`.
    fn drain(epoch: u64) -> process::Command {
        command(command::Kind::Drain(process::Drain { epoch }))
    }

    /// A `Pause` of admissions.
    fn pause(paused: bool) -> process::Command {
        command(command::Kind::Pause(process::Pause { paused }))
    }

    #[test]
    fn the_frontend_admits_requests_only_for_an_epoch_that_serves() {
        let mut state = State::default();
        assert!(!state.admits(), "no epoch serves yet");

        state.apply(serve(4, "unix:/tmp/requests-4")).unwrap();
        assert!(state.admits());
        assert_eq!(state.serving().map(|serving| serving.epoch), Some(4));
        assert_eq!(
            state.serving().map(|serving| serving.requests.as_str()),
            Some("unix:/tmp/requests-4")
        );

        state.apply(drain(4)).unwrap();
        assert!(state.draining());
        assert!(!state.admits(), "a drained frontend admits nothing");

        state.apply(serve(5, "unix:/tmp/requests-5")).unwrap();
        assert!(!state.draining(), "the next epoch serves again");
        assert_eq!(state.serving().map(|serving| serving.epoch), Some(5));
    }

    #[test]
    fn a_pause_holds_admissions_without_ending_a_request() {
        let mut state = State::default();
        state.apply(serve(1, "unix:/tmp/requests")).unwrap();

        state.apply(pause(true)).unwrap();
        assert!(state.paused());
        assert!(!state.admits());

        state.apply(pause(false)).unwrap();
        assert!(!state.paused());
        assert!(state.admits());
        assert_eq!(
            state.serving().map(|serving| serving.epoch),
            Some(1),
            "the serving epoch is unchanged"
        );
    }

    #[test]
    fn a_drain_of_an_epoch_this_frontend_does_not_serve_changes_nothing() {
        let mut state = State::default();
        state.apply(serve(7, "unix:/tmp/requests-7")).unwrap();
        let serving = state.clone();

        state.apply(drain(6)).unwrap();
        assert_eq!(state, serving, "a stale drain leaves the epoch serving");

        // A drain before any epoch served leaves nothing to drain either.
        let mut fresh = State::default();
        fresh.apply(drain(7)).unwrap();
        assert!(!fresh.draining());
    }

    #[test]
    fn a_serve_without_a_request_address_is_refused() {
        let mut state = State::default();
        let refused = state.apply(serve(3, ""));
        assert!(refused.is_err(), "a frontend with nowhere to submit stops");
        assert!(!state.admits());
    }

    #[test]
    fn a_command_the_frontend_does_not_take_ends_its_link() {
        let mut state = State::default();
        let precopy = command(command::Kind::Precopy(process::Precopy::default()));
        assert!(state.apply(precopy).is_err());
    }

    /// The frontend follows a daemon's commands over the real wire: the hello
    /// names the deployment, and the commands reach the state.
    #[test]
    fn the_frontend_follows_the_daemons_commands_over_its_link() {
        let directory = tempfile::tempdir().expect("a directory for the socket");
        let path = directory.path().join("daemon.sock");
        let address = format!("unix:{}", path.display());
        let listener = proto::Listener::bind(&address).expect("a listener");

        let daemon = std::thread::spawn(move || {
            let mut connection = loop {
                if let Some(connection) = listener.try_accept().expect("accepting") {
                    break connection;
                }
                std::thread::sleep(Duration::from_millis(5));
            };
            let hello: process::Hello = connection
                .recv(Duration::from_secs(10))
                .expect("the hello")
                .expect("a hello");
            let Some(hello::Peer::Frontend(frontend)) = hello.peer else {
                panic!("the frontend did not say what it is");
            };
            assert_eq!(frontend.deployment, "qwen3");
            connection
                .send(&process::Welcome {
                    major: proto::PROTOCOL_MAJOR,
                    build: proto::BUILD.to_owned(),
                    refusal: String::new(),
                    nodes: Vec::new(),
                })
                .expect("the welcome");
            for kind in [
                command::Kind::Serve(process::Serve {
                    epoch: 11,
                    requests: "unix:/tmp/requests-11".to_owned(),
                }),
                command::Kind::Pause(process::Pause { paused: true }),
                command::Kind::Pause(process::Pause { paused: false }),
                command::Kind::Drain(process::Drain { epoch: 11 }),
            ] {
                connection
                    .send(&process::Downstream {
                        kind: Some(downstream::Kind::Event(process::Event {
                            kind: Some(event::Kind::Command(process::Command { kind: Some(kind) })),
                        })),
                    })
                    .expect("a command");
            }
            // Hold the link open until the frontend has read everything.
            std::thread::sleep(Duration::from_secs(1));
        });

        let control = Control::connect(&address, "qwen3").expect("the frontend links");
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let state = control.current();
            if state.draining() {
                assert_eq!(state.serving().map(|serving| serving.epoch), Some(11));
                assert_eq!(
                    state.serving().map(|serving| serving.requests.as_str()),
                    Some("unix:/tmp/requests-11")
                );
                assert!(!state.paused(), "the daemon released admissions again");
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the commands did not arrive: {state:?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        // The link ends when the daemon's side does, and says so.
        let deadline = Instant::now() + Duration::from_secs(10);
        while control.ended().is_none() {
            assert!(Instant::now() < deadline, "the link did not report its end");
            std::thread::sleep(Duration::from_millis(5));
        }
        daemon.join().expect("the daemon's side");
    }
}
