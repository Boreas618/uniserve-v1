//! The control thread's channels.
//!
//! The main process's control thread waits on the daemon's request interface
//! and its process supervision; the peer main processes of the epoch are each
//! served by a link thread of their own ([`crate::peers`]), which wakes this
//! thread when a peer is lost. It never waits for a job; every
//! wait for a message ends, at the latest, at the deadline the watchdog gives it
//! (§2.4), so a job that stopped making progress is reported while the channels
//! go on being served.
//!
//! This module owns the connections and the waiting, not the meaning of what
//! arrives. The messages' types are the daemon's append-only Protobuf
//! ([`proto`]); the engine's main process (`engine::process`, whose `update`
//! module holds the generation update) decides what a message does.
//! Keeping the transport here lets that driver stay clear of poll loops and
//! socket addresses, and lets a test drive the loop with a socket pair.

use std::time::{Duration, Instant};

use ipc::{Wake, Waker};
use proto::Connection;

use crate::{Error, Result};

/// How long an attaching daemon has to welcome an engine that connects to it
/// (`Attach`, §7.5.1 step 2). It is the daemon's own connect bound, so a
/// handover that hangs is reported on the daemon's side rather than waiting
/// here forever.
const ATTACH_TIMEOUT: Duration = Duration::from_secs(10);

/// Where the main process reaches the processes on its other ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoints {
    /// The leader daemon's request interface, `unix:<path>` or `<host>:<port>`.
    pub requests: String,
}

/// What the control thread's wait found, for the driver to act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ready {
    /// A frame from the daemon that launched this process is buffered.
    Daemon,
    /// A frame from an attached daemon is buffered: the hot restart successor
    /// whose request socket this engine connected to (`Attach`, §7.5.1 step 2).
    Attached(usize),
    /// The process interface has something to read: the daemon's commands
    /// arrive there, on a socket of their own.
    Process,
    /// Something in this process fired the wake: a scheduler, a worker's
    /// completion, a lost peer link, or the §2.4 watchdog's deadline.
    Woken,
    /// Nothing arrived before the deadline passed. The caller runs its
    /// periodic work (the watchdog's readings) and waits again.
    Idle,
}

/// The control thread's connections to the daemon, and the wake other threads
/// use to end a wait.
pub struct Control {
    daemon: Connection,
    /// The daemons this engine attached to besides the one that launched it,
    /// by the index the request records use: a hot restart's successor, which
    /// takes the responses in flight over while the process goes on serving
    /// (§7.5.1 step 2). An entry stays in place once its daemon detaches, so
    /// that no other front's index moves.
    attached: Vec<Option<Connection>>,
    engine: u64,
    waker: Waker,
    /// The process interface's descriptor, while it is open.
    process: Option<std::os::fd::RawFd>,
}

impl Control {
    /// Connects to the daemon's request interface. The handshake with the
    /// daemon (its hello and the daemon's spec) is the caller's, because the
    /// spec decides how the main process is built.
    pub fn connect(endpoints: &Endpoints, engine: u64) -> Result<Self> {
        let daemon = Connection::connect(&endpoints.requests).map_err(|error| {
            Error::transport_source("connecting to the daemon's request interface", error)
        })?;
        let waker = Waker::new()
            .map_err(|error| Error::transport_source("creating the control wake", error))?;
        Ok(Self {
            daemon,
            attached: Vec::new(),
            engine,
            waker,
            process: None,
        })
    }

    /// A handle that ends the control thread's wait, for any thread of the
    /// process.
    pub fn wake(&self) -> Wake {
        self.waker.wake()
    }

    /// States the process interface's descriptor, or `None` once it closed: a
    /// closed socket stays readable, and polling it would never let a wait end
    /// at its deadline.
    pub fn watch_process(&mut self, process: Option<std::os::fd::RawFd>) {
        self.process = process;
    }

    /// Waits for the next thing to serve, and not later than `deadline`.
    ///
    /// Every stream the thread serves is polled at once: the daemon's request
    /// interface, its process interface, every attached daemon, and the local
    /// wake.
    /// `None` waits indefinitely, which the caller uses only when no job is in
    /// flight; the watchdog always gives a deadline while one is.
    pub fn wait(&mut self, deadline: Option<Instant>) -> Result<Vec<Ready>> {
        let mut descriptors: Vec<libc::pollfd> = Vec::new();
        // The daemon's connection comes first, so the driver can find it at
        // index zero whatever fronts attach later.
        descriptors.push(libc::pollfd {
            fd: self.daemon.descriptor(),
            events: libc::POLLIN,
            revents: 0,
        });
        let process_index = self.process.map(|process| {
            descriptors.push(libc::pollfd {
                fd: process,
                events: libc::POLLIN,
                revents: 0,
            });
            descriptors.len() - 1
        });
        let attached_start = descriptors.len();
        // The front each attached descriptor belongs to: a detached entry keeps
        // its index and has no descriptor, so the two are read together.
        let mut attached_fronts = Vec::new();
        for (index, front) in self.attached.iter().enumerate() {
            let Some(front) = front else { continue };
            attached_fronts.push(index);
            descriptors.push(libc::pollfd {
                fd: front.descriptor(),
                events: libc::POLLIN,
                revents: 0,
            });
        }
        let attached_end = descriptors.len();
        let wake_index = descriptors.len();
        descriptors.push(self.waker.poll_fd());

        let timeout = match deadline {
            Some(deadline) => {
                let left = deadline.saturating_duration_since(Instant::now());
                let millis =
                    left.as_millis() + u128::from(!left.subsec_nanos().is_multiple_of(1_000_000));
                Some(i32::try_from(millis.min(i32::MAX as u128)).unwrap_or(i32::MAX))
            }
            None => None,
        };
        // SAFETY: poll borrows this live array only for the duration of the call.
        let ready = unsafe {
            libc::poll(
                descriptors.as_mut_ptr(),
                descriptors.len() as libc::nfds_t,
                timeout.unwrap_or(-1),
            )
        };
        if ready < 0 {
            return Err(Error::transport_source(
                "polling the control thread's channels",
                std::io::Error::last_os_error(),
            ));
        }
        if ready == 0 {
            return Ok(vec![Ready::Idle]);
        }

        let mut events = Vec::new();
        if descriptors[0].revents != 0 {
            events.push(Ready::Daemon);
        }
        if process_index.is_some_and(|index| descriptors[index].revents != 0) {
            events.push(Ready::Process);
        }
        // An attached front reports its index, which is its place among the
        // fronts the process has ever had, not among the live ones.
        for (descriptor, front) in descriptors[attached_start..attached_end]
            .iter()
            .zip(&attached_fronts)
        {
            if descriptor.revents != 0 {
                events.push(Ready::Attached(*front));
            }
        }
        if descriptors[wake_index].revents != 0 {
            // Drain the wake whether or not the readiness was ours: the byte is
            // the wake's whole state, and the driver acts on the state the
            // firing thread changed, not on the byte.
            self.waker.drain();
            events.push(Ready::Woken);
        }
        Ok(events)
    }

    /// The daemon's request interface.
    pub fn daemon_mut(&mut self) -> &mut Connection {
        &mut self.daemon
    }

    /// The fronts a report of this process's own state may go out on: the
    /// launching daemon's, and every daemon that attached since.
    ///
    /// A handover closes the launching daemon's front and leaves the attached
    /// ones serving (§7.5.1 step 2), so a process that reported only on index 0
    /// would fall silent exactly when a successor took it over — and the
    /// successor's `/metrics` would hold no figure of the engine it inherited.
    /// Every daemon keeps its own registry, so a report on each front is one
    /// figure per registry, not one figure counted twice.
    pub fn report_fronts(&self) -> Vec<usize> {
        let mut fronts = vec![0];
        fronts.extend(
            self.attached
                .iter()
                .enumerate()
                .filter(|(_, front)| front.is_some())
                .map(|(index, _)| index + 1),
        );
        fronts
    }

    /// The connection of the front `front`, when it is live: index 0 is the
    /// daemon that launched this process.
    pub fn front_mut(&mut self, front: usize) -> Option<&mut Connection> {
        match front {
            0 => Some(&mut self.daemon),
            index => self.attached.get_mut(index - 1)?.as_mut(),
        }
    }

    /// Connects to another daemon's request socket and serves it as a further
    /// front (`Attach`, §7.5.1 step 2).
    ///
    /// The engine states itself first - its hello, the daemon's welcome, then
    /// what it serves - because the successor counts this engine as serving on
    /// the `State` that follows. Returns the front's index, which the request
    /// records keep: a response taken over by that daemon names it.
    ///
    /// # Errors
    ///
    /// Fails when the connection cannot be made, when the daemon refuses the
    /// engine, or when it does not welcome it in time. The process keeps
    /// serving the fronts it has; a refused attach is a failed handover, not a
    /// failure of this engine.
    pub fn attach(
        &mut self,
        address: &str,
        state: i32,
        info: Option<proto::request::wire::EngineInfo>,
    ) -> Result<usize> {
        let mut reader = Connection::connect(address).map_err(|error| {
            Error::transport_source(
                "connecting to the attaching daemon's request interface",
                error,
            )
        })?;
        reader
            .send(&proto::request::wire::Hello {
                major: proto::ENGINE_INTERFACE_VERSION,
                build: proto::BUILD.to_owned(),
                engine: self.engine,
                // A request link of a process of the deployment: the successor
                // files it as that engine's request interface, which serves
                // requests and takes no command.
                interface: proto::process::Interface::Requests as i32,
            })
            .map_err(|error| {
                Error::transport_source("stating this engine to the attaching daemon", error)
            })?;
        let welcome: proto::request::wire::Welcome = reader
            .recv(ATTACH_TIMEOUT)
            .map_err(|error| {
                Error::transport_source("awaiting the attaching daemon's welcome", error)
            })?
            .ok_or_else(|| {
                Error::transport("the attaching daemon closed before it welcomed this engine")
            })?;
        if !welcome.refusal.is_empty() {
            return Err(Error::transport(format!(
                "the attaching daemon refused this engine: {}",
                welcome.refusal
            )));
        }
        // What this process is, stated on the new front: the successor counts
        // the engine as serving on this message and on nothing else.
        reader
            .send(&proto::request::wire::FromEngine {
                kind: Some(proto::request::wire::from_engine::Kind::State(
                    proto::request::wire::State {
                        state,
                        info,
                        error: String::new(),
                    },
                )),
            })
            .map_err(|error| {
                Error::transport_source(
                    "stating what this engine serves to the attaching daemon",
                    error,
                )
            })?;
        self.attached.push(Some(reader));
        Ok(self.attached.len())
    }

    /// How many attached daemons are connected now.
    pub fn live_attached(&self) -> usize {
        self.attached.iter().flatten().count()
    }

    /// How many attached fronts this process has ever had, live or not: the
    /// indices stay in place so no request's record moves when one detaches.
    pub fn attached_fronts(&self) -> usize {
        self.attached.len()
    }

    /// Forgets the front `index - 1`'s connection once its daemon detached.
    pub fn detach(&mut self, front: usize) {
        if let Some(entry) = self.attached.get_mut(front - 1) {
            if let Some(connection) = entry.as_mut() {
                connection.close();
            }
            *entry = None;
        }
    }

    /// The engine number the daemon started this main process under.
    pub fn engine(&self) -> u64 {
        self.engine
    }

    /// Closes the daemon's connection.
    pub fn close(&mut self) {
        self.daemon.close();
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;

    use super::*;

    /// A control thread over a daemon connection that answers nothing, and this
    /// process's engine number.
    fn control(engine: u64) -> Control {
        let (ours, theirs) = UnixStream::pair().expect("a socket pair");
        // The daemon's end stays open and answers nothing, which is silence. Dropping it would be a
        // disconnect - a socket that is readable at once, and `Ready::Daemon` is the right answer to
        // that - so a test that asserts `Idle` must keep it.
        #[expect(
            clippy::mem_forget,
            reason = "the daemon's end of the pair must stay open for the test process's life: \
                      closing it would make the daemon connection readable at once"
        )]
        std::mem::forget(theirs);
        Control {
            daemon: Connection::from_unix(ours).expect("a connection over the pair"),
            attached: Vec::new(),
            engine,
            waker: Waker::new().expect("a control wake"),
            process: None,
        }
    }

    #[test]
    fn a_wake_ends_a_wait_without_any_message() -> Result<()> {
        let mut control = control(1);
        let wake = control.wake();
        wake.wake();
        let ready = control.wait(Some(Instant::now() + Duration::from_secs(10)))?;
        assert!(ready.contains(&Ready::Woken), "{ready:?}");
        Ok(())
    }

    #[test]
    fn a_wait_ends_at_its_deadline_with_nothing_to_serve() -> Result<()> {
        let mut control = control(1);
        let started = Instant::now();
        let ready = control.wait(Some(started + Duration::from_millis(50)))?;
        assert_eq!(ready, vec![Ready::Idle]);
        assert!(started.elapsed() >= Duration::from_millis(50));
        Ok(())
    }

    /// A daemon that accepts one attach: it reads the engine's hello and the
    /// state that follows, answers `Welcome`, and then sends `frame` so the
    /// engine's wait reports the new front.
    ///
    /// Returns the thread, which answers with what the daemon read.
    fn attaching_daemon(
        listener: proto::Listener,
        refusal: String,
        frame: Option<proto::request::wire::ToEngine>,
    ) -> std::thread::JoinHandle<(proto::request::wire::Hello, proto::request::wire::State)> {
        // Whether the daemon refuses this engine, kept beside the welcome it is sent in: the
        // fixture's own branch below asks the same question, and the welcome consumes the string.
        let refuses = !refusal.is_empty();
        std::thread::spawn(move || {
            let mut connection = loop {
                match listener.try_accept() {
                    Ok(Some(connection)) => break connection,
                    Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                    Err(error) => panic!("accepting the engine: {error}"),
                }
            };
            let hello: proto::request::wire::Hello = connection
                .recv(Duration::from_secs(10))
                .expect("reading the hello")
                .expect("the engine said hello");
            connection
                .send(&proto::request::wire::Welcome {
                    major: proto::ENGINE_INTERFACE_VERSION,
                    build: proto::BUILD.to_owned(),
                    refusal,
                })
                .expect("answering the hello");
            // A refused engine states nothing: the refusal is the whole answer and the engine
            // closes the connection after reading it, which is the correct behaviour and the
            // reason this fixture must not wait for a state frame it will never receive.
            let state: proto::request::wire::State = if !refuses {
                let reported: Option<proto::request::wire::FromEngine> = connection
                    .recv(Duration::from_secs(10))
                    .expect("reading the state");
                reported
                    .and_then(|message| match message.kind {
                        Some(proto::request::wire::from_engine::Kind::State(state)) => Some(state),
                        _ => None,
                    })
                    .expect("the engine stated what it serves")
            } else {
                proto::request::wire::State::default()
            };
            if let Some(frame) = frame {
                connection.send(&frame).expect("sending a frame");
            }
            // The connection stays open until the test's own end, so the front
            // is live for the whole assertion.
            std::thread::sleep(Duration::from_secs(2));
            (hello, state)
        })
    }

    /// The engine gives a hot restart's successor a front of its own: it
    /// connects to the address the daemon names, says what it is, and serves
    /// the frames that arrive on that connection (§7.5.1 step 2).
    ///
    /// Before the handler existed the process refused the command and kept one
    /// front, so the successor never became a daemon the engine could hand a
    /// response to.
    #[test]
    fn an_engine_attaches_to_a_second_daemons_request_socket() -> Result<()> {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let successor = format!("unix:{}", directory.path().join("requests.sock").display());
        let listener = proto::Listener::bind(&successor).expect("the successor's request socket");
        let daemon = attaching_daemon(
            listener,
            String::new(),
            Some(proto::request::wire::ToEngine {
                kind: Some(proto::request::wire::to_engine::Kind::Engine(
                    proto::request::wire::EngineCommand {
                        kind: Some(proto::request::wire::engine_command::Kind::Shutdown(
                            proto::request::wire::Shutdown {},
                        )),
                    },
                )),
            }),
        );
        let mut control = control(7);

        let front = control.attach(
            &successor,
            proto::request::wire::EngineState::Ready as i32,
            None,
        )?;
        assert_eq!(
            front, 1,
            "the attaching daemon is the process's second front"
        );
        assert_eq!(control.attached_fronts(), 1);

        // The frame the successor sent is readable through that front, and the
        // wait reports it by its index rather than as the launching daemon's.
        let ready = control.wait(Some(Instant::now() + Duration::from_secs(10)))?;
        assert!(
            ready.contains(&Ready::Attached(0)),
            "the attached front was not reported: {ready:?}"
        );
        let message: proto::request::wire::ToEngine = control
            .front_mut(front)
            .expect("the attached front is live")
            .try_recv()
            .map_err(|error| Error::transport_source("reading the attached front", error))?
            .expect("the successor's frame arrived");

        let (hello, state) = daemon.join().expect("the daemon thread");
        assert_eq!(hello.engine, 7, "the engine stated its own number");
        assert_eq!(
            hello.interface,
            proto::process::Interface::Requests as i32,
            "the attach is the engine's request link"
        );
        assert_eq!(
            state.state,
            proto::request::wire::EngineState::Ready as i32,
            "the state that makes the successor count this engine as serving"
        );
        assert!(message.kind.is_some());
        Ok(())
    }

    /// A daemon that refuses the attach changes nothing: the engine keeps the
    /// fronts it has and its process does not end. A refused attach is a failed
    /// handover, and the daemon that asked is still serving its requests.
    #[test]
    fn a_refused_attach_leaves_the_engine_with_the_front_it_had() -> Result<()> {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let successor = format!("unix:{}", directory.path().join("requests.sock").display());
        let listener = proto::Listener::bind(&successor).expect("the successor's request socket");
        let daemon = attaching_daemon(listener, "another build".to_owned(), None);
        let mut control = control(7);

        let refused = control.attach(
            &successor,
            proto::request::wire::EngineState::Ready as i32,
            None,
        );
        assert!(refused.is_err(), "a refused attach must not be a front");
        assert_eq!(control.attached_fronts(), 0);
        assert!(control.front_mut(1).is_none(), "no second front exists");
        daemon.join().expect("the daemon thread");
        Ok(())
    }
}
