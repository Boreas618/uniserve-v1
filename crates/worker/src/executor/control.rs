//! The control thread's channels.
//!
//! An engine's control thread waits on the daemon's request interface and on the daemon's
//! supervision of the process; the other engine processes of the epoch are each served by a link
//! thread of their own ([`super::peers`]), which wakes this thread when a peer is lost. It never
//! waits for a job: every wait for a message ends, at the latest, at the deadline the watchdog
//! gives it ([`super::watchdog`]), so a job that stopped making progress is reported while the
//! channels go on being served.
//!
//! This module owns the connections and the waiting, not the meaning of what arrives. The
//! messages are the daemon's append-only Protobuf ([`proto`]); the engine decides what a message
//! does. Keeping the transport here keeps the engine clear of poll calls and socket addresses,
//! and lets a test drive the wait with a socket pair.

use std::time::{Duration, Instant};

use ipc::{Wake, Waker};
use proto::Connection;

use crate::{Error, Result};

/// How long an attaching daemon has to welcome an engine that connects to it. It is the
/// daemon's own connect bound, so a transfer that hangs is reported on the daemon's side
/// instead of waiting here without end.
const ATTACH_TIMEOUT: Duration = Duration::from_secs(10);

/// What the control thread's wait found, for the engine to act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ready {
    /// A frame from the daemon that launched this process is buffered.
    Daemon,
    /// A frame from an attached daemon is buffered: the hot restart's successor, whose request
    /// socket this engine connected to. The index is the daemon's place among the daemons that
    /// ever attached.
    Attached(usize),
    /// The process interface has something to read: the daemon's commands arrive there, on a
    /// socket of their own.
    Process,
    /// Something in this process fired the wake: a scheduler, a submission's completion, a lost
    /// peer link.
    Woken,
    /// Nothing arrived before the deadline passed. The caller does its periodic work (the
    /// watchdog's readings) and waits again.
    Idle,
}

/// The control thread's connections to the daemon, and the wake other threads end its wait
/// with.
pub struct Control {
    daemon: Connection,
    /// The daemons this engine attached to besides the one that launched it, by the index the
    /// request records use: a hot restart's successor, which takes the responses in flight over
    /// while the process goes on serving. An entry stays in place once its daemon detached, so
    /// no other front's index moves.
    attached: Vec<Option<Connection>>,
    engine: u64,
    waker: Waker,
    /// The process interface's descriptor, while it is open.
    process: Option<std::os::fd::RawFd>,
}

impl Control {
    /// Connects to the daemon's request interface at `requests` (`unix:<path>` or
    /// `<host>:<port>`) as the engine numbered `engine`. The handshake with the daemon is the
    /// caller's, because what the daemon sends decides how the engine is built.
    ///
    /// # Errors
    ///
    /// When the connection or the wake cannot be made.
    pub fn connect(requests: &str, engine: u64) -> Result<Self> {
        let daemon = Connection::connect(requests).map_err(|error| {
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

    /// A handle that ends the control thread's wait, for any thread of the process.
    pub fn wake(&self) -> Wake {
        self.waker.wake()
    }

    /// States the process interface's descriptor, or `None` once it closed: a closed socket
    /// stays readable, and waiting on it would never let a wait end at its deadline.
    pub fn watch_process(&mut self, process: Option<std::os::fd::RawFd>) {
        self.process = process;
    }

    /// Waits for the next thing to serve, and not later than `deadline`.
    ///
    /// Everything the thread serves is waited on at once: the daemon's request interface, its
    /// process interface, every attached daemon, and the local wake. `None` waits indefinitely,
    /// which the caller uses only while no job is in flight; the watchdog gives a deadline while
    /// one is.
    ///
    /// # Errors
    ///
    /// When the wait itself fails.
    pub fn wait(&mut self, deadline: Option<Instant>) -> Result<Vec<Ready>> {
        // The daemon's connection comes first, so it is found at index zero whatever fronts
        // attach later.
        let mut descriptors: Vec<libc::pollfd> = vec![readable(self.daemon.descriptor())];

        let process_index = self.process.map(|process| {
            descriptors.push(readable(process));
            descriptors.len() - 1
        });

        // The front each attached descriptor belongs to: a detached entry keeps its index and
        // has no descriptor.
        let attached_start = descriptors.len();
        let mut attached_fronts = Vec::new();
        for (index, front) in self.attached.iter().enumerate() {
            let Some(front) = front else { continue };
            attached_fronts.push(index);
            descriptors.push(readable(front.descriptor()));
        }
        let attached_end = descriptors.len();

        let wake_index = descriptors.len();
        descriptors.push(self.waker.poll_fd());

        // Milliseconds until the deadline, rounded up so the wait never ends before it.
        let timeout = deadline.map_or(-1, |deadline| {
            let left = deadline.saturating_duration_since(Instant::now());
            let millis =
                left.as_millis() + u128::from(!left.subsec_nanos().is_multiple_of(1_000_000));
            i32::try_from(millis).unwrap_or(i32::MAX)
        });

        // SAFETY: the call borrows this live array for its own duration only.
        let ready = unsafe {
            libc::poll(
                descriptors.as_mut_ptr(),
                descriptors.len() as libc::nfds_t,
                timeout,
            )
        };
        if ready < 0 {
            return Err(Error::transport_source(
                "waiting on the control thread's channels",
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
        for (descriptor, front) in descriptors[attached_start..attached_end]
            .iter()
            .zip(&attached_fronts)
        {
            if descriptor.revents != 0 {
                events.push(Ready::Attached(*front));
            }
        }
        if descriptors[wake_index].revents != 0 {
            // The wake's byte is its whole state: the caller acts on what the thread that fired
            // it changed, not on the byte.
            self.waker.drain();
            events.push(Ready::Woken);
        }
        Ok(events)
    }

    /// The daemon's request interface.
    pub fn daemon_mut(&mut self) -> &mut Connection {
        &mut self.daemon
    }

    /// The fronts a report of this process's own state goes out on: the launching daemon's,
    /// and every daemon that attached since.
    ///
    /// A hot restart closes the launching daemon's front and leaves the attached ones serving,
    /// so a process that reported on index 0 alone would fall silent when a successor took it
    /// over. Every daemon keeps its own registry, so a report on each front is one figure per
    /// registry.
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

    /// The connection of front `front`, when it is live: index 0 is the daemon that launched
    /// this process.
    pub fn front_mut(&mut self, front: usize) -> Option<&mut Connection> {
        match front {
            0 => Some(&mut self.daemon),
            index => self.attached.get_mut(index - 1)?.as_mut(),
        }
    }

    /// Connects to another daemon's request socket at `address` and serves it as a further
    /// front. Returns the front's index, which the request records keep: a response taken over
    /// by that daemon names it.
    ///
    /// The engine states itself first (its hello, the daemon's welcome, then `state` and `info`,
    /// what it serves), because the successor counts this engine as serving on that message.
    ///
    /// # Errors
    ///
    /// When the connection cannot be made, the daemon refuses the engine, or it does not
    /// welcome it in time. The process keeps serving the fronts it has: a refused attach is a
    /// failed hot restart, not a failure of this engine.
    pub fn attach(
        &mut self,
        address: &str,
        state: i32,
        info: Option<proto::request::wire::EngineInfo>,
    ) -> Result<usize> {
        let mut front = Connection::connect(address).map_err(|error| {
            Error::transport_source(
                "connecting to the attaching daemon's request interface",
                error,
            )
        })?;

        front
            .send(&proto::request::wire::Hello {
                major: proto::ENGINE_INTERFACE_VERSION,
                build: proto::BUILD.to_owned(),
                engine: self.engine,
                // A request link of a process of the deployment: the successor files it as that
                // engine's request interface, which serves requests and takes no command.
                interface: proto::process::Interface::Requests as i32,
            })
            .map_err(|error| {
                Error::transport_source("stating this engine to the attaching daemon", error)
            })?;

        let welcome: proto::request::wire::Welcome = front
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

        front
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

        self.attached.push(Some(front));
        Ok(self.attached.len())
    }

    /// How many attached daemons are connected now.
    pub fn live_attached(&self) -> usize {
        self.attached.iter().flatten().count()
    }

    /// How many attached fronts this process has ever had, live or not.
    pub fn attached_fronts(&self) -> usize {
        self.attached.len()
    }

    /// Closes the connection of the attached front `front` (1 or more), whose daemon detached.
    /// Its index stays taken.
    pub fn detach(&mut self, front: usize) {
        let Some(entry) = front
            .checked_sub(1)
            .and_then(|index| self.attached.get_mut(index))
        else {
            return;
        };

        if let Some(connection) = entry.as_mut() {
            connection.close();
        }
        *entry = None;
    }

    /// The engine number the daemon started this process under.
    pub fn engine(&self) -> u64 {
        self.engine
    }

    /// Closes the daemon's connection.
    pub fn close(&mut self) {
        self.daemon.close();
    }
}

/// The entry of a wait for `descriptor` to become readable.
fn readable(descriptor: std::os::fd::RawFd) -> libc::pollfd {
    libc::pollfd {
        fd: descriptor,
        events: libc::POLLIN,
        revents: 0,
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;

    use crossbeam_channel::{Receiver, Sender};

    use super::*;

    /// A control thread's channels over a daemon connection that sends nothing, with the
    /// daemon's end: dropping it closes the connection, which makes it readable.
    fn control(engine: u64) -> (Control, UnixStream) {
        let (ours, theirs) = UnixStream::pair().expect("a socket pair");
        let control = Control {
            daemon: Connection::from_unix(ours).expect("a connection over the pair"),
            attached: Vec::new(),
            engine,
            waker: Waker::new().expect("a control wake"),
            process: None,
        };
        (control, theirs)
    }

    #[test]
    fn a_wake_ends_a_wait_without_any_message() -> Result<()> {
        let (mut control, _daemon) = control(1);

        control.wake().wake();
        let ready = control.wait(None)?;
        assert_eq!(ready, vec![Ready::Woken]);
        Ok(())
    }

    #[test]
    fn a_wait_ends_at_its_deadline_with_nothing_to_serve() -> Result<()> {
        let (mut control, _daemon) = control(1);

        let started = Instant::now();
        let ready = control.wait(Some(started + Duration::from_millis(50)))?;
        assert_eq!(ready, vec![Ready::Idle]);
        assert!(started.elapsed() >= Duration::from_millis(50));
        Ok(())
    }

    /// A daemon that accepts one attach on `listener`: it reads the engine's hello, responds
    /// with a welcome carrying `refusal`, reads the state an engine it did not refuse sends,
    /// sends `frame`, and keeps the connection open until `release` is dropped. The thread
    /// returns what the daemon read.
    fn attaching_daemon(
        listener: proto::Listener,
        refusal: String,
        frame: Option<proto::request::wire::ToEngine>,
    ) -> (
        std::thread::JoinHandle<(proto::request::wire::Hello, proto::request::wire::State)>,
        Sender<()>,
    ) {
        let (release, held): (Sender<()>, Receiver<()>) = crossbeam_channel::bounded(0);
        let refuses = !refusal.is_empty();

        let daemon = std::thread::spawn(move || {
            let mut connection = loop {
                // The listener is readable once the engine connected.
                let mut accepting = [readable(listener.descriptor())];
                // SAFETY: the call borrows this live array for its own duration only.
                unsafe { libc::poll(accepting.as_mut_ptr(), 1, 10_000) };

                if let Some(connection) = listener.try_accept().expect("accepting the engine") {
                    break connection;
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
                .expect("responding to the hello");

            // A refused engine states nothing: it closes the connection once it read the
            // refusal.
            let state = if refuses {
                proto::request::wire::State::default()
            } else {
                let reported: Option<proto::request::wire::FromEngine> = connection
                    .recv(Duration::from_secs(10))
                    .expect("reading the state");
                reported
                    .and_then(|message| match message.kind {
                        Some(proto::request::wire::from_engine::Kind::State(state)) => Some(state),
                        _ => None,
                    })
                    .expect("the engine stated what it serves")
            };
            if let Some(frame) = frame {
                connection.send(&frame).expect("sending a frame");
            }

            // The front stays live until the test is done with it.
            let _ = held.recv();
            (hello, state)
        });
        (daemon, release)
    }

    #[test]
    fn an_engine_attaches_to_a_second_daemon_s_request_socket() -> Result<()> {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let successor = format!("unix:{}", directory.path().join("requests.sock").display());
        let listener = proto::Listener::bind(&successor).expect("the successor's request socket");
        let (daemon, release) = attaching_daemon(
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
        let (mut control, _daemon) = control(7);

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
        assert_eq!(control.report_fronts(), vec![0, 1]);

        // The frame the successor sent is read through that front, and the wait reports the
        // front by its index, not as the launching daemon's.
        let ready = control.wait(Some(Instant::now() + Duration::from_secs(10)))?;
        assert_eq!(ready, vec![Ready::Attached(0)]);
        let message: proto::request::wire::ToEngine = control
            .front_mut(front)
            .expect("the attached front is live")
            .try_recv()
            .map_err(|error| Error::transport_source("reading the attached front", error))?
            .expect("the successor's frame arrived");
        assert!(message.kind.is_some());

        drop(release);
        let (hello, state) = daemon.join().expect("the daemon thread");
        assert_eq!(hello.engine, 7, "the engine stated its own number");
        assert_eq!(hello.interface, proto::process::Interface::Requests as i32);
        assert_eq!(state.state, proto::request::wire::EngineState::Ready as i32);

        // A detached front keeps its index, and no report goes out on it.
        control.detach(front);
        assert_eq!(control.live_attached(), 0);
        assert_eq!(control.attached_fronts(), 1);
        assert_eq!(control.report_fronts(), vec![0]);
        Ok(())
    }

    #[test]
    fn a_refused_attach_leaves_the_engine_with_the_front_it_had() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let successor = format!("unix:{}", directory.path().join("requests.sock").display());
        let listener = proto::Listener::bind(&successor).expect("the successor's request socket");
        let (daemon, release) = attaching_daemon(listener, "another build".to_owned(), None);
        let (mut control, _daemon) = control(7);

        let refused = control.attach(
            &successor,
            proto::request::wire::EngineState::Ready as i32,
            None,
        );
        assert!(refused.is_err(), "a refused attach is not a front");
        assert_eq!(control.attached_fronts(), 0);
        assert!(control.front_mut(1).is_none());

        drop(release);
        daemon.join().expect("the daemon thread");
    }
}
