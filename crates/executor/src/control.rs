//! The control thread's channels (`specs/worker.md` §2.3, §2.4).
//!
//! The main process's control thread is the only thread that waits on another
//! process: the daemon's request interface, the daemon's process supervision,
//! and the peer main processes of the epoch. It never waits for a job; every
//! wait for a message ends, at the latest, at the deadline the watchdog gives it
//! (§2.4), so a job that stopped making progress is reported while the channels
//! go on being served.
//!
//! This module owns the connections and the waiting, not the meaning of what
//! arrives. The messages' types are the daemon's append-only Protobuf
//! ([`proto`]) and the epoch's own Rust peer messages ([`ipc`]); the scheduler
//! node's fan-out and the program-layer driver in `crates/executor`'s control
//! users ([`crate::hot_update`], and `src/engine.rs`) decide what a message does.
//! Keeping the transport here lets that driver stay clear of poll loops and
//! socket addresses, and lets a test drive the loop with a socket pair.
//!
//! A peer main process connects to the address its launch names and sends its
//! [`ipc::PeerRegistration`] first; [`Control::accept`] reads it and refuses
//! another epoch's or another build's registration, because the two ends would
//! decode each other's frames into other meanings.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use ipc::{Channel, Listener, PeerAccepted, PeerRegistration, Wake, Waker};
use proto::Connection;

use crate::{Error, Result};

/// How long a peer's registration may take to arrive before the connection is
/// refused: a peer sends it as the connection's first frame.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(10);

/// Where the main process reaches the processes on its other ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoints {
    /// The leader daemon's request interface, `unix:<path>` or `<host>:<port>`.
    pub requests: String,
    /// The addresses on which peer main processes of this epoch connect, when
    /// this node accepts any. The scheduler node always does; a node whose
    /// ranks are driven by the scheduler node accepts one connection per peer
    /// that reaches it.
    pub peers: Vec<String>,
}

/// What the control thread's wait found, for the driver to act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ready {
    /// A frame from the daemon is buffered.
    Daemon,
    /// A peer main process connected: [`Control::accept`] takes it.
    PeerConnecting,
    /// A peer has a frame buffered; read it with [`Control::peer_mut`].
    Peer(u32),
    /// Something in this process fired the wake: a scheduler, an executor's
    /// completion, or the §2.4 watchdog's deadline.
    Woken,
    /// Nothing arrived before the deadline passed. The caller runs its
    /// periodic work (the watchdog's readings) and waits again.
    Idle,
}

/// The control thread's connections: the daemon and the epoch's peers, and the
/// wake other threads use to end a wait.
pub struct Control {
    daemon: Connection,
    engine: u64,
    listener: Option<Listener>,
    peers: BTreeMap<u32, Channel>,
    waker: Waker,
}

impl Control {
    /// Connects to the daemon's request interface and listens for the epoch's
    /// peers. The handshake with the daemon (its hello and the daemon's spec) is
    /// the caller's, because the spec decides how the main process is built.
    pub fn connect(endpoints: &Endpoints, engine: u64) -> Result<Self> {
        let daemon = Connection::connect(&endpoints.requests).map_err(|error| {
            Error::transport_source("connecting to the daemon's request interface", error)
        })?;
        let listener = if endpoints.peers.is_empty() {
            None
        } else {
            let addresses: Vec<&str> = endpoints.peers.iter().map(String::as_str).collect();
            Some(Listener::bind(&addresses).map_err(|error| {
                Error::transport_source("listening for peer main processes", error)
            })?)
        };
        let waker = Waker::new()
            .map_err(|error| Error::transport_source("creating the control wake", error))?;
        Ok(Self {
            daemon,
            engine,
            listener,
            peers: BTreeMap::new(),
            waker,
        })
    }

    /// A handle that ends the control thread's wait, for any thread of the
    /// process.
    pub fn wake(&self) -> Wake {
        self.waker.wake()
    }

    /// Waits for the next thing to serve, and not later than `deadline`.
    ///
    /// Every stream the thread serves is polled at once: the daemon's request
    /// interface, the peer listener, every connected peer, and the local wake.
    /// `None` waits indefinitely, which the caller uses only when no job is in
    /// flight; the watchdog always gives a deadline while one is.
    pub fn wait(&mut self, deadline: Option<Instant>) -> Result<Vec<Ready>> {
        let mut descriptors: Vec<libc::pollfd> = Vec::new();
        // The daemon's connection comes first, so the driver can find it at
        // index zero whatever peers connect later.
        descriptors.push(libc::pollfd {
            fd: self.daemon.descriptor(),
            events: libc::POLLIN,
            revents: 0,
        });
        let listener_start = descriptors.len();
        if let Some(listener) = &self.listener {
            descriptors.extend(listener.poll_fds());
        }
        let listener_end = descriptors.len();
        let mut peer_ranges: Vec<(u32, usize, usize)> = Vec::new();
        for (rank, channel) in &self.peers {
            let start = descriptors.len();
            descriptors.extend(channel.poll_fds());
            peer_ranges.push((*rank, start, descriptors.len()));
        }
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
        if descriptors[listener_start..listener_end]
            .iter()
            .any(|descriptor| descriptor.revents != 0)
        {
            events.push(Ready::PeerConnecting);
        }
        for (rank, start, end) in peer_ranges {
            if descriptors[start..end]
                .iter()
                .any(|descriptor| descriptor.revents != 0)
            {
                events.push(Ready::Peer(rank));
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

    /// Takes one connecting peer, reads its registration, and keeps its channel
    /// under the rank it serves. Returns the rank, or `None` when no connection
    /// waits.
    pub fn accept(&mut self) -> Result<Option<u32>> {
        let Some(listener) = &self.listener else {
            return Ok(None);
        };
        let mut channel = match listener.try_accept().map_err(|error| {
            Error::transport_source("accepting a peer main process", error)
        })? {
            Some(channel) => channel,
            None => return Ok(None),
        };
        channel.wait(REGISTRATION_TIMEOUT).map_err(|error| {
            Error::transport_source("waiting for a peer's registration", error)
        })?;
        let registration: PeerRegistration = channel
            .try_recv()
            .map_err(|error| Error::transport_source("reading a peer's registration", error))?
            .ok_or_else(|| Error::transport("a peer connected without a registration"))?;
        if !registration.accepted_by(self.engine) {
            return Err(Error::Invalid(format!(
                "rank {} of engine {} connected to engine {}",
                registration.rank, registration.engine, self.engine
            )));
        }
        let rank = registration.rank;
        channel.send(&PeerAccepted { rank }).map_err(|error| {
            Error::transport_source("answering a peer's registration", error)
        })?;
        self.peers.insert(rank, channel);
        Ok(Some(rank))
    }

    /// The daemon's request interface.
    pub fn daemon_mut(&mut self) -> &mut Connection {
        &mut self.daemon
    }

    /// One peer's channel, when it is connected.
    pub fn peer_mut(&mut self, rank: u32) -> Option<&mut Channel> {
        self.peers.get_mut(&rank)
    }

    /// The ranks whose peer channel is connected.
    pub fn ranks(&self) -> impl Iterator<Item = u32> + '_ {
        self.peers.keys().copied()
    }

    /// The engine number the daemon started this main process under.
    pub fn engine(&self) -> u64 {
        self.engine
    }

    /// The addresses this node listens on for peers, as bound; empty when it
    /// accepts none.
    pub fn peer_addresses(&self) -> Vec<String> {
        self.listener
            .as_ref()
            .map(|listener| listener.addresses().to_vec())
            .unwrap_or_default()
    }

    /// Closes every connection, letting each peer observe the close.
    pub fn close(&mut self) {
        for channel in self.peers.values_mut() {
            channel.shutdown();
        }
        self.peers.clear();
        self.daemon.close();
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;

    use super::*;

    /// A control thread over a daemon connection that answers nothing, a peer
    /// listener bound at `address`, and this process's engine number.
    fn control(address: &str, engine: u64) -> Control {
        let (ours, _theirs) = UnixStream::pair().expect("a socket pair");
        Control {
            daemon: Connection::from_unix(ours).expect("a connection over the pair"),
            engine,
            listener: Some(Listener::bind(&[address]).expect("a peer listener")),
            peers: BTreeMap::new(),
            waker: Waker::new().expect("a control wake"),
        }
    }

    fn peer(address: &str) -> Channel {
        Channel::connect(address).expect("a peer connection")
    }

    fn register(peer: &mut Channel, rank: u32, engine: u64) {
        peer.send(&PeerRegistration {
            rank,
            engine,
            version: ipc::VERSION.to_owned(),
        })
        .expect("a registration is sent");
    }

    #[test]
    fn a_peer_is_kept_under_the_rank_it_registers() -> Result<()> {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let address = format!("unix:{}", directory.path().join("peers.sock").display());
        let mut control = control(&address, 1);
        let mut peer = peer(&address);
        register(&mut peer, 3, 1);

        let ready = control.wait(Some(Instant::now() + Duration::from_secs(10)))?;
        assert!(ready.contains(&Ready::PeerConnecting), "{ready:?}");
        assert_eq!(control.accept()?, Some(3));
        assert_eq!(control.ranks().collect::<Vec<_>>(), vec![3]);

        peer.wait(Duration::from_secs(10))
            .map_err(|error| Error::transport_source("waiting for the acceptance", error))?;
        let accepted: PeerAccepted = peer
            .try_recv()
            .map_err(|error| Error::transport_source("reading the acceptance", error))?
            .expect("the acceptance arrived");
        assert_eq!(accepted.rank, 3);
        Ok(())
    }

    #[test]
    fn a_peer_of_another_epoch_is_refused() -> Result<()> {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let address = format!("unix:{}", directory.path().join("peers.sock").display());
        let mut control = control(&address, 1);
        let mut peer = peer(&address);
        register(&mut peer, 0, 2);

        control.wait(Some(Instant::now() + Duration::from_secs(10)))?;
        assert!(control.accept().is_err());
        assert_eq!(control.ranks().count(), 0);
        Ok(())
    }

    #[test]
    fn a_wake_ends_a_wait_without_any_message() -> Result<()> {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let address = format!("unix:{}", directory.path().join("peers.sock").display());
        let mut control = control(&address, 1);
        let wake = control.wake();
        wake.wake();
        let ready = control.wait(Some(Instant::now() + Duration::from_secs(10)))?;
        assert!(ready.contains(&Ready::Woken), "{ready:?}");
        Ok(())
    }

    #[test]
    fn a_wait_ends_at_its_deadline_with_nothing_to_serve() -> Result<()> {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let address = format!("unix:{}", directory.path().join("peers.sock").display());
        let mut control = control(&address, 1);
        let started = Instant::now();
        let ready = control.wait(Some(started + Duration::from_millis(50)))?;
        assert_eq!(ready, vec![Ready::Idle]);
        assert!(started.elapsed() >= Duration::from_millis(50));
        Ok(())
    }
}
