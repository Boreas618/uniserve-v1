//! The server's side of the KV storage (L3), which its Mooncake connector
//! serves.
//!
//! The server runs the connector ([`crate::connector`]) as a child process
//! and starts it again when it ends. The connector connects as
//! [`Role::Connector`](crate::protocol::Role::Connector); the server sends it
//! every page pool and its tasks as events, and the connector reports each
//! task's results with [`Request::Finished`](crate::protocol::Request).
//!
//! While no connector is connected, L3 holds nothing: a lookup counts L2
//! only, a prefetch fetches nothing, and committed pages are not uploaded.
//! A connector that ends fails its tasks in flight.

use std::collections::{BTreeSet, HashMap};
use std::ffi::OsString;
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use storage::protocol::Key;

/// How long the server waits before starting a connector that ended again,
/// so a connector that cannot reach Mooncake does not restart in a loop.
pub(super) const RESTART_DELAY: Duration = Duration::from_secs(5);

/// How the server starts its connector.
#[derive(Debug, Clone)]
pub struct Launch {
    pub program: PathBuf,
    pub arguments: Vec<OsString>,
}

/// A task the connector runs for the server.
pub(super) enum Task {
    /// The L3 part of a lookup: `keys` follow the lookup's L2 run of `l2`
    /// pages. The lookup's client waits for it.
    Lookup {
        client: u64,
        request: u64,
        pool: u64,
        keys: Vec<Key>,
        l2: u32,
    },
    /// Committed pages being written to Mooncake from their slots, which
    /// stay out of eviction until the task finishes.
    Upload { pool: u64, pages: Vec<(Key, u32)> },
    /// Pages being read from Mooncake into slots reserved for them, for the
    /// prefetch `tag` of `head`, which named `keys`.
    Download {
        head: u64,
        tag: u64,
        pool: u64,
        pages: Vec<(Key, u32)>,
        keys: Vec<Key>,
    },
}

/// The connector process and its connection.
pub(super) struct KvStorage {
    launch: Option<Launch>,
    /// The running process, with its pidfd.
    pub(super) child: Option<(Child, OwnedFd)>,
    /// When to start the process again.
    pub(super) restart: Option<Instant>,
    /// The connected connector's client.
    pub(super) client: Option<u64>,
    pub(super) tasks: HashMap<u64, Task>,
    next: u64,
    /// The pools the connected connector was given.
    pub(super) offered: BTreeSet<u64>,
}

impl KvStorage {
    pub(super) fn new(launch: Option<Launch>) -> Self {
        let restart = launch.as_ref().map(|_| Instant::now());
        Self {
            launch,
            child: None,
            restart,
            client: None,
            tasks: HashMap::new(),
            next: 1,
            offered: BTreeSet::new(),
        }
    }

    /// Start the process when its restart is due.
    pub(super) fn start_due(&mut self) {
        let Some(launch) = &self.launch else {
            return;
        };
        if self.child.is_some() || self.restart.is_none_or(|at| Instant::now() < at) {
            return;
        }
        self.restart = None;
        let started = Command::new(&launch.program)
            .args(&launch.arguments)
            .spawn()
            .and_then(|child| {
                let pid = rustix::process::Pid::from_child(&child);
                let pidfd = rustix::process::pidfd_open(pid, rustix::process::PidfdFlags::empty())?;
                Ok((child, pidfd))
            });
        match started {
            Ok((child, pidfd)) => {
                tracing::info!(pid = child.id(), "started the Mooncake connector");
                self.child = Some((child, pidfd));
            }
            Err(error) => {
                tracing::warn!(%error, "the Mooncake connector could not be started");
                self.restart = Some(Instant::now() + RESTART_DELAY);
            }
        }
    }

    /// Reap the process when it ended, and schedule its restart.
    pub(super) fn reap(&mut self) {
        let Some((child, _)) = &mut self.child else {
            return;
        };
        if let Ok(Some(status)) = child.try_wait() {
            tracing::warn!(%status, "the Mooncake connector ended");
            self.child = None;
            self.restart = Some(Instant::now() + RESTART_DELAY);
        }
    }

    /// How long the server may sleep before a restart is due.
    pub(super) fn timeout(&self) -> Option<Duration> {
        self.restart
            .filter(|_| self.launch.is_some() && self.child.is_none())
            .map(|at| at.saturating_duration_since(Instant::now()))
    }

    /// The identity of a new task.
    pub(super) fn next_task(&mut self) -> u64 {
        let id = self.next;
        self.next += 1;
        id
    }
}
