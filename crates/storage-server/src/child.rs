//! The server's child processes: the Mooncake connector and the Transfer
//! Server.
//!
//! Each runs in a process of its own so that a fault in it (Mooncake's
//! client, a CUDA context) ends that process and never the server. The
//! server starts it, polls its pidfd, reaps it when it ends and starts it
//! again after [`RESTART_DELAY`]; the child connects back over the server's
//! socket under its own role. What the server does when a child's
//! connection closes (failing the tasks it held) is the owner's business.

use std::ffi::OsString;
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// How long the server waits before starting a child that ended again, so a
/// child that fails at once (it cannot reach Mooncake, a GPU has no memory
/// for its context) does not restart in a loop.
pub(super) const RESTART_DELAY: Duration = Duration::from_secs(5);

/// How the server starts a child.
#[derive(Debug, Clone)]
pub struct Launch {
    pub program: PathBuf,
    pub arguments: Vec<OsString>,
}

/// One child process the server keeps running.
pub(super) struct Supervised {
    /// What the child is, for the log.
    name: &'static str,
    launch: Option<Launch>,
    /// The running process, with its pidfd.
    pub(super) child: Option<(Child, OwnedFd)>,
    /// When to start the process again.
    restart: Option<Instant>,
}

impl Supervised {
    /// A child started by `launch`, due at once; none without a launch.
    pub(super) fn new(name: &'static str, launch: Option<Launch>) -> Self {
        let restart = launch.as_ref().map(|_| Instant::now());
        Self {
            name,
            launch,
            child: None,
            restart,
        }
    }

    /// Whether the server runs this child.
    pub(super) fn configured(&self) -> bool {
        self.launch.is_some()
    }

    /// The running process's id.
    pub(super) fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(|(child, _)| child.id())
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
                tracing::info!(pid = child.id(), "started the {}", self.name);
                self.child = Some((child, pidfd));
            }
            Err(error) => {
                tracing::warn!(%error, "the {} could not be started", self.name);
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
            tracing::warn!(%status, "the {} ended", self.name);
            self.child = None;
            self.restart = Some(Instant::now() + RESTART_DELAY);
        }
    }

    /// End the child and wait for it, as a hot restart does before its
    /// successor starts one of its own; none starts here again.
    pub(super) fn stop(&mut self) {
        self.launch = None;
        self.restart = None;
        if let Some((mut child, _)) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// How long the server may sleep before a restart is due.
    pub(super) fn timeout(&self) -> Option<Duration> {
        self.restart
            .filter(|_| self.launch.is_some() && self.child.is_none())
            .map(|at| at.saturating_duration_since(Instant::now()))
    }
}
