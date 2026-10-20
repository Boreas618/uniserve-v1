//! Shoot-down: ending a previous writer before its access passes on.
//!
//! The server sends SIGKILL to the victim's process group (the victim alone
//! when it leads no group), waits for its exit on its pidfd, then waits until
//! NVML no longer lists it on any GPU whose memory it held, which is when
//! its context, and every mapping and queued kernel of it, is gone. A
//! process that has not left a GPU by the deadline is a failed shoot-down:
//! it keeps its access, and the requests that waited for it fail.
//!
//! Processes are identified by their pidfd, never by a process id alone, so
//! a reused id is never mistaken for a process that was shot down.

use std::os::fd::{AsFd, OwnedFd};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustix::event::{PollFd, PollFlags, poll};

use super::nvml;

/// How long a victim may take to leave every GPU it held.
const DEADLINE: Duration = Duration::from_secs(60);

struct ShootDown {
    pid: i32,
    pidfd: Arc<OwnedFd>,
    gpus: Vec<[u8; 16]>,
    exited: bool,
    started: Instant,
}

/// The shoot-downs in progress and their outcomes.
#[derive(Default)]
pub(super) struct ShootDowns {
    running: Vec<ShootDown>,
    /// Processes that left every GPU; dropped once nothing else refers to
    /// their pidfd.
    finished: Vec<Arc<OwnedFd>>,
    /// Processes that did not leave a GPU in time.
    failed: Vec<Arc<OwnedFd>>,
}

impl ShootDowns {
    /// Kill the process behind `pidfd` and follow it until it left `gpus`.
    /// A process already being shot down, or already out, is not killed
    /// again.
    pub(super) fn start(&mut self, pid: i32, pidfd: Arc<OwnedFd>, gpus: Vec<[u8; 16]>) {
        let known = |list: &[Arc<OwnedFd>]| list.iter().any(|other| Arc::ptr_eq(other, &pidfd));
        if known(&self.finished)
            || known(&self.failed)
            || self
                .running
                .iter()
                .any(|shot| Arc::ptr_eq(&shot.pidfd, &pidfd))
        {
            return;
        }
        // SAFETY: getpgid has no memory effects.
        let group = unsafe { libc::getpgid(pid) };
        if group == pid {
            // A worker leads its own process group, which holds the children
            // that may have inherited its device descriptors.
            // SAFETY: plain signal delivery.
            unsafe { libc::kill(-pid, libc::SIGKILL) };
        } else if let Err(error) =
            rustix::process::pidfd_send_signal(pidfd.as_fd(), rustix::process::Signal::KILL)
            && error != rustix::io::Errno::SRCH
        {
            tracing::warn!(pid, %error, "the storage server could not kill a previous writer");
        }
        tracing::warn!(pid, "shooting down a previous writer");
        self.running.push(ShootDown {
            pid,
            pidfd,
            gpus,
            exited: false,
            started: Instant::now(),
        });
    }

    /// The pidfds of victims that have not exited, which become readable
    /// when they do.
    pub(super) fn pending_pidfds(&self) -> Vec<Arc<OwnedFd>> {
        self.running
            .iter()
            .filter(|shot| !shot.exited)
            .map(|shot| Arc::clone(&shot.pidfd))
            .collect()
    }

    /// Whether a shoot-down waits on something no descriptor signals: NVML,
    /// or its deadline.
    pub(super) fn polling(&self) -> bool {
        !self.running.is_empty()
    }

    /// Follow every running shoot-down. Returns whether any ended.
    pub(super) fn advance(&mut self) -> bool {
        let mut ended = false;
        let mut still = Vec::with_capacity(self.running.len());
        for mut shot in std::mem::take(&mut self.running) {
            if !shot.exited {
                let mut fd = [PollFd::new(shot.pidfd.as_ref(), PollFlags::IN)];
                shot.exited = matches!(poll(&mut fd, Some(&rustix::time::Timespec::default())), Ok(count) if count > 0);
            }
            let out = shot.exited
                && shot.gpus.iter().all(|gpu| match nvml::lists(gpu, shot.pid) {
                    Ok(listed) => !listed,
                    Err(error) => {
                        tracing::warn!(pid = shot.pid, %error, "NVML cannot tell whether a killed writer left its GPU");
                        false
                    }
                });
            if out {
                tracing::info!(
                    pid = shot.pid,
                    elapsed_ms = shot.started.elapsed().as_millis() as u64,
                    "a previous writer is out"
                );
                self.finished.push(shot.pidfd);
                ended = true;
            } else if shot.started.elapsed() > DEADLINE {
                tracing::error!(
                    pid = shot.pid,
                    exited = shot.exited,
                    "a killed writer did not leave its GPU; the GPU needs a reset"
                );
                self.failed.push(shot.pidfd);
                ended = true;
            } else {
                still.push(shot);
            }
        }
        self.running = still;
        // A record nothing else refers to names no holder any more.
        self.finished.retain(|pidfd| Arc::strong_count(pidfd) > 1);
        ended
    }

    /// Whether the process behind `pidfd` was shot down and is out.
    pub(super) fn finished(&self, pidfd: &Arc<OwnedFd>) -> bool {
        self.finished.iter().any(|other| Arc::ptr_eq(other, pidfd))
    }

    /// Whether the process behind `pidfd` did not leave its GPU in time.
    pub(super) fn failed(&self, pidfd: &Arc<OwnedFd>) -> bool {
        self.failed.iter().any(|other| Arc::ptr_eq(other, pidfd))
    }
}
