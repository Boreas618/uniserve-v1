//! Shoot-down: ending a previous writer before its access passes on.
//!
//! The server sends SIGKILL to the victim's process group (the victim alone
//! when it leads no group), waits for its exit on its pidfd, then waits until
//! NVML no longer lists it on any GPU whose storage it held, which is when
//! its context, and every mapping and queued kernel of it, is gone. A
//! process that has not left a GPU by the deadline is a failed shoot-down,
//! and so is one whose leaving NVML cannot tell: it keeps its access, and the
//! requests that waited for it fail.
//!
//! The same evidence closes the session of an engine that is gone
//! ([`crate::sessions`]): the server follows its processes without killing
//! them ([`ShootDowns::follow`]), and a process is out once it has exited
//! and no GPU lists it. The deadline then runs from its exit.
//!
//! Processes are identified by their pidfd, never by a process id alone, so
//! a reused id is never mistaken for a process that was shot down.

use std::os::fd::{AsFd, OwnedFd};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustix::event::{PollFd, PollFlags, poll};

use platform::{DeviceUuid, Nvml};

struct ShootDown {
    pid: i32,
    pidfd: Arc<OwnedFd>,
    gpus: Vec<[u8; 16]>,
    exited: bool,
    /// When the process was killed, or for a process followed when it was
    /// seen to exit: the deadline runs from then.
    started: Instant,
    /// Whether the server killed the process, or only follows it.
    killed: bool,
}

/// The shoot-downs in progress and their results.
pub(super) struct ShootDowns {
    /// How long a process may take to leave every GPU it held once it exited
    /// (`storage_config.shoot_down_s`).
    deadline: Duration,
    running: Vec<ShootDown>,
    /// Processes that left every GPU; dropped once nothing else refers to
    /// their pidfd.
    finished: Vec<Arc<OwnedFd>>,
    /// Processes that did not leave a GPU in time, or whose leaving NVML could not tell.
    failed: Vec<Arc<OwnedFd>>,
    /// The NVML session the shoot-downs ask whether a GPU lists a process: opened at the first
    /// question, and opened again after a question failed.
    nvml: Option<Nvml>,
}

impl ShootDowns {
    /// No shoot-down, each to come allowed `deadline` to leave its GPUs.
    pub(super) fn new(deadline: Duration) -> Self {
        Self {
            deadline,
            running: Vec::new(),
            finished: Vec::new(),
            failed: Vec::new(),
            nvml: None,
        }
    }

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
        // SAFETY: getpgid has no storage effects.
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
            killed: true,
        });
    }

    /// Follow the process behind `pidfd`, without killing it, until it has
    /// exited and left `gpus`. A process already followed or out is not
    /// followed again.
    pub(super) fn follow(&mut self, pid: i32, pidfd: Arc<OwnedFd>, gpus: Vec<[u8; 16]>) {
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
        self.running.push(ShootDown {
            pid,
            pidfd,
            gpus,
            exited: false,
            started: Instant::now(),
            killed: false,
        });
    }

    /// The processes followed that have not left, with their GPUs, as a
    /// restart in place passes them over.
    pub(super) fn followed(&self) -> Vec<(i32, Arc<OwnedFd>, Vec<[u8; 16]>)> {
        self.running
            .iter()
            .filter(|shot| !shot.killed)
            .map(|shot| (shot.pid, Arc::clone(&shot.pidfd), shot.gpus.clone()))
            .collect()
    }

    /// Whether a process the server killed has not left yet: a restart in
    /// place waits for it.
    pub(super) fn killing(&self) -> bool {
        self.running.iter().any(|shot| shot.killed)
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
    /// or its deadline. A process followed that has not exited waits on its
    /// pidfd.
    pub(super) fn polling(&self) -> bool {
        self.running.iter().any(|shot| shot.killed || shot.exited)
    }

    /// Follow every running shoot-down. Returns whether any ended.
    pub(super) fn advance(&mut self) -> bool {
        let mut ended = false;
        let mut still = Vec::with_capacity(self.running.len());
        for mut shot in std::mem::take(&mut self.running) {
            if !shot.exited {
                let mut fd = [PollFd::new(shot.pidfd.as_ref(), PollFlags::IN)];
                shot.exited = matches!(poll(&mut fd, Some(&rustix::time::Timespec::default())), Ok(count) if count > 0);
                if shot.exited && !shot.killed {
                    shot.started = Instant::now();
                }
            }
            let out = match shot.exited.then(|| self.left_every_gpu(&shot)) {
                None => false,
                Some(Ok(left)) => left,
                Some(Err(error)) => {
                    tracing::error!(
                        pid = shot.pid,
                        %error,
                        "NVML cannot tell whether a process left its GPU; its shoot-down fails"
                    );
                    self.failed.push(shot.pidfd);
                    ended = true;
                    continue;
                }
            };
            if out {
                tracing::info!(
                    pid = shot.pid,
                    elapsed_ms = shot.started.elapsed().as_millis() as u64,
                    killed = shot.killed,
                    "a process is out"
                );
                self.finished.push(shot.pidfd);
                ended = true;
            } else if (shot.killed || shot.exited) && shot.started.elapsed() > self.deadline {
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
    /// The processes shot down that left every GPU.
    pub(super) fn finished_pidfds(&self) -> &[Arc<OwnedFd>] {
        &self.finished
    }

    /// The processes shot down that did not leave a GPU in time.
    pub(super) fn failed_pidfds(&self) -> &[Arc<OwnedFd>] {
        &self.failed
    }

    /// Records, for a server a restart in place resumes, that the process behind
    /// `pidfd` left every GPU.
    pub(super) fn note_finished(&mut self, pidfd: Arc<OwnedFd>) {
        self.finished.push(pidfd);
    }

    /// Records, for a server a restart in place resumes, that the process behind
    /// `pidfd` did not leave a GPU in time.
    pub(super) fn note_failed(&mut self, pidfd: Arc<OwnedFd>) {
        self.failed.push(pidfd);
    }

    pub(super) fn finished(&self, pidfd: &Arc<OwnedFd>) -> bool {
        self.finished.iter().any(|other| Arc::ptr_eq(other, pidfd))
    }

    /// Whether no GPU of `shot` lists its process, through the shoot-downs' NVML session; a
    /// failed question closes the session, so the next opens it again.
    fn left_every_gpu(&mut self, shot: &ShootDown) -> platform::Result<bool> {
        let nvml = match &mut self.nvml {
            Some(nvml) => nvml,
            slot @ None => slot.insert(Nvml::init()?),
        };
        let listed = shot
            .gpus
            .iter()
            .map(|gpu| nvml.lists_process(DeviceUuid(*gpu), shot.pid.unsigned_abs()))
            .collect::<platform::Result<Vec<bool>>>();
        match listed {
            Ok(listed) => Ok(listed.iter().all(|listed| !listed)),
            Err(error) => {
                self.nvml = None;
                Err(error)
            }
        }
    }

    /// Whether the process behind `pidfd` did not leave its GPU in time.
    pub(super) fn failed(&self, pidfd: &Arc<OwnedFd>) -> bool {
        self.failed.iter().any(|other| Arc::ptr_eq(other, pidfd))
    }
}
