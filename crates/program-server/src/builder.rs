//! The compile queue (design section 6.2).
//!
//! One module path is one build. A `prepare` submits a [`Urgency::Background`]
//! build for every module it finds missing, and a `resolve` that misses
//! submits a [`Urgency::Waiting`] one; when a build is already queued for the
//! same path, the later request shares it and raises its urgency, so a module
//! somebody is waiting for is never compiled behind one nobody is.
//!
//! The queue is ordered by urgency first and arrival second. A build starts
//! only when two limits allow it: [`Limits::max_jobs`] builds at once, and the
//! host has [`Limits::host_reserve`] bytes of memory left after the build's
//! [`Toolchain::peak_host_bytes`](crate::toolchain::Toolchain::peak_host_bytes)
//! would be spent. A build that cannot start waits in the queue, in its place;
//! the Program Server launches no kernel, so the only resource it rations is
//! host memory.
//!
//! A build's state is `Queued`, `Building`, `Committed` or `Failed` with the
//! build's log; its waiters are told when it ends.

use std::collections::VecDeque;
use std::path::PathBuf;

use crate::store::ModulePath;

/// How many builds may run at once, and how much host memory must stay free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The number of builds that may run at the same time
    /// (`limits.max_jobs`). Every toolchain process also carries this number
    /// as its `MAX_JOBS`, so a build's own compilers are capped too.
    pub max_jobs: usize,
    /// The host memory that must remain available before a build starts
    /// (`limits.host_reserve`), in bytes.
    pub host_reserve: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_jobs: 4,
            host_reserve: 8 << 30,
        }
    }
}

/// Why a build was submitted: whether somebody waits for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Urgency {
    /// Ahead of time: `prepare` wants the module before anybody needs it.
    Background,
    /// Just in time: a `resolve` waits for the module.
    Waiting,
}

/// The state of one build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildState {
    Queued,
    Building,
    Committed,
    Failed(String),
}

/// One build: one module path, its urgency, its state and who waits for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Build {
    pub path: ModulePath,
    pub urgency: Urgency,
    pub state: BuildState,
    /// The waiters' identities, which the server uses to wake a `resolve`.
    pub waiters: Vec<u64>,
}

impl Build {
    /// The directory this build publishes to, as the queue keys it.
    pub fn key(&self) -> PathBuf {
        self.path.relative()
    }
}

/// The queue of builds.
#[derive(Debug, Default)]
pub struct Builder {
    /// Queued builds, by urgency and arrival.
    queued: VecDeque<Build>,
    /// Builds that have started.
    running: Vec<Build>,
    /// Completed builds, kept until their waiters collected the state.
    finished: Vec<Build>,
    limits: Limits,
}

impl Builder {
    pub fn new(limits: Limits) -> Self {
        Self {
            queued: VecDeque::new(),
            running: Vec::new(),
            finished: Vec::new(),
            limits,
        }
    }

    /// Submits a build for a module path, or raises the urgency of the build
    /// already queued for it.
    ///
    /// Returns the identity of the caller's wait, which the build carries
    /// until it ends.
    pub fn submit(&mut self, path: &ModulePath, urgency: Urgency, waiter: u64) -> BuildTicket {
        self.submit_with(path, urgency, waiter, |_| {})
    }

    /// Submits a build, or joins the one already queued for its path,
    /// reporting through `joined` whether it joined an existing build.
    pub fn submit_with(
        &mut self,
        path: &ModulePath,
        urgency: Urgency,
        waiter: u64,
        joined: impl FnOnce(bool),
    ) -> BuildTicket {
        let key = path.relative();
        if let Some(build) = self
            .queued
            .iter_mut()
            .chain(self.running.iter_mut())
            .find(|build| build.key() == key)
        {
            build.waiters.push(waiter);
            if urgency > build.urgency {
                build.urgency = urgency;
                // A raised build moves ahead of the background builds queued
                // before it; builds of one urgency keep their arrival order.
                if let Some(position) = self.queued.iter().position(|queued| queued.key() == key)
                    && let Some(build) = self.queued.remove(position)
                {
                    let ahead = self
                        .queued
                        .iter()
                        .take_while(|queued| queued.urgency <= build.urgency)
                        .count();
                    self.queued.insert(ahead, build);
                }
            }
            joined(true);
        } else {
            // A path built again (a resolve retrying a failed build) replaces
            // what the queue knew about it.
            self.finished.retain(|build| build.key() != key);
            self.queued.push_back(Build {
                path: path.clone(),
                urgency,
                state: BuildState::Queued,
                waiters: vec![waiter],
            });
            joined(false);
        }
        BuildTicket {
            path: path.clone(),
            waiter,
        }
    }

    /// The builds that may start now, given the host's available memory and
    /// the peak each queued build would spend.
    ///
    /// `peak_host_bytes` is the queue's caller's way of asking the toolchain
    /// of each build for its peak (design section 6.2).
    pub fn start(
        &mut self,
        available_host_bytes: u64,
        peak_host_bytes: impl Fn(&Build) -> u64,
    ) -> Vec<ModulePath> {
        let mut started = Vec::new();
        let mut free = available_host_bytes.saturating_sub(self.limits.host_reserve);
        let mut position = 0;
        while position < self.queued.len() && self.running.len() < self.limits.max_jobs {
            let peak = peak_host_bytes(&self.queued[position]);
            if peak <= free {
                let Some(mut build) = self.queued.remove(position) else {
                    break;
                };
                build.state = BuildState::Building;
                free -= peak;
                started.push(build.path.clone());
                self.running.push(build);
            } else {
                // A larger build may still fit in what is left; a build that
                // cannot start now keeps its place in the queue.
                position += 1;
            }
        }
        started
    }

    /// Records that a running build ended, and returns the builds whose
    /// waiters must be told.
    pub fn finish(&mut self, path: &ModulePath, outcome: Result<(), String>) -> Option<Build> {
        let position = self.running.iter().position(|build| &build.path == path)?;
        let mut build = self.running.remove(position);
        build.state = match outcome {
            Ok(()) => BuildState::Committed,
            Err(log) => BuildState::Failed(log),
        };
        self.finished.push(build.clone());
        Some(build)
    }

    /// The state of a build that was submitted, once it is known.
    pub fn state(&self, path: &ModulePath) -> Option<BuildState> {
        self.running
            .iter()
            .chain(self.queued.iter())
            .chain(self.finished.iter())
            .find(|build| &build.path == path)
            .map(|build| build.state.clone())
    }

    /// The number of queued and running builds.
    pub fn pending(&self) -> usize {
        self.queued.len() + self.running.len()
    }
}

/// A submitted build's identity, with which its waiter collects the outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildTicket {
    pub path: ModulePath,
    pub waiter: u64,
}
