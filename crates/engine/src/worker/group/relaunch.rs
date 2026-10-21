//! Relaunch: the worker groups replaced together after a failure.
//!
//! A relaunch covers one worker group, or every worker group that shares
//! an expert world: their workers hold that world's communicators and
//! exchange tokens at every expert layer, so no group can replace its
//! workers alone. A group that keeps a parked standby promotes it instead
//! (`standby`); every other replacement is a relaunch of the domain here.
//!
//! A failure in any group of a domain starts a replacement round: every
//! group is woken, terminates its workers and arrives with its launch
//! arguments. Arriving never waits, because several groups may be driven by
//! one scheduler thread. The last group to arrive starts the relaunch on a
//! thread of its own, which relaunches every group of the domain (an expert
//! world at a new rendezvous), waits until all of them are ready and posts
//! each group its workers; each group installs them on its next poll. The
//! relaunch takes as long as the ranks need to start, build their models and
//! capture their graphs, and the scheduler threads keep serving their other
//! groups meanwhile. When their group arrives, schedulers end their
//! in-flight token requests on it as migratable, which the daemon submits
//! again and which wait for the relaunched workers, and fail the others;
//! they admit work for the group again once it installed its workers. Weights and exclusive memory come back from the
//! Storage Server.
//!
//! A group that closes during a round cancels its relaunch
//! (`cancel_relaunch`): the relaunched ranks are terminated, whether they
//! were still starting or already posted.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::CommandWaker;

use super::spawn::launch_all;
use crate::worker::WorkerProcess;
use crate::worker::WorkerProcessArgs;
use crate::worker::claims::StorageClaims;
use crate::worker::processes::Processes;
use crate::worker::registry::Registry;

/// The groups replaced together, and their replacement rounds.
pub(super) struct Relaunch {
    registry: Registry,
    processes: Processes,
    claims: StorageClaims,
    /// Each group's wake, in domain order: a failure and a finished
    /// relaunch wake every group's scheduler.
    wakers: Vec<CommandWaker>,
    round: Mutex<Round>,
}

/// The replacement round in progress.
struct Round {
    /// The failure that started the round; `None` between rounds.
    cause: Option<String>,
    /// The launch arguments of each group that has terminated its workers
    /// and arrived; `None` for a group that has not.
    arrived: Vec<Option<WorkerProcessArgs>>,
    /// Each group's relaunched workers, or why the relaunch failed, until the
    /// group takes them.
    replacements: Vec<Option<Result<Vec<WorkerProcess>, String>>>,
    /// The relaunch running, from the last arrival until it posts.
    relaunch: Option<Running>,
    /// Set once a group of the domain closes: no relaunch starts or posts
    /// afterwards.
    closing: bool,
}

/// A relaunch running on its own thread.
struct Running {
    /// The startup cancellation every relaunched rank shares until the
    /// relaunch is ready (`launch_all`).
    cancel: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

impl Relaunch {
    /// A domain of `wakers.len()` groups, which relaunches through the
    /// deployment's `registry`, `processes` and `claims`.
    pub(super) fn new(
        registry: Registry,
        processes: Processes,
        claims: StorageClaims,
        wakers: Vec<CommandWaker>,
    ) -> Self {
        let groups = wakers.len();
        Self {
            registry,
            processes,
            claims,
            wakers,
            round: Mutex::new(Round {
                cause: None,
                arrived: vec![None; groups],
                replacements: (0..groups).map(|_| None).collect(),
                relaunch: None,
                closing: false,
            }),
        }
    }

    fn round(&self) -> std::sync::MutexGuard<'_, Round> {
        self.round
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn wake_all(&self) {
        for waker in &self.wakers {
            waker.wake();
        }
    }

    /// How many groups the domain holds.
    pub(super) fn groups(&self) -> usize {
        self.wakers.len()
    }

    /// Start a replacement round for `cause`, unless one is in progress, and
    /// wake every group.
    pub(super) fn fail(&self, worker: &str, cause: &anyhow::Error) {
        let started = {
            let mut round = self.round();
            if round.cause.is_some() {
                false
            } else {
                round.cause = Some(format!("worker {worker}: {cause:#}"));
                true
            }
        };
        if started {
            self.wake_all();
        }
    }

    /// The failure of the round in progress that group `index` has not yet
    /// arrived for.
    pub(super) fn pending_failure(&self, index: usize) -> Option<String> {
        let round = self.round();
        round
            .cause
            .clone()
            .filter(|_| round.arrived.get(index).is_some_and(Option::is_none))
    }

    /// Whether a round is in progress.
    pub(super) fn failing(&self) -> bool {
        self.round().cause.is_some()
    }

    /// Group `index` terminated its workers and arrives with the arguments
    /// it is relaunched from. The last group to arrive starts the relaunch
    /// on a thread of its own; every arrival returns at once.
    pub(super) fn arrive(self: &Arc<Self>, index: usize, arguments: WorkerProcessArgs) {
        let mut round = self.round();
        if round.closing {
            return;
        }
        if let Some(slot) = round.arrived.get_mut(index) {
            *slot = Some(arguments);
        }
        if round.relaunch.is_some() || !round.arrived.iter().all(Option::is_some) {
            return;
        }

        tracing::warn!(
            groups = round.arrived.len(),
            "relaunching the failure domain's worker groups"
        );
        let arguments = round.arrived.iter().flatten().cloned().collect();
        let cancel = Arc::new(AtomicBool::new(false));
        let domain = Arc::clone(self);
        let startup = Arc::clone(&cancel);
        let spawned = std::thread::Builder::new()
            .name("uniserve-relaunch".into())
            .spawn(move || domain.relaunch(arguments, startup));
        match spawned {
            Ok(thread) => round.relaunch = Some(Running { cancel, thread }),
            Err(error) => {
                let message = format!("starting the relaunch thread: {error}");
                Self::post(&mut round, Err(message));
                drop(round);
                self.wake_all();
            }
        }
    }

    /// The relaunch thread: relaunch every group and post its workers, then
    /// wake every group. A relaunch the domain's closing cancelled posts
    /// nothing and terminates the ranks it started.
    fn relaunch(&self, mut arguments: Vec<WorkerProcessArgs>, cancel: Arc<AtomicBool>) {
        let relaunched = launch_all(
            &mut arguments,
            &self.registry,
            &self.processes,
            &self.claims,
            cancel,
        );
        let mut round = self.round();
        if round.closing {
            drop(round);
            for mut workers in relaunched.into_iter().flatten() {
                workers.iter_mut().for_each(WorkerProcess::terminate);
            }
            if let Ok(mut processes) = crate::worker::processes::lock(&self.processes) {
                for args in &arguments {
                    let _ = processes.stop(&args.worker_id);
                }
            }
            return;
        }
        round.relaunch = None;
        Self::post(&mut round, relaunched.map_err(|error| format!("{error:#}")));
        drop(round);
        self.wake_all();
    }

    /// Posts each group its relaunched workers, or every group the
    /// relaunch's failure, and ends the round.
    fn post(round: &mut Round, relaunched: Result<Vec<Vec<WorkerProcess>>, String>) {
        match relaunched {
            Ok(groups) => {
                for (slot, workers) in round.replacements.iter_mut().zip(groups) {
                    *slot = Some(Ok(workers));
                }
            }
            Err(message) => {
                for slot in &mut round.replacements {
                    *slot = Some(Err(message.clone()));
                }
            }
        }
        round.cause = None;
        round.arrived.iter_mut().for_each(|arrived| *arrived = None);
    }

    /// Group `index`'s relaunched workers, once the round's relaunch
    /// finished.
    pub(super) fn take_replacement(
        &self,
        index: usize,
    ) -> Option<Result<Vec<WorkerProcess>, String>> {
        self.round()
            .replacements
            .get_mut(index)
            .and_then(Option::take)
    }

    /// A group of the domain closes: cancel a running relaunch, wait for its
    /// thread to end, and terminate every relaunched rank no group took. The
    /// cancelled ranks fail their next liveness check, within the startup's
    /// check interval, and are terminated by their launch.
    pub(super) fn cancel_relaunch(&self) {
        let relaunch = {
            let mut round = self.round();
            round.closing = true;
            round.relaunch.take()
        };
        if let Some(relaunch) = relaunch {
            relaunch.cancel.store(true, Ordering::Release);
            if relaunch.thread.join().is_err() {
                tracing::error!("the relaunch thread panicked");
            }
        }
        let posted: Vec<_> = self
            .round()
            .replacements
            .iter_mut()
            .filter_map(Option::take)
            .collect();
        for mut workers in posted.into_iter().flatten() {
            workers.iter_mut().for_each(WorkerProcess::terminate);
        }
    }
}
