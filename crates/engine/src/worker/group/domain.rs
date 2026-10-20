//! Failure domains: the worker groups replaced together after a failure.
//!
//! A failure domain is one worker group, or every worker group that shares
//! an expert world: their workers hold that world's communicators and
//! exchange tokens at every expert layer, so no group can replace its
//! workers alone. A group that is its own domain replaces its workers in
//! `WorkerGroup::recover_workers`; the groups of a shared domain are
//! replaced together here.
//!
//! A failure in any group of a shared domain starts a replacement round:
//! every group is woken, terminates its workers and arrives. Arriving never
//! waits, because several groups may be driven by one scheduler thread. The
//! last group to arrive relaunches every group of the domain on its own
//! thread, at a new rendezvous of the expert world, waits until all of them
//! are ready and posts each group its workers; each group installs them on
//! its next poll. Engine cores fail their in-flight requests on the domain
//! when their group arrives, and admit work for it again once it installed
//! its workers. Weights and exclusive memory come back from the Storage
//! Server.

use std::sync::Mutex;

use uniserve_core::CommandWaker;

use super::spawn::launch_all;
use crate::worker::WorkerProcess;
use crate::worker::WorkerProcessArgs;
use crate::worker::launcher::Launchers;
use crate::worker::registry::Registry;
use crate::worker::storage::StorageClaims;

/// The groups that share an expert world, and their replacement rounds.
pub(super) struct FailureDomain {
    /// Every group of the domain as launched, in domain order, which a
    /// round relaunches.
    arguments: Vec<WorkerProcessArgs>,
    registry: Registry,
    launchers: Option<Launchers>,
    claims: StorageClaims,
    /// Each group's wake, in domain order: a failure and a finished
    /// relaunch wake every group's scheduler.
    wakers: Vec<CommandWaker>,
    round: Mutex<Round>,
}

/// The replacement round in progress.
#[derive(Default)]
struct Round {
    /// The failure that started the round; `None` between rounds.
    cause: Option<String>,
    /// Which groups have terminated their workers and arrived.
    arrived: Vec<bool>,
    /// Each group's relaunched workers, or why the relaunch failed, until the
    /// group takes them.
    replacements: Vec<Option<Result<Vec<WorkerProcess>, String>>>,
}

impl FailureDomain {
    pub(super) fn new(
        arguments: Vec<WorkerProcessArgs>,
        registry: Registry,
        launchers: Option<Launchers>,
        claims: StorageClaims,
        wakers: Vec<CommandWaker>,
    ) -> Self {
        let groups = arguments.len();
        Self {
            arguments,
            registry,
            launchers,
            claims,
            wakers,
            round: Mutex::new(Round {
                cause: None,
                arrived: vec![false; groups],
                replacements: (0..groups).map(|_| None).collect(),
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
            .filter(|_| !round.arrived.get(index).copied().unwrap_or(true))
    }

    /// Whether a round is in progress.
    pub(super) fn failing(&self) -> bool {
        self.round().cause.is_some()
    }

    /// Group `index` terminated its workers. The last group to arrive
    /// relaunches the domain on the calling thread and posts each group's
    /// workers; every other arrival returns at once.
    pub(super) fn arrive(&self, index: usize) {
        let last = {
            let mut round = self.round();
            if let Some(arrived) = round.arrived.get_mut(index) {
                *arrived = true;
            }
            round.arrived.iter().all(|arrived| *arrived)
        };
        if !last {
            return;
        }

        tracing::warn!(
            groups = self.arguments.len(),
            "relaunching the failure domain's worker groups"
        );
        let mut arguments = self.arguments.clone();
        let relaunched = launch_all(
            &mut arguments,
            &self.registry,
            self.launchers.clone(),
            &self.claims,
        );
        {
            let mut round = self.round();
            match relaunched {
                Ok(groups) => {
                    for (slot, (workers, _)) in round.replacements.iter_mut().zip(groups) {
                        *slot = Some(Ok(workers));
                    }
                }
                Err(error) => {
                    let message = format!("{error:#}");
                    for slot in &mut round.replacements {
                        *slot = Some(Err(message.clone()));
                    }
                }
            }
            round.cause = None;
            round
                .arrived
                .iter_mut()
                .for_each(|arrived| *arrived = false);
        }
        self.wake_all();
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
}
