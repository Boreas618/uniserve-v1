//! `Executors`: a node's executors and peer links by worker group, as the engine hands them a
//! group's jobs.
//!
//! A scheduler addresses a batch, a completion of the buffer pool and a request to stamp by
//! worker group ([`GroupId`]), never by device. A node's devices are each a rank of one worker
//! group, and a group may have ranks on other hosts, which the engine processes there serve
//! through a peer link ([`Remote`]). This is the one place a group's jobs are handed to its
//! ranks, so a scheduler never needs to know how many devices or processes it drives:
//!
//! - **Every job goes to every rank.** A batch, a shadow batch, a completion to map and requests
//!   to stamp are handed to each of the group's executors on this node and to each peer link that
//!   holds its other ranks, and their submissions are joined into one that completes when every
//!   rank's did ([`Submission::joined`]): a stamp is the largest over the ranks, and a copy's
//!   `Finish` failed when it failed on any rank.
//! - **Numbers.** A batch and a shadow batch read and write the group's device block tables, so
//!   the group numbers them, from 1, in the order it hands them to its ranks. The number is the
//!   group's for every rank; a peer's ranks take the number the scheduler node sent. Each rank
//!   states the numbers it has ended to the group's [`Sequences`], whose smallest over the ranks
//!   is the group's `completed`.
//! - **One order.** A group's jobs are handed under one lock, so every rank's queue holds them in
//!   the same order: the order their collectives need, and the order in which a completion
//!   mapped before a batch is seen by that batch on every rank.
//! - **A rank alone.** A video decode round or its encode is one unit of a timeline and needs no
//!   other rank: the batch names the rank that runs it ([`ipc::batch::MediaRow::rank`], among this
//!   node's executors of the group), and every other rank, here or on a peer, passes its number.
//!   A rank whose runners do not serve the batch's phase passes it too.

use std::sync::{Arc, Mutex};

use ipc::MainProcessRequest;
use ipc::batch::{Batch, Phase};
use storage::runtime::{Completion, GroupId, Request, RequestId};

use super::peers::Remote;
use super::{Executor, Operation, Sequences, alone};
use crate::completion::Submission;
use crate::{Error, Result, lock};

/// A node's executors and peer links, by worker group, as the engine hands them its jobs.
pub struct Executors {
    groups: Vec<Group>,
}

/// One worker group's ranks, as [`Executors`] hands them the group's jobs.
struct Group {
    sequences: Arc<Sequences>,
    /// This node's executors of the group: executor `i` reports to `sequences` as rank `i`.
    executors: Vec<Arc<Executor>>,
    /// The links of the processes that hold the group's other ranks: link `j` reports as rank
    /// `executors.len() + j`.
    remotes: Vec<Arc<Remote>>,
    /// The number of the group's last numbered job, held while any job is handed to the ranks:
    /// every rank takes the group's jobs in the order they were handed.
    next: Mutex<u64>,
}

impl Executors {
    /// The route over `executors`, this node's, and `remotes`, the links of the processes that
    /// hold other ranks, each with the groups it holds ranks of, for the groups of `sequences`.
    ///
    /// A group's ranks report to its sequences in this order: this node's executors of the group
    /// in the order given, each spawned as its rank there, then the links that hold ranks of it,
    /// in the order given. The executors are shared with the control thread, which reads their
    /// jobs in flight and their counters through the same handles.
    ///
    /// # Errors
    ///
    /// Sequences of one group given twice, or of a group without ranks; sequences that count
    /// other ranks than the group has; an executor of a group without sequences, or spawned as
    /// another rank than its place.
    pub fn new(
        sequences: Vec<Arc<Sequences>>,
        executors: Vec<Arc<Executor>>,
        remotes: Vec<(Vec<GroupId>, Arc<Remote>)>,
    ) -> Result<Self> {
        let mut groups: Vec<Group> = Vec::with_capacity(sequences.len());

        for sequences in sequences {
            let group = sequences.group();
            let ranks: Vec<Arc<Executor>> = executors
                .iter()
                .filter(|executor| executor.group() == group)
                .map(Arc::clone)
                .collect();
            let links: Vec<Arc<Remote>> = remotes
                .iter()
                .filter(|(held, _)| held.contains(&group))
                .map(|(_, remote)| Arc::clone(remote))
                .collect();

            let count = ranks.len() + links.len();
            let twice = groups.iter().any(|route| route.sequences.group() == group);
            if twice || count == 0 || sequences.ranks() != count {
                return Err(Error::Invalid(format!(
                    "group {} is routed to {} executors and {} peer links, with sequences of {} \
                     ranks, or more than once",
                    group.0,
                    ranks.len(),
                    links.len(),
                    sequences.ranks()
                )));
            }

            let misplaced = ranks
                .iter()
                .enumerate()
                .find(|(place, executor)| executor.rank() != *place);
            if let Some((place, executor)) = misplaced {
                return Err(Error::Invalid(format!(
                    "the executor in place {place} of group {} was spawned as rank {}",
                    group.0,
                    executor.rank()
                )));
            }

            groups.push(Group {
                sequences,
                executors: ranks,
                remotes: links,
                next: Mutex::new(0),
            });
        }

        if let Some(stray) = executors.iter().find(|executor| {
            !groups
                .iter()
                .any(|route| route.sequences.group() == executor.group())
        }) {
            return Err(Error::Invalid(format!(
                "an executor of group {} is routed without the group's sequences",
                stray.group().0
            )));
        }

        Ok(Self { groups })
    }

    /// The route of `group`.
    fn group(&self, group: GroupId) -> Result<&Group> {
        self.groups
            .iter()
            .find(|route| route.sequences.group() == group)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "no executor of this node and no peer link serves worker group {}",
                    group.0
                ))
            })
    }

    /// Hands `batch` to every rank of its group under the group's next number: the ranks that
    /// run it run it, the others pass its number. The submission completes with the batch's
    /// output and the `Finish` of each copy that ran with it, stamped by the group. A batch that
    /// a rank of this node fails after copies started with it fails with their
    /// `Finish { ok: false }` ([`Error::finished`]).
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when nothing here serves the batch's group and phase, or a rank it
    /// names; what a queue or a peer link refuses.
    pub fn submit(&self, batch: Batch) -> Result<Submission> {
        self.numbered(batch, false)
    }

    /// Hands the shadow batch `batch` to every rank of its group under the group's next number,
    /// as [`Executors::submit`] hands a batch.
    ///
    /// # Errors
    ///
    /// As [`Executors::submit`].
    pub fn shadow(&self, batch: Batch) -> Result<Submission> {
        self.numbered(batch, true)
    }

    /// Numbers `batch` for its group and hands it to the group's ranks, as a shadow batch when
    /// `shadow` is set.
    fn numbered(&self, batch: Batch, shadow: bool) -> Result<Submission> {
        let group = self.group(batch.group)?;
        let mut next = lock(&group.next);

        // A batch refused before any rank took it takes no number.
        group.check(&batch, false, shadow)?;
        *next += 1;
        group.hand(*next, batch, false, shadow)
    }

    /// Hands `batch`, which the scheduler node numbered `sequence`, to this process's ranks of
    /// its group, as a peer link's far end does: each runs it, or only takes its number when
    /// `pass` is set or its runners do not serve the batch's phase.
    pub(super) fn run(
        &self,
        sequence: u64,
        batch: Batch,
        pass: bool,
        shadow: bool,
    ) -> Result<Submission> {
        let group = self.group(batch.group)?;
        let mut next = lock(&group.next);

        group.check(&batch, pass, shadow)?;
        *next = (*next).max(sequence);
        group.hand(sequence, batch, pass, shadow)
    }

    /// Maps `completion`, the buffer pool's answer to `request` numbered `id`, into the device
    /// block tables of every rank of `group`, under `slot`: the slot row of the request's owner on
    /// the group, or for a group that reads a class it does not write, the slot of the request's
    /// own owner there. Every job handed to the group after it sees its entries. The submission
    /// fails when a rank's tables refuse the pair.
    ///
    /// # Errors
    ///
    /// A group nothing here serves, or what a queue or a peer link refuses.
    pub fn map(
        &self,
        group: GroupId,
        slot: u32,
        id: RequestId,
        request: Request,
        completion: Completion,
    ) -> Result<Submission> {
        let route = self.group(group)?;
        let _order = lock(&route.next);

        let mut parts = Vec::with_capacity(route.executors.len() + route.remotes.len());
        for (index, remote) in route.remotes.iter().enumerate() {
            let frame = MainProcessRequest::map(group, slot, id, &request, &completion);
            parts.push(remote.send(frame, &route.sequences, route.executors.len() + index)?);
        }
        for executor in &route.executors {
            parts.push(executor.map(slot, id, request.clone(), completion.clone())?);
        }

        joined(parts)
    }

    /// Applies `requests` to the device block tables of every rank of `group` and stamps them:
    /// the submission completes with the requests, each with the group's stamp, the largest over
    /// its ranks of the number of the last job each had launched when it applied them. The
    /// requests are those the executors stamp: `Release`, `Detach`, `Commit`, `Transfer`, and the
    /// `Finish { ok: false }` that gives a copy up.
    ///
    /// # Errors
    ///
    /// A request the executors do not stamp, a group nothing here serves, or what a queue or a
    /// peer link refuses. Nothing is handed to any rank when a request is refused.
    pub fn stamp(&self, group: GroupId, requests: Vec<Request>) -> Result<Submission> {
        if let Some(request) = requests.iter().find(|request| !stamped(request)) {
            return Err(Error::Invalid(format!(
                "{request:?} is not a request the executors stamp"
            )));
        }

        let route = self.group(group)?;
        let _order = lock(&route.next);

        let mut parts = Vec::with_capacity(route.executors.len() + route.remotes.len());
        for (index, remote) in route.remotes.iter().enumerate() {
            let frame = MainProcessRequest::stamp(group, &requests);
            parts.push(remote.send(frame, &route.sequences, route.executors.len() + index)?);
        }
        for executor in &route.executors {
            parts.push(executor.stamp(requests.clone())?);
        }

        joined(parts)
    }

    /// The `completed` of `group` as this node knows it: the smallest ended sequence over the
    /// group's ranks here and on peers.
    ///
    /// # Errors
    ///
    /// A group nothing here serves.
    pub fn completed(&self, group: GroupId) -> Result<u64> {
        Ok(self.group(group)?.sequences.completed())
    }

    /// Submits one control operation to every executor of this node, as one submission:
    /// `operation` makes each executor's from its place among the node's executors, group by
    /// group in the order the route was given them.
    ///
    /// # Errors
    ///
    /// When the node holds no executor, or a queue is closed.
    pub fn control(
        &self,
        label: &str,
        operation: impl Fn(usize) -> Operation,
    ) -> Result<Submission> {
        let parts = self
            .groups
            .iter()
            .flat_map(|group| &group.executors)
            .enumerate()
            .map(|(place, executor)| executor.control(label, operation(place)))
            .collect::<Result<Vec<_>>>()?;

        joined(parts)
    }
}

impl Group {
    /// Whether the group's ranks can take `batch`: a rank it names is one of this node's
    /// executors and serves its phase, and a batch that is not passed has a rank to run it.
    fn check(&self, batch: &Batch, pass: bool, shadow: bool) -> Result<()> {
        if pass {
            return Ok(());
        }
        let serving = || {
            self.executors
                .iter()
                .any(|executor| executor.serves(batch.phase))
        };

        match (alone(batch)?, shadow) {
            (Some(rank), false) => {
                let named = self.executors.get(rank as usize);
                if named.is_some_and(|executor| executor.serves(batch.phase)) {
                    return Ok(());
                }
                Err(Error::Invalid(format!(
                    "batch {} names rank {rank} of group {}, which no executor of this node that \
                     serves phase {:?} is",
                    batch.id.0,
                    self.sequences.group().0,
                    batch.phase
                )))
            }
            _ if serving() || !self.remotes.is_empty() => Ok(()),
            _ => Err(route_refusal(self, batch.phase)),
        }
    }

    /// Hands `batch`, numbered `sequence`, to every rank of the group, which `check` accepted:
    /// the ranks that run it run it, and the others pass its number. The peers take it first: it
    /// travels while this node's executors prepare their inputs, and the ranks' collectives meet
    /// in the batch's forward.
    fn hand(&self, sequence: u64, batch: Batch, pass: bool, shadow: bool) -> Result<Submission> {
        let named = if pass || shadow { None } else { alone(&batch)? };
        let peers_pass = pass || named.is_some();

        // The parts whose output counts come first, this node's before the peers': the joined
        // submission's output is its first part's.
        let mut running = Vec::new();
        let mut remote = Vec::new();
        let mut passing = Vec::new();

        for (index, link) in self.remotes.iter().enumerate() {
            let frame = if shadow {
                MainProcessRequest::shadow(sequence, batch.clone())
            } else {
                MainProcessRequest::submit(sequence, peers_pass, batch.clone())
            };
            let part = link.send(frame, &self.sequences, self.executors.len() + index)?;

            if peers_pass {
                passing.push(part);
            } else {
                remote.push(part);
            }
        }

        for (index, executor) in self.executors.iter().enumerate() {
            let runs = !pass
                && named.is_none_or(|rank| rank as usize == index)
                && executor.serves(batch.phase);

            match (runs, shadow) {
                (true, true) => running.push(executor.shadow(sequence, batch.clone())?),
                (true, false) => running.push(executor.submit(sequence, batch.clone())?),
                (false, _) => passing.push(executor.pass(sequence)?),
            }
        }

        running.extend(remote);
        running.extend(passing);
        joined(running)
    }
}

/// `parts`, the same job's submissions on the ranks of a group, as one submission: the first
/// part's outcome, unless another failed, with the requests of every part joined.
fn joined(mut parts: Vec<Submission>) -> Result<Submission> {
    if parts.is_empty() {
        return Err(Error::Invalid(
            "a job was handed to no executor and no peer link".to_owned(),
        ));
    }

    let first = parts.remove(0);
    Ok(first.joined(parts))
}

/// Whether the executors stamp `request`: it takes entries away from a device block table or
/// marks one read-only, or it gives a copy up.
fn stamped(request: &Request) -> bool {
    matches!(
        request,
        Request::Release { .. }
            | Request::Detach { .. }
            | Request::Commit { .. }
            | Request::Transfer { .. }
            | Request::Finish { ok: false, .. }
    )
}

/// The refusal of a batch whose phase no rank of `group` serves.
///
/// The scheduler places a request's phases on groups by the phases their ranks state; when the
/// two disagree the batch arrives here and nothing can serve it. Both are named, the phase asked
/// for and the phases this node's executors of the group serve, because one figure alone cannot
/// show the disagreement. The refusal is `Invalid`: the batch is not servable here, which is not
/// a broken device.
fn route_refusal(group: &Group, phase: Phase) -> Error {
    let served: Vec<Phase> = group
        .executors
        .iter()
        .flat_map(|executor| executor.phases().iter().copied())
        .collect();

    Error::Invalid(format!(
        "no executor of this node in worker group {} serves phase {phase:?}; its executors of the \
         group serve {served:?}",
        group.sequences.group().0
    ))
}
