//! `Route`: a node's workers and peer links by worker group, as the engine hands them a
//! group's jobs.
//!
//! A scheduler addresses a batch, a completion of the buffer pool and a request to stamp by
//! worker group ([`GroupId`]), never by device. A node's devices are each a rank of one worker
//! group, and a group may have ranks on other hosts, which the engine processes there serve
//! through a peer link ([`Remote`]). This is the one place a group's jobs are handed to its
//! ranks, so a scheduler never needs to know how many devices or processes it drives:
//!
//! - **Every job goes to every rank.** A batch, a shadow batch, a completion to map and requests
//!   to stamp are handed to each of the group's workers on this node and to each peer link that
//!   holds its other ranks, and their submissions are joined into one that completes when every
//!   rank's did ([`Submission::joined`]): a stamp is the largest over the ranks, a copy's
//!   `Finish` failed when it failed on any rank, and a call failed when any rank that ran it
//!   answered it as failed. A rank, or a peer's process, none of whose runners serves a batch's
//!   phase passes its number.
//! - **Numbers.** A batch and a shadow batch read and write the group's device block tables, so
//!   the group numbers them, from 1, in the order it hands them to its ranks. The number is the
//!   group's for every rank; a peer's ranks take the number the scheduler node sent. Each rank
//!   states the numbers it has ended to the group's [`Sequences`], whose smallest over the ranks
//!   is the group's `completed`.
//! - **One order.** A group's jobs are handed under one lock, so every rank's queue holds them in
//!   the same order: the order their collectives need, and the order in which a completion
//!   mapped before a batch is seen by that batch on every rank.
//! - **Generations.** A generation's switch, commit and rollback reach every rank of every group,
//!   this node's workers and once each peer link, under every group's lock: each rank takes them
//!   between the same two jobs of each of its groups, and a peer's far end hands them to its
//!   ranks in the order they arrived. They take no number: they address no device block table.
//! - **A rank alone.** A batch whose rows name a rank ([`crate::batch::MediaRow::rank`], among this
//!   node's workers of the group), such as an audio decode, needs no other rank: the named rank
//!   runs it, and every other rank, here or on a peer, passes its number. A video decode round
//!   names none: every rank of the group that serves its phase runs it and decodes its share of
//!   the round's units (`runners::decoder`). A rank whose runners do not serve the batch's phase
//!   passes it too.
//! - **Host work where its blocks lie.** A batch of a host phase (a media request's reading,
//!   encoding and muxing, [`crate::batch::Variant::host`]) is work of the host, not of a rank's
//!   device. A batch whose rows name no rank runs on one rank of each process that runs it, the
//!   first of the process's workers of the group that serves its phase, as if its rows named
//!   that rank, and every other rank passes its number. A video encode round runs in every
//!   process that holds ranks of the group ([`crate::batch::Variant::dealt`]): each encodes the
//!   units its own ranks decoded, which no other process reaches, into their encoded blocks
//!   there; a process none of whose workers serves the phase passes it. Every other host batch
//!   runs once: here when a worker of this node serves its phase, otherwise on the group's
//!   first peer link, the other links passing it. Its rows then start no copy, as no batch one
//!   rank runs alone does; a host batch with no rows is every rank's and runs the copies.

use std::sync::{Arc, Mutex};

use crate::batch::{Batch, Phase};
use crate::executor::peers::messages::MainProcessRequest;
use storage::runtime::{Completion, GroupId, Request, RequestId};

use super::peers::Remote;
use crate::worker::completion::Submission;
use crate::worker::{Error, Result};
use crate::worker::{Sequences, Worker, alone};
use service::lock;

/// A node's workers and peer links, by worker group, as the engine hands them its jobs.
pub struct Route {
    groups: Vec<Group>,
}

/// One worker group's ranks, as [`Route`] hands them the group's jobs.
struct Group {
    sequences: Arc<Sequences>,
    /// This node's workers of the group: worker `i` reports to `sequences` as rank `i`.
    workers: Vec<Arc<Worker>>,
    /// The links of the processes that hold the group's other ranks: link `j` reports as rank
    /// `workers.len() + j`.
    remotes: Vec<Arc<Remote>>,
    /// The number of the group's last numbered job, held while any job is handed to the ranks:
    /// every rank takes the group's jobs in the order they were handed.
    next: Mutex<u64>,
}

impl Route {
    /// The route over `workers`, this node's, and `remotes`, the links of the processes that
    /// hold other ranks, each with the groups it holds ranks of, for the groups of `sequences`.
    ///
    /// A group's ranks report to its sequences in this order: this node's workers of the group
    /// in the order given, each spawned as its rank there, then the links that hold ranks of it,
    /// in the order given. The workers are shared with the control thread, which reads their
    /// jobs in flight and their counters through the same handles.
    ///
    /// # Errors
    ///
    /// Sequences of one group given twice, or of a group without ranks; sequences that count
    /// other ranks than the group has; a worker of a group without sequences, or spawned as
    /// another rank than its place.
    pub fn new(
        sequences: Vec<Arc<Sequences>>,
        workers: Vec<Arc<Worker>>,
        remotes: Vec<(Vec<GroupId>, Arc<Remote>)>,
    ) -> Result<Self> {
        let mut groups: Vec<Group> = Vec::with_capacity(sequences.len());

        for sequences in sequences {
            let group = sequences.group();
            let ranks: Vec<Arc<Worker>> = workers
                .iter()
                .filter(|worker| worker.group() == group)
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
                    "group {} is routed to {} workers and {} peer links, with sequences of {} \
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
                .find(|(place, worker)| worker.rank() != *place);
            if let Some((place, worker)) = misplaced {
                return Err(Error::Invalid(format!(
                    "the worker in place {place} of group {} was spawned as rank {}",
                    group.0,
                    worker.rank()
                )));
            }

            groups.push(Group {
                sequences,
                workers: ranks,
                remotes: links,
                next: Mutex::new(0),
            });
        }

        if let Some(stray) = workers.iter().find(|worker| {
            !groups
                .iter()
                .any(|route| route.sequences.group() == worker.group())
        }) {
            return Err(Error::Invalid(format!(
                "a worker of group {} is routed without the group's sequences",
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
                    "no worker of this node and no peer link serves worker group {}",
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
    /// as [`Route::submit`] hands a batch.
    ///
    /// # Errors
    ///
    /// As [`Route::submit`].
    pub(crate) fn shadow(&self, batch: Batch) -> Result<Submission> {
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
    ///
    /// A process none of whose workers serves the batch's phase passes it: the group's ranks of
    /// other processes serve it, as the scheduler placed the phase on the group by what all of
    /// its ranks serve (a text encoding on a group whose text encoder lies on another host, a
    /// video encode round whose units no rank here decoded). Host work the scheduler node
    /// handed this process to run once is refused when nothing here serves it.
    pub(super) fn run(
        &self,
        sequence: u64,
        batch: Batch,
        pass: bool,
        shadow: bool,
    ) -> Result<Submission> {
        let group = self.group(batch.group)?;
        let once = group
            .host_work(&batch, pass, shadow)?
            .is_some_and(|(dealt, _)| !dealt);
        let serving = group
            .workers
            .iter()
            .any(|worker| worker.serves(batch.phase));
        let pass = pass || (!serving && !once);
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

        let mut parts = Vec::with_capacity(route.workers.len() + route.remotes.len());
        for (index, remote) in route.remotes.iter().enumerate() {
            let frame = MainProcessRequest::map(group, slot, id, &request, &completion);
            let reporting = (&route.sequences, route.workers.len() + index);
            parts.push(remote.send(frame, Some(reporting))?);
        }
        for worker in &route.workers {
            parts.push(worker.map(slot, id, request.clone(), completion.clone())?);
        }

        joined(parts)
    }

    /// Applies `requests` to the device block tables of every rank of `group` and stamps them:
    /// the submission completes with the requests, each with the group's stamp, the largest over
    /// its ranks of the number of the last job each had launched when it applied them. The
    /// requests are those the workers stamp: `Release`, `Detach`, `Commit`, `Transfer`,
    /// `Replicate`, and the `Finish { ok: false }` that gives a copy up.
    ///
    /// # Errors
    ///
    /// A request the workers do not stamp, a group nothing here serves, or what a queue or a
    /// peer link refuses. Nothing is handed to any rank when a request is refused.
    pub fn stamp(&self, group: GroupId, requests: Vec<Request>) -> Result<Submission> {
        if let Some(request) = requests.iter().find(|request| !stamped(request)) {
            return Err(Error::Invalid(format!(
                "{request:?} is not a request the workers stamp"
            )));
        }

        let route = self.group(group)?;
        let _order = lock(&route.next);

        let mut parts = Vec::with_capacity(route.workers.len() + route.remotes.len());
        for (index, remote) in route.remotes.iter().enumerate() {
            let frame = MainProcessRequest::stamp(group, &requests);
            let reporting = (&route.sequences, route.workers.len() + index);
            parts.push(remote.send(frame, Some(reporting))?);
        }
        for worker in &route.workers {
            parts.push(worker.stamp(requests.clone())?);
        }

        joined(parts)
    }

    /// Switches every rank of every group, this node's workers and the peers', to the prepared
    /// generation `generation`, between the jobs of each group handed before and after it: the
    /// switch is handed under every group's lock, in the route's order of groups, to each of this
    /// node's workers and once to each peer link, whose far end hands it to its ranks in the
    /// order it arrived. It takes no number: it addresses no device block table. The submission
    /// completes once every part did.
    ///
    /// # Errors
    ///
    /// A route of no rank, or what a queue or a peer link refuses.
    pub(crate) fn switch(&self, generation: u64) -> Result<Submission> {
        self.everywhere(
            |worker| worker.switch(generation),
            || MainProcessRequest::Switch {
                message_id: None,
                generation,
            },
        )
    }

    /// Commits the trial of generation `generation` on every rank, behind the jobs handed before
    /// it, as [`Route::switch`] hands a switch.
    ///
    /// # Errors
    ///
    /// As [`Route::switch`].
    pub(crate) fn commit(&self, generation: u64) -> Result<Submission> {
        self.everywhere(
            |worker| worker.commit(generation),
            || MainProcessRequest::Commit {
                message_id: None,
                generation,
            },
        )
    }

    /// Rolls generation `generation` back on every rank, between the jobs handed before and
    /// after it, as [`Route::switch`] hands a switch.
    ///
    /// # Errors
    ///
    /// As [`Route::switch`].
    pub(crate) fn rollback(&self, generation: u64) -> Result<Submission> {
        self.everywhere(
            |worker| worker.rollback(generation),
            || MainProcessRequest::Rollback {
                message_id: None,
                generation,
            },
        )
    }

    /// Hands a job of every rank to this node's workers (`local`) and once to each peer link
    /// (`frame`), under every group's lock, and joins their submissions.
    fn everywhere(
        &self,
        local: impl Fn(&Worker) -> Result<Submission>,
        frame: impl Fn() -> MainProcessRequest,
    ) -> Result<Submission> {
        // Every group's order holds the job at one place: no job of any group is handed while
        // it is handed.
        let _order: Vec<_> = self.groups.iter().map(|group| lock(&group.next)).collect();

        let mut links: Vec<&Arc<Remote>> = Vec::new();
        for group in &self.groups {
            for remote in &group.remotes {
                if !links.iter().any(|link| Arc::ptr_eq(link, remote)) {
                    links.push(remote);
                }
            }
        }

        let mut parts = Vec::new();
        for link in links {
            parts.push(link.send(frame(), None)?);
        }
        for group in &self.groups {
            for worker in &group.workers {
                parts.push(local(worker)?);
            }
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
}

impl Group {
    /// Whether `batch`, handed with `pass` and `shadow`, is host work its rows leave to no rank
    /// (module documentation): `None` for any other job; otherwise whether every process runs
    /// it ([`crate::batch::Variant::dealt`]) and the rank of this node that runs it, the first
    /// of this node's workers of the group that serves its phase, `None` when none here does.
    fn host_work(
        &self,
        batch: &Batch,
        pass: bool,
        shadow: bool,
    ) -> Result<Option<(bool, Option<u32>)>> {
        let variant = batch.shape.variant;
        if pass || shadow || !variant.host() || batch.media.is_empty() || alone(batch)?.is_some() {
            return Ok(None);
        }

        let place = self
            .workers
            .iter()
            .position(|worker| worker.serves(batch.phase));
        let rank = place
            .map(|place| {
                u32::try_from(place)
                    .map_err(|_| Error::Invalid(format!("rank {place} of a group overflows a u32")))
            })
            .transpose()?;
        Ok(Some((variant.dealt(), rank)))
    }

    /// Whether the group's ranks can take `batch`: a rank it names is one of this node's
    /// workers and serves its phase, and a batch that is not passed has a rank to run it.
    fn check(&self, batch: &Batch, pass: bool, shadow: bool) -> Result<()> {
        if pass {
            return Ok(());
        }
        let serving = || self.workers.iter().any(|worker| worker.serves(batch.phase));

        match (alone(batch)?, shadow) {
            (Some(rank), false) => {
                let named = self.workers.get(rank as usize);
                if named.is_some_and(|worker| worker.serves(batch.phase)) {
                    return Ok(());
                }
                Err(Error::Invalid(format!(
                    "batch {} names rank {rank} of group {}, which no worker of this node that \
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
    /// travels while this node's workers prepare their inputs, and the ranks' collectives meet
    /// in the batch's forward.
    fn hand(&self, sequence: u64, batch: Batch, pass: bool, shadow: bool) -> Result<Submission> {
        let named = if pass || shadow { None } else { alone(&batch)? };
        let host = self.host_work(&batch, pass, shadow)?;

        // Which peer links run the batch: none for a batch one rank here runs alone; for host
        // work, every link when each process runs its own part, otherwise the first link and
        // only when no worker here runs it; every link for any other batch.
        let link_runs = |index: usize| match host {
            _ if pass || named.is_some() => false,
            Some((true, _)) => true,
            Some((false, here)) => here.is_none() && index == 0,
            None => true,
        };

        // Host work this node runs is named in its rows as the rank that runs it; the links take
        // the batch as the scheduler formed it and name their own.
        let here = host.and_then(|(_, rank)| rank);
        let renamed = here.map(|rank| {
            let mut own = batch.clone();
            for row in &mut own.media {
                row.rank = Some(rank);
            }
            own
        });
        let own = renamed.as_ref().unwrap_or(&batch);
        let named = named.or(here);

        // The parts whose output counts come first, this node's before the peers': the joined
        // submission's output is its first part's.
        let mut running = Vec::new();
        let mut remote = Vec::new();
        let mut passing = Vec::new();

        for (index, link) in self.remotes.iter().enumerate() {
            let link_pass = !link_runs(index);
            let frame = if shadow {
                MainProcessRequest::shadow(sequence, batch.clone())
            } else {
                MainProcessRequest::submit(sequence, link_pass, batch.clone())
            };
            let reporting = (&self.sequences, self.workers.len() + index);
            let part = link.send(frame, Some(reporting))?;

            if link_pass {
                passing.push(part);
            } else {
                remote.push(part);
            }
        }

        for (index, worker) in self.workers.iter().enumerate() {
            let runs = !pass
                && named.is_none_or(|rank| rank as usize == index)
                && worker.serves(batch.phase);

            match (runs, shadow) {
                (true, true) => running.push(worker.shadow(sequence, own.clone())?),
                (true, false) => running.push(worker.submit(sequence, own.clone())?),
                (false, _) => passing.push(worker.pass(sequence)?),
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
            "a job was handed to no worker and no peer link".to_owned(),
        ));
    }

    let first = parts.remove(0);
    Ok(first.joined(parts))
}

/// Whether the workers stamp `request`: it takes entries away from a device block table or
/// marks one read-only, or it gives a copy up.
fn stamped(request: &Request) -> bool {
    matches!(
        request,
        Request::Release { .. }
            | Request::Detach { .. }
            | Request::Commit { .. }
            | Request::Transfer { .. }
            | Request::Replicate { .. }
            | Request::Finish { ok: false, .. }
    )
}

/// The refusal of a batch whose phase no rank of `group` serves.
///
/// The scheduler places a request's phases on groups by the phases their ranks state; when the
/// two disagree the batch arrives here and nothing can serve it. Both are named, the phase asked
/// for and the phases this node's workers of the group serve, because one figure alone cannot
/// show the disagreement. The refusal is `Invalid`: the batch is not servable here, which is not
/// a broken device.
fn route_refusal(group: &Group, phase: Phase) -> Error {
    let served: Vec<Phase> = group
        .workers
        .iter()
        .flat_map(|worker| worker.phases().iter().copied())
        .collect();

    Error::Invalid(format!(
        "no worker of this node in worker group {} serves phase {phase:?}; its workers of the \
         group serve {served:?}",
        group.sequences.group().0
    ))
}
