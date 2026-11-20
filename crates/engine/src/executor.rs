//! A scheduler's executor: the [`Executor`] the scheduler hands its batches, completions and
//! stamps to, and [`WorkerExecutor`], the executor over the node's workers and peers.
//!
//! A scheduler hands its executor a batch to run, a completion of the buffer pool to map into a
//! group's device block tables, and requests for a group to stamp (`specs/scheduler-design.md`
//! §2.1, `Executor`). [`WorkerExecutor`] hands each to the node's [`Route`], which numbers the
//! jobs of the worker group they name and gives them to every rank of the group, the workers of
//! this node and the peers that hold its other ranks ([`peers`]), in the order they were handed,
//! and it returns what they produce to the scheduler as [`Message`]s on its one channel:
//!
//! - a batch's output arrives exactly once, as `Message::Output`: the runner's output, or the
//!   error of a batch the workers refused or failed;
//! - the `Finish` of each copy that ran with a batch arrives just before the batch's output, as
//!   one `Message::Submit`, each `Finish` stamped by the group with that batch. A batch that
//!   failed after copies had started with it carries a `Finish { ok: false }` for each
//!   (`worker::Error::finished`), sent the same way, so the scheduler learns that those copies
//!   ended instead of giving them up at its bound;
//! - the requests of a stamp arrive as one `Message::Submit` once every rank of the group has
//!   applied them to its device block tables, each with the group's stamp;
//! - a map sends nothing. A map or a stamp that fails leaves the group's device block tables
//!   without what the scheduler counts on: its failure arrives as `Message::Failed`, in its place
//!   among the group's messages, and the scheduler fails at once (SD §3.7);
//! - what a worker of the group measured of an expert union's step it joined without tokens of
//!   its own, which another member's worker asked for and no batch of the scheduler carries,
//!   arrives as `Message::Joined` once the join ended (`worker::Worker::joins`).
//!
//! ## Threads and state
//!
//! The scheduler's thread calls `submit`, `map` and `stamp`, which hand the job to the node's
//! route and never wait for it. Each worker group of the scheduler has one waiter thread, which
//! takes the group's submissions in the order they were handed and blocks on each in turn until
//! it completes. The workers complete a group's jobs in that order, so a waiter never waits
//! behind a job that ended after a later one, and since it is the only thread that sends its
//! group's messages, they arrive in the order the group's jobs were done. The same waiter takes
//! the joins of the group's workers on this node as they are given, in between: a join's
//! statistics bind no order of the group's messages. Nothing runs on a
//! worker's thread: a waker would run there, and may only signal.
//!
//! [`WorkerExecutor`] owns the waiters: dropping it closes their channels and joins them once the
//! submissions they hold have completed. The node's route is shared with the control thread and
//! the node's other schedulers, whose groups are other groups.
//!
//! ## Shadow batches
//!
//! A generation update compares each kernel it replaces with the serving one on the batches the
//! scheduler serves (`worker::Worker::shadow`). The scheduler does not know of it: the engine's
//! control thread opens a pass of the prepared generation on a [`Shadow`], and while the pass is
//! open `submit` also hands the route each batch whose group and shape the pass has not shadowed
//! yet as a shadow batch of that generation, right after the batch itself. The pass answers once
//! every shadow batch it started has ended.

pub mod peers;
pub(crate) mod route;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};
use service::lock;
use storage::runtime::{self, Completion, GroupId};

pub use self::route::Route;
use crate::batch::{Batch, BatchOutput, Shape};
use crate::scheduler::{Error, Message};
use crate::worker::completion::{Outcome, Submission};

/// The executor of a scheduler's worker groups, as the scheduler drives it (BP §9 and §17, A1
/// to A4).
///
/// - `submit`: the batch's output arrives exactly once as [`Message::Output`].
/// - `map`: a completion that gives entries (`Attached`, `Allocated`, `Pinned`, `Transferred`,
///   `Copy`, and the answers to a copy's `Finish`: `Loaded`, `Cloned`, `Duplicate`), with the
///   request it answers and that request's number, is written into `group`'s device block tables
///   at the positions the request names, under `slot` (the slot row of the request's owner on
///   that group; for a group that reads a class it does not write, the slot of the request's own
///   owner there), or its copies are queued, before the group's next batch launches. The entries
///   of a `Load` or a `Clone` appear with the answer to its `Finish`, read-only for a `Load` and
///   writable for a `Clone`. The executor writes no entry from any other source. A queued copy
///   runs with the group's next batch, and its `Finish { copy, ok, after }`, with the group's
///   stamp, returns in [`Message::Submit`].
/// - `stamp`: `Release`, `Detach`, `Commit` and `Transfer`, and the `Finish { ok: false }` that
///   gives a copy up, are applied to `group`'s device block tables (the entries a `Release`, a
///   `Detach` or a `Transfer` names are removed, the entry a `Commit` names is marked read-only; a
///   given-up copy joins no further batch and is dropped if it has not started), stamped with
///   the sequence of the last batch the group launched in their `after`, and every one of them
///   returns in [`Message::Submit`]. A given-up copy that has started is covered by that stamp
///   and gets no `Finish` of its own.
/// - A batch writes only entries that came from `Allocated` or `Cloned` and that no `Commit` has
///   marked, and a copy's destination is written only by that copy.
pub trait Executor: Send {
    /// Runs a batch on its group.
    fn submit(&self, batch: Batch);

    /// Writes the entries a completion gives into `group`'s device block tables, or queues its
    /// copies.
    fn map(
        &self,
        group: GroupId,
        slot: u32,
        id: runtime::RequestId,
        request: runtime::Request,
        completion: Completion,
    );

    /// Applies requests to `group`'s device block tables and returns them stamped.
    fn stamp(&self, group: GroupId, requests: Vec<runtime::Request>);
}

/// The executor of one scheduler's worker groups over the node's workers and peers (module
/// documentation).
pub(crate) struct WorkerExecutor {
    /// The route of each worker group's jobs to its ranks: shared with the control thread and
    /// the node's other schedulers.
    route: Arc<Route>,
    /// Per worker group of the scheduler, in the order given: the group and the channel of its
    /// waiter.
    waiters: Vec<(GroupId, Sender<Wait>)>,
    /// The waiters' threads, joined when the executor is dropped.
    threads: Vec<JoinHandle<()>>,
    /// The pass the engine opens over this scheduler's batches while it prepares a generation.
    shadow: Arc<Mutex<Shadow>>,
}

/// One submission a group's waiter waits for, in the order the job was handed to the group's
/// ranks, with what the scheduler is sent once it completes. A job the route refused carries the
/// refusal in place of its submission.
#[expect(
    clippy::large_enum_variant,
    reason = "a batch travels by value once per batch, to its `Message::Output`"
)]
enum Wait {
    /// A batch: the `Finish` of each copy that ran with it goes to the scheduler as
    /// `Message::Submit`, then its output as `Message::Output`.
    Batch(Batch, crate::worker::Result<Submission>),
    /// A shadow batch of the pass numbered here: its outcome goes to the pass.
    Shadow(u64, crate::worker::Result<Submission>),
    /// The map of the completion of the request numbered here: nothing is sent.
    Map(runtime::RequestId, crate::worker::Result<Submission>),
    /// A stamp: its requests go to the scheduler as `Message::Submit`.
    Stamp(crate::worker::Result<Submission>),
}

impl WorkerExecutor {
    /// The executor of a scheduler whose worker groups are `groups`, over `route`, the node's,
    /// with one waiter thread per group, which sends what the group's ranks return to the
    /// scheduler's channel `inbox`. `shadow` is the pass the engine opens over this scheduler's
    /// batches; the engine keeps a clone of it.
    ///
    /// A job of a group that is not one of `groups` is refused without reaching the route: it
    /// holds other schedulers' groups too.
    ///
    /// # Errors
    ///
    /// [`Error::Executor`] for no group, a group given twice, or a waiter thread that cannot
    /// start.
    pub(crate) fn new(
        route: Arc<Route>,
        groups: &[GroupId],
        inbox: Sender<Message>,
        shadow: Arc<Mutex<Shadow>>,
    ) -> Result<Self, Error> {
        let twice = groups
            .iter()
            .enumerate()
            .any(|(index, group)| groups[..index].contains(group));
        if groups.is_empty() || twice {
            return Err(Error::Executor(format!(
                "a scheduler's executor is given the worker groups {groups:?}: none, or one twice"
            )));
        }

        // Built as the waiters start, so a thread that cannot start leaves the ones before it to
        // be joined by `Drop`.
        let mut built = Self {
            route,
            waiters: Vec::with_capacity(groups.len()),
            threads: Vec::with_capacity(groups.len()),
            shadow,
        };
        for &group in groups {
            let (waits, waiting) = crossbeam_channel::unbounded();
            let joins = built.route.joins(group);
            let inbox = inbox.clone();
            let shadow = Arc::clone(&built.shadow);

            let thread = std::thread::Builder::new()
                .name(format!("group-{}", group.0))
                .spawn(move || wait(group, (&waiting, &joins), &inbox, &shadow))
                .map_err(|error| {
                    Error::Executor(format!(
                        "the waiter of worker group {} did not start: {error}",
                        group.0
                    ))
                })?;
            built.waiters.push((group, waits));
            built.threads.push(thread);
        }
        Ok(built)
    }

    /// Hands a job of `group` to the route through `hand`, unless `group` is not one of the
    /// scheduler's.
    fn hand(
        &self,
        group: GroupId,
        hand: impl FnOnce(&Route) -> crate::worker::Result<Submission>,
    ) -> crate::worker::Result<Submission> {
        if self.waiters.iter().any(|(served, _)| *served == group) {
            return hand(&self.route);
        }

        Err(crate::worker::Error::Invalid(format!(
            "worker group {} is not one of this scheduler's",
            group.0
        )))
    }

    /// Gives `wait` to the waiter of `group`. A job of a group that is not the scheduler's was
    /// refused, and its refusal reaches the scheduler through the first group's waiter, since
    /// no group's order binds it.
    fn send(&self, group: GroupId, wait: Wait) {
        let waiter = self
            .waiters
            .iter()
            .find(|(served, _)| *served == group)
            .or_else(|| self.waiters.first());

        // A waiter ends only once its channel is closed, which `Drop` alone does: the send
        // fails only for a waiter that ended abnormally, whose own end is reported there.
        if let Some((_, waits)) = waiter
            && waits.send(wait).is_err()
        {
            tracing::error!(group = group.0, "a worker group's waiter has ended");
        }
    }
}

impl Executor for WorkerExecutor {
    /// Hands `batch` to the ranks of its group, and while a shadow pass is open, a shadow batch
    /// of it right after when the pass has not shadowed its group and shape. The batch is kept
    /// for its `Message::Output`; the ranks take a copy.
    fn submit(&self, batch: Batch) {
        let group = batch.group;
        let submission = self.hand(group, |route| route.submit(batch.clone()));

        let shadow = match &submission {
            Ok(_) => lock(&self.shadow).shadow(&batch),
            Err(_) => None,
        };
        self.send(group, Wait::Batch(batch, submission));

        if let Some((pass, shadow)) = shadow {
            let submission = self.hand(group, |route| route.shadow(shadow));
            self.send(group, Wait::Shadow(pass, submission));
        }
    }

    fn map(
        &self,
        group: GroupId,
        slot: u32,
        id: runtime::RequestId,
        request: runtime::Request,
        completion: Completion,
    ) {
        let submission = self.hand(group, |route| {
            route.map(group, slot, id, request, completion)
        });
        self.send(group, Wait::Map(id, submission));
    }

    fn stamp(&self, group: GroupId, requests: Vec<runtime::Request>) {
        if requests.is_empty() {
            return;
        }

        let submission = self.hand(group, |route| route.stamp(group, requests));
        self.send(group, Wait::Stamp(submission));
    }
}

impl Drop for WorkerExecutor {
    fn drop(&mut self) {
        // A waiter ends once its channel is closed and the submissions in it have completed.
        self.waiters.clear();

        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                tracing::error!("a worker group's waiter panicked");
            }
        }
    }
}

/// The waiter of worker group `group`: waits for each of the group's submissions in the order
/// they were handed and sends the scheduler what each returns ([`Wait`]), until its channel is
/// closed and empty. A map or a stamp that failed is sent as `Message::Failed`, and the
/// scheduler stops on it. In between it takes the joins of `joins`, the readers of the group's
/// workers on this node ([`crate::worker::Worker::joins`]), and sends what each measured as
/// `Message::Joined` once it ended.
fn wait(
    group: GroupId,
    (waits, joins): (&Receiver<Wait>, &[Receiver<Submission>]),
    inbox: &Sender<Message>,
    shadow: &Mutex<Shadow>,
) {
    // Index 0 is the group's own jobs; the joins' readers follow, each dropped from the
    // selection once its worker is gone.
    let mut select = crossbeam_channel::Select::new();
    select.recv(waits);
    for join in joins {
        select.recv(join);
    }

    // A scheduler that stopped reads nothing more: what it is sent then is dropped.
    loop {
        let ready = select.select();
        let index = ready.index();
        if index > 0 {
            match ready.recv(&joins[index - 1]) {
                Ok(submission) => joined(group, submission, inbox),
                Err(_) => select.remove(index),
            }
            continue;
        }
        let Ok(wait) = ready.recv(waits) else {
            break;
        };

        match wait {
            Wait::Batch(batch, submission) => {
                // A failed batch's `Finish`es are those of the copies that had started with it.
                let (output, finished) = match returned(submission) {
                    Ok((output, finished)) => (Ok(output), finished),
                    Err((failure, finished)) => (Err(failure), finished),
                };

                if !finished.is_empty() {
                    let _ = inbox.send(Message::Submit(finished));
                }
                let _ = inbox.send(Message::Output(batch, output.map_err(Error::Executor)));
            }

            Wait::Shadow(pass, submission) => {
                let outcome = returned(submission)
                    .map(|_| ())
                    .map_err(|(failure, _)| failure);
                lock(shadow).ended(pass, outcome);
            }

            Wait::Map(id, submission) => {
                if let Err((failure, _)) = returned(submission) {
                    let failure = format!(
                        "the completion of request {} was not mapped into worker group {}'s \
                         device block tables: {failure}",
                        id.0, group.0
                    );
                    tracing::error!(group = group.0, %failure, "a map failed");
                    let _ = inbox.send(Message::Failed(Error::Executor(failure)));
                }
            }

            Wait::Stamp(submission) => match returned(submission) {
                Ok((_, stamped)) => {
                    let _ = inbox.send(Message::Submit(stamped));
                }
                Err((failure, _)) => {
                    let failure = format!(
                        "worker group {} did not apply and stamp requests: {failure}",
                        group.0
                    );
                    tracing::error!(group = group.0, %failure, "a stamp failed");
                    let _ = inbox.send(Message::Failed(Error::Executor(failure)));
                }
            },
        }
    }
}

/// Sends the scheduler what the join of `submission`, by a worker of `group`, measured once it
/// ended. A failed join reaches the scheduler no other way than the failure of the steps it
/// left (the member that asked for it waits on it in its exchange), so it is logged.
fn joined(group: GroupId, submission: Submission, inbox: &Sender<Message>) {
    match returned(Ok(submission)) {
        Ok((output, _)) => {
            if let Some(stats) = output.forward_stats.filter(|stats| !stats.is_empty()) {
                let _ = inbox.send(Message::Joined(stats));
            }
        }
        Err((failure, _)) => {
            tracing::error!(group = group.0, %failure, "an expert join failed");
        }
    }
}

/// What a submission returned once it completed: a batch's output and the requests the group's
/// workers hand back for the submission queue; or the failure in words, the refusal of a job the
/// route did not take included, with the `Finish { ok: false }` of each copy that had started
/// with a failed batch, which the group's workers applied and stamped
/// (`worker::Error::finished`; none for any other failure).
fn returned(
    submission: crate::worker::Result<Submission>,
) -> Result<(BatchOutput, Vec<runtime::Request>), (String, Vec<runtime::Request>)> {
    match submission.map(|submission| submission.wait()) {
        Ok(Outcome::Success(returned)) => Ok(returned),
        Ok(Outcome::Failed(fault)) => {
            let finished = fault
                .downcast_ref::<crate::worker::Error>()
                .map_or_else(Vec::new, |error| error.finished().to_vec());
            Err((fault.to_string(), finished))
        }
        Ok(Outcome::Cancelled) => Err((
            "the workers closed before they ran the job".to_owned(),
            Vec::new(),
        )),
        Err(refused) => Err((refused.to_string(), refused.finished().to_vec())),
    }
}

/// The shadow pass of a generation update over one scheduler's batches (module documentation),
/// shared behind a lock by the engine's control thread, which opens and ends it, the scheduler's
/// executor, which shadows batches while it is open, and the group waiters, which report how
/// each shadow batch ended.
///
/// A pass shadows the batches handed while it is open, each group and shape once, and closes
/// once its shadow batches have all ended. A shadow batch ends only after the batch it shadows
/// and its own paired forwards, so the batches the scheduler hands one after another as it
/// forms them are shadowed together.
#[derive(Default)]
pub struct Shadow {
    /// The number of the last pass opened, which is the open pass when one is: each shadow batch
    /// names its pass by it.
    passes: u64,
    /// The open pass: the generation it shadows, and where its outcome goes. `None` while no
    /// pass is open.
    pass: Option<(u64, Sender<Result<(), Error>>)>,
    /// The group and shape of each batch the open pass shadowed.
    shadowed: HashSet<(GroupId, Shape)>,
    /// The open pass's shadow batches that have not ended.
    in_flight: usize,
    /// The open pass's first failure.
    failure: Option<String>,
}

impl Shadow {
    /// Opens a pass of `generation`, which every rank of the scheduler's groups has prepared
    /// beside the serving one (`worker::Worker::prepare`). From the next batch on, each batch
    /// whose group and shape the pass has not shadowed is also run as a shadow batch of
    /// `generation`. Once every shadow batch the pass started has ended, and at least one has,
    /// the pass closes and `reply` receives `Ok(())`, or [`Error::Executor`] with its first
    /// failure: a pair of kernels beyond its tolerance, or a shadow batch the ranks refused
    /// or failed. A pass still open is ended first, unanswered.
    pub fn open(&mut self, generation: u64, reply: Sender<Result<(), Error>>) {
        self.passes += 1;
        self.pass = Some((generation, reply));
        self.shadowed.clear();
        self.in_flight = 0;
        self.failure = None;
    }

    /// Ends the open pass, if any, unanswered: its reply's receiver finds the channel
    /// disconnected. No further batch is shadowed; the shadow batches in flight still run, and
    /// how they end is dropped.
    pub fn end(&mut self) {
        self.pass = None;
    }

    /// The shadow batch of `batch` with the number of the open pass, when a pass is open and has
    /// not shadowed a batch of its group and shape: the batch as it is, for the pass's
    /// generation. It counts as in flight from here.
    fn shadow(&mut self, batch: &Batch) -> Option<(u64, Batch)> {
        let (generation, _) = self.pass.as_ref()?;
        let generation = *generation;
        if !self.shadowed.insert((batch.group, batch.shape)) {
            return None;
        }

        self.in_flight += 1;
        let mut shadow = batch.clone();
        shadow.kernels = generation;
        Some((self.passes, shadow))
    }

    /// A shadow batch of pass `pass` ended with `outcome`. The pass closes and answers once none
    /// of its shadow batches is in flight. A shadow batch of a pass that was ended or replaced
    /// changes nothing.
    fn ended(&mut self, pass: u64, outcome: Result<(), String>) {
        if self.pass.is_none() || pass != self.passes {
            return;
        }

        self.in_flight = self.in_flight.saturating_sub(1);
        if let Err(failure) = outcome {
            self.failure.get_or_insert(failure);
        }
        if self.in_flight > 0 {
            return;
        }

        if let Some((generation, reply)) = self.pass.take() {
            let outcome = match self.failure.take() {
                Some(failure) => Err(Error::Executor(format!(
                    "the shadow pass of generation {generation} failed: {failure}"
                ))),
                None => Ok(()),
            };
            // A control thread that stopped waiting has nothing to be told.
            let _ = reply.send(outcome);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
