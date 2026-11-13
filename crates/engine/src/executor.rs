//! The engine's executor: the [`scheduler::Executor`](crate::scheduler::Executor) of one
//! scheduler, over the node's executors ([`worker::executor::Executors`]).
//!
//! A scheduler hands its executor a batch to run, a completion of the buffer pool to map into a
//! group's device block tables, and requests for a group to stamp (`specs/scheduler-design.md`
//! §2.1, `Executor`). [`Groups`] hands each to the node's executors of the worker group it names,
//! which number the group's jobs and give them to every rank of the group in the order they were
//! handed, and returns what they produce to the scheduler as [`Message`]s on its one channel:
//!
//! - a batch's output arrives exactly once, as `Message::Output`: the runner's output, or the
//!   error of a batch the executors refused or failed;
//! - the `Finish` of each copy that ran with a batch arrives just before the batch's output, as
//!   one `Message::Submit`, each `Finish` stamped by the group with that batch. A batch that
//!   failed after copies had started with it carries a `Finish { ok: false }` for each
//!   (`worker::Error::finished`), sent the same way, so the scheduler learns that those copies
//!   ended instead of giving them up at its bound;
//! - the requests of a stamp arrive as one `Message::Submit` once every rank of the group has
//!   applied them to its device block tables, each with the group's stamp;
//! - a map sends nothing. A map or a stamp that fails leaves the group's device block tables
//!   without what the scheduler counts on: its failure arrives as `Message::Failed`, in its place
//!   among the group's messages, and the scheduler fails at once (SD §3.7).
//!
//! ## Threads and state
//!
//! The scheduler's thread calls `submit`, `map` and `stamp`, which hand the job to the node's
//! executors and never wait for it. Each worker group of the scheduler has one waiter thread,
//! which takes the group's submissions in the order they were handed and blocks on each in turn
//! until it completes. The executors complete a group's jobs in that order, so a waiter never
//! waits behind a job that ended after a later one, and since it is the only thread that sends
//! its group's messages, they arrive in the order the group's jobs were done. Nothing runs on an
//! executor's thread: a waker would run there, and may only signal.
//!
//! [`Groups`] owns the waiters: dropping it closes their channels and joins them once the
//! submissions they hold have completed. The node's executors are shared with the control thread
//! and the node's other schedulers, whose groups are other groups.
//!
//! ## Shadow batches
//!
//! A generation update compares each kernel it replaces with the serving one on the batches the
//! scheduler serves (`worker::Executor::shadow`). The scheduler does not know of it: the engine's
//! control thread opens a pass of the prepared generation on a [`Shadow`], and while the pass is
//! open `submit` also hands the executors each batch whose group and shape the pass has not
//! shadowed yet as a shadow batch of that generation, right after the batch itself. The pass
//! answers once every shadow batch it started has ended.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};
use ipc::batch::{Batch, BatchOutput, Shape};
use storage::runtime::{self, Completion, GroupId};
use worker::completion::{Outcome, Submission};
use worker::executor::Executors;
use worker::lock;

use crate::scheduler::{self, Error, Message};

/// The executor of one scheduler's worker groups (module documentation).
pub struct Groups {
    /// The node's executors, by worker group: shared with the control thread and the node's
    /// other schedulers.
    executors: Arc<Executors>,
    /// Per worker group of the scheduler, in the order given: the group and the channel of its
    /// waiter.
    waiters: Vec<(GroupId, Sender<Wait>)>,
    /// The waiters' threads, joined when the executor is dropped.
    threads: Vec<JoinHandle<()>>,
    /// The pass the engine opens over this scheduler's batches while it prepares a generation.
    shadow: Arc<Mutex<Shadow>>,
}

/// One submission a group's waiter waits for, in the order the job was handed to the group's
/// executors, with what the scheduler is sent once it completes. A job the executors refused
/// carries the refusal in place of its submission.
#[expect(
    clippy::large_enum_variant,
    reason = "a batch travels by value once per batch, to its `Message::Output`"
)]
enum Wait {
    /// A batch: the `Finish` of each copy that ran with it goes to the scheduler as
    /// `Message::Submit`, then its output as `Message::Output`.
    Batch(Batch, worker::Result<Submission>),
    /// A shadow batch of the pass numbered here: its outcome goes to the pass.
    Shadow(u64, worker::Result<Submission>),
    /// The map of the completion of the request numbered here: nothing is sent.
    Map(runtime::RequestId, worker::Result<Submission>),
    /// A stamp: its requests go to the scheduler as `Message::Submit`.
    Stamp(worker::Result<Submission>),
}

impl Groups {
    /// The executor of a scheduler whose worker groups are `groups`, over `executors`, the
    /// node's, with one waiter thread per group, which sends what the group's executors return
    /// to the scheduler's channel `inbox`. `shadow` is the pass the engine opens over this
    /// scheduler's batches; the engine keeps a clone of it.
    ///
    /// A job of a group that is not one of `groups` is refused without reaching the executors:
    /// they hold other schedulers' groups too.
    ///
    /// # Errors
    ///
    /// [`Error::Executor`] for no group, a group given twice, or a waiter thread that cannot
    /// start.
    pub fn new(
        executors: Arc<Executors>,
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
            executors,
            waiters: Vec::with_capacity(groups.len()),
            threads: Vec::with_capacity(groups.len()),
            shadow,
        };
        for &group in groups {
            let (waits, waiting) = crossbeam_channel::unbounded();
            let inbox = inbox.clone();
            let shadow = Arc::clone(&built.shadow);

            let thread = std::thread::Builder::new()
                .name(format!("group-{}", group.0))
                .spawn(move || wait(group, &waiting, &inbox, &shadow))
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

    /// Hands a job of `group` to the executors through `hand`, unless `group` is not one of the
    /// scheduler's.
    fn hand(
        &self,
        group: GroupId,
        hand: impl FnOnce(&Executors) -> worker::Result<Submission>,
    ) -> worker::Result<Submission> {
        if self.waiters.iter().any(|(served, _)| *served == group) {
            return hand(&self.executors);
        }

        Err(worker::Error::Invalid(format!(
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

impl scheduler::Executor for Groups {
    /// Hands `batch` to the executors of its group, and while a shadow pass is open, a shadow
    /// batch of it right after when the pass has not shadowed its group and shape. The batch is
    /// kept for its `Message::Output`; the executors take a copy.
    fn submit(&self, batch: Batch) {
        let group = batch.group;
        let submission = self.hand(group, |executors| executors.submit(batch.clone()));

        let shadow = match &submission {
            Ok(_) => lock(&self.shadow).shadow(&batch),
            Err(_) => None,
        };
        self.send(group, Wait::Batch(batch, submission));

        if let Some((pass, shadow)) = shadow {
            let submission = self.hand(group, |executors| executors.shadow(shadow));
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
        let submission = self.hand(group, |executors| {
            executors.map(group, slot, id, request, completion)
        });
        self.send(group, Wait::Map(id, submission));
    }

    fn stamp(&self, group: GroupId, requests: Vec<runtime::Request>) {
        if requests.is_empty() {
            return;
        }

        let submission = self.hand(group, |executors| executors.stamp(group, requests));
        self.send(group, Wait::Stamp(submission));
    }
}

impl Drop for Groups {
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
/// scheduler stops on it.
fn wait(group: GroupId, waits: &Receiver<Wait>, inbox: &Sender<Message>, shadow: &Mutex<Shadow>) {
    // A scheduler that stopped reads nothing more: what it is sent then is dropped.
    for wait in waits {
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

/// What a submission returned once it completed: a batch's output and the requests the group's
/// executors hand back for the submission queue; or the failure in words, the refusal of a job
/// the executors did not take included, with the `Finish { ok: false }` of each copy that had
/// started with a failed batch, which the group's executors applied and stamped
/// (`worker::Error::finished`; none for any other failure).
fn returned(
    submission: worker::Result<Submission>,
) -> Result<(BatchOutput, Vec<runtime::Request>), (String, Vec<runtime::Request>)> {
    match submission.map(|submission| submission.wait()) {
        Ok(Outcome::Success(returned)) => Ok(returned),
        Ok(Outcome::Failed(fault)) => {
            let finished = fault
                .downcast_ref::<worker::Error>()
                .map_or_else(Vec::new, |error| error.finished().to_vec());
            Err((fault.to_string(), finished))
        }
        Ok(Outcome::Cancelled) => Err((
            "the executors closed before they ran the job".to_owned(),
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
    /// beside the serving one (`worker::Executor::prepare`). From the next batch on, each batch
    /// whose group and shape the pass has not shadowed is also run as a shadow batch of
    /// `generation`. Once every shadow batch the pass started has ended, and at least one has,
    /// the pass closes and `reply` receives `Ok(())`, or [`Error::Executor`] with its first
    /// failure: a pair of kernels beyond its tolerance, or a shadow batch the executors refused
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
        shadow.generation = generation;
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
