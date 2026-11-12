//! The executor's queue: its jobs in submission order, the job in flight, and the close.
//!
//! Every thread that has work for a device submits a [`Job`] here and holds the
//! [`Submission`] it gets back; the device's executor thread takes the jobs one at a time, in the
//! order they were submitted, and blocks on the queue while it holds none. The queue keeps three
//! things true:
//!
//! - **Order.** A job is taken after every job submitted before it. A job of several turns (a
//!   warm-up, a generation being prepared, a shadow batch) goes back to the tail after each
//!   turn, so whatever was submitted meanwhile runs between two of its turns, and its
//!   submission completes with its last turn.
//! - **The job in flight.** From the moment the executor takes a job until it posts the job's
//!   completion, the queue states what the executor is performing and since when. The control
//!   thread reads it without waiting for the executor: the watchdog for a job that stopped making
//!   progress, a stall report for the bucket or operator it stopped in. A failed job's error
//!   carries the same statement ([`crate::Error::InFlight`]).
//! - **The close.** A closed queue takes no job, and the jobs it still held end as cancelled, so
//!   a thread waiting on one of them is woken.
//!
//! A batch is refused where it is submitted when no runner of the executor serves its phase: a
//! job in the queue is one the executor can perform.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use ipc::PhaseId;
use ipc::execution::{
    BatchId, BatchOutput, Fault, Outcome, Submission, SubmissionId, SubmissionState,
};

use super::Job;
use crate::{Error, Result, lock};

/// The jobs of one executor, in submission order.
pub struct Queue {
    /// The phases the executor's runners serve: what a submitted batch is checked against.
    phases: Vec<PhaseId>,
    state: Mutex<State>,
    /// Signalled when a job is queued or the queue closes: what an idle executor blocks on.
    ready: Condvar,
}

struct State {
    /// The number of the next submission.
    next: u64,
    pending: VecDeque<Entry>,
    in_flight: Option<InFlight>,
    closed: bool,
}

/// One job in the queue, with the completion its submitter waits on.
pub(super) struct Entry {
    pub(super) job: Job,
    pub(super) submission: Arc<SubmissionState>,
    /// When the job was submitted: where a batch's queued time starts.
    pub(super) submitted: Instant,
}

/// What the executor is performing.
struct InFlight {
    /// When the executor took the job's current turn.
    since: Instant,
    /// The batch the job belongs to, when it executes or shadows one.
    batch: Option<BatchId>,
    /// The work in flight, in words: the job, and once the executor stated it, the bucket,
    /// generation or operator the job is at.
    activity: String,
}

impl Queue {
    /// An open queue for an executor whose runners serve `phases`.
    pub(super) fn new(phases: Vec<PhaseId>) -> Arc<Self> {
        Arc::new(Self {
            phases,
            state: Mutex::new(State {
                next: 0,
                pending: VecDeque::new(),
                in_flight: None,
                closed: false,
            }),
            ready: Condvar::new(),
        })
    }

    /// The phases the executor's runners serve, in the runners' order.
    pub fn phases(&self) -> &[PhaseId] {
        &self.phases
    }

    /// Queues `job` behind every job submitted before it and returns the submission that
    /// completes when the executor has issued the job's work: with the step's output for a batch,
    /// and with an empty output for any other job.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a batch whose phase no runner of the executor serves, and
    /// [`Error::State`] once the queue is closed.
    pub fn submit(&self, job: Job) -> Result<Submission> {
        if let Some(batch) = job.batch()
            && !self.phases.contains(&batch.phase)
        {
            return Err(Error::Invalid(format!(
                "no runner of this executor serves phase {} of batch {}; its runners serve {:?}",
                batch.phase.0,
                batch.id.0,
                self.phases.iter().map(|phase| phase.0).collect::<Vec<_>>()
            )));
        }

        let mut state = lock(&self.state);
        if state.closed {
            return Err(Error::State("the executor's queue is closed"));
        }

        let (submission, completion) = Submission::pending(SubmissionId(state.next));
        state.next += 1;
        state.pending.push_back(Entry {
            job,
            submission: completion,
            submitted: Instant::now(),
        });
        drop(state);

        self.ready.notify_one();
        Ok(submission)
    }

    /// Takes the next job, which is in flight from here on; blocks while the queue holds none.
    /// `None` once the queue is closed.
    pub(super) fn next(&self) -> Option<Entry> {
        let mut state = lock(&self.state);

        loop {
            if let Some(entry) = state.pending.pop_front() {
                state.in_flight = Some(InFlight {
                    since: Instant::now(),
                    batch: entry.job.batch().map(|batch| batch.id),
                    activity: entry.job.to_string(),
                });
                return Some(entry);
            }
            if state.closed {
                return None;
            }

            state = self
                .ready
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// States what the job in flight is at: the bucket it captures, the operator it compares.
    pub(super) fn note(&self, activity: String) {
        if let Some(in_flight) = lock(&self.state).in_flight.as_mut() {
            in_flight.activity = activity;
        }
    }

    /// Ends the turn of the job in flight and puts `entry`, the job's remainder, at the tail: its
    /// submission completes with a later turn. In a queue closed meanwhile the job ends as
    /// cancelled.
    pub(super) fn requeue(&self, entry: Entry) {
        let mut state = lock(&self.state);
        state.in_flight = None;

        if state.closed {
            drop(state);
            entry.submission.finish(Outcome::Cancelled);
            return;
        }

        state.pending.push_back(entry);
        drop(state);
        self.ready.notify_one();
    }

    /// Posts the completion of the job in flight, whose submission is `submission`. A failure is
    /// posted with the work that was in flight when it was taken, in the error and as the
    /// submission's failure label.
    pub(super) fn complete(&self, submission: Arc<SubmissionState>, result: Result<BatchOutput>) {
        let activity = lock(&self.state)
            .in_flight
            .take()
            .map(|in_flight| in_flight.activity);

        let status = match result {
            Ok(output) => Outcome::Success(output),
            Err(error) => {
                let error = match activity {
                    Some(activity) => {
                        submission.note_failure(activity.clone());
                        Error::in_flight(activity, error)
                    }
                    None => error,
                };
                let fault: Arc<Fault> = Arc::new(error);
                Outcome::Failed(fault)
            }
        };
        submission.finish(status);
    }

    /// When the executor took the job in flight and the batch the job belongs to; `None` between
    /// two jobs. This is what the watchdog reads.
    pub fn in_flight(&self) -> Option<(Instant, Option<BatchId>)> {
        lock(&self.state)
            .in_flight
            .as_ref()
            .map(|in_flight| (in_flight.since, in_flight.batch))
    }

    /// What the executor is performing, in words; `None` between two jobs. The executor cannot
    /// say what it stopped in, so whoever reports a stall reads it here.
    pub fn performing(&self) -> Option<String> {
        lock(&self.state)
            .in_flight
            .as_ref()
            .map(|in_flight| in_flight.activity.clone())
    }

    /// Closes the queue: no further job is taken, the queued jobs end as cancelled, and the
    /// executor returns once the job in flight ended.
    pub fn close(&self) {
        let cancelled: Vec<Entry> = {
            let mut state = lock(&self.state);
            state.closed = true;
            state.pending.drain(..).collect()
        };
        self.ready.notify_all();

        for entry in cancelled {
            entry.submission.finish(Outcome::Cancelled);
        }
    }
}

impl std::fmt::Debug for Queue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = lock(&self.state);

        formatter
            .debug_struct("Queue")
            .field("pending", &state.pending.len())
            .field("in_flight", &state.in_flight.is_some())
            .field("closed", &state.closed)
            .finish()
    }
}
