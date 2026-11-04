//! The device executor's queue: what every thread hands the device, in submission order.
//!
//! One executor thread per device performs every command (specs/worker.md 2.4). A command is one
//! of:
//!
//! - a batch to serve, which the thread stages and replays through the runner that serves its
//!   phase, then performs the jobs the runner returned (4.5);
//! - a program-layer step: `precopy` (prepare the next generation), `switchover` (at a batch
//!   boundary), `purge` (after the old generation's last batch), each becoming the jobs the
//!   runners returned (7.3);
//! - a storage owner's recorded issue work (copies, fills, fences and waits) for its own
//!   submission;
//! - a wait for the events deferred storage release hangs on: [`JobQueue::after`] is the closure
//!   [`EventPool::set_submitter`](crate::stream::EventPool::set_submitter) installs, and the
//!   executor completes the event pool's deferred work when the wait ended.
//!
//! A command submitted with a [`Submission`] completes when its work completed on the device; the
//! scheduling mechanism waits on it instead of on the device. A command without a submission (an
//! `after` wait, a storage owner's own work) completes on its own.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use ipc::PhaseId;
use ops::Generation;
use platform::DeviceId;

use crate::batch::{
    Batch, BatchOutput, Job, JobKind, Submission, SubmissionId, SubmissionState,
};
use crate::completion::Outcome;
use crate::stream::events::{EventPool, JobId};
use crate::stream::recorded;
use crate::{Error, Result, lock};

/// One command waiting for the device thread, or being performed by it.
pub(crate) struct Entry {
    /// The command's number, in submission order.
    pub(crate) id: JobId,
    pub(crate) command: Command,
    /// The submission the command belongs to, when a caller waits on it.
    pub(crate) submission: Option<Arc<SubmissionState>>,
    /// The submission's number, for the watchdog's report.
    pub(crate) submission_id: Option<SubmissionId>,
    /// What the command is, for the watchdog's report.
    pub(crate) kind: JobKind,
    /// The batch the command serves, when it serves one.
    pub(crate) batch: Option<crate::batch::BatchId>,
    /// Whether this entry is the last step of its submission (4.5: a precopy or a capture pass is
    /// performed one job per queue step, so serving batches interleave with it).
    pub(crate) final_step: bool,
}

/// One command the device executor performs.
pub(crate) enum Command {
    /// Serve one batch: stage it, choose its bucket, replay its graph.
    Batch(Batch),
    /// Capture every bucket of the serving generation (warmup).
    Warmup,
    /// One queued step of a multi-job command: the jobs not yet performed (4.5's capture pass and
    /// precopy run one job per queue turn, so serving batches interleave with them).
    Steps { generation: Option<u64>, jobs: Vec<Job> },
    /// Prepare the next generation (7.3 step 2).
    Precopy(Arc<Generation>),
    /// Switch the serving generation at a batch boundary (7.3 step 3).
    Switchover(u64),
    /// Discard the update prepared for a generation, keeping the serving one (7.3 step 4).
    Revert(u64),
    /// Release the graphs that reference a generation older than this (7.3 step 6).
    Purge(u64),
    /// A storage owner's recorded issue work.
    Issue(recorded::Batch),
    /// Wait for the events deferred storage release hangs on.
    After(Vec<Arc<platform::Event>>),
}

struct Current {
    submission: Option<SubmissionId>,
    kind: JobKind,
    batch: Option<crate::batch::BatchId>,
    started: std::time::Instant,
}

struct State {
    next: u64,
    pending: std::collections::VecDeque<Entry>,
    current: Option<Current>,
    closed: bool,
}

/// The queue the device executor takes its commands from.
pub struct JobQueue {
    device: DeviceId,
    events: Arc<EventPool>,
    state: Mutex<State>,
    ready: Condvar,
}

impl JobQueue {
    /// A queue for `device`, whose deferred storage release is completed through `events`.
    pub(crate) fn new(device: DeviceId, events: Arc<EventPool>) -> Arc<Self> {
        Arc::new(Self {
            device,
            events,
            state: Mutex::new(State {
                next: 0,
                pending: std::collections::VecDeque::new(),
                current: None,
                closed: false,
            }),
            ready: Condvar::new(),
        })
    }

    /// The device the queue's executor drives.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The event pool deferred work is completed through.
    pub(crate) fn events(&self) -> &Arc<EventPool> {
        &self.events
    }

    /// The number of the next submission the executor will make.
    pub(crate) fn next_submission(&self) -> u64 {
        lock(&self.state).next
    }

    /// Appends one command, with the submission that waits on it.
    pub(crate) fn push(
        &self,
        command: Command,
        submission: Option<(SubmissionId, Arc<SubmissionState>)>,
        kind: JobKind,
        batch: Option<crate::batch::BatchId>,
    ) -> Result<JobId> {
        let mut state = lock(&self.state);
        if state.closed {
            return Err(Error::State("the job queue is closed".into()));
        }
        let id = JobId(state.next);
        state.next += 1;
        let (submission_id, submission) = match submission {
            Some((id, state)) => (Some(id), Some(state)),
            None => (None, None),
        };
        state.pending.push_back(Entry {
            id,
            command,
            submission,
            submission_id,
            kind,
            batch,
            final_step: true,
        });
        drop(state);
        self.ready.notify_one();
        Ok(id)
    }

    /// Puts `command` at the tail as the next step of `entry`'s submission (4.5's interleaving).
    ///
    /// The submission is the same one, so it completes only when the last step does; a serving
    /// batch submitted while the previous step ran is queued ahead of this one and is served
    /// between the two.
    pub(crate) fn requeue(&self, entry: &Entry, command: Command) -> Result<JobId> {
        let mut state = lock(&self.state);
        if state.closed {
            return Err(Error::State("the job queue is closed".into()));
        }
        let id = JobId(state.next);
        state.next += 1;
        state.pending.push_back(Entry {
            id,
            command,
            submission: entry.submission.clone(),
            submission_id: entry.submission_id,
            kind: entry.kind,
            batch: entry.batch,
            final_step: false,
        });
        drop(state);
        self.ready.notify_one();
        Ok(id)
    }

    /// Appends a serving command whose completion the caller waits on.
    pub(crate) fn submit(
        &self,
        command: Command,
        submission: &Submission,
        kind: JobKind,
        batch: Option<crate::batch::BatchId>,
    ) -> Result<JobId> {
        let (state, id) = submission.parts();
        self.push(command, Some((id, state)), kind, batch)
    }

    /// The job that completes once `events` completed: the closure
    /// [`EventPool::set_submitter`] installs. It is called from any thread, so it neither blocks
    /// nor holds the queue's lock for long.
    pub fn after(&self, _device: DeviceId, events: Vec<Arc<platform::Event>>) -> JobId {
        let events = events;
        let mut state = lock(&self.state);
        let id = JobId(state.next);
        state.next += 1;
        state.pending.push_back(Entry {
            id,
            command: Command::After(events),
            submission: None,
            submission_id: None,
            kind: JobKind::After,
            batch: None,
            final_step: true,
        });
        drop(state);
        self.ready.notify_one();
        id
    }

    /// The recorded issue work of one storage submission, named for the queue.
    pub(crate) fn issue(&self, work: recorded::Batch) -> Result<JobId> {
        self.push(Command::Issue(work), None, JobKind::Issue, None)
    }

    /// The next command, once one is pending or `timeout` passed; `None` when the queue closed.
    pub(crate) fn next(&self, timeout: Option<Duration>) -> Option<Entry> {
        let mut state = lock(&self.state);
        loop {
            if state.closed {
                return None;
            }
            if let Some(entry) = state.pending.pop_front() {
                state.current = Some(Current {
                    submission: entry.submission_id,
                    kind: entry.kind,
                    batch: entry.batch,
                    started: std::time::Instant::now(),
                });
                return Some(entry);
            }
            state = match timeout {
                Some(timeout) => {
                    self.ready
                        .wait_timeout(state, timeout)
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .0
                }
                None => self
                    .ready
                    .wait(state)
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            };
        }
    }

    /// Reports that `entry` ended with `result`, and completes the submission that waited on it.
    ///
    /// A serving command hands over the step's output; every other command's output is empty.
    pub(crate) fn complete(&self, entry: Entry, result: Result<BatchOutput>) {
        {
            let mut state = lock(&self.state);
            state.current = None;
        }
        if entry.kind == JobKind::After {
            // The command's only work was its wait, which completing it establishes.
            if let Err(error) = self.events.complete(entry.id, result.is_ok()) {
                tracing::warn!(%error, "the event pool refused a completed wait");
            }
        }
        if let Some(submission) = entry.submission {
            if !entry.final_step {
                // A middle step of a multi-job command: the submission waits for the last one.
                return;
            }
            let outcome = match result {
                Ok(output) => Outcome::Success(output),
                Err(error) => Outcome::Failed(Arc::new(error)),
            };
            submission.finish(outcome);
        }
    }

    /// Whether a command is being performed now; a test-side probe of the watchdog's source.
    #[cfg(test)]
    fn watchdog_is_running(&self) -> bool {
        !self.in_flight().is_empty()
    }

    /// What the control thread's watchdog reads (2.4): the command in flight, if any.
    pub(crate) fn in_flight(&self) -> Vec<crate::device::Watch> {
        let state = lock(&self.state);
        state
            .current
            .as_ref()
            .map(|current| {
                vec![crate::device::Watch {
                    submission: current.submission,
                    job: JobId(0),
                    kind: current.kind,
                    batch: current.batch,
                    started: current.started,
                    elapsed: current.started.elapsed(),
                }]
            })
            .unwrap_or_default()
    }

    /// Stops the queue: no further command is taken, and a waiting executor thread returns.
    pub fn close(&self) {
        let mut state = lock(&self.state);
        state.closed = true;
        state.pending.clear();
        drop(state);
        self.ready.notify_all();
    }
}

impl std::fmt::Debug for JobQueue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = lock(&self.state);
        formatter
            .debug_struct("JobQueue")
            .field("device", &self.device)
            .field("pending", &state.pending.len())
            .field("closed", &state.closed)
            .finish()
    }
}

/// The jobs a runner returned for one command, in order.
pub(crate) fn jobs_of(runner: crate::runner::RunnerId, jobs: crate::batch::Jobs) -> Vec<Job> {
    jobs.iter()
        .map(|job| match job {
            Job::Replay {
                bucket,
                generation,
                batch,
                ..
            } => Job::Replay {
                runner,
                bucket: *bucket,
                generation: *generation,
                batch: *batch,
            },
            Job::Capture {
                bucket, generation, ..
            } => Job::Capture {
                runner,
                bucket: *bucket,
                generation: *generation,
            },
            Job::Shadow {
                operator,
                key,
                bucket,
                generation,
                ..
            } => Job::Shadow {
                runner,
                operator,
                key: key.clone(),
                bucket: *bucket,
                generation: *generation,
            },
            Job::Rebind {
                bucket, generation, ..
            } => Job::Rebind {
                runner,
                bucket: *bucket,
                generation: *generation,
            },
            Job::Issue { .. } | Job::After { .. } => {
                unreachable!("a runner returned a job the device queue owns")
            }
        })
        .collect()
}

/// The kind of a command, for the watchdog.
pub(crate) fn kind_of(command: &Command) -> JobKind {
    match command {
        Command::Batch(_) => JobKind::Batch,
        Command::Warmup => JobKind::Warmup,
        Command::Precopy(_) => JobKind::Precopy,
        Command::Switchover(_) => JobKind::Switchover,
        Command::Revert(_) => JobKind::Revert,
        Command::Purge(_) => JobKind::Purge,
        Command::Issue(_) => JobKind::Issue,
        Command::After(_) => JobKind::After,
    }
}

/// The phase a batch names, for the runner lookup.
pub(crate) fn phase_of(batch: &Batch) -> PhaseId {
    batch.phase
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{BatchColumns, Submission};
    use crate::plan::{Shape, Variant};

    fn queue() -> (Arc<JobQueue>, Arc<EventPool>) {
        let events = EventPool::new();
        (JobQueue::new(platform::DeviceId(0), Arc::clone(&events)), events)
    }

    fn batch() -> Batch {
        Batch {
            id: crate::batch::BatchId(1),
            phase: ipc::PhaseId(0),
            calls: Vec::new(),
            shape: Shape::decode(1, Variant::DecodeGreedy),
            columns: BatchColumns::default(),
            generation: 1,
        }
    }

    /// A command completes its submission only when the device thread performed it, and the
    /// command it performs is what the watchdog reads meanwhile.
    #[test]
    fn a_submission_completes_when_its_command_is_performed() {
        let (queue, _events) = queue();
        let (submission, _) = Submission::pending(crate::batch::SubmissionId(0));
        queue
            .submit(Command::Batch(batch()), &submission, JobKind::Batch, None)
            .expect("a command is queued");

        assert!(submission.outcome().is_none(), "nothing performed yet");
        let entry = queue.next(Some(Duration::from_millis(10))).expect("a command");
        assert_eq!(entry.kind, JobKind::Batch);
        assert!(queue.watchdog_is_running());

        queue.complete(entry, Ok(()));
        assert!(matches!(submission.outcome(), Some(Outcome::Success(()))));
        assert!(queue.in_flight().is_empty());
    }

    /// Commands run in submission order, and a failed command fails its submission.
    #[test]
    fn commands_run_in_order_and_a_failure_is_reported() {
        let (queue, _events) = queue();
        let (first, _) = Submission::pending(crate::batch::SubmissionId(0));
        let (second, _) = Submission::pending(crate::batch::SubmissionId(1));
        queue
            .submit(Command::Batch(batch()), &first, JobKind::Batch, None)
            .expect("queued");
        queue
            .submit(Command::Purge(2), &second, JobKind::Purge, None)
            .expect("queued");

        let entry = queue.next(Some(Duration::from_millis(10))).expect("first");
        assert_eq!(entry.kind, JobKind::Batch);
        queue.complete(entry, Ok(()));
        let entry = queue.next(Some(Duration::from_millis(10))).expect("second");
        assert_eq!(entry.kind, JobKind::Purge);
        queue.complete(entry, Err(Error::State("the purge failed".into())));

        assert!(matches!(first.outcome(), Some(Outcome::Success(()))));
        assert!(matches!(second.outcome(), Some(Outcome::Failed(_))));
    }

    /// A capture pass or a precopy performs one job per queue turn and puts the rest at the tail,
    /// so a serving batch the scheduler submitted meanwhile is served between two of its jobs
    /// (4.5's interleaving, not one long pause).
    #[test]
    fn a_capture_pass_interleaves_with_a_serving_batch() {
        let (queue, _events) = queue();
        let bucket = crate::plan::Bucket::decode(2, crate::plan::Variant::DecodeGreedy);
        let jobs: Vec<Job> = (0..3)
            .map(|_| Job::Capture {
                runner: crate::runner::RunnerId(0),
                bucket,
                generation: 2,
            })
            .collect();
        let (precopy, _) = Submission::pending(crate::batch::SubmissionId(0));
        queue
            .submit(
                Command::Steps {
                    generation: Some(2),
                    jobs,
                },
                &precopy,
                JobKind::Precopy,
                None,
            )
            .expect("the precopy is queued");
        let (served, _) = Submission::pending(crate::batch::SubmissionId(1));
        queue
            .submit(Command::Batch(batch()), &served, JobKind::Batch, None)
            .expect("the batch is queued");

        // The device loop, as the executor performs it: one job of a multi-job command per turn.
        let mut order = Vec::new();
        while let Some(entry) = queue.next(Some(Duration::from_millis(5))) {
            match entry.command {
                Command::Steps { generation, mut jobs } => {
                    order.push("capture");
                    jobs.remove(0);
                    if jobs.is_empty() {
                        queue.complete(entry, Ok(BatchOutput::default()));
                    } else {
                        queue
                            .requeue(&entry, Command::Steps { generation, jobs })
                            .expect("the remainder is queued");
                        assert!(
                            precopy.outcome().is_none(),
                            "the submission waits for the last step"
                        );
                    }
                }
                Command::Batch(_) => {
                    order.push("batch");
                    queue.complete(entry, Ok(BatchOutput::default()));
                    assert!(
                        precopy.outcome().is_none(),
                        "a served batch does not complete the precopy"
                    );
                }
                _ => queue.complete(entry, Ok(BatchOutput::default())),
            }
        }

        // The batch runs between the first and the last capture, not after the whole pass.
        assert_eq!(order, vec!["capture", "batch", "capture", "capture"]);
        assert!(matches!(precopy.outcome(), Some(Outcome::Success(_))));
        assert!(matches!(served.outcome(), Some(Outcome::Success(_))));
    }

    /// A step that fails fails its submission and drops the rest of its command: a precopy whose
    /// shadow comparison failed must not go on capturing.
    #[test]
    fn a_failed_step_fails_its_submission_and_drops_the_rest() {
        let (queue, _events) = queue();
        let bucket = crate::plan::Bucket::decode(2, crate::plan::Variant::DecodeGreedy);
        let jobs: Vec<Job> = (0..3)
            .map(|_| Job::Capture {
                runner: crate::runner::RunnerId(0),
                bucket,
                generation: 2,
            })
            .collect();
        let (precopy, _) = Submission::pending(crate::batch::SubmissionId(0));
        queue
            .submit(
                Command::Steps {
                    generation: Some(2),
                    jobs,
                },
                &precopy,
                JobKind::Precopy,
                None,
            )
            .expect("queued");

        let mut turns = 0;
        while let Some(entry) = queue.next(Some(Duration::from_millis(5))) {
            turns += 1;
            if turns == 1 {
                queue.complete(entry, Err(Error::State("the shadow comparison failed".into())));
            } else {
                panic!("the rest of the command was performed after a failure");
            }
        }
        assert!(matches!(precopy.outcome(), Some(Outcome::Failed(_))));
    }

    /// A deferred-release wait is a command of its own, after everything submitted before it.
    #[test]
    fn an_after_wait_is_a_command_and_closing_stops_the_queue() {
        let (queue, events) = queue();
        let id = queue.after(platform::DeviceId(0), Vec::new());
        let entry = queue
            .next(Some(Duration::from_millis(10)))
            .expect("the wait");
        assert_eq!(entry.id, id);
        assert_eq!(entry.kind, JobKind::After);
        queue.complete(entry, Ok(()));

        // The pool has no deferred work under this job, which is not a failure of the wait.
        assert!(!events.owns(id));

        queue.close();
        assert!(queue.next(Some(Duration::from_millis(10))).is_none());
        assert!(
            queue.push(Command::Purge(3), None, JobKind::Purge, None).is_err(),
            "a closed queue takes no command"
        );
    }
}
