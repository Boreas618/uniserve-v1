//! The device executor: one thread per device, the only issuer of work to that device's streams.
//!
//! The design (specs/worker.md 2.4) gives the executor thread one command at a time, in submission
//! order: serve a batch, prepare or switch a generation, release an old one, perform a storage
//! owner's copies and fences, or wait for the events deferred release hangs on. It blocks only on
//! the current command's device work, so the command it is performing is exactly what the control
//! thread's watchdog reads ([`Watchdog`]).
//!
//! This module is the handle every other thread holds. The scheduling mechanism submits a batch and
//! gets a [`Submission`] it can wait on; the control thread reads the watchdog and probes the
//! device's context for a sticky error (2.4); the program-layer update sends `precopy`,
//! `switchover` and `purge`, which take effect at batch boundaries (7.3).

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use nn::Generation;
use platform::DeviceId;

use crate::batch::{Batch, BatchOutput, Job, JobKind, Submission, SubmissionId};
use crate::completion::Outcome;
use crate::job::{Command, JobQueue, jobs_of, kind_of, phase_of};
use crate::runner::{NamedRunner, RunnerId};
use crate::stream::Streams;
use crate::stream::events::{DRAIN_LIMIT, EventPool, JobId, drain_event};
use crate::{Error, Result};

/// A device executor's handle.
pub struct Executor {
    device: DeviceId,
    queue: Arc<JobQueue>,
    thread: Option<JoinHandle<()>>,
}

impl Executor {
    /// Starts the device thread, which owns `runners`, `streams` and the arena their graphs use.
    ///
    /// From this call on the calling thread issues no device work for this device: everything
    /// reaches the device through [`Executor::submit`], the program-layer calls, and the storage
    /// owners' recorded jobs.
    pub fn spawn(
        device: DeviceId,
        runners: Vec<NamedRunner>,
        streams: Streams,
        events: Arc<EventPool>,
    ) -> Result<Self> {
        let queue = JobQueue::new(device, Arc::clone(&events));
        let state = Device {
            device,
            runners,
            streams,
            queue: Arc::clone(&queue),
            sticky: None,
        };
        let thread_queue = Arc::clone(&queue);
        let thread = std::thread::Builder::new()
            .name(format!("executor-{}", device.0))
            .spawn(move || state.run(&thread_queue))
            .map_err(|error| {
                Error::State(format!(
                    "the executor thread of {device} did not start: {error}"
                ))
            })?;
        Ok(Self {
            device,
            queue,
            thread: Some(thread),
        })
    }

    /// The device this executor drives.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The queue every thread submits device work through.
    pub fn queue(&self) -> &Arc<JobQueue> {
        &self.queue
    }

    /// Submits one batch and returns the submission the mechanism waits on.
    pub fn submit(&self, batch: Batch) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Batch(batch), &submission, JobKind::Batch, None)?;
        Ok(submission)
    }

    /// Asks every runner to capture its buckets of the serving generation, before the executor
    /// serves its first batch, and returns the submission that finishes it.
    pub fn warmup(&self) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Warmup, &submission, JobKind::Warmup, None)?;
        Ok(submission)
    }

    /// Asks every runner to prepare `next` and returns the submission that finishes it (7.3 step
    /// 2).
    pub fn precopy(&self, next: Arc<Generation>) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Precopy(next), &submission, JobKind::Precopy, None)?;
        Ok(submission)
    }

    /// Switches the serving generation at the next batch boundary (7.3 step 3).
    pub fn switchover(&self, generation: u64) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue.submit(
            Command::Switchover(generation),
            &submission,
            JobKind::Switchover,
            None,
        )?;
        Ok(submission)
    }

    /// Discards the update prepared for `generation` and keeps the serving one (7.3 step 4).
    pub fn revert(&self, generation: u64) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Revert(generation), &submission, JobKind::Revert, None)?;
        Ok(submission)
    }

    /// Releases the graphs that still reference a generation older than `generation` (7.3 step 6).
    pub fn purge(&self, generation: u64) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue.submit(
            Command::Purge(generation),
            &submission,
            JobKind::Purge,
            None,
        )?;
        Ok(submission)
    }

    /// What the control thread's watchdog reads: the command in flight and how long it took (2.4).
    pub fn watchdog(&self) -> Watchdog {
        Watchdog {
            jobs: self.queue.in_flight(),
        }
    }

    /// Probes this device's context for a sticky error, which the control thread does once a
    /// second while a command has run longer than a second (2.4).
    pub fn probe(&self) -> Result<Probe> {
        let context = platform::PrimaryContext::retain(self.device)?;
        Ok(match context.sticky_error()? {
            Some(error) => Probe::Sticky(error),
            None => Probe::Idle,
        })
    }

    /// Drains the device's commands and joins its thread. Every submission must have completed
    /// before the caller closes the executor.
    pub fn close(mut self) -> Result<()> {
        self.queue.close();
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| Error::State("the executor thread panicked".into()))?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Executor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Executor")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// One command in flight, as the control thread's stall rule reads it (2.4).
#[derive(Clone, Debug)]
pub struct Watch {
    /// The submission whose command is in flight, when the command belongs to one.
    pub submission: Option<SubmissionId>,
    /// The command's number.
    pub job: JobId,
    /// What kind of command it is.
    pub kind: JobKind,
    /// The batch it serves, when it serves one.
    pub batch: Option<crate::batch::BatchId>,
    /// When the device thread began it.
    pub started: std::time::Instant,
    /// How long it has been in flight at the reading.
    pub elapsed: Duration,
}

/// The watchdog's reading: the commands in flight on one device.
#[derive(Clone, Debug, Default)]
pub struct Watchdog {
    /// The commands in flight, in submission order (at most one).
    pub jobs: Vec<Watch>,
}

impl Watchdog {
    /// The command in flight longest, which the stall rule names when it fires.
    pub fn longest(&self) -> Option<&Watch> {
        self.jobs.iter().max_by_key(|watch| watch.elapsed)
    }

    /// The longest a command has been in flight, `Duration::ZERO` when none is.
    pub fn elapsed(&self) -> Duration {
        self.longest().map_or(Duration::ZERO, |watch| watch.elapsed)
    }
}

/// What a sticky-error probe found (2.4).
#[derive(Debug)]
pub enum Probe {
    /// The device's context is healthy.
    Idle,
    /// The device's context has a sticky error: the executor cannot serve another batch, and the
    /// main process must be replaced (2.2, 2.5).
    Sticky(platform::Error),
}

/// The device thread's state: the runners, the streams only it issues to, and the queue.
struct Device {
    device: DeviceId,
    runners: Vec<NamedRunner>,
    streams: Streams,
    queue: Arc<JobQueue>,
    /// The context's first sticky error, once one has been seen (2.4).
    ///
    /// A sticky error is not recoverable: every launch after it fails, so the executor issues
    /// nothing more and fails every command it is handed with this error. The main process learns
    /// of it through the control thread's probe and replaces the process (2.2, 2.5).
    sticky: Option<platform::Error>,
}

impl Device {
    /// Serves the queue until it closes: one command at a time, in submission order.
    fn run(mut self, queue: &Arc<JobQueue>) {
        // Deferred storage release gets its waiting commands through this queue (2.4).
        let events = Arc::clone(queue.events());
        let submitter = Arc::clone(queue);
        events.set_submitter(Some(Box::new(move |device, events| {
            submitter.after(device, events)
        })));
        while let Some(entry) = queue.next(None) {
            if let Some(sticky) = &self.sticky {
                // The context is unusable: the command is not issued, and its submission fails
                // with the sticky error rather than with a launch error that would hide it.
                queue.complete(entry, Err(Error::Cuda(sticky.clone())));
                continue;
            }
            let result = self.serve(entry.command);
            if let Err(error) = &result {
                tracing::warn!(%error, device = self.device.0, "a device command failed");
            }
            if result.is_err() {
                // A failure may be the sticky error's first symptom; probe before the next command
                // so the executor stops issuing to a broken context (2.4).
                if let Ok(Some(sticky)) = self.probe_sticky() {
                    tracing::error!(
                        device = self.device.0,
                        %sticky,
                        "the device's context has a sticky error; no further command is issued"
                    );
                    self.sticky = Some(sticky);
                }
            }
            queue.complete(entry, result);
        }
        events.set_submitter(None);
    }

    /// Reads this device's context for a sticky error, the check the stall rule ends in (2.4).
    fn probe_sticky(&self) -> Result<Option<platform::Error>> {
        let context = platform::PrimaryContext::retain(self.device)?;
        context.sticky_error()
    }

    /// Performs one command and waits for its device work before returning (2.4: the executor
    /// blocks only on the current command's device work).
    ///
    /// A command that serves a batch answers with that step's output; every other command answers
    /// with an empty one.
    fn serve(&mut self, command: Command) -> Result<BatchOutput> {
        match command {
            Command::Batch(batch) => {
                let runner = self.runner_of(phase_of(&batch))?;
                let jobs = self.runner_mut(runner)?.submit(batch)?;
                self.perform(jobs_of(runner, jobs))
            }
            Command::Warmup => {
                let mut jobs = Vec::new();
                for named in &mut self.runners {
                    jobs.extend(jobs_of(named.id, named.runner.warmup()?));
                }
                self.perform(jobs)
            }
            Command::Precopy(next) => {
                let mut jobs = Vec::new();
                for named in &mut self.runners {
                    jobs.extend(jobs_of(named.id, named.runner.precopy(&next)?));
                }
                self.perform(jobs)
            }
            Command::Switchover(generation) => {
                for named in &mut self.runners {
                    named.runner.switchover(generation);
                }
                self.streams.synchronize()?;
                Ok(BatchOutput::default())
            }
            Command::Revert(generation) => {
                for named in &mut self.runners {
                    named.runner.revert(generation);
                }
                self.streams.synchronize()?;
                Ok(BatchOutput::default())
            }
            Command::Purge(generation) => {
                for named in &mut self.runners {
                    named.runner.purge(generation);
                }
                self.streams.synchronize()?;
                Ok(BatchOutput::default())
            }
            Command::Issue(work) => {
                work.perform()?;
                self.streams.synchronize()?;
                Ok(BatchOutput::default())
            }
            Command::After(events) => {
                // The command establishes that the work deferred on these events may run: wait
                // for them on the host, bounded like every host wait for device work.
                let deadline = std::time::Instant::now() + DRAIN_LIMIT;
                for event in &events {
                    drain_event(event, deadline)?;
                }
                Ok(BatchOutput::default())
            }
        }
    }

    /// Performs a runner's jobs in order, each on the runner that returned it.
    ///
    /// The answer is the output of the jobs that produced one; a job list with no output (every
    /// program-layer command) answers empty.
    fn perform(&mut self, jobs: Vec<Job>) -> Result<BatchOutput> {
        let mut output = BatchOutput::default();
        for job in jobs {
            let runner = job
                .runner()
                .ok_or_else(|| Error::Invariant("a serving job names no runner".into()))?;
            let produced = self.runner_mut(runner)?.serve(&job)?;
            self.streams.synchronize()?;
            if !produced.tokens.is_empty() {
                if !output.tokens.is_empty() {
                    return Err(Error::Invariant(
                        "two jobs of one command produced tokens".into(),
                    ));
                }
                output = produced;
            }
        }
        Ok(output)
    }

    /// The runner that serves `phase`.
    fn runner_of(&self, phase: ipc::PhaseId) -> Result<RunnerId> {
        self.runners
            .iter()
            .find(|named| named.serves(phase))
            .map(|named| named.id)
            .ok_or_else(|| {
                Error::Invariant(format!("no runner of this device serves phase {}", phase.0))
            })
    }

    /// The runner with `id`.
    fn runner_mut(&mut self, id: RunnerId) -> Result<&mut NamedRunner> {
        self.runners
            .iter_mut()
            .find(|named| named.id == id)
            .ok_or_else(|| Error::Invariant(format!("the device has no runner {}", id.0)))
    }
}

/// The kind of a command, for a caller that reports on it.
pub(crate) fn kind(command: &Command) -> JobKind {
    kind_of(command)
}

/// The outcome of a submission that ended with `result`.
pub(crate) fn outcome(result: Result<BatchOutput>) -> Outcome<BatchOutput> {
    match result {
        Ok(output) => Outcome::Success(output),
        Err(error) => Outcome::Failed(Arc::new(error)),
    }
}
