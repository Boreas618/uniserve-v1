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
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ops::Generation;
use platform::DeviceId;

use crate::batch::{Batch, BatchOutput, Job, JobKind, Submission, SubmissionId};
use crate::job::{Command, JobQueue, jobs_of, phase_of};
use crate::runner::{NamedRunner, RunnerId};
use crate::arena::Pools;
use crate::stream::events::{DRAIN_LIMIT, EventPool, JobId, drain_event};
use crate::{Error, Result};

/// A device executor's handle.
pub struct Executor {
    device: DeviceId,
    queue: Arc<JobQueue>,
    thread: Option<JoinHandle<()>>,
    /// The device's graph pools, shared by every runner that issues here (4.1).
    pools: Arc<std::sync::Mutex<Pools>>,
    /// What the device has done since it started, for the metrics report (7.3).
    counters: Arc<SharedCounters>,
}

/// What a device has done since it started: the metrics report's counters (7.3).
///
/// `kernel_launches` counts the kernels of every replayed graph, summed over the device's runners;
/// it is what the report's `kernel_launches` carries. `device_ns` is the time the device thread
/// spent inside issued work, which is device time because the executor synchronizes after each
/// job and nothing else issues on this device's streams (2.4); with CUPTI profiling on, the
/// profiler's own kernel time replaces this figure in the report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Kernel launches issued, summed over every runner of the device.
    pub kernel_launches: u64,
    /// Device time spent in those launches, in nanoseconds.
    pub device_ns: u64,
}

/// The same counters as the device thread and its runners add to them.
///
/// Both are monotonic sums, so a reader on the control thread may take a reading while the device
/// thread adds to it: each field is one atomic add and one atomic load. A runner receives it from
/// [`Runner::attach_counters`](crate::runner::Runner::attach_counters) and adds the launches of
/// every replay; the device thread adds its own time around each job.
#[derive(Debug, Default)]
pub struct SharedCounters {
    kernel_launches: AtomicU64,
    device_ns: AtomicU64,
}

impl SharedCounters {
    /// Adds `launches` kernel launches.
    pub fn launches(&self, launches: u64) {
        self.kernel_launches.fetch_add(launches, Ordering::Relaxed);
    }

    /// Adds device time spent in one job.
    fn time(&self, elapsed: Duration) {
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.device_ns.fetch_add(nanos, Ordering::Relaxed);
    }

    /// The counters as the metrics report reads them.
    fn read(&self) -> Counters {
        Counters {
            kernel_launches: self.kernel_launches.load(Ordering::Relaxed),
            device_ns: self.device_ns.load(Ordering::Relaxed),
        }
    }
}

/// The device's graph storage, as the engine sizes it before the executor starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphStorage {
    /// Bytes the device may map for captured graphs (4.1's budget).
    pub bytes: u64,
    /// The device budget those mappings are booked against.
    pub budget: u64,
    /// How many streams the executor issues on; the range is divided among them.
    pub streams: usize,
}

impl Executor {
    /// Starts the device thread, which owns `runners`, `streams` and the arena their graphs use.
    ///
    /// From this call on the calling thread issues no device work for this device: everything
    /// reaches the device through [`Executor::submit`], the program-layer calls, and the storage
    /// owners' recorded jobs.
    pub fn spawn(
        device: DeviceId,
        mut runners: Vec<NamedRunner>,
        events: Arc<EventPool>,
        graph: GraphStorage,
    ) -> Result<Self> {
        let handle = platform::Device::new(device)?;
        let allocator = platform::Allocator::new(&handle, Arc::new(GraphBudget { bytes: graph.budget }))?;
        let pools = Arc::new(std::sync::Mutex::new(Pools::new(
            allocator,
            graph.bytes,
            graph.budget,
            graph.streams,
        )?));
        let counters = Arc::new(SharedCounters::default());
        for named in &mut runners {
            named.runner.attach_pools(Arc::clone(&pools));
            named.runner.attach_counters(Arc::clone(&counters));
        }
        let queue = JobQueue::new(device, Arc::clone(&events));
        let state = Device {
            device,
            runners,
            queue: Arc::clone(&queue),
            counters: Arc::clone(&counters),
            sticky: None,
        };
        let thread_queue = Arc::clone(&queue);
        let thread = std::thread::Builder::new()
            .name(format!("executor-{}", device.0))
            .spawn(move || state.run(&thread_queue))
            .map_err(|error| {
                Error::Invalid(format!(
                    "the executor thread of {device} did not start: {error}"
                ))
            })?;
        Ok(Self {
            device,
            queue,
            thread: Some(thread),
            pools,
            counters,
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

    /// The device's graph pools as their ledger books them: the reserved range, the bytes mapped
    /// and the budget they are booked against (4.1, item 5's invariant).
    ///
    /// This is the device's figure and not a sum over its runners: one `Pools` is shared by every
    /// runner on the device, so summing runners would count the same mapping twice. The pools are
    /// created when the executor starts, so this is `Some` from then on; the answer stays an
    /// `Option` for a caller that reads a device it has not yet started serving.
    pub fn pool_metrics(&self) -> Option<crate::arena::PoolMetrics> {
        Some(
            self.pools
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .metrics(),
        )
    }

    /// What the device has done since it started: kernel launches and device time (7.3).
    pub fn counters(&self) -> Counters {
        self.counters.read()
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

/// What one turn of the device thread finished (4.5's interleaving).
enum Step {
    /// The command finished; its submission completes with this output.
    Done(BatchOutput),
    /// The command put its remainder at the tail; its submission completes on that step.
    Requeued,
}

/// The device's graph-storage budget, as the allocator books against it (4.1).
struct GraphBudget {
    /// Bytes the device may map for captured graphs.
    bytes: u64,
}

impl platform::Budget for GraphBudget {
    fn book(&self, _device: DeviceId, _pool: platform::PoolKind, bytes: u64) -> platform::Result<()> {
        if bytes > self.bytes {
            return Err(platform::Error::InvalidArgument(format!(
                "a graph pool of {bytes} bytes exceeds the device's {} bytes",
                self.bytes
            )));
        }
        Ok(())
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
    queue: Arc<JobQueue>,
    /// What the device thread has done, shared with the handle the control thread reads.
    counters: Arc<SharedCounters>,
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
            let result = self.serve(&entry);
            if let Err(error) = &result {
                tracing::warn!(%error, device = self.device.0, "a device command failed");
            }
            if matches!(result, Ok(Step::Requeued)) {
                // The entry put the rest of its command at the tail; its submission waits for the
                // last step, so nothing completes here.
                continue;
            }
            let result = match result {
                Ok(Step::Done(output)) => Ok(output),
                Ok(Step::Requeued) => unreachable!("handled above"),
                Err(error) => Err(error),
            };
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
        context.sticky_error().map_err(Into::into)
    }

    /// Performs one command and waits for its device work before returning (2.4: the executor
    /// blocks only on the current command's device work).
    ///
    /// A command that serves a batch answers with that step's output; every other command answers
    /// with an empty one. A multi-job command ([`Command::Steps`]) performs one job and puts the
    /// rest back at the tail of the queue, so a serving batch the scheduler submitted meanwhile is
    /// served between two of its jobs (4.5's interleaving).
    fn serve(&mut self, entry: &crate::job::Entry) -> Result<Step> {
        match &entry.command {
            Command::Batch(batch) => {
                let runner = self.runner_of(phase_of(batch))?;
                let jobs = self.runner_mut(runner)?.runner.submit(batch.clone())?;
                // A batch's jobs are its replay and nothing else: serving it is one turn, because
                // the mechanism waits on this submission to build the next step.
                self.perform(jobs_of(runner, jobs)).map(Step::Done)
            }
            Command::Warmup => {
                let mut jobs = Vec::new();
                for named in &mut self.runners {
                    jobs.extend(jobs_of(named.id, named.runner.warmup()?));
                }
                self.steps(entry, None, jobs)
            }
            Command::Precopy(next) => {
                let mut jobs = Vec::new();
                for named in &mut self.runners {
                    jobs.extend(jobs_of(named.id, named.runner.precopy(next)?));
                }
                self.steps(entry, Some(next.number()), jobs)
            }
            Command::Steps { generation, jobs } => {
                self.steps(entry, *generation, jobs.clone())
            }
            Command::Switchover(generation) => {
                for named in &mut self.runners {
                    named.runner.switchover(*generation);
                }
                self.synchronize()?;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Revert(generation) => {
                for named in &mut self.runners {
                    named.runner.revert(*generation);
                }
                self.synchronize()?;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Purge(generation) => {
                for named in &mut self.runners {
                    named.runner.purge(*generation);
                }
                self.synchronize()?;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Issue(work) => {
                work.lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .perform()?;
                self.synchronize()?;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::After(events) => {
                // The command establishes that the work deferred on these events may run: wait
                // for them on the host, bounded like every host wait for device work.
                let deadline = std::time::Instant::now() + DRAIN_LIMIT;
                for event in events {
                    drain_event(event, deadline)?;
                }
                Ok(Step::Done(BatchOutput::default()))
            }
        }
    }

    /// Performs one job of a multi-job command and puts the rest at the tail (4.5).
    ///
    /// A capture pass and a precopy are sets of independent per-bucket jobs; performing them in
    /// one turn would stop serving for the whole set, which is the long pause the design forbids.
    /// One job per turn lets whatever the scheduler submitted in between be served first, and a
    /// quiet queue still runs the steps back to back.
    fn steps(&mut self, entry: &crate::job::Entry, generation: Option<u64>, jobs: Vec<Job>) -> Result<Step> {
        let mut jobs = jobs;
        if jobs.is_empty() {
            return Ok(Step::Done(BatchOutput::default()));
        }
        let first = jobs.remove(0);
        let output = self.perform(vec![first])?;
        if jobs.is_empty() {
            return Ok(Step::Done(output));
        }
        self.queue.requeue(entry, Command::Steps { generation, jobs })?;
        Ok(Step::Requeued)
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
            let started = Instant::now();
            let produced = self.runner_mut(runner)?.runner.serve(&job)?;
            self.synchronize()?;
            self.counters.time(started.elapsed());
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

    /// Waits for the streams this device's runners issued on.
    ///
    /// A runner owns the stream its graphs replay on, so waiting on those is exactly waiting for
    /// this device's issued work; the executor has no stream of its own to add.
    fn synchronize(&self) -> Result<()> {
        for named in &self.runners {
            if let Some(stream) = named.runner.stream() {
                stream.synchronize()?;
            }
        }
        Ok(())
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

