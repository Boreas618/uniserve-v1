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
    /// The phases this device's runners serve, recorded here because the runners themselves live
    /// on the device thread and a router above the executors has to choose one without asking it.
    phases: Vec<ipc::PhaseId>,
    /// What the device thread is performing, and `None` between commands (2.4).
    activity: Arc<std::sync::Mutex<Option<String>>>,
    /// The media computations this device's runners plan to capture, recorded here for the same
    /// reason: the mechanism asks whether a media computation is served before it forms the batch,
    /// and that answer cannot be a round trip to the device thread (2.4).
    media: crate::plan::Buckets,
    queue: Arc<JobQueue>,
    thread: Option<JoinHandle<()>>,
    /// The device's graph pools, shared by every runner that issues here (4.1).
    pools: Arc<std::sync::Mutex<Pools>>,
    /// What the device has done since it started, for the metrics report (7.3).
    counters: Arc<SharedCounters>,
    /// The operators the last program-layer update could not compare, each with its reason
    /// (7.3 step 2): the runners on this device write it as they prepare and shadow a generation,
    /// and the control thread reports it when the precopy finishes. Empty is a fact - every
    /// operator the update replaced was compared.
    uncompared: Arc<std::sync::Mutex<Vec<String>>>,
    /// The rank's Storage Server client, bound after construction because the control thread builds
    /// it after the launch's claim while the executors exist before it: a sleep and a wake are the
    /// only calls that reach the ledger, and the device thread makes them (2.5).
    storage: Arc<std::sync::Mutex<Option<Arc<crate::memory::retained::Retained>>>>,
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
    /// The device budget every mapping on the device is booked against.
    pub budget: u64,
    /// How many streams the executor issues on; the graph range is divided among them.
    pub streams: usize,
    /// Bytes of the device's scratch pool: the shadow pair's region and the state group's
    /// snapshot, one reservation (7.3, 4.5). Zero when the rank never shadows.
    pub scratch: u64,
    /// Bytes of the shadow region alone, which the paired forward's twins allocate.
    ///
    /// The reservation covers `shadow + snapshot`, and the two tenants take their own figures from
    /// it: giving the twins the reservation would leave the snapshot nothing to allocate, which a
    /// model with no state group cannot show (its snapshot is zero) and one with a state group
    /// shows on the first update it takes.
    pub shadow: u64,
    /// Bytes of the device's resident pool, which the storage owners carve through the same
    /// allocator (4.1). Zero when the rank keeps nothing resident in this process.
    pub resident: u64,
}

impl Executor {
    /// Starts the device thread, which owns the runners and the arena their graphs use.
    ///
    /// The pool set arrives from [`Executor::open`], which the engine calls while it opens the
    /// resident owners: it is the one allocator of the device, so every owner and every runner that
    /// issues here books against the same allowance (4.1). `scratch` is the shadow pair's region,
    /// which the engine sized beside the graph pool.
    ///
    /// From this call on the calling thread issues no device work for this device: everything
    /// reaches the device through [`Executor::submit`], the program-layer calls, and the storage
    /// owners' recorded jobs.
    /// Opens the device's one pool set, before anything that allocates on the device exists.
    ///
    /// The resident owners (4.1) carve their storage from this allocator, and they are opened
    /// before the executor thread - the runner that serves batches is built from them - so the pool
    /// set is built here and handed to [`Executor::spawn`] rather than built inside it.
    pub fn open(device: DeviceId, graph: &GraphStorage) -> Result<Arc<std::sync::Mutex<Pools>>> {
        let handle = platform::Device::new(device)?;
        // One book per device (4.1): the allocator books every pool kind against the same
        // allowance, so the three kinds cannot sum past it.
        let totals = Arc::new(Pools::totals_of(graph.budget));
        let allocator =
            platform::Allocator::new(&handle, Arc::clone(&totals) as Arc<dyn platform::Budget>)?;
        Ok(Arc::new(std::sync::Mutex::new(Pools::new(
            allocator,
            totals,
            graph.bytes,
            graph.scratch,
            graph.resident,
            graph.streams,
        )?)))
    }

    pub fn spawn(
        device: DeviceId,
        mut runners: Vec<NamedRunner>,
        events: Arc<EventPool>,
        pools: Arc<std::sync::Mutex<Pools>>,
        scratch: u64,
        shadow: u64,
    ) -> Result<Self> {
        let counters = Arc::new(SharedCounters::default());
        // What the device thread is performing, read by whoever reports a stall from another thread
        // (2.4): the thread that is stuck cannot say it, and a stall line naming the stage alone
        // says nothing about which job never answered.
        let activity: Arc<std::sync::Mutex<Option<String>>> =
            Arc::new(std::sync::Mutex::new(None));
        let uncompared = Arc::new(std::sync::Mutex::new(Vec::new()));
        let storage = Arc::new(std::sync::Mutex::new(None));
        let phases: Vec<ipc::PhaseId> = runners
            .iter()
            .flat_map(|named| named.phases.iter().copied())
            .collect();
        let media = crate::plan::Buckets::new(
            runners
                .iter()
                .flat_map(|named| named.runner.media_plan())
                .collect::<Vec<_>>(),
        );
        for named in &mut runners {
            named.runner.attach_pools(Arc::clone(&pools));
            named.runner.attach_counters(Arc::clone(&counters));
            named.runner.attach_uncompared(Arc::clone(&uncompared));
        }
        let queue = JobQueue::new(device, Arc::clone(&events));
        let state = Device {
            device,
            runners,
            queue: Arc::clone(&queue),
            scratch,
            shadow,
            pools: Arc::clone(&pools),
            counters: Arc::clone(&counters),
            storage: Arc::clone(&storage),
            shadow_attached: false,
            activity: Arc::clone(&activity),
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
            phases,
            media,
            activity,
            queue,
            thread: Some(thread),
            pools,
            counters,
            storage,
            uncompared,
        })
    }

    /// The operators the last program-layer update could not compare, each with its reason (7.3
    /// step 2), in the order the device's runners recorded them.
    ///
    /// Read by the control thread when a precopy finishes, so the update's answer states what it
    /// has no evidence for instead of leaving that to a log line.
    pub fn uncompared(&self) -> Vec<String> {
        self.uncompared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// What the device thread is performing, or `None` when it is between commands (2.4).
    ///
    /// Read by whoever reports a stall: the thread that is stuck cannot say what it is stuck in, and
    /// a stall line that names only the stage says nothing about which job never answered.
    pub fn performing(&self) -> Option<String> {
        self.activity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// How many graph captures are open in this process on any stream.
    ///
    /// A hang inside a capture and a hang inside a launch look identical from outside and are
    /// different defects, so the stall report states this beside the job in hand (2.4). The count is
    /// the driver's own: a capture opens when `Capture::begin` returns and closes when it ends.
    pub fn capturing(&self) -> usize {
        platform::open_captures()
    }

    /// The device this executor drives.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Gives every runner the shadow harness of its device, carved from the device allocator
    /// (7.3): the main process calls this once, before the epoch serves.
    pub fn set_shadow(&self) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::SetShadow, &submission, JobKind::Shadow, None)?;
        Ok(submission)
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

    /// Runs one shadow batch: the paired forwards of `batch.generation`'s changed operators
    /// against the serving generation (7.3 step 2).
    pub fn shadow(&self, batch: Batch) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Shadow(batch), &submission, JobKind::Shadow, None)?;
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

    /// Releases everything that references `generation`, which was prepared and never served (7.3
    /// step 2).
    ///
    /// A failed precopy abandons the generation it prepared: its kernels unload with it, and the
    /// graphs and arena regions the runner rebound or recaptured for it are released here. `purge`
    /// cannot do this, because purge drops what is *older* than the serving generation and an
    /// abandoned generation is newer than it.
    pub fn discard(&self, generation: u64) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue.submit(
            Command::Discard(generation),
            &submission,
            JobKind::Discard,
            None,
        )?;
        Ok(submission)
    }

    /// Binds the rank's Storage Server client to this device, so a sleep can give segments back and
    /// a wake can map them again (2.5).
    ///
    /// Called once by the control thread after the executors are spawned: the client is built after
    /// the launch's claim, and the devices exist before it.
    pub fn bind_storage(&self, client: Arc<crate::memory::retained::Retained>) {
        *self
            .storage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(client);
    }

    /// Gives the resident cache's segments back, for a standby's park (2.5).
    ///
    /// One call per cache, in the device queue's order: a batch submitted before it finishes first,
    /// and nothing submitted after it runs until admission opens again on the wake's answer.
    pub fn sleep(&self) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Sleep, &submission, JobKind::Sleep, None)?;
        Ok(submission)
    }

    /// Maps the released cache segments again, `limit` of them or all of them (2.5).
    ///
    /// The answer is the measurement of what came back; a shortfall is not an error, because the
    /// limit is a bound rather than a demand. An `Err` is: the addresses are valid and the K/V is
    /// not, so the caller must not admit a request on this device.
    pub fn wake(&self, limit: Option<u32>) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Wake { limit }, &submission, JobKind::Wake, None)?;
        Ok(submission)
    }

    /// Whether one of this device's runners serves `phase` (4.4): what a router above the executors
    /// asks before it hands a batch over.
    pub fn serves(&self, phase: ipc::PhaseId) -> bool {
        self.phases.contains(&phase)
    }

    /// The phases this device's runners serve, in the order they were spawned.
    pub fn phases(&self) -> &[ipc::PhaseId] {
        &self.phases
    }

    /// The bucket this device would serve `computation` in, or `None` when its plan holds no graph
    /// for it (2.4's stated capability).
    ///
    /// The answer comes from the plan the runners stated at spawn, not from the device thread: the
    /// mechanism forms a media batch only where this answers, so a missing answer is the executor
    /// saying it does not serve the computation rather than a check nobody makes.
    pub fn media(&self, computation: &crate::plan::Computation) -> Option<crate::plan::Bucket> {
        self.media.select_media(computation)
    }

    /// The device's one pool set, for an owner that carves its own storage from the same allocator
    /// (the storage owners' resident planes, the paired forward's scratch pages).
    ///
    /// Nothing else on the device constructs a `Pools` (4.1): a second allocator would book the
    /// same device allowance twice.
    pub fn pools(&self) -> Arc<std::sync::Mutex<Pools>> {
        Arc::clone(&self.pools)
    }

    /// The device's mapped bytes per pool kind, from the one allocator that books them (4.1).
    ///
    /// The figures are one booking's totals rather than a sum over owners - one `Pools` is shared
    /// by every runner on the device, so a sum over runners would count the same mapping twice -
    /// and item 5's `graph + scratch + resident <= budget` is therefore decided by one book.
    pub fn memory(&self) -> crate::arena::DeviceMemory {
        self.pools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .memory()
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
                .map_err(|_| Error::State("the executor thread panicked"))?;
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
    /// Bytes of the scratch pool's reservation: the shadow region and the snapshot (7.3, 4.5).
    scratch: u64,
    /// Bytes of the shadow region alone, which the twins allocate from that reservation.
    shadow: u64,
    /// The device's one graph pool, which also lends its allocator for the shadow pair's region.
    pools: Arc<std::sync::Mutex<Pools>>,
    /// What the device thread has done, shared with the handle the control thread reads.
    counters: Arc<SharedCounters>,
    /// Whether this device's runners hold a shadow harness, so `SetShadow` is sent once (7.3).
    shadow_attached: bool,
    /// What the job in hand is doing, for the report of a failed submission and for a stall report
    /// read from another thread. The device thread is the only place that knows it, and a failure
    /// crosses the thread boundary as a bare error.
    activity: Arc<std::sync::Mutex<Option<String>>>,
    /// The context's first sticky error, once one has been seen (2.4).
    ///
    /// A sticky error is not recoverable: every launch after it fails, so the executor issues
    /// nothing more and fails every command it is handed with this error. The main process learns
    /// of it through the control thread's probe and replaces the process (2.2, 2.5).
    sticky: Option<platform::Error>,
    /// The rank's Storage Server client, which a sleep and a wake reach the ledger through (2.5).
    storage: Arc<std::sync::Mutex<Option<Arc<crate::memory::retained::Retained>>>>,
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
                queue.complete(
                    entry,
                    Err(Error::Cuda(sticky.clone())),
                    Some("the device's context already had a sticky error".to_owned()),
                );
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
            let detail = self
                .activity
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            queue.complete(entry, result, detail);
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
        let label = entry.label.clone();
        match &entry.command {
            Command::Batch(batch) => {
                let runner = self.runner_of(phase_of(batch))?;
                let jobs = self.runner_mut(runner)?.runner.submit(batch.clone())?;
                // A batch's jobs are its replay and nothing else: serving it is one turn, because
                // the mechanism waits on this submission to build the next step.
                self.perform(&label, jobs_of(runner, jobs)).map(Step::Done)
            }
            Command::Warmup => {
                let mut jobs = Vec::new();
                for named in &mut self.runners {
                    jobs.extend(jobs_of(named.id, named.runner.warmup()?));
                }
                self.steps(entry, &label, None, jobs)
            }
            Command::Shadow(batch) => {
                let runner = self.runner_of(phase_of(batch))?;
                let jobs = self.runner_mut(runner)?.runner.shadow(batch.clone())?;
                // A shadow batch is a set of independent paired forwards, one per changed operator:
                // they interleave with serving batches like a precopy's steps (7.3 step 2).
                self.steps(entry, &label, Some(batch.generation), jobs_of(runner, jobs))
            }
            Command::Precopy(next) => {
                let mut jobs = Vec::new();
                for named in &mut self.runners {
                    jobs.extend(jobs_of(named.id, named.runner.precopy(next)?));
                }
                self.steps(entry, &label, Some(next.number()), jobs)
            }
            Command::Steps { generation, jobs } => {
                self.steps(entry, &label, *generation, jobs.clone())
            }
            Command::Switchover(generation) => {
                for named in &mut self.runners {
                    named.runner.switchover(*generation)?;
                }
                self.synchronize()?;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::SetShadow => {
                // The shadow pair's region is allocator-backed and comes from the same device
                // allocator as the graph pools, reserved on first use; the harness then goes to
                // every runner, because the update may shadow an operator of any of them.
                // A second send in one process keeps the harness the first made: the region's
                // reservation is idempotent, but re-attaching twins would replace the `Shadow` an
                // update may be mid-comparison with, so the command answers empty instead.
                if self.shadow_attached {
                    return Ok(Step::Done(BatchOutput::default()));
                }
                let mut pools = self
                    .pools
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                // The reservation covers `shadow + snapshot` and is taken once; the twins take the
                // shadow region and the snapshot takes its own figure, so the two tenants of one
                // pool never claim each other's bytes.
                pools.reserve_scratch(self.scratch)?;
                for named in &mut self.runners {
                    let twins = ops::Twins::new(pools.allocator(), self.shadow)
                        .map_err(|error| Error::Interface(error.to_string()))?;
                    let shadow = crate::shadow::Shadow::new(self.device, twins)?;
                    named.runner.attach_shadow(shadow);
                }
                self.shadow_attached = true;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Sleep => {
                let client = self.storage()?;
                let mut storage: &crate::memory::retained::Retained = &client;
                for cache in self.caches() {
                    crate::memory::sleep(&cache, &mut storage)?;
                }
                // The segments are given back by now; the device's own work is finished before the
                // queue hands this command over, so a synchronize here would only wait on nothing.
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Wake { limit } => {
                let client = self.storage()?;
                let mut storage: &crate::memory::retained::Retained = &client;
                // One call per cache, on the stream of the runner that reads it: a second call for a
                // cache two runners share would reset the units the first had just mapped (2.5).
                let mut woken: Vec<*const ()> = Vec::new();
                for named in &mut self.runners {
                    let Some(cache) = named.runner.cache() else {
                        continue;
                    };
                    let address = Arc::as_ptr(&cache) as *const ();
                    if woken.contains(&address) {
                        continue;
                    }
                    woken.push(address);
                    let stream = named
                        .runner
                        .stream()
                        .ok_or_else(|| {
                            Error::Invariant(format!(
                                "runner {} reads a cache but states no stream to wake it on",
                                named.id.0
                            ))
                        })?
                        .clone();
                    crate::memory::wake(&cache, &mut storage, &stream, *limit)?;
                }
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
            Command::Discard(generation) => {
                for named in &mut self.runners {
                    named.runner.discard(*generation)?;
                }
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
    fn steps(
        &mut self,
        entry: &crate::job::Entry,
        label: &str,
        generation: Option<u64>,
        jobs: Vec<Job>,
    ) -> Result<Step> {
        let mut jobs = jobs;
        if jobs.is_empty() {
            return Ok(Step::Done(BatchOutput::default()));
        }
        let first = jobs.remove(0);
        let output = self.perform(label, vec![first])?;
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
    fn perform(&mut self, label: &str, jobs: Vec<Job>) -> Result<BatchOutput> {
        let mut output = BatchOutput::default();
        for job in jobs {
            let runner = job
                .runner()
                .ok_or_else(|| Error::Invariant("a serving job names no runner".into()))?;
            // The job in hand names the bucket, generation and operator a failure is about; the
            // command's label says which submission was running it.
            *self
                .activity
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(format!("{label}, {}", job.describe()));
            let started = Instant::now();
            let produced = self.runner_mut(runner)?.runner.serve(&job)?;
            self.synchronize()?;
            self.counters.time(started.elapsed());
            if !produced.calls.is_empty() {
                if !output.calls.is_empty() {
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
    /// The rank's Storage Server client, or the refusal a sleep and a wake give when none is bound.
    ///
    /// The handle comes back owned so the caller holds it for the length of the call; the slot keeps
    /// its own share, so a control thread that binds it once serves every park of the epoch.
    fn storage(&self) -> Result<Arc<crate::memory::retained::Retained>> {
        self.storage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or_else(|| {
                Error::State(
                    "this device has no Storage Server client: a standby cannot give its resident \
                     cache segments back or map them again"
                        .into(),
                )
            })
    }

    /// The distinct caches this device's runners read, one per cache.
    fn caches(&self) -> Vec<Arc<std::sync::Mutex<crate::memory::prefix_cache::PrefixCache>>> {
        let mut caches = Vec::new();
        let mut addresses: Vec<*const ()> = Vec::new();
        for named in &self.runners {
            let Some(cache) = named.runner.cache() else {
                continue;
            };
            let address = Arc::as_ptr(&cache) as *const ();
            if addresses.contains(&address) {
                continue;
            }
            addresses.push(address);
            caches.push(cache);
        }
        caches
    }

    fn runner_mut(&mut self, id: RunnerId) -> Result<&mut NamedRunner> {
        self.runners
            .iter_mut()
            .find(|named| named.id == id)
            .ok_or_else(|| Error::Invariant(format!("the device has no runner {}", id.0)))
    }
}

