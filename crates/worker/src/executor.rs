//! `Executor`: one thread per device, the only issuer of work on the device's streams.
//!
//! Every other thread of the engine reaches a device through its executor's [`Queue`]: the
//! scheduler submits a batch, the control thread a generation to prepare or switch to, a memory
//! owner a copy it decided on. The executor thread takes the jobs one at a time in the order
//! they were submitted and calls the device's runners, which nothing else calls. A stream's
//! order is therefore the order of the jobs, and a job that must run with nothing in flight on
//! the device only has to be a job.
//!
//! ## Jobs
//!
//! A [`Job`] is one of:
//!
//! - a **batch to execute**, through the runner that serves the batch's phase
//!   ([`Runner::execute`]). An expert-union member with no tokens of its own serves its part of
//!   a peer's step as such a batch, of variant `ExpertJoin`, which the stepping peer submits
//!   through this executor's queue;
//! - a **shadow batch** of the generation being prepared: one paired forward
//!   ([`Graphs::shadow`](crate::Graphs::shadow)) per operator the batch's bucket launches whose
//!   kernel the generation replaces;
//! - the **capture** of the runners' buckets (warm-up), largest first;
//! - a generation to **prepare**, to **switch** to, to **commit** and to **roll back**, each
//!   on every runner of the device;
//! - a change of the device's **profiling session**;
//! - a **wait** for stream events, which deferred storage release hangs on;
//! - a **control operation**: a closure the executor calls with its runners. This is how a
//!   memory owner or the engine runs what must happen at a batch boundary, with nothing in
//!   flight on the device: page copies, sleeping and waking cache segments, suspending and
//!   restoring graph storage, a weight update. Work a storage owner decides on another thread
//!   is issued this way too, so nothing but the executor issues on the device's streams.
//!
//! A capture, a preparation and a shadow batch take one queue turn per bucket or operator and
//! go back to the tail in between, so batches submitted meanwhile are executed between two of
//! their turns instead of behind all of them.
//!
//! ## Completion and waiting
//!
//! A job's [`Submission`] completes when the executor has issued the job's work: for a batch,
//! when the runner returned its output. The executor never polls: an idle executor blocks on its
//! queue, and what must wait for the device waits on a stream event
//! ([`Graphs::wait`](crate::Graphs::wait), `platform::Event::synchronize`).
//!
//! ## Failure
//!
//! A failed job posts its error with the work that was in flight ([`crate::Error::InFlight`]),
//! and the jobs behind it still run. The exception is a device whose context reports a sticky
//! error, which no later launch survives: the executor checks the context after a failure and
//! from then on fails every job with that error without issuing it. The control thread makes the
//! same check on its own schedule ([`Executor::sticky_error`], [`watchdog`]), because a job
//! stuck behind such an error never returns to make it.
//!
//! ## The parts
//!
//! | Module | Holds |
//! | --- | --- |
//! | [`queue`] | [`Queue`]: the jobs in submission order, the job in flight, the close |
//! | [`batches`] | [`Executors`]: the engine's `ipc::execution::Batches` over a node's executors and its peers |
//! | [`watchdog`] | [`Watchdog`](watchdog::Watchdog): when the control thread looks at the jobs in flight, and which it reports |
//! | [`control`] | [`Control`](control::Control): the control thread's connections to the daemon and its wake |
//! | [`peers`] | The channel between the engine processes of an epoch: registration, batches to ranks on other hosts and their outputs |

pub mod batches;
pub mod control;
pub mod peers;
pub mod queue;
pub mod watchdog;

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use ipc::PhaseId;
use ipc::execution::{Batch, BatchId, BatchOutput, Computation, Outcome, Submission};
use ops::Generation;
use platform::cupti::blocking::{Calls, Watched};
use platform::{DeviceId, Event};

pub use self::batches::Executors;
use self::queue::Entry;
pub use self::queue::Queue;
use crate::buckets::Bucket;
use crate::profiling;
use crate::profiling::session::{Change, Session};
use crate::runner::Runner;
use crate::{Error, Result, lock};

/// A control operation: what the executor calls with its runners at a batch boundary, with
/// nothing in flight on the device. The runners are in the order the executor was given them,
/// so `RunnerId(index)` names `runners[index]`.
pub type Operation = Box<dyn FnOnce(&mut [Box<dyn Runner>]) -> Result<()> + Send>;

/// One job of an executor's queue.
pub enum Job {
    /// Execute one batch through the runner that serves its phase.
    Execute(Box<Batch>),
    /// Run the paired forwards of one shadow batch: the generation `batch.generation` names,
    /// which is being prepared, against the serving one, for each operator the batch's bucket
    /// launches whose kernel changed. One paired forward per queue turn.
    Shadow {
        batch: Box<Batch>,
        /// The changed operators already compared: zero when the job is submitted.
        compared: usize,
    },
    /// Capture every bucket the runners state and have no graph of, largest first: the warm-up.
    /// One bucket per queue turn.
    Capture,
    /// Prepare a generation beside the serving one on every runner: the graphs it changes are
    /// rebound or recaptured, one bucket per queue turn.
    Prepare(Arc<Generation>),
    /// Switch every runner to the prepared generation of this number, which then serves on
    /// trial.
    Switch(u64),
    /// Commit the trial of the generation of this number: the generation it replaced is
    /// released.
    Commit(u64),
    /// Roll the generation of this number back, on trial or prepared and never switched to: its
    /// graphs are released, and after a trial the generation it replaced serves again.
    Rollback(u64),
    /// Change the device's profiling session; the rank's response to the daemon's operation is
    /// sent to `response`.
    Profile {
        change: Change,
        response: Sender<serde_json::Value>,
    },
    /// Wait until the device has passed the latest record of each event.
    Wait(Vec<Arc<Event>>),
    /// Call `operation` with the runners, once the device has passed everything they issued.
    Control {
        /// What the operation does, for a stall report and a failure.
        label: String,
        operation: Operation,
    },
}

impl Job {
    /// The batch the job executes or shadows.
    fn batch(&self) -> Option<&Batch> {
        match self {
            Self::Execute(batch) | Self::Shadow { batch, .. } => Some(batch),
            _ => None,
        }
    }

    /// Whether the job runs a runner's forwards, whose time the executor counts.
    fn runs_forwards(&self) -> bool {
        matches!(
            self,
            Self::Execute(_) | Self::Shadow { .. } | Self::Capture | Self::Prepare(_)
        )
    }
}

/// What the job is, as a stall report and a failure name the work in flight.
impl std::fmt::Display for Job {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execute(batch) => write!(
                formatter,
                "executing batch {} of phase {} ({})",
                batch.id.0, batch.phase.0, batch.shape
            ),
            Self::Shadow { batch, .. } => write!(
                formatter,
                "shadowing generation {} on batch {} of phase {} ({})",
                batch.generation, batch.id.0, batch.phase.0, batch.shape
            ),
            Self::Capture => formatter.write_str("warming up: capturing the runners' buckets"),
            Self::Prepare(next) => write!(formatter, "preparing generation {}", next.number()),
            Self::Switch(number) => write!(formatter, "switching to generation {number}"),
            Self::Commit(number) => write!(formatter, "committing generation {number}"),
            Self::Rollback(number) => write!(formatter, "rolling back generation {number}"),
            Self::Profile { .. } => formatter.write_str("changing the profiling session"),
            Self::Wait(events) => write!(
                formatter,
                "waiting for {} events of deferred storage release",
                events.len()
            ),
            Self::Control { label, .. } => formatter.write_str(label),
        }
    }
}

/// What an executor has done since it started; monotonic sums the control thread reads while
/// the executor adds to them.
#[derive(Default)]
struct Counters {
    /// Batches executed.
    steps: AtomicU64,
    /// Nanoseconds the executor spent in jobs that run a runner's forwards.
    forward_ns: AtomicU64,
}

/// The executor of one device, as every other thread holds it.
pub struct Executor {
    device: Option<DeviceId>,
    /// The buckets the runners state, read before the runners moved to the executor thread: the
    /// scheduler asks whether a media computation is served before it forms the batch, and that
    /// is not a round trip to the executor.
    buckets: Vec<Bucket>,
    queue: Arc<Queue>,
    counters: Arc<Counters>,
    /// What the generation update in flight has no comparison for; see [`Executor::uncompared`].
    uncompared: Arc<Mutex<Vec<String>>>,
    thread: Option<JoinHandle<()>>,
}

impl Executor {
    /// Starts the executor of `device` over `runners`, whose graphs serve `kernels`.
    ///
    /// `device` is the CUDA device the runners issue on: the executor keeps its context current
    /// on the thread and checks it for a sticky error. `None` is an executor of runners that
    /// hold no device (host work, or graphs made by [`Graphs::meta`](crate::Graphs::meta)),
    /// which records no profiling session.
    ///
    /// From this call on the runners are called on the executor thread alone.
    ///
    /// # Errors
    ///
    /// When the thread cannot start.
    pub fn spawn(
        device: Option<DeviceId>,
        runners: Vec<Box<dyn Runner>>,
        kernels: Arc<Generation>,
    ) -> Result<Self> {
        let phases: Vec<PhaseId> = runners
            .iter()
            .flat_map(|runner| runner.phases().iter().copied())
            .collect();
        let buckets: Vec<Bucket> = runners.iter().flat_map(|runner| runner.buckets()).collect();

        let queue = Queue::new(phases);
        let counters = Arc::new(Counters::default());
        let uncompared = Arc::new(Mutex::new(Vec::new()));

        let state = Thread {
            device,
            runners,
            queue: Arc::clone(&queue),
            kernels: BTreeMap::from([(kernels.number(), kernels)]),
            counters: Arc::clone(&counters),
            uncompared: Arc::clone(&uncompared),
            unmeasured: Vec::new(),
            sticky: None,
            session: None,
            blocking: Calls::default(),
            renewals: profiling::graph_renewals(),
        };
        let name = match device {
            Some(device) => format!("executor-{}", device.0),
            None => "executor-host".to_owned(),
        };
        let thread = std::thread::Builder::new()
            .name(name)
            .spawn(move || state.run())
            .map_err(|error| Error::transport_source("starting an executor thread", error))?;

        Ok(Self {
            device,
            buckets,
            queue,
            counters,
            uncompared,
            thread: Some(thread),
        })
    }

    /// The CUDA device this executor issues on, if it has one.
    pub fn device(&self) -> Option<DeviceId> {
        self.device
    }

    /// The queue every thread submits this executor's jobs through. A peer of an expert union
    /// holds it to submit the `ExpertJoin` batch of a step this executor's rank has no tokens
    /// for.
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// The phases the runners serve, in the runners' order.
    pub fn phases(&self) -> &[PhaseId] {
        self.queue.phases()
    }

    /// Whether a runner of this executor serves `phase`.
    pub fn serves(&self, phase: PhaseId) -> bool {
        self.queue.phases().contains(&phase)
    }

    /// The smallest bucket the runners state that holds `computation`, or `None` when this
    /// executor serves no such computation.
    ///
    /// A computation that states an extent is matched to the bucket of exactly that extent; one
    /// that states none gets the bucket the runners state ([`Bucket::holds_computation`]).
    pub fn media(&self, computation: &Computation) -> Option<Bucket> {
        self.buckets
            .iter()
            .copied()
            .filter(|bucket| bucket.holds_computation(computation))
            .min()
    }

    /// Submits `batch` for execution; the submission completes with the step's output.
    ///
    /// # Errors
    ///
    /// As [`Queue::submit`]: a batch no runner serves is refused here.
    pub fn submit(&self, batch: Batch) -> Result<Submission> {
        self.queue.submit(Job::Execute(Box::new(batch)))
    }

    /// Submits the paired forwards of the shadow batch `batch` ([`Job::Shadow`]).
    ///
    /// # Errors
    ///
    /// As [`Queue::submit`].
    pub fn shadow(&self, batch: Batch) -> Result<Submission> {
        self.queue.submit(Job::Shadow {
            batch: Box::new(batch),
            compared: 0,
        })
    }

    /// Submits the capture of every bucket the runners state ([`Job::Capture`]).
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub fn warm_up(&self) -> Result<Submission> {
        self.queue.submit(Job::Capture)
    }

    /// Submits the preparation of generation `next` ([`Job::Prepare`]). A preparation that
    /// fails leaves a generation to roll back ([`Executor::rollback`]).
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub fn prepare(&self, next: Arc<Generation>) -> Result<Submission> {
        self.queue.submit(Job::Prepare(next))
    }

    /// Submits the switch to the prepared generation `number`, which takes effect between the
    /// batches submitted before and after it.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub fn switch(&self, number: u64) -> Result<Submission> {
        self.queue.submit(Job::Switch(number))
    }

    /// Submits the commit of generation `number`, which serves on trial.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub fn commit(&self, number: u64) -> Result<Submission> {
        self.queue.submit(Job::Commit(number))
    }

    /// Submits the rollback of generation `number`, on trial or prepared. Rolling back a
    /// generation this executor holds nothing of succeeds and changes nothing.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub fn rollback(&self, number: u64) -> Result<Submission> {
        self.queue.submit(Job::Rollback(number))
    }

    /// Changes the device's profiling session between two batches and returns the rank's
    /// response to the daemon's operation. It waits for the executor, so it is a work thread's
    /// call and never the control thread's.
    ///
    /// # Errors
    ///
    /// When the session refuses the change, the executor has no device, or the queue is closed.
    pub fn profile(&self, change: Change) -> Result<serde_json::Value> {
        let (response, responded) = crossbeam_channel::bounded(1);
        let submission = self.queue.submit(Job::Profile { change, response })?;

        returned(&submission, &responded)
    }

    /// Submits a wait for `events`: the submission completes once the device has passed the
    /// latest record of each, which is when storage whose release hangs on them may be released.
    /// The events are recorded by jobs submitted before this one.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub fn wait(&self, events: Vec<Arc<Event>>) -> Result<Submission> {
        self.queue.submit(Job::Wait(events))
    }

    /// Submits the control operation `operation`, named `label` in a stall report and a failure:
    /// the executor calls it between the jobs submitted before and after it, once the device
    /// has passed everything the runners issued.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub fn control(&self, label: impl Into<String>, operation: Operation) -> Result<Submission> {
        self.queue.submit(Job::Control {
            label: label.into(),
            operation,
        })
    }

    /// Runs `operation` as a control operation ([`Executor::control`]), waits for it and returns
    /// its value. A work thread's call, never the control thread's.
    ///
    /// # Errors
    ///
    /// What the operation fails with, stated with its label, or the close of the queue.
    pub fn call<T: Send + 'static>(
        &self,
        label: impl Into<String>,
        operation: impl FnOnce(&mut [Box<dyn Runner>]) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (value, produced) = crossbeam_channel::bounded(1);
        let submission = self.control(
            label,
            Box::new(move |runners| {
                // A caller that stopped waiting has nothing to be told.
                let _ = value.send(operation(runners)?);
                Ok(())
            }),
        )?;

        returned(&submission, &produced)
    }

    /// When the executor took the job in flight and the batch the job belongs to; `None` between
    /// two jobs. The control thread's watchdog reads it ([`watchdog::Watchdog`]).
    pub fn in_flight(&self) -> Option<(Instant, Option<BatchId>)> {
        self.queue.in_flight()
    }

    /// What the executor is performing, in words: the job, and the bucket, generation or operator
    /// it is at. `None` between two jobs.
    pub fn performing(&self) -> Option<String> {
        self.queue.performing()
    }

    /// The sticky error of the device's context, if it has one: an illegal memory access or a
    /// device-side assert left the context unusable, and the engine must be replaced. The
    /// control thread checks while a job has been in flight for a second, because the jobs
    /// behind such an error never complete.
    ///
    /// # Errors
    ///
    /// When the context cannot be read.
    pub fn sticky_error(&self) -> Result<Option<platform::Error>> {
        match self.device {
            Some(device) => Ok(platform::PrimaryContext::retain(device)?.sticky_error()?),
            None => Ok(None),
        }
    }

    /// The batches executed since the executor started, and the nanoseconds it spent in jobs
    /// that run a runner's forwards: batches, captures, preparations and paired forwards. A
    /// runner's `execute` returns once the device completed the step, so the time is the
    /// device's as well as the host's.
    pub fn counters(&self) -> (u64, u64) {
        (
            self.counters.steps.load(Ordering::Relaxed),
            self.counters.forward_ns.load(Ordering::Relaxed),
        )
    }

    /// What the generation update in flight has no comparison for, each with the reason: the
    /// kernels the generation adds, which nothing serving computes, and the pairs its paired
    /// forwards could not compare. Empty states that every kernel the update replaced and a
    /// shadow batch reached was compared. It is cleared when the next preparation begins.
    pub fn uncompared(&self) -> Vec<String> {
        lock(&self.uncompared).clone()
    }

    /// Closes the queue and joins the thread: the job in flight ends, and the jobs still queued
    /// end as cancelled.
    ///
    /// # Errors
    ///
    /// When the executor thread panicked.
    pub fn close(mut self) -> Result<()> {
        self.join()
    }

    fn join(&mut self) -> Result<()> {
        self.queue.close();

        match self.thread.take() {
            Some(thread) => thread
                .join()
                .map_err(|_| Error::State("the executor thread panicked")),
            None => Ok(()),
        }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        if self.join().is_err() {
            tracing::error!(device = ?self.device, "an executor thread panicked");
        }
    }
}

impl std::fmt::Debug for Executor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Executor")
            .field("device", &self.device)
            .field("queue", &self.queue)
            .finish_non_exhaustive()
    }
}

/// The value a job sent through `receiver`, once `submission` completed.
fn returned<T>(submission: &Submission, receiver: &Receiver<T>) -> Result<T> {
    match submission.wait() {
        Outcome::Success(_) => receiver
            .recv()
            .map_err(|_| Error::State("the executor's job completed without its value")),
        Outcome::Failed(fault) => Err(Error::Shared(fault)),
        Outcome::Cancelled => Err(Error::State("the executor closed before it ran the job")),
    }
}

/// The executor thread's state: the runners nothing else calls, and what the thread keeps
/// between two jobs.
struct Thread {
    device: Option<DeviceId>,
    /// The device's runners; `RunnerId(index)` names `runners[index]`.
    runners: Vec<Box<dyn Runner>>,
    queue: Arc<Queue>,
    /// The generations the runners' graphs hold, by number: the serving one, and the one being
    /// prepared or kept during a trial. Which of them serves is the graphs' to say
    /// ([`Graphs::generation`](crate::Graphs::generation)); a shadow batch reads the two it
    /// compares here.
    kernels: BTreeMap<u64, Arc<Generation>>,
    counters: Arc<Counters>,
    uncompared: Arc<Mutex<Vec<String>>>,
    /// The kernels the generation being prepared adds, which nothing serving is compared with.
    unmeasured: Vec<String>,
    /// The context's sticky error, once one was seen: every launch after it fails, so the
    /// executor issues nothing more and fails each job with this error.
    sticky: Option<platform::Error>,
    /// The profiling session that records the device, while one does.
    session: Option<Session>,
    /// This thread's driver calls that waited for the device, when its last batch output left.
    blocking: Calls,
    /// The process's count of graph renewals the runners' executables were instantiated at.
    renewals: u64,
}

impl Thread {
    /// Performs the queue's jobs until it closes.
    fn run(mut self) {
        // The device's context stays current on this thread: a native module reads the device
        // from the runtime to find its per-device handles, and a thread with no current context
        // reads device zero.
        let context = self.device.map(platform::PrimaryContext::retain);
        let current = match &context {
            Some(Ok(context)) => Some(context.enter()),
            _ => None,
        };
        if let Some(Err(error)) = &context {
            tracing::error!(device = ?self.device, %error, "the device's context is not current");
        }
        if let Some(Err(error)) = &current {
            tracing::error!(device = ?self.device, %error, "the device's context is not current");
        }

        // The driver calls this thread waits in are counted while it serves: waiting for its own
        // device's work is its design, so the figure is a record and not a fault.
        let _watched = self.device.map(|_| Watched::enter());

        while let Some(Entry {
            job,
            submission,
            submitted,
        }) = self.queue.next()
        {
            if let Some(sticky) = &self.sticky {
                // Nothing is issued to an unusable context, and the job fails with the sticky
                // error instead of a launch error that would hide it.
                self.queue
                    .complete(submission, Err(Error::Cuda(sticky.clone())));
                continue;
            }

            let counted = job.runs_forwards();
            let started = Instant::now();
            let turn = self.perform(job, started.saturating_duration_since(submitted));
            if counted {
                let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.counters
                    .forward_ns
                    .fetch_add(elapsed, Ordering::Relaxed);
            }

            match turn {
                Ok(ControlFlow::Continue(job)) => self.queue.requeue(Entry {
                    job,
                    submission,
                    submitted,
                }),
                Ok(ControlFlow::Break(output)) => self.queue.complete(submission, Ok(output)),
                Err(error) => {
                    tracing::warn!(
                        device = ?self.device,
                        %error,
                        in_flight = self.queue.performing().as_deref().unwrap_or_default(),
                        "an executor's job failed"
                    );

                    // A failure may be the first symptom of a sticky error.
                    self.check_sticky();
                    self.queue.complete(submission, Err(error));
                }
            }
        }
    }

    /// Performs one turn of `job`, which waited `queued` in the queue: the job's output when it
    /// ended, or the job's remainder, which goes back to the tail of the queue.
    fn perform(&mut self, job: Job, queued: Duration) -> Result<ControlFlow<BatchOutput, Job>> {
        let done = ControlFlow::Break(BatchOutput::default());

        match job {
            Job::Execute(batch) => Ok(ControlFlow::Break(self.execute(&batch, queued)?)),
            Job::Shadow { batch, compared } => Ok(if self.shadow(&batch, compared)? {
                ControlFlow::Continue(Job::Shadow {
                    batch,
                    compared: compared + 1,
                })
            } else {
                done
            }),
            Job::Capture => Ok(if self.capture()? {
                ControlFlow::Continue(Job::Capture)
            } else {
                done
            }),
            Job::Prepare(next) => Ok(if self.prepare(&next)? {
                ControlFlow::Continue(Job::Prepare(next))
            } else {
                done
            }),
            Job::Switch(number) => {
                self.switch(number)?;
                Ok(done)
            }
            Job::Commit(number) => {
                self.commit(number)?;
                Ok(done)
            }
            Job::Rollback(number) => {
                self.rollback(number)?;
                Ok(done)
            }
            Job::Profile { change, response } => {
                let device = self.device.ok_or(Error::State(
                    "an executor with no device records no profiling session",
                ))?;

                // A session's collectors start and stop with nothing of the device in flight.
                self.settle()?;
                let changed = Session::change(&mut self.session, change, device)?;

                // A caller that stopped waiting has nothing to be told.
                let _ = response.send(changed);
                Ok(done)
            }
            Job::Wait(events) => {
                for event in &events {
                    event.synchronize()?;
                }
                Ok(done)
            }
            Job::Control { operation, .. } => {
                self.settle()?;
                operation(&mut self.runners)?;
                Ok(done)
            }
        }
    }

    /// Executes `batch` through the runner that serves its phase.
    fn execute(&mut self, batch: &Batch, queued: Duration) -> Result<BatchOutput> {
        let index = self.runner_of(batch.phase)?;
        let step = batch.id.0;
        let started = Instant::now();

        // A session's window is made of whole steps: its collectors are armed and disarmed
        // between two steps, with nothing of the device in flight.
        let stage = batch.shape.variant.stage();
        if self
            .session
            .as_mut()
            .is_some_and(|session| session.begin(step, stage))
        {
            self.apply_session()?;
        }
        self.renew()?;

        // The step's range markers are launched by the graphs, around the forward itself.
        self.runners[index].graphs().mark_step(step, stage);
        let mut output = self.runners[index].execute(batch)?;

        if self
            .session
            .as_mut()
            .is_some_and(|session| session.end(step))
        {
            self.apply_session()?;
        }

        // The runner states the step's device and host time; the wait in the queue is the
        // executor's to state.
        output.timing.queued_us = u64::try_from(queued.as_micros()).unwrap_or(u64::MAX);
        if let Some(device) = self.device {
            output.forward_stats =
                profiling::forward_stats(device, self.session.as_mut(), &mut self.blocking);
        }
        self.counters.steps.fetch_add(1, Ordering::Relaxed);

        // A media call is seconds of work a few times per request, so one line each is the
        // request's timeline; a token batch's would be the log.
        if let Some(media) = &batch.media {
            tracing::info!(
                device = ?self.device,
                call = ?media.computation.call,
                batch = step,
                queued_ms = queued.as_millis() as u64,
                total_ms = started.elapsed().as_millis() as u64,
                "a media batch was executed"
            );
        }
        Ok(output)
    }

    /// One paired forward of the shadow batch `batch`, `compared` of whose changed operators
    /// were compared in earlier turns. Returns whether an operator is left for a later turn.
    fn shadow(&mut self, batch: &Batch, compared: usize) -> Result<bool> {
        let index = self.runner_of(batch.phase)?;
        if !self.kernels.contains_key(&batch.generation) {
            return Err(Error::Invariant(format!(
                "shadow batch {} names generation {}, which this executor is not preparing",
                batch.id.0, batch.generation
            )));
        }
        let runner = &mut self.runners[index];

        // The inputs are prepared again for each paired forward, which issues them; a paired
        // forward writes scratch memory only, which the runner's shadow inputs state.
        let bucket = runner.prepare_shadow_inputs(batch)?;
        let operators = runner.graphs().replaced(&bucket);
        let Some(operator) = operators.get(compared).copied() else {
            return Ok(false);
        };

        self.queue.note(format!(
            "shadowing `{operator}` of generation {} in bucket {bucket} on batch {} (runner {})",
            batch.generation, batch.id.0, index
        ));
        let elements = runner.graphs().shadow(&**runner, &bucket, operator)?;
        tracing::info!(
            device = ?self.device,
            runner = index,
            operator,
            %bucket,
            generation = batch.generation,
            elements,
            "a replaced kernel was compared with the one it replaces"
        );

        self.publish_uncompared();
        Ok(compared + 1 < operators.len())
    }

    /// Captures the first bucket a runner states and has no graph of. Returns whether another is
    /// left for a later turn.
    fn capture(&mut self) -> Result<bool> {
        // The runners in their order, and each runner's buckets in its own: largest first.
        let mut uncaptured = Vec::new();
        for (index, runner) in self.runners.iter().enumerate() {
            let captured = runner.graphs().buckets();
            uncaptured.extend(
                runner
                    .buckets()
                    .into_iter()
                    .filter(|bucket| !captured.contains(bucket))
                    .map(|bucket| (index, bucket)),
            );
        }
        let Some(&(index, bucket)) = uncaptured.first() else {
            self.set_warming(false);
            return Ok(false);
        };
        let more = uncaptured.len() > 1;

        // Every runner's forwards are a warm-up's while a bucket is left, so the members of an
        // expert union capture in step; a batch served between two turns runs with it set.
        self.set_warming(true);

        // A capture is long device work with no batch to show for it: the line that it started,
        // with none that it finished, is the shape a stuck warm-up has.
        self.queue.note(format!(
            "warming up: capturing bucket {bucket} of runner {index}"
        ));
        tracing::info!(device = ?self.device, runner = index, %bucket, "a bucket's capture starts");

        let started = Instant::now();
        let runner = &self.runners[index];
        let captured = runner.graphs().capture(&**runner, &bucket);

        // The warm-up ends with its last capture, or with the capture that failed it.
        if captured.is_err() || !more {
            self.set_warming(false);
        }
        captured?;

        tracing::info!(
            device = ?self.device,
            runner = index,
            %bucket,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "the bucket's capture finished"
        );
        Ok(more)
    }

    /// States `warming` on every runner ([`Runner::set_warming`]).
    fn set_warming(&self, warming: bool) {
        for runner in &self.runners {
            runner.set_warming(warming);
        }
    }

    /// One turn of the preparation of `next`: the first begins it on every runner, and each
    /// later one prepares one bucket's graph. Returns whether a bucket is left for a later turn.
    fn prepare(&mut self, next: &Arc<Generation>) -> Result<bool> {
        let number = next.number();
        let began = self.kernels.contains_key(&number)
            && self
                .runners
                .iter()
                .all(|runner| runner.graphs().generation() != number);

        if began {
            if let Some((index, bucket)) = self.pending() {
                self.queue.note(format!(
                    "preparing generation {number}: bucket {bucket} of runner {index}"
                ));

                let runner = &self.runners[index];
                let done = runner.graphs().prepare_bucket(&**runner, &bucket)?;
                tracing::info!(
                    device = ?self.device,
                    runner = index,
                    %bucket,
                    generation = number,
                    ?done,
                    "a bucket's graph was prepared for a generation"
                );
            }
            return Ok(self.pending().is_some());
        }

        // The device prepares a generation on all of its runners or on none: a runner that
        // refuses it ends the preparation of the runners before it.
        for (index, runner) in self.runners.iter().enumerate() {
            let Err(error) = runner.graphs().prepare(&**runner, Arc::clone(next)) else {
                continue;
            };

            for earlier in &self.runners[..index] {
                if let Err(error) = earlier.graphs().rollback(number) {
                    tracing::warn!(%error, generation = number, "a refused preparation was not ended");
                }
            }
            return Err(error);
        }

        // The update's evidence starts empty, but for the kernels it adds: nothing serving
        // computes them, so no paired forward will compare them.
        let serving = self
            .runners
            .first()
            .and_then(|runner| self.kernels.get(&runner.graphs().generation()));
        self.unmeasured = serving.map_or_else(Vec::new, |serving| added_kernels(serving, next));
        self.kernels.insert(number, Arc::clone(next));
        self.publish_uncompared();

        Ok(self.pending().is_some())
    }

    /// Switches every runner to the prepared generation `number`.
    fn switch(&mut self, number: u64) -> Result<()> {
        // Checked for every runner before any switches, so the runners serve one generation.
        if let Some((index, bucket)) = self.pending() {
            return Err(Error::Invariant(format!(
                "generation {number} is not prepared: bucket {bucket} of runner {index} has no \
                 graph of it"
            )));
        }

        for runner in &self.runners {
            runner.graphs().switch(number)?;
        }
        Ok(())
    }

    /// Commits the trial of generation `number` on every runner, and drops the generations it
    /// replaced.
    fn commit(&mut self, number: u64) -> Result<()> {
        let committed = self.on_every_runner(|runner| runner.graphs().commit(number));

        if committed.is_ok() {
            self.kernels.retain(|held, _| *held >= number);
        }
        committed
    }

    /// Rolls generation `number` back on every runner.
    fn rollback(&mut self, number: u64) -> Result<()> {
        // A preparation that a runner refused left nothing of the generation here, and the
        // engine rolls a failed update back on every device alike.
        if !self.kernels.contains_key(&number) {
            return Ok(());
        }

        let rolled_back = self.on_every_runner(|runner| runner.graphs().rollback(number));

        // The generation's kernels go with its graphs, unless it is the one that still serves:
        // a rollback the graphs refused.
        let serves = self
            .runners
            .iter()
            .any(|runner| runner.graphs().generation() == number);
        if !serves {
            self.kernels.remove(&number);
            self.unmeasured.clear();
            self.publish_uncompared();
        }
        rolled_back
    }

    /// Calls `operation` on every runner, whatever an earlier one returned, so a failure on one
    /// does not leave the others holding what the operation releases. Returns the first error.
    fn on_every_runner(&self, operation: impl Fn(&dyn Runner) -> Result<()>) -> Result<()> {
        let mut first = Ok(());

        for runner in &self.runners {
            let result = operation(&**runner);
            if first.is_ok() {
                first = result;
            }
        }
        first
    }

    /// The first bucket the generation being prepared still needs a graph of, with its runner:
    /// the runners in their order and each runner's buckets largest first.
    fn pending(&self) -> Option<(usize, Bucket)> {
        self.runners.iter().enumerate().find_map(|(index, runner)| {
            let graphs = runner.graphs();

            runner
                .buckets()
                .into_iter()
                .find(|bucket| graphs.pending(bucket).is_some())
                .map(|bucket| (index, bucket))
        })
    }

    /// States what the update in flight has no comparison for, for [`Executor::uncompared`].
    fn publish_uncompared(&self) {
        let mut uncompared = self.unmeasured.clone();
        for runner in &self.runners {
            uncompared.extend(runner.graphs().uncompared());
        }

        *lock(&self.uncompared) = uncompared;
    }

    /// Waits for the device to pass the last forward of every runner's graphs: what a job that
    /// needs nothing in flight on the device starts with.
    fn settle(&self) -> Result<()> {
        for runner in &self.runners {
            runner.graphs().wait()?;
        }
        Ok(())
    }

    /// Changes the session's collectors as its window asks, with nothing of the device in
    /// flight.
    fn apply_session(&mut self) -> Result<()> {
        self.settle()?;

        if let Some(session) = self.session.as_mut() {
            session.apply();
        }
        Ok(())
    }

    /// Instantiates the runners' graph executables again when the process declared them stale
    /// since they were instantiated: CUPTI changed how it records a graph's launch, or storage
    /// the graphs address was mapped again.
    fn renew(&mut self) -> Result<()> {
        let renewals = profiling::graph_renewals();
        if renewals == self.renewals {
            return Ok(());
        }

        self.settle()?;
        for runner in &self.runners {
            runner.graphs().renew()?;
        }
        self.renewals = renewals;
        Ok(())
    }

    /// Reads the device's context for a sticky error and keeps the first one found.
    fn check_sticky(&mut self) {
        let Some(device) = self.device else {
            return;
        };

        let sticky =
            platform::PrimaryContext::retain(device).and_then(|context| context.sticky_error());
        if let Ok(Some(sticky)) = sticky {
            tracing::error!(
                device = device.0,
                %sticky,
                "the device's context has a sticky error; no further job is issued"
            );
            self.sticky = Some(sticky);
        }
    }

    /// The runner that serves `phase`.
    fn runner_of(&self, phase: PhaseId) -> Result<usize> {
        self.runners
            .iter()
            .position(|runner| runner.phases().contains(&phase))
            .ok_or_else(|| {
                Error::Invariant(format!(
                    "no runner of this executor serves phase {}",
                    phase.0
                ))
            })
    }
}

/// The kernels `next` holds under keys `serving` has none for, each as a statement of what the
/// update leaves unmeasured, in a stable order.
fn added_kernels(serving: &Generation, next: &Generation) -> Vec<String> {
    let mut added: Vec<String> = next
        .iter()
        .filter(|(operator, key, _)| !serving.contains(operator, key))
        .map(|(operator, key, _)| {
            if serving.keys(operator).next().is_none() {
                format!(
                    "operator `{operator}` is new in generation {}: the serving generation \
                     holds no kernel of it, so its `{key}` kernel is unmeasured",
                    next.number()
                )
            } else {
                format!(
                    "`{operator}` `{key}` is new in generation {}: the serving generation \
                     launches the operator under other keys and none of them computes this one, \
                     so this kernel is unmeasured",
                    next.number()
                )
            }
        })
        .collect();

    added.sort();
    added
}

#[cfg(test)]
mod tests;
