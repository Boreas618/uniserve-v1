//! A worker's capture thread, and what it has made ready.
//!
//! Every worker of a device starts one capture thread beside its own. The thread holds two
//! instances of each runner's forward (`Runner::capturer`): one over graphs beside the runner's
//! ([`Graphs::beside`]) and one over a rehearsal's graphs ([`Graphs::rehearsal`]). It is handed a
//! generation's buckets in order ([`Work`]) at warm-up and when a generation update is prepared,
//! and for each bucket in turn:
//!
//! 1. examines it: a profile run of its forward, which states whether it keeps its graph or is
//!    captured under the generation's kernels, and whether it launches a native kernel the
//!    serving generation does not hold ([`Graphs::examine`]);
//! 2. rehearses it when it does: its forward runs once over idle inputs on a stream of its own,
//!    in memory of its own, so the kernel's first use happens outside any capture
//!    ([`Graphs::rehearse`]);
//! 3. captures it on the capture stream and instantiates the graph ([`Graphs::capture`]);
//! 4. hands the outcome to its worker as a job (`Job::Captured`), which installs it in the
//!    runner's graphs between two batches ([`Graphs::install`]) and states the bucket ready
//!    ([`Readiness`]).
//!
//! A generation of other graph options hands, with its buckets, forward instances of the shapes
//! it prepared (`Runner::reshaped_capturer`), which capture them in place of the runners' own for
//! that work; once the shapes committed, the worker gives the thread instances of them, which
//! capture every later generation ([`Capturer::renew`]).
//!
//! The thread touches nothing of the runners' graphs, which the worker thread alone does: it
//! captures over its own instances, and a graph reaches the runner only through the worker's
//! queue. Its device work is a rehearsal's, on streams no serving forward uses. A capture records
//! and runs nothing.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crossbeam_channel::Sender;
use ops::Generation;
use platform::DeviceId;
use service::lock;

use crate::batch::BatchOutput;
use crate::runner::Forward;
use crate::runner::buckets::Bucket;
use crate::runner::graphs::{Graph, Plan};
use crate::worker::completion::{Fault, Outcome, Submission, SubmissionId, SubmissionState};
use crate::worker::queue::Queue;
use crate::worker::{Error, Job, Result};

/// A bucket of one of a worker's runners: the runner's index among the worker's, and the bucket.
/// The ranks of a group hold the same runners in the same order, each stating the same buckets,
/// so a key names one bucket on every rank.
pub(crate) type Key = (usize, Bucket);

/// A runner's two forward instances on the capture thread: the capture instance and the
/// rehearsal instance (`Runner::capturer`).
pub(crate) type Pair = (Box<dyn Forward>, Box<dyn Forward>);

/// Forward instances of runners, by each runner's index among its worker's.
pub(crate) type Instances = BTreeMap<usize, Pair>;

/// A launch table, as a graph's serving table is handed to the capture thread.
pub(crate) type Table = Vec<(&'static str, String, String)>;

/// A generation's buckets handed to the capture thread.
pub(crate) struct Work {
    /// The generation the buckets are captured under.
    pub(crate) kernels: Arc<Generation>,
    /// The generation that serves beside it, `None` at warm-up, when no bucket has a graph.
    pub(crate) serving: Option<Arc<Generation>>,
    /// The buckets in the order they are examined, each with the launch table of the graph it
    /// serves from (`None` at warm-up, and for a bucket of a shape the generation prepared).
    pub(crate) buckets: Vec<(Key, Option<Table>)>,
    /// The forward instances of the shapes the generation prepared, which capture their
    /// runners' buckets in place of the thread's own for this work.
    pub(crate) instances: Instances,
    /// Whether every bucket is captured again, whatever its examination finds: CUPTI samples
    /// the generation's graphs otherwise than the serving ones.
    pub(crate) recapture: bool,
    /// Set once the generation was rolled back: the thread examines no further bucket of it.
    pub(crate) cancelled: Arc<AtomicBool>,
}

/// What a worker hands its capture thread, in order.
enum Handed {
    /// A generation's buckets to capture.
    Work(Work),
    /// The forward instances a runner's buckets are captured over from here on: those of the
    /// shape it serves once a generation that reshaped it committed.
    Renew(usize, Pair),
}

/// A worker's capture thread.
pub(crate) struct Capturer {
    work: Option<Sender<Handed>>,
    thread: Option<JoinHandle<()>>,
}

impl Capturer {
    /// Starts the capture thread of `device` (`None` for a worker of no device) over
    /// `instances`, each runner's capture instance and rehearsal instance in the worker's order
    /// of runners, handing what it makes to the worker through `queue`.
    ///
    /// # Errors
    ///
    /// When the thread cannot start.
    pub(crate) fn spawn(
        device: Option<DeviceId>,
        instances: Instances,
        queue: Arc<Queue>,
    ) -> Result<Self> {
        let (work, handed) = crossbeam_channel::unbounded::<Handed>();
        let mut instances = instances;
        let name = match device {
            Some(device) => format!("capture-{}", device.0),
            None => "capture-host".to_owned(),
        };

        let thread = std::thread::Builder::new()
            .name(name)
            .spawn(move || {
                // The device's context stays current on the thread, as on its worker's: a native
                // module reads the device from the runtime.
                let context = device.map(platform::PrimaryContext::retain);
                let _current = match &context {
                    Some(Ok(context)) => context.enter().ok(),
                    _ => None,
                };

                for handed in handed {
                    match handed {
                        Handed::Work(work) => capture(device, &instances, &queue, &work),
                        Handed::Renew(runner, pair) => {
                            instances.insert(runner, pair);
                        }
                    }
                }
            })
            .map_err(|error| Error::transport_source("starting a capture thread", error))?;

        Ok(Self {
            work: Some(work),
            thread: Some(thread),
        })
    }

    /// Hands `work` to the thread, behind the work handed before it.
    ///
    /// # Errors
    ///
    /// When the thread has stopped.
    pub(crate) fn submit(&self, work: Work) -> Result<()> {
        self.send(Handed::Work(work))
    }

    /// Has the thread capture runner `runner`'s buckets over `pair` from here on, behind the
    /// work handed before.
    ///
    /// # Errors
    ///
    /// When the thread has stopped.
    pub(crate) fn renew(&self, runner: usize, pair: Pair) -> Result<()> {
        self.send(Handed::Renew(runner, pair))
    }

    fn send(&self, handed: Handed) -> Result<()> {
        self.work
            .as_ref()
            .and_then(|sender| sender.send(handed).ok())
            .ok_or(Error::State("the worker's capture thread has stopped"))
    }
}

impl Drop for Capturer {
    fn drop(&mut self) {
        drop(self.work.take());
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            tracing::error!("a capture thread panicked");
        }
    }
}

/// Examines, rehearses and captures each bucket of `work` in order, and hands each outcome to
/// the worker. The first failure is handed too, and ends the work: no later bucket of the
/// generation is examined.
fn capture(device: Option<DeviceId>, instances: &Instances, queue: &Queue, work: &Work) {
    let generation = work.kernels.number();

    for ((runner, bucket), held) in &work.buckets {
        if work.cancelled.load(Ordering::Acquire) {
            break;
        }

        let outcome = match work.instances.get(runner).or_else(|| instances.get(runner)) {
            Some((capture, rehearsal)) => bucket_outcome(
                device,
                (&**capture, &**rehearsal),
                *runner,
                bucket,
                held,
                work,
            ),
            None => Err(Error::Invariant(format!(
                "generation {generation} names runner {runner}, and the worker holds {}",
                instances.len()
            ))),
        };
        let failed = outcome.is_err();

        let job = Job::Captured {
            generation,
            key: (*runner, *bucket),
            outcome,
        };
        if queue.submit(job).is_err() || failed {
            break;
        }
    }

    // A rehearsal's memory is held only while its generation is captured.
    for (_, rehearsal) in instances.values().chain(work.instances.values()) {
        rehearsal.graphs().release_rehearsal();
    }
}

/// The outcome of one bucket: its graph under `work`'s generation, or `None` when it keeps the
/// graph it serves from.
fn bucket_outcome(
    device: Option<DeviceId>,
    (capture, rehearsal): (&dyn Forward, &dyn Forward),
    runner: usize,
    bucket: &Bucket,
    held: &Option<Table>,
    work: &Work,
) -> Result<Option<Graph>> {
    let generation = work.kernels.number();
    let began = Instant::now();

    let plan: Plan = capture.graphs().examine(
        capture,
        bucket,
        held.as_deref(),
        work.serving.as_deref(),
        &work.kernels,
    )?;
    let examined = began.elapsed();
    if !plan.capture && !work.recapture {
        return Ok(None);
    }

    let began = Instant::now();
    if plan.rehearse {
        rehearsal
            .graphs()
            .rehearse(rehearsal, bucket, &work.kernels)?;
    }
    let rehearsed = began.elapsed();

    let graph = capture.graphs().capture(capture, bucket, &work.kernels)?;
    let costs = graph.costs();
    tracing::info!(
        device = ?device,
        runner,
        %bucket,
        generation,
        examine_ms = examined.as_secs_f64() * 1e3,
        rehearse_ms = rehearsed.as_secs_f64() * 1e3,
        capture_ms = costs.capture.as_secs_f64() * 1e3,
        instantiate_ms = costs.instantiate.as_secs_f64() * 1e3,
        graph_bytes = costs.bytes,
        segments = costs.segments,
        kernel_nodes = costs.kernel_nodes,
        "a bucket was captured beside the serving graphs"
    );
    Ok(Some(graph))
}

/// What a worker's capture thread has made ready, by generation: the buckets each generation
/// examines, those whose outcome the worker installed, and who waits for them
/// ([`Readiness::wait`]).
#[derive(Default)]
pub(crate) struct Readiness {
    generations: Mutex<BTreeMap<u64, Progress>>,
}

/// One generation's readiness on a worker.
#[derive(Default)]
struct Progress {
    /// Every bucket the generation examines, in order, once its preparation began.
    order: Option<Vec<Key>>,
    /// The buckets whose outcome the worker installed.
    ready: BTreeSet<Key>,
    /// Why the generation will not be ready: a capture that failed, or a rollback.
    failed: Option<String>,
    /// The submissions waiting for buckets of the generation, each with the buckets it waits
    /// for (`None` for every one).
    waiters: Vec<(Option<Vec<Key>>, Arc<SubmissionState>)>,
}

impl Progress {
    /// How a wait for `keys` ends now: `None` while it goes on. A key the generation does not
    /// examine is a bucket this worker holds no graph of, which is ready.
    fn settled(&self, keys: Option<&[Key]>) -> Option<std::result::Result<(), String>> {
        if let Some(failed) = &self.failed {
            return Some(Err(failed.clone()));
        }

        let order = self.order.as_ref()?;
        let ready = match keys {
            Some(keys) => keys
                .iter()
                .all(|key| self.ready.contains(key) || !order.contains(key)),
            None => order.iter().all(|key| self.ready.contains(key)),
        };
        ready.then_some(Ok(()))
    }

    /// Takes the waiters whose wait ended, with how.
    fn settle(&mut self) -> Vec<(Arc<SubmissionState>, std::result::Result<(), String>)> {
        let mut settled = Vec::new();
        let mut waiting = Vec::new();

        for (keys, state) in std::mem::take(&mut self.waiters) {
            match self.settled(keys.as_deref()) {
                Some(outcome) => settled.push((state, outcome)),
                None => waiting.push((keys, state)),
            }
        }
        self.waiters = waiting;
        settled
    }
}

impl Readiness {
    /// The preparation of generation `generation` began: it examines the buckets of `order`, in
    /// that order.
    pub(crate) fn begin(&self, generation: u64, order: Vec<Key>) {
        self.change(generation, |progress| progress.order = Some(order));
    }

    /// The worker installed the outcome of `key` for generation `generation`.
    pub(crate) fn ready(&self, generation: u64, key: Key) {
        self.change(generation, |progress| {
            progress.ready.insert(key);
        });
    }

    /// Generation `generation` will not be ready, for `reason`: every wait for it fails.
    pub(crate) fn fail(&self, generation: u64, reason: String) {
        self.change(generation, |progress| {
            progress.failed.get_or_insert(reason);
        });
    }

    /// Forgets every generation up to `generation`, once its update ended: a later wait for one
    /// of them waits for a preparation that does not come.
    pub(crate) fn forget(&self, generation: u64) {
        lock(&self.generations).retain(|held, _| *held > generation);
    }

    /// The buckets generation `generation` examines, in order; `None` before its preparation
    /// began.
    pub(crate) fn order(&self, generation: u64) -> Option<Vec<Key>> {
        lock(&self.generations)
            .get(&generation)
            .and_then(|progress| progress.order.clone())
    }

    /// A submission that completes once the worker installed the outcome of every bucket of
    /// `keys` for generation `generation` (every bucket it examines when `keys` is `None`), and
    /// fails once the generation will not be ready. A bucket the generation does not examine is
    /// ready. A wait for a generation whose preparation has not begun waits for it to begin.
    pub(crate) fn wait(&self, generation: u64, keys: Option<Vec<Key>>) -> Submission {
        let (submission, state) = Submission::pending(SubmissionId(generation));

        let settled = {
            let mut generations = lock(&self.generations);
            let progress = generations.entry(generation).or_default();
            progress.waiters.push((keys, state));
            progress.settle()
        };
        finish(settled);
        submission
    }

    /// Changes generation `generation`'s progress and completes the waits that ended.
    fn change(&self, generation: u64, change: impl FnOnce(&mut Progress)) {
        let settled = {
            let mut generations = lock(&self.generations);
            let progress = generations.entry(generation).or_default();
            change(progress);
            progress.settle()
        };
        finish(settled);
    }
}

/// Completes `settled` waits, outside the readiness's lock: completing one calls its wakers.
fn finish(settled: Vec<(Arc<SubmissionState>, std::result::Result<(), String>)>) {
    for (state, outcome) in settled {
        let outcome = match outcome {
            Ok(()) => Outcome::Success((BatchOutput::default(), Vec::new())),
            Err(reason) => {
                let fault: Arc<Fault> = Arc::new(Error::Invalid(reason));
                Outcome::Failed(fault)
            }
        };
        state.finish(outcome);
    }
}
