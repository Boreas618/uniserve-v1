//! The mechanism's one door to execution.
//!
//! The step submits a batch, waits for results, and asks what the pool can do;
//! it does not know what runs the batch (design §2.4). This module is the door:
//! every path of the mechanism that executes something comes through it.
//!
//! A serving scheduler's door is `::worker::Batches` (2.4, 4.4): the mechanism's
//! own completion type is filled from `::worker::batch::BatchOutput` there, and
//! nothing of the wire is involved. A test supplies a simulated worker through
//! the `Worker` trait instead; that arm is built only with the `testing`
//! feature, so a serving build holds the first alone.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use crate::WorkerGroupId;
use crate::completion::BatchReport;
use crate::worker::ExecutionBatch;
#[cfg(feature = "testing")]
use crate::worker::Worker;
use crate::worker::{WorkerError, WorkerInfo, WorkerSubmitError};
use ::worker::batch::{
    Batch as DeviceBatch, BatchId as DeviceBatchId, Batches, CallId as DeviceCallId, Submission,
};
use ::worker::completion::Outcome;
use ipc::BatchCommand;

/// What dispatch and the views ask of the world: the pools it serves, and
/// which of them can take a submission now.
///
/// The step places calls on every path — a planned call, a media call, a
/// request's first call, a prefill target — and none of that submits anything.
/// Stating it as this trait is what lets one dispatch serve the in-process
/// worker and a simulated one: what differs between them is who answers, not
/// what the step asks.
pub(super) trait World {
    /// The pools, their ranks and their capabilities.
    fn info(&self) -> &WorkerInfo;

    /// Whether `worker` has an unclaimed submission slot.
    fn has_capacity(&self, worker: &WorkerGroupId) -> bool;
}

/// How often the waiting thread looks at the batches in flight.
const REPORT_POLL: std::time::Duration = std::time::Duration::from_micros(50);

/// What the mechanism executes through.
///
/// Public because a scheduler is built with one: `Scheduler::new` takes a
/// `Submissions`, made with `over_batches` over a process's own devices, or
/// with `new` over a simulated worker.
pub struct Submissions {
    path: Path,
}

/// Which worker serves this scheduler's batches.
enum Path {
    /// A worker a test supplies.
    #[cfg(feature = "testing")]
    Supplied(Box<dyn Worker>),
    /// The device worker of this process (2.4). Boxed: the arm holds the
    /// batch path, the waiting thread's two channels and the shadows.
    InProcess(Box<InProcess>),
}

/// The mechanism over `::worker::Batches`: this process's own devices.
///
/// A batch is converted to the worker's own description of it — the shape it
/// pads to, the host columns the kernels read, the generation whose kernels
/// serve it — and submitted. The completion comes back on a waiting thread, as
/// the mechanism's own type: the scheduler thread must never block on a
/// device's completion, because the policy is asked once per step whether or
/// not a device has answered.
struct InProcess {
    batches: Arc<dyn Batches>,
    info: WorkerInfo,
    /// The kernels this step's batches serve on. The scheduler reads its
    /// generation cell once per step and stamps every batch of that step, so a
    /// switchover mid-step cannot split one step across two worlds.
    generation: u64,
    /// Completed reports, in the order the devices finished them.
    ready: crossbeam_channel::Receiver<Result<BatchReport, WorkerError>>,
    /// The sending end of `nudged`, which `Submissions::waker` hands out.
    nudge: crossbeam_channel::Sender<()>,
    /// Ends a park that no result would end (`Submissions::waker`).
    nudged: crossbeam_channel::Receiver<()>,
    /// Submissions still to be waited on, handed to the waiting thread.
    queue: crossbeam_channel::Sender<Pending>,
    /// Shadow submissions in flight (7.3 step 2).
    ///
    /// A shadow batch's paired forwards compare themselves on the device, and
    /// the comparison's verdict is this submission's outcome - no other path
    /// carries it to the update. The scheduler reads them in `shadow_status`,
    /// which is why a submission is kept until it has finished.
    shadows: Vec<Submission>,
    /// The pools each unresolved batch holds a slot of, by batch.
    ///
    /// A pool admits `queue_depth` batches at once, and that bound is what makes a step a batch:
    /// a request becomes ready when its own batch ends, so a mechanism that refills at once sends
    /// it alone, the batch after it carries one row again, and requests that run together never
    /// meet in a step. While the slots are taken the ready requests wait for each other.
    unresolved: HashMap<u64, Vec<WorkerGroupId>>,
    /// How many batches hold a slot of each pool.
    occupied: HashMap<WorkerGroupId, usize>,
}

/// The state of the shadow submissions an update's pass asked for (7.3 step 2).
///
/// The pass is answered by the comparison's verdict, not by the fact that the
/// batch was submitted: a paired forward that failed on the device is the
/// update's failure, and answering earlier let a wrong kernel reach the
/// switchover.
pub(super) enum ShadowStatus {
    /// At least one shadow batch has not finished.
    Pending,
    /// Every shadow batch finished and every pair was within tolerance.
    Settled,
    /// A shadow batch failed, with the reason it failed for.
    Failed(String),
}

/// One submitted batch, with what the report's calls name.
struct Pending {
    batch: u64,
    submission: Submission,
    /// Index for index with the batch's calls: the mechanism's name for the
    /// call the worker numbers by its index.
    calls: Vec<(ipc::RequestKey, ipc::CallId)>,
}

#[cfg_attr(
    not(feature = "testing"),
    allow(
        unused_variables,
        reason = "the arms that read them are a supplied worker's"
    )
)]
impl Submissions {
    /// The mechanism over a worker a test supplies.
    #[cfg(feature = "testing")]
    pub fn new(worker: Box<dyn Worker>) -> Self {
        Self {
            path: Path::Supplied(worker),
        }
    }

    /// The mechanism over this process's own devices (2.4).
    ///
    /// The pools come from the worker's own inventory and nowhere else: an
    /// worker that states no pool is refused here rather than served with a
    /// guessed one, for the same reason `WorkerInfo::from_inventory` refuses
    /// a pool that states no rank or no longest sequence — a mechanism that
    /// assumes a rank's cache makes the host KV path disappear at runtime.
    pub fn over_batches(batches: Arc<dyn Batches>) -> anyhow::Result<Submissions> {
        let info = WorkerInfo::from_inventory(batches.inventory())?;
        anyhow::ensure!(
            !info.workers.is_empty(),
            "the in-process worker states no pool to schedule against"
        );
        // A batch carries the columns the kernels read, and every KV view is
        // shaped to the longest sequence the deployment was built for: without
        // that number no batch is servable, so it is a fact of the boot and not
        // something to discover at the first step as "nothing was dispatched"
        // (design §4.3's layout, §4.4's hard limits).
        anyhow::ensure!(
            info.max_model_len > 0,
            "the in-process worker states no longest sequence; a policy cannot be shaped to a \
             KV view without one"
        );
        let (queue, pending) = crossbeam_channel::unbounded::<Pending>();
        let (reports, ready) = crossbeam_channel::unbounded();
        // One slot: a nudge that finds one waiting has nothing to add.
        let (nudge, nudged) = crossbeam_channel::bounded(1);
        std::thread::Builder::new()
            .name("sched-in-process-wait".into())
            .spawn(move || {
                // One waiting thread for the whole epoch: a submission's
                // outcome is a device's, and the scheduler thread reads only
                // what this thread has already collected. The devices finish
                // in their own order, so a batch is reported when it ends: a
                // report held behind an earlier batch on another device is a
                // next call that many milliseconds late.
                let mut held: Vec<Pending> = Vec::new();
                loop {
                    if held.is_empty() {
                        match pending.recv() {
                            Ok(next) => held.push(next),
                            Err(_) => break,
                        }
                    }
                    held.extend(pending.try_iter());
                    let mut index = 0;
                    let mut reported = false;
                    while index < held.len() {
                        let Some(outcome) = held[index].submission.outcome() else {
                            index += 1;
                            continue;
                        };
                        let done = held.remove(index);
                        let report = BatchReport::from_output(done.batch, outcome, &done.calls);
                        if reports.send(report).is_err() {
                            return;
                        }
                        reported = true;
                    }
                    if !reported && !held.is_empty() {
                        std::thread::sleep(REPORT_POLL);
                    }
                }
            })
            .map_err(|error| anyhow::anyhow!("starting the in-process waiting thread: {error}"))?;
        Ok(Submissions {
            path: Path::InProcess(Box::new(InProcess {
                batches,
                info,
                generation: 0,
                ready,
                nudge,
                nudged,
                queue,
                shadows: Vec::new(),
                unresolved: HashMap::new(),
                occupied: HashMap::new(),
            })),
        })
    }

    /// What ends this scheduler's park when something other than a batch's
    /// result arrives: a command, or a reader that freed room in an event
    /// channel.
    ///
    /// The in-process arm parks on its results, so without this a command to an
    /// idle scheduler waits out the park's whole liveness bound: a request
    /// admitted that much later, and every stage of an update that much slower.
    pub fn waker(&self) -> crate::CommandWaker {
        match &self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(_) => crate::CommandWaker::noop(),
            Path::InProcess(arm) => {
                let nudge = arm.nudge.clone();
                crate::CommandWaker::new(move || {
                    let _ = nudge.try_send(());
                })
            }
        }
    }

    /// States the generation this step's batches serve on (4.2, 7.3).
    ///
    /// The scheduler calls this once per step, before it dispatches anything:
    /// every batch of one step runs on one world's kernels.
    pub(super) fn set_generation(&mut self, generation: u64) {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(_) => {}
            Path::InProcess(arm) => arm.generation = generation,
        }
    }

    /// The pool's capabilities: its workers, components, pools, and limits.
    pub(super) fn info(&self) -> &WorkerInfo {
        match &self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.info(),
            Path::InProcess(arm) => &arm.info,
        }
    }

    /// Whether every rank of the pool `group` is ready to accept execution.
    pub(super) fn is_ready(&self, group: &WorkerGroupId) -> bool {
        match &self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.is_ready(group),
            // The arm serves the pools its inventory states and no others.
            Path::InProcess(arm) => arm.info.workers.iter().any(|(id, _)| id == group),
        }
    }

    /// Whether the pool `group` executes nothing further.
    pub(super) fn is_closed(&self, group: &WorkerGroupId) -> bool {
        match &self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.is_closed(group),
            // A device of this process is closed when the process ends.
            Path::InProcess(_) => false,
        }
    }

    /// Whether the pool `group` has an unclaimed destination submission slot.
    pub(super) fn has_capacity(&self, group: &WorkerGroupId) -> bool {
        match &self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.has_capacity(group),
            Path::InProcess(arm) => arm.has_room(group),
        }
    }

    /// Whether `command` can be handed to the pool now.
    pub(super) fn command_has_capacity(&self, command: &BatchCommand) -> bool {
        match &self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.command_has_capacity(command),
            // A lifecycle command travels inside the batch it belongs to.
            Path::InProcess(_) => true,
        }
    }

    /// Submits one logical batch without blocking for capacity.
    pub(super) fn submit(&mut self, batch: ExecutionBatch) -> Result<(), WorkerSubmitError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.submit(batch),
            Path::InProcess(arm) => arm.submit(batch),
        }
    }

    /// The bucket this worker would serve `computation` in, when it serves one (2.4).
    ///
    /// The mechanism asks before it forms a media batch, because the batch states the extent its
    /// kernels were captured for and only the worker's own plan knows it: a bucket planned for
    /// one canvas serves no other, and deriving the extent in the mechanism would be a second
    /// statement of the runner's plan. `None` is the answer for a computation no device of this
    /// rank serves, and for a supplied worker, which serves media without the mechanism's own
    /// batch assembly.
    pub(super) fn media(
        &self,
        phase: ipc::PhaseId,
        computation: &::worker::plan::Computation,
    ) -> Option<::worker::plan::Bucket> {
        match &self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(_) => None,
            Path::InProcess(arm) => arm.batches.media(phase, computation).ok().flatten(),
        }
    }

    /// Whether this door serves the devices of this process.
    ///
    /// Only those shadow a batch: a scheduler over a supplied worker refuses
    /// the update's ask at command time rather than accepting one nothing will
    /// run (7.3 step 2).
    pub(super) fn serves_in_process(&self) -> bool {
        matches!(self.path, Path::InProcess(_))
    }

    /// Runs `batch`'s paired forwards for `generation` too, without serving
    /// them (7.3 step 2).
    ///
    /// The batch is one the step already submitted for serving: it carries the
    /// serving generation, and the shadow runs on another. A shadow batch never
    /// writes resident memory and never serves a request, and its failure is the
    /// world update's, which the main process reports — the arm keeps the
    /// submission alive and reads nothing from it.
    pub(super) fn shadow(
        &mut self,
        batch: ExecutionBatch,
        generation: u64,
    ) -> Result<(), WorkerSubmitError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(_) => Err(WorkerSubmitError::Failed(anyhow::anyhow!(
                "a supplied worker shadows nothing: only this process's devices do"
            ))),
            Path::InProcess(arm) => arm.shadow(batch, generation),
        }
    }

    /// The state of the shadow submissions of the pass in flight (7.3 step 2).
    ///
    /// A supplied worker submits no shadow batch, so a pass waiting on that
    /// arm cannot exist: `shadow` refuses it at submission, and the pass is
    /// answered with that refusal.
    pub(super) fn shadow_status(&mut self) -> ShadowStatus {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(_) => ShadowStatus::Settled,
            Path::InProcess(arm) => arm.shadow_status(),
        }
    }

    /// Waits up to `timeout` for one partial or terminal batch result, in the
    /// mechanism's own completion type.
    pub(super) fn poll(&mut self, timeout: Duration) -> Result<Option<BatchReport>, WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => Ok(worker.poll(timeout)?.map(BatchReport::from_wire)),
            Path::InProcess(arm) => arm.poll(timeout),
        }
    }

    /// Closes the pool.
    pub(super) fn close(&mut self) -> Result<(), WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.close(),
            // The devices of this process are its own: nothing outlives the call.
            Path::InProcess(_) => Ok(()),
        }
    }

    /// The nodes that hold the deployment's storage.
    pub(super) fn storage_nodes(&self) -> BTreeMap<String, String> {
        match &self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.storage_nodes(),
            // This process's devices hold no deployment storage of their own;
            // the engine's host KV path is served by the storage server it
            // reaches itself.
            Path::InProcess(_) => BTreeMap::new(),
        }
    }

    /// Maps the host KV cache's page pools on every rank of the pool, each
    /// holding `tokens` tokens of every cache group.
    pub(super) fn map_host_cache(
        &mut self,
        tokens: u64,
    ) -> Result<Vec<(WorkerGroupId, Vec<String>)>, WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.map_host_cache(tokens),
            // The ranks of this process map their pools through the worker that owns their
            // caches: its device threads issue the copies.
            Path::InProcess(in_process) => {
                let mapped = in_process
                    .batches
                    .map_host_cache(tokens)
                    .map_err(|error| anyhow::anyhow!(error))?;
                mapped
                    .into_iter()
                    .map(|(group, ranks)| {
                        let results = ranks
                            .iter()
                            .map(|pools| serde_json::to_string(&Some(pools)))
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|error| anyhow::anyhow!(error))?;
                        Ok((WorkerGroupId(group), results))
                    })
                    .collect()
            }
        }
    }

    /// Commits the pool's world: it takes the predecessor world's storage it
    /// mapped read-only.
    pub(super) fn commit(&mut self) -> Result<Vec<String>, WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.commit(),
            // The world this process serves is the worker's own, committed
            // when it was spawned.
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// Publishes this world's immutable pages after its commit decision.
    pub(super) fn publish_world(&mut self) -> Result<(), WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.publish_world(),
            Path::InProcess(_) => Ok(()),
        }
    }

    /// Ends the storage of every world of `worlds` on the pool's nodes.
    pub(super) fn retire(&mut self, worlds: &[u64]) -> Result<(), WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.retire(worlds),
            Path::InProcess(_) => Ok(()),
        }
    }

    /// Keeps the weights alone of every world of `worlds`.
    pub(super) fn keep_weights(
        &mut self,
        worlds: &[u64],
        to_host: bool,
        cache_keys: &[Vec<u8>],
    ) -> Result<Vec<String>, WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.keep_weights(worlds, to_host, cache_keys),
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// Halves, or restores when not `halve`, the deployment's link budgets.
    pub(super) fn throttle(&mut self, halve: bool) -> Result<Vec<String>, WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.throttle(halve),
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// The storage ledgers of the pool's nodes.
    pub(super) fn ledgers(&mut self) -> Result<Vec<String>, WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.ledgers(),
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// Runs the worker hook `name` on every rank while none runs a batch.
    pub(super) fn hook(&mut self, name: &str, payload: &str) -> Result<Vec<String>, WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.hook(name, payload),
            Path::InProcess(arm) => arm
                .batches
                .hook(name, payload)
                .map_err(|error| anyhow::anyhow!(error)),
        }
    }

    /// Runs the worker hook `name`, answering each group with its ranks'
    /// results.
    pub(super) fn hook_groups(
        &mut self,
        name: &str,
        payload: &str,
    ) -> Result<Vec<(WorkerGroupId, Vec<String>)>, WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.hook_groups(name, payload),
            // This process's ranks are the one pool its inventory states, and its data plane
            // answers for them in rank order.
            Path::InProcess(arm) => {
                let results = arm
                    .batches
                    .hook(name, payload)
                    .map_err(|error| anyhow::anyhow!(error))?;
                Ok(arm
                    .info
                    .workers
                    .first()
                    .filter(|_| !results.is_empty())
                    .map(|(group, _)| (group.clone(), results))
                    .into_iter()
                    .collect())
            }
        }
    }

    /// Commits prepared expert allocations and retains their order.
    pub(super) fn commit_experts(
        &mut self,
        order: &BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<String>, WorkerError> {
        match &mut self.path {
            #[cfg(feature = "testing")]
            Path::Supplied(worker) => worker.commit_experts(order),
            Path::InProcess(arm) => arm
                .batches
                .commit_experts(order)
                .map_err(|error| anyhow::anyhow!(error)),
        }
    }
}

impl World for Submissions {
    fn info(&self) -> &WorkerInfo {
        Submissions::info(self)
    }

    fn has_capacity(&self, worker: &WorkerGroupId) -> bool {
        Submissions::has_capacity(self, worker)
    }
}

impl InProcess {
    /// The state of this pass's shadow submissions, reading every one that has
    /// finished and answering the first failure (7.3 step 2).
    ///
    /// A failure carries the device thread's label beside its error, so the
    /// update states which paired forward of which operator and bucket left
    /// tolerance rather than only the element that did.
    pub(super) fn shadow_status(&mut self) -> ShadowStatus {
        let mut failure: Option<String> = None;
        self.shadows
            .retain(|submission| match submission.outcome() {
                None => true,
                Some(Outcome::Success(_)) => false,
                Some(Outcome::Failed(error)) => {
                    let reason = match submission.failure_label() {
                        Some(label) => format!("{label}: {error}"),
                        None => error.to_string(),
                    };
                    failure.get_or_insert(reason);
                    false
                }
                Some(Outcome::Cancelled) => {
                    failure.get_or_insert_with(|| {
                        "a shadow batch was cancelled before its comparison ran".to_owned()
                    });
                    false
                }
            });
        match failure {
            Some(reason) => ShadowStatus::Failed(reason),
            None if self.shadows.is_empty() => ShadowStatus::Settled,
            None => ShadowStatus::Pending,
        }
    }

    /// Whether `worker` is a stated pool with a slot free.
    fn has_room(&self, worker: &WorkerGroupId) -> bool {
        if !self.info.workers.iter().any(|(id, _)| id == worker) {
            return false;
        }
        let depth = self
            .info
            .layouts
            .get(worker)
            .map_or(1, |layout| layout.queue_depth.max(1));
        self.occupied.get(worker).copied().unwrap_or(0) < depth
    }

    /// The pools `batch` takes a slot of: those its calls and its page offloads run on.
    fn pools(batch: &ExecutionBatch) -> Vec<WorkerGroupId> {
        let mut pools: Vec<WorkerGroupId> = batch
            .requests
            .iter()
            .map(|(_, dispatch)| dispatch.worker.clone())
            .chain(
                batch
                    .page_offloads
                    .iter()
                    .map(|offloads| offloads.worker.clone()),
            )
            .collect();
        pools.sort();
        pools.dedup();
        pools
    }

    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), WorkerSubmitError> {
        // Pages that wait while nothing runs ride no forward: they are a batch of their own, which
        // the mechanism reads the end of as it reads any other.
        if batch.requests.is_empty()
            && let Some(offloads) = batch
                .page_offloads
                .as_ref()
                .filter(|offloads| !offloads.pages.is_empty())
        {
            let submission = self
                .batches
                .copy_pages(offloads.pages.clone())
                .map_err(|error| WorkerSubmitError::Failed(anyhow::anyhow!(error)))?;
            return self
                .queue
                .send(Pending {
                    batch: batch.id,
                    submission,
                    calls: Vec::new(),
                })
                .map_err(|error| {
                    WorkerSubmitError::Failed(anyhow::anyhow!(
                        "the in-process waiting thread has gone: {error}"
                    ))
                });
        }
        let pools = Self::pools(&batch);
        if !pools.iter().all(|pool| self.has_room(pool)) {
            return Err(WorkerSubmitError::WouldBlock(Box::new(batch)));
        }
        let (device, calls) = self.describe(&batch).map_err(WorkerSubmitError::Failed)?;
        let submission = self
            .batches
            .submit(device)
            .map_err(|error| WorkerSubmitError::Failed(anyhow::anyhow!(error)))?;
        for pool in &pools {
            *self.occupied.entry(pool.clone()).or_default() += 1;
        }
        self.unresolved.insert(batch.id, pools);
        self.queue
            .send(Pending {
                batch: batch.id,
                submission,
                calls,
            })
            .map_err(|error| {
                WorkerSubmitError::Failed(anyhow::anyhow!(
                    "the in-process waiting thread has gone: {error}"
                ))
            })
    }

    /// Gives back the slots of the batch `report` ends.
    fn resolved(&mut self, report: &BatchReport) {
        if !report.done {
            return;
        }
        for pool in self.unresolved.remove(&report.batch_id).unwrap_or_default() {
            if let Some(count) = self.occupied.get_mut(&pool) {
                *count = count.saturating_sub(1);
            }
        }
    }

    fn shadow(&mut self, batch: ExecutionBatch, generation: u64) -> Result<(), WorkerSubmitError> {
        // A shadow batch reports nothing of its own: the passes are the world
        // update's paired forwards, compared by each operator's tolerance on the
        // generation they name.
        let (device, _) = self
            .describe_with(&batch, generation)
            .map_err(WorkerSubmitError::Failed)?;
        let submission = self
            .batches
            .shadow(device)
            .map_err(|error| WorkerSubmitError::Failed(anyhow::anyhow!(error)))?;
        self.shadows.push(submission);
        self.shadows
            .retain(|submission| submission.outcome().is_none());
        Ok(())
    }

    /// A report the waiting thread collected, with its batch's slots given back.
    fn received(
        &mut self,
        report: Result<BatchReport, WorkerError>,
    ) -> Result<Option<BatchReport>, WorkerError> {
        match report {
            Ok(report) => {
                self.resolved(&report);
                Ok(Some(report))
            }
            // A failure names no batch, and the mechanism retires what it affected: no slot is
            // kept for a batch that will not report.
            Err(error) => {
                self.unresolved.clear();
                self.occupied.clear();
                Err(error)
            }
        }
    }

    fn poll(&mut self, timeout: Duration) -> Result<Option<BatchReport>, WorkerError> {
        let gone = || {
            anyhow::anyhow!("the in-process waiting thread has gone while batches were outstanding")
        };
        // A poll that does not wait takes a result if one is there, and leaves a
        // nudge for the park that follows, which is the one it is meant to end.
        if timeout.is_zero() {
            return match self.ready.try_recv() {
                Ok(report) => self.received(report),
                Err(crossbeam_channel::TryRecvError::Empty) => Ok(None),
                Err(crossbeam_channel::TryRecvError::Disconnected) => Err(gone()),
            };
        }
        crossbeam_channel::select! {
            recv(self.ready) -> report => match report {
                Ok(report) => self.received(report),
                Err(_) => Err(gone()),
            },
            recv(self.nudged) -> _ => Ok(None),
            default(timeout) => Ok(None),
        }
    }

    /// The batch as the worker reads it, with the mechanism's names for its
    /// calls.
    ///
    /// The refusals are the mechanism's, stated here because this is where the
    /// batch crosses into the worker's layout: a batch the mechanism could
    /// not state the columns of, a batch of media or denoise calls (2.4 serves
    /// no media in process yet), and a batch whose call kinds name no device
    /// phase are refused rather than served on invented values. The shape is
    /// the assembly's, not this seam's (`batching::columns::batch_shape`), and
    /// it is checked against the columns before the batch leaves.
    ///
    /// The batch's lifecycle commands do not travel: `::worker::Batch` has no
    /// field for them, because the resources such a command retires —
    /// a request's row, its buffers, its pages — belong to this process's
    /// worker once it holds the columns. A batch that completes reports no
    /// command as failed, which the mechanism reads as its commands having
    /// been applied.
    fn describe(
        &self,
        batch: &ExecutionBatch,
    ) -> anyhow::Result<(DeviceBatch, Vec<(ipc::RequestKey, ipc::CallId)>)> {
        self.describe_with(batch, self.generation)
    }

    /// The batch as the worker reads it, on generation `generation`.
    ///
    /// The serving path stamps the generation the step read; a shadow batch
    /// names the world whose kernels it compares against, which is another.
    fn describe_with(
        &self,
        batch: &ExecutionBatch,
        generation: u64,
    ) -> anyhow::Result<(DeviceBatch, Vec<(ipc::RequestKey, ipc::CallId)>)> {
        // One batch is one computation: a token batch is the typed columns the mechanism wrote,
        // and a media batch is the computation its kernels were captured for beside the request's
        // own stated parameters. A batch that states both would be served twice over, and one that
        // states neither cannot be served at all.
        if let Some(media) = batch.media.clone() {
            // A text encoding is the one media computation that reads token columns: its prompt.
            anyhow::ensure!(
                batch.columns.is_none() || media.computation.call == ipc::MediaCall::TextEncoding,
                "batch {} states token columns and media columns: one batch is one computation",
                batch.id
            );
            media
                .check()
                .map_err(|error| anyhow::anyhow!("batch {}: {error}", batch.id))?;
            let mut calls = Vec::with_capacity(batch.requests.len());
            let mut kind = None;
            for (call, _) in &batch.requests {
                anyhow::ensure!(
                    matches!(call.kind, ipc::CallKind::Media(_)),
                    "batch {} states media columns and carries a call no media phase serves: {:?}",
                    batch.id,
                    call.kind
                );
                let phase = wire_phase(call.kind).ok_or_else(|| {
                    anyhow::anyhow!(
                        "batch {} carries a call kind no device phase serves: {:?}",
                        batch.id,
                        call.kind
                    )
                })?;
                anyhow::ensure!(
                    kind.is_none_or(|served| served == phase),
                    "batch {} mixes phases: one batch serves one phase",
                    batch.id
                );
                kind = Some(phase);
                calls.push((call.request_key, call.call_id));
            }
            let phase =
                kind.ok_or_else(|| anyhow::anyhow!("batch {} carries no call to serve", batch.id))?;
            let computation = media.computation;
            let variant = computation.variant().ok_or_else(|| {
                anyhow::anyhow!(
                    "batch {} states a {:?} computation no variant serves",
                    batch.id,
                    computation.call
                )
            })?;
            let device = DeviceBatch {
                id: DeviceBatchId(batch.id),
                phase,
                calls: (0..calls.len() as u64).map(DeviceCallId).collect(),
                shape: ::worker::plan::Shape {
                    rows: computation.rows,
                    tokens: computation.tokens,
                    variant,
                },
                // A text encoding reads its prompt as token columns. Every other media computation
                // reads none, so its batch carries none: the fields its kernels read are the
                // model's own, staged inside the runner.
                columns: batch.columns.clone().unwrap_or_default(),
                generation,
                cache_units: Vec::new(),
                block_tables: Vec::new(),
                page_loads: Vec::new(),
                page_offloads: Vec::new(),
                media: Some(media),
            };
            return Ok((device, calls));
        }
        let columns = batch.columns.clone().ok_or_else(|| {
            // A media batch that arrives without its own columns is a different fact from a token
            // batch without a layout: the mechanism states a media computation's per-request buffers
            // only for the families it has a record for, so the refusal names the family rather than
            // the token layout that batch never had.
            match batch.requests.first().map(|(call, _)| call.kind) {
                Some(ipc::CallKind::Media(call)) => anyhow::anyhow!(
                    "batch {} states no media columns: the mechanism stages no per-request buffers \
                     for a {:?} computation",
                    batch.id,
                    call
                ),
                _ => anyhow::anyhow!(
                    "batch {} carries no columns: the mechanism could not state the layout its \
                     kernels read",
                    batch.id
                ),
            }
        })?;
        let mut calls = Vec::with_capacity(batch.requests.len());
        let mut kind = None;
        for (call, _) in &batch.requests {
            anyhow::ensure!(
                !matches!(call.kind, ipc::CallKind::Media(_)),
                "in-process execution serves no media call from a token batch (batch {}): {:?}",
                batch.id,
                call.kind
            );
            let phase = wire_phase(call.kind).ok_or_else(|| {
                anyhow::anyhow!(
                    "batch {} carries a call kind no device phase serves: {:?}",
                    batch.id,
                    call.kind
                )
            })?;
            anyhow::ensure!(
                kind.is_none_or(|served| served == phase),
                "batch {} mixes phases: one batch serves one phase",
                batch.id
            );
            kind = Some(phase);
            calls.push((call.request_key, call.call_id));
        }
        let phase =
            kind.ok_or_else(|| anyhow::anyhow!("batch {} carries no call to serve", batch.id))?;
        let shape = crate::batching::columns::batch_shape(&columns);
        columns
            .check(&shape)
            .map_err(|error| anyhow::anyhow!("batch {}: {error}", batch.id))?;
        let device = DeviceBatch {
            id: DeviceBatchId(batch.id),
            phase,
            calls: (0..calls.len() as u64).map(DeviceCallId).collect(),
            shape,
            columns,
            generation,
            // What the mechanism allocated for every call's pages: the units each takes, which the
            // runner resets before the forward writes them, and the copies a continuation's partial
            // last page needs from the page its predecessor published. A batch without them reads
            // that page from whatever its new unit held before.
            cache_units: batch
                .requests
                .iter()
                .flat_map(|(_, dispatch)| dispatch.new_cache_units.iter().cloned())
                .collect(),
            // Every call's tables, flattened: the worker installs these into the cache's pooled
            // table before the replay, because the gather reads row `slot` of that table and a row
            // nobody wrote gathers another request's pages.
            block_tables: batch
                .requests
                .iter()
                .flat_map(|(_, dispatch)| dispatch.block_tables.iter().cloned())
                .collect(),
            // The host KV cache's copies, which the runner issues on its copy stream: what the
            // batch's new units load before the forward, and what it publishes beside it.
            page_loads: batch
                .requests
                .iter()
                .flat_map(|(_, dispatch)| dispatch.page_loads.iter().cloned())
                .collect(),
            page_offloads: batch
                .page_offloads
                .as_ref()
                .map(|offloads| offloads.pages.clone())
                .unwrap_or_default(),
            // A token forward carries no media columns: the media arm above is where a media
            // batch's are passed through, and this arm refuses a media call by name.
            media: None,
        };
        Ok((device, calls))
    }
}

/// The phase id the device worker addresses a batch by (2.4, 4.4).
///
/// The mechanism names phases with `scheduling`'s ids and the worker with
/// `ipc`'s; both are the same fixed vocabulary in the same order, and this is
/// the one place the two meet, so neither type grows a second numbering.
pub(super) fn wire_phase(kind: ipc::CallKind) -> Option<ipc::PhaseId> {
    let id = crate::views::phase_id(crate::views::call_phase(kind));
    u16::try_from(id.0).ok().map(ipc::PhaseId)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use ::worker::batch::SubmissionId;
    use ::worker::batch::{BatchOutput, Inventory, MediaColumns, PoolLayout, StagedRequest};
    use ::worker::plan::{Bucket, Computation, Extent, Variant};
    use ipc::{Call, CallKind, MediaCall};

    /// What one in-process submission carried to the worker.
    #[derive(Clone, Debug)]
    struct Delivered {
        phase: ipc::PhaseId,
        generation: u64,
        media: Option<MediaColumns>,
    }

    /// The worker's own devices, as a double: it states one pool, answers the bucket its plan
    /// would serve, and records every batch the mechanism hands it.
    struct Devices {
        inventory: Vec<Inventory>,
        bucket: Option<Bucket>,
        delivered: Arc<Mutex<Vec<Delivered>>>,
    }

    impl Devices {
        fn one(bucket: Option<Bucket>) -> (Arc<Self>, Arc<Mutex<Vec<Delivered>>>) {
            let delivered = Arc::new(Mutex::new(Vec::new()));
            let inventory = Inventory {
                group: "gpu".to_owned(),
                ranks: vec![ipc::MainProcessInfo::default()],
                layout: PoolLayout {
                    queue_depth: 2,
                    ..PoolLayout::default()
                },
                max_model_len: 4096,
            };
            (
                Arc::new(Self {
                    inventory: vec![inventory],
                    bucket,
                    delivered: Arc::clone(&delivered),
                }),
                delivered,
            )
        }
    }

    impl Batches for Devices {
        fn submit(&self, batch: DeviceBatch) -> ::worker::Result<Submission> {
            self.delivered
                .lock()
                .expect("the recorder is not poisoned")
                .push(Delivered {
                    phase: batch.phase,
                    generation: batch.generation,
                    media: batch.media,
                });
            Ok(Submission::completed(
                SubmissionId(1),
                Outcome::Success(BatchOutput::default()),
            ))
        }

        fn shadow(&self, batch: DeviceBatch) -> ::worker::Result<Submission> {
            self.submit(batch)
        }

        fn media(
            &self,
            _phase: ipc::PhaseId,
            _computation: &Computation,
        ) -> ::worker::Result<Option<Bucket>> {
            Ok(self.bucket)
        }

        fn inventory(&self) -> &[Inventory] {
            &self.inventory
        }
    }

    /// The bucket a denoise is planned into: one step over one request's layout.
    fn denoise_bucket() -> Bucket {
        Bucket::denoise(1, denoise_layout())
    }

    /// A layout of one small canvas: the extent a media batch states.
    fn denoise_layout() -> nn::capability::Layout {
        nn::capability::Layout {
            video: nn::capability::LatentShape {
                channels: 4,
                frames: 2,
                height: 8,
                width: 8,
            },
            audio: None,
            text_rows: 4,
            condition_rows: 0,
            condition_elements: 0,
        }
    }

    fn staged() -> StagedRequest {
        StagedRequest {
            request: ipc::RequestKey::new(1, ipc::RequestId(9), 0),
            conditioning: vec![1, 2, 3],
            step: 2,
            steps: 8,
            seed: 0x5eed,
            width: 768,
            height: 768,
            frames: 121,
            handed: 0,
            unit: 0,
        }
    }

    /// One call as the plan hands it to the mechanism.
    fn call(kind: CallKind, request_id: u64) -> (Call, crate::worker::RequestDispatch) {
        let key = ipc::RequestKey::new(1, ipc::RequestId(request_id), 0);
        let call = Call::new(key, ipc::CallId::new(0, 0), kind);
        (
            call,
            crate::worker::RequestDispatch {
                worker: WorkerGroupId("gpu".to_owned()),
                request_pool_idx: None,
                block_tables: Vec::new(),
                new_cache_units: Vec::new(),
                forward: Default::default(),
                latent: None,
                decode: None,
                buffers: Vec::new(),
                page_loads: Vec::new(),
                readers: None,
            },
        )
    }

    /// A media batch as `batching::attach_media` stamps it: the computation the plan selected,
    /// with the extent the worker answered.
    fn media_batch(id: u64, call_kind: MediaCall, rows: u32) -> ExecutionBatch {
        let computation = Computation {
            call: call_kind,
            rows,
            tokens: 0,
            extent: denoise_bucket().extent,
        };
        let mut batch = ExecutionBatch::new(
            id,
            (0..rows)
                .map(|row| call(CallKind::Media(call_kind), u64::from(row)))
                .collect(),
            Vec::new(),
            Vec::new(),
        );
        let staged = vec![staged(); rows as usize];
        batch.media = Some(MediaColumns::staged(computation, &staged));
        batch
    }

    /// The phase a call kind is served at, as the mechanism's one translation states it.
    fn phase_of(kind: CallKind) -> ipc::PhaseId {
        wire_phase(kind).expect("every call kind has a phase")
    }

    /// A media batch reaches the worker at the phase its call names, with the generation the
    /// step read and the computation its kernels were captured for.
    ///
    /// The phase is checked against what the call kind states and against the token path's own
    /// phase: a media arm that borrowed the token phase, or one that named a constant instead of
    /// reading each call, would pass neither. Both are read here as facts of the call — the arm's
    /// own translation is compared with itself elsewhere, because `views` is the crate's.
    #[test]
    fn a_media_batch_is_served_at_the_phase_its_call_names_and_the_step_generation() {
        let (devices, delivered) = Devices::one(Some(denoise_bucket()));
        let mut submissions = Submissions::over_batches(devices).expect("an in-process mechanism");
        let generation = 7;
        submissions.set_generation(generation);

        submissions
            .submit(media_batch(11, MediaCall::Denoising, 1))
            .expect("a media batch is served in process");
        submissions
            .submit(media_batch(12, MediaCall::ImageDecoding, 1))
            .expect("a media batch is served in process");

        let delivered = delivered.lock().expect("the recorder is not poisoned");
        let [denoising, decoding] = delivered.as_slice() else {
            panic!("two submissions were made, and {} arrived", delivered.len());
        };
        assert_eq!(
            denoising.phase,
            phase_of(CallKind::Media(MediaCall::Denoising)),
            "a denoising call is served at the denoise phase"
        );
        assert_eq!(
            decoding.phase,
            phase_of(CallKind::Media(MediaCall::ImageDecoding)),
            "a decoding call is served at the decode phase"
        );
        assert_ne!(
            denoising.phase,
            phase_of(CallKind::Forward(ipc::ForwardMode::Prefill)),
            "a media batch does not borrow the token path's phase"
        );
        let one = denoising;
        assert_eq!(
            one.generation, generation,
            "a batch runs on the generation the step read"
        );
        let media = one
            .media
            .as_ref()
            .expect("a media batch arrives with its computation");
        assert_eq!(media.computation.call, MediaCall::Denoising);
        assert_eq!(media.computation.rows, 1);
        assert!(
            matches!(media.computation.extent, Extent::Latents(_)),
            "the extent is the bucket's, which is what the worker answered: {:?}",
            media.computation.extent
        );
    }

    /// A pool holds `queue_depth` unresolved batches, and a report gives its slot back.
    ///
    /// This is what makes requests that run together meet in one step: each is ready when its own
    /// batch ends, so a pool that always took the next batch would serve them one row at a time.
    #[test]
    fn a_pool_holds_no_more_batches_than_its_queue_depth() {
        let (devices, delivered) = Devices::one(Some(denoise_bucket()));
        let mut submissions = Submissions::over_batches(devices).expect("an in-process mechanism");
        let pool = submissions.info().workers[0].0.clone();

        assert!(submissions.has_capacity(&pool), "an idle pool has room");
        for batch in 1..=2 {
            submissions
                .submit(media_batch(batch, MediaCall::Denoising, 1))
                .expect("a batch within the depth is taken");
        }
        assert!(!submissions.has_capacity(&pool), "the depth is taken");
        let refused = submissions.submit(media_batch(3, MediaCall::Denoising, 1));
        assert!(
            matches!(refused, Err(WorkerSubmitError::WouldBlock(_))),
            "a batch past the depth waits for a slot"
        );
        assert_eq!(
            delivered
                .lock()
                .expect("the recorder is not poisoned")
                .len(),
            2,
            "a batch that waits reaches no device"
        );

        // The double's batches report no calls, which the mechanism reads as a failure: it ends
        // the batches all the same, so the slots come back.
        let _ = submissions.poll(Duration::from_secs(5));
        assert!(submissions.has_capacity(&pool), "a report frees its slot");
    }

    /// The generation is the step's, for every batch of the step: a switchover mid-step cannot
    /// split one step across two worlds, so a token batch and a media batch of one step carry the
    /// same number.
    #[test]
    fn every_batch_of_a_step_carries_the_same_generation() {
        let (devices, delivered) = Devices::one(Some(denoise_bucket()));
        let mut submissions = Submissions::over_batches(devices).expect("an in-process mechanism");
        submissions.set_generation(3);
        submissions
            .submit(media_batch(1, MediaCall::Denoising, 1))
            .expect("a media batch is served in process");
        submissions.set_generation(4);
        submissions
            .submit(media_batch(2, MediaCall::Denoising, 1))
            .expect("a media batch is served in process");

        let delivered = delivered.lock().expect("the recorder is not poisoned");
        let generations: Vec<u64> = delivered.iter().map(|one| one.generation).collect();
        assert_eq!(
            generations,
            vec![3, 4],
            "each batch carries the generation its step read, in the order the steps ran"
        );
    }

    /// A batch that carries neither token columns nor a media computation is refused by name: the
    /// mechanism could not state what its kernels read, and it is not served on invented values.
    #[test]
    fn a_batch_the_mechanism_could_not_state_is_refused_by_name() {
        let (devices, delivered) = Devices::one(Some(denoise_bucket()));
        let mut submissions = Submissions::over_batches(devices).expect("an in-process mechanism");
        let batch = ExecutionBatch::new(
            5,
            vec![call(CallKind::Media(MediaCall::Denoising), 0)],
            Vec::new(),
            Vec::new(),
        );
        let refusal = submissions
            .submit(batch)
            .expect_err("a batch with no columns of either family is refused");
        let text = format!("{refusal}");
        assert!(
            text.contains("states no media columns") && text.contains("Denoising"),
            "the refusal names the family whose buffers nobody staged: {text}"
        );
        assert!(
            delivered
                .lock()
                .expect("the recorder is not poisoned")
                .is_empty(),
            "nothing reaches the worker"
        );
    }

    /// One batch is one computation: media columns beside token columns, or a media computation
    /// carrying a token call, are each refused rather than served twice over.
    #[test]
    fn one_batch_is_one_computation() {
        let (devices, _) = Devices::one(Some(denoise_bucket()));
        let mut submissions = Submissions::over_batches(devices).expect("an in-process mechanism");

        let mut both = media_batch(6, MediaCall::Denoising, 1);
        both.columns = Some(::worker::batch::BatchColumns::default());
        let refusal = format!(
            "{}",
            submissions
                .submit(both)
                .expect_err("a batch states one layout or the other")
        );
        assert!(
            refusal.contains("one batch is one computation"),
            "the refusal names the rule: {refusal}"
        );

        let mut mixed = media_batch(7, MediaCall::Denoising, 1);
        mixed
            .requests
            .push(call(CallKind::Forward(ipc::ForwardMode::Prefill), 9));
        let refusal = format!(
            "{}",
            submissions
                .submit(mixed)
                .expect_err("a media computation carries media calls")
        );
        assert!(
            refusal.contains("no media phase serves"),
            "the refusal names the call: {refusal}"
        );

        let variant = Computation {
            call: MediaCall::Denoising,
            rows: 1,
            tokens: 0,
            extent: denoise_bucket().extent,
        };
        assert_eq!(
            variant.variant(),
            Some(Variant::Denoise),
            "a denoise computation is the variant the worker plans"
        );
    }
}
