//! The mechanism's one door to execution.
//!
//! The step submits a batch, waits for results, and asks what the pool can do;
//! it does not know whether the batch goes to worker processes over the ipc
//! channel or to the in-process device executor (design §2.4). This module is
//! the door: every path of the mechanism that executes something comes through
//! it, and the two arms differ only in what they ask.
//!
//! The worker-process arm speaks the ipc protocol. The in-process arm is
//! `executor::Batches` (2.4, 4.4): the mechanism's own completion type is filled
//! from `executor::batch::BatchOutput` there, and nothing of the wire is
//! involved. The in-process path is `notes/sched/splice.md`'s splice: it lands
//! first as a constructor (`over_batches`) that the main process may choose, and
//! the worker-process arm goes when it serves a token end to end.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use crate::WorkerGroupId;
use crate::executor::ExecutionBatch;
use crate::executor::{Executor, ExecutorError, ExecutorInfo, ExecutorSubmitError};
use crate::scheduler::completion::BatchReport;
use executor::batch::{
    Batch as DeviceBatch, BatchId as DeviceBatchId, Batches, CallId as DeviceCallId, Submission,
};
use ipc::BatchCommand;

/// What placement and the views ask of the world: the pools it serves, and
/// which of them can take a submission now.
///
/// The step places calls on every path — a planned call, a media call, a
/// request's first call, a prefill target — and none of that submits anything.
/// Stating it as this trait is what lets one placement serve the worker-process
/// executor and the in-process one: what differs between them is who answers,
/// not what the step asks.
pub(super) trait World {
    /// The pools, their ranks and their capabilities.
    fn info(&self) -> &ExecutorInfo;

    /// Whether `worker` has an unclaimed submission slot.
    fn has_capacity(&self, worker: &WorkerGroupId) -> bool;
}

/// What the mechanism executes through.
pub(crate) struct Submissions {
    path: Path,
}

/// Which executor serves this scheduler's batches.
enum Path {
    /// Worker processes, over the ipc protocol.
    Workers(Box<dyn Executor>),
    /// The device executor of this process (2.4).
    InProcess(InProcess),
}

/// The mechanism over `executor::Batches`: this process's own devices.
///
/// A batch is converted to the executor's own description of it — the shape it
/// pads to, the host columns the kernels read, the generation whose kernels
/// serve it — and submitted. The completion comes back on a waiting thread, as
/// the mechanism's own type: the scheduler thread must never block on a
/// device's completion, because the policy is asked once per step whether or
/// not a device has answered.
struct InProcess {
    batches: Arc<dyn Batches>,
    info: ExecutorInfo,
    /// The kernels this step's batches serve on. The scheduler reads its
    /// generation cell once per step and stamps every batch of that step, so a
    /// switchover mid-step cannot split one step across two worlds.
    generation: u64,
    /// Completed reports, in the order the devices finished them.
    ready: crossbeam_channel::Receiver<Result<BatchReport, ExecutorError>>,
    /// Submissions still to be waited on, handed to the waiting thread.
    queue: crossbeam_channel::Sender<Pending>,
    /// Shadow submissions in flight: the executor reports their failure to the
    /// world's own update (7.3), so the mechanism only keeps them alive.
    shadows: Vec<Submission>,
}

/// One submitted batch, with what the report's calls name.
struct Pending {
    batch: u64,
    submission: Submission,
    /// Index for index with the batch's calls: the mechanism's name for the
    /// call the executor numbers by its index.
    calls: Vec<(ipc::RequestKey, ipc::CallId)>,
}

impl Submissions {
    pub(super) fn new(executor: Box<dyn Executor>) -> Self {
        Self {
            path: Path::Workers(executor),
        }
    }

    /// The mechanism over this process's own devices (2.4).
    ///
    /// The pools come from the executor's own inventory and nowhere else: an
    /// executor that states no pool is refused here rather than served with a
    /// guessed one, for the same reason `ExecutorInfo::from_inventory` refuses
    /// a pool that states no rank or no longest sequence — a mechanism that
    /// assumes a rank's cache makes the host KV path disappear at runtime.
    pub(crate) fn over_batches(batches: Arc<dyn Batches>) -> anyhow::Result<Submissions> {
        let info = ExecutorInfo::from_inventory(batches.inventory())?;
        anyhow::ensure!(
            !info.workers.is_empty(),
            "the in-process executor states no pool to schedule against"
        );
        let (queue, pending) = crossbeam_channel::unbounded::<Pending>();
        let (reports, ready) = crossbeam_channel::unbounded();
        std::thread::Builder::new()
            .name("sched-in-process-wait".into())
            .spawn(move || {
                // One waiting thread for the whole epoch: a submission's
                // outcome is a device's, and the scheduler thread reads only
                // what this thread has already collected.
                while let Ok(pending) = pending.recv() {
                    let outcome = pending.submission.wait();
                    let report = BatchReport::from_output(pending.batch, outcome, &pending.calls);
                    if reports.send(report).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| anyhow::anyhow!("starting the in-process waiting thread: {error}"))?;
        Ok(Submissions {
            path: Path::InProcess(InProcess {
                batches,
                info,
                generation: 0,
                ready,
                queue,
                shadows: Vec::new(),
            }),
        })
    }

    /// States the generation this step's batches serve on (4.2, 7.3).
    ///
    /// The scheduler calls this once per step, before it dispatches anything:
    /// every batch of one step runs on one world's kernels.
    pub(super) fn set_generation(&mut self, generation: u64) {
        if let Path::InProcess(arm) = &mut self.path {
            arm.generation = generation;
        }
    }

    /// The pool's capabilities: its workers, components, pools, and limits.
    pub(super) fn info(&self) -> &ExecutorInfo {
        match &self.path {
            Path::Workers(executor) => executor.info(),
            Path::InProcess(arm) => &arm.info,
        }
    }

    /// Whether every worker of `worker` is ready to accept execution.
    pub(super) fn is_ready(&self, worker: &WorkerGroupId) -> bool {
        match &self.path {
            Path::Workers(executor) => executor.is_ready(worker),
            // The arm serves the pools its inventory states and no others.
            Path::InProcess(arm) => arm.info.workers.iter().any(|(id, _)| id == worker),
        }
    }

    /// Whether `worker` executes nothing further.
    pub(super) fn is_closed(&self, worker: &WorkerGroupId) -> bool {
        match &self.path {
            Path::Workers(executor) => executor.is_closed(worker),
            // A device of this process is closed when the process ends.
            Path::InProcess(_) => false,
        }
    }

    /// Whether `worker` has an unclaimed destination submission slot.
    pub(super) fn has_capacity(&self, worker: &WorkerGroupId) -> bool {
        match &self.path {
            Path::Workers(executor) => executor.has_capacity(worker),
            // `Batches::submit` queues on the device's own stream and never
            // refuses for capacity, so a stated pool always has room here.
            Path::InProcess(arm) => arm.info.workers.iter().any(|(id, _)| id == worker),
        }
    }

    /// Whether `command` can be handed to a worker now.
    pub(super) fn command_has_capacity(&self, command: &BatchCommand) -> bool {
        match &self.path {
            Path::Workers(executor) => executor.command_has_capacity(command),
            // A lifecycle command travels inside the batch it belongs to.
            Path::InProcess(_) => true,
        }
    }

    /// Submits one logical batch without blocking for capacity.
    pub(super) fn submit(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        match &mut self.path {
            Path::Workers(executor) => executor.submit(batch),
            Path::InProcess(arm) => arm.submit(batch),
        }
    }

    /// Runs `batch`'s paired forwards without serving them (7.3 step 2).
    ///
    /// The failure of a shadow batch is the world update's, which the main
    /// process reports; the mechanism keeps the submission alive and reads
    /// nothing from it.
    pub(super) fn shadow(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        match &mut self.path {
            Path::Workers(_) => Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                "a worker-process executor shadows a world on its workers, not through this door"
            ))),
            Path::InProcess(arm) => arm.shadow(batch),
        }
    }

    /// Waits up to `timeout` for one partial or terminal batch result, in the
    /// mechanism's own completion type.
    pub(super) fn poll(&mut self, timeout: Duration) -> Result<Option<BatchReport>, ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => Ok(executor.poll(timeout)?.map(BatchReport::from_wire)),
            Path::InProcess(arm) => arm.poll(timeout),
        }
    }

    /// Closes the pool and its physical workers.
    pub(super) fn close(&mut self) -> Result<(), ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.close(),
            // The devices of this process are its own: nothing outlives the call.
            Path::InProcess(_) => Ok(()),
        }
    }

    /// The nodes that hold the deployment's storage.
    pub(super) fn storage_nodes(&self) -> BTreeMap<String, String> {
        match &self.path {
            Path::Workers(executor) => executor.storage_nodes(),
            // This process's devices hold no deployment storage of their own;
            // the engine's host KV path is served by the storage server it
            // reaches itself.
            Path::InProcess(_) => BTreeMap::new(),
        }
    }

    /// Maps the host KV cache's page pools on every worker's ranks, each
    /// holding `tokens` tokens of every cache group.
    pub(super) fn map_host_cache(
        &mut self,
        tokens: u64,
    ) -> Result<Vec<(WorkerGroupId, Vec<String>)>, ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.map_host_cache(tokens),
            // No rank of this process maps a host page pool, so the mechanism
            // plans no host KV write rather than one nothing answers.
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// Commits the pool's world: every worker takes the predecessor world's
    /// storage it mapped read-only.
    pub(super) fn commit(&mut self) -> Result<Vec<String>, ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.commit(),
            // The world this process serves is the executor's own, committed
            // when it was spawned; there is no worker to tell.
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// Publishes this world's immutable pages after its commit decision.
    pub(super) fn publish_world(&mut self) -> Result<(), ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.publish_world(),
            Path::InProcess(_) => Ok(()),
        }
    }

    /// Ends the storage of every world of `worlds` on the pool's nodes.
    pub(super) fn retire(&mut self, worlds: &[u64]) -> Result<(), ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.retire(worlds),
            Path::InProcess(_) => Ok(()),
        }
    }

    /// Keeps the weights alone of every world of `worlds`.
    pub(super) fn keep_weights(
        &mut self,
        worlds: &[u64],
        to_host: bool,
        cache_keys: &[Vec<u8>],
    ) -> Result<Vec<String>, ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.keep_weights(worlds, to_host, cache_keys),
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// Halves, or restores when not `halve`, the deployment's link budgets.
    pub(super) fn throttle(&mut self, halve: bool) -> Result<Vec<String>, ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.throttle(halve),
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// The storage ledgers of the pool's nodes.
    pub(super) fn ledgers(&mut self) -> Result<Vec<String>, ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.ledgers(),
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// Runs the worker hook `name` on every worker while none runs a batch.
    pub(super) fn hook(&mut self, name: &str, payload: &str) -> Result<Vec<String>, ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.hook(name, payload),
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// Runs the worker hook `name`, answering each worker group with its
    /// ranks' results.
    pub(super) fn hook_groups(
        &mut self,
        name: &str,
        payload: &str,
    ) -> Result<Vec<(WorkerGroupId, Vec<String>)>, ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.hook_groups(name, payload),
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }

    /// Commits prepared expert allocations and retains their order.
    pub(super) fn commit_experts(
        &mut self,
        order: &BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<String>, ExecutorError> {
        match &mut self.path {
            Path::Workers(executor) => executor.commit_experts(order),
            // The expert allocation of a world this process serves is its own
            // setup's (7.6), applied where the world is built.
            Path::InProcess(_) => Ok(Vec::new()),
        }
    }
}

impl World for Submissions {
    fn info(&self) -> &ExecutorInfo {
        Submissions::info(self)
    }

    fn has_capacity(&self, worker: &WorkerGroupId) -> bool {
        Submissions::has_capacity(self, worker)
    }
}

impl InProcess {
    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        let (device, calls) = self.describe(&batch).map_err(ExecutorSubmitError::Failed)?;
        let submission = self
            .batches
            .submit(device)
            .map_err(|error| ExecutorSubmitError::Failed(anyhow::anyhow!(error)))?;
        self.queue
            .send(Pending {
                batch: batch.id,
                submission,
                calls,
            })
            .map_err(|error| {
                ExecutorSubmitError::Failed(anyhow::anyhow!(
                    "the in-process waiting thread has gone: {error}"
                ))
            })
    }

    fn shadow(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        // A shadow batch has no call of its own to report: the passes are the
        // world update's paired forwards, served on the generation cells.
        let (device, _) = self.describe(&batch).map_err(ExecutorSubmitError::Failed)?;
        let submission = self
            .batches
            .shadow(device)
            .map_err(|error| ExecutorSubmitError::Failed(anyhow::anyhow!(error)))?;
        self.shadows.push(submission);
        self.shadows
            .retain(|submission| submission.outcome().is_none());
        Ok(())
    }

    fn poll(&mut self, timeout: Duration) -> Result<Option<BatchReport>, ExecutorError> {
        match self.ready.recv_timeout(timeout) {
            Ok(report) => report.map(Some),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Ok(None),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => Err(anyhow::anyhow!(
                "the in-process waiting thread has gone while batches were outstanding"
            )),
        }
    }

    /// The batch as the executor reads it, with the mechanism's names for its
    /// calls.
    ///
    /// The refusals are the mechanism's, stated here because this is where the
    /// batch crosses into the executor's layout: a batch the mechanism could
    /// not state the columns of, a batch of media or denoise calls (2.4 serves
    /// no media in process yet), and a batch whose shape the mechanism cannot
    /// state are refused rather than served on invented values.
    fn describe(
        &self,
        batch: &ExecutionBatch,
    ) -> anyhow::Result<(DeviceBatch, Vec<(ipc::RequestKey, ipc::CallId)>)> {
        let columns = batch.columns.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "batch {} carries no columns: the mechanism could not state the layout its kernels read",
                batch.id
            )
        })?;
        let mut calls = Vec::with_capacity(batch.requests.len());
        let mut kind = None;
        for (call, _) in &batch.requests {
            anyhow::ensure!(
                !matches!(call.kind, ipc::CallKind::Media(_)),
                "in-process execution serves no media call yet (batch {}): {:?}",
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
        let shape = shape_of(&batch.requests, &columns);
        columns
            .check(&shape)
            .map_err(|error| anyhow::anyhow!("batch {}: {error}", batch.id))?;
        let device = DeviceBatch {
            id: DeviceBatchId(batch.id),
            phase,
            calls: (0..calls.len() as u64).map(DeviceCallId).collect(),
            shape,
            columns,
            generation: self.generation,
        };
        Ok((device, calls))
    }
}

/// The phase id the device executor addresses a batch by (2.4, 4.4).
///
/// The mechanism names phases with `scheduling`'s ids and the executor with
/// `ipc`'s; both are the same fixed vocabulary in the same order, and this is
/// the one place the two meet, so neither type grows a second numbering.
fn wire_phase(kind: ipc::CallKind) -> Option<ipc::PhaseId> {
    let id = crate::scheduler::views::phase_id(crate::scheduler::views::call_phase(kind));
    u16::try_from(id.0).ok().map(ipc::PhaseId)
}

/// The shape a batch computes (4.4, 4.5): the rows and tokens its columns
/// hold, and what the runner does with them.
///
/// A batch whose calls prefill is a prefill of `rows` rows holding `tokens`
/// tokens; the phase fixes which. It projects each row's last hidden state
/// exactly when a row's logits are wanted — a prefill that samples or scores
/// from its last row states that in `logits_rows` — and only writes the cache
/// otherwise. A decode row runs the sampler graph, which is what turns a row's
/// parameters into a token; a greedy request's parameters are the sampler's, so
/// the graph is the same one (4.6, `ops::sampler`).
///
/// This rule lives at the mechanism's batch assembly, where the sampling rows
/// are decided; it is stated here until the assembly states the shape itself.
fn shape_of(
    requests: &[(ipc::Call, crate::executor::RequestPlacement)],
    columns: &executor::batch::BatchColumns,
) -> executor::plan::Shape {
    let rows = columns.rows.rows() as u32;
    let tokens = columns.rows.tokens() as u32;
    let prefill = requests
        .iter()
        .any(|(call, _)| matches!(call.kind, ipc::CallKind::Forward(ipc::ForwardMode::Prefill)));
    let variant = if prefill {
        if columns.rows.logits_rows.is_empty() {
            executor::plan::Variant::PrefillWrite
        } else {
            executor::plan::Variant::PrefillOutput
        }
    } else {
        executor::plan::Variant::DecodeSampled
    };
    executor::plan::Shape {
        rows,
        tokens,
        variant,
    }
}
