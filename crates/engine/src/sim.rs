//! GPU-free executor for scheduler and frontend behavior.
//!
//! [`SimExecutor`] implements the `Executor` trait the scheduler drives, over
//! a background thread that runs a [`SimEngine`]. It consumes the scheduler's
//! typed admissions, lifecycle commands, and calls, bounds the batches in
//! flight by a queue depth, and tracks admitted requests and live products for
//! `command_has_capacity`.
//!
//! [`SimEngine`] keeps the per-request progress an executor tracks:
//! positions, cache extent, denoising step, and predicate values. It enforces
//! request identity, lifecycle, call coordinates, and predicate dependencies,
//! and reports deterministic tokens, cache extents, denoising progress, and
//! synthetic media published through POSIX shared memory. The module is built
//! only with the `testing` feature; engine and server tests and the engine's
//! `control_plane` example drive the scheduler through it in place of GPU
//! executors.

use ipc::{ForwardMode, MediaCall, TransferMode};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::CommandWaker;
use crate::executor::{
    BatchResult, CallResult, CommandOutcome, CommandResult, ComponentConfig, ExecutionBatch,
    Executor, ExecutorFailure, ExecutorGroupId, ExecutorInfo, ExecutorSubmitError, PoolPlacement,
};
use crate::scheduler::SpecialTokenIds;
use base64::Engine as _;
use crossbeam_channel::{Receiver, Sender};
use ipc::philox;
use ipc::{
    BatchOutput, Call, CallKind, CallOutput, CallStatus, ErrorCode, FinishFlags, NewRequest,
    SamplingState, TensorRef, TimingCounters, MainProcessInfo,
};
use ipc::{ImageParams, RequestId, SamplingParams, TokenLogprob};

const DEFAULT_TEXT_LEN: usize = 8;
/// The scheduler-facing result of one whole batch: its completions, each with
/// its media claimed, and one `Applied` receipt per lifecycle command, numbered
/// with `Start` admissions skipped.
///
/// A completion's media is claimed before anything else reads it, and a failure
/// to claim it belongs to the call: the batch's other results stay usable.
fn batch_result(report: ipc::BatchOutput, commands: &[ipc::BatchCommand]) -> BatchResult {
    let results = report
        .completions
        .into_iter()
        .map(|mut output| {
            let media = output
                .media_output
                .take()
                .map(|media| {
                    let ipc::ArtifactHandle::PosixShm { name } = media.handle;
                    // SAFETY: export transfers immutable storage after the writer closes.
                    // The mapping owns the bytes even if this result is rejected downstream.
                    unsafe { ipc::SharedMedia::open(&name, media.bytes) }.map(Arc::new)
                })
                .transpose();
            CallResult { output, media }
        })
        .collect();
    BatchResult {
        batch_id: report.batch_id,
        results,
        command_results: commands
            .iter()
            .filter(|command| !matches!(command, ipc::BatchCommand::Start { .. }))
            .enumerate()
            .map(|(index, command)| CommandResult {
                command_index: index as u32,
                request_key: command.request_key(),
                outcome: CommandOutcome::Applied,
            })
            .collect(),
        done: true,
        worker_exec_us: report.worker_exec_us.into_iter().collect(),
        forward_stats: report.forward_stats.into_iter().collect(),
    }
}

/// The failure a stall report amounts to, naming the forward the job belongs
/// to and the calls it held.
fn stall_failure(stall: &ipc::Stall) -> anyhow::Error {
    let calls = stall
        .calls
        .iter()
        .map(|call| format!("{}/{:?}", call.request_key.request_id.0, call.call_id))
        .collect::<Vec<_>>();
    let forward = match stall.batch_id {
        Some(batch_id) => format!(
            "a job of batch {batch_id} ({}, method {}, calls [{}])",
            stall.route.as_deref().unwrap_or("no calls"),
            stall.method.as_deref().unwrap_or("none"),
            calls.join(", "),
        ),
        None => format!(
            "a job of no batch (method {})",
            stall.method.as_deref().unwrap_or("none")
        ),
    };
    anyhow::anyhow!(
        "the builder thread stalled: it has performed {forward} for {:.1} s without completing it",
        Duration::from_millis(stall.running_ms).as_secs_f64()
    )
}

/// Default EOS id the simulator samples after `text_len` tokens.
/// `EngineConfig::sim` configures the same id so the scheduler finishes
/// simulated requests.
const FAKE_EOS_TOKEN: u32 = 151_645;
const SYNTH_VOCAB_SIZE: usize = FAKE_EOS_TOKEN as usize + 1;
/// The longest sequence the simulated pool serves: the model's context length,
/// which the attention descriptor bounds every row by.
///
/// A world states it as the pool's own fact, so the mechanism shapes a KV view
/// from a number the pool declares and not from a guess: batches are
/// refused when it is zero, exactly as a deployment that never states its
/// model's length is refused.
const SIM_MAX_MODEL_LEN: u32 = 32_768;

const DEFAULT_DENOISE_STEPS: u16 = 50;
const DEFAULT_IMAGE_HW: (u32, u32) = (512, 512);

/// Work sent to the simulator thread.
enum Job {
    Batch(ExecutionBatch),
    Shutdown,
}

/// Runs the deterministic model simulator on a bounded asynchronous executor seam.
/// Reports a simulated deployment's executor group lost: the group closes, and
/// the executor answers its next poll with the loss instead of a result.
///
/// The scheduler stops when every model group is closed. Without this a
/// simulated executor's executors never die, so no case could state what
/// happens when they all do.
#[derive(Clone, Default)]
pub struct SimLoss {
    /// The group reported lost: it stays lost for the executor's lifetime, so
    /// it reports not ready after the report has been answered.
    group: Arc<Mutex<Option<ExecutorGroupId>>>,
    /// Whether the lost group is closed as well as lost: a group that cannot
    /// be replaced closes, while one whose replacement is starting is not
    /// ready but not closed.
    closed: Arc<Mutex<bool>>,
    /// The report the executor's next poll answers with, taken once: the
    /// group, why it failed, and the requests whose work is gone.
    pending: Arc<Mutex<Option<(ExecutorGroupId, String, Vec<ipc::RequestKey>)>>>,
    /// Wakes a scheduler parked on this executor, so the loss is answered
    /// without waiting out its park.
    wake: Arc<Mutex<Option<CommandWaker>>>,
}

impl SimLoss {
    /// Reports `group` lost for `message`; the executor's next poll answers
    /// with the loss, and `group` reports not ready and closed from then on.
    ///
    /// # Panics
    ///
    /// Panics if a previous report is still pending: a loss is delivered once.
    pub fn lose(&self, group: ExecutorGroupId, message: impl Into<String>) {
        {
            let mut lost = self
                .group
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(lost.is_none(), "a simulated loss is reported once");
            *lost = Some(group.clone());
            *self
                .closed
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *pending = Some((group, message.into(), Vec::new()));
        }
        if let Some(waker) = self
            .wake
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            waker.wake();
        }
    }

    /// Reports `group` failed by the stall `stall` names, as a rank whose
    /// builder thread stayed inside one job reports it: the executor's next
    /// poll answers with that failure, naming the forward the job belongs to
    /// and the requests whose calls it held.
    ///
    /// # Panics
    ///
    /// Panics if a previous report is still pending: a report is delivered
    /// once.
    pub fn stall(&self, group: ExecutorGroupId, stall: &ipc::Stall) {
        {
            let mut lost = self
                .group
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(lost.is_none(), "a simulated loss is reported once");
            *lost = Some(group.clone());
            *self
                .closed
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
            let requests = stall
                .calls
                .iter()
                .map(|call| call.request_key)
                .collect::<Vec<_>>();
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *pending = Some((
                group,
                stall_failure(stall).to_string(),
                requests,
            ));
        }
        if let Some(waker) = self
            .wake
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            waker.wake();
        }
    }

    /// Reports `group` lost while a replacement is starting: the group is not
    /// ready, and it is not closed either, so the scheduler waits for the
    /// replacement instead of treating the engine as unserved.
    ///
    /// [`Self::restore`] stands a replacement in its place.
    ///
    /// # Panics
    ///
    /// Panics if a previous report is still pending: a loss is delivered once.
    pub fn lose_to_replace(&self, group: ExecutorGroupId, message: impl Into<String>) {
        self.lose(group, message);
        *self
            .closed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = false;
    }

    /// Installs a replacement group: it is ready again and takes work.
    pub fn restore(&self, group: &ExecutorGroupId) {
        let mut lost = self
            .group
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if lost.as_ref().is_some_and(|name| name == group) {
            *lost = None;
        }
    }

    /// Whether `group` was reported lost beyond replacement.
    fn is_closed(&self, group: &ExecutorGroupId) -> bool {
        let lost = self
            .group
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let closed = *self
            .closed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        lost.as_ref().is_some_and(|name| name == group) && closed
    }

    /// Wakes `waker` when a loss is reported, so a scheduler parked on this
    /// executor answers it rather than waiting out its park.
    fn watch(&self, waker: CommandWaker) {
        *self
            .wake
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(waker);
    }

    /// Whether `group` has been reported lost.
    fn is_lost(&self, group: &ExecutorGroupId) -> bool {
        self.group
            .lock()
            .map(|lost| lost.as_ref().is_some_and(|name| name == group))
            .unwrap_or(false)
    }

    /// Takes the report the next poll answers with, if it has not been.
    fn take(&self) -> Option<(ExecutorGroupId, String, Vec<ipc::RequestKey>)> {
        self.pending.lock().ok().and_then(|mut slot| slot.take())
    }
}

pub struct SimExecutor {
    executor_info: ExecutorInfo,
    /// Maximum batches submitted whose results `poll` has not yet returned.
    depth: usize,
    to_worker: Sender<Job>,
    /// Each batch's result paired with the lifecycle commands it was submitted
    /// with, which `poll` needs to interpret the command receipts.
    from_worker: Receiver<anyhow::Result<(BatchResult, Vec<ipc::BatchCommand>)>>,
    /// Capacity-one wake channel: a `command_waker` post makes a blocked
    /// `poll` return early, and posts beyond the first coalesce.
    progress_tx: Sender<()>,
    progress_rx: Receiver<()>,
    in_flight: usize,
    /// Whether a result is delivered only to a poll that waits for it.
    results_on_wait: bool,
    /// Where the boundary events go while a test observes them.
    observer: Option<Sender<BatchEvent>>,
    /// Where the throttles of the deployment's transfers go while a test
    /// observes them: `true` for a halving, `false` for a restoration.
    throttles: Option<Sender<bool>>,
    /// The storage reports of the deployment's nodes a test has the
    /// executor answer with (`Executor::ledgers`), as node daemons report
    /// them; none when unset.
    ledgers: Option<std::sync::Arc<std::sync::Mutex<Vec<String>>>>,
    /// The simulator thread; `None` once the executor is closed.
    handle: Option<JoinHandle<()>>,
    /// The simulator's stop flag: set before the thread is joined, so that a
    /// batch waiting for a permit cannot hold the join open.
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The loss a test reports through [`Self::loss`], answered by `poll`.
    loss: SimLoss,
    /// Requests admitted by a submitted batch whose `Finish` has not been
    /// acknowledged through `poll`.
    admissions: HashSet<ipc::RequestKey>,
    /// Products declared by submitted calls that no acknowledged `Free` or
    /// `Finish` has released yet.
    products: HashSet<TensorRef>,
    /// A result holding failed calls, handed back by the poll after the one
    /// that reports their failure.
    held: Option<(BatchResult, Vec<ipc::BatchCommand>)>,
}

/// What the simulated executor observed at its boundary, in the order it
/// happened: a batch the scheduler submitted and a result it was handed back.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchEvent {
    /// A batch as the executor accepted it.
    Submitted(ExecutionBatch),
    /// The executor handed this batch's result to the scheduler.
    Resolved {
        /// Logical batch identity assigned at submission.
        batch_id: u64,
    },
}

impl SimExecutor {
    /// Starts an asynchronous simulator that admits as many batches in
    /// flight as the pool's queue depth.
    ///
    /// # Panics
    ///
    /// Panics if the simulator thread cannot be spawned.
    #[expect(
        clippy::expect_used,
        reason = "a simulator without its thread cannot run; documented under Panics"
    )]
    pub fn new(mut engine: SimEngine) -> Self {
        let info = engine.info().clone();
        let placement = engine.placement.clone();
        let max_model_len = engine.max_model_len;
        let groups = engine.groups.clone();
        let latent_pools = engine.latent_pools.clone();
        let encoder_pools = engine.encoder_pools.clone();
        let results_on_wait = engine.results_on_wait;
        let stop = std::sync::Arc::clone(&engine.stop);
        let depth = placement.queue_depth.max(1);
        let (to_worker, jobs) = crossbeam_channel::unbounded();
        let (results_tx, from_worker) = crossbeam_channel::unbounded();
        let (progress_tx, progress_rx) = crossbeam_channel::bounded(1);
        let handle = std::thread::Builder::new()
            .name("uniserve-sim-executor".into())
            .spawn(move || {
                // Batches execute one at a time in submission order. Each
                // returns whole, as one terminal result carrying an `Applied`
                // receipt for every command other than `Start`; an execution
                // error is forwarded to `poll` instead.
                while let Ok(job) = jobs.recv() {
                    match job {
                        Job::Batch(batch) => {
                            let commands = batch.commands.clone();
                            let result = engine.execute(batch).map(|report| {
                                (batch_result(report, &commands), commands)
                            });
                            if results_tx.send(result).is_err() {
                                break;
                            }
                        }
                        Job::Shutdown => break,
                    }
                }
            })
            .expect("spawn sim executor thread");
        let mut executor = Self {
            executor_info: {
                ExecutorInfo {
                    workers: groups
                        .iter()
                        .map(|group| (group.clone(), info.clone()))
                        .collect(),
                    placements: groups
                        .iter()
                        .map(|group| (group.clone(), placement.clone()))
                        .collect(),
                    // The pool's own fact, as an inventory states it: the
                    // mechanism shapes a KV view to it, and refuses to shape
                    // one without it.
                    max_model_len,
                    rank_latents: if latent_pools.is_empty() {
                        BTreeMap::new()
                    } else {
                        groups
                            .iter()
                            .map(|group| (group.clone(), latent_pools.clone()))
                            .collect()
                    },
                    rank_encoders: if encoder_pools.is_empty() {
                        BTreeMap::new()
                    } else {
                        groups
                            .iter()
                            .map(|group| (group.clone(), encoder_pools.clone()))
                            .collect()
                    },
                    ..ExecutorInfo::default()
                }
            },
            depth,
            to_worker,
            from_worker,
            progress_tx,
            progress_rx,
            in_flight: 0,
            results_on_wait,
            observer: None,
            throttles: None,
            ledgers: None,
            handle: Some(handle),
            stop,
            loss: SimLoss::default(),
            admissions: HashSet::new(),
            products: HashSet::new(),
            held: None,
        };
        // A loss wakes the scheduler parked on this executor, so it is
        // answered on the next poll rather than at the end of the park.
        executor.loss.watch(executor.command_waker());
        executor
    }

    /// Reports every batch this executor accepts and every result it hands
    /// back, in order, to the returned receiver.
    ///
    /// The receiver disconnects when the executor is dropped, so a reader can
    /// drain the complete record after the scheduler has closed.
    pub fn observe(&mut self) -> Receiver<BatchEvent> {
        let (sender, receiver) = crossbeam_channel::unbounded();
        self.observer = Some(sender);
        receiver
    }

    /// Reports every throttle of the deployment's transfers the executor
    /// takes (`Executor::throttle`), `true` for a halving and `false` for a
    /// restoration, in order, to the returned receiver. The simulator runs
    /// no transfer: a throttle changes nothing else.
    pub fn observe_throttles(&mut self) -> Receiver<bool> {
        let (sender, receiver) = crossbeam_channel::unbounded();
        self.throttles = Some(sender);
        receiver
    }

    /// Has the executor report, as the deployment's nodes' storage reports
    /// (`Executor::ledgers`), whatever the returned list holds when the
    /// engine reads them: a test writes each node's JSON (`{"host",
    /// "ledgers", "host_storage", "transfers"}`) as it means the nodes'
    /// Storage Servers to answer.
    pub fn report_ledgers(&mut self) -> std::sync::Arc<std::sync::Mutex<Vec<String>>> {
        let ledgers = std::sync::Arc::default();
        self.ledgers = Some(std::sync::Arc::clone(&ledgers));
        ledgers
    }

    /// Returns a waker that interrupts executor polling after command enqueue.
    pub fn command_waker(&self) -> CommandWaker {
        let progress = self.progress_tx.clone();
        CommandWaker::new(move || {
            let _ = progress.try_send(());
        })
    }
}

impl SimExecutor {
    /// Waits up to `timeout` for the next batch result and the commands its
    /// batch was submitted with.
    ///
    /// Returns `None` on timeout, on a command wake, after close, and for an
    /// instantaneous poll when results are delivered only on wait. Returns
    /// `Err` for a batch the simulator rejected or a disconnected channel.
    fn poll_batch(
        &mut self,
        timeout: Duration,
    ) -> anyhow::Result<Option<(BatchResult, Vec<ipc::BatchCommand>)>> {
        // A closed simulator produces no results, but the poll still blocks
        // until the timeout or a command wake, as an open one does.
        if self.handle.is_none() {
            let _ = self.progress_rx.recv_timeout(timeout);
            return Ok(None);
        }
        // A paced simulator answers only a caller that waits: an instantaneous
        // check observes nothing, so the batches in flight stay in flight until
        // the scheduler blocks for one of them.
        if self.results_on_wait && timeout.is_zero() {
            return Ok(None);
        }
        let result = crossbeam_channel::select! {
            recv(self.from_worker) -> result => match result {
                Ok(result) => Ok(Some(result?)),
                Err(_) => Err(anyhow::anyhow!("sim executor thread disconnected")),
            },
            recv(self.progress_rx) -> wake => match wake {
                Ok(()) => Ok(None),
                Err(_) => Err(anyhow::anyhow!("sim executor progress channel disconnected")),
            },
            default(timeout) => Ok(None),
        }?;
        if result.is_some() {
            self.in_flight = self.in_flight.saturating_sub(1);
        }
        Ok(result)
    }

    /// A handle that reports one of this executor's groups lost, which the
    /// executor answers on its next poll and reports through `is_ready`.
    ///
    /// The handle outlives the executor's move into a scheduler, so a test
    /// states a loss at the moment it means it.
    pub fn loss(&self) -> SimLoss {
        self.loss.clone()
    }

    /// The failure a result's failed calls amount to: their requests, the
    /// buffers those calls would have produced, and the reason of each request
    /// the simulated model refused.
    /// `None` when every call of the result succeeded.
    fn call_failure(&self, result: &BatchResult) -> Option<ExecutorFailure> {
        let failed = || {
            result
                .results
                .iter()
                .map(|completion| &completion.output)
                .filter(|output| output.status == CallStatus::Error)
        };
        let calls = failed()
            .map(|output| (output.request_key, output.call_id))
            .collect::<HashSet<_>>();
        if calls.is_empty() {
            return None;
        }

        let requests = calls
            .iter()
            .map(|(request, _)| *request)
            .collect::<HashSet<_>>();
        let refusals = failed()
            .filter(|output| output.error_code == Some(ErrorCode::InputError))
            .filter_map(|output| Some((output.request_key, output.error_message.clone()?)))
            .collect();
        let buffers = self
            .products
            .iter()
            .map(TensorRef::buffer_id)
            .filter(|buffer| calls.contains(&(buffer.owner, buffer.producer_call_id)))
            .collect::<HashSet<_>>();
        Some(ExecutorFailure {
            worker_id: ExecutorGroupId("sim".to_owned()),
            endpoints: Vec::new(),
            requests: requests.into_iter().collect(),
            refusals,
            retired: Vec::new(),
            buffers: buffers.into_iter().collect(),
            execution: crate::executor::ExecuteError::non_finite(
                result.batch_id,
                result.results.iter().map(|completion| &completion.output),
            ),
            message: "call failed before its dependent work could complete".into(),
        })
    }

    /// Closes the simulated physical executor.
    fn shutdown(&mut self) -> anyhow::Result<()> {
        // A batch waiting for its permit would otherwise never return, and
        // the join below would wait for it forever.
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        let _ = self.to_worker.send(Job::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        Ok(())
    }
}

impl Executor for SimExecutor {
    fn has_capacity(&self, worker: &ExecutorGroupId) -> bool {
        self.is_ready(worker) && self.in_flight < self.depth
    }

    /// Whether a lifecycle command can be submitted now.
    ///
    /// A command with no owner here (a `Free` of a buffer that holds no
    /// tracked product, or any other command whose request is not admitted)
    /// targets no physical state and always has capacity. A command with an
    /// owner needs an open simulator with a free in-flight slot.
    fn command_has_capacity(&self, command: &ipc::BatchCommand) -> bool {
        use ipc::BatchCommand;
        let has_owner = match command {
            BatchCommand::Free { buffer } => self
                .products
                .iter()
                .any(|product| product.buffer_id() == *buffer),
            _ => self.admissions.contains(&command.request_key()),
        };
        !has_owner || (self.handle.is_some() && self.in_flight < self.depth)
    }

    /// A lost group is closed only once its replacement has failed: while one
    /// is starting it is not ready, and the scheduler waits for it.
    fn is_closed(&self, worker: &ExecutorGroupId) -> bool {
        self.handle.is_none() || self.loss.is_closed(worker)
    }

    fn is_ready(&self, worker: &ExecutorGroupId) -> bool {
        self.handle.is_some()
            && !self.loss.is_lost(worker)
            && self
                .executor_info
                .workers
                .iter()
                .any(|(id, _)| id == worker)
    }

    /// Records the throttle for an observer; the simulator throttles no
    /// host.
    fn throttle(&mut self, halve: bool) -> Result<Vec<String>, crate::executor::ExecutorError> {
        if let Some(throttles) = &self.throttles {
            let _ = throttles.send(halve);
        }
        Ok(Vec::new())
    }

    /// The storage reports a test set (`Self::report_ledgers`); none
    /// otherwise.
    fn ledgers(&mut self) -> Result<Vec<String>, crate::executor::ExecutorError> {
        Ok(self.ledgers.as_ref().map_or_else(Vec::new, |ledgers| {
            ledgers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }))
    }

    /// Returns the executor metadata.
    fn info(&self) -> &ExecutorInfo {
        &self.executor_info
    }

    /// Validates a logical batch and queues it for the simulator thread.
    ///
    /// Returns `WouldBlock` with the batch while `depth` batches are in
    /// flight. Returns `Failed` when the executor is closed, a placement names
    /// an executor other than the simulator, the batch fails validation, or the
    /// simulator thread is gone. The batch's admissions and products are
    /// recorded before it is queued and are not rolled back if queuing fails.
    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        if self.handle.is_none() {
            return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                "Executor is closed"
            )));
        }
        for (_, placement) in &batch.requests {
            if !self.is_ready(&placement.worker) {
                return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                    "Worker cannot accept this request"
                )));
            }
        }
        if self.in_flight >= self.depth {
            return Err(ExecutorSubmitError::WouldBlock(Box::new(batch)));
        }
        batch.validate().map_err(ExecutorSubmitError::Failed)?;

        self.admissions
            .extend(batch.admissions().map(|request| request.request_key));
        self.products.extend(
            batch
                .requests
                .iter()
                .flat_map(|(call, _)| call.tensor_outputs().cloned()),
        );
        if let Some(observer) = &self.observer {
            let _ = observer.send(BatchEvent::Submitted(batch.clone()));
        }
        self.to_worker.send(Job::Batch(batch)).map_err(|_| {
            ExecutorSubmitError::Failed(anyhow::anyhow!("sim executor thread gone"))
        })?;
        self.in_flight += 1;
        Ok(())
    }

    /// Returns the next batch result and applies its acknowledged `Free` and
    /// `Finish` commands to the ownership sets `command_has_capacity` reads.
    fn poll(&mut self, timeout: Duration) -> anyhow::Result<Option<BatchResult>> {
        // A lost group is reported before any result: its work is gone, and a
        // closed group's loss is engine-fatal when no other model group
        // serves.
        if let Some((worker_id, message, requests)) = self.loss.take() {
            return Err(anyhow::Error::new(ExecutorFailure {
                worker_id,
                endpoints: Vec::new(),
                requests,
                refusals: Default::default(),
                retired: Vec::new(),
                buffers: Vec::new(),
                execution: None,
                message,
            }));
        }
        let (result, commands) = match self.held.take() {
            Some(held) => held,
            None => {
                let Some((result, commands)) = self.poll_batch(timeout)? else {
                    return Ok(None);
                };
                if let Some(failure) = self.call_failure(&result) {
                    self.held = Some((result, commands));
                    return Err(failure.into());
                }
                (result, commands)
            }
        };
        for receipt in &result.command_results {
            // A failed command keeps its physical ownership.
            if receipt.outcome == crate::executor::CommandOutcome::Failed {
                continue;
            }
            // Receipts number lifecycle commands with `Start` admissions
            // skipped, as `batch_result` assigns them.
            let command = commands
                .iter()
                .filter(|command| !matches!(command, ipc::BatchCommand::Start { .. }))
                .nth(receipt.command_index as usize)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "command receipt {} names no command of its batch",
                        receipt.command_index
                    )
                })?;
            use ipc::BatchCommand;
            match command {
                BatchCommand::Free { buffer } => self
                    .products
                    .retain(|product| product.buffer_id() != *buffer),
                BatchCommand::Finish {
                    request_key,
                    retained_buffers,
                    ..
                } => {
                    // A finished request keeps only the products in its
                    // retained buffers.
                    self.admissions.remove(request_key);
                    self.products.retain(|product| {
                        product.request_key != *request_key
                            || retained_buffers.contains(&product.buffer_id())
                    });
                }
                _ => {}
            }
        }
        if let Some(observer) = &self.observer {
            let _ = observer.send(BatchEvent::Resolved {
                batch_id: result.batch_id,
            });
        }
        Ok(Some(result))
    }

    /// Stops and joins the simulator thread; later submissions fail.
    fn close(&mut self) -> anyhow::Result<()> {
        self.shutdown()
    }

    /// Answers the engine's own executor hooks as an executor without a device
    /// does: its cache is not split into segments, it gives back no storage,
    /// it has no fence probes, and a profiling session it takes part in runs
    /// no collector and writes no file. It loads and unloads LoRA adapters
    /// as an executor with a LoRA plugin does, answering the slot; other plugin
    /// hooks fail, as no plugin is loaded.
    fn hook(&mut self, name: &str, payload: &str) -> anyhow::Result<Vec<String>> {
        let result = match name {
            "uniserve.cache_storage" => serde_json::json!({
                "segments": [],
                "held": 0,
            }),
            "uniserve.release_cache"
            | "uniserve.restore_cache"
            | "uniserve.sleep"
            | "uniserve.wake" => serde_json::Value::Null,
            "uniserve.update_weights" => serde_json::json!({ "group": null, "page_contents": [] }),
            "uniserve.eager" => {
                let payload: serde_json::Value = serde_json::from_str(payload)?;
                serde_json::json!({ "eager": payload["eager"] })
            }
            "uniserve.profile_start" => {
                let start: proto::profile::SessionStart = serde_json::from_str(payload)?;
                let collectors: serde_json::Map<String, serde_json::Value> = start
                    .request
                    .collectors
                    .iter()
                    .map(|collector| {
                        (
                            collector.as_str().to_owned(),
                            serde_json::json!("unavailable: the simulated executor has no device"),
                        )
                    })
                    .collect();
                serde_json::json!({ "rank": 0, "selected": true, "path": null, "collectors": collectors })
            }
            "uniserve.profile_stop" => {
                let payload: serde_json::Value = serde_json::from_str(payload)?;
                serde_json::json!({
                    "rank": 0,
                    "session": payload["session"],
                    "steps": 0,
                    "reason": payload["reason"],
                    "file": null,
                })
            }
            "uniserve.profile_release" => serde_json::json!({ "rank": 0, "released": false }),
            "uniserve.fence_switch" => {
                let payload: serde_json::Value = serde_json::from_str(payload)?;
                serde_json::json!({
                    "probes": false,
                    "enabled": payload["enabled"],
                    "off": payload["off"],
                })
            }
            "load_lora_adapter" | "unload_lora_adapter" => {
                let payload: serde_json::Value = serde_json::from_str(payload)?;
                serde_json::json!({ "adapter": payload["adapter"] })
            }
            other => anyhow::bail!("no plugin implements the executor hook {other}"),
        };
        Ok(vec![result.to_string()])
    }
}

impl Drop for SimExecutor {
    /// Stops and joins the simulator thread.
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// Current synthetic execution state and products retained for their consumers.
#[derive(Clone)]
struct SimRequestState {
    admission: NewRequest,
    /// Logical position the next chained call starts at.
    logical_position: u32,
    /// Whether the model, rather than this simulator, owns the request's
    /// logical positions. An image contributes positions through its rope
    /// advance, which the simulator does not implement, so once a request has
    /// ingested one it takes the position each call states.
    positions_from_model: bool,
    /// Tokens resident in the prefix cache. The simulator reports it as both the visible
    /// and the computed extent.
    cache_visible_len: u32,
    /// Index of the next synthetic text token, as `synth_logits` takes it.
    emitted: usize,
    /// Tokens the request generated on a predecessor engine
    /// (`GenerationParams::generated_token_ids`): its synthetic tokens
    /// continue after them.
    history: usize,
    /// The step index of the request's trajectory: the solver steps it has
    /// completed. Image decoding resets it to zero, as it does on an executor
    /// rank.
    step_index: u16,
    /// Boolean values of the request's resolved products that later calls
    /// may name as predicates: a token output's continuation value, a
    /// transition output's value (whether a sampled token is a transition
    /// token, otherwise true), a completion output's `true`, and `false` for
    /// every output of a call whose predicate was false.
    predicate_values: HashMap<TensorRef, bool>,
    /// Committed penalty counts in ascending token order.
    ///
    /// Each generated token folds in during execution, allowing a successor to
    /// observe it before the predecessor is host-visible. Device executors encode
    /// the same state as a resident count tensor plus per-call deltas.
    penalty_counts: BTreeMap<u32, u32>,
    /// The latest state-advancing call that completed with status OK, or the
    /// admission root: the call a later chained call follows, as a rank
    /// derives it from its own request state.
    state_call_id: ipc::CallId,
    /// The generation canvas step the request's next canvas call must run:
    /// blocks stopped so far and steps run on the current block.
    canvas: ipc::CanvasStep,
    /// The token the request's last sampling call sampled, which a decode
    /// that relays its predecessor's token writes.
    last_token: Option<u32>,
    /// The image index the request's current trajectory was prepared for.
    image_index: u32,
    /// What each image or feature product the request holds was made from:
    /// the digest of a decoded image's values, and for a feature, the
    /// posterior draw a latent encoder sampled.
    provenance: HashMap<TensorRef, (u64, Option<u32>)>,
}

impl SimRequestState {
    /// Whether a call follows the request's state chain. Every call of a
    /// generation request does; on the media path only a state-advancing
    /// call does, and the branches around it complete independently.
    fn chained(&self, call: &Call) -> bool {
        self.admission.diffusion.is_none() || call.advances_state()
    }

    /// Creates request state from an admitted request.
    fn new(admission: NewRequest) -> Self {
        let prefix_len = admission
            .generation
            .as_ref()
            .map_or(0, |branch| branch.initial_position);
        // A resumed request continues its synthetic tokens and penalties
        // after the ones it generated before.
        let generated = admission
            .generation
            .as_ref()
            .map_or(&[][..], |branch| branch.generated_token_ids.as_slice());
        let mut penalty_counts = BTreeMap::new();
        for token in generated {
            *penalty_counts.entry(*token).or_insert(0_u32) += 1;
        }
        let history = generated.len();
        Self {
            admission,
            logical_position: prefix_len,
            positions_from_model: false,
            cache_visible_len: prefix_len,
            emitted: history,
            history,
            step_index: 0,
            predicate_values: HashMap::new(),
            penalty_counts,
            state_call_id: ipc::CallId::new(0, 0),
            canvas: ipc::CanvasStep::default(),
            last_token: None,
            image_index: 0,
            provenance: HashMap::new(),
        }
    }

    /// Returns the committed penalty histogram in ascending token order.
    fn recent_counts(&self) -> Vec<(u32, u32)> {
        self.penalty_counts
            .iter()
            .map(|(token, count)| (*token, *count))
            .collect()
    }

    /// Folds one generated token into the committed penalty base.
    fn fold_penalty_token(&mut self, token: u32) {
        self.penalty_counts
            .entry(token)
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);
    }

    /// Returns shared access to the simulated sampling configuration.
    fn sampling(&self) -> Option<&SamplingParams> {
        self.admission.generation.as_ref().map(|und| &und.sampling)
    }

    /// Returns shared access to the simulated image configuration.
    fn image(&self) -> Option<&ImageParams> {
        self.admission.image.as_ref()
    }
}

/// Token chosen for one simulated sampling step with its scores.
struct SampleOutput {
    /// Selected vocabulary token.
    token: u32,
    /// Ranked candidates as `(token_id, logprob, rank)`, the sampled token
    /// first, then the most probable alternatives and the explicitly
    /// requested tokens, without duplicates; `rank` is the one-based
    /// competition rank. Empty when the request asks for no logprobs.
    top: Vec<(u32, f32, u32)>,
}

/// Selects a simulated token from `logits` under the request's sampling
/// parameters.
///
/// Applies, in order, the allowed-token and suppression masks, repetition,
/// frequency and presence penalties over `recent_counts` (`(token, count)`
/// pairs), logit bias and temperature, then takes the first maximum when the
/// temperature is zero and otherwise inverts the cumulative distribution at
/// `draw`, a uniform in `[0, 1)`. The simulator exists to exercise the
/// scheduler with tokens that depend on exactly what the scheduler stages, so
/// the rank's truncation processors (top-k, top-p, min-p, typical) are not
/// simulated.
///
/// Returns `None` when the masks leave no finite candidate. `logits` is
/// modified in place.
fn select_token(
    logits: &mut [f32],
    sampling: &SamplingParams,
    recent_counts: &[(u32, u32)],
    allowed: Option<&[u32]>,
    suppress: Option<&[u32]>,
    draw: f32,
) -> Option<SampleOutput> {
    const MASKED: f32 = f32::NEG_INFINITY;
    let vocabulary = logits.len();

    if let Some(allowed) = allowed {
        let mut keep = vec![false; vocabulary];
        for &token in allowed {
            if let Some(slot) = keep.get_mut(token as usize) {
                *slot = true;
            }
        }
        for (logit, keep) in logits.iter_mut().zip(keep) {
            if !keep {
                *logit = MASKED;
            }
        }
    }
    for &token in suppress.unwrap_or_default() {
        if let Some(logit) = logits.get_mut(token as usize) {
            *logit = MASKED;
        }
    }

    // Repetition scales a logit away from selection by its sign; frequency
    // and presence subtract.
    for &(token, count) in recent_counts {
        let Some(logit) = logits.get_mut(token as usize) else {
            continue;
        };
        if count == 0 || *logit == MASKED {
            continue;
        }
        if sampling.repetition_penalty != 1.0 {
            *logit = if *logit > 0.0 {
                *logit / sampling.repetition_penalty
            } else {
                *logit * sampling.repetition_penalty
            };
        }
        *logit -= sampling.frequency_penalty * count as f32 + sampling.presence_penalty;
    }
    for &(token, bias) in &sampling.logit_bias {
        if let Some(logit) = logits.get_mut(token as usize)
            && *logit != MASKED
        {
            *logit += bias;
        }
    }

    let greedy = sampling.temperature <= 0.0;
    if !greedy {
        for logit in logits.iter_mut().filter(|logit| **logit != MASKED) {
            *logit /= sampling.temperature;
        }
    }

    let peak = logits.iter().copied().fold(MASKED, f32::max);
    if !peak.is_finite() {
        return None;
    }
    let weights: Vec<f32> = logits
        .iter()
        .map(|&logit| {
            if logit == MASKED {
                0.0
            } else {
                (logit - peak).exp()
            }
        })
        .collect();
    let total: f32 = weights.iter().sum();

    let token = if greedy {
        logits.iter().position(|&logit| logit == peak)?
    } else {
        // The first candidate whose cumulative mass passes the draw; the last
        // candidate when rounding leaves the total at or below it.
        let target = draw * total;
        let mut cumulative = 0.0;
        let mut selected = 0;
        for (index, &weight) in weights.iter().enumerate() {
            if weight > 0.0 {
                cumulative += weight;
                selected = index;
                if cumulative > target {
                    break;
                }
            }
        }
        selected
    };

    // Scores follow the final masked, temperature-scaled distribution.
    let log_total = total.ln();
    let logprob_of = |index: usize| {
        if logits[index] == MASKED {
            MASKED
        } else {
            logits[index] - peak - log_total
        }
    };
    let rank_of = |value: f32| {
        1 + (0..vocabulary)
            .filter(|&index| logprob_of(index) > value)
            .count() as u32
    };
    let sampled_logprob = logprob_of(token);

    let mut top = Vec::new();
    if sampling.generated_logprobs_requested() {
        top.push((token as u32, sampled_logprob, rank_of(sampled_logprob)));
        let mut order: Vec<usize> = (0..vocabulary)
            .filter(|&index| logits[index] != MASKED)
            .collect();
        // Stable, so tied candidates keep ascending vocabulary order.
        order.sort_by(|&a, &b| logprob_of(b).total_cmp(&logprob_of(a)));
        for index in order.into_iter().take(sampling.n_logprobs as usize) {
            let token_id = index as u32;
            if index < vocabulary && !top.iter().any(|&(existing, ..)| existing == token_id) {
                let value = logprob_of(index);
                top.push((token_id, value, rank_of(value)));
            }
        }
    }

    Some(SampleOutput {
        token: token as u32,
        top,
    })
}

/// Deterministic local model engine that implements the executor lifecycle protocol.
/// How the simulated executor speculates: what its drafter proposes and the
/// executor time each step reports.
#[derive(Clone, Copy, Debug)]
pub struct SimSpeculation {
    /// The leading drafts of each proposal that are the request's next
    /// tokens; the drafts after them are wrong.
    pub accepted: usize,
    /// Executor time of a decoding step none of whose verifications carries
    /// drafts, in microseconds.
    pub decode_us: u64,
    /// Executor time of a decoding step one of whose verifications carries
    /// drafts, in microseconds.
    pub verify_us: u64,
    /// Executor time of a prompt step per token it evaluates, in
    /// microseconds.
    pub prompt_token_us: u64,
}

impl SimSpeculation {
    /// The executor time of a step of `calls`: its prompt tokens' and, with
    /// decodes or verifications among them, one decoding step's.
    fn step_us<'a>(&self, calls: impl Iterator<Item = &'a Call>) -> u64 {
        let mut prompt_tokens = 0;
        let mut decoding = None;
        for call in calls {
            match call.kind {
                CallKind::Forward(ForwardMode::Prefill) => {
                    prompt_tokens += call.input_token_ids.len() as u64;
                }
                CallKind::Forward(ForwardMode::Decode) => {
                    decoding.get_or_insert(false);
                }
                CallKind::Forward(ForwardMode::Verify) => {
                    let drafted = call.input_token_ids.len() > 1;
                    *decoding.get_or_insert(false) |= drafted;
                }
                _ => {}
            }
        }
        let step = decoding.map_or(0, |drafted| {
            if drafted {
                self.verify_us
            } else {
                self.decode_us
            }
        });
        prompt_tokens * self.prompt_token_us + step
    }
}

pub struct SimEngine {
    info: MainProcessInfo,
    /// How the simulated pool is placed: its queue depth and its components'
    /// placement, which `SimExecutor` states to the scheduler.
    placement: PoolPlacement,
    /// The longest sequence the simulated pool serves: the model's context
    /// length, which `SimExecutor` states to the scheduler as
    /// `ExecutorInfo::max_model_len`.
    ///
    /// The mechanism bounds every attention row by it, so a pool that states
    /// none is refused rather than shaped to a bound nobody gave.
    max_model_len: u32,
    /// Synthetic tokens sampled before `fake_eos` dominates.
    text_len: usize,
    fake_eos: u32,
    /// Logit row length. It covers the synthetic text ids, `fake_eos`, and
    /// every configured control token, and never shrinks.
    vocab: usize,
    results_on_wait: bool,
    /// Readout candidate whose log-probability is reported as NaN, as a
    /// executor with a numerical defect reports it.
    non_finite_candidate: Option<u32>,
    /// Block-diffusion sampling the simulated executor runs under.
    canvas_sampling: Option<ipc::CanvasSampling>,
    /// How the simulated executor speculates, when it runs a drafter.
    speculation: Option<SimSpeculation>,
    /// Bytes each published media artifact declares beyond what it stores.
    ///
    /// A rank's report states its payload's extent, and the engine claims the
    /// object by that statement; a test states the disagreement a rank can
    /// report and cannot store ([`SimEngine::overstate_published_media`]).
    media_overstatement: u64,
    /// The executor groups the simulated deployment runs: one named `sim` unless
    /// a test names more.
    ///
    /// A deployment's groups serve different components and fail
    /// independently, so a statement about one group
    /// failing while another serves needs more than one here.
    groups: Vec<ExecutorGroupId>,
    /// Releases for batches that skip a predicated call: such a batch returns
    /// only after taking one release, or once the sender is dropped.
    predicated_release: Option<Receiver<()>>,
    /// Permits for batches: each batch executes only after taking one, or
    /// once the sender is dropped ([`SimEngine::pace`]).
    pace: Option<Receiver<()>>,
    /// Set when the executor that drives this simulator shuts down: a batch
    /// waiting for a permit gives up instead of holding the thread its
    /// executor joins on.
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Admitted requests by id, removed when a `Finish` for the same request
    /// key executes.
    requests: HashMap<RequestId, SimRequestState>,
    /// Requests the simulated model refuses, with the reason it states: each
    /// of their media calls completes with `ErrorCode::InputError`, as a
    /// rank reports a request whose size its model does not serve.
    refused: HashMap<RequestId, String>,
    /// The host latent pools of the simulated pool's ranks, with each
    /// rank's host (`ExecutorInfo::rank_latents`); empty without them.
    latent_pools: Vec<(String, Option<ipc::LatentPoolInfo>)>,
    /// How each simulated denoising rank holds a request's samples, which
    /// every video latent preparation reports (`CallOutput::latent_layouts`).
    latent_layouts: Vec<ipc::RankLatentLayout>,
    /// The host encoder pools of the simulated pool's ranks, with each
    /// rank's host (`ExecutorInfo::rank_encoders`); empty without them.
    encoder_pools: crate::executor::RankEncoderPools,

    /// What each request's calls wrote into its context, shared with the
    /// test that reads it ([`SimEngine::context_log`]).
    context: ContextLog,
}

/// The permits of a paced simulator ([`SimEngine::pace`]).
#[derive(Debug)]
pub struct Pace(Sender<()>);

impl Pace {
    /// Lets the simulator execute one more batch.
    pub fn step(&self) {
        let _ = self.0.send(());
    }

    /// Lets the simulator execute one more batch unless it has not taken the
    /// last permit yet. A simulator with nothing to execute leaves its permits
    /// queued, and each queued permit later runs a batch no step of the test
    /// asked for.
    pub fn offer(&self) {
        if self.0.is_empty() {
            self.step();
        }
    }
}

/// One write a simulated request's call made to its context: the KV cache a
/// executor holds for it, in write order. Two requests whose logs are equal
/// hold the same context, whichever calls wrote it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextWrite {
    /// A token at a logical position: a prompt token, a decode's input or a
    /// cache close.
    Token { position: u32, token: u32 },
    /// An input image's block of a context prefill or a latent block, at the
    /// logical position its call starts at.
    InputBlock { position: u32 },
    /// A generated image's feedback feature at a logical position: the
    /// digest of the decoded image values its encoder read, and the
    /// posterior draw a latent encoder sampled.
    Image {
        position: u32,
        image: u64,
        draw: Option<u32>,
    },
}

/// The context writes of every request a simulator served, by request
/// identity and admission, kept after the request finishes.
#[derive(Debug, Clone, Default)]
pub struct ContextLog(std::sync::Arc<std::sync::Mutex<ContextWrites>>);

/// Each request's admissions, by request identity: each admission's key and
/// the writes of its calls, in order.
type ContextWrites = HashMap<RequestId, Vec<(ipc::RequestKey, Vec<ContextWrite>)>>;

impl ContextLog {
    /// The writes the calls of each admission of request `id` made, in
    /// admission order: a request the engine admits again, as a migrated
    /// one, has a context of its own each time.
    pub fn admissions(&self, id: RequestId) -> Vec<Vec<ContextWrite>> {
        crate::lock(&self.0)
            .get(&id)
            .map(|admissions| {
                admissions
                    .iter()
                    .map(|(_, writes)| writes.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The identities of the requests whose calls wrote a context.
    pub fn requests(&self) -> Vec<RequestId> {
        let mut requests: Vec<RequestId> = crate::lock(&self.0).keys().copied().collect();
        requests.sort_by_key(|id| id.0);
        requests
    }

    fn record(&self, key: ipc::RequestKey, writes: Vec<ContextWrite>) {
        let mut log = crate::lock(&self.0);
        let admissions = log.entry(key.request_id).or_default();
        match admissions.last_mut() {
            Some((admitted, existing)) if *admitted == key => existing.extend(writes),
            _ => admissions.push((key, writes)),
        }
    }
}

impl SimEngine {
    /// Control tokens consistent with the simulator's synthetic text: its fake
    /// EOS id terminates a request.
    pub fn special_tokens() -> SpecialTokenIds {
        SpecialTokenIds {
            bos: 151_644,
            eos: vec![FAKE_EOS_TOKEN, 151_643],
            end_of_image: 151_653,
        }
    }

    /// Model limits of a deployment without a model-imposed bound.
    ///
    /// The scheduler takes the served features from the call kinds the
    /// simulated executor reports and clamps these limits to its capacities.
    pub fn generation_limits() -> crate::GenerationLimits {
        crate::GenerationLimits {
            max_latent_units: u64::MAX,
            latent_downsample: 16,
            max_vae_grid_tokens: u32::MAX,
            max_vit_grid_tokens: u32::MAX,
            // One latent unit of the simulated image: 256 MiB over the
            // simulated executor's 65,536 latent units.
            latent_bytes_per_unit: 4096,
            max_latent_feature_bytes: 256 << 20,
            max_vision_feature_bytes: 256 << 20,
            commit_marker_tokens: 2,
            max_cfg_branches: 3,
            encoder_cache_entries: 256,
            ..Default::default()
        }
    }

    /// Constructs a simulator with deterministic text and image capabilities.
    pub fn new() -> Self {
        // Unset fields, including the prefix cache, come from
        // `MainProcessInfo::default`. One latent page is the reserved sentinel
        // page that `latent_capacity_units` excludes. The pool starts with a
        // queue depth of one and no placed components.
        let info = MainProcessInfo {
            supported_calls: CallKind::ALL.to_vec(),
            latent_page_units: 64,
            latent_pages: 1_025,
            buffer_pool_bytes: 257_u64 * (256 << 20),
            encoder_cache_entries: 256,
            encoder_entry_bytes: 256 << 20,
            max_batch_calls: 1024,
            max_unresolved_calls: 2,
            // One batch in flight (`queue_depth` 1) of up to
            // `max_batch_calls` calls, each holding one transfer entry.
            transfer_tickets: 1024,
            ..MainProcessInfo::default()
        };
        Self {
            info,
            placement: PoolPlacement {
                queue_depth: 1,
                ranks: 1,
                components: BTreeMap::new(),
            },
            groups: vec![ExecutorGroupId("sim".to_owned())],
            max_model_len: SIM_MAX_MODEL_LEN,
            text_len: DEFAULT_TEXT_LEN,
            fake_eos: FAKE_EOS_TOKEN,
            vocab: SYNTH_VOCAB_SIZE,
            results_on_wait: false,
            non_finite_candidate: None,
            canvas_sampling: None,
            speculation: None,
            media_overstatement: 0,
            predicated_release: None,
            pace: None,
            stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            requests: HashMap::new(),
            refused: HashMap::new(),
            latent_pools: Vec::new(),
            latent_layouts: Vec::new(),
            encoder_pools: Vec::new(),
            context: ContextLog::default(),
        }
    }

    /// The longest sequence this simulated pool serves: the number
    /// `SimExecutor` states as `ExecutorInfo::max_model_len`.
    pub const fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    /// States the longest sequence this pool is launched for, as a launch file
    /// states a real rank's: a world built for another model length says so
    /// here rather than inheriting the simulator's default.
    pub const fn set_max_model_len(&mut self, tokens: u32) {
        self.max_model_len = tokens;
    }

    /// Produces deterministic logits for one simulated autoregressive position.
    ///
    /// Before `text_len`, the peak is a natural token in `1_000..6_000`
    /// derived from the request id and `index`; up to two lower-scoring
    /// alternates in the same range and a weak EOS follow it. From `text_len`
    /// on, EOS dominates. `vocab` must cover that range and `fake_eos`.
    fn synth_logits(
        vocab: usize,
        text_len: usize,
        fake_eos: u32,
        request_id: RequestId,
        index: usize,
    ) -> Vec<f32> {
        let mut logits = vec![0.0; vocab];
        let natural = if index >= text_len {
            fake_eos
        } else {
            sim_text_token(request_id, index)
        };
        logits[natural as usize] = 10.0;
        let alternate_one = 1_000 + ((request_id.0 as u32 * 13 + index as u32 + 1) % 5_000);
        let alternate_two = 1_000 + ((request_id.0 as u32 * 29 + index as u32 + 2) % 5_000);
        if alternate_one != natural {
            logits[alternate_one as usize] = 8.0;
        }
        if alternate_two != natural {
            logits[alternate_two as usize] = 6.0;
        }
        logits[fake_eos as usize] = if index >= text_len { 100.0 } else { 1.0 };
        logits
    }

    /// Applies request sampling controls to deterministic simulated logits.
    ///
    /// Returns `None` when the request's masks leave no candidate (see
    /// `select_token`). A stochastic call draws from the admitted seed at the
    /// sampled token's position, as a rank does. A request admitted without
    /// an autoregressive branch takes the natural synthetic token directly.
    fn sample(
        vocab: usize,
        text_len: usize,
        fake_eos: u32,
        call: &Call,
        request: &SimRequestState,
        index: usize,
        state: Option<&SamplingState>,
    ) -> anyhow::Result<Option<SampleOutput>> {
        let request_id = call.request_key.request_id;
        let mut logits = Self::synth_logits(vocab, text_len, fake_eos, request_id, index);
        match request.sampling() {
            Some(sampling) => {
                let draw = if sampling.temperature > 0.0 {
                    // The draw follows the admitted seed and the sampled
                    // token's logical position, which a prefill reaches past
                    // its query tokens and a decode or verify one past its
                    // first, as a rank keys it.
                    let query_tokens = match call.kind {
                        CallKind::Forward(ForwardMode::Prefill) => call.bounds.max_tokens,
                        _ => 1,
                    };
                    let position = u64::from(request.logical_position) + u64::from(query_tokens);
                    let key = philox::request_sampling_key(
                        sampling.seed,
                        call.request_key.engine_id,
                        request_id.0,
                        call.request_key.request_epoch,
                    );
                    philox::sampling_uniform(key, position, 0, 0)
                } else {
                    0.0
                };
                // Penalty counts are device-resident, not carried in the host
                // state: the successor reads the committed base folded from
                // ancestral tokens before any of them is host-observed.
                let recent_counts = request.recent_counts();
                // Per-step masks omit an unchanged request allowlist. A dynamic
                // suppression delta must still preserve that admitted constraint.
                let allowed = state
                    .and_then(|value| value.allowed_token_ids.as_deref())
                    .or(sampling.allowed_token_ids.as_deref());
                let suppress = state
                    .map(|value| value.suppressed_token_ids.as_slice())
                    .filter(|tokens| !tokens.is_empty());
                // Forced-token constraint: every simulated sampling call samples
                // a single span point, so its forced token is the first entry
                // of the schedule and overrides any allowed-token mask.
                let forced = sampling
                    .forced_token_ids
                    .first()
                    .copied()
                    .map(|token| [token]);
                let allowed = forced.as_ref().map(|slot| slot.as_slice()).or(allowed);
                Ok(select_token(
                    &mut logits,
                    sampling,
                    &recent_counts,
                    allowed,
                    suppress,
                    draw,
                ))
            }
            None => Ok(Some(SampleOutput {
                token: if index >= text_len {
                    fake_eos
                } else {
                    1_000 + ((request_id.0 as u32 * 7 + index as u32) % 5_000)
                },
                top: Vec::new(),
            })),
        }
    }

    /// Executes one call against its request and returns its [`CallOutput`].
    ///
    /// Advances the request's positions, cache extent, synthetic text index,
    /// penalty counts, and denoising step as the call kind requires, and
    /// records the boolean values of the call's products in
    /// `predicate_values`. Errors when a chained call states coordinates other
    /// than the request's own, or when synthetic media encoding or export
    /// fails. A sampling pipeline with no valid distribution yields an
    /// `Error` record with `InvalidDescriptor` instead.
    #[expect(
        clippy::too_many_arguments,
        reason = "the simulated model's settings, which the batch's calls borrow beside the engine's request map"
    )]
    fn execute_call(
        vocab: usize,
        text_len: usize,
        fake_eos: u32,
        non_finite_candidate: Option<u32>,
        canvas_sampling: Option<ipc::CanvasSampling>,
        speculation: Option<SimSpeculation>,
        media_overstatement: u64,
        call: &Call,
        request: &mut SimRequestState,
    ) -> anyhow::Result<CallOutput> {
        // A call states the coordinates it executes at. The simulator holds the
        // same request state a rank does, so a disagreement is a scheduling
        // failure and not a condition it can execute through. A media branch
        // call around the chain carries no coordinates, and a draft cache
        // refill states the earlier positions it evaluates again.
        if request.chained(call) && call.draft.refill.is_none() {
            let stated = call.coordinates;
            if request.positions_from_model {
                request.logical_position = stated.logical_position;
            }
            let held = ipc::CallCoordinates {
                logical_position: request.logical_position,
                cache_visible_len: request.cache_visible_len,
                cache_computed_len: request.cache_visible_len,
            };
            anyhow::ensure!(
                stated == held,
                "call {:?} states coordinates {stated:?} that disagree with the request's own progress {held:?}",
                call.call_id
            );
        }
        let mut record = CallOutput {
            token_logprobs: Vec::new(),
            prompt_logprobs: Vec::new(),
            request_key: call.request_key,
            call_id: call.call_id,
            status: CallStatus::Ok,
            error_code: None,
            error_message: None,
            timing_counters: TimingCounters::default(),
            cache_visible_len: 0,
            cache_computed_len: 0,
            step_index: 0,
            committed_tokens: Vec::new(),
            candidate_logprobs: Vec::new(),
            finish_flags: FinishFlags::default(),
            media_output: None,
            cache_output: None,
            draft_token_ids: Vec::new(),
            latent_layouts: Vec::new(),
        };
        set_cache_lengths(&mut record, request.cache_visible_len);

        match call.kind {
            // A draft cache refill evaluates positions the cache holds again
            // for the drafter alone: the request's progress stays.
            CallKind::Forward(ForwardMode::Prefill) if call.draft.refill.is_some() => {}
            work @ (CallKind::Forward(ForwardMode::Prefill)
            | CallKind::Forward(ForwardMode::Decode)
            | CallKind::Forward(ForwardMode::Verify)) => {
                // A call that writes image blocks, alone or among the prompt
                // tokens of a context prefill, extends the prefix cache by its token bound
                // but leaves logical positions to the model.
                let visual_state =
                    !call.vision_inputs.is_empty() || call.latent_feature_input.is_some();
                let samples_token = call.token_output.is_some();
                // A call that extends the prompt scores its tokens when the
                // request asks, as a rank does: every token but the prompt's
                // first, which no position predicts.
                let extends_prompt = work == CallKind::Forward(ForwardMode::Prefill)
                    && call.latent_feature_input.is_none()
                    && call.completion_output.is_none();
                if extends_prompt
                    && request
                        .sampling()
                        .is_some_and(SamplingParams::prompt_logprobs_requested)
                {
                    let scored = usize::from(call.coordinates.logical_position == 0);
                    record.prompt_logprobs = call
                        .input_token_ids
                        .iter()
                        .skip(scored)
                        .map(|&token_id| {
                            vec![TokenLogprob {
                                token_id,
                                logprob: sim_candidate_logprob(
                                    call.request_key.request_id,
                                    token_id,
                                ),
                                rank: 1,
                            }]
                        })
                        .collect();
                }
                if visual_state {
                    request.positions_from_model = true;
                    request.cache_visible_len = request
                        .cache_visible_len
                        .saturating_add(call.bounds.max_tokens);
                    set_cache_lengths(&mut record, request.cache_visible_len);
                    if call.completion_output.is_some() {
                        request.emitted = 0;
                    }
                }
                if !visual_state && !samples_token {
                    // A readout's prompt chunk samples nothing and advances
                    // positions with its tokens; the `CloseCache` write, which
                    // declares a completion, only extends the prefix cache.
                    request.cache_visible_len = request
                        .cache_visible_len
                        .saturating_add(call.bounds.max_tokens);
                    if call.completion_output.is_none() {
                        request.logical_position = request
                            .logical_position
                            .saturating_add(call.bounds.max_tokens);
                    }
                    set_cache_lengths(&mut record, request.cache_visible_len);
                } else if samples_token {
                    let sampling_state = call.sampling_state.as_ref();
                    // A verification accepts its drafts while each is the
                    // token the model selects at its position, short of EOS
                    // and of a finish token; the token after them is
                    // sampled as a decode's.
                    let mut verified = Vec::new();
                    if work == CallKind::Forward(ForwardMode::Verify) {
                        for &draft in call.input_token_ids.get(1..).unwrap_or_default() {
                            let Some(output) = Self::sample(
                                vocab,
                                text_len,
                                fake_eos,
                                call,
                                request,
                                request.emitted + verified.len(),
                                sampling_state,
                            )?
                            else {
                                break;
                            };
                            let finishes = output.token == fake_eos
                                || sampling_state.is_some_and(|state| {
                                    state.finish_token_ids.binary_search(&output.token).is_ok()
                                });
                            if output.token != draft || finishes {
                                break;
                            }
                            request.fold_penalty_token(output.token);
                            verified.push(output);
                        }
                    }
                    let index = request.emitted + verified.len();
                    let Some(output) = Self::sample(
                        vocab,
                        text_len,
                        fake_eos,
                        call,
                        request,
                        index,
                        sampling_state,
                    )?
                    else {
                        record.status = CallStatus::Error;
                        record.error_code = Some(ErrorCode::InvalidDescriptor);
                        return Ok(record);
                    };
                    record.finish_flags.eos = output.token == fake_eos;
                    if let Some(state) = sampling_state {
                        record.finish_flags.stop =
                            state.finish_token_ids.binary_search(&output.token).is_ok()
                                && !record.finish_flags.eos;
                        record.finish_flags.length = state.force_finish;
                    }
                    // The token product's continuation predicate is false when
                    // the token is an admitted or per-call finish token or the
                    // call forces a finish.
                    let admitted_stops = request
                        .admission
                        .generation
                        .as_ref()
                        .map_or(&[][..], |und| und.finish_token_ids.as_slice());
                    let continuation = admitted_stops.binary_search(&output.token).is_err()
                        && sampling_state.is_none_or(|state| {
                            !state.force_finish
                                && state.finish_token_ids.binary_search(&output.token).is_err()
                        });
                    if let Some(token_product) = call.token_output.as_ref() {
                        request
                            .predicate_values
                            .insert(token_product.clone(), continuation);
                    }
                    let transition = sampling_state.is_some_and(|state| {
                        state
                            .transition_token_ids
                            .binary_search(&output.token)
                            .is_ok()
                    });
                    for completion in call.transition_output.iter() {
                        request
                            .predicate_values
                            .insert(completion.clone(), transition);
                    }
                    // Positions and cache advance by the query length: the whole
                    // bound for a prefill, one token for decode, and a
                    // verification's committed tokens.
                    let committed = verified.len() + 1;
                    if !visual_state {
                        let query_tokens = match work {
                            CallKind::Forward(ForwardMode::Prefill) => call.bounds.max_tokens,
                            CallKind::Forward(ForwardMode::Decode) => 1,
                            CallKind::Forward(ForwardMode::Verify) => committed as u32,
                            _ => unreachable!(),
                        };
                        request.logical_position =
                            request.logical_position.saturating_add(query_tokens);
                        request.cache_visible_len =
                            request.cache_visible_len.saturating_add(query_tokens);
                        set_cache_lengths(&mut record, request.cache_visible_len);
                    }
                    match work {
                        CallKind::Forward(ForwardMode::Prefill) => {
                            request.emitted = request.emitted.max(request.history + 1)
                        }
                        CallKind::Forward(ForwardMode::Decode)
                        | CallKind::Forward(ForwardMode::Verify) => {
                            request.emitted = request.emitted.saturating_add(committed)
                        }
                        _ => unreachable!(),
                    }
                    record.committed_tokens = verified
                        .iter()
                        .map(|verified| verified.token)
                        .chain([output.token])
                        .collect();
                    // Fold the generated token into the device-resident penalty
                    // base so the next call's penalties see it before this
                    // one is host-observed. A false-predicate no-op never reaches
                    // this branch, so a retracted point is never folded.
                    request.fold_penalty_token(output.token);
                    // Each committed token's own entry leads its candidates
                    // when the request asks for its generated tokens'
                    // logprobs.
                    if !output.top.is_empty() {
                        record.token_logprobs = verified
                            .iter()
                            .chain([&output])
                            .map(|sampled| {
                                sampled
                                    .top
                                    .iter()
                                    .map(|(token_id, logprob, rank)| TokenLogprob {
                                        token_id: *token_id,
                                        logprob: *logprob,
                                        rank: *rank,
                                    })
                                    .collect()
                            })
                            .collect();
                    }
                    // The drafter proposes after the committed tokens: the
                    // model's next tokens for its accepted count, then
                    // wrong ones.
                    if let Some(speculation) = speculation {
                        for draft in 0..call.draft.proposals as usize {
                            let Some(next) = Self::sample(
                                vocab,
                                text_len,
                                fake_eos,
                                call,
                                request,
                                request.emitted + draft,
                                sampling_state,
                            )?
                            else {
                                break;
                            };
                            record
                                .draft_token_ids
                                .push(if draft < speculation.accepted {
                                    next.token
                                } else if next.token == 1_000 {
                                    1_001
                                } else {
                                    1_000
                                });
                        }
                    }
                }
            }
            CallKind::Forward(ForwardMode::TokenDenoising) => {
                // A canvas pass reads the prefix cache without writing it and reports one
                // deterministic log-probability per readout candidate, or runs
                // one step of the request's generation canvas.
                if let Some(step) = call.canvas {
                    // A step queued behind the one that stopped its block is
                    // predicated on that step's false completion and never
                    // reaches this branch.
                    anyhow::ensure!(
                        step == request.canvas,
                        "canvas call {:?} runs {step:?}, but the request's canvas is at {:?}",
                        call.call_id,
                        request.canvas
                    );
                    let max_steps =
                        canvas_sampling
                            .map(|canvas| canvas.max_steps)
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "canvas call for an executor launched without canvas sampling"
                                )
                            })?;
                    let request_id = call.request_key.request_id;
                    if step.step < sim_canvas_stop_step(request_id, step.block, max_steps) {
                        request.canvas.step += 1;
                    } else {
                        // The stopped block holds the synthetic text from its
                        // first position on, and EOS past the text's end.
                        let length = call.bounds.max_tokens as usize;
                        let first = step.block as usize * length;
                        record.committed_tokens = (first..first + length)
                            .map(|index| {
                                if index < text_len {
                                    sim_text_token(request_id, index)
                                } else {
                                    fake_eos
                                }
                            })
                            .collect();
                        request.canvas = ipc::CanvasStep {
                            block: step.block + 1,
                            step: 0,
                        };
                        // The block stops here, so a step queued behind this
                        // one is a no-op.
                        if let Some(completion) = call.completion_output.as_ref() {
                            request.predicate_values.insert(completion.clone(), false);
                        }
                    }
                }
                if let Some(readout) = &call.readout {
                    record.candidate_logprobs = readout
                        .candidate_ids
                        .iter()
                        .map(|&token| {
                            if non_finite_candidate == Some(token) {
                                f32::NAN
                            } else {
                                sim_candidate_logprob(call.request_key.request_id, token)
                            }
                        })
                        .collect();
                }
            }
            CallKind::Media(MediaCall::MediaReading)
            | CallKind::Media(MediaCall::TextEncoding)
            | CallKind::Media(MediaCall::VisionEncoding)
            | CallKind::Media(MediaCall::LatentEncoding) => {}
            CallKind::Transfer(TransferMode::Tensor)
            | CallKind::Transfer(TransferMode::CacheExport)
            | CallKind::Transfer(TransferMode::CacheInstall) => {
                set_cache_lengths(&mut record, request.cache_visible_len);
            }
            CallKind::Media(MediaCall::LatentPreparation) => {
                // A restored trajectory opens at its stored step; any other
                // at step zero.
                let opened = call
                    .latent_restore
                    .as_ref()
                    .map_or(0, |restore| restore.steps);
                request.step_index = u16::try_from(opened).unwrap_or(u16::MAX);
                record.step_index = opened;
            }
            CallKind::Media(MediaCall::Denoising) => {
                // A denoising call runs one step.
                request.step_index = request.step_index.saturating_add(1);
                let total = request
                    .image()
                    .map(|image| image.steps)
                    .unwrap_or(DEFAULT_DENOISE_STEPS);
                record.step_index = u32::from(request.step_index);
                // Denoise completion surfaces as a length finish once the flow has
                // advanced through every scheduled step.
                record.finish_flags.length = request.step_index >= total;
            }
            CallKind::Media(
                MediaCall::VideoDecoding
                | MediaCall::AudioDecoding
                | MediaCall::VideoEncoding
                | MediaCall::AudioEncoding,
            ) => {}
            CallKind::Media(MediaCall::Muxing) => {
                // The muxer takes each round's encoded media units as they
                // arrive and assembles the artifact on the final call, the one
                // that carries none; the simulator publishes a synthetic
                // payload for that call only.
                if call.inputs.is_empty() {
                    let payload = format!(
                        "uniserve-sim-video request={}",
                        call.request_key.request_id.0
                    );
                    record.media_output =
                        Some(publish_media(
                        "uniserve-video-",
                        payload.as_bytes(),
                        media_overstatement,
                    )?);
                }
            }
            CallKind::Media(MediaCall::ImageDecoding) => {
                // An executor's image decoding releases the request's
                // latent and resets its denoising step to zero.
                request.step_index = 0;
                if let Some(image) = request.image().cloned() {
                    let (height, width) = if image.height > 0 && image.width > 0 {
                        (image.height, image.width)
                    } else {
                        DEFAULT_IMAGE_HW
                    };
                    // A decoding that writes the request's feedback image
                    // product publishes its values after the PNG, as a
                    // executor does.
                    let seed = image.seed.unwrap_or(call.request_key.request_id.0);
                    let mut media =
                        synthetic_png_b64(width, height, seed, request.image_index)?.into_bytes();
                    if call.image_output.is_some() {
                        let values =
                            synthetic_feedback_values(seed, request.image_index, height, width);
                        ipc::append_feedback(&mut media, &values, ipc::PixelRange::SignedUnit);
                    }
                    record.media_output =
                        Some(publish_media("uniserve-image-", &media, media_overstatement)?);
                }
            }
        }

        // Completion and transition outputs the call did not resolve above
        // resolve true.
        for completion in call
            .completion_output
            .iter()
            .chain(call.transition_output.iter())
        {
            request
                .predicate_values
                .entry(completion.clone())
                .or_insert(true);
        }

        Ok(record)
    }

    /// Builds the `Predicated` completion of a call whose predicate is false.
    ///
    /// The completion reports the request's unchanged progress.
    fn predicated_completion(call: &Call, request: &SimRequestState) -> CallOutput {
        CallOutput {
            token_logprobs: Vec::new(),
            prompt_logprobs: Vec::new(),
            request_key: call.request_key,
            call_id: call.call_id,
            status: CallStatus::Predicated,
            error_code: None,
            error_message: None,
            timing_counters: TimingCounters::default(),
            cache_visible_len: request.cache_visible_len,
            cache_computed_len: request.cache_visible_len,
            step_index: u32::from(request.step_index),
            committed_tokens: Vec::new(),
            candidate_logprobs: Vec::new(),
            finish_flags: FinishFlags::default(),
            media_output: None,
            cache_output: None,
            draft_token_ids: Vec::new(),
            latent_layouts: Vec::new(),
        }
    }

    /// Sets the pool's unresolved-batch capacity, clamped to at least one.
    pub fn set_queue_depth(&mut self, depth: u32) {
        self.placement.queue_depth = (depth as usize).max(1);
    }

    /// Names the executor groups this deployment runs, replacing the single
    /// `sim` group.
    ///
    /// Each group reports the same placement, ranks and pools: what differs
    /// between them in a test is which one has failed.
    pub fn name_groups(&mut self, names: &[&str]) {
        self.groups = names
            .iter()
            .map(|name| ExecutorGroupId((*name).to_owned()))
            .collect();
    }

    /// Places the pool's components, as a deployment places an executor
    /// group's, and gives the pool the ranks they span.
    pub fn place_components(&mut self, components: BTreeMap<String, ComponentConfig>) {
        self.placement.ranks = components
            .values()
            .flat_map(|component| component.ranks.iter().copied())
            .max()
            .map_or(1, |last| last + 1);
        self.placement.components = components;
    }

    /// Sets the number of synthetic tokens emitted before the configured EOS.
    pub fn set_text_len(&mut self, length: usize) {
        self.text_len = length;
    }

    /// Delivers a result only to a poll that waits for it.
    ///
    /// An instantaneous check then never observes a completion, so the
    /// scheduler sees every batch it submitted in flight until it blocks for
    /// one, and results arrive one per wait in submission order. This is the
    /// regime of a device slower than its scheduler, and it makes the order of
    /// submissions and completions a function of scheduling decisions alone.
    pub fn set_results_on_wait(&mut self, on: bool) {
        self.results_on_wait = on;
    }

    /// Replaces the simulator's cache groups.
    ///
    /// # Panics
    ///
    /// Panics when the simulated executor does not expose a prefix cache.
    #[expect(
        clippy::expect_used,
        reason = "a test setter whose missing prefix cache is documented under Panics"
    )]
    pub fn set_groups(&mut self, groups: Vec<ipc::CacheGroup>) {
        self.info
            .prefix_cache
            .as_mut()
            .expect("sim AR executor has a prefix cache")
            .groups = groups;
    }

    /// Sets the simulator's cache unit count, including the unit-zero sentinel.
    ///
    /// # Panics
    ///
    /// Panics when the simulated executor does not expose a prefix cache.
    #[expect(
        clippy::expect_used,
        reason = "a test setter whose missing prefix cache is documented under Panics"
    )]
    pub fn set_num_units(&mut self, count: u32) {
        self.info
            .prefix_cache
            .as_mut()
            .expect("sim AR executor has a prefix cache")
            .num_units = count;
    }

    /// Splits the simulated planes into `segments` of `units` each, all
    /// held, the first `inherited` of them after the first mapped from a
    /// predecessor world, as a successor's planes are.
    ///
    /// # Panics
    ///
    /// Panics when the simulated executor does not expose a prefix cache.
    #[expect(
        clippy::expect_used,
        reason = "a test setter whose missing prefix cache is documented under Panics"
    )]
    pub fn set_segments(&mut self, segments: u32, units: u32, inherited: u32) {
        let cache = self
            .info
            .prefix_cache
            .as_mut()
            .expect("sim AR executor has a prefix cache");
        cache.segments = (0..segments)
            .map(|ordinal| ipc::CacheSegment {
                world: if (1..=inherited).contains(&ordinal) {
                    1
                } else {
                    2
                },
                ordinal,
                units,
            })
            .collect();
        cache.held_segments = segments;
        cache.inherited_start = if inherited > 0 { units } else { 0 };
        cache.inherited_units = inherited * units;
    }

    /// Reports a state group after the K/V groups: checkpoints of a layer
    /// kept per request slot, a page of `units_per_page` units at every
    /// boundary of the first group's block size.
    ///
    /// # Panics
    ///
    /// Panics when the simulated executor does not expose a prefix cache.
    #[expect(
        clippy::expect_used,
        reason = "a test setter whose missing prefix cache is documented under Panics"
    )]
    pub fn add_state_group(&mut self, units_per_page: u32) {
        let cache = self
            .info
            .prefix_cache
            .as_mut()
            .expect("sim AR executor has a prefix cache");
        let first = cache.groups[0].clone();
        let layer = cache
            .groups
            .iter()
            .flat_map(|group| group.layer_ids.iter().copied())
            .max()
            .map_or(0, |layer| layer + 1);
        cache.groups.push(ipc::CacheGroup {
            kind: ipc::CacheGroupKind::State,
            block_size: first.block_size,
            units_per_page,
            layer_ids: vec![layer],
            num_kv_heads: 1,
            total_kv_heads: 1,
            kv_head_offset: 0,
            head_dim: 1,
        });
    }

    /// Sets every cache group's block size in tokens.
    ///
    /// # Panics
    ///
    /// Panics when the simulated executor does not expose a prefix cache.
    #[expect(
        clippy::expect_used,
        reason = "a test setter whose missing prefix cache is documented under Panics"
    )]
    pub fn set_block_size(&mut self, tokens: u32) {
        for group in &mut self
            .info
            .prefix_cache
            .as_mut()
            .expect("sim AR executor has a prefix cache")
            .groups
        {
            group.block_size = tokens;
        }
    }

    /// Speculates as `speculation` states: the simulated drafter proposes
    /// for the calls that ask for drafts, verifications accept the drafts
    /// that are the model's tokens, and every step reports its executor time.
    pub fn set_speculation(&mut self, speculation: SimSpeculation) {
        self.speculation = Some(speculation);
    }

    /// Reports a draft group after the other groups: the draft cache, whose
    /// page holds `span` of the first group's pages' tokens in one unit.
    ///
    /// # Panics
    ///
    /// Panics when the simulated executor does not expose a prefix cache.
    #[expect(
        clippy::expect_used,
        reason = "a test setter whose missing prefix cache is documented under Panics"
    )]
    pub fn add_draft_group(&mut self, span: u32) {
        let cache = self
            .info
            .prefix_cache
            .as_mut()
            .expect("sim AR executor has a prefix cache");
        let first = cache.groups[0].clone();
        let layer = cache
            .groups
            .iter()
            .flat_map(|group| group.layer_ids.iter().copied())
            .max()
            .map_or(0, |layer| layer + 1);
        cache.groups.push(ipc::CacheGroup {
            kind: ipc::CacheGroupKind::Draft,
            block_size: first.block_size * span,
            units_per_page: 1,
            layer_ids: vec![layer],
            ..first
        });
    }

    /// Launches the simulated executor with block-diffusion sampling: a canvas
    /// step stops its block no later than the sampling's step limit.
    pub fn set_canvas_sampling(&mut self, sampling: ipc::CanvasSampling) {
        self.canvas_sampling = Some(sampling);
    }

    /// Reports NaN as the log-probability of readout candidate `token`,
    /// wherever a readout reads it.
    pub fn set_non_finite_candidate(&mut self, token: u32) {
        self.non_finite_candidate = Some(token);
    }

    /// Holds every batch that skips a predicated call until the returned
    /// sender releases it, as an executor whose skipped step is still running
    /// holds its batch's result: each message releases one held batch, and
    /// dropping the sender releases the rest.
    pub fn hold_predicated_batches(&mut self) -> Sender<()> {
        let (release, held) = crossbeam_channel::unbounded();
        self.predicated_release = Some(held);
        release
    }

    /// Paces the simulator: each batch executes only once the returned
    /// [`Pace`] permits it, and every batch runs freely once it is dropped.
    /// A test steps a request through its calls with it, to act at a chosen
    /// point of the request's progress.
    pub fn pace(&mut self) -> Pace {
        let (permits, paced) = crossbeam_channel::unbounded();
        self.pace = Some(paced);
        Pace(permits)
    }

    /// States every media artifact the simulator publishes as declaring
    /// `extra_bytes` more than the object stores.
    ///
    /// A rank's report states its payload's extent, and the engine claims the
    /// object by that statement: a declaration that outruns the storage is
    /// what the claim must refuse (`ipc::SharedMediaError::Short`), rather
    /// than map fewer bytes than the report promised.
    pub fn overstate_published_media(&mut self, extra_bytes: u64) {
        self.media_overstatement = extra_bytes;
    }

    /// Refuses request `id` for `reason` at each of its media calls.
    pub fn refuse_request(&mut self, id: RequestId, reason: &str) {
        self.refused.insert(id, reason.to_owned());
    }

    /// Returns mutable access to the simulator's advertised capabilities.
    pub fn mut_info_for_test(&mut self) -> &mut MainProcessInfo {
        &mut self.info
    }

    /// States the host latent pool of each of the simulated pool's ranks,
    /// with the rank's host, as video denoising ranks report them. The
    /// simulator copies no bytes: an offload's batch completes, and a
    /// restored latent preparation commits the trajectory at its stored
    /// step, from which the request's denoising steps continue.
    pub fn set_latent_pools(&mut self, pools: Vec<(String, Option<ipc::LatentPoolInfo>)>) {
        self.latent_pools = pools;
    }

    /// States how each simulated denoising rank's request slot pages hold a
    /// request's samples, which every video latent preparation reports, as
    /// the ranks' executors declare it for every request.
    pub fn set_latent_layouts(&mut self, layouts: Vec<ipc::RankLatentLayout>) {
        self.latent_layouts = layouts;
    }

    /// States the host encoder pool of each of the simulated pool's ranks,
    /// with the rank's host, as ranks executing video encoder calls report
    /// them. The simulator copies no bytes: an encoder call whose output it
    /// reads from the pool completes as one that computes it.
    pub fn set_encoder_pools(&mut self, pools: crate::executor::RankEncoderPools) {
        self.encoder_pools = pools;
    }

    /// The log of what each request's calls write into its context, which
    /// the simulator keeps filling once it runs.
    pub fn context_log(&self) -> ContextLog {
        self.context.clone()
    }
}

/// The values of the decoded image product of request `request_id`'s image
/// `image_index`: BF16 bytes `[3, height, width]` the simulator derives from
/// the two, so an image restarted from noise decodes to the same values.
/// The values a generated image's feedback product carries, derived from the
/// seed the request draws its latent with: one seed yields one image, and the
/// same seed in another request yields the same one.
fn synthetic_feedback_values(seed: u64, image_index: u32, height: u32, width: u32) -> Vec<u8> {
    let bytes = ipc::ImagePixels::value_bytes(height, width) as usize;
    let mut state = (seed << 32) ^ u64::from(image_index) ^ 0x9e37_79b9_7f4a_7c15;
    (0..bytes)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        })
        .collect()
}

/// A digest of image values, which the context log names an image by.
fn digest(values: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    values.hash(&mut hasher);
    hasher.finish()
}

/// What `call`, which completed with `record`, wrote into `request`'s
/// context, as an executor's KV cache holds it, and the provenance of the image
/// products it made.
fn context_writes(
    call: &Call,
    record: &CallOutput,
    request: &mut SimRequestState,
) -> Vec<ContextWrite> {
    let position = call.coordinates.logical_position;
    let mut writes = Vec::new();
    match call.kind {
        CallKind::Forward(ForwardMode::Prefill | ForwardMode::Decode) => {
            let feature = call
                .latent_feature_input
                .as_ref()
                .or(call.vision_inputs.first().map(|input| &input.feature));
            let feedback = call.completion_output.is_some()
                && (call.latent_feature_input.is_some() || !call.vision_inputs.is_empty());
            if feedback {
                let (image, draw) = feature
                    .and_then(|feature| request.provenance.get(feature).copied())
                    .unwrap_or((0, None));
                writes.push(ContextWrite::Image {
                    position,
                    image,
                    draw,
                });
            } else if call.latent_feature_input.is_some() {
                writes.push(ContextWrite::InputBlock { position });
            } else {
                // Prompt tokens with the vision blocks between them, a
                // decode's input or a cache close: an input left empty
                // relays the predecessor's sampled token.
                let tokens: Vec<u32> = if call.input_token_ids.is_empty()
                    && call.kind == CallKind::Forward(ForwardMode::Decode)
                {
                    request.last_token.into_iter().collect()
                } else {
                    call.input_token_ids.clone()
                };
                let mut blocks = call.vision_inputs.iter().peekable();
                for (index, token) in tokens.iter().enumerate() {
                    while blocks
                        .peek()
                        .is_some_and(|block| block.offset as usize == index)
                    {
                        blocks.next();
                        writes.push(ContextWrite::InputBlock {
                            position: position + index as u32,
                        });
                    }
                    writes.push(ContextWrite::Token {
                        position: position + index as u32,
                        token: *token,
                    });
                }
                for _ in blocks {
                    writes.push(ContextWrite::InputBlock {
                        position: position + tokens.len() as u32,
                    });
                }
            }
        }
        CallKind::Media(MediaCall::LatentPreparation) => {
            if let Some(image_index) = call.image_index {
                request.image_index = image_index;
            }
        }
        CallKind::Media(MediaCall::ImageDecoding) => {
            if let (Some(output), Some(image)) = (&call.image_output, request.image()) {
                let (height, width) = if image.height > 0 && image.width > 0 {
                    (image.height, image.width)
                } else {
                    DEFAULT_IMAGE_HW
                };
                let seed = request
                    .image()
                    .and_then(|image| image.seed)
                    .unwrap_or(call.request_key.request_id.0);
                let values = synthetic_feedback_values(seed, request.image_index, height, width);
                request
                    .provenance
                    .insert(output.clone(), (digest(&values), None));
            }
        }
        CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding) => {
            let source = match (&call.image_input, &call.carried_image, &call.input_image) {
                (Some(product), _, _) => request.provenance.get(product).map(|(image, _)| *image),
                (_, Some(carried), _) => Some(digest(&carried.values)),
                (_, _, Some(payload)) => Some(digest(payload.as_bytes())),
                _ => None,
            };
            if let (Some(output), Some(image)) = (&call.encoder_output, source) {
                request
                    .provenance
                    .insert(output.clone(), (image, call.latent_draw));
            }
        }
        _ => {}
    }
    if let Some(&token) = record.committed_tokens.last() {
        request.last_token = Some(token);
    }
    writes
}

/// Synthetic text token `index` of request `request_id`: the natural token
/// the simulator samples at that generated position, in `1_000..6_000`.
pub fn sim_text_token(request_id: RequestId, index: usize) -> u32 {
    1_000 + ((request_id.0 as u32 * 7 + index as u32) % 5_000)
}

/// Zero-based step at which the simulator stops block `block` of request
/// `request_id`: one of its first three steps, within `max_steps`.
pub fn sim_canvas_stop_step(request_id: RequestId, block: u32, max_steps: u32) -> u32 {
    ((request_id.0 as u32).wrapping_add(block) % 3).min(max_steps.saturating_sub(1))
}

/// Sets every logical cache frontier to the visible token position.
/// Deterministic natural-log probability the simulator reports for readout
/// candidate `token` of request `request_id`: a value in `[-7, -1]`.
pub fn sim_candidate_logprob(request_id: RequestId, token: u32) -> f32 {
    -1.0 - ((request_id.0.wrapping_add(u64::from(token))) % 7) as f32
}

fn set_cache_lengths(lengths: &mut CallOutput, visible: u32) {
    lengths.cache_visible_len = visible;
    lengths.cache_computed_len = visible;
}

/// Publishes bytes as an immutable POSIX shared-memory artifact.
///
/// The object is left in place under its generated name; the engine claims it
/// by that name and unlinks it when it opens the result. `extra_bytes` is
/// declared in the report without being written: the disagreement between a
/// statement of extent and its storage that the claim refuses.
fn publish_media(
    prefix: &str,
    bytes: &[u8],
    extra_bytes: u64,
) -> anyhow::Result<ipc::MediaOutput> {
    use std::io::Write;

    let mut storage = tempfile::Builder::new()
        .prefix(prefix)
        .tempfile_in("/dev/shm")?;
    storage.write_all(bytes)?;
    let (_file, path) = storage.keep()?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("shared-memory artifact has no printable name"))?
        .to_owned();

    Ok(ipc::MediaOutput {
        handle: ipc::ArtifactHandle::PosixShm { name },
        bytes: bytes.len() as u64 + extra_bytes,
    })
}

/// Encodes a deterministic RGB gradient as a base64 PNG artifact.
fn synthetic_png_b64(
    width: u32,
    height: u32,
    seed: u64,
    image_index: u32,
) -> anyhow::Result<String> {
    // The simulated denoiser predicts zero velocity, so an image is its initial
    // latent decoded: the seed and the image's index decide its pixels, as they
    // decide its feedback values.
    let latent =
        digest(&[seed.to_le_bytes(), u64::from(image_index).to_le_bytes()].concat()).to_le_bytes();
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        let row_bytes = width as usize * 3;
        let mut image = vec![0; row_bytes * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let index = y * row_bytes + x * 3;
                image[index] = ((x * 255) / width.max(1) as usize) as u8;
                image[index + 1] = ((y * 255) / height.max(1) as usize) as u8;
                image[index + 2] = latent[x % latent.len()];
            }
        }
        writer.write_image_data(&image)?;
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

impl Default for SimEngine {
    /// Equivalent to [`SimEngine::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl SimEngine {
    /// Returns the executor metadata.
    fn info(&self) -> &MainProcessInfo {
        &self.info
    }

    /// Executes a batch's calls in order against deterministic model state.
    ///
    /// Admissions register first, then `Free` commands release predicate
    /// values, then calls execute, and `Finish` commands drop their requests
    /// last, so calls in the same batch still execute against a finishing
    /// request.
    ///
    /// Errors on a readmission with a different descriptor other than a hand-off's, a call for a
    /// request that is not admitted or
    /// whose key differs from the admission, a predicate naming an unresolved
    /// product, or any `execute_call` error. An error abandons the rest of
    /// the batch and its `Finish` commands; admissions, `Free` commands,
    /// earlier calls, and whatever the failing call applied before the error
    /// keep their effects on request state.
    fn execute(&mut self, batch: ExecutionBatch) -> anyhow::Result<BatchOutput> {
        // A paced simulator waits for its permit; a dropped sender frees it,
        // and so does its executor's shutdown, so that a batch waiting for a
        // permit cannot hold the thread the executor joins on.
        if let Some(pace) = self.pace.clone() {
            loop {
                if self.stop.load(std::sync::atomic::Ordering::Acquire) {
                    anyhow::bail!("the simulator stopped while a batch waited for its permit");
                }
                match pace.recv_timeout(std::time::Duration::from_millis(20)) {
                    Ok(()) => break,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        self.pace = None;
                        break;
                    }
                }
            }
        }
        // `SimExecutor::submit` validated the batch.
        let batch_id = batch.id;
        let closed = batch
            .commands
            .iter()
            .filter_map(|control| match control {
                ipc::BatchCommand::Finish { request_key, .. } => Some(*request_key),
                _ => None,
            })
            .collect::<Vec<_>>();

        let admissions = batch.admissions().cloned().collect::<Vec<_>>();
        for admission in admissions {
            // A hand-off's install admits the request on its decode group, which
            // starts from the install's descriptor. The prefill group serves it
            // no more, so one state stands for the request on either.
            let handed_off = batch.requests.iter().any(|(call, _)| {
                call.request_key == admission.request_key
                    && call.kind == CallKind::Transfer(TransferMode::CacheInstall)
            });
            match self.requests.get_mut(&admission.request_key.request_id) {
                Some(request) if handed_off => {
                    let last_token = request.last_token;
                    *request = SimRequestState::new(admission);
                    request.last_token = last_token;
                }
                Some(request) => anyhow::ensure!(
                    request.admission == admission,
                    "request {} was readmitted with a different descriptor",
                    admission.request_key.request_id.0
                ),
                None => {
                    self.requests.insert(
                        admission.request_key.request_id,
                        SimRequestState::new(admission),
                    );
                }
            }
        }

        for command in &batch.commands {
            if let ipc::BatchCommand::Free { buffer } = command
                && let Some(request) = self.requests.get_mut(&buffer.owner.request_id)
            {
                request
                    .predicate_values
                    .retain(|product, _| product.buffer_id() != *buffer);
            }
        }

        let vocab = self.vocab;
        let text_len = self.text_len;
        let fake_eos = self.fake_eos;
        let non_finite_candidate = self.non_finite_candidate;
        let canvas_sampling = self.canvas_sampling;
        let speculation = self.speculation;
        let worker_exec_us = speculation
            .map(|speculation| speculation.step_us(batch.requests.iter().map(|(call, _)| call)));
        let refused = &self.refused;
        let latent_layouts = &self.latent_layouts;
        let mut completions = Vec::with_capacity(batch.requests.len());
        for (call, _) in batch.requests {
            let request = self
                .requests
                .get_mut(&call.request_key.request_id)
                .ok_or_else(|| {
                    anyhow::anyhow!("request {} has no admission", call.request_key.request_id.0)
                })?;
            anyhow::ensure!(
                call.request_key == request.admission.request_key,
                "call identity {:?} does not match its admitted request",
                call.request_key
            );
            let predicate_value = call
                .predicate
                .as_ref()
                .as_ref()
                .map(|predicate| {
                    request
                        .predicate_values
                        .get(predicate)
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "call {:?} predicate names an unresolved product",
                                call.call_id
                            )
                        })
                })
                .transpose()?;

            // As on an executor, a chained call releases the products of
            // the call it follows, and a predicate is released once its
            // consumer has read it. The predicate value is read above, before
            // these releases, because it may be a product of the predecessor.
            if request.chained(&call) {
                let predecessor = request.state_call_id;
                request
                    .predicate_values
                    .retain(|product, _| product.producer_call_id != predecessor);
            }
            if let Some(predicate) = call.predicate.as_ref() {
                request.predicate_values.remove(predicate);
            }
            if let Some(predicate_value) = predicate_value
                && !predicate_value
            {
                for output in call.tensor_outputs() {
                    request.predicate_values.insert(output.clone(), false);
                }
                let completion = Self::predicated_completion(&call, request);
                completions.push(completion);
                continue;
            }

            let mut completion = Self::execute_call(
                vocab,
                text_len,
                fake_eos,
                non_finite_candidate,
                canvas_sampling,
                speculation,
                self.media_overstatement,
                &call,
                request,
            )?;
            if let (CallKind::Media(_), Some(reason)) =
                (call.kind, refused.get(&call.request_key.request_id))
            {
                completion.status = CallStatus::Error;
                completion.media_output = None;
                completion.error_code = Some(ErrorCode::InputError);
                completion.error_message = Some(reason.clone());
            }
            // A video latent preparation reports how each denoising rank
            // holds the request's samples, as every rank's completion does
            // once the engine joined them.
            if completion.status == CallStatus::Ok
                && call.kind == CallKind::Media(MediaCall::LatentPreparation)
            {
                completion.latent_layouts = latent_layouts.clone();
            }
            if completion.status == CallStatus::Ok && call.advances_state() {
                request.state_call_id = call.call_id;
            }
            if completion.status == CallStatus::Ok {
                self.context.record(
                    call.request_key,
                    context_writes(&call, &completion, request),
                );
            }
            completions.push(completion);
        }

        if let Some(release) = &self.predicated_release
            && completions
                .iter()
                .any(|completion| completion.status == CallStatus::Predicated)
        {
            // A disconnected sender releases every later batch.
            let _ = release.recv();
        }

        let report = BatchOutput {
            batch_id,
            completions,
            products: Vec::new(),

            worker_exec_us,
            forward_stats: None,
        };
        // A `Finish` for a stale request epoch leaves a readmitted request
        // under the same id in place.
        for request_key in closed {
            if self
                .requests
                .get(&request_key.request_id)
                .is_some_and(|request| request.admission.request_key == request_key)
            {
                self.drop_request(request_key.request_id)?;
            }
        }
        Ok(report)
    }

    /// Removes all simulated state for a request.
    fn drop_request(&mut self, request_id: RequestId) -> anyhow::Result<()> {
        self.requests.remove(&request_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipc::{
        Bounds, CallId, DType, ForwardBatch, GenerationParams, RequestKey, ShapeBound, TensorRef,
    };

    /// The pool states the longest sequence it serves, and `SimExecutor`
    /// hands that number to the mechanism: a batch is shaped to
    /// a bound the pool declared, and a pool that declares none hands over a
    /// zero the mechanism refuses rather than a guessed bound.
    #[test]
    fn a_simulated_pool_states_the_model_length_it_was_launched_for() {
        let launched = SimEngine::new().max_model_len();
        assert!(launched > 0, "a launched pool states its model's length");
        let executor = SimExecutor::new(SimEngine::new());
        assert_eq!(executor.info().max_model_len, launched);

        let mut silent = SimEngine::new();
        silent.set_max_model_len(0);
        let executor = SimExecutor::new(silent);
        assert_eq!(
            executor.info().max_model_len,
            0,
            "a world that states none states none: the mechanism, not the simulator, refuses it"
        );

        let mut longer = SimEngine::new();
        longer.set_max_model_len(65_536);
        assert_eq!(SimExecutor::new(longer).info().max_model_len, 65_536);
    }

    fn request_key() -> RequestKey {
        RequestKey::new(0, RequestId(9), 1)
    }

    fn admission() -> NewRequest {
        NewRequest::new(
            request_key(),
            1,
            Some(GenerationParams {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                initial_position: 0,
                generated_token_ids: Vec::new(),
                adapter: 0,
            }),
            None,
            0,
        )
        .expect("admission")
    }

    fn token_output(call_id: CallId) -> TensorRef {
        TensorRef {
            request_key: request_key(),
            producer_call_id: call_id,
            output_index: 0,
            generation: 1,
            dtype: DType::I64,
            shape_bound: ShapeBound::default(),
        }
    }

    /// Builds a batch that admits `request_key()` and runs one two-token
    /// prefill with a token output as call `(batch_id, request_index)`.
    fn batch(batch_id: u64, request_index: u32) -> ExecutionBatch {
        let request_key = request_key();
        let admission = admission();
        let call = Call {
            consumer_slots: Vec::new(),
            coordinates: ipc::CallCoordinates::default(),
            token_input: None,

            token_output: Some(token_output(CallId::new(batch_id, request_index))),
            vision_inputs: Vec::new(),
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,

            input_image: None,
            draft: Default::default(),
            cache_input: None,
            cache_output: None,
            input_token_ids: Vec::new(),
            readout: None,
            canvas: None,
            sampling_state: None,
            request_key,
            call_id: CallId::new(batch_id, request_index),
            component: "model".into(),
            kind: CallKind::Forward(ForwardMode::Prefill),
            bounds: Bounds {
                max_tokens: 2,
                ..Bounds::default()
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            predicate: None,
            image_index: None,
            latent_restore: None,
            host_output: None,
            latent_draw: None,
            carried_image: None,
        };
        ExecutionBatch::new(
            batch_id,
            vec![(
                call,
                crate::executor::RequestPlacement {
                    worker: ExecutorGroupId("sim".into()),
                    request_pool_idx: None,
                    block_tables: Vec::new(),
                    new_cache_units: Vec::new(),
                    forward: ForwardBatch::default(),
                    latent: None,
                    decode: None,
                    buffers: Vec::new(),
                    readers: Default::default(),
                    page_loads: Vec::new(),
                },
            )],
            vec![ipc::BatchCommand::Start {
                request: Box::new(admission),
            }],
            Vec::new(),
        )
    }

    #[test]
    fn same_request_calls_preserve_coordinates_and_sequential_tokens() {
        let mut executor = SimExecutor::new(SimEngine::new());
        let mut selected = batch(1, 3);
        let mut successor = batch(1, 4).requests.remove(0);
        // The selected prompt covers two positions, so its successor in the
        // same batch enters where it left off.
        successor.0.coordinates = ipc::CallCoordinates {
            logical_position: 2,
            cache_visible_len: 2,
            cache_computed_len: 2,
        };
        selected.requests.push(successor);

        executor.submit(selected).expect("submit logical batch");
        let report = executor
            .poll(Duration::from_secs(5))
            .expect("poll")
            .expect("result");

        assert!(report.done);
        let outputs = report
            .results
            .iter()
            .map(|result| &result.output)
            .collect::<Vec<_>>();
        assert_eq!(
            outputs
                .iter()
                .map(|output| output.call_id)
                .collect::<Vec<_>>(),
            vec![CallId::new(1, 3), CallId::new(1, 4)]
        );
        // Greedy synthetic tokens are 1000 + request_id * 7 + accepted index.
        assert_eq!(outputs[0].committed_tokens, vec![1063]);
        assert_eq!(outputs[1].committed_tokens, vec![1064]);
        assert_eq!(outputs[1].cache_visible_len, 4);
        executor.close().expect("close simulator");
    }

    #[test]
    fn default_limits_admit_public_image_geometry() {
        let engine = SimEngine::new();
        let info = engine.info();
        let latent_units = (2_048_u64 / 16) * (1_152_u64 / 16);
        assert!(latent_units <= info.latent_capacity_units());
    }
}
