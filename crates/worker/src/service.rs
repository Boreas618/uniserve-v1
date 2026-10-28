//! Rank-channel admission, result delivery, stall reports, and orderly
//! worker shutdown.
//!
//! The control thread serves the channel here and never waits for a job. Each
//! wait for a request or a completion ends, at the latest, when the job in
//! flight longest reaches the rank's stall bound: a job the builder thread has
//! stayed inside that long, such as a driver call that does not return, is
//! reported to the engine ([`WorkerResponse::Stalled`]) with the batch the job
//! belongs to, and the channel goes on being served.

use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ipc::{Channel, IpcError, RequestKind, WorkerInfo, WorkerRequest, WorkerResponse};

use crate::error::WorkerError;
use crate::executor::backend::WorkerBatch;
use crate::memory::retained::Retained;
use crate::{Batch, Error, Executor, Submission};

const EXPERT_POLL: Duration = Duration::from_micros(200);
// Keep a rank with open requests available for its engine's next batch before
// joining an empty expert step. The engine needs time to consume the result
// and schedule its next batch even when it still has admitted requests.
const OWN_STEP_WAIT: Duration = Duration::from_millis(8);

/// How a serving run ended other than by a failure of the rank's own.
pub(crate) enum Ending {
    /// The engine closed the rank, or a fatal response ended admission.
    Closed,
    /// The engine's channel failed (the error says how) while nothing was in
    /// flight: no admitted batch was unfinished and no job given to the builder
    /// thread was without its completion, so nothing the rank issued can still
    /// run on a device.
    Orphaned(WorkerError),
    /// The engine's channel failed (the error says how) while work was in
    /// flight. Nothing can deliver its results, and it may never complete:
    /// it may wait on a peer that stopped or abandoned its own work, and
    /// the builder thread may wait in it for good.
    Abandoned(WorkerError),
}

enum PendingResponse {
    Ready(Box<WorkerResponse>),
    Batch {
        message_id: Option<u64>,
        submission: Arc<Submission>,
    },
}

/// Borrows a worker's executor and channel for one synchronous serving run.
/// All channel waits run on the caller's thread; completion wakes interrupt
/// them without advancing execution on a callback thread.
/// What a parked standby worker released, which its promotion restores;
/// nothing for a standby that keeps its memory resident.
#[derive(Default)]
pub(crate) struct Suspension {
    /// The graph pools and scratch whose contents it moved to host memory.
    pub(crate) pools: Vec<Arc<crate::cuda_graph::GraphPool>>,
    /// What the executor released of the expert transports' workspaces.
    pub(crate) experts: ParkedExperts,
    /// The torch world whose NCCL communicators it suspended, the stream
    /// groups' included.
    pub(crate) world: Option<crate::ffi::Handle>,
}

/// What a parked standby's executor released of its expert union's
/// transport workspaces (`park_experts`).
#[derive(Default, Clone, Copy)]
pub(crate) struct ParkedExperts {
    /// Workspaces that keep their addresses were suspended: the promotion
    /// maps them back (`promote_experts`).
    pub(crate) suspended: bool,
    /// Workspaces that cannot keep their addresses were freed: the promotion
    /// rebuilds them (`rebuild_experts`) and captures the computations' graphs
    /// that address them again.
    pub(crate) released: bool,
}

/// What the engine's built-in worker hooks (`uniserve.*`) reach: the
/// Storage Server connection the cache planes' segments are retained
/// through, and the graph pools a sleeping worker suspends.
pub(crate) struct Maintenance {
    /// `None` for a rank whose planes this process allocated for itself
    /// (`LocalMemory`).
    pub(crate) retained: Option<Arc<Retained>>,
    /// What an in-place weight update rewrites; `None` for a rank that
    /// loaded no model.
    pub(crate) weights: Option<crate::worker::update::WeightUpdate>,
    pub(crate) pools: Vec<Arc<crate::cuda_graph::GraphPool>>,
    /// Those of `pools` that hold captured graphs alone: an eager rank gives
    /// them back, and keeps the streams' scratch its forwards use.
    pub(crate) graph_pools: Vec<Arc<crate::cuda_graph::GraphPool>>,
    /// Whether the worker sleeps: its graph memory suspended and its cache
    /// given back.
    pub(crate) asleep: bool,
    /// CUPTI's always-on counters of the rank, kept while it serves; `None`
    /// without a CUPTI subscription.
    pub(crate) _counters: Option<crate::profiling::counters::Counters>,
    /// The rank, as a profiling session selects ranks.
    pub(crate) rank: u32,
    /// The KV cache of the predecessor world's worker of the rank's scope,
    /// whose flags and scales `uniserve.adopt_cache` copies at the world's
    /// promotion; `None` for a world that succeeds none.
    pub(crate) predecessor: Option<ipc::PredecessorCache>,
    /// The token capacity the deployment configured for the KV cache
    /// (`kv_cache_config.max_tokens`), which bounds a successor world's
    /// pool as it bounds this one's; `None` for a pool sized by memory.
    pub(crate) max_total_tokens: Option<usize>,
    /// The request slots the rank was configured for, whose state
    /// checkpoints a token capacity also holds (`CachePlan::token_units`).
    pub(crate) request_slots: usize,
    /// The rank's CUDA devices, whose contexts the rank probes while jobs
    /// are in flight (`Service::watch_jobs`).
    pub(crate) devices: Vec<cuda::DeviceId>,
    /// Whether a rank of the loaded model captures graphs under a policy
    /// that captures: a world succeeding this one that configures graphs
    /// takes a graph share for its warmup's captures only then.
    pub(crate) capturable: bool,
    /// What the warm rank books in each device's ledger: a promoted standby
    /// books it, a sleep books its graph pools released and a wake books
    /// them again.
    pub(crate) bookings: Vec<crate::capacity::device::DeviceBooking>,
    /// What mapping the rank's host page pools needs, which
    /// `uniserve.host_cache` takes; `None` for a rank that copies no page
    /// through the host KV cache, and once mapped.
    pub(crate) host_cache: Option<crate::memory::prefix_cache::host::HostCacheSetup>,
}

/// A standby worker parked until its promotion: its retained storages,
/// read-only, and what it released while parked.
pub(crate) struct Parked {
    pub(crate) retained: Arc<Retained>,
    pub(crate) suspension: Suspension,
}

impl Parked {
    /// Restore what the standby released, at its addresses, then take write
    /// access to the retained storages: the captured graphs run again.
    /// Restoring the expert transports is collective over the union: the
    /// engine asks every rank of the standby world to promote before it
    /// awaits any. Transports that freed their workspaces are rebuilt and
    /// the executor's graphs captured again, as warmup captured them:
    /// collectively, on the scratch units, with write access; the executor
    /// then refuses further captures.
    fn promote(self, executor: &mut Executor) -> std::result::Result<(), WorkerError> {
        let Parked {
            retained,
            suspension:
                Suspension {
                    pools,
                    experts,
                    world,
                },
        } = self;
        let backend = executor.backend();
        for pool in &pools {
            pool.restore()?;
        }
        if !pools.is_empty() {
            tracing::info!("the promoted standby worker restored its graph memory");
        }
        if let Some(world) = &world {
            crate::ffi::Interpreter::import("uniserve.distributed.world.resume")?
                .invoke(&[world.view()])?;
            tracing::info!("the promoted standby worker resumed its communicators");
        }
        let python = backend.executor().cloned();
        if experts.suspended
            && let Some(python) = &python
        {
            python.promote_experts()?;
            tracing::info!("the promoted standby worker restored its expert transports");
        }
        retained.promote()?;
        let Some(python) = python.filter(|_| experts.released) else {
            return Ok(());
        };
        python.rebuild_experts()?;
        crate::worker::warm_computations(executor.backend_mut())?;
        python.seal()?;
        tracing::info!("the promoted standby worker captured its expert graphs again");
        Ok(())
    }
}

pub(crate) struct Service<'a> {
    executor: &'a mut Executor,
    channel: &'a mut Channel,
    info: WorkerInfo,
    experts: bool,
    pending: VecDeque<PendingResponse>,
    closing: bool,
    shutdown: Option<WorkerResponse>,
    last_result: Instant,
    /// How long one job may run before it is reported.
    stall_bound: Duration,
    /// A stalled job was reported. The engine fails the rank on the report, so
    /// it is made once.
    stalled: bool,
    /// When the devices' contexts were last probed for an error that left
    /// them unusable.
    probed: Instant,
    /// The engine's channel failed: it closed, or a send or receive on it
    /// did.
    channel_failed: Cell<bool>,
    /// The Storage Server connection of a parked standby worker, which
    /// promotion takes write access through; `None` once the rank serves.
    standby: Option<Parked>,
    maintenance: Maintenance,
}

impl<'a> Service<'a> {
    pub(crate) fn new(
        executor: &'a mut Executor,
        channel: &'a mut Channel,
        info: WorkerInfo,
        experts: bool,
        stall_bound: Duration,
        standby: Option<Parked>,
        maintenance: Maintenance,
    ) -> Self {
        executor.set_completion_wake(Some(channel.wake()));
        Self {
            executor,
            channel,
            info,
            experts,
            pending: VecDeque::new(),
            closing: false,
            shutdown: None,
            last_result: Instant::now(),
            stall_bound,
            stalled: false,
            probed: Instant::now(),
            channel_failed: Cell::new(false),
            standby,
            maintenance,
        }
    }

    /// Serve until the engine closes the rank or a fatal response ends
    /// admission, which is [`Ending::Closed`].
    ///
    /// Once the engine's channel fails nothing more can be admitted or
    /// delivered: the run ends [`Ending::Orphaned`] when no admitted batch
    /// is unfinished and the jobs in flight finish before the stall bound,
    /// and [`Ending::Abandoned`] otherwise.
    pub(crate) fn run(&mut self) -> Result<Ending, WorkerError> {
        let error = match self.serve() {
            Ok(()) => return Ok(Ending::Closed),
            Err(error) if self.channel_failed.get() => error,
            Err(error) => return Err(error),
        };
        if self.settled() {
            tracing::warn!(
                "the engine's channel failed with nothing in flight; the rank closes without its peers"
            );
            return Ok(Ending::Orphaned(error));
        }
        tracing::warn!(
            unfinished_batches = self.executor.has_work(),
            jobs_in_flight = self.executor.backend().oldest_job().is_some(),
            "the engine's channel failed with work in flight, which the rank abandons"
        );
        Ok(Ending::Abandoned(error))
    }

    /// Whether nothing the rank issued can still run on a device once its
    /// engine is gone: no admitted batch is unfinished, and every job given to
    /// the builder thread has completed.
    ///
    /// An answered batch can leave jobs in flight, such as the release of
    /// its storage, whose completions announce nothing. Nothing more is
    /// admitted, so those jobs are let finish: their completions are taken
    /// as they arrive, as serving would have, until none is in flight. A
    /// job still in flight when it reaches the stall bound is stuck, and
    /// the rank is not settled.
    fn settled(&mut self) -> bool {
        loop {
            if self.executor.has_work() || self.executor.advance().is_err() {
                return false;
            }
            if self.executor.has_work() {
                return false;
            }
            let Some((_, since)) = self.executor.backend().oldest_job() else {
                return true;
            };
            let Some(left) = self.stall_bound.checked_sub(since.elapsed()) else {
                return false;
            };
            if self.executor.backend().await_completion(left).is_err() {
                return false;
            }
        }
    }

    /// Drain accepted results before acknowledging Close. A fatal result stops
    /// admission too, while preserving responses already owed to the engine.
    fn serve(&mut self) -> Result<(), WorkerError> {
        loop {
            let advanced = self.executor.advance()?;
            if self.send_ready()? || advanced {
                continue;
            }

            if self.closing && self.pending.is_empty() && !self.executor.has_work() {
                if self.experts {
                    match self.executor.backend_mut().join_expert_step(true)? {
                        // Every peer has left the group.
                        Some((_, true)) => {}
                        Some((_, false)) => continue,
                        // The join is in flight: its completion wakes.
                        None => {
                            self.wait(EXPERT_POLL)?;
                            continue;
                        }
                    }
                }

                if let Some(response) = self.shutdown.take() {
                    self.respond(&response)?;
                }

                // The rank closes its connection after this, which discards
                // what is still queued: a fatal error or this acknowledgment
                // may wait behind a large result the engine is still reading.
                self.channel.flush().map_err(|e| self.channel_error(e))?;
                return Ok(());
            }

            if !self.closing && self.pending.len() < self.executor.capacity() {
                let request = self.channel.try_recv().map_err(|e| self.channel_error(e))?;
                if let Some(request) = request {
                    self.accept(request)?;
                    continue;
                }
            }

            // A parked standby joins no expert step: its world's ranks run no
            // collective between parking and the promotion but the one that
            // restores their transports (`Parked::promote`), which each
            // reaches from here in any order.
            let joins = self.experts && self.standby.is_none();
            if joins {
                if self.awaits_own_step()? {
                    self.wait(EXPERT_POLL)?;
                    continue;
                }

                // A step that ran may be followed by another at once; an
                // empty round, or a join in flight, waits below.
                if let Some((true, _)) = self.executor.backend_mut().join_expert_step(false)? {
                    continue;
                }
            }

            if !self.pending.is_empty() || self.executor.has_work() {
                // Remote acknowledgment words have no wake source. Poll only
                // while one is outstanding; ordinary work waits on its wake.
                let timeout = if joins {
                    EXPERT_POLL
                } else if self.executor.backend().awaiting_acknowledgment()? {
                    Duration::from_millis(1)
                } else {
                    Duration::from_secs(60)
                };
                self.wait(timeout)?;
            } else if joins {
                self.wait(EXPERT_POLL)?;
            } else {
                // Nothing is owed or running, so only a request or the engine's
                // close can end this wait.
                self.wait(Duration::from_secs(60))?;
            }
        }
    }

    fn awaits_own_step(&self) -> Result<bool, WorkerError> {
        Ok(self.executor.backend().has_open_requests()?
            && (!self.pending.is_empty()
                || self.executor.has_work()
                || self.last_result.elapsed() < OWN_STEP_WAIT))
    }

    fn accept(&mut self, request: WorkerRequest) -> Result<(), WorkerError> {
        let pending = match request {
            WorkerRequest::Close { message_id } => {
                self.closing = true;
                self.shutdown = Some(WorkerResponse::Ok { message_id });
                return Ok(());
            }
            WorkerRequest::Info { message_id } => {
                PendingResponse::Ready(Box::new(WorkerResponse::Info {
                    message_id,
                    info: self.info.clone(),
                }))
            }
            WorkerRequest::Promote { message_id } => {
                let promoted = match self.standby.take() {
                    // The promoted standby replaces the rank it stood by
                    // for: it books its memory in the ledgers now, which the
                    // larger of its scope's processes' bookings counts.
                    Some(parked) => parked
                        .promote(self.executor)
                        .and_then(|()| self.book_memory(true).map_err(WorkerError::from)),
                    None => Err(WorkerError::from(Error::State(
                        "only a parked standby worker is promoted",
                    ))),
                };
                PendingResponse::Ready(Box::new(match promoted {
                    Ok(()) => WorkerResponse::Ok { message_id },
                    Err(error) => self.error_response(message_id, error)?,
                }))
            }
            WorkerRequest::Hook {
                message_id,
                name,
                payload,
            } => {
                let ran = if name.starts_with(BUILT_IN) {
                    self.built_in(&name, &payload)
                } else {
                    hook(&name, &payload)
                };
                PendingResponse::Ready(Box::new(match ran {
                    Ok(result) => WorkerResponse::Hooked { message_id, result },
                    Err(error) => self.error_response(message_id, WorkerError::from(error))?,
                }))
            }
            WorkerRequest::Submit { message_id, .. } if self.standby.is_some() => {
                PendingResponse::Ready(Box::new(self.error_response(
                    message_id,
                    WorkerError::from(Error::State("a standby worker serves once it is promoted")),
                )?))
            }
            WorkerRequest::Submit { message_id, batch } => {
                let batch = Arc::from(batch);
                let data = WorkerBatch::new(Arc::clone(&batch));
                let admitted = self.executor.admit(Batch::from_plan(&batch, data), false);
                match admitted {
                    Ok(submission) => PendingResponse::Batch {
                        message_id,
                        submission,
                    },
                    Err(error) => {
                        PendingResponse::Ready(Box::new(self.error_response(message_id, error)?))
                    }
                }
            }
        };
        self.pending.push_back(pending);
        Ok(())
    }

    fn send_ready(&mut self) -> Result<bool, WorkerError> {
        // A pending transfer or host task must not hold up independent results.
        for index in 0..self.pending.len() {
            let mut media = Vec::new();
            let response = match &self.pending[index] {
                PendingResponse::Ready(response) => (**response).clone(),
                PendingResponse::Batch {
                    message_id,
                    submission,
                } => {
                    let message_id = *message_id;
                    let result = self.executor.poll(submission);
                    match result {
                        // The result is checked once, here, where the rank
                        // builds its response; the engine does not check it
                        // again. A result that breaks the contract is the
                        // rank's own defect, and its media names are dropped
                        // with it.
                        Ok(Some(result)) => match result.output.validate() {
                            Ok(()) => {
                                media = result.media;
                                WorkerResponse::Result {
                                    message_id,
                                    result: result.output,
                                }
                            }
                            Err(invalid) => self.error_response(
                                message_id,
                                WorkerError::classify(
                                    Error::Invariant(format!("batch result: {invalid}")),
                                    None,
                                ),
                            )?,
                        },
                        Ok(None) => continue,
                        Err(error) => self.error_response(message_id, error)?,
                    }
                }
            };

            self.pending.remove(index);
            self.last_result = Instant::now();
            self.respond(&response)?;
            // A failed response drops the still-owned names. A successful
            // send leaves them available for the engine's claim.
            for source in media {
                source.into_locator();
            }
            if matches!(response, WorkerResponse::Error { ref error, .. } if error.fatal) {
                self.closing = true;
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn error_response(
        &self,
        message_id: Option<u64>,
        error: WorkerError,
    ) -> Result<WorkerResponse, WorkerError> {
        Ok(WorkerResponse::Error {
            message_id,
            error: self
                .executor
                .backend()
                .response_error(RequestKind::Submit, error)?,
        })
    }

    fn respond(&mut self, response: &WorkerResponse) -> Result<(), WorkerError> {
        self.channel
            .send(response)
            .map_err(|e| self.channel_error(e))
    }

    /// Report a job stalled past the bound, then block until a request, a
    /// completion wake, the timeout or the moment the job in flight longest
    /// reaches the bound. Fails once the engine's connection has closed: it is
    /// the engine's liveness, and nothing this rank computes can be delivered
    /// after it.
    fn wait(&mut self, timeout: Duration) -> Result<(), WorkerError> {
        let timeout = match self.watch_jobs()? {
            Some(until) => timeout.min(until),
            None => timeout,
        };
        self.channel
            .wait(timeout)
            .map_err(|e| self.channel_error(e))
    }

    /// Report the job the builder thread has performed for at least the stall
    /// bound, once; otherwise return how long until the job in flight longest
    /// reaches the bound. The pass that waits has taken every completion posted
    /// before its `advance`, so a job still in flight then had not completed.
    fn watch_jobs(&mut self) -> Result<Option<Duration>, WorkerError> {
        if self.stalled {
            return Ok(None);
        }
        let Some((job, since)) = self.executor.backend().oldest_job() else {
            return Ok(None);
        };
        let running = since.elapsed();
        // A context that an illegal memory access or a device-side assert
        // left unusable runs none of the work behind the failure, so the
        // completions of the jobs in flight never post: the rank fails with
        // the driver's error, which leaves its context unusable, at once
        // rather than at the stall bound.
        if running >= CONTEXT_PROBE && self.probed.elapsed() >= CONTEXT_PROBE {
            self.probed = Instant::now();
            let failed = self
                .maintenance
                .devices
                .iter()
                .find_map(|device| device.context_error().err());
            if let Some(error) = failed {
                let error = WorkerError::classify(Error::Cuda(error), Some("a job in flight"));
                // The engine reads the failure as the answer to the oldest
                // batch it awaits and fails the rank with its error, before
                // the connection closes as the rank ends.
                let awaited = self.pending.iter().find_map(|pending| match pending {
                    PendingResponse::Batch { message_id, .. } => Some(*message_id),
                    PendingResponse::Ready(_) => None,
                });
                if let Some(message_id) = awaited {
                    let response = self.error_response(message_id, error.clone())?;
                    self.respond(&response)?;
                    self.channel.flush().map_err(|e| self.channel_error(e))?;
                }
                return Err(error);
            }
        }
        if running < self.stall_bound {
            return Ok(Some((self.stall_bound - running).min(CONTEXT_PROBE)));
        }

        let batch = self.executor.batch_of(job);
        let stall = self.executor.backend().stall(job, running, batch);
        tracing::error!(
            batch_id = ?stall.batch_id,
            route = ?stall.route,
            method = ?stall.method,
            calls = stall.calls.len(),
            running_ms = stall.running_ms,
            "the builder thread has stayed inside one job past the stall bound of {:?}; reporting it to the engine",
            self.stall_bound,
        );
        self.stalled = true;
        self.respond(&WorkerResponse::Stalled { stall })?;
        Ok(None)
    }

    fn channel_error(&self, error: IpcError) -> WorkerError {
        self.channel_failed.set(true);
        WorkerError::classify(Error::transport_source("the engine channel", error), None)
    }
}

impl Drop for Service<'_> {
    fn drop(&mut self) {
        self.executor.set_completion_wake(None);
    }
}

/// How long a job stays in flight before the rank probes its devices'
/// contexts, and how often it probes them while one does.
const CONTEXT_PROBE: Duration = Duration::from_secs(1);

/// The prefix of the hooks the worker runs itself, which no plugin may
/// implement.
const BUILT_IN: &str = "uniserve.";

impl Service<'_> {
    /// Runs the built-in hook `name` with the JSON `payload`, while the rank
    /// runs no batch, and returns its JSON result:
    ///
    /// - `uniserve.cache_memory`: the cache planes' segments (count, units
    ///   and bytes of each, the first given back), the head segment and
    ///   graph share a successor world creates, the rank's scope, world and
    ///   host, the deployment's ledger of the cache's GPU, what the host
    ///   admits of new host memory, and the device's free memory;
    /// - `uniserve.release_cache` (`{"from": segment}`): gives back the
    ///   planes' segments from `segment` on, whose units the engine
    ///   withdrew;
    /// - `uniserve.restore_cache`: creates them again;
    /// - `uniserve.sleep`: suspends the graph memory and gives back every
    ///   segment; `uniserve.wake` restores both;
    /// - `uniserve.host_cache` (`{"tokens": count}`): maps the rank's host
    ///   page pools, each holding the pages of `count` tokens, the count
    ///   the engine sizes the world's host KV cache at, and answers what the
    ///   rank reports of each pool (`ipc::PagePoolInfo`); a rank that copies
    ///   no page through the host KV cache answers `null`;
    /// - `uniserve.eager` (`{"eager": bool}`): serves every forward and
    ///   expert join eagerly and gives back the captured graphs' memory, or
    ///   restores that memory at its addresses and serves by the graphs
    ///   again; a rank whose computations serve by their graphs alone
    ///   refuses;
    /// - `uniserve.update_weights` (`{"model_path": path}`): rewrites the
    ///   weights in place from the checkpoint at `path`
    ///   (`crate::worker::update`), and answers the rank's worker group and
    ///   the new key seed of each host page pool;
    /// - `uniserve.profile_start` (a `proto::profile::SessionStart`): starts
    ///   a profiling session on the rank (`crate::profiling::session`) and
    ///   answers whether the session selects the rank, its file and its
    ///   collectors' states;
    /// - `uniserve.profile_stop` (`{"session": id, "reason": text}`): ends
    ///   it and answers what its file holds;
    /// - `uniserve.profile_release` (`{"session": id, "collector": name,
    ///   "reason": text}`): turns one collector of it off;
    /// - `uniserve.fence_switch` (`{"enabled": bool}`): turns the fence
    ///   probes on or off (`crate::profiling::fences`).
    fn built_in(&mut self, name: &str, payload: &str) -> crate::Result<String> {
        let payload: serde_json::Value = if payload.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(payload)
                .map_err(|error| Error::Invalid(format!("hook payload: {error}")))?
        };
        let planes = self
            .executor
            .backend()
            .worker_storage()
            .and_then(|storage| storage.prefix_cache.as_ref())
            .map(|cache| Arc::clone(cache.planes()));
        let retained = self.maintenance.retained.clone();
        let mut shared = retained.as_deref();
        let mut local = crate::memory::retained::LocalMemory;
        let memory: &mut dyn crate::memory::retained::ExclusiveMemory = match &mut shared {
            Some(retained) => retained,
            None => &mut local,
        };
        let result = match name {
            "uniserve.cache_memory" => {
                // The cache's device; a rank without a cache (an expert
                // worker) reports its primary device's memory.
                let device = planes
                    .as_ref()
                    .and_then(|planes| planes.device())
                    .or_else(|| self.maintenance.devices.first().copied());
                // The device's free memory, which bounds what the cache
                // creates, and the graph share a successor's warmup captures
                // in (the share a serving worker's capacity plan keeps).
                let (free, graph_share) = match device {
                    Some(device) => {
                        let memory = device.memory_info()?;
                        (
                            memory.free,
                            crate::cuda_graph::graph_storage_budget_bytes(memory.total as i64),
                        )
                    }
                    None => (0, 0),
                };
                // The GPU's ledger, from which the engine computes what the
                // cache may hold beside every world of the deployment, and
                // what the host admits of new host memory.
                let (ledger, host_memory) = match (retained.as_deref(), device) {
                    (Some(retained), Some(device)) => {
                        let (ledger, host) = retained.ledger(device)?;
                        (Some(ledger), Some(host))
                    }
                    _ => (None, None),
                };
                let scratch = self
                    .executor
                    .backend()
                    .worker_storage()
                    .and_then(|storage| storage.prefix_cache.as_ref())
                    .map_or(0, crate::memory::PrefixCache::scratch_lent);
                // The head segment a successor creates beside its
                // predecessor's: the sentinel, the scratch of a warmup like
                // this rank's and one page of every group, in whole granules.
                let head_bytes = match (planes.as_ref(), device) {
                    (Some(planes), Some(device)) => {
                        let plan = planes.plan();
                        let granularity = cuda::HandleType::Descriptor.granularity(device)? as u64;
                        let page_units: u32 =
                            plan.groups.iter().map(|group| group.units_per_page).sum();
                        u64::from(crate::memory::PlaneLayout::whole_granules(
                            1 + scratch + page_units,
                            plan.plane_bytes,
                            granularity,
                        )) * planes.unit_bytes()
                    }
                    _ => 0,
                };
                // The units a configured token capacity holds, which bound a
                // successor's head and inherited segments together.
                let capacity_units = match (planes.as_ref(), self.maintenance.max_total_tokens) {
                    (Some(planes), Some(tokens)) => Some(
                        planes
                            .plan()
                            .token_units(tokens as u64, self.maintenance.request_slots as u64),
                    ),
                    _ => None,
                };
                serde_json::json!({
                    "segments": planes.as_ref().and_then(|planes| planes.segments()).unwrap_or_default(),
                    "held": planes.as_ref().map_or(0, |planes| planes.held_segments()),
                    "unit_bytes": planes.as_ref().map_or(0, |planes| planes.unit_bytes()),
                    "head_bytes": head_bytes,
                    "capacity_units": capacity_units,
                    "scratch_units": scratch,
                    "free_bytes": free,
                    "graph_share_bytes": graph_share,
                    "capturable": self.maintenance.capturable,
                    "host": self.info.endpoint.host,
                    "scope": retained.as_deref().map(|retained| retained.scope().clone()),
                    "world": retained.as_deref().map(crate::memory::retained::Retained::world),
                    "ledger": ledger,
                    "host_memory": host_memory,
                })
            }
            "uniserve.host_cache" => {
                let tokens = payload["tokens"]
                    .as_u64()
                    .ok_or_else(|| Error::Invalid("host_cache names no token count".into()))?;
                let Some(setup) = self.maintenance.host_cache.take() else {
                    return Ok(serde_json::Value::Null.to_string());
                };
                let retained = retained.as_deref().ok_or(Error::State(
                    "a host KV cache maps its pools at a Storage Server",
                ))?;
                let cache = self
                    .executor
                    .backend_mut()
                    .worker_storage_mut()
                    .and_then(|storage| storage.prefix_cache.as_mut())
                    .ok_or(Error::State("a host KV cache copies a rank's prefix cache"))?;
                let pools = setup.map(retained, cache, tokens)?;
                tracing::info!(tokens, pools = pools.len(), "mapped the host KV cache");
                serde_json::to_value(pools)
                    .map_err(|error| Error::Invalid(format!("host_cache result: {error}")))?
            }
            "uniserve.eager" => {
                let eager = payload["eager"]
                    .as_bool()
                    .ok_or_else(|| Error::Invalid("eager names no switch".into()))?;
                // A rank joins the union's steps of other schedulers outside
                // its own batches, replaying captured joins from the pools
                // given back below. The engine runs this hook once no
                // scheduler has a batch in flight (`engine::HookBarrier`), so
                // no step is agreed any more; the join in flight completes,
                // and its device work with it.
                while self.executor.backend().joining() {
                    self.executor.backend().await_completion(EXPERT_POLL)?;
                    self.executor.advance().map_err(|error| {
                        Error::Invariant(format!("the expert join in flight failed: {error}"))
                    })?;
                }
                // The executor switches its computations' forwards and its
                // expert union's joins. An expert-only rank holds no
                // computation, but its union replays captured joins from the
                // pools given back below: a replay over unmapped pool memory
                // faults on the device and leaves its peers waiting in the
                // step's exchange.
                let executor = self.executor.backend().executor();
                // No forward or join replays a graph while its memory is given
                // back: the executor switches before it goes and after it
                // returns.
                if eager {
                    if let Some(executor) = executor {
                        executor.run_eager(true)?;
                    }
                    for pool in &self.maintenance.graph_pools {
                        pool.suspend()?;
                    }
                    // The ledgers no longer book the graph memory given back,
                    // which the successor's preparation may then use.
                    self.book_memory(false)?;
                } else {
                    // The graph memory is booked again before it is restored,
                    // refused past the quota.
                    self.book_memory(true)?;
                    for pool in &self.maintenance.graph_pools {
                        pool.restore()?;
                    }
                    if let Some(executor) = executor {
                        executor.run_eager(false)?;
                    }
                }
                tracing::info!(eager, "switched how the rank serves its forwards");
                serde_json::json!({ "eager": eager })
            }
            "uniserve.release_cache" => {
                let first = payload["from"]
                    .as_u64()
                    .and_then(|first| u32::try_from(first).ok())
                    .ok_or_else(|| Error::Invalid("release_cache names no segment".into()))?;
                // A rank that holds no cache, such as an expert worker, has
                // none to give back.
                if let Some(planes) = planes {
                    planes.release_segments(first, memory)?;
                }
                serde_json::Value::Null
            }
            "uniserve.restore_cache" => {
                let limit = payload["segments"]
                    .as_u64()
                    .and_then(|count| u32::try_from(count).ok());
                restore(planes.as_ref(), memory, limit)?;
                serde_json::Value::Null
            }
            "uniserve.adopt_cache" => {
                // The world's promotion: the predecessor parked, so the
                // flags and scales of the units the planes inherit are
                // final, and the rank copies them into its own.
                let (Some(planes), Some(predecessor), Some(retained)) = (
                    planes.as_ref(),
                    self.maintenance.predecessor.as_ref(),
                    retained.as_deref(),
                ) else {
                    return Ok(serde_json::Value::Null.to_string());
                };
                let device = planes
                    .device()
                    .ok_or_else(|| Error::State("a host pool inherits no cache"))?;
                let plan = planes.plan();
                let cells = plan.columns as usize * predecessor.num_units as usize;
                let peers = crate::memory::allocation::Storage::peers(device)?;
                let mut names = Vec::new();
                for field in crate::memory::prefix_cache::planes::FIELDS {
                    names.push((format!("prefix_cache.{field}.initialized"), cells));
                    if plan.quantized {
                        names.push((format!("prefix_cache.{field}.scale"), cells * 4));
                    }
                }
                let specs: Vec<crate::memory::retained::ExclusiveSpec<'_>> = names
                    .iter()
                    .map(|(name, bytes)| crate::memory::retained::ExclusiveSpec {
                        name,
                        device,
                        bytes: *bytes,
                        access: &peers,
                    })
                    .collect();
                let mapped = retained.map_inherited(&specs, predecessor.world)?;
                let storages: Vec<_> = mapped
                    .iter()
                    .map(|(_, storage)| Arc::clone(storage))
                    .collect();
                let stream = cuda::Stream::new(device)?;
                planes.adopt_inherited(&storages, predecessor.num_units, &stream)?;
                let keys: Vec<_> = mapped.iter().map(|(key, _)| *key).collect();
                drop(storages);
                drop(mapped);
                retained.forget(&keys)?;
                let inherited = planes.inherited();
                tracing::info!(
                    units = inherited.end - inherited.start,
                    "adopted the predecessor world's cache flags"
                );
                serde_json::json!({ "inherited": inherited.end - inherited.start })
            }
            "uniserve.take_cache" => {
                // The world's commit: its predecessor is gone, and the rank
                // takes the segments it mapped read-only.
                let taken = match (planes.as_ref(), retained.as_deref()) {
                    (Some(planes), Some(retained)) => planes.take_inherited(retained)?,
                    _ => 0,
                };
                serde_json::json!({ "segments": taken })
            }
            "uniserve.sleep" => {
                // Asleep from the start: a wake restores whatever a sleep
                // that failed part of the way gave back.
                self.maintenance.asleep = true;
                for pool in &self.maintenance.pools {
                    pool.suspend()?;
                }
                if let Some(planes) = &planes
                    && planes.segments().is_some()
                {
                    planes.release_segments(0, memory)?;
                }
                // The ledgers no longer book the released graph memory.
                self.book_memory(false)?;
                tracing::info!("the worker sleeps: its graph memory and cache are released");
                serde_json::Value::Null
            }
            "uniserve.wake" => {
                if self.maintenance.asleep {
                    // The graph memory is booked again before it and the
                    // cache are restored, refused past the quota.
                    self.book_memory(true)?;
                    for pool in &self.maintenance.pools {
                        pool.restore()?;
                    }
                    restore(planes.as_ref(), memory, None)?;
                    self.maintenance.asleep = false;
                    tracing::info!("the worker woke: its graph memory and cache are restored");
                }
                serde_json::Value::Null
            }
            "uniserve.update_weights" => {
                let path = payload["model_path"]
                    .as_str()
                    .ok_or_else(|| Error::Invalid("update_weights names no model_path".into()))?;
                let copies_pages = self
                    .executor
                    .backend()
                    .worker_storage()
                    .and_then(|storage| storage.prefix_cache.as_ref())
                    .is_some_and(crate::memory::PrefixCache::copies_pages);
                let plan = planes.as_ref().map(|planes| Arc::clone(planes.plan()));
                // The requests in flight keep their KV cache through the
                // update: the checkpoint loads beside the cache, in the
                // graph memory as well when the free memory alone does not
                // hold it.
                let keep = payload["keep"].as_bool().unwrap_or(false) && !self.maintenance.asleep;
                let suspended = keep && self.make_room_for_weights()?;
                let updated = match (self.maintenance.weights.as_mut(), retained.as_deref()) {
                    (Some(weights), Some(retained)) => {
                        weights.apply(path, retained, plan.as_deref(), copies_pages)
                    }
                    _ => Ok(serde_json::json!({ "group": null, "page_contents": [] })),
                };
                // The checkpoint copy is gone: its staging booking ends, and
                // the graph memory given back for it returns and is booked
                // again.
                if keep {
                    self.book_staging(&BTreeMap::new())?;
                }
                if suspended {
                    for pool in &self.maintenance.pools {
                        pool.restore()?;
                    }
                    self.book_memory(true)?;
                }
                updated?
            }
            "uniserve.profile_start" => {
                let start: proto::profile::SessionStart = serde_json::from_value(payload)
                    .map_err(|error| Error::Invalid(format!("profile_start: {error}")))?;
                let rank = self.maintenance.rank;
                let device = self.maintenance.devices.first().copied();
                // The device storage the session's collectors will hold is
                // booked with the rank's process before they start, so the
                // ledger's plans count it while the session holds it; a
                // ledger that cannot fit it refuses the session.
                let held =
                    crate::profiling::session::collectors::device_storage(&start.request, rank);
                match self.book_profiling(device, held)? {
                    Some(refusal) => serde_json::json!({
                        "rank": rank,
                        "selected": true,
                        "refused": format!(
                            "range_metrics holds {held} bytes of device storage for its range \
                             profiler pass, which the ledger of the rank's GPU cannot fit: \
                             {refusal}"
                        ),
                    }),
                    None => match self
                        .executor
                        .backend_mut()
                        .start_session(start, rank, device)
                    {
                        Ok(answer) => answer,
                        Err(error) => {
                            self.unbook_profiling(device);
                            return Err(error);
                        }
                    },
                }
            }
            "uniserve.profile_stop" => {
                let session = payload["session"]
                    .as_u64()
                    .ok_or_else(|| Error::Invalid("profile_stop names no session".into()))?;
                let reason = payload["reason"].as_str().unwrap_or("stopped");
                let answer = self.executor.backend_mut().stop_session(
                    session,
                    reason,
                    self.maintenance.rank,
                );
                // The ended session's collectors hold no device storage.
                self.unbook_profiling(self.maintenance.devices.first().copied());
                answer
            }
            "uniserve.profile_release" => {
                let session = payload["session"]
                    .as_u64()
                    .ok_or_else(|| Error::Invalid("profile_release names no session".into()))?;
                let collector: proto::profile::Collector =
                    serde_json::from_value(payload["collector"].clone())
                        .map_err(|error| Error::Invalid(format!("profile_release: {error}")))?;
                let reason = payload["reason"].as_str().unwrap_or("released");
                self.executor.backend_mut().release_collector(
                    session,
                    collector,
                    reason,
                    self.maintenance.rank,
                )
            }
            "uniserve.fence_switch" => {
                let enabled = payload["enabled"]
                    .as_bool()
                    .ok_or_else(|| Error::Invalid("fence_switch states no enabled".into()))?;
                let switched = match crate::profiling::fences::Fences::installed() {
                    Some(fences) => {
                        fences.switch(enabled)?;
                        true
                    }
                    None => false,
                };
                serde_json::json!({ "probes": switched, "enabled": enabled })
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "no built-in worker hook {other}"
                )));
            }
        };
        Ok(result.to_string())
    }
}

impl Service<'_> {
    /// Books the warm rank's memory in each device's ledger, its graph
    /// pools while `graphs` are resident; a rank whose memory the server
    /// does not keep (`LocalMemory`) books none.
    fn book_memory(&self, graphs: bool) -> crate::Result<()> {
        let Some(retained) = self.maintenance.retained.as_deref() else {
            return Ok(());
        };
        for booking in &self.maintenance.bookings {
            booking.submit(retained, graphs)?;
        }
        Ok(())
    }

    /// Books `bytes` of device storage a profiling session's collectors
    /// hold on `device` with the rank's process in the device's ledger,
    /// replacing what an earlier session booked. Answers the ledger's
    /// refusal, with the booking unchanged, when it cannot fit them. A rank
    /// that books nothing (`LocalMemory`, or a parked standby, which books
    /// at its promotion) books none of them either.
    fn book_profiling(
        &mut self,
        device: Option<cuda::DeviceId>,
        bytes: u64,
    ) -> crate::Result<Option<String>> {
        let (Some(retained), Some(device)) = (self.maintenance.retained.clone(), device) else {
            return Ok(None);
        };
        if self.standby.is_some() {
            return Ok(None);
        }
        let graphs = !self.maintenance.asleep;
        let Some(booking) = self
            .maintenance
            .bookings
            .iter_mut()
            .find(|booking| booking.device == device)
        else {
            return Ok(None);
        };
        if booking.profiling == bytes {
            return Ok(None);
        }
        let booked = std::mem::replace(&mut booking.profiling, bytes);
        match booking.submit(&retained, graphs) {
            Ok(_) => Ok(None),
            Err(error) => {
                booking.profiling = booked;
                match error {
                    // `Retained::book` answers the server's refusal of a
                    // booking past the quota so.
                    Error::Unsupported(refusal) => Ok(Some(refusal)),
                    error => Err(error),
                }
            }
        }
    }

    /// Books no profiling storage on `device` any more. A failure leaves
    /// the larger booking in the ledger, which only overstates the rank's
    /// storage, so it is logged rather than failing the session's end.
    fn unbook_profiling(&mut self, device: Option<cuda::DeviceId>) {
        let failure = match self.book_profiling(device, 0) {
            Ok(None) => return,
            Ok(Some(refusal)) => refusal,
            Err(error) => error.to_string(),
        };
        tracing::warn!("unbooking the ended profiling session's device storage failed: {failure}");
    }

    /// Makes the device memory beside the KV cache hold a checkpoint of the
    /// rank's weights, which an update keeping the requests in flight loads
    /// into new memory before it copies each tensor into place. The copy is
    /// booked in each device's ledger as the rank's staging, within the
    /// deployment's quota, and must fit the device's free memory; the graph
    /// memory is given back (and its booking ended) when the quota or the
    /// free memory does not hold the copy beside it. Returns whether it was
    /// given back, for the caller to restore once the update ends.
    ///
    /// # Errors
    ///
    /// Refuses, with the graph memory restored and booked and the staging
    /// unbooked, when even without the graph memory the quota or the free
    /// memory does not hold the checkpoint.
    fn make_room_for_weights(&mut self) -> crate::Result<bool> {
        let Some(weights) = self.maintenance.weights.as_ref() else {
            return Ok(false);
        };
        let mut needed: BTreeMap<cuda::DeviceId, u64> = BTreeMap::new();
        for storage in &weights.storages {
            if let crate::ffi::MemoryKind::Cuda(device) = crate::ffi::Backing::device(&**storage) {
                *needed.entry(device).or_default() += crate::ffi::Backing::bytes(&**storage) as u64;
            }
        }
        let short = |needed: &BTreeMap<cuda::DeviceId, u64>| -> crate::Result<Option<(cuda::DeviceId, u64, u64)>> {
            for (&device, &bytes) in needed {
                let free = device.memory_info()?.free;
                if free < bytes {
                    return Ok(Some((device, bytes, free)));
                }
            }
            Ok(None)
        };
        if self.book_staging(&needed).is_ok() && short(&needed)?.is_none() {
            return Ok(false);
        }
        for pool in &self.maintenance.pools {
            pool.suspend()?;
        }
        self.book_memory(false)?;
        let refusal = match self.book_staging(&needed) {
            Err(error) => Some(format!("{error}")),
            Ok(()) => short(&needed)?.map(|(device, bytes, free)| {
                format!(
                    "the checkpoint's {bytes} bytes on {device} exceed the {free} bytes free with \
                     the graph memory given back"
                )
            }),
        };
        if let Some(refusal) = refusal {
            self.book_staging(&BTreeMap::new())?;
            for pool in &self.maintenance.pools {
                pool.restore()?;
            }
            self.book_memory(true)?;
            return Err(Error::Unsupported(format!(
                "keeping the requests in flight through the weight update loads the checkpoint \
                 beside their KV cache: {refusal}; update with the requests in flight \
                 recomputed or aborted instead"
            )));
        }
        Ok(true)
    }

    /// Books `bytes` of staging on each device of the rank's ledger
    /// bookings, those `bytes` does not name at 0: a transient copy the rank
    /// holds beside its other memory. Refused, past the quota, with the
    /// ledger's entries.
    fn book_staging(&self, bytes: &BTreeMap<cuda::DeviceId, u64>) -> crate::Result<()> {
        let Some(retained) = self.maintenance.retained.as_deref() else {
            return Ok(());
        };
        for booking in &self.maintenance.bookings {
            let staging = bytes.get(&booking.device).copied().unwrap_or(0);
            retained.book(
                booking.device,
                booking.total,
                booking.quota,
                vec![(storage::Item::Staging, staging)],
            )?;
        }
        Ok(())
    }
}

/// Creates the segments `planes` gave back again, through `memory`.
fn restore(
    planes: Option<&Arc<crate::memory::CachePlanes>>,
    memory: &mut dyn crate::memory::retained::ExclusiveMemory,
    limit: Option<u32>,
) -> crate::Result<()> {
    let Some(planes) = planes else {
        return Ok(());
    };
    let Some(device) = planes.device() else {
        return Ok(());
    };
    if planes.segments().is_none() {
        return Ok(());
    }
    let stream = cuda::Stream::new(device)?;
    planes.restore_segments(memory, &stream, limit)
}

/// Runs the worker hook `name` with `payload` (`uniserve.hooks.call`) and
/// returns its JSON result. The engine sends a hook while the rank runs no
/// forward; the hook's device work completes before it returns.
fn hook(name: &str, payload: &str) -> crate::Result<String> {
    use crate::ffi::Interpreter;
    use tvm_ffi::Any;

    let result: tvm_ffi::String = Interpreter::import("uniserve.hooks.call")?
        .call_values(&[
            Any::from(tvm_ffi::String::from(name)),
            Any::from(tvm_ffi::String::from(payload)),
        ])?
        .try_into()
        .map_err(|error| Error::numerical("hook", &error))?;
    Ok(result.as_str().to_owned())
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use ipc::{
        Batch as Plan, BatchCommand, Channel, GenerationParams, IpcError, Listener, NewRequest,
        RequestKey, WorkerRequest, WorkerResponse,
    };
    use ipc::{RequestId, SamplingParams};

    use super::{Ending, Service};
    use crate::error::WorkerError;
    use crate::executor::tests::rank;
    use crate::{Error, ErrorCode};

    type Result<T> = std::result::Result<T, WorkerError>;

    /// How long the engine waits for one response.
    const RESPONSE_WAIT: Duration = Duration::from_secs(30);

    fn transport(error: IpcError) -> WorkerError {
        Error::transport_source("the test channel", error).into()
    }

    /// The rank's end of a connection whose engine end runs `engine` on its
    /// own thread, and the engine's outcome once `serve` returned.
    fn connect<T: Send + 'static, S>(
        engine: impl FnOnce(Channel) -> Result<T> + Send + 'static,
        serve: impl FnOnce(&mut Channel) -> Result<S>,
    ) -> Result<(Result<S>, Result<T>)> {
        let listener = Listener::bind(&["127.0.0.1:0"]).map_err(transport)?;
        let address = listener.addresses()[0].clone();
        let engine =
            std::thread::spawn(move || engine(Channel::connect(&address).map_err(transport)?));

        let deadline = Instant::now() + RESPONSE_WAIT;
        let mut channel = loop {
            if let Some(channel) = listener.try_accept().map_err(transport)? {
                break channel;
            }
            if Instant::now() > deadline {
                return Err(Error::State("the engine did not connect").into());
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        let served = serve(&mut channel);
        // The rank closes its end, as it does when it exits.
        drop(channel);
        let outcome = engine
            .join()
            .map_err(|_| WorkerError::from(Error::State("the engine thread panicked")))?;
        Ok((served, outcome))
    }

    /// The next response the engine receives, or `None` once the rank has
    /// closed the connection.
    fn receive(channel: &mut Channel) -> Result<Option<WorkerResponse>> {
        let deadline = Instant::now() + RESPONSE_WAIT;
        loop {
            match channel.try_recv() {
                Ok(Some(response)) => return Ok(Some(response)),
                Ok(None) => {}
                Err(IpcError::Transport(_)) => return Ok(None),
                Err(error) => return Err(transport(error)),
            }
            if Instant::now() > deadline {
                return Err(Error::State("the rank neither responded nor closed").into());
            }
            match channel.wait(Duration::from_millis(100)) {
                Ok(()) => {}
                Err(IpcError::Transport(_)) => return Ok(None),
                Err(error) => return Err(transport(error)),
            }
        }
    }

    /// Send `request` and receive the response the rank sends next.
    fn ask(channel: &mut Channel, request: WorkerRequest) -> Result<WorkerResponse> {
        channel.send(&request).map_err(transport)?;
        receive(channel)?
            .ok_or_else(|| Error::State("the rank closed the connection unanswered").into())
    }

    /// A Submit of a lifecycle batch starting request `id` in `slot`.
    fn submit(message_id: u64, batch_id: u64, id: u64, slot: u32) -> Result<WorkerRequest> {
        let generation = GenerationParams {
            sampling: SamplingParams::default(),
            negative_token_ids: Vec::new(),
            finish_token_ids: Vec::new(),
            initial_position: 0,
            generated_token_ids: Vec::new(),
            adapter: 0,
        };
        let request = NewRequest::new(
            RequestKey::new(1, RequestId(id), 1),
            slot,
            Some(generation),
            None,
            0,
        )
        .map_err(|error| Error::Invalid(error.to_string()))?;
        let mut batch = Plan::new(batch_id, Vec::new(), Vec::new());
        batch.commands = vec![BatchCommand::Start {
            request: Box::new(request),
        }];
        Ok(WorkerRequest::Submit {
            message_id: Some(message_id),
            batch: Box::new(batch),
        })
    }

    /// Serve the connection with a rank without components that queues
    /// two batches. No job of these batches stays in flight, so the stall
    /// bound (here the default, half the engine's 300 s watchdog timeout)
    /// is never reached.
    fn serve(channel: &mut Channel) -> Result<Ending> {
        let mut executor = rank(2, false)?;
        let info = executor.backend().info().clone();
        Service::new(
            &mut executor,
            channel,
            info,
            false,
            Duration::from_secs(150),
            None,
            super::Maintenance {
                retained: None,
                weights: None,
                pools: Vec::new(),
                asleep: false,
                _counters: None,
                rank: 0,
                predecessor: None,
                max_total_tokens: None,
                request_slots: 0,
                devices: Vec::new(),
                bookings: Vec::new(),
                host_cache: None,
                graph_pools: Vec::new(),
                capturable: false,
            },
        )
        .run()
    }

    #[test]
    fn the_service_answers_each_request_and_acknowledges_close() -> Result<()> {
        let engine = |mut channel: Channel| -> Result<()> {
            let info = ask(
                &mut channel,
                WorkerRequest::Info {
                    message_id: Some(1),
                },
            )?;
            assert!(matches!(
                info,
                WorkerResponse::Info {
                    message_id: Some(1),
                    ..
                }
            ));

            // An admitted batch is answered with its result.
            let result = ask(&mut channel, submit(2, 1, 7, 1)?)?;
            assert!(matches!(
                result,
                WorkerResponse::Result { message_id: Some(2), result } if result.batch_id == 1
            ));

            // A refused batch id and a failed batch are reported under their
            // messages; neither stops the rank.
            for (message_id, request) in [(3, submit(3, 1, 8, 2)?), (4, submit(4, 2, 8, 9)?)] {
                let error = ask(&mut channel, request)?;
                assert!(
                    matches!(
                        &error,
                        WorkerResponse::Error { message_id: Some(id), error }
                            if *id == message_id && !error.fatal
                    ),
                    "{error:?}"
                );
            }
            let info = ask(
                &mut channel,
                WorkerRequest::Info {
                    message_id: Some(5),
                },
            )?;
            assert!(matches!(
                info,
                WorkerResponse::Info {
                    message_id: Some(5),
                    ..
                }
            ));

            // Close is acknowledged, then the rank ends the connection.
            let closed = ask(
                &mut channel,
                WorkerRequest::Close {
                    message_id: Some(6),
                },
            )?;
            assert_eq!(
                closed,
                WorkerResponse::Ok {
                    message_id: Some(6)
                }
            );
            assert_eq!(receive(&mut channel)?, None);
            Ok(())
        };

        let (served, engine) = connect(engine, serve)?;
        engine?;
        assert!(matches!(served?, Ending::Closed));
        Ok(())
    }

    #[test]
    fn the_service_stops_once_the_engine_has_closed_its_connection() -> Result<()> {
        // The engine goes away without asking for anything, so nothing is in
        // flight: the rank is orphaned, and the failure names the channel.
        let (served, engine) = connect(
            |channel: Channel| {
                drop(channel);
                Ok(())
            },
            serve,
        )?;
        engine?;
        let Ending::Orphaned(error) = served? else {
            return Err(Error::State("the service did not end orphaned").into());
        };
        assert_eq!(error.code, ErrorCode::ComputeError);
        Ok(())
    }

    #[test]
    fn an_engine_that_closes_after_its_answer_orphans_the_worker() -> Result<()> {
        // The engine's batch was answered, so nothing is in flight when it
        // goes away.
        let engine = |mut channel: Channel| -> Result<()> {
            let result = ask(&mut channel, submit(1, 1, 7, 1)?)?;
            assert!(matches!(result, WorkerResponse::Result { .. }));
            Ok(())
        };
        let (served, engine) = connect(engine, serve)?;
        engine?;
        assert!(matches!(served?, Ending::Orphaned(_)));
        Ok(())
    }

    #[test]
    fn a_message_that_is_not_a_request_stops_the_service_unanswered() -> Result<()> {
        let engine = |mut channel: Channel| -> Result<bool> {
            // A frame whose message is no request. The rank ends the
            // connection without a response.
            channel.send(&true).map_err(transport)?;
            Ok(receive(&mut channel)?.is_none())
        };
        let (served, unanswered) = connect(engine, serve)?;
        assert!(matches!(served?, Ending::Orphaned(_)));
        assert!(unanswered?, "the rank answered a message it could not read");
        Ok(())
    }
}
