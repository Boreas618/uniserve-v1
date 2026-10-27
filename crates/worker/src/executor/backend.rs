//! The worker's execution backend: request state, lifecycle commands,
//! batch validation, numerical dispatch and result delivery.
//!
//! [`WorkerBackend`] does the work of each step the [`super::Executor`] orders
//! and answers the rank service. It owns the request pool, the request storage
//! owners (through [`RequestStorage`]) and the placed components, and
//! dispatches each batch's calls to what serves their kind (an executor
//! computation or the transfers), whose forward it runs as jobs of the rank's
//! job queue. Lifecycle commands (Start, Finish, Free), batch admission,
//! dependencies, retirement and result delivery are handled here for every
//! batch, with or without calls.

mod steps;

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use ipc::{Batch as BatchPlan, BatchCommand, RequestKind, Stall, WorkerInfo, WorkerResponseError};

use super::computations::Computations;
use super::experts::ExpertMember;
use super::lifecycle::{CommandJob, RequestStorage};
use super::python::PythonExecutor;
use crate::config::placement::{Components, GroupPlacement};
use crate::error::WorkerError;
use crate::job::{Job, JobId, JobQueue};
use crate::profiling::kernels::KernelWatch;
use crate::profiling::profiler::Profiler;
use crate::profiling::session::Session;
use crate::request::RequestPool;
use crate::worker::WorkerMemory;
use crate::{Error, Result};

pub(crate) use steps::WorkerBatch;

/// The backend of one rank.
pub(crate) struct WorkerBackend {
    info: WorkerInfo,
    rank: usize,
    /// The placement the rank's group was launched with.
    placement: GroupPlacement,
    requests: RequestPool,
    /// What serves the rank's components, beside the request pool and the
    /// lifecycle jobs every lifecycle command borrows with its owners.
    owners: Owners,
    components: Components,
    profiler: Option<Profiler>,
    /// The profiling session that records this rank, while one does.
    session: Option<Session>,
    kernels: Option<KernelWatch>,
    /// The rank's job queue: each forward runs as jobs the builder thread
    /// performs.
    jobs: JobQueue,
    /// What lifecycle commands record for the builder thread; submitted after
    /// each command.
    lifecycle: CommandJob,
    /// The submitted lifecycle jobs whose completions have not arrived.
    lifecycle_jobs: HashSet<JobId>,
    /// The job that joins the expert union's step this rank runs no
    /// forward of, while it is in flight.
    joining: Option<JobId>,
    /// How the last such join ended, until the service takes it.
    joined: Option<std::result::Result<(bool, bool), WorkerError>>,
    /// The control thread's counts of driver calls that waited for the
    /// device when the last batch result left ([`cuda::cupti::blocking::Calls`]).
    control_calls: cuda::cupti::blocking::Calls,
}

/// What serves a rank's components and the storage it owns: the rank's storage
/// owners, the executor and its computations, an expert union's member, the
/// weights streamed ahead of their layers and the transfer runtime.
#[derive(Default)]
pub(crate) struct Owners {
    /// The rank's storage owners (events, products, transports, prefix cache).
    worker_storage: Option<WorkerMemory>,
    /// The rank's executor, shared by the computations, the expert union member
    /// and warmup, and closed after them.
    executor: Option<Arc<PythonExecutor>>,
    /// The executor's computations and the resources the worker keeps for them.
    computations: Option<Computations>,
    /// This rank's member of its expert union, whose steps the executor
    /// agrees on and runs.
    experts: Option<ExpertMember>,
    /// Distributed expert weights, on ranks that prefetch them.
    weights: Option<crate::parallel::DwdpManager>,
    /// Layer stacks streamed from host memory, on ranks of components that
    /// keep their weights there.
    host_weights: Vec<crate::parallel::HostWeights>,
    /// The transfer runtime: tensor transfers, cache exports and installs, and
    /// the streams batch imports run on.
    transfer: Option<super::transfer::TransferRuntime>,
}

impl Owners {
    /// Every request storage owner, in lifecycle order: rank storage, the
    /// executor's computations (their output leases and latent ledger), then
    /// the transfer runtime.
    pub(crate) fn request_storage(&mut self) -> impl Iterator<Item = &mut dyn RequestStorage> {
        fn owner<T: RequestStorage>(owner: &mut T) -> &mut dyn RequestStorage {
            owner
        }

        self.worker_storage
            .as_mut()
            .map(owner)
            .into_iter()
            .chain(self.computations.as_mut().map(owner))
            .chain(self.transfer.as_mut().map(owner))
    }
}

/// Disjoint borrows of a backend for one batch's execution.
pub(crate) struct Parts<'a> {
    pub placement: &'a GroupPlacement,
    pub rank: usize,
    pub requests: &'a mut RequestPool,
    pub storage: Option<&'a mut WorkerMemory>,
    pub computations: Option<&'a mut Computations>,
    pub transfer: Option<&'a mut super::transfer::TransferRuntime>,
}

impl WorkerBackend {
    pub(crate) fn new(
        info: WorkerInfo,
        rank: usize,
        placement: GroupPlacement,
        components: Components,
    ) -> Result<Self> {
        Ok(Self {
            requests: RequestPool::new(info.request_slots as usize)?,
            info,
            rank,
            placement,
            owners: Owners::default(),
            components,
            profiler: None,
            session: None,
            kernels: None,
            jobs: JobQueue::default(),
            lifecycle: CommandJob::default(),
            lifecycle_jobs: HashSet::new(),
            joining: None,
            joined: None,
            control_calls: cuda::cupti::blocking::Calls::default(),
        })
    }

    /// Submit the job the last lifecycle command recorded without waiting for
    /// it: the builder thread performs jobs in the order they were submitted,
    /// so a forward submitted later finds its work done. Nothing goes on from
    /// its completion, which is taken with the next one that arrives
    /// ([`WorkerBackend::completed`]); a failed one stops the rank there.
    fn perform_lifecycle(&mut self) -> Result<()> {
        if let Some(job) = self.lifecycle.take() {
            let submitted = self.start(job, false)?;
            self.lifecycle_jobs.insert(submitted);
        }
        Ok(())
    }

    /// Submit `job` to the builder thread. `wake` is as for
    /// [`JobQueue::start`].
    ///
    /// The deferred work the job carries waits for events the job records. The
    /// job whose completion says they completed is started right behind it, and
    /// the event pool runs the work when that completion arrives
    /// ([`WorkerBackend::completed`]). Started earlier, that job would reach
    /// the builder thread before the records and pass at once.
    fn start(&self, job: Job, wake: bool) -> Result<JobId> {
        let deferrals = job.deferrals();
        let submitted = self.jobs.start(job, wake);
        if !deferrals.is_empty() {
            let storage = self.owners.worker_storage.as_ref().ok_or(Error::State(
                "a job defers work on a rank without an event pool",
            ))?;
            let jobs = &self.jobs;
            storage
                .events
                .defer_submitted(deferrals, |device, events| jobs.after(device, events))?;
        }
        Ok(submitted)
    }

    /// Have the rank's event pool start the jobs its deferred work waits for,
    /// from this thread, the control thread.
    fn serve_events(&self) {
        let Some(storage) = &self.owners.worker_storage else {
            return;
        };
        let jobs = self.jobs.clone();
        storage
            .events
            .set_submitter(Some(Box::new(move |device, events| {
                jobs.after(device, events)
            })));
    }

    /// Use the rank's job queue for forward submission.
    pub(crate) fn set_job_queue(&mut self, jobs: JobQueue) {
        self.jobs = jobs;
        self.serve_events();
    }

    /// End a batch's forward: close its profiler and session step and ask
    /// for the count of kernel choices, which tells whether its calls
    /// selected new kernels.
    fn end_forward(&mut self, batch: &mut WorkerBatch) {
        if std::mem::take(&mut batch.stepping) {
            if let Some(profiler) = &mut self.profiler {
                profiler.end();
            }
            if let Some(session) = &mut self.session {
                session.end(batch.plan.batch_id);
            }
            if let Some(watch) = &mut self.kernels {
                watch.count(&self.jobs);
            }
        }
    }

    pub(crate) fn info(&self) -> &WorkerInfo {
        &self.info
    }

    pub(crate) fn requests(&self) -> &RequestPool {
        &self.requests
    }

    pub(crate) fn requests_mut(&mut self) -> &mut RequestPool {
        &mut self.requests
    }

    pub(crate) fn components(&self) -> &Components {
        &self.components
    }

    /// Install the rank's storage owners.
    pub(crate) fn set_worker_storage(&mut self, storage: WorkerMemory) {
        self.owners.worker_storage = Some(storage);
        self.serve_events();
    }

    pub(crate) fn worker_storage(&self) -> Option<&WorkerMemory> {
        self.owners.worker_storage.as_ref()
    }

    pub(crate) fn worker_storage_mut(&mut self) -> Option<&mut WorkerMemory> {
        self.owners.worker_storage.as_mut()
    }

    /// Install the rank's executor, which the backend closes after the
    /// computations and the expert union that call it.
    pub(crate) fn set_executor(&mut self, executor: Arc<PythonExecutor>) {
        self.owners.executor = Some(executor);
    }

    /// The rank's executor.
    pub(crate) fn executor(&self) -> Option<&Arc<PythonExecutor>> {
        self.owners.executor.as_ref()
    }

    /// Install the executor's computations.
    pub(crate) fn set_computations(&mut self, computations: Computations) {
        self.owners.computations = Some(computations);
    }

    /// The executor's computations.
    pub(crate) fn computations(&self) -> Option<&Computations> {
        self.owners.computations.as_ref()
    }

    /// Whether an executor computation serves calls of `kind` on this rank.
    pub(crate) fn executes(&self, kind: ipc::CallKind) -> bool {
        self.owners
            .computations
            .as_ref()
            .is_some_and(|computations| computations.serves(kind))
    }

    /// The parts a computation's batch executes with; `None` on a rank without
    /// executor computations or storage owners.
    pub(crate) fn computation_parts(&mut self) -> Option<super::computations::Parts<'_>> {
        Some(super::computations::Parts {
            placement: &self.placement,
            rank: self.rank,
            info: &self.info,
            requests: &mut self.requests,
            storage: self.owners.worker_storage.as_mut()?,
            computations: self.owners.computations.as_mut()?,
        })
    }

    /// Install this rank's member of its expert union.
    pub(crate) fn set_experts(&mut self, experts: ExpertMember) {
        self.owners.experts = Some(experts);
    }

    /// This rank's member of its expert union.
    pub(crate) fn experts(&self) -> Option<&ExpertMember> {
        self.owners.experts.as_ref()
    }

    /// Install the distributed expert weights.
    pub(crate) fn set_weights(&mut self, weights: crate::parallel::DwdpManager) {
        self.owners.weights = Some(weights);
    }

    /// Install the layer stacks streamed from host memory.
    pub(crate) fn set_host_weights(&mut self, weights: Vec<crate::parallel::HostWeights>) {
        self.owners.host_weights = weights;
    }

    /// The rank's job queue, for work that runs as jobs outside a batch (warmup
    /// captures).
    pub(crate) fn job_queue(&self) -> &JobQueue {
        &self.jobs
    }

    /// Release the expert union member's storage and the weights copied
    /// ahead of their layers after the executor closed the bindings,
    /// graphs and exchanges using them.
    pub(crate) fn close_experts(&mut self, aborted: bool) -> Result<()> {
        if let Some(mut experts) = self.owners.experts.take() {
            experts.close(aborted);
        }
        let weights = match &mut self.owners.weights {
            Some(weights) => weights.close(aborted),
            None => Ok(()),
        };
        // Host weights' copies run on the library's streams, which a close
        // drains; an aborted close keeps their memory until exit, since
        // copies may still be queued.
        let mut host = Ok(());
        for streamed in std::mem::take(&mut self.owners.host_weights) {
            if aborted {
                crate::leak(streamed);
            } else {
                host = host.and(streamed.close());
            }
        }
        weights.and(host)
    }

    /// Install the transfer runtime.
    pub(crate) fn set_transfer(&mut self, transfer: super::transfer::TransferRuntime) {
        self.owners.transfer = Some(transfer);
    }

    /// The primary stream of `device` (none on host devices): imports are read
    /// and published on it.
    pub(crate) fn transfer_stream(
        &self,
        device: crate::ffi::MemoryKind,
    ) -> Result<Option<cuda::Stream>> {
        if !matches!(device, crate::ffi::MemoryKind::Cuda(_)) {
            return Ok(None);
        }
        self.owners
            .transfer
            .as_ref()
            .ok_or_else(|| {
                Error::Unsupported("device transfers require the transfer runtime".into())
            })?
            .stream(device)
    }

    /// Disjoint borrows for executing one batch.
    pub(crate) fn parts(&mut self) -> Parts<'_> {
        Parts {
            placement: &self.placement,
            rank: self.rank,
            requests: &mut self.requests,
            storage: self.owners.worker_storage.as_mut(),
            computations: self.owners.computations.as_mut(),
            transfer: self.owners.transfer.as_mut(),
        }
    }

    pub(crate) fn set_profiler(&mut self, profiler: Option<Profiler>) {
        self.profiler = profiler;
    }

    /// Start the profiling session `start` on this rank, rank `rank` of
    /// CUDA device `device`, while it runs no batch; answers the daemon.
    /// A session that does not select the rank leaves it unrecorded.
    ///
    /// # Errors
    ///
    /// Fails when another session records the rank or the rank cannot
    /// create its file.
    pub(crate) fn start_session(
        &mut self,
        start: proto::profile::SessionStart,
        rank: u32,
        device: Option<cuda::DeviceId>,
    ) -> Result<serde_json::Value> {
        if let Some(session) = &self.session {
            return Err(Error::Invalid(format!(
                "profiling session {} records the rank",
                session.id()
            )));
        }
        let (session, answer) = Session::start(start, rank, device)?;
        self.session = session;
        Ok(answer)
    }

    /// End profiling session `session` for `reason`, while the rank runs no
    /// batch; answers what its file holds. A rank the session did not
    /// select answers so.
    pub(crate) fn stop_session(
        &mut self,
        session: u64,
        reason: &str,
        rank: u32,
    ) -> serde_json::Value {
        match self.session.take() {
            Some(running) if running.id() == session => {
                let mut answer = running.stop(reason);
                answer["rank"] = serde_json::json!(rank);
                answer
            }
            other => {
                self.session = other;
                serde_json::json!({ "rank": rank, "session": session, "selected": false })
            }
        }
    }

    /// Turn `collector` of profiling session `session` off for `reason`.
    pub(crate) fn release_collector(
        &mut self,
        session: u64,
        collector: proto::profile::Collector,
        reason: &str,
        rank: u32,
    ) -> serde_json::Value {
        let mut answer = match &mut self.session {
            Some(running) if running.id() == session => running.release(collector, reason),
            _ => serde_json::json!({ "released": false }),
        };
        answer["rank"] = serde_json::json!(rank);
        answer
    }

    /// Report kernel selections that change while serving.
    pub(crate) fn watch_kernels(&mut self, watch: KernelWatch) {
        self.kernels = Some(watch);
    }

    /// Install every Start, then follow each call's predecessor.
    fn admit_requests(&mut self, batch: &mut WorkerBatch) -> Result<()> {
        let mut installed = Vec::new();
        let started = (|| -> Result<()> {
            for command in &batch.plan.commands {
                if let BatchCommand::Start { request } = command
                    && self.requests.start(request.as_ref().clone())?.is_some()
                {
                    installed.push(request);
                }
            }
            Ok(())
        })();
        started?;
        // The executor admits the started requests in a job before the
        // batch opens (`admit_inputs`); one it does not serve is refused
        // alone, and its calls report the refusal.
        if let Some(computations) = &mut self.owners.computations {
            for request in installed {
                computations.start(request)?;
            }
        }

        batch.predecessors = batch
            .plan
            .calls
            .iter()
            .map(|call| {
                self.requests
                    .predecessor(call.request_key, call.advances_state())
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// Refuse a batch outside this rank's advertised capacity, with the
    /// checks no other owner makes. Buffer allocations, block tables and new
    /// cache units are refused by their owners (`BufferPool::bind`,
    /// `BlockTables::prepare`, `UnitColumn::recycle`).
    fn validate(&self, plan: &BatchPlan) -> Result<()> {
        let unsupported: BTreeSet<&str> = plan
            .calls
            .iter()
            .filter(|call| !self.info.supported_calls.contains(&call.kind))
            .map(|call| call.kind.as_str())
            .collect();
        if !unsupported.is_empty() {
            return Err(Error::Invalid(format!(
                "execution batch contains call kinds unsupported by this worker: {:?}",
                unsupported.into_iter().collect::<Vec<_>>()
            )));
        }
        if !self.placement.components.is_empty() {
            for call in &plan.calls {
                if !self.placement.holds(self.rank, &call.component) {
                    return Err(Error::Invalid(format!(
                        "call targets component {:?} outside this rank",
                        call.component
                    )));
                }
            }
        }
        if plan.calls.len() > self.info.max_batch_calls as usize {
            return Err(Error::Invalid(
                "execution batch exceeds the worker_config call limit".into(),
            ));
        }
        if plan
            .forward
            .request_pool_indices
            .iter()
            .any(|&slot| slot > self.info.request_slots)
        {
            return Err(Error::Invalid(
                "execution batch exceeds request-slot capacity".into(),
            ));
        }
        Ok(())
    }

    /// Install (or clear) the service's completion wake on every storage
    /// owner.
    pub(crate) fn set_storage_wake(&self, wake: Option<ipc::Wake>) {
        self.jobs.set_wake(wake.clone());
        if let Some(storage) = &self.owners.worker_storage {
            storage.set_wake(wake.clone());
        }
        if let Some(computations) = &self.owners.computations {
            computations.set_wake(wake);
        }
    }

    /// Close every storage owner, newest first (the transfer runtime, then
    /// rank storage), after the executor drained and
    /// [`Self::close_executor`] closed what computes; returns the first
    /// failure.
    pub(crate) fn close_storage(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        if let Some(mut transfer) = self.owners.transfer.take() {
            result = result.and(transfer.close(aborted));
        }
        if let Some(mut storage) = self.owners.worker_storage.take() {
            result = result.and(storage.close(aborted));
        }
        result
    }

    /// Close what computes after the jobs drained: the executor, which releases
    /// its computations and the expert union, then the computations' output
    /// storage, image host pool and report pipe; attempts every one before
    /// returning the first failure.
    pub(crate) fn close_executor(&mut self, aborted: bool) -> Result<()> {
        let executor = self
            .owners
            .executor
            .take()
            .map_or(Ok(()), |executor| executor.close(aborted));
        let computations = self
            .owners
            .computations
            .take()
            .map_or(Ok(()), |mut computations| computations.close(aborted));
        executor.and(computations)
    }

    /// Release the outputs of the predecessors this batch's calls follow.
    ///
    /// Before inputs are acquired, a predecessor no call reads can retire
    /// early; once the batch executed, the predecessors it read retire too.
    /// The admission root (batch 0) has no outputs.
    fn release_predecessors(&mut self, batch: &WorkerBatch, inputs_acquired: bool) -> Result<()> {
        let Some(storage) = self.owners.worker_storage.as_mut() else {
            return Ok(());
        };
        let mut consumed = HashSet::new();
        for call in &batch.plan.calls {
            consumed.extend(
                call.tensor_inputs()
                    .chain(call.predicate.iter())
                    .map(|input| (input.request_key, input.producer_call_id)),
            );
            if let Some(input) = call.cache_input {
                consumed.insert((input.owner, input.producer_call_id));
            }
        }
        let releases: Vec<_> = batch
            .plan
            .calls
            .iter()
            .zip(&batch.predecessors)
            .filter_map(|(call, previous)| {
                let previous = previous.as_ref()?;
                let source = (call.request_key, *previous);
                (previous.batch_id > 0 && consumed.contains(&source) == inputs_acquired)
                    .then_some(source)
            })
            .collect();
        storage.release_calls(&releases)?;
        if inputs_acquired {
            // Predicates outside the request's predecessor were consumed
            // here as well; releasing that other call would be too broad.
            let predicates: HashSet<_> = batch
                .plan
                .calls
                .iter()
                .zip(&batch.predecessors)
                .filter_map(|(call, previous)| {
                    call.predicate
                        .as_ref()
                        .filter(|value| Some(value.producer_call_id) != *previous)
                        .map(|value| value.buffer_id())
                })
                .collect();
            if !predicates.is_empty() {
                storage.release_buffers(&predicates, &self.lifecycle)?;
            }
        }
        self.perform_lifecycle()
    }

    /// Release request references after the executor drained its work.
    pub(crate) fn close(&mut self) {
        self.requests.close();
        if let Some(profiler) = &mut self.profiler {
            profiler.close();
        }
    }
}

/// What the rank service asks of the backend beyond the executor's steps.
impl WorkerBackend {
    pub(crate) fn response_error(
        &self,
        kind: RequestKind,
        error: WorkerError,
    ) -> std::result::Result<WorkerResponseError, WorkerError> {
        let calls: Vec<String> = error
            .calls()
            .iter()
            .map(|call| format!("{}/{:?}", call.request_key.request_id.0, call.call_id))
            .collect();
        if crate::error::traces(error.code) {
            tracing::error!(
                "worker request {:?} failed: {error} [code={} calls={calls:?}]{}",
                kind.as_str(),
                error.code,
                error
                    .traceback()
                    .map(|traceback| format!("\n{traceback}"))
                    .unwrap_or_default(),
            );
        } else {
            tracing::warn!(
                "worker request {:?} failed: {error} [code={} calls={calls:?}]",
                kind.as_str(),
                error.code,
            );
        }
        Ok(error.response())
    }

    pub(crate) fn has_open_requests(&self) -> std::result::Result<bool, WorkerError> {
        Ok(self.requests.has_open_requests()?)
    }

    pub(crate) fn awaiting_acknowledgment(&self) -> std::result::Result<bool, WorkerError> {
        Ok(self
            .owners
            .worker_storage
            .as_ref()
            .is_some_and(WorkerMemory::awaiting_acknowledgment))
    }

    /// Whether a join of the union's step is in flight
    /// ([`Self::join_expert_step`]).
    pub(crate) fn joining(&self) -> bool {
        self.joining.is_some()
    }

    pub(crate) fn join_expert_step(
        &mut self,
        leaving: bool,
    ) -> std::result::Result<Option<(bool, bool)>, WorkerError> {
        if let Some(joined) = self.joined.take() {
            return joined.map(Some);
        }
        if self.joining.is_some() {
            return Ok(None);
        }
        let executor = self
            .owners
            .executor
            .as_ref()
            .filter(|_| self.owners.experts.is_some())
            .ok_or_else(|| {
                WorkerError::classify(
                    Error::State("expert-step participation requires an expert union"),
                    None,
                )
            })?;

        // A join is a job like the union's steps: the executor agrees on the
        // union's next step and, when one was agreed, joins it on the stream
        // the steps run on.
        let job = Job::new();
        job.call_returning(executor.join_expert_step(leaving));
        let submitted = self.jobs.start(job, true);
        self.joining = Some(submitted);
        Ok(None)
    }

    pub(crate) fn oldest_job(&self) -> Option<(JobId, std::time::Instant)> {
        self.jobs.oldest()
    }

    /// Wait at most `timeout` for the next completion of a job in flight,
    /// kept for the next [`Executor::advance`]; whether one arrived.
    pub(crate) fn await_completion(&self, timeout: std::time::Duration) -> Result<bool> {
        self.jobs.await_completion(timeout)
    }

    pub(crate) fn stall(
        &self,
        job: JobId,
        running: std::time::Duration,
        batch: Option<(u64, &WorkerBatch)>,
    ) -> Stall {
        let method = self.jobs.performing(job).flatten();
        Stall {
            batch_id: batch.map(|(id, _)| id),
            route: batch
                .and_then(|(_, batch)| batch.route())
                .map(|kind| kind.as_str().to_owned()),
            method,
            calls: batch
                .map(|(_, batch)| batch.identities())
                .unwrap_or_default(),
            running_ms: u64::try_from(running.as_millis()).unwrap_or(u64::MAX),
        }
    }
}
