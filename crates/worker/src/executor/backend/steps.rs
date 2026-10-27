//! The steps of a batch's life on a rank: the submitted batch with its
//! lifecycle and numerical state ([`WorkerBatch`]), and what the backend does
//! at each step the [`crate::executor::Executor`] orders: admission, input
//! preparation, the jobs of its forward and their completions, output
//! readiness, retirement and result delivery.

use std::collections::BTreeSet;
use std::sync::Arc;

use ipc::CallId;
use ipc::{Batch as BatchPlan, BatchCommand, BatchOutput, CallKind, ErrorCallIdentity};

use super::WorkerBackend;
use crate::error::WorkerError;
use crate::executor::batch::{self, Launched};
use crate::executor::image_inputs::InlineImages;
use crate::executor::lifecycle::Retirement;
use crate::executor::python::PythonExecutor;
use crate::executor::{BatchResult, Submission};
use crate::job::{Completion, Job, JobId};
use crate::{Error, Result};

/// Where a batch's host KV cache copies stand on the copy stream
/// (`PrefixCache::issue_pages`).
#[derive(Default)]
enum Pages {
    /// Not recorded yet.
    #[default]
    Unissued,
    /// Submitted as a job the builder thread has not performed yet.
    Issuing(PageFences),
    /// Performed: the fences' records are made.
    Issued(PageFences),
}

/// The fences of a batch's page copies, recorded on the copy stream of the
/// cache's `device`, and of its trajectory offloads, recorded on the copy
/// stream of the latent banks' device. A batch without copies has none.
#[derive(Default)]
struct PageFences {
    device: Option<cuda::DeviceId>,
    /// Behind the loads: the batch's forward waits on it.
    loads: Option<Arc<cuda::Event>>,
    /// Behind the offloads: the batch's output waits on it.
    offloads: Option<Arc<cuda::Event>>,
    /// Behind the trajectory offloads, with the banks' device: the batch's
    /// output waits on it.
    latents: Option<(cuda::DeviceId, Arc<cuda::Event>)>,
}

/// A submitted batch with its lifecycle and numerical state.
pub(crate) struct WorkerBatch {
    pub(super) plan: Arc<BatchPlan>,
    /// The accepted state call each call follows, by call.
    pub(super) predecessors: Vec<Option<CallId>>,
    retirement: Retirement,
    /// Where the batch's forward stands between its jobs.
    stage: batch::Stage,
    /// An executor computation's batch between its jobs.
    computation: Option<crate::executor::computations::Pending>,
    /// A transfer batch between its jobs.
    transfer: Option<crate::executor::transfer::Pending>,
    /// The batch's profiler step is open: its forward began and has not
    /// ended.
    pub(super) stepping: bool,
    /// The job of the profiler's window this batch's forward waits for.
    profiling: Option<JobId>,
    /// The computation's or the transfer runtime's launched work awaiting
    /// materialization.
    running: Option<Launched>,
    /// The host preparation of an image encoding batch's inline images.
    image_inputs: Option<InlineImages>,
    /// The executor admits the requests the batch started: the job that
    /// asks it is in flight.
    starting: bool,
    /// Products and cache transfers imported from other calls.
    inputs: crate::executor::inputs::Inputs,
    /// The batch's host KV cache copies.
    pages: Pages,
    output: Option<BatchResult>,
}

impl WorkerBatch {
    /// Attribute a failure to the batch's calls and its route.
    pub(crate) fn attribute(&self, error: WorkerError) -> WorkerError {
        let error = error.with_calls(self.identities());
        match self.route() {
            Some(kind) => error.with_route(kind.as_str()),
            None => error,
        }
    }

    pub(crate) fn new(plan: Arc<BatchPlan>) -> Self {
        let retirement = Retirement::new(&plan);
        Self {
            plan,
            predecessors: Vec::new(),
            retirement,
            stage: batch::Stage::default(),
            computation: None,
            transfer: None,
            stepping: false,
            profiling: None,
            running: None,
            image_inputs: None,
            starting: false,
            inputs: crate::executor::inputs::Inputs::default(),
            pages: Pages::default(),
            output: None,
        }
    }

    /// The fence of the batch's host KV cache reads once their job issued them
    /// on the copy stream; the batch's forward waits on it.
    pub(crate) fn page_loads(&self) -> Option<&Arc<cuda::Event>> {
        match &self.pages {
            Pages::Issued(fences) => fences.loads.as_ref(),
            Pages::Unissued | Pages::Issuing(_) => None,
        }
    }

    pub(crate) fn shared_plan(&self) -> &Arc<BatchPlan> {
        &self.plan
    }

    /// The batch's call kind; `None` for a lifecycle-only batch.
    pub(crate) fn route(&self) -> Option<CallKind> {
        self.plan.calls.first().map(|call| call.kind)
    }

    pub(crate) fn predecessors(&self) -> &[Option<CallId>] {
        &self.predecessors
    }

    /// The calls a failure of this batch affects.
    pub(super) fn identities(&self) -> Vec<ErrorCallIdentity> {
        self.plan
            .calls
            .iter()
            .map(|call| ErrorCallIdentity {
                request_key: call.request_key,
                call_id: call.call_id,
            })
            .collect()
    }

    pub(crate) fn stage(&self) -> batch::Stage {
        self.stage
    }

    pub(crate) fn set_stage(&mut self, stage: batch::Stage) {
        self.stage = stage;
    }

    /// The batch's plan, its imported inputs and its computation batch between
    /// jobs, borrowed apart.
    pub(crate) fn computation_parts(
        &mut self,
    ) -> (
        &BatchPlan,
        &crate::executor::inputs::Inputs,
        &mut Option<crate::executor::computations::Pending>,
    ) {
        (&self.plan, &self.inputs, &mut self.computation)
    }

    /// The batch's plan, its imported inputs, its host-prepared inline images
    /// and its computation batch between jobs, borrowed apart.
    pub(crate) fn computation_inputs(
        &mut self,
    ) -> (
        &BatchPlan,
        &crate::executor::inputs::Inputs,
        Option<&InlineImages>,
        &mut Option<crate::executor::computations::Pending>,
    ) {
        (
            &self.plan,
            &self.inputs,
            self.image_inputs.as_ref(),
            &mut self.computation,
        )
    }

    /// The batch's plan, its imported inputs and its transfer batch between
    /// jobs, borrowed apart.
    pub(crate) fn transfer_parts(
        &mut self,
    ) -> (
        &BatchPlan,
        &crate::executor::inputs::Inputs,
        &mut Option<crate::executor::transfer::Pending>,
    ) {
        (&self.plan, &self.inputs, &mut self.transfer)
    }

    /// Set the batch's result once its outputs are materialized, or its
    /// calls failed.
    pub(crate) fn complete(&mut self, result: BatchResult) {
        self.output = Some(result);
    }

    /// Hand launched work to materialization by [`WorkerBackend::poll`].
    pub(crate) fn launch(&mut self, running: Launched) {
        self.running = Some(running);
    }

    pub(crate) fn image_inputs(&self) -> Option<&InlineImages> {
        self.image_inputs.as_ref()
    }

    pub(crate) fn set_image_inputs(&mut self, inputs: InlineImages) {
        self.image_inputs = Some(inputs);
    }

    /// The batch's imported inputs.
    pub(crate) fn inputs(&self) -> &crate::executor::inputs::Inputs {
        &self.inputs
    }

    pub(crate) fn inputs_mut(&mut self) -> &mut crate::executor::inputs::Inputs {
        &mut self.inputs
    }
}

/// The batch work of the executor: what each step of a batch's life does
/// on this rank ([`crate::executor::Executor`] orders the steps).
impl WorkerBackend {
    pub(crate) fn admit(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(), WorkerError> {
        // Explicit revocation must not wait behind an allocation-dependent
        // batch; existing readers retain their leases.
        batch.retirement.admit(&mut self.owners, &self.lifecycle)?;
        Ok(self.perform_lifecycle()?)
    }

    pub(crate) fn prepare(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(), WorkerError> {
        self.validate_kinds(batch)?;
        self.admit_requests(batch)?;
        self.validate(&batch.plan)?;
        for command in &batch.plan.commands {
            if let BatchCommand::Finish { request_key, .. } = command {
                self.requests.finish(*request_key)?;
            }
        }
        self.release_predecessors(batch, false)?;
        Ok(())
    }

    pub(crate) fn admit_inputs(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<Option<JobId>, WorkerError> {
        // The batch's host KV cache copies are issued first, in a job of their
        // own on the copy stream, so they overlap on the device the forward the
        // previous batch runs; the batch goes on once that job issued them.
        if matches!(batch.pages, Pages::Unissued)
            && let Some(job) = self.issue_pages(batch)?
        {
            return Ok(Some(job));
        }

        // The executor admits the requests this batch started: it sizes each,
        // resets its slot state and may refuse it. The batch goes on from the
        // job's completion, and so do the requests' later calls, which start
        // behind this batch.
        if let Some(computations) = &mut self.owners.computations
            && let Some(job) = computations.take_started()?
        {
            let submitted = self.jobs.start(job, true);
            batch.starting = true;
            return Ok(Some(submitted));
        }

        // Products other calls published are imported for every batch alike.
        // What the imports recorded is performed even when an import failed
        // after it: the fences are due either way.
        let (job, admitted) = crate::executor::inputs::admit(self, batch);
        let submitted = match batch.route() {
            Some(_) if !job.is_empty() => Some(self.start(job, true).map_err(job_lost)?),
            _ => None,
        };
        admitted.map(|()| submitted)
    }

    /// Record the batch's host KV cache copies and trajectory offloads on
    /// the copy streams and submit them; `None` when the batch copies no
    /// page.
    fn issue_pages(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<Option<JobId>, WorkerError> {
        let plan = &batch.plan;
        if plan.page_loads.is_empty()
            && plan.page_offloads.is_empty()
            && plan.latent_offloads.is_empty()
        {
            batch.pages = Pages::Issued(PageFences::default());
            return Ok(None);
        }
        let mut fences = PageFences::default();
        let job = if plan.page_loads.is_empty() && plan.page_offloads.is_empty() {
            Job::new()
        } else {
            let prefix_cache = self
                .owners
                .worker_storage
                .as_mut()
                .and_then(|storage| storage.prefix_cache.as_mut())
                .ok_or_else(|| {
                    Error::Invalid("a batch copies pages but the rank has no prefix cache".into())
                })?;
            let issued = prefix_cache.issue_pages(&plan.page_loads, &plan.page_offloads)?;
            fences.device = Some(issued.device);
            fences.loads = issued.loads;
            fences.offloads = issued.offloads;
            issued.job
        };
        if !plan.latent_offloads.is_empty() {
            // Each trajectory is its request's slot on this rank.
            let offloads = plan
                .latent_offloads
                .iter()
                .map(|offload| {
                    let request = self
                        .requests
                        .peek(offload.request_key.request_id.0)
                        .filter(|request| request.key() == offload.request_key)
                        .ok_or_else(|| {
                            Error::Invalid(
                                "a batch offloads the trajectory of a request the rank does not \
                                 hold"
                                    .into(),
                            )
                        })?;
                    Ok((request.slot(), offload))
                })
                .collect::<Result<Vec<_>>>()?;
            let ledger = self
                .owners
                .computations
                .as_mut()
                .and_then(|computations| computations.ledger_mut())
                .ok_or_else(|| {
                    Error::Invalid("a batch offloads a trajectory but the rank has none".into())
                })?;
            fences.latents = Some(ledger.offload(&offloads, &job)?);
        }
        let job = self.jobs.start(job, true);
        batch.pages = Pages::Issuing(fences);
        Ok(Some(job))
    }

    pub(crate) fn inputs_admitted(
        &mut self,
        batch: &mut WorkerBatch,
        completion: Completion,
    ) -> std::result::Result<(), WorkerError> {
        // The job issued the batch's page copies: their fences are recorded,
        // whether the copies were issued or failed.
        if let Pages::Issuing(fences) = std::mem::take(&mut batch.pages) {
            batch.pages = Pages::Issued(fences);
            completion.check().map_err(|failure| {
                WorkerError::classify(failure.error, Some("host KV cache copies"))
            })?;
            return Ok(());
        }

        if !std::mem::take(&mut batch.starting) {
            return crate::executor::inputs::admitted(self, batch, completion);
        }

        // The job asked the executor to admit the requests the batch
        // started; those whose size the denoiser or the plan does not serve
        // are refused alone.
        let refused = completion
            .check()
            .map_err(|failure| failure.error)
            .and_then(|mut completion| completion.texts())
            .and_then(|texts| {
                serde_json::from_str::<Vec<(ipc::RequestKey, String)>>(&texts.concat())
                    .map_err(|error| Error::Interface(format!("the executor's refusals: {error}")))
            })
            .map_err(|error| WorkerError::classify(error, Some("request admission")))?;
        for (key, reason) in refused {
            tracing::info!(request = ?key, "refused request: {reason}");
            self.requests.refuse(key, &reason)?;
        }
        Ok(())
    }

    pub(crate) fn prepare_inputs(
        &mut self,
        batch: &mut WorkerBatch,
        submission: &Arc<Submission>,
    ) -> std::result::Result<bool, WorkerError> {
        // The cache transfers other calls published are imported behind the
        // admitted products (`admit_inputs`), and the batch waits for both
        // (`await_inputs` subscribes to what blocks them).
        if !crate::executor::inputs::prepare(self, batch)? {
            return Ok(false);
        }
        // Inline images of an image encoding batch are decoded on the image
        // host pool while other batches execute; the batch launches once they
        // are ready (the pool notifies `submission`). Other batches' inputs are
        // prepared when their computation executes.
        crate::executor::image_inputs::prepare_inputs(self, batch, submission)
    }

    pub(crate) fn await_inputs(
        &mut self,
        batch: &mut WorkerBatch,
        submission: &Arc<Submission>,
    ) -> std::result::Result<(), WorkerError> {
        Ok(crate::executor::inputs::subscribe(self, batch, submission)?)
    }

    pub(crate) fn forward(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<Option<JobId>, WorkerError> {
        let Some(kind) = batch.route() else {
            if batch.output.is_none() {
                batch.output = Some(BatchResult {
                    output: BatchOutput {
                        batch_id: batch.plan.batch_id,
                        completions: Vec::new(),
                        products: Vec::new(),
                        worker_exec_us: None,
                        forward_stats: None,
                    },
                    media: Vec::new(),
                });
            }
            return Ok(None);
        };

        let queue = self.jobs.clone();
        loop {
            let stage = batch.stage();
            match stage {
                // Imported values become resident before the computation
                // consumes them like local ones; an import that failed fails
                // the batch's calls.
                batch::Stage::Inputs => {}
                // Each executed batch with calls is one profiler step.
                batch::Stage::Forward if !batch.stepping => {
                    batch.stepping = true;
                    if let Some(profiler) = &mut self.profiler {
                        profiler.begin();
                    }
                }
                batch::Stage::Forward => {}
                batch::Stage::Done => self.end_forward(batch),
            }

            // The profiler's window opens before the first job of its
            // first step's forward and closes behind the last job of its
            // last step's: each of its jobs is one the batch waits for.
            if stage != batch::Stage::Inputs
                && let Some(profiler) = &mut self.profiler
                && let Some(job) = profiler.next(&queue)
            {
                batch.profiling = Some(job);
                return Ok(Some(job));
            }
            if stage == batch::Stage::Done {
                self.release_predecessors(batch, true)?;
                return Ok(None);
            }

            // A stage that has nothing to issue advances without a job.
            let submitted = match batch::forward(self, batch, kind) {
                Ok(Some(job)) => Some(self.start(job, true).map_err(job_lost)),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            };
            match submitted {
                Some(Ok(job)) => return Ok(Some(job)),
                Some(Err(error)) => {
                    self.end_forward(batch);
                    return Err(error);
                }
                None => {}
            }
        }
    }

    pub(crate) fn complete(
        &mut self,
        batch: &mut WorkerBatch,
        job: JobId,
        completion: Completion,
    ) -> std::result::Result<(), WorkerError> {
        let Some(kind) = batch.route() else {
            return Ok(());
        };
        if batch.profiling == Some(job) {
            batch.profiling = None;
            if let Some(profiler) = &mut self.profiler {
                profiler.complete(completion);
            }
            return Ok(());
        }
        let completed = batch::complete(self, batch, kind, completion);
        if completed.is_err() {
            self.end_forward(batch);
        }
        completed
    }

    pub(crate) fn await_outputs(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<Option<JobId>, WorkerError> {
        // The fences are named by the lease, which holds them whatever the
        // event pool has released since they were recorded.
        let mut fences: Vec<_> = match &batch.running {
            Some(running) => crate::lock(&running.buffer)
                .device_fences()
                .map(|(device, fence)| (device, Arc::clone(fence)))
                .collect(),
            None => Vec::new(),
        };
        // The output reports the batch's page and trajectory offloads
        // complete.
        if let Pages::Issued(PageFences {
            device: Some(device),
            offloads: Some(fence),
            ..
        }) = &batch.pages
        {
            fences.push((*device, Arc::clone(fence)));
        }
        if let Pages::Issued(PageFences {
            latents: Some((device, fence)),
            ..
        }) = &batch.pages
        {
            fences.push((*device, Arc::clone(fence)));
        }
        let Some(&(device, _)) = fences.first() else {
            return Ok(None);
        };
        let fences = fences.into_iter().map(|(_, fence)| fence).collect();

        // The job holds no operation: its completion is posted once the fences
        // completed.
        let job = Job::new();
        job.after(device, fences);
        Ok(Some(self.jobs.start(job, true)))
    }

    pub(crate) fn outputs_ready(
        &mut self,
        batch: &mut WorkerBatch,
        completion: Completion,
    ) -> std::result::Result<(), WorkerError> {
        completion
            .check()
            .map_err(|failure| job_lost(failure.error))?;
        if let Some(running) = &batch.running {
            crate::lock(&running.buffer).fences_completed();
        }
        Ok(())
    }

    pub(crate) fn completed(
        &mut self,
    ) -> std::result::Result<Option<(JobId, Completion)>, WorkerError> {
        while let Some((job, completion)) = self.jobs.take() {
            // What belongs to no batch is applied here: the event pool's
            // deferred work, a lifecycle command's device work, the expert
            // union's step this rank joined, the kernel watch's calls.
            let events = self
                .owners
                .worker_storage
                .as_ref()
                .map(|storage| &storage.events);
            if let Some(events) = events.filter(|events| events.owns(job)) {
                // The device finished what the job's events fence: the
                // storage deferred behind them is released before the
                // executor routes the completions that follow.
                events.complete(job, completion)?;
            } else if self.lifecycle_jobs.remove(&job) {
                // The work a command recorded did not reach its storage:
                // what that storage holds now is unknown.
                completion.check().map_err(|failure| {
                    let mut error = WorkerError::classify(failure.error, Some("lifecycle"));
                    error.fatal = true;
                    error
                })?;
            } else if self.joining == Some(job) {
                self.joining = None;
                // The executor returns whether a step ran and whether every
                // rank has left the union.
                let joined = completion
                    .check()
                    .map_err(|failure| failure.error)
                    .and_then(|mut completion| completion.returned())
                    .and_then(|returned| match returned[..] {
                        [ran, released] => Ok((ran != 0, released != 0)),
                        _ => Err(Error::Interface(
                            "join_expert_step returns whether a step ran and whether the union released"
                                .into(),
                        )),
                    })
                    .map_err(|error| WorkerError::classify(error, Some("expert step")));
                self.joined = Some(joined);
            } else if let Some(watch) = self.kernels.as_mut().filter(|watch| watch.owns(job)) {
                let executor = &self.owners.executor;
                watch.complete(job, completion, &self.jobs, || {
                    executor.as_deref().map(PythonExecutor::kernels_call)
                });
            } else {
                return Ok(Some((job, completion)));
            }
        }
        Ok(None)
    }

    pub(crate) fn begin_retirement(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(), WorkerError> {
        batch
            .retirement
            .begin(&self.requests, &mut self.owners, &self.lifecycle)?;
        Ok(self.perform_lifecycle()?)
    }

    pub(crate) fn poll(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(bool, bool), WorkerError> {
        let mut advanced = false;
        if batch.output.is_none()
            && let Some(running) = batch.running.as_mut()
            && let Some(result) = batch::poll(self, &batch.plan, running)?
        {
            batch.output = Some(result);
            batch.running = None;
            advanced = true;
        }
        if batch.output.is_none() {
            return Ok((advanced, false));
        }
        let retired =
            batch
                .retirement
                .poll(&mut self.requests, &mut self.owners, &self.lifecycle)?;
        self.perform_lifecycle()?;
        Ok((advanced || retired, retired))
    }

    /// Move the completed batch's result out for delivery: the executor
    /// asks once, as it removes the batch.
    pub(crate) fn result(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<BatchResult, WorkerError> {
        let mut result = batch
            .output
            .take()
            .ok_or_else(|| WorkerError::from(Error::State("batch output is not resolved")))?;
        // The streams' device time and the serving threads' driver calls that
        // waited for the device, since the last result, travel with this one;
        // the engine adds every result's into its counters. This runs on the
        // control thread, whose own calls are counted here.
        let control = cuda::cupti::Calls::current();
        crate::profiling::blocking::add("control", &control.since(&self.control_calls));
        self.control_calls = control;
        let activity = cuda::cupti::activity::take();
        if activity.dropped_records > 0 {
            tracing::warn!(
                "CUPTI dropped {} activity records; the streams' device time misses their operations",
                activity.dropped_records
            );
        }
        let stream_us: Vec<(String, u64)> = activity
            .streams
            .iter()
            .flat_map(|stream| {
                [
                    ("busy", stream.busy_us),
                    ("gap", stream.gap_us),
                    ("late", stream.late_us),
                ]
                .map(|(state, us)| (format!("{}/{state}", stream.name), us))
            })
            .filter(|&(_, us)| us > 0)
            .collect();
        let blocking = crate::profiling::blocking::take();
        if !stream_us.is_empty() || !blocking.is_empty() {
            let stats = result
                .output
                .forward_stats
                .get_or_insert_with(Default::default);
            for (key, us) in stream_us {
                *stats.stream_us.entry(key).or_default() += us;
            }
            for (key, count) in blocking {
                *stats.blocking_calls.entry(key).or_default() += count;
            }
        }
        Ok(result)
    }

    pub(crate) fn close_batch(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(), WorkerError> {
        // Calls become pending only when their batch commits; unresolved
        // launched work cancels its pending calls. Lifecycle commands hold
        // no resource beyond their retirement.
        let abandoned = batch
            .running
            .take()
            .map_or(Ok(()), |running| batch::abandon(self, running));
        // Imports the batch did not consume are abandoned; their storage
        // retires with their physical accesses.
        let closed = crate::executor::inputs::close(self, batch).map_err(WorkerError::from);
        abandoned.and(closed)
    }

    pub(crate) fn reap(&mut self) -> std::result::Result<(), WorkerError> {
        for owner in self.owners.request_storage() {
            owner.reap()?;
        }
        Ok(())
    }
}

impl WorkerBackend {
    fn validate_kinds(&self, batch: &WorkerBatch) -> Result<()> {
        let kinds: BTreeSet<CallKind> = batch.plan.calls.iter().map(|call| call.kind).collect();
        if kinds.len() > 1 {
            return Err(Error::Invalid(
                "an execution batch carries calls of one kind".into(),
            ));
        }
        Ok(())
    }
}

/// A job that cannot be submitted, or whose completion cannot be ordered behind
/// its events, leaves the rank unable to serve.
pub(super) fn job_lost(error: Error) -> WorkerError {
    let mut error = WorkerError::classify(error, Some("job"));
    error.fatal = true;
    error
}
