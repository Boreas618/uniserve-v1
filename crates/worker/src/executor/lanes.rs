//! The executor's lanes and the batches they serve.
//!
//! An executor lane is one computation the executor serves under the
//! lane's index: a component's call kinds, which compute on one of the
//! rank's streams (the device's lane stream, a stream of the lane's own, or
//! the host). The executor's manifest declares each lane: whether its
//! batches are prepared (the executor states first what each call reads
//! and writes), whether its batches commit (the executor advances its slot
//! state past the batch's commit point), and the resources the worker keeps
//! for it: the prefix cache's [`pages`](crate::executor::pages), a latent
//! trajectory ledger ([`latents`](crate::executor::latents)), the pipe
//! its host tasks report through ([`reports`](crate::executor::reports)),
//! and the image host lane an image model's lanes prepare inline images and
//! encode decoded ones on ([`images`](crate::executor::image_inputs)). The
//! products, pixels, trajectories and guidance prefixes of an image model's
//! lanes are bound and published by [`crate::executor::image`].
//!
//! Every lane's batch goes through one sequence of jobs on the batch
//! skeleton ([`crate::executor::batch`]):
//!
//! 1. **Opening.** Bind the calls to their requests and read their
//!    host-gated predicates (a token-denoising step's continuation is never
//!    read on the host: its slot state decides on the device). A lane over
//!    the prefix cache orders first after the batches whose writes it reads
//!    on another stream and after the host KV cache reads the copy lane
//!    issued for the batch. A batch that reads predicates ends its job here.
//! 2. **Preparation**, for a lane that prepares: a job asks the executor
//!    what each call reads and writes (`prepare`).
//! 3. **Reservation**, in the batch's last job: lease the output storage
//!    and reserve what the batch uses of the worker's resources: the
//!    prefix cache's tables, units, extents and attention pages and the
//!    relays (`PageReservation`), or the products the preparation named and
//!    the trajectories' banks (`crate::executor::products::bind_prepared`).
//! 4. **Execution.** `execute` computes the batch on what was reserved. A
//!    lane that does not prepare reads host columns and writes its results
//!    into output words the batch lent it, with completion records a host
//!    check on the lane's thread parses before anything else follows, so a
//!    malformed record fails the batch before its commit point; the relayed
//!    values it staged are copied into their products. A lane that prepares
//!    writes the products the preparation named, whose fences and exports
//!    follow.
//! 5. **Seal.** End the read leases, publish the completion flags the host
//!    sets, validate every owner's update and seal the lease: the commit
//!    point. Past it, a lane that commits has the executor advance its slot
//!    state (`commit`), and a committing lane over the prefix cache records
//!    its place in the stream order.
//! 6. **Completion.** The job's results become the calls' pending outputs:
//!    the records' progress and ranges, or each call's graph mode, the
//!    exports and trajectory commits; the products become visible and the
//!    calls pending. The launched batch materializes once the lease's fences
//!    completed and, for a lane whose calls run host tasks, every task
//!    reported.
//!
//! A failure before the commit point retires everything the batch reserved
//! in a job behind what it recorded, and the batch's calls fail; a failure
//! at or past it is fatal: slot state, a trajectory or a product may have
//! become visible.
//!
//! Requests the rank starts reach the executor (`start`) in a job before
//! the batch that started them opens, and the executor may refuse them
//! alone ([`ExecutorLanes::take_started`]).

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use cuda::{DeviceId, Stream};
use ipc::{
    Batch as BatchPlan, BufferId, CallKind, ForwardMode, ForwardStats, RequestKey, WorkerInfo,
};

use crate::config::placement::GroupPlacement;
use crate::cuda_graph::GraphMode;
use crate::error::WorkerError;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::batch::{self, Batch, Failure, HostStage, Launched, Stage, fail, unserved};
use crate::executor::columns::column_step;
use crate::executor::image::{self, ImageBatch, ImageCall};
use crate::executor::image_inputs::{Images, InlineImages};
use crate::executor::inputs::Inputs;
use crate::executor::latents::LatentLedger;
use crate::executor::lifecycle::{LaneJobs, RequestStorage};
use crate::executor::pages::{
    PageCall, PageOwners, PageReservation, Pages, RelayStaging, denoising, relay,
};
use crate::executor::products::{
    PreparedOwners, ProductCall, ProductOwners, Publications, bind_prepared, prepared_step, publish,
};
use crate::executor::python::PythonExecutor;
use crate::executor::records::{COMPLETION_FIELDS, Records};
use crate::executor::reports::{HostCall, HostTasks, Reports, release_borrowed};
use crate::lane::{Completion, Job};
use crate::memory::OutputPool;
use crate::memory::allocation::StorageView;
use crate::request::RequestPool;
use crate::stream::{CUDAStream, Issue as _};
use crate::worker::WorkerMemory;
use crate::{Error, Result, lock};

/// The step a lane batch's failure is reported in.
const PHASE: &str = "executor batch";

/// Bytes of the engine's serialized logprob payload, which bounds a call's
/// completion (`max_completion_bytes`), per output word the executor
/// writes it in: a ranked entry's 12 bytes (token, logprob, rank) take
/// three words, a row's 4 bytes one.
const PAYLOAD_BYTES_PER_WORD: u64 = 4;

/// The worker's resources a lane uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Resources {
    /// Its calls attend to the prefix cache.
    pub(crate) pages: bool,
    /// Its calls step latent trajectories in the rank's ledger.
    pub(crate) latents: bool,
    /// Its calls run host tasks that report through the rank's pipe.
    pub(crate) reports: bool,
    /// Its calls read inline images or encode decoded ones on the rank's
    /// image host lane.
    pub(crate) images: bool,
}

/// One executor lane.
pub(crate) struct ExecutorLane {
    pub(crate) component: String,
    pub(crate) kinds: Vec<CallKind>,
    /// The executor states what each call of a batch reads and writes
    /// before the batch reserves (`prepare`).
    pub(crate) prepares: bool,
    /// The executor advances the lane's slot state past each batch's
    /// commit point (`commit`).
    pub(crate) commits: bool,
    pub(crate) resources: Resources,
    /// The device and stream the lane computes on; neither on the host.
    pub(crate) device: Option<DeviceId>,
    pub(crate) stream: Option<Stream>,
    /// The output leases of the lane's batches.
    pub(crate) outputs: OutputPool,
    /// The relay staging of a lane over the prefix cache.
    pub(crate) staging: Option<RelayStaging>,
}

/// The rank's executor lanes and the resources the worker keeps for them.
pub(crate) struct ExecutorLanes {
    executor: Arc<PythonExecutor>,
    lanes: Vec<ExecutorLane>,
    pages: Option<Pages>,
    ledger: Option<LatentLedger>,
    reports: Option<Reports>,
    images: Option<Images>,
    /// On a CUDA rank whose lanes run host tasks, the stream device
    /// products host tasks wrote are fenced on: the tasks waited for their
    /// copies, so the fences complete at once.
    host_stream: Option<Arc<CUDAStream>>,
    /// Started requests the executor has not admitted yet, as JSON.
    started: Vec<serde_json::Value>,
}

/// The resources of a rank's lanes, as startup allocated them.
pub(crate) struct LaneResources {
    pub(crate) pages: Option<Pages>,
    pub(crate) ledger: Option<LatentLedger>,
    pub(crate) reports: Option<Reports>,
    pub(crate) images: Option<Images>,
    pub(crate) host_stream: Option<Arc<CUDAStream>>,
}

impl ExecutorLanes {
    pub(crate) fn new(
        executor: Arc<PythonExecutor>,
        lanes: Vec<ExecutorLane>,
        resources: LaneResources,
    ) -> Self {
        Self {
            executor,
            lanes,
            pages: resources.pages,
            ledger: resources.ledger,
            reports: resources.reports,
            images: resources.images,
            host_stream: resources.host_stream,
            started: Vec::new(),
        }
    }

    /// The rank's image host resource, when a lane names it.
    pub(crate) fn images(&self) -> Option<&Images> {
        self.images.as_ref()
    }

    /// The latent trajectory ledger, for trajectory transfers and imports.
    pub(crate) fn ledger_mut(&mut self) -> Option<&mut LatentLedger> {
        self.ledger.as_mut()
    }

    /// The lane serving `kind` calls of `component`.
    fn route(&self, component: &str, kind: CallKind) -> Option<usize> {
        self.lanes
            .iter()
            .position(|lane| lane.component == component && lane.kinds.contains(&kind))
    }

    /// Whether a lane serves calls of `kind`.
    pub(crate) fn serves(&self, kind: CallKind) -> bool {
        self.lanes.iter().any(|lane| lane.kinds.contains(&kind))
    }

    /// What the lanes over the prefix cache share, for work beside
    /// `component`'s text batches (cache transfers): `component` has a text
    /// prefill lane over the cache.
    pub(crate) fn pages_of(&self, component: &str) -> Result<&Pages> {
        let lane = self
            .route(component, CallKind::Forward(ForwardMode::Prefill))
            .filter(|&lane| self.lanes[lane].resources.pages);
        match (lane, &self.pages) {
            (Some(_), Some(pages)) => Ok(pages),
            _ => Err(Error::Unsupported(format!(
                "component {component:?} has no text prefill lane over the prefix cache"
            ))),
        }
    }

    /// What the lanes over the prefix cache share, for work on other
    /// streams that orders against their batches.
    pub(crate) fn pages_mut(&mut self) -> Option<&mut Pages> {
        self.pages.as_mut()
    }

    /// The lanes' stream over the prefix cache, for their warmup jobs.
    pub(crate) fn pages_stream(&self) -> Option<Stream> {
        self.pages.as_ref().and_then(Pages::stream).cloned()
    }

    /// Whether lane `lane`'s batches write the prefix cache: a text lane
    /// over it (token-denoising lanes read their prefixes only).
    pub(crate) fn writes_pages(&self, lane: usize) -> bool {
        self.lanes.get(lane).is_some_and(|lane| {
            lane.resources.pages
                && !lane
                    .kinds
                    .contains(&CallKind::Forward(ForwardMode::TokenDenoising))
        })
    }

    /// Hold a started request for the executor's admission, which the batch
    /// that started it asks for next ([`Self::take_started`]). A request
    /// with an image's canvas opens its trajectory's raster in the ledger.
    pub(crate) fn start(&mut self, request: &ipc::NewRequest) -> Result<()> {
        if let (Some(ledger), Some(media)) = (&mut self.ledger, &request.diffusion) {
            ledger.admit(
                request.request_pool_idx as usize,
                (media.width, media.height),
            );
        }
        self.started.push(
            serde_json::to_value(request)
                .map_err(|error| Error::Invalid(format!("encoding a started request: {error}")))?,
        );
        Ok(())
    }

    /// The job that admits the started requests no batch asked the
    /// executor to admit yet, whose completion's text is the JSON of those
    /// it refused, and whether it belongs on the lane: the executor resets
    /// the requests' slot state over the prefix cache, ordered with the
    /// batches over it on every stream, or prepares a trajectory's layout on
    /// its lane's stream. `None` without started requests.
    pub(crate) fn take_started(&mut self) -> Result<Option<(Job, bool)>> {
        if self.started.is_empty() {
            return Ok(None);
        }
        let requests = serde_json::Value::Array(std::mem::take(&mut self.started));
        let job = Job::new();
        let stream = self.pages.as_ref().and_then(Pages::stream).cloned();
        let issue = job.on(stream.as_ref());
        if let Some(pages) = &self.pages {
            pages.order().wait(issue)?;
        }
        job.call_texts(self.executor.start(requests.to_string()));
        if let Some(pages) = &mut self.pages {
            pages.order_mut().record(issue)?;
        }
        Ok(Some((job, self.pages.is_some() || self.ledger.is_some())))
    }

    pub(crate) fn set_wake(&self, wake: Option<ipc::Wake>) {
        if let Some(images) = &self.images {
            images.host.set_wake(wake.clone());
        }
        if let Some(reports) = &self.reports {
            reports.set_wake(wake);
        }
    }

    /// Release the lanes' output storage, stop the image host lane and
    /// close the report pipe, after the lanes drained and the executor
    /// closed (its host thread stopped).
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        self.started.clear();
        let images = self
            .images
            .take()
            .map_or(Ok(()), |images| images.close(aborted));
        for lane in &mut self.lanes {
            lane.outputs.begin_close();
            if !aborted {
                lane.outputs.finish_close();
            }
        }
        if let Some(mut reports) = self.reports.take() {
            reports.close();
        }
        images
    }
}

impl RequestStorage for ExecutorLanes {
    fn release_buffers(&mut self, buffers: &HashSet<BufferId>, _jobs: &LaneJobs) -> Result<()> {
        // Released buffers' published trajectory versions retire once their
        // readers did.
        if let Some(ledger) = &self.ledger {
            ledger.release_buffers(buffers);
        }
        Ok(())
    }

    fn release_requests(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
        _jobs: &LaneJobs,
    ) -> Result<()> {
        match &self.ledger {
            Some(ledger) => ledger.release_requests(requests, retained),
            None => Ok(()),
        }
    }

    fn retirement_ready(
        &mut self,
        _buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        Ok(self
            .ledger
            .as_ref()
            .is_none_or(|ledger| ledger.retirement_ready(requests)))
    }

    fn retire_slots(&mut self, slots: &[usize], _jobs: &LaneJobs) -> Result<()> {
        if let Some(ledger) = &mut self.ledger {
            ledger.release_slots(slots)?;
        }
        self.executor.release(slots);
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        for lane in &mut self.lanes {
            lane.outputs.reap();
        }
        if let Some(ledger) = &mut self.ledger {
            ledger.reap();
        }
        Ok(())
    }
}

/// What a call contributes to its batch beside the state every batch keeps
/// of it ([`batch::CallState`]): what it holds of the prefix cache, of the
/// products a preparation named and of an image model's products and
/// trajectories.
#[derive(Default)]
pub(crate) struct LaneCall {
    pub(crate) pages: PageCall,
    pub(crate) products: ProductCall,
    pub(crate) image: ImageCall,
}

/// A batch whose calls are bound, before its output storage is leased.
pub(crate) struct Opening {
    lane: usize,
    calls: batch::Opening<LaneCall>,
}

/// A batch whose last job is on its lane, or that failed after recording
/// it.
pub(crate) struct Parked {
    lane: usize,
    batch: Batch<LaneCall>,
    /// The executor runs the batch.
    executes: bool,
    /// The job's first operation past the commit point.
    point: usize,
    /// The tensors the executor reads and writes for the batch, kept until
    /// its last job completed.
    _held: Vec<StorageView>,
    /// The records the job's check parsed, for a lane that does not
    /// prepare.
    records: Arc<OnceLock<Records>>,
    /// The lease's first word lent to the executor.
    lent: usize,
    /// The exports of a prepared batch, accepted at its completion and
    /// registered at its commit point.
    publications: Publications,
    /// What an image lane's batch holds through its last job.
    image: ImageBatch,
}

/// A lane's batch between two of its jobs.
pub(crate) enum Pending {
    /// The job on the lane reads the batch's host-gated predicates.
    Predicates(Opening),
    /// The predicates are known: the batch is prepared or records its last
    /// job next.
    Opened(Opening),
    /// The job on the lane asks the executor what the batch's calls read
    /// and write.
    Preparing(Opening, serde_json::Value),
    /// The preparation is known (`step["prepared"]`): the batch records its
    /// last job next.
    Prepared(Opening, serde_json::Value),
    /// The job on the lane is the batch's last.
    Committing(Box<Parked>),
    /// The last job failed before its commit point, or its completion could
    /// not be applied; the batch retires what it reserved next.
    Failing(Box<Parked>, Failure),
    /// The job on the lane retires what the failed batch reserved, or
    /// performs what a batch that reserved nothing had recorded.
    Discarding(Failure),
    /// The batch failed and holds nothing: its calls are reported next.
    Failed(Failure),
}

/// The parts of the backend a lane's batch executes with.
pub(crate) struct Parts<'a> {
    pub(crate) placement: &'a GroupPlacement,
    pub(crate) rank: usize,
    pub(crate) info: &'a WorkerInfo,
    pub(crate) requests: &'a mut RequestPool,
    pub(crate) storage: &'a mut WorkerMemory,
    pub(crate) lanes: &'a mut ExecutorLanes,
}

impl Opening {
    /// Bind the batch's calls to their requests and record into `job` the
    /// read of every host-gated predicate; a lane over the prefix cache
    /// first orders after the batches on other streams and after
    /// `page_loads`, the fence of the host KV cache reads the copy lane
    /// issued for the batch.
    fn open(
        parts: &mut Parts<'_>,
        plan: &BatchPlan,
        job: &Job,
        page_loads: Option<&Arc<cuda::Event>>,
    ) -> Result<Self> {
        let first = plan
            .calls
            .first()
            .ok_or_else(|| Error::Invariant("a batch has calls".into()))?;
        if plan
            .calls
            .iter()
            .any(|call| call.component != first.component)
        {
            return Err(Error::Invalid("a batch serves one component".into()));
        }
        let lanes = &*parts.lanes;
        let index = lanes.route(&first.component, first.kind).ok_or_else(|| {
            Error::Unsupported(format!(
                "component {:?} has no executor lane for {}",
                first.component,
                first.kind.as_str()
            ))
        })?;
        let lane = &lanes.lanes[index];
        let issue = job.on(lane.stream.as_ref());
        if let (true, Some(pages)) = (lane.resources.pages, &lanes.pages) {
            pages.order().wait(issue)?;
            // The copy lane made the fence's record before the batch reached
            // its lane: the batch's inputs waited for the job that records
            // it.
            if let Some(fence) = page_loads {
                issue.wait(fence)?;
            }
        }
        // A token-denoising step's continuation stays on the device: reading
        // it here would serialize steps that can run.
        let calls = batch::Opening::open(
            parts.requests,
            &parts.storage.tensors,
            parts.placement,
            parts.rank,
            plan,
            lane.device,
            issue,
            |call| call.canvas.is_none(),
        )?;
        Ok(Self { lane: index, calls })
    }
}

/// The next job of a lane's batch, or `None` when the batch advanced
/// without one: it failed and its calls are reported.
pub(crate) fn forward(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<Option<Job>, WorkerError> {
    let state = match batch.lane_parts().2.take() {
        Some(Pending::Failed(failure)) => return fail(backend, batch, failure).map(|()| None),
        state => state,
    };
    let page_loads = batch.page_loads().cloned();
    let (plan, imports, images, pending) = batch.lane_inputs();
    let inputs = BatchInputs { imports, images };
    let Some(mut parts) = backend.lane_parts() else {
        return Err(WorkerError::classify(unserved(kind), None));
    };
    let (job, next) = match state {
        None => {
            let job = Job::new();
            match Opening::open(&mut parts, plan, &job, page_loads.as_ref()) {
                Ok(opening) if opening.calls.reads_predicates() => {
                    (Some(job), Pending::Predicates(opening))
                }
                Ok(opening) => opened(&mut parts, plan, inputs, opening, job),
                Err(error) => ended(job, Failure::new(PHASE)(error)),
            }
        }
        Some(Pending::Opened(opening)) => opened(&mut parts, plan, inputs, opening, Job::new()),
        Some(Pending::Prepared(opening, step)) => {
            record(&mut parts, plan, inputs, opening, Some(step), Job::new())
        }
        Some(Pending::Failing(parked, failure)) => retire(
            parts.storage,
            parts.lanes.ledger.as_mut(),
            *parked,
            Job::new(),
            failure,
        ),
        Some(
            state @ (Pending::Predicates(_)
            | Pending::Preparing(..)
            | Pending::Committing(_)
            | Pending::Discarding(_)
            | Pending::Failed(_)),
        ) => {
            // A job of the batch is on the lane: its completion comes before
            // the next job.
            *pending = Some(state);
            return Err(WorkerError::classify(
                Error::Invariant("a batch's job was prepared before its last completed".into()),
                Some(PHASE),
            ));
        }
    };
    *pending = Some(next);
    Ok(job)
}

/// What a batch's calls read that other owners prepared before its
/// forward: products imported from other ranks, and an image encoding's
/// inline images prepared on the host.
#[derive(Clone, Copy)]
struct BatchInputs<'a> {
    imports: &'a Inputs,
    images: Option<&'a InlineImages>,
}

/// Continue an opened batch: a lane that prepares asks the executor for
/// the batch's preparation in `job`; any other records its last job.
fn opened(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    inputs: BatchInputs<'_>,
    opening: Opening,
    job: Job,
) -> (Option<Job>, Pending) {
    let lanes = &*parts.lanes;
    let lane = &lanes.lanes[opening.lane];
    if !lane.prepares {
        return record(parts, plan, inputs, opening, None, job);
    }
    let described = prepared_step(plan, parts.info, opening.lane, opening.calls.calls()).and_then(
        |mut step| {
            // An image model's calls also state their input images and
            // progress.
            if image::serves(lane.resources, lane.prepares) {
                image::describe(
                    &mut step,
                    plan,
                    opening.calls.calls(),
                    parts.requests,
                    &parts.storage.tensors,
                    inputs.images,
                )?;
            }
            Ok(step)
        },
    );
    match described {
        Ok(step) => {
            job.call_texts(lanes.executor.prepare(opening.lane, step.to_string()));
            (Some(job), Pending::Preparing(opening, step))
        }
        Err(error) => ended(job, Failure::new(PHASE)(error)),
    }
}

/// The step of a batch that failed holding no reservation: `job` performs
/// what it recorded before, if anything, and the batch's calls fail from
/// its completion.
fn ended(job: Job, failure: Failure) -> (Option<Job>, Pending) {
    if job.is_empty() {
        (None, Pending::Failed(failure))
    } else {
        (Some(job), Pending::Discarding(failure))
    }
}

/// Retire what a failed batch reserved, recording in `job` the fences that
/// end it: revoke the exports, release borrowed rows, end the reads,
/// abandon the writes and seal and abandon the lease.
fn retire(
    storage: &WorkerMemory,
    ledger: Option<&mut LatentLedger>,
    parked: Parked,
    job: Job,
    failure: Failure,
) -> (Option<Job>, Pending) {
    if failure.committed {
        return (None, Pending::Failed(failure));
    }
    let Parked {
        mut batch,
        mut publications,
        image: mut held,
        ..
    } = parked;
    for state in &mut batch.calls {
        let products = &mut state.forward.products;
        products
            .exports
            .drain(..)
            .for_each(|(_, export)| export.revoke());
        release_borrowed(&products.borrowed);
    }
    let revoked = publications.revoke();
    let stream = batch.stream.clone();
    let withdrawn = image::retire(
        ledger,
        &storage.events,
        &mut batch.calls,
        &mut held,
        job.on(stream.as_ref()),
    );
    let discarded = batch.discard(&storage.tensors, &storage.events, &job);
    match revoked.and(withdrawn).and(discarded) {
        Ok(()) => ended(job, failure),
        // Storage the cleanup could not retire stays reserved; the rank
        // cannot reuse it safely.
        Err(cleanup) => (None, Pending::Failed(Failure::fatal(PHASE)(cleanup))),
    }
}

/// Output words a batch of `plan`'s calls leases: one completion word per
/// call of a prepared batch, beside the words its decoded images are lent
/// (`step`'s preparation); otherwise per call its results' words: a
/// readout's candidate scores, a canvas step's outcome and tokens, or a
/// sampling completion and the words of the logprob payload the engine
/// bounds the call by.
fn lease_words(
    prepares: bool,
    plan: &BatchPlan,
    step: Option<&serde_json::Value>,
) -> Result<usize> {
    let calls = plan.calls.len();
    if prepares {
        let images = step.map_or(Ok(0), image::lease_words)?;
        return Ok(calls.saturating_add(images));
    }
    Ok(plan
        .calls
        .iter()
        .fold(0_usize, |total, call| {
            let words = match (&call.readout, call.canvas) {
                (Some(readout), _) => readout.candidate_count().max(1),
                (None, Some(_)) => 1 + call.bounds.max_tokens as usize,
                (None, None) => COMPLETION_FIELDS.saturating_add(
                    usize::try_from(
                        call.bounds
                            .max_completion_bytes
                            .div_ceil(PAYLOAD_BYTES_PER_WORD),
                    )
                    .unwrap_or(usize::MAX),
                ),
            };
            total.saturating_add(words)
        })
        .max(calls))
}

/// Lease the batch's output storage and record its last job: its
/// reservation, the executor's execution, the seal and what follows the
/// commit point. A failure retires what the batch reserved in the same
/// job.
fn record(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    inputs: BatchInputs<'_>,
    opening: Opening,
    step: Option<serde_json::Value>,
    job: Job,
) -> (Option<Job>, Pending) {
    let Opening { lane, calls } = opening;
    let executor_lane = &mut parts.lanes.lanes[lane];
    let words = match lease_words(executor_lane.prepares, plan, step.as_ref()) {
        Ok(words) => words,
        Err(error) => return ended(job, Failure::new(PHASE)(error)),
    };
    let leased = Batch::lease(
        calls,
        &mut executor_lane.outputs,
        words,
        executor_lane.stream.clone(),
        executor_lane.device,
        &parts.storage.events,
        &job,
    );
    let batch = match leased {
        Ok(batch) => batch,
        Err(error) => return ended(job, Failure::new(PHASE)(error)),
    };
    let mut parked = Parked {
        lane,
        batch,
        executes: false,
        point: 0,
        _held: Vec::new(),
        records: Arc::new(OnceLock::new()),
        lent: 0,
        publications: Publications::default(),
        image: ImageBatch::default(),
    };
    match record_last(parts, plan, inputs, step, &mut parked, &job) {
        Ok(()) => (Some(job), Pending::Committing(Box::new(parked))),
        Err(failure) => retire(
            parts.storage,
            parts.lanes.ledger.as_mut(),
            parked,
            job,
            failure,
        ),
    }
}

/// Record the batch's last job into `job`, up to what follows its commit
/// point.
fn record_last(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    inputs: BatchInputs<'_>,
    step: Option<serde_json::Value>,
    parked: &mut Parked,
    job: &Job,
) -> std::result::Result<(), Failure> {
    let failed = Failure::new(PHASE);
    let index = parked.lane;
    let batch = &mut parked.batch;
    let stream = batch.stream.clone();
    let issue = job.on(stream.as_ref());
    let lanes = &mut *parts.lanes;
    let lane = &lanes.lanes[index];
    let executor = Arc::clone(&lanes.executor);
    let tensors = &parts.storage.tensors;

    if lane.prepares {
        // The products the preparation named, then the execution on them
        // and the publication of what it wrote. A prepared batch runs
        // whichever of its calls are active.
        let mut step = step.ok_or_else(|| {
            failed(Error::Invariant(
                "a prepared lane's batch has no preparation".into(),
            ))
        })?;
        batch.bind_flags(plan, tensors, job).map_err(&failed)?;
        let owners = ProductOwners {
            store: tensors,
            transports: &parts.storage.transports,
        };
        let prepared = PreparedOwners {
            owners,
            info: parts.info,
            ledger: lanes.ledger.as_ref(),
            host_stream: lanes.host_stream.clone(),
            borrowed: &inputs.imports.borrowed(),
        };
        let (mut bound, published) =
            bind_prepared(&prepared, plan, batch, &mut step, issue).map_err(&failed)?;
        // An image model's calls bind their own products, pixels and
        // trajectories, and a denoising batch its rows over the prefix
        // cache; what they wrote is published behind the execution.
        let images = image::serves(lane.resources, lane.prepares);
        let mut image_owners = image::Owners {
            store: tensors,
            transports: &parts.storage.transports,
            events: &parts.storage.events,
            requests: &*parts.requests,
            prefix_cache: parts.storage.prefix_cache.as_mut(),
            pages: lanes.pages.as_mut(),
            ledger: lanes.ledger.as_mut(),
            images: lanes.images.as_ref(),
        };
        if images {
            let tensors = image::bind(
                &mut image_owners,
                plan,
                batch,
                &mut parked.image,
                &mut step,
                job,
                inputs.images,
            )
            .map_err(&failed)?;
            bound.extend(tensors);
        }
        job.call_returning(executor.execute(index, step.to_string(), bound.clone()));
        publish(owners, plan, batch, published, issue).map_err(&failed)?;
        if images {
            image::publish(&mut image_owners, plan, batch, &mut parked.image, job)
                .map_err(&failed)?;
        }
        parked._held = bound;
        parked.executes = true;
    } else {
        let (Some(pages), Some(prefix_cache)) =
            (lanes.pages.as_ref(), parts.storage.prefix_cache.as_mut())
        else {
            return Err(failed(Error::Unsupported(
                "a lane that does not prepare computes over the prefix cache".into(),
            )));
        };
        let reservation = PageReservation::reserve(
            PageOwners {
                pages,
                prefix_cache,
                tensors,
                requests: parts.requests,
            },
            plan,
            batch,
            job,
        )
        .map_err(&failed)?;
        // A batch whose calls were all skipped calls no executor method.
        parked.executes = batch.calls.iter().any(|state| state.active);
        if parked.executes {
            let column =
                column_step(tensors, plan, index, batch, &reservation, issue).map_err(&failed)?;
            job.call_returning(executor.execute(
                index,
                column.step.to_string(),
                column.tensors.clone(),
            ));
            // The records are checked on the lane's thread right behind the
            // call that wrote them, before anything past it is performed.
            let records = Arc::clone(&parked.records);
            let memory = Arc::clone(column.records().memory());
            let (capacity, limits, lent_words) =
                (column.capacity, column.limits.clone(), column.lent_words);
            job.check(Box::new(move |completion: &Completion| {
                let malformed =
                    || Error::Interface("execute returns the record words it wrote".into());
                let used = match completion.last_returned() {
                    Some(&[used]) => usize::try_from(used).map_err(|_| malformed())?,
                    _ => return Err(malformed()),
                };
                // SAFETY: the records' host words, which the executor wrote
                // during the call this check follows and nothing writes
                // since; the parked batch keeps them.
                let words =
                    unsafe { std::slice::from_raw_parts(memory.address() as *const i64, capacity) };
                let parsed = Records::parse(words, used, &limits, lent_words)?;
                records
                    .set(parsed)
                    .map_err(|_| Error::Invariant("a batch's records were checked twice".into()))
            }));
            let staging = lane.staging.as_ref().ok_or_else(|| {
                failed(Error::Invariant(
                    "a lane over the prefix cache has no relay staging".into(),
                ))
            })?;
            relay(tensors, staging, plan, batch, job).map_err(&failed)?;
            // The executor's copies into the lent words lie behind; the
            // lease marks where their timing begins.
            lock(&batch.buffer)
                .begin_copy(batch.device, issue, &parts.storage.events)
                .map_err(&failed)?;
            parked.lent = column.lent;
            parked._held = column.tensors;
        }
    }

    // Readers end on the batch's stream behind the relays when every read
    // has one. The host publishes the completion flags of the executed
    // calls, but a token-denoising step's, which its relay carries.
    let after: Vec<_> = batch
        .calls
        .iter()
        .filter_map(|state| state.forward.pages.token_write.clone())
        .collect();
    batch.end_reads(tensors, job, &after).map_err(&failed)?;
    if !denoising(plan) {
        batch.set_flags(tensors, job).map_err(&failed)?;
    }
    batch.validate(plan, tensors).map_err(&failed)?;
    let updates: Vec<_> = batch
        .calls
        .iter()
        .filter_map(|state| state.forward.products.latent.clone())
        .collect();
    match &lanes.ledger {
        Some(ledger) => ledger.validate(&updates),
        None if updates.is_empty() => Ok(()),
        None => Err(Error::Invariant(
            "a trajectory commit has no latent ledger".into(),
        )),
    }
    .map_err(&failed)?;
    parts
        .requests
        .validate_pending(&Batch::<LaneCall>::pending(plan))
        .map_err(&failed)?;
    batch.seal(&parts.storage.events, job).map_err(&failed)?;

    // Every owner accepted its update. The job's operations from here on
    // change slot state, and any failure from here is fatal.
    parked.point = job.len();
    let lane = &lanes.lanes[index];
    if parked.executes && lane.commits {
        job.call(executor.commit(index));
    }
    if lane.commits
        && lane.resources.pages
        && let Some(pages) = &mut lanes.pages
    {
        // Work on other streams that reads the slot state and K/V the batch
        // wrote orders behind its commit.
        pages
            .order_mut()
            .record(issue)
            .map_err(Failure::fatal(PHASE))?;
    }
    Ok(())
}

/// Apply the completion of a lane batch's job.
pub(crate) fn complete(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    completion: Completion,
) -> std::result::Result<(), WorkerError> {
    let failed = Failure::new(PHASE);
    let state = batch.lane_parts().2.take();
    let next = match (state, completion.check()) {
        // A predicate that could not be read, or a preparation the executor
        // refused, fails the batch, which has reserved nothing yet.
        (Some(Pending::Predicates(_) | Pending::Preparing(..)), Err(failure)) => {
            Pending::Failed(failed(failure.error))
        }
        (Some(Pending::Predicates(mut opening)), Ok(mut completion)) => {
            match opening.calls.predicates(&mut completion) {
                Ok(()) => Pending::Opened(opening),
                Err(error) => Pending::Failed(failed(error)),
            }
        }
        (Some(Pending::Preparing(opening, mut step)), Ok(mut completion)) => {
            let prepared = completion.texts().and_then(|texts| {
                serde_json::from_str::<serde_json::Value>(&texts.concat()).map_err(|error| {
                    Error::Interface(format!("the executor's preparation: {error}"))
                })
            });
            match prepared {
                Ok(prepared) => {
                    step["prepared"] = prepared;
                    Pending::Prepared(opening, step)
                }
                Err(error) => Pending::Failed(failed(error)),
            }
        }
        (Some(Pending::Committing(mut parked)), completed) => {
            let committed = match completed {
                Ok(mut completion) => {
                    let (plan, _, _) = batch.lane_parts();
                    let Some(mut parts) = backend.lane_parts() else {
                        return Err(WorkerError::classify(
                            Error::State("the executor lanes left a batch in flight"),
                            Some(PHASE),
                        ));
                    };
                    // A prepared batch's exports may still be refused, which
                    // retires the batch before its commit point.
                    match accept(&mut parts, &mut parked, &mut completion) {
                        Ok(modes) => commit(&mut parts, plan, parked, &modes, &mut completion)
                            .map_err(|failure| (None, failure)),
                        Err(error) => Err((Some(parked), failed(error))),
                    }
                }
                Err(failure) if failure.operation >= parked.point => {
                    Err((None, Failure::fatal(PHASE)(failure.error)))
                }
                Err(failure) => Err((Some(parked), failed(failure.error))),
            };
            match committed {
                Ok(launched) => {
                    batch.launch(launched);
                    batch.set_stage(Stage::Done);
                    return Ok(());
                }
                Err((Some(parked), failure)) if !failure.committed => {
                    Pending::Failing(parked, failure)
                }
                Err((_, failure)) => Pending::Failed(failure),
            }
        }
        // Storage the cleanup could not retire stays reserved; the rank
        // cannot reuse it safely.
        (Some(Pending::Discarding(_)), Err(failure)) => {
            Pending::Failed(Failure::fatal(PHASE)(failure.error))
        }
        (Some(Pending::Discarding(failure)), Ok(_)) => Pending::Failed(failure),
        (state, _) => {
            *batch.lane_parts().2 = state;
            return Err(WorkerError::classify(
                Error::Invariant("a batch has no job on its lane".into()),
                Some(PHASE),
            ));
        }
    };
    *batch.lane_parts().2 = Some(next);
    Ok(())
}

/// Accept what the batch's last job returned before its commit point: a
/// prepared batch's graph mode per call and the exports its forwards
/// issued, which may still be refused (the caller then retires the
/// batch). A lane that does not prepare had its records checked by the
/// job itself and returns no modes.
fn accept(
    parts: &mut Parts<'_>,
    parked: &mut Parked,
    completion: &mut Completion,
) -> Result<Vec<GraphMode>> {
    if !parts.lanes.lanes[parked.lane].prepares {
        return Ok(Vec::new());
    }
    let tensors = &parts.storage.tensors;
    let modes = completion
        .returned()?
        .into_iter()
        .map(|code| {
            GraphMode::from_code(code).ok_or_else(|| {
                Error::Interface(format!("the executor ran a forward in graph mode {code}"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for state in &mut parked.batch.calls {
        for (write, export) in std::mem::take(&mut state.forward.products.exports) {
            parked
                .publications
                .locate(tensors, &write, export, &mut state.output)?;
        }
    }
    // An image model's trajectory versions, published from the ledger.
    image::accept(
        parts.lanes.ledger.as_ref(),
        &mut parked.batch.calls,
        &mut parked.publications,
    )?;
    parked.publications.validate(tensors)?;
    Ok(modes)
}

/// Apply the completion of the batch's last job, past its commit point:
/// its results to the calls' pending outputs (a lane's records, whose
/// check passed before the commit point, or a prepared batch's accepted
/// exports and trajectory commits with its graph `modes`), the written
/// output words to the lease; then make the products visible and the calls
/// pending. Every failure here is fatal.
fn commit(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    mut parked: Box<Parked>,
    modes: &[GraphMode],
    completion: &mut Completion,
) -> std::result::Result<Launched, Failure> {
    let fatal = Failure::fatal(PHASE);
    let lanes = &mut *parts.lanes;
    let lane = &lanes.lanes[parked.lane];
    let tensors = &parts.storage.tensors;
    let elapsed = parked.batch.started.elapsed().as_micros() as u64;
    let (stats, timer) = if lane.prepares {
        // Every owner accepted its update.
        let updates: Vec<_> = parked
            .batch
            .calls
            .iter()
            .filter_map(|state| state.forward.products.latent.clone())
            .collect();
        if let Some(ledger) = &mut lanes.ledger {
            ledger.apply(&updates).map_err(&fatal)?;
        }
        for state in &mut parked.batch.calls {
            if let Some(update) = state.forward.products.latent.take() {
                state.output.apply_latent_update(&update).map_err(&fatal)?;
            }
        }
        parked.publications.register(tensors).map_err(&fatal)?;
        let timer = if image::serves(lane.resources, lane.prepares) {
            "image_materialize"
        } else {
            "video_materialize"
        };
        (prepared_statistics(modes), timer)
    } else {
        let records = if parked.executes {
            // The integer the executor returned, which the check read.
            completion.returned().map_err(&fatal)?;
            Some(
                parked
                    .records
                    .get()
                    .ok_or_else(|| Error::Invariant("a batch's records were not checked".into()))
                    .map_err(&fatal)?,
            )
        } else {
            None
        };
        if let Some(records) = records {
            for (index, state) in parked.batch.calls.iter_mut().enumerate() {
                if state.active {
                    records
                        .apply(index, &mut state.output, parked.lent)
                        .map_err(&fatal)?;
                }
            }
            lock(&parked.batch.buffer)
                .register_lent(records.written, records.columns())
                .map_err(&fatal)?;
        }
        column_statistics(plan, elapsed, records)
    };
    parked.batch.commit(plan, tensors, parts.requests)?;

    // The host tasks the calls' results wait for.
    let host = if lane.resources.reports {
        let inbox = lanes
            .reports
            .as_ref()
            .map(Reports::inbox)
            .ok_or_else(|| Error::State("a lane's host tasks have no report pipe"))
            .map_err(&fatal)?;
        let calls: Vec<HostCall> = parked
            .batch
            .calls
            .iter_mut()
            .enumerate()
            .filter(|(_, state)| state.forward.products.reports > 0)
            .map(|(index, state)| {
                let products = &mut state.forward.products;
                HostCall::new(
                    index,
                    std::mem::take(&mut products.reports),
                    products.publication.take(),
                    std::mem::take(&mut products.borrowed),
                )
            })
            .collect();
        (!calls.is_empty()).then(|| {
            Box::new(HostTasks {
                calls,
                inbox,
                batch: plan.batch_id,
            }) as Box<dyn HostStage>
        })
    } else if lane.resources.images {
        // The PNG encodings of an image model's decoded images.
        image::host_stage(&mut parked.batch.calls)
    } else {
        None
    };
    let Parked { batch, .. } = *parked;
    Ok(batch.launch(stats, timer, host))
}

/// Count one forward that ran in `mode` under `name`: its mode count and,
/// for a graph, its capture or replay and its graph mode count.
fn count_mode(stats: &mut ForwardStats, name: &str, mode: GraphMode) {
    *stats.mode_counts.entry(name.into()).or_default() += 1;
    match mode {
        GraphMode::Capture => stats.cuda_graph_captures += 1,
        GraphMode::Replay => stats.cuda_graph_replays += 1,
        GraphMode::Eager => return,
    }
    *stats
        .cuda_graph_runtime_mode_counts
        .entry(name.into())
        .or_default() += 1;
}

/// The statistics of a prepared batch: each forward counted by the graph
/// mode it ran in.
fn prepared_statistics(modes: &[GraphMode]) -> ForwardStats {
    let mut stats = ForwardStats::default();
    for &mode in modes {
        count_mode(&mut stats, mode.as_str(), mode);
    }
    stats
}

/// The statistics of a batch from its records, and its materialization's
/// timer: how its forward ran and over how many tokens, and a
/// verification's rows and drafts. A token-denoising batch counts under its
/// call kind, any other under its graph mode.
fn column_statistics(
    plan: &BatchPlan,
    elapsed: u64,
    records: Option<&Records>,
) -> (ForwardStats, &'static str) {
    let canvas = denoising(plan);
    let (execute, materialize) = if canvas {
        ("canvas_execute", "canvas_materialize")
    } else {
        ("text_execute", "text_materialize")
    };
    let mut stats = ForwardStats::default();
    stats.component_us.insert(execute.into(), elapsed);
    let Some((records, mode)) = records.and_then(|records| Some((records, records.mode?))) else {
        return (stats, materialize);
    };
    let name = if canvas {
        CallKind::Forward(ForwardMode::TokenDenoising).as_str()
    } else {
        mode.as_str()
    };
    count_mode(&mut stats, name, mode);
    *stats.mode_tokens.entry(name.into()).or_default() += records.live_tokens;
    if !canvas {
        *stats.mode_us.entry(name.into()).or_default() += elapsed;
    }
    if mode != GraphMode::Eager {
        stats.cuda_graph_unpadded_tokens += records.live_tokens;
        stats.cuda_graph_padded_tokens += records.padding_tokens;
    }
    if plan
        .calls
        .first()
        .is_some_and(|call| call.kind == CallKind::Forward(ForwardMode::Verify))
    {
        stats.spec_verify_rows += records.rows;
        stats.spec_verify_draft_tokens += records.draft_tokens;
    }
    (stats, materialize)
}
