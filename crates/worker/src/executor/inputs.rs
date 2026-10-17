//! Inputs a batch imports from other calls, prepared before it executes.
//!
//! Every capability runtime reads its tensor inputs from the rank's tensor
//! store, and its trajectories from the rank's latent pool. Products
//! another rank or worker published reach this rank as descriptors
//! (`Batch::input_products`): tensors are read into the tensor store,
//! latent trajectory versions into the consuming request's pages; cache
//! transfers its install calls name arrive as `Batch::cache_inputs` or are
//! resident exports of this rank. This module materializes them while
//! other batches execute, so a runtime consumes imported values exactly
//! like values its own rank produced.
//!
//! Preparation runs on the executor thread in phases, each resumed by the
//! readiness notice of what blocked it ([`admit`], [`prepare`],
//! [`subscribe`]):
//!
//! 1. **Storage hazards.** A batch that writes cache storage (newly assigned
//!    units, forward rows, install destinations) first waits for every
//!    physical access still overlapping it: executions of earlier batches,
//!    imports, and exports other ranks are reading; one that writes a
//!    latent bank waits for the exports of versions in it. A failed access
//!    keeps its storage forever, so the batch then fails instead.
//! 2. **Admission.** Product imports are admitted in descriptor order
//!    ([`admit`]). What an import needs on its consumers' stream, the
//!    handoff fence each of its reads waits on, is recorded into one job
//!    of the batch's lane; the reads are handed to their transport threads
//!    from that job's completion ([`admitted`]), when the fences' records
//!    are made. A refusal for read credits resumes at the refused product
//!    once credits return; accepted imports keep their destinations
//!    meanwhile. Cache imports follow ([`prepare`]), after which nothing is
//!    repeatable.
//! 3. **Readiness.** The batch launches once every read and cache copy is
//!    consumable.
//!
//! When the batch executes, [`complete`] publishes each imported product on
//! its consumers' device stream (a new product is committed, a widened
//! shard is fenced over its union, a trajectory version is adopted as its
//! slot's committed one) before the runtime consumes it; imports only
//! predicated-off calls read are left unpublished. [`close`] ends the
//! import leases and abandons every import the batch did not consume, on
//! success and on failure alike.
//!
//! [`BatchInputs`] holds the resources a batch retains while its inputs
//! are prepared.

use super::backend::{WorkerBackend, WorkerBatch};
use super::batch::{flag, request_flag};
use super::transfer::latents::{self, LatentInput};
use crate::error::WorkerError;
use crate::executor::Submission;
use crate::ffi::MemoryKind;
use crate::lane::{Completion, Job};
use crate::storage::{
    Metadata, OutputPlacement, PendingImport, PrefixCache, ProductRead, ProductStore, ValueRange,
};
use crate::transfer::{Notify, Retirement, fetch};
use crate::{Error, Outcome, Result};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use uniserve_ipc::{
    Batch as BatchPlan, BufferId, CacheTransfer, Call, CallKind, DType, MediaCall, TensorExport,
    TensorRef, TransferHandle, TransferMode, TransferTransport,
};

/// Input leases and preparation progress for one scheduler batch: its
/// imported inputs and the physical accesses its storage writes wait for.
/// Readiness queries neither submit work nor invoke completion observers.
pub(crate) struct BatchInputs {
    pub(crate) inputs: HashMap<BufferId, Input>,
    pub(crate) dependencies: Vec<Hazard>,
    pub(crate) started: usize,
    pub(crate) submitted: bool,
    pub(crate) awaiting_reads: bool,
    closed: Arc<AtomicBool>,
}

impl Default for BatchInputs {
    fn default() -> Self {
        Self {
            inputs: HashMap::new(),
            dependencies: Vec::new(),
            started: 0,
            submitted: false,
            awaiting_reads: false,
            closed: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl BatchInputs {
    pub(crate) fn closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Claim cleanup before releasing any resource. A failed release does
    /// not prevent the remaining owners from being closed by the backend.
    pub(crate) fn close(&self) -> bool {
        !self.closed.swap(true, Ordering::AcqRel)
    }

    /// The extra arrival prevents an already-complete dependency from firing
    /// the observer before registration has finished.
    pub(crate) fn wait(&self, dependencies: usize) -> Arc<InputWait> {
        Arc::new(InputWait {
            remaining: AtomicUsize::new(dependencies + 1),
            closed: Arc::clone(&self.closed),
        })
    }

    /// Whether every physical access the batch's writes wait for finished.
    pub(crate) fn storage_ready(&self) -> bool {
        self.dependencies.iter().all(Hazard::ready)
    }

    /// Whether every input is consumable once admission was submitted.
    pub(crate) fn ready(&self) -> bool {
        self.submitted && self.storage_ready() && self.inputs.values().all(Input::ready)
    }
}

/// One snapshot of input dependencies. Each dependency arrives exactly once;
/// only the last arrival may notify the executor, and closing suppresses it.
pub(crate) struct InputWait {
    remaining: AtomicUsize,
    closed: Arc<AtomicBool>,
}

impl InputWait {
    pub(crate) fn arrive(&self) -> bool {
        self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 && !self.closed.load(Ordering::Acquire)
    }
}

/// One imported input of a batch.
pub(crate) enum Input {
    /// A product read into this rank's tensor store on `device`.
    Tensor {
        read: Arc<ProductRead>,
        device: MemoryKind,
    },
    /// A cache transfer copied into its install call's pages.
    Cache(PendingImport),
    /// A latent trajectory version read into the consuming call's slot,
    /// on `device`.
    Latent {
        input: LatentInput,
        device: MemoryKind,
    },
    /// A product a host media call reads in place from a location on this
    /// node; importing it would read unused capacity.
    Borrowed,
}

impl Input {
    /// Whether the input is consumable: its reads or its cache copy
    /// finished.
    fn ready(&self) -> bool {
        match self {
            Self::Tensor { read, .. } => read.reads().iter().all(|read| read.ready()),
            Self::Cache(import) => import.done(),
            Self::Latent { input, .. } => input.reads().iter().all(|read| read.ready()),
            Self::Borrowed => true,
        }
    }
}

/// A physical access a batch's cache writes wait for.
pub(crate) struct Hazard(Arc<Retirement>);

impl Hazard {
    /// Whether the access finished (successfully or not).
    fn ready(&self) -> bool {
        self.0.done()
    }
}

/// A batch's imported inputs and their preparation progress.
#[derive(Default)]
pub(crate) struct Inputs {
    state: BatchInputs,
    /// The storage hazards were resolved.
    resolved: bool,
    /// Resident cache exports the batch's install calls read.
    resident: Vec<Arc<CacheTransfer>>,
    /// The read-credit return sequence a refused import waits past.
    backpressure: Option<u64>,
    /// Preparation stopped on these inputs (not on another runtime's).
    waiting: bool,
    /// A product import was refused for read credits while the admission
    /// job held work: the batch's start from that job's completion admits
    /// nothing, and the refused product is admitted again from the notice
    /// of returned credits.
    refused: bool,
    /// Where the completion of the imports stands between its jobs.
    completing: Completing,
}

/// The completion of a batch's imported inputs between its jobs.
#[derive(Default)]
enum Completing {
    #[default]
    Unstarted,
    /// The job on the lane publishes the imported predicates and reads the
    /// host-gated predicates of these calls, in this order: they decide
    /// which other imports are read.
    Predicates(Vec<usize>),
    /// The predicates are known: whether each of those calls is skipped.
    Decided(HashMap<usize, bool>),
    /// The job on the lane publishes the imports an executing call reads.
    Publishing,
    /// Recording a job failed after it recorded work: the job on the lane
    /// performs that work, whose fences are still due, and the completion
    /// fails with this error.
    Failing(Error),
}

impl Inputs {
    /// The inputs host media calls read in place from their exports.
    pub(crate) fn borrowed(&self) -> HashSet<BufferId> {
        self.state
            .inputs
            .iter()
            .filter(|(_, input)| matches!(input, Input::Borrowed))
            .map(|(id, _)| *id)
            .collect()
    }

    /// The cache import of an install call's source export.
    pub(crate) fn cache_import(&self, source: BufferId) -> Option<&PendingImport> {
        match self.state.inputs.get(&source) {
            Some(Input::Cache(import)) => Some(import),
            _ => None,
        }
    }
}

/// The device a component's calls run on.
pub(crate) fn component_device(backend: &WorkerBackend, component: &str) -> Result<MemoryKind> {
    let placed = backend
        .components()
        .get(component)
        .ok_or_else(|| Error::Invalid(format!("call targets unplaced component {component:?}")))?;
    Ok(placed.device.memory())
}

/// Admit the batch's product imports that can be admitted now, recording
/// what they issue on their consumers' stream into the returned job. The
/// caller submits a job that holds work to the batch's lane, whatever the
/// result, and hands its completion to [`admitted`]: fences recorded by an
/// import that was refused or failed afterwards are still due. With an
/// empty job nothing waits for the lane.
pub(crate) fn admit(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
) -> (Job, std::result::Result<(), WorkerError>) {
    let plan = Arc::clone(batch.shared_plan());
    let job = Job::new();
    let admitted = admit_products(backend, batch, &plan, &job).and_then(|()| {
        if job.is_empty() {
            // No fence is pending: the reads start at once.
            start_reads(batch)
        } else {
            Ok(())
        }
    });
    let admitted = admitted.map_err(|error| failed(backend, batch, error));
    (job, admitted)
}

/// Hand the admitted imports' reads to their transport threads: the job
/// [`admit`] returned completed, so the handoff fences they wait on are
/// recorded.
pub(crate) fn admitted(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    completion: Completion,
) -> std::result::Result<(), WorkerError> {
    let started = completion
        .check()
        .map_err(|failure| failure.error)
        .and_then(|_| start_reads(batch));
    started.map_err(|error| failed(backend, batch, error))
}

/// Advance a batch's input preparation behind its admission; whether its
/// imports are ready.
pub(crate) fn prepare(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
) -> std::result::Result<bool, WorkerError> {
    let plan = Arc::clone(batch.shared_plan());
    let ready = advance(backend, batch, &plan).map_err(|error| failed(backend, batch, error))?;
    batch.inputs_mut().waiting = !ready;
    Ok(ready)
}

/// The failure of a batch's input preparation: accepted imports retire
/// through their owners, and the executor reports the batch failed.
fn failed(backend: &mut WorkerBackend, batch: &mut WorkerBatch, error: Error) -> WorkerError {
    let mut error = WorkerError::classify(error, Some("input preparation"));
    if let Err(cleanup) = close(backend, batch) {
        error.note(&WorkerError::classify(cleanup, None));
    }
    error
}

/// Resolve the batch's storage hazards and, once they retired, admit its
/// product imports from the first one not yet admitted.
fn admit_products(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    plan: &BatchPlan,
    job: &Job,
) -> Result<()> {
    if batch.inputs().state.closed() {
        return Err(Error::State("batch inputs were already consumed"));
    }
    if !batch.inputs().resolved {
        let Resolved {
            resident,
            mut hazards,
        } = resolve(backend, plan)?;
        hazards.extend(latent_hazards(backend, plan)?);
        let inputs = batch.inputs_mut();
        inputs.resident = resident;
        inputs.state.dependencies = hazards.into_iter().map(Hazard).collect();
        inputs.resolved = true;
    }
    // A refusal waits for the notice of returned credits, which starts the
    // batch again; this start follows the admission job's completion.
    if std::mem::take(&mut batch.inputs_mut().refused) {
        return Ok(());
    }
    if batch.inputs().state.submitted || !batch.inputs().state.storage_ready() {
        return Ok(());
    }
    require_storage(&batch.inputs().state)?;
    if !import_products(backend, batch, plan, job)? {
        batch.inputs_mut().refused = !job.is_empty();
    }
    Ok(())
}

/// Start the reads of every product import the batch admitted. A read that
/// was started before stays as it is.
fn start_reads(batch: &WorkerBatch) -> Result<()> {
    for input in batch.inputs().state.inputs.values() {
        match input {
            Input::Tensor { read, .. } => fetch::start(&read.reads())?,
            Input::Latent { input, .. } => fetch::start(input.reads())?,
            Input::Cache(_) | Input::Borrowed => {}
        }
    }
    Ok(())
}

fn advance(backend: &mut WorkerBackend, batch: &mut WorkerBatch, plan: &BatchPlan) -> Result<bool> {
    if batch.inputs().state.closed() {
        return Err(Error::State("batch inputs were already consumed"));
    }
    if !batch.inputs().state.submitted {
        // The hazards were resolved and the products admitted by `admit`,
        // unless a hazard or a refused import still holds them up.
        let inputs = batch.inputs();
        if !inputs.state.storage_ready() || inputs.state.started < plan.input_products.len() {
            return Ok(false);
        }
        import_cache(backend, batch, plan)?;
        batch.inputs_mut().state.submitted = true;
    }
    Ok(batch.inputs().state.ready())
}

/// What a batch's cache transfers and writes depend on.
struct Resolved {
    /// Resident exports the batch's install calls read.
    resident: Vec<Arc<CacheTransfer>>,
    /// Physical accesses overlapping the cache storage the batch writes.
    hazards: Vec<Arc<Retirement>>,
}

/// Resolve the resident exports install calls read and the physical
/// accesses the batch's cache writes overlap. Neither reserves storage.
fn resolve(backend: &mut WorkerBackend, plan: &BatchPlan) -> Result<Resolved> {
    let parts = backend.parts();
    let Some(prefix_cache) = parts
        .storage
        .and_then(|storage| storage.prefix_cache.as_mut())
    else {
        return Ok(Resolved {
            resident: Vec::new(),
            hazards: Vec::new(),
        });
    };

    // Supplied transfers take precedence; other install sources must be
    // exports resident on this rank.
    let mut supplied: HashSet<BufferId> = plan
        .cache_inputs
        .iter()
        .map(|transfer| transfer.source)
        .collect();
    let mut resident = Vec::new();
    for call in installs(plan) {
        let source = call
            .cache_input
            .ok_or_else(|| Error::Invalid("cache installation requires a source export".into()))?;
        if supplied.insert(source) {
            let export = prefix_cache
                .resident(source)
                .ok_or_else(|| Error::Invalid("cache export buffer is not resident".into()))?;
            resident.push(Arc::clone(export));
        }
    }

    if !prefix_cache.has_pending_accesses() {
        return Ok(Resolved {
            resident,
            hazards: Vec::new(),
        });
    }
    let spans = write_spans(prefix_cache, parts.requests, plan, &resident)?;
    let hazards = prefix_cache.write_dependencies(&spans);
    Ok(Resolved { resident, hazards })
}

/// The exports of trajectory versions in the banks the batch's latent
/// preparation and denoising calls write: other ranks still reading a
/// version keep its bank from being overwritten.
///
/// A denoising step writes the bank of the version two steps behind the
/// one it reads, which its trajectory has moved past. That version is
/// revoked here (`LatentPool::supersede`) rather than
/// left published until its Free: the step then waits only for the reads
/// other ranks already claimed, which finish by themselves, and never for
/// a later batch of the head that would carry the Free.
fn latent_hazards(backend: &mut WorkerBackend, plan: &BatchPlan) -> Result<Vec<Arc<Retirement>>> {
    let writes: Vec<(usize, &uniserve_ipc::LatentParams, bool)> = plan
        .latent_params
        .iter()
        .filter_map(|params| {
            let call = plan.calls.iter().find(|call| {
                call.request_key == params.request_key && call.call_id == params.call_id
            })?;
            matches!(
                call.code,
                CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
            )
            .then_some((params, call.code == CallKind::Media(MediaCall::Denoising)))
        })
        .map(|(params, denoising)| {
            let slot = backend
                .requests()
                .get(params.request_key.request_id.0)?
                .slot();
            Ok((slot, params, denoising))
        })
        .collect::<Result<_>>()?;
    if writes.is_empty() {
        return Ok(Vec::new());
    }
    let parts = backend.parts();
    let Some(mut image) = parts.image else {
        return Ok(Vec::new());
    };
    let Ok((pool, _)) = image.latent_pool() else {
        return Ok(Vec::new());
    };
    pool.reap();
    let mut hazards = Vec::new();
    let mut superseded = Vec::new();
    for (slot, params, denoising) in writes {
        let slot = pool.slot(slot as i64)?;
        let pages: Vec<usize> = params
            .page_table
            .iter()
            .map(|&page| page as usize)
            .collect();
        if denoising {
            superseded.extend(pool.supersede(slot, params.request_key, &pages));
        }
        hazards.extend(pool.write_dependencies(slot, &pages));
    }
    if !superseded.is_empty() {
        // A version is published through the product store's locations;
        // revoking them stops new claims and lets each export retire once
        // its claimed reads finished.
        parts
            .storage
            .ok_or_else(|| Error::Invariant("published latents need storage owners".into()))?
            .tensors
            .revoke_buffer_exports(&superseded)?;
    }
    Ok(hazards)
}

/// The install calls of a batch.
fn installs(plan: &BatchPlan) -> impl Iterator<Item = &Call> {
    plan.calls
        .iter()
        .filter(|call| call.code == CallKind::Transfer(TransferMode::CacheInstall))
}

/// Every cache span the batch writes: whole newly assigned units (a recycled
/// unit may have served another group's page size), forward rows that
/// write their queries, and the extent each install copies.
fn write_spans(
    prefix_cache: &mut PrefixCache,
    requests: &crate::request::RequestPool,
    plan: &BatchPlan,
    resident: &[Arc<CacheTransfer>],
) -> Result<Vec<(u32, u32, u32)>> {
    let unit_tokens = prefix_cache.plan().unit_tokens();
    let tables = prefix_cache.tables();
    let mut spans = Vec::new();
    for allocation in &plan.new_cache_units {
        spans.extend(
            allocation
                .unit_ids
                .iter()
                .map(|unit| (unit.0, 0, unit_tokens)),
        );
    }
    let forward = &plan.forward;
    for (row, &write) in forward.write_cache.iter().enumerate() {
        if !write {
            continue;
        }
        let length = u64::from(forward.query_lens[row]);
        let start = u64::from(forward.seq_lens[row]).saturating_sub(length);
        for table in tables.for_batch(forward.request_pool_indices[row], &plan.block_tables)? {
            spans.extend(table.spans(start, length)?);
        }
    }
    let transfers: HashMap<BufferId, &CacheTransfer> = plan
        .cache_inputs
        .iter()
        .chain(resident.iter().map(AsRef::as_ref))
        .map(|transfer| (transfer.source, transfer))
        .collect();
    for call in installs(plan) {
        let Some(transfer) = call.cache_input.and_then(|source| transfers.get(&source)) else {
            continue;
        };
        let slot = requests.get(call.request_key.request_id.0)?.slot() as u32;
        for table in tables.for_batch(slot, &plan.block_tables)? {
            let start = u64::from(transfer.base_extent)
                .max(u64::from(table.start_page) * u64::from(table.shape.page_tokens));
            let end = u64::from(transfer.exported_extent);
            if start < end {
                spans.extend(table.spans(start, end - start)?);
            }
        }
    }
    Ok(spans)
}

/// Fail when an overlapping access ended without retiring its storage.
fn require_storage(state: &BatchInputs) -> Result<()> {
    for Hazard(retirement) in &state.dependencies {
        match retirement.outcome() {
            Some(Outcome::Success(())) => {}
            Some(Outcome::Failed(error)) => {
                return Err(Error::transport_source(
                    "a cache access the batch overwrites failed to retire",
                    error,
                ));
            }
            Some(Outcome::Cancelled) | None => {
                return Err(Error::Resource(
                    "a cache access the batch overwrites did not retire",
                ));
            }
        }
    }
    Ok(())
}

/// The calls reading each supplied product, as an input or a predicate.
fn consumers(plan: &BatchPlan) -> HashMap<&TensorRef, Vec<usize>> {
    let mut consumers: HashMap<&TensorRef, Vec<usize>> = HashMap::new();
    for (index, call) in plan.calls.iter().enumerate() {
        for product in call.tensor_inputs().chain(call.predicate.as_ref()) {
            let indices = consumers.entry(product).or_default();
            if indices.last() != Some(&index) {
                indices.push(index);
            }
        }
    }
    consumers
}

/// Whether a host media consumer reads `export` in place: a muxer always,
/// a video encoder a segment on this node.
fn borrowed(
    backend: &WorkerBackend,
    plan: &BatchPlan,
    readers: &[usize],
    export: &TensorExport,
) -> bool {
    let shared = backend.rank_storage().is_some_and(|storage| {
        storage
            .transports
            .get(crate::transfer::TransportKind::Shm)
            .is_some()
    });
    let node = &backend.info().endpoint.node;
    readers.iter().any(|&index| {
        let call = &plan.calls[index];
        call.inputs.contains(&export.product)
            && (call.code == CallKind::Media(MediaCall::Muxing)
                || (call.code == CallKind::Media(MediaCall::VideoEncoding)
                    && shared
                    && !matches!(export.value, TransferHandle::Latent { .. })
                    && export.value.tensors().iter().any(|tensor| {
                        tensor.locations.iter().any(|location| {
                            matches!(location.transport, TransferTransport::PosixShm { .. })
                                && location.source.node == *node
                        })
                    })))
    })
}

/// The metadata an imported product carries.
fn metadata(handle: &TransferHandle) -> Result<Option<Metadata>> {
    Ok(match handle {
        TransferHandle::Encoder { height, width, .. } => Some(Metadata::Feature {
            height: *height,
            width: *width,
        }),
        TransferHandle::DeviceProduct {
            height,
            width,
            value_range,
            ..
        } if *height != 0 => Some(Metadata::Image {
            height: *height,
            width: *width,
            value_range: match value_range.as_str() {
                "signed_unit" => Some(ValueRange::SignedUnit),
                "unit" => Some(ValueRange::Unit),
                "" => None,
                other => {
                    return Err(Error::Invalid(format!(
                        "image product has an unknown value range {other:?}"
                    )));
                }
            },
        }),
        _ => None,
    })
}

/// Admit product imports from the first one not yet admitted, recording
/// their reads' handoff fences into `job`. Returns false while a refused
/// import waits for read credits.
fn import_products(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    plan: &BatchPlan,
    job: &Job,
) -> Result<bool> {
    let start = batch.inputs().state.started;
    if start >= plan.input_products.len() {
        return Ok(true);
    }
    let readers = consumers(plan);
    let allocations: HashMap<BufferId, uniserve_ipc::BufferAllocation> = plan
        .buffer_allocations
        .iter()
        .map(|allocation| (allocation.buffer, *allocation))
        .collect();

    for (index, export) in plan.input_products.iter().enumerate().skip(start) {
        batch.inputs_mut().state.started = index;
        let product = &export.product;
        let indices = readers.get(product).ok_or_else(|| {
            Error::Invalid("a supplied input product has no consumer in its batch".into())
        })?;
        if borrowed(backend, plan, indices, export) {
            batch
                .inputs_mut()
                .state
                .inputs
                .insert(product.buffer_id(), Input::Borrowed);
            continue;
        }
        if matches!(export.value, TransferHandle::Latent { .. }) {
            match import_latent(backend, plan, export, indices, job) {
                Ok(input) => {
                    batch
                        .inputs_mut()
                        .state
                        .inputs
                        .insert(product.buffer_id(), input);
                    continue;
                }
                Err(Error::ReadBackpressure { returns }) => {
                    let inputs = batch.inputs_mut();
                    inputs.backpressure = Some(returns);
                    inputs.state.awaiting_reads = true;
                    return Ok(false);
                }
                Err(error) => return Err(error),
            }
        }

        // A product may feed several calls, but it has one destination.
        let device = component_device(backend, &plan.calls[indices[0]].component)?;
        for &other in &indices[1..] {
            if component_device(backend, &plan.calls[other].component)? != device {
                return Err(Error::Invalid(
                    "transferred product requires one consumer device per batch".into(),
                ));
            }
        }
        let stream = backend.transfer_stream(plan.calls[indices[0]].code, device)?;
        let tensor = export
            .value
            .tensors()
            .first()
            .ok_or_else(|| Error::Invalid("a supplied product carries no tensor".into()))?;
        let metadata = metadata(&export.value)?;
        let slots: HashMap<_, _> = backend
            .requests()
            .peek(product.request_key.request_id.0)
            .filter(|request| request.key() == product.request_key)
            .map(|request| (product.request_key, request.slot()))
            .into_iter()
            .collect();
        let storage = backend
            .rank_storage()
            .ok_or_else(|| Error::Unsupported("product imports require rank storage".into()))?;
        if storage.transports.is_empty() {
            return Err(Error::Unsupported(
                "cross-call input requires a configured transport".into(),
            ));
        }
        // The reads' handoff fences are recorded into the job of the lane
        // that owns the consumers' stream; the reads start once it ran.
        let imported = storage.tensors.import_tensor(
            product,
            tensor,
            device,
            &storage.transports,
            OutputPlacement {
                request_slots: Some(&slots),
                allocations: Some(&allocations),
                ..OutputPlacement::default()
            },
            metadata,
            job.on(stream.as_ref()),
        );
        match imported {
            Ok(read) => {
                batch
                    .inputs_mut()
                    .state
                    .inputs
                    .insert(product.buffer_id(), Input::Tensor { read, device });
            }
            Err(Error::ReadBackpressure { returns }) => {
                // The refusal started no read; the same product is admitted
                // again once credits return.
                let inputs = batch.inputs_mut();
                inputs.backpressure = Some(returns);
                inputs.state.awaiting_reads = true;
                return Ok(false);
            }
            Err(error) => return Err(error),
        }
    }
    let inputs = batch.inputs_mut();
    inputs.state.started = plan.input_products.len();
    inputs.state.awaiting_reads = false;
    inputs.backpressure = None;
    Ok(true)
}

/// Read a latent trajectory version into the slot of its one consumer,
/// whose scheduler parameters name the pages and the version. The reads'
/// handoff fences are recorded into `job`.
fn import_latent(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    export: &TensorExport,
    readers: &[usize],
    job: &Job,
) -> Result<Input> {
    let [index] = readers else {
        return Err(Error::Invalid(
            "latent transfer must have one consumer".into(),
        ));
    };
    let call = &plan.calls[*index];
    let params = plan
        .latent_params
        .iter()
        .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
        .ok_or_else(|| Error::Invalid("latent transfer has no scheduler parameters".into()))?;
    let device = component_device(backend, &call.component)?;
    let stream = backend.transfer_stream(call.code, device)?;
    let slot = backend
        .requests()
        .peek(export.product.request_key.request_id.0)
        .filter(|request| request.key() == export.product.request_key)
        .map(|request| request.slot())
        .ok_or_else(|| Error::Invalid("latent transfer has no request slot".into()))?;
    let parts = backend.parts();
    let storage = parts
        .storage
        .ok_or_else(|| Error::Unsupported("latent imports require rank storage".into()))?;
    let mut image = parts
        .image
        .ok_or_else(|| Error::Invalid("latent transfer requires physical latent storage".into()))?;
    let (pool, latent) = image.latent_pool()?;
    let input = latents::import(
        pool,
        latent,
        &storage.transports,
        export,
        params,
        slot,
        job.on(stream.as_ref()),
    )?;
    Ok(Input::Latent { input, device })
}

/// Reserve every cache import: each supplied or resident transfer is copied
/// into the pages of its one install call's slot. An accepted import is
/// recorded at once, so a later refusal abandons it with the batch.
fn import_cache(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    plan: &BatchPlan,
) -> Result<()> {
    let resident = batch.inputs().resident.clone();
    if plan.cache_inputs.is_empty() && resident.is_empty() {
        return Ok(());
    }
    let mut consumers: HashMap<BufferId, usize> = HashMap::new();
    for call in installs(plan) {
        if let Some(source) = call.cache_input {
            *consumers.entry(source).or_default() += 1;
        }
    }
    let transfers: Vec<Arc<CacheTransfer>> = plan
        .cache_inputs
        .iter()
        .cloned()
        .map(Arc::new)
        .chain(resident)
        .collect();
    if transfers
        .iter()
        .any(|transfer| consumers.get(&transfer.source) != Some(&1))
    {
        return Err(Error::Invalid(
            "cache input requires one installation consumer".into(),
        ));
    }

    let parts = backend.parts();
    let storage = parts
        .storage
        .ok_or_else(|| Error::Unsupported("cache input requires rank storage".into()))?;
    if storage.transports.is_empty() {
        return Err(Error::Unsupported(
            "cross-call input requires a configured transport".into(),
        ));
    }
    let transports = Arc::clone(&storage.transports);
    let prefix_cache = storage
        .prefix_cache
        .as_mut()
        .ok_or_else(|| Error::Invalid("cache input requires physical cache storage".into()))?;
    for transfer in transfers {
        let source = transfer.source;
        // Starts were installed before preparation; the request pool is the
        // sole source of the consumer's slot.
        let slot = parts.requests.get(source.owner.request_id.0)?.slot() as u32;
        let tables = prefix_cache.tables().for_batch(slot, &plan.block_tables)?;
        let initialized = plan
            .new_cache_units
            .iter()
            .filter(|allocation| allocation.request_pool_idx == slot)
            .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
            .collect();
        let import = prefix_cache.reserve_import(
            transfer,
            slot,
            tables,
            initialized,
            Arc::clone(&transports),
        )?;
        batch
            .inputs_mut()
            .state
            .inputs
            .insert(source, Input::Cache(import));
    }
    Ok(())
}

/// Arrange a readiness notice for `submission` once what stopped the
/// batch's preparation can advance: returned read credits, retired storage
/// hazards, or finished reads and cache copies. Nothing is subscribed when
/// another runtime's inputs stopped the batch.
pub(crate) fn subscribe(
    backend: &WorkerBackend,
    batch: &WorkerBatch,
    submission: &Arc<Submission>,
) -> Result<()> {
    let inputs = batch.inputs();
    if !inputs.waiting || inputs.state.closed() {
        return Ok(());
    }
    let weak: Weak<Submission> = Arc::downgrade(submission);
    let wake = move || {
        if let Some(submission) = weak.upgrade() {
            submission.notify_ready();
        }
    };

    if let Some(returns) = inputs.backpressure {
        let storage = backend
            .rank_storage()
            .ok_or(Error::State("refused imports lost their rank storage"))?;
        let callback: Notify = Arc::new(wake);
        // The return sequence closes the gap between the refusal and this
        // subscription, including a return on a transfer thread.
        if let Some(callback) = storage
            .transports
            .capacity()
            .notify_reads_returned(callback, returns)
        {
            callback();
        }
        return Ok(());
    }

    // One snapshot of everything still pending; only its last arrival
    // notifies, and closing the inputs suppresses it.
    let mut hazards = Vec::new();
    if !inputs.state.submitted {
        hazards.extend(
            inputs
                .state
                .dependencies
                .iter()
                .map(|Hazard(retirement)| Arc::clone(retirement)),
        );
    }
    let mut reads = Vec::new();
    let mut imports = Vec::new();
    for input in inputs.state.inputs.values() {
        match input {
            Input::Tensor { read, .. } => reads.extend(read.reads()),
            Input::Cache(import) => imports.push(import),
            Input::Latent { input, .. } => reads.extend(input.reads().iter().cloned()),
            Input::Borrowed => {}
        }
    }
    let wait = inputs
        .state
        .wait(hazards.len() + reads.len() + imports.len());
    let arrive: Notify = {
        let wait = Arc::clone(&wait);
        Arc::new(move || {
            if wait.arrive() {
                wake();
            }
        })
    };
    for hazard in hazards {
        if let Some(callback) = hazard.subscribe(Arc::clone(&arrive)) {
            callback();
        }
    }
    for read in reads {
        read.add_done_callback(Arc::clone(&arrive));
    }
    for import in imports {
        let arrive = Arc::clone(&arrive);
        import.subscribe(Box::new(move || arrive()));
    }
    // The extra arrival completes registration.
    arrive();
    Ok(())
}

/// The next job publishing the batch's imported products before its
/// runtime executes, or `None` when nothing is left to issue: predicates
/// first, then the imports some call that executes reads (a call whose host
/// predicate is false reads nothing). Each import is published on its
/// consumers' stream; consumers order after its fence.
///
/// The predicates that decide which imports are read are host-gated: the
/// first job publishes the imported ones and reads them, and its
/// completion ([`complete`]) decides what the second job publishes.
pub(crate) fn forward(backend: &mut WorkerBackend, batch: &mut WorkerBatch) -> Result<Option<Job>> {
    let job = Job::new();
    match record(backend, batch, &job) {
        Ok(true) => Ok(Some(job)),
        Ok(false) => Ok(None),
        Err(error) if job.is_empty() => Err(error),
        Err(error) => {
            batch.inputs_mut().completing = Completing::Failing(error);
            Ok(Some(job))
        }
    }
}

/// Record the next job of the imports' completion in `job`; whether the
/// job is to be performed.
fn record(backend: &mut WorkerBackend, batch: &mut WorkerBatch, job: &Job) -> Result<bool> {
    let plan = Arc::clone(batch.shared_plan());
    let imported: HashSet<&TensorRef> = plan
        .calls
        .iter()
        .filter_map(|call| call.predicate.as_ref())
        .collect();
    let readers = consumers(&plan);

    let skipped = match std::mem::take(&mut batch.inputs_mut().completing) {
        Completing::Unstarted => {
            check_resident(backend, batch, &plan)?;
            if plan.input_products.is_empty() {
                return Ok(false);
            }
            for export in &plan.input_products {
                if imported.contains(&export.product) {
                    publish(backend, batch, &plan, &export.product, job)?;
                }
            }

            // The calls whose host-gated predicate decides whether an
            // import that is no predicate is read.
            let mut deciding: Vec<usize> = Vec::new();
            for export in &plan.input_products {
                if imported.contains(&export.product) {
                    continue;
                }
                for &index in readers.get(&export.product).into_iter().flatten() {
                    let gated = plan.calls[index]
                        .predicate
                        .as_ref()
                        .is_some_and(|predicate| predicate.dtype == DType::U8);
                    if gated && !deciding.contains(&index) {
                        deciding.push(index);
                    }
                }
            }
            if !deciding.is_empty() {
                for &index in &deciding {
                    read_predicate(backend, &plan.calls[index], job)?;
                }
                batch.inputs_mut().completing = Completing::Predicates(deciding);
                return Ok(true);
            }
            HashMap::new()
        }
        Completing::Decided(skipped) => skipped,
        state @ (Completing::Predicates(_) | Completing::Publishing | Completing::Failing(_)) => {
            batch.inputs_mut().completing = state;
            return Err(Error::Invariant(
                "an input completion job was prepared before the last completed".into(),
            ));
        }
    };

    for export in &plan.input_products {
        let product = &export.product;
        if imported.contains(product) {
            continue;
        }
        let read = readers
            .get(product)
            .into_iter()
            .flatten()
            .any(|index| !skipped.get(index).copied().unwrap_or(false));
        if read {
            publish(backend, batch, &plan, product, job)?;
        }
    }
    batch.inputs_mut().completing = Completing::Publishing;
    Ok(!job.is_empty())
}

/// Apply the completion of the job [`forward`] prepared last; whether the
/// imports are complete.
pub(crate) fn complete(batch: &mut WorkerBatch, completion: Completion) -> Result<bool> {
    let state = std::mem::take(&mut batch.inputs_mut().completing);
    if let Completing::Failing(error) = state {
        return Err(error);
    }
    let mut completion = completion.check().map_err(|failure| failure.error)?;
    match state {
        Completing::Predicates(calls) => {
            let mut skipped = HashMap::with_capacity(calls.len());
            for index in calls {
                skipped.insert(index, !flag(&mut completion)?);
            }
            batch.inputs_mut().completing = Completing::Decided(skipped);
            Ok(false)
        }
        Completing::Publishing => Ok(true),
        Completing::Unstarted | Completing::Decided(_) | Completing::Failing(_) => Err(
            Error::Invariant("an input completion has no job on its lane".into()),
        ),
    }
}

/// Cache descriptors may have become resident since the imports were admitted;
/// an import must still agree with the resident export it installs.
fn check_resident(
    backend: &mut WorkerBackend,
    batch: &WorkerBatch,
    plan: &BatchPlan,
) -> Result<()> {
    let sources: Vec<BufferId> = installs(plan).filter_map(|call| call.cache_input).collect();
    if sources.is_empty() {
        return Ok(());
    }
    let parts = backend.parts();
    let Some(prefix_cache) = parts
        .storage
        .and_then(|storage| storage.prefix_cache.as_mut())
    else {
        return Ok(());
    };
    for source in sources {
        if let (Some(import), Some(existing)) = (
            batch.inputs().cache_import(source),
            prefix_cache.resident(source),
        ) && existing.as_ref() != import.export().as_ref()
        {
            return Err(Error::Invalid(
                "prepared cache import conflicts with its resident export".into(),
            ));
        }
    }
    Ok(())
}

/// Record in `job` the read of `call`'s host-gated (U8) predicate after
/// its producer's fence, on the stream serving the call; the job's
/// completion carries its byte.
fn read_predicate(backend: &WorkerBackend, call: &Call, job: &Job) -> Result<()> {
    let predicate = call
        .predicate
        .as_ref()
        .ok_or(Error::State("a host-gated call has a predicate"))?;
    let device = component_device(backend, &call.component)?;
    let stream = backend.transfer_stream(call.code, device)?;
    let storage = backend
        .rank_storage()
        .ok_or_else(|| Error::Unsupported("predicates require rank storage".into()))?;
    request_flag(
        &storage.tensors,
        call,
        predicate,
        device.cuda(),
        job.on(stream.as_ref()),
    )
}

/// Record in `job` the publication of one imported product on its
/// consumers' stream.
fn publish(
    backend: &mut WorkerBackend,
    batch: &WorkerBatch,
    plan: &BatchPlan,
    product: &TensorRef,
    job: &Job,
) -> Result<()> {
    let kind = consumer_kind(plan, product)?;
    match batch.inputs().state.inputs.get(&product.buffer_id()) {
        Some(Input::Tensor { read, device }) => {
            let stream = backend.transfer_stream(kind, *device)?;
            tensors(backend)?.complete_import(read, job.on(stream.as_ref()))
        }
        // An imported trajectory version becomes its slot's committed one.
        Some(Input::Latent { input, device }) => {
            let stream = backend.transfer_stream(kind, *device)?;
            let (pool, _) = latent_pool(backend)?;
            latents::adopt(pool, input, job.on(stream.as_ref()))
        }
        // Borrowed inputs stay with their host consumers; cache imports are
        // adopted by their install calls.
        _ => Ok(()),
    }
}

/// The latent pool trajectory transfers use.
fn latent_pool(
    backend: &mut WorkerBackend,
) -> Result<(
    &mut crate::storage::LatentPool,
    &crate::storage::LatentStorage,
)> {
    let latents = backend
        .parts()
        .image
        .ok_or_else(|| Error::Invalid("latent transfer requires physical latent storage".into()))?
        .latents
        .ok_or_else(|| Error::Unsupported("image trajectories require a latent pool".into()))?;
    Ok((&mut latents.pool, &latents.storage))
}

/// The call kind of a product's consumers, whose stream publishes it.
fn consumer_kind(plan: &BatchPlan, product: &TensorRef) -> Result<CallKind> {
    plan.calls
        .iter()
        .find(|call| {
            call.tensor_inputs()
                .chain(call.predicate.as_ref())
                .any(|input| input == product)
        })
        .map(|call| call.code)
        .ok_or_else(|| Error::Invalid("a supplied input product has no consumer".into()))
}

fn tensors(backend: &WorkerBackend) -> Result<&ProductStore> {
    backend
        .rank_storage()
        .map(|storage| &storage.tensors)
        .ok_or_else(|| Error::Unsupported("product imports require rank storage".into()))
}

/// End the batch's import leases and abandon what it did not consume: reads
/// of unpublished imports are cancelled, unadopted cache imports release their
/// pages once their physical accesses retire. Attempts every release before
/// returning the first failure.
pub(crate) fn close(backend: &mut WorkerBackend, batch: &mut WorkerBatch) -> Result<()> {
    let plan = Arc::clone(batch.shared_plan());
    let inputs = batch.inputs_mut();
    if !inputs.state.close() {
        return Ok(());
    }
    let taken: Vec<Input> = inputs
        .state
        .inputs
        .drain()
        .map(|(_, input)| input)
        .collect();
    inputs.state.dependencies.clear();
    let mut result = Ok(());
    let mut reads: HashMap<MemoryKind, Vec<Arc<ProductRead>>> = HashMap::new();
    for input in taken {
        match input {
            Input::Tensor { read, device } => reads.entry(device).or_default().push(read),
            Input::Latent { input, .. } => {
                let closed = if input.adopted() {
                    input.reads().iter().try_for_each(|read| read.close())
                } else {
                    latent_pool(backend).and_then(|(pool, _)| latents::abandon(pool, &input))
                };
                result = result.and(closed);
            }
            Input::Cache(import) => {
                if !import.adopted() {
                    let abandoned = backend
                        .rank_storage_mut()
                        .and_then(|storage| storage.prefix_cache.as_mut())
                        .ok_or(Error::Invariant("a cache import outlived its cache".into()))
                        .and_then(|prefix_cache| prefix_cache.abandon_import(&import));
                    result = result.and(abandoned);
                }
            }
            Input::Borrowed => {}
        }
    }
    let kind = plan.calls.first().map(|call| call.code);
    for (device, reads) in reads {
        // The leases end on the stream that published them.
        let completed = kind
            .map(|kind| backend.transfer_stream(kind, device))
            .transpose()
            .map(Option::flatten)
            .and_then(|stream| tensors(backend)?.complete_reads(&reads, stream.as_ref(), &[]));
        result = result.and(completed);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_products_carry_their_value_range() -> Result<()> {
        let tensor = uniserve_ipc::TensorTransfer {
            shape: vec![1],
            locations: Vec::new(),
        };
        let image = TransferHandle::DeviceProduct {
            height: 4,
            width: 2,
            value_range: "signed_unit".into(),
            tensor: tensor.clone(),
        };
        assert_eq!(
            metadata(&image)?,
            Some(Metadata::Image {
                height: 4,
                width: 2,
                value_range: Some(ValueRange::SignedUnit),
            })
        );
        let plain = TransferHandle::DeviceProduct {
            height: 0,
            width: 0,
            value_range: String::new(),
            tensor,
        };
        assert_eq!(metadata(&plain)?, None);
        Ok(())
    }

    #[test]
    fn concurrent_dependencies_notify_once_unless_the_inputs_closed() -> Result<()> {
        let inputs = BatchInputs::default();
        let wait = inputs.wait(8);
        // The registering caller's own arrival is the extra one.
        assert!(!wait.arrive());

        let notified = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    let wait = Arc::clone(&wait);
                    scope.spawn(move || usize::from(wait.arrive()))
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| {
                    thread
                        .join()
                        .map_err(|_| Error::State("input completion thread panicked"))
                })
                .sum::<Result<usize>>()
        })?;
        assert_eq!(notified, 1);

        let pending = inputs.wait(1);
        assert!(!pending.arrive());
        assert!(inputs.close());
        assert!(!inputs.close());
        assert!(!pending.arrive());
        Ok(())
    }
}
