//! Inputs a batch imports from other calls, prepared before it executes.
//!
//! Every batch reads its tensor inputs from the rank's tensor store, and its
//! trajectories from the computations' latent ledger. Products another rank or
//! worker published reach this rank as descriptors (`Batch::input_products`):
//! tensors are read into the tensor store, latent trajectory versions into the
//! consuming request's pages; cache transfers its install calls name arrive as
//! `Batch::cache_inputs` or are resident exports of this rank. This module
//! materializes them while other batches execute, so a computation consumes
//! imported values exactly like values its own rank produced.
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
//!    ([`admit`]). What an import needs on its consumers' stream, the handoff
//!    fence each of its reads waits on, is recorded into one job of the batch;
//!    the reads are handed to their transport threads from that job's
//!    completion ([`admitted`]), when the fences' records are made. A refusal
//!    for read credits resumes at the refused product once credits return;
//!    accepted imports keep their destinations meanwhile. Cache imports follow
//!    ([`prepare`]), after which nothing is repeatable.
//! 3. **Readiness.** The batch launches once every read and cache copy is
//!    consumable.
//!
//! When the batch executes, [`complete`] publishes each imported product on its
//! consumers' device stream (a new product is committed, a widened shard is
//! fenced over its union, a trajectory version is adopted as its slot's
//! committed one) before the computation consumes it; imports only
//! predicated-off calls read are left unpublished. [`close`] ends the import
//! leases and abandons every import the batch did not consume, on success and
//! on failure alike.
//!
//! [`BatchInputs`] holds the resources a batch retains while its inputs
//! are prepared.

mod hazards;
mod imports;

use super::backend::{WorkerBackend, WorkerBatch};
use super::transfer::latents::{self, LatentInput};
use crate::error::WorkerError;
use crate::executor::Submission;
use crate::ffi::MemoryKind;
use crate::job::{Completion, Job};
use crate::memory::{PendingImport, ProductRead, ProductStore};
use crate::transfer::{Notify, Retirement, fetch};
use crate::{Error, Result};
use ipc::{Batch as BatchPlan, BufferId, CacheTransfer, TensorRef};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use hazards::{Resolved, installs, latent_hazards, require_storage, resolve};
use imports::{consumers, import_cache, import_products};

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
    /// host; importing it would read unused capacity.
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
    /// Preparation stopped on these inputs (not on an inline image's
    /// host preparation).
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
    /// The job in flight publishes the imports the batch's calls read.
    Publishing,
    /// Recording a job failed after it recorded work: the job in flight
    /// performs that work, whose fences are still due, and the completion fails
    /// with this error.
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

/// Admit the batch's product imports that can be admitted now, recording what
/// they issue on their consumers' stream into the returned job. The caller
/// submits a job that holds work, whatever the result, and hands its completion
/// to [`admitted`]: fences recorded by an import that was refused or failed
/// afterwards are still due. With an empty job nothing waits for the builder
/// thread.
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

/// Arrange a readiness notice for `submission` once what stopped the
/// batch's preparation can advance: returned read credits, retired storage
/// hazards, or finished reads and cache copies. Nothing is subscribed when
/// an inline image's host preparation stopped the batch, which notifies
/// `submission` itself.
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
            .worker_storage()
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

/// The next job publishing the batch's imported products before its computation
/// executes, or `None` when nothing is left to issue: every import a call
/// reads, its predicates included. A call its predicate skips on the device
/// reads its imports as any other. Each import is published on its consumers'
/// stream; consumers order after its fence.
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

/// Record the imports' publication in `job`; whether the job is to be
/// performed.
fn record(backend: &mut WorkerBackend, batch: &mut WorkerBatch, job: &Job) -> Result<bool> {
    let plan = Arc::clone(batch.shared_plan());
    match std::mem::take(&mut batch.inputs_mut().completing) {
        Completing::Unstarted => {}
        state @ (Completing::Publishing | Completing::Failing(_)) => {
            batch.inputs_mut().completing = state;
            return Err(Error::Invariant(
                "an input completion job was prepared before the last completed".into(),
            ));
        }
    }
    check_resident(backend, batch, &plan)?;
    if plan.input_products.is_empty() {
        return Ok(false);
    }
    let predicates: HashSet<&TensorRef> = plan
        .calls
        .iter()
        .filter_map(|call| call.predicate.as_ref())
        .collect();
    let readers = consumers(&plan);
    for export in &plan.input_products {
        let product = &export.product;
        let read = predicates.contains(product)
            || readers
                .get(product)
                .is_some_and(|readers| !readers.is_empty());
        if read {
            publish(backend, batch, &plan, product, job)?;
        }
    }
    batch.inputs_mut().completing = Completing::Publishing;
    Ok(!job.is_empty())
}

/// Apply the completion of the job [`forward`] prepared, which completes the
/// imports.
pub(crate) fn complete(batch: &mut WorkerBatch, completion: Completion) -> Result<()> {
    let state = std::mem::take(&mut batch.inputs_mut().completing);
    if let Completing::Failing(error) = state {
        return Err(error);
    }
    completion.check().map_err(|failure| failure.error)?;
    match state {
        Completing::Publishing => Ok(()),
        Completing::Unstarted | Completing::Failing(_) => Err(Error::Invariant(
            "an input completion has no job in flight".into(),
        )),
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

/// Record in `job` the publication of one imported product on its
/// consumers' stream.
fn publish(
    backend: &mut WorkerBackend,
    batch: &WorkerBatch,
    plan: &BatchPlan,
    product: &TensorRef,
    job: &Job,
) -> Result<()> {
    require_consumer(plan, product)?;
    match batch.inputs().state.inputs.get(&product.buffer_id()) {
        Some(Input::Tensor { read, device }) => {
            let stream = backend.transfer_stream(*device)?;
            tensors(backend)?.complete_import(read, job.on(stream.as_ref()))
        }
        // An imported trajectory version becomes its slot's committed one.
        Some(Input::Latent { input, device }) => {
            let stream = backend.transfer_stream(*device)?;
            let (pool, _) = latent_pool(backend)?;
            latents::adopt(pool, input, job.on(stream.as_ref()))
        }
        // Borrowed inputs stay with their host consumers; cache imports are
        // adopted by their install calls.
        _ => Ok(()),
    }
}

/// The latent pool trajectory transfers use: the computations' ledger's.
fn latent_pool(
    backend: &mut WorkerBackend,
) -> Result<(
    &mut crate::memory::LatentPool,
    &crate::memory::LatentStorage,
)> {
    let ledger = backend
        .parts()
        .computations
        .and_then(crate::executor::computations::Computations::ledger_mut)
        .ok_or_else(|| Error::Invalid("latent transfer requires physical latent storage".into()))?;
    Ok(ledger.parts())
}

/// The call kind of a product's consumers, whose stream publishes it.
fn require_consumer(plan: &BatchPlan, product: &TensorRef) -> Result<()> {
    plan.calls
        .iter()
        .find(|call| {
            call.tensor_inputs()
                .chain(call.predicate.as_ref())
                .any(|input| input == product)
        })
        .map(drop)
        .ok_or_else(|| Error::Invalid("a supplied input product has no consumer".into()))
}

fn tensors(backend: &WorkerBackend) -> Result<&ProductStore> {
    backend
        .worker_storage()
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
                        .worker_storage_mut()
                        .and_then(|storage| storage.prefix_cache.as_mut())
                        .ok_or(Error::Invariant("a cache import outlived its cache".into()))
                        .and_then(|prefix_cache| prefix_cache.abandon_import(&import));
                    result = result.and(abandoned);
                }
            }
            Input::Borrowed => {}
        }
    }
    let calls = !plan.calls.is_empty();
    for (device, reads) in reads {
        // The leases end on the stream that published them.
        let completed = calls
            .then(|| backend.transfer_stream(device))
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
