//! One batch in flight: the skeleton every capability's batch is built on,
//! and the dispatch of a batch's jobs to the capability that serves it.
//!
//! A batch carries calls of one kind on one component. Its forward is a
//! sequence of jobs of its lane: [`forward`] prepares the next one and
//! [`complete`] applies its completion. The first job completes the batch's
//! imported inputs ([`crate::executor::inputs`]); the others belong to what
//! serves the kind: the causal LM and diffusion LM runners, an image model's
//! batches ([`crate::executor::image`]), a video model's media batches
//! ([`crate::executor::video`]) and transfers ([`crate::executor::transfer`]).
//! No forward issues its own stream work: the jobs of every batch are
//! operations its owner recorded.
//!
//! Every capability's batch goes through the same steps, which this module
//! implements once:
//!
//! 1. **Open** ([`Opening::open`]). Bind every call's pending output to its
//!    request and record in the opening job the read of every host-gated
//!    predicate. The job's completion decides which calls are active
//!    ([`Opening::predicates`]); a skipped call only publishes false flags.
//! 2. **Lease** ([`Batch::lease`]). Lease the batch's output storage and
//!    record the start of its device work.
//! 3. **Flags** ([`Batch::bind_flags`]). Bind every call's completion flag;
//!    a skipped call publishes false at once.
//! 4. **Seal** ([`Batch::end_reads`], [`Batch::set_flags`],
//!    [`Batch::validate`], [`Batch::seal`]). End the batch's read leases,
//!    publish the executed calls' flags, validate every owner's update and
//!    seal the lease, whose fences retire the batch's storage accesses.
//! 5. **Commit** ([`Batch::commit`]). Once the job that carries the batch's
//!    device work completed, its writes become visible and its calls
//!    pending. A failure before this point retires everything the batch
//!    reserved ([`Batch::discard`]); a failure at or past it is fatal.
//! 6. **Launch and materialization** ([`Batch::launch`], [`materialize`]).
//!    Once the lease's fences complete, every call's result is decoded and
//!    its progress accepted.
//!
//! A failed batch reports its calls failed under the failure's code
//! ([`fail`]); a fatal failure stops the rank. A kind no capability of the
//! rank serves is reported unsupported ([`unsupported`]), which fails the
//! batch with `UnsupportedSetup` and leaves the rank serving. A capability
//! whose results need host work after the device finished (encoding
//! decoded images, publishing what host media tasks wrote) attaches it to
//! its launch as a [`HostStage`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use uniserve_core::CallId;
use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{
    Batch as BatchPlan, BatchOutput, Call, CallKind, CallStatus, DType, ErrorCode, ForwardMode,
    ForwardStats, MediaCall, RequestKey, TensorRef, WorkerInfo,
};

use crate::config::placement::Capability;
use crate::error::WorkerError;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::products::ProductOwners;
use crate::executor::{BatchResult, PendingOutput, request_output};
use crate::ffi::MemoryKind;
use crate::lane::{Completion, Job, Recorder};
use crate::request::RequestPool;
use crate::storage::{
    OutputLease, OutputPlacement, OutputPool, ProductBuffer, ProductRead, ProductStore,
};
use crate::stream::{EventPool, Issue};
use crate::transfer::Retirement;
use crate::{Error, Result, lock};

/// Where a batch's forward stands between its jobs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Its imported inputs are not resident yet.
    #[default]
    Inputs,
    /// Its capability's forward has jobs to run.
    Forward,
    /// Nothing is left for its lane.
    Done,
}

/// A batch's failure, the step it happened in, and whether it happened at
/// or past the commit point, after which the rank cannot roll back what
/// became visible.
pub(crate) struct Failure {
    pub(crate) error: Error,
    pub(crate) phase: &'static str,
    pub(crate) committed: bool,
}

impl Failure {
    /// A failure before the commit point, in `phase`.
    pub(crate) fn new(phase: &'static str) -> impl Fn(Error) -> Self {
        move |error| Self {
            error,
            phase,
            committed: false,
        }
    }

    /// A failure at or past the commit point, in `phase`.
    pub(crate) fn fatal(phase: &'static str) -> impl Fn(Error) -> Self {
        move |error| Self {
            error,
            phase,
            committed: true,
        }
    }
}

/// One call's state through its batch: what every capability tracks, and
/// `forward`, what the call contributes to its capability's forward.
pub(crate) struct CallState<F = ()> {
    pub(crate) output: PendingOutput,
    /// The request slot of the call's request.
    pub(crate) slot: usize,
    /// False when the call's host-gated predicate skipped it.
    pub(crate) active: bool,
    pub(crate) completion_write: Option<ProductBuffer>,
    pub(crate) forward: F,
}

/// A batch whose outputs are bound, before its output storage is leased:
/// its host-gated predicates decide first which calls are active.
pub(crate) struct Opening<F = ()> {
    calls: Vec<CallState<F>>,
    /// The calls whose predicate the opening job reads, in the order of its
    /// reads.
    predicated: Vec<usize>,
    /// This rank reports the component's host results.
    reports: bool,
    started: Instant,
}

impl<F: Default> Opening<F> {
    /// Bind the outputs of `plan`'s calls to their requests' live epochs and
    /// slots, and record in `job` the read of every host-gated predicate of
    /// a call `gated` selects. Each predicate is read after its producer's
    /// fence; its read lease ends behind the read, and the job's completion
    /// carries one byte per predicate.
    #[expect(clippy::too_many_arguments, reason = "the owners a batch opens on")]
    pub(crate) fn open(
        requests: &mut RequestPool,
        tensors: &ProductStore,
        info: &WorkerInfo,
        rank: usize,
        plan: &BatchPlan,
        device: Option<DeviceId>,
        job: Recorder<'_>,
        gated: impl Fn(&Call) -> bool,
    ) -> Result<Self> {
        let started = Instant::now();
        let first = plan
            .calls
            .first()
            .ok_or_else(|| Error::Invariant("a batch has calls".into()))?;
        let reports = reports_output(info, rank, &first.component)?;

        let calls: Vec<_> = plan
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id))
            .collect();
        let bound = requests.bind_calls(&calls)?;
        let mut calls = Vec::with_capacity(plan.calls.len());
        for (index, (call, request)) in plan.calls.iter().zip(&bound).enumerate() {
            calls.push(CallState {
                output: PendingOutput::new(call, request.progress()?, index)?,
                slot: request.slot(),
                active: true,
                completion_write: None,
                forward: F::default(),
            });
        }

        let mut predicated = Vec::new();
        for (index, call) in plan.calls.iter().enumerate() {
            let Some(predicate) = call
                .predicate
                .as_ref()
                .filter(|predicate| predicate.dtype == DType::U8 && gated(call))
            else {
                continue;
            };
            request_flag(tensors, call, predicate, device, job)?;
            predicated.push(index);
        }
        Ok(Self {
            calls,
            predicated,
            reports,
            started,
        })
    }
}

impl<F> Opening<F> {
    /// The calls' states.
    pub(crate) fn calls(&self) -> &[CallState<F>] {
        &self.calls
    }

    /// The calls' states, for a capability that sets what they contribute
    /// to its forward.
    pub(crate) fn calls_mut(&mut self) -> &mut [CallState<F>] {
        &mut self.calls
    }

    /// Whether the opening job reads predicates, so the batch continues
    /// only once that job completed.
    pub(crate) fn reads_predicates(&self) -> bool {
        !self.predicated.is_empty()
    }

    /// Apply the opening job's completion: a call whose predicate read
    /// false is skipped.
    pub(crate) fn predicates(&mut self, completion: &mut Completion) -> Result<()> {
        for index in std::mem::take(&mut self.predicated) {
            if !flag(completion)? {
                let state = &mut self.calls[index];
                state.active = false;
                state.output.output.status = CallStatus::Predicated;
            }
        }
        Ok(())
    }
}

/// A batch whose output storage is leased: its calls and what it
/// reserved. It borrows nothing, so it waits between its jobs as it is.
pub(crate) struct Batch<F = ()> {
    pub(crate) calls: Vec<CallState<F>>,
    /// The batch's output lease: its calls' completion words and payloads.
    pub(crate) buffer: OutputLease,
    /// The products the batch writes, visible once it commits.
    pub(crate) writes: Vec<ProductBuffer>,
    /// The products the batch reads, leased until its read fences.
    pub(crate) reads: Vec<Arc<ProductRead>>,
    /// Signals the physical end of the batch's storage accesses.
    pub(crate) retirement: Arc<Retirement>,
    /// The stream and device the batch's device work runs on.
    pub(crate) stream: Option<Stream>,
    pub(crate) device: Option<DeviceId>,
    /// This rank reports the component's host results.
    pub(crate) reports: bool,
    pub(crate) started: Instant,
}

impl<F> Batch<F> {
    /// Lease the opened batch's output storage, `words` words for its calls'
    /// completions and payloads, and record the start of its device work in
    /// `job`.
    pub(crate) fn lease(
        opening: Opening<F>,
        outputs: &mut OutputPool,
        words: usize,
        stream: Option<Stream>,
        device: Option<DeviceId>,
        events: &EventPool,
        job: &Job,
    ) -> Result<Self> {
        let Opening {
            calls,
            reports,
            started,
            ..
        } = opening;
        let devices: Vec<DeviceId> = device.into_iter().collect();
        let buffer = outputs.lease(calls.len(), words.max(1), &devices, false)?;
        if let Some(stream) = &stream
            && let Err(error) = lock(&buffer).begin_device(device, job.on(Some(stream)), events)
        {
            lock(&buffer).abandon();
            return Err(error);
        }
        Ok(Self {
            calls,
            buffer,
            writes: Vec::new(),
            reads: Vec::new(),
            retirement: Arc::new(Retirement::default()),
            stream,
            device,
            reports,
            started,
        })
    }

    /// Where the batch issues its stream work: `job`, on its stream.
    pub(crate) fn issue<'j>(&'j self, job: &'j Job) -> Recorder<'j> {
        job.on(self.stream.as_ref())
    }

    /// The device the batch's products live on.
    pub(crate) fn target(&self) -> MemoryKind {
        self.device.map_or(MemoryKind::Host, MemoryKind::Cuda)
    }

    /// Indices of the calls that execute.
    pub(crate) fn active(&self) -> Vec<usize> {
        (0..self.calls.len())
            .filter(|&index| self.calls[index].active)
            .collect()
    }

    /// The request slot of every call's request, for relayed products.
    pub(crate) fn slots(&self, plan: &BatchPlan) -> HashMap<RequestKey, usize> {
        plan.calls
            .iter()
            .zip(&self.calls)
            .map(|(call, state)| (call.request_key, state.slot))
            .collect()
    }

    /// Bind every call's completion flag; a skipped call publishes false
    /// now.
    pub(crate) fn bind_flags(
        &mut self,
        plan: &BatchPlan,
        tensors: &ProductStore,
        job: &Job,
    ) -> Result<()> {
        let target = self.target();
        let flags: Vec<(TensorRef, MemoryKind)> = plan
            .calls
            .iter()
            .filter_map(|call| call.completion_output.clone())
            .map(|flag| (flag, target))
            .collect();
        if flags.is_empty() {
            return Ok(());
        }
        let slots = self.slots(plan);
        let bound = tensors.bind_outputs(
            &flags,
            OutputPlacement {
                request_slots: Some(&slots),
                ..OutputPlacement::default()
            },
        )?;
        let owners: HashMap<_, usize> = plan
            .calls
            .iter()
            .enumerate()
            .filter_map(|(index, call)| Some((call.completion_output.as_ref()?.buffer_id(), index)))
            .collect();
        for write in bound {
            self.writes.push(write.clone());
            let index = *owners
                .get(&write.id())
                .ok_or_else(|| Error::Invariant("a completion flag names no call".into()))?;
            let state = &mut self.calls[index];
            if !state.active {
                tensors.write_scalar(&write, 0, job.on(self.stream.as_ref()))?;
            }
            state.completion_write = Some(write);
        }
        Ok(())
    }

    /// Record the end of the batch's read leases in `job`. A read ends at
    /// the fence of `after`'s write it fed when every read has one,
    /// otherwise at a reader fence.
    pub(crate) fn end_reads(
        &mut self,
        tensors: &ProductStore,
        job: &Job,
        after: &[ProductBuffer],
    ) -> Result<()> {
        let reads = std::mem::take(&mut self.reads);
        let after = if after.len() == reads.len() {
            after
        } else {
            &[]
        };
        tensors.complete_reads(&reads, job.on(self.stream.as_ref()), after)
    }

    /// Record the completion flags of the executed calls: they become true.
    pub(crate) fn set_flags(&self, tensors: &ProductStore, job: &Job) -> Result<()> {
        for state in self.calls.iter().filter(|state| state.active) {
            if let Some(write) = &state.completion_write {
                tensors.write_scalar(write, 1, job.on(self.stream.as_ref()))?;
            }
        }
        Ok(())
    }

    /// Check every call's captured output and the batch's writes before
    /// the commit point.
    pub(crate) fn validate(&self, plan: &BatchPlan, tensors: &ProductStore) -> Result<()> {
        for (state, call) in self.calls.iter().zip(&plan.calls) {
            state.output.validate_output(call)?;
        }
        tensors.validate_writes(&self.writes)
    }

    /// Seal the lease in `job` and complete the batch's retirement (the
    /// physical end of its storage accesses) once the lease's fences
    /// complete. The executor learns of their completion from the job it
    /// submits for them
    /// ([`crate::executor::backend::WorkerBackend::await_outputs`]), which
    /// also wakes the service.
    pub(crate) fn seal(&self, events: &EventPool, job: &Job) -> Result<()> {
        let stream = job.on(self.stream.as_ref());
        // A lease of host storage is sealed on no stream.
        let streams = if stream.stream().is_some() {
            vec![stream]
        } else {
            Vec::new()
        };
        let fences = lock(&self.buffer)
            .seal(&streams, events)?
            .unwrap_or_default();
        let lease = Arc::clone(&self.buffer);
        let retirement = Arc::clone(&self.retirement);
        stream.defer(
            events,
            fences,
            Box::new(move || {
                lock(&lease).copies_finished();
                crate::transfer::notify(retirement.complete(Ok(()))?);
                Ok(())
            }),
        )
    }

    /// The pending state of every call of `plan`: its request, its call
    /// and whether it advances its request's state.
    pub(crate) fn pending(plan: &BatchPlan) -> Vec<(RequestKey, CallId, bool)> {
        plan.calls
            .iter()
            .map(|call| (call.request_key, call.call_id, call.advances_state()))
            .collect()
    }

    /// Make the batch's writes visible and its calls pending, once the job
    /// carrying its device work completed. Every failure here is past the
    /// commit point.
    pub(crate) fn commit(
        &mut self,
        plan: &BatchPlan,
        tensors: &ProductStore,
        requests: &mut RequestPool,
    ) -> std::result::Result<(), Failure> {
        let fatal = Failure::fatal("batch commit");
        for state in &mut self.calls {
            state.output.reports_output = self.reports;
        }
        let writes = std::mem::take(&mut self.writes);
        tensors.commit_writes(&writes).map_err(&fatal)?;
        requests.add_pending(&Self::pending(plan)).map_err(&fatal)
    }

    /// Retire every reservation after a failure before the commit point,
    /// recording in `job` the fences that end them: storage returns once
    /// those fences complete.
    pub(crate) fn discard(
        &mut self,
        tensors: &ProductStore,
        events: &EventPool,
        job: &Job,
    ) -> Result<()> {
        let reads = std::mem::take(&mut self.reads);
        let completed = tensors.complete_reads(&reads, job.on(self.stream.as_ref()), &[]);
        let writes = std::mem::take(&mut self.writes);
        let abandoned = tensors.abandon_writes(&writes);
        let sealed = if lock(&self.buffer).sealed() {
            Ok(())
        } else {
            self.seal(events, job)
        };
        lock(&self.buffer).abandon();
        completed.and(abandoned).and(sealed)
    }

    /// Hand the committed batch to materialization: `stats` are its
    /// forward's, `timer` names its materialization in the statistics and
    /// `host` is host work its results need after the device finished.
    pub(crate) fn launch(
        self,
        stats: ForwardStats,
        timer: &'static str,
        host: Option<Box<dyn HostStage>>,
    ) -> Launched {
        Launched {
            outputs: self.calls.into_iter().map(|state| state.output).collect(),
            buffer: self.buffer,
            started: self.started,
            stats,
            timer,
            host,
        }
    }
}

/// Record the read of a host-gated (U8) predicate after its producer's
/// fence: the job's completion carries its byte, which [`flag`] decodes.
pub(crate) fn request_flag(
    tensors: &ProductStore,
    call: &Call,
    predicate: &TensorRef,
    device: Option<DeviceId>,
    job: Recorder<'_>,
) -> Result<()> {
    let read = tensors.consume(predicate, call.call_id, device.map(MemoryKind::Cuda), job)?;
    // One byte of a committed product, which the read lease keeps until
    // the reader's fence that ends it below.
    job.read(read.value().address(), 1);
    tensors.complete_reads(&[read], job, &[])
}

/// The predicate the next read of `completion` carries
/// ([`request_flag`]).
pub(crate) fn flag(completion: &mut Completion) -> Result<bool> {
    match completion.read()?[..] {
        [0] => Ok(false),
        [1] => Ok(true),
        _ => Err(Error::Invalid(
            "call predicate is not a canonical boolean".into(),
        )),
    }
}

/// A batch launched on the device, awaiting its output fences.
pub(crate) struct Launched {
    pub(crate) outputs: Vec<PendingOutput>,
    pub(crate) buffer: OutputLease,
    pub(crate) started: Instant,
    pub(crate) stats: ForwardStats,
    /// The component timer name of the batch's materialization.
    pub(crate) timer: &'static str,
    /// Host work that completes the calls' results after the device.
    pub(crate) host: Option<Box<dyn HostStage>>,
}

/// Host work a launched batch finishes once its output fences completed,
/// such as encoding generated media read back into its output lease, or
/// publishing the products host tasks wrote.
pub(crate) trait HostStage {
    /// Advance the work after the fences completed; whether every call's
    /// host result is ready.
    fn poll(&mut self) -> Result<bool>;

    /// Attach the finished host results to the outputs of `plan`'s calls
    /// (before they resolve). What the host work wrote is published through
    /// `products`: the batch committed, so such a product is committed and
    /// its exports registered at once.
    fn finish(
        &mut self,
        plan: &BatchPlan,
        products: ProductOwners<'_>,
        outputs: &mut [PendingOutput],
    ) -> Result<()>;

    /// Withdraw work of a batch that will not be materialized.
    fn cancel(&mut self);
}

/// The rank that reports a component's host results: its first member.
pub(crate) fn reports_output(info: &WorkerInfo, rank: usize, component: &str) -> Result<bool> {
    match info.components.iter().find(|value| value.name == component) {
        Some(value) => Ok(value.config.ranks.first() == Some(&rank)),
        None if info.world_size == 1 => Ok(true),
        None => Err(Error::Unsupported(format!(
            "component {component:?} has no output rank"
        ))),
    }
}

/// The wire code of a batch failure classified as `code`.
fn error_code(code: crate::ErrorCode) -> ErrorCode {
    match code {
        crate::ErrorCode::ResourceError => ErrorCode::ResourceExhausted,
        crate::ErrorCode::ComputeError => ErrorCode::ComputeError,
        crate::ErrorCode::InvariantViolation | crate::ErrorCode::FatalWorkerFailure => {
            ErrorCode::Internal
        }
        _ => ErrorCode::InvalidCall,
    }
}

/// Every call of `plan` fails with `code`, reporting its accepted progress.
fn failed_output(
    requests: &RequestPool,
    plan: &BatchPlan,
    predecessors: &[Option<CallId>],
    code: ErrorCode,
) -> Result<BatchOutput> {
    let completions = plan
        .calls
        .iter()
        .zip(predecessors)
        .map(|(call, predecessor)| {
            let key = call.request_key;
            // A failed first call has no accepted parent; a stale epoch never
            // reports a replacement request's progress.
            let progress = requests
                .peek(key.request_id.0)
                .filter(|request| request.key() == key && predecessor.is_some())
                .map(|request| request.progress())
                .transpose()?
                .unwrap_or_default();
            let mut output = request_output(key, call.call_id, call.code, progress)?;
            output.status = CallStatus::Error;
            output.error_code = Some(code);
            Ok(output)
        })
        .collect::<Result<_>>()?;
    Ok(BatchOutput {
        batch_id: plan.batch_id,
        completions,
        products: Vec::new(),
        worker_exec_us: None,
        forward_stats: None,
    })
}

/// Report a failed batch: a fatal failure, or one at or past the commit
/// point, stops the rank; any other completes the batch with its calls
/// failed under the failure's code, and the rank keeps serving.
pub(crate) fn fail(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    failure: Failure,
) -> std::result::Result<(), WorkerError> {
    let mut error = WorkerError::classify(failure.error, Some(failure.phase));
    if failure.committed {
        // Products and pending calls may already be visible; the rank cannot
        // roll them back.
        error.fatal = true;
    }
    if error.fatal {
        return Err(error);
    }
    let plan = Arc::clone(batch.shared_plan());
    tracing::warn!(
        "batch {} failed: {error}{}",
        plan.batch_id,
        error
            .traceback()
            .map(|trace| format!("\n{trace}"))
            .unwrap_or_default()
    );
    let output = failed_output(
        backend.requests(),
        &plan,
        batch.predecessors(),
        error_code(error.code),
    )?;
    batch.complete(BatchResult {
        output,
        media: Vec::new(),
    });
    batch.set_stage(Stage::Done);
    Ok(())
}

/// The error of a call kind whose capability is not served on this rank.
pub(crate) fn unsupported(kind: CallKind, capability: &str) -> WorkerError {
    WorkerError::classify(
        Error::Unsupported(format!(
            "{} is served by the {capability} capability",
            kind.as_str()
        )),
        None,
    )
}

/// Materialize a launched batch once its output fences completed: finish
/// its host work, decode every call's result and accept or cancel its
/// pending state. Returns `None` while the device or the host work is still
/// producing.
pub(crate) fn materialize(
    requests: &mut RequestPool,
    products: ProductOwners<'_>,
    plan: &BatchPlan,
    launched: &mut Launched,
) -> Result<Option<BatchResult>> {
    if lock(&launched.buffer).ready_at().is_none() {
        return Ok(None);
    }
    if let Some(host) = &mut launched.host {
        if !host.poll()? {
            return Ok(None);
        }
        host.finish(plan, products, &mut launched.outputs)?;
        launched.host = None;
    }
    lock(&launched.buffer).readback()?;
    // Every call resolves against its request's accepted progress before
    // any call of the batch is accepted.
    for output in &mut launched.outputs {
        let key = output.output.request_key;
        let parent = requests
            .peek(key.request_id.0)
            .filter(|request| request.key() == key)
            .map(|request| request.progress())
            .transpose()?
            .unwrap_or_default();
        let mut buffer = lock(&launched.buffer);
        if let Err(error) = output.resolve(parent, &mut buffer) {
            // A malformed capture fails its call only.
            tracing::error!("{} result materialization failed: {error}", launched.timer);
            output.output.status = CallStatus::Error;
            output.output.error_code = Some(ErrorCode::ComputeError);
            output.progress = parent;
            buffer.discard(output.row);
        }
    }
    for output in &launched.outputs {
        if output.resolved() {
            output.accept(requests)?;
        } else {
            output.cancel(requests)?;
        }
    }

    // The calls' results move into the batch's output: the launched batch
    // ends here. A call whose materialization failed reports its error
    // status alone, and only a successful call publishes its products.
    let outputs = std::mem::take(&mut launched.outputs);
    let mut completions = Vec::with_capacity(outputs.len());
    let mut products = Vec::new();
    let mut media = Vec::new();
    for output in outputs {
        let resolved = output.resolved();
        let (completion, exports, source) = output.into_parts();
        if resolved {
            if completion.status == CallStatus::Ok {
                products.extend(exports);
            }
            media.extend(source);
        }
        completions.push(completion);
    }
    let mut stats = std::mem::take(&mut launched.stats);
    let elapsed = launched.started.elapsed().as_micros() as u64;
    stats.component_us.insert(launched.timer.into(), elapsed);
    Ok(Some(BatchResult {
        output: BatchOutput {
            batch_id: plan.batch_id,
            completions,
            products,
            worker_exec_us: Some(elapsed),
            forward_stats: Some(stats),
        },
        media,
    }))
}

/// Cancel a launched batch's unresolved calls; its storage retires through
/// the fences it recorded.
fn abandon_launched(requests: &mut RequestPool, mut launched: Launched) -> Result<()> {
    if let Some(host) = &mut launched.host {
        host.cancel();
    }
    for output in &launched.outputs {
        output.cancel(requests)?;
    }
    lock(&launched.buffer).abandon();
    Ok(())
}

/// Orders one runner family's per-slot state across the streams of its
/// runners: each batch waits for the previous batch recorded on another
/// stream.
#[derive(Default)]
pub(crate) struct StreamOrder {
    last: Option<(usize, Arc<uniserve_cuda::Event>)>,
}

impl StreamOrder {
    /// Order `stream` after the previous batch on another stream.
    pub(crate) fn wait(&self, stream: impl Issue) -> Result<()> {
        let Some(raw) = stream.stream().map(Stream::raw) else {
            return Ok(());
        };
        if let Some((previous, event)) = &self.last
            && *previous != raw
        {
            stream.wait(event)?;
        }
        Ok(())
    }

    /// Record the end of this batch's work on `stream` for other streams.
    pub(crate) fn record(&mut self, stream: impl Issue) -> Result<()> {
        let Some(raw) = stream.stream().map(Stream::raw) else {
            return Ok(());
        };
        let event = match self.last.take() {
            Some((_, event)) => event,
            None => Arc::new(uniserve_cuda::Event::new(
                uniserve_cuda::EventFlags::default(),
            )),
        };
        stream.record(&event)?;
        self.last = Some((raw, event));
        Ok(())
    }
}

/// Materialize launched work whose device results are ready.
pub(crate) fn poll(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    launched: &mut Launched,
) -> std::result::Result<Option<BatchResult>, WorkerError> {
    let parts = backend.parts();
    // A batch launches only on a rank whose storage owners it opened on.
    let storage = parts
        .storage
        .ok_or_else(|| Error::Invariant("the rank has no storage owners".into()))?;
    let products = ProductOwners {
        store: &storage.tensors,
        transports: &storage.transports,
    };
    Ok(materialize(parts.requests, products, plan, launched)?)
}

/// Cancel launched work that will not be materialized.
pub(crate) fn abandon(
    backend: &mut WorkerBackend,
    launched: Launched,
) -> std::result::Result<(), WorkerError> {
    Ok(abandon_launched(backend.requests_mut(), launched)?)
}

/// The next job of a batch's forward at its stage: the completion of its
/// imported inputs, then the jobs of the runtime serving `kind`. `None`
/// when the stage advanced without needing the lane.
pub(crate) fn forward(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<Option<Job>, WorkerError> {
    use CallKind::Forward;

    match batch.stage() {
        Stage::Inputs => match crate::executor::inputs::forward(backend, batch) {
            Ok(Some(job)) => Ok(Some(job)),
            // Nothing is left to issue for the imports.
            Ok(None) => imported(backend, batch, Ok(())).map(|()| None),
            Err(error) => imported(backend, batch, Err(error)).map(|()| None),
        },
        Stage::Forward => match kind {
            Forward(ForwardMode::Prefill | ForwardMode::Decode | ForwardMode::Verify) => {
                crate::runner::causal_lm::forward(backend, batch, kind)
            }
            Forward(ForwardMode::TokenDenoising) => {
                crate::runner::diffusion_lm::forward(backend, batch, kind)
            }
            CallKind::Transfer(_) => crate::executor::transfer::forward(backend, batch, kind),
            _ if records(backend, kind) => crate::executor::video::forward(backend, batch),
            // A video model's rank records its media calls ([`records`]);
            // no other rank serves these kinds.
            CallKind::Media(
                MediaCall::MediaReading
                | MediaCall::VideoEncoding
                | MediaCall::AudioEncoding
                | MediaCall::Muxing
                | MediaCall::VideoDecoding
                | MediaCall::AudioDecoding,
            ) => Err(unsupported(kind, "video")),
            // The remaining kinds are an image model's: its encoders, its
            // denoiser and its decoder.
            _ => crate::executor::image::forward(backend, batch, kind),
        },
        Stage::Done => Ok(None),
    }
}

/// Apply the completion of the job [`forward`] prepared last.
pub(crate) fn complete(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
    completion: Completion,
) -> std::result::Result<(), WorkerError> {
    use CallKind::Forward;

    if batch.stage() == Stage::Inputs {
        return match crate::executor::inputs::complete(batch, completion) {
            // The imports' second job follows.
            Ok(false) => Ok(()),
            Ok(true) => imported(backend, batch, Ok(())),
            Err(error) => imported(backend, batch, Err(error)),
        };
    }
    match kind {
        Forward(ForwardMode::Prefill | ForwardMode::Decode | ForwardMode::Verify) => {
            crate::runner::causal_lm::complete(backend, batch, completion)
        }
        Forward(ForwardMode::TokenDenoising) => {
            crate::runner::diffusion_lm::complete(backend, batch, completion)
        }
        CallKind::Transfer(_) => crate::executor::transfer::complete(backend, batch, completion),
        _ if records(backend, kind) => crate::executor::video::complete(backend, batch, completion),
        _ => crate::executor::image::complete(backend, batch, kind, completion),
    }
}

/// Whether `kind` is served by a video model's media batches on this rank
/// ([`crate::executor::video::forward`]): its media calls, on its devices
/// and on its host lane.
fn records(backend: &WorkerBackend, kind: CallKind) -> bool {
    use MediaCall::*;

    matches!(
        kind,
        CallKind::Media(
            MediaReading
                | TextEncoding
                | VisionEncoding
                | LatentEncoding
                | LatentPreparation
                | Denoising
                | VideoDecoding
                | AudioDecoding
                | VideoEncoding
                | AudioEncoding
                | Muxing
        )
    ) && backend.serves_video()
}

/// Continue with the forward once the imported inputs are resident; an
/// import that failed fails the batch's calls.
fn imported(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    completed: crate::Result<()>,
) -> std::result::Result<(), WorkerError> {
    match completed {
        Ok(()) => batch.set_stage(Stage::Forward),
        Err(error) => {
            fail(backend, batch, Failure::new("input import")(error))?;
        }
    }
    Ok(())
}

/// Whether a component's model decodes video.
pub(crate) fn decodes_video(capabilities: impl IntoIterator<Item = Capability>) -> bool {
    capabilities.into_iter().any(|capability| {
        matches!(
            capability,
            Capability::VideoDenoiser | Capability::VideoPostprocessor
        )
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use uniserve_core::{RequestId, SamplingParams};
    use uniserve_cuda::{DeviceId, Event, EventFlags, Stream};
    use uniserve_ipc::{ArRequestParams, BufferAllocation, NewRequest, RequestKey, ShapeBound};

    use super::*;
    use crate::storage::BufferPool;

    /// Blocks `stream` in a host function until the returned sender is
    /// used, so the work queued behind it on any stream that waits for it
    /// stays incomplete meanwhile.
    fn hold(stream: &Stream) -> Result<mpsc::Sender<()>> {
        let (release, released) = mpsc::channel::<()>();
        stream.launch_host_fn(move || {
            let _ = released.recv();
        })?;
        Ok(release)
    }

    /// An event capturing the work issued on `stream` so far.
    fn mark(stream: &Stream) -> Result<Arc<Event>> {
        let event = Arc::new(Event::new(EventFlags::default()));
        event.record(stream)?;
        Ok(event)
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_batch_waits_for_the_previous_batch_recorded_on_another_stream() -> Result<()> {
        let device = DeviceId(0);
        let first = Stream::new(device)?;
        let second = Stream::new(device)?;
        let mut order = StreamOrder::default();
        let release = hold(&first)?;

        // Before any batch recorded its end, a batch waits for nothing: its
        // work completes while the other stream is held.
        order.wait(&second)?;
        let unordered = mark(&second)?;
        second.synchronize()?;
        assert!(unordered.query()?);

        // A batch that ended on the held stream holds up the next batch on
        // the other stream until the held work completed.
        order.record(&first)?;
        order.wait(&second)?;
        let ordered = mark(&second)?;
        std::thread::sleep(Duration::from_millis(50));
        let waited = !ordered.query()?;
        release
            .send(())
            .map_err(|_| Error::State("the held stream was released already"))?;
        second.synchronize()?;
        assert!(waited, "the batch did not wait for the previous one");
        assert!(ordered.query()?);
        first.synchronize()?;
        Ok(())
    }

    const HOST: MemoryKind = MemoryKind::Host;

    fn key(request: u64) -> RequestKey {
        RequestKey::new(1, RequestId(request), 1)
    }

    /// A text request in the slot of its id (slots count from one).
    fn admission(request: u64) -> NewRequest {
        NewRequest {
            request_key: key(request),
            request_pool_idx: request as u32,
            ar: Some(ArRequestParams {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                initial_position: 0,
                canvas: None,
            }),
            image: None,
            diffusion: None,
            video: None,
            prompt_token_ids: Vec::new(),
            input_images: 0,
        }
    }

    /// The host-gated predicate the request's previous call produced.
    fn predicate(request: u64) -> TensorRef {
        TensorRef {
            request_key: key(request),
            producer_call_id: CallId::new(1, request as u32),
            output_index: 0,
            generation: 1,
            dtype: DType::U8,
            shape_bound: ShapeBound::default(),
        }
    }

    /// A prefill call of the request in batch 2, gated by `predicate`.
    fn call(request: u64, predicate: Option<TensorRef>) -> Call {
        Call {
            request_key: key(request),
            call_id: CallId::new(2, request as u32),
            coordinates: Default::default(),
            component: "text".into(),
            code: CallKind::Forward(ForwardMode::Prefill),
            bounds: Default::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            consumer_slots: Vec::new(),
            token_input: None,
            token_output: None,
            vision_inputs: Vec::new(),
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,
            predicate,
            rng: None,
            sampling_state: None,
            input_token_ids: Vec::new(),
            readout: None,
            canvas: None,
            input_image: None,
            cache_input: None,
            cache_output: None,
        }
    }

    /// Commit `value` as the one byte of `product` in host storage.
    fn publish(store: &ProductStore, product: &TensorRef, offset: u64, value: u8) -> Result<()> {
        let allocations = HashMap::from([(
            product.buffer_id(),
            BufferAllocation {
                buffer: product.buffer_id(),
                offset,
                bytes: 1,
            },
        )]);
        let mut bound = store.bind_outputs(
            &[(product.clone(), HOST)],
            OutputPlacement {
                allocations: Some(&allocations),
                ..OutputPlacement::default()
            },
        )?;
        let buffer = bound.remove(0);
        // SAFETY: the product's own host byte, without readers.
        unsafe { *(buffer.view().address() as *mut u8) = value };
        store.produced(&buffer, &[], None, None)?;
        store.commit_writes(std::slice::from_ref(&buffer))
    }

    /// Open a batch of one call per request, each gated by a predicate
    /// holding the request's byte of `values` (`None`: no predicate), and
    /// apply the opening job's completion.
    fn open(values: &[Option<u8>]) -> Result<Opening> {
        let mut requests = RequestPool::new(values.len())?;
        let store = ProductStore::new(
            1024,
            1024,
            vec![HOST],
            values.len() + 1,
            1,
            BufferPool::allocate(1024, &[HOST])?,
            EventPool::new(),
        )?;
        let mut calls = Vec::new();
        for (request, value) in (1..).zip(values) {
            requests.start(admission(request))?;
            let gate = match value {
                Some(value) => {
                    let product = predicate(request);
                    publish(&store, &product, request * 64, *value)?;
                    Some(product)
                }
                None => None,
            };
            calls.push(call(request, gate));
        }
        let plan = BatchPlan::new(2, Vec::new(), calls);

        let job = Job::new();
        let mut opening = Opening::open(
            &mut requests,
            &store,
            &WorkerInfo::default(),
            0,
            &plan,
            None,
            job.on(None),
            |_| true,
        )?;
        assert_eq!(
            opening.reads_predicates(),
            values.iter().any(Option::is_some)
        );
        let mut completion = job.perform().check().map_err(|failure| failure.error)?;
        opening.predicates(&mut completion)?;
        Ok(opening)
    }

    #[test]
    fn a_batch_skips_exactly_the_calls_whose_predicate_is_false() -> Result<()> {
        let opening = open(&[Some(1), Some(0), None, Some(0), Some(1)])?;
        let active: Vec<bool> = opening.calls().iter().map(|state| state.active).collect();
        assert_eq!(active, [true, false, true, false, true]);
        let statuses: Vec<CallStatus> = opening
            .calls()
            .iter()
            .map(|state| state.output.output.status)
            .collect();
        assert_eq!(
            statuses,
            [
                CallStatus::Ok,
                CallStatus::Predicated,
                CallStatus::Ok,
                CallStatus::Predicated,
                CallStatus::Ok
            ]
        );

        // A batch without host-gated predicates has nothing to read.
        let opening = open(&[None, None])?;
        assert!(opening.calls().iter().all(|state| state.active));
        Ok(())
    }

    #[test]
    fn a_predicate_that_is_no_canonical_boolean_fails_the_batch() {
        assert!(matches!(open(&[Some(1), Some(2)]), Err(Error::Invalid(_))));
    }
}
