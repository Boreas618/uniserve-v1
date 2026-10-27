//! One batch in flight: the skeleton every batch is built on, and the
//! dispatch of a batch's jobs to what serves its kind.
//!
//! A batch carries calls of one kind on one component. Its forward is a
//! sequence of jobs: [`forward`] prepares the next one and [`complete`] applies
//! its completion. The first job completes the batch's imported inputs
//! ([`crate::executor::inputs`]); the others belong to what serves the kind:
//! the executor's computations ([`crate::executor::computations`]) and
//! transfers ([`crate::executor::transfer`]). No forward issues its own stream
//! work: the jobs of every batch are operations its owner recorded.
//!
//! Every batch goes through the same steps, which this module implements
//! once:
//!
//! 1. **Open** ([`Opening::open`]). Bind every call's pending output to its
//!    request; a call of a request the rank refused is inactive and only
//!    publishes false flags.
//! 2. **Lease** ([`Batch::lease`], [`Batch::lease_predicates`]). Lease the
//!    batch's output storage, record the start of its device work and lease
//!    every call's `uint8` predicate on the device: the executor skips a call
//!    whose predicate is false on the device, and the host learns it from the
//!    call's predicate word once the batch's fences completed. Nothing waits
//!    for a predicate on the host.
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
//! A failed batch reports its calls failed under the failure's code ([`fail`]);
//! a fatal failure stops the rank. A kind no executor computation of the rank
//! serves is reported unsupported ([`unserved`]), which fails the batch with
//! `UnsupportedSetup` and leaves the rank serving. A batch whose results need
//! host work after the device finished (encoding decoded images, publishing
//! what host media tasks wrote) attaches it to its launch as a [`HostStage`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use cuda::{CopyBatch, DeviceId, SourceOrder, Stream};
use ipc::CallId;
use ipc::{
    Batch as BatchPlan, BatchOutput, Call, CallKind, CallStatus, DType, ErrorCode, ForwardStats,
    RequestKey, TensorRef,
};

use crate::config::placement::GroupPlacement;
use crate::error::WorkerError;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::products::ProductOwners;
use crate::executor::{BatchResult, PendingOutput, request_output};
use crate::ffi::MemoryKind;
use crate::job::{Completion, Job};
use crate::memory::allocation::StorageView;
use crate::memory::{
    OutputLease, OutputPlacement, OutputPool, ProductBuffer, ProductRead, ProductStore,
};
use crate::request::RequestPool;
use crate::stream::{EventPool, Issue};
use crate::transfer::Retirement;
use crate::{Error, Result, lock};

/// Where a batch's forward stands between its jobs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Its imported inputs are not resident yet.
    #[default]
    Inputs,
    /// Its forward has jobs to run.
    Forward,
    /// Nothing is left for its forward.
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

/// One call's state through its batch: what every batch tracks, and `forward`,
/// what the call contributes to its computation's forward.
pub(crate) struct CallState<F = ()> {
    pub(crate) output: PendingOutput,
    /// The request slot of the call's request.
    pub(crate) slot: usize,
    /// False when the rank refused the call's request (its output then reports
    /// the refusal). A call whose predicate is false is active: its executor
    /// skips it on the device ([`Self::predicate`]).
    pub(crate) active: bool,
    /// The call's leased `uint8` predicate on the batch's device
    /// ([`Batch::lease_predicates`]): the executor leaves every effect of the
    /// call undone when it is false, and the call's completion flag and
    /// predicate word are copies of it.
    pub(crate) predicate: Option<StorageView>,
    pub(crate) completion_write: Option<ProductBuffer>,
    pub(crate) forward: F,
}

impl<F> CallState<F> {
    /// Whether the rank refused the call's request, so the call reports the
    /// refusal and runs nothing, host work included.
    pub(crate) fn refused(&self) -> bool {
        self.output.output.error_message.is_some()
    }
}

/// A batch whose outputs are bound, before its output storage is leased.
pub(crate) struct Opening<F = ()> {
    calls: Vec<CallState<F>>,
    /// This rank reports the component's host results.
    reports: bool,
    started: Instant,
}

impl<F: Default> Opening<F> {
    /// Bind the outputs of `plan`'s calls to their requests' live epochs and
    /// slots. A call of a request this rank refused is inactive and reports
    /// [`ErrorCode::InputError`] with the refusal's reason.
    pub(crate) fn open(
        requests: &mut RequestPool,
        placement: &GroupPlacement,
        rank: usize,
        plan: &BatchPlan,
    ) -> Result<Self> {
        let started = Instant::now();
        let first = plan
            .calls
            .first()
            .ok_or_else(|| Error::Invariant("a batch has calls".into()))?;
        let reports = placement.reports_output(rank, &first.component)?;

        let calls: Vec<_> = plan
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id))
            .collect();
        let bound = requests.bind_calls(&calls)?;
        let mut calls = Vec::with_capacity(plan.calls.len());
        for (index, (call, request)) in plan.calls.iter().zip(&bound).enumerate() {
            let mut output = PendingOutput::new(call, request.progress()?, index)?;
            let refusal = request.refusal()?;
            if let Some(reason) = &refusal {
                output.output.status = CallStatus::Error;
                output.output.error_code = Some(ErrorCode::InputError);
                output.output.error_message = Some(reason.to_string());
            }
            calls.push(CallState {
                output,
                slot: request.slot(),
                active: refusal.is_none(),
                predicate: None,
                completion_write: None,
                forward: F::default(),
            });
        }

        Ok(Self {
            calls,
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

    /// Lease, on the batch's stream behind its producer, the `uint8` predicate
    /// of every active call that `gated` selects, for the executor that skips
    /// the call on the device when it is false ([`CallState::predicate`]), and
    /// copy each into an output word of its own, which the call's result reads
    /// once the batch's fences completed ([`PendingOutput::predicate_word`]).
    /// The leases end with the batch's other reads. Nothing waits for a
    /// predicate on the host.
    pub(crate) fn lease_predicates(
        &mut self,
        plan: &BatchPlan,
        tensors: &ProductStore,
        events: &EventPool,
        job: &Job,
        gated: impl Fn(&Call) -> bool,
    ) -> Result<()> {
        let stream = self.stream.clone();
        let issue = job.on(stream.as_ref());
        let target = self.target();
        let mut copies = CopyBatch::new(SourceOrder::Stream);
        for (index, call) in plan.calls.iter().enumerate() {
            if !self.calls[index].active {
                continue;
            }
            let Some(predicate) = call
                .predicate
                .as_ref()
                .filter(|predicate| predicate.dtype == DType::U8 && gated(call))
            else {
                continue;
            };
            let read = tensors.consume(predicate, call.call_id, Some(target), issue)?;
            self.reads.push(Arc::clone(&read));
            let value = read.value().clone();
            if value.numel() != 1 {
                return Err(Error::Invalid("a call predicate is one byte".into()));
            }
            let (word, address) = lock(&self.buffer).reserve_words(1)?;
            // The byte lands in the low byte of the zeroed little-endian word.
            copies.push(address, value.address(), 1);
            let state = &mut self.calls[index];
            state.output.predicate_word = Some(word);
            state.predicate = Some(value);
        }
        if copies.is_empty() {
            return Ok(());
        }
        if stream.is_some() {
            lock(&self.buffer).begin_copy(self.device, issue, events)?;
            // SAFETY: each source is a leased predicate, kept until the batch's
            // read fences; each destination a reserved word of the batch's
            // lease, read only after its seal's fences.
            unsafe { issue.copy(&copies) }
        } else {
            // SAFETY: as above, in host memory, in the batch's order.
            unsafe { issue.host().copy(&copies) }
        }
    }

    /// Record the completion flags of the executed calls: true, or a copy of
    /// the call's predicate for a call the executor may have skipped on the
    /// device.
    pub(crate) fn set_flags(&self, tensors: &ProductStore, job: &Job) -> Result<()> {
        let issue = job.on(self.stream.as_ref());
        let (mut writes, mut predicates) = (Vec::new(), Vec::new());
        for state in self.calls.iter().filter(|state| state.active) {
            let Some(write) = &state.completion_write else {
                continue;
            };
            match &state.predicate {
                Some(predicate) => {
                    writes.push(write.clone());
                    predicates.push(predicate.clone());
                }
                None => {
                    tensors.write_scalar(write, 1, issue)?;
                }
            }
        }
        tensors.copy_scalars(&writes, &predicates, issue)
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
            let mut output = request_output(key, call.call_id, progress)?;
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
    let output = failed_output(backend.requests(), &plan, batch.predecessors(), error.code)?;
    batch.complete(BatchResult {
        output,
        media: Vec::new(),
    });
    batch.set_stage(Stage::Done);
    Ok(())
}

/// The error of a call kind no executor computation of this rank serves.
pub(crate) fn unserved(kind: CallKind) -> Error {
    Error::Unsupported(format!(
        "no executor computation of this rank serves {} calls",
        kind.as_str()
    ))
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

/// Orders the batches that share per-slot state or storage across the streams
/// they run on (the computations over the prefix cache, the users of a latent
/// ledger): each batch waits for the previous batch recorded on another stream.
#[derive(Default)]
pub(crate) struct StreamOrder {
    last: Option<(usize, Arc<cuda::Event>)>,
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
            None => Arc::new(cuda::Event::new(cuda::EventFlags::default())),
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
/// imported inputs, then the jobs of what serves `kind` (a transfer or an
/// executor computation). `None` when the stage advanced without needing a job.
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
            CallKind::Transfer(_) => crate::executor::transfer::forward(backend, batch, kind),
            _ if backend.executes(kind) => {
                crate::executor::computations::forward(backend, batch, kind)
            }
            // Only an executor computation serves the other kinds: a call no
            // computation of this rank serves fails with its batch, and the
            // rank serves on.
            Forward(_) | CallKind::Media(_) => {
                fail(backend, batch, Failure::new("computations")(unserved(kind))).map(|()| None)
            }
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
    if batch.stage() == Stage::Inputs {
        let completed = crate::executor::inputs::complete(batch, completion);
        return imported(backend, batch, completed);
    }
    match kind {
        CallKind::Transfer(_) => crate::executor::transfer::complete(backend, batch, completion),
        _ => crate::executor::computations::complete(backend, batch, completion),
    }
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use cuda::{DeviceId, Event, EventFlags, Stream};
    use ipc::{
        BufferAllocation, ForwardMode, GenerationParams, NewRequest, RequestKey, ShapeBound,
    };
    use ipc::{RequestId, SamplingParams};

    use super::*;
    use crate::memory::BufferPool;

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
            generation: Some(GenerationParams {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                initial_position: 0,
                generated_token_ids: Vec::new(),
                adapter: 0,
            }),
            image: None,
            diffusion: None,
            video: None,
            prompt_token_ids: Vec::new(),
            input_images: 0,
        }
    }

    /// The `uint8` predicate the request's previous call produced.
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
            kind: CallKind::Forward(ForwardMode::Prefill),
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
            image_index: None,
            sampling_state: None,
            input_token_ids: Vec::new(),
            readout: None,
            canvas: None,
            input_image: None,
            cache_input: None,
            cache_output: None,
            latent_restore: None,
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

    /// The statuses a batch of one call per request resolves to, each call
    /// gated by a predicate holding the request's byte of `values` (`None`: no
    /// predicate): the batch leases its predicates, copies them into its output
    /// words in its job, and resolves from the words read back.
    fn resolved(values: &[Option<u8>]) -> Result<Vec<CallStatus>> {
        let mut requests = RequestPool::new(values.len())?;
        let store = ProductStore::new(
            1024,
            1024,
            vec![HOST],
            values.len() + 1,
            1,
            BufferPool::allocate(1024, &[HOST], &mut crate::memory::LocalMemory)?,
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
        let placement = GroupPlacement {
            world_size: 1,
            ..GroupPlacement::default()
        };

        let opening = Opening::<()>::open(&mut requests, &placement, 0, &plan)?;
        // A predicate decides nothing on the host: every call is active.
        assert!(opening.calls().iter().all(|state| state.active));
        let events = EventPool::new();
        let mut outputs = OutputPool::new(None, 1, 64)?;
        let job = Job::new();
        let mut batch = Batch::lease(opening, &mut outputs, 64, None, None, &events, &job)?;
        batch.lease_predicates(&plan, &store, &events, &job, |_| true)?;
        job.perform().check().map_err(|failure| failure.error)?;

        let mut buffer = lock(&batch.buffer);
        buffer.seal::<Option<&Stream>>(&[], &events)?;
        buffer.readback()?;
        batch
            .calls
            .iter_mut()
            .map(|state| {
                let parent = state.output.progress;
                state.output.resolve(parent, &mut buffer)?;
                Ok(state.output.output.status)
            })
            .collect()
    }

    #[test]
    fn exactly_the_calls_whose_predicate_is_false_report_themselves_predicated() -> Result<()> {
        assert_eq!(
            resolved(&[Some(1), Some(0), None, Some(0), Some(1)])?,
            [
                CallStatus::Ok,
                CallStatus::Predicated,
                CallStatus::Ok,
                CallStatus::Predicated,
                CallStatus::Ok
            ]
        );
        assert_eq!(resolved(&[None, None])?, [CallStatus::Ok, CallStatus::Ok]);
        Ok(())
    }

    #[test]
    fn a_predicate_that_is_no_canonical_boolean_fails_its_call() {
        assert!(matches!(
            resolved(&[Some(1), Some(2)]),
            Err(Error::State(_))
        ));
    }
}
