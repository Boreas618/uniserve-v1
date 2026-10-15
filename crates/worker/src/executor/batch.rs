//! One batch in flight: the dispatch of its jobs to the capability that
//! serves it, its launch and its materialization.
//!
//! Every batch carries calls of one kind on one component, and its forward
//! is a sequence of jobs of its lane: [`forward`] prepares the next job and
//! [`complete`] applies its completion. The first job completes the
//! batch's imported inputs ([`crate::executor::inputs`]); the others belong
//! to what serves the kind: the causal LM runner, the diffusion LM runner,
//! an image model's batches ([`crate::executor::image`]), a video model's
//! media batches ([`crate::executor::video`]) and transfers
//! ([`crate::executor::transfer`]). A kind no capability of the rank serves
//! is reported unsupported, which fails the batch with `UnsupportedSetup`
//! and leaves the rank serving. No forward issues its own stream work: the
//! jobs of every batch are operations its owner recorded.
//!
//! A capability reserves a batch's outputs, records its device work and the
//! copies of its result words into an output lease, then hands the batch to
//! materialization as a [`Launched`] value. Once the lease's device fences
//! complete, [`materialize`] reads the words back, decodes every call's
//! result and accepts its progress into the request pool. A capability whose
//! results also need host work after the device finished (encoding decoded
//! images) attaches it as a [`HostStage`].

use super::backend::{WorkerBackend, WorkerBatch};
use crate::config::placement::Capability;
use crate::error::WorkerError;
use crate::executor::{BatchResult, PendingOutput, request_output};
use crate::ffi::Device;
use crate::lane::{Completion, Job, Recorder};
use crate::request::RequestPool;
use crate::storage::{OutputBuffer, OutputLease, TensorStore};
use crate::stream::{EventPool, Issue};
use crate::transfer::Retirement;
use crate::{Error, Result};
use std::sync::{Arc, MutexGuard, PoisonError};
use std::time::Instant;
use uniserve_core::CallId;
use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BatchOutput, Call, CallKind, CallStatus, ErrorCode, ForwardMode,
    ForwardStats, MediaCall, TensorRef, WorkerInfo,
};

/// A forward batch launched on the device, awaiting its output fences.
pub(crate) struct Launched {
    pub(crate) outputs: Vec<PendingOutput>,
    pub(crate) buffer: OutputLease,
    pub(crate) started: Instant,
    pub(crate) stats: ForwardStats,
    /// The runtime's component timer name for materialization.
    pub(crate) timer: &'static str,
    /// Host work that completes the calls' results after the device.
    pub(crate) host: Option<Box<dyn HostStage>>,
}

/// Host work a launched batch finishes once its output fences completed,
/// such as encoding generated media read back into its output lease.
pub(crate) trait HostStage {
    /// Advance the work after the fences completed; whether every call's
    /// host result is ready.
    fn poll(&mut self) -> Result<bool>;

    /// Attach the finished host results to the calls' outputs (before they
    /// resolve), returning nothing else.
    fn finish(&mut self, outputs: &mut [PendingOutput]) -> Result<()>;

    /// Withdraw work of a batch that will not be materialized.
    fn cancel(&mut self);
}

/// Lock a batch's output buffer; a poisoned lock keeps its state.
pub(crate) fn lock(buffer: &OutputLease) -> MutexGuard<'_, OutputBuffer> {
    buffer.lock().unwrap_or_else(PoisonError::into_inner)
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
pub(crate) fn error_code(code: crate::ErrorCode) -> ErrorCode {
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
pub(crate) fn failed_output(
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

/// Record the read of a host-gated (U8) predicate after its producer's
/// fence: the job's completion carries its byte, which [`flag`] decodes.
pub(crate) fn request_flag(
    tensors: &TensorStore,
    call: &Call,
    predicate: &TensorRef,
    device: Option<i32>,
    job: Recorder<'_>,
) -> Result<()> {
    let read = tensors.consume(predicate, call.call_id, device.map(Device::Cuda), job)?;
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

/// Seal the lease on `stream` and complete `retirement` (the physical end
/// of the batch's storage accesses) once its fences complete. The batch's
/// executor learns of their completion from the job it submits for them
/// ([`crate::executor::Backend::await_outputs`]), which also wakes the
/// service.
pub(crate) fn seal(
    events: &EventPool,
    buffer: &OutputLease,
    stream: impl Issue,
    retirement: &Arc<Retirement>,
) -> Result<()> {
    // A lease of host storage is sealed on no stream.
    let streams = if stream.stream().is_some() {
        vec![stream]
    } else {
        Vec::new()
    };
    let fences = lock(buffer).seal(&streams, events)?.unwrap_or_default();
    let lease = Arc::clone(buffer);
    let retirement = Arc::clone(retirement);
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

/// Materialize a launched batch once its output fences completed: decode
/// every call's result and accept or cancel its pending state. Returns
/// `None` while the device is still producing.
pub(crate) fn materialize(
    requests: &mut RequestPool,
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
        host.finish(&mut launched.outputs)?;
        launched.host = None;
    }
    lock(&launched.buffer).readback()?;
    let mut completions = Vec::with_capacity(launched.outputs.len());
    let mut products = Vec::new();
    let mut media = Vec::new();
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
            tracing::error!(
                target: "uniserve_worker.execution",
                "{} result materialization failed: {error}",
                launched.timer
            );
            output.output.status = CallStatus::Error;
            output.output.error_code = Some(ErrorCode::ComputeError);
            output.progress = parent;
            buffer.discard(output.row);
            completions.push(output.output.clone());
            continue;
        }
        let completion = output.result()?;
        // Only a successful call publishes its products.
        if completion.status == CallStatus::Ok {
            products.extend_from_slice(output.products());
        }
        completions.push(completion.clone());
        media.extend(output.take_media());
    }
    for output in &launched.outputs {
        if output.resolved() {
            output.accept(requests)?;
        } else {
            output.cancel(requests)?;
        }
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

/// Orders one runtime's per-slot state across the streams of its runners:
/// each batch waits for the previous batch recorded on another stream.
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

/// Materialize launched work whose device results are ready.
pub(crate) fn poll(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    launched: &mut Launched,
) -> std::result::Result<Option<BatchResult>, WorkerError> {
    Ok(materialize(backend.requests_mut(), plan, launched)?)
}

/// Cancel launched work that will not be materialized.
pub(crate) fn abandon(
    backend: &mut WorkerBackend,
    launched: Launched,
) -> std::result::Result<(), WorkerError> {
    Ok(abandon_launched(backend.requests_mut(), launched)?)
}

/// Fail every call of a batch whose execution failed before any runtime
/// committed work: a fatal failure stops the rank, any other completes the
/// batch with its calls failed under the failure's code.
pub(crate) fn fail(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    error: Error,
    context: &str,
) -> std::result::Result<(), WorkerError> {
    let error = WorkerError::classify(error, Some(context));
    if error.fatal {
        return Err(error);
    }
    let plan = batch.shared_plan();
    tracing::warn!(
        target: "uniserve_worker.execution",
        "batch {} failed: {error}",
        plan.batch_id
    );
    let output = failed_output(
        backend.requests(),
        plan,
        batch.predecessors(),
        error_code(error.code),
    )?;
    batch.complete(BatchResult {
        output,
        media: Vec::new(),
    });
    Ok(())
}

/// The error of a call kind whose runtime is not available on this rank.
pub(crate) fn unsupported(kind: CallKind, runtime: &str) -> WorkerError {
    WorkerError::classify(
        Error::Unsupported(format!(
            "{} is served by the {runtime} runtime",
            kind_name(kind)
        )),
        None,
    )
}

fn kind_name(kind: CallKind) -> &'static str {
    match kind {
        CallKind::Forward(mode) => mode.as_str(),
        CallKind::Media(call) => call.as_str(),
        CallKind::Transfer(mode) => mode.as_str(),
    }
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
            _ if records(backend, kind) => {
                let (plan, borrowed, state) = batch.media_parts();
                let mut parts = backend.media_parts()?;
                Ok(Some(crate::executor::video::forward(
                    &mut parts, plan, borrowed, state,
                )?))
            }
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
        _ if records(backend, kind) => {
            let done = {
                let (_, _, state) = batch.media_parts();
                let mut parts = backend.media_parts()?;
                crate::executor::video::complete(&mut parts, state, completion)?
            };
            if done {
                batch.set_stage(Stage::Done);
            }
            Ok(())
        }
        _ => crate::executor::image::complete(backend, batch, kind, completion),
    }
}

/// Whether the forwards of `kind` on this rank are recorded into jobs
/// ([`batch::forward`]): a video model's media calls, on its devices and
/// on its host lane. The others issue their own stream work.
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
            batch.set_stage(Stage::Done);
            fail(backend, batch, error, "input import")?;
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
