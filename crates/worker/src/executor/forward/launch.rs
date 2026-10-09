//! Launched forward batches and their materialization.
//!
//! A capability runtime reserves a batch's outputs, launches its device
//! work and copies its result words into an output lease, then hands the
//! batch to materialization as a [`Launched`] value. Once the lease's device
//! fences complete, [`materialize`] reads the words back, decodes every
//! call's result and accepts its progress into the request pool. These
//! steps are the same for every runtime that executes forward calls. A
//! runtime whose results also need host work after the device finished
//! (encoding decoded images) attaches it as a [`HostStage`].

use std::sync::{Arc, MutexGuard, PoisonError};
use std::time::Instant;

use uniserve_core::CallId;
use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BatchOutput, Call, CallStatus, ErrorCode, ForwardStats, TensorRef,
    WorkerInfo,
};

use crate::executor::{BatchResult, PendingOutput, request_output};
use crate::ffi::Device;
use crate::request::RequestPool;
use crate::storage::{OutputBuffer, OutputLease, TensorStore};
use crate::stream::EventPool;
use crate::transfer::Retirement;
use crate::{Error, Result};

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

/// Read a host-gated (U8) predicate after its producer's fence.
pub(crate) fn read_flag(
    tensors: &TensorStore,
    call: &Call,
    predicate: &TensorRef,
    device: Option<i32>,
    stream: Option<&Stream>,
) -> Result<bool> {
    let read = tensors.consume(predicate, call.call_id, device.map(Device::Cuda), stream)?;
    let mut value = [0_u8; 1];
    let copied = (|| -> Result<()> {
        match stream {
            Some(stream) => {
                // SAFETY: one byte of a committed product, read after the
                // stream waited on its producer; the host byte outlives the
                // synchronized copy.
                unsafe {
                    uniserve_cuda::memcpy_async(
                        value.as_mut_ptr() as u64,
                        read.value().address(),
                        1,
                        stream,
                    )
                }?;
                stream.synchronize()?;
            }
            // SAFETY: a host product byte.
            None => value[0] = unsafe { *(read.value().address() as *const u8) },
        }
        Ok(())
    })();
    tensors.complete_reads(&[read], stream, &[])?;
    copied?;
    match value[0] {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Error::Invalid(
            "call predicate is not a canonical boolean".into(),
        )),
    }
}

/// Seal the lease on `stream` and complete `retirement` (the physical end
/// of the batch's storage accesses) once its fences complete; wake the
/// service when they do.
pub(crate) fn seal(
    events: &EventPool,
    buffer: &OutputLease,
    stream: Option<&Stream>,
    retirement: &Arc<Retirement>,
) -> Result<()> {
    let streams: Vec<&Stream> = stream.into_iter().collect();
    let fences = lock(buffer).seal(&streams, events)?.unwrap_or_default();
    let notify: Vec<_> = lock(buffer).fences().cloned().collect();
    for event in &notify {
        events.wake_after(event)?;
    }
    let lease = Arc::clone(buffer);
    let retirement = Arc::clone(retirement);
    events.defer(
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
    if lock(&launched.buffer).ready_at()?.is_none() {
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
pub(crate) fn abandon(requests: &mut RequestPool, mut launched: Launched) -> Result<()> {
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
    last: Option<(usize, uniserve_cuda::Event)>,
}

impl StreamOrder {
    /// Order `stream` after the previous batch on another stream.
    pub(crate) fn wait(&self, stream: Option<&Stream>) -> Result<()> {
        let Some(stream) = stream else {
            return Ok(());
        };
        if let Some((previous, event)) = &self.last
            && *previous != stream.raw()
        {
            stream.wait_event(event)?;
        }
        Ok(())
    }

    /// Record the end of this batch's work on `stream` for other streams.
    pub(crate) fn record(&mut self, stream: Option<&Stream>) -> Result<()> {
        let Some(stream) = stream else {
            return Ok(());
        };
        let event = match self.last.take() {
            Some((_, event)) => event,
            None => uniserve_cuda::Event::new(uniserve_cuda::EventFlags::default()),
        };
        event.record(stream)?;
        self.last = Some((stream.raw(), event));
        Ok(())
    }
}
