//! The transfer runtime: tensor product transfers, KV exports and KV
//! installs (`CallKind::Transfer`).
//!
//! - **Tensor** (`TransferMode::Tensor`): each call copies its one input
//!   product (local, or imported from another rank or worker before the
//!   batch executed, see [`super::inputs`]) into its one output product and
//!   publishes the copy through the rank's export mechanisms, so other
//!   components, ranks and workers can import it ([`tensor`]).
//! - **KV export** (`TransferMode::KvExport`): each call publishes its
//!   request's visible KV extent ([`kv`]). On a rank with an image runtime
//!   the image runtime serves it as an image's conditioning in its own
//!   batches; the transfer runtime serves it elsewhere.
//! - **KV install** (`TransferMode::KvInstall`): each call adopts the KV
//!   import its batch prepared into its slot's pages ([`kv`]).
//!
//! A batch runs like the other runtimes' ([`batch`]): its calls bind their
//! outputs and completion flags, host predicates skip calls, the transfers
//! run on the batch's stream, and every owner commits together. KV
//! transfers run on the stream of the component's text prefill, ordered
//! with the text batches that read and write the same pages; tensor
//! transfers run on the stream serving them on the component's device.

mod batch;
mod cache;
pub(crate) mod kv;
mod tensor;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{
    Batch as BatchPlan, BufferId, CallKind, ForwardStats, RequestKey, TransferMode,
};

use super::super::backend::{WorkerBackend, WorkerBatch};
use super::inputs::{Inputs, component_device};
use super::launch::{Launched, error_code, failed_output};
use crate::error::WorkerError;
use crate::executor::BatchResult;
use crate::executor::lifecycle::RequestStorage;
use crate::ffi::Device;
use crate::storage::OutputPool;
use crate::worker::Lanes;
use crate::{Error, Result};

use batch::{Failure, TransferBatch};

/// The transfer runtime of a rank: the lane streams its transfers and
/// imports run on, and its output storage.
pub(crate) struct TransferRuntime {
    /// Borrowed views of the lane stream serving each call kind on each
    /// CUDA device; the lanes outlive the runtime.
    streams: HashMap<(CallKind, i32), Stream>,
    outputs: OutputPool,
}

impl TransferRuntime {
    /// The runtime of a rank with `devices`, serving `kinds` on the streams
    /// of `lanes`, with `queue_depth` batches of at most `max_calls` calls
    /// in flight.
    pub(crate) fn start(
        lanes: &Lanes,
        devices: &[i32],
        kinds: &[CallKind],
        queue_depth: usize,
        max_calls: usize,
    ) -> Result<Self> {
        let mut streams = HashMap::new();
        for &device in devices {
            for &kind in kinds {
                if let Some(stream) = lanes.stream_for(kind, DeviceId(device)) {
                    // SAFETY: the lanes own the stream and close after the
                    // runtimes (see `Worker::close`).
                    let borrowed = unsafe { Stream::borrowed(DeviceId(device), stream.handle()?) }?;
                    streams.insert((kind, device), borrowed);
                }
            }
        }
        Ok(Self {
            streams,
            // One completion word per call; transfers read nothing back.
            outputs: OutputPool::new(
                devices.first().copied(),
                queue_depth.max(1) * 2,
                max_calls.max(1),
            ),
        })
    }

    /// The stream serving `kind` calls on `device`; none for host devices.
    pub(crate) fn stream(&self, kind: CallKind, device: Device) -> Result<Option<Stream>> {
        let Device::Cuda(index) = device else {
            return Ok(None);
        };
        let stream = self.streams.get(&(kind, index)).ok_or_else(|| {
            Error::Unsupported(format!(
                "no execution stream serves {} on cuda:{index}",
                kind.as_str()
            ))
        })?;
        // SAFETY: a view of a lane stream that outlives the runtime.
        Ok(Some(unsafe {
            Stream::borrowed(stream.device(), stream.raw())
        }?))
    }

    pub(crate) fn set_wake(&self, _wake: Option<uniserve_ipc::Wake>) {}

    /// Release output storage after the lanes drained.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        self.outputs.begin_close();
        if !aborted {
            self.outputs.finish_close();
        }
        Ok(())
    }
}

impl RequestStorage for TransferRuntime {
    fn admit_slots(&mut self, _slots: &[usize]) -> Result<()> {
        Ok(())
    }

    fn release_buffers(&mut self, _buffers: &std::collections::HashSet<BufferId>) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &std::collections::HashSet<RequestKey>,
        _retained: &std::collections::HashSet<BufferId>,
    ) -> Result<()> {
        Ok(())
    }

    fn retirement_ready(
        &mut self,
        _buffers: &std::collections::HashSet<BufferId>,
        _requests: &std::collections::HashSet<RequestKey>,
        _retained: &std::collections::HashSet<BufferId>,
    ) -> Result<bool> {
        Ok(true)
    }

    fn retire_slots(&mut self, _slots: &[usize]) -> Result<()> {
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        self.outputs.reap();
        Ok(())
    }
}

/// Execute a batch of `kind` calls.
pub(crate) fn execute(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<(), WorkerError> {
    if kind == CallKind::Transfer(TransferMode::KvExport) && backend.image_mut().is_some() {
        return super::image::execute(backend, batch, kind);
    }
    let plan = Arc::clone(batch.shared_plan());
    let predecessors = batch.predecessors().to_vec();
    let failure = match run(backend, &plan, batch.inputs(), kind) {
        Ok(launched) => {
            batch.launch(launched);
            return Ok(());
        }
        Err(failure) => failure,
    };
    let mut error = WorkerError::classify(failure.error, Some("transfer batch"));
    if failure.committed {
        // Products, KV directories and pending calls may already be
        // visible; the rank cannot roll them back.
        error.fatal = true;
    }
    if error.fatal {
        return Err(error);
    }
    tracing::warn!(
        target: "uniserve_worker.execution",
        "transfer batch {} failed: {error}",
        plan.batch_id
    );
    let output = failed_output(
        backend.requests(),
        &plan,
        &predecessors,
        error_code(error.code),
    )?;
    batch.complete(BatchResult {
        output,
        media: Vec::new(),
    });
    Ok(())
}

/// Reserve, transfer and commit one batch.
fn run(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    inputs: &Inputs,
    kind: CallKind,
) -> std::result::Result<Launched, Failure> {
    let started = Instant::now();
    let first = plan
        .calls
        .first()
        .ok_or_else(|| Error::Invariant("a transfer batch has calls".into()))?;
    if plan
        .calls
        .iter()
        .any(|call| call.component != first.component)
    {
        return Err(Error::Invalid("a transfer batch serves one component".into()).into());
    }
    let kv = kind != CallKind::Transfer(TransferMode::Tensor);
    let (device, stream) = if kv {
        // KV transfers read and write the pages text batches use, on their
        // stream.
        let text = backend
            .parts()
            .text
            .ok_or_else(|| Error::Unsupported("KV transfers require the text runtime".into()))?;
        let (device, stream, _) = text.placement(&first.component)?;
        text.order().wait(stream.as_ref())?;
        (device, stream)
    } else {
        let device = component_device(backend, &first.component)?;
        (device.cuda_index(), backend.transfer_stream(kind, device)?)
    };

    let parts = backend.parts();
    let (Some(runtime), Some(storage)) = (parts.transfer, parts.storage) else {
        return Err(Error::Unsupported("transfers require rank storage".into()).into());
    };
    let mut batch = TransferBatch::open(
        &mut runtime.outputs,
        storage,
        parts.requests,
        parts.info,
        parts.rank,
        plan,
        device,
        stream,
        started,
    )?;
    let result = (|| -> std::result::Result<(), Failure> {
        batch.bind_flags()?;
        if kv {
            let text = parts.text.as_deref().ok_or_else(|| {
                Error::Unsupported("KV transfers require the text runtime".into())
            })?;
            batch.reserve_cache(text.unit_column(&first.component)?, inputs)?;
        }
        match kind {
            CallKind::Transfer(TransferMode::Tensor) => batch.copy_products()?,
            CallKind::Transfer(TransferMode::KvExport) => batch.export_cache()?,
            CallKind::Transfer(TransferMode::KvInstall) => batch.install_cache(inputs)?,
            _ => {
                return Err(Error::Unsupported(format!(
                    "{} is served by another runtime",
                    kind.as_str()
                ))
                .into());
            }
        }
        batch.commit()
    })();
    match result {
        Ok(()) => {
            let stream = batch.stream.as_ref().map(|stream| {
                // SAFETY: a view of the lane stream the batch ran on.
                unsafe { Stream::borrowed(stream.device(), stream.raw()) }
            });
            let mut stats = ForwardStats::default();
            stats.component_us.insert(
                "transfer_execute".into(),
                batch.started.elapsed().as_micros() as u64,
            );
            let launched = Launched {
                outputs: batch.calls.into_iter().map(|state| state.output).collect(),
                buffer: batch.buffer,
                started,
                stats,
                timer: "transfer_materialize",
                host: None,
            };
            if kv && let Some(text) = parts.text {
                // Later text batches on other streams order after the
                // installed pages and verified lengths.
                let stream = stream.transpose().map_err(|error| Failure {
                    error: error.into(),
                    committed: true,
                })?;
                text.order_mut()
                    .record(stream.as_ref())
                    .map_err(|error| Failure {
                        error,
                        committed: true,
                    })?;
            }
            Ok(launched)
        }
        Err(failure) if failure.committed => Err(failure),
        Err(failure) => {
            if let Err(cleanup) = batch.discard() {
                // Storage the cleanup could not retire stays reserved; the
                // rank cannot reuse it safely.
                return Err(Failure {
                    error: cleanup,
                    committed: true,
                });
            }
            Err(failure)
        }
    }
}
