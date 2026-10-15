//! The transfer runtime: tensor product transfers, KV exports and KV
//! installs (`CallKind::Transfer`).
//!
//! - **Tensor** (`TransferMode::Tensor`): each call copies its one input
//!   product (local, or imported from another rank or worker before the
//!   batch executed, see [`super::inputs`]) into its one output product and
//!   publishes the copy through the rank's export mechanisms, so other
//!   components, ranks and workers can import it ([`tensor`]). A call with
//!   a latent input and output publishes its request's committed trajectory
//!   version from the latent pool instead ([`latents`]).
//! - **KV export** (`TransferMode::KvExport`): each call publishes its
//!   request's visible KV extent ([`kv`]), which an image's latent
//!   preparation and denoising calls name as their conditioning.
//! - **KV install** (`TransferMode::KvInstall`): each call adopts the KV
//!   import its batch prepared into its slot's pages ([`kv`]).
//!
//! A batch is jobs of its lane ([`forward`], [`complete`], see [`batch`]):
//! its calls bind their outputs, a first job reads the host predicates that
//! skip calls, the last job carries the transfers, and every owner commits
//! together at that job's completion. KV transfers run on the stream of the
//! component's text prefill, ordered with the text batches that read and
//! write the same pages; tensor transfers run on the stream serving them on
//! the component's device.

mod batch;
mod cache;
pub(crate) mod kv;
pub(crate) mod latents;
mod tensor;

use std::collections::HashMap;
use std::sync::Arc;

use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{Batch as BatchPlan, BufferId, CallKind, RequestKey, TransferMode};

use super::super::backend::{WorkerBackend, WorkerBatch};
use super::Stage;
use super::inputs::{Inputs, component_device};
use crate::error::WorkerError;
use crate::executor::lifecycle::{LaneJobs, RequestStorage};
use crate::ffi::Device;
use crate::lane::{Completion, Job, Lanes};
use crate::storage::OutputPool;
use crate::{Error, Result};

pub(crate) use batch::Pending;
use batch::{Failure, Opening, TransferBatch, fatal};

/// The transfer runtime of a rank: the lane streams its transfers and
/// imports run on, and its output storage.
pub(crate) struct TransferRuntime {
    /// The lane stream serving each call kind on each CUDA device.
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
                    streams.insert((kind, device), stream.stream()?.clone());
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
        Ok(Some(stream.clone()))
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
    fn admit_slots(&mut self, _slots: &[usize], _jobs: &LaneJobs) -> Result<()> {
        Ok(())
    }

    fn release_buffers(
        &mut self,
        _buffers: &std::collections::HashSet<BufferId>,
        _jobs: &LaneJobs,
    ) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &std::collections::HashSet<RequestKey>,
        _retained: &std::collections::HashSet<BufferId>,
        _jobs: &LaneJobs,
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

    fn retire_slots(&mut self, _slots: &[usize], _jobs: &LaneJobs) -> Result<()> {
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        self.outputs.reap();
        Ok(())
    }
}

/// Whether `kind` transfers KV, on the stream of the component's text
/// prefill.
fn transfers_cache(kind: CallKind) -> bool {
    kind != CallKind::Transfer(TransferMode::Tensor)
}

/// The device and stream the transfers of `plan`'s component run on.
fn placement(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    kind: CallKind,
) -> Result<(Option<i32>, Option<Stream>)> {
    let first = plan
        .calls
        .first()
        .ok_or_else(|| Error::Invariant("a transfer batch has calls".into()))?;
    if plan
        .calls
        .iter()
        .any(|call| call.component != first.component)
    {
        return Err(Error::Invalid(
            "a transfer batch serves one component".into(),
        ));
    }
    if transfers_cache(kind) {
        // KV transfers read and write the pages text batches use, on their
        // stream.
        let text = backend
            .parts()
            .text
            .ok_or_else(|| Error::Unsupported("KV transfers require the text runtime".into()))?;
        let (device, stream, _) = text.placement(&first.component)?;
        Ok((device, stream))
    } else {
        let device = component_device(backend, &first.component)?;
        Ok((device.cuda_index(), backend.transfer_stream(kind, device)?))
    }
}

/// Bind the batch's outputs and record in `job` its order after the text
/// runtime's batches (KV transfers) and the reads of its predicates.
fn open(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    kind: CallKind,
    device: Option<i32>,
    stream: Option<&Stream>,
    job: &Job,
) -> Result<Opening> {
    let parts = backend.parts();
    let Some(storage) = parts.storage else {
        return Err(Error::Unsupported("transfers require rank storage".into()));
    };
    if transfers_cache(kind) {
        let text = parts
            .text
            .ok_or_else(|| Error::Unsupported("KV transfers require the text runtime".into()))?;
        text.order().wait(job.on(stream))?;
    }
    Opening::open(
        parts.requests,
        &storage.tensors,
        parts.info,
        parts.rank,
        plan,
        device,
        job.on(stream),
    )
}

/// Record the opened batch's last job in `job`: its reservations, its
/// transfers and the seal of its lease. A batch that fails here still has
/// what it recorded performed; what it reserved is retired at that job's
/// completion.
#[expect(clippy::too_many_arguments, reason = "the batch's owners and its job")]
fn record(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    inputs: &Inputs,
    kind: CallKind,
    opening: Opening,
    device: Option<i32>,
    stream: Option<&Stream>,
    job: &Job,
) -> Pending {
    let cache = transfers_cache(kind);
    let mut parts = backend.parts();
    let (Some(runtime), Some(storage)) = (parts.transfer, parts.storage) else {
        let error = Error::Unsupported("transfers require rank storage".into());
        return Pending::Discarding(None, error.into());
    };
    let mut batch = match TransferBatch::begin(
        opening,
        &mut runtime.outputs,
        storage,
        parts.requests,
        plan,
        device,
        stream,
        job,
    ) {
        Ok(batch) => batch,
        Err(error) => return Pending::Discarding(None, error.into()),
    };

    let result = (|| -> Result<()> {
        batch.bind_flags()?;
        if cache {
            let text = parts.text.as_deref().ok_or_else(|| {
                Error::Unsupported("KV transfers require the text runtime".into())
            })?;
            let component = &plan.calls[0].component;
            batch.reserve_cache(text.unit_column(component)?, inputs)?;
        }
        match kind {
            CallKind::Transfer(TransferMode::Tensor) => {
                batch.copy_products(parts.image.as_mut())?
            }
            CallKind::Transfer(TransferMode::KvExport) => batch.export_cache()?,
            CallKind::Transfer(TransferMode::KvInstall) => batch.install_cache(inputs)?,
            _ => {
                return Err(Error::Unsupported(format!(
                    "{} is served by another runtime",
                    kind.as_str()
                )));
            }
        }
        batch.seal()?;
        if cache && let Some(text) = parts.text.as_deref_mut() {
            // Later text batches on other streams order after the
            // installed tables, pages and verified lengths.
            text.order_mut().record(batch.issue())?;
        }
        Ok(())
    })();
    match batch.end(result) {
        (recorded, None) => Pending::Committing(Box::new(recorded)),
        (recorded, Some(failure)) => Pending::Discarding(Some(Box::new(recorded)), failure),
    }
}

/// The next job of a transfer batch on its lane, or `None` when the batch
/// advanced without one: it failed and its calls are reported.
pub(crate) fn forward<'a>(
    backend: &'a mut WorkerBackend,
    batch: &'a mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<Option<Job>, WorkerError> {
    let state = match batch.transfer_parts().2.take() {
        Some(Pending::Failed(failure)) => return fail(backend, batch, failure).map(|()| None),
        state => state,
    };
    let (plan, inputs, pending) = batch.transfer_parts();

    let job = Job::new();
    let next = match placement(backend, plan, kind) {
        Err(error) => Pending::Discarding(None, error.into()),
        Ok((device, stream)) => match state {
            None => match open(backend, plan, kind, device, stream.as_ref(), &job) {
                Ok(opening) if opening.predicated() => Pending::Predicates(opening),
                Ok(opening) => record(
                    backend,
                    plan,
                    inputs,
                    kind,
                    opening,
                    device,
                    stream.as_ref(),
                    &job,
                ),
                Err(error) => Pending::Discarding(None, error.into()),
            },
            Some(Pending::Opened(opening)) => record(
                backend,
                plan,
                inputs,
                kind,
                opening,
                device,
                stream.as_ref(),
                &job,
            ),
            Some(state) => {
                // A job of the batch is on the lane: its completion comes
                // before the next job.
                *pending = Some(state);
                return Err(WorkerError::classify(
                    Error::Invariant(
                        "a transfer batch's job was prepared before its last completed".into(),
                    ),
                    Some("transfer batch"),
                ));
            }
        },
    };

    // A batch that failed before it recorded anything has no job to wait
    // for.
    let (job, next) = match next {
        Pending::Discarding(None, failure) if job.is_empty() => (None, Pending::Failed(failure)),
        next => (Some(job), next),
    };
    *pending = Some(next);
    Ok(job)
}

/// Apply the completion of the transfer batch's job on the lane.
pub(crate) fn complete(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    completion: Completion,
) -> std::result::Result<(), WorkerError> {
    let plan = Arc::clone(batch.shared_plan());
    let state = batch.transfer_parts().2.take();
    let completed = completion.check().map_err(|failure| failure.error);
    let mut parts = backend.parts();
    let next = match (state, completed) {
        // A predicate that could not be read fails the batch, which has
        // reserved nothing yet.
        (Some(Pending::Predicates(_)), Err(error)) => Pending::Failed(error.into()),
        (Some(Pending::Predicates(mut opening)), Ok(mut completion)) => {
            match opening.predicates(&mut completion) {
                Ok(()) => Pending::Opened(opening),
                Err(error) => Pending::Failed(error.into()),
            }
        }
        (Some(Pending::Committing(recorded)), completed) => {
            let Some(storage) = parts.storage else {
                return Err(WorkerError::classify(
                    Error::Invariant("a transfer batch lost its rank storage".into()),
                    Some("transfer batch"),
                ));
            };
            let committed = recorded.commit(
                storage,
                parts.requests,
                parts.image.as_mut(),
                &plan,
                completed.map(drop),
            );
            match committed {
                Ok(launched) => {
                    batch.launch(launched);
                    batch.set_stage(Stage::Done);
                    return Ok(());
                }
                Err(failure) => Pending::Failed(failure),
            }
        }
        (Some(Pending::Discarding(recorded, failure)), completed) => {
            let retired = match (recorded, parts.storage) {
                (Some(recorded), Some(storage)) => recorded.discard(storage, parts.image.as_mut()),
                (Some(_), None) => Err(Error::Invariant(
                    "a transfer batch lost its rank storage".into(),
                )),
                (None, _) => Ok(()),
            };
            // What the failed batch recorded could not be ended, or what it
            // reserved could not be retired: that storage stays reserved,
            // and the rank cannot reuse it safely.
            Pending::Failed(match completed.map(drop).and(retired) {
                Ok(()) => failure,
                Err(cleanup) => fatal(cleanup),
            })
        }
        (state, _) => {
            *batch.transfer_parts().2 = state;
            return Err(WorkerError::classify(
                Error::Invariant("a transfer batch has no job on its lane".into()),
                Some("transfer batch"),
            ));
        }
    };
    *batch.transfer_parts().2 = Some(next);
    Ok(())
}

/// Report a failed batch: a failure past its commit point stops the rank,
/// any other completes the batch with its calls failed.
fn fail(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    failure: Failure,
) -> std::result::Result<(), WorkerError> {
    batch.set_stage(Stage::Done);
    if failure.committed {
        // Products, KV directories and pending calls may already be
        // visible; the rank cannot roll them back.
        let mut error = WorkerError::classify(failure.error, Some("transfer batch"));
        error.fatal = true;
        return Err(error);
    }
    super::fail(backend, batch, failure.error, "transfer batch")
}
