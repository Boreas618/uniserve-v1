//! An image batch: the parts of the backend it executes with
//! ([`ImageParts`]), its reservation, commit and retirement
//! ([`ImageBatch`]), and the dispatch of its calls to the encoder, denoiser
//! and decoder forwards ([`execute`]).
//!
//! [`ImageBatch::open`] binds every call's pending output, resolves
//! host-gated predicates (a false one skips its call, which only publishes
//! false flags) and leases the batch's output storage on its stream.
//! [`ImageBatch::bind_flags`] binds the completion flags the calls publish.
//! After a kind's computation, [`ImageBatch::commit`] validates every
//! owner's update (products, latent trajectories, pending calls) and makes
//! them visible together; a failure before that point retires everything
//! the batch reserved ([`ImageBatch::discard`]), a failure after it is
//! fatal.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, CallKind, CallStatus, DType, ErrorCode,
    ForwardStats, MediaCall, RequestKey, TensorRef,
};

use crate::error::WorkerError;
use crate::executor::backend::{Parts, WorkerBackend, WorkerBatch};
use crate::executor::forward::launch::{
    HostStage, Launched, error_code, failed_output, read_flag, reports_output, seal as seal_lease,
};
use crate::executor::forward::products::Publications;
use crate::executor::forward::transfer::latents::{self, PendingExport, Pool};
use crate::executor::forward::unsupported;
use crate::executor::{BatchResult, PendingOutput};
use crate::ffi::Device;
use crate::host::image::ImageResult;
use crate::host::image::ImageTask;
use crate::host::{HostLane, HostTask};
use crate::lane::Job;
use crate::request::RequestPool;
use crate::request::diffusion::Trajectories;
use crate::runner::decoder::DecoderRunner;
use crate::runner::decoder::runners::DecoderRunners;
use crate::runner::denoiser::image::ImageDiffusion;
use crate::runner::denoiser::image_model::{ImageModel, ImagePolicy};
use crate::runner::denoiser::runners::DenoiserRunners;
use crate::runner::encoder::inputs::Inputs;
use crate::runner::encoder::runners::EncoderRunners;
use crate::storage::{
    ImageLatents, LatentStorage, LatentUpdate, OutputLease, OutputPlacement, OutputPool,
    ProductBuffer, ProductRead, Scalar,
};
use crate::stream::Issue;
use crate::transfer::Retirement;
use crate::worker::RankStorage;
use crate::{Error, Outcome, Result};

/// One call's state through its batch.
pub(crate) struct CallState {
    pub(crate) output: PendingOutput,
    pub(crate) slot: usize,
    /// False when the call's host predicate skipped it.
    pub(crate) active: bool,
    pub(crate) completion_write: Option<ProductBuffer>,
}

/// A decoded image's encoding, run once its pixels reached the host.
pub(crate) struct Encoding {
    /// The call's index in the batch.
    pub(crate) call: usize,
    pub(crate) task: Arc<HostTask<ImageTask>>,
}

/// A failure, and whether it happened past the commit point.
pub(crate) struct Failure {
    pub(crate) error: Error,
    pub(crate) committed: bool,
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Self {
            error,
            committed: false,
        }
    }
}

pub(crate) fn fatal(error: Error) -> Failure {
    Failure {
        error,
        committed: true,
    }
}

pub(crate) use super::super::launch::lock;

/// A batch's persistent buffer allocations by buffer.
pub(crate) fn allocations(plan: &BatchPlan) -> HashMap<BufferId, BufferAllocation> {
    plan.buffer_allocations
        .iter()
        .map(|allocation| (allocation.buffer, *allocation))
        .collect()
}

/// What an image batch's issued commit leaves to apply once its work was
/// performed: the trajectory commits and the calls to record as pending.
pub(crate) struct Commit {
    updates: Vec<LatentUpdate>,
    pending: Vec<(RequestKey, uniserve_core::CallId, bool)>,
}

/// An image batch's borrowed owners and the resources it reserved.
pub(crate) struct ImageBatch<'a> {
    pub(crate) parts: ImageParts<'a>,
    pub(crate) storage: &'a mut RankStorage,
    pub(crate) requests: &'a mut RequestPool,
    pub(crate) plan: &'a BatchPlan,
    /// The device and stream the batch's computation runs on.
    pub(crate) device: Option<i32>,
    pub(crate) stream: Option<Stream>,
    /// What the batch's own steps issue on its stream: the start of its
    /// lease, its flags, its commit and its retirement after a failure.
    pub(crate) job: Job<'static>,
    /// This rank reports the component's host results.
    pub(crate) reports: bool,
    pub(crate) calls: Vec<CallState>,
    pub(crate) buffer: OutputLease,
    pub(crate) writes: Vec<ProductBuffer>,
    pub(crate) reads: Vec<Arc<ProductRead>>,
    /// Trajectory commits and releases by call index, applied at the
    /// commit point.
    pub(crate) updates: Vec<(usize, LatentUpdate)>,
    /// Image encodings to run once the batch's pixels reached the host.
    pub(crate) encodings: Vec<Encoding>,
    /// Product exports (encoder features, trajectory versions),
    /// registered once the batch commits.
    pub(crate) publications: Publications,
    /// Trajectory versions exported from the latent pool, released from it
    /// if the batch fails before committing.
    pub(crate) latent_exports: Vec<BufferId>,
    /// The version exports the batch's forwards issued, by call index,
    /// located when the commit is applied.
    pub(crate) exports: Vec<(usize, PendingExport)>,
    /// Signals the physical end of the batch's KV accesses.
    pub(crate) retirement: Arc<Retirement>,
    /// The batch uses the latent pool on its stream (see
    /// `ImageLatents::enter`).
    pub(crate) latents_entered: bool,
    pub(crate) started: Instant,
}

impl<'a> ImageBatch<'a> {
    /// Bind the batch's outputs, skip the `skipped` calls (those whose host
    /// predicate is false) and lease its output storage on `stream`.
    /// `words` are the payload words the batch reads back beside one
    /// completion word per call.
    #[expect(clippy::too_many_arguments, reason = "the batch's borrowed owners")]
    pub(crate) fn open(
        parts: ImageParts<'a>,
        storage: &'a mut RankStorage,
        requests: &'a mut RequestPool,
        info: &uniserve_ipc::WorkerInfo,
        rank: usize,
        plan: &'a BatchPlan,
        device: Option<i32>,
        stream: Option<Stream>,
        words: usize,
        started: Instant,
        skipped: &HashSet<usize>,
    ) -> Result<Self> {
        let first = plan
            .calls
            .first()
            .ok_or_else(|| Error::Invariant("an image batch has calls".into()))?;
        if plan
            .calls
            .iter()
            .any(|call| call.component != first.component)
        {
            return Err(Error::Invalid("an image batch serves one component".into()));
        }
        let reports = reports_output(info, rank, &first.component)?;
        let rows = plan
            .calls
            .iter()
            .map(|call| {
                let request = requests.get(call.request_key.request_id.0)?;
                Ok((call.request_key, call.call_id, request.slot()))
            })
            .collect::<Result<Vec<_>>>()?;
        let bound = requests.bind_calls(&rows)?;
        let mut calls = Vec::with_capacity(plan.calls.len());
        for (index, (call, request)) in plan.calls.iter().zip(&bound).enumerate() {
            calls.push(CallState {
                output: PendingOutput::new(call, request.progress()?, index)?,
                slot: request.slot(),
                active: true,
                completion_write: None,
            });
        }
        for &index in skipped {
            let state = calls
                .get_mut(index)
                .ok_or_else(|| Error::Invariant("a skipped call is outside its batch".into()))?;
            state.active = false;
            state.output.output.status = CallStatus::Predicated;
        }
        let devices: Vec<i32> = device.into_iter().collect();
        let buffer = parts.outputs.lease(
            plan.calls.len(),
            (plan.calls.len() + words).max(1),
            &devices,
            false,
        )?;
        let job = Job::new();
        if let Some(stream) = &stream {
            lock(&buffer).begin_device(device, job.on(Some(stream)), &storage.events)?;
        }
        Ok(Self {
            parts,
            storage,
            requests,
            plan,
            device,
            stream,
            job,
            reports,
            calls,
            buffer,
            writes: Vec::new(),
            reads: Vec::new(),
            updates: Vec::new(),
            encodings: Vec::new(),
            publications: Publications::default(),
            latent_exports: Vec::new(),
            exports: Vec::new(),
            retirement: Arc::new(Retirement::default()),
            latents_entered: false,
            started,
        })
    }

    /// The batch's product device.
    pub(crate) fn target(&self) -> Device {
        self.device.map_or(Device::Cpu, Device::Cuda)
    }

    /// Indices of the calls that execute.
    pub(crate) fn active(&self) -> Vec<usize> {
        (0..self.calls.len())
            .filter(|&index| self.calls[index].active)
            .collect()
    }

    /// Bind every call's completion flag; skipped calls publish false now.
    pub(crate) fn bind_flags(&mut self) -> Result<()> {
        let target = self.target();
        let slots: HashMap<RequestKey, usize> = self
            .plan
            .calls
            .iter()
            .zip(&self.calls)
            .map(|(call, state)| (call.request_key, state.slot))
            .collect();
        let flags: Vec<(TensorRef, Device)> = self
            .plan
            .calls
            .iter()
            .filter_map(|call| call.completion_output.clone())
            .map(|flag| (flag, target))
            .collect();
        if flags.is_empty() {
            return Ok(());
        }
        let bound = self.storage.tensors.bind_outputs(
            &flags,
            OutputPlacement {
                request_slots: Some(&slots),
                ..OutputPlacement::default()
            },
        )?;
        for write in bound {
            self.writes.push(write.clone());
            let index = self
                .plan
                .calls
                .iter()
                .position(|call| {
                    call.completion_output
                        .as_ref()
                        .is_some_and(|flag| flag.buffer_id() == write.id())
                })
                .ok_or_else(|| Error::Invariant("a completion flag names no call".into()))?;
            let state = &mut self.calls[index];
            if !state.active {
                self.storage.tensors.write_scalar(
                    &write,
                    Scalar::Int(0),
                    self.job.on(self.stream.as_ref()),
                )?;
            }
            state.completion_write = Some(write);
        }
        Ok(())
    }

    /// Perform, on this thread, what the batch recorded so far.
    ///
    /// FIXME: the forwards of an image batch still issue on its stream at
    /// once, so what the batch records around them is performed here, in
    /// their order, instead of being a job of the batch's lane.
    pub(crate) fn perform(&mut self) -> Result<()> {
        std::mem::take(&mut self.job)
            .perform()
            .check()
            .map(drop)
            .map_err(|failure| failure.error)
    }

    /// Validate every owner's update and record the device half of the
    /// commit: the end of the read leases, the completion flags, the end of
    /// the latent pool's use and the seal of the output. The returned
    /// commit is applied once that work was performed
    /// ([`Self::apply_commit`]).
    pub(crate) fn issue_commit(&mut self) -> std::result::Result<Commit, Failure> {
        let stream = self.job.on(self.stream.as_ref());
        let reads = std::mem::take(&mut self.reads);
        self.storage.tensors.complete_reads(&reads, stream, &[])?;

        // Completion flags of executed calls become true.
        for state in self.calls.iter().filter(|state| state.active) {
            if let Some(write) = &state.completion_write {
                self.storage
                    .tensors
                    .write_scalar(write, Scalar::Int(1), stream)?;
            }
        }
        {
            let output = lock(&self.buffer);
            for (state, call) in self.calls.iter().zip(&self.plan.calls) {
                state.output.validate_output(&output, call)?;
            }
        }
        self.storage.tensors.validate_writes(&self.writes)?;
        let updates: Vec<LatentUpdate> = self
            .updates
            .iter()
            .map(|(_, update)| update.clone())
            .collect();
        if !updates.is_empty() {
            self.parts.latents()?.pool.validate_updates(&updates)?;
        }
        self.leave_latents()?;
        self.seal()?;
        for state in &mut self.calls {
            state.output.reports_output = self.reports;
        }
        for (index, update) in &self.updates {
            self.calls[*index].output.apply_latent_update(update);
        }
        let pending: Vec<_> = self
            .plan
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id, call.advances_state()))
            .collect();
        self.requests.validate_pending(&pending)?;
        Ok(Commit { updates, pending })
    }

    /// Apply an issued commit once its work was performed: publish the
    /// exports the forwards issued, then make the writes visible, apply the
    /// trajectory commits, register the exports and record the calls as
    /// pending. A failure after the exports' check is fatal.
    pub(crate) fn apply_commit(&mut self, commit: Commit) -> std::result::Result<(), Failure> {
        let Commit { updates, pending } = commit;
        for (index, export) in std::mem::take(&mut self.exports) {
            let pool = &self.parts.latents()?.pool;
            latents::locate_export(
                pool,
                export,
                &mut self.publications,
                &mut self.calls[index].output,
            )?;
        }
        self.publications.validate(&self.storage.tensors)?;

        // Every owner accepted its update. Any failure from here is fatal.
        let writes = std::mem::take(&mut self.writes);
        self.storage.tensors.commit_writes(&writes).map_err(fatal)?;
        if !updates.is_empty() {
            let cancelled = self
                .parts
                .latents_mut()
                .map_err(fatal)?
                .pool
                .apply_updates(&updates);
            for transfer in cancelled {
                // Released trajectories' pending imports end; their pages
                // retire with the transfer.
                transfer.read().cancel().map_err(fatal)?;
            }
        }
        self.publications
            .register(&self.storage.tensors)
            .map_err(fatal)?;
        self.requests.add_pending(&pending).map_err(fatal)
    }

    /// Seal the lease on the batch's stream and retire its KV accesses and
    /// output storage once the fences complete; wake the service then.
    pub(crate) fn seal(&self) -> Result<()> {
        seal_lease(
            &self.storage.events,
            &self.buffer,
            self.job.on(self.stream.as_ref()),
            &self.retirement,
        )
    }

    /// End the batch's use of the latent pool on its stream.
    fn leave_latents(&mut self) -> Result<()> {
        if std::mem::take(&mut self.latents_entered) {
            self.parts
                .latents_mut()?
                .leave(self.job.on(self.stream.as_ref()))?;
        }
        Ok(())
    }

    /// Retire every reservation after a failure before the commit point,
    /// recording the fences that end them; storage returns once those
    /// fences were performed and complete. What the batch recorded and did
    /// not perform is reduced to its event operations: the failed forward's
    /// remaining work addresses storage this releases.
    pub(crate) fn discard(&mut self) -> Result<()> {
        self.job = std::mem::take(&mut self.job).orders_only();
        let left = self.leave_latents();
        self.exports
            .drain(..)
            .for_each(|(_, export)| export.revoke());
        let revoked = self.publications.revoke();
        let exported = std::mem::take(&mut self.latent_exports);
        if !exported.is_empty()
            && let Ok(latents) = self.parts.latents()
        {
            // The versions retire once their transports drop them.
            latents.pool.release_exports(&exported);
        }
        let stream = self.job.on(self.stream.as_ref());
        let reads = std::mem::take(&mut self.reads);
        let completed = self.storage.tensors.complete_reads(&reads, stream, &[]);
        let writes = std::mem::take(&mut self.writes);
        let abandoned = self.storage.tensors.abandon_writes(&writes);
        for encoding in self.encodings.drain(..) {
            // Unconfigured encodings return their lane capacity.
            let _ = encoding.task.cancel(true);
        }
        let sealed = if lock(&self.buffer).sealed() {
            Ok(())
        } else {
            self.seal()
        };
        lock(&self.buffer).abandon();
        left.and(revoked).and(completed).and(abandoned).and(sealed)
    }
}

/// The parts of the backend an image batch executes with: the image
/// model's facts and host policy, the rank's runners, and the storage its
/// trajectories and products live in.
pub(crate) struct ImageParts<'a> {
    pub(crate) model: &'a ImageModel,
    pub(crate) policy: &'a ImagePolicy,
    pub(crate) encoders: &'a mut EncoderRunners,
    pub(crate) denoisers: &'a mut DenoiserRunners,
    pub(crate) decoders: &'a mut DecoderRunners,
    pub(crate) latents: Option<&'a mut ImageLatents>,
    pub(crate) trajectories: &'a mut Trajectories,
    /// The output leases of image batches.
    pub(crate) outputs: &'a mut OutputPool,
    pub(crate) host: &'a HostLane<ImageTask>,
}

impl ImageParts<'_> {
    pub(crate) fn facts(&self) -> &ImageModel {
        self.model
    }

    pub(crate) fn policy(&self) -> &ImagePolicy {
        self.policy
    }

    pub(crate) fn diffusion(&self) -> Option<&ImageDiffusion> {
        self.denoisers.image.as_ref().map(|(_, runner)| runner)
    }

    pub(crate) fn diffusion_mut(&mut self) -> Option<&mut ImageDiffusion> {
        self.denoisers.image.as_mut().map(|(_, runner)| runner)
    }

    pub(crate) fn latents(&self) -> Result<&ImageLatents> {
        self.latents
            .as_deref()
            .ok_or_else(|| Error::Unsupported("image trajectories require a latent pool".into()))
    }

    fn latents_mut(&mut self) -> Result<&mut ImageLatents> {
        self.latents
            .as_deref_mut()
            .ok_or_else(|| Error::Unsupported("image trajectories require a latent pool".into()))
    }

    /// The latent pool and its storage, for trajectory transfers.
    pub(crate) fn latent_pool(&mut self) -> Result<(&mut Pool, &LatentStorage)> {
        let latents = self.latents_mut()?;
        Ok((&mut latents.pool, &latents.storage))
    }

    /// Order work on `stream` that reads the latent pool's pages after the
    /// pool's previous user; [`Self::leave_latents`] ends it.
    pub(crate) fn enter_latents(&self, stream: impl Issue) -> Result<()> {
        self.latents()?.enter(stream)
    }

    /// End work on `stream` that used the latent pool's pages.
    pub(crate) fn leave_latents(&mut self, stream: impl Issue) -> Result<()> {
        self.latents_mut()?.leave(stream)
    }

    /// The image decoder.
    pub(crate) fn decoder(&self) -> Option<&DecoderRunner> {
        self.decoders.image.as_ref().map(|(_, runner)| runner)
    }
}

/// Execute a batch of `kind` calls.
pub(crate) fn execute(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<(), WorkerError> {
    let plan = Arc::clone(batch.shared_plan());
    let predecessors = batch.predecessors().to_vec();
    let inputs = batch.take_image_inputs();
    let failure = match run(backend.parts(), &plan, kind, inputs) {
        Ok(Some(launched)) => {
            batch.launch(launched);
            return Ok(());
        }
        Ok(None) => return Err(unsupported(kind, "image")),
        Err(failure) => failure,
    };
    let mut error = WorkerError::classify(failure.error, Some("image batch"));
    if failure.committed {
        error.fatal = true;
    }
    if error.fatal {
        return Err(error);
    }
    tracing::warn!(
        target: "uniserve_worker.execution",
        "image batch {} failed: {error}{}",
        plan.batch_id,
        error
            .traceback()
            .map(|trace| format!("\n{trace}"))
            .unwrap_or_default()
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

/// Reserve, compute and commit one batch. `None` on a rank without an
/// image model.
fn run(
    parts: Parts<'_>,
    plan: &BatchPlan,
    kind: CallKind,
    inputs: Option<Inputs>,
) -> std::result::Result<Option<Launched>, Failure> {
    let started = Instant::now();
    let Parts {
        info,
        rank,
        requests,
        storage,
        text,
        image,
        experts,
        ..
    } = parts;
    let (Some(image), Some(storage)) = (image, storage) else {
        return Ok(None);
    };
    let (device, stream, words) = match kind {
        CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding) => {
            crate::runner::encoder::image_placement(&image, plan)?
        }
        CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising) => {
            crate::runner::denoiser::image::image_placement(&image)?
        }
        CallKind::Media(MediaCall::ImageDecoding) => {
            crate::runner::decoder::image_placement(&image, plan)?
        }
        _ => {
            return Err(Error::Unsupported(format!(
                "{} is served by another runtime",
                kind.as_str()
            ))
            .into());
        }
    };
    let mut skipped = HashSet::new();
    for (index, call) in plan.calls.iter().enumerate() {
        if let Some(predicate) = &call.predicate
            && predicate.dtype == DType::U8
            && !read_flag(&storage.tensors, call, predicate, device, stream.as_ref())?
        {
            skipped.insert(index);
        }
    }
    let mut batch = ImageBatch::open(
        image, storage, requests, info, rank, plan, device, stream, words, started, &skipped,
    )?;
    let result = (|| -> std::result::Result<ForwardStats, Failure> {
        batch.bind_flags()?;
        batch.perform()?;
        let stats = match kind {
            CallKind::Media(call @ (MediaCall::VisionEncoding | MediaCall::LatentEncoding)) => {
                let images = inputs.map(Inputs::take).transpose()?.unwrap_or_default();
                batch.encode(call, images)?
            }
            CallKind::Media(MediaCall::LatentPreparation) => batch.prepare_latents()?,
            CallKind::Media(MediaCall::Denoising) => batch.denoise(text, experts)?,
            CallKind::Media(MediaCall::ImageDecoding) => batch.decode()?,
            _ => ForwardStats::default(),
        };
        let commit = batch.issue_commit()?;
        batch.perform()?;
        batch.apply_commit(commit)?;
        Ok(stats)
    })();
    match result {
        Ok(mut stats) => {
            let elapsed = batch.started.elapsed().as_micros() as u64;
            stats.component_us.insert("image_execute".into(), elapsed);
            let encodings = std::mem::take(&mut batch.encodings);
            Ok(Some(Launched {
                outputs: batch.calls.into_iter().map(|state| state.output).collect(),
                host: (!encodings.is_empty()).then(|| {
                    Box::new(Encodings {
                        encodings,
                        submitted: false,
                    }) as Box<dyn HostStage>
                }),
                buffer: batch.buffer,
                started,
                stats,
                timer: "image_materialize",
            }))
        }
        Err(failure) if failure.committed => Err(failure),
        Err(failure) => {
            if let Err(cleanup) = batch.discard().and(batch.perform()) {
                return Err(Failure {
                    error: cleanup,
                    committed: true,
                });
            }
            Err(failure)
        }
    }
}

/// The PNG encodings of a decoding batch's images, run once their pixels
/// reached the host.
struct Encodings {
    encodings: Vec<Encoding>,
    submitted: bool,
}

impl HostStage for Encodings {
    fn poll(&mut self) -> Result<bool> {
        if !self.submitted {
            // The output fences completed: the pixels are on the host.
            for encoding in &self.encodings {
                encoding.task.submit()?;
            }
            self.submitted = true;
        }
        Ok(self
            .encodings
            .iter()
            .all(|encoding| encoding.task.completion.done()))
    }

    fn finish(&mut self, outputs: &mut [PendingOutput]) -> Result<()> {
        for encoding in std::mem::take(&mut self.encodings) {
            let output = outputs
                .get_mut(encoding.call)
                .ok_or_else(|| Error::Invariant("an image encoding names no call".into()))?;
            match encoding.task.completion.outcome() {
                Some(Outcome::Success(result)) => {
                    if let ImageResult::Encoded(bytes) = &*result
                        && output.reports_output
                    {
                        let source =
                            uniserve_core::MediaSource::publish(bytes).map_err(|error| {
                                Error::Invalid(format!(
                                    "publishing an encoded image failed: {error}"
                                ))
                            })?;
                        output.set_media(source)?;
                    }
                }
                Some(Outcome::Failed(error)) => {
                    tracing::error!(
                        target: "uniserve_worker.execution",
                        "image encoding failed: {error}"
                    );
                    output.output.status = CallStatus::Error;
                    output.output.error_code = Some(ErrorCode::InvalidCall);
                }
                Some(Outcome::Cancelled) | None => {
                    output.output.status = CallStatus::Error;
                    output.output.error_code = Some(ErrorCode::Internal);
                }
            }
        }
        Ok(())
    }

    fn cancel(&mut self) {
        for encoding in &self.encodings {
            let _ = encoding.task.cancel(false);
        }
    }
}
