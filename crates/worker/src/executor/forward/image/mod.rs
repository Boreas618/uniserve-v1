//! The image runtime: vision and latent encoding, image latent preparation,
//! KV-conditioned denoising and image decoding (BAGEL, SenseNova-U1 and
//! every model with an image encoder).
//!
//! A batch of one image call kind runs in three stages on the executor
//! thread, like a token batch:
//!
//! 1. **Inputs** ([`prepare_inputs`]): inline request images of encoding
//!    batches are decoded and transformed on the runtime's host lane while
//!    other batches execute; the batch launches once they are ready.
//! 2. **Execution and commit** ([`execute`]): the kind's runner computes on
//!    the batch's stream (encoders, the image denoiser, the image decoder);
//!    products, latent trajectory commits and pending calls become visible
//!    together at the commit point. A failure before it retires everything
//!    the batch reserved and fails every call; a failure after it is fatal.
//! 3. **Materialization** ([`poll`]): once the batch's device fences
//!    complete, decoded images are encoded as PNG on the host lane, then
//!    each call's result is decoded and its progress accepted.
//!
//! Context prefills that place vision features and latent images run with
//! the text runtime's prefill (`token`), through the encoder products and
//! the image denoiser this runtime owns.

mod batch;
mod decode;
mod diffusion;
mod encode;
mod host;
mod latents;
pub(crate) mod start;
pub(crate) mod warmup;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use std::time::Instant;

use uniserve_ipc::{
    Batch as BatchPlan, BufferId, CallKind, CallStatus, ErrorCode, ForwardStats, MediaCall,
    RequestKey, Wake,
};

use super::super::backend::{Parts, WorkerBackend, WorkerBatch};
use super::launch::{HostStage, Launched, error_code, failed_output};
use crate::error::WorkerError;
use crate::executor::forward::token::warmup::with_scratch;
use crate::executor::lifecycle::RequestStorage;
use crate::executor::{BatchResult, PendingOutput, Submission};
use crate::ffi::Device;
use crate::ffi::Handle;
use crate::host::{HostLane, HostTask};
use crate::model::image::{ImageModel, ImagePolicy};
use crate::request::diffusion::Trajectories;
use crate::runner::GraphPool;
use crate::runner::decoder::{DecodeImage, DecoderRunner};
use crate::runner::diffusion::image::{DenoiseRow, ImageDiffusion};
use crate::runner::encoder::{EncoderRunner, PreparedImage, Tower};
use crate::storage::KVCache;
use crate::storage::memory::StorageView;
use crate::storage::{LatentStorage, OutputPool, UnitColumn};
use crate::{Error, Outcome, Result};

use batch::{Failure, ImageBatch};
use host::{ImageResult, ImageTask, ImageWork};
use latents::{ImageLatents, Pool};

/// The image runtime of a rank.
pub(crate) struct ImageRuntime {
    facts: ImageModel,
    policy: ImagePolicy,
    /// Encoders by (component, call kind).
    encoders: Vec<EncoderRunner>,
    encoder_routes: HashMap<(String, CallKind), usize>,
    /// The image denoiser of a component serving latent preparation and
    /// denoising.
    diffusion: Option<(String, ImageDiffusion)>,
    /// The image decoder of a component serving image decoding.
    decoder: Option<(String, DecoderRunner)>,
    latents: Option<ImageLatents>,
    trajectories: Trajectories,
    /// The product arena of each device, which bindings name products in.
    arenas: HashMap<Device, StorageView>,
    /// Resets newly assigned KV units on the denoiser's stream.
    units: Option<UnitColumn>,
    /// Graph pools of packed vision encoders, by encoder.
    pools: HashMap<usize, Arc<GraphPool<Handle>>>,
    /// The denoising steps captured at startup, largest first, and their
    /// graph pools: one on the denoiser's device and one on each other
    /// device its modules run on (a generation tower). Empty when flow
    /// graphs are off.
    flow: Vec<start::FlowShape>,
    flow_pools: Vec<Arc<GraphPool<Handle>>>,
    /// Output words of a decoding batch's pixels at most.
    max_pixel_words: usize,
    outputs: OutputPool,
    host: HostLane<ImageTask>,
}

impl ImageRuntime {
    pub(crate) fn facts(&self) -> &ImageModel {
        &self.facts
    }

    pub(crate) fn policy(&self) -> &ImagePolicy {
        &self.policy
    }

    pub(crate) fn diffusion(&self) -> Option<&ImageDiffusion> {
        self.diffusion.as_ref().map(|(_, runner)| runner)
    }

    pub(crate) fn diffusion_mut(&mut self) -> Option<&mut ImageDiffusion> {
        self.diffusion.as_mut().map(|(_, runner)| runner)
    }

    fn latents(&self) -> Result<&ImageLatents> {
        self.latents
            .as_ref()
            .ok_or_else(|| Error::Unsupported("image trajectories require a latent pool".into()))
    }

    fn latents_mut(&mut self) -> Result<&mut ImageLatents> {
        self.latents
            .as_mut()
            .ok_or_else(|| Error::Unsupported("image trajectories require a latent pool".into()))
    }

    /// The latent pool and its storage, for trajectory transfers.
    pub(crate) fn latent_pool(&mut self) -> Result<(&mut Pool, &LatentStorage)> {
        let latents = self.latents_mut()?;
        Ok((&mut latents.pool, &latents.storage))
    }

    /// Order work on `stream` that reads the latent pool's pages after the
    /// pool's previous user; [`Self::leave_latents`] ends it.
    pub(crate) fn enter_latents(&self, stream: Option<&uniserve_cuda::Stream>) -> Result<()> {
        self.latents()?.enter(stream)
    }

    /// End work on `stream` that used the latent pool's pages.
    pub(crate) fn leave_latents(&mut self, stream: Option<&uniserve_cuda::Stream>) -> Result<()> {
        self.latents_mut()?.leave(stream)
    }

    /// The encoder serving `kind` calls of `component`.
    fn encoder_route(&self, component: &str, kind: CallKind) -> Result<usize> {
        self.encoder_routes
            .get(&(component.to_owned(), kind))
            .copied()
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "component {component:?} has no {} encoder",
                    kind.as_str()
                ))
            })
    }

    /// Warm every unpacked image encoder and the image decoder on one black
    /// image of twice the denoiser's patch side (64 pixels without a
    /// denoiser), in binding order, as serving prepares its images.
    pub(crate) fn warm(&mut self) -> Result<()> {
        let side = self
            .facts
            .denoiser
            .map_or(64, |layout| 2 * layout.downsample);
        for runner in &self.encoders {
            runner.warm(side)?;
            if let Some(stream) = runner.stream() {
                stream.synchronize()?;
            }
        }
        let (Some((_, decoder)), Some(latents), Some(layout)) =
            (&self.decoder, &self.latents, self.facts.denoiser)
        else {
            return Ok(());
        };
        let units = layout.latent_rows(side, side)?;
        let workspace = latents.workspace()?;
        workspace
            .narrow(0, 0, i64::from(units))?
            .zero(decoder.stream())?;
        decoder.decode(
            &[DecodeImage {
                sample: 0,
                units,
                height: side,
                width: side,
                product: None,
            }],
            |_, _| Ok(()),
        )?;
        if let Some(stream) = decoder.stream() {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Capture every packed vision encoder's graphs into its pool, then
    /// the image denoiser's configured steps over scratch KV of `kv`.
    pub(crate) fn capture(&mut self, kv: Option<&mut KVCache>) -> Result<()> {
        for (index, runner) in self.encoders.iter_mut().enumerate() {
            if let Some(pool) = self.pools.get(&index) {
                runner.capture(Some(pool), &mut || Ok(()))?;
            }
        }
        let Some(kv) = kv.filter(|_| !self.flow_pools.is_empty()) else {
            return Ok(());
        };
        let pools = self.flow_pools.clone();
        let shapes = self.flow.clone();
        for shape in &shapes {
            self.capture_flow(shape, &pools, kv)?;
        }
        if let Some((_, diffusion)) = &self.diffusion {
            if let Some(stream) = diffusion.stream() {
                stream.synchronize()?;
            }
            tracing::info!(
                target: "uniserve_worker.warmup",
                "captured {} image denoising graphs",
                diffusion.captured()
            );
        }
        Ok(())
    }

    /// Capture one denoising step: `shape.images` images placed in the
    /// step workspace as serving places them (consecutive whole pages from
    /// its start), each evaluating `shape.branches` guidance branches over
    /// one scratch slot's prefix.
    fn capture_flow(
        &mut self,
        shape: &start::FlowShape,
        pools: &[Arc<GraphPool<Handle>>],
        kv: &mut KVCache,
    ) -> Result<()> {
        let latents = self.latents()?;
        let page_units = latents.pool.page_units() as u32;
        let workspace = latents.workspace()?.clone();
        let Some((_, diffusion)) = &mut self.diffusion else {
            return Ok(());
        };
        let units = diffusion.layout().latent_rows(shape.height, shape.width)?;
        let padded = units.div_ceil(page_units) * page_units;
        let stream = diffusion.stream().cloned();
        // Captured samples start from zeros, as fresh scratch pages do.
        workspace
            .narrow(0, 0, i64::from(padded) * shape.images as i64)?
            .zero(stream.as_ref())?;
        let prefix = start::CAPTURE_PREFIX;
        let mut rows = Vec::with_capacity(shape.images * shape.branches);
        for image in 0..shape.images {
            for _ in 0..shape.branches {
                rows.push(DenoiseRow {
                    sample: image as u32 * padded,
                    units,
                    height: shape.height,
                    width: shape.width,
                    slot: image as u32 + 1,
                    prefix: prefix as u32,
                    temporal: prefix as u64,
                    timestep: 0.5,
                });
            }
        }
        let lengths = vec![prefix; shape.images];
        with_scratch(kv, &lengths, stream.as_ref(), |kv| {
            diffusion.capture(&rows, kv.tables(), pools, &mut || Ok(()))
        })
    }

    /// The product arena of `device`.
    pub(crate) fn arena(&self, device: Device) -> Result<&StorageView> {
        self.arenas
            .get(&device)
            .ok_or_else(|| Error::Invariant("no product arena is bound on this device".into()))
    }

    /// Seal every runner's graphs.
    pub(crate) fn seal(&mut self) {
        for runner in &mut self.encoders {
            runner.seal();
        }
        if let Some((_, runner)) = &mut self.diffusion {
            runner.seal();
        }
    }

    pub(crate) fn set_wake(&self, wake: Option<Wake>) {
        self.host.set_wake(wake);
    }

    /// Release graphs, host work and output storage after the lanes
    /// drained.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        for runner in &mut self.encoders {
            result = result.and(runner.close(aborted));
        }
        if let Some((_, runner)) = &mut self.diffusion {
            result = result.and(runner.close(aborted));
        }
        if aborted {
            self.host.abort();
        } else if let Some(error) = self.host.close().into_iter().next() {
            result = result.and(Err(error));
        }
        self.outputs.begin_close();
        if !aborted {
            self.outputs.finish_close();
        }
        result
    }
}

impl RequestStorage for ImageRuntime {
    fn admit_slots(&mut self, slots: &[usize]) -> Result<()> {
        // A reused slot never continues an earlier request's trajectory.
        self.trajectories.release(slots);
        Ok(())
    }

    fn release_buffers(&mut self, buffers: &HashSet<BufferId>) -> Result<()> {
        if let Some(latents) = &self.latents {
            latents
                .pool
                .release_exports(&buffers.iter().copied().collect::<Vec<_>>());
        }
        Ok(())
    }

    fn release_requests(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<()> {
        let mut result = Ok(());
        if let Some(latents) = &self.latents {
            // Finished requests' published versions retire once their
            // readers did, except retained ones.
            let exports: Vec<BufferId> = latents
                .pool
                .exports()
                .map(|source| source.buffer)
                .filter(|buffer| requests.contains(&buffer.owner) && !retained.contains(buffer))
                .collect();
            latents.pool.release_exports(&exports);
            for transfer in latents.pool.cancel_imports(requests) {
                result = result.and(transfer.read().cancel());
            }
        }
        result
    }

    fn retirement_ready(
        &mut self,
        _buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        Ok(self
            .latents
            .as_ref()
            .is_none_or(|latents| latents.pool.retirement_ready(requests)))
    }

    fn retire_slots(&mut self, slots: &[usize]) -> Result<()> {
        self.trajectories.release(slots);
        if let Some(latents) = &mut self.latents {
            let slots: Vec<i64> = slots.iter().map(|&slot| slot as i64).collect();
            for transfer in latents.pool.release_slots(&slots)? {
                transfer.read().cancel()?;
            }
        }
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        self.outputs.reap();
        if let Some(latents) = &mut self.latents {
            latents.pool.reap();
        }
        Ok(())
    }
}

/// The host-prepared inline images of an encoding batch, by call index.
pub(crate) struct Inputs {
    tasks: Vec<(usize, Arc<HostTask<ImageTask>>)>,
}

impl Inputs {
    fn ready(&self) -> bool {
        self.tasks.iter().all(|(_, task)| task.completion.done())
    }

    /// The prepared images, by call index; a failed preparation fails the
    /// batch.
    fn take(self) -> Result<HashMap<usize, PreparedImage>> {
        let mut images = HashMap::new();
        for (index, task) in self.tasks {
            match task.completion.outcome() {
                Some(Outcome::Success(result)) => match &*result {
                    ImageResult::Prepared(image) => {
                        images.insert(index, image.clone());
                    }
                    ImageResult::Encoded(_) => {
                        return Err(Error::Invariant(
                            "an image preparation produced encoded bytes".into(),
                        ));
                    }
                },
                Some(Outcome::Failed(error)) => {
                    return Err(Error::Invalid(format!(
                        "inline image preparation failed: {error}"
                    )));
                }
                Some(Outcome::Cancelled) | None => {
                    return Err(Error::State("inline image preparation did not complete"));
                }
            }
        }
        Ok(images)
    }

    fn cancel(&self) {
        for (_, task) in &self.tasks {
            let _ = task.cancel(false);
        }
    }
}

/// Start the host preparation of a batch's inline images, and report
/// whether it completed. Notifies `submission` when it does.
pub(crate) fn prepare_inputs(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    submission: &Arc<Submission>,
) -> std::result::Result<bool, WorkerError> {
    if let Some(inputs) = batch.image_inputs() {
        return Ok(inputs.ready());
    }
    let plan = Arc::clone(batch.shared_plan());
    let Some(first) = plan.calls.first() else {
        return Ok(true);
    };
    let tower = match first.code {
        CallKind::Media(MediaCall::VisionEncoding) => Tower::Vit,
        CallKind::Media(MediaCall::LatentEncoding) => Tower::Vae,
        _ => return Ok(true),
    };
    let parts = backend.parts();
    let Some(runtime) = parts.image else {
        return Ok(true);
    };
    let mut tasks = Vec::new();
    let reserved = (|| -> Result<()> {
        for (index, call) in plan.calls.iter().enumerate() {
            let Some(payload) = &call.input_image else {
                continue;
            };
            let request = parts.requests.get(call.request_key.request_id.0)?;
            let input_images = request.admission().input_images;
            let route = runtime.encoder_route(&call.component, call.code)?;
            let pin = runtime.encoders[route].device().is_cuda();
            let task = runtime.host.reserve()?;
            tasks.push((index, Arc::clone(&task)));
            task.configure(ImageTask::new(ImageWork::Prepare {
                function: runtime.policy.prepare_input().clone(),
                loaded: runtime.policy.loaded().clone(),
                tower,
                payload: Arc::clone(payload),
                input_images,
                pin,
            }))?;
            let weak: Weak<Submission> = Arc::downgrade(submission);
            let notify: Box<dyn FnOnce() + Send> = Box::new(move || {
                if let Some(submission) = weak.upgrade() {
                    submission.notify_ready();
                }
            });
            if let Some(notify) = task.completion.subscribe(notify) {
                notify();
            }
            task.submit()?;
        }
        Ok(())
    })();
    let inputs = Inputs { tasks };
    if let Err(error) = reserved {
        inputs.cancel();
        return Err(WorkerError::classify(error, Some("image preparation")));
    }
    let ready = inputs.ready();
    batch.set_image_inputs(inputs);
    Ok(ready)
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
        Ok(None) => return Err(super::unsupported(kind, "image")),
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
/// image runtime.
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
    let (Some(runtime), Some(storage)) = (image, storage) else {
        return Ok(None);
    };
    let (device, stream, words) = match kind {
        CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding) => {
            encode::placement(runtime, plan)?
        }
        CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising) => {
            diffusion::placement(runtime)?
        }
        CallKind::Media(MediaCall::ImageDecoding) => decode::placement(runtime, plan)?,
        _ => {
            return Err(Error::Unsupported(format!(
                "{} is served by another runtime",
                kind.as_str()
            ))
            .into());
        }
    };
    let mut batch = ImageBatch::open(
        runtime, storage, requests, info, rank, plan, device, stream, words, started,
    )?;
    let result = (|| -> std::result::Result<ForwardStats, Failure> {
        batch.bind_flags()?;
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
        batch.commit()?;
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
            if let Err(cleanup) = batch.discard() {
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
    encodings: Vec<batch::Encoding>,
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
