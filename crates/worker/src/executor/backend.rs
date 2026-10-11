//! The worker's execution backend: request state, lifecycle commands,
//! batch validation, numerical dispatch and result delivery.
//!
//! [`WorkerBackend`] is the single [`Backend`] and [`ServiceBackend`] of a
//! rank. It owns the request pool, the request storage owners (through
//! [`RequestStorage`]) and the placed components with their bindings, and
//! dispatches each batch's calls to the capability runtime that serves them
//! ([`super::forward`]). Lifecycle commands (Start, Finish, Free), batch
//! admission, dependencies, retirement and result delivery are handled
//! here for every batch, with or without calls.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use uniserve_core::CallId;
use uniserve_ipc::{
    Batch as BatchPlan, BatchCommand, BatchOutput, CallKind, ErrorCallIdentity, ForwardMode,
    RequestKind, WorkerInfo, WorkerResponseError,
};

use super::forward::{self, launch::Launched};
use super::lifecycle::{RequestStorage, Retirement};
use super::{Backend, BatchResult, Submission};
use crate::error::WorkerError;
use crate::ffi::Function;
use crate::lane::JobQueues;
use crate::model::registry::Components;
use crate::profiling::kernels::KernelRecords;
use crate::profiling::profiler::Profiler;
use crate::request::RequestPool;
use crate::request::video::VideoRequests;
use crate::runner::decoder::runners::DecoderRunners;
use crate::runner::denoiser::runners::DenoiserRunners;
use crate::runner::encoder::runners::EncoderRunners;
use crate::runner::media::MediaRunner;
use crate::service::ServiceBackend;
use crate::worker::RankStorage;
use crate::{Error, OutputPool, Result};

/// Serving-time kernel reporting: a cheap counter tells when any provider
/// choice changed, and only then are the bindings' records gathered.
pub struct KernelWatch {
    pub choices: Function,
    pub last: i64,
    pub records: KernelRecords,
    pub rank: usize,
    pub device: String,
}

/// The backend of one rank.
pub struct WorkerBackend {
    info: WorkerInfo,
    rank: usize,
    requests: RequestPool,
    /// The rank's storage owners (events, products, transports, KV cache).
    rank_storage: Option<RankStorage>,
    /// The text runtime, on ranks holding a causal language model.
    text: Option<crate::runner::causal_lm::CausalLMRunners>,
    /// The image runtime, on ranks holding encoders, image denoisers or
    /// image decoders of a model that is not a video model.
    image: Option<forward::image::ImageRuntime>,
    /// The canvas runtime, on ranks holding a token denoiser.
    canvas: Option<forward::canvas::CanvasRuntime>,
    /// The expert-parallel runtime, on ranks of an expert union.
    experts: Option<forward::experts::ExpertRuntime>,
    /// Distributed expert weights, on ranks that prefetch them.
    weights: Option<forward::experts::WeightRuntime>,
    /// The transfer runtime: tensor transfers, KV exports and installs, and
    /// the streams batch imports run on.
    transfer: Option<forward::transfer::TransferRuntime>,
    /// The admitted video requests, on ranks of a video model.
    video: Option<VideoRequests>,
    /// The rank's encoder, denoiser and decoder runners.
    encoders: EncoderRunners,
    denoisers: DenoiserRunners,
    decoders: DecoderRunners,
    /// The host media of a rank holding host components.
    media: Option<MediaRunner>,
    /// The output leases of media batches.
    media_outputs: Option<OutputPool>,
    /// Other capability runtimes' request storage.
    storage: Vec<Box<dyn RequestStorage>>,
    components: Components,
    profiler: Option<Profiler>,
    kernels: Option<KernelWatch>,
    /// The lanes' job queues: each forward's submission runs on the
    /// execution thread of its lane.
    jobs: JobQueues,
}

/// Disjoint borrows of a backend for one batch's execution.
pub(crate) struct Parts<'a> {
    pub info: &'a WorkerInfo,
    pub rank: usize,
    pub requests: &'a mut RequestPool,
    pub storage: Option<&'a mut RankStorage>,
    pub text: Option<&'a mut crate::runner::causal_lm::CausalLMRunners>,
    pub image: Option<&'a mut forward::image::ImageRuntime>,
    pub canvas: Option<&'a mut forward::canvas::CanvasRuntime>,
    pub experts: Option<&'a forward::experts::ExpertRuntime>,
    pub transfer: Option<&'a mut forward::transfer::TransferRuntime>,
}

/// Every request storage owner, in lifecycle order: rank storage, then the
/// text, image, canvas and transfer runtimes, a video model's requests,
/// denoiser runners and output leases, then the other runtimes' owners.
#[expect(
    clippy::too_many_arguments,
    reason = "one argument per owner the backend holds"
)]
fn owners<'a>(
    rank: &'a mut Option<RankStorage>,
    text: &'a mut Option<crate::runner::causal_lm::CausalLMRunners>,
    image: &'a mut Option<forward::image::ImageRuntime>,
    canvas: &'a mut Option<forward::canvas::CanvasRuntime>,
    transfer: &'a mut Option<forward::transfer::TransferRuntime>,
    video: &'a mut Option<VideoRequests>,
    denoisers: &'a mut DenoiserRunners,
    media_outputs: &'a mut Option<OutputPool>,
    others: &'a mut [Box<dyn RequestStorage>],
) -> Vec<&'a mut dyn RequestStorage> {
    let mut owners: Vec<&'a mut dyn RequestStorage> = Vec::new();
    if let Some(rank) = rank {
        owners.push(rank);
    }
    if let Some(text) = text {
        owners.push(text);
    }
    if let Some(image) = image {
        owners.push(image);
    }
    if let Some(canvas) = canvas {
        owners.push(canvas);
    }
    if let Some(transfer) = transfer {
        owners.push(transfer);
    }
    if let Some(video) = video {
        owners.push(video);
        owners.push(denoisers);
        if let Some(outputs) = media_outputs {
            owners.push(outputs);
        }
    }
    owners.extend(
        others
            .iter_mut()
            .map(|owner| &mut **owner as &mut dyn RequestStorage),
    );
    owners
}

/// A submitted batch with its lifecycle and numerical state.
pub struct WorkerBatch {
    plan: Arc<BatchPlan>,
    /// The accepted state call each call follows, by call.
    predecessors: Vec<Option<CallId>>,
    retirement: Retirement,
    /// A capability runtime's launched work awaiting materialization.
    running: Option<Launched>,
    /// The host preparation of an image encoding batch's inline images.
    image_inputs: Option<forward::image::Inputs>,
    /// Products and KV transfers imported from other calls.
    inputs: forward::inputs::Inputs,
    /// A media batch's calls and completion.
    media: Option<forward::batch::MediaBatch>,
    output: Option<BatchResult>,
}

impl WorkerBatch {
    pub fn new(plan: Arc<BatchPlan>) -> Self {
        let retirement = Retirement::new(&plan);
        Self {
            plan,
            predecessors: Vec::new(),
            retirement,
            running: None,
            image_inputs: None,
            inputs: forward::inputs::Inputs::default(),
            media: None,
            output: None,
        }
    }

    pub fn plan(&self) -> &BatchPlan {
        &self.plan
    }

    pub(crate) fn shared_plan(&self) -> &Arc<BatchPlan> {
        &self.plan
    }

    /// The batch's call kind; `None` for a lifecycle-only batch.
    pub fn route(&self) -> Option<CallKind> {
        self.plan.calls.first().map(|call| call.code)
    }

    pub fn predecessors(&self) -> &[Option<CallId>] {
        &self.predecessors
    }

    /// The calls a failure of this batch affects.
    fn identities(&self) -> Vec<ErrorCallIdentity> {
        self.plan
            .calls
            .iter()
            .map(|call| ErrorCallIdentity {
                request_key: call.request_key,
                call_id: call.call_id,
            })
            .collect()
    }

    /// Set the batch's result; capability runtimes call this once their
    /// outputs are materialized.
    pub fn complete(&mut self, result: BatchResult) {
        self.output = Some(result);
    }

    /// Hand launched work to materialization by [`Backend::poll`].
    pub(crate) fn launch(&mut self, running: Launched) {
        self.running = Some(running);
    }

    pub(crate) fn image_inputs(&self) -> Option<&forward::image::Inputs> {
        self.image_inputs.as_ref()
    }

    pub(crate) fn set_image_inputs(&mut self, inputs: forward::image::Inputs) {
        self.image_inputs = Some(inputs);
    }

    pub(crate) fn take_image_inputs(&mut self) -> Option<forward::image::Inputs> {
        self.image_inputs.take()
    }

    /// The batch's plan, the inputs it reads in place and its video batch
    /// state, borrowed apart.
    pub(crate) fn media_parts(
        &mut self,
    ) -> (
        &BatchPlan,
        HashSet<uniserve_ipc::BufferId>,
        &mut Option<forward::batch::MediaBatch>,
    ) {
        (&self.plan, self.inputs.borrowed(), &mut self.media)
    }

    /// The batch's imported inputs.
    pub(crate) fn inputs(&self) -> &forward::inputs::Inputs {
        &self.inputs
    }

    pub(crate) fn inputs_mut(&mut self) -> &mut forward::inputs::Inputs {
        &mut self.inputs
    }
}

impl WorkerBackend {
    pub fn new(info: WorkerInfo, rank: usize, components: Components) -> Result<Self> {
        Ok(Self {
            requests: RequestPool::new(info.request_slots as usize)?,
            info,
            rank,
            rank_storage: None,
            text: None,
            image: None,
            canvas: None,
            experts: None,
            weights: None,
            transfer: None,
            video: None,
            encoders: EncoderRunners::default(),
            denoisers: DenoiserRunners::default(),
            decoders: DecoderRunners::default(),
            media: None,
            media_outputs: None,
            storage: Vec::new(),
            components,
            profiler: None,
            kernels: None,
            jobs: JobQueues::default(),
        })
    }

    /// Use the job queues of the rank's lanes for forward submission.
    pub fn set_job_queues(&mut self, jobs: JobQueues) {
        self.jobs = jobs;
    }

    /// Issue one batch's forward: complete its imported inputs, then run
    /// its calls on the runtime serving `kind`.
    fn submit_forward(
        &mut self,
        batch: &mut WorkerBatch,
        kind: CallKind,
    ) -> std::result::Result<(), WorkerError> {
        // Imported values become resident before the runtime consumes them
        // like local ones; an import that failed fails the batch's calls.
        if let Err(error) = forward::inputs::complete(self, batch) {
            forward::fail(self, batch, error, "input import")?;
            self.release_predecessors(batch, true)?;
            return Ok(());
        }

        // Each executed batch with calls is one profiler step.
        let mut profiler = self.profiler.take();
        let executed = match &mut profiler {
            Some(profiler) => profiler.step(|| forward::execute(self, batch, kind)),
            None => forward::execute(self, batch, kind),
        };
        self.profiler = profiler;
        self.report_new_kernels();
        executed?;
        self.release_predecessors(batch, true)?;
        Ok(())
    }

    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    pub fn requests(&self) -> &RequestPool {
        &self.requests
    }

    pub fn requests_mut(&mut self) -> &mut RequestPool {
        &mut self.requests
    }

    pub fn components(&self) -> &Components {
        &self.components
    }

    pub fn components_mut(&mut self) -> &mut Components {
        &mut self.components
    }

    /// Install the rank's storage owners.
    pub fn set_rank_storage(&mut self, storage: RankStorage) {
        self.rank_storage = Some(storage);
    }

    pub fn rank_storage(&self) -> Option<&RankStorage> {
        self.rank_storage.as_ref()
    }

    pub fn rank_storage_mut(&mut self) -> Option<&mut RankStorage> {
        self.rank_storage.as_mut()
    }

    /// Install the text runtime.
    pub(crate) fn set_text(&mut self, text: crate::runner::causal_lm::CausalLMRunners) {
        self.text = Some(text);
    }

    pub(crate) fn text_mut(&mut self) -> Option<&mut crate::runner::causal_lm::CausalLMRunners> {
        self.text.as_mut()
    }

    /// Install the image runtime.
    pub(crate) fn set_image(&mut self, image: forward::image::ImageRuntime) {
        self.image = Some(image);
    }

    pub(crate) fn image_mut(&mut self) -> Option<&mut forward::image::ImageRuntime> {
        self.image.as_mut()
    }

    /// Install the canvas runtime.
    pub(crate) fn set_canvas(&mut self, canvas: forward::canvas::CanvasRuntime) {
        self.canvas = Some(canvas);
    }

    pub(crate) fn canvas_mut(&mut self) -> Option<&mut forward::canvas::CanvasRuntime> {
        self.canvas.as_mut()
    }

    /// Install the expert-parallel runtime.
    pub(crate) fn set_experts(&mut self, experts: forward::experts::ExpertRuntime) {
        self.experts = Some(experts);
    }

    /// Install the distributed expert weights.
    pub(crate) fn set_weights(&mut self, weights: forward::experts::WeightRuntime) {
        self.weights = Some(weights);
    }

    pub(crate) fn experts(&self) -> Option<&forward::experts::ExpertRuntime> {
        self.experts.as_ref()
    }

    pub(crate) fn experts_mut(&mut self) -> Option<&mut forward::experts::ExpertRuntime> {
        self.experts.as_mut()
    }

    /// Release the expert runtime after the bindings and graphs using its
    /// transports retired; collective unless `aborted`.
    pub(crate) fn close_experts(&mut self, aborted: bool) -> Result<()> {
        let experts = match &mut self.experts {
            Some(experts) => experts.close(aborted),
            None => Ok(()),
        };
        let weights = match &mut self.weights {
            Some(weights) => weights.close(aborted),
            None => Ok(()),
        };
        experts.and(weights)
    }

    /// Install the transfer runtime.
    pub(crate) fn set_transfer(&mut self, transfer: forward::transfer::TransferRuntime) {
        self.transfer = Some(transfer);
    }

    /// The stream serving `kind` calls on `device` (none on host devices):
    /// imports for such calls are read and published on it.
    pub(crate) fn transfer_stream(
        &self,
        kind: CallKind,
        device: crate::ffi::Device,
    ) -> Result<Option<uniserve_cuda::Stream>> {
        if !matches!(device, crate::ffi::Device::Cuda(_)) {
            return Ok(None);
        }
        self.transfer
            .as_ref()
            .ok_or_else(|| {
                Error::Unsupported("device transfers require the transfer runtime".into())
            })?
            .stream(kind, device)
    }

    /// Register another runtime's storage owner taking part in request
    /// lifecycles.
    pub fn add_storage(&mut self, storage: Box<dyn RequestStorage>) {
        self.storage.push(storage);
    }

    /// Disjoint borrows for executing one batch.
    pub(crate) fn parts(&mut self) -> Parts<'_> {
        Parts {
            info: &self.info,
            rank: self.rank,
            requests: &mut self.requests,
            storage: self.rank_storage.as_mut(),
            text: self.text.as_mut(),
            image: self.image.as_mut(),
            canvas: self.canvas.as_mut(),
            experts: self.experts.as_ref(),
            transfer: self.transfer.as_mut(),
        }
    }

    pub fn set_profiler(&mut self, profiler: Option<Profiler>) {
        self.profiler = profiler;
    }

    pub fn profiler_mut(&mut self) -> Option<&mut Profiler> {
        self.profiler.as_mut()
    }

    /// Report kernel selections that change while serving.
    pub fn watch_kernels(&mut self, watch: KernelWatch) {
        self.kernels = Some(watch);
    }

    /// Whether the rank serves a video model: its calls run as media
    /// batches on the rank's video runners.
    pub fn serves_video(&self) -> bool {
        self.video.is_some()
    }

    /// Install what serves a video model on this rank: its request
    /// records, its runners, its host media and its output leases.
    pub(crate) fn set_video(
        &mut self,
        video: VideoRequests,
        encoders: EncoderRunners,
        denoisers: DenoiserRunners,
        decoders: DecoderRunners,
        media: Option<MediaRunner>,
        outputs: OutputPool,
    ) {
        self.video = Some(video);
        self.encoders = encoders;
        self.denoisers = denoisers;
        self.decoders = decoders;
        self.media = media;
        self.media_outputs = Some(outputs);
    }

    pub(crate) fn encoders(&self) -> &EncoderRunners {
        &self.encoders
    }

    pub(crate) fn denoisers(&self) -> &DenoiserRunners {
        &self.denoisers
    }

    pub(crate) fn decoders(&self) -> &DecoderRunners {
        &self.decoders
    }

    /// The parts a video batch executes with.
    pub(crate) fn media_parts(&mut self) -> Result<forward::batch::Parts<'_>> {
        let unserved =
            || Error::Unsupported("video calls are served on a video model's rank".into());
        Ok(forward::batch::Parts {
            requests: &mut self.requests,
            storage: self
                .rank_storage
                .as_ref()
                .ok_or_else(|| Error::Invariant("the rank has no storage owners".into()))?,
            info: &self.info,
            rank: self.rank,
            video: self.video.as_mut().ok_or_else(unserved)?,
            encoders: &mut self.encoders,
            denoisers: &mut self.denoisers,
            decoders: &mut self.decoders,
            media: self.media.as_ref(),
            outputs: self.media_outputs.as_mut().ok_or_else(unserved)?,
        })
    }

    /// Install every Start, then follow each call's predecessor.
    fn admit_requests(&mut self, batch: &mut WorkerBatch) -> Result<()> {
        let mut admitted = Vec::new();
        let mut installed = Vec::new();
        let started = (|| -> Result<()> {
            for command in &batch.plan.commands {
                if let BatchCommand::Start { request } = command
                    && let Some(slot) = self.requests.start(request.as_ref().clone())?
                {
                    admitted.push(slot);
                    installed.push(request);
                }
            }
            Ok(())
        })();
        // Reset every admitted slot even when a later Start failed.
        if !admitted.is_empty() {
            for owner in owners(
                &mut self.rank_storage,
                &mut self.text,
                &mut self.image,
                &mut self.canvas,
                &mut self.transfer,
                &mut self.video,
                &mut self.denoisers,
                &mut self.media_outputs,
                &mut self.storage,
            ) {
                owner.admit_slots(&admitted)?;
            }
        }
        started?;
        if let Some(video) = &mut self.video {
            for request in installed {
                video.admit(request, &mut self.denoisers)?;
            }
        }

        batch.predecessors = batch
            .plan
            .calls
            .iter()
            .map(|call| {
                self.requests
                    .predecessor(call.request_key, call.advances_state())
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// Refuse a batch outside this rank's advertised capacity.
    fn validate(&self, plan: &BatchPlan) -> Result<()> {
        let unsupported: BTreeSet<&str> = plan
            .calls
            .iter()
            .filter(|call| !self.info.supported_calls.contains(&call.code))
            .map(|call| call.code.as_str())
            .collect();
        if !unsupported.is_empty() {
            return Err(Error::Invalid(format!(
                "execution batch contains call kinds unsupported by this worker: {:?}",
                unsupported.into_iter().collect::<Vec<_>>()
            )));
        }
        if let Some(cache) = &self.info.kv_cache
            && plan
                .new_cache_units
                .iter()
                .flat_map(|allocation| &allocation.unit_ids)
                .any(|unit| unit.0 >= cache.num_units)
        {
            return Err(Error::Invalid(
                "KV allocation exceeds the fixed physical pool".into(),
            ));
        }
        if let Some(allocation) = plan
            .buffer_allocations
            .iter()
            .find(|allocation| allocation.offset + allocation.bytes > self.info.buffer_pool_bytes)
        {
            return Err(Error::Invalid(format!(
                "batch buffer allocation exceeds the worker buffer pool: offset={}, bytes={}, capacity={}, buffer={:?}",
                allocation.offset, allocation.bytes, self.info.buffer_pool_bytes, allocation.buffer,
            )));
        }
        if !self.info.components.is_empty() {
            for call in &plan.calls {
                if !self.info.components.iter().any(|component| {
                    component.name == call.component && component.config.ranks.contains(&self.rank)
                }) {
                    return Err(Error::Invalid(format!(
                        "call targets component {:?} outside this rank",
                        call.component
                    )));
                }
            }
        }
        if plan.calls.len() > self.info.max_batch_calls as usize {
            return Err(Error::Invalid(
                "execution batch exceeds the worker_config call limit".into(),
            ));
        }
        if plan
            .block_tables
            .iter()
            .map(|table| table.request_pool_idx)
            .chain(plan.forward.request_pool_indices.iter().copied())
            .any(|slot| slot > self.info.request_slots)
        {
            return Err(Error::Invalid(
                "execution batch exceeds request-slot capacity".into(),
            ));
        }
        Ok(())
    }

    fn report_new_kernels(&mut self) {
        let Some(watch) = &mut self.kernels else {
            return;
        };
        let Ok(choices) = watch.choices.call_as::<i64>(&[]) else {
            return;
        };
        if choices == watch.last {
            return;
        }
        watch.last = choices;
        let mut current = KernelRecords::default();
        for binding in self.components.bindings() {
            let label = binding_label(binding);
            let added = binding
                .kernels()
                .and_then(|records| current.add(&label, &binding.device.to_string(), &records));
            if let Err(error) = added {
                tracing::warn!(target: "uniserve_worker.profiling", "kernel report failed: {error}");
                return;
            }
        }
        if watch.records.merge(current) {
            watch.records.log("serving", watch.rank, &watch.device);
        }
    }

    /// Install (or clear) the service's completion wake on every storage
    /// owner.
    pub fn set_storage_wake(&self, wake: Option<uniserve_ipc::Wake>) {
        if let Some(storage) = &self.rank_storage {
            storage.set_wake(wake.clone());
        }
        if let Some(text) = &self.text {
            text.set_wake(wake.clone());
        }
        if let Some(image) = &self.image {
            image.set_wake(wake.clone());
        }
        if let Some(transfer) = &self.transfer {
            transfer.set_wake(wake.clone());
        }
        if let Some(media) = &self.media {
            media.lane.set_wake(wake.clone());
        }
        for owner in &self.storage {
            owner.set_wake(wake.clone());
        }
    }

    /// Close every storage owner, newest first (other runtimes, a video
    /// model's owners, the transfer, canvas, image and text runtimes, then
    /// rank storage), after
    /// the executor drained; returns the first failure.
    pub fn close_storage(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        while let Some(mut owner) = self.storage.pop() {
            result = result.and(owner.close(aborted));
        }
        // A video model's owners close before the rank storage their
        // products live in: the host lane, the request records, then the
        // runners and the output leases.
        if let Some(mut video) = self.video.take() {
            if let Some(media) = &self.media {
                result = result.and(media.close(aborted));
            }
            result = result.and(video.close(aborted));
            result = result.and(self.encoders.close_text(aborted));
            result = result.and(self.denoisers.close(aborted));
            result = result.and(self.decoders.close(aborted));
            result = result.and(self.encoders.close_latent(aborted));
            if let Some(mut outputs) = self.media_outputs.take() {
                result = result.and(outputs.close(aborted));
            }
        }
        if let Some(mut transfer) = self.transfer.take() {
            result = result.and(transfer.close(aborted));
        }
        if let Some(mut canvas) = self.canvas.take() {
            result = result.and(canvas.close(aborted));
        }
        if let Some(mut image) = self.image.take() {
            result = result.and(image.close(aborted));
        }
        if let Some(mut text) = self.text.take() {
            result = result.and(text.close(aborted));
        }
        if let Some(mut storage) = self.rank_storage.take() {
            result = result.and(storage.close(aborted));
        }
        result
    }

    /// Close the canvas, image and text runtimes (graphs, then steps)
    /// before the bindings they run close; attempts every one before
    /// returning the first failure.
    pub fn close_runtimes(&mut self, aborted: bool) -> Result<()> {
        let canvas = self
            .canvas
            .take()
            .map_or(Ok(()), |mut canvas| canvas.close(aborted));
        let image = self
            .image
            .take()
            .map_or(Ok(()), |mut image| image.close(aborted));
        let text = self
            .text
            .take()
            .map_or(Ok(()), |mut text| text.close(aborted));
        canvas.and(image).and(text)
    }

    /// Release the outputs of the predecessors this batch's calls follow.
    ///
    /// Before inputs are acquired, a predecessor no call reads can retire
    /// early; once the batch executed, the predecessors it read retire too.
    /// The admission root (batch 0) has no outputs.
    fn release_predecessors(&mut self, batch: &WorkerBatch, inputs_acquired: bool) -> Result<()> {
        let Some(storage) = self.rank_storage.as_mut() else {
            return Ok(());
        };
        let mut consumed = HashSet::new();
        for call in &batch.plan.calls {
            consumed.extend(
                call.tensor_inputs()
                    .chain(call.predicate.iter())
                    .map(|input| (input.request_key, input.producer_call_id)),
            );
            if let Some(input) = call.kv_input {
                consumed.insert((input.owner, input.producer_call_id));
            }
        }
        let releases: Vec<_> = batch
            .plan
            .calls
            .iter()
            .zip(&batch.predecessors)
            .filter_map(|(call, previous)| {
                let previous = previous.as_ref()?;
                let source = (call.request_key, *previous);
                (previous.batch_id > 0 && consumed.contains(&source) == inputs_acquired)
                    .then_some(source)
            })
            .collect();
        storage.release_calls(&releases)?;
        if inputs_acquired {
            // Predicates outside the request's predecessor were consumed
            // here as well; releasing that other call would be too broad.
            let predicates: HashSet<_> = batch
                .plan
                .calls
                .iter()
                .zip(&batch.predecessors)
                .filter_map(|(call, previous)| {
                    call.predicate
                        .as_ref()
                        .filter(|value| Some(value.producer_call_id) != *previous)
                        .map(|value| value.buffer_id())
                })
                .collect();
            if !predicates.is_empty() {
                storage.release_buffers(&predicates)?;
            }
        }
        Ok(())
    }

    /// Release request references after the executor drained its work.
    pub fn close(&mut self) {
        self.requests.close();
        if let Some(profiler) = &mut self.profiler {
            profiler.close();
        }
    }
}

/// A binding's kernel-table label: component, root path and call kinds.
pub fn binding_label(binding: &crate::model::bind::ComponentBinding) -> String {
    if binding.path.is_empty() {
        binding.component.clone()
    } else {
        format!("{}.{}", binding.component, binding.path)
    }
}

impl Backend for WorkerBackend {
    type Batch = WorkerBatch;
    type Output = BatchResult;
    type Error = WorkerError;

    fn error(&self, error: Error) -> WorkerError {
        WorkerError::classify(error, None)
    }

    fn classify(&self, error: WorkerError, batch: &WorkerBatch, _context: &str) -> WorkerError {
        let error = error.with_calls(batch.identities());
        match batch.route() {
            Some(kind) => error.with_route(kind.as_str()),
            None => error,
        }
    }

    fn note_cleanup(&self, error: &mut WorkerError, cleanup: WorkerError) {
        error.note(&cleanup);
    }

    fn admit(&mut self, batch: &mut WorkerBatch) -> std::result::Result<(), WorkerError> {
        // Explicit revocation must not wait behind an allocation-dependent
        // batch; existing readers retain their leases.
        let mut owners = owners(
            &mut self.rank_storage,
            &mut self.text,
            &mut self.image,
            &mut self.canvas,
            &mut self.transfer,
            &mut self.video,
            &mut self.denoisers,
            &mut self.media_outputs,
            &mut self.storage,
        );
        batch.retirement.admit(&mut owners)?;
        Ok(())
    }

    fn prepare(&mut self, batch: &mut WorkerBatch) -> std::result::Result<(), WorkerError> {
        self.validate_kinds(batch)?;
        self.admit_requests(batch)?;
        self.validate(&batch.plan)?;
        for command in &batch.plan.commands {
            if let BatchCommand::Finish { request_key, .. } = command {
                self.requests.finish(*request_key)?;
            }
        }
        self.release_predecessors(batch, false)?;
        Ok(())
    }

    fn prepare_inputs(
        &mut self,
        batch: &mut WorkerBatch,
        submission: &Arc<Submission>,
    ) -> std::result::Result<bool, WorkerError> {
        // Products and KV transfers other calls published are imported
        // first, for every runtime alike (`await_inputs` subscribes to what
        // blocks them).
        if !forward::inputs::prepare(self, batch)? {
            return Ok(false);
        }
        // Inline images of an image encoding batch are decoded on the image
        // runtime's host lane while other batches execute; the batch
        // launches once they are ready (the lane notifies `submission`).
        // Other batches' inputs are prepared when their runtime executes.
        if self.image.is_some() && !self.serves_video() {
            return forward::image::prepare_inputs(self, batch, submission);
        }
        Ok(true)
    }

    fn await_inputs(
        &mut self,
        batch: &mut WorkerBatch,
        submission: &Arc<Submission>,
    ) -> std::result::Result<(), WorkerError> {
        Ok(forward::inputs::subscribe(self, batch, submission)?)
    }

    fn execute(&mut self, batch: &mut WorkerBatch) -> std::result::Result<(), WorkerError> {
        let Some(kind) = batch.route() else {
            batch.output = Some(BatchResult {
                output: BatchOutput {
                    batch_id: batch.plan.batch_id,
                    completions: Vec::new(),
                    products: Vec::new(),
                    worker_exec_us: None,
                    forward_stats: None,
                },
                media: Vec::new(),
            });
            return Ok(());
        };

        // The forward's submission is one job of its lane: the lane's
        // execution thread issues everything the forward puts on the
        // stream, and this thread waits for it.
        let jobs = self.jobs.clone();
        jobs.run(kind, || self.submit_forward(batch, kind))
    }

    fn begin_retirement(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(), WorkerError> {
        let mut owners = owners(
            &mut self.rank_storage,
            &mut self.text,
            &mut self.image,
            &mut self.canvas,
            &mut self.transfer,
            &mut self.video,
            &mut self.denoisers,
            &mut self.media_outputs,
            &mut self.storage,
        );
        batch.retirement.begin(&self.requests, &mut owners)?;
        Ok(())
    }

    fn poll(&mut self, batch: &mut WorkerBatch) -> std::result::Result<(bool, bool), WorkerError> {
        let mut advanced = false;
        if batch.output.is_none()
            && let Some(running) = batch.running.as_mut()
            && let Some(result) = forward::poll(self, &batch.plan, running)?
        {
            batch.output = Some(result);
            batch.running = None;
            advanced = true;
        }
        if batch.output.is_none()
            && let Some(media) = &mut batch.media
        {
            let parts = self.media_parts()?;
            let owners = parts.owners();
            if let Some(result) = media.poll(&batch.plan, owners, parts.requests)? {
                batch.output = Some(result);
                advanced = true;
            }
        }
        if batch.output.is_none() {
            return Ok((advanced, false));
        }
        let mut owners = owners(
            &mut self.rank_storage,
            &mut self.text,
            &mut self.image,
            &mut self.canvas,
            &mut self.transfer,
            &mut self.video,
            &mut self.denoisers,
            &mut self.media_outputs,
            &mut self.storage,
        );
        let retired = batch.retirement.poll(&mut self.requests, &mut owners)?;
        Ok((advanced || retired, retired))
    }

    fn result(&mut self, batch: &mut WorkerBatch) -> std::result::Result<BatchResult, WorkerError> {
        let result = batch
            .output
            .as_mut()
            .ok_or_else(|| WorkerError::from(Error::State("batch output is not resolved")))?;
        Ok(BatchResult {
            output: result.output.clone(),
            media: std::mem::take(&mut result.media),
        })
    }

    fn close(&mut self, batch: &mut WorkerBatch) -> std::result::Result<(), WorkerError> {
        // Calls become pending only when a capability runtime commits them;
        // unresolved launched work cancels its pending calls. Lifecycle
        // commands hold no resource beyond their retirement.
        let mut abandoned = batch
            .running
            .take()
            .map_or(Ok(()), |running| forward::abandon(self, running));
        if let Some(media) = &mut batch.media {
            let closed = self.media_parts().and_then(|parts| {
                let owners = parts.owners();
                media.close(owners, parts.requests)
            });
            abandoned = abandoned.and(closed.map_err(WorkerError::from));
        }
        // Imports the batch did not consume are abandoned; their storage
        // retires with their physical accesses.
        let closed = forward::inputs::close(self, batch).map_err(WorkerError::from);
        abandoned.and(closed)
    }

    fn reap(&mut self) -> std::result::Result<(), WorkerError> {
        for owner in owners(
            &mut self.rank_storage,
            &mut self.text,
            &mut self.image,
            &mut self.canvas,
            &mut self.transfer,
            &mut self.video,
            &mut self.denoisers,
            &mut self.media_outputs,
            &mut self.storage,
        ) {
            owner.reap()?;
        }
        Ok(())
    }
}

impl WorkerBackend {
    fn validate_kinds(&self, batch: &WorkerBatch) -> Result<()> {
        let kinds: BTreeSet<CallKind> = batch.plan.calls.iter().map(|call| call.code).collect();
        if kinds.len() > 1 {
            return Err(Error::Invalid(
                "an execution batch carries calls of one kind".into(),
            ));
        }
        Ok(())
    }
}

impl ServiceBackend for WorkerBackend {
    fn batch(&self, plan: Arc<BatchPlan>) -> std::result::Result<WorkerBatch, WorkerError> {
        Ok(WorkerBatch::new(plan))
    }

    fn response_error(
        &self,
        kind: RequestKind,
        error: WorkerError,
    ) -> std::result::Result<WorkerResponseError, WorkerError> {
        let calls: Vec<String> = error
            .calls()
            .iter()
            .map(|call| format!("{}/{:?}", call.request_key.request_id.0, call.call_id))
            .collect();
        if error.code.traces() {
            tracing::error!(
                target: "uniserve_worker.profiling",
                "worker request {:?} failed: {error} [code={} calls={calls:?}]{}",
                kind.as_str(),
                error.code,
                error
                    .traceback()
                    .map(|traceback| format!("\n{traceback}"))
                    .unwrap_or_default(),
            );
        } else {
            tracing::warn!(
                target: "uniserve_worker.profiling",
                "worker request {:?} failed: {error} [code={} calls={calls:?}]",
                kind.as_str(),
                error.code,
            );
        }
        Ok(error.response())
    }

    fn has_open_requests(&self) -> std::result::Result<bool, WorkerError> {
        Ok(self.requests.has_open_requests()?)
    }

    fn awaiting_acknowledgment(&self) -> std::result::Result<bool, WorkerError> {
        Ok(self
            .rank_storage
            .as_ref()
            .is_some_and(|storage| storage.awaiting_acknowledgment())
            || self
                .storage
                .iter()
                .any(|owner| owner.awaiting_acknowledgment()))
    }

    fn join_expert_step(&self, leaving: bool) -> std::result::Result<(bool, bool), WorkerError> {
        // A join launches on the lane the union's forwards run on.
        self.jobs.run(CallKind::Forward(ForwardMode::Decode), || {
            forward::experts::join_step(self, leaving)
        })
    }
}
