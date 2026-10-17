//! The worker's execution backend: request state, lifecycle commands,
//! batch validation, numerical dispatch and result delivery.
//!
//! [`WorkerBackend`] does the work of each step the [`super::Executor`]
//! orders and answers the rank service. It owns the request pool, the
//! request storage owners (through [`RequestStorage`]) and the placed
//! components with their bindings, and dispatches each batch's calls to the
//! runner that serves them, whose forward it runs as jobs of the batch's
//! lane.
//! Lifecycle commands (Start, Finish, Free), batch admission, dependencies,
//! retirement and result delivery are handled here for every batch, with or
//! without calls.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use uniserve_core::CallId;
use uniserve_ipc::{
    Batch as BatchPlan, BatchCommand, BatchOutput, CallKind, ErrorCallIdentity, ForwardMode,
    RequestKind, WorkerInfo, WorkerResponseError,
};

use super::batch::{self, Launched};
use super::image::ImageParts;
use super::lifecycle::{LaneJobs, RequestStorage, Retirement};
use super::{BatchResult, Submission};
use crate::config::placement::Components;
use crate::error::WorkerError;
use crate::host::HostLane;
use crate::host::image::ImageTask;
use crate::lane::{Completion, Job, JobId, JobQueues};
use crate::profiling::kernels::{KernelCalls, KernelRecord, KernelWatch};
use crate::profiling::profiler::Profiler;
use crate::request::RequestPool;
use crate::request::diffusion::Trajectories;
use crate::request::video::VideoRequests;
use crate::runner::decoder::runners::DecoderRunners;
use crate::runner::denoiser::image_model::{ImageModel, ImagePolicy};
use crate::runner::denoiser::runners::DenoiserRunners;
use crate::runner::encoder::inputs::Inputs;
use crate::runner::encoder::runners::EncoderRunners;
use crate::runner::media::MediaRunner;
use crate::runner::methods::list_kernels;
use crate::runner::start::{ImageOwners, ImageRunner};
use crate::storage::LatentCache;
use crate::worker::RankStorage;
use crate::{Error, OutputPool, Result};

/// The backend of one rank.
pub(crate) struct WorkerBackend {
    info: WorkerInfo,
    rank: usize,
    requests: RequestPool,
    /// The rank's storage owners (events, products, transports, prefix cache).
    rank_storage: Option<RankStorage>,
    /// The text runtime, on ranks holding a causal language model.
    text: Option<crate::runner::causal_lm::CausalLMRunners>,
    /// An image model's facts and host policy, on ranks holding encoders,
    /// image denoisers or image decoders of a model that is not a video
    /// model.
    image: Option<(ImageModel, ImagePolicy)>,
    /// The latent pool of a rank holding an image denoiser.
    latents: Option<LatentCache>,
    /// The open image trajectory of each request slot.
    trajectories: Trajectories,
    /// The lane preparing input images and encoding output images.
    image_host: Option<HostLane<ImageTask>>,
    /// An image model's runners with their names, in construction order.
    image_order: Vec<(ImageRunner, String)>,
    /// The runner of each diffusion LM component placed on the rank.
    diffusion_lm: Vec<crate::runner::diffusion_lm::DiffusionLMRunner>,
    /// This rank's member of its expert union, shared with the jobs that
    /// run an expert step: the lane's execution thread agrees on the step
    /// and runs it.
    experts: Option<Arc<crate::parallel::ExpertUnion>>,
    /// Distributed expert weights, on ranks that prefetch them.
    weights: Option<crate::parallel::DwdpManager>,
    /// The transfer runtime: tensor transfers, cache exports and installs, and
    /// the streams batch imports run on.
    transfer: Option<super::transfer::TransferRuntime>,
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
    /// The lanes' job queues: each forward runs as jobs of its lane.
    jobs: JobQueues,
    /// What lifecycle commands record for the lanes; submitted after each
    /// command.
    lifecycle: LaneJobs,
    /// The submitted lifecycle jobs whose completions have not arrived.
    lifecycle_jobs: HashSet<JobId>,
    /// The expert union's step this rank joins without a forward of its
    /// own, while its job is in flight.
    joining: Option<JobId>,
    /// How the last such join ended, until the service takes it.
    joined: Option<std::result::Result<(bool, bool), WorkerError>>,
}

/// Disjoint borrows of a backend for one batch's execution.
pub(crate) struct Parts<'a> {
    pub info: &'a WorkerInfo,
    pub rank: usize,
    pub requests: &'a mut RequestPool,
    pub storage: Option<&'a mut RankStorage>,
    pub text: Option<&'a mut crate::runner::causal_lm::CausalLMRunners>,
    pub image: Option<ImageParts<'a>>,
    pub diffusion_lm: &'a mut [crate::runner::diffusion_lm::DiffusionLMRunner],
    pub experts: Option<&'a Arc<crate::parallel::ExpertUnion>>,
    pub transfer: Option<&'a mut super::transfer::TransferRuntime>,
}

/// Every request storage owner, in lifecycle order: rank storage, the text
/// runners, an image model's trajectories, output leases and latent pool,
/// the diffusion LM runners, the transfer runtime, a video model's requests, denoiser
/// runners and output leases, then the other runtimes' owners.
#[expect(
    clippy::too_many_arguments,
    reason = "one argument per owner the backend holds"
)]
fn owners<'a>(
    rank: &'a mut Option<RankStorage>,
    text: &'a mut Option<crate::runner::causal_lm::CausalLMRunners>,
    image: bool,
    trajectories: &'a mut Trajectories,
    latents: &'a mut Option<LatentCache>,
    diffusion_lm: &'a mut [crate::runner::diffusion_lm::DiffusionLMRunner],
    transfer: &'a mut Option<super::transfer::TransferRuntime>,
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
    let mut media_outputs = media_outputs.as_mut();
    if image {
        owners.push(trajectories);
        if let Some(outputs) = media_outputs.take() {
            owners.push(outputs);
        }
        if let Some(latents) = latents {
            owners.push(latents);
        }
    }
    for runner in diffusion_lm {
        owners.push(runner);
    }
    if let Some(transfer) = transfer {
        owners.push(transfer);
    }
    if let Some(video) = video {
        owners.push(video);
        owners.push(denoisers);
        if let Some(outputs) = media_outputs.take() {
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
pub(crate) struct WorkerBatch {
    plan: Arc<BatchPlan>,
    /// The accepted state call each call follows, by call.
    predecessors: Vec<Option<CallId>>,
    retirement: Retirement,
    /// Where the batch's forward stands between its jobs.
    stage: batch::Stage,
    /// What a forward that issues itself kept for its submitter: how the
    /// completion of the imported inputs, or a forward inside a converted
    /// batch, ended.
    issued: Option<Result<()>>,
    /// A token batch between its jobs.
    tokens: Option<crate::runner::causal_lm::Pending>,
    /// A diffusion LM batch between its jobs.
    denoising: Option<crate::runner::diffusion_lm::Pending>,
    /// A transfer batch between its jobs.
    transfer: Option<super::transfer::Pending>,
    /// The batch's profiler step is open: its forward began and has not
    /// ended.
    stepping: bool,
    /// The job of the profiler's window this batch's forward waits for.
    profiling: Option<JobId>,
    /// A capability runtime's launched work awaiting materialization.
    running: Option<Launched>,
    /// The host preparation of an image encoding batch's inline images.
    image_inputs: Option<Inputs>,
    /// An image batch between two of its jobs.
    image: Option<super::image::Pending>,
    /// The video requests the batch started, while the library sizes them.
    sizing: Option<crate::request::video::Sizing>,
    /// Products and cache transfers imported from other calls.
    inputs: super::inputs::Inputs,
    /// A video model's media batch between two of its jobs.
    media: Option<super::video::Pending>,
    output: Option<BatchResult>,
}

impl WorkerBatch {
    /// Attribute a failure to the batch's calls and its route.
    pub(crate) fn attribute(&self, error: WorkerError) -> WorkerError {
        let error = error.with_calls(self.identities());
        match self.route() {
            Some(kind) => error.with_route(kind.as_str()),
            None => error,
        }
    }

    pub(crate) fn new(plan: Arc<BatchPlan>) -> Self {
        let retirement = Retirement::new(&plan);
        Self {
            plan,
            predecessors: Vec::new(),
            retirement,
            stage: batch::Stage::default(),
            issued: None,
            tokens: None,
            denoising: None,
            transfer: None,
            stepping: false,
            profiling: None,
            running: None,
            image_inputs: None,
            image: None,
            sizing: None,
            inputs: super::inputs::Inputs::default(),
            media: None,
            output: None,
        }
    }

    pub(crate) fn shared_plan(&self) -> &Arc<BatchPlan> {
        &self.plan
    }

    /// The batch's call kind; `None` for a lifecycle-only batch.
    pub(crate) fn route(&self) -> Option<CallKind> {
        self.plan.calls.first().map(|call| call.code)
    }

    pub(crate) fn predecessors(&self) -> &[Option<CallId>] {
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

    pub(crate) fn stage(&self) -> batch::Stage {
        self.stage
    }

    pub(crate) fn set_stage(&mut self, stage: batch::Stage) {
        self.stage = stage;
    }

    /// The batch's plan, its token batch between jobs and where a forward
    /// inside it keeps its result, borrowed apart.
    pub(crate) fn token_parts(
        &mut self,
    ) -> (
        &BatchPlan,
        &mut Option<crate::runner::causal_lm::Pending>,
        &mut Option<Result<()>>,
    ) {
        (&self.plan, &mut self.tokens, &mut self.issued)
    }

    /// The batch's plan and its diffusion LM batch between jobs, borrowed
    /// apart.
    pub(crate) fn denoising_parts(
        &mut self,
    ) -> (
        &BatchPlan,
        &mut Option<crate::runner::diffusion_lm::Pending>,
    ) {
        (&self.plan, &mut self.denoising)
    }

    /// The batch's plan, its imported inputs and its transfer batch between
    /// jobs, borrowed apart.
    pub(crate) fn transfer_parts(
        &mut self,
    ) -> (
        &BatchPlan,
        &super::inputs::Inputs,
        &mut Option<super::transfer::Pending>,
    ) {
        (&self.plan, &self.inputs, &mut self.transfer)
    }

    /// Set the batch's result; capability runtimes call this once their
    /// outputs are materialized.
    pub(crate) fn complete(&mut self, result: BatchResult) {
        self.output = Some(result);
    }

    /// Hand launched work to materialization by [`WorkerBackend::poll`].
    pub(crate) fn launch(&mut self, running: Launched) {
        self.running = Some(running);
    }

    pub(crate) fn image_inputs(&self) -> Option<&Inputs> {
        self.image_inputs.as_ref()
    }

    pub(crate) fn set_image_inputs(&mut self, inputs: Inputs) {
        self.image_inputs = Some(inputs);
    }

    /// The batch's plan, its image batch between jobs and the inline
    /// images prepared for it, borrowed apart.
    pub(crate) fn image_parts(
        &mut self,
    ) -> (
        &BatchPlan,
        &mut Option<super::image::Pending>,
        &mut Option<Inputs>,
    ) {
        (&self.plan, &mut self.image, &mut self.image_inputs)
    }

    /// The batch's plan, its imported inputs and its media batch between
    /// jobs, borrowed apart.
    pub(crate) fn media_parts(
        &mut self,
    ) -> (
        &BatchPlan,
        &super::inputs::Inputs,
        &mut Option<super::video::Pending>,
    ) {
        (&self.plan, &self.inputs, &mut self.media)
    }

    /// The batch's imported inputs.
    pub(crate) fn inputs(&self) -> &super::inputs::Inputs {
        &self.inputs
    }

    pub(crate) fn inputs_mut(&mut self) -> &mut super::inputs::Inputs {
        &mut self.inputs
    }
}

impl WorkerBackend {
    pub(crate) fn new(info: WorkerInfo, rank: usize, components: Components) -> Result<Self> {
        let lifecycle = LaneJobs::new(&info.supported_calls);
        Ok(Self {
            requests: RequestPool::new(info.request_slots as usize)?,
            info,
            rank,
            rank_storage: None,
            text: None,
            image: None,
            latents: None,
            trajectories: Trajectories::default(),
            image_host: None,
            image_order: Vec::new(),
            diffusion_lm: Vec::new(),
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
            lifecycle,
            lifecycle_jobs: HashSet::new(),
            joining: None,
            joined: None,
        })
    }

    /// Submit the jobs the last lifecycle command recorded, each to the
    /// lane of its kind, without waiting for them: a lane performs them in
    /// the order of its jobs, so a forward submitted later finds their work
    /// done. Nothing goes on from their completions, which are taken with
    /// the next one that arrives ([`WorkerBackend::completed`]); a failed one
    /// stops the rank there.
    fn perform_lifecycle(&mut self) -> Result<()> {
        for (kind, job) in self.lifecycle.take() {
            let submitted = self.start(kind, job, false)?;
            self.lifecycle_jobs.insert(submitted);
        }
        Ok(())
    }

    /// Submit `job` to the lane serving `kind` (the Python thread for a
    /// rank that serves no call kind and so has no lane). `wake` is as for
    /// [`JobQueues::start`].
    ///
    /// The deferred work the job carries waits for events the job records.
    /// The job whose completion says they completed is started right
    /// behind it, on the same lane, and the event pool runs the work when
    /// that completion arrives ([`WorkerBackend::completed`]). Started earlier,
    /// that job would reach the lane before the records and pass at once.
    fn start(&self, kind: Option<CallKind>, job: Job, wake: bool) -> Result<JobId> {
        let deferrals = job.deferrals();
        let records = job.records();
        let lane = kind.and_then(|kind| self.jobs.lane(kind));
        let submitted = match kind {
            Some(kind) => self.jobs.start(kind, job, wake)?,
            None => self.jobs.call(job, wake),
        };
        // Work deferred later behind these records waits on their lane.
        if let (Some(lane), Some(storage)) = (lane, &self.rank_storage)
            && !records.is_empty()
        {
            storage.events.queued(submitted, lane, &records);
        }
        if !deferrals.is_empty() {
            let storage = self.rank_storage.as_ref().ok_or(Error::State(
                "a job defers work on a rank without an event pool",
            ))?;
            let jobs = &self.jobs;
            storage
                .events
                .defer_submitted(deferrals, |device, events| {
                    after(jobs, kind, device, events)
                })?;
        }
        Ok(submitted)
    }

    /// Have the rank's event pool start the jobs its deferred work waits
    /// for on the rank's lanes, from this thread, the control thread.
    fn serve_events(&self) {
        let Some(storage) = &self.rank_storage else {
            return;
        };
        let jobs = self.jobs.clone();
        // Work deferred outside a recording waits on the lane whose job
        // queued its events' records; once they are made, any lane's
        // notification stream waits for them: the first kind a lane serves.
        let mut kinds = self.info.supported_calls.clone();
        kinds.sort_by_key(|&kind| jobs.lane(kind).is_none());
        let kind = kinds.first().copied();
        storage
            .events
            .set_submitter(Some(Box::new(move |device, lane, events| match lane {
                Some(lane) => jobs.after_on(lane, device, events),
                None => after(&jobs, kind, device, events),
            })));
    }

    /// Use the job queues of the rank's lanes for forward submission.
    pub(crate) fn set_job_queues(&mut self, jobs: JobQueues) {
        // Lifecycle work that belongs to no lane's stream (deferred work
        // behind an event) is recorded for the first kind: one a lane
        // serves, where the rank has lanes, since only a lane waits for an
        // event.
        let mut kinds = self.info.supported_calls.clone();
        kinds.sort_by_key(|&kind| jobs.lane(kind).is_none());
        self.lifecycle = LaneJobs::new(&kinds);
        self.jobs = jobs;
        self.serve_events();
    }

    /// End a batch's forward: close its profiler step and ask for the
    /// count of kernel choices, which tells whether its calls selected new
    /// kernels.
    fn end_forward(&mut self, batch: &mut WorkerBatch) {
        if std::mem::take(&mut batch.stepping) {
            if let Some(profiler) = &mut self.profiler {
                profiler.end();
            }
            if let Some(watch) = &mut self.kernels {
                watch.count(&self.jobs);
            }
        }
    }

    pub(crate) fn info(&self) -> &WorkerInfo {
        &self.info
    }

    pub(crate) fn requests(&self) -> &RequestPool {
        &self.requests
    }

    pub(crate) fn requests_mut(&mut self) -> &mut RequestPool {
        &mut self.requests
    }

    pub(crate) fn components(&self) -> &Components {
        &self.components
    }

    /// Install the rank's storage owners.
    pub(crate) fn set_rank_storage(&mut self, storage: RankStorage) {
        self.rank_storage = Some(storage);
        self.serve_events();
    }

    pub(crate) fn rank_storage(&self) -> Option<&RankStorage> {
        self.rank_storage.as_ref()
    }

    pub(crate) fn rank_storage_mut(&mut self) -> Option<&mut RankStorage> {
        self.rank_storage.as_mut()
    }

    /// Install the text runtime.
    pub(crate) fn set_text(&mut self, text: crate::runner::causal_lm::CausalLMRunners) {
        self.text = Some(text);
    }

    pub(crate) fn text_mut(&mut self) -> Option<&mut crate::runner::causal_lm::CausalLMRunners> {
        self.text.as_mut()
    }

    /// Install what serves an image model on this rank.
    pub(crate) fn set_image(&mut self, owners: ImageOwners) {
        self.image = Some((owners.model, owners.policy));
        self.encoders = owners.encoders;
        self.denoisers = owners.denoisers;
        self.decoders = owners.decoders;
        self.latents = owners.latents;
        self.media_outputs = Some(owners.outputs);
        self.image_host = Some(owners.host);
        self.image_order = owners.order;
    }

    /// The kernels each text and image runner's operators selected, as
    /// `(label, device, kernels)`: text runners first, then the image
    /// runners in construction order. Each listing is a call made now,
    /// which startup waits for.
    pub(crate) fn kernels(&self) -> Result<Vec<(String, String, Vec<KernelRecord>)>> {
        kernel_calls(
            self.text.as_ref(),
            &self.image_order,
            &self.encoders,
            &self.denoisers,
            &self.decoders,
        )
        .into_iter()
        .map(|(label, device, calls)| Ok((label, device, list_kernels(calls)?)))
        .collect()
    }

    /// Release what serves an image model: its runners, the host lane and
    /// the output storage, after the lanes drained.
    fn close_image(&mut self, aborted: bool) -> Result<()> {
        let Some(image) = self.image.take() else {
            return Ok(());
        };
        // Each runner releases its graphs and then the execution context
        // its Python half constructed, newest first.
        let mut result = Ok(());
        for (runner, _) in std::mem::take(&mut self.image_order).into_iter().rev() {
            let closed = match runner {
                ImageRunner::Encoder(index) => self
                    .encoders
                    .image
                    .get_mut(index)
                    .map_or(Ok(()), |runner| runner.close(aborted)),
                ImageRunner::Denoiser => self
                    .denoisers
                    .image
                    .as_mut()
                    .map_or(Ok(()), |(_, runner)| runner.close(aborted)),
                ImageRunner::Decoder => self
                    .decoders
                    .image
                    .as_mut()
                    .map_or(Ok(()), |(_, runner)| runner.close(aborted)),
            };
            result = result.and(closed);
        }
        if let Some(host) = self.image_host.take() {
            if aborted {
                host.abort();
            } else if let Some(error) = host.close().into_iter().next() {
                result = result.and(Err(error));
            }
        }
        if let Some(mut outputs) = self.media_outputs.take() {
            outputs.begin_close();
            if !aborted {
                outputs.finish_close();
            }
        }
        // The owners end here, as the model's facts do.
        drop(image);
        self.release_media();
        result
    }

    /// Drop the rank's media runners and the storage their forwards use.
    fn release_media(&mut self) {
        self.encoders = EncoderRunners::default();
        self.denoisers = DenoiserRunners::default();
        self.decoders = DecoderRunners::default();
        self.latents = None;
        self.trajectories = Trajectories::default();
        self.media = None;
    }

    /// Install the diffusion LM runners.
    pub(crate) fn set_diffusion_lm(
        &mut self,
        runners: Vec<crate::runner::diffusion_lm::DiffusionLMRunner>,
    ) {
        self.diffusion_lm = runners;
    }

    pub(crate) fn diffusion_lm_mut(
        &mut self,
    ) -> &mut [crate::runner::diffusion_lm::DiffusionLMRunner] {
        &mut self.diffusion_lm
    }

    /// Install the expert-parallel runtime.
    pub(crate) fn set_experts(&mut self, experts: crate::parallel::ExpertUnion) {
        self.experts = Some(Arc::new(experts));
    }

    /// Install the distributed expert weights.
    pub(crate) fn set_weights(&mut self, weights: crate::parallel::DwdpManager) {
        self.weights = Some(weights);
    }

    pub(crate) fn experts(&self) -> Option<&Arc<crate::parallel::ExpertUnion>> {
        self.experts.as_ref()
    }

    /// The member, to change it between jobs: a job holds it only until
    /// its completion is posted, so holding it now is an error, not a
    /// reason to skip the change.
    pub(crate) fn experts_mut(&mut self) -> Result<Option<&mut crate::parallel::ExpertUnion>> {
        match &mut self.experts {
            Some(experts) => Arc::get_mut(experts)
                .map(Some)
                .ok_or(Error::State("the expert union is held by a job")),
            None => Ok(None),
        }
    }

    /// The lanes' job queues, for work that runs as jobs outside a batch
    /// (warmup captures).
    pub(crate) fn job_queues(&self) -> &JobQueues {
        &self.jobs
    }

    /// Release the expert runtime after the bindings and graphs using its
    /// transports retired; collective unless `aborted`.
    pub(crate) fn close_experts(&mut self, aborted: bool) -> Result<()> {
        let experts = match self.experts_mut() {
            Ok(Some(experts)) => experts.close(aborted),
            Ok(None) => Ok(()),
            Err(error) => Err(error),
        };
        let weights = match &mut self.weights {
            Some(weights) => weights.close(aborted),
            None => Ok(()),
        };
        experts.and(weights)
    }

    /// Install the transfer runtime.
    pub(crate) fn set_transfer(&mut self, transfer: super::transfer::TransferRuntime) {
        self.transfer = Some(transfer);
    }

    /// The stream serving `kind` calls on `device` (none on host devices):
    /// imports for such calls are read and published on it.
    pub(crate) fn transfer_stream(
        &self,
        kind: CallKind,
        device: crate::ffi::MemoryKind,
    ) -> Result<Option<uniserve_cuda::Stream>> {
        if !matches!(device, crate::ffi::MemoryKind::Cuda(_)) {
            return Ok(None);
        }
        self.transfer
            .as_ref()
            .ok_or_else(|| {
                Error::Unsupported("device transfers require the transfer runtime".into())
            })?
            .stream(kind, device)
    }

    /// Disjoint borrows for executing one batch.
    pub(crate) fn parts(&mut self) -> Parts<'_> {
        Parts {
            info: &self.info,
            rank: self.rank,
            requests: &mut self.requests,
            storage: self.rank_storage.as_mut(),
            text: self.text.as_mut(),
            image: match (&self.image, &mut self.media_outputs, &self.image_host) {
                (Some((model, policy)), Some(outputs), Some(host)) => Some(ImageParts {
                    model,
                    policy,
                    encoders: &mut self.encoders,
                    denoisers: &mut self.denoisers,
                    decoders: &mut self.decoders,
                    latents: self.latents.as_mut(),
                    trajectories: &mut self.trajectories,
                    outputs,
                    host,
                }),
                _ => None,
            },
            diffusion_lm: &mut self.diffusion_lm,
            experts: self.experts.as_ref(),
            transfer: self.transfer.as_mut(),
        }
    }

    pub(crate) fn set_profiler(&mut self, profiler: Option<Profiler>) {
        self.profiler = profiler;
    }

    /// Report kernel selections that change while serving.
    pub(crate) fn watch_kernels(&mut self, watch: KernelWatch) {
        self.kernels = Some(watch);
    }

    /// Whether the rank serves a video model: its calls run as media
    /// batches on the rank's video runners.
    pub(crate) fn serves_video(&self) -> bool {
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
    pub(crate) fn media_parts(&mut self) -> Result<super::video::Parts<'_>> {
        let unserved =
            || Error::Unsupported("video calls are served on a video model's rank".into());
        Ok(super::video::Parts {
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
                self.image.is_some(),
                &mut self.trajectories,
                &mut self.latents,
                &mut self.diffusion_lm,
                &mut self.transfer,
                &mut self.video,
                &mut self.denoisers,
                &mut self.media_outputs,
                &mut self.storage,
            ) {
                owner.admit_slots(&admitted, &self.lifecycle)?;
            }
            self.perform_lifecycle()?;
        }
        started?;
        if let Some(video) = &mut self.video {
            for request in installed {
                video.admit(request, self.denoisers.video.is_some())?;
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
        if let Some(cache) = &self.info.prefix_cache
            && plan
                .new_cache_units
                .iter()
                .flat_map(|allocation| &allocation.unit_ids)
                .any(|unit| unit.0 >= cache.num_units)
        {
            return Err(Error::Invalid(
                "cache allocation exceeds the fixed physical pool".into(),
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

    /// Install (or clear) the service's completion wake on every storage
    /// owner.
    pub(crate) fn set_storage_wake(&self, wake: Option<uniserve_ipc::Wake>) {
        self.jobs.set_wake(wake.clone());
        if let Some(storage) = &self.rank_storage {
            storage.set_wake(wake.clone());
        }
        if let Some(host) = &self.image_host {
            host.set_wake(wake.clone());
        }
        if let Some(media) = &self.media {
            media.lane.set_wake(wake.clone());
        }
        for owner in &self.storage {
            owner.set_wake(wake.clone());
        }
    }

    /// Close every storage owner, newest first (other runtimes, a video
    /// model's owners, the transfer runtime, then rank storage), after the
    /// executor drained and [`Self::close_runtimes`] closed the diffusion
    /// LM, image and text runtimes; returns the first failure.
    pub(crate) fn close_storage(&mut self, aborted: bool) -> Result<()> {
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
            self.release_media();
        }
        if let Some(mut transfer) = self.transfer.take() {
            result = result.and(transfer.close(aborted));
        }
        if let Some(mut storage) = self.rank_storage.take() {
            result = result.and(storage.close(aborted));
        }
        result
    }

    /// Close the diffusion LM, image and text runtimes (graphs, then steps)
    /// before the bindings they run close; attempts every one before
    /// returning the first failure.
    pub(crate) fn close_runtimes(&mut self, aborted: bool) -> Result<()> {
        let mut diffusion_lm = Ok(());
        for mut runner in std::mem::take(&mut self.diffusion_lm) {
            diffusion_lm = diffusion_lm.and(runner.close(aborted));
        }
        let image = self.close_image(aborted);
        let text = self
            .text
            .take()
            .map_or(Ok(()), |mut text| text.close(aborted));
        diffusion_lm.and(image).and(text)
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
            if let Some(input) = call.cache_input {
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
                storage.release_buffers(&predicates, &self.lifecycle)?;
            }
        }
        self.perform_lifecycle()
    }

    /// Release request references after the executor drained its work.
    pub(crate) fn close(&mut self) {
        self.requests.close();
        if let Some(profiler) = &mut self.profiler {
            profiler.close();
        }
    }
}

/// The kernel listings of the text runners and of an image model's
/// runners, in the order `order` names those: each with its runner's
/// label and device.
fn kernel_calls(
    text: Option<&crate::runner::causal_lm::CausalLMRunners>,
    order: &[(ImageRunner, String)],
    encoders: &EncoderRunners,
    denoisers: &DenoiserRunners,
    decoders: &DecoderRunners,
) -> Vec<KernelCalls> {
    let mut listed = text.map_or_else(Vec::new, |text| text.kernel_calls());
    for (runner, name) in order {
        let (device, calls) = match runner {
            ImageRunner::Encoder(index) => match encoders.image.get(*index) {
                Some(runner) => (runner.device().to_string(), runner.kernel_calls()),
                None => continue,
            },
            ImageRunner::Denoiser => match &denoisers.image {
                Some((_, runner)) => (runner.device().to_string(), runner.kernel_calls()),
                None => continue,
            },
            ImageRunner::Decoder => match &decoders.image {
                Some((_, runner)) => (runner.device().to_string(), runner.kernel_calls()),
                None => continue,
            },
        };
        listed.push((name.clone(), device, calls));
    }
    listed
}

/// The batch work of the executor: what each step of a batch's life does
/// on this rank ([`super::Executor`] orders the steps).
impl WorkerBackend {
    pub(crate) fn admit(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(), WorkerError> {
        // Explicit revocation must not wait behind an allocation-dependent
        // batch; existing readers retain their leases.
        let mut owners = owners(
            &mut self.rank_storage,
            &mut self.text,
            self.image.is_some(),
            &mut self.trajectories,
            &mut self.latents,
            &mut self.diffusion_lm,
            &mut self.transfer,
            &mut self.video,
            &mut self.denoisers,
            &mut self.media_outputs,
            &mut self.storage,
        );
        batch.retirement.admit(&mut owners, &self.lifecycle)?;
        drop(owners);
        Ok(self.perform_lifecycle()?)
    }

    pub(crate) fn prepare(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(), WorkerError> {
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

    pub(crate) fn admit_inputs(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<Option<JobId>, WorkerError> {
        // A video request this batch started evaluates in a layout the
        // library selects for its size. Asking is a job of the Python
        // thread (it issues nothing on a stream); the batch goes on from
        // its completion, and so do the request's later calls, which start
        // behind this batch.
        if let Some(video) = &mut self.video {
            if batch.sizing.is_none() {
                batch.sizing = video.begin_sizing();
            }
            if let Some(sizing) = &mut batch.sizing {
                let job = Job::new();
                let on_lane = match video.ask(sizing, &mut self.denoisers, &job) {
                    Ok(on_lane) => on_lane,
                    Err(error) => {
                        if let Some(sizing) = batch.sizing.take() {
                            video.abandon(sizing);
                        }
                        return Err(error.into());
                    }
                };
                // A layout's preparation works on the denoiser's stream,
                // so it is a job of that lane; the model's arithmetic is a
                // job of the Python thread alone.
                let submitted = if on_lane {
                    let kind = CallKind::Media(uniserve_ipc::MediaCall::LatentPreparation);
                    self.jobs.start(kind, job, true).map_err(lane_lost)?
                } else {
                    self.jobs.call(job, true)
                };
                return Ok(Some(submitted));
            }
        }

        // Products other calls published are imported for every runtime
        // alike. What the imports recorded for the lane is performed even
        // when an import failed after it: the fences are due either way.
        let (job, admitted) = super::inputs::admit(self, batch);
        let submitted = match batch.route() {
            Some(kind) if !job.is_empty() => {
                Some(self.start(Some(kind), job, true).map_err(lane_lost)?)
            }
            _ => None,
        };
        admitted.map(|()| submitted)
    }

    pub(crate) fn inputs_admitted(
        &mut self,
        batch: &mut WorkerBatch,
        completion: Completion,
    ) -> std::result::Result<(), WorkerError> {
        let Some(mut sizing) = batch.sizing.take() else {
            return super::inputs::admitted(self, batch, completion);
        };

        // The job asked the library to size the requests the batch started.
        let Some(video) = &mut self.video else {
            return Err(Error::State("a batch sized requests on a rank without video").into());
        };
        let answered = completion
            .check()
            .map_err(|failure| crate::runner::denoiser::invalid_size(failure.error))
            .and_then(|mut completion| {
                video.answer(&mut sizing, &mut completion, &mut self.denoisers)
            });
        match answered {
            Ok(true) => Ok(()),
            Ok(false) => {
                batch.sizing = Some(sizing);
                Ok(())
            }
            Err(error) => {
                video.abandon(sizing);
                Err(error.into())
            }
        }
    }

    pub(crate) fn prepare_inputs(
        &mut self,
        batch: &mut WorkerBatch,
        submission: &Arc<Submission>,
    ) -> std::result::Result<bool, WorkerError> {
        // The cache transfers other calls published are imported behind the
        // admitted products (`admit_inputs`), and the batch waits for both
        // (`await_inputs` subscribes to what blocks them).
        if !super::inputs::prepare(self, batch)? {
            return Ok(false);
        }
        // Inline images of an image encoding batch are decoded on the image
        // runtime's host lane while other batches execute; the batch
        // launches once they are ready (the lane notifies `submission`).
        // Other batches' inputs are prepared when their runtime executes.
        if self.image.is_some() && !self.serves_video() {
            return crate::runner::encoder::inputs::prepare_inputs(self, batch, submission);
        }
        Ok(true)
    }

    pub(crate) fn await_inputs(
        &mut self,
        batch: &mut WorkerBatch,
        submission: &Arc<Submission>,
    ) -> std::result::Result<(), WorkerError> {
        Ok(super::inputs::subscribe(self, batch, submission)?)
    }

    pub(crate) fn lane(&self, batch: &WorkerBatch) -> Option<usize> {
        batch.route().and_then(|kind| self.jobs.lane(kind))
    }

    pub(crate) fn forward(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<Option<JobId>, WorkerError> {
        let Some(kind) = batch.route() else {
            if batch.output.is_none() {
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
            }
            return Ok(None);
        };

        let lanes = self.jobs.clone();
        loop {
            let stage = batch.stage();
            match stage {
                // Imported values become resident before the runtime
                // consumes them like local ones; an import that failed
                // fails the batch's calls.
                batch::Stage::Inputs => {}
                // Each executed batch with calls is one profiler step.
                batch::Stage::Forward if !batch.stepping => {
                    batch.stepping = true;
                    if let Some(profiler) = &mut self.profiler {
                        profiler.begin();
                    }
                }
                batch::Stage::Forward => {}
                batch::Stage::Done => self.end_forward(batch),
            }

            // The profiler's window opens before the first job of its
            // first step's forward and closes behind the last job of its
            // last step's: each of its jobs is one the batch waits for.
            if stage != batch::Stage::Inputs
                && let Some(profiler) = &mut self.profiler
                && let Some(job) = profiler.next(&lanes, kind)
            {
                batch.profiling = Some(job);
                return Ok(Some(job));
            }
            if stage == batch::Stage::Done {
                self.release_predecessors(batch, true)?;
                return Ok(None);
            }

            // A stage that needs nothing of the lane advances without a
            // job.
            let submitted = match batch::forward(self, batch, kind) {
                Ok(Some(job)) => {
                    // One forward is in flight at a time (the executor's
                    // ordering): lifecycle work without a stream follows
                    // it to its lane.
                    if lanes.lane(kind).is_some() {
                        self.lifecycle.follow(kind);
                    }
                    Some(self.start(Some(kind), job, true).map_err(lane_lost))
                }
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            };
            match submitted {
                Some(Ok(job)) => return Ok(Some(job)),
                Some(Err(error)) => {
                    self.end_forward(batch);
                    return Err(error);
                }
                None => {}
            }
        }
    }

    pub(crate) fn complete(
        &mut self,
        batch: &mut WorkerBatch,
        job: JobId,
        completion: Completion,
    ) -> std::result::Result<(), WorkerError> {
        let Some(kind) = batch.route() else {
            return Ok(());
        };
        if batch.profiling == Some(job) {
            batch.profiling = None;
            if let Some(profiler) = &mut self.profiler {
                profiler.complete(completion);
            }
            return Ok(());
        }
        let completed = batch::complete(self, batch, kind, completion);
        if completed.is_err() {
            self.end_forward(batch);
        }
        completed
    }

    pub(crate) fn await_outputs(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<Option<JobId>, WorkerError> {
        let (Some(kind), Some(running)) = (batch.route(), &batch.running) else {
            return Ok(None);
        };
        // The fences are named by the lease, which holds them whatever the
        // event pool has released since they were recorded.
        let fences: Vec<_> = crate::lock(&running.buffer)
            .device_fences()
            .map(|(device, fence)| (device, Arc::clone(fence)))
            .collect();
        let Some(&(device, _)) = fences.first() else {
            return Ok(None);
        };
        let fences = fences.into_iter().map(|(_, fence)| fence).collect();

        // The job holds no operation: its completion is posted once the
        // lease's fences completed.
        let job = Job::new();
        job.after(device, fences);
        self.jobs
            .start(kind, job, true)
            .map(Some)
            .map_err(lane_lost)
    }

    pub(crate) fn outputs_ready(
        &mut self,
        batch: &mut WorkerBatch,
        completion: Completion,
    ) -> std::result::Result<(), WorkerError> {
        completion
            .check()
            .map_err(|failure| lane_lost(failure.error))?;
        if let Some(running) = &batch.running {
            crate::lock(&running.buffer).fences_completed();
        }
        Ok(())
    }

    pub(crate) fn completed(
        &mut self,
    ) -> std::result::Result<Option<(JobId, Completion)>, WorkerError> {
        while let Some((job, completion)) = self.jobs.take() {
            if let Some(storage) = &self.rank_storage {
                storage.events.performed(job);
            }
            // What belongs to no batch is applied here: the event pool's
            // deferred work, a lifecycle command's device work, the expert
            // union's step this rank joined, the kernel watch's calls.
            let events = self.rank_storage.as_ref().map(|storage| &storage.events);
            if let Some(events) = events.filter(|events| events.owns(job)) {
                // The device finished what the job's events fence: the
                // storage deferred behind them is released before the
                // executor routes the completions that follow.
                events.complete(job, completion)?;
            } else if self.lifecycle_jobs.remove(&job) {
                // The work a command recorded did not reach its storage:
                // what that storage holds now is unknown.
                completion.check().map_err(|failure| {
                    let mut error = WorkerError::classify(failure.error, Some("lifecycle"));
                    error.fatal = true;
                    error
                })?;
            } else if self.joining == Some(job) {
                self.joining = None;
                let joined = completion
                    .check()
                    .map_err(|failure| failure.error)
                    .and_then(|mut completion| completion.joined())
                    .map_err(|error| WorkerError::classify(error, Some("expert step")));
                self.joined = Some(joined);
            } else if let Some(watch) = self.kernels.as_mut().filter(|watch| watch.owns(job)) {
                watch.complete(job, completion, &self.jobs, || {
                    kernel_calls(
                        self.text.as_ref(),
                        &self.image_order,
                        &self.encoders,
                        &self.denoisers,
                        &self.decoders,
                    )
                });
            } else {
                return Ok(Some((job, completion)));
            }
        }
        Ok(None)
    }

    pub(crate) fn begin_retirement(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(), WorkerError> {
        let mut owners = owners(
            &mut self.rank_storage,
            &mut self.text,
            self.image.is_some(),
            &mut self.trajectories,
            &mut self.latents,
            &mut self.diffusion_lm,
            &mut self.transfer,
            &mut self.video,
            &mut self.denoisers,
            &mut self.media_outputs,
            &mut self.storage,
        );
        batch
            .retirement
            .begin(&self.requests, &mut owners, &self.lifecycle)?;
        drop(owners);
        Ok(self.perform_lifecycle()?)
    }

    pub(crate) fn poll(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(bool, bool), WorkerError> {
        let mut advanced = false;
        if batch.output.is_none()
            && let Some(running) = batch.running.as_mut()
            && let Some(result) = batch::poll(self, &batch.plan, running)?
        {
            batch.output = Some(result);
            batch.running = None;
            advanced = true;
        }
        if batch.output.is_none() {
            return Ok((advanced, false));
        }
        let mut owners = owners(
            &mut self.rank_storage,
            &mut self.text,
            self.image.is_some(),
            &mut self.trajectories,
            &mut self.latents,
            &mut self.diffusion_lm,
            &mut self.transfer,
            &mut self.video,
            &mut self.denoisers,
            &mut self.media_outputs,
            &mut self.storage,
        );
        let retired = batch
            .retirement
            .poll(&mut self.requests, &mut owners, &self.lifecycle)?;
        drop(owners);
        self.perform_lifecycle()?;
        Ok((advanced || retired, retired))
    }

    pub(crate) fn result(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<BatchResult, WorkerError> {
        let result = batch
            .output
            .as_mut()
            .ok_or_else(|| WorkerError::from(Error::State("batch output is not resolved")))?;
        Ok(BatchResult {
            output: result.output.clone(),
            media: std::mem::take(&mut result.media),
        })
    }

    pub(crate) fn close_batch(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(), WorkerError> {
        // Calls become pending only when a capability runtime commits them;
        // unresolved launched work cancels its pending calls. Lifecycle
        // commands hold no resource beyond their retirement.
        let abandoned = batch
            .running
            .take()
            .map_or(Ok(()), |running| batch::abandon(self, running));
        // Imports the batch did not consume are abandoned; their storage
        // retires with their physical accesses.
        let closed = super::inputs::close(self, batch).map_err(WorkerError::from);
        abandoned.and(closed)
    }

    pub(crate) fn reap(&mut self) -> std::result::Result<(), WorkerError> {
        for owner in owners(
            &mut self.rank_storage,
            &mut self.text,
            self.image.is_some(),
            &mut self.trajectories,
            &mut self.latents,
            &mut self.diffusion_lm,
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

/// What the rank service asks of the backend beyond the executor's steps.
impl WorkerBackend {
    pub(crate) fn response_error(
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

    pub(crate) fn has_open_requests(&self) -> std::result::Result<bool, WorkerError> {
        Ok(self.requests.has_open_requests()?)
    }

    pub(crate) fn awaiting_acknowledgment(&self) -> std::result::Result<bool, WorkerError> {
        Ok(self
            .rank_storage
            .as_ref()
            .is_some_and(|storage| storage.awaiting_acknowledgment())
            || self
                .storage
                .iter()
                .any(|owner| owner.awaiting_acknowledgment()))
    }

    pub(crate) fn join_expert_step(
        &mut self,
        leaving: bool,
    ) -> std::result::Result<Option<(bool, bool)>, WorkerError> {
        if let Some(joined) = self.joined.take() {
            return joined.map(Some);
        }
        if self.joining.is_some() {
            return Ok(None);
        }
        let experts = self.experts().ok_or_else(|| {
            WorkerError::classify(
                Error::State("expert-step participation requires an expert union"),
                None,
            )
        })?;

        // A join launches on the lane the union's forwards run on, whose
        // execution thread takes part in the agreement.
        let job = Job::new();
        job.expert_join(Arc::clone(experts), leaving);
        let submitted = self
            .jobs
            .start(CallKind::Forward(ForwardMode::Decode), job, true)
            .map_err(lane_lost)?;
        self.joining = Some(submitted);
        Ok(None)
    }
}

/// Start the job that completes once `events` completed where `kind` is
/// served (the Python thread for a rank without a lane, which has no device
/// event to wait for), with the control thread's wake behind it.
fn after(
    jobs: &JobQueues,
    kind: Option<CallKind>,
    device: uniserve_cuda::DeviceId,
    events: Vec<Arc<uniserve_cuda::Event>>,
) -> Result<JobId> {
    match kind {
        Some(kind) => jobs.after(kind, device, events),
        None => {
            let job = Job::new();
            job.after(device, events);
            Ok(jobs.call(job, true))
        }
    }
}

/// A lane that no longer performs jobs leaves the rank unable to serve.
fn lane_lost(error: Error) -> WorkerError {
    let mut error = WorkerError::classify(error, Some("lane"));
    error.fatal = true;
    error
}
