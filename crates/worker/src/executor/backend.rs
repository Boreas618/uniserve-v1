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

mod steps;

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use ipc::{Batch as BatchPlan, BatchCommand, RequestKind, Stall, WorkerInfo, WorkerResponseError};

use super::image::ImageParts;
use super::lifecycle::{AdmittedSlot, LaneJobs, RequestStorage};
use crate::config::placement::{Components, GroupPlacement};
use crate::error::WorkerError;
use crate::host::HostLane;
use crate::host::image::ImageTask;
use crate::lane::{Job, JobId, JobQueues};
use crate::memory::LatentCache;
use crate::profiling::kernels::{KernelCalls, KernelRecord, KernelWatch};
use crate::profiling::profiler::Profiler;
use crate::request::RequestPool;
use crate::request::diffusion::Trajectories;
use crate::runner::decoder::runners::DecoderRunners;
use crate::runner::denoiser::image_model::{ImageModel, ImagePolicy};
use crate::runner::denoiser::runners::DenoiserRunners;
use crate::runner::encoder::runners::EncoderRunners;
use crate::runner::methods::list_kernels;
use crate::runner::start::{ImageOwners, ImageRunner};
use crate::worker::WorkerMemory;
use crate::{Error, OutputPool, Result};

pub(crate) use steps::WorkerBatch;
use steps::lane_lost;

/// The backend of one rank.
pub(crate) struct WorkerBackend {
    info: WorkerInfo,
    rank: usize,
    /// The placement the rank's group was launched with.
    placement: GroupPlacement,
    requests: RequestPool,
    /// What serves the rank's components, beside the request pool and the
    /// lifecycle jobs every lifecycle command borrows with its owners.
    runtimes: Runtimes,
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

/// The runtimes serving a rank's components and the storage they own: the
/// rank's storage owners, each capability's runners and an image or video
/// model's state.
#[derive(Default)]
pub(crate) struct Runtimes {
    /// The rank's storage owners (events, products, transports, prefix cache).
    worker_storage: Option<WorkerMemory>,
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
    /// Layer stacks streamed from host memory, on ranks of components that
    /// keep their weights there.
    host_weights: Vec<std::sync::Arc<crate::parallel::HostWeights>>,
    /// The transfer runtime: tensor transfers, cache exports and installs, and
    /// the streams batch imports run on.
    transfer: Option<super::transfer::TransferRuntime>,
    /// What serves a video model on its ranks: the executor and the
    /// trajectory ledger.
    video: Option<super::video::VideoRuntime>,
    /// An image model's encoder, denoiser and decoder runners.
    encoders: EncoderRunners,
    denoisers: DenoiserRunners,
    decoders: DecoderRunners,
    /// The output leases of media batches.
    media_outputs: Option<OutputPool>,
    /// Other capability runtimes' request storage.
    storage: Vec<Box<dyn RequestStorage>>,
}

impl Runtimes {
    /// Every request storage owner, in lifecycle order: rank storage, the
    /// text runners, an image model's trajectories, output leases and
    /// latent pool, the diffusion LM runners, the transfer runtime, a video
    /// model's runtime and output leases, then the other runtimes' owners.
    pub(crate) fn owners(&mut self) -> impl Iterator<Item = &mut dyn RequestStorage> {
        fn owner<T: RequestStorage>(owner: &mut T) -> &mut dyn RequestStorage {
            owner
        }

        // The output leases are an image model's where the rank holds one,
        // otherwise a video model's.
        let image = self.image.is_some();
        let video = self.video.is_some();
        let (image_outputs, video_outputs) = match self.media_outputs.as_mut() {
            Some(outputs) if image => (Some(outputs), None),
            Some(outputs) if video => (None, Some(outputs)),
            _ => (None, None),
        };
        let (trajectories, latents) = if image {
            (Some(&mut self.trajectories), self.latents.as_mut())
        } else {
            (None, None)
        };

        self.worker_storage
            .as_mut()
            .map(owner)
            .into_iter()
            .chain(self.text.as_mut().map(owner))
            .chain(trajectories.map(owner))
            .chain(image_outputs.map(owner))
            .chain(latents.map(owner))
            .chain(self.diffusion_lm.iter_mut().map(owner))
            .chain(self.transfer.as_mut().map(owner))
            .chain(self.video.as_mut().map(owner))
            .chain(video_outputs.map(owner))
            .chain(
                self.storage
                    .iter_mut()
                    .map(|owner| &mut **owner as &mut dyn RequestStorage),
            )
    }
}

/// Disjoint borrows of a backend for one batch's execution.
pub(crate) struct Parts<'a> {
    pub placement: &'a GroupPlacement,
    pub rank: usize,
    pub requests: &'a mut RequestPool,
    pub storage: Option<&'a mut WorkerMemory>,
    pub text: Option<&'a mut crate::runner::causal_lm::CausalLMRunners>,
    pub image: Option<ImageParts<'a>>,
    pub diffusion_lm: &'a mut [crate::runner::diffusion_lm::DiffusionLMRunner],
    pub experts: Option<&'a Arc<crate::parallel::ExpertUnion>>,
    pub transfer: Option<&'a mut super::transfer::TransferRuntime>,
}

impl WorkerBackend {
    pub(crate) fn new(
        info: WorkerInfo,
        rank: usize,
        placement: GroupPlacement,
        components: Components,
    ) -> Result<Self> {
        Ok(Self {
            requests: RequestPool::new(info.request_slots as usize)?,
            info,
            rank,
            placement,
            runtimes: Runtimes::default(),
            components,
            profiler: None,
            kernels: None,
            jobs: JobQueues::default(),
            lifecycle: LaneJobs::default(),
            lifecycle_jobs: HashSet::new(),
            joining: None,
            joined: None,
        })
    }

    /// Submit the job the last lifecycle command recorded to the lane
    /// without waiting for it: the lane performs it in the order of its
    /// jobs, so a forward submitted later finds its work done. Nothing goes
    /// on from its completion, which is taken with the next one that
    /// arrives ([`WorkerBackend::completed`]); a failed one stops the rank
    /// there.
    fn perform_lifecycle(&mut self) -> Result<()> {
        if let Some(job) = self.lifecycle.take() {
            let submitted = self.start(job, false)?;
            self.lifecycle_jobs.insert(submitted);
        }
        Ok(())
    }

    /// Submit `job` to the lane (the builder thread for a rank without
    /// one). `wake` is as for [`JobQueues::start`].
    ///
    /// The deferred work the job carries waits for events the job records.
    /// The job whose completion says they completed is started right
    /// behind it, on the lane, and the event pool runs the work when that
    /// completion arrives ([`WorkerBackend::completed`]). Started earlier,
    /// that job would reach the lane before the records and pass at once.
    fn start(&self, job: Job, wake: bool) -> Result<JobId> {
        let deferrals = job.deferrals();
        let submitted = self.jobs.start(job, wake)?;
        if !deferrals.is_empty() {
            let storage = self.runtimes.worker_storage.as_ref().ok_or(Error::State(
                "a job defers work on a rank without an event pool",
            ))?;
            let jobs = &self.jobs;
            storage
                .events
                .defer_submitted(deferrals, |device, events| jobs.after(device, events))?;
        }
        Ok(submitted)
    }

    /// Have the rank's event pool start the jobs its deferred work waits
    /// for on the lane, from this thread, the control thread.
    fn serve_events(&self) {
        let Some(storage) = &self.runtimes.worker_storage else {
            return;
        };
        let jobs = self.jobs.clone();
        storage
            .events
            .set_submitter(Some(Box::new(move |device, events| {
                jobs.after(device, events)
            })));
    }

    /// Use the job queues of the rank's lane for forward submission.
    pub(crate) fn set_job_queues(&mut self, jobs: JobQueues) {
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
    pub(crate) fn set_worker_storage(&mut self, storage: WorkerMemory) {
        self.runtimes.worker_storage = Some(storage);
        self.serve_events();
    }

    pub(crate) fn worker_storage(&self) -> Option<&WorkerMemory> {
        self.runtimes.worker_storage.as_ref()
    }

    pub(crate) fn worker_storage_mut(&mut self) -> Option<&mut WorkerMemory> {
        self.runtimes.worker_storage.as_mut()
    }

    /// Install the text runtime.
    pub(crate) fn set_text(&mut self, text: crate::runner::causal_lm::CausalLMRunners) {
        self.runtimes.text = Some(text);
    }

    pub(crate) fn text_mut(&mut self) -> Option<&mut crate::runner::causal_lm::CausalLMRunners> {
        self.runtimes.text.as_mut()
    }

    /// Install what serves an image model on this rank.
    pub(crate) fn set_image(&mut self, owners: ImageOwners) {
        self.runtimes.image = Some((owners.model, owners.policy));
        self.runtimes.encoders = owners.encoders;
        self.runtimes.denoisers = owners.denoisers;
        self.runtimes.decoders = owners.decoders;
        self.runtimes.latents = owners.latents;
        self.runtimes.media_outputs = Some(owners.outputs);
        self.runtimes.image_host = Some(owners.host);
        self.runtimes.image_order = owners.order;
    }

    /// The kernels each text and image runner's operators selected, as
    /// `(label, device, kernels)`: text runners first, then the image
    /// runners in construction order. Each listing is a call made now,
    /// which startup waits for.
    pub(crate) fn kernels(&self) -> Result<Vec<(String, String, Vec<KernelRecord>)>> {
        kernel_calls(
            self.runtimes.text.as_ref(),
            &self.runtimes.image_order,
            &self.runtimes.encoders,
            &self.runtimes.denoisers,
            &self.runtimes.decoders,
        )
        .into_iter()
        .map(|(label, device, calls)| Ok((label, device, list_kernels(calls)?)))
        .collect()
    }

    /// Release what serves an image model: its runners, the host lane and
    /// the output storage, after the lanes drained.
    fn close_image(&mut self, aborted: bool) -> Result<()> {
        let Some(image) = self.runtimes.image.take() else {
            return Ok(());
        };
        // Each runner releases its graphs and then the execution context
        // its Python half constructed, newest first.
        let mut result = Ok(());
        for (runner, _) in std::mem::take(&mut self.runtimes.image_order)
            .into_iter()
            .rev()
        {
            let closed = match runner {
                ImageRunner::Encoder(index) => self
                    .runtimes
                    .encoders
                    .image
                    .get_mut(index)
                    .map_or(Ok(()), |runner| runner.close(aborted)),
                ImageRunner::Denoiser => self
                    .runtimes
                    .denoisers
                    .image
                    .as_mut()
                    .map_or(Ok(()), |(_, runner)| runner.close(aborted)),
                ImageRunner::Decoder => self
                    .runtimes
                    .decoders
                    .image
                    .as_mut()
                    .map_or(Ok(()), |(_, runner)| runner.close(aborted)),
            };
            result = result.and(closed);
        }
        if let Some(host) = self.runtimes.image_host.take() {
            if aborted {
                host.abort();
            } else if let Some(error) = host.close().into_iter().next() {
                result = result.and(Err(error));
            }
        }
        if let Some(mut outputs) = self.runtimes.media_outputs.take() {
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
        self.runtimes.encoders = EncoderRunners::default();
        self.runtimes.denoisers = DenoiserRunners::default();
        self.runtimes.decoders = DecoderRunners::default();
        self.runtimes.latents = None;
        self.runtimes.trajectories = Trajectories::default();
    }

    /// Install the diffusion LM runners.
    pub(crate) fn set_diffusion_lm(
        &mut self,
        runners: Vec<crate::runner::diffusion_lm::DiffusionLMRunner>,
    ) {
        self.runtimes.diffusion_lm = runners;
    }

    pub(crate) fn diffusion_lm_mut(
        &mut self,
    ) -> &mut [crate::runner::diffusion_lm::DiffusionLMRunner] {
        &mut self.runtimes.diffusion_lm
    }

    /// Install the expert-parallel runtime.
    pub(crate) fn set_experts(&mut self, experts: crate::parallel::ExpertUnion) {
        self.runtimes.experts = Some(Arc::new(experts));
    }

    /// Install the distributed expert weights.
    pub(crate) fn set_weights(&mut self, weights: crate::parallel::DwdpManager) {
        self.runtimes.weights = Some(weights);
    }

    /// Install the layer stacks streamed from host memory.
    pub(crate) fn set_host_weights(
        &mut self,
        weights: Vec<std::sync::Arc<crate::parallel::HostWeights>>,
    ) {
        self.runtimes.host_weights = weights;
    }

    /// Release the graphs of every runtime that runs expert steps and accept
    /// their captures again, once the expert transports were rebuilt at new
    /// addresses. The lanes are idle.
    pub(crate) fn reopen_expert_graphs(&mut self) -> Result<()> {
        if let Some(text) = self.runtimes.text.as_mut() {
            text.reopen()?;
        }
        for runner in &mut self.runtimes.diffusion_lm {
            runner.reopen()?;
        }
        Ok(())
    }

    /// Refuse further captures of the runtimes that run expert steps, once
    /// they were captured again.
    pub(crate) fn seal_expert_graphs(&mut self) {
        if let Some(text) = self.runtimes.text.as_mut() {
            text.seal();
        }
        for runner in &mut self.runtimes.diffusion_lm {
            runner.seal();
        }
    }

    /// The NCCL communicators of every runner's collectives, which a parked
    /// standby suspends.
    pub(crate) fn collectives(&self) -> impl Iterator<Item = &crate::parallel::StreamCollectives> {
        let text = self
            .runtimes
            .text
            .iter()
            .flat_map(|text| text.collectives());
        let diffusion = self
            .runtimes
            .diffusion_lm
            .iter()
            .flat_map(|runner| runner.collectives());
        text.chain(diffusion)
    }

    pub(crate) fn experts(&self) -> Option<&Arc<crate::parallel::ExpertUnion>> {
        self.runtimes.experts.as_ref()
    }

    /// The member, to change it between jobs: a job holds it only until
    /// its completion is posted, so holding it now is an error, not a
    /// reason to skip the change.
    pub(crate) fn experts_mut(&mut self) -> Result<Option<&mut crate::parallel::ExpertUnion>> {
        match &mut self.runtimes.experts {
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

    /// Release the expert runtime and the weights copied ahead of their
    /// layers after the bindings and graphs using them retired; collective
    /// unless `aborted`.
    pub(crate) fn close_experts(&mut self, aborted: bool) -> Result<()> {
        let experts = match self.experts_mut() {
            Ok(Some(experts)) => experts.close(aborted),
            Ok(None) => Ok(()),
            Err(error) => Err(error),
        };
        let weights = match &mut self.runtimes.weights {
            Some(weights) => weights.close(aborted),
            None => Ok(()),
        };
        // Host weights' copies run on their own streams; an aborted close
        // keeps their memory until exit, since copies may still be queued.
        let mut host = Ok(());
        for streamed in std::mem::take(&mut self.runtimes.host_weights) {
            if aborted {
                crate::leak(streamed);
            } else {
                host = host.and(streamed.close());
            }
        }
        experts.and(weights).and(host)
    }

    /// Install the transfer runtime.
    pub(crate) fn set_transfer(&mut self, transfer: super::transfer::TransferRuntime) {
        self.runtimes.transfer = Some(transfer);
    }

    /// The lane's stream on `device` (none on host devices): imports are
    /// read and published on it.
    pub(crate) fn transfer_stream(
        &self,
        device: crate::ffi::MemoryKind,
    ) -> Result<Option<cuda::Stream>> {
        if !matches!(device, crate::ffi::MemoryKind::Cuda(_)) {
            return Ok(None);
        }
        self.runtimes
            .transfer
            .as_ref()
            .ok_or_else(|| {
                Error::Unsupported("device transfers require the transfer runtime".into())
            })?
            .stream(device)
    }

    /// Disjoint borrows for executing one batch.
    pub(crate) fn parts(&mut self) -> Parts<'_> {
        Parts {
            placement: &self.placement,
            rank: self.rank,
            requests: &mut self.requests,
            storage: self.runtimes.worker_storage.as_mut(),
            text: self.runtimes.text.as_mut(),
            image: match (
                &self.runtimes.image,
                &mut self.runtimes.media_outputs,
                &self.runtimes.image_host,
            ) {
                (Some((model, policy)), Some(outputs), Some(host)) => Some(ImageParts {
                    model,
                    policy,
                    encoders: &mut self.runtimes.encoders,
                    denoisers: &mut self.runtimes.denoisers,
                    decoders: &mut self.runtimes.decoders,
                    latents: self.runtimes.latents.as_mut(),
                    trajectories: &mut self.runtimes.trajectories,
                    outputs,
                    host,
                }),
                _ => None,
            },
            diffusion_lm: &mut self.runtimes.diffusion_lm,
            experts: self.runtimes.experts.as_ref(),
            transfer: self.runtimes.transfer.as_mut(),
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
    /// batches on the executor.
    pub(crate) fn serves_video(&self) -> bool {
        self.runtimes.video.is_some()
    }

    /// Install what serves a video model on this rank: the executor with
    /// the trajectory ledger, and the output leases of its batches.
    pub(crate) fn set_video(&mut self, video: super::video::VideoRuntime, outputs: OutputPool) {
        self.runtimes.video = Some(video);
        self.runtimes.media_outputs = Some(outputs);
    }

    /// What serves a video model on this rank.
    pub(crate) fn video(&self) -> Option<&super::video::VideoRuntime> {
        self.runtimes.video.as_ref()
    }

    /// The parts a video batch executes with.
    pub(crate) fn media_parts(&mut self) -> Result<super::video::Parts<'_>> {
        let unserved =
            || Error::Unsupported("video calls are served on a video model's rank".into());
        Ok(super::video::Parts {
            requests: &mut self.requests,
            storage: self
                .runtimes
                .worker_storage
                .as_ref()
                .ok_or_else(|| Error::Invariant("the rank has no storage owners".into()))?,
            info: &self.info,
            placement: &self.placement,
            rank: self.rank,
            video: self.runtimes.video.as_mut().ok_or_else(unserved)?,
            outputs: self.runtimes.media_outputs.as_mut().ok_or_else(unserved)?,
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
            let slots: Vec<AdmittedSlot<'_>> = admitted
                .iter()
                .zip(&installed)
                .map(|(&slot, request)| AdmittedSlot {
                    slot,
                    request: request.as_ref(),
                })
                .collect();
            for owner in self.runtimes.owners() {
                owner.admit_slots(&slots, &self.lifecycle)?;
            }
            self.perform_lifecycle()?;
        }
        started?;
        // The executor admits a video model's started requests in a job
        // before the batch opens (`admit_inputs`); one whose size it does not
        // serve is refused alone, and its calls report the refusal.
        if let Some(video) = &mut self.runtimes.video {
            for request in installed {
                video.start(request)?;
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

    /// Refuse a batch outside this rank's advertised capacity, with the
    /// checks no other owner makes. Buffer allocations, block tables and new
    /// cache units are refused by their owners (`BufferPool::bind`,
    /// `BlockTables::prepare`, `UnitColumn::recycle`).
    fn validate(&self, plan: &BatchPlan) -> Result<()> {
        let unsupported: BTreeSet<&str> = plan
            .calls
            .iter()
            .filter(|call| !self.info.supported_calls.contains(&call.kind))
            .map(|call| call.kind.as_str())
            .collect();
        if !unsupported.is_empty() {
            return Err(Error::Invalid(format!(
                "execution batch contains call kinds unsupported by this worker: {:?}",
                unsupported.into_iter().collect::<Vec<_>>()
            )));
        }
        if !self.placement.components.is_empty() {
            for call in &plan.calls {
                if !self.placement.holds(self.rank, &call.component) {
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
            .forward
            .request_pool_indices
            .iter()
            .any(|&slot| slot > self.info.request_slots)
        {
            return Err(Error::Invalid(
                "execution batch exceeds request-slot capacity".into(),
            ));
        }
        Ok(())
    }

    /// Install (or clear) the service's completion wake on every storage
    /// owner.
    pub(crate) fn set_storage_wake(&self, wake: Option<ipc::Wake>) {
        self.jobs.set_wake(wake.clone());
        if let Some(storage) = &self.runtimes.worker_storage {
            storage.set_wake(wake.clone());
        }
        if let Some(host) = &self.runtimes.image_host {
            host.set_wake(wake.clone());
        }
        if let Some(video) = &self.runtimes.video {
            video.set_wake(wake.clone());
        }
        for owner in &self.runtimes.storage {
            owner.set_wake(wake.clone());
        }
    }

    /// Close every storage owner, newest first (other runtimes, a video
    /// model's owners, the transfer runtime, then rank storage), after the
    /// executor drained and [`Self::close_runtimes`] closed the diffusion
    /// LM, image and text runtimes; returns the first failure.
    pub(crate) fn close_storage(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        while let Some(mut owner) = self.runtimes.storage.pop() {
            result = result.and(owner.close(aborted));
        }
        // A video model's owners close before the rank storage their
        // products live in: the executor, then the output leases.
        if let Some(mut video) = self.runtimes.video.take() {
            result = result.and(video.close(aborted));
            if let Some(mut outputs) = self.runtimes.media_outputs.take() {
                result = result.and(outputs.close(aborted));
            }
            self.release_media();
        }
        if let Some(mut transfer) = self.runtimes.transfer.take() {
            result = result.and(transfer.close(aborted));
        }
        if let Some(mut storage) = self.runtimes.worker_storage.take() {
            result = result.and(storage.close(aborted));
        }
        result
    }

    /// Close the diffusion LM, image and text runtimes (graphs, then steps)
    /// before the bindings they run close; attempts every one before
    /// returning the first failure.
    pub(crate) fn close_runtimes(&mut self, aborted: bool) -> Result<()> {
        let mut diffusion_lm = Ok(());
        for mut runner in std::mem::take(&mut self.runtimes.diffusion_lm) {
            diffusion_lm = diffusion_lm.and(runner.close(aborted));
        }
        let image = self.close_image(aborted);
        let text = self
            .runtimes
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
        let Some(storage) = self.runtimes.worker_storage.as_mut() else {
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
        if crate::error::traces(error.code) {
            tracing::error!(
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
            .runtimes
            .worker_storage
            .as_ref()
            .is_some_and(|storage| storage.awaiting_acknowledgment())
            || self
                .runtimes
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
        let submitted = self.jobs.start(job, true).map_err(lane_lost)?;
        self.joining = Some(submitted);
        Ok(None)
    }

    pub(crate) fn oldest_job(&self) -> Option<(JobId, std::time::Instant)> {
        self.jobs.oldest()
    }

    /// Wait at most `timeout` for the next completion of a job in flight,
    /// kept for the next [`Executor::advance`]; whether one arrived.
    pub(crate) fn await_completion(&self, timeout: std::time::Duration) -> Result<bool> {
        self.jobs.await_completion(timeout)
    }

    pub(crate) fn stall(
        &self,
        job: JobId,
        running: std::time::Duration,
        batch: Option<(u64, &WorkerBatch)>,
    ) -> Stall {
        let (lane, method) = self.jobs.performing(job).unwrap_or_default();
        Stall {
            lane,
            batch_id: batch.map(|(id, _)| id),
            route: batch
                .and_then(|(_, batch)| batch.route())
                .map(|kind| kind.as_str().to_owned()),
            method,
            calls: batch
                .map(|(_, batch)| batch.identities())
                .unwrap_or_default(),
            running_ms: u64::try_from(running.as_millis()).unwrap_or(u64::MAX),
        }
    }
}
