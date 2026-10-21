//! An image model's batch: the parts of the backend it executes with
//! ([`ImageParts`]), what it reserves and commits beside the shared batch
//! skeleton ([`ImageBatch`]), and the dispatch of its calls to the encoder,
//! denoiser and decoder forwards ([`forward`], [`complete`]).
//!
//! The opening job reads the batch's host-gated predicates (a false one
//! skips its call, which only publishes false flags), then
//! [`ImageBatch::open`] leases the batch's output storage on its stream and
//! binds the completion flags the calls publish. After a kind's
//! computation, [`ImageBatch::issue_commit`] validates every owner's update
//! (products, latent trajectories, pending calls) and records the device
//! half of the commit; [`ImageBatch::apply_commit`] makes them visible
//! together from the job's completion. A failure before that point retires
//! everything the batch reserved ([`ImageBatch::discard`]); a failure after
//! it is fatal.

use std::collections::HashMap;
use std::sync::Arc;

use cuda::{DeviceId, Stream};
use ipc::ImageParams;
use ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, CallKind, CallStatus, ErrorCode, ForwardStats,
    MediaCall, RequestKey,
};

use crate::error::WorkerError;
use crate::executor::PendingOutput;
use crate::executor::backend::{Parts, WorkerBackend, WorkerBatch};
use crate::executor::batch::{
    Batch, Failure, HostStage, Launched, Opening, Stage, fail, unsupported,
};
use crate::executor::products::{ProductOwners, Publications};
use crate::executor::transfer::latents::{self, PendingExport};
use crate::ffi::MemoryKind;
use crate::host::image::ImageResult;
use crate::host::image::ImageTask;
use crate::host::{HostLane, HostTask};
use crate::lane::{Completion, Job};
use crate::memory::LatentPool;
use crate::memory::{
    LatentCache, LatentStorage, LatentUpdate, OutputPool, ProductBuffer, ProductExport,
};
use crate::request::RequestPool;
use crate::request::diffusion::{BranchSource, Prefix, Trajectories};
use crate::runner::decoder::DecoderRunner;
use crate::runner::decoder::runners::DecoderRunners;
use crate::runner::denoiser::DenoiserRunner;
use crate::runner::denoiser::image_model::{ImageModel, ImagePolicy};
use crate::runner::denoiser::runners::DenoiserRunners;
use crate::runner::encoder::inputs::Inputs;
use crate::runner::encoder::runners::EncoderRunners;
use crate::stream::Issue;
use crate::worker::WorkerMemory;
use crate::{Error, Outcome, Result};

/// A decoded image's encoding, run once its pixels reached the host.
pub(crate) struct Encoding {
    /// The call's index in the batch.
    pub(crate) call: usize,
    pub(crate) task: Arc<HostTask<ImageTask>>,
}

/// A batch's persistent buffer allocations by buffer.
pub(crate) fn allocations(plan: &BatchPlan) -> HashMap<BufferId, BufferAllocation> {
    plan.buffer_allocations
        .iter()
        .map(|allocation| (allocation.buffer, *allocation))
        .collect()
}

/// What an image batch's issued commit leaves to apply once its work was
/// performed: the trajectory commits.
pub(crate) struct Commit {
    updates: Vec<LatentUpdate>,
}

/// The step an image batch's failure is reported in.
const PHASE: &str = "image batch";

/// A trajectory the batch's job asks the library to plan: it opens from
/// the job's completion, with the guidance plan and the prefixes the calls
/// recorded for it returned.
pub(crate) struct Planned {
    pub(crate) slot: usize,
    pub(crate) key: RequestKey,
    /// Image height and width in pixels.
    pub(crate) size: (u32, u32),
    /// The image parameters the trajectory was admitted with.
    pub(crate) params: Arc<ImageParams>,
    /// The sources a guidance branch may condition on, each with its
    /// prefix where the request gives it (admitted negative tokens); the
    /// others were asked for, in this order.
    pub(crate) prefixes: Vec<(BranchSource, Option<Prefix>)>,
}

/// What an image batch owns between two of its jobs: everything of
/// [`ImageBatch`] but the owners it borrows and its job.
pub(crate) struct Parked {
    batch: Batch,
    updates: Vec<(usize, LatentUpdate)>,
    encodings: Vec<Encoding>,
    publications: Publications,
    latent_exports: Vec<BufferId>,
    exports: Vec<(usize, PendingExport)>,
    features: Vec<(usize, ProductBuffer, ProductExport)>,
    returns: Vec<Vec<i64>>,
    plans: Vec<Planned>,
    latents_entered: bool,
}

/// How far recording a forward got.
pub(crate) enum Progress {
    /// The forward is recorded to its end.
    Done(Box<ForwardStats>),
    /// The job recorded so far ends here: the forward continues from its
    /// completion.
    Boundary(Resume),
}

impl Progress {
    /// A forward recorded to its end, with its statistics.
    pub(crate) fn done(stats: ForwardStats) -> Self {
        Self::Done(Box::new(stats))
    }
}

/// A forward between two of its jobs.
pub(crate) enum Resume {
    Encode(crate::runner::encoder::EncodeResume),
    /// A denoising that opens trajectories waits for their guidance plans.
    Plan(Box<crate::runner::denoiser::image::PlanResume>),
    Denoise(Box<crate::runner::denoiser::image::DenoiseResume>),
}

/// An image batch between two of its jobs.
pub(crate) enum Pending {
    /// The job on the lane reads the batch's host-gated predicates.
    Predicates(Opening),
    /// The predicates are known: the batch leases its storage and records
    /// its forward next.
    Opened(Opening),
    /// The job on the lane ends inside the batch's forward.
    Resuming(Box<(Parked, Resume)>),
    /// That job completed: the forward continues from its completion next.
    Resumed(Box<(Parked, Resume, Completion)>),
    /// The job on the lane is the batch's last: the rest of its forward
    /// and the issued commit.
    Committing(Box<(Parked, Commit, ForwardStats)>),
    /// Recording failed, or a job failed before the commit was applied.
    /// What the batch had recorded is performed; it retires what it
    /// reserved next.
    Failing(Box<Parked>, Failure),
    /// The job on the lane retires what the failed batch reserved, or
    /// performs what a batch that reserved nothing had recorded.
    Discarding(Failure),
}

/// An image batch's borrowed owners, the batch and what it reserves beside
/// the skeleton.
pub(crate) struct ImageBatch<'a> {
    pub(crate) parts: ImageParts<'a>,
    pub(crate) storage: &'a mut WorkerMemory,
    pub(crate) requests: &'a mut RequestPool,
    pub(crate) plan: &'a BatchPlan,
    /// What the batch's own steps issue on its stream: the start of its
    /// lease, its flags, its commit and its retirement after a failure.
    pub(crate) job: Job,
    pub(crate) batch: Batch,
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
    /// The feature exports the batch's forward issued, by call index, with
    /// the product each exports; located when the commit is applied.
    pub(crate) features: Vec<(usize, ProductBuffer, ProductExport)>,
    /// What each returning call the batch recorded must return, in order
    /// (the byte counts a decoder stages), checked when the commit is
    /// applied.
    pub(crate) returns: Vec<Vec<i64>>,
    /// The trajectories the batch's latent preparations planned, in the
    /// order their plan calls were recorded; opened when the commit is
    /// applied.
    pub(crate) plans: Vec<Planned>,
    /// The batch uses the latent pool on its stream (see
    /// `LatentCache::enter`).
    pub(crate) latents_entered: bool,
}

impl<'a> ImageBatch<'a> {
    /// Lease the opened batch's output storage on `stream`, `words` payload
    /// words beside one completion word per call, and bind the completion
    /// flags the calls publish.
    #[expect(clippy::too_many_arguments, reason = "the batch's borrowed owners")]
    pub(crate) fn open(
        parts: ImageParts<'a>,
        storage: &'a mut WorkerMemory,
        requests: &'a mut RequestPool,
        plan: &'a BatchPlan,
        opening: Opening,
        device: Option<DeviceId>,
        stream: Option<Stream>,
        words: usize,
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
        let job = Job::new();
        let batch = Batch::lease(
            opening,
            parts.outputs,
            plan.calls.len() + words,
            stream,
            device,
            &storage.events,
            &job,
        )?;
        let mut batch = Self::resume(
            parts,
            storage,
            requests,
            plan,
            Parked {
                batch,
                updates: Vec::new(),
                encodings: Vec::new(),
                publications: Publications::default(),
                latent_exports: Vec::new(),
                exports: Vec::new(),
                features: Vec::new(),
                returns: Vec::new(),
                plans: Vec::new(),
                latents_entered: false,
            },
        );
        batch.job = job;
        batch
            .batch
            .bind_flags(plan, &batch.storage.tensors, &batch.job)?;
        Ok(batch)
    }

    /// The batch's product device.
    pub(crate) fn target(&self) -> MemoryKind {
        self.batch.target()
    }

    /// Indices of the calls that execute.
    pub(crate) fn active(&self) -> Vec<usize> {
        self.batch.active()
    }

    /// Part the batch from the owners it borrows: its next job and the
    /// state it owns between two jobs.
    pub(crate) fn park(self) -> (Job, Parked) {
        (
            self.job,
            Parked {
                batch: self.batch,
                updates: self.updates,
                encodings: self.encodings,
                publications: self.publications,
                latent_exports: self.latent_exports,
                exports: self.exports,
                features: self.features,
                returns: self.returns,
                plans: self.plans,
                latents_entered: self.latents_entered,
            },
        )
    }

    /// The parked batch on the owners it borrows again, with an empty job.
    pub(crate) fn resume(
        parts: ImageParts<'a>,
        storage: &'a mut WorkerMemory,
        requests: &'a mut RequestPool,
        plan: &'a BatchPlan,
        parked: Parked,
    ) -> Self {
        Self {
            parts,
            storage,
            requests,
            plan,
            job: Job::new(),
            batch: parked.batch,
            updates: parked.updates,
            encodings: parked.encodings,
            publications: parked.publications,
            latent_exports: parked.latent_exports,
            exports: parked.exports,
            features: parked.features,
            returns: parked.returns,
            plans: parked.plans,
            latents_entered: parked.latents_entered,
        }
    }

    /// Validate every owner's update and record the device half of the
    /// commit: the end of the read leases, the completion flags, the end of
    /// the latent pool's use and the seal of the output. The returned
    /// commit is applied once that work was performed
    /// ([`Self::apply_commit`]).
    pub(crate) fn issue_commit(&mut self) -> std::result::Result<Commit, Failure> {
        let failed = Failure::new(PHASE);
        let tensors = &self.storage.tensors;
        self.batch
            .end_reads(tensors, &self.job, &[])
            .map_err(&failed)?;
        self.batch.set_flags(tensors, &self.job).map_err(&failed)?;
        self.batch.validate(self.plan, tensors).map_err(&failed)?;
        let updates: Vec<LatentUpdate> = self
            .updates
            .iter()
            .map(|(_, update)| update.clone())
            .collect();
        if !updates.is_empty() {
            self.parts
                .latents()
                .and_then(|latents| latents.pool.validate_updates(&updates))
                .map_err(&failed)?;
        }
        self.leave_latents().map_err(&failed)?;
        self.batch
            .seal(&self.storage.events, &self.job)
            .map_err(&failed)?;
        for (index, update) in &self.updates {
            self.batch.calls[*index]
                .output
                .apply_latent_update(update)
                .map_err(&failed)?;
        }
        self.requests
            .validate_pending(&Batch::<()>::pending(self.plan))
            .map_err(&failed)?;
        Ok(Commit { updates })
    }

    /// Apply an issued commit from the `completion` of the job that
    /// performed it: check what the forwards' calls returned, read the
    /// guidance plans they asked for, publish the exports they issued,
    /// then apply the trajectory commits, open the planned trajectories,
    /// register the exports, make the writes visible and record the calls
    /// as pending. A failure after the exports' check is fatal.
    pub(crate) fn apply_commit(
        &mut self,
        commit: Commit,
        completion: &mut Completion,
    ) -> std::result::Result<(), Failure> {
        let failed = Failure::new(PHASE);
        let Commit { updates } = commit;
        for expected in std::mem::take(&mut self.returns) {
            let returned = completion.returned().map_err(&failed)?;
            if returned != expected {
                return Err(failed(Error::Interface(format!(
                    "a decoding call staged {returned:?} bytes of images of {expected:?} bytes"
                ))));
            }
        }
        let mut planned = Vec::new();
        for plan in std::mem::take(&mut self.plans) {
            planned.push(self.planned_trajectory(plan, completion).map_err(&failed)?);
        }
        for (index, export) in std::mem::take(&mut self.exports) {
            let pool = &self.parts.latents().map_err(&failed)?.pool;
            latents::locate_export(
                pool,
                export,
                &mut self.publications,
                &mut self.batch.calls[index].output,
            )
            .map_err(&failed)?;
        }
        for (index, write, export) in std::mem::take(&mut self.features) {
            self.publications
                .locate(
                    &self.storage.tensors,
                    &write,
                    export,
                    &mut self.batch.calls[index].output,
                )
                .map_err(&failed)?;
        }
        self.publications
            .validate(&self.storage.tensors)
            .map_err(&failed)?;

        // Every owner accepted its update. Any failure from here is fatal.
        let fatal = Failure::fatal(PHASE);
        if !updates.is_empty() {
            let cancelled = self
                .parts
                .latents_mut()
                .map_err(&fatal)?
                .pool
                .apply_updates(&updates);
            for transfer in cancelled {
                // Released trajectories' pending imports end; their pages
                // retire with the transfer.
                transfer.read().cancel().map_err(&fatal)?;
            }
        }
        for (slot, trajectory) in planned {
            self.parts.trajectories.open(slot, trajectory);
        }
        self.publications
            .register(&self.storage.tensors)
            .map_err(&fatal)?;
        self.batch
            .commit(self.plan, &self.storage.tensors, self.requests)
    }

    /// End the batch's use of the latent pool on its stream.
    fn leave_latents(&mut self) -> Result<()> {
        if std::mem::take(&mut self.latents_entered) {
            self.parts
                .latents_mut()?
                .leave(self.batch.issue(&self.job))?;
        }
        Ok(())
    }

    /// Retire every reservation after a failure before the commit point,
    /// recording the fences that end them; storage returns once those
    /// fences were performed and complete.
    pub(crate) fn discard(&mut self) -> Result<()> {
        let left = self.leave_latents();
        self.exports
            .drain(..)
            .for_each(|(_, export)| export.revoke());
        self.features
            .drain(..)
            .for_each(|(_, _, export)| export.revoke());
        let revoked = self.publications.revoke();
        let exported = std::mem::take(&mut self.latent_exports);
        if !exported.is_empty()
            && let Ok(latents) = self.parts.latents()
        {
            // The versions retire once their transports drop them.
            latents.pool.release_exports(&exported);
        }
        for encoding in self.encodings.drain(..) {
            // Unconfigured encodings return their lane capacity.
            let _ = encoding.task.cancel(true);
        }
        let discarded = self
            .batch
            .discard(&self.storage.tensors, &self.storage.events, &self.job);
        left.and(revoked).and(discarded)
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
    pub(crate) latents: Option<&'a mut LatentCache>,
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

    /// The image denoiser.
    pub(crate) fn denoiser(&self) -> Option<&DenoiserRunner> {
        self.denoisers.image.as_ref().map(|(_, runner)| runner)
    }

    pub(crate) fn denoiser_mut(&mut self) -> Option<&mut DenoiserRunner> {
        self.denoisers.image.as_mut().map(|(_, runner)| runner)
    }

    pub(crate) fn latents(&self) -> Result<&LatentCache> {
        self.latents
            .as_deref()
            .ok_or_else(|| Error::Unsupported("image trajectories require a latent pool".into()))
    }

    fn latents_mut(&mut self) -> Result<&mut LatentCache> {
        self.latents
            .as_deref_mut()
            .ok_or_else(|| Error::Unsupported("image trajectories require a latent pool".into()))
    }

    /// The latent pool and its storage, for trajectory transfers.
    pub(crate) fn latent_pool(&mut self) -> Result<(&mut LatentPool, &LatentStorage)> {
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

/// The next job of an image batch: the read of its host-gated predicates
/// when it has any, then its forward with the issued commit, and after a
/// failure the retirement of what it reserved. `None` when the batch
/// failed without needing the lane again.
pub(crate) fn forward(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<Option<Job>, WorkerError> {
    let step = {
        let (plan, pending, inputs) = batch.image_parts();
        let step = record(backend.parts(), plan, kind, pending.take(), inputs)?;
        match step {
            Step::Job(job, state) => {
                *pending = Some(state);
                return Ok(Some(job));
            }
            Step::Failed(failure) => failure,
        }
    };
    fail(backend, batch, step).map(|()| None)
}

/// What recording an image batch's next job leaves.
enum Step {
    /// The job to submit, and the batch's state while it is on the lane.
    Job(Job, Pending),
    /// The batch failed and holds nothing.
    Failed(Failure),
}

/// The encoder, denoiser or decoder serving a batch of `kind`: its device,
/// its stream and the payload words the batch reads back.
fn placement(
    kind: CallKind,
    image: &ImageParts<'_>,
    plan: &BatchPlan,
) -> Result<(Option<DeviceId>, Option<Stream>, usize)> {
    match kind {
        CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding) => {
            crate::runner::encoder::image_placement(image, plan)
        }
        CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising) => {
            crate::runner::denoiser::image::image_placement(image)
        }
        CallKind::Media(MediaCall::ImageDecoding) => {
            crate::runner::decoder::image_placement(image, plan)
        }
        _ => Err(Error::Unsupported(format!(
            "{} is served by another runtime",
            kind.as_str()
        ))),
    }
}

/// Record the batch's next job from its state between jobs.
fn record(
    parts: Parts<'_>,
    plan: &BatchPlan,
    kind: CallKind,
    state: Option<Pending>,
    inputs: &mut Option<Inputs>,
) -> std::result::Result<Step, WorkerError> {
    let Parts {
        placement: group,
        rank,
        requests,
        storage,
        text,
        image,
        experts,
        ..
    } = parts;
    let (Some(image), Some(storage)) = (image, storage) else {
        return Err(unsupported(kind, "image"));
    };

    let opened = match state {
        Some(Pending::Opened(opening)) => Some(opening),
        None => None,
        Some(Pending::Failing(parked, failure)) => {
            // What the batch recorded was performed: retire what it
            // reserved, behind that work.
            let mut batch = ImageBatch::resume(image, storage, requests, plan, *parked);
            let discarded = batch.discard();
            let (job, _) = batch.park();
            let failure = match discarded {
                Ok(()) => failure,
                // Storage the cleanup could not retire stays reserved; the
                // rank cannot reuse it safely.
                Err(cleanup) => Failure::fatal(PHASE)(cleanup),
            };
            return Ok(Step::Job(job, Pending::Discarding(failure)));
        }
        Some(Pending::Resumed(resumed)) => {
            // The forward continues from the completion of its last job.
            let (parked, resume, mut completion) = *resumed;
            let mut batch = ImageBatch::resume(image, storage, requests, plan, parked);
            let progressed = match resume {
                Resume::Encode(resume) => batch.resume_encode(resume, &mut completion),
                Resume::Plan(resume) => {
                    batch.resume_planned(*resume, &mut completion, text, experts)
                }
                Resume::Denoise(resume) => {
                    batch.resume_denoise(*resume, &mut completion, text, experts)
                }
            };
            return Ok(advance(batch, progressed.map_err(Failure::new(PHASE))));
        }
        Some(
            Pending::Predicates(_)
            | Pending::Resuming(_)
            | Pending::Committing(_)
            | Pending::Discarding(_),
        ) => {
            return Err(WorkerError::classify(
                Error::Invariant(
                    "an image batch's job was prepared before its last completed".into(),
                ),
                Some("image batch"),
            ));
        }
    };
    let (device, stream, words) = match placement(kind, &image, plan) {
        Ok(placed) => placed,
        Err(error) => return Ok(Step::Failed(Failure::new(PHASE)(error))),
    };

    // The batch's first job binds its calls and reads its host-gated
    // predicates: they decide which calls it reserves for.
    let opening = match opened {
        Some(opening) => opening,
        None => {
            let job = Job::new();
            let opened = Opening::open(
                requests,
                &storage.tensors,
                group,
                rank,
                plan,
                device,
                job.on(stream.as_ref()),
                |_| true,
            );
            match opened {
                Ok(opening) if opening.reads_predicates() => {
                    return Ok(Step::Job(job, Pending::Predicates(opening)));
                }
                Ok(opening) => opening,
                // The reads recorded so far are performed; the batch
                // reserved nothing.
                Err(error) if !job.is_empty() => {
                    return Ok(Step::Job(
                        job,
                        Pending::Discarding(Failure::new(PHASE)(error)),
                    ));
                }
                Err(error) => return Ok(Step::Failed(Failure::new(PHASE)(error))),
            }
        }
    };

    let opened = ImageBatch::open(
        image, storage, requests, plan, opening, device, stream, words,
    );
    let mut batch = match opened {
        Ok(batch) => batch,
        Err(error) => return Ok(Step::Failed(Failure::new(PHASE)(error))),
    };
    let progressed = (|| -> Result<Progress> {
        Ok(match kind {
            CallKind::Media(call @ (MediaCall::VisionEncoding | MediaCall::LatentEncoding)) => {
                let images = inputs
                    .take()
                    .map(Inputs::take)
                    .transpose()?
                    .unwrap_or_default();
                batch.encode(call, images)?
            }
            CallKind::Media(MediaCall::LatentPreparation) => {
                Progress::done(batch.prepare_latents()?)
            }
            CallKind::Media(MediaCall::Denoising) => batch.denoise(text, experts)?,
            CallKind::Media(MediaCall::ImageDecoding) => Progress::done(batch.decode()?),
            _ => Progress::done(ForwardStats::default()),
        })
    })()
    .map_err(Failure::new(PHASE));
    Ok(advance(batch, progressed))
}

/// Park the batch behind what its forward recorded: its last job with the
/// issued commit when the forward is recorded to its end, the job that
/// ends inside the forward otherwise, and what a failed recording had
/// recorded.
fn advance(mut batch: ImageBatch<'_>, progressed: std::result::Result<Progress, Failure>) -> Step {
    // The commit is issued behind a forward recorded to its end.
    let progressed = progressed.and_then(|progress| match progress {
        Progress::Done(stats) => batch
            .issue_commit()
            .map(|commit| Recorded::Whole(commit, stats)),
        Progress::Boundary(resume) => Ok(Recorded::Part(resume)),
    });

    let (job, parked) = batch.park();
    let state = match progressed {
        Ok(Recorded::Whole(commit, stats)) => {
            Pending::Committing(Box::new((parked, commit, *stats)))
        }
        Ok(Recorded::Part(resume)) => Pending::Resuming(Box::new((parked, resume))),
        // What the batch recorded before it failed is performed before its
        // reservations are retired: the events it queued are due, and its
        // writes are abandoned only behind its work.
        Err(failure) => Pending::Failing(Box::new(parked), failure),
    };
    Step::Job(job, state)
}

/// What a batch's job holds of its forward.
enum Recorded {
    /// The rest of the forward, with the issued commit.
    Whole(Commit, Box<ForwardStats>),
    /// The forward up to a job boundary.
    Part(Resume),
}

/// Apply the completion of the job [`forward`] prepared last.
pub(crate) fn complete(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
    completion: Completion,
) -> std::result::Result<(), WorkerError> {
    /// What the completion leaves to do on the worker batch.
    enum Done {
        Next(Pending),
        Launched(Box<Launched>),
        Failed(Failure),
    }

    let done = {
        let (plan, pending, _) = batch.image_parts();
        match (pending.take(), completion.check()) {
            // A predicate that could not be read fails the batch, which
            // has reserved nothing yet.
            (Some(Pending::Predicates(_)), Err(failure)) => {
                Done::Failed(Failure::new(PHASE)(failure.error))
            }
            (Some(Pending::Predicates(mut opening)), Ok(mut completion)) => {
                match opening.predicates(&mut completion) {
                    Ok(()) => Done::Next(Pending::Opened(opening)),
                    Err(error) => Done::Failed(Failure::new(PHASE)(error)),
                }
            }
            // A job that ends inside the forward completed: the forward
            // continues from its completion in the next job.
            (Some(Pending::Resuming(resuming)), Ok(completion)) => {
                let (parked, resume) = *resuming;
                Done::Next(Pending::Resumed(Box::new((parked, resume, completion))))
            }
            (Some(Pending::Resuming(resuming)), Err(failure)) => {
                let (parked, _) = *resuming;
                Done::Next(Pending::Failing(
                    Box::new(parked),
                    Failure::new(PHASE)(failure.error),
                ))
            }
            // The job failed on its lane, which still made its event
            // records: the batch retires what it reserved.
            (Some(Pending::Committing(committing)), Err(failure)) => {
                let (parked, _, _) = *committing;
                Done::Next(Pending::Failing(
                    Box::new(parked),
                    Failure::new(PHASE)(failure.error),
                ))
            }
            (Some(Pending::Committing(committing)), Ok(mut completion)) => {
                let (parked, commit, mut stats) = *committing;
                let Parts {
                    requests,
                    storage,
                    image,
                    ..
                } = backend.parts();
                let (Some(image), Some(storage)) = (image, storage) else {
                    return Err(unsupported(kind, "image"));
                };
                let mut resumed = ImageBatch::resume(image, storage, requests, plan, parked);
                match resumed.apply_commit(commit, &mut completion) {
                    Ok(()) => {
                        let elapsed = resumed.batch.started.elapsed().as_micros() as u64;
                        stats.component_us.insert("image_execute".into(), elapsed);
                        let encodings = std::mem::take(&mut resumed.encodings);
                        let host = (!encodings.is_empty()).then(|| {
                            Box::new(Encodings {
                                encodings,
                                submitted: false,
                            }) as Box<dyn HostStage>
                        });
                        let (_, parked) = resumed.park();
                        Done::Launched(Box::new(parked.batch.launch(
                            stats,
                            "image_materialize",
                            host,
                        )))
                    }
                    Err(failure) if failure.committed => Done::Failed(failure),
                    Err(failure) => {
                        let (_, parked) = resumed.park();
                        Done::Next(Pending::Failing(Box::new(parked), failure))
                    }
                }
            }
            // The lane performed what the failed batch had recorded;
            // whether that failed again changes nothing of what follows.
            (Some(Pending::Failing(parked, failure)), _) => {
                Done::Next(Pending::Failing(parked, failure))
            }
            (Some(Pending::Discarding(failure)), Ok(_)) => Done::Failed(failure),
            // Fences the retirement recorded were not all made.
            (Some(Pending::Discarding(_)), Err(cleanup)) => {
                Done::Failed(Failure::fatal(PHASE)(cleanup.error))
            }
            (Some(Pending::Opened(_) | Pending::Resumed(_)) | None, _) => {
                return Err(WorkerError::classify(
                    Error::State("an image batch has no job on its lane"),
                    Some("image batch"),
                ));
            }
        }
    };
    match done {
        Done::Next(state) => {
            *batch.image_parts().1 = Some(state);
            Ok(())
        }
        Done::Launched(launched) => {
            batch.launch(*launched);
            batch.set_stage(Stage::Done);
            Ok(())
        }
        Done::Failed(failure) => fail(backend, batch, failure),
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

    fn finish(
        &mut self,
        _plan: &BatchPlan,
        _products: ProductOwners<'_>,
        outputs: &mut [PendingOutput],
    ) -> Result<()> {
        // The encoded images are delivered as media; nothing is published
        // as a product.
        for encoding in std::mem::take(&mut self.encodings) {
            let output = outputs
                .get_mut(encoding.call)
                .ok_or_else(|| Error::Invariant("an image encoding names no call".into()))?;
            match encoding.task.completion.outcome() {
                Some(Outcome::Success(result)) => {
                    if let ImageResult::Encoded(bytes) = &*result
                        && output.reports_output
                    {
                        let source = ipc::MediaSource::publish(bytes).map_err(|error| {
                            Error::Invalid(format!("publishing an encoded image failed: {error}"))
                        })?;
                        output.set_media(source)?;
                    }
                }
                Some(Outcome::Failed(error)) => {
                    tracing::error!("image encoding failed: {error}");
                    output.output.status = CallStatus::Error;
                    output.output.error_code = Some(ErrorCode::ComputeError);
                }
                // The call's host task always completes before its batch
                // materializes, so a missing outcome breaks the batch's order.
                Some(Outcome::Cancelled) | None => {
                    output.output.status = CallStatus::Error;
                    output.output.error_code = Some(ErrorCode::InvariantViolation);
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
