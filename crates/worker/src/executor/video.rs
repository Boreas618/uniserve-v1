//! A video model's media batch: what its calls contribute beside the shared
//! batch skeleton ([`VideoCall`], [`MediaBatch`]), the parts of the backend
//! it executes with ([`Parts`]), and the dispatch of its calls to the
//! encoder, denoiser, decoder and host media runners ([`forward`],
//! [`complete`]).
//!
//! A media batch goes through the steps of [`crate::executor::batch`] as
//! jobs of its lane:
//!
//! 1. The opening job reads the batch's host-gated predicates, when it has
//!    any. A call whose predicate is false is skipped: it runs nothing and
//!    forwards a false decision to its successor through its completion
//!    flag.
//! 2. When the batch's condition encodings lack an encoder the library
//!    constructs for new input shapes, a job asks for it; the forwards are
//!    recorded from its completion.
//! 3. The last job leases the batch's output storage on the stream of its
//!    kind, binds the completion flags and holds every active call's
//!    forward: device calls on the component's stream, host calls as tasks
//!    of the rank's host lane, submitted while the job is recorded. Behind
//!    the forwards, [`MediaBatch::issue_commit`] records the device half of
//!    the commit (the end of the input leases, the executed calls' flags,
//!    the seal of the lease) after every owner validated its update.
//! 4. From that job's completion, [`MediaBatch::apply_commit`] publishes
//!    the exports the forwards issued, applies the trajectory commits to
//!    the video denoiser's latent pool, makes the writes visible and the
//!    calls pending: the commit point. A failure before it retires everything the batch
//!    reserved in a job behind what it recorded, and the batch's calls fail;
//!    a failure at or past it is fatal.
//! 5. The launched batch materializes once its output fences completed and
//!    its host tasks finished ([`HostTasks`]): what the tasks wrote is then
//!    published as host products, after the commit point, and the output
//!    rank delivers a finished MP4.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use uniserve_core::{CallId, MediaSource};
use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, Call, CallKind, CallStatus, ErrorCode,
    ForwardStats, MediaCall, TensorRef, WorkerInfo,
};

use crate::config::Device;
use crate::cuda_graph::GraphMode;
use crate::error::WorkerError;
use crate::executor::PendingOutput;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::batch::{Batch, Failure, HostStage, Launched, Opening, Stage, fail};
use crate::executor::inputs::Inputs;
use crate::executor::products::{
    OutputShape, Placement, ProductOwners, Publications, bind_outputs, export,
};
use crate::ffi::MemoryKind;
use crate::host::HostTask;
use crate::host::task::{MediaTask, TaskOutput};
use crate::lane::{Completion, Job};
use crate::request::RequestPool;
use crate::request::video::VideoRequests;
use crate::runner::decoder::runners::DecoderRunners;
use crate::runner::denoiser::DenoiserRunner;
use crate::runner::denoiser::runners::DenoiserRunners;
use crate::runner::encoder::ConditionSetup;
use crate::runner::encoder::runners::EncoderRunners;
use crate::runner::media::MediaRunner;
use crate::storage::ProductExport;
use crate::stream::Issue;
use crate::worker::RankStorage;
use crate::{Error, LatentUpdate, OutputPool, ProductBuffer, ProductRead, ProductStore, Result};

/// The step a media batch's failure is reported in.
const PHASE: &str = "media batch";

/// The component timer name of a media batch's materialization.
const MATERIALIZE: &str = "video_materialize";

/// How a call publishes what its host tasks wrote.
pub(crate) enum Publication {
    /// Products the task wrote whole, exported as host products.
    Products(Vec<(TensorRef, ProductBuffer)>),
    /// One encoded unit per task in consecutive rows of one product, each
    /// exported up to its framed length.
    Encoded {
        product: TensorRef,
        write: ProductBuffer,
        /// The product row each task framed into.
        rows: Vec<u64>,
    },
    /// The finished MP4, delivered by the component's output rank.
    Media,
}

/// What a call contributes to its media batch's forward, beside the state
/// every batch keeps of it ([`crate::executor::batch::CallState`]).
#[derive(Default)]
pub(crate) struct VideoCall {
    /// The trajectory commit the call's forward recorded, applied to the
    /// denoiser's latent pool at the commit point.
    pub(crate) latent: Option<LatentUpdate>,
    /// The exports the call's forward issued, located when the batch
    /// commits: their device work is performed by then.
    exports: Vec<(ProductBuffer, ProductExport)>,
    /// The host tasks writing the call's results, submitted while the
    /// batch's job is recorded.
    pub(crate) tasks: Vec<Arc<HostTask<MediaTask>>>,
    /// How the call publishes what its tasks wrote once they finished.
    pub(crate) publication: Option<Publication>,
}

impl VideoCall {
    /// Issue the export of `write`, the call's output `product`, to
    /// `stream`, behind the work that produced it.
    pub(crate) fn export(
        &mut self,
        owners: ProductOwners<'_>,
        call: &Call,
        product: &TensorRef,
        write: &ProductBuffer,
        host: bool,
        stream: impl Issue,
    ) -> Result<()> {
        let export = owners.store.export_buffer(
            call,
            product,
            write,
            owners.transports,
            host,
            None,
            stream,
        )?;
        self.exports.push((write.clone(), export));
        Ok(())
    }
}

/// A media batch whose output storage is leased: the skeleton batch and
/// what a media batch keeps beside it. It borrows nothing, so it waits
/// between its jobs as it is.
pub(crate) struct MediaBatch {
    pub(crate) batch: Batch<VideoCall>,
    /// The inputs host codecs read in place from their shared-memory
    /// exports instead of leasing them from the product store.
    borrowed: HashSet<BufferId>,
    /// The exports the batch accepted, registered at its commit point.
    publications: Publications,
    /// The graph modes of the batch's forwards, reported with its result.
    stats: ForwardStats,
}

impl MediaBatch {
    /// Count one forward of the batch by the graph mode it ran in: a
    /// captured graph's launch, a capture, or an eager call.
    pub(crate) fn record(&mut self, mode: GraphMode) {
        let name = mode.as_str();
        *self.stats.mode_counts.entry(name.into()).or_default() += 1;
        match mode {
            GraphMode::Capture => self.stats.cuda_graph_captures += 1,
            GraphMode::Replay => self.stats.cuda_graph_replays += 1,
            GraphMode::Eager => {}
        }
        if mode != GraphMode::Eager {
            *self
                .stats
                .cuda_graph_runtime_mode_counts
                .entry(name.into())
                .or_default() += 1;
        }
    }

    /// The device the batch's products live on.
    pub(crate) fn target(&self) -> MemoryKind {
        self.batch.target()
    }

    /// Whether an input is read in place from its export.
    pub(crate) fn borrowed(&self, product: &TensorRef) -> bool {
        self.borrowed.contains(&product.buffer_id())
    }

    /// Lease one input for `consumer`, whole, on the batch's device; what
    /// `stream` issues next waits on its writer. The lease ends with the
    /// batch's other reads, when it commits or retires.
    pub(crate) fn read(
        &mut self,
        store: &ProductStore,
        product: &TensorRef,
        consumer: CallId,
        stream: impl Issue,
    ) -> Result<Arc<ProductRead>> {
        let read = store.consume(product, consumer, Some(self.target()), stream)?;
        // Kept before the check below, so a refused read still ends.
        self.batch.reads.push(Arc::clone(&read));
        if read.region().is_some() {
            return Err(Error::Invalid(
                "media computation requires complete input coverage".into(),
            ));
        }
        Ok(read)
    }

    /// The trajectory commits the calls' forwards recorded.
    fn updates(&self) -> Vec<LatentUpdate> {
        self.batch
            .calls
            .iter()
            .filter_map(|state| state.forward.latent.clone())
            .collect()
    }

    /// Validate every owner's update but the exports' and record the
    /// device half of the commit in `job`, behind the forwards: the end of
    /// the input leases, the executed calls' completion flags and the seal
    /// of the output lease, whose fences retire the batch's storage
    /// accesses. [`Self::apply_commit`] applies the updates once the job
    /// was performed.
    fn issue_commit(
        &mut self,
        plan: &BatchPlan,
        storage: &RankStorage,
        requests: &RequestPool,
        trajectories: Option<&DenoiserRunner>,
        job: &Job,
    ) -> std::result::Result<(), Failure> {
        let failed = Failure::new(PHASE);
        let tensors = &storage.tensors;
        self.batch.end_reads(tensors, job, &[]).map_err(&failed)?;
        self.batch.set_flags(tensors, job).map_err(&failed)?;
        self.batch.validate(plan, tensors).map_err(&failed)?;
        let updates = self.updates();
        match trajectories {
            Some(pool) => pool.validate(&updates),
            None if updates.is_empty() => Ok(()),
            None => Err(Error::Invariant(
                "a trajectory commit has no latent pool".into(),
            )),
        }
        .map_err(&failed)?;
        requests
            .validate_pending(&Batch::<()>::pending(plan))
            .map_err(&failed)?;
        self.batch.seal(&storage.events, job).map_err(&failed)
    }

    /// Apply the issued commit once its job was performed: publish the
    /// exports the forwards issued, then apply the trajectory commits to
    /// `trajectories`, register the exports, make the writes visible and
    /// record the calls as pending. A failure after the exports' check is
    /// fatal.
    fn apply_commit(
        &mut self,
        plan: &BatchPlan,
        tensors: &ProductStore,
        requests: &mut RequestPool,
        trajectories: Option<&mut DenoiserRunner>,
    ) -> std::result::Result<(), Failure> {
        let failed = Failure::new(PHASE);
        for state in &mut self.batch.calls {
            for (write, export) in std::mem::take(&mut state.forward.exports) {
                self.publications
                    .locate(tensors, &write, export, &mut state.output)
                    .map_err(&failed)?;
            }
        }
        self.publications.validate(tensors).map_err(&failed)?;

        // Every owner accepted its update. Any failure from here is fatal.
        let fatal = Failure::fatal(PHASE);
        if let Some(pool) = trajectories {
            pool.apply(&self.updates());
        }
        for state in &mut self.batch.calls {
            if let Some(update) = state.forward.latent.take() {
                state.output.apply_latent_update(&update);
            }
        }
        self.publications.register(tensors).map_err(&fatal)?;
        self.batch.commit(plan, tensors, requests)
    }

    /// Retire every reservation after a failure before the commit point,
    /// recording in `job` the fences that end them: withdraw the host tasks
    /// not running yet, revoke the exports, end the reads, abandon the
    /// writes and seal and abandon the lease. A running task keeps its
    /// inputs and outputs to completion, so the products host tasks write
    /// stay deferred rather than abandoned under it.
    fn discard(&mut self, storage: &RankStorage, job: &Job) -> Result<()> {
        for state in &mut self.batch.calls {
            withdraw(&state.forward.tasks);
            state
                .forward
                .exports
                .drain(..)
                .for_each(|(_, export)| export.revoke());
        }
        let revoked = self.publications.revoke();
        let discarded = self.batch.discard(&storage.tensors, &storage.events, job);
        revoked.and(discarded)
    }

    /// Hand the committed batch to materialization, with the host tasks its
    /// calls' results wait for.
    fn launch(self) -> Launched {
        let Self {
            mut batch, stats, ..
        } = self;
        let calls: Vec<HostCall> = batch
            .calls
            .iter_mut()
            .enumerate()
            .filter(|(_, state)| !state.forward.tasks.is_empty())
            .map(|(index, state)| HostCall {
                index,
                tasks: std::mem::take(&mut state.forward.tasks),
                publication: state.forward.publication.take(),
            })
            .collect();
        let host = (!calls.is_empty()).then(|| Box::new(HostTasks { calls }) as Box<dyn HostStage>);
        batch.launch(stats, MATERIALIZE, host)
    }
}

/// Withdraw host tasks that are not running yet; a running task completes.
fn withdraw(tasks: &[Arc<HostTask<MediaTask>>]) {
    for task in tasks {
        if let Err(error) = task.cancel(false) {
            tracing::warn!(target: "uniserve_worker.media", "withdrawing a host media task failed: {error}");
        }
    }
}

/// One call's host tasks in a launched batch, with how the call publishes
/// what they wrote.
struct HostCall {
    /// The call's index in the batch.
    index: usize,
    tasks: Vec<Arc<HostTask<MediaTask>>>,
    publication: Option<Publication>,
}

/// The host tasks a launched media batch's results wait for. What they
/// write is published once every task finished, after the commit point:
/// such a product is committed and its exports registered at once.
struct HostTasks {
    calls: Vec<HostCall>,
}

impl HostStage for HostTasks {
    fn poll(&mut self) -> Result<bool> {
        // The tasks were submitted when the batch's job was recorded.
        Ok(self
            .calls
            .iter()
            .flat_map(|call| &call.tasks)
            .all(|task| task.completion.done()))
    }

    fn finish(
        &mut self,
        plan: &BatchPlan,
        products: ProductOwners<'_>,
        outputs: &mut [PendingOutput],
    ) -> Result<()> {
        for host in std::mem::take(&mut self.calls) {
            let (Some(call), Some(output)) =
                (plan.calls.get(host.index), outputs.get_mut(host.index))
            else {
                return Err(Error::Invariant("a host media task names no call".into()));
            };
            host.finish(call, products, output)?;
        }
        Ok(())
    }

    fn cancel(&mut self) {
        for call in &self.calls {
            withdraw(&call.tasks);
        }
    }
}

impl HostCall {
    /// Publish what the call's finished tasks wrote, or fail the call when
    /// a task did not succeed (its host products are abandoned then).
    fn finish(
        self,
        call: &Call,
        owners: ProductOwners<'_>,
        output: &mut PendingOutput,
    ) -> Result<()> {
        let mut results = Vec::with_capacity(self.tasks.len());
        let mut failure = None;
        for task in &self.tasks {
            match task.completion.outcome() {
                Some(crate::Outcome::Success(result)) => results.push(result),
                Some(crate::Outcome::Failed(error)) => {
                    failure.get_or_insert(error.to_string());
                }
                Some(crate::Outcome::Cancelled) | None => {
                    failure.get_or_insert("a host media task was cancelled".to_owned());
                }
            }
        }
        if let Some(message) = failure {
            tracing::error!(target: "uniserve_worker.media", "host media call failed: {message}");
            output.output.status = CallStatus::Error;
            output.output.error_code = Some(ErrorCode::ComputeError);
            let writes: Vec<ProductBuffer> = match self.publication {
                Some(Publication::Products(written)) => {
                    written.into_iter().map(|(_, write)| write).collect()
                }
                Some(Publication::Encoded { write, .. }) => vec![write],
                _ => Vec::new(),
            };
            return owners.store.abandon_writes(&writes);
        }

        // Written by the host after the commit point: each product is
        // committed, exported and its exports registered at once.
        let mut publications = Publications::default();
        match self.publication {
            Some(Publication::Products(written)) => {
                for (product, write) in written {
                    let view = write.view();
                    owners.store.write(&write, &view, None, None)?;
                    owners.store.commit_writes(std::slice::from_ref(&write))?;
                    let (exported, locations) =
                        export(owners, call, &product, &write, true, None, None)?;
                    publications.accept(write.id(), locations, exported, output)?;
                    publications.register(owners.store)?;
                }
            }
            Some(Publication::Encoded {
                product,
                write,
                rows,
            }) => {
                let view = write.view();
                owners.store.write(&write, &view, None, None)?;
                owners.store.commit_writes(std::slice::from_ref(&write))?;
                let width = view.shape().get(1).copied().unwrap_or(0) as u64;
                let mut regions = Vec::with_capacity(rows.len());
                for (result, row) in results.iter().zip(&rows) {
                    let TaskOutput::Framed(length) = **result else {
                        return Err(Error::Invariant(
                            "a video encoding task framed no unit".into(),
                        ));
                    };
                    if length as u64 > width {
                        return Err(Error::Invariant("a framed unit exceeds its row".into()));
                    }
                    regions.push(vec![*row..*row + 1, 0..length as u64]);
                }
                let (exported, locations) =
                    export(owners, call, &product, &write, true, Some(&regions), None)?;
                publications.accept(write.id(), locations, exported, output)?;
                publications.register(owners.store)?;
            }
            Some(Publication::Media) if output.reports_output => {
                for result in &results {
                    if let TaskOutput::Media(bytes) = &**result {
                        let source = MediaSource::publish(bytes).map_err(|error| {
                            Error::Transport(format!("publishing media failed: {error}"))
                        })?;
                        output.set_media(source)?;
                    }
                }
            }
            Some(Publication::Media) | None => {}
        }
        Ok(())
    }
}

/// A media batch between two of its jobs.
pub(crate) enum Pending {
    /// The job on the lane reads the batch's host-gated predicates.
    Predicates(Opening<VideoCall>),
    /// The predicates are known: the batch records its next job.
    Opened(Opening<VideoCall>),
    /// The job on the lane asks the library for what the batch's condition
    /// encodings lack; the batch goes on from its completion.
    Setup(Opening<VideoCall>, ConditionSetup),
    /// The job on the lane is the batch's last: its forwards and the
    /// issued commit.
    Committing(Box<MediaBatch>),
    /// Recording failed, or the last job failed before the commit was
    /// applied. What the batch had recorded is performed; it retires what
    /// it reserved next.
    Failing(Box<MediaBatch>, Failure),
    /// The job on the lane retires what the failed batch reserved, or
    /// performs what a batch that reserved nothing had recorded.
    Discarding(Failure),
}

/// What recording a media batch's next job leaves.
enum Step {
    /// The job to submit, and the batch's state while it is on the lane.
    Job(Job, Pending),
    /// The batch failed and holds nothing.
    Failed(Failure),
}

/// The parts of the backend a media batch executes with.
pub(crate) struct Parts<'a> {
    pub(crate) requests: &'a mut RequestPool,
    pub(crate) storage: &'a RankStorage,
    pub(crate) info: &'a WorkerInfo,
    pub(crate) rank: usize,
    /// The rank's video requests, with the model and plan that admit them.
    pub(crate) video: &'a mut VideoRequests,
    pub(crate) encoders: &'a mut EncoderRunners,
    pub(crate) denoisers: &'a mut DenoiserRunners,
    pub(crate) decoders: &'a mut DecoderRunners,
    /// The host media of a rank holding host components.
    pub(crate) media: Option<&'a MediaRunner>,
    /// The output leases of media batches.
    pub(crate) outputs: &'a mut OutputPool,
}

impl<'a> Parts<'a> {
    /// The product owners, borrowed for the parts' lifetime rather than
    /// from the parts themselves.
    pub(crate) fn owners(&self) -> ProductOwners<'a> {
        let storage: &'a RankStorage = self.storage;
        ProductOwners {
            store: &storage.tensors,
            transports: &storage.transports,
        }
    }

    /// The stream and device a batch of `kind` on `component` runs on.
    pub(crate) fn placement(
        &self,
        kind: MediaCall,
        component: &str,
    ) -> Result<(Device, Option<Stream>)> {
        match kind {
            MediaCall::TextEncoding | MediaCall::VisionEncoding => {
                let text = self
                    .encoders
                    .text
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no text encoder".into()))?;
                Ok((text.device, text.runner.stream().cloned()))
            }
            MediaCall::LatentPreparation | MediaCall::Denoising => {
                let denoiser = self
                    .denoisers
                    .video
                    .as_ref()
                    .filter(|denoiser| denoiser.name == component);
                let denoiser = denoiser
                    .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
                Ok((denoiser.device, denoiser.runner.stream().cloned()))
            }
            MediaCall::VideoDecoding => {
                let decoder = self
                    .decoders
                    .video
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no video decoder".into()))?;
                Ok((decoder.device, decoder.runner.stream().cloned()))
            }
            MediaCall::AudioDecoding => {
                let audio = self
                    .decoders
                    .audio
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no audio decoder".into()))?;
                Ok((audio.device, audio.runner.stream().cloned()))
            }
            MediaCall::LatentEncoding => {
                let latent =
                    self.encoders.latent.as_ref().ok_or_else(|| {
                        Error::Invalid("this rank holds no latent encoder".into())
                    })?;
                Ok((latent.device, latent.runner.stream().cloned()))
            }
            _ => Ok((Device::Cpu, None)),
        }
    }
}

/// The name of a call's output product.
pub(crate) fn product_name<'a>(
    info: &'a WorkerInfo,
    call: &Call,
    output: &TensorRef,
) -> Result<&'a str> {
    info.components
        .iter()
        .find(|component| component.name == call.component)
        .and_then(|component| component.outputs.get(output.output_index as usize))
        .map(|output| output.name.as_str())
        .ok_or_else(|| {
            Error::Invalid(format!(
                "component {} declares no output {}",
                call.component, output.output_index
            ))
        })
}

fn media_kind(call: &Call) -> MediaCall {
    match call.code {
        CallKind::Media(kind) => kind,
        _ => MediaCall::Denoising,
    }
}

/// The next job of a media batch: the read of its host-gated predicates
/// when it has any, the library's setup of new condition encodings when
/// they need it, then its forwards with the issued commit, and after a
/// failure the retirement of what it reserved. `None` when the batch
/// failed without needing the lane again.
pub(crate) fn forward(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
) -> std::result::Result<Option<Job>, WorkerError> {
    let failure = {
        let mut parts = backend.media_parts()?;
        let (plan, inputs, pending) = batch.media_parts();
        match record(&mut parts, plan, inputs, pending.take())? {
            Step::Job(job, state) => {
                *pending = Some(state);
                return Ok(Some(job));
            }
            Step::Failed(failure) => failure,
        }
    };
    fail(backend, batch, failure).map(|()| None)
}

/// Record the batch's next job from its state between jobs.
fn record(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    inputs: &Inputs,
    state: Option<Pending>,
) -> std::result::Result<Step, WorkerError> {
    let opened = match state {
        None => None,
        Some(Pending::Opened(opening)) => Some(opening),
        // What the batch recorded was performed: retire what it reserved,
        // behind that work.
        Some(Pending::Failing(media, failure)) => {
            return Ok(retire(parts.storage, *media, Job::new(), failure));
        }
        Some(
            Pending::Predicates(_)
            | Pending::Setup(..)
            | Pending::Committing(_)
            | Pending::Discarding(_),
        ) => {
            return Err(WorkerError::classify(
                Error::Invariant(
                    "a media batch's job was prepared before its last completed".into(),
                ),
                Some(PHASE),
            ));
        }
    };
    let failed = Failure::new(PHASE);
    let Some(first) = plan.calls.first() else {
        return Ok(Step::Failed(failed(Error::Invariant(
            "a video batch has calls".into(),
        ))));
    };
    let (device, stream) = match parts.placement(media_kind(first), &first.component) {
        Ok((device, stream)) => (device.cuda(), stream),
        Err(error) => return Ok(Step::Failed(failed(error))),
    };

    // The batch's first job binds its calls and reads its host-gated
    // predicates: they decide which calls run.
    let opening = match opened {
        Some(opening) => opening,
        None => {
            let job = Job::new();
            let opened = Opening::open(
                parts.requests,
                &parts.storage.tensors,
                parts.info,
                parts.rank,
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
                Err(error) => return Ok(ended(job, failed(error))),
            }
        }
    };

    // A condition encoding of input shapes this rank has not encoded
    // before takes an encoder the library constructs. The job makes that
    // call, so the batch's forwards are recorded from its completion.
    let job = Job::new();
    match crate::runner::encoder::setup_conditions(parts, plan, opening.calls(), &job) {
        Ok(Some(setup)) => return Ok(Step::Job(job, Pending::Setup(opening, setup))),
        Ok(None) => {}
        Err(error) => return Ok(ended(job, failed(error))),
    }

    let leased = Batch::lease(
        opening,
        parts.outputs,
        plan.calls.len(),
        stream,
        device,
        &parts.storage.events,
        &job,
    );
    let mut media = match leased {
        Ok(batch) => MediaBatch {
            batch,
            borrowed: inputs.borrowed(),
            publications: Publications::default(),
            stats: ForwardStats::default(),
        },
        Err(error) => return Ok(ended(job, failed(error))),
    };
    let recorded = media
        .batch
        .bind_flags(plan, &parts.storage.tensors, &job)
        .and_then(|()| record_forwards(parts, plan, &mut media, &job))
        .map_err(&failed)
        .and_then(|()| {
            let trajectories = parts
                .denoisers
                .video
                .as_ref()
                .map(|denoiser| &denoiser.runner);
            media.issue_commit(plan, parts.storage, parts.requests, trajectories, &job)
        });
    Ok(match recorded {
        Ok(()) => Step::Job(job, Pending::Committing(Box::new(media))),
        // Nothing was recorded that the retirement must follow.
        Err(failure) if job.is_empty() => retire(parts.storage, media, job, failure),
        // What the batch recorded before it failed is performed before its
        // reservations are retired: the events it queued are due, and its
        // writes are abandoned only behind its work. An export that failed
        // kept only its event operations ([`Issue::attempt`]).
        Err(failure) => Step::Job(job, Pending::Failing(Box::new(media), failure)),
    })
}

/// Record every call's forward into `job`: a host call's tasks are
/// submitted to the rank's host lane, and the job holds only what the
/// batch writes for them; a device call's forward is recorded on the
/// batch's stream, unless its predicate skipped it.
fn record_forwards(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    media: &mut MediaBatch,
    job: &Job,
) -> Result<()> {
    let slots = media.batch.slots(plan);
    let allocations: HashMap<BufferId, BufferAllocation> = plan
        .buffer_allocations
        .iter()
        .map(|allocation| (allocation.buffer, *allocation))
        .collect();
    let placement = Placement {
        allocations: &allocations,
        slots: &slots,
    };
    for (index, call) in plan.calls.iter().enumerate() {
        let slot = media.batch.calls[index].slot;
        match call.code {
            CallKind::Media(MediaCall::MediaReading) => {
                crate::runner::media::read_media(parts, media, index, call, slot, &placement)?;
                continue;
            }
            CallKind::Media(MediaCall::VideoEncoding) => {
                crate::runner::media::encode_units(
                    parts, plan, media, index, call, slot, &placement,
                )?;
                continue;
            }
            CallKind::Media(MediaCall::AudioEncoding) => {
                crate::runner::media::encode_audio(parts, media, index, call, slot)?;
                continue;
            }
            CallKind::Media(MediaCall::Muxing) => {
                crate::runner::media::mux(parts, plan, media, index, call, slot)?;
                continue;
            }
            _ => {}
        }

        if !media.batch.calls[index].active {
            continue;
        }
        match call.code {
            CallKind::Media(MediaCall::TextEncoding) => crate::runner::encoder::encode_text(
                parts, plan, media, index, slot, &placement, job,
            )?,
            CallKind::Media(MediaCall::VisionEncoding) => crate::runner::encoder::encode_vision(
                parts, plan, media, index, slot, &placement, job,
            )?,
            CallKind::Media(MediaCall::LatentEncoding) => {
                crate::runner::encoder::encode_conditions(
                    parts, plan, media, index, slot, &placement, job,
                )?
            }
            CallKind::Media(MediaCall::LatentPreparation) => {
                crate::runner::denoiser::prepare(parts, plan, media, index, slot, job)?
            }
            CallKind::Media(MediaCall::Denoising) => {
                crate::runner::denoiser::denoise(parts, plan, media, index, slot, &placement, job)?
            }
            CallKind::Media(MediaCall::VideoDecoding) => crate::runner::decoder::decode_video(
                parts, plan, media, index, slot, &placement, job,
            )?,
            CallKind::Media(MediaCall::AudioDecoding) => crate::runner::decoder::decode_audio(
                parts, plan, media, index, slot, &placement, job,
            )?,
            kind => {
                return Err(Error::Unsupported(format!(
                    "{} is not a video model's media call",
                    kind.as_str()
                )));
            }
        }
    }
    Ok(())
}

/// The step of a batch that failed holding no reservation: `job`
/// performs what it recorded before, if anything, and the batch's calls
/// fail from its completion.
fn ended(job: Job, failure: Failure) -> Step {
    if job.is_empty() {
        Step::Failed(failure)
    } else {
        Step::Job(job, Pending::Discarding(failure))
    }
}

/// Retire what a failed batch reserved, recording in `job` the fences that
/// end it.
fn retire(storage: &RankStorage, mut media: MediaBatch, job: Job, failure: Failure) -> Step {
    let failure = match media.discard(storage, &job) {
        Ok(()) => failure,
        // Storage the cleanup could not retire stays reserved; the rank
        // cannot reuse it safely.
        Err(cleanup) => Failure::fatal(PHASE)(cleanup),
    };
    ended(job, failure)
}

/// Apply the completion of the job [`forward`] prepared last.
pub(crate) fn complete(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    completion: Completion,
) -> std::result::Result<(), WorkerError> {
    /// What the completion leaves to do on the worker batch.
    enum Done {
        Next(Pending),
        Launched(Box<Launched>),
        Failed(Failure),
    }

    let done = {
        let mut parts = backend.media_parts()?;
        let (plan, _, pending) = batch.media_parts();
        let failed = Failure::new(PHASE);
        match (pending.take(), completion.check()) {
            // A predicate that could not be read, or a setup the library
            // refused, fails the batch, which has reserved nothing yet.
            (Some(Pending::Predicates(_) | Pending::Setup(..)), Err(failure)) => {
                Done::Failed(failed(failure.error))
            }
            (Some(Pending::Predicates(mut opening)), Ok(mut completion)) => {
                match opening.predicates(&mut completion) {
                    Ok(()) => Done::Next(Pending::Opened(opening)),
                    Err(error) => Done::Failed(failed(error)),
                }
            }
            (Some(Pending::Setup(opening, setup)), Ok(mut completion)) => {
                match crate::runner::encoder::accept_conditions(&mut parts, setup, &mut completion)
                {
                    Ok(()) => Done::Next(Pending::Opened(opening)),
                    Err(error) => Done::Failed(failed(error)),
                }
            }
            // The job failed on its lane, which still made its event
            // records: the batch retires what it reserved.
            (Some(Pending::Committing(media)), Err(failure)) => {
                Done::Next(Pending::Failing(media, failed(failure.error)))
            }
            (Some(Pending::Committing(mut media)), Ok(_)) => {
                let trajectories = parts
                    .denoisers
                    .video
                    .as_mut()
                    .map(|denoiser| &mut denoiser.runner);
                match media.apply_commit(plan, &parts.storage.tensors, parts.requests, trajectories)
                {
                    Ok(()) => Done::Launched(Box::new(media.launch())),
                    Err(failure) if failure.committed => Done::Failed(failure),
                    Err(failure) => Done::Next(Pending::Failing(media, failure)),
                }
            }
            // The lane performed what the failed batch had recorded;
            // whether that failed again changes nothing of what follows.
            (Some(Pending::Failing(media, failure)), _) => {
                Done::Next(Pending::Failing(media, failure))
            }
            (Some(Pending::Discarding(failure)), Ok(_)) => Done::Failed(failure),
            // Fences the retirement recorded were not all made.
            (Some(Pending::Discarding(_)), Err(cleanup)) => {
                Done::Failed(Failure::fatal(PHASE)(cleanup.error))
            }
            (Some(Pending::Opened(_)) | None, _) => {
                return Err(WorkerError::classify(
                    Error::State("a media batch has no job on its lane"),
                    Some(PHASE),
                ));
            }
        }
    };
    match done {
        Done::Next(state) => {
            *batch.media_parts().2 = Some(state);
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

pub(crate) fn latent_params<'a>(
    plan: &'a BatchPlan,
    call: &Call,
) -> Result<&'a uniserve_ipc::LatentParams> {
    plan.latent_params
        .iter()
        .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
        .ok_or_else(|| Error::Invalid("a video trajectory call has no latent pages".into()))
}

pub(crate) fn decode_range<'a>(
    plan: &'a BatchPlan,
    call: &Call,
) -> Result<&'a uniserve_ipc::DecodeRange> {
    plan.decode_ranges
        .iter()
        .find(|range| range.request_key == call.request_key && range.call_id == call.call_id)
        .ok_or_else(|| Error::Invalid("a media round has no unit range".into()))
}

/// The batch's stream.
pub(crate) fn stream_of(batch: &MediaBatch) -> Result<Stream> {
    batch
        .batch
        .stream
        .clone()
        .ok_or_else(|| Error::Invariant("a device media call has no stream".into()))
}

/// The most elements a product holds along one dimension.
pub(crate) fn extent(dim: &uniserve_ipc::DimBound) -> u64 {
    match dim {
        uniserve_ipc::DimBound::Static(extent) | uniserve_ipc::DimBound::Device { max: extent } => {
            u64::from(*extent)
        }
    }
}

/// Rows' width of a two-dimensional product.
pub(crate) fn row_width(output: &TensorRef) -> Result<u64> {
    output
        .shape_bound
        .dims
        .get(1)
        .map(extent)
        .ok_or_else(|| Error::Invalid("the product is not rows".into()))
}

/// Bind a call's single output as rows of `shape` on every rank that
/// computes it; the batch commits the write.
pub(crate) fn bind_rows(
    parts: &Parts<'_>,
    batch: &mut MediaBatch,
    output: &TensorRef,
    shape: Vec<u64>,
    placement: &Placement<'_>,
) -> Result<crate::ProductBuffer> {
    let owners = parts.owners();
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: output.clone(),
            shape,
            region: None,
        }],
        batch.target(),
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("an encoder product was not bound".into()))?;
    batch.batch.writes.push(write.clone());
    Ok(write)
}

/// Publish a call's output, written at `shape` from its leading elements,
/// and export it to its readers, behind the work `stream` issued for it.
pub(crate) fn publish(
    parts: &Parts<'_>,
    batch: &mut MediaBatch,
    call: &Call,
    index: usize,
    write: &crate::ProductBuffer,
    shape: &[u64],
    stream: impl Issue,
) -> Result<()> {
    let owners = parts.owners();
    owners.store.produced(write, shape, stream, None)?;
    batch.batch.calls[index]
        .forward
        .export(owners, call, &call.outputs[0], write, false, stream)
}

/// Publish a call's first output, whose bound view the producer filled, and
/// export it to its readers (`host` readers read it from host memory).
///
/// The bound view is this rank's share of the product: its region when
/// other ranks hold the rest, the whole product when the round assigns
/// this rank all of it. Publishing the view itself is right in both cases;
/// the store tracks a share that covers the product as a whole product.
/// The fence and the export follow the work `stream` issued for the value.
pub(crate) fn publish_bound(
    parts: &Parts<'_>,
    batch: &mut MediaBatch,
    call: &Call,
    index: usize,
    write: &crate::ProductBuffer,
    host: bool,
    stream: impl Issue,
) -> Result<()> {
    let owners = parts.owners();
    owners.store.write(write, &write.view(), stream, None)?;
    batch.batch.calls[index]
        .forward
        .export(owners, call, &call.outputs[0], write, host, stream)
}
