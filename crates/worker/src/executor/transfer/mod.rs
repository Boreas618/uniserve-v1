//! The transfer runtime: tensor product transfers, cache exports and cache
//! installs (`CallKind::Transfer`).
//!
//! - **Tensor** (`TransferMode::Tensor`): each call copies its one input
//!   product (local, or imported from another rank or worker before the
//!   batch executed, see [`super::inputs`]) into its one output product and
//!   publishes the copy through the rank's export mechanisms, so other
//!   components, ranks and workers can import it ([`tensor`]). A call with
//!   a latent input and output publishes its request's committed trajectory
//!   version from the latent pool instead ([`latents`]).
//! - **Cache export** (`TransferMode::CacheExport`): each call publishes its
//!   request's visible cache extent ([`prefix_cache`]), which an image's latent
//!   preparation and denoising calls name as their conditioning.
//! - **Cache install** (`TransferMode::CacheInstall`): each call adopts the cache
//!   import its batch prepared into its slot's pages ([`prefix_cache`]).
//!
//! A batch is at most two jobs of its lane ([`forward`], [`complete`]) on
//! the shared batch skeleton ([`crate::executor::batch`]). The first exists
//! when a call has a host-gated predicate: it reads them, and a false one
//! skips its call, which only publishes a false completion flag. The last
//! job carries everything the batch issues: the start of its output lease,
//! the completion flags, the transfers with their exports, the readers'
//! fences and the lease's seal.
//!
//! The completion of that job is the batch's commit point. Exports are
//! located only there, so nothing of an export reaches another process
//! before the device work it describes is complete. Every owner's update is
//! then validated (products and their exports, cache exports and installs,
//! pending calls) and they become visible together. A failure before that
//! point retires everything the batch reserved, also from the job's
//! completion on: the job still carries what the batch recorded before it
//! failed (cache table installs and unit resets do their host bookkeeping
//! when recorded, so their device work follows), and an export is revoked
//! only after the copies into it ran. Of an export that failed itself the
//! job keeps only the event operations ([`crate::stream::Issue::attempt`]).
//! A failure after the commit point is fatal. Cache transfers run on the stream of the
//! component's text prefill, ordered with the text batches that read and
//! write the same pages; tensor transfers run on the stream serving them on
//! the component's device.

pub(crate) mod latents;
pub(crate) mod prefix_cache;
mod tensor;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{
    Batch as BatchPlan, BufferId, CallKind, ForwardStats, RequestKey, TransferMode,
};

use super::backend::{WorkerBackend, WorkerBatch};
use super::batch::{Batch, Failure, Launched, Opening, Stage, fail};
use super::image::ImageParts;
use super::inputs::{Inputs, component_device};
use super::products::Publications;
use crate::error::WorkerError;
use crate::executor::lifecycle::{LaneJobs, RequestStorage};
use crate::ffi::MemoryKind;
use crate::lane::{Completion, Job, Recorder};
use crate::lock;
use crate::request::RequestPool;
use crate::storage::{OutputPool, ProductBuffer, ProductExport};
use crate::stream::Streams;
use crate::worker::RankStorage;
use crate::{Error, Result};
use latents::PendingExport;
use prefix_cache::{Exporting, Installation, Published};

/// The transfer runtime of a rank: the lane streams its transfers and
/// imports run on, and its output storage.
pub(crate) struct TransferRuntime {
    /// The lane stream serving each call kind on each CUDA device.
    streams: HashMap<(CallKind, DeviceId), Stream>,
    outputs: OutputPool,
}

impl TransferRuntime {
    /// The runtime of a rank with `devices`, serving `kinds` on the streams
    /// of `lanes`, with `queue_depth` batches of at most `max_calls` calls
    /// in flight.
    pub(crate) fn start(
        streams: &Streams,
        devices: &[DeviceId],
        kinds: &[CallKind],
        queue_depth: usize,
        max_calls: usize,
    ) -> Result<Self> {
        let mut served = HashMap::new();
        for &device in devices {
            for &kind in kinds {
                if let Some(stream) = streams.stream_for(kind, device) {
                    served.insert((kind, device), stream.stream()?.clone());
                }
            }
        }
        Ok(Self {
            streams: served,
            // One completion word per call; transfers read nothing back.
            outputs: OutputPool::new(
                devices.first().copied(),
                queue_depth.max(1) * 2,
                max_calls.max(1),
            )?,
        })
    }

    /// The stream serving `kind` calls on `device`; none for host devices.
    pub(crate) fn stream(&self, kind: CallKind, device: MemoryKind) -> Result<Option<Stream>> {
        let MemoryKind::Cuda(index) = device else {
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

/// The step a transfer batch's failure is reported in.
const PHASE: &str = "transfer batch";

/// Whether `kind` transfers prefix cache, on the stream of the component's text
/// prefill.
fn transfers_cache(kind: CallKind) -> bool {
    kind != CallKind::Transfer(TransferMode::Tensor)
}

/// The device and stream the transfers of `plan`'s component run on.
fn placement(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    kind: CallKind,
) -> Result<(Option<DeviceId>, Option<Stream>)> {
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
        // Cache transfers read and write the pages text batches use, on their
        // stream.
        let text = backend
            .parts()
            .text
            .ok_or_else(|| Error::Unsupported("cache transfers require the text runtime".into()))?;
        let (device, stream, _) = text.placement(&first.component)?;
        Ok((device, stream))
    } else {
        let device = component_device(backend, &first.component)?;
        Ok((device.cuda(), backend.transfer_stream(kind, device)?))
    }
}

/// Bind the batch's outputs and record in `job` its order after the text
/// runtime's batches (cache transfers) and the reads of its predicates.
fn open(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    kind: CallKind,
    device: Option<DeviceId>,
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
            .ok_or_else(|| Error::Unsupported("cache transfers require the text runtime".into()))?;
        text.order().wait(job.on(stream))?;
    }
    Opening::open(
        parts.requests,
        &storage.tensors,
        parts.placement,
        parts.rank,
        plan,
        device,
        job.on(stream),
        |_| true,
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
    device: Option<DeviceId>,
    stream: Option<&Stream>,
    job: &Job,
) -> Pending {
    let cache = transfers_cache(kind);
    let mut parts = backend.parts();
    let (Some(runtime), Some(storage)) = (parts.transfer, parts.storage) else {
        let error = Error::Unsupported("transfers require rank storage".into());
        return Pending::Discarding(None, Failure::new(PHASE)(error));
    };
    // One completion word per call; transfers read nothing back.
    let calls = plan.calls.len();
    let leased = Batch::lease(
        opening,
        &mut runtime.outputs,
        calls,
        stream.cloned(),
        device,
        &storage.events,
        job,
    );
    let mut batch = match leased {
        Ok(leased) => TransferBatch {
            storage,
            requests: parts.requests,
            plan,
            job,
            stream,
            recorded: Recorded {
                batch: leased,
                products: VecDeque::new(),
                versions: VecDeque::new(),
                exports: VecDeque::new(),
                installations: Vec::new(),
            },
        },
        Err(error) => return Pending::Discarding(None, Failure::new(PHASE)(error)),
    };

    let result = (|| -> Result<()> {
        batch.bind_flags()?;
        if cache {
            let text = parts.text.as_deref().ok_or_else(|| {
                Error::Unsupported("cache transfers require the text runtime".into())
            })?;
            let component = &plan.calls[0].component;
            batch.reserve_cache(text.unit_column(component)?, inputs)?;
        }
        match kind {
            CallKind::Transfer(TransferMode::Tensor) => {
                batch.copy_products(parts.image.as_mut())?
            }
            CallKind::Transfer(TransferMode::CacheExport) => batch.export_cache()?,
            CallKind::Transfer(TransferMode::CacheInstall) => batch.install_cache(inputs)?,
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
        Err(error) => Pending::Discarding(None, Failure::new(PHASE)(error)),
        Ok((device, stream)) => match state {
            None => match open(backend, plan, kind, device, stream.as_ref(), &job) {
                Ok(opening) if opening.reads_predicates() => Pending::Predicates(opening),
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
                Err(error) => Pending::Discarding(None, Failure::new(PHASE)(error)),
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
        (Some(Pending::Predicates(_)), Err(error)) => Pending::Failed(Failure::new(PHASE)(error)),
        (Some(Pending::Predicates(mut opening)), Ok(mut completion)) => {
            match opening.predicates(&mut completion) {
                Ok(()) => Pending::Opened(opening),
                Err(error) => Pending::Failed(Failure::new(PHASE)(error)),
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
                Err(cleanup) => Failure::fatal(PHASE)(cleanup),
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

/// A transfer batch between two of its jobs.
pub(crate) enum Pending {
    /// The job on the lane reads the batch's host-gated predicates.
    Predicates(Opening),
    /// The predicates are known: the batch reserves, transfers and seals
    /// next.
    Opened(Opening),
    /// The job on the lane is the batch's last: its completion commits the
    /// batch.
    Committing(Box<Recorded>),
    /// The job on the lane carries what the failed batch recorded: its
    /// completion retires what the batch reserved, if anything.
    Discarding(Option<Box<Recorded>>, Failure),
    /// The batch failed and holds nothing: its calls are reported next.
    Failed(Failure),
}

/// What a batch keeps from the recording of its last job to that job's
/// completion: the batch, and the exports whose locations exist once the
/// job was performed.
pub(crate) struct Recorded {
    pub(super) batch: Batch,
    /// Product exports, each with its call and the write it publishes.
    pub(super) products: VecDeque<(usize, ProductBuffer, ProductExport)>,
    /// Trajectory versions exported from the latent pool, with their calls.
    pub(super) versions: VecDeque<(usize, PendingExport)>,
    /// Cache exports, and the installs resident once the batch commits.
    pub(super) exports: VecDeque<Exporting>,
    pub(super) installations: Vec<Installation>,
}

/// What a batch's exports became at its job's completion.
#[derive(Default)]
struct Located {
    /// Product exports, registered once the batch commits.
    publications: Publications,
    /// Trajectory versions, released from the latent pool if the batch
    /// fails before committing.
    versions: Vec<BufferId>,
    published: Vec<Published>,
}

/// A transfer batch recording its last job: its borrowed owners and what it
/// reserved.
pub(super) struct TransferBatch<'a> {
    pub(super) storage: &'a mut RankStorage,
    pub(super) requests: &'a mut RequestPool,
    pub(super) plan: &'a BatchPlan,
    /// The job everything the batch issues is recorded into, and the
    /// stream it issues on.
    job: &'a Job,
    stream: Option<&'a Stream>,
    pub(super) recorded: Recorded,
}

impl<'a> TransferBatch<'a> {
    /// Where the batch issues its stream work: its job.
    pub(super) fn issue(&self) -> Recorder<'a> {
        self.job.on(self.stream)
    }

    /// The device the batch's products live on.
    pub(super) fn target(&self) -> crate::ffi::MemoryKind {
        self.recorded.batch.target()
    }

    /// Indices of the calls that execute.
    pub(super) fn active(&self) -> Vec<usize> {
        self.recorded.batch.active()
    }

    /// Request slots of the batch's calls, for relayed outputs.
    pub(super) fn slots(&self) -> HashMap<RequestKey, usize> {
        self.recorded.batch.slots(self.plan)
    }

    /// Bind every call's completion flag; skipped calls publish false.
    fn bind_flags(&mut self) -> Result<()> {
        self.recorded
            .batch
            .bind_flags(self.plan, &self.storage.tensors, self.job)
    }

    /// Record the end of the batch's device work: its readers' fences, the
    /// completion flags of the executed calls and the seal of its lease.
    /// The updates that depend on no export's locations are validated here;
    /// the others at the job's completion ([`Recorded::commit`]).
    fn seal(&mut self) -> Result<()> {
        let batch = &mut self.recorded.batch;
        // A copy's read ends at the fence of the write it fed.
        let after: Vec<ProductBuffer> = batch
            .writes
            .iter()
            .filter(|write| {
                batch.calls.iter().all(|state| {
                    state
                        .completion_write
                        .as_ref()
                        .is_none_or(|flag| flag.id() != write.id())
                })
            })
            .cloned()
            .collect();
        batch.end_reads(&self.storage.tensors, self.job, &after)?;
        batch.set_flags(&self.storage.tensors, self.job)?;
        self.storage.tensors.validate_writes(&batch.writes)?;
        batch.seal(&self.storage.events, self.job)?;
        self.requests
            .validate_pending(&Batch::<()>::pending(self.plan))
    }

    /// End the recording after `result`. A failed batch still records what
    /// ends its device work (the fences of its reads, the seal of its
    /// lease), so the storage they guard returns once its job ran.
    fn end(self, result: Result<()>) -> (Recorded, Option<Failure>) {
        let Err(error) = result else {
            return (self.recorded, None);
        };
        let mut recorded = self.recorded;
        let batch = &mut recorded.batch;
        let completed = batch.end_reads(&self.storage.tensors, self.job, &[]);
        let sealed = if lock(&batch.buffer).sealed() {
            Ok(())
        } else {
            batch.seal(&self.storage.events, self.job)
        };
        let failure = match completed.and(sealed) {
            Ok(()) => Failure::new(PHASE)(error),
            // Storage the cleanup could not fence stays reserved; the rank
            // cannot reuse it safely.
            Err(cleanup) => Failure::fatal(PHASE)(cleanup),
        };
        (recorded, Some(failure))
    }
}

impl Recorded {
    /// Apply the completion of the batch's last job: locate its exports,
    /// validate every owner's update, then commit them together.
    fn commit(
        mut self,
        storage: &mut RankStorage,
        requests: &mut RequestPool,
        mut image: Option<&mut ImageParts<'_>>,
        plan: &BatchPlan,
        completed: Result<()>,
    ) -> std::result::Result<Launched, Failure> {
        let mut located = Located::default();
        let accepted = completed
            .and_then(|()| self.locate(storage, image.as_deref_mut(), &mut located))
            .and_then(|()| self.validate(storage, plan, &located));
        if let Err(error) = accepted {
            return Err(match self.retire(storage, image, located) {
                Ok(()) => Failure::new(PHASE)(error),
                // Storage the cleanup could not retire stays reserved; the
                // rank cannot reuse it safely.
                Err(cleanup) => Failure::fatal(PHASE)(cleanup),
            });
        }

        // Every owner accepted its update. Any failure from here is fatal.
        let fatal = Failure::fatal(PHASE);
        let Located {
            mut publications,
            published,
            ..
        } = located;
        self.batch.commit(plan, &storage.tensors, requests)?;
        publications.register(&storage.tensors).map_err(&fatal)?;
        prefix_cache::apply(storage, published, self.installations).map_err(&fatal)?;

        let mut stats = ForwardStats::default();
        stats.component_us.insert(
            "transfer_execute".into(),
            self.batch.started.elapsed().as_micros() as u64,
        );
        Ok(self.batch.launch(stats, "transfer_materialize", None))
    }

    /// Retire every reservation of a batch that failed while it recorded
    /// its job, once that job completed.
    fn discard(self, storage: &mut RankStorage, image: Option<&mut ImageParts<'_>>) -> Result<()> {
        self.retire(storage, image, Located::default())
    }

    /// The locations of the batch's exports, now that their device work is
    /// complete: each joins its call's completion.
    fn locate(
        &mut self,
        storage: &mut RankStorage,
        image: Option<&mut ImageParts<'_>>,
        located: &mut Located,
    ) -> Result<()> {
        let calls = &mut self.batch.calls;
        while let Some((index, write, export)) = self.products.pop_front() {
            let (export, locations) = storage.tensors.located(&write, export)?;
            located
                .publications
                .accept(write.id(), locations, export, &mut calls[index].output)?;
        }

        if !self.versions.is_empty() {
            let (pool, _) = image
                .ok_or_else(|| {
                    Error::Invariant("exported trajectory versions lost their latent pool".into())
                })?
                .latent_pool()?;
            while let Some((index, version)) = self.versions.pop_front() {
                // The version stays the pool's until the batch commits.
                located.versions.push(version.buffer());
                latents::locate_export(
                    pool,
                    version,
                    &mut located.publications,
                    &mut calls[index].output,
                )?;
            }
        }

        while let Some(export) = self.exports.pop_front() {
            let export = export.locate(storage)?;
            calls[export.call].output.output.cache_output = Some(export.transfer.clone());
            located.published.push(export);
        }
        Ok(())
    }

    /// Check, before the commit point, the updates that name the located
    /// exports: the calls' outputs, the publications and the cache
    /// transfers.
    fn validate(
        &self,
        storage: &mut RankStorage,
        plan: &BatchPlan,
        located: &Located,
    ) -> Result<()> {
        for (state, call) in self.batch.calls.iter().zip(&plan.calls) {
            state.output.validate_output(call)?;
        }
        located.publications.validate(&storage.tensors)?;
        prefix_cache::validate(storage, &located.published, &self.installations)
    }

    /// Retire every reservation after a failure before the commit point:
    /// the exports located so far and those that were not, the versions
    /// exported from `image`'s latent pool, the batch's writes and its
    /// lease. The batch's job completed, so its fences are recorded and
    /// storage returns once they complete.
    fn retire(
        mut self,
        storage: &mut RankStorage,
        image: Option<&mut ImageParts<'_>>,
        located: Located,
    ) -> Result<()> {
        let Located {
            mut publications,
            mut versions,
            published,
        } = located;
        let revoked = publications.revoke();
        for (_, _, export) in self.products.drain(..) {
            export.revoke();
        }

        // A version's locations were revoked with the publications or are
        // revoked here; the pool forgets the export either way.
        for (_, version) in self.versions.drain(..) {
            versions.push(version.buffer());
            version.revoke();
        }
        let released = match (versions.is_empty(), image) {
            (true, _) => Ok(()),
            (false, Some(image)) => image
                .latent_pool()
                .map(|(pool, _)| pool.release_exports(&versions)),
            (false, None) => Err(Error::Invariant(
                "exported trajectory versions lost their latent pool".into(),
            )),
        };

        for export in &published {
            self.batch.calls[export.call].output.output.cache_output = None;
        }
        let mut unpublished = prefix_cache::revoke(storage, published);
        for export in self.exports.drain(..) {
            unpublished = unpublished.and(export.revoke(storage));
        }

        let writes = std::mem::take(&mut self.batch.writes);
        let abandoned = storage.tensors.abandon_writes(&writes);
        lock(&self.batch.buffer).abandon();
        revoked.and(released).and(unpublished).and(abandoned)
    }
}
