//! A video model's media batch: what its calls contribute beside the shared
//! batch skeleton ([`VideoCall`], [`MediaBatch`]), what a video model's rank
//! keeps beside the executor ([`VideoRuntime`]: the trajectory ledger, the
//! call kinds' streams and the output leases), and the jobs of a batch
//! ([`forward`], [`complete`]).
//!
//! The executor (`uniserve.executor`) computes every call; the worker leases
//! what the calls read, binds what they write, keeps the trajectory ledger,
//! publishes the products and delivers the results. A media batch goes
//! through the steps of [`crate::executor::batch`] as jobs of its lane:
//!
//! 1. The opening job reads the batch's host-gated predicates, when it has
//!    any. A call whose predicate is false is skipped: it runs nothing and
//!    forwards a false decision to its successor through its completion
//!    flag.
//! 2. The plan job asks the executor what each call reads and writes
//!    (`plan`): which inputs, and every output's shape, region and memory,
//!    whether a host task fills it after the batch commits and how it is
//!    published.
//! 3. From its completion the last job is recorded: the batch leases its
//!    output storage on the stream of its kind, binds the completion flags,
//!    leases the reads and binds the writes, takes each trajectory's banks
//!    from the ledger, and records the executor's `execute` on those
//!    tensors; behind it the producer fences and exports of what the calls
//!    wrote, and [`MediaBatch::issue_commit`] (the end of the input leases,
//!    the executed calls' flags, the seal of the lease) after every owner
//!    validated its update.
//! 4. From that job's completion, [`MediaBatch::apply_commit`] publishes the
//!    exports, applies the trajectory commits to the ledger, makes the writes
//!    visible and the calls pending: the commit point. A failure before it
//!    retires everything the batch reserved in a job behind what it
//!    recorded, and the batch's calls fail; a failure at or past it is fatal.
//! 5. The launched batch materializes once its output fences completed and
//!    the executor reported its host tasks ([`HostTasks`]): what they wrote
//!    is then published, after the commit point, and the output rank
//!    delivers a finished MP4.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::{Arc, Mutex};

use cuda::Stream;
use ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, Call, CallKind, CallStatus, ErrorCode,
    ForwardStats, LatentParams, MediaCall, TensorRef, TransferTransport, WorkerInfo,
};
use ipc::{CallId, MediaSource};
use serde::Deserialize;

use crate::config::Device;
use crate::config::placement::GroupPlacement;
use crate::cuda_graph::GraphMode;
use crate::error::WorkerError;
use crate::executor::PendingOutput;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::batch::{Batch, Failure, HostStage, Launched, Opening, Stage, fail};
use crate::executor::inputs::Inputs;
use crate::executor::lifecycle::{AdmittedSlot, LaneJobs, RequestStorage};
use crate::executor::products::{
    OutputShape, Placement, ProductOwners, Publications, bind_outputs, export, supplied,
};
use crate::executor::python::{DONE, Inbox, PythonExecutor, Report};
use crate::ffi::{Backing, DType, MemoryKind};
use crate::lane::{Completion, Job};
use crate::memory::ProductExport;
use crate::memory::allocation::{Storage, StorageView};
use crate::request::RequestPool;
use crate::stream::{CUDAStream, Issue};
use crate::transfer::{SharedMapping, SharedRead, TransportKind};
use crate::worker::WorkerMemory;
use crate::{
    Error, LatentPool, LatentStorage, LatentUpdate, OutputPool, ProductBuffer, ProductRead,
    ProductStore, Result, lock,
};

/// The step a media batch's failure is reported in.
const PHASE: &str = "media batch";

/// The component timer name of a media batch's materialization.
const MATERIALIZE: &str = "video_materialize";

/// The trajectory ledger of a denoising rank: the latent pool whose two
/// banks of pages hold every request's samples, which bank holds a
/// request's committed step, and the storage of the banks. A request slot
/// owns `pages` consecutive pages after the padding page 0.
pub(crate) struct LatentLedger {
    pool: LatentPool,
    storage: LatentStorage,
    /// Elements of one page.
    page_units: u64,
    /// Pages a request slot owns.
    pages: u64,
}

impl LatentLedger {
    pub(crate) fn new(
        pool: LatentPool,
        storage: LatentStorage,
        page_units: u64,
        pages: u64,
    ) -> Self {
        Self {
            pool,
            storage,
            page_units,
            pages,
        }
    }

    /// The two banks, `[pages, page_units, 1]` float32 each, which the
    /// executor gathers from and scatters to.
    pub(crate) fn banks(&self) -> Result<[StorageView; 2]> {
        Ok([self.storage.bank(0)?, self.storage.bank(1)?])
    }

    /// The pages, latent units and raster of `slot`'s trajectory: every
    /// slot owns the same run of consecutive pages after the padding page,
    /// so the rank names them itself; the raster is the request's admitted
    /// canvas.
    fn params(&self, slot: usize, call: &Call, canvas: (u32, u32)) -> LatentParams {
        let first = 1 + (slot as u64 - 1) * self.pages;
        LatentParams {
            request_key: call.request_key,
            call_id: call.call_id,
            page_table: (first..first + self.pages)
                .map(|page| page as u32)
                .collect(),
            latent_units: (self.pages * self.page_units) as u32,
            width: canvas.0,
            height: canvas.1,
        }
    }

    fn slot_pages(&self, params: &LatentParams) -> Result<Vec<usize>> {
        let pages: Vec<i64> = params
            .page_table
            .iter()
            .map(|&page| i64::from(page))
            .collect();
        self.pool
            .validate_pages(&pages, i64::from(params.latent_units))
    }

    /// The bank a latent preparation initializes `slot`'s trajectory in, and
    /// its commit.
    fn initial(
        &self,
        slot: usize,
        call: &Call,
        canvas: (u32, u32),
    ) -> Result<(Banks, LatentUpdate)> {
        let params = self.params(slot, call, canvas);
        let pages = self.slot_pages(&params)?;
        self.pool.require_initial(slot, &pages)?;
        let update = LatentUpdate::for_call(slot, call, &params, 0)?;
        Ok((
            Banks {
                read: 1,
                write: 1,
                step: 0,
            },
            update,
        ))
    }

    /// The banks a denoising call reads and writes `slot`'s trajectory in,
    /// the step it evaluates, and its commit: it advances the step the pool
    /// committed by one.
    fn step(&self, slot: usize, call: &Call, canvas: (u32, u32)) -> Result<(Banks, LatentUpdate)> {
        let params = self.params(slot, call, canvas);
        let pages = self.slot_pages(&params)?;
        let step = self.pool.step_index(slot);
        let trajectory = crate::memory::TrajectoryVersion {
            step_index: step,
            generation: step + 1,
            units: i64::from(params.latent_units),
            height: i64::from(params.height),
            width: i64::from(params.width),
        };
        let read = self.pool.current_bank(slot, &trajectory, &pages)?;
        let write = 1 - read;
        self.pool.require_writable(write, &pages)?;
        let update = LatentUpdate::for_call(slot, call, &params, step)?;
        Ok((Banks { read, write, step }, update))
    }

    /// Validate trajectory commits before any becomes visible.
    fn validate(&self, updates: &[LatentUpdate]) -> Result<()> {
        self.pool.validate_updates(updates)
    }

    /// Make validated commits visible.
    fn apply(&mut self, updates: &[LatentUpdate]) {
        drop(self.pool.apply_updates(updates));
    }

    /// Release retired requests' pages.
    fn release_slots(&mut self, slots: &[usize]) -> Result<()> {
        let slots: Vec<i64> = slots.iter().map(|&slot| slot as i64).collect();
        drop(self.pool.release_slots(&slots)?);
        self.pool.reap();
        Ok(())
    }
}

/// The banks a trajectory call reads and writes, and the step it evaluates.
#[derive(Clone, Copy, Debug)]
struct Banks {
    read: u8,
    write: u8,
    step: i64,
}

/// What serves a video model on this rank beside the executor's runners.
pub(crate) struct VideoRuntime {
    pub(crate) executor: PythonExecutor,
    /// The trajectory ledger of a denoising rank.
    pub(crate) ledger: Option<LatentLedger>,
    /// The device and stream of each media call kind this rank serves.
    streams: HashMap<MediaCall, (Device, Option<Stream>)>,
    /// On a CUDA rank holding host components, the stream device products
    /// host tasks wrote are fenced on: the tasks waited for their copies,
    /// so the fences complete at once.
    host_stream: Option<Arc<CUDAStream>>,
    /// Each admitted request's canvas, by slot, which its trajectory's
    /// raster is.
    canvases: HashMap<usize, (u32, u32)>,
    /// Started requests the executor has not admitted yet, as JSON.
    started: Vec<serde_json::Value>,
}

impl VideoRuntime {
    pub(crate) fn new(
        executor: PythonExecutor,
        ledger: Option<LatentLedger>,
        streams: HashMap<MediaCall, (Device, Option<Stream>)>,
        host_stream: Option<Arc<CUDAStream>>,
    ) -> Self {
        Self {
            executor,
            ledger,
            streams,
            host_stream,
            canvases: HashMap::new(),
            started: Vec::new(),
        }
    }

    /// Hold a started request for the executor's admission, which the batch
    /// that started it asks for next ([`Self::take_started`]).
    pub(crate) fn start(&mut self, request: &ipc::NewRequest) -> Result<()> {
        if let Some(media) = &request.diffusion
            && request.video.is_some()
        {
            self.canvases.insert(
                request.request_pool_idx as usize,
                (media.width, media.height),
            );
        }
        self.started.push(
            serde_json::to_value(request)
                .map_err(|error| Error::Invalid(format!("encoding a started request: {error}")))?,
        );
        Ok(())
    }

    /// The started requests no batch asked the executor to admit yet, as the
    /// call that admits them; `None` when there is none.
    pub(crate) fn take_started(&mut self) -> Option<crate::runner::methods::MethodCall> {
        if self.started.is_empty() {
            return None;
        }
        let requests = serde_json::Value::Array(std::mem::take(&mut self.started));
        Some(self.executor.start(requests.to_string()))
    }

    /// Whether the executor's admission works on the denoiser's stream (a
    /// transient layout's preparation), so its job belongs on that lane.
    pub(crate) fn denoising(&self) -> bool {
        self.ledger.is_some()
    }

    /// The device and stream of `kind`: none on the host.
    fn stream(&self, kind: MediaCall) -> (Device, Option<Stream>) {
        self.streams
            .get(&kind)
            .cloned()
            .unwrap_or((Device::Cpu, None))
    }

    pub(crate) fn set_wake(&self, wake: Option<ipc::Wake>) {
        self.executor.set_wake(wake);
    }
}

impl RequestStorage for VideoRuntime {
    fn admit_slots(&mut self, _admitted: &[AdmittedSlot<'_>], _jobs: &LaneJobs) -> Result<()> {
        // A request's state is created when the executor admits it.
        Ok(())
    }

    fn release_buffers(&mut self, _buffers: &HashSet<BufferId>, _jobs: &LaneJobs) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &HashSet<ipc::RequestKey>,
        _retained: &HashSet<BufferId>,
        _jobs: &LaneJobs,
    ) -> Result<()> {
        Ok(())
    }

    fn retirement_ready(
        &mut self,
        _buffers: &HashSet<BufferId>,
        _requests: &HashSet<ipc::RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        Ok(true)
    }

    fn retire_slots(&mut self, slots: &[usize], _jobs: &LaneJobs) -> Result<()> {
        for slot in slots {
            self.canvases.remove(slot);
        }
        if let Some(ledger) = &mut self.ledger {
            ledger.release_slots(slots)?;
        }
        self.executor.release(slots);
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        Ok(())
    }

    fn close(&mut self, aborted: bool) -> Result<()> {
        self.started.clear();
        self.canvases.clear();
        self.executor.close(aborted)
    }
}

/// How a call publishes what its host tasks wrote.
pub(crate) enum Publication {
    /// Products the task wrote whole: host products, or device products the
    /// task copied there (fenced on `stream`), exported over the device
    /// transport.
    Products {
        written: Vec<(TensorRef, ProductBuffer)>,
        stream: Option<Arc<CUDAStream>>,
    },
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
    /// ledger at the commit point.
    pub(crate) latent: Option<LatentUpdate>,
    /// The exports the call's forward issued, located when the batch
    /// commits: their device work is performed by then.
    exports: Vec<(ProductBuffer, ProductExport)>,
    /// The reports the executor's host tasks of the call send.
    reports: usize,
    /// How the call publishes what its tasks wrote once they finished.
    publication: Option<Publication>,
    /// Inputs read in place from their shared-memory exports, released once
    /// the call's host tasks finished.
    borrowed: Vec<Arc<Mutex<SharedRead>>>,
}

impl VideoCall {
    /// Issue the export of `write`, the call's output `product`, to
    /// `stream`, behind the work that produced it.
    fn export(
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
    /// Count one forward of the batch by the graph mode it ran in.
    fn record(&mut self, mode: GraphMode) {
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
    fn target(&self) -> MemoryKind {
        self.batch.target()
    }

    /// Lease one input for `consumer`, whole, on the batch's device; what
    /// `stream` issues next waits on its writer. The lease ends with the
    /// batch's other reads, when it commits or retires.
    fn read(
        &mut self,
        store: &ProductStore,
        product: &TensorRef,
        consumer: CallId,
        memory: MemoryKind,
        stream: impl Issue,
    ) -> Result<Arc<ProductRead>> {
        let read = store.consume(product, consumer, Some(memory), stream)?;
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
        storage: &WorkerMemory,
        requests: &RequestPool,
        ledger: Option<&LatentLedger>,
        job: &Job,
    ) -> std::result::Result<(), Failure> {
        let failed = Failure::new(PHASE);
        let tensors = &storage.tensors;
        self.batch.end_reads(tensors, job, &[]).map_err(&failed)?;
        self.batch.set_flags(tensors, job).map_err(&failed)?;
        self.batch.validate(plan, tensors).map_err(&failed)?;
        let updates = self.updates();
        match ledger {
            Some(ledger) => ledger.validate(&updates),
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
    /// the ledger, register the exports, make the writes visible and record
    /// the calls as pending. A failure after the exports' check is fatal.
    fn apply_commit(
        &mut self,
        plan: &BatchPlan,
        tensors: &ProductStore,
        requests: &mut RequestPool,
        ledger: Option<&mut LatentLedger>,
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
        if let Some(ledger) = ledger {
            ledger.apply(&self.updates());
        }
        for state in &mut self.batch.calls {
            if let Some(update) = state.forward.latent.take() {
                state.output.apply_latent_update(&update).map_err(&fatal)?;
            }
        }
        self.publications.register(tensors).map_err(&fatal)?;
        self.batch.commit(plan, tensors, requests)
    }

    /// Retire every reservation after a failure before the commit point,
    /// recording in `job` the fences that end them: revoke the exports, end
    /// the reads, abandon the writes and seal and abandon the lease.
    fn discard(&mut self, storage: &WorkerMemory, job: &Job) -> Result<()> {
        for state in &mut self.batch.calls {
            state
                .forward
                .exports
                .drain(..)
                .for_each(|(_, export)| export.revoke());
            release_borrowed(&state.forward.borrowed);
        }
        let revoked = self.publications.revoke();
        let discarded = self.batch.discard(&storage.tensors, &storage.events, job);
        revoked.and(discarded)
    }

    /// Hand the committed batch to materialization, with the host tasks its
    /// calls' results wait for.
    fn launch(self, inbox: Inbox, batch_id: u64) -> Launched {
        let Self {
            mut batch, stats, ..
        } = self;
        let calls: Vec<HostCall> = batch
            .calls
            .iter_mut()
            .enumerate()
            .filter(|(_, state)| state.forward.reports > 0)
            .map(|(index, state)| HostCall {
                index,
                reports: std::mem::take(&mut state.forward.reports),
                publication: state.forward.publication.take(),
                borrowed: std::mem::take(&mut state.forward.borrowed),
                received: None,
            })
            .collect();
        let host = (!calls.is_empty()).then(|| {
            Box::new(HostTasks {
                calls,
                inbox,
                batch: batch_id,
            }) as Box<dyn HostStage>
        });
        batch.launch(stats, MATERIALIZE, host)
    }
}

/// Release borrowed shared-memory reads: their producer may reuse the rows.
fn release_borrowed(borrowed: &[Arc<Mutex<SharedRead>>]) {
    for read in borrowed {
        lock(read).release();
    }
}

/// One call's host tasks in a launched batch, with how the call publishes
/// what they wrote.
struct HostCall {
    /// The call's index in the batch.
    index: usize,
    /// The reports the executor sends for it.
    reports: usize,
    publication: Option<Publication>,
    borrowed: Vec<Arc<Mutex<SharedRead>>>,
    /// The reports, once all arrived.
    received: Option<Vec<Report>>,
}

/// The host tasks a launched media batch's results wait for. What they
/// write is published once every task reported, after the commit point:
/// such a product is committed and its exports registered at once.
struct HostTasks {
    calls: Vec<HostCall>,
    inbox: Inbox,
    batch: u64,
}

impl HostStage for HostTasks {
    fn poll(&mut self) -> Result<bool> {
        let mut done = true;
        for call in &mut self.calls {
            if call.received.is_none() {
                call.received = self.inbox.take(self.batch, call.index, call.reports);
                if call.received.is_some() {
                    release_borrowed(&call.borrowed);
                }
            }
            done &= call.received.is_some();
        }
        Ok(done)
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
        // The executor's host thread finishes what it started; its reports
        // are dropped, and borrowed rows are released when they arrive or
        // the inbox closes.
        self.inbox.discard(self.batch);
        for call in &self.calls {
            release_borrowed(&call.borrowed);
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
        let reports = self.received.unwrap_or_default();
        if reports.iter().any(|report| report.status != DONE) {
            tracing::error!("a host media call failed; the executor logged why");
            output.output.status = CallStatus::Error;
            output.output.error_code = Some(ErrorCode::ComputeError);
            let writes: Vec<ProductBuffer> = match self.publication {
                Some(Publication::Products { written, .. }) => {
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
            Some(Publication::Products { written, stream }) => {
                for (product, write) in written {
                    let view = write.view();
                    // A device product's write and export record their
                    // fences on the host stream; its copies drained already.
                    let device = matches!(view.device(), MemoryKind::Cuda(_));
                    let fence = if device {
                        stream.as_deref().map(CUDAStream::stream).transpose()?
                    } else {
                        None
                    };
                    owners.store.write(&write, &view, fence, None)?;
                    owners.store.commit_writes(std::slice::from_ref(&write))?;
                    let (exported, locations) =
                        export(owners, call, &product, &write, fence.is_none(), None, fence)?;
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
                for (local, &row) in rows.iter().enumerate() {
                    let report = reports
                        .iter()
                        .find(|report| report.row == local as i64)
                        .ok_or_else(|| {
                            Error::Invariant("a video encoding task reported no unit".into())
                        })?;
                    let length = u64::try_from(report.value).unwrap_or(u64::MAX);
                    if length > width {
                        return Err(Error::Invariant("a framed unit exceeds its row".into()));
                    }
                    regions.push(vec![row..row + 1, 0..length]);
                }
                let (exported, locations) =
                    export(owners, call, &product, &write, true, Some(&regions), None)?;
                publications.accept(write.id(), locations, exported, output)?;
                publications.register(owners.store)?;
            }
            Some(Publication::Media) if output.reports_output => {
                for report in &reports {
                    if !report.payload.is_empty() {
                        let source = MediaSource::publish(&report.payload).map_err(|error| {
                            Error::transport_source("publishing media failed", error)
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
    /// The predicates are known: the batch asks the executor for its plan
    /// next.
    Opened(Opening<VideoCall>),
    /// The job on the lane asks the executor what the batch's calls read
    /// and write; the batch goes on from its completion.
    Planning(Opening<VideoCall>, serde_json::Value),
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
    pub(crate) storage: &'a WorkerMemory,
    pub(crate) info: &'a WorkerInfo,
    pub(crate) placement: &'a GroupPlacement,
    pub(crate) rank: usize,
    pub(crate) video: &'a mut VideoRuntime,
    /// The output leases of media batches.
    pub(crate) outputs: &'a mut OutputPool,
}

impl<'a> Parts<'a> {
    /// The product owners, borrowed for the parts' lifetime rather than
    /// from the parts themselves.
    fn owners(&self) -> ProductOwners<'a> {
        let storage: &'a WorkerMemory = self.storage;
        ProductOwners {
            store: &storage.tensors,
            transports: &storage.transports,
        }
    }
}

/// The name of a call's output product.
fn product_name<'a>(info: &'a WorkerInfo, call: &Call, output: &TensorRef) -> Result<&'a str> {
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
    match call.kind {
        CallKind::Media(kind) => kind,
        _ => MediaCall::Denoising,
    }
}

/// One output a call writes, as the executor planned it.
#[derive(Debug, Deserialize)]
struct PlannedWrite {
    output: usize,
    shape: Vec<u64>,
    region: Option<Vec<[u64; 2]>>,
    /// `host` or `device`.
    memory: String,
    /// A host task fills it after the batch commits.
    deferred: bool,
    /// Host readers read its export.
    host_export: bool,
}

/// What one call reads and writes, as the executor planned it.
#[derive(Debug, Deserialize)]
struct Planned {
    /// Indices into the call's inputs.
    reads: Vec<usize>,
    writes: Vec<PlannedWrite>,
    /// `products`, `encoded` or `media`.
    publication: Option<String>,
    /// For a call reading rows of a round's decoded frames: the round
    /// position each read takes, read in place from its shared-memory
    /// export or leased.
    #[serde(default)]
    read_rows: Option<Vec<u32>>,
    /// Each input is a round's framed rows, read from the exports that
    /// reach this rank.
    #[serde(default)]
    exported_rows: bool,
    /// The product rows an encoding call's tasks frame into.
    #[serde(default)]
    rows: Vec<u64>,
    /// The reports the call's host tasks send.
    #[serde(default)]
    reports: usize,
}

/// The JSON step the executor plans and executes: every call with what the
/// worker knows of it.
fn step_json(
    plan: &BatchPlan,
    info: &WorkerInfo,
    calls: &[crate::executor::batch::CallState<VideoCall>],
) -> Result<serde_json::Value> {
    let encode = |error: serde_json::Error| Error::Invalid(format!("encoding a batch: {error}"));
    let mut entries = Vec::with_capacity(plan.calls.len());
    for (index, (call, state)) in plan.calls.iter().zip(calls).enumerate() {
        let range = plan
            .decode_ranges
            .iter()
            .find(|range| range.request_key == call.request_key && range.call_id == call.call_id);
        let names = call
            .outputs
            .iter()
            .map(|output| product_name(info, call, output).map(str::to_owned))
            .collect::<Result<Vec<_>>>()?;
        entries.push(serde_json::json!({
            "index": index,
            "slot": state.slot,
            "active": state.active,
            "refused": state.refused(),
            "call": serde_json::to_value(call).map_err(encode)?,
            "range": serde_json::to_value(range).map_err(encode)?,
            "output_names": names,
            "latent": serde_json::Value::Null,
        }));
    }
    Ok(serde_json::json!({
        "batch_id": plan.batch_id,
        "calls": entries,
    }))
}

/// The next job of a media batch: the read of its host-gated predicates
/// when it has any, the executor's plan, then its forwards with the issued
/// commit, and after a failure the retirement of what it reserved. `None`
/// when the batch failed without needing the lane again.
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
    let failed = Failure::new(PHASE);
    let Some(first) = plan.calls.first() else {
        return Ok(Step::Failed(failed(Error::Invariant(
            "a video batch has calls".into(),
        ))));
    };
    let (device, stream) = parts.video.stream(media_kind(first));
    let device = device.cuda();
    let (opening, planned) = match state {
        None => {
            // The batch's first job binds its calls and reads its
            // host-gated predicates: they decide which calls run.
            let job = Job::new();
            let opened = Opening::open(
                parts.requests,
                &parts.storage.tensors,
                parts.placement,
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
                Ok(opening) => return Ok(ask_plan(parts, plan, opening, job)),
                Err(error) => return Ok(ended(job, failed(error))),
            }
        }
        Some(Pending::Opened(opening)) => return Ok(ask_plan(parts, plan, opening, Job::new())),
        Some(Pending::Planning(opening, planned)) => (opening, planned),
        // What the batch recorded was performed: retire what it reserved,
        // behind that work.
        Some(Pending::Failing(media, failure)) => {
            return Ok(retire(parts.storage, *media, Job::new(), failure));
        }
        Some(Pending::Predicates(_) | Pending::Committing(_) | Pending::Discarding(_)) => {
            return Err(WorkerError::classify(
                Error::Invariant(
                    "a media batch's job was prepared before its last completed".into(),
                ),
                Some(PHASE),
            ));
        }
    };

    let job = Job::new();
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
        .and_then(|()| record_forwards(parts, plan, &mut media, planned, &job))
        .map_err(&failed)
        .and_then(|()| {
            media.issue_commit(
                plan,
                parts.storage,
                parts.requests,
                parts.video.ledger.as_ref(),
                &job,
            )
        });
    Ok(match recorded {
        Ok(()) => Step::Job(job, Pending::Committing(Box::new(media))),
        // Nothing was recorded that the retirement must follow.
        Err(failure) if job.is_empty() => retire(parts.storage, media, job, failure),
        // What the batch recorded before it failed is performed before its
        // reservations are retired: the events it queued are due, and its
        // writes are abandoned only behind its work.
        Err(failure) => Step::Job(job, Pending::Failing(Box::new(media), failure)),
    })
}

/// Record into `job` the executor's plan of the opened batch: what each
/// call reads and writes.
fn ask_plan(parts: &Parts<'_>, plan: &BatchPlan, opening: Opening<VideoCall>, job: Job) -> Step {
    match step_json(plan, parts.info, opening.calls()) {
        Ok(step) => {
            job.call_texts(parts.video.executor.plan(step.to_string()));
            Step::Job(job, Pending::Planning(opening, step))
        }
        Err(error) => ended(job, Failure::new(PHASE)(error)),
    }
}

/// Lease what every call reads, bind what it writes, take each trajectory's
/// banks from the ledger and record into `job` the executor's execution of
/// the batch on those tensors, then the producer fences and exports of
/// what the calls wrote.
fn record_forwards(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    media: &mut MediaBatch,
    mut step: serde_json::Value,
    job: &Job,
) -> Result<()> {
    let planned_json = step
        .get("planned")
        .cloned()
        .ok_or_else(|| Error::Invariant("a media batch has no plan".into()))?;
    let planned: Vec<Planned> = serde_json::from_value(planned_json)
        .map_err(|error| Error::Interface(format!("the executor's plan: {error}")))?;
    if planned.len() != plan.calls.len() {
        return Err(Error::Interface(
            "the executor planned another number of calls".into(),
        ));
    }
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
    let owners = parts.owners();
    let stream = media.batch.stream.clone();
    let issue = job.on(stream.as_ref());
    let mut tensors = Vec::new();
    // Each non-deferred write, published behind the execution.
    let mut published: Vec<(usize, TensorRef, ProductBuffer, bool)> = Vec::new();
    for (index, (call, planned)) in plan.calls.iter().zip(&planned).enumerate() {
        let state = &media.batch.calls[index];
        if state.refused() {
            continue;
        }
        let slot = state.slot;

        // Reads.
        for (position, &input_index) in planned.reads.iter().enumerate() {
            let input = call.inputs.get(input_index).ok_or_else(|| {
                Error::Interface("the executor planned a read of an undeclared input".into())
            })?;
            if planned.exported_rows {
                tensors.push(exported_rows(parts, plan, input)?);
                continue;
            }
            if let Some(rows) = &planned.read_rows {
                let row = *rows
                    .get(position)
                    .ok_or_else(|| Error::Interface("a planned row read names no row".into()))?;
                if media.borrowed(input) {
                    let read = borrow_row(parts, plan, input, u64::from(row))?;
                    tensors.push(row_view(&read)?);
                    media.batch.calls[index]
                        .forward
                        .borrowed
                        .push(Arc::new(Mutex::new(read)));
                } else {
                    let read =
                        media.read(owners.store, input, call.call_id, MemoryKind::Host, None)?;
                    tensors.push(read.value().select(0, i64::from(row))?);
                }
                continue;
            }
            let memory = if host_kind(media_kind(call)) {
                MemoryKind::Host
            } else {
                media.target()
            };
            let read = media.read(owners.store, input, call.call_id, memory, issue)?;
            tensors.push(read.value().clone());
        }

        // Writes.
        let mut deferred = Vec::new();
        for write in &planned.writes {
            let product = call.outputs.get(write.output).ok_or_else(|| {
                Error::Interface("the executor planned an undeclared output".into())
            })?;
            let memory = match write.memory.as_str() {
                "host" => MemoryKind::Host,
                "device" => media.target(),
                other => {
                    return Err(Error::Interface(format!(
                        "the executor planned an output in {other:?} memory"
                    )));
                }
            };
            let bound = bind_outputs(
                owners.store,
                &[OutputShape {
                    product: product.clone(),
                    shape: write.shape.clone(),
                    region: write
                        .region
                        .as_ref()
                        .map(|region| region.iter().map(|&[a, b]| a..b).collect()),
                }],
                memory,
                &placement,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invariant("a planned output was not bound".into()))?;
            tensors.push(bound.view());
            if write.deferred {
                owners.store.defer_write(&bound)?;
                deferred.push((product.clone(), bound));
            } else {
                media.batch.writes.push(bound.clone());
                published.push((index, product.clone(), bound, write.host_export));
            }
        }
        let forward = &mut media.batch.calls[index].forward;
        forward.reports = planned.reports;
        forward.publication = match planned.publication.as_deref() {
            Some("products") => Some(Publication::Products {
                written: deferred,
                stream: parts.video.host_stream.clone(),
            }),
            Some("encoded") => {
                let (product, write) = deferred.into_iter().next().ok_or_else(|| {
                    Error::Interface("an encoding call planned no encoded rows".into())
                })?;
                Some(Publication::Encoded {
                    product,
                    write,
                    rows: planned.rows.clone(),
                })
            }
            Some("media") => Some(Publication::Media),
            None => None,
            Some(other) => {
                return Err(Error::Interface(format!(
                    "the executor planned the unknown publication {other:?}"
                )));
            }
        };

        // The trajectory ledger.
        let kind = media_kind(call);
        if matches!(kind, MediaCall::LatentPreparation | MediaCall::Denoising)
            && media.batch.calls[index].active
        {
            let ledger = parts
                .video
                .ledger
                .as_ref()
                .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
            let canvas = *parts
                .video
                .canvases
                .get(&slot)
                .ok_or_else(|| Error::Invalid("a denoising request has no canvas".into()))?;
            let (banks, update) = if kind == MediaCall::LatentPreparation {
                ledger.initial(slot, call, canvas)?
            } else {
                ledger.step(slot, call, canvas)?
            };
            step["calls"][index]["latent"] = serde_json::json!({
                "read": banks.read,
                "write": banks.write,
                "step": banks.step,
            });
            media.batch.calls[index].forward.latent = Some(update);
        }
    }

    job.call_returning(parts.video.executor.execute(step.to_string(), tensors));

    for (index, product, write, host) in published {
        owners.store.write(&write, &write.view(), issue, None)?;
        media.batch.calls[index].forward.export(
            owners,
            &plan.calls[index],
            &product,
            &write,
            host,
            issue,
        )?;
    }
    Ok(())
}

/// Whether the executor runs calls of `kind` on its host thread.
fn host_kind(kind: MediaCall) -> bool {
    matches!(
        kind,
        MediaCall::MediaReading
            | MediaCall::VideoEncoding
            | MediaCall::AudioEncoding
            | MediaCall::Muxing
    )
}

impl MediaBatch {
    /// Whether an input is read in place from its export.
    fn borrowed(&self, product: &TensorRef) -> bool {
        self.borrowed.contains(&product.buffer_id())
    }
}

/// A borrowed shared-memory row as host bytes, viewed as long as the read
/// lives.
struct MappedBytes {
    mapping: Arc<SharedMapping>,
    offset: usize,
    bytes: usize,
}

impl Backing for MappedBytes {
    fn device(&self) -> MemoryKind {
        MemoryKind::Host
    }

    fn address(&self) -> u64 {
        (self.mapping.address() + self.offset) as u64
    }

    fn bytes(&self) -> usize {
        self.bytes
    }
}

/// A host uint8 view of a borrowed row's bytes.
fn row_view(read: &SharedRead) -> Result<StorageView> {
    let bytes = read.nbytes();
    let backing = MappedBytes {
        mapping: read.mapping()?,
        offset: read.offset(),
        bytes,
    };
    StorageView::contiguous(
        Arc::new(backing) as Arc<dyn Backing>,
        0,
        DType::U8,
        &[bytes as i64],
    )
}

/// Row `row` of a round's decoded frames, read in place from its
/// shared-memory export on this host.
fn borrow_row(
    parts: &Parts<'_>,
    plan: &BatchPlan,
    product: &TensorRef,
    row: u64,
) -> Result<SharedRead> {
    let export = supplied(plan, product)
        .ok_or_else(|| Error::Invalid("a borrowed media input has no export".into()))?;
    let tensor = crate::executor::products::tensor(export)?;
    let transport = parts
        .storage
        .transports
        .get(TransportKind::PosixShm)
        .ok_or_else(|| Error::Invalid("video encoding requires local shared storage".into()))?;
    let location = tensor
        .locations
        .iter()
        .find(|location| {
            matches!(location.transport, TransferTransport::PosixShm { .. })
                && location.source.host == parts.info.endpoint.host
                && location.offset[0] <= row
                && row < location.offset[0] + location.shape[0]
        })
        .ok_or_else(|| Error::Invalid("decoded media unit has no local shared storage".into()))?;
    let region: Vec<Range<u64>> = std::iter::once(row..row + 1)
        .chain(tensor.shape[1..].iter().map(|&extent| 0..extent))
        .collect();
    transport.borrow(location, Some(&region))
}

/// A round's framed unit rows, `[rows, width]` host uint8, read from the
/// exports that reach this rank.
fn exported_rows(parts: &Parts<'_>, plan: &BatchPlan, product: &TensorRef) -> Result<StorageView> {
    let export = supplied(plan, product).ok_or_else(|| {
        Error::Invalid("artifact assembly input has no transfer locations".into())
    })?;
    let tensor = crate::executor::products::tensor(export)?;
    if tensor.shape.len() != 2
        || tensor
            .locations
            .iter()
            .any(|location| location.dtype != "uint8")
    {
        return Err(Error::Invalid("encoded units require byte rows".into()));
    }
    let (count, width) = (tensor.shape[0] as usize, tensor.shape[1] as usize);
    let memory = Storage::host((count * width).max(1));
    let mut filled = vec![false; count];
    for location in &tensor.locations {
        if location.offset.get(1).copied().unwrap_or(0) != 0 {
            return Err(Error::Invalid(
                "encoded unit location lacks its length prefix".into(),
            ));
        }
        let first = location.offset[0] as usize;
        let rows = location.shape[0] as usize;
        let row_bytes = location.shape.get(1).copied().unwrap_or(0) as usize;
        if first + rows > count || filled[first..first + rows].iter().all(|&done| done) {
            continue;
        }
        let bytes: Vec<u8> = match &location.transport {
            TransferTransport::Channel { payload, .. } => payload.clone(),
            TransferTransport::PosixShm { .. }
                if location.source.host == parts.info.endpoint.host =>
            {
                let Some(transport) = parts.storage.transports.get(TransportKind::PosixShm) else {
                    continue;
                };
                let mut read = transport.borrow(location, None)?;
                let mapping = read.mapping()?;
                // SAFETY: the claimed range is mapped and ready until the
                // read is released below.
                let payload = unsafe {
                    std::slice::from_raw_parts(
                        (mapping.address() + read.offset()) as *const u8,
                        read.nbytes(),
                    )
                }
                .to_vec();
                read.release();
                payload
            }
            _ => continue,
        };
        if row_bytes == 0 || row_bytes > width || bytes.len() != rows * row_bytes {
            return Err(Error::Invalid(
                "encoded unit rows disagree with their location".into(),
            ));
        }
        for (offset, row) in bytes.chunks_exact(row_bytes).enumerate() {
            let target = (first + offset) * width;
            // SAFETY: the host allocation holds `count * width` bytes, this
            // row's range lies inside it and nothing else references it.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    (memory.address() as *mut u8).add(target),
                    row.len(),
                );
            }
            filled[first + offset] = true;
        }
    }
    if filled.iter().any(|&done| !done) {
        return Err(Error::Invalid(
            "artifact assembly requires every encoded media unit".into(),
        ));
    }
    StorageView::contiguous(
        memory as Arc<dyn Backing>,
        0,
        DType::U8,
        &[count as i64, width as i64],
    )
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
fn retire(storage: &WorkerMemory, mut media: MediaBatch, job: Job, failure: Failure) -> Step {
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
        let parts = backend.media_parts()?;
        let (plan, _, pending) = batch.media_parts();
        let failed = Failure::new(PHASE);
        match (pending.take(), completion.check()) {
            // A predicate that could not be read, or a plan the executor
            // refused, fails the batch, which has reserved nothing yet.
            (Some(Pending::Predicates(_) | Pending::Planning(..)), Err(failure)) => {
                Done::Failed(failed(failure.error))
            }
            (Some(Pending::Predicates(mut opening)), Ok(mut completion)) => {
                match opening.predicates(&mut completion) {
                    Ok(()) => Done::Next(Pending::Opened(opening)),
                    Err(error) => Done::Failed(failed(error)),
                }
            }
            (Some(Pending::Planning(opening, mut step)), Ok(mut completion)) => {
                let planned = completion.texts().and_then(|texts| {
                    serde_json::from_str::<serde_json::Value>(&texts.concat())
                        .map_err(|error| Error::Interface(format!("the executor's plan: {error}")))
                });
                match planned {
                    Ok(planned) => {
                        step["planned"] = planned;
                        Done::Next(Pending::Planning(opening, step))
                    }
                    Err(error) => Done::Failed(failed(error)),
                }
            }
            // The job failed on its lane, which still made its event
            // records: the batch retires what it reserved.
            (Some(Pending::Committing(media)), Err(failure)) => {
                Done::Next(Pending::Failing(media, failed(failure.error)))
            }
            (Some(Pending::Committing(mut media)), Ok(mut completion)) => {
                let modes = completion.returned();
                let committed = match modes {
                    Ok(modes) => {
                        for mode in modes {
                            media.record(match mode {
                                1 => GraphMode::Replay,
                                2 => GraphMode::Capture,
                                _ => GraphMode::Eager,
                            });
                        }
                        media.apply_commit(
                            plan,
                            &parts.storage.tensors,
                            parts.requests,
                            parts.video.ledger.as_mut(),
                        )
                    }
                    Err(error) => Err(failed(error)),
                };
                match committed {
                    Ok(()) => match parts.video.executor.reports() {
                        Ok(reports) => {
                            Done::Launched(Box::new(media.launch(reports.inbox(), plan.batch_id)))
                        }
                        Err(error) => Done::Failed(Failure::fatal(PHASE)(error)),
                    },
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

#[cfg(test)]
mod tests {
    use ipc::RequestId;

    use super::*;
    use crate::memory::{LatentPlan, LocalMemory};

    /// The ledger of a rank whose two request slots own two pages of four
    /// units each.
    fn ledger() -> Result<LatentLedger> {
        let plan = LatentPlan {
            request_pool_size: 2,
            num_pages: 5,
            page_units: 4,
            latent_width: 1,
            dtype: DType::F32,
            with_workspace: false,
        };
        let storage = LatentStorage::allocate(&plan, None, None, "latent_banks", &mut LocalMemory)?;
        Ok(LatentLedger::new(LatentPool::new(2, 5, 4)?, storage, 4, 2))
    }

    fn call(slot: u64, kind: MediaCall) -> Call {
        Call::new(
            ipc::RequestKey::new(1, RequestId(slot), 1),
            CallId::new(1, 0),
            CallKind::Media(kind),
        )
    }

    /// A latent preparation opens a trajectory in bank one at step zero;
    /// each denoising step reads the bank holding the committed step and
    /// writes the other, and a released slot starts a new trajectory.
    #[test]
    fn a_trajectory_alternates_banks_from_its_committed_step() -> Result<()> {
        let mut ledger = ledger()?;
        let canvas = (1344, 768);
        let preparation = call(2, MediaCall::LatentPreparation);
        let (banks, update) = ledger.initial(2, &preparation, canvas)?;
        assert_eq!((banks.read, banks.write, banks.step), (1, 1, 0));
        // Slot two owns the two pages after slot one's and the padding page.
        assert_eq!(update.params.page_table, [3, 4]);
        assert_eq!(update.params.latent_units, 8);
        assert_eq!((update.params.width, update.params.height), canvas);
        ledger.validate(std::slice::from_ref(&update))?;
        ledger.apply(&[update]);

        let denoising = call(2, MediaCall::Denoising);
        for (step, read) in [(0, 1), (1, 0), (2, 1)] {
            let (banks, update) = ledger.step(2, &denoising, canvas)?;
            assert_eq!(
                (banks.read, banks.write, banks.step),
                (read, 1 - read, step)
            );
            ledger.validate(std::slice::from_ref(&update))?;
            ledger.apply(&[update]);
        }
        // A trajectory in use cannot be opened again until its slot retires.
        assert!(ledger.initial(2, &preparation, canvas).is_err());
        ledger.release_slots(&[2])?;
        let (banks, _) = ledger.initial(2, &preparation, canvas)?;
        assert_eq!((banks.read, banks.write, banks.step), (1, 1, 0));
        Ok(())
    }
}
