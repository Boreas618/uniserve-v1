//! The running state of one media batch, from input readiness to its result.
//!
//! A media batch's calls run one after another on their component's stream
//! (device calls) or as host-lane tasks (host calls). Execution then
//! commits the batch in one step: it ends its input leases, writes each
//! call's completion scalar, makes every produced write visible, applies
//! trajectory commits and records the calls as pending in the request pool.
//! The result follows once the batch's device work completed (its output
//! fences are query-ready) and its host tasks finished; host tasks publish
//! the products they wrote before the result is delivered.

pub(crate) mod image;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, PoisonError};

use uniserve_core::MediaSource;
use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BatchOutput, BufferAllocation, BufferId, Call, CallKind, CallStatus,
    ErrorCode, ForwardStats, MediaCall, RequestKey, TensorExport, TensorRef, WorkerInfo,
};

use super::products::Publications;
use crate::cuda_graph::GraphMode;
use crate::error::WorkerError;
use crate::executor::forward::placement::{
    OutputShape, Placement, ProductOwners, Reads, bind_completion, bind_outputs, export, skipped,
};
use crate::executor::pending::{BatchResult, PendingOutput};
use crate::ffi::Device;
use crate::host::HostTask;
use crate::host::task::{MediaTask, TaskOutput};
use crate::request::RequestPool;
use crate::request::video::VideoRequests;
use crate::runner::decoder::runners::DecoderRunners;
use crate::runner::denoiser::runners::DenoiserRunners;
use crate::runner::encoder::runners::EncoderRunners;
use crate::runner::media::MediaRunner;
use crate::storage::OutputLease;
use crate::storage::ProductExport;
use crate::storage::Scalar;
use crate::stream::Issue;
use crate::transfer::exports::Location;
use crate::worker::RankStorage;
use crate::{Error, LatentUpdate, OutputPool, ProductBuffer, Result};

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

/// The latent pool a batch's trajectory commits apply to.
pub(crate) trait Trajectories {
    /// Check every commit before any becomes visible.
    fn validate(&self, updates: &[LatentUpdate]) -> Result<()>;
    /// Make the checked commits visible.
    fn apply(&mut self, updates: &[LatentUpdate]);
}

/// What a batch's issued commit leaves to apply once its work was
/// performed: the writes to make visible, the trajectory commits and the
/// calls to record as pending.
pub(crate) struct Commit {
    writes: Vec<ProductBuffer>,
    updates: Vec<LatentUpdate>,
    pending: Vec<(RequestKey, uniserve_core::CallId, bool)>,
}

/// One call of the batch.
pub(crate) struct CallState {
    pub(crate) pending: PendingOutput,
    /// Progress before the call, restored when it fails or is skipped.
    pub(crate) parent: crate::RequestProgress,
    /// Writes made visible at commit.
    pub(crate) writes: Vec<ProductBuffer>,
    /// The completion scalar a successor's predicate reads.
    pub(crate) completion: Option<ProductBuffer>,
    /// The call's product exports: registered with the store at the commit
    /// point (host products when their task wrote them), so releasing a
    /// product revokes them; revoked when the batch fails before.
    pub(crate) publications: Publications,
    /// The exports the call's forward issued, located when the batch
    /// commits: their device work is performed by then.
    exports: Vec<(ProductBuffer, ProductExport)>,
    pub(crate) latent: Option<LatentUpdate>,
    pub(crate) tasks: Vec<Arc<HostTask<MediaTask>>>,
    pub(crate) publication: Option<Publication>,
    /// The predicate was false: the call ran nothing.
    pub(crate) skipped: bool,
}

impl CallState {
    /// Record an export of `buffer` and add its descriptor to the call's
    /// completion products.
    pub(crate) fn publish(
        &mut self,
        buffer: BufferId,
        locations: Vec<Location>,
        export: TensorExport,
    ) -> Result<()> {
        self.publications
            .accept(buffer, locations, export, &mut self.pending)
    }

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

    /// Publish the exports the call's forward issued, once their work was
    /// performed: each joins the call's completion products.
    fn locate(&mut self, store: &crate::TensorStore) -> Result<()> {
        for (write, export) in std::mem::take(&mut self.exports) {
            let (export, locations) = store.located(&write, export)?;
            self.publish(write.id(), locations, export)?;
        }
        Ok(())
    }
}

/// One media batch's running state.
pub(crate) struct MediaBatch {
    pub(crate) inputs: Reads,
    pub(crate) calls: Vec<CallState>,
    output: OutputLease,
    /// The stream the batch's device work runs on, when it has any.
    pub(crate) stream: Option<Stream>,
    pub(crate) device: Device,
    /// The graph modes of the batch's forwards, reported with its result.
    stats: ForwardStats,
    committed: bool,
    delivered: bool,
}

fn lock(lease: &OutputLease) -> std::sync::MutexGuard<'_, crate::OutputBuffer> {
    lease.lock().unwrap_or_else(PoisonError::into_inner)
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

    /// Open a batch of `calls` calls on `device` and `stream` that reads
    /// `inputs`.
    pub(crate) fn new(
        device: Device,
        stream: Option<Stream>,
        outputs: &mut OutputPool,
        calls: usize,
        inputs: Reads,
    ) -> Result<Self> {
        let devices: Vec<i32> = match device {
            Device::Cuda(index) => vec![index],
            Device::Cpu | Device::CudaHost => Vec::new(),
        };
        let output = outputs.lease(calls.max(1), calls.max(1), &devices, false)?;
        Ok(Self {
            inputs,
            calls: Vec::new(),
            output,
            stream,
            device,
            stats: ForwardStats::default(),
            committed: false,
            delivered: false,
        })
    }

    /// Open each call's pending result against its request's progress.
    pub(crate) fn open_calls(
        &mut self,
        plan: &BatchPlan,
        requests: &RequestPool,
        output_rank: impl Fn(&str) -> bool,
    ) -> Result<()> {
        if !self.calls.is_empty() {
            return Ok(());
        }
        let rows: Vec<_> = plan
            .calls
            .iter()
            .map(|call| {
                let request = requests.get(call.request_key.request_id.0)?;
                Ok((call.request_key, call.call_id, request.slot()))
            })
            .collect::<Result<_>>()?;
        let bound = requests.bind_calls(&rows)?;
        for (row, (call, request)) in plan.calls.iter().zip(bound).enumerate() {
            let parent = request.progress()?;
            let mut pending = PendingOutput::new(call, parent, row)?;
            pending.reports_output = output_rank(&call.component);
            self.calls.push(CallState {
                pending,
                parent,
                writes: Vec::new(),
                completion: None,
                publications: Publications::default(),
                exports: Vec::new(),
                latent: None,
                tasks: Vec::new(),
                publication: None,
                skipped: false,
            });
        }
        Ok(())
    }

    /// Mark the calls whose predicate is false as skipped.
    pub(crate) fn skip(&mut self, skipped: &HashSet<usize>) {
        for &index in skipped {
            if let Some(call) = self.calls.get_mut(index) {
                call.skipped = true;
                call.pending.output.status = CallStatus::Predicated;
            }
        }
    }

    /// Commit the launched batch on its stream at once: issue the commit
    /// and apply it.
    pub(crate) fn commit(
        &mut self,
        plan: &BatchPlan,
        owners: ProductOwners<'_>,
        requests: &mut RequestPool,
        trajectories: Option<&mut dyn Trajectories>,
    ) -> Result<()> {
        let stream = self.stream.clone();
        let commit = self.issue_commit(
            plan,
            owners,
            requests,
            trajectories.as_deref(),
            stream.as_ref(),
        )?;
        self.apply_commit(commit, owners, requests, trajectories)
    }

    /// Issue the device half of the batch's commit to `stream`, behind its
    /// forwards: end its leases, write completion scalars and seal its
    /// output. Every owner but the exports checks its update first, so the
    /// returned commit is applied once that work was performed
    /// ([`MediaBatch::apply_commit`]).
    pub(crate) fn issue_commit(
        &mut self,
        plan: &BatchPlan,
        owners: ProductOwners<'_>,
        requests: &RequestPool,
        trajectories: Option<&dyn Trajectories>,
        stream: impl Issue,
    ) -> Result<Commit> {
        self.inputs.complete(owners.store, stream)?;
        for call in &self.calls {
            if let Some(completion) = &call.completion {
                // Skipped calls forward a false decision to their successor.
                let value = Scalar::Int(i64::from(!call.skipped));
                owners.store.write_scalar(completion, value, stream)?;
            }
        }

        let writes: Vec<ProductBuffer> = self
            .calls
            .iter()
            .flat_map(|call| call.writes.iter().chain(call.completion.iter()))
            .cloned()
            .collect();
        owners.store.validate_writes(&writes)?;
        let updates: Vec<LatentUpdate> = self
            .calls
            .iter()
            .filter_map(|call| call.latent.clone())
            .collect();
        match trajectories {
            Some(pool) => pool.validate(&updates)?,
            None if updates.is_empty() => {}
            None => {
                return Err(Error::Invariant(
                    "a trajectory commit has no latent pool".into(),
                ));
            }
        }
        let pending: Vec<_> = plan
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id, call.advances_state()))
            .collect();
        requests.validate_pending(&pending)?;

        // A lease of host storage is sealed on no stream.
        let streams = if stream.stream().is_some() {
            vec![stream]
        } else {
            Vec::new()
        };
        let fences = lock(&self.output)
            .seal(&streams, owners.events)?
            .unwrap_or_default();
        for fence in &fences {
            stream.wake(owners.events, fence)?;
        }
        let finished = Arc::clone(&self.output);
        owners.events.defer(
            fences,
            Box::new(move || {
                lock(&finished).copies_finished();
                Ok(())
            }),
        )?;

        Ok(Commit {
            writes,
            updates,
            pending,
        })
    }

    /// Apply an issued commit once its work was performed: publish the
    /// calls' exports, make writes visible, apply trajectory commits to
    /// `trajectories` and record the calls as pending.
    pub(crate) fn apply_commit(
        &mut self,
        commit: Commit,
        owners: ProductOwners<'_>,
        requests: &mut RequestPool,
        mut trajectories: Option<&mut dyn Trajectories>,
    ) -> Result<()> {
        let Commit {
            writes,
            updates,
            pending,
        } = commit;
        for call in &mut self.calls {
            call.locate(owners.store)?;
            call.publications.validate(owners.store)?;
        }

        // Every owner accepted its update; failures from here on are fatal.
        self.committed = true;
        owners.store.commit_writes(&writes)?;
        for call in &mut self.calls {
            call.publications.register(owners.store)?;
        }
        if let Some(pool) = trajectories.as_mut() {
            pool.apply(&updates);
        }
        for call in &mut self.calls {
            if let Some(update) = call.latent.take() {
                call.pending.apply_latent_update(&update);
            }
        }
        requests.add_pending(&pending)
    }

    /// Advance completion: publish what finished host tasks wrote, then
    /// resolve the results once every task and device fence completed.
    /// Returns the batch's result when it is complete.
    pub(crate) fn poll(
        &mut self,
        plan: &BatchPlan,
        owners: ProductOwners<'_>,
        requests: &mut RequestPool,
    ) -> Result<Option<BatchResult>> {
        if self.delivered {
            return Ok(None);
        }
        for index in 0..self.calls.len() {
            if !self.finish_tasks(index, plan, owners)? {
                return Ok(None);
            }
        }
        if lock(&self.output).ready_at()?.is_none() {
            return Ok(None);
        }

        let mut completions = Vec::with_capacity(self.calls.len());
        let mut products = Vec::new();
        let mut media = Vec::new();
        for call in &mut self.calls {
            let mut output = lock(&self.output);
            call.pending.resolve(call.parent, &mut output)?;
            drop(output);
            call.pending.accept(requests)?;
            let result = call.pending.result()?;
            if result.status == CallStatus::Ok {
                products.extend_from_slice(call.pending.products());
            }
            completions.push(result.clone());
            if let Some(source) = call.pending.take_media() {
                media.push(source);
            }
        }
        self.delivered = true;
        Ok(Some(BatchResult {
            output: BatchOutput {
                batch_id: plan.batch_id,
                completions,
                products,
                worker_exec_us: None,
                forward_stats: (!self.stats.is_empty()).then(|| self.stats.clone()),
            },
            media,
        }))
    }

    /// Publish a call's host results once its tasks finished. Returns
    /// whether the call has no unfinished task.
    fn finish_tasks(
        &mut self,
        index: usize,
        plan: &BatchPlan,
        owners: ProductOwners<'_>,
    ) -> Result<bool> {
        let call = &mut self.calls[index];
        if call.tasks.is_empty() {
            return Ok(true);
        }
        if call.tasks.iter().any(|task| !task.completion.done()) {
            return Ok(false);
        }
        let tasks = std::mem::take(&mut call.tasks);
        let mut outputs = Vec::with_capacity(tasks.len());
        let mut failure = None;
        for task in &tasks {
            match task.completion.outcome() {
                Some(crate::Outcome::Success(output)) => outputs.push(output),
                Some(crate::Outcome::Failed(error)) => {
                    failure.get_or_insert(error.to_string());
                }
                Some(crate::Outcome::Cancelled) | None => {
                    failure.get_or_insert("a host media task was cancelled".to_owned());
                }
            }
        }
        let publication = call.publication.take();
        if let Some(message) = failure {
            tracing::error!(target: "uniserve_worker.media", "host media call failed: {message}");
            call.pending.output.status = CallStatus::Error;
            call.pending.output.error_code = Some(ErrorCode::ComputeError);
            let writes: Vec<ProductBuffer> = match publication {
                Some(Publication::Products(outputs)) => {
                    outputs.into_iter().map(|(_, write)| write).collect()
                }
                Some(Publication::Encoded { write, .. }) => vec![write],
                _ => Vec::new(),
            };
            owners.store.abandon_writes(&writes)?;
            return Ok(true);
        }

        let plan_call = &plan.calls[index];
        match publication {
            Some(Publication::Products(written)) => {
                for (product, write) in written {
                    let view = write.view();
                    owners.store.write(&write, &view, None, None)?;
                    owners.store.commit_writes(std::slice::from_ref(&write))?;
                    let (exported, locations) =
                        export(owners, plan_call, &product, &write, true, None, None)?;
                    call.publish(write.id(), locations, exported)?;
                    // Written after the commit point: registered at once.
                    call.publications.register(owners.store)?;
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
                for (output, row) in outputs.iter().zip(&rows) {
                    let TaskOutput::Framed(length) = **output else {
                        return Err(Error::Invariant(
                            "a video encoding task framed no unit".into(),
                        ));
                    };
                    if length as u64 > width {
                        return Err(Error::Invariant("a framed unit exceeds its row".into()));
                    }
                    regions.push(vec![*row..*row + 1, 0..length as u64]);
                }
                let (exported, locations) = export(
                    owners,
                    plan_call,
                    &product,
                    &write,
                    true,
                    Some(&regions),
                    None,
                )?;
                call.publish(write.id(), locations, exported)?;
                call.publications.register(owners.store)?;
            }
            Some(Publication::Media) if call.pending.reports_output => {
                for output in &outputs {
                    if let TaskOutput::Media(bytes) = &**output {
                        let source = MediaSource::publish(bytes).map_err(|error| {
                            Error::Transport(format!("publishing media failed: {error}"))
                        })?;
                        call.pending.set_media(source)?;
                    }
                }
            }
            Some(Publication::Media) | None => {}
        }
        Ok(true)
    }

    /// Release what an undelivered batch holds: before commit, abandon its
    /// writes; after commit, withdraw its pending calls. Exports are revoked
    /// and queued host tasks withdrawn either way.
    pub(crate) fn close(
        &mut self,
        owners: ProductOwners<'_>,
        requests: &mut RequestPool,
    ) -> Result<()> {
        if self.delivered {
            return Ok(());
        }
        let mut result = self.inputs.complete(owners.store, self.stream.as_ref());
        for call in &mut self.calls {
            for task in &call.tasks {
                if let Err(error) = task.cancel(false) {
                    tracing::warn!(target: "uniserve_worker.media", "withdrawing a host media task failed: {error}");
                }
            }
            call.exports
                .drain(..)
                .for_each(|(_, export)| export.revoke());
            result = result.and(call.publications.revoke());
        }
        if self.committed {
            for call in &self.calls {
                result = result.and(call.pending.cancel(requests));
            }
        } else {
            let writes: Vec<ProductBuffer> = self
                .calls
                .iter()
                .flat_map(|call| call.writes.iter().chain(call.completion.iter()))
                .cloned()
                .collect();
            result = result.and(owners.store.abandon_writes(&writes));
        }
        lock(&self.output).abandon();
        self.delivered = true;
        result
    }
}

/// The request slots of a batch's calls' requests, for relay outputs.
pub(crate) fn request_slots(
    plan: &BatchPlan,
    requests: &RequestPool,
) -> Result<HashMap<RequestKey, usize>> {
    plan.calls
        .iter()
        .map(|call| {
            let request = requests.get(call.request_key.request_id.0)?;
            Ok((call.request_key, request.slot()))
        })
        .collect()
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
            events: &storage.events,
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
                Ok((denoiser.device, Some(denoiser.runner.stream().clone())))
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

/// Whether `rank` reports a component's host results: its first rank does.
pub(crate) fn reports(info: &WorkerInfo, rank: usize, component: &str) -> bool {
    info.components
        .iter()
        .find(|placed| placed.name == component)
        .and_then(|placed| placed.config.ranks.first())
        .is_none_or(|&first| first == rank)
}

fn media_kind(call: &Call) -> MediaCall {
    match call.code {
        CallKind::Media(kind) => kind,
        _ => MediaCall::Denoising,
    }
}

/// Open a video batch: its output lease on the stream of its kind, its
/// calls' pending results, and the calls its false predicates skip.
/// `borrowed` names the inputs host codecs read in place.
pub(crate) fn open<'b>(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    borrowed: HashSet<BufferId>,
    state: &'b mut Option<MediaBatch>,
) -> Result<&'b mut MediaBatch> {
    let first = plan
        .calls
        .first()
        .ok_or_else(|| Error::Invariant("a video batch has calls".into()))?;
    let (device, stream) = parts.placement(media_kind(first), &first.component)?;
    let batch = state.insert(MediaBatch::new(
        device,
        stream,
        parts.outputs,
        plan.calls.len(),
        Reads::new(borrowed),
    )?);
    let (info, rank) = (parts.info, parts.rank);
    batch.open_calls(plan, parts.requests, |component| {
        reports(info, rank, component)
    })?;
    let skipped = skipped(
        &parts.storage.tensors,
        plan,
        batch.device,
        batch.stream.as_ref(),
    )?;
    batch.skip(&skipped);
    Ok(batch)
}

/// Execute a batch of device video calls and commit it.
pub(crate) fn execute(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    borrowed: HashSet<BufferId>,
    state: &mut Option<MediaBatch>,
) -> std::result::Result<(), WorkerError> {
    let batch = open(parts, plan, borrowed, state)?;

    let slots = request_slots(plan, parts.requests)?;
    let allocations: HashMap<BufferId, BufferAllocation> = plan
        .buffer_allocations
        .iter()
        .map(|allocation| (allocation.buffer, *allocation))
        .collect();
    let placement = Placement {
        allocations: &allocations,
        slots: &slots,
    };
    for index in 0..plan.calls.len() {
        let call = &plan.calls[index];
        let slot = slots[&call.request_key];
        let owners = parts.owners();
        let completion = bind_completion(owners.store, call, batch.device, &slots)?;
        batch.calls[index].completion = completion;
        if batch.calls[index].skipped {
            continue;
        }
        match call.code {
            CallKind::Media(MediaCall::TextEncoding) => {
                crate::runner::encoder::encode_text(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::VisionEncoding) => {
                crate::runner::encoder::encode_vision(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::LatentPreparation) => {
                crate::runner::denoiser::prepare(parts, plan, batch, index, slot)?
            }
            CallKind::Media(MediaCall::Denoising) => {
                crate::runner::denoiser::denoise(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::VideoDecoding) => {
                crate::runner::decoder::decode_video(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::AudioDecoding) => {
                crate::runner::decoder::decode_audio(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::LatentEncoding) => {
                crate::runner::encoder::encode_conditions(
                    parts, plan, batch, index, slot, &placement,
                )?
            }
            kind => {
                return Err(Error::Unsupported(format!(
                    "{} is not a device video call",
                    kind.as_str()
                ))
                .into());
            }
        }
    }
    let owners = parts.owners();
    let trajectories = parts
        .denoisers
        .video
        .as_mut()
        .map(|denoiser| &mut denoiser.runner as &mut dyn Trajectories);
    batch.commit(plan, owners, parts.requests, trajectories)?;
    Ok(())
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
/// computes it.
pub(crate) fn bind_rows(
    parts: &Parts<'_>,
    batch: &mut MediaBatch,
    index: usize,
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
        batch.device,
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("an encoder product was not bound".into()))?;
    batch.calls[index].writes.push(write.clone());
    Ok(write)
}

/// Publish a call's output, written at `shape` from its leading elements,
/// and export it to its readers.
pub(crate) fn publish(
    parts: &Parts<'_>,
    batch: &mut MediaBatch,
    call: &Call,
    index: usize,
    write: &crate::ProductBuffer,
    shape: &[u64],
) -> Result<()> {
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    owners.store.produced(write, shape, Some(&stream), None)?;
    batch.calls[index].export(owners, call, &call.outputs[0], write, false, Some(&stream))
}

/// Publish a call's first output, whose bound view the producer filled, and
/// export it to its readers (`host` readers read it from host memory).
///
/// The bound view is this rank's share of the product: its region when
/// other ranks hold the rest, the whole product when the round assigns
/// this rank all of it. Publishing the view itself is right in both cases;
/// the store tracks a share that covers the product as a whole product.
pub(crate) fn publish_bound(
    parts: &Parts<'_>,
    batch: &mut MediaBatch,
    call: &Call,
    index: usize,
    write: &crate::ProductBuffer,
    host: bool,
) -> Result<()> {
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    owners
        .store
        .write(write, &write.view(), Some(&stream), None)?;
    batch.calls[index].export(owners, call, &call.outputs[0], write, host, Some(&stream))
}
