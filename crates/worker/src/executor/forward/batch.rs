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

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, PoisonError};

use uniserve_core::MediaSource;
use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BatchOutput, BufferId, CallStatus, ErrorCode, RequestKey, TensorExport,
    TensorRef,
};

use super::products::Publications;
use crate::executor::forward::placement::{ProductOwners, Reads, export};
use crate::executor::pending::{BatchResult, PendingOutput};
use crate::ffi::Device;
use crate::host::HostTask;
use crate::host::task::{MediaTask, TaskOutput};
use crate::request::RequestPool;
use crate::storage::OutputLease;
use crate::storage::Scalar;
use crate::transfer::exports::Location;
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
}

/// One media batch's running state.
pub(crate) struct MediaBatch {
    pub(crate) inputs: Reads,
    pub(crate) calls: Vec<CallState>,
    output: OutputLease,
    /// The stream the batch's device work runs on, when it has any.
    pub(crate) stream: Option<Stream>,
    pub(crate) device: Device,
    committed: bool,
    delivered: bool,
}

fn lock(lease: &OutputLease) -> std::sync::MutexGuard<'_, crate::OutputBuffer> {
    lease.lock().unwrap_or_else(PoisonError::into_inner)
}

impl MediaBatch {
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

    /// Commit the launched batch: end its leases, write completion scalars,
    /// make writes visible, apply trajectory commits to `trajectories` and
    /// record the calls as pending.
    pub(crate) fn commit(
        &mut self,
        plan: &BatchPlan,
        owners: ProductOwners<'_>,
        requests: &mut RequestPool,
        mut trajectories: Option<&mut dyn Trajectories>,
    ) -> Result<()> {
        let stream = self.stream.as_ref();
        self.inputs.complete(owners.store, stream)?;
        for call in &mut self.calls {
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
        for call in &self.calls {
            call.publications.validate(owners.store)?;
        }
        let updates: Vec<LatentUpdate> = self
            .calls
            .iter()
            .filter_map(|call| call.latent.clone())
            .collect();
        match &trajectories {
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
        requests.add_pending(&pending)?;

        let fences = {
            let mut output = lock(&self.output);
            let streams: Vec<&Stream> = stream.into_iter().collect();
            output.seal(&streams, owners.events)?
        };
        let fences = fences.unwrap_or_default();
        for fence in &fences {
            owners.events.wake_after(fence)?;
        }
        let finished = Arc::clone(&self.output);
        owners.events.defer(
            fences,
            Box::new(move || {
                lock(&finished).copies_finished();
                Ok(())
            }),
        )?;
        Ok(())
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
                forward_stats: None,
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
