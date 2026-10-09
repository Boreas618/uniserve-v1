//! An image batch's reservation, commit and retirement.
//!
//! [`ImageBatch::open`] binds every call's pending output, resolves
//! host-gated predicates (a false one skips its call, which only publishes
//! false flags) and leases the batch's output storage on its stream.
//! [`ImageBatch::bind_flags`] binds the completion flags the calls publish.
//! After a kind's computation, [`ImageBatch::commit`] validates every
//! owner's update (products, latent trajectories, pending calls) and makes
//! them visible together; a failure before that point retires everything
//! the batch reserved ([`ImageBatch::discard`]), a failure after it is
//! fatal.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, CallStatus, DType, RequestKey, TensorRef,
};

use super::super::launch::{read_flag, reports_output, seal as seal_lease};
use super::super::products::Publications;
use super::ImageRuntime;
use super::host::ImageTask;
use crate::executor::PendingOutput;
use crate::ffi::Device;
use crate::host::HostTask;
use crate::request::RequestPool;
use crate::storage::{
    LatentUpdate, OutputLease, OutputPlacement, ProductBuffer, ProductRead, Scalar,
};
use crate::transfer::Retirement;
use crate::worker::RankStorage;
use crate::{Error, Result};

/// One call's state through its batch.
pub(super) struct CallState {
    pub(super) output: PendingOutput,
    pub(super) slot: usize,
    /// False when the call's host predicate skipped it.
    pub(super) active: bool,
    pub(super) completion_write: Option<ProductBuffer>,
}

/// A decoded image's encoding, run once its pixels reached the host.
pub(super) struct Encoding {
    /// The call's index in the batch.
    pub(super) call: usize,
    pub(super) task: Arc<HostTask<ImageTask>>,
}

/// A failure, and whether it happened past the commit point.
pub(super) struct Failure {
    pub(super) error: Error,
    pub(super) committed: bool,
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Self {
            error,
            committed: false,
        }
    }
}

pub(super) fn fatal(error: Error) -> Failure {
    Failure {
        error,
        committed: true,
    }
}

pub(super) use super::super::launch::lock;

/// A batch's persistent buffer allocations by buffer.
pub(super) fn allocations(plan: &BatchPlan) -> HashMap<BufferId, BufferAllocation> {
    plan.buffer_allocations
        .iter()
        .map(|allocation| (allocation.buffer, *allocation))
        .collect()
}

/// An image batch's borrowed owners and the resources it reserved.
pub(super) struct ImageBatch<'a> {
    pub(super) runtime: &'a mut ImageRuntime,
    pub(super) storage: &'a mut RankStorage,
    pub(super) requests: &'a mut RequestPool,
    pub(super) plan: &'a BatchPlan,
    /// The device and stream the batch's computation runs on.
    pub(super) device: Option<i32>,
    pub(super) stream: Option<Stream>,
    /// This rank reports the component's host results.
    pub(super) reports: bool,
    pub(super) calls: Vec<CallState>,
    pub(super) buffer: OutputLease,
    pub(super) writes: Vec<ProductBuffer>,
    pub(super) reads: Vec<Arc<ProductRead>>,
    /// Trajectory commits and releases by call index, applied at the
    /// commit point.
    pub(super) updates: Vec<(usize, LatentUpdate)>,
    /// Image encodings to run once the batch's pixels reached the host.
    pub(super) encodings: Vec<Encoding>,
    /// Product exports (encoder features, trajectory versions),
    /// registered once the batch commits.
    pub(super) publications: Publications,
    /// Trajectory versions exported from the latent pool, released from it
    /// if the batch fails before committing.
    pub(super) latent_exports: Vec<BufferId>,
    /// Signals the physical end of the batch's KV accesses.
    pub(super) retirement: Arc<Retirement>,
    /// The batch uses the latent pool on its stream (see
    /// `ImageLatents::enter`).
    pub(super) latents_entered: bool,
    pub(super) started: Instant,
}

impl<'a> ImageBatch<'a> {
    /// Bind the batch's outputs, skip calls whose host predicate is false
    /// and lease its output storage on `stream`. `words` are the payload
    /// words the batch reads back beside one completion word per call.
    #[expect(clippy::too_many_arguments, reason = "the batch's borrowed owners")]
    pub(super) fn open(
        runtime: &'a mut ImageRuntime,
        storage: &'a mut RankStorage,
        requests: &'a mut RequestPool,
        info: &uniserve_ipc::WorkerInfo,
        rank: usize,
        plan: &'a BatchPlan,
        device: Option<i32>,
        stream: Option<Stream>,
        words: usize,
        started: Instant,
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
        let reports = reports_output(info, rank, &first.component)?;
        let rows = plan
            .calls
            .iter()
            .map(|call| {
                let request = requests.get(call.request_key.request_id.0)?;
                Ok((call.request_key, call.call_id, request.slot()))
            })
            .collect::<Result<Vec<_>>>()?;
        let bound = requests.bind_calls(&rows)?;
        let mut calls = Vec::with_capacity(plan.calls.len());
        for (index, (call, request)) in plan.calls.iter().zip(&bound).enumerate() {
            calls.push(CallState {
                output: PendingOutput::new(call, request.progress()?, index)?,
                slot: request.slot(),
                active: true,
                completion_write: None,
            });
        }
        for (call, state) in plan.calls.iter().zip(&mut calls) {
            if let Some(predicate) = &call.predicate
                && predicate.dtype == DType::U8
                && !read_flag(&storage.tensors, call, predicate, device, stream.as_ref())?
            {
                state.active = false;
                state.output.output.status = CallStatus::Predicated;
            }
        }
        let devices: Vec<i32> = device.into_iter().collect();
        let buffer = runtime.outputs.lease(
            plan.calls.len(),
            (plan.calls.len() + words).max(1),
            &devices,
            false,
        )?;
        if let Some(stream) = &stream {
            lock(&buffer).begin_device(device, stream, &storage.events)?;
        }
        Ok(Self {
            runtime,
            storage,
            requests,
            plan,
            device,
            stream,
            reports,
            calls,
            buffer,
            writes: Vec::new(),
            reads: Vec::new(),
            updates: Vec::new(),
            encodings: Vec::new(),
            publications: Publications::default(),
            latent_exports: Vec::new(),
            retirement: Arc::new(Retirement::default()),
            latents_entered: false,
            started,
        })
    }

    /// The batch's product device.
    pub(super) fn target(&self) -> Device {
        self.device.map_or(Device::Cpu, Device::Cuda)
    }

    /// Indices of the calls that execute.
    pub(super) fn active(&self) -> Vec<usize> {
        (0..self.calls.len())
            .filter(|&index| self.calls[index].active)
            .collect()
    }

    /// Bind every call's completion flag; skipped calls publish false now.
    pub(super) fn bind_flags(&mut self) -> Result<()> {
        let target = self.target();
        let slots: HashMap<RequestKey, usize> = self
            .plan
            .calls
            .iter()
            .zip(&self.calls)
            .map(|(call, state)| (call.request_key, state.slot))
            .collect();
        let flags: Vec<(TensorRef, Device)> = self
            .plan
            .calls
            .iter()
            .filter_map(|call| call.completion_output.clone())
            .map(|flag| (flag, target))
            .collect();
        if flags.is_empty() {
            return Ok(());
        }
        let bound = self.storage.tensors.bind_outputs(
            &flags,
            OutputPlacement {
                request_slots: Some(&slots),
                ..OutputPlacement::default()
            },
        )?;
        for write in bound {
            self.writes.push(write.clone());
            let index = self
                .plan
                .calls
                .iter()
                .position(|call| {
                    call.completion_output
                        .as_ref()
                        .is_some_and(|flag| flag.buffer_id() == write.id())
                })
                .ok_or_else(|| Error::Invariant("a completion flag names no call".into()))?;
            let state = &mut self.calls[index];
            if !state.active {
                self.storage
                    .tensors
                    .write_scalar(&write, Scalar::Int(0), self.stream.as_ref())?;
            }
            state.completion_write = Some(write);
        }
        Ok(())
    }

    /// Validate every owner's update, then commit them together.
    pub(super) fn commit(&mut self) -> std::result::Result<(), Failure> {
        let stream = self.stream.as_ref();
        let reads = std::mem::take(&mut self.reads);
        self.storage.tensors.complete_reads(&reads, stream, &[])?;

        // Completion flags of executed calls become true.
        for state in self.calls.iter().filter(|state| state.active) {
            if let Some(write) = &state.completion_write {
                self.storage
                    .tensors
                    .write_scalar(write, Scalar::Int(1), stream)?;
            }
        }
        {
            let output = lock(&self.buffer);
            for (state, call) in self.calls.iter().zip(&self.plan.calls) {
                state.output.validate_output(&output, call)?;
            }
        }
        self.storage.tensors.validate_writes(&self.writes)?;
        let updates: Vec<LatentUpdate> = self
            .updates
            .iter()
            .map(|(_, update)| update.clone())
            .collect();
        if !updates.is_empty() {
            self.runtime.latents()?.pool.validate_updates(&updates)?;
        }
        self.publications.validate(&self.storage.tensors)?;
        self.leave_latents()?;
        self.seal()?;
        for state in &mut self.calls {
            state.output.reports_output = self.reports;
        }
        for (index, update) in &self.updates {
            self.calls[*index].output.apply_latent_update(update);
        }
        let pending: Vec<_> = self
            .plan
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id, call.advances_state()))
            .collect();
        self.requests.validate_pending(&pending)?;

        // Every owner accepted its update. Any failure from here is fatal.
        let writes = std::mem::take(&mut self.writes);
        self.storage.tensors.commit_writes(&writes).map_err(fatal)?;
        if !updates.is_empty() {
            let cancelled = self
                .runtime
                .latents_mut()
                .map_err(fatal)?
                .pool
                .apply_updates(&updates);
            for transfer in cancelled {
                // Released trajectories' pending imports end; their pages
                // retire with the transfer.
                transfer.read().cancel().map_err(fatal)?;
            }
        }
        self.publications
            .register(&self.storage.tensors)
            .map_err(fatal)?;
        self.requests.add_pending(&pending).map_err(fatal)
    }

    /// Seal the lease on the batch's stream and retire its KV accesses and
    /// output storage once the fences complete; wake the service then.
    pub(super) fn seal(&self) -> Result<()> {
        seal_lease(
            &self.storage.events,
            &self.buffer,
            self.stream.as_ref(),
            &self.retirement,
        )
    }

    /// End the batch's use of the latent pool on its stream.
    fn leave_latents(&mut self) -> Result<()> {
        if std::mem::take(&mut self.latents_entered) {
            self.runtime.latents_mut()?.leave(self.stream.as_ref())?;
        }
        Ok(())
    }

    /// Retire every reservation after a failure before the commit point;
    /// storage returns once the fences recorded here complete.
    pub(super) fn discard(&mut self) -> Result<()> {
        let left = self.leave_latents();
        let revoked = self.publications.revoke();
        let exported = std::mem::take(&mut self.latent_exports);
        if !exported.is_empty()
            && let Ok(latents) = self.runtime.latents()
        {
            // The versions retire once their transports drop them.
            latents.pool.release_exports(&exported);
        }
        let stream = self.stream.as_ref();
        let reads = std::mem::take(&mut self.reads);
        let completed = self.storage.tensors.complete_reads(&reads, stream, &[]);
        let writes = std::mem::take(&mut self.writes);
        let abandoned = self.storage.tensors.abandon_writes(&writes);
        for encoding in self.encodings.drain(..) {
            // Unconfigured encodings return their lane capacity.
            let _ = encoding.task.cancel(true);
        }
        let sealed = if lock(&self.buffer).sealed() {
            Ok(())
        } else {
            self.seal()
        };
        lock(&self.buffer).abandon();
        left.and(revoked).and(completed).and(abandoned).and(sealed)
    }
}
