//! A transfer batch's reservation, commit and retirement.
//!
//! [`TransferBatch::open`] binds every call's pending output, resolves
//! host-gated predicates (a false one skips its call, which only publishes
//! a false completion flag) and leases the batch's output storage on its
//! stream. After the transfers, [`TransferBatch::commit`] validates every
//! owner's update (products and their exports, KV exports and installs,
//! pending calls) and makes them visible together; a failure before that
//! point retires everything the batch reserved ([`TransferBatch::discard`]),
//! a failure after it is fatal.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use uniserve_cuda::Stream;
use uniserve_ipc::{Batch as BatchPlan, BufferId, CallStatus, DType, RequestKey, TensorRef};

use super::super::image::ImageRuntime;
use super::super::launch::{lock, read_flag, reports_output, seal};
use super::super::products::Publications;
use super::kv::{self, Installation, Published};
use crate::executor::PendingOutput;
use crate::ffi::Device;
use crate::request::RequestPool;
use crate::storage::{
    OutputLease, OutputPlacement, OutputPool, ProductBuffer, ProductRead, Scalar,
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

fn fatal(error: Error) -> Failure {
    Failure {
        error,
        committed: true,
    }
}

/// A transfer batch's borrowed owners and the resources it reserved.
pub(super) struct TransferBatch<'a> {
    pub(super) storage: &'a mut RankStorage,
    pub(super) requests: &'a mut RequestPool,
    pub(super) plan: &'a BatchPlan,
    /// The device and stream the batch's transfers run on.
    pub(super) device: Option<i32>,
    pub(super) stream: Option<Stream>,
    /// This rank reports the component's host results.
    pub(super) reports: bool,
    pub(super) calls: Vec<CallState>,
    pub(super) buffer: OutputLease,
    pub(super) writes: Vec<ProductBuffer>,
    pub(super) reads: Vec<Arc<ProductRead>>,
    /// Product exports, registered once the batch commits.
    pub(super) publications: Publications,
    /// Trajectory versions exported from the latent pool, released from it
    /// if the batch fails before committing.
    pub(super) latent_exports: Vec<BufferId>,
    /// KV exports and installs, resident once the batch commits.
    pub(super) published: Vec<Published>,
    pub(super) installations: Vec<Installation>,
    /// Signals the physical end of the batch's device work.
    pub(super) retirement: Arc<Retirement>,
    pub(super) started: Instant,
}

impl<'a> TransferBatch<'a> {
    /// Bind the batch's outputs, skip calls whose host predicate is false
    /// and lease its output storage on `stream`.
    #[expect(clippy::too_many_arguments, reason = "the batch's borrowed owners")]
    pub(super) fn open(
        outputs: &mut OutputPool,
        storage: &'a mut RankStorage,
        requests: &'a mut RequestPool,
        info: &uniserve_ipc::WorkerInfo,
        rank: usize,
        plan: &'a BatchPlan,
        device: Option<i32>,
        stream: Option<Stream>,
        started: Instant,
    ) -> Result<Self> {
        let first = plan
            .calls
            .first()
            .ok_or_else(|| Error::Invariant("a transfer batch has calls".into()))?;
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
        let buffer = outputs.lease(plan.calls.len(), plan.calls.len(), &devices, false)?;
        if let Some(stream) = &stream {
            lock(&buffer).begin_device(device, stream, &storage.events)?;
        }
        Ok(Self {
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
            publications: Publications::default(),
            latent_exports: Vec::new(),
            published: Vec::new(),
            installations: Vec::new(),
            retirement: Arc::new(Retirement::default()),
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

    /// Request slots of the batch's calls, for relayed outputs.
    pub(super) fn slots(&self) -> HashMap<RequestKey, usize> {
        self.plan
            .calls
            .iter()
            .zip(&self.calls)
            .map(|(call, state)| (call.request_key, state.slot))
            .collect()
    }

    /// Bind every call's completion flag; skipped calls publish false now.
    pub(super) fn bind_flags(&mut self) -> Result<()> {
        let target = self.target();
        let slots = self.slots();
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
        // A copy's read ends at the fence of the write it fed when every
        // read has one; otherwise a reader fence is recorded.
        let reads = std::mem::take(&mut self.reads);
        let after: Vec<ProductBuffer> = self
            .writes
            .iter()
            .filter(|write| {
                self.calls.iter().all(|state| {
                    state
                        .completion_write
                        .as_ref()
                        .is_none_or(|flag| flag.id() != write.id())
                })
            })
            .cloned()
            .collect();
        let after = if after.len() == reads.len() {
            after
        } else {
            Vec::new()
        };
        self.storage
            .tensors
            .complete_reads(&reads, stream, &after)?;

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
        self.publications.validate(&self.storage.tensors)?;
        kv::validate(self.storage, &self.published, &self.installations)?;
        seal(&self.storage.events, &self.buffer, stream, &self.retirement)?;
        for state in &mut self.calls {
            state.output.reports_output = self.reports;
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
        self.publications
            .register(&self.storage.tensors)
            .map_err(fatal)?;
        let published = std::mem::take(&mut self.published);
        let installations = std::mem::take(&mut self.installations);
        kv::apply(self.storage, published, installations).map_err(fatal)?;
        self.requests.add_pending(&pending).map_err(fatal)
    }

    /// Retire every reservation after a failure before the commit point;
    /// storage returns once the fences recorded here complete. Trajectory
    /// versions exported from `image`'s latent pool are released from it.
    pub(super) fn discard(&mut self, image: Option<&mut ImageRuntime>) -> Result<()> {
        let revoked = self.publications.revoke();
        let exported = std::mem::take(&mut self.latent_exports);
        let released = match (exported.is_empty(), image) {
            (true, _) => Ok(()),
            (false, Some(image)) => image
                .latent_pool()
                .map(|(pool, _)| pool.release_exports(&exported)),
            (false, None) => Err(Error::Invariant(
                "exported trajectory versions lost their latent pool".into(),
            )),
        };
        let published = std::mem::take(&mut self.published);
        for export in &published {
            self.calls[export.call].output.output.kv_output = None;
        }
        let unpublished = kv::revoke(self.storage, published);
        let stream = self.stream.as_ref();
        let reads = std::mem::take(&mut self.reads);
        let completed = self.storage.tensors.complete_reads(&reads, stream, &[]);
        let writes = std::mem::take(&mut self.writes);
        let abandoned = self.storage.tensors.abandon_writes(&writes);
        let sealed = if lock(&self.buffer).sealed() {
            Ok(())
        } else {
            seal(&self.storage.events, &self.buffer, stream, &self.retirement)
        };
        lock(&self.buffer).abandon();
        revoked
            .and(released)
            .and(unpublished)
            .and(completed)
            .and(abandoned)
            .and(sealed)
    }
}
