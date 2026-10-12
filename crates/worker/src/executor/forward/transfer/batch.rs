//! A transfer batch between its jobs: reservation, commit and retirement.
//!
//! A batch is at most two jobs of its lane. The first exists when a call has
//! a host-gated predicate: it reads them ([`Opening`]), and a false one
//! skips its call, which only publishes a false completion flag. The last
//! job carries everything the batch issues ([`TransferBatch`]): the start of
//! its output lease, the completion flags, the transfers with their exports,
//! the readers' fences and the lease's seal.
//!
//! The completion of that job is the batch's commit point
//! ([`Recorded::commit`]). Exports are located only there, so nothing of an
//! export reaches another process before the device work it describes is
//! complete. Every owner's update is then validated (products and their
//! exports, KV exports and installs, pending calls) and they become visible
//! together. A failure before that point retires everything the batch
//! reserved ([`Recorded::discard`]), also from the job's completion on: the
//! job still carries what the batch recorded before it failed (KV table
//! installs and unit resets do their host bookkeeping when recorded, so
//! their device work follows), and an export is revoked only after the
//! copies into it ran. A failure after the commit point is fatal.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BufferId, CallStatus, DType, ForwardStats, RequestKey, TensorRef,
};

use super::super::launch::{Launched, flag, lock, reports_output, request_flag, seal};
use super::super::products::Publications;
use super::kv::{self, Exporting, Installation, Published};
use super::latents::VersionExport;
use crate::executor::PendingOutput;
use crate::executor::forward::batch::image::ImageParts;
use crate::ffi::Device;
use crate::lane::{Completion, Job, Recorder};
use crate::request::RequestPool;
use crate::storage::{
    OutputLease, OutputPlacement, OutputPool, ProductBuffer, ProductExport, ProductRead, Scalar,
    TensorStore,
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
pub(crate) struct Failure {
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

/// A batch whose outputs are bound, before its output storage is leased:
/// its host-gated predicates decide first which calls are active.
pub(crate) struct Opening {
    calls: Vec<CallState>,
    /// The calls whose predicate the opening job reads, in the order of its
    /// reads.
    predicated: Vec<usize>,
    /// This rank reports the component's host results.
    reports: bool,
    started: Instant,
}

impl Opening {
    /// Bind the batch's outputs to their requests and record in `job` the
    /// read of every host-gated predicate: the job's completion carries one
    /// byte each.
    pub(super) fn open(
        requests: &mut RequestPool,
        tensors: &TensorStore,
        info: &uniserve_ipc::WorkerInfo,
        rank: usize,
        plan: &BatchPlan,
        device: Option<i32>,
        job: Recorder<'_, '_>,
    ) -> Result<Self> {
        let started = Instant::now();
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

        let mut predicated = Vec::new();
        for (index, call) in plan.calls.iter().enumerate() {
            if let Some(predicate) = &call.predicate
                && predicate.dtype == DType::U8
            {
                request_flag(tensors, call, predicate, device, job)?;
                predicated.push(index);
            }
        }
        Ok(Self {
            calls,
            predicated,
            reports,
            started,
        })
    }

    /// Whether the opening job reads a predicate.
    pub(super) fn predicated(&self) -> bool {
        !self.predicated.is_empty()
    }

    /// Apply the opening job's completion: a call whose predicate read
    /// false is skipped.
    pub(super) fn predicates(&mut self, completion: &mut Completion) -> Result<()> {
        for index in std::mem::take(&mut self.predicated) {
            if !flag(completion)? {
                let state = &mut self.calls[index];
                state.active = false;
                state.output.output.status = CallStatus::Predicated;
            }
        }
        Ok(())
    }
}

/// A batch between two of its jobs.
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
/// completion: its reservations, and the exports whose locations exist
/// once the job was performed.
pub(crate) struct Recorded {
    pub(super) calls: Vec<CallState>,
    buffer: OutputLease,
    pub(super) writes: Vec<ProductBuffer>,
    reports: bool,
    /// Product exports, each with its call and the write it publishes.
    pub(super) products: VecDeque<(usize, ProductBuffer, ProductExport)>,
    /// Trajectory versions exported from the latent pool, with their calls.
    pub(super) versions: VecDeque<(usize, VersionExport)>,
    /// KV exports, and the installs resident once the batch commits.
    pub(super) exports: VecDeque<Exporting>,
    pub(super) installations: Vec<Installation>,
    started: Instant,
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
pub(super) struct TransferBatch<'a, 'j> {
    pub(super) storage: &'a mut RankStorage,
    pub(super) requests: &'a mut RequestPool,
    pub(super) plan: &'a BatchPlan,
    /// The device and stream the batch's transfers run on.
    pub(super) device: Option<i32>,
    stream: Option<&'a Stream>,
    /// The job everything the batch issues is recorded into.
    job: &'a Job<'j>,
    pub(super) reads: Vec<Arc<ProductRead>>,
    /// Signals the physical end of the batch's device work.
    retirement: Arc<Retirement>,
    pub(super) recorded: Recorded,
}

impl<'a, 'j> TransferBatch<'a, 'j> {
    /// Lease the opened batch's output storage and record the start of its
    /// device work in `job`.
    #[expect(clippy::too_many_arguments, reason = "the batch's borrowed owners")]
    pub(super) fn begin(
        opening: Opening,
        outputs: &mut OutputPool,
        storage: &'a mut RankStorage,
        requests: &'a mut RequestPool,
        plan: &'a BatchPlan,
        device: Option<i32>,
        stream: Option<&'a Stream>,
        job: &'a Job<'j>,
    ) -> Result<Self> {
        let Opening {
            calls,
            reports,
            started,
            ..
        } = opening;
        let devices: Vec<i32> = device.into_iter().collect();
        let buffer = outputs.lease(plan.calls.len(), plan.calls.len(), &devices, false)?;
        if stream.is_some() {
            lock(&buffer).begin_device(device, job.on(stream), &storage.events)?;
        }
        Ok(Self {
            storage,
            requests,
            plan,
            device,
            stream,
            job,
            reads: Vec::new(),
            retirement: Arc::new(Retirement::default()),
            recorded: Recorded {
                calls,
                buffer,
                writes: Vec::new(),
                reports,
                products: VecDeque::new(),
                versions: VecDeque::new(),
                exports: VecDeque::new(),
                installations: Vec::new(),
                started,
            },
        })
    }

    /// Where the batch issues its stream work: its job.
    pub(super) fn issue(&self) -> Recorder<'a, 'j> {
        self.job.on(self.stream)
    }

    /// Record an export with `export`, which issues to the recorder it is
    /// given. When it fails, the work it recorded is given up, since its
    /// copies address storage the failed export gave back; its event
    /// operations stay with the batch's job.
    pub(super) fn exporting<T>(
        &mut self,
        export: impl FnOnce(&mut Self, Recorder<'_, 'j>) -> Result<T>,
    ) -> Result<T> {
        let exported = Job::new();
        let result = export(self, exported.on(self.stream));
        self.job.append(match result {
            Ok(_) => exported,
            Err(_) => exported.orders_only(),
        });
        result
    }

    /// The batch's product device.
    pub(super) fn target(&self) -> Device {
        self.device.map_or(Device::Cpu, Device::Cuda)
    }

    /// Indices of the calls that execute.
    pub(super) fn active(&self) -> Vec<usize> {
        let calls = &self.recorded.calls;
        (0..calls.len())
            .filter(|&index| calls[index].active)
            .collect()
    }

    /// Request slots of the batch's calls, for relayed outputs.
    pub(super) fn slots(&self) -> HashMap<RequestKey, usize> {
        self.plan
            .calls
            .iter()
            .zip(&self.recorded.calls)
            .map(|(call, state)| (call.request_key, state.slot))
            .collect()
    }

    /// Bind every call's completion flag; skipped calls publish false.
    pub(super) fn bind_flags(&mut self) -> Result<()> {
        let issue = self.issue();
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
            self.recorded.writes.push(write.clone());
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
            let state = &mut self.recorded.calls[index];
            if !state.active {
                self.storage
                    .tensors
                    .write_scalar(&write, Scalar::Int(0), issue)?;
            }
            state.completion_write = Some(write);
        }
        Ok(())
    }

    /// Record the end of the batch's device work: its readers' fences, the
    /// completion flags of the executed calls and the seal of its lease.
    /// The updates that depend on no export's locations are validated here;
    /// the others at the job's completion ([`Recorded::commit`]).
    pub(super) fn seal(&mut self) -> Result<()> {
        let issue = self.issue();
        let recorded = &self.recorded;
        // A copy's read ends at the fence of the write it fed when every
        // read has one; otherwise a reader fence is recorded.
        let reads = std::mem::take(&mut self.reads);
        let after: Vec<ProductBuffer> = recorded
            .writes
            .iter()
            .filter(|write| {
                recorded.calls.iter().all(|state| {
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
        self.storage.tensors.complete_reads(&reads, issue, &after)?;

        // Completion flags of executed calls become true.
        for state in recorded.calls.iter().filter(|state| state.active) {
            if let Some(write) = &state.completion_write {
                self.storage
                    .tensors
                    .write_scalar(write, Scalar::Int(1), issue)?;
            }
        }
        self.storage.tensors.validate_writes(&recorded.writes)?;
        seal(
            &self.storage.events,
            &recorded.buffer,
            issue,
            &self.retirement,
        )?;
        self.requests.validate_pending(&pending(self.plan))
    }

    /// End the recording after `result`. A failed batch still records what
    /// ends its device work (the fences of its reads, the seal of its
    /// lease), so the storage they guard returns once its job ran.
    pub(super) fn end(mut self, result: Result<()>) -> (Recorded, Option<Failure>) {
        let Err(error) = result else {
            return (self.recorded, None);
        };
        let issue = self.issue();
        let reads = std::mem::take(&mut self.reads);
        let completed = self.storage.tensors.complete_reads(&reads, issue, &[]);
        let sealed = if lock(&self.recorded.buffer).sealed() {
            Ok(())
        } else {
            seal(
                &self.storage.events,
                &self.recorded.buffer,
                issue,
                &self.retirement,
            )
        };
        let failure = match completed.and(sealed) {
            Ok(()) => error.into(),
            // Storage the cleanup could not fence stays reserved; the rank
            // cannot reuse it safely.
            Err(cleanup) => fatal(cleanup),
        };
        (self.recorded, Some(failure))
    }
}

/// The pending state of every call of `plan`.
fn pending(plan: &BatchPlan) -> Vec<(RequestKey, uniserve_ipc::CallId, bool)> {
    plan.calls
        .iter()
        .map(|call| (call.request_key, call.call_id, call.advances_state()))
        .collect()
}

impl Recorded {
    /// Apply the completion of the batch's last job: locate its exports,
    /// validate every owner's update, then commit them together.
    pub(super) fn commit(
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
                Ok(()) => error.into(),
                // Storage the cleanup could not retire stays reserved; the
                // rank cannot reuse it safely.
                Err(cleanup) => fatal(cleanup),
            });
        }
        for state in &mut self.calls {
            state.output.reports_output = self.reports;
        }

        // Every owner accepted its update. Any failure from here is fatal.
        let Located {
            mut publications,
            published,
            ..
        } = located;
        storage.tensors.commit_writes(&self.writes).map_err(fatal)?;
        publications.register(&storage.tensors).map_err(fatal)?;
        kv::apply(storage, published, self.installations).map_err(fatal)?;
        requests.add_pending(&pending(plan)).map_err(fatal)?;

        let mut stats = ForwardStats::default();
        stats.component_us.insert(
            "transfer_execute".into(),
            self.started.elapsed().as_micros() as u64,
        );
        Ok(Launched {
            outputs: self.calls.into_iter().map(|state| state.output).collect(),
            buffer: self.buffer,
            started: self.started,
            stats,
            timer: "transfer_materialize",
            host: None,
        })
    }

    /// Retire every reservation of a batch that failed while it recorded
    /// its job, once that job completed.
    pub(super) fn discard(
        self,
        storage: &mut RankStorage,
        image: Option<&mut ImageParts<'_>>,
    ) -> Result<()> {
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
        while let Some((index, write, export)) = self.products.pop_front() {
            let (export, locations) = storage.tensors.located(&write, export)?;
            located.publications.accept(
                write.id(),
                locations,
                export,
                &mut self.calls[index].output,
            )?;
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
                version.publish(
                    pool,
                    &mut located.publications,
                    &mut self.calls[index].output,
                )?;
            }
        }

        while let Some(export) = self.exports.pop_front() {
            let export = export.locate(storage)?;
            self.calls[export.call].output.output.kv_output = Some(export.transfer.clone());
            located.published.push(export);
        }
        Ok(())
    }

    /// Check, before the commit point, the updates that name the located
    /// exports.
    fn validate(
        &self,
        storage: &mut RankStorage,
        plan: &BatchPlan,
        located: &Located,
    ) -> Result<()> {
        {
            let output = lock(&self.buffer);
            for (state, call) in self.calls.iter().zip(&plan.calls) {
                state.output.validate_output(&output, call)?;
            }
        }
        located.publications.validate(&storage.tensors)?;
        kv::validate(storage, &located.published, &self.installations)
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
            self.calls[export.call].output.output.kv_output = None;
        }
        let mut unpublished = kv::revoke(storage, published);
        for export in self.exports.drain(..) {
            unpublished = unpublished.and(export.revoke(storage));
        }

        let abandoned = storage.tensors.abandon_writes(&self.writes);
        lock(&self.buffer).abandon();
        revoked.and(released).and(unpublished).and(abandoned)
    }
}
