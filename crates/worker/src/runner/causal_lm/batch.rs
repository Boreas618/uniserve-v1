//! A token batch's reservation and retirement.
//!
//! [`Opening::open`] binds every call's pending output and records the
//! reads of the batch's host-gated predicates. [`Batch::lease`] leases the
//! batch's output storage. [`Batch::reserve`] installs the active calls'
//! block tables, resets newly assigned units, retains the cache extents the
//! batch reads and writes until its device work retires, binds its scalar
//! products and leases its relayed predicates. [`Batch::discard`] retires
//! all of it after a failure before the commit point.
//!
//! None of this issues on the batch's stream: every stream operation is
//! recorded in the batch's job. A batch is a sequence of jobs, and between
//! two of them it is a [`TokenBatch`], which borrows nothing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use uniserve_cuda::Stream;
use uniserve_ipc::{Batch as BatchPlan, CallStatus, DType, ForwardMode, RequestKey, TensorRef};

use super::calls::{self, Prepared};
use super::{COMPLETION_FIELDS, CausalLMRunners};
use crate::executor::PendingOutput;
use crate::executor::forward::batch::image::ImageParts;
use crate::executor::forward::launch::{lock, reports_output, seal};
use crate::ffi::Device;
use crate::lane::{Completion, Job};
use crate::parallel::ExpertUnion;
use crate::request::RequestPool;
use crate::sampling::SamplingMetadata;
use crate::storage::{OutputLease, OutputPlacement, ProductBuffer, ProductRead, Scalar};
use crate::transfer::Retirement;
use crate::worker::RankStorage;
use crate::{Error, Result};

/// One call's state through its batch.
pub(super) struct CallState {
    pub(super) output: PendingOutput,
    pub(super) slot: usize,
    /// False when the call's host predicate skipped it.
    pub(super) active: bool,
    pub(super) prepared: Option<Prepared>,
    pub(super) token_write: Option<ProductBuffer>,
    pub(super) transition_write: Option<ProductBuffer>,
    pub(super) completion_write: Option<ProductBuffer>,
    /// The runner of the microbatch the call's row runs in.
    pub(super) microbatch: usize,
    /// The call's input index among its invocation's sampled tasks.
    pub(super) task: Option<usize>,
    pub(super) metadata: Option<SamplingMetadata>,
    /// The call's prompt chunks, in order.
    pub(super) prompt: Vec<super::step::PromptScore>,
}

/// The owners a token batch borrows while one of its jobs is prepared or
/// its completion applied.
pub(super) struct Owners<'a> {
    pub(super) runtime: &'a mut CausalLMRunners,
    /// The image runtime, for prompts placing image features.
    pub(super) image: Option<ImageParts<'a>>,
    pub(super) storage: &'a mut RankStorage,
    pub(super) requests: &'a mut RequestPool,
    pub(super) plan: &'a BatchPlan,
    /// The rank's expert union, when its forwards are expert steps.
    pub(super) experts: Option<Arc<ExpertUnion>>,
}

/// A token batch whose outputs are bound, before its output storage is
/// leased: its host-gated predicates decide first which calls are active.
pub(crate) struct Opening {
    /// The runner group serving the batch.
    group: usize,
    /// The group's lane stream.
    stream: Option<Stream>,
    device: Option<i32>,
    mode: ForwardMode,
    /// This rank reports the component's host results.
    reports: bool,
    calls: Vec<CallState>,
    /// The calls whose host-gated predicate the opening job reads, in the
    /// order of its reads.
    predicated: Vec<usize>,
    started: Instant,
}

/// A token batch between two of its jobs: what it reserved and each call's
/// state.
pub(crate) struct TokenBatch {
    pub(super) group: usize,
    pub(super) stream: Option<Stream>,
    pub(super) device: Option<i32>,
    pub(super) mode: ForwardMode,
    pub(super) reports: bool,
    pub(super) calls: Vec<CallState>,
    pub(super) buffer: OutputLease,
    pub(super) writes: Vec<ProductBuffer>,
    pub(super) reads: Vec<Arc<ProductRead>>,
    pub(super) retirement: Arc<Retirement>,
    pub(super) started: Instant,
}

/// A token batch's borrowed owners, the resources it reserved and the job
/// that records its stream work.
pub(super) struct Batch<'a> {
    pub(super) runtime: &'a mut CausalLMRunners,
    pub(super) image: Option<ImageParts<'a>>,
    pub(super) storage: &'a mut RankStorage,
    pub(super) requests: &'a mut RequestPool,
    pub(super) plan: &'a BatchPlan,
    pub(super) experts: Option<Arc<ExpertUnion>>,
    pub(super) group: usize,
    pub(super) stream: Option<Stream>,
    pub(super) device: Option<i32>,
    pub(super) mode: ForwardMode,
    pub(super) reports: bool,
    pub(super) calls: Vec<CallState>,
    pub(super) buffer: OutputLease,
    pub(super) writes: Vec<ProductBuffer>,
    pub(super) reads: Vec<Arc<ProductRead>>,
    /// Signals the physical end of the batch's cache accesses.
    pub(super) retirement: Arc<Retirement>,
    pub(super) started: Instant,
    /// The batch's next job on its lane.
    pub(super) job: Job,
}

impl Opening {
    /// Bind the batch's outputs to their requests, then record in `job`
    /// the batch's order after the previous one and the read of every
    /// host-gated predicate: the job's completion carries one byte each.
    pub(super) fn open(
        owners: &mut Owners<'_>,
        info: &uniserve_ipc::WorkerInfo,
        rank: usize,
        job: &Job,
    ) -> Result<Self> {
        let started = Instant::now();
        let plan = owners.plan;
        let runtime = &mut *owners.runtime;
        let first = plan
            .calls
            .first()
            .ok_or_else(|| Error::Invariant("a token batch has calls".into()))?;
        let mode = calls::kind(first)?;
        if plan
            .calls
            .iter()
            .any(|call| call.component != first.component)
        {
            return Err(Error::Invalid("a token batch serves one component".into()));
        }
        let group = runtime.route(&first.component, first.code).ok_or_else(|| {
            Error::Unsupported(format!(
                "component {:?} has no text binding for {}",
                first.component,
                first.code.as_str()
            ))
        })?;
        let lane = runtime.groups[group][0];
        let device = runtime.runners[lane].device();
        let stream = runtime.runners[lane].stream().cloned();
        let reports = reports_output(info, rank, &first.component)?;

        // Output rows, bound to their requests' live epochs and slots.
        let rows = plan
            .calls
            .iter()
            .map(|call| {
                let request = owners.requests.get(call.request_key.request_id.0)?;
                Ok((call.request_key, call.call_id, request.slot()))
            })
            .collect::<Result<Vec<_>>>()?;
        let bound = owners.requests.bind_calls(&rows)?;
        let mut calls = Vec::with_capacity(plan.calls.len());
        for (index, (call, request)) in plan.calls.iter().zip(&bound).enumerate() {
            calls.push(CallState {
                output: PendingOutput::new(call, request.progress()?, index)?,
                slot: request.slot(),
                active: true,
                prepared: None,
                token_write: None,
                transition_write: None,
                completion_write: None,
                microbatch: lane,
                task: None,
                metadata: None,
                prompt: Vec::new(),
            });
        }

        // Host-gated predicates resolve before any storage is reserved; a
        // false one skips its call. Each is read after its producer's
        // fence, and its lease ends behind the read.
        let issue = job.on(stream.as_ref());
        runtime.order.wait(issue)?;
        let tensors = &owners.storage.tensors;
        let mut predicated = Vec::new();
        for (index, call) in plan.calls.iter().enumerate() {
            let Some(predicate) = call
                .predicate
                .as_ref()
                .filter(|value| value.dtype == DType::U8)
            else {
                continue;
            };
            let read = tensors.consume(predicate, call.call_id, device.map(Device::Cuda), issue)?;
            issue.read(read.value().address(), 1);
            tensors.complete_reads(&[read], issue, &[])?;
            predicated.push(index);
        }
        Ok(Self {
            group,
            stream,
            device,
            mode,
            reports,
            calls,
            predicated,
            started,
        })
    }

    /// Whether the opening job reads predicates, so the batch continues
    /// only once it completed.
    pub(super) fn reads_predicates(&self) -> bool {
        !self.predicated.is_empty()
    }

    /// Apply the opening job's completion: a call whose predicate read
    /// false is skipped.
    pub(super) fn predicates(&mut self, completion: &mut Completion) -> Result<()> {
        for index in std::mem::take(&mut self.predicated) {
            let active = match completion.read()?[..] {
                [0] => false,
                [1] => true,
                _ => {
                    return Err(Error::Invalid(
                        "call predicate is not a canonical boolean".into(),
                    ));
                }
            };
            if !active {
                let state = &mut self.calls[index];
                state.active = false;
                state.output.output.status = CallStatus::Predicated;
            }
        }
        Ok(())
    }
}

impl<'a> Batch<'a> {
    /// Lease the opened batch's output storage and begin its job.
    pub(super) fn lease(owners: Owners<'a>, opening: Opening, job: Job) -> Result<Self> {
        let Opening {
            group,
            stream,
            device,
            mode,
            reports,
            calls,
            started,
            ..
        } = opening;
        let plan = owners.plan;

        // The output lease: completion fields per call plus payload words.
        let words = plan
            .calls
            .iter()
            .fold(COMPLETION_FIELDS * plan.calls.len(), |total, call| {
                total.saturating_add(
                    usize::try_from(call.bounds.max_completion_bytes.div_ceil(4))
                        .unwrap_or(usize::MAX),
                )
            });
        let devices: Vec<i32> = device.into_iter().collect();
        let buffer =
            owners
                .runtime
                .outputs
                .lease(plan.calls.len(), words.max(1), &devices, false)?;
        if let Some(stream) = &stream {
            lock(&buffer).begin_device(device, job.on(Some(stream)), &owners.storage.events)?;
        }
        Ok(Self::resume(
            owners,
            TokenBatch {
                group,
                stream,
                device,
                mode,
                reports,
                calls,
                buffer,
                writes: Vec::new(),
                reads: Vec::new(),
                retirement: Arc::new(Retirement::default()),
                started,
            },
            job,
        ))
    }

    /// Continue a batch with its owners borrowed again, recording into
    /// `job`.
    pub(super) fn resume(owners: Owners<'a>, batch: TokenBatch, job: Job) -> Self {
        Self {
            runtime: owners.runtime,
            image: owners.image,
            storage: owners.storage,
            requests: owners.requests,
            plan: owners.plan,
            experts: owners.experts,
            group: batch.group,
            stream: batch.stream,
            device: batch.device,
            mode: batch.mode,
            reports: batch.reports,
            calls: batch.calls,
            buffer: batch.buffer,
            writes: batch.writes,
            reads: batch.reads,
            retirement: batch.retirement,
            started: batch.started,
            job,
        }
    }

    /// Give the owners back: the job to submit, the batch as it waits for
    /// the job's completion, and the owners.
    pub(super) fn park(self) -> (Job, TokenBatch, Owners<'a>) {
        (
            self.job,
            TokenBatch {
                group: self.group,
                stream: self.stream,
                device: self.device,
                mode: self.mode,
                reports: self.reports,
                calls: self.calls,
                buffer: self.buffer,
                writes: self.writes,
                reads: self.reads,
                retirement: self.retirement,
                started: self.started,
            },
            Owners {
                runtime: self.runtime,
                image: self.image,
                storage: self.storage,
                requests: self.requests,
                plan: self.plan,
                experts: self.experts,
            },
        )
    }

    /// Install tables, reset units, retain cache extents, bind products and
    /// lease relayed predicates of the active calls.
    pub(super) fn reserve(&mut self) -> Result<()> {
        let stream = self.job.on(self.stream.as_ref());
        let plan = self.plan;
        let prefix_cache = self
            .storage
            .prefix_cache
            .as_mut()
            .ok_or_else(|| Error::Unsupported("text calls require a prefix cache".into()))?;

        // Tables and new units precede any of the batch's writes.
        let active: HashSet<u32> = self
            .calls
            .iter()
            .filter(|state| state.active)
            .map(|state| state.slot as u32)
            .collect();
        let assignments: Vec<_> = plan
            .block_tables
            .iter()
            .filter(|table| active.contains(&table.request_pool_idx))
            .cloned()
            .collect();
        prefix_cache.install(&assignments, stream)?;
        let units: Vec<u32> = plan
            .new_cache_units
            .iter()
            .filter(|allocation| active.contains(&allocation.request_pool_idx))
            .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
            .collect();
        if !units.is_empty() {
            let lane = self.runtime.groups[self.group][0];
            prefix_cache.recycle(&units, &self.runtime.columns[lane], stream)?;
        }

        // Each active call has one forward row on its own slot; a context
        // prefill has one per prompt run and vision block (checked against
        // its rows when they are built). A queued token relay declares a
        // capacity bound: its device state decides the visible length when
        // it executes.
        let forward = &plan.forward;
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); plan.calls.len()];
        for (row, &call) in forward.call_indices.iter().enumerate() {
            rows.get_mut(call as usize)
                .ok_or_else(|| Error::Invalid("forward row names no call".into()))?
                .push(row);
        }
        for ((call, state), call_rows) in plan.calls.iter().zip(&self.calls).zip(&rows) {
            if !state.active {
                continue;
            }
            let (&first, rest) = call_rows
                .split_first()
                .ok_or_else(|| Error::Invalid("a token call has no forward row".into()))?;
            if !rest.is_empty() && !call.writes_context() {
                return Err(Error::Invalid(
                    "a token call has exactly one forward row".into(),
                ));
            }
            if call_rows
                .iter()
                .any(|&row| forward.request_pool_indices[row] as usize != state.slot)
            {
                return Err(Error::Invalid(
                    "a token call's forward row reads another slot".into(),
                ));
            }
            let declared =
                u64::from(forward.seq_lens[first].saturating_sub(forward.query_lens[first]));
            let visible = state.output.progress.cache_visible_len;
            let relayed = call
                .predicate
                .as_ref()
                .is_some_and(|value| value.dtype == DType::I64);
            if declared < visible || (!relayed && declared != visible) {
                return Err(Error::Invalid(
                    "forward row sequence length disagrees with execution state".into(),
                ));
            }
            for &row in call_rows {
                // Readers retain their prefix, writers also their query
                // tokens; a windowed group starts at its first resident page.
                let length = u64::from(if forward.write_cache[row] {
                    forward.seq_lens[row]
                } else {
                    forward.seq_lens[row] - forward.query_lens[row]
                });
                let mut spans = Vec::new();
                let tables = prefix_cache.tables();
                for group in 0..tables.groups().len() as u32 {
                    let table = tables.table(state.slot as u32, group)?;
                    let start = u64::from(table.start_page) * u64::from(table.shape.page_tokens);
                    if length > start {
                        spans.extend(table.spans(start, length - start)?);
                    }
                }
                prefix_cache.manager().retain_execution(
                    call.request_key,
                    &spans,
                    Arc::clone(&self.retirement),
                );
            }
        }

        self.bind_products()?;
        self.lease_predicates()
    }

    /// Bind token relays of active calls and completion and transition flags
    /// of every call; skipped calls publish false flags now.
    fn bind_products(&mut self) -> Result<()> {
        let plan = self.plan;
        let target = self.device.map_or(Device::Cpu, Device::Cuda);
        let slots: HashMap<RequestKey, usize> = plan
            .calls
            .iter()
            .zip(&self.calls)
            .map(|(call, state)| (call.request_key, state.slot))
            .collect();
        let mut groups: indexmap::IndexMap<(DType, String), Vec<(TensorRef, Device)>> =
            indexmap::IndexMap::new();
        for (call, state) in plan.calls.iter().zip(&self.calls) {
            if !call.outputs.is_empty()
                || call.image_output.is_some()
                || call.encoder_output.is_some()
            {
                return Err(Error::Unsupported(
                    "token calls publish no tensor products besides their relays".into(),
                ));
            }
            for scalar in [
                call.token_output.as_ref().filter(|_| state.active),
                call.completion_output.as_ref(),
                call.transition_output.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                // One relay group per dtype and shape.
                groups
                    .entry((scalar.dtype, format!("{:?}", scalar.shape_bound)))
                    .or_default()
                    .push((scalar.clone(), target));
            }
        }
        let groups: Vec<_> = groups.into_values().collect();
        let bound = self.storage.tensors.bind_output_groups(
            &groups,
            OutputPlacement {
                request_slots: Some(&slots),
                ..OutputPlacement::default()
            },
        )?;
        let owners: HashMap<RequestKey, usize> = plan
            .calls
            .iter()
            .enumerate()
            .map(|(index, call)| (call.request_key, index))
            .collect();
        for write in bound.into_iter().flatten() {
            let id = write.id();
            self.writes.push(write.clone());
            let index = owners[&id.owner];
            let call = &plan.calls[index];
            let state = &mut self.calls[index];
            let names = |value: &Option<TensorRef>| {
                value.as_ref().is_some_and(|value| value.buffer_id() == id)
            };
            if names(&call.token_output) {
                state.token_write = Some(write);
            } else if names(&call.transition_output) {
                state.transition_write = Some(write);
            } else if names(&call.completion_output) {
                state.completion_write = Some(write);
            }
        }
        let stream = self.job.on(self.stream.as_ref());
        for state in self.calls.iter().filter(|state| !state.active) {
            for write in [&state.completion_write, &state.transition_write]
                .into_iter()
                .flatten()
            {
                self.storage
                    .tensors
                    .write_scalar(write, Scalar::Int(0), stream)?;
            }
        }
        Ok(())
    }

    /// Lease every active call's relayed predicate on the consuming stream,
    /// which waits on each producer fence.
    fn lease_predicates(&mut self) -> Result<()> {
        let target = self.device.map_or(Device::Cpu, Device::Cuda);
        let requests: Vec<_> = self
            .plan
            .calls
            .iter()
            .zip(&self.calls)
            .filter(|(_, state)| state.active)
            .filter_map(|(call, _)| {
                call.predicate
                    .clone()
                    .filter(|value| value.dtype == DType::I64)
                    .map(|predicate| (predicate, call.call_id, None))
            })
            .collect();
        if !requests.is_empty() {
            self.reads = self.storage.tensors.consume_batch(
                &requests,
                Some(target),
                self.job.on(self.stream.as_ref()),
            )?;
        }
        Ok(())
    }

    /// Seal the lease on the batch's stream and retire its cache accesses and
    /// output storage once the fences complete.
    pub(super) fn seal(&self) -> Result<()> {
        seal(
            &self.storage.events,
            &self.buffer,
            self.job.on(self.stream.as_ref()),
            &self.retirement,
        )
    }

    /// Retire every reservation after a failure before the commit point;
    /// storage returns once the fences recorded here complete.
    pub(super) fn discard(&mut self) -> Result<()> {
        let stream = self.job.on(self.stream.as_ref());
        let reads = std::mem::take(&mut self.reads);
        let completed = self.storage.tensors.complete_reads(&reads, stream, &[]);
        let writes = std::mem::take(&mut self.writes);
        let abandoned = self.storage.tensors.abandon_writes(&writes);
        let sealed = if lock(&self.buffer).sealed() {
            Ok(())
        } else {
            self.seal()
        };
        lock(&self.buffer).abandon();
        completed.and(abandoned).and(sealed)
    }
}
