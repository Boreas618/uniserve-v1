//! A token batch's reservation and retirement.
//!
//! [`TokenBatch::open`] binds every call's pending output, resolves
//! host-gated predicates and leases the batch's output storage.
//! [`TokenBatch::reserve`] installs the active calls' block tables, resets
//! newly assigned units, retains the KV extents the batch reads and writes
//! until its device work retires, binds its scalar products and leases its
//! relayed predicates. [`TokenBatch::discard`] retires all of it after a
//! failure before the commit point.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use uniserve_cuda::Stream;
use uniserve_ipc::{Batch as BatchPlan, CallStatus, DType, ForwardMode, RequestKey, TensorRef};

use super::super::launch::{lock, read_flag, reports_output, seal};
use super::calls::{self, Prepared};
use super::{COMPLETION_FIELDS, TextRuntime};
use crate::executor::PendingOutput;
use crate::executor::forward::image::ImageRuntime;
use crate::ffi::Device;
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

/// A token batch's borrowed owners and the resources it reserved.
pub(super) struct TokenBatch<'a> {
    pub(super) runtime: &'a mut TextRuntime,
    /// The image runtime, for prompts placing image features.
    pub(super) image: Option<&'a mut ImageRuntime>,
    pub(super) storage: &'a mut RankStorage,
    pub(super) requests: &'a mut RequestPool,
    pub(super) plan: &'a BatchPlan,
    /// The runner group serving the batch.
    pub(super) group: usize,
    /// The rank's expert runtime, when its forwards are expert steps.
    pub(super) experts: Option<&'a super::super::experts::ExpertRuntime>,
    /// The group's lane stream.
    pub(super) stream: Option<Stream>,
    pub(super) device: Option<i32>,
    pub(super) mode: ForwardMode,
    /// This rank reports the component's host results.
    pub(super) reports: bool,
    pub(super) calls: Vec<CallState>,
    pub(super) buffer: OutputLease,
    pub(super) writes: Vec<ProductBuffer>,
    pub(super) reads: Vec<Arc<ProductRead>>,
    /// Signals the physical end of the batch's KV accesses.
    pub(super) retirement: Arc<Retirement>,
    pub(super) started: Instant,
}

impl<'a> TokenBatch<'a> {
    /// Bind the batch's outputs and lease its output storage.
    #[expect(clippy::too_many_arguments, reason = "the batch's borrowed owners")]
    pub(super) fn open(
        runtime: &'a mut TextRuntime,
        image: Option<&'a mut ImageRuntime>,
        storage: &'a mut RankStorage,
        requests: &'a mut RequestPool,
        experts: Option<&'a super::super::experts::ExpertRuntime>,
        info: &uniserve_ipc::WorkerInfo,
        rank: usize,
        plan: &'a BatchPlan,
        started: Instant,
    ) -> Result<Self> {
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
        // false one skips its call.
        runtime.order.wait(stream.as_ref())?;
        for (call, state) in plan.calls.iter().zip(&mut calls) {
            if let Some(predicate) = &call.predicate
                && predicate.dtype == DType::U8
                && !read_flag(&storage.tensors, call, predicate, device, stream.as_ref())?
            {
                state.active = false;
                state.output.output.status = CallStatus::Predicated;
            }
        }

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
        let buffer = runtime
            .outputs
            .lease(plan.calls.len(), words.max(1), &devices, false)?;
        if let Some(stream) = &stream {
            lock(&buffer).begin_device(device, stream, &storage.events)?;
        }
        Ok(Self {
            runtime,
            image,
            storage,
            requests,
            plan,
            group,
            experts,
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
        })
    }

    /// Install tables, reset units, retain KV extents, bind products and
    /// lease relayed predicates of the active calls.
    pub(super) fn reserve(&mut self) -> Result<()> {
        let stream = self.stream.as_ref();
        let plan = self.plan;
        let kv = self
            .storage
            .kv
            .as_mut()
            .ok_or_else(|| Error::Unsupported("text calls require a KV cache".into()))?;

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
        kv.install(&assignments, stream)?;
        let units: Vec<u32> = plan
            .new_cache_units
            .iter()
            .filter(|allocation| active.contains(&allocation.request_pool_idx))
            .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
            .collect();
        if !units.is_empty() {
            let lane = self.runtime.groups[self.group][0];
            kv.recycle(&units, &self.runtime.columns[lane], stream)?;
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
            let visible = state.output.progress.kv_visible_len;
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
                let length = u64::from(if forward.write_kv[row] {
                    forward.seq_lens[row]
                } else {
                    forward.seq_lens[row] - forward.query_lens[row]
                });
                let mut spans = Vec::new();
                let tables = kv.tables();
                for group in 0..tables.groups().len() as u32 {
                    let table = tables.table(state.slot as u32, group)?;
                    let start = u64::from(table.start_page) * u64::from(table.shape.page_tokens);
                    if length > start {
                        spans.extend(table.spans(start, length - start)?);
                    }
                }
                kv.manager().retain_execution(
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
        for state in self.calls.iter().filter(|state| !state.active) {
            for write in [&state.completion_write, &state.transition_write]
                .into_iter()
                .flatten()
            {
                self.storage
                    .tensors
                    .write_scalar(write, Scalar::Int(0), self.stream.as_ref())?;
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
                self.stream.as_ref(),
            )?;
        }
        Ok(())
    }

    /// Seal the lease on the batch's stream and retire its KV accesses and
    /// output storage once the fences complete; wake the service then.
    pub(super) fn seal(&self) -> Result<()> {
        seal(
            &self.storage.events,
            &self.buffer,
            self.stream.as_ref(),
            &self.retirement,
        )
    }

    /// Retire every reservation after a failure before the commit point;
    /// storage returns once the fences recorded here complete.
    pub(super) fn discard(&mut self) -> Result<()> {
        let stream = self.stream.as_ref();
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
