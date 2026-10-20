//! A token batch's reservation, on the shared batch skeleton
//! ([`crate::executor::batch`]).
//!
//! [`Opening::open`] orders the batch after the previous one, binds every
//! call's pending output and records the reads of the batch's host-gated
//! predicates. [`TokenBatch::lease`] leases the batch's output storage.
//! [`TokenBatch::reserve`] installs the active calls' block tables, resets
//! newly assigned units, retains the cache extents the batch reads and
//! writes until its device work retires, binds its scalar products and
//! leases its relayed predicates. A failure before the commit point retires
//! all of it ([`Batch::discard`]).
//!
//! None of this issues on the batch's stream: every stream operation is
//! recorded in the batch's job. A batch is a sequence of jobs, and between
//! two of them it is [`Parked`], which borrows nothing.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{
    Batch as BatchPlan, BlockTable, DType, ForwardMode, RequestKey, ShapeBound, TensorRef,
};

use super::calls::{self, Prepared};
use super::{COMPLETION_FIELDS, CausalLMRunners};
use crate::executor::batch::{Batch, Launched};
use crate::executor::image::ImageParts;
use crate::ffi::MemoryKind;
use crate::lane::{Completion, Job};
use crate::parallel::ExpertUnion;
use crate::request::RequestPool;
use crate::runner::causal_lm::context::grid_key;
use crate::runner::denoiser::image_model::GridKey;
use crate::runner::vision::PositionLayout;
use crate::sampling::SamplingControls;
use crate::storage::Metadata;
use crate::storage::{OutputPlacement, ProductBuffer};
use crate::stream::Issue as _;
use crate::worker::WorkerStorage;
use crate::{Error, Result};

/// What a call contributes to its token batch's forward, beside the state
/// every batch keeps of it ([`crate::executor::batch::CallState`]).
#[derive(Default)]
pub(crate) struct TokenCall {
    pub(super) prepared: Option<Prepared>,
    pub(super) token_write: Option<ProductBuffer>,
    pub(super) transition_write: Option<ProductBuffer>,
    /// The runner of the microbatch the call's row runs in.
    pub(super) microbatch: usize,
    /// The call's input index among its invocation's sampled tasks.
    pub(super) task: Option<usize>,
    pub(super) controls: Option<SamplingControls>,
    /// The call's prompt chunks, in order.
    pub(super) prompt: Vec<super::step::PromptScore>,
}

/// A batch plan's forward rows grouped by the call they belong to, each
/// call's rows in row order: a counting sort of the rows' call indices.
pub(super) struct ForwardRows {
    /// Row indices, call by call.
    rows: Vec<usize>,
    /// Where each call's rows start in `rows`, then where the last call's
    /// end.
    starts: Vec<usize>,
}

impl ForwardRows {
    /// Group the rows of `call_indices` among `calls` calls. A row naming
    /// no call is invalid.
    fn new(call_indices: &[u32], calls: usize) -> Result<Self> {
        // Each call's row count at `starts[call + 1]`, summed into where
        // each call's rows start.
        let mut starts = vec![0; calls + 1];
        for &call in call_indices {
            let call = call as usize;
            if call >= calls {
                return Err(Error::Invalid("forward row names no call".into()));
            }
            starts[call + 1] += 1;
        }
        for call in 0..calls {
            starts[call + 1] += starts[call];
        }
        // Placing a call's rows advances its start to the next call's;
        // shifting the starts up by one restores them.
        let mut rows = vec![0; call_indices.len()];
        for (row, &call) in call_indices.iter().enumerate() {
            let next = &mut starts[call as usize];
            rows[*next] = row;
            *next += 1;
        }
        starts.copy_within(0..calls, 1);
        starts[0] = 0;
        Ok(Self { rows, starts })
    }

    /// The forward rows of call `call`, in row order.
    pub(super) fn of(&self, call: usize) -> &[usize] {
        &self.rows[self.starts[call]..self.starts[call + 1]]
    }
}

/// The owners a token batch borrows while one of its jobs is prepared or
/// its completion applied.
pub(super) struct Owners<'a> {
    pub(super) runtime: &'a mut CausalLMRunners,
    /// The image runtime, for prompts placing image features.
    pub(super) image: Option<ImageParts<'a>>,
    pub(super) storage: &'a mut WorkerStorage,
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
    device: Option<DeviceId>,
    mode: ForwardMode,
    calls: crate::executor::batch::Opening<TokenCall>,
}

/// A token batch between two of its jobs: the batch and the runner group
/// serving it.
pub(crate) struct Parked {
    group: usize,
    mode: ForwardMode,
    batch: Batch<TokenCall>,
}

impl Parked {
    /// The committed batch as launched work awaiting materialization.
    pub(super) fn launch(self, stats: uniserve_ipc::ForwardStats) -> Launched {
        self.batch.launch(stats, "text_materialize", None)
    }
}

/// A token batch's borrowed owners, the batch and the job that records its
/// stream work.
pub(super) struct TokenBatch<'a> {
    pub(super) runtime: &'a mut CausalLMRunners,
    pub(super) image: Option<ImageParts<'a>>,
    pub(super) storage: &'a mut WorkerStorage,
    pub(super) requests: &'a mut RequestPool,
    pub(super) plan: &'a BatchPlan,
    pub(super) experts: Option<Arc<ExpertUnion>>,
    /// The runner group serving the batch.
    pub(super) group: usize,
    pub(super) mode: ForwardMode,
    pub(super) batch: Batch<TokenCall>,
    /// The batch's next job on its lane.
    pub(super) job: Job,
}

impl Opening {
    /// Bind the batch's outputs to their requests, then record in `job`
    /// the batch's order after the previous one and after `page_reads`,
    /// the fence of the host page tier reads the copy lane issued for it
    /// (`PrefixCache::issue_pages`), and the read of every host-gated
    /// predicate: the job's completion carries one byte each.
    pub(super) fn open(
        owners: &mut Owners<'_>,
        placement: &crate::config::placement::GroupPlacement,
        rank: usize,
        job: &Job,
        page_reads: Option<&Arc<uniserve_cuda::Event>>,
    ) -> Result<Self> {
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
        let group = runtime.route(&first.component, first.kind).ok_or_else(|| {
            Error::Unsupported(format!(
                "component {:?} has no text binding for {}",
                first.component,
                first.kind.as_str()
            ))
        })?;
        let lane = runtime.groups[group][0];
        let device = runtime.runners[lane].device();
        let stream = runtime.runners[lane].stream().cloned();

        // Host-gated predicates resolve before any storage is reserved; a
        // false one skips its call.
        let issue = job.on(stream.as_ref());
        runtime.order.wait(issue)?;
        // The copy lane made the fence's record before the batch reached
        // its lane: the batch's inputs waited for the job that records it.
        if let Some(fence) = page_reads {
            issue.wait(fence)?;
        }
        let mut calls = crate::executor::batch::Opening::<TokenCall>::open(
            owners.requests,
            &owners.storage.tensors,
            placement,
            rank,
            plan,
            device,
            issue,
            |_| true,
        )?;
        for state in calls.calls_mut() {
            state.forward.microbatch = lane;
        }
        Ok(Self {
            group,
            stream,
            device,
            mode,
            calls,
        })
    }

    /// Whether the opening job reads predicates, so the batch continues
    /// only once it completed.
    pub(super) fn reads_predicates(&self) -> bool {
        self.calls.reads_predicates()
    }

    /// The patch grids the active calls' vision blocks place that the image
    /// policy does not know yet: a context prefill's blocks and a
    /// visual-state row closing an image, of a model whose features take
    /// spatial positions.
    pub(super) fn lacking_grids(&self, owners: &Owners<'_>) -> Result<Vec<GridKey>> {
        let Some(image) = &owners.image else {
            return Ok(Vec::new());
        };
        let spatial = image
            .facts()
            .processing
            .as_ref()
            .and_then(|processing| processing.injection)
            .is_some_and(|injection| injection.positions == PositionLayout::TemporalSpatial);
        if !spatial {
            return Ok(Vec::new());
        }
        let mut keys = Vec::new();
        for (call, state) in owners.plan.calls.iter().zip(self.calls.calls()) {
            let close_image = if call.writes_context() {
                false
            } else if call.writes_visual_state() && call.latent_feature_input.is_none() {
                true
            } else {
                continue;
            };
            if !state.active {
                continue;
            }
            let input_images = owners
                .requests
                .get(call.request_key.request_id.0)?
                .admission()
                .input_images;
            for input in &call.vision_inputs {
                let Some(Metadata::Feature { height, width }) =
                    owners.storage.tensors.metadata(&input.feature)?
                else {
                    return Err(Error::Invalid(
                        "visual input requires encoder feature metadata".into(),
                    ));
                };
                let key = grid_key(height, width, close_image, input_images);
                if image.policy().grid(key).is_none() && !keys.contains(&key) {
                    keys.push(key);
                }
            }
        }
        Ok(keys)
    }

    /// Apply the opening job's completion: a call whose predicate read
    /// false is skipped.
    pub(super) fn predicates(&mut self, completion: &mut Completion) -> Result<()> {
        self.calls.predicates(completion)
    }
}

impl<'a> TokenBatch<'a> {
    /// Lease the opened batch's output storage and begin its job.
    pub(super) fn lease(owners: Owners<'a>, opening: Opening, job: Job) -> Result<Self> {
        let Opening {
            group,
            stream,
            device,
            mode,
            calls,
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
        let batch = Batch::lease(
            calls,
            &mut owners.runtime.outputs,
            words,
            stream,
            device,
            &owners.storage.events,
            &job,
        )?;
        Ok(Self::resume(owners, Parked { group, mode, batch }, job))
    }

    /// Continue a batch with its owners borrowed again, recording into
    /// `job`.
    pub(super) fn resume(owners: Owners<'a>, parked: Parked, job: Job) -> Self {
        Self {
            runtime: owners.runtime,
            image: owners.image,
            storage: owners.storage,
            requests: owners.requests,
            plan: owners.plan,
            experts: owners.experts,
            group: parked.group,
            mode: parked.mode,
            batch: parked.batch,
            job,
        }
    }

    /// Give the owners back: the job to submit, the batch as it waits for
    /// the job's completion, and the owners.
    pub(super) fn park(self) -> (Job, Parked, Owners<'a>) {
        (
            self.job,
            Parked {
                group: self.group,
                mode: self.mode,
                batch: self.batch,
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
    /// lease relayed predicates of the active calls. Returns the batch's
    /// forward rows, grouped by call.
    pub(super) fn reserve(&mut self) -> Result<ForwardRows> {
        let stream = self.job.on(self.batch.stream.as_ref());
        let plan = self.plan;
        let prefix_cache = self
            .storage
            .prefix_cache
            .as_mut()
            .ok_or_else(|| Error::Unsupported("text calls require a prefix cache".into()))?;

        // Tables and new units precede any of the batch's writes.
        let active: HashSet<u32> = self
            .batch
            .calls
            .iter()
            .filter(|state| state.active)
            .map(|state| state.slot as u32)
            .collect();
        // The plan's tables are installed as they are when every one is an
        // active call's, as in a batch that skipped no call.
        let installed = |table: &BlockTable| active.contains(&table.request_pool_idx);
        let assignments: Cow<'_, [BlockTable]> = if plan.block_tables.iter().all(installed) {
            Cow::Borrowed(&plan.block_tables)
        } else {
            Cow::Owned(
                plan.block_tables
                    .iter()
                    .filter(|&table| installed(table))
                    .cloned()
                    .collect(),
            )
        };
        prefix_cache.install(&assignments, stream)?;
        // Units the host page tier's reads filled on the copy lane are
        // overwritten whole there, and are not recycled.
        let read: HashSet<u32> = plan
            .page_reads
            .iter()
            .flat_map(|copy| copy.units.iter().map(|unit| unit.0))
            .collect();
        let units: Vec<u32> = plan
            .new_cache_units
            .iter()
            .filter(|allocation| active.contains(&allocation.request_pool_idx))
            .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
            .filter(|unit| !read.contains(unit))
            .collect();
        let lane = self.runtime.groups[self.group][0];
        if !units.is_empty() {
            prefix_cache.recycle(&units, &self.runtime.columns[lane], stream)?;
        }

        // Each active call has one forward row on its own slot; a context
        // prefill has one per prompt run and vision block (checked against
        // its rows when they are built). A row's lengths are the attention
        // geometry the engine builds from the same request state as the
        // call's coordinates, and each is read by its own consumer.
        let forward = &plan.forward;
        let rows = ForwardRows::new(&forward.call_indices, plan.calls.len())?;
        for (index, (call, state)) in plan.calls.iter().zip(&self.batch.calls).enumerate() {
            if !state.active {
                continue;
            }
            let call_rows = rows.of(index);
            let (_, rest) = call_rows
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
            for &row in call_rows {
                // Readers retain their prefix, writers also their query
                // tokens.
                let length = u64::from(if forward.write_cache[row] {
                    forward.seq_lens[row]
                } else {
                    forward.seq_lens[row] - forward.query_lens[row]
                });
                prefix_cache.retain(
                    call.request_key,
                    state.slot as u32,
                    length,
                    &self.batch.retirement,
                )?;
            }
        }

        self.bind_products()?;
        self.lease_predicates()?;
        Ok(rows)
    }

    /// Bind token relays of active calls and completion and transition flags
    /// of every call; skipped calls publish false flags now.
    fn bind_products(&mut self) -> Result<()> {
        let plan = self.plan;
        let target = self.batch.target();
        let slots: HashMap<RequestKey, usize> = plan
            .calls
            .iter()
            .zip(&self.batch.calls)
            .map(|(call, state)| (call.request_key, state.slot))
            .collect();
        let mut groups: indexmap::IndexMap<(DType, &ShapeBound), Vec<(TensorRef, MemoryKind)>> =
            indexmap::IndexMap::new();
        for (call, state) in plan.calls.iter().zip(&self.batch.calls) {
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
                    .entry((scalar.dtype, &scalar.shape_bound))
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
            self.batch.writes.push(write.clone());
            let index = owners[&id.owner];
            let call = &plan.calls[index];
            let state = &mut self.batch.calls[index];
            let names = |value: &Option<TensorRef>| {
                value.as_ref().is_some_and(|value| value.buffer_id() == id)
            };
            if names(&call.token_output) {
                state.forward.token_write = Some(write);
            } else if names(&call.transition_output) {
                state.forward.transition_write = Some(write);
            } else if names(&call.completion_output) {
                state.completion_write = Some(write);
            }
        }
        let stream = self.job.on(self.batch.stream.as_ref());
        for state in self.batch.calls.iter().filter(|state| !state.active) {
            for write in [&state.completion_write, &state.forward.transition_write]
                .into_iter()
                .flatten()
            {
                self.storage.tensors.write_scalar(write, 0, stream)?;
            }
        }
        Ok(())
    }

    /// Lease every active call's relayed predicate on the consuming stream,
    /// which waits on each producer fence.
    fn lease_predicates(&mut self) -> Result<()> {
        let target = self.batch.target();
        let requests: Vec<_> = self
            .plan
            .calls
            .iter()
            .zip(&self.batch.calls)
            .filter(|(_, state)| state.active)
            .filter_map(|(call, _)| {
                call.predicate
                    .clone()
                    .filter(|value| value.dtype == DType::I64)
                    .map(|predicate| (predicate, call.call_id, None))
            })
            .collect();
        if !requests.is_empty() {
            self.batch.reads = self.storage.tensors.consume_batch(
                &requests,
                Some(target),
                self.job.on(self.batch.stream.as_ref()),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_rows_group_each_calls_rows_in_row_order() -> Result<()> {
        // Rows of calls 2, 0, 2, 3, 0 among four calls; call 1 has none.
        let rows = ForwardRows::new(&[2, 0, 2, 3, 0], 4)?;
        assert_eq!(rows.of(0), &[1, 4]);
        assert!(rows.of(1).is_empty());
        assert_eq!(rows.of(2), &[0, 2]);
        assert_eq!(rows.of(3), &[3]);

        assert!(ForwardRows::new(&[0, 4], 4).is_err());
        Ok(())
    }
}
