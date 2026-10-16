//! Text warmup: graph capture on scratch prefix cache and a representative request.
//!
//! Capture runs every runner's prefill buckets (largest footprint first, so
//! smaller captures reuse pool blocks), then its decode buckets (largest
//! first). Each bucket's rows get startup scratch units installed as block
//! tables of reserved slots `1..=rows`; nothing else is resident during
//! startup. A decode bucket first prefills one real token per row and
//! commits its cache length, so the captured decode attends to
//! model-produced K/V; that commit captures the bucket's commit graph. Every
//! capture runs its step eagerly first, which also compiles the step's
//! kernels.
//!
//! Each step and each commit is one job of the runner's lane: the eager
//! call, then the capture, whose graph the job's completion carries. The
//! runner stores it before the next step is prepared, so later steps launch
//! the graphs of earlier ones. A bucket's scratch tables are two more jobs
//! of the lane: one installs them before the bucket's steps, one clears
//! them and resets the scratch units after.
//!
//! The scenario then admits one request, prefills one token with a token
//! relay output, decodes from that relay and finishes, through the serving
//! executor.

use uniserve_core::{CallId, SamplingParams, UnitId};
use uniserve_ipc::{
    ArRequestParams, Batch as BatchPlan, BatchCommand, BlockTable, Bounds, CacheUnitAllocation,
    Call, CallCoordinates, CallKind, CallStatus, DType, ForwardBatch, ForwardMode, NewRequest,
    ShapeBound, TensorRef,
};

use std::sync::Arc;

use super::CausalLMRunners;
use crate::cuda_graph::capture_order;
use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::lane::{BLOCK_ON_TIMEOUT, Job, JobQueues};
use crate::parallel::ExpertUnion;
use crate::runner::TokenSelection;
use crate::runner::causal_lm::{COMMIT_CACHE, PromptChunks, RowSource, StepRequest, TokenRow};
use crate::storage::{PrefixCache, ScratchUnits};
use crate::stream::Issue;
use crate::warmup::{Scenarios, WarmupHooks};
use crate::{Error, Result};

/// The causal LM runners' warmup work.
pub(crate) struct CausalLMWarmup {
    /// Tokens one capture row may hold.
    pub(crate) row_tokens: usize,
}

impl WarmupHooks for CausalLMWarmup {
    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let lanes = backend.job_queues().clone();
        let parts = backend.parts();
        let (Some(runtime), Some(storage)) = (parts.text, parts.storage) else {
            return Ok(());
        };
        let prefix_cache = storage
            .prefix_cache
            .as_mut()
            .ok_or(Error::State("text warmup requires a prefix cache"))?;
        runtime
            .warmup(prefix_cache, self.row_tokens, parts.experts, &lanes)
            .map_err(|error| {
                WorkerError::classify(error, Some("text graph capture")).with_phase("warmup")
            })
    }

    fn scenarios(&mut self, scenarios: &mut Scenarios<'_>) -> std::result::Result<(), WorkerError> {
        if scenarios.backend().text_mut().is_none() {
            return Ok(());
        }
        tokens(scenarios)
    }
}

/// Lend scratch units to slots `1..` as block tables of `lengths` tokens,
/// issued to `stream`: the units are reset, then the tables installed.
pub(crate) fn install_scratch(
    prefix_cache: &mut PrefixCache,
    lengths: &[usize],
    stream: impl Issue,
) -> Result<ScratchUnits> {
    let shapes = prefix_cache.plan().group_shapes()?;
    let count: usize = lengths
        .iter()
        .map(|&length| {
            shapes
                .iter()
                .map(|shape| {
                    length.max(1).div_ceil(shape.page_tokens as usize)
                        * shape.units_per_page as usize
                })
                .sum::<usize>()
        })
        .sum();
    let scratch = prefix_cache.scratch_units(count as u32, stream)?;
    let mut unit = *scratch.units().start();
    let mut tables = Vec::new();
    for (row, &length) in lengths.iter().enumerate() {
        for (group, shape) in shapes.iter().enumerate() {
            let pages = length.max(1).div_ceil(shape.page_tokens as usize);
            let units = pages * shape.units_per_page as usize;
            tables.push(BlockTable {
                request_pool_idx: row as u32 + 1,
                group_id: group as u32,
                start_page: 0,
                unit_ids: (unit..unit + units as u32).map(UnitId).collect(),
                allocated_tokens: (pages * shape.page_tokens as usize) as u32,
            });
            unit += units as u32;
        }
    }
    prefix_cache.install(&tables, stream)?;
    Ok(scratch)
}

/// Clear the scratch tables of slots `1..=rows` and return their units,
/// issued to `stream`: the stream waits for what wrote the units, resets
/// them and waits for the resets.
pub(crate) fn release_scratch(
    prefix_cache: &mut PrefixCache,
    scratch: ScratchUnits,
    rows: usize,
    stream: impl Issue,
) -> Result<()> {
    let slots: Vec<u32> = (1..=rows as u32).collect();
    let cleared = prefix_cache.release_slots(&slots, stream);
    let finished = scratch.finish(prefix_cache, stream);
    cleared.and(finished)
}

/// Rows of `lengths` host tokens for consecutive slots from `first`.
fn rows(
    first: usize,
    lengths: &[usize],
    prefix: usize,
    selection: TokenSelection,
) -> Vec<TokenRow> {
    lengths
        .iter()
        .enumerate()
        .map(|(row, &length)| TokenRow {
            slot: (first + row) as u32,
            source: RowSource::Host {
                tokens: vec![0; length],
                position: prefix as u64,
            },
            prefix: prefix as u32,
            query: length as u32,
            selection,
            gated: false,
            force_finish: false,
            causal: true,
        })
        .collect()
}

impl CausalLMRunners {
    /// Capture every runner's graphs, each step a job of its lane.
    ///
    /// With experts, each microbatch runner captures every bucket at every
    /// step capacity that holds it, largest first: each capture's eager
    /// warmup is a collective expert step at that capacity (announced to
    /// dedicated expert ranks), in which the other microbatches join.
    fn warmup(
        &mut self,
        prefix_cache: &mut PrefixCache,
        row_tokens: usize,
        experts: Option<&Arc<ExpertUnion>>,
        lanes: &JobQueues,
    ) -> Result<()> {
        let shapes = prefix_cache.plan().group_shapes()?;
        let pages: Vec<(usize, usize)> = shapes
            .iter()
            .map(|shape| (shape.page_tokens as usize, shape.units_per_page as usize))
            .collect();
        let units = prefix_cache.planes().num_units() as usize - 1;
        // Step capacities a graph of `tokens` tokens is captured at.
        let variants = |tokens: usize| -> Vec<Option<usize>> {
            match experts {
                None => vec![None],
                Some(experts) => {
                    let mut capacities: Vec<Option<usize>> = experts
                        .parallel()
                        .capacities()
                        .iter()
                        .copied()
                        .filter(|&capacity| capacity >= tokens)
                        .map(Some)
                        .collect();
                    capacities.reverse();
                    capacities
                }
            }
        };
        for index in 0..self.runners.len() {
            let (group, peer) = self.locate(index)?;
            let prefill = capture_order(self.runners[index].shapes().prefill().to_vec(), |shape| {
                (shape.token_bucket * shape.row_bucket, shape.token_bucket)
            });
            let decode = capture_order(self.runners[index].shapes().decode().to_vec(), |&rows| {
                (rows, rows)
            });
            let prime = self.runners[index]
                .shapes()
                .prefill()
                .iter()
                .map(|shape| shape.row_bucket - 1)
                .max()
                .unwrap_or(usize::MAX)
                .max(1);
            let stream = self.runners[index].stream().cloned();
            let stream = stream.as_ref();
            let kind = self.kind(group)?;
            let run = |runtime: &mut CausalLMRunners,
                       prefix_cache: &mut PrefixCache,
                       rows: &[TokenRow],
                       decode: bool,
                       capacity: Option<usize>,
                       job: Job| {
                runtime.warmup_step(
                    experts,
                    lanes,
                    (group, peer),
                    prefix_cache,
                    rows,
                    decode,
                    capacity,
                    job,
                )
            };
            for shape in prefill {
                let lengths = shape.scratch_lengths(row_tokens, &pages, units);
                let selection = if shape.outputs {
                    TokenSelection::LastLogits
                } else {
                    TokenSelection::Cache
                };
                for capacity in variants(shape.token_bucket) {
                    let rows = rows(1, &lengths, 0, selection);
                    self.warmup_bucket(
                        lanes,
                        kind,
                        prefix_cache,
                        &lengths,
                        stream,
                        |runtime, prefix_cache, _| {
                            run(runtime, prefix_cache, &rows, false, capacity, Job::new())
                        },
                    )?;
                }
            }
            for size in decode {
                let lengths = vec![1; size];
                let tables = vec![2; size];
                self.warmup_bucket(
                    lanes,
                    kind,
                    prefix_cache,
                    &tables,
                    stream,
                    |runtime, prefix_cache, after| {
                        // One real token per row, prefilled in batches no larger than
                        // the scheduler admits (the largest prefill bucket's live
                        // rows) and committed as each slot's cache; then the decode
                        // the graph captures with every row's continuation flag set.
                        for first in (0..size).step_by(prime) {
                            let count = prime.min(size - first);
                            run(
                                runtime,
                                prefix_cache,
                                &rows(first + 1, &lengths[..count], 0, TokenSelection::LastLogits),
                                false,
                                None,
                                Job::new(),
                            )?;
                        }
                        let slots: Vec<usize> = (1..=size).collect();
                        let prefilled: Vec<[i64; 8]> = slots
                            .iter()
                            .map(|&slot| [slot as i64, COMMIT_CACHE, -1, 1, 0, 0, 0, 0])
                            .collect();
                        runtime.warmup_commit(lanes, index, group, &prefilled)?;

                        // The flags are set by the job of the first decode step
                        // and cleared by the job that clears the tables.
                        let mut job = Job::new();
                        runtime.state.set_predicates(&slots, true, job.on(stream))?;
                        runtime
                            .state
                            .set_predicates(&slots, false, after.on(stream))?;
                        for capacity in variants(size) {
                            run(
                                runtime,
                                prefix_cache,
                                &rows(1, &lengths, 1, TokenSelection::LastLogits),
                                true,
                                capacity,
                                std::mem::take(&mut job),
                            )?;
                        }
                        Ok(())
                    },
                )?;
            }
            let runner = &self.runners[index];
            tracing::info!(
                target: "uniserve_worker.warmup",
                "captured {} text graphs: {} decode and {} prefill buckets",
                runner.captured(),
                runner.shapes().decode().len(),
                runner.shapes().prefill().len(),
            );
        }
        Ok(())
    }

    /// Run a capture bucket's `steps` over scratch tables of `lengths`
    /// tokens for slots `1..`. One job of `kind`'s lane installs the tables
    /// before the steps; one clears them and resets the scratch units after
    /// the steps, whether or not they succeeded. `steps` records into that
    /// second job what the device undoes before the tables are cleared.
    fn warmup_bucket(
        &mut self,
        lanes: &JobQueues,
        kind: CallKind,
        prefix_cache: &mut PrefixCache,
        lengths: &[usize],
        stream: Option<&uniserve_cuda::Stream>,
        steps: impl FnOnce(&mut Self, &mut PrefixCache, &Job) -> Result<()>,
    ) -> Result<()> {
        let perform = |job: Job| -> Result<()> {
            let completion = lanes.block_on(kind, job, BLOCK_ON_TIMEOUT)?;
            completion
                .check()
                .map(drop)
                .map_err(|failure| failure.error)
        };

        let before = Job::new();
        let scratch = install_scratch(prefix_cache, lengths, before.on(stream))?;
        perform(before)?;

        let after = Job::new();
        let stepped = steps(self, prefix_cache, &after);
        let released = release_scratch(prefix_cache, scratch, lengths.len(), after.on(stream));
        let performed = perform(after);
        stepped.and(released).and(performed)
    }

    /// One warmup step of `rows` on microbatch `peer` of `group`, as a job
    /// of the group's lane; the group's other microbatches join. `capacity`
    /// forces the expert step capacity. The step continues `job`, which
    /// holds what the device does before it.
    #[expect(
        clippy::too_many_arguments,
        reason = "one warmup step's shape and owners"
    )]
    fn warmup_step(
        &mut self,
        experts: Option<&Arc<ExpertUnion>>,
        lanes: &JobQueues,
        (group, peer): (usize, usize),
        prefix_cache: &mut PrefixCache,
        rows: &[TokenRow],
        decode: bool,
        capacity: Option<usize>,
        job: Job,
    ) -> Result<()> {
        let prompt = PromptChunks::default();
        let request = StepRequest {
            mode: if decode {
                ForwardMode::Decode
            } else {
                ForwardMode::Prefill
            },
            rows,
            finish_ids: &[],
            sampling: None,
            prompt: &prompt,
            embeddings: &[],
        };
        let mut requests: Vec<Option<StepRequest<'_>>> =
            (0..self.groups[group].len()).map(|_| None).collect();
        requests[peer] = Some(request);

        let step = self.forward(experts, group, &requests, prefix_cache, capacity, &job)?;
        let completion = lanes.block_on(self.kind(group)?, job, BLOCK_ON_TIMEOUT)?;
        let mut completion = completion.check().map_err(|failure| failure.error)?;
        self.complete(step, &mut completion).map(drop)
    }

    /// Commit `rows` on runner `index` of `group` as a job of its lane,
    /// storing the commit graph a bucket's first commit captures.
    fn warmup_commit(
        &mut self,
        lanes: &JobQueues,
        index: usize,
        group: usize,
        rows: &[[i64; 8]],
    ) -> Result<()> {
        let job = Job::new();
        let captured = self.runners[index].commit(rows, &job)?;
        let completion = lanes.block_on(self.kind(group)?, job, BLOCK_ON_TIMEOUT)?;
        let mut completion = completion.check().map_err(|failure| failure.error)?;
        match captured {
            Some(key) => self.runners[index].complete_commit(key, &mut completion),
            None => Ok(()),
        }
    }
}

/// One prefill and one relayed decode of a fresh request.
fn tokens(scenarios: &mut Scenarios<'_>) -> std::result::Result<(), WorkerError> {
    let slot = scenarios.slot()?;
    let key = scenarios.request_key();
    let admission = NewRequest::new(
        key,
        slot,
        Some(ArRequestParams {
            sampling: SamplingParams {
                temperature: 0.0,
                ignore_eos: true,
                ..SamplingParams::default()
            },
            negative_token_ids: Vec::new(),
            finish_token_ids: Vec::new(),
            initial_position: 0,
            canvas: None,
        }),
        None,
        0,
    )
    .map_err(|error| Error::Invalid(error.to_string()))?;
    let shapes = scenarios
        .backend()
        .rank_storage()
        .and_then(|storage| storage.prefix_cache.as_ref())
        .ok_or(Error::State("text warmup requires a prefix cache"))?
        .plan()
        .group_shapes()?;

    let (id, sequence) = scenarios.batch_id();
    let prompt_id = CallId::new(id, 0);
    let token = relay(key, prompt_id);
    let mut prompt = token_call(key, prompt_id, ForwardMode::Prefill, vec![0], 0);
    prompt.token_output = Some(token.clone());
    let mut first = token_batch(id, vec![admission], prompt, slot, &shapes, 0, 1);
    first.collective_seq = sequence;
    require_ok(
        scenarios
            .run(first)?
            .output
            .completions
            .iter()
            .map(|output| output.status),
    )?;

    let (id, sequence) = scenarios.batch_id();
    let decode_id = CallId::new(id, 0);
    let mut decode = token_call(key, decode_id, ForwardMode::Decode, Vec::new(), 1);
    decode.token_output = Some(relay(key, decode_id));
    decode.predicate = Some(token.clone());
    let mut second = token_batch(id, Vec::new(), decode, slot, &shapes, 1, 1);
    second.collective_seq = sequence;
    require_ok(
        scenarios
            .run(second)?
            .output
            .completions
            .iter()
            .map(|output| output.status),
    )?;

    let (id, sequence) = scenarios.batch_id();
    let mut finish = BatchPlan::new(id, Vec::new(), Vec::new()).with_commands(vec![
        BatchCommand::Free {
            buffer: token.buffer_id(),
        },
        BatchCommand::Finish {
            request_key: key,
            retained_buffers: Vec::new(),
        },
    ]);
    finish.collective_seq = sequence;
    scenarios.run(finish)?;
    scenarios.release(key);
    tracing::info!(target: "uniserve_worker.warmup", "completed causal LM warmup");
    Ok(())
}

fn require_ok(statuses: impl Iterator<Item = CallStatus>) -> Result<()> {
    for status in statuses {
        if status != CallStatus::Ok {
            return Err(Error::Invariant(format!(
                "text warmup call completed with {status:?}"
            )));
        }
    }
    Ok(())
}

fn relay(key: uniserve_ipc::RequestKey, call: CallId) -> TensorRef {
    TensorRef {
        request_key: key,
        producer_call_id: call,
        output_index: 0,
        generation: 1,
        dtype: DType::I64,
        shape_bound: ShapeBound::default(),
    }
}

fn token_call(
    key: uniserve_ipc::RequestKey,
    id: CallId,
    mode: ForwardMode,
    tokens: Vec<u32>,
    prefix: u32,
) -> Call {
    let mut call = Call::new(key, id, CallKind::Forward(mode));
    call.coordinates = CallCoordinates {
        logical_position: prefix,
        cache_visible_len: prefix,
        cache_computed_len: prefix,
        flow_step: 0,
    };
    call.bounds = Bounds {
        max_tokens: 1,
        ..Bounds::default()
    };
    call.input_token_ids = tokens;
    call
}

/// A one-call batch on `slot`, whose tables hold one page per group from
/// unit 1 (allocated by the first call).
fn token_batch(
    id: u64,
    admissions: Vec<NewRequest>,
    call: Call,
    slot: u32,
    shapes: &[crate::storage::GroupShape],
    prefix: u32,
    query: u32,
) -> BatchPlan {
    let mut batch = BatchPlan::new(id, admissions, vec![call]);
    let mut unit = 1;
    for (group, shape) in shapes.iter().enumerate() {
        let units: Vec<UnitId> = (unit..unit + shape.units_per_page).map(UnitId).collect();
        unit += shape.units_per_page;
        batch.block_tables.push(BlockTable {
            request_pool_idx: slot,
            group_id: group as u32,
            start_page: 0,
            unit_ids: units.clone(),
            allocated_tokens: shape.page_tokens,
        });
        if prefix == 0 {
            batch.new_cache_units.push(CacheUnitAllocation {
                request_pool_idx: slot,
                group_id: group as u32,
                unit_ids: units,
            });
        }
    }
    batch.forward = ForwardBatch {
        call_indices: vec![0],
        request_pool_indices: vec![slot],
        seq_lens: vec![prefix + query],
        query_lens: vec![query],
        write_cache: vec![true],
    };
    batch
}
