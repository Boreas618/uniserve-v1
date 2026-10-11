//! Text warmup: graph capture on scratch KV and a representative request.
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
//! The scenario then admits one request, prefills one token with a token
//! relay output, decodes from that relay and finishes, through the serving
//! executor.

use uniserve_core::{CallId, SamplingParams, UnitId};
use uniserve_ipc::{
    ArRequestParams, Batch as BatchPlan, BatchCommand, BlockTable, Bounds, CacheUnitAllocation,
    Call, CallCoordinates, CallKind, CallStatus, DType, ForwardBatch, ForwardMode, NewRequest,
    ShapeBound, TensorRef,
};

use super::super::experts::ExpertRuntime;
use super::{StepParts, TextRuntime};
use crate::cuda_graph::capture_order;
use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::runner::TokenSelection;
use crate::runner::text::{COMMIT_CACHE, RowSource, StepRequest, TokenRow};
use crate::storage::KVCache;
use crate::warmup::{Scenarios, WarmupHooks};
use crate::{Error, Result};

/// The text runtime's warmup work.
pub(crate) struct TextWarmup {
    /// Tokens one capture row may hold.
    pub(crate) row_tokens: usize,
}

impl WarmupHooks for TextWarmup {
    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let parts = backend.parts();
        let (Some(runtime), Some(storage)) = (parts.text, parts.storage) else {
            return Ok(());
        };
        let kv = storage
            .kv
            .as_mut()
            .ok_or(Error::State("text warmup requires a KV cache"))?;
        capture(runtime, kv, self.row_tokens, parts.experts).map_err(|error| {
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

/// Install scratch tables of `lengths` tokens for slots `1..`, run
/// `operation`, then clear the tables and reset the scratch units.
pub(crate) fn with_scratch<T>(
    kv: &mut KVCache,
    lengths: &[usize],
    stream: Option<&uniserve_cuda::Stream>,
    operation: impl FnOnce(&mut KVCache) -> Result<T>,
) -> Result<T> {
    let shapes = kv.plan().group_shapes()?;
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
    let scratch = kv.scratch_units(count as u32, stream)?;
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
    kv.install(&tables, stream)?;
    let result = operation(kv);
    let slots: Vec<u32> = (1..=lengths.len() as u32).collect();
    let cleared = kv.release_slots(&slots, stream);
    let finished = scratch.finish(kv, stream);
    let value = result?;
    cleared?;
    finished?;
    Ok(value)
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

/// Capture every runner's graphs.
///
/// With experts, each microbatch runner captures every bucket at every step
/// capacity that holds it, largest first: each capture's eager warmup is a
/// collective expert step at that capacity (announced to dedicated expert
/// ranks), in which the other microbatches join.
fn capture(
    runtime: &mut TextRuntime,
    kv: &mut KVCache,
    row_tokens: usize,
    experts: Option<&ExpertRuntime>,
) -> Result<()> {
    let shapes = kv.plan().group_shapes()?;
    let pages: Vec<(usize, usize)> = shapes
        .iter()
        .map(|shape| (shape.page_tokens as usize, shape.units_per_page as usize))
        .collect();
    let units = kv.planes().num_units() as usize - 1;
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
    for index in 0..runtime.runners.len() {
        let (group, peer) = runtime.locate(index)?;
        let prefill = capture_order(
            runtime.runners[index].shapes().prefill().to_vec(),
            |shape| (shape.token_bucket * shape.row_bucket, shape.token_bucket),
        );
        let decode = capture_order(runtime.runners[index].shapes().decode().to_vec(), |&rows| {
            (rows, rows)
        });
        let prime = runtime.runners[index]
            .shapes()
            .prefill()
            .iter()
            .map(|shape| shape.row_bucket - 1)
            .max()
            .unwrap_or(usize::MAX)
            .max(1);
        let stream = runtime.runners[index].stream().cloned();
        let stream = stream.as_ref();
        let run = |runtime: &mut TextRuntime,
                   kv: &mut KVCache,
                   rows: &[TokenRow],
                   decode: bool,
                   capacity: Option<usize>| {
            step(runtime, experts, (group, peer), kv, rows, decode, capacity)
        };
        for shape in prefill {
            let lengths = shape.scratch_lengths(row_tokens, &pages, units);
            let selection = if shape.outputs {
                TokenSelection::LastLogits
            } else {
                TokenSelection::Cache
            };
            for capacity in variants(shape.token_bucket) {
                with_scratch(kv, &lengths, stream, |kv| {
                    run(
                        runtime,
                        kv,
                        &rows(1, &lengths, 0, selection),
                        false,
                        capacity,
                    )
                })?;
            }
        }
        for size in decode {
            let lengths = vec![1; size];
            with_scratch(kv, &vec![2; size], stream, |kv| {
                // One real token per row, prefilled in batches no larger than
                // the scheduler admits (the largest prefill bucket's live
                // rows) and committed as each slot's cache; then the decode
                // the graph captures with every row's continuation flag set.
                for first in (0..size).step_by(prime) {
                    let count = prime.min(size - first);
                    run(
                        runtime,
                        kv,
                        &rows(first + 1, &lengths[..count], 0, TokenSelection::LastLogits),
                        false,
                        None,
                    )?;
                }
                let slots: Vec<usize> = (1..=size).collect();
                let prefilled: Vec<[i64; 8]> = slots
                    .iter()
                    .map(|&slot| [slot as i64, COMMIT_CACHE, -1, 1, 0, 0, 0, 0])
                    .collect();
                runtime.runners[index].commit(&prefilled)?;
                runtime.state.set_predicates(&slots, true, stream)?;
                let mut captured = Ok(());
                for capacity in variants(size) {
                    captured = run(
                        runtime,
                        kv,
                        &rows(1, &lengths, 1, TokenSelection::LastLogits),
                        true,
                        capacity,
                    );
                    if captured.is_err() {
                        break;
                    }
                }
                runtime.state.set_predicates(&slots, false, stream)?;
                captured
            })?;
        }
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        let runner = &runtime.runners[index];
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

/// One warmup step of `rows` on microbatch `peer` of `group`; the group's
/// other microbatches join. `capacity` forces the expert step capacity.
fn step(
    runtime: &mut TextRuntime,
    experts: Option<&ExpertRuntime>,
    (group, peer): (usize, usize),
    kv: &mut KVCache,
    rows: &[TokenRow],
    decode: bool,
    capacity: Option<usize>,
) -> Result<()> {
    let (tables, cache) = kv.split();
    let request = StepRequest {
        decode,
        rows,
        finish_ids: &[],
        sampling: None,
        prompt: &[],
        embeddings: &[],
    };
    let mut requests: Vec<Option<StepRequest<'_>>> =
        (0..runtime.groups[group].len()).map(|_| None).collect();
    requests[peer] = Some(request);
    runtime
        .invoke(
            experts,
            group,
            &requests,
            &mut StepParts { tables, cache },
            capacity,
        )
        .map(drop)
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
        .and_then(|storage| storage.kv.as_ref())
        .ok_or(Error::State("text warmup requires a KV cache"))?
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
    tracing::info!(target: "uniserve_worker.warmup", "completed token runtime warmup");
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
        kv_visible_len: prefix,
        kv_computed_len: prefix,
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
        write_kv: vec![true],
    };
    batch
}
