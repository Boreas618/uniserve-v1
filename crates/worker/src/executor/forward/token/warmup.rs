//! Text warmup: graph capture on scratch KV and a representative request.
//!
//! Capture runs every runner's prefill buckets (largest footprint first, so
//! smaller captures reuse pool blocks), then its decode buckets (largest
//! first). Each bucket's rows get startup scratch units installed as block
//! tables of reserved slots `1..=rows`; nothing else is resident during
//! startup. A decode bucket first prefills one real token per row, so the
//! captured decode attends to model-produced K/V. Every capture runs its step
//! eagerly first, which also compiles the step's kernels.
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

use super::TextRuntime;
use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::runner::TokenSelection;
use crate::runner::graphs::capture_order;
use crate::runner::text::{RowSource, StepContext, StepRequest, TokenRow};
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
        capture(runtime, kv, self.row_tokens).map_err(|error| {
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
fn with_scratch<T>(
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

fn rows(lengths: &[usize], prefix: usize, selection: TokenSelection) -> Vec<TokenRow> {
    lengths
        .iter()
        .enumerate()
        .map(|(row, &length)| TokenRow {
            slot: row as u32 + 1,
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
fn capture(runtime: &mut TextRuntime, kv: &mut KVCache, row_tokens: usize) -> Result<()> {
    let shapes = kv.plan().group_shapes()?;
    let pages: Vec<(usize, usize)> = shapes
        .iter()
        .map(|shape| (shape.page_tokens as usize, shape.units_per_page as usize))
        .collect();
    let units = kv.planes().num_units() as usize - 1;
    for index in 0..runtime.runners.len() {
        let prefill = capture_order(
            runtime.runners[index].shapes().prefill().to_vec(),
            |shape| (shape.token_bucket * shape.row_bucket, shape.token_bucket),
        );
        let decode = capture_order(runtime.runners[index].shapes().decode().to_vec(), |&rows| {
            (rows, rows)
        });
        let stream = runtime.runners[index].stream().map(|stream| {
            // SAFETY: the lanes own the stream for the runtime's lifetime.
            unsafe { uniserve_cuda::Stream::borrowed(stream.device(), stream.raw()) }
        });
        let stream = stream.transpose()?;
        let stream = stream.as_ref();
        for shape in prefill {
            let lengths = shape.scratch_lengths(row_tokens, &pages, units);
            let selection = if shape.outputs {
                TokenSelection::LastLogits
            } else {
                TokenSelection::Cache
            };
            with_scratch(kv, &lengths, stream, |kv| {
                step(runtime, index, kv, &rows(&lengths, 0, selection), false)
            })?;
        }
        for size in decode {
            let lengths = vec![1; size];
            with_scratch(kv, &vec![2; size], stream, |kv| {
                // One real token per row, then the decode the graph captures
                // with every row's continuation flag set.
                step(
                    runtime,
                    index,
                    kv,
                    &rows(&lengths, 0, TokenSelection::LastLogits),
                    false,
                )?;
                let slots: Vec<usize> = (1..=size).collect();
                runtime.state.set_predicates(&slots, true, stream)?;
                let captured = step(
                    runtime,
                    index,
                    kv,
                    &rows(&lengths, 1, TokenSelection::LastLogits),
                    true,
                );
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

fn step(
    runtime: &mut TextRuntime,
    index: usize,
    kv: &mut KVCache,
    rows: &[TokenRow],
    decode: bool,
) -> Result<()> {
    let (tables, cache) = kv.split();
    runtime.runners[index]
        .execute(
            &StepRequest {
                decode,
                rows,
                sampling: None,
                prompt: &[],
                embeddings: &[],
            },
            &mut StepContext {
                tables,
                cache,
                state: &runtime.state,
            },
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
