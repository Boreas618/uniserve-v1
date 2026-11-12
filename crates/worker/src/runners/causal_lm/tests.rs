//! `CausalLmRunner` without a device: the bucket set, the profile run and the meta capture of
//! every bucket, the bytes of the input tensors, the refusals, the page-work seam, the paired
//! forward's inputs, the completion and the step agreement of an expert union, over a small
//! causal language model of `nn` operators on meta weights.
//!
//! The model exercises the cache through the modules a served model uses: its one layer
//! writes its keys and values (`kv_write_`) and attends over them (`paged_attention`) through
//! the page tables the runner gathers.

use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ipc::execution::{
    Batch, BatchColumns, BatchId, BatchRow, BatchRows, CallId, CallOutput, FreshSlot,
    LogitAdjustments, RowKind, RowState, Shape as BatchShape, StateCopy, StateDirection, Variant,
};
use ipc::philox::{DrawAddress, sampling_uniform};
use ipc::{CallStatus, ErrorCode, TokenLogprob};
use nn::capability::Drafter;
use nn::family::{CacheShape, ExpertShape, StateField, StateShape, TextModel};
use nn::functional::{
    WeightOrder, embedding, kv_write_, paged_attention, rms_norm, zero_arrival_counters_,
};
use nn::modules::Linear;
use nn::{CausalLm, ExpertExchange, KvCache, KvPlanes, NO_SLOT, RecurrentState, TokenBatch};
use ops::Generation;
use ops::moe::exchange::{Mode, TransportShape};
use proto::module::commit::{COMMIT_FIELDS, column, mode};
use proto::module::sampler::{PARAMETER_COLUMNS, TEMPERATURE, TOP_K, TOP_P};
use uniserve_tensor::{DType, Device, Tensor, WithDType};

use super::columns::{self, Column, Layout};
use super::experts::{self, JoinHook, Member};
use super::output::calls_output;
use super::wiring;
use super::{CausalLmConfig, CausalLmRunner, PageWork};
use crate::Error;
use crate::buckets::{Bucket, Buckets};
use crate::graphs::Graphs;
use crate::memory::cache::{CacheLayout, KvCacheConfig, KvCacheMemory};
use crate::memory::state::{StateConfig, StateMemory};
use crate::parallel::host::HostGroup;
use crate::runner::Runner;

const VOCABULARY: usize = 64;
const HIDDEN: usize = 16;
/// The model's heads: two of eight elements, for its queries and for its cache.
const HEADS: usize = 2;
const DIM: usize = 8;

/// The runner's request slots, the tokens of a page and the pages a slot's table row names.
const SLOTS: u32 = 6;
const PAGE: u32 = 16;
const TABLE_WIDTH: usize = 4;
const MAX_KV_LEN: u32 = PAGE * TABLE_WIDTH as u32;
/// The pages of the cache's planes, and of each row of the scratch set.
const PAGES: usize = 32;
const SCRATCH_PAGES: u32 = 2;
/// The most likely tokens a scored row reports.
const CANDIDATES: u32 = 3;

/// A meta tensor: a shape and an element type at an address nothing reads.
fn meta(shape: impl Into<uniserve_tensor::Shape>, dtype: DType) -> Tensor {
    backend::retained(0x1_0000, shape, dtype, &backend::meta()).unwrap()
}

/// A causal language model of one attention layer over the paged cache, with one host value.
struct Tiny {
    table: Tensor,
    gain: Tensor,
    key: Linear,
    value: Linear,
    head: Linear,
    /// Whether the model keeps a recurrent state, which its forward then asks for.
    stateful: bool,
    /// Whether the model's experts are split across ranks, whose exchange its forward then
    /// asks for.
    routed: bool,
    /// The model's drafter, when it loaded one.
    drafter: Option<TinyDrafter>,
}

impl Tiny {
    fn new(stateful: bool) -> Self {
        let linear = |rows: usize| {
            Linear::new(meta((rows, HIDDEN), DType::BF16), None, WeightOrder::NxK).unwrap()
        };

        Self {
            table: meta((VOCABULARY, HIDDEN), DType::BF16),
            gain: meta(HIDDEN, DType::BF16),
            key: linear(HEADS * DIM),
            value: linear(HEADS * DIM),
            head: linear(VOCABULARY),
            stateful,
            routed: false,
            drafter: None,
        }
    }

    /// The model with its drafter loaded.
    fn drafting(stateful: bool) -> Self {
        Self {
            drafter: Some(TinyDrafter::new()),
            ..Self::new(stateful)
        }
    }
}

/// A drafter of one attention layer over the draft cache: it reads the sum of the model's
/// hidden row and the next token's embedding.
struct TinyDrafter {
    gain: Tensor,
    key: Linear,
    value: Linear,
}

impl TinyDrafter {
    fn new() -> Self {
        let linear = || {
            Linear::new(
                meta((HEADS * DIM, HIDDEN), DType::BF16),
                None,
                WeightOrder::NxK,
            )
            .unwrap()
        };

        Self {
            gain: meta(HIDDEN, DType::BF16),
            key: linear(),
            value: linear(),
        }
    }
}

impl Drafter for TinyDrafter {
    fn cache(&self) -> CacheShape {
        CacheShape {
            index_keys: None,
            layers: 1,
            kv_heads: HEADS as u32,
            head_dim: DIM as u32,
        }
    }

    fn forward(
        &self,
        batch: &TokenBatch,
        cache: &KvCache,
        embedded: &Tensor,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        let tokens = batch.tokens();
        let fused = embedded.add(hidden)?;

        zero_arrival_counters_(&cache.workspace)?;
        let heads = |projected: Tensor| projected.reshape((tokens, HEADS, DIM));
        kv_write_(
            cache,
            0,
            batch,
            &heads(self.key.forward(&fused)?)?,
            &heads(self.value.forward(&fused)?)?,
        )?;

        let attended = paged_attention(
            &heads(fused.clone())?,
            batch,
            cache,
            0,
            0.35,
            None,
            batch.max_kv_len,
        )?;
        rms_norm(
            &attended.reshape((tokens, HIDDEN))?,
            &self.gain,
            1e-6,
            false,
            fused.dtype(),
        )
    }
}

impl CausalLm for Tiny {
    fn embed_input_ids(&self, ids: &Tensor) -> nn::Result<Tensor> {
        embedding(ids, &self.table, 0)
    }

    fn forward(
        &self,
        batch: &TokenBatch,
        cache: &[KvCache],
        state: Option<&RecurrentState>,
        exchange: Option<&ExpertExchange>,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        // The state's rows are the batch's, or a verification's sequences, which keep a trail.
        let lent = state
            .is_some_and(|state| state.slots.dims() == [batch.rows()] || state.trail.is_some());
        if self.stateful != lent {
            return Err(nn::Error::Unsupported(
                "the model's recurrent state is not the batch's rows".to_owned(),
            ));
        }
        let [group] = cache else {
            return Err(nn::Error::Unsupported("one attention group".to_owned()));
        };
        if self.routed != exchange.is_some_and(|exchange| exchange.step_rows > 0) {
            return Err(nn::Error::Unsupported(
                "the model's expert exchange is not the step's".to_owned(),
            ));
        }
        let tokens = batch.tokens();

        // A host value whose shape follows from the bucket: one FP32 per token.
        let bound = vec![batch.max_kv_len as f32; tokens];
        let bound = Tensor::from_vec(bound, (tokens, 1), &Device::Cpu)?
            .to_device(hidden.device())?
            .to_dtype(hidden.dtype())?;
        let hidden = hidden.broadcast_add(&bound)?;

        zero_arrival_counters_(&group.workspace)?;
        let heads = |projected: Tensor| projected.reshape((tokens, HEADS, DIM));
        kv_write_(
            group,
            0,
            batch,
            &heads(self.key.forward(&hidden)?)?,
            &heads(self.value.forward(&hidden)?)?,
        )?;

        let attended = paged_attention(
            &heads(hidden.clone())?,
            batch,
            group,
            0,
            0.35,
            None,
            batch.max_kv_len,
        )?;
        rms_norm(
            &attended.reshape((tokens, HIDDEN))?,
            &self.gain,
            1e-6,
            false,
            hidden.dtype(),
        )
    }

    fn compute_logits(&self, hidden: &Tensor) -> nn::Result<Tensor> {
        Ok(self.head.forward(hidden)?.to_dtype(DType::F32)?)
    }

    fn join_expert_step(&self, exchange: &ExpertExchange) -> nn::Result<()> {
        exchange.check()
    }
}

impl TextModel for Tiny {
    fn hidden(&self) -> u32 {
        HIDDEN as u32
    }

    fn vocabulary(&self) -> u32 {
        VOCABULARY as u32
    }

    fn dtype(&self) -> DType {
        DType::BF16
    }

    fn cache(&self) -> CacheShape {
        CacheShape {
            index_keys: None,
            layers: 1,
            kv_heads: HEADS as u32,
            head_dim: DIM as u32,
        }
    }

    fn state(&self) -> Option<StateShape> {
        self.stateful.then(|| StateShape {
            layers: 1,
            fields: vec![StateField {
                name: "state",
                dtype: DType::BF16,
                dims: vec![HIDDEN],
            }],
        })
    }

    fn experts(&self) -> Option<ExpertShape> {
        self.routed.then_some(ExpertShape {
            experts: 8,
            intermediate: 32,
            top: 2,
            encoded: false,
        })
    }

    fn drafter(&self) -> Option<&dyn Drafter> {
        self.drafter.as_ref().map(|drafter| drafter as &dyn Drafter)
    }
}

/// One attention group of one layer over `pages` meta pages, with the page table `table`.
fn group(pages: usize, table: Tensor) -> KvCache {
    let plane = || meta((pages, PAGE as usize, HEADS, DIM), DType::BF16);
    let flags = || meta((pages, 1, 1, 1), DType::U8);

    KvCache {
        block_tables: table,
        page_size: PAGE,
        layers: vec![KvPlanes {
            index_keys: None,
            key: plane(),
            value: plane(),
            key_initialized: flags(),
            value_initialized: flags(),
            key_scale: None,
            value_scale: None,
        }],
        workspace: meta(1 << 16, DType::U8),
    }
}

/// Decode buckets of two and four rows, greedy and drawn, and prefill buckets of four rows
/// over eight and eighty tokens in the four prefill variants.
fn config() -> CausalLmConfig {
    CausalLmConfig {
        phases: vec![ipc::PhaseId(0)],
        buckets: Buckets::causal_lm(&[2, 4], &[8, 80], true),
        slots: SLOTS,
        page_size: PAGE,
        max_kv_len: MAX_KV_LEN,
        candidates: CANDIDATES,
        scratch_pages: SCRATCH_PAGES,
        drafts: 0,
        draft_pool: None,
    }
}

/// A runner of [`Tiny`] over graphs that hold no device, with nothing captured.
fn runner(stateful: bool) -> CausalLmRunner {
    let config = config();
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&config).unwrap(),
    );

    let pooled = meta((SLOTS as usize + 1, TABLE_WIDTH), DType::U32);
    let state = stateful.then(|| {
        let buffers = || vec![vec![meta((SLOTS as usize, HIDDEN), DType::BF16)]];
        (buffers(), buffers())
    });

    CausalLmRunner::new(
        Arc::new(Tiny::new(stateful)),
        config,
        graphs,
        None,
        vec![group(PAGES, pooled)],
        state,
    )
    .unwrap()
}

/// A runner with every bucket profiled and captured.
fn captured() -> CausalLmRunner {
    let runner = runner(false);

    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        runner.graphs().capture(&runner, &bucket).unwrap();
    }
    runner
}

/// One row of sampling parameters at `temperature`: no top-k, the whole nucleus.
fn request(temperature: f32) -> [f32; PARAMETER_COLUMNS as usize] {
    let mut row = [0.0; PARAMETER_COLUMNS as usize];
    row[TEMPERATURE as usize] = temperature;
    row[TOP_P as usize] = 1.0;
    row
}

/// The field-major commit table of `rows`, each `(slot, mode, task)`.
fn commit(rows: &[(i64, i64, i64)]) -> Vec<i64> {
    let mut table = vec![0; COMMIT_FIELDS * rows.len()];
    for (row, (slot, mode, task)) in rows.iter().enumerate() {
        table[column::SLOT * rows.len() + row] = *slot;
        table[column::MODE * rows.len() + row] = *mode;
        table[column::TASK * rows.len() + row] = *task;
    }
    table
}

/// The batch of `columns` in `shape`, one call id per call from 100 on.
fn batch(shape: BatchShape, columns: BatchColumns) -> Batch {
    Batch {
        id: BatchId(7),
        phase: ipc::PhaseId(0),
        group: "text".to_owned(),
        calls: (0..columns.calls() as u64)
            .map(|call| CallId(100 + call))
            .collect(),
        shape,
        columns,
        generation: 1,
        cache_units: Vec::new(),
        block_tables: Vec::new(),
        page_loads: Vec::new(),
        page_offloads: Vec::new(),
        latent_restores: Vec::new(),
        host_outputs: Vec::new(),
        images: Vec::new(),
        frees: Vec::new(),
        media: None,
    }
}

/// A decode batch of one row per entry of `slots`, each at its `temperatures` entry: row `r`
/// decodes the token at position `10 + r` of its request, whose draw is at that position of
/// stream `100 + r`.
fn decode(slots: &[i32], temperatures: &[f32]) -> Batch {
    let rows = slots.len();
    let variant = if temperatures.iter().all(|temperature| *temperature == 0.0) {
        Variant::DecodeGreedy
    } else {
        Variant::DecodeSampled
    };

    let columns = BatchColumns {
        rows: BatchRows {
            calls: (0..rows as u32)
                .map(|row| BatchRow {
                    request: slots[row as usize] as u32,
                    kind: RowKind::Decode,
                    cache_group: 0,
                    rows: row..row + 1,
                })
                .collect(),
            positions: (0..rows as i32).map(|row| 10 + row).collect(),
            slot_mapping: (0..rows as i32)
                .map(|row| row * PAGE as i32 + 10 + row)
                .collect(),
            seq_lens: (0..rows as i32).map(|row| 11 + row).collect(),
            query_offsets: (0..=rows as i32).collect(),
            logits_rows: (0..rows as i32).collect(),
        },
        input_ids: vec![0; rows],
        slots: slots.to_vec(),
        parameters: temperatures
            .iter()
            .flat_map(|temperature| request(*temperature))
            .collect(),
        draws: (0..rows as u64)
            .map(|row| DrawAddress {
                key: 100 + row,
                position: 10 + row,
            })
            .collect(),
        commit: commit(
            &slots
                .iter()
                .enumerate()
                .map(|(row, slot)| (i64::from(*slot), mode::DECODE, row as i64))
                .collect::<Vec<_>>(),
        ),
        page_size: PAGE,
        max_kv_len: MAX_KV_LEN,
        ..BatchColumns::default()
    };

    batch(BatchShape::decode(rows as u32, variant), columns)
}

/// A prefill batch of two calls: slot 2's three tokens, which end its prompt, and slot 4's
/// two, which end none.
fn prefill(variant: Variant) -> Batch {
    let call = |request: u32, rows: std::ops::Range<u32>| BatchRow {
        request,
        kind: RowKind::Prefill,
        cache_group: 0,
        rows,
    };

    let mut table = commit(&[(2, mode::EXPLICIT, 0), (4, mode::CACHE, -1)]);
    table[column::CACHE_LENGTH * 2..][..2].copy_from_slice(&[3, 2]);
    table[column::LOGICAL * 2] = 3;
    table[column::SAMPLING * 2] = 1;

    let columns = BatchColumns {
        rows: BatchRows {
            calls: vec![call(2, 0..1), call(4, 1..2)],
            positions: vec![0, 1, 2, 0, 1],
            slot_mapping: vec![0, 1, 2, 16, 17],
            seq_lens: vec![3, 2],
            query_offsets: vec![0, 3, 5],
            logits_rows: vec![0, -1],
        },
        input_ids: vec![5, 6, 7, 8, 9],
        slots: vec![2, 4],
        parameters: [request(0.0), request(0.0)].concat(),
        draws: vec![DrawAddress::default(); 2],
        commit: table,
        page_size: PAGE,
        max_kv_len: MAX_KV_LEN,
        ..BatchColumns::default()
    };

    batch(BatchShape::prefill(2, 5, variant), columns)
}

/// The completion `calls_output` names for `batch` served from `bucket`.
fn completion(
    batch: &Batch,
    bucket: &Bucket,
    vocabulary: u32,
    candidates: u32,
    outputs: &[Tensor],
) -> crate::Result<Vec<ipc::execution::CallOutput>> {
    calls_output(
        &batch.calls,
        &batch.columns,
        bucket,
        vocabulary,
        candidates,
        outputs,
    )
}

/// The input block `columns::fill` writes for `batch` in `bucket`, over bytes that held
/// something else, with the layout its columns are read by.
fn filled(batch: &Batch, bucket: &Bucket, scratch: Option<&[u32]>) -> (Layout, Vec<u8>) {
    let layout = Layout::of(config().buckets.iter(), 0, 0).unwrap();
    let mut block = vec![0xA5_u8; layout.bytes() as usize];

    columns::fill(
        &layout,
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        bucket,
        scratch,
        &mut block,
    )
    .unwrap();
    (layout, block)
}

/// The operators of `bucket`'s launch table, in launch order.
fn operators(runner: &CausalLmRunner, bucket: &Bucket) -> Vec<&'static str> {
    runner
        .graphs()
        .launch_table(bucket)
        .unwrap()
        .into_iter()
        .map(|(operator, _, _)| operator)
        .collect()
}

/// How often `operator` is launched in `launches`.
fn count(launches: &[&'static str], operator: &str) -> usize {
    launches.iter().filter(|name| **name == operator).count()
}

// ---- Buckets, the profile run and the capture ---------------------------------------------------

#[test]
fn the_buckets_are_the_config_s_largest_first() {
    let runner = runner(false);
    let buckets = runner.buckets();

    // Two decode sizes in two selections, and one prefill row size over two token sizes in
    // the four prefill variants.
    assert_eq!(buckets.len(), 2 * 2 + 2 * 4);
    assert!(buckets.windows(2).all(|pair| pair[0] > pair[1]));

    assert_eq!(
        buckets.first(),
        Some(&Bucket::prefill(4, 80, Variant::PrefillScored))
    );
    assert_eq!(
        buckets.last(),
        Some(&Bucket::decode(2, Variant::DecodeGreedy))
    );
    assert!(buckets.contains(&Bucket::decode(4, Variant::DecodeSampled)));
    assert!(buckets.contains(&Bucket::prefill(4, 8, Variant::PrefillWrite)));
}

#[test]
fn a_profile_run_states_the_launches_of_every_bucket_and_the_runner_s_memory() {
    let runner = runner(false);
    runner.graphs().profile_run(&runner).unwrap();

    for bucket in runner.buckets() {
        let launches = operators(&runner, &bucket);

        // Every bucket writes and reads the cache once, selects and scores a token per row,
        // and commits once.
        for operator in ["kv_write", "paged_attention", "logits_shape", "slot_commit"] {
            assert_eq!(count(&launches, operator), 1, "{operator} in {bucket}");
        }

        // A greedy variant takes the argmax and draws nothing; any other draws.
        let greedy = usize::from(bucket.variant.greedy());
        assert_eq!(count(&launches, "argmax"), greedy, "{bucket}");
        assert_eq!(count(&launches, "inverse_cdf"), 1 - greedy, "{bucket}");

        // The chosen tokens are scored once, and a scoring bucket scores its tokens besides,
        // sixty-four to a chunk.
        let chunks = match bucket.variant.scores() {
            true => bucket.tokens.div_ceil(64) as usize,
            false => 0,
        };
        assert_eq!(count(&launches, "log_softmax"), 1 + chunks, "{bucket}");
        assert_eq!(count(&launches, "logprob_gather"), 1 + chunks, "{bucket}");
    }

    // A decode attends as a decode and a prefill as a prefill.
    let phase = |bucket: &Bucket| {
        let table = runner.graphs().launch_table(bucket).unwrap();
        let attention = table
            .iter()
            .find(|(operator, _, _)| *operator == "paged_attention");

        attention.unwrap().1.clone()
    };
    assert!(phase(&Bucket::decode(4, Variant::DecodeGreedy)).contains("phase_decode"));
    assert!(phase(&Bucket::prefill(4, 8, Variant::PrefillWrite)).contains("phase_prefill"));

    let keys = runner.graphs().keys();
    assert!(!keys.is_empty());
    assert!(keys.iter().all(|(_, key)| !key.is_empty()));

    // The region holds the input tensors and the largest forward's activations, and the
    // model's one host value is an FP32 per token of the largest bucket.
    let inputs = CausalLmRunner::input_bytes(&config()).unwrap();
    assert!(runner.graphs().peak() > inputs + (80 * HIDDEN * 2) as u64);
    assert_eq!(runner.graphs().input_peak(), 4 * 80);
}

#[test]
fn every_bucket_is_captured_with_its_profile_run_s_launches() {
    let runner = captured();

    let mut ascending = runner.buckets();
    ascending.reverse();
    assert_eq!(runner.graphs().buckets(), ascending);

    // The host value of each capture is the forward's own: one FP32 per token.
    for bucket in runner.buckets() {
        let copies = runner.graphs().input_copies(&bucket).unwrap();
        let bytes: Vec<u64> = copies.iter().map(|(_, _, bytes)| *bytes).collect();

        assert_eq!(bytes, [4 * u64::from(bucket.tokens)], "{bucket}");
    }
}

#[test]
fn the_runner_s_memory_is_a_function_of_its_config() {
    let config = config();

    // Twenty-one columns, none longer than one 256-byte block but the token columns and the
    // commit table of the largest bucket.
    let inputs = CausalLmRunner::input_bytes(&config).unwrap();
    assert_eq!(inputs % 256, 0);
    assert!(inputs >= 4 * 320 + 21 * 256 - 4 * 256);

    // The two tables of a row per slot and vocabulary entry dominate the slot state.
    let slots = CausalLmRunner::slot_bytes(&config, VOCABULARY as u32).unwrap();
    let tables = 2 * u64::from(SLOTS) * VOCABULARY as u64 * 4;
    assert!(slots >= tables && slots < tables + 8 * 256);

    let mut wider = config;
    wider.slots = 2 * SLOTS;
    assert!(CausalLmRunner::slot_bytes(&wider, VOCABULARY as u32).unwrap() >= 2 * tables);
}

// ---- The bytes of the input tensors -------------------------------------------------------------

#[test]
fn a_decode_batch_is_padded_with_rows_that_address_nothing() {
    // Three rows in a bucket of four: a greedy row beside two drawn ones.
    // The scheduler numbers request slots from one: the last row runs on the last slot.
    let batch = decode(&[3, 1, SLOTS as i32], &[0.0, 0.7, 1.0]);
    let bucket = Bucket::decode(4, Variant::DecodeSampled);
    let (layout, block) = filled(&batch, &bucket, None);

    let u32s = |column, elements| layout.read::<u32>(&block, column, elements);
    let i32s = |column, elements| layout.read::<i32>(&block, column, elements);

    // The batch's tokens write their slots and the padding token writes none.
    assert_eq!(i32s(Column::Positions, 4), [10, 11, 12, 0]);
    assert_eq!(u32s(Column::SlotMapping, 4), [10, 27, 44, NO_SLOT]);
    assert_eq!(u32s(Column::LiveTokens, 1), [3]);

    // The padding row is an empty sequence past the batch's tokens.
    assert_eq!(u32s(Column::SeqLens, 4), [11, 12, 13, 0]);
    assert_eq!(u32s(Column::QueryOffsets, 5), [0, 1, 2, 3, 3]);
    assert_eq!(u32s(Column::LogitsRows, 4), [0, 1, 2, 0]);

    // Request slot `s` owns row `s - 1` of every per-slot buffer. A padding row is no slot
    // to the sampler, and gathers the padding row of the tables, the row past the last slot's.
    assert_eq!(u32s(Column::Slots, 4), [2, 0, 5, NO_SLOT]);
    assert_eq!(u32s(Column::TableRows, 4), [2, 0, 5, SLOTS]);
    assert_eq!(u32s(Column::SlotRows, 4), [2, 0, 5, SLOTS]);
    assert_eq!(u32s(Column::CallSlots, 4), [2, 0, 5, SLOTS]);

    // A greedy row in a drawing bucket draws from its one largest candidate.
    let parameters = layout.read::<f32>(&block, Column::Parameters, 4 * 8);
    let row = |row: usize| &parameters[row * 8..(row + 1) * 8];
    assert_eq!(row(0)[TOP_K as usize], 1.0);
    assert_eq!(row(1)[TEMPERATURE as usize], 0.7);
    assert_eq!(row(1)[TOP_K as usize], 0.0);
    assert_eq!(row(3), [0.0; 8]);

    // Each row's draw is its request's stream at the position it samples.
    let draws = layout.read::<f32>(&block, Column::Draws, 4);
    let expected: Vec<f32> = (0..3)
        .map(|row| sampling_uniform(100 + row, 10 + row, 0, 0))
        .collect();
    assert_eq!(draws[..3], expected);
    assert_eq!(draws[3], 0.0);

    // No row adjusts its logits.
    assert_eq!(u32s(Column::Adjustments, 8), [0, NO_SLOT].repeat(4));

    // The commit table holds the batch's calls and an inert row past them.
    let table = i32s(Column::Commit, COMMIT_FIELDS * 4);
    assert_eq!(table[column::SLOT * 4..][..4], [2, 0, 5, -1]);
    assert_eq!(table[column::TASK * 4..][..4], [0, 1, 2, -1]);
    assert_eq!(i32s(Column::Valid, 4), [1; 4]);
    assert_eq!(i32s(Column::Active, 4), [1; 4]);
    assert_eq!(i32s(Column::Points, 4), [0; 4]);
    assert_eq!(layout.read::<u8>(&block, Column::Continuation, 4), [0; 4]);
}

#[test]
fn a_prefill_batch_is_padded_with_tokens_no_row_holds() {
    let mut batch = prefill(Variant::PrefillScored);
    batch.columns.targets = vec![6, 7, -1, 9, -1];
    let bucket = Bucket::prefill(4, 8, Variant::PrefillScored);
    let (layout, block) = filled(&batch, &bucket, None);

    let u32s = |column, elements| layout.read::<u32>(&block, column, elements);

    assert_eq!(u32s(Column::InputIds, 8), [5, 6, 7, 8, 9, 0, 0, 0]);
    assert_eq!(
        u32s(Column::SlotMapping, 8),
        [0, 1, 2, 16, 17, NO_SLOT, NO_SLOT, NO_SLOT]
    );
    assert_eq!(
        u32s(Column::Targets, 8),
        [6, 7, NO_SLOT, 9, NO_SLOT, NO_SLOT, NO_SLOT, NO_SLOT]
    );
    assert_eq!(u32s(Column::LiveTokens, 1), [5]);

    // The query offsets end at the batch's own tokens: the padding lies in no row's span.
    assert_eq!(u32s(Column::SeqLens, 4), [3, 2, 0, 0]);
    assert_eq!(u32s(Column::QueryOffsets, 5), [0, 3, 5, 5, 5]);

    // The row that ends its prompt reads the last token of its span; a row that samples
    // nothing, and a padding row, read a hidden row whose sample is discarded.
    assert_eq!(u32s(Column::LogitsRows, 4), [2, 0, 0, 0]);

    assert_eq!(u32s(Column::Slots, 4), [1, 3, NO_SLOT, NO_SLOT]);
    assert_eq!(u32s(Column::TableRows, 4), [1, 3, SLOTS, SLOTS]);
    assert_eq!(u32s(Column::CallSlots, 4), [1, 3, SLOTS, SLOTS]);

    let table = layout.read::<i32>(&block, Column::Commit, COMMIT_FIELDS * 4);
    assert_eq!(table[column::SLOT * 4..][..4], [1, 3, -1, -1]);
    assert_eq!(table[column::TASK * 4..][..4], [0, -1, -1, -1]);
    assert_eq!(table[column::CACHE_LENGTH * 4..][..4], [3, 2, 0, 0]);
}

#[test]
fn a_row_s_adjustments_are_its_bias_flag_and_its_ban_mask() {
    let mut batch = decode(&[1, 2], &[0.0, 0.0]);
    batch.columns.adjustments = LogitAdjustments {
        installs: Vec::new(),
        biased: vec![true, false],
        banned: vec![Vec::new(), vec![3, 40]],
    };
    let (layout, block) = filled(&batch, &Bucket::decode(2, Variant::DecodeGreedy), None);

    // The second row is the first that bans, so its mask is mask zero.
    assert_eq!(
        layout.read::<u32>(&block, Column::Adjustments, 4),
        [1, NO_SLOT, 0, 0]
    );
}

#[test]
fn the_idle_inputs_of_every_bucket_write_and_commit_nothing() {
    let runner = runner(false);
    let layout = Layout::of(config().buckets.iter(), 0, 0).unwrap();

    for bucket in runner.buckets() {
        let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
        let mut block = vec![0xA5_u8; layout.bytes() as usize];
        runner.idle_inputs(&bucket, &mut block).unwrap();

        // No token writes a cache slot and no row has a key, a slot or a request's pages.
        assert_eq!(
            layout.read::<u32>(&block, Column::SlotMapping, tokens),
            vec![NO_SLOT; tokens]
        );
        assert_eq!(
            layout.read::<u32>(&block, Column::SeqLens, rows),
            vec![0; rows]
        );
        assert_eq!(
            layout.read::<u32>(&block, Column::Slots, rows),
            vec![NO_SLOT; rows]
        );
        assert_eq!(
            layout.read::<u32>(&block, Column::TableRows, rows),
            vec![SLOTS; rows]
        );

        // The step is a full batch of the bucket: every token lies in a row's span.
        let offsets = layout.read::<u32>(&block, Column::QueryOffsets, rows + 1);
        assert_eq!((offsets[0], offsets[rows]), (0, bucket.tokens));
        assert_eq!(
            layout.read::<u32>(&block, Column::LiveTokens, 1),
            [bucket.tokens]
        );

        // Every row of the commit table is inert.
        let capacity = layout.rows();
        let table = layout.read::<i32>(&block, Column::Commit, COMMIT_FIELDS * capacity);
        assert_eq!(
            table[column::SLOT * capacity..][..capacity],
            vec![-1; capacity]
        );
        assert_eq!(
            table[column::TASK * capacity..][..capacity],
            vec![-1; capacity]
        );
    }

    // A bucket the runner does not state has no idle step.
    let unstated = Bucket::decode(3, Variant::DecodeGreedy);
    assert!(runner.idle_inputs(&unstated, &mut []).is_err());
    assert!(runner.forward(&unstated).is_err());
}

// ---- The choice of a bucket and the refusals ----------------------------------------------------

#[test]
fn a_step_runs_the_smallest_captured_bucket_that_holds_it() {
    let mut runner = captured();

    let small = runner
        .prepare_inputs(&decode(&[1, 2], &[0.0, 0.0]))
        .unwrap();
    assert_eq!(small, Bucket::decode(2, Variant::DecodeGreedy));

    let wide = runner
        .prepare_inputs(&decode(&[1, 2, 3], &[0.0, 0.5, 0.0]))
        .unwrap();
    assert_eq!(wide, Bucket::decode(4, Variant::DecodeSampled));

    let prompt = runner
        .prepare_inputs(&prefill(Variant::PrefillGreedy))
        .unwrap();
    assert_eq!(prompt, Bucket::prefill(4, 8, Variant::PrefillGreedy));
}

#[test]
fn the_last_request_slot_serves_and_gathers_its_own_rows() {
    let mut runner = captured();

    // Every per-slot buffer holds rows `0..SLOTS` for the slots `1..=SLOTS`: the last slot's
    // batch is prepared, and starts its request on the last row.
    let mut batch = decode(&[SLOTS as i32], &[0.0]);
    batch.columns.fresh_slots = vec![FreshSlot {
        slot: SLOTS,
        history: vec![3, 3],
    }];
    batch.columns.adjustments.installs = vec![ipc::execution::SlotBias {
        slot: SLOTS,
        bias: vec![(3, 1.5)],
        allowed: None,
    }];
    runner.prepare_inputs(&batch).unwrap();

    // The slot's writes are made, and the step reaches the replay graphs with no device
    // refuse.
    let refused = runner.execute(&batch).unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");

    // Its rows gather row `SLOTS - 1` of the page tables and of the slot state, and its call
    // commits into that row; the padding row is the one after it.
    let bucket = Bucket::decode(2, Variant::DecodeGreedy);
    let (layout, block) = filled(&batch, &bucket, None);
    let u32s = |column| layout.read::<u32>(&block, column, 2);

    assert_eq!(u32s(Column::Slots), [SLOTS - 1, NO_SLOT]);
    assert_eq!(u32s(Column::TableRows), [SLOTS - 1, SLOTS]);
    assert_eq!(u32s(Column::SlotRows), [SLOTS - 1, SLOTS]);
    assert_eq!(u32s(Column::CallSlots), [SLOTS - 1, SLOTS]);

    let table = layout.read::<i32>(&block, Column::Commit, COMMIT_FIELDS * 4);
    assert_eq!(table[column::SLOT * 4..][..2], [SLOTS as i32 - 1, -1]);
}

#[test]
fn a_batch_no_bucket_holds_is_refused_by_its_shape_and_the_bound() {
    let mut runner = captured();

    let refused = runner
        .prepare_inputs(&decode(&[0, 1, 2, 3, 4], &[0.0; 5]))
        .unwrap_err();
    let message = refused.to_string();

    assert!(matches!(refused, Error::Invalid(_)), "{message}");
    assert!(
        message.contains("decode_greedy/5 rows/5 tokens"),
        "{message}"
    );
    assert!(message.contains("4 rows and 80 tokens"), "{message}");
}

#[test]
fn a_bucket_that_is_not_captured_serves_nothing() {
    let mut runner = runner(false);

    let refused = runner
        .prepare_inputs(&decode(&[1, 2], &[0.0, 0.0]))
        .unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");
}

#[test]
fn a_drawn_row_in_a_greedy_bucket_is_refused_by_the_row() {
    let mut runner = captured();

    // The batch states a greedy shape and its second row states a temperature.
    let mut batch = decode(&[1, 2], &[0.0, 0.0]);
    batch.columns.parameters[PARAMETER_COLUMNS as usize + TEMPERATURE as usize] = 0.8;

    let message = runner.prepare_inputs(&batch).unwrap_err().to_string();
    assert!(message.contains("row 1"), "{message}");
    assert!(message.contains("temperature 0.8"), "{message}");
}

#[test]
fn a_batch_the_runner_s_graphs_were_not_captured_for_is_refused() {
    let mut runner = captured();

    // Another page size than the cache's.
    let mut paged = decode(&[1, 2], &[0.0, 0.0]);
    paged.columns.page_size = 2 * PAGE;
    assert!(matches!(
        runner.prepare_inputs(&paged),
        Err(Error::Invalid(_))
    ));

    // A row on a slot the runner holds no state for.
    let slotted = decode(&[1, SLOTS as i32 + 1], &[0.0, 0.0]);
    let message = runner.prepare_inputs(&slotted).unwrap_err().to_string();
    assert!(message.contains("request slot 7"), "{message}");

    // Slot zero is no request slot: the scheduler numbers them from one.
    let zeroth = decode(&[0, 1], &[0.0, 0.0]);
    let message = runner.prepare_inputs(&zeroth).unwrap_err().to_string();
    assert!(message.contains("request slot 0"), "{message}");

    // A verification, which a runner whose config states no drafts serves none of.
    let mut verified = decode(&[1, 2], &[0.0, 0.0]);
    verified.columns.rows.calls[0].kind = RowKind::Verify;
    assert!(matches!(
        runner.prepare_inputs(&verified),
        Err(Error::Unsupported(_))
    ));

    // Columns that are not the batch's rows.
    let mut short = decode(&[1, 2], &[0.0, 0.0]);
    short.columns.slots.pop();
    assert!(matches!(
        runner.prepare_inputs(&short),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn a_runner_is_built_over_its_model_s_cache_and_state() {
    let build = |stateful: bool, table: Tensor, state: bool| {
        let config = config();
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            CausalLmRunner::input_bytes(&config).unwrap(),
        );
        let state = state.then(|| {
            let buffers = || vec![vec![meta((SLOTS as usize, HIDDEN), DType::BF16)]];
            (buffers(), buffers())
        });

        CausalLmRunner::new(
            Arc::new(Tiny::new(stateful)),
            config,
            graphs,
            None,
            vec![group(PAGES, table)],
            state,
        )
    };
    let pooled = || meta((SLOTS as usize + 1, TABLE_WIDTH), DType::U32);

    assert!(build(false, pooled(), false).is_ok());
    assert!(build(true, pooled(), true).is_ok());

    // A pooled table without the padding row, and a state the model does not keep or is not
    // lent.
    let unpadded = meta((SLOTS as usize, TABLE_WIDTH), DType::U32);
    assert!(build(false, unpadded, false).is_err());
    assert!(build(false, pooled(), true).is_err());
    assert!(build(true, pooled(), false).is_err());

    // Graphs made for fewer bytes than the input tensors take.
    let config = config();
    let short = Graphs::meta(Arc::new(Generation::new(0, [])), 256);
    let pooled = group(PAGES, pooled());
    assert!(
        CausalLmRunner::new(
            Arc::new(Tiny::new(false)),
            config,
            short,
            None,
            vec![pooled],
            None
        )
        .is_err()
    );
}

#[test]
fn a_model_that_keeps_a_recurrent_state_is_lent_it_by_every_forward() {
    let runner = runner(true);

    // The model's forward refuses a batch without its state rows.
    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        runner.graphs().capture(&runner, &bucket).unwrap();
    }
}

// ---- Page work ----------------------------------------------------------------------------------

/// `batch` with one recurrent-state restore: page work of the cache's.
fn with_page_work(mut batch: Batch) -> Batch {
    batch.columns.state_copies = vec![StateCopy {
        slot: 1,
        units: vec![ipc::UnitId(0)],
        direction: StateDirection::Restore,
    }];
    batch
}

#[test]
fn a_batch_s_page_work_is_issued_through_the_hook_before_the_replay() {
    let mut runner = captured();
    let calls = Arc::new(Mutex::new(Vec::new()));

    let seen = Arc::clone(&calls);
    runner.set_page_work(Box::new(move |batch, work| {
        seen.lock().unwrap().push((batch.id, work));
        Ok(())
    }));

    // A batch that carries no page work calls nothing.
    let plain = decode(&[1, 2], &[0.0, 0.0]);
    assert!(runner.execute(&plain).is_err());
    assert!(calls.lock().unwrap().is_empty());

    // A refused batch issues nothing.
    let mut refused = with_page_work(decode(&[1, 2], &[0.0, 0.0]));
    refused.columns.page_size = 2 * PAGE;
    assert!(runner.execute(&refused).is_err());
    assert!(calls.lock().unwrap().is_empty());

    // The work that precedes the forward is issued once the inputs are prepared and before
    // the replay, which graphs that hold no device refuse; nothing follows a replay that was
    // not issued.
    let batch = with_page_work(decode(&[1, 2], &[0.0, 0.0]));
    let refused = runner.execute(&batch).unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");
    assert_eq!(
        *calls.lock().unwrap(),
        [(BatchId(7), PageWork::BeforeReplay)]
    );
}

#[test]
fn a_hook_s_refusal_fails_the_batch_before_its_replay() {
    let mut runner = captured();
    runner.set_page_work(Box::new(|_, _| {
        Err(Error::Resource("the cache holds no page for the batch"))
    }));

    let batch = with_page_work(decode(&[1, 2], &[0.0, 0.0]));
    assert!(matches!(runner.execute(&batch), Err(Error::Resource(_))));
}

#[test]
fn a_runner_with_no_hook_refuses_a_batch_that_carries_page_work() {
    let mut runner = captured();

    let batch = with_page_work(decode(&[1, 2], &[0.0, 0.0]));
    let message = runner.execute(&batch).unwrap_err().to_string();
    assert!(message.contains("carries page work"), "{message}");
}

#[test]
fn a_slot_the_runner_does_not_hold_is_not_started() {
    let mut runner = captured();

    let mut batch = decode(&[1, 2], &[0.0, 0.0]);
    batch.columns.fresh_slots = vec![FreshSlot {
        slot: SLOTS + 1,
        history: vec![1, 2],
    }];
    let message = runner.execute(&batch).unwrap_err().to_string();
    assert!(message.contains("request slot 7"), "{message}");

    // A fresh slot of a decode batch starts from its own row of the batch.
    let mut batch = decode(&[1, 2], &[0.0, 0.0]);
    batch.columns.fresh_slots = vec![FreshSlot {
        slot: 4,
        history: Vec::new(),
    }];
    let message = runner.execute(&batch).unwrap_err().to_string();
    assert!(message.contains("no row in its decode batch"), "{message}");
}

// ---- Paired forwards ----------------------------------------------------------------------------

#[test]
fn a_paired_forward_writes_its_rows_scratch_pages_and_commits_nothing() {
    let batch = prefill(Variant::PrefillOutput);
    let bucket = Bucket::prefill(4, 8, Variant::PrefillOutput);

    let rows = &batch.columns.rows;
    let scratch = columns::scratch_column(
        &rows.positions,
        &rows.query_offsets,
        &rows.slot_mapping,
        8,
        SCRATCH_PAGES,
        PAGE,
    )
    .unwrap();

    // Row `r` owns the pages `2r..2r + 2` of the set: its tokens at positions 0, 1 and 2
    // write slots of page `2r`, and the tokens the batch does not hold write nothing.
    assert_eq!(scratch, [0, 1, 2, 32, 33, NO_SLOT, NO_SLOT, NO_SLOT]);

    let (layout, block) = filled(&batch, &bucket, Some(&scratch));
    assert_eq!(layout.read::<u32>(&block, Column::SlotMapping, 8), scratch);

    // Every row gathers its own row of the scratch set's table, and reads its slot's state.
    assert_eq!(
        layout.read::<u32>(&block, Column::TableRows, 4),
        [0, 1, 2, 3]
    );
    assert_eq!(
        layout.read::<u32>(&block, Column::SlotRows, 4),
        [1, 3, SLOTS, SLOTS]
    );

    let table = layout.read::<i32>(&block, Column::Commit, COMMIT_FIELDS * 4);
    assert_eq!(table[column::SLOT * 4..][..4], [-1; 4]);
    assert_eq!(table[column::TASK * 4..][..4], [-1; 4]);
}

#[test]
fn a_scratch_slot_lies_in_its_row_s_pages_or_is_refused() {
    // Two rows of two tokens around a page boundary; the second row's last token is not
    // written by the served batch.
    let mapping =
        columns::scratch_slot_mapping(&[15, 16, 30, 31], &[0, 2, 4], &[15, 16, 62, -1], 2, PAGE)
            .unwrap();
    assert_eq!(mapping, [15, 16, 2 * PAGE + 30, NO_SLOT]);

    // A position past the row's scratch pages, and columns of different token counts.
    assert!(matches!(
        columns::scratch_slot_mapping(&[32], &[0, 1], &[32], 2, PAGE),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        columns::scratch_slot_mapping(&[0, 1], &[0, 1], &[0, 1], 2, PAGE),
        Err(Error::Invariant(_))
    ));
    assert!(columns::scratch_slot_mapping(&[0], &[0, 1], &[0], 0, PAGE).is_err());
}

#[test]
fn a_paired_forward_is_prepared_over_a_scratch_set_the_cache_lends() {
    let mut runner = captured();
    let batch = decode(&[1, 2], &[0.0, 0.0]);

    // No scratch set was lent.
    assert!(matches!(
        runner.prepare_shadow_inputs(&batch),
        Err(Error::Unsupported(_))
    ));

    // A set of fewer rows than the largest bucket, then one that holds them.
    let table = |rows: usize| meta((rows, SCRATCH_PAGES as usize), DType::U32);
    assert!(runner.set_scratch(vec![group(8, table(2))]).is_err());
    runner.set_scratch(vec![group(8, table(4))]).unwrap();

    let bucket = runner.prepare_shadow_inputs(&batch).unwrap();
    assert_eq!(bucket, Bucket::decode(2, Variant::DecodeGreedy));

    // The forward over the scratch set makes the launches of the serving forward.
    let serving = operators(&runner, &bucket);
    let context = backend::Context::meta(None);
    context.record_launches();
    {
        let _guard = context.enter();
        runner.forward(&bucket).unwrap();
    }
    let paired: Vec<&str> = context
        .take_launches()
        .into_iter()
        .map(|(operator, _, _)| operator)
        .collect();
    assert_eq!(paired, serving);

    // A row whose position lies past its scratch pages is refused.
    let mut far = decode(&[1, 2], &[0.0, 0.0]);
    far.columns.rows.positions[1] = (SCRATCH_PAGES * PAGE) as i32;
    assert!(matches!(
        runner.prepare_shadow_inputs(&far),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn a_model_that_keeps_a_recurrent_state_has_no_paired_forward() {
    let mut runner = runner(true);

    let refused = runner
        .prepare_shadow_inputs(&decode(&[1, 2], &[0.0, 0.0]))
        .unwrap_err();
    assert!(refused.to_string().contains("recurrent state"), "{refused}");
}

// ---- The completion -----------------------------------------------------------------------------

/// The CPU tensor of `values` in `shape`.
fn tensor<T: WithDType>(values: &[T], shape: impl Into<uniserve_tensor::Shape>) -> Tensor {
    Tensor::from_slice(values, shape.into(), &Device::Cpu).unwrap()
}

/// The outputs of a greedy forward of two rows that reports two candidates a row: the tokens
/// `tokens`, then their scores, the calls' positions and the rows' candidates.
fn greedy_outputs(tokens: [u32; 2]) -> Vec<Tensor> {
    vec![
        tensor(&tokens, 2),
        tensor(&[-0.5_f32, -2.0], 2),
        tensor(&[2_i32, 9], 2),
        tensor(&[41_i32, 71], 2),
        tensor(&[5_i32, 8], 2),
        tensor(&[-0.25_f32, -0.5, -0.75, -1.0], (2, 2)),
        tensor(&[7_u32, 11, 3, 4], (2, 2)),
    ]
}

fn entry(token_id: u32, logprob: f32, rank: u32) -> TokenLogprob {
    TokenLogprob {
        token_id,
        logprob,
        rank,
    }
}

#[test]
fn a_call_s_tokens_carry_their_rows_scores_and_its_slot_s_positions() {
    let batch = decode(&[3, 1], &[0.0, 0.0]);
    let bucket = Bucket::decode(2, Variant::DecodeGreedy);

    let output = completion(&batch, &bucket, 64, 2, &greedy_outputs([11, 22])).unwrap();

    let mut first = CallOutput::completed(
        CallId(100),
        vec![11],
        RowState {
            step_index: 5,
            cache_computed_len: 41,
            cache_visible_len: 5,
        },
    );
    // The token itself, then its row's candidates, the largest first.
    first.token_logprobs = vec![vec![
        entry(11, -0.5, 2),
        entry(7, -0.25, 1),
        entry(11, -0.5, 2),
    ]];
    assert_eq!(output[0], first);

    assert_eq!(output[1].call, CallId(101));
    assert_eq!(output[1].tokens, [22]);
    assert_eq!(
        (output[1].step_index, output[1].cache_computed_len),
        (8, 71)
    );
    assert_eq!(
        output[1].token_logprobs,
        [vec![
            entry(22, -2.0, 9),
            entry(3, -0.75, 1),
            entry(4, -1.0, 2)
        ]]
    );
}

#[test]
fn a_row_that_samples_nothing_gives_its_call_no_token() {
    // The second call ends no prompt: the forward chose a token for its row all the same.
    let batch = prefill(Variant::PrefillGreedy);
    let bucket = Bucket::prefill(4, 8, Variant::PrefillGreedy);

    let wide = |values: [u32; 2]| [values[0], values[1], 0, 0];
    let outputs = vec![
        tensor(&wide([11, 22]), 4),
        tensor(&[-0.5_f32, -2.0, 0.0, 0.0], 4),
        tensor(&[1_i32, 1, 0, 0], 4),
        tensor(&[3_i32, 2, 0, 0], 4),
        tensor(&[1_i32, 0, 0, 0], 4),
    ];
    let output = completion(&batch, &bucket, 64, 0, &outputs).unwrap();

    assert_eq!(output.len(), 2);
    assert_eq!(output[0].tokens, [11]);
    assert_eq!(output[0].token_logprobs, [vec![entry(11, -0.5, 1)]]);
    assert_eq!(
        output[1],
        CallOutput::completed(
            CallId(101),
            Vec::new(),
            RowState {
                step_index: 0,
                cache_computed_len: 2,
                cache_visible_len: 0
            }
        )
    );
}

#[test]
fn a_call_of_several_rows_reports_one_state_and_its_rows_tokens_in_order() {
    // One call owns both rows, on one slot.
    let mut batch = decode(&[2, 2], &[0.0, 0.0]);
    batch.columns.rows.calls = vec![BatchRow {
        request: 2,
        kind: RowKind::Decode,
        cache_group: 0,
        rows: 0..2,
    }];
    batch.calls = vec![CallId(9)];
    let bucket = Bucket::decode(2, Variant::DecodeGreedy);

    let output = completion(&batch, &bucket, 64, 2, &greedy_outputs([11, 22])).unwrap();

    assert_eq!(output.len(), 1);
    assert_eq!(output[0].tokens, [11, 22]);
    assert_eq!(output[0].token_logprobs.len(), 2);
    assert_eq!(
        (output[0].step_index, output[0].cache_computed_len),
        (5, 41)
    );

    // A call that runs on no slot reports no state.
    batch.columns.slots = vec![-1, -1];
    let output = completion(&batch, &bucket, 64, 2, &greedy_outputs([11, 22])).unwrap();
    assert_eq!((output[0].step_index, output[0].cache_computed_len), (0, 0));
}

#[test]
fn a_row_that_could_not_be_sampled_fails_its_call_alone() {
    let bucket = Bucket::decode(2, Variant::DecodeGreedy);

    // A greedy row with no numeric logit chose the vocabulary's width.
    let batch = decode(&[3, 1], &[0.0, 0.0]);
    let output = completion(&batch, &bucket, 64, 2, &greedy_outputs([64, 22])).unwrap();

    assert_eq!(output[0].status, CallStatus::Error);
    assert_eq!(output[0].error_code, Some(ErrorCode::NumericalFault));
    assert!(output[0].tokens.is_empty() && output[0].token_logprobs.is_empty());
    assert_eq!(output[1].status, CallStatus::Ok);
    assert_eq!(output[1].tokens, [22]);
    output[0].validate().unwrap();

    // A drawn row whose candidates formed no distribution states it by its flag.
    let batch = decode(&[3, 1], &[0.7, 0.7]);
    let bucket = Bucket::decode(2, Variant::DecodeSampled);
    let mut outputs = greedy_outputs([11, 0]);
    outputs.push(tensor(&[1_i32, 0], 2));

    let output = completion(&batch, &bucket, 64, 2, &outputs).unwrap();
    assert_eq!(output[0].tokens, [11]);
    assert_eq!(output[1].status, CallStatus::Error);
}

#[test]
fn a_scoring_bucket_reports_each_named_target_to_the_call_that_holds_its_token() {
    let mut batch = prefill(Variant::PrefillScored);
    batch.columns.targets = vec![6, 7, -1, 9, -1];
    let bucket = Bucket::prefill(4, 8, Variant::PrefillScored);

    // The row outputs, the rows' validity, then the scores of the bucket's eight tokens.
    let prompt_logprobs: Vec<f32> = (0..8).map(|token| -(token as f32)).collect();
    let outputs = vec![
        tensor(&[11_u32, 22, 0, 0], 4),
        tensor(&[-0.5_f32; 4], 4),
        tensor(&[1_i32; 4], 4),
        tensor(&[3_i32, 2, 0, 0], 4),
        tensor(&[1_i32, 0, 0, 0], 4),
        tensor(&[1_i32; 4], 4),
        tensor(&prompt_logprobs, 8),
        tensor(&[4_i32; 8], 8),
    ];
    let output = completion(&batch, &bucket, 64, 0, &outputs).unwrap();

    // The first call holds tokens 0..3, of which two name a target; the second holds 3..5.
    assert_eq!(
        output[0].prompt_logprobs,
        [vec![entry(6, 0.0, 4)], vec![entry(7, -1.0, 4)]]
    );
    assert_eq!(output[1].prompt_logprobs, [vec![entry(9, -3.0, 4)]]);
    assert_eq!(output[0].tokens, [11]);
}

#[test]
fn outputs_that_are_not_the_bucket_s_are_refused() {
    let batch = decode(&[3, 1], &[0.0, 0.0]);
    let bucket = Bucket::decode(2, Variant::DecodeGreedy);

    // Fewer outputs than the variant states: the candidates are missing.
    let mut outputs = greedy_outputs([11, 22]);
    outputs.truncate(5);
    assert!(matches!(
        completion(&batch, &bucket, 64, 2, &outputs),
        Err(Error::Invariant(_))
    ));

    // A row whose logits the batch asked for and no call owns.
    let mut orphan = decode(&[3, 1], &[0.0, 0.0]);
    orphan.columns.rows.calls.pop();
    orphan.calls.pop();
    assert!(matches!(
        completion(&orphan, &bucket, 64, 2, &greedy_outputs([11, 22])),
        Err(Error::Invariant(_))
    ));

    // Another number of call ids than calls.
    let mut unnamed = decode(&[3, 1], &[0.0, 0.0]);
    unnamed.calls.pop();
    assert!(completion(&unnamed, &bucket, 64, 2, &greedy_outputs([11, 22])).is_err());
}

#[test]
fn the_runner_serves_the_phases_of_its_config() {
    let runner = runner(false);

    assert_eq!(runner.phases(), [ipc::PhaseId(0)]);
    assert_eq!(runner.config().slots, SLOTS);
}

// ---- Expert unions ------------------------------------------------------------------------------

/// The most rows a member of the tested union sends in a step: the largest bucket's tokens.
const UNION_ROWS: u32 = 80;

/// The host groups of a union of `size` members over loopback connections, in member order.
fn host_groups(size: usize) -> Vec<Arc<HostGroup>> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let mut listener = Some(listener);

    std::thread::scope(|scope| {
        let members: Vec<_> = (0..size)
            .map(|rank| {
                // The first member holds the listener the address names.
                let listener = listener.take().filter(|_| rank == 0);
                let address = address.as_str();

                scope.spawn(move || {
                    let timeout = Duration::from_secs(30);
                    Arc::new(HostGroup::connect(address, listener, rank, size, timeout).unwrap())
                })
            })
            .collect();

        members
            .into_iter()
            .map(|member| member.join().unwrap())
            .collect()
    })
}

/// Member `rank` of a union of two over `group`, whose join batches go to `join`.
fn member(group: &Arc<HostGroup>, rank: u32, join: JoinHook) -> Member {
    let shape = TransportShape {
        ranks: 2,
        rank,
        experts: 8,
        hidden: HIDDEN as u32,
        intermediate: 32,
        max_tokens: UNION_ROWS,
        top: 2,
        dtype: DType::BF16,
        device: 0,
        attention_ranks: 0,
    };

    Member::new(Arc::clone(group), shape, Mode::Dispatch, join).unwrap()
}

/// A join hook that hands each join batch to the executor reading `inbox`.
fn deliver(inbox: crossbeam_channel::Sender<Batch>) -> JoinHook {
    Arc::new(move |batch| {
        inbox
            .send(batch)
            .map_err(|_| Error::State("the member's executor is gone"))
    })
}

/// A runner of a model that routes experts, as `member` of a union, with every bucket
/// captured.
fn union_runner(member: Member) -> CausalLmRunner {
    let config = config();
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&config).unwrap(),
    );
    let model = Tiny {
        routed: true,
        ..Tiny::new(false)
    };
    let pooled = meta((SLOTS as usize + 1, TABLE_WIDTH), DType::U32);

    let mut runner = CausalLmRunner::new(
        Arc::new(model),
        config,
        graphs,
        None,
        vec![group(PAGES, pooled)],
        None,
    )
    .unwrap();
    runner.join_union(member).unwrap();

    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        runner.graphs().capture(&runner, &bucket).unwrap();
    }
    runner
}

#[test]
fn a_union_member_captures_each_bucket_at_the_capacities_its_steps_meet() {
    let stated = config().buckets;
    let classes = experts::classes(&stated, UNION_ROWS);

    // Every bucket and the join at the whole capacity; the buckets a step of a quarter of it
    // holds, at the class of their own tokens or of the largest decode.
    let whole = |bucket: Bucket| classes.contains(&bucket.at_capacity(UNION_ROWS));
    assert!(stated.iter().all(|bucket| whole(*bucket)));
    assert!(whole(Bucket::expert_join()));

    let decode = Bucket::decode(2, Variant::DecodeGreedy);
    let short = Bucket::prefill(4, 8, Variant::PrefillOutput);
    let long = Bucket::prefill(4, 80, Variant::PrefillOutput);
    assert!(classes.contains(&decode.at_capacity(4)));
    assert!(classes.contains(&Bucket::expert_join().at_capacity(4)));
    assert!(classes.contains(&short.at_capacity(8)));
    assert_eq!(
        classes
            .iter()
            .filter(|bucket| bucket.at_capacity(0) == long)
            .count(),
        1
    );
    assert_eq!(classes.len(), 2 * 4 + 2 * 4 + 4 + 2);

    // A step runs the smallest class that holds it.
    assert_eq!(classes.at_capacity(decode, 3), Some(decode.at_capacity(4)));
    assert_eq!(
        classes.at_capacity(decode, 5),
        Some(decode.at_capacity(UNION_ROWS))
    );
}

#[test]
fn the_members_of_a_step_agree_on_the_most_tokens_any_of_them_sends() {
    let groups = host_groups(2);
    let (to_first, first_inbox) = crossbeam_channel::unbounded();
    let (to_second, second_inbox) = crossbeam_channel::unbounded();
    let mut first = member(&groups[0], 0, deliver(to_second));
    let mut second = member(&groups[1], 1, deliver(to_first));
    let phase = ipc::PhaseId(0);

    // The first member has a batch of three tokens and the second none: its executor takes
    // the join batch, and both leave the step at the first member's tokens.
    std::thread::scope(|scope| {
        let idle = scope.spawn(|| {
            let join = second_inbox.recv().unwrap();
            assert_eq!(join.shape.variant, Variant::ExpertJoin);
            assert!(join.calls.is_empty());

            second.join(&join).unwrap()
        });

        assert_eq!(first.enter(3, phase, 1).unwrap(), 3);
        assert_eq!(idle.join().unwrap(), Some(3));
    });

    // Both members have a batch: the step holds the larger, and the join each asked of the
    // other finds its step closed.
    std::thread::scope(|scope| {
        let other = scope.spawn(|| second.enter(9, phase, 1).unwrap());

        assert_eq!(first.enter(5, phase, 1).unwrap(), 9);
        assert_eq!(other.join().unwrap(), 9);
    });
    let late = first_inbox.recv().unwrap();
    assert_eq!(first.join(&late).unwrap(), None);
    let late = second_inbox.recv().unwrap();
    assert_eq!(second.join(&late).unwrap(), None);

    // A member enters with tokens of its own, no more than the transport holds, and joins no
    // step past the union's next.
    assert!(matches!(first.enter(0, phase, 1), Err(Error::Invalid(_))));
    assert!(matches!(
        first.enter(UNION_ROWS + 1, phase, 1),
        Err(Error::Invalid(_))
    ));
    let mut ahead = late;
    ahead.id = BatchId(7);
    assert!(matches!(second.join(&ahead), Err(Error::Invariant(_))));
}

#[test]
fn a_member_with_no_batch_is_driven_into_the_step_by_a_join_batch() {
    let groups = host_groups(2);
    let (to_first, _first_inbox) = crossbeam_channel::unbounded();
    let (to_second, second_inbox) = crossbeam_channel::unbounded();
    let mut first = union_runner(member(&groups[0], 0, deliver(to_second)));
    let mut second = union_runner(member(&groups[1], 1, deliver(to_first)));

    // Each bucket at each of its classes and the join at its two, largest first.
    let buckets = first.buckets();
    assert_eq!(buckets.len(), 22);
    assert_eq!(first.graphs().buckets().len(), 22);
    assert!(buckets.contains(&Bucket::expert_join().at_capacity(4)));

    // The join computes the expert layers alone: no output, and no input tensor is read.
    let join = Bucket::expert_join().at_capacity(UNION_ROWS);
    let context = backend::Context::meta(None);
    {
        let _guard = context.enter();
        assert!(first.forward(&join).unwrap().is_empty());
    }
    assert!(first.forward(&Bucket::expert_join()).is_err());

    // The first member's batch blocks in the agreement until the second member's executor
    // takes the join batch. Both then reach the replay of a captured class, which graphs
    // that hold no device refuse.
    std::thread::scope(|scope| {
        let idle = scope.spawn(|| {
            let join = second_inbox.recv().unwrap();
            second.execute(&join)
        });

        let served = first.execute(&decode(&[1, 2, 3], &[0.0; 3]));
        assert!(matches!(served, Err(Error::State(_))), "{served:?}");

        let joined = idle.join().unwrap();
        assert!(matches!(joined, Err(Error::State(_))), "{joined:?}");
    });

    // A member computes no paired forward: its exchange launches wait for the others'.
    assert!(matches!(
        first.prepare_shadow_inputs(&decode(&[1, 2], &[0.0, 0.0])),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn a_union_is_joined_by_a_runner_whose_model_routes_experts() {
    let groups = host_groups(1);
    let hook: JoinHook = Arc::new(|_| Ok(()));
    let shape = TransportShape {
        ranks: 1,
        rank: 0,
        experts: 8,
        hidden: HIDDEN as u32,
        intermediate: 32,
        max_tokens: UNION_ROWS,
        top: 2,
        dtype: DType::BF16,
        device: 0,
        attention_ranks: 0,
    };

    // The model splits no experts.
    let mut runner = runner(false);
    let alone = Member::new(
        Arc::clone(&groups[0]),
        shape,
        Mode::Dispatch,
        Arc::clone(&hook),
    );
    assert!(runner.join_union(alone.unwrap()).is_err());

    // A join batch reaches a runner outside a union.
    let mut join = decode(&[1], &[0.0]);
    join.shape = BatchShape::decode(1, Variant::ExpertJoin);
    assert!(matches!(runner.execute(&join), Err(Error::Invalid(_))));

    // A member that is not the group's.
    let other = TransportShape {
        rank: 1,
        ranks: 2,
        ..shape
    };
    assert!(Member::new(Arc::clone(&groups[0]), other, Mode::Dispatch, hook).is_err());
}

// ---- The runner over the rank's cache and state -------------------------------------------------

/// The recurrent state the stateful test model keeps.
fn state_shape() -> StateShape {
    StateShape {
        layers: 1,
        fields: vec![StateField {
            name: "state",
            dtype: DType::BF16,
            dims: vec![HIDDEN],
        }],
    }
}

/// The rank's cache of the test model: one attention layer of the model's heads over
/// [`PAGES`] units, a row of [`TABLE_WIDTH`] pages per request slot, and the state pages of a
/// model that keeps a recurrent state.
fn cache_config(stateful: bool) -> KvCacheConfig {
    let (heads, dim) = (HEADS as u32, DIM as u32);
    let dtype = ipc::PrefixCacheDtype::BFloat16;
    let layout = if stateful {
        CacheLayout::hybrid(PAGE, 1, heads, dim, dtype, &state_shape())
    } else {
        CacheLayout::dense(PAGE, 1, heads, dim, dtype).validated()
    };

    KvCacheConfig {
        layout: Arc::new(layout.unwrap()),
        units: PAGES as u32,
        slots: SLOTS,
        width: TABLE_WIDTH as u32,
        segments: None,
        workspace_bytes: 1 << 16,
    }
}

/// A runner over `cache` and `state`, with every bucket profiled and captured.
fn over(cache: &KvCacheMemory, state: Option<&StateMemory>) -> CausalLmRunner {
    let config = config();
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&config).unwrap(),
    );
    let model = Arc::new(Tiny::new(state.is_some()));

    let runner = CausalLmRunner::over(model, config, graphs, None, cache, state).unwrap();
    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        runner.graphs().capture(&runner, &bucket).unwrap();
    }
    runner
}

/// The scheduler's page table of `slot` in the attention group: `units` from page zero.
fn block_table(slot: u32, units: &[u32]) -> ipc::BlockTable {
    ipc::BlockTable {
        request_pool_idx: slot,
        group_id: 0,
        start_page: 0,
        unit_ids: units.iter().copied().map(ipc::UnitId).collect(),
        allocated_tokens: units.len() as u32 * PAGE,
    }
}

#[test]
fn a_runner_over_the_rank_s_cache_and_state_profiles_and_captures_every_bucket() {
    // A model that attends over the cache alone, and one that keeps a recurrent state.
    let plain = KvCacheMemory::meta(cache_config(false)).unwrap();
    assert_eq!(over(&plain, None).graphs().buckets().len(), 12);

    let config = cache_config(true);
    let cache = KvCacheMemory::meta(config.clone()).unwrap();
    let state = StateMemory::meta(
        StateConfig {
            shape: state_shape(),
            slots: SLOTS,
            rows: 4,
            trail: 0,
        },
        &config,
    )
    .unwrap();
    assert_eq!(over(&cache, Some(&state)).graphs().buckets().len(), 12);

    // A cache of other request slots than the runner's is not the runner's.
    let mut fewer = cache_config(false);
    fewer.slots = SLOTS - 1;
    let fewer = KvCacheMemory::meta(fewer).unwrap();
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&self::config()).unwrap(),
    );
    let model = Arc::new(Tiny::new(false));
    assert!(CausalLmRunner::over(model, self::config(), graphs, None, &fewer, None).is_err());
}

#[test]
fn the_hook_issues_a_batch_s_page_work_on_the_cache_before_its_replay() {
    let cache = Arc::new(Mutex::new(
        KvCacheMemory::meta(cache_config(false)).unwrap(),
    ));
    let mut runner = over(&cache.lock().unwrap(), None);
    runner.set_page_work(wiring::page_work(
        Arc::clone(&cache),
        None,
        None,
        SCRATCH_PAGES,
    ));

    // Every row of the pooled table names the padding unit before a request holds pages.
    let width = TABLE_WIDTH;
    assert!(
        cache
            .lock()
            .unwrap()
            .host_table(0)
            .unwrap()
            .iter()
            .all(|unit| *unit == 0)
    );

    // A batch on the last slot and the first, with the pages the scheduler gave them. The
    // tables are installed ahead of the replay, which graphs that hold no device refuse.
    let mut batch = decode(&[SLOTS as i32, 1], &[0.0, 0.0]);
    batch.block_tables = vec![block_table(SLOTS, &[3, 5]), block_table(1, &[7])];
    let refused = runner.execute(&batch).unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");

    {
        let cache = cache.lock().unwrap();
        let table = cache.host_table(0).unwrap();

        // Slot `s` owns row `s - 1`, the row the runner's forward gathers for it, and the
        // padding row past the last slot's still names the padding unit.
        assert_eq!(table[..width], [7, 0, 0, 0]);
        assert_eq!(table[(SLOTS as usize - 1) * width..][..width], [3, 5, 0, 0]);
        assert_eq!(table[SLOTS as usize * width..], [0; TABLE_WIDTH]);
        assert_eq!(cache.installed_pages(SLOTS, 0), Some((0, &[3, 5][..])));
        assert_eq!(cache.installed_pages(2, 0), None);
    }

    // A table the cache refuses fails the batch before anything of it is installed.
    let mut outside = decode(&[2, 1], &[0.0, 0.0]);
    outside.block_tables = vec![block_table(2, &[9]), block_table(SLOTS + 1, &[4])];
    let refused = runner.execute(&outside).unwrap_err();
    assert!(matches!(refused, Error::Invalid(_)), "{refused}");
    assert_eq!(cache.lock().unwrap().installed_pages(2, 0), None);
}

#[test]
fn a_paired_forward_computes_over_the_scratch_pages_the_cache_opened() {
    let mut cache = KvCacheMemory::meta(cache_config(false)).unwrap();

    // A cache with no scratch pages lends none, and one whose set is too small for the rows
    // of the largest bucket is refused.
    let mut unlent = over(&cache, None);
    assert!(unlent.lend_scratch(&cache).is_err());
    assert!(matches!(
        unlent.prepare_shadow_inputs(&decode(&[1, 2], &[0.0, 0.0])),
        Err(Error::Unsupported(_))
    ));

    // Four rows of the largest bucket at two scratch pages each.
    cache.open_scratch(4 * SCRATCH_PAGES, None).unwrap();
    let mut runner = over(&cache, None);
    let bucket = runner
        .prepare_shadow_inputs(&decode(&[1, SLOTS as i32], &[0.0, 0.0]))
        .unwrap();
    assert_eq!(bucket, Bucket::decode(2, Variant::DecodeGreedy));

    // The forward over the scratch planes makes the serving forward's launches.
    let serving = operators(&runner, &bucket);
    let context = backend::Context::meta(None);
    context.record_launches();
    {
        let _guard = context.enter();
        runner.forward(&bucket).unwrap();
    }
    let paired: Vec<&str> = context
        .take_launches()
        .into_iter()
        .map(|(operator, _, _)| operator)
        .collect();
    assert_eq!(paired, serving);
}

// ---- Speculation --------------------------------------------------------------------------------

use ipc::DraftWork;

use super::output::verification_output;
use super::speculation::{self, DraftPool};
use crate::lock;

/// The drafts a verification of the speculating test runner holds at most.
const DRAFTS: u32 = 2;

/// The draft pool of the test runner: draft pages of sixteen tokens, each two runs of eight,
/// over sixteen units, in cache group one.
const POOL: DraftPool = DraftPool {
    group: 1,
    span: 2,
    page: 8,
    units: 16,
};

/// The test config with two drafts a verification.
fn speculating_config() -> CausalLmConfig {
    CausalLmConfig {
        drafts: DRAFTS,
        draft_pool: Some(POOL),
        ..config()
    }
}

/// The draft cache of the test drafter: one layer over the pool's runs.
fn draft_cache() -> KvCache {
    let pages = (POOL.span * POOL.units) as usize;
    let plane = || meta((pages, POOL.page as usize, HEADS, DIM), DType::BF16);
    let flags = || meta((pages, 1, 1, 1), DType::U8);

    KvCache {
        block_tables: meta((1, 1), DType::U32),
        page_size: POOL.page,
        layers: vec![KvPlanes {
            index_keys: None,
            key: plane(),
            value: plane(),
            key_initialized: flags(),
            value_initialized: flags(),
            key_scale: None,
            value_scale: None,
        }],
        workspace: meta(1 << 16, DType::U8),
    }
}

/// A speculating runner of [`Tiny`] with its drafter, over graphs that hold no device.
fn speculating(stateful: bool) -> CausalLmRunner {
    let config = speculating_config();
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&config).unwrap(),
    );

    let pooled = meta((SLOTS as usize + 1, TABLE_WIDTH), DType::U32);
    let buffers = |rows: usize| vec![vec![meta((rows, HIDDEN), DType::BF16)]];
    let state = stateful.then(|| (buffers(SLOTS as usize), buffers(SLOTS as usize)));

    let mut runner = CausalLmRunner::new(
        Arc::new(Tiny::drafting(stateful)),
        config,
        graphs,
        None,
        vec![group(PAGES, pooled)],
        state,
    )
    .unwrap();

    // The trail holds the state after each token of the largest verification: four rows of
    // a committed token and two drafts.
    runner
        .set_draft_cache(draft_cache(), stateful.then(|| buffers(12)))
        .unwrap();
    runner
}

/// A verification batch of two calls, stated as the scheduler states it, in a decode's shape:
/// slot 2's committed token 5 and the drafts 6 and 7 from position 10, and slot 4's committed
/// token 8 and the draft 9 from position 20. The first call proposes two drafts and the
/// second one, and the first counts its tokens in its penalties.
fn verification(temperature: f32) -> Batch {
    let call = |request: u32, row: u32| BatchRow {
        request,
        kind: RowKind::Verify,
        cache_group: 0,
        rows: row..row + 1,
    };

    let mut table = commit(&[(2, mode::VERIFY, 0), (4, mode::VERIFY, 1)]);
    table[column::LOGICAL * 2..][..2].copy_from_slice(&[10, 20]);
    table[column::SAMPLING * 2..][..2].copy_from_slice(&[3, 5]);
    table[column::VISIBLE * 2..][..2].copy_from_slice(&[10, 20]);
    table[column::PENALTIES * 2] = 1;

    let columns = BatchColumns {
        rows: BatchRows {
            calls: vec![call(2, 0), call(4, 1)],
            positions: vec![10, 11, 12, 20, 21],
            slot_mapping: vec![26, 27, 28, 52, 53],
            seq_lens: vec![13, 22],
            query_offsets: vec![0, 3, 5],
            logits_rows: vec![0, 1],
        },
        input_ids: vec![5, 6, 7, 8, 9],
        slots: vec![2, 4],
        parameters: [request(temperature), request(temperature)].concat(),
        draws: vec![
            DrawAddress {
                key: 100,
                position: 13,
            },
            DrawAddress {
                key: 101,
                position: 22,
            },
        ],
        commit: table,
        page_size: PAGE,
        max_kv_len: MAX_KV_LEN,
        drafts: vec![
            DraftWork {
                proposals: 2,
                ..DraftWork::default()
            },
            DraftWork {
                proposals: 1,
                ..DraftWork::default()
            },
        ],
        ..BatchColumns::default()
    };

    let variant = if temperature == 0.0 {
        Variant::DecodeGreedy
    } else {
        Variant::DecodeSampled
    };
    batch(BatchShape::prefill(2, 5, variant), columns)
}

/// The layout of the speculating test runner's input tensors, and a block of its bytes that
/// holds something else.
fn speculating_block() -> (Layout, Vec<u8>) {
    let buckets = config().buckets.with_drafts(DRAFTS);
    let layout = Layout::of(buckets.iter(), DRAFTS, POOL.width(MAX_KV_LEN)).unwrap();
    let block = vec![0xA5_u8; layout.bytes() as usize];

    (layout, block)
}

#[test]
fn a_speculating_runner_states_verification_and_drafter_buckets() {
    let runner = speculating(false);
    let buckets = runner.buckets();

    // The eight prefill buckets, a verification bucket for each decode size and selection,
    // and the drafter's bucket behind each prefill and verification size.
    assert_eq!(buckets.len(), 8 + 4 + 4);
    assert!(buckets.windows(2).all(|pair| pair[0] > pair[1]));
    assert!(!buckets.iter().any(|bucket| bucket.variant.is_decode()));

    // A verification of `rows` rows holds a committed token and two drafts a row.
    for rows in [2, 4] {
        for variant in [Variant::VerifyGreedy, Variant::VerifySampled] {
            let bucket = Bucket::verify(rows, DRAFTS, variant);

            assert_eq!(bucket.tokens, 3 * rows);
            assert!(buckets.contains(&bucket), "{bucket}");
        }
        assert!(buckets.contains(&Bucket::draft(rows, 3 * rows)));
    }
    for tokens in [8, 80] {
        assert!(buckets.contains(&Bucket::draft(4, tokens)));
    }

    // A set that verifies no draft is the set itself.
    let plain = config().buckets;
    assert_eq!(
        plain.with_drafts(0).iter().collect::<Vec<_>>(),
        plain.iter().collect::<Vec<_>>()
    );

    // The kept outputs and the input tensors are functions of the config.
    let config = speculating_config();
    let kept = CausalLmRunner::kept_bytes(&config, HIDDEN as u32, DType::BF16).unwrap();
    assert!(kept >= (80 * HIDDEN * 2) as u64);
    assert_eq!(
        CausalLmRunner::kept_bytes(&self::config(), HIDDEN as u32, DType::BF16).unwrap(),
        0
    );
    assert!(
        CausalLmRunner::input_bytes(&config).unwrap()
            > CausalLmRunner::input_bytes(&self::config()).unwrap()
    );
}

#[test]
fn the_verification_and_drafter_buckets_are_profiled_and_captured() {
    let runner = speculating(false);
    runner.graphs().profile_run(&runner).unwrap();

    let reduce = <ops::reduce::Reduce as ops::Op>::NAME;
    let drafts = DRAFTS as usize;

    for bucket in runner.buckets() {
        let launches = operators(&runner, &bucket);

        match bucket.variant {
            Variant::VerifyGreedy | Variant::VerifySampled => {
                // One forward over every token, one selection, one commit.
                for operator in ["kv_write", "paged_attention", "logits_shape", "slot_commit"] {
                    assert_eq!(count(&launches, operator), 1, "{operator} in {bucket}");
                }
                let greedy = usize::from(bucket.variant.greedy());
                assert_eq!(count(&launches, "argmax"), greedy, "{bucket}");
                assert_eq!(count(&launches, "inverse_cdf"), 1 - greedy, "{bucket}");

                // The agreed lengths are one reduction over the rows' runs.
                assert_eq!(count(&launches, reduce), 1, "{bucket}");
            }
            Variant::Draft => {
                // The entries and one chained step write and read the draft cache, and each
                // step proposes; the drafter selects nothing through the sampler and commits
                // nothing.
                for operator in ["kv_write", "paged_attention", "argmax"] {
                    assert_eq!(count(&launches, operator), drafts, "{operator} in {bucket}");
                }
                for operator in ["logits_shape", "slot_commit"] {
                    assert_eq!(count(&launches, operator), 0, "{operator} in {bucket}");
                }
            }
            _ => {}
        }
    }

    // A verification's tokens attend as decode rows; the drafter's entries attend as a
    // prompt's rows and its chained steps as decode rows.
    let phases = |bucket: &Bucket| -> Vec<bool> {
        let table = runner.graphs().launch_table(bucket).unwrap();

        table
            .iter()
            .filter(|(operator, _, _)| *operator == "paged_attention")
            .map(|(_, key, _)| key.contains("phase_decode"))
            .collect()
    };
    assert_eq!(
        phases(&Bucket::verify(4, DRAFTS, Variant::VerifySampled)),
        [true]
    );
    assert_eq!(phases(&Bucket::draft(4, 12)), [false, true]);

    // Every bucket is captured with its profile run's launches.
    for bucket in runner.buckets() {
        runner.graphs().capture(&runner, &bucket).unwrap();
    }
    let mut ascending = runner.buckets();
    ascending.reverse();
    assert_eq!(runner.graphs().buckets(), ascending);

    // The model's one host value is copied by the forwards that run it; the drafter's makes
    // no copy.
    let copies = |bucket: &Bucket| runner.graphs().input_copies(bucket).unwrap().len();
    assert_eq!(copies(&Bucket::verify(2, DRAFTS, Variant::VerifyGreedy)), 1);
    assert_eq!(copies(&Bucket::draft(2, 6)), 0);
}

#[test]
fn a_verification_of_a_model_that_keeps_a_recurrent_state_settles_it_on_its_trail() {
    let index = <ops::index::Index as ops::Op>::NAME;
    let bucket = Bucket::verify(4, DRAFTS, Variant::VerifyGreedy);

    let plain = speculating(false);
    plain.graphs().profile_run(&plain).unwrap();
    let stateful = speculating(true);
    stateful.graphs().profile_run(&stateful).unwrap();

    // The state's one field is settled by a gather from the trail and a scatter into the
    // live buffer, which a model without a state does not launch.
    let gathers = |runner: &CausalLmRunner| count(&operators(runner, &bucket), index);
    assert_eq!(gathers(&stateful), gathers(&plain) + 2);

    for bucket in stateful.buckets() {
        stateful.graphs().capture(&stateful, &bucket).unwrap();
    }

    // A model that keeps a state verifies only with a trail of its largest verification.
    let mut unlent = speculating(true);
    let short = vec![vec![meta((11, HIDDEN), DType::BF16)]];
    assert!(unlent.set_draft_cache(draft_cache(), Some(short)).is_err());
    assert!(unlent.set_draft_cache(draft_cache(), None).is_err());
}

#[test]
fn a_verification_batch_is_one_row_per_token_and_one_sequence_per_call() {
    let batch = verification(0.7);
    let bucket = Bucket::verify(4, DRAFTS, Variant::VerifySampled);
    let (layout, mut block) = speculating_block();

    speculation::fill_verification(
        &layout,
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        &speculation::token_rows(&batch.columns),
        &bucket,
        &mut block,
    )
    .unwrap();

    let u32s = |column, elements| layout.read::<u32>(&block, column, elements);
    let i32s = |column, elements| layout.read::<i32>(&block, column, elements);
    let none = NO_SLOT;

    // ---- Each token is the decode step at its position ----

    assert_eq!(u32s(Column::InputIds, 6), [5, 6, 7, 8, 9, 0]);
    assert_eq!(i32s(Column::Positions, 6), [10, 11, 12, 20, 21, 0]);
    assert_eq!(u32s(Column::SeqLens, 6), [11, 12, 13, 21, 22, 0]);
    assert_eq!(
        u32s(Column::QueryOffsets, 13),
        [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5]
    );
    assert_eq!(u32s(Column::LogitsRows, 6), [0, 1, 2, 3, 4, 0]);

    // A token reads its request's pages and samples on its request's slot.
    assert_eq!(u32s(Column::Slots, 6), [1, 1, 1, 3, 3, none]);
    assert_eq!(u32s(Column::TableRows, 6), [1, 1, 1, 3, 3, SLOTS]);
    assert_eq!(u32s(Column::CallSlots, 3), [1, 3, SLOTS]);

    // A token draws at the position after it in its request's stream.
    let draws = layout.read::<f32>(&block, Column::Draws, 6);
    let expected = [
        sampling_uniform(100, 11, 0, 0),
        sampling_uniform(100, 12, 0, 0),
        sampling_uniform(100, 13, 0, 0),
        sampling_uniform(101, 21, 0, 0),
        sampling_uniform(101, 22, 0, 0),
    ];
    assert_eq!(draws[..5], expected);
    assert_eq!(draws[5], 0.0);

    // ---- Each call is one sequence of the recurrent state ----

    assert_eq!(u32s(Column::StateOffsets, 5), [0, 3, 5, 5, 5]);
    assert_eq!(u32s(Column::StateLens, 4), [13, 22, 0, 0]);
    assert_eq!(u32s(Column::StateSlots, 4), [1, 3, none, none]);
    assert_eq!(u32s(Column::Firsts, 4), [0, 3, 0, 0]);

    // ---- The drafts ----

    // A token's sample is compared with the token after it, and its penalties count the
    // drafts before it.
    assert_eq!(u32s(Column::Drafts, 6), [6, 7, none, 9, none, none]);
    assert_eq!(
        u32s(Column::Extras, 12),
        [none, none, 6, none, 6, 7, none, none, 9, none, none, none]
    );

    // A row's window ends at its last token, whose sample no draft follows.
    assert_eq!(
        u32s(Column::Windows, 12),
        [0, 1, 2, 3, 4, 4, none, none, none, none, none, none]
    );
    assert_eq!(
        u32s(Column::DraftTokens, 8),
        [6, 7, 9, none, none, none, none, none]
    );
    assert_eq!(u32s(Column::DraftSteps, 8), [0, 1, 0, 1, 0, 1, 0, 1]);

    // ---- The commit: a row per call, then a count row per draft ----

    let rows = layout.rows();
    let table = i32s(Column::Commit, COMMIT_FIELDS * rows);
    let field = |field: usize, row: usize| table[field * rows + row];

    assert_eq!((field(column::SLOT, 0), field(column::TASK, 0)), (1, 0));
    assert_eq!((field(column::SLOT, 1), field(column::TASK, 1)), (3, 1));
    assert_eq!(field(column::MODE, 1), mode::VERIFY as i32);
    assert_eq!(field(column::VISIBLE, 1), 20);

    // The counts of the bucket's four rows start behind them: two for the first call, which
    // counts its tokens, and one for the second, which does not.
    for (task, slot, counts) in [(4, 1, 1), (5, 1, 1), (6, 3, 0)] {
        assert_eq!(field(column::SLOT, task), slot);
        assert_eq!(field(column::MODE, task), mode::COUNT as i32);
        assert_eq!(field(column::TASK, task), task as i32);
        assert_eq!(field(column::PENALTIES, task), counts);
    }
    for inert in [2, 3, 7, 8, 11] {
        assert_eq!(
            (field(column::SLOT, inert), field(column::TASK, inert)),
            (-1, -1)
        );
    }
}

#[test]
fn the_drafter_s_columns_follow_a_verification_and_a_prompt() {
    let (layout, mut block) = speculating_block();
    let none = NO_SLOT;

    // ---- Behind a verification ----

    // Slot 2 holds the draft pages 0 and 1 in units 3 and 5; slot 4 holds none.
    let batch = verification(0.0);
    let tables = [(0, vec![3, 5]), (0, Vec::new())];
    let bucket = Bucket::draft(4, 12);
    speculation::fill_drafter(
        &layout,
        &POOL,
        VOCABULARY as u32,
        &batch.columns,
        &tables,
        &bucket,
        &mut block,
    )
    .unwrap();

    {
        let u32s = |column, elements| layout.read::<u32>(&block, column, elements);
        let stride = layout.rows();

        // An entry reads the token its own sampling row chose, and is written while its
        // offset is within the length its verification agreed.
        assert_eq!(u32s(Column::NextRows, 6), [0, 1, 2, 3, 4, none]);
        assert_eq!(u32s(Column::NextIds, 6), [0; 6]);
        assert_eq!(u32s(Column::EntryRows, 6), [0, 0, 0, 1, 1, none]);
        assert_eq!(u32s(Column::EntrySteps, 6), [0, 1, 2, 0, 1, 1]);

        // Position 10 is run 1 of draft page 0, in unit 3: page 16 + 3, offset 2. A slot
        // that holds no draft page writes nothing.
        assert_eq!(
            u32s(Column::DraftSlots, 6),
            [19 * 8 + 2, 19 * 8 + 3, 19 * 8 + 4, none, none, none]
        );
        assert_eq!(u32s(Column::NoSlots, 12), [none; 12]);
        assert_eq!(u32s(Column::TokenCap, 12), [VOCABULARY as u32 - 1; 12]);

        // A drafter row is a call's tokens, and reads the keys up to its last one.
        assert_eq!(u32s(Column::StateOffsets, 5), [0, 3, 5, 5, 5]);
        assert_eq!(u32s(Column::DraftLens, 4), [13, 22, 0, 0]);

        // A row's table names run `c` of the draft page in unit `u` as page `c · 16 + u`,
        // and the padding unit's pages where it holds none.
        assert_eq!(
            u32s(Column::DraftTables, 16),
            [3, 19, 5, 21, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16]
        );

        // The first proposal continues from the entry the agreed length names, from the
        // row's first token on.
        assert_eq!(u32s(Column::LastEntries, 4), [0, 3, 0, 0]);
        assert_eq!(u32s(Column::ProposalRows, 4), [0, 1, none, none]);

        // The chained step evaluates the position after the row's first, moved on the
        // device by the length it agreed.
        let positions = layout.read::<i32>(&block, Column::ChainPositions, 2 * stride);
        assert_eq!(positions[stride..][..4], [11, 21, 0, 0]);
        let lens = u32s(Column::ChainLens, 2 * stride);
        assert_eq!(lens[stride..][..4], [12, 22, 0, 0]);

        // Positions 11 to 13 of slot 2, then cells that hold no slot.
        let slots = u32s(Column::ChainSlots, 12);
        assert_eq!(
            slots[..6],
            [19 * 8 + 3, 19 * 8 + 4, 19 * 8 + 5, none, none, none]
        );
        assert_eq!(slots[6..], [none; 6]);

        // The first call proposes two drafts, so its chained step writes its slot; the
        // second proposes one and a row the batch does not hold none, and both read a cell
        // that holds no slot.
        let index = u32s(Column::ChainIndex, 2 * stride);
        assert_eq!(index[stride..][..4], [0, 6 + 3, 12 + 3, 18 + 3]);

        assert_eq!(u32s(Column::ChainOffsets, 5), [0, 1, 2, 2, 2]);
        assert_eq!(u32s(Column::LiveRows, 1), [2]);
    }

    // ---- Behind a prompt ----

    // Slot 2's three tokens end its prompt and sample; slot 4's two end none, and its call
    // states the token after them.
    let mut batch = prefill(Variant::PrefillGreedy);
    batch.columns.drafts = vec![
        DraftWork {
            proposals: 2,
            ..DraftWork::default()
        },
        DraftWork {
            lookahead: Some(33),
            ..DraftWork::default()
        },
    ];
    let bucket = Bucket::draft(4, 8);
    speculation::fill_drafter(
        &layout,
        &POOL,
        VOCABULARY as u32,
        &batch.columns,
        &tables,
        &bucket,
        &mut block,
    )
    .unwrap();

    {
        let u32s = |column, elements| layout.read::<u32>(&block, column, elements);
        let stride = layout.rows();

        // An entry reads the prompt's next token; the last one of a row that samples reads
        // the row's token on the device, and the last one of a row that does not its call's
        // lookahead.
        assert_eq!(u32s(Column::NextIds, 6), [6, 7, 0, 9, 33, 0]);
        assert_eq!(u32s(Column::NextRows, 6), [none, none, 0, none, none, none]);
        assert_eq!(u32s(Column::EntryRows, 6), [none; 6]);
        assert_eq!(u32s(Column::EntrySteps, 6), [0, 0, 0, 0, 0, 1]);
        assert_eq!(u32s(Column::DraftSlots, 3), [3 * 8, 3 * 8 + 1, 3 * 8 + 2]);

        // A prompt's proposals continue from its last entry, which the batch states.
        assert_eq!(u32s(Column::LastEntries, 4), [2, 4, 0, 0]);
        assert_eq!(u32s(Column::ProposalRows, 4), [none; 4]);
        assert_eq!(u32s(Column::DraftLens, 4), [3, 2, 0, 0]);

        let positions = layout.read::<i32>(&block, Column::ChainPositions, 2 * stride);
        assert_eq!(positions[stride..][..4], [3, 2, 0, 0]);
        let index = u32s(Column::ChainIndex, 2 * stride);
        assert_eq!(index[stride..][..4], [0, 6 + 3, 12 + 3, 18 + 3]);
    }

    // A row whose last token's successor nobody states has no entry there.
    batch.columns.drafts.clear();
    speculation::fill_drafter(
        &layout,
        &POOL,
        VOCABULARY as u32,
        &batch.columns,
        &tables,
        &bucket,
        &mut block,
    )
    .unwrap();

    assert_eq!(
        layout.read::<u32>(&block, Column::EntrySteps, 6),
        [0, 0, 0, 0, 1, 1]
    );
    assert_eq!(layout.read::<u32>(&block, Column::LastEntries, 2), [2, 3]);
}

#[test]
fn a_verification_agrees_the_drafts_its_samples_equal_up_to_the_first_that_differs() {
    let cpu = Device::Cpu;
    let none = NO_SLOT;

    // Four verifications, packed: three drafts of which two agree; two drafts that both
    // agree; two drafts of which the first differs; and a committed token alone.
    let chosen = [7_u32, 8, 4, 5, 7, 8, 3, 1, 8, 3, 5];
    let drafts = [7_u32, 8, 9, none, 7, 8, none, 7, 8, none, none];
    let windows = [0_u32, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 10, 10];
    let firsts = [0_u32, 4, 7, 10];

    let (agreed, last) = speculation::accept(
        &Tensor::from_slice(&chosen, 11, &cpu).unwrap(),
        &Tensor::from_slice(&drafts, 11, &cpu).unwrap(),
        &Tensor::from_slice(&windows, (4, 4), &cpu).unwrap(),
        &Tensor::from_slice(&firsts, 4, &cpu).unwrap(),
    )
    .unwrap();

    // A verification commits its agreed drafts and the sample after them: three, three, one
    // and one tokens.
    assert_eq!(agreed.to_vec1::<u32>().unwrap(), [2, 2, 0, 0]);
    assert_eq!(last.to_vec1::<u32>().unwrap(), [2, 6, 7, 10]);

    // A row the batch does not hold agrees nothing.
    let (agreed, last) = speculation::accept(
        &Tensor::from_slice(&[0_u32, 0], 2, &cpu).unwrap(),
        &Tensor::from_slice(&[none, none], 2, &cpu).unwrap(),
        &Tensor::from_slice(&[none; 4], (2, 2), &cpu).unwrap(),
        &Tensor::from_slice(&[0_u32, 0], 2, &cpu).unwrap(),
    )
    .unwrap();
    assert_eq!(agreed.to_vec1::<u32>().unwrap(), [0, 0]);
    assert_eq!(last.to_vec1::<u32>().unwrap(), [0, 0]);
}

#[test]
fn a_verification_s_call_reports_its_accepted_prefix_and_its_proposals() {
    let batch = verification(0.0);
    let bucket = Bucket::verify(4, DRAFTS, Variant::VerifyGreedy);

    // The tokens sampled after each of the batch's five tokens, their scores, each call's
    // positions after the commit, and the agreed lengths: both of the first call's drafts,
    // and none of the second's one.
    let mut tokens = vec![6_u32, 7, 40, 30, 31];
    tokens.resize(12, 0);
    let mut logprobs = vec![-0.1_f32, -0.2, -0.3, -0.4, -0.5];
    logprobs.resize(12, 0.0);

    let outputs = |tokens: &[u32]| {
        vec![
            tensor(tokens, 12),
            tensor(&logprobs, 12),
            tensor(&[1_i32; 12], 12),
            tensor(&[13_i32, 21, 0, 0], 4),
            tensor(&[6_i32, 6, 0, 0], 4),
            tensor(&[2_u32, 0, 0, 0], 4),
        ]
    };
    let completion = |tokens: &[u32]| {
        verification_output(
            &batch.calls,
            &batch.columns,
            &bucket,
            VOCABULARY as u32,
            0,
            &outputs(tokens),
        )
        .unwrap()
    };
    let mut calls = completion(&tokens);

    // The first call commits its two drafts and the sample after them; the second its
    // first sample alone, as a decode at its position would.
    assert_eq!(calls[0].tokens, [6, 7, 40]);
    assert_eq!(calls[0].token_logprobs.len(), 3);
    assert_eq!(calls[0].token_logprobs[2], [entry(40, -0.3, 1)]);
    assert_eq!(
        (calls[0].cache_computed_len, calls[0].cache_visible_len),
        (13, 6)
    );
    assert_eq!(calls[1].tokens, [30]);
    assert_eq!(calls[1].token_logprobs, [vec![entry(30, -0.4, 1)]]);
    assert_eq!(calls[1].cache_computed_len, 21);

    // Each chained step proposes for every row. A call takes as many as it asked for, up
    // to the first that is no token.
    let proposals = [
        tensor(&[50_u32, 60, 0, 0], 4),
        tensor(&[51_u32, 61, 0, 0], 4),
    ];
    speculation::propose(&batch, VOCABULARY as u32, &proposals, &mut calls).unwrap();
    assert_eq!(calls[0].draft_token_ids, [50, 51]);
    assert_eq!(calls[1].draft_token_ids, [60]);

    let ended = [
        tensor(&[VOCABULARY as u32, 60, 0, 0], 4),
        tensor(&[51_u32, 61, 0, 0], 4),
    ];
    let mut calls = completion(&tokens);
    speculation::propose(&batch, VOCABULARY as u32, &ended, &mut calls).unwrap();
    assert!(calls[0].draft_token_ids.is_empty());

    // A call one of whose samples is no token fails alone and proposes nothing.
    tokens[1] = VOCABULARY as u32;
    let mut calls = completion(&tokens);
    speculation::propose(&batch, VOCABULARY as u32, &proposals, &mut calls).unwrap();
    assert_eq!(calls[0].status, CallStatus::Error);
    assert_eq!(calls[0].error_code, Some(ErrorCode::NumericalFault));
    assert!(calls[0].tokens.is_empty() && calls[0].draft_token_ids.is_empty());
    assert_eq!(calls[1].tokens, [30]);

    // A prompt's call proposes from the row its last entry is in: the first samples the
    // token after its prompt, and the second states no token after its own.
    let prompt = prefill(Variant::PrefillGreedy);
    let mut prompt = Batch {
        columns: BatchColumns {
            drafts: vec![
                DraftWork {
                    proposals: 1,
                    ..DraftWork::default()
                };
                2
            ],
            ..prompt.columns.clone()
        },
        ..prompt
    };
    prompt.columns.rows.query_offsets = vec![0, 3, 4];
    prompt.columns.rows.positions.truncate(4);
    prompt.columns.rows.slot_mapping.truncate(4);
    prompt.columns.input_ids.truncate(4);

    let mut calls = vec![
        ipc::execution::CallOutput::completed(CallId(100), vec![11], RowState::default()),
        ipc::execution::CallOutput::completed(CallId(101), Vec::new(), RowState::default()),
    ];
    speculation::propose(&prompt, VOCABULARY as u32, &proposals, &mut calls).unwrap();
    assert_eq!(calls[0].draft_token_ids, [50]);
    assert!(calls[1].draft_token_ids.is_empty());
}

#[test]
fn a_speculating_runner_serves_a_verification_from_its_verification_bucket() {
    let mut runner = speculating(false);
    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        runner.graphs().capture(&runner, &bucket).unwrap();
    }

    // A batch in a decode's shape whose calls verify takes the verification bucket of its
    // selection, the smallest that holds its calls.
    let bucket = runner.prepare_inputs(&verification(0.0)).unwrap();
    assert_eq!(bucket, Bucket::verify(2, DRAFTS, Variant::VerifyGreedy));
    let bucket = runner.prepare_inputs(&verification(0.7)).unwrap();
    assert_eq!(bucket, Bucket::verify(2, DRAFTS, Variant::VerifySampled));

    // A decode is a verification of no draft.
    let mut decoded = decode(&[1, 2, 3], &[0.0; 3]);
    decoded.columns.input_ids = vec![4, 5, 6];
    let bucket = runner.prepare_inputs(&decoded).unwrap();
    assert_eq!(bucket, Bucket::verify(4, DRAFTS, Variant::VerifyGreedy));

    // A prompt is served from its prefill bucket.
    let bucket = runner
        .prepare_inputs(&prefill(Variant::PrefillGreedy))
        .unwrap();
    assert_eq!(bucket, Bucket::prefill(4, 8, Variant::PrefillGreedy));

    // More drafts than the deployment verifies, and prompt rows beside verifications.
    let mut long = verification(0.0);
    long.columns.rows.calls.truncate(1);
    long.columns.rows.seq_lens = vec![15];
    long.columns.rows.query_offsets = vec![0, 5];
    long.columns.rows.logits_rows.truncate(1);
    long.columns.slots.truncate(1);
    long.columns.parameters.truncate(PARAMETER_COLUMNS as usize);
    long.columns.draws.truncate(1);
    long.columns.commit = commit(&[(2, mode::VERIFY, 0)]);
    long.columns.drafts.truncate(1);
    long.calls.truncate(1);
    let message = runner.prepare_inputs(&long).unwrap_err().to_string();
    assert!(message.contains("carries 4 drafts"), "{message}");

    let mut mixed = verification(0.0);
    mixed.columns.rows.calls[1].kind = RowKind::Prefill;
    let message = runner.prepare_inputs(&mixed).unwrap_err().to_string();
    assert!(message.contains("mixes prompt rows"), "{message}");

    // A verification's token rows share their request's pages: it has no paired forward.
    assert!(matches!(
        runner.prepare_shadow_inputs(&verification(0.0)),
        Err(Error::Unsupported(_))
    ));

    // Graphs that hold no device replay nothing: the step is refused at its replay, after
    // both graphs' columns were padded.
    let refused = runner.execute(&verification(0.0)).unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");
}

#[test]
fn speculation_is_refused_where_the_runner_does_not_serve_it() {
    // A config that states drafts for a model that loaded no drafter, or without the draft
    // pool of its cache.
    let build = |model: Tiny, config: CausalLmConfig| {
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            CausalLmRunner::input_bytes(&config)?,
        );
        let pooled = meta((SLOTS as usize + 1, TABLE_WIDTH), DType::U32);

        CausalLmRunner::new(
            Arc::new(model),
            config,
            graphs,
            None,
            vec![group(PAGES, pooled)],
            None,
        )
    };
    let refused = build(Tiny::new(false), speculating_config()).err().unwrap();
    assert!(
        refused.to_string().contains("loaded no drafter"),
        "{refused}"
    );

    let unpooled = CausalLmConfig {
        draft_pool: None,
        ..speculating_config()
    };
    assert!(matches!(
        build(Tiny::drafting(false), unpooled),
        Err(Error::Invalid(_))
    ));

    // A runner whose config states no drafts is lent no draft cache.
    assert!(runner(false).set_draft_cache(draft_cache(), None).is_err());

    // Experts split across ranks.
    let groups = host_groups(2);
    let (inbox, _outbox) = crossbeam_channel::unbounded();
    let refused = speculating(false)
        .join_union(member(&groups[0], 0, deliver(inbox)))
        .unwrap_err();
    assert!(matches!(refused, Error::Unsupported(_)), "{refused}");

    // A refill of the draft cache of a model that keeps a recurrent state.
    let mut stateful = speculating(true);
    let mut refill = prefill(Variant::PrefillWrite);
    refill.columns.rows.logits_rows = vec![-1, -1];
    refill.columns.commit = commit(&[(2, mode::CACHE, -1), (4, mode::CACHE, -1)]);
    refill.columns.checkpoints = vec![0, 0];
    refill.columns.drafts = vec![
        DraftWork {
            refill: Some(0),
            lookahead: Some(9),
            ..DraftWork::default()
        },
        DraftWork::default(),
    ];
    let refused = stateful.prepare_inputs(&refill).unwrap_err();
    assert!(matches!(refused, Error::Unsupported(_)), "{refused}");
}

#[test]
fn a_draft_page_is_read_as_runs_of_the_pool_s_pages() {
    // Draft pages of 512 tokens in eight runs of 64, in a pool of 100 units.
    let pool = DraftPool {
        group: 1,
        span: 8,
        page: 64,
        units: 100,
    };
    assert_eq!(pool.width(2048), 32);

    // A table from draft page 2 that holds unit 7 there and none for page 3.
    let units = [7, 0];

    // Position 1100 is draft page 2, run 1, offset 12: run 1 of unit 7 is page 107.
    assert_eq!(pool.slot(2, &units, 1100), 107 * 64 + 12);

    // Page 3 holds no unit, and neither does a page before the table's first: nothing is
    // written there.
    assert_eq!(pool.slot(2, &units, 1600), NO_SLOT);
    assert_eq!(pool.slot(2, &units, 10), NO_SLOT);

    // A row names each run of a draft page it holds, and the padding unit's elsewhere.
    let mut row = Vec::new();
    pool.row(2, &units, 18, &mut row);
    assert_eq!(row[16..18], [7, 100 + 7]);
    assert_eq!(row[..2], [0, 100]);
}

#[test]
fn a_speculating_runner_is_built_over_the_rank_s_draft_group_and_trail() {
    let (heads, dim) = (HEADS as u32, DIM as u32);
    let dtype = ipc::PrefixCacheDtype::BFloat16;
    let layout = CacheLayout::hybrid(PAGE, 1, heads, dim, dtype, &state_shape())
        .and_then(|layout| layout.with_draft(1, heads, dim))
        .unwrap();
    let cache_config = KvCacheConfig {
        layout: Arc::new(layout),
        ..cache_config(true)
    };

    let cache = Arc::new(Mutex::new(
        KvCacheMemory::meta(cache_config.clone()).unwrap(),
    ));
    let pool = DraftPool::of(&cache_config.layout, cache_config.units).unwrap();
    assert_eq!(pool.units, PAGES as u32);
    assert_eq!(Some(pool.group as usize), cache_config.layout.draft_group());

    let config = CausalLmConfig {
        drafts: DRAFTS,
        draft_pool: Some(pool),
        ..config()
    };
    let state = |trail: u32| {
        StateMemory::meta(
            StateConfig {
                shape: state_shape(),
                slots: SLOTS,
                rows: 4,
                trail,
            },
            &cache_config,
        )
        .unwrap()
    };
    let over = |config: CausalLmConfig, state: &StateMemory| {
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            CausalLmRunner::input_bytes(&config)?,
        );
        let model = Arc::new(Tiny::drafting(true));

        CausalLmRunner::over(model, config, graphs, None, &lock(&cache), Some(state))
    };

    // The runner is lent the draft group's planes and the state's trail, and captures its
    // verification and drafter buckets over them.
    let mut runner = over(config.clone(), &state(12)).unwrap();
    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        runner.graphs().capture(&runner, &bucket).unwrap();
    }
    assert_eq!(runner.graphs().buckets().len(), 16);

    // A state opened with no trail, or with one shorter than a verification, verifies
    // nothing; neither does a config that states another draft pool than the cache's.
    assert!(over(config.clone(), &state(0)).is_err());
    assert!(over(config.clone(), &state(11)).is_err());
    let moved = CausalLmConfig {
        draft_pool: Some(DraftPool {
            units: pool.units + 1,
            ..pool
        }),
        ..config
    };
    assert!(over(moved, &state(12)).is_err());

    // A slot's draft pages are the ones the scheduler installed for it.
    let mut pages = wiring::draft_pages(Arc::clone(&cache));
    assert_eq!(pages(2), None);

    let installed = ipc::BlockTable {
        request_pool_idx: 2,
        group_id: pool.group,
        start_page: 0,
        unit_ids: vec![ipc::UnitId(3)],
        allocated_tokens: pool.page * pool.span,
    };
    lock(&cache).install(&[installed], None).unwrap();
    assert_eq!(pages(2), Some((0, vec![3])));
    runner.set_draft_pages(pages);
}
