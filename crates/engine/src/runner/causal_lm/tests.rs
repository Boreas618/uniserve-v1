//! `CausalLmRunner` without a device: the bucket set, the profile run and the meta capture of
//! every bucket, the bytes of the input tensors, the cache slots a batch writes through the
//! rank's device block table, the refusals, the page-work seam, the paired forward's inputs,
//! the completion and the step agreement of an expert union, over a small causal language
//! model of `nn` operators on meta weights.
//!
//! The model exercises the cache through the modules a served model uses: its one layer
//! writes its keys and values (`kv_write_`) and attends over them (`paged_attention`) through
//! the page tables the runner gathers.

use std::sync::{Arc, Mutex};

use crate::batch::philox::{DrawAddress, sampling_uniform};
use crate::batch::{
    Batch, BatchId, BatchOutput, CallOutput, CallStatus, Columns, ErrorCode, NewSlot, Phase, Row,
    Rows, Shape as BatchShape, StateCopy, StateDirection, Variant,
};
use nn::capability::Drafter;
use nn::family::{CacheField, ExpertShape, StateField, StateShape, TextModel, TokenHistory};
use nn::functional::{
    WeightOrder, embedding, kv_write_, paged_attention, rms_norm, zero_arrival_counters_,
};
use nn::modules::Linear;
use nn::{CausalLm, ExpertExchange, KvCache, NO_SLOT, RecurrentState, TokenBatch};
use ops::Generation;
use ops::moe::exchange::{Mode, TransportShape};
use proto::module::commit::{COMMIT_FIELDS, column, mode};
use proto::module::sampler::{PARAMETER_COLUMNS, TEMPERATURE, TOP_K, TOP_P};
use proto::request::{RequestId, TokenLogprob};
use storage::Key;
use storage::runtime::{Completion, GroupId, OwnerId, Request, Run};
use tensor::{DType, Device, Tensor, WithDType};

use super::columns::{self, Column, Layout};
use super::experts::{self, JoinHook, Member};
use super::output::{calls_output, refuse};
use super::wiring;
use super::{CausalLmConfig, CausalLmRunner, PageWork};
use crate::memory::cache::{CacheLayout, CacheRegions, KvCacheConfig, KvCacheMemory};
use crate::memory::state::{StateConfig, StateMemory};
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::parallel::{HostGroup, Rendezvous};
use crate::process::ranks::union::Joins;
use crate::runner::buckets::{Bucket, Buckets};
use crate::runner::graphs::Graphs;
use crate::runner::{Forward, Runner};
use crate::worker::Error;
use crate::worker::completion::Outcome;
use crate::worker::{Sequences, Worker};
use testkit::serving::unlaunched_kernel;

/// The first `elements` elements of `column` in `block`, as [`Layout::put`] wrote them: what a
/// prepared step's columns hand the device.
fn read<T: bytemuck::Pod>(
    layout: &Layout,
    block: &[u8],
    column: Column,
    elements: usize,
) -> Vec<T> {
    let start = layout.offset(column) as usize;
    let bytes = elements * std::mem::size_of::<T>();

    bytemuck::pod_collect_to_vec(&block[start..start + bytes])
}

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
/// The pages of the cache's planes, and of the scratch page set the rows of a paired forward
/// share: the largest bucket's eighty tokens are five pages, and its four rows one page more
/// each.
const PAGES: usize = 32;
const SCRATCH_PAGES: u32 = 9;
/// The most likely tokens a scored row reports.
const CANDIDATES: u32 = 3;
/// The ids before each token a model that reads token ids reads, and the id that stands for a
/// position before a request's first token.
const HISTORY: u32 = 2;
const FILL: u32 = 0;

/// The cache fields of the test model and of its drafter: the keys and values of one layer of
/// [`HEADS`] heads of [`DIM`].
fn fields() -> Vec<CacheField> {
    CacheField::keys_and_values(1, HEADS as u32, DIM as u32)
}

/// A meta tensor: a shape and an element type at an address nothing reads.
fn meta(shape: impl Into<tensor::Shape>, dtype: DType) -> Tensor {
    ops::retained(0x1_0000, shape, dtype, &ops::meta()).unwrap()
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
    /// Whether the state keeps a second run of layers, whose field keeps a slot per verified
    /// token of a deployment that drafts [`DRAFTS`] tokens: its forward then asks for the
    /// rows' verify slots.
    stepwise: bool,
    /// Whether the model's layers read token ids, which its forward then asks for.
    reads: bool,
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
            stepwise: false,
            reads: false,
            routed: false,
            drafter: None,
        }
    }

    /// The model with its drafter loaded.
    fn drafting(stateful: bool) -> Self {
        Self {
            drafter: Some(TinyDrafter::new(false)),
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
    /// Whether the layer keeps a state of one layer, [`drafter_ring`]: its forward then asks
    /// for that state, and its rows are read through its norm for their logits.
    stateful: bool,
}

/// The state of a drafter that keeps one: a ring of four hidden rows a request slot.
fn drafter_ring() -> StateShape {
    StateShape {
        layers: 1,
        fields: vec![StateField {
            name: "ring",
            dtype: DType::BF16,
            dims: vec![4, HIDDEN],
            stepwise: false,
            rows: nn::family::StateRows::Whole,
        }],
    }
}

impl TinyDrafter {
    fn new(stateful: bool) -> Self {
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
            stateful,
        }
    }

    /// The layer over the batch: the rows the drafter returns.
    fn attend(
        &self,
        batch: &TokenBatch,
        cache: &KvCache,
        embedded: &Tensor,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        let tokens = batch.tokens();
        let fused = embedded.add(hidden)?;

        zero_arrival_counters_(&cache.workspace, batch.rows(), HEADS as u32)?;
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

impl Drafter for TinyDrafter {
    fn cache(&self) -> Vec<CacheField> {
        fields()
    }

    fn state_layers(&self) -> Vec<StateShape> {
        if self.stateful {
            vec![drafter_ring()]
        } else {
            Vec::new()
        }
    }

    /// A drafter that keeps a state asks for its live rows alone, a state row per row of the
    /// batch: no checkpoint, trail or verify slots, and no row kept; one that keeps none is
    /// lent none.
    fn forward(
        &self,
        batch: &TokenBatch,
        cache: &KvCache,
        state: Option<&RecurrentState>,
        embedded: &Tensor,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        if !self.stateful {
            if state.is_some() {
                return Err(nn::Error::Unsupported(
                    "the drafter keeps no state, and was lent one".to_owned(),
                ));
            }
            return self.attend(batch, cache, embedded, hidden);
        }

        let rows = batch.rows();
        let lent = state.is_some_and(|state| {
            let [layer] = state.live.as_slice() else {
                return false;
            };

            matches!(layer.as_slice(), [ring] if ring.dims() == [SLOTS as usize, 4, HIDDEN])
                && state.slots.dims() == [rows]
                && state.checkpoints.dims() == [rows]
                && state.query_offsets.dims() == [rows + 1]
                && state.checkpoint.is_empty()
                && state.trail.is_none()
                && state.verify.is_none()
        });
        if !lent {
            return Err(nn::Error::Unsupported(
                "the drafter's state is not its live ring over the batch's rows".to_owned(),
            ));
        }
        self.attend(batch, cache, embedded, hidden)
    }

    /// A drafter that keeps a state reads its rows through its own norm for their logits.
    fn logits_rows(&self, hidden: &Tensor) -> nn::Result<Tensor> {
        if !self.stateful {
            return Ok(hidden.clone());
        }
        rms_norm(hidden, &self.gain, 1e-6, false, hidden.dtype())
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
        // The state's rows are the batch's, or a verification's sequences, which keep a trail
        // or are lent their verify slots.
        let lent = state.is_some_and(|state| {
            state.slots.dims() == [batch.rows()] || state.trail.is_some() || state.verify.is_some()
        });
        if self.stateful != lent {
            return Err(nn::Error::Unsupported(
                "the model's recurrent state is not the batch's rows".to_owned(),
            ));
        }

        // A stepwise field reads each state row's slots, `DRAFTS + 1` of them, and its count of
        // standing tokens; the token ids beside the embeddings are read as the batch's pair.
        let verified = state.and_then(|state| state.verify.as_ref().map(|verify| (state, verify)));
        let slotted = verified.is_some_and(|(state, verify)| {
            let rows = state.seq_lens.elem_count();

            verify.state_indices.dims() == [rows, DRAFTS as usize + 1]
                && verify.num_accepted.dims() == [rows]
                && state.trail.is_none()
        });
        if self.stepwise != slotted {
            return Err(nn::Error::Unsupported(
                "the model's stepwise state reads its rows' verify slots (verify)".to_owned(),
            ));
        }
        if self.reads != (batch.token_ids.is_some() && batch.token_history.is_some()) {
            return Err(nn::Error::Unsupported(
                "the model reads token ids exactly when it states it (token_ids, token_history)"
                    .to_owned(),
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

        zero_arrival_counters_(&group.workspace, batch.rows(), HEADS as u32)?;
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

    fn cache(&self) -> Vec<CacheField> {
        fields()
    }

    fn state_layers(&self) -> Vec<StateShape> {
        let mut runs: Vec<StateShape> = self.stateful.then(state_shape).into_iter().collect();
        if self.stepwise {
            runs.push(stepwise_shape());
        }
        runs
    }

    fn token_history(&self) -> Option<TokenHistory> {
        self.reads.then_some(TokenHistory {
            width: HISTORY,
            fill: FILL,
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

/// One attention group of one layer over `pages` meta pages, with the page table `table`; a
/// forward lends it the batch's slot mapping.
fn group(pages: usize, table: Tensor) -> KvCache {
    KvCache {
        block_tables: table,
        slot_mapping: meta(0, DType::U32),
        starts: None,
        page_size: PAGE,
        planes: KvCache::planes_of(&fields(), DType::BF16, pages, PAGE, |shape, dtype| {
            Ok(meta(shape, dtype))
        })
        .unwrap(),
        workspace: meta(1 << 16, DType::U8),
    }
}

/// Decode buckets of two and four rows, greedy and drawn, and prefill buckets of four rows
/// over eight and eighty tokens in the four prefill variants.
fn config() -> CausalLmConfig {
    CausalLmConfig {
        phases: vec![Phase::Prefill, Phase::Decode],
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
        CausalLmRunner::input_bytes(&config, &Tiny::new(false)).unwrap(),
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
        crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
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

/// The batch of `columns` in `shape`, of the phase of its first call, advancing one request per
/// call from request 100 on.
fn batch(shape: BatchShape, columns: Columns) -> Batch {
    let phase = columns
        .rows
        .calls
        .first()
        .map_or(Phase::Prefill, |call| call.kind);

    Batch {
        id: BatchId(7),
        group: GroupId(0),
        phase,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape,
        requests: (0..columns.calls() as u64)
            .map(|call| RequestId(100 + call))
            .collect(),
        columns,
        media: Vec::new(),
        new_slots: Vec::new(),
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

    let columns = Columns {
        rows: Rows {
            calls: (0..rows as u32)
                .map(|row| Row {
                    slot: slots[row as usize] as u32,
                    kind: Phase::Decode,
                    replay: 0..0,
                    rows: row..row + 1,
                })
                .collect(),
            positions: (0..rows as i32).map(|row| 10 + row).collect(),
            cache_positions: (0..rows as i32).map(|row| 10 + row).collect(),
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
        ..Columns::default()
    };

    batch(BatchShape::decode(rows as u32, variant), columns)
}

/// A prefill batch of two calls: slot 2's three tokens, which end its prompt, and slot 4's
/// two, which end none.
fn prefill(variant: Variant) -> Batch {
    let call = |slot: u32, rows: std::ops::Range<u32>| Row {
        slot,
        kind: Phase::Prefill,
        replay: 0..0,
        rows,
    };

    let mut table = commit(&[(2, mode::EXPLICIT, 0), (4, mode::CACHE, -1)]);
    table[column::CACHE_LENGTH * 2..][..2].copy_from_slice(&[3, 2]);
    table[column::LOGICAL * 2] = 3;
    table[column::SAMPLING * 2] = 1;

    let columns = Columns {
        rows: Rows {
            calls: vec![call(2, 0..1), call(4, 1..2)],
            positions: vec![0, 1, 2, 0, 1],
            cache_positions: vec![0, 1, 2, 0, 1],
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
        ..Columns::default()
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
) -> crate::worker::Result<Vec<CallOutput>> {
    calls_output(&batch.columns, bucket, vocabulary, candidates, outputs)
}

/// A performed call's output of `tokens`, with no score, and the slot's positions after it.
fn performed(tokens: Vec<u32>, step_index: u32, computed: u32, visible: u32) -> CallOutput {
    CallOutput {
        status: CallStatus::Ok,
        error_code: None,
        error_message: None,
        tokens,
        step_index,
        cache_computed_len: computed,
        cache_visible_len: visible,
        token_logprobs: Vec::new(),
        prompt_logprobs: Vec::new(),
        media_output: None,
        draft_token_ids: Vec::new(),
    }
}

/// The input block `columns::fill` writes for `batch` in `bucket`, over bytes that held
/// something else, with the layout its columns are read by: `mapping` is the cache slot each
/// token writes, and `scratch` the scratch pages each row of a paired forward takes.
fn filled(
    batch: &Batch,
    bucket: &Bucket,
    mapping: &[u32],
    scratch: Option<&[std::ops::Range<u32>]>,
) -> (Layout, Vec<u8>) {
    let layout = Layout::of(config().buckets.iter(), 1, 0, 0, HISTORY).unwrap();
    let mut block = vec![0xA5_u8; layout.bytes() as usize];

    columns::fill(
        &layout,
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        bucket,
        &[mapping.to_vec()],
        scratch,
        &mut block,
    )
    .unwrap();
    (layout, block)
}

// ---- The rank's device block table ---------------------------------------------------------------

/// The region the test tables' runs lie in: the cache's units.
const UNITS: u8 = 0;

/// The region of the test cache's state units, one checkpoint each.
const STATES: u8 = 1;

/// The unit the test tables give the attention block at `position` of request slot `slot`:
/// each slot's blocks apart from every other's, none of them the padding unit.
fn unit(slot: u32, position: u32) -> u32 {
    slot * TABLE_WIDTH as u32 + position + 1
}

/// Hands `tables` the `Attach` of the owner of slot `slot`, numbered by its slot, with its
/// `Attached`.
fn attach(tables: &mut BlockTables, slot: u32) {
    let request = Request::Attach {
        owner: OwnerId(u64::from(slot)),
        group: GroupId(0),
    };
    tables
        .map(
            slot,
            storage::runtime::RequestId(0),
            &request,
            &Completion::Attached { slot },
        )
        .unwrap();
}

/// Hands `tables` an `Allocate` of one block per run of `runs` at `at` of `class` for the owner
/// of slot `slot`, with its `Allocated`.
fn allocate(tables: &mut BlockTables, slot: u32, class: u8, at: u32, runs: &[Run]) {
    let request = Request::Allocate {
        owner: OwnerId(u64::from(slot)),
        class,
        at,
        count: runs.len() as u32,
        units: 0,
    };
    let completion = Completion::Allocated {
        runs: runs.to_vec(),
        evicted: vec![],
    };
    tables
        .map(slot, storage::runtime::RequestId(1), &request, &completion)
        .unwrap();
}

/// Applies to `tables` the `Commit` of the block at `at` of `class` of the owner of slot
/// `slot`: the entry becomes read-only.
fn commit_entry(tables: &mut BlockTables, slot: u32, class: u8, at: u32) {
    tables
        .apply(&Request::Commit {
            owner: OwnerId(u64::from(slot)),
            class,
            at,
            key: Key([7; 32]),
            tokens: PAGE as u16,
            after: vec![],
        })
        .unwrap();
}

/// The device block table of a rank on which each of `slots` is attached and holds a
/// writable attention block at every position of a row, in [`unit`].
fn held(slots: &[u32]) -> BlockTables {
    let mut tables = BlockTables::new(GroupId(0), &[UNITS, STATES], &[]);
    for &slot in slots {
        attach(&mut tables, slot);
        let runs: Vec<Run> = (0..TABLE_WIDTH as u32)
            .map(|position| (UNITS, unit(slot, position), 1))
            .collect();
        allocate(&mut tables, slot, 0, 0, &runs);
    }
    tables
}

/// The cache slot a token at cache position `position` of slot `slot` writes in [`held`]'s
/// tables.
fn written(slot: u32, position: u32) -> u32 {
    unit(slot, position / PAGE) * PAGE + position % PAGE
}

/// The cache slot each token of `batch` writes through `tables` in the test model's one
/// attention group, of a batch none of whose calls is refused.
fn mapping(batch: &Batch, tables: &BlockTables) -> Vec<u32> {
    let (mapping, refused) =
        columns::slot_mapping(&batch.columns.rows, tables, PAGE, &[false]).unwrap();
    assert_eq!(refused, []);
    mapping.concat()
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
        assert_eq!(
            count(&launches, "sampling_from_logits"),
            1 - greedy,
            "{bucket}"
        );

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
    // model's one host value is an FP32 per token of the largest bucket, at whatever offset its
    // alignment gives it.
    let inputs = CausalLmRunner::input_bytes(&config(), &Tiny::new(false)).unwrap();
    assert!(runner.graphs().peak() > inputs + (80 * HIDDEN * 2) as u64);
    assert_eq!(
        runner.graphs().input_peak(),
        4 * 80 + ops::allocator::ALIGNMENT - 1
    );
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
    let inputs = CausalLmRunner::input_bytes(&config, &Tiny::new(false)).unwrap();
    assert_eq!(inputs % 256, 0);
    assert!(inputs >= 4 * 320 + 21 * 256 - 4 * 256);

    // The two tables of a row per slot and vocabulary entry dominate the slot state.
    let slots = CausalLmRunner::slot_bytes(&config, &Tiny::new(false)).unwrap();
    let tables = 2 * u64::from(SLOTS) * VOCABULARY as u64 * 4;
    assert!(slots >= tables && slots < tables + 8 * 256);

    let mut wider = config;
    wider.slots = 2 * SLOTS;
    assert!(CausalLmRunner::slot_bytes(&wider, &Tiny::new(false)).unwrap() >= 2 * tables);
}

// ---- The bytes of the input tensors -------------------------------------------------------------

#[test]
fn a_decode_batch_is_padded_with_rows_that_address_nothing() {
    // Three rows in a bucket of four: a greedy row beside two drawn ones.
    // The scheduler numbers request slots from one: the last row runs on the last slot.
    let batch = decode(&[3, 1, SLOTS as i32], &[0.0, 0.7, 1.0]);
    let bucket = Bucket::decode(4, Variant::DecodeSampled);
    let tables = held(&[3, 1, SLOTS]);
    let (layout, block) = filled(&batch, &bucket, &mapping(&batch, &tables), None);

    let u32s = |column, elements| read::<u32>(&layout, &block, column, elements);
    let i32s = |column, elements| read::<i32>(&layout, &block, column, elements);

    // The batch's tokens write the slots their entries name at their cache positions, and
    // the padding token writes none.
    assert_eq!(i32s(Column::Positions, 4), [10, 11, 12, 0]);
    assert_eq!(
        u32s(Column::SlotMapping, 4),
        [written(3, 10), written(1, 11), written(SLOTS, 12), NO_SLOT]
    );
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
    let parameters = read::<f32>(&layout, &block, Column::Parameters, 4 * 8);
    let row = |row: usize| &parameters[row * 8..(row + 1) * 8];
    assert_eq!(row(0)[TOP_K as usize], 1.0);
    assert_eq!(row(1)[TEMPERATURE as usize], 0.7);
    assert_eq!(row(1)[TOP_K as usize], 0.0);
    assert_eq!(row(3), [0.0; 8]);

    // Each row's draw is its request's stream at the position it samples.
    let draws = read::<f32>(&layout, &block, Column::Draws, 4);
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
    assert_eq!(read::<u8>(&layout, &block, Column::Continuation, 4), [0; 4]);
}

#[test]
fn a_prefill_batch_is_padded_with_tokens_no_row_holds() {
    let mut batch = prefill(Variant::PrefillScored);
    batch.columns.targets = vec![6, 7, -1, 9, -1];
    let bucket = Bucket::prefill(4, 8, Variant::PrefillScored);
    let (layout, block) = filled(&batch, &bucket, &mapping(&batch, &held(&[2, 4])), None);

    let u32s = |column, elements| read::<u32>(&layout, &block, column, elements);

    assert_eq!(u32s(Column::InputIds, 8), [5, 6, 7, 8, 9, 0, 0, 0]);
    assert_eq!(
        u32s(Column::SlotMapping, 8),
        [
            written(2, 0),
            written(2, 1),
            written(2, 2),
            written(4, 0),
            written(4, 1),
            NO_SLOT,
            NO_SLOT,
            NO_SLOT
        ]
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

    let table = read::<i32>(&layout, &block, Column::Commit, COMMIT_FIELDS * 4);
    assert_eq!(table[column::SLOT * 4..][..4], [1, 3, -1, -1]);
    assert_eq!(table[column::TASK * 4..][..4], [0, -1, -1, -1]);
    assert_eq!(table[column::CACHE_LENGTH * 4..][..4], [3, 2, 0, 0]);
}

#[test]
fn a_row_s_adjustments_are_its_bias_flag_and_its_ban_mask() {
    let mut batch = decode(&[1, 2], &[0.0, 0.0]);
    batch.columns.biased = vec![true, false];
    batch.columns.banned = vec![Vec::new(), vec![3, 40]];
    let (layout, block) = filled(&batch, &Bucket::decode(2, Variant::DecodeGreedy), &[], None);

    // The second row is the first that bans, so its mask is mask zero.
    assert_eq!(
        read::<u32>(&layout, &block, Column::Adjustments, 4),
        [1, NO_SLOT, 0, 0]
    );
}

#[test]
fn the_idle_inputs_of_every_bucket_write_and_commit_nothing() {
    let runner = runner(false);
    let layout = Layout::of(config().buckets.iter(), 1, 0, 0, HISTORY).unwrap();

    for bucket in runner.buckets() {
        let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
        let mut block = vec![0xA5_u8; layout.bytes() as usize];
        runner.idle_inputs(&bucket, &mut block).unwrap();

        // No token writes a cache slot and no row has a key, a slot or a request's pages.
        assert_eq!(
            read::<u32>(&layout, &block, Column::SlotMapping, tokens),
            vec![NO_SLOT; tokens]
        );
        assert_eq!(
            read::<u32>(&layout, &block, Column::SeqLens, rows),
            vec![0; rows]
        );
        assert_eq!(
            read::<u32>(&layout, &block, Column::Slots, rows),
            vec![NO_SLOT; rows]
        );
        assert_eq!(
            read::<u32>(&layout, &block, Column::TableRows, rows),
            vec![SLOTS; rows]
        );

        // The step is a full batch of the bucket: every token lies in a row's span.
        let offsets = read::<u32>(&layout, &block, Column::QueryOffsets, rows + 1);
        assert_eq!((offsets[0], offsets[rows]), (0, bucket.tokens));
        assert_eq!(
            read::<u32>(&layout, &block, Column::LiveTokens, 1),
            [bucket.tokens]
        );

        // Every row of the commit table is inert.
        let capacity = layout.rows();
        let table = read::<i32>(&layout, &block, Column::Commit, COMMIT_FIELDS * capacity);
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

// ---- The cache slots a batch writes ----------------------------------------------------------

/// A prefill of `cache_positions` on slot 2: one call of one row whose tokens are an image's
/// rows, which share one rotary position.
fn chunk(cache_positions: &[i32]) -> Batch {
    let tokens = cache_positions.len();
    let mut batch = prefill(Variant::PrefillWrite);

    batch.columns.rows = Rows {
        calls: vec![Row {
            slot: 2,
            kind: Phase::Prefill,
            replay: 0..0,
            rows: 0..1,
        }],
        positions: vec![3; tokens],
        cache_positions: cache_positions.to_vec(),
        seq_lens: vec![cache_positions.iter().max().map_or(0, |last| last + 1)],
        query_offsets: vec![0, tokens as i32],
        logits_rows: vec![-1],
    };
    batch.columns.input_ids = vec![5; tokens];
    batch.columns.slots = vec![2];
    batch.columns.parameters = request(0.0).to_vec();
    batch.columns.draws = vec![DrawAddress::default()];
    batch.columns.commit = commit(&[(2, mode::CACHE, -1)]);
    batch.requests.truncate(1);
    batch.shape = BatchShape::prefill(1, tokens as u32, Variant::PrefillWrite);
    batch
}

/// The cache slots the runner wrote into its input block for the step it prepared last.
fn prepared_mapping(runner: &CausalLmRunner, tokens: usize) -> Vec<u32> {
    read::<u32>(&runner.layout, &runner.staged, Column::SlotMapping, tokens)
}

#[test]
fn a_token_writes_the_cache_slot_its_entry_names_at_its_cache_position() {
    let mut runner = captured();

    // Slot 2's blocks at positions 0 and 1 are units 9 and 4: a token's slot is the unit of
    // the block its cache position lies in, whatever its rotary position, and a token at a
    // negative position writes nothing.
    let mut tables = held(&[]);
    attach(&mut tables, 2);
    allocate(&mut tables, 2, 0, 0, &[(UNITS, 9, 1), (UNITS, 4, 1)]);

    let batch = chunk(&[14, 15, 16, -1, 17]);
    let bucket = runner.prepare_inputs(&batch, &tables).unwrap();
    assert_eq!(bucket, Bucket::prefill(4, 8, Variant::PrefillWrite));
    assert_eq!(
        prepared_mapping(&runner, 8),
        [
            9 * PAGE + 14,
            9 * PAGE + 15,
            4 * PAGE,
            NO_SLOT,
            4 * PAGE + 1,
            NO_SLOT,
            NO_SLOT,
            NO_SLOT
        ]
    );
}

#[test]
fn a_token_whose_entry_is_read_only_or_missing_writes_nothing_and_fails_its_call() {
    let mut runner = captured();
    let batch = prefill(Variant::PrefillGreedy);

    // Slot 4's first block was committed: it is read-only, and the call that writes into it
    // is refused while slot 2's call writes its tokens.
    let mut tables = held(&[2, 4]);
    commit_entry(&mut tables, 4, 0, 0);
    let (mapping, refused) =
        columns::slot_mapping(&batch.columns.rows, &tables, PAGE, &[false]).unwrap();
    assert_eq!(
        mapping[0],
        [
            written(2, 0),
            written(2, 1),
            written(2, 2),
            NO_SLOT,
            NO_SLOT
        ]
    );
    assert_eq!(refused.len(), 1);
    let (call, reason) = &refused[0];
    assert_eq!(*call, 1);
    assert!(
        reason.contains("slot 4") && reason.contains("read-only"),
        "{reason}"
    );

    // A preparation reports no call, so it refuses the batch, naming the entry.
    let message = runner
        .prepare_inputs(&batch, &tables)
        .unwrap_err()
        .to_string();
    assert!(message.contains("read-only"), "{message}");

    // A slot whose table holds no entry at a position its call writes is refused alike.
    let missing = held(&[2]);
    let (mapping, refused) =
        columns::slot_mapping(&batch.columns.rows, &missing, PAGE, &[false]).unwrap();
    assert_eq!(mapping[0][3..], [NO_SLOT, NO_SLOT]);
    assert!(refused[0].1.contains("holds no entry"), "{}", refused[0].1);
    let message = runner
        .prepare_inputs(&batch, &missing)
        .unwrap_err()
        .to_string();
    assert!(message.contains("holds no entry"), "{message}");

    // A served step fails the refused call alone, as an invariant violation, which the
    // control plane's scheduling of a row outside its writable entries is.
    let outputs = vec![
        tensor(&[11_u32, 22, 0, 0], 4),
        tensor(&[-0.5_f32, -2.0, 0.0, 0.0], 4),
        tensor(&[1_i32, 1, 0, 0], 4),
        tensor(&[3_i32, 2, 0, 0], 4),
        tensor(&[1_i32, 0, 0, 0], 4),
    ];
    let bucket = Bucket::prefill(4, 8, Variant::PrefillGreedy);
    let mut calls = completion(&batch, &bucket, 64, 0, &outputs).unwrap();
    refuse(&mut calls, &refused);

    assert_eq!(calls[0].status, CallStatus::Ok);
    assert_eq!(calls[0].tokens, [11]);
    assert_eq!(calls[1].status, CallStatus::Error);
    assert_eq!(calls[1].error_code, Some(ErrorCode::InvariantViolation));
    assert_eq!(calls[1].cache_computed_len, 0);
    let output = BatchOutput {
        calls,
        ..BatchOutput::default()
    };
    output.check(&batch).unwrap();
}

// ---- The choice of a bucket and the refusals ----------------------------------------------------

#[test]
fn a_step_runs_the_smallest_captured_bucket_that_holds_it() {
    let mut runner = captured();
    let tables = held(&[1, 2, 3]);

    let small = runner
        .prepare_inputs(&decode(&[1, 2], &[0.0, 0.0]), &tables)
        .unwrap();
    assert_eq!(small, Bucket::decode(2, Variant::DecodeGreedy));

    let wide = runner
        .prepare_inputs(&decode(&[1, 2, 3], &[0.0, 0.5, 0.0]), &tables)
        .unwrap();
    assert_eq!(wide, Bucket::decode(4, Variant::DecodeSampled));

    let prompt = runner
        .prepare_inputs(&prefill(Variant::PrefillGreedy), &held(&[2, 4]))
        .unwrap();
    assert_eq!(prompt, Bucket::prefill(4, 8, Variant::PrefillGreedy));
}

#[test]
fn the_last_request_slot_serves_and_gathers_its_own_rows() {
    let mut runner = captured();
    let tables = held(&[SLOTS]);

    // Every per-slot buffer holds rows `0..SLOTS` for the slots `1..=SLOTS`: the last slot's
    // batch is prepared, and starts its request on the last row.
    let mut batch = decode(&[SLOTS as i32], &[0.0]);
    batch.new_slots = vec![NewSlot {
        slot: SLOTS,
        history: vec![3, 3],
        bias: vec![(3, 1.5)],
        allowed: None,
        context: vec![0, 0],
    }];
    runner.prepare_inputs(&batch, &tables).unwrap();

    // The slot's writes are made, and the step reaches the replay graphs with no device
    // refuse.
    let refused = runner.execute(&batch, &tables).unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");

    // Its rows gather row `SLOTS - 1` of the page tables and of the slot state, and its call
    // commits into that row; the padding row is the one after it.
    let bucket = Bucket::decode(2, Variant::DecodeGreedy);
    let (layout, block) = filled(&batch, &bucket, &mapping(&batch, &tables), None);
    let u32s = |column| read::<u32>(&layout, &block, column, 2);

    assert_eq!(u32s(Column::Slots), [SLOTS - 1, NO_SLOT]);
    assert_eq!(u32s(Column::TableRows), [SLOTS - 1, SLOTS]);
    assert_eq!(u32s(Column::SlotRows), [SLOTS - 1, SLOTS]);
    assert_eq!(u32s(Column::CallSlots), [SLOTS - 1, SLOTS]);
    assert_eq!(u32s(Column::SlotMapping), [written(SLOTS, 10), NO_SLOT]);

    let table = read::<i32>(&layout, &block, Column::Commit, COMMIT_FIELDS * 4);
    assert_eq!(table[column::SLOT * 4..][..2], [SLOTS as i32 - 1, -1]);
}

#[test]
fn a_batch_no_bucket_holds_is_refused_by_its_shape_and_the_bound() {
    let mut runner = captured();

    let refused = runner
        .prepare_inputs(
            &decode(&[5, 1, 2, 3, 4], &[0.0; 5]),
            &held(&[1, 2, 3, 4, 5]),
        )
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
        .prepare_inputs(&decode(&[1, 2], &[0.0, 0.0]), &held(&[1, 2]))
        .unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");
}

#[test]
fn a_drawn_row_in_a_greedy_bucket_is_refused_by_the_row() {
    let mut runner = captured();

    // The batch states a greedy shape and its second row states a temperature.
    let mut batch = decode(&[1, 2], &[0.0, 0.0]);
    batch.columns.parameters[PARAMETER_COLUMNS as usize + TEMPERATURE as usize] = 0.8;

    let message = runner
        .prepare_inputs(&batch, &held(&[1, 2]))
        .unwrap_err()
        .to_string();
    assert!(message.contains("row 1"), "{message}");
    assert!(message.contains("temperature 0.8"), "{message}");
}

#[test]
fn a_batch_the_runner_s_graphs_were_not_captured_for_is_refused() {
    let mut runner = captured();
    let tables = held(&[1, 2, SLOTS]);

    // Another page size than the cache's.
    let mut paged = decode(&[1, 2], &[0.0, 0.0]);
    paged.columns.page_size = 2 * PAGE;
    assert!(matches!(
        runner.prepare_inputs(&paged, &tables),
        Err(Error::Invalid(_))
    ));

    // A row on a slot the runner holds no state for, whose table holds nothing either.
    let slotted = decode(&[1, SLOTS as i32 + 1], &[0.0, 0.0]);
    let message = runner
        .prepare_inputs(&slotted, &tables)
        .unwrap_err()
        .to_string();
    assert!(message.contains("slot 7"), "{message}");

    // Slot zero is no request slot: the scheduler numbers them from one.
    let zeroth = decode(&[0, 1], &[0.0, 0.0]);
    let message = runner
        .prepare_inputs(&zeroth, &tables)
        .unwrap_err()
        .to_string();
    assert!(message.contains("slot 0"), "{message}");

    // A verification, which a runner whose config states no drafts serves none of.
    let mut verified = decode(&[1, 2], &[0.0, 0.0]);
    verified.columns.rows.calls[0].kind = Phase::Verify;
    assert!(matches!(
        runner.prepare_inputs(&verified, &tables),
        Err(Error::Unsupported(_))
    ));

    // Columns that are not the batch's rows.
    let mut short = decode(&[1, 2], &[0.0, 0.0]);
    short.columns.slots.pop();
    assert!(matches!(
        runner.prepare_inputs(&short, &tables),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn a_runner_is_built_over_its_model_s_cache_and_state() {
    let build = |stateful: bool, table: Tensor, state: bool| {
        let config = config();
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            CausalLmRunner::input_bytes(&config, &Tiny::new(false)).unwrap(),
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
        crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
    }
}

// ---- Page work ----------------------------------------------------------------------------------

/// `batch` with one recurrent-state restore of slot 1's checkpoint at position 0: page work
/// of the cache's.
fn with_page_work(mut batch: Batch) -> Batch {
    batch.columns.state_copies = vec![StateCopy {
        slot: 1,
        position: 0,
        direction: StateDirection::Restore,
    }];
    batch
}

/// Which part of a batch's page work `work` is, as a test records it.
fn part(work: &PageWork<'_>) -> &'static str {
    match work {
        PageWork::Tables(_) => "tables",
        PageWork::Copies(_) => "copies",
        PageWork::BeforeReplay(..) => "before replay",
        PageWork::AfterReplay => "after replay",
        PageWork::Ended => "ended",
        PageWork::BeforeShadow(..) => "before shadow",
        PageWork::Holds(..) => "holds",
    }
}

#[test]
fn a_batch_s_page_work_is_issued_through_the_hook_around_the_replay() {
    let mut runner = captured();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let tables = held(&[1, 2]);

    let seen = Arc::clone(&calls);
    runner.set_page_work(Box::new(move |work| {
        seen.lock().unwrap().push(part(&work));
        Ok(())
    }));

    // A refused batch issues nothing; the copies taken for it end all the same.
    let mut refused = decode(&[1, 2], &[0.0, 0.0]);
    refused.columns.page_size = 2 * PAGE;
    assert!(runner.execute(&refused, &tables).is_err());
    assert_eq!(*calls.lock().unwrap(), ["ended"]);
    calls.lock().unwrap().clear();

    // The work that precedes the forward is issued once the inputs are prepared and before
    // the replay, which graphs that hold no device refuse; nothing follows a replay that was
    // not issued, but the end.
    let batch = with_page_work(decode(&[1, 2], &[0.0, 0.0]));
    let refused = runner.execute(&batch, &tables).unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");
    assert_eq!(*calls.lock().unwrap(), ["before replay", "ended"]);
    calls.lock().unwrap().clear();

    // The table's changes and the copies reach the hook as the worker hands them.
    runner.write_tables(&[]).unwrap();
    runner.copy(&[]).unwrap();
    assert_eq!(*calls.lock().unwrap(), ["tables", "copies"]);
}

#[test]
fn a_hook_s_refusal_fails_the_batch_before_its_replay() {
    let mut runner = captured();
    runner.set_page_work(Box::new(|work| match work {
        PageWork::BeforeReplay(..) => Err(Error::Resource("the cache holds no page for the batch")),
        _ => Ok(()),
    }));

    let batch = with_page_work(decode(&[1, 2], &[0.0, 0.0]));
    assert!(matches!(
        runner.execute(&batch, &held(&[1, 2])),
        Err(Error::Resource(_))
    ));
}

#[test]
fn a_runner_with_no_hook_refuses_the_page_work_it_is_handed() {
    let mut runner = captured();
    let mut tables = held(&[1, 2]);

    let batch = with_page_work(decode(&[1, 2], &[0.0, 0.0]));
    let message = runner.execute(&batch, &tables).unwrap_err().to_string();
    assert!(message.contains("state copies"), "{message}");

    // Changes of the device block table and copies with runs reach nothing that writes them.
    assert!(runner.write_tables(&tables.take_changes()).is_err());
    let copy = BlockCopy {
        request: storage::runtime::RequestId(3),
        runs: vec![((UNITS, 1, 1), (UNITS, 2, 1))],
        encodings: None,
    };
    assert!(matches!(runner.copy(&[copy]), Err(Error::Unsupported(_))));
}

#[test]
fn a_slot_the_runner_does_not_hold_is_not_started() {
    let mut runner = captured();
    let tables = held(&[1, 2]);
    let new = |slot: u32| NewSlot {
        slot,
        context: vec![0, 0],
        ..NewSlot::default()
    };

    let mut batch = decode(&[1, 2], &[0.0, 0.0]);
    batch.new_slots = vec![new(SLOTS + 1)];
    let message = runner.execute(&batch, &tables).unwrap_err().to_string();
    assert!(message.contains("request slot 7"), "{message}");

    // A new slot of a decode batch starts from its own row of the batch.
    let mut batch = decode(&[1, 2], &[0.0, 0.0]);
    batch.new_slots = vec![new(4)];
    let message = runner.execute(&batch, &tables).unwrap_err().to_string();
    assert!(message.contains("no row in its decode batch"), "{message}");
}

// ---- Paired forwards ----------------------------------------------------------------------------

#[test]
fn the_scratch_page_set_holds_the_largest_bucket_s_tokens_and_a_page_a_row() {
    // The buckets of a deployment of 128 running requests with prefill chunks of 16384 tokens,
    // over pages of 64 tokens: 256 pages of the chunk's tokens and 128 more, one per row.
    let buckets = Buckets::causal_lm(&[1, 2, 4, 8, 16, 32, 64, 128], &[512, 8192, 16384], true);
    assert_eq!(CausalLmRunner::scratch_pages(&buckets, 64), 256 + 128);

    // A partial page of tokens takes a whole one: the tests' eighty tokens of pages of sixteen
    // are five pages, with the four rows' four.
    assert_eq!(
        CausalLmRunner::scratch_pages(&config().buckets, PAGE),
        SCRATCH_PAGES
    );
}

#[test]
fn the_scratch_set_hands_its_pages_to_the_rows_in_order() {
    // A decode at position 20, a chunk of three tokens from position 30, a row of nothing, and
    // a row of four keys whose first token writes position 16 and whose second writes nothing.
    let rows = Rows {
        calls: Vec::new(),
        positions: vec![20, 30, 31, 32, 16, 17],
        cache_positions: vec![20, 30, 31, 32, 16, -1],
        seq_lens: vec![21, 33, 0, 4],
        query_offsets: vec![0, 1, 4, 4, 6],
        logits_rows: vec![0, 3, -1, 5],
    };

    // Each row takes the pages of what it reads or writes, the next row from where it ended.
    let spans = columns::scratch_spans(&rows, PAGE).unwrap();
    assert_eq!(spans, [0..2, 2..5, 5..5, 5..7]);

    // A token writes its position's page among its row's: the decode page 1 of its two, the
    // chunk pages 1, 1 and 2 of its three (set pages 3, 3 and 4), the last row page 1 of its
    // two (set page 6), and the unwritten token nothing.
    let mapping =
        columns::scratch_slot_mapping(&rows.cache_positions, &rows.query_offsets, &spans, PAGE)
            .unwrap();
    assert_eq!(
        mapping,
        [
            PAGE + 4,
            3 * PAGE + 14,
            3 * PAGE + 15,
            4 * PAGE,
            6 * PAGE,
            NO_SLOT
        ]
    );

    // A position past its row's pages, a page size of zero, and lengths of other rows than
    // the offsets', are refused; so is a negative length.
    assert!(matches!(
        columns::scratch_slot_mapping(&[32], &[0, 1], std::slice::from_ref(&(0..2)), PAGE),
        Err(Error::Invariant(_))
    ));
    assert!(columns::scratch_spans(&rows, 0).is_err());
    let mut unmatched = rows.clone();
    unmatched.seq_lens.pop();
    assert!(matches!(
        columns::scratch_spans(&unmatched, PAGE),
        Err(Error::Invariant(_))
    ));
    let mut negative = rows;
    negative.seq_lens[2] = -1;
    assert!(matches!(
        columns::scratch_spans(&negative, PAGE),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn a_paired_forward_writes_its_rows_scratch_pages_and_commits_nothing() {
    let batch = prefill(Variant::PrefillOutput);
    let bucket = Bucket::prefill(4, 8, Variant::PrefillOutput);

    // Slot 2's three tokens take page 0 of the set and slot 4's two take page 1.
    let rows = &batch.columns.rows;
    let spans = columns::scratch_spans(rows, PAGE).unwrap();
    assert_eq!(spans, [0..1, 1..2]);
    let scratch =
        columns::scratch_column(&rows.cache_positions, &rows.query_offsets, 8, &spans, PAGE)
            .unwrap();

    // The tokens at cache positions 0, 1 and 2 of the first row write page 0, those at 0 and
    // 1 of the second page 1, and the tokens the batch does not hold write nothing.
    assert_eq!(
        scratch,
        [0, 1, 2, PAGE, PAGE + 1, NO_SLOT, NO_SLOT, NO_SLOT]
    );

    let (layout, block) = filled(&batch, &bucket, &scratch, Some(&spans));
    assert_eq!(
        read::<u32>(&layout, &block, Column::SlotMapping, 8),
        scratch
    );

    // Each row gathers the row of the set's table that begins at its first page, a row past
    // the batch's row zero; each reads its slot's state.
    assert_eq!(
        read::<u32>(&layout, &block, Column::TableRows, 4),
        [0, 1, 0, 0]
    );
    assert_eq!(
        read::<u32>(&layout, &block, Column::SlotRows, 4),
        [1, 3, SLOTS, SLOTS]
    );

    let table = read::<i32>(&layout, &block, Column::Commit, COMMIT_FIELDS * 4);
    assert_eq!(table[column::SLOT * 4..][..4], [-1; 4]);
    assert_eq!(table[column::TASK * 4..][..4], [-1; 4]);
}

/// The scratch set's page table a runner of [`config`] over [`group`]s is lent: a row for every
/// page of the set, as wide as the pooled table, which is narrower than the set.
fn scratch_table() -> Tensor {
    meta((SCRATCH_PAGES as usize, TABLE_WIDTH), DType::U32)
}

#[test]
fn a_paired_forward_is_prepared_over_a_scratch_set_the_cache_lends() {
    let mut runner = captured();
    let batch = decode(&[1, 2], &[0.0, 0.0]);
    let tables = held(&[1, 2]);

    // No scratch set was lent.
    assert!(matches!(
        runner.prepare_shadow_inputs(&batch, &tables),
        Err(Error::Unsupported(_))
    ));

    // A set of other pages than the config's, then the config's.
    let other = meta((SCRATCH_PAGES as usize - 1, TABLE_WIDTH), DType::U32);
    assert!(runner.set_scratch(vec![group(8, other)]).is_err());
    runner
        .set_scratch(vec![group(SCRATCH_PAGES as usize, scratch_table())])
        .unwrap();

    let bucket = runner.prepare_shadow_inputs(&batch, &tables).unwrap();
    assert_eq!(bucket, Bucket::decode(2, Variant::DecodeGreedy));

    // The forward over the scratch set makes the launches of the serving forward.
    let serving = operators(&runner, &bucket);
    let context = ops::Context::meta(None);
    context.record_launches();
    {
        let _guard = context.enter();
        runner.forward(&bucket).unwrap();
    }
    let paired: Vec<&str> = context
        .take_launches()
        .into_iter()
        .map(|(operator, ..)| operator)
        .collect();
    assert_eq!(paired, serving);
}

/// One entry of `kernel` for every key of `keys`.
fn entries<'a>(
    keys: impl IntoIterator<Item = &'a (&'static str, String)>,
    kernel: &Arc<ops::Kernel>,
) -> Vec<ops::KernelEntry> {
    keys.into_iter()
        .map(|(operator, key)| ops::KernelEntry::keyed(operator, key.clone(), Arc::clone(kernel)))
        .collect()
}

#[test]
fn a_shadow_batch_whose_rows_do_not_fit_the_scratch_set_is_reported_uncompared() {
    use testkit::LONG;

    // The rank's cache with the runner's scratch page set, and the generation the runner
    // serves, which holds a kernel for every key it launches.
    let mut cache = KvCacheMemory::meta(cache_config(false)).unwrap();
    cache.open_scratch(SCRATCH_PAGES, None).unwrap();
    let cache = Arc::new(Mutex::new(cache));
    let keys = over(&lock(&cache), None).graphs().keys();
    let serving = Arc::new(Generation::new(1, entries(&keys, &unlaunched_kernel())));

    // The runner over the cache, each part of a batch's page work recorded before the cache
    // issues it.
    let mut runner = CausalLmRunner::over(
        Arc::new(Tiny::new(false)),
        config(),
        Graphs::meta(
            Arc::clone(&serving),
            CausalLmRunner::input_bytes(&config(), &Tiny::new(false)).unwrap(),
        ),
        None,
        &lock(&cache),
        None,
    )
    .unwrap();
    let issued = Arc::new(Mutex::new(Vec::new()));
    let mut cache_work = wiring::page_work(Arc::clone(&cache), None, None);
    let seen = Arc::clone(&issued);
    runner.set_page_work(Box::new(move |work| {
        seen.lock().unwrap().push(part(&work));
        cache_work(work)
    }));

    let group = GroupId(0);
    let sequences = Sequences::new(group, 1, Vec::new());
    let worker = Worker::spawn(
        None,
        vec![Box::new(runner)],
        Arc::clone(&serving),
        held(&[1, 2, 3]),
        Arc::clone(&sequences),
        0,
    )
    .unwrap();
    let ended = |submission: crate::worker::Result<crate::worker::completion::Submission>| {
        submission.unwrap().wait_timeout(LONG)
    };
    assert!(matches!(ended(worker.warm_up()), Some(Outcome::Success(_))));

    // Generation 2 replaces the attention kernel every bucket launches.
    let attention = keys
        .iter()
        .filter(|(operator, _)| *operator == "paged_attention");
    let next = Arc::new(serving.replace(2, entries(attention, &unlaunched_kernel())));
    assert!(matches!(
        ended(worker.prepare(next, None)),
        Some(Outcome::Success(_))
    ));

    // Three decode rows that each read sixty-four keys take four pages each: twelve of the
    // set's nine. The paired forward is not run, the shadow batch ends, and the update states
    // that it has no comparison of the attention kernel on that batch.
    let mut long = decode(&[1, 2, 3], &[0.0, 0.0, 0.0]);
    long.kernels = 2;
    long.columns.rows.positions = vec![63; 3];
    long.columns.rows.cache_positions = vec![63; 3];
    long.columns.rows.seq_lens = vec![MAX_KV_LEN as i32; 3];
    assert!(matches!(
        ended(worker.shadow(1, long.clone())),
        Some(Outcome::Success(_))
    ));
    let uncompared = worker.uncompared();
    assert_eq!(uncompared.len(), 1, "{uncompared:?}");
    assert!(
        uncompared[0].contains("`paged_attention` in bucket")
            && uncompared[0].contains("take 12 scratch pages")
            && uncompared[0].contains("holds 9"),
        "{uncompared:?}"
    );
    assert!(!issued.lock().unwrap().contains(&"before shadow"));

    // The batch itself is served as every batch is: its page work is issued around the
    // replay, which graphs with no device refuse.
    long.kernels = 1;
    issued.lock().unwrap().clear();
    let served = ended(worker.submit(2, long.clone()));
    let Some(Outcome::Failed(refused)) = served else {
        panic!("graphs with no device replay nothing: {served:?}");
    };
    assert!(refused.to_string().contains("no device"), "{refused}");
    let work = std::mem::take(&mut *issued.lock().unwrap());
    assert_eq!(work[..2], ["before replay", "ended"], "{work:?}");

    // Rows that fit are paired: their history reaches the scratch pages, and the paired
    // forward runs, which graphs with no device refuse.
    let mut short = decode(&[1, 2], &[0.0, 0.0]);
    short.kernels = 2;
    let shadowed = ended(worker.shadow(3, short));
    let Some(Outcome::Failed(refused)) = shadowed else {
        panic!("graphs with no device run no paired forward: {shadowed:?}");
    };
    assert!(refused.to_string().contains("no device"), "{refused}");
    assert!(issued.lock().unwrap().contains(&"before shadow"));
    worker.close().unwrap();
}

#[test]
fn a_model_that_keeps_a_recurrent_state_has_no_paired_forward() {
    let mut runner = runner(true);

    let refused = runner
        .prepare_shadow_inputs(&decode(&[1, 2], &[0.0, 0.0]), &held(&[1, 2]))
        .unwrap_err();
    assert!(refused.to_string().contains("recurrent state"), "{refused}");
}

// ---- The completion -----------------------------------------------------------------------------

/// The CPU tensor of `values` in `shape`.
fn tensor<T: WithDType>(values: &[T], shape: impl Into<tensor::Shape>) -> Tensor {
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

    // The outputs answer the batch's requests in its order.
    let mut first = performed(vec![11], 5, 41, 5);
    // The token itself, then its row's candidates, the largest first.
    first.token_logprobs = vec![vec![
        entry(11, -0.5, 2),
        entry(7, -0.25, 1),
        entry(11, -0.5, 2),
    ]];
    assert_eq!(output[0], first);

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
    assert_eq!(output[1], performed(Vec::new(), 0, 2, 0));
}

#[test]
fn a_call_of_several_rows_reports_one_state_and_its_rows_tokens_in_order() {
    // One call owns both rows, on one slot.
    let mut batch = decode(&[2, 2], &[0.0, 0.0]);
    batch.columns.rows.calls = vec![Row {
        slot: 2,
        kind: Phase::Decode,
        replay: 0..0,
        rows: 0..2,
    }];
    batch.requests = vec![RequestId(9)];
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
    let answered = BatchOutput {
        calls: output,
        ..BatchOutput::default()
    };
    answered.check(&batch).unwrap();

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
    orphan.requests.pop();
    assert!(matches!(
        completion(&orphan, &bucket, 64, 2, &greedy_outputs([11, 22])),
        Err(Error::Invariant(_))
    ));
}

#[test]
fn the_runner_serves_the_phases_of_its_config() {
    let runner = runner(false);

    assert_eq!(runner.phases(), [Phase::Prefill, Phase::Decode]);
}

// ---- Expert unions ------------------------------------------------------------------------------

/// The most rows a member of the tested union sends in a step: the largest bucket's tokens.
const UNION_ROWS: u32 = 80;

/// The rows per rank of the tested union's middle capture class, the deployment's default, which
/// the union's capacity is too small to capture.
const BESIDE: u32 = proto::config::DEFAULT_BESIDE_ROWS.get();

/// The host groups of a union of `size` members of this process, in member order.
fn host_groups(size: usize) -> Vec<Arc<HostGroup>> {
    HostGroup::local(size)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect()
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
        multiprocessors: 148,
        attention_ranks: 0,
    };

    Member::new(Arc::clone(group), shape, Mode::Dispatch, join, BESIDE).unwrap()
}

/// A join hook that hands each join batch to the worker reading `inbox`.
fn deliver(inbox: crossbeam_channel::Sender<Batch>) -> JoinHook {
    Arc::new(move |batch| {
        inbox
            .send(batch)
            .map_err(|_| Error::State("the member's worker is gone"))
    })
}

/// A runner of a model that routes experts, as `member` of a union, with every bucket
/// captured.
fn union_runner(member: Member) -> CausalLmRunner {
    let config = config();
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&config, &Tiny::new(false)).unwrap(),
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
        crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
    }
    runner
}

#[test]
fn a_union_member_captures_each_bucket_at_the_capacities_its_steps_meet() {
    let stated = config().buckets;
    let classes = experts::classes(&stated, UNION_ROWS, BESIDE);

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
    assert_eq!(classes.iter().len(), 2 * 4 + 2 * 4 + 4 + 2);
}

#[test]
fn the_members_of_a_step_agree_on_the_most_tokens_any_of_them_sends() {
    let groups = host_groups(2);
    let (to_first, first_inbox) = crossbeam_channel::unbounded();
    let (to_second, second_inbox) = crossbeam_channel::unbounded();
    let mut first = member(&groups[0], 0, deliver(to_second));
    let mut second = member(&groups[1], 1, deliver(to_first));
    let (group, phase) = (GroupId(0), Phase::Decode);
    let step = Batch {
        id: BatchId(0),
        group,
        phase,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: crate::batch::Shape::decode(1, Variant::DecodeGreedy),
        requests: Vec::new(),
        columns: Columns::default(),
        media: Vec::new(),
        new_slots: Vec::new(),
    };

    // The first member has a batch of three tokens and the second none: its worker takes
    // the join batch, and both leave the step at the first member's tokens.
    std::thread::scope(|scope| {
        let idle = scope.spawn(|| {
            let join = second_inbox.recv().unwrap();
            assert_eq!(join.shape.variant, Variant::ExpertJoin);
            assert_eq!((join.group, join.phase), (group, phase));
            assert!(join.requests.is_empty());

            second.join(&join).unwrap()
        });

        assert_eq!(first.enter(3, &step).unwrap(), 3);
        assert_eq!(idle.join().unwrap(), Some(3));
    });

    // Both members have a batch: the step holds the larger, and the join each asked of the
    // other finds its step closed.
    std::thread::scope(|scope| {
        let other = scope.spawn(|| second.enter(9, &step).unwrap());

        assert_eq!(first.enter(5, &step).unwrap(), 9);
        assert_eq!(other.join().unwrap(), 9);
    });
    let late = first_inbox.recv().unwrap();
    assert_eq!(first.join(&late).unwrap(), None);
    let late = second_inbox.recv().unwrap();
    assert_eq!(second.join(&late).unwrap(), None);

    // A member enters with tokens of its own, no more than the transport holds, and joins no
    // step past the union's next.
    assert!(matches!(first.enter(0, &step), Err(Error::Invalid(_))));
    assert!(matches!(
        first.enter(UNION_ROWS + 1, &step),
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
    let context = ops::Context::meta(None);
    {
        let _guard = context.enter();
        assert!(first.forward(&join).unwrap().is_empty());
    }
    assert!(first.forward(&Bucket::expert_join()).is_err());

    // The first member's batch blocks in the agreement until the second member's worker
    // takes the join batch. Both then reach the replay of a captured class, which graphs
    // that hold no device refuse.
    let tables = held(&[1, 2, 3]);
    std::thread::scope(|scope| {
        let idle = scope.spawn(|| {
            let join = second_inbox.recv().unwrap();
            second.execute(&join, &held(&[]))
        });

        let served = first.execute(&decode(&[1, 2, 3], &[0.0; 3]), &tables);
        assert!(matches!(served, Err(Error::State(_))), "{served:?}");

        let joined = idle.join().unwrap();
        assert!(matches!(joined, Err(Error::State(_))), "{joined:?}");
    });

    // A member computes no paired forward: its exchange launches wait for the others'.
    assert!(matches!(
        first.prepare_shadow_inputs(&decode(&[1, 2], &[0.0, 0.0]), &tables),
        Err(Error::Unsupported(_))
    ));
}

/// A union whose two members two processes hold, each process here a thread whose members meet
/// the other's over TCP: one formation at the rendezvous forms the union's host group and its
/// join link (`crate::process::ranks::union`). A member that enters a step with tokens drives the
/// other process's idle member into it over the link: that member's worker takes the join batch,
/// and both leave the agreement at the asking member's tokens, step after step. Once the other
/// process is gone, the next step fails instead of waiting for it.
#[test]
fn a_member_of_another_process_is_driven_into_the_step_over_the_join_link() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let rendezvous = |listener| Rendezvous {
        listener,
        endpoint: endpoint.clone(),
        epoch: 7,
        timeout: testkit::LONG,
    };
    // Each process's member of the union's host group and of its join link.
    let formed = |formed: Vec<Vec<HostGroup>>| {
        let [mut union, link] = <[Vec<HostGroup>; 2]>::try_from(formed).ok().unwrap();
        (Arc::new(union.remove(0)), link)
    };
    let ((first_group, first_link), (second_group, second_link)) = std::thread::scope(|scope| {
        let second = scope.spawn(|| HostGroup::world(rendezvous(None), &[1], 2, 2).unwrap());
        let first = HostGroup::world(rendezvous(Some(listener)), &[0], 2, 2).unwrap();
        (formed(first), formed(second.join().unwrap()))
    });
    let first_joins = Arc::new(Joins::linked(2, first_link).unwrap());
    let second_joins = Arc::new(Joins::linked(2, second_link).unwrap());

    // The second process's member waits in its worker, which only its join batch reaches.
    let runner = union_runner(member(&second_group, 1, second_joins.hook(1)));
    let group = GroupId(1);
    let worker = Arc::new(
        Worker::spawn(
            None,
            vec![Box::new(runner)],
            Arc::new(Generation::new(0, [])),
            BlockTables::new(group, &[], &[]),
            Sequences::new(group, 1, Vec::new()),
            0,
        )
        .unwrap(),
    );
    second_joins
        .serve(vec![None, Some(Arc::downgrade(&worker))])
        .unwrap();

    // The first process's member is driven here, as its worker drives it.
    let mut first = member(&first_group, 0, first_joins.hook(0));
    first_joins.serve(vec![None, None]).unwrap();
    let step = Batch {
        id: BatchId(0),
        group: GroupId(0),
        phase: Phase::Decode,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: BatchShape::decode(1, Variant::DecodeGreedy),
        requests: Vec::new(),
        columns: Columns::default(),
        media: Vec::new(),
        new_slots: Vec::new(),
    };
    for tokens in [3, 5] {
        assert_eq!(first.enter(tokens, &step).unwrap(), tokens);
    }

    // The other process ends: its worker closes, and with it its member, its relay and its
    // connections.
    Arc::into_inner(worker).unwrap().close().unwrap();
    drop((second_joins, second_group));
    assert!(first.enter(3, &step).is_err());
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
        multiprocessors: 148,
        attention_ranks: 0,
    };

    // The model splits no experts.
    let mut runner = runner(false);
    let alone = Member::new(
        Arc::clone(&groups[0]),
        shape,
        Mode::Dispatch,
        Arc::clone(&hook),
        BESIDE,
    );
    assert!(runner.join_union(alone.unwrap()).is_err());

    // A join batch reaches a runner outside a union.
    let mut join = decode(&[1], &[0.0]);
    join.shape = BatchShape::decode(1, Variant::ExpertJoin);
    assert!(matches!(
        runner.execute(&join, &held(&[1])),
        Err(Error::Invalid(_))
    ));

    // A member that is not the group's.
    let other = TransportShape {
        rank: 1,
        ranks: 2,
        ..shape
    };
    assert!(Member::new(Arc::clone(&groups[0]), other, Mode::Dispatch, hook, BESIDE).is_err());
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
            stepwise: false,
            rows: nn::family::StateRows::Whole,
        }],
    }
}

/// The second run of the stepwise test model's state: one layer whose field keeps a slot per
/// verified token.
fn stepwise_shape() -> StateShape {
    StateShape {
        layers: 1,
        fields: vec![StateField {
            name: "steps",
            dtype: DType::F32,
            dims: vec![HIDDEN],
            stepwise: true,
            rows: nn::family::StateRows::Whole,
        }],
    }
}

/// The rank's cache of the test model: one attention layer of the model's heads over
/// [`PAGES`] units, a row of [`TABLE_WIDTH`] pages per request slot, and a state region of 32
/// checkpoints for a model that keeps a recurrent state.
fn cache_config(stateful: bool) -> KvCacheConfig {
    let dtype = proto::request::PrefixCacheDtype::BFloat16;
    let layout = CacheLayout::of(PAGE, &fields(), dtype).unwrap();
    let layout = if stateful {
        layout.with_state(&[state_shape()]).unwrap()
    } else {
        layout
    };

    KvCacheConfig {
        layout: Arc::new(layout),
        units: PAGES as u32,
        slots: SLOTS,
        width: TABLE_WIDTH as u32,
        segments: None,
        workspace_bytes: 1 << 16,
        states: (if stateful { 32 } else { 0 }, None),
        windows: Vec::new(),
    }
}

/// A runner over `cache` and `state`, with every bucket profiled and captured.
fn over(cache: &KvCacheMemory, state: Option<&StateMemory>) -> CausalLmRunner {
    let config = config();
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&config, &Tiny::new(false)).unwrap(),
    );
    let model = Arc::new(Tiny::new(state.is_some()));

    let runner = CausalLmRunner::over(model, config, graphs, None, cache, state).unwrap();
    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
    }
    runner
}

/// The worker of one rank over a runner of [`Tiny`] over `cache`, serving generation 1, which
/// holds a kernel for every key the runner launches, with every bucket captured.
fn serving_worker(cache: &Arc<Mutex<KvCacheMemory>>) -> (Worker, Arc<Generation>) {
    use testkit::LONG;

    let keys = over(&lock(cache), None).graphs().keys();
    let serving = Arc::new(Generation::new(1, entries(&keys, &unlaunched_kernel())));
    let mut runner = CausalLmRunner::over(
        Arc::new(Tiny::new(false)),
        config(),
        Graphs::meta(
            Arc::clone(&serving),
            CausalLmRunner::input_bytes(&config(), &Tiny::new(false)).unwrap(),
        ),
        None,
        &lock(cache),
        None,
    )
    .unwrap();
    runner.set_page_work(wiring::page_work(Arc::clone(cache), None, None));

    let sequences = Sequences::new(GroupId(0), 1, Vec::new());
    let worker = Worker::spawn(
        None,
        vec![Box::new(runner)],
        Arc::clone(&serving),
        held(&[1, 2, 3]),
        sequences,
        0,
    )
    .unwrap();
    let warmed = worker.warm_up().unwrap().wait_timeout(LONG);
    assert!(matches!(warmed, Some(Outcome::Success(_))), "{warmed:?}");
    (worker, serving)
}

/// The buckets the worker's runner serves from, and the candidates of its shape's samplers.
fn served(worker: &Worker) -> Vec<Bucket> {
    worker
        .call("reading the served buckets", |runners| {
            Ok(runners[0].graphs().buckets())
        })
        .unwrap()
}

/// Prepares on `worker` generation `number` of `options`, built from `serving` with a kernel
/// for every key the options' shape launches, and waits until every bucket of the shape is
/// installed.
fn reshaped(
    worker: &Worker,
    serving: &Generation,
    number: u64,
    options: &crate::runner::GraphOptions,
) {
    use testkit::LONG;

    let measured = worker.measure(options, None).unwrap();
    let shape = measured[0]
        .clone()
        .expect("a causal runner takes graph options");
    let next = Arc::new(serving.replace(number, entries(&shape.keys, &unlaunched_kernel())));
    let reshape = Arc::new(crate::worker::Reshape {
        options: options.clone(),
        model: None,
        regions: vec![(None, shape.values)],
        recapture: false,
    });
    let prepared = worker
        .prepare(next, Some(reshape))
        .unwrap()
        .wait_timeout(LONG);
    assert!(
        matches!(prepared, Some(Outcome::Success(_))),
        "{prepared:?}"
    );
    let ready = worker.ready(number, None).wait_timeout(LONG);
    assert!(matches!(ready, Some(Outcome::Success(_))), "{ready:?}");
}

#[test]
fn a_generation_of_other_graph_options_switches_the_runner_to_its_buckets_whole() {
    use testkit::LONG;

    let cache = Arc::new(Mutex::new(
        KvCacheMemory::meta(cache_config(false)).unwrap(),
    ));
    let (worker, serving) = serving_worker(&cache);
    let before = served(&worker);
    let ended = |submission: crate::worker::Result<crate::worker::completion::Submission>| {
        submission.unwrap().wait_timeout(LONG)
    };

    // The catalog drops the decode bucket of four rows and the prefill of eighty tokens, and
    // gains a decode bucket of eight rows; a scored row reports one candidate fewer.
    let options = crate::runner::GraphOptions {
        buckets: Buckets::causal_lm(&[2, 8], &[8], true),
        candidates: CANDIDATES - 1,
        drafts: 0,
    };
    let after: Vec<Bucket> = Buckets::causal_lm(&[2, 8], &[8], true)
        .iter()
        .copied()
        .collect();
    reshaped(&worker, &serving, 2, &options);

    // Prepared and captured, the shape serves nothing until the switch.
    assert_eq!(served(&worker), before);
    assert!(matches!(ended(worker.switch(2)), Some(Outcome::Success(_))));
    assert_eq!(served(&worker), after);

    // A rollback serves the kept shape again, whole.
    assert!(matches!(
        ended(worker.rollback(2)),
        Some(Outcome::Success(_))
    ));
    assert_eq!(served(&worker), before);

    // Prepared again and committed, the shape serves on, and a later generation of the same
    // options captures its buckets again as any kernel generation does.
    reshaped(&worker, &serving, 3, &options);
    assert!(matches!(ended(worker.switch(3)), Some(Outcome::Success(_))));
    assert!(matches!(ended(worker.commit(3)), Some(Outcome::Success(_))));
    assert_eq!(served(&worker), after);
    let next = Arc::new(Generation::new(
        4,
        entries(
            &worker
                .call("reading the served keys", |runners| {
                    Ok(runners[0].graphs().keys())
                })
                .unwrap(),
            &unlaunched_kernel(),
        ),
    ));
    assert!(matches!(
        ended(worker.prepare(next, None)),
        Some(Outcome::Success(_))
    ));
    let order = worker.order(4).unwrap();
    let examined: Vec<Bucket> = {
        let mut buckets: Vec<Bucket> = order.into_iter().map(|(_, bucket)| bucket).collect();
        buckets.sort();
        buckets
    };
    assert_eq!(examined, after);
    worker.close().unwrap();
}

#[test]
fn options_the_runner_cannot_serve_in_place_prepare_nothing() {
    let cache = Arc::new(Mutex::new(
        KvCacheMemory::meta(cache_config(false)).unwrap(),
    ));
    let (worker, _) = serving_worker(&cache);

    // A runner over a cache that lays out no draft cache verifies no drafts.
    let drafting = crate::runner::GraphOptions {
        buckets: Buckets::causal_lm(&[2, 4], &[8, 80], true),
        candidates: CANDIDATES,
        drafts: 2,
    };
    let model: Arc<dyn TextModel> = Arc::new(Tiny::drafting(false));
    let refused = worker
        .measure(&drafting, Some(model))
        .unwrap_err()
        .to_string();
    assert!(refused.contains("lays no draft cache out"), "{refused}");
    worker.close().unwrap();
}

/// A runner that speculates nothing over a cache that lays the draft cache out: the draft cache
/// is lent to it, kept for a shape that verifies drafts.
fn unspeculating() -> CausalLmRunner {
    let config = CausalLmConfig {
        draft_pool: Some(POOL),
        ..config()
    };
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&config, &Tiny::new(false)).unwrap(),
    );
    let pooled = meta((SLOTS as usize + 1, TABLE_WIDTH), DType::U32);

    let mut runner = CausalLmRunner::new(
        Arc::new(Tiny::new(false)),
        config,
        graphs,
        None,
        vec![group(PAGES, pooled)],
        None,
    )
    .unwrap();
    runner.set_draft_cache(draft_cache(), None, None).unwrap();
    runner
}

/// A generation whose drafts start speculating serves verifications with the model it loaded
/// for them, over the draft cache the runner kept, from its switch on; one whose drafts stop
/// serves decodes again. Each refuses the serving model where it loads no drafter.
#[test]
fn a_generation_of_drafts_from_or_to_none_switches_between_decodes_and_verifications() {
    let catalog = Buckets::causal_lm(&[2, 4], &[8, 80], true);
    let options = |drafts: u32| crate::runner::GraphOptions {
        buckets: catalog.clone(),
        candidates: CANDIDATES,
        drafts,
    };
    let kernels = Arc::new(Generation::new(2, []));

    // Lent no trail and no drafter's state while it speculates nothing.
    let buffers = vec![vec![meta((12, HIDDEN), DType::BF16)]];
    assert!(
        unspeculating()
            .set_draft_cache(draft_cache(), Some(buffers), None)
            .is_err()
    );

    // Drafts need the drafter: the serving model loaded none.
    let mut runner = unspeculating();
    let refused = runner.measure(&options(DRAFTS), None).err().unwrap();
    assert!(refused.to_string().contains("no drafter"), "{refused}");

    let drafter: Arc<dyn TextModel> = Arc::new(Tiny::drafting(false));
    let measured = runner
        .measure(&options(DRAFTS), Some(&drafter))
        .unwrap()
        .unwrap();
    assert!(measured.storage > 0, "a verifying shape keeps its outputs");
    let values = CausalLmRunner::input_bytes(&speculating_config(), &*drafter).unwrap();
    assert!(
        runner
            .reshape(
                2,
                (&options(DRAFTS), Some(&drafter)),
                Arc::clone(&kernels),
                (None, values),
            )
            .unwrap()
    );
    assert_eq!(
        runner.reshaped_phases(),
        [Phase::Prefill, Phase::Verify, Phase::Refill]
    );
    assert_eq!(runner.phases(), [Phase::Prefill, Phase::Decode]);

    assert!(runner.switch_shape(2).unwrap());
    assert_eq!(
        runner.phases(),
        [Phase::Prefill, Phase::Verify, Phase::Refill]
    );
    assert!(
        runner
            .buckets()
            .iter()
            .any(|bucket| super::verifies(bucket.variant))
    );

    // Rolled back, the runner decodes again with the serving model.
    assert!(runner.rollback_shape(2).unwrap());
    assert_eq!(runner.phases(), [Phase::Prefill, Phase::Decode]);
    assert!(
        !runner
            .buckets()
            .iter()
            .any(|bucket| super::verifies(bucket.variant))
    );

    // A speculating runner's generation of no drafts decodes with the model that released the
    // drafter, and its commit releases the verifying shape.
    let mut speculating = speculating(false);
    let released: Arc<dyn TextModel> = Arc::new(Tiny::new(false));
    let values = CausalLmRunner::input_bytes(&config(), &*released).unwrap();
    assert!(
        speculating
            .reshape(3, (&options(0), Some(&released)), kernels, (None, values))
            .unwrap()
    );
    assert!(speculating.switch_shape(3).unwrap());
    assert_eq!(
        speculating.phases(),
        [Phase::Prefill, Phase::Decode, Phase::Refill]
    );
    assert!(speculating.commit_shape(3).unwrap());
    assert_eq!(
        speculating.phases(),
        [Phase::Prefill, Phase::Decode, Phase::Refill]
    );
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
            shapes: vec![state_shape()],
            slots: SLOTS,
            rows: 4,
            trail: 0,
            owned: 1,
            drafter: Vec::new(),
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
        CausalLmRunner::input_bytes(&self::config(), &Tiny::new(false)).unwrap(),
    );
    let model = Arc::new(Tiny::new(false));
    assert!(CausalLmRunner::over(model, self::config(), graphs, None, &fewer, None).is_err());
}

/// The regions a test cache's memory is numbered by: its units, its state units when it keeps a
/// state group, and the host pages of each of its groups.
fn regions(cache: &KvCacheMemory) -> CacheRegions {
    let groups = cache.layout().groups.len();

    CacheRegions {
        units: UNITS,
        states: cache.layout().state_group().map(|_| STATES),
        host: (0..groups).map(|group| Some(4 + group as u8)).collect(),
        windows: Vec::new(),
    }
}

/// A runner over the rank's `cache` and `state`, with the hook that issues its page work on
/// them.
fn wired(
    cache: &Arc<Mutex<KvCacheMemory>>,
    state: Option<&Arc<Mutex<StateMemory>>>,
) -> CausalLmRunner {
    let held = state.map(|state| lock(state));
    let mut runner = over(&lock(cache), held.as_deref());
    drop(held);

    runner.set_page_work(wiring::page_work(
        Arc::clone(cache),
        state.map(Arc::clone),
        None,
    ));
    runner
}

#[test]
fn the_tables_changes_are_written_into_the_pooled_rows_a_forward_gathers() {
    let cache = Arc::new(Mutex::new(
        KvCacheMemory::meta(cache_config(false)).unwrap(),
    ));
    let mut runner = wired(&cache, None);

    // Every row of the pooled table names the padding unit before a request holds a block.
    let width = TABLE_WIDTH;
    assert!(
        lock(&cache)
            .host_table(0)
            .unwrap()
            .iter()
            .all(|unit| *unit == 0)
    );

    // The last slot and the first get blocks from the buffer pool; the worker hands the
    // runner the table's changes before the batch that reads them.
    let mut tables = held(&[]);
    for (slot, units) in [(SLOTS, [3, 5]), (1, [7, 8])] {
        attach(&mut tables, slot);
        allocate(
            &mut tables,
            slot,
            0,
            0,
            &[(UNITS, units[0], 1), (UNITS, units[1], 1)],
        );
    }
    runner.write_tables(&tables.take_changes()).unwrap();

    {
        let cache = lock(&cache);
        let table = cache.host_table(0).unwrap();

        // Slot `s` owns row `s - 1`, the row the runner's forward gathers for it, and the
        // padding row past the last slot's still names the padding unit.
        assert_eq!(table[..width], [7, 8, 0, 0]);
        assert_eq!(table[(SLOTS as usize - 1) * width..][..width], [3, 5, 0, 0]);
        assert_eq!(table[SLOTS as usize * width..], [0; TABLE_WIDTH]);
    }

    // Entries that leave name the padding unit again.
    tables
        .apply(&Request::Release {
            owner: OwnerId(u64::from(SLOTS)),
            class: 0,
            at: 0,
            count: 1,
            after: vec![],
        })
        .unwrap();
    runner.write_tables(&tables.take_changes()).unwrap();
    let row = lock(&cache).host_table(0).unwrap()[(SLOTS as usize - 1) * width..][..width].to_vec();
    assert_eq!(row, [0, 5, 0, 0]);

    // A change the cache refuses fails, and writes nothing: a unit past the pool.
    let mut outside = held(&[]);
    attach(&mut outside, 2);
    allocate(&mut outside, 2, 0, 0, &[(UNITS, PAGES as u32, 1)]);
    let refused = runner.write_tables(&outside.take_changes()).unwrap_err();
    assert!(matches!(refused, Error::Invalid(_)), "{refused}");
    assert!(
        lock(&cache).host_table(0).unwrap()[width..2 * width]
            .iter()
            .all(|unit| *unit == 0)
    );
}

#[test]
fn a_copy_between_two_runs_is_issued_on_the_cache_and_ends_with_the_batch() {
    let cache = Arc::new(Mutex::new(
        KvCacheMemory::meta(cache_config(false)).unwrap(),
    ));
    let mut runner = wired(&cache, None);
    let clone = |runs: Vec<(Run, Run)>| BlockCopy {
        request: storage::runtime::RequestId(5),
        runs,
        encodings: None,
    };

    // A copy names its runs by the regions the buffer pool numbered: before they are stated,
    // none is issued.
    let units = clone(vec![((UNITS, 3, 1), (UNITS, 4, 1))]);
    let unstated = runner.copy(std::slice::from_ref(&units)).unwrap_err();
    assert!(matches!(unstated, Error::State(_)), "{unstated}");

    let stated = regions(&lock(&cache));
    lock(&cache).set_regions(stated).unwrap();
    runner.copy(&[units, clone(Vec::new())]).unwrap();

    // A copy from memory that is no cache's, or into the padding unit, is refused whole.
    let foreign = clone(vec![((9, 3, 1), (UNITS, 4, 1))]);
    assert!(runner.copy(&[foreign]).is_err());
    let padding = clone(vec![((UNITS, 3, 1), (UNITS, 0, 1))]);
    assert!(runner.copy(&[padding]).is_err());

    // A batch's end waits for the copies taken with it, also when the step failed.
    let tables = held(&[1, 2]);
    let refused = runner
        .execute(&decode(&[1, 2], &[0.0, 0.0]), &tables)
        .unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");
}

#[test]
fn a_batch_with_no_rows_writes_the_tables_and_runs_the_copies_without_a_forward() {
    use testkit::LONG;

    let cache = Arc::new(Mutex::new(
        KvCacheMemory::meta(cache_config(false)).unwrap(),
    ));
    let stated = regions(&lock(&cache));
    lock(&cache).set_regions(stated).unwrap();

    // The worker of the one rank of group 0, over the runner wired to the rank's cache. Its
    // graphs hold no device, so any forward the batch ran would fail it.
    let group = GroupId(0);
    let sequences = Sequences::new(group, 1, Vec::new());
    let tables = BlockTables::new(group, &[UNITS], &[(UNITS, 1024)]);
    let worker = Worker::spawn(
        None,
        vec![Box::new(wired(&cache, None))],
        Arc::new(Generation::new(0, [])),
        tables,
        Arc::clone(&sequences),
        0,
    )
    .unwrap();
    let map = |id: u64, request: Request, completion: Completion| {
        let mapped = worker
            .map(1, storage::runtime::RequestId(id), request, completion)
            .unwrap();
        assert!(matches!(
            mapped.wait_timeout(LONG),
            Some(Outcome::Success(_))
        ));
    };

    // Slot 1's owner is given two blocks, and its clone of a cached block into a third is
    // queued.
    let owner = OwnerId(1);
    map(
        1,
        Request::Attach { owner, group },
        Completion::Attached { slot: 1 },
    );
    let allocate = Request::Allocate {
        owner,
        class: 0,
        at: 0,
        count: 2,
        units: 0,
    };
    let allocated = Completion::Allocated {
        runs: vec![(UNITS, 7, 1), (UNITS, 8, 1)],
        evicted: vec![],
    };
    map(2, allocate, allocated);
    let clone = Request::Clone {
        owner,
        class: 0,
        at: 2,
        key: Key([3; 32]),
    };
    let copy = Completion::Copy {
        group,
        keys: vec![Key([3; 32])],
        from: vec![(UNITS, 3, 1)],
        to: vec![(UNITS, 9, 1)],
        evicted: vec![],
        encodings: None,
        parts: Vec::new(),
    };
    map(3, clone, copy);

    // The batch takes its number, writes the changes into the slot's pooled row, starts the
    // clone and ends it, and names no call.
    let empty = Batch {
        id: BatchId(1),
        group,
        phase: Phase::Decode,
        kernels: 0,
        policy: 0,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: BatchShape::decode(0, Variant::DecodeGreedy),
        requests: Vec::new(),
        columns: Columns::default(),
        media: Vec::new(),
        new_slots: Vec::new(),
    };
    let submission = worker.submit(1, empty).unwrap();
    let Some(Outcome::Success((output, requests))) = submission.wait_timeout(LONG) else {
        panic!("the batch with no rows is served");
    };

    assert!(output.calls.is_empty());
    let cloned = Request::Finish {
        copy: storage::runtime::RequestId(3),
        ok: true,
        after: vec![(group, 1)],
    };
    assert_eq!(requests, [cloned]);
    assert_eq!(sequences.completed(), 1);
    assert_eq!(
        lock(&cache).host_table(0).unwrap()[..TABLE_WIDTH],
        [7, 8, 0, 0]
    );
}

#[test]
fn a_state_checkpoint_is_read_and_written_at_the_run_its_position_names() {
    let config = cache_config(true);
    let state_class = config.layout.state_group().unwrap() as u8;
    let cache = KvCacheMemory::meta(config.clone()).unwrap();
    let state = StateMemory::meta(
        StateConfig {
            shapes: vec![state_shape()],
            slots: SLOTS,
            rows: 4,
            trail: 0,
            owned: 1,
            drafter: Vec::new(),
        },
        &config,
    )
    .unwrap();
    let (cache, state) = (Arc::new(Mutex::new(cache)), Arc::new(Mutex::new(state)));
    let mut runner = wired(&cache, Some(&state));

    // Slot 1 holds a checkpoint at position 2 of its state table and slot 2 one at position
    // 0, each one unit of the state region.
    let mut tables = held(&[1, 2]);
    allocate(&mut tables, 1, state_class, 2, &[(STATES, 20, 1)]);
    allocate(&mut tables, 2, state_class, 0, &[(STATES, 24, 1)]);
    let copy = |slot, position, direction| StateCopy {
        slot,
        position,
        direction,
    };

    // The restore reads slot 1's checkpoint at its position and the keep writes slot 2's;
    // both lie in the tables, so the step reaches the replay, which graphs that hold no
    // device refuse.
    let mut batch = prefill(Variant::PrefillWrite);
    batch.columns.rows.calls[0].slot = 1;
    batch.columns.rows.calls[1].slot = 2;
    batch.columns.slots = vec![1, 2];
    batch.columns.commit = commit(&[(1, mode::CACHE, -1), (2, mode::CACHE, -1)]);
    batch.columns.rows.logits_rows = vec![-1, -1];
    batch.columns.checkpoints = vec![0, 2];
    batch.columns.state_copies = vec![
        copy(1, 2, StateDirection::Restore),
        copy(2, 0, StateDirection::Keep),
    ];
    let reached = runner.execute(&batch, &tables).unwrap_err();
    assert!(matches!(reached, Error::State(_)), "{reached}");

    let (restores, keeps) = wiring::state_copies(&lock(&cache), &batch, &tables).unwrap();
    assert_eq!(restores, [(1, 20)]);
    assert_eq!(keeps, [(2, 24)]);

    // An entry of more than one unit is no checkpoint: the batch is refused.
    allocate(&mut tables, 1, state_class, 4, &[(STATES, 26, 2)]);
    let mut wide = batch.clone();
    wide.columns.state_copies[0].position = 4;
    let message = runner.execute(&wide, &tables).unwrap_err().to_string();
    assert!(
        message.contains("one unit of the state region"),
        "{message}"
    );

    // A checkpoint at a position without an entry is refused before anything is launched.
    let mut missing = batch.clone();
    missing.columns.state_copies[0].position = 3;
    let message = runner.execute(&missing, &tables).unwrap_err().to_string();
    assert!(
        message.contains("position 3") && message.contains("holds no entry"),
        "{message}"
    );

    // A keep into a checkpoint a commit made read-only writes nothing: the batch is refused.
    commit_entry(&mut tables, 2, state_class, 0);
    let message = runner.execute(&batch, &tables).unwrap_err().to_string();
    assert!(message.contains("read-only"), "{message}");
}

#[test]
fn a_paired_forward_computes_over_the_scratch_pages_the_cache_opened() {
    let mut cache = KvCacheMemory::meta(cache_config(false)).unwrap();
    let tables = held(&[1, SLOTS]);

    // A cache with no scratch pages lends none, and a set of other pages than the runner's
    // config states is refused.
    let mut unlent = over(&cache, None);
    assert!(unlent.lend_scratch(&cache).is_err());
    assert!(matches!(
        unlent.prepare_shadow_inputs(&decode(&[1, 2], &[0.0, 0.0]), &tables),
        Err(Error::Unsupported(_))
    ));
    let mut other = KvCacheMemory::meta(cache_config(false)).unwrap();
    other.open_scratch(SCRATCH_PAGES + 1, None).unwrap();
    assert!(unlent.lend_scratch(&other).is_err());

    // The set the rows of a batch share.
    cache.open_scratch(SCRATCH_PAGES, None).unwrap();
    let mut runner = over(&cache, None);
    let bucket = runner
        .prepare_shadow_inputs(&decode(&[1, SLOTS as i32], &[0.0, 0.0]), &tables)
        .unwrap();
    assert_eq!(bucket, Bucket::decode(2, Variant::DecodeGreedy));

    // The forward over the scratch planes makes the serving forward's launches.
    let serving = operators(&runner, &bucket);
    let context = ops::Context::meta(None);
    context.record_launches();
    {
        let _guard = context.enter();
        runner.forward(&bucket).unwrap();
    }
    let paired: Vec<&str> = context
        .take_launches()
        .into_iter()
        .map(|(operator, ..)| operator)
        .collect();
    assert_eq!(paired, serving);
}

// ---- Speculation --------------------------------------------------------------------------------

use crate::batch::DraftWork;

use super::output::verification_output;
use super::speculation::{self, DraftPool};
use service::lock;

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

    KvCache {
        block_tables: meta((1, 1), DType::U32),
        slot_mapping: meta(0, DType::U32),
        starts: None,
        page_size: POOL.page,
        planes: KvCache::planes_of(&fields(), DType::BF16, pages, POOL.page, |shape, dtype| {
            Ok(meta(shape, dtype))
        })
        .unwrap(),
        workspace: meta(1 << 16, DType::U8),
    }
}

/// A speculating runner of [`Tiny`] with its drafter, over graphs that hold no device.
fn speculating(stateful: bool) -> CausalLmRunner {
    let config = speculating_config();
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&config, &Tiny::new(false)).unwrap(),
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
        .set_draft_cache(draft_cache(), stateful.then(|| buffers(12)), None)
        .unwrap();
    runner
}

/// A verification batch of two calls, stated as the scheduler states it, in a decode's shape:
/// slot 2's committed token 5 and the drafts 6 and 7 from position 10, and slot 4's committed
/// token 8 and the draft 9 from position 20. The first call proposes two drafts and the
/// second one, and the first counts its tokens in its penalties.
fn verification(temperature: f32) -> Batch {
    let call = |slot: u32, row: u32| Row {
        slot,
        kind: Phase::Verify,
        replay: 0..0,
        rows: row..row + 1,
    };

    let mut table = commit(&[(2, mode::VERIFY, 0), (4, mode::VERIFY, 1)]);
    table[column::LOGICAL * 2..][..2].copy_from_slice(&[10, 20]);
    table[column::SAMPLING * 2..][..2].copy_from_slice(&[3, 5]);
    table[column::VISIBLE * 2..][..2].copy_from_slice(&[10, 20]);
    table[column::PENALTIES * 2] = 1;

    let columns = Columns {
        rows: Rows {
            calls: vec![call(2, 0), call(4, 1)],
            positions: vec![10, 11, 12, 20, 21],
            cache_positions: vec![10, 11, 12, 20, 21],
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
        ..Columns::default()
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
    let layout = Layout::of(buckets.iter(), 1, DRAFTS, POOL.width(MAX_KV_LEN), HISTORY).unwrap();
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
    let kept = CausalLmRunner::kept_bytes(&config, &Tiny::drafting(false)).unwrap();
    assert!(kept >= (80 * HIDDEN * 2) as u64);
    assert_eq!(
        CausalLmRunner::kept_bytes(&self::config(), &Tiny::drafting(false)).unwrap(),
        0
    );
    assert!(
        CausalLmRunner::input_bytes(&config, &Tiny::new(false)).unwrap()
            > CausalLmRunner::input_bytes(&self::config(), &Tiny::new(false)).unwrap()
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
                assert_eq!(
                    count(&launches, "sampling_from_logits"),
                    1 - greedy,
                    "{bucket}"
                );

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
        crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
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
        crate::runner::graphs::tests::warm(&stateful, &bucket).unwrap();
    }

    // A model that keeps a state verifies only with a trail of its largest verification.
    let mut unlent = speculating(true);
    let short = vec![vec![meta((11, HIDDEN), DType::BF16)]];
    assert!(
        unlent
            .set_draft_cache(draft_cache(), Some(short), None)
            .is_err()
    );
    assert!(unlent.set_draft_cache(draft_cache(), None, None).is_err());
}

#[test]
fn a_verification_batch_is_one_row_per_token_and_one_sequence_per_call() {
    let batch = verification(0.7);
    let bucket = Bucket::verify(4, DRAFTS, Variant::VerifySampled);
    let (layout, mut block) = speculating_block();

    let mapping = mapping(&batch, &held(&[2, 4]));
    speculation::fill_verification(
        &layout,
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        &speculation::token_rows(&batch.columns),
        &bucket,
        &[mapping],
        &mut block,
    )
    .unwrap();

    let u32s = |column, elements| read::<u32>(&layout, &block, column, elements);
    let i32s = |column, elements| read::<i32>(&layout, &block, column, elements);
    let none = NO_SLOT;

    // ---- Each token is the decode step at its position ----

    assert_eq!(u32s(Column::InputIds, 6), [5, 6, 7, 8, 9, 0]);
    assert_eq!(i32s(Column::Positions, 6), [10, 11, 12, 20, 21, 0]);

    // Each token writes the cache slot its request's entry names at its cache position.
    assert_eq!(
        u32s(Column::SlotMapping, 6),
        [
            written(2, 10),
            written(2, 11),
            written(2, 12),
            written(4, 20),
            written(4, 21),
            none
        ]
    );
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
    let draws = read::<f32>(&layout, &block, Column::Draws, 6);
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
    let tables = [(0, vec![3, 5], vec![3, 5]), (0, Vec::new(), Vec::new())];
    let bucket = Bucket::draft(4, 12);
    speculation::fill_drafter(
        &layout,
        &POOL,
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        &tables,
        &bucket,
        &mut block,
    )
    .unwrap();

    {
        let u32s = |column, elements| read::<u32>(&layout, &block, column, elements);
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

        // A drafter row is a call's tokens, and reads the keys up to its last one; it advances
        // the drafter's state of its call's slot, which a slot without draft pages keeps too.
        assert_eq!(u32s(Column::StateOffsets, 5), [0, 3, 5, 5, 5]);
        assert_eq!(u32s(Column::DraftLens, 4), [13, 22, 0, 0]);
        assert_eq!(u32s(Column::StateSlots, 4), [1, 3, none, none]);

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
        let positions = read::<i32>(&layout, &block, Column::ChainPositions, 2 * stride);
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
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        &tables,
        &bucket,
        &mut block,
    )
    .unwrap();

    {
        let u32s = |column, elements| read::<u32>(&layout, &block, column, elements);
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
        assert_eq!(u32s(Column::StateSlots, 4), [1, 3, none, none]);

        let positions = read::<i32>(&layout, &block, Column::ChainPositions, 2 * stride);
        assert_eq!(positions[stride..][..4], [3, 2, 0, 0]);
        let index = u32s(Column::ChainIndex, 2 * stride);
        assert_eq!(index[stride..][..4], [0, 6 + 3, 12 + 3, 18 + 3]);
    }

    // A row whose last token's successor nobody states has no entry there.
    batch.columns.drafts.clear();
    speculation::fill_drafter(
        &layout,
        &POOL,
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        &tables,
        &bucket,
        &mut block,
    )
    .unwrap();

    assert_eq!(
        read::<u32>(&layout, &block, Column::EntrySteps, 6),
        [0, 0, 0, 0, 1, 1]
    );
    assert_eq!(read::<u32>(&layout, &block, Column::LastEntries, 2), [2, 3]);

    // A row with no entry, slot 4's one token, advances no drafter state.
    let rows = &mut batch.columns.rows;
    rows.query_offsets = vec![0, 3, 4];
    rows.seq_lens = vec![3, 1];
    rows.positions.truncate(4);
    rows.cache_positions.truncate(4);
    batch.columns.input_ids.truncate(4);
    speculation::fill_drafter(
        &layout,
        &POOL,
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        &tables,
        &bucket,
        &mut block,
    )
    .unwrap();
    assert_eq!(
        read::<u32>(&layout, &block, Column::StateSlots, 2),
        [1, none]
    );
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
        columns: Columns {
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
    prompt.columns.rows.cache_positions.truncate(4);
    prompt.columns.input_ids.truncate(4);

    let mut calls = vec![performed(vec![11], 0, 0, 0), performed(Vec::new(), 0, 0, 0)];
    speculation::propose(&prompt, VOCABULARY as u32, &proposals, &mut calls).unwrap();
    assert_eq!(calls[0].draft_token_ids, [50]);
    assert!(calls[1].draft_token_ids.is_empty());
}

#[test]
fn a_speculating_runner_serves_a_verification_from_its_verification_bucket() {
    let mut runner = speculating(false);
    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
    }

    // A batch in a decode's shape whose calls verify takes the verification bucket of its
    // selection, the smallest that holds its calls.
    let tables = held(&[1, 2, 3, 4]);
    let bucket = runner.prepare_inputs(&verification(0.0), &tables).unwrap();
    assert_eq!(bucket, Bucket::verify(2, DRAFTS, Variant::VerifyGreedy));
    let bucket = runner.prepare_inputs(&verification(0.7), &tables).unwrap();
    assert_eq!(bucket, Bucket::verify(2, DRAFTS, Variant::VerifySampled));

    // A decode is a verification of no draft.
    let mut decoded = decode(&[1, 2, 3], &[0.0; 3]);
    decoded.columns.input_ids = vec![4, 5, 6];
    let bucket = runner.prepare_inputs(&decoded, &tables).unwrap();
    assert_eq!(bucket, Bucket::verify(4, DRAFTS, Variant::VerifyGreedy));

    // A prompt is served from its prefill bucket.
    let bucket = runner
        .prepare_inputs(&prefill(Variant::PrefillGreedy), &tables)
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
    long.requests.truncate(1);
    let message = runner
        .prepare_inputs(&long, &tables)
        .unwrap_err()
        .to_string();
    assert!(message.contains("carries 4 drafts"), "{message}");

    let mut mixed = verification(0.0);
    mixed.columns.rows.calls[1].kind = Phase::Prefill;
    let message = runner
        .prepare_inputs(&mixed, &tables)
        .unwrap_err()
        .to_string();
    assert!(message.contains("mixes prompt rows"), "{message}");

    // A verification's token rows share their request's pages: it has no paired forward.
    assert!(matches!(
        runner.prepare_shadow_inputs(&verification(0.0), &tables),
        Err(Error::Unsupported(_))
    ));

    // Graphs that hold no device replay nothing: the step is refused at its replay, after
    // both graphs' columns were padded.
    let refused = runner.execute(&verification(0.0), &tables).unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");
}

#[test]
fn speculation_is_refused_where_the_runner_does_not_serve_it() {
    // A config that states drafts for a model that loaded no drafter, or without the draft
    // pool of its cache.
    let build = |model: Tiny, config: CausalLmConfig| {
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            CausalLmRunner::input_bytes(&config, &model)?,
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
    assert!(
        runner(false)
            .set_draft_cache(draft_cache(), None, None)
            .is_err()
    );

    // Experts split across ranks.
    let groups = host_groups(2);
    let (inbox, _outbox) = crossbeam_channel::unbounded();
    let refused = speculating(false)
        .join_union(member(&groups[0], 0, deliver(inbox)))
        .unwrap_err();
    assert!(matches!(refused, Error::Unsupported(_)), "{refused}");

    // A refill of the draft cache of a model that keeps a recurrent state.
    let refused = speculating(true)
        .prepare_inputs(&refill(), &held(&[2, 4]))
        .unwrap_err();
    assert!(matches!(refused, Error::Unsupported(_)), "{refused}");
}

/// A prompt batch that refills slot 2's draft cache from its first position.
fn refill() -> Batch {
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
    refill
}

/// A speculating runner of [`Tiny`], which keeps no state, whose drafter keeps one, over graphs
/// that hold no device and lent no draft cache yet.
fn speculating_with_drafter_state() -> CausalLmRunner {
    let config = speculating_config();
    let graphs = Graphs::meta(
        Arc::new(Generation::new(0, [])),
        CausalLmRunner::input_bytes(&config, &Tiny::new(false)).unwrap(),
    );
    let pooled = meta((SLOTS as usize + 1, TABLE_WIDTH), DType::U32);
    let model = Tiny {
        drafter: Some(TinyDrafter::new(true)),
        ..Tiny::new(false)
    };

    CausalLmRunner::new(
        Arc::new(model),
        config,
        graphs,
        None,
        vec![group(PAGES, pooled)],
        None,
    )
    .unwrap()
}

#[test]
fn a_drafter_that_keeps_a_state_is_lent_it_by_each_of_its_forwards() {
    let ring = |slots: u32| vec![vec![meta((slots as usize, 4, HIDDEN), DType::BF16)]];

    // The drafter refuses a forward that withholds its state or lends another: its entries
    // and its chained steps are lent its live ring over their rows, and every bucket is
    // profiled and captured.
    let mut runner = speculating_with_drafter_state();
    runner
        .set_draft_cache(draft_cache(), None, Some(ring(SLOTS)))
        .unwrap();
    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
    }

    // Each proposal is the argmax of the model's head over the drafter's own read of its rows,
    // a norm the drafter that keeps no state does not make.
    let plain = speculating(false);
    plain.graphs().profile_run(&plain).unwrap();
    let norm = <ops::norm::RmsNorm as ops::Op>::NAME;
    let norms = |runner: &CausalLmRunner| count(&operators(runner, &Bucket::draft(4, 12)), norm);
    assert_eq!(norms(&runner), norms(&plain) + DRAFTS as usize);

    // The state is the drafter's own over every request slot, and a drafter that keeps none is
    // lent none.
    let mut unlent = speculating_with_drafter_state();
    assert!(unlent.set_draft_cache(draft_cache(), None, None).is_err());
    assert!(
        unlent
            .set_draft_cache(draft_cache(), None, Some(ring(SLOTS - 1)))
            .is_err()
    );
    assert!(
        speculating(false)
            .set_draft_cache(draft_cache(), None, Some(ring(SLOTS)))
            .is_err()
    );

    // A refill evaluates positions behind its slot's own, which would advance the slot's
    // drafter state from there.
    let refused = runner
        .prepare_inputs(&refill(), &held(&[2, 4]))
        .unwrap_err();
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

    // A slot's draft pages are its entries of the draft group's class: those from its first
    // on, a page between two entries holding none, and the drafter writes only the writable
    // ones.
    let mut tables = held(&[]);
    attach(&mut tables, 2);
    let class = pool.group as u8;
    allocate(&mut tables, 2, class, 2, &[(UNITS, 7, 1)]);
    allocate(&mut tables, 2, class, 4, &[(UNITS, 9, 1)]);
    assert_eq!(
        pool.table(&tables, 2).unwrap(),
        (2, vec![7, 0, 9], vec![7, 0, 9])
    );
    assert_eq!(pool.table(&tables, 3).unwrap(), (0, Vec::new(), Vec::new()));

    commit_entry(&mut tables, 2, class, 2);
    let (first, read, writable) = pool.table(&tables, 2).unwrap();
    assert_eq!((first, &read[..]), (2, &[7, 0, 9][..]));
    assert_eq!(writable, [0, 0, 9]);
    assert_eq!(pool.slot(first, &writable, 1100), NO_SLOT);
    assert_eq!(pool.slot(first, &read, 1100), 107 * 64 + 12);
}

#[test]
fn a_speculating_runner_is_built_over_the_rank_s_draft_group_and_trail() {
    let dtype = proto::request::PrefixCacheDtype::BFloat16;
    let layout = CacheLayout::of(PAGE, &fields(), dtype)
        .and_then(|layout| layout.with_state(&[state_shape()]))
        .and_then(|layout| layout.with_draft(&fields()))
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
                shapes: vec![state_shape()],
                slots: SLOTS,
                rows: 4,
                trail,
                owned: 1,
                drafter: Vec::new(),
            },
            &cache_config,
        )
        .unwrap()
    };
    let over = |config: CausalLmConfig, state: &StateMemory, drafter: TinyDrafter| {
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            CausalLmRunner::input_bytes(&config, &Tiny::new(false))?,
        );
        let model = Tiny {
            drafter: Some(drafter),
            ..Tiny::new(true)
        };

        CausalLmRunner::over(
            Arc::new(model),
            config,
            graphs,
            None,
            &lock(&cache),
            Some(state),
        )
    };
    let plain = || TinyDrafter::new(false);

    // The runner is lent the draft group's planes and the state's trail, and captures its
    // verification and drafter buckets over them.
    let mut runner = over(config.clone(), &state(12), plain()).unwrap();
    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
    }
    assert_eq!(runner.graphs().buckets().len(), 16);

    // A state opened with no trail, or with one shorter than a verification, verifies
    // nothing; neither does a config that states another draft pool than the cache's.
    assert!(over(config.clone(), &state(0), plain()).is_err());
    assert!(over(config.clone(), &state(11), plain()).is_err());

    // A drafter that keeps a state is lent the one the rank's state keeps for it; a state that
    // keeps none for it serves no such drafter.
    let ringed = StateMemory::meta(
        StateConfig {
            shapes: vec![state_shape()],
            slots: SLOTS,
            rows: 4,
            trail: 12,
            owned: 1,
            drafter: vec![drafter_ring()],
        },
        &cache_config,
    )
    .unwrap();
    let drafting = over(config.clone(), &ringed, TinyDrafter::new(true)).unwrap();
    drafting.graphs().profile_run(&drafting).unwrap();
    assert!(over(config.clone(), &state(12), TinyDrafter::new(true)).is_err());
    let moved = CausalLmConfig {
        draft_pool: Some(DraftPool {
            units: pool.units + 1,
            ..pool
        }),
        ..config
    };
    assert!(over(moved, &state(12), plain()).is_err());

    // The drafter's rows read their slots' entries of the draft group's class: a batch whose
    // slot holds a draft page reaches the replay with its drafter's table stated.
    let mut tables = held(&[2, 4]);
    allocate(&mut tables, 2, pool.group as u8, 0, &[(UNITS, 30, 1)]);
    let refused = runner.execute(&verification(0.0), &tables).unwrap_err();
    assert!(matches!(refused, Error::State(_)), "{refused}");
}

// ---- Token ids and verify slots ----------------------------------------------------------------

/// The test model with layers that read token ids, and its drafter when `drafting`.
fn reading(drafting: bool) -> Tiny {
    let model = if drafting {
        Tiny::drafting(false)
    } else {
        Tiny::new(false)
    };

    Tiny {
        reads: true,
        ..model
    }
}

/// A CPU column of `values`.
fn u32_column(values: &[u32]) -> Tensor {
    Tensor::from_slice(values, values.len(), &Device::Cpu).unwrap()
}

/// A context table of the runner's request slots and the padding row, all zero but the rows
/// `started` names: request slot `s` started with the context `[a, b]`, as `Slots::start`
/// writes `NewSlot::context`.
fn context_table(started: &[(u32, [u32; 2])]) -> Vec<[u32; 2]> {
    let mut rows = vec![[0, 0]; SLOTS as usize + 1];
    for (slot, context) in started {
        rows[(*slot - 1) as usize] = *context;
    }
    rows
}

/// The rows of a `[rows, 2]` U32 tensor.
fn pairs(table: &Tensor) -> Vec<[u32; 2]> {
    table
        .to_vec2::<u32>()
        .unwrap()
        .into_iter()
        .map(|row| [row[0], row[1]])
        .collect()
}

/// The rows of `table` that `slots`, each a row of it, gather, as a forward's contexts:
/// `[slots, 2]` U32.
fn gather(table: &[[u32; 2]], slots: &Tensor) -> Tensor {
    let flat: Vec<u32> = table.iter().flatten().copied().collect();
    let table = Tensor::from_slice(&flat, (table.len(), 2), &Device::Cpu).unwrap();

    table.index_select(slots, 0).unwrap()
}

/// Writes `next`, `[rows, 2]`, into the rows of `table` that `slots`, `[rows]`, name, as the
/// forward's scatter does: a row on no slot writes nothing.
fn scatter(table: &mut [[u32; 2]], slots: &Tensor, next: &Tensor) {
    let slots = slots.to_vec1::<u32>().unwrap();
    for (slot, pair) in slots.into_iter().zip(pairs(next)) {
        if slot != NO_SLOT {
            table[slot as usize] = pair;
        }
    }
}

/// A forward of `batch` in `bucket` over the context table `table`, on the host, through the
/// columns the runner pads it with (`columns`, token ids): returns the context each row is lent
/// (`nn::TokenBatch::token_history`), and leaves in `table` what each row's slot continues
/// from. A decode's token is the one its slot's commit left, which the batch states as well.
fn advance(table: &mut [[u32; 2]], batch: &Batch, bucket: &Bucket) -> Vec<[u32; 2]> {
    let (layout, block) = filled(batch, bucket, &[], None);
    let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
    let column = |column, elements| u32_column(&read::<u32>(&layout, &block, column, elements));

    let contexts = gather(table, &column(Column::SlotRows, rows));
    let index = column(Column::NextContext, 2 * rows)
        .reshape((rows, 2))
        .unwrap();
    let next = super::history_ids(&contexts, &column(Column::InputIds, tokens), &index).unwrap();

    scatter(table, &column(Column::Slots, rows), &next);
    pairs(&contexts)
}

/// A prompt chunk of `ids` on slot 2, at cache positions from `start`: one call of one row.
fn prompt(ids: &[i32], start: i32) -> Batch {
    let positions: Vec<i32> = (start..start + ids.len() as i32).collect();
    let mut batch = chunk(&positions);

    batch.columns.input_ids = ids.to_vec();
    batch
}

#[test]
fn a_model_that_reads_token_ids_is_lent_them_by_every_forward() {
    // The model refuses a forward that withholds the token ids it reads, and one that lends
    // them to a model that reads none: every bucket, the drafter's among them, is profiled and
    // captured.
    for drafting in [false, true] {
        let config = if drafting {
            speculating_config()
        } else {
            config()
        };
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            CausalLmRunner::input_bytes(&config, &reading(drafting)).unwrap(),
        );
        let pooled = meta((SLOTS as usize + 1, TABLE_WIDTH), DType::U32);

        let mut runner = CausalLmRunner::new(
            Arc::new(reading(drafting)),
            config,
            graphs,
            None,
            vec![group(PAGES, pooled)],
            None,
        )
        .unwrap();
        if drafting {
            runner.set_draft_cache(draft_cache(), None, None).unwrap();
        }

        runner.graphs().profile_run(&runner).unwrap();
        for bucket in runner.buckets() {
            crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
        }
    }

    // The slots keep two ids each, and the padding row's, for such a model alone.
    let plain = CausalLmRunner::slot_bytes(&config(), &Tiny::new(false)).unwrap();
    let read = CausalLmRunner::slot_bytes(&config(), &reading(false)).unwrap();
    assert!(read >= plain + 2 * 4 * (u64::from(SLOTS) + 1));
}

/// A slot continues from as many ids as its model reads: a row's last ones, led by its
/// context's later ids when it holds fewer.
#[test]
fn a_slot_continues_from_as_many_ids_as_its_model_reads() {
    const WIDTH: usize = 3;
    let layout = Layout::of(config().buckets.iter(), 1, 0, 0, WIDTH as u32).unwrap();
    let bucket = Bucket::prefill(4, 8, Variant::PrefillWrite);

    // Slot 2's context is 1 2 3 before each chunk.
    for (ids, expected) in [(vec![7], [2, 3, 7]), (vec![7, 8, 9, 10], [8, 9, 10])] {
        let batch = prompt(&ids, 4);
        let mut block = vec![0xA5_u8; layout.bytes() as usize];
        columns::fill(
            &layout,
            SLOTS,
            VOCABULARY as u32,
            &batch.columns,
            &bucket,
            &[],
            None,
            &mut block,
        )
        .unwrap();

        let rows = bucket.rows as usize;
        let column = |column, elements| u32_column(&read::<u32>(&layout, &block, column, elements));
        let contexts =
            Tensor::from_slice(&[1_u32, 2, 3].repeat(rows), (rows, WIDTH), &Device::Cpu).unwrap();
        let index = column(Column::NextContext, WIDTH * rows)
            .reshape((rows, WIDTH))
            .unwrap();
        let next = super::history_ids(
            &contexts,
            &column(Column::InputIds, bucket.tokens as usize),
            &index,
        )
        .unwrap();

        assert_eq!(next.to_vec2::<u32>().unwrap()[0], expected, "{ids:?}");
    }
}

#[test]
fn a_slot_s_context_follows_its_request_across_its_prompt_chunks_and_decodes() {
    // Slot 2's request has no token before its prompt, which its start fills with the model's
    // fill id; slot 4's started elsewhere.
    let mut table = context_table(&[(2, [63, 63]), (4, [21, 22])]);
    let bucket = Bucket::prefill(4, 8, Variant::PrefillWrite);

    // Its first chunk, 5 6 7, reads the request's context, and the slot continues from the
    // chunk's last two ids.
    let lent = advance(&mut table, &prompt(&[5, 6, 7], 0), &bucket);
    assert_eq!(lent[0], [63, 63]);
    assert_eq!(table[1], [6, 7]);

    // The second chunk, 8 alone, reads the two ids before it across the chunk boundary, and
    // the slot continues from its context's later id and its token.
    let lent = advance(&mut table, &prompt(&[8], 3), &bucket);
    assert_eq!(lent[0], [6, 7]);
    assert_eq!(table[1], [7, 8]);

    // A decode of 9 continues the same way. A padding row reads the padding row, and writes
    // nothing: every other slot and the padding row keep what they held.
    let mut step = decode(&[2], &[0.0]);
    step.columns.input_ids = vec![9];
    let lent = advance(&mut table, &step, &Bucket::decode(2, Variant::DecodeGreedy));
    assert_eq!(lent, [[7, 8], [0, 0]]);

    let mut expected = vec![[0, 0]; SLOTS as usize + 1];
    expected[1] = [8, 9];
    expected[3] = [21, 22];
    assert_eq!(table, expected);
}

#[test]
fn a_verification_continues_each_slot_from_the_token_its_acceptance_ends_on() {
    // Slot 2 verifies 6 and 7 after its committed token 5, its request's tokens before 5
    // being 3 4; slot 4 verifies 9 after 8, its tokens before being 1 2.
    let batch = verification(0.0);
    let bucket = Bucket::verify(4, DRAFTS, Variant::VerifyGreedy);
    let (layout, mut block) = speculating_block();
    speculation::fill_verification(
        &layout,
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        &speculation::token_rows(&batch.columns),
        &bucket,
        &[],
        &mut block,
    )
    .unwrap();

    let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
    let column = |column, elements| u32_column(&read::<u32>(&layout, &block, column, elements));
    let mut table = context_table(&[(2, [3, 4]), (4, [1, 2])]);

    // Each token is a row of its own, which reads the two ids before it in its request.
    let ids = column(Column::InputIds, tokens);
    let gathered = gather(&table, &column(Column::SlotRows, tokens));
    let index = column(Column::TokenContext, 2 * tokens)
        .reshape((tokens, 2))
        .unwrap();
    let before = super::history_ids(&gathered, &ids, &index).unwrap();

    let mut expected = vec![[3, 4], [4, 5], [5, 6], [1, 2], [2, 8]];
    expected.resize(tokens, [0, 0]);
    assert_eq!(pairs(&before), expected);

    // Slot 2's samples agree with its first draft and not its second, and slot 4's with
    // none: each slot continues after the token its acceptance ends on, 6 and 8.
    let mut chosen = vec![6_u32, 40, 41, 30, 31];
    chosen.resize(tokens, 0);
    let (agreed, last) = speculation::accept(
        &u32_column(&chosen),
        &column(Column::Drafts, tokens),
        &column(Column::Windows, rows * (DRAFTS as usize + 1))
            .reshape((rows, DRAFTS as usize + 1))
            .unwrap(),
        &column(Column::Firsts, rows),
    )
    .unwrap();
    assert_eq!(agreed.to_vec1::<u32>().unwrap()[..2], [1, 0]);

    let next = speculation::continued(&before, &ids, &last).unwrap();
    scatter(&mut table, &column(Column::StateSlots, rows), &next);

    let mut expected = vec![[0, 0]; SLOTS as usize + 1];
    expected[1] = [5, 6];
    expected[3] = [2, 8];
    assert_eq!(table, expected);
}

#[test]
fn a_request_s_verify_slots_are_its_home_row_and_its_rows_past_the_home_rows() {
    let owned = DRAFTS + 1;
    let none = NO_SLOT;

    // The slots of every request slot are the rows of a live buffer of `slots · Q` rows, no
    // two requests sharing one, and each request's home row comes first.
    let homes: Vec<u32> = (0..SLOTS).collect();
    let mut indices = columns::state_indices(&homes, SLOTS, owned);
    assert_eq!(indices[3..6], [1, 8, 9]);
    indices.sort_unstable();
    assert_eq!(indices, (0..SLOTS * owned).collect::<Vec<_>>());
    assert_eq!(columns::state_indices(&[none], SLOTS, owned), [none; 3]);

    // A prompt's rows state the slots of their requests, and a padding row none.
    let (layout, mut block) = speculating_block();
    let u32s = |block: &[u8], column, elements| read::<u32>(&layout, block, column, elements);
    let mut expected = vec![1, 8, 9, 3, 12, 13];
    expected.resize(4 * 3, none);

    let batch = prefill(Variant::PrefillGreedy);
    let bucket = Bucket::prefill(4, 8, Variant::PrefillGreedy);
    columns::fill(
        &layout,
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        &bucket,
        &[],
        None,
        &mut block,
    )
    .unwrap();
    assert_eq!(u32s(&block, Column::StateIndices, 4 * 3), expected);

    // A verification's state rows are its requests, whose tokens are its rows.
    let (_, mut block) = speculating_block();
    let batch = verification(0.0);
    speculation::fill_verification(
        &layout,
        SLOTS,
        VOCABULARY as u32,
        &batch.columns,
        &speculation::token_rows(&batch.columns),
        &Bucket::verify(4, DRAFTS, Variant::VerifyGreedy),
        &[],
        &mut block,
    )
    .unwrap();
    assert_eq!(u32s(&block, Column::StateIndices, 4 * 3), expected);
}

#[test]
fn a_runner_over_stepwise_state_lends_verify_slots_to_every_forward_and_keeps_no_trail() {
    // The stepwise test model, which reads token ids, over the rank's cache of its two runs
    // and its drafter's group.
    let model = Tiny {
        stepwise: true,
        reads: true,
        ..Tiny::drafting(true)
    };
    let runs = model.state_layers();
    let model = Arc::new(model);

    let layout = CacheLayout::of(PAGE, &fields(), proto::request::PrefixCacheDtype::BFloat16)
        .and_then(|layout| layout.with_state(&runs))
        .and_then(|layout| layout.with_draft(&fields()))
        .unwrap();
    let cache_config = KvCacheConfig {
        layout: Arc::new(layout),
        ..cache_config(true)
    };
    let cache = KvCacheMemory::meta(cache_config.clone()).unwrap();
    let config = CausalLmConfig {
        drafts: DRAFTS,
        draft_pool: Some(DraftPool::of(&cache_config.layout, cache_config.units).unwrap()),
        ..config()
    };

    let state = |owned: u32| {
        StateMemory::meta(
            StateConfig {
                shapes: runs.clone(),
                slots: SLOTS,
                rows: 4,
                trail: 0,
                owned,
                drafter: Vec::new(),
            },
            &cache_config,
        )
        .unwrap()
    };
    let over = |model: Arc<Tiny>, config: CausalLmConfig, state: &StateMemory| {
        let graphs = Graphs::meta(
            Arc::new(Generation::new(0, [])),
            CausalLmRunner::input_bytes(&config, &*model)?,
        );
        CausalLmRunner::over(model, config, graphs, None, &cache, Some(state))
    };

    // A request owns a slot of the stepwise field per token of a verification: its live
    // buffer holds them past the home rows, and the runner is lent no trail. The model
    // refuses a forward without its verify slots or its token ids: every bucket, prompts',
    // verifications' and the drafter's, is profiled and captured.
    let owned = DRAFTS + 1;
    assert_eq!(
        state(owned).live().unwrap()[1][0].dims(),
        [(SLOTS * owned) as usize, HIDDEN]
    );
    let runner = over(Arc::clone(&model), config.clone(), &state(owned)).unwrap();
    runner.graphs().profile_run(&runner).unwrap();
    for bucket in runner.buckets() {
        crate::runner::graphs::tests::warm(&runner, &bucket).unwrap();
    }

    // A live buffer of one slot a request holds none of a verification's further slots.
    let refused = over(Arc::clone(&model), config.clone(), &state(1))
        .err()
        .unwrap();
    assert!(refused.to_string().contains("address 18 rows"), "{refused}");

    // A model loaded for a deployment that drafts, under a config that drafts nothing.
    let unspeculating = CausalLmConfig {
        drafts: 0,
        draft_pool: None,
        ..config
    };
    let refused = over(model, unspeculating, &state(1)).err().unwrap();
    assert!(refused.to_string().contains("no drafts"), "{refused}");
}
