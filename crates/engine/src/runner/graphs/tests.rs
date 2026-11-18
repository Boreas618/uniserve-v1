//! `Graphs` without a device: profile runs, captures that record a meta forward, the inputs a
//! replay prepares, bucket choice and the generation state machine with its switch bucket by
//! bucket, over a small causal language model of `nn` modules on meta weights.

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::batch::{Batch, BatchId, BatchOutput, Columns, Phase, Shape as BatchShape, Variant};
use nn::functional::{WeightOrder, embedding, rms_norm};
use nn::modules::Linear;
use nn::{CausalLm, ExpertExchange, KvCache, RecurrentState, TokenBatch};
use ops::fill;
use ops::norm::{RmsNorm, RmsNormArgs, RmsNormDescriptor};
use ops::{Generation, KernelEntry};
use tensor::{DType, Device, Tensor};

use storage::runtime::GroupId;

use super::{Graphs, Plan, Prepared};
use crate::memory::tables::BlockTables;
use crate::runner::buckets::Bucket;
use crate::runner::{Forward, Runner};
use crate::worker::{Error, Result};
use testkit::serving::unlaunched_kernel;

const VOCABULARY: usize = 32;
const HIDDEN: usize = 16;
/// The rows of the largest bucket.
const ROWS: usize = 4;

/// The bytes of one column of the runner's input tensors, and the offsets of its five columns.
const COLUMN: u64 = 256;
const IDS: u64 = 0;
const POSITIONS: u64 = COLUMN;
const SLOTS: u64 = 2 * COLUMN;
const LENGTHS: u64 = 3 * COLUMN;
const OFFSETS: u64 = 4 * COLUMN;
const INPUTS: u64 = 5 * COLUMN;

/// A meta weight: a shape and an element type at an address nothing reads.
fn weight(address: u64, dims: &[usize]) -> Tensor {
    ops::retained(address, dims, DType::BF16, &ops::meta()).unwrap()
}

/// A causal language model of three `nn` operators and one host value.
struct Tiny {
    table: Tensor,
    gain: Tensor,
    head: Linear,
    /// Whether the decoder normalizes twice: another launch table for the same bucket.
    deep: AtomicBool,
}

impl Tiny {
    fn new() -> Self {
        Self {
            table: weight(0x1_0000, &[VOCABULARY, HIDDEN]),
            gain: weight(0x2_0000, &[HIDDEN]),
            head: Linear::new(
                weight(0x3_0000, &[VOCABULARY, HIDDEN]),
                None,
                WeightOrder::NxK,
            )
            .unwrap(),
            deep: AtomicBool::new(false),
        }
    }
}

impl CausalLm for Tiny {
    fn embed_input_ids(&self, ids: &Tensor) -> nn::Result<Tensor> {
        embedding(ids, &self.table, 0)
    }

    fn forward(
        &self,
        batch: &TokenBatch,
        _cache: &[KvCache],
        _state: Option<&RecurrentState>,
        _exchange: Option<&ExpertExchange>,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        nn::profiler::range("Decoder", "model.decoder", || {
            // A host value whose shape follows from the bucket and whose value is the step's:
            // one FP32 per token, moved to the device of the rows it meets.
            let bound = vec![batch.max_kv_len as f32; batch.tokens()];
            let bound = Tensor::from_vec(bound, (batch.tokens(), 1), &Device::Cpu)?
                .to_device(hidden.device())?
                .to_dtype(hidden.dtype())?;

            let shifted = hidden.broadcast_add(&bound)?;
            let normed = rms_norm(&shifted, &self.gain, 1e-6, false, hidden.dtype())?;
            if self.deep.load(Ordering::Relaxed) {
                return rms_norm(&normed, &self.gain, 1e-6, false, hidden.dtype());
            }
            Ok(normed)
        })
    }

    fn compute_logits(&self, hidden: &Tensor) -> nn::Result<Tensor> {
        self.head.forward(hidden)
    }
}

/// A runner of [`Tiny`]: decode buckets of four and of two rows over five input columns.
struct TinyRunner {
    model: Arc<Tiny>,
    graphs: Graphs,
    /// The key bound of the step being prepared: the plain value the model's host value holds.
    bound: Cell<u32>,
}

impl TinyRunner {
    fn new(kernels: Arc<Generation>) -> Self {
        Self {
            model: Arc::new(Tiny::new()),
            graphs: Graphs::meta(kernels, INPUTS),
            bound: Cell::new(16),
        }
    }

    /// The first `elements` elements of the input column at `offset`.
    fn column(&self, offset: u64, elements: usize, dtype: DType) -> Result<Tensor> {
        self.graphs.input(offset, elements, dtype)
    }
}

impl Forward for TinyRunner {
    fn buckets(&self) -> Vec<Bucket> {
        vec![
            Bucket::decode(ROWS as u32, Variant::DecodeGreedy),
            Bucket::decode(2, Variant::DecodeGreedy),
        ]
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let rows = bucket.rows as usize;
        let batch = TokenBatch {
            token_ids: None,
            token_history: None,
            positions: self.column(POSITIONS, rows, DType::I32)?,
            slot_mapping: self.column(SLOTS, rows, DType::U32)?,
            seq_lens: self.column(LENGTHS, rows, DType::U32)?,
            query_offsets: self.column(OFFSETS, rows + 1, DType::U32)?,
            logits_rows: self.column(IDS, rows, DType::U32)?,
            phase: Phase::Decode,
            page_size: 16,
            num_tokens: bucket.tokens,
            max_kv_len: self.bound.get(),
            draft_lens: None,
            rope_positions: None,
            live_tokens: None,
        };
        batch.check()?;

        let ids = self.column(IDS, rows, DType::U32)?;
        let hidden = self.model.embed_input_ids(&ids)?;
        let hidden = self.model.forward(&batch, &[], None, None, &hidden)?;
        Ok(vec![self.model.compute_logits(&hidden)?])
    }

    fn idle_inputs(&self, _bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        columns.fill(0);
        Ok(())
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

impl Runner for TinyRunner {
    fn phases(&self) -> &[Phase] {
        &[Phase::Decode]
    }

    fn capturer(&self, graphs: Graphs) -> Result<Box<dyn Forward>> {
        Ok(Box::new(Self {
            model: Arc::clone(&self.model),
            graphs,
            bound: Cell::new(16),
        }))
    }

    fn prepare_inputs(&mut self, batch: &Batch, _tables: &BlockTables) -> Result<Bucket> {
        let bucket = self
            .graphs
            .select(|bucket| bucket.holds(&batch.shape))
            .ok_or_else(|| Error::Invalid(format!("no bucket holds {}", batch.shape)))?;

        self.bound.set(batch.columns.max_kv_len);
        self.graphs.prepare_inputs(&*self, &bucket, |columns| {
            for (index, id) in batch.columns.input_ids.iter().enumerate() {
                let at = IDS as usize + 4 * index;
                columns[at..at + 4].copy_from_slice(&id.to_ne_bytes());
            }
            Ok(())
        })?;
        Ok(bucket)
    }

    fn execute(&mut self, _batch: &Batch, _tables: &BlockTables) -> Result<BatchOutput> {
        Err(Error::State("a runner with no device executes nothing"))
    }
}

/// A decode batch of `ids`, whose model reads keys up to `bound`.
fn batch(ids: &[i32], bound: u32) -> Batch {
    Batch {
        id: BatchId(1),
        group: GroupId(0),
        phase: Phase::Decode,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: BatchShape::decode(ids.len() as u32, Variant::DecodeGreedy),
        requests: Vec::new(),
        columns: Columns {
            input_ids: ids.to_vec(),
            max_kv_len: bound,
            ..Columns::default()
        },
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

/// The device block table of a rank whose requests hold nothing.
fn tables() -> BlockTables {
    BlockTables::new(GroupId(0), &[], &[])
}

/// Generation `number` holding one kernel for every key a profile run of the runner states.
fn generation_of(runner: &TinyRunner, number: u64) -> Arc<Generation> {
    let kernel = unlaunched_kernel();

    Arc::new(Generation::new(
        number,
        runner
            .graphs
            .keys()
            .into_iter()
            .map(|(operator, key)| KernelEntry::keyed(operator, key, Arc::clone(&kernel))),
    ))
}

/// The generation after `serving` in which every kernel of `operator` is another one.
fn replacing(serving: &Generation, number: u64, operator: &'static str) -> Arc<Generation> {
    let kernel = unlaunched_kernel();
    let keys: Vec<String> = serving.keys(operator).map(str::to_owned).collect();

    Arc::new(
        serving.replace(
            number,
            keys.into_iter()
                .map(|key| KernelEntry::keyed(operator, key, Arc::clone(&kernel))),
        ),
    )
}

/// Warms `bucket` of `runner` up as its worker's capture thread does: a capture instance of its
/// forward captures it under the serving generation, and the graph is installed.
pub(crate) fn warm(runner: &dyn Runner, bucket: &Bucket) -> Result<()> {
    let kernels = runner.graphs().kernels();
    let instance = runner.capturer(runner.graphs().beside()?)?;
    let graph = instance.graphs().capture(&*instance, bucket, &kernels)?;

    runner
        .graphs()
        .install(bucket, kernels.number(), Some(graph))
}

/// What the worker's capture thread does for `runner`'s buckets of generation `kernels` beside
/// `serving` (`None` at warm-up): each is examined over a capture instance of the runner's
/// forward, captured when it needs it, and installed in the runner's graphs. Returns each
/// bucket's plan.
fn captured_beside(
    runner: &TinyRunner,
    buckets: &[Bucket],
    serving: Option<&Generation>,
    kernels: &Arc<Generation>,
) -> Vec<Plan> {
    let instance = runner.capturer(runner.graphs.beside().unwrap()).unwrap();

    let mut plans = Vec::new();
    for bucket in buckets {
        let held = serving.and_then(|_| runner.graphs.launch_table(bucket));
        let plan = instance
            .graphs()
            .examine(&*instance, bucket, held.as_deref(), serving, kernels)
            .unwrap();
        let graph = plan.capture.then(|| {
            instance
                .graphs()
                .capture(&*instance, bucket, kernels)
                .unwrap()
        });

        runner
            .graphs
            .install(bucket, kernels.number(), graph)
            .unwrap();
        plans.push(plan);
    }
    plans
}

/// A runner serving generation 1 with both buckets captured, and that generation.
fn captured() -> (TinyRunner, Arc<Generation>) {
    // The keys come from a profile run, which needs no kernel: the first runner states them and
    // the second serves a generation that holds them.
    let profiled = TinyRunner::new(Arc::new(Generation::new(0, [])));
    profiled.graphs.profile_run(&profiled).unwrap();
    let serving = generation_of(&profiled, 1);

    let runner = TinyRunner::new(Arc::clone(&serving));
    runner.graphs.profile_run(&runner).unwrap();
    captured_beside(&runner, &runner.buckets(), None, &serving);
    (runner, serving)
}

/// Prepares `next` on `runner` and captures each of its buckets beside the serving graphs, the
/// way a worker and its capture thread do; returns the buckets in the order they were examined
/// and each one's plan.
fn prepared(
    runner: &TinyRunner,
    serving: &Generation,
    next: &Arc<Generation>,
) -> (Vec<Bucket>, Vec<Plan>) {
    let buckets = runner.graphs.prepare(runner, Arc::clone(next)).unwrap();
    let plans = captured_beside(runner, &buckets, Some(serving), next);
    (buckets, plans)
}

/// The generation of the graph a step of `bucket` stamped with kernel generation `stamp`
/// replays, as `prepare_inputs` chooses it.
fn replayed(runner: &mut TinyRunner, rows: usize, stamp: u64) -> (Bucket, Option<u64>) {
    runner.graphs.stamp(stamp);
    let ids: Vec<i32> = (1..=rows as i32).collect();
    let bucket = runner.prepare_inputs(&batch(&ids, 16), &tables()).unwrap();

    let state = runner.graphs.state.borrow();
    let Some(Prepared::Inputs(_, _, generation)) = &state.prepared else {
        panic!("inputs are prepared");
    };
    (bucket, *generation)
}

/// The host values the prepared replay's input copies read: the values area of its slot.
fn prepared_values(graphs: &Graphs, bytes: usize) -> Vec<u8> {
    let mut state = graphs.state.borrow_mut();
    let Some(Prepared::Inputs(_, slot, _)) = state.prepared else {
        panic!("inputs are prepared");
    };
    let (base, length) = state.buffer.as_mut().unwrap().values_ptr(slot);
    assert!(bytes as u64 <= length);

    // SAFETY: the area is `length` bytes of the ring's host memory, which the borrow keeps.
    unsafe { std::slice::from_raw_parts(base, bytes) }.to_vec()
}

fn operators(table: &[(&'static str, String, String)]) -> Vec<&'static str> {
    table.iter().map(|(operator, _, _)| *operator).collect()
}

#[test]
fn a_profile_run_states_keys_the_region_and_the_input_copy_bytes() {
    let runner = TinyRunner::new(Arc::new(Generation::new(0, [])));
    runner.graphs.profile_run(&runner).unwrap();

    let large = Bucket::decode(ROWS as u32, Variant::DecodeGreedy);
    let table = runner.graphs.launch_table(&large).unwrap();

    // The model's operators in the order it launches them: the embedding, the host value's
    // cast, its addition, the normalization and the head's product.
    let launched = operators(&table);
    assert_eq!(launched.first(), Some(&"embedding"));
    assert_eq!(launched.last(), Some(&"gemm"));
    assert_eq!(
        launched[1..launched.len() - 1],
        ["cast", "elementwise_binary", "rms_norm"]
    );

    // A launch states the profiler range it was issued in.
    let (_, _, path) = &table[3];
    assert_eq!(path, "model.decoder");
    assert_eq!(table[0].2, "");

    // Two buckets of one model launch the same operators under keys of their own or shared.
    let keys = runner.graphs.keys();
    assert!(keys.iter().any(|(operator, _)| *operator == "rms_norm"));
    assert!(keys.iter().all(|(_, key)| !key.is_empty()));

    // The region holds the input tensors and what the largest forward allocates: at least the
    // embedding's rows and the logits.
    let activations = (ROWS * HIDDEN * 2 + ROWS * VOCABULARY * 2) as u64;
    assert!(runner.graphs.peak() >= INPUTS + activations);

    // One FP32 host value per token of the largest bucket, at whatever offset its alignment
    // gives it.
    assert_eq!(
        runner.graphs.input_peak(),
        4 * ROWS as u64 + ops::allocator::ALIGNMENT - 1
    );
}

/// A runner whose forward writes outputs of known sizes with two operators: `norms`
/// normalizations of its rows of 64 BF16 elements, then one fill of a row of 512 FP32 elements
/// each, in decode buckets of four and two rows.
struct Writes {
    graphs: Graphs,
    weight: Tensor,
    norms: usize,
}

impl Writes {
    fn new(norms: usize) -> Self {
        Self {
            graphs: Graphs::meta(Arc::new(Generation::new(0, [])), 4 * 64 * 2),
            weight: weight(0x4_0000, &[64]),
            norms,
        }
    }
}

impl Forward for Writes {
    fn buckets(&self) -> Vec<Bucket> {
        vec![
            Bucket::decode(4, Variant::DecodeGreedy),
            Bucket::decode(2, Variant::DecodeGreedy),
        ]
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let rows = bucket.rows as usize;
        let rows_in = self.graphs.input(0, (rows, 64), DType::BF16)?;
        let mut outputs = Vec::new();

        let norm = RmsNormDescriptor {
            dtype: DType::BF16,
            columns: 64,
            tile: 64,
            pdl: false,
            centred: false,
            groups: 1,
        };
        for _ in 0..self.norms {
            let out = ops::empty((rows, 64), DType::BF16)?;
            let args = RmsNormArgs {
                x: ops::operand(&rows_in)?,
                weight: ops::operand(&self.weight)?,
                out: ops::operand(&out)?,
                eps: 1e-6,
            };
            ops::launch::<RmsNorm>(&norm, args)?;
            outputs.push(out);
        }

        let filled = ops::empty((rows, 512), DType::F32)?;
        let args = fill::Args {
            destination: ops::operand(&filled)?,
            bits: 0,
        };
        ops::launch::<fill::Fill>(&fill::Descriptor::of(&args), args)?;
        outputs.push(filled);
        Ok(outputs)
    }

    fn idle_inputs(&self, _bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        columns.fill(0);
        Ok(())
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

impl Runner for Writes {
    fn phases(&self) -> &[Phase] {
        &[Phase::Decode]
    }

    fn capturer(&self, graphs: Graphs) -> Result<Box<dyn Forward>> {
        Ok(Box::new(Self {
            graphs,
            weight: self.weight.clone(),
            norms: self.norms,
        }))
    }

    fn prepare_inputs(&mut self, _batch: &Batch, _tables: &BlockTables) -> Result<Bucket> {
        Err(Error::State("the runner only profiles"))
    }

    fn execute(&mut self, _batch: &Batch, _tables: &BlockTables) -> Result<BatchOutput> {
        Err(Error::State("the runner only profiles"))
    }
}

#[test]
fn a_forward_s_shadow_region_holds_its_largest_call_s_twins() {
    // Four rows' fill writes 8192 bytes and each of their normalizations 512: the region holds
    // the fill's twin, in the bucket of four rows, not everything the forward writes.
    let writes = Writes::new(2);
    writes.graphs.profile_run(&writes).unwrap();
    assert_eq!(writes.graphs.shadow_peak(0), 8192);

    // A paired forward compares each call as both kernels ran it, so twenty normalizations of
    // four rows, 10240 bytes in all, take the region no more than one of them does.
    let writes = Writes::new(20);
    writes.graphs.profile_run(&writes).unwrap();
    assert_eq!(writes.graphs.shadow_peak(0), 8192);

    // What a paired forward's cache writes write in place beyond the profiled forward's is
    // held beside the largest call.
    assert_eq!(writes.graphs.shadow_peak(12_000), 8192 + 12_000);
}

#[test]
fn a_capture_records_the_profile_run_s_launches_and_the_forward_s_input_copies() {
    let (runner, _) = captured();

    for (bucket, bytes) in [
        (Bucket::decode(4, Variant::DecodeGreedy), 16),
        (Bucket::decode(2, Variant::DecodeGreedy), 8),
    ] {
        let copies = runner.graphs.input_copies(&bucket).unwrap();
        assert_eq!(copies.len(), 1);

        // The one host value is the first copy of the forward: offset zero of the values.
        let (offset, _, length) = copies[0];
        assert_eq!((offset, length), (0, bytes));
    }

    assert_eq!(
        runner.graphs.buckets(),
        [
            Bucket::decode(2, Variant::DecodeGreedy),
            Bucket::decode(4, Variant::DecodeGreedy)
        ]
    );
}

#[test]
fn a_capture_whose_launches_differ_from_the_profile_run_is_refused() {
    let runner = TinyRunner::new(Arc::new(Generation::new(0, [])));
    runner.graphs.profile_run(&runner).unwrap();

    // The forward the capture runs is not the one the profile run stated keys for.
    let instance = runner.capturer(runner.graphs.beside().unwrap()).unwrap();
    let bucket = Bucket::decode(2, Variant::DecodeGreedy);
    let kernels = generation_of(&runner, 1);
    instance
        .graphs()
        .examine(&*instance, &bucket, None, None, &kernels)
        .unwrap();
    runner.model.deep.store(true, Ordering::Relaxed);
    let Err(refused) = instance.graphs().capture(&*instance, &bucket, &kernels) else {
        panic!("a capture whose launches differ from the profile run's is refused");
    };

    assert!(matches!(refused, Error::Invariant(_)), "{refused}");
}

#[test]
fn prepare_inputs_writes_each_replay_s_host_values_where_the_input_copies_read() {
    let (mut runner, _) = captured();
    let bucket = Bucket::decode(4, Variant::DecodeGreedy);
    let (offset, _, bytes) = runner.graphs.input_copies(&bucket).unwrap()[0];
    assert_eq!(offset, 0);

    for bound in [32_u32, 48] {
        let chosen = runner
            .prepare_inputs(&batch(&[5, 6, 7], bound), &tables())
            .unwrap();
        assert_eq!(chosen, bucket);

        // The forward's host value for this replay: the step's bound, once per token of the
        // bucket, at the offset the capture's input copy reads.
        let expected: Vec<u8> = std::iter::repeat_n(bound as f32, ROWS)
            .flat_map(f32::to_ne_bytes)
            .collect();
        assert_eq!(prepared_values(&runner.graphs, bytes as usize), expected);
    }
}

#[test]
fn a_step_runs_the_smallest_bucket_that_holds_it() {
    let (mut runner, _) = captured();

    let small = runner
        .prepare_inputs(&batch(&[1, 2], 16), &tables())
        .unwrap();
    assert_eq!(small, Bucket::decode(2, Variant::DecodeGreedy));

    let large = runner
        .prepare_inputs(&batch(&[1, 2, 3], 16), &tables())
        .unwrap();
    assert_eq!(large, Bucket::decode(4, Variant::DecodeGreedy));

    // A batch no bucket holds is an error, and so is another computation's.
    assert!(
        runner
            .prepare_inputs(&batch(&[1, 2, 3, 4, 5], 16), &tables())
            .is_err()
    );
    let sampled = BatchShape::decode(2, Variant::DecodeSampled);
    assert_eq!(runner.graphs.select(|bucket| bucket.holds(&sampled)), None);
}

#[test]
fn graphs_with_no_device_replay_nothing() {
    let (mut runner, _) = captured();
    let bucket = runner
        .prepare_inputs(&batch(&[1, 2], 16), &tables())
        .unwrap();

    assert!(matches!(
        runner.graphs.replay(&bucket),
        Err(Error::State(_))
    ));
    assert!(runner.graphs.enter().is_err());
}

/// A batch the runner could not pair compares nothing: its paired forward reports no element
/// count, and the update names the operator uncompared with the runner's reason.
#[test]
fn an_unpaired_batch_is_reported_uncompared_and_not_compared() {
    let (runner, serving) = captured();
    runner
        .graphs
        .prepare(&runner, replacing(&serving, 2, "rms_norm"))
        .unwrap();

    let bucket = runner.buckets()[0];
    runner
        .graphs
        .unpaired(&bucket, "its rows hold no scratch page".to_owned());
    let compared = runner.graphs.shadow(&runner, &bucket, "rms_norm").unwrap();

    assert_eq!(compared, None);
    let uncompared = runner.graphs.uncompared();
    assert_eq!(uncompared.len(), 1, "{uncompared:?}");
    assert!(uncompared[0].contains("`rms_norm`"), "{uncompared:?}");
    assert!(
        uncompared[0].contains("its rows hold no scratch page"),
        "{uncompared:?}"
    );
}

#[test]
fn a_changed_kernel_is_captured_again_beside_the_serving_graph_and_rehearsed_first() {
    let (runner, serving) = captured();
    let next = replacing(&serving, 2, "rms_norm");

    // Both buckets launch the normalization, whose kernel is another: each is captured again,
    // and rehearsed first, since the new kernel is a native entry nothing has used.
    let (buckets, plans) = prepared(&runner, &serving, &next);
    assert_eq!(buckets.len(), 2);
    assert!(plans.iter().all(|plan| plan.capture && plan.rehearse));

    // The serving generation serves whole until the switch, and the replaced graphs stay
    // beside the new ones until the commit.
    assert_eq!(runner.graphs.generation(), 1);
    runner.graphs.switch(2).unwrap();
    assert_eq!(runner.graphs.generation(), 2);
    assert_eq!(runner.graphs.state.borrow().graphs.len(), 4);

    runner.graphs.commit(2).unwrap();
    assert_eq!(runner.graphs.state.borrow().graphs.len(), 2);
    assert_eq!(runner.graphs.buckets().len(), 2);
}

#[test]
fn a_generation_that_changes_no_launched_kernel_keeps_every_graph() {
    let (runner, serving) = captured();

    // Another kernel for an operator the runner never launches.
    let next = Arc::new(serving.replace(
        2,
        [KernelEntry::keyed(
            "softmax",
            "dtype_f32".to_owned(),
            unlaunched_kernel(),
        )],
    ));

    let (_, plans) = prepared(&runner, &serving, &next);
    assert!(plans.iter().all(|plan| !plan.capture && !plan.rehearse));
    runner.graphs.switch(2).unwrap();
    assert_eq!(runner.graphs.generation(), 2);

    // The buckets serve from the graphs they had.
    let shape = BatchShape::decode(2, Variant::DecodeGreedy);
    assert_eq!(
        runner.graphs.select(|bucket| bucket.holds(&shape)),
        Some(Bucket::decode(2, Variant::DecodeGreedy))
    );
    assert_eq!(runner.graphs.state.borrow().graphs.len(), 2);
    runner.graphs.commit(2).unwrap();
}

#[test]
fn a_changed_launch_table_is_captured_again() {
    let (runner, serving) = captured();
    let next = Arc::new(serving.replace(2, []));
    let bucket = Bucket::decode(4, Variant::DecodeGreedy);

    // The forward itself launches differently for the new generation, under the same kernels:
    // its native kernels run at shapes no serving graph ran, so the bucket is rehearsed first.
    runner.model.deep.store(true, Ordering::Relaxed);
    let (_, plans) = prepared(&runner, &serving, &next);
    assert!(plans.iter().all(|plan| plan.capture && plan.rehearse));

    // The serving graph is untouched until the switch, and the new one serves after it.
    let before = runner.graphs.launch_table(&bucket).unwrap();
    runner.graphs.switch(2).unwrap();
    let after = runner.graphs.launch_table(&bucket).unwrap();
    assert_eq!(after.len(), before.len() + 1);
}

/// A bucket switches to the prepared generation ahead of the others once its capture is
/// installed, and a step replays the newest graph at or below both what its bucket switched to
/// and its batch's kernel stamp: every rank, stamping and switching alike, replays a batch with
/// graphs of one generation.
#[test]
fn a_bucket_switches_ahead_of_the_others_within_its_batches_stamp() {
    let (mut runner, serving) = captured();
    let next = replacing(&serving, 2, "rms_norm");
    let (large, small) = (
        Bucket::decode(4, Variant::DecodeGreedy),
        Bucket::decode(2, Variant::DecodeGreedy),
    );

    // Nothing switches before its capture is installed.
    let buckets = runner.graphs.prepare(&runner, Arc::clone(&next)).unwrap();
    assert!(runner.graphs.switch_buckets(2, &[large]).is_err());
    captured_beside(&runner, &buckets, Some(&serving), &next);

    // The large bucket switches; the small one serves its kept graph.
    runner.graphs.switch_buckets(2, &[large]).unwrap();
    assert_eq!(replayed(&mut runner, 3, 2), (large, Some(2)));
    assert_eq!(replayed(&mut runner, 2, 2), (small, Some(1)));

    // A batch stamped with the kept generation replays the kept graph of a switched bucket.
    assert_eq!(replayed(&mut runner, 3, 1), (large, Some(1)));

    // Whole, every bucket replays the new generation's graph.
    runner.graphs.switch(2).unwrap();
    assert_eq!(replayed(&mut runner, 2, 2), (small, Some(2)));
}

/// A rollback while buckets serve from the prepared generation returns every bucket to its kept
/// graph from the next step on, whatever the step's stamp, and releases the rolled-back graphs.
#[test]
fn after_a_rollback_no_step_replays_the_rolled_back_generation() {
    let (mut runner, serving) = captured();
    let next = replacing(&serving, 2, "rms_norm");
    let (buckets, _) = prepared(&runner, &serving, &next);
    runner.graphs.switch_buckets(2, &buckets).unwrap();

    runner.graphs.rollback(2).unwrap();
    for rows in [2, 3] {
        for stamp in [1, 2] {
            assert_eq!(replayed(&mut runner, rows, stamp).1, Some(1));
        }
    }
    assert!(
        runner
            .graphs
            .state
            .borrow()
            .graphs
            .keys()
            .all(|(_, generation)| *generation == 1)
    );
}

#[test]
fn a_rollback_on_trial_serves_the_kept_generation_again() {
    let (runner, serving) = captured();
    let bucket = Bucket::decode(4, Variant::DecodeGreedy);
    let kept = runner.graphs.launch_table(&bucket).unwrap();

    runner.model.deep.store(true, Ordering::Relaxed);
    let next = replacing(&serving, 2, "rms_norm");
    prepared(&runner, &serving, &next);
    runner.graphs.switch(2).unwrap();
    assert_ne!(runner.graphs.launch_table(&bucket).unwrap(), kept);

    // A commit names the generation on trial, and so does a rollback.
    assert!(runner.graphs.commit(1).is_err());
    assert!(runner.graphs.rollback(1).is_err());

    runner.graphs.rollback(2).unwrap();
    assert_eq!(runner.graphs.generation(), 1);
    assert_eq!(runner.graphs.launch_table(&bucket).unwrap(), kept);

    // Nothing of the rolled-back generation is left to commit or roll back.
    assert!(runner.graphs.commit(2).is_err());
    assert!(runner.graphs.rollback(2).is_err());
}

#[test]
fn a_prepared_generation_is_rolled_back_without_touching_the_serving_one() {
    let (runner, serving) = captured();
    let next = replacing(&serving, 2, "rms_norm");
    let buckets = runner.graphs.prepare(&runner, Arc::clone(&next)).unwrap();
    captured_beside(&runner, &buckets[..1], Some(&serving), &next);

    // One update at a time, and no switch while a bucket's capture is not installed.
    assert!(runner.graphs.prepare(&runner, Arc::clone(&next)).is_err());
    assert_eq!(runner.graphs.pending(), buckets[1..]);
    assert!(runner.graphs.switch(2).is_err());

    runner.graphs.rollback(2).unwrap();
    assert_eq!(runner.graphs.generation(), 1);
    assert!(runner.graphs.pending().is_empty());
    assert!(runner.graphs.switch(2).is_err());

    // The next update starts from the serving generation.
    let again = replacing(&serving, 3, "rms_norm");
    assert_eq!(runner.graphs.prepare(&runner, again).unwrap(), buckets);
}

#[test]
fn a_generation_missing_a_launched_kernel_is_refused() {
    let (runner, serving) = captured();
    let next = Arc::new(Generation::new(2, []));
    let buckets = runner.graphs.prepare(&runner, Arc::clone(&next)).unwrap();

    // The bucket it is examined at is refused, and the generation cannot be switched to.
    let instance = runner.capturer(runner.graphs.beside().unwrap()).unwrap();
    let held = runner.graphs.launch_table(&buckets[0]);
    let refused = instance
        .graphs()
        .examine(
            &*instance,
            &buckets[0],
            held.as_deref(),
            Some(&serving),
            &next,
        )
        .unwrap_err();
    assert!(matches!(refused, Error::Interface(_)), "{refused}");
    assert!(runner.graphs.switch(2).is_err());
}

/// The buckets a generation examines come most replayed first: what its switch reaches first.
#[test]
fn a_generation_examines_the_most_replayed_bucket_first() {
    let (runner, serving) = captured();
    runner
        .graphs
        .state
        .borrow_mut()
        .uses
        .insert(Bucket::decode(2, Variant::DecodeGreedy), 7);

    let order = runner
        .graphs
        .prepare(&runner, replacing(&serving, 2, "rms_norm"))
        .unwrap();
    assert_eq!(
        order,
        [
            Bucket::decode(2, Variant::DecodeGreedy),
            Bucket::decode(4, Variant::DecodeGreedy)
        ]
    );
}

#[test]
fn an_input_tensor_lies_inside_the_runner_s_input_block() {
    let runner = TinyRunner::new(Arc::new(Generation::new(0, [])));

    assert!(runner.graphs.input(IDS, ROWS, DType::U32).is_ok());
    // Unaligned, and past the block.
    assert!(runner.graphs.input(4, ROWS, DType::U32).is_err());
    assert!(runner.graphs.input(OFFSETS, 65, DType::U32).is_err());
}
