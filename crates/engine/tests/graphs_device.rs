//! `Graphs` on a device: a capture launches what its profile run stated and a replay computes
//! what the eager forward computes; a replay's columns and host values reach the device through
//! the input buffer; a generation is rebound, shadowed, switched to and rolled back.
//!
//! Every case needs GPU 0; the cases that launch kernels also need a running Build Server
//! (`UNISERVE_BUILD_SERVER_SOCKET`). Run them on a GPU host:
//!
//! ```text
//! cargo test -p engine --test graphs_device -- --ignored --test-threads 1
//! ```

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

mod device;

use std::cell::Cell;
use std::sync::{Arc, Mutex};

use engine::batch::{Batch, BatchId, BatchOutput, Columns, Phase, Shape as BatchShape, Variant};
use engine::memory::tables::BlockTables;
use engine::runner::Runner;
use engine::runner::buckets::Bucket;
use engine::runner::graphs::Graphs;
use engine::runner::graphs::Preparation;
use engine::worker::{Error, Result};
use nn::functional::{WeightOrder, embedding, rms_norm};
use nn::modules::Linear;
use nn::{CausalLm, ExpertExchange, KvCache, RecurrentState, TokenBatch};
use ops::Generation;
use platform::{Budget, DeviceAllocation, DeviceId, DeviceRange, PoolKind, Stream};
use storage::runtime::GroupId;
use tensor::{DType, Device, Tensor};

use device::{DEVICE, Kernels, Outcome};

const VOCABULARY: usize = 32;
const HIDDEN: usize = 16;
const ROWS: usize = 4;

/// The bytes of one column of the runner's input tensors, and the offsets of its columns.
const COLUMN: u64 = 256;
const IDS: u64 = 0;
const POSITIONS: u64 = COLUMN;
const SLOTS: u64 = 2 * COLUMN;
const LENGTHS: u64 = 3 * COLUMN;
const OFFSETS: u64 = 4 * COLUMN;
const INPUTS: u64 = 5 * COLUMN;

/// The table's row `id` holds `id + column / HIDDEN`, the gain is one half everywhere, and the
/// head's row `token` holds `(token % 5) / 4`: values a host computation reproduces exactly.
fn table_values() -> Vec<f32> {
    (0..VOCABULARY * HIDDEN)
        .map(|index| (index / HIDDEN) as f32 + (index % HIDDEN) as f32 / HIDDEN as f32)
        .collect()
}

fn head_values() -> Vec<f32> {
    (0..VOCABULARY * HIDDEN)
        .map(|index| ((index / HIDDEN) % 5) as f32 / 4.0)
        .collect()
}

/// A causal language model of three `nn` operators and one host value.
struct Tiny {
    table: Tensor,
    gain: Tensor,
    head: Linear,
    /// The device memory the weights are over, when they are on a device.
    _memory: Vec<DeviceAllocation>,
}

impl Tiny {
    /// The model over meta weights: what a profile run loads.
    fn meta() -> Self {
        let weight = |address: u64, dims: &[usize]| {
            ops::retained(address, dims, DType::F32, &ops::meta()).unwrap()
        };

        Self {
            table: weight(0x1_0000, &[VOCABULARY, HIDDEN]),
            gain: weight(0x2_0000, &[HIDDEN]),
            head: Linear::new(
                weight(0x3_0000, &[VOCABULARY, HIDDEN]),
                None,
                WeightOrder::NxK,
            )
            .unwrap(),
            _memory: Vec::new(),
        }
    }

    /// The model over weights on the device.
    fn on_device(stream: &Stream) -> Outcome<Self> {
        let (table_memory, table) = device::weight(stream, &table_values(), &[VOCABULARY, HIDDEN])?;
        let (gain_memory, gain) = device::weight(stream, &[0.5; HIDDEN], &[HIDDEN])?;
        let (head_memory, head) = device::weight(stream, &head_values(), &[VOCABULARY, HIDDEN])?;

        Ok(Self {
            table,
            gain,
            head: Linear::new(head, None, WeightOrder::NxK)?,
            _memory: vec![table_memory, gain_memory, head_memory],
        })
    }

    /// The model over host weights: the reference a device result is compared with.
    fn on_host() -> Outcome<Self> {
        let cpu = Device::Cpu;

        Ok(Self {
            table: Tensor::from_vec(table_values(), (VOCABULARY, HIDDEN), &cpu)?,
            gain: Tensor::from_vec(vec![0.5_f32; HIDDEN], HIDDEN, &cpu)?,
            head: Linear::new(
                Tensor::from_vec(head_values(), (VOCABULARY, HIDDEN), &cpu)?,
                None,
                WeightOrder::NxK,
            )?,
            _memory: Vec::new(),
        })
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
        // A host value whose shape follows from the bucket and whose value is the step's.
        let bound = vec![batch.max_kv_len as f32; batch.tokens()];
        let bound = Tensor::from_vec(bound, (batch.tokens(), 1), &Device::Cpu)?
            .to_device(hidden.device())?;

        let shifted = hidden.broadcast_add(&bound)?;
        rms_norm(&shifted, &self.gain, 1e-6, false, hidden.dtype())
    }

    fn compute_logits(&self, hidden: &Tensor) -> nn::Result<Tensor> {
        self.head.forward(hidden)
    }
}

/// The model's batch over five index columns of `rows` rows.
fn token_batch(columns: [Tensor; 5], rows: usize, bound: u32) -> TokenBatch {
    let [ids, positions, slot_mapping, seq_lens, query_offsets] = columns;

    TokenBatch {
        token_ids: None,
        ngram_context: None,
        positions,
        slot_mapping,
        seq_lens,
        query_offsets,
        logits_rows: ids,
        phase: Phase::Decode,
        page_size: 16,
        num_tokens: rows as u32,
        max_kv_len: bound,
        draft_lens: None,
        rope_positions: None,
        live_tokens: None,
    }
}

/// The logits of `ids` under key bound `bound`, computed on the host by the model's own
/// functions on CPU tensors.
fn host_logits(ids: &[u32], bound: u32) -> Outcome<Vec<Vec<f32>>> {
    let model = Tiny::on_host()?;
    let cpu = Device::Cpu;
    let rows = ids.len();
    let column = |dtype: DType, elements: usize| Tensor::zeros(elements, dtype, &cpu);

    let ids = Tensor::from_vec(ids.to_vec(), rows, &cpu)?;
    let batch = token_batch(
        [
            ids.clone(),
            column(DType::I32, rows)?,
            column(DType::U32, rows)?,
            column(DType::U32, rows)?,
            column(DType::U32, rows + 1)?,
        ],
        rows,
        bound,
    );

    let hidden = model.embed_input_ids(&ids)?;
    let hidden = model.forward(&batch, &[], None, None, &hidden)?;
    Ok(model.compute_logits(&hidden)?.to_vec2::<f32>()?)
}

/// A runner of [`Tiny`]: decode buckets of four and of two rows.
struct TinyRunner {
    model: Tiny,
    graphs: Graphs,
    /// The key bound of the step being prepared.
    bound: Cell<u32>,
}

impl TinyRunner {
    fn column(&self, offset: u64, elements: usize, dtype: DType) -> Result<Tensor> {
        self.graphs.input(offset, elements, dtype)
    }

    /// The logits the graphs' last step of `rows` rows left in `outputs`.
    fn read(&self, outputs: &[Tensor], rows: usize) -> Outcome<Vec<Vec<f32>>> {
        self.graphs.wait()?;
        let _guard = self.graphs.enter()?;

        let logits = outputs[0].to_vec2::<f32>()?;
        Ok(logits[..rows].to_vec())
    }
}

impl Runner for TinyRunner {
    fn phases(&self) -> &[Phase] {
        &[Phase::Decode]
    }

    fn buckets(&self) -> Vec<Bucket> {
        vec![
            Bucket::decode(ROWS as u32, Variant::DecodeGreedy),
            Bucket::decode(2, Variant::DecodeGreedy),
        ]
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let rows = bucket.rows as usize;
        let ids = self.column(IDS, rows, DType::U32)?;
        let batch = token_batch(
            [
                ids.clone(),
                self.column(POSITIONS, rows, DType::I32)?,
                self.column(SLOTS, rows, DType::U32)?,
                self.column(LENGTHS, rows, DType::U32)?,
                self.column(OFFSETS, rows + 1, DType::U32)?,
            ],
            rows,
            self.bound.get(),
        );
        batch.check()?;

        let hidden = self.model.embed_input_ids(&ids)?;
        let hidden = self.model.forward(&batch, &[], None, None, &hidden)?;
        Ok(vec![self.model.compute_logits(&hidden)?])
    }

    fn idle_inputs(&self, _bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        columns.fill(0);
        Ok(())
    }

    fn prepare_inputs(&mut self, batch: &Batch, _tables: &BlockTables) -> Result<Bucket> {
        let bucket = self
            .graphs
            .select(|bucket| bucket.holds(&batch.shape))
            .ok_or_else(|| Error::Invalid(format!("no bucket holds {}", batch.shape)))?;

        self.bound.set(batch.columns.max_kv_len);
        self.graphs.prepare_inputs(&*self, &bucket, |columns| {
            columns.fill(0);
            for (index, id) in batch.columns.input_ids.iter().enumerate() {
                let at = IDS as usize + 4 * index;
                columns[at..at + 4].copy_from_slice(&id.to_ne_bytes());
            }
            Ok(())
        })?;
        Ok(bucket)
    }

    fn execute(&mut self, _batch: &Batch, _tables: &BlockTables) -> Result<BatchOutput> {
        Err(Error::State("the cases drive the graphs themselves"))
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

/// A decode batch of `ids`, whose model reads keys up to `bound`.
fn batch(ids: &[u32], bound: u32) -> Batch {
    Batch {
        id: BatchId(1),
        group: GroupId(0),
        phase: Phase::Decode,
        generation: 1,
        shape: BatchShape::decode(ids.len() as u32, Variant::DecodeGreedy),
        requests: Vec::new(),
        columns: Columns {
            input_ids: ids.iter().map(|id| *id as i32).collect(),
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

/// A runner on the device with both buckets captured, the Build Server its kernels come from
/// and the region its graphs address.
struct Bench {
    runner: TinyRunner,
    kernels: Kernels,
    stream: Stream,
    _region: DeviceAllocation,
}

impl Bench {
    fn open() -> Outcome<Self> {
        // The profile run states the keys and the peaks before anything touches the device.
        let profiled = TinyRunner {
            model: Tiny::meta(),
            graphs: Graphs::meta(Arc::new(Generation::new(0, [])), INPUTS),
            bound: Cell::new(16),
        };
        profiled.graphs.profile_run(&profiled)?;

        let mut kernels = Kernels::open()?;
        let serving = kernels.generation(1, &profiled.graphs.keys())?;

        let stream = Stream::new(DEVICE)?;
        let region = DeviceAllocation::linear(DEVICE, profiled.graphs.peak() as usize)?;
        let range = DeviceRange::new(
            platform::DevicePtr::from_address(region.address()),
            region.len(),
        );

        let runner = TinyRunner {
            model: Tiny::on_device(&stream)?,
            graphs: Graphs::new(
                stream.clone(),
                Arc::clone(&serving),
                range,
                INPUTS,
                profiled.graphs.input_peak(),
                2,
            )?,
            bound: Cell::new(16),
        };
        runner.graphs.profile_run(&runner)?;
        for bucket in runner.buckets() {
            runner.graphs.capture(&runner, &bucket)?;
        }

        Ok(Self {
            runner,
            kernels,
            stream,
            _region: region,
        })
    }

    /// One replayed step of `ids` under `bound`: its logits.
    fn step(&mut self, ids: &[u32], bound: u32) -> Outcome<Vec<Vec<f32>>> {
        let bucket = self.runner.prepare_inputs(&batch(ids, bound), &tables())?;
        let outputs = self.runner.graphs.replay(&bucket)?;
        self.runner.read(&outputs, ids.len())
    }
}

/// Whether two logit matrices agree within the rounding of an FP32 normalization and product.
fn close(left: &[Vec<f32>], right: &[Vec<f32>]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.iter()
                .zip(right)
                .all(|(left, right)| (left - right).abs() <= 1e-4 * right.abs().max(1.0))
        })
}

/// A runner whose forward launches nothing: its outputs are a host value moved to the device
/// and one of its input columns.
struct Copies {
    graphs: Graphs,
    value: Cell<f32>,
}

impl Runner for Copies {
    fn phases(&self) -> &[Phase] {
        &[Phase::Decode]
    }

    fn buckets(&self) -> Vec<Bucket> {
        vec![Bucket::decode(ROWS as u32, Variant::DecodeGreedy)]
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let rows = bucket.rows as usize;
        let ids = self.graphs.input(IDS, rows, DType::U32)?;
        let value = Tensor::from_vec(vec![self.value.get(); rows], rows, &Device::Cpu)?
            .to_device(ids.device())?;

        Ok(vec![value, ids])
    }

    fn idle_inputs(&self, _bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        columns.fill(0);
        Ok(())
    }

    fn prepare_inputs(&mut self, batch: &Batch, _tables: &BlockTables) -> Result<Bucket> {
        let bucket = Bucket::decode(ROWS as u32, Variant::DecodeGreedy);

        self.value.set(batch.columns.max_kv_len as f32);
        self.graphs.prepare_inputs(&*self, &bucket, |columns| {
            for (index, id) in batch.columns.input_ids.iter().enumerate() {
                columns[4 * index..4 * index + 4].copy_from_slice(&id.to_ne_bytes());
            }
            Ok(())
        })?;
        Ok(bucket)
    }

    fn execute(&mut self, _batch: &Batch, _tables: &BlockTables) -> Result<BatchOutput> {
        Err(Error::State("the cases drive the graphs themselves"))
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

#[test]
#[ignore = "requires a GPU"]
fn a_replay_copies_its_columns_and_its_host_values_to_the_device() -> Outcome {
    let stream = Stream::new(DEVICE)?;
    let region = DeviceAllocation::linear(DEVICE, 1 << 20)?;
    let range = DeviceRange::new(
        platform::DevicePtr::from_address(region.address()),
        region.len(),
    );

    // Three slots and five replays: a slot is taken again once the stream passed its copies.
    let mut runner = Copies {
        graphs: Graphs::new(
            stream,
            Arc::new(Generation::new(1, [])),
            range,
            COLUMN,
            COLUMN,
            3,
        )?,
        value: Cell::new(0.0),
    };
    let bucket = runner.buckets()[0];
    runner.graphs.capture(&runner, &bucket)?;

    // The graph launches no kernel and has one input copy: the host value.
    assert!(runner.graphs.launch_table(&bucket).unwrap().is_empty());
    assert_eq!(runner.graphs.input_copies(&bucket).unwrap().len(), 1);

    for step in 0..5_u32 {
        let ids: Vec<u32> = (0..ROWS as u32).map(|row| 10 * step + row).collect();
        let prepared = runner.prepare_inputs(&batch(&ids, 100 + step), &tables())?;
        let outputs = runner.graphs.replay(&prepared)?;

        runner.graphs.wait()?;
        let _guard = runner.graphs.enter()?;
        assert_eq!(outputs[0].to_vec1::<f32>()?, [(100 + step) as f32; ROWS]);
        assert_eq!(outputs[1].to_vec1::<u32>()?, ids);
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_replay_computes_what_the_eager_forward_and_the_host_formula_compute() -> Outcome {
    let mut bench = Bench::open()?;

    // The capture launched the profile run's keys, or it would have been refused; the launch
    // tables of the two are the same launches.
    for bucket in bench.runner.buckets() {
        let captured = bench.runner.graphs.launch_table(&bucket).unwrap();
        assert!(!captured.is_empty());
    }

    for (ids, bound) in [(vec![3_u32, 7, 11, 2], 32_u32), (vec![5, 9], 48)] {
        let replayed = bench.step(&ids, bound)?;

        // The same step run eagerly, with no graph.
        let bucket = bench
            .runner
            .prepare_inputs(&batch(&ids, bound), &tables())?;
        let outputs = bench.runner.graphs.eager(&bench.runner, &bucket)?;
        let eager = bench.runner.read(&outputs, ids.len())?;
        assert_eq!(replayed, eager);

        assert!(close(&replayed, &host_logits(&ids, bound)?));
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_generation_is_rebound_switched_to_and_rolled_back() -> Outcome {
    let mut bench = Bench::open()?;
    let ids = [4_u32, 8, 15, 16];
    let served = bench.step(&ids, 24)?;

    // The same modules loaded again are other kernels of an unchanged launch table: a rebind
    // of every bucket.
    let keys = bench.runner.graphs.keys();
    let next = bench.kernels.generation(2, &keys)?;
    let buckets = bench.runner.graphs.prepare(&bench.runner, next)?;
    assert_eq!(buckets, bench.runner.buckets());
    for bucket in &buckets {
        let done = bench.runner.graphs.prepare_bucket(&bench.runner, bucket)?;
        assert_eq!(done, Preparation::Rebind);
    }

    // The serving generation's graphs serve until the switch, and the rebound ones after it.
    assert_eq!(bench.step(&ids, 24)?, served);
    bench.runner.graphs.switch(2)?;
    assert_eq!(bench.runner.graphs.generation(), 2);
    assert_eq!(bench.step(&ids, 24)?, served);

    // A rollback serves from the kept graphs again.
    bench.runner.graphs.rollback(2)?;
    assert_eq!(bench.runner.graphs.generation(), 1);
    assert_eq!(bench.step(&ids, 24)?, served);

    // The next update is committed: the replaced graphs go and the rebound ones serve.
    let again = bench.kernels.generation(3, &keys)?;
    for bucket in bench.runner.graphs.prepare(&bench.runner, again)? {
        bench.runner.graphs.prepare_bucket(&bench.runner, &bucket)?;
    }
    bench.runner.graphs.switch(3)?;
    bench.runner.graphs.commit(3)?;
    assert_eq!(bench.step(&ids, 24)?, served);
    Ok(())
}

/// A budget that never refuses: the case is about a comparison, not about memory.
#[derive(Default)]
struct Unbounded(Mutex<u64>);

impl Budget for Unbounded {
    fn book(&self, _device: DeviceId, _pool: PoolKind, bytes: u64) -> platform::Result<()> {
        *self.0.lock().unwrap() = bytes;
        Ok(())
    }
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_shadow_of_one_implementation_loaded_twice_compares_every_element() -> Outcome {
    let mut bench = Bench::open()?;

    // The shadow region is what the profile run states: the most one paired call carves.
    let device = platform::Device::new(DEVICE)?;
    let mut allocator = platform::Allocator::new(&device, Arc::new(Unbounded::default()))?;
    allocator.reserve(PoolKind::Scratch, 1 << 26)?;
    let region = bench.runner.graphs.shadow_peak(0);
    bench
        .runner
        .graphs
        .set_shadow(ops::Twins::new(&mut allocator, region)?)?;

    let keys = bench.runner.graphs.keys();
    let next = bench.kernels.generation(2, &keys)?;
    bench.runner.graphs.prepare(&bench.runner, next)?;

    // The normalization's two kernels are one implementation: every element of its output
    // agrees, and nothing is left uncompared.
    let bucket = bench
        .runner
        .prepare_inputs(&batch(&[1, 2, 3, 4], 32), &tables())?;
    let compared = bench
        .runner
        .graphs
        .shadow(&bench.runner, &bucket, "rms_norm")?;
    assert_eq!(compared, Some((ROWS * HIDDEN) as u64));
    assert!(bench.runner.graphs.uncompared().is_empty());

    // The paired forward left the serving graphs as they were.
    bench.stream.synchronize()?;
    let served = bench.step(&[1, 2, 3, 4], 32)?;
    assert!(close(&served, &host_logits(&[1, 2, 3, 4], 32)?));
    Ok(())
}
