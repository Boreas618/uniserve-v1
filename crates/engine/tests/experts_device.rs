//! The expert load on a device: the tokens each expert of a routed layer takes, counted by the
//! router's top-k in replays of a captured graph and in eager forwards, read back by the runner's
//! graphs without waiting for a step.
//!
//! Every case needs GPU 0 and this host's Build Server (`uniserve build-server serve`). Run them
//! on a GPU host:
//!
//! ```text
//! cargo test -p engine --test experts_device -- --ignored --test-threads 1
//! ```

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

mod device;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::batch::{Batch, BatchId, BatchOutput, Columns, Phase, Shape as BatchShape, Variant};
use engine::memory::tables::BlockTables;
use engine::runner::buckets::Bucket;
use engine::runner::graphs::Graphs;
use engine::runner::{Forward, Runner};
use engine::worker::{Error, Result};
use nn::functional::{self, WeightOrder, embedding};
use nn::modules::Linear;
use ops::Generation;
use platform::{DeviceAllocation, DeviceRange, Stream};
use storage::runtime::GroupId;
use tensor::{DType, Tensor};

use device::{DEVICE, Kernels, Outcome};

const VOCABULARY: usize = 32;
const HIDDEN: usize = 16;
const EXPERTS: usize = 8;
const TOP: usize = 2;
const ROWS: usize = 4;

/// The routed layer's module path, as the model's routed layers name it.
const LAYER: &str = "layers.0.mlp.experts";

/// The offsets of the runner's two input columns, the token ids and the live count, and the
/// bytes of its input tensors.
const IDS: u64 = 0;
const LIVE: u64 = 256;
const INPUTS: u64 = 512;

/// The router's score of expert `expert` for a token whose embedding is the unit vector of
/// column `column`: `((7 · expert + 3 · column) mod 11) / 4`, distinct over the experts (7 is
/// invertible modulo 11) and exact in FP32, so the picks are the same in any arithmetic.
fn score(expert: usize, column: usize) -> f32 {
    ((7 * expert + 3 * column) % 11) as f32 / 4.0
}

/// The logical experts the router picks for token `id`: its `TOP` highest scores.
fn picks(id: u32) -> Vec<usize> {
    let column = id as usize % HIDDEN;
    let mut experts: Vec<usize> = (0..EXPERTS).collect();
    experts.sort_by(|left, right| score(*right, column).total_cmp(&score(*left, column)));
    experts.truncate(TOP);
    experts
}

/// The slot each logical expert is held in: the experts lie in reverse order.
fn slot(expert: usize) -> u32 {
    (EXPERTS - 1 - expert) as u32
}

/// Device memory holding `values` and the U32 tensor of `dims` over it.
fn indices(stream: &Stream, values: &[u32], dims: &[usize]) -> Outcome<(DeviceAllocation, Tensor)> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let allocation = DeviceAllocation::linear(DEVICE, bytes.len())?;

    // SAFETY: the destination is the allocation just made for these bytes, the source this
    // vector, and the stream is waited for before the vector goes.
    unsafe {
        platform::memcpy_async(
            allocation.address(),
            bytes.as_ptr() as u64,
            bytes.len(),
            stream,
        )?;
    }
    stream.synchronize()?;

    let tensor = ops::retained(
        allocation.address(),
        dims,
        DType::U32,
        &ops::cuda(DEVICE.0 as usize),
    )?;
    Ok((allocation, tensor))
}

/// A routed layer's router: the embedding of each token, its scores over the experts, and the
/// top-k, which writes each route's slot and counts the live routes into the layer's load.
struct Router {
    table: Tensor,
    router: Linear,
    /// The slot of each logical expert.
    map: Tensor,
    /// The weights' and the map's device memory, when they are on a device.
    _memory: Vec<DeviceAllocation>,
}

impl Router {
    fn meta() -> Self {
        let tensor = |address: u64, dims: &[usize], dtype: DType| {
            ops::retained(address, dims, dtype, &ops::meta()).unwrap()
        };

        Self {
            table: tensor(0x1_0000, &[VOCABULARY, HIDDEN], DType::F32),
            router: Linear::new(
                tensor(0x2_0000, &[EXPERTS, HIDDEN], DType::F32),
                None,
                WeightOrder::NxK,
            )
            .unwrap(),
            map: tensor(0x3_0000, &[EXPERTS], DType::U32),
            _memory: Vec::new(),
        }
    }

    fn on_device(stream: &Stream) -> Outcome<Self> {
        let table: Vec<f32> = (0..VOCABULARY * HIDDEN)
            .map(|index| f32::from(u8::from(index % HIDDEN == (index / HIDDEN) % HIDDEN)))
            .collect();
        let router: Vec<f32> = (0..EXPERTS * HIDDEN)
            .map(|index| score(index / HIDDEN, index % HIDDEN))
            .collect();
        let map: Vec<u32> = (0..EXPERTS).map(slot).collect();

        let (table_memory, table) = device::weight(stream, &table, &[VOCABULARY, HIDDEN])?;
        let (router_memory, router) = device::weight(stream, &router, &[EXPERTS, HIDDEN])?;
        let (map_memory, map) = indices(stream, &map, &[EXPERTS])?;

        Ok(Self {
            table,
            router: Linear::new(router, None, WeightOrder::NxK)?,
            map,
            _memory: vec![table_memory, router_memory, map_memory],
        })
    }
}

/// A runner of one decode bucket of [`ROWS`] rows over [`Router`].
struct RouterRunner {
    router: Arc<Router>,
    graphs: Graphs,
}

impl Forward for RouterRunner {
    fn buckets(&self) -> Vec<Bucket> {
        vec![Bucket::decode(ROWS as u32, Variant::DecodeGreedy)]
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let rows = bucket.rows as usize;
        let ids = self.graphs.input(IDS, rows, DType::U32)?;
        let live = self.graphs.input(LIVE, 1, DType::U32)?;

        let hidden = embedding(&ids, &self.router.table, 0)?;
        let scores = self.router.router.forward(&hidden)?;
        let load = ops::moe::layer_load(LAYER);
        let (slots, _) = functional::topk_softmax(
            &scores,
            TOP as u32,
            true,
            None,
            Some(&self.router.map),
            Some(&live),
            load.as_ref(),
        )?;
        Ok(vec![slots])
    }

    fn idle_inputs(&self, _bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        columns.fill(0);
        Ok(())
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

impl Runner for RouterRunner {
    fn phases(&self) -> &[Phase] {
        &[Phase::Decode]
    }

    fn capturer(&self, graphs: Graphs) -> Result<Box<dyn Forward>> {
        Ok(Box::new(Self {
            router: Arc::clone(&self.router),
            graphs,
        }))
    }

    fn prepare_inputs(&mut self, batch: &Batch, _tables: &BlockTables) -> Result<Bucket> {
        let bucket = self.buckets()[0];
        let ids = &batch.columns.input_ids;
        if ids.len() > ROWS {
            return Err(Error::Invalid(format!(
                "{} rows in a bucket of {ROWS}",
                ids.len()
            )));
        }

        // The rows past the batch hold token 0, whose routes the live count leaves uncounted.
        self.graphs.prepare_inputs(&*self, &bucket, |columns| {
            columns.fill(0);
            for (index, id) in ids.iter().enumerate() {
                let at = IDS as usize + 4 * index;
                columns[at..at + 4].copy_from_slice(&id.to_ne_bytes());
            }
            let live = ids.len() as u32;
            columns[LIVE as usize..LIVE as usize + 4].copy_from_slice(&live.to_ne_bytes());
            Ok(())
        })?;
        Ok(bucket)
    }

    fn execute(&mut self, _batch: &Batch, _tables: &BlockTables) -> Result<BatchOutput> {
        Err(Error::State("the cases drive the graphs themselves"))
    }
}

/// A decode batch of `ids`.
fn batch(ids: &[u32]) -> Batch {
    Batch {
        id: BatchId(1),
        group: GroupId(0),
        phase: Phase::Decode,
        kernels: 1,
        policy: 1,
        revision: 0,
        shape: BatchShape::decode(ids.len() as u32, Variant::DecodeGreedy),
        requests: Vec::new(),
        columns: Columns {
            input_ids: ids.iter().map(|id| *id as i32).collect(),
            max_kv_len: 16,
            ..Columns::default()
        },
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

/// The slots the runner's last step wrote for its `rows` rows.
fn slots(runner: &RouterRunner, outputs: &[Tensor], rows: usize) -> Outcome<Vec<u32>> {
    runner.graphs.wait()?;
    let _guard = runner.graphs.enter()?;

    let slots = outputs[0].flatten_all()?.to_vec1::<u32>()?;
    Ok(slots[..rows * TOP].to_vec())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn replays_and_eager_forwards_count_each_live_route_by_its_logical_expert() -> Outcome {
    // The profile run states the keys and the peaks before anything touches the device.
    let profiled = RouterRunner {
        router: Arc::new(Router::meta()),
        graphs: Graphs::meta(Arc::new(Generation::new(0, [])), INPUTS),
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
    let mut runner = RouterRunner {
        router: Arc::new(Router::on_device(&stream)?),
        graphs: Graphs::new(
            stream.clone(),
            Arc::clone(&serving),
            range,
            INPUTS,
            profiled.graphs.input_peak(),
            2,
        )?,
    };
    runner
        .graphs
        .set_expert_load(&BTreeMap::from([(LAYER.to_owned(), EXPERTS as u32)]))?;
    runner.graphs.profile_run(&runner)?;

    // The bucket is captured beside the runner's graphs, as a worker's capture thread does,
    // rehearsed first when a kernel's first use is host work; a rehearsal counts nothing.
    let bucket = runner.buckets()[0];
    let capture = runner.capturer(runner.graphs.beside()?)?;
    let rehearsal = runner.capturer(runner.graphs.rehearsal()?)?;
    let plan = capture
        .graphs()
        .examine(&*capture, &bucket, None, None, &serving)?;
    if plan.rehearse {
        rehearsal
            .graphs()
            .rehearse(&*rehearsal, &bucket, &serving)?;
        rehearsal.graphs().release_rehearsal();
    }
    let graph = capture.graphs().capture(&*capture, &bucket, &serving)?;
    runner.graphs.install(&bucket, 1, Some(graph))?;

    // Two replays, one of a batch that leaves a row of the bucket empty, and an eager forward.
    let steps: [(&[u32], bool); 3] = [
        (&[3, 7, 11, 2], true),
        (&[5, 9, 14], true),
        (&[1, 4, 6, 13], false),
    ];
    let mut expected = vec![0_u64; EXPERTS];
    for (ids, replayed) in steps {
        let prepared = runner.prepare_inputs(&batch(ids), &tables())?;
        let outputs = if replayed {
            runner.graphs.replay(&prepared)?
        } else {
            runner.graphs.eager(&runner, &prepared)?
        };

        // Each route is written as its expert's slot, and counted at the expert's own id.
        let picked: Vec<usize> = ids.iter().flat_map(|id| picks(*id)).collect();
        let written: Vec<u32> = picked.iter().map(|expert| slot(*expert)).collect();
        assert_eq!(slots(&runner, &outputs, ids.len())?, written);
        for expert in picked {
            expected[expert] += 1;
        }
    }

    // The counts reach the host behind the stream, each once, within a few takes.
    let mut counted = vec![0_u64; EXPERTS];
    for _ in 0..3 {
        for (layer, tokens) in runner.graphs.take_expert_tokens()? {
            assert_eq!(layer, LAYER);
            for (count, token) in counted.iter_mut().zip(tokens) {
                *count += token;
            }
        }
        runner.graphs.wait()?;
        stream.synchronize()?;
    }
    assert_eq!(counted, expected);
    assert!(runner.graphs.take_expert_tokens()?.is_empty());
    Ok(())
}

/// The device block table of a rank whose requests hold nothing.
fn tables() -> BlockTables {
    BlockTables::new(GroupId(0), &[], &[])
}
