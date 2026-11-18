//! What a generation's background capture costs on a device, over a layered model of the shape
//! of a 48-layer dense decoder: each bucket's examination, rehearsal, capture and instantiation
//! on a thread beside the one that serves, the device memory each executable takes, and the time
//! of the serving steps while the capture runs beside them against their time without it.
//!
//! A measurement, not a check of behavior: it prints one JSON line per measured item and asserts
//! only that the steps computed what they computed before. Needs GPU 0 and this host's Build
//! Server of the device cases (`device::DEPLOYMENT`):
//!
//! ```text
//! cargo test -p engine --release --test capture_costs -- --ignored --nocapture
//! ```

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

mod device;

use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::batch::{Batch, BatchId, BatchOutput, Columns, Phase, Shape as BatchShape, Variant};
use engine::memory::tables::BlockTables;
use engine::runner::buckets::Bucket;
use engine::runner::graphs::{Costs, Graph, Graphs, Plan};
use engine::runner::{Forward, Runner};
use engine::worker::{Error, Result};
use nn::functional::{WeightOrder, embedding, rms_norm};
use nn::modules::Linear;
use ops::Generation;
use platform::{DeviceAllocation, DeviceRange, Stream};
use storage::runtime::GroupId;
use tensor::{DType, Tensor};

use device::{DEVICE, Kernels, Outcome};

/// The model's width, layers and vocabulary: a 48-layer decoder of hidden width 2048.
const HIDDEN: usize = 2048;
const LAYERS: usize = 48;
const VOCABULARY: usize = 256;

/// The decode buckets, largest first, as a runner captures them.
const BUCKETS: [u32; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];

/// The rows of the steps the measurement serves.
const SERVED_ROWS: u32 = 8;

/// How many times the capture thread rehearses and captures every bucket of the update while
/// the steps run beside it: enough steps fall beside the captures to read their time.
const ROUNDS: usize = 20;

/// The bytes of the input block: one U32 token id per row of the largest bucket.
const INPUTS: u64 = 4 * 256;

/// One layer: a normalization, two products and the residual sum.
struct Layer {
    gain: Tensor,
    up: Linear,
    down: Linear,
}

/// The model's weights, on the meta device for the profile run or on GPU 0.
struct Layered {
    table: Tensor,
    layers: Vec<Layer>,
    head: Linear,
    _memory: Vec<DeviceAllocation>,
}

impl Layered {
    /// The model over meta weights: what a profile run loads.
    fn meta() -> Self {
        let weight = |dims: &[usize]| ops::retained(0x1_0000, dims, DType::F32, &ops::meta());
        let linear = |dims: &[usize]| Linear::new(weight(dims).unwrap(), None, WeightOrder::NxK);

        Self {
            table: weight(&[VOCABULARY, HIDDEN]).unwrap(),
            layers: (0..LAYERS)
                .map(|_| Layer {
                    gain: weight(&[HIDDEN]).unwrap(),
                    up: linear(&[HIDDEN, HIDDEN]).unwrap(),
                    down: linear(&[HIDDEN, HIDDEN]).unwrap(),
                })
                .collect(),
            head: linear(&[VOCABULARY, HIDDEN]).unwrap(),
            _memory: Vec::new(),
        }
    }

    /// The model over small deterministic weights on the device.
    fn on_device(stream: &Stream) -> Outcome<Self> {
        let mut memory = Vec::new();
        let mut weight = |dims: &[usize], scale: f32| -> Outcome<Tensor> {
            let elements: usize = dims.iter().product();
            let values: Vec<f32> = (0..elements)
                .map(|index| scale * ((index % 7) as f32 - 3.0))
                .collect();
            let (allocation, tensor) = device::weight(stream, &values, dims)?;
            memory.push(allocation);
            Ok(tensor)
        };

        let table = weight(&[VOCABULARY, HIDDEN], 0.01)?;
        let mut layers = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            layers.push(Layer {
                gain: weight(&[HIDDEN], 1.0)?,
                up: Linear::new(weight(&[HIDDEN, HIDDEN], 1e-4)?, None, WeightOrder::NxK)?,
                down: Linear::new(weight(&[HIDDEN, HIDDEN], 1e-4)?, None, WeightOrder::NxK)?,
            });
        }
        let head = Linear::new(weight(&[VOCABULARY, HIDDEN], 1e-3)?, None, WeightOrder::NxK)?;

        Ok(Self {
            table,
            layers,
            head,
            _memory: memory,
        })
    }

    /// The decoder over `ids`: its logits.
    fn forward(&self, ids: &Tensor) -> nn::Result<Tensor> {
        let mut hidden = embedding(ids, &self.table, 0)?;
        for layer in &self.layers {
            let normed = rms_norm(&hidden, &layer.gain, 1e-6, false, DType::F32)?;
            let up = layer.up.forward(&normed)?;
            let down = layer.down.forward(&up)?;
            hidden = hidden.broadcast_add(&down)?;
        }
        self.head.forward(&hidden)
    }
}

/// A runner of [`Layered`] with the decode buckets of [`BUCKETS`].
struct LayeredRunner {
    model: Arc<Layered>,
    graphs: Graphs,
}

impl Forward for LayeredRunner {
    fn buckets(&self) -> Vec<Bucket> {
        BUCKETS
            .iter()
            .map(|rows| Bucket::decode(*rows, Variant::DecodeGreedy))
            .collect()
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let ids = self.graphs.input(0, bucket.rows as usize, DType::U32)?;
        Ok(vec![self.model.forward(&ids)?])
    }

    fn idle_inputs(&self, _bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        columns.fill(0);
        Ok(())
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

impl Runner for LayeredRunner {
    fn phases(&self) -> &[Phase] {
        &[Phase::Decode]
    }

    fn capturer(&self, graphs: Graphs) -> Result<Box<dyn Forward>> {
        Ok(Box::new(Self {
            model: Arc::clone(&self.model),
            graphs,
        }))
    }

    fn prepare_inputs(&mut self, batch: &Batch, _tables: &BlockTables) -> Result<Bucket> {
        let bucket = self
            .graphs
            .select(|bucket| bucket.holds(&batch.shape))
            .ok_or_else(|| Error::Invalid(format!("no bucket holds {}", batch.shape)))?;

        self.graphs.prepare_inputs(&*self, &bucket, |columns| {
            columns.fill(0);
            for (index, id) in batch.columns.input_ids.iter().enumerate() {
                columns[4 * index..4 * index + 4].copy_from_slice(&id.to_ne_bytes());
            }
            Ok(())
        })?;
        Ok(bucket)
    }

    fn execute(&mut self, _batch: &Batch, _tables: &BlockTables) -> Result<BatchOutput> {
        Err(Error::State("the measurement drives the graphs itself"))
    }
}

/// A decode step of `rows` rows.
fn step_batch(rows: u32) -> Batch {
    Batch {
        id: BatchId(1),
        group: GroupId(0),
        phase: Phase::Decode,
        kernels: 1,
        policy: 1,
        revision: 0,
        shape: BatchShape::decode(rows, Variant::DecodeGreedy),
        requests: Vec::new(),
        columns: Columns {
            input_ids: (0..rows as i32)
                .map(|row| row % VOCABULARY as i32)
                .collect(),
            ..Columns::default()
        },
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

/// One served step of `rows` rows: its time, from its preparation to the device's completion,
/// and its logits.
fn step(runner: &mut LayeredRunner, rows: u32) -> Outcome<(Duration, Vec<Vec<f32>>)> {
    let tables = BlockTables::new(GroupId(0), &[], &[]);
    let began = Instant::now();
    let bucket = runner.prepare_inputs(&step_batch(rows), &tables)?;
    let outputs = runner.graphs.replay(&bucket)?;
    runner.graphs.wait()?;
    let took = began.elapsed();

    let _guard = runner.graphs.enter()?;
    let logits = outputs[0].to_vec2::<f32>()?;
    Ok((took, logits[..rows as usize].to_vec()))
}

/// The mean, median and 99th percentile of `times`, in microseconds.
fn summary(times: &mut [Duration]) -> (f64, f64, f64) {
    times.sort();
    let micros = |time: Duration| time.as_secs_f64() * 1e6;
    let mean = times.iter().map(|time| micros(*time)).sum::<f64>() / times.len() as f64;
    let at = |fraction: f64| micros(times[((times.len() - 1) as f64 * fraction) as usize]);
    (mean, at(0.5), at(0.99))
}

/// What the capture thread does for one bucket of generation `kernels` beside `serving`, timed:
/// its plan, the examination's and the rehearsal's time, and its graph with what its capture cost.
fn capture_bucket(
    (capture, rehearsal): (&dyn Forward, &dyn Forward),
    bucket: &Bucket,
    held: Option<&[(&'static str, String, String)]>,
    serving: Option<&Generation>,
    kernels: &Arc<Generation>,
) -> Result<(Plan, Duration, Duration, Option<Graph>)> {
    let began = Instant::now();
    let plan = capture
        .graphs()
        .examine(capture, bucket, held, serving, kernels)?;
    let examined = began.elapsed();

    let began = Instant::now();
    if plan.capture && plan.rehearse {
        rehearsal.graphs().rehearse(rehearsal, bucket, kernels)?;
    }
    let rehearsed = began.elapsed();

    let graph = if plan.capture {
        Some(capture.graphs().capture(capture, bucket, kernels)?)
    } else {
        None
    };
    Ok((plan, examined, rehearsed, graph))
}

/// One JSON line of a bucket's capture.
fn print_capture(phase: &str, bucket: &Bucket, timed: (Plan, Duration, Duration), costs: Costs) {
    let (plan, examined, rehearsed) = timed;
    let ms = |time: Duration| time.as_secs_f64() * 1e3;
    println!(
        "{{\"phase\":\"{phase}\",\"rows\":{},\"rehearsed\":{},\"examine_ms\":{:.3},\
         \"rehearse_ms\":{:.3},\"capture_ms\":{:.3},\"instantiate_ms\":{:.3},\"bytes\":{},\
         \"segments\":{},\"kernel_nodes\":{}}}",
        bucket.rows,
        plan.rehearse,
        ms(examined),
        ms(rehearsed),
        ms(costs.capture),
        ms(costs.instantiate),
        costs.bytes,
        costs.segments,
        costs.kernel_nodes,
    );
}

#[test]
#[ignore = "requires a GPU and a Build Server; a measurement"]
fn capture_costs_beside_serving_steps() -> Outcome {
    // The profile run states the keys and the region before anything touches the device.
    let profiled = LayeredRunner {
        model: Arc::new(Layered::meta()),
        graphs: Graphs::meta(Arc::new(Generation::new(0, [])), INPUTS),
    };
    profiled.graphs.profile_run(&profiled)?;
    let keys = profiled.graphs.keys();
    let mut kernels = Kernels::open()?;
    let serving = kernels.generation(1, &keys)?;

    let stream = Stream::new(DEVICE)?;
    let region = DeviceAllocation::linear(DEVICE, profiled.graphs.peak() as usize)?;
    let range = DeviceRange::new(
        platform::DevicePtr::from_address(region.address()),
        region.len(),
    );
    let mut runner = LayeredRunner {
        model: Arc::new(Layered::on_device(&stream)?),
        graphs: Graphs::new(
            stream.clone(),
            Arc::clone(&serving),
            range,
            INPUTS,
            profiled.graphs.input_peak(),
            2,
        )?,
    };
    runner.graphs.profile_run(&runner)?;

    // The warm-up, as the capture thread runs it.
    let capture = runner.capturer(runner.graphs.beside()?)?;
    let rehearsal = runner.capturer(runner.graphs.rehearsal()?)?;
    for bucket in runner.buckets() {
        let (plan, examined, rehearsed, graph) =
            capture_bucket((&*capture, &*rehearsal), &bucket, None, None, &serving)?;
        let graph = graph.expect("a warm-up captures every bucket");
        print_capture(
            "warm-up",
            &bucket,
            (plan, examined, rehearsed),
            graph.costs(),
        );
        runner.graphs.install(&bucket, 1, Some(graph))?;
    }
    rehearsal.graphs().release_rehearsal();

    // The steps alone.
    let (_, served) = step(&mut runner, SERVED_ROWS)?;
    let mut alone = Vec::new();
    for _ in 0..1000 {
        alone.push(step(&mut runner, SERVED_ROWS)?.0);
    }

    // The same modules loaded again are other kernels under every key: every bucket is
    // rehearsed (the products are cuBLASLt's native entries) and captured again on a thread
    // beside the steps.
    let reloaded = kernels.generation(2, &keys)?;
    let instances = (capture, rehearsal);
    let (mut beside, instances) = update_beside(&mut runner, instances, &reloaded, &served)?;
    print_steps("native", &mut alone, &mut beside);

    // Every kernel but the products' loaded again: every bucket is captured again, and none is
    // rehearsed, which is a generation of compiled kernels.
    let compiled: std::collections::BTreeSet<_> = keys
        .iter()
        .filter(|(operator, _)| *operator != "gemm")
        .cloned()
        .collect();
    let fresh = kernels.generation(3, &compiled)?;
    let entries: Vec<ops::KernelEntry> = fresh
        .iter()
        .map(|(operator, key, kernel)| {
            ops::KernelEntry::keyed(operator, key.to_owned(), Arc::clone(kernel))
        })
        .collect();
    let cubins = Arc::new(reloaded.replace(3, entries));
    let (mut beside, _) = update_beside(&mut runner, instances, &cubins, &served)?;
    print_steps("compiled", &mut alone, &mut beside);
    Ok(())
}

/// Prepares `next` on `runner` and has a thread rehearse and capture each of its buckets
/// [`ROUNDS`] times while this one serves steps of [`SERVED_ROWS`] rows, each of which computes
/// `served`; then switches to it and commits it. Returns the steps' times and the instances,
/// which the thread takes and gives back, and prints each bucket's capture of the first round and
/// the time of one round.
fn update_beside(
    runner: &mut LayeredRunner,
    instances: Instances,
    next: &Arc<Generation>,
    served: &[Vec<f32>],
) -> Outcome<(Vec<Duration>, Instances)> {
    let serving = runner.graphs.kernels();
    let number = next.number();
    let buckets = runner.graphs.prepare(&*runner, Arc::clone(next))?;
    let held: Vec<_> = buckets
        .iter()
        .map(|bucket| runner.graphs.launch_table(bucket))
        .collect();

    let (buckets_ref, held_ref, serving_ref) = (&buckets, &held, &serving);
    let mut beside = Vec::new();
    let began = Instant::now();
    let (captured, instances) = std::thread::scope(|scope| -> Outcome<(Vec<Graph>, Instances)> {
        let thread = scope.spawn(move || -> Result<(Vec<Graph>, Instances)> {
            let (capture, rehearsal) = &instances;
            let (capture, rehearsal) = (&**capture, &**rehearsal);
            let context = platform::PrimaryContext::retain(DEVICE)?;
            let _current = context.enter()?;

            let mut graphs = Vec::new();
            for round in 0..ROUNDS {
                graphs.clear();
                for (bucket, held) in buckets_ref.iter().zip(held_ref) {
                    let (plan, examined, rehearsed, graph) = capture_bucket(
                        (capture, rehearsal),
                        bucket,
                        held.as_deref(),
                        Some(serving_ref),
                        next,
                    )?;
                    let graph = graph.expect("other kernels capture every bucket again");
                    if round == 0 {
                        let phase = format!("generation-{number}");
                        print_capture(&phase, bucket, (plan, examined, rehearsed), graph.costs());
                    }
                    graphs.push(graph);
                }
            }
            rehearsal.graphs().release_rehearsal();
            Ok((graphs, instances))
        });

        while !thread.is_finished() {
            let (took, logits) = step(runner, SERVED_ROWS)?;
            assert_eq!(
                logits, served,
                "a step beside the capture computes what it computed"
            );
            beside.push(took);
        }
        Ok(thread.join().expect("the capture thread")?)
    })?;
    println!(
        "{{\"phase\":\"generation-{number}\",\"round_ms\":{:.3}}}",
        (began.elapsed() / ROUNDS as u32).as_secs_f64() * 1e3
    );

    for (bucket, graph) in buckets.iter().zip(captured) {
        runner.graphs.install(bucket, number, Some(graph))?;
    }
    runner.graphs.switch(number)?;
    runner.graphs.stamp(number);
    assert_eq!(step(runner, SERVED_ROWS)?.1, served);
    runner.graphs.commit(number)?;
    Ok((beside, instances))
}

/// A runner's capture and rehearsal instances of its forward, as its capture thread holds them.
type Instances = (Box<dyn Forward>, Box<dyn Forward>);

/// One JSON line of the steps' times alone and beside the captures of `phase`.
fn print_steps(phase: &str, alone: &mut [Duration], beside: &mut [Duration]) {
    let (alone_mean, alone_median, alone_p99) = summary(alone);
    let (beside_mean, beside_median, beside_p99) = summary(beside);
    println!(
        "{{\"phase\":\"steps-{phase}\",\"rows\":{SERVED_ROWS},\"alone_steps\":{},         \"alone_mean_us\":{:.1},\"alone_p50_us\":{:.1},\"alone_p99_us\":{:.1},         \"beside_steps\":{},\"beside_mean_us\":{:.1},\"beside_p50_us\":{:.1},         \"beside_p99_us\":{:.1}}}",
        alone.len(),
        alone_mean,
        alone_median,
        alone_p99,
        beside.len(),
        beside_mean,
        beside_median,
        beside_p99,
    );
}
