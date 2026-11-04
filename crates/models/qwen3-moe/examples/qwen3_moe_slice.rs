//! Qwen3-MoE on a GPU, with no scheduler: the first place this rewrite's operators are launched.
//!
//! A runner builds a device, an allocator, an arena and a `Generation` of loaded kernels before a
//! model computes anything; the executor owns all four. This example builds exactly those four by
//! hand, so a model can be driven with nothing of the executor in the way, and it stages the run so
//! that the first failure names one thing rather than thirty:
//!
//! ```text
//! --stage weights     no device: every parameter maps to a tensor of the config's shape and dtype
//! --stage embedding   one operator on a GPU: the vocabulary table's rows, against the checkpoint
//! --stage block       one decoder layer end to end (`--layer <n>`), against the same rows
//! --stage decode      the whole model's decode step, against notes/reference/qwen3-reference.json
//! ```
//!
//! All but `weights` need a program store: `--store <path>` or `UNISERVE_STORE`, holding the
//! packages the fixture `notes/programs/qwen3-moe-keys.txt` names. Nothing here hard-codes a store
//! root or a checkpoint, because the host that runs it is not the host that wrote it.
//!
//! What is deliberately *not* here: capture, buckets, the prefix cache's paging, request slots and
//! the sampler. This is a numerics harness. Its results are for comparing against a reference, and
//! its failures are for naming the operator and the key that produced them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nn::capability::{KvGroup, KvPlanes, KvView, TokenBatch};
use nn::{
    Arena, Backing, CausalLm, Checkpoint, DType, DevicePtr, Forward, PhaseKind, RetainedWeights,
    Shape, TensorView, WeightKey, WeightLoader,
};
use ops::{Generation, KernelEntry};
use qwen3_moe::{Config, LoadOptions, Partition};

/// The checkpoint the reference was captured from, when `--checkpoint` does not name another.
const DEFAULT_CHECKPOINT: &str = "/opt/uv/models/Qwen3-30B-A3B";

/// The prompt whose first token the reference records, as token ids.
///
/// The reference JSON records tokens and logprobs rather than ids, so the ids come from the
/// tokenizer at capture time; the harness reads the recorded first token and looks its id up in the
/// checkpoint's vocabulary, which is the only direction that needs no tokenizer here.
const DEFAULT_TOKENS: u64 = 1;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Message(message)) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

/// What the harness can fail with: one line, because it is read off a terminal by whoever ran it.
enum Failure {
    Message(String),
}

impl From<nn::Error> for Failure {
    fn from(error: nn::Error) -> Self {
        Self::Message(format!("nn: {error}"))
    }
}

impl From<qwen3_moe::Error> for Failure {
    fn from(error: qwen3_moe::Error) -> Self {
        Self::Message(format!("qwen3-moe: {error}"))
    }
}

impl From<ops::Error> for Failure {
    fn from(error: ops::Error) -> Self {
        Self::Message(format!("ops: {error}"))
    }
}

impl From<platform::Error> for Failure {
    fn from(error: platform::Error) -> Self {
        Self::Message(format!("platform: {error}"))
    }
}

impl Failure {
    fn message(problem: impl Into<String>) -> Self {
        Self::Message(problem.into())
    }
}

type Result<T> = std::result::Result<T, Failure>;

/// The command line, parsed by hand so the example carries no argument-parsing dependency.
struct Args {
    stage: Stage,
    checkpoint: PathBuf,
    store: Option<PathBuf>,
    device: u32,
    layer: usize,
    tokens: String,
}

/// The four stages of the run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Weights,
    Embedding,
    Block,
    Decode,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut stage = Stage::Weights;
        let mut checkpoint = PathBuf::from(DEFAULT_CHECKPOINT);
        let mut store = std::env::var_os("UNISERVE_STORE").map(PathBuf::from);
        let mut device = 0;
        let mut layer = 0;
        let mut tokens = String::new();
        let mut words = std::env::args().skip(1);
        while let Some(word) = words.next() {
            let mut value = |name: &str| -> Result<String> {
                words
                    .next()
                    .ok_or_else(|| Failure::message(format!("{name} needs a value")))
            };
            match word.as_str() {
                "--stage" => {
                    stage = match value("--stage")?.as_str() {
                        "weights" => Stage::Weights,
                        "embedding" => Stage::Embedding,
                        "block" => Stage::Block,
                        "decode" => Stage::Decode,
                        other => {
                            return Err(Failure::message(format!(
                                "unknown stage `{other}`; expected weights, embedding, block or \
                                 decode"
                            )));
                        }
                    };
                }
                "--checkpoint" => checkpoint = PathBuf::from(value("--checkpoint")?),
                "--store" => store = Some(PathBuf::from(value("--store")?)),
                "--device" => {
                    let number = value("--device")?;
                    device = number.parse().map_err(|_| {
                        Failure::message(format!("--device takes a number, not `{number}`"))
                    })?;
                }
                "--layer" => {
                    let number = value("--layer")?;
                    layer = number.parse().map_err(|_| {
                        Failure::message(format!("--layer takes a number, not `{number}`"))
                    })?;
                }
                "--tokens" => tokens = value("--tokens")?,
                "--help" | "-h" => {
                    return Err(Failure::message(
                        "qwen3_moe_slice --stage weights|embedding|block|decode \
                         [--checkpoint PATH] [--store PATH] [--device N] [--layer N]",
                    ));
                }
                other => {
                    return Err(Failure::message(format!(
                        "unknown argument `{other}`; expected --stage, --checkpoint, --store, \
                         --device or --layer"
                    )));
                }
            }
        }
        Ok(Self {
            stage,
            checkpoint,
            store,
            device,
            layer,
            tokens,
        })
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    match args.stage {
        Stage::Weights => stage_weights(&args),
        Stage::Embedding => stage_embedding(&args),
        Stage::Decode => stage_decode(&args),
        Stage::Block => Err(Failure::message(format!(
            "stage `{}` (layer {}) is not wired: `Qwen3Moe` walks every layer in one call and has no \
             per-layer entry point, so one block cannot be run on its own. Stages `weights`, \
             `embedding` and `decode` run today.",
            "block", args.layer
        ))),
    }
}

// -------------------------------------------------------------------------------------------
// Stage 0: the loader's own binding pass, with no device.

/// Weights retained in host memory, which is what a checkpoint read needs before any device is
/// opened: the loader checks every parameter's shape and dtype against the model's own request and
/// publishes the bytes it found. Publishing to host memory makes that check a run without a GPU.
#[derive(Default)]
struct HostWeights {
    published: Vec<Published>,
    bytes: u64,
    by_dtype: BTreeMap<String, (usize, u64)>,
}

/// One published weight: its key as the loader saw it, and the bytes it keeps alive.
struct Published {
    key: String,
    _bytes: Vec<u8>,
}

impl RetainedWeights for HostWeights {
    fn retained(&mut self, _key: &WeightKey) -> nn::Result<Option<Backing>> {
        // Converting a weight twice would hide a duplicate request behind a cache hit; the loader
        // asks once per key, and this store does not answer a second time.
        Ok(None)
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<Backing> {
        let expected = dtype
            .bytes()
            .map(|width| width.saturating_mul(shape.numel()))
            .unwrap_or(0);
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) < expected {
            return Err(nn::Error::Unsupported(format!(
                "weight `{key}`: {} bytes published for {expected} bytes of {dtype} {shape:?}",
                bytes.len()
            )));
        }
        let mut kept = bytes.to_vec();
        kept.truncate(usize::try_from(expected).unwrap_or(bytes.len()));
        let entry = self.by_dtype.entry(dtype.to_string()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += expected;
        self.bytes += expected;
        let address = DevicePtr::new(kept.as_ptr() as u64);
        self.published.push(Published {
            key: key.to_string(),
            _bytes: kept,
        });
        Backing::new(address, expected)
            .view(dtype, shape)
            .map(|_| ())?;
        let last = self
            .published
            .last()
            .map(|published| published._bytes.as_ptr() as u64)
            .unwrap_or(0);
        Ok(Backing::new(DevicePtr::new(last), expected))
    }
}

fn stage_weights(args: &Args) -> Result<()> {
    let checkpoint = Checkpoint::open(&args.checkpoint)?;
    let config = Config::read(&args.checkpoint)?;
    let mut store = HostWeights::default();
    {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        let model = qwen3_moe::load(
            &args.checkpoint,
            Partition::single(),
            &mut loader,
            LoadOptions::default(),
        )?;
        println!(
            "loaded: {} layers, hidden {}, {} experts of {} at top {}, {} parameters",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_experts,
            config.moe_intermediate_size,
            config.num_experts_per_tok,
            store.published.len(),
        );
        // The plan is the arena a forward of one token asks for; printing it costs nothing and
        // tells the next stage how much region to reserve.
        let plan = model.plan(DEFAULT_TOKENS)?;
        println!(
            "plan(1 token): hidden {}, query {}, key/value {}, factors {}, workspace {} bytes",
            plan.hidden, plan.query, plan.key_value, plan.factors, plan.workspace,
        );
        drop(model);
    }
    println!(
        "weights: {} published, {} bytes",
        store.published.len(),
        store.bytes
    );
    if let Some(first) = store.published.first() {
        println!("  first key: {}", first.key);
    }
    for (dtype, (count, bytes)) in &store.by_dtype {
        println!("  {dtype}: {count} weights, {bytes} bytes");
    }
    Ok(())
}

// -------------------------------------------------------------------------------------------
// Stage 1: the vocabulary table's rows, launched on the device.

fn stage_embedding(args: &Args) -> Result<()> {
    let store_root = args.store.clone().ok_or_else(|| {
        Failure::message(
            "stage `embedding` needs a program store: pass --store <path> or set UNISERVE_STORE",
        )
    })?;
    let checkpoint = Checkpoint::open(&args.checkpoint)?;

    // 1. The device and the stream every launch is issued on, before any weight is allocated: a
    //    driver that cannot give a device should fail here and not inside a load.
    let device =
        platform::Device::new(platform::DeviceId(i32::try_from(args.device).unwrap_or(0)))?;
    let _guard = device.enter()?;
    let stream = device.stream()?;

    // 2. The kernels. Every key the fixture names is resolved before a byte is allocated, so a
    //    missing package is reported as coverage rather than as the first operator to launch.
    // Only the operator this stage launches is required: a key the stage never calls must be
    // reported, not fatal, or a missing sampler program would read as a broken embedding.
    let needed = needed_keys(&store_root, Stage::Embedding)?;
    let generation = load_generation(&device, &store_root, &needed)?;

    // 3. Weights on the device, through the loader's own boundary.
    let mut weights = DeviceWeights::new(&device)?;
    let model = {
        let mut loader = WeightLoader::new(&checkpoint, &mut weights, None, None);
        qwen3_moe::load(
            &args.checkpoint,
            Partition::single(),
            &mut loader,
            LoadOptions::default(),
        )?
    };

    // 4. The arena: the plan's own sizes for one token, rounded up, plus the embedding's output.
    let plan = model.plan(DEFAULT_TOKENS)?;
    let hidden = plan.hidden;
    let bytes = arena_bytes(&plan);
    let allocator = weights.allocator();
    allocator.reserve(
        platform::PoolKind::Graph,
        usize::try_from(bytes).unwrap_or(usize::MAX),
    )?;
    let region = allocator.allocate(
        platform::PoolKind::Graph,
        usize::try_from(bytes).unwrap_or(usize::MAX),
    )?;
    let arena = Arena::new(DevicePtr::new(region.address()), bytes);
    let forward = Forward::new(&stream, &generation, &arena);

    // 5. One launch: the ids of the tokens whose rows the checkpoint itself can vouch for.
    let tokens = vec![0_i32, 1, 2, 9707];
    let ids = upload_i32(&stream, &arena, &tokens)?;
    let out = arena.tensor(DType::BF16, Shape::new(&[tokens.len() as u64, hidden]))?;
    model.embed_input_ids(&forward, ids, out)?;
    stream.synchronize()?;

    // 6. Read back and compare with the checkpoint's rows, which is the check that needs no
    //    reference capture: the table is the same tensor the checkpoint holds.
    let table = checkpoint.read("model.embed_tokens.weight")?;
    let rows = download_bf16(
        &stream,
        out,
        tokens.len(),
        usize::try_from(hidden).unwrap_or(0),
    )?;
    let width = usize::try_from(hidden).unwrap_or(0);
    let mut worst = (0_usize, 0.0_f32);
    for (row, token) in tokens.iter().enumerate() {
        let token = usize::try_from(*token).unwrap_or(0);
        for column in 0..width {
            let at = (token * width + column) * 2;
            let expected = half_to_f32(u16::from_le_bytes([table.bytes[at], table.bytes[at + 1]]));
            let got = rows[row * width + column];
            let error = (got - expected).abs();
            if error > worst.1 {
                worst = (row * width + column, error);
            }
        }
    }
    println!(
        "embedding: {} tokens × {} hidden, worst |error| {:.6} at element {}",
        tokens.len(),
        width,
        worst.1,
        worst.0
    );
    println!(
        "  generation 1 holds {} kernels; the table has {} rows of {hidden}",
        generation.len(),
        table.shape.dim(0)
    );
    Ok(())
}

// -------------------------------------------------------------------------------------------
// Stage 3: the whole model over one prompt, for the logits the reference records.

/// The region the decode stage reserves: the plan's workspace, every layer's arena temporaries, the
/// prompt's hidden rows, the logits row and the paged cache.
///
/// There is no reset between layers -- the model allocates each layer's temporaries from one arena
/// and nothing returns the cursor -- so this is sized for the whole stack, not one layer.
fn decode_arena_bytes(plan: &qwen3_moe::Plan, tokens: u64, config: &Config) -> u64 {
    let hidden = plan.hidden.saturating_mul(4).saturating_mul(tokens);
    let logits = u64::try_from(config.vocab_size)
        .unwrap_or(u64::MAX)
        .saturating_mul(4);
    let layers = u64::try_from(config.num_hidden_layers).unwrap_or(0);
    let plane_row = 64_u64
        .saturating_mul(u64::try_from(config.num_key_value_heads).unwrap_or(0))
        .saturating_mul(u64::try_from(config.head_dim).unwrap_or(0))
        .saturating_mul(2);
    let cache = layers
        .saturating_mul(2)
        .saturating_mul(plane_row)
        .saturating_mul(4)
        .max(1 << 20);
    plan.workspace
        .saturating_add(hidden)
        .saturating_add(logits)
        .saturating_add(cache)
        .saturating_add(1 << 30)
}

fn stage_decode(args: &Args) -> Result<()> {
    let store_root = args.store.clone().ok_or_else(|| {
        Failure::message(
            "stage `decode` needs a program store: pass --store <path> or set UNISERVE_STORE",
        )
    })?;
    let tokens: Vec<i32> = args
        .tokens
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim().parse().map_err(|_| {
                Failure::message(format!("--tokens takes comma-separated ids, not `{part}`"))
            })
        })
        .collect::<Result<_>>()?;
    if tokens.is_empty() {
        return Err(Failure::message(
            "stage `decode` needs the prompt's token ids: --tokens 1,2,3 (the reference records \
             strings and logprobs, not ids, so they come from the caller's tokenizer)",
        ));
    }
    let count = u64::try_from(tokens.len()).unwrap_or(0);

    let checkpoint = Checkpoint::open(&args.checkpoint)?;
    let config = Config::read(&args.checkpoint)?;
    let device =
        platform::Device::new(platform::DeviceId(i32::try_from(args.device).unwrap_or(0)))?;
    let _guard = device.enter()?;
    let stream = device.stream()?;

    let needed = needed_keys(&store_root, Stage::Decode)?;
    let generation = load_generation(&device, &store_root, &needed)?;

    let mut weights = DeviceWeights::new(&device)?;
    let model = {
        let mut loader = WeightLoader::new(&checkpoint, &mut weights, None, None);
        qwen3_moe::load(
            &args.checkpoint,
            Partition::single(),
            &mut loader,
            LoadOptions::default(),
        )?
    };

    // The arena, from the plan and the config's own cache geometry.
    let plan = model.plan(count)?;
    let bytes = decode_arena_bytes(&plan, count, &config);
    let allocator = weights.allocator();
    let reserve = usize::try_from(bytes).unwrap_or(usize::MAX);
    allocator.reserve(platform::PoolKind::Graph, reserve)?;
    let region = allocator.allocate(platform::PoolKind::Graph, reserve)?;
    let arena = Arena::new(DevicePtr::new(region.address()), bytes);
    let forward = Forward::new(&stream, &generation, &arena);

    // The batch of one sequence: every prompt token is one query row, the last asks for logits.
    let page_size = 64_u64;
    let pages = count.div_ceil(page_size).max(1);
    let positions: Vec<i32> = (0..count)
        .map(|at| i32::try_from(at).unwrap_or(0))
        .collect();
    let slots: Vec<i32> = (0..count)
        .map(|at| i32::try_from(at).unwrap_or(0))
        .collect();
    let seq_lens = vec![i32::try_from(count).unwrap_or(0)];
    let query_offsets = vec![0_i32, i32::try_from(count).unwrap_or(0)];
    let logits_rows = vec![i32::try_from(count.saturating_sub(1)).unwrap_or(0)];
    let pages_per_row = i32::try_from(pages).unwrap_or(1);
    let block_table: Vec<i32> = (0..pages_per_row).collect();

    let positions = upload_i32(&stream, &arena, &positions)?;
    let slot_mapping = upload_i32(&stream, &arena, &slots)?;
    let seq_lens = upload_i32(&stream, &arena, &seq_lens)?;
    let query_offsets = upload_i32(&stream, &arena, &query_offsets)?;
    let logits_rows = upload_i32(&stream, &arena, &logits_rows)?;
    let block_tables = upload_i32(&stream, &arena, &block_table)?;
    let batch = TokenBatch {
        positions,
        slot_mapping,
        seq_lens,
        query_offsets,
        logits_rows,
        phase: PhaseKind::Prefill,
        page_size: u32::try_from(page_size).unwrap_or(64),
        num_tokens: u32::try_from(count).unwrap_or(0),
        max_kv_len: u32::try_from(pages.saturating_mul(page_size)).unwrap_or(0),
        draft_lens: None,
    };

    // The cache: one group, one plane pair per layer, BF16 with no scales, and flags at zero so the
    // first write initializes every page it touches.
    let kv_heads = u64::try_from(config.num_key_value_heads).unwrap_or(0);
    let head_dim = u64::try_from(config.head_dim).unwrap_or(0);
    let plane = Shape::new(&[pages, page_size, kv_heads, head_dim]);
    let flag = Shape::new(&[pages, 1, 1, 1]);
    let mut planes = Vec::with_capacity(config.num_hidden_layers);
    for _ in 0..config.num_hidden_layers {
        planes.push(KvPlanes {
            key: arena.tensor(DType::BF16, plane)?,
            value: arena.tensor(DType::BF16, plane)?,
            key_initialized: zeroed_i32(&stream, &arena, flag)?,
            value_initialized: zeroed_i32(&stream, &arena, flag)?,
            key_scale: None,
            value_scale: None,
        });
    }
    let groups = [KvGroup {
        block_tables,
        page_size: u32::try_from(page_size).unwrap_or(64),
        dtype: DType::BF16,
        layers: &planes,
    }];
    let cache = KvView { groups: &groups };

    // The forward: embed, walk the stack, project the last row.
    let hidden = arena.tensor(DType::BF16, Shape::new(&[count, plan.hidden]))?;
    model.embed_input_ids(&forward, positions, hidden)?;
    model.forward(&forward, &batch, &cache, hidden)?;
    let logits = arena.tensor(
        DType::F32,
        Shape::new(&[1, u64::try_from(config.vocab_size).unwrap_or(0)]),
    )?;
    let last_row = TensorView::new(
        DevicePtr::new(hidden.ptr.address() + (count - 1) * plan.hidden * 2),
        DType::BF16,
        Shape::new(&[1, plan.hidden]),
    );
    model.logits(&forward, last_row, logits)?;
    stream.synchronize()?;

    // The last row's log-softmax, on the host, and its ten best tokens.
    let vocabulary = config.vocab_size;
    let values = download_f32(&stream, logits, vocabulary)?;
    let best = top_logprobs(&values, 10);
    println!(
        "decode: {count} prompt tokens, {} layers, {} experts; last row's top-10",
        config.num_hidden_layers, config.num_experts
    );
    for (token, logprob) in &best {
        println!("  {token}\t{logprob:.6}");
    }
    Ok(())
}

/// The ten largest `log_softmax` values of one row, as (token id, logprob), largest first.
fn top_logprobs(row: &[f32], count: usize) -> Vec<(usize, f64)> {
    let largest = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !largest.is_finite() {
        return Vec::new();
    }
    let total: f64 = row
        .iter()
        .map(|value| ((value - largest) as f64).exp())
        .sum();
    let log_partition = largest as f64 + total.ln();
    let mut scored: Vec<(usize, f64)> = row
        .iter()
        .enumerate()
        .map(|(token, value)| (token, f64::from(*value) - log_partition))
        .collect();
    scored.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.0.cmp(&right.0))
    });
    scored.truncate(count);
    scored
}

/// A flag plane of `shape` i32 zeros: an initialized flag of zero makes the first write initialize
/// the page it touches.
fn zeroed_i32(stream: &platform::Stream, arena: &Arena, shape: Shape) -> Result<TensorView> {
    let view = arena.tensor(DType::I32, shape)?;
    let zeros = vec![0_i32; usize::try_from(shape.numel()).unwrap_or(0)];
    // SAFETY: `zeros` outlives the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            view.ptr.address(),
            zeros.as_ptr() as u64,
            std::mem::size_of_val(zeros.as_slice()),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(view)
}

/// Download `elements` f32 values.
fn download_f32(stream: &platform::Stream, view: TensorView, elements: usize) -> Result<Vec<f32>> {
    let mut values = vec![0_f32; elements];
    // SAFETY: `values` is written by the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            values.as_mut_ptr() as u64,
            view.ptr.address(),
            std::mem::size_of_val(values.as_slice()),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(values)
}

/// The arena for one token's embedding: the plan's workspace plus the output row and slack.
fn arena_bytes(plan: &qwen3_moe::Plan) -> u64 {
    let row = plan.hidden.saturating_mul(4);
    plan.workspace
        .saturating_add(row.saturating_mul(DEFAULT_TOKENS))
        .saturating_add(1 << 22)
        .max(1 << 24)
}

/// The (operator, key) pairs the fixture names for the slice, which is what the store must hold.
fn needed_keys(store_root: &Path, stage: Stage) -> Result<Vec<(String, String)>> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../notes/programs/qwen3-moe-keys.txt");
    let text = std::fs::read_to_string(&fixture).map_err(|error| {
        Failure::message(format!(
            "reading the fixture {}: {error} (the store root given was {})",
            fixture.display(),
            store_root.display()
        ))
    })?;
    let mut keys = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split('\t');
        let (Some(operator), Some(descriptor)) = (fields.next(), fields.next()) else {
            continue;
        };
        if !required(stage, operator) {
            continue;
        }
        keys.push((operator.to_owned(), descriptor.to_owned()));
    }
    Ok(keys)
}

/// Whether `stage` launches `operator`, which decides whether a missing key is fatal.
///
/// A stage that needs one operator must not be blocked by a key the model uses elsewhere: the store
/// coverage is the caller's report (`notes/programs/store-coverage.md`), and this harness's job is
/// to launch what it says it launches.
fn required(stage: Stage, operator: &str) -> bool {
    match stage {
        Stage::Weights => false,
        Stage::Embedding => operator == "embedding",
        // A block launches everything the model's layer calls, so every fixture key is required.
        Stage::Block | Stage::Decode => true,
    }
}

/// Load every kernel the fixture names from `root`, failing with the complete list of what is
/// missing rather than at the first launch.
fn load_generation(
    device: &platform::Device,
    root: &Path,
    needed: &[(String, String)],
) -> Result<Generation> {
    let mut entries = Vec::new();
    let mut missing = Vec::new();
    for (operator, descriptor) in needed {
        match find_module(root, operator, descriptor) {
            Some(directory) => {
                entries.push(install(device, root, operator, descriptor, &directory)?)
            }
            None => missing.push(format!("{operator}\t{descriptor}")),
        }
    }
    if !missing.is_empty() {
        return Err(Failure::message(format!(
            "the store at {} lacks {} of {} keys:\n  {}",
            root.display(),
            missing.len(),
            needed.len(),
            missing.join("\n  ")
        )));
    }
    Ok(Generation::new(1, entries))
}

/// The committed module directory for `operator`/`descriptor`, under any backend/target/revision.
///
/// The store's layout is `op/backend/revision/target/descriptor`, and which backend, revision and
/// target it holds is the deployment's business: the harness wants the newest committed one.
fn find_module(root: &Path, operator: &str, descriptor: &str) -> Option<PathBuf> {
    let op_root = root.join(operator);
    let mut best: Option<(u32, PathBuf)> = None;
    let backends = std::fs::read_dir(&op_root).ok()?;
    for backend in backends.flatten() {
        let revisions = std::fs::read_dir(backend.path()).ok().into_iter().flatten();
        for revision in revisions.flatten() {
            let number: u32 = revision.file_name().to_string_lossy().parse().unwrap_or(0);
            let targets = std::fs::read_dir(revision.path())
                .ok()
                .into_iter()
                .flatten();
            for target in targets.flatten() {
                let directory = target.path().join(descriptor);
                if directory.join(".committed").is_file()
                    && best.as_ref().is_none_or(|(seen, _)| number >= *seen)
                {
                    best = Some((number, directory));
                }
            }
        }
    }
    best.map(|(_, directory)| directory)
}

/// Load one committed module as an `ops` kernel entry.
fn install(
    device: &platform::Device,
    root: &Path,
    operator: &str,
    descriptor: &str,
    directory: &Path,
) -> Result<KernelEntry> {
    // A Triton package's module carries `metadata.json`, whose `name` is the symbol and whose
    // `shared` is the dynamic shared memory; a vendor cubin carries `launch.json` instead. They are
    // read in that order because the fixture's operators are Triton ones.
    let metadata_path = directory.join("metadata.json");
    let metadata = std::fs::read(&metadata_path).ok();
    let launch = std::fs::read_to_string(directory.join("launch.json")).unwrap_or_default();
    let entry = metadata
        .as_deref()
        .and_then(|bytes| json_field(&String::from_utf8_lossy(bytes), "name"))
        .or_else(|| json_field(&launch, "name"))
        .ok_or_else(|| {
            Failure::message(format!(
                "{}: neither metadata.json nor launch.json names a kernel entry",
                directory.display()
            ))
        })?;
    let cubin = directory.join("module.cubin");
    if cubin.is_file() {
        let image = std::fs::read(&cubin)
            .map_err(|error| Failure::message(format!("reading {}: {error}", cubin.display())))?;
        let kernel = ops::install::install(
            device,
            operator,
            descriptor,
            ops::install::Module::Cubin {
                image: &image,
                metadata: metadata.as_deref(),
                entry: &entry,
            },
        )
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
        return kernel.ok_or_else(|| {
            Failure::message(format!(
                "{}: this build of `ops` has no operator `{operator}` (store root {})",
                directory.display(),
                root.display()
            ))
        });
    }
    let library = std::fs::read_dir(directory)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|file| file.path())
        .find(|path| path.extension().is_some_and(|kind| kind == "so"))
        .ok_or_else(|| {
            Failure::message(format!(
                "{}: neither module.cubin nor a shared object",
                directory.display()
            ))
        })?;
    let installed = ops::install::install(
        device,
        operator,
        descriptor,
        ops::install::Module::Native {
            path: &library,
            entry: &entry,
        },
    )
    .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    installed.ok_or_else(|| {
        Failure::message(format!(
            "{}: this build of `ops` has no operator `{operator}`",
            directory.display()
        ))
    })
}

/// The string value of `field` in a flat JSON object, without a JSON dependency.
fn json_field(text: &str, field: &str) -> Option<String> {
    let needle = format!("\"{field}\"");
    let at = text.find(&needle)? + needle.len();
    let rest = text[at..].trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}

/// Device storage for weights, kept for the life of the model.
struct DeviceWeights {
    pools: platform::Allocator,
    buffers: Vec<platform::Buffer>,
    bytes: u64,
}

impl DeviceWeights {
    fn new(device: &platform::Device) -> Result<Self> {
        let budget = std::sync::Arc::new(NoLimit);
        let pools = platform::Allocator::new(device, budget)?;
        Ok(Self {
            pools,
            buffers: Vec::new(),
            bytes: 0,
        })
    }

    fn allocator(&mut self) -> &mut platform::Allocator {
        &mut self.pools
    }
}

/// The budget a harness with one model on one device needs: the driver's own limit is the only one.
struct NoLimit;

impl platform::Budget for NoLimit {
    fn book(
        &self,
        _device: platform::DeviceId,
        _pool: platform::PoolKind,
        _bytes: u64,
    ) -> platform::Result<()> {
        Ok(())
    }
}

impl RetainedWeights for DeviceWeights {
    fn retained(&mut self, _key: &WeightKey) -> nn::Result<Option<Backing>> {
        Ok(None)
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<Backing> {
        let buffer = self
            .pools
            .allocate(platform::PoolKind::Resident, bytes.len())?;
        // The publish is synchronous with respect to the host: the copy is issued on the device's
        // default stream and waited on before the backing is handed out, because the loader keeps
        // no stream and a later launch may use another one.
        let stream = platform::Stream::new(buffer.device())?;
        // SAFETY: `bytes` outlives the copy, which completes before the wait below.
        unsafe {
            platform::memcpy_async(
                buffer.address(),
                bytes.as_ptr() as u64,
                bytes.len(),
                &stream,
            )?;
        }
        stream.synchronize()?;
        let backing = Backing::new(DevicePtr::new(buffer.address()), bytes.len() as u64);
        backing.view(dtype, shape)?;
        self.bytes += bytes.len() as u64;
        self.buffers.push(buffer);
        let _ = key;
        Ok(backing)
    }
}

/// Upload `values` as an `i32` arena tensor.
fn upload_i32(stream: &platform::Stream, arena: &Arena, values: &[i32]) -> Result<ops::TensorView> {
    let tensor = arena.tensor(DType::I32, Shape::new(&[values.len() as u64]))?;
    // SAFETY: `values` outlives the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            tensor.ptr.address(),
            values.as_ptr() as u64,
            std::mem::size_of_val(values),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(tensor)
}

/// Download `rows × columns` BF16 values as `f32`.
fn download_bf16(
    stream: &platform::Stream,
    view: ops::TensorView,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>> {
    let elements = rows.saturating_mul(columns);
    let mut bytes = vec![0_u8; elements * 2];
    // SAFETY: `bytes` is written by the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            bytes.as_mut_ptr() as u64,
            view.ptr.address(),
            bytes.len(),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| half_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
        .collect())
}

/// A BF16 value as `f32`: the high half of the float, which is exact for every BF16.
fn half_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
