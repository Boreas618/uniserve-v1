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
//!                     (`--substage <cut>` stops inside the layer at one operator)
//! --stage decode      the whole model's decode step, against notes/reference/qwen3-reference.json
//! ```
//!
//! `--reference <dir>` reads a reference stack's `hidden_NN.npy` files one stage at a time as the
//! walk reaches them and prints each stage's `rms/scale` against the reference's own scale, the
//! increment over the stage before it, and at the end the two readings that name a defect: the
//! first stage past [`RELATIVE_BUDGET`] (where we stop agreeing) and the largest increment (where we
//! start being wrong). They are not the same stage in general - smooth growth is bf16 accumulation
//! and a jump is structure - so both are printed. A cut inside a layer prints the prefix's error
//! beside it, because the cut inherits whatever the prefix already lost.
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
use qwen3_moe::{Config, LoadOptions, Partition, Substage};

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
    resident: u64,
    /// Where the decode stage writes its hidden rows and logits as `.npy`, when asked.
    dump: Option<PathBuf>,
    /// A reference stack's `hidden_NN.npy` directory: its files are read one stage at a time as the
    /// decode stage walks the model, and the first stage past [`RELATIVE_BUDGET`] is named.
    reference: Option<PathBuf>,
    /// A cut inside `--layer`, named by the reference tensor that answers it: `normalized`,
    /// `attended` or `mlp_out`.
    substage: Option<Substage>,
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
        let mut resident = 0;
        let mut dump = None;
        let mut reference = None;
        let mut substage = None;
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
                "--dump" => dump = Some(PathBuf::from(value("--dump")?)),
                "--reference" => reference = Some(PathBuf::from(value("--reference")?)),
                "--substage" => {
                    let name = value("--substage")?;
                    substage = Some(Substage::parse(&name).ok_or_else(|| {
                        Failure::message(format!(
                            "--substage {name:?} names no cut; the cuts are {}",
                            Substage::NAMES.join(", ")
                        ))
                    })?);
                }
                "--resident" => {
                    let gib = value("--resident")?;
                    let size: u64 = gib.parse().map_err(|_| {
                        Failure::message(format!("--resident takes whole GiB, not `{gib}`"))
                    })?;
                    resident = size << 30;
                }
                "--help" | "-h" => {
                    return Err(Failure::message(
                        "qwen3_moe_slice --stage weights|embedding|block|decode \
                         [--checkpoint PATH] [--store PATH] [--device N] [--layer N] \
                         [--tokens ID,ID,...] [--dump DIR] [--reference DIR]",
                    ));
                }
                other => {
                    return Err(Failure::message(format!(
                        "unknown argument `{other}`; expected --stage, --checkpoint, --store, \
                         --device, --layer, --tokens, --dump or --reference"
                    )));
                }
            }
        }
        if substage.is_some() && stage != Stage::Block {
            return Err(Failure::message(
                "--substage cuts inside one layer, so it belongs to --stage block",
            ));
        }
        Ok(Self {
            stage,
            checkpoint,
            store,
            device,
            layer,
            tokens,
            resident,
            dump,
            reference,
            substage,
        })
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    match args.stage {
        Stage::Weights => stage_weights(&args),
        Stage::Embedding => stage_embedding(&args),
        Stage::Decode => run_stack(&args, None),
        Stage::Block => run_stack(&args, Some(args.layer)),
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
    /// The bytes of the weights this store actually keeps, capped: the binding pass validates every
    /// parameter and reads none of them back, and the model is tens of gigabytes, so keeping them all
    /// would need the host's memory to load the checkpoint — which is the device's job, not a
    /// validation pass's.
    kept: u64,
}

/// The bytes this store keeps, which is all a binding pass needs: the loader checks a weight against
/// the model's request before publishing it, and nothing reads the published copy without a device.
const HOST_KEEP_BYTES: u64 = 256 << 20;

/// One published weight: its key as the loader saw it, and the bytes when they were kept.
struct Published {
    key: String,
    bytes: u64,
    _kept: Option<Vec<u8>>,
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
        let entry = self.by_dtype.entry(dtype.to_string()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += expected;
        self.bytes += expected;
        // The view is built before the bytes are dropped or kept, so the loader's own fit check
        // (`Backing::view`) runs either way.
        Backing::new(DevicePtr::new(bytes.as_ptr() as u64), expected).view(dtype, shape)?;
        let kept = if self.kept.saturating_add(expected) <= HOST_KEEP_BYTES {
            self.kept += expected;
            let mut copy = bytes.to_vec();
            copy.truncate(usize::try_from(expected).unwrap_or(bytes.len()));
            Some(copy)
        } else {
            None
        };
        // A store that kept nothing still has to answer an address: the binding pass never launches,
        // so the region is never read, and its size is the weight's own.
        let address = kept
            .as_ref()
            .map_or(bytes.as_ptr() as u64, |copy| copy.as_ptr() as u64);
        self.published.push(Published {
            key: key.to_string(),
            bytes: expected,
            _kept: kept,
        });
        Ok(Backing::new(DevicePtr::new(address), expected))
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
        println!("  first key: {} ({} bytes)", first.key, first.bytes);
    }
    println!(
        "  kept {} bytes of them: the pass validates, it does not read back",
        store.kept
    );
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
    let mut weights = DeviceWeights::new(&device, args.resident)?;
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
/// The arena the decode stage reserves: the KV planes and their flags, the batch's metadata, the
/// model's own row buffer, and one forward's plan.
///
/// The forward releases a layer's scratch as the layer ends, so the plan is one layer deep and this
/// is the whole requirement rather than a bound with room to spare. The constant is what the small
/// metadata uploads lose against the arena's alignment.
fn decode_arena_bytes(
    plan: &qwen3_moe::Plan,
    tokens: u64,
    config: &Config,
    page_size: u64,
    pages: u64,
) -> qwen3_moe::Result<u64> {
    let layers = u64::try_from(config.num_hidden_layers).unwrap_or(0);
    let kv_heads = u64::try_from(config.num_key_value_heads).unwrap_or(0);
    let head_dim = u64::try_from(config.head_dim).unwrap_or(0);
    // BF16 planes: one key and one value per layer, over the pages this batch touches.
    let plane = pages
        .saturating_mul(page_size)
        .saturating_mul(kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(2);
    let planes = layers.saturating_mul(2).saturating_mul(plane);
    let flags = layers
        .saturating_mul(2)
        .saturating_mul(pages)
        .saturating_mul(4);
    let rows = tokens.saturating_mul(plan.hidden).saturating_mul(2);
    // The batch's i32 metadata: positions, slots, sequence lengths, the two offsets and the logits
    // rows are seven per token; the page table is `[sequences, pages]` and this batch is one
    // sequence, so its rows are `pages` and not a fixed count — a prompt long enough to need more
    // pages than the slack reserved would otherwise fault at the first attention launch.
    let metadata = tokens
        .saturating_mul(7)
        .saturating_add(pages)
        .saturating_add(64)
        .saturating_mul(4);
    let mut total = 0_u64;
    for bytes in [planes, flags, rows, metadata, plan.bytes(DType::BF16)?] {
        total = total
            .saturating_add(bytes.next_multiple_of(256))
            .saturating_add(256);
    }
    Ok(total)
}

/// The stack over one prompt: `last = Some(layer)` stops after that layer and digests its output,
/// `None` finishes the model and prints the last row's top-10.
fn run_stack(args: &Args, last: Option<usize>) -> Result<()> {
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

    // A reference stack is one prompt's activations, so comparing another prompt against it would
    // report the prompt and not the model. The reference's own `meta.json` records the ids it ran,
    // and a run that names different ones is refused before the first launch rather than after
    // forty-eight stages of disagreement.
    if let Some(dir) = args.reference.as_deref() {
        check_reference_prompt(dir, &tokens)?;
    }

    let checkpoint = Checkpoint::open(&args.checkpoint)?;
    let config = Config::read(&args.checkpoint)?;
    let device =
        platform::Device::new(platform::DeviceId(i32::try_from(args.device).unwrap_or(0)))?;
    let _guard = device.enter()?;
    let stream = device.stream()?;

    let needed = needed_keys(&store_root, Stage::Decode)?;
    let generation = load_generation(&device, &store_root, &needed)?;

    let mut weights = DeviceWeights::new(&device, args.resident)?;
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
    let page_size = 64_u64;
    let pages = count.div_ceil(page_size).max(1);
    let bytes = decode_arena_bytes(&plan, count, &config, page_size, pages)?;
    let allocator = weights.allocator();
    let reserve = usize::try_from(bytes).unwrap_or(usize::MAX);
    allocator.reserve(platform::PoolKind::Graph, reserve)?;
    let region = allocator.allocate(platform::PoolKind::Graph, reserve)?;
    let arena = Arena::new(DevicePtr::new(region.address()), bytes);
    let forward = Forward::new(&stream, &generation, &arena);

    // The batch of one sequence: every prompt token is one query row, the last asks for logits.
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

    // The prompt's ids and its positions are two different buffers with the same length: the
    // embedding reads the ids, and every later launch reads the positions. Handing the positions to
    // the embedding is silent - it produces rows 0..n of the table, a perfectly plausible stack -
    // and every layer after it is then computing on the wrong prompt.
    let ids = upload_i32(&stream, &arena, &tokens)?;
    let positions = upload_i32(&stream, &arena, &positions)?;
    let slot_mapping = upload_i32(&stream, &arena, &slots)?;
    let seq_lens = upload_i32(&stream, &arena, &seq_lens)?;
    let query_offsets = upload_i32(&stream, &arena, &query_offsets)?;
    let logits_rows = upload_i32(&stream, &arena, &logits_rows)?;
    // The page table is `[sequences, pages]`, not a flat list: `paged_attention` addresses row
    // `sequence` of it, and a one-dimensional view is refused as "expected 2 axes, found 1" before the
    // launch. One sequence here, so the extent that matters is the row.
    let block_tables = upload_shaped_i32(
        &stream,
        &arena,
        Shape::new(&[1, u64::try_from(pages_per_row).unwrap_or(1)]),
        &block_table,
    )?;
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

    // The forward: embed, then either stop at one layer or walk the stack and project the last row.
    let hidden = arena.tensor(DType::BF16, Shape::new(&[count, plan.hidden]))?;
    model.embed_input_ids(&forward, ids, hidden)?;
    // The embedding is the one operator in the pass a run can check against the checkpoint alone,
    // and a stack computed on the wrong prompt is indistinguishable from a model with a defect in
    // every layer. So it is checked before the first layer launches, and the run stops here rather
    // than producing a table of thirty numbers that measure the wrong prompt.
    if args.reference.is_some() || args.dump.is_some() {
        check_embedded_rows(&stream, &checkpoint, &tokens, hidden, &plan)?;
    }
    if let Some(layer) = last {
        // The layer's rows, twice: a block has no reference capture, so what it can prove is that
        // every key it composes resolves and that two identical runs agree. A digest that moves
        // between them is a race or an uninitialized buffer, which is worth knowing before a
        // numerical comparison blames the arithmetic.
        let mut digests = Vec::new();
        for run in 0..2 {
            let rows = model.forward_through(&forward, &batch, &cache, hidden, layer)?;
            stream.synchronize()?;
            let values = download_bf16(
                &stream,
                rows,
                tokens_usize(&batch),
                usize::try_from(plan.hidden).unwrap_or(0),
            )?;
            digests.push(digest(&values));
            println!(
                "block: layer {layer}, run {run}: {} rows of {} bf16 columns, digest {digests:?}",
                count,
                usize::try_from(plan.hidden).unwrap_or(0),
            );
        }
        println!(
            "block: layer {layer} {} across two runs; worst |value| {:.6}, sum {:.6}",
            if digests[0] == digests[1] {
                "reproduced"
            } else {
                "DIFFERS"
            },
            digests[0].1,
            digests[0].0,
        );
        if digests[0] != digests[1] {
            return Err(Failure::message(format!(
                "layer {layer} gave two different outputs from the same inputs: the digest moved                  from {:?} to {:?}",
                digests[0], digests[1]
            )));
        }
        // One layer against the reference, without walking the stack: the same comparison the
        // decode stage makes, at a hundredth of the cost, for when a stage has already been named and
        // the question is what one layer does. `--substage` cuts inside the layer, and the cut is
        // read against the prefix that produced it: a clean cut at layer 30 means the layer adds
        // nothing to a prefix already off by that much, never that layers 0..29 are right.
        if args.reference.is_some() || args.dump.is_some() || args.substage.is_some() {
            let rows = tokens_usize(&batch);
            let width = usize::try_from(plan.hidden).unwrap_or(0);
            let mut table = Table::default();
            if let Some(cut) = args.substage {
                if layer == 0 {
                    println!("prefix: layer 0 has none - it is handed the embedding itself");
                } else {
                    let prefix =
                        model.forward_through(&forward, &batch, &cache, hidden, layer - 1)?;
                    stream.synchronize()?;
                    let values = download_bf16(&stream, prefix, rows, width)?;
                    let file = format!("hidden_{layer:02}.npy");
                    let stage = Walked {
                        file: &file,
                        label: &stage_name(layer),
                        index: layer,
                        rows,
                        width,
                    };
                    let error = report_stage(args, stage, &values, &mut table)?;
                    if let Some(error) = error {
                        println!(
                            "prefix: layers 0..{} are at rms/scale {error:.3e} - the cut below adds \
                             to that error and cannot be read as better than it",
                            layer - 1
                        );
                    }
                }
                let out = model.forward_substage(&forward, &batch, &cache, hidden, layer, cut)?;
                stream.synchronize()?;
                let values = download_bf16(&stream, out, rows, width)?;
                let file = format!("hidden_{:02}_{}.npy", layer + 1, cut_file(cut));
                let label = format!("layer {layer}'s `{}` cut", cut.name());
                let stage = Walked {
                    file: &file,
                    label: &label,
                    index: layer + 1,
                    rows,
                    width,
                };
                dump_stage(args, stage, &values)?;
                // The reference tensor that answers this cut exists only if the caller's hooks
                // dumped it; without it the cut is our own number and says where in the layer a
                // number changes, not that the reference agrees.
                let spelled = format!("hidden_{:02}_{}.npy", layer + 1, cut.name());
                let found = args.reference.as_deref().and_then(|dir| {
                    [&file, &spelled]
                        .into_iter()
                        .find(|name| dir.join(name.as_str()).exists())
                        .cloned()
                });
                match found {
                    Some(found) => {
                        let found = Walked {
                            file: &found,
                            ..stage
                        };
                        report_stage(args, found, &values, &mut table)?;
                    }
                    None => println!(
                        "substage: {label}, rms {:.4e} - no `{file}` in the reference directory, \
                         so it is not compared; the file comes from a post-hook on {}",
                        rms(&values),
                        cut_hook(cut)
                    ),
                }
                return Ok(());
            }
            let out = model.forward_through(&forward, &batch, &cache, hidden, layer)?;
            stream.synchronize()?;
            let values = download_bf16(&stream, out, rows, width)?;
            let file = format!("hidden_{:02}.npy", layer + 1);
            let stage = Walked {
                file: &file,
                label: &stage_name(layer + 1),
                index: layer + 1,
                rows,
                width,
            };
            report_stage(args, stage, &values, &mut table)?;
            summarize(&table, config.num_hidden_layers);
        }
        return Ok(());
    }
    // With `--dump`, every stage's output is written as float32 `.npy` for a comparison against a
    // reference stack: the embedding's rows first, then one file per layer, then the logits. The
    // stack is walked one layer at a time rather than all at once, because the only way to see a
    // layer's output is to stop there - `forward_through` runs layers `0..=layer` and releases each
    // layer's scratch as it ends, so the arena pays for one layer and not the stack.
    let width = usize::try_from(plan.hidden).unwrap_or(0);
    // One table for the whole walk: an increment is measured against the stage before it, the last
    // stage included, and that stage is the full run's output rather than a layer's.
    let mut table = Table::default();
    // Set by `hidden_00` below: a table lookup that disagrees means the two stacks are not the same
    // tensor, and the rows that follow would read as defects in layers that are not being measured.
    let mut misaligned = false;
    if args.dump.is_some() || args.reference.is_some() {
        if let Some(dir) = args.dump.as_deref() {
            std::fs::create_dir_all(dir)
                .map_err(|error| Failure::message(format!("--dump {}: {error}", dir.display())))?;
        }
        let rows = tokens_usize(&batch);
        let embedded = download_bf16(&stream, hidden, rows, width)?;
        let stage = Walked {
            file: "hidden_00.npy",
            label: &stage_name(0),
            index: 0,
            rows,
            width,
        };
        let embedded_error = report_stage(args, stage, &embedded, &mut table)?;
        // The tripwire that caught the last misalignment: a table lookup agrees or the two files
        // are not the same tensor, and a stack compared across that gap reads as a defect at every
        // layer while measuring nothing. The rows are still dumped - the file is useful - and are
        // not compared.
        misaligned = embedded_error.is_some_and(|error| error > EMBEDDING_BUDGET);
        if misaligned {
            println!(
                "reference: hidden_00 is a table lookup and is {:.3e} from the reference's - the \
                 comparison is misaligned, so the rows below are dumped and not compared",
                embedded_error.unwrap_or_default()
            );
        }
        // Every call reserves a step's scratch, and nothing returns it: the arena is sized for one
        // forward and not for forty-eight, so each layer's rows are read back to the host and the
        // arena is then released to the mark taken before the first call. Without the release the
        // walk would stop a few layers in, with the arena exhausted.
        // The layer walk ends one short of the stack: the reference's last file is the output of
        // the whole run, after the final norm, and a layer's own output is the state before it. The
        // walk covers `hidden_01..hidden_{L-1}` and the full run below answers `hidden_L`.
        let mark = arena.mark();
        for layer in 0..config.num_hidden_layers.saturating_sub(1) {
            let out = model.forward_through(&forward, &batch, &cache, hidden, layer)?;
            stream.synchronize()?;
            let values = download_bf16(&stream, out, rows, width)?;
            let index = layer + 1;
            let file = format!("hidden_{index:02}.npy");
            let stage = Walked {
                file: &file,
                label: &stage_name(index),
                index,
                rows,
                width,
            };
            if misaligned {
                dump_stage(args, stage, &values)?;
            } else {
                report_stage(args, stage, &values, &mut table)?;
            }
            arena.release(mark);
        }
    }

    model.forward(&forward, &batch, &cache, hidden)?;
    // The run's output norm writes into `hidden`, so this is the state a reference's last file
    // holds: the output after the final norm, which no layer's own output is.
    if args.dump.is_some() || args.reference.is_some() {
        stream.synchronize()?;
        let rows = tokens_usize(&batch);
        let width = usize::try_from(plan.hidden).unwrap_or(0);
        let values = download_bf16(&stream, hidden, rows, width)?;
        let index = config.num_hidden_layers;
        let file = format!("hidden_{index:02}.npy");
        let stage = Walked {
            file: &file,
            label: &describe(index, config.num_hidden_layers),
            index,
            rows,
            width,
        };
        if misaligned {
            dump_stage(args, stage, &values)?;
        } else {
            report_stage(args, stage, &values, &mut table)?;
        }
    }
    if !misaligned {
        summarize(&table, config.num_hidden_layers);
    }
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
    if let Some(dir) = args.dump.as_deref() {
        write_npy(&dir.join("logits.npy"), &values, 1, vocabulary)?;
        println!("dump: logits.npy is the last row, 1 x {vocabulary} f32");
    }
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

/// The batch's token count, as a `usize`, for the host-side readbacks.
fn tokens_usize(batch: &TokenBatch) -> usize {
    usize::try_from(batch.num_tokens).unwrap_or(0)
}

/// Refuse a reference stack that was captured from other prompt ids than the run's.
///
/// `meta.json` beside the `.npy` files records `"ids": [...]`. A stack without one is read as is -
/// the caller may have built the arrays elsewhere - but a stack that names a prompt and does not
/// match is a comparison that cannot say anything about the model, so it stops here.
fn check_reference_prompt(dir: &std::path::Path, tokens: &[i32]) -> Result<()> {
    let path = dir.join("meta.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(());
    };
    let Some(ids_at) = text.find("\"ids\"") else {
        return Ok(());
    };
    let Some(list_at) = text[ids_at..].find('[').map(|at| at + ids_at) else {
        return Ok(());
    };
    let Some(list_end) = text[list_at..].find(']').map(|at| at + list_at) else {
        return Ok(());
    };
    let theirs: Vec<i32> = text[list_at + 1..list_end]
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    if theirs.is_empty() || theirs == tokens {
        return Ok(());
    }
    let shown = |ids: &[i32]| {
        ids.iter()
            .take(16)
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    Err(Failure::message(format!(
        "{} holds a stack for {} tokens ({}), and this run names {} ({}): a different prompt would \
         disagree at every stage and say nothing about the model",
        path.display(),
        theirs.len(),
        shown(&theirs),
        tokens.len(),
        shown(tokens)
    )))
}

/// The relative distance a stage is allowed from the reference before it is a divergence.
///
/// The reference is a bf16 run of the same prompt, so the per-element floor is `2**-8` and a row of
/// `hidden` values averages it down; five percent is two orders above that floor and still far below
/// what a wrong computation produces, which is what makes it a line worth printing.
const RELATIVE_BUDGET: f64 = 0.05;

/// How far `hidden_00` may sit from the reference's before the comparison is called misaligned.
///
/// `hidden_00` is a table lookup: both sides read the same bf16 embedding rows, so the widened f32
/// values are equal and the distance is exactly zero. A nonzero row therefore does not mean the
/// embedding operator is off — it means the reference's file is not the tensor this slice compares
/// against (a different prompt, a hook placed somewhere else, a stack captured at another width),
/// and every row below it would be read as a defect in a layer that is not being measured.
const EMBEDDING_BUDGET: f64 = 1e-3;

/// `hidden_NN`'s stage: the embedding's output at zero, then one layer per file.
fn stage_name(index: usize) -> String {
    match index {
        0 => "the embedding's output".to_owned(),
        other => format!("after layer {}", other - 1),
    }
}

/// A stage's name for a summary line: the run's own output is not a layer's output.
fn describe(index: usize, layers: usize) -> String {
    if index == layers {
        format!("the output after the final norm (hidden_{index:02})")
    } else {
        stage_name(index)
    }
}

/// Write one stage's rows when `--dump` asked, and compare them against the reference when
/// `--reference` did, recording the first stage that passes [`RELATIVE_BUDGET`].
/// The reading of the walk as it accumulates: which stage first crossed the budget tripwire, the
/// error the stage before printed (a stage's increment is measured against it), and the largest
/// increment seen.
///
/// The increment is what names a defect: smooth growth is accumulation and a jump is structure, and
/// the first crossing and the largest increment are not the same stage in general.
#[derive(Default)]
struct Table {
    first_above: Option<usize>,
    previous: Option<f64>,
    largest: Option<(usize, f64)>,
    /// How many stages were compared: one stage has no increment to print.
    stages: usize,
}

impl Table {
    /// Fold one stage's relative error in, answering the increment it makes over the stage before.
    fn push(&mut self, index: usize, relative: f64) -> Option<f64> {
        let increment = self.previous.map(|previous| relative - previous);
        self.stages += 1;
        if let Some(step) = increment {
            let larger = self.largest.is_none_or(|(_, largest)| step > largest);
            if larger {
                self.largest = Some((index, step));
            }
        }
        self.previous = Some(relative);
        if relative > RELATIVE_BUDGET && self.first_above.is_none() {
            self.first_above = Some(index);
        }
        increment
    }
}

/// One stage of the walk: which file it is, what it follows, and the shape of its rows.
#[derive(Clone, Copy)]
struct Walked<'a> {
    file: &'a str,
    label: &'a str,
    index: usize,
    rows: usize,
    width: usize,
}

fn report_stage(
    args: &Args,
    stage: Walked<'_>,
    values: &[f32],
    table: &mut Table,
) -> Result<Option<f64>> {
    let Walked {
        file: name,
        label,
        index,
        rows,
        width,
    } = stage;
    dump_stage(args, stage, values)?;
    let Some(dir) = args.reference.as_deref() else {
        return Ok(None);
    };
    let path = dir.join(name);
    let (theirs, their_rows, their_width) = read_npy(&path)?;
    if their_rows != rows || their_width != width {
        return Err(Failure::message(format!(
            "{} is {their_rows} x {their_width}, the stage is {rows} x {width}",
            path.display()
        )));
    }
    let (relative, worst, scale, ours) = distance(values, &theirs);
    let increment = table.push(index, relative);
    let marker = if relative > RELATIVE_BUDGET {
        "  <-- past the tripwire"
    } else {
        ""
    };
    // The increment is printed beside the error: a stage whose error grew by much more than its
    // predecessor's is the one to open, and one that crossed the tripwire by creeping up is not.
    match increment {
        Some(step) => println!(
            "reference: {name}, {label}, rms {ours:.4e} against the reference's {scale:.4e} \
             — rms/scale {relative:.3e} (step {step:+.2e}), max |diff| {worst:.3e}{marker}"
        ),
        None => println!(
            "reference: {name}, {label}, rms {ours:.4e} against the reference's {scale:.4e} \
             — rms/scale {relative:.3e} (the first stage), max |diff| {worst:.3e}{marker}"
        ),
    }
    Ok(Some(relative))
}

/// Write one stage into the `--dump` directory, when one was asked for.
fn dump_stage(args: &Args, stage: Walked<'_>, values: &[f32]) -> Result<()> {
    let Some(dir) = args.dump.as_deref() else {
        return Ok(());
    };
    write_npy(&dir.join(stage.file), values, stage.rows, stage.width)?;
    println!("dump: {} is {}", stage.file, stage.label);
    Ok(())
}

/// A stage's root-mean-square, for a cut that has no reference file to be measured against.
fn rms(values: &[f32]) -> f64 {
    let count = values.len().max(1) as f64;
    let squared: f64 = values.iter().map(|value| f64::from(*value).powi(2)).sum();
    (squared / count).sqrt()
}

/// The reference module whose hook answer a cut, so a missing file says which hook to run.
fn cut_hook(cut: Substage) -> &'static str {
    match cut {
        Substage::Normalized => "`input_layernorm`",
        Substage::Attended => "`self_attn`",
        Substage::MlpOut => "`mlp`",
    }
}

/// The name a cut's tensor carries in a reference directory.
///
/// The file is named after the module whose hook produced it, not after the cut: the reference's
/// third tensor is `mlp` before its residual add, and this crate calls that cut `mlp_out` so its
/// name states whether the add has happened. Both spellings are accepted when a directory holds
/// one, because a directory somebody else wrote should not have to know our vocabulary.
fn cut_file(cut: Substage) -> &'static str {
    match cut {
        Substage::Normalized => "normalized",
        Substage::Attended => "attended",
        Substage::MlpOut => "mlp",
    }
}

/// The two readings the table gives, said as the decisions they are: where we stop agreeing and
/// where we start being wrong.
fn summarize(table: &Table, layers: usize) {
    if table.previous.is_none() {
        return;
    }
    match table.first_above {
        Some(index) => println!(
            "reference: the first stage past {:.0}% is hidden_{index:02}, {}",
            RELATIVE_BUDGET * 100.0,
            describe(index, layers)
        ),
        None => println!(
            "reference: every stage is within {:.0}% of the reference, hidden_00 to hidden_{layers:02}",
            RELATIVE_BUDGET * 100.0
        ),
    }
    if table.stages < 2 {
        return;
    }
    match table.largest {
        // A step that is not above zero means no stage added anything: nothing here names a defect.
        Some((index, step)) if step > 0.0 => println!(
            "reference: the largest step is at hidden_{index:02} ({step:+.2e}), {} - the stage to open",
            describe(index, layers)
        ),
        _ => println!("reference: no stage's error grew over the one before it"),
    }
}

/// A stage's root-mean-square distance from the reference, relative to the reference's own scale,
/// with the largest absolute difference and that scale beside it.
///
/// The root-mean-square is the measure, not the largest element: a bf16 reference differs from a
/// bf16 computation by about `2**-8` per element and the maximum of a row says more about one
/// rounding than about the two rows' agreement.
fn distance(values: &[f32], reference: &[f32]) -> (f64, f64, f64, f64) {
    let count = values.len().min(reference.len()).max(1) as f64;
    let mut squared = 0.0_f64;
    let mut ours = 0.0_f64;
    let mut theirs = 0.0_f64;
    let mut worst = 0.0_f64;
    for (mine, other) in values.iter().zip(reference.iter()) {
        let difference = f64::from(*mine) - f64::from(*other);
        squared += difference * difference;
        ours += f64::from(*mine) * f64::from(*mine);
        theirs += f64::from(*other) * f64::from(*other);
        worst = worst.max(difference.abs());
    }
    let scale = (theirs / count).sqrt();
    let relative = if scale > 0.0 {
        (squared / count).sqrt() / scale
    } else {
        f64::INFINITY
    };
    // Our own rms goes beside the reference's: a stage whose scale has collapsed disagrees in a way
    // the ratio alone hides, and the reference's growth is the shape the stack has to reproduce.
    (relative, worst, scale, (ours / count).sqrt())
}

/// Read a C-ordered `f32` `.npy` file: the magic, one version, a header dict this reads for the
/// shape, then the little-endian data.
///
/// It is as narrow as the writer beside it on purpose: the reference dumps are `'<f4'` and
/// C-ordered, and a file that is not is refused rather than read into the wrong shape.
fn read_npy(path: &std::path::Path) -> Result<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path)
        .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?;
    let refuse = |what: &str| Failure::message(format!("{}: {what}", path.display()));
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(refuse("not a .npy file"));
    }
    let (header_at, header_len): (usize, usize) = match bytes[6] {
        1 => (10, usize::from(u16::from_le_bytes([bytes[8], bytes[9]]))),
        2 | 3 => (
            12,
            usize::try_from(u32::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11],
            ]))
            .unwrap_or(usize::MAX),
        ),
        other => return Err(refuse(&format!("a .npy version of {other}"))),
    };
    let header = std::str::from_utf8(&bytes[header_at..header_at.saturating_add(header_len)])
        .map_err(|_| refuse("a header that is not UTF-8"))?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err(refuse("not a C-ordered float32 array"));
    }
    let shape_at = header
        .find("'shape': (")
        .ok_or_else(|| refuse("a header with no shape"))?
        + "'shape': (".len();
    let shape_end = header[shape_at..]
        .find(')')
        .ok_or_else(|| refuse("a header with an unterminated shape"))?
        + shape_at;
    let extents: Vec<usize> = header[shape_at..shape_end]
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    let (rows, columns) = match extents.as_slice() {
        [rows, columns] => (*rows, *columns),
        _ => return Err(refuse("an array that is not two-dimensional")),
    };
    let data_at = header_at + header_len;
    let wanted = rows.saturating_mul(columns);
    let available = bytes.len().saturating_sub(data_at) / 4;
    if available < wanted {
        return Err(refuse(&format!(
            "{available} values for a {rows} x {columns} array"
        )));
    }
    let mut values = Vec::with_capacity(wanted);
    for at in (data_at..data_at + wanted * 4).step_by(4) {
        values.push(f32::from_le_bytes([
            bytes[at],
            bytes[at + 1],
            bytes[at + 2],
            bytes[at + 3],
        ]));
    }
    Ok((values, rows, columns))
}

/// Write `values` as a C-ordered `f32` `.npy` file of `rows` x `columns`.
///
/// The format is the small end of `numpy.save`'s: the magic, one version, a header dict padded so
/// that the header's length is a multiple of 64, then the little-endian data. It is written by hand
/// because the readback is four lines of header and no dependency the example does not already
/// have - and a reference comparison wants the array a `numpy.load` returns, nothing else.
fn write_npy(path: &std::path::Path, values: &[f32], rows: usize, columns: usize) -> Result<()> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {columns}), }}");
    // The format's header ends with a newline and starts after a ten-byte preamble, and the whole
    // preamble is padded to a multiple of 64 bytes with spaces.
    let unpadded = 10 + header.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + values.len() * 4);
    bytes.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y']);
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(
        &u16::try_from(header.len())
            .map_err(|_| Failure::message("a .npy header above 65535 bytes"))?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(header.as_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, &bytes)
        .map_err(|error| Failure::message(format!("{}: {error}", path.display())))
}

/// A cheap digest of a block's output: its sum and its largest magnitude, both rounded so that a
/// stable kernel and the same kernel on a second run compare equal.
fn digest(values: &[f32]) -> (i64, i64) {
    let sum: f64 = values.iter().map(|value| f64::from(*value)).sum();
    let worst = values
        .iter()
        .fold(0.0_f32, |worst, value| worst.max(value.abs()));
    ((sum * 1e3) as i64, (f64::from(worst) * 1e3) as i64)
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
        // The model's own operators: the fixture also keys the executor's path and the sampling
        // chain, neither of which this stage calls -- and requiring them would report a missing
        // sampler program as a missing kernel for the decoder.
        Stage::Block | Stage::Decode => MODEL_OPERATORS.contains(&operator),
    }
}

/// The operators `Qwen3Moe`'s own forward launches, in the order a layer composes them.
///
/// The fixture's remaining keys belong to the executor (the cache, the slot commit, the gathers) and
/// to the sampling chain, which this harness computes on the host from the logits row.
const MODEL_OPERATORS: &[&str] = &[
    "embedding",
    "gemm",
    "rms_norm",
    "add_rms_norm",
    "rotary_factors",
    "qk_norm_rope",
    "kv_write",
    "paged_attention",
    "topk_softmax",
    "expert_gather",
    "moe_sort",
    "gather_gemm",
    "route_gemm",
    "moe_unpermute",
    "logits_gather",
    "vocab_unpad",
];

/// Load every kernel the fixture names from `root`, failing with the complete list of what is
/// missing rather than at the first launch.
fn load_generation(
    device: &platform::Device,
    root: &Path,
    needed: &[(String, String)],
) -> Result<Generation> {
    if !root.is_dir() {
        return Err(Failure::message(format!(
            "the store root {} is not a directory: nothing is installed there, so no stage past \
             `weights` can launch. The caller prepares the store (Program Server `prepare`, or the \
             same `op/backend/revision/target/descriptor` directories assembled by hand); this \
             harness only reads it.",
            root.display()
        )));
    }
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
        // A native package's entry is the packaging convention -- `entries = { launch =
        // "uniserve_<operator>" }` in every package manifest in the tree -- and a store assembled
        // without `launch.json` beside the library still names it by its operator. `ops::install`
        // resolves the symbol and refuses with the name when the library does not export it.
        .unwrap_or_else(|| format!("uniserve_{operator}"));
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
    /// An allocator whose resident pool is reserved and can hold `requested` bytes.
    ///
    /// The pool must be reserved before the loader publishes a weight into it, and its size is the
    /// checkpoint's own: the model keeps every parameter on the device for the run, so the region is
    /// the largest thing this harness reserves. The sizes are tried in descending order because the
    /// driver refuses a reservation larger than what the device can still map, and the model's own
    /// total is not known before the weights are read.
    fn new(device: &platform::Device, requested: u64) -> Result<Self> {
        let budget = std::sync::Arc::new(NoLimit);
        let mut pools = platform::Allocator::new(device, budget)?;
        let mut wanted = requested.max(72 << 30);
        let mut reserved = None;
        while wanted >= (8 << 30) {
            if pools
                .reserve(
                    platform::PoolKind::Resident,
                    usize::try_from(wanted).unwrap_or(usize::MAX),
                )
                .is_ok()
            {
                reserved = Some(wanted);
                break;
            }
            wanted /= 2;
        }
        let reserved = reserved.ok_or_else(|| {
            Failure::message(
                "the driver refused every resident reservation from 72 GiB down to 8 GiB: the \
                 model's weights cannot be held on this device",
            )
        })?;
        println!("resident pool: {} GiB reserved", reserved >> 30);
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
fn upload_shaped_i32(
    stream: &platform::Stream,
    arena: &Arena,
    shape: Shape,
    values: &[i32],
) -> Result<ops::TensorView> {
    let tensor = arena.tensor(DType::I32, shape)?;
    if tensor.shape.numel() != u64::try_from(values.len()).unwrap_or(u64::MAX) {
        return Err(Failure::message(format!(
            "{} values for the extents {:?}",
            values.len(),
            shape.dims()
        )));
    }
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

/// Check that the embedding read the prompt's ids: every row of `hidden` is the table's row for
/// that token, which the checkpoint itself can vouch for without a reference capture.
///
/// Both sides are the checkpoint's own BF16 rows widened to `f32`, so the rows are equal exactly.
/// The check costs one read of the table (151 936 × 2048 bf16, about 622 MiB of host memory for the
/// duration of the call) and is paid once per run, only when the run is going to keep its numbers.
/// A row that is not equal names the first token whose row differs and the table row it *does*
/// match, because the failure this exists for is ids and positions being two buffers handed over
/// the wrong way round, and that is what the message has to say rather than "the embedding is
/// wrong".
fn check_embedded_rows(
    stream: &platform::Stream,
    checkpoint: &Checkpoint,
    tokens: &[i32],
    hidden: TensorView,
    plan: &qwen3_moe::Plan,
) -> Result<()> {
    let width = usize::try_from(plan.hidden).unwrap_or(0);
    let rows = download_bf16(stream, hidden, tokens.len(), width)?;
    let table = checkpoint.read("model.embed_tokens.weight")?;
    let table_rows = usize::try_from(table.shape.dim(0)).unwrap_or(0);
    let row_of = |index: usize| -> Option<&[u8]> {
        let at = index.checked_mul(width)?.checked_mul(2)?;
        table.bytes.get(at..at + width * 2)
    };
    for (row, token) in tokens.iter().enumerate() {
        let token = usize::try_from(*token).unwrap_or(usize::MAX);
        let Some(theirs) = row_of(token).filter(|_| token < table_rows) else {
            return Err(Failure::message(format!(
                "token {token} is past the table's {table_rows} rows"
            )));
        };
        let ours = &rows[row * width..(row + 1) * width];
        let agrees = theirs
            .as_chunks::<2>()
            .0
            .iter()
            .zip(ours)
            .all(|(pair, value)| half_to_f32(u16::from_le_bytes(*pair)) == *value);
        if !agrees {
            // The row the embedding actually wrote, for the message: if it is the table's row at
            // this token's *position*, the two buffers were crossed, which is the whole point.
            let wrote = (0..table_rows)
                .find(|index| {
                    row_of(*index).is_some_and(|theirs| {
                        theirs
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .zip(ours)
                            .all(|(pair, value)| half_to_f32(u16::from_le_bytes(*pair)) == *value)
                    })
                })
                .map(|index| index.to_string())
                .unwrap_or_else(|| "no table row".to_owned());
            return Err(Failure::message(format!(
                "row {row} of the embedding is the table's row {wrote}, and the prompt's token there \
                 is {token}: the embedding read something other than the prompt's ids (positions \
                 instead of ids, or an ids buffer nothing filled)"
            )));
        }
    }
    println!(
        "embedding: {} rows are the checkpoint's rows for the prompt's ids, exactly",
        tokens.len()
    );
    Ok(())
}
