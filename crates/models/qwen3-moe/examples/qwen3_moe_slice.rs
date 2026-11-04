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
//! `--routing` on the block stage asks the router alone: which eight experts this layer chooses and
//! with what weights, on the same input the layer would get. It is not one of the cuts — a cut stops
//! a layer partway — and with `--dump` it writes `hidden_NN_router_ids.npy` as `<i4` (an index
//! compared as a float is a metric in which 17 against 18 is "0.8 % off") and
//! `hidden_NN_router_weights.npy` as `<f4`, the two names the reference's hooks write.
//!
//! `--from NN` on the block stage replaces the embedding with the reference's own `hidden_NN`, so
//! the layer that runs is measured on the reference's state: its row is that layer's arithmetic and
//! carries no prefix's error — the one measurement the layer walk cannot make, because the walk runs
//! the prefix on purpose.
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
//! `--generate N` on the decode stage continues the prompt: the prefill's own last row answers the
//! token at position `count`, and every later token is a **one-row decode batch** appended to the
//! cache the prefill filled, at the token's own position, walked by the same `CausalLm::forward` the
//! prefill walks. The id is `nn::Sampler::greedy`'s - the path a request's zero-temperature row
//! takes - and a sampler that cannot run is a refusal, not a host argmax standing in for it, because
//! an id list cannot say which path chose it. `--greedy host` asks for the host argmax deliberately.
//!
//! `--expect <file>` compares the continuation against a reference's `ids` as it goes, with the
//! reference's own `margins` beside ours when the file records them. `--divergence stop` ends the
//! run at the first id that differs, which is what a cache-precision change on an id-exact chain
//! owes; `--divergence report` keeps going and prints every difference with the margin it moved
//! across, because a lossy weight representation moves ids where the choice was a tie and the gap
//! is the number that separates one from a defect.
//!
//! A run measures a **pair**, and its first line states both halves: the representation the
//! checkpoint declares, read from the checkpoint, and the deployment's cache page type, stated by
//! `--kv-pages bf16|fp8`. Neither is a per-run switch for the other. The second line counts what the
//! loader published by the element type each weight arrived in, beside what the checkpoint's index
//! declares, and refuses a checkpoint whose packed tensors arrived as none.
//!
//! What is deliberately *not* here: capture, buckets, request slots, multi-request batching and the
//! serve path. This is a numerics harness. Its results are for comparing against a reference, and
//! its failures are for naming the operator and the key that produced them.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nn::capability::{KvGroup, KvPlanes, KvView, TokenBatch};
use nn::{
    Arena, Backing, CausalLm, Checkpoint, DType, DevicePtr, Forward, PhaseKind, RetainedWeights,
    Sampler, SamplerState, Samples, Shape, TensorView, WeightKey, WeightLoader,
};
use ops::{Generation, KernelEntry, ScaleLayout};
use qwen3_moe::{Config, LoadOptions, Partition, Plan, Representation, Run, Substage};

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
    /// The reference stage a `--stage block` run starts from: its `hidden_NN` is the run's input
    /// instead of the embedding, which measures one layer's arithmetic and no prefix's.
    from: Option<usize>,
    /// Dump one layer's routing instead of measuring the layer: the router's own top-k, as the
    /// reference's hook records it.
    routing: bool,
    /// Dump the gate's raw logits instead of measuring the layer: the last point a scale on the
    /// router is visible as a scale, because the softmax that follows maps any monotone compression
    /// of a row onto nearly uniform weights.
    scores: bool,
    /// How many tokens to generate after the prompt.
    ///
    /// The prompt is one prefill of `count` rows; every generated token is a **one-row decode step
    /// appended to the cache the prefill filled**, at the token's own position. That is the batch
    /// shape serving runs, and the only way to see whether the cache the prefill wrote is the one a
    /// continuation reads.
    generate: Option<usize>,
    /// Where a generated token id comes from: `nn::Sampler::greedy`, or a host argmax over the same
    /// row when the store has no program for the sampler's launches.
    greedy: Greedy,
    /// The reference's greedy continuation, as JSON with an `ids` list: the run compares as it goes
    /// and stops at the first step that differs, which is the step the defect is in.
    expect: Option<PathBuf>,
    /// The deployment's cache precision. It is a deployment fact and not a per-run switch: the
    /// weights' precision comes from the checkpoint and cannot be overridden here.
    kv_pages: KvPages,
    /// What a difference from the reference means for this run.
    divergence: Divergence,
}

/// What a run does at the first id that differs from the reference.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Divergence {
    /// Stop there. A cache-precision change on an id-exact chain is a defect with a step index, and
    /// the steps after the first difference measure nothing.
    Stop,
    /// Keep going and report every difference with its margin. A lossy weight representation moves
    /// ids where the choice was a tie, so the run's question is how wide the margin was, not whether
    /// an id changed.
    Report,
}

fn load_declared<R: RetainedWeights>(
    root: &Path,
    checkpoint: &Checkpoint,
    store: &mut R,
) -> Result<(qwen3_moe::Qwen3Moe, Representation)> {
    // The dispatch is the crate's: `load_checkpoint` reads the declaration, refuses one that
    // quantizes a parameter this tree serves dense, and picks the entry point. The copy read here
    // is the run's log line, and `run_stack` refuses a run whose two reads disagree.
    let declared = Representation::read(root)?;
    let mut loader = WeightLoader::new(checkpoint, store, None, None);
    let model = qwen3_moe::load::load_checkpoint(
        root,
        Partition::single(),
        &mut loader,
        LoadOptions::default(),
        // The scale layout is the one the compiled NVFP4 programs read - `swizzled_128x4` in their
        // key - and a checkpoint written in the other layout would dequantize with the wrong
        // block's scale.
        ScaleLayout::Swizzled128X4,
    )?;
    Ok((model, declared))
}

/// The element type of the paged cache's pages: the deployment's choice, stated by the run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum KvPages {
    /// BF16 pages, the vertical slice's cache.
    Bf16,
    /// FP8 E4M3 pages with one scale per page.
    Fp8,
}

impl fmt::Display for KvPages {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        })
    }
}

/// Where a generated token id comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Greedy {
    /// `nn::Sampler::greedy`: the shaping operator and the argmax a request's zero-temperature row
    /// takes. It is the path a served request runs, which is what makes its id evidence.
    Sampler,
    /// A host argmax over the downloaded logits row: the same question with no launch, so a store
    /// without the sampler's programs still produces a token to compare later.
    Host,
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
        let mut from = None;
        let mut routing = false;
        let mut scores = false;
        let mut generate = None;
        let mut greedy = Greedy::Sampler;
        let mut expect = None;
        let mut kv_pages = KvPages::Bf16;
        let mut divergence = Divergence::Stop;
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
                "--from" => {
                    let number = value("--from")?;
                    from = Some(number.parse().map_err(|_| {
                        Failure::message(format!("--from takes a stage number, not `{number}`"))
                    })?);
                }
                "--routing" => routing = true,
                "--scores" => scores = true,
                "--generate" => {
                    let number = value("--generate")?;
                    generate = Some(number.parse().map_err(|_| {
                        Failure::message(format!("--generate takes a token count, not `{number}`"))
                    })?);
                }
                "--expect" => expect = Some(PathBuf::from(value("--expect")?)),
                "--divergence" => {
                    let name = value("--divergence")?;
                    divergence = match name.as_str() {
                        "stop" => Divergence::Stop,
                        "report" => Divergence::Report,
                        other => {
                            return Err(Failure::message(format!(
                                "--divergence {other:?} names no behaviour; the behaviours are \
                                 stop and report"
                            )));
                        }
                    };
                }
                "--kv-pages" => {
                    let name = value("--kv-pages")?;
                    kv_pages = match name.as_str() {
                        "bf16" => KvPages::Bf16,
                        "fp8" => KvPages::Fp8,
                        other => {
                            return Err(Failure::message(format!(
                                "--kv-pages {other:?} names no page type; the page types are bf16 \
                                 and fp8"
                            )));
                        }
                    };
                }
                "--greedy" => {
                    let name = value("--greedy")?;
                    greedy = match name.as_str() {
                        "sampler" => Greedy::Sampler,
                        "host" => Greedy::Host,
                        other => {
                            return Err(Failure::message(format!(
                                "--greedy {other:?} names no path; the paths are sampler and host"
                            )));
                        }
                    };
                }
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
                         [--tokens ID,ID,...] [--dump DIR] [--reference DIR] \
                         [--substage normalized|attended|mlp_out] [--from NN] [--routing] [--generate N] [--greedy sampler|host] [--expect FILE] [--divergence stop|report] [--kv-pages bf16|fp8]",
                    ));
                }
                other => {
                    return Err(Failure::message(format!(
                        "unknown argument `{other}`; expected --stage, --checkpoint, --store, \
                         --device, --layer, --tokens, --from, --substage, --routing, --scores, \
                         --generate, --greedy, --expect, --divergence, --kv-pages, --dump or \
                         --reference"
                    )));
                }
            }
        }
        if substage.is_some() && stage != Stage::Block {
            return Err(Failure::message(
                "--substage cuts inside one layer, so it belongs to --stage block",
            ));
        }
        if from.is_some() && stage != Stage::Block {
            return Err(Failure::message(
                "--from starts a run at a reference stage, so it belongs to --stage block",
            ));
        }
        if routing && stage != Stage::Block {
            return Err(Failure::message(
                "--routing dumps one layer's routing, so it belongs to --stage block",
            ));
        }
        // A router reads its layer's *input state*. Above layer 0 that state is only in hand when
        // it comes from the reference: our own is whatever the prefix produced, and asking layer 1's
        // router about the embedding would answer a question about a model nobody runs.
        if routing && layer > 0 && from != Some(layer) {
            return Err(Failure::message(format!(
                "--routing --layer {layer} needs --from {layer}: the router reads that layer's own \
                 input state, and ours above layer 0 exists only after running the prefix"
            )));
        }
        if from.is_some() && reference.is_none() {
            return Err(Failure::message(
                "--from reads the reference's own state, so it needs --reference <dir>",
            ));
        }
        if generate.is_some() && stage != Stage::Decode {
            return Err(Failure::message(
                "--generate continues a prefill, so it belongs to --stage decode",
            ));
        }
        if generate.is_some() && reference.is_some() {
            return Err(Failure::message(
                "--generate compares nothing against a reference stack: the reference records one \
                 prompt's activations and no continuation, so the two would be different questions",
            ));
        }
        if expect.is_some() && generate.is_none() {
            return Err(Failure::message(
                "--expect compares a generation step by step, so it needs --generate N",
            ));
        }
        if generate == Some(0) {
            return Err(Failure::message("--generate takes at least one token"));
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
            from,
            routing,
            scores,
            generate,
            greedy,
            expect,
            kv_pages,
            divergence,
        })
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    check_dump_dir(&args)?;
    match args.stage {
        Stage::Weights => stage_weights(&args),
        Stage::Embedding => stage_embedding(&args),
        Stage::Decode => run_stack(&args, None),
        Stage::Block => run_stack(&args, Some(args.layer)),
    }
}

/// Refuse a `--dump` directory that holds the reference's stages.
///
/// The two sides of every comparison are the reference's hooks' answer and the run's own tensors; a
/// dump into the reference's directory overwrites the answer with the thing it is meant to judge, and
/// the comparison that follows reads one file twice — a ratio of exactly 1 with no residual, which is
/// indistinguishable from a correct run. `hidden_00.npy` is the embedding's stage, which a dump never
/// writes (a dump starts at layer 1), and `_normalized`/`_attended` name cuts only the hooks produce.
fn check_dump_dir(args: &Args) -> Result<()> {
    let Some(dir) = args.dump.as_deref() else {
        return Ok(());
    };
    let marks = [
        "hidden_00.npy",
        "hidden_01_normalized.npy",
        "hidden_01_attended.npy",
    ];
    let hit = marks.iter().find(|name| dir.join(name).exists());
    let Some(name) = hit else {
        return Ok(());
    };
    Err(Failure::message(format!(
        "--dump {} holds the reference's stages ({name}), so a dump there would overwrite the tensors \
         every comparison is measured against: write the run's tensors to a directory of their own and \
         read the reference with --reference {}",
        dir.display(),
        dir.display()
    )))
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
        let (model, _) = load_declared(&args.checkpoint, &checkpoint, &mut store)?;
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
    let needed = needed_keys(
        &store_root,
        Stage::Embedding,
        false,
        args.kv_pages,
        Encoded::default(),
    )?;
    let generation = load_generation(&device, &store_root, &needed)?;

    // 3. Weights on the device, through the loader's own boundary, as the checkpoint declares
    // itself: the declaration chooses the entry point, and the loader refuses a mismatch.
    let mut weights = DeviceWeights::new(&device, args.resident)?;
    let (model, _) = load_declared(&args.checkpoint, &checkpoint, &mut weights)?;

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
    sampling: bool,
    kv_pages: KvPages,
) -> qwen3_moe::Result<u64> {
    let layers = u64::try_from(config.num_hidden_layers).unwrap_or(0);
    let kv_heads = u64::try_from(config.num_key_value_heads).unwrap_or(0);
    let head_dim = u64::try_from(config.head_dim).unwrap_or(0);
    // One key and one value per layer, over the pages this batch touches: two bytes a value in a
    // BF16 page and one in an FP8 page, and an FP8 page carries an f32 scale of its own.
    let element = match kv_pages {
        KvPages::Bf16 => 2,
        KvPages::Fp8 => 1,
    };
    let plane = pages
        .saturating_mul(page_size)
        .saturating_mul(kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(element);
    let scale_plane = match kv_pages {
        KvPages::Bf16 => 0,
        KvPages::Fp8 => layers
            .saturating_mul(2)
            .saturating_mul(pages)
            .saturating_mul(4),
    };
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
    // A generation's arena must hold the prefill's step and one decode step at once, because the
    // prefill's answer is read while its reservation is still held, plus a zero count row and a
    // shaped row, each one vocabulary wide, of which the shaped row is claimed and released per
    // step. The second step's workspace is the plan's own, which the attention program's 512 MiB
    // dominates; naming it here is what makes an arena that is one step short fail at the region's
    // stated size and not at an offset nobody can interpret.
    let sampling_bytes = match sampling {
        true => u64::try_from(config.vocab_size)
            .unwrap_or(0)
            .saturating_mul(4)
            .saturating_mul(4)
            .saturating_add(16_384)
            .saturating_add(plan.workspace),
        false => 0,
    };
    let mut total = 0_u64;
    for bytes in [
        planes,
        scale_plane,
        flags,
        rows,
        metadata,
        sampling_bytes,
        plan.bytes(DType::BF16)?,
    ] {
        total = total
            .saturating_add(bytes.next_multiple_of(256))
            .saturating_add(256);
    }
    Ok(total)
}

/// The stack over one prompt: `last = Some(layer)` stops after that layer and digests its output,
/// `None` finishes the model and prints the last row's top-10.
fn run_stack(args: &Args, last: Option<usize>) -> Result<()> {
    // A run measures a pair - the checkpoint's representation and the deployment's cache precision -
    // and a number that does not say which pair it measured cannot enter a table. The pair is the
    // run's first line, read from the checkpoint and stated by the deployment, never overridden.
    let declared = Representation::read(&args.checkpoint)?;
    // A calibrated checkpoint keeps some tensors in the source dtype and packs the rest, so naming
    // the representation is not the same as measuring it: the counts below come from the checkpoint's
    // own index, where a packed projection appears as a `weight_scale_2` and a kept one does not.
    let (packed, kept) = representation_counts(&args.checkpoint);
    // The declaration of a calibrated export excludes forty-eight modules by name, and a first line
    // that lists them is a first line nobody reads: the count is the fact, the list is in the file.
    let named = match &declared {
        Representation::Dense => "dense (nothing quantized)".to_owned(),
        Representation::Nvfp4 {
            excluded,
            also_excluded,
        } => format!(
            "nvfp4, {} + {} module(s) kept in their own dtype",
            excluded.len(),
            also_excluded.len()
        ),
    };
    println!(
        "combination: checkpoint {} declares {named}; tensors packed {packed}, in their own dtype \
         {kept}; kv pages {}",
        args.checkpoint.display(),
        args.kv_pages,
    );
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

    let needed = needed_keys(
        &store_root,
        Stage::Decode,
        args.generate.is_some(),
        args.kv_pages,
        Encoded {
            declared: declared.is_encoded(),
            tokens: Some(count),
            generated: args.generate.map(|_| 1),
        },
    )?;
    let generation = load_generation(&device, &store_root, &needed)?;

    let mut weights = DeviceWeights::new(&device, args.resident)?;
    let (model, loaded) = load_declared(&args.checkpoint, &checkpoint, &mut weights)?;
    // What the model holds, by representation, beside what the index declares: a loader that fell
    // back to a full-precision path for a packed tensor still prints the declaration and still reads
    // as "NVFP4 ran", so the pair is the measurement and a mismatch is a refusal.
    // The counts are in two units and saying so is part of the measurement: the index counts tensors
    // and the loader publishes stacks, because an encoded expert block fuses everything the router
    // names into one publish per projection pair. So the equivalence is stated by the model and not
    // inferred here - what this line can refuse is a load that brought *nothing* through the
    // representation, which is the fallback that would otherwise read as NVFP4 having run.
    println!(
        "loaded: the index declares {packed} packed tensors and {kept} kept in their own dtype; the \
         loader published {} packed weight(s) ({} bytes of E2M1) and {} full precision",
        weights.packed, weights.packed_bytes, weights.dense,
    );
    if packed > 0 && weights.packed == 0 {
        return Err(Failure::message(format!(
            "the checkpoint declares {packed} packed E2M1 tensor(s) and the loader published none: \
             every one of them arrived in a full-precision path, and a run at that model would \
             measure a fallback and call it NVFP4"
        )));
    }
    if packed == 0 && weights.packed > 0 {
        return Err(Failure::message(format!(
            "the checkpoint declares nothing packed and the loader published {} packed weight(s): \
             the model holds a representation the checkpoint did not state",
            weights.packed
        )));
    }
    if loaded != declared {
        return Err(Failure::message(format!(
            "the checkpoint declared {declared:?} before the load and {loaded:?} during it: a \
             declaration that changes between two reads is one this run cannot attribute"
        )));
    }

    // The arena, from the plan and the config's own cache geometry.
    let plan = model.plan(count)?;
    let page_size = 64_u64;
    let pages = count.div_ceil(page_size).max(1);
    let bytes = decode_arena_bytes(
        &plan,
        count,
        &config,
        page_size,
        pages,
        args.generate.is_some(),
        args.kv_pages,
    )?;
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
    // The page type is the deployment's, and it decides both the planes' element type and whether
    // they carry a scale. An FP8 page's scale is one f32 per page and the writer maintains it - it
    // is read for the growth rule and written back - so a plane that starts at zero has no page
    // whose scale a reader could mistake for a settled one.
    let page_dtype = match args.kv_pages {
        KvPages::Bf16 => DType::BF16,
        KvPages::Fp8 => DType::F8E4M3,
    };
    let mut planes = Vec::with_capacity(config.num_hidden_layers);
    for _ in 0..config.num_hidden_layers {
        let (key_scale, value_scale) = match args.kv_pages {
            KvPages::Bf16 => (None, None),
            KvPages::Fp8 => (
                Some(zeroed_f32(&stream, &arena, flag)?),
                Some(zeroed_f32(&stream, &arena, flag)?),
            ),
        };
        planes.push(KvPlanes {
            key: arena.tensor(page_dtype, plane)?,
            value: arena.tensor(page_dtype, plane)?,
            key_initialized: zeroed_i32(&stream, &arena, flag)?,
            value_initialized: zeroed_i32(&stream, &arena, flag)?,
            key_scale,
            value_scale,
        });
    }
    let groups = [KvGroup {
        block_tables,
        page_size: u32::try_from(page_size).unwrap_or(64),
        dtype: page_dtype,
        layers: &planes,
    }];
    let cache = KvView { groups: &groups };

    // The forward: start from the embedding, then either stop at one layer or walk the stack and
    // project the last row. `--from` replaces the embedding with a reference stage, so what the
    // layer below computes is measured against the reference's own state and not a prefix's.
    let hidden = match args.from {
        Some(from) => {
            let dir = args.reference.as_deref().ok_or_else(|| {
                Failure::message(
                    "--from reads the reference's state, so it needs --reference <dir>",
                )
            })?;
            let rows = reference_rows(
                &stream,
                &arena,
                dir,
                from,
                tokens_usize(&batch),
                usize::try_from(plan.hidden).unwrap_or(0),
            )?;
            println!(
                "input: the run starts at layer {from} with the reference's hidden_{from:02}; the \
                 embedding is not run, so a row measured here carries one more bf16 rounding than a \
                 whole-run row (about 4e-03 per value)"
            );
            rows
        }
        None => {
            let hidden = arena.tensor(DType::BF16, Shape::new(&[count, plan.hidden]))?;
            model.embed_input_ids(&forward, ids, hidden)?;
            // The embedding is the one operator in the pass a run can check against the checkpoint
            // alone, and a stack computed on the wrong prompt is indistinguishable from a model with
            // a defect in every layer. So it is checked before the first layer launches, and the run
            // stops here rather than producing a table of thirty numbers about the wrong prompt.
            if args.reference.is_some() || args.dump.is_some() {
                check_embedded_rows(&stream, &checkpoint, &tokens, hidden, &plan)?;
            }
            hidden
        }
    };
    if let Some(layer) = last {
        // The measurement this stage makes, on whichever input the run was given: the embedding's
        // rows, or the reference's own state when `--from` named a stage. A run that starts at
        // `from` has no prefix in its rows, so what it answers is this layer's arithmetic; a run
        // that starts at zero answers the prefix's error plus this layer's.
        let measure = |stage: Option<Substage>| -> Result<TensorView> {
            match args.from {
                Some(from) => Ok(model.forward_from(
                    &forward,
                    &batch,
                    &cache,
                    hidden,
                    Run {
                        start: from,
                        last: layer,
                        stage,
                    },
                )?),
                None => match stage {
                    None => Ok(model.forward_through(&forward, &batch, &cache, hidden, layer)?),
                    Some(cut) => {
                        Ok(model.forward_substage(&forward, &batch, &cache, hidden, layer, cut)?)
                    }
                },
            }
        };
        // The layer's rows, twice: a block has no reference capture, so what it can prove is that
        // every key it composes resolves and that two identical runs agree. A digest that moves
        // between them is a race or an uninitialized buffer, which is worth knowing before a
        // numerical comparison blames the arithmetic.
        // The router's own answer, on the state the forward's MoE is fed. It is not a `Substage`:
        // the cuts stop a layer partway, and this asks a different question - which eight experts
        // the layer chooses - so it runs the router and nothing else. That state is the
        // post-attention sum and not the layer's input rows, because a router fed the input rows is
        // a router no forward executes; with `--from` both halves come from the reference, so a
        // difference in the ids is our gate's arithmetic or our selection, never an inherited error.
        if args.routing {
            let top = u64::try_from(config.num_experts_per_tok).unwrap_or(0);
            let mark = arena.mark();
            let ids = arena.tensor(
                DType::I32,
                Shape::new(&[u64::try_from(tokens_usize(&batch)).unwrap_or(0), top]),
            )?;
            let weights = arena.tensor(
                DType::F32,
                Shape::new(&[u64::try_from(tokens_usize(&batch)).unwrap_or(0), top]),
            )?;
            let state = post_attention_state(&stream, &arena, args, layer, &batch, &plan)?;
            model.route(&forward, state, layer, ids, weights)?;
            stream.synchronize()?;
            let rows = tokens_usize(&batch);
            let top = usize::try_from(top).unwrap_or(0);
            let chosen = download_i32(&stream, ids, rows * top)?;
            let scored = download_f32(&stream, weights, rows * top)?;
            arena.release(mark);
            report_routing(args, layer, rows, top, &chosen, &scored)?;
            return Ok(());
        }
        // The gate's raw logits, on the same state the routing cut uses. The softmax between the
        // logits and the routing table destroys a great deal: any monotone compression of a row
        // becomes nearly uniform weights, so a table of eight equal weights is consistent both with
        // a wrong vector and with a right vector scaled down. This is the tensor that tells them
        // apart, and its rms is what a reference's own logit rms is compared to.
        if args.scores {
            let mark = arena.mark();
            let state = post_attention_state(&stream, &arena, args, layer, &batch, &plan)?;
            let rows = tokens_usize(&batch);
            let experts = config.num_experts;
            let logits = arena.tensor(
                DType::BF16,
                Shape::new(&[
                    u64::try_from(rows).unwrap_or(0),
                    u64::try_from(experts).unwrap_or(0),
                ]),
            )?;
            model.gate(&forward, state, layer, logits)?;
            stream.synchronize()?;
            let values = download_bf16(&stream, logits, rows, experts)?;
            arena.release(mark);
            report_scores(args, layer, rows, experts, &values)?;
            return Ok(());
        }
        // Every measurement reserves the plan's scratch - the attention program's 512 MiB workspace
        // among it - and nothing returns it: the arena is a bump allocator sized for one forward,
        // so a second measurement without a release asks for a second copy of the whole step and
        // fails at `plan.bytes` past the first. The mark is taken once and released after each
        // measurement has been read back to the host.
        let mark = arena.mark();
        // The digest is the block measurement's own evidence that two identical runs agree, and
        // it costs two extra passes of the layer. A cut, the router's selection or the gate's
        // scores ask a different question of the same layer, so their runs pay for one forward.
        if args.substage.is_none() && !args.routing && !args.scores {
            let mut digests = Vec::new();
            for run in 0..2 {
                let rows = measure(None)?;
                stream.synchronize()?;
                let values = download_bf16(
                    &stream,
                    rows,
                    tokens_usize(&batch),
                    usize::try_from(plan.hidden).unwrap_or(0),
                )?;
                arena.release(mark);
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
                if let Some(from) = args.from {
                    println!(
                        "prefix: none - the run starts at layer {from} with the reference's own \
                         state, so this cut is the layer's arithmetic and not a prefix's error"
                    );
                } else if layer == 0 {
                    println!("prefix: layer 0 has none - it is handed the embedding itself");
                } else {
                    let prefix =
                        model.forward_through(&forward, &batch, &cache, hidden, layer - 1)?;
                    stream.synchronize()?;
                    let values = download_bf16(&stream, prefix, rows, width)?;
                    arena.release(mark);
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
                let out = measure(Some(cut))?;
                stream.synchronize()?;
                let values = download_bf16(&stream, out, rows, width)?;
                arena.release(mark);
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
            let out = measure(None)?;
            stream.synchronize()?;
            let values = download_bf16(&stream, out, rows, width)?;
            arena.release(mark);
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

    // A step of the forward reserves the plan's own scratch - the attention program's workspace
    // among it - and nothing returns it. A generation step runs the same forward again, so the mark
    // is taken here and released once the last row's values are on the host; without it the second
    // step asks for the whole step a second time past the region's end. The logits plane is
    // allocated before the mark because every step reads into it and it must outlive the release.
    let logits = arena.tensor(
        DType::F32,
        Shape::new(&[1, u64::try_from(config.vocab_size).unwrap_or(0)]),
    )?;
    let prefill = arena.mark();
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
    arena.release(prefill);
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
    if args.generate.is_some() {
        generate(
            args,
            &forward,
            &stream,
            &arena,
            &model,
            &cache,
            &batch,
            &block_table,
            logits,
            values,
        )?;
    }
    Ok(())
}

/// Continue a prefill: one token per step, each a one-row decode batch on the cache it filled.
///
/// `prompt` is the prefill's own last row - the row that answers the token at position `count` - and
/// every step here appends the token it chooses at that token's own position, so the cache a
/// continuation reads is the one the prefill wrote. The walk is `CausalLm::forward`, the same call
/// the prefill makes: a second implementation of the stack would answer a question about itself.
#[allow(clippy::too_many_arguments)]
fn generate(
    args: &Args,
    forward: &Forward<'_>,
    stream: &platform::Stream,
    arena: &Arena,
    model: &qwen3_moe::Qwen3Moe,
    cache: &KvView<'_>,
    batch: &TokenBatch,
    block_table: &[i32],
    logits: TensorView,
    prompt: Vec<f32>,
) -> Result<()> {
    let Some(steps) = args.generate else {
        return Ok(());
    };
    let vocabulary = prompt.len();
    let capacity = u64::from(batch.max_kv_len);
    println!(
        "generate: {steps} token(s) after a {} row prompt, greedy by {}",
        batch.num_tokens,
        match args.greedy {
            Greedy::Sampler => "`nn::Sampler::greedy`",
            Greedy::Host => "a host argmax over the downloaded logits row",
        }
    );
    // The sampler and its request state: one row, one slot, one candidate. A greedy row is a
    // temperature-scaled one whose argmax is taken, and `logits_shape` divides by the temperature
    // unconditionally (its key carries `divide`), so the row carries 1.0 rather than the 0.0 the
    // reference's table uses to name a greedy request. Every penalty column is zero, and a zero
    // count row cannot move any column even if one were not.
    let sampler = match args.greedy {
        Greedy::Sampler => Some(Sampler::new(
            u32::try_from(vocabulary).unwrap_or(u32::MAX),
            1,
            1,
            0,
        )?),
        Greedy::Host => None,
    };
    let mut parameters =
        vec![0.0_f32; usize::try_from(ops::sampler::PARAMETER_COLUMNS).unwrap_or(0)];
    parameters[usize::try_from(ops::sampler::TEMPERATURE).unwrap_or(0)] = 1.0;
    let state = sampler
        .as_ref()
        .map(|_| -> Result<SamplerState> {
            Ok(SamplerState {
                counts: upload_shaped_i32(
                    stream,
                    arena,
                    Shape::new(&[1, u64::try_from(vocabulary).unwrap_or(0)]),
                    &vec![0_i32; vocabulary],
                )?,
                parameters: upload_f32(
                    stream,
                    arena,
                    Shape::new(&[1, u64::from(ops::sampler::PARAMETER_COLUMNS)]),
                    &parameters,
                )?,
                slots: upload_i32(stream, arena, &[0])?,
                extras: None,
            })
        })
        .transpose()?;
    let samples = Samples::reserve(arena, DType::I32, 1, false)?;
    let hidden_width = model.plan(1)?.hidden;
    let expected = match args.expect.as_deref() {
        Some(path) => Some(read_expectation(path, &args.tokens)?),
        None => None,
    };
    // The reference's margin at a step, when the file records one: an id that moved says nothing
    // about a defect unless the gap it moved across is known, and the file is where that gap lives.
    let expected_margins = args
        .expect
        .as_deref()
        .map(reference_margins)
        .unwrap_or_default();
    let mut differences: Vec<(usize, i32, i32, f64)> = Vec::new();
    let start = u64::from(batch.num_tokens);
    // The loop's arithmetic, stated once: step k appends the token at position `start + k`, writes
    // the slot of that same position and carries a sequence length one past it. A position or a
    // slot that drifts by one is invisible in an id list that happens to be right, and this is the
    // line that makes the drift visible.
    let last = start.saturating_add(u64::try_from(steps.saturating_sub(1)).unwrap_or(0));
    let page_size = u64::from(batch.page_size);
    // The slot a token writes is the one the attention will read for it: the page its sequence's
    // table names, times the page size, plus the position's offset inside that page. It is one
    // number derived twice - here, and by `kv_write` from the same table - so a mismatch is a write
    // nobody reads, which shows up as a repeated token rather than as an error.
    let first_slot = slot_for(start, page_size, block_table)?;
    let last_slot = slot_for(last, page_size, block_table)?;
    println!(
        "generate: step k (k = 0..{}) appends position {start}+k at slot \
         page_table[position/{page_size}]*{page_size} + position%{page_size}, seq_len {}+k; \
         positions {start}..{last}, slots {first_slot}..{last_slot}, page capacity {capacity}",
        steps.saturating_sub(1),
        start.saturating_add(1),
    );
    let mut row = prompt;
    let mut chosen_ids: Vec<i32> = Vec::with_capacity(steps);
    let mut agreed = 0_usize;
    let mut differed = false;
    for step in 0..steps {
        // The prompt's row answers the token at position `start`; a decode step's row answers the
        // one after it, so the position of this step's token is `start + step`.
        let position = start.saturating_add(u64::try_from(step).unwrap_or(0));
        let slot = slot_for(position, page_size, block_table)?;
        check_continuation(
            start,
            step,
            position,
            slot,
            position.saturating_add(1),
            capacity,
        )?;
        let mark = arena.mark();
        let mut chosen = argmax_over(&row);
        let mut source = "host argmax";
        if let (Some(sampler), Some(state)) = (sampler.as_ref(), state.as_ref()) {
            match sampler.greedy(forward, logits, state, &samples) {
                Ok(()) => {
                    stream.synchronize()?;
                    let theirs = download_i32(stream, samples.tokens, 1)?;
                    source = "sampler";
                    let theirs = theirs.first().copied().unwrap_or(-1);
                    // Two reads of one row: the sampler's shaped argmax and the host's over the raw
                    // logits. They agree unless the shaping moved the row, so a disagreement is a
                    // finding about the sampler and not about the model.
                    if theirs != chosen {
                        println!(
                            "generate: step {}: `Sampler::greedy` chose {theirs} and the raw row's \
                             argmax is {chosen}",
                            step + 1
                        );
                    }
                    chosen = theirs;
                }
                Err(error) => {
                    // A decode answers through `greedy` or it does not answer: a host argmax here
                    // would produce an id list that cannot say which path chose it, and a decode
                    // run is the only evidence the decode path has.
                    return Err(Failure::message(format!(
                        "generate: step {}: the sampler path did not run ({error}); pass \
                         `--greedy host` to ask for the host argmax deliberately",
                        step + 1
                    )));
                }
            }
        }
        let best = top_logprobs(&row, 5);
        println!(
            "generate: step {step} of {steps}: position {position}, slot {slot}, seq_len {}, token \
             {chosen} ({source})",
            position.saturating_add(1)
        );
        for (token, logprob) in &best {
            println!("  {token}\t{logprob:.6}");
        }
        chosen_ids.push(chosen);
        if let Some(dir) = args.dump.as_deref() {
            // The file carries the reference's own index, so a first difference names its file.
            let file = dir.join(format!("logits_step_{step:02}.npy"));
            write_npy(&file, &row, 1, vocabulary)?;
            println!(
                "dump: {} is step {step}'s row, 1 x {vocabulary} f32",
                file.display()
            );
        }
        // The comparison is the measurement: a list that differs says nothing about where, and the
        // first differing step names the launch the defect is in - step 0 is the prefill's own
        // answer, step 1 is the decode's first launch, and a later step is the slot or the page.
        if let Some(expected) = expected.as_deref() {
            match expected.get(step) {
                Some(want) if *want == chosen => agreed += 1,
                Some(want) => {
                    differed = true;
                    let ours = margin(&row);
                    let theirs = expected_margins.get(step).copied();
                    let shown = match theirs {
                        Some(margin) => format!("{margin:.4}"),
                        None => "not in the file".to_owned(),
                    };
                    match args.divergence {
                        Divergence::Stop => {
                            println!(
                                "generate: FIRST DIFFERENCE at reference index {step} (position \
                                 {position}): chose {chosen} with margin {ours:.4}, the reference \
                                 chose {want} with margin {shown}; {step} of {steps} ids agree \
                                 before it"
                            );
                            println!("generate: step {step}'s top-5, on our row:");
                            for (token, logprob) in &best {
                                println!("  {token}\t{logprob:.6}");
                            }
                            arena.release(mark);
                            break;
                        }
                        Divergence::Report => {
                            differences.push((step, chosen, *want, ours));
                            println!(
                                "generate: difference at reference index {step} (position \
                                 {position}): chose {chosen} with margin {ours:.4}, the reference \
                                 chose {want} with margin {shown}; continuing"
                            );
                        }
                    }
                }
                None => {}
            }
        }
        if step + 1 == steps {
            arena.release(mark);
            break;
        }
        // The decode step: this token enters the cache at its own position, and the batch carries
        // one query row whose sequence length counts the tokens already cached.
        // A row that carries no slot writes no keys and is attended over as if it were not there,
        // which reads as a repeated token and not as an error. This batch has one token, so a slot
        // it cannot express is a refusal and not a wrap: `-1` is the value that makes `kv_write`
        // skip the write.
        let position = i32::try_from(position).map_err(|_| {
            Failure::message(format!(
                "position {position} does not fit a batch's i32 positions"
            ))
        })?;
        let slot = i32::try_from(slot).map_err(|_| {
            Failure::message(format!(
                "slot {slot} does not fit a batch's i32 slot mapping"
            ))
        })?;
        if position < 0 || slot < 0 {
            return Err(Failure::message(format!(
                "a decode step at position {position} carries slot {slot}: a negative slot writes \
                 no keys, and a one-token row that writes nothing attends the prompt alone"
            )));
        }
        let ids = upload_i32(stream, arena, &[chosen])?;
        let positions = upload_i32(stream, arena, &[position])?;
        let slots = upload_i32(stream, arena, &[slot])?;
        let seq_lens = upload_i32(stream, arena, &[position.saturating_add(1)])?;
        let query_offsets = upload_i32(stream, arena, &[0, 1])?;
        let logits_rows = upload_i32(stream, arena, &[0])?;
        let next = TokenBatch {
            positions,
            slot_mapping: slots,
            seq_lens,
            query_offsets,
            logits_rows,
            phase: PhaseKind::Decode,
            page_size: batch.page_size,
            num_tokens: 1,
            max_kv_len: batch.max_kv_len,
            draft_lens: None,
        };
        next.check()?;
        let hidden = arena.tensor(DType::BF16, Shape::new(&[1, hidden_width]))?;
        model.embed_input_ids(forward, ids, hidden)?;
        model.forward(forward, &next, cache, hidden)?;
        model.logits(forward, hidden, logits)?;
        stream.synchronize()?;
        row = download_f32(stream, logits, vocabulary)?;
        arena.release(mark);
    }
    // One line, so a reader or a `diff` sees the whole continuation at once.
    let ids = chosen_ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!("generate: ids {ids}");
    match expected.as_deref() {
        Some(expected) => {
            let total = expected.len().min(steps);
            match args.divergence {
                Divergence::Stop if differed => println!(
                    "generate: {agreed} of {total} ids agree with the reference before the first \
                     difference; the ids after it were not run"
                ),
                Divergence::Stop => println!("generate: all {total} ids agree with the reference"),
                Divergence::Report => {
                    println!(
                        "generate: {} of {total} ids differ from the reference; each with the \
                         margin it moved across:",
                        differences.len()
                    );
                    for (step, ours, theirs, our_margin) in &differences {
                        let their_margin = match expected_margins.get(*step) {
                            Some(margin) => format!("{margin:.4}"),
                            None => "not in the file".to_owned(),
                        };
                        println!(
                            "  index {step}: ours {ours} (margin {our_margin:.4}), the reference \
                             {theirs} (margin {their_margin})"
                        );
                    }
                    let widest = differences
                        .iter()
                        .map(|(_, _, _, margin)| *margin)
                        .fold(f64::NEG_INFINITY, f64::max);
                    println!(
                        "generate: the widest margin at a difference is {widest:.4} - a difference \
                         inside a margin is a tie and one outside it is a defect, and the reference's \
                         own margins are what settle which"
                    );
                }
            }
        }
        None => println!(
            "generate: {} tokens, no reference to compare against (pass --expect <file>)",
            chosen_ids.len()
        ),
    }
    Ok(())
}

/// Read a reference continuation: JSON with an `ids` list, and the prompt it continues.
///
/// The prompt is the reason this file can be compared at all: a continuation of another prompt is a
/// different question, and an id list that differs then says nothing about the model.
fn read_expectation(path: &std::path::Path, tokens: &str) -> Result<Vec<i32>> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        Failure::message(format!(
            "reading the expectation {}: {error}",
            path.display()
        ))
    })?;
    let list = |key: &str| -> Option<Vec<i32>> {
        let at = text.find(&format!("\"{key}\""))?;
        let open = text[at..].find('[').map(|offset| offset + at)?;
        let close = text[open..].find(']').map(|offset| offset + open)?;
        Some(
            text[open + 1..close]
                .split(',')
                .filter_map(|part| part.trim().parse().ok())
                .collect(),
        )
    };
    if let (Some(theirs), prompt) = (list("prompt_ids"), tokens) {
        let ours: Vec<i32> = prompt
            .split(',')
            .filter_map(|part| part.trim().parse().ok())
            .collect();
        if !theirs.is_empty() && !ours.is_empty() && theirs != ours {
            return Err(Failure::message(format!(
                "{} continues a {} token prompt and this run's --tokens names {}: a continuation of \
                 another prompt would disagree from the first step and say nothing about the \
                 decode",
                path.display(),
                theirs.len(),
                ours.len()
            )));
        }
    }
    let ids = list("ids").ok_or_else(|| {
        Failure::message(format!(
            "{} holds no `\"ids\": [...]` list to compare against",
            path.display()
        ))
    })?;
    if ids.is_empty() {
        return Err(Failure::message(format!(
            "{} holds an empty `ids` list",
            path.display()
        )));
    }
    Ok(ids)
}

/// The two numbers a continuation's batch must satisfy, checked before it is built.
///
/// A decode step's slot is the token's own position and its sequence length is one past it: the
/// cache a prefill filled holds positions `0..position`, and the new token is the one at `position`.
/// A stage that carries a position shows the one it carried - the sibling of the `hidden_00` guard,
/// because a wrong slot is a plausible number that writes a token's keys over another token's, and
/// no output looks wrong until a later token attends over the wrong rows.
fn check_continuation(
    start: u64,
    step: usize,
    position: u64,
    slot: u64,
    seq_len: u64,
    capacity: u64,
) -> Result<()> {
    let wanted = start.saturating_add(u64::try_from(step).unwrap_or(0));
    if position != wanted {
        return Err(Failure::message(format!(
            "decode step {step} of a {start} row prompt is at position {position}, and the token it \
             appends is the one at {wanted}: a continuation that skips or repeats a position writes \
             its keys beside another token's"
        )));
    }
    if slot != position {
        return Err(Failure::message(format!(
            "a decode step at position {position} carries slot {slot}: the KV contract writes a \
             token's keys at the token's own position"
        )));
    }
    if seq_len != position.saturating_add(1) {
        return Err(Failure::message(format!(
            "a decode step at position {position} carries seq_len {seq_len}, and the token makes it \
             {}",
            position.saturating_add(1)
        )));
    }
    if position >= capacity {
        return Err(Failure::message(format!(
            "position {position} has no page: the page table holds {capacity} positions"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{argmax_over, check_continuation, needed_keys, slot_for, KvPages, Stage};

    /// The store rows a run requires follow the deployment's page type: an fp8 cache asks for the
    /// scale-deriving write and the fp8 reads, and a bf16 cache asks for neither, so neither
    /// deployment is held up by the other one's program.
    #[test]
    fn the_required_rows_follow_the_page_type() {
        let root = Path::new(".");
        let bf16 = needed_keys(root, Stage::Decode, false, KvPages::Bf16).expect("the fixture");
        let fp8 = needed_keys(root, Stage::Decode, false, KvPages::Fp8).expect("the fixture");
        let names = |keys: &[(String, String)]| -> Vec<String> {
            keys.iter().map(|(operator, _)| operator.clone()).collect()
        };
        let (bf16, fp8) = (names(&bf16), names(&fp8));
        assert!(bf16.iter().any(|name| name == "kv_write"), "{bf16:?}");
        assert!(
            !bf16.iter().any(|name| name == "kv_write_scaled"),
            "{bf16:?}"
        );
        assert!(fp8.iter().any(|name| name == "kv_write_scaled"), "{fp8:?}");
        // The same operator twice: the fp8 deployment reads its own key/value type.
        assert_eq!(
            fp8.iter().filter(|name| *name == "paged_attention").count(),
            2,
            "{fp8:?}"
        );
        assert_eq!(
            bf16.iter()
                .filter(|name| *name == "paged_attention")
                .count(),
            2,
            "{bf16:?}"
        );
    }

    /// The line a continuation's batch must satisfy, on the numbers a prompt of fifteen tokens
    /// produces: the first generated token is the one at position 15, its sequence length is 16, and
    /// the page table holds more than that.
    #[test]
    fn a_continuation_appends_at_the_prompts_own_position() {
        check_continuation(15, 0, 15, 15, 16, 64).expect("position 15 of a 64 page table");
        check_continuation(15, 1, 16, 16, 17, 64).expect("the second token is at position 16");
    }

    /// A counter used as a slot is the defect this check exists for: step 0's counter is 0, which
    /// writes the sixteenth token's keys over the first token's page.
    #[test]
    fn a_counter_slot_is_refused() {
        let refused = check_continuation(15, 0, 15, 0, 16, 64).expect_err("slot 0 for position 15");
        assert!(refused.to_string().contains("own position"), "{refused}");
    }

    /// A position that skips or repeats is refused before the batch is built.
    #[test]
    fn a_skipped_position_is_refused() {
        let refused =
            check_continuation(15, 0, 16, 16, 17, 64).expect_err("position 16 for step 0");
        assert!(refused.to_string().contains("appends"), "{refused}");
    }

    /// A sequence length that does not count the token being appended is refused.
    #[test]
    fn a_short_sequence_length_is_refused() {
        let refused = check_continuation(15, 0, 15, 15, 15, 64).expect_err("seq_len 15 for 16");
        assert!(refused.to_string().contains("seq_len"), "{refused}");
    }

    /// A position past the page table has no page to write, which is a refusal and not a launch.
    #[test]
    fn a_position_past_the_page_table_is_refused() {
        let refused = check_continuation(63, 0, 63, 63, 64, 64).expect_err("position 64 of 64");
        assert!(refused.to_string().contains("no page"), "{refused}");
    }

    /// The slot a decode writes is the page table's own arithmetic: the page's unit times the page
    /// size plus the offset. An identity table makes it the position, which is why a slot that
    /// drifts is invisible until a sequence spans two non-adjacent pages.
    #[test]
    fn the_slot_is_the_page_tables_arithmetic() {
        assert_eq!(slot_for(15, 64, &[0]).unwrap(), 15);
        // Position 70 is offset 6 of the sequence's second page, and that page is unit 3.
        assert_eq!(slot_for(70, 64, &[0, 3]).unwrap(), 3 * 64 + 6);
    }

    /// A page that is not a page writes no keys, so it is refused rather than skipped.
    #[test]
    fn a_page_out_of_the_table_is_refused() {
        let refused = slot_for(15, 64, &[]).expect_err("no pages");
        assert!(
            refused.to_string().contains("page table names"),
            "{refused}"
        );
        let refused = slot_for(15, 64, &[-1]).expect_err("page -1");
        assert!(refused.to_string().contains("no cache"), "{refused}");
    }

    /// The committed reference carries both halves a difference is read with: the ids and, beside
    /// them, the margin each was chosen by. The reader takes the file as it is, and step 19's margin
    /// of exactly zero is the tie this lane argued about for a night.
    #[test]
    fn the_reference_continuation_carries_its_margins() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../notes/reference");
        let file = dir.join("qwen3-greedy-32-margins.json");
        let ids = super::read_expectation(&file, "").expect("the committed reference");
        let margins = super::reference_margins(&file);
        assert_eq!(ids.len(), 32, "{ids:?}");
        assert_eq!(margins.len(), 32, "{margins:?}");
        assert_eq!(ids[0], 151667);
        assert_eq!(ids[19], 358);
        assert_eq!(margins[19], 0.0, "the tie at step 19");
        // A file with ids alone is a reference this run still compares against, and its margins read
        // as absent rather than as zero.
        let ids_only = dir.join("qwen3-greedy-32.json");
        assert_eq!(
            super::read_expectation(&ids_only, "")
                .expect("its ids")
                .len(),
            32
        );
        assert!(super::reference_margins(&ids_only).is_empty());
    }

    /// An encoded deployment asks the store for its own products, per row count; a dense one asks
    /// for none of them, so neither representation's gap can block the other's run.
    #[test]
    fn an_encoded_deployment_requires_its_own_rows() {
        let root = Path::new(".");
        let dense = super::needed_keys(
            root,
            Stage::Decode,
            false,
            KvPages::Bf16,
            super::Encoded::default(),
        )
        .expect("the fixture");
        assert!(
            !dense.iter().any(|(operator, _)| operator.contains("nvfp4")),
            "{dense:?}"
        );
        let encoded = super::needed_keys(
            root,
            Stage::Decode,
            false,
            KvPages::Bf16,
            super::Encoded {
                declared: true,
                tokens: Some(15),
                generated: Some(1),
            },
        )
        .expect("the fixture");
        let packed = encoded
            .iter()
            .filter(|(operator, _)| operator == "nvfp4_gemm")
            .count();
        // The product is compiled per row count: every `m_15` and `m_1` row the fixture names, and
        // no row for a count this run will not launch.
        assert_eq!(packed, 6, "{encoded:?}");
        for operator in ["nvfp4_quantize", "gather_gemm_nvfp4", "route_gemm_nvfp4"] {
            assert!(
                encoded.iter().any(|(name, _)| name == operator),
                "{operator} missing"
            );
        }
        assert!(
            !encoded
                .iter()
                .any(|(_, descriptor)| descriptor.contains("_m_7_")),
            "a row count this run will not launch"
        );
    }

    /// The host tie rule: the first largest wins, which is the one the sampler's argmax takes.
    #[test]
    fn the_argmax_takes_the_first_largest() {
        assert_eq!(argmax_over(&[0.5, 2.0, 2.0, -1.0]), 1);
        assert_eq!(argmax_over(&[-3.0, -2.0, -4.0]), 1);
    }
}

/// The slot a token at `position` occupies: its sequence's page from the table, times the page size,
/// plus the position's offset in that page.
///
/// This is the number `kv_write` computes from the same two inputs, so printing it beside the
/// position is what separates "the write landed where the read looks" from "the write landed
/// elsewhere and the attention read the prompt alone". A page the table does not name, and a page
/// named `-1`, are refusals: both are a token with no cache.
fn slot_for(position: u64, page_size: u64, block_table: &[i32]) -> Result<u64> {
    let unit = position / page_size;
    let index = usize::try_from(unit).unwrap_or(usize::MAX);
    let entry = block_table.get(index).ok_or_else(|| {
        Failure::message(format!(
            "position {position} lives in page {unit}, and the page table names {} page(s)",
            block_table.len()
        ))
    })?;
    let page = u64::try_from(*entry).map_err(|_| {
        Failure::message(format!(
            "page {unit} of the page table names {entry}: a page that is not a page is a token \
             with no cache"
        ))
    })?;
    Ok(page
        .saturating_mul(page_size)
        .saturating_add(position % page_size))
}

/// How many of a checkpoint's tensors are packed and how many kept in their own dtype.
///
/// A packed projection lists `weight_scale_2` beside its packed `weight`; a kept one lists only its
/// tensor. The index is the checkpoint's own statement of what it holds, so the count is a
/// measurement of the file rather than a restatement of its name.
fn representation_counts(root: &Path) -> (usize, usize) {
    let index =
        std::fs::read_to_string(root.join("model.safetensors.index.json")).unwrap_or_default();
    let mut names = 0_usize;
    let mut packed = 0_usize;
    let mut weights = 0_usize;
    for name in index.split('"') {
        if !name.ends_with(".weight") && !name.ends_with(".weight_scale_2") {
            continue;
        }
        names += 1;
        if name.ends_with(".weight_scale_2") {
            packed += 1;
        } else {
            weights += 1;
        }
    }
    // A tensor counted as packed still has its own `weight` entry, so what the representation kept
    // is every plain weight without a packed sibling.
    let _ = names;
    (packed, weights.saturating_sub(packed))
}

/// The reference's margins, when its file records them under `margins`.
fn reference_margins(path: &Path) -> Vec<f64> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let Some(at) = text.find("\"margins\"") else {
        return Vec::new();
    };
    let Some(open) = text[at..].find('[').map(|offset| offset + at) else {
        return Vec::new();
    };
    let Some(close) = text[open..].find(']').map(|offset| offset + open) else {
        return Vec::new();
    };
    text[open + 1..close]
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect()
}

/// The gap between a row's best and its runner-up, in logit units.
///
/// A log-softmax subtracts one constant from a whole row, so the difference of two log-probabilities
/// is the difference of the two logits: a margin of a thousandth says the choice was a tie and a
/// margin of two says it was not.
fn margin(row: &[f32]) -> f64 {
    let best = top_logprobs(row, 2);
    match best.as_slice() {
        [first, second, ..] => first.1 - second.1,
        _ => f64::NAN,
    }
}

/// The largest value's first index, on the host.
fn argmax_over(row: &[f32]) -> i32 {
    let mut best = (0_usize, f32::NEG_INFINITY);
    for (at, value) in row.iter().enumerate() {
        if *value > best.1 {
            best = (at, *value);
        }
    }
    i32::try_from(best.0).unwrap_or(-1)
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

/// Up to how many rows a comparison prints one by one.
const ROW_DETAIL_LIMIT: usize = 32;

/// One row's magnitude against another's, and the cosine between them.
///
/// The magnitude ratio says how much of the row is there; the cosine says whether what is there
/// points the same way. A defect that scales a row moves the first and leaves the second at 1, and a
/// defect that replaces part of a row moves both — which is why the pair is printed together.
fn row_shape(mine: &[f32], other: &[f32]) -> (f64, f64) {
    let mut dot = 0.0_f64;
    let mut squares = 0.0_f64;
    let mut theirs = 0.0_f64;
    for (one, two) in mine.iter().zip(other.iter()) {
        let one = f64::from(*one);
        let two = f64::from(*two);
        dot += one * two;
        squares += one * one;
        theirs += two * two;
    }
    let ratio = if theirs > 0.0 {
        (squares / theirs).sqrt()
    } else {
        f64::INFINITY
    };
    let cosine = if squares > 0.0 && theirs > 0.0 {
        dot / (squares.sqrt() * theirs.sqrt())
    } else {
        0.0
    };
    (ratio, cosine)
}

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
    // Per row, because a deficit that is one scalar everywhere and a deficit that is scattered
    // across rows are different defects and the aggregate cannot tell them apart: a row's ratio is
    // its magnitude against the reference's, and its cosine is its direction. Fifteen rows is the
    // harness's regime, so the detail is free.
    if rows <= ROW_DETAIL_LIMIT {
        for row in 0..rows {
            let mine = &values[row * width..(row + 1) * width];
            let other = &theirs[row * width..(row + 1) * width];
            let (ratio, cosine) = row_shape(mine, other);
            println!(
                "  row {row:2}: norm {:.4e} against {:.4e} — ratio {ratio:.4}, cosine {cosine:+.5}",
                rms(mine),
                rms(other)
            );
        }
    }
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
        Substage::GateIn => "`post_attention_layernorm`",
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
        Substage::GateIn => "gate_in",
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

/// A plane of `shape` f32 zeros.
fn zeroed_f32(stream: &platform::Stream, arena: &Arena, shape: Shape) -> Result<TensorView> {
    let view = arena.tensor(DType::F32, shape)?;
    let zeros = vec![0_f32; usize::try_from(shape.numel()).unwrap_or(0)];
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
fn needed_keys(
    store_root: &Path,
    stage: Stage,
    sampling: bool,
    kv_pages: KvPages,
    encoded: Encoded,
) -> Result<Vec<(String, String)>> {
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
        // A cache this deployment does not select is not a missing program: the `kv_f8e4m3` rows
        // belong to an fp8 deployment and the bf16 rows to a bf16 one, and a run must not be blocked
        // by the other's gap.
        let scaled = operator == "kv_write_scaled"
            || descriptor.contains("kv_f8e4m3")
            || descriptor.contains("kv_f8e5m2");
        if scaled != (kv_pages == KvPages::Fp8) {
            continue;
        }
        // An encoded checkpoint's programs are its own: a dense deployment must not be held up by
        // them, and an NVFP4 one must not be able to load and then fail at its first product. The
        // packed product is compiled per row count, so only the counts this run will launch are
        // required - the prompt's, and one row when a continuation will follow it.
        let packed_row = operator == "nvfp4_gemm"
            && [encoded.tokens, encoded.generated]
                .into_iter()
                .flatten()
                .any(|rows| descriptor.contains(&format!("_m_{rows}_")));
        if encoded.declared != (packed_row || ENCODED_OPERATORS.contains(&operator)) {
            continue;
        }
        if !required(stage, operator, sampling) {
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
fn required(stage: Stage, operator: &str, sampling: bool) -> bool {
    match stage {
        Stage::Weights => false,
        Stage::Embedding => operator == "embedding",
        // The model's own operators, plus the sampling chain when `--generate` runs it: a store
        // without those programs would otherwise fail at the first launch with a key nobody checked.
        Stage::Block | Stage::Decode => {
            MODEL_OPERATORS.contains(&operator)
                || (sampling && SAMPLER_OPERATORS.contains(&operator))
                // The scale-deriving write is an fp8 deployment's, and the encoded products are an
                // NVFP4 checkpoint's; the filter above has already kept only the rows that
                // deployment selects.
                || operator == "kv_write_scaled"
                || ENCODED_OPERATORS.contains(&operator)
        }
    }
}

/// The operators `nn::Sampler::greedy` launches: the shaping operator and the argmax.
///
/// A greedy step asks for no candidates, no cutoffs and no draw, so `top_k`, `filter_candidates`,
/// `candidate_order`, `inverse_cdf` and the two gathers are not launched and are not required. The
/// log-probability of the chosen token is computed on the host from the same row, which keeps this
/// list at two and keeps a missing `log_softmax` program out of the way of a token's id.
const SAMPLER_OPERATORS: &[&str] = &["logits_shape", "argmax"];

/// The operators an encoded checkpoint's step launches: the activation encoder, the packed product,
/// and the two routed products whose grid is the sorted rows.
const ENCODED_OPERATORS: &[&str] = &["nvfp4_quantize", "gather_gemm_nvfp4", "route_gemm_nvfp4"];

/// What an encoded deployment asks the store for, and which row counts.
#[derive(Clone, Copy, Default)]
struct Encoded {
    /// Whether the checkpoint declares an encoded representation.
    declared: bool,
    /// The prompt's row count, which the prefill's packed product is compiled for.
    tokens: Option<u64>,
    /// One row, which a continuation's step is compiled for, when one will run.
    generated: Option<u64>,
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
    /// Weights published as packed E2M1, which is what an NVFP4 checkpoint's projections arrive as.
    packed: usize,
    /// Weights published in a full-precision element type.
    dense: usize,
    /// Bytes published as packed E2M1: the measure that survives the loader fusing 128 experts into
    /// one stack, because a fused stack is one publish of many tensors' bytes.
    packed_bytes: u64,
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
            packed: 0,
            dense: 0,
            packed_bytes: 0,
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
        // What arrived, by the type it arrived in: a packed projection is published as the packed
        // E2M1 bytes the programs read, and everything kept is published full precision. A loader
        // that fell back would show up here as a packed tensor that never arrived packed, which is
        // the one thing the checkpoint's own index cannot tell a run.
        match dtype {
            DType::U8 => {
                self.packed += 1;
                self.packed_bytes += bytes.len() as u64;
            }
            DType::BF16 | DType::F16 | DType::F32 => self.dense += 1,
            _ => {}
        }
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

/// Upload `values` as an f32 plane of `shape`.
fn upload_f32(
    stream: &platform::Stream,
    arena: &Arena,
    shape: Shape,
    values: &[f32],
) -> Result<ops::TensorView> {
    let tensor = arena.tensor(DType::F32, shape)?;
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

/// Upload `rows × columns` `f32` values as a BF16 tensor, rounding to nearest even.
fn upload_bf16(
    stream: &platform::Stream,
    arena: &Arena,
    values: &[f32],
    rows: usize,
    columns: usize,
) -> Result<TensorView> {
    let bits: Vec<u16> = values.iter().copied().map(f32_to_bf16).collect();
    let bytes: Vec<u8> = bits.iter().flat_map(|bits| bits.to_le_bytes()).collect();
    let tensor = arena.tensor(DType::BF16, Shape::new(&[rows as u64, columns as u64]))?;
    // SAFETY: `bytes` outlives the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            tensor.ptr.address(),
            bytes.as_ptr() as u64,
            bytes.len(),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(tensor)
}

/// An `f32` as BF16, rounding to nearest even: the same rounding the device's own conversion does,
/// so a row uploaded from the host is the row a bf16 stack holds.
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    // NaN stays NaN and infinities stay themselves; a finite value rounds on the 16 kept bits.
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

/// The state a run starts from when `--from` names a reference stage: the reference's own
/// `hidden_NN`, uploaded as this run's input.
///
/// A layer run on the reference's state answers about *that layer*: the prefix's error is not in
/// the rows, so a row that disagrees here is the layer's own arithmetic. The cost is that the input
/// carries one bf16 rounding of its own, about `2**-8` relative, which is why the caller of this
/// function prints the floor it measures against.
/// Layer `layer`'s post-attention state, summed from the reference's own two hooks.
///
/// The router a forward runs is fed `residual + attention`, not the layer's input rows: a cut that
/// normalized the input would answer about a router no forward executes, and the two are
/// indistinguishable by shape - both are `[tokens, hidden]` states. The reference records both
/// halves (`hidden_{layer}` before the attention and `hidden_{layer+1}_attended` its output), so the
/// sum is this layer's state up to one rounding to the model's dtype, which is the same rounding a
/// reference computing the sum from its own hooks makes.
fn post_attention_state(
    stream: &platform::Stream,
    arena: &Arena,
    args: &Args,
    layer: usize,
    batch: &TokenBatch,
    plan: &Plan,
) -> Result<TensorView> {
    let Some(dir) = args.reference.as_deref() else {
        return Err(Failure::message(
            "a routing or score cut runs on the reference's post-attention state, which the run \
             cannot rebuild: pass --reference <dir>",
        ));
    };
    // A run that named no `--from` starts at the embedding, and the state's residual half is the
    // stage the layer would still be handed: `hidden_layer` is that stage in both cases, so the cut
    // reads it from the reference either way rather than from the run's rows.
    let from = args.from.unwrap_or(layer);
    if from != layer {
        return Err(Failure::message(format!(
            "--from {from} runs layer {from}, so it has no state for layer {layer}"
        )));
    }
    let rows = tokens_usize(batch);
    let width = usize::try_from(plan.hidden).unwrap_or(0);
    let residual = read_stage(dir, layer, rows, width)?;
    // The attention's own cut inside this layer carries the next index: `hidden_NN_attended` is the
    // attended output of layer `NN - 1`.
    let attended = read_stage_named(
        dir,
        &format!("hidden_{:02}_attended.npy", layer + 1),
        rows,
        width,
    )?;
    let state: Vec<f32> = residual
        .iter()
        .zip(attended.iter())
        .map(|(left, right)| left + right)
        .collect();
    upload_bf16(stream, arena, &state, rows, width)
}

/// One reference stage's rows, checked against the run's shape.
fn read_stage(dir: &Path, index: usize, rows: usize, width: usize) -> Result<Vec<f32>> {
    read_stage_named(dir, &format!("hidden_{index:02}.npy"), rows, width)
}

/// One named reference stage's rows, checked against the run's shape.
fn read_stage_named(dir: &Path, name: &str, rows: usize, width: usize) -> Result<Vec<f32>> {
    let path = dir.join(name);
    let (values, their_rows, their_width) = read_npy(&path)?;
    if their_rows != rows || their_width != width {
        return Err(Failure::message(format!(
            "{} is {their_rows} x {their_width}, the run's rows are {rows} x {width}",
            path.display()
        )));
    }
    Ok(values)
}

/// Print the gate's raw logits and write them for the comparator.
fn report_scores(
    args: &Args,
    layer: usize,
    rows: usize,
    experts: usize,
    values: &[f32],
) -> Result<()> {
    println!(
        "scores: layer {layer}: rms {:.6e} over every row, {experts} experts per row",
        rms(values)
    );
    for row in 0..rows.min(2) {
        let scores = &values[row * experts..(row + 1) * experts];
        let mut ranked: Vec<(f32, usize)> = scores
            .iter()
            .copied()
            .enumerate()
            .map(|(expert, value)| (value, expert))
            .collect();
        ranked.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.1.cmp(&right.1))
        });
        let top: Vec<String> = ranked
            .iter()
            .take(8)
            .map(|(value, expert)| format!("{expert} {value:.4}"))
            .collect();
        println!(
            "scores: layer {layer} row {row:2}: rms {:.6e}, top 8 [{}]",
            rms(scores),
            top.join(", ")
        );
    }
    let Some(dir) = args.dump.as_deref() else {
        println!("scores: pass --dump to write these for the comparator");
        return Ok(());
    };
    let path = dir.join(format!("hidden_{:02}_gate_scores.npy", layer + 1));
    write_npy(&path, values, rows, experts)?;
    println!("scores: {} (the gate's logits, f32)", path.display());
    Ok(())
}

fn reference_rows(
    stream: &platform::Stream,
    arena: &Arena,
    dir: &Path,
    index: usize,
    rows: usize,
    width: usize,
) -> Result<TensorView> {
    let path = dir.join(format!("hidden_{index:02}.npy"));
    let (values, their_rows, their_width) = read_npy(&path)?;
    if their_rows != rows || their_width != width {
        return Err(Failure::message(format!(
            "{} is {their_rows} x {their_width}, the run's input is {rows} x {width}",
            path.display()
        )));
    }
    upload_bf16(stream, arena, &values, rows, width)
}

/// Write one layer's routing, named as the reference's hook names it, and print what it chose.
///
/// The ids go to disk as **integers**: a float file would hand the comparator a metric in which
/// `17` against `18` is "0.8 % off", which is meaningless for an index and would hide a real
/// mismatch inside a relative pass. The weights go as `f32` — the tensor's own type — and are
/// compared as floats against the reference's bf16-widened copy, whose rounding is their floor.
fn report_routing(
    args: &Args,
    layer: usize,
    rows: usize,
    top: usize,
    ids: &[i32],
    weights: &[f32],
) -> Result<()> {
    for row in 0..rows {
        let chosen = &ids[row * top..(row + 1) * top];
        let scored = &weights[row * top..(row + 1) * top];
        println!(
            "routing: layer {layer} row {row:2}: ids {chosen:?} weights rms {:.6}",
            rms(scored)
        );
    }
    let Some(dir) = args.dump.as_deref() else {
        println!(
            "routing: layer {layer} chose these; pass --dump to write them for the comparator"
        );
        return Ok(());
    };
    let index = layer + 1;
    let ids_path = dir.join(format!("hidden_{index:02}_router_ids.npy"));
    let weights_path = dir.join(format!("hidden_{index:02}_router_weights.npy"));
    write_npy_i32(&ids_path, ids, rows, top)?;
    write_npy(&weights_path, weights, rows, top)?;
    println!(
        "dump: {} is layer {layer}'s router ids (<i4) and {} its weights (<f4)",
        ids_path.display(),
        weights_path.display()
    );
    Ok(())
}

/// Write an `i32` C-ordered `.npy`: the same container as [`write_npy`] with `'<i4'`.
fn write_npy_i32(
    path: &std::path::Path,
    values: &[i32],
    rows: usize,
    columns: usize,
) -> Result<()> {
    let mut header =
        format!("{{'descr': '<i4', 'fortran_order': False, 'shape': ({rows}, {columns}), }}");
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

/// Download `elements` `i32` values.
fn download_i32(stream: &platform::Stream, view: TensorView, elements: usize) -> Result<Vec<i32>> {
    let mut values = vec![0_i32; elements];
    // SAFETY: `values` is written by the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            values.as_mut_ptr() as u64,
            view.ptr.address(),
            std::mem::size_of_val(&values[..]),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(values)
}
