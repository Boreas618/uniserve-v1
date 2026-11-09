//! MiniMax-H3's diffusion transformer on a GPU, one stage of it at a time.
//!
//! A runner builds a device, an allocator, an arena and a `Generation` of loaded kernels before a
//! model computes anything, and the worker owns all four. This example builds exactly those four
//! by hand so the stack can be driven with nothing of the worker in the way, and it stages the run
//! so that the first failure names one thing:
//!
//! ```text
//! --stage weights  no device: every parameter this stack layer reads, against the index and the
//!                  checkpoint's own tensors
//! --stage block    one stack layer on a device: its rotary, its attention, its feed-forward and its
//!                  modulation, over a request's packed rows
//! --stage denoise  the whole sampler on a device: the draw, `--evaluations` schedule evaluations
//!                  through the solver, and the trajectory, one file per evaluation
//! ```
//!
//! **The stack is loaded with `--layers` layers, and the count must be even.** `Transformer::forward`
//! alternates two buffers and refuses an odd stack, because the caller's buffer would not hold the
//! result; `--layers 2` is the smallest honest "one block" and is the default. It is the same arithmetic
//! as a fifty-layer stack, which is why it can stand in for one.
//!
//! All but `weights` need a program store: `--store <path>` or `UNISERVE_STORE`, holding a committed
//! module for every descriptor the layer builds. The store is read, never written; which descriptor a
//! launch binds is `ops`' business, so this harness installs every committed module of the operators
//! the stack uses and lets the launch name a missing one rather than keeping a second copy of the keys
//! here.
//!
//! **What it writes.** `<out>/denoiser/stack_in.h3t` is the input the run used and
//! `<out>/denoiser/stack_out.h3t` what the stack produced, both in the container
//! `tools/dump_tables.py` writes and `compare_blocks` reads. A reference producer reads
//! `stack_in`, runs the same layer stack of today's implementation on it, and writes `stack_out` under
//! the same name in its own tree; the two trees then pair cold.
//!
//! Stage `denoise` writes a trajectory instead: `<out>/denoiser/text_features.h3t` is the prefix
//! source both sides read (synthetic and fixed, so the stage is about the sampler and not about an
//! encoder), `latents_<modality>_e<i>.h3t` is one evaluation state - `e0` the draw and `eN` the sample
//! after the last of the `N` evaluations - and `case.json` names the request, the seed, the count and
//! the checkpoint's own schedule, so the reference producer runs the same first `N` evaluations of it.
//! Every stage writes `walk.txt`: one `operator<TAB>key` per launch, which `tools/walk_gaps.py` reads
//! against the store's list.

use minimax_h3::VideoTask;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::attention::{Factors, Visible};
use minimax_h3::checkpoint::Dit;
use minimax_h3::config::AttentionKind;
use minimax_h3::config::Config;
use minimax_h3::denoiser::{Denoiser, Layouts, denoise_arena};
use minimax_h3::modulation::{self, AdaLnCache, Groups, Projections};
use minimax_h3::packing;
use minimax_h3::partition::Partition;
use minimax_h3::schedule::{Modality, Schedule};
use minimax_h3::transformer::{Transformer, rotated_axes};
// The block's capability methods: `draw_noise`, `layout`, `state_tables` and the rest are the
// trait's, and the stage that drives the whole block calls them the way the worker does.
use nn::capability::Denoiser as _;
use nn::capability::VideoRequest;
use nn::{
    Arena, Backing, Checkpoint, DType, DevicePtr, Forward, RetainedWeights, Shape, TensorView,
    WeightKey, WeightLoader,
};

/// The modulation table's width in `hidden_size` units: three tags of six vectors.
const ADALN_COLUMNS: u64 = 18;

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

impl Failure {
    fn message(text: impl Into<String>) -> Self {
        Self::Message(text.into())
    }
}

impl From<minimax_h3::Error> for Failure {
    fn from(error: minimax_h3::Error) -> Self {
        Failure::Message(error.to_string())
    }
}

impl From<nn::Error> for Failure {
    fn from(error: nn::Error) -> Self {
        Failure::Message(error.to_string())
    }
}

impl From<platform::Error> for Failure {
    fn from(error: platform::Error) -> Self {
        Failure::Message(error.to_string())
    }
}

type Result<T> = std::result::Result<T, Failure>;

/// The two stages of the run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Weights,
    Block,
    Denoise,
}

struct Args {
    stage: Stage,
    root: PathBuf,
    store: Option<PathBuf>,
    dit: Dit,
    device: i32,
    layers: u32,
    frames: u32,
    width: u32,
    height: u32,
    text_tokens: u32,
    arena_gib: u64,
    /// The sampling key of stage `denoise`: one seed for the whole request's stream.
    seed: u64,
    /// The deployment's stated element type; `None` is `dtype = "auto"`, the checkpoint's own.
    ///
    /// This is the loader's policy and not a harness convenience: `WeightLoader` converts once, at
    /// load, and every block takes its type from the weights it was published (specs/worker.md §1).
    /// Stating it here is how the conversion path - a deployment whose `dtype` differs from its
    /// export - reaches the model through the same boundary a served request uses.
    dtype: Option<DType>,
    /// How many of the checkpoint schedule's evaluations stage `denoise` runs; `None` is all of them.
    evaluations: Option<u32>,
    /// The attention kind the case expects; `None` is `auto`, whatever the export states.
    ///
    /// The kind is a property of the checkpoint's `fastvideo_inference.json` and no flag can change
    /// it, so stating it here is a case's *expectation*: a case that says sparse and a checkpoint
    /// that says dense is refused by name instead of producing a dense page that would be read as a
    /// sparse one.
    attention: Option<AttentionKind>,
    out: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            stage: Stage::Weights,
            root: PathBuf::new(),
            store: std::env::var_os("UNISERVE_STORE").map(PathBuf::from),
            dit: Dit::Base,
            device: 0,
            layers: 2,
            frames: 124,
            width: 1344,
            height: 768,
            text_tokens: 64,
            arena_gib: 12,
            seed: 0,
            dtype: None,
            attention: None,
            evaluations: None,
            out: PathBuf::new(),
        };
        let mut words = std::env::args().skip(1);
        while let Some(word) = words.next() {
            let mut value = |name: &str| -> Result<String> {
                words
                    .next()
                    .ok_or_else(|| Failure::message(format!("{name} needs a value")))
            };
            match word.as_str() {
                "--stage" => {
                    args.stage = match value("--stage")?.as_str() {
                        "weights" => Stage::Weights,
                        "block" => Stage::Block,
                        "denoise" => Stage::Denoise,
                        other => {
                            return Err(Failure::message(format!(
                                "unknown stage `{other}`; expected weights, block or denoise"
                            )));
                        }
                    };
                }
                "--root" => args.root = PathBuf::from(value("--root")?),
                "--store" => args.store = Some(PathBuf::from(value("--store")?)),
                "--dit" => {
                    args.dit = match value("--dit")?.as_str() {
                        "base" => Dit::Base,
                        "ref" | "reference" => Dit::Reference,
                        other => {
                            return Err(Failure::message(format!(
                                "unknown network `{other}`; expected base or ref"
                            )));
                        }
                    };
                }
                "--device" => {
                    let number = value("--device")?;
                    args.device = number.parse().map_err(|_| {
                        Failure::message(format!("--device takes a number, not `{number}`"))
                    })?;
                }
                "--layers" => {
                    let number = value("--layers")?;
                    args.layers = number.parse().map_err(|_| {
                        Failure::message(format!("--layers takes a number, not `{number}`"))
                    })?;
                }
                "--frames" => {
                    let number = value("--frames")?;
                    args.frames = number.parse().map_err(|_| {
                        Failure::message(format!("--frames takes a number, not `{number}`"))
                    })?;
                }
                "--width" => {
                    let number = value("--width")?;
                    args.width = number.parse().map_err(|_| {
                        Failure::message(format!("--width takes a number, not `{number}`"))
                    })?;
                }
                "--height" => {
                    let number = value("--height")?;
                    args.height = number.parse().map_err(|_| {
                        Failure::message(format!("--height takes a number, not `{number}`"))
                    })?;
                }
                "--text-tokens" => {
                    let number = value("--text-tokens")?;
                    args.text_tokens = number.parse().map_err(|_| {
                        Failure::message(format!("--text-tokens takes a number, not `{number}`"))
                    })?;
                }
                "--arena-gib" => {
                    let number = value("--arena-gib")?;
                    args.arena_gib = number.parse().map_err(|_| {
                        Failure::message(format!("--arena-gib takes whole GiB, not `{number}`"))
                    })?;
                }
                "--seed" => {
                    let number = value("--seed")?;
                    args.seed = number.parse().map_err(|_| {
                        Failure::message(format!("--seed takes a whole number, not `{number}`"))
                    })?;
                }
                "--evaluations" => {
                    let number = value("--evaluations")?;
                    let count: u32 = number.parse().map_err(|_| {
                        Failure::message(format!(
                            "--evaluations takes a whole number, not `{number}`"
                        ))
                    })?;
                    args.evaluations = Some(count);
                }
                "--dtype" => {
                    args.dtype = match value("--dtype")?.as_str() {
                        "auto" => None,
                        "f32" | "float32" => Some(DType::F32),
                        "bf16" | "bfloat16" => Some(DType::BF16),
                        "f16" | "float16" => Some(DType::F16),
                        other => {
                            return Err(Failure::message(format!(
                                "unknown dtype `{other}`; expected auto, bf16, f32 or f16"
                            )));
                        }
                    };
                }
                "--attention" => {
                    // A case states the attention kind it means to run. The kind is the
                    // checkpoint's, so the flag cannot select one; what it can do is refuse a case
                    // whose expectation the export does not meet - a sparse case served by a dense
                    // export would otherwise produce a dense page that reads as a sparse one.
                    args.attention = match value("--attention")?.as_str() {
                        "auto" => None,
                        "dense" => Some(AttentionKind::Dense),
                        "sparse" => Some(AttentionKind::Sparse {
                            tile: 0,
                            sparsity: 0.0,
                            reference_keep: None,
                        }),
                        other => {
                            return Err(Failure::message(format!(
                                "unknown attention `{other}`; expected auto, dense or sparse"
                            )));
                        }
                    };
                }
                "--out" => args.out = PathBuf::from(value("--out")?),
                "--help" | "-h" => {
                    return Err(Failure::message(
                        "dit_layer --stage weights|block|denoise [--root PATH] [--store PATH] \
                         [--dit base|ref] [--device N] [--layers N] [--frames N] [--width N] \
                         [--height N] [--text-tokens N] [--arena-gib N] [--seed N] \
                         [--evaluations N] [--dtype auto|bf16|f32|f16] \
                         [--attention auto|dense|sparse] [--out DIR]",
                    ));
                }
                other => {
                    return Err(Failure::message(format!(
                        "unknown argument `{other}`; try --help"
                    )));
                }
            }
        }
        if args.out.as_os_str().is_empty() {
            eprintln!("name the output directory with --out");
            std::process::exit(2);
        }
        if args.root.as_os_str().is_empty() {
            args.root = PathBuf::from(minimax_h3::REPO_ID);
        }
        args.root = nn::hub::model_dir(&args.root.to_string_lossy()).unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(2)
        });
        if args.layers < 2 || !args.layers.is_multiple_of(2) {
            return Err(Failure::message(format!(
                "the stack is loaded with {} layers; `Transformer::forward` alternates two buffers \
                 and refuses an odd count, so a two-layer stack is the smallest that returns the \
                 caller's buffer",
                args.layers
            )));
        }
        Ok(args)
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    match args.stage {
        Stage::Weights => stage_weights(&args),
        Stage::Block => stage_block(&args),
        Stage::Denoise => stage_denoise(&args),
    }
}

/// The network's configuration, with the stack loaded at `--layers` layers.
fn configuration(args: &Args) -> Result<(minimax_h3::config::DenoiserConfig, Partition)> {
    let config = Config::read(&args.root, None)?;
    let mut denoiser = config.denoisers.get(&args.dit).cloned().ok_or_else(|| {
        Failure::message(format!(
            "the export at {} holds no {} network; it holds {:?}",
            args.root.display(),
            args.dit.directory(),
            config.denoisers.keys().collect::<Vec<_>>()
        ))
    })?;
    denoiser.transformer.num_hidden_layers = args.layers;
    Ok((denoiser, Partition::single()))
}

/// Stage 1: the mapping, with no device.
///
/// Two checks in one pass. The first is the name level - every tensor the layer and its modulation
/// cache ask for, against the index the export ships, which is the whole of what an index-only host
/// can answer. The second is the tensor level: the same load the device path performs, into a store
/// that validates each tensor's element type and shape as the loader hands it over, so a transposed
/// or truncated weight names itself instead of surfacing as a wrong number.
fn stage_weights(args: &Args) -> Result<()> {
    let (denoiser, partition) = configuration(args)?;
    let config = &denoiser.transformer;

    // The name level first, and from the index alone: this half runs on a host that holds the
    // export's metadata and none of its tensors, which is the only check such a host can answer.
    let index = index_names(&args.root, args.dit)?;
    let asked: Vec<String> = Transformer::keys(config, denoiser.attention)
        .into_iter()
        .chain(AdaLnCache::keys(config))
        .collect();
    let missing: Vec<&String> = asked.iter().filter(|name| !index.contains(name)).collect();
    let unread: Vec<&String> = index.iter().filter(|name| !asked.contains(name)).collect();
    println!(
        "names: {} asked, {} in the index, {} missing, {} unread",
        asked.len(),
        index.len(),
        missing.len(),
        unread.len()
    );
    for name in &missing {
        println!("  missing {name}");
    }
    // The index holds every layer of the export while this load holds `--layers`, so the unread
    // names of the layers above the loaded ones are counted rather than listed.
    let beyond: Vec<&&String> = unread
        .iter()
        .filter(|name| !name.starts_with("transformer_blocks."))
        .collect();
    for name in &beyond {
        println!("  unread {name}");
    }
    if !missing.is_empty() {
        return Err(Failure::message(format!(
            "{} tensors the stack asks for are not in the export's index; the first is `{}`",
            missing.len(),
            missing[0]
        )));
    }

    // The tensor level: the same load the device path performs, into a store that checks each
    // tensor's element type and shape as the loader hands it over, so a transposed or truncated
    // weight names itself instead of surfacing as a wrong number.
    let checkpoint = match Checkpoint::open(&args.root) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            println!("tensors: not read ({error})");
            return Ok(());
        }
    };
    let dit = checkpoint.component(args.dit.directory())?;
    let mut store = HostWeights::default();
    let layouts = Layouts::new(denoiser.clone(), partition.arrangement(&denoiser)?)?;
    let rows = layouts.rows_bound()?;
    // A sparse deployment states its key-tile bound from its own layouts, exactly as the model's
    // loading boundary does (`denoiser.rs`): a zero here is a statement that the deployment's
    // compression reduces over nothing, and `VsaTiles` carries it into the descriptor a plan reads.
    let tiles = match denoiser.attention {
        AttentionKind::Sparse { .. } => layouts.tiles_bound()?,
        AttentionKind::Dense => 0,
    };
    {
        let mut loader = WeightLoader::new(dit, &mut store, args.dtype, None);
        let _stack = Transformer::load(
            config,
            denoiser.attention,
            rows,
            tiles,
            &partition,
            &mut loader,
            None,
        )?;
        let projections = Projections::load(config, &mut loader)?;
        println!(
            "loaded: {} stack layers, {} layers of modulation, hidden {}, {} rows",
            config.num_hidden_layers,
            projections.layers(),
            config.hidden_size,
            rows
        );
    }
    println!(
        "tensors: {} published, {} bytes, checked for element type and shape",
        store.published.len(),
        store.bytes
    );
    for (dtype, (count, bytes)) in &store.by_dtype {
        println!("  {dtype}: {count} tensors, {bytes} bytes");
    }
    Ok(())
}

/// The export's index, as the set of tensor names one network of it declares.
///
/// Read as the JSON it is rather than through `Checkpoint::open`, because opening a checkpoint
/// validates that every file it maps is present: a host that holds an export's metadata and none of
/// its tensors can still answer whether a name is there.
fn index_names(root: &Path, dit: Dit) -> Result<Vec<String>> {
    let directory = root.join(dit.directory());
    let entries = std::fs::read_dir(&directory)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".index.json"))
        })
        .collect();
    files.sort();
    let path = files.first().ok_or_else(|| {
        Failure::message(format!(
            "{} holds no `*.index.json`: nothing states which tensors the network has",
            directory.display()
        ))
    })?;
    let text = std::fs::read_to_string(path)
        .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?;
    let document: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?;
    let map = document
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Failure::message(format!("{}: no `weight_map` object", path.display())))?;
    Ok(map.keys().cloned().collect())
}

/// Stage 3: the sampler, over one request's whole trajectory.
///
/// The subject is the noise, the schedule, the solver and the step table, so nothing else is in the
/// path: the prefix source is a file this harness writes for both sides (`text_features.h3t`,
/// synthetic and stated), no encoder or conditioner runs, and the case's count is a *prefix* of the
/// checkpoint's own schedule rather than a schedule of its own.
///
/// **`e0` is the draw and `eN` the sample after the last of the `N` evaluations.** The state before
/// each evaluation is on disk, counted from the draw, so a wrong draw is a tensor on the page rather
/// than a plausible trajectory that happens to start somewhere else.
fn stage_denoise(args: &Args) -> Result<()> {
    let store_root = args.store.clone().ok_or_else(|| {
        Failure::message(
            "stage `denoise` needs a program store: pass --store <path> or set UNISERVE_STORE",
        )
    })?;
    let (denoiser_config, partition) = configuration(args)?;
    let hidden = u64::from(denoiser_config.transformer.hidden_size);

    // 1. The device and the kernels first: a driver that cannot give a device, or a store missing an
    //    operator, fails before a byte of the checkpoint is read.
    let device = platform::Device::new(platform::DeviceId(args.device))?;
    let _guard = device.enter()?;
    let stream = device.stream()?;
    let generation = load_generation(&device, &store_root)?;
    check_sparse_store(&generation, denoiser_config.attention)?;

    // 2. The case's request, its layout and its packing: the deployment's own rule, so the row count,
    //    the padding and the coordinates are the ones a served request would have.
    let arrangement = partition.arrangement(&denoiser_config)?;
    let layouts = Layouts::new(denoiser_config.clone(), arrangement)?;
    let request = VideoRequest {
        width: args.width,
        height: args.height,
        frames: args.frames,
        fps: 24,
        first_frames: 0,
        last_frames: 0,
        references: 0,
        text_tokens: args.text_tokens,
        conditions: &[],
        vision_spans: &[],
        condition_rows: None,
    };
    let (layout, packing) = layouts.layout(&request)?;
    let frames = packing::align_num_frames(request.frames);
    let canvas = layouts.canvas(&request)?;
    // The attended row count is the *dense* packing's. A sparse deployment reads its tiles instead,
    // so the summary below says which of the two it is: a dense number printed over a sparse run is
    // a page that cannot be read, which is the whole reason the kind is printed at all.
    let tables = packing::dense_tables(&packing, frames, &canvas, request.text_tokens, &[], &[])?;
    let tokens = u64::from(packing.padded_tokens);
    println!(
        "request: {}x{}, {} frames, {} rows packed of {} padded, {}",
        args.width,
        args.height,
        frames,
        packing.video_rows + packing.audio_rows,
        tokens,
        match denoiser_config.attention {
            AttentionKind::Dense => format!("{} attended", tables.used),
            AttentionKind::Sparse { tile, .. } => {
                format!(
                    "sparse over {} tiles of {tile} rows",
                    tokens / u64::from(tile)
                )
            }
        }
    );

    // 3. The whole block in one loader: the stack, the projections, the AdaLN cache and the solver,
    //    which is what the served path builds.
    let checkpoint = Checkpoint::open(&args.root)?;
    let dit = checkpoint.component(args.dit.directory())?;
    let mut weights = DeviceWeights::new(&device, 0)?;
    let rows = layouts.rows_bound()?;
    let block = {
        let mut loader = WeightLoader::new(dit, &mut weights, args.dtype, None);
        Denoiser::load(
            denoiser_config.clone(),
            arrangement,
            rows,
            &partition,
            &mut loader,
            None,
        )?
    };
    let dtype = block.dtype();
    println!(
        "loaded: {} stack layers, hidden {hidden}, {rows} packed rows, {dtype}",
        denoiser_config.transformer.num_hidden_layers
    );
    // Which attention kind this page is produced by. A sparse run and a dense run write files of
    // the same shapes, so a page that does not print its kind cannot be told from the other one -
    // and the kind is a property of the checkpoint (`config/mod.rs` reads it from the export's own
    // `fastvideo_inference.json`), so a run that prints it also records which export it served.
    let attention = denoiser_config.attention;
    let attention_kind = match attention {
        AttentionKind::Dense => "dense",
        AttentionKind::Sparse { .. } => "sparse",
    };
    if let Some(expected) = args.attention
        && std::mem::discriminant(&expected) != std::mem::discriminant(&attention)
    {
        return Err(Failure::message(format!(
            "the case asks for the {} kind and this export states {attention_kind} \
             (`fastvideo_inference.json` decides it, not a flag)",
            match expected {
                AttentionKind::Dense => "dense",
                AttentionKind::Sparse { .. } => "sparse",
            }
        )));
    }
    match attention {
        AttentionKind::Dense => println!("attention: dense"),
        AttentionKind::Sparse {
            tile,
            sparsity,
            reference_keep,
        } => {
            let total = packing.padded_tokens / tile;
            // The extent is the model's own rule, and its alignment is a whole number of tiles:
            // `1` here was the harness's, and it refused every sparse case at its own summary
            // print ("the 1-row alignment is not a whole number of 64-row tiles") before any
            // launch. The alignment moves no tile count, so the printed number is unchanged.
            let video = packing::segment_extent(
                frames,
                &canvas,
                packing.text_rows,
                packing.condition_rows,
                tile,
                tile,
            )
            .map(|extent| extent.video_tiles)
            .map_err(|error| Failure::message(format!("a sparse layout of this case: {error}")))?;
            let keep = packing::kept_tiles(sparsity, video);
            println!(
                "attention: sparse, tiles {tile}, sparsity {sparsity:.2}, kept {keep} of {video} \
                 video tiles of {total}{}",
                match reference_keep {
                    Some(rate) => format!(", reference keep {rate:.2}"),
                    None => String::new(),
                }
            );
        }
    }
    // The stage's files are the prefix source and the trajectory, and it writes them in the
    // checkpoint's own element type: a conversion there would be a second representation of the same
    // numbers on one side of the comparison only. `dtype` is the block's, which is the deployment's -
    // `dtype = "auto"` follows the checkpoint and the loader converts once - so the prefix source is
    // written in the type the stack actually computes in and the case names it for the reference.

    // 4. The block's own view of the request, and the arena. The plan is the model's statement of one
    //    evaluation's temporaries; the harness adds what only it holds - the two kinds of table, the
    //    prefix source and the latents - and reserves the sum. `Denoiser::layout` is the same
    //    `Layouts::cover` the packing above used, so there is one layout for this request.
    let shape = <Denoiser as nn::capability::Denoiser>::layout(&block, &request);
    // The harness derives the packing from `Layouts` and the block answers the layout it will
    // compute in: two statements of one arrangement, from one function today. If they ever diverge,
    // the trajectory would be written against one arrangement and computed in another - the class a
    // dropped condition hid inside - so they are compared rather than asserted in a comment.
    if layout != shape {
        return Err(Failure::message(format!(
            "the harness's layout {layout:?} is not the block's {shape:?}"
        )));
    }
    let planned: u64 = block
        .denoise_arena(&shape)?
        .iter()
        .map(|buffer| buffer.bytes())
        .sum();
    let constant_specs = block.constant_tables();
    let state_specs = block.state_tables(&shape);
    let video = shape.video;
    let video_elements = u64::from(video.channels)
        * u64::from(video.frames)
        * u64::from(video.height)
        * u64::from(video.width);
    let text_rows = u64::from(shape.text_rows);
    let audio_rows = state_specs
        .iter()
        .find(|spec| spec.name == "audio_indices")
        .map(|spec| spec.shape.dim(0));
    let audio_width = u64::from(denoiser_config.transformer.audio_channels);
    let needed = planned
        + table_bytes(&constant_specs)
        + table_bytes(&state_specs)
        + (text_rows + u64::from(shape.condition_rows) + 1) * hidden * 2
        + video_elements * 4
        + audio_rows.unwrap_or(0) * audio_width * 4;
    // The schedule the checkpoint states, and how much of it this case runs - before the allocation,
    // because it is host data and the case states it. The case may shorten the schedule and may not
    // change it: the steps are the schedule's own, each naming its index and the count of the grid
    // it belongs to.
    let schedule = block.schedule();
    let steps = schedule.steps(Modality::Video)?;
    let num_steps = u32::try_from(steps.len()).unwrap_or(u32::MAX);
    if num_steps != block.evaluations() {
        return Err(Failure::message(format!(
            "the schedule states {num_steps} evaluations and the block was loaded for {}",
            block.evaluations()
        )));
    }
    let evaluations = args.evaluations.unwrap_or(num_steps);
    if evaluations > num_steps {
        return Err(Failure::message(format!(
            "the case asks {evaluations} evaluations and the checkpoint's {} grid has {num_steps}",
            schedule.spec().grid.kind()
        )));
    }
    if evaluations == 0 {
        return Err(Failure::message(
            "the case asks no evaluations: `e0` would be its only state, and a stage that draws \
             and stops says nothing about the schedule",
        ));
    }
    println!(
        "schedule: {} grid, {num_steps} evaluations, running {evaluations}, seed {}",
        schedule.spec().grid.kind(),
        args.seed
    );

    // The prefix source's row count, stated before the case that carries it: it is
    // `text_rows + condition_rows + 1` (the prompt, one per packed condition row, and the
    // zero row every padding row gathers), and it is derivable the moment the layout is.
    let prefix_rows = text_rows + u64::from(shape.condition_rows) + 1;

    // The case, before the allocation and before the trajectory. It names the run: a stop in the
    // arena, in the draw or inside the loop is read against it, and the reference producer can be
    // given the same case whatever this side got as far as. Everything the case states is host data
    // - the layout's extents, the schedule's grid, the seed and the counts - so nothing here waits
    // for a device to say what the run would have been.
    let grid = schedule.spec().grid.kind();
    let directory = args.out.join("denoiser");
    std::fs::create_dir_all(&directory)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    // A stage reuses one output directory across tips, so a trajectory file the last run wrote is
    // still there when this one stops earlier than it did - and a comparison would then pair this
    // run's draw with that run's evaluations and report a page no run produced. The files this
    // stage writes are cleared first, and the count is printed so a reader can see a stale tree
    // being cleaned rather than wonder where its numbers went. Everything else in the directory
    // (`case.json`, `walk.txt`, the prefix source) is rewritten by this run anyway.
    let stale = clear_trajectory(&directory)?;
    if stale > 0 {
        println!("cleared {stale} trajectory files from the previous run");
    }
    // The case's `schedule` is the checkpoint's own word for its grid - `ScheduleGrid::kind()`,
    // which is what the checkpoint's config states - and not the reference's class name
    // (`UniformGrid`, `BlockGrid`, `PddGrid`). One vocabulary, and it is the one the model reads:
    // the field's job is to make a FastH3 contract read as another grid a refusal rather than a
    // page, and a spelling difference between two tools would be the same failure with none of the
    // meaning.
    // The latent buffer's extents, which the case states: the same four numbers the arena's video
    // tensor is built with below, taken from the layout so the case does not need an allocation.
    let video_shape = format!(
        "[{}, {}, {}, {}]",
        video.channels, video.frames, video.height, video.width
    );
    let case = format!(
        "{{\"task\": \"t2va\", \"width\": {}, \"height\": {}, \"frames\": {}, \"text_tokens\": {}, \
         \"text_rows\": {text_rows}, \"prefix_rows\": {prefix_rows}, \"condition_rows\": {}, \"layers\": {}, \"hidden\": {hidden}, \
         \"seed\": {}, \"evaluations\": {evaluations}, \"num_steps\": {num_steps}, \
         \"schedule\": {grid:?}, \"attention\": \"{attention_kind}\", \
         \"text_features\": \"synthetic-fixed\", \"sample_dtype\": \"f32\", \"dtype\": \"{dtype}\", \
         \"video_shape\": {}}}\n",
        args.width,
        args.height,
        frames,
        request.text_tokens,
        shape.condition_rows,
        denoiser_config.transformer.num_hidden_layers,
        args.seed,
        video_shape,
    );
    // A case is JSON that a comparator reads, so it is parsed before it is written: a misplaced
    // comma would otherwise cost a device run to discover, and the refusal names the file.
    serde_json::from_str::<serde_json::Value>(&case).map_err(|error| {
        Failure::message(format!(
            "the case this stage states is not JSON ({error}): {case}"
        ))
    })?;
    std::fs::write(directory.join("case.json"), case)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    // The arrangement, stated beside the case in the object the reference's producer writes too
    // (`dump_denoiser.py`, `denoiser/layout.json`). The trajectory files hold the latents, whose
    // shapes do not depend on the packed row count: two sides running different layouts write
    // files of the same shape and different trajectories, which a comparator can only report as an
    // arithmetic difference. This statement is what lets it refuse the pair instead
    // (`compare_trajectory.py`), and the line below prints the same numbers for a reader.
    // The grid and the step count are the third thing a pair can differ in: two sides can agree on
    // every layout field and still have run different schedules - a FastH3 student's `blocks` grid
    // against the base `uniform` one, or a shortened run against a whole one - and their
    // trajectories then differ everywhere for a reason that is not arithmetic. Stated here, in the
    // model's own word for the grid, so the comparator refuses that pair instead of tabulating it.
    let arrangement = format!(
        "{{\"frames\": {frames}, \"canvas\": [{}, {}], \"text_rows\": {}, \
         \"condition_rows\": {}, \"video_rows\": {}, \"audio_rows\": {}, \"packed_rows\": {}, \
         \"schedule\": {grid:?}, \"evaluations\": {evaluations}, \"num_steps\": {num_steps}}}\n",
        canvas.height,
        canvas.width,
        packing.text_rows,
        packing.condition_rows,
        packing.video_rows,
        packing.audio_rows,
        packing.padded_tokens,
    );
    // The arrangement is JSON a comparator parses, so it is parsed here first for the same reason the
    // case is: a format-string typo would otherwise surface on the run host as a comparator refusing
    // a file nobody can see, and the refusal names the statement instead.
    serde_json::from_str::<serde_json::Value>(&arrangement).map_err(|error| {
        Failure::message(format!(
            "the layout statement this stage writes is not JSON ({error}): {arrangement}"
        ))
    })?;
    std::fs::write(directory.join("layout.json"), arrangement)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    // The wording is the reference producer's own, so the two logs can be read side by side
    // (`dump_denoiser.py`, the same line): a reader comparing two arrangements should not have to
    // allow for two spellings of one statement.
    println!(
        "layout  {} packed rows = {} video + {} audio + {} text + {} condition \
         ({frames} frames, {evaluations} of {num_steps} evaluations)",
        packing.padded_tokens,
        packing.video_rows,
        packing.audio_rows,
        packing.text_rows,
        packing.condition_rows,
    );

    let arena_bytes = needed.max(args.arena_gib << 30);
    let allocator = weights.allocator();
    let reserve = usize::try_from(arena_bytes).unwrap_or(usize::MAX);
    // The two numbers a pool refusal is about, printed **before** the reservation that can fail:
    // a run that dies in the allocator otherwise leaves no statement of what it asked for, which is
    // the one line a reader needs to tell a plan that is too large from a pool that is too small.
    println!(
        "tables: {} constants, {} state; arena {:.2} GiB needed of {:.2} GiB reserved",
        constant_specs.len(),
        state_specs.len(),
        needed as f64 / f64::from(1 << 30),
        arena_bytes as f64 / f64::from(1 << 30)
    );
    allocator.reserve(platform::PoolKind::Graph, reserve)?;
    let region = allocator.allocate(platform::PoolKind::Graph, reserve)?;
    let arena = Arena::new(DevicePtr::new(region.address()), arena_bytes);
    let forward = Forward::new(&stream, &generation, &arena);

    // 5. The walk starts at the constants: the AdaLN cache's own projections are launches, and a walk
    //    that began at the forward would report a complete list while the run refused at the cache.
    let mut walk = Walk::new();
    let mut walk_file = WalkFile {
        path: args.out.join("walk.txt"),
        recorded: walk.clone(),
        written: false,
    };
    ops::capture::set_sink(Some(&mut walk));
    let sink = SinkGuard;

    // 6. The constants the block fills once after a load, then the request's own tables, whose host
    //    bytes only the model writes. The evaluation's word is one of them, and this harness is the
    //    worker that writes it before each evaluation.
    let constants = allocate_tables(&arena, &constant_specs)?;
    block.prepare_constants(&forward, &constants)?;
    let mut hosts = host_tables(&state_specs)?;
    block.prepare_state(&request, &shape, &mut hosts)?;
    let state: Vec<TensorView> = state_specs
        .iter()
        .zip(&hosts)
        .map(|(spec, bytes)| {
            let view = arena.tensor(spec.dtype, spec.shape)?;
            upload_bytes(&stream, view, bytes)?;
            Ok(view)
        })
        .collect::<Result<Vec<TensorView>>>()?;
    let step_table = state_specs
        .iter()
        .position(|spec| spec.name == "step")
        .ok_or_else(|| {
            Failure::message(format!(
                "the state tables are {:?} and the solver gathers its coefficient from one named \
                 `step`",
                state_specs.iter().map(|spec| spec.name).collect::<Vec<_>>()
            ))
        })?;

    // 7. The prefix source: one buffer of `text_rows + condition_rows + 1` rows - the prompt's
    //    rows, one row per packed condition row, then the zero row every generated and padding row
    //    gathers, which is the model's own statement of the layout (`denoiser.rs`, "The prefix
    //    source is one buffer") and derives its text capacity by subtraction. A `t2va` request has
    //    no condition rows, so the source is the synthetic prompt followed by that one zero row.
    //
    //    The text rows are fixed bytes, so both sides read the same ones and neither re-derives an
    //    encoder's output; the zero row is not synthetic data and pseudo-noise there would be wrong
    //    in every packed row that reads it. `text_features.h3t` holds the text segment alone,
    //    because the zero row is the runner's and the reference producer appends its own.
    let text_width = hidden
        * dtype
            .bytes()
            .ok_or_else(|| Failure::message(format!("{dtype} is not a whole number of bytes")))?;
    let mut source_bytes = pseudo_bytes(dtype, text_rows, hidden)?;
    source_bytes.resize(usize::try_from(prefix_rows * text_width).unwrap_or(0), 0);
    let text = arena.tensor(dtype, Shape::new(&[prefix_rows, hidden]))?;
    upload_bytes(&stream, text, &source_bytes)?;
    println!(
        "prefix: {prefix_rows} rows of {hidden} - {text_rows} text, {} condition, one zero",
        shape.condition_rows
    );

    // 8. The latents at their native extents - the raster a step patchifies, and the audio's rows -
    //    then the draw. A `t2va` request has no condition rows, so the prefix source carries none.
    let video_latents = arena.tensor(
        DType::F32,
        Shape::new(&[
            u64::from(video.channels),
            u64::from(video.frames),
            u64::from(video.height),
            u64::from(video.width),
        ]),
    )?;
    let audio_latents = match (shape.audio, audio_rows) {
        (Some(audio), Some(rows)) => {
            let declared = u64::from(audio.channels)
                * u64::from(audio.frames)
                * u64::from(audio.height)
                * u64::from(audio.width);
            if declared != rows * audio_width {
                return Err(Failure::message(format!(
                    "the layout's audio latent holds {declared} elements and the audio row table \
                     {rows} rows of {audio_width}"
                )));
            }
            Some(arena.tensor(DType::F32, Shape::new(&[rows, audio_width]))?)
        }
        (None, None) | (None, Some(0)) => None,
        (Some(_), None) => {
            return Err(Failure::message(
                "the layout draws an audio stream and names no audio row table",
            ));
        }
        (None, Some(rows)) => {
            return Err(Failure::message(format!(
                "the layout has {rows} audio rows and no audio latent to draw"
            )));
        }
    };
    let latents = nn::capability::Latents {
        video: video_latents,
        audio: audio_latents,
    };
    block.draw_noise(&forward, nn::capability::Seed(args.seed), shape, latents)?;
    stream.synchronize()?;

    // The schedule's start, applied once (`Denoiser::prepare_latents`). It is a no-op on every
    // released and distilled grid and a scaled draw on a parallel-decoding student, which is why it
    // is a call and not an assumption: a harness that skipped it would agree with this one on every
    // checkpoint shipped today and diverge silently on the first one that scales.
    block.prepare_latents(
        &forward,
        nn::capability::Conditioning {
            text,
            source: None,
            media: &[],
            images: None,
            audio: None,
            state: &state,
            step: None,
        },
        latents,
    )?;
    stream.synchronize()?;

    write_h3t(
        &directory.join("text_features.h3t"),
        &[text_rows, hidden],
        element_code(dtype)?,
        &source_bytes[..usize::try_from(text_rows * text_width).unwrap_or(0)],
    )?;
    let drew = write_sample(&stream, &directory, "e0", video_latents, audio_latents)?;

    // 11. The trajectory. The loop is one closure so that the walk is written whatever the
    //     evaluations did: a stop's most valuable artefact is the list of launches before it.
    let trajectory = (|| -> Result<Vec<u8>> {
        let mut last = drew.clone();
        for (index, step) in steps.iter().enumerate().take(evaluations as usize) {
            if step.index != index as u32 {
                return Err(Failure::message(format!(
                    "the schedule's step {index} names itself {}",
                    step.index
                )));
            }
            // The solver's coefficient is device data the worker writes before each replay; this
            // harness is that worker, so it writes the one word the step table holds.
            let word = i32::try_from(index)
                .map_err(|_| Failure::message(format!("evaluation {index} of a step table")))?;
            upload_bytes(&stream, state[step_table], &word.to_le_bytes())?;
            let cond = nn::capability::Conditioning {
                text,
                source: None,
                media: &[],
                images: None,
                audio: None,
                state: &state,
                step: None,
            };
            block.denoise(&forward, step, latents, cond)?;
            last = write_sample(
                &stream,
                &directory,
                &format!("e{}", index + 1),
                video_latents,
                audio_latents,
            )?;
            println!(
                "evaluations: {} of {evaluations}, sigma {} -> {}",
                index + 1,
                step.sigma,
                step.sigma_next
            );
        }
        Ok(last)
    })();
    ops::capture::set_sink(None);
    drop(sink);

    // 12. What ran, and the one verdict a run cannot fake: a trajectory whose last sample is its own
    //     draw wrote nothing, which the exit code would otherwise pass.
    let walk_path = directory.join("walk.txt");
    if walk.len() != 0 {
        walk_file.write()?;
        println!(
            "walk    {} launches of {} distinct programs -> {}",
            walk.len(),
            walk.distinct(),
            walk_path.display()
        );
    }
    let last = trajectory?;
    if walk.len() == 0 {
        return Err(Failure::message(
            "the walk recorded no launches: the forward replayed a captured graph, or no sink was \
             installed",
        ));
    }
    if last == drew {
        return Err(Failure::message(
            "the last sample is byte-identical to the draw: the evaluations wrote nothing",
        ));
    }
    println!(
        "wrote {} states of {video_elements} latent elements to {}",
        evaluations + 1,
        directory.display()
    );
    Ok(())
}

/// The bytes a list of tables occupies, in each table's own element type.
fn table_bytes(specs: &[nn::capability::StateTable]) -> u64 {
    specs
        .iter()
        .map(|spec| spec.shape.numel() * spec.dtype.bytes().unwrap_or(1))
        .sum()
}

/// Allocate one device tensor per table, in the table's own element type and extents.
fn allocate_tables(arena: &Arena, specs: &[nn::capability::StateTable]) -> Result<Vec<TensorView>> {
    specs
        .iter()
        .map(|spec| Ok(arena.tensor(spec.dtype, spec.shape)?))
        .collect()
}

/// Zeroed host bytes for one table list, sized as the model declared each table.
fn host_tables(specs: &[nn::capability::StateTable]) -> Result<Vec<Vec<u8>>> {
    specs
        .iter()
        .map(|spec| {
            let width = spec.dtype.bytes().ok_or_else(|| {
                Failure::message(format!(
                    "the table `{}` is {} and has no byte width",
                    spec.name, spec.dtype
                ))
            })?;
            let bytes = usize::try_from(spec.shape.numel() * width).map_err(|_| {
                Failure::message(format!(
                    "the table `{}` is {:?} and does not fit a host allocation",
                    spec.name,
                    spec.shape.dims()
                ))
            })?;
            Ok(vec![0_u8; bytes])
        })
        .collect()
}

/// Upload host bytes into a tensor already allocated for them.
fn upload_bytes(stream: &platform::Stream, view: TensorView, bytes: &[u8]) -> Result<()> {
    let width = view
        .dtype
        .bytes()
        .ok_or_else(|| Failure::message(format!("a {} view has no byte width", view.dtype)))?;
    let expected = usize::try_from(view.shape.numel() * width).unwrap_or(usize::MAX);
    if bytes.len() != expected {
        return Err(Failure::message(format!(
            "{} bytes for a {} view of {:?}, which holds {expected}",
            bytes.len(),
            view.dtype,
            view.shape.dims()
        )));
    }
    // SAFETY: `bytes` outlives the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            view.ptr.address(),
            bytes.as_ptr() as u64,
            bytes.len(),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

/// The whole of a tensor's device memory, as bytes.
fn download(stream: &platform::Stream, view: TensorView) -> Result<Vec<u8>> {
    let width = view
        .dtype
        .bytes()
        .ok_or_else(|| Failure::message(format!("a {} view has no byte width", view.dtype)))?;
    let count = usize::try_from(view.shape.numel() * width).map_err(|_| {
        Failure::message(format!("{:?} is not a host allocation", view.shape.dims()))
    })?;
    let mut bytes = vec![0_u8; count];
    // SAFETY: `bytes` is written by the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(bytes.as_mut_ptr() as u64, view.ptr.address(), count, stream)?;
    }
    stream.synchronize()?;
    Ok(bytes)
}

/// Write one evaluation state's latents, and answer the video's bytes.
///
/// The video is the native raster and the audio its rows, both FP32: one file per tensor, named with
/// the state, so a residual in one evaluation is one glance rather than a bisect.
fn write_sample(
    stream: &platform::Stream,
    directory: &Path,
    state: &str,
    video: TensorView,
    audio: Option<TensorView>,
) -> Result<Vec<u8>> {
    let video_bytes = download(stream, video)?;
    write_h3t(
        &directory.join(format!("latents_video_{state}.h3t")),
        video.shape.dims(),
        1,
        &video_bytes,
    )?;
    if let Some(audio) = audio {
        let audio_bytes = download(stream, audio)?;
        write_h3t(
            &directory.join(format!("latents_audio_{state}.h3t")),
            audio.shape.dims(),
            1,
            &audio_bytes,
        )?;
    }
    Ok(video_bytes)
}

/// The video raster's extents, as the case writes them.
/// Stage 2: one stack layer's arithmetic on a device.
///
/// The order is the worker's, done by hand: the device and its stream, the kernels, the weights,
/// the arena, the modulation cache's one post-load computation, then the forward. Each step's failure
/// names itself and nothing else, which is the point of the staging.
fn stage_block(args: &Args) -> Result<()> {
    let store_root = args.store.clone().ok_or_else(|| {
        Failure::message(
            "stage `block` needs a program store: pass --store <path> or set UNISERVE_STORE",
        )
    })?;
    let (denoiser, partition) = configuration(args)?;
    let config = denoiser.transformer;
    let hidden = u64::from(config.hidden_size);
    let checkpoint = Checkpoint::open(&args.root)?;
    let dit = checkpoint.component(args.dit.directory())?;

    // 1. The device and the stream every launch is issued on, before any weight is allocated: a
    //    driver that cannot give a device fails here and not inside a load.
    let device = platform::Device::new(platform::DeviceId(args.device))?;
    let _guard = device.enter()?;
    let stream = device.stream()?;

    // 2. The kernels: every committed module of the operators the stack uses, resolved before a byte
    //    is allocated, so a store that lacks a package is reported as coverage rather than as the
    //    first operator to launch.
    let generation = load_generation(&device, &store_root)?;
    check_sparse_store(&generation, denoiser.attention)?;

    // 3. The request and its packed rows. The layout is the deployment's own rule, so the row count,
    //    the padding and the coordinates are the ones a served request would have.
    // The arrangement is kept beside the layouts rather than rebuilt: the arena's plan is priced at
    // the deployment's own row capacities and sequence degree, and a second `partition.arrangement`
    // call would agree with this one only as long as the partition did.
    let arrangement = partition.arrangement(&denoiser)?;
    let layouts = Layouts::new(denoiser.clone(), arrangement)?;
    let request = VideoRequest {
        width: args.width,
        height: args.height,
        frames: args.frames,
        fps: 24,
        first_frames: 0,
        last_frames: 0,
        references: 0,
        text_tokens: args.text_tokens,
        conditions: &[],
        vision_spans: &[],
        condition_rows: None,
    };
    let (layout, packing) = layouts.layout(&request)?;
    let frames = packing::align_num_frames(request.frames);
    // The canvas is the model's answer for the request - one rounding of the pixels, the same one
    // the layout's rows were computed from. A canvas reconstructed from the layout would be a second
    // statement of that rounding, right only where the rounding is the identity.
    let canvas = layouts.canvas(&request)?;
    // **The tables are the deployment's kind's, not the dense packing's.** A sparse stack reads the
    // valid rows of each tile, the segment of each tile and the keep table, and its row count is the
    // segment packing's - a different packing of the same request. Handing a sparse stack dense
    // tables leaves it with no views at all, which the layer refuses by name (`a sparse layer was
    // handed no segment views`); that refusal is what this harness met while the model's own step
    // passed them.
    let (packed_rows, position_ids, groups, used, sparse_views) = match denoiser.attention {
        AttentionKind::Dense => {
            let tables =
                packing::dense_tables(&packing, frames, &canvas, request.text_tokens, &[], &[])?;
            (
                packing.padded_tokens,
                tables.position_ids,
                tables.groups,
                Some(tables.used),
                None,
            )
        }
        AttentionKind::Sparse {
            sparsity,
            reference_keep,
            ..
        } => {
            let segment = layouts.segment_of(&layout)?;
            let conditions = packing::condition_segments(&[], &canvas, denoiser.clip)?;
            let tables = packing::segment_tables(
                &segment,
                frames,
                &canvas,
                request.text_tokens,
                &conditions,
                &[],
                packing::Sparsity {
                    sparsity,
                    reference_keep,
                },
            )?;
            (
                segment.extent.padded_tokens,
                tables.position_ids,
                tables.groups,
                None,
                Some((
                    tables.variable_block_sizes,
                    tables.tile_segments,
                    tables.segment_keep,
                )),
            )
        }
    };
    let tokens = u64::from(packed_rows);
    if position_ids.len() as u64 != tokens || groups.len() as u64 != tokens {
        return Err(Failure::message(format!(
            "the row tables hold {} coordinates and {} groups for {tokens} packed rows",
            position_ids.len(),
            groups.len()
        )));
    }
    println!(
        "request: {}x{}, {} frames, {} rows packed of {} padded",
        args.width, args.height, frames, packed_rows, tokens,
    );
    match (&used, &sparse_views) {
        (Some(used), None) => println!("attention: dense, {used} attended"),
        (None, Some((sizes, _, _))) => {
            println!("attention: sparse, {} tile(s)", sizes.len());
        }
        _ => {
            return Err(Failure::message(
                "the tables do not match the deployment's attention kind".to_owned(),
            ));
        }
    }

    // 4. Weights on the device, through the loader's own boundary.
    let mut weights = DeviceWeights::new(&device, 0)?;
    let (stack, projections, cache, memory) = {
        let mut loader = WeightLoader::new(dit, &mut weights, args.dtype, None);
        let rows = layouts.rows_bound()?;
        let tiles = match denoiser.attention {
            AttentionKind::Sparse { .. } => layouts.tiles_bound()?,
            AttentionKind::Dense => 0,
        };
        let stack = Transformer::load(
            &config,
            denoiser.attention,
            rows,
            tiles,
            &partition,
            &mut loader,
            None,
        )?;
        let projections = Projections::load(&config, &mut loader)?;
        let memory = adaln_memory(&mut loader, &denoiser, &projections)?;
        let cache = AdaLnCache::new(
            memory.layers.clone(),
            memory.output,
            memory.entries,
            memory.groups,
            config.hidden_size,
            config.time_dim,
        )?;
        (stack, projections, cache, memory)
    };

    // The element type the stack computes in, which is the deployment's: `dtype = "auto"` follows the
    // checkpoint and the loader converts once, so every buffer this harness hands the model - the
    // input rows included - is written in it.
    let dtype = stack.dtype();

    // 5. The arena: the stack's own temporaries are reserved inside the forward, so the region must
    //    cover one evaluation's plan. The plan is **the model's own statement of that sum** - the
    //    same `denoise_arena` the loaded denoiser's `Denoiser::denoise_arena` calls, priced from this
    //    stage's config, arrangement, group count, element type and layout - so the block stage and
    //    the denoise stage reserve the same model number, and `denoise_plan` prices it on a host with
    //    no device (`--check` proves the plan covers every request the forward makes).
    //
    //    What is added to it is what only this harness holds: the row coordinates, the groups, the
    //    prefix source and the latents it writes are its own buffers, not the model's, and the plan
    //    does not know them. The slack covers them; the arena's own reservation refuses by name if
    //    it is short, so a case that outgrows it says so rather than writing past the region.
    let plan = denoise_arena(
        &denoiser,
        &arrangement,
        memory.groups.count(),
        dtype,
        &layout,
    )?;
    let planned: u64 = plan.iter().map(|buffer| buffer.bytes()).sum();
    let bytes = planned + (256 << 20);
    // The directory and the case go out before the arena: the case names the run, everything it
    // states is host data, and a stop in the allocation or in the forward then still leaves the
    // statement the reference producer and the comparator are run against.
    let directory = args.out.join("denoiser");
    std::fs::create_dir_all(&directory)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;

    let evaluation: i32 = 0;
    let width = args.width;
    let height = args.height;
    let text_tokens = request.text_tokens;
    let layers = args.layers;
    let video_rows = packing.video_rows;
    let audio_rows = packing.audio_rows;
    let text_rows = packing.text_rows;
    let condition_rows = packing.condition_rows;
    let checkpoint = args.root.display().to_string();
    let case = format!(
        "{{\"task\": \"t2va\", \"width\": {width}, \"height\": {height}, \"frames\": {frames}, \
         \"text_tokens\": {text_tokens}, \"layers\": {layers}, \"padded_tokens\": {tokens}, \
         \"packed_rows\": {tokens}, \"hidden\": {hidden}, \"video_rows\": {video_rows}, \
         \"audio_rows\": {audio_rows}, \"text_rows\": {text_rows}, \
         \"condition_rows\": {condition_rows}, \"step\": {evaluation}, \"dtype\": \"{dtype}\", \
         \"checkpoint\": {checkpoint:?}}}\n"
    );
    // A case is JSON that a comparator reads, so it is parsed before it is written: a misplaced
    // comma would otherwise cost a device run to discover, and the refusal names the file.
    serde_json::from_str::<serde_json::Value>(&case).map_err(|error| {
        Failure::message(format!(
            "the case this stage states is not JSON ({error}): {case}"
        ))
    })?;
    std::fs::write(directory.join("case.json"), case)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;

    let arena_bytes = bytes.max(args.arena_gib << 30);
    let allocator = weights.allocator();
    // The two numbers a pool refusal is about, printed **before** the reservation that can fail: the
    // model's plan for one evaluation and the region this stage reserves for it.
    println!(
        "arena {:.2} GiB planned ({} buffers) of {:.2} GiB reserved",
        planned as f64 / f64::from(1 << 30),
        plan.len(),
        arena_bytes as f64 / f64::from(1 << 30)
    );
    allocator.reserve(
        platform::PoolKind::Graph,
        usize::try_from(arena_bytes).unwrap_or(usize::MAX),
    )?;
    let region = allocator.allocate(
        platform::PoolKind::Graph,
        usize::try_from(arena_bytes).unwrap_or(usize::MAX),
    )?;
    let arena = Arena::new(DevicePtr::new(region.address()), arena_bytes);
    let forward = Forward::new(&stream, &generation, &arena);

    // 6. The modulation cache, computed once: the schedule's own times, its own groups.
    //
    // The walk starts here rather than at the forward, because this stage's first launches are the
    // cache's own F32 projections - one of them was a stop, and a walk that began at the stack would
    // have reported a complete list while the run still refused at the cache.
    let mut walk = Walk::new();
    let mut walk_file = WalkFile {
        path: args.out.join("walk.txt"),
        recorded: walk.clone(),
        written: false,
    };
    ops::capture::set_sink(Some(&mut walk));
    let sink = SinkGuard;
    cache_projections(&forward, &projections, &memory)?;

    // 7. The inputs: the packed rows' rotary, the row table, the step, and the tokens themselves.
    let positions = transpose_positions(&stream, &arena, &position_ids, tokens)?;
    // A factor table holds one column per rotated pair, not per rotated channel: `rotated_width()`
    // counts channels (three axes of `2 * rope_freq_dim`) and the rotation reads `rotated / 2`.
    let cosine = arena.tensor(
        DType::F32,
        Shape::new(&[tokens, rotated_width(&config) / 2]),
    )?;
    let sine = arena.tensor(
        DType::F32,
        Shape::new(&[tokens, rotated_width(&config) / 2]),
    )?;
    stack.factors(&forward, positions, cosine, sine)?;
    let rows = upload_i32(&stream, &arena, Shape::new(&[tokens]), &groups)?;
    // The evaluation the stack runs: the schedule's first, which is also the row the modulation
    // cache's entry table is indexed at. It is printed and written into `case.json`, because the
    // reference producer runs its own stack at one step and two sides at different steps differ by
    // exactly the modulation, which reads like an arithmetic difference.
    let step = upload_i32(&stream, &arena, Shape::new(&[1]), &[evaluation])?;
    // The visibility the deployment's own state tables state: every packed query row sees the
    // request's **attended** keys and none of the padding rows. Without it the stack attends all
    // `tokens` rows while the reference producer masks each query to `used` keys - the two sides
    // would compute different functions on the same input, and the comparison would report the
    // harness's mask as an arithmetic difference.
    // A sparse stack reads no visibility at all - its layers read the tiles and the keep table -
    // so the pair is stated only for a dense deployment, and the sparse case falls back to the
    // packed row count rather than to a number this harness invented.
    let used =
        match used {
            Some(used) => i32::try_from(used)
                .map_err(|_| Failure::message(format!("{used} attended rows")))?,
            None => i32::try_from(tokens)
                .map_err(|_| Failure::message(format!("{tokens} packed rows")))?,
        };
    let packed =
        i32::try_from(tokens).map_err(|_| Failure::message(format!("{tokens} packed rows")))?;
    let cu_seq_lens_q = upload_i32(&stream, &arena, Shape::new(&[2]), &[0, packed])?;
    let cu_seq_lens_k = upload_i32(&stream, &arena, Shape::new(&[2]), &[0, used])?;
    // The descriptor's key bound is the padded row count, which is the model's own statement
    // (`denoiser.rs`: one program per layout, and a bound is a class rather than a request shape).
    let visible = Visible {
        cu_seq_lens_q,
        cu_seq_lens_k,
        key_rows: u32::try_from(tokens).unwrap_or(u32::MAX),
    };
    let (input, input_bytes) = pseudo_rows(&stream, &arena, dtype, tokens, hidden)?;

    // 8. The forward. `--layers` layers, in place, at the schedule's first evaluation.
    println!("step    {evaluation} of the schedule");
    let total = u32::try_from(tokens).unwrap_or(u32::MAX);
    // The segment views a sparse stack reads: the rows of each tile, the segment of each tile and
    // the key tiles each segment keeps - the three tables the model declares as the block's state,
    // taken from the segment packing above rather than rebuilt here.
    let segment_views = match &sparse_views {
        None => None,
        Some((sizes, segments, keep)) => {
            let tiles = u64::try_from(sizes.len())
                .map_err(|_| Failure::message("more tiles than a u64 states".to_owned()))?;
            let rows = sizes
                .iter()
                .map(|rows| i32::try_from(*rows).unwrap_or(i32::MAX))
                .collect::<Vec<_>>();
            Some(minimax_h3::attention_sparse::Segments {
                sizes: upload_i32(&stream, &arena, Shape::new(&[tiles]), &rows)?,
                segments: upload_i32(&stream, &arena, Shape::new(&[tiles]), segments)?,
                keep: upload_i32(&stream, &arena, Shape::new(&[tiles]), keep)?,
            })
        }
    };
    let forward_result = stack.forward(
        &forward,
        &cache,
        step,
        input,
        rows,
        Factors { cosine, sine },
        Some(&visible),
        segment_views.as_ref(),
        total,
        None,
    );
    // Removed before the result is read: an installed sink is reported to by every later launch on
    // this thread, and a failure must not leave one behind.
    ops::capture::set_sink(None);
    drop(sink);
    forward_result?;
    stream.synchronize()?;
    if walk.len() == 0 {
        return Err(Failure::message(
            "the walk recorded no launches: the forward replayed a captured graph, or no sink was \
             installed",
        ));
    }

    // 9. What ran, written where `compare_blocks` reads it.
    //
    // The input goes out first and from the host's own copy, because the forward overwrites the view
    // it was handed: reading that buffer afterwards would write the output into both files, and a
    // reference producer would then run today's blocks on our output and call the result a
    // comparison. The two payloads are also required to differ - two byte-identical files mean the
    // forward wrote nothing, which is a silent failure the exit code would otherwise pass.
    write_h3t(
        &directory.join("stack_in.h3t"),
        &[tokens, hidden],
        element_code(dtype)?,
        &input_bytes,
    )?;
    // The stack writes in place, so the output is read from the same buffer at the same width.
    let output_bytes = download(&stream, input)?;
    if output_bytes == input_bytes {
        return Err(Failure::message(
            "the stack wrote nothing: `stack_out` is byte-identical to `stack_in`, so the forward \
             left its input in place",
        ));
    }
    write_h3t(
        &directory.join("stack_out.h3t"),
        &[tokens, hidden],
        element_code(dtype)?,
        &output_bytes,
    )?;
    // The row order the comparison indexes into, stated beside the tensor it wrote: the four
    // segments in their packed order and the count they sum to. Without it a table of row blocks
    // would have to re-derive the packing in another language, which is a second statement of the
    // rounding - and a file whose order is not stated cannot be shown to be in the order the
    // reference ran (`compare_stack.py`).
    let walk_path = args.out.join("walk.txt");
    walk_file.write()?;
    println!(
        "walk    {} launches of {} distinct programs -> {}",
        walk.len(),
        walk.distinct(),
        walk_path.display()
    );
    println!(
        "wrote {} and {} ({} rows of {})",
        directory.join("stack_in.h3t").display(),
        directory.join("stack_out.h3t").display(),
        tokens,
        hidden
    );
    Ok(())
}

/// The modulation cache's memory, from the schedule's times.
///
/// The same construction the denoiser performs at load, so the tables are the served ones: the
/// schedule's own timesteps decide how many entries there are, and every layer's table is that many
/// rows of eighteen vectors. The tables are kept here, outside the cache, because the one post-load
/// computation writes into the very views the cache then gathers from.
struct AdaLnMemory {
    groups: Groups,
    /// `[steps, groups]` i32: the entry each group of each step reads.
    entries: nn::Weight,
    /// One `[entries, 18 * hidden]` FP32 table per layer.
    layers: Vec<nn::Weight>,
    /// `[entries, 2 * hidden]` FP32: the final normalization's table.
    output: nn::Weight,
    /// `[entries, frequency]` FP32: the sinusoidal embedding the projections read.
    frequencies: nn::Weight,
}

fn adaln_memory(
    loader: &mut WeightLoader<'_>,
    denoiser: &minimax_h3::config::DenoiserConfig,
    projections: &Projections,
) -> Result<AdaLnMemory> {
    let schedule = Schedule::new(&denoiser.schedule)?;
    let conditioned = denoiser
        .tasks
        .iter()
        .any(|task| !matches!(task, VideoTask::TextToVideoAudio));
    let groups = Groups::of(conditioned);
    let (times, table) = modulation::entries(
        &schedule.timesteps(Modality::Video),
        &schedule.timesteps(Modality::Audio),
        groups,
    )?;
    let entries = times.len() as u64;
    let hidden = u64::from(denoiser.transformer.hidden_size);
    let columns = ADALN_COLUMNS * hidden;
    let frequency = u64::from(denoiser.transformer.frequency_dim);
    let frequencies = loader.constant(
        "minimax_h3.adaln.frequencies",
        DType::F32,
        Shape::new(&[entries, frequency]),
        &projections.frequencies(&times)?,
        "minimax-h3",
    )?;
    let mut entry_bytes = Vec::with_capacity(table.len() * 4);
    for value in &table {
        entry_bytes.extend_from_slice(&value.to_le_bytes());
    }
    // One row per timestep of the ladder - the clean end included - so the row count comes from the
    // table the entries were built into, not from the evaluation count.
    let steps = schedule.num_steps()?;
    let rows = u64::try_from(table.len() / groups.count() as usize).map_err(|_| {
        Failure::message("an AdaLN entry table wider than u64::MAX rows".to_owned())
    })?;
    if rows < u64::from(steps) {
        return Err(Failure::message(format!(
            "the AdaLN entry table has {rows} rows for {steps} evaluations"
        )));
    }
    let entries_table = loader.constant(
        "minimax_h3.adaln.entries",
        DType::I32,
        Shape::new(&[rows, u64::from(groups.count())]),
        &entry_bytes,
        "minimax-h3",
    )?;
    // The tables are the modulation projections' own type - the reference allocates them from the
    // product - so the harness follows the loaded weights rather than naming a type of its own.
    let table_type = projections.dtype();
    let unit = table_type.bytes().unwrap_or(4);
    let zeros = vec![0_u8; usize::try_from(entries * columns * unit).unwrap_or(0)];
    let mut layers = Vec::with_capacity(denoiser.transformer.num_hidden_layers as usize);
    for index in 0..denoiser.transformer.num_hidden_layers {
        layers.push(loader.constant(
            &format!("minimax_h3.adaln.layers.{index}"),
            table_type,
            Shape::new(&[entries, columns]),
            &zeros,
            "minimax-h3",
        )?);
    }
    let output = loader.constant(
        "minimax_h3.adaln.output",
        table_type,
        Shape::new(&[entries, 2 * hidden]),
        &vec![0_u8; usize::try_from(entries * 2 * hidden * unit).unwrap_or(0)],
        "minimax-h3",
    )?;
    Ok(AdaLnMemory {
        groups,
        entries: entries_table,
        layers,
        output,
        frequencies,
    })
}

/// Run the cache's projections once, into the tables the forward gathers rows from.
fn cache_projections(f: &Forward, projections: &Projections, memory: &AdaLnMemory) -> Result<()> {
    let views: Vec<TensorView> = memory.layers.iter().map(nn::Weight::view).collect();
    projections.compute(f, memory.frequencies.view(), &views, memory.output.view())?;
    Ok(())
}

/// The rotary width of one row: the three axes' shares of a head, from the checkpoint's own count.
fn rotated_width(config: &minimax_h3::config::TransformerConfig) -> u64 {
    rotated_axes(config)
        .iter()
        .map(|axis| u64::from(*axis))
        .sum()
}

/// A deterministic input in the deployment's own element type: the same bytes on every host, so a
/// reference can read them back.
///
/// The element type is the **deployment's**, not one this harness picks: `dtype = "auto"` converts
/// nothing at load, so a buffer of another type would be a buffer the model has to convert - or, as
/// the first device run showed, a launch whose descriptor no store holds.
fn pseudo_rows(
    stream: &platform::Stream,
    arena: &Arena,
    dtype: DType,
    rows: u64,
    columns: u64,
) -> Result<(TensorView, Vec<u8>)> {
    let bytes = pseudo_bytes(dtype, rows, columns)?;
    let tensor = arena.tensor(dtype, Shape::new(&[rows, columns]))?;
    upload_bytes(stream, tensor, &bytes)?;
    Ok((tensor, bytes))
}

/// The same rows as host bytes, for a buffer that holds more than they do.
///
/// A fixed function of nothing: no seed, no case, so two runs and both sides read the same bytes.
fn pseudo_bytes(dtype: DType, rows: u64, columns: u64) -> Result<Vec<u8>> {
    let width = dtype
        .bytes()
        .ok_or_else(|| Failure::message(format!("{dtype} has no byte width")))?
        as usize;
    let count = usize::try_from(rows * columns).unwrap_or(0);
    let mut bytes = Vec::with_capacity(count * width);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // A value in `[-1, 1)`: twenty-five bits of the state scaled by 2^24 and shifted down, so the
        // draw is a ramp over a symmetric interval. Symmetry matters here rather than being tidy -
        // a one-sided draw would leave every sign-dependent path of the block (the gated
        // feed-forward's two halves, the norms' shifts) exercised on one side only, and both sides
        // of the comparison would agree on an untested range.
        let value = ((state >> 39) as f32) / ((1_u64 << 24) as f32) - 1.0;
        match dtype {
            DType::BF16 => bytes.extend_from_slice(&half_of(value).to_le_bytes()),
            DType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
            other => {
                return Err(Failure::message(format!(
                    "{other} has no pseudo rows: this harness writes the checkpoint's BF16 or F32"
                )));
            }
        }
    }
    Ok(bytes)
}

/// The `h3t` container's element code, which the comparator reads.
fn element_code(dtype: DType) -> Result<u8> {
    match dtype {
        DType::F32 => Ok(1),
        DType::BF16 => Ok(4),
        DType::F16 => Ok(5),
        other => Err(Failure::message(format!(
            "{other} has no code in the `h3t` container this harness writes"
        ))),
    }
}

/// A BF16 holding the high half of `value`, rounded to nearest even.
fn half_of(value: f32) -> u16 {
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    u16::try_from((bits.wrapping_add(bias)) >> 16).unwrap_or(0)
}

/// The packed rows' coordinates as the `[3, tokens]` table the rotary reads.
fn transpose_positions(
    stream: &platform::Stream,
    arena: &Arena,
    coordinates: &[[f64; 3]],
    tokens: u64,
) -> Result<TensorView> {
    let rows = usize::try_from(tokens).unwrap_or(0);
    let mut values = vec![0_i32; rows * 3];
    for (row, coordinate) in coordinates.iter().take(rows).enumerate() {
        for axis in 0..3 {
            values[axis * rows + row] = coordinate[axis].round() as i32;
        }
    }
    upload_i32(stream, arena, Shape::new(&[3, tokens]), &values)
}

/// Remove a previous run's trajectory files from a stage's `denoiser` directory.
///
/// The names are the stage's own contract - `latents_<modality>_e<index>.h3t` and
/// `latents_<modality>_final.h3t` - and nothing else in the directory is touched. Returns how many
/// were removed.
fn clear_trajectory(directory: &std::path::Path) -> Result<usize> {
    let entries = std::fs::read_dir(directory)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    let mut removed = 0;
    for entry in entries {
        let entry =
            entry.map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("latents_") || !name.ends_with(".h3t") {
            continue;
        }
        std::fs::remove_file(entry.path())
            .map_err(|error| Failure::message(format!("{}: {error}", entry.path().display())))?;
        removed += 1;
    }
    Ok(removed)
}

/// What one eager forward bound: every launch's operator and descriptor key, in launch order.
///
/// A captured graph replays without calling `operate`, so a sink installed around a replay records
/// nothing; the stage this walks is an eager forward, which is why the walk belongs here. The
/// recorder is the worker's own mechanism (`ops::capture`), installed for the duration of the
/// forward and removed whether it succeeds or fails: a sink left installed would be reported to by
/// every later launch on the thread.
///
/// The recording is shared (`Rc<RefCell<..>>`), so the stage keeps a handle it can read while the
/// sink is still installed: a failing launch must be reportable *at* the failure, and a sink that
/// owned its rows outright would borrow them exclusively until it was removed - after the `?` that
/// ends the stage, which is too late to say anything.
#[derive(Clone, Default)]
struct Walk(std::rc::Rc<std::cell::RefCell<Vec<(&'static str, String)>>>);

impl ops::capture::Sink for Walk {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.borrow_mut().push((operator, key.to_owned()));
    }
}

/// Removes the capture sink when it leaves scope, so a failing launch cannot leave one installed.
///
/// The sink is thread-local and every later launch on the thread reports to it; a `?` between the
/// install and the explicit removal would otherwise leak it, and the next stage's walk would then
/// record launches that belong to another pass.
struct SinkGuard;

impl Drop for SinkGuard {
    fn drop(&mut self) {
        ops::capture::set_sink(None);
    }
}

/// Writes the stage's walk when it leaves scope if nothing wrote it earlier.
///
/// A failing launch ends the stage through a `?` long before the end of the pass, and the launches
/// recorded so far are then the only evidence of where it died: the stop line names a native entry
/// and nothing more (`native entry \`uniserve_gemm\` failed with status 1`), while one operator is
/// launched at many descriptor keys in one forward. The guard writes the file and prints the last
/// launches, so a stop says which program launched before it.
struct WalkFile {
    path: std::path::PathBuf,
    recorded: Walk,
    written: bool,
}

impl WalkFile {
    fn write(&mut self) -> Result<()> {
        std::fs::write(&self.path, self.recorded.text())
            .map_err(|error| Failure::message(format!("{}: {error}", self.path.display())))?;
        self.written = true;
        Ok(())
    }
}

impl Drop for WalkFile {
    fn drop(&mut self) {
        if self.written {
            return;
        }
        let _ = std::fs::write(&self.path, self.recorded.text());
        eprintln!(
            "walk: the stage stopped after {} launches; the last ones were\n{}",
            self.recorded.len(),
            self.recorded.tail(5)
        );
    }
}

impl Walk {
    fn new() -> Self {
        Self::default()
    }

    /// The walk as the store's key list is written: one `operator<TAB>key` per launch, in order.
    fn text(&self) -> String {
        let mut text = String::new();
        for (operator, key) in self.0.borrow().iter() {
            text.push_str(operator);
            text.push('\t');
            text.push_str(key);
            text.push('\n');
        }
        text
    }

    /// How many distinct programs the walk bound.
    fn distinct(&self) -> usize {
        let launches = self.0.borrow();
        let mut keys: Vec<(&str, &str)> = launches
            .iter()
            .map(|(operator, key)| (*operator, key.as_str()))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys.len()
    }

    /// The launches this walk has recorded, oldest first, as lines.
    fn tail(&self, count: usize) -> String {
        let launches = self.0.borrow();
        let start = launches.len().saturating_sub(count);
        let mut text = String::new();
        for (operator, key) in &launches[start..] {
            text.push_str(operator);
            text.push('\t');
            text.push_str(key);
            text.push('\n');
        }
        text
    }

    /// The launches recorded so far.
    fn len(&self) -> usize {
        self.0.borrow().len()
    }
}

/// The `.h3t` container: magic, element type, rank, extents, then the payload.
fn write_h3t(path: &Path, shape: &[u64], dtype: u8, payload: &[u8]) -> Result<()> {
    let mut bytes = Vec::with_capacity(6 + shape.len() * 8 + payload.len());
    bytes.extend_from_slice(b"H3D1");
    bytes.push(dtype);
    bytes.push(u8::try_from(shape.len()).unwrap_or(0));
    for extent in shape {
        bytes.extend_from_slice(&extent.to_le_bytes());
    }
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes)
        .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The harness's own runtime objects. A model may not own them, so they live here, in an example.

/// Host storage for weights: validates each tensor, keeps it, and never answers twice.
#[derive(Default)]
struct HostWeights {
    published: Vec<String>,
    by_dtype: std::collections::BTreeMap<String, (u64, u64)>,
    bytes: u64,
}

impl RetainedWeights for HostWeights {
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
        // A validating pass never launches, so the address it answers is only ever asked for its
        // shape and its element size by the loader's own fit check; it need not outlive this call.
        let entry = self.by_dtype.entry(dtype.to_string()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += expected;
        self.bytes += expected;
        self.published.push(key.to_string());
        // The view is built before the bytes are dropped or kept, so the loader's own fit check
        // (`Backing::view`) runs either way.
        Backing::new(DevicePtr::new(bytes.as_ptr() as u64), expected).view(dtype, shape)?;
        Ok(Backing::new(
            DevicePtr::new(bytes.as_ptr() as u64),
            expected,
        ))
    }
}

/// Device storage for weights, kept for the life of the model.
struct DeviceWeights {
    pools: platform::Allocator,
    buffers: Vec<platform::Buffer>,
    bytes: u64,
}

impl DeviceWeights {
    /// The weights' home: a **virtual** range sized from the device's own free storage.
    ///
    /// `Allocator::reserve` maps and books nothing until the first allocation, so this range takes no
    /// storage from the arena the stage reserves beside it - it only caps how much the loaded weights
    /// may ever occupy. Sizing that cap by a figure of this harness's was the defect: a `requested`
    /// of zero fell back to an 8 GiB floor, and the sparse deployment's weights are 8.49 GiB, so the
    /// pool refused a load the device could hold with the message `cannot give out 522190848 more` -
    /// a whole point of the run spent reading a device limit that was this file's. The device is
    /// asked instead, and the halving below still covers a driver that refuses the whole range.
    fn new(device: &platform::Device, requested: u64) -> Result<Self> {
        let budget = std::sync::Arc::new(NoLimit);
        let mut pools = platform::Allocator::new(device, budget)?;
        // `memory_info` answers in this device's primary context; a driver that cannot answer is no
        // reason to refuse, so the floor stands as the fallback rather than the default.
        let free = device.memory_info().map(|info| info.free).unwrap_or(0);
        // Never less than it was before: the 8 GiB floor stays as the fallback for a driver that
        // cannot answer, and the device's own free storage raises the cap above it when it can.
        let mut wanted = requested.max(free.max(8 << 30));
        let mut reserved = None;
        while wanted >= (1 << 30) {
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
                "the driver refused every resident reservation down to 1 GiB: the loaded layers' \
                 weights cannot be held on this device",
            )
        })?;
        println!(
            "resident pool: {} GiB reserved of {} GiB free on the device",
            reserved >> 30,
            free >> 30
        );
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
        // default stream and waited on before the backing is handed out, because the loader keeps no
        // stream and a later launch may use another one.
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

/// Every committed module of the operators the stack uses.
///
/// The store's layout is `op/backend/revision/target/descriptor`, and which backend, revision and
/// target it holds is the deployment's business: this harness wants every committed descriptor,
/// because `ops` binds by key at the launch and a missing one is named there.
/// The six operators a sparse deployment launches that a dense one does not.
///
/// A dense store holds none of them, so a sparse stage with the wrong store launches its
/// projections and the first norm and then stops at the first `vsa_*` cell - which from outside is
/// "it stopped at `vsa_prepare_tiles` after fifteen launches" and says nothing about the store.
/// Counting them before any weight loads names the store as the subject.
const SPARSE_OPERATORS: [&str; 6] = [
    "vsa_prepare_tiles",
    "vsa_scores",
    "vsa_select_tiles",
    "vsa_compress_pooled",
    "vsa_add_gated_tiles",
    "vsa_zero_padding",
];

/// Print the sparse chain's cell counts, and refuse when the store holds none of one.
///
/// The store is the deployment's: its key file states the video-sparse
/// deployment's rows, which include the dense deployment's, so one store built from it serves
/// either - while a store built from a dense-only list stops at the first sparse launch, with no
/// kernel for it.
fn check_sparse_store(generation: &ops::Generation, attention: AttentionKind) -> Result<()> {
    if !matches!(attention, AttentionKind::Sparse { .. }) {
        return Ok(());
    }
    let mut empty: Vec<&'static str> = Vec::new();
    let mut counts: Vec<String> = Vec::new();
    for operator in SPARSE_OPERATORS {
        let keys = generation.keys(operator).count();
        if keys == 0 {
            empty.push(operator);
        }
        counts.push(format!("{operator} {keys}"));
    }
    println!("sparse store: {}", counts.join(", "));
    if !empty.is_empty() {
        return Err(Failure::message(format!(
            "this store holds no kernel for {}, so a sparse stage cannot launch: the store is the \
             deployment's own, and this one was built for a dense deployment",
            empty.join(", ")
        )));
    }
    Ok(())
}

fn load_generation(device: &platform::Device, root: &Path) -> Result<ops::Generation> {
    if !root.is_dir() {
        return Err(Failure::message(format!(
            "the store root {} is not a directory: nothing is installed there, so no stage past \
             `weights` can launch",
            root.display()
        )));
    }
    let mut entries = Vec::new();
    let mut installed = 0_usize;
    let mut operators: Vec<String> = std::fs::read_dir(root)
        .map_err(|error| Failure::message(format!("{}: {error}", root.display())))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter_map(|path| path.file_name()?.to_str().map(str::to_owned))
        .collect();
    operators.sort();
    for operator in &operators {
        for (descriptor, directory) in committed(&root.join(operator)) {
            entries.push(install(device, operator, &descriptor, &directory)?);
            installed += 1;
        }
    }
    println!(
        "store: {installed} committed modules of {} operators",
        operators.len()
    );
    Ok(ops::Generation::new(1, entries))
}

/// The committed descriptor directories under one operator's root, sorted for a stable order.
fn committed(op_root: &Path) -> Vec<(String, PathBuf)> {
    let mut found = Vec::new();
    let backends = std::fs::read_dir(op_root).into_iter().flatten().flatten();
    for backend in backends {
        let revisions = std::fs::read_dir(backend.path())
            .into_iter()
            .flatten()
            .flatten();
        for revision in revisions {
            let targets = std::fs::read_dir(revision.path())
                .into_iter()
                .flatten()
                .flatten();
            for target in targets {
                let descriptors = std::fs::read_dir(target.path())
                    .into_iter()
                    .flatten()
                    .flatten();
                for descriptor in descriptors {
                    if descriptor.path().join(".committed").is_file() {
                        found.push((
                            descriptor.file_name().to_string_lossy().into_owned(),
                            descriptor.path(),
                        ));
                    }
                }
            }
        }
    }
    found.sort_by(|left, right| left.0.cmp(&right.0));
    found
}

/// Load one committed module as an `ops` kernel entry.
fn install(
    device: &platform::Device,
    operator: &str,
    descriptor: &str,
    directory: &Path,
) -> Result<ops::KernelEntry> {
    let metadata_path = directory.join("metadata.json");
    let metadata = std::fs::read(&metadata_path).ok();
    let launch = std::fs::read_to_string(directory.join("launch.json")).unwrap_or_default();
    let entry = metadata
        .as_deref()
        .and_then(|bytes| json_field(&String::from_utf8_lossy(bytes), "name"))
        .or_else(|| json_field(&launch, "name"))
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
                "{}: this build of `ops` has no operator `{operator}`",
                directory.display()
            ))
        });
    }
    let library = std::fs::read_dir(directory)
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

/// Upload `values` as an `i32` tensor of `shape`.
fn upload_i32(
    stream: &platform::Stream,
    arena: &Arena,
    shape: Shape,
    values: &[i32],
) -> Result<TensorView> {
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
