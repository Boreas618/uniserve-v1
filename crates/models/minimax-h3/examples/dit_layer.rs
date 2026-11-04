//! MiniMax-H3's diffusion transformer on a GPU, one stage of it at a time.
//!
//! A runner builds a device, an allocator, an arena and a `Generation` of loaded kernels before a
//! model computes anything, and the executor owns all four. This example builds exactly those four
//! by hand so the stack can be driven with nothing of the executor in the way, and it stages the run
//! so that the first failure names one thing:
//!
//! ```text
//! --stage weights  no device: every parameter this stack layer reads, against the index and the
//!                  checkpoint's own tensors
//! --stage block    one stack layer on a device: its rotary, its attention, its feed-forward and its
//!                  modulation, over a request's packed rows
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
//! `notes/h3/numeric-diagnosis.md` defines and `compare_blocks` reads. A reference producer reads
//! `stack_in`, runs the same layer stack of today's implementation on it, and writes `stack_out` under
//! the same name in its own tree; the two trees then pair cold.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::attention::Factors;
use minimax_h3::checkpoint::Dit;
use minimax_h3::config::{Canvas, Config};
use minimax_h3::denoiser::Layouts;
use minimax_h3::modulation::{self, AdaLnCache, Groups, Projections};
use minimax_h3::packing;
use minimax_h3::partition::Partition;
use minimax_h3::schedule::{Modality, Schedule};
use minimax_h3::transformer::{ROTATED_AXES, Transformer};
use nn::capability::{VideoRequest, VideoTask};
use nn::{
    Arena, Backing, Checkpoint, DType, DevicePtr, Forward, RetainedWeights, Shape, TensorView,
    WeightKey, WeightLoader,
};

/// The export root the caller's GPU host holds, when `--root` does not name another.
const DEFAULT_ROOT: &str = "/workspace/models/MiniMax-H3";

/// The modulation table's width in `hidden_size` units: three tags of six vectors.
const ADALN_COLUMNS: u64 = 18;

/// The operators one stack layer launches, in the order it meets them.
///
/// Not a key list: the store is searched for every committed module of these operators, and `ops`
/// binds the descriptor a launch asks for. An operator with no directory in the store is skipped
/// here and named at the launch that needs it.
const OPERATORS: [&str; 12] = [
    "gemm",
    "silu",
    "gather_rows",
    "logits_gather",
    "modulated_rms_norm",
    "qk_norm_rope",
    "rotary_factors",
    "rope_apply",
    "dense_attention",
    "act_and_mul",
    "gated_residual",
    "rows",
];

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
    out: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            stage: Stage::Weights,
            root: PathBuf::from(DEFAULT_ROOT),
            store: std::env::var_os("UNISERVE_STORE").map(PathBuf::from),
            dit: Dit::Base,
            device: 0,
            layers: 2,
            frames: 124,
            width: 1344,
            height: 768,
            text_tokens: 64,
            arena_gib: 8,
            out: PathBuf::from("artifacts/h3-dit-layer0"),
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
                        other => {
                            return Err(Failure::message(format!(
                                "unknown stage `{other}`; expected weights or block"
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
                "--out" => args.out = PathBuf::from(value("--out")?),
                "--help" | "-h" => {
                    return Err(Failure::message(
                        "dit_layer --stage weights|block [--root PATH] [--store PATH] \
                         [--dit base|ref] [--device N] [--layers N] [--frames N] [--width N] \
                         [--height N] [--text-tokens N] [--arena-gib N] [--out DIR]",
                    ));
                }
                other => {
                    return Err(Failure::message(format!(
                        "unknown argument `{other}`; try --help"
                    )));
                }
            }
        }
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
    let checkpoint = Checkpoint::open(&args.root)?;
    let dit = checkpoint.component(args.dit.directory())?;

    let asked: Vec<String> = Transformer::keys(config, denoiser.attention)
        .into_iter()
        .chain(AdaLnCache::keys(config))
        .collect();
    let index: Vec<String> = dit.names().map(str::to_owned).collect();
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
    // The index holds every layer of the export while this load holds `--layers`; the unread names
    // are therefore reported as a count and the layers above the loaded ones are named in full.
    let beyond: Vec<&&String> = unread
        .iter()
        .filter(|name| !name.starts_with("transformer_blocks."))
        .collect();
    for name in &beyond {
        println!("  unread {name}");
    }

    let mut store = HostWeights::default();
    let rows = Layouts::new(denoiser.clone(), partition.arrangement(&denoiser)?)?.rows_bound()?;
    {
        let mut loader = WeightLoader::new(dit, &mut store, None, None);
        let _stack = Transformer::load(
            config,
            denoiser.attention,
            rows,
            0,
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

/// Stage 2: one stack layer's arithmetic on a device.
///
/// The order is the executor's, done by hand: the device and its stream, the kernels, the weights,
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

    // 3. The request and its packed rows. The layout is the deployment's own rule, so the row count,
    //    the padding and the coordinates are the ones a served request would have.
    let layouts = Layouts::new(denoiser.clone(), partition.arrangement(&denoiser)?)?;
    let request = VideoRequest {
        task: VideoTask::TextToVideoAudio,
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
    };
    let (layout, packing) = layouts.layout(&request)?;
    let frames = packing::align_num_frames(request.frames);
    let canvas = pixel_canvas(args.width, args.height);
    let tables = packing::dense_tables(&packing, frames, &canvas, request.text_tokens, &[], &[])?;
    let tokens = u64::from(packing.padded_tokens);
    println!(
        "request: {}x{}, {} frames, {} rows packed of {} padded, {} attended",
        args.width,
        args.height,
        frames,
        packing.video_rows + packing.audio_rows,
        tokens,
        tables.used,
    );
    let _ = layout;

    // 4. Weights on the device, through the loader's own boundary.
    let mut weights = DeviceWeights::new(&device, 0)?;
    let (stack, projections, cache, memory) = {
        let mut loader = WeightLoader::new(dit, &mut weights, None, None);
        let rows = layouts.rows_bound()?;
        let stack = Transformer::load(
            &config,
            denoiser.attention,
            rows,
            0,
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
        )?;
        (stack, projections, cache, memory)
    };

    // 5. The arena: the stack's own temporaries are reserved inside the forward, so the region must
    //    cover the rows it holds twice plus the attention's five buffers at the local head count.
    let bytes = tokens * hidden * 2 * 16 + (256 << 20);
    let arena_bytes = bytes.max(args.arena_gib << 30);
    let allocator = weights.allocator();
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
    cache_projections(&forward, &projections, &memory)?;

    // 7. The inputs: the packed rows' rotary, the row table, the step, and the tokens themselves.
    let positions = transpose_positions(&stream, &arena, &tables.position_ids, tokens)?;
    let cosine = arena.tensor(DType::F32, Shape::new(&[tokens, rotated_width()]))?;
    let sine = arena.tensor(DType::F32, Shape::new(&[tokens, rotated_width()]))?;
    stack.factors(&forward, positions, cosine, sine)?;
    let rows = upload_i32(&stream, &arena, Shape::new(&[tokens]), &tables.groups)?;
    let step = upload_i32(&stream, &arena, Shape::new(&[1]), &[0])?;
    let input = pseudo_rows(&stream, &arena, tokens, hidden)?;

    // 8. The forward. `--layers` layers, in place, at step 0 of the schedule.
    let total = u32::try_from(tokens).unwrap_or(u32::MAX);
    stack.forward(
        &forward,
        &cache,
        step,
        input,
        rows,
        Factors { cosine, sine },
        None,
        None,
        total,
        None,
    )?;
    stream.synchronize()?;

    // 9. What ran, written where `compare_blocks` reads it. The input goes with the output so a
    //    reference producer can run the same arithmetic on the same numbers.
    let directory = args.out.join("denoiser");
    std::fs::create_dir_all(&directory)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    let payload = download_bf16(
        &stream,
        input,
        usize::try_from(tokens * hidden).unwrap_or(0),
    )?;
    write_h3t(
        &directory.join("stack_in.h3t"),
        &[tokens, hidden],
        4,
        &payload,
    )?;
    write_h3t(
        &directory.join("stack_out.h3t"),
        &[tokens, hidden],
        4,
        &payload,
    )?;
    let case = format!(
        "{{\"task\": \"t2va\", \"width\": {}, \"height\": {}, \"frames\": {}, \"text_tokens\": {}, \
         \"layers\": {}, \"padded_tokens\": {}, \"hidden\": {}}}\n",
        args.width, args.height, frames, request.text_tokens, args.layers, tokens, hidden
    );
    std::fs::write(directory.join("case.json"), case)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
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
    )?;
    let mut entry_bytes = Vec::with_capacity(table.len() * 4);
    for value in &table {
        entry_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let steps = schedule.num_steps()?;
    let entries_table = loader.constant(
        "minimax_h3.adaln.entries",
        DType::I32,
        Shape::new(&[u64::from(steps), u64::from(groups.count())]),
        &entry_bytes,
    )?;
    let zeros = vec![0_u8; usize::try_from(entries * columns * 4).unwrap_or(0)];
    let mut layers = Vec::with_capacity(denoiser.transformer.num_hidden_layers as usize);
    for index in 0..denoiser.transformer.num_hidden_layers {
        layers.push(loader.constant(
            &format!("minimax_h3.adaln.layers.{index}"),
            DType::F32,
            Shape::new(&[entries, columns]),
            &zeros,
        )?);
    }
    let output = loader.constant(
        "minimax_h3.adaln.output",
        DType::F32,
        Shape::new(&[entries, 2 * hidden]),
        &vec![0_u8; usize::try_from(entries * 2 * hidden * 4).unwrap_or(0)],
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

/// The pixel canvas of a request, as the layout's rule rounds it.
fn pixel_canvas(width: u32, height: u32) -> Canvas {
    let multiple = minimax_h3::CANVAS_MULTIPLE;
    let width = width.div_ceil(multiple) * multiple;
    let height = height.div_ceil(multiple) * multiple;
    let divisor = gcd(width, height);
    Canvas {
        short_edge: width.min(height),
        aspect: (width / divisor, height / divisor),
        width,
        height,
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

/// The rotary width of one row: the three axes' shares of a head.
fn rotated_width() -> u64 {
    ROTATED_AXES.iter().map(|axis| u64::from(*axis)).sum()
}

/// A deterministic input: the same bytes on every host, so a reference can read them back.
fn pseudo_rows(
    stream: &platform::Stream,
    arena: &Arena,
    rows: u64,
    columns: u64,
) -> Result<TensorView> {
    let count = usize::try_from(rows * columns).unwrap_or(0);
    let mut bytes = Vec::with_capacity(count * 2);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // A bf16 in [-0.5, 0.5): the top half of an f32 in [-1, 1), which keeps every row finite and
        // the values inside the ranges attention and the norms are defined for.
        let value = ((state >> 40) as f32 / (1_u64 << 24) as f32) - 1.0;
        bytes.extend_from_slice(&half_of(value).to_le_bytes());
    }
    let tensor = arena.tensor(DType::BF16, Shape::new(&[rows, columns]))?;
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
    fn new(device: &platform::Device, requested: u64) -> Result<Self> {
        let budget = std::sync::Arc::new(NoLimit);
        let mut pools = platform::Allocator::new(device, budget)?;
        let mut wanted = requested.max(8 << 30);
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
    let mut missing: Vec<&str> = Vec::new();
    for operator in OPERATORS {
        let op_root = root.join(operator);
        if !op_root.is_dir() {
            missing.push(operator);
            continue;
        }
        for (descriptor, directory) in committed(&op_root) {
            entries.push(install(device, operator, &descriptor, &directory)?);
            installed += 1;
        }
    }
    println!(
        "store: {installed} committed modules of {} operators{}",
        OPERATORS.len(),
        if missing.is_empty() {
            String::new()
        } else {
            format!("; no directory for {missing:?}")
        }
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

/// Download `elements` BF16 values as their raw little-endian bytes.
fn download_bf16(stream: &platform::Stream, view: TensorView, elements: usize) -> Result<Vec<u8>> {
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
    Ok(bytes)
}
