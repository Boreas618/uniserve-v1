//! MiniMax-H3's video decoder on a GPU: one clip, one canvas, one comparison.
//!
//! This is the second device stage, and it is deliberately the smallest thing that can be wrong: the
//! same latent raster through the same 585 tensors must give the same 25 pixel frames. Everything
//! that decides *which* latent a request gets - the conditioner's raster for `fl2va`/`ref2va`, the
//! sampler's state for generation - is a later stage, so this one is handed its input as a file and
//! nothing about a conditioner or a solver can be read into its verdict.
//!
//! ```text
//! --stage weights  no device: the 585 names the decoder reads, against the export's `vae/` index,
//!                  then the load itself into a host weight store that checks every element type
//!                  and every shape as it is handed over
//! --stage decode   one clip on a device: the folded input affine, the 36 layers, the positions
//!                  table, the tiled decode and the stitch, written as one tensor and its case
//! ```
//!
//! All but `weights` need a program store: `--store <path>` or `UNISERVE_STORE`, holding committed
//! modules for the operators the decoder launches. The store is read, never written; which descriptor
//! a launch binds is `ops`' business, so this harness installs every committed module the store holds
//! and lets the launch name a missing one rather than keeping a second copy of the keys here.
//!
//! **What it writes.** `<out>/video_decoder/latents.h3t` is the input the run used and
//! `<out>/video_decoder/raster.h3t` what the decoder produced, both in the `.h3t` container
//! (`H3D1`, one element-type byte, a rank byte, the extents as `u64`, then the payload). The raster is
//! `[frames, rows, columns, channels]`, the layout the reference's module returns, so the reference
//! producer writes what comes back with no transpose and the one transpose on my side, from my
//! `[channels, frames, rows, columns]` raster, is a single place a test can read.
//!
//! The two payloads must differ: a decode that left its input in place is a silent failure the exit
//! code would otherwise pass.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::config::AttentionKind;
use minimax_h3::decoding::{ClipDecoder, video_arena};
use minimax_h3::video_vae::{Canvas, Tiles, VideoDecoder, folded_input};
use nn::{
    Arena, Backing, Checkpoint, DType, DevicePtr, Forward, RetainedWeights, Shape, TensorView,
    WeightKey, WeightLoader,
};

/// The export root the caller's GPU host holds, when `--root` does not name another.
const DEFAULT_ROOT: &str = "/workspace/models/MiniMax-H3";

/// The `vae/` checkpoint directory inside the export.
const VAE: &str = "vae";

/// The `.h3t` element-type codes this harness reads and writes, from `tools/dump_tables.py`'s table:
/// `0` f64, `1` f32, `2` i64, `3` i32, `4` bf16, `5` f16.
const H3T_F16: u8 = 5;
const H3T_F32: u8 = 1;
const H3T_BF16: u8 = 4;

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
    Decode,
}

struct Args {
    stage: Stage,
    root: PathBuf,
    store: Option<PathBuf>,
    device: i32,
    latent_height: u32,
    latent_width: u32,
    latents: Option<PathBuf>,
    /// An explicit arena size in GiB. Zero - the default - means the plan's own number, which is what
    /// a correct run needs; a stated size may only grow the reservation and the line says it did.
    arena_gib: u64,
    out: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            stage: Stage::Weights,
            root: PathBuf::from(DEFAULT_ROOT),
            store: std::env::var_os("UNISERVE_STORE").map(PathBuf::from),
            device: 0,
            latent_height: 16,
            latent_width: 16,
            latents: None,
            arena_gib: 0,
            out: PathBuf::from("artifacts/h3-vae-decode"),
        };
        let mut words = std::env::args().skip(1);
        while let Some(word) = words.next() {
            let mut value = |name: &str| -> Result<String> {
                words
                    .next()
                    .ok_or_else(|| Failure::message(format!("{name} needs a value")))
            };
            let mut number = |name: &str| -> Result<u32> {
                let text = value(name)?;
                text.parse().map_err(|_| {
                    Failure::message(format!("{name} takes a whole number, not `{text}`"))
                })
            };
            match word.as_str() {
                "--stage" => {
                    args.stage = match value("--stage")?.as_str() {
                        "weights" => Stage::Weights,
                        "decode" => Stage::Decode,
                        other => {
                            return Err(Failure::message(format!(
                                "unknown stage `{other}`; expected weights or decode"
                            )));
                        }
                    };
                }
                "--root" => args.root = PathBuf::from(value("--root")?),
                "--store" => args.store = Some(PathBuf::from(value("--store")?)),
                "--device" => {
                    let text = value("--device")?;
                    args.device = text.parse().map_err(|_| {
                        Failure::message(format!("--device takes a number, not `{text}`"))
                    })?;
                }
                "--latent-height" => args.latent_height = number("--latent-height")?,
                "--latent-width" => args.latent_width = number("--latent-width")?,
                "--latents" => args.latents = Some(PathBuf::from(value("--latents")?)),
                "--arena-gib" => {
                    args.arena_gib = u64::from(number("--arena-gib")?);
                }
                "--out" => args.out = PathBuf::from(value("--out")?),
                "--help" | "-h" => {
                    return Err(Failure::message(
                        "vae_decode --stage weights|decode [--root PATH] [--store PATH] [--device N] \
                         [--latent-height N] [--latent-width N] [--latents FILE] [--arena-gib N] \
                         [--out DIR]",
                    ));
                }
                other => {
                    return Err(Failure::message(format!(
                        "unknown argument `{other}`; try --help"
                    )));
                }
            }
        }
        if args.latent_height == 0 || args.latent_width == 0 {
            return Err(Failure::message(
                "the latent extents are the clip's own rows and columns; neither may be zero",
            ));
        }
        Ok(args)
    }

    fn configuration(&self) -> Result<minimax_h3::config::Config> {
        Ok(minimax_h3::config::Config::read(&self.root, None)?)
    }
}

/// What one eager decode bound: every launch's operator and descriptor key, in launch order.
///
/// A captured graph replays without calling `operate`, so a sink installed around a replay records
/// nothing; this stage is an eager decode, which is why the walk belongs here. The text is the store's
/// own key-list format - one `operator<TAB>key` per launch - so `tools/walk_gaps.py` reads this file
/// and `dit_layer`'s with one command.
struct Walk(Vec<(&'static str, String)>);

impl ops::capture::Sink for Walk {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator, key.to_owned()));
    }
}

impl Walk {
    fn text(&self) -> String {
        let mut text = String::new();
        for (operator, key) in &self.0 {
            text.push_str(operator);
            text.push('\t');
            text.push_str(key);
            text.push('\n');
        }
        text
    }

    fn distinct(&self) -> usize {
        let mut keys: Vec<(&str, &str)> = self
            .0
            .iter()
            .map(|(operator, key)| (*operator, key.as_str()))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys.len()
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    match args.stage {
        Stage::Weights => stage_weights(&args),
        Stage::Decode => stage_decode(&args),
    }
}

/// Stage 1: the names and the load, with no device.
///
/// Two checks in one pass. The name level is the whole of what an index-only host can answer: the 585
/// names the decoder reads against the export's `vae/` index, both directions. The tensor level is the
/// load the device path performs, into a store that validates each tensor's element type and shape as
/// the loader hands it over, so a transposed or truncated weight names itself instead of surfacing as
/// a wrong number 25 frames later.
fn stage_weights(args: &Args) -> Result<()> {
    let config = args.configuration()?;
    let index = index_names(&args.root.join(VAE))?;
    let asked = VideoDecoder::keys(&config.video_vae)?;
    let missing: Vec<&String> = asked.iter().filter(|name| !index.contains(name)).collect();
    // The encoder's 118 are the other half of `vae/` and are not the decoder's to read; they are
    // counted rather than listed, because a list of expected leftovers is a list that can be wrong.
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
    if !missing.is_empty() {
        return Err(Failure::message(format!(
            "{} of the {} tensors the video decoder reads are not in the export's `vae/` index; \
             the first is `{}`",
            missing.len(),
            asked.len(),
            missing[0]
        )));
    }

    let checkpoint = match Checkpoint::open(&args.root) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            println!("tensors: not read ({error})");
            return Ok(());
        }
    };
    let component = checkpoint.component(VAE)?;
    let (width, dtype, weight, bias) = folded_input(component, &config.video_vae)?;
    println!("folded input: {width} columns of {dtype} from the checkpoint's three affines");
    let canvas = Canvas::clip(&config.video_vae, args.latent_height, args.latent_width);
    let mut store = HostWeights::default();
    {
        let mut loader = WeightLoader::new(component, &mut store, None, None);
        let weight = loader.constant(
            "video_decoder.input.weight",
            dtype,
            Shape::new(&[width, u64::from(config.video_vae.latent_channels)]),
            &weight,
        )?;
        let bias = loader.constant(
            "video_decoder.input.bias",
            dtype,
            Shape::new(&[width]),
            &bias,
        )?;
        let input = nn::layers::Linear::new(weight, Some(bias), ops::gemm::WeightOrder::NxK)?;
        let source = checkpoint.component(VAE)?;
        let mut shards = loader.reborrow(source)?;
        let decoder = VideoDecoder::load(
            &config.video_vae,
            canvas,
            AttentionKind::Dense,
            dtype,
            input,
            &mut shards,
        )?;
        let _ = &decoder;
        println!(
            "loaded: {} layers, {} registers, clip {}x{} latent cells, {} frames out",
            config.video_vae.decoder_num_layers,
            config.video_vae.decoder_num_register_tokens,
            args.latent_height,
            args.latent_width,
            canvas.frames_out(&config.video_vae),
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

/// The export's `vae/` index, as the set of tensor names it declares.
///
/// Read as the JSON it is rather than through `Checkpoint::open`, because opening a checkpoint
/// validates that every file it maps is present: a host that holds an export's metadata and none of
/// its tensors can still answer whether a name is there.
fn index_names(directory: &Path) -> Result<Vec<String>> {
    let entries = std::fs::read_dir(directory)
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
            "{} holds no `*.index.json`: nothing states which tensors the VAE has",
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

/// Stage 2: one clip's arithmetic on a device.
///
/// The order is the runner's, done by hand: the device and its stream, the kernels, the weights, the
/// folded input affine, the positions table, the arena, the latents, then the decode. Each step's
/// failure names itself and nothing else, which is the point of the staging.
fn stage_decode(args: &Args) -> Result<()> {
    let store_root = args.store.clone().ok_or_else(|| {
        Failure::message(
            "stage `decode` needs a program store: pass --store <path> or set UNISERVE_STORE",
        )
    })?;
    let config = args.configuration()?;
    let canvas = Canvas::clip(&config.video_vae, args.latent_height, args.latent_width);
    let checkpoint = Checkpoint::open(&args.root)?;

    // 1. The device and the stream every launch is issued on, before any weight is allocated.
    let device = platform::Device::new(platform::DeviceId(args.device))?;
    let _guard = device.enter()?;
    let stream = device.stream()?;

    // 2. The kernels: every committed module the store holds, resolved before a byte is allocated,
    //    so a store that lacks a package is reported as coverage rather than as the first launch.
    let generation = load_generation(&device, &store_root)?;

    // 3. Weights on the device, through the loader's own boundary, and the folded input affine first:
    //    it is composed from the checkpoint's bytes at the loading boundary, so a decode never sees
    //    the reference's three separate steps.
    let component = checkpoint.component(VAE)?;
    let (width, dtype, weight, bias) = folded_input(component, &config.video_vae)?;
    // The latent container carries the decode's own element type, whatever the deployment states:
    // `dtype = "auto"` converts nothing, so the export's VAE representation *is* the decode's, and a
    // container of another width would write bytes labelled one width from a buffer of another - a
    // decode reading half a tensor, and a page of nonsense rather than an error. The type is named in
    // `case.json` and in the file's own element code, and a file that disagrees with it is refused
    // rather than reinterpreted.
    let container = element_of(dtype)?;
    let latent_channels = u64::from(config.video_vae.latent_channels);
    // The resident pool's size is the decode's own weights, summed from the index before a byte is
    // mapped: the pool reserves once and gives out of that reservation, so a reservation short of the
    // weights fails inside a publish and names a buffer rather than the plan. At a stated FP32 the
    // decoder is 9.7 GB, which no floor guessed here could cover (`--arena-gib` never sized it either:
    // that flag is the graph arena's).
    let needed = resident_bytes(component, VideoDecoder::keys(&config.video_vae)?)?;
    let mut weights = DeviceWeights::new(&device, needed)?;
    let (decoder, positions, rows) = {
        let mut loader = WeightLoader::new(component, &mut weights, None, None);
        let weight = loader.constant(
            "video_decoder.input.weight",
            dtype,
            Shape::new(&[width, latent_channels]),
            &weight,
        )?;
        let bias = loader.constant(
            "video_decoder.input.bias",
            dtype,
            Shape::new(&[width]),
            &bias,
        )?;
        let input = nn::layers::Linear::new(weight, Some(bias), ops::gemm::WeightOrder::NxK)?;
        let source = checkpoint.component(VAE)?;
        let mut shards = loader.reborrow(source)?;
        let decoder = VideoDecoder::load(
            &config.video_vae,
            canvas,
            AttentionKind::Dense,
            dtype,
            input,
            &mut shards,
        )?;
        // 4. The one table a decode reads: the per-voxel rotary positions, whose rows are the
        //    largest tile's - one table for every canvas of this grid.
        let tables = decoder.tables(args.latent_height, args.latent_width)?;
        let mut bytes: Vec<Vec<u8>> = tables.iter().map(|_| Vec::new()).collect();
        for (buffer, table) in bytes.iter_mut().zip(&tables) {
            buffer.resize(
                usize::try_from(table.shape.numel() * u64::from(table.dtype.bits()) / 8)
                    .unwrap_or(0),
                0,
            );
        }
        decoder.table_bytes(args.latent_height, args.latent_width, &mut bytes)?;
        let [table] = bytes.as_slice() else {
            return Err(Failure::message(format!(
                "the decoder declares {} layout tables; its decode reads one",
                bytes.len()
            )));
        };
        let rows = tables[0].shape.dim(1);
        let positions = loader.constant(
            "video_decoder.positions",
            tables[0].dtype,
            tables[0].shape,
            table,
        )?;
        let positions = positions.view();
        (decoder, positions, rows)
    };
    println!(
        "decoder: {} frames out, {} position rows, {} latent channels, {dtype}",
        canvas.frames_out(&config.video_vae),
        rows,
        latent_channels
    );

    // 5. The arena: `video_arena` is the plan the runner reserves by, and this harness uses it rather
    //    than an estimate, because an estimate that is short would surface as a reservation inside the
    //    decode and name a buffer rather than the plan.
    let plan = video_arena(&config.video_vae, &canvas)?;
    let mut planned = 0_u64;
    for buffer in &plan {
        planned += buffer.elements * buffer.width()?;
    }
    // The plan's own number. A stated `--arena-gib` may raise it - a real over-reserve is honest if it
    // says so - and the line below prints both, so a short plan is visible as a number rather than as
    // a reservation failure inside the decode.
    let arena_bytes = planned.max(args.arena_gib << 30);
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
    println!(
        "arena: {} buffers, {} MiB planned, {} MiB reserved{}",
        plan.len(),
        planned >> 20,
        arena_bytes >> 20,
        if arena_bytes > planned {
            " (the flag raised it; the plan alone is what the decode needs)"
        } else {
            ""
        }
    );

    // 6. The latents: the file the reference's own decoder is called with, packed into the rows this
    //    decoder reads. A missing file is generated from a fixed pattern instead, so the stage can run
    //    before the reference producer exists - and `case.json` says which of the two happened.
    let directory = args.out.join("video_decoder");
    std::fs::create_dir_all(&directory)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    let (latent_bytes, generated) = match &args.latents {
        Some(path) => (
            read_h3t(
                path,
                &[
                    1,
                    latent_channels,
                    canvas.frames.into(),
                    u64::from(args.latent_height),
                    u64::from(args.latent_width),
                ],
                container,
            )?,
            false,
        ),
        None => (
            pattern_latents(
                1,
                latent_channels,
                u64::from(canvas.frames),
                args.latent_height,
                args.latent_width,
                dtype,
            )?,
            true,
        ),
    };
    let tiles = Tiles::of(&config.video_vae, &canvas)?.tiles(&config.video_vae, &canvas);
    let packed = pack_latents(&latent_bytes, &tiles, &canvas)?;
    let voxels: u64 = tiles.iter().map(|tile| tile.canvas.voxels()).sum();
    let latents = upload(
        &stream,
        &arena,
        dtype,
        Shape::new(&[voxels, latent_channels]),
        &packed,
    )?;

    // 7. The decode: one clip, in tile order, with the tables read in order.
    let channels = u64::from(minimax_h3::output::Rgb::CHANNELS);
    let frames = u64::from(canvas.frames_out(&config.video_vae));
    let (height, width_px) = canvas.pixels(&config.video_vae);
    let out = arena.tensor(
        dtype,
        Shape::new(&[channels, frames, u64::from(height), u64::from(width_px)]),
    )?;
    let clip = ClipDecoder::new(&config.video_vae, dtype)?;
    // The walk: what this decode binds, recorded rather than inferred. `tools/walk_gaps.py` diffs it
    // against the store's key list, so a bound key the store has no program for is one command away
    // instead of one device round. Installed for the duration of the decode and removed whether it
    // succeeds or fails: a sink left installed would be reported to by every later launch.
    let mut walk = Walk(Vec::new());
    ops::capture::set_sink(Some(&mut walk));
    let decode_result = clip.decode_clip(&forward, &decoder, &tiles, latents, positions, out);
    ops::capture::set_sink(None);
    decode_result?;
    stream.synchronize()?;
    // The plan against what the decode actually bumped. `video_arena` is a sum of named buffers and
    // the arena only grows, so its cursor is the peak: an arena used past its plan would otherwise
    // surface as a pool refusing a give-out inside a layer, naming a buffer rather than the plan. This
    // line is the one that says which of the two happened.
    let peak = arena.used();
    println!(
        "arena used: {} MiB of {} MiB planned{}",
        peak >> 20,
        planned >> 20,
        if peak > planned {
            " - THE PLAN IS SHORT BY THE DIFFERENCE"
        } else {
            ""
        }
    );
    if walk.0.is_empty() {
        return Err(Failure::message(
            "the walk recorded no launches: the decode replayed a captured graph, or no sink was \
             installed",
        ));
    }

    // 8. What ran, written where the comparison reads it.
    let raster = download(
        &stream,
        out,
        usize::try_from(channels * frames * u64::from(height) * u64::from(width_px)).unwrap_or(0),
    )?;
    if raster.iter().all(|byte| *byte == 0) {
        return Err(Failure::message(
            "the raster is all zeros: the decode wrote nothing, and a comparison of two such trees \
             would be a comparison of two empty buffers",
        ));
    }
    // The one transpose: this side's `[channels, frames, rows x columns]` raster to the container's
    // `[frames, rows x columns, channels]`. Single site, named, and the only place the two layouts
    // meet.
    let transposed = channels_last(
        &raster,
        channels,
        frames,
        u64::from(height) * u64::from(width_px),
    );
    write_h3t(
        &directory.join("latents.h3t"),
        &[
            1,
            latent_channels,
            u64::from(canvas.frames),
            u64::from(args.latent_height),
            u64::from(args.latent_width),
        ],
        container,
        &latent_bytes,
    )?;
    write_h3t(
        &directory.join("raster.h3t"),
        &[frames, u64::from(height), u64::from(width_px), channels],
        element_of(dtype)?,
        &transposed,
    )?;
    let case = format!(
        "{{\"canvas\": {{\"height\": {}, \"width\": {}}}, \"latent_frames\": {}, \
         \"latent_height\": {}, \"latent_width\": {}, \"latent_dtype\": \"{}\", \
         \"latents\": \"{}\", \"tiles\": {}, \"position_rows\": {}}}\n",
        height,
        width_px,
        canvas.frames,
        args.latent_height,
        args.latent_width,
        dtype,
        if generated { "generated" } else { "supplied" },
        tiles.len(),
        rows,
    );
    std::fs::write(directory.join("case.json"), case).map_err(|error| {
        Failure::message(format!(
            "{}: {error}",
            directory.join("case.json").display()
        ))
    })?;
    let walk_path = args.out.join("walk.txt");
    std::fs::write(&walk_path, walk.text())
        .map_err(|error| Failure::message(format!("{}: {error}", walk_path.display())))?;
    println!(
        "walk: {} launches of {} distinct programs -> {}",
        walk.0.len(),
        walk.distinct(),
        walk_path.display()
    );
    println!(
        "wrote {}/raster.h3t: [{frames}, {height}, {width_px}, {channels}] from {} tiles, latents {}",
        directory.display(),
        tiles.len(),
        if generated {
            "generated (no --latents)"
        } else {
            "supplied"
        }
    );
    Ok(())
}

/// The `.h3t` element type of a dtype this harness writes.
fn element_of(dtype: DType) -> Result<u8> {
    match dtype {
        DType::F16 => Ok(H3T_F16),
        DType::F32 => Ok(H3T_F32),
        DType::BF16 => Ok(H3T_BF16),
        other => Err(Failure::message(format!(
            "the decode is {other}; this harness writes float16, bfloat16 and float32 only"
        ))),
    }
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

/// The bytes the decoder's weight iterations publish, summed from the index before a byte is read.
///
/// The pool reserves once, so this is the number that keeps a publish from failing: an index entry's
/// own `shape · dtype` is what the loader hands over for every tensor it reads verbatim. A weight-norm
/// pair is the exception - two tensors fold into the one weight the layer holds, so the pair's second
/// tensor is counted here and not published, and the sum is larger than the pool needs by those
/// vectors. Erring larger is the only direction that is safe: a sum short of the weights is the
/// failure this function exists to prevent.
fn resident_bytes(component: &nn::Checkpoint, keys: Vec<String>) -> Result<u64> {
    let mut total = 0_u64;
    for key in keys {
        let Some(location) = component.tensor(&key) else {
            // Absent names are the loader's to refuse, with the block that asks for them.
            continue;
        };
        total += location.shape.numel() * u64::from(location.dtype.bits()) / 8;
    }
    Ok(total)
}

/// Device storage for weights: a resident pool, one buffer per tensor, and every publish waited on.
struct DeviceWeights {
    pools: platform::Allocator,
    buffers: Vec<platform::Buffer>,
    bytes: u64,
}

impl DeviceWeights {
    fn new(device: &platform::Device, requested: u64) -> Result<Self> {
        let budget = std::sync::Arc::new(NoLimit);
        let mut pools = platform::Allocator::new(device, budget)?;
        // The requested size is the weights' own; a zero request - a caller that did not measure -
        // gets a floor, and the line below prints both numbers either way.
        let mut wanted = requested.max(1 << 30);
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
                "the driver refused every resident reservation down to 1 GiB: the decoder's \
                 weights cannot be held on this device",
            )
        })?;
        println!(
            "resident pool: {} MiB needed (the index), {} MiB reserved{}",
            requested >> 20,
            reserved >> 20,
            if reserved < requested {
                " - a short reservation refuses inside a publish, so an over-reserve here is the one                  that is honest"
            } else {
                ""
            }
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

/// Every committed module the store holds, one entry per descriptor directory.
///
/// The store's layout is `op/backend/revision/target/descriptor`, and which backend, revision and
/// target it holds is the deployment's business: this harness wants every committed descriptor,
/// because `ops` binds by key at the launch and a missing one is named there. A list of the operators
/// a decode uses would be a second copy of that answer and would drift from it.
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
    let operators = std::fs::read_dir(root)
        .map_err(|error| Failure::message(format!("{}: {error}", root.display())))?;
    for operator in operators.flatten() {
        let name = operator.file_name().to_string_lossy().into_owned();
        for (descriptor, directory) in committed(&operator.path()) {
            entries.push(install(device, &name, &descriptor, &directory)?);
            installed += 1;
        }
    }
    println!(
        "store: {installed} committed modules under {}",
        root.display()
    );
    if installed == 0 {
        return Err(Failure::message(format!(
            "{} holds no committed module: the stages past `weights` have nothing to launch",
            root.display()
        )));
    }
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
                    if descriptor.path().is_dir() {
                        found.push((
                            descriptor.file_name().to_string_lossy().into_owned(),
                            descriptor.path(),
                        ));
                    }
                }
            }
        }
    }
    found.sort();
    found
}

/// One committed module, installed through the API the Program Server's own loader uses.
fn install(
    device: &platform::Device,
    operator: &str,
    descriptor: &str,
    directory: &Path,
) -> Result<ops::KernelEntry> {
    let metadata = std::fs::read(directory.join("metadata.json")).ok();
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

/// Upload host bytes into the arena as a tensor of `shape`.
fn upload(
    stream: &platform::Stream,
    arena: &Arena,
    dtype: DType,
    shape: Shape,
    bytes: &[u8],
) -> Result<TensorView> {
    let tensor = arena.tensor(dtype, shape)?;
    // SAFETY: the destination is inside the arena and `bytes` outlives the copy, which is waited on.
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

/// Read a device tensor back to the host.
fn download(stream: &platform::Stream, tensor: TensorView, bytes: usize) -> Result<Vec<u8>> {
    let mut host = vec![0_u8; bytes];
    // SAFETY: `host` is `bytes` long and the source range is inside the tensor being read.
    unsafe {
        platform::memcpy_async(
            host.as_mut_ptr() as u64,
            tensor.ptr.address(),
            bytes,
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(host)
}

// ---------------------------------------------------------------------------------------------
// The diagnostic container and the two transforms between it and the decoder's own layouts.

/// Write the `.h3t` container: magic, element type, rank, extents, then the payload.
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

/// Read a `.h3t` container, refusing a shape or an element type this run cannot use.
fn read_h3t(path: &Path, shape: &[u64], dtype: u8) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path)
        .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?;
    if bytes.len() < 6 || &bytes[..4] != b"H3D1" {
        return Err(Failure::message(format!(
            "{}: not an `.h3t` container",
            path.display()
        )));
    }
    let held = bytes[4];
    let rank = usize::from(bytes[5]);
    let mut extents = Vec::with_capacity(rank);
    for index in 0..rank {
        let start = 6 + index * 8;
        let end = start + 8;
        let word = bytes.get(start..end).ok_or_else(|| {
            Failure::message(format!("{}: the extents are truncated", path.display()))
        })?;
        extents.push(u64::from_le_bytes(word.try_into().unwrap_or_default()));
    }
    if held != dtype || extents != shape {
        return Err(Failure::message(format!(
            "{}: holds {extents:?} of element type {held}, and this stage reads {shape:?} of {dtype}",
            path.display()
        )));
    }
    Ok(bytes[6 + rank * 8..].to_vec())
}

/// The latents a run without `--latents` uses: a fixed pattern, so two runs of this harness agree and
/// the number the comparison produces is about arithmetic rather than about a random draw.
fn pattern_latents(
    batch: u64,
    channels: u64,
    frames: u64,
    height: u32,
    width: u32,
    dtype: DType,
) -> Result<Vec<u8>> {
    let elements = batch * channels * frames * u64::from(height) * u64::from(width);
    let width_bytes = usize::try_from(dtype.bytes().unwrap_or(0)).unwrap_or(0);
    if width_bytes == 0 {
        return Err(Failure::message(format!(
            "the decode is {dtype}, which has no whole-byte element size"
        )));
    }
    let mut bytes = Vec::with_capacity(usize::try_from(elements).unwrap_or(0) * width_bytes);
    for index in 0..elements {
        // A sawtooth over the element index, in [-1, 1): the same arithmetic as the reference's own
        // pseudo-random input, without a generator whose implementation would have to match. The
        // value is rounded into the container's own type, so a bf16 run and an f16 run differ in the
        // rounding and not in the pattern - which is what makes two pages comparable.
        let value = ((index % 1024) as f32 / 512.0) - 1.0;
        match dtype {
            DType::F16 => bytes.extend_from_slice(&half_bits(value).to_le_bytes()),
            DType::BF16 => bytes.extend_from_slice(&bf16_bits(value).to_le_bytes()),
            DType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
            other => {
                return Err(Failure::message(format!(
                    "the decode is {other}; this harness writes float16, bfloat16 and float32 only"
                )));
            }
        }
    }
    Ok(bytes)
}

/// The nearest bfloat16 bit pattern of `value`: the high half of its float32, rounded half to even.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let upper = (bits >> 16) as u16;
    let remainder = bits & 0xffff;
    if remainder > 0x8000 || (remainder == 0x8000 && upper & 1 == 1) {
        upper.wrapping_add(1)
    } else {
        upper
    }
}

/// The nearest float16 bit pattern of `value`, by the round-to-nearest-even rule `DType::F16` keeps.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00;
    }
    let shifted = exponent - 127 + 15;
    if shifted >= 0x1f {
        return sign | 0x7c00;
    }
    if shifted <= 0 {
        return sign;
    }
    let mut half = ((shifted as u32) << 10) | (mantissa >> 13);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half as u16
}

/// Pack `[channels, frames, height, width]` float16 into the rows this decoder reads, in tile order.
fn pack_latents(
    bytes: &[u8],
    tiles: &[minimax_h3::video_vae::Tile],
    canvas: &Canvas,
) -> Result<Vec<u8>> {
    let channels = 24_u64;
    let frames = u64::from(canvas.frames);
    let height = u64::from(canvas.height);
    let width = u64::from(canvas.width);
    let expected = channels * frames * height * width * 2;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != expected {
        return Err(Failure::message(format!(
            "the latents hold {} bytes and a {frames}x{height}x{width} clip of {channels} channels \
             needs {expected}",
            bytes.len()
        )));
    }
    let mut packed = Vec::with_capacity(bytes.len());
    for tile in tiles {
        let (row_start, row_end) = tile.latent_rows;
        let (column_start, column_end) = tile.latent_columns;
        for frame in 0..frames {
            for row in u64::from(row_start)..u64::from(row_end) {
                for column in u64::from(column_start)..u64::from(column_end) {
                    let at = ((frame * height + row) * width + column) * channels * 2;
                    let start = usize::try_from(at).unwrap_or(0);
                    let end = start + usize::try_from(channels * 2).unwrap_or(0);
                    packed.extend_from_slice(&bytes[start..end]);
                }
            }
        }
    }
    Ok(packed)
}

/// `[channels, frames, rows x columns]` to `[frames, rows x columns, channels]`, both float16.
fn channels_last(bytes: &[u8], channels: u64, frames: u64, pixels: u64) -> Vec<u8> {
    let mut out = vec![0_u8; bytes.len()];
    for channel in 0..channels {
        for frame in 0..frames {
            for pixel in 0..pixels {
                let source = (((channel * frames + frame) * pixels + pixel) * 2) as usize;
                let target = (((frame * pixels + pixel) * channels + channel) * 2) as usize;
                out[target..target + 2].copy_from_slice(&bytes[source..source + 2]);
            }
        }
    }
    out
}
