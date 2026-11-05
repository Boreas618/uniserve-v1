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
//!
//! **The stitch's layout.** `--stage decode` decodes the clip's tiles in `Tile` order, because the
//! stitch reads what the tile above and the tile to its left have already written, and it hands the
//! stitch the plan's tiled layout: `latents` in tile order, the largest tile's position table, and the
//! clip's raster. A tile's neighbour is not one contiguous block of a raster wider than the tile, and
//! the blend addresses its sources flat, so the stitch copies the neighbour into one tile-shaped block
//! first - which is why the raster may be any width and why the walk states a `copy_block` row.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::audio_vae::BigVgan;
use minimax_h3::config::AttentionKind;
use minimax_h3::config::offered_canvases;
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
/// The audio VAE's directory in the checkpoint, which is also its component name: the video VAE and
/// the audio VAE are two components of one export, each with its own configuration and shards.
const AUDIO: &str = "audio_vae";

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

/// The stages of the run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Weights,
    Decode,
    Raster,
    Plan,
    Audio,
}

#[derive(Clone)]
struct Args {
    stage: Stage,
    root: PathBuf,
    store: Option<PathBuf>,
    device: i32,
    latent_height: u32,
    latent_width: u32,
    latents: Option<PathBuf>,
    /// The raster whose pixels `--stage raster` encodes: `[frames, rows, columns, channels]` FP32,
    /// one still (one frame) or one clip of the model's clip length - the counts the video encoder
    /// reads. Absent with `--pattern`, the stage draws one instead.
    pixels: Option<PathBuf>,
    /// Draw the raster to encode rather than reading one.
    ///
    /// A decode's own `raster.h3t` is **longer than a clip** - the decoder emits `frames_out` frames
    /// and the clip keeps fewer - so a decoded raster is not an encoder input. This flag is the way to
    /// run the raster path before a reference clip exists: the same habit the decode stage has with a
    /// missing latent, and the page says which of the two produced the pixels.
    pattern: bool,
    /// The latent frames a decode reads, when it is not one clip's.
    ///
    /// A decode's canvas states how many latent frames it decodes, and one clip of the checkpoint is
    /// [`minimax_h3::video_vae::decode_latents`] of them. A reference clip encoded by `--stage raster`
    /// is a *different* count - the video encoder's own frame arithmetic (`clip_length` frames in,
    /// `token_drop` fewer latent frames out) - so a decode of one says its count here rather than
    /// asserting the clip's. Absent, the clip's own number is used, which is what every decode of a
    /// generated trajectory wants.
    frames: Option<u32>,
    /// The latent frames `--stage audio` decodes: one row per stereo channel, `frames` steps of the
    /// decoder's own hop. The reference probe's own default is twenty, so a comparison that does not
    /// name one agrees with the tool's catalogue.
    audio_frames: u32,
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
            pixels: None,
            pattern: false,
            frames: None,
            audio_frames: 20,
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
                        "raster" => Stage::Raster,
                        "plan" => Stage::Plan,
                        "audio" => Stage::Audio,
                        other => {
                            return Err(Failure::message(format!(
                                "unknown stage `{other}`; expected weights, decode, raster, plan or \
                                 audio"
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
                "--pixels" => args.pixels = Some(PathBuf::from(value("--pixels")?)),
                "--pattern" => args.pattern = true,
                "--frames" => args.frames = Some(number("--frames")?),
                "--audio-frames" => args.audio_frames = number("--audio-frames")?,
                "--arena-gib" => {
                    args.arena_gib = u64::from(number("--arena-gib")?);
                }
                "--out" => args.out = PathBuf::from(value("--out")?),
                "--help" | "-h" => {
                    return Err(Failure::message(
                        "vae_decode --stage weights|decode|raster|audio [--root PATH] [--store PATH] \
                         [--device N] [--latent-height N] [--latent-width N] [--latents FILE] \
                         [--pixels FILE | --pattern] [--frames N] [--audio-frames N] \
                         [--arena-gib N] [--out DIR]",
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
        Stage::Raster => stage_raster(&args),
        Stage::Plan => stage_plan(&args),
        Stage::Audio => stage_audio(&args),
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
            "minimax-h3",
        )?;
        let bias = loader.constant(
            "video_decoder.input.bias",
            dtype,
            Shape::new(&[width]),
            &bias,
            "minimax-h3",
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
            "loaded: {} layers, {} registers, clip {}x{} latent cells, {} frames kept of {} decoded",
            config.video_vae.decoder_num_layers,
            config.video_vae.decoder_num_register_tokens,
            args.latent_height,
            args.latent_width,
            kept_frames(&config.video_vae),
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
    // One clip's canvas, unless the caller states the latent frames: a reference clip encoded by
    // `--stage raster` is the video encoder's own count, which is not one clip's.
    let canvas = {
        let mut canvas = Canvas::clip(&config.video_vae, args.latent_height, args.latent_width);
        if let Some(frames) = args.frames {
            canvas.frames = frames;
        }
        canvas
    };
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
    let granularity =
        u64::try_from(platform::HandleType::Descriptor.granularity(device.id())?).unwrap_or(1);
    // The loader also publishes what the export does not hold, and the pool charges a block for every
    // one of them: the input affine this stage folds on the host (`folded_input`, two tensors) and the
    // constants the decoder declares in `video_vae::constants` (the zero row and rows, the two index
    // tables, the rotary frequencies, the voxel rows and the sequence offsets). A reservation built
    // from the checkpoint's names alone is short by every one of these - the failure that made this
    // stage stop on the 588th tensor - and the decoder's list is the one statement of them, checked at
    // each publish. No count is written here: the list is iterated, so a constant added to it is
    // covered without an edit to this comment or this sum.
    let affine = (width * latent_channels + width) * u64::from(dtype.bits()) / 8;
    let mut synthesized = affine.max(1).next_multiple_of(granularity);
    for (_, bytes) in minimax_h3::video_vae::constants(&config.video_vae, canvas, width, dtype)? {
        synthesized += bytes.max(1).next_multiple_of(granularity);
    }
    let needed = resident_bytes(
        component,
        VideoDecoder::keys(&config.video_vae)?,
        granularity,
    )? + synthesized;
    let mut weights = DeviceWeights::new(&device, needed, granularity)?;
    let (decoder, positions, rows) = {
        let mut loader = WeightLoader::new(component, &mut weights, None, None);
        let weight = loader.constant(
            "video_decoder.input.weight",
            dtype,
            Shape::new(&[width, latent_channels]),
            &weight,
            "minimax-h3",
        )?;
        let bias = loader.constant(
            "video_decoder.input.bias",
            dtype,
            Shape::new(&[width]),
            &bias,
            "minimax-h3",
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
            "minimax-h3",
        )?;
        let positions = positions.view();
        (decoder, positions, rows)
    };
    // What the reservation was for, against what it held: the two numbers a reader needs to tell a
    // tight pool from a wrong one, both about publishing and nothing else.
    println!(
        "weights: {} MiB published of {} MiB reserved, {} MiB mapped",
        weights.bytes >> 20,
        weights.reserved >> 20,
        weights.pools.mapped_bytes(platform::PoolKind::Resident) >> 20
    );
    // The decoder decodes more frames than the clip keeps: the temporal upsampler emits
    // `latent_frames · ratio` and the clip's own length crops the padding off. Both numbers are printed
    // because a page comparing a raster against "the frames out" reads the wrong count otherwise - 28
    // decoded against 25 kept is exactly the kind of pair this round has spent rounds on.
    let decoded = canvas.frames_out(&config.video_vae);
    let kept = kept_frames(&config.video_vae);
    println!(
        "decoder: {kept} frames out ({decoded} decoded, {} trimmed), {} position rows, \
         {} latent channels, {dtype}",
        decoded - u32::try_from(kept).unwrap_or(decoded),
        rows,
        latent_channels
    );

    // The tile grid, before the arena: the plan and the harness's own latent buffer both count the
    // tiles' voxels, so the grid is read once and both readings use it.
    let tiles = Tiles::of(&config.video_vae, &canvas)?.tiles(&config.video_vae, &canvas);
    let voxels: u64 = tiles.iter().map(|tile| tile.canvas.voxels()).sum();
    // 5. The arena: `video_arena` is the plan the runner reserves by, and this harness uses it rather
    //    than an estimate, because an estimate that is short would surface as a reservation inside the
    //    decode and name a buffer rather than the plan.
    let plan = video_arena(&config.video_vae, &canvas, dtype)?;
    // **Each buffer at the arena's alignment, not at its raw size.** The arena aligns every give-out
    // to `nn::forward::ALIGNMENT`, so the raw sum is short by up to 255 bytes per buffer - and the
    // first run to meet it was 64 bytes short of a 372 MiB region, which is the two rotary tables'
    // padding and nothing else. `arena_reservation` is that sum, so a runner that reserves by the
    // plan cannot be short by an alignment it did not know about.
    let planned = minimax_h3::decoding::arena_reservation(&plan);
    let raw: u64 = plan.iter().map(minimax_h3::decoding::Buffer::bytes).sum();
    // **The harness is also the runner.** The latents it uploads and the raster it decodes into are,
    // in service, the request's latent and `DecodedMedia.video` - the caller's buffers, which the
    // runner allocates and the plan deliberately does not count (`video_arena`'s note). Here they come
    // out of this region because the harness has one device pointer to give, so they are counted
    // separately and named: a harness that reserved the plan alone would be short by its own two
    // buffers, and the failure would read as the plan again.
    let element_bytes = dtype.bytes().ok_or_else(|| {
        Failure::message(format!(
            "the decode is {dtype}, which has no whole-byte element size"
        ))
    })?;
    let (pixel_height, pixel_width) = canvas.pixels(&config.video_vae);
    let held = element_bytes
        * (voxels * latent_channels
            + u64::from(minimax_h3::output::Rgb::CHANNELS)
                * u64::from(canvas.frames_out(&config.video_vae))
                * u64::from(pixel_height)
                * u64::from(pixel_width));
    // The plan's own number plus what this harness holds. A stated `--arena-gib` may raise it - a real
    // over-reserve is honest if it says so - and the line below prints all three, so a short
    // reservation is visible as a number rather than as a failure inside the decode.
    let reserved = planned + held;
    let arena_bytes = reserved.max(args.arena_gib << 30);
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
        "arena: {} buffers, {} MiB planned at {} ({} KiB of alignment), {} KiB held by this \
         harness, {} MiB reserved{}",
        plan.len(),
        planned >> 20,
        minimax_h3::text_encoder::element_of(dtype),
        (planned - raw) >> 10,
        held >> 10,
        arena_bytes >> 20,
        if arena_bytes > reserved {
            " (the flag raised it; the plan and this harness's two buffers are what the decode needs)"
        } else {
            ""
        }
    );

    // **Where each buffer will land.** A stop inside the arena names a size and an offset, and an
    // offset is only readable against the placement that produced it - the harness's two buffers
    // first, then the plan's entries at the arena's alignment. Printing it costs sixteen lines and
    // turns "an arena that cannot give out" into a name and a number the moment it happens.
    {
        let mut at = held;
        for buffer in &plan {
            at = at.next_multiple_of(nn::forward::ALIGNMENT);
            println!(
                "  arena {:<22} {:>12} B at {at}",
                buffer.name,
                buffer.bytes()
            );
            at += buffer.bytes();
        }
    }

    // 6. The latents: the file the reference's own decoder is called with, packed into the rows this
    //    decoder reads. A missing file is generated from a fixed pattern instead, so the stage can run
    //    before the reference producer exists - and `case.json` says which of the two happened.
    let directory = args.out.join("video_decoder");
    std::fs::create_dir_all(&directory)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    // The page is this run's or it does not exist. A decode that stops in the loader - or anywhere
    // before its last write - would otherwise leave the previous run's `latents.h3t`, `raster.h3t` and
    // `case.json` in place, and a comparator pairing those would be reporting a residual between two
    // runs nobody can attribute. Clearing first makes "a page exists" mean "this run produced it".
    let mut cleared = 0_u32;
    for name in ["latents.h3t", "raster.h3t", "case.json"] {
        let path = directory.join(name);
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?;
            cleared += 1;
        }
    }
    if cleared > 0 {
        println!(
            "cleared {cleared} file(s) of a previous run from {}",
            directory.display()
        );
    }
    // The clip's latent frames, named once: the messages below state the container in both of the
    // forms a caller may hand this stage, and a bare `{}` twice is a place a wrong number can hide.
    let latent_frames = canvas.frames;
    let (latent_bytes, generated) = match &args.latents {
        Some(path) if !path.exists() && is_index(path) => {
            // A bare number is an *evaluation index*, not a path, and three runs have now stopped on
            // one: the flag is the only place in this stage a caller names an input, so it is where
            // the confusion lands. Say the file the index would name, so the corrected command is
            // visible rather than inferable.
            return Err(Failure::message(format!(
                "`--latents {}`: that is an evaluation index, not a path. The trajectory file stage 3 \
                 writes for evaluation {} is `<stage3-out>/latents_video_e{}.h3t` - pass that path, \
                 or leave `--latents` out and this stage draws and writes its own latent",
                path.display(),
                path.display(),
                path.display(),
            )));
        }
        Some(path) if !path.exists() => {
            // An input the caller named is read, never substituted - but the operating system's own
            // message says only `No such file`, and a flag given a word that is not a path (`5`, or a
            // variable that expanded empty) reads exactly like a missing file. Naming the flag, the
            // value and what the stage expects is the difference between a command to correct and a
            // defect to hunt.
            return Err(Failure::message(format!(
                "`--latents {}`: no such file. The flag takes an `.h3t` container holding either the \
                 clip's `[1, {latent_channels}, {latent_frames}, H, W]` latents or a denoiser's trajectory \
                 `[{latent_channels}, {latent_frames}, H, W]` in FP32, which is what stage 3 writes - leave it \
                 out and the stage draws its fixed pattern instead, and writes it to \
                 <out>/video_decoder/latents.h3t",
                path.display(),
            )));
        }
        Some(path) => {
            // Two forms are read: this stage's own `[1, C, frames, H, W]` at the decode's element type,
            // and the **trajectory's** `[C, frames, H, W]` as a denoiser writes it - FP32, because the
            // sample is. The second is narrowed to the decode's type here and says so, so a pipeline of
            // stage 3 then stage 2 is one command rather than a hand conversion nobody records.
            let (held, extents, payload) = read_container(path)?;
            let wanted = [
                1,
                latent_channels,
                canvas.frames.into(),
                u64::from(args.latent_height),
                u64::from(args.latent_width),
            ];
            let trajectory = [
                latent_channels,
                canvas.frames.into(),
                u64::from(args.latent_height),
                u64::from(args.latent_width),
            ];
            let form = latent_form(held, &extents, &wanted, &trajectory, container);
            if form == LatentForm::Own {
                (payload, false)
            } else if form == LatentForm::Trajectory {
                println!(
                    "latents: {} - trajectory form {extents:?} of element code {held}, converted to \
                     {dtype}",
                    path.display()
                );
                (convert_latents(&payload, held, dtype)?, false)
            } else {
                return Err(Failure::message(format!(
                    "{}: holds {extents:?} of element type {held}. This stage reads its own \
                     {wanted:?} of {container}, or a denoiser's trajectory {trajectory:?} of code \
                     {CONTAINER_F32} (FP32), which it narrows to {dtype}",
                    path.display()
                )));
            }
        }
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
    // What the input *was*, when it was supplied: a decode of a stale trajectory file is a page no
    // run produced, and the file's own size and modification time are what say so - the harness cannot
    // know which run wrote it, and a name alone does not age.
    let provenance = match &args.latents {
        Some(path) => {
            let metadata = std::fs::metadata(path)
                .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?;
            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |elapsed| elapsed.as_secs());
            println!(
                "latents: {} - {} bytes, modified {} s after the epoch",
                path.display(),
                metadata.len(),
                modified
            );
            format!(
                "\"latents_source\": {{\"path\": \"{}\", \"bytes\": {}, \"modified_s\": {}}}, ",
                path.display(),
                metadata.len(),
                modified
            )
        }
        None => String::new(),
    };

    let packed = pack_latents(&latent_bytes, &tiles, &canvas, dtype)?;
    // **Both counts in the same unit, and the excess named.** A tile set reads more voxels than the
    // clip holds because neighbouring tiles share rows and columns for the blend to cross; the factor
    // is the grid's overlap and nothing else. Printed as one number in bytes beside one in elements it
    // read as a mismatch of 4x, which is a defect this line invented rather than found.
    let element = dtype.bytes().unwrap_or(1);
    let clip_elements = u64::try_from(latent_bytes.len()).unwrap_or(0) / element;
    let tile_elements = u64::try_from(packed.len()).unwrap_or(0) / element;
    let excess = if clip_elements == 0 {
        1.0
    } else {
        tile_elements as f64 / clip_elements as f64
    };
    println!(
        "latents: {} bytes = {clip_elements} elements at {dtype}; the {} tiles read {} bytes = \
         {tile_elements} elements ({excess:.3}x), and the excess is the rows and columns neighbouring \
         tiles share",
        latent_bytes.len(),
        tiles.len(),
        packed.len(),
    );
    // The grid's spans are the check, not the sentence: every tile reads a span of the clip, so the
    // packed count is the sum of those spans over the frames and channels, and a gather that read the
    // wrong span - a stride where a width belongs - shows here as a count rather than as a raster that
    // is wrong in bands nobody looked at.
    let spanned: u64 = tiles
        .iter()
        .map(|tile| {
            // `(start, end)`, so the span is the difference: multiplying the end is the mistake this
            // check exists to refuse, and it would refuse every run if the fields were read as extents.
            let rows = u64::from(tile.latent_rows.1.saturating_sub(tile.latent_rows.0));
            let columns = u64::from(tile.latent_columns.1.saturating_sub(tile.latent_columns.0));
            rows * columns * 24 * u64::from(canvas.frames)
        })
        .sum();
    if spanned != tile_elements {
        return Err(Failure::message(format!(
            "the {} tiles' spans cover {spanned} elements of a {clip_elements}-element clip, and the \
             packed rows hold {tile_elements}: the excess is the grid's overlap and nothing else, so \
             the gather read a span the grid does not name",
            tiles.len()
        )));
    }
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
    // The walk is written before the outcome is read: the launch after its last line is the one that
    // stopped, so a decode that fails still says where it got to. A walk written only on success makes
    // a stop indistinguishable from a stage that never opened the device.
    let walk_path = args.out.join("walk.txt");
    std::fs::write(&walk_path, walk.text())
        .map_err(|error| Failure::message(format!("{}: {error}", walk_path.display())))?;
    if let Err(error) = &decode_result {
        // **The stop is named here, not left to a line of its own.** A walk of zero launches before a
        // stop is a fact about the decode's first launch, not about its programs: it failed while
        // binding its tables or its first program, so no launch ever reached the recorder. The two
        // read identically in a summary, and they are opposite diagnoses.
        println!(
            "walk: {} launches of {} distinct programs before the stop: {error} -> {}",
            walk.0.len(),
            walk.distinct(),
            walk_path.display()
        );
        if walk.0.is_empty() {
            println!(
                "  the decode stopped before its first launch, so this walk says nothing about the \
                 programs it would have bound"
            );
        }
        for (operator, key) in walk.0.iter().rev().take(5).rev() {
            println!("  {operator}\t{key}");
        }
    }
    decode_result?;
    stream.synchronize()?;
    // The plan against what the decode actually bumped. `video_arena` is a sum of named buffers and
    // the arena only grows, so its cursor is the peak: an arena used past its plan would otherwise
    // surface as a pool refusing a give-out inside a layer, naming a buffer rather than the plan. This
    // line is the one that says which of the two happened.
    let peak = arena.used();
    println!(
        "arena used: {} MiB of {} MiB reserved{}",
        peak >> 20,
        reserved >> 20,
        if peak > reserved {
            " - THE RESERVATION IS SHORT BY THE DIFFERENCE"
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
    // **The raster in numpy's own container, beside the crate's.** `tools/diagnosis/compare.py`
    // pairs the two implementations' trees by relative path and reads `.npy`; a raster written only
    // as `.h3t` would appear there as a file the reference has and this crate does not - a false
    // absence in a delivery artifact, which is worse than a missing row because it reads as a defect
    // in the decoder. The two files hold the same numbers: the payload is the tensor's bytes, and the
    // widening to `f32` is this container's one requirement (`diagnosis::write_npy`).
    let raster_shape = [frames, u64::from(height), u64::from(width_px), channels];
    minimax_h3::diagnosis::write_npy(
        &directory.join("raster.npy"),
        &raster_shape,
        &widen_to_f32(&transposed, dtype)?,
    )
    .map_err(Failure::message)?;
    // **The seams, stated where the comparison reads them.** A blend's arithmetic is invisible in a
    // whole-raster residual: a seam that is faded with the wrong weight moves a band of a few dozen
    // pixels, and the maximum over a 25-frame raster is somewhere else. The grid knows where every
    // overlap is - each tile names the band it shares with the tile above and to its left, in output
    // pixels - so the case states the bands and the comparator reports a row per band. The reference
    // probe copies this `case.json` verbatim, so both sides are compared on the same bands rather
    // than on two readings of the grid.
    let ratio = config.video_vae.spatial_compression();
    let mut seams = Vec::new();
    for tile in &tiles {
        for (axis, extent, start) in [
            ("rows", tile.above, tile.latent_rows.0),
            ("columns", tile.left, tile.latent_columns.0),
        ] {
            if extent == 0 {
                continue;
            }
            let offset = start * ratio;
            seams.push(format!(
                "{{\"axis\": \"{axis}\", \"start\": {offset}, \"end\": {}}}",
                offset + extent
            ));
        }
    }
    let case = format!(
        "{{\"canvas\": {{\"height\": {}, \"width\": {}}}, \"latent_frames\": {}, \
         \"latent_height\": {}, \"latent_width\": {}, \"latent_dtype\": \"{}\", \
         \"latents\": \"{}\", {provenance}\"tiles\": {}, \"position_rows\": {}, \
         \"seams\": [{}], \"checkpoint\": \"{}\"}}\n",
        height,
        width_px,
        canvas.frames,
        args.latent_height,
        args.latent_width,
        dtype,
        if generated { "generated" } else { "supplied" },
        tiles.len(),
        rows,
        seams.join(", "),
        args.root.display(),
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
        "wrote {}/raster.h3t and raster.npy: [{frames}, {height}, {width_px}, {channels}] from {} \
         tiles, latents {}",
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

/// Stage 3: the raster path - `ref2va`'s video half, a reference clip's pixels to a latent and that
/// latent back to a raster.
///
/// The decoder has its own stage and the video encoder is driven by the conditioner, but **no stage
/// had the two meet in one process**, so the pair `ref2va` runs in sequence had never been measured
/// against itself. This stage is that pair in one command: the raster's pixels are cut into the
/// encoder's own tiles, encoded, projected and sampled into a latent, and the latent is handed to the
/// decode of `--stage decode`, which runs here in process on the file this stage writes.
///
/// **Two frame arithmetics, and neither is the other.** The video encoder reads one still or one clip
/// of `clip_length` frames and yields `token_drop` fewer latent frames
/// ([`minimax_h3::encoding::VideoEncoder::latent_frames`]); the decoder decodes `canvas.frames` latent
/// frames into `frames_out`. So this stage encodes the container's frame count, decodes the count the
/// encoder states, and prints both: a decode told a clip's own number would read past its latent.
///
/// The input is the container `--stage decode` writes as `<out>/video_decoder/raster.h3t` -
/// `[frames, rows, columns, channels]` FP32 - so the two stages chain without a conversion nobody
/// recorded.
/// Print the decoder's arena plan, and the tile grid it is derived from, with no device and no weights.
///
/// The plan is what the executor sizes the arena by, so a plan that is wrong is a run that stops with
/// "the arena cannot give out" partway through a clip - the most expensive place to learn it. Every
/// entry here is computed from the config and the canvas alone, so this stage answers on a host with no
/// GPU, no store and no checkpoint shards, and it answers for a canvas the caller names rather than for
/// the one this machine happens to serve.
fn stage_plan(args: &Args) -> Result<()> {
    let config = args.configuration()?;
    let canvas = {
        let mut canvas = Canvas::clip(&config.video_vae, args.latent_height, args.latent_width);
        if let Some(frames) = args.frames {
            canvas.frames = frames;
        }
        canvas
    };
    // The video VAE is F32 in every export this model serves (`component-tensors.json`: 703 tensors,
    // all F32), and `--stage decode` pins the latent container to the checkpoint's own type for the
    // same reason. The plan's element types follow the decode's, so the plan is stated at that type.
    let dtype = nn::DType::F32;
    let (height, width) = canvas.pixels(&config.video_vae);
    let tiles = minimax_h3::video_vae::Tiles::of(&config.video_vae, &canvas)?;
    let grid = tiles.tiles(&config.video_vae, &canvas);
    println!(
        "canvas: {} latent frames of {} x {} -> {} frames of {} x {} pixels, {} tile(s)",
        canvas.frames,
        canvas.height,
        canvas.width,
        canvas.frames_out(&config.video_vae),
        height,
        width,
        grid.len()
    );
    let plan = video_arena(&config.video_vae, &canvas, dtype)?;
    let mut total = 0_u64;
    for buffer in &plan {
        let bytes = buffer.bytes();
        total += buffer.reservation();
        println!(
            "  {:<14} {:<4} {:>12} elements {:>14} bytes",
            buffer.name, buffer.element, buffer.elements, bytes
        );
    }
    println!(
        "plan: {} buffer(s), {} bytes = {} MiB (reserved {} MiB in the arena)",
        plan.len(),
        total,
        total / (1 << 20),
        plan.iter().map(|b| b.reservation()).sum::<u64>() / (1 << 20)
    );
    Ok(())
}

fn stage_raster(args: &Args) -> Result<()> {
    let (latent_frames, latent_height, latent_width, latent_path) = encode_raster(args)?;
    // The decode is the `decode` stage, unchanged: the same arithmetic, the same page, the same
    // comparison. It is given the latent this stage wrote and the count the encoder stated.
    let mut decoded = args.clone();
    decoded.latents = Some(latent_path.clone());
    decoded.frames = Some(latent_frames);
    // The latent's own extents, which the encoder derived from the raster rather than from a flag: a
    // decode told the flag's default would read a canvas the encode never produced.
    decoded.latent_height = latent_height;
    decoded.latent_width = latent_width;
    println!(
        "raster: encoded to {} latent frames -> {}; decoding them with the same stage a generated \
         trajectory uses",
        latent_frames,
        latent_path.display()
    );
    stage_decode(&decoded)
}

/// Stage 5: one audio track's decode on a device, for the reference's block comparison.
///
/// The audio decoder's contract differs from the video one in three ways this stage states rather than
/// inherits, and each is a place a second implementation would drift: its latents are
/// `[2, channels, frames]` - one row per stereo channel - its output is interleaved `i16` PCM rather
/// than a raster, and it reads no layout table, so `state` is empty and the plan is the whole of what
/// it needs besides the caller's two buffers.
///
/// **It writes the latent it decoded as well as the PCM.** A stage comparison pairs two trees on one
/// input; the input the reference gets is this draw and not a second one, so the draw is a file rather
/// than a habit - `latents.h3t` is what `tools/dump_vae_blocks.py --probe decode_audio` reads.
fn stage_audio(args: &Args) -> Result<()> {
    use nn::capability::{DecodedMedia, Decoder as _, LatentShape, Layout};

    let store_root = args.store.clone().ok_or_else(|| {
        Failure::message(
            "stage `audio` needs a program store: pass --store <path> or set UNISERVE_STORE",
        )
    })?;
    // The audio VAE is its own component of the export: its configuration is a file of its own, and
    // the decoder's plan is a function of the track's latent frames and nothing else.
    let config = {
        let path = args.root.join(AUDIO).join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?;
        let values: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?;
        minimax_h3::AudioVaeConfig::parse(&values)
            .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?
    };
    let frames = u64::from(args.audio_frames);
    let channels = u64::from(config.latent_channels);
    let samples = minimax_h3::audio_vae::track_samples(&config, args.audio_frames);
    let checkpoint = Checkpoint::open(&args.root)?;

    // 1. The device and the stream, before any weight is allocated.
    let device = platform::Device::new(platform::DeviceId(args.device))?;
    let _guard = device.enter()?;
    let stream = device.stream()?;

    // 2. The kernels: every committed module the store holds, so a store that lacks a package is
    //    coverage rather than the first launch's failure.
    let generation = load_generation(&device, &store_root)?;

    // 3. The weights, from the component's own key list before a byte is mapped: the pool reserves once,
    //    and a reservation short of the weights fails inside a publish naming a buffer rather than the
    //    list.
    let component = checkpoint.component(AUDIO)?;
    let granularity =
        u64::try_from(platform::HandleType::Descriptor.granularity(device.id())?).unwrap_or(1);
    let needed = resident_bytes(component, BigVgan::keys(&config)?, granularity)?;
    let mut weights = DeviceWeights::new(&device, needed, granularity)?;
    let decoder = {
        let mut loader = WeightLoader::new(component, &mut weights, None, None);
        BigVgan::load(&config, &mut loader)?
    };
    println!(
        "weights: {} MiB published of {} MiB reserved, {} MiB mapped",
        weights.bytes >> 20,
        weights.reserved >> 20,
        weights.pools.mapped_bytes(platform::PoolKind::Resident) >> 20
    );

    // 4. The arena: the plan the capability states for this track, plus this harness's own two
    //    buffers. The latents it uploads and the PCM window the decode writes are the caller's in
    //    service - the runner allocates them and the plan deliberately does not count them - so they
    //    are counted here and the line says which is which.
    let layout = Layout {
        video: LatentShape {
            channels: 0,
            frames: 0,
            height: 0,
            width: 0,
        },
        audio: Some(LatentShape {
            channels: config.latent_channels,
            frames: args.audio_frames,
            height: 0,
            width: 0,
        }),
        text_rows: 0,
        condition_rows: 0,
        condition_elements: 0,
    };
    let planned = decoder.arena_bytes(&layout)?;
    let held = 4 * u64::from(minimax_h3::audio_vae::PCM_CHANNELS) * channels * frames
        + 2 * u64::from(minimax_h3::audio_vae::PCM_CHANNELS) * samples;
    let arena_bytes = (planned + held).max(args.arena_gib << 30);
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
        "arena: {} MiB planned, {} KiB held by this harness, {} MiB reserved",
        planned >> 20,
        held >> 10,
        arena_bytes >> 20
    );

    // 5. The latents, drawn by the device's Philox at the posterior's seed - the same habit the raster
    //    stage has, so the track is reproducible - and then written out, because the reference decodes
    //    *this* draw rather than one of its own.
    let latents = arena.tensor(
        DType::F32,
        Shape::new(&[
            u64::from(minimax_h3::audio_vae::PCM_CHANNELS),
            channels,
            frames,
        ]),
    )?;
    nn::Noise::new().draw(
        &forward,
        latents.reshape(Shape::new(&[
            u64::from(minimax_h3::audio_vae::PCM_CHANNELS) * channels,
            frames,
        ])),
        nn::NoiseKey::Scalar(minimax_h3::denoiser::POSTERIOR_SEED),
        0,
        0,
        0,
    )?;
    // The window's own shape: `[samples, 2]`, interleaved, which is what `DecodedMedia.audio` is.
    let out = arena.tensor(
        DType::I16,
        Shape::new(&[samples, u64::from(minimax_h3::audio_vae::PCM_CHANNELS)]),
    )?;

    // 6. The decode, with the recorder installed for its duration: the walk is what says which programs
    //    the audio half binds, and it is the first time anything has asked.
    let mut walk = Walk(Vec::new());
    ops::capture::set_sink(Some(&mut walk));
    let decoded = decoder.decode(
        &forward,
        latents,
        &[],
        DecodedMedia {
            video: None,
            audio: Some(out),
        },
    );
    ops::capture::set_sink(None);
    stream.synchronize()?;

    let directory = args.out.join("audio_decoder");
    std::fs::create_dir_all(&directory)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    let walk_path = directory.join("walk.txt");
    std::fs::write(&walk_path, walk.text())
        .map_err(|error| Failure::message(format!("{}: {error}", walk_path.display())))?;
    println!(
        "walk: {} launches of {} distinct programs -> {}",
        walk.0.len(),
        walk.distinct(),
        walk_path.display()
    );
    decoded?;
    if walk.0.is_empty() {
        return Err(Failure::message(
            "the walk recorded no launches: the audio decode replayed a captured graph, or no sink \
             was installed, and either way this stage has not measured the decoder",
        ));
    }

    // 7. The two files a comparison needs: the draw and the PCM. The latent is FP32 because the
    //    decoder computes in it, and the PCM is `i16` because the wire format is what the deployment
    //    ships - a comparison at another type would be a comparison of a second quantizer.
    let latent_bytes = download(
        &stream,
        latents,
        usize::try_from(4 * u64::from(minimax_h3::audio_vae::PCM_CHANNELS) * channels * frames)
            .unwrap_or(0),
    )?;
    write_h3t(
        &directory.join("latents.h3t"),
        &[
            u64::from(minimax_h3::audio_vae::PCM_CHANNELS),
            channels,
            frames,
        ],
        CONTAINER_F32,
        &latent_bytes,
    )?;
    let pcm = download(
        &stream,
        out,
        usize::try_from(2 * u64::from(minimax_h3::audio_vae::PCM_CHANNELS) * samples).unwrap_or(0),
    )?;
    write_h3t(
        &directory.join("pcm.h3t"),
        &[samples, u64::from(minimax_h3::audio_vae::PCM_CHANNELS)],
        CONTAINER_I16,
        &pcm,
    )?;
    let case = format!(
        "{{\"component\": \"audio_decoder\", \"dtype\": \"f32\", \"latent_frames\": \
         {frames}, \"latent_channels\": {channels}, \"samples\": {samples}, \"latents\": \
         \"latents.h3t\", \"pcm\": \"pcm.h3t\", \"checkpoint\": \"{}\"}}\n",
        args.root.display(),
    );
    std::fs::write(directory.join("case.json"), case)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    println!(
        "audio: {frames} latent frames of {channels} channel(s) -> {samples} interleaved i16 sample \
         pairs, written to {}",
        directory.display()
    );
    println!(
        "arena used: {} MiB of {} MiB reserved{}",
        arena.used() >> 20,
        arena_bytes >> 20,
        if arena.used() > arena_bytes {
            " - THE RESERVATION IS SHORT BY THE DIFFERENCE"
        } else {
            ""
        }
    );
    Ok(())
}

/// The encode half of [`stage_raster`]: pixels to a latent, written where `--stage decode` reads one.
///
/// The device is opened, used and closed inside this function, because the decode that follows
/// allocates its own pool: two components resident at once would ask the device for both the video
/// encoder's and the decoder's weights, which no deployment of `ref2va` needs at once.
fn encode_raster(args: &Args) -> Result<(u32, u32, u32, PathBuf)> {
    let store_root = args.store.clone().ok_or_else(|| {
        Failure::message(
            "stage `raster` needs a program store: pass --store <path> or set UNISERVE_STORE",
        )
    })?;
    let config = args.configuration()?;
    let ratio = config.video_vae.spatial_compression();

    // 1. The raster, on the host: one still or one clip, FP32, `[frames, rows, columns, channels]`.
    //    Its pixel extents are the encoder's input, so the latent extents are theirs divided by the
    //    spatial compression rather than a flag that could disagree with them.
    let (pixels_path, held, extents, payload) = match (&args.pixels, args.pattern) {
        (Some(path), _) => {
            let (held, extents, payload) = read_container(path)?;
            (path.clone(), held, extents, payload)
        }
        (None, true) => {
            // The **deployment's** canvas, not the latent flags' default: a drawn raster of extents no
            // request can name is a raster no deployment serves, and this path exists to run the pair at
            // the canvas the rows are keyed for. A flag pair that names an offered canvas is honoured;
            // anything else - including the defaults - draws the first of the six.
            let offered = offered_canvases();
            let first = offered.first().ok_or_else(|| {
                Failure::message("the deployment offers no canvas to draw".to_owned())
            })?;
            let canvas = offered
                .iter()
                .find(|canvas| {
                    canvas.height == args.latent_height * ratio
                        && canvas.width == args.latent_width * ratio
                })
                .unwrap_or(first);
            let frames = config.video_vae.clip_length;
            println!(
                "raster: drawing the {} canvas, {frames} frames of {} x {} pixels",
                canvas.name(),
                canvas.height,
                canvas.width
            );
            let (held, extents, payload) = pattern_pixels(frames, canvas.height, canvas.width)?;
            (PathBuf::from("drawn"), held, extents, payload)
        }
        (None, false) => {
            return Err(Failure::message(
                "stage `raster` encodes a raster: pass --pixels <file>, one still or one clip of \
                 `[frames, rows, columns, channels]` FP32 - or --pattern to draw one and run the path \
                 before a reference clip exists",
            ));
        }
    };
    let (frames, rows, columns) = match extents.as_slice() {
        [frames, rows, columns, channels] => {
            if u64::from(minimax_h3::output::Rgb::CHANNELS) != *channels {
                return Err(Failure::message(format!(
                    "{}: {} channels; the video encoder reads {} (RGB)",
                    pixels_path.display(),
                    channels,
                    minimax_h3::output::Rgb::CHANNELS
                )));
            }
            (*frames, *rows, *columns)
        }
        other => {
            return Err(Failure::message(format!(
                "{}: holds {other:?}; this stage encodes one raster of [frames, rows, columns, \
                 channels] - the layout `--stage decode` writes",
                pixels_path.display()
            )));
        }
    };
    if held != CONTAINER_F32 {
        return Err(Failure::message(format!(
            "{}: element code {held}; the video encoder computes in F32, and this stage reads \
             {CONTAINER_F32}",
            pixels_path.display()
        )));
    }
    let frames = u32::try_from(frames)
        .map_err(|_| Failure::message(format!("{}: too many frames", pixels_path.display())))?;
    let rows = u32::try_from(rows)
        .map_err(|_| Failure::message(format!("{}: too many rows", pixels_path.display())))?;
    let columns = u32::try_from(columns)
        .map_err(|_| Failure::message(format!("{}: too many columns", pixels_path.display())))?;
    // **The canvas is the deployment's, and it is refused by name.** A row keyed for a raster no
    // request can name is a cell no launch reads, so the six the deployment offers are the only
    // extents this stage encodes - the same rule the walks apply to their `--rasters`.
    let offered = offered_canvases();
    if !offered
        .iter()
        .any(|canvas| canvas.height == rows && canvas.width == columns)
    {
        let names: Vec<String> = offered
            .iter()
            .map(|canvas| format!("{} ({}x{})", canvas.name(), canvas.width, canvas.height))
            .collect();
        return Err(Failure::message(format!(
            "{}: {rows}x{columns} is not one of the deployment's offered canvases {}",
            pixels_path.display(),
            names.join(", ")
        )));
    }
    // **The frame count, before a weight is opened.** The encoder reads one still or one clip, and the
    // clip's length is the configuration's: a raster of any other count has no path through the
    // network, so the refusal belongs here rather than inside the first launch - a caller with a
    // 3-frame container would otherwise load 4.5 GiB of weights to be told it cannot be read.
    let clip_frames = config.video_vae.clip_length;
    if frames != 1 && frames != clip_frames {
        return Err(Failure::message(format!(
            "{}: {frames} frames; the video encoder reads one still (1) or one {clip_frames}-frame \
             clip, and a condition longer than a clip is cut by the runner into clips of that length",
            pixels_path.display()
        )));
    }
    // And `--frames` is a decode flag: this stage derives the latent's frame count from the raster and
    // the encoder's own arithmetic, so a stated count would be silently replaced - refused instead.
    if args.frames.is_some() {
        return Err(Failure::message(
            "stage `raster` derives the frame count from the raster and the encoder's clip \
             arithmetic; `--frames` belongs to `--stage decode`, which reads a latent of that count"
                .to_owned(),
        ));
    }
    if !rows.is_multiple_of(ratio) || !columns.is_multiple_of(ratio) {
        return Err(Failure::message(format!(
            "{}: {rows} x {columns} pixels is not a multiple of the encoder's {ratio}-pixel spatial \
             compression, so it has no latent raster",
            pixels_path.display()
        )));
    }
    let latent_height = rows / ratio;
    let latent_width = columns / ratio;
    let directory = args.out.join("video_decoder");
    std::fs::create_dir_all(&directory)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    // The checkpoint is opened **after** the raster is read: a wrong `--pixels` is a cheap failure and
    // an index walk is not, so the input a caller typed is checked before the export is touched.
    let checkpoint = Checkpoint::open(&args.root)?;

    // 2. The device, the kernels, and the encoder's weights through the loader's own boundary.
    let device = platform::Device::new(platform::DeviceId(args.device))?;
    let _guard = device.enter()?;
    let stream = device.stream()?;
    let generation = load_generation(&device, &store_root)?;
    let component = checkpoint.component(VAE)?;
    let granularity =
        u64::try_from(platform::HandleType::Descriptor.granularity(device.id())?).unwrap_or(1);
    let needed = resident_bytes(
        component,
        minimax_h3::encoding::VideoEncoder::keys(&config.video_vae)?,
        granularity,
    )?;
    let mut weights = DeviceWeights::new(&device, needed, granularity)?;
    let encoder = {
        let mut loader = WeightLoader::new(component, &mut weights, None, None);
        minimax_h3::encoding::VideoEncoder::load(&config.video_vae, &mut loader)?
    };
    println!(
        "weights: {} MiB published of {} MiB reserved (the video encoder alone - the decode that \
         follows opens its own pool)",
        weights.bytes >> 20,
        weights.reserved >> 20
    );

    // 3. Two statements of the frame count, each its owner's. The encoder's clip arithmetic decides
    //    the latent frames; nothing here restates it.
    let latent_frames = encoder.latent_frames(frames)?;
    let latent = minimax_h3::encoding::Extent {
        frames: latent_frames,
        height: latent_height,
        width: latent_width,
    };
    println!(
        "raster: {frames} frames of {rows} x {columns} pixels -> {latent_frames} latent frames of \
         {latent_height} x {latent_width}"
    );

    // 4. The tiles the encoder's own grid splits the raster into, and the plan the driver executes.
    let clip = minimax_h3::tiled_encode::ClipEncoder::new(rows, columns)?;
    let plan = clip.plan();
    let tile_extents = plan.tile_extents(frames);
    let (origin, end) = plan.extent();
    let run_rows = u64::from(end - origin);
    println!(
        "raster: {} tile(s), latents {} rows of {run_rows} run rows, {} seamless seam width(s)",
        plan.tiles.len(),
        end - origin,
        clip.blend_extents().len()
    );

    // 5. The arena: what a **run** of tiles reserves, which is one set for the grid and not one per tile
    //    (the driver hoists the reservation out of its loop, `tiled_encode.rs:319`), one pixel buffer per
    //    tile because the scene reads every tile's pixels in one call, the conditioning buffers the stint
    //    produces, and the latent this stage reads back. Summing the tile arena over the grid instead
    //    states the tile count times the run - 65,360 MiB against 6,150 MiB at 17 frames of 768x768.
    // The raster plan is the encoder's set **and the stitcher's ring and hand-off**, which
    // `Scene::encode` takes after `VideoEncoder::reserve` returns: a run sized from the encoder's own
    // set alone stops at the first ring block - 737,280 B past the plan for a still and 1,474,560 B
    // for a clip at 768p 16:9. The grid's tile columns are what the ring is a block wide per column of.
    let ring_columns = {
        let mut starts: Vec<u32> = plan.tiles.iter().map(|tile| tile.columns.0).collect();
        starts.sort_unstable();
        starts.dedup();
        starts.len()
    };
    let mut arena_plan = minimax_h3::encoding::video_raster_arena(
        &config.video_vae,
        &tile_extents,
        ring_columns,
        latent_frames,
    )?;
    arena_plan.extend(minimax_h3::encoding::conditioning_arena(
        &config.video_vae,
        frames,
        latent,
    )?);
    for extent in &tile_extents {
        arena_plan.push(minimax_h3::decoding::Buffer {
            name: "raster pixels".to_owned(),
            element: "f32",
            elements: u64::from(minimax_h3::output::Rgb::CHANNELS)
                * u64::from(extent.frames)
                * u64::from(extent.height)
                * u64::from(extent.width),
        });
    }
    let latent_channels = u64::from(config.video_vae.latent_channels);
    let latent_elements =
        latent_channels * u64::from(latent_frames) * run_rows * u64::from(latent_width);
    arena_plan.push(minimax_h3::decoding::Buffer {
        name: "raster latent".to_owned(),
        element: "f32",
        elements: latent_elements,
    });
    let planned = minimax_h3::decoding::arena_reservation(&arena_plan);
    let allocator = weights.allocator();
    let arena_bytes = planned.max(args.arena_gib << 30);
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
        "arena: {} buffers, {} MiB planned, {} MiB reserved",
        arena_plan.len(),
        planned >> 20,
        arena_bytes >> 20
    );

    // 6. Every tile's pixels, cut from the raster on the host and packed: `FramePad` pads a frame's
    //    rows in place of the source's own, so it reads a *packed* view, and a tile cut from a wider
    //    raster carries the raster's row stride instead of the tile's width.
    let mut views = Vec::with_capacity(plan.tiles.len());
    for (tile, extent) in plan.tiles.iter().zip(&tile_extents) {
        let packed = cut_raster_tile(
            &payload,
            frames,
            rows,
            columns,
            tile.rows.0 * ratio,
            tile.columns.0 * ratio,
            extent,
        )?;
        let shape = Shape::new(&[
            1,
            u64::from(minimax_h3::output::Rgb::CHANNELS),
            u64::from(extent.frames),
            u64::from(extent.height),
            u64::from(extent.width),
        ]);
        views.push(upload(&stream, &arena, DType::F32, shape, &packed)?);
    }

    // 7. The encode, the stitch, the projection and the posterior's sample: the crate's own order.
    let blended = arena.tensor(
        DType::F32,
        Shape::new(&[
            2 * latent_channels,
            u64::from(latent_frames),
            run_rows,
            u64::from(latent_width),
        ]),
    )?;
    // The buffers the encode writes into, allocated before the sink is armed: an allocation is not a
    // launch, and a `?` between arming and disarming would leave the sink installed for every later
    // stage of the process.
    let moments = arena.tensor(DType::F32, blended.shape)?;
    let drawn = arena.tensor(
        DType::F32,
        Shape::new(&[
            latent_channels,
            u64::from(latent_frames),
            run_rows,
            u64::from(latent_width),
        ]),
    )?;
    let flat = drawn.reshape(Shape::new(&[
        latent_channels * u64::from(latent_frames) * run_rows,
        u64::from(latent_width),
    ]));
    let latent_view = arena.tensor(
        DType::F32,
        Shape::new(&[
            latent_channels,
            u64::from(latent_frames),
            run_rows,
            u64::from(latent_width),
        ]),
    )?;
    // **Two components run in this stage, so one `walk.txt` cannot say which bound what.** The decode
    // half writes `walk.txt` from its own sink; the encoder's launches - the encode blocks, the
    // projection, the posterior's sample, the stitcher's blends and copies - go to `walk.encode.txt`.
    // Without this the raster path's encoder was recorded nowhere: `walk.txt` named the decode's
    // programs, and a reader who took it for the encode's found a list that did not match the store.
    let mut encode_walk = Walk(Vec::new());
    ops::capture::set_sink(Some(&mut encode_walk));
    let encoded = clip
        .encode(
            &forward,
            &encoder,
            &plan,
            minimax_h3::tiled_encode::Scene {
                pixels: &views,
                blended,
            },
        )
        .and_then(|()| encoder.project(&forward, blended, moments))
        .and_then(|()| {
            // `and_then` needs one error type for the whole chain, and this is the one step that
            // answers with `nn`'s; `?` would convert it, and a chain cannot.
            nn::Noise::new()
                .draw(
                    &forward,
                    flat,
                    nn::NoiseKey::Scalar(minimax_h3::denoiser::POSTERIOR_SEED),
                    0,
                    0,
                    0,
                )
                .map_err(Into::into)
        })
        .and_then(|()| encoder.posterior(&forward, moments, drawn, latent_view));
    ops::capture::set_sink(None);
    // Written before the outcome is read, as the decode half's is: a stop has no other record of how
    // far it got, and the launch after the last line is the one that stopped.
    let encode_walk_path = args.out.join("walk.encode.txt");
    std::fs::write(&encode_walk_path, encode_walk.text())
        .map_err(|error| Failure::message(format!("{}: {error}", encode_walk_path.display())))?;
    println!(
        "encode walk: {} launches of {} distinct programs -> {}",
        encode_walk.0.len(),
        encode_walk.distinct(),
        encode_walk_path.display()
    );
    encoded?;
    if encode_walk.0.is_empty() {
        return Err(Failure::message(
            "the encoder's walk recorded no launches: the encode replayed a captured graph, or no \
             sink was installed, and either way this stage has not measured the raster path",
        ));
    }
    stream.synchronize()?;

    // 8. The latent, read back and written in the container the decode already reads: a denoiser's
    //    `[channels, frames, rows, columns]` in FP32, which `--stage decode` narrows to its own
    //    element type and says so.
    let elements = latent_elements;
    let bytes = usize::try_from(elements * 4).unwrap_or(usize::MAX);
    let host = download(&stream, latent_view, bytes)?;
    if host.iter().all(|byte| *byte == 0) {
        return Err(Failure::message(
            "the encoded latent is all zeros: the encode wrote nothing, and a decode of it would be \
             a decode of an empty buffer",
        ));
    }
    let path = directory.join("latents.h3t");
    write_h3t(
        &path,
        &[
            latent_channels,
            u64::from(latent_frames),
            u64::from(latent_height),
            u64::from(latent_width),
        ],
        CONTAINER_F32,
        &host,
    )?;
    let case = directory.join("raster_case.json");
    let stated = format!(
        "{{\"raster\": \"{}\", \"frames\": {frames}, \"rows\": {rows}, \"columns\": {columns}, \
         \"latent_frames\": {latent_frames}, \"latent_height\": {latent_height}, \
         \"latent_width\": {latent_width}, \"tiles\": {}, \"checkpoint\": \"{}\"}}\n",
        pixels_path.display(),
        plan.tiles.len(),
        args.root.display(),
    );
    std::fs::write(&case, stated)
        .map_err(|error| Failure::message(format!("{}: {error}", case.display())))?;
    println!(
        "wrote {}: [{latent_channels}, {latent_frames}, {latent_height}, {latent_width}] FP32 from \
         {frames} frames of {} - pass it to `--stage decode --frames {latent_frames}`",
        path.display(),
        pixels_path.display()
    );
    Ok((latent_frames, latent_height, latent_width, path))
}

/// A drawn raster: one clip of the model's own clip length, at the canvas's pixel extents.
///
/// The values are a fixed sawtooth, so two runs of the stage encode the same raster and a difference
/// between their latents is the arithmetic rather than the input. A stage that needs real pixels is a
/// stage that passes `--pixels`.
fn pattern_pixels(frames: u32, rows: u32, columns: u32) -> Result<(u8, Vec<u64>, Vec<u8>)> {
    let channels = u64::from(minimax_h3::output::Rgb::CHANNELS);
    let elements = u64::from(frames) * u64::from(rows) * u64::from(columns) * channels;
    let mut payload = Vec::with_capacity(usize::try_from(elements * 4).unwrap_or(0));
    for index in 0..elements {
        let value = f32::from(u16::try_from(index % 251).unwrap_or(0)) / 251.0;
        payload.extend_from_slice(&value.to_le_bytes());
    }
    Ok((
        CONTAINER_F32,
        vec![
            u64::from(frames),
            u64::from(rows),
            u64::from(columns),
            channels,
        ],
        payload,
    ))
}

/// One tile's pixels, cut from a `[frames, rows, columns, channels]` FP32 raster and packed as the
/// encoder's `[1, channels, frames, tile_rows, tile_columns]`.
///
/// The copy is on the host because the raster arrives as host bytes: a caller uploads request pixels,
/// and the encoder's `FramePad` reads a packed view, so the cut has to happen before the upload
/// rather than as a device copy of a strided block.
fn cut_raster_tile(
    payload: &[u8],
    frames: u32,
    rows: u32,
    columns: u32,
    first_row: u32,
    first_column: u32,
    extent: &minimax_h3::encoding::Extent,
) -> Result<Vec<u8>> {
    let channels = u64::from(minimax_h3::output::Rgb::CHANNELS);
    let mut out = Vec::with_capacity(
        usize::try_from(
            channels
                * u64::from(extent.frames)
                * u64::from(extent.height)
                * u64::from(extent.width)
                * 4,
        )
        .unwrap_or(0),
    );
    // The destination is channel-major (`[1, channels, frames, rows, columns]`) and the source is
    // pixel-major (`[frames, rows, columns, channels]`), so the channel is the outer loop and the
    // pixels are read from the one place the two layouts name the same element.
    for channel in 0..channels {
        for frame in 0..extent.frames {
            for row in 0..extent.height {
                for column in 0..extent.width {
                    let pixel = (u64::from(frame) * u64::from(rows) + u64::from(first_row + row))
                        * u64::from(columns)
                        + u64::from(first_column + column);
                    let at = (pixel * channels + channel) * 4;
                    let at = usize::try_from(at).map_err(|_| {
                        Failure::message("the raster's element offset is past usize".to_owned())
                    })?;
                    let four = payload.get(at..at + 4).ok_or_else(|| {
                        Failure::message(format!(
                            "the raster holds {at} bytes, which is short of [{frames}, {rows}, \
                             {columns}, {channels}] at the tile starting at \
                             ({first_row}, {first_column})"
                        ))
                    })?;
                    out.extend_from_slice(four);
                }
            }
        }
    }
    Ok(out)
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
fn resident_bytes(component: &nn::Checkpoint, keys: Vec<String>, granularity: u64) -> Result<u64> {
    let mut total = 0_u64;
    for key in keys {
        // Every key of the block's own list counts, gain tensors included. Skipping a `.weight_g` on
        // the assumption that the fold publishes one tensor was wrong by exactly the amount that
        // failed: the pool charges a block for every publish it is given, and a reservation that
        // leaves one class of them out is short by that class however exact the rest is. The
        // reservation and the give-out must be one sum over one set.
        let Some(location) = component.tensor(&key) else {
            // Absent names are the loader's to refuse, with the block that asks for them.
            continue;
        };
        let bytes = location.shape.numel() * u64::from(location.dtype.bits()) / 8;
        // The pool hands out whole blocks, and a block is the request rounded up to the driver's
        // granularity (`platform::pool::allocate_block`): 585 tensors each padded to that unit is the
        // difference between the index's sum and what the pool needs, and on this device it is the
        // 128 MiB the caller's stop reported rather than a rounding error. Counting the roundings is
        // what makes this the pool's own number instead of a floor plus headroom.
        total += bytes.max(1).next_multiple_of(granularity.max(1));
    }
    Ok(total)
}

/// Device storage for weights: a resident pool, one buffer per tensor, and every publish waited on.
struct DeviceWeights {
    pools: platform::Allocator,
    buffers: Vec<platform::Buffer>,
    bytes: u64,
    /// Every key this store has published, so a second ask answers the first region instead of a
    /// second block. `HostStore` does this on the host, and a device store that did not would charge
    /// the pool twice for one weight while the reservation counted it once - the class that made a
    /// pool one block short in another lane this round.
    published: Vec<(nn::WeightKey, Backing)>,
    /// The reservation the pool was given, so a failed give-out can name it beside the bytes
    /// published: both numbers then describe publishing, and a reader is not comparing a sum of
    /// weights with a capacity that also holds something else.
    reserved: u64,
}

impl DeviceWeights {
    fn new(device: &platform::Device, requested: u64, granularity: u64) -> Result<Self> {
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
            "resident pool: {} MiB needed (the index, each tensor rounded to the {granularity} B block \
             granularity), {} MiB reserved{}",
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
            published: Vec::new(),
            reserved,
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
    fn retained(&mut self, key: &WeightKey) -> nn::Result<Option<Backing>> {
        Ok(self
            .published
            .iter()
            .find(|(published, _)| published == key)
            .map(|(_, backing)| *backing))
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<Backing> {
        // A key already published answers its own region: a store that allocated again would hand two
        // layers two buffers for one weight and charge the pool for both.
        if let Some(backing) = self.retained(key)? {
            return Ok(backing);
        }
        let buffer = self
            .pools
            .allocate(platform::PoolKind::Resident, bytes.len())
            .map_err(|source| {
                // The pool's own message names a byte count and nothing else. Which tensor, how big,
                // and how far the sum had got is what makes the next short reservation a repair
                // instead of a guess: the index's sum under-counts a tensor the loader synthesizes,
                // and this line says which one when it happens.
                nn::Error::Checkpoint {
                    path: key.to_string().into(),
                    problem: format!(
                        "publishing this weight ({dtype} {shape}, {request} bytes) would reach \
                         {total} bytes published of {reserved} reserved, {mapped} mapped: {source}",
                        request = bytes.len(),
                        total = self.bytes + bytes.len() as u64,
                        reserved = self.reserved,
                        mapped = self.pools.mapped_bytes(platform::PoolKind::Resident),
                    ),
                }
            })?;
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
        self.published.push((key.clone(), backing));
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

/// The `f32` bytes of a payload at `dtype`: the widening `diagnosis::write_npy` requires.
///
/// A decode at the checkpoint's own half precision writes its `.h3t` at that width, and the reference
/// covers its `.npy` twin by widening on the reader's side. This crate writes both files from one
/// payload, so the widening happens here: FP32 is copied, FP16 is widened exactly (every half is a
/// value a `f32` holds), and any other width is refused by name rather than guessed at.
fn widen_to_f32(bytes: &[u8], dtype: DType) -> Result<Vec<u8>> {
    match dtype {
        DType::F32 => Ok(bytes.to_vec()),
        DType::F16 => {
            let mut widened = Vec::with_capacity(bytes.len() * 2);
            for half in bytes.as_chunks::<2>().0 {
                widened.extend_from_slice(
                    &nn::weights::half_to_f32(u16::from_le_bytes(*half)).to_le_bytes(),
                );
            }
            Ok(widened)
        }
        other => Err(Failure::message(format!(
            "a raster at {other} has no `f32` twin this stage writes: it widens FP16 and copies \
             FP32, and another width would have to be guessed at"
        ))),
    }
}

/// The frames one clip keeps: its body, the frames the final clip drops and the tail that overlaps the
/// next clip - the span the decoder trims to, which is fewer than the frames it decodes.
fn kept_frames(config: &minimax_h3::VideoVaeConfig) -> u64 {
    u64::from(config.clip_length + config.token_drop + minimax_h3::decoding::OVERLAP_FRAMES)
}

/// Whether a `--latents` value is a bare non-negative integer - an evaluation index a caller meant as
/// a file name.
fn is_index(path: &Path) -> bool {
    path.to_str()
        .is_some_and(|word| !word.is_empty() && word.bytes().all(|byte| byte.is_ascii_digit()))
}

/// Read a `.h3t` container, refusing a shape or an element type this run cannot use.
/// A container's three parts: its element code, its extents, and its payload.
fn read_container(path: &Path) -> Result<(u8, Vec<u64>, Vec<u8>)> {
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
    Ok((held, extents, bytes[6 + rank * 8..].to_vec()))
}

/// Which of the two containers this stage reads a supplied latent in.
///
/// The two forms differ in **rank and in element type**, and both differences are load-bearing: the
/// stage's own container is `[1, C, frames, H, W]` at the decode's type, and a denoiser's trajectory
/// is `[C, frames, H, W]` because the sample is FP32. A file that matches neither is neither - and it
/// is refused with both forms named, never reinterpreted, because the byte count of the two is the same
/// and a wrong rank read as a length would put a permuted latent through the decoder silently.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LatentForm {
    /// `[1, C, frames, H, W]` at the decode's element type: the stage's own, read as it stands.
    Own,
    /// `[C, frames, H, W]`: a trajectory, narrowed to the decode's type by [`convert_latents`].
    Trajectory,
    /// Neither, and the caller must be told both.
    Neither,
}

fn latent_form(
    held: u8,
    extents: &[u64],
    wanted: &[u64; 5],
    trajectory: &[u64; 4],
    container: u8,
) -> LatentForm {
    if held == container && extents == wanted {
        LatentForm::Own
    } else if extents == trajectory {
        LatentForm::Trajectory
    } else {
        LatentForm::Neither
    }
}

/// The container's element code for FP32, which is what a denoiser's trajectory holds.
const CONTAINER_F32: u8 = 1;
/// Interleaved `i16` PCM: the audio decoder's output, which is the wire format rather than a dtype the
/// decode computes in.
///
/// **Code 6, because the container's numbering is established and 2 is `i64`**
/// (`tools/dump_tables.py`'s `CODES`: 0 f64, 1 f32, 2 i64, 3 i32, 4 bf16, 5 f16). A code chosen by
/// what looked free would have written an `i16` payload the reader took for a table of `i64`.
const CONTAINER_I16: u8 = 6;

/// Re-encode a latent payload from element type `held` to the decode's `dtype`.
///
/// The trajectory a denoiser writes is the sample's own type, which is FP32, while this stage holds the
/// VAE in the checkpoint's half precision: the pipeline therefore has one narrowing between the draw and
/// the decode, and the reference makes the same one (`latents.to(dtype)` before its decoder). It is
/// stated on the run's own line, never silent, and only the two float widths this pipeline uses convert -
/// a file of any other type is refused by name rather than reinterpreted.
fn convert_latents(bytes: &[u8], held: u8, dtype: DType) -> Result<Vec<u8>> {
    let from = match held {
        CONTAINER_F32 => 4_usize,
        other if other == element_of(DType::F16).unwrap_or(u8::MAX) => 2,
        other => {
            return Err(Failure::message(format!(
                "a latent of element code {other} cannot be converted: this stage converts FP32 to \
                 the decode's type and nothing else"
            )));
        }
    };
    let to = usize::try_from(dtype.bytes().unwrap_or(0)).map_err(|_| {
        Failure::message(format!(
            "the decode is {dtype}, a width this stage cannot write"
        ))
    })?;
    if from == to {
        return Ok(bytes.to_vec());
    }
    if !bytes.len().is_multiple_of(from) {
        return Err(Failure::message(format!(
            "{} bytes is not a whole number of {from}-byte elements",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / from * to);
    for element in bytes.chunks(from) {
        let value = f32::from_le_bytes(element.try_into().unwrap_or_default());
        match dtype {
            DType::F16 => out.extend_from_slice(&nn::weights::f32_to_half(value).to_le_bytes()),
            DType::BF16 => out.extend_from_slice(&bf16_bits(value).to_le_bytes()),
            DType::F32 => out.extend_from_slice(&value.to_le_bytes()),
            other => {
                return Err(Failure::message(format!(
                    "the decode is {other}, which this stage has no conversion to"
                )));
            }
        }
    }
    Ok(out)
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
            DType::F16 => bytes.extend_from_slice(&nn::weights::f32_to_half(value).to_le_bytes()),
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

/// Pack `[channels, frames, height, width]` float16 into the rows this decoder reads, in tile order.
///
/// The container is **channel-major**, which is the reference's own latent layout (`[B, C, T, H, W]`,
/// permuted to channels-last only inside its decoder), so one voxel's channels are `frames · height ·
/// width` elements apart in the file and adjacent in the rows. Reading them adjacently in the file
/// instead would hand the decode a transposed latent and still produce a plausible raster: the class
/// this branch keeps finding, and here it would have been silent on the harness that has never run.
fn pack_latents(
    bytes: &[u8],
    tiles: &[minimax_h3::video_vae::Tile],
    canvas: &Canvas,
    dtype: DType,
) -> Result<Vec<u8>> {
    let channels = 24_u64;
    let frames = u64::from(canvas.frames);
    let height = u64::from(canvas.height);
    let width = u64::from(canvas.width);
    // The width is **the decode's**, read from its element type and never written down: a `* 2` here
    // meant "f16" and refused every whole-stored export by a factor of two, which read as a size
    // disagreement rather than as the assumption it was.
    let element = dtype.bytes().unwrap_or(0);
    if element == 0 {
        return Err(Failure::message(format!(
            "the decode is {dtype}, which has no whole-byte element size"
        )));
    }
    let elements = channels * frames * height * width;
    let expected = elements * element;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != expected {
        return Err(Failure::message(format!(
            "the latents hold {} bytes = {} elements at {element} B, and a {frames}x{height}x{width} \
             clip of {channels} channels at {dtype} needs {expected} bytes = {elements} elements - \
             the element type is the decode's, so a disagreement here is a file of another type, \
             not a conversion this stage performs silently",
            bytes.len(),
            u64::try_from(bytes.len()).unwrap_or(0) / element,
        )));
    }
    let mut packed = Vec::with_capacity(bytes.len());
    for tile in tiles {
        gather_tile(
            bytes,
            (frames, height, width, channels),
            (tile.latent_rows.0, tile.latent_rows.1),
            (tile.latent_columns.0, tile.latent_columns.1),
            element,
            &mut packed,
        )?;
    }
    Ok(packed)
}

/// One tile's voxels, gathered out of a channel-major raster into rows of `channels` elements each.
///
/// Rows run frame by frame, then row, then column, and each row's channels are adjacent - the order
/// this decoder's `[voxels, channels]` operand reads. The four geometries and the tile's spans are
/// arguments rather than fields so that a small raster states the arrangement in a test.
fn gather_tile(
    bytes: &[u8],
    geometry: (u64, u64, u64, u64),
    rows: (u32, u32),
    columns: (u32, u32),
    element: u64,
    out: &mut Vec<u8>,
) -> Result<()> {
    let (frames, height, width, channels) = geometry;
    let plane = frames
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or_else(|| Failure::message("the latent raster's extent overflows".to_owned()))?;
    if element == 0 {
        return Err(Failure::message(
            "an element of zero bytes cannot be gathered".to_owned(),
        ));
    }
    for frame in 0..frames {
        for row in u64::from(rows.0)..u64::from(rows.1) {
            for column in u64::from(columns.0)..u64::from(columns.1) {
                let voxel = (frame * height + row) * width + column;
                for channel in 0..channels {
                    let at = (channel * plane + voxel) * element;
                    let start = usize::try_from(at).unwrap_or(usize::MAX);
                    let end = start.saturating_add(usize::try_from(element).unwrap_or(0));
                    let held = bytes.get(start..end).ok_or_else(|| {
                        Failure::message(format!(
                            "the latent raster ends before element {at}: it holds {} bytes and a \
                             {frames}x{height}x{width} raster of {channels} channels does not fit",
                            bytes.len()
                        ))
                    })?;
                    out.extend_from_slice(held);
                }
            }
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::{CONTAINER_F32, Failure, LatentForm, convert_latents, gather_tile, latent_form};
    use nn::DType;

    /// The one narrowing the pipeline makes: a denoiser's FP32 latent, read by a decode whose element
    /// type is the checkpoint's half precision.
    #[test]
    fn a_trajectory_latent_narrows_to_the_decodes_type() {
        let values: [f32; 4] = [1.0, 1.5, -2.0, 0.5];
        let source: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let narrowed = match convert_latents(&source, CONTAINER_F32, DType::F16) {
            Ok(narrowed) => narrowed,
            Err(Failure::Message(message)) => panic!("the narrowing: {message}"),
        };
        // Each expectation is the half-precision encoding of the value, written from the format's own
        // fields: sign, exponent biased by 15, and ten mantissa bits.
        let wanted: [u16; 4] = [0x3c00, 0x3e00, 0xc000, 0x3800];
        assert_eq!(narrowed.len(), 2 * values.len());
        for (element, want) in narrowed.as_chunks::<2>().0.iter().zip(wanted) {
            assert_eq!(u16::from_le_bytes([element[0], element[1]]), want);
        }
    }

    /// A decode already in FP32 takes the trajectory's bytes as they are: no conversion, no rounding.
    #[test]
    fn a_float_latent_read_at_its_own_width_is_copied_unchanged() {
        let source: Vec<u8> = [0.25_f32, -1.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let same = match convert_latents(&source, CONTAINER_F32, DType::F32) {
            Ok(same) => same,
            Err(Failure::Message(message)) => panic!("the copy: {message}"),
        };
        assert_eq!(same, source);
    }

    /// The container is channel-major and the decoder's operand is voxel-major: a tile's rows are each
    /// voxel's channels, gathered across the file's channel planes. Two planes of three voxels make the
    /// arrangement visible, and the wrong one (channels adjacent in the file) is a different sequence.
    #[test]
    fn a_tiles_rows_are_gathered_across_the_channels_planes() {
        // Three channels, one frame, one row of three columns: `[3, 1, 1, 3]`, values named by their
        // position so a permutation cannot read as a pass.
        let half = |value: u16| value.to_le_bytes();
        let mut raster = Vec::new();
        for channel in 0..3_u16 {
            for column in 0..3_u16 {
                raster.extend_from_slice(&half(10 * channel + column));
            }
        }
        let mut out = Vec::new();
        match gather_tile(&raster, (1, 1, 3, 3), (0, 1), (0, 3), 2, &mut out) {
            Ok(()) => {}
            Err(Failure::Message(message)) => panic!("the gather: {message}"),
        }
        let got: Vec<u16> = out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|element| u16::from_le_bytes(*element))
            .collect();
        // Voxel 0 takes one element from each plane, then voxel 1, then voxel 2.
        assert_eq!(got, vec![0, 10, 20, 1, 11, 21, 2, 12, 22]);
    }

    /// The gather's element width is the raster's, not a constant: the same geometry at four bytes
    /// per element reads four-byte elements, which is what a whole-stored export writes. A `* 2` that
    /// meant "f16" here refused every f32 latent by a factor of two and read as a size disagreement.
    #[test]
    fn a_gather_reads_its_elements_at_the_rasters_own_width() {
        let word = |value: f32| value.to_le_bytes();
        let mut raster = Vec::new();
        for channel in 0..2_u32 {
            for column in 0..2_u32 {
                raster.extend_from_slice(&word((10 * channel + column) as f32));
            }
        }
        let mut out = Vec::new();
        match gather_tile(&raster, (1, 1, 2, 2), (0, 1), (0, 2), 4, &mut out) {
            Ok(()) => {}
            Err(Failure::Message(message)) => panic!("the gather: {message}"),
        }
        let got: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|element| f32::from_le_bytes(*element))
            .collect();
        // Voxel 0 takes one element from each plane, then voxel 1 - at four bytes each.
        assert_eq!(got, vec![0.0, 10.0, 1.0, 11.0]);
    }

    /// A raster shorter than the tile's spans is refused with the element it ran out at, not read past
    /// the end into another tile's bytes.
    #[test]
    fn a_raster_that_ends_early_is_refused() {
        let mut out = Vec::new();
        assert!(gather_tile(&[0_u8; 4], (1, 1, 3, 3), (0, 1), (0, 3), 2, &mut out).is_err());
    }

    /// The two containers are told apart by rank and by element type, and neither is guessed from a
    /// length: the same byte count reads as a permuted latent if the rank is ignored.
    #[test]
    fn the_two_container_forms_are_told_apart_and_a_third_is_neither() {
        let wanted = [1, 24, 7, 16, 16];
        let trajectory = [24, 7, 16, 16];
        let own = latent_form(5, &wanted, &wanted, &trajectory, 5);
        assert_eq!(own, LatentForm::Own);
        // A trajectory at FP32: the rank is one shorter and the code is the sample's.
        let sampled = latent_form(1, &trajectory, &wanted, &trajectory, 5);
        assert_eq!(sampled, LatentForm::Trajectory);
        // The stage's own shape at the wrong element type is neither: a decode reading bytes labelled
        // one width from a buffer of another is a page of nonsense, not a conversion.
        let mismatched = latent_form(1, &wanted, &wanted, &trajectory, 5);
        assert_eq!(mismatched, LatentForm::Neither);
        // And a trajectory of the right rank but the wrong clip length is neither too.
        let short = latent_form(1, &[24, 7, 16, 8], &wanted, &trajectory, 5);
        assert_eq!(short, LatentForm::Neither);
    }

    /// An element type this stage has no conversion to is refused by name rather than reinterpreted as
    /// bytes of another width.
    #[test]
    fn an_integer_latent_is_refused_by_its_element_code() {
        let refused = convert_latents(&[0_u8; 8], 3, DType::F16);
        assert!(
            refused.is_err(),
            "an i32 container is not a latent this stage reads"
        );
    }
}
