//! MiniMax-H3's conditioned path: `fl2va`, one image, and the condition rows it writes.
//!
//! The fifth device stage, and the only one that binds `VideoEncoder`: a `t2va` request's prefix is
//! `text_rows + 1` rows and never reaches this code, while a conditioned request's is
//! `text_rows + condition_rows + 1` and those condition rows come from the encoder's tiles, the
//! posterior, the anchoring draw and the projection.
//!
//! ```text
//! --stage weights  no device: what the `fl2va` deployment places, the two encoder key lists against
//!                  the export's indexes, and where the case's rows come from
//! --stage encode   one image on a device: the encoder's tile walk, the posterior, the anchoring and
//!                  the projection, then the rows read back through the packing's own placement -
//!                  `<out>/latent_encoder/condition_rows.h3t` with `case.json`, `walk.txt` and
//!                  `readback.txt`
//! ```
//!
//! **It runs the served entry point, not a method of convenience.** `Denoiser::prepare_conditioning`
//! is what a runner calls once per request, and it is the arm that carried the defect that every
//! condition was written where nothing gathers it (`39a41fd55`); `encode_conditions` was the arm that
//! was right. A stage that exercised only the second would have printed a healthy page.
//!
//! **The verdict is a readback, not a row count.** For each piece the page prints the prefix rows the
//! packing gathers for it - `layout.text_rows` plus the rows of the pieces before it - and the mean
//! magnitude of what it found there. A dropped condition reads as zeros; a row count cannot say so.
//! The verdict is only as strong as the model's claim: **a black keyframe is a legal request with
//! zero rows**, so the failure this stage names is *every* piece reading as zeros, not one. The zero
//! row is a separate claim and a hard one: no write may reach it.
//!
//! Untested halves, stated rather than implied: the text region of the prefix is a **synthetic,
//! stated** pattern (stage 4 is the text encoder), and the case carries **no audio** - this export
//! holds no `audio_vae/` weights, so a case that carried audio would be refused by the model for want
//! of samples, and a page that claimed to have proved the audio rows would prove nothing.
//!
//! `--stage encode` needs a program store (`--store <path>` or `UNISERVE_STORE`) holding committed
//! modules for the operators the encoder launches; the store is read, never written.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::checkpoint::Dit;
use minimax_h3::config::Config;
use minimax_h3::denoiser::{Denoiser, Layouts};
use minimax_h3::encoding::Extent;
use minimax_h3::partition::{Component, Partition};
use nn::capability::{Conditioning, Denoiser as _, MediaInput, MediaRole};
use nn::{
    Arena, Backing, Checkpoint, ConditionRole, ConditionShape, DType, DevicePtr, Forward,
    RetainedWeights, Shape, TensorView, VideoRequest, VideoTask, WeightKey, WeightLoader,
};

/// The export root the caller's GPU host holds, when `--root` does not name another.
const DEFAULT_ROOT: &str = "/workspace/models/MiniMax-H3";

/// The two checkpoint directories a conditioned deployment reads its media from.
const VAE: &str = "vae";
const AUDIO_VAE: &str = "audio_vae";

/// The `.h3t` element-type codes this harness writes, from `tools/dump_tables.py`'s table.
const H3T_F32: u8 = 1;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

/// One parsed command line.
struct Args {
    root: PathBuf,
    out: PathBuf,
    stage: String,
    task: VideoTask,
    /// The generated canvas, in pixels.
    width: u32,
    height: u32,
    /// Generated frames the request asks for.
    frames: u32,
    /// Prompt tokens the case presents.
    text_tokens: u32,
    /// The condition image's raster, in pixels; an anchor is fitted to the canvas.
    image: u32,
    /// The request's seed, which addresses the condition's draw.
    seed: u64,
    device: u32,
    /// The element type the deployment loads at. `auto` is the design's default: the checkpoint's own
    /// representation, nothing converted at load, and each request's pin decides.
    dtype: Option<DType>,
    /// The program store `--stage encode` reads its modules from.
    store: Option<PathBuf>,
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    match args.stage.as_str() {
        "weights" => weights(&args),
        "plan" => plan(&args),
        "encode" => encode(&args),
        other => Err(format!(
            "`{other}` is not a stage of this harness; it has `weights` (host), `plan` (host) and \
             `encode` (device)"
        )),
    }
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Args {
            root: PathBuf::from(DEFAULT_ROOT),
            out: PathBuf::from("artifacts/h3-stage5"),
            stage: "weights".to_owned(),
            task: VideoTask::FirstLastToVideoAudio,
            width: 256,
            height: 256,
            frames: 5,
            text_tokens: 15,
            image: 256,
            seed: 0,
            device: 0,
            dtype: None,
            store: std::env::var_os("UNISERVE_STORE").map(PathBuf::from),
        };
        let mut words = std::env::args().skip(1);
        while let Some(word) = words.next() {
            let mut value = || {
                words
                    .next()
                    .ok_or_else(|| format!("`{word}` needs a value"))
            };
            match word.as_str() {
                "--root" => args.root = PathBuf::from(value()?),
                "--out" => args.out = PathBuf::from(value()?),
                "--stage" => args.stage = value()?,
                "--store" => args.store = Some(PathBuf::from(value()?)),
                "--task" => {
                    args.task = match value()?.as_str() {
                        "fl2va" => VideoTask::FirstLastToVideoAudio,
                        "ref2va" => VideoTask::ReferenceToVideoAudio,
                        "t2va" => VideoTask::TextToVideoAudio,
                        other => {
                            return Err(format!(
                                "`{other}` is not a MiniMax-H3 task; this stage serves fl2va, \
                                 ref2va or t2va"
                            ));
                        }
                    }
                }
                "--canvas" => {
                    let text = value()?;
                    let (width, height) = text
                        .split_once('x')
                        .ok_or_else(|| format!("`{text}` is not a canvas; write 256x256"))?;
                    args.width = width
                        .parse()
                        .map_err(|_| format!("`{width}` is not a width; write 256x256"))?;
                    args.height = height
                        .parse()
                        .map_err(|_| format!("`{height}` is not a height; write 256x256"))?;
                    if args.width == 0 || args.height == 0 {
                        return Err(format!("`{text}` is a canvas of no pixels; write 256x256"));
                    }
                }
                "--frames" => {
                    args.frames = value()?
                        .parse()
                        .map_err(|_| "`--frames` takes a count".to_owned())?;
                }
                "--text-tokens" => {
                    args.text_tokens = value()?
                        .parse()
                        .map_err(|_| "`--text-tokens` takes a count".to_owned())?;
                    if args.text_tokens == 0 {
                        return Err("`--text-tokens` takes at least one token".to_owned());
                    }
                }
                "--image" => {
                    args.image = value()?
                        .parse()
                        .map_err(|_| "`--image` takes a pixel side, such as 256".to_owned())?;
                    // A zero-pixel image is a condition of nothing: it would plan no tile and the
                    // case would carry an empty condition rather than refuse it.
                    if args.image == 0 {
                        return Err("`--image` takes at least one pixel, such as 256".to_owned());
                    }
                }
                "--seed" => {
                    args.seed = value()?
                        .parse()
                        .map_err(|_| "`--seed` takes a number".to_owned())?;
                }
                "--device" => {
                    args.device = value()?
                        .parse()
                        .map_err(|_| "`--device` takes an index".to_owned())?;
                }
                "--dtype" => {
                    args.dtype = match value()?.as_str() {
                        "auto" => None,
                        "bf16" => Some(DType::BF16),
                        "f16" => Some(DType::F16),
                        "f32" => Some(DType::F32),
                        other => {
                            return Err(format!(
                                "`{other}` is not an element type; write auto, bf16, f16 or f32"
                            ));
                        }
                    };
                }
                other => return Err(format!("`{other}` is not a flag of this harness")),
            }
        }
        Ok(args)
    }
}

/// The task's name, as the deployment file and the store list spell it.
fn task_name(task: VideoTask) -> &'static str {
    match task {
        VideoTask::TextToVideoAudio => "t2va",
        VideoTask::FirstLastToVideoAudio => "fl2va",
        VideoTask::ReferenceToVideoAudio => "ref2va",
    }
}

/// The blocks this stage's process places: the deployment's own, **without the text encoder**.
///
/// The text encoder is the pipeline stage before this one - its weights are 47.97 GiB beside this
/// stage's 41.46 - and a process that placed both would need 89 GiB, past an 80 GB device and past
/// what a stage is for. Its output is the runner's prefix buffer, which is what a deployment passes
/// between the two stages and what this harness states as a synthetic pattern (stage 4 owns the text
/// encoder). Placing it here would reserve it and read nothing.
fn stage_partition(config: &Config, task: VideoTask) -> Result<Partition, String> {
    let serving = Partition::serving(config, task, 1, 1, 0, 0).map_err(show)?;
    Ok(serving.placing(serving.blocks.without(Component::TextEncoder)))
}

/// Every tensor name the checkpoint's index lists for one component directory.
fn index_names(directory: &Path) -> Result<std::collections::BTreeSet<String>, String> {
    let mut names = std::collections::BTreeSet::new();
    let entries = std::fs::read_dir(directory)
        .map_err(|error| format!("{} is not a directory: {error}", directory.display()))?;
    let mut found = false;
    for entry in entries.flatten() {
        let file = entry.file_name().to_string_lossy().to_string();
        if !file.ends_with(".index.json") {
            continue;
        }
        found = true;
        let text = std::fs::read_to_string(entry.path())
            .map_err(|error| format!("{} is unreadable: {error}", entry.path().display()))?;
        let index: serde_json::Value =
            serde_json::from_str(&text).map_err(|error| format!("{file}: {error}"))?;
        if let Some(map) = index["weight_map"].as_object() {
            names.extend(map.keys().cloned());
        }
    }
    if !found {
        return Err(format!("{directory:?} holds no .safetensors.index.json"));
    }
    Ok(names)
}

/// Print one key list against an index, and whether it resolves.
fn report_names(label: &str, keys: &[String], index: &std::collections::BTreeSet<String>) -> bool {
    let missing: Vec<&str> = keys
        .iter()
        .map(String::as_str)
        .filter(|key| !index.contains(*key))
        .collect();
    println!(
        "{label}: {} asked, {} missing, index {} tensors",
        keys.len(),
        missing.len(),
        index.len()
    );
    for key in missing.iter().take(8) {
        println!("  missing {key}");
    }
    missing.is_empty()
}

/// The host stage: what the deployment places, what the encoders read, and where the rows come from.
fn weights(args: &Args) -> Result<(), String> {
    let layout = minimax_h3::checkpoint::Layout::detect(&args.root).map_err(show)?;
    let sidecars =
        minimax_h3::checkpoint::Sidecars::read(&args.root, None, &layout).map_err(show)?;
    let config = Config::normalize(&layout, &sidecars).map_err(show)?;

    let partition = stage_partition(&config, args.task)?;
    println!(
        "{} places {} blocks (the text encoder is the stage before this one): {}",
        task_name(args.task),
        partition.blocks.iter().count(),
        partition
            .blocks
            .iter()
            .map(|component| component.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !partition.blocks.contains(Component::LatentEncoder) {
        return Err(format!(
            "{} places no condition encoder, so this harness's case has no rows to write; a \
             conditioned task places `latent_encoder`",
            task_name(args.task)
        ));
    }

    // The two encoder lists, against the indexes their components declare. This is the same `keys`
    // the load answers before it reads a byte, so a name that resolves here resolves at load.
    let video = minimax_h3::encoding::VideoEncoder::keys(&config.video_vae).map_err(show)?;
    let vae_index = index_names(&args.root.join(VAE))?;
    let mut resolved = report_names("video encoder", &video, &vae_index);
    match index_names(&args.root.join(AUDIO_VAE)) {
        Ok(audio_index) => {
            let audio =
                minimax_h3::encoding::AudioEncoder::keys(&config.audio_vae).map_err(show)?;
            // A checkpoint without the audio weights is a fact about the export, not a failure of
            // this stage: the case below carries no audio, and the row says so.
            if audio_index.is_empty() {
                println!(
                    "audio encoder: {} asked, no index to answer against",
                    audio.len()
                );
            } else {
                resolved &= report_names("audio encoder", &audio, &audio_index);
            }
        }
        Err(error) => println!("audio encoder: {error}"),
    }

    if !resolved {
        return Err("a name the conditioned path reads is not in the checkpoint".to_owned());
    }
    println!(
        "case: the rows a condition contributes are the model's answer - the segments it packs and \
         the placement the packing gathers them from - so they need the loaded block and are printed \
         by `--stage encode`. The case is {}x{} with a {} px image and {} prompt tokens.",
        args.width, args.height, args.image, args.text_tokens
    );
    Ok(())
}

/// The largest tile the encoder will read for a raster, which is what its arena must hold.
///
/// The encoder reads **one tile per call** (`VideoEncoder::reserve`'s plan), so planning the whole
/// raster would reserve the ladder fifteen times over at a served canvas. The grid's tiles are
/// uniform, so the largest is the one the arena needs - and the frames are the condition's own, a
/// still being one.
fn largest_tile(pixels: Extent) -> Result<Extent, String> {
    let encoder =
        minimax_h3::tiled_encode::ClipEncoder::new(pixels.height, pixels.width).map_err(show)?;
    // `Plan::tile_extents` is the driver's own statement of what `VideoEncoder::reserve` is handed,
    // ratio and seam rule included, so the arena cannot disagree with the reservation.
    encoder
        .plan()
        .tile_extents(pixels.frames)
        .into_iter()
        .max_by_key(|extent| u64::from(extent.height) * u64::from(extent.width))
        .ok_or_else(|| "the tile plan of this raster holds no tile".to_owned())
}

/// The host stage that sizes the case: every plan the device stage will reserve, and its total.
///
/// A reservation is the number this prints - not a flag - and a harness whose fixed pool is smaller
/// than its plan is a stop the caller reads as a defect in the model. So the plan is printed before
/// anything runs, and the device stage reserves exactly this.
fn plan(args: &Args) -> Result<(), String> {
    let layout = minimax_h3::checkpoint::Layout::detect(&args.root).map_err(show)?;
    let sidecars =
        minimax_h3::checkpoint::Sidecars::read(&args.root, None, &layout).map_err(show)?;
    let config = Config::normalize(&layout, &sidecars).map_err(show)?;
    let partition = stage_partition(&config, args.task)?;
    let dit = config
        .denoisers
        .get(&Dit::Base)
        .ok_or_else(|| "the checkpoint states no base DiT".to_owned())?
        .clone();
    let arrangement = partition.arrangement(&dit).map_err(show)?;
    let layouts = Layouts::new(dit.clone(), arrangement).map_err(show)?;

    // The deployment's type for the terms the refiner reads and writes; the condition views are FP32
    // and are counted at 4 bytes below whatever this is.
    let dtype = args.dtype.unwrap_or(DType::BF16);
    let condition = ConditionShape {
        role: ConditionRole::FirstFrame,
        frames: 1,
        height: args.image,
        width: args.image,
        audio_samples: 0,
    };
    let request = VideoRequest {
        task: args.task,
        width: args.width,
        height: args.height,
        frames: args.frames,
        fps: 24,
        first_frames: 1,
        last_frames: 0,
        references: 0,
        text_tokens: args.text_tokens,
        conditions: std::slice::from_ref(&condition),
        vision_spans: &[],
    };
    let (one, packing) = layouts.layout(&request).map_err(show)?;
    let canvas = layouts.canvas(&request).map_err(show)?;
    let raster = minimax_h3::packing::latent_raster(&canvas).map_err(show)?;
    let latent = Extent {
        frames: 1,
        height: raster.height,
        width: raster.width,
    };
    let pixels = Extent {
        frames: 1,
        height: canvas.height,
        width: canvas.width,
    };
    let conditioning =
        minimax_h3::encoding::conditioning_arena(&config.video_vae, 1, latent).map_err(show)?;
    let tile = largest_tile(pixels)?;
    let tiles =
        minimax_h3::encoding::video_encoder_tile_arena(&config.video_vae, tile).map_err(show)?;
    println!(
        "case: {} {}x{} (canvas {}x{}, latent {}x{}), {} frames, {} px image, {} prompt tokens",
        task_name(args.task),
        args.width,
        args.height,
        canvas.width,
        canvas.height,
        raster.width,
        raster.height,
        args.frames,
        args.image,
        args.text_tokens
    );
    println!(
        "rows: text capacity {}, condition capacity {}, prefix {} rows of {}",
        one.text_rows,
        one.condition_rows,
        packing.zero_row() + 1,
        dit.transformer.hidden_size
    );
    let mut total = 0_u64;
    println!(
        "largest tile: {}x{} px, {} frames",
        tile.height, tile.width, tile.frames
    );
    for (label, buffers) in [("conditioning", &conditioning), ("encoder tile", &tiles)] {
        let bytes: u64 = buffers.iter().map(|buffer| buffer.bytes()).sum();
        total += bytes;
        println!(
            "plan {label}: {} buffers, {bytes} B = {} MiB",
            buffers.len(),
            bytes >> 20
        );
        for buffer in buffers.iter() {
            println!("  {:<24} {:>12} elements", buffer.name, buffer.elements);
        }
    }
    // Every term states the element type it is counted at, because the two kinds differ here: the
    // **source** the refiner reads and writes is the deployment's own (`dtype`, BF16 on this export
    // and F16 on the preset), while **every condition view is FP32** - the text encoder's features,
    // the piece's pixels and its anchoring draw, which `precision.py` pins FP32 at every preset. One
    // term counted at 4 bytes for both kinds is a plan whose reader cannot tell which it overstated.
    let width = dtype.bytes().unwrap_or(4);
    // One line, two kinds: the terms the refiner reads and writes are the deployment's type, every
    // condition view is FP32. Stating the two kinds once beats repeating the type on each term, and
    // it is the line a reader needs when a figure looks half of what a block allocates.
    let typed = match args.dtype {
        Some(stated) => {
            format!("the source at {stated} (stated by --dtype), the condition views at f32")
        }
        // `--stage plan` opens no checkpoint, so on a host with indexes and no shards `auto` cannot
        // ask the export; it states the design's default rather than pass a guess off as the export's.
        None => format!(
            "the source at {dtype} (--dtype auto: this stage loads nothing, so it is the design's \
             default), the condition views at f32"
        ),
    };
    let prefix_mib =
        ((packing.zero_row() as u64 + 1) * u64::from(dit.transformer.hidden_size) * width) >> 20;
    let source_mib =
        (u64::from(one.text_rows) * u64::from(dit.transformer.hidden_size) * width) >> 20;
    let text_mib =
        (u64::from(one.text_rows) * u64::from(config.text_encoder.hidden_size) * 4) >> 20;
    let pixel_mib = (u64::from(pixels.frames)
        * u64::from(pixels.height)
        * u64::from(pixels.width)
        * u64::from(config.video_vae.in_channels)
        * 4)
        >> 20;
    let draw_mib = (u64::from(latent.frames)
        * u64::from(latent.height)
        * u64::from(latent.width)
        * u64::from(dit.transformer.video_channels)
        * 4)
        >> 20;
    println!("plan types: {typed}");
    println!(
        "plan buffers: prefix {prefix_mib} MiB, source {source_mib} MiB, text features {text_mib} \
         MiB, pixels {pixel_mib} MiB, draw {draw_mib} MiB"
    );
    println!(
        "plan arena total (the two plans alone): {} MiB",
        total >> 20
    );
    Ok(())
}

/// The device stage: one image through the encoder, the posterior and the projection.
fn encode(args: &Args) -> Result<(), String> {
    let store_root = args.store.clone().ok_or_else(|| {
        "stage `encode` needs a program store: pass --store <path> or set UNISERVE_STORE".to_owned()
    })?;
    if !args
        .image
        .is_multiple_of(minimax_h3::video_vae::TILE_PIXELS)
    {
        return Err(format!(
            "the condition image is {} px and the encoder's tile is {} px, so a raster that is not a \
             whole number of tiles is not this stage's case",
            args.image,
            minimax_h3::video_vae::TILE_PIXELS
        ));
    }
    if args.image != args.width.min(args.height) && args.image != args.width {
        return Err(format!(
            "a first-frame anchor is fitted to the generated canvas, and {} px is neither the \
             canvas's {} px nor its short edge {}: the packing would refuse it",
            args.image,
            args.width,
            args.width.min(args.height)
        ));
    }

    // 1. The device and the kernels first: a driver that cannot give a device, or a store missing an
    //    operator, fails before a byte of the checkpoint is read.
    let device = platform::Device::new(platform::DeviceId(args.device as i32))
        .map_err(|error| format!("device {}: {error}", args.device))?;
    let _guard = device
        .enter()
        .map_err(|error| format!("entering device {}: {error}", args.device))?;
    let stream = device.stream().map_err(show)?;
    let generation = load_generation(&device, &store_root)?;

    // 2. The deployment's own loading boundary, not a harness's. `load` refuses a name it cannot
    //    build, attaches the condition encoders to the denoiser block, and converts nothing.
    let checkpoint = Checkpoint::open(&args.root).map_err(show)?;
    let config = Config::read(&args.root, None).map_err(show)?;
    let partition = stage_partition(&config, args.task)?;
    let dit = config
        .denoisers
        .get(&Dit::Base)
        .ok_or_else(|| "the checkpoint states no base DiT".to_owned())?
        .clone();
    let arrangement = partition.arrangement(&dit).map_err(show)?;
    let layouts = Layouts::new(dit.clone(), arrangement).map_err(show)?;

    // 3. The case: one first-frame anchor at the canvas, and a request the deployment lays out
    //    with its own rule.
    let condition = ConditionShape {
        role: ConditionRole::FirstFrame,
        frames: 1,
        height: args.image,
        width: args.image,
        audio_samples: 0,
    };
    let request = VideoRequest {
        task: args.task,
        width: args.width,
        height: args.height,
        frames: args.frames,
        fps: 24,
        first_frames: 1,
        last_frames: 0,
        references: 0,
        text_tokens: args.text_tokens,
        conditions: std::slice::from_ref(&condition),
        vision_spans: &[],
    };
    let (layout, packing) = layouts.layout(&request).map_err(show)?;

    // The resident pool is sized by what this partition will publish - the checkpoint's own bytes
    // for the blocks it places - not by a default: a pool that cannot give out one more buffer is a
    // stop the caller reads as a defect in the model (`platform: the pool's reservation cannot give
    // out ... more`), and it is the harness's number that was wrong.
    // `auto`, as the design states it: the checkpoint's own representation, which is the mixed one -
    // the timestep pair FP32, the AdaLN projections and the stack BF16 - so every weight is
    // published as stored and each request's pin decides. Nothing converts at load, so asking the
    // component for a single type here would refuse the export the model is designed to load.
    let deployment: Option<DType> = args.dtype;
    let mut published = 0_u64;
    let mut tensors = 0_u64;
    for component in partition.blocks.iter() {
        for directory in component.directories() {
            let (bytes, count) = stored_bytes(&checkpoint, directory)?;
            published += bytes;
            tensors += count;
        }
    }
    // The pool this partition's load gives out of, in three terms rather than a margin:
    //
    // * every byte the directories it places hold, which is an **upper bound** - a fold replaces
    //   tensors with the products it composes, so the load retains less than the index states;
    // * one rounded block per published region, at the device's allocation granularity: the pool
    //   charges a block per publish, so a stage of 600 small tensors pays 600 blocks;
    // * one tensor of the deployment's type when a stated type converts - the largest single one,
    //   since the conversion reads a weight and publishes it before the next is read. At `auto`
    //   nothing converts and this term is zero.
    //
    // A pool that cannot give out one more buffer stops the run as a model defect, so the number has
    // to be derived rather than padded: "half again" was 20 GiB of padding here.
    const BLOCK: u64 = 2 << 20;
    let conversion = match deployment {
        Some(_) => {
            let mut largest = 0_u64;
            for component in partition.blocks.iter() {
                for directory in component.directories() {
                    let source = checkpoint.component(directory).map_err(show)?;
                    for name in source.names() {
                        if let Some(location) = source.tensor(name) {
                            largest =
                                largest.max(location.bytes.1.saturating_sub(location.bytes.0));
                        }
                    }
                }
            }
            largest
        }
        None => 0,
    };
    let resident = (published + tensors * BLOCK + conversion).max(1 << 30);
    println!(
        "weights: {} MiB stored in {} tensors for this partition, plus {} MiB of blocks and {} MiB \
         of conversion: {} MiB reserved",
        published >> 20,
        tensors,
        (tensors * BLOCK) >> 20,
        conversion >> 20,
        resident >> 20
    );
    let mut weights = DeviceWeights::new(&device, resident)?;
    let model = {
        // The deployment's statement reaches the loader and the boundary **together**, so the two
        // can never disagree: `None` is `auto`, the checkpoint's own representation with each
        // request's pin deciding; `Some(stated)` converts every weight once, keyed by the conversion.
        let mut loader = WeightLoader::new(&checkpoint, &mut weights, deployment, None);
        minimax_h3::load(&checkpoint, partition, deployment, &mut loader).map_err(show)?
    };
    let denoiser = std::sync::Arc::clone(
        model
            .transformer()
            .ok_or_else(|| "the deployment placed no denoiser block".to_owned())?,
    );
    // The type this harness's own buffers take is the stack's, which is what the block's first
    // weight was published at: a prefix source, a condition row and a latent are all written in the
    // type the block reads them in, and at `auto` that is the checkpoint's, not a component's.
    let dtype = denoiser_dtype(&denoiser);
    let hidden = u64::from(dit.transformer.hidden_size);
    let source_width = u64::from(config.text_encoder.hidden_size);
    // The prefix source's three regions, from the packing's own indices: the text capacity, then the
    // condition rows, then one zero row. `zero_row` is the model's statement of where the last
    // begins, so nothing here re-derives it.
    let rows = u64::from(packing.zero_row()) + 1;
    println!(
        "loaded: {dtype} is the stack's element type (auto, the checkpoint's own), hidden {hidden}, \
         text {} rows of capacity, \
         conditions {} rows of capacity, prefix {rows} rows",
        layout.text_rows, layout.condition_rows
    );

    // 4. The pieces, from the model: `rows` per piece is the packing's own number, and a harness
    //    that derived it would be a second statement of the arithmetic `39a41fd55` was about.
    let segments = denoiser.segments(&request).map_err(show)?;
    let pieces: Vec<(u32, u32)> = segments
        .iter()
        .map(|segment| (segment.video_rows(), segment.audio_rows()))
        .collect();
    let mut piece_rows = 0_u32;
    let mut starts = Vec::new();
    for segment in &segments {
        starts.push(piece_rows);
        piece_rows += segment.video_rows() + segment.audio_rows();
        println!(
            "piece {} of condition {}: {:?}, {} visual rows, {} audio rows",
            starts.len() - 1,
            segment.condition,
            segment.kind,
            segment.video_rows(),
            segment.audio_rows()
        );
    }
    if pieces.len() != 1 {
        return Err(format!(
            "this stage's case is one image, and the model packed {} pieces for it",
            pieces.len()
        ));
    }
    if piece_rows as u64 > u64::from(layout.condition_rows) {
        return Err(format!(
            "the model packed {piece_rows} condition rows and the layout reserves {}",
            layout.condition_rows
        ));
    }
    let visual = segments[0]
        .visual
        .ok_or_else(|| "the one piece is not visual, so this case has no pixels".to_owned())?;

    // 5. The buffers the runner owns: the prefix source, the text encoder's features, the piece's
    //    pixels and its draw. The draws are FP32 like every condition view; the source is the
    //    deployment's type, since the refiner reads and writes it.
    let constant_specs = denoiser.constant_tables();
    let state_specs = denoiser.state_tables(&layout);
    let latent = Extent {
        frames: visual.frames,
        height: visual.raster.height,
        width: visual.raster.width,
    };
    // The pixels' raster is the model's answer, not this harness's flags: a canvas the request's own
    // rounding produced and one read back from the flags agree only where that rounding is the
    // identity, which is most canvases and not all.
    let canvas = layouts.canvas(&request).map_err(show)?;
    let pixels = Extent {
        frames: u32::from(visual.frames > 0),
        height: canvas.height,
        width: canvas.width,
    };
    let conditioning_bytes: u64 =
        minimax_h3::encoding::conditioning_arena(&config.video_vae, pixels.frames, latent)
            .map_err(show)?
            .iter()
            .map(|buffer| buffer.bytes())
            .sum();
    let tile = largest_tile(pixels)?;
    let encoder_bytes: u64 =
        minimax_h3::encoding::video_encoder_tile_arena(&config.video_vae, tile)
            .map_err(show)?
            .iter()
            .map(|buffer| buffer.bytes())
            .sum();
    let text_elements = u64::from(layout.text_rows) * hidden;
    let source_elements = u64::from(layout.text_rows) * source_width;
    let prefix_elements = rows * hidden;
    let pixel_elements = u64::from(config.video_vae.in_channels)
        * u64::from(pixels.frames)
        * u64::from(pixels.height)
        * u64::from(pixels.width);
    let draw_elements = u64::from(dit.transformer.video_channels)
        * u64::from(latent.frames)
        * u64::from(latent.height)
        * u64::from(latent.width);
    let tables = table_bytes(&constant_specs) + table_bytes(&state_specs);
    // Every element this harness allocates for itself is four bytes except the two text buffers,
    // which are the deployment's type: the draws and the pixels are FP32 by the contract.
    let width = dtype.bytes().unwrap_or(4);
    let needed = conditioning_bytes
        + encoder_bytes
        + tables
        + (text_elements + source_elements + prefix_elements) * width
        + (pixel_elements + draw_elements) * 4;
    let arena_bytes = needed;
    let allocator = weights.allocator();
    let reserve = usize::try_from(arena_bytes).unwrap_or(usize::MAX);
    allocator
        .reserve(platform::PoolKind::Graph, reserve)
        .map_err(|error| format!("reserving {arena_bytes} B of workspace: {error}"))?;
    let region = allocator
        .allocate(platform::PoolKind::Graph, reserve)
        .map_err(|error| format!("allocating {arena_bytes} B of workspace: {error}"))?;
    let arena = Arena::new(DevicePtr::new(region.address()), arena_bytes);
    let forward = Forward::new(&stream, &generation, &arena);
    println!(
        "workspace: {:.2} GiB needed of {:.2} GiB reserved (conditioning {:.0} MiB, encoder tiles \
         {:.0} MiB, tables {:.0} MiB)",
        needed as f64 / f64::from(1 << 30),
        arena_bytes as f64 / f64::from(1 << 30),
        conditioning_bytes as f64 / f64::from(1 << 20),
        encoder_bytes as f64 / f64::from(1 << 20),
        tables as f64 / f64::from(1 << 20)
    );

    // 6. The walk starts at the constants: the denoiser's AdaLN cache is a set of launches, and a
    //    walk that began at the conditioning would report a list missing them.
    let mut walk = Walk(Vec::new());
    ops::capture::set_sink(Some(&mut walk));
    let _sink = SinkGuard;

    let constant_views = allocate_tables(&arena, &constant_specs).map_err(show)?;
    denoiser
        .prepare_constants(&forward, &constant_views)
        .map_err(show)?;

    let mut state_host = host_tables(&state_specs).map_err(show)?;
    denoiser
        .prepare_state(&request, &layout, &mut state_host)
        .map_err(show)?;
    let state_views = allocate_tables(&arena, &state_specs).map_err(show)?;
    for (view, bytes) in state_views.iter().zip(&state_host) {
        upload_bytes(&stream, *view, bytes)?;
    }

    // 7. The runner's own buffers. The text region is a stated pattern rather than the text
    //    encoder's output (stage 4 is that), and it is the *refiner's* input, so it is written in the
    //    deployment's type; the source is the encoder's features, one block wide.
    let text = arena
        .tensor(dtype, Shape::new(&[u64::from(layout.text_rows), hidden]))
        .map_err(show)?;
    let source = arena
        .tensor(
            dtype,
            Shape::new(&[u64::from(layout.text_rows), source_width]),
        )
        .map_err(show)?;
    let prefix = arena
        .tensor(dtype, Shape::new(&[rows, hidden]))
        .map_err(show)?;
    // The pixels are the *encoder's* input, `[channels, frames, height, width]`: the buffer is
    // allocated in that order and filled in it, because a runner that filled the other order and
    // transposed on the way in would be one stride away from a silently permuted image.
    let channels = u64::from(config.video_vae.in_channels);
    let image = arena
        .tensor(
            DType::F32,
            Shape::new(&[
                channels,
                u64::from(pixels.frames),
                u64::from(pixels.height),
                u64::from(pixels.width),
            ]),
        )
        .map_err(show)?;
    let noise = arena
        .tensor(
            DType::F32,
            Shape::new(&[
                u64::from(dit.transformer.video_channels),
                u64::from(latent.frames),
                u64::from(latent.height),
                u64::from(latent.width),
            ]),
        )
        .map_err(show)?;
    let text_bytes = pseudo_bytes(text_elements, dtype);
    upload_bytes(&stream, text, &text_bytes)?;
    upload_bytes(&stream, source, &pseudo_bytes(source_elements, dtype))?;
    // The pixels are the encoder's input: `[frames, height, width, channels]`, normalised. A
    // deterministic ramp stands in for an image and the case says so.
    let mut pixels_host = Vec::with_capacity(pixel_elements as usize * 4);
    for plane in 0..channels {
        for element in 0..(pixel_elements / channels) {
            let value = ((element + plane) % 255) as f32 / 255.0;
            pixels_host.extend_from_slice(&value.to_le_bytes());
        }
    }
    upload_bytes(&stream, image, &pixels_host)?;
    // The anchoring draw sits at the offset the model gives the piece - not at zero by assumption:
    // a condition is drawn before the video and audio latents, and an audio-only reference takes a
    // zero-length entry at the cursor rather than no entry, so the index alignment survives.
    let offsets = layouts.condition_offsets(&request).map_err(show)?;
    let (start, elements) = offsets
        .first()
        .copied()
        .ok_or_else(|| "the model gave no condition offset for a one-piece request".to_owned())?;
    if elements != noise.shape.numel() {
        return Err(format!(
            "the piece's draw is {elements} elements and this harness allocated {}",
            noise.shape.numel()
        ));
    }
    let noise_layer = nn::Noise::new();
    noise_layer
        .draw(
            &forward,
            noise,
            nn::NoiseKey::Scalar(args.seed),
            0,
            0,
            start,
        )
        .map_err(show)?;

    // 8. The served entry point, once per request, over the layout the deployment made.
    let media = [MediaInput {
        role: MediaRole::FirstFrame,
        rows: pieces[0].0,
        audio_rows: pieces[0].1,
        pixels: Some(image),
        audio: None,
        noise: Some(noise),
    }];
    // A stop inside the conditioner still leaves the walk behind: the launches that led to it are the
    // evidence a cold start produces, and a stage that writes nothing on failure costs a round (the
    // rule `dt_layer` and `vae_decode` were both given this round). The page's files are cleared
    // before the call, so what is on disk afterwards is this run's and no earlier run's.
    let directory = args.out.join("latent_encoder");
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    clear_page(&directory)?;
    let outcome = denoiser.prepare_conditioning(
        &forward,
        Conditioning {
            text,
            source: Some(source),
            media: &media,
            images: None,
            audio: None,
            state: &state_views,
            step: None,
        },
    );
    if let Err(error) = outcome {
        write_page(&directory, &walk)?;
        return Err(show(error));
    }
    stream.synchronize().map_err(show)?;

    // 9. The readback: the prefix source, read at the rows the packing gathers each piece from.
    //
    // What this can and cannot say is the model's own claim. A piece whose pixels are black is a
    // legal request with zero rows, so the verdict is **at least one piece non-zero** - the failure
    // it catches is every piece zero, which is a condition the reader gathers nothing for
    // (`39a41fd55`). The zero row is a different claim: no write may reach it, so a non-zero value
    // there is a defect whatever the pieces hold.
    let produced = download(&stream, prefix)?;
    let values = to_f32(&produced, dtype);
    let mut readback = String::new();
    let mut non_zero = 0_usize;
    for (index, segment) in segments.iter().enumerate() {
        let first = u64::from(layout.text_rows) + u64::from(starts[index]);
        let rows_here = u64::from(segment.video_rows() + segment.audio_rows());
        let base = (first * hidden) as usize;
        let count = (rows_here * hidden) as usize;
        let piece = &values[base..base + count];
        let mean = piece.iter().map(|value| value.abs()).sum::<f32>() / count.max(1) as f32;
        if piece.iter().any(|value| *value != 0.0) {
            non_zero += 1;
        }
        readback.push_str(&format!(
            "piece {index} (condition {}): prefix rows [{first}, {}), {} visual rows and {} audio \
             rows, mean |x| {mean:.6e}\n",
            segment.condition,
            first + rows_here,
            segment.video_rows(),
            segment.audio_rows()
        ));
    }
    let zero_base = (u64::from(packing.zero_row()) * hidden) as usize;
    let zero_nonzero = values[zero_base..zero_base + hidden as usize]
        .iter()
        .filter(|value| **value != 0.0)
        .count();
    readback.push_str(&format!(
        "zero row {}: {zero_nonzero} non-zero of {hidden}\n",
        packing.zero_row()
    ));
    let text_nonzero = values[..(u64::from(layout.text_rows) * hidden) as usize]
        .iter()
        .filter(|value| **value != 0.0)
        .count();
    readback.push_str(&format!(
        "text region [0, {}): {text_nonzero} non-zero values\n",
        layout.text_rows
    ));
    print!("{readback}");

    // 10. The files. The container is FP32 with the deployment's type named in the case, so one
    //     writer serves every deployment and the reader is told what the block computed in.
    let mut container = Vec::with_capacity(values.len() * 4);
    for value in &values {
        container.extend_from_slice(&value.to_le_bytes());
    }
    write_h3t(
        &directory.join("condition_rows.h3t"),
        H3T_F32,
        &[rows, hidden],
        &container,
    )?;
    std::fs::write(directory.join("readback.txt"), &readback)
        .map_err(|error| format!("readback.txt: {error}"))?;
    // The anchoring draw, as the model received it: `source = t * clean + (1 - t) * draw`, so a
    // reference that re-derives the draw would be comparing a *convention* - which stream, which
    // counter, which index - and a seed mismatch reads as a defect in the encoder. One file removes
    // the convention; the projection stays the reference's own code.
    let draw = download(&stream, noise)?;
    write_h3t(
        &directory.join("condition_noise.h3t"),
        H3T_F32,
        &[
            u64::from(dit.transformer.video_channels),
            u64::from(latent.frames),
            u64::from(latent.height),
            u64::from(latent.width),
        ],
        &draw,
    )?;
    // The pixels as the encoder received them. A reference probe reads this file instead of
    // rebuilding the pattern, so a comparison of the encoder's arithmetic starts from one input
    // rather than from two decodes and two orders of a channel axis.
    write_h3t(
        &directory.join("pixels.h3t"),
        H3T_F32,
        &[
            channels,
            u64::from(pixels.frames),
            u64::from(pixels.height),
            u64::from(pixels.width),
        ],
        &pixels_host,
    )?;
    let case = format!(
        "{{\n  \"stage\": \"encode\",\n  \"task\": \"{}\",\n  \"canvas\": [{}, {}],\n  \
         \"frames\": {},\n  \"image\": {},\n  \"prompt_tokens\": {},\n  \"seed\": {},\n  \
         \"dtype\": \"{}\",\n  \"text_rows\": {},\n  \"condition_rows\": {},\n  \
         \"prefix_rows\": {},\n  \"hidden\": {},\n  \"pieces\": {},\n  \
         \"piece_rows\": [{}, {}],\n  \"pixels_file\": \"pixels.h3t\",\n  \
         \"pixels_shape\": [{}, {}, {}, {}],\n  \
         \"pixels\": \"deterministic per-channel ramp, stated\",\n  \
         \"text\": \"deterministic pattern, stated (stage 4 owns the text encoder)\",\n  \
         \"rows_file\": \"condition_rows.h3t\",\n  \"noise_file\": \"condition_noise.h3t\",\n  \
         \"rows_container\": \"f32\",\n  \
         \"posterior_fields\": \"not written: the trace that carries them is not on the served entry point\"\n}}\n",
        task_name(args.task),
        args.width,
        args.height,
        args.frames,
        args.image,
        args.text_tokens,
        args.seed,
        dtype,
        layout.text_rows,
        layout.condition_rows,
        rows,
        hidden,
        segments.len(),
        pieces[0].0,
        pieces[0].1,
        channels,
        u64::from(pixels.frames),
        u64::from(pixels.height),
        u64::from(pixels.width)
    );
    std::fs::write(directory.join("case.json"), case)
        .map_err(|error| format!("case.json: {error}"))?;
    std::fs::write(directory.join("walk.txt"), walk.text())
        .map_err(|error| format!("walk.txt: {error}"))?;
    println!(
        "wrote {}: {rows} x {hidden} {} as FP32, {} launches in the walk",
        directory.join("condition_rows.h3t").display(),
        dtype,
        walk.distinct()
    );

    if zero_nonzero != 0 {
        return Err(format!(
            "the prefix's zero row ({}) holds {zero_nonzero} non-zero values: a write reached past \
             the layout's condition capacity",
            packing.zero_row()
        ));
    }
    if text_nonzero == 0 {
        return Err(format!(
            "the prompt rows [0, {}) read as zeros after conditioning: the refiner wrote nothing \
             into the prefix source the denoiser gathers from",
            layout.text_rows
        ));
    }
    if non_zero == 0 {
        return Err(format!(
            "every one of the {segments_len} pieces reads as zeros in the prefix: the reader gathers \
             nothing where the encoder wrote, which is a dropped condition and not a tolerance",
            segments_len = segments.len()
        ));
    }
    Ok(())
}

/// The bytes one component's tensors occupy in the checkpoint, which is what the weights pool must
/// hold before any conversion.
fn stored_bytes(checkpoint: &Checkpoint, directory: &str) -> Result<(u64, u64), String> {
    let source = checkpoint.component(directory).map_err(show)?;
    let mut bytes = 0_u64;
    let mut tensors = 0_u64;
    for name in source.names() {
        if let Some(location) = source.tensor(name) {
            bytes += location.bytes.1.saturating_sub(location.bytes.0);
            tensors += 1;
        }
    }
    Ok((bytes, tensors))
}

/// The deployment's element type, as the loaded block reports it.
fn denoiser_dtype(denoiser: &Denoiser) -> DType {
    denoiser.dtype()
}

/// Remove the page's files, so a failed stage cannot be read as a previous run's success.
fn clear_page(directory: &Path) -> Result<(), String> {
    for name in [
        "case.json",
        "walk.txt",
        "condition_rows.h3t",
        "condition_noise.h3t",
        "pixels.h3t",
    ] {
        let path = directory.join(name);
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|error| format!("clearing {}: {error}", path.display()))?;
        }
    }
    Ok(())
}

/// The one file a stopped stage writes: the walk, so the launches leading to the stop are readable.
fn write_page(directory: &Path, walk: &Walk) -> Result<(), String> {
    let path = directory.join("walk.txt");
    std::fs::write(&path, walk.text()).map_err(|error| format!("{}: {error}", path.display()))?;
    println!(
        "stopped: {} launches written to {}",
        walk.distinct(),
        path.display()
    );
    Ok(())
}

/// One `.h3t` file: the container's magic, one element-type byte, a rank byte, the extents as `u64`,
/// then the payload.
fn write_h3t(path: &Path, element: u8, extents: &[u64], payload: &[u8]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(10 + extents.len() * 8 + payload.len());
    bytes.extend_from_slice(b"H3D1");
    bytes.push(element);
    bytes.push(extents.len() as u8);
    for extent in extents {
        bytes.extend_from_slice(&extent.to_le_bytes());
    }
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
}

/// A deterministic pattern of `elements` values in `dtype`, the same bytes on every host.
fn pseudo_bytes(elements: u64, dtype: DType) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((elements as usize) * dtype.bytes().unwrap_or(4) as usize);
    for element in 0..elements {
        let value = ((element % 97) as f32 + 1.0) / 97.0;
        match dtype {
            DType::BF16 => bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes()),
            DType::F16 => bytes.extend_from_slice(&half_bits(value).to_le_bytes()),
            _ => bytes.extend_from_slice(&value.to_le_bytes()),
        }
    }
    bytes
}

/// The nearest float16 bit pattern of `value`, round to nearest even.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    let mut half = (exponent as u16) << 10 | (mantissa >> 13) as u16;
    if mantissa & 0x1000 != 0 {
        half += 1;
    }
    sign | half
}

/// The bytes a list of tables occupies, in each table's own element type.
fn table_bytes(specs: &[nn::capability::StateTable]) -> u64 {
    specs
        .iter()
        .map(|spec| spec.shape.numel() * spec.dtype.bytes().unwrap_or(1))
        .sum()
}

/// Allocate one device tensor per table, in the table's own element type and extents.
fn allocate_tables(
    arena: &Arena,
    specs: &[nn::capability::StateTable],
) -> Result<Vec<TensorView>, String> {
    specs
        .iter()
        .map(|spec| arena.tensor(spec.dtype, spec.shape).map_err(show))
        .collect()
}

/// Zeroed host bytes for one table list, sized as the model declared each table.
fn host_tables(specs: &[nn::capability::StateTable]) -> Result<Vec<Vec<u8>>, String> {
    specs
        .iter()
        .map(|spec| {
            let bytes = spec
                .dtype
                .bytes()
                .ok_or_else(|| format!("table {} has no byte width", spec.name))?;
            Ok(vec![
                0_u8;
                usize::try_from(spec.shape.numel() * bytes)
                    .unwrap_or(usize::MAX)
            ])
        })
        .collect()
}

/// Upload host bytes into a tensor already allocated for them.
fn upload_bytes(stream: &platform::Stream, view: TensorView, bytes: &[u8]) -> Result<(), String> {
    let width = view
        .dtype
        .bytes()
        .ok_or_else(|| format!("a {} view has no byte width", view.dtype))?;
    let expected = usize::try_from(view.shape.numel() * width).unwrap_or(usize::MAX);
    if bytes.len() != expected {
        return Err(format!(
            "{} bytes for a {} view of {:?}, which holds {expected}",
            bytes.len(),
            view.dtype,
            view.shape.dims()
        ));
    }
    // SAFETY: `bytes` outlives the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            view.ptr.address(),
            bytes.as_ptr() as u64,
            bytes.len(),
            stream,
        )
        .map_err(show)?;
    }
    stream.synchronize().map_err(show)?;
    Ok(())
}

/// The whole of a tensor's device memory, as bytes.
fn download(stream: &platform::Stream, view: TensorView) -> Result<Vec<u8>, String> {
    let width = view
        .dtype
        .bytes()
        .ok_or_else(|| format!("a {} view has no byte width", view.dtype))?;
    let count = usize::try_from(view.shape.numel() * width)
        .map_err(|_| format!("{:?} is not a host allocation", view.shape.dims()))?;
    let mut bytes = vec![0_u8; count];
    // SAFETY: `bytes` is written by the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(bytes.as_mut_ptr() as u64, view.ptr.address(), count, stream)
            .map_err(show)?;
    }
    stream.synchronize().map_err(show)?;
    Ok(bytes)
}

/// The values of a buffer of `dtype` as FP32, so one container serves every deployment type.
fn to_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
            .collect(),
        DType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                let sign = if bits & 0x8000 != 0 { -1.0_f32 } else { 1.0 };
                let exponent = ((bits >> 10) & 0x1f) as i32;
                let mantissa = (bits & 0x3ff) as f32;
                if exponent == 0 {
                    sign * mantissa * 2.0_f32.powi(-24)
                } else if exponent == 0x1f {
                    sign * f32::INFINITY
                } else {
                    sign * (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent - 15)
                }
            })
            .collect(),
        _ => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect(),
    }
}

/// Every committed module of every operator the store holds, as one generation.
///
/// The store is read, never written, and no descriptor is named here: a launch that finds no module
/// says which descriptor it needed, which is the fact a missing cell is fixed from.
fn load_generation(device: &platform::Device, root: &Path) -> Result<ops::Generation, String> {
    if !root.is_dir() {
        return Err(format!(
            "the store root {} is not a directory: nothing is installed there, so no stage past              `weights` can launch",
            root.display()
        ));
    }
    let mut entries = Vec::new();
    let mut installed = 0_usize;
    let mut operators: Vec<String> = std::fs::read_dir(root)
        .map_err(|error| format!("{}: {error}", root.display()))?
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
    for backend in std::fs::read_dir(op_root).into_iter().flatten().flatten() {
        for revision in std::fs::read_dir(backend.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            for target in std::fs::read_dir(revision.path())
                .into_iter()
                .flatten()
                .flatten()
            {
                for descriptor in std::fs::read_dir(target.path())
                    .into_iter()
                    .flatten()
                    .flatten()
                {
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
) -> Result<ops::KernelEntry, String> {
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
            .map_err(|error| format!("reading {}: {error}", cubin.display()))?;
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
        .map_err(|error| format!("{}: {error}", directory.display()))?;
        return kernel.ok_or_else(|| {
            format!(
                "{}: this build of `ops` has no operator `{operator}`",
                directory.display()
            )
        });
    }
    let library = std::fs::read_dir(directory)
        .into_iter()
        .flatten()
        .flatten()
        .map(|file| file.path())
        .find(|path| path.extension().is_some_and(|kind| kind == "so"))
        .ok_or_else(|| {
            format!(
                "{}: neither module.cubin nor a shared object",
                directory.display()
            )
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
    .map_err(|error| format!("{}: {error}", directory.display()))?;
    installed.ok_or_else(|| {
        format!(
            "{}: this build of `ops` has no operator `{operator}`",
            directory.display()
        )
    })
}

/// One string field of a JSON document, without a parser: the store's files are flat.
fn json_field(text: &str, field: &str) -> Option<String> {
    let needle = format!("\"{field}\"");
    let at = text.find(&needle)? + needle.len();
    let rest = text[at..].trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}

/// One launch of the walk: the operator and the descriptor key it bound.
struct Walk(Vec<(&'static str, String)>);

impl ops::capture::Sink for Walk {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator, key.to_owned()));
    }
}

impl Walk {
    /// The walk as the store's key list is written: one `operator<TAB>key` per launch, in order.
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

    /// How many distinct programs the walk bound.
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

/// Removes the capture sink when it leaves scope, so a failing launch cannot leave one installed.
struct SinkGuard;

impl Drop for SinkGuard {
    fn drop(&mut self) {
        ops::capture::set_sink(None);
    }
}

/// Device storage for weights: a resident pool, one buffer per tensor, every publish waited on.
struct DeviceWeights {
    pools: platform::Allocator,
    buffers: Vec<platform::Buffer>,
    bytes: u64,
}

impl DeviceWeights {
    fn new(device: &platform::Device, requested: u64) -> Result<Self, String> {
        let budget = std::sync::Arc::new(NoLimit);
        let mut pools = platform::Allocator::new(device, budget).map_err(show)?;
        let mut wanted = requested;
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
            "the driver refused every resident reservation down to 1 GiB: the deployment's weights \
             cannot be held on this device"
                .to_owned()
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
        _key: &WeightKey,
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
        Ok(backing)
    }
}

fn show<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}
