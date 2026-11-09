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
//!                  the projection, then the rows read back through the packing's own layout -
//!                  `<out>/latent_encoder/condition_rows.h3t` with `case.json`, `walk.txt` and
//!                  `readback.txt`
//! ```
//!
//! **It runs the served entry point, not a method of convenience.** `Denoiser::prepare_conditioning`
//! is what a runner calls once per request, and it is the arm that carried the defect that every
//! condition was written where nothing gathers it (`39a41fd55`).
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

use minimax_h3::VideoTask;
use minimax_h3::checkpoint::Dit;
use minimax_h3::config::Config;
use minimax_h3::denoiser::{Denoiser, Layouts};
use minimax_h3::encoding::Extent;
use minimax_h3::partition::{Component, Components, Partition, Ulysses};
use nn::capability::{Conditioning, Denoiser as _, MediaInput, MediaRole};
use nn::{
    Arena, Backing, Checkpoint, ConditionRole, ConditionShape, DType, DevicePtr, Forward,
    RetainedWeights, Shape, TensorView, VideoRequest, WeightKey, WeightLoader,
};

/// The two checkpoint directories a conditioned deployment reads its media from.
const VAE: &str = "vae";
const AUDIO_VAE: &str = "audio_vae";

/// The `.h3t` element-type codes this harness writes, from `tools/dump_tables.py`'s table.
const H3T_F32: u8 = 1;

/// What the conditional harness takes, printed for `--help`.
///
/// This is the one harness a conditioned deployment (`fl2va`, `ref2va`) is driven from, and its flags
/// are how a task is named: the stage, the task, the condition images, and the canvas the decode is
/// checked at. A caller who guesses wrong gets a refusal that names one flag at a time; this prints
/// the set.
fn usage() -> String {
    "conditioned --root <checkpoint> [--store <dir>] [--stage weights|plan|encode]\n\
     \x20          [--task t2va|fl2va|ref2va] [--image <pixels>] [--canvas <width>x<height>]\n\
     \x20          [--frames <n>] [--text-tokens <n>] [--dtype auto|bf16|f16|f32]\n\
     \x20          [--device <n>] [--seed <n>] [--place deployment|conditioner] [--diag <dir>]\n\
     \x20          [--out <dir>] [--commit <revision>]\n\n\
     --stage weights reads the checkpoint's own layout, `plan` the encoder's tile walk without a\n\
     device, and `encode` one image on a device (which needs `--store`, or `UNISERVE_STORE`).\n"
        .to_owned()
}

fn main() -> ExitCode {
    if std::env::args()
        .skip(1)
        .any(|flag| flag == "--help" || flag == "-h")
    {
        print!("{}", usage());
        return ExitCode::SUCCESS;
    }
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
    /// A **reference**'s raster, in pixels. Unused by `fl2va`: a keyframe is fitted to the
    /// generated canvas, so an anchor's raster is the canvas and no flag states it.
    image: u32,
    /// The request's seed, which addresses the condition's draw.
    seed: u64,
    device: u32,
    /// The element type the deployment loads at. `auto` is the design's default: the checkpoint's own
    /// representation, nothing converted at load, and each request's pin decides.
    dtype: Option<DType>,
    /// The program store `--stage encode` reads its modules from.
    store: Option<PathBuf>,
    /// Which blocks this process places: the deployment's, or only the conditioner's.
    place: Place,
    /// Where the block-by-block diagnosis writes `<block>/<name>.npy` (`specs/worker.md` 10).
    ///
    /// These files are the **candidate's half** of a comparison whose reference half is
    /// `tools/dump_*.py`; each is a tensor this stage already produced, written in the container the
    /// python dumpers write, so a pair is a shape, a maximum difference and the index of the worst
    /// element - never a summary word.
    diag: Option<PathBuf>,
    /// The commit this run computes at, recorded in the diagnosis page: a `.npy` with no commit
    /// beside it is a number nobody can date.
    commit: Option<String>,
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    // The canvas is a **deployment** fact, and the store is built for the six the checkpoint offers:
    // a raster outside them asks for a `frame_pad` class and a `scaled_add` ladder no served canvas
    // needs, so the run would stop at a missing program after the whole load. Every stage refuses it,
    // because the plan's numbers are a claim about the same case the device stage spends.
    if !minimax_h3::NAMED_CANVASES
        .iter()
        .any(|canvas| canvas.width == args.width && canvas.height == args.height)
    {
        let offered = minimax_h3::NAMED_CANVASES
            .iter()
            .map(|canvas| format!("{}x{}", canvas.width, canvas.height))
            .collect::<Vec<String>>()
            .join(", ");
        return Err(format!(
            "`{}x{}` is not a canvas the deployment offers, and the store is built for the ones it \
             does: {}",
            args.width, args.height, offered
        ));
    }
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
            root: PathBuf::new(),
            out: PathBuf::new(),
            stage: "weights".to_owned(),
            task: VideoTask::FirstLastToVideoAudio,
            width: 1344,
            height: 768,
            frames: 5,
            text_tokens: 15,
            // The anchor's raster is the canvas's short edge, which is a whole number of the
            // encoder's tiles at every served canvas: 768 = 3 x 256.
            image: 768,
            seed: 0,
            device: 0,
            dtype: None,
            store: std::env::var_os("UNISERVE_STORE").map(PathBuf::from),
            place: Place::Deployment,
            diag: None,
            commit: std::env::var("H3_COMMIT").ok(),
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
                "--place" => {
                    args.place = match value()?.as_str() {
                        "deployment" => Place::Deployment,
                        "conditioner" => Place::Conditioner,
                        other => {
                            return Err(format!(
                                "`{other}` is not a block selection; write deployment (every block this \
                                 stage serves) or conditioner (the blocks a conditioned encode \
                                 reads)"
                            ));
                        }
                    };
                }
                "--diag" => args.diag = Some(PathBuf::from(value()?)),
                "--commit" => args.commit = Some(value()?),
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
        if args.out.as_os_str().is_empty() {
            eprintln!("name the output directory with --out");
            std::process::exit(2);
        }
        if args.root.as_os_str().is_empty() {
            args.root = std::env::var_os("UNISERVE_H3_MODEL")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    eprintln!("name the checkpoint with --root or UNISERVE_H3_MODEL");
                    std::process::exit(2)
                });
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
/// The condition rows one case asks for, from the packing's own segments.
///
/// A stage's layout is sized **before** the checkpoint is loaded, so the rows cannot come from the
/// loaded denoiser: they come from the packing the request would be laid out under.
///
/// **This is not what sizes the layout.** The packed row bound - and therefore the dense attention's
/// `max_kv_len`, which is part of a program's key - follows the arrangement a stage is *loaded* with,
/// so a layout sized to one case would key its attention at a different bound from every other stage
/// and from any store: it would stop on a missing cell before it computed anything. One store serves
/// every task, so the layout is sized to the widest one (`store_capacity`) and a case's own rows are
/// only compared with it.
fn case_condition_rows(config: &Config, request: &VideoRequest) -> Result<u32, String> {
    // `condition_segments` reduces each shape's pixels to its latent raster through the packing's own
    // rule, and counts an image, a video and a keyframe as the model does.
    let canvas = minimax_h3::denoiser::canvas_of(request).map_err(show)?;
    let clip = minimax_h3::encoding::Clip::of(&config.video_vae).map_err(show)?;
    let segments = minimax_h3::packing::condition_segments(
        &minimax_h3::denoiser::conditions_of(request),
        &canvas,
        clip,
    )
    .map_err(show)?;
    let rows: u32 = segments
        .iter()
        .map(|segment| segment.video_rows() + segment.audio_rows())
        .sum();
    // An arrangement states its capacity in whole units, so the case's rows round up to them.
    minimax_h3::partition::capacity(config, Dit::Base, rows).map_err(show)
}

/// A case whose pieces need more rows than the store's capacity is refused here, by both numbers.
///
/// The layout is the store's, so this cannot be a silent clamp: a request that overflows it would be
/// laid out at a capacity it cannot hold and stop inside the packing with a count instead of a case.
fn require_case_fits(task: VideoTask, case_rows: u32, capacity: u32) -> Result<(), String> {
    if case_rows > capacity {
        return Err(format!(
            "a {task:?} case needs {case_rows} condition rows and this store's capacity is \
             {capacity}: the store is walked for the widest task it serves, so a case beyond it is a \
             deployment the store was not built for"
        ));
    }
    Ok(())
}

/// The condition capacity one store must hold: the widest of the tasks it serves.
///
/// A `max_kv_len` is exact and belongs to a program's key, so a deployment that served two bounds
/// would need two sets of attention cells. The store is one, so the layout is the widest task's -
/// `ref2va` here, whose reference clip reserves two latent frames at the largest canvas - and a
/// narrower task simply does not fill it.
fn store_capacity(config: &Config) -> Result<u32, String> {
    let mut widest = 0;
    for task in [
        VideoTask::TextToVideoAudio,
        VideoTask::FirstLastToVideoAudio,
        VideoTask::ReferenceToVideoAudio,
    ] {
        widest = widest.max(minimax_h3::partition::condition_rows(config, task).map_err(show)?);
    }
    Ok(widest)
}

fn stage_partition(
    config: &Config,
    task: VideoTask,
    place: Place,
    condition_rows: u32,
) -> Result<Partition, String> {
    let serving = Partition::serving(config, task, Ulysses::single(), 1, 0, 0)
        .map_err(show)?
        .sized(minimax_h3::partition::RELEASED_TEXT_ROWS, condition_rows);
    let placed = serving.blocks.without(Component::TextEncoder);
    Ok(serving.placing(match place {
        Place::Deployment => placed,
        // The blocks a **conditioned encode** reads and nothing else: the DiT, which carries both
        // condition encoders, and the `latent_encoder` entry that names their directories. The two
        // decoders are placed by a served deployment because it is one process; a diagnosis of the
        // conditioner does not read them, and on a student checkpoint (`FastVideo-FastH3-8-Step-V2`
        // holds no `audio_vae` weights) placing them would refuse the load for want of a block the
        // encode never launches.
        Place::Conditioner => Components::of([Component::Transformer, Component::LatentEncoder]),
    }))
}

/// Which blocks this process places.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    /// The deployment's own, minus the text encoder: what a served stage holds.
    Deployment,
    /// The conditioner's inputs only, for a diagnosis of the conditioned path.
    Conditioner,
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

    let partition = stage_partition(&config, args.task, args.place, store_capacity(&config)?)?;
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
         the positions the packing gathers them from - so they need the loaded block and are printed \
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

/// The conditions a task's case carries, and the counts a request states for them.
///
/// One statement of the case for both stages: `plan` sizes the buffers the request needs and `encode`
/// runs it, and two constructions of one request is how the printed numbers and the run drift apart.
///
/// `fl2va` anchors the first and the last generated frame; `ref2va` carries one reference clip of
/// [`minimax_h3::CLIP_LATENTS`] latent frames with a soundtrack as long as the clip plays. `t2va` has
/// no condition at all, so it is another stage's case and this one refuses it.
fn case_conditions(args: &Args) -> Result<(Vec<ConditionShape>, u32, u32, u32), String> {
    match args.task {
        VideoTask::FirstLastToVideoAudio => {
            // A keyframe is **fitted to the generated canvas** - the packing refuses any other
            // raster (`H3 keyframes are fitted to the target canvas`) - so the anchor's shape is the
            // canvas and `--image` states nothing here.
            let conditions = [ConditionRole::FirstFrame, ConditionRole::LastFrame]
                .into_iter()
                .map(|role| ConditionShape {
                    role,
                    frames: 1,
                    height: args.height,
                    width: args.width,
                    audio_samples: 0,
                })
                .collect();
            Ok((conditions, 1, 1, 0))
        }
        VideoTask::ReferenceToVideoAudio => {
            // The reference is a still with a soundtrack, at `--image`'s raster: the checkpoint's
            // condition capacity bounds a reference clip, not this harness. It is 192 rows - nine
            // images, three videos and two keyframes of 64-row units (`RELEASED_CONDITION_ROWS`) -
            // so a reference video is admissible at up to three 64-row frames, while a full-canvas
            // reference at 1344x768 is 1008 rows and the packing refuses it, correctly.
            // One clip of pixel frames: the packing refuses a soundtrack on an *image* reference
            // (`an H3 image reference carries no audio`), so the reference is the clip's own length
            // and its audio is the clip's.
            let samples = minimax_h3::CLIP_FRAMES * minimax_h3::AUDIO_SAMPLE_RATE / minimax_h3::FPS;
            let conditions = vec![ConditionShape {
                role: ConditionRole::Reference,
                frames: minimax_h3::CLIP_FRAMES,
                // The shape states **pixels**: the packing reduces them to the latent raster its own
                // `Visual` carries, so dividing here would divide twice.
                height: args.image,
                width: args.image,
                audio_samples: samples,
            }];
            Ok((conditions, 0, 0, 1))
        }
        VideoTask::TextToVideoAudio => Err(
            "`t2va` carries no condition, and this stage is the conditioned path: run `fl2va` or \
             `ref2va` here, and `t2va` in stage 3"
                .to_owned(),
        ),
    }
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
    // The case before the partition: a stage's condition capacity is the rows its case asks for, and
    // the rows cannot be read off a layout sized by a constant.
    let (conditions, first_frames, last_frames, references) = case_conditions(args)?;
    let request = VideoRequest {
        width: args.width,
        height: args.height,
        frames: args.frames,
        fps: minimax_h3::FPS,
        first_frames,
        last_frames,
        references,
        text_tokens: args.text_tokens,
        conditions: &conditions,
        vision_spans: &[],
        condition_rows: None,
    };
    // The layout is the store's, not the case's: a program's key carries the bound the arrangement
    // states, so sizing it to this case would key attention differently from the store that serves it.
    let condition_rows = store_capacity(&config)?;
    let case_rows = case_condition_rows(&config, &request)?;
    require_case_fits(args.task, case_rows, condition_rows)?;
    let partition = stage_partition(&config, args.task, args.place, condition_rows)?;
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
    // The conditions the run will encode, stated by the same function the device stage uses: a plan
    // that named only the canvas would describe a request the run does not make.
    for (index, shape) in conditions.iter().enumerate() {
        println!(
            "condition {index}: {:?}, {} frame(s) at {}x{} px, {} samples",
            shape.role, shape.frames, shape.height, shape.width, shape.audio_samples
        );
    }
    println!(
        "rows: text capacity {}, condition capacity {} (this case needs {case_rows}), prefix {} rows \
         of {}",
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
/// One condition of a request, as this harness builds it.
///
/// Every geometric field is the model's: a piece's latent frames and raster come from the packing's
/// `Visual`, its packed rows from the same place, and the anchoring draw's start and element count
/// from the layout's own cursor. The harness states only what the checkpoint cannot - the pixels, the
/// samples, and the ramp that fills them.
struct PiecePlan {
    /// Where the piece anchors, which is the role the conditioner writes its rows under.
    role: MediaRole,
    /// Packed visual rows the layout reserves for the piece.
    rows: u32,
    /// Packed audio rows it reserves.
    audio_rows: u32,
    /// The pixels' extent in the encoder's own order, absent for an audio-only reference.
    pixels: Option<Extent>,
    /// The anchoring draw's latent extent, absent for an audio-only reference.
    draw: Option<Extent>,
    /// Where the draw starts in the request's noise stream.
    draw_start: u64,
    /// The samples the soundtrack's encoder reads, absent when the piece has no soundtrack.
    samples: Option<u64>,
}

/// One piece's buffers, allocated: the views the conditioner reads and the host bytes they were
/// filled from.
struct PieceBuffers {
    pixels: Option<TensorView>,
    pixels_host: Vec<u8>,
    draw: Option<TensorView>,
    audio: Option<TensorView>,
}

fn encode(args: &Args) -> Result<(), String> {
    let store_root = args.store.clone().ok_or_else(|| {
        "stage `encode` needs a program store: pass --store <path> or set UNISERVE_STORE".to_owned()
    })?;
    // `--image` states a **reference**'s raster. An `fl2va` keyframe is fitted to the generated
    // canvas, so an anchor's raster is the canvas and the flag states nothing for that task.
    let anchors = args.task == VideoTask::FirstLastToVideoAudio;
    if anchors {
        println!(
            "case: {} anchors the generated canvas, so --image {} is unused and each anchor is \
             encoded from the canvas's own {}x{} pixels",
            task_name(args.task),
            args.image,
            args.width,
            args.height
        );
    } else if !args
        .image
        .is_multiple_of(minimax_h3::video_vae::TILE_PIXELS)
    {
        return Err(format!(
            "the reference is {} px and the encoder's tile is {} px, so a raster that is not a \
             whole number of tiles is not this stage's case",
            args.image,
            minimax_h3::video_vae::TILE_PIXELS
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
    let (conditions, first_frames, last_frames, references) = case_conditions(args)?;
    let request = VideoRequest {
        width: args.width,
        height: args.height,
        frames: args.frames,
        fps: minimax_h3::FPS,
        first_frames,
        last_frames,
        references,
        text_tokens: args.text_tokens,
        conditions: &conditions,
        vision_spans: &[],
        condition_rows: None,
    };
    // The layout is the store's, not the case's: a program's key carries the bound the arrangement
    // states, so sizing it to this case would key attention differently from the store that serves it.
    let condition_rows = store_capacity(&config)?;
    let case_rows = case_condition_rows(&config, &request)?;
    require_case_fits(args.task, case_rows, condition_rows)?;
    let partition = stage_partition(&config, args.task, args.place, condition_rows)?;
    let dit = config
        .denoisers
        .get(&Dit::Base)
        .ok_or_else(|| "the checkpoint states no base DiT".to_owned())?
        .clone();
    let arrangement = partition.arrangement(&dit).map_err(show)?;
    let layouts = Layouts::new(dit.clone(), arrangement).map_err(show)?;

    // 3. The case: the conditions the task names, and a request the deployment lays out with its own
    //    rule. `fl2va` anchors the first and the last generated frame; `ref2va` carries one reference
    //    clip with its own soundtrack. `t2va` has no condition at all, so it is another stage's case
    //    and this one refuses it rather than running an unconditioned request down a conditioned path.
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
        // The example's own raster: it exercises the conditioned path against the checkpoint's
        // tile canvas, which is what it can state without a deployment's options.
        let tile = minimax_h3::video_vae::TILE_PIXELS;
        minimax_h3::load(
            &checkpoint,
            partition,
            deployment,
            &[(tile, tile)],
            &mut loader,
        )
        .map_err(show)?
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
    let mode = match deployment {
        Some(stated) => format!("stated by --dtype {stated}"),
        None => "auto, the checkpoint's own".to_owned(),
    };
    println!(
        "loaded: {dtype} is the stack's element type ({mode}), hidden {hidden}, \
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
    // A conditioned task packs one piece per condition: `fl2va` anchors the first and the last
    // generated frame, and `ref2va` carries a reference with its own soundtrack. This stage runs the
    // request's own pieces - refusing a second one is how both tasks stayed unexercised while the
    // only path under test was a single still.
    if segments.is_empty() {
        return Err(format!(
            "a {} request packs no condition, and this stage's case is the conditioned path",
            task_name(args.task)
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
    // **One buffer, not two.** The model's `prepare_conditioning` writes a request's condition rows
    // into `Conditioning::text`, which is the *prefix source* the runner owns - the text capacity,
    // the condition capacity and the zero row - so this stage allocates that one buffer and reads the
    // verdict out of it. A stage that kept a prompt-sized `text` apart from a prefix-sized `prefix`
    // had the model write into the first and compared the second, and it refused at the capacity
    // check before it could show that.
    let text_elements = rows * hidden;
    let source_elements = u64::from(layout.text_rows) * source_width;
    let pixel_elements = u64::from(config.video_vae.in_channels)
        * u64::from(pixels.frames)
        * u64::from(pixels.height)
        * u64::from(pixels.width);
    let draw_elements = u64::from(dit.transformer.video_channels)
        * u64::from(latent.frames)
        * u64::from(latent.height)
        * u64::from(latent.width);
    // The request's pieces, from the layout's own cursors: `condition_offsets` states each draw's
    // start and element count, so a harness that derived either would be a second statement of the
    // packing's arithmetic.
    let offsets = layouts.condition_offsets(&request).map_err(show)?;
    if offsets.len() != segments.len() {
        return Err(format!(
            "the model packed {} conditions and gave {} draw cursors",
            segments.len(),
            offsets.len()
        ));
    }
    let channels = u64::from(config.video_vae.in_channels);
    let video_channels = u64::from(dit.transformer.video_channels);
    // The audio encoder reads a track's samples; its latent frames are what the packing counts, so
    // the hop is the one conversion between the two - and it is the checkpoint's own, which
    // `Config::check` has already refused a mismatch on.
    let hop = u64::from(minimax_h3::AUDIO_HOP);
    let mut plans: Vec<PiecePlan> = Vec::with_capacity(segments.len());
    for (index, segment) in segments.iter().enumerate() {
        let (start, elements) = offsets[index];
        let draw = segment.visual.map(|visual| Extent {
            frames: visual.frames,
            height: visual.raster.height,
            width: visual.raster.width,
        });
        let expected = draw.map_or(0, |extent| {
            video_channels
                * u64::from(extent.frames)
                * u64::from(extent.height)
                * u64::from(extent.width)
        });
        if elements != expected {
            return Err(format!(
                "condition {index} draws {elements} elements and its visual holds {expected}: the \
                 layout's cursor and the packing's raster disagree"
            ));
        }
        plans.push(PiecePlan {
            role: match segment.anchor {
                Some(minimax_h3::packing::Anchor::FirstFrame) => MediaRole::FirstFrame,
                Some(minimax_h3::packing::Anchor::LastFrame) => MediaRole::LastFrame,
                None => MediaRole::Reference,
            },
            rows: segment.video_rows(),
            audio_rows: segment.audio_rows(),
            // A piece's pixels follow its own latent raster, not the canvas: an anchor is fitted to
            // the canvas while a reference keeps the raster it arrives at, and the packing's
            // `Visual` states which.
            pixels: segment.visual.map(|visual| Extent {
                frames: u32::from(visual.frames > 0),
                height: visual.raster.height * minimax_h3::packing::LATENT_COMPRESSION,
                width: visual.raster.width * minimax_h3::packing::LATENT_COMPRESSION,
            }),
            draw,
            draw_start: start,
            samples: (segment.audio_frames > 0).then(|| u64::from(segment.audio_frames) * hop),
        });
    }
    // The pieces after the first are the harness's own cost: the first one's pixels and draw are
    // already in `pixel_elements` and `draw_elements`.
    let mut extras = 0_u64;
    for plan in plans.iter().skip(1) {
        if let Some(extent) = plan.pixels {
            extras += channels
                * u64::from(extent.frames)
                * u64::from(extent.height)
                * u64::from(extent.width)
                * 4;
        }
        if let Some(extent) = plan.draw {
            extras += video_channels
                * u64::from(extent.frames)
                * u64::from(extent.height)
                * u64::from(extent.width)
                * 4;
        }
        if let Some(samples) = plan.samples {
            extras += samples * 4;
        }
    }
    let tables = table_bytes(&constant_specs) + table_bytes(&state_specs);
    // Every element this harness allocates for itself is four bytes except the two text buffers,
    // which are the deployment's type: the draws and the pixels are FP32 by the contract.
    let width = dtype.bytes().unwrap_or(4);
    let needed = conditioning_bytes
        + encoder_bytes
        + tables
        + (text_elements + source_elements) * width
        + (pixel_elements + draw_elements) * 4
        + extras;
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
        .tensor(dtype, Shape::new(&[rows, hidden]))
        .map_err(show)?;
    let source = arena
        .tensor(
            dtype,
            Shape::new(&[u64::from(layout.text_rows), source_width]),
        )
        .map_err(show)?;
    // The diagnosis's own views: one `[rows, channels * 4]` F32 buffer per **visual** condition, for
    // the rows the video encoder emits before the anchor mixes the draw in and before the projection.
    // They are what a reference's encode probe returns, so a harness that offered none could compare
    // only the projected prefix - and a difference there cannot say whether the encoder or the
    // projection made it.
    let mut clean_views = Vec::new();
    for plan in &plans {
        if plan.rows == 0 {
            continue;
        }
        clean_views.push(
            arena
                .tensor(
                    DType::F32,
                    Shape::new(&[
                        u64::from(plan.rows),
                        u64::from(dit.transformer.video_channels) * 4,
                    ]),
                )
                .map_err(show)?,
        );
    }
    // **The prompt rows only, and the rest zeroed.** The prefix is one buffer with three regions,
    // and the verdict reads the last of them: a pattern written across all three would make the zero
    // row non-zero and the page would report a write that never happened. A served runner starts the
    // buffer zeroed too, so this is the request's own initial state rather than a convenience.
    let mut text_bytes = pseudo_bytes(u64::from(layout.text_rows) * hidden, dtype);
    text_bytes.resize(
        usize::try_from(text_elements * dtype.bytes().unwrap_or(4)).map_err(show)?,
        0,
    );
    upload_bytes(&stream, text, &text_bytes)?;
    upload_bytes(&stream, source, &pseudo_bytes(source_elements, dtype))?;

    // The pixels are the *encoder's* input, `[channels, frames, height, width]`, and each anchoring
    // draw is `[video channels, latent frames, height, width]`: both are allocated in that order and
    // filled in it, because a runner that filled the other order and transposed on the way in would
    // be one stride away from a silently permuted image. A deterministic ramp stands in for an image
    // and for a track, one phase per condition, and the case says so.
    let noise_layer = nn::Noise::new();
    let mut buffers: Vec<PieceBuffers> = Vec::with_capacity(plans.len());
    for (index, plan) in plans.iter().enumerate() {
        let mut pixels_host = Vec::new();
        let pixels = match plan.pixels {
            Some(extent) => {
                // **The condition's pixels are a scene, five axes**: `[1, channels, frames, height,
                // width]`. The encoder's own input is the four-axis raster `[channels, frames,
                // height, width]`, and the conditioned path reads the batch axis before it takes the
                // tile views - a four-axis buffer is refused with its own shape, which is what this
                // harness earned until the axis was here.
                let shape = Shape::new(&[
                    1,
                    channels,
                    u64::from(extent.frames),
                    u64::from(extent.height),
                    u64::from(extent.width),
                ]);
                let view = arena.tensor(DType::F32, shape).map_err(show)?;
                let plane =
                    u64::from(extent.frames) * u64::from(extent.height) * u64::from(extent.width);
                pixels_host = vec![0_u8; usize::try_from(channels * plane * 4).map_err(show)?];
                for channel in 0..channels {
                    for element in 0..plane {
                        let value = ((element + channel + index as u64) % 255) as f32 / 255.0;
                        let at = ((channel * plane + element) * 4) as usize;
                        pixels_host[at..at + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
                upload_bytes(&stream, view, &pixels_host)?;
                Some(view)
            }
            None => None,
        };
        let draw = match plan.draw {
            Some(extent) => {
                let shape = Shape::new(&[
                    video_channels,
                    u64::from(extent.frames),
                    u64::from(extent.height),
                    u64::from(extent.width),
                ]);
                let view = arena.tensor(DType::F32, shape).map_err(show)?;
                // The draw sits at the offset the model gives the piece - not at zero by assumption:
                // a condition is drawn before the video and audio latents, and an audio-only
                // reference takes a zero-length entry at the cursor rather than no entry, so the
                // index alignment survives.
                noise_layer
                    .draw(
                        &forward,
                        view,
                        nn::NoiseKey::Scalar(args.seed),
                        0,
                        0,
                        plan.draw_start,
                    )
                    .map_err(show)?;
                Some(view)
            }
            None => None,
        };
        let audio = match plan.samples {
            Some(samples) => {
                // The audio encoder reads `[batch, 1, samples]` and writes channel-major rows, so the
                // harness hands it the track and nothing else.
                let view = arena
                    .tensor(DType::F32, Shape::new(&[1, 1, samples]))
                    .map_err(show)?;
                let mut host = Vec::with_capacity(usize::try_from(samples * 4).map_err(show)?);
                for sample in 0..samples {
                    let phase = (sample % 480) as f32 / 480.0;
                    let value = ((index as f32) * 0.25 + phase).sin() * 0.5;
                    host.extend_from_slice(&value.to_le_bytes());
                }
                upload_bytes(&stream, view, &host)?;
                Some(view)
            }
            None => None,
        };
        buffers.push(PieceBuffers {
            pixels,
            pixels_host,
            draw,
            audio,
        });
    }
    let first = buffers
        .first()
        .ok_or_else(|| "the request packed no condition to encode".to_owned())?;
    let noise = first
        .draw
        .ok_or_else(|| "the first condition carries no draw".to_owned())?;
    let pixels_host = first.pixels_host.clone();

    // 8. The served entry point, once per request, over the layout the deployment made: every
    //    condition the packing named, in request order, with the role its anchor gives it.
    let media: Vec<MediaInput> = plans
        .iter()
        .zip(&buffers)
        .map(|(plan, buffer)| MediaInput {
            role: plan.role,
            rows: plan.rows,
            audio_rows: plan.audio_rows,
            pixels: buffer.pixels,
            audio: buffer.audio,
            noise: buffer.draw,
        })
        .collect();
    // A stop inside the conditioner still leaves the walk behind: the launches that led to it are the
    // evidence a cold start produces, and a stage that writes nothing on failure costs a round (the
    // rule `dt_layer` and `vae_decode` were both given this round). The page's files are cleared
    // before the call, so what is on disk afterwards is this run's and no earlier run's.
    let directory = args.out.join("latent_encoder");
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    clear_page(&directory)?;
    let trace = minimax_h3::denoiser::ConditionTrace {
        source: None,
        refined: None,
        draws: &[],
        clean: &clean_views,
    };
    let outcome = denoiser.prepare_conditioning_traced(
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
        Some(&trace),
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
    let produced = download(&stream, text)?;
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
    // The encoder's unprojected rows, one file per visual condition: the value the reference's
    // `--probe encode_rows` computes from the same pixels, so the comparison is one command rather
    // than a rewrite of the reference's encoder. `rows.h3t` is the name it reads; a request of two
    // visual conditions writes `rows_0.h3t` and `rows_1.h3t` and the case names them.
    let mut rows_files = Vec::new();
    for (index, view) in clean_views.iter().enumerate() {
        let name = if clean_views.len() == 1 {
            "rows.h3t".to_owned()
        } else {
            format!("rows_{index}.h3t")
        };
        let values = download(&stream, *view)?;
        let elements = values.len() / 4;
        write_h3t(
            &directory.join(&name),
            H3T_F32,
            &[
                u64::from(
                    plans
                        .iter()
                        .filter(|plan| plan.rows > 0)
                        .nth(index)
                        .map_or(0, |plan| plan.rows),
                ),
                u64::from(dit.transformer.video_channels) * 4,
            ],
            &values[..elements * 4],
        )?;
        rows_files.push(name);
    }
    if !rows_files.is_empty() {
        println!(
            "wrote   {}: the encoder's unprojected rows, {} x {} FP32",
            rows_files.join(", "),
            plans.iter().map(|plan| plan.rows).max().unwrap_or(0),
            u64::from(dit.transformer.video_channels) * 4
        );
    }
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
    // Every further condition's own input and draw, beside the first piece's: a comparison of
    // `fl2va` needs both anchors and a `ref2va` reference needs its own pixels, and a harness that
    // wrote only piece 0 would leave the second condition without an input to compare against.
    for (index, (plan, buffer)) in plans.iter().zip(&buffers).enumerate().skip(1) {
        if let (Some(extent), Some(_)) = (plan.pixels, buffer.pixels) {
            write_h3t(
                &directory.join(format!("pixels.{index}.h3t")),
                H3T_F32,
                &[
                    channels,
                    u64::from(extent.frames),
                    u64::from(extent.height),
                    u64::from(extent.width),
                ],
                &buffer.pixels_host,
            )?;
        }
        if let (Some(extent), Some(view)) = (plan.draw, buffer.draw) {
            let values = download(&stream, view)?;
            write_h3t(
                &directory.join(format!("condition_noise.{index}.h3t")),
                H3T_F32,
                &[
                    u64::from(dit.transformer.video_channels),
                    u64::from(extent.frames),
                    u64::from(extent.height),
                    u64::from(extent.width),
                ],
                &values,
            )?;
        }
        println!(
            "condition {index} ({}): {} visual rows, {} audio rows, {} samples{}",
            match plan.role {
                MediaRole::FirstFrame => "first frame",
                MediaRole::LastFrame => "last frame",
                MediaRole::Reference => "reference",
            },
            plan.rows,
            plan.audio_rows,
            plan.samples.unwrap_or(0),
            if plan.pixels.is_some() {
                ""
            } else {
                ", audio only"
            }
        );
    }
    // The block-by-block diagnosis (`specs/worker.md` 10): the **candidate's half** of a comparison
    // whose reference half is `tools/dump_*.py`. Each file is a tensor this stage already produced,
    // in the container the python dumpers write, under the block name the protocol's table gives - a
    // shape, a maximum difference and the index of the worst element pair; never a pass or a fail.
    if let Some(diag) = &args.diag {
        let block = diag.join("latent_encoder");
        // The page below writes a `case.json` into the same block directory, so naming the
        // diagnosis' tree as the stage's own page would leave one of the two files silently
        // overwritten - and the one that survived would carry the other's provenance.
        if block == directory {
            return Err(format!(
                "`--diag {}` is this stage's own page directory ({}): the diagnosis and the page \
                 write the same case.json there, so pass a separate tree",
                diag.display(),
                directory.display()
            ));
        }
        std::fs::create_dir_all(&block).map_err(|error| format!("{}: {error}", block.display()))?;
        minimax_h3::diagnosis::write_npy(
            &block.join("pixels.npy"),
            &[
                channels,
                u64::from(pixels.frames),
                u64::from(pixels.height),
                u64::from(pixels.width),
            ],
            &pixels_host,
        )?;
        minimax_h3::diagnosis::write_npy(
            &block.join("condition_noise.npy"),
            &[
                u64::from(dit.transformer.video_channels),
                u64::from(latent.frames),
                u64::from(latent.height),
                u64::from(latent.width),
            ],
            &draw,
        )?;
        let rows_bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        minimax_h3::diagnosis::write_npy(
            &block.join("condition_rows.npy"),
            &[rows, hidden],
            &rows_bytes,
        )?;
        // A `.npy` carries no provenance, so the page beside it does: which commit computed these
        // numbers and which checkpoint they were read from, or a reader cannot date one of them.
        let page = format!(
            "{{\n  \"block\": \"latent_encoder\",\n  \"candidate\": \"minimax-h3\",\n  \
             \"commit\": \"{}\",\n  \"checkpoint\": \"{}\",\n  \"dtype\": \"{dtype}\",\n  \
             \"task\": \"{}\",\n  \"files\": [\"pixels.npy\", \"condition_noise.npy\", \
             \"condition_rows.npy\"],\n  \"shapes\": {{\"pixels\": [{}, {}, {}, {}], \
             \"condition_noise\": [{}, {}, {}, {}], \"condition_rows\": [{}, {}]}},\n  \
             \"reference\": \"tools/dump_encoder.py and tools/dump_vae_blocks.py, --out <dir>\",\n  \
             \"note\": \"a conditioned request's draw is made inside the reference's runner, so the \
             reference half of condition_noise.npy is not producible on the host\"\n}}\n",
            args.commit.as_deref().unwrap_or("unstated"),
            args.root.display(),
            task_name(args.task),
            channels,
            u64::from(pixels.frames),
            u64::from(pixels.height),
            u64::from(pixels.width),
            u64::from(dit.transformer.video_channels),
            u64::from(latent.frames),
            u64::from(latent.height),
            u64::from(latent.width),
            rows,
            hidden
        );
        std::fs::write(block.join("case.json"), page)
            .map_err(|error| format!("diag case.json: {error}"))?;
        println!(
            "diagnosis: {} written at commit {}",
            block.display(),
            args.commit.as_deref().unwrap_or("unstated")
        );
    }
    // What this run conditioned on, in request order: the roles the anchors gave each piece and the
    // rows the packing reserved for it. A `case.json` that named only the count could not tell a
    // two-anchor run from a reference one.
    let conditions = plans
        .iter()
        .map(|plan| {
            format!(
                "{}:{}",
                match plan.role {
                    MediaRole::FirstFrame => "first",
                    MediaRole::LastFrame => "last",
                    MediaRole::Reference => "reference",
                },
                plan.rows + plan.audio_rows
            )
        })
        .collect::<Vec<String>>()
        .join(",");
    // The encoder's unprojected rows, named rather than guessed: the reference's
    // `--probe encode_rows` reads `rows.h3t` beside the pixels, and a request of several
    // visual conditions writes one file each.
    let encoder_rows_files = rows_files.join(",");
    let case = format!(
        "{{\n  \"stage\": \"encode\",\n  \"task\": \"{}\",\n  \"canvas\": [{}, {}],\n  \
         \"frames\": {},\n  \"image\": {},\n  \"prompt_tokens\": {},\n  \"seed\": {},\n  \
         \"dtype\": \"{}\",\n  \"text_rows\": {},\n  \"condition_rows\": {},\n  \
         \"prefix_rows\": {},\n  \"hidden\": {},\n  \"pieces\": {},\n  \
         \"conditions\": \"{}\",\n  \"piece_rows\": [{}, {}],\n  \
         \"pixels_file\": \"pixels.h3t\",\n  \
         \"pixels_shape\": [{}, {}, {}, {}],\n  \
         \"pixels\": \"deterministic per-channel ramp, stated\",\n  \
         \"text\": \"deterministic pattern, stated (stage 4 owns the text encoder)\",\n  \
         \"rows_file\": \"condition_rows.h3t\",\n  \"noise_file\": \"condition_noise.h3t\",\n  \
         \"encoder_rows_file\": \"{encoder_rows_files}\",\n  \
         \"encoder_rows\": \"the video encoder's patch rows before the anchor and the projection, FP32\",\n  \
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
        conditions,
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
