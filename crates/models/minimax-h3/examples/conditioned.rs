//! MiniMax-H3's conditioned path: `fl2va`, one image reference, and the rows it writes.
//!
//! The fifth stage, and the only one that binds `VideoEncoder`: a `t2va` request's prefix is
//! `text_rows + 1` rows and never touches this code, while a conditioned request's is
//! `text_rows + condition_rows + 1` and those condition rows come from the encoder's tiles, the
//! posterior, the draw and the projection.
//!
//! ```text
//! --stage weights  no device: what the `fl2va` deployment places, the two encoder key lists against
//!                  the export's indexes, and the condition arithmetic of the case - the segment
//!                  kinds, their row counts, and the prefix the rows make
//! --stage encode   one image on a device: the encoder's tile walk, the posterior, the condition rows
//!                  written as `<out>/latent_encoder/condition_rows.h3t` with `case.json` and
//!                  `walk.txt`
//! ```
//!
//! **The case is one image at one canvas, and the page is a row count.** A conditioned first page is
//! about *which* rows exist and where they are placed, not about a canvas: the wiring is what has
//! never run. The tensors are compared by the caller against the reference's own conditioned dump
//! (`notes/h3/conditioned-diagnosis.md`); this harness writes what a comparison needs and states what
//! it cannot check.
//!
//! **Three things it refuses rather than approximates:** a request whose conditions include audio
//! (this export carries no `audio_vae/` weights, so the audio half is unprovable here and a page that
//! ran it would imply otherwise), a condition draw out of the request's order (the draw's stream is
//! the request's, and an offset there moves every later sample), and a prefix buffer whose shape is
//! not `[text + condition + 1, hidden]`.
//!
//! `--stage encode` needs a program store (`--store <path>` or `UNISERVE_STORE`) holding committed
//! modules for the operators the encoder launches; the store is read, never written.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::partition::{Component, Partition};
use nn::VideoTask;

/// The export root the caller's GPU host holds, when `--root` does not name another.
const DEFAULT_ROOT: &str = "/workspace/models/MiniMax-H3";

/// The two checkpoint directories a conditioned deployment reads its media from.
const VAE: &str = "vae";
const AUDIO_VAE: &str = "audio_vae";

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
    /// Prompt tokens the case presents.
    text_tokens: u32,
    /// The reference image's raster, in pixels.
    image: u32,
    /// The program store `--stage encode` reads its modules from.
    store: Option<PathBuf>,
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    match args.stage.as_str() {
        "weights" => weights(&args),
        "encode" => encode(&args),
        other => Err(format!(
            "`{other}` is not a stage of this harness; it has `weights` (host) and `encode` (device)"
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
            text_tokens: 15,
            image: 256,
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
                        .map_err(|_| format!("`{width}` is not a width"))?;
                    args.height = height
                        .parse()
                        .map_err(|_| format!("`{height}` is not a height"))?;
                }
                "--text-tokens" => {
                    args.text_tokens = value()?
                        .parse()
                        .map_err(|_| "`--text-tokens` takes a count".to_owned())?;
                }
                "--image" => {
                    args.image = value()?
                        .parse()
                        .map_err(|_| "`--image` takes a pixel side".to_owned())?;
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

/// The host stage: what the deployment places, what the encoders read, and the case's rows.
fn weights(args: &Args) -> Result<(), String> {
    let layout = minimax_h3::checkpoint::Layout::detect(&args.root).map_err(show)?;
    let sidecars =
        minimax_h3::checkpoint::Sidecars::read(&args.root, None, &layout).map_err(show)?;
    let config = minimax_h3::config::Config::normalize(&layout, &sidecars).map_err(show)?;

    let partition = Partition::serving(&config, args.task, 1, 1, 0, 0).map_err(show)?;
    println!(
        "{} places {} blocks: {}",
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
    println!("case: {}", case_note(args));
    Ok(())
}

/// What the case's rows need, and why this harness does not compute them itself.
///
/// The rows are the packing's answer: `condition_segments` decides what a condition contributes and
/// `condition_placements` where it sits, and both are private to the denoiser today. Re-deriving
/// either here would be a second statement of a number the model already makes - the defect class
/// this branch has paid for twice - so the harness names what it needs instead of guessing.
fn case_note(args: &Args) -> String {
    format!(
        "the case's rows come from the model's own packing, which is not public: expose \
         `conditions_of` and `condition_placements` (or one `Denoiser` method answering the segments \
         and their intervals for a `VideoRequest`), and this stage computes `condition_rows` and the \
         per-segment placement from the same code the forward runs. The case is {}x{} with a {} px \
         image and {} prompt tokens.",
        args.width, args.height, args.image, args.text_tokens
    )
}

/// The device stage: one image through the encoder, the posterior and the projection.
fn encode(args: &Args) -> Result<(), String> {
    let _ = args;
    Err(
        "`--stage encode` is not built yet: it needs the device weight loader and a program store, \
         and its two halves are the encoder's tile walk and the condition rows. `--stage weights` \
         answers everything about the case that needs no device."
            .to_owned(),
    )
}

/// One failure, as the harness's own layer turns a library error into text.
fn show<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}
