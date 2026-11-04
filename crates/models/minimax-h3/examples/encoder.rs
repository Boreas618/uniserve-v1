//! Run the encoder stage: the text encoder's own output for one request.
//!
//!     cargo run -p minimax-h3 --example encoder -- --stage weights --root <export>
//!     cargo run -p minimax-h3 --example encoder -- --stage encode --root <export> \
//!         --device 0 --store <store> --out artifacts/h3-stage4
//!
//! A served request's first arrow is prompt -> text encoder -> the denoiser's conditioner, and this
//! is the only stage that exercises it. `--stage weights` is the host half: the language stack's and
//! the tower's names against the export's index, then the real load into a store that checks every
//! element type and shape as it is handed over. `--stage encode` is the device half: one prompt on
//! one rank, decoded to the rows the reference probe (`tools/dump_encoder.py --probe prompt_hidden`)
//! reads back for the same case.
//!
//! The case carries the prompt twice on purpose: `prompt` is the text and `prompt_ids` the ids a
//! front end tokenized it into. The crates have no tokenizer, so ids are data; the string is there
//! so the reference probe can tokenize it with the checkpoint's own tokenizer and refuse a
//! disagreement at the first index where the two lists part, instead of hiding it inside every row.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nn::Checkpoint;
use nn::weights::{HostStore, WeightLoader};

use minimax_h3::config::Config;
use minimax_h3::partition::Partition;
use minimax_h3::text_encoder::TextEncoder;
use minimax_h3::vision::VisionTower;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Failure> {
    let args = Args::parse()?;
    match args.stage {
        Stage::Weights => stage_weights(&args),
        Stage::Encode => stage_encode(&args),
    }
}

/// One failure line, as the series' examples report it.
#[derive(Debug)]
struct Failure(String);

impl std::fmt::Display for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<minimax_h3::Error> for Failure {
    fn from(error: minimax_h3::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<nn::Error> for Failure {
    fn from(error: nn::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Weights,
    Encode,
}

struct Args {
    stage: Stage,
    root: PathBuf,
    device: u32,
    store: Option<PathBuf>,
    out: PathBuf,
    /// The prompt as text, and the ids a front end tokenized it into.
    prompt: String,
    prompt_ids: Vec<u32>,
}

impl Args {
    fn parse() -> Result<Self, Failure> {
        let mut stage = Stage::Weights;
        let mut root = None;
        let mut device = 0;
        let mut store = None;
        let mut out = PathBuf::from("artifacts/h3-stage4");
        let mut prompt = None;
        let mut prompt_ids = None;
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--stage" => {
                    let value = arguments
                        .next()
                        .ok_or_else(|| Failure("--stage needs `weights` or `encode`".into()))?;
                    stage = match value.as_str() {
                        "weights" => Stage::Weights,
                        "encode" => Stage::Encode,
                        other => {
                            return Err(Failure(format!(
                                "--stage takes `weights` or `encode`, not `{other}`"
                            )));
                        }
                    };
                }
                "--root" => {
                    root = Some(PathBuf::from(
                        arguments
                            .next()
                            .ok_or_else(|| Failure("--root needs a path".into()))?,
                    ));
                }
                "--device" => {
                    device = arguments
                        .next()
                        .ok_or_else(|| Failure("--device needs a number".into()))?
                        .parse()
                        .map_err(|error| Failure(format!("--device: {error}")))?;
                }
                "--store" => {
                    store = Some(PathBuf::from(
                        arguments
                            .next()
                            .ok_or_else(|| Failure("--store needs a path".into()))?,
                    ));
                }
                "--out" => {
                    out = PathBuf::from(
                        arguments
                            .next()
                            .ok_or_else(|| Failure("--out needs a path".into()))?,
                    );
                }
                "--prompt" => {
                    prompt = Some(
                        arguments
                            .next()
                            .ok_or_else(|| Failure("--prompt needs text".into()))?,
                    );
                }
                "--prompt-ids" => {
                    let value = arguments.next().ok_or_else(|| {
                        Failure("--prompt-ids needs a comma-separated list".into())
                    })?;
                    prompt_ids = Some(
                        value
                            .split(',')
                            .map(|field| {
                                field.trim().parse::<u32>().map_err(|error| {
                                    Failure(format!("--prompt-ids: {field:?}: {error}"))
                                })
                            })
                            .collect::<Result<Vec<u32>, Failure>>()?,
                    );
                }
                other => {
                    return Err(Failure(format!(
                        "unknown argument {other}; the flags are --stage, --root, --device, \
                         --store, --out, --prompt and --prompt-ids"
                    )));
                }
            }
        }
        let root = root.ok_or_else(|| Failure("--root is required".into()))?;
        let prompt = prompt.unwrap_or_else(|| "a cat walking on the grass".to_owned());
        let prompt_ids = prompt_ids.unwrap_or_else(|| default_ids(&prompt));
        Ok(Self {
            stage,
            root,
            device,
            store,
            out,
            prompt,
            prompt_ids,
        })
    }
}

/// The ids the checkpoint's tokenizer gives the default prompt.
///
/// A case without `--prompt-ids` still has to say what it encoded, so the default prompt's ids are
/// the checkpoint's own answer rather than an empty list: they are printed with the case and the
/// reference probe cross-checks the string against them.
fn default_ids(_prompt: &str) -> Vec<u32> {
    vec![64, 8251, 11435, 389, 279, 16359]
}

/// The names the export's `text_encoder` index holds.
fn index_names(root: &Path) -> Result<Vec<String>, Failure> {
    let path = root
        .join("text_encoder")
        .join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| Failure(format!("{}: {error}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| Failure(format!("{}: {error}", path.display())))?;
    let map = value
        .get("weight_map")
        .and_then(|map| map.as_object())
        .ok_or_else(|| Failure(format!("{}: no weight_map", path.display())))?;
    Ok(map.keys().cloned().collect())
}

/// Stage 1: the names, then the load, with no device.
///
/// Two checks in one pass, each naming its own failure: the language stack's 551 names and the
/// tower's 351 against the export's index, then the same load the device path performs into a store
/// that validates every tensor's element type and shape as the loader hands it over. A checkpoint
/// that lacks a name is refused by name before a byte is read.
fn stage_weights(args: &Args) -> Result<(), Failure> {
    let config = Config::read(&args.root, None)?;
    let text = &config.text_encoder;
    let language = TextEncoder::keys(text);
    let tower = VisionTower::keys(&text.vision);
    let index = index_names(&args.root)?;

    for (half, asked) in [("language stack", &language), ("vision tower", &tower)] {
        let missing: Vec<&String> = asked.iter().filter(|name| !index.contains(name)).collect();
        println!(
            "names: {half} {} asked, {} in the index, {} missing",
            asked.len(),
            index.len(),
            missing.len()
        );
        for name in &missing {
            println!("  missing {name}");
        }
        if !missing.is_empty() {
            return Err(Failure(format!(
                "the encoders ask {} {half} tensors and {} are not in the export",
                asked.len(),
                missing.len()
            )));
        }
    }

    let checkpoint = Checkpoint::open(&args.root)?;
    let source = checkpoint.component("text_encoder")?;
    let mut store = HostStore::new();
    let mut loader = WeightLoader::new(source, &mut store, None, None);
    let encoder = TextEncoder::load(&mut loader, text, &Partition::single(), nn::DType::BF16)?;
    println!(
        "load: published {} weight regions on the host, {} retained layers",
        store.len(),
        text.num_retained_layers
    );
    drop(encoder);
    Ok(())
}

/// Stage 2: one prompt on one rank, through the device.
///
/// Not written yet. It says what it will do rather than pretending, because a case that those
/// numbers describe is what the reference probe reads back - and a harness that wrote a case without
/// running would make a page look like a disagreement.
fn stage_encode(args: &Args) -> Result<(), Failure> {
    let store = args.store.as_ref().map_or_else(
        || "<no store>".to_owned(),
        |path| path.display().to_string(),
    );
    Err(Failure(format!(
        "stage `encode` is not written yet: it would encode {} ids (\"{}\") on device {}, load \
         the text encoder from the store at {store}, and write {}/text_encoder/prompt_hidden.h3t; \
         `--stage weights` is the half that runs on this host",
        args.prompt_ids.len(),
        args.prompt,
        args.device,
        args.out.display()
    )))
}
