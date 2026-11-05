//! A model's programs, recorded from the model's own `forward`.
//!
//! ```text
//! cargo run -p minimax-h3 --example record -- --root /opt/uv/models/MiniMax-H3 --component audio-encoder
//! ```
//!
//! ## Why this exists beside `examples/programs.rs`
//!
//! `programs.rs` enumerates a model's programs in a traversal written by hand beside each `forward`, and on
//! one day that traversal was wrong twice and incomplete once (`silu`'s element type, `act_and_mul`'s packed
//! stride, and four `scaled_add` crossings of the sample/FP32 boundary that no row stated). Each was found by
//! comparing it with a second implementation, none by a test, and the missing four were invisible to
//! `key_coverage` — a row neither the fixture nor the store holds answers "every key is covered".
//!
//! This drives the model's **own** code: `WeightLoader::shapes` states every weight's extents without reading
//! a byte, `Forward::recording` reports each launch to the capture sink instead of launching it, and the
//! component's `forward` runs unchanged. No shard, no device. So the row set is what the model launches, by
//! construction, and the hand-written walk becomes the thing being checked rather than the source.
//!
//! ## What a harness must supply
//!
//! Shape-correct operands, which is exactly the model's interface: the component's `forward` takes views, so
//! the harness makes them from the config and the request's own numbers. The comparison is the point —
//! `--against <walk print>` reports the rows recorded and not stated, and the rows stated and not recorded,
//! in that order of severity.
//!
//! ## What the first harness found, and what it needs next
//!
//! The audio VAE's encoder loads until its first **weight-normalized** convolution, then the shapes-only
//! loader refuses: `weight_norm` reduces `v · g / ‖v‖` and the extents of `v` are a fact of the checkpoint's
//! header, which a metadata-only deployment does not have. Every weight elsewhere is stated by the module that
//! reads it, so the fix is not a loader trick: **the model's load states the extents a weight-normalized
//! convolution expects**, from the configuration it already parses (`[out, in / groups, kernel]`), and the live
//! loader then checks the checkpoint's shape against that declaration instead of deriving it. `Binder` already
//! names every parameter, so it is a table change and a checker, and it strengthens the live load in the same
//! stroke. Until it lands, this harness covers the components whose weights are stored plainly, and refuses the
//! rest by name rather than reporting a row set that reads as served.

use std::process::ExitCode;

use minimax_h3::encoding::AudioEncoder;
use nn::forward::Arena;
use nn::tensor::DevicePtr;
use nn::weights::WeightLoader;
use nn::{DType, Forward, Shape};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(problem) => {
            eprintln!("error: {problem}");
            ExitCode::FAILURE
        }
    }
}

/// The rows a recording pass collected, in the order the model launched them.
#[derive(Default)]
struct Recorded(Vec<(String, String)>);

impl ops::capture::Sink for Recorded {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator.to_owned(), key.to_owned()));
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut root = std::path::PathBuf::from("/opt/uv/models/MiniMax-H3");
    let mut component = String::from("audio-encoder");
    let mut against: Option<std::path::PathBuf> = None;
    let mut extents: Option<std::path::PathBuf> = None;
    let mut list = false;
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next().ok_or_else(|| format!("{argument} needs a value"));
        match argument.as_str() {
            "--root" => root = std::path::PathBuf::from(value()?),
            "--component" => component = value()?,
            "--against" => against = Some(std::path::PathBuf::from(value()?)),
            "--extents" => extents = Some(std::path::PathBuf::from(value()?)),
            "--list" => list = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }

    let table = match extents.as_deref() {
        Some(path) => read_extents(path)?,
        None => std::collections::BTreeMap::new(),
    };
    if list {
        return list_extents(&root, &component);
    }

    let rows = match component.as_str() {
        "audio-encoder" => audio_encoder(&root, &table)?,
        other => return Err(format!("unknown component {other}")),
    };

    if let Some(walk) = against {
        let stated = std::fs::read_to_string(&walk)
            .map_err(|error| format!("{}: {error}", walk.display()))?;
        let stated: std::collections::BTreeSet<(String, String)> = stated
            .lines()
            .filter(|line| !line.starts_with('#') && line.contains('\t'))
            .map(|line| {
                let (operator, key) = line.split_once('\t').unwrap_or((line, ""));
                (operator.to_owned(), key.to_owned())
            })
            .collect();
        let recorded: std::collections::BTreeSet<(String, String)> =
            rows.iter().cloned().collect();
        let only_recorded: Vec<_> = recorded.difference(&stated).collect();
        let only_stated: Vec<_> = stated.difference(&recorded).collect();
        for (operator, key) in &only_recorded {
            println!("recorded, not stated:\t{operator}\t{key}");
        }
        for (operator, key) in &only_stated {
            println!("stated, not recorded:\t{operator}\t{key}");
        }
        eprintln!(
            "# {} recorded, {} stated, {} only recorded, {} only stated",
            recorded.len(),
            stated.len(),
            only_recorded.len(),
            only_stated.len()
        );
        return Ok(());
    }

    let mut rows = rows;
    rows.sort();
    rows.dedup();
    let mut operators: Vec<&str> = rows.iter().map(|(name, _)| name.as_str()).collect();
    operators.sort_unstable();
    operators.dedup();
    for (operator, key) in &rows {
        println!("{operator}\t{key}");
    }
    eprintln!("# {} data rows, {} operators", rows.len(), operators.len());
    Ok(())
}

/// The audio VAE's encoder, as one request: a track of whole latent frames in FP32.
///
/// The component is the first harness because its `forward` takes two views and nothing else — the track and
/// the rows it writes — so the operand construction that a harness owes is visible here in full.
fn audio_encoder(
    root: &std::path::Path,
    extents: &std::collections::BTreeMap<String, (DType, Shape)>,
) -> Result<Vec<(String, String)>, String> {
    let path = root.join("audio_vae/config.json");
    let values: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?,
    )
    .map_err(|error| format!("{}: {error}", path.display()))?;
    let config = minimax_h3::AudioVaeConfig::parse(&values).map_err(|error| error.to_string())?;

    // The shapes-only loader: no checkpoint, no shard, no device. A load that reads bytes refuses by name.
    let mut loader = WeightLoader::stated(DType::F32, Some(extents));
    let encoder = AudioEncoder::load(&config, &mut loader).map_err(|error| error.to_string())?;

    // The arena is a null-backed region of a size no reservation will reach; nothing reads it.
    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);

    let hop = encoder.hop();
    let length = hop * 64;
    let samples = arena
        .tensor(DType::F32, Shape::new(&[1, 1, length]))
        .map_err(|error| error.to_string())?;
    let frames = encoder.frames(length).map_err(|error| error.to_string())?;
    let channels = u64::from(config.latent_channels);
    let out = arena
        .tensor(DType::F32, Shape::new(&[frames, channels]))
        .map_err(|error| error.to_string())?;

    let mut recorded = Recorded::default();
    ops::capture::set_sink(Some(&mut recorded));
    let run = encoder.encode(&forward, samples, out);
    ops::capture::set_sink(None);
    run.map_err(|error| error.to_string())?;
    Ok(recorded.0)
}

/// The checkpoint tensors' extents, as the walk states them: `{"name": {"dtype": "F32", "shape": [...]}}`.
///
/// This is the checkpoint's header content, not a model fact: it is produced where the weights are
/// (`--discover` names every tensor a load asks for) and used where they are not.
fn read_extents(
    path: &std::path::Path,
) -> Result<std::collections::BTreeMap<String, (DType, Shape)>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let object = values
        .as_object()
        .ok_or_else(|| format!("{} is not an object", path.display()))?;
    let mut extents = std::collections::BTreeMap::new();
    for (name, entry) in object {
        let dtype = entry
            .get("dtype")
            .and_then(|value| value.as_str())
            .and_then(dtype_of)
            .ok_or_else(|| format!("{name}: the entry states no element type"))?;
        let dims = entry
            .get("shape")
            .and_then(|value| value.as_array())
            .ok_or_else(|| format!("{name}: the entry states no shape"))?
            .iter()
            .map(|dim| dim.as_u64().ok_or_else(|| format!("{name}: a dimension is not a count")))
            .collect::<Result<Vec<u64>, String>>()?;
        extents.insert(name.clone(), (dtype, Shape::new(&dims)));
    }
    Ok(extents)
}

fn dtype_of(name: &str) -> Option<DType> {
    match name {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::BF16),
        "I32" => Some(DType::I32),
        _ => None,
    }
}

/// Print the checkpoint tensors this component reads, as a skeleton of the extents table.
///
/// The names are the model's own (`AudioEncoder::keys` is the checkpoint signature the load binds
/// against), so the skeleton cannot drift from what the load asks for. The extents are the
/// checkpoint's, filled where the weights are: `safetensors` headers state each tensor's element type
/// and shape, and a metadata-only host has neither.
fn list_extents(root: &std::path::Path, component: &str) -> Result<(), String> {
    let path = root.join("audio_vae/config.json");
    let values: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?,
    )
    .map_err(|error| format!("{}: {error}", path.display()))?;
    let config = minimax_h3::AudioVaeConfig::parse(&values).map_err(|error| error.to_string())?;
    let names = match component {
        "audio-encoder" => AudioEncoder::keys(&config),
        other => return Err(format!("unknown component {other}")),
    }
    .map_err(|error| error.to_string())?;
    for name in &names {
        println!("{name:?}: {{\"dtype\": \"F32\", \"shape\": []}}");
    }
    eprintln!("# {} tensor extents the load binds against", names.len());
    Ok(())
}
