//! The audio components, recorded from the model's own `forward`.
//!
//! ```text
//! cargo run -p minimax-h3 --example audio_rows -- \
//!     --root $UNISERVE_H3_MODEL \
//!     --extents artifacts/h3-reference/component-tensors.json
//! ```
//!
//! ## Why this exists beside `record` and `encoder_rows`
//!
//! `record` reports the direction that stops a run - a launch the enumeration is missing. It cannot see
//! the other one: a row the enumeration states and no launch reads. That row is not harmless; a store
//! built from the enumeration compiles a cell for arithmetic no request performs, and every instrument
//! in the lane accepts it, because `keys_delta` compares the fixture with the walk and `key_coverage`
//! asks whether a package declares the key. Both answers are yes for a row nothing launches.
//!
//! The video encoder had such a row (`frame_pad` at a class only a stage without a downsampler could
//! name), found by driving its forwards rather than by reading its enumeration. This file is the same
//! check for the two audio components: `AudioEncoder::programs` against `AudioEncoder::encode`, and
//! `BigVgan::programs` against `BigVgan::forward`, both ways, at more than one length.
//!
//! ## What it does
//!
//! Each component is built from its weight extents alone - `WeightLoader::stated` reads no byte, and
//! `Forward::recording` reports each launch to a capture sink instead of launching it - so the whole
//! check runs with no device and no checkpoint shard. A launch the enumeration misses fails the run; a
//! stated row no launch reads is counted and printed with its key.
//!
//! The lengths are the deployment's own: the encoder's track is padded to a whole number of latent
//! frames by the caller's rule (`padded_samples`), and the decoder's latent frames come from the
//! packing (`audio_latent_frames`), because a clip's video frame count is not the track's.

use std::collections::BTreeMap;
use std::path::PathBuf;

use minimax_h3::audio_vae::BigVgan;
use minimax_h3::encoding::AudioEncoder;
use nn::weights::WeightLoader;
use nn::{Arena, DType, Shape};
use ops::capture::{self, Sink};

const USAGE: &str = "\
usage: audio_rows --root <checkpoint> --extents <table> [--track N ...] [--clip N ...]

Both audio components are recorded against their own enumerations, in both directions, at the
deployment's track and at the clips given. Exits non-zero when a launch is missing from the
enumeration.";

/// The rows a recording pass collected, in launch order.
#[derive(Default)]
struct Recorded(Vec<(String, String)>);

impl Sink for Recorded {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator.to_owned(), key.to_owned()));
    }
}

fn distinct(recorded: &Recorded) -> Vec<(String, String)> {
    let mut rows = recorded.0.clone();
    rows.sort();
    rows.dedup();
    rows
}

fn main() {
    if let Err(report) = run() {
        eprintln!("{report}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = PathBuf::new();
    let mut extents_path = None;
    let mut tracks: Vec<u32> = Vec::new();
    let mut clips: Vec<u32> = Vec::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--root" => root = PathBuf::from(expect(&mut arguments, "--root")?),
            "--extents" => extents_path = Some(PathBuf::from(expect(&mut arguments, "--extents")?)),
            "--track" => tracks.push(count(&expect(&mut arguments, "--track")?, "--track")?),
            "--clip" => clips.push(count(&expect(&mut arguments, "--clip")?, "--clip")?),
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => {
                return Err(format!(
                    "{other} is not an argument of this program\n{USAGE}"
                ));
            }
        }
    }
    if root.as_os_str().is_empty() {
        root = std::env::var_os("UNISERVE_H3_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                eprintln!("name the checkpoint with --root or UNISERVE_H3_MODEL");
                std::process::exit(2)
            });
    }
    if tracks.is_empty() {
        // The delivery's own track length, so a row is certified at the length it is served at.
        tracks = vec![29600];
    }
    if clips.is_empty() {
        // A short clip, the next packing rung, and one longer than a latent window: the decoder's ladder
        // follows the latent frame count, so three of them is a ladder rather than a point.
        clips = vec![22, 39, 256];
    }

    let extents = extents(&root, extents_path)?;
    let config = audio_config(&root)?;

    println!(
        "# the audio components against their enumerations: tracks {tracks:?}, clips {clips:?}"
    );
    println!(
        "# the audio VAE is F32 by the checkpoint's own storage, and both components state F32 rows"
    );

    let mut short = 0usize;
    let mut excess = 0usize;
    for track in tracks {
        let arena = Arena::new(ops::DevicePtr::new(0), 1u64 << 42);
        let f = nn::Forward::recording(&arena);
        let mut loader = WeightLoader::stated(DType::F32, Some(&extents));
        let encoder =
            AudioEncoder::load(&config, &mut loader).map_err(|error| error.to_string())?;
        let length = minimax_h3::encoding::padded_samples(u64::from(track), encoder.hop())
            .map_err(|error| error.to_string())?;
        // The encoder reads the stereo pair the wire carries - two identical rows for a mono track,
        // as the reference's `unpack_samples` writes it - and its latent is `[batch * frames,
        // latent_channels]` rows, channel-major. A single row encodes half a track and every count
        // downstream accepts the result, so the shape is checked here rather than discovered.
        let samples = f
            .arena()
            .tensor(DType::F32, Shape::new(&[2, 1, length]))
            .map_err(|error| error.to_string())?;
        let frames = encoder.frames(length).map_err(|error| error.to_string())?;
        let out = f
            .arena()
            .tensor(
                DType::F32,
                Shape::new(&[2 * frames, u64::from(config.latent_channels)]),
            )
            .map_err(|error| error.to_string())?;
        let mut recorded = Recorded::default();
        capture::set_sink(Some(&mut recorded));
        let run = encoder.encode(&f, samples, out);
        capture::set_sink(None);
        run.map_err(|error| format!("audio encoder track {track}: {error}"))?;
        let stated =
            AudioEncoder::programs(&config, &[track]).map_err(|error| error.to_string())?;
        let (missing, extra) = compare(
            &format!("audio encoder track {track}"),
            &stated,
            &distinct(&recorded),
        );
        short += missing;
        excess += extra;
    }

    for clip in clips {
        let arena = Arena::new(ops::DevicePtr::new(0), 1u64 << 42);
        let f = nn::Forward::recording(&arena);
        let mut loader = WeightLoader::stated(DType::F32, Some(&extents));
        let decoder = BigVgan::load(&config, &mut loader).map_err(|error| error.to_string())?;
        let frames = minimax_h3::packing::audio_latent_frames(clip);
        let latents = f
            .arena()
            .tensor(
                DType::F32,
                Shape::new(&[2, u64::from(config.latent_channels), u64::from(frames)]),
            )
            .map_err(|error| error.to_string())?;
        let samples = minimax_h3::audio_vae::track_samples(&config, frames);
        // The interleave writes the wire format, so the decode's last view is i16 rather than FP32.
        let out = f
            .arena()
            .tensor(DType::I16, Shape::new(&[samples, 2]))
            .map_err(|error| error.to_string())?;
        let mut recorded = Recorded::default();
        capture::set_sink(Some(&mut recorded));
        let run = decoder.forward(&f, latents, out);
        capture::set_sink(None);
        run.map_err(|error| format!("audio decoder clip {clip}: {error}"))?;
        let track = u32::try_from(samples)
            .map_err(|_| format!("clip {clip} decodes to {samples} samples, past a count"))?;
        let stated = BigVgan::programs(&config, &[track]).map_err(|error| error.to_string())?;
        let (missing, extra) = compare(
            &format!("audio decoder clip {clip} ({frames} latent frames, {samples} samples)"),
            &stated,
            &distinct(&recorded),
        );
        short += missing;
        excess += extra;
    }

    if short == 0 && excess == 0 {
        println!("# every launch is stated and every stated row is launched.");
        Ok(())
    } else {
        Err(format!(
            "{short} row(s) launched and not stated, {excess} stated and not launched: the enumeration and the components disagree"
        ))
    }
}

/// Print one component's comparison and answer with `(launched not stated, stated not launched)`.
fn compare(
    what: &str,
    stated: &[(String, String)],
    launched: &[(String, String)],
) -> (usize, usize) {
    let mut missing: Vec<String> = stated
        .iter()
        .filter(|row| !launched.contains(row))
        .map(|row| format!("{}\t{}", row.0, row.1))
        .collect();
    let mut extra: Vec<String> = launched
        .iter()
        .filter(|row| !stated.contains(row))
        .map(|row| format!("{}\t{}", row.0, row.1))
        .collect();
    missing.sort();
    extra.sort();
    println!(
        "{what}: launched {}, stated {}; launched not stated {}, stated not launched {}",
        launched.len(),
        stated.len(),
        extra.len(),
        missing.len()
    );
    for row in &extra {
        println!("    LAUNCHED NOT STATED {row}");
    }
    for row in &missing {
        println!("    STATED NOT LAUNCHED {row}");
    }
    (extra.len(), missing.len())
}

fn expect(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} takes a value\n{USAGE}"))
}

fn count(text: &str, flag: &str) -> Result<u32, String> {
    text.parse()
        .map_err(|_| format!("{flag} {text} is not a count"))
}

fn audio_config(root: &std::path::Path) -> Result<minimax_h3::AudioVaeConfig, String> {
    let path = root.join("audio_vae/config.json");
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    minimax_h3::AudioVaeConfig::parse(&values).map_err(|error| error.to_string())
}

/// The weight extents a weight-normalized convolution folds from.
///
/// The table is either flat (`{"tensor": {"dtype": F32, "shape": [..]}}`) or the caller's
/// per-component one, whose `audio_vae` entry is the flat form.
fn extents(
    root: &std::path::Path,
    given: Option<PathBuf>,
) -> Result<BTreeMap<String, (DType, Shape)>, String> {
    let path = given.ok_or_else(|| {
        format!(
            "--extents is required: the weight-normalized convolutions are folded at load and their shapes are the checkpoint's, not the config's. The component's table is {}",
            root.join("audio_vae/config.json").display()
        )
    })?;
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let table = values.get("audio_vae").unwrap_or(&values);
    let object = table
        .as_object()
        .ok_or_else(|| format!("{} is not an object", path.display()))?;
    let mut extents = BTreeMap::new();
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
            .map(|dim| {
                dim.as_u64()
                    .ok_or_else(|| format!("{name}: a dimension is not a count"))
            })
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
        _ => None,
    }
}
