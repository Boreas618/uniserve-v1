//! The video encoder's two drivers, recorded from the model's own `forward`.
//!
//! ```text
//! cargo run -p minimax-h3 --example encoder_rows -- \
//!     --root /opt/uv/models/MiniMax-H3 \
//!     --extents artifacts/h3-reference/component-tensors.json
//! ```
//!
//! ## Why this exists
//!
//! `VideoEncoder::programs` and `ClipEncoder::programs` each state what a driver launches, and until this file
//! existed nothing checked either statement in the direction that matters: every other instrument compares one
//! enumeration with another (`keys_delta` the fixture with the walk, `key_coverage` the keys with the packages),
//! and `record` compares the walk with the whole-raster driver only. **The tiled driver had no witness at all,
//! and the whole raster's statement was checked in one direction only** - a row stated and never launched is
//! invisible to a recorder that reports only what it launched and the walk does not state.
//!
//! That gap was real: the encoder's enumeration paced a pad for every stage, including the stages that have no
//! downsampler and read no padded frame, so it named a class no launch asked for at every canvas the model
//! offers. No fixture, store or coverage check could see it, because each of them is a statement about a list.
//!
//! ## What it does
//!
//! The encoder is built from its weight extents alone - `WeightLoader::stated` reads no byte - and
//! `Forward::recording` reports each launch to a capture sink instead of launching it, so both drivers run
//! unchanged with no device and no checkpoint shard. For each offered canvas and each frame count the file
//! prints both directions:
//!
//! - **launched, not stated** - the enumeration is short by a program a launch asks for. This is the direction
//!   that stops a run, and the run fails on it.
//! - **stated, not launched** - the enumeration names a program no launch of that driver reads. This is the
//!   direction that costs a cell nobody selects, and it is reported and counted, not failed.
//!
//! One row is expected in the second list and it is not a defect: `posterior` is launched by
//! [`VideoEncoder::posterior`], which the conditioning path drives separately, and the encoder's enumeration
//! covers both of the encoder's methods.
//!
//! ## The extents table
//!
//! A weight **computed at load** needs its extents: the folded form of a weight-normalized convolution is
//! `v · g / ‖v‖`, and `v`'s shape is a fact of the checkpoint's header. `--extents` supplies it, either flat
//! (`{"tensor": {"dtype": F32, "shape": [..]}}`) or as the caller's per-component table, whose `vae` entry is
//! the flat form.

use std::collections::BTreeMap;
use std::path::PathBuf;

use minimax_h3::encoding::{Stack, VideoEncoder};
use minimax_h3::tiled_encode::{ClipEncoder, Scene};
use nn::weights::WeightLoader;
use nn::{Arena, DType, Shape};
use ops::capture::{self, Sink};

/// Who the program is.
const USAGE: &str = "\
usage: encoder_rows --root <checkpoint> --extents <table> [--canvas HxW ...] [--frames N ...]

Both drivers of the video encoder are recorded against their own enumeration, at every canvas the
deployment offers unless one is named. Exits non-zero when a launch is missing from the enumeration.";

/// The rows a recording pass collected, in launch order.
#[derive(Default)]
struct Recorded(Vec<(String, String)>);

impl Sink for Recorded {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator.to_owned(), key.to_owned()));
    }
}

/// Distinct rows, in a stable order.
fn distinct(recorded: &Recorded) -> Vec<(String, String)> {
    let mut rows = recorded.0.clone();
    rows.sort();
    rows.dedup();
    rows
}

/// What one driver's recording says about one enumeration.
struct Comparison {
    launched: usize,
    stated: usize,
    launched_only: Vec<String>,
    stated_only: Vec<String>,
}

impl Comparison {
    fn of(stated: &[(String, String)], launched: &[(String, String)]) -> Self {
        Self {
            launched: launched.len(),
            stated: stated.len(),
            launched_only: launched
                .iter()
                .filter(|row| !stated.contains(row))
                .map(|row| format!("{}\t{}", row.0, row.1))
                .collect(),
            stated_only: stated
                .iter()
                .filter(|row| !launched.contains(row))
                .map(|row| format!("{}\t{}", row.0, row.1))
                .collect(),
        }
    }

    fn print(&self, canvas: (u32, u32), frames: u32, driver: &str) {
        println!(
            "{driver} {}x{} frames {frames}: launched {}, stated {}; launched not stated {}, stated not launched {}",
            canvas.0,
            canvas.1,
            self.launched,
            self.stated,
            self.launched_only.len(),
            self.stated_only.len()
        );
        for row in &self.launched_only {
            println!("    LAUNCHED NOT STATED {row}");
        }
        for row in &self.stated_only {
            println!("    STATED NOT LAUNCHED {row}");
        }
    }
}

fn main() {
    if let Err(report) = run() {
        eprintln!("{report}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = PathBuf::from("/opt/uv/models/MiniMax-H3");
    let mut extents_path = None;
    let mut canvases: Vec<(u32, u32)> = Vec::new();
    let mut frames: Vec<u32> = Vec::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--root" => root = PathBuf::from(expect(&mut arguments, "--root")?),
            "--extents" => extents_path = Some(PathBuf::from(expect(&mut arguments, "--extents")?)),
            "--canvas" => canvases.push(canvas(&expect(&mut arguments, "--canvas")?)?),
            "--frames" => {
                let count = expect(&mut arguments, "--frames")?;
                frames.push(
                    count
                        .parse()
                        .map_err(|_| format!("--frames {count} is not a count"))?,
                );
            }
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

    let config = video_config(&root)?;
    let stack = Stack::of(&config).map_err(|error| error.to_string())?;
    let extents = extents(&root, extents_path)?;
    let mut loader = WeightLoader::stated(DType::F32, Some(&extents));
    let encoder = VideoEncoder::load(&config, &mut loader).map_err(|error| error.to_string())?;

    if canvases.is_empty() {
        canvases = minimax_h3::config::offered_canvases()
            .iter()
            .map(|offered| (offered.height, offered.width))
            .collect();
    }
    if frames.is_empty() {
        // A still and a clip: the tile's frame operators key on the count, and a harness that drives one of
        // them certifies half of what the encoder does.
        frames = vec![1, stack.clip.frames];
    }

    println!(
        "# the video encoder's drivers against its enumerations: {} canvas(es), frames {frames:?}",
        canvases.len()
    );
    println!("# the VAE is F32 by the checkpoint's own storage, so every row is keyed f32");

    let mut short = 0usize;
    let mut excess = 0usize;
    for (height, width) in canvases {
        // One arena per canvas: the arena never rewinds, and a driver of a whole canvas reserves most of a
        // gigabyte of views.
        let arena = Arena::new(ops::DevicePtr::new(0), 1u64 << 42);
        let f = nn::Forward::recording(&arena);
        for frames in &frames {
            let frames = *frames;
            let latency = encoder
                .latent_frames(frames)
                .map_err(|error| error.to_string())?;
            let rows = u64::from(height) / 16;
            let columns = u64::from(width) / 16;

            // The whole raster: one tile of the raster's own size.
            let pixels = f
                .arena()
                .tensor(
                    DType::F32,
                    Shape::new(&[1, 3, u64::from(frames), u64::from(height), u64::from(width)]),
                )
                .map_err(|error| error.to_string())?;
            let moments = f
                .arena()
                .tensor(
                    DType::F32,
                    Shape::new(&[
                        2 * u64::from(config.latent_channels),
                        u64::from(latency),
                        rows,
                        columns,
                    ]),
                )
                .map_err(|error| error.to_string())?;
            let mut recorded = Recorded::default();
            capture::set_sink(Some(&mut recorded));
            let whole = encoder.encode(&f, pixels, moments);
            capture::set_sink(None);
            whole.map_err(|error| format!("{height}x{width} frames {frames}: {error}"))?;
            // The sampler is the encoder's other method and it is one row: the raster recording
            // alone reports that row as stated and not launched, which reads as an over-statement.
            let noise = f
                .arena()
                .tensor(
                    DType::F32,
                    Shape::new(&[
                        u64::from(config.latent_channels),
                        u64::from(latency),
                        rows,
                        columns,
                    ]),
                )
                .map_err(|error| error.to_string())?;
            let latent = f
                .arena()
                .tensor(
                    DType::F32,
                    Shape::new(&[
                        u64::from(config.latent_channels),
                        u64::from(latency),
                        rows,
                        columns,
                    ]),
                )
                .map_err(|error| error.to_string())?;
            capture::set_sink(Some(&mut recorded));
            let sampled = encoder.posterior(&f, moments, noise, latent);
            capture::set_sink(None);
            sampled
                .map_err(|error| format!("{height}x{width} frames {frames} sampler: {error}"))?;

            let stated = VideoEncoder::programs(&config, &[(height, width)])
                .map_err(|error| error.to_string())?;
            let comparison = Comparison::of(&stated, &distinct(&recorded));
            comparison.print((height, width), frames, "raster and sampler");
            short += comparison.launched_only.len();
            excess += comparison.stated_only.len();

            // The tiled driver: the same raster cut into the grid the encoder states.
            let clip = ClipEncoder::new(height, width).map_err(|error| error.to_string())?;
            let plan = clip.plan();
            let latent = clip.latent_extent();
            let blended = f
                .arena()
                .tensor(
                    DType::F32,
                    Shape::new(&[
                        2 * u64::from(config.latent_channels),
                        u64::from(frames),
                        u64::from(latent.0),
                        u64::from(latent.1),
                    ]),
                )
                .map_err(|error| error.to_string())?;
            // One packed block per tile, in plan order: a tile cut from a wider raster would carry the
            // raster's row stride, and `FramePad` requires a frame's rows contiguous in the view it reads.
            let blocks: Vec<_> = plan
                .tiles
                .iter()
                .map(|tile| {
                    f.arena()
                        .tensor(
                            DType::F32,
                            Shape::new(&[
                                1,
                                3,
                                u64::from(frames),
                                u64::from(tile.rows.1 * 16),
                                u64::from(tile.columns.1 * 16),
                            ]),
                        )
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<_, String>>()?;
            let mut recorded = Recorded::default();
            capture::set_sink(Some(&mut recorded));
            let tiled = clip.encode(
                &f,
                &encoder,
                &plan,
                Scene {
                    pixels: &blocks,
                    blended,
                },
            );
            capture::set_sink(None);
            tiled.map_err(|error| format!("{height}x{width} frames {frames} tiled: {error}"))?;
            // The tiled enumeration covers the encoder, not only the tile loop: the sampler's row is
            // recorded here too, so the comparison is between two statements of the same extent.
            let noise = f
                .arena()
                .tensor(
                    DType::F32,
                    Shape::new(&[
                        u64::from(config.latent_channels),
                        u64::from(latency),
                        rows,
                        columns,
                    ]),
                )
                .map_err(|error| error.to_string())?;
            let latent = f
                .arena()
                .tensor(
                    DType::F32,
                    Shape::new(&[
                        u64::from(config.latent_channels),
                        u64::from(latency),
                        rows,
                        columns,
                    ]),
                )
                .map_err(|error| error.to_string())?;
            capture::set_sink(Some(&mut recorded));
            let sampled = encoder.posterior(&f, moments, noise, latent);
            capture::set_sink(None);
            sampled
                .map_err(|error| format!("{height}x{width} frames {frames} sampler: {error}"))?;

            let stated = ClipEncoder::programs(&config, &[(height, width)], frames)
                .map_err(|error| error.to_string())?;
            let comparison = Comparison::of(&stated, &distinct(&recorded));
            comparison.print((height, width), frames, "tiled");
            short += comparison.launched_only.len();
        }
    }

    if short == 0 && excess == 0 {
        println!(
            "# every launch is stated and every stated row is launched, at every canvas and frame count."
        );
        Ok(())
    } else {
        Err(format!(
            "{short} row(s) launched and not stated, {excess} stated and not launched: the enumeration and the drivers disagree, and a store built from the enumeration would hold a cell no launch selects or lack one a launch asks for"
        ))
    }
}

fn expect(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} takes a value\n{USAGE}"))
}

fn canvas(text: &str) -> Result<(u32, u32), String> {
    let (height, width) = text
        .split_once('x')
        .ok_or_else(|| format!("--canvas {text} is not <height>x<width>"))?;
    Ok((
        height
            .parse()
            .map_err(|_| format!("--canvas {text}: {height} is not a count"))?,
        width
            .parse()
            .map_err(|_| format!("--canvas {text}: {width} is not a count"))?,
    ))
}

fn video_config(root: &std::path::Path) -> Result<minimax_h3::VideoVaeConfig, String> {
    let path = root.join("vae/config.json");
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    minimax_h3::VideoVaeConfig::parse(&values).map_err(|error| error.to_string())
}

/// The weight extents the load folds a weight-normalized convolution from.
///
/// The table is the caller's when one is given, the checkpoint's own header otherwise; the flat form is what
/// the loader asks for, and the per-component form names it under `vae`.
fn extents(
    root: &std::path::Path,
    given: Option<PathBuf>,
) -> Result<BTreeMap<String, (DType, Shape)>, String> {
    let path = match given {
        Some(path) => path,
        None => {
            return Err(format!(
                "--extents is required: the weight-normalized convolutions are folded at load and their shapes are the checkpoint's, not the config's. On a host with the checkpoint's headers, a table made from them; on this host, {}",
                root.join("vae/config.json").display()
            ));
        }
    };
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let table = values.get("vae").unwrap_or(&values);
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
