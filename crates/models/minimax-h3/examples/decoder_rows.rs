//! The video decoder's enumeration, accounted for row by row against the model's own forward.
//!
//! ```text
//! cargo run -p minimax-h3 --example decoder_rows -- \
//!     --root /opt/uv/models/MiniMax-H3 \
//!     --extents artifacts/h3-reference/component-tensors.json
//! ```
//!
//! ## Why this exists
//!
//! `VideoDecoder::programs` states thirty-five rows and a decode launches ten. The other twenty-five
//! are not an over-statement and not a defect: the component's decode is a decoder **and a clip
//! stitcher**, the stitcher owns one unpatchify and one compiled blend per seam width and axis, and a
//! single tile's decode launches neither. The distinction matters because the three instruments that
//! certify a store see only a list: a row with a plausible owner and a row nothing launches look the
//! same to `keys_delta`, `key_coverage` and `build_store`.
//!
//! So this file splits the enumeration into the families it is made of, drives the decoder's own
//! `decode` for the first, reads the stitcher's declared blends for the second, and prints anything
//! left over - which is the only shape in which an unattributed row would appear.
//!
//! ## What it drives
//!
//! The decoder is built from weight extents alone (`WeightLoader::stated` reads no byte) and
//! `Forward::recording` reports each launch to a capture sink, so a canvas's tile decodes with no
//! device and no checkpoint shard. The canvas is the tile the decoder is loaded with, exactly as
//! `load` hands it: one tile, not the clip, because the scratch is sized for a tile's rows.

use std::collections::BTreeMap;
use std::path::PathBuf;

use minimax_h3::AttentionKind;
use minimax_h3::decoding::ClipDecoder;
use minimax_h3::video_vae::{Canvas, Tiles, VideoDecoder};
use nn::forward::Arena;
use nn::layers::{Linear, WeightOrder};
use nn::tensor::DevicePtr;
use nn::weights::{WeightLoader, WeightRequest};
use nn::{DType, Shape};
use ops::capture::{self, Sink};

const USAGE: &str = "\
usage: decoder_rows --root <checkpoint> --extents <table> [--canvas HxW ...]

The video decoder's decode is driven at each canvas's own tile and its rows are compared with
`VideoDecoder::programs`; the rows that belong to the clip stitcher are named rather than counted
against the decoder. Exits non-zero when a launch is missing from the enumeration.";

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
    let mut root = PathBuf::from("/opt/uv/models/MiniMax-H3");
    let mut extents_path = None;
    let mut canvases: Vec<(u32, u32)> = Vec::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--root" => root = PathBuf::from(expect(&mut arguments, "--root")?),
            "--extents" => extents_path = Some(PathBuf::from(expect(&mut arguments, "--extents")?)),
            "--canvas" => canvases.push(canvas(&expect(&mut arguments, "--canvas")?)?),
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
    if canvases.is_empty() {
        canvases = minimax_h3::config::offered_canvases()
            .iter()
            .map(|offered| (offered.height, offered.width))
            .collect();
    }

    let config = video_config(&root)?;
    let extents = extents(&root, extents_path)?;
    let stated = VideoDecoder::programs(&config, DType::F32).map_err(|error| error.to_string())?;
    let ratio = config.spatial_compression();
    let model_width = config.decoder_num_attention_heads * config.decoder_attention_head_dim;

    println!(
        "# the video decoder's {} stated rows, split by the object that launches them",
        stated.len()
    );
    println!("# the VAE is F32 by the checkpoint's own storage");

    let mut union: Vec<(String, String)> = Vec::new();
    // One arena per canvas: the arena never rewinds, and the decoder's reservations are the pair this
    // lane has had to reason about twice.
    for (height, width) in canvases {
        let arena = Arena::new(DevicePtr::new(0), 1u64 << 42);
        let f = nn::Forward::recording(&arena);
        let mut loader = WeightLoader::stated(DType::F32, Some(&extents));
        let clip_canvas = Canvas::clip(&config, height / ratio, width / ratio);
        let tiles = Tiles::of(&config, &clip_canvas).map_err(|error| error.to_string())?;
        let canvas = tiles
            .tiles(&config, &clip_canvas)
            .first()
            .copied()
            .ok_or("a clip with no tiles to decode")?
            .canvas;
        let projection = loader
            .load(&WeightRequest::region(
                "virtual.proj_in.weight",
                0,
                Shape::new(&[u64::from(model_width), u64::from(config.latent_channels)]),
            ))
            .map_err(|error| error.to_string())?;
        let bias = loader
            .load(&WeightRequest::region(
                "virtual.proj_in.bias",
                0,
                Shape::new(&[u64::from(model_width)]),
            ))
            .map_err(|error| error.to_string())?;
        let input = Linear::new(projection, Some(bias), WeightOrder::NxK)
            .map_err(|error| error.to_string())?;
        let decoder = VideoDecoder::load(
            &config,
            canvas,
            AttentionKind::Dense,
            DType::F32,
            input,
            &mut loader,
        )
        .map_err(|error| error.to_string())?;

        let scratch = decoder.scratch(&f).map_err(|error| error.to_string())?;
        let latents = f
            .arena()
            .tensor(
                DType::F32,
                Shape::new(&[canvas.voxels(), u64::from(config.latent_channels)]),
            )
            .map_err(|error| error.to_string())?;
        // The positions table is FP32 by the layout's declaration, not the sample's type.
        let positions = f
            .arena()
            .tensor(DType::F32, Shape::new(&[3, canvas.rows()]))
            .map_err(|error| error.to_string())?;
        let out_channels = u64::from(config.out_channels)
            * u64::from(config.temporal_compression())
            * u64::from(config.spatial_compression())
            * u64::from(config.spatial_compression());
        let out = f
            .arena()
            .tensor(DType::F32, Shape::new(&[canvas.rows(), out_channels]))
            .map_err(|error| error.to_string())?;
        let mut recorded = Recorded::default();
        capture::set_sink(Some(&mut recorded));
        let run = decoder.decode(&f, latents, positions, out, &scratch);
        capture::set_sink(None);
        run.map_err(|error| format!("{height}x{width}: {error}"))?;
        let rows = distinct(&recorded);
        println!(
            "{height}x{width}: one tile of {} rows launches {} distinct rows",
            canvas.rows(),
            rows.len()
        );
        for row in rows {
            if !union.contains(&row) {
                union.push(row);
            }
        }
    }

    // The stitcher: one compiled blend per seam width and axis, and the unpatchify it places tiles with.
    let stitcher = ClipDecoder::new(&config, DType::F32).map_err(|error| error.to_string())?;
    let mut blends: Vec<(String, String)> = stitcher
        .blends()
        .iter()
        .map(|(_, _, layer)| ("cross_fade".to_owned(), layer.descriptor().to_string()))
        .collect();
    blends.sort();
    blends.dedup();
    // The stitcher's other row: the unpatchify it places every decoded tile through, derived the way
    // the decoder's enumeration derives it.
    let unpatchify = nn::layers::UnpatchifyVideo::new(
        DType::F32,
        u64::from(config.out_channels),
        config.temporal_compression(),
        config.spatial_compression().max(1),
    )
    .map_err(|error| error.to_string())?
    .descriptor()
    .to_string();
    let mut stitcher: Vec<(String, String)> = blends.clone();
    stitcher.push(("unpatchify_video".to_owned(), unpatchify));
    // The stitch's third family: each seam's neighbour is copied into one contiguous block before the
    // blend reads it, because `cross_fade` addresses its sources flat and a neighbour in a raster wider
    // than the tile is not one contiguous block. One copy serves every seam of every canvas.
    stitcher.push((
        "copy_block".to_owned(),
        nn::layers::CopyBlock::descriptor_of(DType::F32).to_string(),
    ));
    stitcher.sort();
    println!(
        "# the stitcher declares {} distinct blend row(s), one unpatchify and one copy",
        blends.len()
    );

    union.sort();
    let mut missing: Vec<&(String, String)> =
        stitcher.iter().filter(|r| !stated.contains(r)).collect();
    let mut unattributed: Vec<(String, String)> = stated
        .iter()
        .filter(|row| !union.contains(row) && !stitcher.contains(row))
        .cloned()
        .collect();
    let mut unstated: Vec<&(String, String)> =
        union.iter().filter(|r| !stated.contains(r)).collect();
    missing.sort();
    unattributed.sort();
    unstated.sort();

    println!(
        "# {} row(s) launched by a tile's decode and stated; {} belong to the stitcher and are named below",
        union.len().min(stated.len()),
        stated.len().saturating_sub(union.len())
    );
    for row in &unattributed {
        println!(
            "    STATED AND OWNED BY NO OBJECT NAMED HERE\t{}\t{}",
            row.0, row.1
        );
    }
    for row in &missing {
        println!(
            "    STITCHER RESOLVES A BLEND THE DECODER DOES NOT STATE\t{}",
            row.1
        );
    }
    for row in &unstated {
        println!("    LAUNCHED NOT STATED\t{}\t{}", row.0, row.1);
    }

    if unstated.is_empty() && unattributed.is_empty() && missing.is_empty() {
        println!(
            "# every stated row is launched by the decoder or declared by the stitcher, and no launch is unstated."
        );
        Ok(())
    } else {
        Err(format!(
            "{} launch(es) unstated, {} row(s) owned by no named object, {} blend(s) the stitcher resolves and the decoder does not state",
            unstated.len(),
            unattributed.len(),
            missing.len()
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

fn extents(
    root: &std::path::Path,
    given: Option<PathBuf>,
) -> Result<BTreeMap<String, (DType, Shape)>, String> {
    let path = given.ok_or_else(|| {
        format!(
            "--extents is required: the decoder's weight-normalized convolutions fold at load and their shapes are the checkpoint's. The component's config is {}",
            root.join("vae/config.json").display()
        )
    })?;
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
