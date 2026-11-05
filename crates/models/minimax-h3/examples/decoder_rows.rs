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
//! `VideoDecoder::programs` states more rows than one tile's decode launches, and the difference is
//! not an over-statement: the component's decode is a decoder **and a clip stitcher**, and the stitcher
//! owns one unpatchify, one compiled blend per seam width and axis, and the copy that makes a seam's
//! neighbour readable - a single tile's decode launches none of them. The numbers are printed rather
//! than stated here, because the blend table is **derived from the canvases the deployment offers**:
//! it holds a width exactly when an offered canvas launches it, so a canvas added to the model adds
//! its seams and this program's output changes without this sentence being edited. (A ladder of
//! alignment steps to the tile size was once written out instead; it stated 24 widths, of which 8 ever
//! launch.)
//!
//! The distinction matters because the three instruments that certify a store see only a list: a row
//! with a plausible owner and a row nothing launches look the same to `keys_delta`, `key_coverage` and
//! `build_store`.
//!
//! So this file splits the enumeration into the families it is made of, drives the decoder's own
//! `decode` for the first, reads the stitcher's declared blends for the second, and prints anything
//! left over - which is the only shape in which an unattributed row would appear.
//!
//! ## What it drives
//!
//! ## The arena the plan states
//!
//! The last thing this program checks is the figure a runner sizes storage from: `video_arena` states
//! a clip's buffers, and a whole clip is decoded in an arena of **exactly** those bytes, at every
//! canvas and at every element type a deployment can name. A decode that does not fit fails there
//! rather than on a device
//! partway through the stitch, and a decode that fits with room to spare means the plan reserves a
//! buffer nothing takes - so the check requires the clip to consume the plan entirely, and the one
//! number it prints per canvas (`takes all N bytes of its 17 buffer plan`) is the reservation a
//! runner may allocate.
//!
//! The decoder is built from weight extents alone (`WeightLoader::stated` reads no byte) and
//! `Forward::recording` reports each launch to a capture sink, so a canvas's tile decodes with no
//! device and no checkpoint shard. A decoder is loaded with the **clip's** canvas, because that is
//! the one thing `load` keeps: `layout_of` and `arena_bytes` answer from it, and a decode reads its
//! extent from the raster it writes. The rows a tile launches are the tile grid's, which this tool
//! builds beside the decoder and drives the stitch with.

use std::collections::BTreeMap;
use std::path::PathBuf;

use minimax_h3::AttentionKind;
use minimax_h3::decoding::{ClipDecoder, arena_reservation, video_arena};
use minimax_h3::video_vae::{Canvas, Tiles, VideoDecoder};
use nn::capability::Decoder;
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
    for &(height, width) in &canvases {
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

    let rows_agree = if unstated.is_empty() && unattributed.is_empty() && missing.is_empty() {
        println!(
            "# every stated row is launched by the decoder or declared by the stitcher, and no launch is unstated."
        );
        true
    } else {
        false
    };

    // The plan is the figure a runner sizes storage from, so it is checked by driving the decode it
    // describes rather than by reading it.
    fit_arenas(&config, &extents, &canvases, model_width)?;

    if rows_agree {
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

/// Drive a whole clip's decode in an arena of **exactly** the bytes `video_arena` states for it.
///
/// The plan is the figure a runner reserves, and until this check nothing drove the path it describes:
/// the tile decode above and the probes all hand their forward an arena far larger than any clip needs,
/// so a plan one buffer short would pass every check in the lane and fail on the first clip on a device
/// - as `Arena { needed, available }` partway through the stitch, naming no buffer.
///
/// Two directions, one measurement. A decode that does not fit fails here; a decode that fits with room
/// to spare means the plan states a buffer nothing reserves, which is device memory a runner would
/// reserve for no reason. The plan is exact when the clip takes every byte of it, at every canvas and
/// every element type, so the allowance is zero rather than one buffer.
fn fit_arenas(
    config: &minimax_h3::VideoVaeConfig,
    extents: &BTreeMap<String, (DType, Shape)>,
    canvases: &[(u32, u32)],
    model_width: u32,
) -> Result<(), String> {
    let ratio = config.spatial_compression();
    let mut failures: Vec<String> = Vec::new();
    // The decoder's own element types. **This export stores the video VAE in FP32** - all 703 tensors
    // (`artifacts/h3-reference/component-tensors.json`, and `deployment_sizes.rs`'s weight rows) - so
    // an `auto` deployment decodes at FP32 and converts nothing. FP16 is a released preset's compute
    // type, and BF16 the calibrated preset's (`uniserve/models/minimax_h3/precision.py:183-190`), so
    // all three plans a deployment can ask for are measured.
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for &(height, width) in canvases {
            let clip_canvas = Canvas::clip(config, height / ratio, width / ratio);
            let tiles = Tiles::of(config, &clip_canvas)
                .map_err(|error| error.to_string())?
                .tiles(config, &clip_canvas)
                .to_vec();
            let mut loader = WeightLoader::stated(dtype, Some(extents));
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
                config,
                clip_canvas,
                AttentionKind::Dense,
                dtype,
                input,
                &mut loader,
            )
            .map_err(|error| error.to_string())?;
            let plan =
                video_arena(config, &clip_canvas, dtype).map_err(|error| error.to_string())?;
            let need = arena_reservation(&plan);

            // The number a runner allocates from, asked of the capability rather than recomputed
            // here: it is compared below against the arena a real decode uses, so the tool fails if
            // the two ever become two statements of different things.
            let stated = decoder
                .arena_bytes(
                    &decoder
                        .layout_of(nn::capability::Clip {
                            frames: clip_canvas.frames_out(config),
                            height,
                            width,
                        })
                        .map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?;

            // The runner's own buffers - the clip's latent rows, the position table and the raster the
            // stitch writes - are not part of a clip's plan, so they live in their own arena.
            let caller = Arena::new(DevicePtr::new(0), 1u64 << 42);
            let voxels: u64 = tiles.iter().map(|tile| tile.canvas.voxels()).sum();
            let latents = caller
                .tensor(
                    dtype,
                    Shape::new(&[voxels, u64::from(config.latent_channels)]),
                )
                .map_err(|error| error.to_string())?;
            let positions = caller
                .tensor(DType::F32, Shape::new(&[3, Canvas::rows(&tiles[0].canvas)]))
                .map_err(|error| error.to_string())?;
            let out = caller
                .tensor(
                    dtype,
                    Shape::new(&[
                        u64::from(minimax_h3::output::Rgb::CHANNELS),
                        u64::from(clip_canvas.frames_out(config)),
                        u64::from(height),
                        u64::from(width),
                    ]),
                )
                .map_err(|error| error.to_string())?;

            // Exactly the plan: not one buffer more.
            let clip_arena = Arena::new(DevicePtr::new(0), need);
            let f = nn::Forward::recording(&clip_arena);
            let stitch = ClipDecoder::new(config, dtype).map_err(|error| error.to_string())?;
            let mut recorded = Recorded::default();
            capture::set_sink(Some(&mut recorded));
            let run = stitch.decode_clip(&f, &decoder, &tiles, latents, positions, out);
            capture::set_sink(None);
            let used = clip_arena.used();
            match run {
                Err(error) => failures.push(format!(
                    "{dtype:?} {height}x{width}: a clip does not fit the {} bytes its plan states: {error}",
                    need
                )),
                Ok(()) if used != need => failures.push(format!(
                    "{dtype:?} {height}x{width}: the clip takes {used} of the {} bytes its {} stated buffers reserve, so {} byte(s) are reserved for nothing",
                    need,
                    plan.len(),
                    need - used
                )),
                Ok(()) if stated != used => failures.push(format!(
                    "{dtype:?} {width}x{height}: the decoder states {stated} B of arena and the \
                     clip takes {used} B"
                )),
                Ok(()) => println!(
                    "{dtype:?} {height}x{width}: a {}-tile clip takes all {need} bytes of its {} buffer plan, \
                     and the decoder states that number",
                    tiles.len(),
                    plan.len()
                ),
            }
        }
    }
    if failures.is_empty() {
        println!("# every canvas's clip fits its plan exactly, at every element type a deployment can name.");
        Ok(())
    } else {
        for failure in &failures {
            println!("    PLAN DOES NOT MATCH THE DECODE\t{failure}");
        }
        Err(format!(
            "{} canvas(es) whose decode does not match the arena its plan states",
            failures.len()
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
