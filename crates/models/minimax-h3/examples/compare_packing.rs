//! Dump the denoiser's dense row tables for the block-by-block diagnosis (specs/worker.md 10).
//!
//! ```text
//! cargo run -p minimax-h3 --example compare_packing -- --case <case.json> --out <dir>
//! ```
//!
//! **A dense case** writes two tensors under `<dir>/denoiser/`, under the protocol's own names:
//! `packed_rows` (`[rows, 4]` float32, each row's modulation tag followed by its three rotary
//! coordinates) and `cu_seqlens` (`[2, 2]` int32, `{0, padded}` for the queries and `{0, used}` for
//! the keys). `crates/models/minimax-h3/tools/dump_denoiser.py --probe rows` writes the same two
//! names from today's implementation.
//!
//! **A sparse case** writes the segment packing's own tables, which are the reference's
//! `segment.`-prefixed names (`tools/dump_tables.py:139-161`): `position_ids` (`[padded, 3]`),
//! `token_tags`, `groups` and `prefix_index` (`[padded]`), `variable_block_sizes`, `tile_segments`,
//! `segment_starts` and `segment_keep` (`[tiles]`), `video_indices` and `video_raster_indices`
//! (`[video rows]`), and five one-element counts (`audio_rows`, `audio_start`, `video_tiles`,
//! `video_start`, `padded_tokens`). Those tables are what a sparse request's attention reads - the
//! tile sizes, the keep rate's per-segment selection and every row's rotary coordinate - so a defect
//! in the tiling or in the selection shows here as a table that disagrees, without a device.
//!
//! Either way `tools/diagnosis/compare.py --ref <python dir> --got <this dir>` matches them by
//! relative path and reports their difference.
//!
//! No checkpoint and no device: the tables are host arithmetic ([`minimax_h3::packing`]), which is
//! exactly the half a structural mistake shows in first.
//!
//! This is a **diagnosis, never an acceptance gate**: nothing is gated on the values. A run fails
//! only when the case cannot be read, when a sparse case omits its tile or its sparsity, or when the
//! tables cannot be built - because then there is nothing to compare.
//!
//! ## The case
//!
//! The same file the Python half reads (`tools/dump_denoiser.py`):
//!
//! ```json
//! {
//!   "canvas": {"height": 768, "width": 1344},
//!   "num_frames": 124,
//!   "text_capacity": 256,
//!   "condition_capacity": 0,
//!   "token_multiple": 128,
//!   "num_text_tokens": 24,
//!   "attention": {"kind": "dense"},
//!   "vision_spans": []
//! }
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::config::Canvas;
use minimax_h3::packing;
use serde::Deserialize;

/// The elements one dense layout aligns its padded row count to when the case does not say.
const DEFAULT_TOKEN_MULTIPLE: u32 = 128;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("FAILED: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let arguments = arguments()?;
    let case: Case = read_json(&arguments.case)?;
    let canvas = canvas(&case.canvas);
    let spans: Vec<(u32, u32)> = case
        .vision_spans
        .iter()
        .map(|span| (span[0], span[1]))
        .collect();
    let conditions = conditions(&case)?;
    let clip = clip(&case)?;
    let segments = packing::condition_segments(&conditions, &canvas, clip)
        .map_err(|error| format!("ordering the conditions: {error}"))?;
    match case.attention.kind.to_ascii_lowercase().as_str() {
        "dense" => dense(&case, &canvas, &arguments, &spans, &segments),
        "sparse" => sparse(&case, &canvas, &arguments, &spans, &segments),
        other => Err(format!(
            "`{other}` is not an attention kind: dense or sparse"
        )),
    }
}

/// Dump a dense layout's row tables: the pair the protocol names, without a prefix.
fn dense(
    case: &Case,
    canvas: &Canvas,
    arguments: &Arguments,
    spans: &[(u32, u32)],
    segments: &[packing::Segment],
) -> Result<(), String> {
    let packing = packing::dense_packing(
        case.num_frames,
        canvas,
        case.text_capacity,
        case.condition_capacity,
        case.token_multiple,
    )
    .map_err(|error| format!("sizing the layout: {error}"))?;
    let tables = packing::dense_tables(
        &packing,
        case.num_frames,
        canvas,
        case.num_text_tokens,
        segments,
        spans,
    )
    .map_err(|error| format!("building the tables: {error}"))?;

    // `[rows, 4]`: the tag first, then the row's three rotary coordinates - the protocol's columns,
    // which are the reference's `token_tags` and `position_ids` concatenated exactly this way.
    let mut rows = Vec::with_capacity(tables.token_tags.len() * 16);
    for (tag, position) in tables.token_tags.iter().zip(tables.position_ids.iter()) {
        rows.extend_from_slice(&(*tag as f32).to_le_bytes());
        for coordinate in position {
            rows.extend_from_slice(&(*coordinate as f32).to_le_bytes());
        }
    }
    let boundaries: Vec<i32> = vec![
        0,
        i32::try_from(packing.padded_tokens).map_err(|_| "a layout wider than i32".to_owned())?,
        0,
        i32::try_from(tables.used).map_err(|_| "a layout wider than i32".to_owned())?,
    ];
    let mut boundary_bytes = Vec::with_capacity(boundaries.len() * 4);
    for value in &boundaries {
        boundary_bytes.extend_from_slice(&value.to_le_bytes());
    }

    let (padded, used) = (packing.padded_tokens, tables.used);
    let block = arguments.out.join("denoiser");
    fs::create_dir_all(&block).map_err(|error| format!("making {}: {error}", block.display()))?;
    write(
        &block.join("packed_rows.npy"),
        &npy("<f4", &[u64::from(padded), 4], &rows),
    )?;
    write(
        &block.join("cu_seqlens.npy"),
        &npy("<i4", &[2, 2], &boundary_bytes),
    )?;

    println!("case    {}", arguments.case.display());
    println!("canvas  {}x{} pixels", canvas.width, canvas.height);
    println!("rows    padded {padded}, attended {used}, {} tables", 2);
    println!("wrote   {}", block.display());
    Ok(())
}

/// Dump a segment layout's tables: the reference's `segment.` names, one file each.
///
/// The names and the shapes are `tools/dump_tables.py`'s (`:139-161`), so the comparator pairs each
/// one with the reference's; the values are this crate's own arithmetic. A sparse case must state
/// its tile and its sparsity, because both change the layout rather than only its contents: the tile
/// is the row count of a block, and the sparsity is the rate at which the selection drops generated
/// video tiles.
fn sparse(
    case: &Case,
    canvas: &Canvas,
    arguments: &Arguments,
    spans: &[(u32, u32)],
    segments: &[packing::Segment],
) -> Result<(), String> {
    let tile = case
        .attention
        .tile
        .ok_or_else(|| "a sparse case states its `tile`".to_owned())?;
    let sparsity = case
        .attention
        .sparsity
        .ok_or_else(|| "a sparse case states its `sparsity`".to_owned())?;
    let layout = packing::segment_packing(
        case.num_frames,
        canvas,
        case.text_capacity,
        case.condition_capacity,
        tile,
        case.token_multiple,
    )
    .map_err(|error| format!("sizing the layout: {error}"))?;
    let tables = packing::segment_tables(
        &layout,
        case.num_frames,
        canvas,
        case.num_text_tokens,
        segments,
        spans,
        packing::Sparsity {
            sparsity,
            reference_keep: case.attention.reference_keep,
        },
    )
    .map_err(|error| format!("building the tables: {error}"))?;

    let extent = &layout.extent;
    let padded = u64::from(extent.padded_tokens);
    if tables.token_tags.len() as u64 != padded || tables.position_ids.len() as u64 != padded {
        return Err(format!(
            "the layout holds {padded} padded rows and the tables hold {} tags and {} positions",
            tables.token_tags.len(),
            tables.position_ids.len()
        ));
    }
    let tiles = tables.tile_segments.len() as u64;
    if tables.variable_block_sizes.len() as u64 != tiles
        || tables.segment_starts.len() as u64 != tiles
        || tables.segment_keep.len() as u64 != tiles
    {
        return Err(
            "the layout's four per-tile tables disagree on how many tiles it has".to_owned(),
        );
    }
    let video_rows = layout.video_indices.len() as u64;

    // Three coordinates a row, in double precision: they are the raster's own cell positions, and a
    // rounded float32 would blur a boundary that the comparison is meant to localise.
    let mut coordinates = Vec::with_capacity(tables.position_ids.len() * 24);
    for position in &tables.position_ids {
        for coordinate in position {
            coordinates.extend_from_slice(&coordinate.to_le_bytes());
        }
    }
    let block = arguments.out.join("denoiser");
    fs::create_dir_all(&block).map_err(|error| format!("making {}: {error}", block.display()))?;
    write(
        &block.join("segment.position_ids.npy"),
        &npy("<f8", &[padded, 3], &coordinates),
    )?;
    write(
        &block.join("segment.token_tags.npy"),
        &npy("<i4", &[padded], &i32_bytes(&tables.token_tags)),
    )?;
    write(
        &block.join("segment.groups.npy"),
        &npy("<i4", &[padded], &i32_bytes(&tables.groups)),
    )?;
    write(
        &block.join("segment.prefix_index.npy"),
        &npy("<i4", &[padded], &i32_bytes(&tables.prefix_index)),
    )?;
    write(
        &block.join("segment.variable_block_sizes.npy"),
        &npy("<i4", &[tiles], &u32_bytes(&tables.variable_block_sizes)),
    )?;
    write(
        &block.join("segment.tile_segments.npy"),
        &npy("<i4", &[tiles], &i32_bytes(&tables.tile_segments)),
    )?;
    write(
        &block.join("segment.segment_starts.npy"),
        &npy("<i4", &[tiles], &i32_bytes(&tables.segment_starts)),
    )?;
    write(
        &block.join("segment.segment_keep.npy"),
        &npy("<i4", &[tiles], &i32_bytes(&tables.segment_keep)),
    )?;
    write(
        &block.join("segment.video_indices.npy"),
        &npy("<i4", &[video_rows], &u32_bytes(&layout.video_indices)),
    )?;
    write(
        &block.join("segment.video_raster_indices.npy"),
        &npy(
            "<i4",
            &[video_rows],
            &u32_bytes(&layout.video_raster_indices),
        ),
    )?;
    // The five counts the reference writes as one-element int64 tensors rather than as arrays.
    for (name, value) in [
        ("audio_rows", extent.audio_rows),
        ("audio_start", extent.audio_start()),
        ("video_tiles", extent.video_tiles),
        ("video_start", extent.video_start()),
        ("padded_tokens", extent.padded_tokens),
    ] {
        write(
            &block.join(format!("segment.{name}.npy")),
            &npy("<i8", &[1], &i64_bytes(&[i64::from(value)])),
        )?;
    }

    let kept: i64 = tables
        .segment_keep
        .iter()
        .map(|keep| i64::from(*keep))
        .sum();
    println!("case    {}", arguments.case.display());
    println!("canvas  {}x{} pixels", canvas.width, canvas.height);
    println!(
        "layout  {tiles} tiles of {tile} rows, {video_rows} generated video rows, {padded} padded"
    );
    println!(
        "kept    {kept} key tiles over {} segments",
        extent.video_tiles
    );
    println!("wrote   {} (15 tables)", block.display());
    Ok(())
}

/// An `i32` table's little-endian bytes.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// An `i64` table's little-endian bytes.
fn i64_bytes(values: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// A `u32` table as `i32`: the protocol's indices are non-negative and never above `i32::MAX`, and
/// NumPy has no unsigned type the comparator pairs with the reference's signed tensors.
fn u32_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        let signed = i32::try_from(*value).unwrap_or(i32::MAX);
        bytes.extend_from_slice(&signed.to_le_bytes());
    }
    bytes
}

/// The case's canvas from the pixel extents it states.
///
/// The canvas' name - its short edge and its aspect ratio - is what a request selects it by, and it
/// is **not recoverable** from the raster: the served `16:9` canvas is 1344x768, whose own ratio is
/// 7:4. The layout arithmetic reads the extents alone, so the name is derived here in lowest terms
/// only so a printed plan is recognisable.
fn canvas(size: &Size) -> Canvas {
    let divisor = gcd(size.width, size.height).max(1);
    Canvas {
        short_edge: size.height.min(size.width),
        aspect: (size.width / divisor, size.height / divisor),
        width: size.width,
        height: size.height,
    }
}

/// The request's conditions, in its own order, refusing a role the packing does not know.
fn conditions(case: &Case) -> Result<Vec<packing::Condition>, String> {
    let mut conditions = Vec::with_capacity(case.conditions.len());
    for condition in &case.conditions {
        let role = match condition.role.as_str() {
            "first_frame" => packing::Role::FirstFrame,
            "last_frame" => packing::Role::LastFrame,
            "reference" => packing::Role::Reference,
            other => return Err(format!("`{other}` is not a condition role")),
        };
        let frame = condition.canvas.as_ref().map(|size| packing::Frame {
            canvas: canvas(size),
            num_frames: condition.frames,
        });
        conditions.push(packing::Condition {
            role,
            frame,
            audio_samples: condition.audio_samples,
        });
    }
    Ok(conditions)
}

/// The encoder's clip rule: the case's, or the released checkpoint's
/// (`vae/config.json`: 17 frames, 5 latents, one dropped).
fn clip(case: &Case) -> Result<minimax_h3::encoding::Clip, String> {
    Ok(match &case.clip {
        Some(clip) => minimax_h3::encoding::Clip {
            frames: clip.frames,
            latents: clip.latents,
            token_drop: clip.token_drop,
        },
        None => minimax_h3::encoding::Clip {
            frames: 17,
            latents: 5,
            token_drop: 1,
        },
    })
}

/// The greatest common divisor of two positive extents, for the canvas' ratio.
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A host table as a NumPy `.npy` v1.0 file: little-endian, row-major, one flat buffer.
///
/// The header's dictionary and its padding are the format's: magic, version, a `u16` header length
/// whose total up to the data is a multiple of 64 bytes, `descr`, `fortran_order`, `shape`, then the
/// entries in C order. `crates/models/minimax-h3/examples/compare_encoder_tables.rs` carries the
/// same writer for its own tables; the two are kept apart because the diagnosis' two halves are
/// owned apart, and the format is pinned by each file's test.
fn npy(descriptor: &str, shape: &[u64], bytes: &[u8]) -> Vec<u8> {
    let shape = match shape.len() {
        0 => "()".to_owned(),
        1 => format!("({},)", shape[0]),
        _ => shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", "),
    };
    let mut header =
        format!("{{'descr': '{descriptor}', 'fortran_order': False, 'shape': ({shape}), }}");
    // The dictionary, the length and the magic occupy 10 bytes before the header's text; the data
    // starts on a 64-byte boundary, and the header's last byte is a newline.
    let unpadded = 10 + header.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');

    let mut file = Vec::with_capacity(10 + header.len() + bytes.len());
    file.extend_from_slice(b"\x93NUMPY");
    file.push(1);
    file.push(0);
    file.extend_from_slice(&(header.len() as u16).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(bytes);
    file
}

/// Write one file, naming both sides of a failure.
fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::write(path, bytes).map_err(|error| format!("writing {}: {error}", path.display()))
}

/// Read one JSON file, naming both sides of a failure.
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("parsing {}: {error}", path.display()))
}

/// The two paths a run needs.
struct Arguments {
    case: PathBuf,
    out: PathBuf,
}

/// Read `--case <file>` and `--out <dir>`, refusing anything else by name.
fn arguments() -> Result<Arguments, String> {
    let mut arguments = std::env::args().skip(1);
    let mut case = None;
    let mut out = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--case" => case = arguments.next().map(PathBuf::from),
            "--out" => out = arguments.next().map(PathBuf::from),
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }
    let case = case.ok_or_else(|| "`--case <case.json>` is required".to_owned())?;
    let out = out.ok_or_else(|| "`--out <dir>` is required".to_owned())?;
    Ok(Arguments { case, out })
}

/// The case file, as `tools/dump_denoiser.py` reads it.
#[derive(Deserialize)]
struct Case {
    /// The raster a request generates, in pixels.
    canvas: Size,
    /// The request's frame count, before the alignment to the checkpoint's timeline.
    num_frames: u32,
    /// The prefix's text capacity, in rows.
    text_capacity: u32,
    /// The prefix's condition capacity, in rows.
    #[serde(default)]
    condition_capacity: u32,
    /// The alignment of the padded row count.
    #[serde(default = "default_token_multiple")]
    token_multiple: u32,
    /// The prompt's own row count, which the layout's text rows start from.
    num_text_tokens: u32,
    /// The attention the request runs, whose kind names the packing.
    attention: Attention,
    /// The prompt's vision-token spans, as half-open row ranges.
    #[serde(default)]
    vision_spans: Vec<[u32; 2]>,
    /// The request's conditions, in its own order.
    #[serde(default)]
    conditions: Vec<Condition>,
    /// The encoder's clip rule, which a condition's latent frame count follows.
    #[serde(default)]
    clip: Option<ClipCase>,
}

/// One condition of the request, as the case states it.
#[derive(Deserialize)]
struct Condition {
    /// `first_frame`, `last_frame` or `reference`.
    role: String,
    /// Its raster, absent for an audio-only reference.
    canvas: Option<Size>,
    /// Its frame count; one for a still.
    #[serde(default = "one_frame")]
    frames: u32,
    /// The samples its soundtrack carries.
    #[serde(default)]
    audio_samples: u32,
}

fn one_frame() -> u32 {
    1
}

/// The encoder's clip rule, as the checkpoint's video VAE states it.
#[derive(Deserialize)]
struct ClipCase {
    frames: u32,
    latents: u32,
    #[serde(default)]
    token_drop: u32,
}

fn default_token_multiple() -> u32 {
    DEFAULT_TOKEN_MULTIPLE
}

/// A canvas, in pixels.
#[derive(Deserialize)]
struct Size {
    height: u32,
    width: u32,
}

/// The request's attention: its kind, and - for a sparse one - the tile and the rates.
#[derive(Deserialize)]
struct Attention {
    /// `dense` or `sparse`.
    kind: String,
    /// A sparse layout's tile: the row count of one block, a key of the reference's tile shapes.
    #[serde(default)]
    tile: Option<u32>,
    /// The fraction of a generated video segment's tiles the selection drops.
    #[serde(default)]
    sparsity: Option<f64>,
    /// The fraction of a reference segment's tiles the selection keeps.
    #[serde(default)]
    reference_keep: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_npy_header_pads_to_a_multiple_of_64_bytes() {
        let bytes = vec![0_u8; 24];
        let file = npy("<i4", &[3, 2], &bytes);
        assert_eq!(&file[..6], b"\x93NUMPY");
        let length = u16::from_le_bytes([file[8], file[9]]) as usize;
        assert_eq!((10 + length) % 64, 0);
        assert_eq!(file[10 + length - 1], b'\n');
        let header = std::str::from_utf8(&file[10..10 + length - 1]).expect("ascii");
        assert!(header.contains("'descr': '<i4'"), "{header}");
        assert!(header.contains("'shape': (3, 2)"), "{header}");
        assert_eq!(file.len(), 10 + length + 24);
    }

    /// The protocol's two tensors carry the shapes the comparator pairs: a row of four values, and
    /// two boundaries of two ends.
    #[test]
    fn the_row_tables_carry_the_protocols_shapes() {
        let case: Case = serde_json::from_str(
            r#"{
                "canvas": {"height": 768, "width": 1344},
                "num_frames": 124,
                "text_capacity": 256,
                "token_multiple": 128,
                "num_text_tokens": 24,
                "attention": {"kind": "dense"}
            }"#,
        )
        .expect("a case");
        let canvas = canvas(&case.canvas);
        let packing = packing::dense_packing(
            case.num_frames,
            &canvas,
            case.text_capacity,
            0,
            case.token_multiple,
        )
        .expect("a layout");
        let tables = packing::dense_tables(
            &packing,
            case.num_frames,
            &canvas,
            case.num_text_tokens,
            &[],
            &[],
        )
        .expect("tables");
        // One row of four values per padded row, and the tags are as many as the rows.
        assert_eq!(tables.token_tags.len() as u32, packing.padded_tokens);
        assert_eq!(tables.position_ids.len(), tables.token_tags.len());
        assert!(tables.used <= packing.padded_tokens);
        // The canvas' name is in lowest terms: 1344x768 is 16:9.
        assert_eq!(canvas.aspect, (7, 4), "1344x768 in lowest terms is 7:4");
    }

    /// The segment layout's tables carry one entry per row, per tile and per generated video row -
    /// the invariant the comparator needs, since it pairs each table with the reference's by name and
    /// refuses a shape that differs.
    #[test]
    fn the_segment_tables_carry_one_entry_per_row_and_tile() {
        let case: Case = serde_json::from_str(
            r#"{
                "canvas": {"height": 768, "width": 1344},
                "num_frames": 22,
                "text_capacity": 128,
                "condition_capacity": 64,
                "token_multiple": 64,
                "num_text_tokens": 40,
                "attention": {"kind": "sparse", "tile": 64, "sparsity": 0.8}
            }"#,
        )
        .expect("a case");
        let canvas = canvas(&case.canvas);
        let tile = case.attention.tile.expect("the case states its tile");
        let layout = packing::segment_packing(
            case.num_frames,
            &canvas,
            case.text_capacity,
            case.condition_capacity,
            tile,
            case.token_multiple,
        )
        .expect("a layout");
        let tables = packing::segment_tables(
            &layout,
            case.num_frames,
            &canvas,
            case.num_text_tokens,
            &[],
            &[],
            packing::Sparsity {
                sparsity: 0.8,
                reference_keep: None,
            },
        )
        .expect("tables");

        let padded = layout.extent.padded_tokens;
        let tiles = padded / tile;
        assert_eq!(padded % tile, 0, "a segment layout pads to whole tiles");
        assert_eq!(tables.token_tags.len() as u32, padded);
        assert_eq!(tables.position_ids.len() as u32, padded);
        assert_eq!(tables.groups.len() as u32, padded);
        assert_eq!(tables.prefix_index.len() as u32, padded);
        assert_eq!(tables.variable_block_sizes.len() as u32, tiles);
        assert_eq!(tables.tile_segments.len() as u32, tiles);
        assert_eq!(tables.segment_starts.len() as u32, tiles);
        assert_eq!(tables.segment_keep.len() as u32, tiles);

        // Every generated video row lands in one packed row, in ascending order, and no segment
        // keeps more key tiles than the layout has.
        assert_eq!(
            layout.video_indices.len(),
            layout.video_raster_indices.len()
        );
        assert!(
            layout
                .video_indices
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "the generated video's packed rows ascend"
        );
        assert!(
            layout.video_indices.iter().all(|row| *row < padded),
            "every packed row is inside the layout"
        );
        let kept: i64 = tables
            .segment_keep
            .iter()
            .map(|keep| i64::from(*keep))
            .sum();
        assert!(
            kept <= i64::from(tiles) && kept >= 0,
            "a layout of {tiles} tiles keeps {kept} key tiles"
        );
        // The prefix source holds the text capacity, the condition capacity and one row of zeros, and
        // every packed row addresses one of its rows.
        let prefix = case.text_capacity + case.condition_capacity + 1;
        assert!(
            tables.prefix_index.iter().all(|row| *row >= 0)
                && tables.prefix_index.iter().all(|row| (*row as u32) < prefix),
            "every packed row addresses a row of the prefix source"
        );
    }
}
