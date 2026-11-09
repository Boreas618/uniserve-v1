//! The sparse layout's tile and segment tables for one case, written for comparison.
//!
//! The sparse path's arithmetic runs on a device, but everything that decides *which* tiles a query
//! reads is host data: `variable_block_sizes`, `tile_segments`, `segment_starts` and `segment_keep`.
//! A wrong entry in one of them changes the model silently - a kept tile is a column a query reads -
//! and no shape, type or key refuses it. This example writes those tables, and the case that
//! produced them, so `tools/compare_vsa_tables.py` can build the reference's own tables for the same
//! case and compare them entry by entry.
//!
//! ```text
//! cargo run -q -p minimax-h3 --example vsa_tables -- \
//!     --width 256 --height 256 --frames 22 --text-rows 128 --condition-rows 192 \
//!     --tile 64 --sparsity 0.8 --out artifacts/h3-vsa
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use minimax_h3::config::Canvas;
use minimax_h3::packing::{self, Sparsity};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("FAILURE: {message}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    width: u32,
    height: u32,
    frames: u32,
    text_rows: u32,
    condition_rows: u32,
    text_tokens: u32,
    tile: u32,
    sparsity: f64,
    reference_keep: Option<f64>,
    token_multiple: Option<u32>,
    out: PathBuf,
}

fn run() -> Result<(), String> {
    let mut args = Args {
        width: 256,
        height: 256,
        frames: 22,
        text_rows: 128,
        condition_rows: 192,
        text_tokens: 128,
        tile: 64,
        sparsity: 0.8,
        reference_keep: None,
        token_multiple: None,
        out: PathBuf::new(),
    };
    let mut words = std::env::args().skip(1);
    while let Some(word) = words.next() {
        let mut value = |name: &str| -> Result<String, String> {
            words.next().ok_or_else(|| format!("{name} needs a value"))
        };
        match word.as_str() {
            "--width" => args.width = number(&value("--width")?)?,
            "--height" => args.height = number(&value("--height")?)?,
            "--frames" => args.frames = number(&value("--frames")?)?,
            "--text-rows" => args.text_rows = number(&value("--text-rows")?)?,
            "--condition-rows" => args.condition_rows = number(&value("--condition-rows")?)?,
            "--text-tokens" => args.text_tokens = number(&value("--text-tokens")?)?,
            "--tile" => args.tile = number(&value("--tile")?)?,
            "--token-multiple" => {
                args.token_multiple = Some(number(&value("--token-multiple")?)?);
            }
            "--sparsity" => {
                args.sparsity = value("--sparsity")?
                    .parse()
                    .map_err(|error| format!("--sparsity: {error}"))?;
            }
            "--reference-keep" => {
                args.reference_keep = Some(
                    value("--reference-keep")?
                        .parse()
                        .map_err(|error| format!("--reference-keep: {error}"))?,
                );
            }
            "--out" => args.out = PathBuf::from(value("--out")?),
            "--help" | "-h" => {
                return Err(
                    "vsa_tables --width N --height N --frames N --text-rows N --condition-rows N \
                     --text-tokens N --tile 64|128 --sparsity F [--reference-keep F] \
                     [--token-multiple N] --out DIR"
                        .to_owned(),
                );
            }
            other => return Err(format!("unknown flag `{other}`")),
        }
    }
    if args.out.as_os_str().is_empty() {
        eprintln!("name the output directory with --out");
        std::process::exit(2);
    }

    // The canvas a request of these pixels names: whole multiples of the canvas grid, as the
    // denoiser's own `canvas_of` rounds them.
    let width = round_up(args.width, minimax_h3::CANVAS_MULTIPLE);
    let height = round_up(args.height, minimax_h3::CANVAS_MULTIPLE);
    let divisor = gcd(width, height);
    let canvas = Canvas {
        short_edge: width.min(height),
        aspect: (width / divisor, height / divisor),
        width,
        height,
    };
    // The alignment is whole tiles unless the case states otherwise: a sharded deployment aligns to
    // a multiple of the tile so every rank holds whole tiles, and a single rank needs only the tile.
    let multiple = args.token_multiple.unwrap_or(args.tile);
    let packing = packing::segment_packing(
        args.frames,
        &canvas,
        args.text_rows,
        args.condition_rows,
        args.tile,
        multiple,
    )
    .map_err(|error| format!("a sparse layout of this case: {error}"))?;
    let tables = packing::segment_tables(
        &packing,
        args.frames,
        &canvas,
        args.text_tokens,
        // A `t2va` case: no reference media, so no condition segments and no vision spans.
        &[],
        &[],
        Sparsity {
            sparsity: args.sparsity,
            reference_keep: args.reference_keep,
        },
    )
    .map_err(|error| format!("the tables of this case: {error}"))?;

    let extent = &packing.extent;
    // The segment count is what the tile table states, not what the keep table's fill implies: the
    // keep table is over-allocated to the tile count (`denoiser.rs`, `packing.py:1228-1239`), so its
    // entries past the last segment are unread and hold whatever the writer left there.
    let segments = tables
        .tile_segments
        .iter()
        .map(|segment| segment + 1)
        .max()
        .unwrap_or(0)
        .max(0) as usize;
    let case = format!(
        "{{\"frames\": {}, \"canvas\": [{}, {}], \"text_rows\": {}, \"condition_rows\": {}, \
         \"text_tokens\": {}, \"tile\": {}, \"sparsity\": {}, \"reference_keep\": {}, \
         \"token_multiple\": {}, \"padded_tokens\": {}, \"video_tiles\": {}}}\n",
        args.frames,
        canvas.height,
        canvas.width,
        args.text_rows,
        args.condition_rows,
        args.text_tokens,
        args.tile,
        args.sparsity,
        match args.reference_keep {
            Some(rate) => format!("{rate}"),
            None => "null".to_owned(),
        },
        multiple,
        extent.padded_tokens,
        extent.video_tiles,
    );
    let written = format!(
        "{{\"variable_block_sizes\": {}, \"tile_segments\": {}, \"segment_starts\": {}, \
         \"segment_keep\": {}, \"segments\": {segments}, \"video_start\": {}, \"video_tiles\": {}, \
         \"padded_tokens\": {}}}\n",
        list(&tables.variable_block_sizes),
        list(&tables.tile_segments),
        list(&tables.segment_starts),
        list(&tables.segment_keep),
        extent.video_start(),
        extent.video_tiles,
        extent.padded_tokens,
    );
    std::fs::create_dir_all(&args.out)
        .map_err(|error| format!("{}: {error}", args.out.display()))?;
    std::fs::write(args.out.join("case.json"), case)
        .map_err(|error| format!("{}: {error}", args.out.display()))?;
    std::fs::write(args.out.join("tables.json"), written)
        .map_err(|error| format!("{}: {error}", args.out.display()))?;
    println!(
        "tables: {} tiles over {} rows, {} segments, video tiles {} of {}, keep {:?}",
        extent.padded_tokens / args.tile,
        extent.padded_tokens,
        segments,
        extent.video_tiles,
        packing::kept_tiles(args.sparsity, extent.video_tiles),
        &tables.segment_keep[..segments],
    );
    Ok(())
}

fn number(word: &str) -> Result<u32, String> {
    word.parse().map_err(|error| format!("`{word}`: {error}"))
}

fn list(values: &[impl std::fmt::Display]) -> String {
    let entries: Vec<String> = values.iter().map(|value| value.to_string()).collect();
    format!("[{}]", entries.join(", "))
}

fn round_up(value: u32, multiple: u32) -> u32 {
    value.div_ceil(multiple) * multiple
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}
