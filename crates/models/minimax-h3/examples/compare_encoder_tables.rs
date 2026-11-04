//! Dump the text encoder's prepared tables for the block-by-block diagnosis (design 10).
//!
//! Run it with
//!
//! ```text
//! cargo run -p minimax-h3 --example compare_encoder_tables -- --case <case.json> --out <dir>
//! ```
//!
//! It writes one `.npy` per table under `<dir>/text_encoder/`, the same names
//! `crates/models/minimax-h3/tools/dump_encoder_tables.py` writes from today's implementation, so
//! `tools/diagnosis/compare.py --ref <python dir> --got <this dir>` reports the per-table
//! difference. No checkpoint, no device: every value here is host arithmetic in
//! [`minimax_h3::positions`], which is exactly the half a structural mistake (a wrong row, a wrong
//! order, a wrong rotation axis) would show in first.
//!
//! This is a **diagnosis, never an acceptance gate** (specs/worker.md 10): nothing is gated on the
//! numbers. A run fails only when the case cannot be read or a table cannot be written, because
//! then there is nothing to diagnose.
//!
//! ## The case (`case.json`)
//!
//! ```json
//! {
//!   "image_token_id": 151655, "video_token_id": 151656,
//!   "head_dim": 72, "rope_theta": 10000.0, "position_grid": 48, "merge": 2,
//!   "ids": [9707, 151655, 151655, 151655, 151655, 13],
//!   "image_grids": [[1, 4, 4]],
//!   "video_grids": []
//! }
//! ```
//!
//! `ids` is the prompt's token ids (the tokenizer's output, placeholders included), each block's
//! grid is `[time, height, width]` in patches, and a grid's placeholder run must hold exactly its
//! merged token count on both sides.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::positions::{self, VisionRotary};
use serde::Deserialize;

/// What a case describes: one prompt and the grids of its media.
#[derive(Deserialize)]
struct Case {
    image_token_id: u32,
    video_token_id: u32,
    head_dim: u32,
    rope_theta: f32,
    position_grid: u32,
    merge: u32,
    ids: Vec<u32>,
    #[serde(default)]
    image_grids: Vec<[u32; 3]>,
    #[serde(default)]
    video_grids: Vec<[u32; 3]>,
}

fn main() -> ExitCode {
    match run() {
        Ok(written) => {
            println!("compare_encoder_tables: wrote {written} tables");
            ExitCode::SUCCESS
        }
        Err(problem) => {
            eprintln!("compare_encoder_tables: {problem}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<usize, String> {
    let (case_path, out) = arguments()?;
    let text = fs::read_to_string(&case_path)
        .map_err(|error| format!("reading {}: {error}", case_path.display()))?;
    let case: Case =
        serde_json::from_str(&text).map_err(|error| format!("parsing the case: {error}"))?;
    let directory = out.join("text_encoder");
    fs::create_dir_all(&directory)
        .map_err(|error| format!("creating {}: {error}", directory.display()))?;

    let mut written = 0;
    let mut write = |name: &str, table: Table| -> Result<(), String> {
        let path = directory.join(format!("{name}.npy"));
        fs::write(&path, table.to_npy()).map_err(|error| format!("writing {path:?}: {error}"))?;
        written += 1;
        Ok(())
    };

    // The prompt's M-RoPE coordinates, one row per axis: the reference's `[3, tokens]` layout.
    let coordinates = positions::rope_index(
        &case.ids,
        &case.image_grids,
        &case.video_grids,
        case.image_token_id,
        case.video_token_id,
        case.merge,
    )
    .map_err(problem)?;
    let tokens = coordinates.len();
    let mut positions_table = Vec::with_capacity(tokens * 3);
    for axis in 0..3 {
        for coordinate in &coordinates {
            positions_table.push(coordinate[axis] as i32);
        }
    }
    write("prompt_positions", Table::I32(positions_table, vec![3, tokens as u64]))?;

    let rows = positions::placeholder_rows(&case.ids, case.image_token_id, case.video_token_id);
    write("vision_rows", Table::I32(rows, vec![tokens as u64]))?;

    for (index, grid) in case.image_grids.iter().enumerate() {
        let (cosine, sine, blended, lengths) = media(&case, *grid)?;
        write(&format!("image{index}_cosine"), cosine)?;
        write(&format!("image{index}_sine"), sine)?;
        write(&format!("image{index}_position_row"), blended)?;
        write(&format!("image{index}_lengths"), lengths)?;
    }
    for (index, grid) in case.video_grids.iter().enumerate() {
        let (cosine, sine, blended, lengths) = media(&case, *grid)?;
        write(&format!("video{index}_cosine"), cosine)?;
        write(&format!("video{index}_sine"), sine)?;
        write(&format!("video{index}_position_row"), blended)?;
        write(&format!("video{index}_lengths"), lengths)?;
    }
    Ok(written)
}

/// The four tables of one block's patch grid: its rotary factors, its blended position row and its
/// attention lengths.
///
/// A temporal grid repeats one time step's factors and samples on every time step, because the
/// tower treats each temporal patch as its own sequence over the same `height x width` grid.
fn media(case: &Case, grid: [u32; 3]) -> Result<(Table, Table, Table, Table), String> {
    let [time, height, width] = grid;
    let coordinates = positions::rotary_coordinates(height, width, case.merge).map_err(problem)?;
    let patches = coordinates.len() * time as usize;
    let mut per_step = Vec::with_capacity(patches);
    for _ in 0..time {
        per_step.extend_from_slice(&coordinates);
    }

    let per_axis = positions::VisionRotary::new(case.head_dim).map_err(problem)?.per_axis();
    let frequencies =
        positions::inverse_frequencies(case.head_dim / 2, case.rope_theta).map_err(problem)?;
    let columns = frequencies.len();
    if usize::try_from(per_axis).map(|width| width * 2) != Ok(columns) {
        return Err(format!(
            "the {}-element head has {per_axis} frequencies per axis and {columns} in the table",
            case.head_dim
        ));
    }
    let rotary = VisionRotary::new(case.head_dim).map_err(problem)?;
    let mut cosine = vec![0.0_f32; patches * columns];
    let mut sine = vec![0.0_f32; patches * columns];
    rotary
        .factors(&per_step, &frequencies, &mut cosine, &mut sine)
        .map_err(problem)?;

    // The four table rows and weights of every patch's sample, blended into one number: the table
    // row index it would read, which the reference reproduces by blending a table of row indices.
    let (indices, weights) = positions::position_samples(
        case.position_grid,
        height,
        width,
        case.merge,
    )
    .map_err(problem)?;
    let mut blended = Vec::with_capacity(patches);
    for _ in 0..time {
        for (rows, weights) in indices.iter().zip(&weights) {
            let mut row = 0.0_f32;
            for (index, weight) in rows.iter().zip(weights) {
                row += *weight * *index as f32;
            }
            blended.push(row);
        }
    }

    let lengths = positions::vision_lengths(time, height, width).map_err(problem)?;
    Ok((
        Table::F32(cosine, vec![patches as u64, columns as u64]),
        Table::F32(sine, vec![patches as u64, columns as u64]),
        Table::F32(blended, vec![patches as u64]),
        Table::I32(lengths, vec![time as u64 + 1]),
    ))
}

/// The `--case` and `--out` arguments.
fn arguments() -> Result<(PathBuf, PathBuf), String> {
    let mut case = None;
    let mut out = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--case" => case = arguments.next().map(PathBuf::from),
            "--out" => out = arguments.next().map(PathBuf::from),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let case = case.ok_or("--case <case.json> is required")?;
    let out = out.ok_or("--out <dir> is required")?;
    Ok((case, out))
}

fn problem(error: minimax_h3::Error) -> String {
    error.to_string()
}

/// A table as `.npy` understands it: little-endian, row-major, one flat buffer.
enum Table {
    I32(Vec<i32>, Vec<u64>),
    F32(Vec<f32>, Vec<u64>),
}

impl Table {
    /// The table as a NumPy `.npy` v1.0 file.
    ///
    /// The header's dictionary and its padding are the format's: magic, version, a `u16` header
    /// length whose total up to the data is a multiple of 64 bytes, `descr`, `fortran_order`,
    /// `shape`, then the entries with C order.
    fn to_npy(&self) -> Vec<u8> {
        let (descriptor, shape, bytes) = match self {
            Self::I32(values, shape) => {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                ("<i4", shape, bytes)
            }
            Self::F32(values, shape) => {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                ("<f4", shape, bytes)
            }
        };
        let shape = match shape.len() {
            1 => format!("({},)", shape[0]),
            _ => shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        };
        let mut header = format!(
            "{{'descr': '{descriptor}', 'fortran_order': False, 'shape': ({shape}), }}"
        );
        // The dictionary, the length and the magic occupy 10 bytes before the header's text; the
        // data starts on a 64-byte boundary, and the header's last byte is a newline.
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
        file.extend_from_slice(&bytes);
        file
    }
}

/// A path's final component, for the report line.
#[allow(dead_code)]
fn name_of(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("?")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_npy_header_pads_to_a_multiple_of_64_bytes() {
        let file = Table::I32(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).to_npy();
        assert_eq!(&file[..6], b"\x93NUMPY");
        let length = u16::from_le_bytes([file[8], file[9]]) as usize;
        assert_eq!((10 + length) % 64, 0);
        assert_eq!(file[10 + length - 1], b'\n');
        let header = std::str::from_utf8(&file[10..10 + length - 1]).expect("ascii");
        assert!(header.contains("'descr': '<i4'"));
        assert!(header.contains("'shape': (3, 2)"));
        assert_eq!(file.len(), 10 + length + 24);
    }

    #[test]
    fn a_one_dimensional_shape_is_a_tuple() {
        let file = Table::F32(vec![0.5], vec![1]).to_npy();
        let length = u16::from_le_bytes([file[8], file[9]]) as usize;
        let header = std::str::from_utf8(&file[10..10 + length - 1]).expect("ascii");
        assert!(header.contains("'shape': (1,)"), "{header}");
        assert!(header.contains("'descr': '<f4'"));
    }
}
