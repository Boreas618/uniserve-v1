//! Dump the text encoder's host tables for the block-by-block diagnosis (specs/worker.md 10).
//!
//! ```text
//! cargo run -p minimax-h3 --example compare_encoder_tables -- --case <case.json> --out <dir>
//! ```
//!
//! It writes one `.npy` per table under `<dir>/text_encoder/`, under the names the encoder itself
//! declares in `Encoder::state_tables` (`prompt_positions`, `vision_destinations`,
//! `vision_position_index`, `vision_position_weight`, `vision_cosine`, `vision_sine`,
//! `vision_lengths`). `crates/models/minimax-h3/tools/dump_encoder_tables.py` writes the same names
//! from today's implementation, so `tools/diagnosis/compare.py --ref <python dir> --got <this dir>`
//! reports the per-table difference.
//!
//! No checkpoint and no device: the tables are host arithmetic ([`minimax_h3::text_encoder::
//! table_bytes`]), which is exactly the half a structural mistake - a wrong row, a wrong order, a
//! wrong rotation axis - shows in first. The configuration comes from the checkpoint's own
//! `config.json` and `preprocessor_config.json`, so the diagnosis uses the same numbers serving does.
//!
//! This is a **diagnosis, never an acceptance gate**: nothing is gated on the values. A run fails
//! only when the case or the configuration cannot be read, because then there is nothing to compare.
//!
//! ## The case (`case.json`)
//!
//! ```json
//! {
//!   "ids": [9707, 151655, 151655, 151655, 151655, 13],
//!   "image_grids": [[1, 4, 4]],
//!   "video_grids": []
//! }
//! ```
//!
//! `ids` is the prompt's token ids with its placeholders already spliced in, and each grid is
//! `[time, height, width]` in patches: `image_grids` for the blocks whose placeholders are the image
//! token, `video_grids` for the video token. A grid's placeholder runs must hold exactly its merged
//! token count - one run for an image, one run per temporal patch for a video.

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::config::TextEncoderConfig;
use minimax_h3::text_encoder::table_bytes;
use nn::DType;
use nn::capability::{EncoderLayout, ImageShape};
use serde::Deserialize;

/// The released checkpoint's text encoder directory.
const DEFAULT_CHECKPOINT: &str = "/opt/uv/models/MiniMax-H3/text_encoder";

/// What a case describes: one prompt and the grids of its media.
#[derive(Deserialize)]
struct Case {
    /// The prompt's token ids, placeholders included.
    ids: Vec<u32>,
    /// The patch grids of the image blocks, in prompt order.
    #[serde(default)]
    image_grids: Vec<[u32; 3]>,
    /// The patch grids of the video blocks, in prompt order.
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
    let (case_path, out, checkpoint) = arguments()?;
    let case = read_case(&case_path)?;
    let config = read_config(&checkpoint)?;
    let layout = layout_of(&config, &case)?;
    let tables = table_bytes(&config, &case.ids, &layout).map_err(problem)?;

    let directory = out.join("text_encoder");
    fs::create_dir_all(&directory)
        .map_err(|error| format!("creating {}: {error}", directory.display()))?;
    for (table, bytes) in &tables {
        let file = npy(table.dtype, table.shape.dims(), bytes)?;
        let path = directory.join(format!("{}.npy", table.name));
        fs::write(&path, file).map_err(|error| format!("writing {path:?}: {error}"))?;
    }
    Ok(tables.len())
}

/// The prompt's blocks, in prompt order, from the case's two grid lists and its ids.
///
/// The layout names one block per tower call: an image is one call, a video is one call covering all
/// its temporal patches, and both are consumed in the order the prompt's placeholder runs appear.
fn layout_of(config: &TextEncoderConfig, case: &Case) -> Result<EncoderLayout, String> {
    let merge = config.vision.spatial_merge_size;
    let mut images = VecDeque::from(case.image_grids.clone());
    let mut videos: VecDeque<([u32; 3], u32)> = case
        .video_grids
        .iter()
        .map(|grid| (*grid, grid[0]))
        .collect();
    let mut blocks = Vec::new();
    let mut row = 0;
    while row < case.ids.len() {
        let id = case.ids[row];
        if id != config.image_token_id && id != config.video_token_id {
            row += 1;
            continue;
        }
        let start = row;
        while row < case.ids.len() && case.ids[row] == id {
            row += 1;
        }
        let run = (row - start) as u64;
        // The run must hold exactly its grid's merged rows for one time step.
        if id == config.image_token_id {
            let Some(grid) = images.pop_front() else {
                return Err(format!(
                    "an image run of {run} placeholders with no grid left"
                ));
            };
            let rows = merged_rows(grid, merge)?;
            if run != rows {
                return Err(format!(
                    "an image run of {run} placeholders for a {grid:?} grid"
                ));
            }
            blocks.push(ImageShape { grid });
            continue;
        }
        let Some((grid, remaining)) = videos.front_mut() else {
            return Err(format!(
                "a video run of {run} placeholders with no grid left"
            ));
        };
        let whole = *grid;
        let rows = u64::from(whole[1] / merge) * u64::from(whole[2] / merge);
        if run != rows {
            return Err(format!(
                "a video run of {run} placeholders for one step of a {whole:?} grid"
            ));
        }
        *remaining -= 1;
        if *remaining == 0 {
            blocks.push(ImageShape { grid: whole });
            videos.pop_front();
        }
    }
    if let Some((grid, _)) = videos.front() {
        return Err(format!(
            "the video grid {grid:?} has fewer runs than its steps"
        ));
    }
    if let Some(grid) = images.front() {
        return Err(format!("the image grid {grid:?} has no placeholder run"));
    }
    Ok(EncoderLayout {
        rows: case.ids.len() as u32,
        tokens: case.ids.len() as u32,
        images: blocks,
    })
}

/// A grid's merged rows: one per `merge x merge` block, `time` steps deep.
fn merged_rows(grid: [u32; 3], merge: u32) -> Result<u64, String> {
    let [time, height, width] = grid;
    if merge == 0 || height % merge != 0 || width % merge != 0 {
        return Err(format!(
            "the grid {grid:?} is not whole {merge}x{merge} blocks"
        ));
    }
    Ok(u64::from(time) * u64::from(height / merge) * u64::from(width / merge))
}

/// The checkpoint's text encoder configuration.
fn read_config(root: &Path) -> Result<TextEncoderConfig, String> {
    let metadata = read_json(&root.join("config.json"))?;
    let processor = read_json(&root.join("preprocessor_config.json"))?;
    TextEncoderConfig::parse(&metadata, &processor).map_err(problem)
}

/// One JSON file of the checkpoint.
fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("parsing {}: {error}", path.display()))
}

/// The case's JSON.
fn read_case(path: &Path) -> Result<Case, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("parsing the case: {error}"))
}

/// `--case`, `--out` and the optional `--text-encoder`.
fn arguments() -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let mut case = None;
    let mut out = None;
    let mut checkpoint = PathBuf::from(DEFAULT_CHECKPOINT);
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--case" => case = arguments.next().map(PathBuf::from),
            "--out" => out = arguments.next().map(PathBuf::from),
            "--text-encoder" => {
                checkpoint = arguments
                    .next()
                    .map(PathBuf::from)
                    .ok_or("--text-encoder needs a directory")?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let case = case.ok_or("--case <case.json> is required")?;
    let out = out.ok_or("--out <dir> is required")?;
    Ok((case, out, checkpoint))
}

fn problem(error: minimax_h3::Error) -> String {
    error.to_string()
}

/// A host table as a NumPy `.npy` v1.0 file: little-endian, row-major, one flat buffer.
///
/// The header's dictionary and its padding are the format's: magic, version, a `u16` header length
/// whose total up to the data is a multiple of 64 bytes, `descr`, `fortran_order`, `shape`, then the
/// entries in C order.
fn npy(dtype: DType, shape: &[u64], bytes: &[u8]) -> Result<Vec<u8>, String> {
    let descriptor = match dtype.bits() {
        32 if matches!(dtype, DType::I32) => "<i4",
        32 if matches!(dtype, DType::F32) => "<f4",
        other => return Err(format!("a {other}-bit table has no .npy descriptor here")),
    };
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
    let length =
        u16::try_from(header.len()).map_err(|_| "a header longer than a u16".to_owned())?;
    file.extend_from_slice(&length.to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(bytes);
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_npy_header_pads_to_a_multiple_of_64_bytes() {
        let bytes = vec![0_u8; 24];
        let file = npy(DType::I32, &[3, 2], &bytes).expect("a table");
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
        let file = npy(DType::F32, &[1], &[0; 4]).expect("a table");
        let length = u16::from_le_bytes([file[8], file[9]]) as usize;
        let header = std::str::from_utf8(&file[10..10 + length - 1]).expect("ascii");
        assert!(header.contains("'shape': (1,)"), "{header}");
        assert!(header.contains("'descr': '<f4'"));
    }

    /// The released checkpoint's own configuration, when the assets are present.
    fn released() -> Option<TextEncoderConfig> {
        read_config(Path::new(DEFAULT_CHECKPOINT)).ok()
    }

    #[test]
    fn a_prompts_blocks_follow_its_placeholder_runs() {
        let Some(config) = released() else {
            return;
        };
        let image = config.image_token_id;
        let video = config.video_token_id;
        // Three prompt tokens, one 4x4 image (four placeholders), then a two-step 4x4 video whose
        // two runs of four placeholders sit behind a timestamp token each.
        let case = Case {
            ids: vec![
                1, image, image, image, image, 2, video, video, video, video, 3, video, video,
                video, video, 4,
            ],
            image_grids: vec![[1, 4, 4]],
            video_grids: vec![[2, 4, 4]],
        };
        let layout = layout_of(&config, &case).expect("a well formed prompt");
        assert_eq!(layout.tokens, 16);
        assert_eq!(
            layout.images,
            vec![
                ImageShape { grid: [1, 4, 4] },
                ImageShape { grid: [2, 4, 4] }
            ]
        );
        // A run one short of its grid is refused, and a grid with no run is refused.
        let short = Case {
            ids: vec![1, image, image, image],
            image_grids: vec![[1, 4, 4]],
            video_grids: Vec::new(),
        };
        assert!(layout_of(&config, &short).is_err());
        let unused = Case {
            ids: vec![1, 2],
            image_grids: vec![[1, 4, 4]],
            video_grids: Vec::new(),
        };
        assert!(layout_of(&config, &unused).is_err());
    }

    #[test]
    fn the_tables_are_the_encoders_own_seven() {
        let Some(config) = released() else {
            return;
        };
        let image = config.image_token_id;
        let case = Case {
            ids: vec![1, image, image, image, image, 2],
            image_grids: vec![[1, 4, 4]],
            video_grids: Vec::new(),
        };
        let layout = layout_of(&config, &case).expect("a well formed prompt");
        let tables = table_bytes(&config, &case.ids, &layout).expect("host tables");
        let names = tables
            .iter()
            .map(|(table, _)| table.name)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "prompt_positions",
                "vision_destinations",
                "vision_position_index",
                "vision_position_weight",
                "vision_cosine",
                "vision_sine",
                "vision_lengths",
            ]
        );
        for (table, bytes) in &tables {
            let width = u64::from(table.dtype.bits() / 8);
            assert_eq!(
                bytes.len() as u64,
                table.shape.numel() * width,
                "{}",
                table.name
            );
        }
        // Four placeholders, four merged rows; one time step, one attention sequence.
        assert_eq!(tables[1].0.shape.dims(), &[4]);
        assert_eq!(tables[6].0.shape.dims(), &[2]);
    }
}
