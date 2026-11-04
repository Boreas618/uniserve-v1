//! Compare one request's layout tables against a dump of today's implementation.
//!
//! Run it with
//!
//! ```text
//! cargo run -p minimax-h3 --example compare_blocks -- --dumps <dir> [--out <report.json>]
//! ```
//!
//! `<dir>` holds `case.json`, which describes the request whose layout is compared, and one
//! `.h3t` file per tensor. `crates/models/minimax-h3/tools/dump_tables.py` writes both from
//! `uniserve/models/minimax_h3/packing.py`, today's implementation; `notes/h3/numeric-diagnosis.md`
//! holds the case format, the tensor container and what each name means.
//!
//! This is a **diagnosis, never an acceptance gate** (specs/worker.md 10): every tensor's numbers
//! are reported and nothing is gated on them. A run fails only when the dump itself cannot be read,
//! because then there is nothing to diagnose.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::config::Canvas;
use minimax_h3::packing::{self, Condition, Frame, Role, SegmentExtent};
use serde::Deserialize;
use serde_json::json;

/// The most differing entries one tensor's report lists.
const FIRST_MISMATCHES: usize = 5;

fn main() -> ExitCode {
    match run() {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(problem) => {
            eprintln!("compare_blocks: {problem}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<String, String> {
    let mut arguments = std::env::args().skip(1);
    let mut dumps: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--dumps" => dumps = arguments.next().map(PathBuf::from),
            "--out" => out = arguments.next().map(PathBuf::from),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let dumps = dumps.ok_or("--dumps <dir> names the dump directory to compare against")?;
    let case: Case = serde_json::from_str(
        &fs::read_to_string(dumps.join("case.json"))
            .map_err(|error| format!("{}: {error}", dumps.join("case.json").display()))?,
    )
    .map_err(|error| format!("case.json: {error}"))?;

    let canvas = case.canvas.canvas()?;
    let conditions: Vec<Condition> = case
        .conditions
        .iter()
        .map(|condition| condition.condition())
        .collect::<Result<_, _>>()?;
    let segments = packing::condition_segments(&conditions, &canvas).map_err(problem)?;
    let spans: Vec<(u32, u32)> = case.vision_spans.iter().copied().collect();

    let mut tensors: Vec<Named> = Vec::new();
    match case.attention {
        Attention::Dense => {
            let plan = packing::dense_packing(
                case.num_frames,
                &canvas,
                case.text_capacity,
                case.condition_capacity,
                case.token_multiple,
            )
            .map_err(problem)?;
            let tables = packing::dense_tables(
                &plan,
                case.num_frames,
                &canvas,
                case.num_text_tokens,
                &segments,
                &spans,
            )
            .map_err(problem)?;
            tensors.extend([
                Named::floats(
                    "dense.position_ids",
                    &[tables.position_ids.len() as u64, 3],
                    tables.position_ids.iter().flatten().copied().collect(),
                ),
                Named::signed(
                    "dense.token_tags",
                    &[tables.token_tags.len() as u64],
                    tags(&tables.token_tags),
                ),
                Named::signed(
                    "dense.groups",
                    &[tables.groups.len() as u64],
                    tags(&tables.groups),
                ),
                Named::signed(
                    "dense.prefix_index",
                    &[tables.prefix_index.len() as u64],
                    tags(&tables.prefix_index),
                ),
                Named::scalar("dense.video_rows", i64::from(plan.video_rows)),
                Named::scalar("dense.audio_rows", i64::from(plan.audio_rows)),
                Named::scalar("dense.prefix_start", i64::from(plan.prefix_start())),
                Named::scalar("dense.zero_row", i64::from(plan.zero_row())),
                Named::scalar("dense.padded_tokens", i64::from(plan.padded_tokens)),
                Named::scalar("dense.used", i64::from(tables.used)),
            ]);
        }
        Attention::Sparse {
            tile,
            sparsity,
            reference_keep,
        } => {
            let plan = packing::segment_packing(
                case.num_frames,
                &canvas,
                case.text_capacity,
                case.condition_capacity,
                tile,
                case.token_multiple,
            )
            .map_err(problem)?;
            let tables = packing::segment_tables(
                &plan,
                case.num_frames,
                &canvas,
                case.num_text_tokens,
                &segments,
                &spans,
                sparsity,
                reference_keep,
            )
            .map_err(problem)?;
            let extent: &SegmentExtent = &plan.extent;
            tensors.extend([
                Named::floats(
                    "segment.position_ids",
                    &[tables.position_ids.len() as u64, 3],
                    tables.position_ids.iter().flatten().copied().collect(),
                ),
                Named::signed(
                    "segment.token_tags",
                    &[tables.token_tags.len() as u64],
                    tags(&tables.token_tags),
                ),
                Named::signed(
                    "segment.groups",
                    &[tables.groups.len() as u64],
                    tags(&tables.groups),
                ),
                Named::signed(
                    "segment.prefix_index",
                    &[tables.prefix_index.len() as u64],
                    tags(&tables.prefix_index),
                ),
                Named::signed(
                    "segment.variable_block_sizes",
                    &[tables.variable_block_sizes.len() as u64],
                    tables
                        .variable_block_sizes
                        .iter()
                        .map(|value| i64::from(*value))
                        .collect(),
                ),
                Named::signed(
                    "segment.tile_segments",
                    &[tables.tile_segments.len() as u64],
                    tables
                        .tile_segments
                        .iter()
                        .map(|value| i64::from(*value))
                        .collect(),
                ),
                Named::signed(
                    "segment.segment_starts",
                    &[tables.segment_starts.len() as u64],
                    tables
                        .segment_starts
                        .iter()
                        .map(|value| i64::from(*value))
                        .collect(),
                ),
                Named::signed(
                    "segment.segment_keep",
                    &[tables.segment_keep.len() as u64],
                    tables
                        .segment_keep
                        .iter()
                        .map(|value| i64::from(*value))
                        .collect(),
                ),
                Named::signed(
                    "segment.video_indices",
                    &[plan.video_indices.len() as u64],
                    plan.video_indices
                        .iter()
                        .map(|value| i64::from(*value))
                        .collect(),
                ),
                Named::signed(
                    "segment.video_raster_indices",
                    &[plan.video_raster_indices.len() as u64],
                    plan.video_raster_indices
                        .iter()
                        .map(|value| i64::from(*value))
                        .collect(),
                ),
                Named::scalar("segment.audio_rows", i64::from(extent.audio_rows)),
                Named::scalar("segment.audio_start", i64::from(extent.audio_start())),
                Named::scalar("segment.video_tiles", i64::from(extent.video_tiles)),
                Named::scalar("segment.video_start", i64::from(extent.video_start())),
                Named::scalar("segment.padded_tokens", i64::from(extent.padded_tokens)),
            ]);
        }
    }

    let mut report = Report {
        dumps: dumps.clone(),
        tensors: Vec::new(),
        missing: Vec::new(),
        extra: Vec::new(),
    };
    let mut seen = Vec::new();
    for named in &tensors {
        seen.push(named.name.to_owned());
        let path = dumps.join(format!("{}.h3t", named.name));
        if !path.is_file() {
            report.missing.push(named.name.to_owned());
            continue;
        }
        let dumped = read_dump(&path)?;
        report.tensors.push(compare(named, &dumped));
    }
    for entry in fs::read_dir(&dumps).map_err(|error| format!("{}: {error}", dumps.display()))? {
        let entry = entry.map_err(|error| format!("{}: {error}", dumps.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(name) = name.strip_suffix(".h3t") {
            if !seen.iter().any(|seen| seen == name) {
                report.extra.push(name.to_owned());
            }
        }
    }

    let text = report.render();
    if let Some(out) = out {
        fs::write(&out, &text).map_err(|error| format!("{}: {error}", out.display()))?;
    }
    Ok(text)
}

/// One tensor this run computed, ready to compare.
struct Named {
    name: &'static str,
    shape: Vec<u64>,
    values: Values,
}

impl Named {
    fn floats(name: &'static str, shape: &[u64], values: Vec<f64>) -> Self {
        Self {
            name,
            shape: shape.to_vec(),
            values: Values::Float(values),
        }
    }

    fn signed(name: &'static str, shape: &[u64], values: Vec<i64>) -> Self {
        Self {
            name,
            shape: shape.to_vec(),
            values: Values::Signed(values),
        }
    }

    fn scalar(name: &'static str, value: i64) -> Self {
        Self::signed(name, &[1], vec![value])
    }
}

/// A dumped tensor's entries, converted to the two kinds the tables are made of.
enum Values {
    Signed(Vec<i64>),
    Float(Vec<f64>),
}

impl Values {
    fn len(&self) -> usize {
        match self {
            Self::Signed(values) => values.len(),
            Self::Float(values) => values.len(),
        }
    }
}

fn tags(values: &[i32]) -> Vec<i64> {
    values.iter().map(|value| i64::from(*value)).collect()
}

fn problem(error: minimax_h3::Error) -> String {
    error.to_string()
}

/// One case's layout description, as `tools/dump_tables.py` writes it.
#[derive(Deserialize)]
struct Case {
    canvas: Size,
    num_frames: u32,
    num_text_tokens: u32,
    text_capacity: u32,
    condition_capacity: u32,
    token_multiple: u32,
    attention: Attention,
    #[serde(default)]
    conditions: Vec<CaseCondition>,
    #[serde(default)]
    vision_spans: Vec<(u32, u32)>,
}

#[derive(Deserialize, Clone, Copy)]
struct Size {
    width: u32,
    height: u32,
}

impl Size {
    fn canvas(self) -> Result<Canvas, String> {
        if self.width == 0 || self.height == 0 {
            return Err("a canvas has a positive width and height".to_owned());
        }
        let divisor = gcd(self.width, self.height);
        Ok(Canvas {
            short_edge: self.width.min(self.height),
            aspect: (self.width / divisor, self.height / divisor),
            width: self.width,
            height: self.height,
        })
    }
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let rest = left % right;
        left = right;
        right = rest;
    }
    left.max(1)
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Attention {
    Dense,
    Sparse {
        tile: u32,
        sparsity: f32,
        #[serde(default)]
        reference_keep: Option<f32>,
    },
}

#[derive(Deserialize)]
struct CaseCondition {
    role: String,
    #[serde(default)]
    canvas: Option<Size>,
    #[serde(default)]
    frames: Option<u32>,
    #[serde(default)]
    audio_samples: u32,
}

impl CaseCondition {
    fn condition(&self) -> Result<Condition, String> {
        let role = match self.role.as_str() {
            "first_frame" => Role::FirstFrame,
            "last_frame" => Role::LastFrame,
            "reference" => Role::Reference,
            other => return Err(format!("unknown condition role {other}")),
        };
        let frame = match self.canvas {
            None => None,
            Some(canvas) => Some(Frame {
                canvas: canvas.canvas()?,
                num_frames: self.frames.unwrap_or(1),
            }),
        };
        Ok(Condition {
            role,
            frame,
            audio_samples: self.audio_samples,
        })
    }
}

/// One dumped tensor, in the container `notes/h3/numeric-diagnosis.md` defines.
struct Dump {
    shape: Vec<u64>,
    values: Values,
}

fn read_dump(path: &Path) -> Result<Dump, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let broken = |problem: &str| format!("{}: {problem}", path.display());
    if bytes.len() < 6 || &bytes[..4] != b"H3D1" {
        return Err(broken("not an H3D1 tensor"));
    }
    let dtype = bytes[4];
    let rank = bytes[5] as usize;
    let header = 6 + 8 * rank;
    if bytes.len() < header {
        return Err(broken("a truncated header"));
    }
    let mut shape = Vec::with_capacity(rank);
    for dimension in 0..rank {
        let start = 6 + 8 * dimension;
        let mut word = [0_u8; 8];
        word.copy_from_slice(&bytes[start..start + 8]);
        shape.push(u64::from_le_bytes(word));
    }
    let payload = &bytes[header..];
    let count: u64 = shape.iter().product();
    let values = match dtype {
        0 => {
            let expected = count as usize * 8;
            if payload.len() != expected {
                return Err(broken("a payload that does not match its shape"));
            }
            Values::Float(
                payload
                    .chunks_exact(8)
                    .map(|chunk| {
                        let mut word = [0_u8; 8];
                        word.copy_from_slice(chunk);
                        f64::from_le_bytes(word)
                    })
                    .collect(),
            )
        }
        1 => {
            let expected = count as usize * 4;
            if payload.len() != expected {
                return Err(broken("a payload that does not match its shape"));
            }
            Values::Float(
                payload
                    .chunks_exact(4)
                    .map(|chunk| {
                        let mut word = [0_u8; 4];
                        word.copy_from_slice(chunk);
                        f64::from(f32::from_le_bytes(word))
                    })
                    .collect(),
            )
        }
        2 => Values::Signed(read_signed(payload, 8, &shape).map_err(|error| broken(&error))?),
        3 => Values::Signed(read_signed(payload, 4, &shape).map_err(|error| broken(&error))?),
        other => return Err(broken(&format!("unknown dtype {other}"))),
    };
    Ok(Dump { shape, values })
}

fn read_signed(payload: &[u8], width: usize, shape: &[u64]) -> Result<Vec<i64>, String> {
    let count: u64 = shape.iter().product();
    if payload.len() != count as usize * width {
        return Err("a payload that does not match its shape".to_owned());
    }
    Ok(payload
        .chunks_exact(width)
        .map(|chunk| {
            if width == 8 {
                let mut word = [0_u8; 8];
                word.copy_from_slice(chunk);
                i64::from_le_bytes(word)
            } else {
                let mut word = [0_u8; 4];
                word.copy_from_slice(chunk);
                i64::from(i32::from_le_bytes(word))
            }
        })
        .collect())
}

/// One tensor's comparison: the numbers, and the first places they differ.
struct Comparison {
    name: String,
    shape: Vec<u64>,
    dumped_shape: Vec<u64>,
    entries: usize,
    /// Entries that differ at all, exactly for integers and bitwise for floats.
    differing: usize,
    largest: f64,
    largest_relative: f64,
    first: Vec<[String; 3]>,
}

fn compare(named: &Named, dumped: &Dump) -> Comparison {
    let mut comparison = Comparison {
        name: named.name.to_owned(),
        shape: named.shape.clone(),
        dumped_shape: dumped.shape.clone(),
        entries: named.values.len(),
        differing: 0,
        largest: 0.0,
        largest_relative: 0.0,
        first: Vec::new(),
    };
    if named.shape != dumped.shape {
        return comparison;
    }
    for index in 0..named.values.len() {
        let (difference, shown) = match (&named.values, &dumped.values) {
            (Values::Signed(mine), Values::Signed(theirs)) => {
                let difference = (mine[index] - theirs[index]).abs();
                (
                    difference as f64,
                    [
                        index.to_string(),
                        theirs[index].to_string(),
                        mine[index].to_string(),
                    ],
                )
            }
            (Values::Float(mine), Values::Float(theirs)) => {
                let difference = (mine[index] - theirs[index]).abs();
                let scale = theirs[index].abs().max(f64::MIN_POSITIVE);
                comparison.largest_relative = comparison.largest_relative.max(difference / scale);
                (
                    difference,
                    [
                        index.to_string(),
                        format!("{:.17e}", theirs[index]),
                        format!("{:.17e}", mine[index]),
                    ],
                )
            }
            _ => continue,
        };
        if difference > 0.0 {
            comparison.differing += 1;
            if comparison.first.len() < FIRST_MISMATCHES {
                comparison.first.push(shown);
            }
            comparison.largest = comparison.largest.max(difference);
        }
    }
    comparison
}

/// The whole run's report, printed and optionally written as JSON.
struct Report {
    dumps: PathBuf,
    tensors: Vec<Comparison>,
    missing: Vec<String>,
    extra: Vec<String>,
}

impl Report {
    fn render(&self) -> String {
        let mut text = String::new();
        text.push_str(&format!(
            "# H3 layout diagnosis: {}\n\n",
            self.dumps.display()
        ));
        text.push_str(
            "Diagnosis, not an acceptance gate (specs/worker.md 10): the numbers are recorded as\n\
             they are. `differing` counts entries that are not equal, exactly for the integer\n\
             tables and bitwise for the coordinates; `largest` and `largest_relative` are the\n\
             worst absolute and relative differences of the coordinates.\n\n",
        );
        text.push_str("| tensor | shape | entries | differing | largest | largest relative |\n");
        text.push_str("|---|---|---|---|---|---|\n");
        for tensor in &self.tensors {
            let shape = if tensor.shape == tensor.dumped_shape {
                shape_text(&tensor.shape)
            } else {
                format!(
                    "{} vs {}",
                    shape_text(&tensor.dumped_shape),
                    shape_text(&tensor.shape)
                )
            };
            text.push_str(&format!(
                "| `{}` | {} | {} | {} | {:.3e} | {:.3e} |\n",
                tensor.name,
                shape,
                tensor.entries,
                tensor.differing,
                tensor.largest,
                tensor.largest_relative
            ));
        }
        for tensor in &self.tensors {
            if tensor.first.is_empty() {
                continue;
            }
            text.push_str(&format!(
                "\n`{}` first differences (index, current, ours):\n",
                tensor.name
            ));
            for entry in &tensor.first {
                text.push_str(&format!("- {} {} {}\n", entry[0], entry[1], entry[2]));
            }
        }
        if !self.missing.is_empty() {
            text.push_str(&format!("\nno dump for: {}\n", self.missing.join(", ")));
        }
        if !self.extra.is_empty() {
            text.push_str(&format!(
                "\ndumped but not produced by this case: {}\n",
                self.extra.join(", ")
            ));
        }
        if self
            .tensors
            .iter()
            .any(|tensor| tensor.shape != tensor.dumped_shape)
        {
            text.push_str("\nA tensor's shape differs, which is a structural difference.\n");
        }
        text.push_str(&format!(
            "\n```json\n{}\n```\n",
            serde_json::to_string_pretty(&json!({
                "dumps": self.dumps.display().to_string(),
                "tensors": self.tensors.iter().map(|tensor| json!({
                    "name": &tensor.name,
                    "shape": &tensor.shape,
                    "dumped_shape": &tensor.dumped_shape,
                    "entries": tensor.entries,
                    "differing": tensor.differing,
                    "largest": tensor.largest,
                    "largest_relative": tensor.largest_relative,
                })).collect::<Vec<_>>(),
                "missing": &self.missing,
                "extra": &self.extra,
            }))
            .unwrap_or_default()
        ));
        text
    }
}

fn shape_text(shape: &[u64]) -> String {
    shape
        .iter()
        .map(|dimension| dimension.to_string())
        .collect::<Vec<_>>()
        .join("x")
}
