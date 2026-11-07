//! The vision tower's 2-D rotary factors, written for comparison with the reference.
//!
//! The tower rotates each patch's head by two coordinates - its row and its column in the merged
//! grid - and the reference builds one `RotaryEmbedding(head_dim / 2)` **without sections**
//! (`uniserve/models/qwen3_vl/vision.py:168`) and calls it with `[patches, 2]`
//! (`:253-256`), then flattens `[patches, 2, head_dim / 4]` to `[patches, head_dim / 2]`. Each axis
//! therefore uses the whole 18-frequency spectrum of a 72-element head, and the two axes' factors
//! are contiguous halves - not the text encoder's interleaved cycle and not the DiT's three blocks,
//! which is why neither of those comparators can stand in for this one.
//!
//! ```text
//! cargo run -q -p minimax-h3 --example vision_rope -- --root /workspace/models/MiniMax-H3 \
//!     --height 4 --width 4 --merge 2 --out artifacts/h3-vision-rope
//! python crates/models/minimax-h3/tools/compare_rope_factors.py --vision \
//!     artifacts/h3-vision-rope --checkpoint /workspace/models/MiniMax-H3
//! ```

use std::path::{Path, PathBuf};

use minimax_h3::config::Config;
use minimax_h3::positions::{self, VisionRotary};

/// The container's dtype codes, shared with every other dump this crate writes.
const F32: u8 = 1;
const I32: u8 = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut root = None;
    let mut out = PathBuf::from("artifacts/h3-vision-rope");
    let (mut height, mut width, mut merge) = (4_u32, 4, 2_u32);
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--root" => root = Some(PathBuf::from(value()?)),
            "--out" => out = PathBuf::from(value()?),
            "--height" => height = value()?.parse()?,
            "--width" => width = value()?.parse()?,
            "--merge" => merge = value()?.parse()?,
            other => {
                return Err(format!(
                    "{other} is not a flag this probe takes; it reads --root, --out, --height, \
                     --width and --merge"
                )
                .into());
            }
        }
    }
    let root = root.ok_or("--root is required")?;
    run(&root, height, width, merge, &out).map_err(Into::into)
}

fn run(root: &Path, height: u32, width: u32, merge: u32, out: &Path) -> Result<(), String> {
    let config =
        Config::read(root, None).map_err(|error| format!("{}: {error}", root.display()))?;
    let vision = &config.text_encoder.vision;
    // The tower states its width and its head count, not a head width; the head is the quotient,
    // and `VisionTower::head_dim` is the crate's own statement of it.
    let head_dim =
        minimax_h3::vision::BlockRows::head_dim(vision).map_err(|error| format!("{error}"))?;
    let rotary =
        VisionRotary::new(head_dim).map_err(|error| format!("the tower's head: {error}"))?;
    let frequencies = positions::inverse_frequencies(head_dim / 2, VISION_THETA)
        .map_err(|error| format!("the tower's frequencies: {error}"))?;
    let coordinates = positions::rotary_coordinates(height, width, merge)
        .map_err(|error| format!("the grid's coordinates: {error}"))?;
    let columns = (head_dim / 2) as usize;
    let mut cosine = vec![0.0_f32; coordinates.len() * columns];
    let mut sine = vec![0.0_f32; coordinates.len() * columns];
    rotary
        .factors(&coordinates, &frequencies, &mut cosine, &mut sine)
        .map_err(|error| format!("the grid's factors: {error}"))?;

    std::fs::create_dir_all(out).map_err(|error| format!("{}: {error}", out.display()))?;
    let mut pairs = Vec::with_capacity(coordinates.len() * 2);
    for coordinate in &coordinates {
        for axis in coordinate {
            pairs.extend_from_slice(&(*axis as i32).to_le_bytes());
        }
    }
    write(
        &out.join("coordinates.h3t"),
        &[coordinates.len() as u64, 2],
        I32,
        &pairs,
    )?;
    write(
        &out.join("cosine.h3t"),
        &[coordinates.len() as u64, columns as u64],
        F32,
        &as_bytes(&cosine),
    )?;
    write(
        &out.join("sine.h3t"),
        &[coordinates.len() as u64, columns as u64],
        F32,
        &as_bytes(&sine),
    )?;
    std::fs::write(
        out.join("vision_rope.json"),
        format!(
            "{{\n  \"head_dim\": {},\n  \"theta\": {VISION_THETA},\n  \"height\": {height},\n  \
             \"width\": {width},\n  \"merge\": {merge},\n  \"patches\": {}\n}}\n",
            head_dim,
            coordinates.len()
        ),
    )
    .map_err(|error| format!("{}: {error}", out.display()))?;
    println!(
        "tower   head {} split into two axes of {} frequencies, theta {VISION_THETA}",
        head_dim,
        head_dim / 4
    );
    println!(
        "grid    {height}x{width} merged by {merge}: {} patches",
        coordinates.len()
    );
    println!("wrote   {}", out.display());
    Ok(())
}

/// The tower's rotary base. The reference states it in the module (`vision.py:46`), not in the
/// checkpoint's configuration, so it is stated here once.
const VISION_THETA: f32 = 10_000.0;

fn as_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn write(path: &Path, shape: &[u64], dtype: u8, payload: &[u8]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(6 + shape.len() * 8 + payload.len());
    bytes.extend_from_slice(b"H3D1");
    bytes.push(dtype);
    bytes.push(u8::try_from(shape.len()).map_err(|_| "a rank above 255".to_owned())?);
    for extent in shape {
        bytes.extend_from_slice(&extent.to_le_bytes());
    }
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
}
