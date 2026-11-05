//! The text encoder's M-RoPE factors, written for comparison with the reference.
//!
//! The prompt's rotation reads two tables the encode computes **on the host**: the interleaved
//! M-RoPE coordinates of every token (`positions::rope_index`) and the cosine and sine those
//! coordinates give (`Mrope::factors`, 48-40-40 sections over a 128-element head). Both are pure
//! arithmetic over the prompt's ids and the checkpoint's configuration, so both can be compared
//! without a device - and they must be, because a wrong spectrum is invisible to every other check:
//! the tables have the right shape, the right type and the right count, and the rotation turns the
//! wrong way by an amount that reads as a tolerance problem in a page.
//!
//! The DiT's rotary had exactly that defect and only a factors comparison found it (its height and
//! width axes turned at 0.83 and 0.68 of the reference's angle). The text encoder's is a different
//! arrangement - the reference cycles one spectrum over three axes instead of giving each axis its
//! own band - so its coordinates agreeing does not make its factors agree.
//!
//! ```text
//! cargo run -q -p minimax-h3 --example encoder_rope -- --root /workspace/models/MiniMax-H3 \
//!     --prompt-ids 32,2518,38835,8473,3941,264,89773,2070,518,38393,11,8413,3100,11,6884 \
//!     --out artifacts/h3-stage4/rope
//! .venv/bin/python crates/models/minimax-h3/tools/compare_rope_factors.py \
//!     --case artifacts/h3-stage4/rope --checkpoint /workspace/models/MiniMax-H3
//! ```
//!
//! The first writes `positions.h3t` (`[3, tokens]` int32), `cosine.h3t` and `sine.h3t`
//! (`[tokens, head_dim / 2]` f32) and `role.json` (the ids, the sections and the frequency base);
//! the second recomputes all three from the ids alone, with the reference's own `rope_index` and
//! `RotaryEmbedding`, and prints the worst element of each.

use std::path::{Path, PathBuf};

use minimax_h3::config::Config;
use minimax_h3::positions::{self, Mrope};

/// The container's dtype codes, shared with every other dump this crate writes.
const F32: u8 = 1;
const I32: u8 = 3;

/// The prompt a stage-4 case carries when it states none.
const DEFAULT_IDS: [u32; 15] = [
    32, 2518, 38835, 8473, 3941, 264, 89773, 2070, 518, 38393, 11, 8413, 3100, 11, 6884,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut root = None;
    let mut out = PathBuf::from("artifacts/h3-stage4/rope");
    let mut ids: Option<Vec<u32>> = None;
    let mut grids: Vec<[u32; 3]> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--root" => root = Some(PathBuf::from(value()?)),
            "--out" => out = PathBuf::from(value()?),
            // A prompt with one image block states its grid here; the ids must carry that block's
            // placeholders, and `rope_index` refuses a run whose counts disagree. Without a grid the
            // three axes are equal on every token, which is why a text-only case cannot exercise the
            // interleaving - the axes' own coordinates are what the assignment orders.
            "--image-grid" => {
                let list = value()?;
                let parts = list
                    .split(',')
                    .map(|field| field.trim().parse::<u32>())
                    .collect::<Result<Vec<u32>, _>>()
                    .map_err(|error| format!("--image-grid {list:?}: {error}"))?;
                let [time, height, width] = parts.as_slice() else {
                    return Err("--image-grid takes three extents: time,height,width".into());
                };
                grids.push([*time, *height, *width]);
            }
            "--prompt-ids" => {
                let list = value()?;
                let parsed = list
                    .split(',')
                    .map(|field| {
                        field
                            .trim()
                            .parse::<u32>()
                            .map_err(|error| format!("--prompt-ids: {field:?}: {error}"))
                    })
                    .collect::<Result<Vec<u32>, String>>()?;
                ids = Some(parsed);
            }
            other => {
                return Err(format!(
                    "{other} is not a flag this probe takes; it reads --root, --prompt-ids and --out"
                )
                .into());
            }
        }
    }
    let root = root.ok_or("--root is required")?;
    let ids = ids.unwrap_or_else(|| DEFAULT_IDS.to_vec());
    run(&root, &ids, &grids, &out).map_err(Into::into)
}

/// Write the factors of `ids` under `out`.
///
/// The tables are the ones `TextEncoder::write_state` fills for a layout of this prompt, computed
/// through the same public functions - no second implementation of the spectrum lives here.
fn run(root: &Path, ids: &[u32], grids: &[[u32; 3]], out: &Path) -> Result<(), String> {
    let config =
        Config::read(root, None).map_err(|error| format!("{}: {error}", root.display()))?;
    let text = &config.text_encoder;
    let mrope = Mrope::new(text.mrope_sections, text.head_dim)
        .map_err(|error| format!("the checkpoint's own sections: {error}"))?;
    let frequencies = positions::inverse_frequencies(text.head_dim, text.rope_theta)
        .map_err(|error| format!("the checkpoint's frequency table: {error}"))?;
    // A prompt with an image states its grid; the placeholder run it addresses is the prompt's own
    // business, and `rope_index` refuses a prompt whose placeholders and grids disagree.
    let coordinates = positions::rope_index(
        ids,
        grids,
        &[],
        text.image_token_id,
        text.video_token_id,
        text.vision.spatial_merge_size,
    )
    .map_err(|error| format!("the prompt's coordinates: {error}"))?;
    let columns = (text.head_dim / 2) as usize;
    let mut cosine = vec![0.0_f32; ids.len() * columns];
    let mut sine = vec![0.0_f32; ids.len() * columns];
    mrope
        .factors(&coordinates, &frequencies, &mut cosine, &mut sine)
        .map_err(|error| format!("the prompt's factors: {error}"))?;

    std::fs::create_dir_all(out).map_err(|error| format!("{}: {error}", out.display()))?;
    let mut axis_major = Vec::with_capacity(ids.len() * 3);
    for axis in 0..3 {
        for coordinate in &coordinates {
            axis_major.extend_from_slice(&(coordinate[axis] as i32).to_le_bytes());
        }
    }
    write(
        &out.join("positions.h3t"),
        &[3, ids.len() as u64],
        I32,
        &axis_major,
    )?;
    write(
        &out.join("cosine.h3t"),
        &[ids.len() as u64, columns as u64],
        F32,
        &as_bytes(&cosine),
    )?;
    write(
        &out.join("sine.h3t"),
        &[ids.len() as u64, columns as u64],
        F32,
        &as_bytes(&sine),
    )?;
    let listed = ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<String>>()
        .join(", ");
    std::fs::write(
        out.join("role.json"),
        format!(
            "{{\n  \"prompt_ids\": [{listed}],\n  \"head_dim\": {},\n  \"sections\": [{}, {}, {}],\n  \
             \"theta\": {},\n  \"merge\": {},\n  \"image_grids\": {:?}\n}}\n",
            text.head_dim,
            text.mrope_sections[0],
            text.mrope_sections[1],
            text.mrope_sections[2],
            text.rope_theta,
            text.vision.spatial_merge_size,
            grids,
        ),
    )
    .map_err(|error| format!("{}: {error}", out.display()))?;
    // The first token's factors are printed because they are the only ones a reader can check by
    // hand: token 0 is at coordinate 0 on every axis, so every factor is exactly cos 0 = 1 and
    // sin 0 = 0 - a table that disagrees there is an offset, not a spectrum.
    println!(
        "rope    {} tokens, {} compact frequencies, sections {:?}, theta {}",
        ids.len(),
        columns,
        text.mrope_sections,
        text.rope_theta
    );
    println!(
        "axes    {:?}",
        mrope
            .frequency_axes()
            .iter()
            .map(|axis| axis.to_string())
            .collect::<Vec<String>>()
            .join("")
    );
    println!(
        "token 0 cos {:?} sin {:?}",
        &cosine[..3.min(columns)],
        &sine[..3.min(columns)]
    );
    println!("wrote   {}", out.display());
    Ok(())
}

fn as_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Write one tensor in the container the other dumps use: `H3D1`, a dtype code, a rank, the
/// extents, then the payload little-endian.
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
