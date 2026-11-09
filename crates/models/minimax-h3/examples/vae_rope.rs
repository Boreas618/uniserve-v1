//! The video decoder's rotary spectrum and axis order, written for comparison with the reference.
//!
//! The decoder rotates 48 of its 64-element heads split-half, in three sectioned blocks of 16 columns,
//! and its factors are a `[tokens, 24]` table: **one 8-frequency spectrum per axis**, each axis's block
//! contiguous (`video_vae::inverse_frequencies`, `video_vae::rotary`). That arrangement is the crate's
//! reading of two reference facts:
//!
//! - `AutoencoderKLMiniMaxH3`'s decoder builds `RotaryEmbedding(int(head_dim · ratio) // 3,
//!   theta=decoder_rope_theta)` - a 16-element axis's worth of rotation, i.e. 8 frequencies
//!   (`uniserve/models/minimax_h3/video_vae.py:547-549`);
//! - its factors flatten the three position axes' values in order (`video_vae.py:605-613`,
//!   `uniserve/nn/rope.py:355-380`), so the same eight frequencies serve each axis and the axes are
//!   contiguous blocks.
//!
//! Both facts are wrong in a way no shape, element type or key check can see: 24 distinct frequencies
//! instead of 8 repeated three times, or the axes in the wrong order, produces a table of the same width
//! whose rotation phases drift with position. The DiT's rotary had exactly that class of defect once - its
//! height and width axes turning at 0.83 and 0.68 of the reference's angle - and only a factors comparison
//! found it.
//!
//! ```text
//! cargo run -q -p minimax-h3 --example vae_rope -- --root $UNISERVE_H3_MODEL \
//!     --out artifacts/h3-vae-rope
//! python crates/models/minimax-h3/tools/compare_rope_factors.py --vae artifacts/h3-vae-rope \
//!     --checkpoint $UNISERVE_H3_MODEL
//! ```

use std::path::{Path, PathBuf};

use minimax_h3::VideoVaeConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut root = None;
    let mut out = PathBuf::new();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--root" => root = Some(PathBuf::from(value()?)),
            "--out" => out = PathBuf::from(value()?),
            other => {
                return Err(format!(
                    "{other} is not a flag this probe takes; it reads --root and --out"
                )
                .into());
            }
        }
    }
    if out.as_os_str().is_empty() {
        eprintln!("name the output directory with --out");
        std::process::exit(2);
    }
    let root = root.ok_or("--root is required")?;
    run(&root, &out).map_err(Into::into)
}

fn run(root: &Path, out: &Path) -> Result<(), String> {
    // The export's own sidecar: the probe reads the numbers the decoder reads rather than restating
    // them, so a checkpoint whose theta or rotated ratio differs cannot silently be compared at the
    // wrong one.
    let path = root.join("vae").join("config.json");
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let config =
        VideoVaeConfig::parse(&values).map_err(|error| format!("{}: {error}", path.display()))?;
    let (rotated, widths) = minimax_h3::video_vae::rotary(&config)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let theta = config.decoder_rope_theta;
    // The crate's own table, from the crate's own function, and the axis each column pair takes
    // (`ops::rope::section_axes`, the host statement of the kernel's rule).
    let table = minimax_h3::video_vae::inverse_frequencies(&config)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let axes = ops::rope::section_axes(&widths);
    if axes.len() != table.len() {
        return Err(format!(
            "{} table values for {} axis entries",
            table.len(),
            axes.len()
        ));
    }
    std::fs::create_dir_all(out).map_err(|error| format!("{}: {error}", out.display()))?;
    let spectrum = table
        .iter()
        .map(|value| format!("{value:.9e}"))
        .collect::<Vec<String>>()
        .join(", ");
    let axes_map = axes
        .iter()
        .map(u32::to_string)
        .collect::<Vec<String>>()
        .join("");
    std::fs::write(
        out.join("vae_rope.json"),
        format!(
            "{{\n  \"pairs\": {},\n  \"axes\": 3,\n  \"theta\": {theta},\n  \
             \"rotated\": [{}, {}, {}],\n  \"rotated_total\": {rotated},\n  \
             \"spectrum_axes\": \"{axes_map}\",\n  \"table\": [{spectrum}]\n}}\n",
            widths[0] / 2,
            widths[0],
            widths[1],
            widths[2],
        ),
    )
    .map_err(|error| format!("{}: {error}", out.display()))?;
    println!(
        "vae     {} frequencies per axis, 3 axes, theta {theta}, rotated {rotated} as {widths:?}",
        widths[0] / 2
    );
    println!("table   {} values, axes {axes_map}", table.len());
    println!("wrote   {}", out.display());
    Ok(())
}
