//! The DiT's rotary spectrum and axis order, written for comparison with the reference.
//!
//! The DiT rotates the leading `3 * 2 * rope_freq_dim` channels of each head - 96 of 128 - and its
//! factors are a `[tokens, 48]` table: **one 16-frequency spectrum serving each of the three
//! coordinate axes**, the axes in contiguous blocks (`attention::table`, `transformer.rs:61-69`).
//! That arrangement is the crate's reading of two reference facts:
//!
//! - `Denoiser` builds `RotaryEmbedding(2 * rope_freq_dim)` (`uniserve/models/minimax_h3/denoiser.py:231`),
//!   a 32-element head's worth of rotation, i.e. 16 frequencies;
//! - the attention applies that one table to each of the three coordinates
//!   (`attention.py:94-105`, `axis_dims=(head_dim,)`), and its factors flatten `[axes, tokens,
//!   dim / 2]` axis-major (`uniserve/nn/rope.py:355-380`).
//!
//! Both facts are wrong in a way no shape, type or count can see: a spectrum of 48 distinct
//! frequencies instead of 16 repeated, or the axes in the wrong order, produces a table of the same
//! width whose rotation phases drift with position. The DiT's rotary had exactly that class of
//! defect once already - its height and width axes turning at 0.83 and 0.68 of the reference's
//! angle - and only a factors comparison found it.
//!
//! ```text
//! cargo run -q -p minimax-h3 --example dit_rope -- --root $UNISERVE_H3_MODEL \
//!     --out artifacts/h3-dit-rope
//! python crates/models/minimax-h3/tools/compare_rope_factors.py --dit artifacts/h3-dit-rope \
//!     --checkpoint $UNISERVE_H3_MODEL
//! ```

use std::path::{Path, PathBuf};

use minimax_h3::config::Config;

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
    let config =
        Config::read(root, None).map_err(|error| format!("{}: {error}", root.display()))?;
    // The base DiT is the network every task's request binds; a checkpoint that holds only the
    // conditioned one would have no base and this probe would have nothing to compare.
    let (name, denoiser) = config
        .denoisers
        .iter()
        .find(|(name, _)| format!("{name:?}").to_lowercase().contains("base"))
        .or_else(|| config.denoisers.iter().next())
        .ok_or("the checkpoint states no DiT")?;
    println!("net     {name:?}");
    let transformer = &denoiser.transformer;
    let pairs = transformer.rope_frequency_dim;
    let theta = transformer.rope_theta;
    // The crate's own tables, from the crate's own functions: the spectrum-and-axes table the
    // operator reads (`attention::table`) and the axis each of its columns takes
    // (`ops::rope::section_axes`, the host statement of the kernel's rule).
    let widths = minimax_h3::transformer::rotated_axes(transformer);
    let table = minimax_h3::attention::table(pairs, 3, theta);
    let axes = ops::rope::section_axes(&widths);
    if axes.len() != table.len() {
        return Err(format!(
            "{} table values for {} axis entries",
            table.len(),
            axes.len()
        ));
    }
    std::fs::create_dir_all(out).map_err(|error| format!("{}: {error}", out.display()))?;
    let values = table
        .iter()
        .map(|value| format!("{value:.9e}"))
        .collect::<Vec<String>>()
        .join(", ");
    let axes = axes
        .iter()
        .map(u32::to_string)
        .collect::<Vec<String>>()
        .join("");
    std::fs::write(
        out.join("dit_rope.json"),
        format!(
            "{{\n  \"pairs\": {pairs},\n  \"axes\": 3,\n  \"theta\": {theta},\n  \
             \"rotated\": [{}, {}, {}],\n  \"spectrum_axes\": \"{axes}\",\n  \
             \"table\": [{values}]\n}}\n",
            widths[0], widths[1], widths[2],
        ),
    )
    .map_err(|error| format!("{}: {error}", out.display()))?;
    println!("dit     {pairs} frequencies per axis, 3 axes, theta {theta}, rotated {widths:?}");
    println!("table   {} values, axes {axes}", table.len());
    println!("wrote   {}", out.display());
    Ok(())
}
