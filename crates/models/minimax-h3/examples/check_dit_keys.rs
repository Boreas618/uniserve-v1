//! Check the DiT block's checkpoint keys against the checkpoint's own index, without a device.
//!
//! ```text
//! cargo run -p minimax-h3 --example check_dit_keys -- <pipeline root>
//! ```
//!
//! `Denoiser::keys` states every tensor the block reads, as complete names, from the same
//! construction its `load` uses. This walks that list against each DiT directory's `.index.json` and
//! reports both directions, which is the half a prefix table cannot see:
//!
//! * a name the block asks for that the index lacks - the load would fail on a missing shard;
//! * a name the index holds under the block's own trees that the block never asks for - a weight
//!   nothing reads, or a tensor the loader misses.
//!
//! The index states every tensor of an export whether or not its shard is present, so both answers
//! are host facts. Nothing is loaded and nothing is allocated: the names are arithmetic over the
//! normalized configuration.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use minimax_h3::checkpoint::{Layout, Sidecars};
use minimax_h3::config::Config;
use minimax_h3::denoiser::Denoiser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root: PathBuf = std::env::args()
        .nth(1)
        .ok_or("usage: check_dit_keys <pipeline root>")?
        .into();
    let layout = Layout::detect(&root)?;
    let sidecars = Sidecars::read(&root, None, &layout)?;
    let config = Config::normalize(&layout, &sidecars)?;

    let mut failed = false;
    for (dit, denoiser) in &config.denoisers {
        let directory = dit.directory();
        let index = index_names(&root.join(directory))?;
        if index.is_empty() {
            println!("{directory}: no .index.json under the root, nothing to check");
            continue;
        }
        let asked: BTreeSet<String> = Denoiser::keys(denoiser).into_iter().collect();
        let missing: Vec<&String> = asked.iter().filter(|name| !index.contains(*name)).collect();
        // The block's own trees. A name the index holds under one of them that no key above names
        // is a tensor nothing reads - or a tensor the loader misses, which is the same failure seen
        // from the other side.
        let extra: Vec<&String> = index
            .iter()
            .filter(|name| !asked.contains(*name))
            .filter(|name| {
                name.starts_with("transformer_blocks.")
                    || name.starts_with("token_refiner.")
                    || name.starts_with("context_embedder.")
                    || name.starts_with("time_embedder.")
                    || name.starts_with("norm_out.")
                    || name.starts_with("proj_in.")
                    || name.starts_with("proj_out.")
                    || name.starts_with("audio_proj_")
            })
            .collect();

        println!(
            "{directory}: {} keys asked, {} in the index, {} missing, {} unread",
            asked.len(),
            index.len(),
            missing.len(),
            extra.len()
        );
        for name in &missing {
            println!("  MISSING {name}");
            failed = true;
        }
        for name in &extra {
            println!("  UNREAD  {name}");
        }
    }
    if failed {
        return Err("a key the block asks for is not in the index".into());
    }
    Ok(())
}

/// Every tensor name a directory's safetensors index states.
fn index_names(directory: &Path) -> Result<BTreeSet<String>, Box<dyn std::error::Error>> {
    let mut names = BTreeSet::new();
    if !directory.is_dir() {
        return Ok(names);
    }
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json")
            || !path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".index.json"))
        {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        let index: serde_json::Value = serde_json::from_str(&text)?;
        if let Some(map) = index["weight_map"].as_object() {
            names.extend(map.keys().cloned());
        }
    }
    Ok(names)
}
