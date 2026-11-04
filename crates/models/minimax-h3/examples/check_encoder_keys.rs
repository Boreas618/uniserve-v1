//! Check the encoders' tensor names against a checkpoint's own index.
//!
//!     cargo run -p minimax-h3 --example check_encoder_keys -- <pipeline root>
//!
//! The text encoder and the vision tower ask their tensors by name, and a name that the checkpoint
//! spells differently is the one defect a load cannot report usefully: it surfaces as a read error
//! inside a layer, after the weights before it have already been read. This example asks both
//! components for their complete name list and compares it with the export's own
//! `model.safetensors.index.json`, so the answer is three numbers before any weight is touched:
//!
//!   * **asked** - what the components read (`TextEncoder::keys`, `VisionTower::keys`);
//!   * **missing** - asked, and the index does not hold it: a load would fail;
//!   * **unread** - in the index, and neither component asks for it: either a tensor these two do
//!     not read (a head, a projection the tower drops) or a name the walk spells differently.
//!
//! It opens no tensor and needs no device: the index is metadata, so this runs where the shards are
//! not fetched. Unlike `inspect_checkpoint` it needs no contract, only the two config sidecars.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use minimax_h3::checkpoint::{Layout, Sidecars};
use minimax_h3::config::Config;
use minimax_h3::text_encoder::TextEncoder;
use minimax_h3::vision::VisionTower;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root: PathBuf = std::env::args()
        .nth(1)
        .ok_or("usage: check_encoder_keys <pipeline root>")?
        .into();

    let layout = Layout::detect(&root)?;
    let sidecars = Sidecars::read(&root, None, &layout)?;
    let config = Config::normalize(&layout, &sidecars)?;

    let index = index_names(&root.join("text_encoder"))?;
    println!(
        "index: {} tensors under {}",
        index.len(),
        root.join("text_encoder").display()
    );

    let text = TextEncoder::keys(&config.text_encoder);
    let tower = VisionTower::keys(&config.text_encoder.vision);
    let mut unread: BTreeSet<&str> = index.iter().map(String::as_str).collect();
    let mut failed = false;
    for (name, keys) in [("text encoder", &text), ("vision tower", &tower)] {
        let missing: Vec<&str> = keys
            .iter()
            .map(String::as_str)
            .filter(|key| !index.contains(*key))
            .collect();
        for key in keys {
            unread.remove(key.as_str());
        }
        println!("{name}: {} asked, {} missing", keys.len(), missing.len());
        for key in missing.iter().take(12) {
            println!("  missing  {key}");
        }
        failed |= !missing.is_empty();
    }
    println!("unread: {} of the index's tensors", unread.len());
    for key in unread.iter().take(12) {
        println!("  unread   {key}");
    }

    if failed {
        return Err("a name the encoders read is not in the checkpoint".into());
    }
    Ok(())
}

/// Every tensor name the checkpoint's index lists for one component directory.
///
/// The index is the export's own spelling, read as the JSON it is: `Checkpoint::open` would also
/// validate that each shard file is present, which is exactly what a host without the weights
/// cannot do - and the names are what this check is about.
fn index_names(directory: &Path) -> Result<BTreeSet<String>, Box<dyn std::error::Error>> {
    let mut names = BTreeSet::new();
    let mut found = false;
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        let Some(file) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file.ends_with(".safetensors.index.json") {
            continue;
        }
        found = true;
        let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        let map = index
            .get("weight_map")
            .and_then(|map| map.as_object())
            .ok_or_else(|| format!("{} holds no weight_map", path.display()))?;
        names.extend(map.keys().cloned());
    }
    if !found {
        return Err(format!("{} holds no .safetensors.index.json", directory.display()).into());
    }
    Ok(names)
}
