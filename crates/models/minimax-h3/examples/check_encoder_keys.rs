//! Check the encoders' tensor names against a checkpoint's own index.
//!
//!     cargo run -p minimax-h3 --example check_encoder_keys -- <pipeline root>
//!
//! The text encoder, the vision tower it builds inside itself, and the video VAE encoder ask their
//! tensors by name, and a name the checkpoint spells differently is the one defect a load cannot
//! report usefully: it surfaces as a read error inside a layer, after the weights before it have
//! already been read. This example asks each component for its complete name list and compares it
//! with the export's own index files, so the answer is three numbers before any weight is touched:
//!
//!   * **asked** - what the component reads (`TextEncoder::keys`, `VisionTower::keys`,
//!     `VideoEncoder::keys`);
//!   * **missing** - asked, and the index does not hold it: a load would fail;
//!   * **unread** - in the index, and the component does not ask for it: either a tensor it does
//!     not read by design (an unretained layer, an unused head) or a name the walk spells
//!     differently.
//!
//! It opens no tensor and needs no device: the indexes are metadata, so this runs where the shards
//! are not fetched. Unlike `inspect_checkpoint` it needs no contract, only the config sidecars.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use minimax_h3::audio_vae::AudioDecoder;
use minimax_h3::checkpoint::{Layout, Sidecars};
use minimax_h3::config::Config;
use minimax_h3::encoding::{AudioEncoder, VideoEncoder};
use minimax_h3::text_encoder::TextEncoder;
use minimax_h3::video_vae::VideoDecoder;
use minimax_h3::vision::VisionTower;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root: PathBuf = std::env::args()
        .nth(1)
        .ok_or("usage: check_encoder_keys <pipeline root>")?
        .into();

    let layout = Layout::detect(&root)?;
    let sidecars = Sidecars::read(&root, None, &layout)?;
    let config = Config::normalize(&layout, &sidecars)?;

    // The language stack and the tower share one export directory and one index; the video VAE
    // encoder has its own.
    let text_index = index_names(&root.join("text_encoder"))?;
    let vae_index = index_names(&root.join("vae"))?;
    // The audio directory may hold no index at all - this host's export holds its `config.json`
    // and nothing else - and that is a fact about the export, not a reason to skip the count: the
    // lists are still built, and every name in them is reported missing, which is exactly what
    // `load` refuses with.
    let audio_index = match index_names(&root.join("audio_vae")) {
        Ok(index) => index,
        Err(error) => {
            println!("audio_vae: {error}");
            BTreeSet::new()
        }
    };
    let text = TextEncoder::keys(&config.text_encoder);
    let tower = VisionTower::keys(&config.text_encoder.vision);
    let video = VideoEncoder::keys(&config.video_vae)?;
    // The video VAE directory holds both halves: the encoder reads 118 of its 703 names and the
    // decoder the other 585, so the two lists are partners on one index and nothing is unread.
    let decoder = VideoDecoder::keys(&config.video_vae)?;
    // The audio directory holds both halves too, and against this checkpoint none of the tensors
    // either of them asks for is present - which is the answer `load` must give before a stage of
    // the audio path is built, so it is reported here rather than skipped.
    let audio_decoder = AudioDecoder::keys(&config.audio_vae)?;
    let audio_encoder = AudioEncoder::keys(&config.audio_vae)?;

    let mut failed = false;
    report(
        "text encoder",
        &text,
        &text_index,
        &[&text, &tower],
        &mut failed,
    );
    report(
        "vision tower",
        &tower,
        &text_index,
        &[&text, &tower],
        &mut failed,
    );
    report(
        "video VAE encoder",
        &video,
        &vae_index,
        &[&video, &decoder],
        &mut failed,
    );
    report(
        "video VAE decoder",
        &decoder,
        &vae_index,
        &[&video, &decoder],
        &mut failed,
    );
    report(
        "audio VAE decoder",
        &audio_decoder,
        &audio_index,
        &[&audio_decoder, &audio_encoder],
        &mut failed,
    );
    report(
        "audio VAE encoder",
        &audio_encoder,
        &audio_index,
        &[&audio_decoder, &audio_encoder],
        &mut failed,
    );

    if failed {
        return Err("a name a component reads is not in the checkpoint".into());
    }
    Ok(())
}

/// Print one component's answer, and remember whether it asked for a name the index lacks.
///
/// `partners` are the other key lists that read the same index: a tensor a partner asks for is
/// accounted for even when this component does not read it, so each component is reported without
/// the other's tensors looking unexplained.
fn report(
    name: &str,
    keys: &[String],
    index: &BTreeSet<String>,
    partners: &[&Vec<String>],
    failed: &mut bool,
) {
    let missing: Vec<&str> = keys
        .iter()
        .map(String::as_str)
        .filter(|key| !index.contains(*key))
        .collect();
    let mut unread: BTreeSet<&str> = index.iter().map(String::as_str).collect();
    for list in partners {
        for key in list.iter() {
            unread.remove(key.as_str());
        }
    }
    println!(
        "{name}: {} asked, {} missing, index {} tensors",
        keys.len(),
        missing.len(),
        index.len()
    );
    for key in missing.iter().take(12) {
        println!("  missing  {key}");
    }
    println!("  unread   {} of them", unread.len());
    for key in unread.iter().take(12) {
        println!("    unread   {key}");
    }
    *failed |= !missing.is_empty();
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
