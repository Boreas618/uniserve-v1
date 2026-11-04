//! The programs this deployment compiles, one `operator<TAB>key` line each.
//!
//! ```text
//! cargo run -p minimax-h3 --example programs -- /opt/uv/models/MiniMax-H3 [samples ...]
//! ```
//!
//! The rows are `notes/programs/minimax-h3-keys.txt`: what a kernel package must provide for
//! `crates/models/minimax-h3`, stated by the model from its own configuration and the same
//! construction its layers launch with, so no package and no fixture has to guess a variant.
//!
//! `samples` are the clip lengths the deployment serves. A row program's descriptor carries its
//! width, so the set depends on them; the default is the one the caller's requests use.
//!
//! ## What is walked, and what is not
//!
//! Each component states its own launches through [`minimax_h3::programs::Programs`]. A component
//! that does not yet state its programs is named on stderr rather than omitted silently: a fixture
//! missing a component's rows reads as complete and refuses at the first launch of that component.

use std::path::PathBuf;

use minimax_h3::audio_vae::AudioDecoder;
use minimax_h3::config::AudioVaeConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let root: PathBuf = arguments
        .next()
        .ok_or("usage: programs <pipeline root> [samples ...]")?
        .into();
    let mut samples: Vec<u32> = Vec::new();
    for argument in arguments {
        samples.push(argument.parse()?);
    }
    if samples.is_empty() {
        samples.push(256);
    }

    let mut rows: Vec<(String, String)> = Vec::new();

    let audio = root.join("audio_vae/config.json");
    if audio.exists() {
        let config = AudioVaeConfig::parse(&read(&audio)?)?;
        rows.extend(AudioDecoder::programs(&config, &samples)?);
    } else {
        eprintln!("not walked: {} has no audio VAE config", root.display());
    }

    for component in ["transformer", "transformer_ref", "vae", "text_encoder"] {
        eprintln!("not walked: {component}");
    }

    rows.sort();
    rows.dedup();
    for (operator, key) in &rows {
        println!("{operator}\t{key}");
    }
    eprintln!("{} rows", rows.len());
    Ok(())
}

fn read(path: &std::path::Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}
