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
use minimax_h3::encoding::AudioEncoder;
use minimax_h3::config::{
    AttentionKind, AudioVaeConfig, TextEncoderConfig, TransformerConfig, VideoVaeConfig,
};
use minimax_h3::partition::Partition;
use minimax_h3::output::VideoPostprocessor;
use minimax_h3::text_encoder::TextEncoder;
use minimax_h3::denoiser::Denoiser;
use minimax_h3::modulation::Groups;
use minimax_h3::video_vae::VideoDecoder;
use minimax_h3::vision::VisionTower;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let root: PathBuf = arguments
        .next()
        .ok_or("usage: programs <pipeline root> --samples a,b --rasters HxW,...")?
        .into();

    // The buckets are the deployment's, never the model's: a row program's key carries its width, so
    // a fixture is only valid for the shapes it was generated at. Nothing here has a default — a walk
    // that invented one would describe a deployment nobody runs.
    let mut samples: Vec<u32> = Vec::new();
    let mut rasters: Vec<(u32, u32)> = Vec::new();
    // The video decoder's precision is the checkpoint's, not the config's: the design converts
    // nothing at load, so the deployment states the element type its weights are in.
    let mut dtype: Option<ops::DType> = None;
    // The prompt lengths the deployment serves, and its tensor-parallel degree: both are the
    // deployment's, and the degree decides the shard widths the store must hold.
    let mut tokens: Vec<u32> = Vec::new();
    let mut tensor_parallel: Option<usize> = None;
    // The packed row count the DiT partitions are compiled for: a deployment's layout statement, not
    // a request's count.
    let mut packed: Vec<u32> = Vec::new();
    // The element count of the largest trajectory the solver steps: its residual is one operator over
    // a whole sample, so its width is the deployment's canvas and not a constant.
    let mut elements: Vec<u64> = Vec::new();
    // The longest audio track the deployment encodes: the audio encoder's residual units are one
    // operator each over their own output, so their key carries its channel count times these samples.
    let mut track: Vec<u32> = Vec::new();
    // Whether the deployment serves conditioned tasks (`fl2va`/`ref2va`) or only `t2va`: it decides the
    // modulation cache's group count, which is a program's width.
    let mut conditioned: Option<bool> = None;
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--samples" => {
                for sample in value.split(',') {
                    samples.push(sample.trim().parse()?);
                }
            }
            "--rasters" => {
                for raster in value.split(',') {
                    let (height, width) = raster
                        .trim()
                        .split_once('x')
                        .ok_or_else(|| format!("{raster} is not HxW"))?;
                    rasters.push((height.parse()?, width.parse()?));
                }
            }
            "--tokens" => {
                for token in value.split(',') {
                    tokens.push(token.trim().parse()?);
                }
            }
            "--rows" => {
                for row in value.split(',') {
                    packed.push(row.trim().parse()?);
                }
            }
            "--conditioned" => {
                conditioned = Some(match value.as_str() {
                    "true" => true,
                    "false" => false,
                    other => return Err(format!("{other} is not a boolean").into()),
                });
            }
            "--elements" => {
                for count in value.split(',') {
                    elements.push(count.trim().parse()?);
                }
            }
            "--track" => {
                for samples in value.split(',') {
                    track.push(samples.trim().parse()?);
                }
            }
            "--tp" => tensor_parallel = Some(value.parse()?),
            "--dtype" => {
                dtype = Some(match value.as_str() {
                    "f16" => ops::DType::F16,
                    "bf16" => ops::DType::BF16,
                    "f32" => ops::DType::F32,
                    other => return Err(format!("{other} is not a float type").into()),
                });
            }
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if samples.is_empty() || rasters.is_empty() {
        return Err("both --samples and --rasters are the deployment's and have no default".into());
    }
    let dtype = dtype.ok_or("--dtype is the checkpoint's and has no default")?;
    if tokens.is_empty() {
        return Err("--tokens is the deployment's and has no default".into());
    }
    tokens.sort_unstable();
    tokens.dedup();
    if packed.is_empty() {
        return Err("--rows is the deployment's and has no default".into());
    }
    packed.sort_unstable();
    packed.dedup();
    if track.is_empty() {
        return Err("--track is the deployment's and has no default".into());
    }
    track.sort_unstable();
    track.dedup();
    if elements.is_empty() {
        return Err("--elements is the deployment's and has no default".into());
    }
    elements.sort_unstable();
    elements.dedup();
    let conditioned = conditioned.ok_or("--conditioned is the deployment's and has no default")?;
    let degree = tensor_parallel.ok_or("--tp is the deployment's and has no default")?;
    let partition = Partition {
        tensor_parallel: degree,
        tensor_rank: 0,
        ..Partition::single()
    };
    samples.sort_unstable();
    samples.dedup();
    rasters.sort_unstable();
    rasters.dedup();

    let mut rows: Vec<(String, String)> = Vec::new();

    let audio = root.join("audio_vae/config.json");
    if audio.exists() {
        let config = AudioVaeConfig::parse(&read(&audio)?)?;
        rows.extend(AudioDecoder::programs(&config, &samples)?);
        // The audio encoder runs FP32 by the design's own rows, so it takes no dtype.
        rows.extend(AudioEncoder::programs(&config, &track)?);
    } else {
        eprintln!("not walked: {} has no audio VAE config", root.display());
    }

    // The post-processor is weightless: its programs follow from the raster sizes and the decoder's
    // element type, which the video VAE's config states.
    let vae = root.join("vae/config.json");
    if vae.exists() {
        let config = VideoVaeConfig::parse(&read(&vae)?)?;
        rows.extend(VideoDecoder::programs(&config, dtype)?);
        rows.extend(VideoPostprocessor::programs(dtype, &rasters));
    } else {
        eprintln!("not walked: {} has no video VAE config", root.display());
    }

    let encoder = root.join("text_encoder/config.json");
    if encoder.exists() {
        let metadata = read(&encoder)?;
        let processor = read(&root.join("text_encoder/preprocessor_config.json"))?;
        let config = TextEncoderConfig::parse(&metadata, &processor)?;
        // One row per prompt length: the attention's key bound is the deployment's, and the gated
        // activation's `wide` form follows from it.
        for tokens in &tokens {
            rows.extend(TextEncoder::programs(&config, &partition, dtype, *tokens)?);
        }
        // The tower attends grids of the deployment's largest image; its programs do not depend on
        // them beyond the bound the module is accepted for.
        let sequence = tokens.iter().copied().max().unwrap_or(0);
        rows.extend(VisionTower::programs(&config.vision, dtype, sequence)?);
    } else {
        eprintln!("not walked: text_encoder");
    }

    // The two DiT partitions, each stated from its own `config.json`. The attention kind is a
    // deployment's contract (a sparse contract is a student's), so this walks the base form and the
    // walk itself refuses a sparse partition by name.
    for (component, directory) in [
        ("transformer", "transformer"),
        ("transformer_ref", "transformer_ref"),
    ] {
        let path = root.join(directory).join("config.json");
        if !path.exists() {
            eprintln!("not walked: {component}");
            continue;
        }
        let values = read(&path)?;
        let values = values
            .as_object()
            .ok_or_else(|| format!("{component}/config.json is not an object"))?;
        let config =
            TransformerConfig::parse(values, &format!("{component}/config.json"))?;
        let max = packed.iter().copied().max().unwrap_or(0);
        // The group count is the task set's: a conditioned deployment's modulation cache is another
        // program at the entries' width. The denoiser states its two stacks and the orchestration
        // around them, so the walk asks it rather than a stack.
        rows.extend(Denoiser::programs(
            &config,
            AttentionKind::Dense,
            dtype,
            max,
            Groups::of(conditioned),
            &partition,
            &elements,
        )?);
    }

    println!(
        "# MiniMax-H3 programs for these clip lengths: {samples:?}, raster sizes: {rasters:?}, prompt lengths: {tokens:?}, packed rows: {packed:?}, trajectory elements: {elements:?}, audio tracks: {track:?}, tensor-parallel degree {degree}"
    );
    println!(
        "# generated by `cargo run -p minimax-h3 --example programs -- <pipeline root> --dtype {dtype} --samples {} --rasters {} --tokens {} --rows {} --elements {} --track {} --tp {degree}`",
        samples.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
        rasters.iter().map(|(height, width)| format!("{height}x{width}")).collect::<Vec<_>>().join(","),
        tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
        packed.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
        elements.iter().map(u64::to_string).collect::<Vec<_>>().join(","),
        track.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
    );
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
