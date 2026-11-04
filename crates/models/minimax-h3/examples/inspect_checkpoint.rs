//! Read a checkpoint's own statements into this crate's `Config`, without a device or weights.
//!
//!     cargo run -p minimax-h3 --example inspect_checkpoint -- <pipeline root>
//!
//! Everything it prints is host metadata: the layout the crate recognised, the DiT partitions the
//! checkpoint holds, the tasks and canvases each serves, the attention kind, the precision each
//! block keeps at load, and the sidecars' dtypes. It opens no tensor and allocates nothing on a
//! device, so it is the cheap first step before a load whose weights would cost the whole machine.
//!
//! The `ref2va` export is not a pipeline root on its own: it holds the reference DiT and the two
//! schedulers, and continues a base release for everything else. Point this at the base release to
//! read the checkpoint it continues, or at an engine-opened union once `open_roots` is on the
//! branch.

use std::path::PathBuf;

use minimax_h3::checkpoint::{Layout, Sidecars};
use minimax_h3::config::Config;
use minimax_h3::partition::Component;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root: PathBuf = std::env::args()
        .nth(1)
        .ok_or("usage: inspect_checkpoint <pipeline root>")?
        .into();

    let layout = Layout::detect(&root)?;
    println!("root: {}", root.display());
    println!("packed (ModelOpt manifest): {}", layout.packed);
    let contract = layout.contract.as_ref();
    println!(
        "contract: {}",
        match contract {
            None => "none - the checkpoint is a pipeline export".to_owned(),
            Some(contract) => format!(
                "task {:?}, DiT {}, schedules {:?}, sparse tile {} sparsity {} keep {:?}",
                contract.task,
                contract.dit.directory(),
                contract.schedule,
                contract.attention.tile,
                contract.attention.sparsity,
                contract.attention.reference_keep
            ),
        }
    );
    for (dit, tasks) in &layout.dits {
        println!("DiT {} serves {:?}", dit.directory(), tasks);
    }

    let sidecars = Sidecars::read(&root, None, &layout)?;
    let config = Config::normalize(&layout, &sidecars)?;

    // The checkpoint's depth and the depth the encoder runs are two numbers: the last retained
    // layer's raw hidden state is the prompt, with the final norm removed.
    println!(
        "text encoder: {} of {} layers retained, hidden {}, {} query heads, {} key/value heads, \
         head width {}",
        config.text_encoder.num_retained_layers,
        config.text_encoder.num_checkpoint_layers,
        config.text_encoder.hidden_size,
        config.text_encoder.num_attention_heads,
        config.text_encoder.num_key_value_heads,
        config.text_encoder.head_dim
    );
    // The vision tower is a different head width in the same checkpoint: its rotary spectrum is
    // keyed on that width halved, not on the language stack's head width.
    let vision = &config.text_encoder.vision;
    println!(
        "vision tower: {} blocks, hidden {}, {} heads, head width {}, patch {} x {} x {}, merge {}",
        vision.depth,
        vision.hidden_size,
        vision.num_heads,
        vision.hidden_size / vision.num_heads,
        vision.temporal_patch_size,
        vision.patch_size,
        vision.patch_size,
        vision.spatial_merge_size
    );
    for (dit, denoiser) in &config.denoisers {
        println!(
            "{}: {} layers, {} heads, {} residual width, {} attention width, tasks {:?}",
            dit.directory(),
            denoiser.transformer.num_hidden_layers,
            denoiser.transformer.num_attention_heads,
            denoiser.transformer.hidden_size,
            denoiser.transformer.attention_width(),
            denoiser.tasks
        );
        println!("  attention: {:?}", denoiser.attention);
        match denoiser.canvases {
            None => println!("  canvases: the canvas rule's twelve at 768p and 480p"),
            Some(canvases) => println!(
                "  canvases: {} named, e.g. {:?}",
                canvases.len(),
                canvases[0]
            ),
        }
    }
    println!(
        "video VAE: {} decoder layers, {} latent channels",
        config.video_vae.decoder_num_layers, config.video_vae.latent_channels
    );

    for component in Component::ALL {
        println!(
            "component {} -> block {}, activation {:?}",
            component.name(),
            component.block(),
            component.activation()
        );
    }

    // Which keys each block asks for, checked against the checkpoint's index without loading it.
    //
    // `load` publishes every tensor through `RetainedWeights`, which only the executor implements,
    // so no test in this crate can execute a block's load path; the failure that hides there is the
    // narrow one: a key the loader names that the checkpoint does not hold. The index states every
    // tensor's name whether or not its shard is present, so that failure is a host fact and this
    // walks it. Each prefix is one the loader builds - the loader is the authority and this table
    // only quotes it, one line per group - and a group with no match is an error, not a count.
    let prefixes: &[(&str, &str, &[&str])] = &[
        (
            "vae",
            "video encoder",
            &[
                "encoder.conv_in.",
                "encoder.norm_out.",
                "encoder.conv_out.",
                "encoder.down_blocks.",
                "encoder.down_blocks.0.resnets.",
                "quant_conv.",
            ],
        ),
        (
            "vae",
            "video decoder",
            &[
                "decoder.proj_in.",
                "decoder.register_tokens",
                "decoder.transformer_blocks.",
                "decoder.norm_out.",
                "decoder.proj_out.",
            ],
        ),
        (
            "audio_vae",
            "audio encoder",
            &["encoder.block.", "encoder.conv_in.", "encoder.conv_out."],
        ),
        (
            "audio_vae",
            "audio decoder",
            &[
                "decoder.conv_pre",
                "decoder.ups.",
                "decoder.resblocks.",
                "decoder.activation_post",
                "decoder.conv_post",
            ],
        ),
        (
            "text_encoder",
            "text encoder",
            &[
                "model.language_model.layers.",
                "model.language_model.embed_tokens.",
            ],
        ),
        (
            "transformer",
            "denoiser",
            &["transformer_blocks.", "token_refiner."],
        ),
    ];
    let mut names_by_directory = std::collections::BTreeMap::new();
    for (directory, _, _) in prefixes {
        if names_by_directory.contains_key(*directory) {
            continue;
        }
        let mut names = Vec::new();
        if let Ok(entries) = std::fs::read_dir(root.join(directory)) {
            for entry in entries.flatten() {
                let file = entry.file_name().to_string_lossy().to_string();
                if !file.ends_with(".index.json") {
                    continue;
                }
                let index: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(entry.path())?)?;
                if let Some(map) = index["weight_map"].as_object() {
                    names.extend(map.keys().cloned());
                }
            }
        }
        names_by_directory.insert((*directory).to_owned(), names);
    }
    let mut missing = Vec::new();
    for (directory, block, groups) in prefixes {
        let names = &names_by_directory[*directory];
        if names.is_empty() {
            println!("keys {block}: {directory} states no index here, so its keys are unread");
            continue;
        }
        for group in *groups {
            let found = names.iter().filter(|name| name.starts_with(group)).count();
            if found == 0 {
                missing.push(format!(
                    "{block} asks for {group}* and {directory} holds none"
                ));
            } else {
                println!("keys {block}: {group}* {found} tensors");
            }
        }
    }
    if !missing.is_empty() {
        for line in &missing {
            eprintln!("missing key group: {line}");
        }
        return Err(format!("{} key groups the blocks ask for are absent", missing.len()).into());
    }
    Ok(())
}
