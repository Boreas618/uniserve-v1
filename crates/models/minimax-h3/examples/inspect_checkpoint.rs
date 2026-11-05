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
                "task {:?}, DiT {}, schedules {:?}, backend {:?}, sparse {:?}",
                contract.task,
                contract.dit.directory(),
                contract.schedule,
                contract.backend,
                contract.attention
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
    // walks it. Each list is the block's own `keys` - the table its own load reads - so this half is
    // exact in both directions: every name a component asks for is in the index, and every name in
    // the index belongs to one of the components that read that directory.
    let lists: &[(&str, &str, Vec<String>)] = &[
        (
            "vae",
            "video encoder",
            minimax_h3::encoding::VideoEncoder::keys(&config.video_vae)?,
        ),
        (
            "vae",
            "video decoder",
            minimax_h3::video_vae::VideoDecoder::keys(&config.video_vae)?,
        ),
        (
            "audio_vae",
            "audio encoder",
            minimax_h3::encoding::AudioEncoder::keys(&config.audio_vae)?,
        ),
        (
            "audio_vae",
            "audio decoder",
            minimax_h3::audio_vae::AudioDecoder::keys(&config.audio_vae)?,
        ),
        (
            "text_encoder",
            "text encoder",
            minimax_h3::text_encoder::TextEncoder::keys(&config.text_encoder),
        ),
        (
            "text_encoder",
            "vision tower",
            minimax_h3::vision::VisionTower::keys(&config.text_encoder.vision),
        ),
    ];
    let mut names_by_directory = std::collections::BTreeMap::new();
    // The DiT partitions are checked against their own `keys` too, so their directories are read
    // here whether or not a conditioning list names them.
    let dit_directories = config
        .denoisers
        .keys()
        .map(|dit| dit.directory())
        .collect::<Vec<_>>();
    let list_directories = lists
        .iter()
        .map(|(directory, _, _)| *directory)
        .chain(dit_directories.iter().copied())
        .collect::<Vec<_>>();
    for directory in list_directories {
        if names_by_directory.contains_key(directory) {
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
        names_by_directory.insert(directory.to_owned(), names);
    }
    // Both halves are exact in both directions, because both ask the block rather than a prefix
    // table: every name a component asks for is in the index, and every name in the index belongs
    // to one of the components that read that directory. A prefix table could say the first only.
    let mut missing = Vec::new();
    for (dit, denoiser) in &config.denoisers {
        let block = format!("{} DiT", dit.directory());
        let names = &names_by_directory[dit.directory()];
        if names.is_empty() {
            println!(
                "keys {block}: {} states no index here, so its keys are unread",
                dit.directory()
            );
            continue;
        }
        let keys = minimax_h3::denoiser::Denoiser::keys(denoiser);
        let absent =
            minimax_h3::checkpoint::absent(&keys, |key| names.iter().any(|name| name == key));
        println!(
            "keys {block}: {} asked, {} in the index, {} missing",
            keys.len(),
            keys.len() - absent.len(),
            absent.len()
        );
        for key in absent {
            missing.push(format!("{block} asks for {key} and its index holds none"));
        }
    }
    for (directory, block, keys) in lists {
        let names = &names_by_directory[*directory];
        if names.is_empty() {
            println!("keys {block}: {directory} states no index here, so its keys are unread");
            continue;
        }
        let absent =
            minimax_h3::checkpoint::absent(keys, |key| names.iter().any(|name| name == key));
        println!(
            "keys {block}: {} asked, {} in the index, {} missing",
            keys.len(),
            keys.len() - absent.len(),
            absent.len()
        );
        for key in absent {
            missing.push(format!("{block} asks for {key} and {directory} holds none"));
        }
    }
    // A directory's tensors are read by its components together, so a name none of them asks for is
    // the report a reader wants: a tensor the export ships and the model never uses. That is the
    // half a prefix table could not state at all.
    for directory in names_by_directory.keys() {
        let names = &names_by_directory[directory];
        if names.is_empty() {
            continue;
        }
        let mut asked: Vec<String> = lists
            .iter()
            .filter(|(list_directory, _, _)| list_directory == directory)
            .flat_map(|(_, _, keys)| keys.iter().cloned())
            .collect();
        for (dit, denoiser) in &config.denoisers {
            if dit.directory() == directory {
                asked.extend(minimax_h3::denoiser::Denoiser::keys(denoiser));
            }
        }
        let unread = names
            .iter()
            .filter(|name| !asked.iter().any(|key| key == *name))
            .count();
        println!(
            "keys {directory}: {unread} of {} tensors unread",
            names.len()
        );
    }
    if !missing.is_empty() {
        for line in &missing {
            eprintln!("missing key: {line}");
        }
        return Err(format!("{} keys a block asks for are absent", missing.len()).into());
    }
    Ok(())
}
