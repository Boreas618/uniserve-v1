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
    Ok(())
}
