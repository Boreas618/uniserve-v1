//! Print the row bounds and the per-evaluation buffers of a deployment.
//!
//! Sizing arithmetic for `notes/h3/deployments.md`: the padded row bound of a capacity, the packed
//! hidden it implies, and the attention scratch of either kind.

use minimax_h3::Dit;
use minimax_h3::Error;
use minimax_h3::checkpoint::{Layout, Sidecars};
use minimax_h3::config::{Config, NAMED_CANVASES};
use minimax_h3::partition::{Capacity, Partition};
use nn::VideoTask;

fn main() -> Result<(), Error> {
    let root = std::path::Path::new("/opt/uv/models/MiniMax-H3");
    let layout = Layout::detect(root)?;
    let sidecars = Sidecars::read(root, None, &layout)?;
    let config = Config::normalize(&layout, &sidecars)?;
    let denoiser = &config.denoisers[&Dit::Base];
    let width = u64::from(denoiser.transformer.hidden_size);
    println!(
        "hidden {width}, heads {} x {}",
        denoiser.transformer.num_attention_heads, denoiser.transformer.head_dim
    );

    for frames in [124u32, 362] {
        let capacity = Capacity {
            text_rows: 128,
            condition_rows: 192,
            frames,
        };
        let rows = u64::from(capacity.rows(denoiser)?);
        println!(
            "{frames:>4} frames: bound {rows:>7} rows, packed hidden {} MiB (rows x {width} x 2 B)",
            rows * width * 2 / (1 << 20)
        );
        if let minimax_h3::config::AttentionKind::Sparse { tile, .. } = denoiser.attention {
            let heads = u64::from(denoiser.transformer.num_attention_heads);
            let tiles = rows.div_ceil(u64::from(tile));
            println!(
                "      sparse: {tiles} tiles, compression scratch {} MiB",
                tiles * tiles * heads * 2 / (1 << 20)
            );
        }
    }
    // The video decoder's plan, at each canvas the DiT offers, for one clip.
    let ratio = config.video_vae.spatial_compression();
    for canvas in NAMED_CANVASES {
        let latent = minimax_h3::video_vae::Canvas::clip(
            &config.video_vae,
            canvas.height / ratio,
            canvas.width / ratio,
        );
        let arena: u64 = minimax_h3::decoding::video_arena(&config.video_vae, &latent)?
            .iter()
            .map(|buffer| buffer.elements * if buffer.element == "f16" { 2 } else { 4 })
            .sum();
        let one_tile = minimax_h3::decoding::video_arena(
            &config.video_vae,
            &minimax_h3::video_vae::Canvas::clip(&config.video_vae, 16, 16),
        )?
        .iter()
        .map(|buffer| buffer.elements * if buffer.element == "f16" { 2 } else { 4 })
        .sum::<u64>();
        println!(
            "decode plan {}x{}: clip {} MiB, one tile {} MiB",
            canvas.width,
            canvas.height,
            arena / (1 << 20),
            one_tile / (1 << 20)
        );
    }
    let partition = Partition::serving(&config, VideoTask::TextToVideoAudio, 1, 1, 0)?;
    println!("t2va tp1 ulysses1 blocks: {:?}", partition.blocks);
    Ok(())
}
