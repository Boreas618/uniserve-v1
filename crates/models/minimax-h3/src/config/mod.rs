//! MiniMax-H3's typed configuration: the checkpoint's sidecars, read once at the loading
//! boundary and immutable afterwards.

mod audio_vae;
mod video_vae;

use nn::config::Fields;

pub use audio_vae::AudioVaeConfig;
pub use processing::minimax_h3::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_LATENTS_PER_SECOND, AUDIO_SAMPLE_RATE, FPS, FRAMES,
    LATENT_COMPRESSION, PATCH_SIDE,
};
pub use video_vae::VideoVaeConfig;

use crate::Result;

/// One per-channel latent statistic of a VAE sidecar: a number for every latent channel.
fn statistics(values: &Fields<'_>, key: &str, channels: u32) -> Result<Vec<f32>> {
    Ok(values
        .fixed_numbers(key, channels as usize)?
        .into_iter()
        .map(|value| value as f32)
        .collect())
}

/// The released checkpoint's sidecars, as the tests of this crate use them.
#[cfg(test)]
pub(crate) mod tests_support {
    use serde_json::Value;

    /// The video VAE's configuration.
    pub(crate) fn video_vae() -> Value {
        let statistics = |value: f64| Value::Array((0..24).map(|_| Value::from(value)).collect());

        serde_json::json!({
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 24,
            "block_out_channels": [128, 256, 256, 512, 512, 1024],
            "layers_per_block": 2,
            "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
            "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
            "norm_num_groups": 32,
            "norm_eps": 1e-06,
            "spatial_padding_mode": "reflect",
            "decoder_num_layers": 36,
            "decoder_num_attention_heads": 32,
            "decoder_attention_head_dim": 64,
            "decoder_num_register_tokens": 4,
            "decoder_ffn_mult": 4,
            "decoder_rope_theta": 100.0,
            "decoder_rope_dim_ratio": 0.75,
            "decoder_norm_eps": 1e-05,
            "clip_length": 17,
            "token_drop": 3,
            "latents_mean": statistics(0.0),
            "latents_std": statistics(1.0),
        })
    }

    /// The audio VAE's configuration.
    pub(crate) fn audio_vae() -> Value {
        serde_json::json!({
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 4, 5, 5],
            "latent_dim": 2048,
            "latent_channels": 32,
            "decoder_dim": 1024,
            "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
            "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
            "num_attention_heads": 8,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "sampling_rate": 32000,
            "latents_mean": vec![0.0_f64; 32],
            "latents_std": vec![1.0_f64; 32],
        })
    }
}
