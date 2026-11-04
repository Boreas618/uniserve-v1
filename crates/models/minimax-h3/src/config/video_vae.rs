//! The video VAE's configuration: its encoder stages, the latent reconstruction geometry, and the
//! channel statistics that turn a latent into a raster.
//!
//! The decoder is the piece H3 serves: it inflates a latent of one temporal step and a
//! `spatial_compression x spatial_compression` pixel patch back into frames, in clips of
//! `clip_length` frames. The encoder is the piece `ref2va` uses to encode condition media. Both
//! share one configuration, which the checkpoint's `vae/config.json` states.

use serde_json::{Map, Value};

use super::field;
use crate::{Error, Result};

/// How a convolution pads the spatial axes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpatialPadding {
    /// Mirror the raster about its edge.
    Reflect,
    /// Repeat the edge value.
    Replicate,
    /// Pad with zeros.
    Constant,
}

/// The video VAE's network and reconstruction geometry.
#[derive(Clone, PartialEq, Debug)]
pub struct VideoVaeConfig {
    /// Raster channels the encoder reads.
    pub in_channels: u32,
    /// Raster channels the decoder writes.
    pub out_channels: u32,
    /// Latent channels.
    pub latent_channels: u32,
    /// Each encoder stage's width.
    pub block_out_channels: Vec<u32>,
    /// Residual layers per stage.
    pub layers_per_block: u32,
    /// Each stage's spatial stride.
    pub spatial_downsample_factors: Vec<u32>,
    /// Each stage's temporal stride.
    pub temporal_downsample_factors: Vec<u32>,
    /// Group-normalization groups.
    pub norm_num_groups: u32,
    /// Group-normalization epsilon.
    pub norm_eps: f32,
    /// How the spatial convolutions pad.
    pub spatial_padding_mode: SpatialPadding,
    /// Transformer layers of the decoder's latent stage.
    pub decoder_num_layers: u32,
    /// Attention heads of those layers.
    pub decoder_num_attention_heads: u32,
    /// Width of one head.
    pub decoder_attention_head_dim: u32,
    /// Learned register tokens the decoder's latent stage carries.
    pub decoder_num_register_tokens: u32,
    /// The feed-forward width multiplier of those layers.
    pub decoder_ffn_mult: u32,
    /// The rotary base of those layers.
    pub decoder_rope_theta: f32,
    /// The fraction of a head the 3-axis rotary embedding rotates.
    pub decoder_rope_dim_ratio: f32,
    /// The epsilon of those layers' normalizations.
    pub decoder_norm_eps: f32,
    /// Frames of one clip: the VAE's temporal footprint.
    pub clip_length: u32,
    /// Latent frames the final clip drops, which the next clip's overlap repeats.
    pub token_drop: u32,
    /// Per-channel mean of the latent space.
    pub latents_mean: Vec<f32>,
    /// Per-channel standard deviation of the latent space.
    pub latents_std: Vec<f32>,
}

impl VideoVaeConfig {
    /// Read `vae/config.json`.
    pub fn parse(values: &Value) -> Result<Self> {
        let values = field::object(values, "MiniMax-H3 vae/config.json")?;
        let owner = "MiniMax-H3 video_vae";
        let latent_channels = field::count(values, "latent_channels", owner)?;
        let spatial_padding_mode = match field::text(values, "spatial_padding_mode", owner)? {
            "reflect" => SpatialPadding::Reflect,
            "replicate" => SpatialPadding::Replicate,
            "constant" => SpatialPadding::Constant,
            other => {
                return Err(Error::config(format!(
                    "unsupported H3 video spatial padding mode {other:?}"
                )));
            }
        };
        let config = Self {
            in_channels: field::count(values, "in_channels", owner)?,
            out_channels: field::count(values, "out_channels", owner)?,
            latent_channels,
            block_out_channels: field::counts(values, "block_out_channels", owner)?,
            layers_per_block: field::count(values, "layers_per_block", owner)?,
            spatial_downsample_factors: field::counts(values, "spatial_downsample_factors", owner)?,
            temporal_downsample_factors: field::counts(
                values,
                "temporal_downsample_factors",
                owner,
            )?,
            norm_num_groups: field::count(values, "norm_num_groups", owner)?,
            norm_eps: field::positive(values, "norm_eps", owner)? as f32,
            spatial_padding_mode,
            decoder_num_layers: field::count(values, "decoder_num_layers", owner)?,
            decoder_num_attention_heads: field::count(
                values,
                "decoder_num_attention_heads",
                owner,
            )?,
            decoder_attention_head_dim: field::count(values, "decoder_attention_head_dim", owner)?,
            decoder_num_register_tokens: field::count(
                values,
                "decoder_num_register_tokens",
                owner,
            )?,
            decoder_ffn_mult: field::count(values, "decoder_ffn_mult", owner)?,
            decoder_rope_theta: field::positive(values, "decoder_rope_theta", owner)? as f32,
            decoder_rope_dim_ratio: field::positive(values, "decoder_rope_dim_ratio", owner)?
                as f32,
            decoder_norm_eps: field::positive(values, "decoder_norm_eps", owner)? as f32,
            clip_length: field::count(values, "clip_length", owner)?,
            token_drop: field::index(values, "token_drop", owner)?,
            latents_mean: channel_statistics(values, "latents_mean", latent_channels)?,
            latents_std: channel_statistics(values, "latents_std", latent_channels)?,
        };
        config.validate()?;
        Ok(config)
    }

    /// Check the stages, the latent statistics and the decoder's rotary embedding against each
    /// other.
    fn validate(&self) -> Result<()> {
        let owner = "H3 video_vae";
        if self.block_out_channels.len() != self.spatial_downsample_factors.len()
            || self.block_out_channels.len() != self.temporal_downsample_factors.len()
        {
            return Err(Error::config(format!(
                "{owner}: encoder stages must have matching widths and strides: {} widths, \
                 {} spatial, {} temporal",
                self.block_out_channels.len(),
                self.spatial_downsample_factors.len(),
                self.temporal_downsample_factors.len()
            )));
        }
        if self.token_drop >= self.clip_latents() {
            return Err(Error::config(format!(
                "{owner}.token_drop {} must leave latent frames of a {}-frame clip",
                self.token_drop, self.clip_length
            )));
        }
        if self.latents_std.iter().any(|value| *value <= 0.0) {
            return Err(Error::config(format!(
                "{owner}.latents_std must be positive"
            )));
        }
        // The rotary width splits evenly across the three axes (time, height, width), and each
        // axis rotates an even number of channels.
        let rotary_width =
            self.decoder_attention_head_dim as f64 * self.decoder_rope_dim_ratio as f64;
        let rounded = rotary_width.round();
        if self.decoder_rope_dim_ratio > 1.0
            || (rotary_width - rounded).abs() > f64::EPSILON
            || (rounded as u32) % 6 != 0
        {
            return Err(Error::config(format!(
                "{owner}: the decoder rotary width {rotary_width} must divide into three even axes"
            )));
        }
        Ok(())
    }

    /// The spatial compression: latent columns become this many pixels.
    pub fn spatial_compression(&self) -> u32 {
        self.spatial_downsample_factors.iter().product()
    }

    /// The temporal compression: one latent frame covers this many video frames.
    pub fn temporal_compression(&self) -> u32 {
        self.temporal_downsample_factors.iter().product()
    }

    /// Latent frames one clip encodes to.
    ///
    /// The causal strides map `clip_length` frames to this many latent frames; the final clip
    /// keeps all but `token_drop` of them.
    pub fn clip_latents(&self) -> u32 {
        self.clip_length.div_ceil(self.temporal_compression())
    }

    /// The latent frames of the final clip: what a generated clip's latent actually holds.
    pub fn final_clip_latents(&self) -> u32 {
        self.clip_latents() - self.token_drop
    }
}

/// Read one per-channel latent statistic.
fn channel_statistics(values: &Map<String, Value>, key: &str, channels: u32) -> Result<Vec<f32>> {
    let numbers = field::numbers(values, key, "MiniMax-H3 video_vae")?;
    if numbers.len() != channels as usize {
        return Err(Error::config(format!(
            "H3 video_vae.{key} must describe every one of {channels} latent channels, got {}",
            numbers.len()
        )));
    }
    Ok(numbers.into_iter().map(|value| value as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values() -> Value {
        serde_json::from_str(
            r#"{
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 4,
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
                "latents_mean": [0.5, -0.5, 1.0, -1.0],
                "latents_std": [1.0, 2.0, 3.0, 4.0]
            }"#,
        )
        .expect("the fixture is JSON")
    }

    #[test]
    fn the_released_video_vae_compresses_sixteen_times_spatially_and_four_temporally() {
        let config = VideoVaeConfig::parse(&values()).expect("parsed");
        assert_eq!(config.spatial_compression(), 16);
        assert_eq!(config.temporal_compression(), 4);
        assert_eq!(config.clip_latents(), 5);
        assert_eq!(config.final_clip_latents(), 2);
        assert_eq!(config.spatial_padding_mode, SpatialPadding::Reflect);
        assert_eq!(config.latents_mean.len(), 4);
    }

    #[test]
    fn a_video_vae_refuses_mismatched_stages_and_a_latent_it_cannot_describe() {
        let mut stages = values();
        stages["spatial_downsample_factors"] = serde_json::json!([2, 2, 2]);
        assert!(VideoVaeConfig::parse(&stages).is_err());

        let mut statistics = values();
        statistics["latents_std"] = serde_json::json!([1.0, 2.0, 3.0]);
        assert!(VideoVaeConfig::parse(&statistics).is_err());

        let mut negative = values();
        negative["latents_std"] = serde_json::json!([1.0, -2.0, 3.0, 4.0]);
        assert!(VideoVaeConfig::parse(&negative).is_err());
    }

    #[test]
    fn a_video_vae_refuses_a_rotary_width_that_does_not_split_into_three_even_axes() {
        let mut values = values();
        values["decoder_rope_dim_ratio"] = serde_json::json!(0.5);
        assert!(VideoVaeConfig::parse(&values).is_err());
        values["decoder_rope_dim_ratio"] = serde_json::json!(1.5);
        assert!(VideoVaeConfig::parse(&values).is_err());
    }
}
