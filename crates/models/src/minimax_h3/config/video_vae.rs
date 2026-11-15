//! The video VAE's configuration, as the checkpoint's `vae/config.json` states it: the stages of
//! the convolutional encoder, the transformer of the decoder, the clips a video is encoded in
//! and the per-channel statistics of the latent space.
//!
//! The decoder serves every task: one latent voxel becomes `temporal_compression` frames of
//! `spatial_compression × spatial_compression` pixels. The encoder serves the conditioned
//! tasks, which encode a request's keyframes and reference media. Both read this one config.

use nn::Result;
use nn::config::Fields;
use nn::functional::Padding;
use nn::quantization::Declaration;
use serde_json::Value;

use super::statistics;
use crate::minimax_h3::ID;

/// What `vae/config.json` states of the video VAE (diffusers' `AutoencoderKLMiniMaxH3`).
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
    /// How the causal convolutions fill the rows and columns beyond a frame's edge.
    pub spatial_padding_mode: Padding,
    /// Transformer layers of the decoder.
    pub decoder_num_layers: u32,
    /// Attention heads of those layers.
    pub decoder_num_attention_heads: u32,
    /// Width of one head.
    pub decoder_attention_head_dim: u32,
    /// Learned register tokens that trail each sequence of the decoder.
    pub decoder_num_register_tokens: u32,
    /// The feed-forward width multiplier of those layers.
    pub decoder_ffn_mult: u32,
    /// The rotary base of those layers.
    pub decoder_rope_theta: f32,
    /// The fraction of a head the 3-axis rotary embedding rotates.
    pub decoder_rope_dim_ratio: f32,
    /// The epsilon of those layers' normalizations.
    pub decoder_norm_eps: f32,
    /// Frames of one clip the encoder reads at a time.
    pub clip_length: u32,
    /// Latent frames the final clip of a video drops.
    pub token_drop: u32,
    /// Per-channel mean of the latent space.
    pub latents_mean: Vec<f32>,
    /// Per-channel standard deviation of the latent space.
    pub latents_std: Vec<f32>,
    /// How the stage stores its weights: dense, or the packed decoder projections a ModelOpt
    /// export's manifest calibrates.
    pub quantization: Declaration,
}

impl VideoVaeConfig {
    /// Read `vae/config.json`, the stage's weights dense until the checkpoint's reader states a
    /// calibration.
    pub fn parse(values: &Value) -> Result<Self> {
        let values = Fields::of("vae/config.json", values)?;
        let latent_channels = values.count("latent_channels")?;

        let spatial_padding_mode = match values.text("spatial_padding_mode")? {
            "reflect" => Padding::Reflect,
            "replicate" => Padding::Replicate,
            // A constant border is the convolution's own zero padding, which the causal pad's
            // kernel does not fill: it mirrors or repeats the edge.
            other => {
                return Err(ID.config(format!(
                    "unsupported H3 video spatial padding mode {other:?}"
                )));
            }
        };

        let config = Self {
            in_channels: values.count("in_channels")?,
            out_channels: values.count("out_channels")?,
            latent_channels,
            block_out_channels: values.counts("block_out_channels")?,
            layers_per_block: values.count("layers_per_block")?,
            spatial_downsample_factors: values.counts("spatial_downsample_factors")?,
            temporal_downsample_factors: values.counts("temporal_downsample_factors")?,
            norm_num_groups: values.count("norm_num_groups")?,
            norm_eps: values.positive("norm_eps")? as f32,
            spatial_padding_mode,
            decoder_num_layers: values.count("decoder_num_layers")?,
            decoder_num_attention_heads: values.count("decoder_num_attention_heads")?,
            decoder_attention_head_dim: values.count("decoder_attention_head_dim")?,
            decoder_num_register_tokens: values.count("decoder_num_register_tokens")?,
            decoder_ffn_mult: values.count("decoder_ffn_mult")?,
            decoder_rope_theta: values.positive("decoder_rope_theta")? as f32,
            decoder_rope_dim_ratio: values.positive("decoder_rope_dim_ratio")? as f32,
            decoder_norm_eps: values.positive("decoder_norm_eps")? as f32,
            clip_length: values.count("clip_length")?,
            token_drop: values.index("token_drop")?,
            latents_mean: statistics(&values, "latents_mean", latent_channels)?,
            latents_std: statistics(&values, "latents_std", latent_channels)?,
            quantization: Declaration::dense(),
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
            return Err(ID.config(format!(
                "{owner}: encoder stages must have matching widths and strides: {} widths, \
                 {} spatial, {} temporal",
                self.block_out_channels.len(),
                self.spatial_downsample_factors.len(),
                self.temporal_downsample_factors.len()
            )));
        }
        if self.token_drop >= self.clip_latents() {
            return Err(ID.config(format!(
                "{owner}.token_drop {} must leave latent frames of a {}-frame clip",
                self.token_drop, self.clip_length
            )));
        }
        if self.latents_std.iter().any(|value| *value <= 0.0) {
            return Err(ID.config(format!("{owner}.latents_std must be positive")));
        }

        // The rotary width splits evenly across the three axes (time, height, width), and each
        // axis rotates an even number of channels.
        let rotary_width =
            self.decoder_attention_head_dim as f64 * self.decoder_rope_dim_ratio as f64;
        let rounded = rotary_width.round();
        if self.decoder_rope_dim_ratio > 1.0
            || (rotary_width - rounded).abs() > f64::EPSILON
            || !(rounded as u32).is_multiple_of(6)
        {
            return Err(ID.config(format!(
                "{owner}: the decoder rotary width {rotary_width} must divide into three even axes"
            )));
        }
        Ok(())
    }

    /// The spatial compression: one latent row or column is this many pixels.
    pub fn spatial_compression(&self) -> u32 {
        self.spatial_downsample_factors.iter().product()
    }

    /// The temporal compression: one latent frame covers this many video frames.
    pub fn temporal_compression(&self) -> u32 {
        self.temporal_downsample_factors.iter().product()
    }

    /// Latent frames one clip encodes to: the causal strides map `clip_length` frames to
    /// `ceil(clip_length / temporal_compression)`, and the final clip keeps all but `token_drop`
    /// of them.
    pub fn clip_latents(&self) -> u32 {
        self.clip_length.div_ceil(self.temporal_compression())
    }

    /// The latent frames the final clip of a video keeps.
    pub fn final_clip_latents(&self) -> u32 {
        self.clip_latents() - self.token_drop
    }
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
        assert_eq!(config.spatial_padding_mode, Padding::Reflect);
        // The parsing keeps every statistic the sidecar declares, one per entry of its array.
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
