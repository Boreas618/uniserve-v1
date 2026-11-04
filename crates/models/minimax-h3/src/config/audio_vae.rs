//! The audio VAE's configuration: the BigVGAN decoder that turns an audio latent into stereo PCM,
//! the latent space's channel statistics, and the sample clock.
//!
//! The product of `encoder_rates` is the samples one latent frame stands for - the encoder's input
//! hop and the decoder's output rate - and H3's timeline counts audio latents in those hops
//! (specs/worker.md 9, `docs/minimax_h3/minimax_h3.md`).

use serde_json::{Map, Value};

use super::field;
use crate::{Error, Result};

/// The audio VAE's network geometry and latent statistics.
#[derive(Clone, PartialEq, Debug)]
pub struct AudioVaeConfig {
    /// Width of the encoder's first stage.
    pub encoder_dim: u32,
    /// Each encoder stage's stride.
    pub encoder_rates: Vec<u32>,
    /// Width of one latent frame.
    pub latent_dim: u32,
    /// Latent channels: the generated audio stream's.
    pub latent_channels: u32,
    /// Width of the decoder's first stage.
    pub decoder_dim: u32,
    /// Each decoder stage's stride, the encoder's reversed.
    pub decoder_rates: Vec<u32>,
    /// Each decoder stage's transposed convolution kernel.
    pub decoder_kernel_sizes: Vec<u32>,
    /// Attention heads of the latent stage.
    pub num_attention_heads: u32,
    /// Each residual stack's kernel sizes.
    pub resblock_kernel_sizes: Vec<u32>,
    /// Each residual stack's dilations, one row per kernel size.
    pub resblock_dilation_sizes: Vec<Vec<u32>>,
    /// The sample clock the checkpoint was trained at, in hertz.
    pub sampling_rate: u32,
    /// Per-channel mean of the latent space.
    pub latents_mean: Vec<f32>,
    /// Per-channel standard deviation of the latent space.
    pub latents_std: Vec<f32>,
}

impl AudioVaeConfig {
    /// Read `audio_vae/config.json`.
    pub fn parse(values: &Value) -> Result<Self> {
        let values = field::object(values, "MiniMax-H3 audio_vae/config.json")?;
        let owner = "MiniMax-H3 audio_vae";
        let latent_channels = field::count(values, "latent_channels", owner)?;
        let config = Self {
            encoder_dim: field::count(values, "encoder_dim", owner)?,
            encoder_rates: field::counts(values, "encoder_rates", owner)?,
            latent_dim: field::count(values, "latent_dim", owner)?,
            latent_channels,
            decoder_dim: field::count(values, "decoder_dim", owner)?,
            decoder_rates: field::counts(values, "decoder_rates", owner)?,
            decoder_kernel_sizes: field::counts(values, "decoder_kernel_sizes", owner)?,
            num_attention_heads: field::count(values, "num_attention_heads", owner)?,
            resblock_kernel_sizes: field::counts(values, "resblock_kernel_sizes", owner)?,
            resblock_dilation_sizes: field::nested_counts(
                values,
                "resblock_dilation_sizes",
                owner,
            )?,
            sampling_rate: field::count(values, "sampling_rate", owner)?,
            latents_mean: channel_statistics(values, "latents_mean", latent_channels)?,
            latents_std: channel_statistics(values, "latents_std", latent_channels)?,
        };
        config.validate()?;
        Ok(config)
    }

    /// Check the stages, the residual stacks and the latent width against each other.
    fn validate(&self) -> Result<()> {
        let owner = "H3 audio_vae";
        if self.decoder_rates.len() != self.decoder_kernel_sizes.len() {
            return Err(Error::config(format!(
                "{owner}: {} decoder stages and {} kernels",
                self.decoder_rates.len(),
                self.decoder_kernel_sizes.len()
            )));
        }
        if self.resblock_dilation_sizes.len() != self.resblock_kernel_sizes.len() {
            return Err(Error::config(format!(
                "{owner}: {} residual kernels and {} dilation rows",
                self.resblock_kernel_sizes.len(),
                self.resblock_dilation_sizes.len()
            )));
        }
        if self.latent_dim % self.num_attention_heads != 0 {
            return Err(Error::config(format!(
                "{owner}: latent width {} is not divisible by {} attention heads",
                self.latent_dim, self.num_attention_heads
            )));
        }
        if self.latents_std.iter().any(|value| *value <= 0.0) {
            return Err(Error::config(format!(
                "{owner}.latents_std must be positive"
            )));
        }
        Ok(())
    }

    /// Samples one latent frame covers: the encoder's hop and the decoder's output rate.
    pub fn hop(&self) -> u32 {
        self.encoder_rates.iter().product()
    }

    /// Samples one latent frame of the decoder's output covers.
    pub fn decoder_hop(&self) -> u32 {
        self.decoder_rates.iter().product()
    }
}

/// Read one per-channel latent statistic.
fn channel_statistics(values: &Map<String, Value>, key: &str, channels: u32) -> Result<Vec<f32>> {
    let numbers = field::numbers(values, key, "MiniMax-H3 audio_vae")?;
    if numbers.len() != channels as usize {
        return Err(Error::config(format!(
            "H3 audio_vae.{key} must describe every one of {channels} latent channels, got {}",
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
                "encoder_dim": 64,
                "encoder_rates": [2, 4, 4, 5, 5],
                "latent_dim": 2048,
                "latent_channels": 2,
                "decoder_dim": 1024,
                "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
                "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
                "num_attention_heads": 8,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "sampling_rate": 32000,
                "latents_mean": [0.1, -0.1],
                "latents_std": [1.5, 2.5]
            }"#,
        )
        .expect("the fixture is JSON")
    }

    #[test]
    fn the_released_audio_vae_hops_the_timeline_sample_count_per_latent() {
        let config = AudioVaeConfig::parse(&values()).expect("parsed");
        assert_eq!(config.hop(), super::super::AUDIO_HOP);
        assert_eq!(config.decoder_hop(), super::super::AUDIO_HOP);
        assert_eq!(config.sampling_rate, super::super::AUDIO_SAMPLE_RATE);
        assert_eq!(config.latent_channels, 2);
    }

    #[test]
    fn an_audio_vae_refuses_mismatched_stages_and_indivisible_latents() {
        let mut kernels = values();
        kernels["decoder_kernel_sizes"] = serde_json::json!([9, 9, 4]);
        assert!(AudioVaeConfig::parse(&kernels).is_err());

        let mut dilations = values();
        dilations["resblock_dilation_sizes"] = serde_json::json!([[1, 3, 5]]);
        assert!(AudioVaeConfig::parse(&dilations).is_err());

        let mut heads = values();
        heads["num_attention_heads"] = serde_json::json!(7);
        assert!(AudioVaeConfig::parse(&heads).is_err());
    }
}
