//! H3's audio VAE decoder: BigVGAN's stage arithmetic, its alias-free activation, and the stereo PCM
//! it ends in.
//!
//! `uniserve/models/minimax_h3/audio_vae.py` and diffusers'
//! `autoencoders/autoencoder_kl_minimax_h3_audio.py` are the reference. The decoder is a
//! BigVGAN: a 1×1 input projection to `latent_dim`, a kernel-7 pre-convolution, seven transposed-conv
//! upsampling stages (each followed by three anti-aliased multi-periodicity blocks whose outputs are
//! summed and averaged), one alias-free activation, and a kernel-7 post-convolution to mono, clamped
//! to [-1, 1].
//!
//! ## What this module owns
//!
//! * the stage table a deployment builds from [`AudioVaeConfig`]: every convolution's channel pair,
//!   rate, kernel and padding, so the layer stack and its weight names are derived, never transcribed;
//! * the alias-free activation's length arithmetic ([`Resample`], [`Activation`]) — the two replicate
//!   pads, the crop after the transposed convolution and the stride, which together map `length` to
//!   `length · ratio` and back to `length`;
//! * [`receptive_field`]: the latent frames of context one decoded sample needs on each side, which is
//!   what the runner carries as a halo around a media window;
//! * [`Pcm::quantize`]: the exact arithmetic of the `pcm_interleave` program, stated here so the
//!   operator's implementation has one reference.
//!
//! ## Precision
//!
//! The audio decoder is FP32 throughout (specs/worker.md 9; `precision.py` keeps it unquantized and
//! unconverted). Its 128 weight-normalized convolutions are folded once at the loading boundary
//! (`nn::WeightLoader::weight_norm`), so no layer carries the magnitude/direction pair, and its
//! convolutions run through cuDNN without TF32: spec 9 keeps TF32 for the video VAE's FP32 `conv3d`
//! alone.

use crate::{AudioVaeConfig, Error, Result};

/// One upsampling stage of the decoder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Stage {
    /// Input channels.
    pub channels_in: u32,
    /// Output channels: half the input channels.
    pub channels_out: u32,
    /// The transposed convolution's stride, i.e. the stage's upsampling rate.
    pub rate: u32,
    /// The transposed convolution's kernel.
    pub kernel: u32,
    /// The transposed convolution's zero padding: `(kernel − rate) / 2`.
    pub padding: u32,
}

/// The decoder's stages and the channel count it starts from.
///
/// The reference builds the stages from `decoder_dim` and halves the channel count at each one; a
/// sidecar whose rates and kernels disagree, or whose halving does not divide `decoder_dim`, is
/// refused at the loading boundary rather than producing a stack nobody can name.
pub fn stages(config: &AudioVaeConfig) -> Result<Vec<Stage>> {
    if config.decoder_rates.len() != config.decoder_kernel_sizes.len() {
        return Err(Error::config(format!(
            "the audio decoder has {} rates and {} kernels",
            config.decoder_rates.len(),
            config.decoder_kernel_sizes.len()
        )));
    }
    let mut channels = config.decoder_dim;
    let mut stages = Vec::with_capacity(config.decoder_rates.len());
    for (index, (&rate, &kernel)) in config
        .decoder_rates
        .iter()
        .zip(&config.decoder_kernel_sizes)
        .enumerate()
    {
        if channels < 2 || !channels.is_multiple_of(2) {
            return Err(Error::config(format!(
                "audio decoder stage {index} halves {channels} channels"
            )));
        }
        let channels_out = channels / 2;
        // `(kernel - rate) / 2` in the reference's integer arithmetic: the kernel is wider than the
        // stride at every stage of the checkpoint, so the difference is never negative.
        let padding = kernel.saturating_sub(rate) / 2;
        if rate == 0 || kernel < rate {
            return Err(Error::config(format!(
                "audio decoder stage {index} has rate {rate} and kernel {kernel}"
            )));
        }
        stages.push(Stage {
            channels_in: channels,
            channels_out,
            rate,
            kernel,
            padding,
        });
        channels = channels_out;
    }
    if stages.is_empty() {
        return Err(Error::config(
            "the audio decoder has no upsampling stage".to_owned(),
        ));
    }
    Ok(stages)
}

/// The alias-free resampler's geometry: one direction of the activation's two passes.
///
/// The upsample replicate-pads by `pad` on both sides, runs a transposed depthwise convolution of
/// stride `ratio`, and crops `pad_left` samples from the front and `pad_right` from the end; the
/// downsample replicate-pads the same way and runs a depthwise convolution of stride `ratio`. Both
/// formulas are the reference's, and the operator refuses an output length they do not produce.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Resample {
    /// The upsampling factor; 2 in the checkpoint.
    pub ratio: u32,
    /// The Kaiser-sinc filter's length; 12 in the checkpoint.
    pub kernel: u32,
    /// Samples replicate-padded on each side before the convolution.
    pub pad: u32,
    /// Samples cropped from the front after the transposed convolution.
    pub pad_left: u32,
    /// Samples cropped from the end after the transposed convolution.
    pub pad_right: u32,
}

impl Resample {
    /// The geometry of ratio `ratio` with a `kernel`-long filter.
    ///
    /// The reference states the three paddings once, for the upsampling direction; the downsampling
    /// direction pads its own convolution's input, which is why [`Resample::upsample_length`] and
    /// [`Resample::downsample_padding`] are separate.
    pub fn new(ratio: u32, kernel: u32) -> Result<Self> {
        if ratio == 0 || kernel == 0 {
            return Err(Error::config(
                "a resampler needs a positive ratio and kernel".to_owned(),
            ));
        }
        if kernel < ratio || !kernel.is_multiple_of(ratio) {
            return Err(Error::config(format!(
                "a {kernel}-long filter does not divide a rate of {ratio}"
            )));
        }
        Ok(Self {
            ratio,
            kernel,
            pad: kernel / ratio - 1,
            pad_left: (kernel / ratio - 1) * ratio + (kernel - ratio) / 2,
            pad_right: (kernel / ratio - 1) * ratio + (kernel - ratio + 1) / 2,
        })
    }

    /// The output length of the upsampling direction for `length` input samples.
    ///
    /// The transposed convolution of a `length + 2·pad` input, stride `ratio` and kernel `kernel`
    /// emits `(length + 2·pad − 1)·ratio + kernel`, and the crop removes `pad_left + pad_right`; the
    /// two cancel to `length · ratio` for the kernel this constructor admits.
    pub fn upsample_length(&self, length: u32) -> u32 {
        (u64::from(length) * u64::from(self.ratio)) as u32
    }

    /// The samples the upsampling direction's transposed convolution emits before the crop.
    pub fn transposed_length(&self, length: u32) -> u32 {
        (u64::from(length) + 2 * u64::from(self.pad) - 1) as u32 * self.ratio
            + self.kernel
    }

    /// The padding of the downsampling direction's convolution: `[pad_left, pad_right]`.
    pub fn downsample_padding(&self) -> (u32, u32) {
        let even = self.kernel.is_multiple_of(2);
        let left = self.kernel / 2 - u32::from(even);
        (left, self.kernel / 2)
    }

    /// The output length of the downsampling direction for `length` input samples.
    pub fn downsample_length(&self, length: u32) -> u32 {
        let (left, right) = self.downsample_padding();
        let padded = u64::from(length) + u64::from(left) + u64::from(right);
        // `conv1d` with stride `ratio` and no padding: `⌊(padded − kernel)/ratio⌋ + 1`.
        ((padded - u64::from(self.kernel)) / u64::from(self.ratio) + 1) as u32
    }
}

/// The alias-free activation: upsample, activate, low-pass back down.
///
/// The activation's length is unchanged; the operator that runs it fuses the five passes (pad,
/// transposed depthwise convolution, crop, SnakeBeta, pad, depthwise convolution) into one program
/// (specs/worker.md 9, the BigVGAN row).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Activation {
    /// The upsampling factor of both passes.
    pub ratio: u32,
    /// The Kaiser-sinc filter's length.
    pub kernel: u32,
}

impl Activation {
    /// The activation of `ratio` and `kernel`.
    pub fn new(ratio: u32, kernel: u32) -> Result<Self> {
        Resample::new(ratio, kernel)?;
        Ok(Self { ratio, kernel })
    }

    /// The activation of the checkpoint's decoder: ratio 2, kernel 12.
    pub fn bigvgan() -> Self {
        Self {
            ratio: 2,
            kernel: 12,
        }
    }

    /// The output length for `length` input samples, which is `length` itself.
    ///
    /// Stated as the two passes so that [`Resample`]'s arithmetic is the only place the formulas live.
    pub fn length(&self, length: u32) -> u32 {
        let up = Resample::new(self.ratio, self.kernel).expect("an admitted activation");
        up.downsample_length(up.upsample_length(length))
    }
}

/// The samples one latent frame decodes to.
pub fn samples_per_latent(config: &AudioVaeConfig) -> u32 {
    config.decoder_hop()
}

/// The latent frames a sample count needs: `ceil(samples / samples_per_latent)`.
pub fn latent_frames(config: &AudioVaeConfig, samples: u32) -> Result<u32> {
    let hop = samples_per_latent(config);
    if samples == 0 {
        return Err(Error::config(
            "an audio duration of zero samples has no latent frame".to_owned(),
        ));
    }
    Ok(samples.div_ceil(hop))
}

/// The samples a latent frame count decodes to.
pub fn track_samples(config: &AudioVaeConfig, latent_frames: u32) -> u64 {
    u64::from(latent_frames) * u64::from(samples_per_latent(config))
}

/// The latent frames of context one decoded sample depends on, per side.
///
/// This is the reference `receptive_field`: it walks the decoder in forward order, tracking how far
/// one output sample's dependency spreads, with the residual branches of a stage running in parallel
/// (so a stage contributes its widest branch, not their sum) and each alias-free activation spreading
/// its own upsample, filter and low-pass. The result rounds up and may exceed the exact field, which
/// costs context and never correctness — it is the halo the runner carries around a media window.
pub fn receptive_field(config: &AudioVaeConfig) -> Result<u32> {
    let stages = stages(config)?;
    if config.resblock_kernel_sizes.len() != config.resblock_dilation_sizes.len() {
        return Err(Error::config(format!(
            "the audio decoder has {} residual kernels and {} dilation lists",
            config.resblock_kernel_sizes.len(),
            config.resblock_dilation_sizes.len()
        )));
    }
    let activation = Activation::bigvgan();
    let mut spread = 0.0f64;
    let mut rate = 1.0f64;

    // A convolution's spread is `(kernel − 1) · dilation / rate` output samples at the current rate.
    let convolution = |kernel: u32, dilation: u32, rate: f64| -> f64 {
        let width = f64::from(kernel) - 1.0;
        width * f64::from(dilation) / rate
    };
    // One alias-free activation spreads its upsampled convolution and its low-pass, at the current
    // rate: the reference computes the first pass with `ceil` because the crop only takes whole
    // samples.
    let activation_spread = |rate: f64| -> f64 {
        let up = (f64::from(activation.kernel) / f64::from(activation.ratio)).ceil() - 1.0;
        up / rate
            + (f64::from(activation.kernel) - 1.0) / (rate * f64::from(activation.ratio))
    };

    spread += convolution(7, 1, rate);
    for stage in &stages {
        spread += (f64::from(stage.kernel) / f64::from(stage.rate)).ceil() / rate - 1.0 / rate;
        rate *= f64::from(stage.rate);
        let mut widest = 0.0f64;
        for (kernel, dilations) in config
            .resblock_kernel_sizes
            .iter()
            .zip(&config.resblock_dilation_sizes)
        {
            let mut branch = 0.0f64;
            for &dilation in dilations {
                branch += activation_spread(rate) + convolution(*kernel, dilation, rate);
                branch += activation_spread(rate) + convolution(*kernel, 1, rate);
            }
            widest = widest.max(branch);
        }
        spread += widest;
    }
    spread += activation_spread(rate) + convolution(7, 1, rate);

    // The spread covers both sides of one dependency, so one side is half.
    Ok(spread.div_ceil(2.0) as u32)
}

/// The arithmetic of the stereo PCM the audio decoder ends in.
///
/// The reference converts `[stereo, samples]` float waveforms to `[samples, 2]` `int16`:
/// `clamp(x, −1, 1) · 32767`, `torch.round` (half to even), then the integer cast. `pcm_interleave`
/// states the same thing on the device; this is the one reference an implementation is checked
/// against.
pub struct Pcm;

impl Pcm {
    /// The full-scale sample value: the scale the reference multiplies by.
    pub const FULL_SCALE: f32 = 32767.0;

    /// One sample's PCM value.
    ///
    /// `x` is clamped to [-1, 1] first, so the result never leaves `[-32767, 32767]` and the cast is
    /// total.
    pub fn quantize(x: f32) -> i16 {
        let clamped = x.clamp(-1.0, 1.0) * Self::FULL_SCALE;
        // `f32::round_ties_even` is `torch.round`'s rule; a value that is not finite cannot leave the
        // clamp, so the cast is defined.
        clamped.round_ties_even() as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint's audio VAE config (`audio_vae/config.json`).
    pub(crate) fn audio_config() -> AudioVaeConfig {
        let sidecar = serde_json::json!({
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
            "latents_mean": vec![0.0; 32],
            "latents_std": vec![1.0; 32],
        });
        AudioVaeConfig::parse(&sidecar).expect("the checkpoint's audio VAE sidecar")
    }

    #[test]
    fn the_stage_table_is_the_checkpoints() {
        let stages = stages(&audio_config()).expect("stages");
        let table: Vec<(u32, u32, u32, u32, u32)> = stages
            .iter()
            .map(|s| (s.channels_in, s.channels_out, s.rate, s.kernel, s.padding))
            .collect();
        assert_eq!(
            table,
            vec![
                (1024, 512, 5, 9, 2),
                (512, 256, 5, 9, 2),
                (256, 128, 2, 4, 1),
                (128, 64, 2, 4, 1),
                (64, 32, 2, 4, 1),
                (32, 16, 2, 4, 1),
                (16, 8, 2, 4, 1),
            ]
        );
    }

    #[test]
    fn the_decoder_hop_is_eight_hundred_samples_per_latent() {
        let config = audio_config();
        assert_eq!(samples_per_latent(&config), 800);
        assert_eq!(latent_frames(&config, 800).expect("one frame"), 1);
        assert_eq!(latent_frames(&config, 801).expect("two frames"), 2);
        assert_eq!(latent_frames(&config, 0).is_err(), true);
        assert_eq!(track_samples(&config, 5), 4000);
        // A five-second track at 32 kHz.
        let samples = 5 * 32000;
        let frames = latent_frames(&config, samples).expect("frames");
        assert_eq!(frames, 200);
        assert_eq!(track_samples(&config, frames), u64::from(samples));
    }

    #[test]
    fn the_alias_free_activation_keeps_its_length() {
        let activation = Activation::bigvgan();
        assert_eq!(activation.ratio, 2);
        assert_eq!(activation.kernel, 12);
        for length in [1u32, 2, 7, 40, 1000, 4000] {
            assert_eq!(activation.length(length), length, "length {length}");
        }
        let resample = Resample::new(2, 12).expect("the checkpoint's resampler");
        assert_eq!(resample.pad, 5);
        assert_eq!(resample.pad_left, 15);
        assert_eq!(resample.pad_right, 15);
        assert_eq!(resample.transposed_length(40), 110);
        assert_eq!(resample.upsample_length(40), 80);
        assert_eq!(resample.downsample_padding(), (5, 6));
        assert_eq!(resample.downsample_length(80), 40);
    }

    #[test]
    fn a_resampler_refuses_a_kernel_its_ratio_does_not_divide() {
        assert!(Resample::new(2, 11).is_err());
        assert!(Resample::new(1, 12).is_ok());
        assert!(Resample::new(0, 12).is_err());
        assert!(Activation::new(3, 4).is_err());
    }

    #[test]
    fn the_receptive_field_is_the_references_thirty_latent_frames() {
        // Computed from the reference `receptive_field`: spread 58.31 output samples, one side 30.
        assert_eq!(receptive_field(&audio_config()).expect("field"), 30);
    }

    #[test]
    fn pcm_rounds_half_to_even_and_clamps() {
        assert_eq!(Pcm::quantize(0.0), 0);
        assert_eq!(Pcm::quantize(1.0), 32767);
        assert_eq!(Pcm::quantize(-1.0), -32767);
        assert_eq!(Pcm::quantize(2.0), 32767);
        assert_eq!(Pcm::quantize(-2.0), -32767);
        // 0.5 / 32767 lies halfway between 0 and 1: to even is 0; 1.5 / 32767 rounds to 2.
        assert_eq!(Pcm::quantize(0.5 / Pcm::FULL_SCALE), 0);
        assert_eq!(Pcm::quantize(1.5 / Pcm::FULL_SCALE), 2);
        assert_eq!(Pcm::quantize(2.5 / Pcm::FULL_SCALE), 2);
    }
}
