//! H3's audio VAE decoder: the latent restore and input projection around a BigVGAN generator, and
//! the stereo waveform it ends in.
//!
//! Diffusers' `autoencoders/autoencoder_kl_minimax_h3_audio.py` is the reference. The decoder
//! restores the latent's channel statistics, projects it to `latent_dim` with a 1×1 convolution
//! (`dec_in_proj`), and runs the checkpoint's BigVGAN (`decoder`, [`nn::vocoder::BigVgan`]): a
//! kernel-7 pre-convolution, transposed-convolution upsampling stages of anti-aliased
//! multi-periodicity blocks, one alias-free activation and a kernel-7 post-convolution to mono
//! without a bias, whose output the runtime clamps to [-1, 1] as it quantizes the PCM.
//!
//! ## Precision
//!
//! The audio decoder is FP32 throughout, unquantized and unconverted. Its weight-normalized
//! convolutions are folded once at the loading boundary, so no layer carries the magnitude and
//! direction pair, and its convolutions run through cuDNN without TF32.

use crate::{AudioVaeConfig, Error, Result};
use nn::layers::{Bias, ChannelAffine, Conv1d, ConvSpec};
use nn::vocoder::{BigVgan, BigVganSpec};
use nn::{DType, DecodedMedia, Decoder, Forward, Shape, StateTable, TensorView, WeightLoader};

/// The alias-free activations' up- and down-sampling ratio: BigVGAN's, which the config does not
/// state.
const ACTIVATION_RATIO: u32 = 2;

/// The channels of the waveform a decode writes: a stereo pair.
///
/// Public because a caller sizes `DecodedMedia.audio` before a graph exists - the waveform is
/// `[PCM_CHANNELS, track_samples(config, frames)]` - and a channel count restated outside this crate
/// would be a second statement of the checkpoint's.
pub const PCM_CHANNELS: u32 = 2;

/// The samples a latent frame count decodes to.
pub fn track_samples(config: &AudioVaeConfig, latent_frames: u32) -> u64 {
    u64::from(latent_frames) * u64::from(config.decoder_hop())
}

/// The latent statistics the decoder inverts before its first convolution.
///
/// A generated audio latent is normalized per channel; the decoder restores it with
/// `value * std + mean` (`LatentDecoder` in the reference) before the input projection. The two
/// vectors are load-time constants of the checkpoint, one entry per latent channel.
#[derive(Clone, PartialEq, Debug)]
pub struct Statistics {
    /// The per-channel mean.
    pub mean: Vec<f32>,
    /// The per-channel standard deviation.
    pub std: Vec<f32>,
}

impl Statistics {
    /// The statistics the config states, one entry per latent channel.
    pub fn of(config: &AudioVaeConfig) -> Result<Self> {
        let channels = config.latent_channels as usize;
        if config.latents_mean.len() != channels || config.latents_std.len() != channels {
            return Err(Error::config(format!(
                "the audio latent statistics hold {} means and {} deviations for {channels} channels",
                config.latents_mean.len(),
                config.latents_std.len()
            )));
        }
        for (index, (mean, std)) in config
            .latents_mean
            .iter()
            .zip(config.latents_std.iter())
            .enumerate()
        {
            if !mean.is_finite() || !std.is_finite() || *std == 0.0 {
                return Err(Error::config(format!(
                    "the audio latent channel {index} has mean {mean} and deviation {std}"
                )));
            }
        }
        Ok(Self {
            mean: config.latents_mean.clone(),
            std: config.latents_std.clone(),
        })
    }
}

/// The architecture of the checkpoint's BigVGAN, from its audio VAE config.
pub fn generator(config: &AudioVaeConfig) -> BigVganSpec {
    BigVganSpec {
        in_channels: config.latent_dim,
        channels: config.decoder_dim,
        upsample_rates: config.decoder_rates.clone(),
        upsample_kernels: config.decoder_kernel_sizes.clone(),
        resblock_kernels: config.resblock_kernel_sizes.clone(),
        resblock_dilations: config.resblock_dilation_sizes.clone(),
        activation_ratio: ACTIVATION_RATIO,
        // `use_bias_at_final=False`; the final `tanh` is off and the clamp is the PCM step's.
        final_bias: Bias::Absent,
    }
}

/// The audio decoder: the checkpoint's restore, input projection and BigVGAN, one call per run of
/// latent frames.
///
/// Layout everywhere is `[batch, channels, samples]`, the form the one-dimensional convolutions
/// take. The batch is two because the reference decodes a stereo pair of identical rows and
/// interleaves them; nothing in the network mixes the rows, so both output channels carry the same
/// signal.
pub struct AudioDecoder {
    config: AudioVaeConfig,
    /// `value · std + mean` per latent channel, a product then a sum.
    denormalize: ChannelAffine,
    input: Conv1d,
    generator: BigVgan,
    /// Moves the track from the denoiser's rows into the decoder's channel-major layout.
    track: nn::layers::Copy,
}

impl AudioDecoder {
    /// The sample clock the checkpoint was trained at, in hertz.
    ///
    /// It is the audio vae config's own number (`audio_vae/config.json`'s `sampling_rate`), not a
    /// constant this crate types: the muxer builds the artifact's timeline from it.
    pub fn sampling_rate(&self) -> u32 {
        self.config.sampling_rate
    }

    /// The audio decoder the checkpoint's `dec_in_proj` and `decoder` modules state, read in FP32.
    pub fn load(config: &AudioVaeConfig, loader: &mut WeightLoader) -> Result<Self> {
        let statistics = Statistics::of(config)?;
        let channels = u64::from(config.latent_channels);
        let denormalize = {
            let mut constants = loader.scope("audio_vae.denormalize");
            // The latents are `[batch, channels, frames]`: the channel is axis one.
            ChannelAffine::new(
                constants.constant_f32("scale", &[channels], &statistics.std)?,
                constants.constant_f32("shift", &[channels], &statistics.mean)?,
                1,
            )?
        };
        // The audio decoder computes in FP32 throughout, so every weight is read in it whatever the
        // checkpoint stores.
        let mut root = loader.scope("");
        let mut root = root.pinned(DType::F32);
        // A plain 1x1 convolution in the reference, outside the weight-normalized generator.
        let input = Conv1d::load(
            &mut root.scope("dec_in_proj"),
            ConvSpec::new(config.latent_channels, config.latent_dim, [1]),
        )?;
        let generator = BigVgan::load(&mut root.scope("decoder"), &generator(config))?;
        Ok(Self {
            config: config.clone(),
            denormalize,
            input,
            generator,
            track: nn::layers::Copy::new(DType::F32, nn::layers::Access::Element)?,
        })
    }

    /// The denoiser's track in the layout [`Self::forward`] reads.
    ///
    /// The denoiser publishes its audio rows as they are: `[frames, 2 · latent_channels]` over the
    /// bytes of `[2 · frames, latent_channels]`, the first stereo channel's rows before the second's.
    /// The decoder's convolutions run along time, so each stereo channel's latent channels must
    /// become the outer axis: this is the reference's `reshape(2, frames, channels).permute(0, 2, 1)`.
    fn channel_major(&self, f: &Forward, track: TensorView) -> Result<TensorView> {
        let channels = u64::from(self.config.latent_channels);
        let stereo = u64::from(PCM_CHANNELS);
        let shape = track.shape;
        if shape.rank() != 2 || shape.dim(1) != stereo * channels {
            return Err(Error::config(format!(
                "{:?} is not a track of [frames, {}]",
                shape.dims(),
                stereo * channels
            )));
        }
        let frames = shape.dim(0);
        let major = self.buffer(f, self.config.latent_channels, frames)?;
        // Four axes are what a block copy moves: `[stereo, channels, 1, frames]`.
        let block = Shape::new(&[stereo, channels, 1, frames]);
        let source = TensorView::with_strides(
            track.offset_elements(0),
            track.dtype,
            block,
            nn::Strides::new(&[frames * channels, 1, 1, channels]),
        );
        let strides = [channels * frames, frames, frames, 1];
        let destination = TensorView::with_strides(
            major.offset_elements(0),
            major.dtype,
            block,
            nn::Strides::new(&strides),
        );
        self.track
            .forward(f, source, destination, &strides.map(|stride| stride as i64))?;
        Ok(major)
    }

    /// Decode `latents [2, latent_channels, frames]` into the waveform `out [2, samples]` FP32.
    ///
    /// The two rows of the latent are the stereo pair the reference decodes; both hold the same
    /// latent, so the network's two output rows are equal. Every intermediate is FP32, and the
    /// length grows only through the generator's transposed convolutions.
    pub fn forward(&self, f: &Forward, latents: TensorView, out: TensorView) -> Result<()> {
        let shape = latents.shape;
        if shape.rank() != 3
            || shape.dim(0) != u64::from(PCM_CHANNELS)
            || shape.dim(1) != u64::from(self.config.latent_channels)
        {
            return Err(Error::config(format!(
                "{:?} is not [{PCM_CHANNELS}, {}, frames] latents",
                shape.dims(),
                self.config.latent_channels
            )));
        }
        let length = shape.dim(2);
        // The reference restores each latent channel before the decoder's first convolution.
        let restored = self.buffer(f, self.config.latent_channels, length)?;
        self.denormalize.forward(f, latents, restored)?;
        let projected = self.buffer(f, self.config.latent_dim, length)?;
        self.input.forward(f, restored, projected)?;
        let samples = self.generator.samples(length);
        let planes = [u64::from(PCM_CHANNELS), samples];
        if out.dtype != DType::F32 || out.shape.dims() != planes || !out.is_row_major() {
            return Err(Error::config(format!(
                "the waveform is a contiguous f32 {planes:?}, not {} {:?}",
                out.dtype,
                out.shape.dims()
            )));
        }
        // The generator's batch axis is the stereo pair and its channel axis one.
        let mono = out.reshape(Shape::new(&[u64::from(PCM_CHANNELS), 1, samples]));
        Ok(self.generator.forward(f, projected, mono)?)
    }

    /// A FP32 buffer of the audio network's layout.
    fn buffer(&self, f: &Forward, channels: u32, length: u64) -> Result<TensorView> {
        Ok(f.arena().tensor(
            DType::F32,
            Shape::new(&[u64::from(PCM_CHANNELS), u64::from(channels), length]),
        )?)
    }
}

impl Decoder for AudioDecoder {
    /// The element type this decoder computes in.
    ///
    /// `F32`: the audio decoder's arithmetic is single precision throughout, so the latents column
    /// and the waveform it writes are `f32` (`DecodedMedia::audio`).
    fn element_type(&self) -> nn::DType {
        nn::DType::F32
    }

    fn layout_tables(&self, _layout: &nn::capability::Layout) -> nn::Result<Vec<StateTable>> {
        // Every table this decoder reads is a weight, bound at load: nothing follows from the
        // layout, so the runner uploads nothing for it.
        Ok(Vec::new())
    }

    fn prepare_layout(
        &self,
        _layout: &nn::capability::Layout,
        out: &mut [nn::TableValues],
    ) -> nn::Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        Err(Error::config(format!(
            "the audio decoder has no layout tables; {} buffers were offered",
            out.len()
        ))
        .into())
    }

    /// The waveform one decode of `audio` writes: the track's latent frames times the decoder's hop,
    /// in [`PCM_CHANNELS`] channels.
    fn pcm_window(&self, audio: &nn::capability::LatentShape) -> nn::Result<(u64, u32)> {
        Ok((track_samples(&self.config, audio.frames), PCM_CHANNELS))
    }

    /// Decode one run of latent frames into its waveform.
    ///
    /// `state` must be empty, `out.audio` is the request's `[2, samples]` f32 waveform and
    /// `out.video` must be absent: this path decodes sound only.
    fn decode(
        &self,
        f: &Forward,
        latents: TensorView,
        state: &[TensorView],
        out: DecodedMedia,
    ) -> nn::Result<()> {
        if !state.is_empty() {
            return Err(Error::config(format!(
                "the audio decoder reads no layout table, {} were given",
                state.len()
            ))
            .into());
        }
        let audio = out.audio.ok_or_else(|| {
            Error::config("an audio decode writes the request's waveform".to_owned())
        })?;
        if out.video.is_some() {
            return Err(
                Error::config("the audio decoder writes no video raster".to_owned()).into(),
            );
        }
        let latents = self.channel_major(f, latents)?;
        Ok(self.forward(f, latents, audio)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {

    use super::*;

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
    fn a_latent_frame_decodes_to_eight_hundred_samples() {
        let config = audio_config();
        assert_eq!(track_samples(&config, 5), 4000);
        assert_eq!(track_samples(&config, 200), 5 * 32000);
    }

    #[test]
    fn latent_statistics_of_the_wrong_count_or_a_zero_deviation_are_refused() {
        let config = audio_config();
        let statistics = Statistics::of(&config).expect("the statistics");
        assert_eq!(statistics.mean.len(), 32);
        assert_eq!(statistics.std.len(), 32);
        let mut broken = config.clone();
        broken.latents_mean.pop();
        assert!(Statistics::of(&broken).is_err());
        let mut broken = config.clone();
        broken.latents_std[3] = 0.0;
        assert!(Statistics::of(&broken).is_err());
    }

    #[test]
    fn the_generator_is_the_checkpoints_bigvgan() {
        let spec = generator(&audio_config());
        assert_eq!(spec.in_channels, 2048);
        assert_eq!(spec.channels, 1024);
        assert_eq!(spec.upsample_rates.iter().product::<u32>(), 800);
        assert_eq!(spec.resblock_kernels, vec![3, 7, 11]);
        assert_eq!(spec.final_bias, Bias::Absent);
    }
}
