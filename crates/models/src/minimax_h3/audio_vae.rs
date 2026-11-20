//! The audio VAE: the BigVGAN decoder that turns an audio latent into a stereo waveform, and
//! the DAC encoder that turns a reference track into its latent.
//!
//! The reference is `DacAudioVAE`
//! (`sglang:python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_audio_vae/audio_vae.py:230-339`).
//!
//! ## The decoder
//!
//! A decode restores the latent's channel statistics (`latent · std + mean`), projects it to
//! `latent_dim` features with a one-by-one convolution (`dec_in_proj`) and runs the
//! checkpoint's BigVGAN generator (`decoder`, [`BigVgan`]) to one waveform per stereo channel.
//! The two channels are the batch: nothing in the network mixes them. The generator's output
//! is not clamped here: execution clamps to `[-1, 1]` as it quantizes the PCM.
//!
//! ## The encoder
//!
//! An encode runs the DAC encoder (`encoder.block`, [`DacEncoder`]) to `latent_dim` features
//! per latent frame, narrows them to the latent channels with the checkpoint's attention
//! projection (`pre_block`), takes the posterior's mean (`mean_proj`, no sample is drawn) and
//! normalizes it, `(mean − latents_mean) / latents_std`
//! (`sglang:python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:422-447`).
//! The projection's attention is causal: a latent frame depends only on the audio before it.
//!
//! ## Precision
//!
//! Both directions are FP32 throughout, whatever the deployment's preset: every weight is read
//! in FP32, the weight-normalized convolutions are folded once at load, and no convolution
//! allows TF32, because the values are amplitudes.

use nn::functional::{self, Activation, Rounding};
use nn::modules::{
    BigVgan, BigVganConfig, Conv1d, ConvConfig, DacEncoder, DacEncoderConfig, GateUp, LayerNorm,
    LayerNormConfig, Linear, LinearConfig, Mlp, MlpConfig, Parallel, QkvBias, QkvLinear,
    QkvLinearConfig, QkvNames,
};
use nn::{DType, Decoder, LatentShape, Result, Split, Tensor, VarBuilder};
use processing::minimax_h3::AUDIO_CHANNELS;
use proto::video::Layout;

use crate::minimax_h3::{AudioVaeConfig, ID};

/// The up- and down-sampling ratio of BigVGAN's alias-free activations, which the config does
/// not state (`sglang:python/sglang/multimodal_gen/runtime/models/vaes/alias_free.py:154-166`).
const ACTIVATION_RATIO: u32 = 2;

/// `nn.LayerNorm`'s default epsilon, which the reference's attention projection keeps.
const LAYER_NORM_EPS: f32 = 1e-5;

/// The samples a track of `samples` is right-padded to with zeros before it is encoded: the
/// next whole number of `hop`-sample latent frames (the reference's `preprocess`,
/// `audio_vae.py:314-323`).
///
/// # Errors
///
/// A track of no sample, which encodes to no latent frame, or a hop of no sample.
pub fn padded_samples(samples: u64, hop: u64) -> Result<u64> {
    if samples == 0 || hop == 0 {
        return Err(ID.request(format!(
            "an audio track of {samples} samples in {hop}-sample latent frames encodes to \
             nothing"
        )));
    }

    Ok(samples.div_ceil(hop) * hop)
}

/// The latent space's per-channel statistics as two constants of `shape`, published under
/// `vb` as `latents_mean` and `latents_std`.
fn statistics(
    config: &AudioVaeConfig,
    vb: &VarBuilder<'_, '_>,
    shape: &[usize],
) -> Result<(Tensor, Tensor)> {
    let channels = config.latent_channels as usize;
    if config.latents_mean.len() != channels || config.latents_std.len() != channels {
        return Err(ID.config(format!(
            "the audio latent statistics hold {} means and {} deviations for {channels} channels",
            config.latents_mean.len(),
            config.latents_std.len()
        )));
    }

    Ok((
        vb.constant("latents_mean", DType::F32, shape, &config.latents_mean)?,
        vb.constant("latents_std", DType::F32, shape, &config.latents_std)?,
    ))
}

/// The audio decoder: the latent restore, the input projection and the BigVGAN generator.
pub struct AudioDecoder {
    /// `[latent_channels, 1]` FP32: the latent space's per-channel mean, over the frames.
    latents_mean: Tensor,
    /// `[latent_channels, 1]` FP32: the latent space's per-channel deviation.
    latents_std: Tensor,
    /// A plain one-by-one convolution, outside the weight-normalized generator.
    dec_in_proj: Conv1d,
    decoder: BigVgan,
    latent_channels: usize,
    sampling_rate: u32,
    /// The path the decoder was loaded under: its profiler range's.
    path: String,
}

impl AudioDecoder {
    /// The decoder at `vb`, the root of the checkpoint's `audio_vae` stage: `dec_in_proj` and
    /// the generator `decoder`, every tensor read in FP32.
    ///
    /// # Errors
    ///
    /// Latent statistics of another count than the channels, a generator the config does not
    /// describe, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(config: &AudioVaeConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let vb = vb.to_dtype(DType::F32);
        let channels = config.latent_channels as usize;
        let (latents_mean, latents_std) = statistics(config, &vb, &[channels, 1])?;

        // `use_bias_at_final=False`, and `use_tanh_at_final=False`: the clamp is execution's.
        let generator = BigVganConfig {
            in_channels: config.latent_dim,
            channels: config.decoder_dim,
            upsample_rates: config.decoder_rates.clone(),
            upsample_kernels: config.decoder_kernel_sizes.clone(),
            resblock_kernels: config.resblock_kernel_sizes.clone(),
            resblock_dilations: config.resblock_dilation_sizes.clone(),
            activation_ratio: ACTIVATION_RATIO,
            final_bias: false,
        };

        Ok(Self {
            latents_mean,
            latents_std,
            dec_in_proj: Conv1d::load(
                &ConvConfig::new(config.latent_channels, config.latent_dim, [1]),
                vb.pp("dec_in_proj"),
            )?,
            decoder: BigVgan::load(&generator, vb.pp("decoder"))?,
            latent_channels: channels,
            sampling_rate: config.sampling_rate,
            path: vb.pp("decoder").path().to_owned(),
        })
    }

    /// The sample clock of the waveform a decode writes, in hertz: the checkpoint's own
    /// (`audio_vae/config.json`'s `sampling_rate`).
    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }

    /// The track of `frames` latent frames decoded from `latents` into `audio`
    /// ([`Decoder::decode`]).
    fn decode_track(&self, frames: usize, latents: &Tensor, audio: &Tensor) -> Result<()> {
        let (stereo, channels) = (AUDIO_CHANNELS as usize, self.latent_channels);
        if latents.elem_count() != stereo * frames * channels || latents.dtype() != DType::F32 {
            return Err(ID.request(format!(
                "a track of {frames} latent frames is {stereo} · {frames} FP32 rows of \
                 {channels} channels, not {:?} {:?}",
                latents.dtype(),
                latents.dims()
            )));
        }
        let samples = self.decoder.samples(frames);
        if audio.dims() != [stereo, samples] || audio.dtype() != DType::F32 {
            return Err(ID.request(format!(
                "the waveform of {frames} latent frames is FP32 [{stereo}, {samples}], not \
                 {:?} {:?}",
                audio.dtype(),
                audio.dims()
            )));
        }

        // The denoiser's rows are channel-major, the first stereo channel's frames before the
        // second's; the convolutions run along time, so the latent channels become the
        // middle axis (the reference's `reshape(2, frames, channels).permute(0, 2, 1)`).
        let track = latents
            .reshape((stereo, frames, channels))?
            .permute((0, 2, 1))?;
        let restored = track
            .broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)?;

        let waveform = self
            .decoder
            .forward(&self.dec_in_proj.forward(&restored)?)?;
        Ok(audio.slice_set(&waveform.reshape((stereo, samples))?, 0, 0)?)
    }
}

impl Decoder for AudioDecoder {
    /// FP32: the latents a decode reads and the waveform it writes.
    fn element_type(&self) -> DType {
        DType::F32
    }

    /// The track's latent frames times the product of the generator's rates, in
    /// [`AUDIO_CHANNELS`] channels.
    fn pcm_window(&self, audio: &LatentShape) -> nn::Result<(u64, u32)> {
        let samples = self.decoder.samples(audio.frames as usize);

        Ok((samples as u64, AUDIO_CHANNELS))
    }

    /// Decodes one track into its waveform.
    ///
    /// `latents` holds the track of `layout.audio` as the denoiser publishes it: the elements
    /// of `[2 · frames, latent_channels]` FP32 rows, the first stereo channel's frames before
    /// the second's, in any shape of that element count. `audio` is the `[2, samples]` FP32
    /// waveform of [`Self::pcm_window`], and `video` must be absent.
    fn decode(
        &self,
        layout: &Layout,
        latents: &Tensor,
        video: Option<&Tensor>,
        audio: Option<&Tensor>,
    ) -> nn::Result<()> {
        let (Some(track), Some(audio), None) = (layout.audio, audio, video) else {
            return Err(
                ID.request("an audio decode writes the waveform of a layout's track and no raster")
            );
        };

        nn::profiler::range("AudioDecoder", &self.path, || {
            self.decode_track(track.frames as usize, latents, audio)
        })
    }
}

/// The attention projection that narrows the encoder's features to the latent channels (the
/// reference's `AttnProjection` over `CausalAttention` and `GeGluMlp`, `audio_vae.py:21-136`).
///
/// ```text
/// h   = proj(norm3(x)) + attn.proj(pool(attention(qkv(norm1(x)))))
/// out = mlp(norm2(h)) + h        mlp(y) = w2(gelu_tanh(w0(norm(y))) · w1(norm(y)))
/// ```
///
/// The attention is causal over a track's latent frames. Its heads are averaged and each
/// head's width pooled down to the latent channels in one pass
/// ([`functional::adaptive_avg_pool2d`]).
struct AttnProjection {
    norm3: LayerNorm,
    proj: Linear,
    norm1: LayerNorm,
    /// `attn.qkv` with the bias windows `attn.{q_bias, zero_k_bias, v_bias}`. The key window
    /// is a tensor the checkpoint stores like the other two, so it is read, not assumed zero.
    qkv: QkvLinear,
    /// `attn.proj`, over the pooled heads.
    attn_proj: Linear,
    norm2: LayerNorm,
    /// `mlp.norm`.
    mlp_norm: LayerNorm,
    mlp: Mlp,
    heads: usize,
    head_dim: usize,
    latent_channels: u32,
}

impl AttnProjection {
    /// The projection at `vb` (`pre_block`), from `latent_dim` features to the latent
    /// channels.
    ///
    /// # Errors
    ///
    /// A latent width the latent channels do not divide or do not narrow (the reference then
    /// projects to another width, which no released checkpoint states), or a tensor the
    /// checkpoint lacks.
    fn load(config: &AudioVaeConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (wide, channels) = (config.latent_dim, config.latent_channels);
        let heads = config.num_attention_heads;
        if channels == 0
            || wide <= channels
            || !wide.is_multiple_of(channels)
            || heads == 0
            || !wide.is_multiple_of(heads)
        {
            return Err(ID.config(format!(
                "the audio encoder projects {wide} features to {channels} latent channels over \
                 {heads} heads: the channels and the heads each divide the features, which are \
                 wider than the channels"
            )));
        }

        let norm = |width: u32, name: &str| {
            LayerNorm::load(
                &LayerNormConfig {
                    width,
                    eps: LAYER_NORM_EPS,
                    bias: true,
                },
                vb.pp(name),
            )
        };
        let linear = |inputs: u32, outputs: u32, vb: VarBuilder<'_, '_>| {
            Linear::load(
                &LinearConfig {
                    inputs,
                    outputs,
                    bias: true,
                    parallel: Parallel::None,
                },
                vb,
            )
        };
        let attn = vb.pp("attn");
        let mlp = vb.pp("mlp");

        Ok(Self {
            norm3: norm(wide, "norm3")?,
            proj: linear(wide, channels, vb.pp("proj"))?,
            norm1: norm(wide, "norm1")?,
            qkv: QkvLinear::load(
                &QkvLinearConfig {
                    hidden: wide,
                    heads,
                    kv_heads: heads,
                    head_dim: wide / heads,
                    gated: false,
                    names: QkvNames::Fused("qkv"),
                    bias: QkvBias::Windows(["q_bias", "zero_k_bias", "v_bias"]),
                    split: Split::WHOLE,
                },
                attn.clone(),
            )?,
            attn_proj: linear(channels, channels, attn.pp("proj"))?,
            norm2: norm(channels, "norm2")?,
            mlp_norm: LayerNorm::load(
                &LayerNormConfig {
                    width: channels,
                    eps: LAYER_NORM_EPS,
                    bias: true,
                },
                mlp.pp("norm"),
            )?,
            // `hidden_features = out_dim · mlp_ratio`, with the reference's ratio of two.
            mlp: Mlp::load(
                &MlpConfig {
                    hidden: channels,
                    intermediate: 2 * channels,
                    activation: Activation::GeluTanh,
                    rounding: Rounding::Fused,
                    limit: None,
                    gate_up: GateUp::Split {
                        gate: "w0",
                        up: "w1",
                    },
                    down: "w2",
                    bias: true,
                    split: Split::WHOLE,
                },
                mlp,
            )?,
            heads: heads as usize,
            head_dim: (wide / heads) as usize,
            latent_channels: channels,
        })
    }

    /// One track's features `x`, `[frames, latent_dim]` in FP32, projected:
    /// `[frames, latent_channels]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let frames = x.dims()[0];
        let heads = (frames, self.heads, self.head_dim);

        let shortcut = self.proj.forward(&self.norm3.forward(x)?)?;

        // Causal attention over the frames, at the default scale of one over the root of a
        // head's width.
        let normalized = self.norm1.forward(x)?;
        let attended = functional::dense_attention(
            &self.qkv.query().forward(&normalized)?.reshape(heads)?,
            &self.qkv.key().forward(&normalized)?.reshape(heads)?,
            &self.qkv.value().forward(&normalized)?.reshape(heads)?,
            (self.head_dim as f32).sqrt().recip(),
            true,
            None,
        )?;

        // The mean over the heads and the pool of a head's width, as one pool whose rows are
        // the heads.
        let pooled = functional::adaptive_avg_pool2d(&attended, [1, self.latent_channels])?
            .reshape((frames, self.latent_channels as usize))?;
        let hidden = (shortcut + self.attn_proj.forward(&pooled)?)?;

        let update = self
            .mlp
            .forward(&self.mlp_norm.forward(&self.norm2.forward(&hidden)?)?)?;
        Ok((update + hidden)?)
    }
}

/// The audio encoder: a reference track's waveform to the normalized mean of its posterior.
pub struct AudioEncoder {
    encoder: DacEncoder,
    pre_block: AttnProjection,
    /// A one-by-one convolution over the projected features: the posterior's mean.
    mean_proj: Conv1d,
    /// `[latent_channels]` FP32: the latent space's per-channel mean.
    latents_mean: Tensor,
    /// `[latent_channels]` FP32: the latent space's per-channel deviation.
    latents_std: Tensor,
    latent_channels: usize,
    /// The path the encoder was loaded under: its profiler range's.
    path: String,
}

impl AudioEncoder {
    /// The encoder at `vb`, the root of the checkpoint's `audio_vae` stage: `encoder.block`,
    /// `pre_block` and `mean_proj`, every tensor read in FP32.
    ///
    /// # Errors
    ///
    /// A config the DAC encoder or the projection refuses, statistics of another count than
    /// the channels, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(config: &AudioVaeConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let vb = vb.to_dtype(DType::F32);
        let channels = config.latent_channels as usize;
        let (latents_mean, latents_std) = statistics(config, &vb, &[channels])?;
        let encoder = vb.pp("encoder").pp("block");

        Ok(Self {
            encoder: DacEncoder::load(
                &DacEncoderConfig {
                    channels: config.encoder_dim,
                    strides: config.encoder_rates.clone(),
                    latent_channels: config.latent_dim,
                },
                encoder.clone(),
            )?,
            pre_block: AttnProjection::load(config, vb.pp("pre_block"))?,
            mean_proj: Conv1d::load(
                &ConvConfig::new(config.latent_channels, config.latent_channels, [1]),
                vb.pp("mean_proj"),
            )?,
            latents_mean,
            latents_std,
            latent_channels: channels,
            path: vb.pp("encoder").path().to_owned(),
        })
    }

    /// The samples one latent frame covers: the product of the encoder's rates.
    pub fn hop(&self) -> u64 {
        self.encoder.hop() as u64
    }

    /// A track's normalized latent rows: `samples`, `[2, 1, padded]` in FP32, one row per
    /// stereo channel, right-padded with zeros to whole hops ([`padded_samples`]), to
    /// `[2 · frames, latent_channels]` in FP32, the first channel's frames before the
    /// second's: the layout the denoiser's audio input projection reads.
    ///
    /// The stereo pair is the batch: each channel is encoded and attended alone. A mono track
    /// is expanded to two identical channels before it reaches the encoder, as the reference
    /// expands it; read as one row it would encode half a track.
    ///
    /// # Errors
    ///
    /// A track that is not the stereo pair in FP32, a length that is not whole hops, or what a
    /// module refuses.
    pub fn encode(&self, samples: &Tensor) -> Result<Tensor> {
        nn::profiler::range("AudioEncoder", &self.path, || self.encode_track(samples))
    }

    /// [`Self::encode`] inside its profiler range.
    fn encode_track(&self, samples: &Tensor) -> Result<Tensor> {
        let (stereo, hop) = (AUDIO_CHANNELS as usize, self.encoder.hop());
        let whole = |length: usize| length > 0 && length.is_multiple_of(hop);

        let frames = match samples.dims() {
            &[batch, 1, length]
                if batch == stereo && whole(length) && samples.dtype() == DType::F32 =>
            {
                length / hop
            }
            dims => {
                return Err(ID.request(format!(
                    "the audio encoder reads the stereo pair as FP32 [{stereo}, 1, samples] of \
                     whole {hop}-sample latent frames, a mono track expanded to two identical \
                     channels and a short one right-padded with zeros, not {:?} {dims:?}",
                    samples.dtype()
                )));
            }
        };

        // `[2, latent_dim, frames]` features; the projection reads each channel's frames as
        // rows and never mixes the two.
        let features = self.encoder.forward(samples)?;
        let projected = (0..stereo)
            .map(|channel| {
                let rows = features.get(channel)?.t()?.contiguous()?;
                self.pre_block.forward(&rows)
            })
            .collect::<Result<Vec<Tensor>>>()?;

        // The posterior's mean over `[2, latent_channels, frames]`, then the latent space's
        // normalization over `[2, frames, latent_channels]`, a subtraction and a division as
        // the reference's (`reference_encoding.py:441-447`).
        let mean = self
            .mean_proj
            .forward(&Tensor::stack(&projected, 0)?.permute((0, 2, 1))?)?;
        let normalized = mean
            .permute((0, 2, 1))?
            .broadcast_sub(&self.latents_mean)?
            .broadcast_div(&self.latents_std)?;

        Ok(normalized.reshape((stereo * frames, self.latent_channels))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_track_is_padded_to_whole_latent_frames() {
        // A partial frame rounds up to the next whole one, and a whole track is unchanged.
        assert_eq!(padded_samples(801, 800).expect("padded"), 1600);
        assert_eq!(padded_samples(1600, 800).expect("whole"), 1600);
        assert_eq!(padded_samples(1, 800).expect("one sample"), 800);

        assert!(
            padded_samples(0, 800).is_err(),
            "no sample encodes to nothing"
        );
        assert!(padded_samples(800, 0).is_err(), "a hop of no sample");
    }
}
