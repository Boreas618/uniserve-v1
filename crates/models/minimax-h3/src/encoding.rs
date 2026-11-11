//! The H3 condition encoders: the causal video encoder that turns a clip of raster frames into
//! posterior moments, and the audio encoder that turns a reference waveform into latents.
//!
//! Only `ref2va` reads them: a reference video or audio clip is encoded once at
//! the request boundary and the latents join the packed sequence as clean condition rows. The rest of
//! this module is the encoders' arithmetic — the stage table and the causal extents — and their
//! parameter tables. The numerical bodies follow the shared layers, as the decoders' do.
//!
//! ## Precision
//!
//! Every convolution of the video encoder runs FP32 with TF32 allowed because the checkpoint's VAE
//! encoder is evaluated at the quality preset. The audio encoder's convolutions and its `Snake`
//! activations run FP32 and keep no TF32: its magnitudes are amplitudes, where the reference
//! compares samples, not feature maps.
//!
//! **The audio encoder's attention is FP32 throughout** — query, key and value projections, the
//! scores, the softmax and the output — whatever the checkpoint's preset says: FP32 attention for
//! `ref2va`'s audio encoder and the VSA compression branch, as a cuBLASLt SGEMM plus a softmax
//! kernel program. The model declares it instead of leaning on `torch.autocast`, and it is the
//! opposite of the video VAE decoder's rule (one element type for both parameters and residual,
//! whatever the deployment's preset - F32 under `dtype = "auto"`, because the checkpoint stores the
//! VAE in F32 and auto converts nothing; FP16 is a *converted* deployment's element, and every size
//! halves with it). The layer is [`nn::layers::DenseAttention`] over FP32 operands; the operands'
//! element type, not a backend flag, selects the FP32 program.
//!
//! ## The encoder's input
//!
//! A reference video reaches the encoder as `[frames, height, width, 3]` **uint8** pixels — the
//! same raster the decoder writes (`crate::output`) — and is normalized per channel by
//! `(pixel / 255 − mean) / std` with the ImageNet constants before the input convolution. A
//! condition is cut into the frame rule's clips ([`crate::config::FRAMES`]), each encoded alone; a
//! single frame stays alone.
//!
//! ## Causal extents
//!
//! A stride-`s` temporal convolution pads two zero frames in front and none behind, so it maps `T`
//! frames to `ceil(T / s)`: one 17-frame clip becomes five latent frames, and a single frame stays
//! one. A spatial downsampling pads the bottom and the right by one pixel and strides by two, so it
//! halves an even extent exactly and floors an odd one.
//!
//! ## Tiles
//!
//! A tile is 256 input pixels square with at least 64 pixels of overlap, both multiples of the 16-pixel
//! spatial compression, so every seam spans whole latent positions. Adjacent tiles cross-fade over
//! `overlap / 16` latent positions in the encoder's output precision — unlike the decoder, which blends
//! decoded pixels, the encoder blends latents ([`crate::tiled_encode`]).

/// `nn.LayerNorm`'s default epsilon, which the reference's audio projection does not override.
const NORM_EPS: f32 = 1e-5;
use crate::config::{FRAMES, VideoVaeConfig};
use crate::packing::LATENT_COMPRESSION;
use crate::{Error, Result};

/// The channels of a condition's raster: RGB.
pub const PIXEL_CHANNELS: u32 = 3;

/// How H3's encoders read a request's decoded media, with the vision tower at `dtype`: the video
/// encoder's planar pixels normalized by the ImageNet constants ([`crate::output::PIXEL_NORM`]), the
/// vision tower's patch rows by the processor's statistics, and the audio encoder's channel rows
/// padded to whole latent hops ([`crate::config::AUDIO_HOP`]).
pub fn media_format(config: &crate::Config, dtype: DType) -> Result<nn::capability::MediaFormat> {
    Ok(nn::capability::MediaFormat {
        pixels: crate::output::PIXEL_NORM,
        patches: qwen3::vision::patch_format(config.vision()?, config.pixels, dtype),
        hop: u64::from(crate::config::AUDIO_HOP),
    })
}

/// Refuses a video VAE config whose encoder stages do not form one encoder of the H3 latent
/// space: widths, strides and blocks of one count each, no zero stride, and a spatial compression
/// of [`LATENT_COMPRESSION`].
fn check_stages(config: &VideoVaeConfig) -> Result<()> {
    let stages = config.block_out_channels.len();
    if config.temporal_downsample_factors.len() != stages
        || config.spatial_downsample_factors.len() != stages
    {
        return Err(Error::config(format!(
            "the video VAE's {stages} widths do not match its downsample factors"
        )));
    }
    if config.block_out_channels.is_empty() || config.layers_per_block == 0 {
        return Err(Error::config(
            "the video encoder has no stage or no block".to_owned(),
        ));
    }
    if config.latent_channels == 0 {
        return Err(Error::config(
            "the video VAE has no latent channel".to_owned(),
        ));
    }
    for (index, stride) in config
        .temporal_downsample_factors
        .iter()
        .chain(config.spatial_downsample_factors.iter())
        .enumerate()
    {
        if *stride == 0 {
            return Err(Error::config(format!(
                "the video encoder's stride {} is zero",
                index + 1
            )));
        }
    }
    let compression: u32 = config
        .spatial_downsample_factors
        .iter()
        .try_fold(1u32, |product, stride| product.checked_mul(*stride))
        .ok_or_else(|| Error::config("the video encoder's strides overflow".to_owned()))?;
    if compression != LATENT_COMPRESSION {
        return Err(Error::config(format!(
            "the video encoder compresses space {compression}-fold, not {LATENT_COMPRESSION}"
        )));
    }
    Ok(())
}

/// The length the encoder must be given for a track of `samples`: the ceiling's own multiple.
///
/// The padding is the encoder's latent count ([`processing::minimax_h3::audio_latents`]) times
/// the hop less the track, so a whole-frame track is
/// unchanged and any other is right-padded with zeros exactly as the reference's `unpack_samples`
/// does.
///
/// # Errors
///
/// A track of no sample, which encodes to nothing.
pub fn padded_samples(samples: u64, hop: u64) -> Result<u64> {
    if samples == 0 {
        return Err(Error::config(
            "an audio track of no samples encodes to no latent frames".to_owned(),
        ));
    }
    Ok(samples.div_ceil(hop) * hop)
}

// ---------------------------------------------------------------------------------------------
// The video encoder's device path.
// ---------------------------------------------------------------------------------------------

use nn::Forward;
use nn::blocks::{GateUp, GatedMlp, GatedMlpSpec};
use nn::family::Split;
use nn::layers::{
    AdaptiveAvgPool2d, Bias, CausalPad, Conv1d, Conv3d, ConvSpec, DenseAttention, DenseSpec,
    GroupNorm, LayerNorm, Linear, Padding, Posterior, Residual, ResnetBlock3d, ResnetSpec,
    Rounding, Transpose, WeightOrder,
};
use nn::weights::{Scope, WeightLoader};
use nn::{DType, Shape, TensorView};

use nn::codec::{DacEncoder, DacSpec};

/// The video encoder: the causal convolutional network that turns a clip's pixels into the posterior
/// moments of its latent.
///
/// One call encodes one raster: the caller tiles a clip as `Grid` states, and a band of latent rows
/// may be encoded on its own through `Grid::band`. Every convolution runs FP32 with TF32 allowed, the
/// precision the checkpoint's preset declares for the video VAE's encoder.
pub struct VideoEncoder {
    /// The raster channels the input convolution reads.
    input_channels: u32,
    /// The input convolution, over raw pixels.
    input: Conv3d,
    /// Each encoder level: its residual blocks, then the convolution that halves space and strides
    /// time.
    levels: Vec<(Vec<ResnetBlock3d>, Option<Conv3d>)>,
    /// The final normalization, which writes its padded output: one pixel on each spatial side,
    /// mirrored or replicated, and two zero frames in front, as the reference's
    /// `CausalConv3d.forward(norm=…)` pads the normalized and activated frames, so no output frame
    /// reads a later input frame.
    norm_out: GroupNorm,
    /// The moment convolution over the padded normalization, `2 · latent_channels` wide.
    conv_out: Conv3d,
    /// The one-by-one projection after the moments.
    moments: Conv3d,
    /// The latent space's per-channel affine, `(sample − mean) / std` as a one-by-one convolution.
    normalize: Conv3d,
    /// The posterior's epilogue: the sample a condition reads, from the projected moments.
    sampler: Posterior,
    /// How the causal pads fill the rows and columns beyond a frame's edge (the checkpoint's mode).
    padding: Padding,
}

/// The per-channel affine of the latent space, in the encoder's direction: `(sample − mean) / std`.
///
/// Multiplying by `1 / std` and shifting by `−mean / std` is a one-by-one convolution whose weight is
/// the diagonal, so the encoder undoes the decoder's `value · std + mean` with one operator of the
/// same shape as the projection beside it (the audio decoder's `ChannelAffine` restore is that other direction).
///
/// **Declared deviation.** The reference's `(lat - mean) / std` is a subtraction and a division in
/// f32 - two roundings - while a convolution's multiply-add is one. The values agree to at most 1 ulp
/// of the f32 sample (6e-8 relative), against the 1e-4 the conditioned path is compared at; the
/// epilogue before it is bit-exact, so this is the only step that differs from torch.
pub fn normalize_tables(mean: &[f32], std: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let channels = mean.len();
    if channels == 0 || std.len() != channels {
        return Err(Error::config(format!(
            "the latent statistics hold {} means and {} deviations",
            mean.len(),
            std.len()
        )));
    }
    let mut scale = vec![0.0_f32; channels * channels];
    let mut shift = vec![0.0_f32; channels];
    for channel in 0..channels {
        let deviation = std[channel];
        if !deviation.is_finite() || deviation <= 0.0 {
            return Err(Error::config(format!(
                "latent channel {channel} has a deviation of {deviation}, which cannot be divided by"
            )));
        }
        scale[channel * channels + channel] = 1.0 / deviation;
        shift[channel] = -mean[channel] / deviation;
    }
    Ok((scale, shift))
}

/// Read one convolution of the checkpoint: `from` input channels, `to` output, `kernel` taps.
///
/// **`channels_last` is the operand's storage, not a preference.** The kernel derives every
/// operand's strides from the flag, so a convolution declared channels-last over a channels-first
/// buffer is wrong numbers rather than a refusal (`nn::Conv3d::forward` refuses it).
///
/// **Every convolution here is channel-first, and the reference says why that costs no number.**
/// Its stride-1 padded convolutions run under `torch.channels_last_3d` as a cuDNN speed path: "it
/// selects the same algorithm for either order, which leaves the encoding unchanged".
/// The order is therefore a throughput choice
/// rather than a fidelity one, and this encoder takes it only against a measured encode-convolution
/// time. The same comment is why its strided downsamplers stay channel-first even there: under
/// channels-last cuDNN picks another algorithm "whose rounding differs", so the reference keeps
/// those two out of the speed path itself.
fn convolution(
    scope: &mut Scope<'_, '_>,
    from: u32,
    to: u32,
    kernel: u32,
    stride: [u32; 3],
    channels_last: bool,
) -> Result<Conv3d> {
    // No zero padding of its own: a causal pad (`CausalPad`, alone or fused into the
    // normalization) supplies the padded input. `tf32`: the encoder's convolutions are FP32 and the
    // reference keeps cuDNN's TF32 for them.
    Ok(Conv3d::load(
        scope,
        ConvSpec {
            stride,
            tf32: true,
            channels_last,
            ..ConvSpec::new(from, to, [kernel; 3])
        },
    )?)
}

impl VideoEncoder {
    /// Read the encoder's weights. Every convolution is stored plain, so nothing is folded.
    pub fn load(config: &VideoVaeConfig, loader: &mut WeightLoader) -> Result<Self> {
        check_stages(config)?;
        let widths = &config.block_out_channels;
        let latent_channels = config.latent_channels;
        let groups = config.norm_num_groups;
        let epsilon = config.norm_eps;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::config(format!(
                "the video encoder's normalization epsilon is {epsilon}"
            )));
        }
        let padding = config.spatial_padding_mode;
        // One pixel on each spatial side and two zero frames in front, before every normalized
        // convolution.
        let pad = CausalPad {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
            front: 2,
            padding,
        };

        let input = convolution(
            &mut loader.scope("encoder.conv_in"),
            config.in_channels,
            widths[0],
            3,
            [1; 3],
            false,
        )?;
        let mut width = widths[0];
        let mut levels = Vec::with_capacity(widths.len());
        for (index, out) in widths.iter().enumerate() {
            let stage = format!("encoder.down_blocks.{index}");
            let mut blocks = Vec::with_capacity(config.layers_per_block as usize);
            for block in 0..config.layers_per_block {
                let base = format!("{stage}.resnets.{block}");
                let from = if block == 0 { width } else { *out };
                blocks.push(ResnetBlock3d::load(
                    &mut loader.scope(base),
                    ResnetSpec {
                        in_channels: from,
                        out_channels: *out,
                        groups,
                        eps: epsilon,
                        pad,
                        kernel: 3,
                        tf32: true,
                    },
                )?);
            }
            let stride = [
                config.temporal_downsample_factors[index],
                config.spatial_downsample_factors[index],
                config.spatial_downsample_factors[index],
            ];
            // A stage with both factors one has no convolution, so it has no padded frame either.
            let downsample = if stride.iter().product::<u32>() > 1 {
                Some(convolution(
                    &mut loader.scope(format!("{stage}.downsamplers.0.conv")),
                    *out,
                    *out,
                    3,
                    stride,
                    false,
                )?)
            } else {
                None
            };
            levels.push((blocks, downsample));
            width = *out;
        }
        let norm_out = {
            let mut norm = loader.scope("encoder.norm_out");
            let weight = norm.tensor("weight", &[u64::from(width)])?;
            let bias = norm.tensor("bias", &[u64::from(width)])?;
            GroupNorm::new(groups, epsilon, Some(weight), Some(bias), pad)?
        };
        let conv_out = convolution(
            &mut loader.scope("encoder.conv_out"),
            width,
            2 * latent_channels,
            3,
            [1; 3],
            false,
        )?;
        let moments = convolution(
            &mut loader.scope("quant_conv"),
            2 * latent_channels,
            2 * latent_channels,
            1,
            [1; 3],
            false,
        )?;
        // The latent space's affine: the checkpoint states its statistics, not a tensor for them.
        let latent = u64::from(latent_channels);
        let (scale, shift) = normalize_tables(&config.latents_mean, &config.latents_std)?;
        let mut constants = loader.scope("video_vae.normalize");
        let normalize = Conv3d::new(
            constants.constant_f32("weight", &[latent, latent, 1, 1, 1], &scale)?,
            Some(constants.constant_f32("bias", &[latent], &shift)?),
            ConvSpec {
                tf32: true,
                ..ConvSpec::new(latent_channels, latent_channels, [1; 3])
            },
        )?;
        Ok(Self {
            input_channels: config.in_channels,
            input,
            levels,
            norm_out,
            conv_out,
            moments,
            normalize,
            sampler: Posterior,
            padding,
        })
    }
}
/// A `[channels, frames, rows, columns]` raster as the one sample a convolution reads.
fn batched(view: TensorView) -> TensorView {
    let dims = view.shape.dims();
    view.reshape(Shape::new(&[1, dims[0], dims[1], dims[2], dims[3]]))
}

/// A `[1, channels, frames, rows, columns]` sample as its raster.
fn unbatched(view: TensorView) -> TensorView {
    let dims = view.shape.dims();
    view.reshape(Shape::new(&[dims[1], dims[2], dims[3], dims[4]]))
}

impl VideoEncoder {
    /// Encode one tile: `pixels` `[1, in_channels, frames, height, width]` FP32, normalized by
    /// [`crate::output::PIXEL_NORM`], to its moments `[2 · latent_channels, frames', height / 16,
    /// width / 16]`.
    ///
    /// The one-by-one projection that follows the moment convolution runs once on the stitched
    /// raster ([`Self::project`]), as the reference's `quant_conv` reads the stitched moments.
    pub fn encode_tile(&self, f: &Forward, pixels: TensorView) -> Result<TensorView> {
        if pixels.shape.rank() != 5 || pixels.shape.dim(0) != 1 {
            return Err(Error::config(format!(
                "the video encoder reads one [1, channels, frames, height, width] tile, not {:?}",
                pixels.shape.dims()
            )));
        }
        if pixels.shape.dim(1) != u64::from(self.input_channels) {
            return Err(Error::config(format!(
                "{} input channels for a stack of {}",
                pixels.shape.dim(1),
                self.input_channels
            )));
        }
        let frames = pixels.shape.dim(2);
        if frames == 0 || frames > u64::from(FRAMES.clip_frames) {
            return Err(Error::config(format!(
                "a video encoder reads one still or one clip of at most {} frames, not {frames}",
                FRAMES.clip_frames
            )));
        }

        // The input convolution reads padded pixels; it has no normalization of its own.
        let padded = CausalPad {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
            front: 2,
            padding: self.padding,
        }
        .forward(f, pixels)?;
        let mut stream = self.input.forward(f, padded)?;
        for (blocks, downsample) in &self.levels {
            for block in blocks {
                stream = block.forward(f, stream)?;
            }
            if let Some(downsample) = downsample {
                // The strided downsampler pads only after the source, one row and one column, and
                // two zero frames in front.
                let padded = CausalPad {
                    top: 0,
                    bottom: 1,
                    left: 0,
                    right: 1,
                    front: 2,
                    padding: self.padding,
                }
                .forward(f, stream)?;
                stream = downsample.forward(f, padded)?;
            }
        }
        let padded = self.norm_out.forward(f, stream)?;
        Ok(unbatched(self.conv_out.forward(f, padded)?))
    }

    /// The stitched moments `[2 · latent_channels, frames, rows, columns]` projected to the
    /// posterior's two fields by `quant_conv`: one call, after the stitch, as the reference's
    /// `Encoder` runs it on the stitched moments.
    pub fn project(&self, f: &Forward, moments: TensorView) -> Result<TensorView> {
        Ok(unbatched(self.moments.forward(f, batched(moments))?))
    }

    /// The posterior sample a condition reads: the projected `moments` and the posterior's
    /// `noise`, both `[.., frames, rows, columns]` channel-major FP32, to the normalized latent
    /// `[latent_channels, frames, rows, columns]`.
    ///
    /// The reference's `DiagonalGaussian` and the tail of `LatentEncoder`: mean and clamped
    /// log-variance, `mean + exp(0.5 · logvar) · noise` rounded through FP16 as a step
    /// (`ops::posterior`), then the latent space's per-channel affine ([`normalize_tables`]).
    pub fn posterior(
        &self,
        f: &Forward,
        moments: TensorView,
        noise: TensorView,
    ) -> Result<TensorView> {
        let drawn = self.sampler.sample(f, moments, noise)?;
        Ok(unbatched(self.normalize.forward(f, batched(drawn))?))
    }
}

/// The longest track the model supports, in latent frames.
///
/// A request is at most [`crate::config::MAX_FRAMES`] frames of video at [`crate::config::FPS`], and its
/// audio lasts as long, so the longest track is that duration at [`crate::config::AUDIO_LATENTS_PER_SECOND`]
/// latents, rounded up to the next latent. The encoder's attention is built for this bound and a longer
/// track is refused rather than read past it.
pub const LONGEST_TRACK: u32 = (crate::config::MAX_FRAMES
    * crate::config::AUDIO_LATENTS_PER_SECOND)
    .div_ceil(crate::config::FPS);

/// The DAC-style audio encoder `ref2va` conditions on: a waveform's posterior mean as latent frames.
///
/// `[1, 1, samples]` mono audio in, `[latent_channels, frames]` latents out, with `frames =
/// samples / 800` — the product of the checkpoint's encoder rates `(2, 4, 4, 5, 5)` and the hop the
/// decoder inverts. The network is the reference's DAC `Encoder` ([`DacEncoder`]) to `latent_dim`
/// features, then this family's attention projection (`pre_block`) that every frame of the track
/// attends over its predecessors, the mean convolution and the latent normalization.
///
/// **The projection's attention is causal** (`DenseInput(causal=True)` in the reference), which is what
/// makes a latent frame depend only on the audio before it. It runs in FP32 like everything else in this
/// encoder, on the deployment's longest track, which `load` takes as the attention's key bound.
pub struct AudioEncoder {
    config: crate::AudioVaeConfig,
    /// The most latent frames one track may have: the attention's key bound.
    frames: u32,
    /// The DAC network from the waveform to `latent_dim` features.
    encoder: DacEncoder,
    projection: Projection,
    mean: Conv1d,
    /// The latent space's affine, `(latent - mean) / std` per channel. The reference's `LatentEncoder`
    /// applies it to the posterior mode, and the DiT's audio input reads normalized rows: the
    /// statistics are the same ones the audio decoder restores.
    normalize: Conv1d,
    transpose: Transpose,
}

/// The attention projection that narrows latent features to the latent channels.
struct Projection {
    shortcut_norm: LayerNorm,
    shortcut: Linear,
    attention_norm: LayerNorm,
    /// Three matrices over the checkpoint's fused query-key-value weight, so each head group is one
    /// contiguous block the attention can read.
    query: Linear,
    key: Linear,
    value: Linear,
    attention: DenseAttention,
    attention_output: Linear,
    /// The head average and the head width pooled down to the latent channels, in one pass: the
    /// heads are the pool's rows.
    pool: AdaptiveAvgPool2d,
    /// The latent channels the pool writes.
    channels: u32,
    mlp_norm: LayerNorm,
    /// The reference's `GeGLU`: its own norm, then `output(gelu_tanh(gate(x)) · value(x))` with the
    /// gated product in one rounding.
    geglu_norm: LayerNorm,
    geglu: GatedMlp,
}

impl AudioEncoder {
    /// The latent frames `samples` samples encode to: `ceil(samples / hop)`.
    ///
    /// This is the reference's rule and the reason it is a ceiling is the pad its
    /// The batch the track must carry: **the stereo pair is the batch**, and a mono track is refused.
    ///
    /// The reference expands a mono track to two identical channels before the encoder
    /// (`unpack_samples`: `waveform.expand(2, -1)`, refusing a channel count other than one or two),
    /// and the encoder's own output is `2 * latent_frames` rows. A caller that hands a mono track
    /// unexpanded gets **half a track of latent frames** with every count downstream still
    /// consistent - `[1, 1, samples]` is a legal shape on its own - so the check is that the batch is
    /// exactly the stereo pair, and it names the expansion the reference performs.
    fn check_track(shape: Shape) -> Result<u64> {
        if shape.rank() != 3 || shape.dim(1) != 1 {
            return Err(Error::config(format!(
                "the audio encoder reads a [batch, 1, samples] track, not {:?}",
                shape.dims()
            )));
        }
        let batch = shape.dim(0);
        if batch != u64::from(crate::config::AUDIO_CHANNELS) {
            return Err(Error::config(format!(
                "the audio encoder reads the stereo pair ({} batch rows) and was given {batch}: a \
                 mono track must be expanded to two identical channels before it, as the \
                 reference's `unpack_samples` does - read as one row it encodes half a track of \
                 latent frames, which every count downstream accepts",
                crate::config::AUDIO_CHANNELS
            )));
        }
        Ok(batch)
    }

    /// own `unpack_samples` applies first: a track that is not a whole number of latent frames is
    /// **right-padded with zeros** to `ceil(samples / hop) · hop`, so every stage's strided convolution
    /// divides exactly and the last latent frame reads those zeros. A track of any positive length
    /// therefore encodes, and [`Self::encode`] states what it needs to do the same.
    pub fn frames(&self, samples: u64) -> Result<u64> {
        if samples == 0 {
            return Err(Error::config(
                "an audio track of no samples encodes to no latent frames".to_owned(),
            ));
        }
        let frames = samples.div_ceil(self.hop());
        if frames > u64::from(LONGEST_TRACK) {
            return Err(Error::config(format!(
                "{frames} latent frames exceed the {LONGEST_TRACK}-frame track the encoder's \
                 attention is built for"
            )));
        }
        Ok(frames)
    }

    /// The samples one latent frame covers: the product of the encoder's rates.
    pub fn hop(&self) -> u64 {
        self.encoder.hop()
    }

    /// The encoder the checkpoint's `encoder`, `pre_block` and `mean_proj` modules state, read in
    /// FP32.
    ///
    /// The encoder is DAC's network - `encoder.block` is the input convolution, one entry per stage
    /// (three dilated residual units, the stage's Snake and the strided widening convolution), a
    /// Snake and the output convolution - followed by the latent projection (`pre_block`) and the
    /// mean convolution (`mean_proj`). Every convolution of the network is weight-normalized and
    /// folded once here. The attention is built for [`LONGEST_TRACK`], the longest track the model
    /// supports; a longer track is refused by [`Self::frames`] rather than read past the bound.
    pub fn load(config: &crate::AudioVaeConfig, loader: &mut WeightLoader) -> Result<Self> {
        let frames = LONGEST_TRACK;
        // The audio encoder computes in FP32 throughout, so every weight is read in it.
        let mut root = loader.scope("");
        let mut root = root.pinned(DType::F32);
        let encoder = DacEncoder::load(
            &mut root.scope("encoder.block"),
            &DacSpec {
                channels: config.encoder_dim,
                strides: config.encoder_rates.clone(),
                latent_channels: config.latent_dim,
            },
        )?;
        let projection = Projection::load(config, &mut root.scope("pre_block"), frames)?;
        let channels = u64::from(config.latent_channels);
        // The reference's mean projection is `nn.Conv1d(latent_channels, latent_channels, 1)`, so the
        // checkpoint stores a three-dimensional filter `[out, in, kernel]` and not a matrix: a kernel
        // of one makes the two the same arithmetic but not the same rank, and `Conv1d::new` refuses a
        // filter that is not 3-d.
        let mean = Conv1d::load(
            &mut root.scope("mean_proj"),
            ConvSpec::new(config.latent_channels, config.latent_channels, [1]),
        )?;
        // The latent space's affine: the config states its statistics, not a tensor for them.
        let (scale, shift) = normalize_tables(&config.latents_mean, &config.latents_std)?;
        let mut constants = loader.scope("audio_vae.normalize");
        let normalize = Conv1d::new(
            constants.constant_f32("weight", &[channels, channels, 1], &scale)?,
            Some(constants.constant_f32("bias", &[channels], &shift)?),
            ConvSpec::new(config.latent_channels, config.latent_channels, [1]),
        )?;
        Ok(Self {
            config: config.clone(),
            frames,
            encoder,
            projection,
            mean,
            normalize,
            transpose: Transpose::new(DType::F32)?,
        })
    }

    /// Encode a `[batch, 1, length]` track into fresh `[batch · frames, latent_channels]` F32 rows
    /// of the normalized latent, `(mode - mean) / std`.
    ///
    /// The encoder is mono per batch row and its latent is a two-row batch: a stereo reference
    /// arrives as `[2, 1, samples]`. The rows are channel-major, batch outermost, which is the form
    /// the DiT's audio conditioning reads (`[2 · frames, audio_channels]`). The length must be a
    /// whole number of hops, right-padded with zeros by the caller as the reference's
    /// `unpack_samples` does ([`padded_samples`]); the projection attends over the frame axis, so
    /// it runs once per batch row and never mixes them.
    pub fn encode(&self, f: &Forward, samples: TensorView) -> Result<TensorView> {
        let batch = Self::check_track(samples.shape)?;
        let length = samples.shape.dim(2);
        let frames = self.frames(length)?;
        let hop = self.hop();
        if !length.is_multiple_of(hop) {
            let padded = padded_samples(length, hop)?;
            return Err(Error::config(format!(
                "{length} samples is not a whole number of {hop}-sample latent frames: the track \
                 is right-padded with zeros to {padded} samples before it is encoded, as the \
                 reference's `unpack_samples` does"
            )));
        }
        if frames > u64::from(self.frames) {
            return Err(Error::config(format!(
                "a {frames}-frame track is longer than the {} the encoder is built for",
                self.frames
            )));
        }
        let channels = u64::from(self.config.latent_channels);
        let wide = u64::from(self.config.latent_dim);
        let features = self.encoder.forward(f, samples)?;
        let rows = f.empty(DType::F32, Shape::new(&[batch * frames, channels]))?;
        for row in 0..batch {
            // One row's `[wide, frames]` features, frame-major for the attention, then its mean
            // head's `[channels, frames]` transposed into its block of channel-major rows.
            let features = features
                .narrow(0, row, 1)?
                .reshape(Shape::new(&[wide, frames]));
            let frame_major = self.transpose.forward(f, features)?;
            let pooled = self.projection.forward(f, frame_major)?;
            let pooled = self
                .transpose
                .forward(f, pooled)?
                .reshape(Shape::new(&[1, channels, frames]));
            let mean = self.mean.forward(f, pooled)?;
            let normalized = self
                .normalize
                .forward(f, mean)?
                .reshape(Shape::new(&[channels, frames]));
            let block = self.transpose.forward(f, normalized)?;
            nn::layers::Axpby.forward(f, rows.narrow(0, row * frames, frames)?, block, 1.0, 0.0)?;
        }
        Ok(rows)
    }
}

impl Projection {
    /// The latent projection module at `scope` (`pre_block`), at the latent width.
    fn load(
        config: &crate::AudioVaeConfig,
        scope: &mut Scope<'_, '_>,
        frames: u32,
    ) -> Result<Self> {
        let wide = u64::from(config.latent_dim);
        let channels = u64::from(config.latent_channels);
        let heads = u64::from(config.num_attention_heads);
        if heads == 0 || !wide.is_multiple_of(heads) {
            return Err(Error::config(format!(
                "{wide} latent channels over {heads} attention heads"
            )));
        }
        let shortcut_norm =
            LayerNorm::load(&mut scope.scope("norm3"), config.latent_dim, NORM_EPS)?;
        let attention_norm =
            LayerNorm::load(&mut scope.scope("norm1"), config.latent_dim, NORM_EPS)?;
        let mut shortcut = scope.scope("proj");
        let shortcut = Linear::new(
            shortcut.tensor("weight", &[channels, wide])?,
            Some(shortcut.tensor("bias", &[channels])?),
            WeightOrder::NxK,
        )?;
        // The checkpoint fuses query, key and value into one `[3 · wide, wide]` matrix whose thirds
        // are contiguous rows, and keeps the three bias windows as three tensors: each head group is
        // its own layer over its rows. `zero_k_bias` is the checkpoint's stored key window - a
        // vector it carries like the other two - so it is loaded, never substituted by a constant
        // on the strength of its name: a buffer placed in a learned slot changes the arithmetic.
        let third = wide;
        let mut attention = scope.scope("attn");
        let mut linear = |index: u64, bias: &str| -> Result<Linear> {
            let weight = attention
                .scope("qkv")
                .rows("weight", index * third, &[third, wide])?;
            let bias = attention.tensor(bias, &[third])?;
            Ok(Linear::new(weight, Some(bias), WeightOrder::NxK)?)
        };
        let query = linear(0, "q_bias")?;
        let key = linear(1, "zero_k_bias")?;
        let value = linear(2, "v_bias")?;
        let head_dim = u32::try_from(wide / heads)
            .map_err(|_| Error::config("a head wider than a launch holds".to_owned()))?;
        let heads_u32 = u32::try_from(heads)
            .map_err(|_| Error::config("more heads than a launch holds".to_owned()))?;
        let attention = DenseAttention::new(DenseSpec {
            head_dim,
            heads: heads_u32,
            kv_heads: heads_u32,
            max_keys: frames,
            causal: true,
            window: None,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })?;
        let mut output = scope.scope("attn.proj");
        let attention_output = Linear::new(
            output.tensor("weight", &[channels, channels])?,
            Some(output.tensor("bias", &[channels])?),
            WeightOrder::NxK,
        )?;
        let pool = AdaptiveAvgPool2d::new([1, config.latent_channels])?;
        let mlp_norm =
            LayerNorm::load(&mut scope.scope("norm2"), config.latent_channels, NORM_EPS)?;
        let mut geglu = scope.scope("mlp");
        let geglu_norm =
            LayerNorm::load(&mut geglu.scope("norm"), config.latent_channels, NORM_EPS)?;
        let geglu = GatedMlp::load(
            &mut geglu,
            GatedMlpSpec {
                hidden: config.latent_channels,
                intermediate: 2 * config.latent_channels,
                activation: ops::activation::Activation::GeluTanh,
                rounding: Rounding::Fused,
                gate_up: GateUp::Split {
                    gate: "w0",
                    up: "w1",
                },
                down: "w2",
                bias: Bias::Present,
            },
            Split::WHOLE,
        )?;
        Ok(Self {
            shortcut_norm,
            shortcut,
            attention_norm,
            query,
            key,
            value,
            attention,
            attention_output,
            pool,
            channels: config.latent_channels,
            mlp_norm,
            geglu_norm,
            geglu,
        })
    }

    /// Project `values` `[frames, latent_dim]` to a fresh `[frames, latent_channels]`.
    fn forward(&self, f: &Forward, values: TensorView) -> Result<TensorView> {
        let frames = values.shape.dim(0);
        let head_dim = u64::from(self.attention.spec().head_dim);
        let heads = values.shape.dim(1) / head_dim;
        let shortcut = self
            .shortcut
            .forward(f, self.shortcut_norm.forward(f, values, None)?)?;
        let normed = self.attention_norm.forward(f, values, None)?;
        // Three contiguous head-group blocks, then causal attention over the frames.
        let head_shape = Shape::new(&[frames, heads, head_dim]);
        let [query, key, value] =
            [&self.query, &self.key, &self.value].map(|linear| linear.forward(f, normed));
        let attended = self.attention.forward(
            f,
            query?.reshape(head_shape),
            key?.reshape(head_shape),
            value?.reshape(head_shape),
            None,
        )?;
        // The head average and the head width pooled to the latent channels in one pass.
        let pooled = self
            .pool
            .forward(f, attended)?
            .reshape(Shape::new(&[frames, u64::from(self.channels)]));
        let projected = self.attention_output.forward(f, pooled)?;
        let stream = Residual::default().forward(f, shortcut, projected, None)?;
        // The normalized GeGLU residual.
        let normed = self.mlp_norm.forward(f, stream, None)?;
        let updated = self
            .geglu
            .forward(f, self.geglu_norm.forward(f, normed, None)?)?;
        Ok(Residual::default().forward(f, stream, updated, None)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One pixel's normalized channels: `(pixel / 255 − mean) / std`.
    ///
    /// The inverse of the decoder's byte step (`value * std + mean`, clamped to [0, 1], scaled to 255
    /// and rounded half to even), and the input the video encoder reads. A kernel that normalizes the
    /// raster computes this; the round trip through the decoder's quantization recovers the byte.
    fn normalize(pixel: [u8; 3]) -> [f32; 3] {
        let mut values = [0.0f32; 3];
        for (channel, byte) in pixel.iter().enumerate() {
            values[channel] = (f32::from(*byte) / 255.0 - crate::output::PIXEL_NORM.mean[channel])
                / crate::output::PIXEL_NORM.std[channel];
        }
        values
    }

    use crate::video_vae::tests::video_config;

    #[test]
    fn a_pixel_normalizes_and_quantizes_back_to_itself() {
        // The decoder's byte step, per channel: restore, clamp to [0, 1], scale and round half to
        // even.
        let quantize = |normalized: [f32; 3]| -> [u8; 3] {
            let mut bytes = [0u8; 3];
            for (channel, value) in normalized.iter().enumerate() {
                let restored = value * crate::output::PIXEL_NORM.std[channel]
                    + crate::output::PIXEL_NORM.mean[channel];
                bytes[channel] = (restored.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8;
            }
            bytes
        };
        for pixel in [
            [0u8, 0, 0],
            [255, 255, 255],
            [124, 116, 104],
            [1, 2, 3],
            [200, 10, 77],
        ] {
            let restored = quantize(normalize(pixel));
            for (channel, (got, want)) in restored.iter().zip(pixel.iter()).enumerate() {
                assert!(
                    got.abs_diff(*want) <= 1,
                    "channel {channel}: {got} for {want} from {pixel:?}"
                );
            }
        }
        // The extremes land outside the band the decoder works in - the byte step maps a value
        // back through `value · std + mean` and clamps to the byte range, so one inside [-1, 1]
        // loses nothing. Being outside is the property; how far outside is the checkpoint's
        // statistics' business, not this test's.
        for channel in 0..3 {
            let low = normalize([0, 0, 0])[channel];
            let high = normalize([255, 255, 255])[channel];
            assert!(low < -1.0, "channel {channel}: {low} from zero");
            assert!(high > 1.0, "channel {channel}: {high} from 255");
        }
    }

    #[test]
    fn the_encoders_affine_inverts_the_latent_statistics() {
        let config = video_config();
        let channels = config.latents_mean.len();
        let (scale, shift) =
            normalize_tables(&config.latents_mean, &config.latents_std).expect("the affine");
        assert_eq!(scale.len(), channels * channels);
        assert_eq!(shift.len(), channels);
        for channel in 0..channels {
            assert_eq!(
                scale[channel * channels + channel],
                1.0 / config.latents_std[channel]
            );
            assert_eq!(
                shift[channel],
                -config.latents_mean[channel] / config.latents_std[channel]
            );
        }
        // Off the diagonal there is nothing: each channel scales on its own.
        let off = scale
            .iter()
            .enumerate()
            .filter(|(index, value)| index % (channels + 1) != 0 && **value != 0.0)
            .count();
        assert_eq!(off, 0);

        // A deviation of zero cannot be divided by, and a shorter table names the wrong channel count.
        let mut zeroed = config.latents_std.clone();
        zeroed[3] = 0.0;
        assert!(normalize_tables(&config.latents_mean, &zeroed).is_err());
        assert!(normalize_tables(&config.latents_mean, &config.latents_std[..2]).is_err());
    }

    #[test]
    fn a_track_is_padded_to_whole_latent_frames() {
        // A partial frame rounds up, and the length the network is given is the ceiling's own multiple:
        // the reference right-pads the track with zeros, so nothing is dropped.
        assert!(padded_samples(0, 800).is_err());
        assert_eq!(padded_samples(801, 800).expect("padded"), 1600);
        assert_eq!(padded_samples(1600, 800).expect("whole"), 1600);
    }

    #[test]
    fn a_stack_with_no_stage_is_refused() {
        let mut config = video_config();
        config.block_out_channels.clear();
        assert!(check_stages(&config).is_err());
        let mut config = video_config();
        config.spatial_downsample_factors.pop();
        assert!(check_stages(&config).is_err());
        let mut config = video_config();
        config.temporal_downsample_factors[1] = 0;
        assert!(check_stages(&config).is_err());
    }

    #[test]
    fn a_mono_track_is_refused_where_a_stereo_pair_passes() {
        // The stereo pair is the batch. A mono track is legal as a shape and halves the latent
        // frames, so nothing downstream notices - the reference expands it before the encoder and
        // this refuses it here, naming that expansion.
        assert_eq!(
            AudioEncoder::check_track(Shape::new(&[2, 1, 44_100])).unwrap(),
            2
        );
        let refused = AudioEncoder::check_track(Shape::new(&[1, 1, 44_100])).unwrap_err();
        let message = refused.to_string();
        assert!(message.contains("stereo pair"), "{message}");
        assert!(message.contains("unpack_samples"), "{message}");
        // And three batch rows are not "more stereo": the reference refuses a channel count that is
        // neither one nor two, and so does this.
        assert!(AudioEncoder::check_track(Shape::new(&[3, 1, 44_100])).is_err());
        // The two shapes the encoder never took are still refused, and by the other message.
        assert!(AudioEncoder::check_track(Shape::new(&[2, 3, 44_100])).is_err());
        assert!(AudioEncoder::check_track(Shape::new(&[2, 44_100])).is_err());
    }
}
