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
//! `(pixel / 255 − mean) / std` with the ImageNet constants before the input convolution. A clip
//! shorter than the checkpoint's [`crate::config::CLIP_FRAMES`] repeats its last frame; a single
//! frame stays alone.
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
use crate::config::VideoVaeConfig;
use crate::{Error, Result};

/// The spatial compression of the video encoder: input pixels per latent position.
pub const SPATIAL_RATIO: u32 = 16;

/// The channels of a condition's raster: RGB.
pub const PIXEL_CHANNELS: u32 = 3;

/// How H3's encoders read a request's decoded media, with the vision tower at `dtype`: the video
/// encoder's planar pixels normalized by the ImageNet constants ([`crate::output::PIXEL_NORM`]), the
/// vision tower's patch rows by the processor's statistics, and the audio encoder's channel rows
/// padded to whole latent hops.
pub fn media_format(config: &crate::Config, dtype: DType) -> Result<nn::capability::MediaFormat> {
    Ok(nn::capability::MediaFormat {
        pixels: crate::output::PIXEL_NORM,
        patches: qwen3::processor::patch_format(&config.text_encoder.vision, config.pixels, dtype),
        hop: u64::from(AudioStack::of(&config.audio_vae)?.samples_per_latent),
    })
}

/// The extents of one tensor of the encoder's stack, in frames, rows and columns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Extent {
    /// Frames.
    pub frames: u32,
    /// Rows.
    pub height: u32,
    /// Columns.
    pub width: u32,
}

impl Extent {
    /// The extent one stage of the encoder's stack produces: `frame_stride` in time, and a halving in
    /// each spatial axis when `spatial_stride` is two.
    pub fn stage(self, frame_stride: u32, spatial_stride: u32) -> Self {
        let frames = if frame_stride <= 1 {
            self.frames
        } else {
            self.frames.div_ceil(frame_stride)
        };
        let halve = |extent: u32| {
            if spatial_stride <= 1 {
                extent
            } else {
                // Two pixels of padding in front (bottom and right), a three-pixel kernel and a
                // stride of two: `floor((extent + 1 - 3) / 2) + 1`.
                extent.saturating_sub(2) / 2 + 1
            }
        };
        Self {
            frames,
            height: halve(self.height),
            width: halve(self.width),
        }
    }
}

/// The encoder's stage table, read from the checkpoint's config once at the loading boundary.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stack {
    /// The raster channels the input convolution reads.
    pub input_channels: u32,
    /// Each stage's width, in order.
    pub widths: Vec<u32>,
    /// Each stage's temporal stride.
    pub temporal_strides: Vec<u32>,
    /// Each stage's spatial stride.
    pub spatial_strides: Vec<u32>,
    /// Blocks per stage.
    pub blocks: u32,
    /// Latent channels.
    pub latent_channels: u32,
    /// The clip the network reads: its frames and what one clip yields.
    pub clip: Clip,
}

/// The clip a causal encoder cuts a raster into, and what its frames are worth.
///
/// Three numbers of one geometry, stated together because every use needs all three: the frames the
/// network reads at once, the latent frames one complete clip yields, and the latent frames a
/// condition drops from its last clip because a later clip of a generated video would overlap them.
/// A condition's own latent count follows from them ([`Clip::latent_frames`]), and packing takes this
/// value rather than three loose numbers so that the count has one statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clip {
    /// Frames one clip of the encoder's raster is.
    pub frames: u32,
    /// Latent frames one complete clip yields.
    pub latents: u32,
    /// Latent frames a condition drops from its last clip.
    pub token_drop: u32,
}

impl Clip {
    /// The clip a checkpoint's video VAE config states.
    pub fn of(config: &VideoVaeConfig) -> Result<Self> {
        let clip = Self {
            frames: config.clip_length,
            latents: config.clip_latents(),
            token_drop: config.token_drop,
        };
        if clip.frames == 0 || clip.latents <= clip.token_drop {
            return Err(Error::config(format!(
                "a clip of {} frames yields {} latent frames and drops {}; the drop must leave some",
                clip.frames, clip.latents, clip.token_drop
            )));
        }
        Ok(clip)
    }

    /// The latent frames a **condition** of `frames` frames encodes to.
    ///
    /// A condition is cut into clips of [`Self::frames`] frames, each yielding [`Self::latents`] latent
    /// frames, and the last clip's trailing [`Self::token_drop`] latent frames are dropped - so 17
    /// frames are 2 and 22 are 7. A single frame is a still and yields one: the network's causal front
    /// padding makes one frame one latent frame, and the reference encodes
    /// an image or a keyframe through the same network alone.
    ///
    /// This is a **different statement** from a generated timeline's (`packing::video_latent_frames`:
    /// `17n + 5` frames to `5n + 2` latents). The two agree on the counts a generation uses, and only
    /// this one accepts the counts a condition may be - a condition is cut and padded, not generated.
    pub fn latent_frames(&self, frames: u32) -> Result<u32> {
        if frames == 0 {
            return Err(Error::config(
                "a video condition of no frame encodes nothing".to_owned(),
            ));
        }
        if frames == 1 {
            return Ok(1);
        }
        Ok(frames.div_ceil(self.frames) * self.latents - self.token_drop)
    }

    /// The clips a condition of `frames` frames is cut into, in order.
    ///
    /// Each clip is encoded alone, as the reference's `clips` and `latent_slices` cut it:
    /// the frames it reads and the latent frames it writes, the last clip
    /// holding what is left of both. A still is one clip of one frame.
    pub fn spans(&self, frames: u32) -> Result<Vec<ClipSpan>> {
        let total = self.latent_frames(frames)?;
        let mut spans = Vec::with_capacity(frames.div_ceil(self.frames) as usize);
        let (mut first, mut latent) = (0, 0);
        while first < frames {
            let length = self.frames.min(frames - first);
            let latents = self.latents.min(total - latent);
            spans.push(ClipSpan {
                first,
                frames: length,
                latent,
                latents,
            });
            first += length;
            latent += latents;
        }
        Ok(spans)
    }
}

impl Stack {
    /// Whether stage `index` ends in a downsampler, which is whether it halves space or strides time.
    ///
    /// The load builds the convolution from this answer and the call pads its input from the same
    /// one: a stage with both factors one has no convolution, so it has no padded frame either.
    pub fn downsamples(&self, index: usize) -> bool {
        self.temporal_strides[index] * self.spatial_strides[index] > 1
    }

    /// The stack the config states.
    pub fn of(config: &VideoVaeConfig) -> Result<Self> {
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
        if compression != SPATIAL_RATIO {
            return Err(Error::config(format!(
                "the video encoder compresses space {compression}-fold, not {SPATIAL_RATIO}"
            )));
        }
        Ok(Self {
            input_channels: config.in_channels,
            widths: config.block_out_channels.clone(),
            temporal_strides: config.temporal_downsample_factors.clone(),
            spatial_strides: config.spatial_downsample_factors.clone(),
            blocks: config.layers_per_block,
            latent_channels: config.latent_channels,
            clip: Clip::of(config)?,
        })
    }

    /// The extents the stack visits for one clip: the raster, one entry per stage, and the latent
    /// raster the output convolution produces.
    pub fn extents(&self, raster: Extent) -> Vec<Extent> {
        let mut extents = vec![raster];
        let mut current = raster;
        for (temporal, spatial) in self
            .temporal_strides
            .iter()
            .zip(self.spatial_strides.iter())
        {
            current = current.stage(*temporal, *spatial);
            extents.push(current);
        }
        extents
    }

    /// The extent of the posterior moments: `2 * latent_channels` channels over the latent raster.
    pub fn moments(&self, raster: Extent) -> Result<Extent> {
        self.extents(raster)
            .last()
            .copied()
            .ok_or_else(|| Error::config("the video encoder's stack has no level".to_owned()))
    }
}

/// The audio encoder's stack: DAC's strided widening network and the latent projection.
///
/// The decoder's rates are the encoder's reversed only in effect: the encoder's five strided stages
/// multiply to the 800 samples of one latent frame, so a whole number of latent frames in gives whole
/// latent frames out. `encoder_rates` are [2, 4, 4, 5, 5] and each stage doubles the width, starting
/// from `encoder_dim` after the input convolution.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AudioStack {
    /// Width after the input convolution.
    pub encoder_dim: u32,
    /// Each strided stage's rate, in order.
    pub rates: Vec<u32>,
    /// The width of each stage's output: `encoder_dim * 2^(stage + 1)`.
    pub widths: Vec<u32>,
    /// The encoder's features before the latent projection.
    pub latent_dim: u32,
    /// Latent channels.
    pub latent_channels: u32,
    /// Attention heads of the latent projection.
    pub heads: u32,
    /// Samples per latent frame: the product of the rates.
    pub samples_per_latent: u32,
}

impl AudioStack {
    /// The stack the config states.
    pub fn of(config: &crate::AudioVaeConfig) -> Result<Self> {
        if config.encoder_rates.is_empty() || config.encoder_dim == 0 {
            return Err(Error::config(
                "the audio encoder has no stage or no input width".to_owned(),
            ));
        }
        if config.latent_channels == 0 || config.num_attention_heads == 0 {
            return Err(Error::config(
                "the audio encoder has no latent channel or no head".to_owned(),
            ));
        }
        if !config.latent_dim.is_multiple_of(config.num_attention_heads) {
            return Err(Error::config(format!(
                "{} heads do not divide the audio encoder's width {}",
                config.num_attention_heads, config.latent_dim
            )));
        }
        let samples_per_latent = config
            .encoder_rates
            .iter()
            .try_fold(1u32, |product, rate| product.checked_mul(*rate))
            .ok_or_else(|| Error::config("the audio encoder's rates overflow".to_owned()))?;
        let mut width = config.encoder_dim;
        let widths = config
            .encoder_rates
            .iter()
            .map(|_| {
                width *= 2;
                width
            })
            .collect();
        Ok(Self {
            encoder_dim: config.encoder_dim,
            rates: config.encoder_rates.clone(),
            widths,
            latent_dim: config.latent_dim,
            latent_channels: config.latent_channels,
            heads: config.num_attention_heads,
            samples_per_latent,
        })
    }

    /// The length after each stage, from the samples to the latent frames.
    ///
    /// A strided stage is a convolution of `2 * rate` taps with `ceil(rate / 2)` samples of padding
    /// each side, so it maps `L` to `(L + 2 * ceil(rate / 2) - 2 * rate) / rate + 1`: 800 samples
    /// become 400, 100, 25, 5 and 1.
    pub fn lengths(&self, samples: u32) -> Vec<u32> {
        let mut lengths = vec![samples];
        let mut current = samples;
        for rate in &self.rates {
            let pad = rate.div_ceil(2);
            let padded = current + 2 * pad;
            current = if padded < 2 * rate {
                0
            } else {
                (padded - 2 * rate) / rate + 1
            };
            lengths.push(current);
        }
        lengths
    }

    /// The latent frames a track of `samples` encodes to, and the length the stack must be given.
    pub fn latent_frames(&self, samples: u32) -> Result<u32> {
        let frames = audio_latent_frames(u64::from(samples), u64::from(self.samples_per_latent))?;
        u32::try_from(frames)
            .map_err(|_| Error::config(format!("{samples} samples exceed a latent frame count")))
    }
}

/// The latent frames a track of `samples` encodes to: `ceil(samples / hop)`.
///
/// One statement of the rule, because two callers need it: the stack's geometry and the encoder's
/// own output extent. **This is the encode's direction:** a requested duration is covered, so the
/// track is `+0..hop-1` samples longer than the request and never short.
/// [`crate::packing::audio_latent_frames`] is the other direction - a request's video frame count
/// to its latents, rounded to nearest - and the two are deliberately named alike and documented
/// against each other, because a reader who takes one for the other gets a figure that looks right.
/// The ceiling is the reference's, and it is a ceiling because its `unpack_samples` right-pads the
/// track with zeros to [`padded_samples`] first, so every strided convolution divides exactly and
/// the last latent frame reads those zeros.
pub fn audio_latent_frames(samples: u64, hop: u64) -> Result<u64> {
    if samples == 0 || hop == 0 {
        return Err(Error::config(
            "an audio track of no samples encodes to no latent frames".to_owned(),
        ));
    }
    Ok(samples.div_ceil(hop))
}

/// The length the encoder must be given for a track of `samples`: the ceiling's own multiple.
///
/// The padding is [`audio_latent_frames`]' multiple less the track, so a whole-frame track is unchanged
/// and any other is right-padded with zeros exactly as the reference's `unpack_samples` does.
pub fn padded_samples(samples: u64, hop: u64) -> Result<u64> {
    Ok(audio_latent_frames(samples, hop)? * hop)
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

/// The encoder's output convolution and the normalization that feeds it.
///
/// The reference's `CausalConv3d.forward(norm=…)` normalizes each input frame, activates it, pads the
/// result with a reflection of the edge rows and columns and two zero frames in front, and convolves;
/// no output frame therefore reads a later input frame. The bias belongs to the convolution here,
/// where the reference adds it in a second pass to keep its rounding equal to `conv(x) + bias` — the
/// same value, one rounding apart. The residual blocks run the same stage twice
/// ([`ResnetBlock3d`]).
struct Normalized {
    norm: GroupNorm,
    conv: Conv3d,
    /// The source width the normalization reduces.
    channels: u32,
    /// The frame groups its statistics are taken over.
    groups: u32,
}

/// One encoder level: its residual blocks, then the convolution that halves space and strides
/// time.
struct Level {
    blocks: Vec<ResnetBlock3d>,
    downsample: Option<Conv3d>,
}

/// The video encoder: the causal convolutional network that turns a clip's pixels into the posterior
/// moments of its latent.
///
/// One call encodes one raster: the caller tiles a clip as `Grid` states, and a band of latent rows
/// may be encoded on its own through `Grid::band`. Every convolution runs FP32 with TF32 allowed, the
/// precision the checkpoint's preset declares for the video VAE's encoder.
pub struct VideoEncoder {
    stack: Stack,
    /// The input convolution, over raw pixels.
    input: Conv3d,
    levels: Vec<Level>,
    /// The final normalization and the moment convolution, `2 · latent_channels` wide.
    output: Normalized,
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
        let stack = Stack::of(config)?;
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
            left: 1,
            front: 2,
            padding,
        };

        let input = convolution(
            &mut loader.scope("encoder.conv_in"),
            stack.input_channels,
            stack.widths[0],
            3,
            [1; 3],
            false,
        )?;
        let mut width = stack.widths[0];
        let mut levels = Vec::with_capacity(stack.widths.len());
        for (index, out) in stack.widths.iter().enumerate() {
            let stage = format!("encoder.down_blocks.{index}");
            let mut blocks = Vec::with_capacity(stack.blocks as usize);
            for block in 0..stack.blocks {
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
                stack.temporal_strides[index],
                stack.spatial_strides[index],
                stack.spatial_strides[index],
            ];
            let downsample = if stack.downsamples(index) {
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
            levels.push(Level { blocks, downsample });
            width = *out;
        }
        let output = {
            let mut norm = loader.scope("encoder.norm_out");
            let weight = norm.tensor("weight", &[u64::from(width)])?;
            let bias = norm.tensor("bias", &[u64::from(width)])?;
            Normalized {
                norm: GroupNorm::new(groups, epsilon, Some(weight), Some(bias), pad)?,
                conv: convolution(
                    &mut loader.scope("encoder.conv_out"),
                    width,
                    2 * stack.latent_channels,
                    3,
                    [1; 3],
                    false,
                )?,
                channels: width,
                groups,
            }
        };
        let moments = convolution(
            &mut loader.scope("quant_conv"),
            2 * stack.latent_channels,
            2 * stack.latent_channels,
            1,
            [1; 3],
            false,
        )?;
        // The latent space's affine: the checkpoint states its statistics, not a tensor for them.
        let latent = u64::from(stack.latent_channels);
        let (scale, shift) = normalize_tables(&config.latents_mean, &config.latents_std)?;
        let mut constants = loader.scope("video_vae.normalize");
        let normalize = Conv3d::new(
            constants.constant_f32("weight", &[latent, latent, 1, 1, 1], &scale)?,
            Some(constants.constant_f32("bias", &[latent], &shift)?),
            ConvSpec {
                tf32: true,
                ..ConvSpec::new(stack.latent_channels, stack.latent_channels, [1; 3])
            },
        )?;
        Ok(Self {
            stack,
            input,
            levels,
            output,
            moments,
            normalize,
            sampler: Posterior,
            padding,
        })
    }
}
impl Normalized {
    /// Normalize each frame, activate it, pad the result and convolve it.
    ///
    /// The padding is one pixel on each spatial side, mirrored or replicated, and two zero frames in
    /// front, so the convolution's own geometry carries none and no output frame reads a later input
    /// frame.
    /// `pad` is `[1, channels, frames + 2, height + 2, width + 2]` and `mean`/`rstd` are
    /// `[frames, groups]`, reserved once per tile extent with the encoder's other buffers.
    fn run(
        &self,
        f: &Forward,
        values: TensorView,
        pad: TensorView,
        mean: TensorView,
        rstd: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let (_, channels, frames, height, width) = extent(values)?;
        if channels != u64::from(self.channels)
            || pad.shape.rank() != 5
            || pad.shape.dims()
                != [
                    1,
                    u64::from(self.channels),
                    frames + 2,
                    height + 2,
                    width + 2,
                ]
        {
            return Err(Error::config(format!(
                "a normalization of {} channels got {:?} to carry its padded input",
                self.channels,
                pad.shape.dims()
            )));
        }
        self.norm.forward(f, values, mean, rstd, pad)?;
        Ok(self.conv.forward(f, pad, out)?)
    }
}

/// A `[channels, frames, rows, columns]` raster as the one sample a convolution reads.
fn batched(view: TensorView) -> TensorView {
    let dims = view.shape.dims();
    view.reshape(Shape::new(&[1, dims[0], dims[1], dims[2], dims[3]]))
}

/// The extent of a five-axis tensor: `(batch, channels, frames, height, width)`.
fn extent(view: TensorView) -> Result<(u64, u64, u64, u64, u64)> {
    if view.shape.rank() != 5 {
        return Err(Error::config(format!(
            "a video tensor is {:?}, five axes expected",
            view.shape.dims()
        )));
    }
    Ok((
        view.shape.dim(0),
        view.shape.dim(1),
        view.shape.dim(2),
        view.shape.dim(3),
        view.shape.dim(4),
    ))
}

/// The distinct extents of a run, in the order they are first named.
///
/// A grid names every tile's extent and a uniform grid names the same one 28 times; the reservation is per
/// distinct extent, so this is what keeps a uniform run to one set. The order is first-seen because a
/// plan's sets are reported in reservation order.
fn distinct(extents: &[Extent]) -> Vec<Extent> {
    let mut seen: Vec<Extent> = Vec::with_capacity(extents.len());
    for extent in extents {
        if !seen.contains(extent) {
            seen.push(*extent);
        }
    }
    seen
}

/// One clip of a condition: the frames it reads and the latent frames it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipSpan {
    /// The clip's first frame.
    pub first: u32,
    /// Its frames.
    pub frames: u32,
    /// The first latent frame it writes.
    pub latent: u32,
    /// The latent frames it writes.
    pub latents: u32,
}

/// The buffers one tile extent's encode works in, reserved once for a whole run of tiles.
///
/// A run's tiles are not uniform - the right and bottom edges of a grid are shorter - so a scratch holds
/// one set per **distinct** tile extent and each call takes the set its own extent names. A set's buffers
/// are then exactly the call's shape, which matters because a convolution writes the view it is handed:
/// a sub-window of a larger buffer would be strided, and the convolution path takes a destination's
/// address and shape, not its strides.
pub struct EncoderScratch {
    sets: Vec<(Extent, EncoderSet)>,
}

impl EncoderScratch {
    /// The set a tile of this extent encodes in.
    fn of(&self, extent: Extent) -> Result<&EncoderSet> {
        self.sets
            .iter()
            .find(|(known, _)| *known == extent)
            .map(|(_, set)| set)
            .ok_or_else(|| {
                Error::config(format!(
                    "no encoder buffers were reserved for a {} by {} tile of {} frames",
                    extent.height, extent.width, extent.frames
                ))
            })
    }
}

/// One tile extent's buffers: the input pad, the input convolution's output and a set per level.
///
/// The moment convolution writes the caller's `out`, so no buffer is held for it: the tile's result is
/// the moments, and the one-by-one projection that follows runs once, on the stitched raster
/// ([`VideoEncoder::project`]).
struct EncoderSet {
    padded: TensorView,
    stream: TensorView,
    levels: Vec<LevelSet>,
    /// The last normalization's own padded input, mean and inverse standard deviation.
    output_pad: TensorView,
    output_mean: TensorView,
    output_rstd: TensorView,
}

/// One level's buffers: the pair its blocks alternate between, and the pad and strided output where
/// the level downsamples. A block reserves and releases its own working buffers.
///
/// The pair is the level's **own** width, not the buffer the level was handed. Alternating with the
/// incoming buffer instead reads a level's second block against its input width whenever the level widens
/// it - which three of the six levels do - so a two-block level would write a wider value into a narrower
/// view.
struct LevelSet {
    first: TensorView,
    second: TensorView,
    pad: Option<TensorView>,
    strided: Option<TensorView>,
}

impl VideoEncoder {
    /// Reserve the buffers one run of tiles encodes in, one set per distinct tile extent.
    ///
    /// The arena never rewinds, so a driver that encodes a grid of tiles must reserve once and hand the
    /// same scratch to every tile: a reservation inside the loop is multiplied by the tile count, and one
    /// tile's buffers are most of a gigabyte at the served canvas.
    pub fn reserve(&self, f: &Forward, extents: &[Extent]) -> Result<EncoderScratch> {
        if extents.is_empty() {
            return Err(Error::config(
                "an encoder run with no tile extent has nothing to reserve".to_owned(),
            ));
        }
        let mut sets: Vec<(Extent, EncoderSet)> = Vec::new();
        for extent in distinct(extents) {
            sets.push((extent, self.set(f, extent)?));
        }
        Ok(EncoderScratch { sets })
    }

    /// The buffers one tile extent needs.
    fn set(&self, f: &Forward, raster: Extent) -> Result<EncoderSet> {
        let extents = self.stack.extents(raster);
        let buffer = |channels: u64, extent: &Extent| -> Result<TensorView> {
            Ok(f.arena().tensor(
                DType::F32,
                Shape::new(&[
                    1,
                    channels,
                    u64::from(extent.frames),
                    u64::from(extent.height),
                    u64::from(extent.width),
                ]),
            )?)
        };
        // One normalization's padded input: one pixel wider and taller on each side and two frames in
        // front, exactly the shape `Normalized::run` checks.
        let pad_buffer = |channels: u64, extent: &Extent| -> Result<TensorView> {
            Ok(f.arena().tensor(
                DType::F32,
                Shape::new(&[
                    1,
                    channels,
                    u64::from(extent.frames) + 2,
                    u64::from(extent.height) + 2,
                    u64::from(extent.width) + 2,
                ]),
            )?)
        };
        let moments_buffer = |groups: u64, extent: &Extent| -> Result<TensorView> {
            Ok(f.arena()
                .tensor(DType::F32, Shape::new(&[u64::from(extent.frames), groups]))?)
        };
        let padded = pad_buffer(u64::from(self.stack.input_channels), &raster)?;
        let stream = buffer(u64::from(self.stack.widths[0]), &extents[0])?;
        let mut levels = Vec::with_capacity(extents.len() - 1);
        for index in 0..extents.len() - 1 {
            let width = u64::from(self.stack.widths[index]);
            let extent = &extents[index];
            let next = &extents[index + 1];
            let downsample = self.levels[index].downsample.is_some();
            levels.push(LevelSet {
                first: buffer(width, extent)?,
                second: buffer(width, extent)?,
                pad: match downsample {
                    true => Some(f.arena().tensor(
                        DType::F32,
                        Shape::new(&[
                            1,
                            width,
                            u64::from(extent.frames) + 2,
                            u64::from(extent.height) + 1,
                            u64::from(extent.width) + 1,
                        ]),
                    )?),
                    false => None,
                },
                strided: match downsample {
                    true => Some(buffer(width, next)?),
                    false => None,
                },
            });
        }
        let last = extents[extents.len() - 1];
        let output_pad = pad_buffer(u64::from(self.output.channels), &last)?;
        let output_groups = u64::from(self.output.groups);
        let output_mean = moments_buffer(output_groups, &last)?;
        let output_rstd = moments_buffer(output_groups, &last)?;

        Ok(EncoderSet {
            padded,
            stream,
            levels,
            output_pad,
            output_mean,
            output_rstd,
        })
    }

    /// Encode one tile into a reserved set of buffers.
    ///
    /// This is the entry point a driver of several tiles uses: it reserves once with [`Self::reserve`]
    /// and hands the same scratch to every tile, which is the difference between one tile's buffers and
    /// one tile's buffers times the grid. `pixels` is `[1, in_channels, frames, height, width]` FP32 -
    /// one tile of a clip, normalized by [`normalize`] - and `out` is
    /// `[1, 2 · latent_channels, frames', height / 16, width / 16]`. A tile whose extent was not reserved
    /// is refused by name rather than read past the buffers.
    pub fn encode_tile(
        &self,
        f: &Forward,
        scratch: &EncoderScratch,
        pixels: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let (batch, channels, frames, height, width) = extent(pixels)?;
        if batch != 1 {
            return Err(Error::config(format!(
                "the video encoder reads one tile at a time, not {batch}"
            )));
        }
        if channels != u64::from(self.stack.input_channels) {
            return Err(Error::config(format!(
                "{channels} input channels for a stack of {}",
                self.stack.input_channels
            )));
        }
        if frames == 0 || frames > u64::from(self.stack.clip.frames) {
            return Err(Error::config(format!(
                "a video encoder reads one still or one clip of at most {} frames, not {frames}",
                self.stack.clip.frames
            )));
        }
        let raster = Extent {
            frames: u32::try_from(frames)
                .map_err(|_| Error::config("too many frames".to_owned()))?,
            height: u32::try_from(height).map_err(|_| Error::config("too tall".to_owned()))?,
            width: u32::try_from(width).map_err(|_| Error::config("too wide".to_owned()))?,
        };
        let set = scratch.of(raster)?;

        // The input convolution reads padded pixels; it has no normalization of its own.
        CausalPad {
            top: 1,
            left: 1,
            front: 2,
            padding: self.padding,
        }
        .forward(f, pixels, set.padded)?;
        self.input.forward(f, set.padded, set.stream)?;

        // A level's blocks alternate between its own pair of buffers, and the level's input is a third
        // buffer it never writes: the first block reads the previous level's output, each later block
        // reads the one before it, and every block writes the level's width.
        let mut source = set.stream;
        for (index, level) in self.levels.iter().enumerate() {
            let buffers = &set.levels[index];
            for (at, block) in level.blocks.iter().enumerate() {
                let target = match at % 2 {
                    0 => buffers.first,
                    _ => buffers.second,
                };
                block.forward(f, source, target)?;
                source = target;
            }
            if let Some(downsample) = &level.downsample {
                let (Some(pad), Some(strided)) = (buffers.pad, buffers.strided) else {
                    return Err(Error::config(format!(
                        "level {index} downsamples without the buffers for it"
                    )));
                };
                // The strided downsampler pads only after the source (its view is one row and
                // one column larger), and two zero frames in front.
                CausalPad {
                    top: 0,
                    left: 0,
                    front: 2,
                    padding: self.padding,
                }
                .forward(f, source, pad)?;
                downsample.forward(f, pad, strided)?;
                source = strided;
            }
        }

        // The moment convolution writes `2 · latent_channels` into the caller's view. The one-by-one
        // projection that follows it runs **once for the whole raster**, after the tiles are stitched
        // (`project`), because that is the reference's order: the tiles produce moments, the tiler
        // blends them, and the projection reads the stitched result.
        self.output.run(
            f,
            source,
            set.output_pad,
            set.output_mean,
            set.output_rstd,
            out,
        )
    }

    /// The latent frames a **condition** of `frames` frames encodes to.
    ///
    /// A condition is cut into clips of `Stack::clip_frames` frames, each yielding
    /// `Stack::clip_latents` latent frames, and the last clip's trailing `Stack::token_drop` latent
    /// frames are dropped - so 17 frames are 2 latent frames and 22 are 7. A single frame is a still and
    /// yields one (the network's causal front padding makes one frame one latent frame).
    ///
    /// This is a **different statement** from the generated timeline's `packing::video_latent_frames`
    /// (`17n + 5` frames to `5n + 2` latents). The two agree on the counts a generation uses and this
    /// one alone accepts the counts a condition may be - 17, 30, any length - because a condition is
    /// cut and padded, not generated.
    pub fn latent_frames(&self, frames: u32) -> Result<u32> {
        self.stack.clip.latent_frames(frames)
    }

    /// The clips a **condition** of `frames` frames is cut into, in order ([`Clip::spans`]).
    pub fn clips(&self, frames: u32) -> Result<Vec<ClipSpan>> {
        self.stack.clip.spans(frames)
    }

    /// Project the stitched moments to the posterior's two fields: one call, after the stitch.
    ///
    /// The reference's `Encoder` wraps the tiled network and runs `quant_conv` on the stitched
    /// moments, so a tile's own output is not yet projected. A one-by-one convolution commutes with the
    /// blend in real arithmetic - the blend's weights do not depend on the channel - so projecting per
    /// tile would be the same value for more work; it is written this way because it is also the order
    /// the posterior is sampled in, which does depend on it: the sample reads a mean and a log-variance
    /// of the same position, and a per-tile projection would entangle two tiles' roundings before they
    /// were blended.
    ///
    /// `moments` is a tile's or the whole run's `[2 · latent_channels, frames, rows, columns]`; `out` is
    /// the same shape. Nothing is reserved: the projection reads and writes the caller's views.
    pub fn project(&self, f: &Forward, moments: TensorView, out: TensorView) -> Result<()> {
        self.moments
            .forward(f, batched(moments), batched(out))
            .map_err(Error::from)
    }

    /// The posterior sample a condition reads: `moments` and the runner's `noise` to `out`.
    ///
    /// The reference's `DiagonalGaussian` and the tail of `LatentEncoder`: the channel axis splits into
    /// mean and log-variance, the log-variance is clamped to `(-30, 20)`, scaled by `exp(0.5 · ·)`,
    /// mixed with the noise and added to the mean; the sum is rounded through FP16 **as a step**; and
    /// the per-channel affine of [`normalize_tables`] turns it into the latent the conditioner reads.
    /// The epilogue is `ops::posterior`, exact by contract, so the only rounding that differs from
    /// torch is the affine's fused multiply-add ([`normalize_tables`] declares it).
    ///
    /// `moments` is the projected moments of the whole stitched raster and `out` is the condition's
    /// latent, both `[channels, frames, rows, columns]` FP32; `noise` is the runner's draw in the same
    /// order - the reference's `condition_noise_shapes` is channel-major and the draw is addressed by
    /// element index, so that order is what reproduces it. The operator addresses every operand
    /// through two strides, so a channel-major raster needs **no copy and no transpose**: the moments'
    /// two halves are one buffer cut by an offset, the sample is written into one scratch raster (the
    /// affine cannot read and write `out` at once), and the affine reads that raster into `out`.
    pub fn posterior(
        &self,
        f: &Forward,
        moments: TensorView,
        noise: TensorView,
        out: TensorView,
    ) -> Result<()> {
        if out.shape.rank() != 4 || out.shape.dim(0) != u64::from(self.stack.latent_channels) {
            return Err(Error::config(format!(
                "the condition's latent is [{}, frames, rows, columns], not {:?}",
                self.stack.latent_channels,
                out.shape.dims()
            )));
        }
        // The sample is written into one scratch raster, because the affine cannot read and write
        // `out` at once.
        let drawn = f.arena().tensor(DType::F32, out.shape)?;
        self.sampler.sample(f, moments, noise, drawn)?;
        self.normalize
            .forward(f, batched(drawn), batched(out))
            .map_err(Error::from)
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
    mlp: GeGlu,
}

/// A normalized tanh-GELU gated linear unit: the reference's `GeGLU`.
struct GeGlu {
    norm: LayerNorm,
    /// `output(gelu_tanh(gate(x)) · value(x))`, the gated product in one rounding: the reference's
    /// `F.gelu(gate, approximate="tanh") * value`.
    mlp: GatedMlp,
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
        let frames = audio_latent_frames(samples, self.hop())?;
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

    /// Encode a `[batch, 1, length]` track into `out [batch · frames, latent_channels]` F32 rows of
    /// the normalized latent, `(mode - mean) / std`.
    ///
    /// The encoder is mono per batch row (it encodes `[batch, 1, samples]` waveforms) and its
    /// latent is a **two-row batch**: a stereo reference arrives as `[2, 1, samples]` and its
    /// latent is `[2, latent_channels, frames]`, the same two-row shape the audio decoder reads.
    /// The rows are written channel-major, `[batch · frames, latent_channels]` with batch
    /// outermost, because that is the form the DiT's audio conditioning reads (`[2 · frames,
    /// audio_channels]`) - so no caller transposes.
    ///
    /// The length must be a whole number of hops and no longer than the deployment's bound. The
    /// activation of every stage runs at the stage's inner width, which is half of the width the
    /// widening convolution writes. The projection attends over frame 2, so it runs once per batch row
    /// and never mixes them.
    pub fn encode(&self, f: &Forward, samples: TensorView, out: TensorView) -> Result<()> {
        let batch = Self::check_track(samples.shape)?;
        let length = samples.shape.dim(2);
        let frames = self.frames(length)?;
        // The network's stages divide the length exactly, so a track that is not a whole number of
        // latent frames must be right-padded with zeros to one - what the reference's `unpack_samples`
        // does before it convolves. **The pad is the caller's**, and that is not a punt: the padded
        // span depends on the request's own sample count, so it is a request buffer, which the runner
        // allocates and may fill on the host before it uploads. `padded_samples` states the length and
        // this refusal states the length too, so a caller that meets it needs no operator: truncating
        // instead would drop the last latent frame, which is why this is a refusal and not a rounding.
        let hop = self.hop();
        if !length.is_multiple_of(hop) {
            let padded = padded_samples(length, hop)?;
            return Err(Error::config(format!(
                "{length} samples is not a whole number of {hop}-sample latent frames: hand this call \
                 the track right-padded with zeros to {padded} samples (`padded_samples`), as the \
                 reference's `unpack_samples` does before it convolves - the pad is a request \
                 buffer's tail, so it is the caller's to fill"
            )));
        }
        if frames > u64::from(self.frames) {
            return Err(Error::config(format!(
                "a {frames}-frame track is longer than the {} the deployment declared",
                self.frames
            )));
        }
        let channels = u64::from(self.config.latent_channels);
        let rows = batch
            .checked_mul(frames)
            .ok_or_else(|| Error::config("the audio latent's row count overflows".to_owned()))?;
        if out.shape.rank() != 2 || out.shape.dims() != [rows, channels] {
            return Err(Error::config(format!(
                "the audio encoder writes [{rows}, {channels}] channel-major rows, not {:?}",
                out.shape.dims()
            )));
        }

        let extent = frames;
        let wide = u64::from(self.config.latent_dim);
        let features = f
            .arena()
            .tensor(DType::F32, Shape::new(&[batch, wide, extent]))?;
        self.encoder.forward(f, samples, features)?;

        // The projection attends over the frame axis, so it runs once per batch row and never mixes
        // rows: a row's features are a `[wide, extent]` matrix, transposed to frame-major for the
        // attention, and the mean head's `[channels, extent]` is transposed again into the row-major
        // channel-major block the conditioning reads.
        let per_row = wide * extent;
        for row in 0..batch {
            let features = TensorView::new(
                features.ptr.offset(row * per_row * 4),
                DType::F32,
                Shape::new(&[wide, extent]),
            );
            let frames_view = f.arena().tensor(DType::F32, Shape::new(&[extent, wide]))?;
            self.transpose.forward(f, features, frames_view)?;
            let pooled = self.projection.forward(f, frames_view, extent)?;
            let pooled_matrix = f
                .arena()
                .tensor(DType::F32, Shape::new(&[channels, extent]))?;
            self.transpose.forward(f, pooled, pooled_matrix)?;
            let batched = TensorView::new(
                pooled_matrix.ptr,
                DType::F32,
                Shape::new(&[1, channels, extent]),
            );
            let narrowed = f
                .arena()
                .tensor(DType::F32, Shape::new(&[1, channels, extent]))?;
            self.mean.forward(f, batched, narrowed)?;
            let normalized = f
                .arena()
                .tensor(DType::F32, Shape::new(&[1, channels, extent]))?;
            self.normalize.forward(f, narrowed, normalized)?;
            // One row's latent, channel-major rows: `[extent, channels]` at the row's place in `out`.
            let plane =
                TensorView::new(normalized.ptr, DType::F32, Shape::new(&[channels, extent]));
            let written = TensorView::new(
                out.ptr.offset(row * extent * channels * 4),
                DType::F32,
                Shape::new(&[extent, channels]),
            );
            self.transpose.forward(f, plane, written)?;
        }
        Ok(())
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
        let mlp = GeGlu::load(config, &mut scope.scope("mlp"))?;
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
            mlp,
        })
    }

    /// Project `values [frames, latent_dim]` to `[frames, latent_channels]`.
    fn forward(&self, f: &Forward, values: TensorView, frames: u64) -> Result<TensorView> {
        let shape = values.shape;
        let channels = u64::from(self.channels);
        let heads = shape.dim(1) / u64::from(self.attention.spec().head_dim);
        let shortcut = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        let normed = f.arena().tensor(DType::F32, shape)?;
        self.shortcut_norm.forward(f, values, None, normed)?;
        self.shortcut.forward(f, normed, shortcut)?;
        self.attention_norm.forward(f, values, None, normed)?;
        // Three contiguous head-group blocks, then causal attention over the frames.
        let head_shape = Shape::new(&[frames, heads, u64::from(self.attention.spec().head_dim)]);
        let attended = f.arena().tensor(DType::F32, head_shape)?;
        let mut head_views = Vec::with_capacity(3);
        for linear in [&self.query, &self.key, &self.value] {
            let out = f.arena().tensor(DType::F32, shape)?;
            linear.forward(f, normed, out)?;
            head_views.push(TensorView::new(out.ptr, DType::F32, head_shape));
        }
        self.attention.forward(
            f,
            head_views[0],
            head_views[1],
            head_views[2],
            None,
            attended,
        )?;
        let pooled = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.pool.forward(
            f,
            attended,
            pooled.reshape(Shape::new(&[frames, 1, channels])),
        )?;
        let projected = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.attention_output.forward(f, pooled, projected)?;
        Residual::default().forward(f, shortcut, projected, None, shortcut)?;
        // The normalized GeGLU residual.
        let normed = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.mlp_norm.forward(f, shortcut, None, normed)?;
        let updated = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.mlp.forward(f, normed, updated)?;
        Residual::default().forward(f, shortcut, updated, None, shortcut)?;
        Ok(shortcut)
    }
}

impl GeGlu {
    /// The normalized tanh-GELU gated linear unit at `scope` over the latent channels: its `norm`,
    /// the gate `w0`, the value `w1` and the output `w2`.
    fn load(config: &crate::AudioVaeConfig, scope: &mut Scope<'_, '_>) -> Result<Self> {
        let channels = config.latent_channels;
        Ok(Self {
            norm: LayerNorm::load(&mut scope.scope("norm"), channels, NORM_EPS)?,
            mlp: GatedMlp::load(
                scope,
                GatedMlpSpec {
                    hidden: channels,
                    intermediate: 2 * channels,
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
            )?,
        })
    }

    /// `output(gelu_tanh(gate(norm(x))) · value(norm(x)))` into `out`.
    fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        let normed = f.arena().tensor(DType::F32, x.shape)?;
        self.norm.forward(f, x, None, normed)?;
        Ok(self.mlp.forward(f, normed, out)?)
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

    /// A video encoder over the fixture's config: the stack is all `latent_frames` needs, so no
    /// weights are loaded.
    fn encoder_stack() -> Stack {
        Stack::of(&video_config()).expect("the fixture's stack")
    }

    /// A condition's clips cover its frames and its latent frames, whole clips and then what is
    /// left, the reference's `clips` and `latent_slices`.
    #[test]
    fn a_condition_is_cut_into_its_clips() {
        let clip = encoder_stack().clip;
        assert_eq!(
            clip.spans(1).unwrap(),
            [ClipSpan {
                first: 0,
                frames: 1,
                latent: 0,
                latents: 1
            }]
        );
        let video = clip.spans(39).unwrap();
        assert_eq!(video.len(), 3);
        assert_eq!(
            video[2],
            ClipSpan {
                first: 34,
                frames: 5,
                latent: 10,
                latents: 2
            }
        );
        for frames in [22, 39, 56, 124, 362] {
            let spans = clip.spans(frames).unwrap();
            let last = spans.last().unwrap();
            assert_eq!(last.first + last.frames, frames);
            assert_eq!(
                last.latent + last.latents,
                clip.latent_frames(frames).unwrap()
            );
        }
    }

    #[test]
    fn a_condition_is_counted_by_its_own_rule() {
        // The reference counts a condition in clips: every clip of `clip_length` frames yields
        // `clip_latents` latent frames and the last clip's trailing `token_drop` are dropped, so the
        // canonical 22 frames (`clip_length + 5`) are two clips less three.
        let clip = encoder_stack().clip;
        assert_eq!(clip.latent_frames(1).expect("a still"), 1);
        assert_eq!(
            clip.latent_frames(clip.frames).expect("one clip"),
            clip.latents - clip.token_drop
        );
        let tail = clip.frames + crate::config::CLIP_LATENTS;
        assert_eq!(
            clip.latent_frames(tail).expect("a clip and a tail"),
            2 * clip.latents - clip.token_drop
        );
        // A length no generation produces is a condition all the same: two clips and a short tail.
        assert_eq!(
            clip.latent_frames(2 * clip.frames - 4)
                .expect("a short last clip"),
            2 * clip.latents - clip.token_drop
        );
        assert!(clip.latent_frames(0).is_err());

        // The generated timeline's rule agrees on the canonical count, which is why the two must be
        // stated separately rather than one standing in for the other.
        assert_eq!(
            clip.latent_frames(tail).expect("a clip and a tail"),
            crate::packing::video_latent_frames(tail).expect("the generated rule")
        );
    }

    #[test]
    fn a_run_reserves_one_set_per_distinct_tile_extent() {
        let tile = |height: u32, width: u32| Extent {
            frames: 25,
            height,
            width,
        };
        // A uniform grid names one extent many times and reserves once; a grid with shorter edges names
        // two, in the order they first appear.
        assert_eq!(distinct(&[tile(256, 256); 28]), vec![tile(256, 256)]);
        assert_eq!(
            distinct(&[
                tile(256, 256),
                tile(256, 192),
                tile(256, 256),
                tile(256, 192)
            ]),
            vec![tile(256, 256), tile(256, 192)]
        );
        assert_eq!(
            distinct(&[tile(192, 192), tile(256, 256)]),
            vec![tile(192, 192), tile(256, 256)],
            "the order is the order they are named"
        );
    }

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
    fn a_clip_of_seventeen_frames_becomes_five_latent_frames() {
        let stack = Stack::of(&video_config()).expect("the stack");
        let extents = stack.extents(Extent {
            frames: 17,
            height: 256,
            width: 256,
        });
        // 17 frames, then 17, 9, 5, 5, 5, 5: one 17-frame clip is five latent frames.
        assert_eq!(
            extents
                .iter()
                .map(|extent| extent.frames)
                .collect::<Vec<_>>(),
            vec![17, 17, 9, 5, 5, 5, 5]
        );
        // 256 pixels halve four times: the 16-pixel compression.
        assert_eq!(
            extents
                .iter()
                .map(|extent| extent.height)
                .collect::<Vec<_>>(),
            vec![256, 128, 64, 32, 16, 16, 16]
        );
        assert_eq!(stack.moments(extents[0]).expect("moments").frames, 5);
        // A single frame stays one latent frame.
        let single = stack
            .moments(Extent {
                frames: 1,
                height: 16,
                width: 16,
            })
            .expect("moments");
        assert_eq!(single.frames, 1);
        assert_eq!((single.height, single.width), (1, 1));
    }

    #[test]
    fn the_audio_encoder_stages_land_on_whole_latent_frames() {
        let stack = AudioStack::of(&crate::audio_vae::tests::audio_config()).expect("the stack");
        assert_eq!(stack.samples_per_latent, 800);
        assert_eq!(stack.lengths(800), vec![800, 400, 100, 25, 5, 1]);
        assert_eq!(stack.lengths(1600), vec![1600, 800, 200, 50, 10, 2]);
        assert_eq!(stack.latent_frames(800).expect("one frame"), 1);
        assert_eq!(stack.latent_frames(24000).expect("thirty frames"), 30);
        // A partial frame rounds up, and the length the network is given is the ceiling's own multiple:
        // the reference right-pads the track with zeros, so nothing is dropped.
        assert_eq!(stack.latent_frames(801).expect("rounds up"), 2);
        assert_eq!(padded_samples(801, 800).expect("padded"), 1600);
        assert_eq!(padded_samples(1600, 800).expect("whole"), 1600);
        // Each stage doubles the width from the input convolution's.
        assert_eq!(stack.widths, vec![128, 256, 512, 1024, 2048]);
        assert_eq!(stack.latent_dim, 2048);
    }

    #[test]
    fn a_stack_with_no_stage_is_refused() {
        let mut config = video_config();
        config.block_out_channels.clear();
        assert!(Stack::of(&config).is_err());
        let mut config = video_config();
        config.spatial_downsample_factors.pop();
        assert!(Stack::of(&config).is_err());
        let mut config = video_config();
        config.temporal_downsample_factors[1] = 0;
        assert!(Stack::of(&config).is_err());
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
