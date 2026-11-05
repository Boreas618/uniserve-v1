//! The H3 condition encoders: the causal video encoder that turns a clip of raster frames into
//! posterior moments, and the audio encoder that turns a reference waveform into latents.
//!
//! Only `ref2va` reads them (specs/worker.md 9): a reference video or audio clip is encoded once at
//! the request boundary and the latents join the packed sequence as clean condition rows. The rest of
//! this module is the encoders' arithmetic — the stage table, the causal extents, the tile grid and
//! the band rule a partial encoding follows — and their parameter tables. The numerical bodies follow
//! the shared layers, as the decoders' do.
//!
//! `uniserve/models/minimax_h3/video_vae.py` (`CausalCNN`, `Encoder`) and `uniserve/nn/vae/tiled.py`
//! (`TiledEncoder`) are the reference.
//!
//! ## Precision
//!
//! Every convolution of the video encoder runs FP32 with TF32 allowed — `specs/worker.md` 9's
//! `conv3d` row — because the checkpoint's VAE encoder is evaluated at the quality preset. The audio
//! encoder's convolutions and its `Snake` activations run FP32 and keep no TF32: its magnitudes are
//! amplitudes, where the reference compares samples, not feature maps.
//!
//! **The audio encoder's attention is FP32 throughout** — query, key and value projections, the
//! scores, the softmax and the output — whatever the checkpoint's preset says. `specs/worker.md` 9
//! states the row in its own words: 「FP32 注意力（`ref2va` 的音频编码器、VSA 压缩分支） | cuBLASLt
//! SGEMM 加 softmax kernel program」 — FP32 attention for `ref2va`'s audio encoder and the VSA
//! compression branch, as a cuBLASLt SGEMM plus a softmax kernel program. It is the declaration §9
//! asks the model to make instead of leaning on `torch.autocast`, and it is the opposite of the
//! video VAE decoder's rule (one FP16 element type for parameters and residual). The layer is
//! [`nn::layers::Attention::fp32`]; no call site may construct the dense backend for these blocks.
//!
//! ## The encoder's input
//!
//! A reference video reaches the encoder as `[frames, height, width, 3]` **uint8** pixels — the same
//! raster the decoder writes ([`crate::output::Rgb`]) — and is normalized per channel by
//! `(pixel / 255 − mean) / std` with the ImageNet constants before the input convolution
//! (`uniserve/models/minimax_h3/encoding.py`, `unpack_pixels`). A clip shorter than the checkpoint's
//! [`crate::config::CLIP_FRAMES`] repeats its last frame; a single frame stays alone.
//!
//! ## Causal extents
//!
//! A stride-`s` temporal convolution pads two zero frames in front and none behind, so it maps `T`
//! frames to `ceil(T / s)`: one 17-frame clip becomes five latent frames, and a single frame stays
//! one. A spatial downsampling pads the bottom and the right by one pixel and strides by two, so it
//! halves an even extent exactly and floors an odd one.
//!
//! ## Tiles and bands
//!
//! A tile is 256 input pixels square with at least 64 pixels of overlap, both multiples of the 16-pixel
//! spatial compression, so every seam spans whole latent positions. Adjacent tiles cross-fade over
//! `overlap / 16` latent positions in the encoder's output precision — unlike the decoder, which blends
//! decoded pixels, the encoder blends latents. A band of latent rows can be encoded on its own: the
//! tile rows it reads are chosen so that the band's rows equal those rows of the whole encoding.

use crate::audio_vae::Binder;

/// `nn.LayerNorm`'s default epsilon, which the reference's audio projection does not override.
const NORM_EPS: f32 = 1e-5;
use crate::config::VideoVaeConfig;
use crate::video_vae::{MINIMUM_OVERLAP, Seams, TILE_PIXELS};
use crate::{Error, Result};

/// The spatial compression of the video encoder: input pixels per latent position.
pub const SPATIAL_RATIO: u32 = 16;

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
    /// padding makes one frame one latent frame (`video_vae.py:1105-1113`), and the reference encodes
    /// an image or a keyframe through the same network alone (`encoding.py:41-51`).
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
}

impl Stack {
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

    /// The tile grid of a raster, in input pixels, and the latent extent it produces.
    pub fn grid(&self, height: u32, width: u32) -> Result<Grid> {
        Grid::of(height, width)
    }
}

/// The tile grid of one raster, in input pixels, with the latent positions each seam shares.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Grid {
    /// The tile rows.
    pub rows: Seams,
    /// The tile columns.
    pub columns: Seams,
}

impl Grid {
    /// The grid of a raster whose extents are multiples of the spatial compression.
    pub fn of(height: u32, width: u32) -> Result<Self> {
        Ok(Self {
            rows: Seams::split(height, TILE_PIXELS, MINIMUM_OVERLAP, SPATIAL_RATIO)?,
            columns: Seams::split(width, TILE_PIXELS, MINIMUM_OVERLAP, SPATIAL_RATIO)?,
        })
    }

    /// The latent rows each tile row owns: where it starts in latent rows, then the latent height.
    ///
    /// A stitched tile row owns the latent rows from its own start up to the next row's start, so the
    /// first seams of a run cross-fade and the last row keeps its whole lower overlap.
    pub fn owned(&self, latent_height: u32) -> Vec<u32> {
        let mut owned: Vec<u32> = self
            .rows
            .starts
            .iter()
            .map(|start| start / SPATIAL_RATIO)
            .collect();
        owned.push(latent_height);
        owned
    }

    /// The tile rows a band of latent rows `[start, stop)` reads, as an inclusive range.
    ///
    /// The band's first tile row is the last one starting at or before `start`; it is preceded by one
    /// more row when `start` falls inside the cross-fade that row shares, because those latent rows
    /// read the row above. The last row is the last one starting before `stop`.
    pub fn band(&self, latent_height: u32, start: u32, stop: u32) -> Result<(usize, usize)> {
        if start >= stop || stop > latent_height {
            return Err(Error::config(format!(
                "the latent band [{start}, {stop}) is not a run of the {latent_height} rows"
            )));
        }
        let owned = self.owned(latent_height);
        let first = (0..self.rows.len())
            .rev()
            .find(|index| owned[*index] <= start)
            .ok_or_else(|| Error::config("the band starts before the first tile".to_owned()))?;
        let last = (0..self.rows.len())
            .rev()
            .find(|index| owned[*index] < stop)
            .ok_or_else(|| Error::config("the band ends before the first tile".to_owned()))?;
        let begin = match first.checked_sub(1) {
            Some(above) if start < owned[first] + self.rows.overlaps[above] / SPATIAL_RATIO => {
                above
            }
            _ => first,
        };
        Ok((begin, last))
    }

    /// The latent positions each seam of an axis cross-fades.
    pub fn blends(seams: &Seams) -> Vec<u32> {
        seams
            .overlaps
            .iter()
            .map(|overlap| overlap / SPATIAL_RATIO)
            .collect()
    }
}

/// One pixel's normalized channels: `(pixel / 255 − mean) / std`.
///
/// The inverse of [`crate::output::Rgb::quantize`], and the input the video encoder reads. A kernel
/// that normalizes the raster computes this; the round trip through the decoder's quantization
/// recovers the byte.
pub fn normalize(pixel: [u8; 3]) -> [f32; 3] {
    let mut values = [0.0f32; 3];
    for (channel, byte) in pixel.iter().enumerate() {
        values[channel] = (f32::from(*byte) / crate::output::BYTE_MAX
            - crate::output::PIXEL_MEAN[channel])
            / crate::output::PIXEL_STD[channel];
    }
    values
}

impl VideoEncoder {
    /// Every checkpoint tensor the video encoder reads, by name.
    ///
    /// The encoder is a component whose names were already data (`sources` maps each parameter to
    /// the checkpoint key that fills it), so this list is that mapping's other column rather than a
    /// second spelling of it: a weight-normalized pair names both of its tensors, and a fused
    /// request names every part. A checkpoint that lacks one is refused by block and name before a
    /// byte is read.
    pub fn keys(config: &VideoVaeConfig) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        for source in sources(config)? {
            match source.tensor {
                crate::audio_vae::Tensor::Region(tensor) => keys.push(tensor),
                crate::audio_vae::Tensor::Parts(parts) => keys.extend(parts),
                crate::audio_vae::Tensor::Normalized { value, gain } => {
                    keys.push(value);
                    keys.push(gain);
                }
            }
        }
        Ok(keys)
    }
}

/// Every parameter of the video encoder, with the checkpoint key that fills it.
///
/// The encoder is the input convolution, six stages of two blocks each, the final normalization and
/// the moment convolution, plus the one-by-one `quant_conv` after it: 118 tensors in the released
/// checkpoint. A shortcut convolution exists only where a block changes width, and a downsampling
/// only where a stage strides.
pub fn sources(config: &VideoVaeConfig) -> Result<Vec<crate::audio_vae::Source>> {
    use crate::audio_vae::{Source, Tensor};

    let stack = Stack::of(config)?;
    let mut sources = Vec::new();
    let mut push = |parameter: String, tensor: String| {
        sources.push(Source {
            parameter,
            tensor: Tensor::Region(tensor),
        });
    };

    for (parameter, tensor) in [
        ("quant_conv.weight", "quant_conv.weight"),
        ("quant_conv.bias", "quant_conv.bias"),
        ("input.weight", "encoder.conv_in.weight"),
        ("input.bias", "encoder.conv_in.bias"),
        ("norm.weight", "encoder.norm_out.weight"),
        ("norm.bias", "encoder.norm_out.bias"),
        ("output.weight", "encoder.conv_out.weight"),
        ("output.bias", "encoder.conv_out.bias"),
    ] {
        push(parameter.to_owned(), tensor.to_owned());
    }

    let mut input_width = stack.widths[0];
    for (index, width) in stack.widths.iter().enumerate() {
        let stage = format!("encoder.down_blocks.{index}");
        for block in 0..stack.blocks {
            let base = format!("{stage}.resnets.{block}");
            let at = |name: &str| format!("stages.{index}.blocks.{block}.{name}");
            for half in 0..2u32 {
                for leaf in ["weight", "bias"] {
                    push(
                        at(&format!("norms.{half}.{leaf}")),
                        format!("{base}.norm{}.{leaf}", half + 1),
                    );
                    push(
                        at(&format!("convolutions.{half}.{leaf}")),
                        format!("{base}.conv{}.{leaf}", half + 1),
                    );
                }
            }
            // A block whose width changes projects its residual with a one-by-one convolution.
            let shortcut_from = if block == 0 { input_width } else { *width };
            if shortcut_from != *width {
                for leaf in ["weight", "bias"] {
                    push(
                        at(&format!("shortcut.{leaf}")),
                        format!("{base}.conv_shortcut.{leaf}"),
                    );
                }
            }
        }
        // A stage strides through one causal convolution, or not at all.
        if stack.temporal_strides[index] * stack.spatial_strides[index] > 1 {
            for leaf in ["weight", "bias"] {
                push(
                    format!("stages.{index}.downsample.{leaf}"),
                    format!("{stage}.downsamplers.0.conv.{leaf}"),
                );
            }
        }
        input_width = *width;
    }
    Ok(sources)
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
/// One statement of the rule, because two callers need it: the stack's geometry and the encoder's own
/// output extent. The ceiling is the reference's (`encoding.py:244`), and it is a ceiling because its
/// `unpack_samples` right-pads the track with zeros to [`padded_samples`] first, so every strided
/// convolution divides exactly and the last latent frame reads those zeros.
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

/// Every parameter of the audio encoder, with the checkpoint keys that fill it.
///
/// The encoder is DAC's network — the input convolution, five stages of three dilated residual units
/// each and a strided widening convolution, then the output convolution — followed by the latent
/// projection (`pre_block` in the checkpoint) and the mean convolution (`mean_proj`). Every
/// convolution is weight-normalized. The projection's query, key and value biases are three separate
/// checkpoint tensors that fill one fused parameter, the key's being a stored zero vector.
pub fn audio_sources(config: &crate::AudioVaeConfig) -> Result<Vec<crate::audio_vae::Source>> {
    use crate::audio_vae::{Source, Tensor};

    fn region(sources: &mut Vec<Source>, parameter: String, tensor: String) {
        sources.push(Source {
            parameter,
            tensor: Tensor::Region(tensor),
        });
    }
    fn normalized(sources: &mut Vec<Source>, parameter: String, base: String) {
        sources.push(Source {
            parameter,
            tensor: Tensor::Normalized {
                value: format!("{base}.weight_v"),
                gain: format!("{base}.weight_g"),
            },
        });
    }

    let stack = AudioStack::of(config)?;
    let mut sources = Vec::new();

    // `network` is the checkpoint's `encoder.block` sequence: the input convolution, one entry per
    // stage, a Snake, and the output convolution.
    normalized(
        &mut sources,
        "input.weight".to_owned(),
        "encoder.block.0".to_owned(),
    );
    region(
        &mut sources,
        "input.bias".to_owned(),
        "encoder.block.0.bias".to_owned(),
    );
    // Three residual units per stage, at dilations 1, 3 and 9; the dilation names the branch in the
    // plan, not a parameter.
    let units = 3;
    for index in 0..stack.rates.len() {
        let base = format!("encoder.block.{}.block", index + 1);
        let at = |name: &str| format!("stages.{index}.{name}");
        for unit in 0..units {
            normalized(
                &mut sources,
                at(&format!("units.{unit}.convs.0.weight")),
                format!("{base}.{unit}.block.1"),
            );
            region(
                &mut sources,
                at(&format!("units.{unit}.convs.0.bias")),
                format!("{base}.{unit}.block.1.bias"),
            );
            normalized(
                &mut sources,
                at(&format!("units.{unit}.convs.1.weight")),
                format!("{base}.{unit}.block.3"),
            );
            region(
                &mut sources,
                at(&format!("units.{unit}.convs.1.bias")),
                format!("{base}.{unit}.block.3.bias"),
            );
            // Each residual unit's two Snakes carry one frequency per channel.
            for (side, slot) in [("0", 0), ("1", 2)] {
                region(
                    &mut sources,
                    at(&format!("units.{unit}.snakes.{side}.alpha")),
                    format!("{base}.{unit}.block.{slot}.alpha"),
                );
            }
        }
        normalized(&mut sources, at("widen.weight"), format!("{base}.5"));
        region(&mut sources, at("widen.bias"), format!("{base}.5.bias"));
        // The stage's own Snake sits between the last unit and the widening convolution.
        region(&mut sources, at("snake.alpha"), format!("{base}.4.alpha"));
    }
    // The network ends with a Snake and the output convolution.
    region(
        &mut sources,
        "output_snake.alpha".to_owned(),
        format!("encoder.block.{}.alpha", stack.rates.len() + 1),
    );
    let last = stack.rates.len() + 2;
    normalized(
        &mut sources,
        "output.weight".to_owned(),
        format!("encoder.block.{last}"),
    );
    region(
        &mut sources,
        "output.bias".to_owned(),
        format!("encoder.block.{last}.bias"),
    );

    for (parameter, tensor) in [
        ("projection.shortcut_norm.weight", "pre_block.norm3.weight"),
        ("projection.shortcut_norm.bias", "pre_block.norm3.bias"),
        ("projection.shortcut.weight", "pre_block.proj.weight"),
        ("projection.shortcut.bias", "pre_block.proj.bias"),
        ("projection.attention_norm.weight", "pre_block.norm1.weight"),
        ("projection.attention_norm.bias", "pre_block.norm1.bias"),
        ("projection.qkv.weight", "pre_block.attn.qkv.weight"),
        (
            "projection.attention_output.weight",
            "pre_block.attn.proj.weight",
        ),
        (
            "projection.attention_output.bias",
            "pre_block.attn.proj.bias",
        ),
        ("projection.mlp_norm.weight", "pre_block.norm2.weight"),
        ("projection.mlp_norm.bias", "pre_block.norm2.bias"),
        ("projection.mlp.norm.weight", "pre_block.mlp.norm.weight"),
        ("projection.mlp.norm.bias", "pre_block.mlp.norm.bias"),
        ("projection.mlp.gate.weight", "pre_block.mlp.w0.weight"),
        ("projection.mlp.gate.bias", "pre_block.mlp.w0.bias"),
        ("projection.mlp.value.weight", "pre_block.mlp.w1.weight"),
        ("projection.mlp.value.bias", "pre_block.mlp.w1.bias"),
        ("projection.mlp.output.weight", "pre_block.mlp.w2.weight"),
        ("projection.mlp.output.bias", "pre_block.mlp.w2.bias"),
        ("mean_projection.weight", "mean_proj.weight"),
        ("mean_projection.bias", "mean_proj.bias"),
    ] {
        region(&mut sources, parameter.to_owned(), tensor.to_owned());
    }
    // One fused query-key-value bias from three checkpoint vectors, in query, key, value order.
    sources.push(Source {
        parameter: "projection.qkv.bias".to_owned(),
        tensor: Tensor::Parts(vec![
            "pre_block.attn.q_bias".to_owned(),
            "pre_block.attn.zero_k_bias".to_owned(),
            "pre_block.attn.v_bias".to_owned(),
        ]),
    });
    Ok(sources)
}

// ---------------------------------------------------------------------------------------------
// The video encoder's device path.
// ---------------------------------------------------------------------------------------------

use nn::Forward;
use nn::layers::{
    Activation, AdaptivePool, Attention, Conv1d, Conv3d, ConvGeometry, FrameMoments, FrameNormPad,
    FramePad, GatedActivation, LayerNorm, Linear, Posterior, ScaledAdd, Snake, Transpose, WeightOrder,
};
use nn::weights::{Weight, WeightLoader, WeightRequest};
use nn::{DType, Shape, Strides, TensorView};

/// One causal convolution of the encoder and the normalization that feeds it.
///
/// The reference's `CausalConv3d.forward(norm=…)` normalizes each input frame, activates it, pads the
/// result with a reflection of the edge rows and columns and two zero frames in front, and convolves;
/// no output frame therefore reads a later input frame. The bias belongs to the convolution here,
/// where the reference adds it in a second pass to keep its rounding equal to `conv(x) + bias` — the
/// same value, one rounding apart.
struct Normalized {
    norm: FrameNormPad,
    conv: Conv3d,
    /// The source width the normalization reduces.
    channels: u32,
    /// The frame groups its statistics are taken over.
    groups: u32,
    /// The variance floor of those statistics.
    epsilon: f32,
}

/// One residual block: two normalized causal convolutions added to the input's projection.
struct Block {
    first: Normalized,
    second: Normalized,
    /// The one-by-one projection of the input when the width changes.
    shortcut: Option<Conv3d>,
}

/// One encoder level: its blocks, then the convolution that halves space and strides time.
struct Level {
    blocks: Vec<Block>,
    downsample: Option<Conv3d>,
}

/// The video encoder: the causal convolutional network that turns a clip's pixels into the posterior
/// moments of its latent.
///
/// One call encodes one raster: the caller tiles a clip as `Grid` states, and a band of latent rows
/// may be encoded on its own through `Grid::band`. Every convolution runs FP32 with TF32 allowed, the
/// precision the checkpoint's preset declares for the video VAE's encoder (specs/worker.md 9).
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
    /// Whether the spatial padding mirrors the edge rows and columns (the checkpoint's mode) or
    /// replicates them.
    reflect: bool,
}

/// The geometry every causal convolution of the encoder is built for: three taps, no padding of its
/// own (the frame pad supplies it) and unit dilation.
fn geometry(kernel: u32, stride: [u32; 3]) -> Result<ConvGeometry> {
    Ok(ConvGeometry::new(
        3,
        [kernel; 3],
        stride,
        [0; 3],
        [1; 3],
        1,
    )?)
}

/// Read one weight of `shape`.
fn read(loader: &mut WeightLoader, name: &str, shape: &[u64]) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(
        name.to_owned(),
        0,
        Shape::new(shape),
    ))?)
}

/// The per-channel affine of the latent space, in the encoder's direction: `(sample − mean) / std`.
///
/// Multiplying by `1 / std` and shifting by `−mean / std` is a one-by-one convolution whose weight is
/// the diagonal, so the encoder undoes the decoder's `value · std + mean` with one operator of the
/// same shape as the projection beside it (`audio_vae::denormalize_tables` is that other direction).
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

/// A constant weight of `values` in FP32, published once at load.
///
/// The checkpoint has no tensor for the latent statistics: they are config fields, and this is where
/// arithmetic on them becomes a weight the convolution can read. `audio_vae::constants` publishes the
/// decoder's pair the same way.
fn constants(
    loader: &mut WeightLoader<'_>,
    name: &str,
    shape: Shape,
    values: &[f32],
) -> Result<Weight> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    Ok(loader.constant(name, DType::F32, shape, &bytes)?)
}

/// The geometry of one posterior sample: the positions it reads and the channels it writes.
///
/// The sample is `[latent_channels, frames, rows, columns]` like every other raster here, and
/// `ops::posterior` reads `[positions, channels]` - so the call states the positions
/// (`frames · rows · columns`) once, here, and both the strides and the operator's channel count come
/// from it. Every refusal is on this type rather than inside a forward, because a raster of the wrong
/// channel count is a configuration mistake and not a runtime condition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Sample {
    /// Positions the sample has: `frames · rows · columns`.
    pub positions: u64,
    /// Latent channels of one position.
    pub channels: u64,
}

impl Sample {
    /// The sample of `latent_channels` channels over the raster `moments`, `noise` and `out` share.
    ///
    /// `moments` holds both fields of the posterior, so it has twice the sample's channels; `noise` and
    /// `out` hold the sample's own. All three are FP32 rasters of one geometry, packed as the encoder
    /// writes them: the call addresses them through two strides, and a plane that is not packed has no
    /// single pair of them.
    pub fn of(
        latent_channels: u32,
        moments: TensorView,
        noise: TensorView,
        out: TensorView,
    ) -> Result<Self> {
        let channels = u64::from(latent_channels);
        if channels == 0 {
            return Err(Error::config(
                "a latent raster with no channels has no sample".to_owned(),
            ));
        }
        let wide = 2 * channels;
        for (name, view, expected) in [
            ("moments", moments, wide),
            ("noise", noise, channels),
            ("out", out, channels),
        ] {
            if view.dtype != DType::F32 {
                return Err(Error::config(format!(
                    "the {name} raster is {}, and a condition's posterior is FP32 throughout: the \
                     reference draws and samples in f32",
                    view.dtype
                )));
            }
            if view.shape.rank() != 4 {
                return Err(Error::config(format!(
                    "the {name} raster is {:?}, a [channels, frames, rows, columns] raster expected",
                    view.shape.dims()
                )));
            }
            if view.shape.dim(0) != expected {
                return Err(Error::config(format!(
                    "the {name} raster has {} channels, {expected} expected for a {latent_channels}-\
                     channel latent",
                    view.shape.dim(0)
                )));
            }
            let packed = Strides::row_major(view.shape);
            if view.strides != packed {
                return Err(Error::config(format!(
                    "the {name} raster's strides are {:?} over {:?}: a channel-major sample \
                     addresses a packed raster through two strides, and a view is refused rather than \
                     copied",
                    view.strides.as_slice(),
                    view.shape.dims()
                )));
            }
        }
        if noise.shape != out.shape || moments.shape.dims()[1..] != out.shape.dims()[1..] {
            return Err(Error::config(format!(
                "the posterior's rasters disagree: moments {:?}, noise {:?}, out {:?}",
                moments.shape.dims(),
                noise.shape.dims(),
                out.shape.dims()
            )));
        }
        let positions = moments.shape.dims()[1..].iter().product::<u64>();
        Ok(Self {
            positions,
            channels,
        })
    }
}

/// Read one convolution of the checkpoint: `from` input channels, `to` output, `kernel` taps.
fn convolution(
    loader: &mut WeightLoader,
    name: &str,
    from: u32,
    to: u32,
    kernel: u32,
    stride: [u32; 3],
) -> Result<Conv3d> {
    let weight = read(
        loader,
        &format!("{name}.weight"),
        &[
            u64::from(to),
            u64::from(from),
            u64::from(kernel),
            u64::from(kernel),
            u64::from(kernel),
        ],
    )?;
    let bias = read(loader, &format!("{name}.bias"), &[u64::from(to)])?;
    Ok(Conv3d::new(
        weight,
        Some(bias),
        geometry(kernel, stride)?,
        true,
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
        let reflect = matches!(
            config.spatial_padding_mode,
            crate::config::SpatialPadding::Reflect
        );
        // A normalized convolution reads its affine parameters from `norm` and its filter from `conv`.
        let normalized = |loader: &mut WeightLoader,
                          norm_name: &str,
                          conv_name: &str,
                          channels: u32,
                          to: u32,
                          kernel: u32|
         -> Result<Normalized> {
            let weight = read(
                loader,
                &format!("{norm_name}.weight"),
                &[u64::from(channels)],
            )?;
            let bias = read(loader, &format!("{norm_name}.bias"), &[u64::from(channels)])?;
            Ok(Normalized {
                norm: FrameNormPad::new(u64::from(groups), Some(weight), Some(bias), reflect)?,
                conv: convolution(loader, conv_name, channels, to, kernel, [1; 3])?,
                channels,
                groups,
                epsilon,
            })
        };

        let input = convolution(
            loader,
            "encoder.conv_in",
            stack.input_channels,
            stack.widths[0],
            3,
            [1; 3],
        )?;
        let mut width = stack.widths[0];
        let mut levels = Vec::with_capacity(stack.widths.len());
        for (index, out) in stack.widths.iter().enumerate() {
            let stage = format!("encoder.down_blocks.{index}");
            let mut blocks = Vec::with_capacity(stack.blocks as usize);
            for block in 0..stack.blocks {
                let base = format!("{stage}.resnets.{block}");
                let from = if block == 0 { width } else { *out };
                let first = normalized(
                    loader,
                    &format!("{base}.norm1"),
                    &format!("{base}.conv1"),
                    from,
                    *out,
                    3,
                )?;
                let second = normalized(
                    loader,
                    &format!("{base}.norm2"),
                    &format!("{base}.conv2"),
                    *out,
                    *out,
                    3,
                )?;
                let shortcut = if from == *out {
                    None
                } else {
                    Some(convolution(
                        loader,
                        &format!("{base}.conv_shortcut"),
                        from,
                        *out,
                        1,
                        [1; 3],
                    )?)
                };
                blocks.push(Block {
                    first,
                    second,
                    shortcut,
                });
            }
            let temporal = stack.temporal_strides[index];
            let spatial = stack.spatial_strides[index];
            let downsample = if temporal * spatial > 1 {
                Some(convolution(
                    loader,
                    &format!("{stage}.downsamplers.0.conv"),
                    *out,
                    *out,
                    3,
                    [temporal, spatial, spatial],
                )?)
            } else {
                None
            };
            levels.push(Level { blocks, downsample });
            width = *out;
        }
        let output = normalized(
            loader,
            "encoder.norm_out",
            "encoder.conv_out",
            width,
            2 * stack.latent_channels,
            3,
        )?;
        let moments = convolution(
            loader,
            "quant_conv",
            2 * stack.latent_channels,
            2 * stack.latent_channels,
            1,
            [1; 3],
        )?;
        // The latent space's affine: the checkpoint states its statistics, not a tensor for them.
        let latent = u64::from(stack.latent_channels);
        let (scale, shift) = normalize_tables(&config.latents_mean, &config.latents_std)?;
        let normalize = Conv3d::new(
            constants(
                loader,
                "video_vae.normalize.weight",
                Shape::new(&[latent, latent, 1, 1, 1]),
                &scale,
            )?,
            Some(constants(
                loader,
                "video_vae.normalize.bias",
                Shape::new(&[latent]),
                &shift,
            )?),
            geometry(1, [1; 3])?,
            true,
        )?;
        Ok(Self {
            stack,
            input,
            levels,
            output,
            moments,
            normalize,
            sampler: Posterior::new(),
            reflect,
        })
    }
}
impl Normalized {
    /// Normalize each frame, activate it, pad the result and convolve it.
    ///
    /// The padding is one pixel on each spatial side, mirrored or replicated, and two zero frames in
    /// front, so the convolution's own geometry carries none and no output frame reads a later input
    /// frame.
    fn run(&self, f: &Forward, values: TensorView, out: TensorView) -> Result<()> {
        let (batch, _, frames, height, width) = extent(values)?;
        let mean = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, u64::from(self.groups)]))?;
        let rstd = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, u64::from(self.groups)]))?;
        FrameMoments::new(u64::from(self.groups), self.epsilon).forward(f, values, mean, rstd)?;
        let padded = f.arena().tensor(
            DType::F32,
            Shape::new(&[
                batch,
                u64::from(self.channels),
                frames + 2,
                height + 2,
                width + 2,
            ]),
        )?;
        self.norm.forward(f, values, mean, rstd, padded, 1, 1, 2)?;
        Ok(self.conv.forward(f, padded, out)?)
    }
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

/// A five-axis tensor as the two-axis view an elementwise operator reads.
fn rows(view: TensorView) -> Result<TensorView> {
    let columns = view.shape.dim(4);
    if columns == 0 {
        return Err(Error::config("a tensor of no column".to_owned()));
    }
    let rows = view.shape.numel() / columns;
    Ok(view.reshape(Shape::new(&[rows, columns])))
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

    /// How many distinct tile extents the scratch covers.
    pub fn sets(&self) -> usize {
        self.sets.len()
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
}

/// One level's buffers: the pair its blocks alternate between, one working buffer the block reads back,
/// its shortcut projection where the first block widens, and the pad and strided output where the level
/// downsamples.
///
/// The pair is the level's **own** width, not the buffer the level was handed. Alternating with the
/// incoming buffer instead reads a level's second block against its input width whenever the level widens
/// it - which three of the six levels do - so a two-block level would write a wider value into a narrower
/// view.
struct LevelSet {
    first: TensorView,
    second: TensorView,
    working: TensorView,
    projected: Option<TensorView>,
    pad: Option<TensorView>,
    strided: Option<TensorView>,
}

impl VideoEncoder {

    /// The programs the video encoder launches, from the configuration alone and one list of rasters.
    ///
    /// **The raster is load-bearing here, unlike everywhere else in this model**: three of the
    /// encoder's operators key on classes derived from a frame's extent — the statistics' reduction
    /// (`channels/groups · pixels`, padded to a power of two), the fused normalize-pad's same class
    /// with its layout and affine flags, and each block's residual add over its level's whole extent.
    /// So a walk states a raster and gets the rows that raster launches.
    ///
    /// Every convolution runs FP32 with TF32 allowed, the design's declared row for this VAE, so there
    /// is no dtype argument.
    pub fn programs(
        config: &crate::VideoVaeConfig,
        rasters: &[(u32, u32)],
    ) -> Result<Vec<(String, String)>> {
        use crate::programs::Programs;
        use ops::conv::Conv3d as Conv3dOp;
        use ops::frame::{
            FrameMoments as FrameMomentsOp, FrameNormPad as FrameNormPadOp, FramePad as FramePadOp,
        };
        use ops::posterior::Posterior as PosteriorOp;
        use ops::residual::ScaledAdd as ScaledAddOp;

        let stack = Stack::of(config)?;
        let groups = u64::from(config.norm_num_groups);
        let epsilon = config.norm_eps;
        let dtype = DType::F32;
        let mut programs = Programs::default();

        // Every convolution of the network — the input, each block's two, each level's downsampler, the
        // output and the two of the posterior head — takes the same key: element types and a bias.
        programs.launch::<Conv3dOp>(&Conv3d::descriptor_of(
            dtype,
            dtype,
            true,
            dtype,
            crate::audio_vae::geometry(3, 1, 1)?,
            true,
        ));
        // The head's sample: the moments and the noise, one operator.
        programs.launch::<PosteriorOp>(&Posterior::new().descriptor());

        let reflect = matches!(
            config.spatial_padding_mode,
            crate::config::SpatialPadding::Reflect
        );
        // A pad's block split is its output's column count, so each extent's pad is its own row.
        let pad = |programs: &mut Programs, columns: u64| {
            let (block_pixels, block_channels) =
                ops::frame::FramePadDescriptor::tiles_of(true, true, columns);
            programs.launch::<FramePadOp>(&FramePad::descriptor_of(
                dtype,
                reflect,
                block_pixels,
                block_channels,
            ));
        };

        for (height, width) in rasters {
            // The encoder reads one still or one clip, and the clip's length is the configuration's.
            for frames in [1, stack.clip.frames] {
            let extents = stack.extents(Extent {
                frames,
                height: *height,
                width: *width,
            });
            // The input convolution reads padded pixels and normalizes nothing of its own.
            pad(&mut programs, u64::from(*width) + 2);
            // Each level's blocks run at the level's incoming extent and its incoming width.
            for (index, out) in stack.widths.iter().enumerate() {
                let extent = extents[index];
                let channels = if index == 0 {
                    stack.input_channels
                } else {
                    stack.widths[index - 1]
                };
                let pixels = u64::from(extent.height) * u64::from(extent.width);
                let padded = (u64::from(extent.height) + 2) * (u64::from(extent.width) + 2);
                // The level's downsampler reads a padded frame of its own.
                pad(&mut programs, u64::from(extent.width) + 2);
                // Each block takes the statistics of its input and then the fused normalize-pad of the
                // same frame; the classes follow from the frame and are deduplicated by `finish`.
                for _ in 0..stack.blocks {
                    programs.launch::<FrameMomentsOp>(&FrameMoments::descriptor_of(
                        dtype,
                        u64::from(channels),
                        pixels,
                        groups,
                        epsilon,
                    ));
                    programs.launch::<FrameNormPadOp>(&FrameNormPad::descriptor_of(
                        dtype,
                        u64::from(channels),
                        pixels,
                        groups,
                        false,
                        true,
                        padded,
                    ));
                    // The block's residual add covers the whole level: `frames · extent · channels`.
                    let columns = u64::from(extent.frames) * pixels * u64::from(*out);
                    if let Ok(descriptor) = ScaledAdd.descriptor(dtype, dtype, columns) {
                        programs.launch::<ScaledAddOp>(&descriptor);
                    }
                }
            }
            // The output normalization convolves the last level's extent at the last width.
            let last = extents[stack.widths.len()];
            let channels = *stack.widths.last().unwrap_or(&stack.input_channels);
            let pixels = u64::from(last.height) * u64::from(last.width);
            programs.launch::<FrameMomentsOp>(&FrameMoments::descriptor_of(
                dtype,
                u64::from(channels),
                pixels,
                groups,
                epsilon,
            ));
            programs.launch::<FrameNormPadOp>(&FrameNormPad::descriptor_of(
                dtype,
                u64::from(channels),
                pixels,
                groups,
                false,
                true,
                (u64::from(last.height) + 2) * (u64::from(last.width) + 2),
            ));
            }
        }

        Ok(programs.finish())
    }

    /// Encode one raster into the posterior moments of its latent.
    ///
    /// One tile, with its buffers reserved for the call: a driver of several tiles reserves once with
    /// [`Self::reserve`] and calls [`Self::encode_tile`] instead, or the arena carries one tile's buffers
    /// per tile.
    pub fn encode(&self, f: &Forward, pixels: TensorView, out: TensorView) -> Result<()> {
        let (_, _, frames, height, width) = extent(pixels)?;
        let raster = Extent {
            frames: u32::try_from(frames)
                .map_err(|_| Error::config("too many frames".to_owned()))?,
            height: u32::try_from(height).map_err(|_| Error::config("too tall".to_owned()))?,
            width: u32::try_from(width).map_err(|_| Error::config("too wide".to_owned()))?,
        };
        let scratch = self.reserve(f, &[raster])?;
        self.encode_tile(f, &scratch, pixels, out)
    }

    /// Reserve the buffers one run of tiles encodes in, one set per distinct tile extent.
    ///
    /// The arena never rewinds, so a driver that encodes a grid of tiles must reserve once and hand the
    /// same scratch to every tile: a reservation inside the loop is multiplied by the tile count, and one
    /// tile's buffers are most of a gigabyte at the served canvas. [`video_encoder_tile_arena`] states
    /// what one set holds.
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
        let padded = f.arena().tensor(
            DType::F32,
            Shape::new(&[
                1,
                u64::from(self.stack.input_channels),
                u64::from(raster.frames + 2),
                u64::from(raster.height + 2),
                u64::from(raster.width + 2),
            ]),
        )?;
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
                working: buffer(width, extent)?,
                // The first block of a level widens from the previous level's width; later blocks add to
                // their own width and project nothing.
                projected: match self.levels[index]
                    .blocks
                    .first()
                    .map(|block| block.shortcut.is_some())
                {
                    Some(true) => Some(buffer(width, extent)?),
                    _ => None,
                },
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
        Ok(EncoderSet {
            padded,
            stream,
            levels,
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
        if frames != 1 && frames != u64::from(self.stack.clip.frames) {
            return Err(Error::config(format!(
                "a video encoder reads one still (one frame) or one {}-frame clip, not {frames}",
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
        FramePad::new(self.reflect).forward(f, pixels, set.padded, 1, 1, 2)?;
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
                self.block(f, block, source, target, buffers.working, buffers.projected)?;
                source = target;
            }
            if let Some(downsample) = &level.downsample {
                let (Some(pad), Some(strided)) = (buffers.pad, buffers.strided) else {
                    return Err(Error::config(format!(
                        "level {index} downsamples without the buffers for it"
                    )));
                };
                FramePad::new(self.reflect).forward(f, source, pad, 0, 0, 2)?;
                downsample.forward(f, pad, strided)?;
                source = strided;
            }
        }

        // The moment convolution writes `2 · latent_channels` into the caller's view. The one-by-one
        // projection that follows it runs **once for the whole raster**, after the tiles are stitched
        // (`project`), because that is the reference's order: the tiles produce moments, the tiler
        // blends them, and the projection reads the stitched result.
        self.output.run(f, source, out)
    }

    /// The latent frames a **condition** of `frames` frames encodes to.
    ///
    /// A condition is cut into clips of [`Stack::clip_frames`] frames, each yielding
    /// [`Stack::clip_latents`] latent frames, and the last clip's trailing [`Stack::token_drop`] latent
    /// frames are dropped - so 17 frames are 2 latent frames and 22 are 7. A single frame is a still and
    /// yields one (`encoding.py:104-111`; the network's causal front padding makes one frame one latent
    /// frame, `video_vae.py:1105-1113`).
    ///
    /// This is a **different statement** from the generated timeline's `packing::video_latent_frames`
    /// (`17n + 5` frames to `5n + 2` latents). The two agree on the counts a generation uses and this
    /// one alone accepts the counts a condition may be - 17, 30, any length - because a condition is
    /// cut and padded, not generated.
    pub fn latent_frames(&self, frames: u32) -> Result<u32> {
        self.stack.clip.latent_frames(frames)
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
        self.moments.forward(f, moments, out).map_err(Error::from)
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
        let sample = Sample::of(self.stack.latent_channels, moments, noise, out)?;
        let positions = sample.positions;
        let channels = sample.channels;
        // A channel-major raster: neighbouring positions are neighbouring elements, and one channel is
        // one whole plane away. The two halves of the moments differ by exactly `channels` planes.
        let strides = Strides::new(&[1, positions]);
        let view = |ptr| {
            TensorView::with_strides(ptr, DType::F32, Shape::new(&[positions, channels]), strides)
        };
        let drawn = f.arena().tensor(DType::F32, out.shape)?;
        self.sampler.forward(
            f,
            view(moments.ptr),
            view(moments.offset_elements(channels * positions)),
            view(noise.ptr),
            view(drawn.ptr),
        )?;
        self.normalize.forward(f, drawn, out).map_err(Error::from)
    }

    /// One residual block: two normalized convolutions, added to the input or its projection.
    fn block(
        &self,
        f: &Forward,
        block: &Block,
        values: TensorView,
        out: TensorView,
        working: TensorView,
        projected: Option<TensorView>,
    ) -> Result<()> {
        block.first.run(f, values, working)?;
        block.second.run(f, working, out)?;
        let residual = match (&block.shortcut, projected) {
            (Some(shortcut), Some(projected)) => {
                shortcut.forward(f, values, projected)?;
                projected
            }
            (Some(_), None) => {
                return Err(Error::config(
                    "a block that widens its input has no buffer to project it into".to_owned(),
                ));
            }
            (None, _) => values,
        };
        // The reference groups the sum as `residual + (values + bias)` and adds the convolution's
        // bias in a pass of its own; cuDNN adds that bias in a pass of its own too, and IEEE addition
        // commutes, so `(values + bias) + residual` here is the same rounded value
        // (`video_vae.py:852-859`, `:1009-1020`).
        Ok(ScaledAdd.forward(f, rows(out)?, rows(residual)?, 1.0, 1.0)?)
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
/// decoder inverts. The network is the reference's `Encoder`: an input convolution of width
/// `encoder_dim`, five stages that each run three dilated residual units at the incoming width and then
/// a strided widening convolution, a Snake and a convolution to `latent_dim`, and then the attention
/// projection that every frame of the track attends over its predecessors.
///
/// **The projection's attention is causal** (`DenseInput(causal=True)` in the reference), which is what
/// makes a latent frame depend only on the audio before it. It runs in FP32 like everything else in this
/// encoder, on the deployment's longest track, which `load` takes as the attention's key bound.
pub struct AudioEncoder {
    config: crate::AudioVaeConfig,
    /// The most latent frames one track may have: the attention's key bound.
    frames: u32,
    input: Conv1d,
    stages: Vec<EncoderStage>,
    tail: Snake,
    tail_alpha: Weight,
    output: Conv1d,
    projection: Projection,
    mean: Conv1d,
    transpose: Transpose,
}

/// One encoder stage: three dilated residual units at `inner`, then a widening strided convolution.
struct EncoderStage {
    /// The width the units and the Snake run at: half the stage's output width.
    inner: u32,
    /// Latent samples divide by this.
    rate: u32,
    units: Vec<EncoderUnit>,
    snake: Snake,
    alpha: Weight,
    widen: Conv1d,
}

/// One residual unit: snake, dilated convolution, snake, one-by-one-wide convolution, added back.
struct EncoderUnit {
    convs: [Conv1d; 2],
    snakes: [Snake; 2],
    alphas: [Weight; 2],
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
    attention: Attention,
    attention_output: Linear,
    /// The head average and the head width pooled down to the latent channels, in one pass.
    pool: AdaptivePool,
    mlp_norm: LayerNorm,
    mlp: GeGlu,
}

/// A normalized tanh-GELU gated linear unit: the reference's `GeGLU`.
struct GeGlu {
    norm: LayerNorm,
    gate: Linear,
    value: Linear,
    output: Linear,
    /// `gelu_tanh(gate) · value`, the reference's `F.gelu(gate, approximate="tanh") * value`.
    product: GatedActivation,
}

impl AudioEncoder {
    /// Every checkpoint tensor the audio encoder reads, by name.
    ///
    /// Built from [`audio_sources`], the table the load drives, so the list cannot drift from what is
    /// bound: a weight-normalized convolution names its direction and its magnitude, and the fused
    /// attention projections name every part. The export this tree was checked against ships
    /// `audio_vae/config.json` and no tensor file, so against that checkpoint the honest verdict is
    /// "all absent" - a refusal naming the block, not a read error from inside a layer.
    pub fn keys(config: &crate::AudioVaeConfig) -> Result<Vec<String>> {
        Ok(crate::audio_vae::tensor_keys(&audio_sources(config)?))
    }

    /// The programs the audio encoder launches, from the configuration and the deployment's longest
    /// track.
    ///
    /// **FP32 throughout and no dtype argument**: the encoder's convolutions, its Snake activations and
    /// its whole attention are the design's declared FP32 rows, not the checkpoint's representation.
    /// `samples` states the residual units' reach: each unit's output is one operator, so its width is
    /// `inner · samples` and a longer track is a different program.
    pub fn programs(
        config: &crate::AudioVaeConfig,
        samples: &[u32],
    ) -> Result<Vec<(String, String)>> {
        use crate::programs::Programs;
        use ops::activation::ActAndMul as ActAndMulOp;
        use ops::attention::dense::DenseAttention;
        use ops::conv::Conv1d as Conv1dOp;
        use ops::gemm::Gemm;
        use ops::norm::LayerNorm as LayerNormOp;
        use ops::adaptive_pool::AdaptivePool as PoolOp;
        use ops::residual::ScaledAdd as ScaledAddOp;
        use ops::snake::Snake as SnakeOp;
        use ops::transpose::Transpose as TransposeOp;

        let dtype = DType::F32;
        // The widths the encoder runs at: `encoder_dim`, doubled per stage, and the final width that
        // the tail Snake and the output convolution read.
        let mut widths = Vec::new();
        let mut width = config.encoder_dim;
        for _ in &config.encoder_rates {
            widths.push(width);
            width = width
                .checked_mul(2)
                .ok_or_else(|| Error::config("the encoder's widths overflow".to_owned()))?;
        }
        widths.push(width);
        widths.sort_unstable();
        widths.dedup();
        let channels = config.latent_channels;
        let mut programs = Programs::default();

        // Every convolution of the encoder: the input, each unit's two, each stage's widening one and
        // the output. The window, the stride, the dilation and the bias travel with the call.
        programs.launch::<Conv1dOp>(&Conv1d::descriptor_of(
            dtype,
            dtype,
            true,
            dtype,
            crate::audio_vae::geometry(7, 1, 3)?,
            false,
        ));
        // One Snake per channel width it runs at.
        for channels in &widths {
            programs.launch::<SnakeOp>(&Snake::descriptor_of(dtype, u64::from(*channels)));
        }

        // The projection: three layer normalizations, four dense products, the attention, the pool and
        // the transpose that hands the latent features to the moments convolution.
        for columns in [width, channels] {
            programs.launch::<LayerNormOp>(&LayerNorm::descriptor_of(dtype, columns, true));
        }
        programs.launch::<Gemm>(&Linear::descriptor_of(
            channels,
            dtype,
            WeightOrder::NxK,
            true,
            dtype,
            dtype,
        ));
        let head_dim = (width / config.num_attention_heads.max(1)).max(1);
        let attention = Attention::dense(
            head_dim,
            config.num_attention_heads,
            config.num_attention_heads,
            LONGEST_TRACK,
            false,
            None,
            1.0 / (head_dim as f32).sqrt(),
        )?;
        // The audio encoder's attention is the **unpacked** FP32 form: one track, no offsets table
        // (`encoding.rs`'s launch passes none). The `f32` package serves exactly that key; the flash
        // backend serves the packed form only, and no package serves f16 unpacked — which is why this
        // row names `cu_seq_lens_false` and must.
        programs.launch::<DenseAttention>(&attention.dense_descriptor(
            dtype, dtype, dtype, LONGEST_TRACK, false,
        ));
        programs.launch::<PoolOp>(
            &AdaptivePool::new(
                u64::from(config.num_attention_heads),
                u64::from(width / config.num_attention_heads.max(1)),
                u64::from(channels),
            )?
            .descriptor(dtype),
        );
        programs.launch::<TransposeOp>(&Transpose::new(dtype)?.descriptor());

        // The GeGLU's gated product: the reference's normalized tanh-GELU gate.
        programs.launch::<ActAndMulOp>(&GatedActivation::descriptor_of(
            dtype,
            u64::from(2 * channels),
            Activation::GeluTanh,
            u64::from(2 * channels),
            u64::from(2 * channels),
            u64::from(2 * channels),
            false,
        )?);

        // Each residual unit's output is one addition over the whole unit, so its width is the inner
        // channel count times the track's samples.
        for samples in samples {
            for inner in &widths {
                let columns = u64::from(*inner) * u64::from(*samples);
                if let Ok(descriptor) = ScaledAdd.descriptor(dtype, dtype, columns) {
                    programs.launch::<ScaledAddOp>(&descriptor);
                }
            }
        }

        Ok(programs.finish())
    }

    /// The latent frames `samples` samples encode to: `ceil(samples / hop)`.
    ///
    /// This is the reference's rule (`encoding.py:244`) and the reason it is a ceiling is the pad its
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
        self.stages
            .iter()
            .map(|stage| u64::from(stage.rate))
            .product()
    }

    /// Bind the encoder every checkpoint field it has, through [`audio_sources`].
    ///
    /// The attention is built for [`LONGEST_TRACK`], the longest track the model supports; a longer
    /// track is refused by [`Self::frames`] rather than read past the bound. The table is consumed
    /// exactly, the same discipline the audio decoder's load keeps.
    pub fn load(config: &crate::AudioVaeConfig, loader: &mut WeightLoader) -> Result<Self> {
        let frames = LONGEST_TRACK;
        let table = audio_sources(config)?;
        let mut binder = Binder::new(loader, table);
        let mut width = config.encoder_dim;
        let input = Conv1d::new(
            binder.normalized("input.weight")?,
            Some(binder.region("input.bias", &[u64::from(width)])?),
            crate::audio_vae::geometry(7, 1, 3)?,
            false,
        )?;

        let mut stages = Vec::with_capacity(config.encoder_rates.len());
        for (index, rate) in config.encoder_rates.iter().enumerate() {
            let inner = width;
            width = width.checked_mul(2).ok_or_else(|| {
                Error::config("the encoder's widths overflow a descriptor".to_owned())
            })?;
            let at = |name: &str| format!("stages.{index}.{name}");
            let mut units = Vec::with_capacity(3);
            // The three units are the same shape at dilations 1, 3 and 9; the first convolution is the
            // dilated one, whose padding keeps the length, and the second returns to it.
            for (unit, dilation) in [1u32, 3, 9].into_iter().enumerate() {
                let name = |slot: &str| at(&format!("units.{unit}.{slot}"));
                let first = Conv1d::new(
                    binder.normalized(&name("convs.0.weight"))?,
                    Some(binder.region(&name("convs.0.bias"), &[u64::from(inner)])?),
                    crate::audio_vae::dilated_geometry(7, 1, 3 * dilation, dilation)?,
                    false,
                )?;
                let second = Conv1d::new(
                    binder.normalized(&name("convs.1.weight"))?,
                    Some(binder.region(&name("convs.1.bias"), &[u64::from(inner)])?),
                    crate::audio_vae::geometry(7, 1, 3)?,
                    false,
                )?;
                units.push(EncoderUnit {
                    convs: [first, second],
                    snakes: [Snake::new(u64::from(inner))?, Snake::new(u64::from(inner))?],
                    alphas: [
                        binder.region(&name("snakes.0.alpha"), &[u64::from(inner)])?,
                        binder.region(&name("snakes.1.alpha"), &[u64::from(inner)])?,
                    ],
                });
            }
            let snake = Snake::new(u64::from(inner))?;
            let alpha = binder.region(&at("snake.alpha"), &[u64::from(inner)])?;
            let widen = Conv1d::new(
                binder.normalized(&at("widen.weight"))?,
                Some(binder.region(&at("widen.bias"), &[u64::from(width)])?),
                crate::audio_vae::geometry(2 * rate, *rate, rate.div_ceil(2))?,
                false,
            )?;
            stages.push(EncoderStage {
                inner,
                rate: *rate,
                units,
                snake,
                alpha,
                widen,
            });
        }

        let tail = Snake::new(u64::from(width))?;
        let tail_alpha = binder.region("output_snake.alpha", &[u64::from(width)])?;
        // The network's last convolution is three wide, not seven: the reference's `Conv1d(width,
        // latent_dim, 3, padding=1)`.
        let output = Conv1d::new(
            binder.normalized("output.weight")?,
            Some(binder.region("output.bias", &[u64::from(config.latent_dim)])?),
            crate::audio_vae::geometry(3, 1, 1)?,
            false,
        )?;

        let projection = Projection::load(config, &mut binder, frames)?;
        let channels = u64::from(config.latent_channels);
        let mean = Conv1d::new(
            binder.region("mean_projection.weight", &[channels, channels])?,
            Some(binder.region("mean_projection.bias", &[channels])?),
            crate::audio_vae::geometry(1, 1, 0)?,
            false,
        )?;
        binder.finish()?;
        Ok(Self {
            config: config.clone(),
            frames,
            input,
            stages,
            tail,
            tail_alpha,
            output,
            projection,
            mean,
            transpose: Transpose::new(DType::F32)?,
        })
    }

    /// Encode a `[batch, 1, length]` track into `out [batch · frames, latent_channels]` F32 rows.
    ///
    /// The reference's encoder is mono per batch row (`audio_vae.py:448-452`, "encode mono
    /// `[batch, 1, samples]` waveforms") and its latent is a **two-row batch**: a stereo reference
    /// arrives as `[2, 1, samples]` and its latent is `[2, latent_channels, frames]`, the same two-row
    /// shape the audio decoder reads. The rows are written channel-major, `[batch · frames,
    /// latent_channels]` with batch outermost, because that is the form the DiT's audio conditioning
    /// reads (`denoiser.py:518`, `[2 · frames, audio_channels]`) - so no caller transposes.
    ///
    /// The length must be a whole number of hops and no longer than the deployment's bound. The
    /// activation of every stage runs at the stage's inner width, which is half of the width the
    /// widening convolution writes. The projection attends over frame 2, so it runs once per batch row
    /// and never mixes them.
    pub fn encode(&self, f: &Forward, samples: TensorView, out: TensorView) -> Result<()> {
        let shape = samples.shape;
        if shape.rank() != 3 || shape.dim(0) == 0 || shape.dim(1) != 1 {
            return Err(Error::config(format!(
                "the audio encoder reads a [batch, 1, samples] track, not {:?}",
                shape.dims()
            )));
        }
        let batch = shape.dim(0);
        let length = shape.dim(2);
        let frames = self.frames(length)?;
        // The network's stages divide the length exactly, so a track that is not a whole number of
        // latent frames is right-padded with zeros to one, which is what the reference's
        // `unpack_samples` does before it convolves. Writing that pad needs a zero-fill the operator set
        // does not have yet (a `zero_pad` over the trailing axis); until it exists this is refused by
        // name rather than truncated, because truncating drops the last latent frame.
        let hop = self.hop();
        if !length.is_multiple_of(hop) {
            let padded = padded_samples(length, hop)?;
            return Err(Error::config(format!(
                "{length} samples is not a whole number of {hop}-sample latent frames: the reference \
                 right-pads the track with zeros to {padded} samples and this build has no zero-fill \
                 program"
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

        let width = u64::from(self.config.encoder_dim);
        let mut hidden = f
            .arena()
            .tensor(DType::F32, Shape::new(&[batch, width, length]))?;
        self.input.forward(f, samples, hidden)?;

        // A stage runs its three units at the stage's inner width, which is the width it receives, then
        // snaked and widened into the next width. Each unit needs two buffers - the two its convolutions
        // write, because the second snake reads what the first convolution wrote - and they are the
        // stage's shape, so the stage reserves them once and the three units take turns: a unit writes
        // `first` from the residual stream before reading it, and `second` from its own convolution. The
        // residual stream itself is `hidden`, which no unit writes.
        let mut extent = length;
        for stage in &self.stages {
            let inner = u64::from(stage.inner);
            let inner_shape = Shape::new(&[batch, inner, extent]);
            let first = f.arena().tensor(DType::F32, inner_shape)?;
            let second = f.arena().tensor(DType::F32, inner_shape)?;
            for unit in &stage.units {
                unit.snakes[0].forward(f, hidden, unit.alphas[0].view(), first)?;
                unit.convs[0].forward(f, first, second)?;
                unit.snakes[1].forward(f, second, unit.alphas[1].view(), first)?;
                unit.convs[1].forward(f, first, second)?;
                let residual = matrix(hidden)?;
                ScaledAdd.forward(f, residual, matrix(second)?, 1.0, 1.0)?;
            }
            let widened = extent / u64::from(stage.rate);
            let next = f.arena().tensor(
                DType::F32,
                Shape::new(&[batch, u64::from(self.width_of(stage)), widened]),
            )?;
            let scratch = f.arena().tensor(DType::F32, inner_shape)?;
            stage
                .snake
                .forward(f, hidden, stage.alpha.view(), scratch)?;
            stage.widen.forward(f, scratch, next)?;
            hidden = next;
            extent = widened;
        }

        // The tail Snake and the convolution to the latent width.
        let wide = u64::from(self.config.latent_dim);
        let tail_shape = Shape::new(&[batch, wide, extent]);
        let tail_scratch = f.arena().tensor(DType::F32, tail_shape)?;
        self.tail
            .forward(f, hidden, self.tail_alpha.view(), tail_scratch)?;
        let features = f.arena().tensor(DType::F32, tail_shape)?;
        self.output.forward(f, tail_scratch, features)?;

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
            // One row's latent, channel-major rows: `[extent, channels]` at the row's place in `out`.
            let plane = TensorView::new(narrowed.ptr, DType::F32, Shape::new(&[channels, extent]));
            let written = TensorView::new(
                out.ptr.offset(row * extent * channels * 4),
                DType::F32,
                Shape::new(&[extent, channels]),
            );
            self.transpose.forward(f, plane, written)?;
        }
        Ok(())
    }

    /// The width a stage writes: its inner width doubled.
    fn width_of(&self, stage: &EncoderStage) -> u32 {
        stage.inner * 2
    }
}

/// Every buffer **one tile's** video encode takes from `Forward::arena()`, in reservation order.
///
/// `VideoEncoder::encode` reads one tile, so this is one call's list: the padded pixels, the input
/// convolution's output, then per level the alternating pair its blocks swap (one buffer for the level's
/// width, plus the block's own working buffer and the shortcut's projection), the pad and the strided
/// output where the level downsamples, and the moments convolution's output.
///
/// **A caller that encodes several tiles must hoist these buffers out of its loop.** The arena never
/// rewinds - its only rewind is the executor's per-forward `reset` - so a driver that calls `encode` once
/// per tile multiplies this whole list by the tile count. The level-0 buffers are the pixels at the
/// encoder's first width, so at a 256-by-64-pixel tile they are ~143 MB each: four of them is most of a
/// gigabyte for one tile, and a tiled reference clip is tens of tiles. This function states one tile so
/// The conditioning pass's own reservations for one visual condition.
///
/// A condition's sizes are the **condition's**, not a request's: the pass runs once per condition and
/// the encoder's own set is reserved beside it, so nothing here multiplies by a request's clip or frame
/// count. Three conditions in one request are three sequential uses of one plan, not three plans.
///
/// `latent` is the condition's **latent** extent - the encoder's own rule decided the frame count
/// (`VideoEncoder::latent_frames`), so a still is one frame. The six buffers, in the order the pass
/// reserves them:
///
/// | name | extent | why |
/// | --- | --- | --- |
/// | `tile` | `in_channels · frames · tile_height · tile_width` | **one** buffer, refilled per tile: the tiles are uniform, so a per-tile buffer would grow with the grid and buy nothing. The extents are the plan's largest tile, taken from the same `Tiles` the decoder's grid uses |
/// | `stitch` | `2 · latent_channels · frames · rows · columns` | the moments' two halves, stitched over the grid |
/// | `moments` | the same | the projection the posterior reads |
/// | `sample` | `latent_channels · frames · rows · columns` | the posterior's draw |
/// | `clean` | `patch_rows · latent_channels · 4` | the condition's latent patchified into rows |
/// | `anchored` | the same | those rows at the condition's timestep |
///
/// `patch_rows` is the latent's voxels divided by four: the condition's rows are `2 x 2` latent
/// patches, so `latent_channels · 4` columns, and a 84 x 48 latent gives 42 x 24 of them a frame.
pub fn conditioning_arena(
    config: &crate::VideoVaeConfig,
    frames: u32,
    latent: Extent,
) -> Result<Vec<crate::decoding::Buffer>> {
    let stack = Stack::of(config)?;
    // The tile buffer is in **pixels**, so its frame axis is the condition's own, while every other
    // buffer is in latents. Two frame counts in one plan is exactly the confusion this check removes.
    let expected = stack.clip.latent_frames(frames)?;
    if expected != latent.frames {
        return Err(Error::config(format!(
            "{frames} frames of condition are {expected} latent frames, and the plan was given {}",
            latent.frames
        )));
    }
    // The tile is the **encoder's** own grid, not the decoder's: this buffer is the encoder's input, so
    // the plan reads the enumeration the encoder's driver reads. The two are one rule today - the
    // checkpoint's 256-pixel tiles with a 64-pixel minimum overlap - and taking it from this side is
    // what keeps the plan right if either ever moves.
    let ratio = config.spatial_compression();
    let grid = Grid::of(latent.height * ratio, latent.width * ratio)?;
    let tile = (
        grid.rows.lengths.iter().copied().max().unwrap_or(0),
        grid.columns.lengths.iter().copied().max().unwrap_or(0),
    );
    if tile.0 == 0 || tile.1 == 0 {
        return Err(Error::config(
            "a condition with no tile has no plan".to_owned(),
        ));
    }
    if !latent.height.is_multiple_of(2) || !latent.width.is_multiple_of(2) {
        return Err(Error::config(format!(
            "a condition's rows are 2 x 2 latent patches, and {} x {} has an odd axis",
            latent.height, latent.width
        )));
    }
    let latent_frames = u64::from(latent.frames);
    let rows = u64::from(latent.height);
    let columns = u64::from(latent.width);
    let voxels = latent_frames * rows * columns;
    let channels = u64::from(config.latent_channels);
    let patch_rows = voxels / 4;
    let buffer = |name: &str, elements: u64| crate::decoding::Buffer {
        name: name.to_owned(),
        element: "f32",
        elements,
    };
    Ok(vec![
        buffer(
            "tile",
            u64::from(stack.input_channels)
                * u64::from(frames)
                * u64::from(tile.0)
                * u64::from(tile.1),
        ),
        buffer("stitch", 2 * channels * voxels),
        buffer("moments", 2 * channels * voxels),
        buffer("sample", channels * voxels),
        // The posterior's own draw, which the encoder samples under its module seed
        // (`denoiser::POSTERIOR_SEED`), at the latent's shape.
        buffer("posterior", channels * voxels),
        buffer("clean", patch_rows * channels * 4),
        buffer("anchored", patch_rows * channels * 4),
    ])
}

/// the multiplication is visible rather than hidden inside a driver.
pub fn video_encoder_tile_arena(
    config: &VideoVaeConfig,
    raster: Extent,
) -> Result<Vec<crate::decoding::Buffer>> {
    let stack = Stack::of(config)?;
    let extents = stack.extents(raster);
    let mut plan = Vec::new();
    let mut push = |name: String, elements: u64| {
        plan.push(crate::decoding::Buffer {
            name,
            element: "f32",
            elements,
        });
    };
    let pixels = |extent: &Extent| {
        u64::from(extent.frames) * u64::from(extent.height) * u64::from(extent.width)
    };
    push(
        "padded".to_owned(),
        u64::from(stack.input_channels)
            * u64::from(raster.frames + 2)
            * u64::from(raster.height + 2)
            * u64::from(raster.width + 2),
    );
    for (index, level) in extents.iter().take(extents.len() - 1).enumerate() {
        let width = u64::from(stack.widths[index]);
        let input = match index {
            0 => u64::from(stack.widths[0]),
            _ => u64::from(stack.widths[index - 1]),
        };
        // The pair a level's blocks alternate between, and the working buffer each block reads back.
        push(format!("level{index}.first"), width * pixels(level));
        push(format!("level{index}.second"), width * pixels(level));
        push(format!("level{index}.working"), width * pixels(level));
        // Only the first block of a level widens its input, and only when the level does.
        if input != width {
            push(format!("level{index}.projected"), width * pixels(level));
        }
        let next = &extents[index + 1];
        if next.frames < level.frames || next.height < level.height || next.width < level.width {
            push(
                format!("level{index}.pad"),
                width
                    * u64::from(level.frames + 2)
                    * u64::from(level.height + 1)
                    * u64::from(level.width + 1),
            );
            push(format!("level{index}.strided"), width * pixels(next));
        }
    }
    Ok(plan)
}

/// Every buffer one audio encode takes from `Forward::arena()`, in reservation order.
///
/// The encoder's `encode` reserves the input convolution's output, then per stage the two buffers its
/// three units take turns in plus the snake's buffer and the widened one it passes on, then the tail
/// snake's buffer and the convolution's features. The projection that follows runs once per batch row
/// and reserves its own working set each time, which is why the rows appear separately here.
///
/// The list matters because the arena never rewinds, so everything accumulates for the length of the
/// encode. The dominant entries are the first stage's, which run at the *sample* rate: at 207 latent
/// frames a three-minute track is 165,600 samples, and one `[2, 64, 165600]` f32 buffer is 84.8 MB, so
/// the encoder's plan is several hundred MB even though its weights are 0.12 GiB.
pub fn audio_encoder_arena(
    config: &crate::AudioVaeConfig,
    batch: u64,
    samples: u64,
) -> Result<Vec<crate::decoding::Buffer>> {
    let mut plan = Vec::new();
    let mut push = |name: String, elements: u64| {
        plan.push(crate::decoding::Buffer {
            name,
            element: "f32",
            elements,
        });
    };
    let stack = AudioStack::of(config)?;
    let width = u64::from(stack.encoder_dim);
    push("input".to_owned(), batch * width * samples);
    let mut extent = samples;
    let mut inner = width;
    for (index, rate) in stack.rates.iter().enumerate() {
        let rate = u64::from(*rate);
        if rate == 0 || extent == 0 {
            return Err(Error::config(format!(
                "audio encoder stage {index} has rate {rate} at extent {extent}"
            )));
        }
        let widened = extent / rate;
        push(format!("stage{index}.first"), batch * inner * extent);
        push(format!("stage{index}.second"), batch * inner * extent);
        push(format!("stage{index}.snake"), batch * inner * extent);
        push(format!("stage{index}.next"), batch * 2 * inner * widened);
        extent = widened;
        inner *= 2;
    }
    let wide = u64::from(stack.latent_dim);
    let channels = u64::from(stack.latent_channels);
    push("tail".to_owned(), batch * wide * extent);
    push("features".to_owned(), batch * wide * extent);
    // The projection, per row: its transposed features, its four residual-shaped buffers, the attention's
    // three projections and its output, the gated unit's four, and the two views the row's latent is
    // written through.
    for row in 0..batch {
        let frames = extent;
        for (name, elements) in [
            ("transpose", frames * wide),
            ("shortcut", frames * channels),
            ("normed", frames * wide),
            ("attended", frames * wide),
            ("query", frames * wide),
            ("key", frames * wide),
            ("value", frames * wide),
            ("pooled", frames * channels),
            ("projected", frames * channels),
            ("mlp_normed", frames * channels),
            ("updated", frames * channels),
            ("gate_normed", frames * channels),
            ("gate", frames * 2 * channels),
            ("gated_value", frames * 2 * channels),
            ("product", frames * 2 * channels),
            ("pooled_matrix", channels * frames),
            ("narrowed", channels * frames),
        ] {
            push(format!("row{row}.{name}"), elements);
        }
    }
    Ok(plan)
}

/// A `[batch, rows, columns]` view as `[batch · rows, columns]`: the row axis the operators that work
/// a row at a time read.
///
/// Valid for a contiguous view, which every arena tensor and every slice of one is: the batch axis is
/// the outermost, so folding it into the row axis keeps every row where it is.
fn matrix(view: TensorView) -> Result<TensorView> {
    if view.shape.rank() != 3 || view.shape.dim(0) == 0 {
        return Err(Error::config(format!(
            "{:?} is not a batched matrix",
            view.shape.dims()
        )));
    }
    let rows = view.shape.dim(0) * view.shape.dim(1);
    Ok(TensorView::new(
        view.ptr,
        view.dtype,
        Shape::new(&[rows, view.shape.dim(2)]),
    ))
}

impl Projection {
    /// Bind the attention projection at the latent width.
    fn load(
        config: &crate::AudioVaeConfig,
        binder: &mut Binder<'_, '_>,
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
        let norm = |binder: &mut Binder<'_, '_>, name: &str| -> Result<LayerNorm> {
            let weight = binder.region(&format!("projection.{name}.weight"), &[wide])?;
            let bias = binder.region(&format!("projection.{name}.bias"), &[wide])?;
            Ok(LayerNorm::new(weight, Some(bias), NORM_EPS)?)
        };
        let shortcut_norm = norm(binder, "shortcut_norm")?;
        let attention_norm = norm(binder, "attention_norm")?;
        let shortcut = Linear::new(
            binder.region("projection.shortcut.weight", &[channels, wide])?,
            Some(binder.region("projection.shortcut.bias", &[channels])?),
            WeightOrder::NxK,
        )?;
        // The checkpoint fuses query, key and value into one `[3 · wide, wide]` matrix and one
        // `[3 · wide]` bias; each third is contiguous, so each head group is its own layer.
        let third = wide;
        let slice = |binder: &mut Binder<'_, '_>,
                     name: &str,
                     index: u64,
                     shape: &[u64]|
         -> Result<Weight> {
            binder.slice(name, index * shape.iter().product::<u64>(), shape)
        };
        let linear = |binder: &mut Binder<'_, '_>, index: u64| -> Result<Linear> {
            let weight = slice(binder, "projection.qkv.weight", index, &[third, wide])?;
            let bias = slice(binder, "projection.qkv.bias", index, &[third])?;
            Ok(Linear::new(weight, Some(bias), WeightOrder::NxK)?)
        };
        let query = linear(binder, 0)?;
        let key = linear(binder, 1)?;
        let value = linear(binder, 2)?;
        let head_dim = u32::try_from(wide / heads)
            .map_err(|_| Error::config("a head wider than a launch holds".to_owned()))?;
        let heads_u32 = u32::try_from(heads)
            .map_err(|_| Error::config("more heads than a launch holds".to_owned()))?;
        let attention = Attention::dense(
            head_dim,
            heads_u32,
            heads_u32,
            frames,
            true,
            None,
            1.0 / (head_dim as f32).sqrt(),
        )?;
        let attention_output = Linear::new(
            binder.region("projection.attention_output.weight", &[channels, channels])?,
            Some(binder.region("projection.attention_output.bias", &[channels])?),
            WeightOrder::NxK,
        )?;
        let pool = AdaptivePool::new(heads, wide / heads, channels)?;
        let mlp_norm = LayerNorm::new(
            binder.region("projection.mlp_norm.weight", &[channels])?,
            Some(binder.region("projection.mlp_norm.bias", &[channels])?),
            NORM_EPS,
        )?;
        let mlp = GeGlu::load(config, binder)?;
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
            mlp_norm,
            mlp,
        })
    }

    /// Project `values [frames, latent_dim]` to `[frames, latent_channels]`.
    fn forward(&self, f: &Forward, values: TensorView, frames: u64) -> Result<TensorView> {
        let shape = values.shape;
        let channels = u64::from(self.pool.output());
        let heads = shape.dim(1) / u64::from(self.attention.head_dim());
        let shortcut = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        let normed = f.arena().tensor(DType::F32, shape)?;
        self.shortcut_norm.forward(f, values, normed)?;
        self.shortcut.forward(f, normed, shortcut)?;
        self.attention_norm.forward(f, values, normed)?;
        // Three contiguous head-group blocks, then causal attention over the frames.
        let attended = f.arena().tensor(
            DType::F32,
            Shape::new(&[frames, heads, u64::from(self.attention.head_dim())]),
        )?;
        let mut head_views = Vec::with_capacity(3);
        for linear in [&self.query, &self.key, &self.value] {
            let out = f.arena().tensor(DType::F32, shape)?;
            linear.forward(f, normed, out)?;
            head_views.push(out);
        }
        self.attention.attend_dense(
            f,
            head_views[0],
            head_views[1],
            head_views[2],
            None,
            None,
            u32::try_from(frames)
                .map_err(|_| Error::config("a track longer than a launch bound".to_owned()))?,
            attended,
        )?;
        let pooled = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.pool.forward(f, attended, pooled)?;
        let projected = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.attention_output.forward(f, pooled, projected)?;
        ScaledAdd.forward(f, shortcut, projected, 1.0, 1.0)?;
        // The normalized GeGLU residual.
        let normed = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.mlp_norm.forward(f, shortcut, normed)?;
        let updated = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.mlp.forward(f, normed, updated)?;
        ScaledAdd.forward(f, shortcut, updated, 1.0, 1.0)?;
        Ok(shortcut)
    }
}

impl GeGlu {
    /// Bind a normalized tanh-GELU gated linear unit over the latent channels.
    fn load(config: &crate::AudioVaeConfig, binder: &mut Binder<'_, '_>) -> Result<Self> {
        let channels = u64::from(config.latent_channels);
        let hidden = 2 * channels;
        let norm = LayerNorm::new(
            binder.region("projection.mlp.norm.weight", &[channels])?,
            Some(binder.region("projection.mlp.norm.bias", &[channels])?),
            NORM_EPS,
        )?;
        let linear = |binder: &mut Binder<'_, '_>, name: &str, out: u64| -> Result<Linear> {
            Ok(Linear::new(
                binder.region(&format!("projection.mlp.{name}.weight"), &[out, channels])?,
                Some(binder.region(&format!("projection.mlp.{name}.bias"), &[out])?),
                WeightOrder::NxK,
            )?)
        };
        let gate = linear(binder, "gate", hidden)?;
        let value = linear(binder, "value", hidden)?;
        let output = Linear::new(
            binder.region("projection.mlp.output.weight", &[channels, hidden])?,
            Some(binder.region("projection.mlp.output.bias", &[channels])?),
            WeightOrder::NxK,
        )?;
        Ok(Self {
            norm,
            gate,
            value,
            output,
            product: GatedActivation::new(ops::activation::Activation::GeluTanh),
        })
    }

    /// `output(gelu_tanh(gate(norm(x))) · value(norm(x)))` into `out`.
    fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        let (frames, hidden) = (x.shape.dim(0), 2 * x.shape.dim(1));
        let normed = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, x.shape.dim(1)]))?;
        self.norm.forward(f, x, normed)?;
        let gate = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, hidden]))?;
        let value = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, hidden]))?;
        self.gate.forward(f, normed, gate)?;
        self.value.forward(f, normed, value)?;
        let product = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, hidden]))?;
        self.product.forward(f, gate, value, product)?;
        Ok(self.output.forward(f, product, out)?)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_audio_encoders_keys_are_the_encoders_half_and_name_every_part() {
        let config = crate::audio_vae::tests::audio_config();
        let keys = AudioEncoder::keys(&config).expect("the encoder's keys");
        let mut unique = keys.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), keys.len(), "no tensor is named twice");
        // The encoder's 171 names are 37 weight-normalized pairs and the rest single tensors; the
        // export this tree was checked against ships `audio_vae/config.json` and no tensor file, so
        // against it the honest verdict is "all absent" rather than a load that reads nothing.
        assert_eq!(keys.len(), 171);
        assert_eq!(
            keys.iter().filter(|key| key.ends_with(".weight_g")).count(),
            37
        );
        assert_eq!(
            keys.iter().filter(|key| key.ends_with(".weight_v")).count(),
            37
        );
        let decoder = crate::audio_vae::AudioDecoder::keys(&config).expect("the decoder's keys");
        assert!(
            keys.iter().all(|key| !decoder.contains(key)),
            "the two audio halves read disjoint tensors"
        );
    }


    /// The video encoder states its programs from its configuration and the rasters it serves: one
    /// convolution whatever the window or stride, the statistics and the fused normalize-pad of each
    /// frame, the posterior head's sample, and one residual per block at its level's own extent.
    ///
    /// **The raster reaches these rows through the residual and not through the frame operators**: a
    /// frame's reduction class is a declared bound in the descriptor and not part of the key, which is
    /// what lets one program serve every raster a deployment serves.
    #[test]
    fn the_video_encoder_states_its_programs_from_its_config_and_raster() {
        let config = crate::video_vae::tests::video_config();
        let raster = [(480, 832)];
        let programs = VideoEncoder::programs(&config, &raster).expect("the video encoder's programs");
        let count = |operator: &str| {
            programs
                .iter()
                .filter(|(name, _)| name == operator)
                .count()
        };
        assert_eq!(count("conv3d"), 1, "one program whatever the window or stride");
        assert_eq!(count("posterior"), 1);
        assert_eq!(
            count("frame_moments"),
            1,
            "the reduction is a bound, not a key: {programs:?}"
        );
        assert_eq!(count("frame_norm_pad"), 1);
        assert!(
            count("frame_pad") > 1,
            "each extent's pad splits its blocks by its own column count: {programs:?}"
        );
        assert!(
            count("scaled_add") > 1,
            "each level's residual is its own width: {programs:?}"
        );
        for (operator, key) in &programs {
            assert!(
                key.contains("f32") || operator == "posterior" || operator == "scaled_add",
                "{operator} {key} is not FP32"
            );
        }

        // A different raster is different residual programs and different pad classes — the pad's block
        // split is its output's column count — while the statistics, the normalize-pad and the network's
        // convolutions keep one row each.
        let other = VideoEncoder::programs(&config, &[(480, 448)]).expect("a narrower raster");
        assert_ne!(other, programs);
        let fixed = |rows: &[(String, String)]| {
            rows.iter()
                .filter(|(name, _)| name != "scaled_add" && name != "frame_pad")
                .cloned()
                .collect::<Vec<_>>()
        };
        assert_eq!(
            fixed(&other),
            fixed(&programs),
            "a raster changes the residuals' extent and the pads' columns, not the frame statistics"
        );
    }

    /// The audio encoder states its programs from its own configuration: one convolution, one Snake per
    /// channel width it runs at, the projection's two normalizations and its pool, the FP32 attention,
    /// the GeGLU, and the residual of each unit at its own width times the track's samples.
    #[test]
    fn the_audio_encoder_states_its_programs_from_its_config() {
        let config = crate::audio_vae::tests::audio_config();
        let programs =
            AudioEncoder::programs(&config, &[44_100]).expect("the audio encoder's programs");
        let count = |operator: &str| {
            programs
                .iter()
                .filter(|(name, _)| name == operator)
                .count()
        };
        assert_eq!(count("conv1d"), 1, "one program whatever the window or stride");
        assert_eq!(count("adaptive_pool"), 1);
        assert_eq!(count("transpose"), 1);
        assert_eq!(count("act_and_mul"), 1);
        assert_eq!(count("gemm"), 1);
        assert_eq!(count("dense_attention"), 1);
        assert_eq!(count("layer_norm"), 2);

        // One Snake per channel width: `encoder_dim` doubled per stage, plus the tail's width.
        let mut widths = Vec::new();
        let mut width = config.encoder_dim;
        for _ in &config.encoder_rates {
            widths.push(width);
            width *= 2;
        }
        widths.push(width);
        widths.sort_unstable();
        widths.dedup();
        assert_eq!(count("snake"), widths.len(), "{programs:?}");

        // Every launch of the encoder is FP32: the design's declared rows, not the checkpoint's type.
        for (operator, key) in &programs {
            assert!(
                key.contains("f32") || operator == "scaled_add" || operator == "act_and_mul",
                "{operator} {key} is not FP32"
            );
        }

        // The attention's key, stated whole: FP32 operands, head_dim from the final width, plain MHA
        // (q_heads == kv_heads), and no offsets table because one track is one unpacked sequence.
        // The `f32` package declares exactly this cell; an f16 form of it is served by no package and
        // the encoder never asks for one, since its `programs` takes no dtype at all.
        let attention: Vec<&str> = programs
            .iter()
            .filter(|(name, _)| name == "dense_attention")
            .map(|(_, key)| key.as_str())
            .collect();
        assert_eq!(
            attention,
            ["dtype_f32_kv_f32_out_f32_head_dim_256_q_heads_8_kv_heads_8_mask_dense_window_false_cu_seq_lens_false"],
            "{programs:?}"
        );

        // A longer track is another residual program, because the unit's whole output is its width.
        let longer = AudioEncoder::programs(&config, &[88_200]).expect("a longer track");
        assert_ne!(longer, programs);
    }
    use super::*;
    use crate::video_vae::tests::video_config;

    /// A video encoder over the fixture's config: the stack is all `latent_frames` needs, so no
    /// weights are loaded.
    fn encoder_stack() -> Stack {
        Stack::of(&video_config()).expect("the fixture's stack")
    }

    /// The encoder names only its own tensors, and each of them once.
    ///
    /// Two properties the checkpoint makes a load depend on: a name asked twice would be read twice
    /// and reported twice by a missing-name check, and a name outside the encoder's own two prefixes
    /// would be the decoder's or the post-quant convolution's - a different component's tensors in
    /// the same export, which this encoder must never read.
    #[test]
    fn keys_name_only_the_encoder_and_once_each() {
        let keys = VideoEncoder::keys(&video_config()).expect("the encoder's names");
        let distinct: std::collections::BTreeSet<&String> = keys.iter().collect();
        assert_eq!(distinct.len(), keys.len(), "a tensor is asked for twice");
        assert!(
            keys.iter()
                .all(|key| key.starts_with("encoder.") || key.starts_with("quant_conv.")),
            "a key lies outside the encoder's own tensors"
        );
    }

    #[test]
    fn the_conditioning_plan_is_one_tile_per_run_not_one_per_tile() {
        // A condition's plan is the pass's, so nothing in it scales with the tile grid: the two
        // canvases here are the served ones and their grids differ in tile count, while the plan's
        // six buffers are the same six of the same size. A per-tile buffer would fail this on the
        // larger canvas and nowhere else.
        let config = video_config();
        let stack = Stack::of(&config).expect("the stack");
        // The condition's latent frame count is the encoder's own rule, so the test derives it rather
        // than naming it: the two frame counts in this plan - the tile's, in pixels, and every other
        // buffer's, in latents - are what `conditioning_arena` checks against each other.
        let frames = 25;
        let latent_frames = stack
            .clip
            .latent_frames(frames)
            .expect("the condition's latent frames");
        let vertical = Extent {
            frames: latent_frames,
            height: 84,
            width: 48,
        };
        let wide = Extent {
            frames: latent_frames,
            height: 48,
            width: 84,
        };
        let first = conditioning_arena(&config, frames, vertical).expect("the vertical condition");
        let second = conditioning_arena(&config, frames, wide).expect("the wide condition");
        let names: Vec<&str> = first.iter().map(|buffer| buffer.name.as_str()).collect();
        assert_eq!(
            names,
            ["tile", "stitch", "moments", "sample", "clean", "anchored"]
        );
        assert_eq!(
            second
                .iter()
                .map(|buffer| buffer.name.as_str())
                .collect::<Vec<_>>(),
            names,
            "the same six buffers, whatever the canvas"
        );
        // The tile buffer is **one** tile's pixels, not the grid's: its extent is the largest tile of
        // the decoder's own grid, which is the same `Tiles` the decode reads, and the grid's count
        // never enters it.
        let tile = first
            .iter()
            .find(|buffer| buffer.name == "tile")
            .expect("a tile buffer");
        let canvas = crate::video_vae::Canvas::clip(&config, vertical.height, vertical.width);
        let tiles = crate::video_vae::Tiles::of(&config, &canvas).expect("the grid");
        let grid = tiles.tiles(&config, &canvas);
        assert!(grid.len() > 1, "the fixture's canvas is more than one tile");
        let (tile_height, tile_width) = grid
            .iter()
            .map(|tile| tile.canvas.pixels(&config))
            .max_by_key(|(height, width)| u64::from(*height) * u64::from(*width))
            .expect("a largest tile");
        assert_eq!(
            tile.elements,
            u64::from(stack.input_channels)
                * u64::from(frames)
                * u64::from(tile_height)
                * u64::from(tile_width),
            "the tile buffer is one tile's pixels, refilled per tile"
        );
    }

    #[test]
    fn a_still_is_one_frame_of_tile_and_the_pair_must_agree() {
        // A still is the case the plan's two frame counts can disagree about, in both directions: its
        // own frame count is one and so is its latent count, and a still's latent extent paired with a
        // generated piece's frame count describes a condition that cannot exist.
        let config = video_config();
        let still = Extent {
            frames: 1,
            height: 84,
            width: 48,
        };
        let plan = conditioning_arena(&config, 1, still).expect("a still's plan");
        assert_eq!(plan.len(), 6);
        let stack = Stack::of(&config).expect("the stack");
        let canvas = crate::video_vae::Canvas::clip(&config, still.height, still.width);
        let grid = crate::video_vae::Tiles::of(&config, &canvas).expect("the grid");
        let (tile_height, tile_width) = grid
            .tiles(&config, &canvas)
            .iter()
            .map(|tile| tile.canvas.pixels(&config))
            .max_by_key(|(height, width)| u64::from(*height) * u64::from(*width))
            .expect("a largest tile");
        // One frame of pixels: the tile term scales with the condition's own frame count, and that is
        // the whole difference between a still's plan and a 25-frame piece's.
        assert_eq!(
            plan[0].elements,
            u64::from(stack.input_channels) * u64::from(tile_height) * u64::from(tile_width)
        );
        // A still's latents with a generated piece's frame count is refused rather than sized.
        let mismatched = Extent { frames: 1, ..still };
        assert!(conditioning_arena(&config, 25, mismatched).is_err());
    }

    #[test]
    fn a_condition_that_is_not_whole_patches_is_refused() {
        // The condition's rows are `2 x 2` latent patches, so an odd axis has no row count at all and
        // must be refused here rather than sized by a division that rounds.
        let config = video_config();
        let odd = Extent {
            frames: 1,
            height: 83,
            width: 48,
        };
        assert!(conditioning_arena(&config, 1, odd).is_err());
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
    fn the_video_encoders_tile_plan_is_one_tile_and_says_so() {
        let config = video_config();
        let stack = Stack::of(&config).expect("the stack");
        let raster = Extent {
            frames: 25,
            height: 64,
            width: 256,
        };
        let plan = video_encoder_tile_arena(&config, raster).expect("the plan");
        // A tiled driver looks every entry up by name, so what a plan owes the driver is its set of
        // names: the tile's pad, and per level the pair its blocks alternate between and the working
        // buffer each block reads back, a projection exactly where the level widens, and the pad and
        // strided output exactly where it downsamples. Nothing else, and no name twice. This is stated
        // as the set and not as a count because a count of a formula is a second copy of the plan's
        // arithmetic, and the copy is what drifts - the input convolution's output is `level0.first`,
        // not a buffer of its own.
        let extents = stack.extents(raster);
        let mut expected = vec!["padded".to_owned()];
        for (index, level) in extents.iter().take(extents.len() - 1).enumerate() {
            let width = stack.widths[index];
            let input = match index {
                0 => stack.widths[0],
                _ => stack.widths[index - 1],
            };
            expected.push(format!("level{index}.first"));
            expected.push(format!("level{index}.second"));
            expected.push(format!("level{index}.working"));
            if input != width {
                expected.push(format!("level{index}.projected"));
            }
            let next = &extents[index + 1];
            if next.frames < level.frames || next.height < level.height || next.width < level.width
            {
                expected.push(format!("level{index}.pad"));
                expected.push(format!("level{index}.strided"));
            }
        }
        let mut planned: Vec<&str> = plan.iter().map(|buffer| buffer.name.as_str()).collect();
        planned.sort_unstable();
        let mut wanted: Vec<&str> = expected.iter().map(String::as_str).collect();
        wanted.sort_unstable();
        assert_eq!(
            planned, wanted,
            "the plan reserves exactly the buffers a tiled driver looks up, once each"
        );
        // Level 0 runs at the tile's pixels and the encoder's first width, and it is the entry a tiled
        // driver multiplies by its tile count.
        let first = plan
            .iter()
            .find(|buffer| buffer.name == "level0.first")
            .expect("level 0's first buffer");
        assert_eq!(
            first.elements,
            u64::from(stack.widths[0]) * 25 * 64 * 256,
            "level 0 is the tile's pixels at the first width"
        );
    }

    #[test]
    fn the_audio_encoders_arena_names_every_buffer_it_reserves() {
        let config = crate::audio_vae::tests::audio_config();
        let stack = AudioStack::of(&config).expect("the stack");
        let samples = 207 * u64::from(stack.samples_per_latent);
        let batch = 2;
        let plan = audio_encoder_arena(&config, batch, samples).expect("the plan");
        // The input convolution, four buffers per stage, the tail pair, and each row's projection.
        assert_eq!(
            plan.len(),
            1 + 4 * stack.rates.len() + 2 + 17 * batch as usize
        );
        // The first stage runs at the sample rate, at the input width: the encoder's largest buffer.
        let first = plan
            .iter()
            .find(|buffer| buffer.name == "stage0.first")
            .expect("stage 0's first buffer");
        assert_eq!(
            first.elements,
            batch * u64::from(stack.encoder_dim) * samples
        );
        // The lengths divide by each rate in turn, so the tail sees one row per latent frame.
        let tail = plan
            .iter()
            .find(|buffer| buffer.name == "tail")
            .expect("the tail snake's buffer");
        assert_eq!(tail.elements, batch * u64::from(stack.latent_dim) * 207);
        let mut names: Vec<&str> = plan.iter().map(|buffer| buffer.name.as_str()).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "every buffer is named once");
    }

    #[test]
    fn a_pixel_normalizes_and_quantizes_back_to_itself() {
        let rgb = crate::output::Rgb::of(1, 1, 1);
        for pixel in [
            [0u8, 0, 0],
            [255, 255, 255],
            [124, 116, 104],
            [1, 2, 3],
            [200, 10, 77],
        ] {
            let restored = rgb.quantize(normalize(pixel));
            for (channel, (got, want)) in restored.iter().zip(pixel.iter()).enumerate() {
                assert!(
                    got.abs_diff(*want) <= 1,
                    "channel {channel}: {got} for {want} from {pixel:?}"
                );
            }
        }
        // The extremes land outside the band the decoder works in - `Rgb::quantize` maps a value
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

    /// A raster view of `channels` over one frame's `rows × columns`, for the shape tests: nothing
    /// reads it, so the pointer is the null one.
    fn raster(channels: u64, frames: u64, rows: u64, columns: u64, dtype: DType) -> TensorView {
        TensorView::new(
            nn::DevicePtr::new(0),
            dtype,
            Shape::new(&[channels, frames, rows, columns]),
        )
    }

    #[test]
    fn the_posterior_sample_is_the_rasters_own_geometry() {
        let config = video_config();
        let channels = u64::from(config.latent_channels);
        let moments = raster(2 * channels, 7, 30, 52, DType::F32);
        let noise = raster(channels, 7, 30, 52, DType::F32);
        let out = raster(channels, 7, 30, 52, DType::F32);
        let sample = Sample::of(config.latent_channels, moments, noise, out).expect("a sample");
        assert_eq!(sample.channels, channels);
        assert_eq!(sample.positions, 7 * 30 * 52);
        // The moments' rows are twice the sample's, which is what makes the two fields one tensor.
        assert_eq!(moments.shape.dim(0), 2 * sample.channels);

        // A moments raster of the sample's own width has no log-variance to read.
        let narrow = raster(channels, 7, 30, 52, DType::F32);
        assert!(Sample::of(config.latent_channels, narrow, noise, out).is_err());
        // A noise raster of the moments' width is two draws where the sample wants one.
        let wide_noise = raster(2 * channels, 7, 30, 52, DType::F32);
        assert!(Sample::of(config.latent_channels, moments, wide_noise, out).is_err());
        // An FP16 raster is the decoder's precision, not the encoder's posterior.
        let half = raster(channels, 7, 30, 52, DType::F16);
        assert!(Sample::of(config.latent_channels, moments, half, out).is_err());
        // A view whose planes overlap has no single channel stride.
        let strided = TensorView::with_strides(
            nn::DevicePtr::new(0),
            DType::F32,
            Shape::new(&[channels, 7, 30, 52]),
            Strides::new(&[30 * 52, 30 * 52, 52, 1]),
        );
        assert!(Sample::of(config.latent_channels, moments, strided, out).is_err());
        // A noise raster of another clip is a different sample.
        let other = raster(channels, 5, 30, 52, DType::F32);
        assert!(Sample::of(config.latent_channels, moments, other, out).is_err());
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
    fn the_encoder_is_one_hundred_and_eighteen_tensors() {
        let config = video_config();
        let table = sources(&config).expect("the encoder's parameters");
        assert_eq!(table.len(), 118);
        let names: Vec<&str> = table
            .iter()
            .map(|source| source.parameter.as_str())
            .collect();
        for expected in [
            "quant_conv.weight",
            "input.weight",
            "norm.bias",
            "output.weight",
            "stages.0.blocks.1.norms.0.weight",
            "stages.5.blocks.0.convolutions.1.bias",
            "stages.1.blocks.0.shortcut.weight",
            "stages.3.downsample.weight",
        ] {
            assert!(names.contains(&expected), "{expected} is missing");
        }
        // The stages whose width is unchanged carry no shortcut, and the last two stride nothing.
        assert!(!names.contains(&"stages.0.blocks.0.shortcut.weight"));
        assert!(!names.contains(&"stages.5.blocks.1.shortcut.weight"));
        assert!(!names.contains(&"stages.4.downsample.weight"));
        assert!(!names.contains(&"stages.5.downsample.weight"));
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn the_tile_grid_is_the_decoders_grid_in_input_pixels() {
        let grid = Grid::of(1344, 768).expect("the grid");
        // The decoder's grid at 16 pixels per latent position: seven rows of 256 with overlaps
        // [80, 80, 80, 80, 64, 64] and four columns with [96, 80, 80].
        assert_eq!(grid.rows.starts, vec![0, 176, 352, 528, 704, 896, 1088]);
        assert_eq!(grid.rows.overlaps, vec![80, 80, 80, 80, 64, 64]);
        assert_eq!(grid.columns.starts, vec![0, 160, 336, 512]);
        // Every seam spans whole latent positions: 80/16 = 5, 64/16 = 4, 96/16 = 6.
        assert_eq!(Grid::blends(&grid.rows), vec![5, 5, 5, 5, 4, 4]);
        assert_eq!(Grid::blends(&grid.columns), vec![6, 5, 5]);
        // The rows a raster of tiles owns: each start in latent rows, then the height.
        assert_eq!(grid.owned(84), vec![0, 11, 22, 33, 44, 56, 68, 84]);
    }

    #[test]
    fn a_band_reads_the_tile_rows_it_cross_fades_with() {
        let grid = Grid::of(1344, 768).expect("the grid");
        let latent_height = 1344 / SPATIAL_RATIO;
        assert_eq!(latent_height, 84);
        // The first band starts at the top: no row above it.
        assert_eq!(grid.band(84, 0, 11).expect("band"), (0, 0));
        // A band that starts on a row's own first latent row reads only that row and below.
        assert_eq!(grid.band(84, 16, 33).expect("band"), (1, 2));
        // A band that starts inside a row's cross-fade reads the row above.
        let owned = grid.owned(84);
        assert_eq!(owned[1], 11);
        assert_eq!(grid.rows.overlaps[0] / SPATIAL_RATIO, 5);
        assert_eq!(grid.band(84, 13, 20).expect("band"), (0, 1));
        // Past the end is refused.
        assert!(grid.band(84, 80, 90).is_err());
        assert!(grid.band(84, 5, 5).is_err());
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
    fn the_audio_encoder_table_names_every_parameter() {
        let table = audio_sources(&crate::audio_vae::tests::audio_config())
            .expect("the encoder's parameters");
        // The input pair, five stages of twenty-one — eighteen for the three residual units, two for
        // the widening convolution and one for the stage's Snake — the output Snake and its pair, and
        // the projection block with the mean convolution: twenty-one tensors and the fused
        // query-key-value bias.
        assert_eq!(table.len(), 2 + 5 * 21 + 3 + 22);
        let names: Vec<&str> = table
            .iter()
            .map(|source| source.parameter.as_str())
            .collect();
        for expected in [
            "input.weight",
            "stages.0.units.0.convs.0.weight",
            "stages.4.units.2.convs.1.bias",
            "stages.4.widen.weight",
            "stages.4.units.2.snakes.0.alpha",
            "stages.4.snake.alpha",
            "output_snake.alpha",
            "output.weight",
            "projection.qkv.weight",
            "projection.qkv.bias",
            "projection.mlp.gate.weight",
            "mean_projection.bias",
        ] {
            assert!(names.contains(&expected), "{expected} is missing");
        }
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len());
        // Seven weight-normalized convolutions per stage — six in the three residual units and the
        // widening one — plus the input and the output convolution. The projection block's weights and
        // the mean convolution are plain, as the checkpoint stores them.
        let normalized = table
            .iter()
            .filter(|source| matches!(source.tensor, crate::audio_vae::Tensor::Normalized { .. }))
            .count();
        assert_eq!(normalized, 1 + 5 * 7 + 1);
        // The fused query-key-value bias names its three checkpoint vectors in order.
        let bias = table
            .iter()
            .find(|source| source.parameter == "projection.qkv.bias")
            .expect("the fused bias");
        match &bias.tensor {
            crate::audio_vae::Tensor::Parts(parts) => assert_eq!(
                parts,
                &vec![
                    "pre_block.attn.q_bias".to_owned(),
                    "pre_block.attn.zero_k_bias".to_owned(),
                    "pre_block.attn.v_bias".to_owned(),
                ]
            ),
            other => panic!("the fused bias is {other:?}"),
        }
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
}
