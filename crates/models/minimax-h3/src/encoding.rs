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
            return Err(Error::config("the video VAE has no latent channel".to_owned()));
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
    pub fn moments(&self, raster: Extent) -> Extent {
        *self.extents(raster).last().expect("a nonempty stack")
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
            Some(above) if start < owned[first] + self.rows.overlaps[above] / SPATIAL_RATIO => above,
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
        values[channel] =
            (f32::from(*byte) / crate::output::BYTE_MAX - crate::output::PIXEL_MEAN[channel])
                / crate::output::PIXEL_STD[channel];
    }
    values
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

    /// The latent frames a track of `samples` encodes to, which must be a whole number of frames.
    pub fn latent_frames(&self, samples: u32) -> Result<u32> {
        if !samples.is_multiple_of(self.samples_per_latent) {
            return Err(Error::config(format!(
                "{samples} samples are not whole {} sample latent frames",
                self.samples_per_latent
            )));
        }
        Ok(*self.lengths(samples).last().expect("a nonempty stack"))
    }
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
    normalized(&mut sources, "input.weight".to_owned(), "encoder.block.0".to_owned());
    region(&mut sources, "input.bias".to_owned(), "encoder.block.0.bias".to_owned());
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
        }
        normalized(&mut sources, at("widen.weight"), format!("{base}.5"));
        region(&mut sources, at("widen.bias"), format!("{base}.5.bias"));
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_vae::tests::video_config;

    #[test]
    fn a_pixel_normalizes_and_quantizes_back_to_itself() {
        let rgb = crate::output::Rgb::of(1, 1, 1);
        for pixel in [[0u8, 0, 0], [255, 255, 255], [124, 116, 104], [1, 2, 3], [200, 10, 77]] {
            let restored = rgb.quantize(normalize(pixel));
            for (channel, (got, want)) in restored.iter().zip(pixel.iter()).enumerate() {
                assert!(
                    got.abs_diff(*want) <= 1,
                    "channel {channel}: {got} for {want} from {pixel:?}"
                );
            }
        }
        // The extremes are exact: the normalization maps 0 and 255 outside the [-1, 1] range the
        // decoder works in, which is expected — the encoder reads the raster, not latents.
        assert!(normalize([0, 0, 0])[0] < -2.0);
        assert!(normalize([255, 255, 255])[0] > 2.0);
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
            extents.iter().map(|extent| extent.frames).collect::<Vec<_>>(),
            vec![17, 17, 9, 5, 5, 5, 5]
        );
        // 256 pixels halve four times: the 16-pixel compression.
        assert_eq!(
            extents.iter().map(|extent| extent.height).collect::<Vec<_>>(),
            vec![256, 128, 64, 32, 16, 16, 16]
        );
        assert_eq!(stack.moments(extents[0]).frames, 5);
        // A single frame stays one latent frame.
        let single = stack.moments(Extent {
            frames: 1,
            height: 16,
            width: 16,
        });
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
        // A partial frame is refused: the encoder's output is whole latent frames.
        assert!(stack.latent_frames(801).is_err());
        // Each stage doubles the width from the input convolution's.
        assert_eq!(stack.widths, vec![128, 256, 512, 1024, 2048]);
        assert_eq!(stack.latent_dim, 2048);
    }

    #[test]
    fn the_audio_encoder_table_names_every_parameter() {
        let table = audio_sources(&crate::audio_vae::tests::audio_config())
            .expect("the encoder's parameters");
        // The input pair, five stages of twenty-one, the output pair, the projection's twenty-two and
        // the mean convolution's pair.
        assert_eq!(table.len(), 3 + 5 * 21 + 3 + 22);
        let names: Vec<&str> = table
            .iter()
            .map(|source| source.parameter.as_str())
            .collect();
        for expected in [
            "input.weight",
            "stages.0.units.0.convs.0.weight",
            "stages.4.units.2.convs.1.bias",
            "stages.4.widen.weight",
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
        // Every convolution is weight-normalized: four per stage plus the input, the output and the
        // mean convolution.
        let normalized = table
            .iter()
            .filter(|source| matches!(source.tensor, crate::audio_vae::Tensor::Normalized { .. }))
            .count();
        assert_eq!(normalized, 3 + 5 * 7 + 3);
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
