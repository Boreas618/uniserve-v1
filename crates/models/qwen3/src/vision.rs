//! The Qwen3-VL vision tower (`model.visual`): tubelet patches to language-width tokens.
//!
//! One image block's patches (a still image is one time step) are projected by the patch
//! embedding, given a learned 2-D position resampled to their grid, rotated by a 2-D rotary
//! position, encoded by `depth` pre-norm blocks, and merged so that every `merge × merge` block of
//! patches becomes one token of the language model's width. The blocks the checkpoint names in
//! `deepstack_visual_indexes` are merged the same way and returned beside the last one: these are
//! the DeepStack features the language model adds at its own layer boundaries.
//!
//! ## What a grid decides
//!
//! The forward computes a grid's host values and places them on the device
//! ([`nn::Forward::upload`]): the learned table's four taps and FP32 bilinear weights per patch
//! ([`positions::position_samples`]), each patch's `(row, column)` rotary coordinates
//! ([`positions::rotary_coordinates`]), and the attention offsets of the grid's time steps
//! ([`positions::vision_lengths`]). Attention never crosses a time step: each temporal patch of a
//! grid is one bidirectional sequence of `height · width` patches.
//!
//! The 2-D rotary is the reference's `RotaryEmbedding(head_dim / 2)` evaluated at two coordinates
//! per patch: each axis reads the whole `head_dim / 4`-frequency spectrum with its own coordinate,
//! the axes in contiguous blocks ([`Spectrum::Repeated`]), computed on the device from the
//! uploaded coordinates.
//!
//! ## Precision and layout
//!
//! The tower's activations are the checkpoint's own representation. The position blend accumulates
//! in FP32 and rounds once into the activation type; the rotary factors are FP32. The patch
//! embedding reads the processor's flat channel-major tubelets against the checkpoint's Conv3d
//! weight flattened in the same order: its kernel equals its stride, so it is one matrix product.
//! The blocks' MLPs take the checkpoint's `gelu_pytorch_tanh`, the mergers PyTorch's exact
//! `nn.GELU()`.

use nn::blocks::{AttentionKind, Keys, QkNormNames, QkvAttention, QkvAttentionSpec, RopeFactors};
use nn::capability::{PatchFormat, PixelNorm, Vision};
use nn::config::Fields;
use nn::family::Split;
use nn::layers::{
    Bias, Gelu, LayerNorm, Linear, Parallel, Precision, QkNorm, QkvBias, QkvNames, QkvSpec,
    Residual, Rotary, RotarySpec, Spectrum, Varlen,
};
use nn::weights::{Scope, WeightLoader};
use nn::{DType, Forward, Operand, Shape, Weight};
use ops::position::{Args as InterpolateArgs, Descriptor as InterpolateDescriptor};
use processing::qwen_vl::ProcessorConfig;

use crate::config::{POSITION_AXES, VisionConfig};
use crate::{Error, Result, positions, write};

/// The checkpoint's module path of the vision tower.
const PREFIX: &str = "model.visual";

/// Every normalization's variance floor (`LAYER_NORM_EPS` in the reference).
const LAYER_NORM_EPS: f32 = 1e-6;

/// The 2-D rotary frequency base.
const ROPE_THETA: f32 = 10_000.0;

/// The fence class of one tower block, at path `model.visual.blocks.<i>`.
const BLOCK: &str = "models.qwen3.vision.VisionBlock";

/// The checkpoint's processor rules and its images' pixel normalization, or `None` when it states
/// no processor files: such a checkpoint serves text alone.
///
/// # Errors
///
/// A file that is not a Qwen-VL processor config.
pub fn open_processor(checkpoint: &nn::Checkpoint) -> Result<Option<(ProcessorConfig, PixelNorm)>> {
    let [image_file, video_file] = ProcessorConfig::FILES;
    let (Some(image), Some(video)) = (
        checkpoint.sidecar(image_file)?,
        checkpoint.sidecar(video_file)?,
    ) else {
        return Ok(None);
    };
    let norm = pixel_norm(&Fields::of(image_file, &image)?)?;
    let processor = ProcessorConfig::from_values(&image, &video).map_err(Error::Config)?;
    Ok(Some((processor, norm)))
}

/// The pixel normalization a processor's `image_mean` and `image_std` state: each RGB channel maps
/// `x` in `[0, 255]` to `(x / 255 - mean) / std` ([`PixelNorm`]).
///
/// # Errors
///
/// Fields of the wrong type or length, or a deviation that is not positive.
pub fn pixel_norm(values: &Fields<'_>) -> Result<PixelNorm> {
    let owner = values.owner();
    let mean = values.fixed_numbers("image_mean", 3)?;
    let std = values.fixed_numbers("image_std", 3)?;
    let mean: [f32; 3] = [mean[0] as f32, mean[1] as f32, mean[2] as f32];
    let std: [f32; 3] = [std[0] as f32, std[1] as f32, std[2] as f32];
    if std.iter().any(|value| *value <= 0.0) {
        return Err(Error::Config(format!(
            "{owner}.image_std must be positive, got {std:?}"
        )));
    }
    Ok(PixelNorm { mean, std })
}

/// How the tower reads a clip's pixels as patch rows, which the runtime writes ([`PatchFormat`]).
///
/// A row is one tubelet, channel-major (`channels · temporal · patch²`, the layout Qwen-VL
/// processors emit), each element `(byte - 255 · mean) / (255 · std)` in FP32 rounded once to the
/// tower's element type `dtype` (`tubelets.to(self.weight.dtype)`). The rows run over each temporal
/// patch in merge-block order, and a temporal patch short of frames repeats the clip's last frame:
/// an image is one frame repeated.
pub fn patch_format(vision: &VisionConfig, norm: PixelNorm, dtype: DType) -> PatchFormat {
    PatchFormat {
        side: vision.patch_size,
        temporal: vision.temporal_patch_size,
        merge: vision.spatial_merge_size,
        channels: vision.in_channels,
        norm,
        dtype,
    }
}

/// One pre-norm block: attention and an ungated MLP, each added to the residual stream.
struct VisionLayer {
    norm1: LayerNorm,
    /// The query, key and value windows of the checkpoint's fused `attn.qkv`, the 2-D rotation
    /// without a norm, dense attention per time step, and `attn.proj`.
    attention: QkvAttention,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    path: String,
}

impl VisionLayer {
    /// The block at `scope`, its softmax scale `scale`.
    fn load(scope: &mut Scope<'_, '_>, config: &VisionConfig, scale: f32) -> Result<Self> {
        let hidden = config.hidden_size;
        let attention = QkvAttention::load(
            &mut scope.scope("attn"),
            QkvAttentionSpec {
                qkv: QkvSpec {
                    hidden,
                    heads: config.num_heads,
                    kv_heads: config.num_heads,
                    head_dim: config.head_dim(),
                    gated: false,
                },
                names: QkvNames::Fused("qkv"),
                bias: QkvBias::Present,
                output: "proj",
                output_bias: Bias::Present,
                qk_norm: QkNormNames::None,
                // A block's key bound is its own grid's patches, a launch value.
                kind: AttentionKind::Dense {
                    max_keys: u32::MAX,
                    causal: false,
                    window: None,
                },
                scale,
            },
            Split::WHOLE,
        )?;
        let norm1 = LayerNorm::load(&mut scope.scope("norm1"), hidden, LAYER_NORM_EPS)?;
        let norm2 = LayerNorm::load(&mut scope.scope("norm2"), hidden, LAYER_NORM_EPS)?;
        let linear = |scope: &mut Scope<'_, '_>, module: &str, inputs: u32, outputs: u32| {
            Linear::load(
                &mut scope.scope(module),
                inputs,
                outputs,
                Bias::Present,
                Parallel::None,
            )
        };
        let mut mlp = scope.scope("mlp");
        let fc1 = linear(&mut mlp, "linear_fc1", hidden, config.intermediate_size)?;
        let fc2 = linear(&mut mlp, "linear_fc2", config.intermediate_size, hidden)?;
        Ok(Self {
            norm1,
            attention,
            norm2,
            fc1,
            fc2,
            path: scope.path().to_owned(),
        })
    }

    /// The block over `hidden`, `[patches, hidden]`, updated in place.
    fn forward(
        &self,
        f: &Forward,
        hidden: Operand,
        rope: RopeFactors<'_>,
        varlen: Varlen,
        gelu: Gelu,
    ) -> Result<()> {
        f.scoped(|f| {
            let normalized = self.norm1.forward(f, hidden, None)?;
            let attended =
                self.attention
                    .forward(f, normalized, Some(rope), Keys::Dense(Some(varlen)))?;
            Residual::default().forward(f, hidden, attended, None)?;
            let normalized = self.norm2.forward(f, hidden, None)?;
            let activated = gelu.forward(f, self.fc1.forward(f, normalized)?)?;
            let update = self.fc2.forward(f, activated)?;
            Residual::default().forward(f, hidden, update, None)?;
            Ok(())
        })
    }
}

/// A patch merger: normalize, concatenate every `merge²` patch rows into one, project twice.
///
/// The final merger normalizes each patch before the concatenation; a DeepStack merger normalizes
/// the concatenated row (the reference's `use_postshuffle_norm`).
struct PatchMerger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    width: u32,
    normalizes_merged: bool,
}

impl PatchMerger {
    /// The merger at `scope`.
    fn load(
        scope: &mut Scope<'_, '_>,
        config: &VisionConfig,
        normalizes_merged: bool,
    ) -> Result<Self> {
        let width = config.merged_width();
        let normalized = if normalizes_merged {
            width
        } else {
            config.hidden_size
        };
        Ok(Self {
            norm: LayerNorm::load(&mut scope.scope("norm"), normalized, LAYER_NORM_EPS)?,
            fc1: Linear::load(
                &mut scope.scope("linear_fc1"),
                width,
                width,
                Bias::Present,
                Parallel::None,
            )?,
            fc2: Linear::load(
                &mut scope.scope("linear_fc2"),
                width,
                config.out_hidden_size,
                Bias::Present,
                Parallel::None,
            )?,
            width,
            normalizes_merged,
        })
    }

    /// `hidden`'s patch rows merged: `[patches / merge², out_hidden]`.
    fn forward(&self, f: &Forward, hidden: Operand, gelu: Gelu) -> Result<Operand> {
        let width = u64::from(self.width);
        let merged = Shape::new(&[hidden.numel() / width, width]);
        let normalized = if self.normalizes_merged {
            self.norm.forward(f, hidden.reshape(merged), None)?
        } else {
            self.norm.forward(f, hidden, None)?.reshape(merged)
        };
        let activated = gelu.forward(f, self.fc1.forward(f, normalized)?)?;
        Ok(self.fc2.forward(f, activated)?)
    }
}

/// The Qwen3-VL vision tower, with the processor rules its images are sized and laid out by.
pub struct VisionTower {
    config: VisionConfig,
    processor: ProcessorConfig,
    format: PatchFormat,
    dtype: DType,
    patch: Linear,
    position: Weight,
    rotary: Rotary,
    layers: Vec<VisionLayer>,
    merger: PatchMerger,
    /// One merger per DeepStack block, in `deepstack_visual_indexes` order.
    deepstack: Vec<PatchMerger>,
}

impl VisionTower {
    /// The tower of `config` read through `loader` in `dtype`, its images sized by `processor` and
    /// normalized by `norm`.
    ///
    /// # Errors
    ///
    /// A processor whose patch geometry is not the tower's, or a module the loader refuses.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        config: &VisionConfig,
        processor: ProcessorConfig,
        norm: PixelNorm,
        dtype: DType,
    ) -> Result<Self> {
        if (
            processor.patch_size,
            processor.temporal_patch_size,
            processor.merge_size,
        ) != (
            config.patch_size,
            config.temporal_patch_size,
            config.spatial_merge_size,
        ) {
            return Err(Error::Config(format!(
                "the processor patches {} over {} frames merging {}, the tower {} over {} merging {}",
                processor.patch_size,
                processor.temporal_patch_size,
                processor.merge_size,
                config.patch_size,
                config.temporal_patch_size,
                config.spatial_merge_size
            )));
        }
        let head_dim = config.head_dim();
        let spectrum = Spectrum::Repeated {
            theta: ROPE_THETA,
            pairs: head_dim / 4,
            axes: 2,
            precision: Precision::F32,
        };
        let scale = spectrum.score_scale(head_dim)?;
        let mut tower = loader.scope(PREFIX);
        let rotary = Rotary::load(
            &mut tower.scope("rotary_pos_emb"),
            spectrum,
            RotarySpec {
                head_dim,
                heads: config.num_heads,
                kv_heads: config.num_heads,
                qk_norm: QkNorm::None,
            },
        )?;
        let layers = (0..config.depth)
            .map(|index| {
                VisionLayer::load(&mut tower.scope(format!("blocks.{index}")), config, scale)
            })
            .collect::<Result<Vec<_>>>()?;
        let deepstack = (0..config.deepstack_visual_indexes.len())
            .map(|index| {
                PatchMerger::load(
                    &mut tower.scope(format!("deepstack_merger_list.{index}")),
                    config,
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch: Linear::load(
                &mut tower.scope("patch_embed.proj"),
                config.tubelet(),
                config.hidden_size,
                Bias::Present,
                Parallel::None,
            )?,
            position: tower.scope("pos_embed").tensor(
                "weight",
                &[
                    u64::from(config.num_position_embeddings),
                    u64::from(config.hidden_size),
                ],
            )?,
            merger: PatchMerger::load(&mut tower.scope("merger"), config, false)?,
            format: patch_format(config, norm, dtype),
            config: config.clone(),
            processor,
            dtype,
            rotary,
            layers,
            deepstack,
        })
    }

    /// The tower's configuration, the placeholder tokens its rows replace included.
    pub const fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// The patch rows and merged rows of one block of `grid` (patches), refused when a side is
    /// not whole merge blocks.
    fn rows(&self, grid: [u32; 3]) -> Result<(u64, u64)> {
        let [time, height, width] = grid;
        let merge = self.config.spatial_merge_size;
        if time == 0 || height == 0 || width == 0 || height % merge != 0 || width % merge != 0 {
            return Err(Error::Config(format!(
                "the grid {grid:?} is not whole {merge}x{merge} merge blocks"
            )));
        }
        let patches = u64::from(time) * u64::from(height) * u64::from(width);
        Ok((patches, patches / u64::from(merge * merge)))
    }

    /// Encode one block of `grid` from `pixels`, `[patches, tubelet]`: the merged features,
    /// `[merged, out_hidden]`, and one DeepStack table of the same shape per DeepStack block.
    ///
    /// # Errors
    ///
    /// Pixels that do not describe the grid, or what a layer refuses.
    pub fn forward(
        &self,
        f: &Forward,
        grid: [u32; 3],
        pixels: Operand,
    ) -> Result<(Operand, Vec<Operand>)> {
        let [time, height, width] = grid;
        let (patches, _) = self.rows(grid)?;
        if pixels.shape.dims() != [patches, u64::from(self.config.tubelet())] {
            return Err(Error::Config(format!(
                "pixels of {:?} for a {grid:?} grid",
                pixels.shape.dims()
            )));
        }
        let merge = self.config.spatial_merge_size;
        // Every time step samples the same grid, so one step's values repeat `time` times.
        let repeat = |values: Vec<i32>| -> Vec<i32> {
            values
                .iter()
                .copied()
                .cycle()
                .take(values.len() * time as usize)
                .collect()
        };
        let (taps, blend) =
            positions::position_samples(self.config.position_side(), height, width, merge)?;
        let taps = f.upload(
            &repeat(taps.into_iter().flatten().collect()),
            Shape::new(&[patches, 4]),
        )?;
        let step: Vec<f32> = blend.into_iter().flatten().collect();
        let blend = f.upload(
            &step
                .iter()
                .copied()
                .cycle()
                .take(step.len() * time as usize)
                .collect::<Vec<_>>(),
            Shape::new(&[patches, 4]),
        )?;
        // `[axes, patches]` axis-major: every patch's row, then every patch's column.
        let coordinates = positions::rotary_coordinates(height, width, merge)?;
        let mut axes = Vec::with_capacity(2 * patches as usize);
        for axis in 0..2 {
            for _ in 0..time {
                axes.extend(coordinates.iter().map(|coordinate| coordinate[axis] as i32));
            }
        }
        let coordinates = f.upload(&axes, Shape::new(&[2, patches]))?;
        let lengths = f.upload(
            &positions::vision_lengths(time, height, width)?,
            Shape::new(&[u64::from(time) + 1]),
        )?;

        let hidden = self.patch.forward(f, pixels)?;
        let resampled = self.interpolate(f, taps, blend, patches)?;
        Residual::default().forward(f, hidden, resampled, None)?;
        let rope = RopeFactors {
            rotary: &self.rotary,
            factors: self.rotary.factors(f, coordinates)?,
        };
        let varlen = Varlen {
            cu_q: lengths,
            cu_k: lengths,
            max_q: Some(height * width),
            max_k: u32::try_from(patches)
                .map_err(|_| Error::Config(format!("a grid of {patches} patches")))?,
        };
        let mut deepstack = Vec::with_capacity(self.deepstack.len());
        for (index, layer) in self.layers.iter().enumerate() {
            f.fenced(BLOCK, format_args!("{}", layer.path), || {
                layer.forward(f, hidden, rope, varlen, Gelu::tanh())
            })?;
            if let Some(position) = self
                .config
                .deepstack_visual_indexes
                .iter()
                .position(|block| *block as usize == index)
            {
                deepstack.push(self.deepstack[position].forward(f, hidden, Gelu::exact())?);
            }
        }
        Ok((self.merger.forward(f, hidden, Gelu::exact())?, deepstack))
    }

    /// The learned position table resampled at `taps` with `weights`: `[patches, hidden]`, blended
    /// in FP32 and rounded once into the activation type.
    fn interpolate(
        &self,
        f: &Forward,
        taps: Operand,
        weights: Operand,
        patches: u64,
    ) -> Result<Operand> {
        let out = f.empty(
            self.dtype,
            Shape::new(&[patches, u64::from(self.config.hidden_size)]),
        )?;
        f.operate::<ops::position::PositionInterpolate>(
            &InterpolateDescriptor {
                dtype: self.dtype,
                channels: self.config.hidden_size,
                in_side: self.config.position_side(),
                block: 1024,
                pdl: false,
            },
            InterpolateArgs {
                table: self.position.view(),
                indices: taps,
                weights,
                out,
            },
        )?;
        Ok(out)
    }
}

impl Vision for VisionTower {
    fn image_grid(&self, height: u32, width: u32) -> nn::Result<[u32; 3]> {
        let grid = self
            .processor
            .image_grid(height, width)
            .map_err(|error| nn::Error::from(Error::Config(error.to_string())))?;
        Ok([grid.t, grid.h, grid.w])
    }

    fn patch_format(&self) -> PatchFormat {
        self.format
    }

    fn image_rows(&self, grid: [u32; 3]) -> nn::Result<u32> {
        if grid[0] != 1 {
            return Err(
                Error::Config(format!("an image is one time step, not the grid {grid:?}")).into(),
            );
        }
        let (_, merged) = self.rows(grid)?;
        Ok(u32::try_from(merged).map_err(|_| Error::Config(format!("the grid {grid:?}")))?)
    }

    fn max_image_rows(&self) -> u32 {
        // A resized image's area is at most the processor's pixel budget, in whole merged tokens.
        let token = u64::from(self.processor.patch_size * self.processor.merge_size).pow(2);
        u32::try_from(self.processor.image_max_pixels / token.max(1)).unwrap_or(u32::MAX)
    }

    fn position_axes(&self) -> u32 {
        POSITION_AXES
    }

    fn image_positions(
        &self,
        grid: [u32; 3],
        start: u32,
        out: &mut [&mut [i32]],
    ) -> nn::Result<()> {
        let rows = self.image_rows(grid)? as usize;
        if out.len() != POSITION_AXES as usize || out.iter().any(|axis| axis.len() != rows) {
            return Err(Error::Config(format!(
                "{} coordinate rows for an image of {rows} rows over {POSITION_AXES} axes",
                out.len()
            ))
            .into());
        }
        // The block alone, as `rope_index` places one run of placeholders at origin zero; `start`
        // is added on every axis.
        const PLACEHOLDER: u32 = 1;
        let coordinates = positions::rope_index(
            &vec![PLACEHOLDER; rows],
            &[grid],
            &[],
            PLACEHOLDER,
            PLACEHOLDER + 1,
            self.config.spatial_merge_size,
        )?;
        for (row, coordinate) in coordinates.iter().enumerate() {
            for (axis, value) in coordinate.iter().enumerate() {
                out[axis][row] = i32::try_from(i64::from(start) + value).map_err(|_| {
                    nn::Error::from(Error::Config(
                        "an M-RoPE coordinate wider than an i32".to_owned(),
                    ))
                })?;
            }
        }
        Ok(())
    }

    fn encode_image(
        &self,
        f: &Forward,
        grid: [u32; 3],
        pixels: Operand,
        out: Operand,
    ) -> nn::Result<()> {
        f.scoped(|f| {
            let (features, _) = self.forward(f, grid, pixels)?;
            write(f, features, out)
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_processor_states_three_positive_deviations() {
        let fields =
            serde_json::json!({"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.0, 0.5]});
        let error =
            pixel_norm(&Fields::of("preprocessor_config.json", &fields).expect("an object"))
                .expect_err("a zero deviation is refused");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("image_std")),
            "{error}"
        );

        let fields =
            serde_json::json!({"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]});
        let norm = pixel_norm(&Fields::of("preprocessor_config.json", &fields).unwrap()).unwrap();
        assert_eq!((norm.mean, norm.std), ([0.5; 3], [0.5; 3]));
    }
}
