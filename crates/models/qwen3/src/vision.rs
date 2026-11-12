//! The Qwen3-VL vision tower (`model.visual`): tubelet patches to language-width tokens.
//!
//! One vision block's patches (a still image is one time step) are projected by the patch
//! embedding, given a learned 2-D position resampled to their frame, rotated by a 2-D rotary
//! position, encoded by `depth` pre-norm blocks, and merged so that every `merge × merge` block
//! of patches becomes one token of the language model's width. The blocks the checkpoint names
//! in `deepstack_visual_indexes` are merged the same way and returned beside the last one:
//! these are the DeepStack features the language model adds at its own layer boundaries.
//!
//! The module tree is vLLM's `Qwen3_VisionTransformer`
//! (`refs/vllm/vllm/model_executor/models/qwen3_vl.py:381-908`) and transformers'
//! `Qwen3VLVisionModel` (`transformers/models/qwen3_vl/modeling_qwen3_vl.py:609-732`).
//!
//! ## What a grid decides
//!
//! The forward computes a grid's position tables as CPU tensors ([`crate::positions`]) and
//! moves them to the device of the patches: the learned table's four rows and bilinear weights
//! per patch, each patch's `(row, column)` rotary coordinates, and the offsets of the grid's
//! time steps. Attention never crosses a time step: each temporal patch of a grid is one
//! bidirectional sequence of `height · width` patches.
//!
//! The 2-D rotary is the reference's `RotaryEmbedding(head_dim / 2)` evaluated at two
//! coordinates per patch: each axis reads the whole `head_dim / 4`-frequency spectrum with its
//! own coordinate, the axes in contiguous blocks ([`Spectrum::Repeated`]).
//!
//! ## Precision and layout
//!
//! The tower's activations are the checkpoint's own representation. The position blend
//! multiplies the table's rows by FP32 weights, sums in FP32 and rounds once into the
//! activation type (`modeling_qwen3_vl.py:702-703`); the rotary factors are FP32. The patch
//! embedding reads the processor's flat channel-major tubelets against the checkpoint's Conv3d
//! weight flattened in the same order: its kernel equals its stride, so it is one matrix
//! product. The blocks' MLPs take the checkpoint's `gelu_pytorch_tanh`, the mergers PyTorch's
//! exact `nn.GELU()`.

use nn::modules::{
    Attention, AttentionConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Parallel,
    Precision, QkNorm, QkvBias, QkvLinearConfig, QkvNames, Rotary, RotaryConfig, Spectrum,
};
use nn::{DType, PatchFormat, PixelNorm, Split, Tensor, VarBuilder, Vision, functional};
use processing::qwen_vl::ProcessorConfig;

use crate::config::{POSITION_AXES, VisionConfig};
use crate::{Error, Result, positions};

/// Every normalization's variance floor (`partial(nn.LayerNorm, eps=1e-6)` in the reference).
const LAYER_NORM_EPS: f32 = 1e-6;

/// The 2-D rotary frequency base.
const ROPE_THETA: f32 = 10_000.0;

/// The profiler class of one tower block, at path `model.visual.blocks.<i>`.
const BLOCK: &str = "models.qwen3.vision.VisionBlock";

/// How the tower reads a clip's pixels as patch rows, which execution writes
/// ([`PatchFormat`]).
///
/// A row is one tubelet, channel-major (`channels · temporal · patch²`, the layout Qwen-VL
/// processors emit), each element `(byte - 255 · mean) / (255 · std)` in FP32 rounded once to
/// the tower's element type `dtype` (`tubelets.to(self.weight.dtype)`). The rows run over each
/// temporal patch in merge-block order, and a temporal patch short of frames repeats the
/// clip's last frame: an image is one frame repeated.
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

/// A projection with a bias at `vb`, whole on one rank.
fn linear(vb: VarBuilder<'_, '_>, inputs: u32, outputs: u32) -> Result<Linear> {
    let config = LinearConfig {
        inputs,
        outputs,
        bias: true,
        parallel: Parallel::None,
    };

    Ok(Linear::load(&config, vb)?)
}

/// A layer normalization with a bias over rows of `width` at `vb`.
fn layer_norm(vb: VarBuilder<'_, '_>, width: u32) -> Result<LayerNorm> {
    let config = LayerNormConfig {
        width,
        eps: LAYER_NORM_EPS,
        bias: true,
    };

    Ok(LayerNorm::load(&config, vb)?)
}

/// One pre-norm block of the tower: attention and an ungated MLP, each added to the stream
/// (`Qwen3_VisionBlock`).
pub struct VisionBlock {
    norm1: LayerNorm,
    /// The query, key and value rows of the checkpoint's fused `attn.qkv`, the 2-D rotation
    /// without a norm, bidirectional attention per time step, and `attn.proj`.
    attn: Attention,
    norm2: LayerNorm,
    linear_fc1: Linear,
    linear_fc2: Linear,
    /// The block's module path, which names its profiler range.
    path: String,
}

impl VisionBlock {
    /// The block at `vb` (`model.visual.blocks.<i>`), its softmax scale `scale`.
    ///
    /// # Errors
    ///
    /// A module the loader refuses.
    pub fn load(config: &VisionConfig, vb: VarBuilder<'_, '_>, scale: f32) -> Result<Self> {
        let hidden = config.hidden_size;
        let attention = AttentionConfig {
            qkv: QkvLinearConfig {
                hidden,
                heads: config.num_heads,
                kv_heads: config.num_heads,
                head_dim: config.head_dim(),
                gated: false,
                names: QkvNames::Fused("qkv"),
                bias: QkvBias::Present,
                split: Split::WHOLE,
            },
            output: "proj",
            output_bias: true,
            qk_norm: QkNorm::None,
            scale,
            causal: false,
            window: None,
            // A block's key bound is its own grid's patches, which each forward states.
            max_kv_len: u32::MAX,
        };
        let mlp = vb.pp("mlp");

        Ok(Self {
            norm1: layer_norm(vb.pp("norm1"), hidden)?,
            attn: Attention::load(&attention, vb.pp("attn"))?,
            norm2: layer_norm(vb.pp("norm2"), hidden)?,
            linear_fc1: linear(mlp.pp("linear_fc1"), hidden, config.intermediate_size)?,
            linear_fc2: linear(mlp.pp("linear_fc2"), config.intermediate_size, hidden)?,
            path: vb.path().to_owned(),
        })
    }

    /// The block over `hidden`, `[patches, hidden]`: the stream after it.
    ///
    /// `rope` is the grid's `(cos, sin)` factor tables and `offsets` the `[time + 1]` offsets
    /// of its time steps, each one sequence of `frame` patches.
    ///
    /// # Errors
    ///
    /// Tensors a module refuses.
    pub fn forward(
        &self,
        hidden: &Tensor,
        rope: (&Tensor, &Tensor),
        offsets: &Tensor,
        frame: u32,
    ) -> Result<Tensor> {
        nn::profiler::range(BLOCK, &self.path, || {
            let patches = hidden.dim(0)? as u32;
            let attended = self.attn.forward_varlen(
                &self.norm1.forward(hidden)?,
                Some(rope),
                offsets,
                offsets,
                Some(frame),
                patches,
            )?;
            let hidden = (hidden + attended)?;

            let activated =
                functional::gelu_tanh(&self.linear_fc1.forward(&self.norm2.forward(&hidden)?)?)?;
            Ok((&hidden + self.linear_fc2.forward(&activated)?)?)
        })
    }
}

/// A patch merger: normalize, read every `merge²` patch rows as one, project twice
/// (`Qwen3_VisionPatchMerger`).
///
/// The final merger normalizes each patch before the rows are joined; a DeepStack merger
/// normalizes the joined row (the reference's `use_postshuffle_norm`).
pub struct PatchMerger {
    norm: LayerNorm,
    linear_fc1: Linear,
    linear_fc2: Linear,
    /// The width of one joined row: `merge² · hidden`.
    width: usize,
    use_postshuffle_norm: bool,
}

impl PatchMerger {
    /// The merger at `vb`.
    ///
    /// # Errors
    ///
    /// A module the loader refuses.
    pub fn load(
        config: &VisionConfig,
        vb: VarBuilder<'_, '_>,
        use_postshuffle_norm: bool,
    ) -> Result<Self> {
        let width = config.merged_width();
        let normalized = if use_postshuffle_norm {
            width
        } else {
            config.hidden_size
        };

        Ok(Self {
            norm: layer_norm(vb.pp("norm"), normalized)?,
            linear_fc1: linear(vb.pp("linear_fc1"), width, width)?,
            linear_fc2: linear(vb.pp("linear_fc2"), width, config.out_hidden_size)?,
            width: width as usize,
            use_postshuffle_norm,
        })
    }

    /// The patch rows of `hidden`, `[patches, hidden]`, merged: `[patches / merge²,
    /// out_hidden]`.
    ///
    /// # Errors
    ///
    /// Tensors a module refuses.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let merged = (hidden.elem_count() / self.width, self.width);
        let normalized = if self.use_postshuffle_norm {
            self.norm.forward(&hidden.reshape(merged)?)?
        } else {
            self.norm.forward(hidden)?.reshape(merged)?
        };

        let activated = functional::gelu(&self.linear_fc1.forward(&normalized)?)?;
        Ok(self.linear_fc2.forward(&activated)?)
    }
}

/// The Qwen3-VL vision tower, with the processor rules its images are sized and laid out by.
pub struct VisionTower {
    config: VisionConfig,
    processor: ProcessorConfig,
    pixels: PixelNorm,
    patch_embed: Linear,
    /// `[num_position_embeddings, hidden]`: the learned position table, a square grid.
    pos_embed: Tensor,
    rotary_pos_emb: Rotary,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    /// One merger per DeepStack block, in `deepstack_visual_indexes` order.
    deepstack_merger_list: Vec<PatchMerger>,
}

impl VisionTower {
    /// The tower of `config` at `vb` (`model.visual`), its images sized by `processor` and
    /// normalized by `pixels`.
    ///
    /// # Errors
    ///
    /// A processor whose patch sizes are not the tower's, or a module the loader refuses.
    pub fn load(
        config: &VisionConfig,
        vb: VarBuilder<'_, '_>,
        processor: ProcessorConfig,
        pixels: PixelNorm,
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
                "the processor patches {} over {} frames merging {}, the tower {} over {} \
                 merging {}",
                processor.patch_size,
                processor.temporal_patch_size,
                processor.merge_size,
                config.patch_size,
                config.temporal_patch_size,
                config.spatial_merge_size
            )));
        }

        // Each of the two axes turns `head_dim / 4` pairs by the same spectrum.
        let head_dim = config.head_dim();
        let rotary = RotaryConfig {
            theta: ROPE_THETA,
            dim: head_dim,
            spectrum: Spectrum::Repeated { axes: 2 },
            precision: Precision::F32,
        };
        let scale = rotary.score_scale(head_dim)?;

        let blocks = (0..config.depth)
            .map(|index| VisionBlock::load(config, vb.pp("blocks").pp(index), scale))
            .collect::<Result<Vec<_>>>()?;
        let deepstack_merger_list = (0..config.deepstack_visual_indexes.len())
            .map(|index| PatchMerger::load(config, vb.pp("deepstack_merger_list").pp(index), true))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            patch_embed: linear(
                vb.pp("patch_embed.proj"),
                config.tubelet(),
                config.hidden_size,
            )?,
            pos_embed: vb.pp("pos_embed").get(
                (
                    config.num_position_embeddings as usize,
                    config.hidden_size as usize,
                ),
                "weight",
            )?,
            rotary_pos_emb: Rotary::load(&rotary, vb.pp("rotary_pos_emb"))?,
            merger: PatchMerger::load(config, vb.pp("merger"), false)?,
            config: config.clone(),
            processor,
            pixels,
            blocks,
            deepstack_merger_list,
        })
    }

    /// The tower's configuration, the placeholder tokens its rows replace included.
    pub const fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// The patch rows of one block of `grid`, refused when a side is not whole merge blocks.
    fn patches(&self, grid: [u32; 3]) -> Result<u32> {
        positions::check_grid(grid, self.config.spatial_merge_size)?;
        let [time, height, width] = grid;

        time.checked_mul(height * width)
            .ok_or_else(|| Error::Config(format!("a grid of {grid:?} patches")))
    }

    /// The learned position of every patch of `grid`: `[patches, hidden]` in the table's
    /// element type.
    ///
    /// Each patch blends the four table rows around its point with FP32 weights; the blend
    /// accumulates in FP32 and rounds once. Every time step samples the same frame.
    fn position_embedding(&self, grid: [u32; 3]) -> Result<Tensor> {
        let [time, height, width] = grid;
        let (side, merge) = (self.config.position_side(), self.config.spatial_merge_size);
        let (frame, hidden) = ((height * width) as usize, self.config.hidden_size as usize);
        let device = self.pos_embed.device();

        let taps = positions::position_taps(side, height, width, merge)?.to_device(device)?;
        let weights = positions::position_weights(side, height, width, merge)?
            .unsqueeze(2)?
            .to_device(device)?;

        let rows = self
            .pos_embed
            .index_select(&taps.flatten_all()?, 0)?
            .reshape((4, frame, hidden))?
            .to_dtype(DType::F32)?;
        let blended = rows.broadcast_mul(&weights)?.sum(0)?;

        Ok(blended
            .to_dtype(self.pos_embed.dtype())?
            .repeat((time as usize, 1))?)
    }

    /// Encodes one block of `grid` from `pixels`, `[patches, tubelet]`: `[merged,
    /// out_hidden · (1 + deepstack)]`, each merged row's features followed by its DeepStack
    /// features in `deepstack_visual_indexes` order, as the reference returns them
    /// (`qwen3_vl.py:905-906`).
    ///
    /// # Errors
    ///
    /// Pixels that do not describe the grid, or what a module refuses.
    pub fn forward(&self, grid: [u32; 3], pixels: &Tensor) -> Result<Tensor> {
        let patches = self.patches(grid)?;
        if pixels.dims() != [patches as usize, self.config.tubelet() as usize] {
            return Err(Error::Config(format!(
                "pixels of {:?} for a {grid:?} grid",
                pixels.dims()
            )));
        }
        let [_, height, width] = grid;
        let device = pixels.device();

        let hidden = (self.patch_embed.forward(pixels)? + self.position_embedding(grid)?)?;

        let coordinates = positions::patch_coordinates(grid, self.config.spatial_merge_size)?
            .to_device(device)?;
        let (cos, sin) = self.rotary_pos_emb.forward(&coordinates)?;
        let offsets = positions::frame_offsets(grid)?.to_device(device)?;

        let mut hidden = hidden;
        let mut deepstack_features = Vec::with_capacity(self.deepstack_merger_list.len());
        for (index, block) in (0..).zip(&self.blocks) {
            hidden = block.forward(&hidden, (&cos, &sin), &offsets, height * width)?;

            let deepstack = self.config.deepstack_visual_indexes.iter();
            if let Some(merger) = deepstack
                .zip(&self.deepstack_merger_list)
                .find_map(|(block, merger)| (*block == index).then_some(merger))
            {
                deepstack_features.push(merger.forward(&hidden)?);
            }
        }

        let features = [vec![self.merger.forward(&hidden)?], deepstack_features].concat();
        Ok(Tensor::cat(&features, 1)?)
    }
}

impl Vision for VisionTower {
    fn image_grid(&self, height: u32, width: u32) -> nn::Result<[u32; 3]> {
        let grid = self
            .processor
            .image_grid(height, width)
            .map_err(|error| Error::Config(error.to_string()))?;

        Ok([grid.t, grid.h, grid.w])
    }

    fn patch_format(&self) -> PatchFormat {
        patch_format(&self.config, self.pixels, self.patch_embed.weight().dtype())
    }

    fn image_rows(&self, grid: [u32; 3]) -> nn::Result<u32> {
        if grid[0] != 1 {
            return Err(
                Error::Config(format!("an image is one time step, not the grid {grid:?}")).into(),
            );
        }
        let merge = self.config.spatial_merge_size;

        Ok(self.patches(grid)? / (merge * merge))
    }

    fn max_image_rows(&self) -> u32 {
        // A resized image's area is at most the processor's pixel budget, in whole merged
        // tokens.
        let token = u64::from(self.processor.patch_size * self.processor.merge_size).pow(2);

        u32::try_from(self.processor.image_max_pixels / token.max(1)).unwrap_or(u32::MAX)
    }

    fn position_axes(&self) -> u32 {
        POSITION_AXES
    }

    fn image_positions(&self, grid: [u32; 3], start: u32) -> nn::Result<Tensor> {
        self.image_rows(grid)?;

        Ok(positions::vision_block(
            grid,
            self.config.spatial_merge_size,
            start,
        )?)
    }

    fn encode_image(&self, grid: [u32; 3], pixels: &Tensor, out: &Tensor) -> nn::Result<()> {
        let rows = self.image_rows(grid)? as usize;
        let width = self.config.out_hidden_size as usize;
        if out.dims() != [rows, width] {
            return Err(Error::Config(format!(
                "an image of {rows} rows of {width} written into {:?}",
                out.dims()
            ))
            .into());
        }

        // A causal language model's tower states no DeepStack block, so the features are the
        // merged rows alone.
        let features = self.forward(grid, pixels).map_err(nn::Error::from)?;
        Ok(out.slice_set(&features.narrow(1, 0, width)?.contiguous()?, 0, 0)?)
    }
}
