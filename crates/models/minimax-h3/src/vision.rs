//! The Qwen3-VL vision tower: tubelet patches to language-width tokens.
//!
//! `uniserve/models/qwen3_vl/vision.py` is the reference, and `uniserve/nn/vision/position.py` the
//! position helper it composes. One image block's patches - a still image is one time step - are
//! projected by the patch embedding, given a learned 2-D position resampled to their grid, rotated
//! by a 2-D rotary position, encoded by `depth` pre-norm blocks, and merged so that every
//! `merge x merge` block of patches becomes one token of the language model's width. Three
//! intermediate blocks' hidden states are merged the same way and returned beside the last one:
//! these are the DeepStack features the language model adds at its own block boundaries.
//!
//! ## What the request decides, and where it enters
//!
//! Every value a request decides enters as a device table the encoder runner prepared on the host
//! from the request's grids (`crate::positions`, specs/worker.md 9): the position table's four row
//! indices and FP32 weights per patch, the 2-D rotary cosine and sine per patch, and the attention
//! lengths of the grid's time steps. Attention never crosses a time step - each temporal patch of a
//! grid is one bidirectional sequence of `height * width` patches - so those lengths are both the
//! query and the key offsets of the one dense attention call.
//!
//! The rotary factors cannot come from a sectioned `rotary_factors` call. The reference builds its
//! 2-D rotary with `head_dim / 2` as the *dimension* and no sections, and passes the two
//! coordinates as an ordinary trailing position axis: each axis therefore reads the whole
//! `head_dim / 4`-frequency spectrum with its own coordinate, so compact frequency
//! `a · (head_dim / 4) + f` of a patch at `(row, column)` turns by `coordinate(a) · inv_freq(f)`
//! (`uniserve/nn/rope.py::RotaryEmbedding._factors` over a `[patches, 2]` position). That is
//! neither the contiguous nor the interleaved M-RoPE cut, both of which give each axis its own band
//! of frequencies, so the factors are the runner's tables and
//! [`crate::positions::VisionRotary::factors`] computes them in the reference's FP32 formulas.
//!
//! ## Precision
//!
//! The tower declares one element type for its activations: the checkpoint's own representation,
//! BF16 for the released vision tower, which the loading boundary passes in as `dtype` and never
//! converts to. Two values keep their own types because their formulas have them: the position
//! table's bilinear blend accumulates in FP32 and rounds once into `dtype` (the reference's
//! `(table[indices] * weights).sum(0)` before `.to(hidden.dtype)`), and the rotary factors are FP32.
//!
//! ## The tubelet layout
//!
//! The patch embedding reads the processor's flat tubelets, channel-major
//! (`in_channels · temporal_patch · patch²`, the layout Qwen-VL image processors emit) against the
//! checkpoint's Conv3d weight flattened in the same order (`uniserve/nn/vision/modules.py:37-46`,
//! `TubeletEmbed`): the convolution's kernel equals its stride, so one matrix product over the
//! flattened tubelets evaluates it, and no permutation sits between the checkpoint and the product.
//!
//! ## The two GELUs
//!
//! The reference uses both of PyTorch's GELU forms: the block MLPs take the checkpoint's
//! `gelu_pytorch_tanh` (`VisionConfig::hidden_act`), the patch mergers a plain `nn.GELU()`, which is
//! the exact `erf` form. [`Gelu`] takes the approximation, and the tower holds one of each.

use nn::capability::ImageInput;
use nn::layers::{Attention, Gelu, LayerNorm, Linear, PositionInterpolate, Rotate, ScaledResidual};
use nn::weights::{WeightLoader, WeightRequest};
use nn::{Arena, DevicePtr, Forward, Shape, TensorView, Weight};
use ops::gemm::WeightOrder;
use ops::rope::Rotation;

use crate::config::{VisionActivation, VisionConfig};
use crate::positions::invalid;
use crate::{Error, Result};

/// Every normalization of the tower's variance floor: `LAYER_NORM_EPS` in the reference.
const LAYER_NORM_EPS: f32 = 1e-6;

/// The 2-D rotary frequency base (`ROPE_THETA` in the reference).
pub const ROPE_THETA: f32 = 10_000.0;

/// The checkpoint's tensor names under the text encoder's component.
const PREFIX: &str = "model.visual";

/// One image block's patch grid and the row counts its forward covers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Geometry {
    /// Temporal patches of the block: one for a still image.
    pub time: u32,
    /// Patch rows per time step.
    pub height: u32,
    /// Patch columns per time step.
    pub width: u32,
    /// Patch rows the block holds: `time * height * width`.
    pub patches: u32,
    /// Merged rows the block produces: `patches / merge²`, one language-width token each.
    pub merged: u32,
    /// Rows of one attention sequence: `height * width`, one sequence per time step.
    pub sequence: u32,
}

impl Geometry {
    /// The geometry of the grid `(time, height, width)` in patches.
    ///
    /// The grid is refused when a side is not a whole number of merge blocks (every merged token
    /// reads whole `merge x merge` blocks of patches) or when one time step holds more patches than
    /// the learned position table has entries: the table is the deployment's declared bound on an
    /// image's grid, and the attention descriptor's key bound follows it.
    pub fn of(config: &VisionConfig, grid: [u32; 3]) -> Result<Self> {
        let [time, height, width] = grid;
        let merge = config.spatial_merge_size;
        if time == 0 || height == 0 || width == 0 {
            return Err(invalid(
                "VisionTower",
                format!("the grid {grid:?} has an empty side"),
            ));
        }
        if !height.is_multiple_of(merge) || !width.is_multiple_of(merge) {
            return Err(invalid(
                "VisionTower",
                format!("the grid {height}x{width} is not whole {merge}x{merge} merge blocks"),
            ));
        }
        let patches = u64::from(time) * u64::from(height) * u64::from(width);
        let sequence = u64::from(height) * u64::from(width);
        let bound = u64::from(config.num_position_embeddings);
        if sequence > bound {
            return Err(invalid(
                "VisionTower",
                format!(
                    "one time step of the grid {height}x{width} holds {sequence} patches and the \
                     learned position table has {bound} entries"
                ),
            ));
        }
        let patches = u32::try_from(patches).map_err(|_| {
            invalid(
                "VisionTower",
                format!("the grid {grid:?} holds more patches than a u32 counts"),
            )
        })?;
        let sequence = u32::try_from(sequence)
            .map_err(|_| invalid("VisionTower", "an attention sequence longer than a u32"))?;
        Ok(Self {
            time,
            height,
            width,
            patches,
            merged: patches / (merge * merge),
            sequence,
        })
    }

    /// The tubelet width the patch embedding reads: `in_channels · temporal_patch_size · patch²`.
    pub fn tubelet(config: &VisionConfig) -> u32 {
        config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
    }

    /// The width of one merged row before the merger projects it: `merge² · hidden`.
    pub fn merged_width(config: &VisionConfig) -> u32 {
        config.hidden_size * config.spatial_merge_size * config.spatial_merge_size
    }

    /// The elements of one head: the hidden state split evenly between the heads.
    ///
    /// The checkpoint states no separate head width for the tower - the reference derives it the
    /// same way (`VisionConfig::head_dim` in `uniserve/models/qwen3_vl/config.py`).
    pub fn head_dim(config: &VisionConfig) -> Result<u32> {
        if config.num_heads == 0 || !config.hidden_size.is_multiple_of(config.num_heads) {
            return Err(invalid(
                "VisionTower",
                format!(
                    "{} heads do not split the {} element hidden state",
                    config.num_heads, config.hidden_size
                ),
            ));
        }
        Ok(config.hidden_size / config.num_heads)
    }

    /// The side of the square learned position table (`position_grid` in the reference).
    pub fn position_grid(config: &VisionConfig) -> Result<u32> {
        let side = u32::try_from(
            (u64::from(config.num_position_embeddings) as f64).sqrt().round() as u64,
        )
        .map_err(|_| invalid("VisionTower", "a position grid wider than a u32"))?;
        if u64::from(side) * u64::from(side) != u64::from(config.num_position_embeddings) {
            return Err(invalid(
                "VisionTower",
                format!(
                    "the {} entry position table is not a square grid",
                    config.num_position_embeddings
                ),
            ));
        }
        Ok(side)
    }
}

/// The DeepStack blocks, in checkpoint order, as layer indices.
///
/// The checkpoint states the blocks whose hidden state is merged and returned with the last one
/// (`deepstack_visual_indexes`: 8, 16, 24). The list is a contract, so a repeated, unsorted or
/// out-of-range index is refused rather than served in a surprising order: the language model adds
/// feature `j` after its own block `j`.
fn deepstack_layers(config: &VisionConfig) -> Result<Vec<usize>> {
    let mut layers: Vec<usize> = Vec::with_capacity(config.deepstack_visual_indexes.len());
    for &index in &config.deepstack_visual_indexes {
        if u64::from(index) >= u64::from(config.depth) {
            return Err(invalid(
                "VisionTower",
                format!("DeepStack block {index} of a {} block tower", config.depth),
            ));
        }
        if layers.last().is_some_and(|last| *last >= index as usize) {
            return Err(invalid(
                "VisionTower",
                format!(
                    "the DeepStack blocks {:?} are not increasing",
                    config.deepstack_visual_indexes
                ),
            ));
        }
        layers.push(index as usize);
    }
    Ok(layers)
}

/// The buffers one image block's forward reads and writes.
///
/// Every row count follows from the block's patch count, so a captured bucket reserves one set and
/// every replay addresses the same regions (`nn::Forward::arena`). The residual stream lives in
/// `hidden`: the patch embedding writes it, every block updates it in place, and the mergers read it.
pub struct Buffers {
    /// `[patches, hidden]` the residual stream.
    hidden: TensorView,
    /// `[patches, hidden]` one normalization, or the position blend the stream starts from.
    normalized: TensorView,
    /// `[patches, heads · head_dim]` the queries.
    query: TensorView,
    /// `[patches, heads · head_dim]` the keys.
    key: TensorView,
    /// `[patches, heads · head_dim]` the values.
    value: TensorView,
    /// `[patches, heads · head_dim]` the rotated queries.
    rotated_query: TensorView,
    /// `[patches, heads · head_dim]` the rotated keys.
    rotated_key: TensorView,
    /// `[patches, heads · head_dim]` the attended rows.
    attended: TensorView,
    /// `[patches, hidden]` the attention projection.
    projected: TensorView,
    /// `[patches, intermediate]` the MLP's first projection.
    gate: TensorView,
    /// `[patches, intermediate]` its activation.
    activated: TensorView,
    /// `[patches, hidden]` its second projection.
    down: TensorView,
    /// `[merged, merged_width]` a merger's first projection.
    merged: TensorView,
}

impl Buffers {
    /// Reserve the block's buffers in `arena` for a grid of `patches` patches.
    ///
    /// The reservations happen in one fixed order, so the plan an executor makes for a captured
    /// bucket is the plan every replay follows.
    pub fn new(arena: &Arena, config: &VisionConfig, patches: u32, dtype: nn::DType) -> Result<Self> {
        let rows = u64::from(patches);
        let hidden = u64::from(config.hidden_size);
        let intermediate = u64::from(config.intermediate_size);
        let merge = u64::from(config.spatial_merge_size);
        let blocks = rows / (merge * merge);
        let width = u64::from(Geometry::merged_width(config));
        let reserve = |shape: Shape| -> Result<TensorView> {
            arena.tensor(dtype, shape).map_err(Error::from)
        };
        Ok(Self {
            hidden: reserve(Shape::new(&[rows, hidden]))?,
            normalized: reserve(Shape::new(&[rows, hidden]))?,
            query: reserve(Shape::new(&[rows, hidden]))?,
            key: reserve(Shape::new(&[rows, hidden]))?,
            value: reserve(Shape::new(&[rows, hidden]))?,
            rotated_query: reserve(Shape::new(&[rows, hidden]))?,
            rotated_key: reserve(Shape::new(&[rows, hidden]))?,
            attended: reserve(Shape::new(&[rows, hidden]))?,
            projected: reserve(Shape::new(&[rows, hidden]))?,
            gate: reserve(Shape::new(&[rows, intermediate]))?,
            activated: reserve(Shape::new(&[rows, intermediate]))?,
            down: reserve(Shape::new(&[rows, hidden]))?,
            merged: reserve(Shape::new(&[blocks, width]))?,
        })
    }

    /// Use `view` as the residual stream instead of this block's own arena buffer.
    ///
    /// The diagnosis of specs/worker.md 10 dumps the tower's residual stream after its last block,
    /// which lives in an arena reservation whose address the runner does not know. Handing the
    /// runner's own view in as the stream costs nothing and changes no number: the same kernels
    /// write the same rows, at another address (`notes/h3/text-encoder-tables.md`).
    pub fn with_stream(mut self, view: TensorView) -> Result<Self> {
        let rows = self.hidden.rows();
        let columns = self.hidden.last();
        if view.shape != self.hidden.shape {
            return Err(invalid(
                "VisionTower",
                format!(
                    "the diagnostic stream is {:?}, [{rows}, {columns}] expected",
                    view.shape.dims()
                ),
            ));
        }
        self.hidden = view;
        Ok(self)
    }

    /// The bytes [`Buffers::new`] reserves for a grid of `patches` patches.
    ///
    /// The reservations are made against a region of exactly that size, so the number is the one
    /// the executor plans a captured bucket with and cannot drift from the reservations.
    pub fn bytes(config: &VisionConfig, patches: u32, dtype: nn::DType) -> Result<u64> {
        let arena = Arena::new(DevicePtr::new(0), u64::MAX);
        Self::new(&arena, config, patches, dtype)?;
        Ok(arena.used())
    }

    /// `view`'s rows as `[patches, heads, head_dim]` for the attention and rotation operators.
    fn as_heads(&self, view: TensorView, heads: u32, head_dim: u32) -> Result<TensorView> {
        let rows = view.rows();
        if view.last() != u64::from(heads) * u64::from(head_dim) {
            return Err(invalid(
                "VisionTower",
                format!(
                    "a {}-wide row does not split into {heads} heads of {head_dim}",
                    view.last()
                ),
            ));
        }
        Ok(view.reshape(Shape::new(&[
            rows,
            u64::from(heads),
            u64::from(head_dim),
        ])))
    }
}

/// Where one image block's features go.
pub struct Outputs<'a> {
    /// `[merged, out_hidden]`: the merger's language-width tokens.
    pub features: TensorView,
    /// One `[merged, out_hidden]` per DeepStack block, in checkpoint order.
    pub deepstack: &'a [TensorView],
}

/// One pre-norm block: attention and an ungated MLP, each added to the residual stream.
struct Block {
    input_norm: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    output_norm: LayerNorm,
    first: Linear,
    second: Linear,
}

impl Block {
    fn load(loader: &mut WeightLoader<'_>, config: &VisionConfig, index: u32) -> Result<Self> {
        let hidden = u64::from(config.hidden_size);
        let intermediate = u64::from(config.intermediate_size);
        let block = format!("{PREFIX}.blocks.{index}");
        Ok(Self {
            input_norm: normalization(loader, &format!("{block}.norm1"), hidden)?,
            query: fused_projection(loader, &format!("{block}.attn.qkv"), 0, hidden)?,
            key: fused_projection(loader, &format!("{block}.attn.qkv"), 1, hidden)?,
            value: fused_projection(loader, &format!("{block}.attn.qkv"), 2, hidden)?,
            output: projection(loader, &format!("{block}.attn.proj"), hidden, hidden)?,
            output_norm: normalization(loader, &format!("{block}.norm2"), hidden)?,
            first: projection(loader, &format!("{block}.mlp.linear_fc1"), intermediate, hidden)?,
            second: projection(loader, &format!("{block}.mlp.linear_fc2"), hidden, intermediate)?,
        })
    }

    /// Run the block over `buffers.hidden` in place.
    fn forward(
        &self,
        tower: &VisionTower,
        f: &Forward,
        image: &ImageInput,
        buffers: &Buffers,
    ) -> Result<()> {
        let heads = tower.config.num_heads;
        let head_dim = Geometry::head_dim(&tower.config)?;
        let hidden = buffers.hidden;

        self.input_norm
            .forward(f, hidden, buffers.normalized)
            .map_err(Error::from)?;
        self.query
            .forward(f, buffers.normalized, buffers.query)
            .map_err(Error::from)?;
        self.key
            .forward(f, buffers.normalized, buffers.key)
            .map_err(Error::from)?;
        self.value
            .forward(f, buffers.normalized, buffers.value)
            .map_err(Error::from)?;

        // Both projections rotate by the patch's own 2-D factors; nothing normalizes them.
        for (unrotated, rotated) in [
            (buffers.query, buffers.rotated_query),
            (buffers.key, buffers.rotated_key),
        ] {
            tower.rotate.apply(
                f,
                buffers.as_heads(unrotated, heads, head_dim)?,
                image.cosine,
                image.sine,
                buffers.as_heads(rotated, heads, head_dim)?,
            )?;
        }

        tower.attention.attend_dense(
            f,
            buffers.as_heads(buffers.rotated_query, heads, head_dim)?,
            buffers.as_heads(buffers.rotated_key, heads, head_dim)?,
            buffers.as_heads(buffers.value, heads, head_dim)?,
            image.lengths,
            image.lengths,
            buffers.as_heads(buffers.attended, heads, head_dim)?,
        )?;
        self.output
            .forward(f, buffers.attended, buffers.projected)
            .map_err(Error::from)?;
        tower
            .residual
            .forward(f, hidden, buffers.projected)
            .map_err(Error::from)?;

        self.output_norm
            .forward(f, hidden, buffers.normalized)
            .map_err(Error::from)?;
        self.first
            .forward(f, buffers.normalized, buffers.gate)
            .map_err(Error::from)?;
        tower
            .blocks_gelu
            .forward(f, buffers.gate, buffers.activated)?;
        self.second
            .forward(f, buffers.activated, buffers.down)
            .map_err(Error::from)?;
        tower
            .residual
            .forward(f, hidden, buffers.down)
            .map_err(Error::from)
    }
}

/// A patch merger: normalize, concatenate every `merge²` patch rows into one, project twice.
///
/// The two variants differ only in where the normalization sits, and the reference states the
/// difference as `normalize_merged`: the final merger normalizes each patch *before* the
/// concatenation (`normalize_merged = false`), a DeepStack merger normalizes the concatenated row
/// (`true`).
struct Merger {
    norm: LayerNorm,
    first: Linear,
    second: Linear,
    merge: u32,
    width: u32,
    out_hidden: u32,
    normalizes_merged: bool,
}

impl Merger {
    fn load(
        loader: &mut WeightLoader<'_>,
        name: &str,
        config: &VisionConfig,
        normalizes_merged: bool,
    ) -> Result<Self> {
        let width = u64::from(Geometry::merged_width(config));
        let out_hidden = u64::from(config.out_hidden_size);
        let norm_width = if normalizes_merged {
            width
        } else {
            u64::from(config.hidden_size)
        };
        Ok(Self {
            norm: normalization(loader, &format!("{name}.norm"), norm_width)?,
            first: projection(loader, &format!("{name}.linear_fc1"), width, width)?,
            second: projection(loader, &format!("{name}.linear_fc2"), out_hidden, width)?,
            merge: config.spatial_merge_size,
            width: u32::try_from(width)
                .map_err(|_| invalid("VisionTower", "a merged row wider than a u32"))?,
            out_hidden: u32::try_from(out_hidden)
                .map_err(|_| invalid("VisionTower", "an out_hidden wider than a u32"))?,
            normalizes_merged,
        })
    }

    /// Merge `hidden`'s patch rows into the rows of `out`.
    fn forward(
        &self,
        tower: &VisionTower,
        f: &Forward,
        hidden: TensorView,
        buffers: &Buffers,
        out: TensorView,
    ) -> Result<()> {
        let rows = hidden.rows();
        let block = u64::from(self.merge) * u64::from(self.merge);
        if !rows.is_multiple_of(block) {
            return Err(invalid(
                "VisionTower",
                format!("{rows} patch rows are not whole {block}-patch merge blocks"),
            ));
        }
        let merged = Shape::new(&[rows / block, u64::from(self.width)]);
        if out.rows() != merged.dim(0) || out.last() != u64::from(self.out_hidden) {
            return Err(invalid(
                "VisionTower",
                format!(
                    "a merger writes {} rows of {} but the output is {} rows of {}",
                    merged.dim(0),
                    self.out_hidden,
                    out.rows(),
                    out.last()
                ),
            ));
        }

        // The variants differ in whether the rows are concatenated before the normalization or
        // after it; the normalized values are `[blocks, width]` either way.
        if self.normalizes_merged {
            self.norm
                .forward(
                    f,
                    hidden.reshape(merged),
                    buffers.normalized.reshape(merged),
                )
                .map_err(Error::from)?;
        } else {
            self.norm
                .forward(f, hidden, buffers.normalized)
                .map_err(Error::from)?;
        }
        self.first
            .forward(
                f,
                buffers.normalized.reshape(merged),
                buffers.merged.reshape(merged),
            )
            .map_err(Error::from)?;
        tower
            .merger_gelu
            .forward(f, buffers.merged.reshape(merged), buffers.gate.reshape(merged))?;
        self.second
            .forward(f, buffers.gate.reshape(merged), out)
            .map_err(Error::from)
    }
}

/// The vision tower of the Qwen3-VL text encoder.
pub struct VisionTower {
    config: VisionConfig,
    dtype: nn::DType,
    patch: Linear,
    position: Weight,
    /// The learned table's side and the side it is resampled from: one square grid.
    position_grid: u32,
    attention: Attention,
    rotate: Rotate,
    layers: Vec<Block>,
    /// The layers whose hidden state a DeepStack merger reads, in checkpoint order.
    deepstack_layers: Vec<usize>,
    merger: Merger,
    deepstack: Vec<Merger>,
    blocks_gelu: Gelu,
    merger_gelu: Gelu,
    residual: ScaledResidual,
}

impl VisionTower {
    /// The tower of `config`, reading the checkpoint through `loader`.
    ///
    /// `dtype` is the activation element type the loading boundary chose: the checkpoint's own
    /// representation unless the deployment asked for another, and every weight is read in it.
    pub fn load(loader: &mut WeightLoader<'_>, config: &VisionConfig, dtype: nn::DType) -> Result<Self> {
        let hidden = u64::from(config.hidden_size);
        let tubelet = u64::from(Geometry::tubelet(config));
        // The checkpoint's activation, and the merger's exact form, are two different GELUs.
        let blocks_gelu = match config.hidden_act {
            VisionActivation::GeluTanh => Gelu::tanh(),
        };
        let heads = config.num_heads;
        let head_dim = Geometry::head_dim(config)?;
        // The position table must be a square grid: its rows are indexed by `row · side + column`.
        let position_grid = Geometry::position_grid(config)?;
        let layers = (0..config.depth)
            .map(|index| Block::load(loader, config, index))
            .collect::<Result<Vec<_>>>()?;
        let deepstack = (0..config.deepstack_visual_indexes.len())
            .map(|index| {
                Merger::load(
                    loader,
                    &format!("{PREFIX}.deepstack_merger_list.{index}"),
                    config,
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // The position table's entries bound one time step's patch count, and with it the keys the
        // tower's one dense attention call reads. The bound is a descriptor value of the
        // deployment (specs/worker.md 9), so it is fixed here and not per call.
        let bound = config.num_position_embeddings;
        let attention =
            Attention::dense(head_dim, heads, heads, bound, false, None, (head_dim as f32).sqrt().recip())
                .map_err(Error::from)?;
        Ok(Self {
            deepstack_layers: deepstack_layers(config)?,
            patch: projection(loader, &format!("{PREFIX}.patch_embed.proj"), hidden, tubelet)?,
            position: region(
                loader,
                &format!("{PREFIX}.pos_embed.weight"),
                Shape::new(&[u64::from(bound), hidden]),
            )?,
            position_grid,
            attention,
            rotate: Rotate::new(head_dim, heads, Rotation::Half).map_err(Error::from)?,
            layers,
            merger: Merger::load(loader, &format!("{PREFIX}.merger"), config, false)?,
            deepstack,
            blocks_gelu,
            merger_gelu: Gelu::exact(),
            residual: ScaledResidual::new(
                ones(
                    loader,
                    "minimax-h3.vision.residual",
                    dtype,
                    u64::from(config.hidden_size),
                )?,
                None,
            )
            .map_err(Error::from)?,
            config: config.clone(),
            dtype,
        })
    }

    /// The tower's configuration.
    pub const fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// The element type of the tower's activations.
    pub const fn dtype(&self) -> nn::DType {
        self.dtype
    }

    /// How many DeepStack feature tables one block returns.
    pub fn deepstack_features(&self) -> usize {
        self.deepstack.len()
    }

    /// Reserve the buffers of a block of `patches` patches.
    pub fn buffers(&self, arena: &Arena, patches: u32) -> Result<Buffers> {
        Buffers::new(arena, &self.config, patches, self.dtype)
    }

    /// The bytes a block of `patches` patches reserves in the arena.
    pub fn bytes(&self, patches: u32) -> Result<u64> {
        Buffers::bytes(&self.config, patches, self.dtype)
    }

    /// Encode one image block's patches into `out`.
    ///
    /// `image` carries the patches and the four prepared tables of its grid; `out` receives tokens
    /// of the language model's width, `merge²` patches each: the tower's own features and one table
    /// per DeepStack block. One call per block, in prompt order.
    pub fn encode(
        &self,
        f: &Forward,
        image: &ImageInput,
        buffers: &Buffers,
        out: Outputs<'_>,
    ) -> Result<()> {
        let geometry = Geometry::of(&self.config, image.grid_shape)?;
        if out.deepstack.len() != self.deepstack.len() {
            return Err(invalid(
                "VisionTower",
                format!(
                    "{} DeepStack outputs for a checkpoint that names {}",
                    out.deepstack.len(),
                    self.deepstack.len()
                ),
            ));
        }
        self.check(image, &geometry, buffers, &out)?;

        // The patch embedding reads the tubelets the processor packed; the position blend is added
        // into the same rows, and the residual stream starts there.
        self.patch
            .forward(f, image.pixels, buffers.hidden)
            .map_err(Error::from)?;
        // The resample's descriptor names both grids, so the layer is built for the image's own
        // (rectangular) patch grid; the taps and weights are the request's tables.
        PositionInterpolate::new(
            self.dtype,
            u64::from(self.config.hidden_size),
            self.position_grid,
            geometry.height,
            geometry.width,
        )
        .map_err(Error::from)?
        .forward(
            f,
            self.position.view(),
            image.position_index,
            image.position_weight,
            buffers.normalized,
        )
        .map_err(Error::from)?;
        self.residual
            .forward(f, buffers.hidden, buffers.normalized)
            .map_err(Error::from)?;

        let mut next = 0;
        for (index, block) in self.layers.iter().enumerate() {
            block.forward(self, f, image, buffers)?;
            if self
                .deepstack_layers
                .get(next)
                .is_some_and(|layer| *layer == index)
            {
                self.deepstack[next].forward(self, f, buffers.hidden, buffers, out.deepstack[next])?;
                next += 1;
            }
        }
        self.merger
            .forward(self, f, buffers.hidden, buffers, out.features)
    }

    /// Refuse a block whose prepared tables, buffers or outputs do not describe its grid.
    fn check(
        &self,
        image: &ImageInput,
        geometry: &Geometry,
        buffers: &Buffers,
        out: &Outputs<'_>,
    ) -> Result<()> {
        let patches = u64::from(geometry.patches);
        let merged = u64::from(geometry.merged);
        let hidden = u64::from(self.config.hidden_size);
        let columns = u64::from(Geometry::head_dim(&self.config)? / 2);
        let tubelet = u64::from(Geometry::tubelet(&self.config));
        let width = u64::from(Geometry::merged_width(&self.config));
        let out_hidden = u64::from(self.config.out_hidden_size);
        let tables = [
            ("pixels", image.pixels, Shape::new(&[patches, tubelet])),
            ("position_index", image.position_index, Shape::new(&[patches, 4])),
            ("position_weight", image.position_weight, Shape::new(&[patches, 4])),
            ("cosine", image.cosine, Shape::new(&[patches, columns])),
            ("sine", image.sine, Shape::new(&[patches, columns])),
            ("grid", image.grid, Shape::new(&[1, 3])),
            ("features", out.features, Shape::new(&[merged, out_hidden])),
        ];
        for (name, view, shape) in tables {
            if view.shape != shape {
                return Err(invalid(
                    "VisionTower",
                    format!(
                        "the {name} table is {:?} for a {:?} block, {shape:?} expected",
                        view.shape.dims(),
                        image.grid_shape
                    ),
                ));
            }
        }
        for deepstack in out.deepstack {
            if deepstack.shape != Shape::new(&[merged, out_hidden]) {
                return Err(invalid(
                    "VisionTower",
                    format!(
                        "a DeepStack output is {:?} for a {:?} block, [{merged}, {out_hidden}] \
                         expected",
                        deepstack.shape.dims(),
                        image.grid_shape
                    ),
                ));
            }
        }
        let lengths = u64::from(geometry.time) + 1;
        if image.lengths.numel() != lengths {
            return Err(invalid(
                "VisionTower",
                format!(
                    "{} attention lengths for {} time steps",
                    image.lengths.numel(),
                    geometry.time
                ),
            ));
        }
        if image.cosine.dtype != nn::DType::F32 || image.sine.dtype != nn::DType::F32 {
            return Err(invalid(
                "VisionTower",
                "the rotary factors are FP32".to_owned(),
            ));
        }
        if buffers.hidden.rows() != patches
            || buffers.hidden.last() != hidden
            || buffers.merged.rows() != merged
            || buffers.merged.last() != width
        {
            return Err(invalid(
                "VisionTower",
                format!(
                    "the block's buffers hold {} rows of {} and {} merged rows of {}, the {:?} \
                     grid needs {patches}, {hidden}, {merged} and {width}",
                    buffers.hidden.rows(),
                    buffers.hidden.last(),
                    buffers.merged.rows(),
                    buffers.merged.last(),
                    image.grid_shape
                ),
            ));
        }
        Ok(())
    }
}

/// A region of one checkpoint tensor, in the operator's NxK order.
fn region(loader: &mut WeightLoader<'_>, name: &str, shape: Shape) -> Result<Weight> {
    loader
        .load(&WeightRequest::region(name, 0, shape))
        .map_err(Error::from)
}

/// A projection whose checkpoint tensor is `[outputs, inputs]` (`n · k`).
fn projection(
    loader: &mut WeightLoader<'_>,
    name: &str,
    outputs: u64,
    inputs: u64,
) -> Result<Linear> {
    let weight = region(
        loader,
        &format!("{name}.weight"),
        Shape::new(&[outputs, inputs]),
    )?;
    let bias = region(loader, &format!("{name}.bias"), Shape::new(&[outputs]))?;
    Linear::new(weight, Some(bias), WeightOrder::NxK).map_err(Error::from)
}

/// One branch of a fused `qkv` projection: branch 0, 1 and 2 are `q`, `k` and `v`.
///
/// The checkpoint fuses the three `[hidden, hidden]` matrices into one `[3 · hidden, hidden]`
/// tensor, `q` first, so each branch is a contiguous region of it: three calls read the same bytes
/// as the reference's one fused product.
fn fused_projection(
    loader: &mut WeightLoader<'_>,
    name: &str,
    branch: u64,
    hidden: u64,
) -> Result<Linear> {
    let weight = loader
        .load(&WeightRequest::region(
            format!("{name}.weight"),
            branch * hidden * hidden,
            Shape::new(&[hidden, hidden]),
        ))
        .map_err(Error::from)?;
    let bias = loader
        .load(&WeightRequest::region(
            format!("{name}.bias"),
            branch * hidden,
            Shape::new(&[hidden]),
        ))
        .map_err(Error::from)?;
    Linear::new(weight, Some(bias), WeightOrder::NxK).map_err(Error::from)
}

/// A LayerNorm of `width` channels, with the tower's one variance floor.
fn normalization(loader: &mut WeightLoader<'_>, name: &str, width: u64) -> Result<LayerNorm> {
    let weight = region(loader, &format!("{name}.weight"), Shape::new(&[width]))?;
    let bias = region(loader, &format!("{name}.bias"), Shape::new(&[width]))?;
    LayerNorm::new(weight, Some(bias), LAYER_NORM_EPS).map_err(Error::from)
}

/// A retained weight of `columns` ones: the block residual stream's coefficient.
///
/// The fused residual operator reads its coefficient as a weight; these blocks add each sub-layer's
/// output with coefficient one, so the constant is published once per element type and width under
/// the retained key `name`.
pub(crate) fn ones(
    loader: &mut WeightLoader<'_>,
    name: &str,
    dtype: nn::DType,
    columns: u64,
) -> Result<Weight> {
    let element = dtype.bytes().ok_or_else(|| {
        invalid(
            "VisionTower",
            format!("a residual coefficient of {dtype} has no whole-byte element"),
        )
    })?;
    let value: Vec<u8> = match dtype {
        nn::DType::F32 => 1.0_f32.to_le_bytes().to_vec(),
        nn::DType::BF16 => (1.0_f32.to_bits() >> 16).to_le_bytes().to_vec(),
        nn::DType::F16 => half_bits(1.0).to_le_bytes().to_vec(),
        other => {
            return Err(invalid(
                "VisionTower",
                format!("a residual coefficient of {other} is not a whole-byte float"),
            ));
        }
    };
    debug_assert_eq!(value.len(), usize::try_from(element).unwrap_or(0));
    let mut bytes = Vec::with_capacity(value.len() * usize::try_from(columns).unwrap_or(0));
    for _ in 0..columns {
        bytes.extend_from_slice(&value);
    }
    loader
        .constant(name, dtype, Shape::new(&[columns]), &bytes)
        .map_err(Error::from)
}

/// The FP16 bit pattern of a value in the normal range.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3ff;
    ((exponent as u16) << 10) | mantissa as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released vision tower's configuration.
    fn config() -> VisionConfig {
        crate::config::tests_support::released_config()
            .text_encoder
            .vision
    }

    #[test]
    fn one_image_is_one_time_step_of_whole_merge_blocks() {
        let config = config();
        let geometry = Geometry::of(&config, [1, 24, 24]).expect("a 24x24 patch grid");
        assert_eq!(geometry.patches, 576);
        assert_eq!(geometry.merged, 144);
        assert_eq!(geometry.sequence, 576);
        assert_eq!(Geometry::tubelet(&config), 1536);
        assert_eq!(Geometry::merged_width(&config), 4608);
    }

    #[test]
    fn a_video_block_keeps_one_sequence_per_time_step() {
        let config = config();
        let geometry = Geometry::of(&config, [4, 16, 16]).expect("four time steps");
        assert_eq!(geometry.patches, 1024);
        assert_eq!(geometry.merged, 256);
        assert_eq!(geometry.sequence, 256);
    }

    #[test]
    fn a_grid_outside_whole_merge_blocks_is_refused() {
        let config = config();
        assert!(Geometry::of(&config, [1, 23, 24]).is_err());
        assert!(Geometry::of(&config, [1, 24, 0]).is_err());
    }

    #[test]
    fn a_time_step_larger_than_the_position_table_is_refused() {
        let config = config();
        // 48 x 48 is the table's own grid; 50 x 50 patches have no entry for every row.
        assert!(Geometry::of(&config, [1, 48, 48]).is_ok());
        assert!(Geometry::of(&config, [1, 50, 50]).is_err());
    }

    #[test]
    fn the_deepstack_blocks_are_the_checkpoints_increasing_list() {
        let config = config();
        assert_eq!(
            deepstack_layers(&config).expect("the released list"),
            [8, 16, 24]
        );
        let mut repeated = config.clone();
        repeated.deepstack_visual_indexes = vec![8, 8];
        assert!(deepstack_layers(&repeated).is_err());
        let mut late = config.clone();
        late.deepstack_visual_indexes = vec![config.depth];
        assert!(deepstack_layers(&late).is_err());
    }

    #[test]
    fn the_buffers_of_a_block_grow_with_its_patches() {
        let config = config();
        let dtype = nn::DType::BF16;
        let small = Buffers::bytes(&config, 576, dtype).expect("a 24x24 block");
        let large = Buffers::bytes(&config, 1024, dtype).expect("a 4x16x16 block");
        assert!(large > small);
        // Ten of the thirteen reservations are `patches · hidden` rows, so the region cannot be
        // smaller than ten of them; and every reservation starts on the arena's alignment, so the
        // total is a multiple of it and larger than the rows alone.
        let hidden = 576 * u64::from(config.hidden_size) * 2;
        assert!(small > 10 * hidden);
    }

    #[test]
    fn the_half_pattern_of_a_power_of_two_is_its_exponent() {
        assert_eq!(half_bits(1.0), 0x3c00);
        assert_eq!(half_bits(2.0), 0x4000);
        assert_eq!(half_bits(0.5), 0x3800);
    }
}
