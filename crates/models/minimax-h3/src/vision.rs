//! The Qwen3-VL vision tower: tubelet patches to language-width tokens.
//!
//! The tower is Qwen3-VL's, with its learned position table resampled to each grid. One image
//! block's patches - a still image is one time step - are
//! projected by the patch embedding, given a learned 2-D position resampled to their grid, rotated
//! by a 2-D rotary position, encoded by `depth` pre-norm blocks, and merged so that every
//! `merge x merge` block of patches becomes one token of the language model's width. Three
//! intermediate blocks' hidden states are merged the same way and returned beside the last one:
//! these are the DeepStack features the language model adds at its own block boundaries.
//!
//! ## What the request decides, and where it enters
//!
//! Every value a request decides enters as a device table the encoder runner prepared on the host
//! from the request's grids ([`crate::positions`]): the position table's four row
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
//! over a `[patches, 2]` position. That is
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
//! The patch embedding reads the processor's flat tubelets, channel-major (`in_channels ·
//! temporal_patch · patch²`, the layout Qwen-VL image processors emit) against the checkpoint's
//! Conv3d weight flattened in the same order: the convolution's kernel equals its stride, so one
//! matrix product over the flattened tubelets evaluates it, and no permutation sits between the
//! checkpoint and the product.
//!
//! ## The two GELUs
//!
//! The reference uses both of PyTorch's GELU forms: the block MLPs take the checkpoint's
//! `gelu_pytorch_tanh` (`VisionConfig::hidden_act`), the patch mergers a plain `nn.GELU()`, which is
//! the exact `erf` form. [`Gelu`] takes the approximation, and the tower holds one of each.

use nn::capability::ImageInput;
use nn::family::Split;
use nn::layers::{
    Attention, Bias, Gelu, LayerNorm, Linear, Parallel, PositionInterpolate, QkvBias, QkvLinear,
    QkvNames, QkvSpec, Residual, Rotate,
};
use nn::weights::{Scope, WeightLoader};
use nn::{Arena, Forward, Shape, TensorView, Weight};

use crate::config::VisionConfig;
use crate::positions::invalid;
use crate::{Error, Result};

/// Every normalization of the tower's variance floor: `LAYER_NORM_EPS` in the reference.
const LAYER_NORM_EPS: f32 = 1e-6;

/// The 2-D rotary frequency base (`ROPE_THETA` in the reference).
pub const ROPE_THETA: f32 = 10_000.0;

/// The checkpoint's tensor names under the text encoder's component.
const PREFIX: &str = "model.visual";

/// One image block's row counts, as its forward and its tables see them.
///
/// The grid's two spatial extents are a host fact of the *request*: they are known when the tables
/// are prepared (`BlockRows::of`) and are carried by the tables themselves at call time, because a
/// model may not read `ImageInput.grid` back and a descriptor that only counts points does not need
/// them (`BlockRows::of_counts`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockRows {
    /// Temporal patches of the block: one for a still image.
    pub time: u32,
    /// Patch rows the block holds: `time * height * width`.
    pub patches: u32,
    /// Merged rows the block produces: `patches / merge²`, one language-width token each.
    pub merged: u32,
    /// Rows of one attention sequence: `height * width`, one sequence per time step.
    pub sequence: u32,
}

impl BlockRows {
    /// The row counts of the grid `(time, height, width)` in patches.
    ///
    /// The grid is refused when a side is not a whole number of merge blocks (every merged token
    /// reads whole `merge x merge` blocks of patches). Nothing else bounds it here: the learned
    /// position table is resampled to the grid, and the tower's attention descriptor takes each
    /// time step's own `height · width` as its key bound.
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
        // The learned table bounds nothing: it is resampled to whatever grid an image has
        // (`PositionEmbedding.interpolate`), and the processor's own pixel cap is far above the
        // table's entry count - H3's 768x1344 keyframe is 48x84 patches against a 48x48 table.
        let patches = u64::from(time) * u64::from(height) * u64::from(width);
        let sequence = u64::from(height) * u64::from(width);
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
            patches,
            merged: patches / (merge * merge),
            sequence,
        })
    }

    /// The row counts a call's tables describe: `patches` rows of one time step each, over `time`
    /// time steps.
    ///
    /// The tables are what a captured call has; their row counts fix `patches` (`cosine`'s rows)
    /// and `time` (`lengths`'). The spatial split is not needed: every table already holds the grid.
    pub fn of_counts(config: &VisionConfig, patches: u32, time: u32) -> Result<Self> {
        let merge = config.spatial_merge_size;
        if time == 0 || patches == 0 {
            return Err(invalid(
                "VisionTower",
                format!("a block of {patches} patches over {time} time steps"),
            ));
        }
        if !patches.is_multiple_of(time) {
            return Err(invalid(
                "VisionTower",
                format!("{patches} patches do not spread over {time} time steps"),
            ));
        }
        let sequence = patches / time;
        if !sequence.is_multiple_of(merge * merge) {
            return Err(invalid(
                "VisionTower",
                format!(
                    "a time step of {sequence} patches is not whole {merge}x{merge} merge blocks"
                ),
            ));
        }
        Ok(Self {
            time,
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
    /// The checkpoint states no separate head width for the tower, so it is the hidden width over
    /// the head count.
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
            (u64::from(config.num_position_embeddings) as f64)
                .sqrt()
                .round() as u64,
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
    /// `[merged, merged_width]` its activation: a merged row is `merge² · hidden` wide, which the
    /// MLP's `[patches, intermediate]` buffers do not fit.
    merged_activated: TensorView,
}

impl Buffers {
    /// Reserve the block's buffers in `arena` for a grid of `patches` patches.
    ///
    /// The reservations happen in one fixed order, so the plan a worker makes for a captured
    /// bucket is the plan every replay follows.
    pub fn new(
        arena: &Arena,
        config: &VisionConfig,
        patches: u32,
        dtype: nn::DType,
    ) -> Result<Self> {
        let rows = u64::from(patches);
        let hidden = u64::from(config.hidden_size);
        let intermediate = u64::from(config.intermediate_size);
        let merge = u64::from(config.spatial_merge_size);
        let blocks = rows / (merge * merge);
        let width = u64::from(BlockRows::merged_width(config));
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
            merged_activated: reserve(Shape::new(&[blocks, width]))?,
        })
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
        Ok(view.reshape(Shape::new(&[rows, u64::from(heads), u64::from(head_dim)])))
    }
}

/// The prepared tables of one image block: what the request's grid becomes on the device.
///
/// Every value depends only on the block's patch grid, so the runner uploads one set per layout
/// (`Encoder::state_tables`) and a captured call reads them. The two spatial extents are not among
/// them: each table already carries the grid, and a model never reads a device view back.
#[derive(Clone, Copy)]
pub struct ImageTables {
    /// `[patches, 4]` I32: the learned table rows each patch blends (`positions::position_samples`).
    pub position_index: TensorView,
    /// `[patches, 4]` F32: their bilinear weights.
    pub position_weight: TensorView,
    /// `[patches, head_dim / 2]` F32: each patch's 2-D rotary cosine.
    pub cosine: TensorView,
    /// `[patches, head_dim / 2]` F32: its sine.
    pub sine: TensorView,
    /// `[time + 1]` I32: the attention offsets, one sequence per time step (`positions::vision_lengths`).
    pub lengths: TensorView,
    /// The block's row counts, as the tables state them.
    pub rows: BlockRows,
    /// The patches the block reads, `[patches, tubelet]`.
    pub pixels: TensorView,
}

impl ImageTables {
    /// The tables of `image`, whose row counts are `patches` over `time` time steps.
    ///
    /// The caller states the counts because a table's *shape* is a host value it already holds
    /// (`EncoderLayout`); the model checks that every table agrees with them.
    pub fn new(
        config: &VisionConfig,
        image: &ImageInput,
        patches: u32,
        time: u32,
        tables: &[TensorView],
    ) -> Result<Self> {
        let [position_index, position_weight, cosine, sine, lengths] = tables else {
            return Err(invalid(
                "VisionTower",
                format!(
                    "{} prepared tables for one image, five expected",
                    tables.len()
                ),
            ));
        };
        Ok(Self {
            position_index: *position_index,
            position_weight: *position_weight,
            cosine: *cosine,
            sine: *sine,
            lengths: *lengths,
            rows: BlockRows::of_counts(config, patches, time)?,
            pixels: image.pixels,
        })
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
    /// The query, key and value windows of the checkpoint's fused `attn.qkv`.
    qkv: QkvLinear,
    output: Linear,
    output_norm: LayerNorm,
    first: Linear,
    second: Linear,
}

impl Block {
    /// The block the module at `scope` states.
    fn load(scope: &mut Scope<'_, '_>, config: &VisionConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let head_dim = BlockRows::head_dim(config)?;
        let mut attention = scope.scope("attn");
        // The checkpoint fuses the three `[hidden, hidden]` matrices into one `qkv` module, query
        // first; each projection is its own window of it, so three products read the same bytes
        // as the reference's one fused product.
        let qkv = QkvLinear::load(
            &mut attention,
            QkvSpec {
                hidden,
                heads: config.num_heads,
                kv_heads: config.num_heads,
                head_dim,
            },
            QkvNames::Fused("qkv"),
            QkvBias::Present,
            Split::WHOLE,
        )?;
        let output = Linear::load(
            &mut attention.scope("proj"),
            hidden,
            hidden,
            Bias::Present,
            Parallel::None,
        )?;
        let mut mlp = scope.scope("mlp");
        let first = Linear::load(
            &mut mlp.scope("linear_fc1"),
            hidden,
            intermediate,
            Bias::Present,
            Parallel::None,
        )?;
        let second = Linear::load(
            &mut mlp.scope("linear_fc2"),
            intermediate,
            hidden,
            Bias::Present,
            Parallel::None,
        )?;
        Ok(Self {
            input_norm: LayerNorm::load(&mut scope.scope("norm1"), hidden, LAYER_NORM_EPS)?,
            qkv,
            output,
            output_norm: LayerNorm::load(&mut scope.scope("norm2"), hidden, LAYER_NORM_EPS)?,
            first,
            second,
        })
    }

    /// Run the block over `buffers.hidden` in place.
    fn forward(
        &self,
        tower: &VisionTower,
        f: &Forward,
        image: &ImageTables,
        attention: &Attention,
        rows: &BlockRows,
        buffers: &Buffers,
    ) -> Result<()> {
        let heads = tower.config.num_heads;
        let head_dim = BlockRows::head_dim(&tower.config)?;
        let hidden = buffers.hidden;

        self.input_norm
            .forward(f, hidden, None, buffers.normalized)
            .map_err(Error::from)?;
        self.qkv.forward(
            f,
            buffers.normalized,
            buffers.query,
            buffers.key,
            buffers.value,
        )?;

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

        // One sequence per time step, packed in one launch: the key bound is every step together
        // and the longest row is one step, so a video block of several steps launches as an image.
        attention.attend_packed(
            f,
            buffers.as_heads(buffers.rotated_query, heads, head_dim)?,
            buffers.as_heads(buffers.rotated_key, heads, head_dim)?,
            buffers.as_heads(buffers.value, heads, head_dim)?,
            image.lengths,
            rows.patches,
            rows.sequence,
            buffers.as_heads(buffers.attended, heads, head_dim)?,
        )?;
        self.output
            .forward(f, buffers.attended, buffers.projected)
            .map_err(Error::from)?;
        tower
            .residual
            .forward(f, hidden, buffers.projected, None, hidden)
            .map_err(Error::from)?;

        self.output_norm
            .forward(f, hidden, None, buffers.normalized)
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
            .forward(f, hidden, buffers.down, None, hidden)
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
    /// The merger the module at `scope` states.
    fn load(
        scope: &mut Scope<'_, '_>,
        config: &VisionConfig,
        normalizes_merged: bool,
    ) -> Result<Self> {
        let width = BlockRows::merged_width(config);
        let out_hidden = config.out_hidden_size;
        let norm_width = if normalizes_merged {
            width
        } else {
            config.hidden_size
        };
        Ok(Self {
            norm: LayerNorm::load(&mut scope.scope("norm"), norm_width, LAYER_NORM_EPS)?,
            first: Linear::load(
                &mut scope.scope("linear_fc1"),
                width,
                width,
                Bias::Present,
                Parallel::None,
            )?,
            second: Linear::load(
                &mut scope.scope("linear_fc2"),
                width,
                out_hidden,
                Bias::Present,
                Parallel::None,
            )?,
            merge: config.spatial_merge_size,
            width,
            out_hidden,
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
                    None,
                    buffers.normalized.reshape(merged),
                )
                .map_err(Error::from)?;
        } else {
            self.norm
                .forward(f, hidden, None, buffers.normalized)
                .map_err(Error::from)?;
        }
        self.first
            .forward(
                f,
                buffers.normalized.reshape(merged),
                buffers.merged.reshape(merged),
            )
            .map_err(Error::from)?;
        tower.merger_gelu.forward(
            f,
            buffers.merged.reshape(merged),
            buffers.merged_activated.reshape(merged),
        )?;
        self.second
            .forward(f, buffers.merged_activated.reshape(merged), out)
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
    /// The query/key row counts of one dense bidirectional call; its key bound is the grid's.
    heads: u32,
    head_dim: u32,
    rotate: Rotate,
    layers: Vec<Block>,
    /// The layers whose hidden state a DeepStack merger reads, in checkpoint order.
    deepstack_layers: Vec<usize>,
    merger: Merger,
    deepstack: Vec<Merger>,
    blocks_gelu: Gelu,
    merger_gelu: Gelu,
    residual: Residual,
}

impl VisionTower {
    /// The tower of `config`, reading the checkpoint through `loader`.
    ///
    /// `dtype` is the activation element type the loading boundary chose: the checkpoint's own
    /// representation unless the deployment asked for another, and every weight is read in it.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        config: &VisionConfig,
        dtype: nn::DType,
    ) -> Result<Self> {
        let hidden = u64::from(config.hidden_size);
        let tubelet = u64::from(BlockRows::tubelet(config));
        // The checkpoint's activation, and the merger's exact form, are two different GELUs.
        let blocks_gelu = match config.hidden_act {
            ops::activation::Activation::GeluTanh => Gelu::tanh(),
            other => {
                return Err(Error::config(format!(
                    "the vision tower gates with gelu_pytorch_tanh, not {other}"
                )));
            }
        };
        let heads = config.num_heads;
        let head_dim = BlockRows::head_dim(config)?;
        // The position table must be a square grid: its rows are indexed by `row · side + column`.
        let position_grid = BlockRows::position_grid(config)?;
        let mut tower = loader.scope(PREFIX);
        let layers = (0..config.depth)
            .map(|index| Block::load(&mut tower.scope(format!("blocks.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let deepstack = (0..config.deepstack_visual_indexes.len())
            .map(|index| {
                Merger::load(
                    &mut tower.scope(format!("deepstack_merger_list.{index}")),
                    config,
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // The patch embedding is a kernel-equals-stride convolution: its `[hidden, C, T, P, P]`
        // weight is read as the `[hidden, tubelet]` matrix of the product it is.
        let patch = Linear::load(
            &mut tower.scope("patch_embed.proj"),
            u32::try_from(tubelet)
                .map_err(|_| invalid("VisionTower", "a tubelet wider than a u32"))?,
            config.hidden_size,
            Bias::Present,
            Parallel::None,
        )?;
        let position = tower.scope("pos_embed").tensor(
            "weight",
            &[u64::from(config.num_position_embeddings), hidden],
        )?;
        let merger = Merger::load(&mut tower.scope("merger"), config, false)?;
        Ok(Self {
            deepstack_layers: deepstack_layers(config)?,
            patch,
            position,
            position_grid,
            heads,
            head_dim,
            rotate: Rotate::new(u64::from(head_dim), u64::from(heads), crate::SPLIT_HALF)
                .map_err(Error::from)?,
            layers,
            merger,
            deepstack,
            blocks_gelu,
            merger_gelu: Gelu::exact(),
            // Every sub-layer's output joins the stream unscaled.
            residual: Residual::default(),
            config: config.clone(),
            dtype,
        })
    }

    /// How many DeepStack feature tables one block returns.
    pub fn deepstack_features(&self) -> usize {
        self.deepstack.len()
    }

    /// Reserve the buffers of a block of `patches` patches.
    pub fn buffers(&self, arena: &Arena, patches: u32) -> Result<Buffers> {
        Buffers::new(arena, &self.config, patches, self.dtype)
    }

    /// Encode one image block's patches into `out`.
    ///
    /// `image` carries the patches and the five prepared tables of its grid; `out` receives tokens
    /// of the language model's width, `merge²` patches each: the tower's own features and one table
    /// per DeepStack block. One call per block, in prompt order: the rows of one block never meet
    /// another block's, so a call per block is the arithmetic of one call over their concatenation,
    /// row by row.
    pub fn encode(
        &self,
        f: &Forward,
        image: &ImageTables,
        buffers: &Buffers,
        out: Outputs<'_>,
    ) -> Result<()> {
        let rows = image.rows;
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
        self.check(image, &rows, buffers, &out)?;

        // The patch embedding reads the tubelets the processor packed; the position blend is added
        // into the same rows, and the residual stream starts there.
        self.patch
            .forward(f, image.pixels, buffers.hidden)
            .map_err(Error::from)?;
        // The resample reads the request's own tables and only counts output points, so the
        // descriptor names the learned side and the point count. The block's two spatial extents
        // exist on the host (`positions::position_samples`) and nowhere in a captured call: a model
        // may not read the grid tensor back.
        PositionInterpolate::new(
            self.dtype,
            u64::from(self.config.hidden_size),
            self.position_grid,
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
            .forward(f, buffers.hidden, buffers.normalized, None, buffers.hidden)
            .map_err(Error::from)?;

        // One bidirectional launch over the grid's time steps: the descriptor's key bound is the
        // block's own patches (the reference's attention declares none), and its scale the head's
        // `head_dim^-0.5`.
        let attention = Attention::dense(
            self.head_dim,
            self.heads,
            self.heads,
            rows.patches,
            false,
            None,
            (self.head_dim as f32).sqrt().recip(),
        )
        .map_err(Error::from)?;

        let mut next = 0;
        for (index, block) in self.layers.iter().enumerate() {
            block.forward(self, f, image, &attention, &rows, buffers)?;
            if self
                .deepstack_layers
                .get(next)
                .is_some_and(|layer| *layer == index)
            {
                self.deepstack[next].forward(
                    self,
                    f,
                    buffers.hidden,
                    buffers,
                    out.deepstack[next],
                )?;
                next += 1;
            }
        }
        self.merger
            .forward(self, f, buffers.hidden, buffers, out.features)
    }

    /// Refuse a block whose prepared tables, buffers or outputs do not describe its row counts.
    fn check(
        &self,
        image: &ImageTables,
        rows: &BlockRows,
        buffers: &Buffers,
        out: &Outputs<'_>,
    ) -> Result<()> {
        let patches = u64::from(rows.patches);
        let merged = u64::from(rows.merged);
        let hidden = u64::from(self.config.hidden_size);
        let columns = u64::from(BlockRows::head_dim(&self.config)? / 2);
        let tubelet = u64::from(BlockRows::tubelet(&self.config));
        let width = u64::from(BlockRows::merged_width(&self.config));
        let out_hidden = u64::from(self.config.out_hidden_size);
        let tables = [
            ("pixels", image.pixels, Shape::new(&[patches, tubelet])),
            (
                "position_index",
                image.position_index,
                Shape::new(&[patches, 4]),
            ),
            (
                "position_weight",
                image.position_weight,
                Shape::new(&[patches, 4]),
            ),
            ("cosine", image.cosine, Shape::new(&[patches, columns])),
            ("sine", image.sine, Shape::new(&[patches, columns])),
            ("features", out.features, Shape::new(&[merged, out_hidden])),
        ];
        for (name, view, shape) in tables {
            if view.shape != shape {
                return Err(invalid(
                    "VisionTower",
                    format!(
                        "the {name} table is {:?} for a {}-patch block over {} time steps, \
                         {shape:?} expected",
                        view.shape.dims(),
                        rows.patches,
                        rows.time
                    ),
                ));
            }
        }
        for deepstack in out.deepstack {
            if deepstack.shape != Shape::new(&[merged, out_hidden]) {
                return Err(invalid(
                    "VisionTower",
                    format!(
                        "a DeepStack output is {:?} for a {}-patch block, [{merged}, {out_hidden}] \
                         expected",
                        deepstack.shape.dims(),
                        rows.patches
                    ),
                ));
            }
        }
        let lengths = u64::from(rows.time) + 1;
        if image.lengths.numel() != lengths {
            return Err(invalid(
                "VisionTower",
                format!(
                    "{} attention lengths for {} time steps",
                    image.lengths.numel(),
                    rows.time
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
                    "the block's buffers hold {} rows of {} and {} merged rows of {}, a {} row \
                     block needs {patches}, {hidden}, {merged} and {width}",
                    buffers.hidden.rows(),
                    buffers.hidden.last(),
                    buffers.merged.rows(),
                    buffers.merged.last(),
                    rows.patches
                ),
            ));
        }
        Ok(())
    }
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
        let merge = config.spatial_merge_size;
        let rows = BlockRows::of(&config, [1, 24, 24]).expect("a 24x24 patch grid");
        // A patch is one 16x16 pixel block, so a 24 by 24 patch grid is 24 * 24 of them, merged in
        // 2 by 2 blocks, with one sequence position per patch of the grid.
        assert_eq!(rows.patches, 24 * 24);
        assert_eq!(rows.merged, (24 / merge) * (24 / merge));
        assert_eq!(rows.sequence, rows.patches);
        // The tubelet is a patch's channels with its temporal depth, and a merged row is the merged
        // block's features - the two numbers the tower's first and last projections see.
        let patch = config.patch_size;
        assert_eq!(
            BlockRows::tubelet(&config),
            config.in_channels * config.temporal_patch_size * patch * patch
        );
        assert_eq!(
            BlockRows::merged_width(&config),
            config.hidden_size * merge * merge
        );
    }

    #[test]
    fn a_video_block_keeps_one_sequence_per_time_step() {
        let config = config();
        let merge = config.spatial_merge_size;
        let rows = BlockRows::of(&config, [4, 16, 16]).expect("four time steps");
        // Four time steps of a 16 by 16 patch grid: every patch is a row, and the merged rows are the
        // 2 by 2 blocks of each step.
        assert_eq!(rows.patches, 4 * 16 * 16);
        assert_eq!(rows.merged, 4 * (16 / merge) * (16 / merge));
        // The attention sequence is one step's **patches**, not its merged rows: the tower attends
        // over patch rows and merges afterwards, so a video block runs four sequences of 16 * 16.
        assert_eq!(rows.sequence, 16 * 16);
        assert_eq!(rows.sequence, rows.patches / rows.time);
    }

    #[test]
    fn a_grid_outside_whole_merge_blocks_is_refused() {
        let config = config();
        assert!(BlockRows::of(&config, [1, 23, 24]).is_err());
        assert!(BlockRows::of(&config, [1, 24, 0]).is_err());
    }

    #[test]
    fn a_calls_geometry_follows_the_tables_row_counts() {
        let config = config();
        // The tables state a block's patch rows and its time steps; the grid's split is a host fact
        // the tables already carry.
        let of_grid = BlockRows::of(&config, [4, 16, 24]).expect("four steps of 16x24");
        let of_counts =
            BlockRows::of_counts(&config, of_grid.patches, of_grid.time).expect("the same block");
        assert_eq!(of_counts, of_grid);
        assert!(BlockRows::of_counts(&config, 1000, 3).is_err());
        assert!(BlockRows::of_counts(&config, 0, 1).is_err());
        // A time step of six patches is not whole 2x2 merge blocks.
        assert!(BlockRows::of_counts(&config, 6, 1).is_err());
    }

    #[test]
    fn a_grid_larger_than_the_position_table_is_resampled_not_refused() {
        let config = config();
        // The table is 48 x 48 entries; the served 768x1344 canvas is 48 x 84 patches, and the
        // processor's pixel cap allows far more, so the grid is never bounded by the table.
        assert!(BlockRows::of(&config, [1, 48, 48]).is_ok());
        let merge = config.spatial_merge_size;
        let keyframe = BlockRows::of(&config, [1, 48, 84]).expect("a 768x1344 keyframe");
        // One time step of a 48 by 84 patch grid, merged 2 by 2: the numbers a 768p 16:9 frame gives.
        assert_eq!(keyframe.patches, 48 * 84);
        assert_eq!(keyframe.merged, (48 / merge) * (84 / merge));
        assert_eq!(keyframe.sequence, 48 * 84);
        assert_eq!(keyframe.sequence, keyframe.patches / keyframe.time);
        assert!(BlockRows::of(&config, [1, 256, 256]).is_ok());
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
}
