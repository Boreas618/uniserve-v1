//! The Qwen3-VL text encoder of MiniMax-H3: a prompt's hidden states, its media spliced in.
//!
//! `uniserve/models/minimax_h3/encoder.py` is the reference: H3 places the Qwen3-VL language model
//! and the vision tower, runs only the first [`RETAINED_LAYERS`](crate::RETAINED_LAYERS) of the
//! checkpoint's decoder layers, and returns the last retained layer's raw hidden state - the
//! checkpoint's final normalization is not part of H3's conditioning path (`network.norm =
//! Identity`). The tower's merged tokens replace the prompt's image and video placeholder tokens,
//! and the three DeepStack feature tables are added to the residual stream after the block each was
//! merged from (`uniserve/models/transformer.py::_traverse`, lines 408-419).
//!
//! ## The layer recurrence
//!
//! The stack keeps the reference's *separate* residual stream (`separate_residual = True`): a layer
//! returns its output unsummed, and the addition is fused into the next normalization. So a layer
//! reads the previous layer's update `x` and the residual `residual`, computes
//! `residual, hidden = add_rms_norm(x, residual, input_norm)`, then the attention and
//! `residual, hidden = add_rms_norm(attn, residual, post_attention_norm)`, and returns
//! `mlp(hidden)` unsummed. The first layer receives the embeddings with no residual yet, and the
//! last update is added once at the end, which is the reference's `norm(hidden + residual)` with
//! `norm` the identity (`uniserve/models/qwen3/transformer.py:178-201` for the layer, and
//! `uniserve/models/transformer.py:436-451` for the final fold).
//!
//! A DeepStack addition is the one place the recurrence is cut: the reference adds the layer's own
//! output to the residual (`residual + hidden`, one rounding), then the features (a second
//! rounding), and starts the next layer from a zero update. Adding zero is exact, so this
//! implementation drops the zero and normalizes the residual alone, which is the same bits.
//!
//! ## Precision and partitioning
//!
//! `dtype` is the checkpoint's own representation (BF16) unless the deployment asked for another,
//! and it is decided once at the loading boundary: no layer converts. The text stack is
//! tensor-parallel: query, key and value heads are split over the ranks, the MLP's intermediate
//! width is split, the output and down projections reduce over the group, and the embedding's
//! vocabulary is split (`nn::layers::Embedding` reduces its own lookups). The vision tower is
//! *replicated* on every rank: it is small, every rank needs its features, and its row-parallel
//! projections would otherwise need a bias added once after the reduction.

use nn::capability::{
    Encoder, EncoderInputs, EncoderLayout, EncoderOutputs, EncoderPrompt, ImageInput, ImageShape,
    StateTable,
};
use nn::layers::{
    Attention, Embedding, GatedActivation, Linear, RmsNorm, Rotary, Rows, ScaledResidual, Sections,
};
use nn::model::{Block, BlockName};
use nn::weights::{WeightLoader, WeightRequest};
use nn::{Arena, DevicePtr, Forward, Shape, TensorView, Weight};
use ops::activation::Activation;
use ops::gemm::WeightOrder;
use ops::rope::Rotation;

use crate::config::TextEncoderConfig;
use crate::load::Component;
use crate::partition::Partition;
use crate::positions::{self, Mrope};
use crate::vision::{ones, Geometry, ImageTables, Outputs, ROPE_THETA, VisionTower};
use crate::{Error, Result};

/// The language layers' gate: the checkpoint states `hidden_act: "silu"` and H3 fixes it.
const ACTIVATION: Activation = Activation::Silu;

/// The checkpoint's tensor names under the text encoder's component.
const PREFIX: &str = "model.language_model";

/// What one encode call of a layout covers.
///
/// The runner plans a bucket from these numbers: the prompt's rows, the images of the request (one
/// patch grid each) and the widths this rank holds. [`Plan::bytes`] then states the arena region one
/// such call reserves, which the executor keeps for as long as a captured graph addresses it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Plan {
    /// Rows the call writes: the packed prompt tokens, placeholders included.
    ///
    /// An encoder's output is one row per presented token, so this is the batch's `num_tokens`; the
    /// layout's `rows` is the capacity an executor reserves and must equal it (the model refuses a
    /// padded output, whose rows the batch's offsets do not describe).
    pub rows: u32,
    /// Merged vision rows of all the request's images together.
    pub merged: u32,
    /// Patch rows per image, in prompt order.
    pub images: Vec<u32>,
    /// Time steps per image, in prompt order.
    pub time: Vec<u32>,
    /// The images' patch grids as the layout stated them; empty in a plan built from a call.
    pub grids: Vec<[u32; 3]>,
    /// Width of one query projection on this rank.
    pub query: u32,
    /// Width of one key/value projection on this rank.
    pub key_value: u32,
    /// Width of the MLP's intermediate projection on this rank.
    pub intermediate: u32,
    /// The language model's width.
    pub hidden: u32,
    /// Elements of one head.
    pub head_dim: u32,
    /// DeepStack feature tables the checkpoint's tower returns.
    pub deepstack: u32,
}

impl Plan {
    /// The plan of one prompt of `tokens` rows whose images have the patch grids `grids`.
    ///
    /// The grids are the request's host values; `grids.len()` is the number of tower calls. A
    /// decode-style call ([`Plan::of_text`]) is a prompt with no images.
    pub fn new(
        config: &TextEncoderConfig,
        partition: &Partition,
        tokens: u32,
        grids: &[[u32; 3]],
    ) -> Result<Self> {
        let rank = Rank::new(config, partition)?;
        if tokens == 0 {
            return Err(Error::config(
                "a prompt of no tokens has nothing to encode".to_owned(),
            ));
        }
        let mut images = Vec::with_capacity(grids.len());
        let mut time = Vec::with_capacity(grids.len());
        let mut merged = 0_u32;
        for &grid in grids {
            let geometry = Geometry::of(&config.vision, grid)?;
            merged = merged
                .checked_add(geometry.merged)
                .ok_or_else(|| Error::config("the images merge to more than a u32 of rows".to_owned()))?;
            images.push(geometry.patches);
            time.push(geometry.time);
        }
        Ok(Self {
            rows: tokens,
            merged,
            images,
            time,
            grids: grids.to_vec(),
            query: rank.query,
            key_value: rank.key_value,
            intermediate: rank.intermediate,
            hidden: config.hidden_size,
            head_dim: config.head_dim,
            deepstack: config.vision.deepstack_visual_indexes.len() as u32,
        })
    }

    /// The plan of a text-only prompt of `tokens` rows.
    pub fn of_text(config: &TextEncoderConfig, partition: &Partition, tokens: u32) -> Result<Self> {
        Self::new(config, partition, tokens, &[])
    }

    /// The plan of the layout an executor prepared.
    ///
    /// A layout that reserves more output rows than the prompt presents is refused: the batch's
    /// query offsets describe exactly the presented rows, and nothing masks a pad row.
    pub fn of_layout(
        config: &TextEncoderConfig,
        partition: &Partition,
        layout: &EncoderLayout,
    ) -> Result<Self> {
        let grids = layout
            .images
            .iter()
            .map(|image| image.grid)
            .collect::<Vec<_>>();
        let plan = Self::new(config, partition, layout.tokens, &grids)?;
        if layout.rows != plan.rows {
            return Err(Error::config(format!(
                "the layout reserves {} output rows for a {} token prompt",
                layout.rows, plan.rows
            )));
        }
        Ok(plan)
    }

    /// The layout this plan describes, as the executor keys its state by.
    pub fn layout(&self) -> EncoderLayout {
        EncoderLayout {
            rows: self.rows,
            tokens: self.rows,
            images: self
                .grids
                .iter()
                .map(|grid| ImageShape { grid: *grid })
                .collect(),
        }
    }

    /// The bytes one encode call of this plan reserves in the arena.
    pub fn bytes(&self, config: &TextEncoderConfig, dtype: nn::DType) -> Result<u64> {
        let mut bytes = Buffers::bytes(self, dtype)?;
        for &patches in &self.images {
            bytes = bytes
                .checked_add(Buffers::tower_bytes(&config.vision, patches, dtype)?)
                .ok_or_else(|| Error::config("an encoder plan larger than a u64".to_owned()))?;
        }
        Ok(bytes)
    }
}

/// The widths one tensor-parallel rank of the text stack holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Rank {
    /// This rank's index.
    index: u32,
    /// The group's degree.
    degree: u32,
    /// Query heads this rank projects.
    query_heads: u32,
    /// Key/value heads this rank projects.
    key_value_heads: u32,
    /// Width of this rank's query projection.
    query: u32,
    /// Width of this rank's key and value projections.
    key_value: u32,
    /// Width of this rank's intermediate projection.
    intermediate: u32,
    /// The `[first, stop)` vocabulary rows this rank's embedding holds.
    vocabulary: (u32, u32),
}

impl Rank {
    /// The shard of `config` this rank holds under `partition`.
    fn new(config: &TextEncoderConfig, partition: &Partition) -> Result<Self> {
        let degree = u32::try_from(partition.tensor_parallel)
            .map_err(|_| Error::config("a tensor-parallel degree wider than a u32".to_owned()))?;
        let index = u32::try_from(partition.tensor_rank)
            .map_err(|_| Error::config("a tensor-parallel rank wider than a u32".to_owned()))?;
        if degree == 0 || index >= degree {
            return Err(Error::config(format!(
                "tensor-parallel rank {index} of a group of {degree}"
            )));
        }
        for (name, heads) in [
            ("query", config.num_attention_heads),
            ("key/value", config.num_key_value_heads),
        ] {
            if !heads.is_multiple_of(degree) {
                return Err(Error::config(format!(
                    "the text encoder's {heads} {name} heads do not split over {degree} ranks"
                )));
            }
        }
        if !config.intermediate_size.is_multiple_of(degree) {
            return Err(Error::config(format!(
                "the text encoder's {} intermediate channels do not split over {degree} ranks",
                config.intermediate_size
            )));
        }
        let query_heads = config.num_attention_heads / degree;
        let key_value_heads = config.num_key_value_heads / degree;
        let (first, stop) = partition.vocabulary_shard(config.vocab_size);
        Ok(Self {
            index,
            degree,
            query_heads,
            key_value_heads,
            query: query_heads * config.head_dim,
            key_value: key_value_heads * config.head_dim,
            intermediate: config.intermediate_size / degree,
            vocabulary: (first, stop),
        })
    }

    /// The element offset of this rank's shard of a `[rows, columns]` matrix split along `rows`.
    fn row_start(&self, local_rows: u32, columns: u32) -> u64 {
        u64::from(self.index) * u64::from(local_rows) * u64::from(columns)
    }

    /// The element offset of this rank's shard of a `[rows, columns]` matrix split along `columns`.
    fn column_start(&self, local_columns: u32) -> u64 {
        u64::from(self.index) * u64::from(local_columns)
    }
}

/// The device buffers one encode call reads and writes.
struct Buffers {
    /// `[tokens, hidden]` one normalization.
    normalized: TensorView,
    /// `[tokens, query]` the query projection.
    query: TensorView,
    /// `[tokens, query]` the normalized, rotated query.
    rotated_query: TensorView,
    /// `[tokens, key_value]` the key projection.
    key: TensorView,
    /// `[tokens, key_value]` the normalized, rotated key.
    rotated_key: TensorView,
    /// `[tokens, key_value]` the value projection.
    value: TensorView,
    /// `[tokens, query]` the attended rows.
    attended: TensorView,
    /// `[tokens, hidden]` the attention's output projection.
    projected: TensorView,
    /// `[tokens, intermediate]` the gate projection.
    gate: TensorView,
    /// `[tokens, intermediate]` the up projection.
    up: TensorView,
    /// `[tokens, intermediate]` its activation.
    activated: TensorView,
    /// `[tokens, hidden]` the down projection: a layer's unsummed output.
    down: TensorView,
    /// `[tokens, head_dim / 2]` FP32: the prompt's rotary cosine.
    cosine: TensorView,
    /// `[tokens, head_dim / 2]` FP32: the prompt's rotary sine.
    sine: TensorView,
    /// `[merged, out_hidden]` every image's merged language-width features.
    features: TensorView,
    /// `[deepstack · merged, out_hidden]` their DeepStack features, one table after another.
    deepstack: TensorView,
}

impl Buffers {
    /// Reserve the prompt's buffers; the tower's per-image buffers follow in the same arena.
    fn new(arena: &Arena, plan: &Plan, dtype: nn::DType) -> Result<Self> {
        let tokens = u64::from(plan.rows);
        let hidden = u64::from(plan.hidden);
        let query = u64::from(plan.query);
        let key_value = u64::from(plan.key_value);
        let intermediate = u64::from(plan.intermediate);
        let factors = u64::from(plan.head_dim / 2);
        let merged = u64::from(plan.merged);
        let out_hidden = hidden;
        let reserve = |shape: Shape| -> Result<TensorView> {
            arena.tensor(dtype, shape).map_err(Error::from)
        };
        let factors_of = |columns: u64| -> Result<TensorView> {
            arena
                .tensor(nn::DType::F32, Shape::new(&[tokens, columns]))
                .map_err(Error::from)
        };
        Ok(Self {
            normalized: reserve(Shape::new(&[tokens, hidden]))?,
            query: reserve(Shape::new(&[tokens, query]))?,
            rotated_query: reserve(Shape::new(&[tokens, query]))?,
            key: reserve(Shape::new(&[tokens, key_value]))?,
            rotated_key: reserve(Shape::new(&[tokens, key_value]))?,
            value: reserve(Shape::new(&[tokens, key_value]))?,
            attended: reserve(Shape::new(&[tokens, query]))?,
            projected: reserve(Shape::new(&[tokens, hidden]))?,
            gate: reserve(Shape::new(&[tokens, intermediate]))?,
            up: reserve(Shape::new(&[tokens, intermediate]))?,
            activated: reserve(Shape::new(&[tokens, intermediate]))?,
            down: reserve(Shape::new(&[tokens, hidden]))?,
            cosine: factors_of(factors)?,
            sine: factors_of(factors)?,
            features: reserve(Shape::new(&[merged, out_hidden]))?,
            deepstack: reserve(Shape::new(&[u64::from(plan.deepstack) * merged, out_hidden]))?,
        })
    }

    /// The bytes [`Buffers::new`] reserves.
    fn bytes(plan: &Plan, dtype: nn::DType) -> Result<u64> {
        let arena = Arena::new(DevicePtr::new(0), u64::MAX);
        Self::new(&arena, plan, dtype)?;
        Ok(arena.used())
    }

    /// The bytes one image's tower reservations take.
    fn tower_bytes(
        config: &crate::config::VisionConfig,
        patches: u32,
        dtype: nn::DType,
    ) -> Result<u64> {
        crate::vision::Buffers::bytes(config, patches, dtype)
    }

    /// `view`'s rows as `[rows, heads, head_dim]` for the rotation and attention operators.
    fn as_heads(&self, view: TensorView, heads: u32, head_dim: u32) -> Result<TensorView> {
        let rows = view.rows();
        Ok(view.reshape(Shape::new(&[
            rows,
            u64::from(heads),
            u64::from(head_dim),
        ])))
    }

    /// The `count` elements of a one-dimensional view starting at `first`.
    fn span_of(view: TensorView, first: u64, count: u64) -> TensorView {
        TensorView::new(
            view.offset_elements(first),
            view.dtype,
            Shape::new(&[count]),
        )
    }

    /// The `count` rows of `view` starting at `first`, as a contiguous `[count, columns]` view.
    fn rows_of(view: TensorView, first: u64, count: u64) -> TensorView {
        let columns = view.last();
        TensorView::new(
            view.offset_elements(first * columns),
            view.dtype,
            Shape::new(&[count, columns]),
        )
    }
}

/// One decoder layer: pre-norm attention, then the gated MLP.
struct Layer {
    input_norm: RmsNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    /// The queries' per-head normalization.
    query_norm: Weight,
    /// The keys' per-head normalization.
    key_norm: Weight,
    rotary: Rotary,
    attention: Attention,
    post_norm: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Layer {
    /// The layer at `index` of the checkpoint's stack.
    fn load(
        loader: &mut WeightLoader<'_>,
        config: &TextEncoderConfig,
        rank: &Rank,
        index: u32,
    ) -> Result<Self> {
        let hidden = u64::from(config.hidden_size);
        let head_dim = u64::from(config.head_dim);
        let name = format!("{PREFIX}.layers.{index}");
        let attention = format!("{name}.self_attn");
        // Query, key and value heads split over the ranks; the checkpoint stores each projection
        // as `[outputs, inputs]`, so a rank's heads are a contiguous block of rows.
        let query = sharded(
            loader,
            &format!("{attention}.q_proj.weight"),
            rank.row_start(rank.query, config.hidden_size),
            u64::from(rank.query),
            hidden,
        )?;
        let key = sharded(
            loader,
            &format!("{attention}.k_proj.weight"),
            rank.row_start(rank.key_value, config.hidden_size),
            u64::from(rank.key_value),
            hidden,
        )?;
        let value = sharded(
            loader,
            &format!("{attention}.v_proj.weight"),
            rank.row_start(rank.key_value, config.hidden_size),
            u64::from(rank.key_value),
            hidden,
        )?;
        // The output projection is row parallel: the contraction is over the query heads, so this
        // rank reads the columns of the heads it holds.
        let output = strided(
            loader,
            &format!("{attention}.o_proj.weight"),
            rank.column_start(rank.query),
            Shape::new(&[hidden, u64::from(rank.query)]),
            u64::from(config.query_width()),
        )?;
        // The M-RoPE sections cut the head into one band per axis; the operator counts rotated
        // elements, so each axis's frequency width is doubled.
        let mrope = Mrope::new(config.mrope_sections, config.head_dim)?;
        let widths = mrope.widths().map(|width| width * 2);
        let sections = Sections::new(&widths)?;
        Ok(Self {
            input_norm: normalization(loader, &format!("{name}.input_layernorm"), hidden, config)?,
            query: Linear::new(query, None, WeightOrder::NxK).map_err(Error::from)?,
            key: Linear::new(key, None, WeightOrder::NxK).map_err(Error::from)?,
            value: Linear::new(value, None, WeightOrder::NxK).map_err(Error::from)?,
            output: Linear::new(output, None, WeightOrder::NxK).map_err(Error::from)?,
            query_norm: region(
                loader,
                &format!("{attention}.q_norm.weight"),
                Shape::new(&[head_dim]),
            )?,
            key_norm: region(
                loader,
                &format!("{attention}.k_norm.weight"),
                Shape::new(&[head_dim]),
            )?,
            rotary: Rotary::new(
                frequencies(loader, config)?,
                config.head_dim,
                rank.query_heads,
                rank.key_value_heads,
                config.rms_norm_eps,
                Rotation::Half,
                1.0,
            )
            .map_err(Error::from)?
            .sectioned(sections)
            .map_err(Error::from)?,
            attention: Attention::dense(
                config.head_dim,
                rank.query_heads,
                rank.key_value_heads,
                config.max_position_embeddings,
                true,
                None,
                (config.head_dim as f32).sqrt().recip(),
            )
            .map_err(Error::from)?,
            post_norm: normalization(
                loader,
                &format!("{name}.post_attention_layernorm"),
                hidden,
                config,
            )?,
            gate: sharded_projection(
                loader,
                &format!("{name}.mlp.gate_proj.weight"),
                rank.row_start(rank.intermediate, config.hidden_size),
                u64::from(rank.intermediate),
                hidden,
            )?,
            up: sharded_projection(
                loader,
                &format!("{name}.mlp.up_proj.weight"),
                rank.row_start(rank.intermediate, config.hidden_size),
                u64::from(rank.intermediate),
                hidden,
            )?,
            down: Linear::new(
                strided(
                    loader,
                    &format!("{name}.mlp.down_proj.weight"),
                    rank.column_start(rank.intermediate),
                    Shape::new(&[hidden, u64::from(rank.intermediate)]),
                    u64::from(config.intermediate_size),
                )?,
                None,
                WeightOrder::NxK,
            )
            .map_err(Error::from)?,
        })
    }

    /// Run the layer: normalize, attend, normalize, gate; the update is left unsummed.
    ///
    /// `carried` is the previous layer's update, `None` for the first layer of the stack and after
    /// a DeepStack addition (the residual then holds every contribution). `residual` carries the
    /// stream and is written in place.
    #[allow(
        clippy::too_many_arguments,
        reason = "the call carries the reference's recurrence: the carried update, the residual \
                  stream, the prompt's factor tables and the plan's buffers"
    )]
    fn forward(
        &self,
        encoder: &TextEncoder,
        f: &Forward,
        offsets: TensorView,
        buffers: &Buffers,
        carried: Option<TensorView>,
        residual: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        match carried {
            Some(carried) => self
                .input_norm
                .forward_with_residual(f, carried, residual, residual, buffers.normalized)
                .map_err(Error::from)?,
            None => self
                .input_norm
                .forward(f, residual, buffers.normalized)
                .map_err(Error::from)?,
        }

        let heads = (encoder.rank.query_heads, encoder.rank.key_value_heads);
        self.query
            .forward(f, buffers.normalized, buffers.query)
            .map_err(Error::from)?;
        self.key
            .forward(f, buffers.normalized, buffers.key)
            .map_err(Error::from)?;
        self.value
            .forward(f, buffers.normalized, buffers.value)
            .map_err(Error::from)?;
        self.rotary
            .apply(
                f,
                buffers.as_heads(buffers.query, heads.0, encoder.config.head_dim)?,
                buffers.as_heads(buffers.key, heads.1, encoder.config.head_dim)?,
                Some(self.query_norm.view()),
                Some(self.key_norm.view()),
                cosine,
                sine,
                buffers.as_heads(buffers.rotated_query, heads.0, encoder.config.head_dim)?,
                buffers.as_heads(buffers.rotated_key, heads.1, encoder.config.head_dim)?,
            )
            .map_err(Error::from)?;
        self.attention
            .attend_dense(
                f,
                buffers.as_heads(buffers.rotated_query, heads.0, encoder.config.head_dim)?,
                buffers.as_heads(buffers.rotated_key, heads.1, encoder.config.head_dim)?,
                buffers.as_heads(buffers.value, heads.1, encoder.config.head_dim)?,
                Some(offsets),
                Some(offsets),
                buffers.as_heads(buffers.attended, heads.0, encoder.config.head_dim)?,
            )
            .map_err(Error::from)?;
        self.output
            .forward(f, buffers.attended, buffers.projected)
            .map_err(Error::from)?;
        encoder.reduce(f, buffers.projected)?;

        self.post_norm
            .forward_with_residual(
                f,
                buffers.projected,
                residual,
                residual,
                buffers.normalized,
            )
            .map_err(Error::from)?;

        self.gate
            .forward(f, buffers.normalized, buffers.gate)
            .map_err(Error::from)?;
        self.up
            .forward(f, buffers.normalized, buffers.up)
            .map_err(Error::from)?;
        encoder
            .activation
            .forward(f, buffers.gate, buffers.up, buffers.activated)?;
        self.down
            .forward(f, buffers.activated, buffers.down)
            .map_err(Error::from)?;
        encoder.reduce(f, buffers.down)
    }

}

/// The Qwen3-VL text encoder of H3, its vision tower included.
pub struct TextEncoder {
    config: TextEncoderConfig,
    partition: Partition,
    dtype: nn::DType,
    rank: Rank,
    embedding: Embedding,
    /// The prompt's M-RoPE factor tables; the layers apply the same factors.
    rotary: Rotary,
    layers: Vec<Layer>,
    tower: VisionTower,
    /// The row operator that splices the vision rows into the prompt and adds the DeepStack taps.
    rows: Rows,
    activation: GatedActivation,
    /// The coefficient of the residual stream: one, read as a weight by the fused operator.
    residual: ScaledResidual,
    /// The layers whose hidden state a DeepStack table is added after.
    deepstack_layers: Vec<usize>,
}

impl TextEncoder {
    /// The encoder of `config`, reading the checkpoint through `loader`.
    ///
    /// `dtype` is the activation element type the loading boundary chose. The stack is built for
    /// `partition`'s tensor-parallel position, and the vision tower is loaded whole on every rank.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        config: &TextEncoderConfig,
        partition: &Partition,
        dtype: nn::DType,
    ) -> Result<Self> {
        let rank = Rank::new(config, partition)?;
        let hidden = u64::from(config.hidden_size);
        let retained = config.num_retained_layers;
        if retained == 0 || retained > config.num_checkpoint_layers {
            return Err(Error::config(format!(
                "{retained} retained layers of {}",
                config.num_checkpoint_layers
            )));
        }
        let (first, stop) = rank.vocabulary;
        let rows = stop - first;
        let embedding = Embedding::new(
            region(
                loader,
                &format!("{PREFIX}.embed_tokens.weight"),
                Shape::new(&[u64::from(rows), hidden]),
            )
            .map_err(|error| {
                Error::config(format!("the embedding shard {first}..{stop}: {error}"))
            })?,
            first,
            rows,
            config.vocab_size,
        )
        .map_err(Error::from)?;
        let layers = (0..retained)
            .map(|index| Layer::load(loader, config, &rank, index))
            .collect::<Result<Vec<_>>>()?;
        let deepstack_layers = config
            .vision
            .deepstack_visual_indexes
            .iter()
            .map(|index| *index as usize)
            .collect::<Vec<_>>();
        for &layer in &deepstack_layers {
            if layer >= retained as usize {
                return Err(Error::config(format!(
                    "DeepStack feature after block {layer}, but H3 retains {retained} blocks"
                )));
            }
        }
        let mrope = Mrope::new(config.mrope_sections, config.head_dim)?;
        let widths = mrope.widths().map(|width| width * 2);
        let sections = Sections::new(&widths)?;
        Ok(Self {
            config: config.clone(),
            partition: partition.clone(),
            dtype,
            embedding,
            rotary: Rotary::new(
                frequencies(loader, config)?,
                config.head_dim,
                rank.query_heads,
                rank.key_value_heads,
                config.rms_norm_eps,
                Rotation::Half,
                1.0,
            )
            .map_err(Error::from)?
            .sectioned(sections)
            .map_err(Error::from)?,
            layers,
            tower: VisionTower::load(loader, &config.vision, dtype)?,
            rows: Rows::new(dtype, hidden).map_err(Error::from)?,
            activation: GatedActivation::new(ACTIVATION),
            residual: ScaledResidual::new(
                ones(
                    loader,
                    "minimax-h3.text_encoder.residual",
                    dtype,
                    u64::from(config.hidden_size),
                )?,
                None,
            )
            .map_err(Error::from)?,
            deepstack_layers,
            rank,
        })
    }

    /// The checkpoint configuration this encoder was built for.
    pub const fn config(&self) -> &TextEncoderConfig {
        &self.config
    }

    /// The deployment's mathematical partition.
    pub const fn partition(&self) -> &Partition {
        &self.partition
    }

    /// The element type of the encoder's activations.
    pub const fn dtype(&self) -> nn::DType {
        self.dtype
    }

    /// The vision tower this encoder composes.
    pub const fn tower(&self) -> &VisionTower {
        &self.tower
    }

    /// How many prompt tokens one encode call of `grids` covers at most: the packed tokens.
    pub fn plan(&self, tokens: u32, grids: &[[u32; 3]]) -> Result<Plan> {
        Plan::new(&self.config, &self.partition, tokens, grids)
    }

    /// The table names and shapes a layout needs, in the order [`TextEncoder::prepare_state`] fills
    /// them.
    ///
    /// Two tables follow the *prompt* - its M-RoPE positions and the prompt rows its vision rows
    /// land on - so their bytes change with every request; the five vision tables follow the
    /// *layout* alone, so an executor uploads them once per bucket. A layout with no images needs
    /// the two prompt tables only.
    fn tables(&self, plan: &Plan, rows: u32) -> Vec<StateTable> {
        table_specs(&self.config, rows, &plan.images, &plan.time, plan.merged)
    }

    /// Write the host bytes of every table of `layout`, in [`TextEncoder::tables`]' order.
    ///
    /// The arithmetic is the reference's request arithmetic, evaluated in FP32 on the host
    /// (`positions::*`): a captured call reads only the device copies. Nothing here allocates or
    /// touches the device.
    fn state_bytes(
        &self,
        prompt: &EncoderPrompt<'_>,
        layout: &EncoderLayout,
        out: &mut [Vec<u8>],
    ) -> Result<()> {
        let plan = Plan::of_layout(&self.config, &self.partition, layout)?;
        let tables = self.tables(&plan, layout.rows);
        if out.len() != tables.len() {
            return Err(Error::config(format!(
                "{} buffers for {} tables",
                out.len(),
                tables.len()
            )));
        }
        let vision = &self.config.vision;
        let merge = vision.spatial_merge_size;
        let tokens = plan.rows as usize;
        if prompt.ids.len() != tokens {
            return Err(Error::config(format!(
                "{} token ids for a {tokens} token prompt",
                prompt.ids.len()
            )));
        }

        // A vision grid reaches `rope_index` in the shape the prompt presents it: an image block is
        // one run of `time · rows · columns` placeholders with its whole grid, a video block is one
        // run per temporal patch, each of `rows · columns` placeholders on a single time step
        // (`uniserve/models/qwen3_vl/positions.py:31-52`).
        let (image_grids, video_grids) = run_grids(
            prompt.ids,
            &plan.grids,
            self.config.image_token_id,
            self.config.video_token_id,
            merge,
        )?;
        let coordinates = positions::rope_index(
            prompt.ids,
            &image_grids,
            &video_grids,
            self.config.image_token_id,
            self.config.video_token_id,
            merge,
        )?;
        // `[tokens][3]` to the operator's `[axes, tokens]`.
        let axes = self.config.mrope_sections.len();
        let mut positions = vec![0_i32; axes * tokens];
        for (token, coordinate) in coordinates.iter().enumerate() {
            for axis in 0..axes {
                positions[axis * tokens + token] = i32::try_from(coordinate[axis]).map_err(|_| {
                    Error::config("an M-RoPE coordinate wider than an i32".to_owned())
                })?;
            }
        }
        write_i32(&mut out[0], &positions, "prompt_positions")?;

        let destinations = positions::vision_destinations(
            prompt.ids,
            self.config.image_token_id,
            self.config.video_token_id,
        );
        if destinations.len() != plan.merged as usize {
            return Err(Error::config(format!(
                "{} placeholders for {} merged vision rows",
                destinations.len(),
                plan.merged
            )));
        }
        write_i32(&mut out[1], &destinations, "vision_destinations")?;
        if plan.images.is_empty() {
            return Ok(());
        }

        let side = Geometry::position_grid(vision)?;
        let head_dim = Geometry::head_dim(vision)?;
        let rotary = positions::VisionRotary::new(head_dim)?;
        let frequencies = positions::inverse_frequencies(head_dim / 2, ROPE_THETA)?;
        let mut index = Vec::new();
        let mut weight = Vec::new();
        let mut cosine = Vec::new();
        let mut sine = Vec::new();
        let mut lengths = Vec::new();
        for (grid, patches) in plan.grids.iter().zip(&plan.images) {
            let [time, height, width] = *grid;
            let (step_index, step_weight) = positions::position_samples(side, height, width, merge)?;
            let step_coordinates = positions::rotary_coordinates(height, width, merge)?;
            let columns = (head_dim / 2) as usize;
            for _ in 0..time {
                index.extend(step_index.iter().flatten().copied());
                weight.extend(step_weight.iter().flatten().copied());
                let mut step_cosine = vec![0.0_f32; step_coordinates.len() * columns];
                let mut step_sine = vec![0.0_f32; step_coordinates.len() * columns];
                rotary
                    .factors(
                        &step_coordinates,
                        &frequencies,
                        &mut step_cosine,
                        &mut step_sine,
                    )
                    .map_err(|error| {
                        Error::config(format!("the rotary factors of {grid:?}: {error}"))
                    })?;
                cosine.extend_from_slice(&step_cosine);
                sine.extend_from_slice(&step_sine);
            }
            lengths.extend_from_slice(&positions::vision_lengths(time, height, width)?);
            let expected = *patches as usize;
            if index.len() != expected * positions::POSITION_TAPS as usize
                || weight.len() != expected * positions::POSITION_TAPS as usize
                || cosine.len() != expected * columns
            {
                return Err(Error::config(format!(
                    "the tables of {grid:?} cover {} patches, {expected} expected",
                    cosine.len() / columns
                )));
            }
        }
        write_i32(&mut out[2], &index, "vision_position_index")?;
        write_f32(&mut out[3], &weight, "vision_position_weight")?;
        write_f32(&mut out[4], &cosine, "vision_cosine")?;
        write_f32(&mut out[5], &sine, "vision_sine")?;
        write_i32(&mut out[6], &lengths, "vision_lengths")
    }

    /// The encoder's tables read the way a call reads them.
    fn prepared<'a>(&self, inputs: &EncoderInputs<'a>, plan: &Plan) -> Result<Prepared<'a>> {
        let tables = self.tables(plan, plan.rows);
        let state = inputs.state;
        if state.len() != tables.len() {
            return Err(invalid_input(format!(
                "{} state tables for a layout of {}",
                state.len(),
                tables.len()
            )));
        }
        for (view, table) in state.iter().zip(&tables) {
            if view.dtype != table.dtype || view.shape != table.shape {
                return Err(invalid_input(format!(
                    "the {} table is {:?} {:?}, {:?} {:?} expected",
                    table.name,
                    view.dtype,
                    view.shape.dims(),
                    table.dtype,
                    table.shape.dims()
                )));
            }
        }
        Ok(Prepared {
            positions: state[0],
            destinations: state[1],
            vision: &state[2..],
        })
    }

    /// The encoder as the executor sees it: one block under the deployment's component name.
    pub fn into_block(self) -> (BlockName, Block) {
        (
            BlockName::new(Component::TextEncoder.name()),
            Block::Encoder(Box::new(self)),
        )
    }

    /// Sum a row-parallel projection's partial product over the group.
    fn reduce(&self, f: &Forward, view: TensorView) -> Result<()> {
        let Some(group) = f.group else {
            return if self.rank.degree == 1 {
                Ok(())
            } else {
                Err(Error::config(format!(
                    "the text encoder is split over {} ranks and the forward carries no group",
                    self.rank.degree
                )))
            };
        };
        group.all_reduce_sum(view, f.stream).map_err(Error::from)
    }

    /// Refuse a call whose batch, layout or outputs do not agree.
    ///
    /// The tables are checked against their declared shapes by [`TextEncoder::prepared`]; this
    /// checks the three host values a captured call is addressed by.
    fn check(
        &self,
        inputs: &EncoderInputs<'_>,
        out: &EncoderOutputs,
        plan: &Plan,
    ) -> Result<()> {
        let layout = inputs.layout;
        let tokens = u64::from(plan.rows);
        if layout.tokens != plan.rows {
            return Err(invalid_input(format!(
                "the layout presents {} tokens and the plan {}",
                layout.tokens, plan.rows
            )));
        }
        if inputs.batch.num_tokens != plan.rows {
            return Err(invalid_input(format!(
                "the batch packs {} tokens and the layout {}",
                inputs.batch.num_tokens, plan.rows
            )));
        }
        if layout.rows != plan.rows {
            return Err(invalid_input(format!(
                "the layout reserves {} output rows for a {} token prompt: an encoder writes one \
                 row per presented token",
                layout.rows, plan.rows
            )));
        }
        let hidden = u64::from(plan.hidden);
        if out.hidden.rows() != tokens || out.hidden.last() != hidden {
            return Err(invalid_input(format!(
                "the output is {} rows of {}, a {tokens} row prompt of {hidden} expected",
                out.hidden.rows(),
                out.hidden.last()
            )));
        }
        if inputs.ids.numel() != tokens {
            return Err(invalid_input(format!(
                "{} token ids for a {tokens} row prompt",
                inputs.ids.numel()
            )));
        }
        if inputs.images.len() != plan.images.len() {
            return Err(invalid_input(format!(
                "{} images for a layout of {}",
                inputs.images.len(),
                plan.images.len()
            )));
        }
        Ok(())
    }
}

impl TextEncoder {
    /// Encode one prompt: its media's features, then the retained layers over the spliced rows.
    fn encode_prompt(
        &self,
        f: &Forward,
        inputs: EncoderInputs<'_>,
        out: EncoderOutputs,
    ) -> Result<()> {
        let grids = inputs
            .layout
            .images
            .iter()
            .map(|image| image.grid)
            .collect::<Vec<_>>();
        let plan = self.plan(inputs.layout.tokens, &grids)?;
        self.check(&inputs, &out, &plan)?;
        let prepared = self.prepared(&inputs, &plan)?;

        let buffers = Buffers::new(f.arena(), &plan, self.dtype)?;
        let hidden = out.hidden;

        // The diagnosis of specs/worker.md 10 asks for two values the runner cannot otherwise
        // reach: the tower's residual stream after its last block and the merged feature table.
        // Both arrive as borrowed views, are used in place of the arena's own buffers, and change
        // no number; a serving call passes none.
        let diagnostics = inputs.diagnostics;
        let mut buffers = buffers;
        if !diagnostics.is_empty() && diagnostics.len() != 2 {
            return Err(invalid_input(format!(
                "{} diagnostic views, [vision_hidden, merged] or none expected",
                diagnostics.len()
            )));
        }
        let vision_hidden = diagnostics.first().copied();
        if let Some(merged) = diagnostics.get(1).copied() {
            if merged.rows() != u64::from(plan.merged) || merged.last() != u64::from(plan.hidden) {
                return Err(invalid_input(format!(
                    "the diagnostic feature table is {} rows of {}, [{}, {}] expected",
                    merged.rows(),
                    merged.last(),
                    plan.merged,
                    plan.hidden
                )));
            }
            buffers.features = merged;
        }

        // 1. The vision tower, one call per image, in prompt order: the k-th merged row of the
        //    concatenation takes the k-th vision placeholder of the prompt.
        let merged = u64::from(plan.merged);
        let mut offset = 0_u64;
        let mut patch_offset = 0_u64;
        let mut length_offset = 0_u64;
        let mut deepstack = Vec::with_capacity(self.deepstack_layers.len());
        for (index, image) in inputs.images.iter().enumerate() {
            let patches = plan.images[index];
            let time = plan.time[index];
            let tables = [
                Buffers::rows_of(prepared.vision[0], patch_offset, u64::from(patches)),
                Buffers::rows_of(prepared.vision[1], patch_offset, u64::from(patches)),
                Buffers::rows_of(prepared.vision[2], patch_offset, u64::from(patches)),
                Buffers::rows_of(prepared.vision[3], patch_offset, u64::from(patches)),
                Buffers::span_of(prepared.vision[4], length_offset, u64::from(time) + 1),
            ];
            let image_tables =
                ImageTables::new(&self.config.vision, image, patches, time, &tables)?;
            let mut tower = self.tower.buffers(f.arena(), patches)?;
            if let Some(stream) = vision_hidden {
                tower = tower.with_stream(stream).map_err(|error| {
                    invalid_input(format!(
                        "the diagnostic stream of the {patches} patch block: {error}"
                    ))
                })?;
            }
            let rows = u64::from(image_tables.geometry.merged);
            for table in 0..self.tower.deepstack_features() {
                deepstack.push(Buffers::rows_of(
                    buffers.deepstack,
                    table as u64 * merged + offset,
                    rows,
                ));
            }
            self.tower.encode(
                f,
                &image_tables,
                &tower,
                Outputs {
                    features: Buffers::rows_of(buffers.features, offset, rows),
                    deepstack: &deepstack[deepstack.len() - self.tower.deepstack_features()..],
                },
            )?;
            offset += rows;
            patch_offset += u64::from(patches);
            length_offset += u64::from(time) + 1;
        }
        debug_assert_eq!(offset, merged);

        // 2. The prompt's embeddings, with the placeholders replaced by the vision features.
        self.embedding
            .forward(f, inputs.ids, hidden)
            .map_err(Error::from)?;
        if !inputs.images.is_empty() {
            self.rows
                .replace(f, hidden, buffers.features, prepared.destinations)?;
        }

        // 3. The prompt's M-RoPE factors, once for the whole stack.
        self.rotary
            .factors(f, prepared.positions, buffers.cosine, buffers.sine)
            .map_err(Error::from)?;

        // 4. The retained layers, with each DeepStack table added after the block it came from.
        let offsets = inputs.batch.query_offsets;
        let mut carried = None;
        let mut next = 0;
        for (index, layer) in self.layers.iter().enumerate() {
            layer.forward(
                self,
                f,
                offsets,
                &buffers,
                carried,
                hidden,
                buffers.cosine,
                buffers.sine,
            )?;
            carried = Some(buffers.down);
            if self
                .deepstack_layers
                .get(next)
                .is_some_and(|block| *block == index)
            {
                // The reference's cut: the layer's own output joins the stream, then its features,
                // and the next layer normalizes the stream alone.
                self.residual
                    .forward(f, hidden, buffers.down)
                    .map_err(Error::from)?;
                if merged > 0 {
                    let features =
                        Buffers::rows_of(buffers.deepstack, next as u64 * merged, merged);
                    self.rows
                        .add(f, hidden, features, prepared.destinations)?;
                }
                carried = None;
                next += 1;
            }
        }
        if let Some(carried) = carried {
            self.residual
                .forward(f, hidden, carried)
                .map_err(Error::from)?;
        }
        Ok(())
    }
}

impl Encoder for TextEncoder {
    /// The layout of a batch and its images.
    ///
    /// The encoder writes one row per presented token, so the row capacity is the batch's; a grid
    /// that cannot be served is refused where a failure can be reported - [`Encoder::prepare_state`]
    /// and [`Encoder::encode`] - because this method answers with the layout alone.
    fn prepare_layout(&self, batch: &nn::capability::TokenBatch, images: &[ImageShape]) -> EncoderLayout {
        EncoderLayout {
            rows: batch.num_tokens,
            tokens: batch.num_tokens,
            images: images.to_vec(),
        }
    }

    fn state_tables(&self, layout: &EncoderLayout) -> Vec<StateTable> {
        match Plan::of_layout(&self.config, &self.partition, layout) {
            Ok(plan) => self.tables(&plan, layout.rows),
            // A layout the model cannot serve still needs its table list answered; the shapes are
            // arithmetic on the grids, and `prepare_state` refuses the layout itself.
            Err(_) => {
                let merge = self.config.vision.spatial_merge_size;
                let (patches, time, merged) = layout_counts(layout, merge);
                table_specs(&self.config, layout.rows, &patches, &time, merged)
            }
        }
    }

    fn prepare_state(
        &self,
        prompt: &EncoderPrompt<'_>,
        layout: &EncoderLayout,
        out: &mut [Vec<u8>],
    ) -> nn::Result<()> {
        self.state_bytes(prompt, layout, out).map_err(capability)
    }

    fn encode(
        &self,
        f: &Forward,
        inputs: EncoderInputs<'_>,
        out: EncoderOutputs,
    ) -> nn::Result<()> {
        self.encode_prompt(f, inputs, out).map_err(capability)
    }
}

/// The tables a call reads, in the order [`TextEncoder::tables`] declares them.
struct Prepared<'a> {
    /// `[axes, tokens]` I32: the prompt's M-RoPE coordinates.
    positions: TensorView,
    /// `[merged]` I32: the prompt row of every merged vision row, in prompt order.
    destinations: TensorView,
    /// The five vision tables, concatenated over the prompt's images in order.
    vision: &'a [TensorView],
}

/// The five vision tables' names and shapes for a concatenation of `patches` patch rows over
/// `time` time steps.
///
/// Total in its arguments: the shapes are arithmetic on the row counts, so an executor can ask for
/// them before it knows whether the layout can be served.
fn table_specs(
    config: &TextEncoderConfig,
    rows: u32,
    patches: &[u32],
    time: &[u32],
    merged: u32,
) -> Vec<StateTable> {
    let tokens = u64::from(rows);
    let patch_rows: u64 = patches.iter().map(|patches| u64::from(*patches)).sum();
    let steps: u64 = time.iter().map(|time| u64::from(*time) + 1).sum();
    let columns = u64::from(config.head_dim / 2);
    let table = |name: &'static str, dtype: nn::DType, shape: Shape| StateTable {
        name,
        dtype,
        shape,
    };
    let mut tables = vec![
        table(
            "prompt_positions",
            nn::DType::I32,
            Shape::new(&[u64::from(config.mrope_sections.len() as u32), tokens]),
        ),
        table(
            "vision_destinations",
            nn::DType::I32,
            Shape::new(&[u64::from(merged)]),
        ),
    ];
    if !patches.is_empty() {
        tables.push(table(
            "vision_position_index",
            nn::DType::I32,
            Shape::new(&[patch_rows, positions::POSITION_TAPS]),
        ));
        tables.push(table(
            "vision_position_weight",
            nn::DType::F32,
            Shape::new(&[patch_rows, positions::POSITION_TAPS]),
        ));
        tables.push(table(
            "vision_cosine",
            nn::DType::F32,
            Shape::new(&[patch_rows, columns]),
        ));
        tables.push(table(
            "vision_sine",
            nn::DType::F32,
            Shape::new(&[patch_rows, columns]),
        ));
        tables.push(table(
            "vision_lengths",
            nn::DType::I32,
            Shape::new(&[steps]),
        ));
    }
    tables
}

/// The patch rows, time steps and merged rows a layout states, by plain arithmetic.
///
/// Used where a failure cannot be reported (a table list) so that the caller still learns the
/// shapes; [`Plan::of_layout`] is what refuses a grid that cannot be served.
fn layout_counts(layout: &EncoderLayout, merge: u32) -> (Vec<u32>, Vec<u32>, u32) {
    let mut patches = Vec::with_capacity(layout.images.len());
    let mut time = Vec::with_capacity(layout.images.len());
    let mut merged = 0_u32;
    for image in &layout.images {
        let [steps, height, width] = image.grid;
        let rows = steps.saturating_mul(height).saturating_mul(width);
        let blocks = (height / merge.max(1))
            .saturating_mul(width / merge.max(1))
            .max(1);
        patches.push(rows);
        time.push(steps);
        merged = merged.saturating_add(steps.saturating_mul(blocks));
    }
    (patches, time, merged)
}

/// The two grid lists [`positions::rope_index`] reads, one entry per vision placeholder run.
///
/// A vision grid reaches `rope_index` in the shape the prompt presents it
/// (`uniserve/models/qwen3_vl/positions.py:31-52`): an image block is one run of
/// `time · rows · columns` placeholders under its whole grid, a video block is one run per temporal
/// patch, each of `rows · columns` placeholders on a single time step. The run lengths are what
/// tells the two apart, and the grids arrive in the prompt's block order.
fn run_grids(
    ids: &[u32],
    grids: &[[u32; 3]],
    image_token_id: u32,
    video_token_id: u32,
    merge: u32,
) -> Result<(Vec<[u32; 3]>, Vec<[u32; 3]>)> {
    if merge == 0 {
        return Err(Error::config("the merge side must be positive".to_owned()));
    }
    // The prompt's vision runs, in order: which placeholder they hold and how many.
    let mut runs: Vec<(u32, u64)> = Vec::new();
    let mut row = 0;
    while row < ids.len() {
        let id = ids[row];
        if id != image_token_id && id != video_token_id {
            row += 1;
            continue;
        }
        let start = row;
        while row < ids.len() && ids[row] == id {
            row += 1;
        }
        runs.push((id, (row - start) as u64));
    }

    let mut images = Vec::with_capacity(grids.len());
    let mut videos = Vec::new();
    let mut next = 0_usize;
    for &grid in grids {
        let [time, height, width] = grid;
        if height % merge != 0 || width % merge != 0 {
            return Err(Error::config(format!(
                "the vision grid {grid:?} does not hold whole {merge}x{merge} merge blocks"
            )));
        }
        let step = u64::from(height / merge) * u64::from(width / merge);
        let Some(&(id, run)) = runs.get(next) else {
            return Err(Error::config(format!(
                "the vision grid {grid:?} has no placeholder run"
            )));
        };
        if id == image_token_id {
            if run != step * u64::from(time) {
                return Err(Error::config(format!(
                    "an image run of {run} placeholders for a {grid:?} grid of {}",
                    step * u64::from(time)
                )));
            }
            images.push(grid);
            next += 1;
            continue;
        }
        for _ in 0..time {
            let Some(&(id, run)) = runs.get(next) else {
                return Err(Error::config(format!(
                    "the video grid {grid:?} has fewer than {time} placeholder runs"
                )));
            };
            if id != video_token_id || run != step {
                return Err(Error::config(format!(
                    "a video run of {run} placeholders of type {id} for one {height}x{width} \
                     time step of {step}"
                )));
            }
            videos.push([1, height, width]);
            next += 1;
        }
    }
    if next != runs.len() {
        return Err(Error::config(format!(
            "{} vision placeholder runs for {} grids",
            runs.len() - next,
            grids.len()
        )));
    }
    Ok((images, videos))
}

/// Write `values` into `buffer` as little-endian up to its stated length.
fn write_i32(buffer: &mut [u8], values: &[i32], name: &'static str) -> Result<()> {
    let bytes = values.len() * 4;
    if buffer.len() != bytes {
        return Err(Error::config(format!(
            "the {name} table holds {} bytes for {bytes}",
            buffer.len()
        )));
    }
    for (chunk, value) in buffer.chunks_exact_mut(4).zip(values) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

/// Write `values` into `buffer` as little-endian FP32 up to its stated length.
fn write_f32(buffer: &mut [u8], values: &[f32], name: &'static str) -> Result<()> {
    let bytes = values.len() * 4;
    if buffer.len() != bytes {
        return Err(Error::config(format!(
            "the {name} table holds {} bytes for {bytes}",
            buffer.len()
        )));
    }
    for (chunk, value) in buffer.chunks_exact_mut(4).zip(values) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

/// The capability boundary's view of this crate's failures: an `nn` failure is already one, and
/// every other refusal names this block.
fn capability(error: Error) -> nn::Error {
    match error {
        Error::Nn(error) => error,
        other => nn::Error::Shape {
            owner: "MiniMax-H3 text encoder",
            problem: other.to_string(),
        },
    }
}

/// A refusal of a call whose inputs do not describe the plan.
fn invalid_input(problem: String) -> Error {
    Error::Nn(nn::Error::from(ops::Error::Shape(format!(
        "TextEncoder: {problem}"
    ))))
}

/// A region of one checkpoint tensor, in the operator's NxK order.
fn region(loader: &mut WeightLoader<'_>, name: &str, shape: Shape) -> Result<Weight> {
    loader
        .load(&WeightRequest::region(name, 0, shape))
        .map_err(Error::from)
}

/// A region of one checkpoint tensor starting at element `start`.
fn sharded(
    loader: &mut WeightLoader<'_>,
    name: &str,
    start: u64,
    rows: u64,
    columns: u64,
) -> Result<Weight> {
    loader
        .load(&WeightRequest::region(name, start, Shape::new(&[rows, columns])))
        .map_err(Error::from)
}

/// A sub-matrix of one checkpoint tensor whose rows are `stride` elements apart.
fn strided(
    loader: &mut WeightLoader<'_>,
    name: &str,
    start: u64,
    shape: Shape,
    stride: u64,
) -> Result<Weight> {
    loader
        .load(&WeightRequest::strided(name, start, shape, stride))
        .map_err(Error::from)
}

/// A projection of a sharded `[rows, columns]` weight: the checkpoint's tensors carry no bias.
fn sharded_projection(
    loader: &mut WeightLoader<'_>,
    name: &str,
    start: u64,
    rows: u64,
    columns: u64,
) -> Result<Linear> {
    Linear::new(sharded(loader, name, start, rows, columns)?, None, WeightOrder::NxK)
        .map_err(Error::from)
}

/// A normalization of `width` channels with the checkpoint's epsilon.
fn normalization(
    loader: &mut WeightLoader<'_>,
    name: &str,
    width: u64,
    config: &TextEncoderConfig,
) -> Result<RmsNorm> {
    RmsNorm::new(
        region(loader, &format!("{name}.weight"), Shape::new(&[width]))?,
        config.rms_norm_eps,
    )
    .map_err(Error::from)
}

/// The rotary frequency table: `theta^(-2j/head_dim)` for every rotated frequency, in FP32.
///
/// It is computed once here from the configuration (specs/worker.md 9: the frequency table is a
/// loading-boundary value) and published as a retained constant, so the factors op reads it as a
/// device weight like any other.
fn frequencies(loader: &mut WeightLoader<'_>, config: &TextEncoderConfig) -> Result<Weight> {
    let frequencies = positions::inverse_frequencies(config.head_dim, config.rope_theta)?;
    let mut bytes = Vec::with_capacity(frequencies.len() * 4);
    for value in &frequencies {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    loader
        .constant(
            "minimax-h3.text_encoder.rotary.inv_freq",
            nn::DType::F32,
            Shape::new(&[frequencies.len() as u64]),
            &bytes,
        )
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released text encoder's configuration.
    fn config() -> TextEncoderConfig {
        crate::config::tests_support::released_config().text_encoder
    }

    #[test]
    fn one_rank_holds_every_head_and_the_whole_mlp() {
        let config = config();
        let rank = Rank::new(&config, &Partition::single()).expect("one rank");
        assert_eq!(rank.degree, 1);
        assert_eq!(rank.query_heads, 64);
        assert_eq!(rank.key_value_heads, 8);
        assert_eq!(rank.query, 64 * 128);
        assert_eq!(rank.key_value, 8 * 128);
        assert_eq!(rank.intermediate, 25_600);
        assert_eq!(rank.vocabulary, (0, config.vocab_size));
    }

    #[test]
    fn four_ranks_split_the_heads_the_intermediate_width_and_the_vocabulary() {
        let config = config();
        let partition = Partition {
            ulysses_degree: 1,
            tensor_parallel: 4,
            tensor_rank: 2,
        };
        let rank = Rank::new(&config, &partition).expect("four ranks");
        assert_eq!(rank.query_heads, 16);
        assert_eq!(rank.key_value_heads, 2);
        assert_eq!(rank.query, 16 * 128);
        assert_eq!(rank.key_value, 2 * 128);
        assert_eq!(rank.intermediate, 6_400);
        // The third rank's rows of the query projection start after two whole shards.
        assert_eq!(rank.row_start(rank.query, config.hidden_size), 2 * 2048 * 5120);
        // Its columns of the output projection start after two shards of the head split.
        assert_eq!(rank.column_start(rank.query), 2 * 2048);
        let (first, stop) = rank.vocabulary;
        assert_eq!(first, 2 * config.vocab_size.div_ceil(4));
        assert!(stop <= config.vocab_size);
    }

    #[test]
    fn a_degree_that_does_not_split_the_heads_is_refused() {
        let config = config();
        let partition = Partition {
            ulysses_degree: 1,
            tensor_parallel: 3,
            tensor_rank: 0,
        };
        assert!(Rank::new(&config, &partition).is_err());
        let out_of_range = Partition {
            ulysses_degree: 1,
            tensor_parallel: 4,
            tensor_rank: 4,
        };
        assert!(Rank::new(&config, &out_of_range).is_err());
    }

    #[test]
    fn the_plan_counts_the_prompt_and_its_images() {
        let config = config();
        let plan = Plan::new(
            &config,
            &Partition::single(),
            4096,
            &[[1, 24, 24], [4, 16, 16]],
        )
        .expect("a prompt with two images");
        assert_eq!(plan.rows, 4096);
        assert_eq!(plan.images, [576, 1024]);
        assert_eq!(plan.merged, 144 + 256);
        assert_eq!(plan.deepstack, 3);
        assert_eq!(plan.hidden, 5120);
        assert_eq!(plan.head_dim, 128);
    }

    #[test]
    fn the_plan_of_a_text_prompt_has_no_vision_rows() {
        let config = config();
        let plan = Plan::of_text(&config, &Partition::single(), 128).expect("one prompt");
        assert!(plan.images.is_empty());
        assert_eq!(plan.merged, 0);
        assert_eq!(plan.rows, 128);
    }

    #[test]
    fn the_plan_reserves_more_with_images_and_more_rows() {
        let config = config();
        let single = Partition::single();
        let text = Plan::of_text(&config, &single, 512).expect("a text prompt");
        let prompt = Plan::new(&config, &single, 512, &[[1, 24, 24]]).expect("one image");
        let longer = Plan::of_text(&config, &single, 1024).expect("a longer prompt");
        let dtype = nn::DType::BF16;
        let text_bytes = text.bytes(&config, dtype).expect("bytes");
        let prompt_bytes = prompt.bytes(&config, dtype).expect("bytes");
        let longer_bytes = longer.bytes(&config, dtype).expect("bytes");
        assert!(prompt_bytes > text_bytes);
        assert!(longer_bytes > text_bytes);
        // The text plan holds twelve `[tokens, width]` rows plus two FP32 factor tables.
        assert!(text_bytes > 12 * 512 * u64::from(config.hidden_size) * 2);
    }

    #[test]
    fn the_frequency_table_is_the_checkpoints_rotary_recipe() {
        let config = config();
        let frequencies =
            crate::positions::inverse_frequencies(config.head_dim, config.rope_theta)
                .expect("the released recipe");
        assert_eq!(frequencies.len(), config.head_dim as usize / 2);
        assert_eq!(frequencies[0], 1.0);
        assert!(frequencies[63] < frequencies[0]);
    }
}
