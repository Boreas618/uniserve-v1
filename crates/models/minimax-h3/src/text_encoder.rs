//! The Qwen3-VL text encoder of MiniMax-H3: a prompt's hidden states, its media spliced in.
//!
//! H3 places the Qwen3-VL language model
//! and the vision tower, runs only the first [`RETAINED_LAYERS`](crate::RETAINED_LAYERS) of the
//! checkpoint's decoder layers, and returns the last retained layer's raw hidden state - the
//! checkpoint's final normalization is not part of H3's conditioning path (`network.norm =
//! Identity`). The tower's merged tokens replace the prompt's image and video placeholder tokens,
//! and DeepStack feature `j` is added to the residual stream after retained layer `j`, whichever
//! tower block it was merged from.
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
//! `norm` the identity.
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
//! vocabulary is split (`nn::layers::VocabParallelEmbedding` reduces its own lookups). The vision tower is
//! *replicated* on every rank: it is small, every rank needs its features, and its row-parallel
//! projections would otherwise need a bias added once after the reduction.

use nn::capability::{
    Encoder, EncoderInputs, EncoderLayout, EncoderOutputs, EncoderPrompt, ImageShape, StateTable,
    TableValues,
};
use nn::family::Split;
use nn::layers::{
    Attention, Bias, GatedActivation, Linear, Parallel, QkvBias, QkvLinear, QkvNames, QkvSpec,
    Residual, ResidualStream, RmsNorm, Rotary, Rows, Sections, VocabParallelEmbedding,
};
use nn::weights::{Scope, WeightLoader};
use nn::{Arena, Forward, Shape, TensorView, Weight};
use ops::activation::Activation;

use crate::config::TextEncoderConfig;
use crate::partition::Partition;
use crate::positions::{self, Mrope};
use crate::vision::{BlockRows, ImageTables, Outputs, ROPE_THETA, VisionTower};
use crate::{Error, Result};

/// The component a deployment's fence probes name (`profiling_config.fence_components`): one layer
/// of the text encoder's stack.
const TRANSFORMER_LAYER: &str = "models.minimax_h3.text_encoder.TransformerLayer";

/// The language layers' gate: the checkpoint states `hidden_act: "silu"` and H3 fixes it.
const ACTIVATION: Activation = Activation::Silu;

/// The checkpoint's tensor names under the text encoder's component.
const PREFIX: &str = "model.language_model";

/// What one encode call of a layout covers.
///
/// An encode call reads these numbers: the prompt's rows, the images of the request (one patch grid
/// each) and the widths this rank holds. They shape the call's tables and its arena reservations.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Plan {
    /// Rows the call writes: the packed prompt tokens, placeholders included.
    ///
    /// An encoder's output is one row per presented token, so this is the batch's `num_tokens`; the
    /// layout's `rows` is the capacity a worker reserves and must equal it (the model refuses a
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
    /// text-only prompt has no grids.
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
            let rows = BlockRows::of(&config.vision, grid)?;
            merged = merged.checked_add(rows.merged).ok_or_else(|| {
                Error::config("the images merge to more than a u32 of rows".to_owned())
            })?;
            images.push(rows.patches);
            time.push(rows.time);
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

    /// The plan of the layout a worker prepared.
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
}

/// The M-RoPE cut of one head: the checkpoint's widths, in rotated elements, interleaved.
///
/// The checkpoint states `mrope_section: [24, 20, 20]` and `mrope_interleaved: true`
/// (`text_encoder/config.json`, `rope_scaling`). The widths count *frequencies*, and the rotary
/// operator counts *rotated elements*, so each is doubled; the interleaved cut assigns every
/// compact frequency the axis `ops::rope::interleaved_axes` gives it (`j % 3` while
/// `j < widths[a] * 3`), and the contiguous cut is a different coordinate assignment.
fn sections_of(config: &TextEncoderConfig) -> Result<Sections> {
    let mrope = Mrope::new(config.mrope_sections, config.head_dim)?;
    let widths = mrope.widths().map(|width| width * 2);
    Ok(Sections::new(&widths)?.interleaved())
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
        Ok(Self {
            index,
            degree,
            query_heads,
            key_value_heads,
            query: query_heads * config.head_dim,
            key_value: key_value_heads * config.head_dim,
            intermediate: config.intermediate_size / degree,
        })
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
            deepstack: reserve(Shape::new(&[
                u64::from(plan.deepstack) * merged,
                out_hidden,
            ]))?,
        })
    }

    /// `view`'s rows as `[rows, heads, head_dim]` for the rotation and attention operators.
    fn as_heads(&self, view: TensorView, heads: u32, head_dim: u32) -> Result<TensorView> {
        let rows = view.rows();
        Ok(view.reshape(Shape::new(&[rows, u64::from(heads), u64::from(head_dim)])))
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
    /// The query, key and value projections, split over the tensor-parallel ranks by heads.
    qkv: QkvLinear,
    /// The output projection, row parallel over the query heads: its forward sums the ranks'
    /// partial outputs.
    output: Linear,
    /// The queries' per-head normalization.
    query_norm: Weight,
    /// The keys' per-head normalization.
    key_norm: Weight,
    rotary: Rotary,
    post_norm: RmsNorm,
    gate: Linear,
    up: Linear,
    /// The down projection, row parallel over the intermediate channels.
    down: Linear,
}

impl Layer {
    /// The layer the module at `scope` states, at `split`'s rank of the tensor-parallel group.
    ///
    /// `frequencies` is the stack's one rotary frequency table, which every layer rotates by.
    fn load(
        scope: &mut Scope<'_, '_>,
        config: &TextEncoderConfig,
        rank: &Rank,
        split: Split,
        frequencies: Weight,
    ) -> Result<Self> {
        let hidden = u64::from(config.hidden_size);
        let head_dim = u64::from(config.head_dim);
        let sections = sections_of(config)?;
        let mut attention = scope.scope("self_attn");
        let qkv = QkvLinear::load(
            &mut attention,
            QkvSpec {
                hidden: config.hidden_size,
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
            },
            QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
            QkvBias::Absent,
            split,
        )?;
        let output = Linear::load(
            &mut attention.scope("o_proj"),
            config.query_width(),
            config.hidden_size,
            Bias::Absent,
            Parallel::Row(split),
        )?;
        let query_norm = attention.scope("q_norm").tensor("weight", &[head_dim])?;
        let key_norm = attention.scope("k_norm").tensor("weight", &[head_dim])?;
        let mut mlp = scope.scope("mlp");
        let mut column = |module: &str| {
            Linear::load(
                &mut mlp.scope(module),
                config.hidden_size,
                config.intermediate_size,
                Bias::Absent,
                Parallel::Column(split),
            )
        };
        let gate = column("gate_proj")?;
        let up = column("up_proj")?;
        let down = Linear::load(
            &mut mlp.scope("down_proj"),
            config.intermediate_size,
            config.hidden_size,
            Bias::Absent,
            Parallel::Row(split),
        )?;
        Ok(Self {
            input_norm: RmsNorm::load(
                &mut scope.scope("input_layernorm"),
                hidden,
                config.rms_norm_eps,
            )?,
            qkv,
            output,
            query_norm,
            key_norm,
            rotary: Rotary::new(
                frequencies,
                config.head_dim,
                rank.query_heads,
                rank.key_value_heads,
                config.rms_norm_eps,
                crate::SPLIT_HALF,
                1.0,
            )
            .map_err(Error::from)?
            .sectioned(sections)
            .map_err(Error::from)?,
            post_norm: RmsNorm::load(
                &mut scope.scope("post_attention_layernorm"),
                hidden,
                config.rms_norm_eps,
            )?,
            gate,
            up,
            down,
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
    #[allow(
        clippy::too_many_arguments,
        reason = "the call carries the reference's recurrence, the prompt's factor tables, the \
                  batch's attention descriptor and the plan's buffers"
    )]
    fn forward(
        &self,
        encoder: &TextEncoder,
        f: &Forward,
        attention: &Attention,
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
                .forward(
                    f,
                    carried,
                    Some(ResidualStream::in_place(residual)),
                    buffers.normalized,
                )
                .map_err(Error::from)?,
            None => self
                .input_norm
                .forward(f, residual, None, buffers.normalized)
                .map_err(Error::from)?,
        }

        let heads = (encoder.rank.query_heads, encoder.rank.key_value_heads);
        self.qkv.forward(
            f,
            buffers.normalized,
            buffers.query,
            buffers.key,
            buffers.value,
        )?;
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
        attention
            .attend_dense(
                f,
                buffers.as_heads(buffers.rotated_query, heads.0, encoder.config.head_dim)?,
                buffers.as_heads(buffers.rotated_key, heads.1, encoder.config.head_dim)?,
                buffers.as_heads(buffers.value, heads.1, encoder.config.head_dim)?,
                Some(offsets),
                Some(offsets),
                u32::try_from(buffers.value.rows()).map_err(|_| {
                    invalid_input("a prompt longer than a key bound counts".to_owned())
                })?,
                buffers.as_heads(buffers.attended, heads.0, encoder.config.head_dim)?,
            )
            .map_err(Error::from)?;
        // Row parallel: the projection sums the ranks' partial outputs itself.
        self.output
            .forward(f, buffers.attended, buffers.projected)
            .map_err(Error::from)?;

        self.post_norm
            .forward(
                f,
                buffers.projected,
                Some(ResidualStream::in_place(residual)),
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
        Ok(self.down.forward(f, buffers.activated, buffers.down)?)
    }
}

/// The Qwen3-VL text encoder of H3, its vision tower included.
pub struct TextEncoder {
    config: TextEncoderConfig,
    partition: Partition,
    dtype: nn::DType,
    rank: Rank,
    embedding: VocabParallelEmbedding,
    /// The prompt's M-RoPE factor tables; the layers apply the same factors.
    rotary: Rotary,
    layers: Vec<Layer>,
    tower: VisionTower,
    /// The row operator that splices the vision rows into the prompt and adds the DeepStack taps.
    rows: Rows,
    activation: GatedActivation,
    /// The add of every sub-layer's update into the residual stream.
    residual: Residual,
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
        let split = Split {
            size: rank.degree,
            rank: rank.index,
        };
        let embedding = VocabParallelEmbedding::load(
            &mut loader.scope(format!("{PREFIX}.embed_tokens")),
            config.vocab_size,
            config.hidden_size,
            split,
        )?;
        // One frequency table rotates the prompt's factors and every layer's.
        let frequencies = frequencies(loader, config)?;
        let layers = (0..retained)
            .map(|index| {
                Layer::load(
                    &mut loader.scope(format!("{PREFIX}.layers.{index}")),
                    config,
                    &rank,
                    split,
                    frequencies,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // The checkpoint's `deepstack_visual_indexes` name the *tower's* blocks; the language model
        // adds the features in that order after its own first layers.
        let features = config.vision.deepstack_visual_indexes.len();
        if features > retained as usize {
            return Err(Error::config(format!(
                "{features} DeepStack features, but H3 retains {retained} blocks"
            )));
        }
        let deepstack_layers = (0..features).collect::<Vec<_>>();
        let sections = sections_of(config)?;
        Ok(Self {
            config: config.clone(),
            partition: *partition,
            dtype,
            embedding,
            rotary: Rotary::new(
                frequencies,
                config.head_dim,
                rank.query_heads,
                rank.key_value_heads,
                config.rms_norm_eps,
                crate::SPLIT_HALF,
                1.0,
            )
            .map_err(Error::from)?
            .sectioned(sections)
            .map_err(Error::from)?,
            layers,
            tower: VisionTower::load(loader, &config.vision, dtype)?,
            rows: Rows::new(dtype, hidden).map_err(Error::from)?,
            activation: GatedActivation::new(ACTIVATION),
            // Every sub-layer's output joins the stream unscaled.
            residual: Residual::default(),
            deepstack_layers,
            rank,
        })
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
    /// *layout* alone, so a worker uploads them once per bucket. A layout with no images needs
    /// the two prompt tables only.
    fn tables(&self, plan: &Plan, rows: u32) -> Vec<StateTable> {
        table_specs(&self.config, rows, &plan.images, &plan.time, plan.merged)
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

    /// Write the host bytes of every table of `layout`, in [`TextEncoder::tables`]' order.
    fn prepare_state_bytes(
        &self,
        prompt: &EncoderPrompt<'_>,
        layout: &EncoderLayout,
        out: &mut [TableValues],
    ) -> Result<()> {
        write_state(&self.config, prompt.ids, layout, out)
    }

    /// Refuse a call whose batch, layout or outputs do not agree.
    ///
    /// The tables are checked against their declared shapes by [`TextEncoder::prepared`]; this
    /// checks the three host values a captured call is addressed by.
    fn check(&self, inputs: &EncoderInputs<'_>, out: &EncoderOutputs, plan: &Plan) -> Result<()> {
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
            // A block's tower buffers are dead once its features are written, and the next block's
            // launches follow on the stream, so each block reuses the bytes of the one before it: a
            // request of several enlarged images took more than the arena holds.
            let block = f.arena().mark();
            let tower = self.tower.buffers(f.arena(), patches)?;
            let rows = u64::from(image_tables.rows.merged);
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
            f.arena().release(block);
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

        // 3. The prompt's M-RoPE factors, once for the whole stack, and the attention descriptor
        //    whose key bound is the batch's own (the reference declares none).
        if inputs.batch.max_kv_len == 0 {
            return Err(invalid_input(
                "the batch states no key bound its rows read".to_owned(),
            ));
        }
        let attention = Attention::dense(
            self.config.head_dim,
            self.rank.query_heads,
            self.rank.key_value_heads,
            inputs.batch.max_kv_len,
            true,
            None,
            (self.config.head_dim as f32).sqrt().recip(),
        )
        .map_err(Error::from)?;

        self.rotary
            .factors(f, prepared.positions, buffers.cosine, buffers.sine)
            .map_err(Error::from)?;

        // 4. The retained layers, with DeepStack table `j` added after layer `j`.
        let offsets = inputs.batch.query_offsets;
        let mut carried = None;
        let mut next = 0;
        for (index, layer) in self.layers.iter().enumerate() {
            f.fenced(TRANSFORMER_LAYER, format_args!("layers.{index}"), || {
                layer.forward(
                    self,
                    f,
                    &attention,
                    offsets,
                    &buffers,
                    carried,
                    hidden,
                    buffers.cosine,
                    buffers.sine,
                )
            })?;
            carried = Some(buffers.down);
            if self
                .deepstack_layers
                .get(next)
                .is_some_and(|block| *block == index)
            {
                // The reference's cut: the layer's own output joins the stream, then its features,
                // and the next layer normalizes the stream alone.
                self.residual
                    .forward(f, hidden, buffers.down, None, hidden)
                    .map_err(Error::from)?;
                if merged > 0 {
                    let features =
                        Buffers::rows_of(buffers.deepstack, next as u64 * merged, merged);
                    self.rows.add(f, hidden, features, prepared.destinations)?;
                }
                carried = None;
                next += 1;
            }
        }
        if let Some(carried) = carried {
            self.residual
                .forward(f, hidden, carried, None, hidden)
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
    fn layout(&self, batch: &nn::capability::TokenBatch, images: &[ImageShape]) -> EncoderLayout {
        EncoderLayout {
            rows: batch.num_tokens,
            tokens: batch.num_tokens,
            images: images.to_vec(),
        }
    }

    fn state_tables(&self, layout: &EncoderLayout) -> nn::Result<Vec<StateTable>> {
        // A layout the model cannot serve is refused here, by the plan that cannot hold it, rather
        // than answered with tables no encode of it would read.
        let plan = Plan::of_layout(&self.config, &self.partition, layout)?;
        Ok(self.tables(&plan, layout.rows))
    }

    fn prepare_state(
        &self,
        prompt: &EncoderPrompt<'_>,
        layout: &EncoderLayout,
        out: &mut [TableValues],
    ) -> nn::Result<()> {
        self.prepare_state_bytes(prompt, layout, out)
            .map_err(nn::Error::from)
    }

    fn encode(
        &self,
        f: &Forward,
        inputs: EncoderInputs<'_>,
        out: EncoderOutputs,
    ) -> nn::Result<()> {
        self.encode_prompt(f, inputs, out).map_err(nn::Error::from)
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

/// Write the host bytes of every table of `layout`, in the order `table_specs` declares them.
///
/// The arithmetic is the reference's request arithmetic, evaluated in FP32 on the host
/// (`positions::*`): a captured call reads only the device copies. Nothing here allocates device
/// memory, synchronizes or reads anything back - `out` holds the worker's host buffers.
fn write_state(
    config: &TextEncoderConfig,
    ids: &[u32],
    layout: &EncoderLayout,
    out: &mut [TableValues],
) -> Result<()> {
    let grids = layout
        .images
        .iter()
        .map(|image| image.grid)
        .collect::<Vec<_>>();
    let plan = Plan::new(config, &Partition::single(), layout.tokens, &grids)?;
    if plan.rows != layout.rows {
        return Err(Error::config(format!(
            "the layout reserves {} output rows for a {} token prompt",
            layout.rows, plan.rows
        )));
    }
    let tables = table_specs(config, layout.rows, &plan.images, &plan.time, plan.merged);
    if out.len() != tables.len() {
        return Err(Error::config(format!(
            "{} buffers for {} tables",
            out.len(),
            tables.len()
        )));
    }
    let vision = &config.vision;
    let merge = vision.spatial_merge_size;
    let tokens = plan.rows as usize;
    if ids.len() != tokens {
        return Err(Error::config(format!(
            "{} token ids for a {tokens} token prompt",
            ids.len()
        )));
    }

    // A vision grid reaches `rope_index` in the shape the prompt presents it: an image block is
    // one run of `time · rows · columns` placeholders with its whole grid, a video block is one
    // run per temporal patch, each of `rows · columns` placeholders on a single time step.
    let (image_grids, video_grids) = run_grids(
        ids,
        &plan.grids,
        config.image_token_id,
        config.video_token_id,
        merge,
    )?;
    let coordinates = positions::rope_index(
        ids,
        &image_grids,
        &video_grids,
        config.image_token_id,
        config.video_token_id,
        merge,
    )?;
    // `[tokens][3]` to the operator's `[axes, tokens]`.
    let axes = config.mrope_sections.len();
    let mut positions = vec![0_i32; axes * tokens];
    for (token, coordinate) in coordinates.iter().enumerate() {
        for axis in 0..axes {
            positions[axis * tokens + token] = i32::try_from(coordinate[axis])
                .map_err(|_| Error::config("an M-RoPE coordinate wider than an i32".to_owned()))?;
        }
    }
    out[0].write("prompt_positions", &TableValues::I32(positions.to_vec()))?;

    let destinations =
        positions::vision_destinations(ids, config.image_token_id, config.video_token_id);
    if destinations.len() != plan.merged as usize {
        return Err(Error::config(format!(
            "{} placeholders for {} merged vision rows",
            destinations.len(),
            plan.merged
        )));
    }
    out[1].write(
        "vision_destinations",
        &TableValues::I32(destinations.to_vec()),
    )?;
    if plan.images.is_empty() {
        return Ok(());
    }

    let side = BlockRows::position_grid(vision)?;
    let head_dim = BlockRows::head_dim(vision)?;
    let rotary = positions::VisionRotary::new(head_dim)?;
    let frequencies = positions::inverse_frequencies(head_dim / 2, ROPE_THETA)?;
    let mut index = Vec::new();
    let mut weight = Vec::new();
    let mut cosine = Vec::new();
    let mut sine = Vec::new();
    let mut lengths = Vec::new();
    // The tables are one run over the prompt's blocks, in prompt order, so a block's rows sit after
    // its predecessors' and the check below counts against the running total, not the block's own.
    let mut covered = 0_usize;
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
        covered += *patches as usize;
        if index.len() != covered * positions::POSITION_TAPS as usize
            || weight.len() != covered * positions::POSITION_TAPS as usize
            || cosine.len() != covered * columns
            || sine.len() != covered * columns
        {
            return Err(Error::config(format!(
                "the tables through the {grid:?} grid cover {} patches, {covered} expected",
                cosine.len() / columns
            )));
        }
    }
    out[2].write("vision_position_index", &TableValues::I32(index.to_vec()))?;
    out[3].write("vision_position_weight", &TableValues::F32(weight.to_vec()))?;
    out[4].write("vision_cosine", &TableValues::F32(cosine.to_vec()))?;
    out[5].write("vision_sine", &TableValues::F32(sine.to_vec()))?;
    Ok(out[6].write("vision_lengths", &TableValues::I32(lengths.to_vec()))?)
}

/// The grids [`positions::rope_index`] reads: one list per placeholder kind, in prompt order.
type GridRuns = (Vec<[u32; 3]>, Vec<[u32; 3]>);

/// The five vision tables' names and shapes for a concatenation of `patches` patch rows over
/// `time` time steps.
///
/// Total in its arguments: the shapes are arithmetic on the row counts, so a worker can ask for
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
    // The tower rotates on its own head width, which is not the language model's: the checkpoint
    // states 1152 elements over 16 heads, so the factors carry 36 frequencies rather than the text
    // stack's 64. The division is total here (this function reports nothing) and any configuration
    // that does not divide is refused by `write_state`, which derives the same width through
    // `BlockRows::head_dim`.
    let tower_columns = u64::from(config.vision.hidden_size / config.vision.num_heads.max(1) / 2);
    let table =
        |name: &'static str, dtype: nn::DType, shape: Shape| StateTable { name, dtype, shape };
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
            Shape::new(&[patch_rows, tower_columns]),
        ));
        tables.push(table(
            "vision_sine",
            nn::DType::F32,
            Shape::new(&[patch_rows, tower_columns]),
        ));
        tables.push(table(
            "vision_lengths",
            nn::DType::I32,
            Shape::new(&[steps]),
        ));
    }
    tables
}

/// The two grid lists [`positions::rope_index`] reads, one entry per vision placeholder run.
///
/// A vision grid reaches `rope_index` in the shape the prompt presents it:
/// an image block is one run of
/// `time · rows · columns` placeholders under its whole grid, a video block is one run per temporal
/// patch, each of `rows · columns` placeholders on a single time step. The run lengths are what
/// tells the two apart, and the grids arrive in the prompt's block order.
fn run_grids(
    ids: &[u32],
    grids: &[[u32; 3]],
    image_token_id: u32,
    video_token_id: u32,
    merge: u32,
) -> Result<GridRuns> {
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

/// A refusal of a call whose inputs do not describe the plan.
fn invalid_input(problem: String) -> Error {
    Error::Nn(nn::Error::from(ops::Error::Shape(format!(
        "TextEncoder: {problem}"
    ))))
}

/// The rotary frequency table: `theta^(-2j/head_dim)` for every rotated frequency, in FP32.
///
/// It is computed once here from the configuration (the frequency table is a loading-boundary
/// value) and published as a retained constant, so the factors op reads it as a
/// device weight like any other.
fn frequencies(loader: &mut WeightLoader<'_>, config: &TextEncoderConfig) -> Result<Weight> {
    let frequencies = positions::inverse_frequencies(config.head_dim, config.rope_theta)?;
    Ok(loader.scope(format!("{PREFIX}.rotary_emb")).constant_f32(
        "inv_freq",
        &[frequencies.len() as u64],
        &frequencies,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint's placeholder token ids.
    const IMAGE: u32 = 151_655;
    const VIDEO: u32 = 151_656;

    /// The released text encoder's configuration.
    fn config() -> TextEncoderConfig {
        crate::config::tests_support::released_config().text_encoder
    }

    #[test]
    fn the_mrope_cut_is_interleaved_and_agrees_with_the_host_rule() {
        let config = config();
        let sections = sections_of(&config).expect("the checkpoint's sections");
        assert!(sections.is_interleaved());
        assert_eq!(sections.axes(), 3);
        // The checkpoint's widths count frequencies; the operator's count rotated elements.
        let widths = Mrope::new(config.mrope_sections, config.head_dim)
            .expect("sections")
            .widths()
            .map(|width| width * 2);
        assert_eq!(sections.widths(), widths);
        assert_eq!(
            u64::from(widths.iter().sum::<u32>()),
            u64::from(config.head_dim)
        );
        // The device's assignment (`ops::rope::interleaved_axes`, the kernel's rule) names one
        // axis per compact frequency.
        assert_eq!(
            ops::rope::interleaved_axes(&widths).len(),
            (config.head_dim / 2) as usize
        );
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
    }

    #[test]
    fn four_ranks_split_the_heads_and_the_intermediate_width() {
        let config = config();
        let partition = Partition {
            tensor_parallel: 4,
            tensor_rank: 2,
            ..Partition::single()
        };
        let rank = Rank::new(&config, &partition).expect("four ranks");
        assert_eq!(rank.query_heads, 16);
        assert_eq!(rank.key_value_heads, 2);
        assert_eq!(rank.query, 16 * 128);
        assert_eq!(rank.key_value, 2 * 128);
        assert_eq!(rank.intermediate, 6_400);
    }

    #[test]
    fn a_degree_that_does_not_split_the_heads_is_refused() {
        let config = config();
        let partition = Partition {
            tensor_parallel: 3,
            ..Partition::single()
        };
        assert!(Rank::new(&config, &partition).is_err());
        let out_of_range = Partition {
            tensor_parallel: 4,
            tensor_rank: 4,
            ..Partition::single()
        };
        assert!(Rank::new(&config, &out_of_range).is_err());
    }

    /// A host table of the type and extents `table` states, as the worker hands it over.
    fn buffer_for(table: &StateTable) -> TableValues {
        TableValues::zeroed(table).expect("a host table")
    }

    /// The `index`-th element of an i32 host table.
    fn i32_at(table: &TableValues, index: usize) -> i32 {
        match table {
            TableValues::I32(values) => values[index],
            TableValues::F32(_) => panic!("an f32 table read as i32"),
        }
    }

    /// The `index`-th element of an f32 host table.
    fn f32_at(table: &TableValues, index: usize) -> f32 {
        match table {
            TableValues::F32(values) => values[index],
            TableValues::I32(_) => panic!("an i32 table read as f32"),
        }
    }

    #[test]
    fn a_text_prompt_needs_the_two_prompt_tables_only() {
        let config = config();
        let tables = table_specs(&config, 5, &[], &[], 0);
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0].name, "prompt_positions");
        assert_eq!(tables[0].shape.dims(), &[3, 5]);
        assert_eq!(tables[1].name, "vision_destinations");
        assert!(tables[1].shape.numel() == 0);

        let ids = [7, 8, 9, 10, 11];
        let layout = EncoderLayout {
            rows: 5,
            tokens: 5,
            images: Vec::new(),
        };
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        write_state(&config, &ids, &layout, &mut out).expect("a text prompt");

        // The positions are the reference's own, axis-major: axis `a`'s coordinates start at
        // `a * tokens`.
        let coordinates = positions::rope_index(
            &ids,
            &[],
            &[],
            config.image_token_id,
            config.video_token_id,
            config.vision.spatial_merge_size,
        )
        .expect("positions");
        for (token, coordinate) in coordinates.iter().enumerate() {
            for (axis, value) in coordinate.iter().enumerate() {
                assert_eq!(i32_at(&out[0], axis * 5 + token), *value as i32);
            }
        }
    }

    #[test]
    fn an_images_tables_are_one_time_steps_samples_repeated_over_its_grid() {
        let config = config();
        let merge = config.vision.spatial_merge_size;
        let side = BlockRows::position_grid(&config.vision).expect("a square table");
        let head_dim = BlockRows::head_dim(&config.vision).expect("a head width");
        // One 4x6 image block: 24 patches, 6 merged rows, after three prompt tokens.
        let ids = [7, 8, 9, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, 10];
        let layout = EncoderLayout {
            rows: 10,
            tokens: 10,
            images: vec![ImageShape { grid: [1, 4, 6] }],
        };
        let tables = table_specs(&config, 10, &[24], &[1], 6);
        assert_eq!(tables.len(), 7);
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        write_state(&config, &ids, &layout, &mut out).expect("an image prompt");

        // The six merged rows take the six placeholders, in order.
        let destinations = (3..9)
            .map(|row| i32_at(&out[1], row - 3))
            .collect::<Vec<_>>();
        assert_eq!(destinations, vec![3, 4, 5, 6, 7, 8]);

        // The resample's taps are one time step's, in merge-block patch order.
        let (index, weight) = positions::position_samples(side, 4, 6, merge).expect("samples");
        for patch in 0..24 {
            for tap in 0..4 {
                assert_eq!(i32_at(&out[2], patch * 4 + tap), index[patch][tap]);
                assert_eq!(f32_at(&out[3], patch * 4 + tap), weight[patch][tap]);
            }
        }
        // The rotary factors are the 2-D rotation of each patch's own coordinates.
        let coordinates = positions::rotary_coordinates(4, 6, merge).expect("coordinates");
        let frequencies = positions::inverse_frequencies(head_dim / 2, crate::vision::ROPE_THETA)
            .expect("frequencies");
        let rotary = positions::VisionRotary::new(head_dim).expect("a rotary head");
        let columns = (head_dim / 2) as usize;
        let mut cosine = vec![0.0_f32; coordinates.len() * columns];
        let mut sine = vec![0.0_f32; coordinates.len() * columns];
        rotary
            .factors(&coordinates, &frequencies, &mut cosine, &mut sine)
            .expect("factors");
        for element in 0..cosine.len() {
            assert_eq!(f32_at(&out[4], element), cosine[element]);
            assert_eq!(f32_at(&out[5], element), sine[element]);
        }
        // One time step is one attention sequence.
        assert_eq!(i32_at(&out[6], 0), 0);
        assert_eq!(i32_at(&out[6], 1), 24);

        // The prompt's positions start the block at the running origin.
        let coordinates = positions::rope_index(
            &ids,
            &[[1, 4, 6]],
            &[],
            config.image_token_id,
            config.video_token_id,
            merge,
        )
        .expect("positions");
        assert_eq!(i32_at(&out[0], 3), coordinates[3][0] as i32);
        assert_eq!(i32_at(&out[0], 10 + 3), coordinates[3][1] as i32);
    }

    #[test]
    fn a_video_is_one_run_per_temporal_patch_and_two_time_steps() {
        let config = config();
        let merge = config.vision.spatial_merge_size;
        // Two temporal patches of a 4x6 video: six placeholders each, behind a timestamp.
        let ids = [
            7, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO, 8, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO,
            VIDEO, 9,
        ];
        let layout = EncoderLayout {
            rows: 15,
            tokens: 15,
            images: vec![ImageShape { grid: [2, 4, 6] }],
        };
        let tables = table_specs(&config, 15, &[48], &[2], 12);
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        write_state(&config, &ids, &layout, &mut out).expect("a video prompt");

        // 48 patches merge to 12 rows, on the twelve placeholders.
        let destinations = (0..12).map(|row| i32_at(&out[1], row)).collect::<Vec<_>>();
        assert_eq!(destinations, vec![1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]);
        // Two time steps, each one sequence of 24 patches.
        assert_eq!(i32_at(&out[6], 0), 0);
        assert_eq!(i32_at(&out[6], 1), 24);
        assert_eq!(i32_at(&out[6], 2), 48);

        // The reference reads a video as one single-step grid per temporal patch, and each run
        // advances the origin.
        let (images, videos) = run_grids(
            &ids,
            &[[2, 4, 6]],
            config.image_token_id,
            config.video_token_id,
            merge,
        )
        .expect("runs");
        assert!(images.is_empty());
        assert_eq!(videos, vec![[1, 4, 6], [1, 4, 6]]);
        let coordinates = positions::rope_index(
            &ids,
            &images,
            &videos,
            config.image_token_id,
            config.video_token_id,
            merge,
        )
        .expect("positions");
        // One clock serves all three axes: a coordinate is the origin plus the patch's merged row
        // and column. The prompt's first token takes origin 0, the first run's patches sit at
        // `(1 + i, 1 + j)` because that token left origin 1, the run advances it by
        // `max(4, 6) / 2 = 3` to 4, the timestamp takes 4 and leaves 5, and the second run starts
        // there.
        assert_eq!(coordinates[0], [0, 0, 0]);
        assert_eq!(coordinates[1], [1, 1, 1]);
        assert_eq!(coordinates[6], [1, 2, 3]);
        assert_eq!(coordinates[7], [4, 4, 4]);
        assert_eq!(coordinates[8], [5, 5, 5]);
    }

    #[test]
    fn a_prompt_with_an_image_and_a_video_puts_both_in_one_table_run() {
        let config = config();
        let merge = config.vision.spatial_merge_size;
        // An image block of 4x4 after two text tokens, a comma, then a two-step 4x4 video: two
        // runs of four placeholders, behind its own text tokens.
        let ids = [
            7, 8, IMAGE, IMAGE, IMAGE, IMAGE, 9, VIDEO, VIDEO, VIDEO, VIDEO, 10, VIDEO, VIDEO,
            VIDEO, VIDEO, 11,
        ];
        let tokens = ids.len() as u32;
        let layout = EncoderLayout {
            rows: tokens,
            tokens,
            images: vec![
                ImageShape { grid: [1, 4, 4] },
                ImageShape { grid: [2, 4, 4] },
            ],
        };
        // The tables are one run over the prompt's blocks: sixteen patches of image, then the
        // video's thirty-two, in that order.
        let tables = table_specs(&config, tokens, &[16, 32], &[1, 2], 12);
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        write_state(&config, &ids, &layout, &mut out).expect("a prompt with an image and a video");

        // Four placeholders of an image, then four per temporal patch of the video: twelve merged
        // rows on twelve placeholders, in prompt order.
        let destinations = (0..12).map(|row| i32_at(&out[1], row)).collect::<Vec<_>>();
        assert_eq!(destinations, vec![2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15]);
        // Both calls' rows are in the same tables: forty-eight patches of taps, factors and
        // sequence lengths, the image's one single-step sequence first. The factors carry the
        // tower's own head width, not the language model's.
        let tower_head = BlockRows::head_dim(&config.vision).expect("a head width");
        assert_eq!(out[2].len(), 48 * positions::POSITION_TAPS as usize);
        assert_eq!(out[4].len(), 48 * (tower_head / 2) as usize);
        let lengths = (0..5).map(|row| i32_at(&out[6], row)).collect::<Vec<_>>();
        assert_eq!(lengths, vec![0, 16, 0, 16, 32]);

        // The blocks are consumed in prompt order, so the video's two runs come after the image's
        // and each block advances the origin: the image leaves 2, its timestamp takes 3, the video
        // leaves 5, and its own timestamp leaves 6 for the second run.
        let (images, videos) = run_grids(
            &ids,
            &[[1, 4, 4], [2, 4, 4]],
            config.image_token_id,
            config.video_token_id,
            merge,
        )
        .expect("runs");
        assert_eq!(images, vec![[1, 4, 4]]);
        assert_eq!(videos, vec![[1, 4, 4], [1, 4, 4]]);
        let coordinates = positions::rope_index(
            &ids,
            &images,
            &videos,
            config.image_token_id,
            config.video_token_id,
            merge,
        )
        .expect("positions");
        // The clock again: two text tokens leave origin 2, the image's four merged rows sit at
        // `(2 + i, 2 + j)`, the block advances by `max(4, 4) / 2 = 2` to 4, its timestamp takes 4
        // and leaves 5, the video's first run sits there, advances to 7, its timestamp leaves 8,
        // and the second run starts at 8.
        assert_eq!(coordinates[1], [1, 1, 1]);
        assert_eq!(coordinates[2], [2, 2, 2]);
        assert_eq!(coordinates[5], [2, 3, 3]);
        assert_eq!(coordinates[6], [4, 4, 4]);
        assert_eq!(coordinates[7], [5, 5, 5]);
        assert_eq!(coordinates[12], [8, 8, 8]);
    }

    #[test]
    fn a_run_that_does_not_hold_its_grid_is_refused() {
        let config = config();
        let merge = config.vision.spatial_merge_size;
        // A 4x6 image block merges to six rows; five placeholders are one short.
        let ids = [7, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE];
        assert!(
            run_grids(
                &ids,
                &[[1, 4, 6]],
                config.image_token_id,
                config.video_token_id,
                merge
            )
            .is_err()
        );
        // A grid with no run left, and a run with no grid, are both refused.
        assert!(
            run_grids(
                &[7, 8],
                &[[1, 4, 6]],
                config.image_token_id,
                config.video_token_id,
                merge
            )
            .is_err()
        );
        assert!(
            run_grids(
                &[7, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, 8],
                &[],
                config.image_token_id,
                config.video_token_id,
                merge
            )
            .is_err()
        );
        // A video grid of two time steps needs two runs.
        assert!(
            run_grids(
                &[7, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO, VIDEO, 8],
                &[[2, 4, 6]],
                config.image_token_id,
                config.video_token_id,
                merge
            )
            .is_err()
        );
    }

    #[test]
    fn a_buffer_of_the_wrong_size_is_refused() {
        let config = config();
        let layout = EncoderLayout {
            rows: 3,
            tokens: 3,
            images: Vec::new(),
        };
        let tables = table_specs(&config, 3, &[], &[], 0);
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        out[0] = TableValues::I32(vec![0; 2]);
        assert!(write_state(&config, &[1, 2, 3], &layout, &mut out).is_err());
        // A layout that reserves rows the prompt does not present is refused too.
        let mut out = tables.iter().map(buffer_for).collect::<Vec<_>>();
        let wide = EncoderLayout {
            rows: 4,
            tokens: 3,
            images: Vec::new(),
        };
        assert!(write_state(&config, &[1, 2, 3], &wide, &mut out).is_err());
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
        let merge = config.vision.spatial_merge_size;
        // The prompt's rows are the capacity the caller asked for; each image's patch rows are its
        // grid's patches, and its merged rows are the 2 by 2 blocks of each time step.
        assert_eq!(plan.rows, 4096);
        assert_eq!(plan.images, [24 * 24, 4 * 16 * 16]);
        assert_eq!(
            plan.merged,
            (24 / merge) * (24 / merge) + 4 * (16 / merge) * (16 / merge)
        );
        // One DeepStack tap per checkpoint layer the tower taps, and the width and head width the text
        // config states rather than a number this file owns.
        assert_eq!(
            plan.deepstack,
            config.vision.deepstack_visual_indexes.len() as u32
        );
        assert_eq!(plan.hidden, config.hidden_size);
        assert_eq!(plan.head_dim, config.head_dim);
    }

    #[test]
    fn the_plan_of_a_text_prompt_has_no_vision_rows() {
        let config = config();
        let plan = Plan::new(&config, &Partition::single(), 128, &[]).expect("one prompt");
        assert!(plan.images.is_empty());
        assert_eq!(plan.merged, 0);
        assert_eq!(plan.rows, 128);
    }

    #[test]
    fn the_frequency_table_is_the_checkpoints_rotary_recipe() {
        let config = config();
        let frequencies = crate::positions::inverse_frequencies(config.head_dim, config.rope_theta)
            .expect("the released recipe");
        assert_eq!(frequencies.len(), config.head_dim as usize / 2);
        assert_eq!(frequencies[0], 1.0);
        assert!(frequencies[63] < frequencies[0]);
    }
}
