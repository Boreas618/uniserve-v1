//! `Qwen3MoeForCausalLM`: the causal-LM block itself, composed of `nn` layers.
//!
//! ## What one forward does
//!
//! The layer equations are Hugging Face's `Qwen3MoeDecoderLayer`, `Qwen3MoeAttention` and
//! `Qwen3MoeSparseMoeBlock`:
//!
//! ```text
//! embedding:  hidden = Σ_ranks embed_shard(ids)                       (vocabulary parallel)
//! layer 0:    residual = hidden; hidden = input_norm(hidden)
//! layer i>0:  hidden, residual = add_rms_norm(hidden, residual, input_norm)
//! attention:  q, k, v = q_proj(hidden), k_proj(hidden), v_proj(hidden)
//!             q, k = qk_norm_rope(q, k)                               (per-head RMSNorm + RoPE)
//!             hidden = o_proj(attend(q, k, v))                        (row parallel)
//!             hidden, residual = add_rms_norm(hidden, residual, post_attention_norm)
//! expert:     hidden = combine(routes, experts(hidden))               (top-k, FP32 weights)
//! final:      hidden = add_rms_norm(hidden, residual, norm)
//! ```
//!
//! The residual stream is carried separately, summed in FP32, and normalized once per sublayer
//! ([`RmsNorm::forward`] with a [`ResidualStream`]): the model never computes `hidden + residual` and then
//! normalizes the rounded sum.
//!
//! ## Buffers
//!
//! Every activation comes from [`Forward::arena`], reserved in a fixed order that depends on the
//! token count and the model's geometry alone, so a captured bucket replays over the same
//! addresses. There is one set of buffers per forward, not one per layer: the normalized values,
//! the layer's output, the residual stream and the attention operands are reused as the layer loop
//! advances, as the reference reuses its tuple.
//!
//! ## What this crate may do
//!
//! The model receives borrowed execution resources in [`Forward`] and holds weights, the partition
//! and the checkpoint-derived configuration. It never allocates device memory, creates a stream,
//! captures a graph, selects a program or owns a runtime object (`AGENTS.md`, Model Library
//! Boundary).

use nn::blocks::{GatedMlp, Keys, QkvAttention, RopeFactors};
use nn::capability::{CausalLm, KvView, TokenBatch};
use nn::layers::{
    Moe, PagedAttention, ParallelLMHead, ResidualStream, RmsNorm, Rotary, TRTLLM_WORKSPACE_BYTES,
    VocabParallelEmbedding,
};
use nn::{Arena, DType, Forward, Shape, Strides, TensorView};

use crate::{Config, Error, Partition, Result};

/// The components a deployment's fence probes name (`profiling_config.fence_components`): the
/// decoder stack, one decoder layer, and a layer's two sublayers.
const TRANSFORMER: &str = "models.qwen3.transformer.Transformer";
const TRANSFORMER_LAYER: &str = "models.qwen3.transformer.TransformerLayer";
const ATTENTION: &str = "models.qwen3.transformer.Attention";
const MOE: &str = "models.qwen3.transformer.MoE";

/// What a deployment sets when it loads a Qwen3-MoE model.
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// Bytes of transient scratch the loaded attention program needs per forward.
    pub attention_workspace_bytes: u64,
    /// Tokens one prefix-cache page holds; the deployment's cache decides it.
    pub page_tokens: u32,
    /// Whether the checkpoint's experts were read as calibrated NVFP4, which makes every routed
    /// layer's step encode its activations and read the packed products.
    pub experts_encoded: bool,
}

impl Default for LoadOptions {
    /// A single-rank BF16 deployment: 64-token pages and the trtllm-gen attention program's
    /// workspace.
    fn default() -> Self {
        Self {
            attention_workspace_bytes: TRTLLM_WORKSPACE_BYTES,
            page_tokens: 64,
            experts_encoded: false,
        }
    }
}

/// The Qwen3-MoE causal language model.
pub struct Qwen3Moe {
    config: Config,
    partition: Partition,
    dtype: DType,
    embedding: VocabParallelEmbedding,
    /// The stack's one rotary embedding: every layer turns by the factors it computes once.
    rotary: Rotary,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    head: ParallelLMHead,
    options: LoadOptions,
}

/// One decoder layer: two pre-norms and the two sublayers between them.
pub(crate) struct DecoderLayer {
    pub(crate) input_norm: RmsNorm,
    pub(crate) post_attention_norm: RmsNorm,
    /// The attention: Q/K/V column parallel over the heads, per-head gains fused into the stack's
    /// rotation, paged attention, and the row-parallel output projection.
    pub(crate) attention: QkvAttention,
    pub(crate) mlp: Mlp,
}

/// The feed-forward sublayer.
///
/// The expert block is by far the larger variant and the common one (every layer of
/// Qwen3-30B-A3B routes); boxing it would put an indirection on every forward to save a few
/// hundred bytes in a vector of 48 layers.
#[allow(
    clippy::large_enum_variant,
    reason = "the routed block is the common case and is held once per layer"
)]
pub(crate) enum Mlp {
    /// Routed experts: a replicated router and the grouped products over the local expert set.
    Experts(Moe),
    /// A dense block: `down(up(x) · silu(gate(x)))`, column parallel in and row parallel out.
    Dense(Box<GatedMlp>),
}

/// One forward's activation extents, in elements: the widths the forward reserves its buffers at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Extents {
    /// Tokens this forward processes.
    tokens: u64,
    /// The row class the activation buffers are reserved at ([`nn::layers::packed_rows`]).
    ///
    /// `tokens` unless a projection is packed: the packed product is compiled per row class, so it
    /// writes whole tiles and every buffer it writes must hold the class. The request's rows are a
    /// prefix of each, and every op that reads such a buffer reads the prefix, which is why the
    /// buffers are padded here and not re-shaped downstream.
    padded: u64,
    /// The hidden width.
    hidden: u64,
    /// Elements per rotary factor row (`head_dim / 2`).
    factors: u64,
    /// Bytes of the attention program's workspace.
    workspace: u64,
}

impl Extents {
    /// The extents of one forward over `tokens` tokens; `packed` states whether the checkpoint's
    /// projections are packed, which reserves the row-padded buffers at the class that holds them.
    fn new(config: &Config, tokens: u64, options: &LoadOptions, packed: bool) -> Result<Self> {
        let padded = if packed {
            let counted = u32::try_from(tokens)
                .map_err(|_| Error::Config(format!("{tokens} activation rows")))?;
            u64::from(nn::layers::packed_rows(counted)?)
        } else {
            tokens
        };
        Ok(Self {
            tokens,
            padded,
            hidden: element(config.hidden_size, "hidden_size")?,
            factors: element(config.head_dim / 2, "head_dim / 2")?,
            workspace: options.attention_workspace_bytes,
        })
    }
}

/// The first `rows` rows of a row-major `buffer` whose rows are `width` elements wide.
///
/// The pointer is the buffer's: a shorter view of a larger region is what lets one reservation hold
/// the row class a packed product writes and the request's prefix every other op reads.
fn prefix_rows(buffer: TensorView, rows: u64, width: u64) -> TensorView {
    let shape = Shape::new(&[rows, width]);
    TensorView::with_strides(buffer.ptr, buffer.dtype, shape, Strides::row_major(shape))
}

/// A configuration count as an element count.
fn element(count: u32, _what: &str) -> Result<u64> {
    Ok(u64::from(count))
}

/// One forward's activation views, reserved in a fixed order.
struct Buffers {
    workspace: TensorView,
    normalized: TensorView,
    update: TensorView,
    residual: TensorView,
}

impl Buffers {
    /// Reserve the forward's regions from `arena`, after the two factor tables the caller already
    /// reserved.
    fn reserve(arena: &Arena, extents: &Extents, dtype: DType) -> Result<Self> {
        // Two heights, and both are needed: the region is the row class an encoded projection
        // writes whole tiles into, while the view every other op reads is the request's prefix of
        // it. A buffer reserved at the request's rows is one a 128-row launch writes past.
        let rows = |width: u64| Shape::new(&[extents.tokens, width]);
        let padded = |width: u64| -> Result<TensorView> {
            let capacity = arena
                .tensor(dtype, Shape::new(&[extents.padded, width]))
                .map_err(Error::from)?;
            Ok(prefix_rows(capacity, extents.tokens, width))
        };
        Ok(Self {
            workspace: arena
                .tensor(DType::U8, Shape::new(&[extents.workspace.max(1)]))
                .map_err(Error::from)?,
            normalized: arena
                .tensor(dtype, rows(extents.hidden))
                .map_err(Error::from)?,
            update: padded(extents.hidden)?,
            residual: arena
                .tensor(dtype, rows(extents.hidden))
                .map_err(Error::from)?,
        })
    }
}

impl Qwen3Moe {
    /// The loaded model from its parts; the loading boundary is [`crate::load`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the model is its configuration, its partition, its three module groups and the \
                  deployment's options; the loader names each once"
    )]
    pub(crate) fn new(
        config: Config,
        partition: Partition,
        dtype: DType,
        embedding: VocabParallelEmbedding,
        rotary: Rotary,
        layers: Vec<DecoderLayer>,
        norm: RmsNorm,
        head: ParallelLMHead,
        options: LoadOptions,
    ) -> Result<Self> {
        if layers.len() != config.num_hidden_layers as usize {
            return Err(Error::Config(format!(
                "{} layers for {} blocks",
                layers.len(),
                config.num_hidden_layers
            )));
        }
        if layers.is_empty() {
            return Err(Error::Config(
                "a causal language model needs at least one layer".to_owned(),
            ));
        }
        Ok(Self {
            config,
            partition,
            dtype,
            embedding,
            rotary,
            layers,
            norm,
            head,
            options,
        })
    }

    /// The checkpoint configuration this model was built for.
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// The deployment's mathematical partition.
    pub const fn partition(&self) -> &Partition {
        &self.partition
    }

    /// The element type the activations of this model have.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// The activation extents of a forward over `tokens` tokens.
    fn extents(&self, tokens: u64) -> Result<Extents> {
        Extents::new(
            &self.config,
            tokens,
            &self.options,
            self.projections_packed(),
        )
    }

    /// Whether the checkpoint stored this model's projections packed, which reserves the row-padded
    /// buffers at the packed product's row class.
    fn projections_packed(&self) -> bool {
        self.layers
            .first()
            .is_some_and(|layer| layer.attention.query().is_packed())
    }

    /// Run every decoder layer over `hidden`, answering the last layer's update together with the
    /// residual stream it has not yet been added into; the caller's output norm performs that add.
    ///
    /// `extents` are the extents of `batch`'s token count, and `hidden` has already been checked
    /// against them. The rotary tables, the buffers and the attention counter are reserved from `f`'s
    /// arena in a fixed order before the first layer, so a replayed graph finds every operand at the
    /// address it was captured with.
    fn layers(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        hidden: TensorView,
        extents: &Extents,
    ) -> Result<(TensorView, TensorView)> {
        let tokens = batch.tokens();

        // The two rotary factor tables are one forward's worth of trigonometry, shared by every
        // layer: the positions and the head geometry do not change between them.
        let cosine = f
            .arena
            .tensor(DType::F32, Shape::new(&[tokens, extents.factors]))?;
        let sine = f
            .arena
            .tensor(DType::F32, Shape::new(&[tokens, extents.factors]))?;
        self.rotary.factors(f, batch.positions, cosine, sine)?;
        let buffers = Buffers::reserve(f.arena, extents, self.dtype)?;
        // The arena is not zero here: every bucket's extents share one range, so the decode
        // program's arrival counters are cleared once before the first layer reads them.
        PagedAttention::zero_counters(f, buffers.workspace)?;

        let mut residual = hidden;
        let mut input = hidden;
        for (index, layer) in self.layers.iter().enumerate() {
            // One layer's scratch is dead once its launches are enqueued: the scratch is one layer
            // deep and the next layer is handed the same addresses, which is what a replayed graph
            // expects. The layer-independent buffers above were reserved before this mark.
            let mark = f.arena.mark();
            let out = f.fenced(TRANSFORMER_LAYER, format_args!("layers.{index}"), || {
                layer.forward(
                    f,
                    batch,
                    cache,
                    index,
                    &self.rotary,
                    &buffers,
                    residual,
                    buffers.residual,
                    cosine,
                    sine,
                    input,
                )
            })?;
            f.arena.release(mark);
            // The first layer read the embedding in `hidden` and wrote the first sum into the
            // forward's residual region; from here on the stream lives there and every layer
            // accumulates it in place.
            input = out;
            residual = buffers.residual;
        }
        Ok((input, residual))
    }

    /// Whether the checkpoint's expert stacks were read as calibrated NVFP4.
    pub const fn experts_encoded(&self) -> bool {
        self.options.experts_encoded
    }

    /// The tokens one prefix-cache page holds, which the deployment's cache chose.
    pub const fn page_tokens(&self) -> u32 {
        self.options.page_tokens
    }
}

impl DecoderLayer {
    /// Run the layer: normalize (or add and normalize for later layers), attend, then route,
    /// writing the layer's output into `update` and the new residual into `residual_out`.
    ///
    /// `residual` and `residual_out` differ for the first layer only: its input residual is the
    /// caller's `hidden` — the embedding — while its output must land in the forward's residual
    /// region, where every later layer accumulates in place.
    ///
    /// `layer` is the model's own index of this layer: it names the cache plane the attention
    /// writes and reads, and layer 0 is the one whose input residual is the embedding.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer runs the reference's tuple through the forward's buffers"
    )]
    fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        layer: usize,
        rotary: &Rotary,
        buffers: &Buffers,
        residual: TensorView,
        residual_out: TensorView,
        cosine: TensorView,
        sine: TensorView,
        hidden: TensorView,
    ) -> Result<TensorView> {
        let normalized = buffers.normalized;
        let update = buffers.update;

        // Layer 0's input residual is the embedding itself, so it normalizes alone and the
        // addition starts at the next layer.
        if layer == 0 {
            self.input_norm.forward(f, hidden, None, normalized)?;
        } else {
            self.input_norm
                .forward(
                    f,
                    hidden,
                    Some(ResidualStream::in_place(residual)),
                    normalized,
                )
                .map_err(Error::from)?;
        }

        f.fenced(ATTENTION, format_args!("layers.{layer}.self_attn"), || {
            let rope = RopeFactors {
                rotary,
                cosine,
                sine,
            };
            let keys = Keys::Paged {
                batch,
                cache,
                group: 0,
                layer,
                workspace: buffers.workspace,
            };
            Ok::<(), Error>(
                self.attention
                    .forward(f, normalized, Some(rope), keys, update)?,
            )
        })?;

        let stream = ResidualStream {
            input: residual,
            output: residual_out,
        };
        self.post_attention_norm
            .forward(f, update, Some(stream), normalized)
            .map_err(Error::from)?;

        f.fenced(MOE, format_args!("layers.{layer}.mlp"), || {
            match &self.mlp {
                // Both blocks sum their tensor-parallel partial outputs themselves: the routed
                // block after its experts, the dense one in its row-parallel down projection.
                Mlp::Experts(moe) => moe.forward(f, normalized, update).map_err(Error::from)?,
                Mlp::Dense(dense) => dense.forward(f, normalized, update)?,
            }
            Ok::<(), Error>(())
        })?;
        Ok(update)
    }
}

impl CausalLm for Qwen3Moe {
    fn join_expert_step(&self, f: &Forward) -> nn::Result<()> {
        for layer in &self.layers {
            if let Mlp::Experts(moe) = &layer.mlp {
                // A layer's scratch is dead once its launches are enqueued, as in a forward.
                let mark = f.arena.mark();
                moe.join_expert_step(f)?;
                f.arena.release(mark);
            }
        }
        Ok(())
    }

    fn embed_input_ids(&self, f: &Forward, ids: TensorView, out: TensorView) -> nn::Result<()> {
        self.embedding.forward(f, ids, out)
    }

    fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        hidden: TensorView,
    ) -> nn::Result<()> {
        let tokens = batch.tokens();
        let extents = self.extents(tokens).map_err(nn::Error::from)?;
        if hidden.shape.rank() != 2
            || hidden.shape.dim(0) != tokens
            || hidden.shape.dim(1) != extents.hidden
        {
            return Err(nn::Error::from(Error::Config(format!(
                "hidden is {:?}, [{tokens}, {}] expected",
                hidden.shape.dims(),
                extents.hidden
            ))));
        }
        if hidden.dtype != self.dtype {
            return Err(nn::Error::from(Error::Config(format!(
                "hidden is {}, the model is {}",
                hidden.dtype, self.dtype
            ))));
        }

        f.fenced(TRANSFORMER, format_args!("model"), || {
            let (input, residual) = self
                .layers(f, batch, cache, hidden, &extents)
                .map_err(nn::Error::from)?;

            // The output norm: the last layer's output added into the residual stream and
            // normalized. The sum lands back in the stream, which is the last thing this forward
            // needs from it, and `add_rms_norm` loads a whole row before storing any of it, so the
            // alias is safe.
            self.norm
                .forward(f, input, Some(ResidualStream::in_place(residual)), hidden)
        })
    }

    fn logits(&self, f: &Forward, hidden: TensorView, out: TensorView) -> nn::Result<()> {
        // The head reserves its shard's projection and the group's gather itself.
        self.head.forward(f, hidden, out)
    }
}

impl std::fmt::Debug for Qwen3Moe {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen3Moe")
            .field("layers", &self.layers.len())
            .field("hidden", &self.config.hidden_size)
            .field("experts", &self.config.num_experts)
            .field("partition", &self.partition)
            .finish()
    }
}
