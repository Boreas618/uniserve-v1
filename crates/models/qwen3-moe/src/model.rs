//! `Qwen3MoeForCausalLM`: the causal-LM block itself, composed of `nn` layers.
//!
//! ## What one forward does
//!
//! The layer equations are `uniserve/models/qwen3/transformer.py` (`Qwen3MoeDecoderLayer`,
//! `Qwen3MoeAttention`, `Qwen3MoeSparseMoeBlock`) and specs/worker.md 9's Qwen3-MoE rows:
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
//! ([`RmsNorm::forward_with_residual`]): the model never computes `hidden + residual` and then
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
//! Boundary; specs/worker.md 4.3).

use nn::capability::{CausalLm, KvView, TokenBatch};
use nn::forward::ALIGNMENT;
use nn::layers::{
    Attention, Embedding, Linear, Moe, ParallelLMHead, RmsNorm, Rotary, SwiGlu, TILE_ROWS,
    TRTLLM_WORKSPACE_BYTES,
};
use nn::model::{Block, BlockName, Model};
use nn::weights::Weight;
use nn::{Arena, DType, Forward, Shape, TensorView};
use std::sync::Arc;

use crate::{Config, Error, Partition, Result};

/// What a deployment sets when it loads a Qwen3-MoE model.
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// Bytes of transient scratch the loaded attention program needs per forward.
    pub attention_workspace_bytes: u64,
    /// Tokens one prefix-cache page holds; the deployment's cache decides it.
    pub page_tokens: u32,
    /// Whether the checkpoint's experts were read as calibrated NVFP4, which makes every routed
    /// layer's step encode its activations and read the packed products.
    ///
    /// It is a load option because it is a property of the checkpoint the loader read, and the plan
    /// has to know it: the encoded step's buffers are the plan's to reserve, and a plan built for
    /// dense experts would size an arena the encoded step writes past.
    pub experts_encoded: bool,
    /// Tokens one rank's expert exchange carries, on a deployment whose experts are sharded.
    ///
    /// Zero on a single rank, which has no exchange. On a group it is the **load-time** bound the
    /// plan is built with: a rank sorts `ranks · capacity` received rows over the experts it holds,
    /// so the step's buffers are shaped by this number and not by the batch's token count. A request
    /// is admitted within it; a batch larger than it is a deployment error rather than a buffer that
    /// grows.
    pub exchange_capacity: u64,
}

impl Default for LoadOptions {
    /// The BF16 vertical slice's deployment: 64-token pages and the trtllm-gen attention
    /// program's workspace.
    fn default() -> Self {
        Self {
            attention_workspace_bytes: TRTLLM_WORKSPACE_BYTES,
            page_tokens: 64,
            experts_encoded: false,
            exchange_capacity: 0,
        }
    }
}

/// The Qwen3-MoE causal language model.
pub struct Qwen3Moe {
    config: Config,
    partition: Partition,
    dtype: DType,
    embedding: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    head: ParallelLMHead,
    options: LoadOptions,
}

/// One decoder layer: two pre-norms and the two sublayers between them.
pub(crate) struct DecoderLayer {
    pub(crate) input_norm: RmsNorm,
    pub(crate) post_attention_norm: RmsNorm,
    pub(crate) attention: AttentionLayer,
    pub(crate) mlp: Mlp,
}

/// The attention sublayer: four projections, the fused Q/K normalization and rotation, and the
/// paged attention layer.
pub(crate) struct AttentionLayer {
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) o_proj: Linear,
    pub(crate) q_norm: Weight,
    pub(crate) k_norm: Weight,
    pub(crate) rotary: Rotary,
    pub(crate) attention: Attention,
}

/// The feed-forward sublayer.
///
/// The expert block is by far the larger variant and the common one (every layer of the slice
/// routes); boxing it would put an indirection on every forward to save a few hundred bytes in a
/// vector of 48 layers.
#[allow(
    clippy::large_enum_variant,
    reason = "the routed block is the common case and is held once per layer"
)]
pub(crate) enum Mlp {
    /// Routed experts: a replicated router and the grouped products over the local expert set.
    Experts(Moe),
    /// A dense block: `down(up(x) · silu(gate(x)))`.
    Dense(Box<DenseMlp>),
}

/// A dense feed-forward block.
pub(crate) struct DenseMlp {
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
    pub(crate) activation: SwiGlu,
}

/// One forward's activation geometry: the extents of the arena reservations, in reservation order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Plan {
    /// Tokens this forward processes.
    pub tokens: u64,
    /// The hidden width.
    pub hidden: u64,
    /// The fused output width of the query projection (this rank's heads).
    pub query: u64,
    /// The fused width of the key and value projections (this rank's heads).
    pub key_value: u64,
    /// Elements per rotary factor row (`head_dim / 2`).
    pub factors: u64,
    /// Elements per dense feed-forward row; zero when every layer routes to experts.
    pub feed_forward: u64,
    /// Elements per row of this rank's vocabulary shard: the head's output width.
    pub vocabulary: u64,
    /// Elements per row of the gathered logits matrix, `vocabulary × tensor parallel`; zero when
    /// one rank projects the whole vocabulary and no gather happens.
    pub gathered: u64,
    /// Bytes of the attention program's workspace.
    pub workspace: u64,
    /// Bytes one routed layer's expert step reserves; zero when no layer routes.
    ///
    /// One reservation serves every routed layer, as the rest of the plan does: the step's buffers
    /// are released as the layer loop advances, so a plan is one layer deep.
    ///
    /// What it covers depends on the deployment, and the difference is not a constant. A single rank
    /// reserves the router, the sort's metadata at the batch's height, and the permuted pair. An
    /// expert-parallel rank reserves the router at the batch's height, the sort's metadata at the
    /// **received** height (`ranks · capacity`) over the experts it holds, the pair at the rows that
    /// height permutes to ([`Self::exchange_rows`]), and one **combined** row per received token
    /// (`received · hidden · dtype`), which no single-rank path has.
    pub moe: u64,
    /// The exchange this plan was sized for, when the model's experts are sharded.
    pub exchange: Option<nn::layers::Group>,
    /// The permuted rows an expert-parallel rank sorts over: zero on a single rank.
    pub exchange_rows: u64,
}

impl Plan {
    /// The plan of one forward over `tokens` tokens of `dtype`'s activations.
    pub fn new(
        config: &Config,
        partition: &Partition,
        tokens: u64,
        options: &LoadOptions,
        dtype: DType,
    ) -> Result<Self> {
        // The exchange is a deployment, not a request: its rank count is the partition's and its
        // capacity is the load option's, so the plan can reserve the group's worst case before any
        // request exists. A sharded deployment that states no capacity has nothing to size with, and
        // guessing one is how a buffer ends up smaller than the rows that reach it.
        let exchange = match partition.expert_parallel {
            1 => None,
            ranks => {
                if options.exchange_capacity == 0 {
                    return Err(Error::Config(format!(
                        "an expert-parallel deployment of {ranks} ranks states no exchange \
                         capacity: the plan is sized from the deployment's bound, not from a request"
                    )));
                }
                Some(nn::layers::Group {
                    ranks: u64::try_from(ranks).map_err(|_| {
                        Error::Config(format!("{ranks} ranks exceed a plan's row counts"))
                    })?,
                    capacity: options.exchange_capacity,
                })
            }
        };
        // The block's own geometry, which both the single-rank and the exchanged step are stated
        // over: the vocabulary's expert count, and the tile the permutation packs in.
        let geometry = nn::layers::MoeGeometry {
            hidden: element(config.hidden_size, "hidden_size")?,
            intermediate: element(config.moe_intermediate_size, "moe_intermediate_size")?,
            top: element(config.num_experts_per_tok, "num_experts_per_tok")?,
            experts: element(config.num_experts, "num_experts")?,
            tile: u64::from(TILE_ROWS),
        };
        Ok(Self {
            tokens,
            hidden: element(config.hidden_size, "hidden_size")?,
            query: element(
                partition.local_attention_heads(config) * config.head_dim,
                "the query width",
            )?,
            key_value: element(
                partition.local_key_value_heads(config) * config.head_dim,
                "the key/value width",
            )?,
            factors: element(config.head_dim / 2, "head_dim / 2")?,
            feed_forward: if config.has_dense_layers() {
                element(config.intermediate_size, "intermediate_size")?
            } else {
                0
            },
            vocabulary: element(
                partition.padded_vocabulary_shard(config.vocab_size),
                "the vocabulary shard",
            )?,
            gathered: if partition.tensor_parallel > 1 {
                element(
                    partition.padded_vocabulary_shard(config.vocab_size)
                        * partition.tensor_parallel,
                    "the gathered vocabulary",
                )?
            } else {
                0
            },
            workspace: options.attention_workspace_bytes,
            moe: match exchange {
                None => {
                    // One routed layer's whole step: the routing prefix, the permuted pair, and the
                    // encoded planes when the checkpoint stores them encoded. Sized from the same
                    // `max_permuted_rows` the runtime reserves, so plan and step cannot disagree. The
                    // activation encoding's block scales are the CUTLASS product's plain layout.
                    if config.num_experts > 0 {
                        nn::layers::step_bytes(
                            tokens,
                            geometry,
                            ops::fp4::ScaleLayout::Linear,
                            dtype,
                            options.experts_encoded,
                        )?
                    } else {
                        0
                    }
                }
                Some(group) => {
                    // The exchanged step: the received rows over this rank's experts, and one
                    // combined row per received token — the shape the single-rank step cannot state.
                    if config.num_experts == 0 {
                        return Err(Error::Config(
                            "an expert-parallel deployment of a model with no experts".to_owned(),
                        ));
                    }
                    nn::layers::exchanged_bytes(tokens, geometry, group, dtype)?
                }
            },
            exchange,
            exchange_rows: match exchange {
                None => 0,
                Some(group) => nn::layers::exchanged_rows(tokens, geometry, group)?,
            },
        })
    }

    /// The bytes this plan hands out, in reservation order, every reservation aligned.
    pub fn bytes(&self, dtype: DType) -> Result<u64> {
        let mut total = 0u64;
        for (elements, element) in [
            (self.tokens * self.factors, DType::F32), // cosine
            (self.tokens * self.factors, DType::F32), // sine
            (self.workspace, DType::U8),              // attention workspace
            (self.tokens * self.hidden, dtype),       // normalized values
            (self.tokens * self.hidden, dtype),       // the layer's output
            (self.tokens * self.hidden, dtype),       // the residual stream
            (self.tokens * self.query, dtype),        // query
            (self.tokens * self.query, dtype),        // rotated query
            (self.tokens * self.key_value, dtype),    // key
            (self.tokens * self.key_value, dtype),    // rotated key
            (self.tokens * self.key_value, dtype),    // value
            (self.tokens * self.query, dtype),        // attended output
            (self.tokens * self.feed_forward, dtype), // dense gate
            (self.tokens * self.feed_forward, dtype), // dense up
            (self.moe, DType::U8),                    // the encoded expert step
            (self.tokens * self.vocabulary, dtype),   // the vocabulary shard's logits
            (self.tokens * self.gathered, dtype),     // the gathered logits, absent on one rank
        ] {
            let bytes = element
                .bytes()
                .ok_or_else(|| Error::Config(format!("{element} has no byte size")))?
                .checked_mul(elements)
                .ok_or_else(|| Error::Config("the activation plan overflows".to_owned()))?;
            total = at_alignment(total)
                .checked_add(bytes)
                .ok_or_else(|| Error::Config("the activation plan overflows".to_owned()))?;
        }
        Ok(total)
    }
}

/// `bytes` rounded up to the arena's alignment.
fn at_alignment(bytes: u64) -> u64 {
    bytes.next_multiple_of(ALIGNMENT)
}

/// A configuration count as an element count.
fn element(count: usize, what: &str) -> Result<u64> {
    u64::try_from(count).map_err(|_| Error::Config(format!("{what} does not fit an element count")))
}

/// One forward's activation views, reserved in the plan's order.
struct Buffers {
    workspace: TensorView,
    normalized: TensorView,
    update: TensorView,
    residual: TensorView,
    query: TensorView,
    query_rotated: TensorView,
    key: TensorView,
    key_rotated: TensorView,
    value: TensorView,
    attended: TensorView,
    /// The dense feed-forward pair; absent when every layer routes to experts, so the plan's
    /// zero feed-forward extent reserves nothing.
    dense: Option<DenseBuffers>,
}

/// The two dense feed-forward intermediates.
struct DenseBuffers {
    gate: TensorView,
    up: TensorView,
}

impl Buffers {
    /// Reserve the plan's regions from `arena`, in the plan's order, after the two factor tables
    /// the caller already reserved.
    fn reserve(arena: &Arena, plan: &Plan, dtype: DType) -> Result<Self> {
        let rows = |width: u64| Shape::new(&[plan.tokens, width]);
        Ok(Self {
            workspace: arena
                .tensor(DType::U8, Shape::new(&[plan.workspace.max(1)]))
                .map_err(Error::from)?,
            normalized: arena
                .tensor(dtype, rows(plan.hidden))
                .map_err(Error::from)?,
            update: arena
                .tensor(dtype, rows(plan.hidden))
                .map_err(Error::from)?,
            residual: arena
                .tensor(dtype, rows(plan.hidden))
                .map_err(Error::from)?,
            query: arena.tensor(dtype, rows(plan.query)).map_err(Error::from)?,
            query_rotated: arena.tensor(dtype, rows(plan.query)).map_err(Error::from)?,
            key: arena
                .tensor(dtype, rows(plan.key_value))
                .map_err(Error::from)?,
            key_rotated: arena
                .tensor(dtype, rows(plan.key_value))
                .map_err(Error::from)?,
            value: arena
                .tensor(dtype, rows(plan.key_value))
                .map_err(Error::from)?,
            attended: arena.tensor(dtype, rows(plan.query)).map_err(Error::from)?,
            dense: if plan.feed_forward == 0 {
                None
            } else {
                Some(DenseBuffers {
                    gate: arena
                        .tensor(dtype, rows(plan.feed_forward))
                        .map_err(Error::from)?,
                    up: arena
                        .tensor(dtype, rows(plan.feed_forward))
                        .map_err(Error::from)?,
                })
            },
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
        embedding: Embedding,
        layers: Vec<DecoderLayer>,
        norm: RmsNorm,
        head: ParallelLMHead,
        options: LoadOptions,
    ) -> Result<Self> {
        if layers.len() != config.num_hidden_layers {
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

    /// The activation plan of a forward over `tokens` tokens.
    /// The vocabulary embedding.
    pub(crate) fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    /// The decoder layers, in order.
    pub(crate) fn layers(&self) -> &[DecoderLayer] {
        &self.layers
    }

    /// The final normalization.
    pub(crate) fn norm(&self) -> &RmsNorm {
        &self.norm
    }

    /// The vocabulary head.
    pub(crate) fn head(&self) -> &ParallelLMHead {
        &self.head
    }

    pub fn plan(&self, tokens: u64) -> Result<Plan> {
        Plan::new(
            &self.config,
            &self.partition,
            tokens,
            &self.options,
            self.dtype,
        )
    }

    /// Run the decoder layers `0..=last` over `hidden`, and answer the layer's output rows.
    ///
    /// This is the bisection tool a wrong number needs: `forward` walks all 48 layers and then
    /// normalizes, so a defect shows up only as a bad logit row with nothing saying which layer
    /// produced it. Naming a layer runs the same code up to and including it, with the same rotary
    /// tables and the same residual stream, and returns the rows that layer wrote.
    ///
    /// `last` past the last layer is refused rather than clamped: a stage asking for layer 60 of a
    /// 48-layer model is a caller error, and clamping it would answer a different question.
    ///
    /// The answer is the **state** after layer `last` — the reference's `hidden_{last + 1}` — and
    /// not the block's update, which is one fold short of it (see [`Self::fold`]).
    ///
    /// **Each call reserves the plan's step from `f`'s arena and leaves it reserved**, because the
    /// answer lives in it: the rows a caller compares are the arena's own. A caller that runs two
    /// measurements in a row — the same layer twice, or a layer and then its prefix — must take a
    /// mark before the first and release it after it has read the first answer, or the second call
    /// asks for a second step past the region's end.
    pub fn forward_through(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        hidden: TensorView,
        last: usize,
    ) -> Result<TensorView> {
        if last >= self.layers.len() {
            return Err(Error::Config(format!(
                "layer {last} was asked for but the model has {}",
                self.layers.len()
            )));
        }
        let tokens = batch.tokens();
        let plan = self.plan(tokens)?;
        if hidden.shape.rank() != 2
            || hidden.shape.dim(0) != tokens
            || hidden.shape.dim(1) != plan.hidden
        {
            return Err(Error::Config(format!(
                "hidden is {:?}, [{tokens}, {}] expected",
                hidden.shape.dims(),
                plan.hidden
            )));
        }
        self.forward_from(
            f,
            batch,
            cache,
            hidden,
            Run {
                start: 0,
                last,
                stage: None,
            },
        )
    }

    /// The router's choice at one layer, from that layer's own input rows.
    ///
    /// `hidden` is the layer's input — the reference's `hidden_{layer}`, not its output — `[tokens,
    /// hidden]`; `ids` and `weights` are the caller's `[tokens, top]` i32 and F32. The layer's input
    /// norm runs, then its router: no attention, no expert, no other layer, and no fold into the
    /// stream, so nothing here can be confused with a state.
    ///
    /// The ids are the checkpoint's expert indices, before the expert map. This is the only way a
    /// caller gets the model's own routing for a layer: [`nn::layers::Moe::route`] takes rows the
    /// input norm has already produced, and the layers are private, so a comparison against a
    /// reference's `layer.mlp.gate` needs this call or none.
    ///
    /// A layer whose MLP is dense has no router, and is refused by name.
    ///
    /// **`hidden` is a state, not a block's update.** The residual stream below the layer and the
    /// update a layer's block produces have the same shape, so handing this method an update answers
    /// a well-formed question about the wrong tensor: nothing here can refuse it, and the difference
    /// shows only when the ids are compared. Above layer 0 the state exists only after the prefix,
    /// which is why a caller holding the reference's files passes `hidden_{layer}` and a caller
    /// without them has nothing correct to pass.
    pub fn route(
        &self,
        f: &Forward,
        hidden: TensorView,
        layer: usize,
        ids: TensorView,
        weights: TensorView,
    ) -> Result<()> {
        if layer >= self.layers.len() {
            return Err(Error::Config(format!(
                "layer {layer} was asked for but the model has {}",
                self.layers.len()
            )));
        }
        let tokens = hidden.rows();
        let plan = self.plan(tokens)?;
        if hidden.shape.rank() != 2
            || hidden.shape.dim(0) != tokens
            || hidden.shape.dim(1) != plan.hidden
        {
            return Err(Error::Config(format!(
                "hidden is {:?}, [{tokens}, {}] expected",
                hidden.shape.dims(),
                plan.hidden
            )));
        }
        let normalized = f
            .arena()
            .tensor(hidden.dtype, Shape::new(&[tokens, plan.hidden]))?;
        self.layers[layer].route(f, hidden, normalized, ids, weights)
    }

    /// The router's raw scores at one layer, from that layer's own input rows.
    ///
    /// `hidden` is the layer's input — the reference's `hidden_{layer}` — `[tokens, hidden]`, and
    /// `scores` is the caller's `[tokens, experts]` buffer of the same dtype, which receives the
    /// gate's product before the top-k softens it. The layer's input norm runs, then its gate, and
    /// nothing else.
    ///
    /// This is the measurement a routing difference needs first. `topk_softmax` applies the softmax
    /// itself, so the scores are the last point at which a *factor* is visible as a factor: scores
    /// that are a fraction of a reference's give nearly equal weights, an order that follows those
    /// scores, and ids that mostly agree — the exact signature of a routing defect that a comparison
    /// of weights alone would call "flat weights" and a comparison of ids alone would call "close".
    ///
    /// Like [`Self::route`], the answer is about a **state**, not a block's update, and a layer whose
    /// MLP is dense has no gate to answer from.
    pub fn gate(
        &self,
        f: &Forward,
        hidden: TensorView,
        layer: usize,
        scores: TensorView,
    ) -> Result<()> {
        if layer >= self.layers.len() {
            return Err(Error::Config(format!(
                "layer {layer} was asked for but the model has {}",
                self.layers.len()
            )));
        }
        let tokens = hidden.rows();
        let plan = self.plan(tokens)?;
        if hidden.shape.rank() != 2
            || hidden.shape.dim(0) != tokens
            || hidden.shape.dim(1) != plan.hidden
        {
            return Err(Error::Config(format!(
                "hidden is {:?}, [{tokens}, {}] expected",
                hidden.shape.dims(),
                plan.hidden
            )));
        }
        let normalized = f
            .arena()
            .tensor(hidden.dtype, Shape::new(&[tokens, plan.hidden]))?;
        self.layers[layer].gate(f, hidden, normalized, scores)
    }

    /// Run layers `start..=last` on the caller's rows, and answer the state after `last` (with
    /// `stage = None`) or that cut inside it.
    ///
    /// This is the clean-input measurement: the rows handed in become the run's residual stream, so
    /// the first layer of the run normalizes them alone instead of adding a previous layer's output,
    /// and nothing of `0..start` enters the answer's error. It is how a reference's own
    /// `hidden_start` measures one layer or a range of them without inheriting a prefix's divergence.
    ///
    /// Two facts a caller needs. **The cache planes are per layer**, and a run touches only the
    /// planes of the layers it runs: the keys and values the attention reads are the ones those
    /// layers write from these rows, so a prefill needs a cache whose pages hold no other sequence's
    /// rows and a decode needs the prefix already written into those same planes by an earlier run.
    /// And **the answer carries the layers' own roundings, no more**: the state after `last` is the
    /// fold of its block's update into the stream, computed by the same launch the next layer's input
    /// norm runs, so a cut inside a layer and the state above it differ by one such fold.
    pub fn forward_from(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        hidden: TensorView,
        run: Run,
    ) -> Result<TensorView> {
        let (update, residual, buffers) = self.walk(f, batch, cache, hidden, run)?;
        match run.stage {
            Some(_) => Ok(update),
            None => self.fold(f, update, residual, buffers.normalized),
        }
    }

    /// The state after a layer: its block's update folded into the residual stream.
    ///
    /// This is the tensor the reference calls `hidden_{layer+1}` — the whole state a layer hands
    /// the next one — and not the block's update, which is an order of magnitude smaller and on its
    /// own is not a state at all. Comparing one against the other would read as a collapse at every
    /// layer.
    ///
    /// The fold is the layer boundary's own operator, `add_rms_norm`, whose residual output is the
    /// sum: a boundary has no separate add, and the run accumulates the stream with exactly this
    /// launch at the next layer's input norm, so the state answered here is bit-identical to the
    /// one the run produces. The normalization the operator fuses is computed and discarded; its
    /// weights do not touch the sum.
    /// The sum lands in `update` itself, which the operator may alias, and the normalization it
    /// fuses is discarded into `scratch` — the last layer's dead `normalized` buffer — so the fold
    /// costs no arena beyond the step the walk already reserved.
    fn fold(
        &self,
        f: &Forward,
        update: TensorView,
        residual: TensorView,
        scratch: TensorView,
    ) -> Result<TensorView> {
        self.layers[0]
            .input_norm
            .forward_with_residual(f, update, residual, update, scratch)
            .map_err(Error::from)?;
        Ok(update)
    }

    /// Run layers `0..=layer` and stop inside `layer` after `stage`, answering what it wrote there.
    ///
    /// The prefix is run in full, so the layer's residual stream and every operand are the ones the
    /// whole run hands it: an in-layer walk whose prefix differed from the model's would answer a
    /// question about a model nobody runs. The cuts sit *inside* the layer, so `MlpOut` is the
    /// block's update and [`Self::forward_through`] is that update folded into the stream: the two
    /// differ by the boundary's residual add. The three cuts are one operator each, which is what
    /// lets a reference post-hook on `input_layernorm`, on `self_attn` and on `mlp` be their
    /// counterparts.
    ///
    /// The prefix's error is the cut's error floor: what the cut shows is a divergence *on top of*
    /// the state layers `0..layer` already produced, which is the walk's `hidden_{layer}` row. A cut
    /// that looks clean at layer 30 therefore means "layer 30 adds nothing to a prefix that is
    /// already off by the walk's error at 29", and never "layers 0..29 are right".
    pub fn forward_substage(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        hidden: TensorView,
        layer: usize,
        stage: Substage,
    ) -> Result<TensorView> {
        self.forward_from(
            f,
            batch,
            cache,
            hidden,
            Run {
                start: 0,
                last: layer,
                stage: Some(stage),
            },
        )
    }

    /// Run layers `0..=last`, or `0..last` and then `last` up to `tail`, and answer the rows the
    /// last run wrote together with the residual stream as it then stands.
    ///
    /// One loop serves the whole run, the layer walk and the in-layer walk. Three loops that
    /// reserve the same buffers and add the residual in the same order are three chances to
    /// disagree about the layer equations, and a bisection whose prefix differs from the model it
    /// bisects answers nothing.
    fn walk(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        hidden: TensorView,
        run: Run,
    ) -> Result<(TensorView, TensorView, Buffers)> {
        let Run { start, last, stage } = run;
        let tokens = batch.tokens();
        let plan = self.plan(tokens)?;
        if start > last || last >= self.layers.len() {
            return Err(Error::Config(format!(
                "layers {start}..={last} were asked for and the model has {}",
                self.layers.len()
            )));
        }
        if hidden.shape.rank() != 2
            || hidden.shape.dim(0) != tokens
            || hidden.shape.dim(1) != plan.hidden
        {
            return Err(Error::Config(format!(
                "hidden is {:?}, [{tokens}, {}] expected",
                hidden.shape.dims(),
                plan.hidden
            )));
        }
        if hidden.dtype != self.dtype {
            return Err(Error::Config(format!(
                "hidden is {}, the model is {}",
                hidden.dtype, self.dtype
            )));
        }

        // The two rotary factor tables are one forward's worth of trigonometry, shared by every
        // layer: the positions and the head geometry do not change between them.
        let cosine = f
            .arena
            .tensor(DType::F32, Shape::new(&[tokens, plan.factors]))?;
        let sine = f
            .arena
            .tensor(DType::F32, Shape::new(&[tokens, plan.factors]))?;
        self.factors(f, batch.positions, cosine, sine)?;
        let buffers = Buffers::reserve(f.arena, &plan, self.dtype)?;
        let mut residual = hidden;
        let mut input = hidden;
        for (offset, layer) in self.layers[start..=last].iter().enumerate() {
            let index = start + offset;
            // One layer's scratch is dead once its launches are enqueued: the plan is one layer
            // deep and the next layer is handed the same addresses, which is what a replayed graph
            // expects. The layer-independent buffers above were reserved before this mark.
            let mark = f.arena.mark();
            let out = layer.forward(
                f,
                batch,
                cache,
                index,
                offset == 0,
                &buffers,
                residual,
                buffers.residual,
                cosine,
                sine,
                input,
                if index == last { stage } else { None },
            )?;
            f.arena.release(mark);
            // The first layer read the embedding in `hidden` and wrote the first sum into the
            // plan's residual region; from here on the stream lives there and every layer
            // accumulates it in place.
            input = out;
            residual = buffers.residual;
        }
        Ok((input, residual, buffers))
    }

    /// The model as the executor sees it: one causal-LM block named `backbone`.
    pub fn into_model(self) -> Model {
        Model::new(vec![(
            BlockName::new("backbone"),
            Block::CausalLm(Arc::new(self)),
        )])
    }

    /// The number of tokens one page of the prefix cache holds.
    pub const fn page_tokens(&self) -> u32 {
        self.options.page_tokens
    }

    /// The rotary factor tables of `positions`, written into `cosine` and `sine`.
    fn factors(
        &self,
        f: &Forward,
        positions: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        self.layers[0]
            .attention
            .rotary
            .factors(f, positions, cosine, sine)
            .map_err(Error::from)
    }
}

impl DecoderLayer {
    /// Run the layer: normalize (or add and normalize for later layers), attend, then route,
    /// writing the layer's output into `update` and the new residual into `residual_out`.
    ///
    /// `residual` and `residual_out` differ for the first layer only: its input residual is the
    /// caller's `hidden` — the embedding — while its output must land in the plan's residual
    /// region, where every later layer accumulates in place.
    ///
    /// `stop` cuts the layer short for an in-layer walk and answers the tensor that cut produces;
    /// the sublayers it skips are not launched at all. With `stop = None` the answer is `update`,
    /// the layer's output.
    /// The router's raw scores at this layer: its own input norm, then its gate, and nothing else.
    fn gate(
        &self,
        f: &Forward,
        hidden: TensorView,
        normalized: TensorView,
        scores: TensorView,
    ) -> Result<()> {
        self.input_norm.forward(f, hidden, normalized)?;
        self.mlp.gate(f, normalized, scores)
    }

    /// The router's choice at this layer: its own input norm, then its router, and nothing else.
    ///
    /// `hidden` is the layer's *input* — the residual stream below it — and `normalized` is the
    /// caller's row buffer, because the norm's answer is what the router reads. This is the cut a
    /// routing comparison needs: the ids it produces come from the same rows through the same norm
    /// as the forward's, so a reference's `layer.mlp.gate` output compares with them directly.
    fn route(
        &self,
        f: &Forward,
        hidden: TensorView,
        normalized: TensorView,
        ids: TensorView,
        weights: TensorView,
    ) -> Result<()> {
        self.input_norm.forward(f, hidden, normalized)?;
        self.mlp.route(f, normalized, ids, weights)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the layer runs the reference's tuple through the plan's buffers"
    )]
    fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        layer: usize,
        first: bool,
        buffers: &Buffers,
        residual: TensorView,
        residual_out: TensorView,
        cosine: TensorView,
        sine: TensorView,
        hidden: TensorView,
        stop: Option<Substage>,
    ) -> Result<TensorView> {
        let normalized = buffers.normalized;
        let update = buffers.update;

        // `first` is the first layer *of this run*, which is layer 0 for the whole model and layer
        // `start` for a run that begins part way up: its input residual is the caller's rows, so it
        // normalizes alone and the addition starts at the next layer. `layer` is the cache plane and
        // the rotary table's layer, which is always the model's own index.
        if first {
            self.input_norm
                .forward(f, hidden, normalized)
                .map_err(Error::from)?;
        } else {
            self.input_norm
                .forward_with_residual(f, hidden, residual, residual, normalized)
                .map_err(Error::from)?;
        }
        if stop == Some(Substage::Normalized) {
            return Ok(normalized);
        }

        self.attention.forward(
            f, batch, cache, layer, buffers, normalized, cosine, sine, update,
        )?;
        if stop == Some(Substage::Attended) {
            return Ok(update);
        }

        self.post_attention_norm
            .forward_with_residual(f, update, residual, residual_out, normalized)
            .map_err(Error::from)?;

        match &self.mlp {
            Mlp::Experts(moe) => moe.forward(f, normalized, update).map_err(Error::from),
            Mlp::Dense(dense) => dense.forward(f, normalized, buffers, update),
        }?;
        Ok(update)
    }
}

impl AttentionLayer {
    /// Project, normalize and rotate Q and K, write them to the cache, attend, and project out.
    #[allow(
        clippy::too_many_arguments,
        reason = "the call carries the reference's operands and the plan's buffers"
    )]
    fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        layer: usize,
        buffers: &Buffers,
        normalized: TensorView,
        cosine: TensorView,
        sine: TensorView,
        out: TensorView,
    ) -> Result<()> {
        self.q_proj
            .forward(f, normalized, buffers.query)
            .map_err(Error::from)?;
        self.k_proj
            .forward(f, normalized, buffers.key)
            .map_err(Error::from)?;
        self.v_proj
            .forward(f, normalized, buffers.value)
            .map_err(Error::from)?;

        let query = buffers.query.reshape(headed(buffers.query, &self.q_norm)?);
        let key = buffers.key.reshape(headed(buffers.key, &self.k_norm)?);
        let value = buffers.value.reshape(key.shape);
        let query_rotated = buffers.query_rotated.reshape(query.shape);
        let key_rotated = buffers.key_rotated.reshape(key.shape);
        let attended = buffers.attended.reshape(query.shape);

        self.rotary
            .apply(
                f,
                query,
                key,
                Some(self.q_norm.view()),
                Some(self.k_norm.view()),
                cosine,
                sine,
                query_rotated,
                key_rotated,
            )
            .map_err(Error::from)?;

        self.attention
            .forward(
                f,
                batch,
                cache,
                0,
                layer,
                query_rotated,
                key_rotated,
                value,
                attended,
                buffers.workspace,
            )
            .map_err(Error::from)?;

        self.o_proj
            .forward(f, attended.reshape(buffers.attended.shape), out)
            .map_err(Error::from)?;
        // Row parallel: each rank's partial output is summed over the query-head split.
        if let Some(group) = f.group {
            group.all_reduce_sum(out, f.stream).map_err(Error::from)?;
        }
        Ok(())
    }
}

impl Mlp {
    /// The router's choice for this sublayer, on rows the layer's input norm already produced.
    ///
    /// A dense MLP has no router: the checkpoint's `mlp_only_layers` give such a layer a plain
    /// gated MLP, so there are no expert ids to answer and this refuses rather than inventing them.
    fn route(
        &self,
        f: &Forward,
        normalized: TensorView,
        ids: TensorView,
        weights: TensorView,
    ) -> Result<()> {
        match self {
            Mlp::Experts(moe) => moe.route(f, normalized, ids, weights).map_err(Error::from),
            Mlp::Dense(_) => Err(Error::Config(
                "this layer's MLP is dense: it chooses no experts, so it has no routing to answer"
                    .to_owned(),
            )),
        }
    }
}

impl Mlp {
    /// The router's raw scores for this sublayer, on rows the layer's input norm already produced.
    fn gate(&self, f: &Forward, normalized: TensorView, scores: TensorView) -> Result<()> {
        match self {
            Mlp::Experts(moe) => moe.gate(f, normalized, scores).map_err(Error::from),
            Mlp::Dense(_) => Err(Error::Config(
                "this layer's MLP is dense: it has no gate to answer scores from".to_owned(),
            )),
        }
    }
}

impl DenseMlp {
    /// `down(up · silu(gate))`.
    fn forward(
        &self,
        f: &Forward,
        normalized: TensorView,
        buffers: &Buffers,
        out: TensorView,
    ) -> Result<()> {
        let dense = buffers.dense.as_ref().ok_or_else(|| {
            Error::Config(
                "a dense block runs in a forward planned without dense buffers".to_owned(),
            )
        })?;
        self.gate_proj
            .forward(f, normalized, dense.gate)
            .map_err(Error::from)?;
        self.up_proj
            .forward(f, normalized, dense.up)
            .map_err(Error::from)?;
        // `silu(gate) · up`, written into the gate buffer: the activation applies to the branch
        // named `gate`, so the argument order is `(f, gate, value, out)` as in every gated layer.
        self.activation
            .forward(f, dense.gate, dense.up, dense.gate)
            .map_err(Error::from)?;
        self.down_proj
            .forward(f, dense.gate, out)
            .map_err(Error::from)
    }
}

/// The shape of `projection` seen as `[rows, heads, head_dim]`, with `head_dim` from a per-head
/// normalization weight.
fn headed(projection: TensorView, norm: &Weight) -> Result<Shape> {
    let head_dim = norm.view().shape.numel();
    let rows = projection.rows();
    let columns = projection.shape.last();
    if head_dim == 0 || rows == 0 || !columns.is_multiple_of(head_dim) {
        return Err(Error::Config(format!(
            "a {head_dim}-element head does not divide {columns} columns of {rows} rows"
        )));
    }
    Ok(Shape::new(&[rows, columns / head_dim, head_dim]))
}

/// Which layers a walk runs, and whether it stops inside the last of them.
///
/// The two ends are named rather than passed side by side: `forward_from(start, last)` with adjacent
/// `usize`s is a transposition waiting for a busy afternoon, and a run that silently executed
/// `last..=start` would answer a question nobody asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    /// The first layer the run executes.
    pub start: usize,
    /// The last layer it executes.
    pub last: usize,
    /// Where inside `last` it stops, or `None` to run that layer whole.
    pub stage: Option<Substage>,
}

/// How far through one layer an in-layer walk goes.
///
/// A whole-run bisection answers which *layer* diverges; a layer is four operators deep and its
/// output alone cannot say which of them. These are the three cuts a layer can be stopped at, in
/// the order the layer runs them, each naming the tensor the reference's module holds at that point:
///
/// - [`Substage::Normalized`] is `input_layernorm`'s output, residual add included for every layer
///   after the first (the first's input residual is the embedding itself);
/// - [`Substage::Attended`] is `self_attn`'s output, projected out and **before** the residual add;
/// - [`Substage::MlpOut`] is the routed (or dense) block's output, **before** its residual add —
///   transformers' `mlp` post-hook — and one fold short of the layer's output state, which is what
///   `forward_through` answers and which no cut names because the layer walk already shows it.
///
/// Each cut is named after the reference tensor that answers it, and a cut exists only where such a
/// tensor exists: a name ambiguous about the residual would produce a number that looks like
/// evidence and is off by a vector the size of the stream (rms ~7.3 in this model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Substage {
    /// The residual add and the input normalization.
    Normalized,
    /// The attention block's projection out, before the residual add.
    Attended,
    /// The routed (or dense) block's output, before its residual add (`mlp`).
    MlpOut,
}

impl Substage {
    /// Every arm with its spelling, in the order a layer runs them.
    ///
    /// The spellings are the contract with a command line that names a stage: the model refuses
    /// anything else, so a mistyped stage cannot silently run a different cut.
    pub const NAMES: [&'static str; 3] = ["normalized", "attended", "mlp_out"];

    /// The stage `name` asks for, or `None` when it names no cut.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "normalized" => Some(Self::Normalized),
            "attended" => Some(Self::Attended),
            "mlp_out" => Some(Self::MlpOut),
            _ => None,
        }
    }

    /// The spelling of this stage, one of [`Self::NAMES`].
    pub const fn name(self) -> &'static str {
        match self {
            Self::Normalized => "normalized",
            Self::Attended => "attended",
            Self::MlpOut => "mlp_out",
        }
    }
}

impl CausalLm for Qwen3Moe {
    fn embed_input_ids(&self, f: &Forward, ids: TensorView, out: TensorView) -> nn::Result<()> {
        self.embedding
            .forward(f, ids, out)
            .map_err(Error::from)
            .map_err(capability)
    }

    fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        hidden: TensorView,
    ) -> nn::Result<()> {
        let tokens = batch.tokens();
        let plan = self.plan(tokens).map_err(capability)?;
        if hidden.shape.rank() != 2
            || hidden.shape.dim(0) != tokens
            || hidden.shape.dim(1) != plan.hidden
        {
            return Err(capability(Error::Config(format!(
                "hidden is {:?}, [{tokens}, {}] expected",
                hidden.shape.dims(),
                plan.hidden
            ))));
        }
        if hidden.dtype != self.dtype {
            return Err(capability(Error::Config(format!(
                "hidden is {}, the model is {}",
                hidden.dtype, self.dtype
            ))));
        }

        // The rotary tables, the buffers and the residual stream are `walk`'s to reserve: the whole
        // run and a bisection must share one loop, or the bisection measures a different model.
        let last =
            self.layers.len().checked_sub(1).ok_or_else(|| {
                capability(Error::Config("the model has no decoder layer".into()))
            })?;
        let (input, residual, _) = self
            .walk(
                f,
                batch,
                cache,
                hidden,
                Run {
                    start: 0,
                    last,
                    stage: None,
                },
            )
            .map_err(capability)?;

        // The output norm: the last layer's output added into the residual stream and normalized.
        // The sum lands back in the stream, which is the last thing this forward needs from it, and
        // `add_rms_norm` loads a whole row before storing any of it, so the alias is safe.
        self.norm
            .forward_with_residual(f, input, residual, residual, hidden)
            .map_err(Error::from)
            .map_err(capability)
    }

    fn logits(&self, f: &Forward, hidden: TensorView, out: TensorView) -> nn::Result<()> {
        let rows = hidden.rows();
        let plan = self.plan(rows).map_err(capability)?;
        // The plan's two widths are the model's, not the launch's: a loaded head whose shard width
        // disagrees with the partition would otherwise be given a buffer of the wrong width and
        // write past it.
        if plan.vocabulary != u64::from(self.head.padded()) {
            return Err(capability(Error::Config(format!(
                "the plan reserves {} columns for the vocabulary shard, the head projects {}",
                plan.vocabulary,
                self.head.padded()
            ))));
        }
        let projected = f
            .arena
            .tensor(self.dtype, Shape::new(&[rows, plan.vocabulary]))
            .map_err(Error::from)
            .map_err(capability)?;
        let gathered = if plan.gathered == 0 {
            None
        } else {
            Some(
                f.arena
                    .tensor(self.dtype, Shape::new(&[rows, plan.gathered]))
                    .map_err(Error::from)
                    .map_err(capability)?,
            )
        };
        self.head
            .forward(f, hidden, projected, gathered, out)
            .map_err(Error::from)
            .map_err(capability)
    }
}

/// A model refusal as the capability trait's error.
///
/// The layer calls already answer [`nn::Error`]; this crate's own refusals (a projection that no
/// head divides, a hidden state of the wrong type) are shape disagreements of the model.
fn capability(error: Error) -> nn::Error {
    match error {
        Error::Nn(error) => error,
        other => nn::Error::Shape {
            owner: "Qwen3Moe",
            problem: other.to_string(),
        },
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

#[cfg(test)]
mod tests {
    /// The stage spellings are the contract with a command line: every name parses back to the
    /// stage it names, in the layer's own order, and a name that is not one of them answers nothing
    /// rather than a default. A bisection that silently ran a different cut would report the wrong
    /// operator with full confidence.
    #[test]
    fn a_substage_name_round_trips_and_anything_else_is_refused() {
        for (index, name) in Substage::NAMES.iter().enumerate() {
            let stage = Substage::parse(name).expect("a declared name parses");
            assert_eq!(stage.name(), *name);
            assert_eq!(Substage::NAMES[index], stage.name());
        }
        assert_eq!(Substage::parse("normalized"), Some(Substage::Normalized));
        assert_eq!(Substage::parse("attended"), Some(Substage::Attended));
        assert_eq!(Substage::parse("mlp_out"), Some(Substage::MlpOut));
        for refused in [
            "",
            "norm",
            "Normalized",
            "attention",
            "attn",
            "moe",
            "mlp",
            "mlpout",
        ] {
            assert_eq!(Substage::parse(refused), None, "{refused:?} named no cut");
        }
    }

    use super::*;
    use crate::Activation;

    fn config() -> Config {
        Config {
            vocab_size: 151936,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 2,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 128,
            hidden_act: Activation::Silu,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            max_position_embeddings: 40960,
            attention_bias: false,
            tie_word_embeddings: false,
            num_experts: 128,
            num_experts_per_tok: 8,
            moe_intermediate_size: 768,
            norm_topk_prob: true,
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
        }
    }

    #[test]
    fn the_plan_of_the_slice() {
        let plan = Plan::new(
            &config(),
            &Partition::single(),
            4,
            &LoadOptions::default(),
            DType::BF16,
        )
        .unwrap();
        assert_eq!(plan.hidden, 2048);
        assert_eq!(plan.query, 32 * 128);
        assert_eq!(plan.key_value, 4 * 128);
        assert_eq!(plan.factors, 64);
        // Every layer routes to experts, so no dense feed-forward buffer is planned.
        assert_eq!(plan.feed_forward, 0);
        // One rank projects the whole vocabulary and gathers nothing.
        assert_eq!(plan.vocabulary, 151936);
        assert_eq!(plan.gathered, 0);
    }

    #[test]
    fn a_dense_layer_is_planned() {
        let mut config = config();
        config.mlp_only_layers = vec![1];
        assert!(config.has_dense_layers());
        let plan = Plan::new(
            &config,
            &Partition::single(),
            2,
            &LoadOptions::default(),
            DType::BF16,
        )
        .unwrap();
        assert_eq!(plan.feed_forward, 6144);
    }

    #[test]
    fn tensor_parallel_heads_divide_the_projection() {
        let config = config();
        let partition = Partition::new(&config, 4, 1, 1, 0).unwrap();
        let plan = Plan::new(&config, &partition, 3, &LoadOptions::default(), DType::BF16).unwrap();
        assert_eq!(plan.query, 8 * 128);
        assert_eq!(plan.key_value, 128);
    }

    /// A sharded head plans both logits matrices: its own padded shard, and the gather every rank
    /// writes into. The plan is what sizes the arena, so a gather it forgot would be a write past
    /// the reservation on the first sharded deployment.
    #[test]
    fn a_sharded_head_plans_its_shard_and_the_gather() {
        let config = config();
        let single = Plan::new(
            &config,
            &Partition::single(),
            1,
            &LoadOptions::default(),
            DType::BF16,
        )
        .unwrap();
        let sharded = Plan::new(
            &config,
            &Partition::new(&config, 2, 0, 1, 0).unwrap(),
            1,
            &LoadOptions::default(),
            DType::BF16,
        )
        .unwrap();
        assert_eq!(sharded.vocabulary, 75968);
        assert_eq!(sharded.gathered, 151936);
        assert_eq!(single.gathered, 0);
        assert!(
            sharded.bytes(DType::BF16).unwrap() > sharded.vocabulary * 2,
            "the gather's own rows are reserved"
        );
    }

    /// An encoded expert step's buffers are the plan's to reserve: the arena is sized from the
    /// plan, and the step writes past a plan that forgot them. A dense deployment reserves its
    /// routing prefix and its permuted pair — the combine indexes that pair through the sort's map on
    /// either path — and an encoded one adds its activation planes and the products' scratch.
    #[test]
    fn an_encoded_expert_step_is_planned_and_a_dense_one_reserves_less() {
        let config = config();
        let partition = Partition::single();
        let dense = Plan::new(
            &config,
            &partition,
            15,
            &LoadOptions::default(),
            DType::BF16,
        )
        .unwrap();
        let encoded = Plan::new(
            &config,
            &partition,
            15,
            &LoadOptions {
                experts_encoded: true,
                ..LoadOptions::default()
            },
            DType::BF16,
        )
        .unwrap();
        // The permutation's worst case: 120 expanded routes over 128 experts at 128-row tiles is one
        // tile per route, so the step reserves 15360 permuted rows rather than 120.
        let routes = 120 * 128;
        let down_rows = routes * 2048 * 2;
        let gated_rows = routes * 768 * 2;
        assert!(
            encoded.moe >= down_rows,
            "the encoded step reserves {} bytes, the down product's rows alone are {down_rows}",
            encoded.moe
        );
        // The permuted pair is permuted-height on both paths: the combine indexes it through the
        // sort's map, so the dense path pays it too and the encoded path's addition is its
        // activation planes.
        assert!(
            dense.moe >= down_rows + gated_rows,
            "the dense step reserves {} bytes against the pair's {}",
            dense.moe,
            down_rows + gated_rows
        );
        assert!(encoded.moe > dense.moe);
        assert!(encoded.bytes(DType::BF16).unwrap() > dense.bytes(DType::BF16).unwrap());
    }

    #[test]
    fn the_plan_bytes_follow_the_token_count() {
        let config = config();
        let partition = Partition::single();
        let plan = |tokens: u64| {
            Plan::new(
                &config,
                &partition,
                tokens,
                &LoadOptions::default(),
                DType::BF16,
            )
            .unwrap()
        };
        let one = plan(1);
        let two = plan(2);
        // Per token: two F32 factor columns, three hidden buffers, three query-width buffers
        // (query, rotated query, attended output), three key/value-width buffers and the logits'
        // row. Two rank's worth of logits are planned only where a gather happens, and the single
        // rank plans none.
        let per_token = 2 * 64 * 4 + 3 * 2048 * 2 + 3 * 4096 * 2 + 3 * 512 * 2 + 151936 * 2;
        // The expert step grows with the batch as well, at the permutation's padded height rather
        // than at `tokens · top`, so its own term is the difference the plan states for it.
        let step = two.moe - one.moe;
        assert_eq!(
            two.bytes(DType::BF16).unwrap() - one.bytes(DType::BF16).unwrap(),
            per_token + step
        );
    }
}
