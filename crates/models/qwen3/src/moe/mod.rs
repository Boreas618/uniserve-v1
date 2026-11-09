//! `Qwen3MoeForCausalLM`: the Qwen3 decoder stack with routed experts, under a
//! vocabulary-parallel embedding and head.
//!
//! The layer equations are Hugging Face's `Qwen3MoeDecoderLayer`, `Qwen3MoeAttention` and
//! `Qwen3MoeSparseMoeBlock`, run by the family's [`DecoderStack`]:
//!
//! ```text
//! embedding:  hidden = Σ_ranks embed_shard(ids)                       (vocabulary parallel)
//! layers:     the decoder stack, paged attention over the prefix cache  (crate::decoder)
//! expert:     hidden = combine(routes, experts(hidden))               (top-k, FP32 weights)
//! final:      hidden = add_rms_norm(hidden, residual, norm)
//! ```
//!
//! - the embedding is vocabulary-parallel over the tensor ranks, zero-filled outside a rank's shard
//!   and summed with an all-reduce;
//! - attention applies per-head RMSNorm to Q and K, a single-axis rotary embedding, paged KV
//!   attention over the prefix cache, and a row-parallel output projection;
//! - the expert block routes each token to its top-k experts of a replicated router, renormalizes
//!   the FP32 weights when the checkpoint says so, and combines the experts' outputs in FP32;
//! - the final RMSNorm feeds the vocabulary-parallel head.
//!
//! ## Buffers
//!
//! Every activation comes from [`Forward::arena`], reserved in a fixed order that depends on the
//! token count and the model's dimensions alone, so a captured bucket replays over the same
//! addresses: the rotary factors, the attention program's workspace, then the stack's buffers.
//!
//! ## What this module may do
//!
//! The model receives borrowed execution resources in [`Forward`] and holds weights, its splits
//! and the checkpoint-derived configuration. It never allocates device memory, creates a stream,
//! captures a graph, selects a program or owns a runtime object (`AGENTS.md`, Model Library
//! Boundary).

pub mod load;

use nn::blocks::Keys;
use nn::capability::{CausalLm, KvView, TokenBatch};
use nn::family::Split;
use nn::layers::{PagedAttention, ParallelLMHead, ResidualStream, RmsNorm, VocabParallelEmbedding};
use nn::{DType, Forward, TensorView};

use crate::config::MoeConfig;
use crate::decoder::{Components, DecoderStack, Hidden, Mlp};
use crate::{Error, Result};

/// The components a deployment's fence probes name (`profiling_config.fence_components`): the
/// decoder stack, one decoder layer, and a layer's two sublayers.
const TRANSFORMER: &str = "models.qwen3.transformer.Transformer";
pub(crate) const COMPONENTS: Components = Components {
    layer: "models.qwen3.transformer.TransformerLayer",
    attention: Some("models.qwen3.transformer.Attention"),
    mlp: Some("models.qwen3.transformer.MoE"),
};

/// The Qwen3-MoE causal language model.
pub struct Qwen3Moe {
    config: MoeConfig,
    /// The tensor-parallel split every layer is sharded by.
    tensor: Split,
    dtype: DType,
    embedding: VocabParallelEmbedding,
    stack: DecoderStack,
    norm: RmsNorm,
    head: ParallelLMHead,
    /// Whether the checkpoint's experts were read as calibrated NVFP4, which makes every routed
    /// layer's step encode its activations and read the packed products.
    experts_encoded: bool,
}

impl Qwen3Moe {
    /// The checkpoint configuration this model was built for.
    pub const fn config(&self) -> &MoeConfig {
        &self.config
    }

    /// The tensor-parallel split this rank's layers hold.
    pub const fn tensor(&self) -> Split {
        self.tensor
    }

    /// The element type the activations of this model have.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Whether the checkpoint's expert stacks were read as calibrated NVFP4.
    pub const fn experts_encoded(&self) -> bool {
        self.experts_encoded
    }

    /// Run every decoder layer over `hidden`, answering the last layer's update together with the
    /// residual stream it has not yet been added into; the caller's output norm performs that add.
    ///
    /// The rotary tables and the stack's buffers are reserved from `f`'s arena in a fixed order
    /// before the first layer, so a replayed graph finds every operand at the address it was
    /// captured with; the attention program's workspace is the runtime's ([`KvView::workspace`]).
    fn layers(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        hidden: TensorView,
    ) -> Result<Hidden> {
        let tokens = batch.tokens();
        let (cosine, sine) = self.stack.factors(f, batch.positions, tokens)?;
        let buffers = self.stack.buffers(
            f.arena(),
            tokens,
            self.config.text.hidden_size,
            self.dtype,
            None,
        )?;
        // The workspace is the runtime's and not zero here: every bucket's forward reuses it, so the
        // decode program's arrival counters are cleared once before the first layer reads them.
        PagedAttention::zero_counters(f, cache.workspace)?;

        // The first layer reads the embedding in `hidden` and writes the first sum into the
        // forward's residual region; from there on the stream lives there and every layer
        // accumulates it in place.
        self.stack.forward(
            f,
            Hidden {
                hidden,
                residual: None,
            },
            cosine,
            sine,
            |layer| Keys::Paged {
                batch,
                cache,
                group: 0,
                layer,
            },
            &buffers,
            |_, state| Ok(state),
        )
    }
}

impl CausalLm for Qwen3Moe {
    fn join_expert_step(&self, f: &Forward) -> nn::Result<()> {
        for layer in self.stack.layers() {
            if let Mlp::Experts(moe) = &layer.mlp {
                // A layer's scratch is dead once its launches are enqueued, as in a forward.
                let mark = f.arena().mark();
                moe.join_expert_step(f)?;
                f.arena().release(mark);
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
        let width = u64::from(self.config.text.hidden_size);
        if hidden.shape.rank() != 2 || hidden.shape.dim(0) != tokens || hidden.shape.dim(1) != width
        {
            return Err(nn::Error::from(Error::Config(format!(
                "hidden is {:?}, [{tokens}, {width}] expected",
                hidden.shape.dims(),
            ))));
        }
        if hidden.dtype != self.dtype {
            return Err(nn::Error::from(Error::Config(format!(
                "hidden is {}, the model is {}",
                hidden.dtype, self.dtype
            ))));
        }

        f.fenced(TRANSFORMER, format_args!("model"), || {
            let last = self
                .layers(f, batch, cache, hidden)
                .map_err(nn::Error::from)?;
            let residual = last.residual.ok_or_else(|| {
                nn::Error::from(Error::Config(
                    "a causal language model needs at least one layer".to_owned(),
                ))
            })?;

            // The output norm: the last layer's output added into the residual stream and
            // normalized. The sum lands back in the stream, which is the last thing this forward
            // needs from it, and `add_rms_norm` loads a whole row before storing any of it, so the
            // alias is safe.
            self.norm.forward(
                f,
                last.hidden,
                Some(ResidualStream::in_place(residual)),
                hidden,
            )
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
            .field("layers", &self.stack.layers().len())
            .field("hidden", &self.config.text.hidden_size)
            .field("experts", &self.config.num_experts)
            .field("tensor", &self.tensor)
            .finish()
    }
}
