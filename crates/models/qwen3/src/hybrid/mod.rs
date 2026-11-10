//! The Qwen3.5-MoE language model (`Qwen3_5MoeForConditionalGeneration`'s text model, for example
//! Qwen3.6-35B-A3B): a hybrid stack of gated delta networks and gated full attention, every layer
//! routed with a gated shared expert, under a vocabulary-parallel embedding and head.
//!
//! The layer equations are Hugging Face's `Qwen3_5MoeDecoderLayer`, run by the family's
//! [`DecoderStack`]:
//!
//! ```text
//! embedding:  hidden = Σ_ranks embed_shard(ids)                         (vocabulary parallel)
//! linear:     hidden = linear_attn(hidden)     a gated delta network over the slot's state
//! attention:  hidden = o_proj(attention(q, k, v) · sigmoid(gate))   q/k zero-centred RMSNorm,
//!                                              partial rotary, paged attention over the cache
//! experts:    hidden = combine(routes, experts(hidden)) + sigmoid(g · hidden) · shared(hidden)
//! norms:      x̂ · (1 + w)                       every RMSNorm but the delta network's output norm
//! ```
//!
//! ## Rotary positions
//!
//! The checkpoint rotates the leading quarter of each attention head by interleaved M-RoPE over
//! three position axes. A chat text request places every token at the same position on all three
//! axes, under which interleaved M-RoPE is the default spectrum over the rotated channels: the
//! frequencies are the same and each reads the one position. The model therefore rotates text by
//! that spectrum ([`text_spectrum`]); a request carrying vision input states three axes and reads
//! the sectioned spectrum the configuration holds.
//!
//! ## State
//!
//! Each gated delta network keeps a short-convolution window and a delta-rule state per request
//! slot ([`nn::family::StateShape`]); the cache the forward receives carries their banks, which the
//! infrastructure allocates, restores and checkpoints. The model indexes the key/value cache by
//! attention layer and the state by linear layer, each in stack order.
//!
//! ## What this module may do
//!
//! The model receives borrowed execution resources in [`Forward`] and holds weights, its splits and
//! the checkpoint-derived configuration. It never allocates device memory, creates a stream,
//! captures a graph, selects a program or owns a runtime object (`AGENTS.md`, Model Library
//! Boundary).

pub mod load;

use nn::blocks::{GatedDeltaNet, GatedDeltaNetSpec, Keys};
use nn::capability::{CausalLm, KvView, TokenBatch};
use nn::family::{Split, StateShape};
use nn::layers::{
    PagedAttention, ParallelLMHead, ResidualStream, RmsNorm, Spectrum, VocabParallelEmbedding,
};
use nn::{DType, Forward, TensorView};

use crate::config::{HybridConfig, Mixer};
use crate::decoder::{Components, DecoderStack, Hidden, Mlp};
use crate::{Error, Result};

/// The components a deployment's fence probes name (`profiling_config.fence_components`): the
/// decoder stack, one decoder layer, and a layer's two sublayers.
const TRANSFORMER: &str = "models.qwen3.hybrid.Transformer";
pub(crate) const COMPONENTS: Components = Components {
    layer: "models.qwen3.hybrid.TransformerLayer",
    attention: Some("models.qwen3.hybrid.Mixer"),
    mlp: Some("models.qwen3.hybrid.MoE"),
};

/// The spectrum a text request rotates by: the default spectrum over the rotated channels of the
/// configuration's interleaved M-RoPE (module documentation).
///
/// # Errors
///
/// A spectrum other than a sectioned one, which no Qwen3.5 configuration states.
pub fn text_spectrum(rope: Spectrum) -> Result<Spectrum> {
    match rope {
        Spectrum::Sectioned {
            theta,
            dim,
            precision,
            ..
        } => Ok(Spectrum::Default {
            theta,
            dim,
            precision,
        }),
        other => Err(Error::Config(format!(
            "a hybrid stack rotates by interleaved M-RoPE, not {other:?}"
        ))),
    }
}

/// The geometry of the stack's gated delta networks, before any split.
pub fn delta_net_spec(config: &HybridConfig) -> GatedDeltaNetSpec {
    let linear = &config.linear;
    GatedDeltaNetSpec {
        hidden: config.text.hidden_size,
        key_heads: linear.num_key_heads,
        value_heads: linear.num_value_heads,
        key_dim: linear.key_head_dim,
        value_dim: linear.value_head_dim,
        conv_width: linear.conv_kernel_dim,
        eps: config.text.rms_norm_eps,
    }
}

/// The Qwen3.5-MoE causal language model.
pub struct Qwen35Moe {
    config: HybridConfig,
    /// The tensor-parallel split every layer is sharded by.
    tensor: Split,
    dtype: DType,
    embedding: VocabParallelEmbedding,
    stack: DecoderStack,
    norm: RmsNorm,
    head: ParallelLMHead,
    /// Each layer's index into the key/value cache (an attention layer) or the recurrent state (a
    /// linear one), in stack order.
    ordinals: Vec<usize>,
    /// Whether the checkpoint's experts were read as calibrated NVFP4.
    experts_encoded: bool,
}

impl Qwen35Moe {
    /// The checkpoint configuration this model was built for.
    pub const fn config(&self) -> &HybridConfig {
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

    /// The recurrent state this rank's linear layers keep per request slot.
    pub fn state_shape(&self) -> StateShape {
        let linear = self.config.layers_of(Mixer::Linear).count() as u32;
        StateShape {
            layers: linear,
            fields: GatedDeltaNet::state_fields(&delta_net_spec(&self.config), self.dtype),
        }
    }

    /// Run every decoder layer over `hidden`, answering the last layer's update together with the
    /// residual stream it has not yet been added into; the caller's output norm performs that add.
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

        let mixers = &self.config.mixers;
        self.stack.forward(
            f,
            Hidden {
                hidden,
                residual: None,
            },
            cosine,
            sine,
            |index| match mixers[index] {
                Mixer::Attention => Keys::Paged {
                    batch,
                    cache,
                    group: 0,
                    layer: self.ordinals[index],
                },
                Mixer::Linear => Keys::Recurrent {
                    batch,
                    cache,
                    layer: self.ordinals[index],
                },
            },
            &buffers,
            |_, state| Ok(state),
        )
    }
}

impl CausalLm for Qwen35Moe {
    fn join_expert_step(&self, f: &Forward) -> nn::Result<()> {
        for layer in self.stack.layers() {
            if let Mlp::Experts { routed, .. } = &layer.mlp {
                // A layer's scratch is dead once its launches are enqueued, as in a forward.
                let mark = f.arena().mark();
                routed.join_expert_step(f)?;
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
            // The output norm adds the last update into the stream and normalizes it; the alias is
            // safe because `add_rms_norm` loads a whole row before storing any of it.
            self.norm.forward(
                f,
                last.hidden,
                Some(ResidualStream::in_place(residual)),
                hidden,
            )
        })
    }

    fn compute_logits(&self, f: &Forward, hidden: TensorView, out: TensorView) -> nn::Result<()> {
        self.head.forward(f, hidden, out)
    }
}

impl std::fmt::Debug for Qwen35Moe {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen35Moe")
            .field("layers", &self.stack.layers().len())
            .field("hidden", &self.config.text.hidden_size)
            .field("experts", &self.config.num_experts)
            .field("tensor", &self.tensor)
            .finish()
    }
}
