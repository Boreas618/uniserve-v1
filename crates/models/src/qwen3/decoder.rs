//! The Qwen3 decoder layer, which every Qwen3 model composes.
//!
//! The layer is Hugging Face's `Qwen3DecoderLayer`, with the feed-forward block of
//! `Qwen3MoeDecoderLayer` and the token mixer and shared expert of `Qwen3_5MoeDecoderLayer`,
//! in vLLM's residual form (`Qwen3NextDecoderLayer.forward(hidden_states, residual,
//! positions)`, `refs/vllm/vllm/model_executor/models/qwen3_next.py:583-655`):
//!
//! ```text
//! residual None:     residual = hidden;            x = input_layernorm(hidden)
//! residual Some(r):  x, residual = input_layernorm(hidden, r)       add, then normalize
//! mixer:             hidden = self_attn(x)          q/k RMSNorm fused into the rotation
//!                 or hidden = linear_attn(x)        a gated delta network over the slot's state
//!                    x, residual = post_attention_layernorm(hidden, residual)
//! feed-forward:      hidden = mlp(x)                gated MLP, or routed experts and a shared one
//! ```
//!
//! A layer returns `(hidden, residual)`: its feed-forward output, not yet added to the residual
//! stream, and the stream. The next layer's input norm, or the model's final norm, adds the
//! two in the operator that normalizes their sum.
//!
//! A layer holds no rotation: a model turns its queries and keys by the factors of its
//! positions, computed once per forward, and may hold more than one spectrum (a text batch and
//! an image batch of a Qwen3.5 model turn by different spectra of the same rotary width).

use nn::family::TextPlacement;
use nn::functional::Rounding;
use nn::modules::{
    Attention, AttentionConfig, GateUp, GatedDeltaNet, GatedDeltaNetConfig, Mlp, MlpConfig, MoE,
    MoEConfig, QkNorm, QkvBias, QkvLinearConfig, QkvNames, RmsNorm, RmsNormConfig, SharedExpert,
    SharedExpertConfig,
};
use nn::{ExpertExchange, KvCache, RecurrentState, Result, Split, Tensor, TokenBatch, VarBuilder};

use crate::qwen3::ID;
use crate::qwen3::config::{Config, TextConfig};

/// The profiler class of one decoder layer, at path `<stack>.layers.<i>`, which a deployment's
/// range patterns select (`docs/profiling.md`).
pub const LAYER: &str = "models.qwen3.transformer.TransformerLayer";

/// The profiler class of a layer's token mixer, at path `<layer>.self_attn` or
/// `<layer>.linear_attn`.
pub const ATTENTION: &str = "models.qwen3.transformer.Attention";

/// The profiler class of a layer's feed-forward block, at path `<layer>.mlp`.
pub const MLP: &str = "models.qwen3.transformer.MoE";

/// The gated MLP of `hidden → intermediate → hidden` gated by `text`'s activation, named as a
/// Qwen3 checkpoint names it, at `split`'s rank.
///
/// The product `act(gate) · up` is one FP32 expression rounded once, as the references' fused
/// `SiluAndMul` evaluates it.
fn gated_mlp(text: &TextConfig, intermediate: u32, split: Split) -> MlpConfig {
    MlpConfig {
        hidden: text.hidden_size,
        intermediate,
        activation: text.hidden_act,
        rounding: Rounding::Fused,
        gate_up: GateUp::Split {
            gate: "gate_proj",
            up: "up_proj",
        },
        down: "down_proj",
        bias: false,
        split,
    }
}

/// The RMS normalization of the residual stream of `text`, at its epsilon and gain.
pub(crate) fn stream_norm(text: &TextConfig) -> RmsNormConfig {
    RmsNormConfig {
        width: text.hidden_size,
        eps: text.rms_norm_eps,
        centred: text.rms_norm_centred,
        output: None,
    }
}

/// One decoder layer, its modules named as the checkpoint names them.
///
/// A layer mixes its tokens with `self_attn` or with `linear_attn`, and feeds them forward
/// through `mlp` or through `experts` (with `shared_expert` beside them when the checkpoint
/// has one): exactly one of each pair is loaded, as the references construct one of each
/// (`qwen3_next.py:525-560`).
pub struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Option<Attention>,
    linear_attn: Option<GatedDeltaNet>,
    post_attention_layernorm: RmsNorm,
    mlp: Option<Mlp>,
    experts: Option<MoE>,
    shared_expert: Option<SharedExpert>,
    /// The layer's module path, which names its profiler ranges.
    path: String,
}

impl DecoderLayer {
    /// The layer the module at `vb` (`<stack>.layers.<i>`) states, at the ranks `placement`
    /// states: its rank of the tensor-parallel group, its attention exchanging its rows over the
    /// Ulysses sequence split ([`nn::family::SequenceSplit::WHOLE`] for none).
    ///
    /// `linear` loads a gated delta network as the mixer in place of attention. `routed` loads
    /// routed experts (and the shared expert the configuration states) in place of the dense
    /// MLP, held at the placement's share of them (its window of the expert group's slots, or
    /// of the tensor group's when that group holds whole experts) in the slots `order` states,
    /// or in the checkpoint's order when it states none.
    ///
    /// # Errors
    ///
    /// A linear layer or a routed layer of a configuration that states neither network, a
    /// module the loader refuses, or a split that does not divide what it shards.
    pub fn load(
        config: &Config,
        vb: VarBuilder<'_, '_>,
        linear: bool,
        routed: bool,
        placement: &TextPlacement,
        order: Option<&Vec<u32>>,
    ) -> Result<Self> {
        let (tensor, sequence) = (placement.tensor, placement.sequence);
        let text = &config.text;
        let norm = stream_norm(text);
        let refuse = |missing: &str| ID.config(format!("{} states no {missing}", vb.path()));

        let (self_attn, linear_attn) = if linear {
            let network = text
                .gated_delta_net
                .ok_or_else(|| refuse("gated delta network"))?;
            let network = GatedDeltaNetConfig {
                split: tensor,
                ..network
            };

            (
                None,
                Some(GatedDeltaNet::load(&network, vb.pp("linear_attn"))?),
            )
        } else {
            let attention = AttentionConfig {
                qkv: QkvLinearConfig {
                    hidden: text.hidden_size,
                    heads: text.num_attention_heads,
                    kv_heads: text.num_key_value_heads,
                    head_dim: text.head_dim,
                    gated: text.attn_output_gate,
                    names: QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
                    bias: if text.attention_bias {
                        QkvBias::Present
                    } else {
                        QkvBias::Absent
                    },
                    split: tensor,
                },
                output: "o_proj",
                output_bias: text.attention_bias,
                qk_norm: QkNorm::Fused {
                    gains: Some(["q_norm", "k_norm"]),
                    eps: text.rms_norm_eps,
                    centred: text.rms_norm_centred,
                },
                scale: text.rope.score_scale(text.head_dim)?,
                causal: true,
                window: None,
                max_kv_len: text.max_position_embeddings,
                sequence,
            };

            (Some(Attention::load(&attention, vb.pp("self_attn"))?), None)
        };

        let (mlp, routed_experts, shared_expert) = if routed {
            let moe = config
                .moe
                .as_ref()
                .ok_or_else(|| refuse("routed experts"))?;
            let block = MoEConfig {
                hidden: text.hidden_size,
                intermediate: moe.moe_intermediate_size,
                experts: moe.num_experts,
                top: moe.num_experts_per_tok,
                activation: text.hidden_act,
                renormalize: moe.norm_topk_prob,
                router: "gate",
                list: "experts",
                up: "up_proj",
                gate: "gate_proj",
                down: "down_proj",
                // Released Qwen3.5 BF16 checkpoints stack the experts; Qwen3-MoE and every
                // calibrated export store one module per expert.
                stacked: Some(["gate_up_proj", "down_proj"]),
                order: order
                    .cloned()
                    .unwrap_or_else(|| (0..moe.num_experts).collect()),
                split: tensor,
                tensor_experts: placement.tensor_experts,
                exchange: placement.experts,
            };
            let shared = match moe.shared_expert_intermediate_size {
                Some(intermediate) => Some(SharedExpert::load(
                    &SharedExpertConfig {
                        mlp: gated_mlp(text, intermediate, tensor),
                        module: "shared_expert",
                        gate: "shared_expert_gate",
                    },
                    vb.pp("mlp"),
                )?),
                None => None,
            };

            (None, Some(MoE::load(&block, vb.pp("mlp"))?), shared)
        } else {
            let dense = gated_mlp(text, text.intermediate_size, tensor);
            (Some(Mlp::load(&dense, vb.pp("mlp"))?), None, None)
        };

        Ok(Self {
            input_layernorm: RmsNorm::load(&norm, vb.pp("input_layernorm"))?,
            self_attn,
            linear_attn,
            post_attention_layernorm: RmsNorm::load(&norm, vb.pp("post_attention_layernorm"))?,
            mlp,
            experts: routed_experts,
            shared_expert,
            path: vb.path().to_owned(),
        })
    }

    /// Whether the layer's mixer is a gated delta network, which reads the recurrent state
    /// and not the key/value cache.
    pub const fn is_linear(&self) -> bool {
        self.linear_attn.is_some()
    }

    /// The layer's routed experts, when it routes.
    pub const fn experts(&self) -> Option<&MoE> {
        self.experts.as_ref()
    }

    /// The layer's module path in the checkpoint.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The layer over the rows of `batch`: `(hidden, residual)`, the feed-forward output and
    /// the residual stream it is not yet added to.
    ///
    /// `hidden` is `[tokens, hidden]`, with the stream it is not yet added to (`None` for a
    /// stack's first layer, whose `hidden` becomes the stream). `rope` is the forward's
    /// `(cos, sin)` factor tables. An attention layer writes its keys and values into layer
    /// `ordinal` of `cache` and attends them; a linear layer reads and advances recurrent
    /// layer `ordinal` of `state`. `exchange` is the expert group's exchange, lent when the
    /// layer's experts are split over one.
    ///
    /// # Errors
    ///
    /// A linear layer run without a state, or tensors a module refuses.
    #[expect(
        clippy::too_many_arguments,
        reason = "the tensors a decoder layer reads and the state execution lends it"
    )]
    pub fn forward(
        &self,
        hidden: &Tensor,
        residual: Option<&Tensor>,
        rope: (&Tensor, &Tensor),
        batch: &TokenBatch,
        cache: Option<&KvCache>,
        state: Option<&RecurrentState>,
        exchange: Option<&ExpertExchange>,
        ordinal: usize,
    ) -> Result<(Tensor, Tensor)> {
        nn::profiler::range(LAYER, &self.path, || {
            let (normalized, residual) = self.normalize(hidden, residual)?;

            let mixed = match (&self.self_attn, &self.linear_attn) {
                (Some(attention), _) => {
                    let cache = cache.ok_or_else(|| self.lent_no("key/value cache"))?;

                    nn::profiler::range(ATTENTION, &format!("{}.self_attn", self.path), || {
                        attention.forward_paged(&normalized, Some(rope), batch, cache, ordinal)
                    })?
                }
                (None, Some(network)) => {
                    let state = state.ok_or_else(|| self.lent_no("recurrent state"))?;

                    nn::profiler::range(ATTENTION, &format!("{}.linear_attn", self.path), || {
                        network.forward(&normalized, batch, state, ordinal)
                    })?
                }
                (None, None) => return Err(self.lent_no("token mixer")),
            };

            self.feed_forward(&mixed, &residual, batch.live_tokens.as_ref(), exchange)
        })
    }

    /// The layer over rows that attend as the variable-length sequences `offsets` names, with
    /// no cache: `(hidden, residual)` as [`Self::forward`] returns them.
    ///
    /// `offsets` is the `[sequences + 1]` U32 cumulative row counts of the sequences, which
    /// attend themselves under the causal mask, and `max_kv_len` the bound on one sequence's
    /// keys.
    ///
    /// # Errors
    ///
    /// A linear layer, whose state only a batch's slots address, or tensors a module refuses.
    pub fn forward_varlen(
        &self,
        hidden: &Tensor,
        residual: Option<&Tensor>,
        rope: (&Tensor, &Tensor),
        offsets: &Tensor,
        max_kv_len: u32,
    ) -> Result<(Tensor, Tensor)> {
        nn::profiler::range(LAYER, &self.path, || {
            let (normalized, residual) = self.normalize(hidden, residual)?;

            let attention = self
                .self_attn
                .as_ref()
                .ok_or_else(|| self.lent_no("attention over its own rows"))?;
            let mixed =
                nn::profiler::range(ATTENTION, &format!("{}.self_attn", self.path), || {
                    attention.forward_varlen(
                        &normalized,
                        Some(rope),
                        offsets,
                        offsets,
                        None,
                        max_kv_len,
                    )
                })?;

            self.feed_forward(&mixed, &residual, None, None)
        })
    }

    /// The refusal of a forward that needs `what` and has none.
    fn lent_no(&self, what: &str) -> nn::Error {
        ID.config(format!("{} has no {what}", self.path))
    }

    /// The input norm: `(normalized, residual)`, the rows the mixer reads and the stream,
    /// which `hidden` is added into when there is one and is itself otherwise.
    fn normalize(&self, hidden: &Tensor, residual: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        match residual {
            None => Ok((self.input_layernorm.forward(hidden)?, hidden.clone())),
            Some(residual) => Ok(self.input_layernorm.forward_residual(hidden, residual)?),
        }
    }

    /// The second half of the layer: `mixed`, the mixer's output, is added into the stream,
    /// and the feed-forward block runs over the normalized sum.
    fn feed_forward(
        &self,
        mixed: &Tensor,
        residual: &Tensor,
        live_tokens: Option<&Tensor>,
        exchange: Option<&ExpertExchange>,
    ) -> Result<(Tensor, Tensor)> {
        let (normalized, residual) = self
            .post_attention_layernorm
            .forward_residual(mixed, residual)?;

        let hidden = nn::profiler::range(MLP, &format!("{}.mlp", self.path), || {
            match (&self.mlp, &self.experts) {
                (Some(mlp), _) => mlp.forward(&normalized),
                (None, Some(experts)) => {
                    let routed = experts.forward(&normalized, live_tokens, exchange)?;

                    match &self.shared_expert {
                        Some(shared) => shared.forward(&normalized, routed),
                        None => Ok(routed),
                    }
                }
                (None, None) => Err(nn::Error::shape(
                    "DecoderLayer",
                    format!("{} has no feed-forward block", self.path),
                )),
            }
        })?;

        Ok((hidden, residual))
    }
}
