//! The decoder layer over hyper-connected streams (vLLM's `Qwen4ExpDecoderLayer`,
//! `refs/vllm/vllm/models/qwen4_exp/nvidia/model.py:295-368`).
//!
//! ```text
//! PLE (the layers that hold one):  S ← S + ple(S, ids)           after any pending combine
//! mixer stage:   S, x, r ← attn_hyper_connection(S, pending)     combine, norm, mix
//!                y ← linear_attn(x)  or  self_attn(x)
//! expert stage:  S, x, r ← mlp_hyper_connection(S, (y, r))
//!                y ← Σ_routes weight · expert(x) + sigmoid(g · x) · shared_expert(x)
//! ```
//!
//! A layer has no norm of its own: each stage's hyper-connection normalizes the streams it
//! reads. A layer returns `(S, y, r)`: the streams its expert stage read, the experts' output
//! and its combine logits, which the next layer's first read (or the model's read-out) folds
//! into the streams. That boundary is the one vLLM's kernel chain and its fused layer kernels
//! have, so a layer is the unit either computes.

use nn::family::TextPlacement;
use nn::functional::Rounding;
use nn::modules::{
    GateUp, GatedDeltaNet, GatedDeltaNetConfig, MlpConfig, MoE, MoEConfig, Rotary, SharedExpert,
    SharedExpertConfig,
};
use nn::{DType, ExpertExchange, KvCache, RecurrentState, Result, Tensor, TokenBatch, VarBuilder};

use crate::qwen4_exp::ID;
use crate::qwen4_exp::attention::IndexedAttention;
use crate::qwen4_exp::config::Config;
use crate::qwen4_exp::hyper_connection::HyperConnection;
use crate::qwen4_exp::ngram::PleLayer;

/// The profiler class of one decoder layer, at path `<stack>.layers.<i>`, which a deployment's
/// range patterns select (`docs/profiling.md`).
pub const LAYER: &str = "models.qwen4_exp.transformer.TransformerLayer";

/// The profiler class of a layer's token mixer, at path `<layer>.self_attn` or
/// `<layer>.linear_attn`.
pub const ATTENTION: &str = "models.qwen4_exp.transformer.Attention";

/// The profiler class of a layer's experts, at path `<layer>.mlp`.
pub const MLP: &str = "models.qwen4_exp.transformer.MoE";

/// One decoder layer, its modules named as the checkpoint names them.
///
/// A layer mixes its tokens with `linear_attn` or with `self_attn`: exactly one is loaded.
pub struct DecoderLayer {
    ple: Option<PleLayer>,
    attn_hyper_connection: HyperConnection,
    linear_attn: Option<GatedDeltaNet>,
    self_attn: Option<IndexedAttention>,
    mlp_hyper_connection: HyperConnection,
    experts: MoE,
    shared_expert: SharedExpert,
    /// The residual streams, `C`.
    streams: usize,
    /// The layer's module path, which names its profiler ranges.
    path: String,
}

impl DecoderLayer {
    /// The layer the module at `vb` (`<stack>.layers.<i>`) states, at the ranks `placement`
    /// states: its rank of the tensor-parallel group, for a deployment that drafts the
    /// placement's draft tokens a step.
    ///
    /// `linear` loads a gated delta network as the mixer in place of indexed attention, and
    /// `ple` the PLE layer the layer holds. The routed experts are held at the placement's share
    /// of them (its window of the expert group's slots, or of the tensor group's when that group
    /// holds whole experts) in the slots `order` states, or in the checkpoint's order when it
    /// states none.
    ///
    /// # Errors
    ///
    /// A module the loader refuses, or a split that does not divide what it shards.
    pub fn load(
        config: &Config,
        vb: VarBuilder<'_, '_>,
        linear: bool,
        ple: bool,
        placement: &TextPlacement,
        order: Option<&Vec<u32>>,
    ) -> Result<Self> {
        let (tensor, draft_tokens) = (placement.tensor, placement.draft_tokens);
        let (text, moe) = (&config.text, &config.moe);

        let (linear_attn, self_attn) = if linear {
            let network = GatedDeltaNetConfig {
                split: tensor,
                draft_tokens,
                ..text.gated_delta_net
            };

            (
                Some(GatedDeltaNet::load(&network, vb.pp("linear_attn"))?),
                None,
            )
        } else {
            (
                None,
                Some(IndexedAttention::load(config, vb.pp("self_attn"), tensor)?),
            )
        };

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
            // The RadixArk export stacks the drafter's experts; every other module list holds
            // one module per expert.
            stacked: Some(["gate_up_proj", "down_proj"]),
            order: order
                .cloned()
                .unwrap_or_else(|| (0..moe.num_experts).collect()),
            split: tensor,
            tensor_experts: placement.tensor_experts,
            exchange: placement.experts,
        };
        let shared = SharedExpertConfig {
            mlp: MlpConfig {
                hidden: text.hidden_size,
                intermediate: moe.shared_expert_intermediate_size.unwrap_or(0),
                activation: text.hidden_act,
                // `act(gate) · up` is one FP32 expression rounded once, as vLLM's
                // `act_and_mul` evaluates it.
                rounding: Rounding::Fused,
                gate_up: GateUp::Split {
                    gate: "gate_proj",
                    up: "up_proj",
                },
                down: "down_proj",
                bias: false,
                split: tensor,
            },
            module: "shared_expert",
            gate: "shared_expert_gate",
        };

        Ok(Self {
            ple: if ple {
                Some(PleLayer::load(config, vb.pp("ple"), tensor)?)
            } else {
                None
            },
            attn_hyper_connection: HyperConnection::load(
                config,
                vb.pp("attn_hyper_connection"),
                true,
            )?,
            linear_attn,
            self_attn,
            mlp_hyper_connection: HyperConnection::load(
                config,
                vb.pp("mlp_hyper_connection"),
                true,
            )?,
            experts: MoE::load(&block, vb.pp("mlp"))?,
            shared_expert: SharedExpert::load(&shared, vb.pp("mlp"))?,
            streams: config.hyper_connection.streams as usize,
            path: vb.path().to_owned(),
        })
    }

    /// Whether the layer's mixer is a gated delta network, which reads the recurrent state
    /// and not the key/value cache.
    pub const fn is_linear(&self) -> bool {
        self.linear_attn.is_some()
    }

    /// Whether the layer holds a PLE layer.
    pub const fn has_ple(&self) -> bool {
        self.ple.is_some()
    }

    /// The layer's routed experts.
    pub const fn experts(&self) -> &MoE {
        &self.experts
    }

    /// The layer's module path in the checkpoint.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The layer over the rows of `batch`: `(streams, output, logits)`, the streams its expert
    /// stage read, `[tokens, C · H]`, the experts' output, `[tokens, H]`, not yet folded into
    /// them, and the combine logits it is folded by, `[tokens, C]`.
    ///
    /// `streams` are the streams the block before read, with that block's output and logits
    /// `pending` on them (`None` for a stack's first layer). `rope` is the forward's
    /// `(cos, sin)` factor tables of `coordinates` under `rotary`, `columns` its per-token
    /// columns ([`crate::qwen4_exp::causal_lm::token_columns`]) and `context` each token's n-gram
    /// context, which a layer with a PLE layer reads. An attention layer writes layer
    /// `ordinal` of `cache`. `state_layers` names the layer's recurrent layers of `state`, as
    /// the model orders them: its mixer's (a gated delta network's fields or an attention's
    /// ring) and its PLE layer's window. `exchange` is the expert group's exchange, lent when
    /// the experts are split over one.
    ///
    /// # Errors
    ///
    /// A PLE layer run without a context, an attention layer without a cache, or tensors a
    /// module refuses.
    #[expect(
        clippy::too_many_arguments,
        reason = "the tensors a decoder layer reads and the memory execution lends it"
    )]
    pub fn forward(
        &self,
        streams: &Tensor,
        pending: Option<(&Tensor, &Tensor)>,
        rope: (&Tensor, &Tensor),
        rotary: &Rotary,
        coordinates: &Tensor,
        columns: &[Tensor],
        context: Option<&Tensor>,
        batch: &TokenBatch,
        cache: Option<&KvCache>,
        state: &RecurrentState,
        exchange: Option<&ExpertExchange>,
        ordinal: usize,
        (mixer_state, ple_state): (usize, usize),
    ) -> Result<(Tensor, Tensor, Tensor)> {
        nn::profiler::range(LAYER, &self.path, || {
            // A PLE layer reads the streams themselves, so what is pending on them is folded
            // first (`model.py:309-316`).
            let (streams, pending) = match &self.ple {
                None => (streams.clone(), pending),
                Some(ple) => {
                    let folded = match pending {
                        Some((output, logits)) => {
                            HyperConnection::combine(streams, output, Some(logits), self.streams)?
                        }
                        None => streams.clone(),
                    };
                    let context = context.ok_or_else(|| self.lent_no("n-gram context"))?;
                    let added = ple.forward(&folded, context, batch, state, ple_state)?;

                    let sum = folded
                        .to_dtype(DType::F32)?
                        .add(&added.to_dtype(DType::F32)?)?
                        .to_dtype(folded.dtype())?;
                    (sum, None)
                }
            };

            let (streams, input, logits) = self.attn_hyper_connection.forward(&streams, pending)?;
            let logits = logits.ok_or_else(|| self.lent_no("combine logits"))?;

            let mixed = match (&self.linear_attn, &self.self_attn) {
                (Some(network), _) => {
                    nn::profiler::range(ATTENTION, &format!("{}.linear_attn", self.path), || {
                        network.forward(&input, batch, state, mixer_state)
                    })?
                }
                (None, Some(attention)) => {
                    let cache = cache.ok_or_else(|| self.lent_no("key/value cache"))?;

                    nn::profiler::range(ATTENTION, &format!("{}.self_attn", self.path), || {
                        attention.forward(
                            &input,
                            rope,
                            rotary,
                            coordinates,
                            columns,
                            batch,
                            cache,
                            ordinal,
                            state,
                            mixer_state,
                        )
                    })?
                }
                (None, None) => return Err(self.lent_no("token mixer")),
            };

            let (streams, input, logits) = self
                .mlp_hyper_connection
                .forward(&streams, Some((&mixed, &logits)))?;
            let logits = logits.ok_or_else(|| self.lent_no("combine logits"))?;

            let output = nn::profiler::range(MLP, &format!("{}.mlp", self.path), || {
                let routed = self
                    .experts
                    .forward(&input, batch.live_tokens.as_ref(), exchange)?;

                self.shared_expert.forward(&input, routed)
            })?;

            Ok((streams, output, logits))
        })
    }

    /// The refusal of a forward that needs `what` and has none.
    fn lent_no(&self, what: &str) -> nn::Error {
        ID.config(format!("{} has no {what}", self.path))
    }
}
