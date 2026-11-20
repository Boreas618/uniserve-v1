//! A hyper-connected decoder layer (`inference/model.py:909-996`): an n-gram memory on the
//! layers that hold one, then the attention and the mixture of experts, each between two
//! manifold-constrained connections of the residual streams.
//!
//! ```text
//! h      ← engram(h)                         layers 1 and 14: every stream plus its gated recall
//! h, a   ← connection_attn(h; attention)     the input read with the carried pre-mix, attn_norm
//! h, f   ← connection_ffn(h; experts)        the input read with the attention's pre-mix, ffn_norm
//! experts(x) = routed(x) + shared(x)         384 MXFP4 experts, top 6, beside one MXFP8 expert
//! ```
//!
//! The connections are [`MhcResidual`]'s: each sub-layer's output stays pending until the next
//! boundary writes it back, and the memory reads the streams with every output written.

use nn::collective;
use nn::family::{EXPERT_AXIS, ExpertOrder, TextPlacement};
use nn::functional::{Activation, Rounding, Scoring};
use nn::modules::{
    ExpertLayout, GateUp, MhcConnection, MhcResidual, MlpConfig, MoE, MoEConfig, NgramMemory,
    SharedExpert, SharedExpertConfig, Tokens,
};
use nn::{DType, ExpertExchange, Result, Split, Tensor, VarBuilder};

use crate::deepseek_v41::attention::{Attention, Carried, Inputs};
use crate::deepseek_v41::config::{Config, ExpertConfig};

/// The module path an expert order and the expert load name layer `layer`'s routed experts by.
pub fn expert_module(layer: u32) -> String {
    format!("layers.{layer}.ffn.experts")
}

/// The routed experts `experts` of a layer whose expert load and order name them `name`, in
/// the slots `order` states (the checkpoint's own where it states none), split as `splits`
/// states: the tensor split, whether a tensor-parallel group holds the experts whole, and the
/// expert union of ranks that share them (`model.py:794-906`).
pub(crate) fn routed(
    config: &Config,
    experts: &ExpertConfig,
    name: String,
    order: Option<Vec<u32>>,
    (split, tensor_experts, exchange): (Split, bool, Split),
) -> MoEConfig {
    MoEConfig {
        hidden: config.hidden_size,
        intermediate: experts.intermediate,
        experts: experts.experts,
        top: experts.top,
        activation: Activation::Silu,
        renormalize: true,
        scoring: Scoring::SqrtSoftplus,
        selection_bias: Some("bias"),
        routed_scale: Some(experts.routed_scale),
        limit: Some(experts.limit),
        router: "gate",
        // The reference scores in FP32: `linear(x.float(), weight.float())`.
        router_dtype: Some(DType::F32),
        list: "experts",
        up: "w3",
        gate: "w1",
        down: "w2",
        layout: ExpertLayout::Modules,
        order: order.unwrap_or_else(|| (0..experts.experts).collect()),
        split,
        tensor_experts,
        exchange,
        name: Some(name),
    }
}

/// The shared expert beside the routed experts `experts`, of their width (`model.py:832-854`).
pub(crate) fn shared(config: &Config, experts: &ExpertConfig) -> SharedExpertConfig {
    SharedExpertConfig {
        mlp: MlpConfig {
            hidden: config.hidden_size,
            intermediate: experts.intermediate,
            activation: Activation::Silu,
            // The reference's `Expert`: the clamped SwiGLU in FP32, rounded once.
            rounding: Rounding::Fused,
            limit: Some(experts.limit),
            gate_up: GateUp::Split {
                gate: "w1",
                up: "w3",
            },
            down: "w2",
            bias: false,
            split: Split::WHOLE,
        },
        module: "shared_experts",
        gate: None,
    }
}

/// One decoder layer (module documentation).
pub struct DecoderLayer {
    engram: Option<NgramMemory>,
    attention_connection: MhcConnection,
    attention: Attention,
    ffn_connection: MhcConnection,
    experts: MoE,
    shared: SharedExpert,
}

impl DecoderLayer {
    /// Layer `layer` at `vb` (`layers.<layer>`), its n-gram memory `engram` already built, at
    /// the ranks `placement` states, its routed experts in the slots `order` states.
    ///
    /// # Errors
    ///
    /// A weight the checkpoint lacks or holds otherwise, or a split the experts refuse.
    pub fn load(
        config: &Config,
        layer: u32,
        vb: VarBuilder<'_, '_>,
        engram: Option<NgramMemory>,
        placement: &TextPlacement,
        order: &ExpertOrder,
    ) -> Result<Self> {
        let mhc = config.mhc();
        let name = expert_module(layer);
        let routed = routed(
            config,
            &config.experts,
            name.clone(),
            order.get(&name).cloned(),
            (
                placement.tensor,
                placement.tensor_experts,
                placement.experts,
            ),
        );
        let shared = shared(config, &config.experts);

        Ok(Self {
            engram,
            attention_connection: MhcConnection::load(&mhc, &vb, "hc_attn", vb.pp("attn_norm"))?,
            attention: Attention::load(config, layer, vb.pp("attn"))?,
            ffn_connection: MhcConnection::load(&mhc, &vb, "hc_ffn", vb.pp("ffn_norm"))?,
            experts: MoE::load(&routed, vb.pp("ffn"))?,
            shared: SharedExpert::load(&shared, vb.pp("ffn"))?,
        })
    }

    /// The layer's routed experts.
    pub fn experts(&self) -> &MoE {
        &self.experts
    }

    /// The layer's n-gram memory, when it holds one.
    pub fn engram(&self) -> Option<&NgramMemory> {
        self.engram.as_ref()
    }

    /// The layer over `residual`: the n-gram memory over the streams written back, then the
    /// attention and the experts as sub-layers of the residual.
    ///
    /// `context` is the batch's n-gram context (`[tokens, N]`), read by a layer with a memory, and
    /// `tokens` whose tokens its table's other ranks look up; `live_tokens` and `exchange` are
    /// the routed experts' ([`MoE::forward`]).
    ///
    /// # Errors
    ///
    /// What a sub-layer or a connection refuses.
    pub fn forward(
        &self,
        residual: &mut MhcResidual,
        (context, tokens): (Option<&Tensor>, Tokens),
        inputs: &Inputs<'_>,
        carried: &mut Carried,
        live_tokens: Option<&Tensor>,
        exchange: Option<&ExpertExchange>,
    ) -> Result<()> {
        if let (Some(memory), Some(context)) = (&self.engram, context) {
            let streams = residual.streams()?;
            // Ranks that serve their own tokens complete each other's lookups along their union.
            let recalled = match tokens {
                Tokens::Own { .. } => {
                    collective::along(EXPERT_AXIS, || memory.forward(streams, context, tokens))?
                }
                Tokens::Shared => memory.forward(streams, context, tokens)?,
            };
            residual.set_streams(recalled)?;
        }
        residual.sublayer(&self.attention_connection, |x| {
            self.attention.forward(x, inputs, carried)
        })?;
        residual.sublayer(&self.ffn_connection, |x| {
            let routed = self.experts.forward(x, live_tokens, exchange)?;
            self.shared.forward(x, routed)
        })
    }
}
