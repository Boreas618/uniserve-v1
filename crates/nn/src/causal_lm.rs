//! What causal language model families of the transformers layout share: how a checkpoint opens
//! and loads, how its routed experts are named and laid out, and the configuration of the
//! sparse mixture-of-experts block their decoder layers hold.
//!
//! A family implements [`CausalLmConfig`] on its configuration, the one value its loading
//! boundary normalizes a checkpoint into, and registers [`CausalLmFamily`] of it: the family
//! value implements [`TextFamily`] and the opened checkpoint [`TextCheckpoint`] once, here,
//! from what the configuration states. The mathematics stays the family's own: its module tree,
//! its forward and the model its [`CausalLmConfig::build`] returns.
//!
//! ## The routed experts
//!
//! A decoder layer that routes holds transformers' sparse MoE block under `mlp`: a `gate`
//! router over the module list `experts`, each expert a gated MLP of `gate_proj`, `up_proj` and
//! `down_proj`, beside an optional gated shared expert ([`MoeConfig`]). An expert
//! redistribution names a layer's experts by the module path [`expert_module`] states, and a
//! layer's slots follow the order the deployment states for it, the checkpoint's own where it
//! states none.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use proto::config::Quantization;

use crate::error::{ModelId, Result};
use crate::family::{ExpertOrder, TextCheckpoint, TextFamily, TextModel, TextPlacement};
use crate::functional::{Activation, Scoring};
use crate::modules::{ExpertLayout, MoE, MoEConfig};
use crate::quantization::Declaration;
use crate::{Checkpoint, ExpertExchange, VarBuilder, WeightLoader};

/// A causal language model checkpoint's configuration, as its family reads it once at the
/// loading boundary: what the shared scaffolding ([`CausalLmFamily`]) asks of a checkpoint.
///
/// The required methods state the checkpoint's facts and build the family's model; the provided
/// ones are the behaviour every such family shares, implemented once over those facts.
pub trait CausalLmConfig: Send + Sync + Sized + 'static {
    /// The family's id, which its refusals name ([`ModelId`]).
    const ID: ModelId;

    /// Reads and normalizes `checkpoint`'s configuration: its `config.json` and the
    /// quantization its export declares.
    ///
    /// # Errors
    ///
    /// When the configuration cannot be read or states a model the family cannot build.
    fn open(checkpoint: &Checkpoint) -> Result<Self>;

    /// What the checkpoint's export declares about its weights' representation, which every
    /// later weight is read as.
    fn quantization(&self) -> &Declaration;

    /// The decoder layers the checkpoint holds.
    fn layers(&self) -> u32;

    /// The routed experts of the decoder stack, `None` for a dense stack.
    fn experts(&self) -> Option<&MoeConfig>;

    /// Whether the checkpoint carries a drafter for speculative decoding (multi-token
    /// prediction layers) that a placement may load ([`TextPlacement::drafter`]). A family whose
    /// checkpoints carry none states none.
    fn drafter(&self) -> bool {
        false
    }

    /// Builds the family's model of this configuration, the configuration of `checkpoint`, its
    /// modules read under `vb` (the checkpoint's root), at the ranks `placement` states, its
    /// routed experts in the slots `order` states.
    ///
    /// # Errors
    ///
    /// When the placement does not divide the model or a weight cannot be loaded.
    fn build(
        &self,
        checkpoint: &Checkpoint,
        vb: VarBuilder<'_, '_>,
        placement: &TextPlacement,
        order: &ExpertOrder,
    ) -> Result<Arc<dyn TextModel>>;

    /// Whether decoder layer `layer` routes its tokens to experts.
    fn routes(&self, layer: u32) -> bool {
        self.experts().is_some_and(|moe| moe.routes(layer))
    }

    /// The routed expert layers by module path ([`expert_module`]), each with its expert count:
    /// what an expert redistribution may order. Empty for a dense stack.
    fn expert_layers(&self) -> BTreeMap<String, u32> {
        let Some(moe) = self.experts() else {
            return BTreeMap::new();
        };

        (0..self.layers())
            .filter(|layer| moe.routes(*layer))
            .map(|layer| (expert_module(layer), moe.num_experts))
            .collect()
    }

    /// Refuses an expert order that names a module no layer routes through.
    ///
    /// # Errors
    ///
    /// The family's refusal naming the module.
    fn check_order(&self, order: &ExpertOrder) -> Result<()> {
        let routed = self.expert_layers();

        match order.keys().find(|module| !routed.contains_key(*module)) {
            Some(stray) => Err(Self::ID.config(format!(
                "{stray} is not a routed expert layer of this model"
            ))),
            None => Ok(()),
        }
    }
}

/// The family value of the configuration `C`: how a checkpoint of it opens and loads.
pub struct CausalLmFamily<C> {
    config: PhantomData<fn() -> C>,
}

impl<C> CausalLmFamily<C> {
    /// The family of `C`.
    pub const fn new() -> Self {
        Self {
            config: PhantomData,
        }
    }
}

impl<C> Default for CausalLmFamily<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: CausalLmConfig> TextFamily for CausalLmFamily<C> {
    /// Opens the checkpoint's configuration. The family serves each checkpoint as its export
    /// declares it, BF16 or quantized, and refuses any other policy.
    fn open(
        &self,
        checkpoint: &Checkpoint,
        quantization: &Quantization,
    ) -> Result<Box<dyn TextCheckpoint>> {
        crate::quantization::own_representation(C::ID.0, quantization)?;

        Ok(Box::new(Opened(C::open(checkpoint)?)))
    }
}

/// An opened checkpoint of the configuration `C`.
struct Opened<C>(C);

impl<C: CausalLmConfig> TextCheckpoint for Opened<C> {
    fn expert_layers(&self) -> BTreeMap<String, u32> {
        self.0.expert_layers()
    }

    fn drafter(&self) -> bool {
        self.0.drafter()
    }

    fn load(
        &self,
        checkpoint: &Checkpoint,
        loader: &mut WeightLoader<'_>,
        placement: &TextPlacement,
        experts: &ExpertOrder,
    ) -> Result<Arc<dyn TextModel>> {
        // Every later weight is read as the export declares it stored.
        loader.declare(self.0.quantization())?;
        self.0.check_order(experts)?;

        self.0.build(checkpoint, loader.root(), placement, experts)
    }
}

/// The path an [`ExpertOrder`] names the routed experts of decoder layer `layer` by.
pub fn expert_module(layer: u32) -> String {
    format!("layers.{layer}.mlp.experts")
}

/// The routed experts of a decoder stack: the fields of a transformers MoE configuration its
/// sparse MoE blocks read, and how the checkpoint lays its experts out.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    /// Experts per routed layer (`num_experts`).
    pub num_experts: u32,
    /// Experts each token is routed to (`num_experts_per_tok`).
    pub num_experts_per_tok: u32,
    /// Width of one expert's intermediate projection (`moe_intermediate_size`).
    pub moe_intermediate_size: u32,
    /// Whether the selected experts' weights renormalize to one (`norm_topk_prob`).
    pub norm_topk_prob: bool,
    /// Every `decoder_sparse_step`-th layer routes when `mlp_only_layers` does not name it.
    pub decoder_sparse_step: u32,
    /// Layers that keep the dense MLP (`mlp_only_layers`).
    pub mlp_only_layers: Vec<u32>,
    /// Width of the gated shared expert every token also passes through, when the block has
    /// one (`shared_expert_intermediate_size`).
    pub shared_expert_intermediate_size: Option<u32>,
    /// How the checkpoint's architecture stores the dense experts ([`ExpertLayout`]).
    pub layout: ExpertLayout,
}

impl MoeConfig {
    /// Whether `layer` routes its tokens to experts.
    pub fn routes(&self, layer: u32) -> bool {
        !self.mlp_only_layers.contains(&layer)
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
    }

    /// The sparse MoE block of a layer of `hidden`-wide rows whose experts gate by
    /// `activation`, at the ranks `placement` states: a `gate` router over the list `experts`,
    /// each expert's `gate_proj`, `up_proj` and `down_proj`, laid out as the architecture
    /// declares ([`Self::layout`]).
    ///
    /// `stack` is the block's decoder layer in the model's own stack and the deployment's
    /// expert order: the block is named by the layer's module ([`expert_module`]), its experts
    /// held in the slots the order states for it (the checkpoint's own order where it states
    /// none). A block outside the stack, such as a drafter's, takes `None`: it is named by
    /// nothing and holds its experts in the checkpoint's order.
    pub fn block(
        &self,
        hidden: u32,
        activation: Activation,
        placement: &TextPlacement,
        stack: Option<(u32, &ExpertOrder)>,
    ) -> MoEConfig {
        let name = stack.map(|(layer, _)| expert_module(layer));
        let order = stack.and_then(|(_, order)| order.get(name.as_deref()?));

        MoEConfig {
            hidden,
            intermediate: self.moe_intermediate_size,
            experts: self.num_experts,
            top: self.num_experts_per_tok,
            activation,
            renormalize: self.norm_topk_prob,
            scoring: Scoring::Softmax,
            selection_bias: None,
            routed_scale: None,
            limit: None,
            router: "gate",
            list: "experts",
            up: "up_proj",
            gate: "gate_proj",
            down: "down_proj",
            layout: self.layout,
            order: order
                .cloned()
                .unwrap_or_else(|| (0..self.num_experts).collect()),
            split: placement.tensor,
            tensor_experts: placement.tensor_experts,
            exchange: placement.experts,
            name,
        }
    }
}

/// The layout transformers' fused experts store a sparse MoE block's dense experts in: the
/// stacked `gate_up_proj` and `down_proj` under the module list.
pub const FUSED_EXPERTS: ExpertLayout = ExpertLayout::Stacked {
    gate_up: "gate_up_proj",
    down: "down_proj",
};

/// Runs every routed block of `blocks` for one expert step the rank holds no tokens of: what
/// [`crate::CausalLm::join_expert_step`] does for a model whose layers route.
///
/// # Errors
///
/// What a block refuses.
pub fn join_expert_step<'a>(
    blocks: impl IntoIterator<Item = &'a MoE>,
    exchange: &ExpertExchange,
) -> Result<()> {
    for block in blocks {
        block.join_expert_step(exchange)?;
    }
    Ok(())
}
